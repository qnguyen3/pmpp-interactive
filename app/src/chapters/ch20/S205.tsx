import type { ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { defineLabs, solutionTabStatus } from '../../components/LabKit';
import { Aside, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { FlashBanks } from './FlashBanks';
import { FlashBlockScene } from './FlashBlockScene';
import { FlashCausalTiles } from './FlashCausalTiles';
import { FlashFit } from './FlashFit';
import { FlashLanes } from './FlashLanes';
import { FlashOutcomes, FlashStepper } from './FlashStepper';
import { OnlineMergeScene } from './OnlineMergeScene';
import { OnlineSoftmaxStepper } from './OnlineSoftmaxStepper';

// Section 20.5, Flash attention (book pp. 492–504). The chapter's centre: the online
// softmax merge, the single-kernel forward pass, its storage plan and lane layout, the
// book's slips, and what changes on HIP and Metal. Book claims come from notes/ch20.md;
// vendor and project claims from notes/today/ch20-facts.md (letters A–F below) and the
// files it cites. Every interactive runs on a tested model in src/sim.

const [LAB2] = defineLabs('ch20', [{ num: '20.2', dir: 'lab2_flash', name: 'flash attention forward', vi: { name: 'flash attention lượt forward' } }]);

/** The listing's status lines from this machine's runs, in the labs page's words. */
function statuses(reports: Reports) {
  return { cu: solutionTabStatus(reports, LAB2, 'kernel.cu'), metal: solutionTabStatus(reports, LAB2, 'kernel.metal') };
}

const FLASH_TIERS: Record<string, Tier> = {
  Q: 'global', K: 'global', V: 'global', O: 'global', L: 'global', out_O: 'global', out_L: 'global', out_D: 'global',
  KT_j: 'shared', S_i: 'shared', V_j: 'shared',
  Q_i: 'reg', O_i: 'reg', m_i: 'reg', D_i: 'reg', last_m: 'reg', curr_max: 'reg', curr_sum: 'reg', S_ij: 'reg', O_ij: 'reg', P_ij: 'reg',
};

// llama.cpp's per-tile update, quoted from notes/today/ch20-facts.md D1 (fattn-vec.cuh on
// master, read 2026-11-20). A quotation, not course code: it is not compiled here.
const LLAMA = `const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
KQ_max[j] = KQ_max_new[j];

KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
KQ[j*nthreads + tid] = KQ_reg[j];
// ... every VKQ accumulator is multiplied by KQ_max_scale ...
ggml_cuda_syncwarp(); // Needed for KQ synchronization.`;

function Eq({ children, n }: { children: ReactNode; n?: string }) {
  return (
    <p style={{ fontFamily: 'var(--serif)', textAlign: 'center', margin: '0.9rem 0', overflowX: 'auto' }}>
      {children}
      {n ? <span className="muted" style={{ marginLeft: '1.2em' }}>({n})</span> : null}
    </p>
  );
}

export const questionsS205: Record<string, Question> = {
  'ch20/20.5/q-why-hard': {
    kind: 'choice',
    id: 'ch20/20.5/q-why-hard',
    prompt: (
      <Translated vi={<>Một khối đã tính một ô B_r × B_c của S cho các hàng của nó. Vì sao nó chưa thể biến bất kỳ phần tử nào của ô đó thành một xác suất p?</>}>
        <>A block has computed one B_r × B_c tile of S for its rows. Why can it not yet turn any element of that tile into a probability p?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="p cần max và tổng của cả hàng, còn ô chỉ chứa B_c trong N cột.">
            p needs the max and the sum of the whole row, and the tile holds B_c of N columns.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Ô S vẫn nằm trong thanh ghi và chưa được ghi vào bộ nhớ chia sẻ.">
            The S tile is still in registers and has not been written to shared memory yet.
          </Translated>
        ),
        why: (
          <Translated vi="Nó đã được ghi ở đó. Nó nằm ở đâu không thay đổi việc p phụ thuộc vào cái gì.">
            It is written there. Where it lives does not change what p depends on.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Mặt nạ nhân quả chỉ có thể áp dụng khi đã biết cả hàng điểm số.">
            The causal mask can only be applied once the whole row of scores is known.
          </Translated>
        ),
        why: (
          <Translated vi="Mặt nạ chỉ phụ thuộc vào chỉ số hàng và cột, mà ô nào cũng biết.">
            The mask depends only on the row and column indices, which every tile knows.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="exp quá chậm để áp dụng từng ô một, nên nó chờ cả hàng điểm số.">
            exp is too slow to apply tile by tile, so it waits for the whole row of scores.
          </Translated>
        ),
        why: (
          <Translated vi="Số phép lũy thừa là như nhau theo cách nào cũng vậy.">
            The number of exponentials is the same either way.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mỗi p trong một hàng là e<sup>l − m</sup> chia cho tổng trên hàng, cả hai đều cần đủ N cột. Một cách phân ô ngây
            thơ sẽ cần một lượt cho max và tổng, rồi một lượt nữa cho P, với toàn bộ S nằm trong bộ nhớ giữa hai lượt. Quy
            tắc trộn mang theo (m, D) cho mỗi hàng và chỉnh tỉ lệ khi các ô đến, nhờ đó bỏ được lượt thứ hai.
          </p>
        }
      >
        <p>
          Every p in a row is e<sup>l − m</sup> divided by the sum over the row, both of which need all N columns. A naive
          tiling would need one pass for the max and the sum and another for P, with the whole of S in memory between them.
          The merge rule carries (m, D) per row and rescales as tiles arrive, which removes the second pass.
        </p>
      </Translated>
    ),
    source: 'ch. 20 §20.5, p. 493',
  },
  'ch20/20.5/q-factor': {
    kind: 'numeric',
    id: 'ch20/20.5/q-factor',
    prompt: (
      <Translated
        vi={
          <>
            Một hàng, hai ô. Ô A có điểm số {'{1, 3}'}, nên sau nó m = 3 và D = e<sup>−2</sup> + 1 ≈ 1.135. Ô B có điểm số{' '}
            {'{2, 5}'}. Khi B đến, D bạn đang có được nhân với hệ số bao nhiêu? (Ba chữ số thập phân.)
          </>
        }
      >
        <>
          One row, two tiles. Tile A has scores {'{1, 3}'}, so after it m = 3 and D = e<sup>−2</sup> + 1 ≈ 1.135. Tile B
          has scores {'{2, 5}'}. When B arrives, by what factor is the D you already have multiplied? (Three decimals.)
        </>
      </Translated>
    ),
    answer: 0.1353,
    tol: 0.01,
    explain: (
      <Translated
        vi={
          <p>
            Max tăng từ 3 lên 5, nên mọi thứ đã giữ được đo so với một max giờ nhỏ hơn 2: nhân với e<sup>3 − 5</sup> = e
            <sup>−2</sup> ≈ 0.135. Đây là hệ số của số hạng A trong Eq. 20.4, e<sup>m_A − m_A∪B</sup>. Nếu max của B dưới
            3, hệ số sẽ đúng bằng 1.
          </p>
        }
      >
        <p>
          The max rises from 3 to 5, so everything kept was measured against a max that is now 2 too small: multiply by
          e<sup>3 − 5</sup> = e<sup>−2</sup> ≈ 0.135. This is the A term's factor in Eq. 20.4, e<sup>m_A − m_A∪B</sup>.
          Had B's max been below 3, the factor would have been exactly 1.
        </p>
      </Translated>
    ),
    source: 'ch. 20 §20.5, Eq. 20.4, p. 494',
  },
  'ch20/20.5/q-merged-d': {
    kind: 'numeric',
    id: 'ch20/20.5/q-merged-d',
    prompt: (
      <Translated vi={<>Cùng hàng đó. D bằng bao nhiêu sau khi trộn B? (Ba chữ số thập phân.)</>}>
        <>Same row. What is D after merging B? (Three decimals.)</>
      </Translated>
    ),
    answer: 1.203,
    tol: 0.002,
    explain: (
      <Translated
        vi={
          <p>
            1.135 × 0.135 + e<sup>2 − 5</sup> + e<sup>5 − 5</sup> = 0.154 + 0.050 + 1 = 1.203. Kiểm tra trực tiếp với max
            cuối cùng: e<sup>1−5</sup> + e<sup>3−5</sup> + e<sup>2−5</sup> + e<sup>0</sup> = 0.018 + 0.135 + 0.050 + 1 =
            1.203. Phép trộn cho cùng con số như khi thấy cả hàng một lúc.
          </p>
        }
      >
        <p>
          1.135 × 0.135 + e<sup>2 − 5</sup> + e<sup>5 − 5</sup> = 0.154 + 0.050 + 1 = 1.203. Check it directly against
          the final max: e<sup>1−5</sup> + e<sup>3−5</sup> + e<sup>2−5</sup> + e<sup>0</sup> = 0.018 + 0.135 + 0.050 + 1
          = 1.203. The merge gives the same number as having seen the whole row at once.
        </p>
      </Translated>
    ),
    source: 'ch. 20 §20.5, Eq. 20.4, p. 494',
  },
  'ch20/20.5/q-a-term': {
    kind: 'choice',
    id: 'ch20/20.5/q-a-term',
    prompt: (
      <Translated
        vi={
          <>
            Dòng 6 của Fig. 20.14 là <code>D *= exp(last_m - m_i[ii]);</code>, áp dụng cho mẫu số đã giữ đến lúc đó. Nó
            tính số hạng nào của Eq. 20.4?
          </>
        }
      >
        <>
          Line 6 of Fig. 20.14 is <code>D *= exp(last_m - m_i[ii]);</code>, applied to the denominator kept so far. Which
          term of Eq. 20.4 does it compute?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi={<>D<sub>A</sub>·e<sup>m_A − m_A∪B</sup>, số hạng A.</>}>
            <>D<sub>A</sub>·e<sup>m_A − m_A∪B</sup>, the A term.</>
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi={<>D<sub>B</sub>·e<sup>m_B − m_A∪B</sup>, số hạng B, như văn bản ở p. 502 nói.</>}>
            <>D<sub>B</sub>·e<sup>m_B − m_A∪B</sup>, the B term, as the text on p. 502 says.</>
          </Translated>
        ),
        why: (
          <Translated vi="D ở đây là mẫu số chạy của các cột đã trộn, và last_m là max của chúng: đó là A, không phải ô mới.">
            D here is the running denominator of the columns already merged, and last_m is their max: that is A, not the new
            tile.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi={<>Toàn bộ D<sub>A∪B</sub>.</>}>
            <>The whole of D<sub>A∪B</sub>.</>
          </Translated>
        ),
        why: (
          <Translated vi="Phần đóng góp của ô mới được cộng sau, trong compute_P_and_update_D.">
            The new tile’s contribution is added later, in compute_P_and_update_D.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không cái nào: đó là một phép chuẩn hóa.">Neither: it is a normalisation.</Translated>
        ),
        why: (
          <Translated vi="Không có gì bị chia ở đây; việc chuẩn hóa xảy ra một lần, trong store_O.">
            Nothing is divided here; normalisation happens once, in store_O.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Văn bản của sách gán dòng này cho số hạng B; mã thì chỉnh tỉ lệ mẫu số cũ, tức là số hạng A. Số hạng B được dựng
            trong Fig. 20.15 trực tiếp theo max đã cập nhật, dưới dạng Σ e<sup>l − m_A∪B</sup>, bằng D<sub>B</sub>·e
            <sup>m_B − m_A∪B</sup> mà không bao giờ phải tạo ra D<sub>B</sub>.
          </p>
        }
      >
        <p>
          The book's text attributes this line to the B term; the code rescales the old denominator, which is the A term.
          The B term is built in Fig. 20.15 directly against the updated max, as Σ e<sup>l − m_A∪B</sup>, which equals
          D<sub>B</sub>·e<sup>m_B − m_A∪B</sup> without ever forming D<sub>B</sub>.
        </p>
      </Translated>
    ),
    source: 'notes/ch20.md, Unclear in the PDF, p. 502', // i18n-ignore: citation of a notes section
  },
  'ch20/20.5/q-rescale-order': {
    kind: 'choice',
    id: 'ch20/20.5/q-rescale-order',
    prompt: (
      <Translated vi={<>Trong compute_O, điều gì sai nếu phép nhân với exp(last_m − m_new) đến sau khi cộng P·V_j thay vì trước?</>}>
        <>In compute_O, what goes wrong if the multiply by exp(last_m − m_new) comes after adding P·V_j instead of before?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Các số hạng của ô mới cũng bị nhân hệ số, và ra quá nhỏ.">
            The new tile’s terms get the factor too, and come out too small.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không gì cả: hệ số và phép cộng giao hoán, nên thứ tự tùy ý.">
            Nothing: the factor and the addition commute, so the order is free.
          </Translated>
        ),
        why: (
          <Translated vi="Hệ số chỉ nên áp cho phần cũ. Áp sau phép cộng thì nó cũng tác động lên phần mới.">
            The factor should apply to the old part only. Applied after the addition it also hits the new part.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ ô đầu tiên sai, vì O bắt đầu từ 0 ở đó.">
            Only the first tile is wrong, because O starts at 0 there.
          </Translated>
        ),
        why: (
          <Translated vi="Ở ô đầu tiên O bằng 0, nên thứ tự không thể quan trọng ở đó; nó quan trọng ở mọi ô sau mà max tăng.">
            On the first tile O is 0, so the order cannot matter there; it matters on every later tile where the max rises.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="D trở nên sai, vì nó được chỉnh tỉ lệ ở cùng chỗ.">
            D becomes wrong, because it is rescaled in the same place.
          </Translated>
        ),
        why: (
          <Translated vi="D được cập nhật ở các hàm khác; compute_O chỉ chạm vào O.">
            D is updated in other functions; compute_O touches only O.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            P đã được lấy lũy thừa theo max đã cập nhật, nên nó đã ở hệ quy chiếu mới. O cũ ở hệ quy chiếu cũ và cần hệ số.
            Áp sau phép cộng, nó co giãn các số hạng mới thêm một lần nữa, và chúng ra quá nhỏ mỗi khi max tăng. Hãy chỉnh
            tỉ lệ trước, rồi cộng, như Fig. 20.13 làm.
          </p>
        }
      >
        <p>
          P was exponentiated against the updated max, so it is already in the new frame. The old O is in the old frame
          and needs the factor. Applied after the addition, it scales the new terms a second time, and they come out too
          small whenever the max rose. Rescale first, then add, as Fig. 20.13 does.
        </p>
      </Translated>
    ),
    source: 'ch. 20 §20.5, Fig. 20.13, p. 503',
  },
  'ch20/20.5/q-init': {
    kind: 'choice',
    id: 'ch20/20.5/q-init',
    prompt: (
      <Translated vi={<>Bài tập 3 của sách: vì sao O_i và D_i phải bắt đầu từ 0, và m_i nên bắt đầu từ đâu?</>}>
        <>The book's Exercise 3: why must O_i and D_i start at 0, and what should m_i start at?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="O và D bằng 0, như các tổng rỗng; m dưới mọi điểm số, tốt nhất là giữ hữu hạn.">
            O and D at 0, as empty sums; m below every score, best kept finite.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Cả ba bằng 0, vì chưa có gì được cộng dồn vào chúng.">
            All three at 0, since nothing has been accumulated in any of them yet.
          </Translated>
        ),
        why: (
          <Translated vi="0 không nằm dưới mọi điểm số. Nếu các điểm số của một hàng đều rất âm, max không bao giờ rời khỏi 0, mọi e^(l − 0) có thể bị underflow về 0, và phép chia cuối là 0/0.">
            0 is not below every score. If a row’s scores are all very negative, the max never moves off 0, every e^(l − 0)
            can underflow to 0, and the final division is 0/0.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Cả ba bằng âm vô cực, để ô đầu tiên ghi đè tất cả.">
            All three at minus infinity, so that the first tile overwrites them all.
          </Translated>
        ),
        why: (
          <Translated vi="O và D là các tổng: âm vô cực nhân một hệ số, cộng một số hạng hữu hạn, vẫn là âm vô cực (hoặc thành NaN).">
            O and D are sums: minus infinity times a factor, plus a finite term, stays minus infinity (or turns NaN).
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="O và D bằng 1 và m bằng 0, để phép chia cuối luôn an toàn.">
            O and D at 1 and m at 0, so that the final division is always safe.
          </Translated>
        ),
        why: (
          <Translated vi="Phép chia là cho D, không phải O, và một số 1 trong O sẽ bị mang vào đầu ra.">
            The division is by D, not O, and a 1 in O would be carried into the output.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            O và D cộng dồn: mỗi ô nhân giá trị đang có với một hệ số và cộng phần của chính nó, nên giá trị khởi đầu bị mang
            theo, được chỉnh tỉ lệ, vào đáp án trừ khi nó bằng 0. Với m, Algorithm 1 của FlashAttention-2 bắt đầu từ âm vô
            cực. Mối nguy là số học IEEE: nếu một ô đầu tiên không có cột nhìn thấy nào, phép chỉnh tỉ lệ tính (−∞) − (−∞),
            tức là NaN; thứ tự nhân quả từ trái sang phải của sách không bao giờ gặp trường hợp đó, các thứ tự khác thì có.
            Các kernel production bắt đầu từ một giá trị hữu hạn, và các lab cũng vậy.
          </p>
        }
      >
        <p>
          O and D accumulate: each tile multiplies what is there by a factor and adds its own part, so the start value is
          carried, rescaled, into the answer unless it is 0. For m, FlashAttention-2's Algorithm 1 starts at minus
          infinity. The hazard is IEEE arithmetic: if a first tile has no visible column, the rescale computes (−∞) − (−∞),
          which is NaN; the book&rsquo;s left-to-right causal order never meets that case, others do. Production kernels start from a finite value, and
          the labs do too.
        </p>
      </Translated>
    ),
    source: 'ch. 20 Exercise 3, p. 511; ch20 facts A5, A7, D1',
    vi: { source: 'ch. 20 Bài tập 3, p. 511; ch20 facts A5, A7, D1' },
  },
  'ch20/20.5/q-lane': {
    kind: 'numeric',
    id: 'ch20/20.5/q-lane',
    prompt: (
      <Translated
        vi={
          <>
            Một mô hình có độ rộng head 256 (bench ở trên dừng ở 128), warp 32 lane, bố cục của Fig. 20.10. Mỗi lane cần bao
            nhiêu ô thanh ghi cho một hàng Q, và lane nào giữ phần tử dd = 201? Hãy cho biết lane.
          </>
        }
      >
        <>
          A model with a head width of 256 (the bench above stops at 128), 32-lane warps, the layout of Fig. 20.10. How many
          register slots does each lane need for a Q row, and which lane holds element dd = 201? Give the lane.
        </>
      </Translated>
    ),
    answer: 9,
    explain: (
      <Translated
        vi={
          <p>
            256 / 32 = 8 ô mỗi lane. 201 = 6 · 32 + 9: lane 9, ô thanh ghi 6. Trong compute_S_and_max, bước dd = 201 là một
            phép phát tán từ lane 9, <code>__shfl_sync(mask, Q_i[ii][6], 9)</code>. Quy tắc như nhau ở mọi độ rộng: lane
            dd mod 32, ô dd / 32.
          </p>
        }
      >
        <p>
          256 / 32 = 8 slots per lane. 201 = 6 · 32 + 9: lane 9, register slot 6. In compute_S_and_max, step dd = 201 is a
          broadcast from lane 9, <code>__shfl_sync(mask, Q_i[ii][6], 9)</code>. The rule is the same at any width: lane
          dd mod 32, slot dd / 32.
        </p>
      </Translated>
    ),
    source: 'ch. 20 §20.5, Figs. 20.10 and 20.12, pp. 498–500',
    vi: { source: 'ch. 20 §20.5, Figs. 20.10 và 20.12, pp. 498–500' },
  },
  'ch20/20.5/q-hip-lane': {
    kind: 'numeric',
    id: 'ch20/20.5/q-hip-lane',
    prompt: (
      <Translated
        vi={
          <>
            Một wavefront AMD rộng 64 chạy mã của sách với WARP_SIZE 32. Phép shuffle của nó dùng độ rộng mặc định của HIP,
            64. Lane phần cứng 40 (lane 8 của warp logic thứ hai) yêu cầu lane nguồn 5. Nó nhận được giá trị của lane phần
            cứng nào?
          </>
        }
      >
        <>
          A 64-wide AMD wavefront runs the book's code with WARP_SIZE 32. Its shuffle uses HIP's default width, 64.
          Hardware lane 40 (lane 8 of the second logical warp) asks for source lane 5. Which hardware lane's value does it
          get?
        </>
      </Translated>
    ),
    answer: 5,
    explain: (
      <Translated
        vi={
          <p>
            HIP tính nguồn là (srcLane &amp; (width − 1)) + (self &amp; ~(width − 1)) = 5 + 0 = 5: lane 5 của warp logic
            thứ nhất, vốn giữ một hàng Q khác. Với width 32 nó là 5 + 32 = 37, đúng lane mong muốn.
          </p>
        }
      >
        <p>
          HIP computes the source as (srcLane &amp; (width − 1)) + (self &amp; ~(width − 1)) = 5 + 0 = 5: lane 5 of the
          first logical warp, which holds a different row of Q. With width 32 it is 5 + 32 = 37, the intended lane.
        </p>
      </Translated>
    ),
    source: 'ch20 facts B2 (HIP amd_warp_functions.h)', // i18n-ignore: citation
  },
  'ch20/20.5/q-metal-bc': {
    kind: 'numeric',
    id: 'ch20/20.5/q-metal-bc',
    prompt: (
      <Translated vi={<>d = 128, B_r = 32, đệm cho 32 bank. Giá trị lớn nhất trong B_c = 32, 16 và 8 mà ba ô của nó vừa trong 32 KB bộ nhớ threadgroup là bao nhiêu?</>}>
        <>d = 128, B_r = 32, padding for 32 banks. Which is the largest of B_c = 32, 16 and 8 whose three tiles fit in 32 KB of threadgroup memory?</>
      </Translated>
    ),
    answer: 16,
    explain: (
      <Translated vi={<p>B_c = 32 cần 37,376 byte, vượt 32,768. B_c = 16 cần (2048 + 64) + 512 + 2048 float = 18,688 byte.</p>}>
        <p>B_c = 32 needs 37,376 bytes, over 32,768. B_c = 16 needs (2048 + 64) + 512 + 2048 floats = 18,688 bytes.</p>
      </Translated>
    ),
    source: 'ch20 facts C1, F1', // i18n-ignore: citation
  },
  'ch20/20.5/q-syncwarp': {
    kind: 'choice',
    id: 'ch20/20.5/q-syncwarp',
    prompt: (
      <Translated vi={<>Kernel của sách cần một rào chắn warp mà nó không có. Nó đặt ở đâu, và vì sao các phép shuffle đã có ở đó không làm được việc của nó?</>}>
        <>The book's kernel needs one warp barrier it does not have. Where does it go, and why do the shuffles already there not do its job?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Giữa lần ghi P và compute_O; shuffle không sắp thứ tự bộ nhớ.">
            Between the P write and compute_O; shuffles do not order memory.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Trước compute_S_and_max, vì Q đến đó qua các phép shuffle.">
            Before compute_S_and_max, because Q arrives there through shuffles.
          </Translated>
        ),
        why: (
          <Translated vi="Q di chuyển bằng shuffle, từ thanh ghi sang thanh ghi; không có lần trao tay qua bộ nhớ chia sẻ nào ở đó.">
            Q moves by shuffle, register to register; there is no shared-memory hand-off there.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Sau compute_O, trước khi warp chuyển sang hàng tiếp theo.">
            After compute_O, before the warp moves on to its next row.
          </Translated>
        ),
        why: (
          <Translated vi="Hàng tiếp theo dùng một hàng khác của S_i, nên không có gì bị ghi đè ở đó.">
            The next row uses a different row of S_i, so nothing is overwritten there.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không cần cái nào, vì một warp luôn thực thi đồng bộ từng bước (lockstep).">
            None is needed, because a warp always executes in lockstep.
          </Translated>
        ),
        why: (
          <Translated vi="NVIDIA đã không còn hứa điều đó kể từ independent thread scheduling, và trên Metal lockstep không được công bố.">
            NVIDIA has not promised that since independent thread scheduling, and on Metal lockstep is undocumented.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lane L ghi phần tử P thứ L trong compute_P_and_update_D; compute_O bắt mọi lane đọc cả B_c phần tử. Đó là một
            đọc-sau-ghi giữa các lane qua bộ nhớ chia sẻ, và chỉ một rào chắn có sắp thứ tự bộ nhớ mới bảo vệ được nó.
            Shuffle trao đổi thanh ghi nhưng &ldquo;do not guarantee any memory ordering&rdquo;; <code>__syncwarp()</code>{' '}
            thì có. Kernel attention của llama.cpp có một <code>ggml_cuda_syncwarp()</code> đúng ở chỗ này.
          </p>
        }
      >
        <p>
          Lane L writes P element L in compute_P_and_update_D; compute_O makes every lane read all B_c elements. That is a
          read-after-write between lanes through shared memory, and only a barrier with memory ordering protects it.
          Shuffles exchange registers but &ldquo;do not guarantee any memory ordering&rdquo;; <code>__syncwarp()</code>{' '}
          does. llama.cpp's attention kernel
          has a <code>ggml_cuda_syncwarp()</code> at exactly this point.
        </p>
      </Translated>
    ),
    source: 'ch20 facts A3, D1', // i18n-ignore: citation
  },
  'ch20/20.5/q-skip': {
    kind: 'numeric',
    id: 'ch20/20.5/q-skip',
    prompt: (
      <Translated vi={<>N = 4096, B_r = B_c = 32, nhân quả. Tỉ lệ bao nhiêu cặp (dải Q, ô K/V) nằm hoàn toàn phía trên đường chéo? (Ba chữ số thập phân.)</>}>
        <>N = 4096, B_r = B_c = 32, causal. What fraction of the (Q panel, K/V tile) pairs lies wholly above the diagonal? (Three decimals.)</>
      </Translated>
    ),
    answer: 0.496,
    tol: 0.003,
    explain: (
      <Translated
        vi={
          <p>
            128 dải nhân 128 ô = 16,384 cặp. Dải i có 127 − i ô ở bên phải, tổng cộng 8,128: 8,128 / 16,384 = 127/256 ≈
            0.496. FlashAttention-2 bỏ qua chúng; kernel của sách tính chúng, và mặt nạ của nó biến mọi điểm số thành âm vô
            cực.
          </p>
        }
      >
        <p>
          128 panels by 128 tiles = 16,384 pairs. Panel i has 127 − i tiles to its right, 8,128 in all: 8,128 / 16,384 =
          127/256 ≈ 0.496. FlashAttention-2 skips them; the book's kernel computes them, and its mask turns every score
          to minus infinity.
        </p>
      </Translated>
    ),
    source: 'ch20 facts A7', // i18n-ignore: citation
  },
  'ch20/20.5/q-kv-coalesced': {
    kind: 'choice',
    id: 'ch20/20.5/q-kv-coalesced',
    prompt: (
      <Translated vi={<>Bài tập 4 của sách: các lần nạp từ K và V trong Fig. 20.11 có được hợp nhất (coalesced) không, và luồng nào làm việc với d = 128 và 512 luồng?</>}>
        <>The book's Exercise 4: are the loads from K and V in Fig. 20.11 coalesced, and which threads do the work with d = 128 and 512 threads?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Có; các luồng 0 đến 127 nạp, mỗi luồng một phần tử, còn lại nhàn rỗi.">
            Yes; threads 0 to 127 load, one element each, and the rest idle.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không: phần lưu K^T bị chuyển vị, nên các lần nạp K từ bộ nhớ toàn cục có bước nhảy.">
            No: the K^T store is transposed, so the K loads from global are strided.
          </Translated>
        ),
        why: (
          <Translated vi="Việc chuyển vị xảy ra ở phía bộ nhớ chia sẻ. Các lần đọc toàn cục đi dọc một hàng của K.">
            The transposition happens on the shared-memory side. The global reads walk along a row of K.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Có, và cả 512 luồng mỗi luồng nạp bốn phần tử.">Yes, and all 512 threads load four elements each.</Translated>
        ),
        why: (
          <Translated vi="Vòng lặp trong chạy qua dd với bước nhảy blockDim.x = 512, nên nó chạy một lần, với dd = t < 128.">
            The inner loop is over dd with stride blockDim.x = 512, so it runs once, for dd = t {'<'} 128.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ các lần nạp V được hợp nhất; các lần nạp K đi dọc xuống một cột.">
            Only the V loads are coalesced; the K loads walk down a column.
          </Translated>
        ),
        why: (
          <Translated vi="K và V đều là N × d theo thứ tự hàng và được đọc với cùng một chỉ số.">
            K and V are both N × d row-major and are read with the same index.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Cả hai mảng đều theo thứ tự hàng, và chỉ số vòng lặp trong là chỉ số luồng: với một jj cố định, luồng t đọc phần
            tử t của một hàng K và của một hàng V, nên một warp đọc 32 float liền kề. Chỉ các luồng 0 đến 127 có dd &lt; d;
            số còn lại bỏ qua vòng lặp. Chi phí của việc chuyển vị rơi vào phần lưu vào bộ nhớ chia sẻ, và đó là lý do có
            phần đệm.
          </p>
        }
      >
        <p>
          Both arrays are row-major, and the inner loop index is the thread index: for a fixed jj, thread t reads element t
          of one K row and of one V row, so a warp reads 32 neighbouring floats. Only threads 0 to 127 have dd &lt; d; the
          rest skip the loop. The cost of the transposition lands on the shared-memory store, which is what the padding
          is for.
        </p>
      </Translated>
    ),
    source: 'ch. 20 Exercise 4, p. 511; notes/ch20.md',
    vi: { source: 'ch. 20 Bài tập 4, p. 511; notes/ch20.md' },
  },
  'ch20/20.5/q-shuffles': {
    kind: 'numeric',
    id: 'ch20/20.5/q-shuffles',
    prompt: (
      <Translated vi={<>Trong compute_S_and_max với d = 128, một lane gọi __shfl_sync bao nhiêu lần cho mỗi điểm số nó tính?</>}>
        <>In compute_S_and_max with d = 128, how many __shfl_sync calls does a lane make for each score it computes?</>
      </Translated>
    ),
    answer: 128,
    explain: (
      <Translated
        vi={
          <p>
            Một lần cho mỗi phép nhân-cộng: phần tử Q thứ dd đến từ lane dd % 32 ở bước dd. Đó là một lý do kernel này còn
            xa tốc độ của thư viện, và vì sao các phiên bản dùng tensor core tổ chức hai phép nhân theo cách khác (chú thích
            8).
          </p>
        }
      >
        <p>
          One per multiply-add: Q element dd comes from lane dd % 32 at step dd. That is one reason this kernel is far from
          library speed, and why tensor-core versions organise the two products differently (footnote 8).
        </p>
      </Translated>
    ),
    source: 'ch. 20 §20.5, Fig. 20.12, pp. 500–501',
  },
  'ch20/20.5/q-lse': {
    kind: 'choice',
    id: 'ch20/20.5/q-lse',
    prompt: (
      <Translated vi={<>Sách ghi mẫu số cuối cùng D "cho huấn luyện". Vì sao chỉ D không thể dựng lại P cho lượt backward, và FlashAttention-2 lưu gì?</>}>
        <>The book writes the final denominator D "for training". Why can D alone not rebuild P for the backward pass, and what does FlashAttention-2 store?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="P còn cần max của hàng; FA2 lưu L = m + log D.">
            P also needs the row max; FA2 stores L = m + log D.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="D là đủ, vì max m của hàng luôn bằng 0.">D is enough, because the row max m is always 0.</Translated>
        ),
        why: (
          <Translated vi="m là max của các điểm số trong hàng, có thể là bất kỳ giá trị nào.">
            m is the row max of the scores, which can be anything.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="FlashAttention-2 lưu cả m và D cho mỗi hàng.">FlashAttention-2 stores both m and D for each row.</Translated>
        ),
        why: (
          <Translated vi="Nó lưu một số cho mỗi hàng, logsumexp, mang cả hai.">
            It stores one number per row, the logsumexp, which carries both.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="FlashAttention-2 lưu chính P cho lượt backward.">FlashAttention-2 stores P itself for the backward pass.</Translated>
        ),
        why: (
          <Translated vi="P là N × N; không lưu nó chính là mục đích.">P is N × N; not storing it is the point.</Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            P = e<sup>s − m</sup> / D, và D được đo so với max cuối cùng, thứ mà store_O không ghi ra. Logsumexp L = m + log
            D gộp cả hai lại: e<sup>s − L</sup> = e<sup>s − m</sup> / D. Lab 20.2 ghi L.
          </p>
        }
      >
        <p>
          P = e<sup>s − m</sup> / D, and D is measured against the final max, which store_O does not write. The logsumexp
          L = m + log D folds the two together: e<sup>s − L</sup> = e<sup>s − m</sup> / D. Lab 20.2 writes L.
        </p>
      </Translated>
    ),
    source: 'ch20 facts A7', // i18n-ignore: citation
  },
};

const Q = questionsS205;

export function S205() {
  const { reports } = useReports();
  const st = statuses(reports);
  const cu = useLabFile('ch20/lab2_flash/solution/kernel.cu');
  const metal = useLabFile('ch20/lab2_flash/solution/kernel.metal');
  const mm = useLabFile('ch20/lab2_flash/solution/launch.mm');
  return (
    <>
      <Translated
        vi={
          <p>
            Mục 20.3 dựng attention từ các kernel riêng rẽ: một GEMM ghi S ra bộ nhớ toàn cục (global memory), một kernel
            softmax đọc nó và ghi P, một GEMM thứ hai đọc P. Bạn đã đếm cái giá đó trong{' '}
            <a href={href('ch20', '20.3')}>trình khám phá ở đó</a>: hai ma trận N × N đi ra bộ nhớ rồi quay về, và mỗi ranh
            giới kernel là một rào chắn (barrier) trên toàn lưới. Mục này loại bỏ cả hai. Một kernel tính một dải của O từ
            một dải của Q, và S và P chỉ tồn tại dưới dạng các ô (tile) trên chip.
          </p>
        }
      >
        <p>
          Section 20.3 built attention from separate kernels: a GEMM writes S to global memory, a softmax kernel reads it
          and writes P, a second GEMM reads P. You counted what that costs in <a href={href('ch20', '20.3')}>the explorer there</a>: two N × N matrices go out
          to memory and back, and every kernel boundary is a barrier across the whole grid. This section removes both. One
          kernel computes a panel of O from a panel of Q, and S and P exist only as tiles on chip.
        </p>
      </Translated>
      <Source section="20.5" pages="492–493" />

      <h2>
        <Translated vi="Vì sao một kernel lại khó">Why one kernel is hard</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Phân ô một phép nhân ma trận thì dễ vì mỗi tích riêng phần có thể được cộng vào ngay khi tính xong. Softmax không
            như vậy. Mỗi xác suất trong một hàng chia cho một tổng trên cả hàng và trừ max của hàng trước, nên một khối giữ
            một ô của S không thể hoàn tất dù chỉ một p. Câu trả lời ngây thơ là hai lượt qua mỗi hàng, với toàn bộ S được
            lưu giữa hai lượt.
          </p>
        }
      >
        <p>
          Tiling a matrix product is easy because each partial product can be added on as it is computed. Softmax is not
          like that. Every probability in a row divides by a sum over the whole row and subtracts the row's max first, so a
          block holding one tile of S cannot finish a single p. The naive answer is two passes over each row, with all of
          S stored between them.
        </p>
      </Translated>
      <Source section="20.5" pages="493" />
      <Translated
        vi={
          <p>
            Sách nêu hai cách để có một rào chắn trên toàn lưới: kết thúc kernel và khởi chạy một kernel khác, hoặc giữ một
            lưới thường trực và đồng bộ hóa nó bằng cooperative groups. Cả hai đều tồn tại, nhưng phép nhân ma trận và
            softmax muốn các hình dạng lưới khác nhau, và theo cách nào thì S cũng đi qua bộ nhớ toàn cục. Flash attention
            thay vào đó sắp xếp lại phép tính, để softmax có thể làm từng ô một. Kết quả là chính xác, không phải xấp xỉ.
          </p>
        }
      >
        <p>
          The book names the two ways to get a barrier across the grid: end the kernel and launch another, or keep one
          persistent grid and synchronise it with cooperative groups. Both exist, but matrix multiplication and softmax want
          different grid shapes, and either way S travels through global memory. Flash attention reorders the arithmetic
          instead, so that softmax can be done tile by tile. The result is exact, not an approximation.
        </p>
      </Translated>
      <Source section="20.5" pages="492–493" />
      <Ask q={Q['ch20/20.5/q-why-hard']} />
      <Translated
        vi={
          <p>
            Chương này chỉ đề cập lượt forward. Sách nhận xét rằng lượt backward không cần chỉnh tỉ lệ nhưng tốn năm phép
            nhân ma trận so với hai của lượt forward; bài báo FlashAttention-2 đưa ra cùng con số.
          </p>
        }
      >
        <p>
          The chapter covers the forward pass only. The book remarks that the backward pass needs no rescaling but costs
          five matrix multiplications to the forward pass's two; the FlashAttention-2 paper gives the same count.
        </p>
      </Translated>
      <Source section="20.5" pages="493" />

      <h2>
        <Translated vi="Trộn hai softmax riêng phần">Merging two partial softmaxes</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chia các cột của hàng r thành hai tập rời nhau A và B. Chỉ trên A, bạn có thể tính một max riêng phần m
            <sub>A</sub> và một mẫu số riêng phần đo so với nó:
          </p>
        }
      >
        <p>
          Split the columns of row r into disjoint sets A and B. Over A alone you can compute a partial max m<sub>A</sub> and
          a partial denominator measured against it:
        </p>
      </Translated>
      <Eq n="20.3">D<sub>A</sub> = Σ<sub>j∈A</sub> e<sup>l_j − m_A</sup></Eq>
      <Translated
        vi={
          <p>
            Hai kết quả riêng phần kết hợp lại thành kết quả bạn sẽ có từ A ∪ B, với m<sub>A∪B</sub> = max(m<sub>A</sub>, m
            <sub>B</sub>):
          </p>
        }
      >
        <p>
          The two partial results combine into the one you would have got from A ∪ B, with m<sub>A∪B</sub> = max(m<sub>A</sub>,
          m<sub>B</sub>):
        </p>
      </Translated>
      <Eq n="20.4">D<sub>A∪B</sub> = D<sub>A</sub>·e<sup>m_A − m_A∪B</sup> + D<sub>B</sub>·e<sup>m_B − m_A∪B</sup></Eq>
      <Translated
        vi={
          <p>
            Mỗi số hạng đưa một tổng riêng phần về max chung. Hệ số không bao giờ lớn hơn 1, và đúng bằng 1 với phía nào giữ
            max lớn hơn. Hai tập không được chồng lên nhau, nếu không một cột sẽ bị đếm hai lần; các ô của một hàng không bao
            giờ chồng nhau.
          </p>
        }
      >
        <p>
          Each term moves one partial sum to the common max. The factor is never above 1, and it is exactly 1 for whichever
          side holds the larger max. The two sets must not overlap, or a column would be counted twice; tiles of a row
          never do.
        </p>
      </Translated>
      <Source section="20.5" pages="493–494" figs={tr('Eqs. 20.3, 20.4', 'Pt. 20.3, 20.4')} />
      <Ask q={Q['ch20/20.5/q-factor']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={Q['ch20/20.5/q-merged-d']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Cùng bước đi đó áp dụng cho đầu ra. Eq. 20.5 định nghĩa một đầu ra riêng phần o<sub>A</sub> được chuẩn hóa bởi D
            <sub>A</sub>, và Eq. 20.6 trộn hai đầu ra như vậy, mỗi cái có trọng số theo phần đã chỉnh tỉ lệ của nó trong mẫu
            số đã trộn.
          </p>
        }
      >
        <p>
          The same move works for the output. Eq. 20.5 defines a partial output o<sub>A</sub> normalised by D<sub>A</sub>,
          and Eq. 20.6 merges two of them, each weighted by its rescaled share of the merged denominator.
        </p>
      </Translated>
      <Source section="20.5" pages="494–495" figs={tr('Eqs. 20.5, 20.6', 'Pt. 20.5, 20.6')} />
      <Translated
        vi={
          <p>
            Kernel làm một việc đơn giản hơn nhưng là cùng phép tính: nó giữ Õ = Σ e<sup>l − m</sup> v chưa chuẩn hóa, chỉnh
            tỉ lệ nó bằng e<sup>m_old − m_new</sup> mỗi khi max tăng, cộng các số hạng của ô mới (đã đo so với max mới), và
            chia cho D một lần ở tận cuối. FlashAttention-2 mô tả đúng điều này.
          </p>
        }
      >
        <p>
          The kernel does something simpler that is the same arithmetic: it keeps the un-normalised Õ = Σ e<sup>l − m</sup>{' '}
          v, rescales it by e<sup>m_old − m_new</sup> whenever the max rises, adds the new tile's terms (already measured
          against the new max), and divides by D once at the very end. FlashAttention-2 describes exactly this.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              <b>Errata, p. 495.</b> Ở dòng đầu của phần suy dẫn tiếp theo của Eq. 20.6, số mũ thứ hai được in là l − m
              <sub>B</sub>; các dòng trước và sau cần l − m<sub>A∪B</sub>.
            </p>
          }
        >
          <p>
            <b>Erratum, p. 495.</b> In the first line of the continued derivation of Eq. 20.6 the second exponent is printed
            l − m<sub>B</sub>; the lines before and after need l − m<sub>A∪B</sub>.
          </p>
        </Translated>
      </Aside>
      <Translated vi={<p>Hãy xem phép trộn diễn ra một lần, rồi tự điều khiển nó.</p>}>
        <p>Watch the merge happen once, then take control of it.</p>
      </Translated>
      <OnlineMergeScene />
      <Translated
        vi={
          <p>
            Trình chạy từng bước chạy cùng hàng đó. Hãy thử độ rộng ô 1 và 12, và ba thứ tự: phép kiểm tra cuối không bao giờ
            thay đổi. Rồi nhìn cột hệ số ở mỗi bước và tìm khi nào nó đúng bằng 1. Dùng dữ liệu "max arrives late" để thấy các
            giá trị đã giữ co lại mạnh ở ô cuối, và dữ liệu "+ 100" để thấy vì sao phải trừ max. Chuyển sang dạng chuẩn hóa để
            thấy Eq. 20.6 theo nghĩa đen.
          </p>
        }
      >
        <p>
          The stepper runs the same row. Try a tile width of 1 and of 12, and the three orders: the final check never
          changes. Then look at the factor column on each step and find when it is exactly 1. Use the "max arrives late"
          data to see the kept values shrink hard on the last tile, and the "+ 100" data to see why the max is subtracted at
          all. Switch to the normalised form to see Eq. 20.6 literally.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Scene và trình chạy từng bước: một hàng, không phải một khối nhiều hàng. float32 được giả lập bằng cách làm
              tròn mọi phép toán với Math.fround, và exp là của JavaScript, làm tròn về float32, không phải của GPU (có thể
              khác ở các bit cuối). Ở đây không có luồng hay bộ nhớ chia sẻ; trình chạy từng bước của kernel ở phía dưới có
              những thứ đó.
            </p>
          }
        >
          <p>
            The scene and the stepper: one row, not a block of rows. float32 is emulated by rounding every operation with Math.fround, and exp is
            JavaScript's, rounded to float32, not a GPU's (which may differ in the last bits). No threads or shared memory
            here; the kernel stepper further down has those.
          </p>
        </Translated>
      </Simplification>
      <OnlineSoftmaxStepper />

      <h3>
        <Translated vi="m bắt đầu từ đâu">What m starts at</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Bài tập 3 yêu cầu hàm khởi tạo O<sub>i</sub>, D<sub>i</sub> và m<sub>i</sub>. O và D là các tổng chưa có số hạng
            nào, nên bằng 0. Với m, Algorithm 1 của FlashAttention-2 dùng âm vô cực.
          </p>
        }
      >
        <p>
          Exercise 3 asks for the function that initialises O<sub>i</sub>, D<sub>i</sub> and m<sub>i</sub>. O and D are
          sums over nothing yet, so 0. For m, FlashAttention-2's Algorithm 1 uses minus infinity.
        </p>
      </Translated>
      <Source section="20.9" pages="511" figs={tr('Exercise 3', 'Bài tập 3')} />
      <Translated
        vi={
          <p>
            Giờ hãy đặt trình chạy từng bước về hàng 2, ô cỡ 4, từ phải sang trái, bắt đầu từ âm vô cực: ô đầu tiên không có
            cột nhìn thấy nào, max của nó vẫn là âm vô cực, và phép chỉnh tỉ lệ tính (−∞) − (−∞), mà số học IEEE định nghĩa là
            NaN. Đầu ra là NaN từ đó trở đi. Chuyển giá trị khởi đầu sang −FLT_MAX/2 và cùng thứ tự đó cho kết quả chính xác.
          </p>
        }
      >
        <p>
          Now set the stepper to row 2, tiles of 4, right to left, starting at minus infinity: the first tile has no visible
          column, its max is still minus infinity, and the rescale computes (−∞) − (−∞), which IEEE arithmetic defines as
          NaN. The output is NaN from then on. Switch the start to −FLT_MAX/2 and the same order is exact.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md A5, A7, C5, D1, D4)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              Hướng dẫn của CUDA liệt kê Inf − Inf trong các dạng cho ra NaN. llama.cpp bắt đầu max chạy của nó ở −FLT_MAX/2
              với một chú thích giải thích lý do; các kernel Metal của MLX bắt đầu ở một giá trị nhỏ nhất hữu hạn. Trên Metal
              còn một lý do thứ hai: dưới fast math, hành vi của vô cực và NaN là không xác định.
            </p>
          }
        >
          <p>
            CUDA's guide lists Inf − Inf among the forms that give NaN. llama.cpp starts its running max at −FLT_MAX/2 with
            a comment saying why; MLX's Metal kernels start at a finite minimum. On Metal there is a second reason: under fast
            math the behaviour of infinities and NaN is undefined.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Kernel của sách không bao giờ gặp NaN, vì nó đi qua các ô từ trái sang phải bắt đầu từ cột 0, nên mọi hàng đều
            thấy một cột nhìn thấy ở ô đầu tiên. Chia các cột cho nhiều khối, đi qua chúng theo thứ tự khác, hoặc thêm một mặt
            nạ đệm hay một cửa sổ trượt, và nó sẽ gặp. Các lab bắt đầu m ở −FLT_MAX/2 và che theo chỉ số, không bao giờ lưu
            một giá trị vô cực.
          </p>
        }
      >
        <p>
          The book's kernel never meets the NaN, because it walks tiles left to right from column 0, so every row sees a
          visible column in the first tile. Split the columns across blocks, walk them in another order, or add a padding
          mask or a sliding window, and it does. The labs start m at −FLT_MAX/2 and mask by index, never storing an
          infinity.
        </p>
      </Translated>
      <Ask q={Q['ch20/20.5/q-init']} />

      <h2>
        <Translated vi="Kế hoạch của kernel">The kernel's plan</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Kernel của sách cố ý hẹp: lượt forward của một head nhân quả cho một chuỗi, với float 32-bit, d = 128, B_r = B_c
            = 32 và 512 luồng mỗi khối cố định lúc biên dịch. Chính sách nói rằng các cách cài đặt thật chạy với float 16-bit
            hoặc 8-bit trên tensor core. Thứ mang sang được là cấu trúc.
          </p>
        }
      >
        <p>
          The book's kernel is deliberately narrow: the forward pass of one causal head for one sequence, in 32-bit floats,
          with d = 128, B_r = B_c = 32 and 512 threads per block fixed at compile time. The book says itself that real
          implementations run in 16-bit or 8-bit floats on tensor cores. What carries over is the structure.
        </p>
      </Translated>
      <Source section="20.5" pages="495–496" figs={tr('Fig. 20.9', 'Hình 20.9')} />
      <Translated
        vi={
          <p>
            Mỗi khối sở hữu một dải B_r hàng của Q và tạo ra cùng các hàng đó của O. Một lưới nhỏ hơn số dải vẫn phủ hết
            chúng, vì vòng lặp ngoài bước theo lưới: <code>for (i = blockIdx.x; i &lt; T_r; i += gridDim.x)</code>. Sách nói
            nhiều head hơn nghĩa là nhiều chiều lưới hơn; lab 20.2 dùng chiều y của lưới cho batch × heads.
          </p>
        }
      >
        <p>
          Each block owns a panel of B_r rows of Q and produces the same rows of O. A grid smaller than the number of panels
          still covers them, because the outer loop strides by the grid: <code>for (i = blockIdx.x; i &lt; T_r; i +=
          gridDim.x)</code>. More heads, the book says, mean more grid dimensions; lab 20.2 uses the grid's y dimension for
          batch × heads.
        </p>
      </Translated>
      <Source section="20.5" pages="495, 497" />
      <Translated
        vi={
          <p>
            Mỗi mảnh nằm ở đâu quyết định toàn bộ thiết kế. Các hàng Q của khối, các hàng O của nó và m và D theo từng hàng
            nằm trong thanh ghi, trải ra trên các luồng. Ô hiện tại của K (lưu chuyển vị, dưới tên KT_j), ô hiện tại của V và
            ô S nằm trong bộ nhớ chia sẻ, và ô S bị P ghi đè tại chỗ. Trong một khối, B_r hàng được chia cho các warp, nên các
            warp không bao giờ trao đổi gì với nhau; mọi trao đổi là giữa các lane của một warp. Cách chia Q giữa các warp đó
            là của FlashAttention-2. Phiên bản 1 chia K và V giữa các warp, buộc các warp phải kết hợp kết quả riêng phần qua
            bộ nhớ chia sẻ.
          </p>
        }
      >
        <p>
          Where each piece lives decides the whole design. The block's rows of Q, its rows of O and the per-row m and D sit
          in registers, spread over the threads. The current tile of K (stored transposed, as KT_j), the current tile of V
          and the S tile sit in shared memory, and the S tile is overwritten in place by P. Inside a block, the B_r rows are
          divided among the warps, so warps never exchange anything; all exchange is between lanes of one warp. That split
          of Q across warps is FlashAttention-2's. Version 1 split K and V across warps, which forced the warps to combine
          partial results through shared memory.
        </p>
      </Translated>
      <Source section="20.5" pages="497–498" figs={tr('Fig. 20.8; footnote 7', 'Hình 20.8; chú thích 7')} />
      <Translated
        vi={
          <p>
            Lời khuyên của sách về kích thước: B_r lớn nhất mà bộ nhớ trên chip và số khối cần để lấp đầy thiết bị cho phép,
            vì nó quyết định mức dùng lại của mỗi ô K và V; B_c lớn nhất mà bộ nhớ chia sẻ cho phép, vì mỗi ô tốn một lần
            chỉnh tỉ lệ O.
          </p>
        }
      >
        <p>
          The book's advice on sizes: B_r as large as on-chip memory and the number of blocks needed to fill the device
          allow, since it sets the reuse of every K and V tile; B_c as large as shared memory allows, since every tile costs
          one rescale of O.
        </p>
      </Translated>
      <Source section="20.5" pages="497, 499" />
      <Translated
        vi={
          <p>
            Cả hai cùng kéo về cùng một ngân sách. Các ô chia sẻ tăng theo B_c, và các hàng Q và O theo từng lane tăng theo
            B_r. Hãy tìm các con số của sách trong trình khám phá, rồi một cấu hình vừa với Metal.
          </p>
        }
      >
        <p>
          Both pull against the same budget. The shared tiles grow with B_c, and the per-lane rows of Q and O grow with B_r.
          Find the book's numbers in the explorer, then a configuration that fits Metal.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Chỉ là số học tĩnh: byte và thanh ghi, không phải tốc độ. Số thanh ghi là số mà các mảng cần, không phải số mà
              trình biên dịch cấp phát. Con số của AMD là LDS nhỏ nhất trong bảng của AMD.
            </p>
          }
        >
          <p>
            Static arithmetic only: bytes and registers, not speed. The register count is what the arrays need, not what a
            compiler allocates. The AMD figure is the smallest LDS in AMD's table.
          </p>
        </Translated>
      </Simplification>
      <FlashFit />
      <Translated
        vi={
          <p>
            37,376 byte vừa với 48 KB bộ nhớ chia sẻ tĩnh của CUDA và mọi LDS của AMD. Nó không vừa với 32 KB của Metal, và
            lỗi đến muộn: trình biên dịch chấp nhận kernel, và việc dựng pipeline thất bại với "Threadgroup memory size
            (37376) exceeds the maximum threadgroup memory allowed (32768)".
          </p>
        }
      >
        <p>
          37,376 bytes fits CUDA's 48 KB of static shared memory and any AMD LDS. It does not fit Metal's 32 KB, and the
          failure is late: the compiler accepts the kernel, and building the pipeline fails with "Threadgroup memory size
          (37376) exceeds the maximum threadgroup memory allowed (32768)".
        </p>
      </Translated>
      <Source section="20.5" pages="496" />
      <Translated
        vi={
          <p>
            Vậy các kích thước của sách là một lựa chọn cho một máy, không phải một tính chất của thuật toán. Lab 20.2 đọc
            giới hạn từ thiết bị và chọn B_c theo nó.
          </p>
        }
      >
        <p>
          So the book's sizes are a choice for one machine, not a property of the algorithm. Lab 20.2 reads the limit from
          the device and picks B_c from it.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md C1; ch05 A3, C1)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              Thông báo lỗi lấy từ một lần chạy trên một máy Mac trong lúc kiểm tra các ghi chú này; giới hạn 32 KB là từ
              bảng tính năng của Apple cho Apple4 đến Apple11. Giới hạn của chính bạn nằm trong báo cáo Metal của bạn.
            </p>
          }
        >
          <p>
            The error message is from a run on one Mac while these notes were checked; the 32 KB limit is Apple's feature
            tables for Apple4 to Apple11. Your own limit is in your Metal report.
          </p>
        </Translated>
      </Today>
      <Ask q={Q['ch20/20.5/q-metal-bc']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Thêm một điều kế hoạch quyết định: những ô nào cần tính. Với mặt nạ nhân quả, một ô K/V mà mọi cột của nó nằm bên
            phải mọi hàng của dải thì không đóng góp gì.
          </p>
        }
      >
        <p>
          One more thing the plan decides: which tiles to compute at all. With a causal mask, a K/V tile whose every column
          lies to the right of every row of the panel contributes nothing.
        </p>
      </Translated>
      <Simplification>
        <Translated vi={<p>Đếm các cặp (dải, ô), không phải thời gian: trong một kernel thật, một ô bị che còn tốn cả chi phí của mặt nạ.</p>}>
          <p>Counts pairs of (panel, tile), not time: in a real kernel a masked tile also costs the mask itself.</p>
        </Translated>
      </Simplification>
      <FlashCausalTiles />
      <Translated
        vi={
          <p>
            FlashAttention-2 bỏ qua các ô rỗng và chỉ áp dụng mặt nạ cho các ô cắt qua đường chéo, tiết kiệm khoảng một nửa
            công việc với các chuỗi dài. Kernel của sách tính mọi ô; mặt nạ của nó sau đó biến mọi điểm số trong các ô rỗng
            thành âm vô cực và mọi xác suất thành 0. Bản tham chiếu của lab 20.2 bỏ qua chúng, như FlashAttention-2, nên
            kernel đã sửa bên dưới cũng khác trang sách ở điểm này.
          </p>
        }
      >
        <p>
          FlashAttention-2 skips the empty tiles and applies the mask only to tiles that cross the diagonal, about half the
          work saved for long sequences. The book's kernel computes every tile; its mask then turns every score in the empty
          ones to minus infinity and every probability to 0. Lab 20.2's reference skips them, as FlashAttention-2 does,
          so the corrected kernel below differs from the page here too.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Việc bỏ qua ô lấy từ bài báo FlashAttention-2 (ch20 facts A7). Kernel Metal của llama.cpp phân loại các khối
              của mặt nạ theo cùng cách trước vòng lặp chính của nó (D2).
            </p>
          }
        >
          <p>
            Tile skipping is from the FlashAttention-2 paper (ch20 facts A7). llama.cpp's Metal kernel classifies blocks of
            the mask the same way before its main loop (D2).
          </p>
        </Translated>
      </Beyond>
      <Ask q={Q['ch20/20.5/q-skip']} label={tr('Work it out', 'Tự tính')} />

      <h2>
        <Translated vi="Ai giữ cái gì">Who holds what</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mỗi warp sở hữu B_r_warp = B_r / N_WARPS hàng, 2 hàng trong cấu hình của sách. Một hàng của Q được chia cho 32
            lane của warp theo cách chia bài: lane L giữ các phần tử L, L + 32, L + 64 và L + 96, mỗi phần tử một ô thanh ghi.
            Các lane cạnh nhau nạp các phần tử cạnh nhau, nên mọi lần nạp đều được hợp nhất. O được bố trí theo cùng cách.
          </p>
        }
      >
        <p>
          Each warp owns B_r_warp = B_r / N_WARPS rows, 2 in the book's configuration. A row of Q is dealt across the
          warp's 32 lanes the way cards are dealt: lane L holds elements L, L + 32, L + 64 and L + 96, one per register slot.
          Lanes next to each other load elements next to each other, so every load is coalesced. O is laid out the same way.
        </p>
      </Translated>
      <Source section="20.5" pages="498" figs={tr('Figs. 20.8, 20.10', 'Hình 20.8, 20.10')} />
      <Translated
        vi={
          <p>
            Để tính một điểm số, một lane cần cả một hàng Q, nhưng chỉ giữ 4 trong 128 phần tử của nó. Vì vậy ở bước dd của
            tích vô hướng, lane sở hữu phần tử dd phát tán nó:{' '}
            <code>__shfl_sync(mask, Q_i[ii][dd / 32], dd % 32)</code>. Mọi lane nhận cùng giá trị và nhân nó với cột KT_j
            của riêng mình.
          </p>
        }
      >
        <p>
          To compute a score, a lane needs a whole row of Q, but holds only 4 of its 128 elements. So at step dd of the inner product,
          the lane that owns element dd broadcasts it: <code>__shfl_sync(mask, Q_i[ii][dd / 32], dd % 32)</code>. Every lane
          receives the same value and multiplies it by its own column of KT_j.
        </p>
      </Translated>
      <Source section="20.5" pages="500" figs={tr('Fig. 20.12', 'Hình 20.12')} />
      <Translated
        vi={
          <p>
            Đó là một phép shuffle cho mỗi phép nhân-cộng. Bên dưới, hãy đi qua vòng lặp trong từng phần tử một và xem lane
            nào phát tán mỗi phần tử; rồi chuyển sang một wavefront AMD rộng 64 chạy cùng mã đó.
          </p>
        }
      >
        <p>
          That is one shuffle per multiply-add. Below, walk the inner loop element by element and watch which lane
          broadcasts each one; then switch to a 64-wide AMD wavefront running the same code.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Hình chỉ thể hiện cách đánh chỉ số. Trên AMD nó theo mã nguồn shuffle của HIP; không có gì được chạy trên phần
              cứng AMD.
            </p>
          }
        >
          <p>
            The picture is the indexing only. On AMD it follows HIP's shuffle source code; nothing was run on AMD hardware.
          </p>
        </Translated>
      </Simplification>
      <FlashLanes />
      <Ask q={Q['ch20/20.5/q-lane']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={Q['ch20/20.5/q-shuffles']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Sách lưu ý rằng đây là một phép nhân ma trận cố ý đơn giản. Với tensor core, hai phép nhân bên trong vòng lặp sẽ
            được tổ chức khác đi, và đó là nơi các kernel thư viện có được tốc độ.
          </p>
        }
      >
        <p>
          The book notes that this is a deliberately plain matrix product. With tensor cores, the two products inside the
          loop would be organised differently, which is where library kernels get their speed.
        </p>
      </Translated>
      <Source section="20.5" pages="500–501" figs={tr('footnote 8', 'chú thích 8')} />

      <h3>
        <Translated vi="Cùng bố cục trên HIP và Metal">The same layout on HIP and Metal</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Trên HIP đoạn mã không biên dịch được như khi in: mask <code>0xFFFFFFFF</code> không qua phép kiểm tra của HIP
            rằng lane mask phải là 64 bit, và CUB là hipCUB. Tệ hơn là cái biên dịch được. Trên một wavefront rộng 64 với{' '}
            <code>WARP_SIZE</code> cố định là 32, một wavefront chứa hai warp của sách, và <code>__shfl</code> của HIP mặc
            định độ rộng là cả wavefront. Khi đó warp logic thứ hai đọc Q của warp thứ nhất. Phép rút gọn warp của hipCUB
            cũng mặc định 64 lane logic trên CDNA, rút gọn hai hàng thành một. Hãy chuyển trình khám phá ở trên sang cấu hình
            AMD để thấy điều đó.
          </p>
        }
      >
        <p>
          On HIP the listing does not compile as printed: the <code>0xFFFFFFFF</code> mask fails HIP's check that a lane mask
          is 64 bits, and CUB is hipCUB. Worse is what compiles. On a 64-wide wavefront with <code>WARP_SIZE</code> fixed at
          32, one wavefront holds two of the book's warps, and HIP's <code>__shfl</code> defaults to a width of the whole
          wavefront. The second logical warp then reads the first one's Q. hipCUB's warp reduction also defaults to 64
          logical lanes on CDNA, reducing two rows into one. Switch the explorer above to the AMD settings to see it.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              HIP và Metal không có trong sách. Dữ kiện HIP: ch20 B1, B2 (mã nguồn HIP và hipCUB ở tag ROCm 10.2). Dữ kiện
              Metal: C3; ch04 C2, D2.
            </p>
          }
        >
          <p>
            HIP and Metal are not in the book. HIP facts: ch20 B1, B2 (HIP and hipCUB source at the ROCm 10.2 tag). Metal
            facts: C3; ch04 C2, D2.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Có hai cách sửa nhất quán: biến độ rộng warp logic thành một tham số lúc biên dịch bằng wavefront (64 cho 8 warp
            mỗi warp 4 hàng với khối của sách), hoặc giữ warp logic 32 lane và truyền width 32 cho mọi phép shuffle và mọi
            phép rút gọn. Các lab làm cách thứ hai: một hằng số 32 lane có tên với một chú thích giải thích giả định của sách,
            width 32 trên mọi phép shuffle, và một lane mask có kiểu.
          </p>
        }
      >
        <p>
          There are two consistent fixes: make the logical warp width a compile-time parameter equal to the wavefront (64
          gives 8 warps of 4 rows for the book's block), or keep 32-lane logical warps and pass width 32 to every shuffle and
          every reduction. The labs do the second: a named 32-lane constant with a comment explaining the book's
          assumption, width 32 on every shuffle, and a typed lane mask.
        </p>
      </Translated>
      <Ask q={Q['ch20/20.5/q-hip-lane']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Trên Metal, lane và SIMD group đến từ <code>[[thread_index_in_simdgroup]]</code> và{' '}
            <code>[[simdgroup_index_in_threadgroup]]</code>. Phép phát tán là <code>simd_broadcast(value, lane)</code>, được
            định nghĩa khi mọi lane chỉ định cùng một nguồn, như ở đây. Apple không công bố độ rộng SIMD group, còn bố cục
            cần 32, nên mã host của lab 20.2 kiểm tra <code>threadExecutionWidth</code> của pipeline và thất bại với một
            thông báo rõ ràng nếu nó khác. Backend Metal của llama.cpp cũng giả định 32.
          </p>
        }
      >
        <p>
          On Metal, lane and SIMD-group come from <code>[[thread_index_in_simdgroup]]</code> and{' '}
          <code>[[simdgroup_index_in_threadgroup]]</code>. The broadcast is <code>simd_broadcast(value, lane)</code>, which
          is defined when every lane names the same source, as here. Apple does not document the SIMD-group width, and the
          layout needs 32, so lab 20.2's host code checks the pipeline's <code>threadExecutionWidth</code> and fails with a
          clear message if it differs. llama.cpp's Metal backend assumes 32 as well.
        </p>
      </Translated>

      <h2>
        <Translated vi="Nạp K ở dạng chuyển vị">Loading K transposed</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mỗi ô của K đến dưới dạng hàng (một key mỗi hàng), nhưng vòng lặp điểm số muốn các key là cột, để lane jj đi dọc
            xuống cột jj. Phép nạp đọc K từng hàng và ghi nó ở dạng chuyển vị. Khi đó luồng dd lưu vào từ dd · B_c + jj: các
            luồng liên tiếp cách nhau B_c từ, và với B_c = 32 trên 32 bank, tất cả đều trúng cùng một bank.
          </p>
        }
      >
        <p>
          Each tile of K arrives as rows (one key per row), but the score loop wants keys as columns, so that lane jj walks
          down column jj. The load reads K row by row and writes it transposed. Thread dd then stores to word dd · B_c + jj:
          consecutive threads are B_c words apart, and with B_c = 32 on 32 banks, all of them hit the same bank.
        </p>
      </Translated>
      <Source section="20.5" pages="499" figs={tr('Fig. 20.11', 'Hình 20.11')} />
      <Translated
        vi={
          <p>
            Sách đệm mọi địa chỉ bằng <code>addr(x) = x + (x {'>>'} LOG_NUM_BANKS)</code>, chèn một từ dự phòng cho mỗi 32
            từ và biến bước nhảy thành 33.
          </p>
        }
      >
        <p>
          The book pads every address with <code>addr(x) = x + (x {'>>'} LOG_NUM_BANKS)</code>, which inserts one spare word
          per 32 and turns the stride into 33.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Văn bản ở p. 499 gọi hàm là <code>load_KTV()</code>; đoạn mã gọi nó là <code>load_KT_and_V</code>. Nó nói{' '}
              <code>addr()</code> đến từ chương 16, nơi macro có tên <code>pad()</code>. Cả <code>LOG_NUM_BANKS</code> lẫn{' '}
              <code>addr()</code> đều không được định nghĩa trong chương.
            </p>
          }
        >
          <p>
            The text on p. 499 calls the function <code>load_KTV()</code>; the listing calls it <code>load_KT_and_V</code>.
            It says <code>addr()</code> comes from chapter 16, where the macro is named <code>pad()</code>. Neither{' '}
            <code>LOG_NUM_BANKS</code> nor <code>addr()</code> is defined in the chapter.
          </p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            <code>LOG_NUM_BANKS</code> = 5 là 32 bank 4 byte của NVIDIA. AMD có 32 bank trên CDNA 1 đến 3 và 64 trên CDNA 4
            và RDNA 2 đến 4, và dù sao cũng phục vụ một wavefront qua nhiều chu kỳ. Apple không công bố cấu trúc bank nào cả.
            Vì vậy phần đệm là một phép đo cần làm trên từng thiết bị, không phải một hằng số để sao chép.
          </p>
        }
      >
        <p>
          <code>LOG_NUM_BANKS</code> = 5 is NVIDIA's 32 banks of 4 bytes. AMD has 32 banks on CDNA 1 to 3 and 64 on CDNA 4
          and RDNA 2 to 4, and serves a wavefront over several cycles anyway. Apple documents no bank structure at all. So
          the padding is a measurement to make on each device, not a constant to copy.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/infra-facts.md A10, C13; ch20 B3)">{/* i18n-ignore: date and fact references */}
        <Translated vi={<p>Số bank theo từng hãng như được công bố hiện nay; bộ nhớ threadgroup của Apple không có bố cục bank được công bố.</p>}>
          <p>Bank counts per vendor as documented today; Apple's threadgroup memory has no published bank layout.</p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Hãy đếm số lượt mà một lần lưu chuyển vị cần, có và không có phần đệm, với số bank của từng hãng. Dự đoán trước.
          </p>
        }
      >
        <p>
          Count the passes one transposed store needs, with and without the padding, for each vendor&rsquo;s bank count.
          Predict first.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Quy tắc của NVIDIA (bank = từ mod số bank; một lượt cho mỗi từ phân biệt trên bank bận nhất) áp dụng cho số
              bank của từng hãng. Đó không phải mô hình chi phí của AMD, và không có mô hình nào cho Apple.
            </p>
          }
        >
          <p>
            NVIDIA's rule (bank = word mod bank count; a pass per distinct word on the busiest bank) applied to each vendor's
            bank count. It is not AMD's cost model, and there is none for Apple.
          </p>
        </Translated>
      </Simplification>
      <FlashBanks />
      <Ask q={Q['ch20/20.5/q-kv-coalesced']} />

      <h2>
        <Translated vi="Một hàng, từng bước một">One row, step by step</Translated>
      </h2>
      <Translated
        vi={<p>Với mỗi hàng ii của dải, một warp chạy bốn hàm trên ô hiện tại. Chúng là Eq. 20.4 và Eq. 20.6 dưới dạng mã.</p>}
      >
        <p>
          For each row ii of its panel, a warp runs four functions against the current tile. They are Eq. 20.4 and Eq. 20.6
          in code.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <b>compute_S_and_max</b> tính B_c điểm số của hàng, mỗi lane một cột, nhân chúng với 1/√d, và che mỗi cột nằm
            sau hàng bằng cách so sánh chỉ số. Nó ghi các điểm số vào hàng của warp trong S_i và rút gọn max của chúng trên
            warp. Phép rút gọn CUB của sách để max ở lane 0, nên một phép shuffle từ lane 0 trao nó cho mọi lane.
          </p>
        }
      >
        <p>
          <b>compute_S_and_max</b> computes the row's B_c scores, one column per lane, scales them by 1/√d, and masks each
          column past the row by comparing indices. It writes the scores into the warp's row of S_i and reduces their max
          across the warp. The book's CUB reduction leaves the max in lane 0, so a shuffle from lane 0 hands it to everyone.
        </p>
      </Translated>
      <Source section="20.5" pages="500" figs={tr('Fig. 20.12', 'Hình 20.12')} />
      <Translated
        vi={
          <p>
            <b>update_m_and_D</b> nâng max chạy nếu max của ô lớn hơn, nhân D với e<sup>old − new</sup>, và trả về max{' '}
            <i>cũ</i>, vì compute_O vẫn cần nó.
          </p>
        }
      >
        <p>
          <b>update_m_and_D</b> raises the running max if the tile's max is larger, multiplies D by e<sup>old − new</sup>,
          and returns the <i>old</i> max, because compute_O still needs it.
        </p>
      </Translated>
      <Source section="20.5" pages="502" figs={tr('Fig. 20.14', 'Hình 20.14')} />
      <Translated
        vi={
          <p>
            Theo các số hạng của Eq. 20.4, <code>last_m</code> là m<sub>A</sub>, max của ô là m<sub>B</sub> và max đã cập
            nhật là m<sub>A∪B</sub>. Vì vậy dòng 6, <code>D *= exp(last_m - m_i[ii])</code>, đưa mẫu số đã giữ đến lúc đó
            về max mới.
          </p>
        }
      >
        <p>
          In Eq. 20.4&rsquo;s terms, <code>last_m</code> is m<sub>A</sub>, the tile&rsquo;s max is m<sub>B</sub> and the
          updated max is m<sub>A∪B</sub>. So line 6, <code>D *= exp(last_m - m_i[ii])</code>, moves the denominator kept so
          far to the new max.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              <b>Errata, p. 502.</b> Văn bản nói dòng 6 tính số hạng B của Eq. 20.4. Nó chỉnh tỉ lệ mẫu số cũ: số hạng A, D
              <sub>A</sub>·e<sup>m_A − m_A∪B</sup>.
            </p>
          }
        >
          <p>
            <b>Erratum, p. 502.</b> The text says line 6 computes the B term of Eq. 20.4. It rescales the old denominator: the
            A term, D<sub>A</sub>·e<sup>m_A − m_A∪B</sup>.
          </p>
        </Translated>
      </Aside>
      <Ask q={Q['ch20/20.5/q-a-term']} />
      <Translated
        vi={
          <p>
            <b>compute_P_and_update_D</b> thay mỗi điểm số trong S_i bằng e<sup>s − m</sup> theo max vừa cập nhật (các cột
            bị che nhận 0 theo chỉ số), cộng chúng trên warp và cộng tổng vào D.
          </p>
        }
      >
        <p>
          <b>compute_P_and_update_D</b> replaces each score in S_i by e<sup>s − m</sup> against the max just updated (masked
          columns get 0 by index), sums them across the warp and adds the sum to D.
        </p>
      </Translated>
      <Source section="20.5" pages="502–503" figs={tr('Fig. 20.15', 'Hình 20.15')} />
      <Translated
        vi={
          <p>
            Tổng đó là toàn bộ số hạng B của Eq. 20.4, đã ở hệ quy chiếu mới: mỗi số mũ trừ đi m<sub>A∪B</sub>, nên không bao
            giờ tạo ra D<sub>B</sub> và không có gì cần chỉnh tỉ lệ sau đó.
          </p>
        }
      >
        <p>
          That sum is the whole B term of Eq. 20.4, already in the new frame: each exponent subtracts m<sub>A∪B</sub>, so
          no D<sub>B</sub> is ever formed and nothing needs rescaling afterwards.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              <b>Errata, p. 502.</b> Văn bản nói tổng này là Σ e<sup>l − m_B</sup>. Mã trừ đi max đã cập nhật, nên nó tính Σ e
              <sup>l − m_A∪B</sup>, tức là D<sub>B</sub>·e<sup>m_B − m_A∪B</sup>.
            </p>
          }
        >
          <p>
            <b>Erratum, p. 502.</b> The text says this sum is Σ e<sup>l − m_B</sup>. The code subtracts the updated max, so
            it computes Σ e<sup>l − m_A∪B</sup>, which is D<sub>B</sub>·e<sup>m_B − m_A∪B</sup>.
          </p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            <b>compute_O</b> nhân mỗi phần tử O của lane với e<sup>last_m − m</sup> và chỉ sau đó mới cộng hàng P nhân với
            các cột V_j của lane. Sau ô cuối, <b>store_O</b> chia cho D, một lần. Nó chứa phép kiểm tra biên duy nhất của
            kernel.
          </p>
        }
      >
        <p>
          <b>compute_O</b> multiplies each of the lane's O elements by e<sup>last_m − m</sup> and only then adds the P row
          times the lane's columns of V_j. After the last tile, <b>store_O</b> divides by D, once. It has the kernel's only
          bounds test.
        </p>
      </Translated>
      <Source section="20.5" pages="501, 503–504" figs={tr('Figs. 20.13, 20.16', 'Hình 20.13, 20.16')} />
      <Ask q={Q['ch20/20.5/q-rescale-order']} />

      <h3>
        <Translated vi="Rào chắn mà sách bỏ sót">The barrier the book leaves out</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Hãy nhìn luồng dữ liệu giữa compute_P_and_update_D và compute_O. Lane L ghi phần tử P thứ L vào bộ nhớ chia sẻ;
            rồi mọi lane đọc cả B_c phần tử của hàng. Đó là một lần trao tay giữa các lane qua bộ nhớ chia sẻ, và đoạn mã
            không có rào chắn nào cho nó.
          </p>
        }
      >
        <p>
          Look at the data flow between compute_P_and_update_D and compute_O. Lane L writes P element L to shared memory;
          then every lane reads all B_c elements of the row. That is a hand-off between lanes through shared memory, and the
          listing has no barrier for it.
        </p>
      </Translated>
      <Source section="20.5" pages="501–503" figs={tr('Figs. 20.13, 20.15', 'Hình 20.13, 20.15')} />
      <Translated
        vi={
          <p>
            Các phép shuffle ở giữa trao đổi thanh ghi, nhưng hướng dẫn của NVIDIA nói rõ rằng chúng "do not guarantee any
            memory ordering". Một <code>__syncwarp()</code> giữa hai lời gọi là thứ các quy tắc đòi hỏi.
          </p>
        }
      >
        <p>
          The shuffles in between exchange registers, but NVIDIA's guide says plainly that they "do not guarantee any
          memory ordering". One <code>__syncwarp()</code> between the two calls is what the rules require.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md A3, B4, C4, D1)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              CUDA guide 5.4.6.5 về shuffle và 5.4.4.2 về <code>__syncwarp</code>. Các header của HIP cũng khai báo lời gọi
              này, với kiểu mask 64-bit cho dạng có mask (unverified: không có gì ở đây được build bằng ROCm); trên Metal
              thứ tương ứng là <code>simdgroup_barrier(mem_flags::mem_threadgroup)</code>. Kernel attention dạng vectơ của
              llama.cpp có <code>ggml_cuda_syncwarp(); // Needed for KQ synchronization.</code> ở cùng điểm đó.
            </p>
          }
        >
          <p>
            CUDA guide 5.4.6.5 on shuffles and 5.4.4.2 on <code>__syncwarp</code>. HIP's headers declare the call too, with a
            64-bit mask type for the masked form (unverified: nothing here has been built with ROCm); on Metal the
            counterpart is <code>simdgroup_barrier(mem_flags::mem_threadgroup)</code>. llama.cpp's vector attention kernel
            has <code>ggml_cuda_syncwarp(); // Needed for KQ synchronization.</code> at the same point.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Một sơ suất thứ hai thì kín đáo hơn. Fig. 20.12 và 20.15 khai báo một <code>TempStorage</code> CUB{' '}
            <code>__shared__</code> bên trong hàm trợ giúp, khiến nó chỉ có một cho mỗi khối, dùng chung bởi cả 16 warp. CUB
            ghi trong tài liệu là một cho mỗi warp. CUB ngày nay nói phép rút gọn warp của nó không dùng bộ nhớ chia sẻ, nên
            có thể không lộ ra, nhưng đó không phải là hợp đồng. Các lab viết tay các phép rút gọn dưới dạng butterfly xor,
            để kết quả ở mọi lane (trên Metal, <code>simd_max</code> và <code>simd_sum</code> làm điều tương tự).
          </p>
        }
      >
        <p>
          A second slip is quieter. Figs. 20.12 and 20.15 declare one <code>__shared__</code> CUB <code>TempStorage</code>
          inside the helper function, which makes it one per block, shared by all 16 warps. CUB documents one per warp.
          Today's CUB says its warp reduction does not use shared memory, so it may not show, but it is not the contract.
          The labs write the reductions by hand as an xor butterfly, which leaves the result in every lane (on Metal,{' '}
          <code>simd_max</code> and <code>simd_sum</code> do the same).
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md A2; infra A11)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              Trang WarpReduce của CUB: một TempStorage cho mỗi warp, một <code>__syncwarp()</code> trước khi dùng lại.{' '}
              <code>cub::Max()</code> và <code>cub::Sum()</code> đã bị gỡ bỏ trong CCCL 3.0.
            </p>
          }
        >
          <p>
            CUB's WarpReduce page: one TempStorage per warp, a <code>__syncwarp()</code> before reuse. <code>cub::Max()</code>{' '}
            and <code>cub::Sum()</code> were removed in CCCL 3.0.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Giờ hãy xem một khối làm tất cả những điều đó, rồi tự đi từng bước qua nó. Trình chạy từng bước chạy kernel đã
            sửa ở kích thước vừa màn hình. Dùng <i>Run to next barrier</i> để xem mỗi rào chắn chờ cái gì. Rồi bật từng lỗi
            một. <code>__syncwarp</code> bị thiếu không gây ra gì khi các lane còn đi đồng bộ từng bước: hãy cho chúng chạy
            lệch nhau, và nhấn <i>Go to the first failure</i>. Sơ suất "new max" không có lần đọc sai nào mà vẫn sai.
          </p>
        }
      >
        <p>
          Now watch one block do all of it, then step through it yourself. The stepper runs the corrected kernel at a size
          that fits on screen. Use <i>Run to next barrier</i> to see what each barrier waits for. Then switch on one defect
          at a time. The missing <code>__syncwarp</code> does nothing while the lanes stay in lockstep: let them run apart,
          and press <i>Go to the first failure</i>. The "new max" slip has no bad read at all and is still wrong.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Bốn lane mỗi warp thay vì 32, d = 8, và số học double precision, nên chỉ các lỗi cấu trúc lộ ra. Một lane tiến
              trọn một hàm mỗi lần. "Lanes run apart" cho mỗi lane chạy đến rào chắn kế tiếp trước khi lane kế bắt đầu, và
              một phép shuffle trao cho mọi lane giá trị mà các lane khác tính cho lời gọi đó bất cứ khi nào chúng đến: trao
              đổi thanh ghi được định nghĩa, thứ tự bộ nhớ thì không. Đó là trường hợp xấu nhất mà các quy tắc cho phép, không
              phải một khẳng định về điều một GPU cụ thể làm. Bộ nhớ chia sẻ chưa khởi tạo là NaN.
            </p>
          }
        >
          <p>
            Four lanes per warp instead of 32, d = 8, and arithmetic in double precision, so only structural mistakes show.
            A lane advances one whole function at a time. "Lanes run apart" lets each lane run to its next barrier before the
            next lane starts, and a shuffle hands every lane the value the others compute for that call whenever they get
            there: register exchange is defined, memory order is not. That is the worst case the rules allow, not a claim
            about what a given GPU does. Uninitialised shared memory is NaN.
          </p>
        </Translated>
      </Simplification>
      <FlashBlockScene />
      <FlashStepper />
      <FlashOutcomes />
      <Ask q={Q['ch20/20.5/q-syncwarp']} />

      <h2>
        <Translated vi="Kernel đã sửa">The corrected kernel</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Đây là lời giải tham chiếu của lab 20.2. Hãy tự viết bản của bạn trong lab trước khi đọc kỹ nó. Bảng liệt kê mọi
            chỗ nó khác đoạn mã như khi in, và lý do.
          </p>
        }
      >
        <p>
          Here is lab 20.2's reference solution. Write your own in the lab before reading it closely. The table lists every
          place where it departs from the listing as printed, and why.
        </p>
      </Translated>
      {cu && metal && mm ? (
        <Code
          tabs={[
            { label: 'CUDA', code: cu, tiers: FLASH_TIERS, status: st.cu },
            { label: tr('Metal kernel', 'Kernel Metal'), code: metal, tiers: FLASH_TIERS, status: st.metal },
            { label: tr('Metal host', 'Host Metal'), code: mm, tiers: {}, status: st.metal },
          ]}
          caption={
            <Translated
              vi={
                <>
                  Được tải từ <code>labs/ch20/lab2_flash/solution/</code>. Tệp CUDA cũng là mã nguồn cho HIP và trình giả
                  lập. Các trạng thái lấy từ các lần chạy trên máy này.
                </>
              }
            >
              <>
                Loaded from <code>labs/ch20/lab2_flash/solution/</code>. The CUDA file is also the HIP and emulator source.
                Statuses come from the runs on this machine.
              </>
            </Translated>
          }
        />
      ) : null}
      <table className="data wide" style={{ maxWidth: '58rem' }} data-corrections>
        <thead>
          <tr>
            <th>
              <Translated vi="Như khi in">As printed</Translated>
            </th>
            <th>
              <Translated vi="Như lab 20.2 viết">As lab 20.2 writes it</Translated>
            </th>
            <th>
              <Translated vi="Lý do">Why</Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              <Translated vi={<><code>NEG_INFINITY</code> cho các điểm số bị che và max khởi đầu</>}>
                <code>NEG_INFINITY</code> for masked scores and the initial max
              </Translated>
            </td>
            <td>
              <Translated vi={<><code>-FLT_MAX/2</code> làm giá trị khởi đầu; các cột bị che được bỏ qua theo chỉ số, không bao giờ lưu dưới dạng vô cực</>}>
                <code>-FLT_MAX/2</code> as the start; masked columns skipped by index, never stored as infinity
              </Translated>
            </td>
            <td>
              <Translated vi="NEG_INFINITY không phải tên của CUDA; (−∞) − (−∞) là NaN; vô cực là không xác định dưới fast math của Metal (A5, C5)">
                NEG_INFINITY is not a CUDA name; (−∞) − (−∞) is NaN; infinities are undefined under Metal fast math (A5, C5)
              </Translated>
            </td>
          </tr>
          <tr>
            <td>
              <Translated vi={<><code>WarpReduce</code> của CUB với một <code>TempStorage</code> dùng chung, kết quả ở lane 0, rồi một phép shuffle</>}>
                CUB <code>WarpReduce</code> with one shared <code>TempStorage</code>, result in lane 0, then a shuffle
              </Translated>
            </td>
            <td>
              <Translated vi={<>các butterfly xor viết tay cho max và tổng, kết quả ở mọi lane; <code>simd_max</code>, <code>simd_sum</code> trên Metal</>}>
                hand-written xor butterflies for max and sum, result in every lane; <code>simd_max</code>, <code>simd_sum</code> on Metal
              </Translated>
            </td>
            <td>
              <Translated vi="một TempStorage cho mỗi warp là cách dùng được CUB ghi trong tài liệu; các functor không còn nữa (A2, infra A11)">
                one TempStorage per warp is CUB's documented use; the functors are gone (A2, infra A11)
              </Translated>
            </td>
          </tr>
          <tr>
            <td>
              <Translated vi={<><code>__shfl_sync(0xFFFFFFFF, ...)</code> với độ rộng của phần cứng</>}>
                <code>__shfl_sync(0xFFFFFFFF, ...)</code> with the hardware's width
              </Translated>
            </td>
            <td>
              <Translated vi="một lane mask có kiểu và width 32 trên mọi phép shuffle">a typed lane mask and width 32 on every shuffle</Translated>
            </td>
            <td>
              <Translated vi="HIP cần mask 64-bit; trên wave64 độ rộng mặc định vượt qua ranh giới các warp logic (B1, B2)">
                HIP needs a 64-bit mask; on wave64 the default width crosses logical warps (B1, B2)
              </Translated>
            </td>
          </tr>
          <tr>
            <td>
              <Translated vi={<>các mảng <code>__shared__</code> được khai báo bên trong các hàm trợ giúp</>}>
                <code>__shared__</code> arrays declared inside the helpers
              </Translated>
            </td>
            <td>
              <Translated vi="khai báo trong kernel, truyền cho các hàm trợ giúp dưới dạng con trỏ">
                declared in the kernel, passed to the helpers as pointers
              </Translated>
            </td>
            <td>
              <Translated vi="Metal từ chối biến threadgroup trong hàm trợ giúp; tài liệu của AMD không nói gì (C2, B3)">
                Metal rejects threadgroup variables in helpers; AMD's documentation is silent (C2, B3)
              </Translated>
            </td>
          </tr>
          <tr>
            <td>
              <code>T_r = N / B_r</code>, <code>T_c = N / B_c</code>
            </td>
            <td>
              <Translated vi="làm tròn lên; các ô được điền 0 sau N; các cột ≥ N bị che theo chỉ số; các hàng ≥ N không được lưu">
                ceilings; tiles zero-filled past N; columns ≥ N masked by index; rows ≥ N not stored
              </Translated>
            </td>
            <td>
              <Translated vi="phép chia nguyên lặng lẽ bỏ các hàng cuối khi N không phải bội của kích thước ô (notes)">
                the integer division silently drops the last rows when N is not a multiple of the tiles (notes)
              </Translated>
            </td>
          </tr>
          <tr>
            <td>
              <Translated vi="B_c là bội của 32">B_c a multiple of 32</Translated>
            </td>
            <td>
              <Translated vi="B_c bằng 8, 16 hoặc 32; các lane không có cột mang phần tử đơn vị vào các phép rút gọn">
                B_c of 8, 16 or 32; lanes without a column bring the identity to the reductions
              </Translated>
            </td>
            <td>
              <Translated vi="giới hạn của Metal buộc B_c = 16 ở d = 128 (C1)">Metal's limit forces B_c = 16 at d = 128 (C1)</Translated>
            </td>
          </tr>
          <tr>
            <td>
              <Translated vi="không có rào chắn giữa lần ghi P và compute_O">no barrier between the P write and compute_O</Translated>
            </td>
            <td>
              <Translated vi={<><code>__syncwarp()</code>; <code>simdgroup_barrier(mem_flags::mem_threadgroup)</code> trên Metal</>}>
                <code>__syncwarp()</code>; <code>simdgroup_barrier(mem_flags::mem_threadgroup)</code> on Metal
              </Translated>
            </td>
            <td>
              <Translated vi="shuffle không sắp thứ tự bộ nhớ (A3, C4)">shuffles do not order memory (A3, C4)</Translated>
            </td>
          </tr>
          <tr>
            <td>
              <Translated vi={<><code>out_D</code>, mẫu số cuối cùng</>}>
                <code>out_D</code>, the final denominator
              </Translated>
            </td>
            <td>
              <Translated vi={<><code>L = m + log D</code>, logsumexp</>}>
                <code>L = m + log D</code>, the logsumexp
              </Translated>
            </td>
            <td>
              <Translated vi="chỉ D không thể dựng lại P; FlashAttention-2 lưu L (A7)">D alone cannot rebuild P; FlashAttention-2 stores L (A7)</Translated>
            </td>
          </tr>
          <tr>
            <td>
              <Translated vi="mọi ô cho mọi dải">every tile for every panel</Translated>
            </td>
            <td>
              <Translated vi={<>các ô nằm hoàn toàn phía trên đường chéo được bỏ qua (cận vòng lặp <code>lastTile</code>)</>}>
                tiles wholly above the diagonal skipped (loop bound <code>lastTile</code>)
              </Translated>
            </td>
            <td>
              <Translated vi="FlashAttention-2 làm vậy (A7); các ô bị che chỉ tốn thời gian">
                FlashAttention-2 does (A7); the masked tiles only cost time
              </Translated>
            </td>
          </tr>
          <tr>
            <td>
              <Translated vi={<>độ rộng head là macro <code>d</code></>}>
                the head width is the macro <code>d</code>
              </Translated>
            </td>
            <td>
              <Translated vi={<><code>HEAD_DIM</code>; <code>d_size</code> thành <code>D_SIZE</code></>}>
                <code>HEAD_DIM</code>; <code>d_size</code> is <code>D_SIZE</code>
              </Translated>
            </td>
            <td>
              <Translated vi="một macro tên d sẽ viết lại mọi chữ d trong tệp">a macro named d would rewrite every d in the file</Translated>
            </td>
          </tr>
          <tr>
            <td>
              <Translated vi={<>các kích thước dạng <code>#define</code></>}>
                <code>#define</code> sizes
              </Translated>
            </td>
            <td>
              <Translated vi={<>cùng các tên đó; trên Metal, được truyền dưới dạng <code>preprocessorMacros</code> khi build thư viện</>}>
                the same names; on Metal, passed as <code>preprocessorMacros</code> when the library is built
              </Translated>
            </td>
            <td>
              <Translated vi="d và các kích thước ô vẫn là hằng số lúc biên dịch trên mọi backend (C6)">
                d and the tile sizes stay compile-time constants on every backend (C6)
              </Translated>
            </td>
          </tr>
        </tbody>
      </table>
      <Ask q={Q['ch20/20.5/q-lse']} />

      <h2>
        <Translated vi="Kernel này áp dụng ở đâu">Where this kernel applies</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Đây là kernel prefill: nó hợp với transformer cơ sở và giai đoạn tóm tắt, nơi Q có nhiều hàng. Khi decode với KV
            cache, Q, S, P và O trở thành các hàng đơn. Một dải B_r hàng của Q không còn tồn tại, và cách chia việc của Fig.
            20.8 để phần lớn một khối nhàn rỗi. Câu trả lời của sách là gộp batch các yêu cầu để Q lại có nhiều hàng (mục{' '}
            <a href={href('ch20', '20.6')}>20.6</a>); lab 20.4 viết một kernel decode với cách chia việc riêng.
          </p>
        }
      >
        <p>
          This is the prefill kernel: it fits the baseline transformer and the summarisation phase, where Q has many rows. In
          decoding with a KV cache, Q, S, P and O become single rows. A panel of B_r rows of Q no longer exists, and the work
          division of Fig. 20.8 leaves most of a block idle. The book's answer is to batch requests so Q has rows again
          (section <a href={href('ch20', '20.6')}>20.6</a>); lab 20.4 writes a decode kernel with its own division of work.
        </p>
      </Translated>
      <Source section="20.5" pages="504" />
      <Translated
        vi={
          <p>
            Sách kết thúc mục này với FlashAttention-3 trên Hopper: TMA, WGMMA bất đồng bộ, FP8 và chuyên biệt hóa warp. Từ
            đó đến nay FlashAttention-4, viết bằng CuTeDSL cho Hopper và Blackwell, được phát hành dưới dạng beta. SDPA của
            cuDNN cài đặt thuật toán FlashAttention-2 trên SM80 trở lên ở fp16, bf16 và fp8, có hỗ trợ paged cache, và không
            có đường fp32. Trên ROCm, FlashAttention-2 chạy qua Composable Kernel hoặc Triton (aiter), và PyTorch dùng
            AOTriton và CK. Apple cung cấp <code>scaledDotProductAttention</code> của MPSGraph (macOS 15; tùy chọn causal từ
            macOS 27) mà không công bố thuật toán của nó, và MLX có các kernel attention phân ô &ldquo;steel&rdquo; của riêng
            mình.
          </p>
        }
      >
        <p>
          The book closes the section with FlashAttention-3 on Hopper: TMA, asynchronous WGMMA, FP8 and warp
          specialisation. Since then FlashAttention-4, written in CuTeDSL for Hopper and Blackwell, ships as betas. cuDNN&rsquo;s
          SDPA implements the FlashAttention-2 algorithm on SM80 and later in fp16, bf16 and fp8, with paged caches, and has
          no fp32 path. On ROCm, FlashAttention-2 runs through Composable Kernel or Triton (aiter), and PyTorch uses
          AOTriton and CK. Apple offers MPSGraph&rsquo;s <code>scaledDotProductAttention</code> (macOS 15; a causal option
          from macOS 27) without documenting its algorithm, and MLX has its own tiled &ldquo;steel&rdquo; attention kernels.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md A6, B5, C7, D4)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              Mọi thứ sau FlashAttention-3 đều không có trong sách; mỗi mục lấy từ trang riêng của dự án hoặc của hãng, được
              trích trong tệp facts.
            </p>
          }
        >
          <p>Everything after FlashAttention-3 is not in the book; each item is from the project&rsquo;s or vendor&rsquo;s own pages, quoted in the facts file.</p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Khi bạn đọc một backend production, mục này là thứ bạn đang tìm. Kernel vectơ CUDA của llama.cpp làm
            update_m_and_D, compute_P_and_update_D và phép chỉnh tỉ lệ của compute_O trong vài dòng, cùng với rào chắn warp
            mà mục này đã thêm vào:
          </p>
        }
      >
        <p>
          When you read a production backend, this section is what you are looking for. llama.cpp's CUDA vector kernel does
          update_m_and_D, compute_P_and_update_D and the rescale of compute_O in a handful of lines, with the warp barrier
          this section added:
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Cách đọc llama.cpp là của chúng tôi, từ mã nguồn của nó ngày 2026-11-20 (ch20 facts D1, D2). Tệp
              fattn-common.cuh của nó (<code>flash_attn_combine_results</code>) trộn các kết quả riêng phần của các khối riêng
              rẽ bằng Pt. 20.4 và 20.6, chính là phần cuối của scene.
            </p>
          }
        >
          <p>
            The llama.cpp reading is ours, from its sources on 2026-11-20 (ch20 facts D1, D2). Its fattn-common.cuh (
            <code>flash_attn_combine_results</code>) merges partial results of separate blocks with Eqs. 20.4 and 20.6,
            which is the scene's last beat.
          </p>
        </Translated>
      </Beyond>
      <Code
        tabs={[
          {
            label: tr('llama.cpp, fattn-vec.cuh (quoted)', 'llama.cpp, fattn-vec.cuh (trích dẫn)'),
            code: LLAMA,
            tiers: {},
            status: tr(
              'quoted from llama.cpp master as read on 2026-11-20 (facts D1); not compiled here',
              'trích từ nhánh master của llama.cpp như đọc ngày 2026-11-20 (facts D1); không biên dịch ở đây',
            ),
          },
        ]}
        caption={
          <Translated
            vi={
              <>
                KQ_max là m, KQ_sum là D, KQ_max_scale là hệ số chỉnh tỉ lệ, KQ là hàng P trong bộ nhớ chia sẻ. Kernel Metal
                của nó làm điều tương tự với <code>simd_max</code> và <code>simd_sum</code> (facts D2).
              </>
            }
          >
            <>
              KQ_max is m, KQ_sum is D, KQ_max_scale is the rescale factor, KQ is the P row in shared memory. Its Metal
              kernel does the same with <code>simd_max</code> and <code>simd_sum</code> (facts D2).
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Bạn sẽ viết kernel này, và cuộc săn race của nó, trên <a href={href('ch20', 'labs')}>trang lab</a>: lab 20.2 là
            lượt forward, và lab 20.3 là một cuộc săn race trong cùng kernel đó.
          </p>
        }
      >
        <p>
          You will write this kernel, and its race hunt, on the <a href={href('ch20', 'labs')}>labs page</a>: lab 20.2 is
          the forward pass, and lab 20.3 is a race hunt in the same kernel.
        </p>
      </Translated>
    </>
  );
}
