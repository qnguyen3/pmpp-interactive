import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Bench, Beyond, NoteGroup, Simplification, Source, TierLegend, Today } from '../../components/notes';
import type { Tier } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { useReports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  WARP_CLASS_LABEL,
  basicLineNumbers,
  basicTaps,
  conv2d,
  firstKernel,
  mapThread,
  ownerOf,
  partOf,
  pyramidFilter,
  stripsOf,
  toyImage,
  warpDivergence,
  warpsInCorner,
  type BasicLine,
  type WarpClass,
} from '../../sim/convolve';
import { widthChoices } from '../../sim/warps';
import { OneThreadScene } from './OneThreadScene';

// Section 7.2: one thread per output. The index mapping, the printed kernel and its
// corrections, a stepper through one thread's taps, which warps diverge at the edges,
// and the narrated animation. The kernel shown is lab 7.1's reference solution.

const LAB = 'ch07/lab1_constconv';
const SOL = LAB + '/solution';
const CONV_TIERS: Record<string, Tier> = {
  N: 'global', F: 'global', P: 'global',
  Pvalue: 'reg', outRow: 'reg', outCol: 'reg', inRow: 'reg', inCol: 'reg', fRow: 'reg', fCol: 'reg', gid: 'reg',
};
const big = (v: number) => v.toLocaleString('en-US');
const pct = (v: number) => (v === 0 ? '0' : v < 0.001 ? '<0.1' : String(Math.round(v * 1000) / 10));

export const questionsS72: Record<string, Question> = {
  'ch07/7.2/q-map': {
    kind: 'numeric',
    id: 'ch07/7.2/q-map',
    prompt: (
      <Translated
        vi={
          <>
            Một ảnh 16×16, khối 4×4, r = 2. Luồng <code>(tx 3, ty 0)</code> của khối <code>(bx 2, by 1)</code>. Chỉ số
            đầu vào phẳng của góc trên bên trái mảng con của nó là bao nhiêu?
          </>
        }
      >
        <>
          A 16×16 image, 4×4 blocks, r = 2. Thread <code>(tx 3, ty 0)</code> of block <code>(bx 2, by 1)</code>. What is
          the flat input index of its patch&rsquo;s upper-left corner?
        </>
      </Translated>
    ),
    answer: 41,
    explain: (
      <Translated
        vi={
          <p>
            outCol = 2·4 + 3 = 11 và outRow = 1·4 + 0 = 4, nên góc là (outRow − r, outCol − r) = (2, 9), và chỉ số phẳng
            của nó là 2·16 + 9 = 41. Đặt khối (2, 1) và luồng (3, 0) trong trình chạy từng bước của mục 7.2 để thấy điều
            đó.
          </p>
        }
      >
        <p>
          outCol = 2·4 + 3 = 11 and outRow = 1·4 + 0 = 4, so the corner is (outRow − r, outCol − r) = (2, 9), and its flat
          index is 2·16 + 9 = 41. Set block (2, 1) and thread (3, 0) in section 7.2&rsquo;s stepper to see it.
        </p>
      </Translated>
    ),
    source: 'ch. 7 §7.2, pp. 165–166',
  },
  'ch07/7.2/q-skip': {
    kind: 'numeric',
    id: 'ch07/7.2/q-skip',
    prompt: (
      <Translated vi={<>Với r = 2, luồng tính P[0][1] bỏ qua bao nhiêu trong 25 tap của nó?</>}>
        <>With r = 2, how many of its 25 taps does the thread computing P[0][1] skip?</>
      </Translated>
    ),
    answer: 13,
    unit: 'taps',
    explain: (
      <Translated
        vi={
          <p>
            Mảng con của nó phủ các hàng −2 đến 2 và các cột −1 đến 3. Các hàng 0 đến 2 (ba hàng) và các cột 0 đến 3 (bốn
            cột) nằm trong ảnh, nên 3·4 = 12 tap là thật và 25 − 12 = 13 tap bị bỏ qua. Luồng cho P[0][0] bỏ qua 16; ý của
            sách là mỗi luồng ở biên bỏ qua một số lượng khác nhau.
          </p>
        }
      >
        <p>
          Its patch covers rows −2 to 2 and columns −1 to 3. Rows 0 to 2 (three of them) and columns 0 to 3 (four) are
          inside, so 3·4 = 12 taps are real and 25 − 12 = 13 are skipped. The thread for P[0][0] skips 16; the book&rsquo;s
          point is that every edge thread skips a different number.
        </p>
      </Translated>
    ),
    source: 'ch. 7 §7.2, p. 166',
    vi: { unit: 'tap' },
  },
  'ch07/7.2/q-guard': {
    kind: 'choice',
    id: 'ch07/7.2/q-guard',
    prompt: (
      <Translated
        vi={
          <>
            Kernel được in trong sách không có lệnh chặn đầu ra. Chỉ sửa đủ để biên dịch được, nó chạy với bộ lọc 5×5 (r =
            2) trên ảnh 100×100 với khối 32×32 (một lưới 4×4 khối). Chuyện gì xảy ra?
          </>
        }
      >
        <>
          The printed kernel has no output guard. Corrected only so that it compiles, it runs with a 5×5 filter (r = 2)
          on a 100×100 image with 32×32 blocks (a 4×4 grid of them). What happens?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không có gì xấu: dòng 09 bỏ qua mọi tap nằm ngoài ảnh, nên các luồng vượt quá biên không chạm vào bộ nhớ nào.">
            Nothing bad: line 09 skips every tap outside the image, so threads past the edge touch no memory.
          </Translated>
        ),
        why: (
          <Translated vi="Dòng 09 kiểm tra nơi mỗi tap đọc, không phải nơi luồng ghi. Một luồng tại outCol = 101 vẫn chạy các tap của nó (tap ở cột 99 vượt qua), và dù tổng là gì, nó vẫn ghi ra.">
            Line 09 tests where each tap reads, not where the thread writes. A thread at outCol = 101 still runs its taps
            (the one at column 99 passes), and whatever it sums, it still stores.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các luồng vượt quá biên phải ghi vào đầu ra của hàng kế tiếp; các luồng vượt quá đáy ghi ra ngoài P.">
            {'Threads past the right edge store into the next row\'s outputs; threads past the bottom write beyond P.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Lần khởi chạy thất bại: runtime từ chối một lưới 128 × 128 luồng cho một ảnh 100 × 100.">
            The launch fails: the runtime rejects a grid of 128 × 128 threads for an image of 100 × 100.
          </Translated>
        ),
        why: (
          <Translated vi="Một lần khởi chạy không biết gì về ảnh. 16 khối 1024 luồng là một lần khởi chạy hợp lệ.">
            A launch knows nothing about the image. 16 blocks of 1024 threads is a valid launch.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các luồng thừa thấy mọi tap của chúng nằm ngoài ảnh, tổng bằng không, và lệnh ghi của chúng bị bỏ.">
            The extra threads find all their taps outside the image, sum zero, and their stores are dropped.
          </Translated>
        ),
        why: (
          <Translated vi="Một số đúng là có tổng bằng không (từ outCol = 102 trở đi, mọi tap đều ở ngoài), nhưng một lệnh ghi vào P[outRow*100 + outCol] đi tới bất cứ đâu chỉ số đó trỏ tới.">
            Some do sum zero (from outCol = 102 on, every tap is outside), but a store to P[outRow*100 + outCol] goes
            wherever that index points.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Lưới phủ 128×128 luồng. Một luồng có outRow = 5 và outCol = 101 ghi vào P[5·100 + 101] = P[601], tức là
              P[6][1]: một đầu ra mà luồng khác sở hữu. Giá trị nào còn lại tùy vào thời điểm. Một luồng có outRow ≥ 100,
              hoặc ở hàng 99 với outCol ≥ 100, ghi vượt quá 10,000 phần tử của P.
            </p>
            <p>
              Trên trình giả lập (emulator) của khóa học, vốn chạy dưới AddressSanitizer, lệnh ghi ngoài biên được báo
              lỗi. Trên GPU thì không có gì được bảo đảm: kết quả có thể trông đúng, hoặc dữ liệu khác có thể bị ghi đè.
              Sách giả định một lưới phủ vừa khít ảnh; một lệnh <code>if</code> ở đầu kernel loại bỏ giả định đó.
            </p>
          </>
        }
      >
        <>
          <p>
            The grid covers 128×128 threads. A thread with outRow = 5 and outCol = 101 stores to P[5·100 + 101] = P[601],
            which is P[6][1]: an output another thread owns. Which value survives depends on timing. A thread with outRow
            ≥ 100, or on row 99 with outCol ≥ 100, stores past the 10,000 elements of P.
          </p>
          <p>
            On the course&rsquo;s emulator, which runs under AddressSanitizer, the out-of-bounds store is reported. On a
            GPU nothing is guaranteed: the result may look right, or other data may be overwritten. The book assumes a
            grid that covers the image exactly; one <code>if</code> at the top of the kernel removes the assumption.
          </p>
        </>
      </Translated>
    ),
    source: 'our analysis of Fig. 7.7 (notes/ch07.md, Code listings)',
    vi: { source: 'phân tích của khóa học này về Hình 7.7 (notes/ch07.md, Code listings)' },
  },
  'ch07/7.2/q-warps1d': {
    kind: 'numeric',
    id: 'ch07/7.2/q-warps1d',
    prompt: (
      <Translated
        vi={
          <>
            Một phép tích chập 1D với mỗi đầu ra một luồng: 4096 đầu ra, r = 3, warp 32 luồng. Có bao nhiêu warp phân kỳ
            tại phép kiểm tra ô ma?
          </>
        }
      >
        <>
          A 1D convolution with one thread per output: 4096 outputs, r = 3, warps of 32. How many warps diverge at the
          ghost test?
        </>
      </Translated>
    ),
    answer: 2,
    unit: 'warps',
    explain: (
      <Translated
        vi={
          <p>
            Chỉ các đầu ra 0 đến 2 và 4093 đến 4095 có tap nằm ngoài mảng. Chúng nằm trong warp đầu tiên (đầu ra 0 đến 31)
            và warp cuối cùng (4064 đến 4095), và trong mỗi warp đó có những lane bỏ qua các tap mà lane bên cạnh không bỏ.
            126 warp còn lại trong 128 warp không bao giờ bất đồng. Chế độ đặt sẵn 1D của trình khám phá phân kỳ ở mục 7.2
            cho thấy điều đó.
          </p>
        }
      >
        <p>
          Only outputs 0 to 2 and 4093 to 4095 have taps outside the array. They sit in the first warp (outputs 0 to 31)
          and the last (4064 to 4095), and in each of those some lanes skip taps that their neighbours do not. The other
          126 of the 128 warps never disagree. The 1D preset of section 7.2&rsquo;s divergence explorer shows it.
        </p>
      </Translated>
    ),
    source: 'ch. 7 §7.2, p. 166; ch. 4 §4.5, pp. 79–82',
    vi: { unit: 'warp', source: 'ch. 7 §7.2, tr. 166; ch. 4 §4.5, tr. 79–82' },
  },
  'ch07/7.2/q-blockshape': {
    kind: 'choice',
    id: 'ch07/7.2/q-blockshape',
    prompt: (
      <Translated
        vi={
          <>
            Cùng ảnh và bộ lọc (r ≥ 1), warp 32 luồng. Với khối 32×32, rồi với khối 16×16: các warp có phân kỳ tại phép
            kiểm tra ô ma dọc theo cạnh <b>trên</b> của ảnh, xa các góc, không?
          </>
        }
      >
        <>
          Same image and filter (r ≥ 1), warps of 32. With 32×32 blocks, and then with 16×16 blocks: do warps diverge at
          the ghost test along the <b>top</b> edge of the image, away from the corners?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không với khối 32×32; có với khối 16×16.">Not with 32×32 blocks; yes with 16×16 blocks.</Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Có với cả hai: cạnh trên cũng là một cạnh.">Yes with both: the top edge is an edge.</Translated>,
        why: (
          <Translated vi="Phân kỳ cần các lane của một warp bất đồng với nhau. Hãy xem một warp trải qua những hàng nào.">
            Divergence needs lanes of one warp to disagree. Check which rows one warp spans.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không với cả hai: chỉ cạnh trái và cạnh phải mới có thể gây phân kỳ.">
            No with both: only the left and right edges can cause divergence.
          </Translated>
        ),
        why: (
          <Translated vi="Điều đó chỉ đúng khi một warp nằm gọn trong một hàng.">
            That is true only when a warp lies within one row.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Có với khối 32×32; không với khối 16×16.">Yes with 32×32 blocks; not with 16×16 blocks.</Translated>
        ),
        why: <Translated vi="Ngược lại mới đúng.">It is the other way round.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một warp là 32 luồng liên tiếp theo thứ tự x chạy nhanh nhất. Với khối rộng 32, đó đúng là một hàng của khối,
            nên mọi lane của một warp ở hàng 0 bỏ qua cùng các hàng của bộ lọc: không có bất đồng. Với khối rộng 16, một
            warp trải qua hai hàng, 0 và 1, vốn bỏ qua số hàng bộ lọc khác nhau, nên warp bị tách. Cùng kernel, cùng ảnh;
            chỉ hình dạng khối thay đổi.
          </p>
        }
      >
        <p>
          A warp is 32 consecutive threads in x-fastest order. With 32-wide blocks that is exactly one row of the block,
          so every lane of a warp on row 0 skips the same filter rows: no disagreement. With 16-wide blocks a warp spans
          two rows, 0 and 1, which skip different numbers of filter rows, so the warp splits. Same kernel, same image;
          only the block shape changed.
        </p>
      </Translated>
    ),
    source: 'our analysis with the explorer\'s model; ch. 4 §4.4, pp. 74–79',
    vi: { source: 'phân tích của khóa học này bằng mô hình của trình khám phá; ch. 4 §4.4, tr. 74–79' },
  },
};

// ---------------------------------------------------------------- erratum hunt
type Verdict = 'ok' | 'undeclared' | 'subscript';
const VERDICT_LABEL: Record<Verdict, string> = {
  get ok() {
    return tr('fine as printed', 'đúng như in');
  },
  get undeclared() {
    return tr('uses a variable that is never declared', 'dùng một biến chưa bao giờ được khai báo');
  },
  get subscript() {
    return tr('subscripts a float* twice', 'dùng chỉ số hai lần trên một float*');
  },
};
interface Printed {
  n: string;
  text: string;
  answer: Verdict;
  explain: ReactNode;
}
const PRINTED: Printed[] = [
  {
    n: '02',
    text: 'int outCol = blockIdx.x*blockDim.x + threadIdx.x;',
    answer: 'ok',
    explain: (
      <Translated
        vi={
          <>
            Đúng. Các biến dựng sẵn là số không dấu, nhưng kết quả được giữ trong một <code>int</code>, chính điều đó khiến{' '}
            <code>outCol - r</code> có thể âm về sau.
          </>
        }
      >
        <>
          Fine. The built-ins are unsigned, but the result is kept in an <code>int</code>, which is what makes{' '}
          <code>outCol - r</code> able to go negative later.
        </>
      </Translated>
    ),
  },
  {
    n: '07',
    text: 'inRow = outRow - r + fRow;',
    answer: 'undeclared',
    explain: (
      <Translated
        vi={
          <>
            <code>inRow</code> được gán nhưng chưa bao giờ được khai báo. Nó cần <code>int inRow = …</code>; trình biên
            dịch dừng ở đây với &ldquo;use of undeclared identifier&rdquo;.
          </>
        }
      >
        <>
          <code>inRow</code> is assigned but never declared. It needs <code>int inRow = …</code>; a compiler stops here with
          &ldquo;use of undeclared identifier&rdquo;.
        </>
      </Translated>
    ),
  },
  {
    n: '08',
    text: 'inCol = outCol - r + fCol;',
    answer: 'undeclared',
    explain: (
      <Translated vi={<>Tương tự với <code>inCol</code>.</>}>
        <>The same for <code>inCol</code>.</>
      </Translated>
    ),
  },
  {
    n: '09',
    text: 'if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {',
    answer: 'ok',
    explain: (
      <Translated
        vi={
          <>
            Đúng. Trình biên dịch cũng báo lỗi ở đây, nhưng chỉ vì 07 và 08 chưa khai báo hai tên đó; khai báo chúng là{' '}
            <code>int</code> thì dòng này biên dịch được. Với <code>unsigned</code>, hai phép kiểm tra <code>&gt;= 0</code>{' '}
            sẽ luôn đúng.
          </>
        }
      >
        <>
          Fine. A compiler complains here too, but only because 07 and 08 never declared the two names; declare them as{' '}
          <code>int</code> and this line compiles. With <code>unsigned</code> the two <code>&gt;= 0</code> tests would
          always pass.
        </>
      </Translated>
    ),
  },
  {
    n: '10',
    text: 'Pvalue += F[fRow][fCol]*N[inRow*width + inCol];',
    answer: 'subscript',
    explain: (
      <Translated
        vi={
          <>
            <code>F</code> là một <code>float*</code>, nên <code>F[fRow]</code> đã là một <code>float</code> và không thể
            dùng chỉ số thêm lần nữa. Trọng số là <code>F[fRow*(2*r+1) + fCol]</code>. Chỉ số vào <code>N</code> thì đúng.
          </>
        }
      >
        <>
          <code>F</code> is a <code>float*</code>, so <code>F[fRow]</code> is already a <code>float</code> and cannot be
          subscripted again. The weight is <code>F[fRow*(2*r+1) + fCol]</code>. The index into <code>N</code> is right.
        </>
      </Translated>
    ),
  },
  {
    n: '14',
    text: 'P[outRow][outCol] = Pvalue;',
    answer: 'subscript',
    explain: (
      <Translated
        vi={
          <>
            Cùng vấn đề với <code>P</code>: nó phải là <code>P[outRow*width + outCol]</code>. Đoạn mã kế tiếp của sách,
            Hình 7.9, in dòng này ở dạng đã tuyến tính hóa.
          </>
        }
      >
        <>
          The same problem for <code>P</code>: it must be <code>P[outRow*width + outCol]</code>. The book&rsquo;s next
          listing, Fig. 7.9, prints this line linearised.
        </>
      </Translated>
    ),
  },
];

function ErratumHunt({ onChecked }: { onChecked: () => void }) {
  const [pick, setPick] = useState<Record<string, Verdict | ''>>({});
  const [checked, setChecked] = useState(false);
  const right = PRINTED.filter((l) => pick[l.n] === l.answer).length;
  const all = PRINTED.every((l) => pick[l.n]);
  return (
    <Bench
      title={tr('Fig. 7.7 as printed', 'Hình 7.7 như được in')}
      task={tr(
        'For each printed line, say whether the line itself is wrong, and how. Then check.',
        'Với mỗi dòng được in, hãy cho biết bản thân dòng đó có sai không, và sai thế nào. Rồi kiểm tra.',
      )}
      foot={tr(
        "The quoted lines are the book's, as printed on p. 166, with its line numbers. They are shown because some are wrong; do not copy them.",
        'Các dòng được trích là của sách, đúng như in ở tr. 166, kèm số dòng của sách. Chúng được hiển thị vì một số dòng sai; đừng chép lại.',
      )}
    >
      <p className="muted" style={{ margin: '0 0 0.6rem' }}>
        <Translated vi="Dòng 01, để tham khảo:">Line 01, for reference:</Translated> <code>__global__ void convolution_2D_basic_kernel(float *N, float *F, float *P, int r, int width, int height)</code>
      </p>
      <table className="data" style={{ margin: 0, background: 'var(--page)' }}>
        <tbody>
          {PRINTED.map((l) => {
            const v = pick[l.n] ?? '';
            const ok = v === l.answer;
            return (
              <tr key={l.n} data-line={l.n} data-result={checked ? (ok ? 'right' : 'wrong') : ''}>
                <td className="mono" style={{ paddingLeft: '0.6rem', whiteSpace: 'nowrap' }}>{l.n}</td>
                <td style={{ minWidth: '16rem' }}>
                  <code>{l.text}</code>
                  {checked ? (
                    <div style={{ marginTop: '0.3rem' }}>
                      <b className={ok ? 't-ok' : 't-fail'}>{ok ? tr('Right.', 'Đúng.') : tr(`It ${l.answer === 'ok' ? 'is fine as printed' : l.answer === 'undeclared' ? 'uses a variable that is never declared' : 'subscripts a float* twice'}.`, `Dòng này ${l.answer === 'ok' ? 'đúng như in' : l.answer === 'undeclared' ? 'dùng một biến chưa bao giờ được khai báo' : 'dùng chỉ số hai lần trên một float*'}.`)}</b>{' '}
                      <span className="muted">{l.explain}</span>
                    </div>
                  ) : null}
                </td>
                <td>
                  <select value={v} disabled={checked} aria-label={tr(`Line ${l.n}`, `Dòng ${l.n}`)} onChange={(e) => setPick({ ...pick, [l.n]: e.target.value as Verdict })}>
                    <option value="">{tr('choose…', 'chọn…')}</option>
                    {(Object.keys(VERDICT_LABEL) as Verdict[]).map((k) => (
                      <option key={k} value={k}>{VERDICT_LABEL[k]}</option>
                    ))}
                  </select>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="row" style={{ marginTop: '0.8rem' }}>
        <button className="primary" disabled={checked || !all} onClick={() => { setChecked(true); onChecked(); }}>
          <Translated vi="Kiểm tra">Check</Translated>
        </button>
        {checked ? (
          <button className="quiet" onClick={() => { setChecked(false); setPick({}); }}>
            <Translated vi="Thử lại">Try again</Translated>
          </button>
        ) : null}
      </div>
      {checked ? (
        <div className={'q'} style={{ margin: '0.8rem 0 0', border: 0, padding: 0 }}>
          <div className={'verdict ' + (right === PRINTED.length ? 'right' : 'info')} data-out="hunt" data-right={right}>
            <Translated
              vi={
                <p>
                  <b>Đúng {right} trên {PRINTED.length}.</b> Bốn trong sáu dòng làm trình biên dịch dừng lại: hai chỗ thiếu
                  khai báo và hai chỗ dùng chỉ số hai lần trên một <code>float*</code>. Không có gì trên trang sách sai về
                  số học.
                </p>
              }
            >
              <p>
                <b>{right} of {PRINTED.length} right.</b> Four of the six lines stop a compiler: two missing declarations
                and two double subscripts on a <code>float*</code>. Nothing on the page is wrong in its arithmetic.
              </p>
            </Translated>
            <Translated
              vi={
                <p>
                  Còn một vấn đề nữa mà trình biên dịch không thấy được: kernel không có lệnh chặn đầu ra, nên nó chỉ đúng khi
                  lưới phủ vừa khít ảnh. Kernel đã sửa bên dưới thêm một lệnh như vậy, và câu hỏi sau nó cho thấy điều gì sai
                  khi thiếu nó.
                </p>
              }
            >
              <p>
                One more problem is invisible to a compiler: the kernel has no output guard, so it is correct only when the
                grid covers the image exactly. The corrected kernel below adds one, and the question after it shows what
                goes wrong without it.
              </p>
            </Translated>
          </div>
        </div>
      ) : null}
    </Bench>
  );
}

function useLabStatus() {
  const { reports, loading } = useReports();
  if (loading) return { cuda: tr('reading reports', 'đang đọc báo cáo'), metal: tr('reading reports', 'đang đọc báo cáo') };
  // The labs page's words (LabKit's solutionTabStatus), so a failed run reads as failed here too.
  return { cuda: solutionTabStatus(reports, LAB, 'kernel.cu'), metal: solutionTabStatus(reports, LAB, 'kernel.metal') };
}

function CorrectedKernel() {
  const cu = useLabFile(SOL + '/kernel.cu');
  const metal = useLabFile(SOL + '/kernel.metal');
  const mm = useLabFile(SOL + '/launch.mm');
  const status = useLabStatus();
  if (!cu || !metal || !mm)
    return (
      <p className="muted">
        <Translated vi="Đang tải lời giải tham chiếu của lab 7.1.">Loading lab 7.1&rsquo;s reference solution.</Translated>
      </p>
    );
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: partOf(cu, 'basic'), tiers: CONV_TIERS, status: status.cuda },
        { label: tr('Metal kernel', 'Kernel Metal'), code: partOf(metal, 'basic'), tiers: CONV_TIERS, status: status.metal },
        { label: tr('Metal host', 'Mã host Metal'), code: partOf(mm, 'basic'), tiers: {}, status: status.metal },
      ]}
      caption={
        <Translated
          vi={
            <>
              Kernel cơ bản từ lời giải tham chiếu của lab 7.1, tải từ <code>labs/{SOL}/</code> (phần được đánh dấu{' '}
              <code>@part basic</code>); trạng thái bên cạnh các tab được đọc từ các lần chạy lab đã ghi lại trên máy này.
              Nó khác trang sách ở bốn chỗ: <code>int</code> cho <code>inRow</code> và <code>inCol</code>, các chỉ số{' '}
              <code>F</code> và <code>P</code> đã tuyến tính hóa, và lệnh chặn đầu ra, là phần của khóa học này. Bản build
              HIP được thiết lập để nhận nguyên tệp <code>.cu</code> này qua <code>pmpp/vendors/hip.h</code>; nó chưa được
              biên dịch hay chạy trên phần cứng AMD (unverified).
            </>
          }
        >
          <>
            The basic kernel from lab 7.1&rsquo;s reference solution, loaded from <code>labs/{SOL}/</code> (the part marked{' '}
            <code>@part basic</code>); the status beside the tabs is read from this machine&rsquo;s recorded lab runs. It
            differs from the page in four places: <code>int</code> on <code>inRow</code> and <code>inCol</code>, the
            linearised <code>F</code> and <code>P</code> indices, and the output guard, which is ours. The HIP build is
            set up to take this <code>.cu</code> file unchanged through <code>pmpp/vendors/hip.h</code>; it has not been
            compiled or run on AMD hardware (unverified).
          </>
        </Translated>
      }
    />
  );
}

function HuntAndKernel() {
  const [show, setShow] = useState(false);
  return (
    <>
      <ErratumHunt onChecked={() => setShow(true)} />
      {show ? (
        <CorrectedKernel />
      ) : (
        <p className="ui">
          <Translated vi="Kernel đã sửa sẽ hiện ở đây khi bạn đã kiểm tra câu trả lời.">
            The corrected kernel appears here once you have checked your answers.
          </Translated>{' '}
          <button className="quiet" onClick={() => setShow(true)}>
            <Translated vi="Hiện kernel đã sửa ngay">Show the corrected kernel now</Translated>
          </button>
        </p>
      )}
    </>
  );
}

// ---------------------------------------------------------------- stepper: from thread to taps
const W = 16;
const TOY = toyImage(W, W);

function Stepper() {
  const cu = useLabFile(SOL + '/kernel.cu');
  const code = useMemo(() => (cu ? firstKernel(partOf(cu, 'basic')) : null), [cu]);
  const lines = useMemo(() => (code ? basicLineNumbers(code) : {}), [code]);
  const [bd, setBd] = useState(4);
  const [bx, setBx] = useState(1);
  const [by, setBy] = useState(1);
  const [tx, setTx] = useState(1);
  const [ty, setTy] = useState(1);
  const [r, setR] = useState(2);
  const [n, setN] = useState(0);
  const [playing, setPlaying] = useState(false);

  const m = 2 * r + 1;
  const F = useMemo(() => pyramidFilter(r), [r]);
  const ref = useMemo(() => conv2d(TOY, F), [F]);
  const map = mapThread({ bx, by, tx, ty, bdx: bd, bdy: bd }, r);
  const run = basicTaps(W, W, r, map.outRow, map.outCol, TOY.data, F.data);
  const total = run.taps.length + 1;
  const at = Math.min(n, total);
  const tap = at >= 1 && at <= run.taps.length ? run.taps[at - 1] : null;
  const stored = at === total;
  const blocks = W / bd;

  useEffect(() => setN(0), [bd, bx, by, tx, ty, r]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setN((v) => v + 1), 380);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);

  const pick = (row: number, col: number) => {
    const o = ownerOf(row, col, bd, bd);
    setBx(o.bx);
    setBy(o.by);
    setTx(o.tx);
    setTy(o.ty);
  };
  const setBlockDim = (v: number) => {
    const o = ownerOf(map.outRow, map.outCol, v, v);
    setBd(v);
    setBx(o.bx);
    setBy(o.by);
    setTx(o.tx);
    setTy(o.ty);
  };

  const hl = (keys: BasicLine[]) => keys.flatMap((k) => (lines[k] ? [lines[k]!] : []));
  const highlight = at === 0 ? hl(['outCol', 'outRow']) : tap ? hl(tap.inside ? ['inRow', 'inCol', 'test', 'accumulate'] : ['inRow', 'inCol', 'test']) : hl(['store']);
  const pvalue = tap ? tap.pvalue! : at === 0 ? 0 : run.pvalue;

  const where = (t: typeof tap) =>
    !t ? '' : t.inRow < 0 ? tr(`row ${t.inRow} is above the image`, `hàng ${t.inRow} nằm phía trên ảnh`) : t.inRow >= W ? tr(`row ${t.inRow} is below the image`, `hàng ${t.inRow} nằm phía dưới ảnh`) : t.inCol < 0 ? tr(`column ${t.inCol} is left of the image`, `cột ${t.inCol} nằm bên trái ảnh`) : tr(`column ${t.inCol} is right of the image`, `cột ${t.inCol} nằm bên phải ảnh`);

  let say: ReactNode;
  if (at === 0)
    say = (
      <Translated
        vi={
          <>
            outCol = blockIdx.x·blockDim.x + threadIdx.x = {bx}·{bd} + {tx} = <b>{map.outCol}</b>; outRow = blockIdx.y·blockDim.y +
            threadIdx.y = {by}·{bd} + {ty} = <b>{map.outRow}</b>. Mảng con phủ các hàng {map.outRow - r} đến {map.outRow + r} và
            các cột {map.outCol - r} đến {map.outCol + r}; góc trên bên trái của nó là (outRow − r, outCol − r) ={' '}
            <b>({map.cornerRow}, {map.cornerCol})</b>. Nhấn Bước để đi qua các tap.
          </>
        }
      >
        <>
          outCol = blockIdx.x·blockDim.x + threadIdx.x = {bx}·{bd} + {tx} = <b>{map.outCol}</b>; outRow = blockIdx.y·blockDim.y +
          threadIdx.y = {by}·{bd} + {ty} = <b>{map.outRow}</b>. The patch covers rows {map.outRow - r} to {map.outRow + r} and
          columns {map.outCol - r} to {map.outCol + r}; its upper-left corner is (outRow − r, outCol − r) ={' '}
          <b>({map.cornerRow}, {map.cornerCol})</b>. Press Step to walk the taps.
        </>
      </Translated>
    );
  else if (tap)
    say = (
      <Translated
        vi={
          <>
            Tap {at} trên {m * m} (fRow {tap.fRow}, fCol {tap.fCol}): inRow = {map.outRow} − {r} + {tap.fRow} = {tap.inRow}, inCol ={' '}
            {map.outCol} − {r} + {tap.fCol} = {tap.inCol}.{' '}
            {tap.inside ? (
              <>
                Phép kiểm tra ô ma vượt qua. Pvalue += F[{tap.fRow}·{m} + {tap.fCol}] × N[{tap.inRow}·{W} + {tap.inCol}] = F[{tap.fIndex}] × N[{tap.nIndex}] ={' '}
                {tap.weight} × {tap.value} = {tap.product}. Pvalue giờ là {tap.pvalue}.
              </>
            ) : (
              <>Phép kiểm tra ô ma thất bại ({where(tap)}), nên tap bị bỏ qua: giống như nhân với một ô ma bằng không. Pvalue vẫn là {tap.pvalue}.</>
            )}
          </>
        }
      >
        <>
          Tap {at} of {m * m} (fRow {tap.fRow}, fCol {tap.fCol}): inRow = {map.outRow} − {r} + {tap.fRow} = {tap.inRow}, inCol ={' '}
          {map.outCol} − {r} + {tap.fCol} = {tap.inCol}.{' '}
          {tap.inside ? (
            <>
              The ghost test passes. Pvalue += F[{tap.fRow}·{m} + {tap.fCol}] × N[{tap.inRow}·{W} + {tap.inCol}] = F[{tap.fIndex}] × N[{tap.nIndex}] ={' '}
              {tap.weight} × {tap.value} = {tap.product}. Pvalue is now {tap.pvalue}.
            </>
          ) : (
            <>The ghost test fails ({where(tap)}), so the tap is skipped: the same as multiplying by a zero ghost cell. Pvalue stays {tap.pvalue}.</>
          )}
        </>
      </Translated>
    );
  else
    say = (
      <Translated
        vi={
          <>
            Sau các vòng lặp, một lệnh ghi: P[outRow·width + outCol] = P[{map.outRow}·{W} + {map.outCol}] = P[{run.store}] ← {run.pvalue}.{' '}
            {run.skipped} trên {m * m} tap đã bị bỏ qua.
          </>
        }
      >
        <>
          After the loops, one store: P[outRow·width + outCol] = P[{map.outRow}·{W} + {map.outCol}] = P[{run.store}] ← {run.pvalue}.{' '}
          {run.skipped} of {m * m} taps were skipped.
        </>
      </Translated>
    );

  const inPatch = (row: number, col: number) => Math.abs(row - map.outRow) <= r && Math.abs(col - map.outCol) <= r;
  const doneUpTo = tap ? at : stored ? run.taps.length : 0;
  const tapAt = new Map(run.taps.slice(0, doneUpTo).map((t) => [t.inRow * 100 + t.inCol, t]));
  const cell = { minWidth: '1.25rem', height: '1.25rem', fontSize: '0.62rem' };
  const G = W + 2 * r;

  return (
    <Bench
      title={tr('From thread to taps', 'Từ luồng đến các tap')}
      task={tr(
        'Pick a block and a thread, or click an output. Then step through its taps; try a corner and an edge.',
        'Chọn một khối và một luồng, hoặc nhấn vào một đầu ra. Rồi đi từng bước qua các tap của nó; hãy thử một góc và một cạnh.',
      )}
      foot={
        <Translated
          vi={
            <>
              Ảnh có hình dạng của ảnh mẫu 16×16 trong Hình 7.6, được điền các số nguyên nhỏ của khóa học này để bạn có thể
              tự kiểm tra các tổng bằng tay. Với r = 2, bộ lọc là bộ lọc 5×5 của sách trong Hình 7.4; với r = 1 và 3, nó là
              bộ lọc 3×3 và 7×7 có cùng dạng kim tự tháp (của khóa học này). <TierLegend tiers={['reg', 'global']} />
            </>
          }
        >
          <>
            The image has the shape of Fig. 7.6&rsquo;s 16×16 toy, filled with small integers of ours so you can check
            the sums by hand. For r = 2 the filter is the book&rsquo;s 5×5 filter of Fig. 7.4; for r = 1 and 3 it is a
            3×3 and a 7×7 filter of the same pyramid shape (ours). <TierLegend tiers={['reg', 'global']} />
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('Block size', 'Kích thước khối')}>
          {[4, 8].map((v) => (
            <button key={v} aria-pressed={bd === v} onClick={() => setBlockDim(v)}>
              <Translated vi={<>khối {v}×{v}{v === 4 ? ' (ví dụ nhỏ của sách)' : ''}</>}>
                {v}×{v} blocks{v === 4 ? " (the book's toy)" : ''}
              </Translated>
            </button>
          ))}
        </span>
        <label className="field">
          <Translated vi="blockIdx (x, y)">blockIdx (x, y)</Translated>
          <select value={`${bx},${by}`} aria-label="blockIdx" onChange={(e) => { const [x, y] = e.target.value.split(',').map(Number); setBx(x); setBy(y); }}>
            {Array.from({ length: blocks * blocks }, (_, k) => [k % blocks, Math.floor(k / blocks)]).map(([x, y]) => (
              <option key={`${x},${y}`} value={`${x},${y}`}>({x}, {y})</option>
            ))}
          </select>
        </label>
        <label className="field">
          <Translated vi="threadIdx (x, y)">threadIdx (x, y)</Translated>
          <select value={`${tx},${ty}`} aria-label="threadIdx" onChange={(e) => { const [x, y] = e.target.value.split(',').map(Number); setTx(x); setTy(y); }}>
            {Array.from({ length: bd * bd }, (_, k) => [k % bd, Math.floor(k / bd)]).map(([x, y]) => (
              <option key={`${x},${y}`} value={`${x},${y}`}>({x}, {y})</option>
            ))}
          </select>
        </label>
        <span className="seg" role="group" aria-label={tr('Radius', 'Bán kính')}>
          {[1, 2, 3].map((v) => (
            <button key={v} aria-pressed={r === v} onClick={() => setR(v)}>r = {v}</button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setN(0); }} disabled={at === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => { setPlaying(false); setN(at - 1); }} disabled={at === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => { setPlaying(false); setN(at + 1); }} disabled={at >= total}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? tr('Pause', 'Tạm dừng') : tr('Play', 'Chạy')}</button>
        <button onClick={() => { setPlaying(false); setN(total); }} disabled={at >= total}><Translated vi="Chạy đến lệnh ghi">Run to the store</Translated></button>
        <span className="muted" aria-live="polite"><Translated vi={<>bước {at} trên {total}</>}>step {at} of {total}</Translated></span>
      </div>
      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }}
        data-out="step" data-at={at} data-outrow={map.outRow} data-outcol={map.outCol} data-corner={`${map.cornerRow},${map.cornerCol}`}
        data-pvalue={pvalue} data-ref={ref.data[map.outRow * W + map.outCol]}>
        {say}
      </div>
      <div className="row" style={{ alignItems: 'flex-start', gap: '1rem' }}>
        <div>
          <div className="mx-label t-global">
            <Translated vi="N (đầu vào, bộ nhớ toàn cục), ô ma có viền nét đứt">N (input, global memory), ghost cells dashed</Translated>
          </div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${G}, auto)`, gap: '1px' }} data-grid="N">
            {Array.from({ length: G * G }, (_, k) => {
              const row = Math.floor(k / G) - r;
              const col = (k % G) - r;
              const ghost = row < 0 || row >= W || col < 0 || col >= W;
              const on = inPatch(row, col);
              const isCur = tap && tap.inRow === row && tap.inCol === col;
              if (ghost)
                return on ? (
                  <span key={k} className="void" style={{ ...cell, ...(isCur ? { borderColor: 'var(--ink)', borderWidth: 2 } : {}) }}>
                    {tapAt.has(row * 100 + col) || isCur ? '·' : ''}
                  </span>
                ) : (
                  <span key={k} style={{ ...cell, visibility: 'hidden' }} />
                );
              const done = tapAt.has(row * 100 + col);
              return (
                <span key={k} className={isCur ? 'hot' : done || (on && at === 0) ? 'on' : ''} style={cell}>
                  {TOY.data[row * W + col]}
                </span>
              );
            })}
          </div>
        </div>
        <div>
          <p style={{ margin: '0 0 0.8rem' }}>
            <span className="chip reg" data-out="pvalue">Pvalue = {pvalue}</span>{' '}
            {stored ? (
              run.pvalue === ref.data[map.outRow * W + map.outCol] ? (
                <span className="chip ok" data-out="match">
                  <Translated vi={<>khớp với giá trị tham chiếu, {ref.data[map.outRow * W + map.outCol]}</>}>
                    matches the reference, {ref.data[map.outRow * W + map.outCol]}
                  </Translated>
                </span>
              ) : (
                <span className="chip fail" data-out="match">
                  <Translated vi="khác với giá trị tham chiếu">differs from the reference</Translated>
                </span>
              )
            ) : null}
          </p>
          <div className="mx-label t-global">F ({m}×{m})</div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${m}, auto)`, gap: '1px' }}>
            {F.data.map((v, k) => (
              <span key={k} className={tap && tap.fIndex === k ? 'hot' : ''} style={cell}>{v}</span>
            ))}
          </div>
          <div className="mx-label t-global" style={{ marginTop: '0.8rem' }}>
            <Translated vi="P (nhấn vào một đầu ra)">P (click an output)</Translated>
          </div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${W}, auto)`, gap: '1px' }} data-grid="P">
            {Array.from({ length: W * W }, (_, k) => {
              const row = Math.floor(k / W);
              const col = k % W;
              const sel = row === map.outRow && col === map.outCol;
              const shade = (Math.floor(row / bd) + Math.floor(col / bd)) % 2 === 1;
              return (
                <span key={k} role="button" tabIndex={-1} aria-label={`P[${row}][${col}]`} data-rc={`${row},${col}`}
                  className={sel ? 'hot' : ''} onClick={() => pick(row, col)}
                  style={{ ...cell, minWidth: '0.95rem', height: '0.95rem', cursor: 'pointer', ...(sel ? {} : { background: shade ? 'var(--rule-soft)' : 'var(--cell)' }) }} />
              );
            })}
          </div>
          <p className="muted" style={{ margin: '0.3rem 0 0', fontSize: '0.78rem' }}>
            <Translated vi="Màu nền xen kẽ theo khối.">Shading alternates by block.</Translated>
          </p>
        </div>
      </div>
      <div style={{ marginTop: '1rem' }}>
        {code ? (
          <Code tabs={[{ label: tr('kernel.cu (lab 7.1)', 'kernel.cu (bài thực hành 7.1)'), code, tiers: CONV_TIERS }]} highlight={highlight} />
        ) : (
          <p className="muted">
            <Translated vi={<>Đang tải kernel từ labs/{SOL}/kernel.cu.</>}>Loading the kernel from labs/{SOL}/kernel.cu.</Translated>
          </p>
        )}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: which warps diverge
const SIZES = [1, 16, 64, 100, 256, 1000, 1024, 4096];
const SHAPES: [number, number][] = [[32, 1], [8, 8], [16, 16], [32, 32]];
const CLS_FILL: Record<WarpClass, string> = {
  uniform: 'color-mix(in srgb, var(--ink) 14%, var(--cell))', // i18n-ignore: CSS value
  ghost: 'var(--ink)',
  guard: 'color-mix(in srgb, var(--ink) 55%, var(--cell))', // i18n-ignore: CSS value
  idle: 'var(--cell)',
};
const HATCH = 'ch07-warp-idle';

function Swatch({ cls }: { cls: WarpClass }) {
  return (
    <svg width="14" height="12" style={{ verticalAlign: '-1px', marginRight: '0.3rem' }} aria-hidden="true">
      <rect x="0.5" y="0.5" width="13" height="11" fill={CLS_FILL[cls]} stroke="var(--rule)" />
      {cls === 'idle' ? <rect x="0.5" y="0.5" width="13" height="11" fill={`url(#${HATCH})`} /> : null}
    </svg>
  );
}

function WarpsExplorer() {
  const { reports } = useReports();
  const { progress } = useProgress();
  const choices = useMemo(() => widthChoices(reports), [reports]);
  const [width, setWidth] = useState(256);
  const [height, setHeight] = useState(256);
  const [shape, setShape] = useState(2);
  const [r, setR] = useState(2);
  const [wkey, setWkey] = useState('book');
  const choice = choices.find((c) => c.key === wkey) ?? choices[0];
  const warp = choice.width;
  const [bxs, bys] = SHAPES[shape];
  const d = useMemo(() => warpDivergence(width, height, bxs, bys, r, warp), [width, height, bxs, bys, r, warp]);
  // The question "One dimension" asks for this case: its counts and pictures wait until it is answered.
  const locked = width === 4096 && height === 1 && r === 3 && !progress.answers['ch07/7.2/q-warps1d'];
  const strips = useMemo(() => (locked ? [] : stripsOf(d.marked)), [d, locked]);

  // Overview: the whole grid of threads, scaled; thin strips are drawn at least 2 px wide.
  const gw = d.blocksX * bxs;
  const gh = d.blocksY * bys;
  const S = 300;
  let sx = S / Math.max(gw, gh);
  let sy = sx;
  if (gh * sy < 18) sy = 18 / gh;
  if (gw * sx < 18) sx = 18 / gw;
  const OW = gw * sx;
  const OH = gh * sy;

  // Close-up: the top-left corner, two blocks each way, between 32 and 64 threads.
  const zc = Math.min(gw, Math.max(32, Math.min(64, 2 * bxs)));
  const zr = Math.min(gh, Math.max(1, Math.min(64, Math.max(32, 2 * bys))));
  const corner = useMemo(() => warpsInCorner(width, height, bxs, bys, r, warp, zr, zc), [width, height, bxs, bys, r, warp, zr, zc]);
  // Square cells, except that a close-up only a few threads tall gets taller cells.
  const zsx = Math.min(300 / zc, 300 / Math.max(zr, 8));
  const zsy = zr < 8 ? 20 : zsx;
  const ZW = zc * zsx;
  const ZH = zr * zsy;

  const oneD = () => {
    setWidth(4096);
    setHeight(1);
    setShape(0);
    setR(3);
    setWkey('book');
  };

  return (
    <Bench
      title={tr('Which warps diverge', 'Warp nào phân kỳ')}
      task={tr(
        'Change the image size, the block shape, the radius and the warp width. Watch which warps split, and where.',
        'Thay đổi kích thước ảnh, hình dạng khối, bán kính và độ rộng warp. Quan sát warp nào bị tách, và ở đâu.',
      )}
      foot={
        <Translated
          vi={
            <>
              Các luồng được cắt thành warp theo thứ tự x chạy nhanh nhất, quy tắc mà CUDA ghi trong tài liệu.{' '}
              <Swatch cls="uniform" />{WARP_CLASS_LABEL.uniform}; <Swatch cls="ghost" />{WARP_CLASS_LABEL.ghost};{' '}
              <Swatch cls="guard" />{WARP_CLASS_LABEL.guard} (và có thể cả phép kiểm tra ô ma);{' '}
              <Swatch cls="idle" />{WARP_CLASS_LABEL.idle}.
            </>
          }
        >
          <>
            Threads are cut into warps in x-fastest order, the rule CUDA documents. <Swatch cls="uniform" />{WARP_CLASS_LABEL.uniform};{' '}
            <Swatch cls="ghost" />{WARP_CLASS_LABEL.ghost}; <Swatch cls="guard" />{WARP_CLASS_LABEL.guard} (and maybe the ghost
            test too); <Swatch cls="idle" />{WARP_CLASS_LABEL.idle}.
          </>
        </Translated>
      }
    >
      <svg width="0" height="0" style={{ position: 'absolute' }} aria-hidden="true">
        <defs>
          <pattern id={HATCH} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" strokeWidth="1.1" />
          </pattern>
        </defs>
      </svg>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Chiều rộng ảnh">Image width</Translated>
          <select value={width} onChange={(e) => setWidth(+e.target.value)} aria-label={tr('Image width', 'Chiều rộng ảnh')}>
            {SIZES.filter((v) => v > 1).map((v) => <option key={v} value={v}>{big(v)}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Chiều cao ảnh">Image height</Translated>
          <select value={height} onChange={(e) => setHeight(+e.target.value)} aria-label={tr('Image height', 'Chiều cao ảnh')}>
            {SIZES.map((v) => <option key={v} value={v}>{big(v)}</option>)}
          </select>
        </label>
        <span className="seg" role="group" aria-label={tr('Block shape', 'Hình dạng khối')}>
          {SHAPES.map(([x, y], k) => (
            <button key={k} aria-pressed={shape === k} onClick={() => setShape(k)}>{x}×{y}</button>
          ))}
        </span>
        <label className="field">
          <Translated vi={<>Bán kính r = {r}</>}>Radius r = {r}</Translated>
          <input type="range" min={1} max={7} value={r} onChange={(e) => setR(+e.target.value)} aria-label={tr('Radius', 'Bán kính')} style={{ width: '8rem' }} />
        </label>
        <label className="field">
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <select value={choice.key} onChange={(e) => setWkey(e.target.value)} aria-label={tr('Warp width', 'Độ rộng warp')}>
            {choices.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
          </select>
        </label>
        <button onClick={oneD}><Translated vi="1D: 4096 đầu ra, r = 3">1D: 4096 outputs, r = 3</Translated></button>
      </div>
      {locked ? (
        <p className="ui" style={{ margin: '0 0 0.8rem' }} data-out="warps" data-locked="q-warps1d" data-warps={d.warps} data-width={warp}>
          <Translated
            vi={
              <>
                {big(d.blocksX * d.blocksY)} khối {bxs}×{bys} luồng, mỗi khối {d.warpsPerBlock} warp:{' '}
                <b>{big(d.warps)}</b> warp. Có bao nhiêu warp trong số đó phân kỳ là câu hỏi &ldquo;Một chiều&rdquo; bên
                dưới: hãy trả lời nó để xem con số và các hình vẽ.
              </>
            }
          >
            {big(d.blocksX * d.blocksY)} blocks of {bxs}×{bys} threads, {d.warpsPerBlock} warp{d.warpsPerBlock === 1 ? '' : 's'} each:{' '}
            <b>{big(d.warps)}</b> warps. How many of them diverge is the question &ldquo;One dimension&rdquo; below: answer it
            to see the count and the pictures.
          </Translated>
        </p>
      ) : (
        <p className="ui" style={{ margin: '0 0 0.8rem' }} data-out="warps" data-warps={d.warps} data-ghost={d.ghost} data-guard={d.guard}
          data-divergent={d.divergent} data-idle={d.idle} data-width={warp}>
          <Translated
            vi={
              <>
                {big(d.blocksX * d.blocksY)} khối {bxs}×{bys} luồng, mỗi khối {d.warpsPerBlock} warp:{' '}
                <b>{big(d.warps)}</b> warp. Tại phép kiểm tra ô ma có <b>{big(d.ghost)}</b> warp phân kỳ; tại lệnh chặn đầu
                ra có <b>{big(d.guard)}</b>. Tính cả hai, <b>{big(d.divergent)}</b> ({pct(d.divergent / d.warps)} %).{' '}
                {d.idle ? <>{big(d.idle)} warp hoàn toàn không có đầu ra nào.</> : null}
              </>
            }
          >
            {big(d.blocksX * d.blocksY)} blocks of {bxs}×{bys} threads, {d.warpsPerBlock} warp{d.warpsPerBlock === 1 ? '' : 's'} each:{' '}
            <b>{big(d.warps)}</b> warps. At the ghost test <b>{big(d.ghost)}</b> diverge; at the output guard <b>{big(d.guard)}</b>.
            Either way, <b>{big(d.divergent)}</b> ({pct(d.divergent / d.warps)} %). {d.idle ? <>{big(d.idle)} have no output at all.</> : null}
          </Translated>
        </p>
      )}
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
        <div>
          <div className="mx-label">
            <Translated vi="Toàn bộ lưới luồng">The whole grid of threads</Translated>
          </div>
          <svg viewBox={`-3 -3 ${OW + 6} ${OH + 6}`} role="img" aria-label={tr('Warps over the whole image', 'Các warp trên toàn bộ ảnh')} style={{ width: '100%', maxWidth: OW + 4, display: 'block' }} data-strips={strips.length}>
            <rect x={0} y={0} width={OW} height={OH} fill={CLS_FILL.uniform} />
            <rect x={-1.5} y={-1.5} width={width * sx + 3} height={height * sy + 3} fill="none" stroke="var(--ink-2)" strokeWidth="1" strokeDasharray="4 2" />
            {strips.map((s, k) => (
              <g key={k}>
                <rect x={s.c0 * sx} y={s.row0 * sy} width={Math.max(3, (s.c1 - s.c0 + 1) * sx)} height={Math.max(3, (s.row1 - s.row0 + 1) * sy)}
                  fill={CLS_FILL[s.cls]} />
                {s.cls === 'idle' ? (
                  <rect x={s.c0 * sx} y={s.row0 * sy} width={Math.max(3, (s.c1 - s.c0 + 1) * sx)} height={Math.max(3, (s.row1 - s.row0 + 1) * sy)} fill={`url(#${HATCH})`} />
                ) : null}
              </g>
            ))}
          </svg>
          <p className="muted" style={{ margin: '0.3rem 0 0', fontSize: '0.78rem' }}>
            <Translated
              vi={
                <>
                  {gw}×{gh} luồng cho một ảnh {big(width)}×{big(height)}. Ảnh được viền nét đứt; các dải mỏng hơn 3 px được
                  vẽ rộng 3 px.
                </>
              }
            >
              {gw}×{gh} threads for a {big(width)}×{big(height)} image. The image is outlined dashed; strips thinner than 3 px are drawn 3 px wide.
            </Translated>
          </p>
        </div>
        <div>
          <div className="mx-label">
            <Translated vi={<>Cận cảnh: {zc}×{zr} luồng ở góc trên bên trái, mỗi warp một ô</>}>
              Close-up: the top-left {zc}×{zr} threads, one box per warp
            </Translated>
          </div>
          <svg viewBox={`-2 -2 ${ZW + 4} ${ZH + 4}`} role="img" aria-label={tr('Warps in the top-left corner', 'Các warp ở góc trên bên trái')} style={{ width: '100%', maxWidth: ZW + 4, display: 'block' }}>
            {(locked ? [] : corner).flatMap((w, wi) =>
              w.segments
                .filter((sg) => sg.row < zr && sg.c0 < zc)
                .map((sg, si) => {
                  const x = sg.c0 * zsx;
                  const y = sg.row * zsy;
                  const ww = (Math.min(sg.c1, zc - 1) - sg.c0 + 1) * zsx;
                  return (
                    <g key={`${wi}-${si}`} data-cls={w.cls}>
                      <rect x={x + 0.4} y={y + 0.4} width={ww - 0.8} height={zsy - 0.8} fill={CLS_FILL[w.cls]} />
                      {w.cls === 'idle' ? <rect x={x + 0.4} y={y + 0.4} width={ww - 0.8} height={zsy - 0.8} fill={`url(#${HATCH})`} /> : null}
                    </g>
                  );
                }),
            )}
            {r < zc ? <line x1={r * zsx} x2={r * zsx} y1={0} y2={ZH} stroke="var(--page)" strokeWidth="1.2" strokeDasharray="3 2" /> : null}
            {r < zr ? <line x1={0} x2={ZW} y1={r * zsy} y2={r * zsy} stroke="var(--page)" strokeWidth="1.2" strokeDasharray="3 2" /> : null}
            {width < zc ? <line x1={width * zsx} x2={width * zsx} y1={0} y2={ZH} stroke="var(--ink)" strokeWidth="1.5" /> : null}
            {height < zr ? <line x1={0} x2={ZW} y1={height * zsy} y2={height * zsy} stroke="var(--ink)" strokeWidth="1.5" /> : null}
            {Array.from({ length: Math.ceil(zc / bxs) * Math.ceil(zr / bys) }, (_, k) => {
              const bx = k % Math.ceil(zc / bxs);
              const by = Math.floor(k / Math.ceil(zc / bxs));
              return <rect key={'b' + k} x={bx * bxs * zsx} y={by * bys * zsy} width={Math.min(bxs, zc - bx * bxs) * zsx} height={Math.min(bys, zr - by * bys) * zsy} fill="none" stroke="var(--ink-2)" strokeWidth="1" />;
            })}
          </svg>
          <p className="muted" style={{ margin: '0.3rem 0 0', fontSize: '0.78rem' }}>
            <Translated
              vi={
                <>
                  Nét đứt: cách cạnh trên và cạnh trái r = {r}; các đầu ra nằm ngoài những đường này có mọi tap nằm trong ảnh.
                  Viền mảnh: các khối.
                </>
              }
            >
              Dashed: r = {r} from the top and left edges; outputs beyond these lines have every tap inside the image.
              Thin outlines: blocks.
            </Translated>
          </p>
        </div>
      </div>
      <p className="muted" style={{ margin: '0.8rem 0 0' }}>{choice.note}</p>
    </Bench>
  );
}

// ---------------------------------------------------------------- the section
export function S72() {
  return (
    <>
      <Translated
        vi={
          <p>
            Mỗi đầu ra của một phép tích chập có thể được tính mà không cần biết bất kỳ đầu ra nào khác, nên kernel tự nhiên
            nhất giao cho mỗi đầu ra một luồng (thread) riêng. Trong 2D, đó là một lưới (grid) 2D gồm các khối (block) 2D,
            cách sắp xếp của kernel chuyển ảnh xám ở chương 3 (<a href={href('ch03', '3.2')}>§3.2</a>). Hình của sách là
            một ảnh 16×16 được phủ bởi 4×4 khối, mỗi khối 4×4 luồng; khối thực tế lớn hơn, thường là 32×32.
          </p>
        }
      >
        <p>
          Every output of a convolution can be computed without knowing any other output, so the natural kernel gives each
          output its own thread. In 2D that is a 2D grid of 2D blocks, the arrangement of chapter 3&rsquo;s grayscale
          kernel (<a href={href('ch03', '3.2')}>§3.2</a>). The book&rsquo;s picture is a 16×16 image covered by 4×4 blocks
          of 4×4 threads; real blocks are bigger, typically 32×32.
        </p>
      </Translated>
      <Source section="7.2" pages="164–165" figs={tr('Fig. 7.6', 'Hình 7.6')} />
      <Translated
        vi={
          <p>
            Kernel nhận đầu vào <code>N</code>, bộ lọc <code>F</code>, đầu ra <code>P</code>, bán kính <code>r</code>, cùng{' '}
            <code>width</code> và <code>height</code> của ảnh. Mã của sách là 2D và giả định bộ lọc vuông; 1D và 3D là bài
            tập.
          </p>
        }
      >
        <p>
          The kernel takes the input <code>N</code>, the filter <code>F</code>, the output <code>P</code>, the radius{' '}
          <code>r</code>, and the image&rsquo;s <code>width</code> and <code>height</code>. The book&rsquo;s code is 2D and
          assumes a square filter; 1D and 3D are exercises.
        </p>
      </Translated>

      <h2>
        <Translated vi="Đầu ra nào, mảng con nào">Which output, which patch</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mỗi luồng tìm đầu ra của nó đúng như ở chương 3: <code>outCol = blockIdx.x*blockDim.x + threadIdx.x</code>{' '}
            và tương tự theo y cho <code>outRow</code>. Mảng con của nó trải từ hàng <code>outRow - r</code> đến{' '}
            <code>outRow + r</code> và các cột tương ứng, nên <code>(outRow - r, outCol - r)</code> là góc trên bên trái của
            mảng con. Trong ví dụ của sách, luồng (1, 1) của khối (1, 1) tính P[5][5] từ các hàng và cột 3 đến 7.
          </p>
        }
      >
        <p>
          Each thread finds its output exactly as in chapter 3: <code>outCol = blockIdx.x*blockDim.x + threadIdx.x</code>{' '}
          and the same in y for <code>outRow</code>. Its patch spans rows <code>outRow - r</code> to{' '}
          <code>outRow + r</code> and the same columns, so <code>(outRow - r, outCol - r)</code> is the patch&rsquo;s
          upper-left corner. In the book&rsquo;s example, thread (1, 1) of block (1, 1) computes P[5][5] from rows and
          columns 3 to 7.
        </p>
      </Translated>
      <Source section="7.2" pages="166" />
      <Ask q={questionsS72['ch07/7.2/q-map']} label={tr('Work it out', 'Tự tính')} />

      <h2>
        <Translated vi="Kernel, như được in và như khi biên dịch">The kernel, as printed and as it compiles</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hình 7.7 của sách là hai vòng lặp lồng nhau trên bộ lọc bên trong mỗi luồng, với một lệnh <code>if</code> bỏ qua
            các tap rơi ra ngoài ảnh. Như được in, nó không biên dịch được. Trước khi xem bản đã sửa, hãy tìm ra lý do: một
            số dòng dưới đây sai như đang viết, và một số thì đúng.
          </p>
        }
      >
        <p>
          The book&rsquo;s Fig. 7.7 is two nested loops over the filter inside each thread, with an <code>if</code> that
          skips taps falling outside the image. As printed it does not compile. Before you see the corrected version,
          find out why: some of these lines are wrong as they stand, and some are fine.
        </p>
      </Translated>
      <Source section="7.2" pages="166" figs={tr('Fig. 7.7', 'Hình 7.7')} />
      <HuntAndKernel />
      <Ask q={questionsS72['ch07/7.2/q-guard']} label={tr('Without the guard', 'Khi không có lệnh chặn')} />

      <h2>
        <Translated vi="Xem một luồng làm việc">Watch one thread work</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trình chạy từng bước chạy một luồng của kernel đó trên ảnh mẫu của sách. Bước đầu tiên là phép tính chỉ số; mỗi
            bước sau là một tap: các dòng tính <code>inRow</code> và <code>inCol</code>, phép kiểm tra ô ma, và, khi phép
            kiểm tra vượt qua, phép nhân-cộng. Bước cuối cùng là lệnh ghi duy nhất.
          </p>
        }
      >
        <p>
          The stepper runs one thread of that kernel on the book&rsquo;s toy image. The first step is the index arithmetic;
          each later step is one tap: lines that compute <code>inRow</code> and <code>inCol</code>, the ghost test, and,
          when the test passes, the multiply-add. The last step is the single store.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một luồng, mỗi lần một câu lệnh, không có gì về thời gian hay lưu lượng bộ nhớ. Các dòng được tô sáng được tìm
              trong tệp của lab 7.1 theo nội dung văn bản, nên chúng đi theo tệp nếu tệp thay đổi.
            </p>
          }
        >
          <p>
            One thread, one statement at a time, nothing about time or memory traffic. The highlighted lines are found in
            lab 7.1&rsquo;s file by their text, so they follow the file if it changes.
          </p>
        </Translated>
      </Simplification>
      <Stepper />
      <Ask q={questionsS72['ch07/7.2/q-skip']} label={tr('Count the taps', 'Đếm các tap')} />
      <Translated
        vi={
          <p>
            Hai chi tiết của kernel có ý nghĩa vượt ra ngoài chương này. <code>Pvalue</code> nằm trong một thanh ghi
            (register) suốt vòng lặp và được ghi vào <code>P</code> một lần, nên việc cộng dồn không tốn lưu lượng DRAM. Và
            bỏ qua một tap ô ma cho đúng tổng mà phép nhân với một ô ma bằng không sẽ cho: lệnh <code>if</code> chính là
            chính sách số không, và không có gì được đọc từ ngoài ảnh.
          </p>
        }
      >
        <p>
          Two details of the kernel matter beyond this chapter. <code>Pvalue</code> lives in a register for the whole loop
          and is written to <code>P</code> once, so the accumulation costs no DRAM traffic. And skipping a ghost tap gives
          exactly the sum that multiplying by a zero ghost cell would: the <code>if</code> is the zero policy, and nothing
          is read from outside the image.
        </p>
      </Translated>
      <Source section="7.2" pages="166" />

      <h2>
        <Translated vi="Phân kỳ ở biên">Divergence at the edges</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách đưa ra hai nhận xét về kernel này. Thứ nhất là phân kỳ điều khiển: các luồng gần bốn cạnh bỏ qua số tap
            khác nhau, nên các lane của một warp có thể bất đồng tại phép kiểm tra ô ma. Cái giá của nó phụ thuộc vào kích
            thước ảnh và bán kính; với ảnh lớn và bộ lọc nhỏ, chỉ một phần nhỏ đầu ra bị ảnh hưởng, nên sách cho rằng hiệu
            ứng này &ldquo;từ khiêm tốn đến không đáng kể&rdquo;.
          </p>
        }
      >
        <p>
          The book makes two observations about this kernel. The first is control divergence: threads near the four edges
          skip different numbers of taps, so the lanes of a warp can disagree at the ghost test. Its cost depends on the
          image size and the radius; for large images and small filters only a small share of outputs is affected, so the
          book expects the effect to be &ldquo;modest to insignificant&rdquo;.
        </p>
      </Translated>
      <NoteGroup>
        <Source section="7.2" pages="166" />
        <Simplification>
          <Translated
            vi={
              <p>
                Hoạt ảnh cho mọi lane đi qua các tap cùng nhau, mỗi lần một tap. Nó cho thấy lane nào ngồi ngoài một tap,
                không phải điều đó tốn bao nhiêu: không có thời gian, predication hay hội tụ lại. Con số cuối cùng của nó dùng
                cách CUDA cắt thành warp 32 luồng.
              </p>
            }
          >
            <p>
              The animation steps every lane through the taps together, one tap at a time. It shows which lanes sit out a
              tap, not what that costs: no timing, predication or reconvergence. Its last count uses CUDA&rsquo;s cut into
              warps of 32.
            </p>
          </Translated>
        </Simplification>
      </NoteGroup>
      <OneThreadScene />
      <Translated
        vi={
          <p>
            Một điều sách không nói là warp nào phân kỳ phụ thuộc vào hình dạng khối, vì một warp là một dãy chỉ số luồng
            liên tiếp theo thứ tự x chạy nhanh nhất (chương 4, <a href={href('ch04', '4.4')}>§4.4</a>). Với khối rộng 32,
            một warp là một hàng của khối, và ở cạnh trên và cạnh dưới, mọi lane của một hàng bỏ qua cùng các hàng bộ lọc.
            Khối hẹp hơn đặt hai hàng trở lên vào một warp, và khi đó cạnh trên và cạnh dưới cũng tách warp.
          </p>
        }
      >
        <p>
          One thing the book does not say is that which warps diverge depends on the block shape, because a warp is a run
          of consecutive thread indices in x-fastest order (chapter 4, <a href={href('ch04', '4.4')}>§4.4</a>). With 32-wide
          blocks a warp is one row of a block, and at the top and bottom edges every lane of a row skips the same filter
          rows. Narrower blocks put two or more rows in a warp, and then the top and bottom edges split warps too.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một warp được tính là phân kỳ khi các lane của nó bất đồng tại một nhánh ít nhất một lần. Mô hình không nói gì
              về cái giá của điều đó: không có thời gian, không có predication, không có hội tụ lại. Nó dùng cách cắt warp
              mà CUDA ghi trong tài liệu cho mọi độ rộng; AMD ghi tài liệu về thứ tự nhưng không nói gọn cách cắt trong một
              câu, còn Metal gọi cách nhóm của nó là do cài đặt quyết định (notes/today/ch04-facts.md).
            </p>
          }
        >
          <p>
            A warp counts as divergent when its lanes disagree at a branch at least once. The model says nothing about
            what that costs: no timing, no predication, no reconvergence. It uses CUDA&rsquo;s documented cut into warps
            for every width; AMD documents the order but not the cut in one sentence, and Metal calls its grouping
            implementation-defined (notes/today/ch04-facts.md).
          </p>
        </Translated>
      </Simplification>
      <WarpsExplorer />
      <Translated
        vi={
          <p>
            Hãy tìm ba điều. Tỉ lệ warp phân kỳ giảm khi ảnh lớn lên. Với khối 16×16 hoặc 8×8, nó tăng theo r, vì có thêm
            nhiều hàng gần cạnh trên và cạnh dưới có kiểu bỏ tap riêng. Với khối 32×32 và warp rộng 32, chỉ cạnh trái và
            cạnh phải phân kỳ, nên con số không đổi theo r chừng nào các dải biên rộng r còn nằm trong cột khối đầu tiên và
            cuối cùng: mọi r đến 7 với ảnh mặc định rộng 256. Hãy thử chiều rộng 100 với r = 4 rồi r = 5 để thấy điều đó
            hỏng, và tìm xem cột khối nào tham gia. Và một warp rộng 64, như trên GPU AMD wave64, trải qua hai hàng của một
            khối 32×32, khiến cạnh trên và cạnh dưới phân kỳ trở lại.
          </p>
        }
      >
        <p>
          Find three things. The share of divergent warps falls as the image grows. With 16×16 or 8×8 blocks it grows
          with r, because more rows near the top and bottom have their own pattern of skipped taps. With 32×32 blocks and
          32-wide warps only the left and right edges diverge, so the count does not change with r as long as the
          borders of width r stay inside the first and last column of blocks: every r up to 7 for the default 256-wide
          image. Try width 100 with r = 4 and then r = 5 to see it fail, and work out which block column joins in. And a
          64-wide warp, as on wave64 AMD GPUs, spans two rows of a 32×32 block, which brings the top and bottom edges back.
        </p>
      </Translated>
      <Ask q={questionsS72['ch07/7.2/q-warps1d']} label={tr('One dimension', 'Một chiều')} />
      <Ask q={questionsS72['ch07/7.2/q-blockshape']} label={tr('Block shape', 'Hình dạng khối')} />
      <Translated
        vi={
          <p>
            Nhận xét thứ hai của sách mới là điều quan trọng: băng thông bộ nhớ. Mỗi tap nạp một phần tử của <code>N</code>{' '}
            và một của <code>F</code> từ bộ nhớ toàn cục (global memory) cho hai phép toán dấu phẩy động. Mục{' '}
            <a href={href('ch07', '7.3')}>7.3</a> tính xem điều đó tốn bao nhiêu.
          </p>
        }
      >
        <p>
          The book&rsquo;s second observation is the one that matters: memory bandwidth. Every tap loads one element of{' '}
          <code>N</code> and one of <code>F</code> from global memory for two floating-point operations. Section{' '}
          <a href={href('ch07', '7.3')}>7.3</a> counts what that costs.
        </p>
      </Translated>
      <Source section="7.2" pages="166" />

      <h2>
        <Translated vi="Trong một backend thực tế">In a real backend</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Backend CUDA của llama.cpp có một phép tích chập 2D trực tiếp đúng dạng này, mỗi đầu ra một luồng, với hai điểm
            khác đáng học theo. Nó tính khoảng tap hợp lệ của mỗi luồng một lần, trước các vòng lặp, thay vì kiểm tra từng
            tap, và nó nhận các kích thước dưới dạng một struct truyền theo giá trị.
          </p>
        }
      >
        <p>
          llama.cpp&rsquo;s CUDA backend has a direct 2D convolution of exactly this shape, one thread per output, with
          two differences worth copying. It works out each thread&rsquo;s range of valid taps once, before the loops,
          instead of testing every tap, and it takes its sizes as a struct passed by value.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              <code>conv2d_kernel</code> trong <code>ggml/src/ggml-cuda/conv2d.cu</code> gọi <code>calculate_kernel_bounds</code>{' '}
              và nhận <code>const conv_params P</code> theo giá trị, còn các tensor là con trỏ <code>__restrict__</code>.
              Kernel đệm của nó ghi ô ma bằng không với <code>(unsigned) x &lt; (unsigned) iw</code>: một chỉ số âm trở thành
              một số không dấu rất lớn, nên một phép so sánh mỗi trục thay cho hai (facts D2).
            </p>
          }
        >
          <p>
            <code>conv2d_kernel</code> in <code>ggml/src/ggml-cuda/conv2d.cu</code> calls <code>calculate_kernel_bounds</code>{' '}
            and takes <code>const conv_params P</code> by value and its tensors as <code>__restrict__</code> pointers. Its
            padding kernel writes zero ghost cells with <code>(unsigned) x &lt; (unsigned) iw</code>: a negative index
            becomes a huge unsigned one, so one compare per axis replaces two (facts D2).
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Cắt cận của vòng lặp cho cùng tổng mà không có nhánh cho mỗi tap. Các lane của một warp ở biên vẫn chạy số lần
            lặp khác nhau, nên sự phân kỳ chuyển từ lệnh <code>if</code> sang điều kiện vòng lặp chứ không biến mất.
          </p>
        }
      >
        <p>
          Clipping the loop bounds gives the same sum with no per-tap branch. Lanes of an edge warp still run different
          numbers of iterations, so the divergence moves from the <code>if</code> to the loop condition rather than
          disappearing.
        </p>
      </Translated>

      <h2>
        <Translated vi="HIP và Metal">HIP and Metal</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bản build HIP được thiết lập để nhận cùng tệp <code>.cu</code> qua <code>pmpp/vendors/hip.h</code>, vì luồng,
            khối và các biến chỉ số dựng sẵn giữ nguyên cách viết của CUDA; nó chưa được biên dịch hay chạy trên phần cứng
            AMD (unverified). Kernel Metal (tab thứ hai ở trên) nhận vị trí của nó dưới dạng một <code>uint2</code> với{' '}
            <code>[[thread_position_in_grid]]</code>, và Metal chỉ chấp nhận kiểu không dấu cho thuộc tính đó. Vì vậy kernel
            chuyển sang <code>int</code> trước khi trừ <code>r</code>, cùng lý do kernel CUDA lưu <code>outCol</code> trong
            một <code>int</code>: một <code>outCol - r</code> không dấu sẽ quay vòng thay vì thành số âm, và phép{' '}
            <code>&gt;= 0</code> của phép kiểm tra ô ma sẽ luôn đúng.
          </p>
        }
      >
        <p>
          The HIP build is set up to take the same <code>.cu</code> file through <code>pmpp/vendors/hip.h</code>, because
          threads, blocks and the index built-ins keep their CUDA spellings; it has not been compiled or run on AMD
          hardware (unverified). The Metal kernel (the second tab above) gets its position as
          a <code>uint2</code> with <code>[[thread_position_in_grid]]</code>, and Metal accepts only unsigned types for
          that attribute. So the kernel converts to <code>int</code> before subtracting <code>r</code>, for the same
          reason the CUDA kernel stores <code>outCol</code> in an <code>int</code>: an unsigned <code>outCol - r</code>{' '}
          wraps around instead of going negative, and the ghost test&rsquo;s <code>&gt;= 0</code> would always pass.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 (notes/today/ch03-facts.md C6; ch07-facts.md A8; ch05-facts.md B5, C3)',
          '2026-11-20 (notes/today/ch03-facts.md C6; ch07-facts.md A8; ch05-facts.md B5 và C3)',
        )}
      >
        <Translated
          vi={
            <p>
              Thuộc tính vị trí của MSL chỉ chấp nhận kiểu <code>ushort</code> và <code>uint</code>. Warp của AMD rộng 64
              trên gfx9 và 32 trên gfx10 trở về sau; hãy đọc <code>warpSize</code>. Độ rộng của Metal là{' '}
              <code>threadExecutionWidth</code> của pipeline.
            </p>
          }
        >
          <p>
            MSL position attributes accept only <code>ushort</code> and <code>uint</code> types. AMD warps are 64 wide on
            gfx9 and 32 on gfx10 and later; read <code>warpSize</code>. Metal&rsquo;s width is the pipeline&rsquo;s{' '}
            <code>threadExecutionWidth</code>.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Không có gì trong kernel này phụ thuộc vào độ rộng warp, nhưng hình ảnh phân kỳ thì có: hãy đọc{' '}
            <code>warpSize</code> trên CUDA và HIP và <code>threadExecutionWidth</code> của pipeline trên Metal thay vì giả
            định 32. Lab 7.1 trên <a href={href('ch07', 'labs')}>trang lab</a> chạy kernel này so với hai cách giữ bộ lọc mà
            mục <a href={href('ch07', '7.4')}>7.4</a> giới thiệu.
          </p>
        }
      >
        <p>
          Nothing in this kernel depends on the warp width, but the divergence picture does: read <code>warpSize</code>{' '}
          on CUDA and HIP and the pipeline&rsquo;s <code>threadExecutionWidth</code> on Metal rather than assuming 32.
          Lab 7.1 on the <a href={href('ch07', 'labs')}>labs page</a> runs this kernel against two ways of holding the
          filter that section <a href={href('ch07', '7.4')}>7.4</a> introduces.
        </p>
      </Translated>
    </>
  );
}
