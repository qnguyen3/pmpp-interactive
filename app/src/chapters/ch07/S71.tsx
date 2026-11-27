import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  ASYMMETRIC,
  BOOK_1D,
  BOOK_2D,
  EX2,
  EX3_FILTERS,
  POLICIES,
  POLICY_LABEL,
  conv1d,
  conv2d,
  ghostCounts2d,
  ghostFormula2d,
  patch2d,
  radiusOf,
  skippedMap,
  window1d,
  type Policy,
} from '../../sim/convolve';

// Section 7.1: the definition, the boundary policies, the flip that the book (and
// PyTorch) does not do, the 2D form, and counting ghost cells (Exercises 1 to 6).
// Every number shown comes from sim/convolve.ts.

const fmtNum = (v: number) => (Number.isInteger(v) ? String(v) : String(Math.round(v * 100) / 100));

export const questionsS71: Record<string, Question> = {
  'ch07/7.1/q-ex2': {
    kind: 'choice',
    id: 'ch07/7.1/q-ex2',
    prompt: (
      <Translated vi={<>Bài tập 2. Tích chập <code>N = {'{4, 1, 3, 2, 3}'}</code> với <code>F = {'{2, 1, 4}'}</code>, dùng công thức của sách và ô ma (ghost cell) bằng 0. Đầu ra là gì?</>}>
<>
        Exercise 2. Convolve <code>N = {'{4, 1, 3, 2, 3}'}</code> with <code>F = {'{2, 1, 4}'}</code>, using the
        book&rsquo;s formula and zero ghost cells. What is the output?
      </>
</Translated>
    ),
    options: [
      { label: '{8, 21, 13, 20, 7}', correct: true },
      {
        label: '{6, 23, 11, 20, 11}',
        why: <Translated vi="Đó là bộ lọc bị đảo ngược: tích chập kiểu giáo trình. Công thức của sách, giống PyTorch, nhân f[0] với phần tử ngoài cùng bên trái của cửa sổ.">{'That is the filter reversed: textbook convolution. The book\'s formula, like PyTorch\'s, multiplies f[0] with the leftmost element of the window.'}</Translated>,
      },
      {
        label: '{21, 13, 20}',
        why: <Translated vi="Đó là kết quả khi hoàn toàn không có ô ma, như PyTorch làm với padding=0. Sách cho mỗi đầu vào một đầu ra.">{'That is what you get with no ghost cells at all, as PyTorch does with padding=0. The book gives every input an output.'}</Translated>,
      },
      {
        label: '{16, 21, 13, 20, 19}',
        why: <Translated vi="Đó là nhân bản giá trị ở biên vào các ô ma. Bài tập dùng mặc định của sách, là số 0.">{'That replicates the edge value into the ghost cells. The exercise uses the book\'s default, zeros.'}</Translated>,
      },
    ],
    explain: (
      <Translated vi={<p>Với r = 1, y[i] = 2·x[i−1] + 1·x[i] + 4·x[i+1]. Vậy y[0] = 2·0 + 4 + 4·1 = 8, y[1] = 2·4 + 1 + 4·3 = 21, y[2] = 2·1 + 3 + 4·2 = 13, y[3] = 2·3 + 2 + 4·3 = 20, và y[4] = 2·2 + 3 + 4·0 = 7. Hãy nạp Bài tập 2 vào trình khám phá cửa sổ trượt của mục 7.1 và bật chế độ lật để thấy phương án thứ hai từ đâu ra.</p>}>
<p>
        With r = 1, y[i] = 2·x[i−1] + 1·x[i] + 4·x[i+1]. So y[0] = 2·0 + 4 + 4·1 = 8, y[1] = 2·4 + 1 + 4·3 = 21, y[2] =
        2·1 + 3 + 4·2 = 13, y[3] = 2·3 + 2 + 4·3 = 20, and y[4] = 2·2 + 3 + 4·0 = 7. Load Exercise 2 into section 7.1&rsquo;s
        sliding-window explorer and switch the flip on to see where the second option comes from.
      </p>
</Translated>
    ),
    source: 'ch. 7 Exercise 2, p. 180; notes/today/ch07-facts.md D1',
    vi: { source: 'ch. 7 Bài tập 2, p. 180; notes/today/ch07-facts.md D1' },
  },
  'ch07/7.1/q-ex3': {
    kind: 'choice',
    id: 'ch07/7.1/q-ex3',
    prompt: (
      <Translated vi={<>Bài tập 3, một phần của nó. Bộ lọc nào trong số này dịch mọi giá trị sang phải một vị trí, để{' '}<code>y[i] = x[i-1]</code>?</>}>
<>
        Exercise 3, one part of it. Which of these filters moves every value one place to the right, so that{' '}
        <code>y[i] = x[i-1]</code>?
      </>
</Translated>
    ),
    options: [
      { label: '[0, 1, 0]', why: <Translated vi="Phép đồng nhất: y[i] = x[i].">{'The identity: y[i] = x[i].'}</Translated> },
      { label: '[0, 0, 1]', why: <Translated vi="f[2] nhân với x[i+1], nên y[i] = x[i+1]: mọi giá trị dịch sang trái một vị trí.">{'f[2] multiplies x[i+1], so y[i] = x[i+1]: every value moves one place to the left.'}</Translated> },
      { label: '[1, 0, 0]', correct: true },
      { label: '[−1/2, 0, 1/2]', why: <Translated vi="Bộ lọc này tính (x[i+1] − x[i−1]) / 2.">{'This one computes (x[i+1] − x[i−1]) / 2.'}</Translated> },
      { label: '[1/3, 1/3, 1/3]', why: <Translated vi="Trung bình của ba phần tử lân cận.">{'An average of three neighbours.'}</Translated> },
    ],
    explain: (
      <>
        <Translated vi={<p>Trong công thức của sách, f[0] nhân với phần tử ngoài cùng bên trái của cửa sổ, x[i−1], nên [1, 0, 0] chép x[i−1] vào y[i]. Theo định nghĩa kiểu giáo trình có lật bộ lọc, [1, 0, 0] và [0, 0, 1] đổi nghĩa cho nhau, và đó là lý do đáp án của bài tập này phụ thuộc vào định nghĩa bạn dùng.</p>}>
<p>
          In the book&rsquo;s formula f[0] multiplies the leftmost element of the window, x[i−1], so [1, 0, 0] copies
          x[i−1] into y[i]. Under the flipped, textbook definition, [1, 0, 0] and [0, 0, 1] swap meanings, which is why
          the answer to this exercise depends on which definition you use.
        </p>
</Translated>
        <Translated vi={<p>Hai bộ lọc còn lại đáng được gọi tên. [−1/2, 0, 1/2] là sai phân trung tâm, đạo hàm bậc nhất rời rạc, và là điểm khởi đầu của các stencil ở chương 8. [1/3, 1/3, 1/3] là trung bình trượt ba điểm, một bộ lọc làm mượt.</p>}>
<p>
          The other two are worth naming. [−1/2, 0, 1/2] is a central difference, the discrete first derivative, which
          is where chapter 8&rsquo;s stencils start. [1/3, 1/3, 1/3] is a three-point moving average, a smoothing
          filter.
        </p>
</Translated>
      </>
    ),
    source: 'ch. 7 Exercise 3, p. 180',
    vi: { source: 'ch. 7 Bài tập 3, p. 180' },
  },
  'ch07/7.1/q-policy': {
    kind: 'choice',
    id: 'ch07/7.1/q-policy',
    prompt: (
      <Translated vi={<>Một mảng 7 đầu vào và một bộ lọc 5 tap. Chuyển từ ô ma bằng 0 sang nhân bản giá trị biên có thể làm thay đổi những đầu ra nào?</>}>
        <>An array of 7 inputs and a 5-tap filter. Switching from zero ghost cells to replicating the edge can change which outputs?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Chỉ y[0] và y[6].">{'Only y[0] and y[6].'}</Translated>, why: <Translated vi="Với r = 2, cửa sổ của y[1] cũng vươn quá điểm đầu một vị trí.">{'With r = 2 the window of y[1] also reaches one place past the start.'}</Translated> },
      { label: <Translated vi="y[0], y[1], y[5] và y[6].">{'y[0], y[1], y[5] and y[6].'}</Translated>, correct: true },
      { label: <Translated vi="Cả bảy.">{'All seven.'}</Translated>, why: <Translated vi="Cửa sổ của y[2], y[3] và y[4] nằm trọn trong mảng; chúng không bao giờ đọc ô ma.">{'The windows of y[2], y[3] and y[4] lie entirely inside the array; they never read a ghost cell.'}</Translated> },
      { label: <Translated vi="Không đầu ra nào: chính sách chỉ ảnh hưởng tới bộ lọc, không ảnh hưởng tới x.">{'None: the policy only matters for the filter, not for x.'}</Translated>, why: <Translated vi="Ô ma đứng thay cho các phần tử bị thiếu của x.">{'Ghost cells stand in for missing elements of x.'}</Translated> },
    ],
    explain: (
      <Translated vi={<p>Một cửa sổ vươn ra r = 2 vị trí mỗi bên. Chỉ những đầu ra cách một đầu mảng không quá r mới có cửa sổ thò ra ngoài, nên chỉ r đầu ra đầu tiên và r đầu ra cuối cùng phụ thuộc vào chính sách. Trong một ảnh lớn đó là một đường viền mỏng, nên chính sách là câu hỏi về tính đúng đắn nhiều hơn hẳn là về hiệu năng.</p>}>
<p>
        A window reaches r = 2 places to each side. Only outputs within r of an end have a window that sticks out, so
        only the first r and the last r outputs depend on the policy. In a large image that is a thin border, which is
        why the policy is a correctness question far more than a performance one.
      </p>
</Translated>
    ),
    source: 'ch. 7 §7.1, pp. 161–162',
  },
  'ch07/7.1/q-ex5': {
    kind: 'numeric',
    id: 'ch07/7.1/q-ex5',
    prompt: (
      <Translated vi={<>Bài tập 5 cho một ảnh 8×8 và một bộ lọc 3×3. Nếu các tap rơi vào ô ma được bỏ qua thay vì nhân với 0, còn lại bao nhiêu phép nhân?</>}>
<>
        Exercise 5 for an 8×8 image and a 3×3 filter. If ghost taps are skipped instead of multiplied by zero, how many
        multiplications are left?
      </>
</Translated>
    ),
    answer: 484,
    unit: 'multiplications',
    vi: { unit: 'phép nhân', source: 'ch. 7 Bài tập 5, p. 180' },
    explain: (
      <>
        <Translated vi={<p>Đếm một chiều trước. Dọc theo một hàng 8 đầu ra với r = 1, đầu ra đầu tiên và cuối cùng mỗi cái mất một tap, nên có 8·3 − 2 = 22 tap thật theo chiều hàng. Một tap là thật khi nó thật theo cả hai chiều, nên tổng là 22² = 484.</p>}>
<p>
          Count one dimension first. Along a row of 8 outputs with r = 1, the first and the last output each lose one
          tap, so there are 8·3 − 2 = 22 real taps per row direction. A tap is real when it is real in both directions,
          so the total is 22² = 484.
        </p>
</Translated>
        <Translated vi={<p>Tính cả tap rơi vào ô ma thì có 64·9 = 576, và bản thân các ô ma tạo thành một khung rộng một ô quanh ảnh: 10² − 8² = 36. Bộ đếm ô ma trong mục 7.1 kiểm tra cả ba con số.</p>}>
<p>
          With ghost taps there are 64·9 = 576, and the ghost cells themselves form a frame one wide around the image:
          10² − 8² = 36. The ghost counter in section 7.1 checks all three.
        </p>
</Translated>
      </>
    ),
    source: 'ch. 7 Exercise 5, p. 180',
  },
  'ch07/7.1/q-pytorch': {
    kind: 'choice',
    id: 'ch07/7.1/q-pytorch',
    prompt: (
      <Translated vi={<><code>torch.nn.functional.conv2d</code> trên một đầu vào 1×1×7×7 với trọng số 5×5 và <code>padding=0</code>. Theo cách gọi của sách, nó tính gì?</>}>
<>
        <code>torch.nn.functional.conv2d</code> on a 1×1×7×7 input with a 5×5 weight and <code>padding=0</code>. In the
        book&rsquo;s terms, what does it compute?
      </>
</Translated>
    ),
    options: [
      {
        label: <Translated vi="Chỉ 3×3 đầu ra có vùng không chứa ô ma nào.">{'Only the 3×3 outputs whose patch contains no ghost cell.'}</Translated>,
        correct: true,
      },
      { label: <Translated vi="Cả 7×7 đầu ra, với ô ma bằng 0.">{'All 7×7 outputs, with zero ghost cells.'}</Translated>, why: <Translated vi='Điều đó cần padding=2 (hoặc padding="same").'>{'That needs padding=2 (or padding="same").'}</Translated> },
      { label: <Translated vi="Cả 7×7 đầu ra, với giá trị biên được nhân bản.">{'All 7×7 outputs, with the edge values replicated.'}</Translated>, why: <Translated vi="Nhân bản là một padding_mode của nn.Conv2d, và nó chỉ có tác dụng khi có padding.">{'Replication is a padding_mode of nn.Conv2d, and it only matters when there is padding.'}</Translated> },
      { label: <Translated vi="Giống sách, nhưng bộ lọc bị lật.">{'The same as the book, but with the filter flipped.'}</Translated>, why: <Translated vi="PyTorch ghi trong tài liệu rằng Conv2d là tương quan chéo (cross-correlation): không lật, giống sách.">{'PyTorch documents Conv2d as cross-correlation: no flip, like the book.'}</Translated> },
    ],
    explain: (
      <Translated vi={<p>Không có padding nghĩa là không có ô ma, nên chỉ những đầu ra có vùng 5×5 nằm trọn trong ảnh mới tồn tại: các hàng và cột 2 đến 4, một kết quả 3×3. <code>padding=2</code> cho kết quả 7×7 của sách với ô ma bằng 0; một <code>nn.Conv2d</code> với <code>padding_mode=&apos;replicate&apos;</code> hoặc{' '}<code>&apos;circular&apos;</code> cho hai chính sách còn lại của sách.</p>}>
<p>
        No padding means no ghost cells, so only outputs whose 5×5 patch lies entirely inside the image exist: rows and
        columns 2 to 4, a 3×3 result. <code>padding=2</code> gives the book&rsquo;s 7×7 result with zero ghost cells;
        an <code>nn.Conv2d</code> with <code>padding_mode=&apos;replicate&apos;</code> or{' '}
        <code>&apos;circular&apos;</code> gives the book&rsquo;s other two policies.
      </p>
</Translated>
    ),
    source: 'notes/today/ch07-facts.md D1', // i18n-ignore: citation
  },
};

// ---------------------------------------------------------------- explorer: the sliding window
interface Preset {
  key: string;
  label: string;
  x: readonly number[];
  f: readonly number[];
}
const PRESETS: Preset[] = [
  { key: 'book', get label() { return tr("the book's x and f (Figs. 7.1 to 7.3)", 'x và f của sách (Figs. 7.1 đến 7.3)'); }, x: BOOK_1D.x, f: BOOK_1D.f },
  { key: 'ex2', get label() { return tr('Exercise 2', 'Bài tập 2'); }, x: EX2.x, f: EX2.f },
  ...EX3_FILTERS.map((e) => ({ key: 'ex3' + e.key, get label() { return tr(`Exercise 3 (${e.key}): ${e.printed}`, `Bài tập 3 (${e.key}): ${e.printed}`); }, x: BOOK_1D.x, f: e.f })),
  { key: 'asym', get label() { return tr('an asymmetric filter, [3, 2, 1] (ours)', 'một bộ lọc bất đối xứng, [3, 2, 1] (của khóa học này)'); }, x: BOOK_1D.x, f: ASYMMETRIC },
];

/** Parse "8, 2, 5" or "1/3 1/3 1/3"; null when anything is not a number. */
function parseList(s: string): number[] | null {
  const parts = s.split(/[\s,]+/).filter(Boolean);
  if (!parts.length) return null;
  const out: number[] = [];
  for (const p of parts) {
    const m = /^(-?\d+(?:\.\d+)?)(?:\/(\d+(?:\.\d+)?))?$/.exec(p.replace('−', '-'));
    if (!m) return null;
    const v = m[2] ? Number(m[1]) / Number(m[2]) : Number(m[1]);
    if (!isFinite(v)) return null;
    out.push(v);
  }
  return out;
}

function SlidingWindow() {
  const { progress } = useProgress();
  const [preset, setPreset] = useState('book');
  const [xText, setXText] = useState(BOOK_1D.x.join(', '));
  const [fText, setFText] = useState(BOOK_1D.f.join(', '));
  const [i, setI] = useState(2);
  const [policy, setPolicy] = useState<Policy>('zero');
  const [flip, setFlip] = useState(false);

  const px = parseList(xText);
  const pf = parseList(fText);
  const xErr = !px ? tr('x must be a list of numbers.', 'x phải là một danh sách số.') : px.length > 16 ? tr('x can have at most 16 elements here.', 'Ở đây x có tối đa 16 phần tử.') : null;
  const fErr = !pf ? tr('f must be a list of numbers.', 'f phải là một danh sách số.') : pf.length % 2 === 0 ? tr('f needs an odd number of weights (2r + 1).', 'f cần một số lẻ trọng số (2r + 1).') : pf.length > 9 ? tr('f can have at most 9 weights here.', 'Ở đây f có tối đa 9 trọng số.') : null;
  const p = PRESETS.find((q) => q.key === preset) ?? PRESETS[0];
  const x = !xErr && px ? px : [...p.x];
  const f = !fErr && pf ? pf : [...p.f];
  const n = x.length;
  const r = radiusOf(f.length);
  const ii = Math.min(i, n - 1);

  const w = window1d(x, f, ii, { policy, flip });
  const y = conv1d(x, f, { policy, flip });
  const yZero = conv1d(x, f, { policy: 'zero', flip });
  const yUnflipped = conv1d(x, f, { policy, flip: false });
  const touches = x.map((_, k) => window1d(x, f, k).touchesGhost);
  const byPolicy = y.map((v, k) => (Math.abs(v - yZero[k]) > 1e-9 ? k : -1)).filter((k) => k >= 0);
  const byFlip = y.map((v, k) => (Math.abs(v - yUnflipped[k]) > 1e-9 ? k : -1)).filter((k) => k >= 0);

  // y[0] of the book's data is Exercise 1: hidden until a prediction is committed.
  const edited = xText.trim() !== p.x.join(', ') || fText.trim() !== p.f.join(', ');
  const locked = preset === 'book' && !edited && policy === 'zero' && !flip && !progress.answers['ch07/7.1/p-y0'];
  // Exercise 2's data, however it got here (preset or typed): every output hidden until the question is answered.
  const same = (a: readonly number[], b: readonly number[]) => a.length === b.length && a.every((v, k) => Math.abs(v - b[k]) < 1e-9);
  const ex2Locked = same(x, EX2.x) && same(f, EX2.f) && !progress.answers['ch07/7.1/q-ex2'];
  const hide = (k: number) => ex2Locked || (locked && k === 0);

  const load = (key: string) => {
    const q = PRESETS.find((z) => z.key === key)!;
    setPreset(key);
    setXText(q.x.map(fmtNum).join(', '));
    setFText(q.f.map(fmtNum).join(', '));
    setI(Math.min(i, q.x.length - 1));
  };

  // Drawing: one column per logical index -r .. n-1+r.
  const C = 46;
  const L = 92;
  const col = (k: number) => L + (k + r) * C;
  const W = L + (n + 2 * r) * C + 8;
  const rows = { x: 26, f: 104, prod: 170, y: 262 };

  return (
    <Bench
      title={tr('The sliding window', 'Cửa sổ trượt')}
      task={tr('Drag the output index across the array, then change the boundary policy and switch the flip on.', 'Kéo chỉ số đầu ra dọc theo mảng, rồi đổi chính sách biên và bật chế độ lật.')}
      foot={
        <Translated vi={<>Các ô nét đứt là ô ma (ghost cell): những vị trí nằm ngoài x. Giá trị chúng giữ phụ thuộc vào chính sách. Các đầu ra vẽ nét đứt có cửa sổ chạm tới một ô ma. Có thể gõ phân số như 1/3 vào x và f.</>}>
        <>
          Dashed cells are ghost cells: positions outside x. What they hold depends on the policy. Outputs drawn dashed
          have a window that reaches a ghost cell. Fractions such as 1/3 can be typed into x and f.
        </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Dữ liệu">Data</Translated>
          <select value={preset} onChange={(e) => load(e.target.value)} aria-label={tr('Preset', 'Bộ dữ liệu mẫu')}>
            {PRESETS.map((q) => (
              <option key={q.key} value={q.key}>
                {q.label}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          x
          <input type="text" value={xText} onChange={(e) => setXText(e.target.value)} aria-label="x" style={{ width: '14rem' }} />
        </label>
        <label className="field">
          f
          <input type="text" value={fText} onChange={(e) => setFText(e.target.value)} aria-label="f" style={{ width: '9rem' }} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi={<>Chỉ số đầu ra i = {ii}</>}>Output index i = {ii}</Translated>
          <input type="range" min={0} max={n - 1} value={ii} onChange={(e) => setI(+e.target.value)} aria-label={tr('Output index', 'Chỉ số đầu ra')} style={{ width: '12rem' }} />
        </label>
        <span className="seg" role="group" aria-label={tr('Boundary policy', 'Chính sách biên')}>
          {POLICIES.map((q) => (
            <button key={q} aria-pressed={policy === q} onClick={() => setPolicy(q)}>
              {tr(POLICY_LABEL[q], { zero: 'ô ma bằng 0', replicate: 'nhân bản giá trị biên', wrap: 'quấn vòng' }[q])}
            </button>
          ))}
        </span>
        <label className="check">
          <input type="checkbox" checked={flip} onChange={(e) => setFlip(e.target.checked)} />{' '}
          <Translated vi="lật bộ lọc (tích chập kiểu giáo trình)">flip the filter (textbook
          convolution)</Translated>
        </label>
      </div>
      {xErr || fErr ? (
        <p className="t-fail" role="alert" style={{ margin: '0 0 0.6rem' }}>
          {xErr ?? fErr} <Translated vi="Đang hiển thị dữ liệu hợp lệ gần nhất.">Showing the last valid data.</Translated>
        </p>
      ) : null}
      <svg
        viewBox={`0 0 ${W} 300`}
        role="img"
        aria-label={tr(`Window of y[${ii}]`, `Cửa sổ của y[${ii}]`)}
        data-i={ii}
        data-sum={hide(ii) ? '' : w.sum}
        style={{ width: '100%', maxWidth: W, minWidth: 460, display: 'block' }}
      >
        <text x={8} y={rows.x + 24} fontSize="13" fontWeight="600">x</text>
        {Array.from({ length: n + 2 * r }, (_, c) => {
          const k = c - r;
          const ghost = k < 0 || k >= n;
          const inWin = k >= ii - r && k <= ii + r;
          const tap = inWin ? w.taps[k - (ii - r)] : null;
          const shown = ghost ? (inWin ? tap!.value : null) : x[k];
          return (
            <g key={c}>
              <rect x={col(k)} y={rows.x} width={C - 4} height={34} rx="2"
                fill={inWin ? 'var(--rule-soft)' : ghost ? 'transparent' : 'var(--cell)'}
                stroke={inWin ? 'var(--ink)' : 'var(--rule)'} strokeWidth={inWin ? 1.8 : 1}
                strokeDasharray={ghost ? '4 3' : undefined} />
              {shown != null ? (
                <text x={col(k) + (C - 4) / 2} y={rows.x + 22} fontSize="13" textAnchor="middle"
                  fontWeight={k === ii ? 700 : 400} style={{ fill: ghost ? 'var(--ink-2)' : 'var(--ink)' }}>
                  {fmtNum(shown)}
                </text>
              ) : null}
              <text x={col(k) + (C - 4) / 2} y={rows.x - 6} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>
                {`x[${k}]`}
              </text>
            </g>
          );
        })}
        <text x={8} y={rows.f + 22} fontSize="13" fontWeight="600">{flip ? tr('f, reversed', 'f, đảo ngược') : 'f'}</text>
        {w.taps.map((t) => (
          <g key={'f' + t.j}>
            <rect x={col(ii + t.j)} y={rows.f} width={C - 4} height={30} rx="2" fill="var(--cell)" stroke="var(--ink-2)" />
            <text x={col(ii + t.j) + (C - 4) / 2} y={rows.f + 20} fontSize="13" textAnchor="middle">{fmtNum(t.weight)}</text>
            <text x={col(ii + t.j) + (C - 4) / 2} y={rows.f + 44} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>
              f[{t.fIndex}]
            </text>
          </g>
        ))}
        <text x={8} y={rows.prod + 22} fontSize="13" fontWeight="600"><Translated vi="tích">products</Translated></text>
        {w.taps.map((t) => (
          <g key={'p' + t.j}>
            <rect x={col(ii + t.j)} y={rows.prod} width={C - 4} height={30} rx="2" fill="var(--rule-soft)" stroke="var(--rule)"
              strokeDasharray={t.ghost ? '4 3' : undefined} />
            <text x={col(ii + t.j) + (C - 4) / 2} y={rows.prod + 20} fontSize="13" textAnchor="middle">
              {hide(ii) ? '?' : fmtNum(t.product)}
            </text>
          </g>
        ))}
        <text x={col(ii - r)} y={rows.prod + 56} fontSize="13.5" fontWeight="600" data-out="sum">
          {ex2Locked
            ? tr('These outputs are Exercise 2: answer it below to see them', 'Các đầu ra này là Bài tập 2: trả lời nó ở dưới để xem')
            : hide(ii)
              ? tr('y[0] is Exercise 1: commit a prediction above to see it', 'y[0] là Bài tập 1: chốt một dự đoán ở trên để xem')
              : `y[${ii}] = ${w.taps.map((t) => fmtNum(t.product)).join(' + ').replace(/\+ -/g, '− ')} = ${fmtNum(w.sum)}`}
        </text>
        <text x={8} y={rows.y + 22} fontSize="13" fontWeight="600">y</text>
        {y.map((v, k) => (
          <g key={'y' + k} style={{ cursor: 'pointer' }} onClick={() => setI(k)}>
            <rect x={col(k)} y={rows.y} width={C - 4} height={30} rx="2" fill={k === ii ? 'var(--rule-soft)' : 'var(--cell)'}
              stroke={k === ii ? 'var(--ink)' : 'var(--rule)'} strokeWidth={k === ii ? 1.8 : 1}
              strokeDasharray={touches[k] ? '4 3' : undefined} />
            <text x={col(k) + (C - 4) / 2} y={rows.y + 20} fontSize="13" textAnchor="middle" fontWeight={k === ii ? 700 : 400}>
              {hide(k) ? '?' : fmtNum(v)}
            </text>
          </g>
        ))}
      </svg>
      <p className="ui" style={{ margin: '0.6rem 0 0' }} data-out="y" data-y={JSON.stringify(y.map((v, k) => (hide(k) ? null : Math.round(v * 1000) / 1000)))}>
        y = [{y.map((v, k) => (hide(k) ? '?' : fmtNum(v))).join(', ')}].{' '}
        <span data-out="policy-changed">
          {policy === 'zero'
            ? tr(`Outputs whose window reaches a ghost cell: ${touches.map((t, k) => (t ? `y[${k}]` : null)).filter(Boolean).join(', ') || 'none'}.`, `Các đầu ra có cửa sổ chạm tới ô ma: ${touches.map((t, k) => (t ? `y[${k}]` : null)).filter(Boolean).join(', ') || 'không có'}.`)
            : tr(`Compared with zero ghost cells, this policy changed ${byPolicy.length ? byPolicy.map((k) => `y[${k}]`).join(', ') : 'nothing'}.`, `So với ô ma bằng 0, chính sách này đã thay đổi ${byPolicy.length ? byPolicy.map((k) => `y[${k}]`).join(', ') : 'không gì cả'}.`)}
        </span>{' '}
        {flip ? (
          <span data-out="flip-changed">
            <Translated vi={<>So với dạng không lật của sách, việc lật đã thay đổi {byFlip.length ? byFlip.map((k) => `y[${k}]`).join(', ') : 'không gì cả'}.</>}>
              Compared with the book&rsquo;s un-flipped form, the flip changed {byFlip.length ? byFlip.map((k) => `y[${k}]`).join(', ') : 'nothing'}.
            </Translated>
          </span>
        ) : null}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: one output's patch
function PatchExplorer() {
  const { N, F } = BOOK_2D;
  const [sel, setSel] = useState({ row: 2, col: 2 });
  const P = useMemo(() => conv2d(N, F), [N, F]);
  const p = patch2d(N, F, sel.row, sel.col);
  const r = p.ry;
  const rowSums = [0, 1, 2, 3, 4].map((k) => p.taps.slice(k * 5, k * 5 + 5).reduce((s, t) => s + t.product, 0));
  const inPatch = (row: number, col: number) => Math.abs(row - sel.row) <= r && Math.abs(col - sel.col) <= r;
  const cell = { minWidth: '1.9rem' };
  return (
    <Bench
      title={tr("One output's patch", 'Vùng của một đầu ra')}
      task={tr('Click any output of P. Try the centre, an edge and a corner.', 'Nhấp vào bất kỳ đầu ra nào của P. Thử ở tâm, ở cạnh và ở góc.')}
      foot={tr("The input and filter are the book's Fig. 7.4. Ghost cells (dashed) hold 0, the book's default.", 'Đầu vào và bộ lọc là Fig. 7.4 của sách. Ô ma (nét đứt) giữ giá trị 0, mặc định của sách.')}
    >
      <div className="row" style={{ alignItems: 'flex-start', gap: '1.4rem' }}>
        <div>
          <div className="mx-label"><Translated vi={<>N, với vùng của P[{sel.row}][{sel.col}]</>}>N, with the patch of P[{sel.row}][{sel.col}]</Translated></div>
          <div className="mx" style={{ gridTemplateColumns: `repeat(${N.width + 2 * r}, auto)` }} data-grid="N">
            {Array.from({ length: (N.height + 2 * r) * (N.width + 2 * r) }, (_, k) => {
              const row = Math.floor(k / (N.width + 2 * r)) - r;
              const col = (k % (N.width + 2 * r)) - r;
              const ghost = row < 0 || row >= N.height || col < 0 || col >= N.width;
              const on = inPatch(row, col);
              if (ghost)
                return on ? (
                  <span key={k} className="void" style={cell}>0</span>
                ) : (
                  <span key={k} style={{ ...cell, visibility: 'hidden' }} />
                );
              const centre = row === sel.row && col === sel.col;
              return (
                <span key={k} style={{ ...cell, ...(on ? { background: 'var(--rule-soft)', borderColor: 'var(--ink)' } : {}), fontWeight: centre ? 700 : 400 }}>
                  {N.data[row * N.width + col]}
                </span>
              );
            })}
          </div>
        </div>
        <div>
          <div className="mx-label"><Translated vi="bộ lọc">filter</Translated></div>
          <div className="mx" style={{ gridTemplateColumns: 'repeat(5, auto)' }}>
            {F.data.map((v, k) => (
              <span key={k} style={cell}>{v}</span>
            ))}
          </div>
        </div>
        <div>
          <div className="mx-label"><Translated vi="tích">products</Translated></div>
          <div className="mx" style={{ gridTemplateColumns: 'repeat(5, auto)' }} data-grid="products">
            {p.taps.map((t, k) => (
              <span key={k} className={t.ghost ? 'void' : ''} style={cell}>{t.product}</span>
            ))}
          </div>
        </div>
        <div>
          <div className="mx-label"><Translated vi="P (nhấp vào một đầu ra)">P (click an output)</Translated></div>
          <div className="mx" style={{ gridTemplateColumns: `repeat(${N.width}, auto)` }} data-grid="P">
            {P.data.map((v, k) => {
              const row = Math.floor(k / N.width);
              const col = k % N.width;
              const on = row === sel.row && col === sel.col;
              return (
                <span key={k} role="button" tabIndex={0} aria-label={`P[${row}][${col}]`} data-rc={`${row},${col}`}
                  onClick={() => setSel({ row, col })} onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && setSel({ row, col })}
                  style={{ ...cell, cursor: 'pointer', ...(on ? { background: 'var(--ink)', color: 'var(--page)', fontWeight: 700 } : {}) }}>
                  {v}
                </span>
              );
            })}
          </div>
        </div>
      </div>
      <p className="ui" style={{ margin: '0.9rem 0 0' }} data-out="patch" data-sum={p.sum} data-ghost={p.ghostTaps}>
        <Translated vi={<>P[{sel.row}][{sel.col}] = {rowSums.join(' + ')} = <b>{p.sum}</b> (mỗi hàng tích một số hạng).{' '}</>}>
          P[{sel.row}][{sel.col}] = {rowSums.join(' + ')} = <b>{p.sum}</b> (one term per row of products).{' '}
        </Translated>
        {p.ghostTaps ? tr(`${p.ghostTaps} of its 25 taps fall on ghost cells and add nothing.`, `${p.ghostTaps} trên 25 tap của nó rơi vào ô ma và không cộng thêm gì.`) : tr('Its whole patch lies inside the image.', 'Toàn bộ vùng của nó nằm trong ảnh.')}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: count the ghost cells
const M_CHOICES = [1, 3, 5, 7, 9, 11, 13, 15];

function GhostCounter() {
  const { progress } = useProgress();
  const [n, setN] = useState(7);
  const [m, setM] = useState(5);
  const [showF, setShowF] = useState(false);
  const r = radiusOf(m);
  const b = ghostCounts2d(n, n, m, m);
  const fm = ghostFormula2d(n, n, m, m);
  const answered = !!progress.answers['ch07/7.1/q-ex5'];
  // 8×8 with a 3×3 filter is the Exercise 5 question: its counts of what is skipped stay hidden until it is answered.
  const ex5Locked = !answered && n === 8 && m === 3;
  const skipped = useMemo(() => skippedMap(n, n, m, m), [n, m]);
  const maxSkip = Math.max(1, ...skipped);

  // Picture: the image with each output shaded by how many of its taps are skipped, inside
  // its frame of ghost cells.
  const S = 300;
  const cellPx = S / (n + 2 * r);
  const o = r * cellPx;
  return (
    <Bench
      title={tr('Count the ghost cells', 'Đếm ô ma')}
      task={tr('Set the image and filter sizes. Find how each count grows, then compare with the formulas.', 'Đặt kích thước ảnh và bộ lọc. Tìm xem mỗi con số tăng thế nào, rồi so với các công thức.')}
      foot={tr("Every count is made by visiting every tap of every output. Dashed: the frame of ghost cells. Shading: how many of an output's taps are skipped.", 'Mỗi con số được đếm bằng cách duyệt mọi tap của mọi đầu ra. Nét đứt: khung ô ma. Độ đậm: số tap của một đầu ra bị bỏ qua.')}
    >
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 2fr)' }}>
        <div>
          <div className="row" style={{ marginBottom: '0.8rem' }}>
            <label className="field">
              <Translated vi={<>Ảnh N×N, N = {n}</>}>Image N×N, N = {n}</Translated>
              <input type="range" min={1} max={64} value={n} onChange={(e) => setN(+e.target.value)} aria-label={tr('Image width N', 'Chiều rộng ảnh N')} style={{ width: '13rem' }} />
            </label>
            <label className="field">
              <Translated vi="Bộ lọc M×M">Filter M×M</Translated>
              <select value={m} onChange={(e) => setM(+e.target.value)} aria-label={tr('Filter width M', 'Chiều rộng bộ lọc M')}>
                {M_CHOICES.map((v) => (
                  <option key={v} value={v}>
                    {v} × {v} (r = {radiusOf(v)})
                  </option>
                ))}
              </select>
            </label>
          </div>
          <table className="data" style={{ margin: 0 }} data-out="counts">
            <thead>
              <tr>
                <th><Translated vi="Đại lượng">Count</Translated></th>
                <th><Translated vi="Bằng cách duyệt mọi tap">By visiting every tap</Translated></th>
                {showF ? <th><Translated vi="Công thức">Formula</Translated></th> : null}
              </tr>
            </thead>
            <tbody>
              <tr data-row="ghost">
                <td><Translated vi="Ô ma">Ghost cells</Translated></td>
                <td>{b.ghostCells.toLocaleString('en-US')}</td>
                {showF ? <td>{fm.ghostCells.toLocaleString('en-US')}</td> : null}
              </tr>
              <tr data-row="all">
                <td><Translated vi="Phép nhân, tính cả tap rơi vào ô ma">Multiplications, ghost taps included</Translated></td>
                <td>{b.allTaps.toLocaleString('en-US')}</td>
                {showF ? <td>{fm.allTaps.toLocaleString('en-US')}</td> : null}
              </tr>
              <tr data-row="real">
                <td><Translated vi="Phép nhân, bỏ qua tap rơi vào ô ma">Multiplications, ghost taps skipped</Translated></td>
                <td>{ex5Locked ? '?' : b.realTaps.toLocaleString('en-US')}</td>
                {showF ? <td>{fm.realTaps.toLocaleString('en-US')}</td> : null}
              </tr>
            </tbody>
          </table>
          <p className="ui" style={{ margin: '0.7rem 0 0' }} data-out="share">
            {ex5Locked ? (
              <Translated vi={<>Đây là câu hỏi Bài tập 5: trả lời nó ở trên để xem việc bỏ qua tiết kiệm được bao nhiêu ở đây.</>}>
                <>This is the Exercise 5 question: answer it above to see what skipping saves here.</>
              </Translated>
            ) : (
              <Translated vi={<>Bỏ qua các tap rơi vào ô ma tiết kiệm {fmtNum(Math.round(((b.allTaps - b.realTaps) / b.allTaps) * 1000) / 10)} % số phép nhân ở đây.</>}>
              <>
                Skipping ghost taps saves {fmtNum(Math.round(((b.allTaps - b.realTaps) / b.allTaps) * 1000) / 10)} % of
                the multiplications here.
              </>
              </Translated>
            )}
          </p>
          <div style={{ marginTop: '0.9rem' }}>
            {showF ? (
              <div className="panel" data-out="formulas">
                <Translated vi={<p style={{ margin: '0 0 0.4rem' }}>Với r = (M − 1)/2, cho một ảnh N×N và một bộ lọc M×M (Bài tập 5):</p>}>
<p style={{ margin: '0 0 0.4rem' }}>
                  With r = (M − 1)/2, for an N×N image and an M×M filter (Exercise 5):
                </p>
</Translated>
                <ul style={{ margin: '0 0 0.4rem', paddingLeft: '1.2rem' }}>
                  <li><Translated vi="ô ma: (N + M − 1)² − N² = 4r(N + r)">ghost cells: (N + M − 1)² − N² = 4r(N + r)</Translated></li>
                  <li><Translated vi="phép nhân tính cả tap rơi vào ô ma: N²·M²">multiplications with ghost taps: N²·M²</Translated></li>
                  <li><Translated vi="không tính chúng: (N·M − r(r + 1))²">without them: (N·M − r(r + 1))²</Translated></li>
                </ul>
                <Translated vi={<p style={{ margin: '0 0 0.4rem' }}>Trong 1D (Bài tập 4) các con số tương ứng là M − 1, N·M và N·M − r(r + 1): mỗi đầu mất 1 + 2 + … + r tap. Với một ảnh N1×N2 và một bộ lọc M1×M2 (Bài tập 6), nhân hai chiều lại: (N1 + M1 − 1)(N2 + M2 − 1) − N1·N2, N1·N2·M1·M2, và (N1·M1 − r1(r1 + 1))·(N2·M2 − r2(r2 + 1)).</p>}>
<p style={{ margin: '0 0 0.4rem' }}>
                  In 1D (Exercise 4) the same counts are M − 1, N·M and N·M − r(r + 1): each end loses 1 + 2 + … + r
                  taps. For an N1×N2 image and an M1×M2 filter (Exercise 6) multiply the two directions: (N1 + M1 − 1)(N2
                  + M2 − 1) − N1·N2, N1·N2·M1·M2, and (N1·M1 − r1(r1 + 1))·(N2·M2 − r2(r2 + 1)).
                </p>
</Translated>
                <p className="muted" style={{ margin: 0 }}>
                  <Translated vi={<>Công thức cuối giả định N ≥ r: mỗi đầu có r đầu ra để mất tap.{' '}</>}>
                    The last formula assumes N ≥ r: every end has r outputs to lose taps from.{' '}
                  </Translated>
                  {n < r ? <b><Translated vi={<>Ở đây N &lt; r, và nó không khớp với số đếm.</>}>Here N &lt; r, and it disagrees with the count.</Translated></b> : null}
                </p>
              </div>
            ) : (
              <p className="ui" style={{ margin: 0 }}>
                <button onClick={() => setShowF(true)} disabled={!answered}>
                  <Translated vi="Hiện các công thức">Show the formulas</Translated>
                </button>{' '}
                {answered ? null : <span className="muted"><Translated vi="Hãy trả lời câu hỏi Bài tập 5 ở trên trước.">Answer the Exercise 5 question above first.</Translated></span>}
              </p>
            )}
          </div>
        </div>
        <svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label={tr('Image with its frame of ghost cells', 'Ảnh cùng khung ô ma của nó')} style={{ width: '100%', maxWidth: 320 }}>
          {r > 0 ? (
            <path d={`M0 0H${S}V${S}H0Z M${o} ${o}V${o + n * cellPx}H${o + n * cellPx}V${o}Z`} fillRule="evenodd" /* i18n-ignore: SVG value */
              fill="var(--rule-soft)" stroke="var(--ink-2)" strokeWidth="1" strokeDasharray="4 3" />
          ) : null}
          {r > 0 && cellPx >= 7
            ? Array.from({ length: (n + 2 * r) * (n + 2 * r) }, (_, k) => {
                const row = Math.floor(k / (n + 2 * r)) - r;
                const col = (k % (n + 2 * r)) - r;
                if (row >= 0 && row < n && col >= 0 && col < n) return null;
                return (
                  <rect key={'g' + k} x={o + col * cellPx + 1} y={o + row * cellPx + 1} width={cellPx - 2} height={cellPx - 2}
                    fill="none" stroke="var(--ink-3)" strokeWidth="0.8" strokeDasharray="2 2" />
                );
              })
            : null}
          <rect x={o} y={o} width={n * cellPx} height={n * cellPx} fill="var(--cell)" stroke="var(--ink)" strokeWidth="1" />
          {skipped.map((s, k) =>
            s ? (
              <rect key={k} x={o + (k % n) * cellPx} y={o + Math.floor(k / n) * cellPx} width={cellPx} height={cellPx}
                fill="var(--ink)" fillOpacity={0.12 + 0.6 * (s / maxSkip)} />
            ) : null,
          )}
          <text x={S / 2} y={S / 2 + 4} fontSize="12" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
            {n} × {n}
          </text>
        </svg>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the section
export function S71() {
  return (
    <>
      <Translated vi={<p>Một phép tích chập (convolution) thay mỗi phần tử của một mảng bằng tổng có trọng số của chính nó và các phần tử lân cận. Các trọng số giống nhau cho mọi phần tử và đến từ một mảng nhỏ gọi là <b>bộ lọc</b> (filter). Phần lớn tài liệu gọi nó là convolution kernel; sách gọi là bộ lọc, để dành chữ &ldquo;kernel&rdquo; cho các hàm CUDA tính nó, và khóa học này cũng vậy.</p>}>
<p>
        A convolution replaces every element of an array with a weighted sum of itself and its neighbours. The weights
        are the same for every element and come from a small array called the <b>filter</b>. Most of the literature
        calls it the convolution kernel; the book says filter, to keep &ldquo;kernel&rdquo; for the CUDA functions that
        compute it, and so does this course.
      </p>
</Translated>
      <Source section="7.1" pages="159–160" />
      <Translated vi={<p>Một bộ lọc có số trọng số lẻ, 2r + 1, để nó được căn giữa tại phần tử đang tính với r phần tử lân cận mỗi bên; r là <b>bán kính</b> của nó. Với đầu vào x và bộ lọc f, đầu ra i là</p>}>
<p>
        A filter has an odd number of weights, 2r + 1, so that it is centred on the element being computed with r
        neighbours on each side; r is its <b>radius</b>. For an input x and a filter f, output i is
      </p>
</Translated>
      <p style={{ textAlign: 'center' }}>
        <code>y[i] = f[0]·x[i−r] + f[1]·x[i−r+1] + … + f[2r]·x[i+r]</code>,
      </p>
      <Translated vi={<p>tổng theo j từ −r đến r của <code>f[j+r]·x[i+j]</code>. Hãy đọc nó như một tích vô hướng: bộ lọc đặt lên một cửa sổ của x có tâm tại i. Với x = [8, 2, 5, 4, 1, 7, 3] và f = [1, 3, 5, 3, 1] của sách, r = 2, cửa sổ của y[2] là x[0] đến x[4] và y[2] = 8 + 6 + 25 + 12 + 1 = 52. Trượt một vị trí thì y[3] = 2 + 15 + 20 + 3 + 7 = 47.</p>}>
<p>
        a sum over j from −r to r of <code>f[j+r]·x[i+j]</code>. Read it as an inner product: the filter laid over a
        window of x centred on i. With the book&rsquo;s x = [8, 2, 5, 4, 1, 7, 3] and f = [1, 3, 5, 3, 1], r = 2, the
        window for y[2] is x[0] to x[4] and y[2] = 8 + 6 + 25 + 12 + 1 = 52. Slide one place and y[3] = 2 + 15 + 20 + 3
        + 7 = 47.
      </p>
</Translated>
      <Source section="7.1" pages="160–161" figs="Figs. 7.1, 7.2" />
      <Translated vi={<p>Vì mỗi đầu ra là một tích vô hướng với cùng một bộ lọc, cả phép tích chập có thể viết thành một phép nhân ma trận. Sách để điều đó cho chương 19 (<a href={href('ch19', '19.3')}>§19.3</a>), và các thư viện thường chạy nó theo cách này.</p>}>
<p>
        Since every output is an inner product with the same filter, a whole convolution can be written as one matrix
        multiplication. The book leaves that to chapter 19 (<a href={href('ch19', '19.3')}>§19.3</a>), and it is how libraries usually run it.
      </p>
</Translated>
      <Source pages="161" figs={tr('footnote 1', 'chú thích 1')} />

      <h2><Translated vi="Công thức của sách không lật bộ lọc">The book&rsquo;s formula does not flip the filter</Translated></h2>
      <Translated vi={<p>Hãy xem trọng số nào gặp đầu vào nào. Trong công thức trên, f[0] nhân với phần tử ngoài cùng bên trái của cửa sổ. Phép tích chập trong các giáo trình xử lý tín hiệu đảo ngược bộ lọc trước, để f[0] gặp phần tử ngoài cùng bên phải. Sách không bao giờ nhắc tới khác biệt này, và mọi bộ lọc sách dùng đều đối xứng, nơi hai cách cho cùng kết quả.</p>}>
<p>
        Look at which weight meets which input. In the formula above f[0] multiplies the leftmost element of the
        window. The convolution of signal-processing textbooks reverses the filter first, so that f[0] meets the
        rightmost one. The book never mentions the difference, and every filter it uses is symmetric, where the two
        agree.
      </p>
</Translated>
      <Translated vi={<p>Lựa chọn của sách chính là cách bạn vẫn đang dùng. PyTorch ghi trong tài liệu rằng <code>Conv2d</code> là &ldquo;valid 2D cross-correlation operator&rdquo;, tức dạng không lật này; nó yêu cầu cuDNN dùng chế độ cross-correlation, nhánh ROCm của nó gọi <code>miopenConvolution</code> của MIOpen, được tài liệu mô tả là &ldquo;Cross-Correlation convolution&rdquo;, và <code>conv_general</code> của MLX chỉ lật khi bạn truyền{' '}<code>flip=True</code>. Vì vậy trọng số của một lớp tích chập đã huấn luyện được áp dụng đúng như khi lưu.</p>}>
<p>
        The book&rsquo;s choice is the one you already use. PyTorch documents <code>Conv2d</code> as the &ldquo;valid
        2D cross-correlation operator&rdquo;, which is this un-flipped form; it asks cuDNN for its cross-correlation
        mode, its ROCm path calls MIOpen&rsquo;s <code>miopenConvolution</code>, documented as
        &ldquo;Cross-Correlation convolution&rdquo;, and MLX&rsquo;s <code>conv_general</code> only flips when you pass{' '}
        <code>flip=True</code>. So the weights of a trained convolution layer are applied exactly as stored.
      </p>
</Translated>
      <Today checked="2026-11-20 (notes/today/ch07-facts.md, D1)" /* i18n-ignore: citation */>
        <Translated vi={<p>PyTorch 2.15 <code>Conv2d</code>: &ldquo;valid 2D cross-correlation&rdquo;. Chế độ cuDNN{' '}<code>CUDNN_CROSS_CORRELATION</code> do PyTorch đặt; MIOpen <code>miopenConvolution</code>; MLX 0.32.4{' '}<code>conv_general(..., flip=False)</code>.</p>}>
<p>
          PyTorch 2.15 <code>Conv2d</code>: &ldquo;valid 2D cross-correlation&rdquo;. cuDNN mode{' '}
          <code>CUDNN_CROSS_CORRELATION</code> set by PyTorch; MIOpen <code>miopenConvolution</code>; MLX 0.32.4{' '}
          <code>conv_general(..., flip=False)</code>.
        </p>
</Translated>
      </Today>

      <h2><Translated vi="Ô ma (ghost cell)">Ghost cells</Translated></h2>
      <Translated vi={<p>Gần hai đầu, cửa sổ chạy ra ngoài mảng. Cách sửa của sách là giả vờ các phần tử bị thiếu tồn tại và giữ một giá trị mặc định, thường là 0 (một tín hiệu âm thanh im lặng trước khi bắt đầu ghi). Sách gọi các phần tử tưởng tượng này là <b>ô ma</b> (ghost cell). Với một ô ma ở bên trái, y[1] = 1·0 + 3·8 + 5·2 + 3·5 + 1·4 = 53. Phân ô (tiling) sẽ tạo ra loại ô thừa thứ hai ở phần sau của chương, vùng biên (halo).</p>}>
<p>
        Near the ends the window runs off the array. The book&rsquo;s fix is to pretend the missing elements exist and
        hold a default value, usually 0 (an audio signal is silent before the recording starts). It calls these
        imaginary elements <b>ghost cells</b>. With one ghost cell on the left, y[1] = 1·0 + 3·8 + 5·2 + 3·5 + 1·4 = 53.
        Tiling will create a second kind of extra cell later in the chapter, the halo.
      </p>
</Translated>
      <Source section="7.1" pages="161–162" figs="Fig. 7.3" />
      <Translated vi={<p>Số 0 là một chính sách, không phải định luật. Sách nêu hai chính sách khác: chép phần tử thật gần nhất vào các ô ma, hoặc coi mảng là vòng tròn để nó quấn vòng.</p>}>
<p>
        Zero is a policy, not a law. The book names two others: copy the nearest real element into the ghost cells, or
        treat the array as circular so that it wraps around.
      </p>
</Translated>
      <Predict
        id="ch07/7.1/p-y0"
        prompt={
          <Translated vi={<>Bài tập 1. Với ô ma bằng 0, y[0] bằng bao nhiêu cho x = [8, 2, 5, 4, 1, 7, 3] và f = [1, 3, 5, 3, 1] của sách?</>}>
<>
            Exercise 1. With zero ghost cells, what is y[0] for the book&rsquo;s x = [8, 2, 5, 4, 1, 7, 3] and f = [1, 3,
            5, 3, 1]?
          </>
</Translated>
        }
      >
        {(v) => (
          <>
            <Translated vi={<p>{v === 51 ? 'Đúng: ' : ''}y[0] = 1·0 + 3·0 + 5·8 + 3·2 + 1·5 = <b data-out="y0">51</b>. Lần này có hai ô ma ở bên trái. Trình khám phá bên dưới giờ cũng hiển thị nó.</p>}>
<p>
              {v === 51 ? 'Right: ' : ''}y[0] = 1·0 + 3·0 + 5·8 + 3·2 + 1·5 = <b data-out="y0">51</b>. Two ghost cells on
              the left this time. The explorer below now shows it too.
            </p>
</Translated>
            <Translated vi={<p className="muted">Nguồn: ch. 7 Bài tập 1, p. 180 (bài tập gọi đầu ra là P[0]; hình gọi nó là y).</p>}>
<p className="muted">Source: ch. 7 Exercise 1, p. 180 (the exercise calls the output P[0]; the figure calls it y).</p>
</Translated>
          </>
        )}
      </Predict>
      <SlidingWindow />
      <Translated vi={<p>Ba điều cần tìm trong trình khám phá. Chỉ r đầu ra đầu tiên và r đầu ra cuối cùng thay đổi theo chính sách. Việc lật không thay đổi gì với bộ lọc đối xứng của sách và thay đổi gần như mọi thứ với bộ lọc bất đối xứng. Và các bộ lọc của Bài tập 3 chỉ làm đúng điều hình dạng của chúng gợi ý khi bạn biết định nghĩa nào đang được dùng.</p>}>
<p>
        Three things to find in the explorer. Only the first r and last r outputs change with the policy. The flip
        changes nothing for the book&rsquo;s symmetric filter and nearly everything for the asymmetric one. And the
        Exercise 3 filters do what their shapes suggest only once you know which definition is in force.
      </p>
</Translated>
      <Ask q={questionsS71['ch07/7.1/q-ex2']} label={tr('Exercise 2', 'Bài tập 2')} />
      <Ask q={questionsS71['ch07/7.1/q-ex3']} label={tr('Exercise 3', 'Bài tập 3')} />
      <Ask q={questionsS71['ch07/7.1/q-policy']} />
      <Translated vi={<p>PyTorch đưa ra cùng lựa chọn này. Một <code>nn.Conv2d</code> nhận <code>padding_mode</code>{' '}<code>&apos;zeros&apos;</code>, <code>&apos;replicate&apos;</code> hoặc <code>&apos;circular&apos;</code>, chính là ba chính sách của sách, và <code>&apos;reflect&apos;</code>, chính sách thứ tư mà sách không nhắc tới. Mặc định <code>padding=0</code> của nó nghĩa là hoàn toàn không có ô ma: chỉ những đầu ra có cửa sổ nằm gọn trong đầu vào mới tồn tại, nên đầu ra nhỏ hơn đầu vào.</p>}>
<p>
        PyTorch exposes the same choice. An <code>nn.Conv2d</code> takes <code>padding_mode</code>{' '}
        <code>&apos;zeros&apos;</code>, <code>&apos;replicate&apos;</code> or <code>&apos;circular&apos;</code>, which are
        the book&rsquo;s three policies, and <code>&apos;reflect&apos;</code>, a fourth the book does not mention. Its
        default <code>padding=0</code> means no ghost cells at all: only outputs whose window fits inside the input
        exist, so the output is smaller than the input.
      </p>
</Translated>
      <Beyond>
        <Translated vi={<p>GPU có thể áp dụng các chính sách này bằng phần cứng khi một ảnh được đọc như texture: các address mode Border, Clamp, Wrap và Mirror của CUDA và HIP; các sampler <code>clamp_to_zero</code>, <code>clamp_to_edge</code> và{' '}<code>repeat</code> của Metal. Programming Guide của NVIDIA nói texture không mang lại lợi thế hiệu năng cho mã phi đồ họa trên các GPU hiện nay, nên khóa học, giống sách, đọc các buffer thông thường (facts A6, B4, C5).</p>}>
<p>
          GPUs can apply these policies in hardware when an image is read as a texture: CUDA and HIP address modes
          Border, Clamp, Wrap and Mirror; Metal samplers <code>clamp_to_zero</code>, <code>clamp_to_edge</code> and{' '}
          <code>repeat</code>. NVIDIA&rsquo;s Programming Guide says textures give no performance advantage for
          non-graphics code on current GPUs, so the course, like the book, reads plain buffers (facts A6, B4, C5).
        </p>
</Translated>
      </Beyond>

      <h2><Translated vi="Hai chiều">Two dimensions</Translated></h2>
      <Translated vi={<p>Trong 2D, bộ lọc là một hình vuông nhỏ (không bắt buộc, nhưng thường là vậy) gồm 2r + 1 hàng và cột, và một đầu ra là tổng các tích từng phần tử của bộ lọc với vùng đầu vào có tâm tại nó:{' '}<code>P[y][x]</code> là tổng theo j và k từ −r đến r của <code>f[j+r][k+r]·N[y+j][x+k]</code>. Sách viết các chỉ số dưới dạng chỉ số dưới và lưu ý rằng mã thật tuyến tính hóa chúng, vì N và P được cấp phát động: <code>N[(y+j)*width + (x+k)]</code>, như ở chương 3.</p>}>
<p>
        In 2D the filter is a small square (it need not be, but usually is) of 2r + 1 rows and columns, and an output is
        the sum of the element-by-element products of the filter with the patch of the input centred on it:{' '}
        <code>P[y][x]</code> is the sum over j and k from −r to r of <code>f[j+r][k+r]·N[y+j][x+k]</code>. The book
        writes the indices as subscripts and warns that real code linearises them, because N and P are allocated
        dynamically: <code>N[(y+j)*width + (x+k)]</code>, as in chapter 3.
      </p>
</Translated>
      <Source section="7.1" pages="162–163" figs="Fig. 7.4" />
      <Translated vi={<p>Ví dụ của sách là một đầu vào 7×7 và một bộ lọc 5×5. P[2][2] có toàn bộ vùng nằm trong đầu vào; 25 tích của nó cộng lại được 321 (theo hàng, 27 + 56 + 95 + 84 + 59; các tổng theo hàng là của khóa học này). Tại P[1][0], vùng thò ra ngoài cạnh trên một hàng và ngoài cạnh trái hai cột, và với số 0 ở đó tổng là 112.</p>}>
<p>
        The book&rsquo;s example is a 7×7 input and a 5×5 filter. P[2][2] has its whole patch inside the input; its 25
        products add up to 321 (by rows, 27 + 56 + 95 + 84 + 59; the row sums are ours). At P[1][0] the patch hangs over
        the top edge by one row and over the left edge by two columns, and with zeros there the sum is 112.
      </p>
</Translated>
      <Source section="7.1" pages="163–164" figs="Fig. 7.5" />
      <PatchExplorer />
      <Ask q={questionsS71['ch07/7.1/q-pytorch']} label={tr('In PyTorch', 'Trong PyTorch')} />

      <h2><Translated vi="Bao nhiêu phần công việc nằm ở các cạnh?">How much of the work is at the edges?</Translated></h2>
      <Translated vi={<p>Bài tập 4 đến 6 hỏi có bao nhiêu ô ma và một kernel làm bao nhiêu phép nhân khi tính và khi không tính các tap rơi vào ô ma. Câu trả lời quan trọng cho mục 7.2: một luồng bỏ qua đúng những tap rơi vào ô ma của nó, nên các con số này cho biết bao nhiêu phần ảnh nằm ở đường viền khó xử. Hãy giải trường hợp 8×8 trước, rồi dùng bộ đếm để kiểm tra và tìm các công thức tổng quát.</p>}>
<p>
        Exercises 4 to 6 ask how many ghost cells there are and how many multiplications a kernel does with and without
        the ghost taps. The answers matter for section 7.2: a thread skips exactly its ghost taps, so these counts say
        how much of the image sits in the awkward border. Work the 8×8 case first, then use the counter to check it and
        to find the general formulas.
      </p>
</Translated>
      <Source pages="180" figs={tr('Exercises 4–6', 'Bài tập 4–6')} />
      <Ask q={questionsS71['ch07/7.1/q-ex5']} label={tr('Exercise 5', 'Bài tập 5')} />
      <Translated vi={<p>Bộ đếm duyệt mọi tap của mọi đầu ra và đếm. Hãy đổi N và M lần lượt từng cái và tìm quy luật trước khi mở các công thức.</p>}>
<p>
        The counter visits every tap of every output and counts. Change N and M one at a time and look for the pattern
        before you open the formulas.
      </p>
</Translated>
      <Simplification>
        <Translated vi={<p>Bộ đếm đếm tap, không đếm thời gian. Một tap bị bỏ qua có rẻ hơn phép nhân với 0 hay không tùy vào phần cứng và trình biên dịch; mục 7.2 xem nhánh rẽ đó gây ra gì cho một warp.</p>}>
<p>
          The counter counts taps, not time. Whether a skipped tap is cheaper than multiplying by a zero depends on the
          hardware and the compiler; section 7.2 looks at what the branch does to a warp.
        </p>
</Translated>
      </Simplification>
      <GhostCounter />
      <Translated vi={<p>Với một ảnh lớn, các tap bị bỏ qua chỉ là một đường viền mỏng: phần chúng tiết kiệm giảm xấp xỉ theo 1/N. Vì vậy chính sách biên là câu hỏi về việc ra đúng đáp án, còn các luồng ở cạnh chỉ là một phần nhỏ của chi phí. Mục{' '}<a href={href('ch07', '7.2')}>7.2</a> biến công thức thành một kernel với mỗi đầu ra một luồng.</p>}>
<p>
        For a large image the skipped taps are a thin border: the share they save falls roughly as 1/N. So the boundary
        policy is a question of getting the right answer, and the edge threads are a small part of the cost. Section{' '}
        <a href={href('ch07', '7.2')}>7.2</a> turns the formula into a kernel with one thread per output.
      </p>
</Translated>
    </>
  );
}
