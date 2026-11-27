import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Aside, Bench, Beyond, NoteGroup, Simplification, Source, Today, type Tier } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { OPENING_PAIR, TEMP_ROWS, argmax, entropyBits, reciprocalMismatch, softmax, temperature } from '../../sim/chain';
import { functionSource } from '../../sim/toolchain';

const FACTS = 'notes/today/x1-facts.md';
const LAB = 'ch10/lab3_softmax_argmax';
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const TIERS: Record<string, Tier> = {
  x: 'global', y: 'global', in: 'global', out: 'global',
  scratch: 'shared',
  m: 'reg', s: 'reg', c: 'reg', inv: 'reg', scale: 'reg',
};

// log slider: 0..1000 maps to T in [0.05, 4]
const T_MIN = 0.05;
const T_MAX = 4;
const tFrom = (v: number) => Number((T_MIN * Math.pow(T_MAX / T_MIN, v / 1000)).toPrecision(2));
const toSlider = (T: number) => Math.round((Math.log(T / T_MIN) / Math.log(T_MAX / T_MIN)) * 1000);

// ---------------------------------------------------------------- questions
export const questionsSX12: Record<string, Question> = {
  'x1/X1.2/q-tie-limit': {
    kind: 'choice',
    id: 'x1/X1.2/q-tie-limit',
    prompt: <Translated vi={<>Logits [3.0, 3.0, 1.0]: hai giá trị lớn nhất bằng nhau tuyệt đối. softmax(l / T) tiến tới đâu khi T tiến về 0 từ phía trên?</>}><>Logits [3.0, 3.0, 1.0]: two exact maxima. What does softmax(l / T) approach as T goes to 0 from above?</></Translated>,
    options: [
      { label: '[0.5, 0.5, 0]', correct: true },
      { label: '[1, 0, 0]', why: <Translated vi="Không có gì trong công thức ưu tiên logit đầu tiên trong hai logit bằng nhau. Sự ưu tiên đó là một quy tắc phân định hòa, thứ softmax không có.">Nothing in the formula prefers the first of two equal logits. That preference is a tie rule, which softmax does not have.</Translated> },
      { label: <Translated vi="Nó không xác định, vì phép chia cho 0">It is undefined, because of the division by zero</Translated>, why: <Translated vi="Mọi T lớn hơn 0 đều ổn, và giới hạn tồn tại: trừ đi giá trị lớn nhất và xem mỗi số mũ làm gì.">Every T above 0 is fine, and the limit exists: subtract the maximum and look at what each exponent does.</Translated> },
      { label: '[0.33, 0.33, 0.33]', why: <Translated vi="Đó là giới hạn khi T tăng, không phải khi nó giảm.">That is the limit as T grows, not as it shrinks.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
          Sau khi trừ giá trị lớn nhất, hai logit hòa nhau mỗi cái đóng góp exp(0) = 1 ở mọi T, còn cái thứ ba đóng góp
          exp(−2 / T), tiến về 0. Vậy cặp hòa nhau chia đều khối xác suất. "T → 0 là greedy" cần một quy tắc phân định hòa
          để gọi tên một token; quy tắc của M5 là id thấp nhất.
        </p>
        }
      >
      <p>
        After subtracting the maximum, the two tied logits contribute exp(0) = 1 each at every T, and the third
        contributes exp(−2 / T), which goes to 0. So the tied pair share the mass equally. "T → 0 is greedy" needs a tie
        rule to name one token; M5's is the lowest id.
      </p>
      </Translated>
    ),
    source: 'ours; softmax with the maximum subtracted: ch. 20 Eq. 20.2, p. 484',
    vi: { source: 'của khóa học này; softmax có trừ giá trị lớn nhất: ch. 20 Eq. 20.2, p. 484' },
  },
  'x1/X1.2/q-fold-bytes': {
    kind: 'numeric',
    id: 'x1/X1.2/q-fold-bytes',
    prompt: (
      <Translated
        vi={
          <>
            Áp dụng temperature bằng một kernel theo từng phần tử riêng trước softmax của lab 10.3, thay vì qua đối số{' '}
            <code>scale</code> của softmax: kernel đó di chuyển thêm ít nhất bao nhiêu byte cho mỗi logit float32?
          </>
        }
      >
      <>
        Applying temperature as its own elementwise kernel before lab 10.3's softmax, instead of through softmax's{' '}
        <code>scale</code> argument: how many extra bytes per float32 logit does that kernel move, at least?
      </>
      </Translated>
    ),
    answer: 8,
    unit: 'bytes',
    explain: (
      <Translated
        vi={
          <p>
          Nó đọc mỗi logit (4 byte) và ghi logit đã chia tỉ lệ (4 byte). Sau đó softmax đọc lại nó, như nó lẽ ra đã đọc bản
          gốc. Truyền 1/T làm <code>scale</code> tốn một phép nhân cho mỗi phần tử, điều mà softmax bị giới hạn bởi bộ nhớ
          che giấu hoàn toàn.
        </p>
        }
      >
      <p>
        It reads each logit (4 bytes) and writes the scaled logit (4 bytes). The softmax then reads it again, as it would
        have read the original. Passing 1/T as <code>scale</code> costs one multiplication per element, which the
        memory-bound softmax hides entirely.
      </p>
      </Translated>
    ),
    source: 'ours; lab 10.3 README (launchSoftmaxRows takes scale)',
    vi: { unit: 'byte', source: 'của khóa học này; README của lab 10.3 (launchSoftmaxRows nhận scale)' },
  },
  'x1/X1.2/q-reciprocal': {
    kind: 'choice',
    id: 'x1/X1.2/q-reciprocal',
    prompt: (
      <Translated
        vi={
          <>
            Phía host của llama.cpp chia mỗi logit cho T; backend của nó nhân với 1/T đã tính trước. Hai cách có thể cho ra
            các logit khác nhau không?
          </>
        }
      >
      <>
        llama.cpp's host divides each logit by T; its backend multiplies by a precomputed 1/T. Can the two give different
        logits?
      </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Có, ở bit cuối: 1/T được làm tròn một lần và tích được làm tròn thêm lần nữa, trong khi phép chia chỉ làm tròn một lần.">Yes, in the last bit: 1/T is rounded once and the product rounded again, while the division rounds once.</Translated>, correct: true },
      { label: <Translated vi="Không: nhân với nghịch đảo là chính xác trong số học IEEE.">No: multiplying by the reciprocal is exact in IEEE arithmetic.</Translated>, why: <Translated vi="1/T chỉ chính xác khi T là lũy thừa của hai. Nếu không, nó đã bị làm tròn trước phép nhân.">1/T is exact only when T is a power of two. Otherwise it is already rounded before the multiplication.</Translated> },
      { label: <Translated vi="Chỉ trên Metal, nơi chế độ toán học mặc định không chặt chẽ.">Only on Metal, whose default math mode is not strict.</Translated>, why: <Translated vi="Hai phép toán được làm tròn đúng có thể khác với một phép trên bất kỳ thiết bị IEEE nào. Metal còn thêm lý do riêng của nó.">Two correctly rounded operations can differ from one on any IEEE device. Metal adds its own reasons on top.</Translated> },
      { label: <Translated vi="Chỉ với logit âm.">Only for negative logits.</Translated>, why: <Translated vi="Dấu không đóng vai trò gì; việc làm tròn 1/T mới có.">The sign plays no part; the rounding of 1/T does.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
          l / T là thương được làm tròn chính xác. l × fl(1/T) nhân với một nghịch đảo đã được làm tròn rồi làm tròn lần nữa,
          nên nó có thể lệch một bậc. Khác biệt rất nhỏ, nhưng khi hai token hòa nhau chính xác hoặc gần hòa, nó có thể thay
          đổi token nào thắng, và đây là lý do đầu tiên trong nhiều lý do khiến đường host và đường device có thể bất đồng.
          Mục X1.8 xử lý phần còn lại.
        </p>
        }
      >
      <p>
        l / T is the exactly rounded quotient. l × fl(1/T) multiplies by an already rounded reciprocal and rounds again,
        so it can land one step away. The difference is tiny, but on an exact or near tie between two tokens it can change
        which one wins, and it is the first of several reasons a host path and a device path can disagree. Section X1.8
        deals with the rest.
      </p>
      </Translated>
    ),
    source: 'notes/today/x1-facts.md D2 (our note on the last bit)',
    vi: { source: 'notes/today/x1-facts.md D2 (ghi chú của chúng tôi về bit cuối)' },
  },
  'x1/X1.2/q-greedy-ties': {
    kind: 'choice',
    id: 'x1/X1.2/q-greedy-ties',
    prompt: <Translated vi={<>Một hàng có các giá trị lớn nhất bằng nhau ở cột 3 và 7. Phát biểu nào đúng?</>}><>A row has equal maxima at columns 3 and 7. Which statement is right?</></Translated>,
    options: [
      {
        label: <Translated vi="Tài liệu torch.argmax ghi cột 3 (giá trị lớn nhất đầu tiên), M5 trả về 3, argmax Metal của llama.cpp trả về 7 (theo cách chúng tôi đọc mã).">torch.argmax documents column 3 (the first maximum), M5 returns 3, llama.cpp's Metal argmax returns 7 (our reading).</Translated>,
        correct: true,
      },
      { label: <Translated vi="Mọi backend trả về 3: argmax là một phép rút gọn với cây cố định, và mỗi bước kết hợp giữ toán hạng bên trái khi hòa, nên giá trị lớn nhất đầu tiên thắng.">Every backend returns 3: argmax is a reduction with a fixed tree, and each combine keeps the left operand on a tie, so the first maximum wins.</Translated>, why: <Translated vi="Không phải bước kết hợp của dự án nào cũng giữ toán hạng bên trái: theo cách chúng tôi đọc, argmax Metal của llama.cpp lấy chỉ số lớn nhất trong các lane đang giữ giá trị lớn nhất, nên nó trả về 7.">Not every project's combine keeps the left operand: by our reading, llama.cpp's Metal argmax takes the largest index among the lanes holding the maximum, so it returns 7.</Translated> },
      { label: <Translated vi="Mọi backend trả về 7: mỗi bước kết hợp so sánh bằng ≥, nên giá trị lớn nhất đến sau thay thế giá trị đến trước.">Every backend returns 7: each combine compares with ≥, so a later maximum replaces an earlier one.</Translated>, why: <Translated vi="Tài liệu PyTorch ghi giá trị lớn nhất đầu tiên, và bước kết hợp của M5 giữ chỉ số thấp hơn.">PyTorch documents the first maximum, and M5's combine keeps the lower index.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
          Tài liệu PyTorch ghi giá trị lớn nhất đầu tiên và bộ so sánh của nó giữ chỉ số thấp hơn. Kernel Metal của llama.cpp
          lấy chỉ số lớn nhất trong các lane đang giữ giá trị lớn nhất, còn đáp án của kernel CUDA của nó phụ thuộc vào lane
          nào giữ chỉ số nào. Vì vậy một bước greedy có thể chọn các token khác nhau trên các backend khác nhau của cùng một dự
          án. Quy tắc của M5, từ lab 10.3, là chỉ số thấp nhất ở mọi nơi.
        </p>
        }
      >
      <p>
        PyTorch documents the first maximal value and its comparator keeps the lower index. llama.cpp's Metal kernel
        takes the largest index among the lanes holding the maximum, and its CUDA kernel's answer depends on which lane
        held which index. A greedy step can therefore pick different tokens on different backends of one project. M5's
        rule, from lab 10.3, is the lowest index everywhere.
      </p>
      </Translated>
    ),
    source: 'notes/today/ch10-facts.md D2, D4', // i18n-ignore: citation
  },
  'x1/X1.2/q-softmax-share': {
    kind: 'choice',
    id: 'x1/X1.2/q-softmax-share',
    prompt: <Translated vi={<>Trong M5 với k = 40, phần việc softmax nào chiếm ưu thế trong một bước lấy mẫu với bộ từ vựng 151,936?</>}><>In M5 with k = 40, which softmax work dominates a sampling step at a vocabulary of 151,936?</></Translated>,
    options: [
      { label: <Translated vi="Không phần nào: softmax chỉ thấy 40 token còn lại; các penalty và top-k mới là thứ đọc bộ từ vựng.">None of it: softmax sees only the 40 survivors; the penalties and top-k are what read the vocabulary.</Translated>, correct: true },
      { label: <Translated vi="Softmax trên toàn bộ từ vựng, thứ M5 phải tính trước khi top-k có thể xếp hạng các token theo xác suất">The softmax over the whole vocabulary, which M5 has to compute before top-k can rank the tokens by probability</Translated>, why: <Translated vi="M5 không bao giờ tính nó. Top-k chạy trên logits và không cần xác suất.">M5 never computes one. Top-k runs on logits and needs no probabilities.</Translated> },
      { label: <Translated vi="Các lời gọi exp của softmax">The softmax's exp calls</Translated>, why: <Translated vi="Chỉ có 40 lời gọi mỗi hàng, so với một lượt đọc cỡ bộ từ vựng ở chỗ khác.">There are 40 of them per row, against a vocabulary-sized read elsewhere.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
          Top-k làm việc trên logits (thứ tự của chúng là tất cả những gì nó cần), nên chưa có xác suất nào cho đến khi có k
          ứng viên. Sau đó softmax, scan và lần rút mỗi phép chạm tới 40 số mỗi hàng. Phần việc cỡ bộ từ vựng là lượt penalty
          và phép chọn.
        </p>
        }
      >
      <p>
        Top-k works on logits (their order is all it needs), so no probability exists until there are k candidates. Then
        the softmax, the scan and the draw each touch 40 numbers per row. The vocabulary-sized work is the penalty pass
        and the selection.
      </p>
      </Translated>
    ),
    source: 'ours; the M5 order of section X1.1',
    vi: { source: 'của khóa học này; thứ tự M5 của mục X1.1' },
  },
};

// ---------------------------------------------------------------- the explorer
type Tie = 'lowest' | 'highest';
const TIE_LABEL: Record<Tie, string> = {
  get lowest() { return tr('lowest index (M5, PyTorch)', 'chỉ số thấp nhất (M5, PyTorch)'); },
  get highest() { return tr('highest index (llama.cpp Metal)', 'chỉ số cao nhất (llama.cpp Metal)'); },
};

function TemperatureExplorer() {
  const [rowId, setRowId] = useState('peaked');
  const [T, setT] = useState(0.5);
  const [view, setView] = useState<'probs' | 'logits'>('probs');
  const [float32, setFloat32] = useState(false);
  const [sel, setSel] = useState<number[]>([...OPENING_PAIR]);
  const [tie, setTie] = useState<Tie>('lowest');
  const [greedy, setGreedy] = useState<number | null>(null);
  const row = TEMP_ROWS.find((r) => r.id === rowId)!;
  const l = row.logits;
  const probs = useMemo(() => softmax(l, 1 / T, float32), [l, T, float32]);
  const scaled = useMemo(() => (float32 ? temperature(l, T, true, 'reciprocal') : l.map((v) => v / T)), [l, T, float32]);
  const mismatch = useMemo(() => (float32 ? reciprocalMismatch(l, T) : []), [l, T, float32]);
  const top = Math.max(...probs);
  const H = entropyBits(probs);
  const [a, b] = sel.length === 2 ? sel : [sel[0], sel[0]];
  const ratio = probs[b] > 0 ? probs[a] / probs[b] : Infinity;
  const theory = Math.exp((l[a] - l[b]) / T);
  const n = l.length;
  const W = 640;
  const X0 = 30;
  const bw = (W - X0 - 10) / n;
  const BASE = 190;
  const PH = 160;
  const maxAbs = Math.max(...scaled.map(Math.abs), 1e-9);
  const click = (i: number) => {
    setSel((s) => (s.includes(i) ? s : [s[s.length - 1], i]));
  };
  const pickRow = (id: string) => {
    setRowId(id);
    setGreedy(null);
    if (id === 'peaked') setSel([...OPENING_PAIR]);
    else if (id === 'tied') setSel([1, 4]);
    else setSel([7, 5]);
  };

  return (
    <Bench
      title={tr('Temperature', 'Temperature')}
      task={tr('Drag T down towards 0 and up to 4. Click any two bars to compare them. Then press greedy on each row.', 'Kéo T xuống gần 0 và lên tới 4. Nhấn vào hai thanh bất kỳ để so sánh chúng. Rồi nhấn greedy trên từng hàng.')}
      foot={
        <Translated
          vi={
            <>
              Xác suất được vẽ trên thang cố định từ 0 đến 1; logits được chia tỉ lệ theo giá trị lớn nhất. Float32 làm tròn
              mọi phép toán theo thứ tự của lab 10.3 (scale, trừ giá trị lớn nhất, exp, cộng, một nghịch đảo, một phép nhân).
            </>
          }
        >
        <>
          Probabilities are drawn on a fixed scale from 0 to 1; logits are scaled to the largest. Float32 rounds every
          operation in lab 10.3's order (scale, subtract the maximum, exp, sum, one reciprocal, one multiply).
        </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end', flexWrap: 'wrap' }}>
        <span style={FIELD}>
          <Translated vi="Hàng">Row</Translated>
          <span className="seg" role="group" aria-label={tr('Row', 'Hàng')}>
            {TEMP_ROWS.map((r) => (
              <button key={r.id} aria-pressed={rowId === r.id} onClick={() => pickRow(r.id)}>{r.label}</button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          <Translated vi="Hiển thị">Show</Translated>
          <span className="seg" role="group" aria-label={tr('Show', 'Hiển thị')}>
            <button aria-pressed={view === 'probs'} onClick={() => setView('probs')}>{tr('probabilities', 'xác suất')}</button>
            <button aria-pressed={view === 'logits'} onClick={() => setView('logits')}>{tr('logits / T', 'logits / T')}</button>
          </span>
        </span>
        <label className="check" style={{ alignSelf: 'flex-end' }}>
          <input type="checkbox" checked={float32} onChange={(e) => setFloat32(e.target.checked)} /> float32{/* i18n-ignore: type name */}
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end', flexWrap: 'wrap' }}>
        <label className="field" style={{ flex: '1 1 16rem' }}>
          T = {T < 1 ? T.toFixed(2) : T.toFixed(1)}
          <input type="range" min={0} max={1000} step={1} value={toSlider(T)} aria-label="T"
            onChange={(e) => { setT(tFrom(Number(e.target.value))); }} style={{ width: '100%' }} />
        </label>
        <span style={FIELD}>
          <Translated vi="Quy tắc hòa của greedy">Greedy tie rule</Translated>
          <span className="seg" role="group" aria-label={tr('Tie rule', 'Quy tắc hòa')}>
            {(['lowest', 'highest'] as Tie[]).map((t) => (
              <button key={t} aria-pressed={tie === t} onClick={() => { setTie(t); setGreedy(null); }}>{TIE_LABEL[t]}</button>
            ))}
          </span>
        </span>
        <button className="primary" onClick={() => setGreedy(argmax(l, tie))}>{tr('greedy', 'greedy')}</button>
      </div>

      <svg viewBox={`0 0 ${W} 222`} role="img" aria-label={tr("The row's probabilities at this temperature", 'Xác suất của hàng ở temperature này')} data-part="bars"
        style={{ width: '100%', maxWidth: W, minWidth: 420, display: 'block' }}>
        {view === 'probs'
          ? [0.25, 0.5, 0.75, 1].map((g) => (
              <g key={g}>
                <line x1={X0} y1={BASE - g * PH} x2={W - 10} y2={BASE - g * PH} stroke="var(--rule-soft)" />
                <text x={X0 - 4} y={BASE - g * PH + 4} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{g}</text>
              </g>
            ))
          : null}
        <line x1={X0} y1={view === 'probs' ? BASE : BASE - PH / 2} x2={W - 10} y2={view === 'probs' ? BASE : BASE - PH / 2} stroke="var(--rule)" />
        {l.map((_, i) => {
          const x = X0 + i * bw + 3;
          const picked = sel.includes(i);
          const won = greedy === i;
          let y: number;
          let h: number;
          if (view === 'probs') {
            h = Math.max(1, probs[i] * PH);
            y = BASE - h;
          } else {
            const v = (scaled[i] / maxAbs) * (PH / 2);
            h = Math.max(1, Math.abs(v));
            y = v >= 0 ? BASE - PH / 2 - h : BASE - PH / 2;
          }
          return (
            <g key={i} onClick={() => click(i)} style={{ cursor: 'pointer' }} data-token={i} data-prob={String(probs[i])}>
              <rect x={x} y={y} width={bw - 6} height={h} rx="2"
                fill={won ? 'var(--ink)' : 'var(--ink-3)'} fillOpacity={won ? 0.9 : 0.5}
                stroke={picked ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth={picked ? 2.4 : 1} />
              <rect x={x} y={20} width={bw - 6} height={BASE - 20} fill="transparent" />
              <text x={x + (bw - 6) / 2} y={BASE + 14} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{i}</text>
              {picked ? (
                <text x={x + (bw - 6) / 2} y={BASE + 27} fontSize="10.5" textAnchor="middle" fontWeight="700">{i === a ? 'a' : 'b'}</text>
              ) : null}
              <text x={x + (bw - 6) / 2} y={14} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
                {view === 'probs' ? (probs[i] >= 0.0005 ? probs[i].toFixed(3) : probs[i] === 0 ? '0' : '<.001') : scaled[i].toFixed(1)}
              </text>
            </g>
          );
        })}
      </svg>

      <table className="data" style={{ margin: '0.6rem 0 0', maxWidth: '40rem' }} data-part="readouts">
        <tbody>
          <tr data-readout="top1" data-top={top}><th><Translated vi="xác suất top-1">top-1 probability</Translated></th><td className="num">{top.toFixed(4)}</td></tr>
          <tr data-readout="entropy"><th><Translated vi="entropy">entropy</Translated></th><td className="num"><Translated vi={<>{H.toFixed(3)} bit <span className="muted">(tối đa {Math.log2(n).toFixed(3)})</span></>}>{H.toFixed(3)} bits <span className="muted">(of at most {Math.log2(n).toFixed(3)})</span></Translated></td></tr>
          <tr data-readout="ratio" data-ratio={ratio.toPrecision(6)}>
            <th>p(token {a}) / p(token {b}){/* i18n-ignore: formula */}</th>
            <td className="num">
              {isFinite(ratio) ? ratio.toPrecision(4) : tr('infinite', 'vô cùng')} <span className="muted">= exp(({l[a]} − {l[b]}) / {T}) = {theory.toPrecision(4)}</span>
            </td>
          </tr>
          {float32 ? (
            <tr data-readout="reciprocal" data-mismatch={mismatch.length}>
              <th><Translated vi="chia cho T so với nhân với 1/T">divide by T vs multiply by 1/T</Translated></th>
              <td>
                {mismatch.length === 0
                  ? tr(`the same float32 logits at T = ${T}`, `cùng các logit float32 tại T = ${T}`)
                  : tr(`${mismatch.length} of ${n} logits differ in the last bit (tokens ${mismatch.join(', ')})`, `${mismatch.length} trên ${n} logit khác nhau ở bit cuối (token ${mismatch.join(', ')})`)}
              </td>
            </tr>
          ) : null}
          {greedy != null ? (
            <tr data-readout="greedy" data-greedy={greedy}>
              <th><Translated vi={<>greedy, chỉ số {tie === 'lowest' ? 'thấp nhất' : 'cao nhất'} khi hòa</>}>greedy, {tie === 'lowest' ? 'lowest' : 'highest'} index on a tie</Translated></th>
              <td><Translated vi={<>token <b>{greedy}</b></>}>token <b>{greedy}</b></Translated>{argmax(l, 'lowest') !== argmax(l, 'highest') ? tr(' (the two tie rules disagree on this row)', ' (hai quy tắc hòa bất đồng trên hàng này)') : tr(' (both tie rules agree on this row)', ' (cả hai quy tắc hòa thống nhất trên hàng này)')}</td>
            </tr>
          ) : null}
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- the listing
function SoftmaxListing() {
  const { reports } = useReports();
  const STATUS = { cu: solutionTabStatus(reports, LAB, 'kernel.cu'), metal: solutionTabStatus(reports, LAB, 'kernel.metal') };
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  const metal = useLabFile(LAB + '/solution/kernel.metal');
  const k = cu ? functionSource(cu, 'softmaxRowsKernel') : null;
  const launch = cu ? functionSource(cu, 'launchSoftmaxRows') : null;
  const mk = metal ? functionSource(metal, 'softmax_rows') : null;
  if (!k || !launch || !mk)
    return <Translated vi={<p className="muted">Kernel được nạp từ <code>labs/{LAB}/solution/</code>.</p>}><p className="muted">The kernel is loaded from <code>labs/{LAB}/solution/</code>.</p></Translated>;
  const text = k.text + '\n\n' + launch.text;
  const scaleLines = (t: string) => t.split('\n').flatMap((ln, i) => (/\bscale\b/.test(ln) && !/^\s*\/\//.test(ln) ? [i + 1] : []));
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: text, tiers: TIERS, status: STATUS.cu, highlight: scaleLines(text) },
        { label: tr('Metal kernel', 'Kernel Metal'), code: mk.text, tiers: TIERS, status: STATUS.metal, highlight: scaleLines(mk.text) },
      ]}
      caption={
        <Translated
          vi={
            <>
              Softmax của lab 10.3 (<code>labs/{LAB}/solution/kernel.cu</code>, dòng {k.firstLine} đến {launch.lastLine}, và{' '}
              <code>kernel.metal</code>). Các dòng được đánh dấu mang <code>scale</code>: truyền 1/T là xong temperature.
            </>
          }
        >
        <>
          Lab 10.3's softmax (<code>labs/{LAB}/solution/kernel.cu</code>, lines {k.firstLine} to {launch.lastLine}, and{' '}
          <code>kernel.metal</code>). The marked lines carry <code>scale</code>: pass 1/T and temperature is done.
        </>
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- the section
export function SX12() {
  const q = questionsSX12;
  return (
    <>
      <Translated
        vi={
          <p>
          Temperature là tham số lấy mẫu duy nhất mà mọi API đều có. Nó chia logits cho T trước softmax: dưới 1 nó làm phân
          phối nhọn về phía token được ưa chuộng, trên 1 nó làm phẳng phân phối, và khi T tiến về 0 nó trở thành greedy
          decoding. Mục này gộp nó vào softmax bạn đã viết, và xác định rõ greedy trả về gì khi hai logit bằng nhau, đó là chỗ
          các backend lặng lẽ bất đồng.
        </p>
        }
      >
      <p>
        Temperature is the one sampling parameter every API exposes. It divides the logits by T before the softmax: below
        1 it sharpens the distribution towards the favourite, above 1 it flattens it, and as T goes to 0 it becomes greedy
        decoding. This section folds it into the softmax you already wrote, and pins down what greedy returns when two
        logits are equal, which is where backends quietly disagree.
      </p>
      </Translated>
      <NoteGroup>
        <Beyond title={tr("Ours, not the book's", 'Của khóa học này, không phải của sách')}>
          <Translated
            vi={
              <p>
              Temperature và greedy decoding không có trong sách. Softmax có trừ giá trị lớn nhất là Eq. 20.2 của sách, và một
              khối cho mỗi hàng là Fig. 20.4 của sách. Các kernel của M4 là của lab 10.3: phép rút gọn warp hai tầng của §10.8
              (p. 243) và vòng lặp gộp việc cho luồng của §10.10 (p. 247).
            </p>
            }
          >
          <p>
            Temperature and greedy decoding are not in the book. The softmax with the maximum subtracted is the book's Eq.
            20.2, and one block per row is its Fig. 20.4. M4's kernels are lab 10.3's: the two-stage warp reduction of §10.8
            (p. 243) and the coarsening loop of §10.10 (p. 247).
          </p>
          </Translated>
        </Beyond>
        <Source section="20.2" pages="482–486" />
        <Source section="20.3" pages="486–488" />
      </NoteGroup>

      <h2><Translated vi="Chia cho T làm gì">What dividing by T does</Translated></h2>
      <Translated
        vi={
          <p>
          Viết softmax của l / T cho hai token a và b rồi chia cái này cho cái kia: tổng triệt tiêu, và cái còn lại là
          p<sub>a</sub> / p<sub>b</sub> = exp((l<sub>a</sub> − l<sub>b</sub>) / T). Chênh lệch một logit nghĩa là gấp e lần
          tại T = 1. Temperature đổi tỉ lệ các chênh lệch, và xác suất theo sau theo hàm mũ.
        </p>
        }
      >
      <p>
        Write the softmax of l / T for two tokens a and b and divide one by the other: the sum cancels, and what is left is
        p<sub>a</sub> / p<sub>b</sub> = exp((l<sub>a</sub> − l<sub>b</sub>) / T). A gap of one logit means a factor of e at
        T = 1. Temperature rescales the gaps, and the probabilities follow exponentially.
      </p>
      </Translated>
      <Predict
        id="x1/X1.2/p-ratio"
        prompt={<Translated vi={<>Logit của hai token chênh nhau 1.0. Tại T = 0.5, tỉ số xác suất của chúng là bao nhiêu?</>}><>Two tokens' logits differ by 1.0. At T = 0.5, what is the ratio of their probabilities?</></Translated>}
      >
        {(c) => (
          <div data-reveal="ratio">
            <Translated
              vi={
                <p>
                  exp(1 / 0.5) = e² = <b>{Math.exp(2).toFixed(3)}</b>
                  {Math.abs(c - Math.exp(2)) < 0.01 ? ', đúng như bạn nói' : ''}. Giảm T một nửa làm gấp đôi mọi chênh lệch
                  trong số mũ, nên tỉ số được bình phương: e tại T = 1 thành e² tại T = 0.5. Trình khám phá mở ra đúng cặp
                  này, token{' '}{OPENING_PAIR[0]} và {OPENING_PAIR[1]} của hàng nhọn.
                </p>
              }
            >
            <p>
              exp(1 / 0.5) = e² = <b>{Math.exp(2).toFixed(3)}</b>
              {Math.abs(c - Math.exp(2)) < 0.01 ? ', as you said' : ''}. Halving T doubles every gap in the exponent, so
              the ratio is squared: e at T = 1 becomes e² at T = 0.5. The explorer opens on exactly this pair, tokens{' '}
              {OPENING_PAIR[0]} and {OPENING_PAIR[1]} of the peaked row.
            </p>
            </Translated>
          </div>
        )}
      </Predict>
      <Translated
        vi={
          <p>
          Trình khám phá hiển thị một hàng mười hai token tại bất kỳ T nào từ 0.05 đến 4. Theo dõi xác suất top-1 và entropy
          khi bạn kéo, và kiểm tra tỉ số của hai thanh bất kỳ với công thức.
        </p>
        }
      >
      <p>
        The explorer shows one twelve-token row at any T from 0.05 to 4. Watch the top-1 probability and the entropy as you
        drag, and check the ratio of any two bars against the formula.
      </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
            Một hàng 12 token. Độ chính xác double trừ khi bật float32; khi đó mọi phép toán được làm tròn theo thứ tự của lab
            10.3, với exp của JavaScript thay cho exp của thiết bị (<code>expf</code> của CUDA sai số trong phạm vi 2 ulp, <code>exp</code>{' '}
            chính xác của Metal trong 4 và bản nhanh thì lỏng hơn; notes/today/ch10-facts.md A11, C4). Không có fused
            multiply-add.
          </p>
          }
        >
        <p>
          One row of 12 tokens. Double precision unless float32 is on; then every operation is rounded in lab 10.3's order,
          with JavaScript's exp standing in for the device's (CUDA's <code>expf</code> is within 2 ulp, Metal's precise{' '}
          <code>exp</code> within 4 and its fast one less tightly; notes/today/ch10-facts.md A11, C4). No fused
          multiply-add.
        </p>
        </Translated>
      </Simplification>
      <TemperatureExplorer />
      <Translated
        vi={
          <p>
          Điều cần để ý. Thanh lớn nhất không bao giờ đổi khi T di chuyển: chia cho một số dương giữ nguyên thứ tự. Khi T tiến
          về 0, hàng nhọn trở thành one-hot, còn hàng có hai giá trị lớn nhất hòa nhau chia đều giữa chúng, nên riêng giới hạn
          không gọi tên được một token. Tỉ số của hai thanh bất kỳ luôn là exp(hiệu / T). Và hai quy tắc hòa của greedy chỉ cho
          token khác nhau trên hàng có hòa.
        </p>
        }
      >
      <p>
        What to notice. The largest bar never changes as T moves: dividing by a positive number keeps the order. Towards T =
        0 the peaked row becomes one-hot, and the row with two tied maxima splits evenly between them, so the limit alone
        does not name a token. The ratio of any two bars is always exp(difference / T). And the two greedy tie rules give
        different tokens on the tied row only.
      </p>
      </Translated>
      <Ask q={q['x1/X1.2/q-tie-limit']} />

      <h2><Translated vi="Temperature không tốn gì khi được gộp">Temperature costs nothing when it is fused</Translated></h2>
      <Translated
        vi={
          <p>
          <code>launchSoftmaxRows</code> của lab 10.3 đã nhận một <code>scale</code>: nó tính softmax(scale · x). Truyền 1/T
          là temperature được thực hiện bên trong một kernel vốn đọc hàng đó rồi. Một kernel temperature riêng sẽ đọc và ghi
          mỗi logit thêm một lần, 8 byte mỗi phần tử trên một kernel vốn đã bị giới hạn bởi bộ nhớ.
        </p>
        }
      >
      <p>
        Lab 10.3's <code>launchSoftmaxRows</code> already takes a <code>scale</code>: it computes softmax(scale · x). Pass
        1/T and temperature is done inside a kernel that reads the row anyway. A separate temperature kernel would read and
        write every logit once more, 8 bytes per element on a kernel that is already memory-bound.
      </p>
      </Translated>
      <SoftmaxListing />
      <Ask q={q['x1/X1.2/q-fold-bytes']} />
      <Translated
        vi={
          <p>
          Trong M5 temperature còn tiến thêm một bước. Nó chạy cuối cùng, nên nó chỉ thấy các token còn lại: kernel rút chia
          tỉ lệ k giá trị, không phải cả bộ từ vựng (mục <a href={href('x1', 'X1.1')}>X1.1</a> có thứ tự).
        </p>
        }
      >
      <p>
        In M5 temperature goes one step further. It runs last, so it only ever sees the survivors: the draw kernel scales
        k values, not the vocabulary (section <a href={href('x1', 'X1.1')}>X1.1</a> has the order).
      </p>
      </Translated>

      <h2><Translated vi="Chia, hay nhân với nghịch đảo?">Divide, or multiply by the reciprocal?</Translated></h2>
      <Translated
        vi={
          <p>
          Sampler phía host của llama.cpp chia mỗi logit cho T; đường backend của nó nhân với 1/T tính một lần. Hai cách không
          giống nhau trong dấu phẩy động: 1/T được làm tròn, rồi tích lại được làm tròn lần nữa. Bật float32 trong trình khám
          phá và thử T = 3, rồi T = 2: tại lũy thừa của hai nghịch đảo là chính xác và không gì thay đổi.
        </p>
        }
      >
      <p>
        llama.cpp's host sampler divides each logit by T; its backend path multiplies by 1/T computed once. The two are
        not the same in floating point: 1/T is rounded, then the product is rounded again. Turn on float32 in the explorer
        and try T = 3, then T = 2: at a power of two the reciprocal is exact and nothing changes.
      </p>
      </Translated>
      <Today checked={tr(`2026-11-20 in the facts pass (${FACTS}, D2)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${FACTS}, D2)`)}>
        <Translated
          vi={
            <p>
            Host: <code>cur_p-&gt;data[i].logit /= temp;</code>. Backend:{' '}
            <code>ggml_scale(ctx, data-&gt;logits, 1.0f / temp)</code>. Việc hai cách có thể khác nhau ở bit cuối là ghi chú
            của chúng tôi.
          </p>
          }
        >
        <p>
          Host: <code>cur_p-&gt;data[i].logit /= temp;</code>. Backend:{' '}
          <code>ggml_scale(ctx, data-&gt;logits, 1.0f / temp)</code>. That the two can differ in the last bit is our note.
        </p>
        </Translated>
      </Today>
      <Ask q={q['x1/X1.2/q-reciprocal']} />

      <h2><Translated vi="T → 0 là greedy, và greedy cần một quy tắc phân định hòa">T → 0 is greedy, and greedy needs a tie rule</Translated></h2>
      <Translated
        vi={
          <p>
          Greedy decoding lấy logit lớn nhất: đó là <code>launchArgmaxRows</code> của lab 10.3, mốc dự án M4, hòa thì chọn chỉ
          số thấp nhất. Cùng bước đó trong mã khác không phải lúc nào cũng thống nhất khi hòa. Tài liệu PyTorch ghi giá trị
          lớn nhất đầu tiên, và bước kết hợp của MLX giữ chỉ số thấp hơn. Greedy sampler phía host của llama.cpp lấy chỉ số
          thấp nhất, argmax Metal của nó lấy chỉ số cao nhất trong các lane đang giữ giá trị lớn nhất, còn argmax CUDA của nó
          lấy bất cứ thứ gì thứ tự của các lane cho ra. transformers từ chối T = 0 và làm greedy qua{' '}
          <code>do_sample=False</code>; temp 0 của mlx-lm trả về argmax. Quy tắc của M5 là chỉ số thấp nhất ở mọi nơi, vì lab
          10.3 đã kiểm tra nó với các trường hợp hòa cài sẵn.
        </p>
        }
      >
      <p>
        Greedy decoding takes the largest logit: it is lab 10.3's <code>launchArgmaxRows</code>, milestone M4, with ties to
        the lowest index. The same step in other code does not always agree on ties. PyTorch documents the first maximum,
        and MLX's combine keeps the lower index. llama.cpp's host greedy sampler takes the lowest index, its Metal argmax
        the highest index among the lanes holding the maximum, and its CUDA argmax whatever the lanes' order gives.
        transformers refuses T = 0 and does greedy through <code>do_sample=False</code>; mlx-lm's temp 0 returns the
        argmax. M5's rule is the lowest index everywhere, because lab 10.3 already tests it with planted ties.
      </p>
      </Translated>
      <Today checked={tr(`2026-11-20 in the facts pass (notes/today/ch10-facts.md D2, D3, D4; ${FACTS}, D1, D4, D8)`, `2026-11-20 trong lượt kiểm tra dữ kiện (notes/today/ch10-facts.md D2, D3, D4; ${FACTS}, D1, D4, D8)`)}>
        <Translated
          vi={
            <p>
            <code>torch.argmax</code>: "the indices of the first maximal value" (chỉ số của giá trị lớn nhất đầu tiên).
            llama.cpp Metal:{' '}<code>simd_max(select(-1, larg, lmax == max_val))</code>. llama.cpp CUDA: một cặp chỉ bị thay
            thế{' '}<code>if (val &gt; maxval)</code>, nên các trường hợp hòa phụ thuộc vào lane.
          </p>
          }
        >
        <p>
          <code>torch.argmax</code>: "the indices of the first maximal value". llama.cpp Metal:{' '}
          <code>simd_max(select(-1, larg, lmax == max_val))</code>. llama.cpp CUDA: a pair is replaced only{' '}
          <code>if (val &gt; maxval)</code>, so ties depend on lanes.
        </p>
        </Translated>
      </Today>
      <Ask q={q['x1/X1.2/q-greedy-ties']} />

      <h2><Translated vi="Softmax trên các ứng viên, không phải trên bộ từ vựng">Softmax over candidates, not the vocabulary</Translated></h2>
      <Translated
        vi={
          <p>
          Trong M5 softmax không bao giờ thấy 151,936 logit. Top-k chạy trước, trên logits, và chỉ cần thứ tự của chúng; sau
          đó softmax chạy trên k giá trị mỗi hàng. Với k = 40, một khối 256 luồng của lab 10.3 có hầu hết luồng rảnh trong
          softmax đó, và điều đó ổn: nó là phần nhỏ của một bước, thứ mà lab X1.3 cho bạn đo trên thiết bị của mình.
        </p>
        }
      >
      <p>
        In M5 the softmax never sees 151,936 logits. Top-k runs first, on logits, and needs only their order; the softmax
        then runs on k values per row. With k = 40, a block of lab 10.3's 256 threads has most of its threads idle in
        that softmax, and that is fine: it is a small share of the step, which lab X1.3 lets you measure on your device.
      </p>
      </Translated>
      <Ask q={q['x1/X1.2/q-softmax-share']} />

      <h2><Translated vi="Trên HIP và Metal">On HIP and Metal</Translated></h2>
      <Translated
        vi={
          <p>
          Không có gì mới ngoài lab 10.3. Trên HIP phép rút gọn warp nhận một lane mask 64-bit có kiểu và đọc{' '}
          <code>warpSize</code>; trên Metal độ rộng SIMD group lấy từ <code>threadExecutionWidth</code> của pipeline. Chế độ
          toán học mặc định của Metal không chặt chẽ theo IEEE, nên một softmax tính trên Metal được so với tham chiếu CPU
          trong một dung sai, không bao giờ từng bit.
        </p>
        }
      >
      <p>
        Nothing new beyond lab 10.3. On HIP the warp reduction takes a typed 64-bit lane mask and reads{' '}
        <code>warpSize</code>; on Metal the SIMD-group width comes from the pipeline's <code>threadExecutionWidth</code>.
        Metal's default math mode is not IEEE-strict, so a softmax computed on Metal is compared with the CPU reference
        within a tolerance, never bit for bit.
      </p>
      </Translated>
      <NoteGroup>
        <Today checked={tr('2026-11-20 in the facts pass (notes/today/ch05-facts.md C7; notes/today/infra-facts.md)', '2026-11-20 trong lượt kiểm tra dữ kiện (notes/today/ch05-facts.md C7; notes/today/infra-facts.md)')}>
          <Translated
            vi={
              <p>
              <code>fastMathEnabled</code> mặc định là YES và đã bị deprecated, thay bằng <code>mathMode</code>; các nguồn của
              Apple bất đồng về chế độ mặc định trên Apple silicon. Lane mask của HIP là 64-bit trên mọi target.
            </p>
            }
          >
          <p>
            <code>fastMathEnabled</code> defaults to YES and is deprecated in favour of <code>mathMode</code>; Apple's
            sources disagree on the default mode on Apple silicon. HIP lane masks are 64-bit on every target.
          </p>
          </Translated>
        </Today>
        <Aside>
          <Translated
            vi={
              <p>
              Dung sai mà lab 10.3 dùng, và lý do, nằm trong README của nó, được suy ra từ cận sai số exp được tài liệu hóa của
              từng backend tại hàng dài nhất mà nó kiểm tra.
            </p>
            }
          >
          <p>
            The tolerance lab 10.3 uses, and why, is in its README, derived from each backend's documented exp error
            bound at the longest row it tests.
          </p>
          </Translated>
        </Aside>
      </NoteGroup>
    </>
  );
}
