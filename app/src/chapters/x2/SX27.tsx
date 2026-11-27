import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { fmt, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { gpuBackends, measuredDevices, thresholdOf, type Device, type ReportsLike, type StepShape } from '../../sim/batching';
import { MAT_FORMAT_LABEL, type MatFormat } from '../../sim/quant';
import {
  BOOK_DEVICE,
  M8_DEFAULT,
  STEP_FORMATS,
  STEP_FORMAT_LABEL,
  TOY,
  bandwidthUtilisation,
  greedyRun,
  matrixWeightBytes,
  measuredSpeedup,
  measuredSteps,
  predictedSpeedup,
  stepBytes,
  toyModel,
  toyWeightBytes,
  type StepFormat,
  type StepFormats,
  type StepBytes,
} from '../../sim/qstep';

const FACTS = 'notes/today/x2-facts.md';
const CHECKED = (items: string) => tr(`2026-11-20 in the facts pass (${FACTS}, ${items})`, `2026-11-20 trong lượt kiểm tra dữ kiện (${FACTS}, ${items})`);
const MB = (b: number) => `${(b / 1e6).toFixed(1)} MB`; // i18n-ignore: a unit
const fmtTime = (s: number) => (s < 1e-3 ? `${fmt(s * 1e6)} µs` : `${fmt(s * 1e3)} ms`); // i18n-ignore: units
const F32F: StepFormats = { layers: 'f32', embed: 'f32' };
const ALLQ4: StepFormats = { layers: 'q4_0', embed: 'q4_0' };
const LAYERSQ4: StepFormats = { layers: 'q4_0', embed: 'f32' };

// ---------------------------------------------------------------- questions
const ratio = (f: StepFormats) => matrixWeightBytes(M8_DEFAULT, F32F) / matrixWeightBytes(M8_DEFAULT, f);
export const questionsSX27: Record<string, Question> = {
  'x2/X2.7/q-step-ratio': {
    kind: 'numeric',
    id: 'x2/X2.7/q-step-ratio',
    prompt: (
      <Translated
        vi={
          <>
            Bước mặc định của lab 22.2: 2 lớp 4096 × 4096 và một LM head 32,000 × 4096, tất cả đều float32. Chỉ lượng tử
            hóa hai lớp sang Q4_0 và giữ head ở float32. Số byte trọng số mỗi bước giảm đi bao nhiêu lần? (Hai chữ số thập
            phân.)
          </>
        }
      >
        <>
          Lab 22.2's default step: 2 layers of 4096 × 4096 and an LM head of 32,000 × 4096, all float32. Quantise only the
          two layers to Q4_0 and keep the head in float32. By how many times do the weight bytes per step shrink? (Two
          decimals.)
        </>
      </Translated>
    ),
    answer: Number(ratio(LAYERSQ4).toFixed(2)),
    tol: 0.01,
    unit: '×',
    explain: (
      <Translated
        vi={
          <p>
            Float32: 4 × (2 × 4096² + 32,000 × 4096) = {MB(matrixWeightBytes(M8_DEFAULT, F32F))}. Các lớp ở Q4_0 với 0.5625
            byte mỗi trọng số: 18.9 MB, cộng 524.3 MB của head = {MB(matrixWeightBytes(M8_DEFAULT, LAYERSQ4))}. Tỉ lệ{' '}
            {ratio(LAYERSQ4).toFixed(2)}. Lượng tử hóa cả head nữa thì tỉ lệ là {ratio(ALLQ4).toFixed(2)}: head chiếm 131 triệu
            trong 165 triệu trọng số.
          </p>
        }
      >
        <p>
          Float32: 4 × (2 × 4096² + 32,000 × 4096) = {MB(matrixWeightBytes(M8_DEFAULT, F32F))}. Layers in Q4_0 at 0.5625
          bytes per weight: 18.9 MB, plus the head's 524.3 MB = {MB(matrixWeightBytes(M8_DEFAULT, LAYERSQ4))}. Ratio{' '}
          {ratio(LAYERSQ4).toFixed(2)}. Quantise the head too and it is {ratio(ALLQ4).toFixed(2)}: the head is 131 million of
          the 165 million weights.
        </p>
      </Translated>
    ),
    source: `ours: ${FACTS} E1 bytes per weight; lab 22.2's default shape (labs/ch22/lab2_batching)`,
    vi: { source: `của khóa học này: ${FACTS} E1 về số byte mỗi trọng số; kích thước mặc định của lab 22.2 (labs/ch22/lab2_batching)` },
  },
  'x2/X2.7/q-embed': {
    kind: 'choice',
    id: 'x2/X2.7/q-embed',
    prompt: (
      <Translated vi={<>Khi bảng embedding E đã được lượng tử hóa, vì sao bước decode cần một phép gather có giải lượng tử hóa?</>}>
        <>Once the embedding table E is quantised, why does the step need a gather that dequantises?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="E được dùng chung với head, và một bản sao float riêng sẽ tốn đúng số byte vừa tiết kiệm được.">
            E is tied to the head, and a separate float copy would cost the bytes just saved.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Gather là phần chậm nhất của bước, nên nó phải đọc ít byte hơn.">
            The gather is the slowest piece of the step, so it must read fewer bytes.
          </Translated>
        ),
        why: (
          <Translated vi="Ở batch 1 nó đọc một hàng của E: khoảng 16 KB ở float32, không đáng kể so với head.">
            At batch 1 it reads one row of E: about 16 KB in float32, nothing next to the head.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Embedding phải giữ float32 cho softmax, nên gather chuyển đổi chúng.">
            Embeddings must stay float32 for the softmax, so the gather converts them.
          </Translated>
        ),
        why: (
          <Translated vi="Gather cấp dữ liệu cho lớp đầu tiên, không phải cho softmax; và đầu ra của nó đằng nào cũng là float32.">
            The gather feeds the first layer, not the softmax; and its output is float32 either way.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong lab 22.2, LM head nhân với chính E. Lượng tử hóa E cho head thì gather cũng phải đọc đúng những byte đó:
            một hàng khối cho mỗi chuỗi, giải lượng tử hóa vào H. Một bản sao float thứ hai của E sẽ là 524 MB ở kích thước
            mặc định, tức phần lớn những gì lượng tử hóa đã tiết kiệm được.
          </p>
        }
      >
        <p>
          In lab 22.2 the LM head multiplies by E itself. Quantise E for the head and the same bytes are what the gather must
          read: one row of blocks per sequence, dequantised into H. A second, float copy of E would be 524 MB at the
          default shape, which is most of what quantising saved.
        </p>
      </Translated>
    ),
    source: 'ours; lab 22.2’s tied LM head (labs/ch22/lab2_batching/decode.h)',
    vi: { source: 'của khóa học này; LM head dùng chung trọng số của lab 22.2 (labs/ch22/lab2_batching/decode.h)' },
  },
  'x2/X2.7/q-margin': {
    kind: 'choice',
    id: 'x2/X2.7/q-margin',
    prompt: (
      <Translated vi={<>Những bước greedy nào có nguy cơ khi trọng số bị lượng tử hóa?</>}>
        <>Which greedy steps are at risk when the weights are quantised?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Những bước có khoảng chênh top-2 ở float32 nhỏ hơn mức thay đổi của logits.">
            Steps whose float32 top-2 margin is below the logits' change.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Những bước đầu của chuỗi, nơi sai số lượng tử hóa lớn nhất.">
            The first steps of the sequence, where the quantisation error is largest.
          </Translated>
        ),
        why: (
          <Translated vi="Ở đây mỗi bước bắt đầu từ một token, không từ trạng thái tích lũy; nguy cơ tính theo từng bước.">
            Each step here starts from a token, not from accumulated state; the risk is per step.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Những bước có token thắng mang logit lớn nhất của chuỗi.">
            The steps whose winning token has the largest logit of the sequence.
          </Translated>
        ),
        why: (
          <Translated vi="Logit thắng lớn vẫn ổn nếu token về nhì bị bỏ xa.">
            A large winning logit is fine if the runner-up is far behind.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Tất cả như nhau, vì mọi logit đều dịch đi khoảng cùng một lượng.">
            All of them equally, since every logit moves by about the same amount.
          </Translated>
        ),
        why: (
          <Translated vi="Trong mô hình đồ chơi, một bước có khoảng chênh gấp vài lần sai số logits không bao giờ đổi token.">
            In the toy model a step whose margin is several times the logits error never flips.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Decode greedy giữ argmax. Lượng tử hóa làm mọi logit dịch đi một chút; nếu khoảng dẫn của token thắng so với một
            đối thủ nhỏ hơn mức hai logit đó dịch tương đối với nhau, đối thủ sẽ thắng. Những bước có khoảng chênh rộng thì an
            toàn dù sai số thế nào. Trong mô hình đồ chơi, hãy xem cột nào màu đỏ.
          </p>
        }
      >
        <p>
          Greedy decoding keeps the argmax. Quantisation moves every logit a little; if the winner's lead over some rival
          is smaller than how far the two move relative to each other, the rival wins. Steps with a wide margin are safe
          whatever the error. In the toy model, look at which bars are red.
        </p>
      </Translated>
    ),
    source: 'ours (toy model in app/src/sim/qstep.ts)',
    vi: { source: 'của khóa học này (mô hình đồ chơi trong app/src/sim/qstep.ts)' },
  },
  'x2/X2.7/q-weights-vs-output': {
    kind: 'choice',
    id: 'x2/X2.7/q-weights-vs-output',
    prompt: (
      <Translated vi={<>Q4_0 có sai số RMS tương đối khoảng 9% trên trọng số phân phối chuẩn. Từ đó bạn có dự đoán được sai số của logits không?</>}>
        <>Q4_0 has about 9% relative RMS error on normal weights. Can you predict the error in the logits from that?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không: sai số trộn lẫn qua mọi lớp và qua residual.">
            No: the error mixes through every layer and the residual.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Có: cũng khoảng 9%, vì logits tuyến tính theo trọng số.">
            Yes: also about 9%, because the logits are linear in the weights.
          </Translated>
        ),
        why: (
          <Translated vi="Logits không tuyến tính theo trọng số: mỗi lớp cấp cho lớp sau, và E xuất hiện hai lần, ở gather và ở head.">
            They are not linear in the weights: each layer feeds the next, and E appears twice, at the gather and at the head.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Có: khoảng 9% nhân với số lớp mà sai số đi qua.">
            Yes: about 9% times the number of layers the error passes through.
          </Translated>
        ),
        why: (
          <Translated vi="Sai số không cộng dồn như vậy; một phần triệt tiêu nhau, và residual mang phần chưa lượng tử hóa đi tiếp.">
            Errors do not add like that; some cancel, and the residual carries the unquantised part forward.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong mô hình đồ chơi, sai số của trọng số khoảng 9% còn sai số của logits mỗi bước thì lớn hơn, và khác nhau
            giữa các bước. Độ nhạy của một mô hình thật lại khác nữa. Vì thế các khẳng định của chính PyTorch và llama.cpp
            nói về đầu ra (độ chính xác, perplexity), và vì thế lab X2.4 kiểm tra logits và token, không kiểm tra trọng số.
          </p>
        }
      >
        <p>
          In the toy model the weights' error is about 9% and the logits' error per step comes out larger, and it differs
          from step to step. A real model's sensitivity differs again. That is why PyTorch's and llama.cpp's own claims are
          about outputs (accuracy, perplexity), and why lab X2.4 checks logits and tokens, not weights.
        </p>
      </Translated>
    ),
    source: `ours; ${FACTS} E2 for the 9%`,
    vi: { source: `của khóa học này; ${FACTS} E2 cho con số 9%` },
  },
  'x2/X2.7/q-batch8': {
    kind: 'choice',
    id: 'x2/X2.7/q-batch8',
    prompt: (
      <Translated
        vi={
          <>
            Ở batch 8, mỗi lần đọc trọng số phục vụ 8 hàng. Trong mô hình byte, mức tăng tốc dự đoán của Q4_0 so với
            float32 tăng hay giảm từ batch 1 lên batch 8, và vì sao vẫn phải đo?
          </>
        }
      >
        <>
          At batch 8 every weight read serves 8 rows. In the bytes model, does Q4_0's predicted speed-up over float32 grow
          or shrink from batch 1 to batch 8, and why measure it anyway?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Giảm: các byte khác tăng theo batch, và head có thể chuyển sang bị giới hạn bởi tính toán.">
            It shrinks: other bytes grow with the batch, and the head can turn compute-bound.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Tăng: công giải lượng tử hóa được 8 hàng chia nhau, nên mỗi hàng tốn ít hơn.">
            It grows: the dequantise work is shared by 8 rows, so each row costs less.
          </Translated>
        ),
        why: (
          <Translated vi="Mô hình hoàn toàn không có công giải lượng tử hóa; và chia sẻ nó chỉ cắt bớt một chi phí, không nâng tỉ lệ byte.">
            The model has no dequantise work at all; and sharing it would only cut a cost, not raise the byte ratio.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Giữ nguyên ở tỉ lệ byte, 7.11, vì byte trọng số vẫn chiếm phần lớn.">
            It stays at the byte ratio, 7.11, because weight bytes still dominate.
          </Translated>
        ),
        why: (
          <Translated vi="Chỉ byte trọng số giảm 7.11 lần; phần còn lại của bước không giảm chút nào, và ở batch 8 nó không còn nhỏ nữa.">
            Only the weight bytes shrink 7.11 times; the rest of the step does not shrink at all, and at batch 8 it is no longer small.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Giảm, vì hệ số tỉ lệ của Q4_0 được đọc lại cho mỗi hàng của batch.">
            It shrinks, because Q4_0's scales are read again for each row of the batch.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi lần đọc trọng số, kể cả hệ số tỉ lệ, phục vụ cả 8 hàng: hệ số tỉ lệ được đọc một lần mỗi bước, không phải một lần mỗi hàng.">
            Each weight read, scales included, serves all 8 rows: the scales are read once per step, not once per row.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Ở batch 1, bước gần như toàn là byte trọng số, nên tỉ lệ gần 7.1. Ở batch 8, logits, softmax và argmax di chuyển
            gấp 8 lần số byte, và trên thiết bị ví dụ của sách, một GEMV Q4_0 ở batch 8 làm 2 × 8 / 0.5625 ≈ 28 phép toán
            trên mỗi byte trọng số, vượt ngưỡng 20 của nó: trong mô hình, head trở nên bị giới hạn bởi tính toán. Kernel thật
            còn tốn phép tính để giải nén và nhân hệ số tỉ lệ, điều mà mô hình bỏ qua, và chính điều đó quyết định phần còn lại. Chỉ
            phép đo mới cho biết bao nhiêu phần của tỉ lệ còn giữ được.
          </p>
        }
      >
        <p>
          At batch 1 the step is nearly all weight bytes, so the ratio is near 7.1. At batch 8 the logits, softmax and
          argmax move 8 times as many bytes, and on the book's example device a Q4_0 GEMV at batch 8 does 2 × 8 / 0.5625 ≈
          28 operations per byte of weights, above its threshold of 20: the head becomes compute-bound in the model. Real
          kernels also spend arithmetic unpacking and scaling, which the model leaves out, and that decides the rest. Only a
          measurement says how much of the ratio survives.
        </p>
      </Translated>
    ),
    source: `ours; ${FACTS} E1, A1; the book's example device from ch. 5 p. 94`,
    vi: { source: `của khóa học này; ${FACTS} E1, A1; thiết bị ví dụ của sách từ ch. 5 p. 94` },
  },
};

// ---------------------------------------------------------------- explorer: where the step's bytes go
interface DevOption { key: string; label: string; dev: Device; backend?: string }
const BOOK_OPTION: DevOption = {
  key: 'book',
  get label() {
    return tr("The book's example (H100)", 'Ví dụ của sách (H100)');
  },
  dev: BOOK_DEVICE,
};

function useDevices() {
  const { reports } = useReports();
  const rep = reports as unknown as ReportsLike;
  const options: DevOption[] = useMemo(
    () => [BOOK_OPTION, ...measuredDevices(rep).map((m) => ({
      key: m.backend, backend: m.backend,
      get label() {
        return tr(`Yours: ${m.name} (${m.backend}, lab 5.1)`, `Của bạn: ${m.name} (${m.backend}, lab 5.1)`);
      },
      dev: { peak: m.peak, bw: m.bw },
    }))],
    [rep],
  );
  return { rep, options };
}

function StackedBar({ s, max, label }: { s: StepBytes; max: number; label: string }) {
  const Wd = 600;
  let x = 0;
  return (
    <div style={{ marginBottom: '0.5rem' }}>
      <div className="muted" style={{ marginBottom: '0.15rem' }}>
        <Translated vi={<>{label}: {MB(s.bytes)}, trong đó {MB(s.weightBytes)} là trọng số</>}>
          {label}: {MB(s.bytes)}, {MB(s.weightBytes)} of it weights
        </Translated>
      </div>
      <svg viewBox={`0 0 ${Wd} 22`} role="img" aria-label={tr(`Bytes per piece, ${label}`, `Số byte theo từng phần, ${label}`)} style={{ width: '100%', display: 'block' }} data-bar={label} data-bytes={s.bytes}>
        {s.pieces.map((p) => {
          const w = (p.weightBytes / max) * Wd;
          const o = ((p.bytes - p.weightBytes) / max) * Wd;
          const g = (
            <g key={p.key} data-piece={p.key}>
              <rect x={x} y={2} width={w} height={18} fill="var(--global)" />
              <rect x={x + w} y={2} width={o} height={18} fill="var(--global-wash)" stroke={o > 1 ? 'var(--global)' : 'none'} strokeWidth={0.5} />
              {w + o > 50 ? <text x={x + 4} y={15} fontSize="10" style={{ fill: w > 40 ? 'var(--page)' : 'var(--ink)' }}>{p.label}</text> : null}
            </g>
          );
          x += w + o;
          return g;
        })}
      </svg>
    </div>
  );
}

function StepExplorer() {
  const { rep, options } = useDevices();
  const [devKey, setDevKey] = useState('book');
  const [custom, setCustom] = useState<Device>({ peak: 10000, bw: 400 });
  const [shapeKey, setShapeKey] = useState<'m8' | 'custom'>('m8');
  const [cshape, setCshape] = useState<StepShape>({ kind: 'step', d: 2048, layers: 8, vocab: 32000 });
  const [layers, setLayers] = useState<StepFormat>('q4_0');
  const [embed, setEmbed] = useState<StepFormat>('f32');
  const [B, setB] = useState(1);
  const opt = devKey === 'custom' ? { key: 'custom', label: tr('Custom', 'Tùy chỉnh'), dev: custom } : (options.find((o) => o.key === devKey) ?? BOOK_OPTION);
  const shape = shapeKey === 'm8' ? M8_DEFAULT : cshape;
  const fm: StepFormats = { layers, embed };
  const base = stepBytes(shape, F32F, B, opt.dev);
  const q = stepBytes(shape, fm, B, opt.dev);
  const speed = base.seconds / q.seconds;
  const wratio = matrixWeightBytes(shape, F32F) / matrixWeightBytes(shape, fm);
  const meas = opt.backend ? measuredSteps(rep, opt.backend) : [];
  const fmtName = `${layers}/${embed}`;
  const m = opt.backend ? measuredSpeedup(meas, fmtName, B) : null;
  const seg = (v: StepFormat, set: (f: StepFormat) => void, aria: string) => (
    <span className="seg" role="group" aria-label={aria}>
      {STEP_FORMATS.map((f) => <button key={f} aria-pressed={v === f} onClick={() => set(f)}>{STEP_FORMAT_LABEL[f]}</button>)}
    </span>
  );
  const numIn = (label: string, val: number, set: (v: number) => void) => (
    <label className="field">
      {label}
      <input type="number" aria-label={label} min="1" value={val} onChange={(e) => set(Math.max(1, Math.round(+e.target.value || 1)))} />
    </label>
  );
  return (
    <Bench
      title={tr("Where the step's bytes go", 'Byte của bước đi về đâu')}
      task={tr('Quantise only the layers, then only the head. Then raise the batch to 8.', 'Chỉ lượng tử hóa các lớp, rồi chỉ head. Sau đó tăng batch lên 8.')}
      foot={
        <Translated
          vi={
            <>
              Mỗi phần của bước trong lab 22.2 ở cận roofline của nó, max(byte / băng thông, phép toán / đỉnh), lần lượt
              nối tiếp nhau. Đậm: byte trọng số; nhạt: activation, bias, logits và id token, tất cả float32. Không mô hình
              hóa: phép tính giải lượng tử hóa, chi phí khởi chạy, bộ nhớ đệm. Các hàng embedding được đọc theo định dạng của
              E, nguyên khối. Thiết bị:{' '}
              {opt.label}, ngưỡng {fmt(thresholdOf(opt.dev))} FLOP/B.
            </>
          }
        >
          <>
            Every piece of lab 22.2's step at its roofline bound, max(bytes / bandwidth, operations / peak), one after
            another. Solid: weight bytes; light: activations, bias, logits and token ids, all float32. Not modelled: the
            dequantise arithmetic, launches, caches. Embedding rows are read in E's format, whole blocks. Device:{' '}
            {opt.label}, threshold {fmt(thresholdOf(opt.dev))} FLOP/B.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Kích thước bước">Step shape</Translated>
          <span className="seg" role="group" aria-label={tr('shape', 'kích thước')}>
            <button aria-pressed={shapeKey === 'm8'} onClick={() => setShapeKey('m8')}><Translated vi="mặc định của lab 22.2">lab 22.2 default</Translated></button>
            <button aria-pressed={shapeKey === 'custom'} onClick={() => setShapeKey('custom')}><Translated vi="tùy chỉnh">custom</Translated></button>
          </span>
        </label>
        {shapeKey === 'custom' ? (
          <>
            {numIn('d', cshape.d, (v) => setCshape({ ...cshape, d: Math.max(32, Math.round(v / 32) * 32) }))}
            {numIn(tr('layers', 'số lớp'), cshape.layers, (v) => setCshape({ ...cshape, layers: v }))}
            {numIn(tr('vocabulary', 'từ vựng'), cshape.vocab, (v) => setCshape({ ...cshape, vocab: v }))}
          </>
        ) : (
          <Translated vi={<span className="muted">d {M8_DEFAULT.d}, {M8_DEFAULT.layers} lớp, từ vựng {M8_DEFAULT.vocab.toLocaleString('en-US')}</span>}>
            <span className="muted">d {M8_DEFAULT.d}, {M8_DEFAULT.layers} layers, vocabulary {M8_DEFAULT.vocab.toLocaleString('en-US')}</span>
          </Translated>
        )}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field"><Translated vi="Trọng số các lớp">Layers' weights</Translated>{seg(layers, setLayers, tr('layer format', 'định dạng lớp'))}</label>
        <label className="field"><Translated vi="Embedding và LM head">Embedding and LM head</Translated>{seg(embed, setEmbed, tr('embedding format', 'định dạng embedding'))}</label>
        <label className="field">
          <Translated vi="Batch">Batch</Translated> {B}
          <input type="range" aria-label={tr('batch', 'batch')} min="1" max="8" value={B} onChange={(e) => setB(+e.target.value)} />
        </label>
        <label className="field">
          <Translated vi="Thiết bị">Device</Translated>
          <select aria-label={tr('device', 'thiết bị')} value={devKey} onChange={(e) => setDevKey(e.target.value)}>
            {options.map((o) => <option key={o.key} value={o.key}>{o.label}</option>)}
            <option value="custom">{tr('Custom numbers', 'Số liệu tùy chỉnh')}</option>
          </select>
        </label>
        {devKey === 'custom' ? (
          <>
            {numIn(tr('peak GFLOP/s', 'đỉnh GFLOP/s'), custom.peak, (v) => setCustom({ ...custom, peak: v }))}
            {numIn(tr('bandwidth GB/s', 'băng thông GB/s'), custom.bw, (v) => setCustom({ ...custom, bw: v }))}
          </>
        ) : null}
      </div>
      <StackedBar s={base} max={base.bytes} label="F32" />
      <StackedBar s={q} max={base.bytes} label={tr(`layers ${STEP_FORMAT_LABEL[layers]}, head ${STEP_FORMAT_LABEL[embed]}`, `các lớp ${STEP_FORMAT_LABEL[layers]}, head ${STEP_FORMAT_LABEL[embed]}`)} />
      <div className="narrate" role="status" data-part="step" data-speedup={speed} data-wratio={wratio} style={{ margin: '0.7rem 0' }}>
        <Translated
          vi={
            <>
              Byte trọng số giảm {wratio.toFixed(2)} lần. Bước của mô hình: {fmtTime(base.seconds)} ở F32, {fmtTime(q.seconds)} ở đây:{' '}
              <b>nhanh hơn {speed.toFixed(2)} lần</b> ở batch {B}.{' '}
              {m != null ? <>Lab X2.4 đo được {m.toFixed(2)} lần trên thiết bị này.</> : opt.backend ? <>Chưa có phép đo nào của lab X2.4 cho định dạng này ở batch {B} trên thiết bị này.</> : null}
            </>
          }
        >
          Weight bytes shrink {wratio.toFixed(2)} times. The model's step: {fmtTime(base.seconds)} in F32, {fmtTime(q.seconds)} here:{' '}
          <b>{speed.toFixed(2)} times faster</b> at batch {B}.{' '}
          {m != null ? <>Lab X2.4 measured {m.toFixed(2)} times on this device.</> : opt.backend ? <>No lab X2.4 measurement of this format at batch {B} on this device yet.</> : null}
        </Translated>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" data-part="pieces" style={{ margin: 0, maxWidth: '44rem' }}>
          <thead><tr><th><Translated vi="phần">piece</Translated></th><th className="num"><Translated vi="byte, F32">bytes, F32</Translated></th><th className="num"><Translated vi="byte, ở đây">bytes, here</Translated></th><th className="num"><Translated vi="thời gian mô hình, ở đây">model time, here</Translated></th><th><Translated vi="giới hạn bởi">bound</Translated></th></tr></thead>
          <tbody>
            {q.pieces.map((p, i) => {
              const bw = p.bytes / (opt.dev.bw * 1e9);
              return (
                <tr key={p.key} data-piece={p.key}>
                  <td>{p.label}</td>
                  <td className="num">{base.pieces[i].bytes.toLocaleString('en-US')}</td>
                  <td className="num">{p.bytes.toLocaleString('en-US')}</td>
                  <td className="num">{fmtTime(p.seconds)}</td>
                  <td>{p.seconds > bw * 1.0001 ? tr('arithmetic', 'tính toán') : tr('bytes', 'byte')}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

function StepReveal({ predicted }: { predicted: number }) {
  const { rep, options } = useDevices();
  const book = predictedSpeedup(M8_DEFAULT, ALLQ4, 1, BOOK_DEVICE);
  const mine = options.filter((o) => o.backend);
  const meas = gpuBackends(rep).flatMap((b) => {
    const s = measuredSpeedup(measuredSteps(rep, b), 'q4_0/q4_0', 1);
    return s != null ? [{ b, s, name: String(rep[b].device?.name ?? b) }] : [];
  });
  return (
    <>
      <Translated
        vi={
          <p data-part="reveal" data-ceiling={book}>
            Trần của mô hình byte ở kích thước của lab 22.2, mọi thứ ở Q4_0, batch 1: <b>{book.toFixed(2)} lần</b> trên
            thiết bị ví dụ của sách{mine.length ? <>, và {mine.map((o) => `${predictedSpeedup(M8_DEFAULT, ALLQ4, 1, o.dev).toFixed(2)} trên ${o.label.replace('Của bạn: ', '')}`).join('; ')}</> : null}.
            Đó là một trần: nó đếm byte, không đếm công giải nén.
          </p>
        }
      >
        <p data-part="reveal" data-ceiling={book}>
          The bytes model's ceiling at lab 22.2's shape, everything in Q4_0, batch 1: <b>{book.toFixed(2)} times</b> on the
          book's example device{mine.length ? <>, and {mine.map((o) => `${predictedSpeedup(M8_DEFAULT, ALLQ4, 1, o.dev).toFixed(2)} on ${o.label.replace('Yours: ', '')}`).join('; ')}</> : null}.
          It is a ceiling: it counts bytes, not the unpacking.
        </p>
      </Translated>
      {meas.length ? (
        meas.map((x) => (
          <Translated
            key={x.b}
            vi={
              <p>
                Lab X2.4 trên <b>{x.name}</b> ({x.b}) đo được <b>{x.s.toFixed(2)} lần</b>; bạn đã dự đoán {predicted}.
              </p>
            }
          >
            <p key={x.b}>
              Lab X2.4 on <b>{x.name}</b> ({x.b}) measured <b>{x.s.toFixed(2)} times</b>; you predicted {predicted}.
            </p>
          </Translated>
        ))
      ) : (
        <Translated
          vi={
            <p className="muted">
              Chưa có phép đo nào của lab X2.4 trên máy này. Hãy chạy nó từ <a href={href('x2', 'labs')}>trang lab</a>; khi
              đó ô này sẽ đặt phép đo cạnh dự đoán của bạn.
            </p>
          }
        >
          <p className="muted">
            No lab X2.4 measurement on this machine yet. Run it from the <a href={href('x2', 'labs')}>labs page</a>; this box
            will then put the measurement beside your prediction.
          </p>
        </Translated>
      )}
    </>
  );
}

// ---------------------------------------------------------------- explorer: a toy model, quantised
const TOY_FORMATS: MatFormat[] = ['f32', 'q8_0', 'q4_0', 'affine3', 'affine2'];

function ToyExplorer() {
  const model = useMemo(() => toyModel(), []);
  const [wf, setWf] = useState<MatFormat[]>(['q8_0', 'q8_0']);
  const [ef, setEf] = useState<MatFormat>('q8_0');
  const f = { W: wf, E: ef };
  const run = useMemo(() => greedyRun(model, f), [model, wf, ef]); // eslint-disable-line react-hooks/exhaustive-deps
  const flips = run.steps.filter((s) => s.flipped);
  const Wd = 640;
  const H = 170;
  const maxM = Math.max(...run.steps.map((s) => s.margin), 1);
  const maxE = Math.max(...run.steps.map((s) => s.errRms));
  const top = Math.max(maxM, 2 * maxE) * 1.05;
  const Y = (v: number) => 140 - (v / top) * 125;
  const cw = Wd / TOY.steps;
  const sel = (label: string, v: MatFormat, set: (x: MatFormat) => void) => (
    <label className="field">
      {label}
      <select aria-label={label} value={v} onChange={(e) => set(e.target.value as MatFormat)}>
        {TOY_FORMATS.map((x) => <option key={x} value={x}>{MAT_FORMAT_LABEL[x]}</option>)}
      </select>
    </label>
  );
  return (
    <Bench
      title={tr('A toy model, quantised', 'Một mô hình đồ chơi, được lượng tử hóa')}
      task={tr('Start with Q8_0 everywhere, then Q4_0, then affine 3-bit. Where do the red bars fall?', 'Bắt đầu với Q8_0 ở mọi nơi, rồi Q4_0, rồi affine 3 bit. Các cột đỏ rơi vào đâu?')}
      foot={
        <Translated
          vi={
            <>
              Một decoder ngẫu nhiên có seed, cùng cấu trúc với lab 22.2: d {TOY.d}, từ vựng {TOY.vocab}, {TOY.layers} lớp,
              embedding dùng chung, {TOY.steps} bước greedy từ token {TOY.start}. Cột: khoảng chênh top-2 ở float32 mỗi bước;
              chấm: hai lần mức thay đổi RMS của logits. Mỗi bước được chấm từ token của lượt chạy float32 (teacher-forced);
              các hàng token bên dưới chạy tự do. Trọng số {toyWeightBytes(model, f).toLocaleString('en-US')} byte ở đây so với{' '}
              {toyWeightBytes(model, { W: ['f32', 'f32'], E: 'f32' }).toLocaleString('en-US')} ở F32.
            </>
          }
        >
          <>
            A seeded random decoder with lab 22.2's structure: d {TOY.d}, vocabulary {TOY.vocab}, {TOY.layers} layers, tied
            embedding, {TOY.steps} greedy steps from token {TOY.start}. Bars: the float32 top-2 margin per step; dots: twice
            the RMS change in the logits. Each step is scored from the float32 run's token (teacher-forced); the token rows
            below run freely. Weights {toyWeightBytes(model, f).toLocaleString('en-US')} bytes here against{' '}
            {toyWeightBytes(model, { W: ['f32', 'f32'], E: 'f32' }).toLocaleString('en-US')} in F32.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        {sel(tr('W0 (layer 0)', 'W0 (lớp 0)'), wf[0], (x) => setWf([x, wf[1]]))}
        {sel(tr('W1 (layer 1)', 'W1 (lớp 1)'), wf[1], (x) => setWf([wf[0], x]))}
        {sel(tr('E (embedding and head)', 'E (embedding và head)'), ef, setEf)}
      </div>
      <Translated
        vi={
          <p className="muted" style={{ margin: '0 0 0.3rem', fontSize: '0.85rem' }} data-legend="">
            Mỗi bước: cột là khoảng chênh top-2 ở float32, chấm là hai lần mức thay đổi RMS của logits. Cột đỏ là bước mà mô
            hình lượng tử hóa chọn một token khác.
          </p>
        }
      >
        <p className="muted" style={{ margin: '0 0 0.3rem', fontSize: '0.85rem' }} data-legend="">
          Per step: the bar is the float32 top-2 margin, the dot twice the RMS change in the logits. A red bar is a step where
          the quantised model picks a different token.
        </p>
      </Translated>
      <div style={{ overflowX: 'auto' }}>
      <svg viewBox={`0 0 ${Wd} ${H}`} role="img" aria-label={tr('Top-2 margin and logits error per step', 'Khoảng chênh top-2 và sai số logits mỗi bước')} style={{ width: '100%', display: 'block', minWidth: 520 }}>
        <line x1={0} x2={Wd} y1={140} y2={140} stroke="var(--rule)" />
        {run.steps.map((s) => (
          <g key={s.step} data-step={s.step} data-flipped={s.flipped ? 'yes' : 'no'}>
            <rect x={s.step * cw + 3} y={Y(s.margin)} width={cw - 6} height={140 - Y(s.margin)} fill={s.flipped ? 'var(--fail)' : 'var(--ink-3)'} />
            <circle cx={s.step * cw + cw / 2} cy={Y(2 * s.errRms)} r={3} fill="var(--reg)" />
            {s.step % 4 === 0 ? <text x={s.step * cw + cw / 2} y={158} fontSize="12" textAnchor="middle">{s.step}</text> : null}
          </g>
        ))}
      </svg>
      </div>
      <div className="narrate" role="status" data-part="toy" data-flips={flips.length} data-div={run.firstDivergence ?? ''} style={{ margin: '0.6rem 0' }}>
        <Translated
          vi={
            <>
              Sai số của trọng số: E {(100 * run.weightRel.E).toFixed(1)}%, W0 {(100 * run.weightRel.W[0]).toFixed(1)}%, W1{' '}
              {(100 * run.weightRel.W[1]).toFixed(1)}%. Sai số của logits, trung bình qua các bước: {(100 * run.meanRel).toFixed(1)}%.{' '}
              {flips.length} trong {TOY.steps} bước chọn một token khác
              {flips.length ? `, ở các khoảng chênh ${flips.map((s) => s.margin.toFixed(1)).join(', ')}` : ''}.{' '}
              {run.firstDivergence == null ? 'Các chuỗi chạy tự do không bao giờ tách nhau.' : `Các chuỗi chạy tự do tách nhau ở vị trí ${run.firstDivergence}.`}
            </>
          }
        >
          Weights' error: E {(100 * run.weightRel.E).toFixed(1)}%, W0 {(100 * run.weightRel.W[0]).toFixed(1)}%, W1{' '}
          {(100 * run.weightRel.W[1]).toFixed(1)}%. Logits' error, mean over steps: {(100 * run.meanRel).toFixed(1)}%.{' '}
          {flips.length} of {TOY.steps} steps pick a different token
          {flips.length ? `, at margins ${flips.map((s) => s.margin.toFixed(1)).join(', ')}` : ''}.{' '}
          {run.firstDivergence == null ? 'The free-running sequences never diverge.' : `The free-running sequences part at position ${run.firstDivergence}.`}
        </Translated>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" data-part="tokens" style={{ margin: 0, fontVariantNumeric: 'tabular-nums' }}>
          <tbody>
            {(['F32', 'quantised'] as const).map((who) => {
              const toks = who === 'F32' ? run.f32Tokens : run.qTokens;
              return (
                <tr key={who}>
                  <th style={{ whiteSpace: 'nowrap' }}><Translated vi={who === 'F32' ? 'F32' : 'lượng tử hóa'}>{who}</Translated></th>
                  {toks.map((t, i) => (
                    <td key={i} className={who === 'quantised' && t !== run.f32Tokens[i] ? 't-fail' : ''} style={{ paddingRight: '0.35rem' }}>{t}</td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the page
export function SX27() {
  const q = questionsSX27;
  const headShare = (M8_DEFAULT.vocab * M8_DEFAULT.d) / (M8_DEFAULT.layers * M8_DEFAULT.d ** 2 + M8_DEFAULT.vocab * M8_DEFAULT.d);
  return (
    <>
      <Translated
        vi={
          <p>
            Mọi thứ đến giờ chỉ là một ma trận. Mốc dự án M8, bước decode của lab 22.2, có nhiều ma trận: một phép gather
            embedding, một lớp tuyến tính với bias và residual cho mỗi khối, một LM head dùng chung trọng số với bảng
            embedding, một softmax và một argmax, tất cả ở float32 với chiều batch đứng đầu. Decode là phép nhân vector với ma
            trận, bị giới hạn bởi số byte của trọng số. Mục này đưa các ma trận lượng tử hóa vào bước đó, dự đoán cả bước
            được lợi bao nhiêu, và đánh giá thiệt hại ở chỗ quan trọng: trong logits và token.
          </p>
        }
      >
        <p>
          Everything so far was one matrix. Milestone M8, lab 22.2's decode step, is several: an embedding gather, a linear
          layer with bias and residual per block, an LM head tied to the embedding table, a softmax and an argmax, all in
          float32 with the batch first. Decode is vector-matrix work, bound by the bytes of the weights. This section puts
          quantised matrices into that step, predicts what the whole step gains, and judges the damage where it matters: in
          the logits and the tokens.
        </p>
      </Translated>
      <Beyond title={tr("Ours, not the book's", 'Của khóa học này, không phải của sách')}>
        <Translated
          vi={
            <p>
              Việc batching biến các phép nhân vector-ma trận của decode thành phép nhân ma trận là của sách (chương 22,
              §22.5). Bước decode của lab 22.2 là của dự án, xây trên đó; lượng tử hóa bước này là của khóa học này.
            </p>
          }
        >
          <p>
            That batching turns decode's vector-matrix products into matrix products is the book's (chapter 22, §22.5).
            Lab 22.2's step is the project's, built on it; quantising the step is ours.
          </p>
        </Translated>
      </Beyond>
      <Source section="22.5" pages="539–540" />
      <Translated
        vi={
          <p>
            Byte của bước nằm ở đâu? Ở kích thước mặc định của lab 22.2 (d 4096, 2 lớp, từ vựng 32,000), riêng LM head đã
            chiếm 131 triệu trong 165 triệu trọng số, {(100 * headShare).toFixed(0)}%. Hai lớp cộng lại là phần còn lại. Các
            byte không phải trọng số (activation, bias, logits, xác suất) dưới một megabyte khá xa ở batch 1.
          </p>
        }
      >
        <p>
          Where are the step's bytes? At lab 22.2's default shape (d 4096, 2 layers, vocabulary 32,000) the LM head alone is
          131 million of the 165 million weights, {(100 * headShare).toFixed(0)}%. The two layers together are the rest. The
          non-weight bytes (activations, the bias, the logits, the probabilities) are well under a megabyte at batch 1.
        </p>
      </Translated>
      <Source section="20.6" pages="504–506" />

      <h2><Translated vi="Điều gì thay đổi ở mốc dự án M9">What changes in milestone M9</Translated></h2>
      <Translated
        vi={
          <p>
            Lab X2.4 thực hiện việc thay thế ở cấp kernel: mọi lớp tuyến tính đọc trọng số lượng tử hóa qua kernel int8 của
            lab X2.1 hoặc kernel Q4_0 của lab X2.2, và vì E được gather và head dùng chung, gather phải giải lượng tử hóa một
            hàng khối cho mỗi chuỗi. Bias, residual và bộ lấy mẫu vẫn ở float32, đúng như quy ước của dự án. Batch giữ ở 8
            trở xuống: đó là ngưỡng mà chính llama.cpp dùng kernel ma trận-vector cho trọng số lượng tử hóa.
          </p>
        }
      >
        <p>
          Lab X2.4 makes the swap at the kernel level: every linear reads quantised weights through lab X2.1's int8 kernel
          or lab X2.2's Q4_0 kernel, and because E is shared by the gather and the head, the gather must dequantise one row
          of blocks per sequence. The bias, the residual and the sampler stay float32, as the project's convention says.
          Batches stay at 8 or below: that is where llama.cpp itself uses matrix-vector kernels for quantised weights.
        </p>
      </Translated>
      <Today checked={tr(`${CHECKED('D5')}; ch15 D1 for the batch limit`, `${CHECKED('D5')}; ch15 D1 cho giới hạn batch`)}>
        <Translated
          vi={
            <p>
              <code>mlx.nn.QuantizedLinear</code>, theo lời của MLX, là "the quantized equivalent of mlx.nn.Linear" (bản
              lượng tử hóa tương đương của mlx.nn.Linear), mặc định dùng nhóm 64 và 4 bit. <code>MMVQ_MAX_BATCH_SIZE</code>{' '}
              của llama.cpp là 8.
            </p>
          }
        >
          <p>
            <code>mlx.nn.QuantizedLinear</code> is, in MLX's words, "the quantized equivalent of mlx.nn.Linear", with
            groups of 64 and 4 bits by default. llama.cpp's <code>MMVQ_MAX_BATCH_SIZE</code> is 8.
          </p>
        </Translated>
      </Today>
      <Ask q={q['x2/X2.7/q-embed']} />

      <h2><Translated vi="Dự đoán từ số byte của cả bước">Predict from the bytes of the whole step</Translated></h2>
      <Translated
        vi={
          <p>
            Mục X2.1 dự đoán mức tăng tốc của một GEMV bằng tỉ lệ byte của nó. Với cả bước, hãy dùng số byte của bước. Chỉ
            lượng tử hóa các lớp thì bước gần như không đổi, vì head chiếm phần lớn; lượng tử hóa mọi thứ thì byte trọng số
            giảm 7.11 lần, nhưng các phần không phải trọng số và các lần khởi chạy không giảm chút nào. Hãy thử cả hai trong
            trình khám phá trước khi trả lời.
          </p>
        }
      >
        <p>
          Section X2.1 predicted one GEMV's speed-up as its byte ratio. For a step, use the step's bytes. Quantise only the
          layers and the step barely moves, because the head dominates; quantise everything and the weight bytes fall by
          7.11 times, but the non-weight pieces and the launches do not shrink at all. Try both in the explorer before you
          answer.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi phần được tính thời gian ở cận roofline của nó với activation float32 của lab. Không có phép tính giải lượng
              tử hóa, không có chi phí khởi chạy, không có bộ nhớ đệm, không có attention (bước của lab 22.2 không có). Số liệu
              thiết bị của sách chỉ là ví dụ của sách; số liệu của bạn đến từ lab 5.1.
            </p>
          }
        >
          <p>
            Each piece is timed at its roofline bound with the lab's float32 activations. No dequantise arithmetic, no
            launch cost, no caches, no attention (lab 22.2's step has none). The book's device numbers are its example
            only; your own come from lab 5.1.
          </p>
        </Translated>
      </Simplification>
      <StepExplorer />
      <Ask q={q['x2/X2.7/q-step-ratio']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={q['x2/X2.7/q-batch8']} />
      <Predict
        id="x2/X2.7/p-step-speedup"
        prompt={
          <Translated
            vi={
              <>
                Trước khi chạy lab X2.4: bạn dự đoán bước toàn Q4_0 nhanh hơn bước float32 bao nhiêu lần ở batch 1 trên thiết
                bị <em>của bạn</em>?
              </>
            }
          >
            <>
              Before you run lab X2.4: how many times faster than the float32 step do you expect the all-Q4_0 step to be at
              batch 1 on <em>your</em> device?
            </>
          </Translated>
        }
        unit={tr('× faster', '× nhanh hơn')}
      >
        {(v) => <StepReveal predicted={v} />}
      </Predict>

      <h2><Translated vi="Sai số đáng quan tâm">The error that matters</Translated></h2>
      <Translated
        vi={
          <p>
            Mục X2.2 đo sai số trên trọng số. Một mô hình được đánh giá qua đầu ra của nó. Vậy hãy đo hai thứ so với bước
            float32: sai số RMS tương đối của logits, và việc decode greedy có chọn cùng token hay không. Mô hình đồ chơi bên
            dưới có cấu trúc emulator của lab X2.4 ở kích thước trình duyệt chạy được. Bắt đầu mỗi ma trận ở Q8_0, rồi hạ từng
            cái một và xem bước nào đổi token.
          </p>
        }
      >
        <p>
          Section X2.2 measured error on the weights. A model is judged by its outputs. So measure two things against the
          float32 step: the relative RMS error of the logits, and whether greedy decoding picks the same token. The toy
          model below has lab X2.4's emulator structure at a size the browser can run. Start each matrix at Q8_0, then lower
          one at a time and watch which steps flip.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mô hình đồ chơi là ngẫu nhiên, chưa được huấn luyện. Việc token của nó khớp hay không không nói gì về perplexity
              hay độ chính xác của một mô hình thật; nó cho thấy cơ chế: một lần đổi token cần một khoảng chênh nhỏ.
            </p>
          }
        >
          <p>
            The toy model is random, not trained. Its tokens agreeing or not says nothing about a real model's perplexity or
            accuracy; it shows the mechanism: a flip needs a small margin.
          </p>
        </Translated>
      </Simplification>
      <ToyExplorer />
      <Ask q={q['x2/X2.7/q-margin']} />
      <Ask q={q['x2/X2.7/q-weights-vs-output']} />
      <Translated
        vi={
          <p>
            Với mô hình thật, chính các dự án tự nói điều đó. Nhóm PyTorch báo cáo lượng tử hóa int8 chỉ trọng số không làm
            giảm độ chính xác một cách nhận thấy được, còn int4 thì có, và với int4, một hệ số tỉ lệ cho mỗi 32 trọng số là lựa
            chọn phổ biến. Đó là khẳng định của họ về mô hình của họ, không phải điều trang này đã đo.
          </p>
        }
      >
        <p>
          For real models, the projects say it themselves. PyTorch's team reports no perceptible accuracy loss from int8
          weight-only quantisation and some from int4, for which one scale per 32 weights is a common choice. Those are
          their statements about their models, not something this page measured.
        </p>
      </Translated>
      <Today checked={CHECKED('A1')}>
        <Translated
          vi={
            <p>
              PyTorch, "Accelerating Generative AI with PyTorch II: GPT, Fast": "using int8 weight-only quantization has no
              perceptible accuracy degradation, using int4 weight-only quantization does" (lượng tử hóa int8 chỉ trọng số
              không làm giảm độ chính xác một cách nhận thấy được, int4 thì có); với int4, "one scaling factor per 32
              elements ... is also a common choice among the community" (một hệ số tỉ lệ cho mỗi 32 phần tử cũng là lựa
              chọn phổ biến trong cộng đồng).
            </p>
          }
        >
          <p>
            PyTorch, "Accelerating Generative AI with PyTorch II: GPT, Fast": "using int8 weight-only quantization has no
            perceptible accuracy degradation, using int4 weight-only quantization does"; for int4, "one scaling factor per 32
            elements ... is also a common choice among the community".
          </p>
        </Translated>
      </Today>
      <Translated vi={<p>M9 chỉ lượng tử hóa trọng số, không gì khác.</p>}>
        <p>M9 quantises the weights and nothing else.</p>
      </Translated>
      <Beyond title={tr('The KV cache', 'KV cache')}>
        <Translated
          vi={
            <p>
              M9 để KV cache ở float32. llama.cpp cũng có thể lưu nó ở dạng <code>q8_0</code> hoặc <code>q4_0</code> (34 và 18
              byte cho mỗi 32 giá trị; notes/today/ch20-facts.md D3), cách này tấn công vào lưu lượng attention mà batching
              không giúp được.
            </p>
          }
        >
          <p>
            M9 leaves the KV cache in float32. llama.cpp can store it as <code>q8_0</code> or <code>q4_0</code> too (34 and 18
            bytes per 32 values; notes/today/ch20-facts.md D3), which attacks the attention traffic that batching does not
            help.
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Chuyển sang lab X2.4">Hand-off to lab X2.4</Translated></h2>
      <Translated
        vi={
          <p>
            Lab X2.4 dựng bước này trên GPU của bạn và báo cáo thời gian mỗi bước cho từng định dạng và batch. Trang lab đặt
            phép đo cạnh dự đoán bạn đã chốt ở đây, và đổi nó thành mức sử dụng băng thông: byte trọng số mỗi bước chia cho
            thời gian mỗi bước, tính theo tỉ lệ của băng thông đỉnh của bạn từ lab 5.1. Với số liệu giả định: 100 MB trọng số
            trong 1 ms so với 200 GB/s đo được là {(100 * bandwidthUtilisation(100e6, 1e-3, 200)).toFixed(0)}%. Hãy sang{' '}
            <a href={href('x2', 'labs')}>trang lab</a> khi bạn sẵn sàng.
          </p>
        }
      >
        <p>
          Lab X2.4 builds this step on your GPU and reports time per step for each format and batch. The labs page puts the
          measurement beside the prediction you committed here, and turns it into bandwidth utilisation: the weight bytes
          per step divided by the time per step, as a fraction of your peak bandwidth from lab 5.1. With made-up numbers:
          100 MB of weights in 1 ms against a measured 200 GB/s is {(100 * bandwidthUtilisation(100e6, 1e-3, 200)).toFixed(0)}%. Go to the{' '}
          <a href={href('x2', 'labs')}>labs page</a> when you are ready.
        </p>
      </Translated>
    </>
  );
}
