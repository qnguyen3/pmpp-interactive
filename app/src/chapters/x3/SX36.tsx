import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';
import {
  BINS, IMPLS, OFFSETS, ORDER, STAGE_LIST, SYSTEMS, TIE_VALUE,
  argmaxBy, cellOf, cudaButterfly, metalLanes, metalMaxLanes, tieRow,
  type Bin, type Impl, type Stage, type System,
} from '../../sim/argmaxties';
import { functionSource } from '../../sim/toolchain';

const X3 = 'notes/today/x3-facts.md';
const X1 = 'notes/today/x1-facts.md';
const C10 = 'notes/today/ch10-facts.md';
const LAB = 'ch10/lab3_softmax_argmax';
const TIERS: Record<string, Tier> = { v: 'reg', i: 'reg', otherV: 'reg', otherI: 'reg' };
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };

// ---------------------------------------------------------------- questions
export const questionsSX36: Record<string, Question> = {
  'x3/X3.6/q-tie-cuda': {
    kind: 'numeric',
    id: 'x3/X3.6/q-tie-cuda',
    prompt: (
      <Translated
        vi={
          <>
            Một hàng 32 cột trên kernel argmax CUDA của llama.cpp, mỗi lane của một warp giữ một cột. Cột 1 và cột 2 có
            cùng một giá trị cực đại. Kernel trả về chỉ số nào?
          </>
        }
      >
        <>
          A 32-column row on llama.cpp's CUDA argmax, one column per lane of one warp. Columns 1 and 2 hold the same
          maximum. Which index does the kernel return?
        </>
      </Translated>
    ),
    answer: 2,
    explain: (
      <Translated
        vi={
          <p>
            Lane 0 ghi kết quả. Ở các offset 16, 8 và 4, không có gì đến được nó. Ở offset 2 nó ghép cặp với lane 2, lane đang
            giữ cực đại ở cột 2, và lấy cặp đó. Ở offset 1 nó ghép cặp với lane 1, lane giữ cùng giá trị đó ở cột 1, và giữ
            nguyên cặp của mình, vì một cặp chỉ bị thay bởi một giá trị lớn hơn hẳn. Vậy đáp án là 2: không phải cực đại "đầu
            tiên" cũng không phải "cuối cùng", mà là cực đại nào đến lane 0 trước qua butterfly.
          </p>
        }
      >
        <p>
          Lane 0 writes the result. Nothing reaches it at offsets 16, 8 and 4. At offset 2 it pairs with lane 2, which holds
          the maximum at column 2, and takes that pair. At offset 1 it pairs with lane 1, which holds the same value at
          column 1, and keeps its own pair, because a pair is replaced only by a strictly larger value. So 2: neither "the
          first" nor "the last" maximum, but whichever reached lane 0 first through the butterfly.
        </p>
      </Translated>
    ),
    source: `${C10} D2 (our reading of argmax.cu)`,
    vi: { source: `${C10} D2 (cách chúng tôi đọc argmax.cu)` },
  },
  'x3/X3.6/q-tie-metal': {
    kind: 'choice',
    id: 'x3/X3.6/q-tie-metal',
    prompt: (
      <Translated vi={<>Hai cực đại bằng nhau ở cột 0 và cột 1 của một hàng 32 cột. Kernel Metal của llama.cpp, kernel CUDA của llama.cpp và M4 của khóa học lần lượt trả về gì?</>}>
        <>Equal maxima at columns 0 and 1 of a 32-column row. What do llama.cpp's Metal kernel, llama.cpp's CUDA kernel and the course's M4 return, in that order?</>
      </Translated>
    ),
    options: [
      { label: '1, 0, 0', correct: true },
      { label: '0, 0, 0', why: <Translated vi="Metal lấy chỉ số lớn nhất trong các lane đang giữ cực đại: simd_max trên các chỉ số của chúng.">Metal takes the largest index among the lanes holding the maximum: simd_max over their indices.</Translated> },
      { label: '1, 1, 0', why: <Translated vi="Trên CUDA, lane 0 chỉ gặp lane 1 ở offset 1, lúc lane 1 giữ một giá trị bằng nó, và lane 0 giữ nguyên cặp của mình.">On CUDA lane 0 meets lane 1 only at offset 1, holding an equal value, and keeps its own pair.</Translated> },
      { label: '0, 1, 0', why: <Translated vi="Hai đáp án của llama.cpp bị đổi chỗ cho nhau.">Both llama.cpp answers are swapped.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Metal: <code>simd_max(select(-1, larg, lmax == max_val))</code> chọn chỉ số lớn hơn trong hai chỉ số, tức 1. CUDA: lane 0
            đã giữ cực đại của cột 0 và giữ nó trước giá trị bằng nhau của lane 1. Phép kết hợp của M4 gửi các giá trị bằng nhau
            về chỉ số nhỏ hơn, tức 0, trên mọi backend.
          </p>
        }
      >
        <p>
          Metal: <code>simd_max(select(-1, larg, lmax == max_val))</code> picks the larger of the two indices, 1. CUDA: lane 0
          already holds column 0's maximum and keeps it against lane 1's equal value. M4's combine sends equal values to the
          lower index, 0, on every backend.
        </p>
      </Translated>
    ),
    source: `${C10} D2; lab 10.3's argmaxCombine`,
    vi: { source: `${C10} D2; argmaxCombine của lab 10.3` },
  },
  'x3/X3.6/q-compare-greedy': {
    kind: 'choice',
    id: 'x3/X3.6/q-compare-greedy',
    prompt: (
      <Translated
        vi={
          <>
            Bạn so sánh các token id greedy của llama.cpp trên Metal với transformers trên CPU. Chúng khác nhau ở một bước,
            nơi hai logit lớn nhất được in ra bằng nhau. Đây có phải là lỗi không?
          </>
        }
      >
        <>
          You compare greedy token ids from llama.cpp on Metal with transformers on the CPU. They differ at one step, where
          the top two logits printed equal. Is this a bug?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Chưa chắc: quy tắc phân định hòa khác nhau. Hãy dùng một bản tham chiếu có cùng quy tắc, hoặc kiểm tra khoảng cách giữa hai giá trị lớn nhất.">
            Not necessarily: the tie rules differ. Use a reference with the same rule, or check the top-two margin.
          </Translated>
        ),
        correct: true,
      },
      { label: <Translated vi="Có: một kernel đúng trả về cùng token với torch.argmax trên mọi đầu vào.">Yes: a correct kernel returns the same token as torch.argmax on every input.</Translated>, why: <Translated vi="Chỉ khi nó được viết theo cùng quy tắc phân định hòa. Kernel Metal của llama.cpp thì không.">Only if it is written to the same tie rule. llama.cpp's Metal kernel is not.</Translated> },
      { label: <Translated vi="Có: logit in ra bằng nhau nghĩa là các số float bằng nhau, và với các float bằng nhau thì mọi argmax đúng đều trả về cùng một chỉ số.">Yes: equal printed logits mean equal floats, and with equal floats every correct argmax returns the same index.</Translated>, why: <Translated vi="Ngay cả float bằng nhau cũng để việc chọn cho quy tắc phân định hòa. Và giá trị in ra đã được làm tròn: trên hai backend, các float có thể khác nhau ở vài bit cuối.">Even equal floats leave the choice to the tie rule. And printed values are rounded: the floats may differ in the last bits on two backends.</Translated> },
      { label: <Translated vi="Không: giải mã greedy vốn không được kỳ vọng tái lập được giữa các thiết bị.">No: greedy decoding is not expected to be reproducible across devices.</Translated>, why: <Translated vi="Có được kỳ vọng, ở bất cứ đâu khoảng cách lớn hơn sai khác do làm tròn. Đó là bài kiểm thử cần viết.">It is, wherever the margin exceeds the rounding differences. That is the test to write.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            torch.argmax ghi trong tài liệu là lấy cực đại đầu tiên; kernel Metal của llama.cpp lấy chỉ số lớn nhất trong các
            cực đại. Khi hòa chính xác, hai bên khác nhau theo thiết kế, và sau một token khác nhau thì mọi bước sau đó đều có
            thể khác. Vì vậy một bài kiểm thử về sự trùng khớp greedy chính xác cần một bản tham chiếu có cùng quy tắc phân định
            hòa, và nó nên đánh dấu các bước có khoảng cách giữa hai giá trị lớn nhất nằm trong dung sai làm tròn thay vì báo lỗi
            ở các bước đó.
          </p>
        }
      >
        <p>
          torch.argmax documents the first maximum; llama.cpp's Metal kernel takes the largest index among the maxima. On an
          exact tie the two disagree by design, and after one different token every later step may differ. A test of exact
          greedy agreement therefore needs a reference with the same tie rule, and it should flag steps whose top-two margin
          is within the rounding tolerance instead of failing on them.
        </p>
      </Translated>
    ),
    source: `${X3} D7, D9; ${C10} D2, D4`,
  },
  'x3/X3.6/q-multinomial': {
    kind: 'choice',
    id: 'x3/X3.6/q-multinomial',
    prompt: (
      <Translated vi={<>Cùng xác suất, cùng các số ngẫu nhiên đều. Vì sao torch.multinomial và một sampler dùng CDF ngược có thể cho ra token khác nhau?</>}>
        <>Same probabilities, same uniform random numbers. Why can torch.multinomial and an inverse-CDF sampler give different tokens?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="torch.multinomial(probs, 1) lấy argmax(p / q) với nhiễu mũ q: cùng phân phối, nhưng token khác.">
            torch.multinomial(probs, 1) takes argmax(p / q) with exponential noise q: same distribution, other tokens.
          </Translated>
        ),
        correct: true,
      },
      { label: <Translated vi="torch.multinomial sắp xếp các xác suất theo thứ tự giảm dần trước, nên CDF của nó được duyệt theo một thứ tự token khác.">torch.multinomial sorts the probabilities in descending order first, so its CDF is walked in another token order.</Translated>, why: <Translated vi="Nó không sắp xếp. Với một mẫu, nó hoàn toàn không dựng CDF.">It does not sort. For one sample it never builds a CDF at all.</Translated> },
      { label: <Translated vi="Sai số làm tròn float trong tổng tích lũy, làm dịch mọi biên của các ngăn đi vài đơn vị ở chữ số cuối.">Float rounding in the cumulative sum, which moves every bucket boundary by a few units in the last place.</Translated>, why: <Translated vi="Điều đó có thể dịch một biên đi vài ulp, nhưng ở đây chính các thuật toán khác nhau.">That can move a boundary by a few ulps, but here the algorithms themselves differ.</Translated> },
      { label: <Translated vi="Không thể: cả hai đều là sampler chính xác của cùng một phân phối.">They cannot: both are exact samplers of the same distribution.</Translated>, why: <Translated vi="Các sampler chính xác của cùng một phân phối vẫn có thể ánh xạ cùng các bit ngẫu nhiên ra các token khác nhau.">Exact samplers of one distribution can still map the same random bits to different tokens.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Chính chú thích của PyTorch suy ra nó từ Gumbel-max: s = argmax(p / q) với q lấy từ Exp(1). Cả hai phương pháp đều lấy
            mẫu chính xác từ phân phối, nên qua nhiều lần rút thì tần suất khớp nhau; với cùng một bộ bit ngẫu nhiên, chúng chọn
            token khác nhau. Bản tìm kiếm CDF ngược trên CUDA của PyTorch có tồn tại, trong <code>MultinomialKernel.cu</code>, nhưng
            chỉ dùng khi lấy mẫu có hoàn lại và nhiều mẫu.
          </p>
        }
      >
        <p>
          PyTorch's own comment derives it from Gumbel-max: s = argmax(p / q) with q drawn from Exp(1). Both methods sample
          the distribution exactly, so over many draws the frequencies agree; for one set of random bits they pick different
          tokens. PyTorch's CUDA inverse-CDF search exists, in <code>MultinomialKernel.cu</code>, but only with replacement
          and several samples.
        </p>
      </Translated>
    ),
    source: `${X1} D5`,
  },
  'x3/X3.6/q-default-place': {
    kind: 'choice',
    id: 'x3/X3.6/q-default-place',
    prompt: (
      <Translated vi={<>Với các giá trị mặc định của llama.cpp, chuỗi lấy mẫu của quá trình sinh thông thường chạy ở đâu?</>}>
        <>With llama.cpp's defaults, where does the sampling chain of ordinary generation run?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Trên host: backend_sampling mặc định là false, và -bs, đường đi trên GPU, còn ở dạng thử nghiệm.">On the host: backend_sampling is false by default, and -bs, the GPU path, is experimental.</Translated>, correct: true },
      { label: <Translated vi="Trên GPU, trong cùng graph với mô hình, nên chỉ token được chọn quay về host.">On the GPU, in the same graph as the model, so only the chosen token returns to the host.</Translated>, why: <Translated vi="Đó là đường -bs, mặc định tắt khi sinh. Bản nháp speculative là ngoại lệ, mặc định bật.">That is the -bs path, which is off by default for generation. The speculative draft is the exception, on by default.</Translated> },
      { label: <Translated vi="Các hình phạt chạy trên host, nơi có lịch sử token; mọi thứ sau đó chạy trên GPU.">Penalties on the host, where the token history is; everything after them on the GPU.</Translated>, why: <Translated vi="Mặc định không có cách chia nào như vậy: cả chuỗi là mã host.">No such split exists by default: the whole chain is host code.</Translated> },
      { label: <Translated vi="Trên backend nào đã tính logits, vì chuỗi được viết bằng các tensor op.">On whichever backend computed the logits, since the chain is written as tensor ops.</Translated>, why: <Translated vi="Đó là mô tả của transformers và mlx-lm, nơi sampler là các tensor op.">That describes transformers and mlx-lm, whose samplers are tensor ops.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>common_params_sampling</code> có <code>bool backend_sampling = false;</code> cho quá trình sinh và{' '}
            <code>true</code> cho bản nháp speculative. Cờ là <code>-bs</code>, "enable backend sampling (experimental)
            (default: disabled)". Logits quay về host và chuỗi trong <code>src/llama-sampler.cpp</code> chạy ở đó.
          </p>
        }
      >
        <p>
          <code>common_params_sampling</code> has <code>bool backend_sampling = false;</code> for generation and{' '}
          <code>true</code> for the speculative draft. The flag is <code>-bs</code>, "enable backend sampling (experimental)
          (default: disabled)". The logits come back to the host and the chain in <code>src/llama-sampler.cpp</code> runs
          there.
        </p>
      </Translated>
    ),
    source: `${X1} D1`,
  },
};

// ---------------------------------------------------------------- the tie stepper
const PRESETS: { label: string; cols: number[] }[] = [
  { get label() { return tr('1 and 2', '1 và 2'); }, cols: [1, 2] },
  { get label() { return tr('0 and 1', '0 và 1'); }, cols: [0, 1] },
  { get label() { return tr('0 and 16', '0 và 16'); }, cols: [0, 16] },
  { get label() { return tr('5 and 9', '5 và 9'); }, cols: [5, 9] },
  { get label() { return tr('3, 7 and 30', '3, 7 và 30'); }, cols: [3, 7, 30] },
];
// Read at render time so the stage names follow the selected language.
const STAGE_NAMES = () => [
  tr('each lane loads its column', 'mỗi lane nạp cột của nó'),
  ...OFFSETS.map((o) => tr(`xor offset ${o}`, `offset xor ${o}`)),
  tr('Metal: simd_max of the indices', 'Metal: simd_max trên các chỉ số'),
];

function TieStepper() {
  const [planted, setPlanted] = useState<number[]>([1, 2]);
  const [guess, setGuess] = useState<Record<Impl, string>>({ cuda: '', metal: '', host: '', torch: '', m4: '' });
  const [revealed, setRevealed] = useState(false);
  const [stage, setStage] = useState(0);
  const [lane, setLane] = useState(0);
  const row = useMemo(() => tieRow(planted), [planted]);
  const bf = useMemo(() => cudaButterfly(row), [row]);
  const metal = useMemo(() => metalLanes(row), [row]);
  const maxLanes = useMemo(() => metalMaxLanes(row), [row]);
  const results = useMemo(() => Object.fromEntries(IMPLS.map((k) => [k.id, argmaxBy(k.id, row)])) as Record<Impl, number>, [row]);
  const last = STAGE_NAMES().length - 1;
  const isMetal = stage === last;
  const st = bf[Math.min(stage, bf.length - 1)];
  const off = st.offset;

  const reset = (cols: number[]) => {
    setPlanted([...cols].sort((a, b) => a - b));
    setRevealed(false);
    setStage(0);
    setGuess({ cuda: '', metal: '', host: '', torch: '', m4: '' });
  };
  const toggle = (c: number) => reset(planted.includes(c) ? planted.filter((x) => x !== c) : [...planted, c]);

  return (
    <Bench
      title={tr('Ties in one warp', 'Giá trị hòa trong một warp')}
      task={
        <Translated vi="Cài các cực đại bằng nhau bằng cách bấm vào các cột, ghi lại kết quả mỗi cách cài đặt trả về, rồi chạy từng bước qua các phép rút gọn.">
          Plant equal maxima by clicking columns, write down what each implementation returns, then step through the reductions.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Hàng nằm trong bộ nhớ toàn cục (global memory): một cột được cài giữ {TIE_VALUE}; mọi cột khác giữ một giá trị khác,
              nhỏ hơn nó, nên chỉ các cột được cài là hòa nhau. Lane k nạp cột k vào thanh ghi (register) của nó. Lane 0 ghi kết quả
              của kernel CUDA.
            </>
          }
        >
          <>
            The row is in global memory: a planted column holds {TIE_VALUE}; every other column holds a different value below
            it, so only the planted columns tie. Lane k loads column k into its registers. Lane 0 writes the CUDA kernel's
            result.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Các cột hòa nhau">Tied columns</Translated>
          <span className="seg" role="group" aria-label={tr('Presets', 'Mẫu có sẵn')} style={{ flexWrap: 'wrap' }}>
            {PRESETS.map((p) => (
              <button key={p.label} aria-pressed={p.cols.join() === planted.join()} onClick={() => reset(p.cols)}>{p.label}</button>
            ))}
          </span>
        </span>
        <button onClick={() => reset([])} disabled={!planted.length}><Translated vi="xóa">clear</Translated></button>
      </div>
      <div data-part="row" style={{ display: 'grid', gridTemplateColumns: 'repeat(8, minmax(0, 1fr))', gap: '0.25rem', maxWidth: '34rem', marginBottom: '0.8rem' }}>
        {row.map((v, c) => (
          <button key={c} data-col={c} data-value={v} data-planted={planted.includes(c) ? 1 : 0} onClick={() => toggle(c)}
            aria-label={tr(
              `column ${c}, value ${v.toFixed(1)}${planted.includes(c) ? ', planted maximum' : ''}`,
              `cột ${c}, giá trị ${v.toFixed(1)}${planted.includes(c) ? ', cực đại được cài' : ''}`,
            )}
            style={{
              padding: '0.25rem 0', fontVariantNumeric: 'tabular-nums', fontWeight: planted.includes(c) ? 700 : 400,
              background: planted.includes(c) ? 'var(--global-wash)' : undefined,
              outline: planted.includes(c) ? '2px solid var(--global)' : undefined, outlineOffset: -1,
            }}
            title={tr(`column ${c}`, `cột ${c}`)}>
            <span className="muted" style={{ fontSize: '0.7rem', display: 'block' }}>{c}</span>
            {v.toFixed(1)}
          </button>
        ))}
      </div>

      <div style={{ overflowX: 'auto', marginBottom: '0.8rem' }}>
      <table className="data" style={{ margin: 0, background: 'var(--page)', maxWidth: '48rem' }} data-part="guesses">
        <thead>
          <tr><th style={{ paddingLeft: '0.6rem' }}><Translated vi="Cách cài đặt">Implementation</Translated></th><th><Translated vi="đáp án của bạn">your answer</Translated></th>{revealed ? <><th><Translated vi="trả về">returns</Translated></th><th><Translated vi="quy tắc">rule</Translated></th></> : null}</tr>
        </thead>
        <tbody>
          {IMPLS.map((k) => {
            const g = guess[k.id].trim() === '' ? null : Number(guess[k.id]);
            const r = results[k.id];
            return (
              <tr key={k.id} data-impl={k.id} data-result={revealed ? r : undefined}>
                <td style={{ paddingLeft: '0.6rem' }}>{tr(k.label, { cuda: 'llama.cpp CUDA (argmax.cu, cũng là file HIP)', metal: 'llama.cpp Metal (kernel_argmax_f32)', host: 'sampler greedy trên host của llama.cpp', torch: 'torch.argmax (greedy của transformers)', m4: 'M4 của khóa học (lab 10.3)' }[k.id])}</td>
                <td>
                  <input type="number" step="1" aria-label={tr(`your answer for ${k.label}`, `đáp án của bạn cho ${k.label}`)} value={guess[k.id]} style={{ width: '4.5rem' }}
                    onChange={(e) => { setGuess({ ...guess, [k.id]: e.target.value }); }} />
                </td>
                {revealed ? (
                  <>
                    <td className="num">
                      <b>{r}</b>{' '}
                      {g === null ? null : g === r ? <span className="chip ok"><Translated vi="đúng như bạn nói">as you said</Translated></span> : <span className="chip fail"><Translated vi={<>bạn nói {g}</>}>you said {g}</Translated></span>}
                    </td>
                    <td className="muted">{tr(k.rule, { cuda: 'cặp của lane 0 sau butterfly xor; một cặp chỉ bị thay bởi một giá trị lớn hơn hẳn', metal: 'chỉ số lớn nhất trong các lane đang giữ cực đại', host: 'chỉ số nhỏ nhất: quét bằng phép so sánh > chặt từ chỉ số 0', torch: 'theo tài liệu: giá trị cực đại đầu tiên', m4: 'chỉ số nhỏ nhất trong các cực đại, theo quy tắc kết hợp của nó' }[k.id])} <span className="nowrap">({k.cite})</span></td>
                  </>
                ) : null}
              </tr>
            );
          })}
        </tbody>
      </table>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button className="primary" onClick={() => { setRevealed(true); setStage(0); }} disabled={revealed || planted.length === 0}><Translated vi="Hiện đáp án">Reveal</Translated></button>
        {planted.length === 0 ? <span className="muted"><Translated vi="Hãy cài ít nhất một cực đại.">Plant at least one maximum.</Translated></span> : null}
        {revealed ? (
          <>
            <button onClick={() => setStage(0)} disabled={stage === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
            <button onClick={() => setStage(stage - 1)} disabled={stage === 0}><Translated vi="Lùi">Back</Translated></button>
            <button className="primary" onClick={() => setStage(stage + 1)} disabled={stage >= last}><Translated vi="Bước">Step</Translated></button>
            <span className="muted" aria-live="polite" data-stage={stage}>{STAGE_NAMES()[stage]}</span>
          </>
        ) : null}
      </div>

      {revealed ? (
        <>
          <div className="narrate" role="status" style={{ marginBottom: '0.7rem' }} data-part="narrate">
            {isMetal ? (
              <Translated
                vi={
                  <>
                    Metal không trao đổi cặp. <code>simd_max</code> tìm giá trị cực đại, {TIE_VALUE}, ở mọi lane cùng lúc; sau đó
                    mỗi lane đưa ra chỉ số của nó nếu nó giữ giá trị đó, và -1 nếu không, rồi một <code>simd_max</code> thứ hai lấy
                    giá trị lớn nhất được đưa ra. Các lane giữ cực đại: {maxLanes.join(', ')}. Kết quả: <b>{results.metal}</b>.
                  </>
                }
              >
                <>
                  Metal does not exchange pairs. <code>simd_max</code> finds the maximum value, {TIE_VALUE}, in every lane at once;
                  then each lane offers its index if it holds that value and -1 if not, and a second <code>simd_max</code> takes the
                  largest offer. Lanes holding the maximum: {maxLanes.join(', ')}. Result: <b>{results.metal}</b>.
                </>
              </Translated>
            ) : off === null ? (
              <Translated vi={<>Mỗi lane bắt đầu từ (-FLT_MAX, -1) và chỉ giữ cột của nó nếu cột đó lớn hơn. Giờ mỗi lane giữ cột của chính nó. Bấm Bước.</>}>
                <>Each lane starts from (-FLT_MAX, -1) and keeps its column only if it is larger. Every lane now holds its own column. Press Step.</>
              </Translated>
            ) : (
              <Translated
                vi={
                  <>
                    Offset {off}: lane k đọc cặp của lane k xor {off} và chỉ lấy nó nếu giá trị đó lớn hơn hẳn. Lane{' '}
                    {lane} ghép cặp với lane {lane ^ off}
                    {st.changed.includes(lane) ? <> và lấy cặp của nó</> : <> và giữ nguyên cặp của mình</>}: giờ là ({st.lanes[lane].v.toFixed(1)}, {st.lanes[lane].i}).{' '}
                    {st.changed.length} lane đã thay đổi.
                    {off === 1 ? <> Lane 0 giữ chỉ số <b>{st.lanes[0].i}</b>, chỉ số mà kernel ghi ra. Không phải mọi lane đều thống nhất: butterfly đưa giá trị cực đại đến mọi lane, nhưng không phải cùng một chỉ số.</> : null}
                  </>
                }
              >
                <>
                  Offset {off}: lane k reads the pair of lane k xor {off} and takes it only if that value is strictly larger. Lane{' '}
                  {lane} paired with lane {lane ^ off}
                  {st.changed.includes(lane) ? <> and took its pair</> : <> and kept its own</>}: now ({st.lanes[lane].v.toFixed(1)}, {st.lanes[lane].i}).{' '}
                  {st.changed.length} lanes changed.
                  {off === 1 ? <> Lane 0 holds index <b>{st.lanes[0].i}</b>, which the kernel writes. Not every lane agrees: the butterfly gives every lane the maximum value, but not the same index.</> : null}
                </>
              </Translated>
            )}
          </div>
          <div data-part="lanes" style={{ display: 'grid', gridTemplateColumns: 'repeat(8, minmax(0, 1fr))', gap: '0.25rem', maxWidth: '40rem' }}>
            {(isMetal ? metal : st.lanes).map((p, k) => {
              const hot = isMetal ? maxLanes.includes(k) : st.changed.includes(k);
              const partner = !isMetal && off !== null && k === (lane ^ off);
              const max = p.v === TIE_VALUE;
              return (
                <button key={k} data-lane={k} data-idx={p.i} data-hot={hot ? 1 : 0} data-sel={k === lane ? 1 : 0} onClick={() => setLane(k)}
                  style={{
                    padding: '0.2rem 0', fontVariantNumeric: 'tabular-nums', lineHeight: 1.25,
                    background: hot ? 'var(--reg-wash)' : undefined,
                    outline: k === lane ? '2px solid var(--ink)' : partner ? '2px dashed var(--reg)' : undefined,
                    outlineOffset: -1,
                  }}
                  title={tr(`lane ${k}`, `lane ${k}`)}>
                  <span className="muted" style={{ fontSize: '0.68rem', display: 'block', fontWeight: k === 0 && !isMetal ? 700 : 400 }}>{k}{k === 0 && !isMetal ? ' ✎' : ''}</span>
                  <span className="t-reg" style={{ fontWeight: max ? 700 : 400 }}>{p.i < 0 ? '–' : p.v.toFixed(1)}</span>
                  <span style={{ display: 'block', fontSize: '0.8rem' }}>i {p.i}</span>
                </button>
              );
            })}
          </div>
          <Translated
            vi={
              <p className="muted" style={{ margin: '0.6rem 0 0' }}>
                Mỗi ô là một lane, số thứ tự ghi ở dòng trên cùng. Bấm vào một ô để theo dõi nó (viền liền); lane đối tác của nó ở offset này có viền đứt. Lane 0, đánh dấu ✎, ghi kết quả. Các lane tô nền {isMetal ? 'đang giữ cực đại' : 'đã thay đổi ở offset này'}.
                Các giá trị là thanh ghi, mỗi lane một cặp.
              </p>
            }
          >
            <p className="muted" style={{ margin: '0.6rem 0 0' }}>
              Each box is a lane, numbered in its top line. Click one to follow it (solid outline); its partner at this offset is dashed. Lane 0, marked ✎, writes the result. Shaded lanes {isMetal ? 'hold the maximum' : 'changed at this offset'}.
              Values are registers, one pair per lane.
            </p>
          </Translated>
        </>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- where each stage runs
function WhereItRuns() {
  const [sys, setSys] = useState<System>('llama-default');
  const [pick, setPick] = useState<Record<string, Bin>>({});
  const [checked, setChecked] = useState<Record<string, boolean>>({});
  const key = (s: System, st: Stage) => `${s}/${st}`;
  const done = checked[sys];
  const all = STAGE_LIST.every((st) => pick[key(sys, st)]);
  const right = STAGE_LIST.filter((st) => pick[key(sys, st)] === cellOf(sys, st)!.bin).length;
  return (
    <Bench
      title={tr('Where each stage runs', 'Mỗi giai đoạn chạy ở đâu')}
      task={
        <Translated vi="Với mỗi hệ thống, đặt từng giai đoạn vào nơi mã của nó chạy, rồi kiểm tra với mã nguồn.">
          For each system, put every stage where its code runs, then check against the source.
        </Translated>
      }
      foot={
        <Translated vi={<>Mỗi ô được xác định bởi một mục dữ kiện, trích dẫn ngay bên cạnh. Một giai đoạn mà mã nguồn của dự án không xác định sẽ bị bỏ ra, không đoán.</>}>
          <>Each cell is settled by a facts item, cited beside it. A stage a project does not settle in its source would be left out, not guessed.</>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', flexWrap: 'wrap' }}>
        <span className="seg" role="group" aria-label={tr('System', 'Hệ thống')} style={{ flexWrap: 'wrap' }}>
          {SYSTEMS.map((s) => (
            <button key={s.id} aria-pressed={sys === s.id} onClick={() => setSys(s.id)}>
              {tr(s.label, { 'llama-default': 'llama.cpp, mặc định', 'llama-bs': 'llama.cpp với -bs', transformers: 'transformers generate', 'mlx-lm': 'mlx-lm' }[s.id])}{checked[s.id] ? ' ✓' : ''}
            </button>
          ))}
        </span>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0, background: 'var(--page)', minWidth: '36rem' }} data-part="where" data-system={sys}>
          <thead>
            <tr><th style={{ paddingLeft: '0.6rem' }}><Translated vi="Giai đoạn">Stage</Translated></th><th><Translated vi="Chạy dưới dạng">Runs as</Translated></th>{done ? <th><Translated vi="Nguồn">Source</Translated></th> : null}</tr>
          </thead>
          <tbody>
            {STAGE_LIST.map((st) => {
              const c = cellOf(sys, st)!;
              const p = pick[key(sys, st)];
              return (
                <tr key={st} data-stage={st} data-verdict={done ? (p === c.bin ? 'right' : 'wrong') : undefined}>
                  <td style={{ paddingLeft: '0.6rem' }}>{tr(st, { penalties: 'hình phạt', 'top-k': 'top-k', 'top-p': 'top-p', temperature: 'nhiệt độ', draw: 'phép rút' }[st])}</td>
                  <td>
                    <span className="seg" role="group" aria-label={tr(`${st} runs as`, `${st} chạy dưới dạng`)} style={{ flexWrap: 'wrap' }}>
                      {BINS.map((b) => (
                        <button key={b.id} data-bin={b.id} aria-pressed={p === b.id} disabled={done}
                          onClick={() => setPick({ ...pick, [key(sys, st)]: b.id })}>{tr(b.label, { host: 'mã CPU trên host', kernels: 'graph GPU của dự án (mỗi op một kernel backend)', framework: 'tensor op của framework trên thiết bị chứa logits' }[b.id])}</button>
                      ))}
                    </span>
                  </td>
                  {done ? (
                    <td>
                      {p === c.bin ? <span className="chip ok"><Translated vi="đúng">right</Translated></span> : <span className="chip fail">{tr(BINS.find((b) => b.id === c.bin)!.label, { host: 'mã CPU trên host', kernels: 'graph GPU của dự án (mỗi op một kernel backend)', framework: 'tensor op của framework trên thiết bị chứa logits' }[c.bin])}</span>}{' '}
                      <span className="muted">{c.evidence} ({c.cite})</span>
                    </td>
                  ) : null}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="row" style={{ marginTop: '0.7rem' }}>
        <button className="primary" disabled={!all || done} onClick={() => setChecked({ ...checked, [sys]: true })}><Translated vi="Kiểm tra">Check</Translated></button>
        {done ? (
          <>
            <span data-part="score"><Translated vi={<>Đúng {right} trên {STAGE_LIST.length}.</>}>{right} of {STAGE_LIST.length} right.</Translated></span>
            <button onClick={() => { setChecked({ ...checked, [sys]: false }); }}><Translated vi="Thử lại">Try again</Translated></button>
          </>
        ) : !all ? <span className="muted"><Translated vi="Hãy đặt cả năm giai đoạn trước.">Place all five stages first.</Translated></span> : null}
      </div>
      {done && ORDER[sys] ? (
        <p className="muted" style={{ margin: '0.6rem 0 0' }} data-part="order"><Translated vi={<>Thứ tự trong hệ thống này: {({ 'llama-default': 'hình phạt → DRY → top-n-sigma → top-k → typical → top-p → min-p → XTC → nhiệt độ → dist hoặc greedy', transformers: 'hình phạt lặp lại và các processor khác → nhiệt độ → top-k → top-p → min-p → softmax → torch.multinomial', 'mlx-lm': 'logits processor (hình phạt) → top-p → min-p → XTC → top-k → nhiệt độ và categorical' } as Partial<Record<System, string>>)[sys] ?? ORDER[sys]!.join(' → ')}.</>}>Order in this system: {ORDER[sys]!.join(' → ')}.</Translated></p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the M4 listing
function M4Listing() {
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  // The labs page's words (LabKit's solutionTabStatus), from this machine's runs.
  const { reports } = useReports();
  const STATUS = { cu: solutionTabStatus(reports, LAB, 'kernel.cu'), metal: solutionTabStatus(reports, LAB, 'kernel.metal') };
  const metal = useLabFile(LAB + '/solution/kernel.metal');
  const a = cu ? functionSource(cu, 'argmaxCombine') : null;
  const w = cu ? functionSource(cu, 'warpArgmax') : null;
  const ma = metal ? functionSource(metal, 'argmax_combine') : null;
  const mw = metal ? functionSource(metal, 'simd_argmax') : null;
  if (!a || !w || !ma || !mw) return <p className="muted"><Translated vi={<>Phép kết hợp được nạp từ <code>labs/{LAB}/solution/</code>.</>}>The combine is loaded from <code>labs/{LAB}/solution/</code>.</Translated></p>;
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: a.text + '\n\n' + w.text, tiers: TIERS, status: STATUS.cu },
        { label: tr('Metal kernel', 'Kernel Metal'), code: ma.text + '\n\n' + mw.text, tiers: TIERS, status: STATUS.metal },
      ]}
      caption={
        <Translated
          vi={
            <>
              Quy tắc phân định hòa của M4, từ lab 10.3 (<code>labs/{LAB}/solution/kernel.cu</code>, dòng {a.firstLine} đến {w.lastLine}, và{' '}
              <code>kernel.metal</code>). Các giá trị bằng nhau đi về chỉ số nhỏ hơn, nên thứ tự các lane gặp nhau không quan trọng.
            </>
          }
        >
          <>
            M4's tie rule, from lab 10.3 (<code>labs/{LAB}/solution/kernel.cu</code>, lines {a.firstLine} to {w.lastLine}, and{' '}
            <code>kernel.metal</code>). Equal values go to the lower index, so the order in which lanes meet does not matter.
          </>
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- the section
export function SX36() {
  const q = questionsSX36;
  return (
    <>
      <Translated
        vi={
          <p>
            Module X1 đã xây một sampler; phần này đọc những sampler đang được phát hành. Với mỗi dự án, bạn sẽ tìm sampler và
            argmax, nói mỗi cái chạy ở đâu theo mặc định, và dự đoán mỗi cái chọn token nào khi hai logit bằng nhau tuyệt đối.
            Phần cuối là chỗ ba cách cài đặt trong cùng một dự án không thống nhất với nhau, và là chỗ bài kiểm thử M10 của bạn
            phải cẩn thận.
          </p>
        }
      >
        <p>
          Module X1 built a sampler; this section reads the ones that ship. For each project you will find the sampler and
          the argmax, say where each one runs by default, and predict which token each picks when two logits are exactly
          equal. The last part is where three implementations inside one project disagree, and where your M10 test has to be
          careful.
        </p>
      </Translated>
      <Beyond title={tr("Ours, not the book's", 'Của khóa học này, không phải của sách')}>
        <Translated
          vi={
            <p>
              Lấy mẫu không có trong sách. Mọi nhận định về mã của một dự án ở đây đều được đọc trong các lượt kiểm tra dữ kiện
              ngày 2026-11-20 ({X1}, {C10}, {X3} B8, D9). Thao tác warp shuffle mà argmax CUDA dùng là của sách.
            </p>
          }
        >
          <p>
            Sampling is not in the book. Every statement about a project's code here was read in the facts passes on
            2026-11-20 ({X1}, {C10}, {X3} B8, D9). The warp shuffle the CUDA argmax uses is the book's.
          </p>
        </Translated>
      </Beyond>
      <Source section="10.7" pages="237–241" />

      <h2><Translated vi="llama.cpp: mặc định là một chuỗi trên host">llama.cpp: a host chain by default</Translated></h2>
      <Translated
        vi={
          <p>
            Chuỗi sampler nằm trong <code>src/llama-sampler.cpp</code>, được cấu hình bởi <code>common_params_sampling</code>. Thứ
            tự mặc định là penalties, DRY, top-n-sigma, top-k, typical, top-p, min-p, XTC và temperature, rồi đến phép rút cuối
            cùng (<code>dist</code>) hoặc, ở temperature 0, <code>greedy</code>. Các giá trị mặc định là top-k 40, top-p 0.95, min-p 0.05
            và temperature 0.8. Temperature đứng sau mọi bước cắt bớt. Đây là thứ tự mà mốc dự án M5 sao chép.
          </p>
        }
      >
        <p>
          The sampler chain lives in <code>src/llama-sampler.cpp</code>, configured by <code>common_params_sampling</code>. The
          default order is penalties, DRY, top-n-sigma, top-k, typical, top-p, min-p, XTC and temperature, then the final
          draw (<code>dist</code>) or, at temperature 0, <code>greedy</code>. The defaults are top-k 40, top-p 0.95, min-p 0.05
          and temperature 0.8. Temperature comes after every truncation. This is the order milestone M5 copies.
        </p>
      </Translated>
      <Today checked={tr(`2026-11-20 in the facts pass (${X1} D1, D3)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${X1} D1, D3)`)}>
        <Translated
          vi={
            <p>
              <code>bool backend_sampling = false;</code> cho quá trình sinh, <code>true</code> cho bản nháp speculative;{' '}
              <code>-bs</code>: "enable backend sampling (experimental) (default: disabled)".
            </p>
          }
        >
          <p>
            <code>bool backend_sampling = false;</code> for generation, <code>true</code> for the speculative draft;{' '}
            <code>-bs</code>: "enable backend sampling (experimental) (default: disabled)".
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Với <code>-bs</code>, các giai đoạn đó trở thành các phép toán trong graph, được các kernel của backend chạy như mọi op
            khác. Khi đó phép rút cuối cùng là <code>ggml_soft_max</code>, <code>ggml_cumsum</code>, <code>ggml_step</code> và{' '}
            <code>ggml_sum</code>: đếm xem có bao nhiêu xác suất tích lũy vượt quá u, và token là độ dài hàng trừ đi số đếm đó.
            Số đều u vẫn được rút trên host, từ cùng <code>mt19937</code>, và được tải lên ở mỗi bước. Top-k trên GPU của nó là{' '}
            <code>top-k.cu</code>, một phép chọn radix 8 bit, và trên Metal là phép sắp xếp threadgroup và top-k của{' '}
            <code>argsort.metal</code>: phép chọn của phần <a href={href('x1', 'X1.4')}>X1.4</a>.
          </p>
        }
      >
        <p>
          With <code>-bs</code> the same stages become graph operations, run by the backend's kernels like any other op. The
          final draw is then <code>ggml_soft_max</code>, <code>ggml_cumsum</code>, <code>ggml_step</code> and{' '}
          <code>ggml_sum</code>: count how many cumulative probabilities exceed u, and the token is the row length minus that
          count. The uniform number u is still drawn on the host, from the same <code>mt19937</code>, and uploaded each step.
          Its GPU top-k is <code>top-k.cu</code>, an 8-bit radix select, and on Metal the threadgroup sort and top-k of{' '}
          <code>argsort.metal</code>: the selection of section <a href={href('x1', 'X1.4')}>X1.4</a>.
        </p>
      </Translated>
      <Today checked={tr(`2026-11-20 in the facts pass (${X1} D3; notes/today/ch14-facts.md D2, D3)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${X1} D3; notes/today/ch14-facts.md D2, D3)`)}>
        <Translated
          vi={
            <p>
              Phép rút trên host: ứng viên đầu tiên có tổng tích lũy ít nhất bằng u nhân tổng. Phép rút trên backend: ứng viên đầu
              tiên có xác suất tích lũy bao gồm lớn hơn hẳn u. Ở đúng biên, hai cách lệch nhau một token.
            </p>
          }
        >
          <p>
            Host draw: the first candidate whose running sum is at least u times the total. Backend draw: the first whose
            inclusive cumulative probability is strictly greater than u. On an exact boundary the two differ by one token.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Vậy câu hỏi "llama.cpp lấy mẫu ở đâu?" có hai câu trả lời, và mọi dự án khác cũng vậy. Hãy sắp xếp chúng trước khi đọc
            tiếp.
          </p>
        }
      >
        <p>
          So "where does llama.cpp sample?" has two answers, and so does every other project. Sort them before you read on.
        </p>
      </Translated>
      <WhereItRuns />
      <Ask q={q['x3/X3.6/q-default-place']} />

      <h2><Translated vi="Argmax: một dự án, ba quy tắc phân định hòa">Argmax: one project, three tie rules</Translated></h2>
      <Translated
        vi={
          <p>
            Giải mã greedy là một phép argmax, và argmax cần một quy tắc cho các giá trị bằng nhau. llama.cpp có ba quy tắc. Kernel
            CUDA của nó, <code>argmax_f32</code>, giao cho mỗi luồng (thread) một phần của hàng theo bước nhảy (stride) rồi chạy một
            butterfly với <code>__shfl_xor_sync</code>: ở mỗi offset, mọi lane đọc cặp của lane có số thứ tự khác nó đúng một bit, và
            chỉ thay cặp của mình <code>if (val &gt; maxval)</code>. Kernel Metal của nó lấy <code>simd_max</code> của các giá trị rồi{' '}
            <code>simd_max</code> của chỉ số các lane đang giữ cực đại. Sampler greedy trên host của nó quét từ chỉ số 0 với phép so
            sánh chặt.
          </p>
        }
      >
        <p>
          Greedy decoding is an argmax, and an argmax needs a rule for equal values. llama.cpp has three. Its CUDA kernel,{' '}
          <code>argmax_f32</code>, gives each thread a strided share of the row and then runs a butterfly with{' '}
          <code>__shfl_xor_sync</code>: at each offset every lane reads the pair of the lane whose number differs in one bit,
          and replaces its own only <code>if (val &gt; maxval)</code>. Its Metal kernel takes <code>simd_max</code> of the
          values and then <code>simd_max</code> of the indices of the lanes that hold the maximum. Its host greedy sampler
          scans from index 0 with a strict comparison.
        </p>
      </Translated>
      <Today checked={tr(`2026-11-20 in the facts pass (${C10} D2)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${C10} D2)`)}>
        <Translated
          vi={
            <p>
              <code>argmax.cu</code> cũng là file HIP: llama.cpp biên dịch <code>ggml-cuda/*.cu</code> thành HIP ({X3} B9).
              Metal có kernel riêng trong <code>kernels/misc.metal</code>.
            </p>
          }
        >
          <p>
            <code>argmax.cu</code> is also the HIP file: llama.cpp compiles <code>ggml-cuda/*.cu</code> as HIP ({X3} B9).
            Metal has its own kernel in <code>kernels/misc.metal</code>.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Phép rút gọn (reduction) warp của sách chạy các bước nhảy 16, 8, 4, 2, 1 bằng shuffle-down và để tổng ở lane 0.
            Butterfly xor đi qua đúng các offset đó nhưng mọi lane đều kết thúc với một kết quả, và với phép so sánh chặt thì các
            lane không nhất thiết thống nhất về chỉ số. Hãy dự đoán trước khi chạy từng bước.
          </p>
        }
      >
        <p>
          The book's warp reduction runs its strides 16, 8, 4, 2, 1 with shuffle-down and leaves the sum in lane 0. The xor
          butterfly visits the same offsets but every lane ends with a result, and with a strict comparison the lanes need not
          agree on the index. Predict before you step through it.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một warp, mỗi lane một cột: vòng lặp theo bước nhảy trên một hàng dài hơn và giai đoạn thứ hai giữa các warp bị bỏ
              ra. Điểm khởi đầu của mỗi lane trên Metal, (−∞, −1) với phép so sánh chặt, là giả định của chúng tôi; lượt kiểm tra dữ
              kiện chỉ trích hai dòng <code>simd_max</code>, và Metal không ghi trong tài liệu cách <code>simd_max</code> xử lý NaN.
            </p>
          }
        >
          <p>
            One warp, one column per lane: the strided loop over a longer row and the second stage across warps are left
            out. Metal's per-lane start, (−∞, −1) with a strict comparison, is our assumption; the facts pass quotes only the
            two <code>simd_max</code> lines, and Metal does not document how <code>simd_max</code> treats NaN.
          </p>
        </Translated>
      </Simplification>
      <Ask q={q['x3/X3.6/q-tie-cuda']} label={tr('Predict, then check', 'Dự đoán, rồi kiểm tra')} />
      <TieStepper />
      <Translated
        vi={
          <p>
            Điều cần để ý. Metal luôn trả về chỉ số hòa lớn nhất, còn host, torch.argmax và M4 trả về chỉ số nhỏ nhất. Đáp án của
            CUDA không thuộc bên nào: với cột 1 và 2 là 2, với 0 và 1 là 0, với 0 và 16 là 0. Nó phụ thuộc vào vị trí các cột hòa
            nhau trong butterfly, tức là một tính chất của vị trí lane, không phải của thứ tự.
          </p>
        }
      >
        <p>
          What to notice. Metal always returns the largest tied index, and the host, torch.argmax and M4 the smallest. The CUDA
          answer is neither: for columns 1 and 2 it is 2, for 0 and 1 it is 0, for 0 and 16 it is 0. It depends on where the
          tied columns sit in the butterfly, which is a property of lane positions, not of order.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Hòa chính xác giữa các logit float là hiếm trong một mô hình thật. Trong kiểm thử thì không hiếm: một mô hình tổng
              hợp, một mô hình đã lượng tử hóa, hay một hàng bị che bằng âm vô cùng đều tạo ra chúng, và một hàng toàn −∞ khiến
              kernel CUDA của llama.cpp trả về −1.
            </p>
          }
        >
          <p>
            Exact ties in float logits are rare in a real model. They are not rare in tests: a synthetic model, a quantised
            one, or a masked row of minus infinity produces them, and a row that is entirely −∞ makes llama.cpp's CUDA kernel
            return −1.
          </p>
        </Translated>
      </Aside>
      <Ask q={q['x3/X3.6/q-tie-metal']} />
      <M4Listing />

      <h2><Translated vi="transformers và PyTorch">transformers and PyTorch</Translated></h2>
      <Translated
        vi={
          <p>
            transformers biến logits thành một token trong <code>generation/utils.py</code>. Các logits processor chạy trước, trong
            đó có repetition penalty; rồi đến các warper, theo thứ tự temperature, top-k, top-p, min-p, ngược với vị trí đặt
            temperature của llama.cpp. Phép rút là <code>torch.multinomial(probs, num_samples=1)</code>. Greedy là{' '}
            <code>torch.argmax(next_token_scores, dim=-1)</code>, mà tài liệu của nó cam kết trả về "the indices of the first maximal
            value".
          </p>
        }
      >
        <p>
          transformers turns logits into a token in <code>generation/utils.py</code>. Logits processors run first, the
          repetition penalty among them; then the warpers, in the order temperature, top-k, top-p, min-p, the reverse of
          llama.cpp's placement of temperature. The draw is <code>torch.multinomial(probs, num_samples=1)</code>. Greedy is{' '}
          <code>torch.argmax(next_token_scores, dim=-1)</code>, whose documentation promises "the indices of the first maximal
          value".
        </p>
      </Translated>
      <Today checked={tr(`2026-11-20 in the facts pass (${X1} D4, D5; ${C10} D4)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${X1} D4, D5; ${C10} D4)`)}>
        <Translated
          vi={
            <p>
              <code>multinomial_out</code>: "Fast-path for no replacement or if only one sample is drawn", tính{' '}
              <code>argmax(p / q)</code> với <code>q ~ Exp(1)</code> trên CPU, CUDA, MPS và XPU.
            </p>
          }
        >
          <p>
            <code>multinomial_out</code>: "Fast-path for no replacement or if only one sample is drawn", computing{' '}
            <code>argmax(p / q)</code> with <code>q ~ Exp(1)</code> on CPU, CUDA, MPS and XPU.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Điều bất ngờ nằm ở phép rút. <code>torch.multinomial(probs, 1)</code> không phải là "rút u, duyệt CDF". Nó rút một số mũ
            cho mỗi token và lấy argmax của p / q, tức mẹo Gumbel-max dưới một dạng khác. Kernel CDF ngược trên CUDA của PyTorch,{' '}
            <code>native/cuda/MultinomialKernel.cu</code>, chỉ chạy khi lấy mẫu có hoàn lại và nhiều hơn một mẫu. Mọi giai đoạn, kể cả
            phép rút, đều chạy trên thiết bị chứa logits dưới dạng các tensor op của framework.
          </p>
        }
      >
        <p>
          The draw is the surprise. <code>torch.multinomial(probs, 1)</code> is not "draw u, walk the CDF". It draws one
          exponential number per token and takes the argmax of p / q, the Gumbel-max trick in another form. PyTorch's CUDA
          inverse-CDF kernel, <code>native/cuda/MultinomialKernel.cu</code>, runs only with replacement and more than one
          sample. Every stage, the draw included, runs on the logits' device as framework tensor operations.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Nếu bạn gọi <code>model.generate(do_sample=False)</code>, đây là thứ bạn nhận được: các processor, rồi argmax trên các
            logits đã xử lý, lấy cực đại đầu tiên khi hòa.
          </p>
        }
      >
        <p>
          If you call <code>model.generate(do_sample=False)</code>, this is what you get: the processors, then argmax over the
          processed logits, first maximum on a tie.
        </p>
      </Translated>
      <Ask q={q['x3/X3.6/q-multinomial']} />

      <h2><Translated vi="MLX và mlx-lm">MLX and mlx-lm</Translated></h2>
      <Translated
        vi={
          <p>
            Số ngẫu nhiên của MLX đến từ một bộ sinh Threefry dựa trên bộ đếm và có thể tách nhánh, theo thiết kế của JAX. Kernel
            Metal là <code>kernels/random.metal</code>; backend CUDA của nó có bản song sinh, <code>backend/cuda/random.cu</code>,
            với cùng <code>threefry2x32_hash</code>. <code>mx.random.categorical</code> dùng CDF ngược cho một phân phối đơn lẻ có
            sample shape và dùng Gumbel-max trong các trường hợp khác, nên theo cách chúng tôi đọc, kích thước batch có thể làm
            thay đổi token được rút với cùng một khóa.
          </p>
        }
      >
        <p>
          MLX's random numbers come from a splittable, counter-based Threefry generator, following JAX's design. The Metal
          kernel is <code>kernels/random.metal</code>; its CUDA backend has the twin, <code>backend/cuda/random.cu</code>,
          with the same <code>threefry2x32_hash</code>. <code>mx.random.categorical</code> uses an inverse CDF for a single
          distribution with a sample shape and Gumbel-max otherwise, so by our reading the batch size can change the token
          drawn for the same key.
        </p>
      </Translated>
      <Today checked={tr(`2026-11-20 in the facts pass (${X3} B8; ${X1} D7, D8)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${X3} B8; ${X1} D7, D8)`)}>
        <Translated
          vi={
            <p>
              mlx-lm <code>make_sampler</code>: top-p, rồi min-p, XTC, top-k, rồi <code>categorical_sampling</code> với
              temperature; min-p dưới dạng <code>top_logprobs + math.log(min_p)</code>. Tất cả là op của MLX, chạy trên GPU dưới Metal.
            </p>
          }
        >
          <p>
            mlx-lm <code>make_sampler</code>: top-p, then min-p, XTC, top-k, then <code>categorical_sampling</code> with
            temperature; min-p as <code>top_logprobs + math.log(min_p)</code>. All MLX ops, on the GPU under Metal.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            <code>sample_utils.py</code> của mlx-lm áp mọi bước cắt bớt trước temperature, giống llama.cpp, nhưng top-p trước
            top-k, khác cả hai dự án kia. Min-p được so trong không gian log: log p<sub>i</sub> so với log-xác suất lớn nhất cộng
            ln(min-p), cùng phép kiểm tra với tỉ lệ xác suất của transformers. Temperature 0 trả về <code>mx.argmax</code>.
          </p>
        }
      >
        <p>
          mlx-lm's <code>sample_utils.py</code> applies every truncation before temperature, like llama.cpp, but top-p before
          top-k, unlike both other projects. Min-p is compared in log space: log p<sub>i</sub> against the top log-probability
          plus ln(min-p), the same test as transformers' probability ratio. Temperature 0 returns <code>mx.argmax</code>.
        </p>
      </Translated>

      <h2><Translated vi="Khóa học làm gì, và vì sao">What the course does, and why</Translated></h2>
      <Translated
        vi={
          <p>
            M5 lấy thứ tự của llama.cpp (cắt bớt trước, temperature sau cùng). Argmax của M4 giữ chỉ số nhỏ nhất khi hòa, trên mọi
            backend, vì lab 10.3 kiểm tra nó bằng các giá trị hòa được cài sẵn. M5 suy ra u trên device từ Philox4x32-10, với khóa
            là seed, stream và bước, nên CUDA, HIP, Metal và bản tham chiếu CPU rút cùng một số (phần{' '}
            <a href={href('x1', 'X1.7')}>X1.7</a>). llama.cpp có cùng u trên mọi backend theo cách khác: nó rút u trên host rồi
            tải lên.
          </p>
        }
      >
        <p>
          M5 takes llama.cpp's order (truncation first, temperature last). M4's argmax keeps the lowest index on a tie, on every
          backend, because lab 10.3 tests it with planted ties. M5 derives u on the device from Philox4x32-10, keyed by seed,
          stream and step, so CUDA, HIP, Metal and the CPU reference draw the same number (section{' '}
          <a href={href('x1', 'X1.7')}>X1.7</a>). llama.cpp gets the same u on every backend another way: it draws it on the
          host and uploads it.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Với bản tham chiếu của M10, hệ quả là trực tiếp. "Cực đại đầu tiên" đúng với <code>sample_argmax</code> của llama2.c và
            với torch.argmax, không đúng với các kernel GPU của llama.cpp. Bản tham chiếu CPU của M10 phải dùng quy tắc của M4, chỉ
            số nhỏ nhất, và bài kiểm thử của nó phải báo khoảng cách giữa hai giá trị lớn nhất ở mọi bước, để một trường hợp gần
            hòa được đánh dấu thay vì bị tính là lỗi.
          </p>
        }
      >
        <p>
          For M10's reference the consequence is direct. "First maximum" holds for llama2.c's <code>sample_argmax</code> and for
          torch.argmax, not for llama.cpp's GPU kernels. M10's CPU reference must use M4's rule, the lowest index, and its test
          must report the top-two margin at every step, so that a near tie is flagged rather than counted as a failure.
        </p>
      </Translated>
      <Today checked={tr(`2026-11-20 in the facts pass (${X3} D7, D9)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${X3} D7, D9)`)}>
        <Translated
          vi={
            <p>
              <code>test-backend-ops</code> của llama.cpp so sánh các backend bằng một sai số chuẩn hóa, ngưỡng mặc định 1e-7 cho
              mỗi op, không phải bằng phép so sánh bằng nhau từng phần tử.
            </p>
          }
        >
          <p>
            llama.cpp's <code>test-backend-ops</code> compares backends with a normalised error, default threshold 1e-7 per
            op, not with elementwise equality.
          </p>
        </Translated>
      </Today>
      <Ask q={q['x3/X3.6/q-compare-greedy']} />
      <Translated
        vi={
          <p>
            Lab X3.2 chạy bài kiểm thử đó trên chính các kernel của khóa học: xem <a href={href('x3', 'labs')}>trang lab</a>.
          </p>
        }
      >
        <p>
          Lab X3.2 runs that test on the course's own kernels: see the <a href={href('x3', 'labs')}>labs page</a>.
        </p>
      </Translated>

      <h2><Translated vi="Trên HIP và Metal">On HIP and Metal</Translated></h2>
      <Translated
        vi={
          <p>
            Bản build HIP của llama.cpp biên dịch <code>argmax.cu</code> và <code>top-k.cu</code> từ thư mục CUDA. llama.cpp định
            nghĩa một warp logic 32 lane và truy vấn độ rộng vật lý riêng; cách butterfly của <code>argmax.cu</code> ánh xạ lên một
            wavefront rộng 64 chưa được kiểm tra, nên hãy coi hành vi khi hòa của nó trên AMD là unverified. MLX hoàn toàn không có
            backend HIP. Trên Metal, argmax của llama.cpp là một kernel riêng với một quy tắc phân định hòa khác, và đó là bài học
            chung của phần này: hai backend của một dự án là hai cách cài đặt, và trường hợp hòa là nơi bạn thấy điều đó.
          </p>
        }
      >
        <p>
          llama.cpp's HIP build compiles <code>argmax.cu</code> and <code>top-k.cu</code> from the CUDA directory. llama.cpp
          defines a logical warp of 32 lanes and queries the physical width separately; how <code>argmax.cu</code>'s butterfly
          maps onto a 64-wide wavefront was not checked, so treat its tie behaviour on AMD as unverified. MLX has no HIP
          backend at all. On Metal, llama.cpp's argmax is a separate kernel with a different tie rule, which is the general lesson of
          this section: two backends of one project are two implementations, and a tie is where you see it.
        </p>
      </Translated>
      <Today checked={tr(`2026-11-20 in the facts pass (${X3} B9; notes/today/ch04-facts.md D1)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${X3} B9; notes/today/ch04-facts.md D1)`)}>
        <Translated
          vi={
            <p>
              llama.cpp: <code>#define WARP_SIZE 32</code>, một độ rộng logic, với độ rộng vật lý được truy vấn riêng (64 trên GFX8
              và GFX9). MLX: <code>mlx/backend/</code> không có thư mục HIP hay ROCm.
            </p>
          }
        >
          <p>
            llama.cpp: <code>#define WARP_SIZE 32</code>, a logical width, with the physical width queried separately
            (64 on GFX8 and GFX9). MLX: <code>mlx/backend/</code> has no HIP or ROCm directory.
          </p>
        </Translated>
      </Today>
    </>
  );
}
