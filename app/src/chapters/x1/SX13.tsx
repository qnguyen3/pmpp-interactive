import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Bench, Beyond, NoteGroup, Simplification, Source, Today, type Tier } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { num, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { SEGMENT_MODELS, type SegmentModelId } from '../../sim/coalesce';
import {
  MAPPINGS,
  MAPPING_LABEL,
  PENALTY_ORDER_LABEL,
  PEN_CONTEXT,
  PEN_LOGITS,
  PEN_VOCAB,
  QWEN3_VOCAB,
  REP_RULES,
  REP_RULE_FACT,
  REP_RULE_LABEL,
  applyPenalties,
  argmaxLowest,
  bytesPerLogit,
  countTokens,
  f32,
  mappingAccess,
  penaltyEffects,
  segmentFor,
  softmaxDouble,
  type Mapping,
  type PenaltyOrder,
  type RepRule,
} from '../../sim/penalties';
import { functionSource } from '../../sim/toolchain';

const FACTS = 'notes/today/x1-facts.md';
const LAB = 'ch09/lab3_tokencounts';
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const TIERS: Record<string, Tier> = { logits: 'global', counts: 'global', c: 'reg', i: 'reg', n: 'reg' };
const fmtInt = (v: number) => v.toLocaleString('en-US');
const f3 = (v: number) => (Object.is(v, -0) ? '0.000' : v.toFixed(3));

// ---------------------------------------------------------------- questions
export const questionsSX13: Record<string, Question> = {
  'x1/X1.3/q-zero': {
    kind: 'choice',
    id: 'x1/X1.3/q-zero',
    prompt: (
      <Translated
        vi={
          <>
            Logit của một token lặp lại đúng bằng 0.0 và repetition penalty là 1.25. llama.cpp nhân khi logit ≤ 0;
            transformers nhân khi nó &lt; 0 và chia trong trường hợp còn lại. Hai bên có cho kết quả khác nhau không?
          </>
        }
      >
        A repeated token's logit is exactly 0.0 and the repetition penalty is 1.25. llama.cpp multiplies when the logit is
        ≤ 0; transformers multiplies when it is &lt; 0 and divides otherwise. Do they disagree?
      </Translated>
    ),
    options: [
      { label: <Translated vi="Không: cả hai đều cho 0">No: both give 0</Translated>, correct: true },
      {
        label: (
          <Translated vi="Có: llama.cpp cho 0, transformers cho −0.25">
            Yes: llama.cpp gives 0, transformers gives −0.25
          </Translated>
        ),
        why: (
          <Translated vi="Ở đây transformers chia, và 0 / 1.25 là 0.">
            transformers divides here, and 0 / 1.25 is 0.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Có: transformers cho 0.8">Yes: transformers gives 0.8</Translated>,
        why: (
          <Translated vi="0.8 sẽ là 1 / 1.25. Logit là 0, không phải 1.">
            0.8 would be 1 / 1.25. The logit is 0, not 1.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Có: llama.cpp cho −1.25">Yes: llama.cpp gives −1.25</Translated>,
        why: (
          <Translated vi="0 × 1.25 là 0; penalty này không trừ đi gì cả.">
            0 × 1.25 is 0; nothing is subtracted by this penalty.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            0 × 1.25 = 0 và 0 / 1.25 = 0, nên hai quy tắc theo dấu không thể khác nhau tại 0. Chúng chỉ khác ở chỗ nhánh nào
            xử lý một giá trị mà kết quả đằng nào cũng như nhau. Quy tắc đáng quan tâm là quy tắc kia: của CTRL, chia cả các
            logit âm.
          </p>
        }
      >
        <p>
          0 × 1.25 = 0 and 0 / 1.25 = 0, so the two sign rules cannot disagree at zero. They differ only in which branch
          handles a value whose result is the same either way. The rule that matters is the other one: CTRL's, which divides
          negative logits too.
        </p>
      </Translated>
    ),
    source: 'notes/today/x1-facts.md D2, D4, D8', // i18n-ignore: citation
  },
  'x1/X1.3/q-freq': {
    kind: 'numeric',
    id: 'x1/X1.3/q-freq',
    prompt: (
      <Translated
        vi={
          <>
            Logit 1.2, count 3, frequency penalty 0.1, presence penalty 0.5, repetition penalty 1 (tắt). Logit sau các
            penalty là bao nhiêu?
          </>
        }
      >
        Logit 1.2, count 3, frequency penalty 0.1, presence penalty 0.5, repetition penalty 1 (off). What is the logit
        after the penalties?
      </Translated>
    ),
    answer: 0.4,
    tol: 0.001,
    explain: (
      <Translated
        vi={
          <p>
            1.2 − (3 × 0.1 + 1 × 0.5) = 1.2 − 0.8 = 0.4. Frequency tăng theo mỗi lần lặp; presence chỉ bị tính một lần, cho bất
            kỳ token nào đã xuất hiện.
          </p>
        }
      >
        <p>
          1.2 − (3 × 0.1 + 1 × 0.5) = 1.2 − 0.8 = 0.4. Frequency grows with every repeat; presence is paid once, by any
          token that occurred at all.
        </p>
      </Translated>
    ),
    source: 'OpenAI’s formula as llama.cpp implements it: notes/today/x1-facts.md D9; notes/today/ch09-facts.md D4',
    vi: {
      source: 'Công thức của OpenAI như llama.cpp hiện thực: notes/today/x1-facts.md D9; notes/today/ch09-facts.md D4',
    },
  },
  'x1/X1.3/q-order': {
    kind: 'numeric',
    id: 'x1/X1.3/q-order',
    prompt: (
      <Translated
        vi={
          <>
            Logit 2.0, count 2, repetition 1.25, frequency 0.5, presence 0, theo thứ tự của llama.cpp (repetition trước,
            rồi đến phép trừ). Kết quả là bao nhiêu?
          </>
        }
      >
        Logit 2.0, count 2, repetition 1.25, frequency 0.5, presence 0, in llama.cpp's order (repetition first, then the
        subtraction). What is the result?
      </Translated>
    ),
    answer: 0.6,
    tol: 0.001,
    explain: (
      <Translated
        vi={
          <p>
            2.0 / 1.25 = 1.6, rồi 1.6 − 2 × 0.5 = 0.6. Thứ tự ngược lại cho (2.0 − 1.0) / 1.25 = 0.8. Phép chia và phép trừ
            không giao hoán, vì vậy hợp đồng của M5 nêu rõ thứ tự.
          </p>
        }
      >
        <p>
          2.0 / 1.25 = 1.6, then 1.6 − 2 × 0.5 = 0.6. The reverse order gives (2.0 − 1.0) / 1.25 = 0.8. Division and
          subtraction do not commute, which is why M5's contract states the order.
        </p>
      </Translated>
    ),
    source: 'notes/today/x1-facts.md D2 (the order in llama_sampler_penalties_apply)',
    vi: { source: 'notes/today/x1-facts.md D2 (thứ tự trong llama_sampler_penalties_apply)' },
  },
  'x1/X1.3/q-bytes': {
    kind: 'numeric',
    id: 'x1/X1.3/q-bytes',
    prompt: (
      <Translated
        vi={
          <>
            Repetition penalty là một kernel, và frequency cộng presence (<code>applyPenalties</code> của lab 9.3) là một
            kernel khác, mỗi kernel đọc logit và count của nó rồi ghi logit. Số byte tối thiểu được di chuyển cho mỗi logit
            với cặp kernel này?
          </>
        }
      >
        The repetition penalty as one kernel, and frequency plus presence (lab 9.3's <code>applyPenalties</code>) as
        another, each reading the logit and its count and writing the logit. Minimum bytes moved per logit for the pair?
      </Translated>
    ),
    answer: bytesPerLogit(2),
    unit: 'bytes',
    explain: (
      <Translated
        vi={
          <p>
            Mỗi lượt di chuyển 4 + 4 + 4 = 12 byte cho mỗi logit, nên hai lượt di chuyển 24. Gộp thành một lượt thì 12: một
            nửa lưu lượng của một kernel gần như không làm phép tính nào. Lab X1.3 giữ hai lượt (và một bản sao trước
            chúng), nên khoản tiết kiệm đó để dành cho bạn.
          </p>
        }
      >
        <p>
          Each pass moves 4 + 4 + 4 = 12 bytes per logit, so two passes move 24. Fused into one pass, 12: half the traffic
          of a kernel that does almost no arithmetic. Lab X1.3 keeps the two passes (and a copy before them), so that
          saving is left for you to take.
        </p>
      </Translated>
    ),
    source: 'ours',
    vi: { unit: 'byte', source: 'của khóa học này' },
  },
  'x1/X1.3/q-hf-counts': {
    kind: 'choice',
    id: 'x1/X1.3/q-hf-counts',
    prompt: (
      <Translated vi={<>Phát biểu nào về các penalty của transformers là đúng?</>}>
        <>Which statement about transformers' penalties is right?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Nó phạt mọi token trong input_ids, kể cả prompt, mỗi token một lần, và không có frequency hay presence penalty.">
            It penalises every token in input_ids, prompt included, once each, and has no frequency or presence penalty.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Nó chỉ phạt các token được sinh ra, không bao giờ phạt prompt, và điều chỉnh penalty theo số lần mỗi token xuất hiện.">
            It penalises only the generated tokens, never the prompt, and scales the penalty by how often each token appeared.
          </Translated>
        ),
        why: (
          <Translated vi="input_ids cũng chứa prompt, trừ khi có prompt_ignore_length.">
            input_ids holds the prompt too, unless prompt_ignore_length is given.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó hiện thực công thức frequency và presence của OpenAI bên cạnh repetition penalty có xét dấu.">
            It implements OpenAI's frequency and presence formula alongside its sign-aware repetition penalty.
          </Translated>
        ),
        why: (
          <Translated vi="Không từ nào trong hai từ đó xuất hiện như một tham số trong mã sinh văn bản của nó.">
            Neither word appears as a parameter in its generation code.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó đếm từng lần xuất hiện, nên một token được dùng ba lần trong ngữ cảnh bị phạt ba lần.">
            It counts each occurrence, so a token used three times in the context is penalised three times over.
          </Translated>
        ),
        why: (
          <Translated vi="Penalty được áp một lần cho mỗi token phân biệt, bất kể count là bao nhiêu.">
            The penalty is applied once per distinct token, whatever the count.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>RepetitionPenaltyLogitsProcessor</code> gom các điểm tại các id trong <code>input_ids</code> và áp penalty
            có xét dấu một lần cho mỗi id. Không có số hạng frequency hay presence, nên một sampler phải khớp với llama.cpp
            hoặc OpenAI API không thể dựng chỉ từ các processor của transformers.
          </p>
        }
      >
        <p>
          <code>RepetitionPenaltyLogitsProcessor</code> gathers the scores at the ids in <code>input_ids</code> and applies
          the sign-aware penalty once to each. There is no frequency or presence term, so a sampler that must match
          llama.cpp or the OpenAI API cannot be built from transformers' processors alone.
        </p>
      </Translated>
    ),
    source: 'notes/today/x1-facts.md D4', // i18n-ignore: citation
  },
  'x1/X1.3/q-append': {
    kind: 'choice',
    id: 'x1/X1.3/q-append',
    prompt: (
      <Translated
        vi={
          <>
            <code>appendTokens</code> của lab 9.3 thêm một token mới vào count của mỗi hàng ở mỗi bước decode, không dùng
            atomic. Vì sao điều đó an toàn?
          </>
        }
      >
        Lab 9.3's <code>appendTokens</code> adds one new token to each row's counts per decode step, with no atomic. Why is
        that safe?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Mỗi hàng do một luồng sở hữu, nên không có hai luồng nào cùng đọc và ghi một count trong một lần khởi chạy.">
            One thread owns each row, so no two threads ever read and write the same count in one launch.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Các count là số nguyên nhỏ, nên một lần tăng mỗi bước không thể tràn kể cả qua một lần sinh dài.">
            The counts are small integers, so one increment per step cannot overflow even over a long generation.
          </Translated>
        ),
        why: (
          <Translated vi="Tràn số không phải là mối nguy. Mối nguy là hai luồng làm mất lần tăng của nhau.">
            Overflow is not the hazard. Two threads losing each other’s increment is.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="GPU tuần tự hóa các lần ghi thường vào cùng một địa chỉ, nên hai lần tăng một count không thể chồng lên nhau.">
            The GPU serialises plain writes to the same address, so two increments of one count cannot overlap.
          </Translated>
        ),
        why: (
          <Translated vi="Nó không tuần tự hóa một thao tác đọc-sửa-ghi: đó chính là lần cập nhật bị mất của chương 9.">
            It does not serialise a read-modify-write: that is the lost update of chapter 9.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Mỗi lần khởi chạy cập nhật các hàng của nó lần lượt, nên không có hai hàng nào thay đổi count cùng lúc.">
            Each launch updates its rows one after another, so no two rows' counts change at the same time.
          </Translated>
        ),
        why: (
          <Translated vi="Các hàng cập nhật cùng lúc, mỗi hàng một luồng. Điều làm nó an toàn là count của mỗi hàng chỉ có một chủ sở hữu.">
            Rows update at once, one thread each. What makes it safe is that each row's counts have one owner.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mỗi hàng nhận đúng một token mỗi bước, và luồng của hàng r là luồng duy nhất chạm vào hàng r. Đó là quy tắc
            owner-computes ở phần mở đầu chương 9, được khôi phục: một histogram chỉ cần atomic khi bất kỳ luồng nào cũng có
            thể chạm vào bất kỳ bin nào.
          </p>
        }
      >
        <p>
          Each row gains exactly one token per step, and the thread for row r is the only one that touches row r. That is
          the owner-computes rule of chapter 9's opening, restored: a histogram needs atomics only when any thread may hit
          any bin.
        </p>
      </Translated>
    ),
    source: 'ch. 9, p. 201; lab 9.3 README',
    vi: { source: 'ch. 9, p. 201; README của lab 9.3' },
  },
};

// ---------------------------------------------------------------- explorer 1: penalties
const BEFORE = PEN_LOGITS.map(f32);
const P_BEFORE = softmaxDouble(BEFORE);
function PenaltyExplorer() {
  const [context, setContext] = useState<number[]>([...PEN_CONTEXT]);
  const [repeat, setRepeat] = useState(1.25);
  const [freq, setFreq] = useState(0);
  const [presence, setPresence] = useState(0);
  const [rule, setRule] = useState<RepRule>('llama');
  const [order, setOrder] = useState<PenaltyOrder>('repeat-first');
  const counts = useMemo(() => countTokens(context, PEN_VOCAB), [context]);
  const before = BEFORE;
  const after = useMemo(() => applyPenalties(BEFORE, counts, { repeat, freq, presence, rule, order, float32: true }), [counts, repeat, freq, presence, rule, order]);
  const pB = P_BEFORE;
  const pA = softmaxDouble(after);
  const add = (t: number) => setContext((c) => [...c, t]);
  const next = argmaxLowest(after);
  const n = PEN_VOCAB;
  const W = 660;
  const X0 = 34;
  const bw = (W - X0 - 8) / n;
  const ZERO = 92;
  const LS = 26;
  const PB = 236;
  const PS = 100;
  const pMax = Math.max(...pA, ...pB);
  const { raised, gained } = useMemo(() => penaltyEffects(BEFORE, after, counts), [after, counts]);
  const up = PEN_LOGITS.map((_, i) => raised.includes(i));
  const plural = (a: number[], one: string, many: string) => (a.length > 1 ? many : one);

  return (
    <Bench
      title={tr('Penalties', 'Các penalty')}
      task={
        <Translated vi="Đặt một penalty, đổi quy tắc theo dấu, và thêm token vào ngữ cảnh bằng cách bấm vào một cột.">
          Set a penalty, change the sign rule, and add tokens to the context by clicking a column.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Trên: logits trước (viền nét đứt) và sau (tô đặc); ×n là count của token. Dưới: xác suất sau một softmax,
              trước (nét đứt) và sau (tô đặc), đậm hơn ở chỗ một penalty đã nâng logit lên. Mỗi phép toán penalty được làm
              tròn về float32 theo thứ tự đã chọn.
            </>
          }
        >
          <>
            Top: logits before (dashed outline) and after (filled); ×n is the token's count. Bottom: probabilities after a
            softmax, before (dashed) and after (filled), darker where a penalty raised the logit. Every penalty operation is
            rounded to float32 in the order chosen.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem', flexWrap: 'wrap', gap: '0.6rem 1rem' }}>
        <label className="field" style={{ flex: '1 1 10rem' }}>
          repetition = {repeat.toFixed(2)}
          <input type="range" min={100} max={200} step={5} value={Math.round(repeat * 100)} aria-label={tr('repetition', 'repetition')} onChange={(e) => setRepeat(Number(e.target.value) / 100)} style={{ width: '100%' }} />
        </label>
        <label className="field" style={{ flex: '1 1 10rem' }}>
          frequency = {freq.toFixed(2)}
          <input type="range" min={0} max={100} step={5} value={Math.round(freq * 100)} aria-label={tr('frequency', 'frequency')} onChange={(e) => setFreq(Number(e.target.value) / 100)} style={{ width: '100%' }} />
        </label>
        <label className="field" style={{ flex: '1 1 10rem' }}>
          presence = {presence.toFixed(2)}
          <input type="range" min={0} max={100} step={5} value={Math.round(presence * 100)} aria-label={tr('presence', 'presence')} onChange={(e) => setPresence(Number(e.target.value) / 100)} style={{ width: '100%' }} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.6rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Quy tắc repetition">Repetition rule</Translated>
          <span className="seg" role="group" aria-label={tr('Repetition rule', 'Quy tắc repetition')} style={{ flexWrap: 'wrap' }}>
            {REP_RULES.map((r) => (
              <button key={r} aria-pressed={rule === r} onClick={() => setRule(r)} title={REP_RULE_FACT[r]}>{REP_RULE_LABEL[r]}</button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          <Translated vi="Thứ tự">Order</Translated>
          <span className="seg" role="group" aria-label={tr('Penalty order', 'Thứ tự penalty')}>
            {(['repeat-first', 'subtract-first'] as PenaltyOrder[]).map((o) => (
              <button key={o} aria-pressed={order === o} onClick={() => setOrder(o)}>{PENALTY_ORDER_LABEL[o]}</button>
            ))}
          </span>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.6rem', flexWrap: 'wrap', alignItems: 'center' }}>
        <span className="ui" style={{ fontSize: '0.84rem' }} data-readout="context" data-length={context.length}>
          <Translated vi={<>ngữ cảnh ({context.length} token):</>}>context ({context.length} tokens):</Translated>{' '}
          <span className="mono">{context.join(' ')}</span>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.6rem', flexWrap: 'wrap' }}>
        <button onClick={() => add(next)}>
          <Translated vi={<>Thêm token khả dĩ nhất ({next})</>}>Append the most likely token ({next})</Translated>
        </button>
        <button className="quiet" onClick={() => setContext([...PEN_CONTEXT])}>
          <Translated vi="Đặt lại ngữ cảnh">Reset the context</Translated>
        </button>
      </div>

      <svg viewBox={`0 0 ${W} 262`} role="img" aria-label={tr('Logits and probabilities before and after the penalties', 'Logits và xác suất trước và sau các penalty')} data-part="pen-bars"
        style={{ width: '100%', maxWidth: W, minWidth: 440, display: 'block' }}>
        <line x1={X0} y1={ZERO} x2={W - 8} y2={ZERO} stroke="var(--rule)" />
        <text x={X0 - 4} y={ZERO + 4} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>0</text>
        <text x={X0 - 4} y={14} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi="logit">logit</Translated></text>
        <line x1={X0} y1={PB} x2={W - 8} y2={PB} stroke="var(--rule)" />
        <text x={X0 - 4} y={PB - PS + 4} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>p</text>
        {PEN_LOGITS.map((_, i) => {
          const x = X0 + i * bw + 4;
          const w = bw - 8;
          const bar = (v: number) => (v >= 0 ? { y: ZERO - v * LS, h: Math.max(1, v * LS) } : { y: ZERO, h: Math.max(1, -v * LS) });
          const b = bar(before[i]);
          const a = bar(after[i]);
          return (
            <g key={i} onClick={() => add(i)} style={{ cursor: 'pointer' }} data-token={i} data-count={counts[i]} data-after={after[i]}>
              <rect x={x - 3} y={4} width={w + 6} height={PB + 22} fill="transparent" />
              <rect x={x} y={a.y} width={w} height={a.h} rx="1.5" fill="var(--ink-3)" fillOpacity={counts[i] > 0 ? 0.65 : 0.3} />
              <rect x={x} y={b.y} width={w} height={b.h} rx="1.5" fill="none" stroke="var(--ink-2)" strokeDasharray="3 2" />
              <text x={x + w / 2} y={ZERO + (after[i] >= 0 ? 12 : -4)} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
                {counts[i] > 0 ? `×${counts[i]}` : ''}
              </text>
              <rect x={x} y={PB - (pA[i] / pMax) * PS} width={w} height={Math.max(1, (pA[i] / pMax) * PS)} rx="1.5"
                fill={up[i] ? 'var(--ink)' : 'var(--ink-3)'} fillOpacity={up[i] ? 0.85 : 0.5} />
              <rect x={x} y={PB - (pB[i] / pMax) * PS} width={w} height={Math.max(1, (pB[i] / pMax) * PS)} rx="1.5" fill="none" stroke="var(--ink-2)" strokeDasharray="3 2" />
              <text x={x + w / 2} y={PB + 13} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{i}</text>
            </g>
          );
        })}
      </svg>

      <div style={{ overflowX: 'auto', marginTop: '0.5rem' }}>
        <table className="data" style={{ margin: 0, minWidth: '34rem' }} data-part="pen-table">
          <tbody>
            <tr><th><Translated vi="token">token</Translated></th>{PEN_LOGITS.map((_, i) => <td key={i} className="num muted">{i}</td>)}</tr>
            <tr data-row="count"><th><Translated vi="count">count</Translated></th>{counts.map((c, i) => <td key={i} className="num">{c}</td>)}</tr>
            <tr data-row="before"><th><Translated vi="logit trước">logit before</Translated></th>{before.map((v, i) => <td key={i} className="num">{f3(v)}</td>)}</tr>
            <tr data-row="after"><th><Translated vi="logit sau">logit after</Translated></th>{after.map((v, i) => <td key={i} className="num" style={counts[i] > 0 ? { fontWeight: 650 } : undefined}>{f3(v)}</td>)}</tr>
            <tr data-row="p-before"><th><Translated vi="p trước">p before</Translated></th>{pB.map((v, i) => <td key={i} className="num">{v.toFixed(3)}</td>)}</tr>
            <tr data-row="p-after"><th><Translated vi="p sau">p after</Translated></th>{pA.map((v, i) => <td key={i} className="num">{v.toFixed(3)}</td>)}</tr>
          </tbody>
        </table>
      </div>
      <p style={{ margin: '0.5rem 0 0' }} data-readout="pen" data-raised={raised.join(' ')} data-gained={gained.join(' ')}>
        {raised.length ? (
          <Translated
            vi={
              <>
                Theo quy tắc này, {plural(raised, 'token', 'các token')} lặp lại <b>{raised.join(', ')}</b> nhận logit{' '}
                <b>cao hơn</b>: chia một logit âm cho θ đưa nó về gần 0, nên penalty lại thưởng cho việc lặp.
              </>
            }
          >
            <>
              Under this rule, repeated {plural(raised, 'token', 'tokens')} <b>{raised.join(', ')}</b> got a <b>higher</b>{' '}
              logit: dividing a negative logit by θ moves it towards 0, so the penalty rewards the repeat.
            </>
          </Translated>
        ) : (
          <Translated vi={<>Không logit nào của token lặp lại tăng lên.</>}>
            <>No repeated token's logit went up.</>
          </Translated>
        )}{' '}
        {gained.length ? (
          <Translated
            vi={
              <>
                {plural(gained, 'Token', 'Các token')} {gained.join(', ')} đã bị phạt mà vẫn tăng xác suất: softmax chuẩn
                hóa lại, và những token được dùng nhiều hơn mất nhiều hơn.{' '}
              </>
            }
          >
            <>
              {plural(gained, 'Token', 'Tokens')} {gained.join(', ')} {plural(gained, 'was', 'were')} penalised and still gained
              probability: the softmax renormalises, and tokens used more often lost more.{' '}
            </>
          </Translated>
        ) : null}
        <Translated vi={<>Token khả dĩ nhất là {next}.</>}>The most likely token is {next}.</Translated>
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer 2: who reads which logit
const BATCHES = [1, 8, 64];
const MODELS: SegmentModelId[] = ['nvidia', 'amd64', 'amd128', 'metal'];
const METAL_SIZES = [32, 64, 128];

/** Warp widths your own devices reported: lab 6.1's simd_width row, else the device's warpSize. */
function deviceWidths(reports: Reports): { device: string; width: number }[] {
  return realBackends(reports)
    .filter((b) => !/emulator/i.test(String(reports[b].device?.name ?? '')))
    .flatMap((b) => {
      const rows = reports[b].results.filter((r) => r.lab === 'ch06/lab1_coalescing').flatMap((r) => r.bench);
      const w = num(rows.find((r) => r.name === 'simd_width')?.width) ?? num(Number(reports[b].device.warpSize));
      return w ? [{ device: String(reports[b].device.name ?? b), width: w }] : [];
    });
}

function AccessExplorer() {
  const { reports } = useReports();
  const mine = deviceWidths(reports);
  const [mapping, setMapping] = useState<Mapping>('element');
  const [batch, setBatch] = useState(64);
  const [width, setWidth] = useState(32);
  const [model, setModel] = useState<SegmentModelId>('nvidia');
  const [metalSize, setMetalSize] = useState(64);
  const seg = segmentFor(model, metalSize);
  const acc = mappingAccess(mapping, { batch, vocab: QWEN3_VOCAB, warpWidth: width, blockDim: 256 }, seg);
  const widths = [...new Set([32, 64, ...mine.map((m) => m.width)])].sort((a, b) => a - b);
  const W = 700;
  const laneW = (W - 20) / width;
  const segs = acc.segments;
  const segW = Math.min(60, (W - 20) / Math.max(segs.length, 1));
  const segX = (k: number) => 10 + k * segW + ((W - 20) - segs.length * segW) / 2;
  const unit = SEGMENT_MODELS[model].unit;

  return (
    <Bench
      title={tr('Who reads which logit', 'Ai đọc logit nào')}
      task={
        <Translated vi="Chọn một cách ánh xạ và một batch. Đếm số segment mà lần nạp đầu tiên của một warp chạm tới.">
          Pick a mapping and a batch. Count the segments one warp's first load touches.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Một lệnh nạp của warp 0, ở lần lặp đầu tiên, trên một tensor float32 [batch × {fmtInt(QWEN3_VOCAB)}]. Các
              đường nối mỗi lane đang hoạt động với segment căn chỉnh chứa 4 byte của nó. Không có cache: một segment được
              đếm mỗi lần bị chạm tới.
            </>
          }
        >
          <>
            One load instruction of warp 0, its first iteration, over a [batch × {fmtInt(QWEN3_VOCAB)}] float32 tensor. Lines
            join each active lane to the aligned segment its 4 bytes fall in. No cache: a segment is counted every time it is
            touched.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Cách ánh xạ">Mapping</Translated>
          <span className="seg" role="group" aria-label={tr('Mapping', 'Cách ánh xạ')} style={{ flexWrap: 'wrap' }}>
            {MAPPINGS.map((m) => (
              <button key={m} aria-pressed={mapping === m} onClick={() => setMapping(m)}>{MAPPING_LABEL[m]}</button>
            ))}
          </span>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.6rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Batch">Batch</Translated>
          <span className="seg" role="group" aria-label={tr('Batch', 'Batch')}>
            {BATCHES.map((b) => <button key={b} aria-pressed={batch === b} onClick={() => setBatch(b)}>{b}</button>)}
          </span>
        </span>
        <span style={FIELD}>
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <span className="seg" role="group" aria-label={tr('Warp width', 'Độ rộng warp')}>
            {widths.map((w) => (
              <button key={w} aria-pressed={width === w} onClick={() => setWidth(w)}>
                {w}{mine.some((m) => m.width === w) ? tr(' (yours)', ' (của bạn)') : ''}
              </button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          <Translated vi="Segment">Segment</Translated>
          <span className="seg" role="group" aria-label={tr('Segment model', 'Mô hình segment')} style={{ flexWrap: 'wrap' }}>
            {MODELS.map((m) => <button key={m} aria-pressed={model === m} onClick={() => setModel(m)}>{tr(SEGMENT_MODELS[m].label, ({ nvidia: 'NVIDIA: sector 32 byte trong dòng 128 byte', amd64: 'AMD: dòng cache 64 byte', amd128: 'AMD: dòng cache 128 byte', metal: 'Apple: không có tài liệu' } as Record<string, string>)[m] ?? SEGMENT_MODELS[m].label)}</button>)}
          </span>
        </span>
        {model === 'metal' ? (
          <span style={FIELD}>
            <Translated vi="Lựa chọn của bạn cho Metal">Your choice for Metal</Translated>
            <span className="seg" role="group" aria-label={tr('Metal segment size', 'Kích thước segment Metal')}>
              {METAL_SIZES.map((s) => <button key={s} aria-pressed={metalSize === s} onClick={() => setMetalSize(s)}>{s} B</button>)}
            </span>
          </span>
        ) : null}
      </div>
      {mine.length ? (
        <p className="ui muted" style={{ margin: '0 0 0.5rem', fontSize: '0.8rem' }}>
          <Translated vi={<>Thiết bị của bạn: {mine.map((m) => `${m.device}, độ rộng ${m.width}`).join('; ')}.</>}>
            Your devices: {mine.map((m) => `${m.device}, width ${m.width}`).join('; ')}.
          </Translated>
        </p>
      ) : null}

      <svg viewBox={`0 0 ${W} 170`} role="img" aria-label={tr('Lanes of one warp and the segments they touch', 'Các lane của một warp và các segment chúng chạm tới')} data-part="lanes"
        style={{ width: '100%', maxWidth: W, minWidth: 460, display: 'block' }}>
        <text x={10} y={12} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
          <Translated vi={<>lane 0 đến {width - 1} của warp 0</>}>lanes 0 to {width - 1} of warp 0</Translated>
        </text>
        {Array.from({ length: width }, (_, lane) => {
          const r = acc.lanes.find((l) => l.lane === lane);
          return (
            <rect key={lane} x={10 + lane * laneW + 0.5} y={18} width={Math.max(1, laneW - 1)} height={16}
              fill={r ? 'var(--ink-3)' : 'none'} fillOpacity={r ? 0.6 : 0} stroke="var(--ink-3)" strokeWidth="0.6" strokeDasharray={r ? undefined : '2 2'} />
          );
        })}
        {acc.lanes.map((l) => {
          const k = segs.indexOf(Math.floor(l.byte / seg));
          return <line key={l.lane} x1={10 + l.lane * laneW + laneW / 2} y1={34} x2={segX(k) + segW / 2} y2={112} stroke="var(--ink-2)" strokeWidth="0.7" />;
        })}
        {segs.map((sIdx, k) => (
          <g key={sIdx} data-seg={sIdx}>
            <rect x={segX(k) + 1} y={112} width={Math.max(2, segW - 2)} height={18} fill="var(--ink)" fillOpacity={0.12} stroke="var(--ink-2)" />
          </g>
        ))}
        <text x={10} y={150} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
          <Translated
            vi={
              <>
                {segs.length} {({ sector: 'sector', line: 'dòng cache', segment: 'đoạn' } as Record<string, string>)[unit] ?? unit} {seg} byte
                {segs.length > 1 && mapping === 'row' ? `, cách nhau ${fmtInt(QWEN3_VOCAB * 4)} byte (mỗi hàng một)` : segs.length > 1 ? ', nằm cạnh nhau' : ''}
              </>
            }
          >
            {segs.length} {unit}{segs.length === 1 ? '' : 's'} of {seg} bytes
            {segs.length > 1 && mapping === 'row' ? `, ${fmtInt(QWEN3_VOCAB * 4)} bytes apart (one row each)` : segs.length > 1 ? ', side by side' : ''}
          </Translated>
        </text>
      </svg>

      <table className="data" style={{ margin: '0.5rem 0 0', maxWidth: '40rem' }} data-part="access" data-transactions={acc.transactions} data-active={acc.lanes.length}>
        <tbody>
          <tr>
            <th><Translated vi="lane hoạt động trong lần nạp này">active lanes in this load</Translated></th>
            <td className="num"><Translated vi={<>{acc.lanes.length} trên {width}</>}>{acc.lanes.length} of {width}</Translated></td>
          </tr>
          <tr><th><Translated vi="số byte được yêu cầu">bytes asked for</Translated></th><td className="num">{acc.bytesNeeded}</td></tr>
          <tr>
            <th><Translated vi={<>số {({ sector: 'sector', line: 'dòng cache', segment: 'đoạn' } as Record<string, string>)[unit] ?? unit} bị chạm tới</>}>{unit}s touched</Translated></th>
            <td className="num">{acc.transactions} × {seg} B = {acc.bytesFetched} B</td>
          </tr>
          <tr><th><Translated vi="phần hữu ích trong số byte được nạp">useful share of the bytes fetched</Translated></th><td className="num">{Math.round(acc.efficiency * 100)}%</td></tr>
          <tr><th><Translated vi="số luồng có việc trong lần khởi chạy">threads with any work in the launch</Translated></th><td className="num">{fmtInt(acc.threadsWithWork)}</td></tr>
          <tr><th><Translated vi="số logit mỗi luồng có việc nạp">logits each working thread loads</Translated></th><td className="num">{fmtInt(acc.loadsPerThread)}</td></tr>
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- the listing
function PenaltyListing() {
  const { reports } = useReports();
  const STATUS = { cu: solutionTabStatus(reports, LAB, 'kernel.cu'), metal: solutionTabStatus(reports, LAB, 'kernel.metal') };
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  const metal = useLabFile(LAB + '/solution/kernel.metal');
  const k = cu ? functionSource(cu, 'applyPenalties') : null;
  const launch = cu ? functionSource(cu, 'launchApplyPenalties') : null;
  const mk = metal ? functionSource(metal, 'apply_penalties') : null;
  if (!k || !launch || !mk)
    return (
      <p className="muted">
        <Translated vi={<>Kernel được nạp từ <code>labs/{LAB}/solution/</code>.</>}>
          The kernel is loaded from <code>labs/{LAB}/solution/</code>.
        </Translated>
      </p>
    );
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: k.text + '\n\n' + launch.text, tiers: TIERS, status: STATUS.cu },
        { label: tr('Metal kernel', 'Kernel Metal'), code: mk.text, tiers: TIERS, status: STATUS.metal },
      ]}
      caption={
        <Translated
          vi={
            <>
              <code>applyPenalties</code> của lab 9.3 (<code>labs/{LAB}/solution/kernel.cu</code>, dòng {k.firstLine} đến{' '}
              {launch.lastLine}, và <code>kernel.metal</code>): một luồng cho mỗi phần tử của tensor đã làm phẳng, 12 byte
              cho mỗi logit.
            </>
          }
        >
          <>
            Lab 9.3's <code>applyPenalties</code> (<code>labs/{LAB}/solution/kernel.cu</code>, lines {k.firstLine} to{' '}
            {launch.lastLine}, and <code>kernel.metal</code>): one thread per element of the flattened tensor, 12 bytes per
            logit.
          </>
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- the section
export function SX13() {
  const q = questionsSX13;
  return (
    <>
      <Translated
        vi={
          <p>
            Để mặc, một mô hình sẽ lặp lại. Cách chống thông thường là hạ logits của những token đã xuất hiện, với mức hạ
            tùy theo số lần. Việc đó cần một count cho mỗi token của mỗi request, tức là histogram của chương 9 với bộ từ
            vựng làm các bin, và một lượt đi qua logits áp một trong ba công thức. Các công thức đơn giản. Quy tắc theo dấu
            trong một công thức là chỗ các dự án, và cả bài báo mà họ trích dẫn, đi theo những hướng khác nhau.
          </p>
        }
      >
        <p>
          A model left to itself repeats. The usual counter is to lower the logits of tokens that already appeared, by an
          amount that depends on how often. That needs a count per token per request, which is chapter 9's histogram with the
          vocabulary as bins, and a pass over the logits that applies one of three formulas. The formulas are simple. The
          sign rule in one of them is where projects, and the paper they cite, part ways.
        </p>
      </Translated>
      <NoteGroup>
        <Beyond title={tr("Ours, not the book's", 'Của khóa học này, không phải của sách')}>
          <Translated
            vi={
              <p>
                Penalty và việc đếm token id không có trong sách. Các atomic, việc riêng tư hóa (privatization) và quy tắc
                owner-computes là của chương 9.
              </p>
            }
          >
            <p>
              Penalties and counting token ids are not in the book. The atomics, the privatisation and the owner-computes rule
              are chapter 9's.
            </p>
          </Translated>
        </Beyond>
        <Source section="9.2" pages="203–209" />
        <Source section="9.4" pages="211–213" />
      </NoteGroup>

      <h2>
        <Translated vi="Các count đến từ đâu">Where the counts come from</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lab 9.3 đã dựng chúng. <code>countTokens</code> đếm toàn bộ một ngữ cảnh vào một hàng count cho mỗi request, bằng
            atomic ghi thẳng vào bộ nhớ toàn cục (global memory): count của cả một bộ từ vựng không vừa bộ nhớ on-chip của
            bất kỳ khối nào, và các hàng của những request khác nhau không bao giờ dùng chung một bin.{' '}
            <code>appendTokens</code> sau đó thêm một token cho mỗi hàng ở mỗi bước decode, với một luồng sở hữu mỗi hàng,
            nên không cần atomic. Đó là quy tắc owner-computes mà chương 9 mở đầu bằng, nay quay trở lại.
          </p>
        }
      >
        <p>
          Lab 9.3 built them. <code>countTokens</code> counts a whole context into one row of counts per request, with
          atomics straight into global memory: a vocabulary of counts does not fit in any block's on-chip memory, and rows
          of different requests never share a bin. <code>appendTokens</code> then adds one token per row per decode step,
          with one thread owning each row, so no atomic is needed. That is the owner-computes rule the chapter opened with,
          back again.
        </p>
      </Translated>
      <Ask q={q['x1/X1.3/q-append']} />
      <Translated
        vi={
          <p>
            Token nào được đếm là lựa chọn của bên gọi, và các dự án khác nhau. llama.cpp giữ một map trên host cho 64 token
            gần nhất (<code>penalty_last_n</code>). transformers phạt mọi token có mặt trong <code>input_ids</code>, kể cả
            prompt, mỗi token một lần. vLLM đếm các token được sinh vào các bin trên GPU và đánh dấu các token của prompt
            trong một bitmask. mlx-lm xét một cửa sổ 20 token gần nhất. Tài liệu của OpenAI định nghĩa count là số lần token
            đã được lấy mẫu trước vị trí hiện tại.
          </p>
        }
      >
        <p>
          Which tokens count is the caller's choice, and the projects differ. llama.cpp keeps a map on the host over the
          last 64 tokens (<code>penalty_last_n</code>). transformers penalises every token present in{' '}
          <code>input_ids</code>, prompt included, each once. vLLM counts the generated tokens into GPU bins and marks the
          prompt's tokens in a bitmask. mlx-lm looks at a window of the last 20 tokens. OpenAI's documentation defines the
          count as how often the token was sampled before the current position.
        </p>
      </Translated>
      <Today
        checked={tr(
          `2026-11-20 in the facts pass (notes/today/ch09-facts.md D4, D5, D6; ${FACTS}, D1, D4, D9)`,
          `2026-11-20 trong lượt kiểm tra dữ kiện (notes/today/ch09-facts.md D4, D5, D6; ${FACTS}, D1, D4, D9)`,
        )}
      >
        <Translated
          vi={
            <p>
              llama.cpp: <code>std::unordered_map&lt;llama_token, int&gt; token_count</code>,{' '}
              <code>penalty_last_n = 64</code>. vLLM: <code>output_bin_counts</code>,{' '}
              <code>[max_num_reqs, vocab_size]</code> int32, với <code>tl.atomic_add</code>. mlx-lm:{' '}
              <code>context_size: int = 20</code>.
            </p>
          }
        >
          <p>
            llama.cpp: <code>std::unordered_map&lt;llama_token, int&gt; token_count</code>, <code>penalty_last_n = 64</code>.
            vLLM: <code>output_bin_counts</code>, <code>[max_num_reqs, vocab_size]</code> int32, with{' '}
            <code>tl.atomic_add</code>. mlx-lm: <code>context_size: int = 20</code>.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Ba công thức, một thứ tự">Three formulas, one order</Translated>
      </h2>
      <Translated
        vi={
          <p>
            <b>Frequency và presence</b> trừ đi: logit −= count × frequency + (count &gt; 0) × presence. Frequency tăng theo
            mỗi lần lặp; presence chỉ bị tính một lần cho bất kỳ token nào đã xuất hiện. Đây là công thức được OpenAI ghi trong
            tài liệu, và llama.cpp hiện thực đúng nó; <code>applyPenalties</code> của lab 9.3 chính là dòng này.
            transformers không có cái nào trong hai.
          </p>
        }
      >
        <p>
          <b>Frequency and presence</b> subtract: logit −= count × frequency + (count &gt; 0) × presence. Frequency grows
          with every repeat; presence is paid once by any token that appeared. This is OpenAI's documented formula, and
          llama.cpp implements exactly it; lab 9.3's <code>applyPenalties</code> is this line. transformers has neither.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <b>Repetition</b> chia, với mọi token có count lớn hơn 0, cho một hệ số θ như 1.2, nhưng chỉ khi logit dương.
            Một logit âm thì được nhân với θ, để trong cả hai trường hợp token đều khó được chọn hơn. llama.cpp nhân khi logit ≤ 0;
            transformers và mlx-lm nhân khi nó &lt; 0, cho cùng kết quả tại 0. Bài báo mà kỹ thuật này xuất phát, CTRL,
            chia bất kể dấu, và mã nguồn llama.cpp nói trong một chú thích vì sao nó không làm vậy: chia một logit âm cho θ
            sẽ nâng nó lên.
          </p>
        }
      >
        <p>
          <b>Repetition</b> divides, for any token with a count above 0, by a factor θ such as 1.2, but only if the logit is
          positive. A negative logit is multiplied by θ instead, so that both move away from being chosen. llama.cpp
          multiplies when the logit is ≤ 0; transformers and mlx-lm when it is &lt; 0, which gives the same result at 0. The
          paper the technique comes from, CTRL, divides regardless of sign, and llama.cpp's source says in a comment why it
          does not: dividing a negative logit by θ raises it.
        </p>
      </Translated>
      <Today
        checked={tr(
          `2026-11-20 in the facts pass (${FACTS}, D2, D4, D8, D9)`,
          `2026-11-20 trong lượt kiểm tra dữ kiện (${FACTS}, D2, D4, D8, D9)`,
        )}
      >
        <Translated
          vi={
            <p>
              llama.cpp (trích nguyên văn): "The academic publication that described this technique actually just only
              divided, but that would cause tokens with negative logits to become more likely, which is obviously wrong."
              transformers: <code>torch.where(score &lt; 0, score * self.penalty, score / self.penalty)</code>.
            </p>
          }
        >
          <p>
            llama.cpp: "The academic publication that described this technique actually just only divided, but that would
            cause tokens with negative logits to become more likely, which is obviously wrong." transformers:{' '}
            <code>torch.where(score &lt; 0, score * self.penalty, score / self.penalty)</code>.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            llama.cpp áp repetition trước rồi mới trừ, và thứ tự có ý nghĩa: phép chia và phép trừ không giao hoán. M5 dùng
            thứ tự của llama.cpp.
          </p>
        }
      >
        <p>
          llama.cpp applies repetition first and subtracts afterwards, and the order matters: division and subtraction do
          not commute. M5 uses llama.cpp's order.
        </p>
      </Translated>
      <Predict
        id="x1/X1.3/p-rep-neg"
        prompt={
          <Translated vi={<>Một token lặp lại có logit −2.0 và repetition penalty là 1.25. Quy tắc của llama.cpp biến nó thành bao nhiêu?</>}>
            <>A repeated token has logit −2.0 and the repetition penalty is 1.25. What does llama.cpp's rule make it?</>
          </Translated>
        }
      >
        {(c) => (
          <div data-reveal="rep-neg">
            <Translated
              vi={
                <p>
                  <b>−2.5</b>
                  {c === -2.5 ? ', đúng như bạn nói' : ''}: logit ≤ 0, nên nó được nhân, và token trở nên ít khả dĩ hơn.
                  Quy tắc "luôn chia" của CTRL cho −2.0 / 1.25 = −1.6, tức là <i>cao hơn</i>: penalty sẽ khiến token lặp
                  lại khả dĩ hơn, đúng lỗi mà chú thích của llama.cpp mô tả. Token 1 trong trình khám phá bên dưới có đúng
                  logit này; hãy đổi quy tắc sang của CTRL và theo dõi xác suất của nó.
                </p>
              }
            >
              <p>
                <b>−2.5</b>
                {c === -2.5 ? ', as you said' : ''}: the logit is ≤ 0, so it is multiplied, and the token becomes less
                likely. CTRL's "always divide" gives −2.0 / 1.25 = −1.6, which is <i>higher</i>: the penalty would make the
                repeated token more likely, the bug llama.cpp's comment describes. Token 1 in the explorer below has exactly
                this logit; switch the rule to CTRL's and watch its probability.
              </p>
            </Translated>
          </div>
        )}
      </Predict>
      <Ask q={q['x1/X1.3/q-zero']} />
      <Ask q={q['x1/X1.3/q-freq']} />
      <Ask q={q['x1/X1.3/q-order']} />
      <Ask q={q['x1/X1.3/q-hf-counts']} />

      <h2>
        <Translated vi="Thử các penalty">Try the penalties</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mười hai token với logits ở cả hai phía của 0 và một ngữ cảnh hai mươi token lặp lại vài token của mỗi dấu. Bấm
            vào bất kỳ cột nào để thêm token đó vào ngữ cảnh, hoặc thêm token khả dĩ nhất, như một bước decode sẽ làm.
          </p>
        }
      >
        <p>
          Twelve tokens with logits on both sides of zero and a context of twenty tokens that repeats some of each sign.
          Click any column to add that token to the context, or append the most likely one, as one decode step would.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi phép toán penalty được làm tròn về float32 theo thứ tự của llama.cpp, không có fused multiply-add (CUDA
              mặc định gộp <code>a * b + c</code>, điều có thể thay đổi bit cuối; X1.8 và lab X1.3 xử lý chuyện đó).
              Softmax bên dưới các cột được tính bằng double.
            </p>
          }
        >
          <p>
            Each penalty operation is rounded to float32 in llama.cpp's order, with no fused multiply-add (CUDA contracts{' '}
            <code>a * b + c</code> by default, which can change the last bit; X1.8 and lab X1.3 deal with that). The
            softmax below the bars is in double.
          </p>
        </Translated>
      </Simplification>
      <PenaltyExplorer />
      <Translated
        vi={
          <p>
            Bốn điều cần tìm. Theo quy tắc của CTRL, một token lặp lại có logit âm trở nên khả dĩ hơn, và bạn càng tăng θ,
            nó càng khả dĩ hơn. Theo hai quy tắc kia không logit nào tăng, vậy mà một số token bị phạt vẫn tăng xác suất:
            penalty là tương đối, và softmax trao phần xác suất mà một token bị lặp nhiều mất đi cho các token còn lại.
            Presence dịch mọi token lặp lại một lượng như nhau, một lần; frequency dịch một token được dùng năm lần xa gấp
            năm lần. Và khi bật repetition penalty cùng một trong hai phép trừ, đổi thứ tự sẽ thay đổi mọi logit lặp lại.
          </p>
        }
      >
        <p>
          Four things to find. Under CTRL's rule a repeated token with a negative logit becomes more likely, and the more
          you raise θ, the more likely it gets. Under the other two rules no logit goes up, yet some penalised tokens still
          gain probability: a penalty is relative, and the softmax hands the mass a heavily repeated token loses to the
          rest. Presence moves every repeated token by the same amount, once; frequency moves
          a token used five times five times as far. And with the repetition penalty and either subtraction on, swapping the
          order changes every repeated logit.
        </p>
      </Translated>

      <h2>
        <Translated vi="Kernel: một lượt được hợp nhất truy cập">The kernel: one coalesced pass</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Penalty là theo từng phần tử: logit i cần count i và không gì khác. Vì vậy kernel coi logits và counts{' '}
            <code>[rows × vocab]</code> là hai mảng phẳng và giao một luồng cho một phần tử, đúng như{' '}
            <code>applyPenalties</code> của lab 9.3. Repetition penalty cũng theo từng phần tử như thế, nên nó có thể dùng
            chung lượt đó: một lần đọc logit, một lần đọc count, một lần ghi, 12 byte cho mỗi logit, thay vì 24 cho hai
            kernel riêng. Lab X1.3 giữ chúng tách riêng để có thể dùng lại kernel của lab 9.3 nguyên vẹn. Bạn viết
            repetition penalty thành kernel riêng, sau một bản sao logits vào vùng làm việc, và phép trừ của lab 9.3 chạy
            sau nó: ba lượt khoảng 8, 12 và 12 byte cho mỗi logit. Gộp repetition penalty vào bản sao, hoặc vào lượt của lab
            9.3, là một bài tập mà benchmark của lab cho bạn đo.
          </p>
        }
      >
        <p>
          The penalty is elementwise: logit i needs count i and nothing else. So the kernel treats the{' '}
          <code>[rows × vocab]</code> logits and counts as two flat arrays and gives one thread to one element, exactly as
          lab 9.3's <code>applyPenalties</code> does. The repetition penalty is elementwise in the same way, so it could
          share that pass: one read of the logit, one read of the count, one write, 12 bytes per logit, instead of 24 for
          two separate kernels. Lab X1.3 keeps them separate so it can reuse lab 9.3's kernel unchanged. You write the
          repetition penalty as its own kernel, after a copy of the logits into the workspace, and lab 9.3's subtraction
          runs after it: three passes of about 8, 12 and 12 bytes per logit. Fusing the repetition penalty into the copy, or
          into lab 9.3's pass, is an exercise the lab's benchmark lets you measure.
        </p>
      </Translated>
      <PenaltyListing />
      <Ask q={q['x1/X1.3/q-bytes']} />

      <h2>
        <Translated vi="Ai đọc logit nào">Who reads which logit</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Vì sao phải làm phẳng? Vì cách thay thế hiển nhiên, một luồng cho mỗi hàng lặp qua bộ từ vựng, đặt các lane của
            một warp cách nhau cả một hàng: {fmtInt(QWEN3_VOCAB * 4)} byte. Khi đó mỗi lane cần segment bộ nhớ riêng, và ở
            batch 1 chỉ có một luồng có việc trong cả lần khởi chạy. Một khối cho mỗi hàng với vòng lặp có bước nhảy (stride),
            hình dạng của mọi kernel M5 khác, được hợp nhất truy cập (coalesced) như cách ánh xạ phẳng.
          </p>
        }
      >
        <p>
          Why flatten? Because the obvious alternative, one thread per row looping over the vocabulary, puts the lanes of a
          warp a whole row apart: {fmtInt(QWEN3_VOCAB * 4)} bytes. Each lane then needs its own memory segment, and at batch 1
          there is one working thread in the whole launch. One block per row with a strided loop, the shape of every other M5
          kernel, is as coalesced as the flat mapping.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một lần nạp của một warp, với các kích thước segment mà nhà cung cấp ghi trong tài liệu (sector 32 byte của
              NVIDIA, line 64 hoặc 128 byte của AMD); Apple không ghi gì, nên với Metal bạn tự chọn. Không mô hình hóa cache.
              Độ rộng warp là một nút điều khiển: 32 hoặc 64, và độ rộng của thiết bị của bạn khi lab 6.1 hoặc{' '}
              <code>pmpp.py doctor</code> đã báo cáo chúng.
            </p>
          }
        >
          <p>
            One load of one warp, with the segment sizes the vendors document (NVIDIA 32-byte sectors, AMD 64- or 128-byte
            lines); Apple documents none, so for Metal you pick one. No cache is modelled. The warp width is a control:
            32 or 64, and your own devices' widths when lab 6.1 or <code>pmpp.py doctor</code> has reported them.
          </p>
        </Translated>
      </Simplification>
      <AccessExplorer />
      <Translated
        vi={
          <p>
            Với một luồng cho mỗi phần tử hoặc một khối cho mỗi hàng, các lane liên tiếp đọc các float liên tiếp: một warp 32
            lane (độ rộng của NVIDIA) yêu cầu 128 byte và chạm tới bốn sector 32 byte, tất cả đều hữu ích. Với một luồng cho
            mỗi hàng, mỗi lane đang hoạt động chạm tới segment riêng và dùng 4 byte của nó, và warp chỉ có số lane hoạt động
            bằng số hàng.
          </p>
        }
      >
        <p>
          With one thread per element or one block per row, consecutive lanes read consecutive floats: a warp of 32 lanes
          (NVIDIA's width) asks for 128 bytes and touches four 32-byte sectors, all of them useful. With one thread per row, every active lane touches
          its own segment and uses 4 bytes of it, and the warp has only as many active lanes as there are rows.
        </p>
      </Translated>

      <h2>
        <Translated vi="Trên HIP và Metal">On HIP and Metal</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Không có gì riêng cho backend: một kernel theo từng phần tử là như nhau trên cả ba, với kích thước lưới tính từ độ
            dài đã làm phẳng. Trên Metal, các count trong kernel này là một <code>device const uint*</code> thông thường;
            kiểu atomic <code>atomic_uint</code> chỉ cần ở chỗ lab 9.3 đếm, vì Metal không có góc nhìn atomic lên một mảng
            thông thường. Xem <a href={href('ch09', 'labs')}>lab 9.3</a> cho các kernel đếm trên từng backend.
          </p>
        }
      >
        <p>
          Nothing backend-specific: an elementwise kernel is the same on all three, with the grid size computed from the
          flattened length. On Metal the counts are a plain <code>device const uint*</code> in this kernel; the atomic type{' '}
          <code>atomic_uint</code> is needed only where lab 9.3 counts, because Metal has no atomic view of a plain array.
          See <a href={href('ch09', 'labs')}>lab 9.3</a> for the counting kernels on each backend.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 in the facts pass (notes/today/ch09-facts.md C1)',
          '2026-11-20 trong lượt kiểm tra dữ kiện (notes/today/ch09-facts.md C1)',
        )}
      >
        <Translated
          vi={
            <p>
              Metal không có <code>atomic_ref</code>: một buffer được cập nhật bằng atomic phải được khai báo với một kiểu
              atomic, và các lần đọc thường trên nó trở thành <code>atomic_load_explicit</code>.
            </p>
          }
        >
          <p>
            Metal has no <code>atomic_ref</code>: a buffer that is updated atomically must be declared with an atomic type,
            and plain reads of it become <code>atomic_load_explicit</code>.
          </p>
        </Translated>
      </Today>
    </>
  );
}
