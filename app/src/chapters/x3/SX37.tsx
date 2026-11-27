import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { href } from '../../lib/router';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { HEAD, JOB_LABEL, SYNTHETIC, launchCount, stepLaunches, totals, validate, type Config, type Launch } from '../../sim/llamastep';
import { compare, makeCase, pairOf, ropeTables, type Pairing } from '../../sim/ropepair';
import { functionSource } from '../../sim/toolchain';
import { LlamaStepScene } from './LlamaStepScene';
import { RopePairsScene } from './RopePairsScene';

const X3 = 'notes/today/x3-facts.md';
const LAB1 = 'x3/lab1_llama_pieces';
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const TIERS: Record<string, Tier> = {
  x: 'global', cosT: 'global', sinT: 'global', pos: 'global', v: 'global',
  c: 'reg', s: 'reg', x0: 'reg', x1: 'reg', a: 'reg', b: 'reg', i: 'reg',
};
const MiB = (b: number) => (b / 2 ** 20).toFixed(2) + ' MiB';
const bytes = (b: number) => (b >= 2 ** 20 ? MiB(b) : b >= 1024 ? (b / 1024).toFixed(1) + ' KiB' : b + ' B');
/** A function name that may wrap at its camel-case humps and before a parenthesis, so a narrow table never cuts it mid-word. */
const breakable = (name: string) => name.split(/(?=[A-Z(])/).flatMap((part, i) => (i ? [<wbr key={i} />, part] : [part]));

// ---------------------------------------------------------------- questions
export const questionsSX37: Record<string, Question> = {
  'x3/X3.7/q-rope-mismatch': {
    kind: 'choice',
    id: 'x3/X3.7/q-rope-mismatch',
    prompt: (
      <Translated vi={<>Bạn chạy một checkpoint Llama của Hugging Face qua một kernel RoPE ghép cặp liên tiếp mà không hoán vị Q và K. Chuyện gì xảy ra?</>}>
        <>You run a Hugging Face Llama checkpoint through a consecutive-pair RoPE kernel without permuting Q and K. What happens?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Nó chạy, cho ra các điểm attention hữu hạn nhưng sai; chỉ phép so sánh với một bản tham chiếu mới bắt được lỗi.">
            It runs, with finite but wrong attention scores; only a comparison with a reference catches it.
          </Translated>
        ),
        correct: true,
      },
      { label: <Translated vi="Kernel đọc vượt quá cuối mỗi head, vì cách chia nửa đòi một kích thước head khác với cách ghép cặp liên tiếp.">The kernel reads past the end of each head, because the half split expects a different head size from consecutive pairs.</Translated>, why: <Translated vi="Cả hai cách ghép cặp chạm vào cùng các phần tử của head; chỉ có đối tác là khác.">Both pairings touch the same elements of the head; only the partners differ.</Translated> },
      { label: <Translated vi="Các điểm số ra NaN ngay từ lớp đầu tiên.">The scores come out as NaN from the first layer.</Translated>, why: <Translated vi="Phép quay bảo toàn độ dài. Mọi giá trị vẫn hữu hạn và trông hợp lý, và chính điều đó làm lỗi này nguy hiểm.">A rotation preserves lengths. Every value stays finite and plausible, which is what makes the bug dangerous.</Translated> },
      { label: <Translated vi="Không có gì thay đổi: quay thì vẫn là quay.">Nothing changes: a rotation is a rotation.</Translated>, why: <Translated vi="Giờ mỗi phép quay trộn hai phần tử vốn được huấn luyện như thành phần của hai cặp khác nhau.">Each rotation now mixes two elements that were trained as parts of different pairs.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>rotate_half</code> của Hugging Face ghép phần tử i với phần tử i + d/2; một kernel ghép cặp liên tiếp ghép 2i với
            2i + 1. Kernel vẫn quay đúng góc, nhưng mỗi phép quay trộn hai giá trị mà các trọng số đã huấn luyện chưa bao giờ coi
            là một cặp, nên q · k thay đổi. Trong trình khám phá bên dưới, đường sai trơn tru và trông hợp lý. Đầu ra của mô hình
            xấu đi mà không có lỗi nào, nên bài kiểm thử bắt được nó là phép so sánh logits với một bản tham chiếu.
          </p>
        }
      >
        <p>
          Hugging Face's <code>rotate_half</code> pairs element i with element i + d/2; a consecutive-pair kernel pairs 2i with
          2i + 1. The kernel still rotates by the right angles, but each rotation mixes two values that the trained weights
          never meant to be a pair, so q · k changes. In the explorer below the wrong curve is smooth and plausible. The
          model's output degrades without any error, so the test that catches it is a comparison of logits with a reference.
        </p>
      </Translated>
    ),
    source: `${X3} B6`,
  },
  'x3/X3.7/q-permute-k': {
    kind: 'choice',
    id: 'x3/X3.7/q-permute-k',
    prompt: (
      <Translated vi={<>Vì sao <code>permute</code> của llama.cpp chuyển sang <code>n_head_kv</code> khi được áp cho <code>k_proj</code>?</>}>
        <>Why does llama.cpp's <code>permute</code> switch to <code>n_head_kv</code> when it is applied to <code>k_proj</code>?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Với GQA, K chỉ có n_kv_heads × head_size hàng, nên các head của nó phải được đếm bằng n_head_kv.">
            Under GQA, K has only n_kv_heads × head_size rows, so its heads must be counted with n_head_kv.
          </Translated>
        ),
        correct: true,
      },
      { label: <Translated vi="Trong llama.cpp, K được quay với một tần số cơ sở khác Q, nên các hàng của nó cần một cách nhóm khác.">K is rotated with a different base frequency than Q in llama.cpp, so its rows need a different grouping.</Translated>, why: <Translated vi="Q và K dùng cùng các góc; đó là điều làm q · k chỉ phụ thuộc vào khoảng cách.">Q and K use the same angles; that is what makes q · k depend only on the distance.</Translated> },
      { label: <Translated vi="Chỉ K được lưu vào cache, nên chỉ K cần hoán vị.">Only K is cached, so only K needs the permutation.</Translated>, why: <Translated vi="Cả q_proj và k_proj đều được hoán vị; cache không liên quan gì ở đây.">Both q_proj and k_proj are permuted; the cache has nothing to do with it.</Translated> },
      { label: <Translated vi="Đó là một tối ưu: ít head hơn, ít hàng phải di chuyển hơn.">It is an optimisation: fewer heads, fewer rows to move.</Translated>, why: <Translated vi="Với số head của query, phép reshape nhóm sai hàng, và đồng nhất thức không còn đúng.">With the query head count the reshape groups the wrong rows, and the identity fails.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>permute</code> reshape trọng số thành (heads, 2, head_size / 2, columns) và đổi chỗ hai trục giữa, việc này chỉ
            đặt hai nửa của mỗi head cạnh nhau nếu "heads" đúng là số head mà các hàng đó thực sự tạo thành. Với SmolLM2, q_proj là
            [576, 576] (9 head 64) và k_proj là [192, 576] (3 head 64). <code>export.py</code> của llama2.c{' '}
            hoán vị K bằng số head của query, mà theo cách chúng tôi đọc thì không khớp với một K grouped-query.
          </p>
        }
      >
        <p>
          <code>permute</code> reshapes the weight into (heads, 2, head_size / 2, columns) and swaps the middle axes, which
          only puts the two halves of each head side by side if "heads" is the number of heads those rows really form. For
          SmolLM2, q_proj is [576, 576] (9 heads of 64) and k_proj [192, 576] (3 heads of 64). llama2.c's{' '}
          <code>export.py</code> permutes K with the query head count, which by our reading does not fit a grouped-query K.
        </p>
      </Translated>
    ),
    source: `${X3} B6 (conversion/llama.py lines 177 to 182), D4, D6`,
    vi: { source: `${X3} B6 (conversion/llama.py dòng 177 đến 182), D4, D6` },
  },
  'x3/X3.7/q-rope-v': {
    kind: 'choice',
    id: 'x3/X3.7/q-rope-v',
    prompt: (
      <Translated vi={<>RoPE quay những cái nào trong q, k và v, và cache chứa gì?</>}>
        <>Which of q, k and v does RoPE rotate, and what does the cache hold?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="q và k; k được lưu vào cache sau khi quay; v không bao giờ được quay.">q and k; k is cached after the rotation; v is never rotated.</Translated>, correct: true },
      { label: <Translated vi="q, k và v, để tổng có trọng số thẳng hàng với các key.">q, k and v, so that the weighted sum lines up with the keys.</Translated>, why: <Translated vi="Vị trí chỉ đi vào qua q · k. V là phần dữ liệu mang theo.">The position enters through q · k only. V is a payload.</Translated> },
      { label: <Translated vi="q và k; cache chứa k trước khi quay, được quay lại ở mỗi bước.">q and k; the cache holds k before the rotation, rotated again at each step.</Translated>, why: <Translated vi="Như thế sẽ phải quay mọi key trong cache ở mỗi bước. Mã tham chiếu quay k một lần, ở vị trí của chính nó, rồi lưu vào cache.">That would rotate every cached key at every step. The reference code rotates k once, at its own position, then caches it.</Translated> },
      { label: <Translated vi="Chỉ q: các key đã mang sẵn vị trí của chúng.">Only q: the keys already carry their positions.</Translated>, why: <Translated vi="Chúng mang vị trí vì đã được quay trước khi được lưu.">They carry them because they were rotated before they were stored.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            llama2.c quay q và k bên trong lớp, rồi ghi k vào cache ở hàng pos; mã tham chiếu của Meta cũng làm vậy. Trong lab X3.2
            đó là hai lần khởi chạy RoPE, mỗi buffer một lần, trước bước append.
          </p>
        }
      >
        <p>
          llama2.c rotates q and k inside the layer, then writes k into the cache at row pos; Meta's reference code does the
          same. In lab X3.2 that is two RoPE launches, one per buffer, before the append.
        </p>
      </Translated>
    ),
    source: `ch20/20.1/q-rope; ${X3} D1; notes/today/ch20-facts.md D3`, // i18n-ignore: citation
  },
  'x3/X3.7/q-host-tables': {
    kind: 'choice',
    id: 'x3/X3.7/q-host-tables',
    prompt: (
      <Translated vi={<>Vì sao khóa học truyền các bảng cos và sin tính trên host cho kernel RoPE thay vì gọi cosf và sinf trong kernel như llama.cpp?</>}>
        <>Why does the course pass host-computed cos and sin tables to its RoPE kernel instead of calling cosf and sinf in the kernel, as llama.cpp does?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Khi đó mọi backend đều quay bằng cùng các giá trị, bất kể độ chính xác cosf hay chế độ toán học của nó; cái giá là một lần đọc bảng.">
            Every backend then rotates by the same values, whatever its cosf accuracy or math mode; the cost is a table read.
          </Translated>
        ),
        correct: true,
      },
      { label: <Translated vi="Metal không có hàm cos và sin độ chính xác đơn, nên bảng là dạng duy nhất chạy được trên cả ba backend.">Metal has no single-precision cos and sin functions, so a table is the only form that works on all three backends.</Translated>, why: <Translated vi="Cả ba backend đều có. Độ chính xác và các biến thể nhanh của chúng thì khác nhau.">All three backends have them. Their accuracy and their fast variants differ.</Translated> },
      { label: <Translated vi="Đọc bảng luôn nhanh hơn tính cosf và sinf, mỗi hàm tốn hàng chục lệnh.">A table read is always faster than computing cosf and sinf, which take dozens of instructions each.</Translated>, why: <Translated vi="Nó thay một chút số học bằng các lần đọc bộ nhớ. Nhanh hơn hay không thì chưa được đo, và đó không phải lý do.">It replaces a little arithmetic with memory reads. Whether that is faster was not measured, and it is not the reason.</Translated> },
      { label: <Translated vi="Nó sao chép llama.cpp, vốn cũng dựng bảng cos và sin trên host và tải lên ở mỗi bước.">It copies llama.cpp, which also builds its cos and sin tables on the host and uploads them at every step.</Translated>, why: <Translated vi="llama.cpp tính powf, cosf và sinf bên trong rope.cu.">llama.cpp computes powf, cosf and sinf inside rope.cu.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Bảng được tính một lần bằng double và làm tròn về float, nên CUDA, HIP, Metal và bản tham chiếu CPU đều nhân với đúng
            cùng các giá trị cos và sin, bất kể cosf của mỗi thiết bị chính xác đến đâu. Chế độ toán học mặc định của Metal không
            IEEE-strict, và một lần đọc bảng không bị nó ảnh hưởng. Cái giá là một bảng seqLen × head/2 giá trị cos và chừng ấy giá trị sin, mỗi lần khởi chạy đọc một hàng. Đây là của chúng tôi,
            và cũng là lựa chọn mà các file cũ nhất của llama2.c từng làm: chúng lưu các bảng như vậy (freq_cis), mà run.c ngày nay bỏ qua.
          </p>
        }
      >
        <p>
          The table is computed once in double and rounded to float, so CUDA, HIP, Metal and the CPU reference all multiply by
          the very same cos and sin, whatever accuracy each device's own cosf has. Metal's default math mode is not
          IEEE-strict, and a table read is untouched by it. The cost is a table of seqLen × head/2 cos values and as many sines, of which each launch reads one row. Ours,
          and a choice llama2.c's oldest files also made: they stored such tables (freq_cis), which run.c now skips.
        </p>
      </Translated>
    ),
    source: `ours; ${X3} B6, D1; notes/today/ch05-facts.md C7`,
    vi: { source: `của chúng tôi; ${X3} B6, D1; notes/today/ch05-facts.md C7` },
  },
  'x3/X3.7/q-swiglu-bytes': {
    kind: 'numeric',
    id: 'x3/X3.7/q-swiglu-bytes',
    prompt: (
      <Translated vi={<>SwiGLU cho một token của SmolLM2-135M (intermediate size 1536, float32) dưới dạng một kernel riêng: nó đọc cộng ghi bao nhiêu byte?</>}>
        <>SwiGLU for one token of SmolLM2-135M (intermediate size 1536, float32) as a separate kernel: how many bytes does it read plus write?</>
      </Translated>
    ),
    answer: 18432,
    unit: 'bytes',
    explain: (
      <Translated
        vi={
          <p>
            Nó đọc gate và up projection, mỗi cái 1536 float, và ghi một hàng 1536: 3 × 1536 × 4 = 18,432 byte, cho khoảng 1536 ×
            vài phép toán. So với 3 × 576 × 1536 trọng số của ba phép chiếu quanh nó thì nhỏ, nhưng đó là thêm một lần khởi chạy và
            thêm một lượt đi về qua bộ nhớ.
          </p>
        }
      >
        <p>
          It reads the gate and the up projection, 1536 floats each, and writes one row of 1536: 3 × 1536 × 4 = 18,432 bytes,
          for about 1536 × a handful of operations. Against the 3 × 576 × 1536 weights of the three projections around it, it
          is small, but it is one more launch and one more round trip through memory.
        </p>
      </Translated>
    ),
    source: `${X3} D4 (intermediate_size 1536); ours`,
    vi: { unit: 'byte', source: `${X3} D4 (intermediate_size 1536); của chúng tôi` },
  },
  'x3/X3.7/q-fused': {
    kind: 'choice',
    id: 'x3/X3.7/q-fused',
    prompt: (
      <Translated vi={<>llama.cpp có một kernel gộp cho RMSNorm, phép nhân trọng số và RoPE. Việc gộp tiết kiệm được gì cho một token?</>}>
        <>llama.cpp has a fused kernel for RMSNorm, the weight multiply and RoPE. What does fusing save for one token?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Hai lần khởi chạy, và các hàng mà một lần khởi chạy ghi ra rồi lần sau đọc lại.">Two launches, and the rows written by one launch and read back by the next.</Translated>, correct: true },
      { label: <Translated vi="Các byte trọng số của phép chiếu q và k, mà kernel gộp đọc một lần thay vì hai.">The weight bytes of the q and k projections, which the fused kernel reads once instead of twice.</Translated>, why: <Translated vi="Những byte đó được các phép nhân ma trận đọc, và chúng không thuộc phép gộp này.">Those are read by the matrix products, which are not part of this fusion.</Translated> },
      { label: <Translated vi="Một nửa số học, vì phép chuẩn hóa và phép quay dùng chung các phép nhân mà việc gộp chỉ tính một lần.">Half the arithmetic, because the norm and the rotation share multiplications that fusion computes only once.</Translated>, why: <Translated vi="Mọi phép nhân vẫn diễn ra; chỉ là chúng diễn ra trong thanh ghi thay vì sau một lượt đi qua bộ nhớ.">Every multiply still happens; it happens in registers instead of after a trip through memory.</Translated> },
      { label: <Translated vi="Không gì cả ở batch 1: với một token các hàng rất nhỏ, nên gộp chỉ có lợi với batch lớn.">Nothing at batch 1: with one token the rows are tiny, so fusion only pays off for large batches.</Translated>, why: <Translated vi="Ở batch 1 lưu lượng tiết kiệm được nhỏ, nhưng mọi thứ trừ trọng số cũng nhỏ, và số lần khởi chạy tiết kiệm được có ý nghĩa.">At batch 1 the saved traffic is small, but so is everything except the weights, and the saved launches count.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Khi tách riêng, phép chuẩn hóa ghi một hàng, phép nhân đọc và ghi nó, rồi RoPE lại đọc và ghi nó. Khi gộp, hàng được đọc
            một lần và ghi một lần, các bước ở giữa nằm trong thanh ghi. Số FLOP như nhau. Backend CUDA của llama.cpp cũng mặc định
            gộp RMSNorm với phép nhân (và một phép cộng), trước khi dispatch, nên kernel chạy cho <code>GGML_OP_RMS_NORM</code> thường
            là kernel gộp.
          </p>
        }
      >
        <p>
          Separately, the norm writes a row, the multiply reads and writes it, and RoPE reads and writes it again. Fused, the
          row is read once and written once and the steps in between stay in registers. The FLOPs are the same. llama.cpp's
          CUDA backend also fuses RMSNorm with the multiply (and an add) by default, before dispatch, which is why the kernel
          that runs for <code>GGML_OP_RMS_NORM</code> is usually the fused one.
        </p>
      </Translated>
    ),
    source: `${X3} A1, B6 (rms_norm_mul_rope_f32); ours`,
    vi: { source: `${X3} A1, B6 (rms_norm_mul_rope_f32); của chúng tôi` },
  },
};

// ---------------------------------------------------------------- the step stepper
/** A whole-number field that keeps what you type and applies it only when it is a valid size. */
function IntField({ label, value, max, width, onValue }: { label: string; value: number; max: number; width: string; onValue: (v: number) => void }) {
  const [text, setText] = useState(String(value));
  const v = Number(text);
  const ok = text.trim() !== '' && Number.isInteger(v) && v >= 1 && v <= max;
  return (
    <label className="field">
      {label}
      <input type="number" min={1} max={max} step={1} value={text} aria-label={label} aria-invalid={!ok} style={{ width }}
        onChange={(e) => {
          setText(e.target.value);
          const n = Number(e.target.value);
          if (e.target.value.trim() !== '' && Number.isInteger(n) && n >= 1 && n <= max) onValue(n);
        }}
        onBlur={() => { if (!ok) setText(String(value)); }} />
    </label>
  );
}

const VOCABS = [4099, 32000, 49152];
type Gate = 'rope' | 'append' | 'share';

function StepThrough() {
  const [heads, setHeads] = useState(SYNTHETIC.heads);
  const [kvHeads, setKvHeads] = useState(SYNTHETIC.kvHeads);
  const [hidden, setHidden] = useState(SYNTHETIC.hidden);
  const [layers, setLayers] = useState(SYNTHETIC.layers);
  const [vocab, setVocab] = useState(SYNTHETIC.vocab);
  const [pos, setPos] = useState(5);
  const [n, setN] = useState(0);
  const [answered, setAnswered] = useState<Partial<Record<Gate, string>>>({});
  const [draft, setDraft] = useState<Record<Gate, string>>({ rope: '', append: '', share: '' });
  const cfg: Config = { ...SYNTHETIC, d: heads * HEAD, heads, kvHeads, hidden, layers, vocab };
  const bad = validate(cfg);
  const locale = useLocale();
  const ls = useMemo(() => (bad ? [] : stepLaunches(cfg, pos)), [bad, heads, kvHeads, hidden, layers, vocab, pos, locale]); // eslint-disable-line react-hooks/exhaustive-deps
  const t = useMemo(() => totals(ls), [ls]);
  const at = Math.min(n, ls.length);
  const cur: Launch | undefined = at > 0 ? ls[at - 1] : undefined;
  const firstRope = ls.findIndex((l) => l.job === 'rope-q');
  const firstAppend = ls.findIndex((l) => l.job === 'append');
  const lmIdx = ls.findIndex((l) => l.job === 'lm-head');
  // The stepper stops before a launch whose question is still open.
  const gateAt: Record<Gate, number> = { rope: firstRope, append: firstAppend, share: lmIdx };
  const blocking = (Object.keys(gateAt) as Gate[]).find((g) => at === gateAt[g] && !answered[g]);
  const sharePct = 100 * t.lmHeadShare;
  const open = (Object.keys(gateAt) as Gate[]).filter((g) => !answered[g] && gateAt[g] >= at).map((g) => gateAt[g]);
  const nextStop = open.length ? Math.min(...open) : ls.length;

  const reset = () => { setN(0); setAnswered({}); setDraft({ rope: '', append: '', share: '' }); };
  const onCfg = (f: () => void) => { f(); setN(0); };
  const layerOf = (l: Launch | undefined) => (l?.layer ?? null);
  const shown = ls.filter((l) => l.layer === null || l.layer === (layerOf(cur) ?? 0));

  const kvOptions = Array.from({ length: heads }, (_, i) => i + 1).filter((k) => heads % k === 0);

  return (
    <Bench
      title={tr('Step through one token', 'Đi từng bước qua một token')}
      task={
        <Translated vi="Đặt một mô hình, rồi bấm Bước để đi qua một bước decode. Ba câu hỏi sẽ dừng bạn lại trên đường.">
          Set a model, then press Step through one decode step. Three questions stop you on the way.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Kích thước head cố định ở {HEAD}, giá trị <code>HEAD_DIM</code> mà các kernel decode của M7 được biên dịch cho, nên d = heads × {HEAD}.
              Byte tính theo float32. FLOP chỉ đếm các phép nhân ma trận và attention.
            </>
          }
        >
          <>
            Head size is fixed at {HEAD}, the <code>HEAD_DIM</code> M7's decode kernels are compiled for, so d = heads × {HEAD}.
            Bytes are float32. FLOPs count the matrix products and attention only.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi="Số head query">Query heads</Translated>
          <select value={heads} aria-label={tr('Query heads', 'Số head query')} onChange={(e) => onCfg(() => { const h = +e.target.value; setHeads(h); if (h % kvHeads) setKvHeads(h); })}>
            {[1, 2, 4, 6, 8, 12, 16].map((h) => <option key={h} value={h}>{h} (d = {h * HEAD})</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Số head KV">KV heads</Translated>
          <select value={kvHeads} aria-label={tr('KV heads', 'Số head KV')} onChange={(e) => onCfg(() => setKvHeads(+e.target.value))}>
            {kvOptions.map((k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
        <IntField label={tr('Hidden', 'Chiều ẩn')} value={hidden} max={65536} width="6rem" onValue={(v) => onCfg(() => setHidden(v))} />
        <IntField label={tr('Layers', 'Số lớp')} value={layers} max={64} width="4.5rem" onValue={(v) => onCfg(() => setLayers(v))} />
        <label className="field">
          <Translated vi="Bộ từ vựng">Vocabulary</Translated>
          <select value={vocab} aria-label={tr('Vocabulary', 'Bộ từ vựng')} onChange={(e) => onCfg(() => setVocab(+e.target.value))}>
            {VOCABS.map((v) => <option key={v} value={v}>{v.toLocaleString('en-US')}</option>)}
          </select>
        </label>
        <label className="field" style={{ flex: '1 1 12rem' }}>
          pos = {pos}
          <input type="range" min={0} max={SYNTHETIC.seqLen - 1} value={pos} aria-label="pos" onChange={(e) => onCfg(() => setPos(+e.target.value))} style={{ width: '100%' }} />{/* i18n-ignore: variable name */}
        </label>
      </div>
      {bad ? <p className="t-fail">{bad}</p> : null}

      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <button onClick={reset} disabled={at === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => setN(at - 1)} disabled={at === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => setN(at + 1)} disabled={at >= ls.length || !!blocking}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => setN(nextStop)} disabled={at >= ls.length || !!blocking}><Translated vi="Chạy đến câu hỏi tiếp theo">Run to the next question</Translated></button>
        <span className="muted" aria-live="polite" data-at={at}><Translated vi={<>lần khởi chạy {at} trên {ls.length}</>}>launch {at} of {ls.length}</Translated></span>
      </div>

      {blocking ? (
        <div className="q" data-gate={blocking}>
          <div className="kind"><Translated vi="Dự đoán trước">Predict first</Translated></div>
          {blocking === 'rope' ? (
            <>
              <div className="prompt"><Translated vi="Lần khởi chạy tiếp theo là RoPE. Nó quay những buffer nào của bước này?">The next launch is RoPE. Which of the step's buffers does it rotate?</Translated></div>
              <div className="opts" role="radiogroup">
                {['q only', 'q and k', 'q, k and v', 'the cached keys'].map((o) => ( // i18n-ignore: answer keys, shown through tr()
                  <label key={o}><input type="radio" name="gate-rope" checked={draft.rope === o} onChange={() => setDraft({ ...draft, rope: o })} />{/* i18n-ignore: radio group name */} <span>{tr(o, ({ 'q only': 'chỉ q', 'q and k': 'q và k', 'q, k and v': 'q, k và v', 'the cached keys': 'các key trong cache' } as Record<string, string>)[o] ?? o)}</span></label>
                ))}
              </div>
            </>
          ) : blocking === 'append' ? (
            <div className="prompt">
              <Translated vi="Lần khởi chạy tiếp theo nối k và v vào cache của lớp này. Nó ghi vào hàng nào?">The next launch appends k and v to this layer's cache. Which row does it write?</Translated>{' '}
              <input type="number" step="1" aria-label={tr('cache row', 'hàng cache')} value={draft.append} style={{ width: '5rem' }} onChange={(e) => setDraft({ ...draft, append: e.target.value })} />
            </div>
          ) : (
            <div className="prompt">
              <Translated vi="Lần khởi chạy tiếp theo là LM head. Ma trận của nó chiếm bao nhiêu phần trăm tổng số byte trọng số mà bước này đọc?">The next launch is the LM head. What share of all the weight bytes this step reads is its matrix?</Translated>{' '}
              <input type="number" step="any" aria-label={tr('LM head share', 'Tỉ lệ của LM head')} value={draft.share} style={{ width: '5rem' }} onChange={(e) => setDraft({ ...draft, share: e.target.value })} /> %
            </div>
          )}
          <button className="primary" disabled={draft[blocking].trim() === ''} onClick={() => setAnswered({ ...answered, [blocking]: draft[blocking] })}><Translated vi="Chốt">Commit</Translated></button>
        </div>
      ) : null}
      {(Object.keys(answered) as Gate[]).map((g) => {
        const v = answered[g]!;
        const ok = g === 'rope' ? v === 'q and k' : g === 'append' ? Number(v) === pos : Math.abs(Number(v) - sharePct) <= 2;
        return (
          <p key={g} data-answer={g} data-ok={ok ? 1 : 0} style={{ margin: '0 0 0.5rem' }}>
            <span className={'chip ' + (ok ? 'ok' : 'fail')}>{ok ? tr('right', 'đúng') : tr('not quite', 'chưa đúng')}</span>{' '}
            {g === 'rope'
              ? <Translated vi={<>Bạn nói {({ 'q only': 'chỉ q', 'q and k': 'q và k', 'q, k and v': 'q, k và v', 'the cached keys': 'các key trong cache' } as Record<string, string>)[v] ?? v}. RoPE quay q và k, bằng hai lần khởi chạy cùng một kernel, mỗi buffer một lần; v không bao giờ được quay, và cache đã chứa sẵn các key đã quay.</>}><>You said {v}. RoPE rotates q and k, in two launches of the same kernel, one per buffer; v is never rotated, and the cache already holds rotated keys.</></Translated>
              : g === 'append'
                ? <Translated vi={<>Bạn nói hàng {v}. Bước append ghi hàng pos = {pos}: các hàng 0 đến {pos - 1} chứa các token trước đó, và attention sau đó đọc {pos + 1} hàng.</>}><>You said row {v}. The append writes row pos = {pos}: rows 0 to {pos - 1} hold the earlier tokens, and attention then reads {pos + 1} rows.</></Translated>
                : <Translated vi={<>Bạn nói {v} %. LM head là {MiB(t.lmHeadBytes)} trên {MiB(t.weightBytes)}: <b>{sharePct.toFixed(1)} %</b>. Đó là một ma trận vocabulary × d, so với {layers} lớp trọng số attention và feed-forward.</>}><>You said {v} %. The LM head is {MiB(t.lmHeadBytes)} of {MiB(t.weightBytes)}: <b>{sharePct.toFixed(1)} %</b>. It is one matrix of vocabulary × d, against {layers} layers of attention and feed-forward weights.</></Translated>}
          </p>
        );
      })}

      <div className={'narrate'} role="status" style={{ marginBottom: '0.8rem' }} data-part="narrate">
        {cur ? (
          <Translated
            vi={
              <>
                <b>{cur.fn}</b> ({cur.lab.replace(' (or ', ' (hoặc ').replace(' (yours)', ' (của bạn)')}, {cur.milestone}{cur.exists ? '' : ', mới trong module này'}){cur.layer !== null ? `, lớp ${cur.layer}` : ''}:{' '}
                {cur.inputs} → {cur.outputs}. Trọng số đọc: {bytes(cur.weightBytes)}
                {cur.cacheBytes ? <>; cache {cur.job === 'append' ? 'ghi' : 'đọc'}: {bytes(cur.cacheBytes)}</> : null}
                {cur.tableBytes ? <>; bảng đọc: {bytes(cur.tableBytes)}</> : null}.{' '}
                <span className="muted">Trong llama.cpp: {cur.project}.</span>
              </>
            }
          >
            <>
              <b>{cur.fn}</b> ({cur.lab}, {cur.milestone}{cur.exists ? '' : ', new in this module'}){cur.layer !== null ? `, layer ${cur.layer}` : ''}:{' '}
              {cur.inputs} → {cur.outputs}. Weights read: {bytes(cur.weightBytes)}
              {cur.cacheBytes ? <>; cache {cur.job === 'append' ? 'written' : 'read'}: {bytes(cur.cacheBytes)}</> : null}
              {cur.tableBytes ? <>; table read: {bytes(cur.tableBytes)}</> : null}.{' '}
              <span className="muted">In llama.cpp: {cur.project}.</span>
            </>
          </Translated>
        ) : (
          <Translated vi={<>Chưa có gì chạy. Token vào ở vị trí {pos}; {pos} token trước đó đang nằm trong cache. Bấm Bước.</>}>
            <>Nothing has run. Token in at position {pos}; {pos} earlier tokens are in the caches. Press Step.</>
          </Translated>
        )}
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0, background: 'var(--page)' }} data-part="launches">
          <thead>
            <tr><th style={{ paddingLeft: '0.6rem' }}>#</th><th><Translated vi="lần khởi chạy">launch</Translated></th><th><Translated vi="kernel của khóa học">course kernel</Translated></th><th><Translated vi="từ">from</Translated></th><th className="num"><Translated vi="trọng số">weights</Translated></th><th className="num"><Translated vi="cache">cache</Translated></th></tr>
          </thead>
          <tbody>
            {shown.map((l, k) => {
              const prev = shown[k - 1];
              const gap = prev && l.layer === null && prev.layer !== null && (prev.layer as number) < layers - 1;
              return (
                <tr key={l.n} data-n={l.n} data-job={l.job} className={cur?.n === l.n ? 'cur' : ''}
                  style={{ opacity: l.n < at ? 1 : 0.6, background: cur?.n === l.n ? 'var(--rule-soft)' : undefined, fontWeight: cur?.n === l.n ? 600 : undefined }}>
                  <td style={{ paddingLeft: '0.6rem' }}>{gap ? '… ' : ''}{l.n + 1}</td>
                  <td>{JOB_LABEL[l.job]}{l.layer !== null ? <span className="muted"><Translated vi={<> (lớp {l.layer})</>}> (layer {l.layer})</Translated></span> : null}</td>
                  <td><code style={{ overflowWrap: 'normal' }}>{breakable(l.fn)}</code></td>
                  <td>{tr(l.lab, l.lab.replace(' (or ', ' (hoặc ').replace(' (yours)', ' (của bạn)'))}{l.exists ? null : <b><Translated vi=" · mới"> · new</Translated></b>}</td>
                  <td className="num">{l.weightBytes ? bytes(l.weightBytes) : '–'}</td>
                  <td className="num">{l.cacheBytes ? bytes(l.cacheBytes) : '–'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <Translated
        vi={
          <p className="muted" style={{ margin: '0.5rem 0 0' }}>
            Bảng hiển thị lớp mà trình chạy từng bước đang ở; {layers - 1} lớp còn lại lặp lại đúng 16 lần khởi chạy đó. Các lần khởi chạy đánh dấu mới
            đến từ module này: lab X3.1 xây RMSNorm, RoPE và SwiGLU, lab X3.2 xây bước decode và phần quản lý đi kèm.
          </p>
        }
      >
        <p className="muted" style={{ margin: '0.5rem 0 0' }}>
          The table shows the layer the stepper is in; the other {layers - 1} layers repeat the same 16 launches. Launches marked new
          come from this module: lab X3.1 builds RMSNorm, RoPE and SwiGLU, lab X3.2 the step and its bookkeeping.
        </p>
      </Translated>
      <table className="data" style={{ margin: '0.7rem 0 0', maxWidth: '40rem' }} data-part="totals">
        <tbody>
          <tr data-total="launches" data-value={t.launches}><th><Translated vi="số lần khởi chạy mỗi token">launches per token</Translated></th><td className="num">{t.launches} <span className="muted">= 1 + 16 × {layers} + 4</span></td></tr>
          <tr data-total="weights" data-value={t.weightBytes}><th><Translated vi="byte trọng số mỗi token">weight bytes per token</Translated></th><td className="num">{MiB(t.weightBytes)}</td></tr>
          <tr data-total="cache" data-value={t.cacheBytes}><th><Translated vi={<>byte cache tại pos {pos}</>}>cache bytes at pos {pos}</Translated></th><td className="num">{bytes(t.cacheBytes)}</td></tr>
          <tr data-total="flops" data-value={t.flops}><th><Translated vi="FLOP (phép nhân ma trận, attention)">FLOPs (matrix products, attention)</Translated></th><td className="num">{(t.flops / 1e6).toFixed(1)} M</td></tr>
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- the pairing explorer
const HEADS = [8, 10, 12, 14, 16];
const MAXD = 15;

function TwoPairings() {
  const [head, setHead] = useState(8);
  const [pos, setPos] = useState(15);
  const [mode, setMode] = useState<Pairing>('norm');
  const [permuted, setPermuted] = useState(false);
  const t = useMemo(() => ropeTables(64, head), [head]);
  const c = useMemo(() => makeCase(head), [head]);
  const r = useMemo(() => compare(c, pos, mode, permuted, t, MAXD), [c, pos, mode, permuted, t]);
  const half = head / 2;
  const pairIdx = (e: number) => Array.from({ length: half }, (_, j) => j).find((j) => pairOf(j, head, mode).includes(e))!;

  // the chart
  const W = 640, X0 = 50, X1 = 620, Y0 = 200, Y1 = 30;
  const ref = r.ref.scores;
  const ker = r.ker.scores;
  const vals = [...ref, ...ker].filter((v) => !Number.isNaN(v));
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const px = (d: number) => X0 + (d / MAXD) * (X1 - X0);
  const py = (s: number) => Y0 - ((s - lo) / (hi - lo || 1)) * (Y0 - Y1);
  const path = (s: number[]) => s.map((v, d) => (Number.isNaN(v) ? '' : `${d && !Number.isNaN(s[d - 1]) ? 'L' : 'M'}${px(d).toFixed(1)},${py(v).toFixed(1)}`)).join(' ');
  const col = r.same ? 'var(--ok)' : 'var(--fail)';

  return (
    <Bench
      title={tr('Two pairings', 'Hai cách ghép cặp')}
      task={
        <Translated vi="Các trọng số được huấn luyện cho cách chia nửa. Hãy thử kernel ghép cặp liên tiếp, rồi bật phép hoán vị.">
          The weights were trained for the half split. Try the consecutive-pair kernel, then switch on the permutation.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Một head của q và một head của k, mỗi cái là một phép chiếu W x của một đầu vào 6 giá trị với các trọng số ngẫu nhiên cố
              định. Bảng: cơ số 10000, dựng bằng double và lưu dưới dạng float, như host của lab X3.1 làm. Điểm số: q ở pos so với k ở pos − khoảng cách.
            </>
          }
        >
          <>
            One head of q and one of k, each a projection W x of a 6-value input with fixed random weights. Tables: base
            10000, built in double and stored as float, as lab X3.1's host does. Scores: q at pos against k at pos − distance.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi="Kích thước head">Head size</Translated>
          <select value={head} aria-label={tr('Head size', 'Kích thước head')} onChange={(e) => setHead(+e.target.value)}>
            {HEADS.map((h) => <option key={h} value={h}>{h}</option>)}
          </select>
        </label>
        <span style={FIELD}>
          <Translated vi="Cách ghép cặp của kernel">Kernel pairing</Translated>
          <span className="seg" role="group" aria-label={tr('Kernel pairing', 'Cách ghép cặp của kernel')}>
            <button aria-pressed={mode === 'norm'} onClick={() => setMode('norm')}><Translated vi="liên tiếp (NORM)">consecutive (NORM)</Translated></button>
            <button aria-pressed={mode === 'neox'} onClick={() => setMode('neox')}><Translated vi="chia nửa (NEOX)">half split (NEOX)</Translated></button>
          </span>
        </span>
        <label className="check" style={{ alignSelf: 'flex-end' }}>
          <input type="checkbox" checked={permuted} onChange={(e) => setPermuted(e.target.checked)} /> <span><Translated vi="hoán vị các hàng của Wq và Wk như bộ chuyển đổi của llama.cpp làm">permute the rows of Wq and Wk as llama.cpp's converter does</Translated></span>
        </label>
        <label className="field" style={{ flex: '1 1 10rem' }}>
          pos = {pos}
          <input type="range" min={0} max={40} value={pos} aria-label={tr('position', 'vị trí')} onChange={(e) => setPos(+e.target.value)} style={{ width: '100%' }} />
        </label>
      </div>

      <div data-part="pairs" style={{ display: 'grid', gridTemplateColumns: `repeat(${head}, minmax(0, 1fr))`, gap: '0.2rem', maxWidth: `${head * 3.4}rem`, marginBottom: '0.3rem' }}>
        {Array.from({ length: head }, (_, e) => (
          <div key={e} data-el={e} data-pair={pairIdx(e)} style={{ textAlign: 'center', border: '1px solid var(--rule)', borderRadius: 3, padding: '0.15rem 0', background: 'var(--cell)' }}>
            <div className="muted" style={{ fontSize: '0.7rem' }}>q{e}</div>
            <div style={{ fontWeight: 600 }}>{String.fromCharCode(65 + pairIdx(e))}</div>
          </div>
        ))}
      </div>
      <Translated
        vi={
          <p className="muted" style={{ margin: '0 0 0.7rem' }}>
            Các phần tử cùng chữ cái được kernel này quay cùng nhau. {permuted ? 'Kernel thấy q đã hoán vị, nên phần tử 2j + 1 giờ chứa thứ trước đây là phần tử j + ' + half + '.' : ''}
          </p>
        }
      >
        <p className="muted" style={{ margin: '0 0 0.7rem' }}>
          Elements with the same letter are rotated together by this kernel. {permuted ? 'The kernel sees the permuted q, so element 2j + 1 now holds what was element j + ' + half + '.' : ''}
        </p>
      </Translated>

      <div style={{ overflowX: 'auto' }}>
      <svg viewBox={`0 0 ${W} 228`} role="img" aria-label={tr('q · k against distance', 'q · k theo khoảng cách')} data-part="scores"
        style={{ width: '100%', maxWidth: W, minWidth: 460, display: 'block' }}>
        <line x1={X0} y1={Y0} x2={X1} y2={Y0} stroke="var(--rule)" />
        {Array.from({ length: MAXD + 1 }, (_, d) => (
          <text key={d} x={px(d)} y={Y0 + 14} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{d}</text>
        ))}
        <text x={(X0 + X1) / 2} y={Y0 + 27} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}><Translated vi="khoảng cách giữa query và key">distance between query and key</Translated></text>
        <text x={X0 - 6} y={py(hi) + 4} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{hi.toFixed(2)}</text>
        <text x={X0 - 6} y={py(lo) + 4} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{lo.toFixed(2)}</text>
        <path d={path(ref)} fill="none" stroke="var(--ink)" strokeWidth="2" data-curve="reference" />
        <path d={path(ker)} fill="none" stroke={col} strokeWidth="2.4" strokeDasharray={r.same ? '6 4' : undefined} data-curve="kernel" />
        <g transform={`translate(${X0 + 8}, ${Y1 - 14})`}>
          <line x1={0} y1={0} x2={20} y2={0} stroke="var(--ink)" strokeWidth="2" />
          <text x={26} y={4} fontSize="10" style={{ fill: 'var(--ink)' }}><Translated vi="tham chiếu: chia nửa trên W x">reference: half split on W x</Translated></text>
          <line x1={200} y1={0} x2={220} y2={0} stroke={col} strokeWidth="2.4" strokeDasharray={r.same ? '6 4' : undefined} />
          <text x={226} y={4} fontSize="10" style={{ fill: 'var(--ink)' }}><Translated vi="kernel này">this kernel</Translated></text>
        </g>
      </svg>
      </div>
      <p style={{ margin: '0.5rem 0 0' }} data-part="verdict" data-same={r.same ? 1 : 0} data-diff={r.maxScoreDiff}>
        {r.same ? (
          <span className="chip ok"><Translated vi="khớp với tham chiếu">matches the reference</Translated></span>
        ) : (
          <span className="chip fail"><Translated vi={<>khác tham chiếu tối đa {r.maxScoreDiff.toFixed(3)}</>}>differs from the reference by up to {r.maxScoreDiff.toFixed(3)}</Translated></span>
        )}{' '}
        {r.same
          ? permuted
            ? tr('The rotated q, put back in the original order, equals the reference element for element, and so does every score.', 'q đã quay, khi đặt lại theo thứ tự ban đầu, bằng tham chiếu đến từng phần tử, và mọi điểm số cũng vậy.')
            : tr('Same pairing as training: nothing to fix.', 'Cùng cách ghép cặp như lúc huấn luyện: không có gì phải sửa.')
          : tr('Every value is finite and the curve looks plausible. Nothing would fail; the scores are simply not the trained ones.', 'Mọi giá trị đều hữu hạn và đường cong trông hợp lý. Không có gì báo lỗi; chỉ là các điểm số không phải những điểm số đã được huấn luyện.')}
      </p>
      {pos < MAXD ? <p className="muted" style={{ margin: '0.3rem 0 0' }}><Translated vi={<>Key chỉ tồn tại ở các vị trí 0 đến {pos}, nên đường cong dừng ở khoảng cách {pos}.</>}>Keys exist only at positions 0 to {pos}, so the curve stops at distance {pos}.</Translated></p> : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the RoPE listing
function RopeListing() {
  const cu = useLabFile(LAB1 + '/solution/kernel.cu');
  // The labs page's words (LabKit's solutionTabStatus), from this machine's runs.
  const { reports } = useReports();
  const metal = useLabFile(LAB1 + '/solution/kernel.metal');
  const k = cu ? functionSource(cu, 'ropeKernel') : null;
  const mk = metal ? functionSource(metal, 'rope_rows') : null;
  if (!k || !mk) return <p className="muted"><Translated vi={<>Kernel được nạp từ <code>labs/{LAB1}/solution/</code>.</>}>The kernel is loaded from <code>labs/{LAB1}/solution/</code>.</Translated></p>;
  const lines = (txt: string) => txt.split('\n').flatMap((ln, i) => (/cosT|sinT/.test(ln) && !/^\s*\/\//.test(ln) ? [i + 1] : []));
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: k.text, tiers: TIERS, status: solutionTabStatus(reports, LAB1, 'kernel.cu'), highlight: lines(k.text) },
        { label: tr('Metal kernel', 'Kernel Metal'), code: mk.text, tiers: TIERS, status: solutionTabStatus(reports, LAB1, 'kernel.metal'), highlight: lines(mk.text) },
      ]}
      caption={
        <Translated
          vi={
            <>
              RoPE của lab X3.1 (<code>labs/{LAB1}/solution/kernel.cu</code>, dòng {k.firstLine} đến {k.lastLine}, và{' '}
              <code>kernel.metal</code>). Mỗi cặp một luồng; các dòng được đánh dấu đọc bảng của host. Mode 0 là cách ghép cặp
              liên tiếp, mode 1 là cách chia nửa.
            </>
          }
        >
          <>
            Lab X3.1's RoPE (<code>labs/{LAB1}/solution/kernel.cu</code>, lines {k.firstLine} to {k.lastLine}, and{' '}
            <code>kernel.metal</code>). One thread per pair; the marked lines read the host's tables. Mode 0 is the consecutive
            pairing, mode 1 the half split.
          </>
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- the section
export function SX37() {
  const q = questionsSX37;
  const syn = totals(stepLaunches(SYNTHETIC, 0));
  return (
    <>
      <Translated
        vi={
          <p>
            Bạn đã xây gần như mọi kernel mà một bước decode của Llama cần. Phần này trải bước đó ra thành một chuỗi các lần khởi
            chạy những kernel ấy, gọi tên ba mảnh còn thiếu (RMSNorm, RoPE và SwiGLU), và dành phần lớn thời gian cho mảnh âm thầm
            làm hỏng các checkpoint thật: hai cách ghép cặp các phần tử của một head cho RoPE.
          </p>
        }
      >
        <p>
          You have built almost every kernel a Llama decode step needs. This section lays the step out as a sequence of
          launches of those kernels, names the three pieces still missing (RMSNorm, RoPE and SwiGLU), and spends most of its
          time on the one that silently breaks real checkpoints: the two ways to pair the elements of a head for RoPE.
        </p>
      </Translated>
      <Beyond title={tr("Ours, not the book's", 'Của khóa học này, không phải của sách')}>
        <Translated
          vi={
            <p>
              RMSNorm, RoPE và SwiGLU không có trong sách; transformer của sách dùng LayerNorm, ReLU và một mã hóa vị trí được cộng
              vào. Phần attention của bước này là phép tính theo từng token của sách với KV cache. Việc mỗi dự án làm được đọc trong
              lượt kiểm tra dữ kiện ngày 2026-11-20 ({X3}).
            </p>
          }
        >
          <p>
            RMSNorm, RoPE and SwiGLU are not in the book; its transformer uses LayerNorm, a ReLU and an added positional
            encoding. The step's attention part is the book's per-token computation with a KV cache. What each project does
            was read in the facts pass on 2026-11-20 ({X3}).
          </p>
        </Translated>
      </Beyond>
      <Source section="20.4" pages="488–492" />

      <h2><Translated vi="Một bước decode, từ llama2.c">One decode step, from llama2.c</Translated></h2>
      <Translated
        vi={
          <p>
            Bản tham chiếu là <code>forward</code> của llama2.c, khoảng 130 dòng C. Embed token. Rồi, với mỗi lớp: RMSNorm; các phép
            chiếu q, k và v; RoPE trên q và k; nối k và v vào cache của lớp ở hàng pos; attention trên pos + 1 key, với các head
            grouped-query; phép chiếu đầu ra wo; một phép cộng residual; RMSNorm lần nữa; w1 và w3; SwiGLU; w2; một phép cộng residual
            thứ hai. Sau lớp cuối: một RMSNorm cuối, bộ phân loại (dùng lại bảng embedding), và argmax hoặc sampler. Prompt đi qua
            cùng hàm đó, mỗi lần một token.
          </p>
        }
      >
        <p>
          The reference is llama2.c's <code>forward</code>, about 130 lines of C. Embed the token. Then, per layer: RMSNorm;
          the q, k and v projections; RoPE on q and k; append k and v to the layer's cache at row pos; attention over pos + 1
          keys, with grouped-query heads; the output projection wo; a residual add; RMSNorm again; w1 and w3; SwiGLU; w2; a
          second residual add. After the last layer, a final RMSNorm, the classifier, which reuses the embedding table, and
          argmax or the sampler. The prompt goes through the same function one token at a time.
        </p>
      </Translated>
      <Today checked={tr(`2026-11-20 in the facts pass (${X3} D1, project notes)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${X3} D1, ghi chú dự án)`)}>
        <Translated
          vi={
            <p>
              llama2.c <code>run.c</code>, commit cuối 2023-11-28: epsilon của RMSNorm <code>1e-5f</code> và cơ số RoPE{' '}
              <code>10000.0f</code> được viết cứng; GQA bằng <code>(h / kv_mul) * head_size</code>.
            </p>
          }
        >
          <p>
            llama2.c <code>run.c</code>, last commit 2023-11-28: RMSNorm epsilon <code>1e-5f</code> and RoPE base{' '}
            <code>10000.0f</code> are hard-coded; GQA by <code>(h / kv_mul) * head_size</code>.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Mọi kernel đều là kernel bạn đã viết, trừ bốn cái (RMSNorm, RoPE và SwiGLU từ lab X3.1, và bước advance bạn viết trong
            lab X3.2), và ba cái đầu chạy nhiều lần mỗi token: riêng RMSNorm đã hai lần mỗi lớp và thêm một lần ở cuối. Phần còn lại:{' '}
            <code>launchEmbed</code> của lab 22.2 (M8),{' '}
            <code>LinearFn</code> của các lab 6.4, 15.6 và 22.1 (ở batch 1 là GEMV gather của lab 22.1), <code>launchKvAppend</code>{' '}
            của lab 20.4 và attention decode (M7, được nhóm trong lab 20.5), <code>launchAddResidual</code> của lab 2.3{' '}
            (M0), và <code>launchArgmaxRows</code> của lab 10.3 (M4) hoặc sampler của lab X1.3 (M5). Các kernel của M7 được biên dịch cho
            kích thước head 128, nên mô hình tổng hợp của lab X3.2, điểm khởi đầu của trình chạy từng bước, dùng 128.
          </p>
        }
      >
        <p>
          Every kernel but four is one you have written (RMSNorm, RoPE and SwiGLU from lab X3.1, and the advance you write in
          lab X3.2), and the first three run many times per token: RMSNorm alone twice per layer and once more at the end. The rest:
          lab 22.2's <code>launchEmbed</code> (M8), the{' '}
          <code>LinearFn</code> of labs 6.4, 15.6 and 22.1 (at batch 1, lab 22.1's gather GEMV), lab 20.4's{' '}
          <code>launchKvAppend</code> and decode attention (M7, grouped in lab 20.5), lab 2.3's <code>launchAddResidual</code>{' '}
          (M0), and lab 10.3's <code>launchArgmaxRows</code> (M4) or lab X1.3's sampler (M5). M7's kernels are compiled for a
          head size of 128, so lab X3.2's synthetic model, the stepper's starting point, uses 128.
        </p>
      </Translated>
      <Today checked={tr(`2026-11-20 in the facts pass (${X3} E2)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${X3} E2)`)}>
        <Translated
          vi={
            <p>
              Không checkpoint thật cỡ nhỏ nào có kích thước head 128: stories260K là 8, stories15M là 48, SmolLM2-135M là 64. Chạy một
              checkpoint như vậy cần một kernel decode nhận kích thước head làm tham số.
            </p>
          }
        >
          <p>
            No small real checkpoint has head size 128: stories260K has 8, stories15M 48, SmolLM2-135M 64. Running one needs a
            decode kernel with the head size as a parameter.
          </p>
        </Translated>
      </Today>
      <Simplification>
        <Translated
          vi={
            <p>
              Batch 1 và float32. Byte là lượng tối thiểu mỗi kernel phải đọc và ghi, không có bộ nhớ đệm (cache). Các phép toán từng
              phần tử không được tính là FLOP. Danh sách lần khởi chạy là của lab X3.2, như phần đầu file của nó khai báo.
            </p>
          }
        >
          <p>
            Batch 1 and float32. Bytes are what each kernel must read and write at least, with no caches. Elementwise
            operations are not counted as FLOPs. The launch list is lab X3.2's, as its header declares it.
          </p>
        </Translated>
      </Simplification>
      <StepThrough />
      <Translated
        vi={
          <p>
            Điều cần để ý. Ở pos 0, trọng số gần như là tất cả những gì bước này di chuyển; kéo pos lên và chỉ cột cache tăng. LM head
            chỉ là một ma trận, vậy mà với mô hình của lab X3.2 nó chiếm {(100 * syn.lmHeadShare).toFixed(1)} % byte trọng số; với bộ
            từ vựng 49,152 và ít lớp hơn, nó có thể áp đảo. Và bước này là {launchCount(4)} lần khởi chạy cho bốn lớp, phần lớn là nhỏ:
            đó là lý do các dự án gộp kernel.
          </p>
        }
      >
        <p>
          What to notice. At pos 0 the weights are almost everything this step moves; drag pos up and only the cache column
          grows. The LM head is a single matrix, yet for lab X3.2's model it is {(100 * syn.lmHeadShare).toFixed(1)} % of the
          weight bytes; with a vocabulary of 49,152 and fewer layers it can dominate. And the step is {launchCount(4)} launches
          for four layers, most of them small: the reason projects fuse.
        </p>
      </Translated>
      <Ask q={q['x3/X3.7/q-rope-v']} />
      <LlamaStepScene />

      <h2><Translated vi="SwiGLU">SwiGLU</Translated></h2>
      <Translated
        vi={
          <p>
            Khối feed-forward của một lớp Llama là down(silu(gate(x)) · up(x)): hai phép chiếu song song, một SiLU trên một nhánh,
            một phép nhân từng phần tử, và phép chiếu thứ ba đưa về d. Hugging Face viết nó là{' '}
            <code>down_proj(act_fn(gate_proj(x)) * up_proj(x))</code>; llama2.c viết w2(silu(w1 x) · w3 x). Dưới dạng kernel, nó là
            kernel từng phần tử của chương 2 trên hai đầu vào.
          </p>
        }
      >
        <p>
          The feed-forward of a Llama layer is down(silu(gate(x)) · up(x)): two projections side by side, a SiLU on one, an
          elementwise product, and a third projection back to d. Hugging Face writes it as{' '}
          <code>down_proj(act_fn(gate_proj(x)) * up_proj(x))</code>; llama2.c as w2(silu(w1 x) · w3 x). As a kernel it is
          chapter 2's elementwise kernel over two inputs.
        </p>
      </Translated>
      <Today checked={tr(`2026-11-20 in the facts pass (${X3} B1, D4)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${X3} B1, D4)`)}>
        <Translated
          vi={
            <p>
              llama.cpp: <code>unary_gated_op_kernel</code> trong <code>unary.cu</code>, mỗi đầu ra một luồng, và{' '}
              <code>kernel_swiglu</code> trên Metal, một vòng lặp grid-stride. mlx-lm: <code>swiglu</code>, dưới{' '}
              <code>mx.compile</code>, nên MLX sinh kernel của nó lúc chạy.
            </p>
          }
        >
          <p>
            llama.cpp: <code>unary_gated_op_kernel</code> in <code>unary.cu</code>, one thread per output, and{' '}
            <code>kernel_swiglu</code> on Metal, a grid-stride loop. mlx-lm: <code>swiglu</code>, under{' '}
            <code>mx.compile</code>, so MLX generates its kernel at run time.
          </p>
        </Translated>
      </Today>
      <Ask q={q['x3/X3.7/q-swiglu-bytes']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            RMSNorm là phép rút gọn theo hàng của chương 10 với tổng bình phương thay cho cực đại, rồi một phép nhân tỉ lệ mỗi hàng:{' '}
            x · 1/√(mean(x²) + ε) · w. Trên CUDA, llama.cpp mặc định gộp nó với phép nhân trọng số, và cũng có một kernel gộp RMSNorm,
            phép nhân và RoPE, <code>rms_norm_mul_rope_f32</code>.
          </p>
        }
      >
        <p>
          RMSNorm is chapter 10's row reduction with a sum of squares instead of a maximum, then one scale per row:{' '}
          x · 1/√(mean(x²) + ε) · w. llama.cpp fuses it with the weight multiply by default on CUDA, and it also has a kernel
          that fuses RMSNorm, the multiply and RoPE, <code>rms_norm_mul_rope_f32</code>.
        </p>
      </Translated>
      <Ask q={q['x3/X3.7/q-fused']} />

      <h2><Translated vi="RoPE: hai cách ghép cặp">RoPE: two pairings</Translated></h2>
      <Translated
        vi={
          <p>
            RoPE chia mỗi head thành các cặp và quay cặp j một góc pos × θ<sub>j</sub>, với θ<sub>j</sub> = base<sup>−2j/d</sup>.
            Các góc ở đâu cũng như nhau; hai phần tử nào tạo thành một cặp thì không. Cách ghép cặp liên tiếp quay (0, 1), (2, 3),
            và cứ thế: llama.cpp gọi nó là "normal" (<code>rope_norm</code>), MLX gọi là "traditional", nơi khác gọi là "interleaved",
            và llama2.c dùng nó. Cách chia nửa quay phần tử j với phần tử j + d/2: <code>rope_neox</code> của llama.cpp,{' '}
            <code>rotate_half</code> của Hugging Face, mặc định <code>rope_traditional = False</code> của mlx-lm, và{' '}
            <code>"rope_interleaved": false</code> của SmolLM2.
          </p>
        }
      >
        <p>
          RoPE splits each head into pairs and rotates pair j by pos × θ<sub>j</sub>, with θ<sub>j</sub> = base<sup>−2j/d</sup>.
          The angles are the same everywhere; which two elements form a pair is not. The consecutive pairing rotates (0, 1),
          (2, 3), and so on: llama.cpp calls it "normal" (<code>rope_norm</code>), MLX "traditional", others "interleaved",
          and llama2.c uses it. The half split rotates element j with element j + d/2: llama.cpp's <code>rope_neox</code>,
          Hugging Face's <code>rotate_half</code>, mlx-lm's default <code>rope_traditional = False</code>, and SmolLM2's{' '}
          <code>"rope_interleaved": false</code>.
        </p>
      </Translated>
      <Today checked={tr(`2026-11-20 in the facts pass (${X3} B6, C1)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${X3} B6, C1)`)}>
        <Translated
          vi={
            <p>
              <code>llama_model_rope_type</code> trả về <code>LLAMA_ROPE_TYPE_NORM</code> cho <code>LLM_ARCH_LLAMA</code>. MLX
              chọn cách ghép cặp bằng một function constant trên Metal và một tham số template trên CUDA. PyTorch không có operator
              RoPE.
            </p>
          }
        >
          <p>
            <code>llama_model_rope_type</code> returns <code>LLAMA_ROPE_TYPE_NORM</code> for <code>LLM_ARCH_LLAMA</code>. MLX
            picks the pairing with a function constant on Metal and a template parameter on CUDA. PyTorch has no RoPE
            operator.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Vậy llama.cpp chạy Llama với các cặp liên tiếp, trong khi các checkpoint Hugging Face được huấn luyện với cách chia nửa.
            Bộ chuyển đổi của nó nối hai bên bằng cách hoán vị các hàng của q_proj và k_proj, để mỗi cặp liên tiếp chứa đúng thứ mà
            cách chia nửa đã ghép: <code>permute(weights, n_head, n_head_kv)</code> trong <code>conversion/llama.py</code>, với{' '}
            <code>n_head_kv</code> cho k_proj.
          </p>
        }
      >
        <p>
          So llama.cpp runs Llama with consecutive pairs, while Hugging Face checkpoints were trained with the half split. Its
          converter bridges them by permuting the rows of q_proj and k_proj, so that each consecutive pair holds what the half
          split paired: <code>permute(weights, n_head, n_head_kv)</code> in <code>conversion/llama.py</code>, with{' '}
          <code>n_head_kv</code> for k_proj.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Trong trình khám phá bên dưới: một head, một đầu vào 6 giá trị, một bộ trọng số ngẫu nhiên cố định cho mỗi kích thước head. cos và sin của JavaScript, bằng double,
              đứng thay cho bộ dựng bảng của host; bản thân phép quay được làm tròn về float ở mọi phép toán, như một kernel tính nó,
              không có fused multiply-add.
            </p>
          }
        >
          <p>
            In the explorer below: one head, a 6-value input, one fixed set of random weights per head size. JavaScript's cos and sin, in double,
            stand in for the host's table builder; the rotation itself is rounded to float at every operation, as a kernel
            computes it, without fused multiply-adds.
          </p>
        </Translated>
      </Simplification>
      <RopePairsScene />
      <TwoPairings />
      <Translated
        vi={
          <p>
            Điều cần để ý. Cách ghép cặp sai không làm sập gì và cho ra một đường cong trơn tru, đáng tin. Phép hoán vị không chỉ đưa nó
            đến gần tham chiếu mà làm nó giống hệt tham chiếu, vì khi đó kernel thực hiện đúng các phép nhân trên đúng các giá trị theo một thứ
            tự bộ nhớ khác. Và hoán vị trọng số cho một kernel vốn đã dùng cách chia nửa lại làm hỏng nó: phép chuyển đổi thuộc về
            một cách ghép cặp.
          </p>
        }
      >
        <p>
          What to notice. The wrong pairing crashes nothing and produces a smooth, believable curve. The permutation does not
          bring it close to the reference; it makes it identical, because the kernel then does the same multiplications on the
          same values in another order of memory. And permuting the weights for a kernel that already uses the half split
          breaks it again: the conversion belongs to one pairing.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Trong transformers, phép quay là <code>rotate_half</code> bên trong <code>modeling_llama.py</code>, các phép toán tensor
              thông thường trên q và k. Đó là dòng cần đặt cạnh kernel của bạn khi kiểm tra một cách ghép cặp.
            </p>
          }
        >
          <p>
            In transformers the rotation is <code>rotate_half</code> inside <code>modeling_llama.py</code>, ordinary tensor
            operations on q and k. That is the line to put next to your kernel when you check a pairing.
          </p>
        </Translated>
      </Aside>
      <Ask q={q['x3/X3.7/q-rope-mismatch']} />
      <Ask q={q['x3/X3.7/q-permute-k']} />

      <h2><Translated vi="Lựa chọn của khóa học: bảng từ host">The course's choice: tables from the host</Translated></h2>
      <Translated
        vi={
          <p>
            llama.cpp tính góc bên trong kernel: <code>powf</code> cho tần số, rồi <code>cosf</code> và{' '}
            <code>sinf</code>. Khóa học thì không. Host của lab X3.1 tính mọi cos và sin một lần, bằng double, làm tròn về float và
            truyền hai bảng; kernel chỉ nhân. Khi đó phép quay giống hệt nhau trên CUDA, HIP, Metal và bản tham chiếu CPU, bất kể độ
            chính xác lượng giác của mỗi thiết bị, và chế độ toán học mặc định của Metal không thể động đến nó. Các file cũ nhất của
            llama2.c cũng từng lưu các bảng như vậy, dưới tên freq_cis; run.c ngày nay bỏ qua chúng.
          </p>
        }
      >
        <p>
          llama.cpp computes the angle inside the kernel: <code>powf</code> for the frequency, then <code>cosf</code> and{' '}
          <code>sinf</code>. The course does not. Lab X3.1's host computes every cos and sin once, in double, rounds them to
          float and passes the two tables; the kernel only multiplies. The rotation is then identical on CUDA, HIP, Metal and
          the CPU reference, whatever each device's trigonometric accuracy, and Metal's default math mode cannot touch it.
          llama2.c's oldest files stored such tables too, as freq_cis; today's run.c skips them.
        </p>
      </Translated>
      <Today checked={tr(`2026-11-20 in the facts pass (${X3} B6, D1; notes/today/ch05-facts.md C7)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${X3} B6, D1; notes/today/ch05-facts.md C7)`)}>
        <Translated
          vi={
            <p>
              llama.cpp: <code>theta_base = pos[i2]*powf(theta_scale, iw/2.0f)</code>, rồi <code>cosf(theta) * mscale</code>.
              Metal: <code>fastMathEnabled</code> mặc định là YES; các nguồn của Apple không thống nhất về chế độ mặc định.
            </p>
          }
        >
          <p>
            llama.cpp: <code>theta_base = pos[i2]*powf(theta_scale, iw/2.0f)</code>, then <code>cosf(theta) * mscale</code>.
            Metal: <code>fastMathEnabled</code> defaults to YES; Apple's sources disagree on the default mode.
          </p>
        </Translated>
      </Today>
      <RopeListing />
      <Ask q={q['x3/X3.7/q-host-tables']} />
      <Translated
        vi={
          <p>
            Lab X3.1 xây RMSNorm, RoPE ở cả hai cách ghép cặp và SwiGLU; lab X3.2 ghép cả bước lại và so sánh các token của nó với
            một bản tham chiếu CPU. Cả hai đều ở <a href={href('x3', 'labs')}>trang lab</a>. Phần{' '}
            <a href={href('x3', 'X3.1')}>X3.1</a> ánh xạ mỗi lần khởi chạy ở trên đến file của nó trong llama.cpp, MLX và PyTorch.
          </p>
        }
      >
        <p>
          Lab X3.1 builds RMSNorm, RoPE in both pairings and SwiGLU; lab X3.2 composes the whole step and compares its tokens
          with a CPU reference. Both are on the <a href={href('x3', 'labs')}>labs page</a>. Section{' '}
          <a href={href('x3', 'X3.1')}>X3.1</a> maps each launch above to its file in llama.cpp, MLX and PyTorch.
        </p>
      </Translated>

      <h2><Translated vi="Trên HIP và Metal">On HIP and Metal</Translated></h2>
      <Translated
        vi={
          <p>
            <code>rope.cu</code> và <code>norm.cu</code> của llama.cpp cũng là các file HIP của nó; Metal có{' '}
            <code>kernel_rope_norm</code> và <code>kernel_rope_neox</code> riêng. RoPE CUDA của MLX không có bản tương ứng trên HIP, vì MLX
            không có backend HIP. Ba kernel của khóa học là một file <code>.cu</code> cho CUDA, HIP và các trình giả lập, và một{' '}
            <code>kernel.metal</code> làm cùng phép tính trên cùng các bảng. Một file đó phục vụ được HIP chỉ vì thao tác warp shuffle
            của nó nhận một lane mask có kiểu và đọc <code>warpSize</code>: các intrinsic <code>_sync</code> của HIP cần một mask 64
            bit, nên một literal <code>0xffffffff</code> không khả chuyển. Nó chưa được chạy trên HIP.
          </p>
        }
      >
        <p>
          llama.cpp's <code>rope.cu</code> and <code>norm.cu</code> are also its HIP files; Metal has its own{' '}
          <code>kernel_rope_norm</code> and <code>kernel_rope_neox</code>. MLX's CUDA RoPE has no HIP counterpart, because MLX
          has no HIP backend. The course's three kernels are one <code>.cu</code> file for CUDA, HIP and the emulators, and a{' '}
          <code>kernel.metal</code> that does the same arithmetic on the same tables. The one file serves HIP only because its
          warp shuffle takes a typed lane mask and reads <code>warpSize</code>: HIP's <code>_sync</code> intrinsics need a
          64-bit mask, so a <code>0xffffffff</code> literal is not portable. It has not been run on HIP.
        </p>
      </Translated>
      <Today checked={tr(`2026-11-20 in the facts pass (${X3} B6, B9)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${X3} B6, B9)`)}>
        <Translated
          vi={
            <p>
              <code>ggml-hip/CMakeLists.txt</code> biên dịch <code>../ggml-cuda/*.cu</code> thành HIP. <code>mlx/backend/</code>{' '}
              không có thư mục HIP hay ROCm.
            </p>
          }
        >
          <p>
            <code>ggml-hip/CMakeLists.txt</code> compiles <code>../ggml-cuda/*.cu</code> as HIP. <code>mlx/backend/</code>{' '}
            lists no HIP or ROCm directory.
          </p>
        </Translated>
      </Today>
    </>
  );
}
