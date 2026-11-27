import { useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { NMSE_LLAMACPP, SLIDER, START_S, STEPS, compare, nmse, type Comparison } from '../../sim/tokencompare';
import { LateDivergenceScene } from './LateDivergenceScene';

// Section X3.9: testing a GPU run against a reference, and contributing. The explorer and the
// scene run on src/sim/tokencompare.ts (pinned in tokencompare.test.ts). Vendor and project
// facts are from notes/today/x3-facts.md D7, D8, F1, F2 (and B3, ch10 D3 for MLX's softmax).

const MONO: CSSProperties = { fontFamily: 'var(--sans)', fontVariationSettings: "'MONO' 1" };// i18n-ignore: CSS value
const e1 = (v: number) => (v === 0 ? '0' : v.toExponential(1));
const sLabel = (e: number) => {
  const [m, x] = (10 ** e).toExponential(1).split('e');
  return (
    <>
      {m} × 10<sup>{String(Number(x)).replace('-', '−')}</sup>
    </>
  );
};

// ---------------------------------------------------------------- charts

const CW = 720;
const CH = 150;
const L = 52;
const R = 12;
const T = 24;
const B = 24;
const xAt = (t: number) => L + ((t + 0.5) / STEPS) * (CW - L - R);

function LogChart({ lo, hi, label, children, part }: { lo: number; hi: number; label: string; children: (y: (v: number) => number) => ReactNode; part: string }) {
  const y = (v: number) => {
    const l = Math.log10(Math.max(v, 10 ** lo));
    return T + ((hi - Math.min(l, hi)) / (hi - lo)) * (CH - T - B);
  };
  const ticks: number[] = [];
  for (let e = Math.ceil(lo); e <= hi; e += lo < -12 ? 4 : 2) ticks.push(e);
  return (
    <div style={{ overflowX: 'auto' }}>
    <svg viewBox={`0 0 ${CW} ${CH}`} role="img" aria-label={label} data-part={part}
      style={{ width: '100%', minWidth: 540, display: 'block', background: 'var(--page)', border: '1px solid var(--rule)', borderRadius: 4 }}>
      {ticks.map((e) => (
        <g key={e}>
          <line x1={L} y1={y(10 ** e)} x2={CW - R} y2={y(10 ** e)} stroke="var(--rule-soft)" />
          <text x={L - 5} y={y(10 ** e) + 3.5} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>1e{e}</text>
        </g>
      ))}
      {Array.from({ length: STEPS }, (_, t) => (t % 2 === 0 ? (
        <text key={t} x={xAt(t)} y={CH - 8} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{t}</text>
      ) : null))}
      <text x={L} y={12} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{label}</text>
      {children(y)}
    </svg>
    </div>
  );
}

function NmseChart({ c }: { c: Comparison }) {
  return (
    <LogChart lo={-18} hi={2} label={tr('NMSE per step against the reference (log scale); x: step', 'NMSE mỗi bước so với tham chiếu (thang log); x: bước')} part="nmse" /* i18n-ignore: data-part key */>
      {(y) => (
        <>
          <line x1={L} y1={y(NMSE_LLAMACPP)} x2={CW - R} y2={y(NMSE_LLAMACPP)} stroke="var(--ink)" strokeDasharray="5 3" />
          <text x={CW - R - 4} y={y(NMSE_LLAMACPP) - 4} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink)' }}><Translated vi="mặc định của llama.cpp cho mỗi op, 1e-7">llama.cpp's default per op, 1e-7</Translated></text>
          {c.steps.map((r) => (
            <g key={r.t}>
              <circle cx={xAt(r.t) + 4} cy={y(r.frNmse)} r={3.6} fill="none" stroke={r.frTok !== r.refTok || r.frIn !== r.refIn ? 'var(--fail)' : 'var(--ink-3)'} strokeWidth={1.4} />
              <circle cx={xAt(r.t) - 4} cy={y(r.tfNmse)} r={3.4} fill={r.tfTok !== r.refTok ? 'var(--fail)' : 'var(--ink)'} />
            </g>
          ))}
        </>
      )}
    </LogChart>
  );
}

function MarginChart({ c }: { c: Comparison }) {
  const lo = -7;
  return (
    <LogChart lo={lo} hi={1} label={tr('Reference top-two margin per step (log scale); bars at or below the line are flagged', 'Khoảng cách giữa hai logit cao nhất của tham chiếu mỗi bước (thang log); các cột chạm hoặc dưới đường kẻ bị gắn cờ')} part="margins" /* i18n-ignore: data-part key */>
      {(y) => (
        <>
          {c.steps.map((r) => (
            <rect key={r.t} x={xAt(r.t) - 7} y={y(r.margin)} width={14} height={Math.max(1, y(10 ** lo) - y(r.margin))} rx="1.5"
              fill={r.flagged ? 'var(--fail)' : 'var(--ink-3)'} opacity={r.flagged ? 0.9 : 0.55} data-step={r.t} data-flagged={r.flagged ? 'yes' : 'no'} />
          ))}
          <line x1={L} y1={y(2 * c.tol)} x2={CW - R} y2={y(2 * c.tol)} stroke="var(--ink)" strokeDasharray="5 3" />
          <text x={CW - R - 4} y={y(2 * c.tol) - 4} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink)' }}>
            <Translated vi={<>2 × sai số logit lớn nhất = {e1(2 * c.tol)}</>}>
              2 × largest logit error = {e1(2 * c.tol)}
            </Translated>
          </text>
        </>
      )}
    </LogChart>
  );
}

function Streams({ c }: { c: Comparison }) {
  const rows = [
    { id: 'ref', label: tr('reference', 'tham chiếu'), tok: c.steps.map((r) => r.refTok) },
    { id: 'fr', label: tr('GPU, free running', 'GPU, chạy tự do'), tok: c.steps.map((r) => r.frTok) },
    { id: 'tf', label: tr('GPU, teacher forced', 'GPU, teacher forcing'), tok: c.steps.map((r) => r.tfTok) },
  ];
  const cell: CSSProperties = { ...MONO, width: '1.55rem', textAlign: 'center', padding: '0.12rem 0', borderRadius: 3, fontSize: 'var(--step--1)' };// i18n-ignore: CSS value
  return (
    <div style={{ overflowX: 'auto' }} data-part="streams">
      <table style={{ borderCollapse: 'separate', borderSpacing: 2, margin: 0 }}>
        <thead>
          <tr>
            <th />
            {c.steps.map((r) => (
              <th key={r.t} style={{ ...cell, fontWeight: 400, color: r.flagged ? 'var(--fail)' : 'var(--ink-3)' }} title={r.flagged ? tr('flagged: margin too small to judge', 'bị gắn cờ: khoảng cách quá nhỏ để phán xét') : undefined}>
                {r.flagged ? '?' : r.t}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id} data-row={row.id}>
              <td style={{ paddingRight: '0.5rem', whiteSpace: 'nowrap', color: 'var(--ink-2)', fontSize: 'var(--step--1)' }}>{row.label}</td>
              {row.tok.map((tok, t) => {
                const bad = row.id !== 'ref' && tok !== c.steps[t].refTok;
                const first = row.id === 'fr' && t === c.firstDivergence;
                return (
                  <td key={t} data-bad={bad ? 'yes' : 'no'}
                    style={{ ...cell, background: bad ? 'var(--fail-wash)' : 'var(--cell)', color: bad ? 'var(--fail)' : 'var(--ink)', border: `1px solid ${first ? 'var(--fail)' : bad ? 'var(--fail-wash)' : 'var(--rule)'}`, outline: first ? '1px solid var(--fail)' : undefined }}>
                    {tok}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CompareRuns() {
  const [e, setE] = useState(Math.log10(START_S));
  const c = useMemo(() => compare(10 ** e), [e]);
  const flips = c.tfDiffers;
  return (
    <Bench
      title={tr('Compare two runs', 'So sánh hai lần chạy')}
      task={
        <Translated vi="Kéo nhiễu lên. Sự trùng khớp token thất bại lần đầu ở đâu, và lúc đó NMSE nói gì?">
          Drag the perturbation up. Where does token agreement first fail, and what does NMSE say then?
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Một mô hình đồ chơi gồm 16 token và 24 bước greedy; "GPU" cộng một mẫu nhiễu cố định, nhân theo thanh trượt, vào mọi logit.
              Chấm đặc: NMSE khi teacher forcing. Vòng tròn: NMSE khi chạy tự do (màu đỏ khi đầu vào đã khác nhau). Các dòng token: ô đỏ khác
              với tham chiếu; dấu ? trên một bước đánh dấu nó bị gắn cờ.
            </>
          }
        >
          <>
            A toy model of 16 tokens and 24 greedy steps; the "GPU" adds a fixed pattern of noise, scaled by the slider, to every logit.
            Filled dots: teacher-forced NMSE. Rings: free-running NMSE (red once the inputs differ). Streams: red cells differ from the
            reference; a ? over a step marks it flagged.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ flex: '1 1 18rem' }}>
          <Translated vi="nhiễu trên mỗi logit, tối đa s =">perturbation per logit, up to s =</Translated> <span data-part="s" data-s={10 ** e}>{sLabel(e)}</span>
          <input type="range" min={SLIDER.min} max={SLIDER.max} step={SLIDER.step} value={e} aria-label={tr('perturbation size', 'độ lớn nhiễu')}
            onChange={(ev) => setE(Math.round(Number(ev.target.value) * 10) / 10)} style={{ width: '100%' }} />
        </label>
      </div>
      <NmseChart c={c} />
      <div style={{ height: '0.6rem' }} />
      <MarginChart c={c} />
      <div style={{ height: '0.8rem' }} />
      <Streams c={c} />
      <p style={{ margin: '0.8rem 0 0' }} data-part="readout" data-first={c.firstDivergence ?? ''} data-fr={c.frDiffers.length}
        data-tf={flips.join(',')} data-flagged={c.flagged.join(',')} data-max-nmse={c.maxTfNmse}>
        <Translated
          vi={
            <>
              {c.firstDivergence == null ? (
                <>
                  <span className="chip ok">chạy tự do: cả {STEPS} token đều khớp</span>{' '}
                </>
              ) : (
                <>
                  <span className="chip fail">chạy tự do: phân kỳ đầu tiên ở bước {c.firstDivergence}</span>, sau đó {c.frDiffers.length} trong {STEPS}{' '}
                  token khác nhau.{' '}
                </>
              )}
              Teacher forcing: {flips.length ? <>bước <b>{flips.join(', ')}</b> khác nhau</> : 'không bước nào khác nhau'}
              {c.flagged.length ? <>; gắn cờ bước {c.flagged.join(', ')}</> : '; không gì bị gắn cờ'}. NMSE teacher forcing
              lớn nhất <b>{e1(c.maxTfNmse)}</b>, {c.maxTfNmse < NMSE_LLAMACPP ? 'dưới' : 'trên'} ngưỡng 1e-7 của llama.cpp.
            </>
          }
        >
          {c.firstDivergence == null ? (
            <>
              <span className="chip ok">free running: all {STEPS} tokens match</span>{' '}
            </>
          ) : (
            <>
              <span className="chip fail">free running: first divergence at step {c.firstDivergence}</span>, then {c.frDiffers.length} of {STEPS}{' '}
              tokens differ.{' '}
            </>
          )}
          Teacher forced: {flips.length ? <>{flips.length === 1 ? 'step' : 'steps'} <b>{flips.join(', ')}</b> differ</> : 'no step differs'}
          {c.flagged.length ? <>; flagged {c.flagged.length === 1 ? 'step' : 'steps'} {c.flagged.join(', ')}</> : '; nothing flagged'}. Largest
          teacher-forced NMSE <b>{e1(c.maxTfNmse)}</b>, {c.maxTfNmse < NMSE_LLAMACPP ? 'under' : 'over'} llama.cpp's 1e-7.
        </Translated>
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- the four entry points

const ENTRY: { fw: string; call: string; backend: string; note: string }[] = [
  {
    fw: 'MLX', call: 'mx.fast.metal_kernel(name, input_names, output_names, source, ...)', backend: 'Metal',
    get note() {
      return tr('source is the function body; MLX writes the signature. math_mode "safe" by default.', 'source là thân hàm; MLX viết chữ ký hàm. Mặc định math_mode là "safe".');
    },
  },
  {
    fw: 'MLX', call: 'mx.fast.cuda_kernel(...)', backend: 'CUDA',
    get note() {
      return tr('grid counts threads, not blocks, "for compatibility with metal_kernel".', 'grid đếm số luồng, không phải số khối, "for compatibility with metal_kernel".');
    },
  },
  {
    fw: 'PyTorch', call: 'torch.mps.compile_shader(source)', backend: 'MPS (Metal)',
    get note() {
      return tr('a whole MSL kernel; call it as lib.<kernel name>(tensors, values).', 'một kernel MSL hoàn chỉnh; gọi nó dưới dạng lib.<kernel name>(tensors, values).');
    },
  },
  {
    fw: 'PyTorch', call: 'torch.utils.cpp_extension.load_inline(name, cpp_sources, cuda_sources=None, ...)',
    get backend() {
      return tr('CUDA; ROCm builds', 'CUDA; các bản build ROCm');
    },
    get note() {
      return tr('a C++ extension compiled at run time. ROCm not checked separately.', 'một extension C++ được biên dịch lúc chạy. ROCm chưa được kiểm tra riêng.');
    },
  },
  {
    fw: 'llama.cpp',
    get call() {
      return tr('none: a new GGML_OP_*', 'không có: một GGML_OP_* mới');
    },
    get backend() {
      return tr('every backend', 'mọi backend');
    },
    get note() {
      return tr('CPU first, then each backend, each with test-backend-ops cases.', 'CPU trước, rồi từng backend, mỗi backend kèm các ca test-backend-ops.');
    },
  },
];

// ---------------------------------------------------------------- questions

export const questionsSX39: Record<string, Question> = {
  'x3/X3.9/q-nmse': {
    kind: 'numeric',
    id: 'x3/X3.9/q-nmse',
    prompt: (
      <Translated vi={<>Tham chiếu tính ra a = (1, 2, 2) và backend của bạn ra b = (1, 2, 2.03). test-backend-ops báo NMSE bằng bao nhiêu?</>}>
        <>
          The reference computes a = (1, 2, 2) and your backend b = (1, 2, 2.03). What NMSE does test-backend-ops report?
        </>
      </Translated>
    ),
    answer: nmse([1, 2, 2], [1, 2, 2.03]),
    tol: 1e-6,
    explain: (
      <Translated
        vi={
          <p>
            Tổng bình phương các hiệu: 0.03² = 0.0009. Tổng bình phương của tham chiếu: 1 + 4 + 4 = 9. NMSE = 0.0009 / 9 = 1e-4, gấp một
            nghìn lần ngưỡng mặc định 1e-7 cho mỗi op của llama.cpp. Chỉ riêng một giá trị đó đã lệch 1.5%, đó là lỗi, không phải sai số làm tròn; ngưỡng
            chặt hơn thế rất nhiều.
          </p>
        }
      >
        <p>
          Sum of squared differences: 0.03² = 0.0009. Sum of squares of the reference: 1 + 4 + 4 = 9. NMSE = 0.0009 / 9 = 1e-4, a thousand
          times llama.cpp's default threshold of 1e-7 per op. That one value is off by 1.5%, which is a bug, not rounding; the threshold is
          far tighter than that.
        </p>
      </Translated>
    ),
    source: 'x3-facts D7 (tests/test-backend-ops.cpp, nmse and max_nmse_err)',
    vi: { source: 'x3-facts D7 (tests/test-backend-ops.cpp, nmse và max_nmse_err)' },
  },
  'x3/X3.9/q-late-flip': {
    kind: 'choice',
    id: 'x3/X3.9/q-late-flip',
    prompt: (
      <Translated
        vi={
          <>
            Lần chạy GPU của bạn khớp với tham chiếu từng token một trong 40 bước; từ bước 41 mọi token đều khác. Logit khi teacher forcing
            khớp với tham chiếu tới NMSE 1e-10 ở mọi bước. Khả năng cao nhất là gì?
          </>
        }
      >
        <>
          Your GPU run matches the reference token for token for 40 steps; from step 41 every token differs. Teacher-forced logits agree with
          the reference to an NMSE of 1e-10 at every step. What is most likely?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Hai logit cao nhất của bước 41 cách nhau trong phạm vi sai số làm tròn; hãy kiểm tra khoảng cách của nó.">Step 41’s top two logits were within rounding of each other; check its margin.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Một kernel có lỗi chỉ lộ ra sau 40 token, chẳng hạn một KV cache tràn khỏi vùng cấp phát.">A kernel has a bug that only shows after 40 tokens, such as a KV cache that overflows its allocation.</Translated>,
        why: <Translated vi="Khi đó logit teacher forcing ở bước 41 trở đi cũng sẽ lệch, lệch hơn 1e-10 rất nhiều.">Then teacher-forced logits at step 41 and later would disagree too, by far more than 1e-10.</Translated>,
      },
      {
        label: <Translated vi="Tham chiếu sai từ bước 41 trở đi, và GPU của bạn mới là bên tính đúng.">The reference is wrong from step 41 on, and your GPU is the one computing correctly.</Translated>,
        why: <Translated vi="Có thể nhưng không có dấu hiệu: logit teacher forcing khớp với nó ở mọi nơi.">Possible but not indicated: the teacher-forced logits agree with it everywhere.</Translated>,
      },
      {
        label: <Translated vi="NMSE không thấy được lỗi ở các bước muộn, vì nó lấy trung bình trên cả chuỗi.">NMSE cannot see errors in late steps, because it averages over the whole sequence.</Translated>,
        why: <Translated vi="Nó được tính cho từng bước; ở 1e-10 mọi bước đều khớp.">It is computed per step; at 1e-10 every step agrees.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Khớp tới 1e-10 nghĩa là mọi kernel tính cùng một hàm như tham chiếu, sai khác chỉ do làm tròn. Một lần chạy tự do mà sau đó phân
            kỳ hoàn toàn từ một bước trở đi đã gặp một bước có hai logit cao nhất gần nhau hơn mức chênh lệch làm tròn: GPU chọn ứng viên thứ
            hai, và mọi bước sau đó bắt đầu từ một token khác. Hãy tra khoảng cách của bước 41. Nếu nó nhỏ hơn hai lần sai số logit, bước đó
            quá sát để phán xét, và lần chạy không phải là thất bại.
          </p>
        }
      >
        <p>
          Agreement to 1e-10 says every kernel computes the same function as the reference, up to rounding. A free-running run that then
          diverges completely from one step on has met a step whose top two logits were closer than the rounding difference: the GPU took the
          runner-up, and every later step started from a different token. Look up step 41's margin. If it is below twice the logit error, the
          step is too close to judge, and the run is not a failure.
        </p>
      </Translated>
    ),
    source: 'ours, from x3-facts D7 (two tests for M10)',
    vi: { source: 'của chúng tôi, từ x3-facts D7 (hai phép kiểm tra cho M10)' },
  },
  'x3/X3.9/q-cuda-grid': {
    kind: 'choice',
    id: 'x3/X3.9/q-cuda-grid',
    prompt: (
      <Translated
        vi={
          <>
            Bạn gọi <code>mx.fast.cuda_kernel</code> với <code>grid=(n // 256, 1, 1)</code> và <code>threadgroup=(256, 1, 1)</code>, như
            khi bạn khởi chạy một kernel CUDA trên n phần tử. Cái gì sẽ chạy?
          </>
        }
      >
        <>
          You call <code>mx.fast.cuda_kernel</code> with <code>grid=(n // 256, 1, 1)</code> and <code>threadgroup=(256, 1, 1)</code>, as you
          would launch a CUDA kernel over n elements. What runs?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Chỉ n / 256 luồng: grid của MLX đếm số luồng, không phải số khối, nên phần lớn phần tử bị bỏ qua.">Only n / 256 threads: MLX’s grid counts threads, not blocks, so most elements are skipped.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="n luồng, như trong CUDA: n / 256 khối, mỗi khối 256 luồng.">n threads, as in CUDA: n / 256 blocks of 256 threads each.</Translated>,
        why: <Translated vi="Đó là nghĩa của grid trong CUDA: một số đếm khối. Của MLX là số đếm luồng.">That is CUDA’s meaning of the grid: a count of blocks. MLX’s is a count of threads.</Translated>,
      },
      {
        label: <Translated vi="256 × n luồng, vì MLX nhân grid với kích thước threadgroup.">256 × n threads, because MLX multiplies the grid by the threadgroup size.</Translated>,
        why: <Translated vi="Không có phép nhân nào ở đây; grid đã là tổng số.">Nothing multiplies here; the grid is already the total.</Translated>,
      },
      {
        label: <Translated vi="Một lỗi: cuda_kernel của MLX không nhận tham số threadgroup, vì nó tự chọn kích thước khối.">An error: MLX’s cuda_kernel takes no threadgroup argument, because it chooses the block size itself.</Translated>,
        why: <Translated vi="Cả kích thước grid lẫn threadgroup đều được truyền vào; chỉ đơn vị của grid là khác.">The grid and threadgroup sizes are both given; only the grid’s unit differs.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Tài liệu của MLX cho <code>cuda_kernel</code> nói grid tính bằng luồng chứ không bằng threadgroup, để tương thích với{' '}
            <code>metal_kernel</code>. Thói quen CUDA truyền số khối sẽ khởi chạy ít hơn 256 lần số luồng cần, và một kernel có kiểm tra biên
            thất bại trong im lặng: phần lớn đầu ra không bao giờ được ghi. Hãy truyền <code>grid=(n, 1, 1)</code>.
          </p>
        }
      >
        <p>
          MLX's documentation for <code>cuda_kernel</code> says the grid is in threads and not in threadgroups, for compatibility with{' '}
          <code>metal_kernel</code>. A CUDA habit of passing the number of blocks launches 256 times too few threads, and a kernel with a
          bounds check fails silently: most of the output is never written. Pass <code>grid=(n, 1, 1)</code>.
        </p>
      </Translated>
    ),
    source: 'x3-facts D8 (python/src/fast.cpp)',// i18n-ignore: citation only
  },
  'x3/X3.9/q-math-mode': {
    kind: 'choice',
    id: 'x3/X3.9/q-math-mode',
    prompt: (
      <Translated
        vi={
          <>
            Bạn chuyển softmax của lab 10.3 vào <code>mx.fast.metal_kernel</code>. Trên cùng đầu vào, nó khác <code>mx.softmax</code> ở
            những bit cuối. Các dữ kiện ủng hộ cách giải thích nào?
          </>
        }
      >
        <>
          You port lab 10.3's softmax into <code>mx.fast.metal_kernel</code>. On the same input it differs from <code>mx.softmax</code> in the
          last bits. Which explanation do the facts support?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi='metal_kernel mặc định biên dịch với math_mode "safe" trong khi softmax của MLX gọi fast::exp, và các phép cộng chạy theo thứ tự khác.'>
            metal_kernel compiles in math_mode "safe" by default while MLX’s softmax calls fast::exp, and the sums run in other orders.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Bản chuyển của bạn hẳn có lỗi: một bản chuyển đúng của cùng thuật toán trên cùng GPU sẽ khớp mx.softmax từng bit.">Your port must have a bug: a correct port of the same algorithm on the same GPU matches mx.softmax bit for bit.</Translated>,
        why: <Translated vi="Cách cài đặt exp khác nhau và thứ tự cộng khác nhau làm thay đổi những bit cuối của một kết quả đúng.">Different exp implementations and summation orders change the last bits of a correct result.</Translated>,
      },
      {
        label: <Translated vi="mx.softmax mặc định chạy trên CPU, nên bạn đang so một kết quả CPU với một kết quả GPU.">mx.softmax runs on the CPU by default, so you are comparing a CPU result with a GPU one.</Translated>,
        why: <Translated vi="Nó có một kernel Metal, softmax_single_row.">It has a Metal kernel, softmax_single_row.</Translated>,
      },
      {
        label: <Translated vi="metal_kernel chuyển đầu vào sang độ chính xác một nửa (half).">metal_kernel converts the input to half precision.</Translated>,
        why: <Translated vi="Tài liệu của nó không nói vậy; dtype là do bạn chọn.">Nothing in its documentation says so; the dtypes are yours.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Softmax Metal của MLX dùng <code>fast::exp</code>; một kernel tùy chỉnh được biên dịch với math mode "safe" trừ khi bạn chỉ định
            khác. Thêm một thứ tự rút gọn khác (<code>simd_sum</code> hai giai đoạn của lab 10.3 so với của chính MLX), và kết quả giống hệt
            nhau mới là điều bất ngờ. Hãy so sánh với một dung sai, như mọi lab trong khóa học.
          </p>
        }
      >
        <p>
          MLX's Metal softmax uses <code>fast::exp</code>; a custom kernel is compiled with math mode "safe" unless you say otherwise. Add a
          different reduction order (lab 10.3's two-stage <code>simd_sum</code> against MLX's own), and identical results would be the
          surprise. Compare with a tolerance, as every lab in the course does.
        </p>
      </Translated>
    ),
    source: 'x3-facts D8 (math_mode), B3 and ch10-facts D3 (fast::exp in softmax_single_row); ch10-facts A9 (summation order)',
    vi: { source: 'x3-facts D8 (math_mode), B3 và ch10-facts D3 (fast::exp trong softmax_single_row); ch10-facts A9 (thứ tự cộng)' },
  },
  'x3/X3.9/q-new-op': {
    kind: 'choice',
    id: 'x3/X3.9/q-new-op',
    prompt: (
      <Translated vi={<>Bạn muốn thêm một op gộp mới vào llama.cpp, với một kernel CUDA và một kernel Metal. CONTRIBUTING.md của nó yêu cầu gì?</>}>
        <>You want to add a new fused op to llama.cpp, with a CUDA and a Metal kernel. What does its CONTRIBUTING.md ask?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Hỗ trợ CPU trước, rồi từng backend, cùng các ca test-backend-ops và một lần chạy trên hai backend.">CPU support first, then each backend, plus test-backend-ops cases and a run on two backends.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Đăng ký kernel của bạn qua API plug-in của backend; một op chỉ cho GPU không cần phiên bản CPU.">Register your kernel through the backend plug-in API; no CPU version is needed for a GPU-only op.</Translated>,
        why: <Translated vi="Không có API plug-in nào: một op mới là một GGML_OP_* có cài đặt trên CPU.">There is no plug-in API: a new op is a GGML_OP_* with a CPU implementation.</Translated>,
      },
      {
        label: <Translated vi="Một benchmark cho thấy tăng tốc so với các op chưa gộp trên mọi backend mà op hỗ trợ.">A benchmark showing a speed-up over the unfused ops on every backend the op supports.</Translated>,
        why: <Translated vi="Các quy tắc được trích dẫn yêu cầu kiểm tra tính nhất quán, không yêu cầu tăng tốc trên mọi backend.">The rules quoted ask for consistency tests, not a speed-up on every backend.</Translated>,
      },
      {
        label: <Translated vi="Chỉ kernel CUDA; những người bảo trì sẽ chuyển nó sang Metal và HIP sau khi merge.">The CUDA kernel only; the maintainers port it to Metal and HIP once it is merged.</Translated>,
        why: <Translated vi="HIP đúng là biên dịch tệp CUDA, nhưng Metal là mã riêng, và các quy tắc bắt đầu từ CPU.">HIP does compile the CUDA file, but Metal is separate code, and the rules start from the CPU.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            "When adding support for a new model or feature, focus on CPU support only in the initial PR." Rồi đến từng backend, và với mọi
            toán tử ggml mới hoặc bị thay đổi, các ca trong <code>test-backend-ops</code>, công cụ so sánh một backend với CPU. Một bản sửa lỗi
            cũng cần một bài kiểm tra hồi quy thất bại trước thay đổi và vượt qua sau đó.
          </p>
        }
      >
        <p>
          "When adding support for a new model or feature, focus on CPU support only in the initial PR." Then each backend, and for any new or
          changed ggml operator, cases in <code>test-backend-ops</code>, which compares a backend with the CPU. A bug fix also needs a
          regression test that fails before the change and passes after.
        </p>
      </Translated>
    ),
    source: 'x3-facts D8, F1 (llama.cpp CONTRIBUTING.md)',// i18n-ignore: citation only
  },
  'x3/X3.9/q-ai-policy': {
    kind: 'choice',
    id: 'x3/X3.9/q-ai-policy',
    prompt: (
      <Translated vi={<>Theo chính sách đóng góp của llama.cpp, điều nào sau đây được phép?</>}>
        <>Under llama.cpp's contribution policy, which of these is allowed?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Gửi mã kernel có sự hỗ trợ của AI mà bạn đã kiểm tra từng dòng, có khai báo việc dùng AI.">Submitting AI-assisted kernel code that you have checked line by line, with the AI use disclosed.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Một mô tả pull request do AI viết, có khai báo như vậy.">An AI-written pull request description, disclosed as such.</Translated>,
        why: <Translated vi="Bị cấm: AI không được viết bài đăng, mô tả PR hay trả lời người khác, dù có khai báo hay không.">Prohibited: AI may not write posts, PR descriptions or replies to people, disclosed or not.</Translated>,
      },
      {
        label: <Translated vi="Mã do AI sinh ra vượt qua test-backend-ops, mà không nhắc tới AI.">AI-generated code that passes test-backend-ops, without mentioning the AI.</Translated>,
        why: <Translated vi="Dùng AI mà không khai báo có thể khiến tài khoản bị cấm đóng góp.">Undisclosed AI use may get the account banned from contributing.</Translated>,
      },
      {
        label: <Translated vi="Để AI trả lời câu hỏi của người review thay bạn.">Letting an AI answer the reviewers’ questions on your behalf.</Translated>,
        why: <Translated vi={'"Responding to humans" nằm trong danh sách bị cấm.'}>"Responding to humans" is on the prohibited list.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mã do AI sinh ra được phép, và bạn chịu hoàn toàn trách nhiệm về từng dòng. Bạn phải khai báo AI đã được dùng như thế nào. Dùng AI
            để viết các bài đăng của bạn (báo lỗi, yêu cầu tính năng, mô tả pull request, thảo luận, trả lời) bị nghiêm cấm.
          </p>
        }
      >
        <p>
          AI-generated code is allowed, and you are fully responsible for every line. You must disclose how AI was used. Using AI to write
          your posts (bug reports, feature requests, pull request descriptions, discussions, replies) is strictly prohibited.
        </p>
      </Translated>
    ),
    source: 'x3-facts F1 (CONTRIBUTING.md, AI policy)',
    vi: { source: 'x3-facts F1 (CONTRIBUTING.md, chính sách AI)' },
  },
};

// ---------------------------------------------------------------- the section

export function SX39() {
  return (
    <>
      <Translated
        vi={
          <p>
            GPU của bạn sinh ra token. Chúng có đúng không? "Cùng các token như tham chiếu" nghe như phép kiểm tra hiển nhiên, và đó chính là
            phép kiểm tra thất bại mà chẳng vì lý do chính đáng nào. Trang này cho thấy các dự án so sánh một backend với một tham chiếu ra
            sao, vì sao sự trùng khớp greedy chính xác có thể thất bại muộn dù mọi kernel đều đúng, cách chạy kernel của riêng bạn bên trong
            MLX hoặc PyTorch, và mỗi dự án đòi hỏi gì ở một thay đổi kernel.
          </p>
        }
      >
        <p>
          Your GPU produces tokens. Are they right? "The same tokens as the reference" sounds like the obvious test, and it is the one that
          fails for no good reason. This page shows how the projects compare a backend with a reference, why exact greedy agreement can fail
          late with every kernel correct, how to run your own kernel inside MLX or PyTorch, and what each project asks of a kernel change.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Module X3 là của chúng tôi. Dữ kiện về các dự án lấy từ notes/today/x3-facts.md (D7, D8, F1, F2), đọc tại các commit được ghim
              vào 2026-11-20. Thiết kế hai phép kiểm tra và mô hình đồ chơi cũng là của chúng tôi.
            </p>
          }
        >
          <p>
            Module X3 is ours. Project facts from notes/today/x3-facts.md (D7, D8, F1, F2), read at pinned commits on 2026-11-20. The
            two-test design and the toy model are ours too.
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="llama.cpp kiểm tra một backend như thế nào">How llama.cpp tests a backend</Translated></h2>
      <Translated
        vi={
          <p>
            <code>test-backend-ops</code> của llama.cpp chạy cùng các op GGML trên hai backend và so mỗi kết quả với kết quả của backend CPU.
            Không phải bằng so sánh bằng nhau: mà bằng một sai số chuẩn hóa, NMSE = Σ (a − b)² / Σ a², với a là tham chiếu. Ngưỡng đạt mặc
            định là 1e-7, và một op có thể ghi đè nó. Bạn chọn op bằng <code>-o</code> (ví dụ <code>-o ADD,MUL_MAT</code>) và backend bằng <code>-b</code>:{' '}
            <code>CUDA0</code>, <code>MTL0</code>.
          </p>
        }
      >
        <p>
          llama.cpp's <code>test-backend-ops</code> runs the same GGML ops on two backends and compares each result with the CPU backend's.
          Not by equality: by a normalised error, NMSE = Σ (a − b)² / Σ a², with a the reference. The default pass mark is 1e-7, and an op
          can override it. You pick ops with <code>-o</code> (for example <code>-o ADD,MUL_MAT</code>) and a backend with <code>-b</code>:{' '}
          <code>CUDA0</code>, <code>MTL0</code>.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/x3-facts.md D7; table “Where the backends really differ”)', '2026-11-20 (notes/today/x3-facts.md D7; bảng “Where the backends really differ”)')}>
        <Translated
          vi={
            <p>
              <code>max_nmse_err()</code> mặc định trả về 1e-7 (test-backend-ops.cpp dòng 1266). Backend HIP được kiểm tra bằng cùng công
              cụ, từ cùng mã nguồn CUDA; trên Metal backend là <code>MTL0</code>.
            </p>
          }
        >
          <p>
            <code>max_nmse_err()</code> returns 1e-7 by default (test-backend-ops.cpp line 1266). The HIP backend is tested by the same tool,
            from the same CUDA sources; on Metal the backend is <code>MTL0</code>.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Có hai điều cần để ý. Chia cho Σ a² làm cho thước đo không phụ thuộc vào độ lớn của đầu ra, nên một ngưỡng dùng được cho cả
            softmax lẫn matmul. Và 1e-7 là con số cho mỗi op, trên đầu ra của một op. Nó không nói gì về việc logit của cả một mô hình có thể
            trôi đi bao xa sau ba mươi lớp.
          </p>
        }
      >
        <p>
          Two things to notice. Dividing by Σ a² makes the measure independent of the scale of the output, so one threshold serves a softmax
          and a matmul. And 1e-7 is a figure per op, on one op's output. It says nothing about how far the logits of a whole model may drift
          after thirty layers.
        </p>
      </Translated>
      <Ask q={questionsSX39['x3/X3.9/q-nmse']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Vì sao một phép kiểm tra là không đủ">Why one test is not enough</Translated></h2>
      <Translated
        vi={
          <p>
            Logit từ hai backend khác nhau do làm tròn: thứ tự cộng khác nhau (chương 10), <code>exp</code> được cài đặt khác nhau, và math
            mode mặc định của Metal không phải IEEE nghiêm ngặt. Thường thì mức chênh đó nhỏ hơn nhiều so với khoảng cách giữa hai logit cao
            nhất, và giải mã greedy chọn cùng một token. Nhưng đôi khi hai logit cao nhất gần nhau hơn mức chênh làm tròn. Khi đó GPU có thể
            chọn ứng viên thứ hai, và từ bước đó trở đi nó sinh ra một văn bản khác.
          </p>
        }
      >
        <p>
          Logits from two backends differ by rounding: summation orders differ (chapter 10), <code>exp</code> is implemented differently, and
          Metal's default math mode is not strict IEEE. Usually that difference is far smaller than the gap between the top two logits, and
          greedy decoding picks the same token. But sometimes the top two are closer than the rounding difference. Then the GPU can take the
          runner-up, and from that step on it is generating a different text.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch10-facts.md A9, A11; x1-facts C5; ch05-facts C7)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Phép cộng số thực dấu phẩy động không có tính kết hợp, nên một thứ tự rút gọn khác làm thay đổi những bit cuối; độ chính xác
              của <code>expf</code> khác nhau theo backend; Metal không mặc định dùng phép toán IEEE nghiêm ngặt, và các nguồn của Apple không
              thống nhất về mode nào là mặc định (ch05 C7).
            </p>
          }
        >
          <p>
            Float summation is not associative, so a different reduction order changes the last bits; <code>expf</code>'s accuracy differs by
            backend; Metal does not default to IEEE-strict math, and Apple's sources disagree on which mode is the default (ch05 C7).
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Vì vậy M10 chạy hai phép kiểm tra. <b>Teacher forcing</b>: đưa token của tham chiếu vào cả hai lần chạy, để mọi bước thấy cùng một
            đầu vào, và yêu cầu logit khớp nhau trong một dung sai được nêu rõ. <b>Chạy tự do</b>: để GPU tự đưa lại token của nó và yêu cầu
            cùng các token greedy, nhưng chỉ ở những bước có khoảng cách giữa hai logit cao nhất lớn hơn hai lần dung sai đó. Một bước sát
            hơn thế bị gắn cờ là quá sát để phán xét, không bị tính là thất bại.
          </p>
        }
      >
        <p>
          So M10 runs two tests. <b>Teacher forced</b>: feed both runs the reference's tokens, so every step sees the same input, and require
          the logits to agree within a stated tolerance. <b>Free running</b>: let the GPU feed back its own tokens and require the same greedy
          tokens, but only on steps whose top-two margin is more than twice that tolerance. A step closer than that is flagged as too close to
          judge, not counted as a failure.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Nhiễu là nhân tạo: cùng một mẫu cố định của nhiễu độc lập trên mỗi logit, nhân theo thanh trượt. Khác biệt thật giữa các backend
              có cấu trúc và tương quan giữa các logit và các bước. Logit của mô hình đồ chơi có phản ứng với token trước đó, như một mô hình
              thật, nhưng các bước sát nút của nó được dựng sẵn.
            </p>
          }
        >
          <p>
            The perturbation is synthetic: the same fixed pattern of independent noise per logit, scaled by the slider. Real backend
            differences are structured and correlated across logits and steps. The toy model's logits do react to the previous token, as a
            real model's do, but its close steps are built in.
          </p>
        </Translated>
      </Simplification>
      <CompareRuns />
      <Translated
        vi={
          <p>
            Có ba điều cần tìm. Sự trùng khớp token chính xác thất bại lần đầu ở bước 19, trong khi mọi NMSE vẫn dưới 1e-11, thấp hơn ngưỡng
            của llama.cpp ít nhất mười nghìn lần. Một khi đã thất bại thì nó tiếp tục thất bại: khi chạy tự do, mọi token sau đó đều khác, và
            các vòng tròn trong biểu đồ thứ nhất nhảy lên khoảng 1, vì hai lần chạy không còn tính trên cùng một đầu vào. Khi teacher forcing,
            chỉ những bước sát nút mới có thể khác, và mỗi bước như vậy đều đã bị gắn cờ trước. Kéo tới 1e-3 và phân kỳ đầu tiên dời về bước
            13, trong khi mọi NMSE vẫn dưới 1e-7.
          </p>
        }
      >
        <p>
          Three things to find. Exact token agreement fails first at step 19, while every NMSE is still below 1e-11, at least ten thousand
          times under llama.cpp's threshold. Once it fails it keeps failing: free running, every later token differs, and the rings in the
          first chart jump to about 1, because the two runs are no longer computing on the same input. Teacher forced, only the close steps
          can differ, and each one that does was flagged first. Drag to 1e-3 and the first divergence moves to step 13, with every NMSE
          still under 1e-7.
        </p>
      </Translated>
      <Ask q={questionsSX39['x3/X3.9/q-late-flip']} />
      <LateDivergenceScene />

      <h2><Translated vi="Kernel của bạn bên trong framework bạn dùng">Your kernel inside the framework you use</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn không cần fork một framework để thử một kernel trong đó. MLX và PyTorch đều nhận mã nguồn kernel từ Python và biên dịch nó lúc
            chạy, trên Metal và trên CUDA. llama.cpp không có cánh cửa như vậy: một kernel ở đó là một op GGML mới.
          </p>
        }
      >
        <p>
          You do not need to fork a framework to try a kernel in it. MLX and PyTorch each take kernel source from Python and compile it at run
          time, on Metal and on CUDA. llama.cpp has no such door: a kernel there is a new GGML op.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x3-facts.md D8)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              MLX <code>python/src/fast.cpp</code> ở commit được ghim; tài liệu PyTorch 2.15 cho <code>compile_shader</code> và{' '}
              <code>load_inline</code>. MLX không có backend AMD (B9).
            </p>
          }
        >
          <p>
            MLX <code>python/src/fast.cpp</code> at the pinned commit; PyTorch docs 2.15 for <code>compile_shader</code> and{' '}
            <code>load_inline</code>. MLX has no AMD backend (B9).
          </p>
        </Translated>
      </Today>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ width: '100%' }} data-part="entry">
          <thead>
            <tr><th><Translated vi="Framework">Framework</Translated></th><th><Translated vi="Backend">Backend</Translated></th><th><Translated vi="Điểm vào, và những gì cần biết">Entry point, and what to know</Translated></th></tr>
          </thead>
          <tbody>
            {ENTRY.map((r) => (
              <tr key={r.call}>
                <td>{r.fw}</td>
                <td>{r.backend}</td>
                <td>
                  <code style={{ fontSize: 'var(--step--1)', wordBreak: 'break-word' }}>{r.call}</code>
                  <div className="muted">{r.note}</div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Ask q={questionsSX39['x3/X3.9/q-cuda-grid']} />
      <Translated
        vi={
          <p>
            Cây cầu từ khóa học này: chuyển <code>softmax_rows</code> của lab 10.3 vào <code>mx.fast.metal_kernel</code> và so sánh nó với{' '}
            <code>mx.softmax</code>. Chuỗi mã nguồn chỉ là thân kernel; MLX sinh chữ ký hàm từ tên các đầu vào và đầu ra của bạn.
            Lab 10.3 nhận buffer tạm từ <code>launch.mm</code>; trong thân hàm bạn có thể khai báo nó thay vào đó, dưới dạng{' '}
            <code>threadgroup float scratch[32];</code>. Lần khởi chạy của nó là một threadgroup 256 luồng cho mỗi hàng, nên grid là số hàng × 256 luồng,
            không phải số hàng. Chữ ký được sinh ra cung cấp những thuộc tính chỉ số nào thì có trong tài liệu của MLX cho <code>metal_kernel</code>; trang này
            chưa chạy bản chuyển đó.
          </p>
        }
      >
        <p>
          The bridge from this course: port lab 10.3's <code>softmax_rows</code> into <code>mx.fast.metal_kernel</code> and compare it with{' '}
          <code>mx.softmax</code>. The source string is the kernel's body only; MLX generates the signature from your input and output names.
          Lab 10.3 receives its scratch buffer from <code>launch.mm</code>; in a body you can declare it instead, as{' '}
          <code>threadgroup float scratch[32];</code>. Its launch was one threadgroup of 256 threads per row, so the grid is rows × 256 threads,
          not rows. Which index attributes the generated signature provides is in MLX's documentation for <code>metal_kernel</code>; this page
          has not run the port.
        </p>
      </Translated>
      <Ask q={questionsSX39['x3/X3.9/q-math-mode']} />

      <h2><Translated vi="Đóng góp một thay đổi kernel">Contributing a kernel change</Translated></h2>
      <Translated
        vi={
          <p>
            <b>llama.cpp.</b> Nếu bạn thay đổi ggml, hãy chạy <code>test-backend-ops</code>, công cụ cần hai backend. Nếu bạn thay đổi hoặc
            thêm một toán tử, hãy thêm các ca của nó vào đó. Một bản sửa lỗi cần một issue tái hiện được và một bài kiểm tra hồi quy thất bại
            trước thay đổi của bạn và vượt qua sau đó. Một tính năng mới bắt đầu chỉ với hỗ trợ CPU. Phong cách: 4 dấu cách, dấu ngoặc trên
            cùng dòng, <code>void * ptr</code>,{' '}
            <code>int &amp; a</code>.
          </p>
        }
      >
        <p>
          <b>llama.cpp.</b> If you change ggml, run <code>test-backend-ops</code>, which needs two backends. If you change or add an operator,
          add its cases there. A bug fix needs a reproducible issue and a regression test that fails before your change and passes after. A
          new feature starts with CPU support only. Style: 4 spaces, brackets on the same line, <code>void * ptr</code>,{' '}
          <code>int &amp; a</code>.
        </p>
      </Translated>
      <Ask q={questionsSX39['x3/X3.9/q-new-op']} />
      <Translated
        vi={
          <p>
            Chính sách AI của nó rất rõ ràng. Mã do AI sinh ra được phép, và bạn chịu trách nhiệm về từng dòng. Bạn phải khai báo AI đã được
            dùng như thế nào; dùng mà không khai báo có thể khiến một tài khoản bị cấm vĩnh viễn. Dùng AI để viết bài đăng, báo lỗi, mô tả pull
            request hay trả lời người khác là bị cấm.
          </p>
        }
      >
        <p>
          Its AI policy is explicit. AI-generated code is allowed, and you are responsible for every line. You must disclose how AI was used;
          undisclosed use can get an account permanently banned. Using AI to write posts, bug reports, pull request descriptions or replies to
          people is prohibited.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x3-facts.md F1, F2)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              CONTRIBUTING.md của llama.cpp (thay đổi lần cuối 2026-11-16); CONTRIBUTING.md của MLX và PyTorch ở các commit được ghim. Mục
              "AI-Assisted Development" của PyTorch chưa được đọc.
            </p>
          }
        >
          <p>
            llama.cpp CONTRIBUTING.md (last changed 2026-11-16); MLX and PyTorch CONTRIBUTING.md at the pinned commits. PyTorch's
            "AI-Assisted Development" section was not read.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsSX39['x3/X3.9/q-ai-policy']} />
      <Translated
        vi={
          <p>
            <b>MLX.</b> Mọi đóng góp đều cần kiểm thử kỹ lưỡng; chạy <code>python3 python/tests/run.py</code> và định dạng bằng{' '}
            <code>pre-commit run --all</code>. Lập trình có hỗ trợ của AI được phép, và đầu ra của nó phải đạt cùng chuẩn như mọi thứ khác.{' '}
            <b>PyTorch.</b> Chạy <code>python test/run_test.py</code>, hoặc một tệp, ví dụ <code>pytest test/test_nn.py -k Loss -v</code>.
            Hướng dẫn của nó có một mục về phát triển có hỗ trợ của AI, mà khóa học này chưa đọc: hãy đọc nó trước khi bạn mở pull request.
          </p>
        }
      >
        <p>
          <b>MLX.</b> Every contribution needs thorough tests; run <code>python3 python/tests/run.py</code> and format with{' '}
          <code>pre-commit run --all</code>. AI-assisted coding is allowed, and its output must meet the same standard as anything else.{' '}
          <b>PyTorch.</b> Run <code>python test/run_test.py</code>, or one file with, for example, <code>pytest test/test_nn.py -k Loss -v</code>.
          Its guide has a section on AI-assisted development, which this course has not read: read it before you open a pull request.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Dù bạn đóng góp cho dự án nào, thói quen vẫn là thói quen trang này đã xây dựng: so sánh với một tham chiếu bằng sai số chuẩn
            hóa, nêu rõ dung sai, và khi sự trùng khớp chính xác thất bại, hãy xem khoảng cách trước khi xem kernel. Lab X3.2 trên{' '}
            <a href={href('x3', 'labs')}>trang lab</a> chạy cả hai phép kiểm tra trên M10.
          </p>
        }
      >
        <p>
          Whichever project you contribute to, the habit is the one this page built: compare with a reference by a normalised error, state the
          tolerance, and when exact agreement fails, look at the margin before you look at the kernel. Lab X3.2 on the{' '}
          <a href={href('x3', 'labs')}>labs page</a> runs both tests on M10.
        </p>
      </Translated>
    </>
  );
}
