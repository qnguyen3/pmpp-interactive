import { Fragment, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';
import {
  BLOCKS,
  MILESTONES,
  MILESTONE_LABEL,
  PRIMITIVES,
  PRIMITIVE_LABEL,
  crossover,
  decodeBytes,
  judge,
  layerFlops,
  type FfnKind,
  type LayerShape,
  type Milestone,
  type Phase,
  type Primitive,
} from '../../sim/llmflops';

export const questionsS201: Record<string, Question> = {
  'ch20/20.1/q-layernorm': {
    kind: 'choice',
    id: 'ch20/20.1/q-layernorm',
    prompt: (
      <Translated
        vi={
          <>
            LayerNorm (Hình 20.1: <code>(x − μ)/σ · γ + β</code> cho hàng của mỗi token) gần nhất với kernel nào bạn đã
            viết?
          </>
        }
      >
        <>
          LayerNorm (Fig. 20.1: <code>(x − μ)/σ · γ + β</code> for each token&rsquo;s row) is closest to which kernel you
          have already written?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Softmax theo hàng của M4: theo từng hàng, các phép rút gọn rồi một lượt theo từng phần tử.">{'M4\'s row softmax: per row, reductions then an elementwise pass.'}</Translated>, correct: true },
      { label: <Translated vi="Một GEMM phân ô, vì γ và β được học giống như một ma trận trọng số.">{'A tiled GEMM, since γ and β are learned like a weight matrix.'}</Translated>, why: <Translated vi="γ và β là các vectơ áp dụng theo từng phần tử. Không có gì được nhân với một ma trận.">{'γ and β are vectors applied elementwise. Nothing is multiplied by a matrix.'}</Translated> },
      { label: <Translated vi="Một phép scan, vì phương sai cần trung bình trước.">{'A scan, since the variance needs the mean first.'}</Translated>, why: <Translated vi="Cần một tổng trước một tổng khác là hai phép rút gọn nối tiếp, không phải một tổng tiền tố đang chạy.">{'Needing one total before another is two reductions in sequence, not a running prefix.'}</Translated> },
      { label: <Translated vi="Một histogram riêng tư hóa, mỗi đặc trưng một bin.">{'A privatised histogram, one bin per feature.'}</Translated>, why: <Translated vi="Không có gì được đếm vào bin; mọi phần tử được đọc và ghi lại tại chỗ.">{'Nothing is counted into bins; every element is read and rewritten in place.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Theo từng hàng: rút gọn ra trung bình, rút gọn ra phương sai (hoặc cả hai cùng lúc), rồi ghi lại từng phần tử. Đó là
            dạng của softmax trong lab 10.3, một khối cho mỗi hàng, với các toán tử khác. Các mô hình mở ngày nay dùng RMSNorm,
            bỏ phần trung bình và chỉ cần một phép rút gọn (facts E8).
          </p>
        }
      >
        <p>
          Per row: reduce to the mean, reduce to the variance (or both at once), then rewrite each element. That is the
          shape of lab 10.3&rsquo;s softmax, one block per row, with different operators. Today&rsquo;s open models use
          RMSNorm, which drops the mean and needs one reduction (facts E8).
        </p>
      </Translated>
    ),
    source: 'ch. 20 p. 481, Fig. 20.1; ch. 10 and lab 10.3',
    vi: { source: 'ch. 20 p. 481, Hình 20.1; ch. 10 và lab 10.3' },
  },
  'ch20/20.1/q-flops-share': {
    kind: 'numeric',
    id: 'ch20/20.1/q-flops-share',
    prompt: (
      <Translated
        vi={
          <>
            Một lớp với các kích thước của sách: bốn phép chiếu d_model × d_model, một mạng feed-forward ReLU hai ma trận với
            d_ff = 4 d_model, không bỏ qua nửa bị che. Ở N = 4096 và d_model = 4096, số FLOP của Q K<sup>T</sup> và P V cộng
            lại, chia cho số FLOP của các lớp tuyến tính, là bao nhiêu?
          </>
        }
      >
        <>
          One layer with the book&rsquo;s shapes: four d_model × d_model projections, a two-matrix ReLU feed-forward with
          d_ff = 4 d_model, no skipping of the masked half. At N = 4096 and d_model = 4096, what are the FLOPs of Q K
          <sup>T</sup> and P V together, divided by the FLOPs of the linear layers?
        </>
      </Translated>
    ),
    answer: 0.1667,
    tol: 0.03,
    explain: (
      <Translated
        vi={
          <p>
            Mỗi token, hai phép nhân tốn 2 × 2 N d_model = 4 N d_model FLOP và các lớp tuyến tính 2 × (4 + 8) d_model² = 24
            d_model². Tỷ lệ là N / (6 d_model) = 1/6. Hai phép nhân chỉ đuổi kịp ở N = 6 d_model, ở đây là 24,576 token.
          </p>
        }
      >
        <p>
          Per token the two products cost 2 × 2 N d_model = 4 N d_model FLOPs and the linear layers 2 × (4 + 8)
          d_model² = 24 d_model². The ratio is N / (6 d_model) = 1/6. The products only catch up at N = 6 d_model, here
          24,576 tokens.
        </p>
      </Translated>
    ),
    source: "our arithmetic from the book's shapes (pp. 480–483); the explorer above",
    vi: { source: 'phép tính của chúng tôi từ các kích thước của sách (pp. 480–483); trình khám phá ở trên' },
  },
  'ch20/20.1/q-new-kernels': {
    kind: 'choice',
    id: 'ch20/20.1/q-new-kernels',
    prompt: (
      <Translated vi={<>Những khối nào của Hình 20.1 cần một kernel mà đến hết chương 15 bạn chưa viết?</>}>
        <>Which blocks of Fig. 20.1 need a kernel you had not written by the end of chapter 15?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Q K^T với mặt nạ nhân quả, softmax theo hàng của nó, và P V.">{'Q K^T with the causal mask, its row softmax, and P V.'}</Translated>, correct: true },
      { label: <Translated vi="Mạng feed-forward, vì hàm kích hoạt nằm giữa hai lớp của nó.">{'The feed-forward, because of the activation between its two layers.'}</Translated>, why: <Translated vi="Hai GEMM và một ReLU theo từng phần tử: M1 đến M3 và M0.">{'Two GEMMs and an elementwise ReLU: M1 to M3 and M0.'}</Translated> },
      { label: <Translated vi="LayerNorm, vì γ và β được học của nó.">{'LayerNorm, because of its learned γ and β.'}</Translated>, why: <Translated vi="Hai phép rút gọn theo hàng và một lượt theo từng phần tử, dạng của M4.">{'Two row reductions and an elementwise pass, the shape of M4.'}</Translated> },
      { label: <Translated vi="Phép tra embedding, vì nó đánh chỉ số theo token.">{'The embedding lookup, because it indexes by token.'}</Translated>, why: <Translated vi="Một phép gather: mỗi token ID chép một hàng, không có phép tính số học.">{'A gather: one row copied per token ID, no arithmetic.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mọi thứ khác là một GEMM hoặc GEMV, một phép rút gọn theo hàng, việc theo từng phần tử hoặc một phép gather. Ngay cả
            ba bước attention cũng chạy được ở dạng không gộp trên các kernel cũ (hai GEMM và một softmax); cái mới là mặt nạ
            nhân quả và, ở mục 20.5, việc gộp chúng để các ma trận N × N không bao giờ tồn tại. Sách cũng nói vậy về phiên
            bản không gộp của nó: softmax là mảnh mới duy nhất (p. 486).
          </p>
        }
      >
        <p>
          Everything else is a GEMM or GEMV, a row reduction, elementwise work or a gather. Even the three attention steps
          can run unfused on old kernels (two GEMMs and a softmax); what is new is the causal mask and, in section 20.5,
          fusing them so the N × N matrices never exist. The book says the same of its unfused version: the softmax is
          the only new piece (p. 486).
        </p>
      </Translated>
    ),
    source: 'ch. 20 Fig. 20.1, p. 479, and p. 486',
    vi: { source: 'ch. 20 Hình 20.1, p. 479, và p. 486' },
  },
  'ch20/20.1/q-rope': {
    kind: 'choice',
    id: 'ch20/20.1/q-rope',
    prompt: (
      <Translated
        vi={
          <>
            Các mô hình họ Llama xoay Q và K theo vị trí bên trong mỗi lớp attention, và lưu K vào cache sau khi xoay. Ở mỗi
            bước, một kernel decode phải làm gì với các key trong cache?
          </>
        }
      >
        <>
          Llama-family models rotate Q and K by position inside each attention layer, and cache K after the rotation. What
          must a decode kernel do to the cached keys at each step?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Không gì cả: chỉ q và k mới được xoay, theo vị trí của chúng.">{'Nothing: only the new q and k get rotated, at their position.'}</Translated>, correct: true },
      { label: <Translated vi="Xoay lại mọi key trong cache, theo khoảng cách tới vị trí mới.">{'Rotate every cached key again, by the distance to the new position.'}</Translated>, why: <Translated vi="Mỗi key đã được xoay một lần, theo vị trí của chính nó, trước khi được lưu. Đó là mục đích của việc lưu vào cache sau khi xoay.">{'Each key was rotated once, by its own position, before it was stored. That is the point of caching after the rotation.'}</Translated> },
      { label: <Translated vi="Xoay V theo cùng cách, để P V khớp với các key.">{'Rotate V the same way, so that P V lines up with the keys.'}</Translated>, why: <Translated vi="Chỉ Q và K được xoay; tích vô hướng của chúng mang thông tin vị trí. V không bị động đến.">{'Only Q and K are rotated; their inner product carries the position. V is untouched.'}</Translated> },
      { label: <Translated vi="Cộng một mã hóa vị trí vào cache trước khi đọc nó.">{'Add a positional encoding to the cache before reading it.'}</Translated>, why: <Translated vi="Đó là cách của transformer gốc, áp dụng một lần lên các embedding, không phải lên một cache.">{'That is the original transformer\'s scheme, applied once to the embeddings, not to a cache.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mã tham chiếu của Meta áp dụng rotary embedding lên Q và K rồi mới ghi K vào cache (facts D3), nên cache chứa các key
            đã xoay và bước decode chỉ xoay query và key của token mới tại vị trí của chính nó. Một cache được ghi theo một quy
            ước và đọc theo quy ước khác là một lỗi thầm lặng.
          </p>
        }
      >
        <p>
          Meta&rsquo;s reference code applies the rotary embedding to Q and K and then writes K into the cache (facts D3),
          so the cache holds rotated keys and the decode step rotates only the new token&rsquo;s query and key at its own
          position. A cache written by one convention and read by another is a silent bug.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch20-facts.md E8, D3', // i18n-ignore: citation
  },
  'ch20/20.1/q-hidden': {
    kind: 'numeric',
    id: 'ch20/20.1/q-hidden',
    prompt: (
      <Translated
        vi={
          <>
            DeepSeek-V2 có 128 head attention rộng 128 và số chiều ẩn 5120. Công thức &ldquo;số chiều ẩn = h_q × d&rdquo; của
            sách (p. 505) cho ra con số nào với nó?
          </>
        }
      >
        <>
          DeepSeek-V2 has 128 attention heads of width 128 and a hidden dimension of 5120. What does the book&rsquo;s
          &ldquo;hidden dimension = h_q × d&rdquo; (p. 505) give for it?
        </>
      </Translated>
    ),
    answer: 16384,
    explain: (
      <Translated
        vi={
          <p>
            128 × 128 = 16,384, hơn ba lần con số thật 5120. Số head × độ rộng head không nhất thiết bằng độ rộng mô hình, nên
            các công thức cache của mục 20.6 cần h và d, không bao giờ cần kích thước ẩn. Đẳng thức của sách đúng với GPT-3 (96
            × 128 = 12,288), Llama 2 7B và Mistral 7B (32 × 128 = 4096).
          </p>
        }
      >
        <p>
          128 × 128 = 16,384, more than three times the real 5120. Heads × head width need not equal the model width, so
          the cache formulas of section 20.6 need h and d, never the hidden size. The book&rsquo;s identity holds for
          GPT-3 (96 × 128 = 12,288), Llama 2 7B and Mistral 7B (32 × 128 = 4096).
        </p>
      </Translated>
    ),
    source: 'notes/today/ch20-facts.md E2, E8', // i18n-ignore: citation
  },
};

// ---------------------------------------------------------------- which kernel is this block?
function Classify() {
  const [prim, setPrim] = useState<Record<string, Primitive | ''>>({});
  const [mile, setMile] = useState<Record<string, Milestone | ''>>({});
  const [checked, setChecked] = useState(false);
  const verdicts = BLOCKS.map((b) => judge(b, prim[b.id] || null, mile[b.id] || null));
  const right = verdicts.filter((v) => v.primitive && v.milestone).length;
  const answered = BLOCKS.filter((b) => prim[b.id] && mile[b.id]).length;
  const reveal = () => {
    setPrim(Object.fromEntries(BLOCKS.map((b) => [b.id, b.primitive[0]])));
    setMile(Object.fromEntries(BLOCKS.map((b) => [b.id, b.milestone[0]])));
    setChecked(true);
  };
  return (
    <Bench
      title={tr("Which kernel is this block?", 'Khối này là kernel nào?')}
      task={
        <Translated vi="Với mỗi khối của Hình 20.1, chọn thứ chạy nó và mốc dự án xây nó. Rồi kiểm tra.">
          For each block of Fig. 20.1, pick what runs it and which milestone builds it. Then check.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Từ trên xuống qua một lớp decoder, cộng với những gì xảy ra trước và sau chồng lớp. Một số khối chấp nhận hai đáp
              án; phần phản hồi nói vì sao.
            </>
          }
        >
          <>
            Top to bottom through one decoder layer, plus what happens before and after the stack. Some blocks accept two
            answers; the feedback says why.
          </>
        </Translated>
      }
    >
      <div style={{ overflowX: 'auto' }}>
        <table className="data" data-table="classify" style={{ margin: 0, background: 'var(--page)' }}>
          <thead>
            <tr>
              <th><Translated vi="Khối">Block</Translated></th>
              <th><Translated vi="Thứ chạy nó">What runs it</Translated></th>
              <th><Translated vi="Mốc dự án">Milestone</Translated></th>
              {checked ? <th><Translated vi="Kết luận">Verdict</Translated></th> : null}
            </tr>
          </thead>
          <tbody>
            {BLOCKS.map((b, i) => {
              const v = verdicts[i];
              const ok = v.primitive && v.milestone;
              return (
                <Fragment key={b.id}>
                <tr data-block={b.id} data-verdict={checked ? (ok ? 'right' : 'wrong') : ''} style={checked ? { borderBottom: 0 } : undefined}>
                  <td style={{ minWidth: '9rem' }}>
                    {b.name}
                    {b.fig ? <span className="muted"> ({b.fig})</span> : null}
                  </td>
                  <td>
                    <select
                      aria-label={tr(`What runs ${b.name}`, `Thứ chạy ${b.name}`)}
                      value={prim[b.id] ?? ''}
                      onChange={(e) => {
                        setPrim({ ...prim, [b.id]: e.target.value as Primitive });
                        setChecked(false);
                      }}
                    >
                      <option value="">{tr('choose', 'chọn')}</option>
                      {PRIMITIVES.map((p) => (
                        <option key={p} value={p}>
                          {PRIMITIVE_LABEL[p]}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td>
                    <select
                      aria-label={tr(`Milestone for ${b.name}`, `Mốc dự án cho ${b.name}`)}
                      value={mile[b.id] ?? ''}
                      onChange={(e) => {
                        setMile({ ...mile, [b.id]: e.target.value as Milestone });
                        setChecked(false);
                      }}
                    >
                      <option value="">{tr('choose', 'chọn')}</option>
                      {MILESTONES.map((m) => (
                        <option key={m} value={m}>
                          {MILESTONE_LABEL[m]}
                        </option>
                      ))}
                    </select>
                  </td>
                  {checked ? (
                    <td>
                      {ok ? <span className="chip ok"><Translated vi="đúng">right</Translated></span> : <span className="chip fail">{!v.primitive && !v.milestone ? tr('both wrong', 'sai cả hai') : !v.primitive ? tr('what runs it', 'thứ chạy nó') : tr('milestone', 'mốc dự án')}</span>}
                    </td>
                  ) : null}
                </tr>
                {checked ? (
                  <tr data-why={b.id}>
                    <td colSpan={4} className="muted" style={{ whiteSpace: 'normal', paddingTop: 0 }}>
                      {b.why}
                    </td>
                  </tr>
                ) : null}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="row" style={{ marginTop: '0.8rem' }}>
        <button className="primary" onClick={() => setChecked(true)} disabled={answered === 0}>
          <Translated vi="Kiểm tra">Check</Translated>
        </button>
        <button className="quiet" onClick={reveal}>
          <Translated vi="Xem đáp án">Show the answers</Translated>
        </button>
        <span className="muted" role="status" data-score={checked ? right : ''}>
          {checked ? tr(`${right} of ${BLOCKS.length} rows right.`, `${right} / ${BLOCKS.length} hàng đúng.`) : tr(`${answered} of ${BLOCKS.length} rows answered.`, `Đã trả lời ${answered} / ${BLOCKS.length} hàng.`)}
        </span>
      </div>
      {checked && right === BLOCKS.length ? (
        <Translated
          vi={
            <p style={{ margin: '0.8rem 0 0' }} data-discovery>
              Chỉ có ba hàng ghi M6. Mọi thứ khác trong lớp là kernel bạn đã có: việc mới của chương là ba bước attention, và lý do
              chúng cần việc mới là kích thước của thứ được truyền giữa chúng.
            </p>
          }
        >
          <p style={{ margin: '0.8rem 0 0' }} data-discovery>
            Only three rows say M6. Everything else in the layer is a kernel you already have: the chapter&rsquo;s new
            work is the three attention steps, and the reason they need new work is the size of what passes between them.
          </p>
        </Translated>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- where the arithmetic goes
const D_MODELS = [512, 1024, 2048, 4096, 8192, 16384];
const PARTS: { key: 'proj' | 'qk' | 'pv' | 'ffn'; label: string; alpha: number }[] = [
  { key: 'proj', get label() { return tr('Q, K, V, O projections', 'các phép chiếu Q, K, V, O'); }, alpha: 0.82 },
  { key: 'ffn', get label() { return tr('feed-forward', 'feed-forward'); }, alpha: 0.5 },
  { key: 'qk', label: 'Q K^T', alpha: 0.24 },
  { key: 'pv', label: 'P V', alpha: 0.1 },
];
const fmtFlops = (x: number) => {
  if (x >= 1e12) return (x / 1e12).toPrecision(3) + ' TFLOP';
  if (x >= 1e9) return (x / 1e9).toPrecision(3) + ' GFLOP';
  if (x >= 1e6) return (x / 1e6).toPrecision(3) + ' MFLOP';
  return x.toPrecision(3) + ' FLOP';
};
const fmtMiB = (x: number) => {
  const m = x / 2 ** 20;
  return (m >= 10 ? Math.round(m).toLocaleString('en-US') : m.toPrecision(2)) + ' MiB';
};
const pct = (x: number) => (100 * x).toFixed(x < 0.01 ? 2 : 1) + '%';
const MAX_LOG = 17; // 131,072 tokens

function Explorer() {
  const [logN, setLogN] = useState(12);
  const [dModel, setDModel] = useState(4096);
  const [ratio, setRatio] = useState(4);
  const [ffn, setFfn] = useState<FfnKind>('relu');
  const [skip, setSkip] = useState(false);
  const [phase, setPhase] = useState<Phase>('prefill');
  const n = Math.round(2 ** logN);
  const shape: LayerShape = { dModel, ffRatio: ratio, ffn, causalSkip: skip };
  const f = layerFlops(n, shape, phase);
  const cross = crossover(shape, phase);
  const bytes = decodeBytes(n, shape, 1, 2);
  const curve = useMemo(
    () =>
      Array.from({ length: MAX_LOG * 8 + 1 }, (_, i) => {
        const x = i / 8;
        return { x, y: layerFlops(Math.round(2 ** x), shape, phase).attnShare };
      }),
    [dModel, ratio, ffn, skip, phase], // eslint-disable-line react-hooks/exhaustive-deps
  );
  // Plot geometry.
  const W = 640;
  const H = 210;
  const L = 44;
  const R = 12;
  const T = 12;
  const Bm = 34;
  const px = (x: number) => L + (x / MAX_LOG) * (W - L - R);
  const py = (y: number) => T + (1 - y) * (H - T - Bm);
  const crossX = Math.log2(Math.max(1, cross));
  return (
    <Bench
      title={tr("Where a layer's arithmetic goes", 'Phép tính số học của một lớp đi về đâu')}
      task={
        <Translated vi="Tìm độ dài ngữ cảnh mà tại đó Q K^T và P V tốn ngang các lớp tuyến tính. Rồi chuyển sang một token decode.">
          Find the context length at which Q K^T and P V cost as much as the linear layers. Then switch to one decode token.
        </Translated>
      }
      foot={
        <Translated vi={<>Phép tính của chúng tôi từ các kích thước của sách, không phải của sách. FLOP chỉ tính các phép nhân ma trận, 2 cho mỗi phép nhân-cộng.</>}>
          <>Our arithmetic from the book&rsquo;s shapes, not the book&rsquo;s. FLOPs count the matrix products only, 2 per multiply-add.</>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field" style={{ minWidth: '14rem' }}>
          <Translated vi="Độ dài ngữ cảnh N: ">Context length N: </Translated><b data-n>{n.toLocaleString('en-US')}</b>
          <input type="range" min={0} max={MAX_LOG} step={0.25} value={logN} onChange={(e) => setLogN(+e.target.value)} aria-label={tr("context length (log scale)", 'độ dài ngữ cảnh (thang log)')} />
        </label>
        <label className="field">{/* i18n-ignore: variable name */}
          d_model
          <select value={dModel} onChange={(e) => setDModel(+e.target.value)} aria-label="d_model">{/* i18n-ignore: variable name */}
            {D_MODELS.map((d) => (
              <option key={d} value={d}>
                {d.toLocaleString('en-US')}
              </option>
            ))}
          </select>
        </label>
        <label className="field" style={{ minWidth: '10rem' }}>{/* i18n-ignore: variable names */}
          d_ff / d_model: <b>{ratio}</b>
          <input type="range" min={1} max={8} step={0.25} value={ratio} onChange={(e) => setRatio(+e.target.value)} aria-label={tr("d_ff over d_model", 'd_ff chia d_model')} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <span className="seg" role="group" aria-label={tr("Feed-forward", 'Feed-forward')}>
          {(['relu', 'swiglu'] as FfnKind[]).map((k) => (
            <button key={k} aria-pressed={ffn === k} onClick={() => setFfn(k)}>
              {k === 'relu' ? tr('ReLU, 2 matrices', 'ReLU, 2 ma trận') : tr('SwiGLU, 3 matrices', 'SwiGLU, 3 ma trận')}
            </button>
          ))}
        </span>
        <span className="seg" role="group" aria-label={tr("Phase", 'Giai đoạn')}>
          {(['prefill', 'decode'] as Phase[]).map((p) => (
            <button key={p} aria-pressed={phase === p} onClick={() => setPhase(p)}>
              {p === 'prefill' ? tr('prefill of N tokens', 'prefill N token') : tr('one decode token', 'một token decode')}
            </button>
          ))}
        </span>
        <label className="check">
          <input type="checkbox" checked={skip} onChange={(e) => setSkip(e.target.checked)} disabled={phase === 'decode'} /> <Translated vi="bỏ qua nửa bị che">skip the masked half</Translated>
        </label>
      </div>

      <div aria-label={tr("Share of FLOPs", 'Tỷ lệ FLOP')} data-shares style={{ display: 'flex', height: '1.6rem', border: '1px solid var(--rule)', borderRadius: 3, overflow: 'hidden' }}>
        {PARTS.map((p) => (
          <div
            key={p.key}
            title={`${p.label}: ${pct(f[p.key] / f.total)}`}
            data-part={p.key}
            data-share={(f[p.key] / f.total).toFixed(4)}
            style={{ width: `${(100 * f[p.key]) / f.total}%`, background: 'var(--ink)', opacity: p.alpha }}
          />
        ))}
      </div>
      <div className="legend" style={{ marginTop: '0.4rem', display: 'flex', flexWrap: 'wrap', gap: '0.3rem 1rem' }}>
        {PARTS.map((p) => (
          <span key={p.key}>
            <i style={{ background: 'var(--ink)', opacity: p.alpha }} />
            {p.label} {pct(f[p.key] / f.total)}
          </span>
        ))}
      </div>
      <p style={{ margin: '0.8rem 0' }} data-readout>
        <Translated
          vi={
            <>
              {phase === 'prefill' ? 'Prefill' : 'Một token decode ở ngữ cảnh'} {n.toLocaleString('en-US')}
              {phase === 'prefill' ? ' token' : ''}: {fmtFlops(f.total)} mỗi lớp. Các phép nhân attention bằng{' '}
              <b data-ratio={f.attnOverLinear.toFixed(4)}>{f.attnOverLinear < 10 ? f.attnOverLinear.toPrecision(3) : Math.round(f.attnOverLinear)}</b> lần các
              lớp tuyến tính; chúng ngang bằng nhau ở N = <b data-cross={Math.round(cross)}>{Math.round(cross).toLocaleString('en-US')}</b>.
            </>
          }
        >
          {phase === 'prefill' ? 'Prefill of' : 'One decode token at context'} {n.toLocaleString('en-US')}
          {phase === 'prefill' ? ' tokens' : ''}: {fmtFlops(f.total)} per layer. The attention products are{' '}
          <b data-ratio={f.attnOverLinear.toFixed(4)}>{f.attnOverLinear < 10 ? f.attnOverLinear.toPrecision(3) : Math.round(f.attnOverLinear)}</b> times the
          linear layers; they draw level at N = <b data-cross={Math.round(cross)}>{Math.round(cross).toLocaleString('en-US')}</b>.
        </Translated>
      </p>
      {phase === 'decode' ? (
        <Translated
          vi={
            <p style={{ margin: '0 0 0.8rem' }} data-bytes>
              Số byte token này đọc mỗi lớp, 16-bit, một chuỗi: trọng số <b data-wbytes={bytes.weights}>{fmtMiB(bytes.weights)}</b>, K cache
              và V cache của nó <b data-cbytes={bytes.cache}>{fmtMiB(bytes.cache)}</b>. Một batch gồm b chuỗi đọc trọng số một lần và
              b cache, nên lượng đọc cache lớn hơn khi b × N vượt quá{' '}
              <b data-bcross={Math.round(bytes.crossBatchN)}>{Math.round(bytes.crossBatchN).toLocaleString('en-US')}</b>.
            </p>
          }
        >
          <p style={{ margin: '0 0 0.8rem' }} data-bytes>
            Bytes this token reads per layer, 16-bit, one sequence: weights <b data-wbytes={bytes.weights}>{fmtMiB(bytes.weights)}</b>, its K
            and V cache <b data-cbytes={bytes.cache}>{fmtMiB(bytes.cache)}</b>. A batch of b sequences reads the weights once and
            b caches, so the cache read is the larger once b × N passes{' '}
            <b data-bcross={Math.round(bytes.crossBatchN)}>{Math.round(bytes.crossBatchN).toLocaleString('en-US')}</b>.
          </p>
        </Translated>
      ) : null}

      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={tr("Attention products as a share of the layer's FLOPs, against context length", 'Tỷ lệ FLOP của lớp nằm ở các phép nhân attention, theo độ dài ngữ cảnh')}>
        {[0, 0.25, 0.5, 0.75, 1].map((y) => (
          <g key={y}>
            <line x1={L} x2={W - R} y1={py(y)} y2={py(y)} stroke="var(--rule)" strokeWidth={y === 0.5 ? 1 : 0.6} strokeDasharray={y === 0.5 ? '4 3' : undefined} />
            <text x={L - 6} y={py(y) + 4} textAnchor="end" fontSize="11">
              {y * 100}%
            </text>
          </g>
        ))}
        {[0, 4, 8, 12, 16].map((x) => (
          <text key={x} x={px(x)} y={H - Bm + 16} textAnchor="middle" fontSize="11">
            {(2 ** x).toLocaleString('en-US')}
          </text>
        ))}
        <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle" fontSize="11">
          <Translated vi="độ dài ngữ cảnh N (thang log)">context length N (log scale)</Translated>
        </text>
        <text x={L + 8} y={T + 10} fontSize="11" fill="var(--ink-2)">
          <Translated vi="tỷ lệ FLOP của lớp nằm ở Q K^T và P V">share of the layer&rsquo;s FLOPs in Q K^T and P V</Translated>
        </text>
        <polyline fill="none" stroke="var(--ink)" strokeWidth={2} points={curve.map((p) => `${px(p.x)},${py(p.y)}`).join(' ')} />
        {crossX <= MAX_LOG ? (
          <g data-crossmark>
            <line x1={px(crossX)} x2={px(crossX)} y1={T} y2={H - Bm} stroke="var(--ink-2)" strokeDasharray="3 3" />
            <text x={px(crossX) + (crossX > MAX_LOG - 4 ? -4 : 4)} y={T + 10} fontSize="11" textAnchor={crossX > MAX_LOG - 4 ? 'end' : 'start'}>
              <Translated vi="phép nhân = lớp tuyến tính">products = linear layers</Translated>
            </text>
          </g>
        ) : null}
        <circle cx={px(logN)} cy={py(f.attnShare)} r={5} fill="var(--page)" stroke="var(--ink)" strokeWidth={2} />
      </svg>
    </Bench>
  );
}

export function S201() {
  return (
    <>
      <Translated
        vi={
          <p>
            Bạn biết kiến trúc này từ phía PyTorch, nên mục này chỉ làm một việc: gọi tên kernel đứng sau mỗi khối. Một LLM chỉ
            có decoder và tự hồi quy (auto-regressive). Nó chạy cả chồng lớp một lần cho mỗi token sinh ra, đưa mỗi token mới
            trở lại làm đầu vào, cho đến khi nó phát ra một token kết thúc chuỗi. (Transformer gốc còn có các lớp encoder trộn
            đầu vào một lần; các tác vụ phân biệt như phân loại không cần decoder. Sinh văn bản là việc LLM làm, và là thứ
            chương này tối ưu.)
          </p>
        }
      >
        <p>
          You know this architecture from the PyTorch side, so this section only does one job: name the kernel behind each
          block. An LLM is decoder-only and auto-regressive. It runs the whole stack once per generated token, feeding
          each new token back in, until it emits an end-of-sequence token. (The original transformer also had encoder
          layers that mix the input once; discriminative tasks such as classification need no decoder. Generation is what
          LLMs do, and it is what this chapter optimises.)
        </p>
      </Translated>
      <Source section="20.1" pages="478–479" />

      <h2><Translated vi="Hình 20.1 dưới dạng một danh sách kernel">Fig. 20.1 as a list of kernels</Translated></h2>
      <Translated
        vi={
          <p>
            Đọc hình từ trên xuống. Tokenizer biến văn bản thành các ID số nguyên qua một bảng dựng trong lúc huấn luyện; lớp
            embedding tra mỗi ID trong một bảng đã học; transformer gốc cộng một mã hóa vị trí vào mỗi embedding. Các hàng, mỗi
            token cho đến giờ một hàng, tạo thành ma trận X đi vào lớp đầu tiên. Mỗi lớp gồm multi-head attention, một Add &amp;
            Norm, một mạng feed-forward và một Add &amp; Norm thứ hai. Sau lớp cuối, lớp un-embedding biến hàng cuối cùng thành
            một phân phối trên bộ từ vựng, và một cơ chế nào đó chọn token tiếp theo từ đó; sách dành cho lựa chọn ấy một câu.
          </p>
        }
      >
        <p>
          Read the figure from the top. The tokenizer turns text into integer IDs through a table built in training; the
          embedding layer looks each ID up in a learned table; the original transformer adds a positional encoding to each
          embedding. The rows, one per token so far, form the matrix X that enters the first layer. Each layer is
          multi-head attention, an Add &amp; Norm, a feed-forward network and a second Add &amp; Norm. After the last
          layer, the un-embedding layer turns the final row into a distribution over the vocabulary, and something
          chooses the next token from it; the book gives that choice one sentence.
        </p>
      </Translated>
      <Source section="20.1" pages="479–481" figs={tr("Fig. 20.1", 'Hình 20.1')} />
      <Translated
        vi={
          <p>
            Chú thích số 5 phóng to các head attention: một phép chiếu tuyến tính ra Q, K và V, rồi Q K<sup>T</sup>, một
            softmax, và một phép nhân với V; đầu ra của các head được nối lại và chiếu thêm một lần. Hãy phân loại mọi khối trước
            khi đọc tiếp.
          </p>
        }
      >
        <p>
          Call-out 5 zooms into the attention heads: a linear projection to Q, K and V, then Q K<sup>T</sup>, a softmax,
          and a product with V; the heads&rsquo; outputs are concatenated and projected once more. Sort every block
          before reading on.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Chú thích của Hình 20.1 kết thúc bằng &ldquo;S x V&rdquo;. Thực ra là P × V: đầu ra softmax nhân với V, như Hình 20.2
              và Eq. 20.1 ghi.
            </p>
          }
        >
          <p>
            Fig. 20.1&rsquo;s call-out ends with &ldquo;S x V&rdquo;. It is P × V: the softmax output times V, as Fig. 20.2
            and Eq. 20.1 have it.
          </p>
        </Translated>
      </Aside>
      <Classify />
      <Ask q={questionsS201['ch20/20.1/q-layernorm']} />
      <Ask q={questionsS201['ch20/20.1/q-new-kernels']} />

      <h2><Translated vi="Các chiều của mô hình">The model&rsquo;s dimensions</Translated></h2>
      <Translated
        vi={
          <p>
            Sách nêu ba chiều: độ sâu (số lớp), độ rộng của mỗi vectơ embedding, và số head attention mỗi lớp. Rồi sách nói số
            tham số của một mô hình là tích của ba chiều này.
          </p>
        }
      >
        <p>
          The book names three: depth (the number of layers), the width of each embedding vector, and the number of
          attention heads per layer. It then says a model&rsquo;s parameter count is the product of the three.
        </p>
      </Translated>
      <Source section="20.1" pages="481" />
      <Translated
        vi={
          <p>
            Hãy hiểu đó là một cách nói. Với các kích thước của sách, một lớp chứa 4 d_model² trọng số attention và 8 d_model²
            trọng số feed-forward; 32 lớp rộng 4096 cho ra 6.4 tỷ, trong khi 32 × 4096 × 32 chỉ khoảng bốn triệu.
          </p>
        }
      >
        <p>
          Take that as a figure of speech. With the book&rsquo;s shapes a layer holds 4 d_model² attention weights and 8
          d_model² feed-forward weights; 32 layers of width 4096 come to 6.4 billion, while 32 × 4096 × 32 is about four
          million.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>Số tham số là phép tính của chúng tôi từ các kích thước của sách.</p>
          }
        >
          <p>The parameter count is our arithmetic from the book&rsquo;s shapes.</p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Thêm một ký hiệu cần chốt ngay bây giờ. Sách viết d cho cả độ rộng embedding lẫn độ rộng của một head, điều này vô
            hại khi sách chỉ vẽ một head có mọi ma trận đều rộng d. Trong các lab của khóa học này, d là <b>độ rộng head</b> (128
            ở GPT-3, Llama 2 7B và Mistral 7B), còn d_model là độ rộng mô hình, ở hầu hết mô hình bằng số head nhân d.
          </p>
        }
      >
        <p>
          One more piece of notation to fix now. The book writes d both for the embedding width and for a head&rsquo;s
          width, which is harmless while it draws one head whose matrices are all d wide. In this course&rsquo;s labs d is
          the <b>head width</b> (128 in GPT-3, Llama 2 7B and Mistral 7B), and d_model is the model width, the number of heads
          times d in most models.
        </p>
      </Translated>
      <Ask q={questionsS201['ch20/20.1/q-hidden']} label={tr("Check the identity", 'Kiểm tra đẳng thức')} />

      <h2><Translated vi="Phép tính số học đi về đâu">Where the arithmetic goes</Translated></h2>
      <Translated
        vi={
          <p>
            Sách gọi các head attention là phần tốn kém nhất của lớp và dành cả chương cho chúng. Về số học, ở những độ dài ngữ
            cảnh bạn thường chạy, chúng không phải vậy: các lớp tuyến tính chiếm phần lớn. Về byte thì còn tùy cách việc được
            chạy, và những lý do đó chính là nội dung phần còn lại của chương. Hãy tự tìm điểm giao nhau.
          </p>
        }
      >
        <p>
          The book calls the attention heads the most expensive part of the layer and spends the chapter on them. In
          arithmetic, at the context lengths you usually run, they are not: the linear layers dominate. In bytes it
          depends on how the work is run, and the reasons are what the rest of the chapter is about. Find the crossover
          yourself.
        </p>
      </Translated>
      <Source section="20.1" pages="481–482" />
      <Translated
        vi={
          <p>
            Trình khám phá đếm số phép nhân-cộng của các phép nhân ma trận trong một lớp, cho cả một prompt (prefill) hoặc cho một
            token mới (decode), và vẽ tỷ lệ của các phép nhân attention khi ngữ cảnh dài ra.
          </p>
        }
      >
        <p>
          The explorer counts the multiply-adds of one layer&rsquo;s matrix products, for a whole prompt (prefill) or for
          one new token (decode), and plots the attention products&rsquo; share as the context grows.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Chỉ các phép nhân ma trận. Softmax, chuẩn hóa, bias, hàm kích hoạt, embedding và un-embedding ra bộ từ vựng đều bị bỏ
              qua; các phép chiếu K và V nhỏ hơn của grouped-query attention cũng vậy.
            </p>
          }
        >
          <p>
            Matrix products only. Softmax, norms, biases, activations, the embedding and the un-embedding to the
            vocabulary are left out; so are grouped-query attention&rsquo;s smaller K and V projections.
          </p>
        </Translated>
      </Simplification>
      <Explorer />
      <Translated
        vi={
          <p>
            Mỗi token, bốn phép chiếu và mạng feed-forward ReLU tốn 24 d_model² FLOP, còn hai phép nhân attention 4 N d_model.
            Tỷ lệ của chúng là N / (6 d_model): ở d_model = 4096, các phép nhân chỉ ngang bằng khi vượt quá 24,000 token, và
            bỏ qua nửa bị che đẩy con số đó lên khoảng 49,000. Decode có cùng tỷ lệ, vì một token gặp N key. Vậy ở độ dài thông
            thường, vấn đề của attention không phải là FLOP. Ở chỗ nó là vấn đề, đó là{' '}
            <b>byte</b>, ở hai nơi. Trong prefill không gộp, các ma trận N × N S và P mà pipeline ghi rồi đọc (mục 20.3)
            chiếm phần lớn lưu lượng. Trong decode, một chuỗi đọc 2 N d_model phần tử cache mỗi lớp so với 12 d_model² trọng số:
            vẫn là N / (6 d_model), nên với một chuỗi ở độ dài thông thường, trọng số vẫn chiếm phần lớn. Nhưng một batch dùng
            chung trọng số chứ không dùng chung cache, nên lượng đọc cache thắng khi b × N vượt quá 6 d_model, điều mà batching
            (mục 20.6) đạt tới rất nhanh. Chuyển trình khám phá sang một token decode để thấy cả hai.
          </p>
        }
      >
        <p>
          Per token, the four projections and the ReLU feed-forward cost 24 d_model² FLOPs, and the two attention
          products 4 N d_model. Their ratio is N / (6 d_model): at d_model = 4096 the products draw level only past 24,000
          tokens, and skipping the masked half pushes that to about 49,000. Decode has the same ratio, because one token
          meets N keys. So attention&rsquo;s problem at ordinary lengths is not FLOPs. Where it is a problem, it is{' '}
          <b>bytes</b>, in two places. In unfused prefill, the N × N matrices S and P that the pipeline writes and reads
          (section 20.3) dominate the traffic. In decode, one sequence reads 2 N d_model cache elements per layer against
          12 d_model² weights: the same N / (6 d_model), so for one sequence at ordinary lengths the weights still
          dominate. But a batch shares the weights and not the caches, so the cache read wins once b × N passes 6
          d_model, which batching (section 20.6) reaches quickly. Switch the explorer to one decode token to see both.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách không đưa ra số FLOP hay số byte nào. Đây là của chúng tôi, từ các kích thước của sách: bốn phép chiếu d_model ×
              d_model, d_ff = 4 d_model, 2 FLOP cho mỗi phép nhân-cộng, mỗi trọng số và phần tử cache được đọc một lần.
            </p>
          }
        >
          <p>
            The book gives no FLOP or byte counts. These are ours, from its shapes: four d_model × d_model projections,
            d_ff = 4 d_model, 2 FLOPs per multiply-add, each weight and cache element read once.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS201['ch20/20.1/q-flops-share']} label={tr("Work it out", 'Tự tính')} />

      <h2><Translated vi="Các mô hình mở ngày nay đã thay đổi gì">What today&rsquo;s open models changed</Translated></h2>
      <Translated
        vi={
          <p>
            Hình 20.1 là transformer gốc. Các mô hình bạn sẽ nạp khác nó ở ba điểm chạm tới tận kernel. Chuẩn hóa đứng trước mỗi
            lớp con và là RMSNorm, một phép rút gọn thay vì hai. Mạng feed-forward là SwiGLU, ba ma trận thay vì hai (chuyển
            trình khám phá sang nó). Và vị trí là một phép xoay áp lên Q và K bên trong mỗi lớp attention, không phải một vectơ
            cộng một lần ở đáy; vì vậy KV cache lưu các key đã được xoay.
          </p>
        }
      >
        <p>
          Fig. 20.1 is the original transformer. The models you will load differ in three ways that reach into the
          kernels. Normalisation comes before each sub-layer and is RMSNorm, one reduction instead of two. The
          feed-forward is SwiGLU, three matrices instead of two (switch the explorer to it). And position is a rotation
          applied to Q and K inside every attention layer, not a vector added once at the bottom; the KV cache therefore
          stores keys that are already rotated.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md E8, D3, E2)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Llama 2: &ldquo;pre-normalization using RMSNorm&rdquo;, &ldquo;SwiGLU&rdquo;, &ldquo;rotary positional
              embeddings&rdquo;; PaLM cũng liệt kê rotary embedding. Số head × độ rộng head bằng độ rộng mô hình với GPT-3, Llama 2
              7B và Mistral 7B, không đúng với DeepSeek-V2.
            </p>
          }
        >
          <p>
            Llama 2: &ldquo;pre-normalization using RMSNorm&rdquo;, &ldquo;SwiGLU&rdquo;, &ldquo;rotary positional
            embeddings&rdquo;; PaLM also lists rotary embeddings. Heads × head width equals the model width for GPT-3, Llama 2 7B
            and Mistral 7B, not for DeepSeek-V2.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS201['ch20/20.1/q-rope']} />
      <Translated
        vi={
          <p>
            Thêm hai nhận xét từ mục này. Huấn luyện bị chi phối bởi các phép nhân ma trận lớn trong các head attention và các
            lớp feed-forward, vốn là địa hạt của chương 15. Và retrieval-augmented generation dán các tài liệu được truy xuất vào
            prompt, khiến N lớn hơn nhiều: phần còn lại của chương nói về cái giá của một N lớn hơn. Tiếp theo, mục{' '}
            <a href={href('ch20', '20.2')}>20.2</a> mở một head ra.
          </p>
        }
      >
        <p>
          Two more remarks from the section. Training is dominated by large matrix products in the attention heads and the
          feed-forward layers, which is chapter 15&rsquo;s territory. And retrieval-augmented generation pastes retrieved
          documents into the prompt, which makes N much larger: the rest of the chapter is about what a larger N costs.
          Next, section <a href={href('ch20', '20.2')}>20.2</a> opens one head.
        </p>
      </Translated>
      <Source section="20.1" pages="481–482" />
    </>
  );
}
