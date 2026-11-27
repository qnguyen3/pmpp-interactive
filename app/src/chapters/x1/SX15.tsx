import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, realBackends, useReports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { CROSSING, MLX_TIES, VOCAB_PRESETS, rowView, sortPlan, sortShare } from '../../sim/nucleus';
import { topP } from '../../sim/scan';

// Section X1.5: top-p over a sorted row. The explorer's numbers come from src/sim/nucleus.ts
// (rowView, sortPlan), the worked rows from src/sim/scan.ts's topP; both pinned in their tests.

const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const n0 = (v: number) => v.toLocaleString('en-US');
/** Lab X1.2's directory, by the prefix pmpp.py matches. */
const LAB_X12 = 'x1/lab2';

// ---------------------------------------------------------------- the crossing row
function CrossingBars() {
  const r = topP(CROSSING.probs, CROSSING.p, 'exclusive');
  const W = 560;
  const X0 = 14;
  const sx = (v: number) => X0 + v * (W - 2 * X0);
  const crossing = r.before.findIndex((b, i) => b < CROSSING.p && r.upto[i] >= CROSSING.p);
  return (
    <svg viewBox={`0 0 ${W} 92`} role="img" aria-label={tr("Each token's piece of the unit interval, and p", 'Phần của mỗi token trên đoạn đơn vị, và p')} data-part="crossing"
      style={{ width: '100%', maxWidth: W, display: 'block', margin: '0.4rem 0 0.6rem' }}>
      {CROSSING.probs.map((v, i) => (
        <g key={i} data-token={i} data-keep={r.keep[i] ? '1' : '0'} data-before={r.before[i]}>
          <rect x={sx(r.before[i])} y={28} width={sx(r.before[i] + v) - sx(r.before[i])} height={24}
            fill={r.keep[i] ? 'var(--ink-3)' : 'var(--cell)'} fillOpacity={r.keep[i] ? 0.35 : 1}
            stroke={i === crossing ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth={i === crossing ? 2.2 : 1} />
          <text x={(sx(r.before[i]) + sx(r.before[i] + v)) / 2} y={44} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink)' }}>{i}</text>
          <text x={sx(r.before[i])} y={68} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{r.before[i]}</text>
        </g>
      ))}
      <line x1={sx(CROSSING.p)} y1={16} x2={sx(CROSSING.p)} y2={58} stroke="var(--ink)" strokeWidth={2} />
      <text x={sx(CROSSING.p)} y={11} fontSize="11.5" textAnchor="middle" fontWeight="650" style={{ fill: 'var(--ink)' }}>p = {CROSSING.p}</text>
      <text x={X0} y={86} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{tr('numbers under the bars: the mass ranked before each token', 'số dưới các thanh: khối xác suất xếp hạng trước mỗi token')}</text>
    </svg>
  );
}

// ---------------------------------------------------------------- sort what you need
const KS = [0, 10, 40, 100, 1000];
const fmtShare = (x: number) => (x >= 0.1 ? `${fmt(x * 100, 3)}%` : x >= 0.001 ? `${fmt(x * 100, 2)}%` : `${fmt(x * 100, 1)}%`);

function SortExplorer() {
  const [V, setV] = useState(151936);
  const [s, setS] = useState(1.5);
  const [seed, setSeed] = useState(1);
  const [p, setP] = useState(0.95);
  const [k, setK] = useState(0);
  const row = useMemo(() => rowView(V, s, seed), [V, s, seed]);
  const plan = sortPlan(row, p, k, 0.05);
  const strategies = [
    { id: 'full', label: tr('sort the whole row', 'sắp xếp cả hàng'), n: plan.sizes.full, note: tr('what lab 14.3 does with the row as it is', 'điều lab 14.3 làm với hàng như nó vốn có') },
    { id: 'topk', label: tr('top-k first, then sort k', 'top-k trước, rồi sắp xếp k'), n: plan.sizes.topK, note: plan.sizes.topK == null ? tr('top-k is off', 'top-k đang tắt') : tr('lab X1.1, then lab 14.3 on k values', 'lab X1.1, rồi lab 14.3 trên k giá trị') },
    {
      id: 'adaptive', label: tr("llama.cpp's host route", 'đường host của llama.cpp'), n: plan.sizes.adaptive,
      note: V <= 1024 ? tr('1,024 candidates or fewer: a full sort', '1,024 ứng viên trở xuống: sắp xếp toàn bộ') : plan.widened ? tr('the top 256 hold less than p: widened to the row', '256 token đầu giữ ít hơn p: mở rộng ra cả hàng') : tr('the top 256 reach p: a partial sort of 256', '256 token đầu đạt p: sắp xếp một phần 256'),
    },
    { id: 'minp', label: tr('min-p 0.05 first, then sort the survivors', 'min-p 0.05 trước, rồi sắp xếp các token còn lại'), n: plan.sizes.minPFirst, note: tr("when the chain has min-p, as llama.cpp's default does: section X1.6", 'khi chuỗi có min-p, như mặc định của llama.cpp: mục X1.6') },
  ];
  // The cumulative mass against rank on a log axis, sampled at about 160 ranks.
  const PW = 560;
  const PH = 170;
  const L = 38;
  const B = 24;
  const lx = (rank: number) => L + (Math.log10(rank) / Math.log10(V)) * (PW - L - 10);
  const ly = (m: number) => PH - B - m * (PH - B - 10);
  const pts: [number, number][] = [];
  for (let j = 0; j <= 160; j++) {
    const rank = Math.max(1, Math.round(Math.pow(V, j / 160)));
    pts.push([lx(rank), ly(row.before[rank - 1] + row.probs[rank - 1])]);
  }
  const kept = plan.keptFull;
  const { reports } = useReports();
  const measured = realBackends(reports).flatMap((b) =>
    sortShare(benchRows(reports[b], labDir(reports[b]))).map((m) => ({ ...m, backend: b, device: String(reports[b].device.name ?? b) })),
  );
  return (
    <Bench
      title={tr('Sort what you need', 'Chỉ sắp xếp những gì bạn cần')}
      task={tr("Start peaked, then flatten the row until llama.cpp's route widens; then turn top-k on.", 'Bắt đầu với hàng nhọn, rồi làm phẳng hàng cho đến khi đường của llama.cpp mở rộng; rồi bật top-k.')}
      foot={
        <Translated
          vi={
            <>
              Đường cong là khối xác suất của các token có khả năng cao nhất, tính đến và gồm cả mỗi hạng (thang log). Top-p giữ
              mọi token có khối xác suất trước nó dưới p: chỗ đường cong cắt p, cộng thêm token cắt qua.
            </>
          }
        >
        <>
          The curve is the mass of the most likely tokens, up to and including each rank (log scale). Top-p keeps every token
          whose mass before it is below p: the curve's crossing of p, plus the token that crosses.
        </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Bộ từ vựng">Vocabulary</Translated>
          <span className="seg" role="group" aria-label={tr('Vocabulary', 'Bộ từ vựng')}>
            {VOCAB_PRESETS.map((v) => (
              <button key={v} aria-pressed={V === v} onClick={() => setV(v)}>{n0(v)}</button>
            ))}
          </span>
        </span>
        <label className="field" style={{ flex: '1 1 12rem' }}>
          <Translated vi={<>hình dạng s = {s.toFixed(2)} ({s < 0.8 ? 'phẳng' : s < 1.4 ? 'ở giữa' : 'nhọn'})</>}>
            shape s = {s.toFixed(2)} ({s < 0.8 ? 'flat' : s < 1.4 ? 'in between' : 'peaked'})
          </Translated>
          <input type="range" min={0} max={3} step={0.05} value={s} aria-label={tr('shape', 'hình dạng')} onChange={(e) => setS(Number(e.target.value))} style={{ width: '100%' }} />
        </label>
        <button onClick={() => setSeed(seed + 1)}>{tr('another row', 'hàng khác')}</button>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ flex: '1 1 12rem' }}>
          p = {p.toFixed(2)}
          <input type="range" min={1} max={100} step={1} value={Math.round(p * 100)} aria-label="p" onChange={(e) => setP(Number(e.target.value) / 100)} style={{ width: '100%' }} />
        </label>
        <span style={FIELD}>
          <Translated vi="Top-k trước">Top-k first</Translated>
          <span className="seg" role="group" aria-label={tr('Top-k', 'Top-k')}>
            {KS.map((x) => (
              <button key={x} aria-pressed={k === x} onClick={() => setK(x)}>{x === 0 ? tr('off', 'tắt') : x}</button>
            ))}
          </span>
        </span>
      </div>
      <div className="cols plot">
        <svg viewBox={`0 0 ${PW} ${PH}`} role="img" aria-label={tr('Cumulative mass against rank', 'Khối xác suất tích lũy theo hạng')} data-part="curve"
          style={{ width: '100%', maxWidth: PW, display: 'block', background: 'var(--page)', border: '1px solid var(--rule)', borderRadius: 4 }}>
          <line x1={L} y1={ly(0)} x2={PW - 10} y2={ly(0)} stroke="var(--rule)" />
          <line x1={L} y1={ly(1)} x2={PW - 10} y2={ly(1)} stroke="var(--rule-soft)" />
          <text x={L - 4} y={ly(1) + 4} fontSize="10" textAnchor="end">1</text>
          <text x={L - 4} y={ly(0) + 4} fontSize="10" textAnchor="end">0</text>
          <line x1={L} y1={ly(p)} x2={PW - 10} y2={ly(p)} stroke="var(--ink)" strokeDasharray="4 3" />
          <text x={(L + PW) / 2} y={ly(p) + 13} fontSize="10" textAnchor="middle">p = {p.toFixed(2)}</text>
          <polyline points={pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ')} fill="none" stroke="var(--global)" strokeWidth={2} />
          <line x1={lx(Math.min(kept, V))} y1={ly(0)} x2={lx(Math.min(kept, V))} y2={ly(1)} stroke="var(--ink)" />
          <text x={lx(Math.min(kept, V)) + (lx(Math.min(kept, V)) > PW - 140 ? -4 : 4)} y={ly(0) - 6} fontSize="10.5" fontWeight="650"
            textAnchor={lx(Math.min(kept, V)) > PW - 140 ? 'end' : 'start'}><Translated vi={<>top-p giữ {n0(kept)}</>}>top-p keeps {n0(kept)}</Translated></text>
          {V > 256 ? <line x1={lx(256)} y1={ly(0)} x2={lx(256)} y2={ly(0) + 6} stroke="var(--ink-2)" /> : null}
          {V > 256 ? <text x={lx(256)} y={PH - 6} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>256</text> : null}
          <text x={L} y={PH - 6} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>{tr('rank 1', 'hạng 1')}</text>
          <text x={PW - 10} y={PH - 6} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{n0(V)}</text>
        </svg>
        <div>
          <p style={{ marginTop: 0 }} data-readout="kept" data-kept={kept} data-kept-k={plan.keptAfterTopK ?? ''}>
            <Translated vi={<>Top-p tại p = {p.toFixed(2)} giữ <b>{n0(kept)}</b> trên {n0(V)} token ({fmtShare(kept / V)} của hàng).</>}>
              Top-p at p = {p.toFixed(2)} keeps <b>{n0(kept)}</b> of {n0(V)} tokens ({fmtShare(kept / V)} of the row).
            </Translated>
            {plan.keptAfterTopK != null ? (
              <Translated vi={<> Sau top-k = {k}, chuẩn hóa lại trên {k} token đó, nó giữ <b>{n0(plan.keptAfterTopK)}</b>.</>}>
                <> After top-k = {k}, renormalised over those {k}, it keeps <b>{n0(plan.keptAfterTopK)}</b>.</>
              </Translated>
            ) : null}
          </p>
          <p className="muted" style={{ marginBottom: 0 }}>
            <Translated vi={<>Token có khả năng cao nhất giữ {fmtShare(row.probs[0])} khối xác suất.</>}>
              The most likely token holds {fmtShare(row.probs[0])} of the mass.
            </Translated>
          </p>
        </div>
      </div>
      <div style={{ overflowX: 'auto', marginTop: '0.8rem' }}>
        <table className="data" style={{ margin: 0, background: 'var(--page)', minWidth: '34rem' }} data-part="strategies">
          <thead>
            <tr><th style={{ paddingLeft: '0.6rem' }}><Translated vi="Chiến lược">Strategy</Translated></th><th className="num"><Translated vi="Số giá trị được sắp xếp">Values sorted</Translated></th><th style={{ width: '30%' }}><Translated vi="thang log, tối đa bằng cả hàng">log scale, up to the row</Translated></th><th /></tr>
          </thead>
          <tbody>
            {strategies.map((x) => (
              <tr key={x.id} data-strategy={x.id} data-n={x.n ?? ''}>
                <td style={{ paddingLeft: '0.6rem' }}>{x.label}</td>
                <td className="num">{x.n == null ? tr('off', 'tắt') : n0(x.n)}</td>
                <td>
                  {x.n != null ? (
                    <span style={{ display: 'inline-block', height: 10, background: 'var(--global)', width: `${Math.max(1, (Math.log10(Math.max(1, x.n)) / Math.log10(V)) * 100)}%` }} />
                  ) : null}
                </td>
                <td className="muted">{x.id === 'minp' ? <a href={href('x1', 'X1.6')}>{x.note}</a> : x.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {measured.length ? (
        <div className="panel" style={{ marginTop: '0.8rem' }} data-part="measured">
          {measured.map((m, i) => (
            <p key={i} style={{ margin: 0 }}>
              <Translated vi={<>Lab X1.2 trên {m.device} ({m.backend}), {m.rows} × {n0(m.cols)}, {m.k ? `top-k ${m.k}` : 'top-k tắt'}: phép sắp xếp chiếm{' '}<b>{fmtShare(m.share)}</b> của chuỗi ghép ({fmt(m.sort * 1e3)} trên {fmt(m.chain * 1e3)} ms).</>}>
                Lab X1.2 on {m.device} ({m.backend}), {m.rows} × {n0(m.cols)}, {m.k ? `top-k ${m.k}` : 'top-k off'}: the sort was{' '}
                <b>{fmtShare(m.share)}</b> of the composed chain ({fmt(m.sort * 1e3)} of {fmt(m.chain * 1e3)} ms).
              </Translated>
            </p>
          ))}
        </div>
      ) : (
        <Translated
          vi={
            <p className="muted" style={{ margin: '0.6rem 0 0' }} data-part="measured-none">
              Khi lab X1.2 đã chạy ở đây (<code>python3 labs/pmpp.py run x1/lab2 --solution --bench</code>), tỉ lệ thời gian
              sắp xếp đo được của nó hiện bên cạnh mô hình.
            </p>
          }
        >
        <p className="muted" style={{ margin: '0.6rem 0 0' }} data-part="measured-none">
          When lab X1.2 has run here (<code>python3 labs/pmpp.py run x1/lab2 --solution --bench</code>), its measured sort
          share appears beside the model.
        </p>
        </Translated>
      )}
    </Bench>
  );
}

/** The lab directory under x1/ whose name starts lab2_, as this machine's reports name it. */
function labDir(run: { results: { lab: string }[] } | undefined): string {
  return run?.results.find((r) => r.lab.startsWith(LAB_X12 + '_'))?.lab ?? LAB_X12;
}

function MeasuredShare({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const found = realBackends(reports).flatMap((b) =>
    sortShare(benchRows(reports[b], labDir(reports[b]))).map((m) => ({ ...m, backend: b, device: String(reports[b].device.name ?? b) })),
  );
  if (!found.length)
    return (
      <Translated
        vi={
          <p>
          Chưa có phép đo nào trên máy này. Chạy <code>python3 labs/pmpp.py run x1/lab2 --solution --bench</code> rồi quay
          lại: ô này so sánh dự đoán của bạn với số liệu từ thiết bị của chính bạn, với top-k tắt và với top-k = 40.
        </p>
        }
      >
      <p>
        No measurement on this machine yet. Run <code>python3 labs/pmpp.py run x1/lab2 --solution --bench</code> and come
        back: this box compares your prediction with your own device's numbers, with top-k off and with top-k = 40.
      </p>
      </Translated>
    );
  return (
    <>
      {found.map((m, i) => (
        <p key={i} data-reveal={m.backend}>
          <Translated vi={<>{m.device} ({m.backend}), {m.k ? `top-k = ${m.k}` : 'top-k tắt'}: phép sắp xếp chiếm <b>{fmtShare(m.share)}</b> của chuỗi{m.k ? '.' : `; bạn đã dự đoán ${predicted}%.`}</>}>
            {m.device} ({m.backend}), {m.k ? `top-k = ${m.k}` : 'top-k off'}: the sort took <b>{fmtShare(m.share)}</b> of the
            chain{m.k ? '.' : `; you predicted ${predicted}%.`}
          </Translated>
        </p>
      ))}
      <Translated vi={<p className="muted">Số đo từ máy của bạn, không phải giá trị kỳ vọng.</p>}><p className="muted">Measurements from your machine, not expected values.</p></Translated>
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSX15: Record<string, Question> = {
  'x1/X1.5/q-mlx-ties': {
    kind: 'numeric',
    id: 'x1/X1.5/q-mlx-ties',
    prompt: (
      <Translated
        vi={
          <>
            Các xác suất đã sắp xếp [0.5, 0.125, 0.125, 0.125, 0.125] và p = 0.6. M5 giữ 2 token. <code>apply_top_p</code>{' '}
            của mlx-lm giữ bao nhiêu?
          </>
        }
      >
      <>
        Sorted probabilities [0.5, 0.125, 0.125, 0.125, 0.125] and p = 0.6. M5 keeps 2 tokens. How many does mlx-lm's{' '}
        <code>apply_top_p</code> keep?
      </>
      </Translated>
    ),
    answer: topP(MLX_TIES.probs, MLX_TIES.p, 'mlx').kept,
    unit: 'tokens',
    explain: (
      <Translated
        vi={
          <p>
          Khối xác suất trước token 1 là 0.5, dưới 0.6, nên token biên là token 1, với xác suất 0.125. Sau đó mlx-lm giữ mọi
          token có khả năng ít nhất bằng token biên: cả bốn giá trị 0.125, và 0.5. transformers giữ 2, và nó giữ token nào
          trong bốn token hòa nhau tùy vào phép sắp xếp không ổn định của nó.
        </p>
        }
      >
      <p>
        The mass before token 1 is 0.5, below 0.6, so the boundary token is token 1, with probability 0.125. mlx-lm then
        keeps every token at least as likely as the boundary token: all four 0.125s, and the 0.5. transformers keeps 2,
        and which of the four ties it keeps depends on its unstable sort.
      </p>
      </Translated>
    ),
    source: 'notes/today/ch11-facts.md D3 (our reading of mlx-lm), D4',
    vi: { unit: 'token', source: 'notes/today/ch11-facts.md D3 (cách chúng tôi đọc mlx-lm), D4' },
  },
  'x1/X1.5/q-inclusive-bug': {
    kind: 'choice',
    id: 'x1/X1.5/q-inclusive-bug',
    prompt: <Translated vi={<>Một kernel giữ token i khi tổng bao gồm đến i còn dưới p. Trên [0.5, 0.25, 0.125, 0.125] với p = 0.75, điều gì sai?</>}><>A kernel keeps token i while the inclusive sum up to i is below p. On [0.5, 0.25, 0.125, 0.125] with p = 0.75, what goes wrong?</></Translated>,
    options: [
      { label: <Translated vi="Nó chỉ giữ token 0: token cắt qua, có tổng bao gồm đúng bằng 0.75, bị loại">It keeps only token 0: the crossing token, whose inclusive sum is exactly 0.75, is dropped</Translated>, correct: true },
      { label: <Translated vi="Nó giữ cả token 2, thừa một token, vì 0.75 không nhỏ hơn 0.75">It keeps token 2 as well, one token too many, because 0.75 is not below 0.75</Translated>, why: <Translated vi="Token 2 bị loại theo cả hai quy tắc; khối trước nó là 0.75, không dưới p.">Token 2 is dropped by both rules; its mass before is 0.75, not below p.</Translated> },
      { label: <Translated vi="Không gì sai trên hàng này; hai scan chỉ khác nhau khi các xác suất hòa">Nothing on this row; the two scans only differ when probabilities tie</Translated>, why: <Translated vi="Chúng khác nhau ở mọi biên: tổng bao gồm tính cả chính token đó.">They differ at every boundary: the inclusive sum includes the token itself.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
          Quy tắc giữ tập nhỏ nhất có khối xác suất đạt p, nên token cắt qua p được giữ: ở đây là token 0 và 1. Phép kiểm tra
          bao gồm loại token 1, mà chính xác suất của nó đưa tổng lên p. Lệch một đó chính là khác biệt giữa scan bao gồm và
          scan loại trừ.
        </p>
        }
      >
      <p>
        The rule keeps the smallest set whose mass reaches p, so the token that crosses p stays: here tokens 0 and 1. The
        inclusive test drops token 1, whose own probability carries the sum to p. That off-by-one is the difference between
        the inclusive and the exclusive scan.
      </p>
      </Translated>
    ),
    source: 'ch. 11 §11.1, p. 253; x1-facts D9', // i18n-ignore: citation
  },
  'x1/X1.5/q-p1': {
    kind: 'choice',
    id: 'x1/X1.5/q-p1',
    prompt: <Translated vi={<>Vì sao p ≥ 1 phải giữ mọi token một cách tường minh thay vì dựa vào "khối trước &lt; p"?</>}><>Why must p ≥ 1 keep every token explicitly instead of relying on "mass before &lt; p"?</></Translated>,
    options: [
      { label: <Translated vi="Trong float, khối trước các token cuối có thể làm tròn thành 1 hoặc hơn, và khi đó p = 1 sẽ loại chúng">In float the mass before the last tokens can round to 1 or above, and p = 1 would then drop them</Translated>, correct: true },
      { label: <Translated vi="Vì p = 1 nằm ngoài khoảng một kernel scan chấp nhận, và hàm khởi chạy sẽ từ chối lời gọi">Because p = 1 is outside the range a scan kernel accepts, and the launcher would reject the call</Translated>, why: <Translated vi="Scan không biết gì về p; phép so sánh mới biết.">A scan does not know about p; the comparison does.</Translated> },
      { label: <Translated vi="Vì các hàng softmax không bao giờ có tổng đúng bằng 1, nên khối trước token cuối luôn trên 1">Because softmax rows never sum to exactly 1, so the last token's mass before is always above 1</Translated>, why: <Translated vi="Không phải luôn trên 1: nó có thể làm tròn theo cả hai hướng, đó mới là vấn đề.">Not always above 1: it can round either way, which is the problem.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
          Người dùng đặt p = 1 nghĩa là "không top-p". Sampler phía host của llama.cpp trả về sớm khi p ≥ 1, và hợp đồng của
          lab 11.6 giữ mọi thứ tại p ≥ 1 vì cùng lý do.
        </p>
        }
      >
      <p>
        A user who sets p = 1 means "no top-p". llama.cpp's host sampler returns early for p ≥ 1, and lab 11.6's contract
        keeps everything at p ≥ 1 for the same reason.
      </p>
      </Translated>
    ),
    source: 'notes/today/ch11-facts.md D2; labs/ch11/lab6_topp', // i18n-ignore: citation
  },
  'x1/X1.5/q-adaptive': {
    kind: 'choice',
    id: 'x1/X1.5/q-adaptive',
    prompt: <Translated vi={<>Khi nào top-p phía host của llama.cpp phải sắp xếp nhiều hơn 256 ứng viên đầu tiên?</>}><>When does llama.cpp's host top-p have to sort more than its first 256 candidates?</></Translated>,
    options: [
      { label: <Translated vi="Khi 256 token đó giữ ít hơn p khối xác suất, điều xảy ra trên các phân phối phẳng">When those 256 hold less than p of the mass, which happens on flat distributions</Translated>, correct: true },
      { label: <Translated vi="Bất cứ khi nào bộ từ vựng có hơn 1,024 token, tức mọi mô hình hiện nay">Whenever the vocabulary has more than 1,024 tokens, which is every current model</Translated>, why: <Translated vi="Hơn 1,024 ứng viên là lúc nó bắt đầu bằng sắp xếp một phần 256, không phải lúc nó mở rộng.">More than 1,024 candidates is when it starts with the partial sort of 256, not when it widens.</Translated> },
      { label: <Translated vi="Không bao giờ: sắp xếp một phần 256 luôn đủ cho mọi p dưới 1">Never: a partial sort of 256 is always enough for any p below 1</Translated>, why: <Translated vi="Trên một hàng phẳng có thể cần hàng nghìn token để đạt p.">On a flat row thousands of tokens can be needed to reach p.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
          Nó sắp xếp một phần 256 token đầu, chạy tổng tích lũy, và chỉ mở rộng khi tổng chưa đạt p. Trong trình khám phá,
          làm phẳng hàng cho đến khi đường chuyển từ 256 sang cả hàng.
        </p>
        }
      >
      <p>
        It partial-sorts the top 256, runs the cumulative sum, and widens only when the sum has not reached p. In the
        explorer, flatten the row until the route switches from 256 to the whole row.
      </p>
      </Translated>
    ),
    source: 'notes/today/ch14-facts.md D3', // i18n-ignore: citation
  },
  'x1/X1.5/q-why-scan': {
    kind: 'choice',
    id: 'x1/X1.5/q-why-scan',
    prompt: <Translated vi={<>Vì sao lab 11.6 tính scan loại trừ thay vì scan bao gồm cho top-p?</>}><>Why does lab 11.6 compute an exclusive scan rather than an inclusive one for top-p?</></Translated>,
    options: [
      { label: <Translated vi="Phép kiểm tra giữ cần khối xác suất xếp hạng hẳn trước mỗi token, thứ scan loại trừ cho trực tiếp">The keep test needs the mass ranked strictly before each token, which the exclusive scan gives directly</Translated>, correct: true },
      { label: <Translated vi="Scan loại trừ rẻ hơn scan bao gồm trên GPU, nên lab tiết kiệm một lượt qua hàng">An exclusive scan is cheaper on a GPU than an inclusive one, so the lab saves a pass over the row</Translated>, why: <Translated vi="Chúng tốn như nhau; cái này là cái kia dịch đi một vị trí.">They cost the same; each is the other shifted by one place.</Translated> },
      { label: <Translated vi="Scan bao gồm không thể tính trong float mà không làm tròn giá trị cuối lên trên 1">An inclusive scan cannot be computed in float without rounding the last value above 1</Translated>, why: <Translated vi="Cả hai scan đều làm tròn; đó không phải điều quyết định lựa chọn.">Both scans round; that is not what decides the choice.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
          Token có tổng bao gồm đầu tiên đạt p vẫn có khối trước nó dưới p, nên phép kiểm tra loại trừ giữ nó, đúng như định
          nghĩa. Với scan bao gồm bạn sẽ phải so phần tử trước đó, cùng một việc nhưng lệch một vị trí.
        </p>
        }
      >
      <p>
        The token whose inclusive sum first reaches p still has mass before it below p, so the exclusive test keeps it,
        which is the definition. With an inclusive scan you would compare the previous element, the same thing one place
        over.
      </p>
      </Translated>
    ),
    source: 'ch. 11 §11.1, p. 253; labs/ch11/lab6_topp', // i18n-ignore: citation
  },
};

// ---------------------------------------------------------------- the section
export function SX15() {
  return (
    <>
      <Translated
        vi={
          <p>
          Top-p, hay nucleus, giữ các token có khả năng cao nhất cho đến khi khối xác suất của chúng đạt p. Bạn đã có sẵn ba
          mảnh: lab 14.3 sắp xếp một hàng từ lớn nhất và ổn định, lab 10.3 biến logits thành xác suất, và lab 11.6 scan các
          xác suất đã sắp xếp và giữ một tiền tố. Trang này nêu chính xác quy tắc và cách phân định hòa, ghép ba mảnh lại, và
          hỏi thực sự cần sắp xếp bao nhiêu phần của hàng.
        </p>
        }
      >
      <p>
        Top-p, the nucleus, keeps the most likely tokens until their mass reaches p. You have the three pieces already:
        lab 14.3 sorts a row largest first and stably, lab 10.3 turns logits into probabilities, and lab 11.6 scans the
        sorted probabilities and keeps a prefix. This page states the rule and its tie-break exactly, composes the three,
        and asks how much of the row really needs sorting.
      </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
            Top-p không có trong sách. Scan loại trừ và hình xúc xích là §11.1 (pp. 252–254); phép sắp xếp key-value ổn định
            là §14.1 (p. 330) và lab 14.3.
          </p>
          }
        >
        <p>
          Top-p is not in the book. The exclusive scan and the sausage picture are §11.1 (pp. 252–254); the stable key-value
          sort is §14.1 (p. 330) and lab 14.3.
        </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Quy tắc, một cách chính xác">The rule, exactly</Translated></h2>
      <Translated
        vi={
          <p>
          Holtzman et al. định nghĩa tập top-p là tập token nhỏ nhất có tổng xác suất ít nhất bằng p. Vậy token cắt qua p
          nằm trong tập. Trên một hàng sắp xếp từ lớn nhất, tập đó là một tiền tố, và phép kiểm tra cho mỗi token là: giữ
          token i khi i = 0, hoặc p ≥ 1, hoặc khối xác suất xếp hạng trước nó dưới p. Đó là quy tắc của lab 11.6. Các sampler
          host và backend của llama.cpp, cách viết tăng dần của transformers và vLLM đều quy về nó trên các hàng có tổng bằng 1.
        </p>
        }
      >
      <p>
        Holtzman et al. define the top-p set as the smallest set of tokens whose total probability is at least p. So the
        token that crosses p is in. On a row sorted largest first that set is a prefix, and the test for each token is: keep
        token i when i = 0, or p ≥ 1, or the mass ranked before it is below p. That is lab 11.6's rule. llama.cpp's host
        and backend samplers, transformers' ascending formulation and vLLM all reduce to it on rows that sum to 1.
      </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x1-facts.md D9; ch11-facts D2, D4)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
            Bài báo định nghĩa V(p) là "the smallest set such that Σ P ≥ p" (tập nhỏ nhất sao cho Σ P ≥ p) và không đưa ra
            quy tắc phân định hòa nào.
          </p>
          }
        >
        <p>
          The paper defines V(p) as "the smallest set such that Σ P ≥ p" and gives no tie rule.
        </p>
        </Translated>
      </Today>
      <Predict
        id="x1/X1.5/p-crossing"
        prompt={<Translated vi={<>Các xác suất đã sắp xếp [0.5, 0.25, 0.125, 0.0625, 0.0625] và p = 0.875. M5 giữ bao nhiêu token?</>}><>Sorted probabilities [0.5, 0.25, 0.125, 0.0625, 0.0625] and p = 0.875. How many tokens does M5 keep?</></Translated>}
        unit={tr('tokens', 'token')}
      >
        {(v) => {
          const r = topP(CROSSING.probs, CROSSING.p, 'exclusive');
          return (
            <>
              <Translated
                vi={
                  <p data-reveal="crossing">
                    <b>{r.kept}</b>{v === r.kept ? ', đúng như bạn nói' : ''}. Khối trước các token là 0, 0.5, 0.75, 0.875 và
                    0.9375; ba giá trị đầu dưới p. Tổng bao gồm của token 2 đúng bằng 0.875: nó là token cắt qua và được giữ.
                    Mọi giá trị ở đây đều là lũy thừa của hai, nên số học float cũng quyết định nó chính xác.
                  </p>
                }
              >
              <p data-reveal="crossing">
                <b>{r.kept}</b>{v === r.kept ? ', as you said' : ''}. The masses before the tokens are 0, 0.5, 0.75, 0.875 and
                0.9375; the first three are below p. Token 2's inclusive sum is exactly 0.875: it is the crossing token and
                it stays. Every value here is a power of two, so float arithmetic decides it exactly too.
              </p>
              </Translated>
              <CrossingBars />
            </>
          );
        }}
      </Predict>
      <Ask q={questionsSX15['x1/X1.5/q-inclusive-bug']} />
      <Ask q={questionsSX15['x1/X1.5/q-why-scan']} />

      <h2><Translated vi="Phép sắp xếp phân định các trường hợp hòa">Ties are broken by the sort</Translated></h2>
      <Translated
        vi={
          <p>
          "Tập nhỏ nhất" không duy nhất khi các token hòa nhau ở biên. Trong M5 việc phân định hòa dựa vào vị trí, tức là vào
          phép sắp xếp. Các ứng viên đến từ lab X1.1 theo thứ tự cột tăng dần, và phép sắp xếp của lab 14.3 là ổn định, nên
          các xác suất bằng nhau giữ thứ tự token id tăng dần và id thấp hơn qua được biên.
        </p>
        }
      >
      <p>
        "Smallest set" is not unique when tokens tie at the boundary. In M5 the tie-break is position, which means the
        sort. The candidates arrive from lab X1.1 in ascending column order, and lab 14.3's sort is stable, so equal
        probabilities stay in order of increasing token id and the lower id survives the boundary.
      </p>
      </Translated>
      <Source section="14.1" pages="330" />
      <Translated
        vi={
          <p>
          Các dự án khác nhau ở đây. mlx-lm giữ mọi token hòa với token biên, nên nó có thể giữ nhiều hơn. transformers sắp
          xếp tăng dần với <code>torch.sort</code> mặc định, không ổn định, nên nó giữ token hòa nào là không xác định. Trình
          khám phá <a href={href('ch11', '11.1')}>"Nucleus kết thúc ở đâu"</a> của chương 11 đặt các quy tắc cạnh nhau trên
          những hàng bạn chọn.
        </p>
        }
      >
      <p>
        The projects differ here. mlx-lm keeps every token tied with the boundary token, so it can keep more. transformers
        sorts ascending with the default, unstable <code>torch.sort</code>, so which tied token it keeps is not specified.
        Chapter 11's <a href={href('ch11', '11.1')}>"Where the nucleus ends"</a> explorer puts the rules side by side on
        rows you choose.
      </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch11-facts.md D3, D4; ch14-facts D5)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
            mlx-lm <code>apply_top_p</code>: che <code>logprobs &lt; threshold</code>, trong đó ngưỡng là log-xác suất của
            chính token biên, nên các token hòa với nó được giữ (theo cách chúng tôi đọc).
          </p>
          }
        >
        <p>
          mlx-lm <code>apply_top_p</code>: masks <code>logprobs &lt; threshold</code>, where the threshold is the boundary
          token's own log-probability, so its ties survive (our reading).
        </p>
        </Translated>
      </Today>
      <Ask q={questionsSX15['x1/X1.5/q-mlx-ties']} label={tr('Work it out', 'Tự tính ra')} />

      <h2><Translated vi="Ghép từ ba lab">Composed from three labs</Translated></h2>
      <Translated
        vi={
          <p>
          Trong M5, top-p chạy trên các token còn lại sau top-k, một tensor [rows × k] gồm logits cùng token id của chúng.{' '}
          <code>launchSortRowsDesc</code> của lab 14.3 sắp xếp mỗi hàng từ lớn nhất. <code>launchSoftmaxRows</code> của lab
          10.3 biến chúng thành xác suất với scale 1, vì llama.cpp quyết định top-p trước temperature.{' '}
          <code>launchTopPMask</code> của lab 11.6 ghi khối trước mỗi token, mặt nạ giữ và số lượng được giữ. Các xác suất được
          chuẩn hóa lại trên k token còn lại, như trong llama.cpp và transformers.
        </p>
        }
      >
      <p>
        In M5, top-p runs on the top-k survivors, a [rows × k] tensor of logits with their token ids. Lab 14.3's{' '}
        <code>launchSortRowsDesc</code> sorts each row largest first. Lab 10.3's <code>launchSoftmaxRows</code> turns
        them into probabilities with scale 1, because llama.cpp decides top-p before temperature. Lab 11.6's{' '}
        <code>launchTopPMask</code> writes the mass before each token, the keep mask and the kept count. The probabilities
        are renormalised over the k survivors, as in llama.cpp and transformers.
      </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x1-facts.md D1, D4)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
            Chuỗi mặc định của llama.cpp chạy top-k, rồi top-p và min-p, và temperature cuối cùng; transformers áp dụng
            temperature trước.
          </p>
          }
        >
        <p>
          llama.cpp's default chain runs top-k, then top-p and min-p, and temperature last; transformers applies
          temperature first.
        </p>
        </Translated>
      </Today>
      <div className="wide" style={{ maxWidth: '58rem', display: 'flex', flexWrap: 'wrap', gap: '0.4rem', alignItems: 'center', margin: '0.4rem 0 1.4rem', fontFamily: 'var(--sans)', fontSize: '0.82rem' }} data-part="chain">
        {[
          ['candidates', tr('[rows × k] logits and ids, column order (lab X1.1)', '[rows × k] logits và id, thứ tự cột (lab X1.1)')], // i18n-ignore: variable name shown as code
          ['launchSortRowsDesc', tr('lab 14.3: largest first, stable', 'lab 14.3: lớn nhất trước, ổn định')],
          ['launchSoftmaxRows', 'lab 10.3: scale 1'], // i18n-ignore: lab number and argument value
          ['launchTopPMask', tr('lab 11.6: mass before, keep, kept', 'lab 11.6: khối trước, giữ, số được giữ')],
          ['kept count', tr('a prefix of each sorted row', 'một tiền tố của mỗi hàng đã sắp xếp')], // i18n-ignore: variable name shown as code
        ].map(([a, b], i, all) => (
          <span key={a} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
            <span className="panel" style={{ padding: '0.35rem 0.6rem' }}>
              <code>{a}</code>
              <br />
              <span className="muted">{b}</span>
            </span>
            {i < all.length - 1 ? <span aria-hidden="true">→</span> : null}
          </span>
        ))}
      </div>

      <h2><Translated vi="Bạn cần sắp xếp bao nhiêu?">How much do you need to sort?</Translated></h2>
      <Translated
        vi={
          <p>
          Với top-k chạy trước, phép sắp xếp thấy k giá trị, không phải cả bộ từ vựng. Không có nó, một top-p đơn giản sắp
          xếp cả hàng, và trên một phân phối nhọn hầu như toàn bộ phép sắp xếp đó bị lãng phí: p = 0.95 có thể chỉ cần vài
          token. Sampler phía host của llama.cpp phòng hờ. Với hơn 1,024 ứng viên, nó chỉ sắp xếp một phần 256 token đầu, và
          chỉ mở rộng nếu chúng giữ ít hơn p. Mặc định của vLLM tránh hẳn phép sắp xếp bằng một phép tìm pivot. Mục X1.6 cho
          thấy cách thứ ba, lọc bằng min-p trước khi sắp xếp.
        </p>
        }
      >
      <p>
        With top-k first, the sort sees k values, not the vocabulary. Without it, a straightforward top-p sorts the whole
        row, and on a peaked distribution almost all of that sort is wasted: p = 0.95 may need a handful of tokens. llama.cpp's
        host sampler hedges. With more than 1,024 candidates it partial-sorts only the top 256, and widens only if those
        hold less than p. vLLM's default avoids the sort altogether with a pivot search. Section X1.6 shows a third way,
        filtering by min-p before sorting.
      </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md D3, D7)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
            llama.cpp <code>llama_sampler_top_p_apply</code>: "try adaptive top-k sorting" (thử sắp xếp top-k thích ứng) với
            256 khi có hơn 1,024 ứng viên. vLLM: <code>apply_top_k_top_p_triton</code>, dựa trên bài báo Qrita (chưa đọc ở đây).
          </p>
          }
        >
        <p>
          llama.cpp <code>llama_sampler_top_p_apply</code>: "try adaptive top-k sorting" of 256 when there are more than
          1,024 candidates. vLLM: <code>apply_top_k_top_p_triton</code>, based on the Qrita paper (not read here).
        </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
          Trình khám phá bên dưới vẽ một hàng cỡ cả bộ từ vựng, tìm số lượng top-p giữ, và đếm xem mỗi chiến lược này đưa bao
          nhiêu giá trị cho phép sắp xếp.
        </p>
        }
      >
      <p>
        The explorer below draws a whole vocabulary row, finds top-p's kept count, and counts how many values each of
        these strategies hands to the sort.
      </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
            Các hàng là tổng hợp: logits kiểu Zipf có seed, hạng r nhận −s ln(r + 1) cộng một nhiễu nhỏ, với s từ phẳng (0) đến
            nhọn (3). Không phải đầu ra của một mô hình thật. Số lượng được giữ tính ở double.
          </p>
          }
        >
        <p>
          The rows are synthetic: seeded Zipf-like logits, rank r getting −s ln(r + 1) plus a small jitter, with s from flat
          (0) to peaked (3). Not a real model's output. Kept counts are computed in double.
        </p>
        </Translated>
      </Simplification>
      <SortExplorer />
      <Translated
        vi={
          <p>
          Trên một hàng nhọn, nucleus chỉ là vài token và sắp xếp 150,000 giá trị để tìm chúng hầu như là lãng phí hoàn toàn.
          Trên một hàng phẳng, riêng top-p giữ hàng chục nghìn, và khi đó chính top-k giới hạn phép sắp xếp.
        </p>
        }
      >
      <p>
        On a peaked row the nucleus is a few tokens and sorting 150,000 values to find them is almost all waste. On a flat
        row top-p alone keeps tens of thousands, and then it is top-k that caps the sort.
      </p>
      </Translated>
      <Ask q={questionsSX15['x1/X1.5/q-adaptive']} />

      <h2><Translated vi="Số float ở biên">Floats at the boundary</Translated></h2>
      <Translated
        vi={
          <p>
          Một quyết định chỉ chính xác đến mức khối trước token chính xác. Lab 11.6 chấp nhận cả hai đáp án cho một token có khối trước
          chính xác nằm trong khoảng 1e-4 quanh p (tương đối so với khối đó), vì các backend và thư viện cộng theo những thứ
          tự khác nhau. Các hàng có xác suất là lũy thừa của hai quyết định chính xác trên mọi backend, đó là lý do các bài
          kiểm tra biên của lab dùng chúng. Và p ≥ 1 phải giữ mọi thứ một cách tường minh.
        </p>
        }
      >
      <p>
        A decision is only as exact as the mass before the token. Lab 11.6 accepts either answer for a token whose exact
        mass before it lies within 1e-4 of p (relative to that mass), because backends and libraries add in different
        orders. Rows whose probabilities are powers of two decide exactly on every backend, which is why the lab's
        boundary tests use them. And p ≥ 1 must keep everything explicitly.
      </p>
      </Translated>
      <Source section="11.1" pages="252–254" />
      <Ask q={questionsSX15['x1/X1.5/q-p1']} />

      <h2><Translated vi="Phép sắp xếp tốn bao nhiêu trong chuỗi">What the sort costs in the chain</Translated></h2>
      <Translated
        vi={
          <p>
          Lab X1.2 ghép chuỗi từ ba lab và đo thời gian từng bước, với top-k tắt (phép sắp xếp thấy cả hàng) và với top-k =
          40. Trước khi chạy, hãy chốt một con số.
        </p>
        }
      >
      <p>
        Lab X1.2 composes the chain from the three labs and times each step, with top-k off (the sort sees the whole row)
        and with top-k = 40. Before you run it, commit to a number.
      </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch11-facts.md D2 to D4; ch14-facts D3, D7)', '2026-11-20 (notes/today/ch11-facts.md D2 đến D4; ch14-facts D3, D7)')}>
        <Translated
          vi={
            <p>
            Top-p phía backend của llama.cpp là argsort, softmax, cumsum và một mặt nạ bậc thang, với thêm một token được tính
            bao gồm. transformers sắp xếp tăng dần và loại các token có tổng tích lũy tối đa 1 − p. Mặc định của vLLM không cần
            sắp xếp.
          </p>
          }
        >
        <p>
          llama.cpp's backend top-p is argsort, softmax, cumsum and a step mask, with one extra token made inclusive.
          transformers sorts ascending and removes tokens whose cumulative sum is at most 1 − p. vLLM's default needs no
          sort.
        </p>
        </Translated>
      </Today>
      <Predict
        id="x1/X1.5/p-sort-share"
        prompt={
          <Translated
            vi={
              <>
                Với top-k tắt, 64 hàng 151,936: phép sắp xếp chiếm bao nhiêu phần trăm thời gian của chuỗi ghép? (Phần đáp án
                cũng cho thấy tỉ lệ tại top-k = 40.)
              </>
            }
          >
          <>
            With top-k off, 64 rows of 151,936: what percentage of the composed chain's time is the sort? (The reveal also
            shows the share at top-k = 40.)
          </>
          </Translated>
        }
        unit="%"
      >
        {(v) => <MeasuredShare predicted={v} />}
      </Predict>
      <Translated
        vi={
          <p>
          Số lượng được giữ là độ dài tiền tố của mỗi hàng. Trang tiếp theo, <a href={href('x1', 'X1.6')}>min-p</a>, cũng là
          một tiền tố, và hai cái ghép với nhau bằng cách lấy cái ngắn hơn.
        </p>
        }
      >
      <p>
        The kept count is a prefix length per row. The next page, <a href={href('x1', 'X1.6')}>min-p</a>, is a prefix too,
        and the two compose by taking the shorter.
      </p>
      </Translated>
    </>
  );
}
