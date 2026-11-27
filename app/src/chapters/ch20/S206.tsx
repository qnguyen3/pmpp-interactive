import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { benchRows, fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { batchToReach, batchedDecodeAttentionAI, linearAI } from '../../sim/kvmath';
import { FIG_PROMPT, FIG_STEPS, expectedPerPass, figCounters, replayFigure, simulate } from '../../sim/specdecode';
import { BatchShareScene } from './BatchShareScene';
import { KvExplorer } from './KvExplorer';

// ---------------------------------------------------------------- ridges from lab 5.1

interface Ridge {
  key: string;
  label: string;
  ridge: number;
  bw?: number; // GB/s
}
const BOOK_RIDGE: Ridge = {
  key: 'book',
  get label() {
    return tr('the book’s example', 'ví dụ của sách');
  },
  ridge: 66900 / 3350,
  bw: 3350,
};
function ridges(reports: Reports): Ridge[] {
  return realBackends(reports).flatMap((b) => {
    const r = benchRows(reports[b], 'ch05/lab1_roofline').find((x) => x.name === 'roofline');
    const peak = num(r?.compute_gflops);
    const bw = num(r?.bandwidth_gb_per_s);
    // A getter, because the list is memoised on the reports alone and the label must follow the language.
    return peak && bw
      ? [
          {
            key: b,
            get label() {
              return tr(`yours, ${String(reports[b].device.name ?? b)} (${b})`, `của bạn, ${String(reports[b].device.name ?? b)} (${b})`);
            },
            ridge: peak / bw,
            bw,
          },
        ]
      : [];
  });
}

// ---------------------------------------------------------------- batching explorer

const LAYERS = [
  {
    key: 'book',
    get label() {
      return tr('1,024 → 4,096 (the book’s)', '1,024 → 4,096 (của sách)');
    },
    inF: 1024,
    outF: 4096,
  },
  {
    key: 'qkv',
    get label() {
      return tr('4,096 → 12,288 (fused Q, K, V)', '4,096 → 12,288 (gộp Q, K, V)');
    },
    inF: 4096,
    outF: 12288,
  },
  { key: 'square', label: '4,096 → 4,096', inF: 4096, outF: 4096 },
];
const N_ATTN = 4096;

function BatchPlot({ inF, outF, p, b, lines }: { inF: number; outF: number; p: number; b: number; lines: Ridge[] }) {
  const W = 600;
  const H = 300;
  const pad = { l: 52, r: 14, t: 14, b: 38 };
  const yMin = 0.1;
  const yMax = 2000;
  const X = (bb: number) => pad.l + (Math.log2(bb) / 9) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * (H - pad.t - pad.b);
  const bs = Array.from({ length: 91 }, (_, i) => 2 ** (i / 10));
  const lin = bs.map((x) => `${X(x).toFixed(1)},${Y(linearAI(x, inF, outF, p)).toFixed(1)}`).join(' ');
  const att = batchedDecodeAttentionAI(1, N_ATTN, p);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Intensity against batch size', 'Cường độ tính toán theo kích thước batch')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {[1, 2, 4, 8, 16, 32, 64, 128, 256, 512].map((x) => (
        <g key={x}>
          <line x1={X(x)} x2={X(x)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(x)} y={H - pad.b + 14} fontSize="10.5" textAnchor="middle">{x}</text>
        </g>
      ))}
      {[0.1, 1, 10, 100, 1000].map((v) => (
        <g key={v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 6} y={Y(v) + 4} fontSize="10.5" textAnchor="end">{v}</text>
        </g>
      ))}
      <text x={(W + pad.l) / 2} y={H - 5} fontSize="11" textAnchor="middle">
        <Translated vi="kích thước batch">batch size</Translated>
      </text>
      <text x={12} y={(H - pad.b + pad.t) / 2} fontSize="11" textAnchor="middle" transform={`rotate(-90 12 ${(H - pad.b + pad.t) / 2})`}>
        <Translated vi="FLOP trên mỗi byte">FLOP per byte</Translated>
      </text>
      {lines.map((r, i) => (
        <g key={r.key} data-ridge={r.ridge.toFixed(3)}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(r.ridge)} y2={Y(r.ridge)} stroke="var(--ink)" strokeDasharray={r.key === 'book' ? '2 4' : '6 3'} />
          <text x={W - pad.r - 4} y={Y(r.ridge) - 4 - i * 12} fontSize="10" textAnchor="end">
            <Translated vi={<>điểm gãy, {r.label}: {fmt(r.ridge)}</>}>ridge, {r.label}: {fmt(r.ridge)}</Translated>
          </text>
        </g>
      ))}
      <polyline points={lin} fill="none" stroke="var(--global)" strokeWidth="2.4" />
      <line x1={pad.l} x2={W - pad.r} y1={Y(att)} y2={Y(att)} stroke="var(--reg)" strokeWidth="2.4" />
      <text x={pad.l + 6} y={Y(att) + 15} fontSize="10.5" style={{ fill: 'var(--reg)' }}>
        <Translated vi="decode attention, mỗi yêu cầu một cache riêng">decode attention, each request its own cache</Translated>
      </text>
      <g data-linear-key="">
        <line x1={pad.l + 8} x2={pad.l + 30} y1={pad.t + 14} y2={pad.t + 14} stroke="var(--global)" strokeWidth="2.4" />
        <text x={pad.l + 36} y={pad.t + 18} fontSize="10.5" style={{ fill: 'var(--global)' }}>
          <Translated vi="lớp tuyến tính, trọng số dùng chung">linear layer, weights shared</Translated>
        </text>
      </g>
      <circle cx={X(b)} cy={Y(linearAI(b, inF, outF, p))} r="5" fill="var(--global)" />
      <circle cx={X(b)} cy={Y(att)} r="5" fill="var(--reg)" />
    </svg>
  );
}

function BatchingBench() {
  const { reports } = useReports();
  const lines = useMemo(() => [...ridges(reports), BOOK_RIDGE], [reports]);
  const [logB, setLogB] = useState(0);
  const [layerKey, setLayerKey] = useState('book');
  const [p, setP] = useState(2);
  const b = 2 ** logB;
  const L = LAYERS.find((x) => x.key === layerKey)!;
  const lin = linearAI(b, L.inF, L.outF, p);
  const att = batchedDecodeAttentionAI(b, N_ATTN, p);
  return (
    <Bench
      title={tr('Batching raises one intensity, not the other', 'Batching nâng một cường độ tính toán, không nâng cái kia')}
      task={
        <Translated vi="Tìm batch mà tại đó lớp tuyến tính vượt qua điểm gãy của bạn. Rồi thử làm cho attention vượt qua nó.">
          Find the batch at which the linear layer crosses your ridge. Then try to make attention cross it.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Lớp tuyến tính đếm trọng số của nó một lần cùng mọi hàng đầu vào và đầu ra (cách đó tái tạo con số 1 và 315 của
              sách). Attention là một bước decode trên một cache {N_ATTN.toLocaleString('en-US')} token cho mỗi yêu cầu. Các
              điểm gãy là đỉnh chia cho băng thông từ lab 5.1, hoặc ví dụ của sách (66.9 TFLOP/s chia 3.35 TB/s, ch. 5 p.
              94).
            </>
          }
        >
          <>
            The linear layer counts its weights once and every input and output row (that reproduces the book’s 1 and 315).
            Attention is one decode step over a {N_ATTN.toLocaleString('en-US')}-token cache per request. Ridges are peak
            over bandwidth from lab 5.1, or the book’s example (66.9 TFLOP/s over 3.35 TB/s, ch. 5 p. 94).
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          <Translated vi={<>Batch b = {b}</>}>Batch b = {b}</Translated>
          <input type="range" min="0" max="9" step="1" value={logB} aria-label={tr('Batch size', 'Kích thước batch')} onChange={(e) => setLogB(Number(e.target.value))} />
        </label>
        <label className="field" style={{ maxWidth: '100%', minWidth: 0 }}>
          <Translated vi="Lớp tuyến tính">Linear layer</Translated>
          <select value={layerKey} onChange={(e) => setLayerKey(e.target.value)} aria-label={tr('Linear layer', 'Lớp tuyến tính')} style={{ maxWidth: '100%' }}>
            {LAYERS.map((x) => <option key={x.key} value={x.key}>{x.label}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Phần tử">Element</Translated>
          <select value={p} onChange={(e) => setP(Number(e.target.value))} aria-label={tr('Element size', 'Kích thước phần tử')} style={{ maxWidth: '100%' }}>
            <option value={4}>{tr('fp32, 4 bytes', 'fp32, 4 byte')}</option>
            <option value={2}>{tr('16-bit, 2 bytes', '16-bit, 2 byte')}</option>
            <option value={1}>{tr('8-bit, 1 byte', '8-bit, 1 byte')}</option>
          </select>
        </label>
      </div>
      <div className="cols plot">
        <BatchPlot inF={L.inF} outF={L.outF} p={p} b={b} lines={lines} />
        <div>
          <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-batching="">
            <tbody>
              <tr><td><Translated vi={<>Lớp tuyến tính ở b = {b}</>}>Linear layer at b = {b}</Translated></td><td className="num" data-linear={lin.toFixed(4)}><b>{fmt(lin, 4)}</b> FLOP/B</td></tr>
              <tr><td><Translated vi={<>Decode attention ở b = {b}</>}>Decode attention at b = {b}</Translated></td><td className="num" data-attn={att.toFixed(4)}><b>{fmt(att, 4)}</b> FLOP/B</td></tr>
            </tbody>
          </table>
          <table className="data" style={{ background: 'var(--page)', margin: '0.8rem 0 0' }}>
            <thead>
              <tr>
                <th><Translated vi="Điểm gãy">Ridge</Translated></th>
                <th className="num"><Translated vi="lớp tuyến tính vượt qua ở">linear crosses at</Translated></th>
                <th className="num">attention{/* i18n-ignore: term kept in English */}</th>
              </tr>
            </thead>
            <tbody>
              {lines.map((r) => {
                const at = batchToReach(r.ridge, L.inF, L.outF, p);
                return (
                  <tr key={r.key} data-cross={at ?? 'never'}>
                    <td>{r.label}, {fmt(r.ridge)}</td>
                    <td className="num">{at == null ? tr('never', 'không bao giờ') : `b = ${at}`}</td>
                    <td className="num">{att >= r.ridge ? tr('already', 'đã vượt') : tr('never', 'không bao giờ')}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- speculative decoding

const ROUNDS = replayFigure();

function Tok({ t, tone }: { t: string; tone?: 'prefix' | 'draft' | 'ok' | 'bad' | 'mute' }) {
  const style: CSSProperties = {
    display: 'inline-block',
    padding: '0.05rem 0.4rem', // i18n-ignore: CSS value
    margin: '0.1rem 0.2rem 0.1rem 0', // i18n-ignore: CSS value
    borderRadius: 3,
    fontFamily: 'var(--sans)',
    fontSize: '0.8rem',
    border: '1px solid var(--rule)', // i18n-ignore: CSS value
    background: tone === 'prefix' ? 'var(--bench)' : tone === 'ok' ? 'var(--ok-wash)' : tone === 'bad' ? 'var(--fail-wash)' : 'var(--cell)',
    color: tone === 'ok' ? 'var(--ok)' : tone === 'bad' ? 'var(--fail)' : tone === 'mute' ? 'var(--ink-3)' : 'var(--ink)',
    borderColor: tone === 'ok' ? 'var(--ok)' : tone === 'bad' ? 'var(--fail)' : 'var(--rule)',
  };
  return <span style={style}>{t}</span>;
}

function FigureReplay() {
  const [s, setS] = useState(0);
  const c = figCounters(s);
  const shown = (round: number, phase: 'draft' | 'verify' | 'accept') => FIG_STEPS.findIndex((x) => x.round === round && x.phase === phase) <= s;
  const prefixText = (prefix: string[]) => (prefix.length > FIG_PROMPT.length ? '… ' + prefix.slice(FIG_PROMPT.length).join(' ') : prefix.slice(-2).join(' '));
  return (
    <div data-replay-step={s}>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <button onClick={() => setS(0)} disabled={s === 0}>
          <Translated vi="Bắt đầu lại">Restart</Translated>
        </button>
        <button onClick={() => setS(s - 1)} disabled={s === 0}>
          <Translated vi="Lùi">Back</Translated>
        </button>
        <button className="primary" onClick={() => setS(s + 1)} disabled={s >= FIG_STEPS.length - 1}>
          <Translated vi="Bước">Step</Translated>
        </button>
        <span className="muted" aria-live="polite">
          {s === 0
            ? tr('the prompt', 'prompt')
            : tr(
                `round ${FIG_STEPS[s].round + 1}: ${FIG_STEPS[s].phase}`,
                `vòng ${FIG_STEPS[s].round + 1}: ${({ start: 'bắt đầu', draft: 'phác thảo', verify: 'kiểm chứng', accept: 'chấp nhận' } as Record<string, string>)[FIG_STEPS[s].phase] ?? FIG_STEPS[s].phase}`,
              )}
        </span>
      </div>
      <p style={{ margin: '0 0 0.6rem' }}>{FIG_PROMPT.map((t, i) => <Tok key={i} t={t} tone="prefix" />)}</p>
      {ROUNDS.map((r, ri) =>
        shown(ri, 'draft') ? (
          <div key={ri} className="panel" style={{ marginBottom: '0.6rem' }} data-round={ri + 1}>
            <div className="mx-label">
              <Translated vi={<>Vòng {ri + 1}</>}>Round {ri + 1}</Translated>
            </div>
            <div><span className="muted"><Translated vi="Dự đoán (mô hình nháp): ">Predict (draft model): </Translated></span>{r.draft.map((t, i) => <Tok key={i} t={t} tone="draft" />)}</div>
            {shown(ri, 'verify') ? (
              <div style={{ marginTop: '0.3rem' }}>
                <span className="muted">
                  <Translated vi={<>Kiểm chứng (mô hình đích, một lượt, {r.draft.length} hàng):</>}>
                    Verify (target model, one pass, {r.draft.length} rows):
                  </Translated>
                </span>
                {r.draft.map((t, i) => (
                  <div key={i} style={{ marginLeft: '1rem' }} data-verify={r.ok[i] ? 'ok' : 'bad'}>
                    <span className="muted">
                      <Translated vi={<>sau “{prefixText([...r.prefix, ...r.draft.slice(0, i)])}” mô hình đích nói </>}>
                        after “{prefixText([...r.prefix, ...r.draft.slice(0, i)])}” the target says{' '}
                      </Translated>
                    </span>
                    <Tok t={r.target[i]} tone={r.ok[i] ? 'ok' : 'bad'} />
                    <span className={r.ok[i] ? 't-ok' : 't-fail'}>{r.ok[i] ? tr('✓ matches', '✓ khớp') : tr(`✗ the draft said “${t}”`, `✗ bản nháp nói “${t}”`)}</span>
                  </div>
                ))}
              </div>
            ) : null}
            {shown(ri, 'accept') ? (
              <div style={{ marginTop: '0.3rem' }} data-accepted={r.accepted.join(' ')}>
                <span className="muted"><Translated vi="Chấp nhận: ">Accept: </Translated></span>
                {r.accepted.map((t, i) => <Tok key={i} t={t} tone="ok" />)}
              </div>
            ) : null}
          </div>
        ) : null,
      )}
      <p className="muted" style={{ margin: 0 }} data-counters={`${c.targetPasses}/${c.tokens.length}`}>
        <Translated
          vi={
            <>
              Số lượt của mô hình đích đến giờ: <b>{c.targetPasses}</b>. Số token được giữ: <b>{c.tokens.length}</b>
              {c.tokens.length ? <> ({c.tokens.join(' ')})</> : null}. Decode thông thường sẽ cần {c.tokens.length} lượt
              của mô hình đích cho chúng.
            </>
          }
        >
          Target passes so far: <b>{c.targetPasses}</b>. Tokens kept: <b>{c.tokens.length}</b>
          {c.tokens.length ? <> ({c.tokens.join(' ')})</> : null}. Plain decoding would have needed {c.tokens.length} target{' '}
          {c.tokens.length === 1 ? 'pass' : 'passes'} for them.
        </Translated>
      </p>
    </div>
  );
}

function OwnSpec() {
  const [k, setK] = useState(4);
  const [alpha, setAlpha] = useState(0.7);
  const [seed, setSeed] = useState(1);
  const [keep, setKeep] = useState(false);
  const r = useMemo(() => simulate({ k, alpha, seed, tokens: 64, keepTarget: keep }), [k, alpha, seed, keep]);
  const e = expectedPerPass(k, alpha, keep);
  return (
    <div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi={<>Độ sâu suy đoán k = {k}</>}>Speculation depth k = {k}</Translated>
          <input type="range" min="1" max="8" value={k} aria-label={tr('Speculation depth', 'Độ sâu suy đoán')} onChange={(ev) => setK(Number(ev.target.value))} />
        </label>
        <label className="field">
          <Translated vi={<>Tỉ lệ chấp nhận mỗi token α = {alpha.toFixed(2)}</>}>Acceptance per token α = {alpha.toFixed(2)}</Translated>
          <input type="range" min="0.05" max="1" step="0.05" value={alpha} aria-label={tr('Acceptance probability', 'Xác suất chấp nhận')} onChange={(ev) => setAlpha(Number(ev.target.value))} />
        </label>
        <button onClick={() => setSeed(seed + 1)}>
          <Translated vi="Chạy lần khác">Another run</Translated>
        </button>
        <label className="check">
          <input type="checkbox" checked={keep} onChange={(ev) => setKeep(ev.target.checked)} />{' '}
          <Translated vi="giữ luôn token của mô hình đích ở chỗ không khớp đầu tiên (của khóa học này)">
            also keep the target’s token at the first mismatch (ours)
          </Translated>
        </label>
      </div>
      <div className="cols plot">
        <div aria-label={tr('Rounds', 'Các vòng')} data-rounds={r.rounds.length}>
          {r.rounds.slice(0, 18).map((rd, i) => (
            <div key={i} style={{ display: 'flex', gap: 3, alignItems: 'center', marginBottom: 3 }}>
              <span className="muted" style={{ fontSize: '0.72rem', width: '3.2rem' }}><Translated vi={<>lượt {i + 1}</>}>pass {i + 1}</Translated></span>
              {Array.from({ length: k }, (_, j) => {
                const ok = j < rd.matched;
                const bad = j === rd.matched;
                return (
                  <span key={j} style={{
                    width: 18, height: 14, borderRadius: 2,
                    background: ok ? 'var(--ok)' : bad ? 'var(--fail-wash)' : 'transparent',
                    border: `1px ${ok || bad ? 'solid' : 'dashed'} ${ok ? 'var(--ok)' : bad ? 'var(--fail)' : 'var(--rule)'}`,
                  }} />
                );
              })}
              <span style={{ fontSize: '0.75rem', marginLeft: 6 }}><Translated vi={<>giữ {rd.kept}</>}>{rd.kept} kept</Translated></span>
            </div>
          ))}
          {r.rounds.length > 18 ? <p className="muted" style={{ margin: 0 }}><Translated vi={<>… thêm {r.rounds.length - 18} lượt</>}>… {r.rounds.length - 18} more passes</Translated></p> : null}
        </div>
        <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-spec="">
          <tbody>
            <tr><td><Translated vi={<>Số lượt của mô hình đích cho {r.tokens} token</>}>Target passes for {r.tokens} tokens</Translated></td><td className="num" data-passes={r.targetPasses}>{r.targetPasses}</td></tr>
            <tr><td><Translated vi="Số token mỗi lượt của mô hình đích, lần chạy này">Tokens per target pass, this run</Translated></td><td className="num" data-per-pass={r.tokensPerPass.toFixed(3)}>{fmt(r.tokensPerPass)}</td></tr>
            <tr><td><Translated vi="Kỳ vọng (công thức đóng)">Expected (closed form)</Translated></td><td className="num" data-expected={e.toFixed(4)}>{fmt(e)}</td></tr>
            <tr><td><Translated vi="Số lần đọc toàn bộ cache cho mỗi token được giữ">Reads of the whole cache per token kept</Translated></td><td className="num">{fmt(r.cacheReadsPerToken)}</td></tr>
            <tr><td><Translated vi="Decode thông thường">Plain decoding</Translated></td><td className="num">1</td></tr>
            <tr><td><Translated vi="Số hàng truy vấn dùng chung mỗi lần đọc">Query rows sharing each read</Translated></td><td className="num">{r.rowsPerRead}</td></tr>
          </tbody>
        </table>
      </div>
      <p className="muted" style={{ marginBottom: 0 }}>
        <Translated
          vi={
            <>
              Số token kỳ vọng mỗi lượt của mô hình đích ở α = {alpha.toFixed(2)} với độ sâu 1 đến 8:{' '}
              {Array.from({ length: 8 }, (_, i) => fmt(expectedPerPass(i + 1, alpha, keep))).join(', ')}.
            </>
          }
        >
          Expected tokens per target pass at α = {alpha.toFixed(2)} for depths 1 to 8:{' '}
          {Array.from({ length: 8 }, (_, i) => fmt(expectedPerPass(i + 1, alpha, keep))).join(', ')}.
        </Translated>
      </p>
    </div>
  );
}

function SpecBench() {
  const [mode, setMode] = useState<'book' | 'own'>('book');
  return (
    <Bench
      title={tr('Speculative decoding', 'Speculative decoding')}
      task={
        mode === 'book'
          ? tr('Step through the book’s figure: draft, verify, accept, twice.', 'Đi từng bước qua hình của sách: phác thảo, kiểm chứng, chấp nhận, hai lần.')
          : tr('Find the acceptance below which deeper speculation stops paying.', 'Tìm tỉ lệ chấp nhận mà dưới mức đó suy đoán sâu hơn không còn đáng giá.')
      }
      foot={
        mode === 'book'
          ? tr(
              'The rounds of Fig. 20.18, replayed from src/sim/specdecode.ts. Each verify is one pass of the target model over all the drafted rows.',
              'Các vòng của Fig. 20.18, phát lại từ src/sim/specdecode.ts. Mỗi lần kiểm chứng là một lượt của mô hình đích trên mọi hàng đã phác thảo.',
            )
          : tr(
              'Green: a draft token the target agreed with. Red: the first mismatch. Dashed: drafted but never reached. Model: src/sim/specdecode.ts.',
              'Xanh lá: một token nháp mà mô hình đích đồng ý. Đỏ: chỗ không khớp đầu tiên. Nét đứt: đã phác thảo nhưng không bao giờ tới. Mô hình: src/sim/specdecode.ts.',
            )
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('Mode', 'Chế độ')}>
          <button aria-pressed={mode === 'book'} onClick={() => setMode('book')}><Translated vi="Hình của sách">The book’s figure</Translated></button>
          <button aria-pressed={mode === 'own'} onClick={() => setMode('own')}><Translated vi="Của riêng bạn">Your own</Translated></button>
        </span>
      </div>
      {mode === 'book' ? <FigureReplay /> : <OwnSpec />}
    </Bench>
  );
}

// ---------------------------------------------------------------- lab 20.4 prediction

function DecodeMeasured({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const found = realBackends(reports).flatMap((b) => {
    const roof = benchRows(reports[b], 'ch05/lab1_roofline').find((x) => x.name === 'roofline');
    const bw = num(roof?.bandwidth_gb_per_s);
    const rows = benchRows(reports[b], 'ch20/lab4_decode').filter((x) => x.name === 'decode' && num(x.gb_per_s));
    return rows.length ? [{ b, device: String(reports[b].device.name ?? b), bw, rows }] : [];
  });
  if (!found.length)
    return (
      <Translated
        vi={
          <p>
            Chưa có số đo nào trên máy này. Chạy <code>python3 labs/pmpp.py run 20.4 --solution --bench</code> rồi quay lại:
            ô này sẽ so sánh dự đoán của bạn với GB/s của lab 20.4 và của lab 5.1.
          </p>
        }
      >
        <p>
          No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 20.4 --solution --bench</code> and come back:
          this box will compare your prediction with lab 20.4’s GB/s and lab 5.1’s.
        </p>
      </Translated>
    );
  return (
    <>
      {found.map((f) => {
        const at = f.rows.find((r) => r.batch === 1 && r.len === 8192) ?? f.rows[f.rows.length - 1];
        const pct = f.bw ? (100 * num(at.gb_per_s)!) / f.bw : null;
        return (
          <div key={f.b}>
            <Translated
              vi={
                <p>
                  Trên <b>{f.device}</b> ({f.b}) lời giải tham chiếu đọc cache fp32 của nó ở <b>{fmt(num(at.gb_per_s)!)} GB/s</b>{' '}
                  (batch {String(at.batch)}, {Number(at.len).toLocaleString('en-US')} token)
                  {pct != null ? (
                    <>
                      : <b>{fmt(pct)}%</b> của {fmt(f.bw!)} GB/s mà lab 5.1 đo được. Bạn đã dự đoán {predicted}%.
                    </>
                  ) : (
                    '.'
                  )}
                </p>
              }
            >
              <p>
                On <b>{f.device}</b> ({f.b}) the reference solution read its fp32 cache at <b>{fmt(num(at.gb_per_s)!)} GB/s</b>{' '}
                (batch {String(at.batch)}, {Number(at.len).toLocaleString('en-US')} tokens)
                {pct != null ? <>: <b>{fmt(pct)}%</b> of the {fmt(f.bw!)} GB/s lab 5.1 measured. You predicted {predicted}%.</> : '.'}
              </p>
            </Translated>
            <table className="data" style={{ margin: '0 0 0.6rem' }}>
              <thead>
                <tr>
                  <th className="num"><Translated vi="batch">batch</Translated></th>
                  <th className="num"><Translated vi="token">tokens</Translated></th>
                  <th className="num">GB/s</th>
                  {f.bw ? <th className="num"><Translated vi="so với lab 5.1">of lab 5.1</Translated></th> : null}
                </tr>
              </thead>
              <tbody>
                {f.rows.map((r, i) => (
                  <tr key={i}>
                    <td className="num">{String(r.batch)}</td>
                    <td className="num">{Number(r.len).toLocaleString('en-US')}</td>
                    <td className="num">{fmt(num(r.gb_per_s)!)}</td>
                    {f.bw ? <td className="num">{fmt((100 * num(r.gb_per_s)!) / f.bw)}%</td> : null}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      })}
      <Translated
        vi={
          <p className="muted">
            Số đo từ máy của bạn, không phải kỳ vọng. Các tác giả Flash-Decoding lưu ý rằng một truy vấn mỗi chuỗi có thể để
            phần lớn GPU nhàn rỗi khi batch nhỏ, đó là lý do các kernel production cũng chia cache ra nhiều khối
            (ch20-facts A8); hãy so sánh các hàng batch 1 và batch 8 của bạn.
          </p>
        }
      >
        <p className="muted">
          Measurements from your machine, not expectations. The Flash-Decoding authors note that one query per sequence can
          leave most of a GPU idle when the batch is small, which is why production kernels also split the cache across
          blocks (ch20-facts A8); compare your batch 1 and batch 8 rows.
        </p>
      </Translated>
    </>
  );
}

// ---------------------------------------------------------------- questions

export const questionsS206: Record<string, Question> = {
  'ch20/20.6/q-linear-ai': {
    kind: 'numeric',
    id: 'ch20/20.6/q-linear-ai',
    prompt: (
      <Translated vi={<>Một lớp tuyến tính có 1,024 đầu vào và 4,096 đầu ra chạy ở 16-bit trên một batch 512 hàng. Đếm trọng số một lần cộng mọi hàng đầu vào và đầu ra, nó làm bao nhiêu FLOP trên mỗi byte?</>}>
        <>A linear layer with 1,024 inputs and 4,096 outputs runs in 16-bit on a batch of 512 rows. Counting the weights once plus every input and output row, how many FLOPs does it do per byte?</>
      </Translated>
    ),
    answer: 315,
    tol: 0.0035,
    unit: 'FLOP/B',
    explain: (
      <Translated
        vi={
          <p>
            FLOP: 512 × 2 × 1,024 × 4,096 = 4,294,967,296. Byte: trọng số, 1,024 × 4,096 × 2 = 8,388,608, cộng 512 × (1,024
            + 4,096) × 2 = 5,242,880 cho các hàng vào và ra. Tỉ số là 315.08, con số 315 của sách. Ở batch 1 cùng cách đếm
            cho 0.9988, &ldquo;khoảng 1&rdquo; của sách.
          </p>
        }
      >
        <p>
          FLOPs: 512 × 2 × 1,024 × 4,096 = 4,294,967,296. Bytes: the weights, 1,024 × 4,096 × 2 = 8,388,608, plus 512 × (1,024
          + 4,096) × 2 = 5,242,880 for the rows in and out. The ratio is 315.08, the book’s 315. At batch 1 the same count
          gives 0.9988, the book’s “about 1”.
        </p>
      </Translated>
    ),
    source: 'ch. 20 §20.6, p. 505; notes/today/ch20-facts.md F1', // i18n-ignore: citation
  },
  'ch20/20.6/q-attn-batch': {
    kind: 'choice',
    id: 'ch20/20.6/q-attn-batch',
    prompt: (
      <Translated vi={<>Vì sao gộp batch 64 yêu cầu decode lại với nhau không nâng cường độ tính toán của attention của chúng?</>}>
        <>Why does batching 64 decode requests together not raise the intensity of their attention?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Mỗi yêu cầu đọc cache riêng của nó, nên số byte tăng cùng số FLOP.">
            Each request reads its own cache, so bytes grow with the FLOPs.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Softmax không thể gộp batch giữa các yêu cầu có độ dài khác nhau.">
            The softmax cannot be batched across requests of different lengths.
          </Translated>
        ),
        why: (
          <Translated vi="Softmax là một phần nhỏ của công việc và gộp batch tốt; vấn đề là thứ được đọc.">
            The softmax is a small part of the work and batches fine; the issue is what is read.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Mặt nạ nhân quả khác nhau giữa các yêu cầu trong batch.">
            The causal mask differs from request to request in the batch.
          </Translated>
        ),
        why: <Translated vi="Một bước decode hoàn toàn không cần mặt nạ.">A decode step needs no mask at all.</Translated>,
      },
      {
        label: (
          <Translated vi="Nó có nâng, gấp 64 lần, giống như với các phép chiếu.">
            It does raise it, 64 times, just as it does for the projections.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là điều xảy ra với các phép chiếu, vốn dùng chung trọng số. Các cache thì không dùng chung.">
            That is what happens to the projections, whose weights are shared. The caches are not.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Cường độ tính toán tăng khi một byte được nạp nuôi nhiều phép tính hơn. Trọng số chung cho mọi yêu cầu; một cache
            thuộc về một yêu cầu, nên 64 yêu cầu đọc 64 cache và không byte nào được dùng chung giữa chúng.
          </p>
        }
      >
        <p>Intensity rises when one loaded byte feeds more arithmetic. Weights are common to every request; a cache belongs to one request, so 64 requests read 64 caches and no byte is shared between them.</p>
      </Translated>
    ),
    source: 'ch. 20 §20.6, p. 505, Fig. 20.17',
  },
  'ch20/20.6/q-gpt3': {
    kind: 'numeric',
    id: 'ch20/20.6/q-gpt3',
    prompt: (
      <Translated vi={<>GPT-3 có 96 lớp và 96 head cỡ 128. Ở 16-bit, KV cache của một chuỗi ở cửa sổ ngữ cảnh thực của nó là 2,048 token lớn bao nhiêu?</>}>
        <>GPT-3 has 96 layers and 96 heads of 128. In 16-bit, how big is the KV cache of one sequence at its real context window of 2,048 tokens?</>
      </Translated>
    ),
    answer: 9,
    tol: 0.02,
    unit: 'GiB',
    explain: (
      <Translated
        vi={
          <p>
            Eq. 20.8: 1 × 2,048 × 2 × 96 × (96 × 128) × 2 byte = 9,663,676,416 byte = 9 GiB. Con số 18 GB của sách dùng 4,096
            token, gấp đôi cửa sổ mà GPT-3 được xây dựng với.
          </p>
        }
      >
        <p>
          Eq. 20.8: 1 × 2,048 × 2 × 96 × (96 × 128) × 2 bytes = 9,663,676,416 bytes = 9 GiB. The book’s 18 GB uses 4,096
          tokens, twice the window GPT-3 was built with.
        </p>
      </Translated>
    ),
    source: 'ch. 20 §20.6, p. 506; notes/today/ch20-facts.md E2, F1', // i18n-ignore: citation
  },
  'ch20/20.6/q-batch-fit': {
    kind: 'numeric',
    id: 'ch20/20.6/q-batch-fit',
    prompt: (
      <Translated vi={<>Sau trọng số, còn 8 GiB bộ nhớ cho cache. Bao nhiêu chuỗi Llama 2 7B dài 4,096 token vừa ở 16-bit?</>}>
        <>After the weights, 8 GiB of memory remain for the cache. How many 4,096-token Llama 2 7B sequences fit in 16-bit?</>
      </Translated>
    ),
    answer: 4,
    explain: (
      <Translated
        vi={
          <p>
            Cache của mỗi chuỗi là 2 GiB (ví dụ của sách: 4,096 × 2 × 32 × 4,096 × 2 byte), và 8 / 2 = 4. Giảm một nửa ngữ
            cảnh thì vừa gấp đôi: tích batch nhân ngữ cảnh mới là thứ bộ nhớ giới hạn.
          </p>
        }
      >
        <p>Each sequence’s cache is 2 GiB (the book’s example: 4,096 × 2 × 32 × 4,096 × 2 bytes), and 8 / 2 = 4. Halve the context and twice as many fit: batch times context is what memory caps.</p>
      </Translated>
    ),
    source: 'ch. 20 §20.6, pp. 505–506',
  },
  'ch20/20.6/q-palm': {
    kind: 'choice',
    id: 'ch20/20.6/q-palm',
    prompt: (
      <Translated vi={<>Hàng &ldquo;PaLM 2&rdquo; của sách (64 lớp, chiều ẩn 8,192) cho cache 8 GB với 4,096 token. Nó sai ở đâu?</>}>
        <>The book’s “PaLM 2” row (64 layers, hidden dimension 8,192) gives a cache of 8 GB for 4,096 tokens. What is wrong with it?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Hình dạng đó là của PaLM 62B, vốn cache một head K/V, không phải 32.">
            The shape is PaLM 62B’s, which caches one K/V head, not 32.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không có gì sai: phép tính của Eq. 20.8 đúng với hình dạng đó.">
            Nothing: the arithmetic of Eq. 20.8 is right for that shape.
          </Translated>
        ),
        why: (
          <Translated vi="Phép tính thì đúng; mô hình mà nó mô tả không tồn tại ở dạng đó.">
            The arithmetic is right; the model it describes does not exist in that form.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Lẽ ra phải là 16 GB, vì cả K và V đều được cache cho mỗi lớp.">
            It should be 16 GB, because K and V are both cached per layer.
          </Translated>
        ),
        why: (
          <Translated vi="Eq. 20.8 đã có thừa số 2 cho K và V.">Eq. 20.8 already has the factor 2 for K and V.</Translated>
        ),
      },
      {
        label: (
          <Translated vi="Độ chính xác lẽ ra phải là 4 byte, vì mô hình được huấn luyện ở fp32.">
            The precision should be 4 bytes, as the model was trained in fp32.
          </Translated>
        ),
        why: (
          <Translated vi="Sách nêu 16-bit, đúng là dạng mà các mô hình như vậy được phục vụ.">
            The book states 16-bit, which is what such models are served in.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Google không công bố kiến trúc của PaLM 2, nên đây không phải các con số của nó. Bài báo PaLM đầu tiên liệt kê
            PaLM 62B với 64 lớp, 32 head và d_model 8,192, và dùng multi-query attention với kích thước head 256. Với một head
            K và một head V cỡ 256 mỗi lớp, 4,096 token chiếm 4,096 × 2 × 64 × 256 × 2 byte = 256 MiB, ít hơn 32 lần con số
            của sách.
          </p>
        }
      >
        <p>
          Google withheld PaLM 2’s architecture, so these are not its numbers. The first PaLM paper lists PaLM 62B with 64 layers, 32 heads and d_model
          8,192, and uses multi-query attention with a head size of 256. With one K and one V head of 256 per layer, 4,096
          tokens take 4,096 × 2 × 64 × 256 × 2 bytes = 256 MiB, 32 times less than the book&rsquo;s figure.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch20-facts.md E2, F1; book p. 506',
    vi: { source: 'notes/today/ch20-facts.md E2, F1; sách p. 506' },
  },
  'ch20/20.6/q-spec-ai': {
    kind: 'choice',
    id: 'ch20/20.6/q-spec-ai',
    prompt: (
      <Translated vi={<>Speculative decoding kiểm chứng k token được phác thảo trong một lượt của mô hình đích. Số byte cache đọc cho mỗi token được kiểm chứng thay đổi thế nào?</>}>
        <>Speculative decoding verifies k drafted tokens in one pass of the target model. What happens to the cache bytes read per verified token?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Chúng giảm khoảng k lần: một lần đọc tiền tố phục vụ cả k hàng.">
            They fall about k times: one read of the prefix serves all k rows.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không thay đổi: mỗi token phác thảo vẫn tự đọc cache.">
            Nothing: each drafted token still reads the cache on its own.
          </Translated>
        ),
        why: (
          <Translated vi="Không cần phải vậy: k hàng dùng chung một tiền tố, nên một lượt có thể nạp mỗi K và V đã cache một lần cho cả k (p. 508). Đọc theo từng token là điều mà việc gộp batch các yêu cầu riêng rẽ bắt buộc.">
            They need not: the k rows share one prefix, so one pass can load each cached K and V once for all k (p. 508).
            Reading per token is what batching separate requests forces.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chúng tăng gấp đôi, vì hai mô hình chạy và mỗi mô hình đọc một cache.">
            They double, because two models run and each reads a cache.
          </Translated>
        ),
        why: (
          <Translated vi="Mô hình nháp nhỏ và có cache nhỏ riêng; cache của mô hình đích được đọc một lần mỗi lượt.">
            The draft model is small and has its own small cache; the target’s cache is read once per pass.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chúng giảm về không, vì các bản nháp đã được tính sẵn.">
            They fall to zero, because the drafts were already computed.
          </Translated>
        ),
        why: (
          <Translated vi="Cache vẫn được đọc, một lần mỗi lượt.">The cache is still read, once per pass.</Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Ý của sách ở p. 508: một batch suy đoán đặc biệt vì mọi phần tử của nó dùng chung một tiền tố, nên cùng K và V
            đã cache phục vụ mọi token đầu ra thay vì được nạp một lần cho mỗi token. Cường độ tính toán của attention tăng
            khoảng k lần.
          </p>
        }
      >
        <p>The book’s point on p. 508: a speculative batch is special because all its entries share a common prefix, so the same cached K and V serve every output token instead of being loaded once per token. Attention&rsquo;s intensity rises about k times.</p>
      </Translated>
    ),
    source: 'ch. 20 §20.6, p. 508; notes/today/ch20-facts.md E4', // i18n-ignore: citation
  },
  'ch20/20.6/q-inflight': {
    kind: 'choice',
    id: 'ch20/20.6/q-inflight',
    prompt: (
      <Translated vi={<>Sách nói in-flight batching đặt một yêu cầu mới vào batch có khối lượng công việc thấp nhất (p. 506). Bài báo mà sách trích dẫn và tài liệu của NVIDIA thực sự mô tả gì?</>}>
        <>The book says in-flight batching slots a new request into the batch with the lowest workload (p. 506). What do the paper it cites and NVIDIA’s documentation actually describe?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Sau mỗi vòng lặp, các yêu cầu đã xong rời đi và các yêu cầu đang chờ tham gia.">
            After every iteration, finished requests leave and waiting ones join.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Mô tả của sách: một yêu cầu mới tham gia batch nào có ít việc nhất.">
            The book’s description: a new request joins whichever batch has the least work.
          </Translated>
        ),
        why: (
          <Translated vi="Không nguồn nào mô tả việc chọn batch theo khối lượng công việc.">
            Neither source describes choosing a batch by workload.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Đệm mọi yêu cầu về một độ dài, để batch giữ dạng hình chữ nhật.">
            Padding every request to one length, so that the batch stays rectangular.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là điều mà lập lịch theo từng vòng lặp tránh.">That is what iteration-level scheduling avoids.</Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chạy prefill và decode trên các GPU riêng, mỗi GPU có batch riêng.">
            Running prefill and decode on separate GPUs, each with its own batch.
          </Translated>
        ),
        why: (
          <Translated vi="TensorRT-LLM nói điều ngược lại: các chuỗi ở giai đoạn ngữ cảnh và giai đoạn sinh được xử lý cùng nhau.">
            TensorRT-LLM says the opposite: context-phase and generation-phase sequences are processed together.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đây là lập lịch theo từng vòng lặp (iteration-level scheduling). Bài báo vLLM ghi công cho Orca: &ldquo;After
            each iteration, completed requests are removed from the batch, and new ones are added.&rdquo; TensorRT-LLM:
            &ldquo;in-flight batching of requests (also known as continuous batching or iteration-level batching)&rdquo;.
          </p>
        }
      >
        <p>
          This is iteration-level scheduling. The vLLM paper credits it to Orca: “After each iteration, completed requests are removed from
          the batch, and new ones are added.” TensorRT-LLM: “in-flight batching of requests (also known as continuous batching
          or iteration-level batching)”.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch20-facts.md E6', // i18n-ignore: citation
  },
};

// ---------------------------------------------------------------- the page

export function S206() {
  return (
    <>
      <Translated
        vi={
          <p>
            Mục 20.4 để lại một bước decode như một chuỗi các phép nhân vectơ-ma trận làm khoảng một phép toán trên mỗi
            byte. Mục này đặt hai câu hỏi về điều đó. Cái gì có thể nâng tỉ số? Và cache tốn bao nhiêu bộ nhớ, khi nhiều yêu
            cầu được phục vụ cùng lúc?
          </p>
        }
      >
        <p>
          Section 20.4 left a decode step as a string of vector-matrix products doing about one operation per byte. This
          section asks two questions about that. What can raise the ratio? And how much memory does the cache cost, once
          many requests are served at the same time?
        </p>
      </Translated>
      <Source section="20.6" pages="504" />

      <h2>
        <Translated vi="Gộp batch các phép chiếu">Batching the projections</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Batching là cách mọi mạng nơ-ron nâng cường độ tính toán của mình: xếp nhiều hàng đầu vào thành một ma trận, để
            mỗi trọng số nạp từ bộ nhớ phục vụ tất cả chúng.
          </p>
        }
      >
        <p>
          Batching is how every neural network raises its intensity: stack several input rows into a matrix, so that each
          weight loaded from memory serves all of them.
        </p>
      </Translated>
      <Source section="20.6" pages="504–505" />
      <Translated
        vi={
          <p>
            Ví dụ của sách là một lớp tuyến tính có 1,024 đầu vào và 4,096 đầu ra ở 16-bit: khoảng 1 FLOP trên mỗi byte với
            một hàng, 315 với 512 hàng. Chú thích của nó nói thêm rằng đây là mức lợi lý thuyết; một kernel đạt được bao
            nhiêu phụ thuộc vào kích thước ô nó dùng.
          </p>
        }
      >
        <p>
          The book&rsquo;s example is a linear layer with 1,024 inputs and 4,096 outputs in 16-bit: about 1 FLOP per byte
          with one row, 315 with 512 rows. Its footnote adds that this is the theoretical gain; what a kernel gets depends
          on the tile size it uses.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Sách không trình bày cách đếm của mình. Trọng số một lần, cộng mọi hàng đầu vào và đầu ra, cho đúng 0.9988 và
              315.08 (facts F1). Chương 22 đưa cùng cách đếm này vào cả một bước decode.
            </p>
          }
        >
          <p>The book does not show its count. Weights once, plus every input and output row, gives 0.9988 and 315.08 exactly (facts F1). Chapter 22 builds the same count into a whole decode step.</p>
        </Translated>
      </Aside>
      <BatchingBench />
      <Ask q={questionsS206['ch20/20.6/q-linear-ai']} label={tr('Work it out', 'Tự tính')} />

      <h2>
        <Translated vi="Vì sao attention không gộp batch được">Why attention does not batch</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Fig. 20.17 vẽ một batch gồm bốn bước decode. Trong các phép chiếu, cả bốn truy vấn dùng chung W_Q, W_K và W_V,
            nên batching nâng cường độ tính toán của các phép chiếu theo kích thước batch. Trong attention, mỗi truy vấn cần
            cache của cuộc hội thoại của riêng nó: bốn truy vấn đọc bốn cache khác nhau, và số byte tăng nhanh như phép tính.
            Khi đó batching bị giới hạn bởi bộ nhớ: trọng số cộng một cache cho mỗi yêu cầu.
          </p>
        }
      >
        <p>
          Fig. 20.17 draws a batch of four decode steps. In the projections, all four queries share the same W_Q, W_K and
          W_V, so batching raises the projections’ intensity by the batch size. In attention, each query needs the cache of
          its own conversation: four queries read four different caches, and the bytes grow as fast as the arithmetic.
          Batching is then limited by memory: the weights plus one cache per request.
        </p>
      </Translated>
      <Source section="20.6" pages="505" figs={tr('Fig. 20.17', 'Hình 20.17')} />
      <Translated vi={<p>Hãy xem cả hai luồng dữ liệu khi batch lớn lên: cường độ tính toán của luồng nào thay đổi, và luồng nào lấp đầy bộ nhớ.</p>}>
        <p>Watch both streams as the batch grows: which one&rsquo;s intensity moves, and which one fills memory.</p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một lớp có hình dạng Llama 2 7B, 16-bit, mọi yêu cầu có cùng cache 1,024 token. Chỉ các lần đọc trọng số và
              cache được đếm: phép chiếu Q, K và V và attention, không có phép chiếu đầu ra, không có feed-forward.
            </p>
          }
        >
          <p>
            One layer of a Llama 2 7B shape, 16-bit, every request with the same 1,024-token cache. Only the weight and cache
            reads are counted: the Q, K and V projection and attention, no output projection, no feed-forward.
          </p>
        </Translated>
      </Simplification>
      <BatchShareScene />
      <Ask q={questionsS206['ch20/20.6/q-attn-batch']} />

      <h2>
        <Translated vi="Cache lớn đến đâu">How big the cache is</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trọng số trước: một mô hình vài tỉ tham số là điển hình, và 7 tỉ ở định dạng 16-bit chiếm khoảng 14 GB. Rồi đến
            cache. Với l lớp, h_q head truy vấn độ rộng d và p byte mỗi phần tử, mỗi token thêm một hàng K và một hàng V vào
            mọi head của mọi lớp:
          </p>
        }
      >
        <p>
          Weights first: a model of several billion parameters is typical, and 7 billion in a 16-bit format take about 14 GB.
          Then the cache. With l layers, h_q query heads of width d and p bytes per element, every token adds a K row and a
          V row to every head of every layer:
        </p>
      </Translated>
      <Source section="20.6" pages="505–506" />
      <p style={{ textAlign: 'center' }}>
        <Translated
          vi={
            <>
              mỗi token (Eq. 20.7): 2 × l × h_q × d × p &nbsp;&nbsp;·&nbsp;&nbsp; b chuỗi gồm N token (Eq. 20.8): b × N × 2 ×
              l × h_q × d × p
            </>
          }
        >
          per token (Eq. 20.7): 2 × l × h_q × d × p &nbsp;&nbsp;·&nbsp;&nbsp; b sequences of N tokens (Eq. 20.8): b × N × 2 × l × h_q × d × p
        </Translated>
      </p>
      <Translated
        vi={
          <p>
            Sách làm ba ví dụ, mỗi ví dụ một chuỗi 16-bit dài 4,096 token: Llama 2 7B cần 2 GB, GPT-3 18 GB và &ldquo;PaLM
            2&rdquo; 8 GB. Phép tính đúng ở cả ba. Hai trong số các đầu vào thì không.
          </p>
        }
      >
        <p>
          The book works three examples, each one 16-bit sequence of 4,096 tokens: Llama 2 7B needs 2 GB, GPT-3 18 GB and
          “PaLM 2” 8 GB. The arithmetic is right in all three. Two of the inputs are not.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch20-facts E2, F1)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              Llama 2 7B: 2 GiB, đã xác nhận. Cửa sổ ngữ cảnh của GPT-3 là 2,048 token, nên 9 GiB, không phải 18. Hình dạng
              của PaLM 2 không được công bố; hàng đó là PaLM 62B, vốn dùng multi-query attention với một head K/V cỡ 256: 256
              MiB, không phải 8 GB. Đơn vị GB của sách là nhị phân (GiB).
            </p>
          }
        >
          <p>Llama 2 7B: 2 GiB, confirmed. GPT-3’s context window is 2,048 tokens, so 9 GiB, not 18. PaLM 2’s shape was withheld; the row is PaLM 62B, which uses multi-query attention with one K/V head of 256: 256 MiB, not 8 GB. The book’s GB are binary (GiB).</p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Kết luận của sách vẫn đứng vững sau các chỉnh sửa: gộp batch nhiều yêu cầu thì các cache nhanh chóng vượt quá bộ
            nhớ của một GPU, đó là lý do việc phục vụ LLM thường trải trên nhiều GPU. Hãy tự tính kích thước, so với bộ nhớ
            mà thiết bị của chính bạn báo cáo.
          </p>
        }
      >
        <p>
          The book’s conclusion survives the corrections: batch many requests and the caches quickly outgrow one GPU’s
          memory, which is why LLM serving commonly spans several GPUs. Size it for yourself, against the memory your own
          device reported.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Hàng cuối là của khóa học này, không phải của sách: một cận dưới cho thời gian của một bước decode, nếu mọi trọng
            số và mọi byte đã cache phải được đọc một lần ở băng thông mà lab 5.1 đo được. Nó trả lời &ldquo;decode có thể
            nhanh đến đâu ở đây?&rdquo; bằng một phép chia.
          </p>
        }
      >
        <p>
          The last row is ours, not the book&rsquo;s: a floor on the time of one decode step, if every weight and every
          cached byte had to be read once at the bandwidth lab 5.1 measured. It answers &ldquo;how fast could decode
          possibly go here?&rdquo; with one division.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Cận dưới chỉ đếm các lần đọc, giả định mọi byte chảy qua một lần ở băng thông của lab 5.1, và bỏ qua bộ nhớ
              đệm, chồng lấp, các lần khởi chạy kernel và mọi thứ ngoài attention và trọng số.
            </p>
          }
        >
          <p>The floor counts reads only, assumes every byte streams once at lab 5.1’s bandwidth, and ignores caches, overlap, kernel launches and everything outside attention and the weights.</p>
        </Translated>
      </Simplification>
      <KvExplorer mode="size" />
      <Ask q={questionsS206['ch20/20.6/q-gpt3']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS206['ch20/20.6/q-batch-fit']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS206['ch20/20.6/q-palm']} />

      <h2>
        <Translated vi="Các yêu cầu có độ dài khác nhau">Requests of different lengths</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Phân tích đến giờ cho mọi yêu cầu cùng độ dài. Các yêu cầu thật khác nhau cả về độ dài đầu vào lẫn đầu ra, và
            sách lưu ý rằng điều này có thể làm mất cân bằng công việc và để GPU bị dùng chưa hết. Sách nêu cách chữa mà các
            hệ thống phục vụ dùng, in-flight batching, và mô tả nó là đặt một yêu cầu mới vào batch có khối lượng công việc
            thấp nhất ngay khi một yêu cầu trước đó được phục vụ xong.
          </p>
        }
      >
        <p>
          The analysis so far gave every request the same length. Real requests differ in both input and output length, and
          the book notes that this can unbalance the work and leave the GPU underused. It names the remedy serving systems
          use, in-flight batching, and describes it as slotting a new request into the batch with the lowest workload as soon
          as an earlier one is served.
        </p>
      </Translated>
      <Source section="20.6" pages="506" />
      <Translated
        vi={
          <p>
            Nguồn mà sách trích dẫn, và tài liệu của NVIDIA, mô tả một điều đơn giản hơn. Sau mỗi vòng lặp, các yêu cầu đã
            xong rời batch đang chạy và các yêu cầu đang chờ tham gia, nên batch được tạo lại ở mỗi bước decode thay vì một
            lần cho mỗi batch yêu cầu. Lời phàn nàn của cùng bài báo đó về bộ nhớ, rằng cache bị lãng phí do phân mảnh giới
            hạn kích thước batch, là điều mà paged cache ở mục 20.7 giải quyết.
          </p>
        }
      >
        <p>
          The source the book cites, and NVIDIA’s documentation, describe something simpler. After every iteration, finished
          requests leave the running batch and waiting ones join it, so the batch is re-formed at every decode step rather
          than once per batch of requests. The same paper’s complaint about memory, that cache wasted by fragmentation
          limits the batch size, is what section 20.7’s paged cache answers.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch20-facts E6)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              Bài báo vLLM ghi công việc lập lịch theo từng vòng lặp này cho Orca. TensorRT-LLM gọi nó là &ldquo;in-flight
              batching … (also known as continuous batching or iteration-level batching)&rdquo;. Không nguồn nào đặt yêu cầu
              theo khối lượng công việc thấp nhất.
            </p>
          }
        >
          <p>The vLLM paper credits this iteration-level scheduling to Orca. TensorRT-LLM calls it “in-flight batching … (also known as continuous batching or iteration-level batching)”. Neither slots requests by lowest workload.</p>
        </Translated>
      </Today>
      <Ask q={questionsS206['ch20/20.6/q-inflight']} />
      <Translated
        vi={
          <p>
            Chương này chỉ lập luận theo cường độ tính toán và bộ nhớ. Nó không nói gì về độ trễ mà mỗi người dùng phải trả
            cho thông lượng mà một batch lớn hơn mua được. <a href={href('ch22', '22.5')}>Mục 22.5</a> và mốc dự án M8 xem
            xét điều đó trên thiết bị của chính bạn.
          </p>
        }
      >
        <p>
          The chapter argues only in terms of intensity and memory. It says nothing about the latency each user pays for the
          throughput a bigger batch buys. <a href={href('ch22', '22.5')}>Section 22.5</a> and milestone M8 take that up on your
          own device.
        </p>
      </Translated>

      <h2>
        <Translated vi="Cường độ tính toán của decode attention">The intensity of decode attention</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Eq. 20.9 đếm một lớp decode attention có cache: nó đọc 2 × h_q × d × N phần tử đã cache, cộng Q′ và O′, và làm 2
            × N × h_q × d phép toán cho (Q′Kᵀ)V. Đó là N / (1 + N), khoảng một phép toán mỗi phần tử, và batching không thể
            thay đổi điều đó.
          </p>
        }
      >
        <p>
          Eq. 20.9 counts one layer of decode attention with the cache: it reads 2 × h_q × d × N cached elements, plus Q′ and
          O′, and does 2 × N × h_q × d operations for (Q′Kᵀ)V. That is N / (1 + N), about one operation per element, and
          batching cannot change it.
        </p>
      </Translated>
      <Source section="20.6" pages="506–507" figs={tr('Eq. 20.9', 'Pt. 20.9')} />
      <Translated
        vi={
          <p>
            Sách đếm một phép nhân-cộng là một phép toán và một phần tử là một byte (p = 1). Tính bằng FLOP trên mỗi byte,
            đó là 1 ở 16-bit và 0.5 ở fp32 mà các lab dùng.
          </p>
        }
      >
        <p>
          The book counts a multiply-add as one operation and an element as one byte (p = 1). In FLOP per byte, that is 1 in
          16-bit and 0.5 in the fp32 the labs use.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch20-facts E4)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              Nguồn của sách, [13], nêu nó theo byte: &ldquo;Every loaded BF16 (2 bytes) element of the cached key matrix
              performs one MAC (2 FLOPs) … yielding a 1:1 FLOP-to-byte ratio.&rdquo;
            </p>
          }
        >
          <p>The book’s source, [13], states it per byte: “Every loaded BF16 (2 bytes) element of the cached key matrix performs one MAC (2 FLOPs) … yielding a 1:1 FLOP-to-byte ratio.”</p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Vậy tốc độ của một kernel decode được quyết định bởi tốc độ nó đọc cache. Lab 20.4 yêu cầu bạn viết một kernel
            như vậy, và benchmark của nó báo GB/s đạt được. Hãy cam kết một con số trước.
          </p>
        }
      >
        <p>
          So a decode kernel’s speed is set by how fast it can read the cache. Lab 20.4 asks you to write one, and its
          benchmark reports the GB/s it reached. Commit to a number first.
        </p>
      </Translated>
      <Predict
        id="ch20/20.6/p-decode-bw"
        prompt={
          <Translated
            vi={
              <>
                Kernel decode của lab 20.4 đọc một cache fp32 gồm 32 head cỡ 128 cho một chuỗi 8,192 token. Bạn kỳ vọng nó
                đạt bao nhiêu phần trăm băng thông mà lab 5.1 đo được trên thiết bị của bạn?
              </>
            }
          >
            <>
              Lab 20.4’s decode kernel reads an fp32 cache of 32 heads of 128 for one sequence of 8,192 tokens. What
              percentage of the bandwidth lab 5.1 measured on your device do you expect it to reach?
            </>
          </Translated>
        }
        unit="%"
      >
        {(v) => <DecodeMeasured predicted={v} />}
      </Predict>

      <h2>
        <Translated vi="Speculative decoding">Speculative decoding</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Speculative decoding tấn công cùng tỉ số đó từ một phía khác. Một mô hình nháp nhỏ đoán k token tiếp theo của một
            mô hình đích lớn (k là độ sâu suy đoán, 3 trong hình của sách). Mô hình đích, chậm hơn nhiều trên mỗi token, kiểm
            tra cả k dự đoán trong một lượt, thực thi song song như một batch. Các token đến trước chỗ bất đồng đầu tiên được
            chấp nhận, và vòng tiếp theo bắt đầu sau chúng.
          </p>
        }
      >
        <p>
          Speculative decoding attacks the same ratio from another side. A small draft model guesses the next k tokens of a
          large target model (k is the speculation depth, 3 in the book’s figure). The target, much slower per token, checks
          all k guesses in one pass, executed in parallel like a batch. Tokens up to the first disagreement are accepted, and
          the next round starts after them.
        </p>
      </Translated>
      <Source section="20.6" pages="507–508" figs={tr('Fig. 20.18', 'Hình 20.18')} />
      <Translated
        vi={
          <p>
            Sách đưa ra hai lý do nó có ích. Giống một batch, nó cho mỗi trọng số phép chiếu nhiều hàng để phục vụ. Và nó là
            một batch đặc biệt: mọi hàng của nó dùng chung một tiền tố, nên K và V đã cache của tiền tố đó được nạp một lần và
            được mọi hàng dùng. Điều đó nâng cường độ tính toán của attention, thứ mà việc gộp batch các yêu cầu riêng rẽ
            không làm được.
          </p>
        }
      >
        <p>
          The book gives two reasons it helps. Like a batch, it gives every projection weight several rows to serve. And it
          is a special batch: all its rows share one prefix, so the cached K and V of that prefix are loaded once and used by
          every row. That raises attention’s intensity, which batching separate requests cannot do.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch20-facts E4)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              [13], Fig. 3: với độ dài truy vấn 2, như trong speculative decoding, h_q = 128 đã &ldquo;climbs beyond the
              roof&rdquo; trên thiết bị ví dụ của họ. Các con số H100 của họ là của họ, không phải của bạn.
            </p>
          }
        >
          <p>[13], Fig. 3: with a query length of 2, as in speculative decoding, h_q = 128 already “climbs beyond the roof” on their example device. Their H100 figures are theirs, not yours.</p>
        </Translated>
      </Today>
      <Translated
        vi={<p>Hãy đi từng bước qua hình, rồi chuyển sang các lần chạy của riêng bạn và tìm chỗ suy đoán sâu hơn không còn đáng giá.</p>}
      >
        <p>
          Step through the figure, then switch to your own runs and look for where deeper speculation stops paying.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi token nháp được chấp nhận độc lập với xác suất α, điều mà các mô hình thật không làm. Mô hình nháp không tốn
              gì ở đây. Quy tắc mặc định là của hình: một vòng chỉ giữ các token nháp khớp. Ô đánh dấu giữ luôn token của
              chính mô hình đích ở chỗ không khớp đầu tiên, thứ mà lượt kiểm chứng đã tính sẵn; biến thể đó là của khóa học
              này.
            </p>
          }
        >
          <p>Each draft token is accepted independently with probability α, which real models do not do. The draft model costs nothing here. The default rule is the figure’s: a round keeps only the matching draft tokens. The checkbox keeps the target’s own token at the first mismatch too, which the verify pass has already computed; that variant is ours.</p>
        </Translated>
      </Simplification>
      <SpecBench />
      <Translated
        vi={
          <p>
            Theo quy tắc của hình, một vòng có thể không giữ được gì, nên suy đoán cần α trên một nửa để thắng decode thông
            thường ngay cả ở độ sâu lớn, và cao hơn nữa ở độ sâu nhỏ. Giữ token của mô hình đích làm mỗi lượt đáng ít nhất
            một token. Dù theo cách nào, mỗi lượt của mô hình đích đọc cache một lần cho k hàng truy vấn.
          </p>
        }
      >
        <p>
          Under the figure’s rule a round can keep nothing at all, so speculation needs α above one half to beat plain
          decoding even at great depth, and higher still at small depths. Keeping the target’s token makes every pass worth
          at least one token. Either way, each target pass reads the cache once for k query rows.
        </p>
      </Translated>
      <Ask q={questionsS206['ch20/20.6/q-spec-ai']} />
    </>
  );
}
