import { useMemo, useState, type KeyboardEvent } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Beyond, Bench, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { benchRows, fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  MAT_NAMES,
  aiPoints,
  baselineStep,
  cacheBytes,
  cachedStep,
  diffStep,
  generate,
  mustCompute,
  scoreMarks,
  type CellKind,
  type MatName,
} from '../../sim/kvcache';
import { DecodeStreamScene } from './DecodeStreamScene';
import { fmtBytes } from './KvExplorer';

// ---------------------------------------------------------------- formatting

const minus = (s: string) => s.replace(/^-/, '−');
function cellText(name: MatName, v: number): string {
  if (v === -Infinity) return '−∞';
  if (Math.abs(v) < 0.005) return '0';
  if (name === 'P') return v.toFixed(2).replace(/^0/, '');
  if (name === 'S' || name === 'O') return Math.abs(v) < 0.05 ? '0' : minus(v.toFixed(1));
  return minus(String(Number(v.toFixed(2))));
}
function fmtFlops(f: number): string {
  const units: [number, string][] = [[1e18, 'EFLOP'], [1e15, 'PFLOP'], [1e12, 'TFLOP'], [1e9, 'GFLOP'], [1e6, 'MFLOP'], [1e3, 'kFLOP']];
  for (const [s, u] of units) if (f >= s) return `${fmt(f / s)} ${u}`;
  return `${fmt(f)} FLOP`; // i18n-ignore: number and unit
}

// ---------------------------------------------------------------- which cells change

const N0 = 6;
const D0 = 4;
const TITLE: Record<MatName, string> = {
  get X() {
    return tr('X (embeddings)', 'X (embedding)');
  },
  Q: 'Q = X W_Q',
  K: 'K = X W_K',
  V: 'V = X W_V',
  S: 'S = Q Kᵀ/√d + M',
  P: 'P = softmax(S)',
  O: 'O = P V',
};
const empty = (): Record<MatName, Set<number>> => ({ X: new Set(), Q: new Set(), K: new Set(), V: new Set(), S: new Set(), P: new Set(), O: new Set() });

function CellsBench() {
  const [seed, setSeed] = useState(1);
  const diff = useMemo(() => diffStep(N0, D0, seed), [seed]);
  const [marks, setMarks] = useState(empty);
  const [revealed, setRevealed] = useState(false);
  const score = useMemo(() => scoreMarks(diff, marks), [diff, marks]);
  const need = useMemo(() => mustCompute(diff), [diff]);
  const total = MAT_NAMES.reduce((s, m) => s + need[m].size, 0);

  const toggle = (name: MatName, i: number) => {
    if (revealed) return;
    setMarks((m) => {
      const next = { ...m, [name]: new Set(m[name]) };
      if (next[name].has(i)) next[name].delete(i);
      else next[name].add(i);
      return next;
    });
  };
  const onKey = (e: KeyboardEvent, name: MatName, i: number) => {
    if (e.key === ' ' || e.key === 'Enter') {
      e.preventDefault();
      toggle(name, i);
    }
  };
  const marked = MAT_NAMES.reduce((s, m) => s + marks[m].size, 0);

  const grid = (name: MatName) => {
    const a = diff.after[name];
    const b = diff.before[name];
    return (
      <div key={name} data-mat={name}>
        <div className="mx-label t-global">{TITLE[name]}</div>
        <div className="mx global" style={{ gridTemplateColumns: `repeat(${a.cols}, auto)` }}>
          {diff.kind[name].map((k: CellKind, i) => {
            const r = Math.floor(i / a.cols);
            const c = i % a.cols;
            const old = r < b.rows && c < b.cols;
            const isMarked = marks[name].has(i);
            let cls = '';
            let strike = false;
            let text = '';
            if (!revealed) {
              cls = isMarked ? 'on' : old ? '' : 'void';
              text = old ? cellText(name, b.v[r * b.cols + c]) : '';
            } else {
              text = cellText(name, a.v[i]);
              if (k === 'new' || k === 'changed') cls = isMarked ? 'good' : 'bad';
              else if (k === 'masked') cls = 'void';
              else if (isMarked) {
                cls = 'bad';
                strike = true;
              }
            }
            return (
              <span key={i} className={cls} role="checkbox" aria-checked={isMarked} tabIndex={revealed ? -1 : 0}
                aria-label={tr(`${name} row ${r} column ${c}`, `${name} hàng ${r} cột ${c}`)} data-kind={revealed ? k : undefined} data-marked={isMarked ? '' : undefined}
                onClick={() => toggle(name, i)} onKeyDown={(e) => onKey(e, name, i)}
                style={{ minWidth: '2.2rem', height: '1.7rem', fontSize: '0.7rem', cursor: revealed ? 'default' : 'pointer', textDecoration: strike ? 'line-through' : undefined }}>
                {text}
              </span>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <Bench
      title={tr('Which cells change?', 'Những ô nào thay đổi?')}
      task={tr(
        `Token ${N0 + 1} arrives. Mark every cell that needs new arithmetic, then reveal.`,
        `Token ${N0 + 1} đến. Đánh dấu mọi ô cần tính toán mới, rồi xem đáp án.`,
      )}
      foot={
        <Translated
          vi={
            <>
              Các giá trị được làm tròn để hiển thị; phần đáp án so sánh chúng khi chưa làm tròn. Các ô nét đứt trước khi
              xem đáp án là những ô mà token mới thêm vào. Sau đó: xanh lá, một ô cần tính và bạn đã đánh dấu; đỏ, một ô bạn
              bỏ sót; đỏ và gạch ngang, một ô bạn đánh dấu nhưng không thay đổi; nét đứt, một ô bị che mà giá trị đã biết
              không cần tính toán.
            </>
          }
        >
          <>
            Values are rounded for display; the reveal compares them unrounded. Dashed cells before the reveal are the slots
            the new token adds. After it: green, a cell that needs computing and you marked; red, one you missed; red and
            struck through, a cell you marked that did not change; dashed, a masked cell whose value is known without arithmetic.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ alignItems: 'flex-start', gap: '1rem', marginBottom: '0.9rem' }}>
        {(['X', 'Q', 'K', 'V'] as MatName[]).map(grid)}
      </div>
      <div className="row" style={{ alignItems: 'flex-start', gap: '1rem', marginBottom: '0.9rem' }}>
        {(['S', 'P', 'O'] as MatName[]).map(grid)}
      </div>
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <button className="primary" onClick={() => setRevealed(true)} disabled={revealed}>
          <Translated vi="Xem đáp án">Reveal</Translated>
        </button>
        <button onClick={() => { setMarks(empty()); setRevealed(false); }}>
          <Translated vi="Xóa">Clear</Translated>
        </button>
        <button onClick={() => { setSeed((s) => s + 1); setMarks(empty()); setRevealed(false); }}>
          <Translated vi="Bài khác">Another problem</Translated>
        </button>
        <span className="muted" aria-live="polite">
          <Translated vi={<>đã đánh dấu {marked}</>}>{marked} marked</Translated>
        </span>
      </div>
      {revealed ? (
        <div className="narrate" role="status" data-score={`${score.hit}/${score.needed}/${score.missed}/${score.extraSame}/${score.extraMasked}`}>
          {score.missed === 0 && score.extraSame === 0 && score.extraMasked === 0 ? (
            <b className="t-ok">
              <Translated vi={<>Đúng chính xác cả {total} ô. </>}>All {total} cells exactly. </Translated>
            </b>
          ) : (
            <Translated
              vi={
                <>
                  Bạn đã đánh dấu {score.hit} trên {total} ô cần tính toán mới
                  {score.missed ? `, và bỏ sót ${score.missed}` : ''}.{' '}
                </>
              }
            >
              <>You marked {score.hit} of the {total} cells that need new arithmetic{score.missed ? `, and missed ${score.missed}` : ''}. </>
            </Translated>
          )}
          {score.extraSame ? (
            <Translated vi={<>Bạn đã đánh dấu {score.extraSame} ô hoàn toàn không thay đổi: kể cả ở bit cuối cùng. </>}>
              <>You marked {score.extraSame} {score.extraSame === 1 ? 'cell' : 'cells'} that did not change at all: not even in the last bit. </>
            </Translated>
          ) : null}
          {score.extraMasked ? (
            <Translated
              vi={
                <>
                  Bạn đã đánh dấu {score.extraMasked} ô bị che của cột mới, như Fig. 20.5 tô sẫm chúng: chúng tồn tại trong
                  ma trận lớn hơn, nhưng mỗi ô là −∞ trong S và 0 trong P, đã biết mà không cần tính toán.{' '}
                </>
              }
            >
              <>You marked {score.extraMasked} masked {score.extraMasked === 1 ? 'cell' : 'cells'} of the new column, as Fig. 20.5 darkens them: they exist in the bigger matrix, but each is −∞ in S and 0 in P, known without arithmetic. </>
            </Translated>
          ) : null}
          <Translated
            vi={
              <>
                Phần việc mới là một hàng của mỗi ma trận: mỗi ma trận X, Q, K, V và O {D0} giá trị, và S và P mỗi ma trận{' '}
                {N0 + 1} giá trị.
              </>
            }
          >
            The new work is one row of each matrix: {D0} values each for X, Q, K, V and O, and {N0 + 1} each for S and P.
          </Translated>
        </div>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- recompute or cache

const WIDTHS = [512, 1024, 2048, 4096, 5120, 8192, 12288];

function PerTokenPlot({ D, layers, at }: { D: number; layers: number; at: number }) {
  const W = 560;
  const H = 230;
  const pad = { l: 64, r: 14, t: 12, b: 36 };
  const nMax = 32768;
  const yMin = cachedStep(1, D, layers).total / 2;
  const yMax = baselineStep(nMax, D, layers).total * 2;
  const X = (n: number) => pad.l + (Math.log10(n) / Math.log10(nMax)) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * (H - pad.t - pad.b);
  const ns = Array.from({ length: 61 }, (_, i) => Math.round(nMax ** (i / 60)));
  const line = (f: (n: number) => number) => ns.map((n) => `${X(n).toFixed(1)},${Y(f(n)).toFixed(1)}`).join(' ');
  const yt: number[] = [];
  for (let e = Math.ceil(Math.log10(yMin)); e <= Math.floor(Math.log10(yMax)); e += 2) yt.push(10 ** e);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('FLOPs per generated token against context length', 'FLOP cho mỗi token sinh ra theo độ dài ngữ cảnh')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {[1, 10, 100, 1000, 10000].map((n) => (
        <g key={n}>
          <line x1={X(n)} x2={X(n)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(n)} y={H - pad.b + 14} fontSize="10.5" textAnchor="middle">{n.toLocaleString('en-US')}</text>
        </g>
      ))}
      {yt.map((v) => (
        <g key={v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 6} y={Y(v) + 4} fontSize="10" textAnchor="end">10<tspan dy="-5" fontSize="7.5">{Math.round(Math.log10(v))}</tspan></text>
        </g>
      ))}
      <text x={(W + pad.l) / 2} y={H - 4} fontSize="11" textAnchor="middle">
        <Translated vi="số token trong ngữ cảnh, n">tokens in the context, n</Translated>
      </text>
      <polyline points={line((n) => baselineStep(n, D, layers).total)} fill="none" stroke="var(--fail)" strokeWidth="2" />
      <polyline points={line((n) => cachedStep(n, D, layers).total)} fill="none" stroke="var(--ok)" strokeWidth="2" />
      <text x={X(3)} y={Y(baselineStep(3, D, layers).total) - 10} fontSize="11" style={{ fill: 'var(--fail)' }}>
        <Translated vi="tính lại mọi thứ">recompute everything</Translated>
      </text>
      <text x={X(200)} y={Y(cachedStep(200, D, layers).total) - 8} fontSize="11" style={{ fill: 'var(--ok)' }}>
        <Translated vi="với KV cache">with a KV cache</Translated>
      </text>
      <line x1={X(at)} x2={X(at)} y1={pad.t} y2={H - pad.b} stroke="var(--ink)" strokeDasharray="3 3" />
    </svg>
  );
}

function RecomputeBench() {
  const [logN, setLogN] = useState(11);
  const [D, setD] = useState(4096);
  const [layers, setLayers] = useState(32);
  const [logT, setLogT] = useState(8);
  const [p, setP] = useState(2);
  const N = Math.round(2 ** logN);
  const T = Math.round(2 ** logT);
  const n = N + 1;
  const base = baselineStep(n, D, layers);
  const cach = cachedStep(n, D, layers);
  const gen = generate(N, T, D, layers);
  const held = cacheBytes(N + T, D, layers, p);
  const row = (label: string, a: number, b: number) => (
    <tr key={label}>
      <td>{label}</td>
      <td className="num">{fmtFlops(a)}</td>
      <td className="num">{fmtFlops(b)}</td>
    </tr>
  );
  return (
    <Bench
      title={tr('Recompute or cache', 'Tính lại hay dùng cache')}
      task={
        <Translated vi="Di chuyển độ dài ngữ cảnh và xem chi phí nào tăng theo nó, và tăng nhanh đến đâu.">
          Move the context length and watch which costs grow with it, and how fast.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              FLOP đếm một phép nhân-cộng là hai, cho mỗi token sinh ra, cộng dồn qua các lớp. D là độ rộng của các hàng Q,
              K và V của một token trên mọi head (chiều ẩn h_q × d của sách, p. 505). Mô hình: <code>src/sim/kvcache.ts</code>.
            </>
          }
        >
          <>
            FLOPs count a multiply-add as two, per generated token, summed over the layers. D is the width of a token’s Q, K
            and V rows over all heads (the book’s hidden dimension h_q × d, p. 505). Model: <code>src/sim/kvcache.ts</code>.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          <Translated vi={<>Số token đã có trong ngữ cảnh, N = {N.toLocaleString('en-US')}</>}>
            Tokens already in the context, N = {N.toLocaleString('en-US')}
          </Translated>
          <input type="range" min="0" max="15" step="0.25" value={logN} aria-label={tr('Context length', 'Độ dài ngữ cảnh')} onChange={(e) => setLogN(Number(e.target.value))} />
        </label>
        <label className="field">
          <Translated vi="Độ rộng D">Width D</Translated>
          <select value={D} onChange={(e) => setD(Number(e.target.value))} aria-label={tr('Width', 'Độ rộng')}>
            {WIDTHS.map((w) => <option key={w} value={w}>{w.toLocaleString('en-US')}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi={<>Số lớp = {layers}</>}>Layers = {layers}</Translated>
          <input type="range" min="1" max="96" value={layers} aria-label={tr('Layers', 'Số lớp')} onChange={(e) => setLayers(Number(e.target.value))} />
        </label>
        <label className="field">
          <Translated vi={<>Sinh T = {T.toLocaleString('en-US')} token</>}>Generate T = {T.toLocaleString('en-US')} tokens</Translated>
          <input type="range" min="0" max="13" step="1" value={logT} aria-label={tr('Tokens to generate', 'Số token cần sinh')} onChange={(e) => setLogT(Number(e.target.value))} />
        </label>
        <label className="field">
          <Translated vi="Phần tử cache">Cache element</Translated>
          <select value={p} onChange={(e) => setP(Number(e.target.value))} aria-label={tr('Cache element', 'Phần tử cache')}>
            <option value={4}>{tr('fp32, 4 bytes', 'fp32, 4 byte')}</option>
            <option value={2}>{tr('16-bit, 2 bytes', '16-bit, 2 byte')}</option>
          </select>
        </label>
      </div>
      <div className="cols plot">
        <PerTokenPlot D={D} layers={layers} at={n} />
        <div>
          <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-recompute="">
            <thead>
              <tr>
                <th>
                  <Translated vi={<>Token tiếp theo (n = {n.toLocaleString('en-US')})</>}>
                    The next token (n = {n.toLocaleString('en-US')})
                  </Translated>
                </th>
                <th className="num">
                  <Translated vi="tính lại">recompute</Translated>
                </th>
                <th className="num">
                  <Translated vi="dùng cache">cached</Translated>
                </th>
              </tr>
            </thead>
            <tbody>
              {row('X W_Q, X W_K, X W_V', base.projections, cach.projections)}
              {row('Q Kᵀ', base.scores, cach.scores)}
              {row('P V', base.values, cach.values)}
              <tr>
                <td>
                  <b>
                    <Translated vi="mỗi token">per token</Translated>
                  </b>
                </td>
                <td className="num" data-base={base.total}><b>{fmtFlops(base.total)}</b></td>
                <td className="num" data-cached={cach.total}><b>{fmtFlops(cach.total)}</b></td>
              </tr>
              <tr>
                <td>
                  <Translated vi={<>cả {T.toLocaleString('en-US')} token</>}>all {T.toLocaleString('en-US')} tokens</Translated>
                </td>
                <td className="num">{fmtFlops(gen.baseline)}</td>
                <td className="num">{fmtFlops(gen.cached)}</td>
              </tr>
            </tbody>
          </table>
          <p style={{ margin: '0.7rem 0 0' }} data-ratio={base.total / cach.total}>
            <Translated
              vi={
                <>
                  Tính lại tốn gấp <b>{fmt(base.total / cach.total)}×</b> bước dùng cache cho token này,{' '}
                  <b>{fmt(gen.baseline / gen.cached)}×</b> trên cả {T.toLocaleString('en-US')} token. Cái giá: sau chúng các
                  cache chứa <b data-held={held}>{fmtBytes(held)}</b> ({(N + T).toLocaleString('en-US')} token × {layers}{' '}
                  lớp × K và V × {D.toLocaleString('en-US')} × {p} byte).
                </>
              }
            >
              Recomputing costs <b>{fmt(base.total / cach.total)}×</b> the cached step for this token,{' '}
              <b>{fmt(gen.baseline / gen.cached)}×</b> over all {T.toLocaleString('en-US')}. The price: after them the caches
              hold <b data-held={held}>{fmtBytes(held)}</b> ({(N + T).toLocaleString('en-US')} tokens × {layers} layers × K and V × {D.toLocaleString('en-US')} × {p} bytes).
            </Translated>
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- roofline

interface Hardware {
  key: string;
  label: string;
  peak: number; // GFLOP/s
  bw: number; // GB/s
  note: string;
}
const BOOK: Hardware = {
  key: 'book',
  get label() {
    return tr('The book’s example (66.9 TFLOP/s, 3.35 TB/s)', 'Ví dụ của sách (66.9 TFLOP/s, 3.35 TB/s)');
  },
  peak: 66900,
  bw: 3350,
  get note() {
    return tr(
      'The figures the book quotes in chapter 5 (p. 94), shown as the book’s example only.',
      'Các con số sách đưa ra ở chương 5 (p. 94), chỉ hiển thị như ví dụ của sách.',
    );
  },
};
function measured(reports: Reports): Hardware[] {
  return realBackends(reports).flatMap((b) => {
    const r = benchRows(reports[b], 'ch05/lab1_roofline').find((x) => x.name === 'roofline');
    const peak = num(r?.compute_gflops);
    const bw = num(r?.bandwidth_gb_per_s);
    if (!peak || !bw) return [];
    // Getters, because the list is memoised on the reports alone and the text must follow the language.
    return [
      {
        key: b,
        get label() {
          return tr(`Yours: ${String(reports[b].device.name ?? b)} (${b})`, `Của bạn: ${String(reports[b].device.name ?? b)} (${b})`);
        },
        peak,
        bw,
        get note() {
          return tr(
            `Lab 5.1’s two probe kernels on ${b}, measured on ${reports[b].generated.slice(0, 10)}.`,
            `Hai kernel thăm dò của lab 5.1 trên ${b}, đo ngày ${reports[b].generated.slice(0, 10)}.`,
          );
        },
      },
    ];
  });
}

function RooflinePlot({ hw, points }: { hw: Hardware; points: { label: string; ai: number; prefill: boolean }[] }) {
  const W = 620;
  const H = 330;
  const pad = { l: 60, r: 14, t: 14, b: 40 };
  const xMin = 0.1;
  const xMax = 1000;
  const yMax = hw.peak * 3;
  const yMin = hw.bw * xMin * 0.5;
  const X = (v: number) => pad.l + ((Math.log10(v) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin))) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * (H - pad.t - pad.b);
  const ridge = hw.peak / hw.bw;
  const roof = (x: number) => Math.min(hw.peak, hw.bw * x);
  const yt: number[] = [];
  for (let e = Math.ceil(Math.log10(yMin)); e <= Math.floor(Math.log10(yMax)); e++) yt.push(10 ** e);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Roofline with prefill and decode products', 'Roofline với các phép nhân của prefill và decode')} style={{ width: '100%', maxWidth: W, display: 'block' }} data-ridge={ridge.toFixed(3)}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      <rect x={pad.l} y={pad.t} width={Math.max(0, X(ridge) - pad.l)} height={H - pad.t - pad.b} fill="var(--global-wash)" />
      <text x={pad.l + 6} y={pad.t + 15} fontSize="10.5" style={{ fill: 'var(--global)' }}>
        <Translated vi="giới hạn bởi băng thông">bandwidth-bound</Translated>
      </text>
      <text x={X(ridge) + 6} y={pad.t + 15} fontSize="10.5" style={{ fill: 'var(--reg)' }}>
        <Translated vi="giới hạn bởi tính toán">compute-bound</Translated>
      </text>
      {[0.1, 1, 10, 100, 1000].map((v) => (
        <g key={v}>
          <line x1={X(v)} x2={X(v)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(v)} y={H - pad.b + 14} fontSize="10.5" textAnchor="middle">{v}</text>
        </g>
      ))}
      {yt.map((v) => (
        <g key={v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 6} y={Y(v) + 4} fontSize="10" textAnchor="end">{v >= 1000 ? `${v / 1000}k` : v}</text>
        </g>
      ))}
      <text x={(W + pad.l) / 2} y={H - 6} fontSize="11" textAnchor="middle">
        <Translated vi="FLOP trên mỗi byte">FLOP per byte</Translated>
      </text>
      <text x={13} y={(H - pad.b + pad.t) / 2} fontSize="11" textAnchor="middle" transform={`rotate(-90 13 ${(H - pad.b + pad.t) / 2})`}>
        <Translated vi="trần GFLOP/s">GFLOP/s ceiling</Translated>
      </text>
      <polyline points={`${X(xMin)},${Y(hw.bw * xMin)} ${X(ridge)},${Y(hw.peak)} ${X(xMax)},${Y(hw.peak)}`} fill="none" stroke="var(--ink)" strokeWidth="2.2" />
      <line x1={X(ridge)} x2={X(ridge)} y1={Y(hw.peak)} y2={H - pad.b} stroke="var(--ink)" strokeDasharray="3 3" />
      <text x={X(ridge)} y={H - pad.b - 5} fontSize="10.5" textAnchor="middle">
        <Translated vi={<>điểm gãy {fmt(ridge)}</>}>ridge {fmt(ridge)}</Translated>
      </text>
      {points.map((pt, i) => {
        const x = X(Math.min(Math.max(pt.ai, xMin), xMax));
        const y = Y(roof(pt.ai));
        return (
          <g key={pt.label} data-point={pt.label}>
            <circle cx={x} cy={y} r="5" fill={pt.prefill ? 'var(--reg)' : 'var(--page)'} stroke={pt.prefill ? 'var(--reg)' : 'var(--global)'} strokeWidth="2" />
            <text x={x + 8} y={y + (i % 2 ? 15 : -7)} fontSize="10.5">{pt.label}</text>
          </g>
        );
      })}
    </svg>
  );
}

function RooflineBench() {
  const { reports } = useReports();
  const options = useMemo(() => [...measured(reports), BOOK], [reports]);
  const [key, setKey] = useState<string | null>(null);
  const hw = options.find((o) => o.key === key) ?? options[0];
  const [logN, setLogN] = useState(11);
  const [p, setP] = useState(2);
  const [d, setD] = useState(128);
  const N = Math.round(2 ** logN);
  const pts = aiPoints(N, d, p);
  const ridge = hw.peak / hw.bw;
  return (
    <Bench
      title={tr('Prefill and decode on your roofline', 'Prefill và decode trên roofline của bạn')}
      task={
        <Translated vi="Trượt độ dài prompt: những điểm nào di chuyển, và những điểm nào không bao giờ rời khỏi sườn dốc?">
          Slide the prompt length: which points move, and which never leave the slope?
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Vòng tròn rỗng là decode, vòng tròn đặc là prefill; mỗi điểm nằm ở trần của nó, min(peak, bandwidth × FLOP/B).{' '}
              {hw.note}
            </>
          }
        >
          <>Open circles are decode, filled circles prefill; each sits at its ceiling, min(peak, bandwidth × FLOP/B). {hw.note}</>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          <Translated vi="Mái">Roof</Translated>
          <select value={hw.key} onChange={(e) => setKey(e.target.value)} aria-label={tr('Roof', 'Mái')}>
            {options.map((o) => <option key={o.key} value={o.key}>{o.label}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi={<>Độ dài prompt N = {N.toLocaleString('en-US')}</>}>Prompt length N = {N.toLocaleString('en-US')}</Translated>
          <input type="range" min="0" max="14" step="0.25" value={logN} aria-label={tr('Prompt length', 'Độ dài prompt')} onChange={(e) => setLogN(Number(e.target.value))} />
        </label>
        <label className="field">
          <Translated vi="Phần tử">Element</Translated>
          <select value={p} onChange={(e) => setP(Number(e.target.value))} aria-label={tr('Element size', 'Kích thước phần tử')}>
            <option value={4}>{tr('fp32, 4 bytes', 'fp32, 4 byte')}</option>
            <option value={2}>{tr('16-bit, 2 bytes', '16-bit, 2 byte')}</option>
          </select>
        </label>
        <label className="field">
          <Translated vi="Độ rộng head d">Head width d</Translated>
          <select value={d} onChange={(e) => setD(Number(e.target.value))} aria-label={tr('Head width', 'Độ rộng head')}>
            {[64, 128, 256].map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
      </div>
      <div className="cols plot">
        <RooflinePlot hw={hw} points={pts.map((x) => ({ label: x.label, ai: x.ai, prefill: x.key.startsWith('prefill') }))} />
        <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-roofline="">
          <thead>
            <tr>
              <th>
                <Translated vi="Phép nhân">Product</Translated>
              </th>
              <th className="num">FLOP/B</th>
              <th>
                <Translated vi="Giới hạn">Bound</Translated>
              </th>
            </tr>
          </thead>
          <tbody>
            {pts.map((x) => (
              <tr key={x.key} data-key={x.key} data-ai={x.ai.toFixed(4)} data-bound={x.ai < ridge ? 'bandwidth' : 'compute'}>
                <td>{x.label}</td>
                <td className="num">{fmt(x.ai)}</td>
                <td>{x.ai < ridge ? (
                    <span className="chip global">
                      <Translated vi="băng thông">bandwidth</Translated>
                    </span>
                  ) : (
                    <span className="chip reg">
                      <Translated vi="tính toán">compute</Translated>
                    </span>
                  )}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions

export const questionsS204: Record<string, Question> = {
  'ch20/20.4/q-unchanged': {
    kind: 'choice',
    id: 'ch20/20.4/q-unchanged',
    prompt: (
      <Translated vi={<>Khi token N tham gia, vì sao các hàng 0 đến N − 1 của P giữ nguyên chính xác, đến tận bit cuối cùng?</>}>
        <>When token N joins, why do rows 0 to N − 1 of P stay exactly the same, down to the last bit?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Mỗi hàng đó có thêm một phần tử, và phần tử đó bị che, nên nó cộng 0 vào mọi tổng.">
            Each of those rows gains one entry, and it is masked, so it adds 0 to every sum.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chúng thay đổi chút ít; KV cache là một phép xấp xỉ bỏ qua thay đổi đó.">
            They change slightly; the KV cache is an approximation that ignores the change.
          </Translated>
        ),
        why: (
          <Translated vi="Phần đáp án so sánh chúng khi chưa làm tròn: giống hệt nhau. Cache là chính xác.">
            The reveal compared them unrounded: identical. The cache is exact.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chúng thay đổi, nhưng O thì không, vì nhân với V triệt tiêu thay đổi đó.">
            They change, but O does not, because multiplying by V cancels the change.
          </Translated>
        ),
        why: (
          <Translated vi="Ngay từ đầu P đã không thay đổi, và các hàng cũ của O cũng vậy.">
            P does not change in the first place, and O’s old rows do not either.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Phép trừ max của hàng triệt tiêu phần tử mới trước khi lấy tổng.">
            Subtracting the row maximum cancels the new entry before the sum is taken.
          </Translated>
        ),
        why: (
          <Translated vi="Max giữ nguyên vì phần tử mới là −∞; lý do là mặt nạ, không phải phép trừ.">
            The maximum stays the same because the new entry is −∞; the reason is the mask, not the subtraction.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hàng r của S + M có thêm cột N, và N &gt; r đặt nó phía trên đường chéo, nơi M chứa −∞. Max của hàng không đổi
            và tổng các lũy thừa của nó có thêm e<sup>−∞</sup> = 0, không thêm gì vào mẫu số và không thêm gì vào các tử
            số, nên mọi p trong hàng giữ nguyên giá trị. Lời của sách: các phần tử bị che &ldquo;do not dilute any of the
            probabilities&rdquo;.
          </p>
        }
      >
        <p>
          Row r of S + M gains column N, and N &gt; r puts it above the diagonal, where M holds −∞. The row’s maximum is
          unchanged and its sum of exponentials gains e<sup>−∞</sup> = 0, which adds nothing to the denominator and nothing to
          the numerators, so every p in the row keeps its value. The book’s
          words: the masked elements “do not dilute any of the probabilities”.
        </p>
      </Translated>
    ),
    source: 'ch. 20 §20.4, pp. 489–490, Fig. 20.5',
  },
  'ch20/20.4/q-new-row-cost': {
    kind: 'numeric',
    id: 'ch20/20.4/q-new-row-cost',
    prompt: (
      <Translated
        vi={
          <>
            Với một KV cache, một head của một lớp, N = 4,096 token đã cache và d = 128: q′Kᵀ và p′V cùng nhau tốn bao
            nhiêu FLOP? Đếm một phép nhân-cộng là 2.
          </>
        }
      >
        <>
          With a KV cache, one head of one layer, N = 4,096 cached tokens and d = 128: how many FLOPs do q′Kᵀ and p′V take
          together? Count a multiply-add as 2.
        </>
      </Translated>
    ),
    answer: 2097152,
    explain: (
      <Translated
        vi={
          <p>
            q′Kᵀ là N tích vô hướng độ dài d: 2Nd FLOP. p′V cộng N hàng gồm d giá trị, mỗi hàng được nhân hệ số: thêm 2Nd.
            Tổng cộng 4Nd = 4 × 4,096 × 128 = 2,097,152. Ba phép chiếu của hàng mới là riêng biệt và không phụ thuộc vào N.
          </p>
        }
      >
        <p>
          q′Kᵀ is N dot products of length d: 2Nd FLOPs. p′V adds N rows of d values, each scaled: another 2Nd. Together
          4Nd = 4 × 4,096 × 128 = 2,097,152. The three projections of the new row are separate and do not depend on N.
        </p>
      </Translated>
    ),
    source: 'ch. 20 §20.4, p. 490, Fig. 20.6',
  },
  'ch20/20.4/q-fp32-ai': {
    kind: 'numeric',
    id: 'ch20/20.4/q-fp32-ai',
    prompt: (
      <Translated vi={<>Bước decode đó đọc mọi hàng của K và của V một lần, ở fp32 (4 byte mỗi phần tử). Nó làm bao nhiêu FLOP trên mỗi byte đọc?</>}>
        <>That decode step reads every row of K and of V once, in fp32 (4 bytes per element). How many FLOPs does it do per byte read?</>
      </Translated>
    ),
    answer: 0.5,
    tol: 0.02,
    unit: 'FLOP/B',
    explain: (
      <Translated
        vi={
          <p>
            4Nd FLOP trên 2Nd phần tử × 4 byte = 0.5 FLOP/B, bất kể N. Ở 16-bit cùng bước đó làm 1 FLOP/B: đó là Eq. 20.9
            của sách (khoảng một phép toán mỗi phần tử, đếm với p = 1), và nguồn được trích dẫn nêu nó là tỉ lệ FLOP trên
            byte 1:1 cho BF16 (facts E4).
          </p>
        }
      >
        <p>
          4Nd FLOPs over 2Nd elements × 4 bytes = 0.5 FLOP/B, whatever N. In 16-bit the same step does 1 FLOP/B: that is the
          book’s Eq. 20.9 (about one operation per element, counted with p = 1), and the cited source states it as a 1:1
          FLOP-to-byte ratio for BF16 (facts E4).
        </p>
      </Translated>
    ),
    source: 'ch. 20 §20.6, pp. 506–507, Eq. 20.9; notes/today/ch20-facts.md E4', // i18n-ignore: citation
  },
  'ch20/20.4/q-is-causal': {
    kind: 'choice',
    id: 'ch20/20.4/q-is-causal',
    prompt: (
      <Translated
        vi={
          <>
            Một bước decode gọi <code>F.scaled_dot_product_attention(q, k, v, is_causal=True)</code> với một truy vấn và
            N key đã cache. Truy vấn tính attention với những key nào?
          </>
        }
      >
        <>
          A decode step calls <code>F.scaled_dot_product_attention(q, k, v, is_causal=True)</code> with one query and N cached
          keys. Which keys does the query attend to?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Chỉ key 0: PyTorch căn mặt nạ nhân quả theo góc trên bên trái.">
            Only key 0: PyTorch aligns the causal mask to the upper left.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Cả N key, như một bước decode cần, vì không có gì đến sau nó.">
            All N keys, as a decode step should, since nothing comes after it.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là điều bạn muốn, và là điều FlashAttention (2.1 trở đi) và MLX làm, nhưng không phải điều cờ của PyTorch làm với một mặt nạ không vuông.">
            That is what you want, and what FlashAttention (2.1 and later) and MLX do, but not what PyTorch’s flag does for a
            non-square mask.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ key cuối cùng, key vừa được nối thêm vào cache.">
            Only the last key, the one that was just appended to the cache.
          </Translated>
        ),
        why: (
          <Translated vi="Cách căn theo góc trên bên trái giữ cột đầu tiên của hàng đầu tiên, không phải cột cuối.">
            Upper-left alignment keeps the first column of the first row, not the last.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không key nào: PyTorch báo lỗi với một mặt nạ nhân quả không vuông.">
            None: PyTorch raises an error for a non-square causal mask.
          </Translated>
        ),
        why: (
          <Translated vi="Không phải vậy: nó ghi trong tài liệu cách căn theo góc trên bên trái.">
            It does not: it documents the upper-left alignment.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Tài liệu PyTorch mô tả <code>is_causal</code> là một causal bias căn theo góc trên bên trái khi truy vấn ngắn hơn
            các key: hàng 0 chỉ giữ cột 0. Với một truy vấn, đừng truyền mặt nạ nào cả (bản tham chiếu Llama chỉ dựng mặt
            nạ khi xử lý nhiều hơn một token) hoặc truyền một causal bias căn theo góc dưới bên phải.
          </p>
        }
      >
        <p>
          PyTorch documents <code>is_causal</code> as an upper-left causal bias when the query is shorter than the keys:
          row 0 keeps column 0 only. For one query, pass no mask at all (the Llama reference builds a mask only when more than one token is
          processed) or a lower-right causal bias.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch20-facts.md D5, D3, A8, D4', // i18n-ignore: citation
  },
  'ch20/20.4/q-prefill-bound': {
    kind: 'choice',
    id: 'ch20/20.4/q-prefill-bound',
    prompt: (
      <Translated vi={<>Prefill một prompt 2,000 token và một bước decode đều tính q Kᵀ. Vì sao cái đầu thường bị giới hạn bởi tính toán còn cái sau bị giới hạn bởi băng thông?</>}>
        <>Prefill of a 2,000-token prompt and one decode step both compute q Kᵀ. Why is the first usually compute-bound and the second bandwidth-bound?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Mỗi hàng K được nạp gặp 2,000 truy vấn trong prefill và một truy vấn trong decode.">
            Each loaded K row meets 2,000 queries in prefill and one in decode.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Prefill chạy trên tensor core còn decode chạy trên các lõi (core) thông thường.">
            Prefill runs on the tensor cores and decode on the ordinary cores.
          </Translated>
        ),
        why: (
          <Translated vi="Lựa chọn phần cứng đi theo hình dạng, không phải ngược lại; một GEMV không cho tensor core thứ gì để dùng lại.">
            Hardware choice follows from the shape, not the other way round; a GEMV gives tensor cores nothing to reuse.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Decode chạy một softmax mỗi bước; prefill chạy nó một lần ở cuối.">
            Decode runs a softmax every step; prefill runs it once at the end.
          </Translated>
        ),
        why: (
          <Translated vi="Cả hai đều chạy một softmax trên các điểm số của mình, và cách đếm của sách bỏ qua nó ở cả hai.">
            Both run one over their scores, and the book’s count ignores it in both.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Tập dữ liệu làm việc của prefill vừa trong cache, còn KV cache của decode thì không.">
            Prefill’s working set fits in cache, and decode’s KV cache does not.
          </Translated>
        ),
        why: (
          <Translated vi="Lập luận không cần bộ nhớ đệm nào cả: nó là về số lần mỗi byte được nạp được dùng.">
            The argument needs no cache at all: it is about how many times each loaded byte is used.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Bài học của chương 5 ở một chỗ mới. Một hàng key được nạp tốn cùng số byte ở cả hai giai đoạn; điều khác biệt
            là nó nuôi bao nhiêu phép nhân-cộng. Prefill là một GEMM có cường độ tính toán tăng theo N về phía khoảng 2d/p
            FLOP/B; decode là một GEMV gần 2/p, bất kể N. Trình khám phá roofline cho thấy điểm prefill trượt sang phải
            theo N trong khi điểm decode đứng yên.
          </p>
        }
      >
        <p>
          Chapter 5’s lesson in a new place. A loaded key row costs the same bytes in both phases; what differs is how many
          multiply-adds it feeds. Prefill is a GEMM whose intensity grows with N towards about 2d/p FLOP/B; decode is a GEMV
          near 2/p, whatever N. The roofline explorer shows the prefill point sliding right with N while the decode point
          stays where it is.
        </p>
      </Translated>
    ),
    source: 'ch. 20 §20.4, pp. 491–492; ch. 5 §5.1', // i18n-ignore: citation
  },
  'ch20/20.4/q-per-layer': {
    kind: 'choice',
    id: 'ch20/20.4/q-per-layer',
    prompt: (
      <Translated vi={<>Fig. 20.7 trao K và V từ mỗi lớp cho cùng lớp đó ở vòng lặp tiếp theo. Vì sao một cache không thể phục vụ mọi lớp?</>}>
        <>Fig. 20.7 hands K and V from each layer to the same layer of the next iteration. Why can’t one cache serve every layer?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Mỗi lớp tính K và V từ đầu vào riêng của nó với trọng số riêng của nó.">
            Each layer computes K and V from its own input with its own weights.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Có thể được; một cache mỗi lớp chỉ là một lựa chọn sổ sách trong mã.">
            It could; one cache per layer is only a bookkeeping choice in the code.
          </Translated>
        ),
        why: (
          <Translated vi="Các giá trị khác nhau giữa các lớp, nên chúng phải được lưu riêng.">
            The values differ from layer to layer, so they must be stored separately.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các lớp có độ rộng head khác nhau, nên các hàng sẽ không khớp nhau.">
            The layers have different head widths, so the rows would not line up.
          </Translated>
        ),
        why: (
          <Translated vi="Trong các mô hình ở đây thì không; lý do là các giá trị khác nhau.">
            In the models here they do not; the reason is that the values differ.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Mặt nạ nhân quả khác nhau giữa các lớp, nên các key sẽ không khớp.">
            The causal mask differs from layer to layer, so the keys would not match.
          </Translated>
        ),
        why: (
          <Translated vi="Mặt nạ nhân quả giống nhau ở mọi lớp.">The causal mask is the same in every layer.</Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            X cho lớp 1 là các embedding; X cho lớp 2 là đầu ra của lớp 1, và cứ thế: K của một token ở lớp l phụ thuộc vào
            mọi thứ các lớp trước l đã làm với nó. Vì vậy K = X W_K là một ma trận khác nhau ở mọi lớp, và cache tăng theo
            số lớp cũng như theo N (Eq. 20.7 có thừa số l).
          </p>
        }
      >
        <p>
          X for layer 1 is the embeddings; X for layer 2 is layer 1’s output, and so on: a token’s K in layer l depends on
          everything the layers before l did to it. So K = X W_K is a different matrix in every layer, and the cache grows with the number of layers as well as with N (Eq. 20.7 has the factor l).
        </p>
      </Translated>
    ),
    source: 'ch. 20 §20.4, p. 491, Fig. 20.7',
  },
};

// ---------------------------------------------------------------- the page

export function S204() {
  return (
    <>
      <Translated
        vi={
          <p>
            Mục 20.2 đưa mỗi token mới qua mọi lớp bằng cách tính lại mọi thứ: trong mỗi bước sinh, mỗi lớp nhân ra X W_Q,
            X W_K, X W_V, rồi Q Kᵀ, rồi P V, năm phép nhân ma trận trên cả N hàng của X. N có thể lên tới hàng chục nghìn
            token. Thế nhưng từ bước này sang bước kế, X chỉ có thêm đúng một hàng. Trước khi đọc câu trả lời của sách, hãy
            tự tính xem bao nhiêu phần của khối tính toán đó tạo ra thứ gì mới.
          </p>
        }
      >
        <p>
          Section 20.2 ran each new token through every layer by recomputing everything: in every generation step, each layer
          multiplies out X W_Q, X W_K, X W_V, then Q Kᵀ, then P V, five matrix products over all N rows of X. N can reach tens
          of thousands of tokens. Yet from one step to the next, X gains exactly one row. Before reading the book’s answer,
          work out for yourself how much of that arithmetic produces something new.
        </p>
      </Translated>
      <Source section="20.4" pages="488" />

      <h2>
        <Translated vi="Dự đoán những ô nào thay đổi">Predict which cells change</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Dưới đây là một head với sáu token và d = 4, vẽ như Fig. 20.2 của sách, với hệ số 1/√d và mặt nạ được gộp vào S.
            Một token thứ bảy sắp đến. Mỗi lưới đã có sẵn kích thước nó sẽ có với bảy token; các ô nét đứt là những ô mà
            token mới thêm vào. Nhấn vào mọi ô bạn nghĩ cần tính toán mới, rồi xem đáp án. Phần đáp án chạy attention ở sáu
            token và ở bảy token rồi so sánh mọi giá trị.
          </p>
        }
      >
        <p>
          Below is one head with six tokens and d = 4, drawn like the book’s Fig. 20.2, with the 1/√d scale and the mask
          folded into S. A seventh token is about to arrive. Each grid is already the size it will have with seven tokens;
          the dashed slots are the cells the new token adds. Click every cell you expect to need new arithmetic, then reveal.
          The reveal runs the attention at six tokens and at seven and compares every value.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một head, bốn cột, các đầu vào nhỏ và tròn để bạn đọc được giá trị. Các mô hình ở mục 20.6 có head rộng 128
              hoặc 256 và hàng chục head mỗi lớp; khuôn mẫu là như nhau.
            </p>
          }
        >
          <p>One head, four columns, small round inputs so that you can read the values. The models of section 20.6 have heads 128 or 256 wide and dozens of them per layer; the pattern is the same.</p>
        </Translated>
      </Simplification>
      <CellsBench />
      <Translated
        vi={
          <p>
            X, Q, K và V mỗi ma trận có thêm một hàng dưới cùng: embedding của token mới, và ba phép nhân vectơ-ma trận của
            nó với W_Q, W_K và W_V. Trong S, mọi ô cũ (r, c) chỉ phụ thuộc vào hàng truy vấn r và hàng key c, nên không ô
            nào thay đổi. Hàng mới của S là truy vấn mới đối với mọi key, q′Kᵀ, thêm một phép nhân vectơ-ma trận. Cột mới là
            các truy vấn cũ đối với key mới, và mọi ô của nó trừ (N, N) nằm phía trên đường chéo, nên mặt nạ biến chúng
            thành −∞. Sách gọi chúng là &ldquo;all zero elements&rdquo;; đó là giá trị của chúng sau softmax, không phải
            trước.
          </p>
        }
      >
        <p>
          X, Q, K and V each gain one bottom row: the new token’s embedding, and three vector-matrix products of it with
          W_Q, W_K and W_V. In S, every old cell (r, c) depends only on query row r and key row c, so none moves. The new row
          of S is the new query against all keys, q′Kᵀ, one more vector-matrix product. The new column is the old queries
          against the new key, and every cell of it except (N, N) lies above the diagonal, so the mask makes it −∞. The book
          calls these “all zero elements”; that is their value after the softmax, not before it.
        </p>
      </Translated>
      <Source section="20.4" pages="489" figs={tr('Fig. 20.5', 'Hình 20.5')} />
      <Translated
        vi={
          <p>
            Các hàng cũ của P và O cũng không thay đổi: không phải xấp xỉ, mà từng bit một, như phần đáp án kiểm tra. Câu
            hỏi bên dưới hỏi vì sao.
          </p>
        }
      >
        <p>
          The old rows of P and O do not change either: not approximately, but bit for bit, as the reveal checks. The
          question below asks why.
        </p>
      </Translated>
      <Ask q={questionsS204['ch20/20.4/q-unchanged']} />

      <h2>
        <Translated vi="Giữ lại K và V">Keep K and V</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Vậy một bước decode đưa một hàng mới của X vào một head attention và phải tạo ra một hàng mới của O. Nó vẫn cần
            toàn bộ K và V: mọi key trước đó để chấm điểm truy vấn mới, và mọi value trước đó để trộn. Chúng đã được tính ở
            các bước trước và không thay đổi, nên hãy lưu chúng thay vì tính lại. Kho lưu đó là <b>KV cache</b>. (Chú thích
            6 của sách nói thêm rằng &ldquo;KV&rdquo; ở đây không phải là một kho key-value, dù ý tưởng gần giống: đầu ra là
            một tổng các value, có trọng số theo mức độ truy vấn gần với key của từng value.)
          </p>
        }
      >
        <p>
          So a decode step takes one new row of X into an attention head and must produce one new row of O. It still needs
          all of K and V: every earlier key to score the new query against, and every earlier value to mix. Those were
          computed in earlier steps and do not change, so store them instead of recomputing them. That store is the{' '}
          <b>KV cache</b>. (The book&rsquo;s footnote 6 adds that &ldquo;KV&rdquo; here is not a key-value store, though
          the idea is close: the output is a sum of values, weighted by how near the query is to each value&rsquo;s key.)
        </p>
      </Translated>
      <Source section="20.4" pages="490" figs={tr('footnote 6', 'chú thích 6')} />

      <h2>
        <Translated vi="Phần việc của một token">One token’s work</Translated>
      </h2>
      <Translated vi={<p>Với một cache, một bước của một head là năm thao tác nhỏ (Fig. 20.6):</p>}>
        <p>With a cache, one step of one head is five small operations (Fig. 20.6):</p>
      </Translated>
      <ol>
        <li>
          <Translated vi="q′ = x′W_Q, k′ = x′W_K, v′ = x′W_V: ba phép nhân vectơ-ma trận của hàng mới, mỗi phép 1 × d;">
            q′ = x′W_Q, k′ = x′W_K, v′ = x′W_V: three vector-matrix products of the new row, 1 × d each;
          </Translated>
        </li>
        <li>
          <Translated vi="nối k′ vào K và v′ vào V, giờ có N + 1 hàng;">append k′ to K and v′ to V, which now have N + 1 rows;</Translated>
        </li>
        <li>
          <Translated vi="s′ = q′Kᵀ, một hàng gồm N + 1 điểm số;">s′ = q′Kᵀ, a row of N + 1 scores;</Translated>
        </li>
        <li>p′ = softmax(s′);{/* i18n-ignore: equation */}</li>
        <li>
          <Translated vi="o′ = p′V, một hàng đầu ra.">o′ = p′V, one row of output.</Translated>
        </li>
      </ol>
      <Translated
        vi={
          <p>
            Không có mặt nạ. Token mới nhất được phép tính attention với mọi token đã cache, và chưa có token nào đến sau để
            nó phải bị chặn khỏi.
          </p>
        }
      >
        <p>
          There is no mask. The newest token may attend to every cached token, and no later token exists yet for it to be
          kept from.
        </p>
      </Translated>
      <Source section="20.4" pages="490–491" figs={tr('Fig. 20.6', 'Hình 20.6')} />
      <Translated
        vi={
          <p>
            Mã Llama tham chiếu của Meta đồng ý: nó chỉ dựng mặt nạ nhân quả khi có nhiều hơn một token đi qua cùng lúc,
            tức là trong prefill. Một kernel decode có thể bỏ hẳn mặt nạ, tiết kiệm một phép so sánh cho mỗi điểm số và bớt
            đi một nguồn lỗi.
          </p>
        }
      >
        <p>
          Meta’s reference Llama code agrees: it builds a causal mask only when more than one token goes through at once,
          that is, in prefill. A decode kernel can skip the mask entirely, which is one comparison per score saved and one
          source of bugs gone.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch20-facts D3)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              Trong <code>llama/model.py</code> mặt nạ là <code>None</code> trừ khi <code>seqlen &gt; 1</code>; khi đó nó
              là một mặt nạ <code>torch.triu</code> với các số 0 được thêm vào trước cho các vị trí đã cache.
            </p>
          }
        >
          <p>In <code>llama/model.py</code> the mask is <code>None</code> unless <code>seqlen &gt; 1</code>; then it is a <code>torch.triu</code> mask with zeros prepended for the cached positions.</p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Prefill lấp đầy cache của mọi lớp">Prefill fills every layer’s cache</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lượt đầu tiên qua prompt, mà mục 20.2 gọi là tóm tắt (summarization), còn được gọi là <b>prefill</b>: mọi lớp
            chạy phiên bản ma trận đầy đủ trên cả prompt và để lại K và V của nó. Mỗi lớp có cache riêng. Trong mỗi bước
            sinh, token đầu ra cuối cùng trở thành một hàng mới cho lớp 1; mỗi lớp tính q′, k′ và v′, nối vào cache riêng
            của nó, tính o′ và chuyển một hàng sang lớp tiếp theo.
          </p>
        }
      >
        <p>
          The first pass over the prompt, which section 20.2 called summarization, is also called <b>prefill</b>: every
          layer runs the full matrix version on the whole prompt and leaves its K and V behind. Every layer has its own
          cache. In each generation step the last output token becomes one new row for layer 1; each layer computes q′, k′
          and v′, appends to its own cache, computes o′ and passes one row on to the next layer.
        </p>
      </Translated>
      <Source section="20.4" pages="491" figs={tr('Fig. 20.7', 'Hình 20.7')} />
      <Ask q={questionsS204['ch20/20.4/q-per-layer']} />

      <h2>
        <Translated vi="Tính lại hay dùng cache">Recompute or cache</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Giờ hãy đếm. Trình khám phá cộng năm phép nhân cho token tiếp theo khi có và không có cache. Trước khi di chuyển
            bất cứ thứ gì, hãy quyết định hàng nào trong ba hàng của bảng sẽ tăng nhanh nhất khi N tăng.
          </p>
        }
      >
        <p>
          Now count. The explorer adds up the five products for the next token with and without the cache. Before you move
          anything, decide which of the three rows of the table will grow fastest as N grows.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Chỉ năm phép nhân của sách được đếm: không có phép chiếu đầu ra, không có lớp feed-forward, không có softmax.
              Chúng tốn như nhau dù có hay không có cache, trừ softmax, vốn cũng thu nhỏ còn một hàng.
            </p>
          }
        >
          <p>Only the book’s five products are counted: no output projection, no feed-forward layer, no softmax. Those cost the same with or without a cache, except the softmax, which also shrinks to one row.</p>
        </Translated>
      </Simplification>
      <RecomputeBench />
      <Translated
        vi={
          <p>
            Tính lại tăng theo N² trong Q Kᵀ và P V và theo N D² trong các phép chiếu. Bước dùng cache tăng theo N D trong
            attention và hoàn toàn không tăng trong các phép chiếu. Và tỉ số giữa chúng đúng bằng số token n: bản cơ sở làm
            lại cùng một hàng công việc cho mỗi token, 6nD² + 4n²D = n(6D² + 4nD). Chi phí đã chuyển từ tính toán sang bộ
            nhớ: cache chứa 2 × n × D × layers × p byte, và mục tiếp theo tính kích thước của nó.
          </p>
        }
      >
        <p>
          Recomputing grows with N² in Q Kᵀ and P V and with N D² in the projections. The cached step grows with N D in
          attention and not at all in the projections. And the ratio between them is exactly the number of tokens n: the
          baseline redoes the same row of work for every one of them, 6nD² + 4n²D = n(6D² + 4nD). The cost has moved from
          arithmetic to memory: the cache holds 2 × n × D × layers × p bytes, and the next section sizes it.
        </p>
      </Translated>
      <Ask q={questionsS204['ch20/20.4/q-new-row-cost']} label={tr('Work it out', 'Tự tính')} />

      <h2>
        <Translated vi="Prefill bị giới hạn bởi tính toán, decode bị giới hạn bởi băng thông">
          Compute-bound prefill, bandwidth-bound decode
        </Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách mô tả hai giai đoạn, mỗi giai đoạn bằng một câu. Prefill là một lượt gồm các phép nhân ma trận-ma trận lớn,
            thường bị giới hạn bởi tính toán, và có thể giữ GPU bận rộn. Sinh token là một lượt cho mỗi token, chủ yếu gồm
            các phép nhân vectơ-ma trận, thường bị giới hạn bởi băng thông bộ nhớ, và để phần lớn các đơn vị tính toán nhàn
            rỗi.
          </p>
        }
      >
        <p>
          The book characterises the two phases in one sentence each. Prefill is one pass of large matrix-matrix
          multiplications, typically compute-bound, and can keep a GPU busy. Generation is one pass per token of mostly
          vector-matrix products, typically memory-bandwidth-bound, and leaves most of the arithmetic units idle.
        </p>
      </Translated>
      <Source section="20.4" pages="491–492" />
      <Translated
        vi={
          <p>
            Chương 5 cho biết lý do. Trong một GEMM, mỗi hàng được nạp của K, hay của một ma trận trọng số, được mọi hàng
            của toán hạng kia dùng lại; trong một GEMV, mỗi byte được nạp chỉ nuôi một phép nhân-cộng và không bao giờ được
            dùng lại. Hãy đặt bốn phép nhân của một head lên roofline của chính bạn, từ{' '}
            <a href={href('ch05', 'labs')}>lab 5.1</a>.
          </p>
        }
      >
        <p>
          Chapter 5 gives the reason. In a GEMM each loaded row of K, or of a weight matrix, is reused by every row of the
          other operand; in a GEMV each loaded byte feeds one multiply-add and is never used again. Place the four products
          of one head on your own roofline, from <a href={href('ch05', 'labs')}>lab 5.1</a>.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Chỉ lưu lượng bắt buộc: mỗi toán hạng đọc một lần và kết quả ghi một lần, không có bộ nhớ đệm, các phép nhân
              của một head. Q Kᵀ của prefill ghi N × N điểm số của nó, như bản cơ sở không gộp của mục 20.3.
            </p>
          }
        >
          <p>Compulsory traffic only: each operand read once and the result written once, no caches, one head’s products. Prefill’s Q Kᵀ writes its N × N scores, as the unfused baseline of section 20.3 does.</p>
        </Translated>
      </Simplification>
      <RooflineBench />
      <Translated
        vi={
          <p>
            Các điểm của prefill trượt sang phải khi prompt dài ra và vượt qua điểm gãy; Q Kᵀ của nó tiến gần 2d/p FLOP
            trên mỗi byte. Các điểm decode ở gần 2/p dù cache dài bao nhiêu: 1 FLOP trên mỗi byte ở 16-bit, 0.5 ở fp32. Hãy
            xem cả một bước decode chảy qua bộ nhớ.
          </p>
        }
      >
        <p>
          Prefill’s points slide right as the prompt grows and cross the ridge; its Q Kᵀ approaches 2d/p FLOP per byte. The
          decode points stay near 2/p however long the cache is: 1 FLOP per byte in 16-bit, 0.5 in fp32. Watch a whole
          decode step stream through memory.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một lớp có hình dạng Llama 2 7B (32 head cỡ 128) ở 4,096 token, 16-bit. Mỗi byte được đếm một lần: không dùng
              lại bộ nhớ đệm, không chồng lấp. Chỉ các phép chiếu Q, K và V và hai phép nhân attention; không có phép chiếu
              đầu ra, không có feed-forward.
            </p>
          }
        >
          <p>
            One layer of a Llama 2 7B shape (32 heads of 128) at 4,096 tokens, 16-bit. Each byte is counted once: no cache
            reuse, no overlap. Only the Q, K and V projections and the two attention products; no output projection, no
            feed-forward.
          </p>
        </Translated>
      </Simplification>
      <DecodeStreamScene />
      <Ask q={questionsS204['ch20/20.4/q-fp32-ai']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS204['ch20/20.4/q-prefill-bound']} />

      <h2>
        <Translated vi="Cache được bố trí thế nào trong mã bạn sẽ đọc">How the cache is laid out in code you will read</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách dừng ở phần đại số: nó không đưa ra cấu trúc dữ liệu nào cho cache, không có phép append và không có kernel
            decode. Ba codebase bạn có khả năng sẽ đọc lưu nó theo ba cách: mảng cấp phát trước của Meta, vòng các ô token
            của llama.cpp, và cache tăng theo từng nấc của mlx-lm.
          </p>
        }
      >
        <p>
          The book stops at the algebra: it shows no data structure for the cache, no append and no decode kernel. Three
          code bases you are likely to read store it three ways: Meta&rsquo;s preallocated array, llama.cpp&rsquo;s ring of
          token slots, and mlx-lm&rsquo;s cache that grows in steps.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch20-facts D3)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              Bản tham chiếu Llama của Meta cấp phát trước <code>cache_k</code> dạng [max_batch, max_seq, kv_heads,
              head_dim] và ghi <code>cache_k[:bsz, start_pos : start_pos + seqlen]</code>. llama.cpp giữ một tensor K và
              một tensor V cho mỗi lớp, một vòng các ô token được ghi theo chỉ số bằng <code>ggml_set_rows</code>; nó không
              có bảng khối. mlx-lm tăng cache theo từng nấc 256 token.
            </p>
          }
        >
          <p>Meta’s Llama reference preallocates <code>cache_k</code> as [max_batch, max_seq, kv_heads, head_dim] and writes <code>cache_k[:bsz, start_pos : start_pos + seqlen]</code>. llama.cpp keeps one K and one V tensor per layer, a ring of token slots written by index with <code>ggml_set_rows</code>; it has no block table. mlx-lm grows its cache in steps of 256 tokens.</p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Lab 20.4 dùng bố cục đầu tiên, một cache cho mỗi lớp có hình dạng [batch, max_seq, kv_heads, head_dim], và yêu
            cầu bạn viết phép append và bước decode một token. Các token đứng trước các head, nên nối thêm một token ghi một
            dải liên tục gồm kv_heads × head_dim giá trị.
          </p>
        }
      >
        <p>
          Lab 20.4 uses the first layout, one cache per layer of shape [batch, max_seq, kv_heads, head_dim], and asks you to
          write the append and the single-token decode step. Tokens come before heads, so appending one token writes
          one contiguous run of kv_heads × head_dim values.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Lab 20.4 là của khóa học này: sách không có mã cache hay decode. Xem{' '}
              <a href={href('ch20', 'labs')}>trang lab</a>.
            </p>
          }
        >
          <p>Lab 20.4 is ours: the book has no cache or decode code. See the <a href={href('ch20', 'labs')}>labs page</a>.</p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Hai cái bẫy khi bạn gọi một thư viện">Two traps when you call a library</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một bước decode có một truy vấn và N key, nên ma trận điểm số của nó là 1 × N, không vuông. Các thư viện không
            thống nhất về chỗ đặt mặt nạ nhân quả trong một ma trận không vuông, và một lời gọi phổ biến làm sai bước decode
            nếu bạn truyền cờ causal theo thói quen.
          </p>
        }
      >
        <p>
          A decode step has one query and N keys, so its score matrix is 1 × N, not square. Libraries disagree about where a
          causal mask goes in a non-square matrix, and one popular call gets the decode step wrong if you pass the causal
          flag out of habit.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch20-facts D5, A8, D4)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              <code>is_causal=True</code> của PyTorch căn mặt nạ theo góc trên bên trái khi truy vấn ngắn hơn các key.
              FlashAttention (từ 2.1) và MLX căn nó theo góc dưới bên phải, nơi truy vấn cuối gặp key cuối.
            </p>
          }
        >
          <p>PyTorch’s <code>is_causal=True</code> aligns the mask to the upper left when the query is shorter than the keys. FlashAttention (since 2.1) and MLX align it to the lower right, where the last query meets the last key.</p>
        </Translated>
      </Today>
      <Ask q={questionsS204['ch20/20.4/q-is-causal']} />
      <Translated
        vi={
          <p>
            Cái bẫy thứ hai là thứ cache chứa. Trong transformer của sách, thông tin vị trí được cộng một lần vào các
            embedding. Trong các mô hình mở bạn sẽ nạp, một rotary embedding được áp dụng cho Q và K bên trong mọi lớp
            attention, trước khi K được ghi vào cache. Vì vậy cache chứa các key đã xoay, và một kernel gộp phép append
            phải xoay trước.
          </p>
        }
      >
        <p>
          The second trap is what the cache holds. In the book’s transformer, position information is added once to the
          embeddings. In the open models you will load, a rotary embedding is applied to Q and K inside every attention
          layer, before K is written to the cache. So the cache holds rotated keys, and a kernel that fuses the append must
          rotate first.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch20-facts E8, D3)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              Llama 2 dùng rotary positional embedding; mã tham chiếu gọi <code>apply_rotary_emb</code> trên{' '}
              <code>xq</code> và <code>xk</code> rồi ghi <code>xk</code> vào <code>cache_k</code>.
            </p>
          }
        >
          <p>Llama 2 uses rotary positional embeddings; the reference code calls <code>apply_rotary_emb</code> on <code>xq</code> and <code>xk</code> and then writes <code>xk</code> into <code>cache_k</code>.</p>
        </Translated>
      </Today>
    </>
  );
}
