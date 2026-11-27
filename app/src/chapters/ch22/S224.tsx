import { useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source } from '../../components/notes';
import { useNarrow } from '../../lib/narrow';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';
import {
  COULOMB,
  GiB,
  LLAMA2_7B,
  SERIES,
  advantage,
  crossover,
  curves,
  kvBytes,
  slope,
  tieFactor,
  time,
  type Preset,
  type Series,
} from '../../sim/formulation';

const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const WRAP: CSSProperties = { flexWrap: 'wrap' };
const int = (v: number) => Math.round(v).toLocaleString('en-US');
const sig = (v: number, d = 3) => (v >= 1000 ? int(v) : Number(v.toPrecision(d)).toLocaleString('en-US'));
const sci = (v: number) => {
  if (v < 1e4) return sig(v);
  const e = Math.floor(Math.log10(v));
  const m = v / 10 ** e;
  return `${Number(m.toFixed(1))} × 10${String(e).replace(/\d/g, (c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+c])}`;
};


// ---------------------------------------------------------------- the explorer
const PRESET_LABEL: Record<Preset, string> = {
  get coulomb() { return tr("Electrostatic map (the book's)", 'Bản đồ tĩnh điện (của sách)'); },
  get kv() { return tr('Decoding a sequence (ours, from chapter 20)', 'Decode một chuỗi (của chúng tôi, từ chương 20)'); },
};
const NAMES: Record<Preset, { direct: string; reformed: string; directText: string; reformedText: string; size: string; unit: (x: number) => string }> = {
  coulomb: {
    get direct() { return tr('direct summation', 'tổng trực tiếp'); },
    get reformed() { return tr('cutoff with bins', 'cutoff với bin'); },
    get directText() { return tr('direct summation', 'tổng trực tiếp'); },
    get reformedText() { return tr('cutoff summation with bins', 'tổng cutoff với bin'); },
    get size() { return tr('volume, relative to the reference system', 'thể tích, so với hệ tham chiếu'); },
    unit: (x) => tr(`${sig(x)}× the volume`, `${sig(x)}× thể tích`),
  },
  kv: {
    get direct() { return tr('recompute the prefix', 'tính lại tiền tố'); },
    get reformed() { return tr('KV cache', 'KV cache'); },
    get directText() { return tr('recomputing the prefix', 'việc tính lại tiền tố'); },
    get reformedText() { return tr('the KV cache', 'KV cache'); },
    get size() { return tr('position of the token being generated', 'vị trí của token đang được sinh'); },
    unit: (x) => tr(`token ${int(x)}`, `token ${int(x)}`),
  },
};
const SERIES_STYLE: Record<Series, { stroke: string; dash?: string; width: number }> = {
  'direct-slow': { stroke: 'var(--ink-3)', width: 1.8 },
  'direct-fast': { stroke: 'var(--ink)', width: 2.2 },
  'reformed-slow': { stroke: 'var(--ink-3)', dash: '7 4', width: 1.8 },
  'reformed-fast': { stroke: 'var(--ink)', dash: '7 4', width: 2.2 },
};
const seriesName = (p: Preset, s: Series, S: number) =>
  `${s.startsWith('direct') ? NAMES[p].direct : NAMES[p].reformed}, ${s.endsWith('fast') ? `GPU (${S}×)` : 'CPU'}`; // i18n-ignore: processor names and a factor

/** Size sliders: volume in tenths of a decade from 0.01 to 1000; token position in quarter-doublings from 1 to 65,536. */
const SIZE_SLIDER: Record<Preset, { max: number; at: number; of: (v: number) => number }> = {
  coulomb: { max: 50, at: 20, of: (v) => Number((10 ** (v / 10 - 2)).toPrecision(3)) },
  kv: { max: 64, at: 40, of: (v) => Math.round(2 ** (v / 4)) },
};
/** Points along a log axis, for drawing. */
const logOf = (v: number, lo: number, hi: number) => Math.exp(Math.log(lo) + (v / 100) * (Math.log(hi) - Math.log(lo)));
const SPEEDS = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000];

function CostPlot({ preset, x, S, cross }: { preset: Preset; x: number; S: number; cross: number | null }) {
  const c = curves(preset);
  const narrow = useNarrow(600); // a narrower drawing on a phone keeps the labels legible
  const W = narrow ? 400 : 640;
  const H = narrow ? 300 : 330;
  const pad = { l: narrow ? 48 : 64, r: 12, t: 16, b: 44 };
  const xs = Array.from({ length: 121 }, (_, i) => logOf((100 * i) / 120, c.min, c.max));
  const all = SERIES.flatMap((s) => xs.map((v) => time(c, s, v, S)));
  const yMin = Math.min(...all) / 2;
  const yMax = Math.max(...all) * 2;
  const X = (v: number) => pad.l + ((Math.log10(v) - Math.log10(c.min)) / (Math.log10(c.max) - Math.log10(c.min))) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * (H - pad.t - pad.b);
  const xt: number[] = [];
  for (let e = Math.ceil(Math.log10(c.min)); e <= Math.floor(Math.log10(c.max)); e += narrow ? 2 : 1) xt.push(10 ** e);
  const yt: number[] = [];
  const step = Math.max(1, Math.ceil((Math.log10(yMax) - Math.log10(yMin)) / (narrow ? 5 : 7)));
  for (let e = Math.ceil(Math.log10(yMin)); e <= Math.floor(Math.log10(yMax)); e += step) yt.push(10 ** e);
  return (
    <div>
    <div className="ui" style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem 1.1rem', fontSize: '0.85em', margin: '0 0 0.3rem' }}>
      {SERIES.map((s) => (
        <span key={s} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }} data-legend={s}>
          <svg width="28" height="8" aria-hidden="true">
            <line x1="1" x2="27" y1="4" y2="4" stroke={SERIES_STYLE[s].stroke} strokeWidth={SERIES_STYLE[s].width} strokeDasharray={SERIES_STYLE[s].dash} />
          </svg>
          {s.startsWith('direct') ? NAMES[preset].direct : NAMES[preset].reformed}, {s.endsWith('fast') ? 'GPU' : 'CPU'}
        </span>
      ))}
    </div>
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Cost against problem size, log-log', 'Chi phí theo kích thước bài toán, thang log-log')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {xt.map((v) => (
        <g key={'x' + v}>
          <line x1={X(v)} x2={X(v)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(v)} y={H - pad.b + 15} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
            {v < 1 ? v : v >= 1000 ? v / 1000 + 'k' : v}
          </text>
        </g>
      ))}
      {yt.map((v) => (
        <g key={'y' + v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 6} y={Y(v) + 4} fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
            10{String(Math.round(Math.log10(v))).replace(/-?\d/g, (ch) => ('-' === ch ? '⁻' : '⁰¹²³⁴⁵⁶⁷⁸⁹'[+ch]))}
          </text>
        </g>
      ))}
      <text x={(W + pad.l) / 2} y={H - 8} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
        {NAMES[preset].size}
      </text>
      <text x={14} y={(H - pad.b + pad.t) / 2} fontSize="11.5" textAnchor="middle" transform={`rotate(-90 14 ${(H - pad.b + pad.t) / 2})`} style={{ fill: 'var(--ink-2)' }}>
        <Translated vi="thời gian (số phép toán ÷ tốc độ)">time (operations ÷ speed)</Translated>
      </text>
      {SERIES.map((s) => (
        <polyline
          key={s}
          data-series={s}
          points={xs.map((v) => `${X(v)},${Y(time(c, s, v, S))}`).join(' ')}
          fill="none"
          stroke={SERIES_STYLE[s].stroke}
          strokeWidth={SERIES_STYLE[s].width}
          strokeDasharray={SERIES_STYLE[s].dash}
        />
      ))}
      <line x1={X(x)} x2={X(x)} y1={pad.t} y2={H - pad.b} stroke="var(--ink-2)" strokeWidth="1" />
      {cross != null ? (
        <g data-mark="crossover">
          <circle cx={X(cross)} cy={Y(time(c, 'direct-fast', cross, S))} r="5.5" fill="none" stroke="var(--ink)" strokeWidth="2" />
          <text x={X(cross) + 8} y={Y(time(c, 'direct-fast', cross, S)) + 18} fontSize="11.5" style={{ fill: 'var(--ink)' }}>
            <Translated vi={<>hòa tại {preset === 'coulomb' ? `${sig(cross)}×` : `token ${int(cross)}`}</>}>
              tie at {preset === 'coulomb' ? `${sig(cross)}×` : `token ${int(cross)}`}
            </Translated>
          </text>
        </g>
      ) : null}
    </svg>
    </div>
  );
}

function MemoryPlot({ length, batch }: { length: number; batch: number }) {
  const W = 300;
  const H = 170;
  const pad = { l: 46, r: 10, t: 10, b: 34 };
  const lo = 1;
  const hi = 65536;
  const yLo = kvBytes(lo, 1) / GiB;
  const yHi = kvBytes(hi, 256) / GiB;
  const X = (v: number) => pad.l + (Math.log10(v / lo) / Math.log10(hi / lo)) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(v) - Math.log10(yLo)) / (Math.log10(yHi) - Math.log10(yLo))) * (H - pad.t - pad.b);
  const line = (b: number) => `${X(lo)},${Y(kvBytes(lo, b) / GiB)} ${X(hi)},${Y(kvBytes(hi, b) / GiB)}`;
  const yt = [1 / 1024, 1, 1024];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('KV-cache memory against length', 'Bộ nhớ KV cache theo độ dài')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {yt.map((v) => (
        <g key={v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 5} y={Y(v) + 4} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
            {v < 1 ? '1 MiB' : v === 1 ? '1 GiB' : '1 TiB'}{/* i18n-ignore: units */}
          </text>
        </g>
      ))}
      {[1, 256, 65536].map((v) => (
        <text key={v} x={X(v)} y={H - pad.b + 14} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
          {v >= 1024 ? v / 1024 + 'k' : v}
        </text>
      ))}
      <text x={(W + pad.l) / 2} y={H - 6} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi="độ dài chuỗi (token)">sequence length (tokens)</Translated>
      </text>
      <polyline points={line(1)} fill="none" stroke="var(--ink-3)" strokeWidth="1.6" />
      {batch > 1 ? <polyline points={line(batch)} fill="none" stroke="var(--ink)" strokeWidth="2" /> : null}
      <circle cx={X(length)} cy={Y(kvBytes(length, batch) / GiB)} r="4" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.8" />
      <text x={X(hi) - 4} y={Y(kvBytes(hi, 1) / GiB) + 14} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi="batch 1">batch 1</Translated>
      </text>
      {batch > 1 ? (
        <text x={X(hi) - 4} y={Y(kvBytes(hi, batch) / GiB) - 6} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink)' }}>
          <Translated vi={<>batch {batch}</>}>batch {batch}</Translated>
        </text>
      ) : null}
    </svg>
  );
}

function Reformulate() {
  const [preset, setPreset] = useState<Preset>('coulomb');
  const c = curves(preset);
  const [pos, setPos] = useState<Record<Preset, number>>({ coulomb: SIZE_SLIDER.coulomb.at, kv: SIZE_SLIDER.kv.at });
  const [si, setSi] = useState(SPEEDS.indexOf(1000));
  const [batch, setBatch] = useState(8);
  const S = SPEEDS[si];
  const x = SIZE_SLIDER[preset].of(pos[preset]);
  const cross = crossover(c, S);
  const adv = advantage(c, x, S);
  const slopeDirect = slope((v) => time(c, 'direct-fast', v, S), x);
  const slopeReformed = slope((v) => time(c, 'reformed-fast', v, S), x);
  const n = NAMES[preset];

  return (
    <Bench
      title={tr('Reformulate or accelerate?', 'Phát biểu lại hay tăng tốc?')}
      task={tr('Drag the speed-up and watch what moves. Then drag the size past the tie.', 'Kéo hệ số tăng tốc và xem cái gì dịch chuyển. Sau đó kéo kích thước vượt qua điểm hòa.')}
      foot={
        preset === 'coulomb' ? (
          <Translated
            vi={
              <>
                Hệ tham chiếu (kích thước 1), hằng số của chúng tôi: {int(COULOMB.atoms)} nguyên tử, {int(COULOMB.points)}{' '}
                điểm lưới, một vùng lân cận cutoff gồm {int(COULOMB.neighbourhood)} nguyên tử. Một đơn vị cho mỗi cặp nguyên
                tử - điểm lưới. Mô hình: <code>src/sim/formulation.ts</code>.
              </>
            }
          >
            <>
              Reference system (size 1), our constants: {int(COULOMB.atoms)} atoms, {int(COULOMB.points)} grid points, a
              cutoff neighbourhood of {int(COULOMB.neighbourhood)} atoms. One unit per atom-grid point pair. Model:{' '}
              <code>src/sim/formulation.ts</code>.
            </>
          </Translated>
        ) : (
          <Translated
            vi={
              <>
                Một lớp với độ rộng ẩn {int(LLAMA2_7B.hidden)} (ví dụ Llama 2 7B của sách, tr. 506); một đơn vị cho mỗi phép
                nhân-cộng, bỏ qua softmax. Tính lại: 3·T·d² + 2·T²·d. Có cache: 3·d² + 2·T·d. Mô hình:{' '}
                <code>src/sim/formulation.ts</code>.
              </>
            }
          >
            <>
              One layer of hidden width {int(LLAMA2_7B.hidden)} (the book&rsquo;s Llama 2 7B example, p. 506); one unit per
              multiply-add, softmax ignored. Recompute: 3·T·d² + 2·T²·d. Cached: 3·d² + 2·T·d. Model:{' '}
              <code>src/sim/formulation.ts</code>.
            </>
          </Translated>
        )
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Bài toán">Problem</Translated>
          <span className="seg" role="group" aria-label={tr('Problem', 'Bài toán')} style={WRAP}>
            {(['coulomb', 'kv'] as Preset[]).map((p) => (
              <button key={p} aria-pressed={preset === p} onClick={() => setPreset(p)}>
                {PRESET_LABEL[p]}
              </button>
            ))}
          </span>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <span>
            <Translated vi="Kích thước:">Size:</Translated> <b data-value="size">{n.unit(x)}</b>
          </span>
          <input
            type="range"
            min={0}
            max={SIZE_SLIDER[preset].max}
            value={pos[preset]}
            aria-label={tr('Problem size', 'Kích thước bài toán')}
            onChange={(e) => setPos({ ...pos, [preset]: Number(e.target.value) })}
            style={{ width: '14rem' }}
          />
        </label>
        <label className="field">
          <span>
            <Translated vi="Hệ số tăng tốc của GPU so với CPU:">GPU speed-up over the CPU:</Translated> <b data-value="speedup">{S}×</b>
          </span>
          <input type="range" min={0} max={SPEEDS.length - 1} value={si} aria-label={tr('GPU speed-up', 'Hệ số tăng tốc GPU')} onChange={(e) => setSi(Number(e.target.value))} style={{ width: '14rem' }} />
        </label>
        {preset === 'kv' ? (
          <label className="field">
            <span>
              <Translated vi={<>Batch: <b data-value="batch">{batch}</b> chuỗi</>}>
                Batch: <b data-value="batch">{batch}</b> sequences
              </Translated>
            </span>
            <input type="range" min={0} max={8} value={Math.log2(batch)} aria-label={tr('Batch', 'Batch')} onChange={(e) => setBatch(2 ** Number(e.target.value))} style={{ width: '10rem' }} />
          </label>
        ) : null}
      </div>
      <div className="cols plot">
        <CostPlot preset={preset} x={x} S={S} cross={cross} />
        <div className="panel" aria-live="polite" data-readout="formulation" data-crossover={cross ?? ''} data-slope-direct={slopeDirect.toFixed(3)} data-slope-reformed={slopeReformed.toFixed(3)}>
          <Translated
            vi={
              <p style={{ margin: 0 }}>
                Tại {n.unit(x)}, {n.directText} trên GPU{' '}
                {Math.abs(adv - 1) < 0.005 ? (
                  <>
                    mất <b data-value="advantage">cùng thời gian</b> như
                  </>
                ) : (
                  <>
                    <b data-value="advantage">{adv > 1 ? `nhanh hơn ${sig(adv)} lần` : `chậm hơn ${sig(1 / adv)} lần`}</b> so với
                  </>
                )}{' '}
                {n.reformedText} trên CPU.
              </p>
            }
          >
            <p style={{ margin: 0 }}>
              At {n.unit(x)}, {n.directText} on the GPU{' '}
              {Math.abs(adv - 1) < 0.005 ? (
                <>
                  takes <b data-value="advantage">the same time</b> as
                </>
              ) : (
                <>
                  is <b data-value="advantage">{adv > 1 ? `${sig(adv)} times faster` : `${sig(1 / adv)} times slower`}</b> than
                </>
              )}{' '}
              {n.reformedText} on the CPU.
            </p>
          </Translated>
          <Translated
            vi={
              <p style={{ margin: '0.6rem 0 0' }}>
                Hai bên hòa tại <b data-value="crossover">{cross == null ? 'không kích thước nào được hiển thị' : n.unit(cross)}</b>
                {cross != null ? `; vượt quá đó, ${n.reformedText} thắng ngay cả trên bộ xử lý chậm hơn.` : '.'}
              </p>
            }
          >
            <p style={{ margin: '0.6rem 0 0' }}>
              They tie at <b data-value="crossover">{cross == null ? 'no size shown' : n.unit(cross)}</b>
              {cross != null ? `; beyond it ${n.reformedText} wins even on the slower processor.` : '.'}
            </p>
          </Translated>
          <Translated
            vi={
              <p style={{ margin: '0.6rem 0 0' }}>
                Độ dốc trên đồ thị này tại kích thước này: {n.direct} <b data-value="slope-direct">{slopeDirect.toFixed(2)}</b>,{' '}
                {n.reformed} <b data-value="slope-reformed">{slopeReformed.toFixed(2)}</b>. Hệ số tăng tốc dịch mọi đường
                cong GPU xuống cùng một lượng và giữ nguyên độ dốc của nó.
              </p>
            }
          >
            <p style={{ margin: '0.6rem 0 0' }}>
              Slope on this plot at this size: {n.direct} <b data-value="slope-direct">{slopeDirect.toFixed(2)}</b>,{' '}
              {n.reformed} <b data-value="slope-reformed">{slopeReformed.toFixed(2)}</b>. The speed-up shifts every GPU curve
              down by the same amount and leaves its slope alone.
            </p>
          </Translated>
          <table className="data" style={{ margin: '0.7rem 0 0' }}>
            <tbody>
              {SERIES.map((s) => (
                <tr key={s} data-series-row={s}>
                  <td>{seriesName(preset, s, S)}</td>
                  <td className="num">{sci(time(c, s, x, S))}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      {preset === 'kv' ? (
        <div className="cols" style={{ marginTop: '0.9rem', alignItems: 'start' }}>
          <div className="panel" data-readout="kv-memory" data-bytes={kvBytes(x, batch)}>
            <Translated
              vi={
                <p style={{ margin: 0 }}>
                  <b>Thứ mà cache phải trả thay vào đó</b>: bộ nhớ. Pt. 20.8 với Llama 2 7B của sách (32 lớp, độ rộng ẩn
                  4096, 2 byte): {int(kvBytes(1, 1) / 1024)} KiB cho mỗi token của mỗi chuỗi.
                </p>
              }
            >
              <p style={{ margin: 0 }}>
                <b>What the cache costs instead</b>: memory. Eq. 20.8 with the book&rsquo;s Llama 2 7B (32 layers, hidden
                width 4096, 2 bytes): {int(kvBytes(1, 1) / 1024)} KiB per token per sequence.
              </p>
            </Translated>
            <Translated
              vi={
                <p style={{ margin: '0.5rem 0 0' }}>
                  {batch} × {int(x)} token: <b data-value="kv-gib">{sig(kvBytes(x, batch) / GiB)} GiB</b>, tăng tỷ lệ
                  thuận với cả hai. Chính bộ nhớ đó, cùng với trọng số, là thứ giới hạn batch (chương 20, tr. 505) ở mục{' '}
                  <a href={href('ch22', '22.5')}>22.5</a>.
                </p>
              }
            >
              <p style={{ margin: '0.5rem 0 0' }}>
                {batch} × {int(x)} tokens: <b data-value="kv-gib">{sig(kvBytes(x, batch) / GiB)} GiB</b>, growing in
                proportion to both. That memory, with the weights, is what limits the batch (chapter 20, p. 505) in
                section <a href={href('ch22', '22.5')}>22.5</a>.
              </p>
            </Translated>
          </div>
          <div className="panel">
            <MemoryPlot length={x} batch={batch} />
          </div>
        </div>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const TIE = tieFactor(10, 2, 1);
const EX_S = 1000;
const EX_ADV = advantage(curves('coulomb'), 1, EX_S);
const EX_CROSS = crossover(curves('coulomb'), EX_S)!;

export const questionsS224: Record<string, Question> = {
  'ch22/22.4/q-what-relaxed': {
    kind: 'choice',
    id: 'ch22/22.4/q-what-relaxed',
    prompt: (
      <Translated vi="Phép phát biểu lại nào dưới đây hy sinh độ chính xác để giảm độ phức tạp?">
        <>Which of these reformulations gives up exactness to cut the complexity?</>
      </Translated>
    ),
    options: [
      { label: tr('Cutoff summation: far atoms are not summed one by one but treated together.', 'Tổng cutoff: các nguyên tử ở xa không được cộng từng cái một mà được xử lý chung.'), correct: true },
      {
        label: tr('The KV cache: each decode step reuses the keys and values of earlier steps.', 'KV cache: mỗi bước decode dùng lại key và value của các bước trước.'),
        why: tr('It reuses the very numbers it would recompute, so nothing is given up. It spends memory instead.', 'Nó dùng lại đúng những con số mà nó lẽ ra phải tính lại, nên không hy sinh gì cả. Thay vào đó nó tốn bộ nhớ.'),
      },
      {
        label: tr('Splitting a long sum across the warps of a block and adding the partial sums.', 'Chia một tổng dài cho các warp của một khối rồi cộng các tổng riêng phần.'),
        why: tr('That changes the order of the additions, and so the rounding, but not how many there are.', 'Điều đó đổi thứ tự các phép cộng, và do đó đổi cách làm tròn, nhưng không đổi số lượng phép cộng.'),
      },
      {
        label: tr('Batching the requests of several users into one matrix product per layer.', 'Gom yêu cầu của nhiều người dùng thành một phép nhân ma trận cho mỗi lớp.'),
        why: tr('It changes how the work is grouped, not the result any user gets.', 'Nó đổi cách công việc được gom nhóm, chứ không đổi kết quả mà bất kỳ người dùng nào nhận được.'),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Chỉ có tổng cutoff thay đổi thứ được tính: đóng góp riêng lẻ của các nguyên tử ở xa bị bỏ đi và được xử lý bằng
            một phương pháp gộp rẻ hơn, và lượng việc giảm từ bình phương thể tích xuống tuyến tính. KV cache đổi chi phí từ
            tính lại sang lưu trữ mà không thay đổi một con số nào.
          </p>
        }
      >
        <p>
          Only cutoff summation changes what is computed: the far atoms&rsquo; individual contributions are dropped and
          handled by a cheaper collective method, and the work falls from the square of the volume to linear. The KV cache
          changes the cost from recomputation to storage without changing a single number.
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.4, p. 538; ch. 20 §20.4, pp. 488–492',
    vi: { source: 'ch. 22 §22.4, tr. 538; ch. 20 §20.4, tr. 488–492' },
  },
  'ch22/22.4/q-kv-cost': {
    kind: 'choice',
    id: 'ch22/22.4/q-kv-cost',
    prompt: (
      <Translated
        vi={
          <>
            KV cache làm cho bước sinh token T rẻ hơn khoảng T lần so với tính lại toàn bộ tiền tố. Thay vào đó nó tốn thứ
            gì, và điều đó sẽ gây khó khăn ở đâu tiếp theo?
          </>
        }
      >
        <>
          The KV cache makes the step that generates token T about T times cheaper than recomputing the whole prefix.
          What does it spend instead, and where does that bite next?
        </>
      </Translated>
    ),
    options: [
      {
        label: tr('Memory that grows with the length of each sequence and with the number of sequences, which caps how many fit in a batch.', 'Bộ nhớ tăng theo độ dài của mỗi chuỗi và theo số chuỗi, điều này giới hạn số chuỗi vừa trong một batch.'),
        correct: true,
      },
      {
        label: tr('Arithmetic: appending each new key and value to the cache costs about as much as recomputing the prefix it replaces.', 'Tính toán: nối mỗi key và value mới vào cache tốn gần bằng việc tính lại tiền tố mà nó thay thế.'),
        why: tr('An append copies one row of K and one of V per layer. The recomputation it replaces is about T times the whole cached step.', 'Một lần nối chép một hàng của K và một hàng của V cho mỗi lớp. Phần tính lại mà nó thay thế lớn khoảng T lần toàn bộ bước có cache.'),
      },
      {
        label: tr('Accuracy, because the cached keys and values were computed for a shorter prefix and grow stale as tokens arrive.', 'Độ chính xác, vì key và value trong cache được tính cho một tiền tố ngắn hơn và trở nên lỗi thời khi các token mới đến.'),
        why: tr('Under the causal mask the keys and values of earlier tokens never change (ch. 20 p. 489), so the cache is exact.', 'Dưới mặt nạ nhân quả (causal mask), key và value của các token trước không bao giờ thay đổi (ch. 20 tr. 489), nên cache là chính xác.'),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Pt. 20.8: b · N · 2 · l · h_q · d · p byte cho b chuỗi có độ dài N. Với số liệu Llama 2 7B của sách, một chuỗi
            4096 token chiếm 2 GiB cache. Mỗi chuỗi trong một batch có cache riêng, nên cache quyết định batch ở mục{' '}
            <a href={href('ch22', '22.5')}>22.5</a> có thể lớn đến đâu. Cache còn có hệ quả thứ hai: mỗi bước giờ là một phép
            nhân vector-ma trận, mà chương 20 gọi là bị giới hạn bởi băng thông, và đó là lý do còn lại khiến mục 22.5 dùng
            batch.
          </p>
        }
      >
        <p>
          Eq. 20.8: b · N · 2 · l · h_q · d · p bytes for b sequences of length N. With the book&rsquo;s Llama 2 7B
          figures, one sequence of 4096 tokens holds 2 GiB of cache. Every sequence in a batch has its own, so the cache
          decides how large the batch of section <a href={href('ch22', '22.5')}>22.5</a> can be. The cache has a second
          consequence: each step is now a vector-matrix product, which chapter 20 calls bandwidth-bound, and that is the
          other reason section 22.5 batches.
        </p>
      </Translated>
    ),
    source: 'ours, from ch. 20 §20.4, pp. 488–492, and §20.6, pp. 505–506 (Eq. 20.8)',
    vi: { source: 'của chúng tôi, từ ch. 20 §20.4, tr. 488–492, và §20.6, tr. 505–506 (Pt. 20.8)' },
  },
  'ch22/22.4/q-why-domain': {
    kind: 'choice',
    id: 'ch22/22.4/q-why-domain',
    prompt: (
      <Translated
        vi={
          <>
            Ba đề xuất để tăng tốc decode với ngữ cảnh dài. Đề xuất nào là một phép phát biểu lại theo nghĩa của sách: một
            thay đổi mà chỉ kiến thức chuyên ngành mới đánh giá được nó có an toàn hay không, như tổng cutoff đã cần đến vật
            lý?
          </>
        }
      >
        <>
          Three proposals to speed up decoding at long context. Which one is a reformulation in the book&rsquo;s sense:
          a change whose safety only knowledge of the domain can judge, as cutoff summation needed physics?
        </>
      </Translated>
    ),
    options: [
      {
        label: tr('Attend only to the most recent 4,096 tokens. Whether the outputs survive is a question about the trained model; skipping the older keys is then a kernel job.', 'Chỉ attention tới 4,096 token gần nhất. Đầu ra có còn dùng được hay không là câu hỏi về mô hình đã huấn luyện; việc bỏ qua các key cũ hơn sau đó là việc của kernel.'),
        correct: true,
      },
      {
        label: tr('Split each attention row over more thread blocks and combine their partial softmaxes at the end. It changes how the work is cut, so it needs a judgement about the model.', 'Chia mỗi hàng attention cho nhiều khối luồng hơn và kết hợp các softmax riêng phần ở cuối. Nó thay đổi cách cắt công việc, nên cần một đánh giá về mô hình.'),
        why: tr('The combined result is the same attention up to rounding. That is a decomposition, which section 22.2’s criteria judge without knowing anything about the model.', 'Kết quả kết hợp chính là cùng attention đó, sai khác chỉ do làm tròn. Đó là một phép phân rã, mà các tiêu chí của mục 22.2 đánh giá được mà không cần biết gì về mô hình.'),
      },
      {
        label: tr('Store the KV cache in a layout whose reads coalesce. Reading keys in a different order changes what the model sees, so it needs a judgement about the model.', 'Lưu KV cache theo một bố cục mà các lần đọc được hợp nhất. Đọc key theo thứ tự khác làm thay đổi thứ mô hình nhìn thấy, nên cần một đánh giá về mô hình.'),
        why: tr('The same values reach the same arithmetic; only where the bytes sit changes. Chapter 6’s rules decide this one.', 'Cùng các giá trị đi vào cùng các phép tính; chỉ vị trí của các byte thay đổi. Các quy tắc của chương 6 quyết định trường hợp này.'),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Cutoff với bin cần cả hai nửa. Biết rằng các nguyên tử ở xa đóng góp rất ít, và tác động của chúng có thể được xử
            lý gộp, là vật lý. Biến phần tính toán còn lại thành một kernel lấy lưới làm trung tâm, biết cần quét những nguyên
            tử nào, là một phép phân rã và một cấu trúc dữ liệu. Một cửa sổ trên các key có cùng hình dạng: nó bỏ bớt phần
            việc mà phát biểu chính xác vẫn làm, và chỉ hành vi của mô hình, đo trên đầu ra của nó, mới cho biết điều đó có
            chấp nhận được hay không. Hai đề xuất kia tính cùng một thứ và không cần đánh giá chuyên ngành nào cả. Cửa sổ là
            ví dụ của chúng tôi, không phải của sách.
          </p>
        }
      >
        <p>
          Cutoff binning needs both halves. Knowing that far atoms contribute little, and that their effect can be handled
          collectively, is physics. Turning the remaining computation into a grid-centric kernel that knows which atoms to
          scan is a decomposition and a data structure. A window over the keys has the same shape: it drops work the exact
          formulation does, and only the model&rsquo;s behaviour, measured on its outputs, can say whether that is
          acceptable. The other two proposals compute the same thing and need no domain judgement at all. The window is
          our example, not the book&rsquo;s.
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.4, p. 538; the attention example is ours',
    vi: { source: 'ch. 22 §22.4, tr. 538; ví dụ attention là của chúng tôi' },
  },
};

export function S224() {
  const q = questionsS224;
  return (
    <>
      <Translated
        vi={
          <p>
            Sách gọi phát biểu bài toán là bước có lẽ mang tính quyết định nhất khi phát triển một ứng dụng song song, và dẫn
            bài báo của Wing về tư duy điện toán cho nhận định này. Các mục trước đã chọn thuật toán và cách phân rã cho một bài
            toán như nó được đưa ra. Mục này hỏi liệu bài toán có nên được đưa ra theo cách đó hay không.
          </p>
        }
      >
        <p>
          The book calls problem formulation arguably the most consequential step in developing a parallel application, and
          cites Wing&rsquo;s paper on computational thinking for it. The earlier sections chose algorithms and decompositions
          for a problem as given. This one asks whether the problem should be given that way at all.
        </p>
      </Translated>
      <Source section="22.4" pages="538" />
      <Translated
        vi={
          <p>
            Theo sách, điều không nên làm là lấy một phương pháp số tổng quát, ánh xạ bài toán chuyên ngành lên nó từng bước
            một, rồi cài đặt các bước bằng những kỹ năng trong cuốn sách này. Con đường đó thường dẫn đến độ phức tạp tính toán
            cao, khả năng song song hạn chế, hoặc cả hai. Thay vào đó, hãy nghĩ lại phương pháp và nới lỏng một số ràng buộc về
            cách nó được thực hiện.
          </p>
        }
      >
        <p>
          What not to do, according to the book, is to take a generic numerical method, map the domain problem onto it
          step by step, and implement the steps with the skills of this book. That route tends to give high computational
          complexity, limited parallelism, or both. What to do instead is rethink the method and relax some of the
          constraints on how it is carried out.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Ví dụ của sách là cutoff với bin ở chương 21. Quyết định rằng có thể đổi một phần độ chính xác để giảm mạnh độ phức
            tạp đòi hỏi chuyên môn trong lĩnh vực. Làm cho kết quả chạy nhanh thì cần khoa học máy tính: một cách phân rã lấy lưới
            làm trung tâm và các kỹ thuật chia bin cho mỗi khối biết cần quét những nguyên tử nào.
          </p>
        }
      >
        <p>
          The book&rsquo;s example is cutoff binning from chapter 21. Deciding that some accuracy can be traded for a
          dramatic drop in complexity takes domain expertise. Making the result fast takes computer science: a grid-centric
          decomposition and the binning techniques that tell each block which atoms to scan.
        </p>
      </Translated>
      <Source section="22.4" pages="538" figs="ch. 21 §21.5" />

      <h2><Translated vi="Một hằng số nhân đối đầu với một độ dốc">A constant factor against a slope</Translated></h2>
      <Translated
        vi={
          <p>
            Sự khác biệt giữa tăng tốc một phương pháp và phát biểu lại nó lộ ra ngay khi bài toán lớn lên. Hãy dự đoán trước.
          </p>
        }
      >
        <p>
          The difference between accelerating a method and reformulating it shows up as soon as the problem grows. Predict
          it first.
        </p>
      </Translated>
      <Predict
        id="ch22/22.4/p-crossover"
        prompt={
          <Translated
            vi={
              <>
                Ở kích thước bài toán hiện tại, một GPU chạy tổng trực tiếp nhanh hơn 10 lần so với một CPU chạy tổng cutoff.
                Chi phí trực tiếp tăng theo bình phương thể tích, chi phí cutoff tăng tuyến tính. Thể tích phải tăng lên bao
                nhiêu lần thì hai bên mất cùng thời gian?
              </>
            }
          >
            <>
              At today&rsquo;s problem size, a GPU running direct summation is 10 times faster than a CPU running cutoff
              summation. Direct cost grows with the square of the volume, cutoff cost linearly. By what factor must the
              volume grow before the two take the same time?
            </>
          </Translated>
        }
        unit="×"
      >
        {(v) => (
          <Translated
            vi={
              <p>
                <b>{sig(TIE)}</b>. Viết thời gian trực tiếp là k·V² và thời gian cutoff là 10·k·V; chúng gặp nhau tại V = 10.{' '}
                {Math.abs(v - TIE) < 0.05 * TIE
                  ? 'Đó đúng là đáp án của bạn.'
                  : v > TIE
                    ? 'Cả hai chi phí đều tăng, nên tỷ số giữa chúng chỉ tăng tỷ lệ thuận với thể tích: lợi thế 10 lần biến mất ở thể tích gấp 10, không phải 100.'
                    : 'Hệ số √10 sẽ đúng nếu chi phí cutoff đứng yên. Nhưng nó cũng tăng, nên tỷ số giữa hai bên chỉ tăng tỷ lệ thuận với thể tích.'}{' '}
                Trong trình khám phá bên dưới, hệ số tăng tốc {EX_S}× cho đúng trường hợp này: tổng trực tiếp của GPU nhanh hơn{' '}
                {sig(EX_ADV)} lần ở kích thước 1, và các đường cong hòa nhau tại {sig(EX_CROSS)}× thể tích.
              </p>
            }
          >
            <p>
              <b>{sig(TIE)}</b>. Write the direct time as k·V² and the cutoff time as 10·k·V; they meet where V = 10.{' '}
              {Math.abs(v - TIE) < 0.05 * TIE
                ? 'That is what you had.'
                : v > TIE
                  ? 'Both costs grow, so the ratio between them grows only in proportion to the volume: a lead of 10 is gone at 10 times the volume, not 100.'
                  : 'A factor of √10 would be right if the cutoff cost stayed fixed. It grows too, so the ratio between the two grows only in proportion to the volume.'}{' '}
              In the explorer below, a speed-up of {EX_S}× gives exactly this case: the GPU&rsquo;s direct sum is{' '}
              {sig(EX_ADV)} times faster at size 1, and the curves tie at {sig(EX_CROSS)}× the volume.
            </p>
          </Translated>
        )}
      </Predict>
      <Reformulate />
      <Translated
        vi={
          <p>
            Kéo hệ số tăng tốc xuống từ 1000× rồi kéo lên lại. Mỗi đường cong GPU trượt theo, và điểm hòa dời về kích thước lớn
            hơn khi hệ số tăng tốc tăng, nhưng độ dốc trong phần số liệu không đổi: một bộ xử lý nhanh hơn chia chi phí cho một
            hằng số. Phép phát biểu lại thay đổi độ dốc, nên vượt qua một kích thước nào đó nó thắng ngay cả trên bộ xử lý chậm
            hơn, và không hằng số nào trì hoãn được kích thước đó mãi mãi. Với bản đồ tĩnh điện, mỗi hệ số 10 về tốc độ phần cứng
            chỉ dời điểm hòa đi một hệ số 10 về thể tích.
          </p>
        }
      >
        <p>
          Drag the speed-up down from 1000× and back up. Each GPU curve slides with it, and the tie moves to larger sizes
          as the speed-up grows, but the slopes in the readout do not change: a faster processor divides the cost by a
          constant. The reformulation changes the
          slope, so beyond some size it wins even on the slower processor, and no constant can postpone that size forever.
          For the electrostatic map, every factor of 10 in hardware speed only moves the tie by a factor of 10 in volume.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Chi phí là số phép toán với các hằng số do chúng tôi chọn, ghi dưới đồ thị. Không tính lưu lượng bộ nhớ, điều quan
              trọng với một bước decode có cache: nó bị giới hạn bởi băng thông. Phần mất độ chính xác của cutoff và chi phí xử lý
              các nguyên tử ở xa không được mô hình hóa. Cùng một hệ số tăng tốc áp dụng cho cả hai phương pháp trên GPU.
            </p>
          }
        >
          <p>
            Costs are operation counts with constants we chose, stated under the plot. No memory traffic, which matters for
            a cached decode step: it is bandwidth-bound. The cutoff&rsquo;s loss of accuracy and the cost of handling far
            atoms are not modelled. The same speed-up applies to both methods on the GPU.
          </p>
        </Translated>
      </Simplification>

      <h2><Translated vi="Một phép phát biểu lại bạn vẫn đang dùng">A reformulation you already use</Translated></h2>
      <Translated
        vi={
          <p>
            KV cache của chương 20 là cùng nước đi đó trong chính lĩnh vực của bạn. Không có nó, việc sinh token T phải tính lại K
            và V cho toàn bộ tiền tố và attention trên tất cả chúng, ở mỗi bước. Có nó, mỗi bước tính một hàng mới của Q, K và V,
            nối K và V vào cache, và attention với một hàng. Theo cách đếm dùng ở đây, bước có cache rẻ hơn đúng T lần, nên độ dốc
            giảm đi một ở mọi độ dài, và một GPU tính lại tiền tố nhanh hơn S lần so với một CPU có cache sẽ thua nó từ token S trở
            đi. Hãy chuyển trình khám phá sang bài toán thứ hai.
          </p>
        }
      >
        <p>
          Chapter 20&rsquo;s KV cache is the same move in your own field. Without it, generating token T recomputes K and V
          for the whole prefix and the attention over all of it, every step. With it, each step computes one new row of Q,
          K and V, appends K and V to the cache, and attends with one row. In the count used here the cached step is
          exactly T times cheaper, so the slope drops by one at every length, and a GPU recomputing the prefix S times
          faster than a CPU with a cache loses to it from token S on. Switch the explorer to the second problem.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Cầu nối này là của chúng tôi. KV cache nằm ở chương 20 (§20.4, tr. 488–492; Pt. 20.8, tr. 505–506), không phải ở mục
              này.
            </p>
          }
        >
          <p>
            This bridge is ours. The KV cache is in chapter 20 (§20.4, pp. 488–492; Eq. 20.8, pp. 505–506), not in this
            section.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Khác với tổng cutoff, cache không hy sinh gì về mặt số học. Thay vào đó nó tốn bộ nhớ: key và value cho mọi token của
            mọi chuỗi, tăng theo độ dài và theo batch. Chính bộ nhớ đó giới hạn batching ở mục{' '}
            <a href={href('ch22', '22.5')}>22.5</a>, nên phép phát biểu lại đã làm decode rẻ đi cũng là thứ đặt ra ràng buộc tiếp
            theo. Các mục về KV cache của chương 20 là <a href={href('ch20', '20.4')}>20.4</a> và{' '}
            <a href={href('ch20', '20.6')}>20.6</a>.
          </p>
        }
      >
        <p>
          Unlike cutoff summation, the cache gives up nothing numerically. It spends memory instead: keys and values for
          every token of every sequence, growing with length and with batch. That memory is what limits batching in section{' '}
          <a href={href('ch22', '22.5')}>22.5</a>, so the reformulation that made decoding cheap is also what sets the
          next constraint. Chapter 20&rsquo;s KV-cache sections are{' '}
          <a href={href('ch20', '20.4')}>20.4</a> and <a href={href('ch20', '20.6')}>20.6</a>.
        </p>
      </Translated>

      <h2><Translated vi="Những cải thiện lớn đến từ đâu">Where the large gains come from</Translated></h2>
      <Translated
        vi={
          <p>
            Nhận định khép lại mục này của sách là phát biểu bài toán là nơi sinh ra những lợi thế hiệu năng lớn nhất, và cả
            những khả năng thực sự mới: chẳng hạn một mô phỏng độ trung thực cao cho một hệ sinh hóa có kích thước mà các phương
            pháp truyền thống không thể đạt tới. Nó cần hiểu biết sâu từ cả lĩnh vực chuyên ngành lẫn khoa học máy tính, và sách
            cho rằng thành quả xứng đáng với công sức bỏ ra.
          </p>
        }
      >
        <p>
          The book&rsquo;s closing claim for the section is that formulation is where the largest performance advantages,
          and genuinely new capabilities, come from: a high-fidelity simulation of a biochemical system of a size that
          traditional methods could not reach, for example. It needs insight from both the domain and computer science, and
          the book says the payoff is worth the effort.
        </p>
      </Translated>
      <Source section="22.4" pages="538" />
      <Ask q={q['ch22/22.4/q-what-relaxed']} />
      <Ask q={q['ch22/22.4/q-kv-cost']} />
      <Ask q={q['ch22/22.4/q-why-domain']} />
    </>
  );
}
