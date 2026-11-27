import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';
import {
  BF16, E2M1, E4M3, E5M2, FP16, FP32,
  anatomy, binade, bitString, classify, decode, fromFields, makeFormat, maxFinite, minNormal, minSubnormal,
  round, show, topCode, ulp, values,
  type Classified, type Format, type Subnormals,
} from '../../sim/minifloat';

// ---------------------------------------------------------------- small pieces
const minus = (n: number | string) => String(n).replace(/-/g, '−');
function parseValue(t: string): number | null {
  const s = t.trim().toLowerCase().replace(/[−–]/g, '-').replace(/[,_\s]/g, '');
  if (s === '') return null;
  const p = /^2\^(-?\d+(\.\d+)?)$/.exec(s);
  if (p) return 2 ** Number(p[1]);
  const v = Number(s);
  return Number.isNaN(v) ? null : v;
}
const P_GAP = 'appA/A.2/p-gap';

/** Width of a container in px, so a drawing can be laid out for it instead of scaled. */
function useWidth(initial = 900) {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(initial);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver((entries) => setW(Math.round(entries[0].contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, Math.max(300, Math.min(900, w))] as const;
}

// ---------------------------------------------------------------- explorer: the number line (Figs. A.4–A.7)
const CONVS: { id: Subnormals; label: string; short: string; fig: string }[] = [
  {
    id: 'nozero',
    get label() {
      return tr('no zero', 'không có số 0');
    },
    get short() {
      return tr('no zero', 'không 0');
    },
    fig: 'Fig. A.5',
  },
  {
    id: 'abrupt',
    get label() {
      return tr('abrupt underflow', 'underflow đột ngột');
    },
    get short() {
      return tr('abrupt', 'đột ngột');
    },
    fig: 'Fig. A.6',
  },
  {
    id: 'denorm',
    get label() {
      return tr('denormals', 'số denormal');
    },
    get short() {
      return tr('denorm', 'denorm');
    },
    fig: 'Fig. A.7',
  },
];

function Lines({
  fmts,
  conv,
  lo,
  hi,
  probe,
  showProbe,
  label,
}: {
  fmts: Record<Subnormals, Format>;
  conv: Subnormals;
  lo: number;
  hi: number;
  probe: number;
  /** False until the prediction above is committed: the rings would give its answer away. */
  showProbe: boolean;
  label: string;
}) {
  const [ref, W] = useWidth();
  const narrow = W < 560;
  const L0 = narrow ? 64 : 132;
  const R0 = W - 18;
  const rowH = 44;
  const top = 18;
  const H = top + rowH * 3 + 30;
  const X = (v: number) => L0 + ((v - lo) / (hi - lo)) * (R0 - L0);
  const f0 = fmts.denorm;
  // Ticks: 0 and the powers of two that start intervals, skipping labels that would collide.
  const ticks: number[] = [0];
  for (let k = Math.round(Math.log2(minNormal(fmts.nozero))); 2 ** k <= hi; k++) if (2 ** k >= lo) ticks.push(2 ** k);
  if (hi <= 8) for (let i = 3; i <= hi; i += 2) ticks.push(i);
  ticks.sort((p, q) => p - q);
  const shown: number[] = [];
  for (const t of ticks) if (!shown.length || X(t) - X(shown[shown.length - 1]) >= 30) shown.push(t);
  const probeIn = showProbe && probe >= lo && probe <= hi;
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={label} style={{ maxWidth: '100%', height: 'auto', display: 'block' }}>
        {CONVS.map((c, r) => {
          const f = fmts[c.id];
          const y = top + rowH * r + rowH / 2;
          const pos = values(f).filter((v) => v >= 0);
          const inView = pos.filter((v) => v >= lo && v <= hi);
          const sel = c.id === conv;
          const smallest = pos.find((v) => v > 0);
          const near = round(f, probe);
          return (
            <g key={c.id} data-conv={c.id} data-count={pos.filter((v) => v > 0).length} opacity={sel ? 1 : 0.55}>
              <text x={4} y={y + 4} fontSize="12" fontWeight={sel ? 650 : 400}>
                {narrow ? c.short : c.label}
              </text>
              {smallest !== undefined && smallest > lo ? (
                <rect x={X(Math.max(lo, 0))} y={y - 7} width={Math.max(0, X(Math.min(smallest, hi)) - X(Math.max(lo, 0)))} height={14}
                  style={{ fill: 'var(--rule-soft)', stroke: 'var(--rule)' }} strokeDasharray="3 2" />
              ) : null}
              <line x1={L0} x2={R0} y1={y} y2={y} style={{ stroke: 'var(--ink-3)' }} />
              {inView.map((v) => (
                <circle key={v} cx={X(v)} cy={y} r={sel ? 3.6 : 3} style={{ fill: 'var(--ink)' }} />
              ))}
              {!pos.includes(0) && lo <= 0 ? (
                <text x={X(0)} y={y + 4} fontSize="12" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>×</text>
              ) : null}
              {probeIn && near >= lo && near <= hi ? (
                <circle cx={X(near)} cy={y} r={7} style={{ fill: 'none', stroke: 'var(--ink)' }} strokeWidth={1.6} />
              ) : null}
            </g>
          );
        })}
        <line x1={L0} x2={R0} y1={H - 26} y2={H - 26} style={{ stroke: 'var(--rule)' }} />
        {shown.map((t) => (
          <g key={t}>
            <line x1={X(t)} x2={X(t)} y1={H - 29} y2={H - 23} style={{ stroke: 'var(--ink-3)' }} />
            <text x={X(t)} y={H - 9} fontSize="11" textAnchor="middle">
              {t === 0 || t >= 1 / 16 ? String(t) : `2^${minus(Math.log2(t))}`}
            </text>
          </g>
        ))}
        {probeIn ? (
          <g>
            <line x1={X(probe)} x2={X(probe)} y1={top - 4} y2={H - 26} style={{ stroke: 'var(--ink-2)' }} strokeDasharray="4 3" />
            <text x={X(probe) + 4} y={top + 6} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
              x
            </text>
          </g>
        ) : null}
        <text x={R0} y={top - 2} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
          <Translated vi={<>định dạng {f0.name}, các giá trị dương (giá trị âm đối xứng với chúng)</>}>
            <>{f0.name} format, positive values (negatives mirror them)</>
          </Translated>
        </text>
      </svg>
    </div>
  );
}

function NumberLineBench() {
  const { progress } = useProgress();
  const revealed = progress.answers[P_GAP] != null;
  const [e, setE] = useState(2);
  const [m, setM] = useState(2);
  const [conv, setConv] = useState<Subnormals>('nozero');
  const [probe, setProbe] = useState(0.2);
  const [text, setText] = useState('0.2');
  const fmts = useMemo(
    () => ({ nozero: makeFormat(e, m, 'nozero'), abrupt: makeFormat(e, m, 'abrupt'), denorm: makeFormat(e, m, 'denorm') }),
    [e, m],
  );
  const f = fmts[conv];
  const max = maxFinite(f);
  const xmax = max * 1.06;
  const zoomHi = 2 ** (3 - f.bias);
  const zoom = zoomHi < xmax / 4;
  const pos = values(f).filter((v) => v > 0);
  const intervals = new Set(pos.filter((v) => v >= minNormal(f)).map(binade)).size;
  const gap = pos[0];
  const spacing = pos.length > 1 ? pos[1] - pos[0] : null;

  const setFromSlider = (v: number) => {
    setProbe(v);
    setText(show(v, 4));
  };
  const setFromText = (t: string) => {
    setText(t);
    const v = parseValue(t);
    if (v !== null && Number.isFinite(v) && v >= 0) setProbe(v);
  };

  // Fig. A.4 for the chosen convention: one row per exponent code.
  const rows: { E: number; rule: ReactNode; vals: number[]; spacing: number | null }[] = [];
  for (let E = 0; E < 2 ** e; E++) {
    if (E > topCode(f)) {
      rows.push({ E, rule: tr('reserved', 'dành riêng'), vals: [], spacing: null });
      continue;
    }
    const a = anatomy(f, fromFields(f, 0, E, 0));
    const vals = Array.from({ length: 2 ** m }, (_, M) => decode(f, fromFields(f, 0, E, M)).value);
    const rule =
      a.rule === 'flushed-pattern' ? (
        tr('all read as 0', 'đều đọc thành 0')
      ) : (
        <>
          {a.rule === 'denormal' ? '0.M' : '1.M'} × 2<sup>{minus(a.scale!)}</sup>
        </>
      );
    rows.push({ E, rule, vals, spacing: a.scale === null ? null : 2 ** (a.scale - m) });
  }

  return (
    <Bench
      title={tr('Number line', 'Trục số')}
      task={tr(
        'Move e and m, switch the convention near 0, and probe a value under all three conventions.',
        'Thay đổi e và m, chuyển quy ước gần 0, và thăm dò một giá trị dưới cả ba quy ước.',
      )}
      foot={
        <Translated
          vi={
            <>
              Định dạng kiểu sách: 1 bit dấu, e bit số mũ dạng excess-(2<sup>e−1</sup> − 1) với mã cao nhất được dành riêng, m
              bit mantissa. e = 2, m = 2 là định dạng 5-bit của sách ở Fig. A.4. Vẽ đúng tỉ lệ; các hộp nét đứt là khoảng
              trống giữa 0 và giá trị dương nhỏ nhất, dấu × đánh dấu số 0 mà định dạng không biểu diễn được, các vòng tròn đánh
              dấu giá trị gần nhất với điểm thăm dò.
            </>
          }
        >
          <>
            Book-style formats: 1 sign bit, e exponent bits in excess-(2<sup>e−1</sup> − 1) with the top code reserved, m
            mantissa bits. e = 2, m = 2 is the book's 5-bit format of Fig. A.4. Drawn to scale; dashed boxes are the gap
            between 0 and the smallest positive value, × marks a 0 the format cannot hold, rings mark the probe's nearest value.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          <Translated vi={<>Số bit mũ e = {e}</>}>
            <>Exponent bits e = {e}</>
          </Translated>
          <input type="range" min="1" max="4" value={e} aria-label={tr('exponent bits', 'số bit mũ')} onChange={(ev) => setE(+ev.target.value)} />
        </label>
        <label className="field">
          <Translated vi={<>Số bit mantissa m = {m}</>}>
            <>Mantissa bits m = {m}</>
          </Translated>
          <input type="range" min="0" max="4" value={m} aria-label={tr('mantissa bits', 'số bit mantissa')} onChange={(ev) => setM(+ev.target.value)} />
        </label>
        <label className="field">
          <Translated vi="Gần 0">Near 0</Translated>
          <span className="seg" role="group" aria-label={tr('Convention near 0', 'Quy ước gần 0')}>
            {CONVS.map((c) => (
              <button key={c.id} aria-pressed={c.id === conv} onClick={() => setConv(c.id)}>
                {c.label}
              </button>
            ))}
          </span>
        </label>
      </div>

      <Lines fmts={fmts} conv={conv} lo={0} hi={xmax} probe={probe} showProbe={revealed} label={tr('Positive representable values under the three conventions', 'Các giá trị dương biểu diễn được dưới ba quy ước')} />
      {zoom ? (
        <>
          <div className="mx-label" style={{ margin: '0.6rem 0 0.2rem' }}>
            <Translated vi={<>Gần 0, phóng to: 0 đến {show(zoomHi)}</>}>
              <>Near 0, magnified: 0 to {show(zoomHi)}</>
            </Translated>
          </div>
          <Lines fmts={fmts} conv={conv} lo={0} hi={zoomHi} probe={probe} showProbe={revealed} label={tr('The region near 0, magnified', 'Vùng gần 0, phóng to')} />
        </>
      ) : null}

      <Translated
        vi={
          <p style={{ margin: '0.7rem 0 0.9rem' }} data-part="summary">
            <b>{CONVS.find((c) => c.id === conv)!.label}:</b> {pos.length} giá trị dương
            {intervals ? <> trong {intervals} khoảng</> : null}; nhỏ nhất {show(gap)}, lớn nhất{' '}
            {show(max)}.{' '}
            {conv === 'nozero' ? '0 không phải là một giá trị. ' : ''}
            {spacing !== null ? (
              <>
                Khoảng trống từ 0 đến giá trị dương nhỏ nhất là {show(gap)}, gấp <b data-part="ratio">{show(gap / spacing)}</b>{' '}
                lần khoảng cách ngay phía trên nó ({show(spacing)}).
              </>
            ) : null}
          </p>
        }
      >
        <p style={{ margin: '0.7rem 0 0.9rem' }} data-part="summary">
          <b>{CONVS.find((c) => c.id === conv)!.label}:</b> {pos.length} positive values
          {intervals ? <> in {intervals} interval{intervals === 1 ? '' : 's'}</> : null}; smallest {show(gap)}, largest{' '}
          {show(max)}.{' '}
          {conv === 'nozero' ? '0 is not a value. ' : ''}
          {spacing !== null ? (
            <>
              The gap from 0 to the smallest positive value is {show(gap)}, <b data-part="ratio">{show(gap / spacing)}</b> times the
              spacing just above it ({show(spacing)}).
            </>
          ) : null}
        </p>
      </Translated>

      <div className="cols">
        <div className="panel">
          <div className="row" style={{ marginBottom: '0.7rem' }}>
            <label className="field">
              <Translated vi="Điểm thăm dò x">Probe x</Translated>
              <input type="range" min="0" max={xmax} step={xmax / 2000} value={Math.min(probe, xmax)} aria-label={tr('probe slider', 'thanh trượt điểm thăm dò')}
                onChange={(ev) => setFromSlider(+ev.target.value)} />
            </label>
            <label className="field">
              <Translated vi="hoặc nhập vào">or type it</Translated>
              <input type="text" inputMode="decimal" value={text} aria-label={tr('probe value', 'giá trị điểm thăm dò')} style={{ width: '7rem' }}
                onChange={(ev) => setFromText(ev.target.value)} />
            </label>
          </div>
          {revealed ? (
            <table className="data" style={{ margin: 0 }} aria-label={tr('The probe under the three conventions', 'Điểm thăm dò dưới ba quy ước')}>
              <thead>
                <tr>
                  <th>
                    <Translated vi="Quy ước">Convention</Translated>
                  </th>
                  <th className="num">
                    <Translated vi="Giá trị gần nhất">Nearest value</Translated>
                  </th>
                  <th className="num">
                    <Translated vi="Sai số làm tròn">Rounding error</Translated>
                  </th>
                </tr>
              </thead>
              <tbody>
                {CONVS.map((c) => {
                  const v = round(fmts[c.id], probe);
                  return (
                    <tr key={c.id} data-conv={c.id} data-nearest={String(v)} data-error={String(Math.abs(v - probe))}
                      style={c.id === conv ? { fontWeight: 650 } : undefined}>
                      <td>{c.label}</td>
                      <td className="num">{show(v)}</td>
                      <td className="num">{show(Math.abs(v - probe), 4)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          ) : (
            <p className="muted" style={{ margin: 0 }} data-part="locked">
              <Translated vi="Hãy chốt dự đoán ở trên để thấy điểm thăm dò trên trục số và các sai số làm tròn của nó.">
                Commit your prediction above to see the probe on the number line and its rounding errors.
              </Translated>
            </p>
          )}
        </div>
        <div className="panel">
          <table className="data" style={{ margin: 0 }} aria-label={tr('Values for each exponent code', 'Các giá trị cho từng mã số mũ')}>
            <thead>
              <tr>
                <th>E</th>
                <th>
                  <Translated vi="Đọc thành">Read as</Translated>
                </th>
                <th>
                  <Translated vi="Các giá trị (từ M = 0 trở lên)">Values (M = 0 up)</Translated>
                </th>
                <th className="num">
                  <Translated vi="Khoảng cách">Spacing</Translated>
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.E} data-e={r.E}>
                  <td className="mono">{bitString(r.E, e)}</td>
                  <td>{r.rule}</td>
                  <td>{r.vals.map((v) => show(v, 5)).join(', ')}</td>
                  <td className="num">{r.spacing === null ? '–' : show(r.spacing)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: spacing in real formats
const REAL: Format[] = [FP32, FP16, BF16, E4M3, E5M2, E2M1];
const FP8 = new Set(['e4m3', 'e5m2']);

function statusText(c: Classified): ReactNode {
  switch (c.status) {
    case 'normal':
      return c.exact ? tr('normal, exact', 'chuẩn, chính xác') : tr('normal', 'chuẩn');
    case 'subnormal':
      return c.exact ? tr('subnormal, exact', 'subnormal, chính xác') : tr('subnormal', 'số subnormal');
    case 'zero':
      return c.input === 0 ? tr('zero', 'số 0') : <span className="chip fail">{tr('underflow to 0', 'underflow về 0')}</span>;
    case 'flushed':
      return <span className="chip fail">{tr('flushed to 0', 'bị ép về 0')}</span>;
    case 'inf':
      return <span className="chip fail">{tr('overflow to infinity', 'tràn thành vô cực')}</span>;
    case 'nan':
      return <span className="chip fail">{tr('overflow to NaN', 'tràn thành NaN')}</span>;
    case 'saturated':
      return <span className="chip fail"><Translated vi={<>bão hòa ở {show(c.value)}</>}><>saturated at {show(c.value)}</></Translated></span>;
  }
}

function RangeChart({ x, ftz }: { x: number; ftz: boolean }) {
  const [ref, W] = useWidth();
  const L0 = W < 560 ? 66 : 92;
  const R0 = W - 14;
  const tlo = -34;
  const thi = 22;
  const T = (t: number) => L0 + ((Math.min(thi, Math.max(tlo, t)) - tlo) / (thi - tlo)) * (R0 - L0);
  const rowH = 24;
  const top = 8;
  const H = top + rowH * REAL.length + 30;
  const tx = x === 0 ? null : Math.log2(Math.abs(x));
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={tr('Range of each format on a log scale', 'Phạm vi của từng định dạng trên thang log')} style={{ maxWidth: '100%', height: 'auto', display: 'block' }}>
        {REAL.map((f, r) => {
          const y = top + r * rowH;
          const sub = minSubnormal(f);
          const a = Math.log2(sub ?? minNormal(f));
          const b = Math.log2(minNormal(f));
          const c = Math.log2(maxFinite(f));
          return (
            <g key={f.id} data-format={f.id}>
              <text x={4} y={y + 14} fontSize="11.5">
                {W < 560 ? f.name.replace(/^FP[84] /, '') : f.name}
              </text>
              {sub !== null && !ftz ? (
                <rect x={T(a)} y={y + 5} width={T(b) - T(a)} height={12} style={{ fill: 'none', stroke: 'var(--ink-3)' }} strokeDasharray="3 2" />
              ) : null}
              <rect x={T(b)} y={y + 5} width={T(c) - T(b)} height={12} style={{ fill: 'var(--ink-3)', opacity: 0.35 }} />
              {a < tlo ? (
                <text x={L0 + 3} y={y + 15} fontSize="10" style={{ fill: 'var(--ink)' }}>
                  ◀ {`2^${minus(ftz ? b : a)}`}
                </text>
              ) : null}
              {c > thi ? (
                <text x={R0 - 3} y={y + 15} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink)' }}>
                  {`2^${Math.round(c)}`} ▶
                </text>
              ) : null}
            </g>
          );
        })}
        {[-30, -20, -10, 0, 10, 20].map((t) => (
          <g key={t}>
            <line x1={T(t)} x2={T(t)} y1={top} y2={H - 26} style={{ stroke: 'var(--rule-soft)' }} />
            <text x={T(t)} y={H - 10} fontSize="11" textAnchor="middle">{`2^${minus(t)}`}</text>
          </g>
        ))}
        {tx !== null ? (
          <g>
            <line x1={T(tx)} x2={T(tx)} y1={top - 4} y2={H - 26} style={{ stroke: 'var(--ink)' }} strokeWidth={1.5} />
            <text x={T(tx) + 4} y={H - 28} fontSize="11">
              |x|
            </text>
          </g>
        ) : null}
      </svg>
    </div>
  );
}

function SpacingBench() {
  const [t, setT] = useState(0);
  const [text, setText] = useState('1');
  const [x, setX] = useState(1);
  const [ftz, setFtz] = useState(false);
  const [sat, setSat] = useState(false);
  const setFromSlider = (v: number) => {
    setT(v);
    const xv = 2 ** v;
    setX(xv);
    setText(show(xv, 6));
  };
  const setFromText = (s: string) => {
    setText(s);
    const v = parseValue(s);
    if (v === null || !Number.isFinite(v)) return;
    setX(v);
    if (v !== 0) setT(Math.min(20, Math.max(-30, Math.log2(Math.abs(v)))));
  };
  const presets: [string, number][] = [
    ['70,000', 70000],
    ['1.001', 1.001],
    ['0.1', 0.1],
    ['500', 500],
    ['2^-20', 2 ** -20],
  ];
  return (
    <Bench
      title={tr('Spacing in real formats', 'Khoảng cách trong các định dạng thực')}
      task={tr(
        'Move x across the range, or pick one of the values. Then turn denormals off.',
        'Di chuyển x khắp phạm vi, hoặc chọn một trong các giá trị. Rồi tắt denormal.',
      )}
      foot={
        <Translated
          vi={
            <>
              Mỗi hàng làm tròn x về gần nhất, hòa thì về số chẵn. Thanh: nét liền là phạm vi chuẩn, nét đứt là phạm vi dưới
              chuẩn, trên thang log. FP8 và FP4 theo định nghĩa của OCP (appA-facts A3, A4); E2M1 không có vô cực hay NaN, nên
              nó luôn bão hòa. Phép chuyển đổi E4M3 của NVIDIA và AMD mặc định bão hòa, còn của Metal bão hòa với E4M3 nhưng
              không với E5M2 (A5, B2, C4).
            </>
          }
        >
          <>
            Each row rounds x to nearest, ties to even. Bars: solid is the normal range, dashed the subnormal range, on a
            log scale. FP8 and FP4 follow OCP's definitions (appA-facts A3, A4); E2M1 has no infinity or NaN, so it always
            saturates. NVIDIA's and AMD's E4M3 conversions saturate by default, and Metal's saturate for E4M3 but not E5M2
            (A5, B2, C4).
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Trượt x (thang log)">Slide x (log scale)</Translated>
          <input type="range" min="-30" max="20" step="0.01" value={t} aria-label={tr('log2 of x', 'log2 của x')} onChange={(ev) => setFromSlider(+ev.target.value)} />
        </label>
        <label className="field">
          <Translated vi="hoặc nhập x">or type x</Translated>
          <input type="text" inputMode="decimal" value={text} aria-label="x" style={{ width: '8rem' }} onChange={(ev) => setFromText(ev.target.value)} />
        </label>
        <span className="row" style={{ gap: '0.35rem' }}>
          {presets.map(([label, v]) => (
            <button key={label} className="quiet" onClick={() => setFromText(label === '2^-20' ? '2^-20' : String(v))}>
              {label}
            </button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="check">
          <input type="checkbox" checked={ftz} onChange={(ev) => setFtz(ev.target.checked)} />
          <Translated vi="tắt denormal (ép về 0)">denormals off (flush to zero)</Translated>
        </label>
        <label className="check">
          <input type="checkbox" checked={sat} onChange={(ev) => setSat(ev.target.checked)} />
          <Translated vi="FP8: chuyển đổi bão hòa">FP8: saturating conversion</Translated>
        </label>
      </div>
      <RangeChart x={x} ftz={ftz} />
      <table className="data" style={{ margin: '0.8rem 0 0' }} aria-label={tr('x in each format', 'x trong từng định dạng')}>
        <thead>
          <tr>
            <th>
              <Translated vi="Định dạng">Format</Translated>
            </th>
            <th className="num">
              <Translated vi="Giá trị lưu">Stored value</Translated>
            </th>
            <th className="num">
              <Translated vi="Sai số tương đối">Relative error</Translated>
            </th>
            <th className="num">
              <Translated vi="Khoảng cách tại x (tương đối)">Spacing at x (relative)</Translated>
            </th>
            <th>
              <Translated vi="Trạng thái">Status</Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          {REAL.map((f) => {
            const c = classify(f, x, { flushDenormals: ftz, overflow: sat && FP8.has(f.id) ? 'saturate' : 'ieee' });
            const rel = x === 0 ? NaN : ulp(f, x) / Math.abs(x);
            return (
              <tr key={f.id} data-format={f.id} data-value={String(c.value)} data-status={c.status}>
                <td>{f.name}</td>
                <td className="num" data-part="stored">{show(c.value)}</td>
                <td className="num">{Number.isNaN(c.relError) ? '–' : c.exact ? '0' : show(c.relError, 2)}</td>
                <td className="num">
                  {show(ulp(f, x), 3)}
                  {Number.isNaN(rel) ? null : <span className="muted"> ({show(rel, 2)})</span>}
                </td>
                <td>{statusText(c)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSA2: Record<string, Question> = {
  'appA/A.2/q-ex1': {
    kind: 'numeric',
    id: 'appA/A.2/q-ex1',
    prompt: (
      <Translated
        vi={
          <>
            Một định dạng 6-bit có 1 bit dấu, phần số mũ 2-bit (excess-1, mã cao nhất dành riêng) và phần mantissa 3-bit,
            không có số 0, giống Fig. A.5. Nó biểu diễn được bao nhiêu giá trị dương?
          </>
        }
      >
        <>
          A 6-bit format has 1 sign bit, a 2-bit exponent (excess-1, top code reserved) and a 3-bit mantissa, with no zero,
          like Fig. A.5. How many positive values can it represent?
        </>
      </Translated>
    ),
    answer: 24,
    tol: 0,
    explain: (
      <Translated
        vi={
          <p>
            Số bit mũ không đổi, nên vẫn có ba khoảng: [0.5, 1), [1, 2) và [2, 4). Mỗi khoảng giờ chứa 2
            <sup>3</sup> = 8 giá trị thay vì 4, với khoảng cách bằng một nửa: 24. Phạm vi hầu như không đổi (giá trị lớn
            nhất đi từ 3.5 lên 3.75), và khoảng trống cạnh 0 giờ gấp 8 lần khoảng cách phía trên nó. Thêm một bit mantissa
            mua được độ phân giải, không phải phạm vi.
          </p>
        }
      >
        <p>
          The exponent bits did not change, so there are still three intervals: [0.5, 1), [1, 2) and [2, 4). Each now holds 2
          <sup>3</sup> = 8 values instead of 4, at half the spacing: 24. The range barely moves (the largest value goes from
          3.5 to 3.75), and the gap next to 0 is now 8 times the spacing above it. An extra mantissa bit buys resolution, not
          range.
        </p>
      </Translated>
    ),
    source: 'Exercise 1, p. 599; the answer is ours',
    vi: { source: 'Bài tập 1, p. 599; lời giải là của khóa học này' },
  },
  'appA/A.2/q-ex2': {
    kind: 'numeric',
    id: 'appA/A.2/q-ex2',
    prompt: (
      <Translated
        vi={
          <>
            Giờ là 1 bit dấu, phần số mũ 3-bit (excess-3, mã cao nhất dành riêng) và phần mantissa 2-bit, không có số 0. Giá
            trị lớn nhất biểu diễn được là bao nhiêu?
          </>
        }
      >
        <>
          Now 1 sign bit, a 3-bit exponent (excess-3, top code reserved) and a 2-bit mantissa, no zero. What is the largest
          representable value?
        </>
      </Translated>
    ),
    answer: 14,
    tol: 0,
    explain: (
      <Translated
        vi={
          <p>
            Các mã 000 đến 110 cho số mũ từ −3 đến 3: bảy khoảng bắt đầu ở 1/8, 1/4, 1/2, 1, 2, 4 và 8. Khoảng cao nhất chứa
            8, 10, 12 và 14. Thêm một bit mũ gần như nhân đôi số khoảng, kéo giãn phạm vi về cả phía độ lớn lớn hơn lẫn nhỏ
            hơn, và giữ nguyên số điểm trong mỗi khoảng.
          </p>
        }
      >
        <p>
          Codes 000 to 110 give exponents −3 to 3: seven intervals starting at 1/8, 1/4, 1/2, 1, 2, 4 and 8. The top one holds
          8, 10, 12 and 14. An extra exponent bit roughly doubles the number of intervals, stretching the range toward both
          larger and smaller magnitudes, and leaves the points per interval alone.
        </p>
      </Translated>
    ),
    source: 'Exercise 2, p. 599; the answer is ours',
    vi: { source: 'Bài tập 2, p. 599; lời giải là của khóa học này' },
  },
  'appA/A.2/q-int-half': {
    kind: 'numeric',
    id: 'appA/A.2/q-int-half',
    prompt: (
      <Translated vi={<>Số nguyên dương nhỏ nhất mà FP16 không biểu diễn chính xác được là bao nhiêu?</>}>
        <>What is the smallest positive integer that FP16 cannot represent exactly?</>
      </Translated>
    ),
    answer: 2049,
    tol: 0,
    explain: (
      <Translated
        vi={
          <p>
            FP16 có 11 bit có nghĩa (10 bit được lưu, 1 bit ẩn). Mọi số nguyên đến 2<sup>11</sup> = 2048 đều vừa. Trong khoảng
            từ 2048 đến 4096, khoảng cách là 2, nên 2049 rơi vào giữa 2048 và 2050. 8 bit có nghĩa của BF16 dừng ở 256, nên
            số đầu tiên nó bỏ lỡ là 257; 24 bit của FP32 dừng ở 16,777,216, nên số đầu tiên nó bỏ lỡ là 16,777,217. Một chỉ
            số hay một bộ đếm giữ trong kiểu float sẽ âm thầm sai khi vượt qua những điểm đó.
          </p>
        }
      >
        <p>
          FP16 has 11 significant bits (10 stored, 1 hidden). Every integer up to 2<sup>11</sup> = 2048 fits. In the interval
          from 2048 to 4096 the spacing is 2, so 2049 falls between 2048 and 2050. BF16's 8 significant bits stop at 256, so it
          first misses 257; FP32's 24 stop at 16,777,216, so it first misses 16,777,217. An index or a count kept in a float
          type goes wrong silently past those points.
        </p>
      </Translated>
    ),
    source: 'our arithmetic from the §A.2 observations (mantissa bits set the points per interval)',
    vi: { source: 'phép tính của khóa học này từ các nhận xét ở §A.2 (số bit mantissa quyết định số điểm trong mỗi khoảng)' },
  },
  'appA/A.2/q-range-vs-bits': {
    kind: 'choice',
    id: 'appA/A.2/q-range-vs-bits',
    fixedOrder: true,
    prompt: (
      <Translated vi={<>Định dạng nào giữ được cả giá trị 100,000 lẫn hiệu giữa 1000 và 1001?</>}>
        <>Which format keeps both a value of 100,000 and the difference between 1000 and 1001?</>
      </Translated>
    ),
    options: [
      {
        label: 'FP16',
        why: (
          <Translated vi="Gần 1000, các giá trị FP16 cách nhau 0.5, nên 1001 chính xác; nhưng giá trị lớn nhất của nó là 65504, và 100,000 thành vô cực.">
            Near 1000 FP16 values are 0.5 apart, so 1001 is exact; but its largest value is 65504, and 100,000 becomes infinity.
          </Translated>
        ),
      },
      {
        label: 'BF16',
        why: (
          <Translated vi="BF16 có phạm vi của FP32, nên 100,000 sống sót (thành 99,840). Nhưng gần 1000 các giá trị của nó cách nhau 4, và 1001 bị làm tròn về 1000.">
            BF16 has FP32's range, so 100,000 survives (as 99,840). But near 1000 its values are 4 apart, and 1001 rounds to 1000.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Cả hai">Both</Translated>,
        why: (
          <Translated vi="Mỗi định dạng mất một trong hai: FP16 mất phạm vi, BF16 mất độ phân giải.">
            Each loses one of the two: FP16 the range, BF16 the resolution.
          </Translated>
        ),
      },
      { label: 'FP32', correct: true },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Giữa 512 và 1024, các giá trị của một định dạng cách nhau 2<sup>9</sup> × 2<sup>−m</sup>: 2<sup>−1</sup> với FP16
            (m = 10), 2<sup>2</sup> = 4 với BF16 (m = 7). FP16 dùng các bit của nó cho độ phân giải và dừng ở 65504; BF16
            dùng chúng cho phạm vi, lên đến khoảng 3.4 × 10<sup>38</sup>. Chỉ FP32 có cả hai. Hãy nhập từng giá trị vào trình
            khám phá ở trên để kiểm tra.
          </p>
        }
      >
        <p>
          Between 512 and 1024 a format's values are 2<sup>9</sup> × 2<sup>−m</sup> apart: 2<sup>−1</sup> for FP16 (m =
          10), 2<sup>2</sup> = 4 for BF16 (m = 7). FP16 spends its bits on resolution and stops at 65504; BF16 spends them
          on range, up to about 3.4 × 10<sup>38</sup>. Only FP32 has both. Type each value into the explorer above to
          check.
        </p>
      </Translated>
    ),
    source: '§A.2 (range from exponent bits, precision from mantissa bits); NVIDIA Table 51 for the limits (appA-facts A2)',
    vi: {
      source: '§A.2 (phạm vi đến từ số bit mũ, độ chính xác đến từ số bit mantissa); NVIDIA Table 51 cho các giới hạn (appA-facts A2)',
    },
  },
  'appA/A.2/q-ftz': {
    kind: 'choice',
    id: 'appA/A.2/q-ftz',
    prompt: (
      <Translated
        vi={
          <>
            <code>nvcc --ftz=true</code> làm gì?
          </>
        }
      >
        <>
          What does <code>nvcc --ftz=true</code> do?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Ép các số denormal độ chính xác đơn về 0, tức là tắt denormal. Mặc định là false; --use_fast_math ngầm bật nó thành true.">
            Flushes single-precision denormals to zero, so denormals are off. The default is false; --use_fast_math implies true.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Bật hỗ trợ denormal; không có nó, nvcc mặc định ép các số denormal độ chính xác đơn về 0.">
            Turns denormal support on; without it, nvcc flushes single-precision denormals to zero by default.
          </Translated>
        ),
        why: (
          <Translated vi={'Đó là cách đọc theo nghĩa đen chữ "(on)" của sách. Tài liệu của NVIDIA: --ftz=true ép các số denormal về 0, còn giá trị mặc định, false, giữ chúng lại.'}>
            {"That is the book's \"(on)\" read literally. NVIDIA's documentation: --ftz=true flushes denormals, and the default, false, preserves them."}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Ép denormal về 0 ở mọi độ chính xác, kể cả double; --use_fast_math cũng làm vậy.">
            Flushes denormals to zero in every precision, double included; --use_fast_math does the same.
          </Translated>
        ),
        why: (
          <Translated vi="Cờ này chỉ điều khiển các số denormal độ chính xác đơn.">
            The flag controls single-precision denormals only.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Làm cho một phép atomicAdd float trên bộ nhớ toàn cục (global memory) giữ lại denormal thay vì ép chúng về 0.">
            Makes a float atomicAdd on global memory keep its denormals instead of flushing them.
          </Translated>
        ),
        why: (
          <Translated vi="Phép toán đó luôn ép denormal về 0, bất kể cờ nói gì.">
            That operation always flushes denormals, whatever the flag says.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            ftz là viết tắt của flush to zero (ép về 0). Đặt thành true, nó đánh đổi phạm vi denormal lấy tốc độ, và{' '}
            <code>--use_fast_math</code> đặt nó giúp bạn. Nếu kết quả của một kernel thay đổi khi bạn thêm{' '}
            <code>--use_fast_math</code>, việc các giá trị rất nhỏ bị ép về 0 là một trong những nghi phạm.
          </p>
        }
      >
        <p>
          ftz stands for flush to zero. Set to true, it trades the denormal range for speed, and <code>--use_fast_math</code>{' '}
          sets it for you. If a kernel's results change when you add <code>--use_fast_math</code>, tiny values being flushed
          is one of the suspects.
        </p>
      </Translated>
    ),
    source: '§A.2, p. 589; nvcc documentation 4.2.7.8 and the CUDA guide (appA-facts A6; ch10 facts A6)',
    vi: { source: '§A.2, p. 589; tài liệu nvcc 4.2.7.8 và hướng dẫn CUDA (appA-facts A6; ch10 facts A6)' },
  },
};

// ---------------------------------------------------------------- the section
export function SA2() {
  return (
    <>
      <Translated
        vi={
          <p>
            Các số biểu diễn được của một định dạng là những giá trị nó lưu được chính xác; mọi giá trị khác bị làm tròn về
            một trong số đó. Một số nguyên không dấu 3-bit biểu diễn 0 đến 7 và không gì khác: không có −1, không có 9, không
            có 2.5. Các định dạng dấu phẩy động cũng có một tập hữu hạn, nhưng phân bố rất không đều, và hình dạng của sự phân
            bố đó quyết định sai số của bạn nằm ở đâu.
          </p>
        }
      >
        <p>
          A format's representable numbers are the values it can store exactly; anything else gets rounded to one of them. A
          3-bit unsigned integer represents 0 to 7 and nothing else: not −1, not 9, not 2.5. Floating-point formats also have
          a finite set, but spread very unevenly, and the shape of that spread is what decides where your errors are.
        </p>
      </Translated>
      <Source section="A.2" pages="586" figs="Figs. A.2–A.3" />
      <Translated
        vi={
          <p>
            Để thấy toàn bộ một tập cùng lúc, sách dùng một định dạng 5-bit: 1 bit dấu, phần số mũ 2-bit dạng excess-1 và
            phần mantissa 2-bit. Các mã số mũ 00, 01 và 10 ứng với −1, 0 và +1; 11 được dành riêng. Với 2 bit mantissa, mỗi
            mã số mũ cho bốn giá trị. Các định dạng chỉ khác nhau ở ý nghĩa của mã E = 00, và sách đặt ba quy ước cạnh nhau.
          </p>
        }
      >
        <p>
          To see a whole set at once the book uses a 5-bit format: 1 sign bit, a 2-bit exponent in excess-1 and a 2-bit
          mantissa. Exponent codes 00, 01 and 10 stand for −1, 0 and +1; 11 is reserved. With 2 mantissa bits each exponent
          code gives four values. The formats differ only in what the code E = 00 means, and the book lays out three
          conventions side by side.
        </p>
      </Translated>
      <Source section="A.2" pages="586–587" figs="Fig. A.4" />

      <h2>
        <Translated vi="Khoảng trống nằm ở đâu?">Where is the gap?</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Theo công thức đơn giản của §A.1, mà sách gọi là quy ước không có số 0, E = 00 được đọc như mọi mã khác: 1.M ×
            2<sup>−1</sup>, tức 0.5, 0.625, 0.75 và 0.875. Trước khi nhìn trục số, hãy tính xem chuyện gì xảy ra với một giá
            trị nhỏ.
          </p>
        }
      >
        <p>
          Under the plain formula of §A.1, which the book calls the no-zero convention, E = 00 is read like any other code:
          1.M × 2<sup>−1</sup>, so 0.5, 0.625, 0.75 and 0.875. Before you look at the number line, work out what happens to a
          small value.
        </p>
      </Translated>
      <Predict
        id={P_GAP}
        prompt={tr(
          'In the 5-bit no-zero format, 0.2 is rounded to the nearest representable value. What is the rounding error?',
          'Trong định dạng 5-bit không có số 0, 0.2 được làm tròn về giá trị biểu diễn được gần nhất. Sai số làm tròn là bao nhiêu?',
        )}
      >
        {(v) => (
          <Translated
            vi={
              <p>
                Các giá trị dương bắt đầu ở 0.5, và không có gì nằm giữa −0.5 và 0.5, nên 0.2 thành 0.5: sai số là{' '}
                <b>0.3</b>.{' '}
                {Math.abs(v - 0.3) < 0.005
                  ? 'Đúng như bạn đã tính.'
                  : Math.abs(v - 0.05) < 0.005
                    ? '0.05 là sai số khi có denormal, thứ mà định dạng này không có.'
                    : Math.abs(v - 0.2) < 0.005
                      ? '0.2 sẽ là sai số khi làm tròn về 0, nhưng định dạng này không có số 0.'
                      : 'Hãy xem những giá trị nào nằm gần 0.2 nhất.'}{' '}
                Bảng dưới trục số giờ cho thấy hai quy ước còn lại: underflow đột ngột cho 0 (sai số 0.2) và denormal cho
                0.25 (sai số 0.05).
              </p>
            }
          >
            <p>
              The positive values start at 0.5, and nothing lies between −0.5 and 0.5, so 0.2 becomes 0.5: an error of{' '}
              <b>0.3</b>.{' '}
              {Math.abs(v - 0.3) < 0.005
                ? 'That is what you had.'
                : Math.abs(v - 0.05) < 0.005
                  ? '0.05 is the error with denormals, which this format does not have.'
                  : Math.abs(v - 0.2) < 0.005
                    ? '0.2 would be the error of rounding to 0, but this format has no 0.'
                    : 'Check which values lie closest to 0.2.'}{' '}
              The table under the number line now shows the other two conventions: abrupt underflow gives 0 (error 0.2) and
              denormals give 0.25 (error 0.05).
            </p>
          </Translated>
        )}
      </Predict>
      <NumberLineBench />
      <Translated
        vi={
          <p>
            Hình vẽ của chính sách cho các giá trị này, Fig. A.5, không đúng tỉ lệ gần 0: nó vẽ đoạn từ 0 đến 1 ngắn hơn đoạn
            từ 1 đến 2, nên 0.5 đến 0.875 không nằm đúng chỗ (bản thân các giá trị thì rõ từ Fig. A.4). Trình khám phá vẽ đúng
            tỉ lệ, đó là lý do đoạn trống cạnh 0 trông lớn đúng như nó vốn có.
          </p>
        }
      >
        <p>
          The book's own drawing of these values, Fig. A.5, is not to scale near 0: it draws the stretch from 0 to 1 shorter
          than the one from 1 to 2, so 0.5 to 0.875 are not where they belong (the values themselves are clear from Fig.
          A.4). The explorer is to scale, which is why the empty stretch next to 0 looks as large as it is.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Chỉ các định dạng kiểu sách: mã số mũ cao nhất được dành riêng và không có vô cực, nên một điểm thăm dò vượt quá
              giá trị lớn nhất bị kẹp về giá trị đó. Một điểm thăm dò nằm đúng giữa khoảng trống underflow đột ngột đi về 0;
              sách không đưa ra quy tắc cho trường hợp hòa đó.
            </p>
          }
        >
          <p>
            Book-style formats only: the top exponent code is reserved and there is no infinity, so a probe beyond the largest
            value is clamped to it. A probe exactly halfway across the abrupt-underflow gap goes to 0; the book gives no rule
            for that tie.
          </p>
        </Translated>
      </Simplification>

      <h2>
        <Translated vi="Năm điều trục số cho thấy">Five things the number line shows</Translated>
      </h2>
      <Translated vi={<p>Đặt trình khám phá về lại e = 2, m = 2 và không có số 0, rồi kiểm tra từng điều sau.</p>}>
        <p>Set the explorer back to e = 2, m = 2 and no zero, then check each of these.</p>
      </Translated>
      <Translated
        vi={
          <ol>
            <li>
              <b>Bit mũ tạo ra các khoảng.</b> Mỗi mã số mũ có chứa giá trị mở đầu một khoảng tại một lũy thừa của hai: ở
              đây có ba khoảng mỗi bên của 0. Chuyển e lên 3 và sẽ có bảy.
            </li>
            <li>
              <b>Bit mantissa lấp đầy chúng.</b> Mỗi khoảng chứa 2<sup>m</sup> giá trị, bất kể độ rộng của nó. Chuyển m lên
              3: tám giá trị mỗi khoảng với khoảng cách bằng một nửa, và các khoảng vẫn ở nguyên chỗ cũ. Nhiều bit mantissa
              hơn, sai số làm tròn nhỏ hơn.
            </li>
            <li>
              <b>Thiếu số 0.</b> Trong quy ước không có số 0, không mẫu bit nào nghĩa là 0. Sách gọi đây là một khiếm khuyết
              nghiêm trọng.
            </li>
            <li>
              <b>Các giá trị dồn về phía 0.</b> Mỗi khoảng rộng bằng một nửa khoảng kế tiếp phía ngoài và chứa cùng số giá
              trị, nên sai số làm tròn lớn nhất trong một khoảng co lại theo độ lớn của các số trong đó. Đó là hình dạng đúng:
              sai số một đơn vị hầu như không đáng kể với giá trị một triệu nhưng là tất cả với giá trị hai.
            </li>
            <li>
              <b>Trừ ngay sát 0.</b> Khoảng trống giữa 0 và giá trị dương nhỏ nhất gấp 2<sup>m</sup> lần khoảng cách ngay
              phía trên nó (tỉ số của trình khám phá). Các giá trị trong khoảng trống bị làm tròn với sai số lớn gấp
              2<sup>m</sup> lần so với các giá trị ngay bên ngoài, điều này gây hại cho mã kiểm tra hội tụ trên các giá trị
              rất nhỏ hoặc chia cho chúng.
            </li>
          </ol>
        }
      >
        <ol>
          <li>
            <b>Exponent bits make the intervals.</b> Each exponent code that holds values starts an interval at a power of two:
            three on each side of 0 here. Move e to 3 and there are seven.
          </li>
          <li>
            <b>Mantissa bits fill them.</b> Every interval holds 2<sup>m</sup> values, whatever its width. Move m to 3: eight
            per interval at half the spacing, and the intervals stay where they were. More mantissa bits, smaller rounding
            errors.
          </li>
          <li>
            <b>0 is missing.</b> In the no-zero convention no pattern means 0. The book calls this a serious deficiency.
          </li>
          <li>
            <b>Values crowd toward 0.</b> Each interval is half as wide as the next one out and holds as many values, so the
            largest rounding error in an interval shrinks with the size of the numbers in it. That is the right shape: an error
            of one unit hardly matters on a value of a million and is everything on a value of two.
          </li>
          <li>
            <b>Except right next to 0.</b> The gap between 0 and the smallest positive value is 2<sup>m</sup> times the
            spacing just above it (the explorer's ratio). Values in the gap are rounded with errors 2<sup>m</sup> times larger
            than values just outside it, which hurts code that tests for convergence on tiny values or divides by them.
          </li>
        </ol>
      </Translated>
      <Source section="A.2" pages="587–588" figs="Fig. A.5" />
      <Aside>
        <Translated
          vi={
            <p>
              <b>Trong PDF.</b> p. 588 nói số 0 bị thiếu trong "cột no-zero của Fig. A.5". Các cột nằm ở Fig. A.4; Fig. A.5 là
              trục số.
            </p>
          }
        >
          <p>
            <b>In the PDF.</b> p. 588 says 0 is missing from "the no-zero column of Fig. A.5". The columns are in Fig. A.4;
            Fig. A.5 is the number line.
          </p>
        </Translated>
      </Aside>
      <Ask q={questionsSA2['appA/A.2/q-ex1']} />
      <Ask q={questionsSA2['appA/A.2/q-ex2']} />

      <h2>
        <Translated vi="Underflow đột ngột">Abrupt underflow</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Cách sửa thô bạo nhất: hễ E bằng 0 thì giá trị là 0, bất kể mantissa nói gì. Trong định dạng 5-bit, điều đó biến
            tám mẫu bit, bốn mẫu mỗi bên, thành 0. Số 0 giờ là một giá trị, nhưng khoảng trống tệ hơn: giá trị dương nhỏ nhất
            giờ là 1, nên khoảng trống cạnh 0 rộng gấp đôi trước. Chuyển trình khám phá sang underflow đột ngột và tỉ số vẫn
            là 2<sup>m</sup>, đo so với một khoảng trống đã tăng gấp đôi. Sách nói một số máy minicomputer những năm 1980
            hoạt động theo cách này, và một số khối số học phải rất nhanh vẫn còn làm vậy.
          </p>
        }
      >
        <p>
          The bluntest fix: whenever E is 0, the value is 0, whatever the mantissa says. In the 5-bit format that turns eight
          patterns, four on each side, into 0. Zero is now a value, but the gap gets worse: the smallest positive value is now
          1, so the gap next to 0 is twice as wide as before. Switch the explorer to abrupt underflow and the ratio stays 2
          <sup>m</sup>, measured against a gap that has doubled. The book says some 1980s minicomputers worked this way, and
          some arithmetic units that must be very fast still do.
        </p>
      </Translated>
      <Source section="A.2" pages="588–589" figs="Fig. A.6" />

      <h2>
        <Translated vi="Denormal">Denormals</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Câu trả lời của IEEE là phi chuẩn hóa (denormalisation). Khi E bằng 0, mantissa được đọc là 0.M thay vì 1.M, và
            số mũ lấy từ khoảng phía trên, ở đây là 0 thay vì −1. Khi đó mẫu bit 0 00 01 nghĩa là 0.01B × 2<sup>0</sup> = 0.25.
            Bốn mẫu bit có E = 00 giờ trải đều trên [0, 1) với cùng khoảng cách như [1, 2): 0, 0.25, 0.5, 0.75. Khoảng trống
            biến mất, và tỉ số của trình khám phá giảm xuống 1.
          </p>
        }
      >
        <p>
          IEEE's answer is denormalisation. When E is 0, the mantissa is read as 0.M instead of 1.M, and the exponent is taken
          from the interval above, 0 instead of −1 here. Pattern 0 00 01 then means 0.01B × 2<sup>0</sup> = 0.25. The four
          patterns with E = 00 now spread evenly over [0, 1) at the same spacing as [1, 2): 0, 0.25, 0.5, 0.75. The gap is
          gone, and the explorer's ratio drops to 1.
        </p>
      </Translated>
      <Source section="A.2" pages="589" figs={tr('Fig. A.4 (right columns), Fig. A.7', 'Fig. A.4 (các cột bên phải), Fig. A.7')} />
      <Translated
        vi={
          <p>
            Với trường số mũ n bit, một mẫu bit có E = 0 nghĩa là 0.M × 2<sup>−2<sup>n−1</sup> + 2</sup>. NVIDIA viết cùng điều
            đó là 0.M × 2<sup>1 − bias</sup>; với bias = 2<sup>n−1</sup> − 1 thì hai số mũ bằng nhau.
          </p>
        }
      >
        <p>
          For an n-bit exponent field, a pattern with E = 0 means 0.M × 2<sup>−2<sup>n−1</sup> + 2</sup>. NVIDIA writes the same
          thing as 0.M × 2<sup>1 − bias</sup>; with bias = 2<sup>n−1</sup> − 1 the two exponents are equal.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              <b>Trong PDF.</b> p. 589 trỏ tới Fig. A.6 cho denormal; ý là Fig. A.7. "Khoảng trước đó" của nó nghĩa là khoảng
              lớn hơn kế tiếp, như công thức xác nhận. Và Fig. A.7 vẽ một ngôi sao thứ 13 gần 4, thứ mà định dạng không chứa
              được: E = 11 được dành riêng.
            </p>
          }
        >
          <p>
            <b>In the PDF.</b> p. 589 points to Fig. A.6 for denormals; Fig. A.7 is meant. Its "previous interval" means the
            next larger one, as the formula confirms. And Fig. A.7 draws a 13th star near 4, which the format cannot hold:
            E = 11 is reserved.
          </p>
        </Translated>
      </Aside>

      <h2>
        <Translated vi="Denormal tốn gì, và công tắc bật tắt">What denormals cost, and the switch</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Cái giá nằm ở phần cứng: một khối phải nhận ra rằng một giá trị nằm trong phạm vi denormal và chuyển cách biểu
            diễn. Sách nói các cài đặt với phần cứng vừa phải có thể tốn hàng nghìn chu kỳ xung nhịp mỗi lần một denormal
            được tạo ra hoặc được dùng, đó là lý do các thiết bị CUDA đời đầu bỏ chúng ra, và hầu như mọi thiết bị CUDA gần
            đây đều hỗ trợ chúng.
          </p>
        }
      >
        <p>
          The price is in the hardware: a unit has to notice that a value is in the denormal range and switch representation.
          The book says implementations with moderate hardware can spend thousands of clock cycles each time a denormal is
          produced or consumed, which is why early CUDA devices left them out, and that virtually all recent CUDA devices
          support them.
        </p>
      </Translated>
      <Source section="A.2" pages="589" />
      <Translated
        vi={
          <p>
            Sách nói thêm rằng một cờ của trình biên dịch nvcc bật tắt denormal, và viết thiết lập đó là "ftz = true (on)
            hoặc ftz = false (off)". Hãy kiểm tra xem chiều nào là chiều nào trước khi dựa vào nó.
          </p>
        }
      >
        <p>
          It adds that an nvcc compiler flag switches denormals, and writes the setting as "ftz = true (on) or ftz = false
          (off)". Check which way round that is before you rely on it.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 against the nvcc 13.5 and CUDA guide, Clang, MSL 4.1 and PyTorch 2.15 docs (appA-facts A6, B3, C2, D1; ch10 facts A6)',
          '2026-11-20 đối chiếu với tài liệu nvcc 13.5 và hướng dẫn CUDA, Clang, MSL 4.1 và PyTorch 2.15 (appA-facts A6, B3, C2, D1; ch10 facts A6)',
        )}
      >
        <Translated
          vi={
            <p>
              "ftz = true (on)" của sách đọc ngược: <code>--ftz=true</code> bật việc ép về 0 và tắt denormal. NVIDIA chỉ nói
              denormal "tốn kém hơn về tính toán"; "hàng nghìn chu kỳ xung nhịp" là khẳng định của sách.
            </p>
          }
        >
          <p>
            The book's "ftz = true (on)" reads backwards: <code>--ftz=true</code> turns flushing on and denormals off. NVIDIA
            says only that denormals are "computationally more expensive"; "thousands of clock cycles" is the book's claim.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Ngày nay backend nào cũng để denormal bị mất ở đâu đó, nên một kernel mà các giá trị nhỏ của nó có ý nghĩa không
            nên giả định rằng chúng sống sót:
          </p>
        }
      >
        <p>
          Today every backend lets denormals be lost somewhere, so a kernel whose small values matter should not assume they
          survive:
        </p>
      </Translated>
      <table className="data" aria-label={tr('Denormals on each backend', 'Denormal trên từng backend')}>
        <thead>
          <tr>
            <th>
              <Translated vi="Backend">Backend</Translated>
            </th>
            <th>
              <Translated vi="Denormal FP32">FP32 denormals</Translated>
            </th>
            <th>
              <Translated vi="Ngoại lệ bạn nên biết">Exceptions you should know</Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>CUDA</td>
            <Translated
              vi={
                <td>
                  Mặc định được giữ. <code>--ftz=true</code> ép chúng về 0 (chỉ độ chính xác đơn); <code>--use_fast_math</code>{' '}
                  ngầm bật nó.
                </td>
              }
            >
              <td>
                Kept by default. <code>--ftz=true</code> flushes them (single precision only); <code>--use_fast_math</code>{' '}
                implies it.
              </td>
            </Translated>
            <Translated vi={<td>Một phép <code>atomicAdd</code> float trên bộ nhớ toàn cục luôn ép về 0, bất kể cờ.</td>}>
              <td>A float <code>atomicAdd</code> on global memory always flushes, whatever the flag.</td>
            </Translated>
          </tr>
          <tr>
            <td>HIP</td>
            <Translated
              vi={
                <td>
                  Mặc định không có trong tài liệu. <code>-fgpu-flush-denormals-to-zero</code> của Clang ép chúng về 0.
                </td>
              }
            >
              <td>
                Default not documented. Clang's <code>-fgpu-flush-denormals-to-zero</code> flushes them.
              </td>
            </Translated>
            <Translated vi={<td>PyTorch ghi trong tài liệu rằng các lệnh ma trận FP16 và BF16 của MI200 ép denormal về 0.</td>}>
              <td>PyTorch documents that MI200's FP16 and BF16 matrix instructions flush denormals.</td>
            </Translated>
          </tr>
          <tr>
            <td>Metal</td>
            <Translated vi={<td>Đặc tả nói denormal "có thể bị ép về 0", với float, half và bfloat.</td>}>
              <td>The specification says denormals "may be flushed to zero", for float, half and bfloat.</td>
            </Translated>
            <Translated vi={<td>GPU Apple nào ép về 0 thì không có trong tài liệu.</td>}>
              <td>Which Apple GPUs flush is not documented.</td>
            </Translated>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Lab A.1 chạy trên chính các GPU của bạn và báo cáo chúng làm gì với một giá trị nhỏ hơn số FP32 chuẩn nhỏ nhất (
            <a href={href('appA', 'labs')}>các lab</a>).
          </p>
        }
      >
        <p>
          Lab A.1 runs on your own GPUs and reports what they do with a value below the smallest normal FP32 number (
          <a href={href('appA', 'labs')}>labs</a>).
        </p>
      </Translated>
      <Ask q={questionsSA2['appA/A.2/q-ftz']} />

      <h2>
        <Translated vi="Độ chính xác">Precision</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách định nghĩa độ chính xác của một định dạng là sai số lớn nhất nó có thể gây ra khi lưu một số thành một
            trong các giá trị biểu diễn được của nó. Sai số nhỏ hơn, độ chính xác cao hơn. Mỗi bit mantissa thêm vào nhân đôi
            số giá trị trong mỗi khoảng và vì thế giảm một nửa sai số đó; chỉ vậy thôi, không có gì bí ẩn hơn, là khác biệt
            giữa độ chính xác đơn và độ chính xác kép.
          </p>
        }
      >
        <p>
          The book defines the precision of a format as the largest error it can introduce by storing a number as one of its
          representable values. Smaller error, higher precision. Each extra mantissa bit doubles the values per interval and so
          halves that error; that, and nothing more mysterious, is the difference between single and double precision.
        </p>
      </Translated>
      <Source section="A.2" pages="590" />

      <h2>
        <Translated vi="Các định dạng thực trên cùng một hình">Real formats on one picture</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mọi điều ở trên đều áp dụng cho các định dạng bạn dùng. Trong phạm vi chuẩn, khoảng cách gần x bằng 2<sup>−m</sup>{' '}
            lần lũy thừa của hai ngay dưới x, nên khoảng cách tương đối so với giá trị luôn nằm trong khoảng một thừa số hai
            quanh 2<sup>−m</sup> ở mọi nơi: 2<sup>−23</sup> với FP32, 2<sup>−10</sup> với FP16, 2<sup>−7</sup> với BF16. Phạm
            vi mới là chỗ các định dạng khác nhau, và nó đến từ số bit mũ: FP16 chạy từ 2<sup>−14</sup> đến 65504 với các giá
            trị chuẩn, BF16 và FP32 từ 2<sup>−126</sup> đến khoảng 3.4 × 10<sup>38</sup>.
          </p>
        }
      >
        <p>
          Everything above carries over to the formats you use. In the normal range the spacing near x is 2<sup>−m</sup> times
          the power of two just below x, so the spacing relative to the value stays within a factor of two of 2<sup>−m</sup>{' '}
          everywhere: 2<sup>−23</sup> for FP32, 2<sup>−10</sup> for FP16, 2<sup>−7</sup> for BF16. Range is where the formats
          differ, and it comes from the exponent bits: FP16 runs from 2<sup>−14</sup> to 65504 in normal values, BF16 and FP32
          from 2<sup>−126</sup> to about 3.4 × 10<sup>38</sup>.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Phần định dạng thực là của khóa học này, từ Table 51 của NVIDIA (cột epsilon của nó là 2<sup>−m</sup>) và các
              đặc tả FP8 và MX của OCP (appA-facts A2 đến A4). §A.5 nói vì sao các định dạng hẹp tồn tại.
            </p>
          }
        >
          <p>
            Real formats are ours, from NVIDIA's Table 51 (its epsilon column is 2<sup>−m</sup>) and OCP's FP8 and MX
            specifications (appA-facts A2 to A4). §A.5 says why the narrow ones exist.
          </p>
        </Translated>
      </Beyond>
      <SpacingBench />
      <Ask q={questionsSA2['appA/A.2/q-range-vs-bits']} />
      <Translated
        vi={
          <p>
            Thêm hai giá trị nữa để thấy rõ sự đánh đổi. Một activation bằng 70,000 thành vô cực trong FP16 và sống sót trong
            BF16. Hiệu giữa 1 và 1.001 sống sót trong FP16 và biến mất trong BF16. Nói chung không định dạng nào chính xác
            hơn định dạng nào; chúng dùng 16 bit theo cách khác nhau, và FP16 hết phạm vi từ lâu trước khi BF16 hết bit.
          </p>
        }
      >
        <p>
          Two more values make the trade concrete. An activation of 70,000 becomes infinity in FP16 and survives in BF16. The
          difference between 1 and 1.001 survives in FP16 and vanishes in BF16. Neither format is more accurate than the other
          in general; they spend 16 bits differently, and FP16 runs out of range long before BF16 runs out of bits.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Ép về 0 được mô hình hóa bằng cách làm tròn có denormal rồi thay kết quả subnormal bằng 0. Phần cứng có thể
              ép cả đầu vào, và các nhà cung cấp ghi trong tài liệu những chế độ khác nhau. Mọi phép chuyển đổi ở đây làm
              tròn về số chẵn gần nhất.
            </p>
          }
        >
          <p>
            Flush-to-zero is modelled as rounding with denormals and then replacing a subnormal result by zero. Hardware may
            also flush inputs, and vendors document different modes. Every conversion here rounds to nearest even.
          </p>
        </Translated>
      </Simplification>
      <Ask q={questionsSA2['appA/A.2/q-int-half']} />
    </>
  );
}
