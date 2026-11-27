import { useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { fmt } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BOOK_CASES,
  FRACTIONS,
  RATES,
  SPEEDUPS,
  THREADS,
  ceiling,
  corners,
  fractionIndex,
  fractionNeeded,
  kernelNeeded,
  levers,
  limits,
  limitsCurve,
  nearestIndex,
  remaining,
  speedup,
  speedupCurve,
  untouchedShare,
} from '../../sim/amdahl';
import { useWidth } from './useWidth';

// ---------------------------------------------------------------- formatting
/** A speed-up: "1.42×", "1,000×", or "unbounded". Rounded first, so that 999.9999 reads as 1,000. */
const times = (v: number) => (isFinite(v) ? fmt(Number(v.toPrecision(6))) + '×' : tr('unbounded', 'không giới hạn'));
/** A share of something, as a percentage with four significant digits. */
const share = (v: number) => fmt(v * 100, 4) + '%';
/** A parallel fraction exactly as the slider holds it: "99%", "99.95%". */
const frac = (p: number) => Number((p * 100).toFixed(4)) + '%';
const count = (v: number) => v.toLocaleString('en-US', { maximumFractionDigits: 3 });

// The cases the page works with. Every number shown for them comes from the model.
const DEFAULT_P = 0.99;
const DEFAULTS = { threads: 1000, rate: 1 / 4, cap: 10 };
const book30 = BOOK_CASES[0];
const book99 = BOOK_CASES[2];
/** q-fivehundred: the part made this many times faster, and the overall figure asked for. */
export const FIVE_HUNDRED = { s: 1000, target: 500 };

// ---------------------------------------------------------------- plot helpers
const tick = (v: number) => (v >= 1000 ? v / 1000 + 'k' : String(v));
const path = (pts: [number, number][]) => pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');

function FractionSlider({ p, setP }: { p: number; setP: (p: number) => void }) {
  return (
    <label className="field">
      <span>
        <Translated vi={<>Phần song song p = <b data-out="p">{frac(p)}</b></>}>
          Parallel fraction p = <b data-out="p">{frac(p)}</b>
        </Translated>
      </span>
      <input
        type="range"
        aria-label={tr('parallel fraction', 'phần song song')}
        min="0"
        max={FRACTIONS.length - 1}
        step="1"
        value={fractionIndex(p)}
        onChange={(e) => setP(FRACTIONS[+e.target.value])}
        style={{ width: '15rem' }}
      />
    </label>
  );
}

// ---------------------------------------------------------------- explorer 1: Amdahl's law
function TimeBars({ p, s }: { p: number; s: number }) {
  const r = remaining(p, s);
  const us = untouchedShare(p, s);
  const seg = (part: 'untouched' | 'accelerated', width: number) => (
    <div
      data-part={part}
      style={{ width: `${width * 100}%`, background: part === 'untouched' ? 'var(--ink)' : 'var(--ink-3)', height: '100%' }}
    />
  );
  const row = (id: string, name: string, figure: ReactNode, parts: ReactNode) => (
    <div className="row" style={{ gap: '0.3rem 1rem', marginBottom: '0.55rem', alignItems: 'center' }}>
      <div style={{ flex: '0 0 13.5rem', lineHeight: 1.25 }}>
        <b>{name}</b>
        <div className="muted" style={{ fontSize: '0.76rem' }}>{figure}</div>
      </div>
      <div
        data-bar={id}
        style={{
          flex: '1 1 14rem', display: 'flex', height: '1.5rem', background: 'var(--cell)',
          border: '1px solid var(--rule)', borderRadius: 2, overflow: 'hidden',
        }}
      >
        {parts}
      </div>
    </div>
  );
  return (
    <div className="panel" style={{ marginBottom: '1rem' }} data-timebars>
      {row('before', tr('Before', 'Trước'), tr('the whole run: 100%', 'toàn bộ lần chạy: 100%'), <>{seg('untouched', r.untouched)}{seg('accelerated', p)}</>)}
      {row(
        'after',
        tr('After, same scale', 'Sau, cùng tỷ lệ'),
        <Translated vi={<><span data-out="left">{share(r.total)}</span> thời gian ban đầu</>}>
          <span data-out="left">{share(r.total)}</span> of the original time
        </Translated>,
        <>{seg('untouched', r.untouched)}{seg('accelerated', r.accelerated)}</>,
      )}
      {row(
        'zoom',
        tr('After, magnified', 'Sau, phóng to'),
        <Translated vi={<>phần không đụng tới chiếm <span data-out="share">{share(us)}</span> trong đó</>}>
          the untouched part is <span data-out="share">{share(us)}</span> of it
        </Translated>,
        <>{seg('untouched', us)}{seg('accelerated', 1 - us)}</>,
      )}
      <span className="legend" style={{ marginTop: '0.2rem' }}>
        <span><i style={{ background: 'var(--ink)' }} /><Translated vi="để nguyên: 1 − p">left alone: 1 − p</Translated></span>
        <span><i style={{ background: 'var(--ink-3)' }} /><Translated vi="phần được tăng tốc: p trước, p / s sau">the accelerated part: p before, p / s after</Translated></span>
      </span>
    </div>
  );
}

function AmdahlPlot({ p, s }: { p: number; s: number }) {
  const [ref, W] = useWidth(600);
  const H = Math.round(Math.min(330, Math.max(250, W * 0.56)));
  const pad = { l: 52, r: 10, t: 24, b: 44 };
  const slot = 80; // room at the right for "unbounded"
  const x1 = W - pad.r - slot; // where s = 1,000 sits
  const xInf = W - pad.r - 22;
  const X = (v: number) => (isFinite(v) ? pad.l + (Math.log10(v) / 3) * (x1 - pad.l) : xInf);
  const Y = (v: number) => H - pad.b - (Math.log10(Math.min(Math.max(v, 1), 10000)) / 4) * (H - pad.t - pad.b);
  const top = ceiling(p);
  const now = speedup(p, s);
  const curve = speedupCurve(p).map((q): [number, number] => [X(q.s), Y(q.overall)]);
  const px = X(s);
  const py = Y(now);
  return (
    <div ref={ref} style={{ width: '100%', overflow: 'hidden' }}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Overall speed-up against the speed-up of the accelerated part', 'Mức tăng tốc tổng thể theo mức tăng tốc của phần được tăng tốc')} style={{ display: 'block' }}>
        <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
        {/* past s = 1,000 the axis is not to scale: one column for the unbounded case */}
        <rect x={x1 + 14} y={pad.t} width={W - pad.r - x1 - 14} height={H - pad.t - pad.b} fill="var(--rule-soft)" />
        {[1, 10, 100, 1000].map((v) => (
          <g key={'x' + v}>
            <line x1={X(v)} x2={X(v)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
            <text x={X(v)} y={H - pad.b + 15} fontSize="11" textAnchor="middle">{v.toLocaleString('en-US')}</text>
          </g>
        ))}
        <text x={W - pad.r} y={H - pad.b + 15} fontSize="11" textAnchor="end"><Translated vi="không giới hạn">unbounded</Translated></text>
        {[1, 10, 100, 1000, 10000].map((v) => (
          <g key={'y' + v}>
            <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
            <text x={pad.l - 6} y={Y(v) + 4} fontSize="11" textAnchor="end">{tick(v)}</text>
          </g>
        ))}
        <text x={(pad.l + x1) / 2} y={H - 8} fontSize="11.5" textAnchor="middle"><Translated vi="mức tăng tốc của phần được tăng tốc, s">speed-up of the accelerated part, s</Translated></text>
        <text x={14} y={(H - pad.b + pad.t) / 2} fontSize="11.5" textAnchor="middle" transform={`rotate(-90 14 ${(H - pad.b + pad.t) / 2})`}><Translated vi="mức tăng tốc tổng thể">overall speed-up</Translated></text>
        {/* what the whole run would gain if nothing were left alone */}
        <line x1={X(1)} y1={Y(1)} x2={X(1000)} y2={Y(1000)} stroke="var(--ink-3)" strokeWidth="1" strokeDasharray="2 4" />
        {/* the label gives way when the ceiling line would run through it */}
        {Math.abs(Y(top) - (Y(1000) - 11)) < 13 ? null : (
          <text x={X(1000) - 6} y={Y(1000) - 7} fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi="nếu p là 100%">if p were 100%</Translated></text>
        )}
        {/* the ceiling 1 / (1 - p) */}
        <line x1={pad.l} x2={W - pad.r} y1={Y(top)} y2={Y(top)} stroke="var(--ink)" strokeWidth="1.2" strokeDasharray="6 4" />
        <text x={pad.l + 8} y={Y(top) - 6} fontSize="11.5" data-out="plot-ceiling"><Translated vi={<>trần {times(top)}</>}>ceiling {times(top)}</Translated></text>
        {/* overall speed-up against s, and where it ends up when s has no bound */}
        <path d={path(curve)} fill="none" stroke="var(--ink)" strokeWidth="2.2" />
        <circle cx={xInf} cy={Y(top)} r="4.5" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.8" data-limit />
        {/* where the controls are now */}
        <line x1={px} x2={px} y1={py} y2={H - pad.b} stroke="var(--ink-3)" strokeWidth="1" />
        <circle cx={px} cy={py} r="5.5" fill="var(--ink)" stroke="var(--page)" strokeWidth="1.5" data-point />
      </svg>
    </div>
  );
}

function AmdahlBench({ p, setP }: { p: number; setP: (p: number) => void }) {
  const [finite, setFinite] = useState(100);
  const [unbounded, setUnbounded] = useState(false);
  const s = unbounded ? Infinity : finite;
  const now = speedup(p, s);
  const top = ceiling(p);
  const r = remaining(p, s);
  const need = kernelNeeded(100, p);
  const preset = (c: (typeof BOOK_CASES)[number]) => {
    setP(c.p);
    setUnbounded(!isFinite(c.s));
    if (isFinite(c.s)) setFinite(c.s);
  };
  return (
    <Bench
      title={tr("Amdahl's law", 'Định luật Amdahl')}
      task={tr(
        'Find the smallest parallel fraction that lets the overall figure reach 100× at all.',
        'Tìm phần song song nhỏ nhất cho phép con số tổng thể đạt được 100×.',
      )}
      foot={tr(
        'Log scales on both axes of the plot. Heavy line: overall speed-up; the filled dot is where the controls are now. Dashed line: the ceiling, 1 / (1 − p), which the open dot in the last column reaches. Dotted diagonal: what the run would gain if none of it were left alone. The bars are drawn to one scale except the last, which stretches the remaining time to full width.',
        'Thang log trên cả hai trục của đồ thị. Đường đậm: mức tăng tốc tổng thể; chấm đặc là vị trí hiện tại của các nút điều khiển. Đường đứt: trần, 1 / (1 − p), mà chấm rỗng ở cột cuối chạm tới. Đường chéo chấm: lần chạy sẽ được lợi bao nhiêu nếu không phần nào bị để nguyên. Các thanh được vẽ cùng một tỷ lệ trừ thanh cuối, thanh này kéo giãn thời gian còn lại ra hết chiều rộng.',
      )}
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <div className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
          <Translated vi="Các trường hợp của sách (p. 9)">The book's cases (p. 9)</Translated>
          <span className="seg" data-presets>
            {BOOK_CASES.map((c) => (
              <button key={c.label} aria-pressed={p === c.p && s === c.s} onClick={() => preset(c)}>
                {tr(c.label, c.label.replace('unbounded', 'không giới hạn'))}
              </button>
            ))}
          </span>
        </div>
        <FractionSlider p={p} setP={setP} />
        <label className="field">
          <span>
            <Translated vi={<>Mức tăng tốc của phần đó s = <b data-out="s">{times(s)}</b></>}>
              Speed-up of that part s = <b data-out="s">{times(s)}</b>
            </Translated>
          </span>
          <input
            type="range"
            aria-label={tr('speed-up of the accelerated part', 'mức tăng tốc của phần được tăng tốc')}
            min="0"
            max={SPEEDUPS.length - 1}
            step="1"
            disabled={unbounded}
            value={nearestIndex(SPEEDUPS, finite)}
            onChange={(e) => setFinite(SPEEDUPS[+e.target.value])}
            style={{ width: '13rem' }}
          />
        </label>
        <label className="check">
          <input type="checkbox" aria-label={tr('unbounded speed-up', 'mức tăng tốc không giới hạn')} checked={unbounded} onChange={(e) => setUnbounded(e.target.checked)} />
          <Translated vi="không giới hạn">unbounded</Translated>
        </label>
      </div>
      <TimeBars p={p} s={s} />
      <div className="cols plot">
        <AmdahlPlot p={p} s={s} />
        <div className="panel" data-readout>
          <p className="muted" style={{ margin: 0 }}><Translated vi="Mức tăng tốc tổng thể">Overall speed-up</Translated></p>
          <p style={{ fontSize: '1.9rem', fontWeight: 600, margin: '0 0 0.5rem', lineHeight: 1.15 }} data-out="overall">{times(now)}</p>
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              <tr>
                <td><Translated vi="Trần, 1 / (1 − p)">Ceiling, 1 / (1 − p)</Translated></td>
                <td className="num" data-out="ceiling">{times(top)}</td>
              </tr>
              <tr>
                <td><Translated vi="Thời gian chạy còn lại, so với ban đầu">Run time left, of the original</Translated></td>
                <td className="num">{share(r.total)}</td>
              </tr>
              <tr>
                <td><Translated vi="Phần không đụng tới, tính theo tỷ lệ thời gian còn lại">Untouched part, as a share of the time left</Translated></td>
                <td className="num">{share(untouchedShare(p, s))}</td>
              </tr>
              <tr>
                <td><Translated vi="Giá trị s mà 100× tổng thể cần">The s that 100× overall would need</Translated></td>
                <td className="num nowrap" data-out="needed">
                  {need === null ? tr('none is enough', 'không giá trị nào đủ') : isFinite(need) ? times(need) : tr('unbounded', 'không giới hạn')}
                </td>
              </tr>
            </tbody>
          </table>
          <Translated
            vi={
              <p className="muted" style={{ margin: '0.6rem 0 0' }}>
                Con số thứ ba cũng cho biết bạn đã chạm tới bao nhiêu phần của trần: phần thời gian còn lại không bị đụng tới càng
                nhiều, thì một s nhanh hơn càng thêm được ít.
              </p>
            }
          >
            <p className="muted" style={{ margin: '0.6rem 0 0' }}>
              The third figure is also how much of the ceiling you have reached: the more of the remaining time is
              untouched, the less a faster s can add.
            </p>
          </Translated>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer 2: what caps the kernel
function CapsPlot({ p, threads, rate, cap }: { p: number; threads: number; rate: number; cap: number }) {
  const [ref, W] = useWidth(600);
  const H = Math.round(Math.min(340, Math.max(260, W * 0.58)));
  const pad = { l: 52, r: 12, t: 24, b: 44 };
  const xLo = 1;
  const xHi = 100000;
  const yLo = 0.1;
  const yHi = 10000;
  const X = (v: number) => pad.l + ((Math.log10(Math.min(Math.max(v, xLo), xHi)) - Math.log10(xLo)) / 5) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(Math.min(Math.max(v, yLo), yHi)) - Math.log10(yLo)) / 5) * (H - pad.t - pad.b);
  const c = corners(rate, cap);
  const pts = limitsCurve(p, rate, cap);
  const now = limits(p, threads, rate, cap);
  const top = ceiling(p);
  const atSat = speedup(p, cap);
  const satRight = X(c.saturation) > W - pad.r - 120;
  return (
    <div ref={ref} style={{ width: '100%', overflow: 'hidden' }}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Overall speed-up against the number of threads', 'Mức tăng tốc tổng thể theo số luồng')} style={{ display: 'block' }}>
        <defs>
          <pattern id="s13-hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="7" stroke="var(--rule)" strokeWidth="2" />
          </pattern>
        </defs>
        <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
        {/* three stretches of the thread axis */}
        {c.breakEven > xLo ? <rect x={X(xLo)} y={pad.t} width={X(c.breakEven) - X(xLo)} height={H - pad.t - pad.b} fill="url(#s13-hatch)" data-region="slower" /> : null}
        <rect x={X(c.saturation)} y={pad.t} width={W - pad.r - X(c.saturation)} height={H - pad.t - pad.b} fill="var(--global-wash)" data-region="memory" />
        {[1, 10, 100, 1000, 10000, 100000].map((v) => (
          <g key={'x' + v}>
            <line x1={X(v)} x2={X(v)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
            <text x={X(v)} y={H - pad.b + 15} fontSize="11" textAnchor={v === xHi ? 'end' : 'middle'}>{tick(v)}</text>
          </g>
        ))}
        {[0.1, 1, 10, 100, 1000, 10000].map((v) => (
          <g key={'y' + v}>
            <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke={v === 1 ? 'var(--ink-3)' : 'var(--rule-soft)'} />
            <text x={pad.l - 6} y={Y(v) + 4} fontSize="11" textAnchor="end">{tick(v)}</text>
          </g>
        ))}
        <text x={(W + pad.l) / 2} y={H - 8} fontSize="11.5" textAnchor="middle"><Translated vi="số luồng, n">threads, n</Translated></text>
        <text x={14} y={(H - pad.b + pad.t) / 2} fontSize="11.5" textAnchor="middle" transform={`rotate(-90 14 ${(H - pad.b + pad.t) / 2})`}><Translated vi="mức tăng tốc so với một lõi CPU">speed-up over one CPU core</Translated></text>
        {/* the ceiling from the untouched part */}
        <line x1={pad.l} x2={W - pad.r} y1={Y(top)} y2={Y(top)} stroke="var(--ink)" strokeWidth="1.2" strokeDasharray="6 4" />
        <text x={pad.l + 8} y={Y(top) - 6} fontSize="11.5"><Translated vi={<>trần {times(top)}</>}>ceiling {times(top)}</Translated></text>
        {/* the kernel alone: min(n r, M) */}
        <path d={path(pts.map((q): [number, number] => [X(q.threads), Y(q.s)]))} fill="none" stroke="var(--ink-2)" strokeWidth="1.2" />
        {/* the label gives way when the ceiling line would run through it */}
        {Math.abs(Y(top) - (Y(cap) - 10)) < 12 ? null : (
          <text x={W - pad.r - 6} y={Y(cap) - 6} fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>riêng kernel {times(cap)}</>}>kernel alone {times(cap)}</Translated></text>
        )}
        {/* the whole run */}
        <path d={path(pts.map((q): [number, number] => [X(q.threads), Y(q.overall)]))} fill="none" stroke="var(--ink)" strokeWidth="2.4" />
        {/* the two corners */}
        {c.breakEven > xLo ? (
          <g data-corner="break-even">
            <circle cx={X(c.breakEven)} cy={Y(1)} r="4.5" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.8" />
            {cap >= 4 ? <text x={X(c.breakEven) + 8} y={Y(1) + 16} fontSize="11"><Translated vi="hòa vốn">break-even</Translated></text> : null}
          </g>
        ) : null}
        <g data-corner="saturation">
          <circle cx={X(c.saturation)} cy={Y(atSat)} r="4.5" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.8" />
          <text
            x={satRight ? W - pad.r - 6 : X(c.saturation) + 8}
            y={H - pad.b - 8}
            fontSize="11"
            textAnchor={satRight ? 'end' : 'start'}
            style={{ fill: 'var(--global)' }}
          >
            <Translated vi="bộ nhớ bão hòa">memory saturated</Translated>
          </text>
        </g>
        {/* where the controls are now */}
        <line x1={X(threads)} x2={X(threads)} y1={Y(now.overall)} y2={H - pad.b} stroke="var(--ink-3)" strokeWidth="1" />
        <circle cx={X(threads)} cy={Y(now.overall)} r="5.5" fill="var(--ink)" stroke="var(--page)" strokeWidth="1.5" data-point />
      </svg>
    </div>
  );
}

function CapsBench({ p, setP }: { p: number; setP: (p: number) => void }) {
  const [threads, setThreads] = useState(DEFAULTS.threads);
  const [rate, setRate] = useState(DEFAULTS.rate);
  const [cap, setCap] = useState(DEFAULTS.cap);
  const now = limits(p, threads, rate, cap);
  const c = corners(rate, cap);
  const lv = levers(p, threads, rate, cap);
  const top = ceiling(p);
  const reset = () => {
    setP(DEFAULT_P);
    setThreads(DEFAULTS.threads);
    setRate(DEFAULTS.rate);
    setCap(DEFAULTS.cap);
  };
  const tickAt = SPEEDUPS.indexOf(10) / (SPEEDUPS.length - 1);
  const lever = (id: string, label: string, v: number) => (
    <tr data-lever={id}>
      <td>{label}</td>
      <td className="num nowrap">{times(v)}</td>
      <td className="muted nowrap">{v <= lv.now * 1.001 ? tr('no change', 'không đổi') : tr(`${fmt(v / lv.now)} times more`, `gấp ${fmt(v / lv.now)} lần`)}</td>
    </tr>
  );
  const goal = (target: number) => (
    <span className={now.overall > target ? 'chip ok' : 'chip'} data-goal={target} style={now.overall > target ? undefined : { border: '1px solid var(--rule)' }}>
      <Translated vi={<>vượt {target}×: {now.overall > target ? 'đã đạt' : 'chưa đạt'}</>}>
        past {target}×: {now.overall > target ? 'reached' : 'not yet'}
      </Translated>
    </span>
  );
  return (
    <Bench
      title={tr('What caps the kernel', 'Điều gì chặn trần kernel')}
      task={tr(
        'From the defaults, get the overall figure past 50×, then past 100×. Note which control you had to move each time.',
        'Từ giá trị mặc định, đưa con số tổng thể vượt 50×, rồi vượt 100×. Ghi lại mỗi lần bạn phải dịch nút điều khiển nào.',
      )}
      foot={tr(
        'Log scales on both axes. Heavy line: the whole run. Thin line: the kernel alone, min(n × r, M). Dashed line: the ceiling set by the untouched part. Hatched: too few threads to match the CPU core. Shaded: thread counts at which memory is saturated.',
        'Thang log trên cả hai trục. Đường đậm: toàn bộ lần chạy. Đường mảnh: riêng kernel, min(n × r, M). Đường đứt: trần do phần không đụng tới đặt ra. Vùng gạch chéo: quá ít luồng để bằng lõi CPU. Vùng tô: số luồng mà tại đó bộ nhớ bão hòa.',
      )}
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-start' }}>
        <label className="field">
          <span>
            <Translated vi={<>Số luồng n = <b data-out="n">{count(threads)}</b></>}>
              Threads n = <b data-out="n">{count(threads)}</b>
            </Translated>
          </span>
          <input
            type="range"
            aria-label={tr('threads', 'số luồng')}
            min="0"
            max={THREADS.length - 1}
            step="1"
            value={nearestIndex(THREADS, threads)}
            onChange={(e) => setThreads(THREADS[+e.target.value])}
            style={{ width: '15rem' }}
          />
        </label>
        <div className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
          <Translated vi="Một luồng GPU so với lõi CPU, r">One GPU thread against the CPU core, r</Translated>
          <span className="seg" data-rates>
            {RATES.map((o) => (
              <button key={o.label} aria-pressed={rate === o.value} onClick={() => setRate(o.value)}>
                {o.label}
              </button>
            ))}
          </span>
        </div>
        <label className="field" style={{ position: 'relative', paddingBottom: '1.15rem' }}>
          <span>
            <Translated vi={<>Trần bộ nhớ M = <b data-out="M">{times(cap)}</b></>}>
              Memory cap M = <b data-out="M">{times(cap)}</b>
            </Translated>
          </span>
          <input
            type="range"
            aria-label={tr('memory cap', 'trần bộ nhớ')}
            min="0"
            max={SPEEDUPS.length - 1}
            step="1"
            value={nearestIndex(SPEEDUPS, cap)}
            onChange={(e) => setCap(SPEEDUPS[+e.target.value])}
            style={{ width: '17rem' }}
          />
          <span
            data-tick="book"
            style={{
              position: 'absolute', bottom: 0, left: `calc(${tickAt * 100}% + ${(8 - 16 * tickAt).toFixed(1)}px - 0.36em)`,
              fontSize: '0.72rem', whiteSpace: 'nowrap', pointerEvents: 'none',
            }}
          >
            <Translated vi="▲ quy tắc kinh nghiệm của sách, p. 10">▲ the book's rule of thumb, p. 10</Translated>
          </span>
        </label>
        <FractionSlider p={p} setP={setP} />
        <button className="quiet" onClick={reset} style={{ alignSelf: 'center' }}>
          <Translated vi="Về giá trị mặc định">Back to the defaults</Translated>
        </button>
      </div>
      <div className="cols plot">
        <CapsPlot p={p} threads={threads} rate={rate} cap={cap} />
        <div className="panel" data-readout>
          <p className="muted" style={{ margin: 0 }}><Translated vi="Mức tăng tốc kernel, s = min(n × r, M)">Kernel speed-up, s = min(n × r, M)</Translated></p>
          <p style={{ margin: '0 0 0.6rem' }} data-out="kernel">
            <Translated
              vi={
                <>
                  min({fmt(threads * rate)}, {fmt(cap)}) = <b>{times(now.s)}</b>, do{' '}
                  {now.limitedBy === 'memory' ? <span className="chip global">bộ nhớ</span> : <b>số luồng</b>} quyết định
                </>
              }
            >
              min({fmt(threads * rate)}, {fmt(cap)}) = <b>{times(now.s)}</b>, set by{' '}
              {now.limitedBy === 'memory' ? <span className="chip global">memory</span> : <b>the thread count</b>}
            </Translated>
          </p>
          <p className="muted" style={{ margin: 0 }}><Translated vi="Mức tăng tốc tổng thể">Overall speed-up</Translated></p>
          <p style={{ fontSize: '1.9rem', fontWeight: 600, margin: '0 0 0.4rem', lineHeight: 1.15 }} data-out="overall">{times(now.overall)}</p>
          {now.s < 1 ? (
            <p style={{ margin: '0 0 0.5rem' }} data-out="slower">
              <Translated vi="Với số luồng ít như vậy, kernel chậm hơn lõi CPU mà nó thay thế.">
                With this few threads the kernel is slower than the CPU core it replaces.
              </Translated>
            </p>
          ) : null}
          <p style={{ margin: '0 0 0.6rem' }} data-out="detail">
            <Translated
              vi={
                <>
                  Trần do phần không đụng tới: {times(top)}. Phần không đụng tới chiếm {share(now.untouchedShare)} thời gian
                  còn lại. Hòa vốn ở {count(c.breakEven)} luồng; bộ nhớ bão hòa ở {count(c.saturation)}.
                </>
              }
            >
              Ceiling from the untouched part: {times(top)}. The untouched part is {share(now.untouchedShare)} of the time
              left. Break-even at {count(c.breakEven)} threads; memory saturates at {count(c.saturation)}.
            </Translated>
          </p>
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr>
                <th><Translated vi="Một thay đổi từ đây">One change from here</Translated></th>
                <th className="num"><Translated vi="Tổng thể">Overall</Translated></th>
                <th />
              </tr>
            </thead>
            <tbody>
              {lever('threads', tr('10× the threads', '10× số luồng'), lv.threads)}
              {lever('memory', tr('10× the memory cap', '10× trần bộ nhớ'), lv.memory)}
              {lever('untouched', tr('1/10 of the untouched part', '1/10 phần không đụng tới'), lv.untouched)}
            </tbody>
          </table>
          <p style={{ margin: '0.7rem 0 0' }} className="row" data-goals>
            {goal(50)}
            {goal(100)}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- classification
type Challenge = 'work' | 'memory' | 'input' | 'sync';
const CHALLENGES: { id: Challenge; label: string }[] = [
  { id: 'work', get label() { return tr('Algorithm and work', 'Thuật toán và khối lượng việc'); } },
  { id: 'memory', get label() { return tr('Memory-bound', 'Bị giới hạn bởi bộ nhớ'); } },
  { id: 'input', get label() { return tr('Input sensitivity', 'Nhạy cảm với đầu vào'); } },
  { id: 'sync', get label() { return tr('Synchronisation', 'Đồng bộ hóa'); } },
];

interface Card {
  id: string;
  text: ReactNode;
  answer: Challenge;
  /** Shown once the card is placed: why, and where the course deals with it. */
  done: ReactNode;
  /** Shown for each wrong choice. */
  why: Partial<Record<Challenge, ReactNode>>;
}

// All eight situations are ours, not the book's. Each is justified from the notes of
// the chapter it names; the comment above a card says where.
const CARDS: Card[] = [
  // ch. 1 p. 11: recurrences need reformulating and prefix sum is the standard tool.
  // notes/ch11.md, "Relevance": top-p as an inclusive scan over sorted probabilities (our mapping).
  {
    id: 'topp',
    text: (
      <Translated
        vi={
          <>
            Lấy mẫu top-p cần tổng tích lũy của các xác suất đã sắp xếp. Viết thành vòng lặp, mỗi tổng riêng phần là tổng trước
            nó cộng thêm một số hạng.
          </>
        }
      >
        <>
          Top-p sampling needs the running sum of the sorted probabilities. Written as a loop, each partial sum is the
          one before it plus one more term.
        </>
      </Translated>
    ),
    answer: 'work',
    done: (
      <Translated
        vi={
          <>
            Một hệ thức truy hồi: bước i cần bước i − 1, nên như đang viết thì luồng thứ hai không có gì để làm.{' '}
            <a href={href('ch11')}>Chương 11</a> phát biểu lại nó thành tổng tiền tố (prefix sum) song song, và module X1 dùng
            điều đó cho top-p.
          </>
        }
      >
        <>
          A recurrence: step i needs step i − 1, so as written there is nothing for a second thread to do.{' '}
          <a href={href('ch11')}>Chapter 11</a> reformulates it as a parallel prefix sum, and module X1 uses that for
          top-p.
        </>
      </Translated>
    ),
    why: {
      memory: <Translated vi={<>Một hàng xác suất không có nhiều thứ để nạp. Trở ngại là mỗi tổng dường như cần tổng trước nó.</>}>One row of probabilities is not much to fetch. The obstacle is that each sum seems to need the one before it.</Translated>,
      input: <Translated vi={<>Mọi hàng đều có cùng chuỗi phụ thuộc, bất kể nó chứa giá trị gì.</>}>Every row has the same chain of dependences, whatever values it holds.</Translated>,
      sync: <Translated vi={<>Chờ đợi là cái giá của một số lời giải. Khó khăn đến trước: như đang viết, không có hai bước nào của vòng lặp chạy cùng lúc được.</>}>Waiting is what some solutions cost. The difficulty comes first: as written, no two steps of the loop can run at once.</Translated>,
    },
  },
  // ch. 5 §5.1 (pp. 94–97), the FLOP/B count applied to a matrix-vector product;
  // ch. 20 §20.6 (p. 505) reaches the same conclusion for decoding. See ch05/5.1/q-decode.
  {
    id: 'matvec',
    text: (
      <Translated
        vi={
          <>
            Sinh một token với batch 1: mỗi ma trận trọng số nhân với một vector duy nhất, nên mỗi trọng số được nạp một lần và
            dùng một lần.
          </>
        }
      >
        <>
          Generating one token at batch size 1: each weight matrix multiplies a single vector, so every weight is fetched
          once and used once.
        </>
      </Translated>
    ),
    answer: 'memory',
    done: (
      <Translated
        vi={
          <>
            Hai phép toán cho mỗi trọng số được nạp, dù kernel viết thế nào, nên tốc độ do trọng số đến nhanh cỡ nào quyết định.{' '}
            <a href={href('ch05', '5.1')}>Chương 5, §5.1</a> đưa ra phép đếm; batching và lượng tử hóa là những lối thoát.
          </>
        }
      >
        <>
          Two operations per weight fetched, however the kernel is written, so the speed is set by how fast weights
          arrive. <a href={href('ch05', '5.1')}>Chapter 5, §5.1</a> gives the count; batching and quantisation are the
          ways out.
        </>
      </Translated>
    ),
    why: {
      work: <Translated vi={<>Kernel song song thực hiện đúng các phép nhân và cộng như vòng lặp, không hơn.</>}>The parallel kernel performs the same multiplications and additions as the loop, no more.</Translated>,
      input: <Translated vi={<>Trọng số dày đặc: mọi luồng nhận cùng một lượng việc.</>}>Dense weights: every thread gets the same amount of work.</Translated>,
      sync: <Translated vi={<>Mỗi phần tử đầu ra thuộc về một luồng. Không ai chờ ai.</>}>Each output element belongs to one thread. Nobody waits for anybody.</Translated>,
    },
  },
  // ch. 20 §20.6 (p. 506): unequal sequence lengths may unbalance work across thread
  // blocks and under-use the GPU; in-flight batching is the serving-side answer.
  {
    id: 'lengths',
    text: <Translated vi={<>Một lần khởi chạy phục vụ một batch ba prompt dài 12, 900 và 4,000 token.</>}>One launch serves a batch of three prompts that are 12, 900 and 4,000 tokens long.</Translated>,
    answer: 'input',
    done: (
      <Translated
        vi={
          <>
            Độ dài không đều giao cho các khối luồng lượng việc không đều và để một phần GPU ngồi không. Sách ghi nhận điều này
            cho batch LLM ở <a href={href('ch20')}>chương 20</a> (p. 506).
          </>
        }
      >
        <>
          Unequal lengths give thread blocks unequal work and leave part of the GPU idle. The book notes this for LLM
          batches in <a href={href('ch20')}>chapter 20</a> (p. 506).
        </>
      </Translated>
    ),
    why: {
      work: <Translated vi={<>Thuật toán không thay đổi gì. Cùng kernel đó trên ba prompt bằng nhau sẽ cân bằng.</>}>Nothing about the algorithm changed. The same kernel on three equal prompts would be balanced.</Translated>,
      memory: <Translated vi={<>Có thể cả điều đó, nhưng nó cũng đúng y như vậy với ba prompt bằng nhau. Điều đặc biệt ở đây là sự pha trộn độ dài.</>}>Perhaps that too, but it would be just as true of three equal prompts. What is special here is the mix of lengths.</Translated>,
      sync: <Translated vi={<>Không luồng nào cần kết quả của luồng khác. Thời gian ngồi không đến từ lượng việc không đều, không phải từ việc chờ ở rào chắn (barrier).</>}>No thread needs another's result. The idle time comes from uneven work, not from waiting at a barrier.</Translated>,
    },
  },
  // ch. 9 §9.2–9.3 (pp. 206, 209): atomic updates to one location are serialised.
  // notes/ch09.md, "Relevance": token-count histograms for repetition penalties (our mapping).
  {
    id: 'counters',
    text: (
      <Translated
        vi={
          <>
            Một hình phạt lặp lại (repetition penalty) cần số đếm cho mỗi token id đã gặp. Mỗi luồng ứng với một token ngữ cảnh
            tăng bộ đếm cho token của nó, và một token thường gặp bị nhiều luồng cùng đụng tới.
          </>
        }
      >
        <>
          A repetition penalty needs a count for every token id seen so far. One thread per context token increments the
          counter for its token, and a frequent token is hit by many threads.
        </>
      </Translated>
    ),
    answer: 'sync',
    done: (
      <Translated
        vi={
          <>
            Các cập nhật phải là nguyên tử (atomic), và các cập nhật nguyên tử vào một vị trí chạy lần lượt trong khi các luồng
            khác chờ. <a href={href('ch09')}>Chương 9</a> nói về cái giá này và cách giảm nó.
          </>
        }
      >
        <>
          The updates must be atomic, and atomic updates to one location run one after another while the other threads
          wait. <a href={href('ch09')}>Chapter 9</a> is about this cost and how to reduce it.
        </>
      </Translated>
    ),
    why: {
      work: <Translated vi={<>Đếm N token cần N lần tăng, song song cũng như tuần tự. Cái giá nằm ở việc làm cho chúng an toàn.</>}>Counting N tokens takes N increments in parallel as in sequence. The cost is in making them safe.</Translated>,
      memory: <Translated vi={<>Các bộ đếm nhỏ. Điều làm chậm các luồng là phải chờ lượt, không phải việc nạp dữ liệu.</>}>The counters are small. What slows the threads is taking turns, not fetching.</Translated>,
      input: <Translated vi={<>Phân bố token quyết định tình hình tệ đến mức nào, và chương 9 cũng nói vậy. Bản thân cái giá là các luồng phải chờ lượt trên một bộ đếm.</>}>The token distribution decides how bad it gets, and chapter 9 says as much. The cost itself is threads taking turns on a counter.</Translated>,
    },
  },
  // ch. 11 §11.5 (pp. 265–266): Kogge-Stone performs N·log2N − (N − 1) additions against
  // a minimum of N − 1; for 512 elements, between 8 and 9 times the sequential work.
  {
    id: 'worksum',
    text: (
      <Translated
        vi={
          <>
            Một phép tổng tiền tố song song trên N giá trị xong trong log<sub>2</sub> N bước. Nó thực hiện khoảng N ×{' '}
            log<sub>2</sub> N phép cộng, trong khi vòng lặp tuần tự thực hiện N − 1.
          </>
        }
      >
        <>
          A parallel prefix sum over N values finishes in log<sub>2</sub> N steps. It performs about N × log<sub>2</sub> N
          additions, where the sequential loop performs N − 1.
        </>
      </Translated>
    ),
    answer: 'work',
    done: (
      <Translated
        vi={
          <>
            Tổng khối lượng việc nhiều hơn mã tuần tự: với 512 giá trị, gấp từ 8 đến 9 lần (pp. 265–266).{' '}
            <a href={href('ch11')}>Chương 11</a> gọi thước đo này là hiệu quả công việc (work efficiency), rồi sửa nó.
          </>
        }
      >
        <>
          More total work than the sequential code: for 512 values, between 8 and 9 times as much (pp. 265–266).{' '}
          <a href={href('ch11')}>Chapter 11</a> calls the measure work efficiency, then repairs it.
        </>
      </Translated>
    ),
    why: {
      memory: <Translated vi={<>Đếm phép cộng, không đếm byte. Thẻ này nói về tổng cộng có bao nhiêu phép tính.</>}>Count the additions, not the bytes. The card is about how much arithmetic there is in total.</Translated>,
      input: <Translated vi={<>Số đếm chỉ phụ thuộc vào N, không phụ thuộc vào các giá trị.</>}>The count depends on N only, not on the values.</Translated>,
      sync: <Translated vi={<>Các bước đúng là cần rào chắn giữa chúng. Nhưng con số trong thẻ là số phép cộng được thực hiện, và nó lớn hơn của vòng lặp.</>}>The steps do need barriers between them. But the number in the card is additions performed, and it is larger than the loop's.</Translated>,
    },
  },
  // ch. 17 p. 401 (irregularity costs: load imbalance, control divergence) and §17.3
  // pp. 408–410 (CSR: one thread per row; the trip count is the row length).
  {
    id: 'sparse',
    text: <Translated vi={<>Một ma trận thưa có từ 1 đến 10,000 phần tử khác không mỗi hàng. Kernel giao mỗi hàng cho một luồng.</>}>A sparse matrix has between 1 and 10,000 non-zeros per row. The kernel gives each row to one thread.</Translated>,
    answer: 'input',
    done: (
      <Translated
        vi={
          <>
            Mỗi luồng lặp qua hàng của riêng nó, nên các luồng cạnh nhau làm lượng việc rất khác nhau.{' '}
            <a href={href('ch17')}>Chương 17</a> gọi tên cái giá này (mất cân bằng tải và phân kỳ điều khiển, pp. 401 và 410)
            và làm dữ liệu đều đặn hơn để giảm nó.
          </>
        }
      >
        <>
          Each thread loops over its own row, so neighbouring threads do very different amounts of work.{' '}
          <a href={href('ch17')}>Chapter 17</a> names the cost (load imbalance and control divergence, pp. 401 and 410)
          and makes the data more regular to reduce it.
        </>
      </Translated>
    ),
    why: {
      work: <Translated vi={<>Mỗi phần tử khác không được nhân một lần, như trong vòng lặp tuần tự.</>}>Each non-zero is multiplied once, as in the sequential loop.</Translated>,
      memory: <Translated vi={<>Kernel thưa cũng thiếu băng thông, nhưng điều đó vẫn đúng với các hàng bằng nhau. Độ chênh từ 1 đến 10,000 mới là điểm chính ở đây.</>}>Sparse kernels are short of bandwidth too, but that would hold with equal rows. The spread from 1 to 10,000 is the point here.</Translated>,
      sync: <Translated vi={<>Kết quả của mỗi hàng thuộc về một luồng, nên không có cập nhật nguyên tử và không có rào chắn.</>}>Each row's result belongs to one thread, so there are no atomic updates and no barriers.</Translated>,
    },
  },
  // ch. 5 §5.4 (pp. 108–112): the tiled kernel and its two __syncthreads().
  {
    id: 'barrier',
    text: (
      <Translated
        vi={
          <>
            Trong kernel nhân ma trận phân ô, mỗi luồng của một khối nạp một phần tử vào ô (tile) của khối, rồi dừng lại cho đến
            khi tất cả các luồng khác đã nạp xong phần của chúng.
          </>
        }
      >
        <>
          In the tiled matrix multiplication kernel, every thread of a block loads one element into the block's tile,
          then stops until all the others have loaded theirs.
        </>
      </Translated>
    ),
    answer: 'sync',
    done: (
      <Translated
        vi={
          <>
            Một rào chắn: các luồng đến sớm phải chờ thay vì làm việc. <a href={href('ch05', '5.4')}>Chương 5, §5.4</a> cho thấy
            vì sao kernel cần hai rào chắn như vậy.
          </>
        }
      >
        <>
          A barrier: threads that arrive early wait instead of working. <a href={href('ch05', '5.4')}>Chapter 5, §5.4</a>{' '}
          shows why the kernel needs two of them.
        </>
      </Translated>
    ),
    why: {
      work: <Translated vi={<>Kernel phân ô thực hiện đúng các phép nhân-cộng như kernel ngây thơ.</>}>The tiled kernel performs the same multiply-adds as the naive one.</Translated>,
      memory: <Translated vi={<>Phân ô là cách chữa vấn đề bộ nhớ. Điểm dừng là cái giá nó phải trả, và cái giá đó có tên riêng.</>}>Tiling is the cure for the memory problem. The stop is the price it pays, and the price has its own name.</Translated>,
      input: <Translated vi={<>Ma trận dày đặc: mọi luồng làm cùng một việc.</>}>Dense matrices: every thread does the same thing.</Translated>,
    },
  },
  // ch. 5 §5.1 (p. 96): vector addition does 1 FLOP per 12 bytes, 0.083 FLOP/B, deeply
  // memory-bound; ch. 1 p. 12 for "embarrassingly parallel".
  {
    id: 'bias',
    text: (
      <Translated
        vi={
          <>
            Cộng một bias vào mọi phần tử của một tensor lớn. Không luồng nào cần gì từ luồng nào khác. Một trong bốn khó khăn
            vẫn áp dụng.
          </>
        }
      >
        <>
          Adding a bias to every element of a large tensor. No thread needs anything from any other thread. One of the
          four still applies.
        </>
      </Translated>
    ),
    answer: 'memory',
    done: (
      <Translated
        vi={
          <>
            Giống phép cộng vector của sách, thực hiện một phép toán cho mỗi 12 byte nó di chuyển (
            <a href={href('ch05', '5.1')}>chương 5, §5.1</a>, p. 96): song song hiển nhiên (embarrassingly parallel), vậy mà vẫn
            bị bộ nhớ kìm lại.
          </>
        }
      >
        <>
          Like the book's vector addition, which performs one operation for every 12 bytes it moves (
          <a href={href('ch05', '5.1')}>chapter 5, §5.1</a>, p. 96): embarrassingly parallel, and held back by memory all
          the same.
        </>
      </Translated>
    ),
    why: {
      work: <Translated vi={<>Một phép cộng cho mỗi phần tử, song song cũng như tuần tự.</>}>One addition per element, in parallel as in sequence.</Translated>,
      input: <Translated vi={<>Mọi phần tử tốn như nhau.</>}>Every element costs the same.</Translated>,
      sync: <Translated vi={<>Không có gì phải trả. Thuật ngữ của sách cho điều đó là embarrassingly parallel, và đó là lý do một thứ khác quyết định tốc độ.</>}>There is none to pay. The book's term for that is embarrassingly parallel, and it is why something else sets the speed.</Translated>,
    },
  },
];

function WhichChallenge() {
  const [picks, setPicks] = useState<Record<string, Challenge | undefined>>({});
  const placed = CARDS.filter((c) => picks[c.id] === c.answer).length;
  return (
    <Bench
      title={tr('Which challenge is it?', 'Đó là khó khăn nào?')}
      task={tr(
        'Eight situations. Give each the one of the four challenges that fits it best.',
        'Tám tình huống. Gán cho mỗi tình huống khó khăn phù hợp nhất trong bốn khó khăn.',
      )}
      foot={tr(
        "The situations are ours, not the book's. The feedback names the chapter where this course deals with each.",
        'Các tình huống là của chúng tôi, không phải của sách. Phần phản hồi nêu tên chương mà khóa học này xử lý từng khó khăn.',
      )}
    >
      <div data-which>
        <div className="panel">
          <div className="row" style={{ justifyContent: 'space-between', marginBottom: '0.2rem' }}>
            <b><Translated vi="Tám tình huống">Eight situations</Translated></b>
            <span className="muted" data-which-score>
              <Translated vi={<>đã xếp {placed} trên {CARDS.length}</>}>{placed} of {CARDS.length} placed</Translated>
            </span>
          </div>
          {CARDS.map((c, n) => {
            const pick = picks[c.id];
            const done = pick === c.answer;
            return (
              <div
                key={c.id}
                data-card={c.id}
                data-solved={done ? 'true' : 'false'}
                style={{ padding: '0.75rem 0', borderTop: n ? '1px solid var(--rule-soft)' : undefined }}
              >
                <p style={{ margin: '0 0 0.5rem', maxWidth: '46rem', fontSize: '0.92rem' }}>
                  <span className="muted">{n + 1}.</span> {c.text}
                </p>
                <div className="row" style={{ gap: '0.4rem' }} role="group" aria-label={tr(`Challenge for situation ${n + 1}`, `Khó khăn cho tình huống ${n + 1}`)}>
                  {CHALLENGES.map((ch) => (
                    <button
                      key={ch.id}
                      className="quiet"
                      data-choice={ch.id}
                      aria-pressed={pick === ch.id}
                      disabled={done && ch.id !== c.answer}
                      onClick={() => (done ? undefined : setPicks((s) => ({ ...s, [c.id]: ch.id })))}
                    >
                      {ch.label}
                    </button>
                  ))}
                </div>
                {pick ? (
                  <p data-feedback={c.id} role="status" style={{ margin: '0.5rem 0 0', maxWidth: '46rem' }}>
                    {done ? (
                      <>
                        <span className="chip ok"><Translated vi="đã xếp">placed</Translated></span> {c.done}
                      </>
                    ) : (
                      <>
                        <span className="chip fail"><Translated vi="không phải cái này">not this one</Translated></span> {c.why[pick]}
                      </>
                    )}
                  </p>
                ) : null}
              </div>
            );
          })}
        </div>
        <div className="row" style={{ marginTop: '0.8rem', justifyContent: 'flex-end' }}>
          <button className="quiet" onClick={() => setPicks({})}>
            <Translated vi="Bắt đầu lại">Start again</Translated>
          </button>
        </div>
        {placed === CARDS.length ? (
          <div className="panel" style={{ marginTop: '0.8rem' }} data-which-done>
            <Translated
              vi={
                <p style={{ margin: 0 }}>
                  <b>Đã xếp cả tám.</b> Bốn cái tên là bộ từ vựng làm việc cho các kernel bạn sẽ viết: hãy nói kernel đang mắc khó khăn
                  nào trước khi quyết định thay đổi gì. Thẻ cuối là khuôn mẫu cần nhớ. Một kernel có thể dễ ở ba mặt và khó ở mặt thứ
                  tư.
                </p>
              }
            >
              <p style={{ margin: 0 }}>
                <b>All eight placed.</b> The four names are a working vocabulary for the kernels you will write: say
                which one a kernel suffers from before you decide what to change. The last card is the pattern to
                remember. A kernel can be easy on three counts and hard on the fourth.
              </p>
            </Translated>
          </div>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questions13: Record<string, Question> = {
  'ch01/1.3/q-fivehundred': {
    kind: 'numeric',
    id: 'ch01/1.3/q-fivehundred',
    prompt: (
      <Translated
        vi={
          <>
            Không kernel thật nào nhanh vô hạn. Giả sử bạn làm phần song song nhanh hơn {fmt(FIVE_HUNDRED.s)} lần. Phần song song p
            bằng bao nhiêu thì cho khoảng {FIVE_HUNDRED.target}× tổng thể? Trả lời với một chữ số thập phân.
          </>
        }
      >
        <>
          No real kernel is infinitely fast. Say you make the parallel part {fmt(FIVE_HUNDRED.s)} times faster. What
          parallel fraction p gives about {FIVE_HUNDRED.target}× overall? Give it to one decimal place.
        </>
      </Translated>
    ),
    answer: Number((100 * fractionNeeded(FIVE_HUNDRED.target, FIVE_HUNDRED.s)!).toFixed(1)),
    tol: 0.0001,
    unit: '%',
    explain: (
      <Translated
        vi={
          <p>
            {FIVE_HUNDRED.target}× tổng thể nghĩa là lần chạy mất 1/{FIVE_HUNDRED.target} = {fmt(100 / FIVE_HUNDRED.target)}% thời
            gian cũ. Phần được tăng tốc, nhanh hơn {fmt(FIVE_HUNDRED.s)} lần, mất khoảng p / {fmt(FIVE_HUNDRED.s)}, gần 0.1%.
            Còn lại khoảng 0.1% cho phần để nguyên, nên p khoảng 99.9% (đến bốn chữ số,{' '}
            {frac(fractionNeeded(FIVE_HUNDRED.target, FIVE_HUNDRED.s)!)}). Phần 0.1% không đụng tới giờ mất thời gian bằng toàn
            bộ phần việc được tăng tốc, giống như 1% không đụng tới trong dự đoán thứ hai của bạn. Vì thế 99% của nhiệm vụ chỉ
            làm 100× khả thi với một phần có tốc độ không giới hạn. Với một mức tăng tốc thật, hữu hạn, kết quả vài trăm lần cần
            "hơn 99.9%" như sách dẫn. Đặt trình khám phá Amdahl ở 99.9% và {times(FIVE_HUNDRED.s)}: nó hiện{' '}
            {times(speedup(0.999, FIVE_HUNDRED.s))}.
          </p>
        }
      >
        <p>
          {FIVE_HUNDRED.target}× overall means the run takes 1/{FIVE_HUNDRED.target} = {fmt(100 / FIVE_HUNDRED.target)}%
          of its old time. The accelerated part, {fmt(FIVE_HUNDRED.s)} times faster, takes about p /{' '}
          {fmt(FIVE_HUNDRED.s)}, close to 0.1%. That leaves about 0.1% for the part left alone, so p is about 99.9%
          (to four places, {frac(fractionNeeded(FIVE_HUNDRED.target, FIVE_HUNDRED.s)!)}). The untouched 0.1% now takes
          as long as all the accelerated work, as the untouched 1% did in your second prediction. So the 99% of the
          task makes 100× possible only with a part of unbounded speed. With a real, finite speed-up, results of
          several hundred times need the "more than 99.9%" the book quotes. Set the Amdahl explorer to 99.9% and{' '}
          {times(FIVE_HUNDRED.s)}: it shows {times(speedup(0.999, FIVE_HUNDRED.s))}.
        </p>
      </Translated>
    ),
    source: "our numbers; the law and the 99.9% figure are the book's, ch. 1 §1.3, pp. 9–10",
    vi: { source: 'số liệu của chúng tôi; định luật và con số 99.9% là của sách, ch. 1 §1.3, pp. 9–10' },
  },
  'ch01/1.3/q-kernelneeded': {
    kind: 'numeric',
    id: 'ch01/1.3/q-kernelneeded',
    prompt: (
      <Translated
        vi={
          <>
            Một bản profile cho thấy các kernel chiếm 96% thời gian của một bước decode. Bạn cần bước đó chạy nhanh hơn 20 lần từ
            đầu đến cuối, và bạn chỉ được thay đổi các kernel. Chúng phải nhanh hơn bao nhiêu lần?
          </>
        }
      >
        <>
          A profile shows that kernels take 96% of the time of a decode step. You need the step to run 20 times faster
          end to end, and you can only change the kernels. How many times faster must they become?
        </>
      </Translated>
    ),
    answer: 96,
    tol: 0.01,
    unit: '×',
    explain: (
      <Translated
        vi={
          <p>
            Một bước 20× được phép mất 1/20 = 0.05 thời gian ban đầu. Phần 4% không đụng tới dùng 0.04 trong đó, còn lại 0.01 cho
            các kernel: 0.96 / 0.01 = 96×. Nếu các kernel chỉ chiếm 95% của bước, phần 5% không đụng tới sẽ dùng hết ngân sách.
            Khi đó 20× chính là trần, và không mức tăng tốc kernel hữu hạn nào đạt tới.
          </p>
        }
      >
        <p>
          A 20× step may take 1/20 = 0.05 of the original time. The untouched 4% uses 0.04 of that, which leaves 0.01 for
          the kernels: 0.96 / 0.01 = 96×. Had the kernels been 95% of the step, the untouched 5% would use the whole
          budget. 20× would then be the ceiling itself, and no finite kernel speed-up reaches it.
        </p>
      </Translated>
    ),
    source: "our numbers; the law is the book's, ch. 1 §1.3, p. 9",
    vi: { source: 'số liệu của chúng tôi; định luật là của sách, ch. 1 §1.3, p. 9' },
  },
  'ch01/1.3/q-hostshare': {
    kind: 'numeric',
    id: 'ch01/1.3/q-hostshare',
    prompt: (
      <Translated
        vi={
          <>
            Trong một bước decode, việc lấy mẫu và các lần sao chép cấp dữ liệu cho nó chạy trên host và chiếm 8% thời gian. Mọi thứ
            khác là kernel. Dù bạn làm các kernel nhanh đến đâu, bước đó có thể được lợi tối đa bao nhiêu từ đầu đến cuối?
          </>
        }
      >
        <>
          In a decode step, sampling and the copies that feed it run on the host and take 8% of the time. Everything else
          is kernels. However fast you make the kernels, what is the most the step can gain end to end?
        </>
      </Translated>
    ),
    answer: 12.5,
    tol: 0.005,
    unit: '×',
    explain: (
      <Translated
        vi={
          <p>
            Trần là 1 / (1 − p) với p = 0.92: 1 / 0.08 = 12.5×. Vượt qua đó, nước đi duy nhất còn lại là thu nhỏ chính phần 8%.
            Đó là lý do chạy bộ lấy mẫu trên GPU, việc mà module X1 làm. Trong llama.cpp, chuỗi lấy mẫu mặc định chạy trên host.
          </p>
        }
      >
        <p>
          The ceiling is 1 / (1 − p) with p = 0.92: 1 / 0.08 = 12.5×. Past that, the only move left is to shrink the 8%
          itself. That is the case for running the sampler on the GPU, which module X1 does. In llama.cpp the default
          sampler chain runs on the host.
        </p>
      </Translated>
    ),
    source: "our framing; the law is the book's, ch. 1 §1.3, p. 9; llama.cpp checked 2026-11-20 (notes/today/ch01-facts.md, D2)",
    vi: {
      source: 'cách đặt vấn đề của chúng tôi; định luật là của sách, ch. 1 §1.3, p. 9; llama.cpp kiểm tra ngày 2026-11-20 (notes/today/ch01-facts.md, D2)',
    },
  },
  'ch01/1.3/q-bigslice': {
    kind: 'choice',
    id: 'ch01/1.3/q-bigslice',
    prompt: (
      <Translated
        vi={
          <>
            Bạn chỉ có thời gian cho một trong hai việc. Phương án A làm một kernel chiếm 60% thời gian chạy nhanh hơn 10 lần.
            Phương án B làm một kernel chiếm 30% thời gian chạy nhanh vô hạn. Phương án nào cho mức tăng tốc tổng thể lớn hơn?
          </>
        }
      >
        <>
          You have time for one of two jobs. Option A makes a kernel that takes 60% of the run time 10 times faster.
          Option B makes a kernel that takes 30% of the run time infinitely fast. Which gives the larger overall
          speed-up?
        </>
      </Translated>
    ),
    options: [
      { label: 'A.', correct: true },
      {
        label: 'B.',
        get why() {
          return tr(
            'Infinite beats ten on the part, but the part is only 30% of the time. The other 70% caps the whole at 1.43×.',
            'Vô hạn thắng mười trên phần đó, nhưng phần đó chỉ chiếm 30% thời gian. 70% còn lại chặn trần toàn bộ ở 1.43×.',
          );
        },
      },
      {
        get label() {
          return tr(
            'They come out the same.',
            'Hai phương án như nhau.',
          );
        },
        get why() {
          return tr(
            'A leaves 40% + 6% = 46% of the time. B leaves 70%.',
            'A để lại 40% + 6% = 46% thời gian. B để lại 70%.',
          );
        },
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            A: 1 / (0.4 + 0.6 / 10) = {fmt(speedup(0.6, 10))}×. B: 1 / 0.7 = {fmt(speedup(0.3, Infinity))}×. Kích thước của
            phần việc quan trọng hơn việc bạn làm nó nhanh đến đâu. Hãy profile trước, rồi chọn kernel.
          </p>
        }
      >
        <p>
          A: 1 / (0.4 + 0.6 / 10) = {fmt(speedup(0.6, 10))}×. B: 1 / 0.7 = {fmt(speedup(0.3, Infinity))}×. The size of
          the slice counts for more than how fast you make it. Profile first, then choose the kernel.
        </p>
      </Translated>
    ),
    source: "our numbers; the law is the book's, ch. 1 §1.3, p. 9",
    vi: { source: 'số liệu của chúng tôi; định luật là của sách, ch. 1 §1.3, p. 9' },
  },
  'ch01/1.3/q-saturated': {
    kind: 'choice',
    id: 'ch01/1.3/q-saturated',
    prompt: (
      <Translated
        vi={
          <>
            Một bản chuyển đổi thẳng thắn của một kernel đã giữ bộ nhớ ở mức bão hòa. Bạn viết lại phần số học để mỗi phần tử chỉ
            cần một nửa số lệnh, và không thay đổi gì về những gì nó đọc và ghi. Thời gian chạy của nó thay đổi thế nào?
          </>
        }
      >
        <>
          A straightforward port of a kernel already keeps memory saturated. You rewrite its arithmetic so that each
          element needs half as many instructions, and you change nothing about what it reads and writes. What happens to
          its run time?
        </>
      </Translated>
    ),
    options: [
      {
        get label() {
          return tr(
            'It roughly halves.',
            'Giảm khoảng một nửa.',
          );
        },
        get why() {
          return tr(
            'That would be true of a compute-bound kernel. The threads of this one were already waiting for data.',
            'Điều đó đúng với một kernel bị giới hạn bởi tính toán. Các luồng của kernel này vốn đã đang chờ dữ liệu.',
          );
        },
      },
      {
        get label() {
          return tr(
            'Almost nothing.',
            'Gần như không đổi.',
          );
        },
        correct: true,
      },
      {
        get label() {
          return tr(
            'It gets slower.',
            'Nó chậm đi.',
          );
        },
        get why() {
          return tr(
            'Nothing was added. The same bytes move at the same rate; the arithmetic finishes sooner and waits longer.',
            'Không có gì được thêm vào. Cùng số byte di chuyển với cùng tốc độ; phần số học xong sớm hơn và chờ lâu hơn.',
          );
        },
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Kernel bị giới hạn bởi bộ nhớ: tốc độ của nó do byte đến nhanh cỡ nào quyết định, và cùng số byte đó vẫn phải di
            chuyển. Điều giúp được là ít byte hơn trên mỗi phép toán, bằng cách tái sử dụng dữ liệu từ bộ nhớ trên chip (chương
            5, 6 và 15) hoặc làm mỗi giá trị nhỏ hơn (lượng tử hóa, module X2).
          </p>
        }
      >
        <p>
          The kernel is memory-bound: its speed is set by how fast bytes arrive, and the same bytes still have to move.
          What helps is fewer bytes per operation, either by reusing data from on-chip memory (chapters 5, 6 and 15) or
          by making each value smaller (quantisation, module X2).
        </p>
      </Translated>
    ),
    source: 'ch. 1 §1.3–1.4, pp. 10–12; chapter 5 §5.1 gives the method',
    vi: { source: 'ch. 1 §1.3–1.4, pp. 10–12; chương 5 §5.1 đưa ra phương pháp' },
  },
};
// The same record under the name the file-based convention gives it.
export const questionsS13 = questions13;

// ---------------------------------------------------------------- the page
export function S13() {
  const [p, setP] = useState(DEFAULT_P);
  const left30 = remaining(book30.p, book30.s);
  const left99 = remaining(book99.p, book99.s);
  const atDefaults = limits(DEFAULT_P, DEFAULTS.threads, DEFAULTS.rate, DEFAULTS.cap);
  return (
    <>
      <Translated
        vi={
          <p>
            Trước khi viết một kernel, hãy tính xem toàn bộ chương trình có thể được lợi bao nhiêu từ nó. Trang này cho bạn phép
            tính đó, rồi đến bốn khó khăn quyết định mức lợi đó khó đạt đến đâu. Chương 1 của sách không có mã và không có bài
            tập, nên chính các ví dụ đã tính sẵn của nó được dùng làm các dự đoán ở đây.
          </p>
        }
      >
        <p>
          Before you write a kernel, work out how much the whole program can gain from it. This page gives you the
          arithmetic for that, and then the four difficulties that decide how hard the gain is to get. Chapter 1 of the
          book has no code and no exercises, so its own worked cases serve as the predictions here.
        </p>
      </Translated>
      <Source section="1.3" pages="9–11" />

      <h2><Translated vi="Tăng tốc">Speed-up</Translated></h2>
      <Translated
        vi={
          <p>
            <b>Mức tăng tốc</b> (speed-up) của hệ thống A so với hệ thống B là thời gian B cần chia cho thời gian A cần. Thứ tự
            là quan trọng: một con số dưới 1 nghĩa là A chậm hơn. Ví dụ của sách: 200 s trên B và 10 s trên A là mức tăng tốc
            200 / 10 = 20× của A so với B.
          </p>
        }
      >
        <p>
          The <b>speed-up</b> of system A over system B is the time B takes divided by the time A takes. The order
          matters: a figure below 1 means A is the slower one. The book's example: 200 s on B and 10 s on A is a
          speed-up of 200 / 10 = 20× for A over B.
        </p>
      </Translated>
      <Source section="1.3" pages="9" />
      <Translated
        vi={
          <p>
            Con số này nói về mốc so sánh nhiều ngang với về bản chuyển đổi: CPU hợp với nhiệm vụ đến đâu, và phiên bản CPU tốt
            đến đâu. Vì thế sách yêu cầu bạn cho CPU một cơ hội công bằng. Hầu hết ứng dụng có những phần chạy tốt hơn trên đó,
            và mã GPU được dùng để bổ sung cho chúng.
          </p>
        }
      >
        <p>
          The figure says as much about the baseline as about the port: how well the CPU suits the task, and how good
          the CPU version was. So the book asks you to give the CPU a fair chance. Most applications have parts that run
          better there, and GPU code is meant to complement them.
        </p>
      </Translated>
      <Source section="1.3" pages="10" />

      <h2><Translated vi="Hai dự đoán">Two predictions</Translated></h2>
      <Translated
        vi={
          <p>
            Một kernel không bao giờ là toàn bộ chương trình. Hãy chốt một con số cho mỗi câu dưới đây trước khi đọc tiếp. Không
            cần công thức: hãy nghĩ theo thời gian chạy, từng phần một.
          </p>
        }
      >
        <p>
          A kernel is never the whole program. Commit to a number for each of these before you read on. No formula is
          needed: think in run time, part by part.
        </p>
      </Translated>
      <Source section="1.3" pages="9" />
      <Predict
        id="ch01/1.3/p-thirty"
        prompt={
          <Translated
            vi={
              <>
                Một chương trình dành 30% thời gian chạy cho một phần bạn có thể song song hóa. Bạn làm phần đó nhanh hơn 100 lần và
                để nguyên phần còn lại. Mức tăng tốc tổng thể là bao nhiêu?
              </>
            }
          >
            <>
              A program spends 30% of its run time in a part you can parallelise. You make that part 100 times faster and
              leave the rest alone. What is the overall speed-up?
            </>
          </Translated>
        }
        unit="×"
      >
        {(v) => (
          <Translated
            vi={
              <p>
                Phần được tăng tốc co từ 30% thời gian xuống {share(left30.accelerated)}. Phần {share(left30.untouched)} còn lại không
                bị đụng tới, nên lần chạy mất {share(left30.total)} so với trước: 1 / {fmt(left30.total)} ={' '}
                <b>{times(speedup(book30.p, book30.s))}</b>.{' '}
                {v >= 1.395 && v < 1.4275
                  ? 'Đúng như bạn đã dự đoán.'
                  : v >= 1.4275 && v <= 1.44
                    ? 'Bạn đã bỏ quên 0.3% còn lại của phần được tăng tốc. Đó là trường hợp tiếp theo.'
                    : v >= 1.25 && v <= 1.35
                      ? 'Bạn tính đúng phần tiết kiệm: khoảng 30% thời gian. Nhưng một lần chạy mất 70.3% thời gian thì nhanh gấp 1 / 0.703 lần, không phải 1.3 lần.'
                      : v >= 20
                        ? 'Nếu bạn đoán gần 30 hay 100: 70% còn lại không hề nhúc nhích, và giờ nó gần như là toàn bộ lần chạy.'
                        : 'Hãy tính theo thời gian, không theo tốc độ: cộng phần còn lại của mỗi phần, rồi chia.'}{' '}
                Làm cùng phần đó nhanh vô hạn thì lần chạy vẫn mất 70% thời gian: không quá {times(ceiling(book30.p))}. Cả hai con số
                đều là của sách.
              </p>
            }
          >
            <p>
              The accelerated part shrinks from 30% of the time to {share(left30.accelerated)}. The other{' '}
              {share(left30.untouched)} is untouched, so the run takes {share(left30.total)} of what it did: 1 /{' '}
              {fmt(left30.total)} = <b>{times(speedup(book30.p, book30.s))}</b>.{' '}
              {v >= 1.395 && v < 1.4275
                ? 'That is what you had.'
                : v >= 1.4275 && v <= 1.44
                  ? 'You dropped the 0.3% that is left of the accelerated part. That is the next case.'
                  : v >= 1.25 && v <= 1.35
                    ? 'You have the saving right: about 30% of the time. But a run that takes 70.3% as long is 1 / 0.703 times as fast, not 1.3 times.'
                    : v >= 20
                      ? 'If you were near 30 or 100: the other 70% did not move, and it is now nearly all of the run.'
                      : 'Work in time, not in speed: add up what is left of each part, then divide.'}{' '}
              Make the same part infinitely fast and the run still takes 70% of the time: no more than{' '}
              {times(ceiling(book30.p))}. Both figures are the book's.
            </p>
          </Translated>
        )}
      </Predict>
      <Predict
        id="ch01/1.3/p-ninetynine"
        prompt={
          <Translated vi={<>Giờ phần đó chiếm 99% thời gian chạy, và một lần nữa bạn làm nó nhanh hơn 100 lần. Mức tăng tốc tổng thể là bao nhiêu?</>}>
            Now the part is 99% of the run time, and again you make it 100 times faster. What is the overall speed-up?
          </Translated>
        }
        unit="×"
      >
        {(v) => (
          <Translated
            vi={
              <p>
                99% co lại còn {share(left99.accelerated)}. 1% không đụng tới vẫn là 1%. Lần chạy mất {share(left99.total)} so với
                trước: 1 / {fmt(left99.total)} = <b>{times(speedup(book99.p, book99.s))}</b>; sách nói khoảng 50.{' '}
                {Math.abs(v - speedup(book99.p, book99.s)) <= 1.5
                  ? 'Đúng như bạn đã dự đoán.'
                  : v >= 90
                    ? 'Gần như mọi thứ đều được tăng tốc, nên 99 hay 100 là phỏng đoán tự nhiên. Nó quên rằng 1% không đụng tới giờ lớn ngang toàn bộ phần việc được tăng tốc.'
                    : 'Cùng phương pháp như trước: 1% không đụng tới, cộng 0.99% còn lại của phần được tăng tốc.'}{' '}
                Một phần trăm của chương trình đã lấy mất một nửa của con số một trăm.
              </p>
            }
          >
            <p>
              The 99% shrinks to {share(left99.accelerated)}. The untouched 1% is still 1%. The run takes{' '}
              {share(left99.total)} of what it did: 1 / {fmt(left99.total)} = <b>{times(speedup(book99.p, book99.s))}</b>
              ; the book says about 50.{' '}
              {Math.abs(v - speedup(book99.p, book99.s)) <= 1.5
                ? 'That is what you had.'
                : v >= 90
                  ? 'Nearly everything was accelerated, so 99 or 100 is the natural guess. It forgets that the untouched 1% is now as large as all the accelerated work.'
                  : 'Same method as before: 1% untouched, plus 0.99% left of the accelerated part.'}{' '}
              One percent of the program has taken half of the hundred.
            </p>
          </Translated>
        )}
      </Predict>

      <h2><Translated vi="Định luật Amdahl">Amdahl's law</Translated></h2>
      <Translated
        vi={
          <p>
            Điều bạn vừa tính ra là định luật Amdahl: mức tăng tốc của toàn bộ chương trình bị giới hạn bởi phần thời gian mà
            bạn để nguyên. Sách phát biểu nó bằng lời và minh họa qua các trường hợp trên. Viết ra, với p là phần thời gian chạy
            ban đầu nằm trong phần bạn tăng tốc và s là mức tăng tốc của phần đó, con số tổng thể là 1 / ((1 − p) + p / s).
          </p>
        }
      >
        <p>
          What you just worked out is Amdahl's law: the speed-up of a whole program is limited by the share of its time
          that you leave alone. The book states it in words and shows it through those cases. Written out, with p the
          fraction of the original run time in the part you speed up and s the speed-up of that part, the overall figure
          is 1 / ((1 − p) + p / s).
        </p>
      </Translated>
      <Beyond title={tr('Our formula', 'Công thức của chúng tôi')}>
        <Translated
          vi={
            <p>
              Sách không in công thức nào, chỉ có các con số (p. 9). Công thức này là của chúng tôi, và ba trường hợp của sách khớp
              với nó: 1.42, 1.43 và khoảng 50.
            </p>
          }
        >
          <p>
            The book prints no formula, only numbers (p. 9). This one is ours, and the book's three cases match it: 1.42,
            1.43 and about 50.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Trình khám phá bắt đầu ở trường hợp bạn vừa dự đoán. Dùng các cấu hình sẵn để kiểm tra hai trường hợp kia, rồi làm
            nhiệm vụ ở phần đầu của nó.
          </p>
        }
      >
        <p>
          The explorer starts on the case you just predicted. Use the presets to check the other two, then work on the
          task in its header.
        </p>
      </Translated>
      <AmdahlBench p={p} setP={setP} />
      <Translated
        vi={
          <p>
            Ba điều cần để ý. Các cấu hình sẵn cho {times(speedup(0.3, 100))}, {times(speedup(0.3, Infinity))} và{' '}
            {times(speedup(0.99, 100))}, đúng các con số của sách. Trần chỉ phụ thuộc vào p: không gì bạn làm với s di chuyển
            được đường đứt. Và khi gần trần, công sức thôi sinh lời. Ở p = 99%, đưa phần đó từ 100× lên 1,000× chỉ đưa toàn bộ từ
            khoảng {fmt(speedup(0.99, 100), 2)}× lên khoảng {fmt(speedup(0.99, 1000), 2)}×, vì đến lúc đó phần lớn thời gian còn
            lại thuộc về phần bạn không đụng tới.
          </p>
        }
      >
        <p>
          Three things to notice. The presets give {times(speedup(0.3, 100))}, {times(speedup(0.3, Infinity))} and{' '}
          {times(speedup(0.99, 100))}, the book's figures. The ceiling depends on p alone: nothing you do to s moves the
          dashed line. And close to the ceiling, effort stops paying. At p = 99%, taking the part from 100× to 1,000×
          moves the whole only from about {fmt(speedup(0.99, 100), 2)}× to about {fmt(speedup(0.99, 1000), 2)}×, because
          by then most of the time that is left belongs to the part you did not touch.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Chỉ có hai phần, và chúng không chồng lấn về thời gian; chương 22 (<a href={href('ch22', '22.3')}>§22.3</a>) nới
              lỏng điều đó. Phần không đụng tới được giả định giữ nguyên kích thước, dù việc chuyển công việc sang GPU thêm vào đó
              các lần sao chép và khởi chạy kernel. Kích thước bài toán cố định.
            </p>
          }
        >
          <p>
            Two parts only, and they do not overlap in time; chapter 22 (
            <a href={href('ch22', '22.3')}>§22.3</a>) relaxes that. The untouched part is
            assumed to stay the same size, although moving work to a GPU adds copies and kernel launches to it. The
            problem size is fixed.
          </p>
        </Translated>
      </Simplification>
      <Translated
        vi={
          <p>
            Đây là bối cảnh của một nhận xét sách đưa ra ngay sau các ví dụ. Sách nói đã có báo cáo về mức tăng tốc trên 100×,
            nhưng thường chỉ sau khi tối ưu và thay đổi thuật toán để hơn 99.9% công việc nằm trong phần song song. Vì sao là
            99.9% chứ không phải 99% của nhiệm vụ? Nhiệm vụ là nửa đầu của lý do: nó giả định một phần có tốc độ không giới hạn.
            Câu hỏi tiếp theo là nửa sau.
          </p>
        }
      >
        <p>
          This is the background to a remark the book makes right after its examples. Speed-ups above 100× have been
          reported, it says, but typically only after optimisation and algorithm changes that put more than 99.9% of the
          work in the parallel part. Why 99.9% and not the 99% of the task? The task was the first half of the reason:
          it assumed a part of unbounded speed. The next question is the second half.
        </p>
      </Translated>
      <Source section="1.3" pages="9–10" />
      <Ask q={questions13['ch01/1.3/q-fivehundred']} />
      <Ask q={questions13['ch01/1.3/q-kernelneeded']} />

      <h2><Translated vi="Cùng phép tính cho một bước decode">The same sum for a decode step</Translated></h2>
      <Translated
        vi={
          <p>
            Giờ áp dụng nó vào thứ bạn xây dựng. Sinh một token là một chuỗi kernel (các phép nhân ma trận, attention, chuẩn hóa)
            cộng với công việc trên host: tokeniser, chuỗi lấy mẫu, và chép logits về để bộ lấy mẫu đọc được. Nếu bạn chỉ tăng
            tốc các kernel, công việc trên host là phần 1 − p của định luật. Trong llama.cpp, chuỗi lấy mẫu mặc định chạy trên
            host trên một mảng ứng viên, nên theo mặc định nó thuộc về phần không được tăng tốc.
          </p>
        }
      >
        <p>
          Now apply it to what you build. Generating one token is a chain of kernels (the matrix products, attention,
          normalisation) plus work on the host: the tokeniser, the sampler chain, and copying the logits back so that
          the sampler can read them. If you speed up only the kernels, the host work is the 1 − p of the law. In
          llama.cpp the default sampler chain runs on the host over an array of candidates, so by default it belongs to
          the part that does not speed up.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Cách ánh xạ sang một bước decode là của chúng tôi. Sách áp dụng cùng định luật cho một ứng dụng động lực học phân tử
              ở <a href={href('ch22', '22.3')}>§22.3</a> (pp. 537–538), nơi các module để lại trên host chặn trần toàn bộ.
            </p>
          }
        >
          <p>
            The mapping to a decode step is ours. The book works the same law on a molecular-dynamics application in{' '}
            <a href={href('ch22', '22.3')}>§22.3</a> (pp. 537–538), where the modules left on the host cap the whole.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Đó là mặc định, không phải định luật. llama.cpp cũng có một giao diện thử nghiệm chạy bộ lấy mẫu bên trong backend,
            cạnh các kernel, và <a href={href('x1')}>module X1</a> của khóa học này cho bạn viết loại kernel đó: chọn greedy,
            top-k và top-p trên GPU. Hai câu hỏi dưới đây là lý do việc đó đáng làm.
          </p>
        }
      >
        <p>
          That is a default, not a law. llama.cpp also has an experimental interface that runs samplers inside the
          backend, beside the kernels, and <a href={href('x1')}>module X1</a> of this course has you write that kind of
          kernel: greedy, top-k and top-p selection on the GPU. The two questions below are the reason it is worth
          doing.
        </p>
      </Translated>
      <Today checked={tr("2026-11-20 against llama.cpp's llama.h (notes/today/ch01-facts.md, D2)", '2026-11-20, đối chiếu với llama.h của llama.cpp (notes/today/ch01-facts.md, D2)')}>
        <Translated
          vi={
            <p>
              Header đánh dấu giao diện lấy mẫu trong backend là thử nghiệm. Chúng tôi không lần theo xem bộ lấy mẫu nào hiện thực
              nó, hay có front end nào bật nó lên không.
            </p>
          }
        >
          <p>
            The header marks the backend sampling interface as experimental. We did not trace which samplers implement
            it, or whether any front end turns it on.
          </p>
        </Translated>
      </Today>
      <Ask q={questions13['ch01/1.3/q-hostshare']} />
      <Ask q={questions13['ch01/1.3/q-bigslice']} />

      <h2><Translated vi="10× đầu tiên đến từ đâu, và dừng ở đâu">Where the first 10× comes from, and where it stops</Translated></h2>
      <Translated
        vi={
          <p>
            Sách đưa ra ba quy tắc kinh nghiệm, đều từ kinh nghiệm của các tác giả. Hai quy tắc đầu: một hiện thực GPU tốt cho
            một ứng dụng phù hợp có thể chạy nhanh hơn 100× so với mã tuần tự trên một lõi CPU, và khi ứng dụng có song song dữ
            liệu, 10× thường đạt được chỉ với vài giờ làm việc.
          </p>
        }
      >
        <p>
          The book gives three rules of thumb, all from its authors' experience. The first two: a good GPU implementation
          of a suitable application can run more than 100× faster than sequential code on one CPU core, and where the
          application has data parallelism, 10× is often reached with a few hours of work.
        </p>
      </Translated>
      <Source section="1.2" pages="7" />
      <Translated
        vi={
          <p>
            Quy tắc thứ ba là quy tắc mà khóa học này xoay quanh. Một bản chuyển đổi thẳng thắn thường làm bão hòa băng thông
            DRAM và dừng ở khoảng 10×. Đi xa hơn nghĩa là thay đổi mã để dữ liệu được tái sử dụng từ các bộ nhớ trên chip của GPU
            và số truy cập tới DRAM ít đi nhiều, rồi xoay xở với việc các bộ nhớ đó nhỏ đến mức nào. Trong khóa học này đó là các
            chương 5, 6 và 15. Chúng tôi không tìm thấy nguồn nào từ nhà sản xuất cho cả ba con số; hãy coi mỗi con số là của
            sách.
          </p>
        }
      >
        <p>
          The third is the one this course is built around. A straightforward port often saturates DRAM bandwidth and
          stops at about 10×. Going further means changing the code so that data is reused from the GPU's on-chip
          memories and far fewer accesses go to DRAM, and then working around how small those memories are. In this
          course that is chapters 5, 6 and 15. We found no vendor source for any of the three figures; take each as the
          book's.
        </p>
      </Translated>
      <Source section="1.3" pages="10" />
      <Translated
        vi={
          <p>
            Ở phần trước của chương, sách cũng nói rằng chip đồ họa có băng thông bộ nhớ khoảng 10× so với CPU cùng thời (p. 5).
            Sách nêu cả hai con số mười và không nối chúng với nhau. Bạn không cần tin suông con số nào.{' '}
            <a href={href('ch05', '5.1')}>Chương 5</a> biến nhận xét về sự bão hòa thành một phép tính: roofline của nó cho ra
            trần của một kernel từ số phép toán trên mỗi byte và băng thông của thiết bị, và lab 5.1 đo băng thông của chính
            thiết bị của bạn. Chương 5 cũng cho thấy một trường hợp mà mô hình và phép đo chênh nhau rất xa, và vì sao (
            <a href={href('ch05', '5.1')}>"Khi một phép đo nằm trên mái"</a>). Vì thế khóa học cho bạn dự đoán rồi đo, và quy tắc
            kinh nghiệm này không phải ngoại lệ.
          </p>
        }
      >
        <p>
          Earlier in the chapter the book also says that graphics chips have run at roughly 10× the memory bandwidth of
          the CPUs of their day (p. 5). It states both tens and does not link them. You do not have to take either
          on trust. <a href={href('ch05', '5.1')}>Chapter 5</a> turns the remark about saturation into a calculation: its
          roofline gives a kernel's ceiling from its operations per byte and the device's bandwidth, and lab 5.1
          measures the bandwidth of your own device. Chapter 5 also shows a case where the model and a measurement
          disagree by a wide margin, and why (<a href={href('ch05', '5.1')}>"When a measurement lands above the
          roof"</a>). So the course has you predict and then measure, and this rule of thumb is no exception.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 against the CUDA programming guide and vendor specification pages (notes/today/ch01-facts.md, A8)',
          '2026-11-20, đối chiếu với hướng dẫn lập trình CUDA và các trang thông số của nhà sản xuất (notes/today/ch01-facts.md, A8)',
        )}
      >
        <Translated
          vi={
            <p>
              Hướng dẫn của NVIDIA xác nhận chiều hướng: băng thông bộ nhớ cao hơn một CPU cùng tầm giá và công suất. Theo các con
              số của nhà sản xuất mà chúng tôi đã kiểm tra, tỷ lệ của một GPU trung tâm dữ liệu so với một socket CPU máy chủ là
              dưới mười. Trên Apple silicon, một bộ nhớ phục vụ cả hai bộ xử lý, nên không có tỷ lệ nào.
            </p>
          }
        >
          <p>
            NVIDIA's guide confirms the direction: more memory bandwidth than a CPU in a similar price and power
            envelope. On the vendor figures we checked, the ratio for one data-centre GPU against one server CPU socket
            is below ten. On Apple silicon one memory serves both processors, so there is no ratio.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Trình khám phá dưới đây đặt ba giới hạn vào một chỗ: bao nhiêu luồng chạy, bộ nhớ cấp dữ liệu cho chúng nhanh đến
            đâu, và bạn để nguyên bao nhiêu phần của chương trình. Sách nêu từng giới hạn bằng lời; mô hình kết hợp chúng là của
            chúng tôi. Nó dùng chung phần song song với trình khám phá ở trên.
          </p>
        }
      >
        <p>
          The explorer below puts three limits in one place: how many threads run, how fast memory can feed them, and how
          much of the program you left alone. The book gives each of them in words; the model that combines them is
          ours. It shares its parallel fraction with the explorer above.
        </p>
      </Translated>
      <CapsBench p={p} setP={setP} />
      <Translated
        vi={
          <p>
            Ba điều cần để ý. Qua điểm bộ nhớ bão hòa, thêm luồng không mua được gì: với trần bộ nhớ {DEFAULTS.cap} và 99% thời
            gian nằm trong kernel, kết quả là {times(atDefaults.overall)} dù chạy bao nhiêu luồng. Nâng trần là thay đổi duy nhất
            dịch chuyển nó đáng kể: lên {times(limits(DEFAULT_P, DEFAULTS.threads, DEFAULTS.rate, 100).overall)} ở trần 100.
            Trong một kernel thật, điều đó nghĩa là ít byte hơn trên mỗi phép toán, qua tái sử dụng (chương 5, 6 và 15) hoặc qua
            giá trị nhỏ hơn (lượng tử hóa). Sau đó kernel không còn là thứ kìm bạn lại, và chỉ một phần song song lớn hơn mới
            giúp được. Với một bước decode, điều đó nghĩa là chuyển công việc trên host như lấy mẫu sang GPU.
          </p>
        }
      >
        <p>
          Three things to notice. Past the point where memory saturates, more threads buy nothing: with a memory cap of{' '}
          {DEFAULTS.cap} and 99% of the time in the kernel, the result is {times(atDefaults.overall)} however many
          threads run. Raising the cap is the only change that moves it much: to{' '}
          {times(limits(DEFAULT_P, DEFAULTS.threads, DEFAULTS.rate, 100).overall)} at a cap of 100. In a real kernel that means fewer bytes per operation, through reuse (chapters 5, 6 and 15) or
          through smaller values (quantisation). After that the kernel is no longer what holds you back, and only a
          larger parallel fraction helps. For a decode step, that means moving host work such as sampling onto the GPU.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Ba giới hạn cứng với các góc sắc, trong khi đường cong thật thì uốn cong. Trần bộ nhớ ở đây là một con số; chương 5
              tính nó từ số phép toán trên mỗi byte của kernel và băng thông của thiết bị, và bộ nhớ đệm làm nó thay đổi. Tốc độ
              của một luồng là hằng số, và các luồng mở rộng hoàn hảo cho đến khi bộ nhớ bão hòa. Sách nêu con số 10× mà không suy
              ra nó.
            </p>
          }
        >
          <p>
            Three hard limits with sharp corners, where real curves bend. The memory cap is one number here; chapter 5
            computes it from a kernel's operations per byte and a device's bandwidth, and caches change it. The rate of
            a thread is a constant, and threads scale perfectly until memory saturates. The book states the 10× figure
            and does not derive it.
          </p>
        </Translated>
      </Simplification>
      <Ask q={questions13['ch01/1.3/q-saturated']} />
      <Translated
        vi={
          <p>
            Sách hình dung tất cả điều này như một quả đào. Hạt là phần tuần tự, có thể chiếm phần lớn mã nhưng thường chỉ chiếm
            phần nhỏ thời gian chạy; thịt quả là phần song song dữ liệu, nơi thời gian nằm. Các thanh thời gian của trình khám
            phá đầu tiên là hình ảnh của chúng tôi cho cùng điều đó.
          </p>
        }
      >
        <p>
          The book pictures all of this as a peach. The pit is the sequential part, which can be a large share of the
          code and is usually a small share of the run time; the flesh is the data-parallel part, where the time is. The
          time bars of the first explorer are our picture of the same thing.
        </p>
      </Translated>
      <Source section="1.3" pages="10–11" figs="Fig. 1.2" />

      <h2><Translated vi="Bốn khó khăn mà sách liên tục quay lại">Four difficulties the book keeps returning to</Translated></h2>
      <Translated
        vi={
          <p>
            Sách nhận xét rằng lập trình song song thì dễ, miễn là bạn không quan tâm đến hiệu năng. Sách gọi tên bốn điều làm
            phiên bản nhanh trở nên khó. Phần còn lại của sách, và của khóa học này, phần lớn là một tập câu trả lời cho chúng.
          </p>
        }
      >
        <p>
          Parallel programming is easy, the book remarks, as long as you do not care about performance. It names four
          things that make the fast version hard. The rest of the book, and of this course, is largely a set of answers
          to them.
        </p>
      </Translated>
      <Source section="1.4" pages="11–12" />
      <Translated
        vi={
          <p>
            <b>Độ phức tạp thuật toán.</b> Một thuật toán song song có thể làm tổng khối lượng việc nhiều hơn thuật toán tuần tự
            mà nó thay thế, đôi khi đủ để thua trên đầu vào lớn. Một bài toán vốn là hệ thức truy hồi, nơi mỗi bước cần kết quả
            của bước trước, phải được phát biểu lại trước khi luồng thứ hai có việc để làm, và dạng mới có thể thêm việc thừa.
            Công cụ chuẩn là tổng tiền tố (prefix sum), và thước đo phần việc thêm là <b>hiệu quả công việc</b> (work
            efficiency). Cả hai đến ở chương 11.
          </p>
        }
      >
        <p>
          <b>Algorithmic complexity.</b> A parallel algorithm can do more total work than the sequential one it replaces,
          sometimes enough to lose on large inputs. A problem that is naturally a recurrence, where each step needs the
          result of the one before, has to be reformulated before a second thread has anything to do, and the new form
          may add redundant work. The standard tool is the prefix sum, and the measure of the extra work is{' '}
          <b>work efficiency</b>. Both arrive in chapter 11.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <b>Thực thi bị giới hạn bởi bộ nhớ.</b> Một chương trình có tốc độ do dữ liệu đến nhanh cỡ nào quyết định thì{' '}
            <b>bị giới hạn bởi bộ nhớ</b>; một chương trình có tốc độ do phần số học quyết định thì{' '}
            <b>bị giới hạn bởi tính toán</b>. Với loại thứ nhất, phép tính nhanh hơn không mua được gì. Chương 5 và 6 nói về việc
            rút được nhiều phép toán hơn từ mỗi byte.
          </p>
        }
      >
        <p>
          <b>Memory-bound execution.</b> A program whose speed is set by how fast its data arrives is <b>memory-bound</b>
          ; one whose speed is set by its arithmetic is <b>compute-bound</b>. For the first kind, faster arithmetic buys
          nothing. Chapters 5 and 6 are about getting more operations out of each byte.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <b>Nhạy cảm với dữ liệu đầu vào.</b> Chương trình song song phản ứng với hình dạng đầu vào mạnh hơn chương trình tuần
            tự. Kích thước thất thường và phân bố không đều giao cho các luồng lượng việc không đều. Cách chữa là làm dữ liệu đều
            đặn hơn hoặc điều chỉnh cách gán dữ liệu cho luồng, và sách đặt chúng trong các chương về mẫu hình và ứng dụng.
          </p>
        }
      >
        <p>
          <b>Sensitivity to input data.</b> Parallel programs react more than sequential ones to the shape of their
          input. Erratic sizes and uneven distributions hand the threads uneven amounts of work. The remedies are to
          make the data more regular or to fit the assignment of data to threads, and the book places them in its
          pattern and application chapters.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <b>Chi phí đồng bộ hóa.</b> Công việc mà các luồng gần như không cần gì từ nhau được gọi là{' '}
            <b>embarrassingly parallel</b> (song song hiển nhiên). Mọi thứ khác cần các thao tác đồng bộ hóa như rào chắn
            (barrier) và phép toán nguyên tử (atomic), và mỗi thao tác khiến các luồng phải chờ thay vì làm việc. Cắt giảm sự chờ
            đợi đó là một chủ đề xuyên suốt cả cuốn sách.
          </p>
        }
      >
        <p>
          <b>Synchronisation overhead.</b> Work whose threads need almost nothing from each other is called{' '}
          <b>embarrassingly parallel</b>. Everything else needs synchronisation operations such as barriers and atomic
          operations, and each of them makes threads wait instead of work. Cutting that waiting is a theme of the whole
          book.
        </p>
      </Translated>
      <table className="data wide" data-challenges style={{ maxWidth: '62rem' }}>
        <thead>
          <tr>
            <th style={{ minWidth: '8.5rem' }}><Translated vi="Khó khăn">Challenge</Translated></th>
            <th style={{ minWidth: '9.5rem' }}><Translated vi="Thuật ngữ của sách">The book's term</Translated></th>
            <th style={{ minWidth: '12rem' }}><Translated vi="Nơi sách gặp nó">Where the book meets it</Translated></th>
            <th style={{ minWidth: '11rem' }}><Translated vi="Trong các module riêng của khóa học này">In this course's own modules</Translated></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="Phiên bản song song làm nhiều việc hơn">The parallel version does more work</Translated></td>
            <td><Translated vi="độ phức tạp thuật toán; hiệu quả công việc">algorithmic complexity; work efficiency</Translated></td>
            <td><Translated vi="ch. 11 (được nêu tên ở p. 11)">ch. 11 (named on p. 11)</Translated></td>
            <td><Translated vi="X1 lấy mẫu: top-p cần một tổng tiền tố">X1 sampling: top-p needs a prefix sum</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Bộ nhớ quyết định tốc độ">Memory sets the speed</Translated></td>
            <td><Translated vi="bị giới hạn bởi bộ nhớ, so với bị giới hạn bởi tính toán">memory-bound, against compute-bound</Translated></td>
            <td><Translated vi="chs. 5 và 6 (được nêu tên ở pp. 11–12); ch. 15 là phần chúng tôi thêm">chs. 5 and 6 (named on pp. 11–12); ch. 15 is our addition</Translated></td>
            <td><Translated vi="X2 kernel lượng tử hóa: ít byte hơn trên mỗi phép toán">X2 quantised kernels: fewer bytes per operation</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Các luồng nhận lượng việc không đều">Threads get uneven work</Translated></td>
            <td><Translated vi="nhạy cảm với dữ liệu đầu vào">sensitivity to input data</Translated></td>
            <td><Translated vi="các chương về mẫu hình và ứng dụng của sách (p. 12); chs. 13, 17 và 18 là cách đọc của chúng tôi">its pattern and application chapters (p. 12); chs. 13, 17 and 18 are our reading</Translated></td>
            <td><Translated vi="không module nào xoay quanh nó">neither module is built around it</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Các luồng chờ nhau">Threads wait for each other</Translated></td>
            <td><Translated vi="chi phí đồng bộ hóa; embarrassingly parallel">synchronisation overhead; embarrassingly parallel</Translated></td>
            <td><Translated vi="một chủ đề xuyên suốt (p. 12); chs. 5, 9, 10 và 11 là cách đọc của chúng tôi">a theme throughout (p. 12); chs. 5, 9, 10 and 11 are our reading</Translated></td>
            <td><Translated vi="X1 lấy mẫu: các bộ đếm mà nhiều luồng cùng cập nhật">X1 sampling: counters that many threads update</Translated></td>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Ở chỗ bảng ghi số trang, số chương là của chính sách. Phần còn lại của việc ánh xạ là của chúng tôi, dựa trên ghi chú
            của chúng tôi về các chương đó, và cột cuối cũng vậy: sách không có chương nào về lấy mẫu hay lượng tử hóa.
          </p>
        }
      >
        <p>
          Where the table gives a page, the chapter number is the book's own. The rest of the mapping is ours, made from
          our notes on those chapters, and so is the last column: the book has no sampling or quantisation chapter.
        </p>
      </Translated>

      <h3><Translated vi="Đưa các cái tên vào sử dụng">Put the names to work</Translated></h3>
      <Translated
        vi={
          <p>
            Mỗi tình huống dưới đây là một tình huống bạn sẽ gặp khi viết kernel suy luận. Hãy quyết định nó chủ yếu là một ví dụ
            của khó khăn nào trong bốn. Vài tình huống dễ ở hầu hết các mặt, chính điều đó làm cho mặt còn lại đáng được gọi tên.
          </p>
        }
      >
        <p>
          Each situation below is one you will meet when you write inference kernels. Decide which of the four it is
          mainly an instance of. Several are easy on most counts, which is what makes the remaining one worth naming.
        </p>
      </Translated>
      <WhichChallenge />
      <Translated
        vi={
          <p>
            Tiếp theo, <a href={href('ch01', '1.7')}>sách, ấn bản này, và lộ trình của bạn qua nó</a>: những chương nào mang bốn
            khó khăn này đến phép nhân ma trận, attention và lấy mẫu.
          </p>
        }
      >
        <p>
          Next, <a href={href('ch01', '1.7')}>the book, this edition, and your path through it</a>: which chapters carry
          these four through to matrix multiplication, attention and sampling.
        </p>
      </Translated>
    </>
  );
}
