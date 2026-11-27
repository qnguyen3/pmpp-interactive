import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useReports, type Reports } from '../../lib/reports';
import { useNarrow } from '../../lib/narrow';
import { href } from '../../lib/router';
import { Translated, localized, tr } from '../../lib/i18n';
import {
  ALGOS,
  ALGO_LABEL,
  N_MAX_LOG,
  N_MIN_LOG,
  brentKungUpTo,
  compare,
  curve,
  hybridLag,
  koggeStoneFrom,
  networkWinner,
  overallWinner,
  type AlgoId,
  type CurvePoint,
} from '../../sim/algoselect';
import {
  BINS,
  BOOK_SHARE,
  DISTRIBUTIONS,
  DIST_LABEL,
  GRID,
  POINTS_PER_BIN,
  bestCapacity,
  binWith,
  capacityForShare,
  countsFor,
  histogram,
  work,
  type Distribution,
} from '../../sim/binning';
import { CHARGE_LABEL, type Charge } from '../../sim/prefixnet';
import { widthChoices } from '../../sim/warps';

/** A caption above a group of buttons (a <label> would forward clicks to the first button). */
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const WRAP: CSSProperties = { flexWrap: 'wrap' };
const int = (v: number) => Math.round(v).toLocaleString('en-US');
const dec = (v: number, d = 2) => Number(v.toFixed(d)).toLocaleString('en-US');
const pct = (v: number) => (v === 0 ? '0' : v < 0.001 ? '<0.1' : dec(100 * v, 1));


// ---------------------------------------------------------------- explorer 1: which scan
const CHARGES: Charge[] = ['adds', 'warps'];
const CHARGE_NOTE: Record<Charge, string> = {
  get adds() {
    return tr(
      'Only the threads that add are charged: a step with a additions costs ⌈a / P⌉ time units.',
      'Chỉ những luồng thực hiện phép cộng mới bị tính: một bước có a phép cộng tốn ⌈a / P⌉ đơn vị thời gian.',
    );
  },
  get warps() {
    return tr(
      'A warp with at least one adding lane is charged all its lanes, the reason chapter 11 gives for real resource use above the formulas.',
      'Một warp có ít nhất một lane đang cộng bị tính cho mọi lane của nó, đây là lý do chương 11 đưa ra cho việc tài nguyên dùng thực tế cao hơn công thức.',
    );
  },
  threads: '',
};
const DASH: Record<AlgoId, string | undefined> = { 'kogge-stone': undefined, 'brent-kung': '7 4', coarsened: '2 3' };
const STROKE: Record<AlgoId, string> = { 'kogge-stone': 'var(--ink)', 'brent-kung': 'var(--ink)', coarsened: 'var(--ink-2)' };

function Bar({ value, max }: { value: number; max: number }) {
  const w = max > 0 ? Math.max(1, (value / max) * 100) : 0;
  return (
    <span aria-hidden="true" style={{ display: 'block', height: 6, marginTop: 3, background: 'var(--rule-soft)', borderRadius: 2 }}>
      <span style={{ display: 'block', height: 6, width: `${w}%`, background: 'var(--ink-2)', borderRadius: 2 }} />
    </span>
  );
}

function Legend() {
  return (
    <div className="ui" style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem 1.1rem', fontSize: '0.85em', margin: '0 0 0.3rem' }}>
      {ALGOS.map((a) => (
        <span key={a} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
          <svg width="28" height="8" aria-hidden="true">
            <line x1="1" x2="27" y1="4" y2="4" stroke={STROKE[a]} strokeWidth={a === 'coarsened' ? 2.4 : 1.8} strokeDasharray={DASH[a]} />
          </svg>
          {localized(ALGO_LABEL[a], ({ 'kogge-stone': 'Kogge-Stone', 'brent-kung': 'Brent-Kung', coarsened: 'Ba pha có gộp việc' } as Record<AlgoId, string>)[a])}
        </span>
      ))}
    </div>
  );
}

function TimePlot({ pts, n, p, bkUpTo, ksFrom }: { pts: CurvePoint[]; n: number; p: number; bkUpTo: number | null; ksFrom: number | null }) {
  const narrow = useNarrow(600); // a narrower drawing on a phone keeps the labels legible
  const W = narrow ? 380 : 640;
  const H = narrow ? 260 : 290;
  const pad = { l: 50, r: 12, t: 10, b: 42 };
  const ys = pts.flatMap((q) => ALGOS.map((a) => q[a]));
  const yMin = Math.max(1, Math.min(...ys) / 1.4);
  const yMax = Math.max(...ys) * 1.4;
  const lx = (v: number) => Math.log2(v);
  const X = (v: number) => pad.l + (lx(v) / Math.max(1, lx(n))) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * (H - pad.t - pad.b);
  const every = narrow ? (pts.length > 9 ? 4 : 2) : pts.length > 9 ? 2 : 1;
  const xt = pts.filter((_, i) => i % every === 0).map((q) => q.p);
  const yt: number[] = [];
  for (let e = Math.ceil(Math.log10(yMin)); e <= Math.floor(Math.log10(yMax)); e++) yt.push(10 ** e);
  return (
    <div>
    <Legend />
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Time units against execution units', 'Đơn vị thời gian theo số đơn vị thực thi')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {bkUpTo ? (
        <rect x={pad.l} y={pad.t} width={Math.max(0, X(bkUpTo) - pad.l)} height={H - pad.t - pad.b} fill="var(--rule)" opacity="0.35" data-region="bk" />
      ) : null}
      {xt.map((v) => (
        <g key={'x' + v}>
          <line x1={X(v)} x2={X(v)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(v)} y={H - pad.b + 15} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
            {v >= 1024 ? v / 1024 + 'k' : v}
          </text>
        </g>
      ))}
      {yt.map((v) => (
        <g key={'y' + v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 6} y={Y(v) + 4} fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
            {v >= 1000 ? v / 1000 + 'k' : v}
          </text>
        </g>
      ))}
      <text x={(W + pad.l) / 2} y={H - 8} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi="số đơn vị thực thi P">execution units P</Translated>
      </text>
      <text x={14} y={(H - pad.b + pad.t) / 2} fontSize="11.5" textAnchor="middle" transform={`rotate(-90 14 ${(H - pad.b + pad.t) / 2})`} style={{ fill: 'var(--ink-2)' }}>
        <Translated vi="đơn vị thời gian">time units</Translated>
      </text>
      {ksFrom ? (
        <g data-mark="ks-from">
          <line x1={X(ksFrom)} x2={X(ksFrom)} y1={pad.t} y2={H - pad.b} stroke="var(--ink-3)" strokeDasharray="3 3" />
        </g>
      ) : null}
      {ALGOS.map((a) => (
        <polyline
          key={a}
          data-curve={a}
          points={pts.map((q) => `${X(q.p)},${Y(q[a])}`).join(' ')}
          fill="none"
          stroke={STROKE[a]}
          strokeWidth={a === 'coarsened' ? 2.4 : 1.8}
          strokeDasharray={DASH[a]}
        />
      ))}
      <line x1={X(p)} x2={X(p)} y1={pad.t} y2={H - pad.b} stroke="var(--ink)" strokeWidth="1.2" />
      {ALGOS.map((a) => {
        const q = pts.find((x) => x.p === p);
        return q ? <circle key={'c' + a} cx={X(p)} cy={Y(q[a])} r="3.6" fill="var(--page)" stroke={STROKE[a]} strokeWidth="1.6" /> : null;
      })}
    </svg>
    </div>
  );
}

const PRESETS = [
  { key: 'few', get label() { return tr('1,024 elements, 32 units', '1,024 phần tử, 32 đơn vị'); }, logN: 10, logP: 5 },
  { key: 'many', get label() { return tr('1,024 elements, 1,024 units', '1,024 phần tử, 1,024 đơn vị'); }, logN: 10, logP: 10 },
  { key: 'big', get label() { return tr('16,384 elements, 1,024 units', '16,384 phần tử, 1,024 đơn vị'); }, logN: 14, logP: 10 },
];

function ScanChoice() {
  const { reports } = useReports();
  const widths = useMemo(() => widthChoices(reports as Reports), [reports]);
  const [logN, setLogN] = useState(10);
  const [logP, setLogP] = useState(5);
  const [charge, setCharge] = useState<Charge>('adds');
  const [wKey, setWKey] = useState('book');
  const warp = (widths.find((c) => c.key === wKey) ?? widths[0]).width;
  const n = 2 ** logN;
  const lp = Math.min(logP, logN);
  const p = 2 ** lp;
  const rows = compare(n, p, charge, warp);
  const pts = useMemo(() => curve(n, charge, warp), [n, charge, warp]);
  const bkUpTo = useMemo(() => brentKungUpTo(n, charge, warp), [n, charge, warp]);
  const ksFrom = useMemo(() => koggeStoneFrom(n, charge, warp), [n, charge, warp]);
  const net = networkWinner(rows);
  const all = overallWinner(rows);
  const lag = hybridLag(rows);
  const maxTime = Math.max(...ALGOS.map((a) => rows[a].time));
  const maxAdds = Math.max(...ALGOS.map((a) => rows[a].additions));

  return (
    <Bench
      title={tr('Which scan for this hardware?', 'Scan nào cho phần cứng này?')}
      task={tr(
        "Start at the question's case. Then give the block more units, one doubling at a time, until Kogge-Stone wins.",
        'Bắt đầu từ trường hợp của câu hỏi. Sau đó cho khối thêm đơn vị, mỗi lần gấp đôi, cho đến khi Kogge-Stone thắng.',
      )}
      foot={
        <Translated
          vi={
            <>
              Mọi con số đều lấy từ các mô hình của chương 11: các mạng từ mục 11.2 và 11.10, scan có gộp việc từ mục 11.6
              (2N/T + log₂T − 1 bước cho T luồng, mỗi luồng một đơn vị). Một bước có nhiều lane bị tính hơn số đơn vị thì
              tốn nhiều vòng. Mô hình: <code>src/sim/algoselect.ts</code>.
            </>
          }
        >
          <>
            Every count comes from chapter 11's models: the networks from section 11.2 and 11.10, the coarsened scan from
            section 11.6 (2N/T + log₂T − 1 steps for T threads, one per unit). A step with more charged lanes than units
            takes several rounds. Model: <code>src/sim/algoselect.ts</code>.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated
            vi={
              <span>
                N = <b data-value="n">{int(n)}</b> phần tử
              </span>
            }
          >
            <span>
              N = <b data-value="n">{int(n)}</b> elements
            </span>
          </Translated>
          <input
            type="range"
            min={N_MIN_LOG}
            max={N_MAX_LOG}
            value={logN}
            aria-label={tr('Elements, as a power of two', 'Số phần tử, dạng lũy thừa của hai')}
            onChange={(e) => setLogN(Number(e.target.value))}
            style={{ width: '12rem' }}
          />
        </label>
        <label className="field">
          <Translated
            vi={
              <span>
                P = <b data-value="p">{int(p)}</b> đơn vị thực thi
              </span>
            }
          >
            <span>
              P = <b data-value="p">{int(p)}</b> execution units
            </span>
          </Translated>
          <input
            type="range"
            min={0}
            max={logN}
            value={lp}
            aria-label={tr('Execution units, as a power of two', 'Số đơn vị thực thi, dạng lũy thừa của hai')}
            onChange={(e) => setLogP(Number(e.target.value))}
            style={{ width: '12rem' }}
          />
        </label>
        <span style={FIELD}>
          <Translated vi="Mỗi bước bị tính theo">Charge each step for</Translated>
          <span className="seg" role="group" aria-label={tr('Charge each step for', 'Mỗi bước bị tính theo')} style={WRAP}>
            {CHARGES.map((c) => (
              <button key={c} aria-pressed={charge === c} onClick={() => setCharge(c)}>
                {localized(CHARGE_LABEL[c], ({ adds: 'các phép cộng của nó', warps: 'trọn warp', threads: 'mọi luồng (sách)' } as Record<Charge, string>)[c])}
              </button>
            ))}
          </span>
        </span>
        <label className="field">
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <select value={wKey} aria-label={tr('warp width', 'độ rộng warp')} disabled={charge !== 'warps'} onChange={(e) => setWKey(e.target.value)}>
            {widths.map((c) => (
              <option key={c.key} value={c.key}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span style={FIELD}>
          <Translated vi="Thiết lập sẵn">Presets</Translated>
          <span className="seg" role="group" aria-label={tr('Presets', 'Thiết lập sẵn')} style={WRAP}>
            {PRESETS.map((q) => (
              <button
                key={q.key}
                onClick={() => {
                  setLogN(q.logN);
                  setLogP(q.logP);
                }}
              >
                {q.label}
              </button>
            ))}
          </span>
        </span>
      </div>
      <p className="muted" style={{ margin: '0 0 0.7rem' }} data-readout="charge">
        {CHARGE_NOTE[charge]}
      </p>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0, minWidth: '30rem' }} data-part="scan-table">
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}><Translated vi="Scan">Scan</Translated></th>
              <th className="num"><Translated vi="đơn vị thời gian trên P">time units on P</Translated></th>
              <th className="num"><Translated vi="phép cộng">additions</Translated></th>
              <th className="num"><Translated vi="số bước, không giới hạn đơn vị">steps, unlimited units</Translated></th>
              <th className="num"><Translated vi="luồng">threads</Translated></th>
            </tr>
          </thead>
          <tbody>
            {ALGOS.map((a) => (
              <tr key={a} data-row={a} data-time={rows[a].time} data-adds={rows[a].additions}>
                <td style={{ paddingLeft: '0.6rem' }}>{localized(ALGO_LABEL[a], ({ 'kogge-stone': 'Kogge-Stone', 'brent-kung': 'Brent-Kung', coarsened: 'Ba pha có gộp việc' } as Record<AlgoId, string>)[a])}</td>
                <td className="num">
                  {int(rows[a].time)}
                  <Bar value={rows[a].time} max={maxTime} />
                </td>
                <td className="num">
                  {int(rows[a].additions)}
                  <Bar value={rows[a].additions} max={maxAdds} />
                </td>
                <td className="num">{int(rows[a].span)}</td>
                <td className="num">{int(rows[a].threads)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="cols plot" style={{ marginTop: '0.9rem' }}>
        <TimePlot pts={pts} n={n} p={p} bkUpTo={bkUpTo} ksFrom={ksFrom} />
        <div className="panel" aria-live="polite" data-readout="winner" data-network={net} data-overall={all}>
          <p style={{ margin: 0 }}>
            <Translated vi="Mạng nhanh hơn trong hai mạng:">Faster of the two networks:</Translated>{' '}
            {net === 'tie' ? (
              <span className="chip" data-value="network-winner">
                <Translated vi="hòa">a tie</Translated>
              </span>
            ) : (
              <span className="chip ok" data-value="network-winner">
                {localized(ALGO_LABEL[net], ({ 'kogge-stone': 'Kogge-Stone', 'brent-kung': 'Brent-Kung', coarsened: 'Ba pha có gộp việc' } as Record<AlgoId, string>)[net])}
              </span>
            )}{' '}
            <span data-value="network-times">
              {net === 'kogge-stone'
                ? tr(
                    `${int(rows['kogge-stone'].time)} time units against Brent-Kung's ${int(rows['brent-kung'].time)}.`,
                    `${int(rows['kogge-stone'].time)} đơn vị thời gian so với ${int(rows['brent-kung'].time)} của Brent-Kung.`,
                  )
                : net === 'brent-kung'
                  ? tr(
                      `${int(rows['brent-kung'].time)} time units against Kogge-Stone's ${int(rows['kogge-stone'].time)}.`,
                      `${int(rows['brent-kung'].time)} đơn vị thời gian so với ${int(rows['kogge-stone'].time)} của Kogge-Stone.`,
                    )
                  : tr(`both take ${int(rows['brent-kung'].time)} time units.`, `cả hai đều tốn ${int(rows['brent-kung'].time)} đơn vị thời gian.`)}
            </span>
          </p>
          <p style={{ margin: '0.6rem 0 0' }} data-readout="hybrid">
            <Translated
              vi={
                <>
                  Scan có gộp việc: {int(rows.coarsened.time)} đơn vị thời gian với {int(rows.coarsened.threads)}{' '}
                  {rows.coarsened.threads === 1 ? 'luồng, tức là nó thành vòng lặp tuần tự' : 'luồng'};{' '}
                </>
              }
            >
              The coarsened scan: {int(rows.coarsened.time)} time units with {int(rows.coarsened.threads)}{' '}
              {rows.coarsened.threads === 1 ? 'thread, which makes it the sequential loop' : 'threads'};{' '}
            </Translated>
            {lag > 0
              ? tr(`${int(lag)} behind the better network.`, `chậm hơn mạng tốt hơn ${int(lag)}.`)
              : lag === 0
                ? tr('level with the better network.', 'ngang với mạng tốt hơn.')
                : tr(`${int(-lag)} ahead of the better network.`, `nhanh hơn mạng tốt hơn ${int(-lag)}.`)}{' '}
            {all === 'coarsened' ? <span className="chip ok"><Translated vi="nhanh nhất trong ba">fastest of the three</Translated></span> : null}
          </p>
          <p className="muted" style={{ margin: '0.6rem 0 0' }} data-readout="crossover" data-bk-upto={bkUpTo ?? ''} data-ks-from={ksFrom ?? ''}>
            <Translated
              vi={
                <>
                  Xét mọi số nguyên đơn vị, không chỉ lũy thừa của hai: Brent-Kung nhanh hơn với mọi P đến{' '}
                  <b data-value="bk-upto">{bkUpTo == null ? 'không có' : int(bkUpTo)}</b> (vùng tô), và Kogge-Stone với mọi P từ{' '}
                  <b data-value="ks-from">{ksFrom == null ? 'không có' : int(ksFrom)}</b> (đường nét đứt).
                  {bkUpTo != null && ksFrom != null && ksFrom > bkUpTo + 1 ? ' Ở giữa, chúng hòa hoặc thay nhau dẫn.' : ''}
                </>
              }
            >
              Over every whole number of units, not only powers of two: Brent-Kung is faster for every P up to{' '}
              <b data-value="bk-upto">{bkUpTo == null ? 'none' : int(bkUpTo)}</b> (shaded), and Kogge-Stone for every P from{' '}
              <b data-value="ks-from">{ksFrom == null ? 'none' : int(ksFrom)}</b> (dashed line).
              {bkUpTo != null && ksFrom != null && ksFrom > bkUpTo + 1 ? ' In between they tie or take turns.' : ''}
            </Translated>
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer 2: bin capacity
const sOf = (v: number) => Math.round(10 ** (v / 10));
const S_DEFAULT_V = 13; // S = 20

function BinMap({ counts, capacity, fullest }: { counts: number[]; capacity: number; fullest: number }) {
  const cell = 12;
  const size = GRID * cell;
  return (
    <svg viewBox={`-1 -1 ${size + 2} ${size + 2}`} role="img" aria-label={tr('Atoms per bin, over the domain', 'Số nguyên tử mỗi bin, trên toàn miền')} style={{ width: '100%', maxWidth: 230, display: 'block' }}>
      {counts.map((k, i) => {
        const x = (i % GRID) * cell;
        const y = Math.floor(i / GRID) * cell;
        const over = k > capacity;
        return (
          <rect
            key={i}
            x={x + 0.5}
            y={y + 0.5}
            width={cell - 1}
            height={cell - 1}
            fill="var(--global)"
            fillOpacity={fullest ? 0.06 + 0.94 * (k / fullest) : 0.06}
            stroke={over ? 'var(--ink)' : 'none'}
            strokeWidth={over ? 1.6 : 0}
            data-over={over ? '' : undefined}
          />
        );
      })}
    </svg>
  );
}

function Histogram({ counts, capacity, fullest }: { counts: number[]; capacity: number; fullest: number }) {
  const h = histogram(counts);
  const buckets = Math.min(48, fullest + 1);
  const width = Math.ceil((fullest + 1) / buckets);
  const nb = Math.ceil((fullest + 1) / width);
  const fit: number[] = new Array(nb).fill(0);
  const over: number[] = new Array(nb).fill(0);
  h.forEach((bins, k) => {
    const b = Math.floor(k / width);
    if (k > capacity) over[b] += bins;
    else fit[b] += bins;
  });
  const top = Math.max(1, ...fit.map((v, i) => v + over[i]));
  const W = 420;
  const H = 210;
  const pad = { l: 40, r: 10, t: 26, b: 40 };
  const bw = (W - pad.l - pad.r) / nb;
  const Y = (v: number) => H - pad.b - (v / top) * (H - pad.t - pad.b);
  const cx = pad.l + ((capacity + 0.5) / width) * bw;
  const ticks = [0, Math.round(fullest / 2), fullest];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Histogram of atoms per bin with the capacity', 'Histogram số nguyên tử mỗi bin cùng với sức chứa')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <line x1={pad.l} x2={W - pad.r} y1={H - pad.b} y2={H - pad.b} stroke="var(--rule)" />
      {fit.map((v, i) => (
        <g key={i}>
          {v ? <rect x={pad.l + i * bw + 0.5} y={Y(v + over[i])} width={Math.max(1, bw - 1)} height={Y(over[i]) - Y(v + over[i])} fill="var(--global)" /> : null}
          {over[i] ? <rect x={pad.l + i * bw + 0.5} y={Y(over[i])} width={Math.max(1, bw - 1)} height={H - pad.b - Y(over[i])} fill="var(--ink-2)" /> : null}
        </g>
      ))}
      {ticks.map((t) => (
        <text key={t} x={pad.l + ((t + 0.5) / width) * bw} y={H - pad.b + 14} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
          {t}
        </text>
      ))}
      <text x={(W + pad.l) / 2} y={H - 8} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi="số nguyên tử trong một bin">atoms in a bin</Translated>
      </text>
      <text x={pad.l - 6} y={Y(top) + 4} fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
        {top}
      </text>
      <text x={pad.l - 6} y={H - pad.b} fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
        0
      </text>
      <text x={12} y={(H - pad.b + pad.t) / 2} fontSize="11.5" textAnchor="middle" transform={`rotate(-90 12 ${(H - pad.b + pad.t) / 2})`} style={{ fill: 'var(--ink-2)' }}>
        <Translated vi="số bin">bins</Translated>
      </text>
      <line x1={cx} x2={cx} y1={pad.t} y2={H - pad.b} stroke="var(--ink)" strokeWidth="1.6" strokeDasharray="4 3" data-mark="capacity" />
      <line x1={cx} x2={cx} y1={pad.t - 8} y2={pad.t} stroke="var(--ink)" strokeWidth="1.6" />
      <text x={Math.max(pad.l, Math.min(cx - 30, W - pad.r - 80))} y={pad.t - 12} fontSize="11.5" style={{ fill: 'var(--ink)' }}>
        <Translated vi={<>sức chứa {capacity}</>}>capacity {capacity}</Translated>
      </text>
    </svg>
  );
}

function HBar({ label, value, max, fill, data }: { label: string; value: number; max: number; fill: string; data: string }) {
  const w = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div style={{ marginBottom: '0.45rem' }} data-bar={data} data-units={Math.round(value)}>
      <div className="ui" style={{ display: 'flex', justifyContent: 'space-between', gap: '0.6rem', fontSize: '0.9em' }}>
        <span>{label}</span>
        <span className="num">{int(value)}</span>
      </div>
      <span aria-hidden="true" style={{ display: 'block', height: 9, background: 'var(--rule-soft)', borderRadius: 2 }}>
        <span style={{ display: 'block', height: 9, width: `${Math.max(value > 0 ? 0.6 : 0, w)}%`, background: fill, borderRadius: 2 }} />
      </span>
    </div>
  );
}

function BinCapacity() {
  const [dist, setDist] = useState<Distribution>('pocket');
  const [atoms, setAtoms] = useState(4096);
  const [cap, setCap] = useState(Infinity);
  const [sv, setSv] = useState(S_DEFAULT_V);
  const S = sOf(sv);
  const counts = useMemo(() => countsFor(dist, atoms), [dist, atoms]);
  const fullest = Math.max(...counts);
  const c = Math.max(0, Math.min(cap, fullest));
  const b = binWith(counts, c);
  const w = work(counts, c, S);
  const wFull = work(counts, fullest, S);
  const c3 = capacityForShare(counts, BOOK_SHARE);
  const cBest = useMemo(() => bestCapacity(counts, S), [counts, S]);
  const scale = Math.max(wFull.serial, w.serial);
  const oneInS = BINS / S;

  return (
    <Bench
      title={tr('Bin capacity and the overflow list', 'Sức chứa của bin và danh sách tràn')}
      task={tr(
        "Start sized for the fullest bin. Drag the capacity down until the host's pass overtakes the device, then find the lowest total.",
        'Bắt đầu với kích thước cho bin đầy nhất. Kéo sức chứa xuống cho đến khi lượt chạy của host vượt device, rồi tìm tổng thấp nhất.',
      )}
      foot={
        <Translated
          vi={
            <>
              {GRID} × {GRID} bin, mỗi bin một khối gồm {POINTS_PER_BIN} điểm lưới (8 × 8), vùng lân cận 3 × 3 bin. Một
              đơn vị là một lần kiểm tra một ô nguyên tử cho một điểm lưới. Các nguyên tử được sinh từ một seed cố định. Mô
              hình:{' '}
              <code>src/sim/binning.ts</code>.
            </>
          }
        >
          <>
            {GRID} × {GRID} bins, one block of {POINTS_PER_BIN} grid points (8 × 8) per bin, a 3 × 3 neighbourhood of bins.
            One unit is one test of one atom slot for one grid point. Atoms are generated from a fixed seed. Model:{' '}
            <code>src/sim/binning.ts</code>.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Cách đặt nguyên tử">Atoms placed</Translated>
          <span className="seg" role="group" aria-label={tr('Distribution', 'Phân bố')} style={WRAP}>
            {DISTRIBUTIONS.map((d) => (
              <button key={d} aria-pressed={dist === d} onClick={() => setDist(d)}>
                {localized(DIST_LABEL[d], ({ uniform: 'đều', clustered: 'theo cụm', pocket: 'một túi dày đặc' } as Record<Distribution, string>)[d])}
              </button>
            ))}
          </span>
        </span>
        <label className="field">
          <Translated
            vi={
              <span>
                <b data-value="atoms">{int(atoms)}</b> nguyên tử
              </span>
            }
          >
            <span>
              <b data-value="atoms">{int(atoms)}</b> atoms
            </span>
          </Translated>
          <input type="range" min={1024} max={16384} step={1024} value={atoms} aria-label={tr('Atoms', 'Số nguyên tử')} onChange={(e) => setAtoms(Number(e.target.value))} style={{ width: '11rem' }} />
        </label>
        <label className="field">
          <Translated
            vi={
              <span>
                S = <b data-value="speedup">{S}</b>: số lần kiểm tra trên device cho mỗi lần kiểm tra trên host
              </span>
            }
          >
            <span>
              S = <b data-value="speedup">{S}</b>: device tests per host test
            </span>
          </Translated>
          <input type="range" min={0} max={30} value={sv} aria-label={tr('Device speed-up over the host', 'Mức tăng tốc của device so với host')} onChange={(e) => setSv(Number(e.target.value))} style={{ width: '11rem' }} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated
            vi={
              <span>
                Sức chứa c = <b data-value="capacity">{c}</b> nguyên tử mỗi bin
              </span>
            }
          >
            <span>
              Capacity c = <b data-value="capacity">{c}</b> atoms per bin
            </span>
          </Translated>
          <input type="range" min={0} max={fullest} value={c} aria-label={tr('Bin capacity', 'Sức chứa của bin')} onChange={(e) => setCap(Number(e.target.value))} style={{ width: '16rem' }} />
        </label>
        <span style={FIELD}>
          <Translated vi="Đặt sức chứa">Set the capacity</Translated>
          <span className="seg" role="group" aria-label={tr('Set the capacity', 'Đặt sức chứa')} style={WRAP}>
            <button onClick={() => setCap(fullest)} aria-pressed={c === fullest}>
              <Translated vi={<>bin đầy nhất ({fullest})</>}>fullest bin ({fullest})</Translated>
            </button>
            <button onClick={() => setCap(c3)} aria-pressed={c === c3}>
              <Translated vi={<>nhỏ nhất dưới 3% ({c3})</>}>smallest under 3% ({c3})</Translated>
            </button>
            <button onClick={() => setCap(cBest)} aria-pressed={c === cBest}>
              <Translated vi={<>tổng thời gian thấp nhất ({cBest})</>}>lowest total time ({cBest})</Translated>
            </button>
          </span>
        </span>
      </div>
      <div className="cols" style={{ alignItems: 'start' }}>
        <div className="panel">
          <Translated
            vi={
              <p className="muted" style={{ margin: '0 0 0.4rem' }}>
                Miền tính: bin càng đậm chứa càng nhiều nguyên tử (các bin nằm trong bộ nhớ toàn cục (global memory)); bin có viền là bin vượt sức chứa.
              </p>
            }
          >
            <p className="muted" style={{ margin: '0 0 0.4rem' }}>
              The domain: darker bins hold more atoms (bins live in global memory); outlined bins are over capacity.
            </p>
          </Translated>
          <BinMap counts={counts} capacity={c} fullest={fullest} />
        </div>
        <div className="panel">
          <Translated
            vi={
              <p className="muted" style={{ margin: '0 0 0.4rem' }}>
                Các bin theo số nguyên tử chúng chứa. <span className="chip global">vừa trong bin của nó</span>{' '}
                <span className="chip" style={{ background: 'var(--ink-2)', color: 'var(--page)' }}>
                  vượt sức chứa
                </span>
              </p>
            }
          >
            <p className="muted" style={{ margin: '0 0 0.4rem' }}>
              Bins by how many atoms they hold. <span className="chip global">fits in its bin</span>{' '}
              <span className="chip" style={{ background: 'var(--ink-2)', color: 'var(--page)' }}>
                over capacity
              </span>
            </p>
          </Translated>
          <Histogram counts={counts} capacity={c} fullest={fullest} />
        </div>
      </div>
      <div className="cols" style={{ marginTop: '0.9rem', alignItems: 'start' }}>
        <div className="panel" data-readout="memory">
          <Translated
            vi={
              <p style={{ margin: 0 }}>
                <b>Bộ nhớ cho các bin</b>: {BINS} × {c} = <b data-value="slots">{int(b.slots)}</b> ô, trong đó{' '}
                <span data-value="real">{int(b.real)}</span> nguyên tử thật và <span data-value="dummies">{int(b.dummies)}</span>{' '}
                nguyên tử giả điện tích 0 ({b.slots ? pct(b.dummies / b.slots) : '0'}%).
              </p>
            }
          >
            <p style={{ margin: 0 }}>
              <b>Memory for the bins</b>: {BINS} × {c} = <b data-value="slots">{int(b.slots)}</b> slots, of which{' '}
              <span data-value="real">{int(b.real)}</span> real atoms and <span data-value="dummies">{int(b.dummies)}</span>{' '}
              zero-charge dummies ({b.slots ? pct(b.dummies / b.slots) : '0'}%).
            </p>
          </Translated>
          <p className="muted" style={{ margin: '0.4rem 0 0' }} data-readout="full-sizing">
            {c === fullest
              ? tr(
                  `This is the sizing for the fullest bin: every bin has room for ${fullest} atoms.`,
                  `Đây là kích thước cho bin đầy nhất: mọi bin đều có chỗ cho ${fullest} nguyên tử.`,
                )
              : tr(
                  `Sized for the fullest bin it would be ${BINS} × ${fullest} = ${int(BINS * fullest)} slots${b.slots ? `, ${dec((BINS * fullest) / b.slots, 1)} times as many` : ''}.`,
                  `Nếu định kích thước cho bin đầy nhất thì sẽ là ${BINS} × ${fullest} = ${int(BINS * fullest)} ô${b.slots ? `, gấp ${dec((BINS * fullest) / b.slots, 1)} lần` : ''}.`,
                )}
          </p>
          <p style={{ margin: '0.7rem 0 0' }} data-readout="overflow" data-share={b.overflowShare}>
            <Translated
              vi={
                <>
                  <b>Danh sách tràn</b>: <b data-value="overflow">{int(b.overflow)}</b> nguyên tử từ{' '}
                  <span data-value="overflowing-bins">{b.overflowingBins}</span> bin, <b data-value="share">{pct(b.overflowShare)}%</b> tổng số nguyên tử{' '}
                </>
              }
            >
              <b>Overflow list</b>: <b data-value="overflow">{int(b.overflow)}</b> atoms from{' '}
              <span data-value="overflowing-bins">{b.overflowingBins}</span> bins, <b data-value="share">{pct(b.overflowShare)}%</b> of all atoms{' '}
            </Translated>
            {b.overflowShare <= BOOK_SHARE ? (
              <span className="chip ok"><Translated vi="trong mức 3% của sách">within the book's 3%</Translated></span>
            ) : (
              <span className="chip"><Translated vi="trên mức 3% của sách">above the book's 3%</Translated></span>
            )}
          </p>
          <Translated
            vi={
              <p className="muted" style={{ margin: '0.4rem 0 0' }}>
                Tăng c có lợi chừng nào còn hơn khoảng một bin trên S bị tràn: ở đây là khoảng {dec(oneInS, 0)} trên {BINS}.
              </p>
            }
          >
            <p className="muted" style={{ margin: '0.4rem 0 0' }}>
              Raising c pays while more than about one bin in S overflows: about {dec(oneInS, 0)} of {BINS} here.
            </p>
          </Translated>
        </div>
        <div className="panel" data-readout="work">
          <HBar label={tr(`Device: ${int(w.device)} slot tests ÷ S`, `Device: ${int(w.device)} lần kiểm tra ô ÷ S`)} value={w.deviceTime} max={scale} fill="var(--ink-2)" data="device" />{/* i18n-ignore: data key */}
          <HBar label={tr(`Host: ${int(b.overflow)} overflow atoms, one test at a time`, `Host: ${int(b.overflow)} nguyên tử tràn, mỗi lần một phép kiểm tra`)} value={w.hostTime} max={scale} fill="var(--ink)" data="host" />{/* i18n-ignore: data key */}
          <HBar label={tr('Kernel, then the host pass', 'Kernel, rồi đến lượt chạy của host')} value={w.serial} max={scale} fill="var(--ink-3)" data="serial" />{/* i18n-ignore: data key */}
          <Translated
            vi={
              <p className="muted" style={{ margin: '0.5rem 0 0' }}>
                Tính bằng thời gian một lần kiểm tra trên host. {w.device ? `${pct(w.wasted / w.device)}% số lần kiểm tra của device rơi vào nguyên tử giả. ` : ''}
                Với kích thước cho bin đầy nhất: {int(wFull.serial)}. Giấu lượt chạy của host sau kernel kế tiếp (cải tiến của
                chương 21) sẽ chỉ còn lại phần lớn hơn trong hai phần: {int(w.overlapped)}.
              </p>
            }
          >
            <p className="muted" style={{ margin: '0.5rem 0 0' }}>
              In host-test times. {w.device ? `${pct(w.wasted / w.device)}% of the device's tests are on dummies. ` : ''}
              Sized for the fullest bin: {int(wFull.serial)}. Hiding the host pass behind the next kernel (chapter 21's
              refinement) would leave the larger of the two: {int(w.overlapped)}.
            </p>
          </Translated>
          <p style={{ margin: '0.5rem 0 0' }} data-readout="bottleneck">
            {w.hostTime > w.deviceTime ? (
              <span className="chip fail"><Translated vi="lượt chạy tuần tự của host lâu hơn kernel">the host's serial pass takes longer than the kernel</Translated></span>
            ) : (
              <span className="chip ok"><Translated vi="kernel lâu hơn lượt chạy của host">the kernel takes longer than the host's pass</Translated></span>
            )}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const Q = compare(1024, 32, 'adds', 32);
const Q_BK = Q['brent-kung'].time;
const Q_KS = Q['kogge-stone'].time;
const KS_FROM_1024 = koggeStoneFrom(1024, 'adds', 32)!;
const BK_UPTO_1024 = brentKungUpTo(1024, 'adds', 32)!;

export const questionsS221: Record<string, Question> = {
  'ch22/22.1/q-scan-units': {
    kind: 'choice',
    id: 'ch22/22.1/q-scan-units',
    prompt: (
      <Translated
        vi={
          <>
            Một khối thực hiện scan trên 1,024 phần tử và chỉ giữ được 32 bộ cộng bận rộn trong một bước. Nếu chỉ đếm phép
            cộng, scan nào xong trước?
          </>
        }
      >
        <>
          A block scans 1,024 elements and can keep only 32 adders busy in a step. Counting additions only, which scan
          finishes first?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Brent-Kung, dù tốn 19 bước so với 10 bước của Kogge-Stone: với 32 bộ cộng, tổng số phép cộng mới là yếu tố quyết định.">Brent-Kung, despite 19 steps to Kogge-Stone’s 10: with 32 adders, the total additions decide.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Kogge-Stone, vì nó tốn 10 bước thay vì 19, và số bước là thứ một scan phải chờ.">Kogge-Stone, because it takes 10 steps instead of 19, and the number of steps is what a scan waits for.</Translated>,
        why: <Translated vi="Điều đó đúng khi mọi phép cộng của một bước có thể chạy cùng lúc. Với 32 đơn vị, mỗi bước của Kogge-Stone tốn tới 32 vòng.">That holds when every addition of a step can run at once. With 32 units each of Kogge-Stone’s steps takes up to 32 rounds.</Translated>,
      },
      {
        label: <Translated vi="Hòa nhau, vì rốt cuộc cả hai làm cùng số phép cộng, chỉ nhóm vào các bước khác nhau.">They tie, because both end up doing the same additions, only grouped into steps differently.</Translated>,
        why: <Translated vi="Kogge-Stone làm 9,217 phép cộng trên 1,024 phần tử, Brent-Kung làm 2,036.">Kogge-Stone does 9,217 additions on 1,024 elements, Brent-Kung 2,036.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mỗi bước trong 10 bước của Kogge-Stone có từ 512 đến 1,023 phép cộng, nên trên 32 đơn vị mỗi bước tốn 16 đến 32
            vòng: tổng cộng {Q_KS}. 19 bước của Brent-Kung co lại nhanh, phần lớn chỉ còn một vòng: {Q_BK}. Xét mọi số nguyên
            đơn vị, Brent-Kung dẫn trước đến {BK_UPTO_1024} và Kogge-Stone chỉ thắng từ{' '}
            {KS_FROM_1024}, tức ba phần tư đơn vị cho mỗi phần tử. Trình khám phá bên dưới cho bạn kiểm tra cả hai.
          </p>
        }
      >
        <p>
          Kogge-Stone&rsquo;s 10 steps have between 512 and 1,023 additions each, so on 32 units each takes 16 to 32
          rounds: {Q_KS} in all. Brent-Kung&rsquo;s 19 steps shrink fast, most of them to a single round: {Q_BK}. Counting
          every whole number of units, Brent-Kung stays ahead up to {BK_UPTO_1024} and Kogge-Stone wins only from{' '}
          {KS_FROM_1024}, three quarters of a unit per element. The explorer below lets you check both.
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.1, p. 530; ch. 11 §11.5 and §11.10, pp. 265–267, 282–285',
    vi: { source: 'ch. 22 §22.1, p. 530; ch. 11 §11.5 và §11.10, pp. 265–267, 282–285' },
  },
  'ch22/22.1/q-sort-logits': {
    kind: 'choice',
    id: 'ch22/22.1/q-sort-logits',
    prompt: (
      <Translated
        vi={
          <>
            Bạn cần sắp xếp 128,000 logits kiểu float cho lấy mẫu top-p. Radix sort là cách sắp xếp nhanh nhất bạn có, nhưng
            sách nói nó chỉ dùng được cho một số kiểu khóa. Bạn có dùng được nó không?
          </>
        }
      >
        <>
          You need 128,000 float logits sorted for top-p sampling. Radix sort is your fastest sort, but the book says it
          works only for some key types. Can you use it?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Có, sau khi ánh xạ các bit của mỗi float sang một khóa không dấu giữ nguyên thứ tự.">Yes, after mapping each float’s bits to an order-preserving unsigned key.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Không: radix sort cần khóa số nguyên còn logits là float, nên đây chính là trường hợp mà tính tổng quát của merge sort sinh ra để giải quyết.">No: radix sort needs integer keys and logits are floats, so this is the case merge sort’s generality exists for.</Translated>,
        why: <Translated vi="Một phép biến đổi bit biến mỗi float thành một số nguyên không dấu có cùng thứ tự, và khi đó radix sort áp dụng được.">A bit transform turns each float into an unsigned integer with the same order, and then radix sort applies.</Translated>,
      },
      {
        label: <Translated vi="Có, giữ nguyên như vậy: các bit của một float, đọc như số nguyên không dấu, vốn đã sắp xếp theo thứ tự của các float.">Yes, as they are: the bits of a float, read as an unsigned integer, already sort in the order of the floats.</Translated>,
        why: <Translated vi="Chỉ đúng với giá trị không âm. Float âm có bit dấu được bật, nên chúng sẽ nằm trên mọi số dương, theo thứ tự ngược.">Only for non-negative values. Negative floats have the sign bit set, so they would land above every positive one, in reverse order.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Bật bit dấu của một float không âm và đảo mọi bit của một float âm: đọc như số nguyên không dấu, kết quả nằm theo
            đúng thứ tự của các float. Đó là phép biến đổi CUB mô tả cho radix sort của nó, và là phép bạn đã gặp ở{' '}
            <a href={href('ch14', '14.4')}>chương 14</a>. Tính tổng quát là thứ merge sort giữ được: nó sắp xếp theo bất kỳ
            phép so sánh nào bạn viết được, kể cả phép so sánh mà chưa ai có cách mã hóa bit giữ nguyên thứ tự.
          </p>
        }
      >
        <p>
          Set the sign bit of a non-negative float and invert every bit of a negative one: read as unsigned integers, the
          results are in the floats&rsquo; order. That is the transform CUB documents for its radix sort, and the one you
          met in <a href={href('ch14', '14.4')}>chapter 14</a>. Generality is what merge sort keeps: it sorts by any
          comparison you can write, including one for which nobody has an order-preserving bit encoding.
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.1, p. 530; ch. 14 §14.4 lesson (ours) and notes/today/ch14-facts.md A3',
    vi: { source: 'ch. 22 §22.1, p. 530; bài học ch. 14 §14.4 (của khóa học này) và notes/today/ch14-facts.md A3' },
  },
  'ch22/22.1/q-overflow': {
    kind: 'choice',
    id: 'ch22/22.1/q-overflow',
    prompt: (
      <Translated
        vi={
          <>
            Các nguyên tử dồn vào một vài bin. So với việc định kích thước mọi bin theo bin đầy nhất, một sức chứa gần mức
            thông thường cộng với một danh sách tràn phải trả giá gì?
          </>
        }
      >
        <>
          Atoms crowd into a few bins. Compared with sizing every bin for the fullest one, what does a capacity near the
          typical count plus an overflow list cost?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Một lượt chạy tuần tự ngắn trên số ít nguyên tử tràn sau kernel, đổi lại tốn ít bộ nhớ hơn nhiều và quét ít ô trống hơn hẳn.">A short serial pass over the few overflow atoms after the kernel, for much less memory and far fewer empty slots scanned.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Không gì cả: các nguyên tử không vừa bị bỏ đi, giống như ngưỡng cắt vốn đã bỏ các nguyên tử ở xa.">Nothing at all: atoms that do not fit are dropped, just as the cutoff already drops the far atoms.</Translated>,
        why: <Translated vi="Nguyên tử ở xa bị bỏ qua có chủ đích. Nguyên tử tràn thì ở gần, và phần đóng góp của chúng phải được cộng vào để hoàn tất kết quả (p. 531).">Far atoms are left out by design. Overflow atoms are near, and their contributions must be added to complete the result (p. 531).</Translated>,
      },
      {
        label: <Translated vi="Thêm việc cho device, vì giờ mỗi khối còn phải tìm trong danh sách tràn các nguyên tử gần điểm lưới của mình trước khi ghi chúng.">More work on the device, because every block must now also search the overflow list for atoms near its grid points before it writes them.</Translated>,
        why: <Translated vi="Device không bao giờ đọc danh sách này. Nó được xử lý sau kernel, bởi host trong chương 21 (p. 527).">The device never reads the list. It is processed after the kernel, by the host in chapter 21 (p. 527).</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mọi ô của mọi bin đều được quét bởi mọi điểm lưới trong vùng lân cận của nó, dù là nguyên tử giả hay không, nên sức
            chứa nhân lên cả khối lượng việc của device lẫn bộ nhớ. Danh sách tràn chuyển phần dư hiếm hoi sang một lượt chạy
            tuần tự. Nó chỉ rẻ khi còn nhỏ: trong trình khám phá, sức chứa quá thấp khiến host trở thành nút thắt cổ chai.
          </p>
        }
      >
        <p>
          Every slot of every bin is scanned by every grid point in its neighbourhood, dummy or not, so the capacity
          multiplies the device&rsquo;s work and the memory. The overflow list moves the rare excess to a serial pass. It
          stays cheap only while it stays small: in the explorer, a capacity too low makes the host the bottleneck.
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.1, p. 531; ch. 21 §21.5, p. 527', // i18n-ignore: citation
  },
  'ch22/22.1/q-cutoff-axis': {
    kind: 'choice',
    id: 'ch22/22.1/q-cutoff-axis',
    prompt: (
      <Translated
        vi={
          <>
            Trong bốn trục (số bước, mức song song lộ ra, tính tổng quát, độ chính xác), phép cộng có ngưỡng cắt đánh đổi hai
            trục nào với nhau, so với phép cộng trực tiếp?
          </>
        }
      >
        <>
          Of the four axes (steps, exposed parallelism, generality, accuracy), which two does cutoff summation trade
          against each other, compared with direct summation?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Độ chính xác, hy sinh một chút, để đổi lấy số bước, giảm đi rất nhiều.">Accuracy, given up a little, for steps, cut a great deal.</Translated>, correct: true },
      {
        label: <Translated vi="Mức song song đổi lấy số bước: phiên bản có ngưỡng cắt có ít điểm lưới có thể tính cùng lúc hơn.">Parallelism for steps: the cutoff version has fewer grid points it can compute at once.</Translated>,
        why: <Translated vi="Cả hai đều chạy mỗi điểm lưới một luồng. Sách nói cả hai đều lộ ra rất nhiều song song (p. 530).">Both run one thread per grid point. The book says both expose ample parallelism (p. 530).</Translated>,
      },
      {
        label: <Translated vi="Tính tổng quát đổi lấy độ chính xác: phiên bản có ngưỡng cắt chỉ dùng được cho một số loại nguyên tử.">Generality for accuracy: the cutoff version works only for certain kinds of atoms.</Translated>,
        why: <Translated vi="Nó dùng được cho mọi nguyên tử. Thứ nó cần là một định luật vật lý mà theo đó các đóng góp ở xa là nhỏ (p. 531).">It works for any atoms. What it needs is a physical law under which far contributions are small (p. 531).</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Khối lượng việc của phép cộng trực tiếp tăng theo bình phương thể tích; của phép cộng có ngưỡng cắt tăng tuyến tính,
            vì mỗi điểm lưới chỉ cộng các nguyên tử gần nó. Cái giá là phần đóng góp riêng lẻ của các nguyên tử ở xa, mà vật lý
            cho biết là nhỏ và có thể xử lý gộp. Mục <a href={href('ch22', '22.4')}>22.4</a>{' '}
            gọi kiểu bước đi này là đặt lại bài toán. Cả hai đều lộ ra rất nhiều song song, mỗi điểm lưới một luồng (p. 530),
            nên song song không phải là thứ bị đánh đổi.
          </p>
        }
      >
        <p>
          Direct summation&rsquo;s work grows with the square of the volume; cutoff summation&rsquo;s grows linearly,
          because each grid point only sums the atoms near it. The price is the far atoms&rsquo; individual contributions,
          which the physics says are small and can be handled together. Section <a href={href('ch22', '22.4')}>22.4</a>{' '}
          calls this kind of move a reformulation. Both expose ample parallelism, one thread per grid point (p. 530), so
          parallelism is not what is traded.
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.1, pp. 530–531',
  },
  'ch22/22.1/q-two-kernels': {
    kind: 'choice',
    id: 'ch22/22.1/q-two-kernels',
    prompt: (
      <Translated
        vi={
          <>
            llama.cpp dùng kernel nhân ma trận với vector cho tối đa khoảng 8 token và một kernel ma trận trên mức đó, với các
            ngoại lệ theo từng GPU được ghi &ldquo;tuned on RTX 4090&rdquo;. Vì sao là một ngưỡng đo được chứ không phải suy ra?
          </>
        }
      >
        <>
          llama.cpp uses matrix-vector kernels for up to about 8 tokens and a matrix kernel above that, with per-GPU
          exceptions marked &ldquo;tuned on RTX 4090&rdquo;. Why a measured threshold rather than a derived one?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Thuật toán nào thắng phụ thuộc vào phần cứng và định dạng trọng số, nên điểm giao phải được đo trên từng trường hợp.">Which algorithm wins depends on the hardware and the weight format, so the crossover has to be measured on each.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Kernel nhân ma trận với vector chính xác hơn, nên nó được giữ cho các batch nhỏ, nơi người dùng nhận ra lỗi.">The matrix-vector kernel is more accurate, so it is kept for the small batches where users notice errors.</Translated>,
        why: <Translated vi="Cả hai tính cùng các tích. Không có gì trong việc chọn liên quan đến độ chính xác.">Both compute the same products. Nothing in the selection is about accuracy.</Translated>,
      },
      {
        label: <Translated vi="Đó là giới hạn phần cứng: GPU không thể chạy kernel ma trận trên một batch ít hơn 8 hàng.">It is a hardware limit: a GPU cannot run a matrix kernel on a batch of fewer than 8 rows.</Translated>,
        why: <Translated vi="Kernel ma trận vẫn chạy được trên một hàng. Nó chỉ làm thêm phần việc đệm, bị lãng phí, như M3 làm ở batch 3.">A matrix kernel runs on one row too. It just does padded, wasted work, as M3 does at batch 3.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Quy tắc của sách là chọn phương án dung hòa tốt nhất cho phần cứng đang có. Ở đây hai thuật toán khác nhau ở cách
            tái sử dụng trọng số và ở lượng việc giải mã chúng lặp lại cho mỗi token, và nơi các chi phí đó giao nhau phụ thuộc
            vào thiết bị và định dạng. Một giới hạn được hạ xuống cho một số định dạng trên những GPU cụ thể chính là quy tắc đó
            được viết vào một hàm dispatch.
          </p>
        }
      >
        <p>
          The book&rsquo;s rule is to choose the best compromise for the given hardware. Here the two algorithms differ in
          how they reuse weights and in how much decoding work they repeat per token, and where those costs cross depends
          on the device and on the format. A limit lowered for some formats on particular GPUs is that rule written into a
          dispatch function.
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.1, p. 530; notes/today/ch22-facts.md D6', // i18n-ignore: citation
  },
};

export function S221() {
  const q = questionsS221;
  return (
    <>
      <Translated
        vi={
          <p>
            Theo định nghĩa của sách, một thuật toán là một thủ tục trong đó mọi bước được phát biểu chính xác (tính xác
            định), mọi bước đều có thể được máy tính thực hiện (tính khả thi), và toàn bộ được bảo đảm kết thúc (tính hữu
            hạn). Điều quan trọng ở đây là hầu hết các bài toán đều có nhiều thuật toán như vậy.
          </p>
        }
      >
        <p>
          An algorithm, in the book&rsquo;s definition, is a procedure in which every step is stated precisely
          (definiteness), every step can be carried out by a computer (effective computability), and the whole is
          guaranteed to end (finiteness). What matters here is that most problems have several of them.
        </p>
      </Translated>
      <Source section="22.1" pages="530" />
      <Translated
        vi={
          <p>
            Các thuật toán cho cùng một bài toán khác nhau theo ít nhất bốn trục: chúng tốn bao nhiêu bước (độ phức tạp), lộ ra
            bao nhiêu song song, tổng quát đến đâu, và chính xác hay ổn định số học đến đâu. Thường không thuật toán nào thắng
            trên cả bốn, nên bạn chọn phương án dung hòa tốt nhất cho phần cứng mình có. Sách minh họa ba sự đánh đổi như vậy
            bằng các chương bạn đã học, mỗi sự đánh đổi là một cặp trục khác nhau.
          </p>
        }
      >
        <p>
          Algorithms for one problem differ along at least four axes: how many steps they take (their complexity), how much
          parallelism they expose, how general they are, and how accurate or numerically stable they are. Usually none wins
          on all four, so you choose the best compromise for the hardware you have. The book shows three of these
          trade-offs with chapters you have done, each a different pair of axes.
        </p>
      </Translated>

      <h2><Translated vi="Ít bước hơn, hay ít phép cộng hơn">Fewer steps, or fewer additions</Translated></h2>
      <Translated
        vi={
          <p>
            Hai mạng scan của chương 11 là cặp đầu tiên. Kogge-Stone xong trong log₂N bước nhưng làm khoảng N·log₂N phép cộng.
            Brent-Kung làm ít hơn 2N phép cộng nhưng cần khoảng gấp đôi số bước. Số bước chỉ quyết định thời gian khi mọi phép
            cộng của một bước có thể chạy cùng lúc. Khi phần cứng có ít bộ cộng hơn số phép cộng của một bước, bước đó tốn nhiều
            vòng, và tổng số phép cộng bắt đầu có ý nghĩa.
          </p>
        }
      >
        <p>
          Chapter 11&rsquo;s two scan networks are the first pair. Kogge-Stone finishes in log₂N steps but does about
          N·log₂N additions. Brent-Kung does fewer than 2N additions but needs about twice as many steps. Steps decide the
          time only when every addition of a step can run at once. When the hardware has fewer adders than a step has
          additions, the step takes several rounds, and the total number of additions starts to count.
        </p>
      </Translated>
      <Source section="22.1" pages="530" figs={tr("ch. 11 §11.5 and §11.10", "ch. 11 §11.5 và §11.10")} />
      <Translated
        vi={
          <p>
            Sách thêm một cách làm dịu phần việc thừa: một thuật toán lai, hoặc kết hợp hai thuật toán song song, hoặc kết hợp
            một thuật toán song song với một thuật toán tuần tự rẻ hơn qua gộp việc cho luồng (thread coarsening). Scan ba pha
            của chương 11 thuộc loại thứ hai. Mỗi luồng scan tuần tự đoạn phần tử của riêng nó, một scan song song nối các tổng
            của từng đoạn, và mỗi luồng cộng thêm tổng của đoạn đứng trước.
          </p>
        }
      >
        <p>
          The book adds a way to soften the extra work: a hybrid, either two parallel algorithms combined or a parallel one
          combined with a cheaper sequential one through thread coarsening. Chapter 11&rsquo;s three-phase scan is the
          second kind. Each thread scans its own run of elements sequentially, one parallel scan joins the run totals, and
          each thread adds its predecessor&rsquo;s total.
        </p>
      </Translated>
      <Ask q={q['ch22/22.1/q-scan-units']} label={tr('Reason it out, then check it below', 'Suy luận trước, rồi kiểm tra ở bên dưới')} />
      <ScanChoice />
      <Translated
        vi={
          <p>
            Có ba điều cần tìm. Thứ nhất, Kogge-Stone cần bao nhiêu đơn vị mới thắng: với 1,024 phần tử, đếm phép cộng, nó chưa
            dẫn trước cho đến khi P đạt {KS_FROM_1024}. Thứ hai, cách tính trọn warp làm gì với N nhỏ, khi các lane rảnh của một
            warp dùng dở tốn ngang các lane bận. Thứ ba, scan có gộp việc: ở mọi P nó ngang mạng tốt hơn, nhanh hơn, hoặc chậm
            hơn một đơn vị thời gian. Nó chạy mỗi đơn vị một luồng và giao cho mỗi luồng một đoạn tuần tự, nên nó điều chỉnh
            khối lượng việc theo phần cứng thay vì gắn chặt với một mạng.
          </p>
        }
      >
        <p>
          Three things to find. First, how many units Kogge-Stone needs before it wins: with 1,024 elements, counting
          additions, it is not ahead until P reaches {KS_FROM_1024}. Second, what whole-warp charging does with a small N,
          where idle lanes of a partly used warp cost as much as busy ones. Third, the coarsened scan: at every P it is
          level with the better network, ahead of it, or one time unit behind. It runs one thread per unit and gives each
          thread a sequential run, so it adapts its work to the hardware instead of committing to one network.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mô hình của chương 11: mỗi phép cộng một đơn vị, một bước tốn ⌈charge / P⌉ đơn vị thời gian, các bước không bao giờ
              chồng lấp. Không có lưu lượng bộ nhớ và rào chắn (barrier) không tốn gì. P là lũy thừa của hai, vì scan có gộp việc
              theo cách đếm này cần số luồng là lũy thừa của hai.
            </p>
          }
        >
          <p>
            Chapter 11&rsquo;s model: one unit per addition, a step costs ⌈charge / P⌉ time units, steps never overlap. No
            memory traffic and no cost for barriers. P is a power of two, because the coarsened scan as counted needs a
            power-of-two thread count.
          </p>
        </Translated>
      </Simplification>

      <h2><Translated vi="Tổng quát, hay nhanh">General, or fast</Translated></h2>
      <Translated
        vi={
          <p>
            Sắp xếp cho ta cặp thứ hai. Odd-even sort làm nhiều việc nhất trong các cách sắp xếp của chương 14 nhưng song song
            hóa dễ nhất. Radix sort có thể thắng merge sort về độ phức tạp vì nó không bao giờ so sánh khóa, và nó song song hóa
            rất tốt, nhưng chỉ dùng được cho một số kiểu khóa. Merge sort so sánh, nên nó nhận mọi kiểu khóa có phép so sánh
            được định nghĩa rõ.
          </p>
        }
      >
        <p>
          Sorting gives the second pair. Odd-even sort does the most work of chapter 14&rsquo;s sorts but is the most
          trivial to parallelise. Radix sort can beat merge sort on complexity because it never compares keys, and it
          parallelises very well, but it works only for certain key types. Merge sort compares, so it takes any key type
          with a well-defined comparison.
        </p>
      </Translated>
      <Source section="22.1" pages="530" figs="ch. 14" />
      <Translated vi={<p>Khóa bạn có nhiều khả năng phải sắp xếp nhất là logits, và logits là float.</p>}>
        <p>The keys you are most likely to sort are logits, and logits are floats.</p>
      </Translated>
      <Ask q={q['ch22/22.1/q-sort-logits']} />

      <h2><Translated vi="Chính xác, hay nhanh">Exact, or fast</Translated></h2>
      <Translated
        vi={
          <p>
            Cặp thứ ba đến từ bản đồ tĩnh điện của chương 21. Phép cộng Coulomb trực tiếp cộng phần đóng góp của mọi nguyên tử
            tại mọi điểm lưới. Nó chính xác, nhưng khối lượng việc tăng theo bình phương thể tích, vì cả số nguyên tử lẫn số điểm
            lưới đều tăng theo thể tích, và với hệ lớn thì như vậy là quá nhiều ngay cả với một thiết bị song song quy mô lớn.
            Phép cộng có ngưỡng cắt chỉ cộng các nguyên tử nằm trong một khoảng cách cố định quanh mỗi điểm lưới. Nó hy sinh một
            chút độ chính xác để được tăng tốc lớn, và cả hai đều lộ ra rất nhiều song song.
          </p>
        }
      >
        <p>
          The third pair comes from chapter 21&rsquo;s electrostatic map. Direct Coulomb summation adds every atom&rsquo;s
          contribution at every grid point. It is exact, but its work grows with the square of the volume, because the
          atoms and the grid points both grow with it, and for large systems that is too much even for a massively
          parallel device. Cutoff summation adds only the atoms within a fixed distance of each grid point. It gives up a
          little accuracy for a large gain in speed, and both expose ample parallelism.
        </p>
      </Translated>
      <Source section="22.1" pages="530–531" />
      <Translated
        vi={
          <p>
            Vì sao có thể hy sinh độ chính xác là một sự thật về vật lý, không phải về tính toán. Theo các định luật mà những
            phép tính trên lưới này dựa vào, các hạt ở xa một điểm lưới đóng góp rất ít, và phần đóng góp của chúng có thể được
            xử lý gộp bằng một phương pháp riêng có độ phức tạp thấp hơn nhiều.
          </p>
        }
      >
        <p>
          Why the accuracy may be given up is a fact about physics, not about computing. Under the laws these grid
          calculations rest on, particles far from a grid point contribute very little, and their contributions can be
          handled together by a separate method of much lower complexity.
        </p>
      </Translated>

      <h3><Translated vi="Vì sao phiên bản song song khó hơn">Why the parallel version is harder</Translated></h3>
      <Translated
        vi={
          <p>
            Sự đánh đổi này cũng có trong mã tuần tự; mã song song khiến việc thu được lợi ích khó hơn. Vòng lặp tuần tự có ngưỡng
            cắt đơn giản lấy từng nguyên tử một và ghé các điểm lưới nằm trong ngưỡng cắt của nó, việc này dễ vì các điểm lưới
            nằm trong một mảng đánh chỉ số theo tọa độ. Chạy song song như thế, mỗi nguyên tử một luồng, thì các luồng scatter:
            nhiều luồng cùng cộng vào một điểm lưới.
          </p>
        }
      >
        <p>
          The trade exists in sequential code too; parallel code makes it harder to collect. The simple sequential cutoff
          loop takes one atom at a time and visits the grid points within the cutoff of it, which is easy because the grid
          points sit in an array indexed by coordinate. Run that in parallel, one thread per atom, and the threads scatter:
          several of them add into the same grid point.
        </p>
      </Translated>
      <Source section="22.1" pages="531" />
      <Translated
        vi={
          <p>
            Vì vậy phiên bản song song đảo ngược lại: mỗi điểm lưới một luồng, theo kiểu gather. Để cho một luồng biết nguyên tử
            nào ở gần mà không phải quét hết, các nguyên tử trước tiên được xếp vào các bin, tức các hộp trong không gian lưới.
            Vùng lân cận của một điểm lưới là tập các bin có thể chứa nguyên tử trong ngưỡng cắt của nó. Mọi luồng của một khối
            cùng nhau quét các nguyên tử trong vùng lân cận chung của chúng, và mỗi luồng kiểm tra mọi nguyên tử với ngưỡng cắt
            của riêng nó. Mục{' '}
            <a href={href('ch22', '22.2')}>22.2</a> đặt tên cho hai cách sắp xếp này: lấy nguyên tử làm trung tâm là lấy đầu vào
            làm trung tâm, lấy lưới làm trung tâm là lấy đầu ra làm trung tâm.
          </p>
        }
      >
        <p>
          So the parallel version turns it around: one thread per grid point, gathering. To tell a thread which atoms are
          near without scanning all of them, the atoms are first sorted into bins, boxes of grid space. A grid point&rsquo;s
          neighbourhood is the set of bins that can hold atoms within its cutoff. All the threads of a block scan the atoms
          of their combined neighbourhood together, and each tests every atom against its own cutoff. Section{' '}
          <a href={href('ch22', '22.2')}>22.2</a> names the two arrangements: atom-centric is input-centric, grid-centric is
          output-centric.
        </p>
      </Translated>

      <h3><Translated vi="Các bin không đều">Uneven bins</Translated></h3>
      <Translated
        vi={
          <p>
            Các bin không đầy đều nhau: có bin trống, có bin chật. Dành chỗ trong mọi bin cho số nguyên tử nhiều nhất mà một bin
            bất kỳ có thể chứa thì đắt đến mức không chấp nhận được. Cách sửa quen thuộc là một sức chứa thấp hơn hẳn trường hợp
            xấu nhất cộng với một danh sách tràn. Trong lúc xếp bin, một nguyên tử có bin đã đầy sẽ vào danh sách thay vì vào bin,
            và sau kernel các nguyên tử đó được xử lý để cộng phần đóng góp còn thiếu.
          </p>
        }
      >
        <p>
          Bins do not fill evenly: some are empty, some crowded. Room in every bin for the most atoms any bin could hold is
          prohibitively expensive. The well-known fix is a capacity well below the worst case and an overflow list. During
          binning, an atom whose bin is already full goes on the list instead, and after the kernel those atoms are
          processed to add their missing contributions.
        </p>
      </Translated>
      <Source section="22.1" pages="531" />
      <Translated
        vi={
          <p>
            Trang này của sách không nói danh sách tràn được xử lý ở đâu. Chương 21 thì có: host chạy một lượt tuần tự có ngưỡng
            cắt trên danh sách đó. Chương 21 cũng đệm mọi bin cho đủ sức chứa bằng các nguyên tử giả có điện tích bằng không, để
            mọi bin có cùng một kích thước và các lần nạp vẫn được hợp nhất; các nguyên tử giả tốn bộ nhớ, và chúng kéo dài lượt
            quét của những khối có bin chưa đầy. Với tỉ lệ tràn nhỏ, chẳng hạn dưới 3%, lượt chạy của host thường tốn ít thời gian
            hơn device.
          </p>
        }
      >
        <p>
          This page of the book does not say where the overflow list is processed. Chapter 21 does: the host runs a
          sequential cutoff pass over it. Chapter 21 also pads every bin to the capacity with dummy atoms of charge zero, so
          that all bins have one size and loads stay coalesced; the dummies cost memory, and they lengthen the scans of
          blocks whose bins are not full. With a small overflow share, under 3% for example, the host pass typically takes
          less time than the device.
        </p>
      </Translated>
      <Source section="21.5" pages="527" />
      <BinCapacity />
      <Translated
        vi={
          <p>
            Bắt đầu với một túi dày đặc, kích thước theo bin đầy nhất: phần lớn bộ nhớ và phần lớn số lần kiểm tra của device
            rơi vào nguyên tử giả. Kéo sức chứa về phía phần lớn của histogram. Bộ nhớ và thời gian của device giảm, danh sách
            tràn tăng chậm, và tổng thời gian giảm theo. Kéo tiếp thì lượt chạy tuần tự của host vượt kernel. Đó là định luật
            Amdahl đến sớm một mục: phần tuần tự, dù lúc đầu nhỏ đến đâu, rốt cuộc sẽ quyết định thời gian.
          </p>
        }
      >
        <p>
          Start with one dense pocket, sized for the fullest bin: most of the memory and most of the device&rsquo;s tests
          go to dummies. Drag the capacity toward the bulk of the histogram. Memory and device time fall, the overflow list
          grows slowly, and the total falls with them. Keep going and the host&rsquo;s serial pass overtakes the kernel.
          That is Amdahl&rsquo;s law arriving one section early: the serial part, however small it starts, ends up setting
          the time.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Hai chiều, vùng lân cận cố định 3 × 3, mỗi bin một khối. Mỗi ô được quét tốn một đơn vị dù nguyên tử có nằm trong
              ngưỡng cắt hay không, nên không có phân kỳ luồng (divergence). S là một tham số tự do, không phải số liệu của một
              thiết bị. Không có lưu lượng bộ nhớ, không có lượt xếp bin, không có sao chép giữa host và device.
            </p>
          }
        >
          <p>
            Two dimensions, a fixed 3 × 3 neighbourhood, one block per bin. Every scanned slot costs one unit whether or not
            the atom is inside the cutoff, so there is no divergence. S is a free parameter, not a device figure. No memory
            traffic, no binning pass, no copies between host and device.
          </p>
        </Translated>
      </Simplification>
      <Translated
        vi={
          <p>
            Sức chứa tốt nhất nằm ở đâu? Thêm một ô cho mỗi bin là thêm một ô vào mọi bin, nên mọi khối đều quét nhiều hơn; nó
            chỉ bớt nguyên tử cho host ở những bin vẫn còn tràn. Vì thế nó có lợi chừng nào còn hơn khoảng một bin trên S bị
            tràn, và nút &ldquo;tổng thời gian thấp nhất&rdquo; dừng đúng ở chỗ điều đó không còn đúng.
          </p>
        }
      >
        <p>
          Where is the best capacity? One more slot per bin adds a slot to every bin, so every block scans more; it takes
          atoms off the host only in the bins that still overflow. So it pays while more than about one bin in S still
          overflows, and the &ldquo;lowest total time&rdquo; button lands where that stops being true.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>Quy tắc này là của khóa học này, suy ra từ mô hình khối lượng việc của trình khám phá; sách chỉ đưa ví dụ 3%.</p>
          }
        >
          <p>This rule is ours, derived from the explorer&rsquo;s work model; the book gives only the 3% example.</p>
        </Translated>
      </Beyond>
      <Ask q={q['ch22/22.1/q-overflow']} />

      <h2><Translated vi="Chọn bằng đo đạc">Selection by measurement</Translated></h2>
      <Translated
        vi={
          <p>
            Cùng lập luận đó chọn kernel trong mã đang chạy thực tế. llama.cpp giữ hai thuật toán cho một phép toán: một ma trận
            trọng số nhân với các token của một batch. Đến 8 token nó chạy kernel nhân ma trận với vector; trên mức đó, một kernel
            nhân ma trận với ma trận. Ngưỡng này không được suy ra. Nó bị hạ xuống cho một số kiểu trọng số đã lượng tử hóa trên
            những GPU cụ thể, kèm chú thích nêu tên GPU đã dùng để tinh chỉnh, và có các nhánh riêng cho AMD. Một lý do nằm ngay
            trong mã nguồn: với một số định dạng, kernel nhân ma trận với vector giải mã lại trọng số cho mỗi token, nên kernel ma
            trận thắng sớm hơn. Chọn theo một điểm giao đo được chính là phương án dung hòa tốt nhất cho phần cứng đang có của
            sách, được viết vào một hàm dispatch.
          </p>
        }
      >
        <p>
          The same reasoning picks kernels in shipping code. llama.cpp keeps two algorithms for one operation, a weight
          matrix times the tokens of a batch. Up to 8 tokens it runs matrix-vector kernels; above that, a matrix-matrix
          kernel. The threshold is not derived. It is lowered for some quantised weight types on particular GPUs, with
          comments naming the GPU it was tuned on, and it has separate branches for AMD. One reason is in the source itself:
          for some formats the matrix-vector kernel redoes the decoding of the weights for every token, so the matrix
          kernel wins sooner. Choosing by a measured crossover is the book&rsquo;s best compromise for the given hardware,
          written into a dispatch function.
        </p>
      </Translated>
      <Today checked={tr("2026-11-20 against llama.cpp's master branch (ch22-facts D6)", '2026-11-20, đối chiếu nhánh master của llama.cpp (ch22-facts D6)')}>
        <Translated
          vi={
            <p>
              CUDA và HIP dùng chung{' '}
              <a href="https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda" target="_blank" rel="noreferrer">
                <code>ggml-cuda/</code>
              </a>
              : <code>mmvf</code> và <code>mmvq</code> cho tối đa 8 token, với các ngoại lệ ghi &ldquo;tuned on RTX
              4090&rdquo;. Metal,{' '}
              <a href="https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-metal" target="_blank" rel="noreferrer">
                <code>ggml-metal/</code>
              </a>
              : <code>mul_mv_ext</code> cho 2 đến 8 token, rồi <code>mul_mm</code>. Số dòng có thể thay đổi.
            </p>
          }
        >
          <p>
            CUDA and HIP share{' '}
            <a href="https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda" target="_blank" rel="noreferrer">
              <code>ggml-cuda/</code>
            </a>
            : <code>mmvf</code> and <code>mmvq</code> up to 8 tokens, with exceptions marked &ldquo;tuned on RTX
            4090&rdquo;. Metal,{' '}
            <a href="https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-metal" target="_blank" rel="noreferrer">
              <code>ggml-metal/</code>
            </a>
            : <code>mul_mv_ext</code> for 2 to 8 tokens, then <code>mul_mm</code>. Line numbers move.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Lab 22.1 trên <a href={href('ch22', 'labs')}>trang lab</a> cắt một phép nhân ma trận với vector lúc decode theo hai
            cách; phiên bản gather của nó là một kernel thuộc loại thứ nhất.
          </p>
        }
      >
        <p>
          Lab 22.1 on the <a href={href('ch22', 'labs')}>labs page</a> cuts a decode-time matrix-vector product two ways;
          its gather version is a kernel of the first kind.
        </p>
      </Translated>
      <Ask q={q['ch22/22.1/q-cutoff-axis']} />
      <Ask q={q['ch22/22.1/q-two-kernels']} />
    </>
  );
}
