import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  FILTER_WIDTHS, basicIntensity, ceiling, constIntensity, idealIntensity, multipleOfCeiling, regime,
  smallestComputeBoundWidth, threshold, type Dims, type Hardware,
} from '../../sim/convai';

// Section 7.3: how many operations per byte a convolution could have, how many the
// basic kernel asks for, and why measurements land above the model. Every number on
// the page comes from sim/convai.ts; device numbers come from the reader's reports.

const LAB71 = 'ch07/lab1_constconv';

// ---------------------------------------------------------------- hardware
interface Device extends Hardware {
  key: string;
  label: string;
  note: string;
}
// The book's worked example (ch. 5, p. 94; used again on p. 167). Only ever offered as that.
const BOOK: Device = {
  key: 'book',
  get label() {
    return tr("The book's example (Hopper H100)", 'Ví dụ của sách (Hopper H100)');
  },
  peak: 66900,
  bw: 3350,
  get note() {
    return tr('The figures the book quotes in chapter 5 (p. 94): 66.9 TFLOPS single precision, 3.35 TB/s.', 'Các số liệu sách trích ở chương 5 (p. 94): 66.9 TFLOPS độ chính xác đơn, 3.35 TB/s.');
  },
};

function lab51(reports: Reports, b: string): Hardware | null {
  const r = benchRows(reports[b], 'ch05/lab1_roofline').find((x) => x.name === 'roofline');
  const peak = num(r?.compute_gflops);
  const bw = num(r?.bandwidth_gb_per_s);
  return peak && bw ? { peak, bw } : null;
}

function measuredDevices(reports: Reports): Device[] {
  return realBackends(reports).flatMap((b) => {
    const hw = lab51(reports, b);
    if (!hw) return [];
    return [
      {
        key: b,
        get label() {
          return tr(`Yours: ${String(reports[b].device.name ?? b)} (${b})`, `Của bạn: ${String(reports[b].device.name ?? b)} (${b})`);
        },
        ...hw,
        get note() {
          return tr(`Measured by lab 5.1 on ${reports[b].generated.slice(0, 10)}: two simple probe kernels, not the vendor's peak figures.`, `Đo bằng lab 5.1 ngày ${reports[b].generated.slice(0, 10)}: hai kernel thăm dò đơn giản, không phải số liệu đỉnh của nhà sản xuất.`);
        },
      },
    ];
  });
}

const BYTES = [
  { b: 4, label: '4 (float32)' }, // i18n-ignore: data type names
  { b: 2, label: '2 (fp16, bf16)' }, // i18n-ignore: data type names
  { b: 1, label: '1 (int8)' }, // i18n-ignore: data type names
] as const;

const filterName = (m: number, dims: Dims) => (dims === 2 ? `${m}×${m}` : tr(`${m}-tap`, `${m} tap`));

// ---------------------------------------------------------------- the chart
function IntensityChart({ m, bytes, dims, thr }: { m: number; bytes: number; dims: Dims; thr: number }) {
  const W = 760;
  const H = 330;
  const pad = { l: 62, r: 132, t: 14, b: 48 };
  const yMin = 0.1;
  const yMax = Math.max(400, thr * 2);
  const X = (w: number) => pad.l + ((w - 1) / 14) * (W - pad.l - pad.r);
  const Y = (v: number) =>
    H - pad.b - ((Math.log10(Math.max(v, yMin)) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * (H - pad.t - pad.b);
  const ideal = FILTER_WIDTHS.map((w) => ({ w, v: idealIntensity(w, bytes, dims) }));
  const basic = basicIntensity(bytes);
  const cst = constIntensity(bytes);
  const yt: number[] = [];
  for (let e = Math.ceil(Math.log10(yMin)); e <= Math.floor(Math.log10(yMax)); e++) yt.push(10 ** e);
  const cur = idealIntensity(m, bytes, dims);
  const right = W - pad.r;
  // Labels at the right edge, nudged apart when two lines are close.
  const labels = [
    { key: 'thr', y: Y(thr), text: tr(`threshold ${fmt(thr)}`, `ngưỡng ${fmt(thr)}`), color: 'var(--ink)' },
    { key: 'const', y: Y(cst), text: tr(`constant ${fmt(cst)}`, `hằng ${fmt(cst)}`), color: 'var(--const)' },
    { key: 'basic', y: Y(basic), text: tr(`basic ${fmt(basic)}`, `cơ bản ${fmt(basic)}`), color: 'var(--global)' },
    { key: 'ideal', y: Y(ideal[ideal.length - 1].v), text: tr('ideal', 'lý tưởng'), color: 'var(--ink)' },
  ].sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < 14) labels[i].y = labels[i - 1].y + 14;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Intensity against filter width', 'Cường độ tính toán theo chiều rộng bộ lọc')} style={{ width: '100%', maxWidth: W, minWidth: 520, display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={right - pad.l} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      <rect x={pad.l} y={Y(thr)} width={right - pad.l} height={Math.max(0, H - pad.b - Y(thr))} fill="var(--global-wash)" />
      {yt.map((v) => (
        <g key={'y' + v}>
          <line x1={pad.l} x2={right} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 7} y={Y(v) + 4} fontSize="12" textAnchor="end">{v}</text>
        </g>
      ))}
      {FILTER_WIDTHS.map((w) => (
        <g key={'x' + w}>
          <line x1={X(w)} x2={X(w)} y1={H - pad.b} y2={H - pad.b + 4} stroke="var(--ink-3)" />
          <text x={X(w)} y={H - pad.b + 18} fontSize="12" textAnchor="middle" fontWeight={w === m ? 700 : 400}>{w}</text>
        </g>
      ))}
      <text x={(pad.l + right) / 2} y={H - 8} fontSize="12.5" textAnchor="middle">
        <Translated vi={<>chiều rộng bộ lọc m {dims === 2 ? '(một bộ lọc m × m)' : '(m tap)'}</>}>
          filter width m {dims === 2 ? '(an m × m filter)' : '(m taps)'}
        </Translated>
      </text>
      <text x={16} y={(H - pad.b + pad.t) / 2} fontSize="12.5" textAnchor="middle" transform={`rotate(-90 16 ${(H - pad.b + pad.t) / 2})`}>
        <Translated vi="FLOP trên mỗi byte (thang log)">FLOP per byte (log scale)</Translated>
      </text>
      {/* the device's threshold */}
      <line x1={pad.l} x2={right} y1={Y(thr)} y2={Y(thr)} stroke="var(--ink)" strokeDasharray="5 4" strokeWidth="1.4" />
      <text x={pad.l + 8} y={Y(thr) + 16} fontSize="12" style={{ fill: 'var(--global)' }}><Translated vi="dưới ngưỡng: bị giới hạn bởi bộ nhớ">memory-bound below the threshold</Translated></text>
      <text x={pad.l + 8} y={Y(thr) - 7} fontSize="12" style={{ fill: 'var(--reg)' }}><Translated vi="trên ngưỡng: bị giới hạn bởi tính toán">compute-bound above</Translated></text>
      {/* the two kernels as written: flat */}
      <line x1={pad.l} x2={right} y1={Y(basic)} y2={Y(basic)} stroke="var(--global)" strokeWidth="2.2" />
      <line x1={pad.l} x2={right} y1={Y(cst)} y2={Y(cst)} stroke="var(--const)" strokeWidth="2.2" />
      {/* the ideal: grows with the filter */}
      <polyline points={ideal.map((p) => `${X(p.w)},${Y(p.v)}`).join(' ')} fill="none" stroke="var(--ink)" strokeWidth="2.2" />
      {ideal.map((p) => (
        <circle key={p.w} cx={X(p.w)} cy={Y(p.v)} r={p.w === m ? 6.5 : 4} fill={p.w === m ? 'var(--ink)' : 'var(--page)'} stroke="var(--ink)" strokeWidth="1.8" />
      ))}
      {labels.map((l) => (
        <text key={l.key} x={right + 8} y={l.y + 4} fontSize="12" style={{ fill: l.color }} data-label={l.key}
          data-threshold={l.key === 'thr' ? fmt(thr) : undefined}>{l.text}</text>
      ))}
      {/* the gap at the chosen width */}
      <line x1={X(m)} x2={X(m)} y1={Y(cur)} y2={Y(basic)} stroke="var(--ink-2)" strokeDasharray="2 3" />
      <text x={X(m) + (m >= 13 ? -9 : 9)} y={(Y(cur) + Y(basic)) / 2 + 4} fontSize="12.5" fontWeight="600" textAnchor={m >= 13 ? 'end' : 'start'} data-gap>
        <Translated vi={<>khoảng cách ×{fmt(cur / basic)}</>}>gap ×{fmt(cur / basic)}</Translated>
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------- explorer: ideal against what the kernel asks
function IdealExplorer() {
  const { reports } = useReports();
  const options = useMemo(() => [BOOK, ...measuredDevices(reports)], [reports]);
  const [key, setKey] = useState('book');
  const [custom, setCustom] = useState({ peak: 1000, bw: 100 });
  const [wi, setWi] = useState(2); // index into FILTER_WIDTHS: m = 5
  const [bytes, setBytes] = useState<number>(4);
  const [dims, setDims] = useState<Dims>(2);
  const hw: Device =
    key === 'custom'
      ? { key: 'custom', label: tr('Custom', 'Tùy chỉnh'), ...custom, note: tr('Numbers you typed.', 'Các số bạn đã gõ.') }
      : (options.find((o) => o.key === key) ?? BOOK);
  const m = FILTER_WIDTHS[wi];
  const thr = threshold(hw);
  const rows = [
    { id: 'basic', label: tr('Basic kernel (Fig. 7.7)', 'Kernel cơ bản (Fig. 7.7)'), ai: basicIntensity(bytes) },
    { id: 'const', label: tr('Filter in constant memory (Fig. 7.9, next section)', 'Bộ lọc trong bộ nhớ hằng (Fig. 7.9, mục kế tiếp)'), ai: constIntensity(bytes) },
    { id: 'ideal', label: tr(`Ideal for a ${filterName(m, dims)} filter`, `Lý tưởng cho một bộ lọc ${filterName(m, dims)}`), ai: idealIntensity(m, bytes, dims) },
  ];
  const first = smallestComputeBoundWidth(thr, bytes, dims);
  return (
    <Bench
      title={tr('Ideal against what the kernel asks', 'Lý tưởng so với những gì kernel đòi hỏi')}
      task={tr("Find the smallest filter that could ever be compute-bound on the book's example device, then halve the bytes.", 'Tìm bộ lọc nhỏ nhất có thể bị giới hạn bởi tính toán trên thiết bị ví dụ của sách, rồi giảm số byte đi một nửa.')}
      foot={
        <Translated vi={<>Mức lý tưởng đếm mỗi phần tử đầu vào và đầu ra một lần và bỏ qua số byte của chính bộ lọc; với một ảnh rộng 1024, con số chính xác ở thiết lập này là {fmt(idealIntensity(m, bytes, dims, 1024), 6)}. Trần là min(peak, bandwidth × intensity). {hw.note}</>}>
        <>
          The ideal counts each input and output element once and drops the filter's own bytes; for a 1024-wide image
          the exact figure at this setting is {fmt(idealIntensity(m, bytes, dims, 1024), 6)}. Ceilings are min(peak,
          bandwidth × intensity). {hw.note}
        </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          <Translated vi={<>Chiều rộng bộ lọc m = {m} (bán kính {(m - 1) / 2})</>}>Filter width m = {m} (radius {(m - 1) / 2})</Translated>
          <input type="range" min="0" max={FILTER_WIDTHS.length - 1} value={wi} onChange={(e) => setWi(+e.target.value)} aria-label={tr('filter width', 'chiều rộng bộ lọc')} />
        </label>
        <label className="field">
          <Translated vi="Số byte mỗi phần tử">Bytes per element</Translated>
          <span className="seg" data-control="bytes">
            {BYTES.map((o) => (
              <button key={o.b} aria-pressed={bytes === o.b} onClick={() => setBytes(o.b)}>{o.label}</button>
            ))}
          </span>
        </label>
        <label className="field">
          <Translated vi="Số chiều">Dimensions</Translated>
          <span className="seg" data-control="dims">
            {([2, 1] as Dims[]).map((d) => (
              <button key={d} aria-pressed={dims === d} onClick={() => setDims(d)}>{d === 2 ? tr('2D image', 'ảnh 2D') : tr('1D signal', 'tín hiệu 1D')}</button>
            ))}
          </span>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          <Translated vi="Phần cứng">Hardware</Translated>
          <select value={key} onChange={(e) => setKey(e.target.value)} aria-label={tr('hardware', 'phần cứng')}>
            {options.map((o) => (
              <option key={o.key} value={o.key}>{o.label}</option>
            ))}
            <option value="custom">{tr('Custom numbers', 'Số tùy chỉnh')}</option>
          </select>
        </label>
        {key === 'custom' ? (
          <>
            <label className="field">
              <Translated vi="Thông lượng đỉnh (GFLOP/s)">Peak throughput (GFLOP/s)</Translated>
              <input type="number" min="1" value={custom.peak} onChange={(e) => setCustom({ ...custom, peak: Math.max(1, +e.target.value) })} />
            </label>
            <label className="field">
              <Translated vi="Băng thông bộ nhớ (GB/s)">Memory bandwidth (GB/s)</Translated>
              <input type="number" min="1" value={custom.bw} onChange={(e) => setCustom({ ...custom, bw: Math.max(1, +e.target.value) })} />
            </label>
          </>
        ) : null}
      </div>
      <IntensityChart m={m} bytes={bytes} dims={dims} thr={thr} />
      <div className="cols" style={{ marginTop: '0.9rem', gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 2fr)' }}>
        <div className="panel" data-panel="ceilings">
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr>
                <th><Translated vi="Kernel">Kernel</Translated></th>
                <th className="num">FLOP/B</th>
                <th><Translated vi="Chế độ">Regime</Translated></th>
                <th className="num"><Translated vi="Trần, GFLOP/s">Ceiling, GFLOP/s</Translated></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const g = regime(r.ai, thr);
                return (
                  <tr key={r.id} data-row={r.id} data-ai={r.ai} data-regime={g}>
                    <td>{r.label}</td>
                    <td className="num">{fmt(r.ai)}</td>
                    <td>
                      <span className={'chip ' + (g === 'memory-bound' ? 'global' : 'reg')}>{tr(g, g === 'memory-bound' ? 'bị giới hạn bởi bộ nhớ' : 'bị giới hạn bởi tính toán')}</span>
                    </td>
                    <td className="num">{fmt(ceiling(r.ai, hw))}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="muted" style={{ margin: '0.7rem 0 0' }}>
            <Translated vi={<>Đỉnh {fmt(hw.peak)} GFLOP/s, băng thông {fmt(hw.bw)} GB/s, ngưỡng {fmt(thr)} FLOP/B.</>}>
              Peak {fmt(hw.peak)} GFLOP/s, bandwidth {fmt(hw.bw)} GB/s, threshold {fmt(thr)} FLOP/B.
            </Translated>
          </p>
        </div>
        <div className="panel">
          <p style={{ margin: 0 }} data-first={first}>
            <Translated vi={<>{dims === 2 ? 'Chiều rộng bộ lọc lẻ' : 'Số tap lẻ'} nhỏ nhất có mức lý tưởng chạm ngưỡng này với{' '}{bytes} byte mỗi phần tử: <b>{first}</b>{first > FILTER_WIDTHS[FILTER_WIDTHS.length - 1] ? ', nằm ngoài biểu đồ này' : ''}. Dưới mức đó, không kernel nào có thể bị giới hạn bởi tính toán ở đây.</>}>
            Smallest {dims === 2 ? 'odd filter width' : 'odd number of taps'} whose ideal reaches this threshold at{' '}
            {bytes} byte{bytes > 1 ? 's' : ''} per element: <b>{first}</b>
            {first > FILTER_WIDTHS[FILTER_WIDTHS.length - 1] ? ', beyond this chart' : ''}. Below it, no kernel can be
            compute-bound here.
            </Translated>
          </p>
          <p style={{ margin: '0.7rem 0 0' }}>
            <Translated vi={<>Mức lý tưởng gấp {fmt(idealIntensity(m, bytes, dims) / basicIntensity(bytes))} lần những gì kernel cơ bản đòi hỏi: {dims === 2 ? 'm²' : 'm'}, số đầu ra mà mỗi phần tử đầu vào tham gia.</>}>
            The ideal is {fmt(idealIntensity(m, bytes, dims) / basicIntensity(bytes))} times what the basic kernel asks
            for: {dims === 2 ? 'm²' : 'm'}, the number of outputs each input element takes part in.
            </Translated>
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- your kernels against their ceilings
interface ConvRow {
  radius: number;
  m: number;
  gflops: number;
  ai: number;
}
function convRows(reports: Reports, b: string, name: 'basic' | 'const'): ConvRow[] {
  const fallback = name === 'basic' ? basicIntensity() : constIntensity();
  return benchRows(reports[b], LAB71)
    .filter((r) => r.name === name)
    .flatMap((r) => {
      const radius = num(r.radius);
      const gflops = num(r.gflops);
      if (radius == null || gflops == null) return [];
      return [{ radius, m: num(r.m) ?? 2 * radius + 1, gflops, ai: num(r.model_intensity) ?? fallback }];
    })
    .sort((a, b2) => a.radius - b2.radius);
}

// Above its model ceiling is not a failure (caches served the reuse), so no colour either way.
function Multiple({ x }: { x: number }) {
  const above = x > 1.02;
  return (
    <span className="chip" data-multiple={x.toFixed(3)}>
      <Translated vi={<>{fmt(x)}× {above ? 'vượt' : 'so với'} trần của nó</>}>{fmt(x)}× {above ? 'above' : 'of'} its ceiling</Translated>
    </span>
  );
}

/** Lab 5.1's naive matmul against the ceiling its 0.25 FLOP/B gives it, on each device that has run it. */
function Lab51Naive() {
  const { reports, loading } = useReports();
  const rows = realBackends(reports)
    .filter((b) => b !== 'race')
    .flatMap((b) => {
      const r = benchRows(reports[b], 'ch05/lab1_roofline').find((x) => x.name === 'roofline');
      const g = num(r?.naive_gflops);
      const c = num(r?.naive_model_ceiling_gflops);
      return g && c ? [{ b, device: String(reports[b].device.name ?? b), x: g / c }] : [];
    });
  if (loading || !rows.length) return <span data-lab51-naive=""><Translated vi="lab 5.1 đo đúng điều này trên thiết bị của bạn">lab 5.1 measures the same thing on your device</Translated></span>;
  return (
    <span data-lab51-naive={rows.map((r) => r.x.toFixed(3)).join(',')}>
      <Translated vi={<>trên thiết bị của bạn, kernel ngây thơ của lab 5.1 chạy được{' '}{rows.map((r) => `${fmt(r.x, 2)} lần mức trần mà 0.25 FLOP/B cho nó trên ${r.device} (${r.b})`).join(', và ')}</>}>
        on your device, lab 5.1&rsquo;s naive kernel ran{' '}
        {rows.map((r) => `${fmt(r.x, 2)} times the ceiling its 0.25 FLOP/B gave it on ${r.device} (${r.b})`).join(', and ')}
      </Translated>
    </span>
  );
}

/** Constant over basic, as measured, beside the model's ratio of the two ceilings. */
function SpeedupLine({ basic, cst }: { basic: ConvRow[]; cst: ConvRow[] }) {
  const ratios = basic.flatMap((a) => {
    const c = cst.find((x) => x.radius === a.radius);
    return c ? [c.gflops / a.gflops] : [];
  });
  if (!ratios.length) return null;
  const lo = Math.min(...ratios);
  const hi = Math.max(...ratios);
  return (
    <p style={{ margin: '0 0 0.5rem' }} data-speedup={`${lo.toFixed(3)}-${hi.toFixed(3)}`}>
      <Translated vi={<>Kernel dùng bộ nhớ hằng chạy nhanh gấp {fmt(lo)}{hi - lo > 0.005 ? <> đến {fmt(hi)}</> : null} lần kernel cơ bản ở đây; tỷ lệ giữa hai mức trần theo mô hình của chúng là 2. Một tỷ lệ thấp hẳn dưới 2 gợi ý rằng bộ nhớ đệm đã phục vụ phần lớn các lần đọc bộ lọc của kernel cơ bản, nên bộ nhớ hằng chẳng còn mấy để tiết kiệm.</>}>
      The constant-memory kernel ran {fmt(lo)}
      {hi - lo > 0.005 ? <> to {fmt(hi)}</> : null} times as fast as the basic one here; the ratio of their model
      ceilings is 2. A ratio well under 2 suggests the caches were already serving most of the basic kernel&rsquo;s
      filter reads, which left constant memory little to save.
      </Translated>
    </p>
  );
}

function YourKernels() {
  const { reports, loading } = useReports();
  const backends = realBackends(reports).filter((b) => convRows(reports, b, 'basic').length || convRows(reports, b, 'const').length);
  return (
    <Bench
      title={tr('Your kernels against their ceilings', 'Kernel của bạn so với mức trần của chúng')}
      task={tr('Read only. Where lab 7.1 has run, each measurement is set against the ceiling its model intensity gives it on the same device.', 'Chỉ để đọc. Ở nơi lab 7.1 đã chạy, mỗi số đo được đặt cạnh mức trần mà cường độ tính toán theo mô hình cho nó trên cùng thiết bị.')}
      foot={
        <Translated vi={<>Trần = min(peak, bandwidth × model intensity), với đỉnh và băng thông lấy từ lab 5.1 trên cùng backend. GFLOP/s đếm 2·m² phép toán mỗi đầu ra, tính cả tap rơi vào ô ma, như sách làm. Lời giải tham chiếu (solution) của lab 7.1; kernel của chính bạn nằm ở <a href={href('ch07', 'labs')}>trang lab</a>.</>}>
        <>
          Ceiling = min(peak, bandwidth × model intensity), with peak and bandwidth from lab 5.1 on the same backend.
          GFLOP/s counts 2·m² operations per output, ghost taps included, as the book does. Lab 7.1's reference
          solution; your own kernels are on the <a href={href('ch07', 'labs')}>labs page</a>.
        </>
        </Translated>
      }
    >
      {loading ? (
        <p className="muted" style={{ margin: 0 }}><Translated vi="Đang đọc các báo cáo lab của máy này.">Reading this machine's lab reports.</Translated></p>
      ) : !backends.length ? (
        <p style={{ margin: 0 }} data-empty>
          <Translated vi={<>Chưa có số đo nào. Hãy chạy <code>python3 labs/pmpp.py run 7.1 --solution --bench</code> (và lab 5.1 với{' '}<code>--bench</code> nếu bạn chưa chạy), rồi tải lại trang này.</>}>
          No measurement yet. Run <code>python3 labs/pmpp.py run 7.1 --solution --bench</code> (and lab 5.1 with{' '}
          <code>--bench</code> if you have not), then reload this page.
          </Translated>
        </p>
      ) : (
        backends.map((b) => {
          const hw = lab51(reports, b);
          const basic = convRows(reports, b, 'basic');
          const cst = convRows(reports, b, 'const');
          const radii = [...new Set([...basic, ...cst].map((r) => r.radius))].sort((x, y) => x - y);
          return (
            <div key={b} className="panel" data-backend={b} style={{ marginBottom: '0.8rem' }}>
              <p style={{ margin: '0 0 0.4rem' }}>
                <b>{String(reports[b].device.name ?? b)}</b> ({b})
                {hw ? (
                  <span className="muted"> · lab 5.1: {fmt(hw.bw)} GB/s, {fmt(hw.peak)} GFLOP/s{/* i18n-ignore: lab name and units */}</span>
                ) : (
                  <span className="muted"><Translated vi=" · chưa có băng thông từ lab 5.1 trên backend này: hãy chạy lab 5.1 với --bench để có các mức trần"> · no lab 5.1 bandwidth on this backend: run lab 5.1 with --bench to get the ceilings</Translated></span>
                )}
              </p>
              <SpeedupLine basic={basic} cst={cst} />
              <div style={{ overflowX: 'auto' }}>
                <table className="data" style={{ margin: 0 }}>
                  <thead>
                    <tr>
                      <th><Translated vi="Bộ lọc">Filter</Translated></th>
                      <th className="num"><Translated vi="Cơ bản, GFLOP/s">Basic, GFLOP/s</Translated></th>
                      <th><Translated vi="so với 0.25 FLOP/B">against 0.25 FLOP/B</Translated></th>
                      <th className="num"><Translated vi="Hằng, GFLOP/s">Constant, GFLOP/s</Translated></th>
                      <th><Translated vi="so với 0.5 FLOP/B">against 0.5 FLOP/B</Translated></th>
                    </tr>
                  </thead>
                  <tbody>
                    {radii.map((rad) => {
                      const a = basic.find((x) => x.radius === rad);
                      const c = cst.find((x) => x.radius === rad);
                      return (
                        <tr key={rad} data-radius={rad}>
                          <td>{2 * rad + 1}×{2 * rad + 1}</td>
                          <td className="num">{a ? fmt(a.gflops) : 'n/a'}</td>
                          <td>{a && hw ? <Multiple x={multipleOfCeiling(a.gflops, a.ai, hw)} /> : '·'}</td>
                          <td className="num">{c ? fmt(c.gflops) : 'n/a'}</td>
                          <td>{c && hw ? <Multiple x={multipleOfCeiling(c.gflops, c.ai, hw)} /> : '·'}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          );
        })
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS73: Record<string, Question> = {
  'ch07/7.3/q-regime': {
    kind: 'choice',
    id: 'ch07/7.3/q-regime',
    prompt: (
      <Translated vi={<>Trên thiết bị ví dụ của sách (ngưỡng 20 FLOP/B), lấy một kernel hoàn hảo: mỗi đầu vào đọc một lần, mỗi đầu ra ghi một lần, số float 4 byte. Với những bộ lọc vuông nào nó có thể bị giới hạn bởi tính toán?</>}>
<>
        On the book&rsquo;s example device (threshold 20 FLOP/B), take a perfect kernel: every input read once, every
        output written once, 4-byte floats. For which square filters could it be compute-bound?
      </>
</Translated>
    ),
    options: [
      { label: <Translated vi="Mọi kích thước, nếu kernel đủ tốt.">{'Every size, if the kernel is good enough.'}</Translated>, why: <Translated vi="Một kernel hoàn hảo đạt tới mức lý tưởng và không hơn. Ở 3×3 đó là 2.25 FLOP/B, một phần chín của ngưỡng.">{'A perfect kernel reaches the ideal and no further. At 3×3 that is 2.25 FLOP/B, a ninth of the threshold.'}</Translated> },
      { label: <Translated vi="7×7 trở lên.">{'7×7 and larger.'}</Translated>, why: <Translated vi="7×7 cho 49/4 = 12.25, vẫn dưới 20.">{'7×7 gives 49/4 = 12.25, still below 20.'}</Translated> },
      { label: <Translated vi="9×9 trở lên.">{'9×9 and larger.'}</Translated>, correct: true },
      { label: <Translated vi="11×11 trở lên.">{'11×11 and larger.'}</Translated>, why: <Translated vi="Sách gọi 11×11 là bị giới hạn bởi tính toán, nhưng 9×9 đã vượt ngưỡng: 81/4 = 20.25.">{'The book names 11×11 as compute-bound, but 9×9 already clears the threshold: 81/4 = 20.25.'}</Translated> },
      { label: <Translated vi="Không bộ lọc nào: tích chập luôn bị giới hạn bởi bộ nhớ.">{'None: convolution is always memory-bound.'}</Translated>, why: <Translated vi="Mức lý tưởng tăng theo diện tích bộ lọc, nên một bộ lọc đủ lớn sẽ vượt bất kỳ ngưỡng cố định nào.">{'The ideal grows with the filter\'s area, so a large enough filter crosses any fixed threshold.'}</Translated> },
    ],
    explain: (
      <Translated vi={<p>Mức lý tưởng là m²/4: 2.25, 6.25 và 12.25 cho 3, 5 và 7; 20.25 và 30.25 cho 9 và 11. Chỉ từ 9×9 mức lý tưởng mới chạm 20, và chỉ vừa đủ. Cũng những con số này là cột Ideal của Fig. 7.14, được tô màu theo cùng cách. Một kernel thật có tiến gần mức lý tưởng hay không là chủ đề của mục 7.5.</p>}>
<p>
        The ideal is m²/4: 2.25, 6.25 and 12.25 for 3, 5 and 7; 20.25 and 30.25 for 9 and 11. Only from 9×9 does the
        ideal reach 20, and only just. The same numbers are the Ideal column of Fig. 7.14, which is shaded the same
        way. Whether a real kernel gets near its ideal is the subject of section 7.5.
      </p>
</Translated>
    ),
    source: 'ch. 7 §7.3, p. 167; Fig. 7.14, p. 177; notes/today/ch07-facts.md A7', // i18n-ignore: citation
  },
  'ch07/7.3/q-store': {
    kind: 'choice',
    id: 'ch07/7.3/q-store',
    prompt: (
      <Translated vi={<>Vì sao sách có thể bỏ qua lần ghi <code>P</code> khi cho kernel cơ bản 0.25 FLOP/B?</>}>
        <>Why may the book leave the store of <code>P</code> out when it gives the basic kernel 0.25 FLOP/B?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Mỗi luồng ghi 4 byte một lần nhưng nạp 8 byte mỗi tap, nên lần ghi hầu như không làm thay đổi tỷ lệ.">{'Each thread stores 4 bytes once but loads 8 bytes per tap, so the store barely moves the ratio.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Lần ghi viết vào bộ nhớ chứ không đọc, và cường độ tính toán chỉ đếm số byte kernel đọc.">{'Stores write memory rather than read it, and the intensity counts only the bytes a kernel reads.'}</Translated>,
        why: <Translated vi="Chúng cũng là byte được di chuyển như mọi byte khác. Mục 7.5 có đếm chúng (p. 176).">{'They are bytes moved like any other. Section 7.5 counts them (p. 176).'}</Translated>,
      },
      {
        label: <Translated vi="Lần ghi rơi vào cache L2 và nằm đó tới khi kernel kết thúc, nên nó không bao giờ tốn DRAM.">{'The store lands in the L2 cache and stays there until the kernel ends, so it never costs DRAM.'}</Translated>,
        why: <Translated vi="Mô hình đếm những gì kernel đòi hỏi ở bộ nhớ và bỏ qua bộ nhớ đệm cho cả lần nạp lẫn lần ghi. Bộ nhớ đệm là lý do số đo cao hơn, không phải lý do lần ghi bị bỏ qua.">{'The model counts what the kernel asks of memory and ignores caches for loads and stores alike. Caches are why measurements come out higher, not why the store is left out.'}</Translated>,
      },
      {
        label: <Translated vi="Lần ghi chồng lên các lần nạp của luồng kế tiếp, nên số byte của nó ẩn sau lưu lượng đã được đếm.">{'The store overlaps with the next thread\'s loads, so its bytes hide behind traffic already counted.'}</Translated>,
        why: <Translated vi="Chồng lấp có thể giấu thời gian, nhưng số byte vẫn được di chuyển. Cường độ tính toán đếm byte.">{'Overlap can hide time, but the bytes are still moved. Intensity counts bytes.'}</Translated>,
      },
    ],
    explain: (
      <Translated vi={<p>Ở 5×5, vòng lặp nạp 25 × 8 = 200 byte và lần ghi thêm 4: tỷ lệ đổi từ 0.25 thành 50/204 ≈ 0.245, khoảng 2%. Ở mục 7.5 các lần nạp ra khỏi vòng lặp (mỗi luồng nạp một phần tử đầu vào vào bộ nhớ chia sẻ), nên lần nạp và lần ghi trở nên tương đương và lần ghi phải được đếm.</p>}>
<p>
        At 5×5 the loop loads 25 × 8 = 200 bytes and the store adds 4: the ratio moves from 0.25 to 50/204 ≈ 0.245, about
        2%. In section 7.5 the loads leave the loop (each thread loads one input element into shared memory), so loads
        and stores become comparable and the store has to be counted.
      </p>
</Translated>
    ),
    source: 'ch. 7 §7.3, p. 167 and §7.5, p. 176',
    vi: { source: 'ch. 7 §7.3, p. 167 và §7.5, p. 176' },
  },
  'ch07/7.3/q-1d': {
    kind: 'numeric',
    id: 'ch07/7.3/q-1d',
    prompt: (
      <Translated vi={<>FLOP/B lý tưởng của một phép tích chập 1D với bộ lọc 9 tap trên số float 4 byte là bao nhiêu? Bỏ qua số byte của chính bộ lọc, như sách làm trong 2D.</>}>
        <>What is the ideal FLOP/B of a 1D convolution with a 9-tap filter on 4-byte floats? Ignore the filter&rsquo;s own bytes, as the book does in 2D.</>
      </Translated>
    ),
    answer: 2.25,
    tol: 0.01,
    unit: 'FLOP/B',
    explain: (
      <Translated vi={<p>n đầu ra, mỗi đầu ra 9 tap, là 18n phép toán; đọc đầu vào một lần và ghi đầu ra một lần là 8n byte. 18n / 8n = 2.25, tức m/4. Trong 1D mỗi phần tử đầu vào tham gia m đầu ra, không phải m², nên mức lý tưởng chỉ tăng tuyến tính theo bộ lọc: 9 tap trong 1D mua được đúng thứ mà một bộ lọc 3×3 mua được trong 2D.</p>}>
<p>
        n outputs of 9 taps is 18n operations; reading the input once and writing the output once is 8n bytes. 18n / 8n =
        2.25, which is m/4. In 1D each input element takes part in m outputs, not m², so the ideal grows only linearly
        with the filter: 9 taps in 1D buy what a 3×3 filter buys in 2D.
      </p>
</Translated>
    ),
    source: 'the method of ch. 7 §7.3, p. 167, applied to 1D (our arithmetic)',
    vi: { source: 'phương pháp của ch. 7 §7.3, p. 167, áp dụng cho 1D (phép tính của chúng tôi)' },
  },
  'ch07/7.3/q-fp16': {
    kind: 'numeric',
    id: 'ch07/7.3/q-fp16',
    prompt: (
      <Translated vi={<>FLOP/B lý tưởng của một phép tích chập 3×3 có đầu vào và đầu ra là các giá trị 2 byte là bao nhiêu?</>}>
        <>What is the ideal FLOP/B of a 3×3 convolution whose inputs and outputs are 2-byte values?</>
      </Translated>
    ),
    answer: 4.5,
    tol: 0.01,
    unit: 'FLOP/B',
    explain: (
      <Translated vi={<p>Số phép toán không đổi, 2·n²·9, còn số byte giảm một nửa thành 2·n²·2, nên mức lý tưởng là 9/2 = 4.5: m² chia cho số byte mỗi phần tử. Mọi cường độ tính toán trên trang này co giãn theo cùng cách, và đó là lý do trọng số lượng tử hóa giúp một kernel bị giới hạn bởi bộ nhớ ngay cả khi không phép tính nào rẻ hơn.</p>}>
<p>
        The operations are unchanged, 2·n²·9, and the bytes halve to 2·n²·2, so the ideal is 9/2 = 4.5: m² divided by
        the bytes per element. Every intensity on this page scales the same way, which is why quantised weights help a
        memory-bound kernel even when no arithmetic gets cheaper.
      </p>
</Translated>
    ),
    source: 'the method of ch. 7 §7.3, p. 167; other element sizes are our extension (notes/ch07.md, "Relevance")',
    vi: { source: 'phương pháp của ch. 7 §7.3, p. 167; các kích thước phần tử khác là phần mở rộng của chúng tôi (notes/ch07.md, "Relevance")' },
  },
};

// ---------------------------------------------------------------- the page
export function S73() {
  const q = questionsS73;
  return (
    <>
      <Translated vi={<p>Kernel của mục 7.2 là đúng. Mục này hỏi nó có thể nhanh đến đâu, bằng cách đếm mà chương 5 đã dùng cho phép nhân ma trận: số phép toán dấu phẩy động thực hiện trên mỗi byte lưu lượng bộ nhớ toàn cục (global memory). Phép đếm ngắn, và nó cho hai con số chênh nhau một hệ số mà bạn gọi tên được trước khi chạy bất cứ thứ gì.</p>}>
<p>
        The kernel of section 7.2 is correct. This section asks how fast it can be, with the counting chapter 5 used
        for matrix multiplication: floating-point operations performed per byte of global-memory traffic. The count is
        short, and it gives two numbers that differ by a factor you can name before you run anything.
      </p>
</Translated>
      <Source section="7.3" pages="167" />

      <h2><Translated vi="Phép tính cần những gì">What the computation needs</Translated></h2>
      <Translated vi={<p>Lấy một đầu vào n × n, một bộ lọc m × m với m = 2r + 1, và số float 4 byte. Mỗi đầu ra gồm m² tap, mỗi tap một phép nhân và một phép cộng, nên cả phép tích chập là 2·n²·m² phép toán. Các tap rơi vào ô ma (ghost cell) cũng được đếm, như sách đếm.</p>}>
<p>
        Take an n × n input, an m × m filter with m = 2r + 1, and 4-byte floats. Every output is m² taps of one multiply
        and one add, so the whole convolution is 2·n²·m² operations. Taps that land on ghost cells are counted too, as
        the book counts them.
      </p>
</Translated>
      <Translated vi={<p>Lưu lượng ít nhất mà bất kỳ kernel nào có thể có là đọc mỗi phần tử đầu vào một lần, đọc bộ lọc một lần và ghi mỗi đầu ra một lần: 4n² + 4m² + 4n² byte. Bộ lọc rất nhỏ so với ảnh, nên bỏ 4m² của nó đi, và tỷ lệ tốt nhất có thể là</p>}>
<p>
        The least traffic any kernel could have reads every input element once, reads the filter once and writes every
        output once: 4n² + 4m² + 4n² bytes. The filter is tiny next to the image, so drop its 4m², and the best
        possible ratio is
      </p>
</Translated>
      <p style={{ textAlign: 'center', fontSize: '1.15rem' }}>
        2·n²·m² / 8n² = <b>m²/4 FLOP/B</b>{/* i18n-ignore: formula */}
      </p>
      <Translated vi={<p>Nó tăng theo diện tích bộ lọc: 2.25 cho bộ lọc 3 × 3, 30.25 cho 11 × 11. Lý do là tái sử dụng. Mỗi phần tử đầu vào tham gia m² đầu ra, và một kernel nạp nó một lần sẽ được m² lần dùng từ mỗi lần nạp.</p>}>
<p>
        It grows with the filter&rsquo;s area: 2.25 for a 3 × 3 filter, 30.25 for 11 × 11. The reason is reuse. Each
        input element takes part in m² outputs, and a kernel that fetched it once would get m² uses out of every fetch.
      </p>
</Translated>
      <Source section="7.3" pages="167" />

      <h2><Translated vi="Kích thước bộ lọc quyết định chế độ">The filter size decides the regime</Translated></h2>
      <Translated vi={<p>Giờ hãy đặt các con số đó cạnh một thiết bị. Sách dùng GPU của chương 5 làm ví dụ, có ngưỡng khoảng 20 FLOP/B. Một phép tích chập 3 × 3 chỉ đạt 2.25 ngay cả trong trường hợp lý tưởng, nên không kernel nào, dù viết tốt đến đâu, giữ được các đơn vị số học của thiết bị đó bận rộn. Một phép tích chập 11 × 11, ở mức 30.25, có thể chạm đỉnh số học, nhưng chỉ khi kernel thực sự tiến gần mức lý tưởng của nó, và đó là điều phần còn lại của chương giải quyết.</p>}>
<p>
        Now put those numbers against a device. The book uses the GPU of chapter 5 as its example, whose threshold is
        about 20 FLOP/B. A 3 × 3 convolution reaches only 2.25 even in the ideal case, so no kernel, however well
        written, keeps that device&rsquo;s arithmetic units busy. An 11 × 11 convolution, at 30.25, could reach the
        arithmetic peak, but only if the kernel actually gets close to its ideal, which is what the rest of the
        chapter works on.
      </p>
</Translated>
      <Today checked={tr('2026-11-20 against nvidia.com (notes/today/ch07-facts.md A7, ch05-facts.md A8)', '2026-11-20, đối chiếu với nvidia.com (notes/today/ch07-facts.md A7, ch05-facts.md A8)')}>
        <Translated vi={<p>Thiết bị của sách là NVIDIA H100 SXM. Với 67 TFLOPS (FP32) và 3.35 TB/s mà NVIDIA công bố, ngưỡng là 20.0 FLOP/B; con số 66.9 của sách cho 19.97. Cách tô màu bị giới hạn bởi bộ nhớ và bị giới hạn bởi tính toán của Fig. 7.14 khớp với nó từng ô một. Ngưỡng của chính bạn đến từ lab 5.1.</p>}>
<p>
          The book&rsquo;s device is the NVIDIA H100 SXM. With NVIDIA&rsquo;s printed 67 TFLOPS (FP32) and 3.35 TB/s the
          threshold is 20.0 FLOP/B; the book&rsquo;s 66.9 gives 19.97. The memory-bound and compute-bound shading of
          Fig. 7.14 matches it cell for cell. Your own threshold comes from lab 5.1.
        </p>
</Translated>
      </Today>
      <IdealExplorer />
      <Translated vi={<p>Ba điều cần tìm trong đó. Di chuyển m thì mức lý tưởng leo lên theo diện tích bộ lọc, trong khi cả hai kernel như đã viết vẫn nằm ngang: khoảng cách giữa chúng là m². Chuyển sang 2 byte mỗi phần tử thì mọi đường đều tăng gấp đôi, cả mức lý tưởng lẫn các kernel. Chuyển sang 1D thì mức lý tưởng chỉ tăng theo m, nên một bộ lọc 1D trên float32 cần 81 tap thì thiết bị ví dụ của sách mới có thể bị giới hạn bởi tính toán.</p>}>
<p>
        Three things to find in it. Move m and the ideal climbs with the filter&rsquo;s area, while both kernels as
        written stay flat: the gap between them is m². Switch to 2 bytes per element and every line doubles, ideal and
        kernels alike. Switch to 1D and the ideal grows only with m, so a 1D filter in float32 needs 81 taps before the
        book&rsquo;s example device could be compute-bound.
      </p>
</Translated>
      <Simplification>
        <Translated vi={<p>Mỗi thiết bị một đỉnh và một băng thông. Đổi số byte mỗi phần tử chỉ thay đổi lưu lượng: đỉnh vẫn là đỉnh bạn đã chọn, dù các thiết bị thật có đỉnh khác nhau cho số học 16-bit và 8-bit.</p>}>
<p>
          One peak and one bandwidth per device. Changing the bytes per element changes the traffic only: the peak
          stays the one you picked, although real devices have different peaks for 16-bit and 8-bit arithmetic.
        </p>
</Translated>
        <Translated vi={<p>Không có bộ nhớ đệm: các đường là những gì một kernel đòi hỏi ở bộ nhớ (xem bên dưới).</p>}>
<p>No caches: the lines are what a kernel asks of memory (see below).</p>
</Translated>
      </Simplification>
      <Ask q={q['ch07/7.3/q-regime']} />

      <h2><Translated vi="Kernel cơ bản đòi hỏi những gì">What the basic kernel asks for</Translated></h2>
      <Translated vi={<p>Giờ đến kernel bạn đang có. Câu lệnh bên trong của nó, ở dạng đã sửa của lab 7.1, là{' '}<code>Pvalue += F[fRow*(2*r+1) + fCol] * N[inRow*width + inCol];</code></p>}>
<p>
        Now the kernel you have. Its inner statement, in the corrected form of lab 7.1, is{' '}
        <code>Pvalue += F[fRow*(2*r+1) + fCol] * N[inRow*width + inCol];</code>
      </p>
</Translated>
      <Predict
        id="ch07/7.3/p-basic"
        prompt={tr('For an 11 × 11 filter, how many operations does that statement perform per byte it loads from global memory, counting only the loop?', 'Với một bộ lọc 11 × 11, câu lệnh đó thực hiện bao nhiêu phép toán trên mỗi byte nó nạp từ bộ nhớ toàn cục, chỉ tính vòng lặp?')}
        unit="FLOP/B"
      >
        {(v) => (
          <Translated vi={<p>Mỗi lần lặp nạp một phần tử của F và một của N, 8 byte, cho một phép nhân và một phép cộng: 2/8 ={' '}<b>0.25 FLOP/B</b>.{' '}{Math.abs(v - 0.25) < 0.02 ? 'Chính xác.' : v > 1 ? 'Nếu bạn ra số lớn hơn, có lẽ bạn đang nghĩ tới mức lý tưởng: kernel không chia sẻ gì giữa các luồng, nên nó không được chút tái sử dụng nào.' : 'Hãy đếm byte, không đếm lượt truy cập: mỗi lần nạp trong hai lần nạp di chuyển 4 byte.'}{' '}Với 3 × 3 cũng vậy: mỗi tap mang theo hai lần nạp của riêng nó, nên kích thước bộ lọc bị triệt tiêu. So với mức lý tưởng 30.25, kernel này đòi bộ nhớ gấp 121 lần số byte nó cần.</p>}>
<p>
            Each iteration loads one element of F and one of N, 8 bytes, for one multiply and one add: 2/8 ={' '}
            <b>0.25 FLOP/B</b>.{' '}
            {Math.abs(v - 0.25) < 0.02
              ? 'Exactly.'
              : v > 1
                ? 'If you had more, you were probably thinking of the ideal: the kernel does not share anything between threads, so it gets none of the reuse.'
                : 'Count bytes, not accesses: each of the two loads moves 4 of them.'}{' '}
            It is the same for 3 × 3: each tap brings its own two loads, so the filter size cancels. Against the ideal of
            30.25, this kernel asks memory for 121 times the bytes it needs.
          </p>
</Translated>
        )}
      </Predict>
      <Source section="7.3" pages="167" />
      <Translated vi={<p>Hệ số m² đó chính là phần tái sử dụng mà mức lý tưởng trông cậy vào còn kernel thì vứt bỏ: mỗi luồng nạp lại toàn bộ vùng của nó, dù các luồng lân cận vừa nạp phần lớn vùng đó một lúc trước.</p>}>
<p>
        That factor of m² is exactly the reuse the ideal counts on and the kernel throws away: every thread fetches its
        whole patch again, although its neighbours fetched most of it a moment ago.
      </p>
</Translated>
      <Translated vi={<p>Còn lần ghi thì sao? Mỗi luồng ghi <code>P</code> một lần, sau các vòng lặp của nó. Sách bỏ nó ra khỏi con số 0.25, và điều đó hợp lý khi vòng lặp di chuyển 8 byte mỗi tap: ở 5 × 5 lần ghi chỉ thêm 4 byte vào 200. Mục 7.5 phải đếm lần ghi, vì ở đó các lần nạp ra khỏi vòng lặp và không còn lấn át chúng nữa.</p>}>
<p>
        What about the store? Each thread writes <code>P</code> once, after its loops. The book leaves it out of the
        0.25, and that is fair while the loop moves 8 bytes per tap: at 5 × 5 the store adds 4 bytes to 200. Section 7.5
        has to count stores, because there the loads leave the loop and stop dwarfing them.
      </p>
</Translated>
      <Source section="7.3, 7.5" pages="167, 176" />
      <Ask q={q['ch07/7.3/q-store']} />

      <h2><Translated vi="Cường độ đo được cao hơn mức này">Measured intensity is higher than this</Translated></h2>
      <Translated vi={<p>Con số 0.25 tính mọi lần nạp vào DRAM. Sách nói ngay sau đó rằng con số thật cao hơn: một phần tử bộ lọc được mọi luồng đọc và một phần tử đầu vào được tới m² luồng đọc, nên nhiều lần nạp trong số đó lại được tìm thấy trong cache L1 hoặc L2. Chương 5 đã cho thấy điều tương tự với phép nhân ma trận: <Lab51Naive />.</p>}>
<p>
        The 0.25 charges every load to DRAM. The book says in the same breath that the real figure is higher: a filter
        element is read by every thread and an input element by up to m² threads, so many of those loads are found in
        the L1 or L2 cache instead. Chapter 5 showed the same thing for matrix multiplication: <Lab51Naive />.
      </p>
</Translated>
      <Source section="7.3" pages="168" />
      <Translated vi={<p>Vậy hãy đọc mô hình như một phát biểu về những gì kernel đòi hỏi ở bộ nhớ. Thứ thực sự tới DRAM tùy vào thiết bị và bộ nhớ đệm của nó, và chỉ một phép đo mới cho thấy được. Lab 7.1 đo thời gian kernel cơ bản và kernel dùng bộ nhớ hằng (constant memory) của mục kế tiếp ở nhiều bán kính. Với băng thông từ lab 5.1 cho cùng thiết bị, bảng bên dưới đặt mỗi số đo cạnh mức trần theo mô hình của nó.</p>}>
<p>
        So read the model as a statement about what the kernel asks of memory. What actually reaches DRAM is up to the
        device and its caches, and only a measurement shows it. Lab 7.1 times the basic kernel and the constant-memory
        kernel of the next section at several radii. With lab 5.1&rsquo;s bandwidth for the same device, the panel below
        sets each measurement against its model ceiling.
      </p>
</Translated>
      <YourKernels />

      <h2><Translated vi="Câu trả lời của sách: bộ nhớ trên chip, một cách có chủ đích">The book&rsquo;s answer: on-chip memory, on purpose</Translated></h2>
      <Translated vi={<p>Bộ nhớ đệm có giúp, nhưng bạn không quyết định được chúng giữ gì. Kế hoạch của sách cho phần còn lại của chương là chủ động đặt dữ liệu được tái sử dụng vào bộ nhớ trên chip: bộ lọc vào bộ nhớ hằng, nơi bộ nhớ đệm hằng phục vụ nó (<a href={href('ch07', '7.4')}>mục 7.4</a>, đưa kernel từ 0.25 lên 0.5), và các ô của đầu vào vào bộ nhớ chia sẻ (shared memory) (<a href={href('ch07', '7.5')}>mục 7.5</a>, nhắm vào hệ số m²).</p>}>
<p>
        Caches help, but you do not decide what they keep. The book&rsquo;s plan for the rest of the chapter is to put the
        reused data in on-chip memory deliberately: the filter in constant memory, where the constant cache serves it
        (<a href={href('ch07', '7.4')}>section 7.4</a>, which takes the kernel from 0.25 to 0.5), and tiles of the input
        in shared memory (<a href={href('ch07', '7.5')}>section 7.5</a>, which goes after the factor of m²).
      </p>
</Translated>
      <Source section="7.3" pages="168" />

      <h2><Translated vi="Số byte mỗi phần tử, và trường hợp 1D">Bytes per element, and the 1D case</Translated></h2>
      <Translated vi={<p>Hai biến thể bạn sẽ gặp trong mã suy luận. Với ít byte hơn mỗi phần tử, mọi cường độ tính toán trên trang này được chia cho số byte, nên giá trị 16-bit nhân đôi tất cả và giá trị 8-bit nhân bốn, cả mức lý tưởng lẫn các kernel. Với một chiều thay vì hai, mỗi đầu vào tham gia m đầu ra thay vì m², nên mức lý tưởng là m/4 trên float32: nó tăng theo chiều rộng bộ lọc, không theo diện tích.</p>}>
<p>
        Two variations you will meet in inference code. With fewer bytes per element every intensity on this page is
        divided by the bytes, so 16-bit values double all of them and 8-bit values quadruple them, ideal and kernels
        alike. With one dimension instead of two, each input takes part in m outputs instead of m², so the ideal is
        m/4 in float32: it grows with the filter&rsquo;s width, not its area.
      </p>
</Translated>
      <Beyond>
        <Translated vi={<p>Cả hai là phép tính của chúng tôi theo phương pháp của sách; sách cố định 2D và số float 4 byte. Một bộ lọc 1D 4 tap trên float32 có mức lý tưởng 1 FLOP/B, thấp hơn nhiều so với ngưỡng 20 của sách: một kernel như vậy được đánh giá theo băng thông nó đạt được. Mục 7.5 cho thấy một bộ lọc như vậy chạy ở đâu trong việc phục vụ LLM.</p>}>
<p>
          Both are our arithmetic with the book&rsquo;s method; the book fixes 2D and 4-byte floats. A 4-tap 1D filter in
          float32 has an ideal of 1 FLOP/B, far below the book&rsquo;s threshold of 20: a kernel like that is judged by the
          bandwidth it reaches. Section 7.5 shows where such a filter runs in LLM serving.
        </p>
</Translated>
      </Beyond>
      <Ask q={q['ch07/7.3/q-1d']} />
      <Ask q={q['ch07/7.3/q-fp16']} />
    </>
  );
}
