import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, NoteGroup, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { benchRows, fmt, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BOOK_RULE,
  H100,
  H100_RATIO,
  better,
  breakEven,
  copyFirstCost,
  lifetimeTables,
  mappedCost,
  mixedCurve,
  mixedTime,
  passCurve,
  ratioFromProbe,
} from '../../sim/zerocopy';

export const LAB = 'appC/lab1_feeding';

// ---------------------------------------------------------------- link ratios
interface RatioOption {
  key: string;
  label: string;
  rho: number;
  note: string;
}

// Built inside a useMemo keyed on the reports only, so the reader text is read through getters at render time.
export function ratioOptions(reports: Reports): RatioOption[] {
  const out: RatioOption[] = [
    {
      key: 'book',
      get label() {
        return tr("the book's rule of thumb (10%)", 'quy tắc kinh nghiệm của sách (10%)');
      },
      rho: BOOK_RULE,
      get note() {
        return tr(
          'The book: interconnect bandwidth is "typically less than 10%" of global-memory bandwidth (p. 625).',
          'Sách: băng thông kênh kết nối "thường dưới 10%" băng thông bộ nhớ toàn cục (p. 625).',
        );
      },
    },
    {
      key: 'h100',
      get label() {
        return tr("the book's example device (H100 SXM)", 'thiết bị trong ví dụ của sách (H100 SXM)');
      },
      rho: H100_RATIO,
      get note() {
        return tr(
          `${H100.linkGBps} GB/s of PCIe Gen5 (NVIDIA's page) against ${H100.deviceGBps / 1000} TB/s of device memory (the book, ch. 5 p. 94): about ${fmt(100 * H100_RATIO, 2)}%, our arithmetic (appC-facts A4).`,
          `${H100.linkGBps} GB/s của PCIe Gen5 (trang của NVIDIA) so với ${H100.deviceGBps / 1000} TB/s của bộ nhớ device (sách, ch. 5 p. 94): khoảng ${fmt(100 * H100_RATIO, 2)}%, theo phép tính của chúng tôi (appC-facts A4).`,
        );
      },
    },
  ];
  for (const b of realBackends(reports)) {
    if (b === 'metal' || b === 'race') continue;
    const row = benchRows(reports[b], LAB).find((r) => r.name === 'probe');
    const rho = ratioFromProbe(row);
    if (rho == null) continue;
    out.push({
      key: 'yours-' + b,
      get label() {
        return tr(`yours: ${String(reports[b].device.name ?? b)} (${b})`, `của bạn: ${String(reports[b].device.name ?? b)} (${b})`);
      },
      rho,
      get note() {
        return tr(
          `Lab C.1's probe on ${reports[b].generated.slice(0, 10)}: ${fmt(Number(row!.link_gb_per_s))} GB/s for a pinned host-to-device copy against ${fmt(Number(row!.device_gb_per_s))} GB/s for a device-to-device copy. Copy bandwidths stand in for in-kernel reads.`,
          `Phép dò của lab C.1 ngày ${reports[b].generated.slice(0, 10)}: ${fmt(Number(row!.link_gb_per_s))} GB/s cho một lần sao chép từ bộ nhớ ghim của host sang device, so với ${fmt(Number(row!.device_gb_per_s))} GB/s cho một lần sao chép từ device sang device. Băng thông sao chép được dùng thay cho các lần đọc bên trong kernel.`,
        );
      },
    });
  }
  return out;
}

// ---------------------------------------------------------------- a small log-scale plot
interface Series {
  label: string;
  pts: [number, number][];
  dash?: boolean;
}
function Plot({
  series,
  xLog,
  xDomain,
  yDomain,
  xLabel,
  yLabel,
  marks,
  vline,
  aria,
}: {
  series: Series[];
  xLog: boolean;
  xDomain: [number, number];
  yDomain: [number, number];
  xLabel: string;
  yLabel: string;
  marks: { x: number; y: number; label: string }[];
  vline?: { x: number; label: string };
  aria: string;
}) {
  const W = 460;
  const H = 260;
  const pad = { l: 52, r: 12, t: 12, b: 40 };
  const lx = (v: number) => (xLog ? Math.log10(v) : v);
  const X = (v: number) => pad.l + ((lx(v) - lx(xDomain[0])) / (lx(xDomain[1]) - lx(xDomain[0]))) * (W - pad.l - pad.r);
  const Y = (v: number) =>
    H - pad.b - ((Math.log10(v) - Math.log10(yDomain[0])) / (Math.log10(yDomain[1]) - Math.log10(yDomain[0]))) * (H - pad.t - pad.b);
  const yt: number[] = [];
  for (let e = Math.floor(Math.log10(yDomain[0])); e <= Math.ceil(Math.log10(yDomain[1])); e++)
    for (const m of [1, 2, 5]) {
      const v = m * 10 ** e;
      if (v >= yDomain[0] * 0.999 && v <= yDomain[1] * 1.001) yt.push(v);
    }
  const xt = xLog ? [1, 10, 100, 1000].filter((v) => v >= xDomain[0] && v <= xDomain[1]) : [0, 0.25, 0.5, 0.75, 1];
  const inX = (v: number) => v >= xDomain[0] && v <= xDomain[1];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={aria} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {yt.map((v) => (
        <g key={'y' + v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 5} y={Y(v) + 4} fontSize="10" textAnchor="end">
            {fmt(v)}
          </text>
        </g>
      ))}
      {xt.map((v) => (
        <g key={'x' + v}>
          <line x1={X(v)} x2={X(v)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(v)} y={H - pad.b + 14} fontSize="10" textAnchor="middle">
            {xLog ? v : `${v * 100}%`}
          </text>
        </g>
      ))}
      <text x={(W + pad.l) / 2} y={H - 6} fontSize="11" textAnchor="middle">
        {xLabel}
      </text>
      <text x={12} y={(H - pad.b + pad.t) / 2} fontSize="11" textAnchor="middle" transform={`rotate(-90 12 ${(H - pad.b + pad.t) / 2})`}>
        {yLabel}
      </text>
      {vline && inX(vline.x) ? (
        <g>
          <line x1={X(vline.x)} x2={X(vline.x)} y1={pad.t} y2={H - pad.b} stroke="var(--ink-2)" strokeDasharray="2 3" />
          <text x={X(vline.x) + 4} y={pad.t + 12} fontSize="10" style={{ fill: 'var(--ink-2)' }}>
            {vline.label}
          </text>
        </g>
      ) : null}
      {series.map((s) => (
        <polyline
          key={s.label}
          fill="none"
          stroke={s.dash ? 'var(--ink-2)' : 'var(--ink)'}
          strokeWidth="2"
          strokeDasharray={s.dash ? '6 4' : undefined}
          points={s.pts
            .filter(([, y]) => y >= yDomain[0] * 0.999 && y <= yDomain[1] * 1.001)
            .map(([x, y]) => `${X(x)},${Y(y)}`)
            .join(' ')}
        />
      ))}
      {marks.map((m, i) => (
        <g key={i}>
          <circle cx={X(m.x)} cy={Y(m.y)} r="4.5" fill="var(--page)" stroke="var(--ink)" strokeWidth="2" />
          <text x={i ? X(m.x) + 7 : X(m.x) - 7} y={Y(m.y) - 7} fontSize="10.5" textAnchor={i ? 'start' : 'end'}>
            {m.label}
          </text>
        </g>
      ))}
    </svg>
  );
}

// ---------------------------------------------------------------- explorer
const pct = (v: number) => `${fmt(100 * v, 3)}%`;

function MappedCost() {
  const { reports } = useReports();
  const options = useMemo(() => ratioOptions(reports), [reports]);
  const [key, setKey] = useState('book');
  const [custom, setCustom] = useState(0.05);
  const [f, setF] = useState(0.1);
  const [sExp, setSExp] = useState(6); // s = 1 / 2^sExp
  const [kExp, setKExp] = useState(1); // k = 10^kExp
  const opt = options.find((o) => o.key === key);
  const rho = key === 'custom' ? custom : (opt?.rho ?? BOOK_RULE);
  const s = 1 / 2 ** sExp;
  const k = Math.max(1, Math.round(10 ** kExp));
  const none = mixedTime(f, rho, 'none');
  const perfect = mixedTime(f, rho, 'perfect');
  const top = Math.max(2, 1 / rho);
  const kStar = breakEven(s, rho);
  const m = mappedCost(k, s, rho);
  const c = copyFirstCost(k, s, rho);
  const win = better(k, s, rho);
  const curveB = passCurve(s, rho);
  const yMin = Math.min(...curveB.map((p) => Math.min(p.mapped, p.copy))) * 0.8;
  const yMax = Math.max(...curveB.map((p) => Math.max(p.mapped, p.copy))) * 1.25;
  const hasYours = options.some((o) => o.key.startsWith('yours-'));

  return (
    <Bench
      title={tr('What a mapped read costs', 'Một lần đọc ánh xạ tốn bao nhiêu')}
      task={tr(
        'Choose a link, then move the mapped fraction, the share touched per pass and the number of passes.',
        'Chọn một đường truyền, rồi kéo tỷ lệ ánh xạ, phần được chạm tới mỗi lượt và số lượt.',
      )}
      foot={
        <Translated
          vi={
            <>
              Chỉ tính băng thông. Băng thông device là 1; đường truyền có băng thông ρ. Biểu đồ thứ nhất: một kernel bị
              giới hạn bởi bộ nhớ lấy một tỷ lệ f số byte của nó qua đường truyền, so với chính kernel đó khi mọi byte nằm
              trong bộ nhớ device. Biểu đồ thứ hai: một bảng được đọc k lần, mỗi lượt một tỷ lệ s, tính bằng đơn vị "số
              byte của một bảng ở băng thông device".
            </>
          }
        >
          <>
            Bandwidth only. Device bandwidth is 1; the link has bandwidth ρ. First plot: a memory-bound kernel that takes a
            fraction f of its bytes over the link, relative to the same kernel with every byte in device memory. Second plot: a
            table read k times, a fraction s per pass, in units of "one table's bytes at device bandwidth".
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Băng thông đường truyền tính theo tỷ lệ băng thông device">
            Link bandwidth as a share of device bandwidth
          </Translated>
          <select style={{ maxWidth: '100%' }} value={key} onChange={(e) => setKey(e.target.value)} aria-label={tr('link', 'đường truyền')}>
            {options.map((o) => (
              <option key={o.key} value={o.key}>
                {o.label}
              </option>
            ))}
            <option value="custom">{tr('custom', 'tùy chọn')}</option>
          </select>
        </label>
        {key === 'custom' ? (
          <label className="field">
            ρ = {pct(custom)}
            <input type="range" min={0.01} max={0.5} step={0.005} value={custom} aria-label={tr('custom ratio', 'tỷ lệ tùy chọn')} onChange={(e) => setCustom(+e.target.value)} />
          </label>
        ) : null}
        <span className="ui" data-rho={rho}>
          ρ = {pct(rho)}
        </span>
      </div>
      <p className="ui muted" style={{ marginTop: 0, fontSize: '0.82rem' }} data-rho-note>
        {key === 'custom' ? tr('A ratio you chose.', 'Một tỷ lệ do bạn chọn.') : opt?.note}
        {hasYours ? null : (
          <Translated
            vi={
              <>
                {' '}
                Chưa có "của bạn": hãy chạy lab C.1 với <code>--bench</code> trên một máy CUDA hoặc HIP (
                <code>python3 labs/pmpp.py run appC/lab1 --bench</code>) rồi tải lại trang. Trên Apple silicon không có
                đường truyền nào để đo.
              </>
            }
          >
            <>
              {' '}
              No "yours" yet: run lab C.1 with <code>--bench</code> on a CUDA or HIP machine (
              <code>python3 labs/pmpp.py run appC/lab1 --bench</code>) and reload. On Apple silicon there is no link to
              measure.
            </>
          </Translated>
        )}
      </p>
      <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(19rem, 1fr))' }}>
        <div>
          <label className="field ui">
            <Translated vi="Số byte từ bộ nhớ host ánh xạ: f =">Bytes from mapped host memory: f =</Translated> {pct(f)}
            <input type="range" min={0} max={1} step={0.01} value={f} aria-label={tr('mapped fraction', 'tỷ lệ ánh xạ')} onChange={(e) => setF(+e.target.value)} />
          </label>
          <Plot
            aria={tr('Kernel time against the mapped fraction', 'Thời gian kernel theo tỷ lệ ánh xạ')}
            series={[
              { label: 'no overlap', pts: mixedCurve(rho, 'none').map((p) => [p.f, p.t]) }, // i18n-ignore: React key, never shown
              { label: 'perfect overlap', pts: mixedCurve(rho, 'perfect').map((p) => [p.f, p.t]), dash: true }, // i18n-ignore: React key, never shown
            ]}
            xLog={false}
            xDomain={[0, 1]}
            yDomain={[1, top]}
            xLabel={tr('share of bytes from mapped memory (f)', 'tỷ lệ byte từ bộ nhớ ánh xạ (f)')}
            yLabel={tr('time vs all-device', 'thời gian so với toàn device')}
            marks={[
              { x: f, y: none, label: `${fmt(none)}×` },
              { x: f, y: Math.max(1, perfect), label: `${fmt(perfect)}×` },
            ]}
          />
          <p className="ui" style={{ fontSize: '0.85rem' }} data-mixed>
            <Translated
              vi={
                <>
                  Nét liền: hai luồng dữ liệu chạy lần lượt, <b>{fmt(none)}×</b>. Nét đứt: chồng lấp hoàn hảo,{' '}
                  <b>{fmt(perfect)}×</b>. Đường truyền mang {pct(f)} số byte và chiếm{' '}
                  <b>{pct(f / rho / none)}</b> thời gian khi không chồng lấp.
                </>
              }
            >
              <>
                Solid: the two streams one after the other, <b>{fmt(none)}×</b>. Dashed: perfectly overlapped,{' '}
                <b>{fmt(perfect)}×</b>. The link carries {pct(f)} of the bytes and takes{' '}
                <b>{pct(f / rho / none)}</b> of the time without overlap.
              </>
            </Translated>
          </p>
        </div>
        <div>
          <div className="row" style={{ gap: '0.4rem 1rem' }}>
            <label className="field ui">
              <Translated vi="Phần được chạm tới mỗi lượt: s = 1/">Touched per pass: s = 1/</Translated>{2 ** sExp}
              <input type="range" min={0} max={10} step={1} value={sExp} aria-label={tr('share touched per pass', 'phần được chạm tới mỗi lượt')} onChange={(e) => setSExp(+e.target.value)} />
            </label>
            <label className="field ui">
              <Translated vi="Số lượt: k =">Passes: k =</Translated> {k}
              <input type="range" min={0} max={3} step={0.05} value={kExp} aria-label={tr('passes', 'số lượt')} onChange={(e) => setKExp(+e.target.value)} />
            </label>
          </div>
          <Plot
            aria={tr('Read in place against copy first, by number of passes', 'Đọc tại chỗ so với sao chép trước, theo số lượt')}
            series={[
              { label: 'read in place', pts: curveB.map((p) => [p.k, p.mapped]), dash: true }, // i18n-ignore: React key, never shown
              { label: 'copy, then read', pts: curveB.map((p) => [p.k, p.copy]) }, // i18n-ignore: React key, never shown
            ]}
            xLog
            xDomain={[1, 1000]}
            yDomain={[yMin, yMax]}
            xLabel={tr('passes over the table (k)', 'số lượt đi qua bảng (k)')}
            yLabel={tr('cost', 'chi phí')}
            marks={[
              { x: k, y: m, label: tr('in place', 'tại chỗ') },
              { x: k, y: c, label: tr('copy first', 'sao chép trước') },
            ]}
            vline={{ x: kStar, label: tr(`break-even ${fmt(kStar)}`, `hòa vốn ${fmt(kStar)}`) }}
          />
          <p className="ui" style={{ fontSize: '0.85rem' }} data-passes data-winner={win}>
            <Translated
              vi={
                <>
                  Nét đứt: đọc tại chỗ, k·s/ρ = <b>{fmt(m)}</b>. Nét liền: sao chép một lần rồi đọc, 1/ρ + k·s ={' '}
                  <b>{fmt(c)}</b>.{' '}
                  {win === 'mapped' ? 'Đọc tại chỗ thắng' : win === 'copy' ? 'Sao chép trước thắng' : 'Hòa'}. Hòa vốn
                  ở <b data-kstar>{fmt(kStar)}</b> lượt. Trong suốt vòng đời, kernel đọc lượng dữ liệu bằng {fmt(k * s)}{' '}
                  bảng; đọc tại chỗ thắng khi con số này dưới {fmt(lifetimeTables(rho))}.
                </>
              }
            >
              <>
                Dashed: read in place, k·s/ρ = <b>{fmt(m)}</b>. Solid: copy once, then read, 1/ρ + k·s = <b>{fmt(c)}</b>.{' '}
                {win === 'mapped' ? 'Reading in place wins' : win === 'copy' ? 'Copying first wins' : 'A tie'}. Break-even at{' '}
                <b data-kstar>{fmt(kStar)}</b> passes. Over its life the kernel reads {fmt(k * s)} tables' worth; in place wins
                below {fmt(lifetimeTables(rho))}.
              </>
            </Translated>
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSC2: Record<string, Question> = {
  'appC/C.2/q-mixed': {
    kind: 'numeric',
    id: 'appC/C.2/q-mixed',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel bị giới hạn bởi bộ nhớ lấy 90% số byte từ bộ nhớ device và 10% từ bộ nhớ host ánh xạ. Đường truyền
            có 10% băng thông device, và hai luồng dữ liệu không chồng lấp nhau. Kernel chạy lâu gấp bao nhiêu lần so với
            khi mọi byte nằm trong bộ nhớ device?
          </>
        }
      >
        <>
          A memory-bound kernel takes 90% of its bytes from device memory and 10% from mapped host memory. The link has
          10% of device bandwidth, and the two streams do not overlap. How many times longer does the kernel take than
          with every byte in device memory?
        </>
      </Translated>
    ),
    answer: 1.9,
    tol: 0.02,
    unit: '×',
    explain: (
      <Translated
        vi={
          <p>
            0.9 / 1 + 0.1 / 0.1 = 0.9 + 1 = 1.9. Một phần mười số byte chiếm hơn một nửa thời gian. Ngay cả khi chồng lấp
            hoàn hảo, kernel cũng không thể chạy nhanh hơn luồng dữ liệu chậm hơn của nó, ở đây là 1.0: không nhanh hơn
            chút nào so với kernel toàn device.
          </p>
        }
      >
        <p>
          0.9 / 1 + 0.1 / 0.1 = 0.9 + 1 = 1.9. One tenth of the bytes takes more than half of the time. Even with perfect
          overlap the kernel cannot run faster than its slower stream, here 1.0: no gain over the all-device kernel at
          all.
        </p>
      </Translated>
    ),
    source: 'book p. 625 for the 10% and the reasoning; the arithmetic is ours',
    vi: { source: 'sách p. 625 cho con số 10% và lập luận; phép tính là của chúng tôi' },
  },
  'appC/C.2/q-breakeven': {
    kind: 'numeric',
    id: 'appC/C.2/q-breakeven',
    prompt: (
      <Translated
        vi={
          <>
            Mỗi lượt đọc 1/64 của một bảng, và đường truyền có 10% băng thông device. Từ bao nhiêu lượt trở lên thì sao
            chép bảng trước sẽ thắng đọc tại chỗ?
          </>
        }
      >
        <>
          Each pass reads 1/64 of a table, and the link has 10% of device bandwidth. Above how many passes does copying
          the table first beat reading it in place?
        </>
      </Translated>
    ),
    answer: 71.1,
    tol: 0.02,
    unit: 'passes',
    explain: (
      <Translated
        vi={
          <p>
            Đọc tại chỗ tốn k·s/ρ; sao chép tốn 1/ρ + k·s. Hai đường gặp nhau tại k = 1/(s(1 − ρ)) = 64 / 0.9 ≈ 71.1. Từ 72
            lượt trở đi, hãy sao chép trước. Nói cách khác: đọc tại chỗ chỉ thắng khi, trong suốt vòng đời, kernel đọc ít
            hơn khoảng một bảng dữ liệu.
          </p>
        }
      >
        <p>
          In place costs k·s/ρ; copying costs 1/ρ + k·s. They meet at k = 1/(s(1 − ρ)) = 64 / 0.9 ≈ 71.1. From 72 passes
          on, copy first. Put differently: in place wins only while the kernel reads, over its whole life, less than
          about one table's worth.
        </p>
      </Translated>
    ),
    source: 'our derivation from the book\'s rule, p. 625',
    vi: { unit: 'lượt', source: 'phép suy ra của chúng tôi từ quy tắc của sách, p. 625' },
  },
  'appC/C.2/q-today-pointer': {
    kind: 'choice',
    id: 'appC/C.2/q-today-pointer',
    prompt: (
      <Translated
        vi={
          <>
            Ngày nay, trong một tiến trình 64-bit trên một device CUDA hiện hành, bạn cấp phát một buffer bằng{' '}
            <code>cudaMallocHost</code> để một kernel đọc tại chỗ. Bạn truyền gì cho kernel?
          </>
        }
      >
        <>
          Today, in a 64-bit process on a current CUDA device, you <code>cudaMallocHost</code> a buffer for a kernel to
          read in place. What do you pass to the kernel?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Chính con trỏ đó.">The same pointer.</Translated>,
        correct: true,
        why: (
          <Translated vi="Với định địa chỉ hợp nhất (unified addressing), nó dùng được trực tiếp; cudaHostGetDevicePointer chỉ cần cho bộ nhớ được đăng ký hoặc bộ nhớ write-combined.">
            Under unified addressing it works directly; cudaHostGetDevicePointer is needed only for registered or write-combined memory.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Giá trị lấy từ cudaHostGetDevicePointer, như năm 2009.">
            The value from cudaHostGetDevicePointer, as in 2009.
          </Translated>
        ),
        why: (
          <Translated vi="Vẫn được chấp nhận, nhưng không còn cần cho loại bộ nhớ này.">
            Still accepted, but no longer needed for this memory.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không gì cả: kernel không đọc được bộ nhớ ghim.">Nothing: kernels cannot read pinned memory.</Translated>
        ),
        why: (
          <Translated vi="Có đọc được, qua kênh kết nối. Phần này nói về chính điều đó.">
            They can, over the interconnect. That is what this section is about.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Một bản sao của nó bằng cudaMalloc.">A cudaMalloc copy of it.</Translated>,
        why: (
          <Translated vi="Đó là mô hình sao chép, chính là điều mà đọc tại chỗ tránh được.">
            That is the copy model, which is what reading in place avoids.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            NVIDIA: bộ nhớ từ <code>cudaMallocHost</code> và <code>cudaHostAlloc</code> "luôn truy cập trực tiếp được từ
            mọi device hỗ trợ unified addressing", dù có cờ Mapped hay không.
          </p>
        }
      >
        <p>
          NVIDIA: memory from <code>cudaMallocHost</code> and <code>cudaHostAlloc</code> is "always directly accessible
          from all devices that support unified addressing", with or without the Mapped flag.
        </p>
      </Translated>
    ),
    source: 'appC-facts A3',
    vi: { source: 'appC-facts A3' },
  },
  'appC/C.2/q-pinmemory': {
    kind: 'choice',
    id: 'appC/C.2/q-pinmemory',
    prompt: (
      <Translated
        vi={
          <>
            Vòng lặp huấn luyện của bạn chuyển từng batch bằng <code>batch.to("cuda", non_blocking=True)</code>, vậy mà
            profile cho thấy host bị đứng ở mỗi lần sao chép thay vì đi tiếp sang bước sau. <code>DataLoader</code> được tạo
            với các giá trị mặc định. Điều gì giải thích việc đứng này, và cách nào khắc phục?
          </>
        }
      >
        <>
          Your training loop moves each batch with <code>batch.to("cuda", non_blocking=True)</code>, yet a profile shows
          the host stalling at every copy instead of moving on to the next step. The <code>DataLoader</code> was built with
          its defaults. What explains the stall, and what fixes it?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Các batch nằm trong bộ nhớ phân trang (pageable), và một lần sao chép non_blocking từ bộ nhớ phân trang có thể chặn. DataLoader(pin_memory=True) ghim chúng, nên lần sao chép chạy bất đồng bộ.">
            {
              'The batches are pageable, and a non_blocking copy from pageable memory may block. DataLoader(pin_memory=True) pins them, so the copy runs asynchronously.'
            }
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="non_blocking=True chỉ ảnh hưởng đến các lần sao chép từ GPU sang host, không phải từ host sang GPU; không gì ở phía host khắc phục được việc đứng, nên hãy chồng lấp nó bằng thêm CUDA stream.">
            {'non_blocking=True affects only GPU-to-host copies, not host-to-GPU ones; nothing on the host side fixes the stall, so overlap it with more CUDA streams.'}
          </Translated>
        ),
        why: (
          <Translated vi="PyTorch: khi một tensor đã được ghim, bạn có thể dùng sao chép GPU bất đồng bộ bằng cách truyền non_blocking=True. Cờ này hoạt động; vấn đề nằm ở bộ nhớ mà nó sao chép từ đó.">
            {'PyTorch: once a tensor is pinned, you can use asynchronous GPU copies by passing non_blocking=True. The flag works; the memory it copies from was the problem.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Cho mô hình đọc từng batch tại chỗ từ bộ nhớ host ánh xạ, để host hoàn toàn không phải chờ lần sao chép nào.">
            {'Have the model read each batch in place from mapped host memory, so there is no copy for the host to wait for at all.'}
          </Translated>
        ),
        why: (
          <Translated vi="Cách đó bỏ được lần sao chép nhưng đặt đường truyền dưới mọi lần truy cập của mô hình. PyTorch ghim bộ nhớ để lần sao chép nhanh và bất đồng bộ, không phải để đọc batch tại chỗ.">
            {'That removes the copy and puts the link under every access the model makes. PyTorch pins memory to make the copy fast and asynchronous, not to read batches in place.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Ghim luôn cả trọng số của mô hình, để mọi tensor mà bước đó chạm tới đều bị khóa trang và các lần sao chép thôi bị đứng.">
            {"Pin the model's weights as well, so every tensor the step touches is page-locked and the copies stop stalling."}
          </Translated>
        ),
        why: (
          <Translated vi="Trọng số vốn đã nằm trên GPU. Thứ đi qua đường truyền ở mỗi bước là batch, và batch mới là thứ cần bộ nhớ khóa trang.">
            {'The weights already live on the GPU. What crosses the link at each step is the batch, and the batch is what needs page-locked memory.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Ghi chú CUDA của PyTorch: sao chép từ bộ nhớ ghim nhanh hơn nhiều, tensor được ghim cho phép sao chép bất đồng
            bộ với <code>non_blocking=True</code>, và "với non_blocking=True và bộ nhớ CPU phân trang, lời gọi có thể chặn
            host". Bên dưới, caching host allocator gọi <code>cudaHostAlloc(ptr, size, cudaHostAllocDefault)</code>: được
            ghim, và với unified addressing thì kernel đọc được, nhưng được dùng cho lần sao chép. Đừng ghim vô hạn độ:
            cũng các ghi chú đó cảnh báo rằng lạm dụng bộ nhớ ghim gây ra vấn đề khi RAM sắp cạn.
          </p>
        }
      >
        <p>
          PyTorch's CUDA notes: copies from pinned memory are much faster, pinned tensors allow asynchronous copies with{' '}
          <code>non_blocking=True</code>, and "With non_blocking=True and pageable CPU memory, the call may block the host".
          Under the hood the caching host allocator calls <code>cudaHostAlloc(ptr, size, cudaHostAllocDefault)</code>:
          pinned, and under unified addressing readable by kernels, but used for the copy. Do not pin without limit: the
          same notes warn that overusing pinned memory causes problems when RAM runs low.
        </p>
      </Translated>
    ),
    source: 'not in the book; appC-facts D1 (PyTorch source and docs, 2026-11-20)',
    vi: { source: 'không có trong sách; appC-facts D1 (mã nguồn và tài liệu PyTorch, 2026-11-20)' },
  },
};

// ---------------------------------------------------------------- the page
export function SC2() {
  return (
    <>
      <Translated
        vi={
          <p>
            CUDA 2.2, năm 2009, cho phép một kernel đọc và ghi bộ nhớ host trực tiếp, qua kênh kết nối (interconnect), không
            cần <code>cudaMemcpy</code>. Công thức của sách có ba bước. Cấp phát bằng{' '}
            <code>cudaHostAlloc(&amp;h, bytes, cudaHostAllocMapped)</code>. Xin giá trị phía device bằng{' '}
            <code>cudaHostGetDevicePointer(&amp;d, h, 0)</code>. Truyền <code>d</code>, không phải <code>h</code>, cho kernel.
            Mã host và mã device dùng hai giá trị con trỏ khác nhau cho cùng những byte đó.
          </p>
        }
      >
        <p>
          CUDA 2.2, in 2009, let a kernel read and write host memory directly, over the interconnect, with no{' '}
          <code>cudaMemcpy</code>. The book's recipe has three steps. Allocate with{' '}
          <code>cudaHostAlloc(&amp;h, bytes, cudaHostAllocMapped)</code>. Ask for the device-side value with{' '}
          <code>cudaHostGetDevicePointer(&amp;d, h, 0)</code>. Pass <code>d</code>, not <code>h</code>, to the kernel. Host
          code and device code use two different pointer values for the same bytes.
        </p>
      </Translated>
      <Source section="C.2" pages="624" />
      <Translated
        vi={
          <p>
            Bộ nhớ zero-copy này là bộ nhớ ghim (pinned memory): hệ điều hành không được đẩy nó ra khỏi RAM khi GPU đang
            đọc. Chương 23 giải thích việc ghim (<a href={href('ch23', '23.3')}>mục 23.3</a> ở đây), và tự cấp phát các buffer ghim của mình bằng <code>cudaMallocHost</code>; sách không bao giờ liên hệ hai lời gọi này với nhau.
          </p>
        }
      >
        <p>
          This zero-copy memory is pinned memory: the operating system must not page it out while the GPU reads it.
          Chapter 23 explains pinning (<a href={href('ch23', '23.3')}>section 23.3</a> here), and allocates its own pinned
          buffers with <code>cudaMallocHost</code> instead; the book never relates the two calls.
        </p>
      </Translated>
      <NoteGroup>
        <Source section="C.2" pages="624" figs={tr('pinning: ch. 23, pp. 562–563', 'ghim bộ nhớ: ch. 23, pp. 562–563')} />
        <Aside>
          <Translated
            vi={
              <p>
                <b>Trong bản PDF</b> phụ lục nói "chúng tôi đã nhắc" rằng các giá trị khác của cờ dành cho cách dùng nâng
                cao. Chúng tôi không tìm thấy lần nhắc nào trước đó trong lần xuất bản này (tìm trong văn bản chỉ thấy{' '}
                <code>cudaHostAlloc</code> ở Phụ lục C).
              </p>
            }
          >
            <p>
              <b>In the PDF</b> the appendix says "we mentioned" that the flag's other values are for advanced use. We found
              no earlier mention in this edition (a search of the text finds <code>cudaHostAlloc</code> only in Appendix C).
            </p>
          </Translated>
        </Aside>
      </NoteGroup>
      <Translated
        vi={
          <p>
            Từ đó đến nay công thức đã gọn lại. NVIDIA giờ gọi loại này là <b>bộ nhớ ánh xạ (mapped memory)</b>. Với định
            địa chỉ hợp nhất (unified addressing), thứ mà mọi tiến trình 64-bit đều có (mục C.3), bộ nhớ từ{' '}
            <code>cudaMallocHost</code> hoặc <code>cudaHostAlloc</code> dùng được trong kernel qua chính con trỏ của host,
            dù có cờ Mapped hay không. Bước hai con trỏ chỉ còn lại cho bộ nhớ được đăng ký bằng{' '}
            <code>cudaHostRegister</code> và cho bộ nhớ write-combined.
          </p>
        }
      >
        <p>
          Since then the recipe has shrunk. NVIDIA now calls this <b>mapped memory</b>. Under unified addressing, which
          every 64-bit process has (section C.3), memory from <code>cudaMallocHost</code> or <code>cudaHostAlloc</code>{' '}
          works in kernels through the host's own pointer, with or without the Mapped flag. The two-pointer step survives
          only for memory registered with <code>cudaHostRegister</code> and for write-combined memory.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 against the CUDA 13.5 runtime reference and guide (notes/today/appC-facts.md, A3)',
          '2026-11-20 đối chiếu với tài liệu tham chiếu và hướng dẫn runtime CUDA 13.5 (notes/today/appC-facts.md, A3)',
        )}
      >
        <Translated
          vi={
            <p>
              CUDA 2.2 ra đời tháng 5 năm 2009, đúng như sách nói. NVIDIA thêm một cảnh báo mà sách không có: các phép
              toán nguyên tử (atomic) trên bộ nhớ ánh xạ "không nguyên tử khi nhìn từ phía host hoặc các GPU khác".
            </p>
          }
        >
          <p>
            CUDA 2.2 is from May 2009, as the book says. NVIDIA adds a warning the book does not: atomics on mapped memory
            "are not atomic from the point of view of the host or other GPUs".
          </p>
        </Translated>
      </Today>
      <Ask q={questionsSC2['appC/C.2/q-today-pointer']} />

      <h2>
        <Translated vi="Một lần đọc qua đường truyền tốn bao nhiêu">What a read over the link costs</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bộ nhớ ánh xạ phải trả độ trễ dài và băng thông hạn chế của kênh kết nối ở mọi lần truy cập. Quy tắc kinh
            nghiệm của sách: kênh kết nối có băng thông "thường dưới 10%" băng thông bộ nhớ toàn cục. Chương 5 đã cho thấy
            một kernel không đủ tái sử dụng sẽ chạy ở tốc độ của bộ nhớ của nó. Nếu giờ phần lớn số byte của nó đến qua
            đường truyền, thì đường truyền chính là bộ nhớ của nó, và nó chạy ở một phần mười tốc độ hoặc chậm hơn.
          </p>
        }
      >
        <p>
          Mapped memory pays the interconnect's long latency and limited bandwidth on every access. The book's rule of
          thumb: the interconnect has "typically less than 10%" of global-memory bandwidth. Chapter 5 showed that a kernel
          without enough reuse runs at the speed of its memory. If most of its bytes now come over the link, the link is
          its memory, and it runs at a tenth of the speed or less.
        </p>
      </Translated>
      <NoteGroup>
        <Source section="C.2" pages="625" />
        <Today
          checked={tr(
            "2026-11-20 against NVIDIA's H100 and Grace Hopper pages (notes/today/appC-facts.md, A4)",
            '2026-11-20 đối chiếu với các trang H100 và Grace Hopper của NVIDIA (notes/today/appC-facts.md, A4)',
          )}
        >
          <Translated
            vi={
              <p>
                Đúng với PCIe. Trên thiết bị trong ví dụ của sách, H100 SXM, 128 GB/s của PCIe Gen5 so với 3.35 TB/s của
                bộ nhớ device là khoảng 3.8% (phép tính của chúng tôi). Các hệ thống NVLink-C2C của NVIDIA thu hẹp khoảng
                cách này rất nhiều. NVIDIA không còn nêu một tỷ lệ nào: hãy đo của bạn.
              </p>
            }
          >
            <p>
              True for PCIe. On the book's example device, H100 SXM, 128 GB/s of PCIe Gen5 against 3.35 TB/s of device
              memory is about 3.8% (our arithmetic). NVIDIA's NVLink-C2C systems narrow the gap a lot. NVIDIA no longer
              states a ratio: measure yours.
            </p>
          </Translated>
        </Today>
      </NoteGroup>
      <Translated
        vi={
          <p>
            Kernel hỗn hợp mới là trường hợp thực tế: phần lớn số byte từ bộ nhớ device, một ít từ host. Hãy tính bài này
            trước khi chạm vào trình khám phá, rồi kiểm tra lại ở đó với con số 10% của sách.
          </p>
        }
      >
        <p>
          A mixed kernel is the realistic case: most bytes from device memory, a few from the host. Work this one out
          before you touch the explorer, then check it there with the book's 10%.
        </p>
      </Translated>
      <Ask q={questionsSC2['appC/C.2/q-mixed']} />
      <Translated
        vi={
          <p>
            Trình khám phá chạy cùng phép tính đó cho bất kỳ đường truyền nào. Hãy thử thiết bị trong ví dụ của sách và,
            khi lab C.1 đã chạy trên máy CUDA hoặc HIP của bạn, đường truyền của chính bạn. Trên biểu đồ thứ hai, hãy xem
            hai đường cắt nhau ở đâu khi bạn làm mỗi lượt thưa hơn.
          </p>
        }
      >
        <p>
          The explorer runs the same arithmetic for any link. Try the book's example device and, once lab C.1 has run on
          your CUDA or HIP machine, your own link. On the second plot, watch where the two lines cross as you make each pass
          sparser.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Chỉ tính băng thông, không tính độ trễ. Byte được tính như được chuyển, trong khi đường truyền chuyển từng
              mảnh có kích thước cố định, nên các lần đọc thưa tốn hơn số byte của chúng. Không có bộ nhớ đệm (cache) cho
              dữ liệu ánh xạ trên GPU, và lần sao chép không được chồng lấp với tính toán. "Của bạn" dùng băng thông sao
              chép thay cho các lần đọc bên trong kernel.
            </p>
          }
        >
          <p>
            Bandwidth only, no latency. Bytes as moved, whereas a link moves fixed-size pieces, so sparse reads cost more
            than their bytes. No caching of mapped data on the GPU, and the copy is not overlapped with compute. "Yours"
            uses copy bandwidths as stand-ins for reads inside a kernel.
          </p>
        </Translated>
      </Simplification>
      <MappedCost />
      <Translated
        vi={
          <p>
            Có hai điều cần rút ra từ biểu đồ thứ nhất. Một tỷ lệ ánh xạ nhỏ đã đủ chi phối một kernel bị giới hạn bởi bộ
            nhớ: với 10% của sách, năm phần trăm số byte làm thời gian chạy tăng thêm gần một nửa. Và chồng lấp không cứu
            được bạn một khi luồng dữ liệu qua đường truyền là luồng dài hơn.
          </p>
        }
      >
        <p>
          Two things to take from the first plot. A small mapped fraction already dominates a memory-bound kernel: at the
          book's 10%, five percent of the bytes adds almost half the run time again. And overlap does not rescue you once the
          link stream is the longer one.
        </p>
      </Translated>

      <h2>
        <Translated vi={'"Thỉnh thoảng và thưa thớt", nói cho chính xác'}>"Occasionally and sparsely", made exact</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lời khuyên của sách là chỉ dùng bộ nhớ ánh xạ cho dữ liệu mà kernel chạm tới thỉnh thoảng và thưa thớt. Biểu
            đồ thứ hai biến điều đó thành một con số. Đọc tại chỗ tốn số byte bạn chạm tới, ở tốc độ đường truyền, mỗi
            lần. Sao chép trước tốn toàn bộ bảng một lần ở tốc độ đường truyền, rồi mỗi lần chạm ở tốc độ device. Đọc tại
            chỗ thắng đúng khi số byte kernel đọc trong suốt vòng đời của nó còn dưới 1/(1 − ρ) bảng: nhỉnh hơn một bảng dữ
            liệu một chút.
          </p>
        }
      >
        <p>
          The book's advice is to use mapped memory only for data a kernel touches occasionally and sparsely. The second
          plot turns that into a number. Reading in place costs the bytes you touch, at link speed, every time. Copying
          first costs the whole table once at link speed, then each touch at device speed. Reading in place wins exactly
          while the bytes the kernel reads over its whole life stay below 1/(1 − ρ) tables: a little more than one
          table's worth.
        </p>
      </Translated>
      <NoteGroup>
        <Source section="C.2" pages="625" />
        <Beyond>
          <Translated
            vi={
              <p>
                Quy tắc hòa vốn là phép suy ra của chúng tôi từ lời khuyên của sách, trong mô hình chỉ tính băng thông của
                trình khám phá.
              </p>
            }
          >
            <p>
              The break-even rule is our derivation from the book's advice, in the bandwidth-only model of the explorer.
            </p>
          </Translated>
        </Beyond>
      </NoteGroup>
      <Translated
        vi={
          <p>
            Một lần tra embedding, bước đầu tiên của transformer theo mô tả của sách (ch. 20, p. 480), là ví dụ kinh điển
            của đọc thưa. Hãy chốt một con số:
          </p>
        }
      >
        <p>
          An embedding lookup, the book's description of the first step of a transformer (ch. 20, p. 480), is the
          textbook sparse read. Commit to a number:
        </p>
      </Translated>
      <Predict
        id="appC/C.2/p-embedding"
        prompt={
          <Translated
            vi={
              <>
                Một bảng embedding 1 GiB nằm trong bộ nhớ host. Một bước tra 64 hàng, mỗi hàng 16 KiB. Số byte đi qua đường
                truyền ít hơn bao nhiêu lần nếu kernel đọc các hàng tại chỗ so với khi bạn sao chép bảng trước?
              </>
            }
          >
            <>
              An embedding table of 1 GiB sits in host memory. One step looks up 64 rows of 16 KiB each. How many times fewer
              bytes cross the link if the kernel reads the rows in place than if you copy the table first?
            </>
          </Translated>
        }
        unit="×"
      >
        {(v) => (
          <Translated
            vi={
              <>
                <p>
                  {v === 1024 ? 'Đúng: ' : 'Đáp án là 1024: '}2<sup>30</sup> byte của bảng so với 64 × 2
                  <sup>14</sup> = 2<sup>20</sup> byte của các hàng.
                </p>
                <p>
                  Một bước chưa phải là toàn bộ câu chuyện. Nếu các bước sau chạm tới những hàng khác, thì sau khoảng một
                  nghìn bước kernel đã đọc một bảng dữ liệu qua đường truyền, và một lần sao chép duy nhất lẽ ra đã rẻ hơn.
                  Câu hỏi tiếp theo đặt một con số cho điều đó.
                </p>
              </>
            }
          >
            <>
              <p>
                {v === 1024 ? 'Right: ' : 'The answer is 1024: '}2<sup>30</sup> bytes of table against 64 × 2
                <sup>14</sup> = 2<sup>20</sup> bytes of rows.
              </p>
              <p>
                One step is not the whole story. If later steps touch other rows, then after about a thousand steps the
                kernel has read a table's worth over the link, and a single copy would have been cheaper. The next
                question puts a number on that.
              </p>
            </>
          </Translated>
        )}
      </Predict>
      <Ask q={questionsSC2['appC/C.2/q-breakeven']} />

      <h2>
        <Translated vi="Bạn đã gặp bộ nhớ ghim ở đâu">Where you have met pinned memory</Translated>
      </h2>
      <Translated
        vi={
          <p>
            <code>DataLoader(pin_memory=True)</code> và <code>tensor.pin_memory()</code> cấp phát đúng loại bộ nhớ này:
            caching host allocator của PyTorch gọi <code>cudaHostAlloc</code> với <code>cudaHostAllocDefault</code>. Điều
            PyTorch muốn ở nó là sao chép nhanh hơn, và các lần sao chép có thể chạy bất đồng bộ với{' '}
            <code>non_blocking=True</code>. Nó vẫn sao chép từng batch lên GPU; không có gì đọc batch tại chỗ.
          </p>
        }
      >
        <p>
          <code>DataLoader(pin_memory=True)</code> and <code>tensor.pin_memory()</code> allocate exactly this kind of
          memory: PyTorch's caching host allocator calls <code>cudaHostAlloc</code> with <code>cudaHostAllocDefault</code>.
          What PyTorch wants from it is faster copies, and copies that can run asynchronously with{' '}
          <code>non_blocking=True</code>. It still copies each batch to the GPU; nothing reads the batch in place.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Mã nguồn và ghi chú CUDA của PyTorch, đọc ngày 2026-11-20 (appC-facts D1).</p>}>
          <p>PyTorch's source and CUDA notes, read on 2026-11-20 (appC-facts D1).</p>
        </Translated>
      </Beyond>
      <Ask q={questionsSC2['appC/C.2/q-pinmemory']} />

      <h2>
        <Translated vi="HIP và Metal">HIP and Metal</Translated>
      </h2>
      <Translated
        vi={
          <p>
            HIP có cùng hình dạng: <code>hipHostMalloc</code> với <code>hipHostMallocMapped</code>, và{' '}
            <code>hipHostGetDevicePointer</code> cho giá trị phía device. AMD thêm một lựa chọn mà CUDA không có, tính nhất
            quán (coherence). Bộ nhớ coarse-grained chỉ được cập nhật sau một lần đồng bộ hóa; bộ nhớ fine-grained giữ nhất
            quán trong khi bên nào thay đổi nó cũng vậy, thường bằng cách không đưa nó vào bộ nhớ đệm trên GPU. Các cờ là{' '}
            <code>hipHostMallocCoherent</code> và <code>hipHostMallocNonCoherent</code>, hoặc biến môi trường{' '}
            <code>HIP_HOST_COHERENT</code>, và bộ nhớ Mapped luôn là fine-grained. Còn mặc định bạn nhận được loại nào, chính
            trang của AMD nêu theo cả hai cách, nên hãy đặt nó một cách tường minh.
          </p>
        }
      >
        <p>
          HIP has the same shape: <code>hipHostMalloc</code> with <code>hipHostMallocMapped</code>, and{' '}
          <code>hipHostGetDevicePointer</code> for the device-side value. AMD adds a choice CUDA does not have, coherence.
          Coarse-grained memory is up to date only after a synchronisation; fine-grained memory stays coherent while
          either side changes it, often by leaving it uncached on the GPU. The flags are{' '}
          <code>hipHostMallocCoherent</code> and <code>hipHostMallocNonCoherent</code>, or the environment variable{' '}
          <code>HIP_HOST_COHERENT</code>, and Mapped memory is always fine-grained. Which one you get by default, AMD's own
          page states both ways, so set it explicitly.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Các trang về bộ nhớ host và tính nhất quán của HIP do AMD viết, ROCm 10.2 (appC-facts B2).</p>}>
          <p>AMD's HIP host-memory and coherence pages, ROCm 10.2 (appC-facts B2).</p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Trên Apple silicon không có kênh kết nối nào giữa bộ nhớ CPU và bộ nhớ GPU để phải trả giá, vì chỉ có một bộ
            nhớ. Một buffer shared, mặc định ở đó, được mọi kernel đọc tại chỗ, và Apple không ghi nhận sự phân biệt nào
            giữa bộ nhớ ghim và bộ nhớ phân trang. Để dùng bộ nhớ bạn đã có, <code>newBufferWithBytesNoCopy</code> bọc vùng
            nhớ căn theo trang thành một buffer. llama.cpp làm đúng như vậy với trọng số mô hình được ánh xạ bộ nhớ: zero
            copy, áp dụng cho trọng số, trên một bộ nhớ duy nhất.
          </p>
        }
      >
        <p>
          On Apple silicon there is no interconnect between CPU memory and GPU memory to pay for, because there is one
          memory. A shared buffer, the default there, is read in place by every kernel, and Apple documents no distinction
          between pinned and pageable memory. To use memory you already have, <code>newBufferWithBytesNoCopy</code> wraps
          page-aligned memory as a buffer. llama.cpp does exactly that with memory-mapped model weights: zero copy, applied
          to the weights, on one memory.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Tài liệu của Apple (appC-facts C2; infra-facts C12; ch02-facts C2) và backend Metal của llama.cpp, đọc ngày
              2026-11-20 (appC-facts D4).
            </p>
          }
        >
          <p>
            Apple's documentation (appC-facts C2; infra-facts C12; ch02-facts C2) and llama.cpp's Metal backend, read on
            2026-11-20 (appC-facts D4).
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            <a href={href('appC', 'labs')}>Lab C.1</a> cấp dữ liệu cho một kernel theo ba cách trên mỗi backend, trong đó có
            bộ nhớ ánh xạ, và hàng probe của nó là nơi "của bạn" trong trình khám phá được lấy ra. Quay lại{' '}
            <a href={href('appC', 'start')}>bản đồ</a>: bước này là C.2.
          </p>
        }
      >
        <p>
          <a href={href('appC', 'labs')}>Lab C.1</a> feeds one kernel three ways on each backend, mapped memory among them,
          and its probe row is where "yours" in the explorer comes from. Back to <a href={href('appC', 'start')}>the map</a>
          : this step is C.2.
        </p>
      </Translated>
    </>
  );
}
