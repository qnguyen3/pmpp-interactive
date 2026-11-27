import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Bench, Beyond, NoteGroup as Notes, Simplification, Source, Today, type Tier } from '../../components/notes';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { fmt, realBackends, useReports, type BackendRun, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BARRIER,
  FINISH,
  INDEX_LABEL,
  JITTER,
  LOCAL,
  MP_TABLE,
  ORDER_LABEL,
  PHASE_LABEL,
  PROGRESS_LABEL,
  SCHEME_LABEL,
  METAL_NOTES,
  messagePassing,
  rows as rowsOf,
  simulate,
  type BlockIndex,
  type LoadOrder,
  type LookbackConfig,
  type LookbackRun,
  type Phase,
  type Progress,
  type Scheme,
  type Scope,
  type StartOrder,
  type StoreOrder,
} from '../../sim/lookback';
import {
  BOOK_DEVICE,
  BOOK_PRESET,
  DESIGNS,
  DESIGN_LABEL,
  ceiling,
  designCeilings,
  designTraffic,
  labRows,
  measuredBandwidth,
  rssOverSsa,
  type Design,
  type RunLike,
} from '../../sim/scantraffic';
import { LookbackScene } from './LookbackScene';

// Section 11.9: stitching block scans into a global scan. Bytes per element of each design
// (src/sim/scantraffic.ts), blocks waiting for blocks and the flag protocol
// (src/sim/lookback.ts), the narrated scene, and lab 11.5's single-pass kernel.

const LAB4 = 'ch11/lab4_';
/** Lab 11.5's reference solution (CUDA family only). */
export const LAB5_FILE = 'ch11/lab5_singlepass/solution/kernel.cu';
const fmtG = (elementsPerSec: number) => fmt(elementsPerSec / 1e9);

const frac = (S: number) => (1 / S).toPrecision(2);

// ---------------------------------------------------------------- devices
interface Device {
  key: string;
  label: string;
  bw: number;
  note: string;
}
const BOOK: Device = {
  key: 'book',
  get label() {
    return tr("The book's example (H100 SXM, 3.35 TB/s)", 'Ví dụ của sách (H100 SXM, 3.35 TB/s)');
  },
  bw: BOOK_DEVICE.bandwidthGBs,
  get note() {
    return tr("the book's example device (3.35 TB/s, p. 272)", 'thiết bị ví dụ của sách (3.35 TB/s, p. 272)');
  },
};
function devicesOf(reports: Reports): Device[] {
  return [
    BOOK,
    ...realBackends(reports).flatMap((b) => {
      const m = measuredBandwidth(reports[b] as BackendRun & RunLike);
      return m
        ? [{ key: b, label: tr(`Yours: ${String(reports[b].device.name ?? b)} (${b}), ${fmt(m.gbs)} GB/s`, `Của bạn: ${String(reports[b].device.name ?? b)} (${b}), ${fmt(m.gbs)} GB/s`), bw: m.gbs, note: tr(`${m.source} on ${reports[b].generated.slice(0, 10)}`, `${m.source} ngày ${reports[b].generated.slice(0, 10)}`) }]
        : [];
    }),
  ];
}

// ---------------------------------------------------------------- explorer: bytes per element
const SEGMENTS = [256, 512, 1024, 2048, 4096, 8192];

function BytesExplorer() {
  const { reports } = useReports();
  const locale = useLocale();
  const devices = useMemo(() => devicesOf(reports), [reports, locale]);
  const [design, setDesign] = useState<Design>('ssa');
  const [S, setS] = useState(1024);
  const [elem, setElem] = useState(4);
  const [key, setKey] = useState('book');
  const [extra, setExtra] = useState(false);
  const dev = devices.find((d) => d.key === key) ?? BOOK;
  const t = designTraffic(design, S, elem, extra);
  const caps = designCeilings(dev.bw, S, elem, extra);
  const preset = () => {
    setKey('book');
    setS(BOOK_PRESET.S);
    setElem(4);
    setExtra(BOOK_PRESET.bookkeeping);
  };
  const isPreset = key === 'book' && S === BOOK_PRESET.S && elem === 4 && !extra;
  const per = (x: number) => (x === 1 ? 'N' : `N/S`);
  const maxBytes = Math.max(...DESIGNS.map((d) => designTraffic(d, S, elem, extra).bytesPerElement));

  return (
    <Bench
      title={tr('Bytes per element', 'Số byte mỗi phần tử')}
      task={
        <Translated vi="Bấm vào các số của sách, rồi so sánh hai thiết kế ba kernel theo từng kernel. Reduce-scan-scan tránh được những lần truy cập nào?">
          Press the book's numbers, then compare the two three-kernel designs kernel by kernel. Which accesses does reduce-scan-scan avoid?
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Mọi lần truy cập đều tính vào bộ nhớ toàn cục; S là đoạn mà một khối scan. Sách chỉ đếm các lần truy cập in chữ
              thường; công tắc thêm vào phần ghi chép theo từng khối mà sách bỏ qua (của khóa học này). Băng thông: {dev.note}.
            </>
          }
        >
          <>
            Every access charged to global memory; S is the segment one block scans. The book counts only the accesses in plain
            type; the switch adds the per-block bookkeeping it leaves out (ours). Bandwidth: {dev.note}.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('Design', 'Thiết kế')}>
          {DESIGNS.map((d) => (
            <button key={d} aria-pressed={design === d} onClick={() => setDesign(d)}>{DESIGN_LABEL[d]}</button>
          ))}
        </span>
        <button className={isPreset ? 'quiet' : 'primary'} onClick={preset}><Translated vi="Các số của sách">The book's numbers</Translated></button>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          <Translated vi="Đoạn S (số phần tử mỗi khối)">Segment S (elements per block)</Translated>
          <select value={S} onChange={(e) => setS(+e.target.value)} aria-label={tr('segment size', 'kích thước đoạn')}>
            {SEGMENTS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Kích thước phần tử">Element size</Translated>
          <span className="seg" role="group" aria-label={tr('Element size', 'Kích thước phần tử')}>
            {[4, 2].map((e) => (
              <button key={e} aria-pressed={elem === e} onClick={() => setElem(e)}>{e === 4 ? tr('float, 4 bytes', 'float, 4 byte') : tr('half, 2 bytes', 'half, 2 byte')}</button>
            ))}
          </span>
        </label>
        <label className="field">
          <Translated vi="Băng thông">Bandwidth</Translated>
          <select value={key} onChange={(e) => setKey(e.target.value)} aria-label={tr('bandwidth', 'băng thông')}>
            {devices.map((d) => <option key={d.key} value={d.key}>{d.label}</option>)}
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={extra} onChange={(e) => setExtra(e.target.checked)} /> <Translated vi="tính cả phần ghi chép theo từng khối (của khóa học này)">count the per-block bookkeeping (ours)</Translated>
        </label>
      </div>

      <div style={{ overflowX: 'auto', marginBottom: '0.8rem' }}>
      <table className="data" style={{ margin: 0, minWidth: 600 }} aria-label={tr('Kernels of this design', 'Các kernel của thiết kế này')}>
        <thead>
          <tr>
            <th><Translated vi="Kernel">Kernel</Translated></th>
            <th><Translated vi="Làm gì">Does</Translated></th>
            <th><Translated vi="Nạp">Loads</Translated></th>
            <th><Translated vi="Ghi">Stores</Translated></th>
            <th className="num"><Translated vi="Byte mỗi phần tử">Bytes per element</Translated></th>
          </tr>
        </thead>
        <tbody>
          {t.kernels.map((k) => {
            const shown = k.accesses;
            const sum = shown.filter((a) => extra || !a.bookkeeping).reduce((s, a) => s + a.perElement * a.bytes, 0);
            const list = (kind: 'load' | 'store') =>
              shown.filter((a) => a.kind === kind).map((a, i) => (
                <div key={i} className={a.bookkeeping && !extra ? 'muted' : ''} style={a.bookkeeping && !extra ? { textDecoration: 'line-through' } : undefined}>
                  <span className="t-global">{a.what}</span>, {a.bookkeeping && a.bytes === 4 && design === 'single' ? tr('1 per block', '1 mỗi khối') : per(a.perElement)}
                  {a.bookkeeping ? <span className="muted"> <Translated vi="(ghi chép)">(bookkeeping)</Translated></span> : null}
                </div>
              ));
            return (
              <tr key={k.kernel} data-kernel={k.kernel}>
                <td>{tr(k.kernel, k.kernel === 'one kernel' ? 'một kernel' : k.kernel)}</td>
                <td>{k.does}</td>
                <td>{list('load')}</td>
                <td>{list('store')}</td>
                <td className="num" data-bytes={sum.toFixed(6)}>{fmt(sum, 5)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      </div>

      <div className="cols">
        <div className="panel" data-testid="total">
          <Translated
            vi={
              <p style={{ margin: 0 }}>
                {DESIGN_LABEL[design]}, S = {S}:{' '}
                <b data-testid="bpe">{fmt(t.bytesPerElement, 6)}</b> byte mỗi phần tử
                {extra ? null : <> = {t.formula.replace('S', String(S))}</>}.
              </p>
            }
          >
            <p style={{ margin: 0 }}>
              {DESIGN_LABEL[design]}, S = {S}:{' '}
              <b data-testid="bpe">{fmt(t.bytesPerElement, 6)}</b> bytes per element
              {extra ? null : <> = {t.formula.replace('S', String(S))}</>}.
            </p>
          </Translated>
          <p style={{ fontSize: '1.4rem', margin: '0.5rem 0 0.2rem', fontWeight: 600 }}>
            <span data-testid="cap">{fmtG(ceiling(dev.bw, t.bytesPerElement))}</span> <Translated vi="× 10⁹ phần tử/s">× 10⁹ elements/s</Translated>
          </p>
          <Translated
            vi={
              <p className="muted" style={{ margin: 0 }}>
                {fmt(dev.bw)} GB/s ÷ {fmt(t.bytesPerElement, 5)} byte. S chỉ đi vào qua các số hạng 1/S (ở đây là {frac(S)}), vì
                vậy các đoạn lớn làm cho tổng của khối gần như miễn phí.
              </p>
            }
          >
            <p className="muted" style={{ margin: 0 }}>
              {fmt(dev.bw)} GB/s ÷ {fmt(t.bytesPerElement, 5)} bytes. S enters only through the 1/S terms ({frac(S)} here), which is
              why large segments make the block sums almost free.
            </p>
          </Translated>
        </div>
        <div className="panel">
          <svg viewBox="0 0 360 132" role="img" aria-label={tr('The three designs side by side', 'Ba thiết kế đặt cạnh nhau')} style={{ width: '100%', maxWidth: 360, display: 'block' }}>
            {DESIGNS.map((d, i) => {
              const b = designTraffic(d, S, elem, extra).bytesPerElement;
              const y = 10 + i * 40;
              const w = (b / maxBytes) * 170;
              return (
                <g key={d} data-design={d} data-ceiling={(caps[d] / 1e9).toFixed(3)} opacity={d === design ? 1 : 0.55}>
                  <text x="0" y={y + 9} fontSize="11" fontWeight={d === design ? 600 : 400}>{DESIGN_LABEL[d].split(' (')[0]}</text>
                  <rect x="0" y={y + 14} width={w} height="12" fill="var(--global)" />
                  <text x={w + 6} y={y + 24} fontSize="11">{fmt(b, 4)} B</text>
                  <text x="360" y={y + 24} fontSize="11" textAnchor="end" fontWeight="600">{fmtG(caps[d])} × 10⁹/s</text>
                </g>
              );
            })}
          </svg>
          <Translated
            vi={
              <p className="muted" style={{ margin: '0.3rem 0 0' }}>
                Scan-scan-add đạt {Math.round((100 * caps.ssa) / caps.single)}% của single pass, reduce-scan-scan{' '}
                {Math.round((100 * caps.rss) / caps.single)}%.
              </p>
            }
          >
            <p className="muted" style={{ margin: '0.3rem 0 0' }}>
              Scan-scan-add reaches {Math.round((100 * caps.ssa) / caps.single)}% of the single pass, reduce-scan-scan{' '}
              {Math.round((100 * caps.rss) / caps.single)}%.
            </p>
          </Translated>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- measured reveal (lab 11.4)
function RssReveal({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const found = realBackends(reports).flatMap((b) => {
    const r = rssOverSsa(labRows(reports[b] as BackendRun & RunLike, LAB4));
    return r ? [{ b, device: String(reports[b].device.name ?? b), ...r }] : [];
  });
  if (!found.length)
    return (
      <Translated
        vi={
          <p>
            Phép đếm byte dự đoán 16/12 ≈ 1.33. Máy này chưa có số đo nào: hãy chạy{' '}
            <code>python3 labs/pmpp.py run 11.4 --solution --bench</code> trên một máy có GPU, rồi tải lại.
          </p>
        }
      >
        <p>
          The byte count predicts 16/12 ≈ 1.33. No measurement on this machine yet: run{' '}
          <code>python3 labs/pmpp.py run 11.4 --solution --bench</code> on a machine with a GPU, then reload.
        </p>
      </Translated>
    );
  return (
    <>
      {found.map((f) => (
        <Translated
          key={f.b}
          vi={
            <p data-backend={f.b} data-ratio={f.ratio.toFixed(4)}>
              Trên <b>{f.device}</b> ({f.b}), với n = {f.n.toLocaleString('en-US')}: reduce-scan-scan chạy nhanh gấp{' '}
              <b>{fmt(f.ratio)}×</b> scan-scan-add ({fmtG(f.rss)} so với {fmtG(f.ssa)} × 10⁹ phần tử/s). Phép đếm byte nói 1.33;
              bạn nói {predicted}.{' '}
              {f.ratio < 1.15
                ? 'Ít hơn nhiều so với những gì số byte hứa hẹn: một thứ khác ngoài lưu lượng, như chi phí khởi chạy hay một phép rút gọn chậm hơn, chiếm một phần thời gian.'
                : f.ratio > 1.45
                  ? 'Nhiều hơn những gì số byte hứa hẹn: các kernel scan-scan-add mất thêm thứ gì đó ngoài số byte thừa của chúng trên thiết bị này.'
                  : 'Gần với những gì số byte hứa hẹn: lưu lượng là phần lớn câu chuyện ở đây.'}
            </p>
          }
        >
          <p key={f.b} data-backend={f.b} data-ratio={f.ratio.toFixed(4)}>
            On <b>{f.device}</b> ({f.b}), at n = {f.n.toLocaleString('en-US')}: reduce-scan-scan ran <b>{fmt(f.ratio)}×</b> as fast
            as scan-scan-add ({fmtG(f.rss)} against {fmtG(f.ssa)} × 10⁹ elements/s). The byte count says 1.33; you said {predicted}.{' '}
            {f.ratio < 1.15
              ? 'Much less than the bytes promise: something other than traffic, such as launch overhead or a slower reduction, takes part of the time.'
              : f.ratio > 1.45
                ? 'More than the bytes promise: the scan-scan-add kernels lose something beyond their extra bytes on this device.'
                : 'Close to what the bytes promise: traffic is most of the story here.'}
          </p>
        </Translated>
      ))}
    </>
  );
}

// ---------------------------------------------------------------- simulator: who waits for whom
const PH_FILL: Record<Phase, string> = {
  local: 'var(--ink-3)',
  look: 'var(--ink)',
  spin: 'url(#ww-spin)',
  barrier: 'url(#ww-bar)',
  scan: 'var(--ink)',
  finish: 'var(--ink-2)',
  starved: 'none',
};

/** True on a phone-width screen, where the timeline and its table switch to a compact layout. */
function useNarrow(query = '(max-width: 640px)'): boolean { // i18n-ignore: CSS media query
  const matches = () => typeof window !== 'undefined' && typeof window.matchMedia === 'function' && window.matchMedia(query).matches;
  const [narrow, setNarrow] = useState(matches);
  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return;
    const m = window.matchMedia(query);
    const on = () => setNarrow(m.matches);
    on();
    m.addEventListener('change', on);
    return () => m.removeEventListener('change', on);
  }, [query]);
  return narrow;
}

function Timeline({ run, narrow }: { run: LookbackRun; narrow: boolean }) {
  const rs = rowsOf(run);
  const G = run.config.G;
  const RH = G > 16 ? 15 : G > 8 ? 18 : 24;
  // On a phone the drawing keeps its full width on screen: a short label gutter ("i4 b7" for
  // index 4 on blockIdx.x 7) and a narrower canvas, so nothing has to scroll sideways.
  const W = narrow ? 360 : 680;
  const L = narrow ? 62 : 150;
  const right = narrow ? 18 : 50;
  const top = 18;
  const T = Math.max(run.time + (run.outcome === 'done' ? 1 : 4), 10);
  const X = (t: number) => L + (t / T) * (W - L - right);
  const H = top + rs.length * RH + 22;
  const counter = run.config.index === 'counter';
  const step = [2, 5, 10, 20, 50, 100].find((x) => T / x <= (narrow ? 5 : 8)) ?? 200;
  const ticks = Array.from({ length: Math.floor(T / step) + 1 }, (_, i) => i * step);
  const label = (b: { hw: number; idx: number | null; start: number | null }) =>
    narrow
      ? counter ? (b.start != null ? `i${b.idx} b${b.hw}` : tr(`b${b.hw} queued`, `b${b.hw} chờ`)) : `b${b.hw}`
      : counter ? (b.start != null ? tr(`index ${b.idx} (blockIdx ${b.hw})`, `chỉ số ${b.idx} (blockIdx ${b.hw})`) : tr(`blockIdx ${b.hw}, not started`, `blockIdx ${b.hw}, chưa bắt đầu`)) : tr(`block ${b.hw}`, `khối ${b.hw}`);
  const stuckX = X(run.time + 1);
  const stuckLeft = stuckX > W / 2;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Timeline of every block', 'Dòng thời gian của mọi khối')} data-layout={narrow ? 'narrow' : 'wide'}
      style={{ width: '100%', minWidth: narrow ? 0 : 600, display: 'block' }}>
      <defs>
        <pattern id="ww-spin" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="6" height="6" fill="var(--global-wash)" />
          <line x1="0" y1="0" x2="0" y2="6" stroke="var(--global)" strokeWidth="2" />
        </pattern>
        <pattern id="ww-bar" width="6" height="6" patternUnits="userSpaceOnUse">
          <rect width="6" height="6" fill="var(--rule-soft)" />
          <line x1="0" y1="3" x2="6" y2="3" stroke="var(--ink-2)" strokeWidth="1.4" />
        </pattern>
      </defs>
      {ticks.map((tk) => (
        <g key={tk}>
          <line x1={X(tk)} x2={X(tk)} y1={top - 4} y2={top + rs.length * RH} stroke="var(--rule-soft)" />
          <text x={X(tk)} y={top + rs.length * RH + 13} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{tk}</text>
        </g>
      ))}
      <text x={narrow ? L - 8 : W - 4} y={top + rs.length * RH + 13} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-3)' }}><Translated vi="vòng">rounds</Translated></text>
      {run.outcome === 'deadlock' || run.outcome === 'stall' ? (
        <g data-stuck={run.time} data-x={stuckX.toFixed(1)} data-w={W}>
          <line x1={stuckX} x2={stuckX} y1={top - 10} y2={top + rs.length * RH} stroke="var(--fail)" strokeWidth="2" />
          <text x={stuckLeft ? stuckX - 4 : stuckX + 4} y={top - 4} fontSize="10" textAnchor={stuckLeft ? 'end' : 'start'} style={{ fill: 'var(--fail)' }}>
            {narrow ? tr('stuck', 'kẹt') : tr('stuck: nothing can move', 'kẹt: không gì có thể nhúc nhích')}
          </text>
        </g>
      ) : null}
      {rs.map((b, k) => {
        const y = top + k * RH;
        const started = b.start != null;
        return (
          <g key={b.hw} data-row={b.hw} data-idx={b.idx ?? ''} data-start={b.start ?? ''} data-end={b.end ?? ''}>
            <text x={L - 6} y={y + RH * 0.72} fontSize={RH < 16 ? 10.5 : 11.5} textAnchor="end">{label(b)}</text>
            {!started ? <rect x={L} y={y + 2} width={W - L - right} height={RH - 4} fill="none" stroke="var(--rule)" strokeDasharray="3 3" /> : null}
            {b.spans.map((s, i) =>
              s.phase === 'starved' ? (
                <rect key={i} x={X(s.from)} y={y + 2} width={X(s.to) - X(s.from)} height={RH - 4} fill="none" stroke="var(--fail)" strokeDasharray="3 2" />
              ) : (
                <rect key={i} x={X(s.from)} y={y + 2} width={X(s.to) - X(s.from)} height={RH - 4} fill={PH_FILL[s.phase]} opacity={s.phase === 'local' ? 0.6 : 1} />
              ),
            )}
            {b.end != null ? <text x={X(b.end) + 3} y={y + RH * 0.72} fontSize={RH < 16 ? 10 : 11} style={{ fill: 'var(--ok)' }}>✓</text> : null}
          </g>
        );
      })}
    </svg>
  );
}

/** Swatches drawn with the timeline's own fills, so the legend is right in both themes. */
function PhaseLegend() {
  const items: [string, ReactNode][] = [
    [PHASE_LABEL.local, <rect width="16" height="10" fill="var(--ink-3)" opacity="0.6" />],
    [PHASE_LABEL.look, <rect width="16" height="10" fill="var(--ink)" />],
    [PHASE_LABEL.finish, <rect width="16" height="10" fill="var(--ink-2)" />],
    [tr(`${PHASE_LABEL.spin} (global memory)`, `${PHASE_LABEL.spin} (bộ nhớ toàn cục)`), <rect width="16" height="10" fill="url(#ww-spin)" />],
    [PHASE_LABEL.barrier, <rect width="16" height="10" fill="url(#ww-bar)" />],
    [PHASE_LABEL.starved, <rect x="0.5" y="0.5" width="15" height="9" fill="none" stroke="var(--fail)" strokeDasharray="3 2" />],
  ];
  return (
    <p className="legend" style={{ margin: '0.5rem 0 0' }} data-testid="legend">
      {items.map(([label, sw]) => (
        <span key={label}>
          <svg width="16" height="10" style={{ marginRight: '0.35em', verticalAlign: '-0.05em' }} aria-hidden="true">{sw}</svg>
          {label}
        </span>
      ))}
    </p>
  );
}

/** The scheme table's columns: wide screens show one row per scheme, phones one column per scheme. */
const SCHEME_COLUMNS: { head: string; short: string; cell: (r: LookbackRun) => ReactNode }[] = [
  { head: 'Outcome', short: 'Outcome', cell: (r) => (r.outcome === 'done' ? tr('finished', 'xong') : r.outcome === 'stall' ? tr('stalled', 'bị treo') : <Translated vi={r.outcome === 'refused' ? 'bị từ chối' : r.outcome}>{r.outcome}</Translated>) }, // i18n-ignore: also the column key; translated where rendered
  { head: 'Rounds', short: 'Rounds', cell: (r) => (r.outcome === 'refused' ? '—' : r.outcome === 'done' ? r.time : tr(`stuck at ${r.time}`, `kẹt ở ${r.time}`)) }, // i18n-ignore: also the column key; translated where rendered
  { head: 'Longest chain', short: 'Longest chain', cell: (r) => (r.outcome !== 'done' ? '—' : r.config.scheme === 'barrier' ? tr('2 barriers', '2 rào chắn') : r.criticalPath) }, // i18n-ignore: also the column key; translated where rendered
  { head: 'Rounds spent waiting', short: 'Waiting', cell: (r) => (r.outcome === 'refused' ? '—' : r.waitTotal) }, // i18n-ignore: also the column key; translated where rendered
  { head: 'Additions (repeated)', short: 'Additions (repeated)', cell: (r) => (r.outcome !== 'done' ? '—' : `${r.additions} (${r.repeated})`) }, // i18n-ignore: also the column key; translated where rendered
];
const PRESETS: { readonly label: string; c: Partial<LookbackConfig> }[] = [
  {
    get label() {
      return tr('blockIdx.x, later blocks first', 'blockIdx.x, khối sau chạy trước');
    },
    c: { G: 8, R: 4, order: 'reversed', index: 'hardware', scheme: 'single', progress: 'started', even: false },
  },
  {
    get label() {
      return tr('the same with the counter', 'như trên nhưng dùng bộ đếm');
    },
    c: { G: 8, R: 4, order: 'reversed', index: 'counter', scheme: 'single', progress: 'started', even: false },
  },
  {
    get label() {
      return tr('no progress guarantee', 'không bảo đảm tiến triển');
    },
    c: { G: 6, R: 3, order: 'index', seed: 6, index: 'counter', scheme: 'single', progress: 'none', even: false },
  },
  {
    get label() {
      return tr('a long chain', 'một chuỗi dài');
    },
    c: { G: 32, R: 32, order: 'index', index: 'counter', scheme: 'single', progress: 'started', even: false },
  },
];

function WaitSimulator() {
  const [c, setC] = useState<LookbackConfig>({ G: 8, R: 4, order: 'reversed', seed: 1, index: 'hardware', scheme: 'single', progress: 'started', even: false });
  const set = (p: Partial<LookbackConfig>) => setC((o) => ({ ...o, ...p }));
  const locale = useLocale();
  const run = useMemo(() => simulate(c), [c, locale]);
  const all = useMemo(() => (['barrier', 'single', 'decoupled'] as Scheme[]).map((s) => simulate({ ...c, scheme: s })), [c, locale]);
  const narrow = useNarrow();
  const short: Record<Scheme, string> = { barrier: 'barrier', single: 'single', decoupled: 'decoupled' };
  const outcomeText: Record<string, ReactNode> = {
    done: <span className="chip ok"><Translated vi="mọi khối đều xong">every block finished</Translated></span>,
    deadlock: <span className="chip fail"><Translated vi="deadlock">deadlock</Translated></span>,
    stall: <span className="chip fail"><Translated vi="bị treo">stalled</Translated></span>,
    refused: <span className="chip fail"><Translated vi="lần khởi chạy bị từ chối">launch refused</Translated></span>,
  };
  return (
    <Bench
      title={tr('Who waits for whom', 'Ai chờ ai')}
      task={
        <Translated vi="Làm cho preset đầu tiên chạy xong chỉ với một thay đổi. Sau đó tìm thiết lập duy nhất mà ở đó bộ đếm không giúp được gì.">
          Make the first preset finish with one change. Then find the only setting under which the counter does not help.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Vòng là đơn vị thời gian trừu tượng (của khóa học này): một đoạn mất {LOCAL} đến {LOCAL + JITTER - 1} vòng (có
              seed), cộng và ghi mất {FINISH}, một lần đọc look-back mất 1, một rào chắn toàn lưới mất {BARRIER} sau lần đến
              cuối cùng.
            </>
          }
        >
          <>
            Rounds are abstract time units (ours): a segment takes {LOCAL} to {LOCAL + JITTER - 1} rounds (seeded), adding and
            storing {FINISH}, a look-back read 1, a grid barrier {BARRIER} after its last arrival.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <span className="muted"><Translated vi="Preset:">Presets:</Translated></span>
        {PRESETS.map((p) => (
          <button key={p.label} className="quiet" onClick={() => set(p.c)}>{p.label}</button>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <label className="field">
          <Translated vi={<span>Số khối G = <b data-testid="G">{c.G}</b></span>}><span>Blocks G = <b data-testid="G">{c.G}</b></span></Translated>
          <input type="range" min="2" max="32" value={c.G} aria-label={tr('blocks', 'số khối')} onChange={(e) => set({ G: +e.target.value })} />
        </label>
        <label className="field">
          <Translated vi={<span>Số suất thường trú R = <b data-testid="R">{c.R}</b></span>}><span>Resident slots R = <b data-testid="R">{c.R}</b></span></Translated>
          <input type="range" min="1" max="32" value={c.R} aria-label={tr('slots', 'số suất')} onChange={(e) => set({ R: +e.target.value })} />
        </label>
        <label className="field">
          <Translated vi="Thứ tự bắt đầu">Start order</Translated>
          <span className="seg" role="group" aria-label={tr('Start order', 'Thứ tự bắt đầu')}>
            {(['index', 'reversed', 'shuffle'] as StartOrder[]).map((o) => (
              <button key={o} aria-pressed={c.order === o} onClick={() => set({ order: o })}>{ORDER_LABEL[o]}</button>
            ))}
          </span>
        </label>
        {c.order === 'shuffle' ? (
          <button onClick={() => set({ seed: c.seed + 1 })}><Translated vi={<>Xáo trộn lại (seed {c.seed})</>}>New shuffle (seed {c.seed})</Translated></button>
        ) : null}
      </div>
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <label className="field">
          <Translated vi="Chỉ số khối">Block index</Translated>
          <span className="seg" role="group" aria-label={tr('Block index', 'Chỉ số khối')}>
            {(['hardware', 'counter'] as BlockIndex[]).map((x) => (
              <button key={x} aria-pressed={c.index === x} onClick={() => set({ index: x })}>{INDEX_LABEL[x]}</button>
            ))}
          </span>
        </label>
        <label className="field">
          <Translated vi="Phương án">Scheme</Translated>
          <span className="seg" role="group" aria-label={tr('Scheme', 'Phương án')}>
            {(['barrier', 'single', 'decoupled'] as Scheme[]).map((s) => (
              <button key={s} aria-pressed={c.scheme === s} onClick={() => set({ scheme: s })}>{SCHEME_LABEL[s]}</button>
            ))}
          </span>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          <Translated vi="Quy tắc tiến triển">Progress rule</Translated>
          <span className="seg" role="group" aria-label={tr('Progress rule', 'Quy tắc tiến triển')}>
            {(['started', 'none'] as Progress[]).map((p) => (
              <button key={p} aria-pressed={c.progress === p} onClick={() => set({ progress: p })}>{PROGRESS_LABEL[p]}</button>
            ))}
          </span>
        </label>
        <label className="check">
          <input type="checkbox" checked={!!c.even} onChange={(e) => set({ even: e.target.checked })} /> <Translated vi="các đoạn bằng nhau (không jitter)">equal segments (no jitter)</Translated>
        </label>
      </div>

      <div className={'narrate' + (run.outcome === 'done' ? '' : ' bad')} role="status" data-outcome={run.outcome} style={{ marginBottom: '0.8rem' }}>
        {outcomeText[run.outcome]}{' '}
        {run.outcome === 'refused'
          ? run.refused
          : run.outcome === 'done'
            ? tr(`After ${run.time} rounds; ${run.config.scheme === 'barrier' ? 'two grid barriers' : `longest chain ${run.criticalPath} link${run.criticalPath === 1 ? '' : 's'}`}; ${run.waitTotal} rounds spent waiting in all.`, `Sau ${run.time} vòng; ${run.config.scheme === 'barrier' ? 'hai rào chắn toàn lưới' : `chuỗi dài nhất ${run.criticalPath} mắt xích`}; tổng cộng ${run.waitTotal} vòng phải chờ.`)
            : run.outcome === 'deadlock'
              ? tr(`Stuck at round ${run.time}: every resident block spins, and what they wait for cannot run. ${run.stuck.map((s) => `Index ${s.idx} waits on ${s.on}${s.onStarted ? '' : ', which has not started'}.`).join(' ')}`, `Kẹt ở vòng ${run.time}: mọi khối thường trú đều quay chờ, và thứ chúng chờ không thể chạy. ${run.stuck.map((s) => `Chỉ số ${s.idx} chờ ${s.on}${s.onStarted ? '' : ', khối này chưa bắt đầu'}.`).join(' ')}`)
              : tr(`Stuck at round ${run.time}: a block is spinning, and this schedule only runs spinning blocks, so the block it waits for never moves. ${run.stuck.map((s) => `Index ${s.idx} waits on ${s.on}.`).join(' ')}`, `Kẹt ở vòng ${run.time}: một khối đang quay chờ, và lịch chạy này chỉ chạy các khối đang quay chờ, nên khối mà nó chờ không bao giờ nhúc nhích. ${run.stuck.map((s) => `Chỉ số ${s.idx} chờ ${s.on}.`).join(' ')}`)}
      </div>

      {run.outcome !== 'refused' ? (
        <div style={{ overflowX: 'auto' }} data-testid="timeline">
          <Timeline run={run} narrow={narrow} />
        </div>
      ) : null}
      {narrow && run.outcome !== 'refused' && c.index === 'counter' ? (
        <p className="muted" style={{ margin: '0.3rem 0 0', fontSize: '0.85em' }}><Translated vi="i: chỉ số đã nhận; b: blockIdx.x.">i: the claimed index; b: blockIdx.x.</Translated></p>
      ) : null}

      <div style={{ overflowX: 'auto', margin: '0.8rem 0 0' }} data-testid="schemes">
      {narrow ? (
        // On a phone, one column per scheme, so the table fits without scrolling.
        <table className="data" style={{ margin: 0 }} aria-label={tr('The three schemes on this launch', 'Ba phương án trên lần khởi chạy này')} data-layout="narrow">
          <thead>
            <tr>
              <th><Translated vi="Cùng lần khởi chạy">Same launch</Translated></th>
              {all.map((r) => (
                <th key={r.config.scheme} className="num" data-scheme={r.config.scheme} data-outcome={r.outcome}
                  style={r.config.scheme === c.scheme ? { background: 'var(--rule-soft)' } : undefined}>{short[r.config.scheme]}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {SCHEME_COLUMNS.map((col) => (
              <tr key={col.head}>
                <th scope="row"><Translated vi={({ Outcome: 'Kết quả', Rounds: 'Số vòng', 'Longest chain': 'Chuỗi dài nhất', 'Rounds spent waiting': 'Số vòng phải chờ', Waiting: 'Chờ', 'Additions (repeated)': 'Phép cộng (lặp lại)' } as Record<string, string>)[col.short]}>{col.short}</Translated></th>
                {all.map((r) => (
                  <td key={r.config.scheme} className="num" style={r.config.scheme === c.scheme ? { background: 'var(--rule-soft)' } : undefined}>{col.cell(r)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <table className="data" style={{ margin: 0 }} aria-label={tr('The three schemes on this launch', 'Ba phương án trên lần khởi chạy này')}>
          <thead>
            <tr>
              <th><Translated vi="Phương án, cùng lần khởi chạy">Scheme, same launch</Translated></th>
              {SCHEME_COLUMNS.map((col) => <th key={col.head} className={col.head === 'Outcome' ? undefined : 'num'}><Translated vi={({ Outcome: 'Kết quả', Rounds: 'Số vòng', 'Longest chain': 'Chuỗi dài nhất', 'Rounds spent waiting': 'Số vòng phải chờ', Waiting: 'Chờ', 'Additions (repeated)': 'Phép cộng (lặp lại)' } as Record<string, string>)[col.head]}>{col.head}</Translated></th>)}
            </tr>
          </thead>
          <tbody>
            {all.map((r) => (
              <tr key={r.config.scheme} data-scheme={r.config.scheme} data-outcome={r.outcome} data-time={r.time} data-chain={r.criticalPath} data-repeated={r.repeated}
                style={r.config.scheme === c.scheme ? { background: 'var(--rule-soft)' } : undefined}>
                <td>{SCHEME_LABEL[r.config.scheme]}</td>
                {SCHEME_COLUMNS.map((col) => <td key={col.head} className={col.head === 'Outcome' ? undefined : 'num'}>{col.cell(r)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      )}
      </div>
      {run.outcome !== 'refused' ? <PhaseLegend /> : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- classification: make the flag safe
const SAFE_TIERS: Record<string, Tier> = {
  partialSums: 'global', flags: 'global', blockSums: 'global', blockCounter: 'global',
  prev: 'reg', blockSum: 'reg', bid: 'reg', f: 'reg', tid: 'reg',
};
/** The checked files behind the classification's code (labs/ch11/snippets; S119.snippets.test.ts). */
export const FLAG_CU = 'ch11/snippets/s119_flag_protocol.cu';
export const FLAG_MSL: { file: string; readonly label: string; status: string }[] = [
  {
    file: 'ch11/snippets/s119_flag_msl41.metal',
    get label() {
      return tr('MSL 4.1', 'MSL 4.1');
    },
    status: 'unverified: compiled at MSL 4.1 by the snippet test, never run',
  },
  {
    file: 'ch11/snippets/s119_flag_msl32.metal',
    get label() {
      return tr('MSL 3.2, fences', 'MSL 3.2, fence');
    },
    status: 'unverified: compiled at MSL 3.2 and 4.1 by the snippet test, never run',
  },
];
const CU_STATUS: Record<string, string> = {
  guaranteed: 'front-end-checked',
  race: 'front-end-checked; a data race on purpose',
  scope: 'front-end-checked; undefined on purpose',
};

function OrderPicture({ store, load }: { store: StoreOrder; load: LoadOrder }) {
  // Two blocks, four events, each side in program order. The line between a side's two events
  // says whether the plain access may move across the flag access (dashed red) or not (solid green).
  const W = 560;
  const box = (x: number, y: number, label: string, flag: boolean) => (
    <g>
      <rect x={x} y={y} width={196} height={26} rx="3" fill="var(--global-wash)" stroke="var(--global)" strokeWidth={flag ? 2 : 1} />
      <text x={x + 98} y={y + 17} fontSize="11.5" textAnchor="middle">{label}</text>
    </g>
  );
  const side = (y: number, first: string, second: string, flagFirst: boolean, kept: boolean, keptText: string, freeText: string) => (
    <g data-kept={kept ? 1 : 0}>
      {box(20, y, first, flagFirst)}
      {box(300, y, second, !flagFirst)}
      <line x1={218} x2={298} y1={y + 13} y2={y + 13} stroke={kept ? 'var(--ok)' : 'var(--fail)'} strokeWidth="2" strokeDasharray={kept ? undefined : '4 3'} />
      <text x={258} y={y + 42} fontSize="11" textAnchor="middle" style={{ fill: kept ? 'var(--ok)' : 'var(--fail)' }}>{kept ? keptText : freeText}</text>
    </g>
  );
  return (
    <svg viewBox={`0 0 ${W} 150`} role="img" aria-label={tr('Which of the four events may be reordered', 'Sự kiện nào trong bốn sự kiện có thể bị đổi thứ tự')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <text x="20" y="14" fontSize="11.5" fontWeight="600"><Translated vi="khối b − 1 (bên ghi), theo thứ tự chương trình">block b − 1 (writer), in program order</Translated></text>
      {side(20, tr('1. store partialSums[b−1]', '1. ghi partialSums[b−1]'), tr('2. set flags[b−1]', '2. đặt flags[b−1]'), false, store === 'release', tr('release: 1 stays before 2', 'release: 1 luôn đứng trước 2'), tr('relaxed: 2 may be seen before 1', 'relaxed: 2 có thể được thấy trước 1'))}
      <text x="20" y="92" fontSize="11.5" fontWeight="600"><Translated vi="khối b (bên đọc), theo thứ tự chương trình">block b (reader), in program order</Translated></text>
      {side(98, tr('3. read flags[b−1]', '3. đọc flags[b−1]'), tr('4. load partialSums[b−1]', '4. nạp partialSums[b−1]'), true, load === 'acquire', tr('acquire: 4 stays after 3', 'acquire: 4 luôn đứng sau 3'), tr('relaxed: 4 may happen before 3', 'relaxed: 4 có thể xảy ra trước 3'))}
    </svg>
  );
}

function FlagClassifier() {
  const [store, setStore] = useState<StoreOrder>('relaxed');
  const [load, setLoad] = useState<LoadOrder>('relaxed');
  const [scope, setScope] = useState<Scope>('device');
  const [tried, setTried] = useState<Set<string>>(() => new Set());
  // Each control sets only its own value; the combinations seen so far are recorded after each change.
  useEffect(() => setTried((t) => new Set(t).add(`${store}/${load}/${scope}`)), [store, load, scope]);
  const msl41 = useLabFile(FLAG_MSL[0].file);
  const msl32 = useLabFile(FLAG_MSL[1].file);
  const m = messagePassing(store, load, scope);
  const code = ['// block b − 1, the writer (its leader thread)', ...m.writer, '', '// block b, the reader (its leader thread)', ...m.reader].join('\n');
  const chip = m.verdict === 'guaranteed' ? 'ok' : 'fail';
  return (
    <Bench
      title={tr('Make the flag safe', 'Làm cho cờ an toàn')}
      task={
        <Translated vi="Tìm tổ hợp duy nhất bảo đảm khối b thấy được tổng. Sau đó thử phạm vi block với tổ hợp đó.">
          Find the one combination that guarantees block b sees the sum. Then try block scope with it.
        </Translated>
      }
      foot={tr(
        "The verdicts follow NVIDIA's memory-model example (facts A5) and the book's pp. 278–279. The flag is read with load, NVIDIA's spelling; the book's fetch_add(0, ...) behaves the same here.",
        'Các kết luận theo ví dụ mô hình bộ nhớ của NVIDIA (facts A5) và pp. 278–279 của sách. Cờ được đọc bằng load, cách viết của NVIDIA; fetch_add(0, ...) của sách hoạt động giống hệt ở đây.',
      )}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Lệnh ghi cờ của bên ghi">Writer's flag store</Translated>
          <span className="seg" role="group" aria-label={tr('Store order', 'Thứ tự của lệnh ghi')}>
            {(['relaxed', 'release'] as StoreOrder[]).map((o) => (
              <button key={o} aria-pressed={store === o} onClick={() => setStore(o)}>{o}</button>
            ))}
          </span>
        </label>
        <label className="field">
          <Translated vi="Lệnh đọc cờ của bên đọc">Reader's flag load</Translated>
          <span className="seg" role="group" aria-label={tr('Load order', 'Thứ tự của lệnh đọc')}>
            {(['relaxed', 'acquire'] as LoadOrder[]).map((o) => (
              <button key={o} aria-pressed={load === o} onClick={() => setLoad(o)}>{o}</button>
            ))}
          </span>
        </label>
        <label className="field">
          <Translated vi="Phạm vi atomic">Atomic scope</Translated>
          <span className="seg" role="group" aria-label={tr('Scope', 'Phạm vi')}>
            {(['block', 'device'] as Scope[]).map((s) => (
              <button key={s} aria-pressed={scope === s} onClick={() => setScope(s)}>thread_scope_{s}{/* i18n-ignore: CUDA identifier */}</button>
            ))}
          </span>
        </label>
      </div>
      <OrderPicture store={store} load={load} />
      <Code
        tabs={[{ label: 'CUDA', code, tiers: SAFE_TIERS, status: CU_STATUS[m.verdict] }]}
        caption={
          <Translated
            vi={
              <>
                Thân của <code>publish</code> và <code>waitFor</code> trong <code>labs/{FLAG_CU}</code>, với thứ tự và phạm vi bạn
                chọn. Cả tám tổ hợp đều biên dịch được.
              </>
            }
          >
            <>
              The bodies of <code>publish</code> and <code>waitFor</code> in <code>labs/{FLAG_CU}</code>, with your orders and
              scope. Every one of the eight combinations compiles.
            </>
          </Translated>
        }
      />
      <div className={'narrate' + (m.verdict === 'guaranteed' ? '' : ' bad')} role="status" data-verdict={m.verdict}>
        <span className={`chip ${chip}`}>{m.title}</span> {m.why} <span className="muted">({m.source})</span>
      </div>
      {tried.size >= 3 ? (
        <table className="data" style={{ margin: '0.8rem 0 0' }} aria-label={tr('All eight combinations', 'Cả tám tổ hợp')}>
          <thead>
            <tr><th><Translated vi="Phạm vi">Scope</Translated></th><th><Translated vi="Ghi">Store</Translated></th><th><Translated vi="Đọc">Load</Translated></th><th><Translated vi="Kết luận">Verdict</Translated></th></tr>
          </thead>
          <tbody>
            {MP_TABLE.map((r) => (
              <tr key={`${r.scope}${r.store}${r.load}`} data-verdict={r.verdict} style={r.scope === scope && r.store === store && r.load === load ? { background: 'var(--rule-soft)' } : undefined}>
                <td>{r.scope}</td><td>{r.store}</td><td>{r.load}</td>
                <td>{r.verdict === 'guaranteed' ? <span className="t-ok"><Translated vi="thấy được tổng">sees the sum</Translated></span> : r.verdict === 'race' ? <span className="t-fail"><Translated vi="race condition (tranh chấp dữ liệu)">data race</Translated></span> : <span className="t-fail"><Translated vi="không xác định (phạm vi)">undefined (scope)</Translated></span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
      <div className="panel" style={{ marginTop: '0.8rem' }} data-testid="metal">
        <Translated
          vi={
            <p style={{ margin: '0 0 0.4rem' }}>
              <b>Giao thức an toàn trên Metal</b>, theo hai cách viết. Bài kiểm tra snippet biên dịch mỗi tệp ở phiên bản ngôn
              ngữ của nó; không tệp nào được chạy, và không tệp nào làm cho phép scan an toàn ở đó.
            </p>
          }
        >
          <p style={{ margin: '0 0 0.4rem' }}>
            <b>The safe protocol on Metal</b>, in its two spellings. The snippet test compiles each file at its language
            version; neither is run, and neither makes the scan safe there.
          </p>
        </Translated>
        <Code
          tabs={FLAG_MSL.map((f, i) => ({
            label: f.label,
            code: (i === 0 ? msl41 : msl32) ?? `// loading labs/${f.file}`,
            tiers: SAFE_TIERS,
            status: f.status,
          }))}
          caption={
            <Translated vi={<>Lấy từ <code>labs/ch11/snippets/</code>. Chỉ luồng dẫn đầu của mỗi threadgroup tham gia.</>}>
              <>From <code>labs/ch11/snippets/</code>. Only the leader thread of each threadgroup takes part.</>
            </Translated>
          }
        />
        <ul style={{ margin: '0.5rem 0 0', paddingLeft: '1.1rem' }}>
          {METAL_NOTES.map((n) => <li key={n}>{tr(n, ({ 'partialSums must be declared coherent(device): by default device memory is only threadgroup-coherent (MSL 3.2 and later).': 'partialSums phải được khai báo coherent(device): mặc định bộ nhớ device chỉ coherent trong threadgroup (MSL 3.2 trở lên).', 'There is no sleep for the spin, and no documented forward progress between threadgroups: the single-pass scan is still unsafe on Metal.': 'Không có sleep cho vòng quay chờ, và tài liệu không ghi bảo đảm tiến triển nào giữa các threadgroup: phép scan single-pass vẫn không an toàn trên Metal.' } as Record<string, string>)[n] ?? n)}</li>)}
        </ul>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- lab 11.5's single-pass kernel
const SCAN_TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', partialSums: 'global', flags: 'global', blockCounter: 'global',
  buffer_s: 'shared', threadSums: 'shared', bid_s: 'shared', prevBlockPartialSum_s: 'shared', warpSums_s: 'shared',
  buffer_r: 'reg', bid: 'reg', threadSum: 'reg', prevPartialSum: 'reg', prevBlockPartialSum: 'reg', c: 'reg', val: 'reg',
};
const linesWith = (text: string, needles: string[]) =>
  text.split('\n').flatMap((l, i) => (needles.some((n) => l.includes(n)) ? [i + 1] : []));

function SinglePassListing() {
  const cu = useLabFile(LAB5_FILE);
  // The labs page's words (LabKit's solutionTabStatus), from this machine's runs. Lab 11.5 has no Metal files.
  const { reports } = useReports();
  if (cu == null) return <p className="muted"><Translated vi="Đang nạp kernel của lab 11.5…">Loading lab 11.5's kernel…</Translated></p>;
  if (cu.startsWith('// could not load'))
    return (
      <Translated
        vi={
          <p className="muted" data-testid="lab5-missing">
            Lời giải tham chiếu (solution) của lab 11.5 sẽ hiện ở đây khi lab đã có: <code>labs/{LAB5_FILE}</code>.
          </p>
        }
      >
        <p className="muted" data-testid="lab5-missing">
          Lab 11.5's reference solution is shown here once the lab is in place: <code>labs/{LAB5_FILE}</code>.
        </p>
      </Translated>
    );
  return (
    <Code
      tabs={[
        {
          label: tr('CUDA (lab 11.5)', 'CUDA (lab 11.5)'),
          code: cu,
          tiers: SCAN_TIERS,
          status: solutionTabStatus(reports, 'ch11/lab5_singlepass', 'kernel.cu'),
          highlight: linesWith(cu, ['memory_order_acquire', 'memory_order_release', 'blockCounter']),
        },
      ]}
      caption={
        <Translated
          vi={
            <>
              Lời giải tham chiếu của lab 11.5, <code>labs/{LAB5_FILE}</code>: Fig. 11.18 với giao thức cờ của Fig. 11.17, đã sửa
              và chạy trên các trình giả lập của khóa học. Được đánh dấu: chỉ số khối đã nhận, cùng acquire và release trên các
              cờ. Chỉ cho họ CUDA: Metal không thể chờ an toàn giữa các threadgroup (xem bên dưới). Hãy thử làm lab trước khi
              nghiên cứu phần này.
            </>
          }
        >
          <>
            Lab 11.5's reference solution, <code>labs/{LAB5_FILE}</code>: Fig. 11.18 with Fig. 11.17's flag protocol, corrected
            and run on the course's emulators. Marked: the claimed block index and the acquire and release on the flags. CUDA family only: Metal cannot
            wait safely across threadgroups (see below). Try the lab before you study this.
          </>
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- questions
export const questionsS119: Record<string, Question> = {
  'ch11/11.9/q-ssa': {
    kind: 'numeric',
    id: 'ch11/11.9/q-ssa',
    prompt: <Translated vi={<>Scan-scan-add ba kernel trên số float, với S = 4096 phần tử mỗi khối. Nó di chuyển bao nhiêu byte mỗi phần tử, lấy ba chữ số thập phân?</>}><>Three-kernel scan-scan-add over floats, with S = 4096 elements per block. How many bytes per element does it move, to three decimal places?</></Translated>,
    answer: 16.002,
    tol: 0.00006,
    unit: 'bytes',
    explain: (
      <Translated
        vi={
          <p>
            16 + 8/S = 16 + 8/4096 = 16.00195, nên là 16.002. Kernel 1 và 3 mỗi kernel nạp và ghi cả N phần tử (cộng lại 16
            byte mỗi phần tử); kernel 2 nạp và ghi N/S tổng của khối (8/S). Tổng của khối gần như không tốn gì; 16 mới là vấn
            đề, vì kernel 1 ghi ra mọi phần tử đã scan một phần và kernel 3 đọc lại tất cả.
          </p>
        }
      >
        <p>
          16 + 8/S = 16 + 8/4096 = 16.00195, so 16.002. Kernels 1 and 3 each load and store all N elements (16 bytes per element
          between them); kernel 2 loads and stores the N/S block sums (8/S). The block sums cost almost nothing; the 16 is the
          problem, because kernel 1 stores every partially scanned element and kernel 3 reads it all back.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.9, p. 274',
    vi: { unit: 'byte' },
  },
  'ch11/11.9/q-rss-why': {
    kind: 'choice',
    id: 'ch11/11.9/q-rss-why',
    prompt: <Translated vi={<>Sách ưu tiên reduce-scan-scan giữa các khối nhưng lại dùng scan-scan-add bên trong một khối. Vì sao có sự khác biệt?</>}><>The book prefers reduce-scan-scan across blocks but scan-scan-add inside a block. Why the difference?</></Translated>,
    options: [
      { label: <Translated vi="Reduce-scan-scan làm ít phép cộng hơn tổng cộng, và việc tiết kiệm phép cộng chỉ có lợi khi dữ liệu trải trên nhiều khối.">Reduce-scan-scan does fewer additions in total, and saving additions only pays once the data spans many blocks.</Translated>, why: <Translated vi="Cả hai đều làm hai lượt việc trên dữ liệu; khác biệt mà sách chỉ ra là lưu lượng và đồng bộ hóa.">Both do two passes of work over the data; the difference the book points to is traffic and synchronisation.</Translated> },
      {
        label: <Translated vi="Giữa các khối, byte là thứ khan hiếm và một phép reduce chỉ ghi một giá trị mỗi khối; bên trong một khối, giá trị nằm trong thanh ghi và cây của phép reduce tốn rào chắn.">Across blocks bytes are scarce and a reduce writes one value per block; inside one, values sit in registers and a reduce's tree costs barriers.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Scan-scan-add không đúng giữa các khối: kernel 3 có thể đọc những tổng mà kernel 2 chưa ghi.">Scan-scan-add is not correct across blocks: kernel 3 can read sums kernel 2 has not written.</Translated>,
        why: <Translated vi="Nó đúng; phiên bản ba kernel của Fig. 11.14 chạy được, vì ranh giới kernel sắp thứ tự cho chúng. Nó chậm hơn: 16 byte mỗi phần tử so với 12.">It is correct; the three-kernel version of Fig. 11.14 works, since a kernel boundary orders them. It is slower: 16 bytes per element against 12.</Translated>,
      },
      {
        label: <Translated vi="Bên trong một khối không có primitive rút gọn, nên một phép reduce sẽ cần một kernel riêng.">Inside a block there is no reduction primitive, so a reduce would need a kernel of its own.</Translated>,
        why: <Translated vi="Chương 10 xây dựng phép rút gọn khối; chúng có tồn tại. Chỉ là chúng tốn nhiều đồng bộ hóa hơn một phép cộng.">Chapter 10 builds block reductions; they exist. They just cost more synchronisation than an add.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Cả hai thiết kế đều có hai phép scan và thêm một bước: một phép add hoặc một phép reduce. Phép reduce ghi ít đầu ra
            hơn nhiều; phép add không cần cây có rào chắn giữa các mức. Bên nào thắng phụ thuộc vào thứ gì khan hiếm. Giữa các
            kernel đó là bộ nhớ toàn cục, nên reduce thắng (12 so với 16 byte). Giữa các warp hay các luồng, các lần truy cập
            thêm rơi vào thanh ghi và gần như miễn phí, còn mỗi rào chắn thì không.
          </p>
        }
      >
        <p>
          Both designs have two scans and one more step: an add or a reduce. A reduce writes far fewer outputs; an add needs no
          tree with barriers between its levels. Which one wins depends on what is scarce. Between kernels it is global memory,
          so the reduce wins (12 against 16 bytes). Between warps or threads, the extra accesses hit registers and are almost
          free, while every barrier is not.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.9, p. 276',
  },
  'ch11/11.9/q-commutative': {
    kind: 'choice',
    id: 'ch11/11.9/q-commutative',
    prompt: (
      <Translated
        vi={
          <>
            Toán tử scan của bạn có tính kết hợp nhưng không giao hoán: ví dụ phép hợp các ánh xạ affine x ↦ a·x + b, như một
            hệ thức truy hồi tuyến tính cần. Bạn phải kiểm tra gì trước khi dùng reduce-scan-scan?
          </>
        }
      >
        <>
          Your scan operator is associative but not commutative: for example composing affine maps x ↦ a·x + b, as a linear
          recurrence needs. What must you check before using reduce-scan-scan?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Không gì cả: tính kết hợp là tất cả những gì một phép scan cần, nên phép rút gọn nào của chương 10 cũng được.">Nothing: associativity is all a scan needs, so any of chapter 10's reductions will do.</Translated>,
        why: <Translated vi="Hai phép scan chỉ cần tính kết hợp. Phép reduce trong kernel 1 mới là bước cần kiểm tra: một số tối ưu hóa của chương 10 giả định tính giao hoán.">The two scans only need associativity. The reduce in kernel 1 is the step to check: some of chapter 10's optimisations assume commutativity.</Translated>,
      },
      { label: <Translated vi="Rằng phép rút gọn chỉ kết hợp các khoảng kề nhau, theo thứ tự.">That the reduction combines only neighbouring ranges, in order.</Translated>, correct: true },
      {
        label: <Translated vi="Rằng các tổng của khối được scan theo chiều ngược lại, để các ánh xạ hợp từ phải sang trái.">That the block sums are scanned in reverse, so the maps compose right to left.</Translated>,
        why: <Translated vi="Đảo ngược thứ tự sẽ hợp các ánh xạ theo chiều sai.">Reversing the order would compose the maps the wrong way round.</Translated>,
      },
      { label: <Translated vi="Rằng mọi khối có cùng số phần tử, để các ánh xạ khớp nhau.">That every block has the same number of elements, so the maps line up.</Translated>, why: <Translated vi="Kích thước đoạn không thay đổi các tính chất của toán tử.">Segment sizes do not change the operator's properties.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mọi thứ trong chương cho đến giờ chỉ giả định tính kết hợp, cho phép bạn nhóm lại các phần tử kề nhau nhưng không
            bao giờ đổi thứ tự chúng. Một phép rút gọn trong đó luồng t kết hợp phần tử t với phần tử t + N/2 nối những phần tử
            không kề nhau, nên nó ổn với phép cộng nhưng sai với phép hợp ánh xạ. Trong reduce-scan-scan, phép rút gọn của
            kernel 1 chỉ được phép kết hợp các khoảng kề nhau, khoảng bên trái trước.
          </p>
        }
      >
        <p>
          Everything in the chapter so far assumed only associativity, which lets you regroup neighbours but never reorder
          them. A reduction in which thread t combines element t with element t + N/2 joins elements that are not neighbours,
          so it is fine for addition and wrong for composing maps. In reduce-scan-scan, kernel 1's reduction must only ever
          combine adjacent ranges, the left one first.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.9, p. 276; the affine-map example is ours',
    vi: { source: 'ch. 11 §11.9, p. 276; ví dụ ánh xạ affine là của khóa học này' },
  },
  'ch11/11.9/q-acquire': {
    kind: 'choice',
    id: 'ch11/11.9/q-acquire',
    prompt: <Translated vi={<>Khối đang chờ đọc cờ bằng <code>memory_order_relaxed</code> thay vì acquire. Bên ghi vẫn dùng release. Điều gì có thể hỏng?</>}><>The waiting block reads the flag with <code>memory_order_relaxed</code> instead of acquire. The writer still uses release. What can go wrong?</></Translated>,
    options: [
      { label: <Translated vi="Không gì cả: release ở bên ghi là đủ.">Nothing: release on the writer is enough.</Translated>, why: <Translated vi="Release chỉ sắp thứ tự phía bên ghi. Lệnh nạp thường của bên đọc vẫn có thể bị dời đi.">Release orders the writer's side only. The reader's plain load can still be moved.</Translated> },
      {
        label: <Translated vi="Lệnh nạp partialSums[b−1] có thể dời lên trước lần đọc cờ và thấy một tổng cũ: một race condition.">The load of partialSums[b−1] may move before the flag read and see a stale sum: a data race.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Vòng lặp quay chờ có thể không bao giờ kết thúc, vì một lệnh nạp relaxed có thể bị cache mãi mãi.">The spin loop may never end, because a relaxed load can be cached forever.</Translated>, why: <Translated vi="Một lệnh nạp atomic relaxed vẫn đọc biến atomic; CUDA tính một lần đọc atomic là tiến triển (facts A6). Vấn đề là lệnh nạp thường sau nó.">A relaxed atomic load still reads the atomic; CUDA counts an atomic read as progress (facts A6). The problem is the plain load after it.</Translated> },
      { label: <Translated vi="Bản thân cờ có thể bị đọc rách (torn).">The flag itself may be read torn.</Translated>, why: <Translated vi="Atomic relaxed vẫn là atomic: không có lần đọc rách nào.">Relaxed atomics are still atomic: no torn reads.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Acquire ở lần đọc là thứ giữ cho lệnh nạp tổng phía sau không xảy ra trước lần kiểm tra cờ (p. 279). Không có nó,
            trình biên dịch hoặc phần cứng có thể nạp <code>partialSums[b−1]</code> sớm, thấy giá trị cũ, và chỉ sau đó mới
            thấy cờ được đặt. Mô hình bộ nhớ của NVIDIA gọi cặp không được sắp thứ tự đó là một data race, tức hành vi không xác
            định.
          </p>
        }
      >
        <p>
          Acquire on the read is what keeps the following load of the sum from happening before the flag check (p. 279). Without
          it the compiler or the hardware may load <code>partialSums[b−1]</code> early, see the old value, and only then see the
          flag set. NVIDIA's memory model calls the unordered pair a data race, which is undefined.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.9, p. 279; notes/today/ch11-facts.md A5', // i18n-ignore: citation only
  },
  'ch11/11.9/q-scope': {
    kind: 'choice',
    id: 'ch11/11.9/q-scope',
    prompt: <Translated vi={<>Ai đó viết giao thức cờ với <code>cuda::thread_scope_block</code> trên cả hai tham chiếu atomic, vẫn giữ release và acquire. Kết quả là gì?</>}><>Someone writes the flag protocol with <code>cuda::thread_scope_block</code> on both atomic references, keeping release and acquire. What is the result?</></Translated>,
    options: [
      {
        label: <Translated vi="Đúng, và nhanh hơn: một bên ghi và một bên đọc mỗi cờ thì không cần phạm vi rộng hơn.">Correct, and faster: one writer and one reader per flag need no wider scope.</Translated>,
        why: <Translated vi="Nhanh hơn, có thể; đúng, thì không. Phạm vi block chỉ bao gồm khối của chính bên ghi, còn bên đọc lại ở khối khác.">Faster, perhaps; correct, no. Block scope covers only the writer's own block, and the reader is in another.</Translated>,
      },
      {
        label: tr("Undefined behaviour: with block scope on the store, only the writer's own block is ordered with it.", 'Hành vi không xác định: với phạm vi block trên lệnh ghi, chỉ khối của chính bên ghi được sắp thứ tự với nó.'),
        correct: true,
      },
      { label: <Translated vi="Chỉ đúng trên Hopper trở về sau.">Correct on Hopper and later only.</Translated>, why: <Translated vi="Mô hình bộ nhớ ở đây không phụ thuộc vào kiến trúc.">The memory model does not depend on the architecture here.</Translated> },
      {
        label: <Translated vi="Lỗi biên dịch: một atomic phạm vi block không thể tham chiếu bộ nhớ toàn cục.">A compile error: a block-scope atomic cannot refer to global memory.</Translated>,
        why: <Translated vi="Nó biên dịch được: phạm vi nói những luồng nào được sắp thứ tự, không phải biến nằm ở đâu. NVIDIA in biến thể này như mã có data race, không phải một lỗi. Chính điều đó làm nó nguy hiểm.">It compiles: a scope says which threads are ordered, not where the variable lives. NVIDIA prints this variant as code that is a data race, not an error. That is what makes it dangerous.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Phạm vi cho biết atomic đồng bộ hóa với những luồng nào. Cờ được một khối ghi và một khối khác đọc, nên nó cần{' '}
            <code>thread_scope_device</code>. Hướng dẫn của NVIDIA in ví dụ message-passing của họ với phạm vi block trên lệnh
            ghi và chú thích nó là "UB: data race": lệnh ghi "only includes all other threads of Block 0". Thu hẹp phạm vi của
            bên đọc nữa không thể mở rộng phạm vi của bên ghi, nên phạm vi block ở cả hai phía cũng không khá hơn.
          </p>
        }
      >
        <p>
          Scope says which threads the atomic synchronises with. The flag is written by one block and read by another, so it
          needs <code>thread_scope_device</code>. NVIDIA's guide prints its message-passing example with block scope on the
          store and comments it "UB: data race": the store "only includes all other threads of Block 0". Narrowing the
          reader's scope as well cannot widen the writer's, so block scope on both sides is no better.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch11-facts.md A5 (NVIDIA memory model, section 5.8.5); ch. 11 p. 278',
    vi: { source: 'notes/today/ch11-facts.md A5 (mô hình bộ nhớ của NVIDIA, mục 5.8.5); ch. 11 p. 278' },
  },
  'ch11/11.9/q-deadlock': {
    kind: 'choice',
    id: 'ch11/11.9/q-deadlock',
    prompt: <Translated vi={<>Vì sao một phép scan single-pass chờ khối có <code>blockIdx.x − 1</code> có thể bị treo trên một lưới lớn, và sách tránh điều đó thế nào?</>}><>Why can a single-pass scan that waits on the block with <code>blockIdx.x − 1</code> hang on a large grid, and how does the book avoid it?</></Translated>,
    options: [
      { label: <Translated vi="Các cờ bị tràn trên lưới lớn; sách chuyển sang cờ 64-bit để tránh điều đó.">The flags overflow on a large grid; the book switches to 64-bit flags to avoid it.</Translated>, why: <Translated vi="Một cờ chỉ bao giờ giữ 0 hoặc 1.">A flag only ever holds 0 or 1.</Translated> },
      {
        label: <Translated vi="Các khối thường trú có thể cùng chờ một khối chưa bao giờ bắt đầu; một chỉ số nhận từ bộ đếm atomic thuộc về một khối đã bắt đầu.">Resident blocks may all wait on one that never started; an index claimed from an atomic counter belongs to a block that has started.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Các khối có chỉ số liên tiếp chạy trên các SM khác nhau và không thấy được bộ nhớ của nhau.">Blocks with consecutive indices run on different SMs and cannot see each other's memory.</Translated>, why: <Translated vi="Atomic phạm vi device được thấy qua các SM; đó là mục đích của phạm vi.">Device-scope atomics are visible across SMs; that is what the scope is for.</Translated> },
      { label: <Translated vi="Nó không thể treo: CUDA bắt đầu các khối theo thứ tự chỉ số, nên khối b − 1 luôn bắt đầu trước khối b.">It cannot hang: CUDA starts blocks in index order, so block b − 1 always starts before block b.</Translated>, why: <Translated vi="Hướng dẫn không hứa thứ tự nào. CUB dựa vào thứ tự khởi chạy quan sát được, vốn không phải một bảo đảm được ghi trong tài liệu (facts A6).">The guide promises no order. CUB relies on observed launch order, which is not a documented guarantee (facts A6).</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Chờ chỉ an toàn trên một khối chắc chắn sẽ chạy. Với bộ đếm, chỉ số b − 1 thuộc về một khối đã lấy nó bằng một
            atomic, nên nó đã thực thi một bước, và mô hình thực thi của CUDA khi đó bảo đảm nó tiếp tục tiến triển. Chờ{' '}
            <code>blockIdx.x − 1</code> không có chỗ dựa như vậy: khối đó có thể chưa bao giờ được lên lịch. Preset đầu tiên
            của trình mô phỏng cho thấy cảnh treo.
          </p>
        }
      >
        <p>
          Waiting is safe only on a block that is sure to run. With the counter, index b − 1 belongs to a block that took it with
          an atomic, so it has executed a step, and CUDA's execution model then guarantees it keeps making progress. Waiting on{' '}
          <code>blockIdx.x − 1</code> has no such backing: that block may never have been scheduled. The simulator's first
          preset shows the hang.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.9, pp. 277, 280, 285; notes/today/ch11-facts.md A6', // i18n-ignore: citation only
  },
  'ch11/11.9/q-zero': {
    kind: 'choice',
    id: 'ch11/11.9/q-zero',
    prompt: <Translated vi={<>Một chương trình khởi chạy kernel single-pass hai lần liên tiếp và không đặt lại các cờ hay bộ đếm khối giữa hai lần. Điều gì xảy ra ở lần khởi chạy thứ hai?</>}><>A program launches the single-pass kernel twice in a row and does not reset the flags or the block counter between launches. What happens in the second launch?</></Translated>,
    options: [
      { label: <Translated vi="Cùng kết quả: mỗi lần khởi chạy ghi đè các cờ trước khi khối nào đọc chúng.">The same result: each launch overwrites the flags before any block reads them.</Translated>, why: <Translated vi="Một khối đặt cờ của nó; không có gì trong kernel xóa cờ.">A block sets its flag; nothing in the kernel clears one.</Translated> },
      {
        label: <Translated vi="Bộ đếm phát ra các chỉ số vượt quá lưới và các cờ đã đọc ra 1, nên các khối chạy ra ngoài phạm vi hoặc bỏ qua bước chờ.">The counter hands out indices past the grid and the flags already read 1, so blocks run out of range or skip the wait.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Một deadlock: mọi khối chờ một cờ đã được đặt sẵn.">A deadlock: every block waits for a flag that is already set.</Translated>, why: <Translated vi="Một cờ đã đặt kết thúc việc chờ ngay lập tức; mối nguy là không chờ.">A set flag ends the wait immediately; the danger is not waiting.</Translated> },
      { label: <Translated vi="Chỉ khối đầu tiên sai, vì chỉ nó không có khối trước để chờ.">Only the first block is wrong, since it alone has no predecessor to wait for.</Translated>, why: <Translated vi="Mọi khối nhận chỉ số tính từ chỗ bộ đếm đã dừng.">Every block claims an index from where the counter stopped.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Kernel giả định bộ đếm ở 0 và mọi cờ ở 0. Sách nói bộ đếm được khởi tạo bằng 0; các cờ cũng phải như vậy, và phải
            làm cho mọi lần khởi chạy, tức là một lệnh <code>cudaMemset</code> (hoặc <code>hipMemset</code>) trên host trước
            mỗi lần.
          </p>
        }
      >
        <p>
          The kernel assumes a counter at 0 and every flag at 0. The book says the counter is zero-initialised; the flags must be
          too, and so for every launch, which is a <code>cudaMemset</code> (or <code>hipMemset</code>) on the host before each one.
        </p>
      </Translated>
    ),
    source: 'ch. 11 p. 280; notes/ch11.md "Unclear in the PDF"; notes/today/ch11-facts.md A5',
    vi: { source: 'ch. 11 p. 280; notes/ch11.md mục "Unclear in the PDF"; notes/today/ch11-facts.md A5' },
  },
  'ch11/11.9/q-metal': {
    kind: 'choice',
    id: 'ch11/11.9/q-metal',
    prompt: <Translated vi={<>Bạn cần một phép scan bao gồm (inclusive) trên 10⁸ số float trên Metal. Thiết kế nào?</>}><>You need an inclusive scan of 10⁸ floats on Metal. Which design?</></Translated>,
    options: [
      { label: <Translated vi="Kernel single-pass với chỉ số đã nhận, acquire và release của MSL 4.1, và các buffer coherent(device).">The single-pass kernel with a claimed index, MSL 4.1 acquire and release, and coherent(device) buffers.</Translated>, why: <Translated vi="Cả ba đều cần cho cờ, và không thứ nào trong số đó đem lại forward progress giữa các threadgroup, điều mà Metal không ghi trong tài liệu.">All three are needed for the flag, and none of them gives forward progress between threadgroups, which Metal does not document.</Translated> },
      { label: <Translated vi="Một rào chắn toàn lưới.">A grid-wide barrier.</Translated>, why: <Translated vi="Metal không có rào chắn toàn lưới và không có cooperative launch.">Metal has no grid-wide barrier and no cooperative launch.</Translated> },
      {
        label: <Translated vi="Một phép scan phân cấp trong nhiều lần dispatch của một compute encoder, vốn chạy chúng theo thứ tự.">A hierarchical scan in several dispatches of one compute encoder, which runs them in order.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Một threadgroup cho cả mảng.">One threadgroup for the whole array.</Translated>, why: <Translated vi="Đúng, nhưng nó chỉ dùng một threadgroup của cả GPU.">Correct, but it uses a single threadgroup of the whole GPU.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Phép scan single-pass cần một khối đang chờ phải chắc rằng khối mà nó chờ tiếp tục chạy. Đặc tả của Metal không hứa
            gì về các threadgroup với nhau, và một kỹ sư Apple đã gọi đích danh giả định đó là không xác định. Các lần dispatch
            riêng biệt không cần lời hứa nào như vậy: một compute encoder mặc định chạy chúng tuần tự và các buffer của nó được
            theo dõi hazard. Đó là những gì llama.cpp và PyTorch đưa vào sản phẩm trên Metal, và là thứ lab 11.4 xây dựng.
          </p>
        }
      >
        <p>
          The single-pass scan needs a block that waits to be sure the block it waits for keeps running. Metal's specification
          promises nothing about threadgroups relative to each other, and an Apple engineer named exactly that assumption as
          undefined. Separate dispatches need no such promise: a default compute encoder runs them serially and its buffers are
          hazard-tracked. That is what llama.cpp and PyTorch ship on Metal, and what lab 11.4 builds.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch11-facts.md C3, C4, C5, D1, D4; ch02-facts.md C9', // i18n-ignore: citation only
  },
  'ch11/11.9/q-ex5': {
    kind: 'choice',
    id: 'ch11/11.9/q-ex5',
    prompt: <Translated vi={<>Bài tập 5 yêu cầu decoupled lookback. Một giá trị cờ mỗi khối (0 hoặc 1) không còn đủ nữa. Một khối cần công bố những trạng thái nào?</>}><>Exercise 5 asks for decoupled lookback. One flag value per block (0 or 1) is no longer enough. Which states does a block need to publish?</></Translated>,
    options: [
      {
        label: <Translated vi="Hai, như trong single lookback: chưa sẵn sàng, và sẵn sàng.">Two, as in single lookback: not ready, and ready.</Translated>,
        why: <Translated vi="Đó là single lookback. Một khối đang chờ sẽ không phân biệt được tổng riêng của khối với một tiền tố đầy đủ.">That is single lookback. A waiting block could not tell an own total from a full prefix.</Translated>,
      },
      { label: <Translated vi="Ba: chưa có gì, tổng riêng đã sẵn sàng, tiền tố bao gồm đã sẵn sàng.">Three: nothing yet, own total ready, inclusive prefix ready.</Translated>, correct: true },
      {
        label: <Translated vi="Mỗi khối trước mà nó đã cộng một trạng thái, để khối đang chờ thấy nó đã đi được bao xa.">One per predecessor it has added, so a waiting block can see how far it got.</Translated>,
        why: <Translated vi="Một khối đã cộng những gì là việc riêng của nó; các khối khác chỉ cần biết được phép dùng giá trị nào trong hai giá trị của nó.">What a block has added is its own business; others only need to know which of its two values they may use.</Translated>,
      },
      {
        label: <Translated vi="Không cần gì: decoupled lookback đọc chính các tổng và không cần cờ.">None: decoupled lookback reads the sums themselves and needs no flags.</Translated>,
        why: <Translated vi="Một khối đang chờ vẫn phải biết khi nào một giá trị sẵn sàng, điều đó cần một lần đọc cờ với acquire.">A waiting block still has to know when a value is ready, which takes a flag read with acquire.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một khối đang chờ đi ngược từ b − 1. Ở mỗi khối trước, nó chờ cho đến khi có thứ gì đó được công bố. Nếu nó thấy
            tiền tố bao gồm, nó cộng vào và dừng; nếu chỉ thấy tổng riêng, nó cộng tổng đó và lùi thêm một khối. Cả hai giá trị
            được ghi trước trạng thái của chúng với release và đọc sau trạng thái với acquire, như trong Fig. 11.17.
          </p>
        }
      >
        <p>
          A waiting block walks back from b − 1. At each predecessor it waits until something is published. If it finds the
          inclusive prefix it adds it and stops; if it finds only the own total, it adds that and moves one further back. Both
          values are written before their state with release and read after it with acquire, as in Fig. 11.17.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.9, pp. 279–280; exercise 5, p. 286',
    vi: { source: 'ch. 11 §11.9, pp. 279–280; bài tập 5, p. 286' },
  },
};

export function S119() {
  return (
    <>
      <Translated
        vi={
          <p>
            Một phép scan khối phủ đoạn của một khối, thứ mà bộ nhớ trên chip giữ ở kích thước nhỏ. Một số ứng dụng không cần
            gì hơn. Những ứng dụng khác scan hàng triệu hay hàng tỷ phần tử, và kết quả của mỗi phần tử phụ thuộc vào mọi khối
            trước nó. Phần này ghép các phép scan khối thành một phép scan toàn cục, trước hết bằng nhiều kernel, rồi bằng một,
            và đếm chi phí của mỗi cách theo đơn vị mà phần 11.8 đã cho bạn: số byte mỗi phần tử.
          </p>
        }
      >
        <p>
          A block scan covers one block's segment, which on-chip memory keeps small. Some applications need nothing more. Others
          scan millions or billions of elements, and every element's result depends on every block before it. This section
          stitches block scans into one global scan, first with several kernels, then with one, and counts the cost of each in
          the unit section 11.8 gave you: bytes per element.
        </p>
      </Translated>
      <Source section="11.9" pages="273" />

      <h2><Translated vi="Ba kernel: scan, scan, add">Three kernels: scan, scan, add</Translated></h2>
      <Translated
        vi={
          <p>
            Ý tưởng đầu tiên là phép phân rã bạn đã dùng hai lần, bên trong một warp và bên trong một luồng, giờ lên thêm một
            cấp (Fig. 11.14). Kernel 1 scan đoạn của mỗi khối và ghi nó ra, cùng với giá trị cuối của mỗi khối, tức tổng của
            khối. Kernel 2 là một khối duy nhất scan các tổng của khối đó. Kernel 3 cộng vào mọi phần tử của khối b tổng đã scan
            của mọi khối đứng trước nó.
          </p>
        }
      >
        <p>
          The first idea is the decomposition you already used twice, inside a warp and inside a thread, now one level up (Fig.
          11.14). Kernel 1 scans each block's segment and writes it out, along with each block's last value, its sum. Kernel 2 is
          a single block that scans those block sums. Kernel 3 adds to every element of block b the scanned sum of all the blocks
          before it.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Hãy đếm lưu lượng với S phần tử mỗi khối. Kernel 1 nạp N và ghi N. Kernel 2 nạp và ghi N/S. Kernel 3 nạp N và ghi
            N. Tổng là (4 + 2/S)·N phần tử, hay (16 + 8/S)·N byte với số float: khoảng 16 byte mỗi phần tử. Trên thiết bị ví dụ
            của sách, 3.35 × 10¹² / 16 = 209 × 10⁹ phần tử mỗi giây, một nửa con số 419 × 10⁹ của phần 11.8. Lý do rất rõ: cả N
            giá trị đã scan một phần được kernel 1 ghi ra và kernel 3 nạp lại.
          </p>
        }
      >
        <p>
          Count the traffic with S elements per block. Kernel 1 loads N and stores N. Kernel 2 loads and stores N/S. Kernel 3
          loads N and stores N. That is (4 + 2/S)·N elements, or (16 + 8/S)·N bytes for floats: about 16 bytes per element. On
          the book's example device, 3.35 × 10¹² / 16 = 209 × 10⁹ elements per second, half the 419 × 10⁹ of section 11.8.
          The reason is plain: all N partly scanned values are stored by kernel 1 and loaded again by kernel 3.
        </p>
      </Translated>
      <Source section="11.9" pages="273–274" figs="Fig. 11.14" />

      <h2><Translated vi="Reduce, scan, scan">Reduce, scan, scan</Translated></h2>
      <Translated
        vi={
          <p>
            Kernel 1 không cần ghi đoạn đã scan nếu tất cả những gì kernel 2 muốn từ nó là tổng. Vậy hãy reduce thay vào đó
            (Fig. 11.15): kernel 1 rút gọn mỗi đoạn thành một tổng, bằng bất kỳ{' '}
            <a href={href('ch10', '10.9')}>kernel rút gọn nào của chương 10</a>; kernel 2 scan các tổng của khối; kernel 3 cộng
            tổng của các khối trước vào phần tử đầu tiên của đoạn rồi scan đoạn đó, tạo ra trực tiếp các giá trị cuối cùng.
          </p>
        }
      >
        <p>
          Kernel 1 does not need to store the scanned segment if all kernel 2 wants from it is its total. So reduce instead
          (Fig. 11.15): kernel 1 reduces each segment to one sum, with any of{' '}
          <a href={href('ch10', '10.9')}>chapter 10's reduction kernels</a>; kernel 2 scans the block sums; kernel 3 adds the
          sum of the earlier blocks to the first element of its segment and then scans the segment, which produces final values
          directly.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Giờ kernel 1 nạp N và chỉ ghi N/S. Kernel 2 nạp và ghi N/S. Kernel 3 nạp N và ghi N. Tổng là (3 + 3/S)·N phần tử,
            (12 + 12/S)·N byte, và 279 × 10⁹ phần tử mỗi giây trên ví dụ của sách: hai phần ba mức lý tưởng. Nó tiết kiệm lần
            ghi dữ liệu trung gian; nó vẫn đọc đầu vào hai lần.
          </p>
        }
      >
        <p>
          Kernel 1 now loads N and stores only N/S. Kernel 2 loads and stores N/S. Kernel 3 loads N and stores N. That is (3 +
          3/S)·N elements, (12 + 12/S)·N bytes, and 279 × 10⁹ elements per second on the book's example: two thirds of the
          ideal. It saves the store of the intermediate; it still reads the input twice.
        </p>
      </Translated>
      <Source section="11.9" pages="274–276" figs="Fig. 11.15" />
      <BytesExplorer />
      <Translated
        vi={
          <p>
            Khi bấm các số của sách, ba trần đọc ra 419, 209 và 279. Thay đổi S thì chỉ chữ số có nghĩa thứ ba dịch chuyển: các
            tổng của khối chỉ là sai số làm tròn với bất kỳ đoạn thực tế nào. Bật phần ghi chép lên và điều tương tự cũng đúng
            với vài lần truy cập theo từng khối mà tổng của sách bỏ qua. Thứ quyết định trần là số lần N phần tử đi qua bus bộ
            nhớ.
          </p>
        }
      >
        <p>
          With the book's numbers pressed, the three ceilings read 419, 209 and 279. Move S and only the third significant digit
          moves: the block sums are a rounding error for any realistic segment. Switch on the bookkeeping and the same is true of
          the few per-block accesses the book's totals leave out. What decides the ceiling is how many times the N elements cross
          the memory bus.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Chỉ tính byte: không có chi phí khởi chạy giữa ba kernel, không có cache, không có thời gian cho khối duy nhất của
              kernel 2. Đó là những gì một phép đo thêm vào.
            </p>
          }
        >
          <p>
            Bytes only: no launch overhead between the three kernels, no caches, no time for kernel 2's single block. Those are
            what a measurement adds.
          </p>
        </Translated>
      </Simplification>
      <Ask q={questionsS119['ch11/11.9/q-ssa']} />
      <Predict
        id="ch11/11.9/p-rss"
        prompt={
          <Translated vi="Trên thiết bị của bạn, ở n = 2^26, reduce-scan-scan sẽ chạy nhanh gấp bao nhiêu lần scan-scan-add? Phép đếm byte nói 16/12.">
            How many times faster than scan-scan-add will reduce-scan-scan run at n = 2^26 on your device? The byte count says 16/12.
          </Translated>
        }
        unit="×"
      >
        {(v) => <RssReveal predicted={v} />}
      </Predict>
      <Translated
        vi={
          <p>
            Một lưu ý trước khi bạn dùng lại một kernel của chương 10 ở bước 1. Mọi thứ trong chương này cho đến giờ chỉ cần
            toán tử có tính kết hợp, không gì hơn. Một số tối ưu hóa phép rút gọn còn đổi thứ tự toán hạng, điều này giả định
            tính giao hoán. Với một toán tử không giao hoán, hãy chọn phép rút gọn cẩn thận.
          </p>
        }
      >
        <p>
          One caution before you reuse a chapter 10 kernel in step 1. Everything in this chapter so far needs the operator to be
          associative and nothing more. Some reduction optimisations also reorder operands, which assumes commutativity. For an
          operator that is not commutative, choose the reduction with care.
        </p>
      </Translated>
      <Source section="11.9" pages="276" />
      <Ask q={questionsS119['ch11/11.9/q-commutative']} />

      <h2><Translated vi="Sao không dùng reduce-scan-scan bên trong một khối?">Why not reduce-scan-scan inside a block?</Translated></h2>
      <Translated
        vi={
          <p>
            Cả hai phép phân rã đều có hai phép scan và thêm một bước. Một phép reduce ghi ít đầu ra hơn nhiều so với một phép
            add; một phép add không cần cây nhiều mức với đồng bộ hóa giữa các mức. Điều nào quan trọng phụ thuộc vào thứ gì khan
            hiếm. Giữa các khối, đó là lưu lượng bộ nhớ toàn cục, nên reduce thắng. Giữa các warp hay các luồng, các phần tử đã
            nằm sẵn trong thanh ghi, nơi các lần truy cập thêm thì rẻ còn rào chắn thì không, nên scan-scan-add thắng.
          </p>
        }
      >
        <p>
          Both decompositions have two scans and one more step. A reduce writes far fewer outputs than an add; an add needs no
          multi-level tree with synchronisation between levels. Which matters depends on what is scarce. Across blocks it is
          global-memory traffic, so the reduce wins. Across warps or threads the elements are already in registers, where extra
          accesses are cheap and barriers are not, so scan-scan-add wins.
        </p>
      </Translated>
      <Notes>
      <Source section="11.9" pages="276" />
      <Today checked={tr('2026-11-20 against CCCL main (notes/today/ch11-facts.md A4)', '2026-11-20, đối chiếu với CCCL main (notes/today/ch11-facts.md A4)')}>
        <Translated
          vi={
            <p>
              <code>BlockScan</code> mặc định của CUB là một "raking reduce-then-scan"; tùy chọn{' '}
              <code>BLOCK_SCAN_WARP_SCANS</code> của nó là scan-scan-add của sách. Cả hai đều là lựa chọn được ghi trong tài
              liệu, nên sự ưu tiên của sách bên trong một khối là một nhận định bạn có thể đo được.
            </p>
          }
        >
          <p>
            CUB's default <code>BlockScan</code> is a "raking reduce-then-scan"; its <code>BLOCK_SCAN_WARP_SCANS</code> option is the
            book's scan-scan-add. Both are documented choices, so the book's preference inside a block is a judgement you can
            measure.
          </p>
        </Translated>
      </Today>
      </Notes>
      <Ask q={questionsS119['ch11/11.9/q-rss-why']} />

      <h2><Translated vi="Một kernel">One kernel</Translated></h2>
      <Translated
        vi={
          <p>
            Để đạt 8 byte mỗi phần tử, đầu vào phải được nạp một lần và đầu ra được ghi một lần, và không thiết kế nhiều kernel
            nào làm được điều đó: phải có thứ gì đó vượt qua ranh giới kernel qua bộ nhớ toàn cục. Bước giữa các khối phải diễn
            ra bên trong kernel, khi đoạn đã scan vẫn còn trên chip. Sách đưa ra ba cách (Fig. 11.16).
          </p>
        }
      >
        <p>
          To reach 8 bytes per element the input must be loaded once and the output stored once, and no multi-kernel design can
          do that: something must cross the kernel boundary through global memory. The inter-block step has to happen inside the
          kernel, while the scanned segment is still on chip. The book shows three ways (Fig. 11.16).
        </p>
      </Translated>
      <Source section="11.9" pages="276" />
      <Translated
        vi={
          <p>
            <b>Một rào chắn toàn lưới.</b> Mọi khối trao tổng của mình cho một khối, khối này scan chúng và trao kết quả lại, với
            một rào chắn toàn lưới trước và sau. Điều đó đòi mọi khối chạy cùng lúc, giới hạn số khối, và vì đoạn của mỗi khối bị
            giới hạn bởi bộ nhớ trên chip, nó cũng giới hạn kích thước mảng. Và sách đánh giá rào chắn toàn lưới là rất đắt, theo
            cả hai nghĩa.
          </p>
        }
      >
        <p>
          <b>A grid-wide barrier.</b> Every block hands its sum to one block, which scans them and hands the results back, with a
          grid-wide barrier before and after. That needs every block running at the same time, which caps the number of blocks,
          and since each block's segment is capped by on-chip memory, it caps the array size too. And the book judges grid-wide
          barriers very expensive, twice over.
        </p>
      </Translated>
      <Notes>
      <Source section="11.9" pages="276–277" figs="Fig. 11.16(a)" />
      <Today checked={tr('2026-11-20 (notes/today/infra-facts.md A4, B4; ch04-facts.md C3; ch11-facts.md, no vendor fact)', '2026-11-20 (notes/today/infra-facts.md A4, B4; ch04-facts.md C3; ch11-facts.md, không có thông tin từ nhà cung cấp)')}>
        <Translated
          vi={
            <p>
              CUDA và HIP cung cấp điều này qua cooperative launch, bị từ chối nếu không thể để cả lưới thường trú. Metal không có
              rào chắn lưới. Không tìm thấy con số chi phí nào của NVIDIA cho rào chắn này; "very expensive" là nhận định của sách.
            </p>
          }
        >
          <p>
            CUDA and HIP offer this through a cooperative launch, refused if the grid cannot all be resident. Metal has no grid
            barrier. No NVIDIA cost figure for the barrier was found; "very expensive" is the book's.
          </p>
        </Translated>
      </Today>
      </Notes>
      <Translated
        vi={
          <p>
            <b>Single lookback.</b> Mỗi khối chờ tổng đã scan của khối trước, cộng tổng của mình và công bố kết quả. Một khối chỉ
            bao giờ chờ các khối trước nó, điều mà sách gọi là đồng bộ hóa một chiều. Khác với rào chắn, nó không cần mọi khối
            thường trú cùng lúc; nó cần mọi khối mà nó chờ đã được lên lịch. Điểm yếu của nó là chuỗi: kết quả của khối cuối
            cùng đi qua cả N/S khối, mỗi lần một mắt xích.
          </p>
        }
      >
        <p>
          <b>Single lookback.</b> Each block waits for the previous block's scanned sum, adds its own and publishes the result.
          A block only ever waits for earlier blocks, which the book calls unidirectional synchronisation. Unlike a barrier, it
          does not need every block resident at once; it needs every block it waits for to have been scheduled. Its weakness is
          the chain: the last block's result passes through all N/S blocks, one link at a time.
        </p>
      </Translated>
      <Source section="11.9" pages="277, 279" figs="Fig. 11.16(b)" />
      <Translated
        vi={
          <p>
            <b>Multiple và decoupled lookback.</b> Nếu khối b thấy khối b − 1 đã có tổng riêng nhưng chưa có tổng đã scan, nó có
            thể nhìn lùi thêm một khối và tự cộng các mảnh. Decoupled lookback tiếp tục cho đến khi tìm thấy một tổng đã scan
            sẵn sàng, cộng các tổng thường của mọi khối ở giữa. Ít chờ hơn, với cái giá là những phép cộng mà một khối khác cũng
            làm. Ở mức cực đoan, Fig. 11.16(c), mọi khối cộng tổng của mọi khối trước nó: lại là phép scan ngây thơ của phần
            11.2, giờ là sự đánh đổi đúng, vì đồng bộ hóa tốn hơn tính toán rất nhiều và số tổng của khối ít hơn số phần tử rất
            nhiều.
          </p>
        }
      >
        <p>
          <b>Multiple and decoupled lookback.</b> If block b sees that block b − 1 has its own sum ready but not yet its scanned
          sum, it can look one further back and add the pieces itself. Decoupled lookback keeps going until it finds a ready
          scanned sum, adding the plain sums of every block in between. Less waiting, at the price of additions some other block
          also does. At the extreme, Fig. 11.16(c), every block adds all its predecessors' sums: the naive scan of section 11.2
          again, now the right trade, because synchronisation costs far more than computation and there are far fewer block sums
          than elements.
        </p>
      </Translated>
      <Source section="11.9" pages="279–280" figs="Fig. 11.16(c)" />
      <LookbackScene />
      <Translated
        vi={
          <p>
            Giờ hãy thử. Trình mô phỏng bên dưới chạy cùng mô hình với hoạt hình. Bắt đầu từ preset đầu tiên: nó chờ{' '}
            <code>blockIdx.x − 1</code> trong khi bộ lập lịch khởi động các khối sau trước.
          </p>
        }
      >
        <p>
          Now try it. The simulator below runs the same model as the animation. Start from the first preset: it waits on{' '}
          <code>blockIdx.x − 1</code> while the scheduler starts the later blocks first.
        </p>
      </Translated>
      <WaitSimulator />
      <Translated
        vi={
          <p>
            Ba điều cần tìm. Với <code>blockIdx.x</code> và các khối sau chạy trước, lần khởi chạy bị deadlock ngay khi G vượt
            quá R; khi mọi khối đều thường trú thì nó chạy xong. Bộ đếm loại bỏ deadlock theo quy tắc của CUDA với mọi thứ tự và
            kích thước, nhưng không phải khi không có bảo đảm tiến triển. Và trên một lần khởi chạy mà mọi khối đều thường trú,
            decoupled lookback cắt chuỗi xuống còn một hoặc hai mắt xích đổi lấy vài phép cộng lặp lại, trong khi với chỉ vài
            suất, các đợt khối dù sao cũng che phần lớn chuỗi.
          </p>
        }
      >
        <p>
          Three things to find. With <code>blockIdx.x</code> and later blocks first, the launch deadlocks as soon as G exceeds R;
          with every block resident it finishes. The counter removes the deadlock under CUDA's rule for every order and size, but
          not without a progress guarantee. And on a launch where every block is resident, decoupled lookback cuts the chain to
          one or two links for a handful of repeated additions, while with only a few slots the waves of blocks hide most of the
          chain anyway.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Các đoạn bằng nhau với jitter có seed, vòng trừu tượng, một khối đang quay chờ giữ suất của nó, một vòng look-back
              đọc mọi khối trước cùng lúc, không có hệ thống bộ nhớ. "Không bảo đảm" là một lịch chạy đối nghịch (chỉ các khối
              đang quay chờ được chạy), không phải mô hình của bất kỳ GPU nào.
            </p>
          }
        >
          <p>
            Equal segments with seeded jitter, abstract rounds, a spinning block holds its slot, a look-back round reads every
            predecessor at once, no memory system. "No guarantee" is one adversarial schedule (only spinning blocks run), not a
            model of any GPU.
          </p>
        </Translated>
      </Simplification>
      <Ask q={questionsS119['ch11/11.9/q-deadlock']} />

      <h2><Translated vi="Cờ">The flag</Translated></h2>
      <Translated
        vi={
          <p>
            Single lookback dưới dạng mã (Fig. 11.17): luồng cuối của mỗi khối giữ tổng của khối và đóng vai luồng dẫn đầu. Một
            mảng <code>flags</code> giữ mỗi khối một phần tử, bằng 0 cho đến khi tổng đã scan của khối đó nằm trong{' '}
            <code>partialSums</code>. Luồng dẫn đầu của khối b quay chờ trên <code>flags[b−1]</code>, đọc{' '}
            <code>partialSums[b−1]</code>, ghi tổng đã scan của chính nó và đặt cờ của mình. Các truy cập cờ là atomic, vì một
            khối khác ghi thứ mà khối này đọc, và có phạm vi device, để cập nhật được thấy giữa các khối.
          </p>
        }
      >
        <p>
          Single lookback in code (Fig. 11.17): the last thread of each block holds the block's sum and acts as its leader. A{' '}
          <code>flags</code> array holds one entry per block, 0 until that block's scanned sum is in <code>partialSums</code>. The
          leader of block b spins on <code>flags[b−1]</code>, reads <code>partialSums[b−1]</code>, stores its own scanned sum and
          sets its own flag. The flag accesses are atomic, because another block writes what this one reads, and device-scope,
          so the update is visible across blocks.
        </p>
      </Translated>
      <Source section="11.9" pages="278–279" figs="Fig. 11.17" />
      <Translated
        vi={
          <p>
            Bản thân tổng là một lệnh nạp và ghi thường; chỉ cờ là atomic. Hai thứ tự bộ nhớ làm cho điều đó an toàn. Lần đọc cờ
            dùng <b>acquire</b>, nên lệnh nạp tổng không thể xảy ra trước lần kiểm tra cờ. Lần ghi dùng <b>release</b>, nên lệnh
            ghi tổng không thể bị dời ra sau khi cờ được đặt. Hãy so sánh với{' '}
            <a href={href('ch09', '9.2')}>histogram của chương 9</a>, nơi relaxed là đủ vì không có truy cập nào khác phụ thuộc
            vào thứ tự.
          </p>
        }
      >
        <p>
          The sum itself is a plain load and store; only the flag is atomic. Two orderings make that safe. The read of the flag
          uses <b>acquire</b>, so the load of the sum cannot happen before the flag check. The write uses <b>release</b>, so the
          store of the sum cannot be moved after the flag is set. Compare{' '}
          <a href={href('ch09', '9.2')}>chapter 9's histogram</a>, where relaxed was enough because no other access depended on
          the order.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 against the CUDA Programming Guide v13.5, section 5.8.5 (notes/today/ch11-facts.md A5)', '2026-11-20, đối chiếu với CUDA Programming Guide v13.5, mục 5.8.5 (notes/today/ch11-facts.md A5)')}>
        <Translated
          vi={
            <p>
              Mô hình bộ nhớ của NVIDIA in giao thức này làm ví dụ "message passing" của họ. Nó đọc bằng{' '}
              <code>load(acquire)</code>; <code>fetch_add(0, acquire)</code> của sách chạy được nhưng đồng thời cũng là một lần
              ghi. Cờ và tổng phải được host đặt về 0 trước mỗi lần khởi chạy.
            </p>
          }
        >
          <p>
            NVIDIA's memory model prints this protocol as its "message passing" example. It reads with{' '}
            <code>load(acquire)</code>; the book's <code>fetch_add(0, acquire)</code> works but is a write as well. Flags and sums
            must be zeroed by the host before every launch.
          </p>
        </Translated>
      </Today>
      <FlagClassifier />
      <Ask q={questionsS119['ch11/11.9/q-acquire']} />
      <Ask q={questionsS119['ch11/11.9/q-scope']} />

      <h2><Translated vi="Phép scan một kernel">The single-kernel scan</Translated></h2>
      <Translated
        vi={
          <p>
            Fig. 11.18 ghép mọi thứ lại: một lần nạp được hợp nhất (coalesced), các phép scan luồng phân ô thanh ghi của phần
            11.7, một phép scan khối trên các tổng của luồng, lookback, một phép cộng trong thanh ghi, một lần ghi được hợp nhất.
            Bên trong một kernel không có gì phải đi ra bộ nhớ toàn cục giữa phép scan đầu tiên và phép cộng, nên lý do ưu tiên
            reduce-scan-scan không còn và kernel dùng scan-scan-add.
          </p>
        }
      >
        <p>
          Fig. 11.18 puts it together: a coalesced load, the register-tiled thread scans of section 11.7, a block scan of the
          thread sums, the lookback, one add in registers, a coalesced store. Inside one kernel nothing has to go to global memory
          between the first scan and the add, so the reason for preferring reduce-scan-scan is gone and the kernel uses
          scan-scan-add.
        </p>
      </Translated>
      <Source section="11.9" pages="280–282" figs="Fig. 11.18" />
      <Translated
        vi={
          <p>
            Còn một mảnh nữa dễ bị bỏ sót. Khối không dùng <code>blockIdx.x</code> làm chỉ số của nó. Luồng 0 tăng một bộ đếm
            toàn cục bằng atomic và chia sẻ giá trị cũ qua bộ nhớ chia sẻ; khối scan đoạn ứng với chỉ số đó. Vì các chỉ số được
            phát ra khi các khối thực sự bắt đầu, một khối không bao giờ chờ một khối chưa được lên lịch. Bộ đếm được host khởi
            tạo bằng 0 và truyền vào.
          </p>
        }
      >
        <p>
          One more piece is easy to miss. The block does not use <code>blockIdx.x</code> as its index. Thread 0 increments a
          global counter atomically and shares the old value through shared memory; the block scans the segment with that index.
          Because indices are handed out as blocks actually start, a block never waits on one that has not been scheduled. The
          counter is zero-initialised by the host and passed in.
        </p>
      </Translated>
      <Source section="11.9" pages="280" />
      <Translated
        vi={
          <p>
            Như được in, mã có ba chỗ nhầm: thiếu dấu phẩy giữa hai tham số cuối, nên nó không biên dịch được;{' '}
            <code>N</code> không bao giờ được dùng, nên N phải là bội số của đoạn khối; và dòng 12 dùng <code>blockDim.x</code>{' '}
            trong khi dòng 14 dùng <code>BLOCK_DIM</code>. Lab 11.5 là phiên bản đã sửa.
          </p>
        }
      >
        <p>
          As printed, the listing has three slips: a comma is missing between the last two parameters, so it does not compile;{' '}
          <code>N</code> is never used, so N must be a multiple of the block segment; and line 12 uses <code>blockDim.x</code>{' '}
          where line 14 uses <code>BLOCK_DIM</code>. Lab 11.5 is the corrected version.
        </p>
      </Translated>
      <Source section="11.9" pages="281" figs={tr('Fig. 11.18, lines 02, 12, 14', 'Fig. 11.18, dòng 02, 12, 14')} />
      <SinglePassListing />
      <Ask q={questionsS119['ch11/11.9/q-zero']} />

      <h2><Translated vi="Ai hứa hẹn sự tiến triển mà điều này cần">Who promises the progress this needs</Translated></h2>
      <Translated
        vi={
          <p>
            Bộ đếm bảo đảm khối bạn chờ đã bắt đầu. Điều đó chỉ đủ nếu một khối đã bắt đầu sau đó được bảo đảm tiếp tục chạy
            trong khi bạn quay chờ. Bảo đảm đó không có trong sách; nó là của nền tảng, và ba backend của khóa học này khác nhau.
          </p>
        }
      >
        <p>
          The counter makes sure the block you wait for has started. That is enough only if a started block is then guaranteed to
          keep running while you spin. That guarantee is not in the book; it is the platform's, and the three backends of this
          course differ.
        </p>
      </Translated>
      <Beyond title={tr('Not in the book: what each platform documents', 'Không có trong sách: mỗi nền tảng ghi gì trong tài liệu')}>
        <Translated vi={<p>Chỉ lấy từ thông tin của các nhà cung cấp. Các suy luận từ đó là của chúng tôi.</p>}>
          <p>From the vendor facts only. The derivations from them are ours.</p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            <b>CUDA có ghi trong tài liệu.</b> Mô hình thực thi của nó cho các luồng device forward progress song song: một khi
            một luồng đã thực thi một bước, nó tiếp tục tiến triển, và cả khối của nó cũng vậy. Một khối đã nhận chỉ số b − 1 đã
            thực thi một bước, nên việc chờ nó sẽ kết thúc. Chờ <code>blockIdx.x − 1</code> không có chỗ dựa như vậy. Vòng quay
            chờ phải là một lần đọc atomic, như của sách, và <code>__nanosleep</code> là cách được ghi trong tài liệu để lùi lại
            bên trong nó.
          </p>
        }
      >
        <p>
          <b>CUDA documents it.</b> Its execution model gives device threads parallel forward progress: once a thread has executed
          a step, it keeps making progress, and with it its whole block. A block that claimed index b − 1 has executed a step, so
          waiting for it ends. Waiting on <code>blockIdx.x − 1</code> has no such backing. The spin must be an atomic read, as the
          book's is, and <code>__nanosleep</code> is the documented way to back off inside it.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 against the CUDA Programming Guide v13.5, sections 5.9, 5.4.8.5; CCCL main (facts A6, A7)', '2026-11-20, đối chiếu với CUDA Programming Guide v13.5, mục 5.9, 5.4.8.5; CCCL main (facts A6, A7)')}>
        <Translated
          vi={
            <p>
              Chính CUB của NVIDIA không dùng bộ đếm: nó đánh chỉ số các tile bằng <code>blockIdx.x</code> dưới chú thích "Blocks
              are launched in increasing order", điều mà hướng dẫn không hứa.
            </p>
          }
        >
          <p>
            NVIDIA's own CUB does not use the counter: it indexes tiles by <code>blockIdx.x</code> under the comment "Blocks are
            launched in increasing order", which the guide does not promise.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            <b>HIP thì không.</b> AMD không ghi quy tắc forward progress nào giữa các workgroup. Thư viện của chính AMD mặc định
            dùng lookback với chỉ số khối phần cứng, chỉ chuyển sang bộ đếm trên một số kiến trúc như một "hotfix", và ngủ trong
            vòng quay chờ trên các chip MI100 đời đầu để tránh treo. Không có <code>__nanosleep</code> được ghi trong tài liệu.
            Trong khóa học này, kernel single-pass là <b>unverified</b> trên HIP; phép scan phân cấp là mặc định cho HIP.
          </p>
        }
      >
        <p>
          <b>HIP does not.</b> AMD documents no forward-progress rule between workgroups. AMD's own library uses lookback with the
          hardware block index by default, switches to the counter only on some architectures as a "hotfix", and sleeps in the
          spin on early MI100 parts to avoid a hang. There is no documented <code>__nanosleep</code>. In this course the
          single-pass kernel is <b>unverified</b> on HIP; the hierarchical scan is the HIP default.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 against HIP and rocPRIM at therock-10.2 (facts B3, B4)', '2026-11-20, đối chiếu với HIP và rocPRIM ở therock-10.2 (facts B3, B4)')}>
        <Translated
          vi={
            <p>
              rocPRIM chỉ bật bộ đếm cho gfx942, gfx950 và gfx1250, và gọi <code>__builtin_amdgcn_s_sleep</code> trên các bản
              gfx908 đời đầu.
            </p>
          }
        >
          <p>
            rocPRIM enables the counter for gfx942, gfx950 and gfx1250 only, and calls <code>__builtin_amdgcn_s_sleep</code> on
            early gfx908 revisions.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            <b>Metal thì không, và còn nói vì sao không nên dựa vào nó.</b> Đặc tả chỉ hứa tiến triển giữa các SIMD group của
            một threadgroup. Một kỹ sư Apple, khi trả lời về một phép tổng tiền tố dạng chuỗi trên diễn đàn của Apple, đã nói nó
            dựa vào hành vi không xác định và gọi tên việc lập lịch công bằng cho các threadgroup chạy cùng lúc là một trong
            những giả định không hợp lệ của nó. Bản thân cờ thì diễn đạt được, với thêm sự cẩn thận: các buffer được đọc giữa các
            threadgroup cần <code>coherent(device)</code>, và acquire cùng release chỉ có trong các overload của MSL 4.1 nhận{' '}
            <code>mem_flags</code> (hoặc dưới dạng atomic relaxed kèm fence từ MSL 3.2). Không có lệnh ngủ. Trên Metal, phép scan
            toàn cục là phép scan nhiều lần dispatch, lab 11.4.
          </p>
        }
      >
        <p>
          <b>Metal does not, and says why not to rely on it.</b> The specification promises progress only between SIMD-groups of
          one threadgroup. An Apple engineer, answering a chained prefix sum on Apple's forum, said it relied on undefined behaviour
          and named fair scheduling of threadgroups that run at the same time as one of its invalid assumptions. The flag itself is expressible, with extra care: buffers read across
          threadgroups need <code>coherent(device)</code>, and acquire and release exist only in MSL 4.1's overloads that take{' '}
          <code>mem_flags</code> (or as relaxed atomics with fences from MSL 3.2). There is no sleep. On Metal the global scan is the multi-dispatch one, lab 11.4.
        </p>
      </Translated>
      <Today checked={tr("2026-11-20 against the MSL specification 4.1 and Apple's developer forum (facts C3, C4, C5)", '2026-11-20, đối chiếu với đặc tả MSL 4.1 và diễn đàn nhà phát triển của Apple (facts C3, C4, C5)')}>
        <Translated
          vi={
            <p>
              Câu trả lời trên diễn đàn là của một kỹ sư, không phải tài liệu; cùng với sự im lặng của đặc tả, đó là nguồn mạnh
              nhất từ Apple tìm được. Một câu hỏi mới hơn trên diễn đàn về cùng điểm này chưa có trả lời vào ngày kiểm tra.
            </p>
          }
        >
          <p>
            The forum answer is an engineer's, not documentation; with the specification's silence it is the strongest Apple source
            found. A newer forum question on the same point had no reply on the check date.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS119['ch11/11.9/q-metal']} />

      <h2><Translated vi="Các thư viện làm gì">What the libraries do</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn sẽ hiếm khi tự viết kernel này cho production; sách cũng nói vậy. Phép scan toàn device của CUB đã dùng decoupled
            lookback từ năm 2013, và bài tập 5 là một phiên bản nhỏ của thứ nó đưa vào sản phẩm. Trên AMD, rocPRIM làm điều
            tương tự. Trên GPU của Apple, MPSGraph có hàm tổng tích lũy với các công tắc cho scan loại trừ (exclusive) và scan
            ngược.
          </p>
        }
      >
        <p>
          You will rarely write this kernel for production; the book says so too. CUB's device-wide scan has used decoupled
          lookback since 2013, and exercise 5 is a small version of what it ships. On AMD, rocPRIM does the same. On Apple GPUs,
          MPSGraph has a cumulative sum with switches for exclusive and reverse scans.
        </p>
      </Translated>
      <Today checked="2026-11-20 (facts A8, B2, C6)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              CUB và rocPRIM đều ghi trong tài liệu rằng một phép scan float có thể khác nhau giữa các lần chạy: CUB trừ khi yêu
              cầu tính tất định <code>run_to_run</code>{' '}, rocPRIM trừ khi bạn gọi <code>deterministic_inclusive_scan</code>.
            </p>
          }
        >
          <p>
            CUB and rocPRIM both document that a float scan may differ from run to run: CUB unless <code>run_to_run</code>{' '}
            determinism is requested, rocPRIM unless you call <code>deterministic_inclusive_scan</code>.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS119['ch11/11.9/q-ex5']} />

      <h2><Translated vi="Bên dưới một batch các hàng lấy mẫu">Underneath a batch of sampling rows</Translated></h2>
      <Translated
        vi={
          <p>
            Với một batch các hàng vocabulary, MLX và PyTorch giao mỗi hàng cho một khối hoặc threadgroup (PyTorch gói nhiều hàng
            ngắn vào một khối), khối này đi qua hàng theo từng chunk và mang tổng chạy từ chunk này sang chunk khác. Trên Metal,
            một hàng quá dài cho cách đó được chuyển sang các lần dispatch riêng: reduce-scan-carry trong PyTorch, scan-scan-add
            trong llama.cpp.
          </p>
        }
      >
        <p>
          For a batch of vocabulary rows, MLX and PyTorch give each row to one block or threadgroup (PyTorch packs several
          short rows into one block), which walks the row in chunks and carries the running total from chunk to chunk. On
          Metal a row too long for that goes to separate dispatches instead: reduce-scan-carry in PyTorch, scan-scan-add in
          llama.cpp.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Bản build CUDA của llama.cpp thì khác. Khi một hàng dài so với batch (độ dài của nó chia cho số hàng lớn hơn 4,096),
            nó giao mỗi hàng cho phép scan device của CUB, tức decoupled lookback. Một hàng Qwen3 gồm 151,936 token đủ điều kiện
            ở mọi kích thước batch đến 37. Nếu không, nó dùng các phép scan mức khối: <code>BlockScan</code> của CUB cho các hàng
            từ 1,024 trở lên, và kernel một-khối-mỗi-hàng của riêng nó dưới mức đó. Bản build HIP của nó không bao giờ dùng CUB
            và luôn chạy kernel của riêng nó. Vậy cơ chế của phần này thật sự chạy bên dưới top-p theo batch, bên trong một thư
            viện, ở batch nhỏ trên CUDA. Thứ bạn tự viết cho các hàng theo batch là một khối mỗi hàng kèm phần mang (carry): lab
            11.3 và 11.6.
          </p>
        }
      >
        <p>
          llama.cpp's CUDA build is different. When a row is long compared with the batch (its length divided by the number
          of rows is above 4,096), it hands each row to CUB's device scan, which is decoupled lookback. A Qwen3 row of 151,936
          tokens qualifies at every batch size up to 37. Otherwise it uses block-level scans: CUB's <code>BlockScan</code> for
          rows of 1,024 or more, its own one-block-per-row kernel below that. Its HIP build never uses CUB and always runs its
          own kernel. So this section's machinery does run under batched top-p, inside a library, at small batches on CUDA.
          What you write for batched rows yourself is one block per row with a carry: labs 11.3 and 11.6.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 against llama.cpp master, MLX main, PyTorch main (facts D1, D3, D4; ch01-facts D2; ch09-facts D7)', '2026-11-20, đối chiếu với llama.cpp master, MLX main, PyTorch main (facts D1, D3, D4; ch01-facts D2; ch09-facts D7)')}>
        <Translated
          vi={
            <p>
              llama.cpp dùng phép scan device của CUB khi <code>nrows == 1 &amp;&amp; ne00 &gt; 1024</code> hoặc{' '}
              <code>ne00 / nrows &gt; 4096</code>, và chỉ ở nơi CUB được biên dịch vào, điều này loại trừ HIP. Con số batch 37 là
              phép tính của chúng tôi. Mặc định llama.cpp lấy mẫu trên host; đây là đường đi backend thử nghiệm của nó.
            </p>
          }
        >
          <p>
            llama.cpp takes CUB's device scan when <code>nrows == 1 &amp;&amp; ne00 &gt; 1024</code> or{' '}
            <code>ne00 / nrows &gt; 4096</code>, and only where CUB is compiled in, which excludes HIP. The batch of 37 is
            our arithmetic. llama.cpp samples on the host by default; this is its experimental backend path.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Lab 11.4 xây dựng cả hai thiết kế ba kernel và đo chúng, trên mọi backend. Lab 11.5 xây dựng kernel single-pass cho
            họ CUDA. Cả hai đều có trên <a href={href('ch11', 'labs')}>trang lab</a>.
          </p>
        }
      >
        <p>
          Lab 11.4 builds both three-kernel designs and measures them, on every backend. Lab 11.5 builds the single-pass kernel
          for the CUDA family. Both are on the <a href={href('ch11', 'labs')}>labs page</a>.
        </p>
      </Translated>
    </>
  );
}
