import { useEffect, useRef, useState, type ReactNode, type RefObject } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { fmt } from '../../lib/reports';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';
import {
  DECODE_STEP,
  MD_STEP,
  SPEEDUPS,
  ceiling,
  moduleSpeedup,
  nearestIndex,
  overlapped,
  rebalance,
  speedup,
  type AppModule,
  type ModuleResult,
  type Place,
} from '../../sim/amdahl';
import {
  FILLS,
  HIP_DEFAULT_QUEUES,
  KERNEL_COUNTS,
  LAUNCH_RATIOS,
  compare,
  simulate,
  type LanesInput,
  type LanesResult,
  type Mode,
} from '../../sim/tasklanes';

// ---------------------------------------------------------------- formatting
/** A speed-up: "16.8×", "1,000×", or "unbounded". */
const times = (v: number) => (isFinite(v) ? fmt(Number(v.toPrecision(6))) + '×' : tr('unbounded', 'không giới hạn'));
/** A fraction as a percentage, three significant digits: "5.95%". */
const pct = (v: number) => fmt(v * 100, 3) + '%';
/** A time in units of one kernel's duration d. */
const dur = (v: number) => fmt(v, 3) + ' d';

// The cases the page works with. Every number shown for them comes from the models.
const P_OVERLAP = { p: 0.9, s: 50 };
const BOOK = { p: 0.95, s: 100 };

// ---------------------------------------------------------------- small helpers (this section only)
/** Width of an element in CSS pixels, so a drawing keeps legible text on a phone. */
function useWidth(initial: number, min = 280): [RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(initial);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const read = () => setW(Math.max(min, Math.round(el.clientWidth)));
    read();
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, [min]);
  return [ref, w];
}

const SWATCH: Record<Place, string> = { host: 'var(--ink)', device: 'var(--ink-3)' };

// ---------------------------------------------------------------- explorer 1: which modules go on the GPU
type PresetKey = 'md' | 'decode';
const PRESETS: Record<PresetKey, { label: string; modules: AppModule[]; share: string }> = {
  md: {
    get label() { return tr("The book's time step (pp. 537–538)", 'Bước thời gian của sách (pp. 537–538)'); },
    modules: MD_STEP,
    get share() { return tr('Share of the original time', 'Tỉ lệ trong thời gian ban đầu'); },
  },
  decode: {
    get label() { return tr('A decode step (ours)', 'Một bước decode (của khóa học này)'); },
    modules: DECODE_STEP,
    get share() { return tr("Share of today's step", 'Tỉ lệ trong bước hiện nay'); },
  },
};

/** One bar of spans placed in time; `scale` is the time the full width stands for. */
function SpanBar({ label, figure, spans, scale, lane }: {
  label: string;
  figure: ReactNode;
  spans: { module: number; start: number; end: number; place: Place }[];
  scale: number;
  lane: string;
}) {
  return (
    <div className="row" style={{ gap: '0.3rem 1rem', marginBottom: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
      <div style={{ flex: '0 0 11rem', lineHeight: 1.25 }}>
        <b>{label}</b>
        <div className="muted" style={{ fontSize: '0.76rem' }}>{figure}</div>
      </div>
      <div
        data-lane={lane}
        style={{
          flex: '1 1 14rem', position: 'relative', height: '1.6rem', background: 'var(--cell)',
          border: '1px solid var(--rule)', borderRadius: 2, overflow: 'hidden', minWidth: '12rem',
        }}
      >
        {spans
          .filter((s) => s.end > s.start)
          .map((s) => {
            const left = (s.start / scale) * 100;
            const width = ((s.end - s.start) / scale) * 100;
            return (
              <div
                key={s.module}
                data-span={s.module + 1}
                title={tr(`module ${s.module + 1}`, `mô-đun ${s.module + 1}`)}
                style={{
                  position: 'absolute', top: 0, bottom: 0, left: `${left}%`, width: `${width}%`,
                  background: SWATCH[s.place], borderRight: '1px solid var(--page)', color: 'var(--page)',
                  fontSize: '0.72rem', fontWeight: 600, display: 'grid', placeItems: 'center', overflow: 'hidden',
                }}
              >
                {width >= 3.2 ? s.module + 1 : ''}
              </div>
            );
          })}
      </div>
    </div>
  );
}

function ModuleCard({
  m, i, shareLabel, onChange,
}: {
  m: AppModule;
  i: number;
  shareLabel: string;
  onChange: (patch: Partial<AppModule> & { share?: number }) => void;
}) {
  const onDevice = m.place === 'device';
  return (
    <div className="panel" data-module={i + 1} style={{ borderLeft: `5px solid ${SWATCH[m.place]}`, display: 'grid', gap: '0.55rem' }}>
      <b style={{ lineHeight: 1.25 }}>
        {i + 1}. <Translated vi={({ 'Neighbour list': 'Danh sách lân cận', 'Non-bonded forces': 'Lực không liên kết', 'Vibrational and rotational forces': 'Lực dao động và lực quay', 'Update positions and velocities': 'Cập nhật vị trí và vận tốc', 'Projections and attention': 'Các phép chiếu và attention', 'Elementwise kernels': 'Các kernel theo từng phần tử', Sampling: 'Lấy mẫu', 'Kernel launches': 'Khởi chạy kernel', 'Detokenise and stream out': 'Giải token và stream đầu ra' } as Record<string, string>)[m.name] ?? m.name}>{m.name}</Translated>
      </b>
      <label className="field">
        <span>
          {shareLabel}: <b data-out="share">{pct(m.share)}</b>
        </span>
        <input
          type="range" min="0" max="100" step="0.5" value={Math.round(m.share * 200) / 2}
          aria-label={tr(`share of ${m.name}`, `tỉ lệ của ${({ 'Neighbour list': 'Danh sách lân cận', 'Non-bonded forces': 'Lực không liên kết', 'Vibrational and rotational forces': 'Lực dao động và lực quay', 'Update positions and velocities': 'Cập nhật vị trí và vận tốc', 'Projections and attention': 'Các phép chiếu và attention', 'Elementwise kernels': 'Các kernel theo từng phần tử', Sampling: 'Lấy mẫu', 'Kernel launches': 'Khởi chạy kernel', 'Detokenise and stream out': 'Giải token và stream đầu ra' } as Record<string, string>)[m.name] ?? m.name}`)}
          onChange={(e) => onChange({ share: +e.target.value / 100 })}
        />
      </label>
      <div className="row" style={{ gap: '0.4rem 0.8rem' }}>
        <span className="seg" data-place>
          <button aria-pressed={!onDevice} disabled={m.fixed === 'device'} onClick={() => onChange({ place: 'host' })}><Translated vi="host">host</Translated></button>
          <button aria-pressed={onDevice} disabled={m.fixed === 'host'} onClick={() => onChange({ place: 'device' })}><Translated vi="GPU">GPU</Translated></button>
        </span>
        {m.fixed ? <span className="muted" style={{ fontSize: '0.78rem' }}><Translated vi={<>ở lại trên {m.fixed === 'host' ? 'host' : 'GPU'}</>}>stays on the {m.fixed === 'host' ? 'host' : 'GPU'}</Translated></span> : null}
      </div>
      <label className="field" style={{ opacity: onDevice ? 1 : 0.55 }}>
        <Translated
          vi={
            <span>
              Trên GPU, nhanh hơn <b data-out="s">{times(m.s)}</b>
            </span>
          }
        >
          <span>
            On the GPU, <b data-out="s">{times(m.s)}</b> faster
          </span>
        </Translated>
        <input
          type="range" min="0" max={SPEEDUPS.length - 1} step="1" value={nearestIndex(SPEEDUPS, m.s)} disabled={!onDevice}
          aria-label={tr(`speed-up of ${m.name} on the GPU`, `mức tăng tốc của ${({ 'Neighbour list': 'Danh sách lân cận', 'Non-bonded forces': 'Lực không liên kết', 'Vibrational and rotational forces': 'Lực dao động và lực quay', 'Update positions and velocities': 'Cập nhật vị trí và vận tốc', 'Projections and attention': 'Các phép chiếu và attention', 'Elementwise kernels': 'Các kernel theo từng phần tử', Sampling: 'Lấy mẫu', 'Kernel launches': 'Khởi chạy kernel', 'Detokenise and stream out': 'Giải token và stream đầu ra' } as Record<string, string>)[m.name] ?? m.name} trên GPU`)}
          onChange={(e) => onChange({ s: SPEEDUPS[+e.target.value] })}
        />
      </label>
      <label className="check" style={{ opacity: onDevice ? 0.55 : 1 }}>
        <input type="checkbox" checked={!!m.chained} disabled={onDevice} aria-label={tr(`${m.name} chained to the device`, `${({ 'Neighbour list': 'Danh sách lân cận', 'Non-bonded forces': 'Lực không liên kết', 'Vibrational and rotational forces': 'Lực dao động và lực quay', 'Update positions and velocities': 'Cập nhật vị trí và vận tốc', 'Projections and attention': 'Các phép chiếu và attention', 'Elementwise kernels': 'Các kernel theo từng phần tử', Sampling: 'Lấy mẫu', 'Kernel launches': 'Khởi chạy kernel', 'Detokenise and stream out': 'Giải token và stream đầu ra' } as Record<string, string>)[m.name] ?? m.name} bị buộc với device`)}
          onChange={(e) => onChange({ chained: e.target.checked })} />
        <Translated vi="bị buộc với device">chained to the device</Translated>
      </label>
    </div>
  );
}

function ModuleTimes({ mods, r }: { mods: AppModule[]; r: ModuleResult }) {
  let t = 0;
  const before = mods.map((m, i) => {
    const s = { module: i, start: t, end: t + m.share, place: m.place };
    t += m.share;
    return s;
  });
  const after = (lane: Place) => r.spans.filter((s) => s.lane === lane).map((s) => ({ ...s, place: lane }));
  return (
    <div className="panel" data-timebars style={{ marginBottom: '1rem' }}>
      <SpanBar label={tr('Before', 'Trước')} figure={tr('the original run, 100%; colour shows where each module goes now', 'lần chạy ban đầu, 100%; màu cho biết mỗi mô-đun giờ chạy ở đâu')} spans={before} scale={1} lane="before" />{/* i18n-ignore: data key */}
      <SpanBar
        label={tr('After: host', 'Sau: host')}
        figure={
          <Translated vi={<>đã kéo giãn: toàn bộ chiều rộng là <span data-out="after-total">{pct(r.total)}</span> thời gian ban đầu</>}>
            <>stretched: the full width is <span data-out="after-total">{pct(r.total)}</span> of the original time</>
          </Translated>
        }
        spans={after('host')} scale={r.total} lane="host" // i18n-ignore: data key
      />
      <SpanBar label={tr('After: GPU', 'Sau: GPU')} figure={tr('same scale as the host row', 'cùng tỉ lệ với hàng host')} spans={after('device')} scale={r.total} lane="device" />{/* i18n-ignore: data key */}
      <span className="legend" style={{ marginTop: '0.3rem' }}>
        <span><i style={{ background: SWATCH.host }} /><Translated vi="chạy trên host">runs on the host</Translated></span>
        <span><i style={{ background: SWATCH.device }} /><Translated vi="chạy trên GPU">runs on the GPU</Translated></span>
        <span><Translated vi="các số là các mô-đun bên dưới">numbers are the modules below</Translated></span>
      </span>
    </div>
  );
}

function ModulesBench() {
  const [preset, setPreset] = useState<PresetKey>('md');
  const [mods, setMods] = useState<AppModule[]>(MD_STEP);
  const [overlap, setOverlap] = useState(false);
  const r = moduleSpeedup(mods, overlap);
  const load = (k: PresetKey) => {
    setPreset(k);
    setMods(PRESETS[k].modules);
    setOverlap(false);
  };
  const change = (i: number) => (patch: Partial<AppModule> & { share?: number }) =>
    setMods((ms) => {
      const { share, ...rest } = patch;
      const next = share === undefined ? ms : rebalance(ms, i, share);
      return next.map((m, j) => (j === i ? { ...m, ...rest } : m));
    });
  const unchanged = JSON.stringify(mods) === JSON.stringify(PRESETS[preset].modules);
  return (
    <Bench
      title={tr('Which modules go on the GPU', 'Những mô-đun nào lên GPU')}
      task={tr('Find a placement whose ceiling is above 20×, and say what it had to move.', 'Tìm một cách bố trí có trần trên 20×, và nói nó đã phải chuyển những gì.')}
      foot={
        <Translated
          vi={
            <>
              Thanh trên cùng là lần chạy ban đầu, mọi thứ trên host theo tỉ lệ của từng mô-đun. Hai hàng bên dưới là lần chạy
              mới, được kéo giãn sao cho toàn bộ chiều rộng là độ dài của nó; khi bật chồng lấp, hàng GPU chạy song song với
              hàng host. Một mô-đun bị buộc thì cấp dữ liệu cho GPU hoặc chờ kết quả của nó trong cùng một bước, nên không thể
              chồng lấp. Kéo một thanh trượt tỉ lệ sẽ co giãn các mô-đun khác để tổng tỉ lệ vẫn là 100%. Trong thiết lập sẵn của
              sách, cách chia 5% và mức 4× cho ba mô-đun nhỏ là của khóa học này; thiết lập sẵn decode hoàn toàn là của khóa học
              này.
            </>
          }
        >
          <>
            The top bar is the original run, everything on the host at the module's share. The two rows below are the new
            run, stretched so the full width is its length; with overlap on, the GPU row runs beside the host row. A chained
            module feeds the GPU or waits for its result within one step, so it cannot overlap. Moving a share slider scales
            the other modules so the shares still add up to 100%. In the book's preset, the split of the 5% and the 4× for
            the three small modules are ours; the decode preset is entirely ours.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <div className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
          <Translated vi="Thiết lập sẵn">Preset</Translated>
          <span className="seg" data-presets>
            {(Object.keys(PRESETS) as PresetKey[]).map((k) => (
              <button key={k} aria-pressed={preset === k} onClick={() => load(k)}>
                {PRESETS[k].label}
              </button>
            ))}
          </span>
        </div>
        <label className="check">
          <input type="checkbox" aria-label={tr('host and device overlap', 'host và device chồng lấp')} checked={overlap} onChange={(e) => setOverlap(e.target.checked)} />
          <Translated vi="host và GPU chồng lấp">host and GPU overlap</Translated>
        </label>
      </div>
      <ModuleTimes mods={mods} r={r} />
      <div className="cols plot">
        <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(13.5rem, 1fr))', gap: '0.7rem' }}>
          {mods.map((m, i) => (
            <ModuleCard key={i} m={m} i={i} shareLabel={PRESETS[preset].share} onChange={change(i)} />
          ))}
        </div>
        <div className="panel" data-readout>
          <p className="muted" style={{ margin: 0 }}><Translated vi="Mức tăng tốc của ứng dụng">Application speed-up</Translated></p>
          <p style={{ fontSize: '1.9rem', fontWeight: 600, margin: '0 0 0.5rem', lineHeight: 1.15 }} data-out="overall">
            {times(r.speedup)}
          </p>
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              <tr>
                <td><Translated vi="Thời gian host, so với ban đầu">Host time, of the original</Translated></td>
                <td className="num" data-out="host">{pct(r.host)}</td>
              </tr>
              <tr>
                <td><Translated vi="Thời gian GPU, so với ban đầu">GPU time, of the original</Translated></td>
                <td className="num" data-out="device">{pct(r.device)}</td>
              </tr>
              {overlap ? (
                <tr>
                  <td><Translated vi="Thời gian GPU được giấu sau phần việc host không bị buộc">GPU time hidden behind unchained host work</Translated></td>
                  <td className="num" data-out="hidden">{pct(r.hidden)}</td>
                </tr>
              ) : null}
              <tr>
                <td><Translated vi="Thời gian chạy mới, so với ban đầu">New run time, of the original</Translated></td>
                <td className="num" data-out="total">{pct(r.total)}</td>
              </tr>
              <tr>
                <td><Translated vi="Trần, 1 / thời gian host">Ceiling, 1 / host time</Translated></td>
                <td className="num" data-out="ceiling">{times(r.ceiling)}</td>
              </tr>
            </tbody>
          </table>
          <p className="muted" style={{ margin: '0.6rem 0 0' }}>
            {preset === 'md' && unchanged
              ? tr(
                  "The book's setting. The page prints 17× without overlap (exactly 16.8×) and 20× with it.",
                  'Thiết lập của sách. Trang sách in 17× khi không chồng lấp (chính xác là 16.8×) và 20× khi có.',
                )
              : r.host > 0 && r.total <= r.host * 1.0001
                ? tr(
                    'The GPU is fully hidden: only the host part is left, so this is the ceiling.',
                    'GPU được giấu hoàn toàn: chỉ còn phần host, nên đây là trần.',
                  )
                : r.host === 0
                  ? tr(
                      'Nothing is left on the host, so there is no ceiling from host work.',
                      'Không còn gì trên host, nên không có trần do phần việc host.',
                    )
                  : tr(`The host part is ${pct(r.host / r.total)} of the new run time.`, `Phần host chiếm ${pct(r.host / r.total)} thời gian chạy mới.`)}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer 2: many small kernels
interface LanesState {
  k: number;
  f: number;
  ratio: number;
  mode: Mode;
  queues: number;
  pairs: boolean;
  concurrent: boolean;
  grouped: boolean;
}
const MODES: { mode: Mode; label: string }[] = [
  { mode: 'stream', get label() { return tr('One stream', 'Một stream'); } },
  { mode: 'streams', get label() { return tr('A stream each', 'Mỗi kernel một stream'); } },
  { mode: 'encoder', get label() { return tr('Metal encoder', 'Encoder của Metal'); } },
  { mode: 'graph', get label() { return tr('Graph', 'Graph'); } },
];
const compareKey = (s: LanesState) =>
  s.mode === 'streams' ? (isFinite(s.queues) ? 'streams4' : 'streams') : s.mode === 'encoder' ? (s.concurrent ? 'encoder' : 'serial') : s.mode;

function LanesPlot({ r, scale }: { r: LanesResult; scale: number }) {
  const [ref, W] = useWidth(720);
  const labelW = W < 520 ? 66 : 96;
  const rowH = 22;
  const gap = 6;
  const top = 16;
  const occH = 34;
  const rowsN = 1 + r.rows.length;
  const occY = top + rowsN * (rowH + gap) + 4;
  const H = occY + occH + 34;
  const x0 = labelW;
  const x1 = W - 10;
  const X = (t: number) => x0 + (Math.min(t, scale) / scale) * (x1 - x0);
  const rowY = (j: number) => top + j * (rowH + gap);
  const step = scale <= 12 ? 1 : scale <= 30 ? 2 : scale <= 60 ? 5 : 10;
  const ticks: number[] = [];
  for (let t = 0; t <= scale + 1e-9; t += step) ticks.push(t);
  const occ = r.occupancy;
  const occPath =
    `M${X(0)},${occY + occH} ` +
    occ.map((o, j) => {
      const next = j + 1 < occ.length ? occ[j + 1].t : r.total;
      const y = occY + occH - o.fill * occH;
      return `L${X(o.t).toFixed(1)},${y.toFixed(1)} L${X(next).toFixed(1)},${y.toFixed(1)}`;
    }).join(' ') +
    ` L${X(r.total)},${occY + occH} Z`;
  return (
    <div ref={ref} style={{ width: '100%', overflow: 'hidden' }}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Timeline of the host and the device', 'Dòng thời gian của host và device')} style={{ display: 'block' }}>
        <rect x={x0} y={top - 4} width={x1 - x0} height={occY + occH - top + 4} fill="var(--page)" stroke="var(--rule)" />
        {ticks.map((t) => (
          <g key={t}>
            <line x1={X(t)} x2={X(t)} y1={top - 4} y2={occY + occH} stroke="var(--rule-soft)" />
            <text x={X(t)} y={occY + occH + 14} fontSize="11" textAnchor="middle">{t}</text>
          </g>
        ))}
        <text x={(x0 + x1) / 2} y={H - 4} fontSize="11.5" textAnchor="middle"><Translated vi="thời gian, theo đơn vị thời lượng d của một kernel">time, in units of one kernel's duration d</Translated></text>
        {/* host lane */}
        <text x={x0 - 6} y={rowY(0) + rowH / 2 + 4} fontSize="11" textAnchor="end"><Translated vi="host">host</Translated></text>
        {r.host.map((h, j) => {
          const w = X(h.end) - X(h.start);
          return (
            <g key={'h' + j} data-host={h.kind}>
              <rect x={X(h.start)} y={rowY(0)} width={Math.max(1, w - 0.6)} height={rowH} fill="var(--ink)" />
              {w >= 16 ? (
                <text x={X(h.start) + w / 2} y={rowY(0) + rowH / 2 + 4} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--page)' }}>
                  {h.kind === 'graph' ? (w >= 40 ? tr('graph', 'graph') : 'g') : h.i}
                </text>
              ) : null}
            </g>
          );
        })}
        {r.commit !== null ? (
          <g data-commit>
            <line x1={X(r.commit)} x2={X(r.commit)} y1={rowY(0) - 3} y2={rowY(0) + rowH + 3} stroke="var(--ink)" strokeWidth="2" />
            <text x={X(r.commit) + 4} y={rowY(0) - 2} fontSize="10.5"><Translated vi="commit">commit</Translated></text>
          </g>
        ) : null}
        {/* device rows */}
        {r.rows.map((label, j) => (
          <text key={'l' + j} x={x0 - 6} y={rowY(j + 1) + rowH / 2 + 4} fontSize="11" textAnchor="end">{label}</text>
        ))}
        {r.kernels.map((k) => {
          const w = X(k.end) - X(k.start);
          return (
            <g key={'k' + k.i} data-kernel={k.i}>
              <rect x={X(k.start)} y={rowY(k.row + 1)} width={Math.max(1, w - 1)} height={rowH} rx="2"
                fill="var(--ink-3)" stroke={k.dependsOn !== null ? 'var(--ink)' : 'none'} strokeDasharray={k.dependsOn !== null ? '3 2' : undefined} />
              {w >= 16 ? (
                <text x={X(k.start) + w / 2} y={rowY(k.row + 1) + rowH / 2 + 4} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--page)' }}>
                  {k.i}
                </text>
              ) : null}
            </g>
          );
        })}
        {r.barriers.map((b, j) => (
          <g key={'b' + j} data-barrier>
            <line x1={X(b)} x2={X(b)} y1={rowY(1) - 3} y2={rowY(rowsN) - 3} stroke="var(--ink)" strokeWidth="1.4" strokeDasharray="4 3" />
            {j === 0 ? <text x={X(b) + 3} y={rowY(1) - 4} fontSize="10"><Translated vi="rào chắn">barrier</Translated></text> : null}
          </g>
        ))}
        {/* occupancy */}
        <text x={x0 - 6} y={occY + occH / 2 - 2} fontSize="11" textAnchor="end"><Translated vi="device">device</Translated></text>
        <text x={x0 - 6} y={occY + occH / 2 + 11} fontSize="11" textAnchor="end"><Translated vi="đang dùng">in use</Translated></text>
        <line x1={x0} x2={x1} y1={occY} y2={occY} stroke="var(--rule)" strokeDasharray="2 3" />
        <text x={x1 - 3} y={occY + 10} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>100%</text>
        <path d={occPath} fill="var(--ink-3)" fillOpacity="0.45" stroke="var(--ink-3)" strokeWidth="1" />
      </svg>
    </div>
  );
}

function ManyKernelsBench() {
  const [s, setS] = useState<LanesState>({ k: 8, f: 0.25, ratio: 0.2, mode: 'stream', queues: Infinity, pairs: false, concurrent: true, grouped: false });
  const set = (patch: Partial<LanesState>) => setS((v) => ({ ...v, ...patch }));
  const base = { k: s.k, f: s.f, d: 1, L: s.ratio, pairs: s.pairs, grouped: s.grouped };
  const input: LanesInput = { ...base, mode: s.mode, queues: s.queues, concurrent: s.concurrent };
  const r = simulate(input);
  const all = compare(base);
  const scale = Math.max(...all.map((x) => x.total));
  const current = compareKey(s);
  const segs = <T,>(name: string, value: T, options: { v: T; label: string }[], onPick: (v: T) => void, data: string) => (
    <div className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
      {name}
      <span className="seg" data-seg={data}>
        {options.map((o) => (
          <button key={o.label} aria-pressed={o.v === value} onClick={() => onPick(o.v)}>{o.label}</button>
        ))}
      </span>
    </div>
  );
  return (
    <Bench
      title={tr('Many small kernels', 'Nhiều kernel nhỏ')}
      task={tr(
        'Find a setting where streams beat one stream by 4× or more, then one where nothing but the graph helps.',
        'Tìm một thiết lập mà nhiều stream thắng một stream từ 4× trở lên, rồi một thiết lập mà chỉ graph mới giúp được.',
      )}
      foot={
        <Translated
          vi={
            <>
              Hàng host: mỗi lần khởi chạy, mỗi dispatch Metal được encode, hoặc lần khởi chạy duy nhất của graph là một vạch.
              Các hàng device: mỗi stream hoặc hàng đợi phần cứng một hàng, hoặc các dispatch của encoder khi chúng chồng lấp.
              Viền nét đứt đánh dấu một kernel đọc kernel ngay trước nó; các đường dọc nét đứt là rào chắn của Metal. Dải dưới
              cùng là phần device đang được dùng. Trục thời gian giống nhau cho mọi chế độ, nên chuyển chế độ là so sánh cùng
              một thước đo.
            </>
          }
        >
          <>
            Host row: one mark per launch, per encoded Metal dispatch, or the graph's single launch. Device rows: one per
            stream or hardware queue, or the encoder's dispatches as they overlap. A dashed outline marks a kernel that reads
            the one before it; dashed vertical lines are Metal barriers. The bottom strip is the fraction of the device in use.
            The time axis is the same for every mode, so switching modes compares like with like.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi={<span>Số kernel k = <b data-out="k">{s.k}</b></span>}>
            <span>Kernels k = <b data-out="k">{s.k}</b></span>
          </Translated>
          <input type="range" aria-label={tr('number of kernels', 'số kernel')} min="0" max={KERNEL_COUNTS.length - 1} step="1"
            value={KERNEL_COUNTS.indexOf(s.k)} onChange={(e) => set({ k: KERNEL_COUNTS[+e.target.value] })} style={{ width: '11rem' }} />
        </label>
        <label className="field">
          <Translated vi={<span>Mỗi kernel lấp f = <b data-out="f">{pct(s.f)}</b> device</span>}>
            <span>Each fills f = <b data-out="f">{pct(s.f)}</b> of the device</span>
          </Translated>
          <input type="range" aria-label={tr('fill fraction', 'tỉ lệ lấp đầy')} min="0" max={FILLS.length - 1} step="1"
            value={FILLS.indexOf(s.f)} onChange={(e) => set({ f: FILLS[+e.target.value] })} style={{ width: '11rem' }} />
        </label>
        <label className="field">
          <Translated vi={<span>Chi phí khởi chạy L = <b data-out="ratio">{fmt(s.ratio)}</b> d</span>}>
            <span>Launch cost L = <b data-out="ratio">{fmt(s.ratio)}</b> d</span>
          </Translated>
          <input type="range" aria-label={tr('launch cost over kernel duration', 'chi phí khởi chạy so với thời lượng kernel')} min="0" max={LAUNCH_RATIOS.length - 1} step="1"
            value={LAUNCH_RATIOS.indexOf(s.ratio)} onChange={(e) => set({ ratio: LAUNCH_RATIOS[+e.target.value] })} style={{ width: '11rem' }} />
        </label>
        {segs(tr('Dependencies', 'Phụ thuộc'), s.pairs, [{ v: false, label: tr('none', 'không có') }, { v: true, label: tr('pairs', 'theo cặp') }], (v) => set({ pairs: v }), 'deps')}
      </div>
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        {segs(tr('How they are issued', 'Cách phát lệnh'), s.mode, MODES.map((m) => ({ v: m.mode, label: m.label })), (v) => set({ mode: v }), 'mode')}
        {s.mode === 'streams'
          ? segs(tr('Hardware queues', 'Hàng đợi phần cứng'), s.queues, [{ v: Infinity, label: tr('unlimited', 'không giới hạn') }, { v: HIP_DEFAULT_QUEUES, label: tr(`${HIP_DEFAULT_QUEUES} (HIP's default)`, `${HIP_DEFAULT_QUEUES} (mặc định của HIP)`) }], (v) => set({ queues: v }), 'queues')
          : null}
        {s.mode === 'encoder'
          ? segs(tr('Dispatch type', 'Kiểu dispatch'), s.concurrent, [{ v: true, label: tr('concurrent', 'đồng thời') }, { v: false, label: tr('serial', 'tuần tự') }], (v) => set({ concurrent: v }), 'dispatch')
          : null}
        {s.mode === 'encoder' && s.concurrent && s.pairs
          ? segs(tr('Encode order', 'Thứ tự encode'), s.grouped, [{ v: false, label: tr('as issued', 'theo thứ tự phát') }, { v: true, label: tr('producers first', 'bên tạo dữ liệu trước') }], (v) => set({ grouped: v }), 'order')
          : null}
      </div>
      <div className="panel" style={{ marginBottom: '1rem' }}>
        <LanesPlot r={r} scale={scale} />
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 3fr)' }}>
        <div className="panel" data-readout>
          <p className="muted" style={{ margin: 0 }}><Translated vi={<>Thời gian cho cả {s.k} kernel</>}>Time for all {s.k} kernels</Translated></p>
          <p style={{ fontSize: '1.7rem', fontWeight: 600, margin: '0 0 0.5rem', lineHeight: 1.15 }} data-out="total">{dur(r.total)}</p>
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              <tr>
                <td><Translated vi="Phần device được dùng, trung bình">Device in use, on average</Translated></td>
                <td className="num" data-out="util">{pct(r.utilisation)}</td>
              </tr>
              <tr>
                <td><Translated vi="Số kernel chạy cùng lúc nhiều nhất">Most kernels running at once</Translated></td>
                <td className="num" data-out="max">{r.maxConcurrent}</td>
              </tr>
              <tr>
                <td><Translated vi="Device rảnh trong khi host vẫn đang phát lệnh">Device idle while the host is still issuing</Translated></td>
                <td className="num nowrap" data-out="idle">{dur(r.idleWhileLaunching)}</td>
              </tr>
              {r.barriers.length ? (
                <tr>
                  <td><Translated vi="Rào chắn">Barriers</Translated></td>
                  <td className="num" data-out="barriers">{r.barriers.length}</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <table className="data" style={{ margin: 0 }} data-compare>
            <thead>
              <tr>
                <th><Translated vi="Cùng các kernel, phát lệnh theo kiểu">Same kernels, issued as</Translated></th>
                <th className="num"><Translated vi="Thời gian">Time</Translated></th>
                <th className="num"><Translated vi="Đang dùng">In use</Translated></th>
                <th className="num"><Translated vi="Cùng lúc">At once</Translated></th>
              </tr>
            </thead>
            <tbody>
              {all.map((x) => (
                <tr key={x.key} data-key={x.key} style={x.key === current ? { background: 'var(--rule-soft)', fontWeight: 600 } : undefined}>
                  <td><Translated vi={({ stream: 'Một stream', streams: 'Mỗi kernel một stream, không giới hạn hàng đợi', streams4: `Mỗi kernel một stream, ${HIP_DEFAULT_QUEUES} hàng đợi (mặc định của HIP)`, encoder: 'Encoder đồng thời của Metal', serial: 'Encoder tuần tự của Metal', graph: 'Graph, khởi chạy một lần' } as Record<string, string>)[x.key] ?? x.label}>{x.label}</Translated></td>
                  <td className="num nowrap" data-cell="total">{dur(x.total)}</td>
                  <td className="num">{pct(x.utilisation)}</td>
                  <td className="num">{x.maxConcurrent}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS223: Record<string, Question> = {
  'ch22/22.3/q-when-concurrent': {
    kind: 'choice',
    id: 'ch22/22.3/q-when-concurrent',
    prompt: (
      <Translated
        vi={
          <>
            Bạn có sáu kernel nhỏ, độc lập trong một bước và đặt mỗi kernel vào một stream riêng để chúng có thể chạy cùng lúc.
            Khi nào việc đó làm bước ngắn lại?
          </>
        }
      >
        <>
          You have six small, independent kernels in one step and put each on its own stream so they can run at the same
          time. When does that make the step shorter?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Khi mỗi kernel, chạy một mình, để phần lớn device rảnh."}>When each kernel, running alone, leaves much of the device idle.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Bất cứ khi nào chúng ở các stream khác nhau: kernel trên các stream riêng chạy song song nhau."}>Whenever they are on different streams: kernels on separate streams run side by side.</Translated>,
        why: <Translated vi={"Các stream khác nhau làm cho chồng lấp có thể xảy ra, không phải chắc chắn. Một kernel lấp đầy device thì không để lại gì cho các kernel khác."}>Different streams make overlap possible, not certain. A kernel that fills the device leaves nothing for the others.</Translated>,
      },
      {
        label: <Translated vi={"Khi các kernel dùng chung đầu vào, để mỗi kernel thấy dữ liệu của mình đã sẵn trong bộ nhớ đệm."}>When the kernels share inputs, so that each one finds its data warm in the caches.</Translated>,
        why: <Translated vi={"Dữ liệu dùng chung không phải là điều kiện. Điều kiện là các kernel độc lập, mỗi kernel dùng một phần nhỏ của device."}>Shared data is not the condition. Independent kernels that each use a small part of the device are.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Chồng lấp lấp đầy những phần của device mà một kernel để rảnh. Nếu mỗi kernel đã chiếm toàn bộ device, kernel kế tiếp
            phải chờ nó trên bất kỳ stream nào: đặt f thành 100% trong trình khám phá và không chế độ nào xong trong ít hơn k·d.
            Điều kiện của sách là mỗi kernel nhỏ khi chạy một mình dùng GPU kém hiệu quả (p. 538). AMD nêu cùng điều kiện đó:
            chạy đồng thời "is most effective when the kernels do not fully utilize the GPU's resources".
          </p>
        }
      >
        <p>
          Overlap fills parts of the device that one kernel leaves idle. If each kernel already occupies the whole device,
          the next one waits for it on any stream: set f to 100% in the explorer and no mode finishes in less than k·d. The book's condition is that each small kernel alone uses the GPU poorly (p. 538). AMD states
          the same one: concurrency "is most effective when the kernels do not fully utilize the GPU's resources".
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.3, p. 538; notes/today/ch22-facts.md B1', // i18n-ignore: citation
  },
  'ch22/22.3/q-hip-queues': {
    kind: 'numeric',
    id: 'ch22/22.3/q-hip-queues',
    prompt: (
      <Translated
        vi={
          <>
            Trên ROCm với thiết lập mặc định, một chương trình đặt 8 kernel độc lập vào 8 stream, mỗi stream một kernel, trên một
            GPU. Mỗi kernel lấp 10% device và chạy trong thời gian d, và chi phí khởi chạy không đáng kể. Sau bao lâu thì cả 8
            kernel xong, tính theo đơn vị d? Giả sử một hàng đợi phần cứng dùng chung bởi nhiều stream chạy các kernel của chúng
            lần lượt từng cái một.
          </>
        }
      >
        <>
          On ROCm with default settings, a program puts 8 independent kernels on 8 streams, one each, on one GPU. Each
          kernel fills 10% of the device and runs for a time d, and launch cost is negligible. How long until all 8 have
          finished, in units of d? Assume a hardware queue shared by several streams runs their kernels one after another.
        </>
      </Translated>
    ),
    answer: 2,
    unit: 'd',
    explain: (
      <Translated
        vi={
          <p>
            8 kernel, mỗi kernel 10%, sẽ vừa trên device cùng lúc, nên device không phải là giới hạn. Hàng đợi mới là giới hạn:{' '}
            <code>GPU_MAX_HW_QUEUES</code> mặc định là 4, HIP runtime tạo tối đa chừng đó hàng đợi phần cứng cho mỗi device trong
            mỗi tiến trình, và các stream vượt quá sẽ dùng lại các hàng đợi đó theo vòng tròn. Hai stream dùng chung mỗi hàng
            đợi, nên các kernel chạy thành hai lượt, mỗi lượt 4: 2d, so với d khi mỗi stream có một hàng đợi. Việc một hàng đợi
            dùng chung chạy các kernel của nó theo thứ tự là giả định của mô hình, và trang của AMD không nói rõ điều này. Trong
            trình khám phá ở trên, đặt 8 kernel, f bằng 10%, mỗi kernel một stream và 4 hàng đợi để thấy điều đó. Hướng dẫn của
            CUDA đưa ra một giới hạn khác, 128 lưới (grid) thường trú trên mỗi device.
          </p>
        }
      >
        <p>
          8 kernels at 10% each would fit on the device at once, so the device is not the limit. The queues are:{' '}
          <code>GPU_MAX_HW_QUEUES</code> defaults to 4, the HIP runtime creates at most that many hardware queues per device
          per process, and streams beyond them reuse the same queues round-robin. Two streams share each queue, so the
          kernels run in two rounds of 4: 2d, against d with a queue per stream. That a shared queue runs its kernels in
          order is the model&rsquo;s assumption, and AMD&rsquo;s page does not spell it out. In the explorer above, set 8
          kernels, f at 10%, a stream each and 4 queues to see it. CUDA&rsquo;s guide gives a different limit, 128 resident grids per device.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch22-facts.md B1, A2 (HIP docs at ROCm 10.2; CUDA Programming Guide 13.5); our model (sim/tasklanes.ts)',
    vi: { source: "notes/today/ch22-facts.md B1, A2 (tài liệu HIP ở ROCm 10.2; CUDA Programming Guide 13.5); mô hình của khóa học này (sim/tasklanes.ts)" },
  },
  'ch22/22.3/q-default-stream': {
    kind: 'choice',
    id: 'ch22/22.3/q-default-stream',
    prompt: (
      <Translated vi={<>Điều nào dưới đây theo mặc định ngăn các kernel độc lập chồng lấp, ngay cả khi bản thân các kernel không có vấn đề gì?</>}>
        <>Which of these keeps independent kernels from overlapping by default, even when the kernels themselves are fine?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Default stream kiểu cũ của CUDA, được dùng khi bạn khởi chạy không chỉ định stream, và compute encoder thường của Metal.">CUDA's legacy default stream, used when you launch with no stream, and Metal's plain compute encoder.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Các stream tạo với cudaStreamNonBlocking hoặc hipStreamNonBlocking, mỗi kernel một stream, trên một device."}>Streams created with cudaStreamNonBlocking or hipStreamNonBlocking, one per kernel, on one device.</Translated>,
        why: <Translated vi={"Các cờ đó tồn tại chính là để stream không đồng bộ hóa với default stream."}>Those flags exist precisely so that the stream does not synchronise with the default stream.</Translated>,
      },
      {
        label: <Translated vi={"Một compute encoder của Metal tạo với dispatch type concurrent, không encode rào chắn nào giữa các dispatch."}>A Metal compute encoder made with dispatch type concurrent, with no barrier encoded between the dispatches.</Translated>,
        why: <Translated vi="Đó là cách Metal ghi trong tài liệu để cho các dispatch chồng lấp; bạn chỉ thêm rào chắn ở chỗ một dispatch đọc đầu ra của dispatch khác.">That is Metal's documented way to let dispatches overlap; you add barriers only where one reads another's output.</Translated>,
      },
    ],
    explain: (
      <>
        <Translated
          vi={
            <p>
              CUDA: một kernel trên default stream kiểu cũ chỉ bắt đầu sau khi mọi công việc trước đó trên device đã hoàn tất, và
              không gì trên bất kỳ stream nào bắt đầu cho đến khi nó xong; các stream non-blocking được miễn. HIP: AMD gọi default
              stream là "a blocking operation" và cung cấp <code>hipStreamNonBlocking</code>; việc nó chặn có giống hệt stream kiểu
              cũ của CUDA hay không là unverified. Metal: encoder mà lời gọi thường trả về dispatch các lệnh của nó một cách tuần
              tự.
            </p>
          }
        >
          <p>
            CUDA: a kernel on the legacy default stream begins only after all earlier work on the device has completed,
            and nothing in any stream starts until it has finished; non-blocking streams are exempt. HIP: AMD calls the
            default stream "a blocking operation" and offers <code>hipStreamNonBlocking</code>; whether it blocks exactly as
            CUDA's legacy stream does is unverified. Metal: the encoder the plain call returns dispatches its commands
            serially.
          </p>
        </Translated>
      </>
    ),
    source: 'notes/today/ch22-facts.md A2, B1; ch02-facts.md C9', // i18n-ignore: citation
  },
  'ch22/22.3/q-metal-overlap': {
    kind: 'choice',
    id: 'ch22/22.3/q-metal-overlap',
    prompt: (
      <Translated
        vi={
          <>
            Trên Metal, dispatch A và B độc lập và nên chồng lấp; dispatch C đọc đầu ra của B. Apple ghi trong tài liệu cách nào
            cho trường hợp này?
          </>
        }
      >
        <>
          On Metal, dispatches A and B are independent and should overlap; dispatch C reads B's output. What does Apple
          document for this?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Một compute encoder với dispatch type concurrent, có memoryBarrier(scope:) được encode trước C."}>One compute encoder of dispatch type concurrent, with memoryBarrier(scope:) encoded before C.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"A và B trên hai command queue khác nhau, được GPU chạy cùng lúc, với C được commit sau cả hai."}>A and B on two different command queues, which the GPU runs at the same time, with C committed after both.</Translated>,
        why: <Translated vi={"Apple ghi lại thứ tự trong một queue. Apple không ghi rằng hai queue chạy cùng lúc."}>Apple documents ordering within a queue. It does not document that two queues run at the same time.</Translated>,
      },
      {
        label: <Translated vi={"Một compute encoder thường, cho chồng lấp mọi dispatch không xung đột."}>A plain compute encoder, which overlaps any dispatches that do not conflict.</Translated>,
        why: <Translated vi="Theo tài liệu của Apple, encoder thường là tuần tự: nó chạy A, B và C lần lượt từng cái.">The plain encoder is serial by Apple's documentation: it runs A, B and C one after another.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Rào chắn buộc mọi dispatch được encode trước nó phải xong, và các lần ghi của chúng phải hiển thị, trước khi bất cứ thứ
            gì sau nó bắt đầu. Nó chờ cả A lẫn B. Đó là lý do thứ tự encode quan trọng: hãy thử chế độ theo cặp với encoder của
            Metal trong trình khám phá, theo thứ tự phát và bên tạo dữ liệu trước. Với các kiểu của Metal 4, mọi tài nguyên đều
            không được theo dõi, nên đặt rào chắn luôn là việc của bạn.
          </p>
        }
      >
        <p>
          The barrier makes every dispatch encoded before it finish, and its writes visible, before anything after it
          starts. It waits for A as well as B. That is why the order you encode in matters: try pairs with the Metal encoder
          in the explorer, as issued and producers first. With the Metal 4 types every resource is untracked, so placing the
          barriers is always your job.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch22-facts.md C1; infra-facts.md C12; ch02-facts.md C9', // i18n-ignore: citation
  },
  'ch22/22.3/q-graphs': {
    kind: 'choice',
    id: 'ch22/22.3/q-graphs',
    prompt: (
      <Translated
        vi={
          <>
            PyTorch ghi trong tài liệu rằng chế độ "reduce-overhead" của <code>torch.compile</code>, dùng CUDA graph, là "useful
            for small batches". Vì sao lại là batch nhỏ?
          </>
        }
      >
        <>
          PyTorch documents <code>torch.compile</code>'s "reduce-overhead" mode, which uses CUDA graphs, as "useful for small
          batches". Why small batches?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Mỗi kernel ngắn, nên chi phí CPU cố định của một lần khởi chạy chiếm phần lớn hơn trong bước.">Each kernel is short, so a launch's fixed CPU cost is a larger share of the step.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Graph cho mỗi kernel giữ ít bộ nhớ thường trú hơn giữa các lần khởi chạy, điều quan trọng nhất ở batch nhỏ."}>A graph lets each kernel keep less memory resident between launches, which matters most at small batch.</Translated>,
        why: <Translated vi={"Graph thay đổi cách công việc được gửi đi, không thay đổi thứ kernel lưu trữ. PyTorch nêu chi phí phụ trên CPU là lợi ích chính."}>A graph changes how work is submitted, not what a kernel stores. PyTorch names CPU overhead as the main gain.</Translated>,
      },
      {
        label: <Translated vi={"Batch nhỏ không thể lấp đầy device qua stream, nên graph là cách để chúng chồng lấp các kernel."}>Small batches cannot fill the device through streams, so a graph is their way to overlap kernels.</Translated>,
        why: <Translated vi={"Stream có thể chồng lấp các kernel nhỏ ở mọi kích thước batch, và chồng lấp không phải điều PyTorch nêu: nó nêu chi phí phụ, thứ mà graph trả một lần cho cả chuỗi khởi chạy."}>Streams can overlap small kernels at any batch size, and overlap is not what PyTorch cites: it names overhead, which a graph pays once for the whole sequence of launches.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đó là định luật Amdahl với lần khởi chạy là phần còn lại. NVIDIA: với một kernel có thời gian thực thi ngắn, chi phí
            khởi chạy "can be a significant fraction" của toàn bộ; graph trả nó một lần cho cả chuỗi. AMD gọi chi phí của một lần
            khởi chạy HIP trực tiếp là không đáng kể còn các lớp mà framework thêm vào là đáng kể, và graph của PyTorch bỏ qua
            đúng những lớp đó. Trong trình khám phá, đặt L lớn hơn d và so sánh một stream với graph.
          </p>
        }
      >
        <p>
          It is Amdahl's law with the launch as the remainder. NVIDIA: for a kernel with a short execution time, the launch
          overhead "can be a significant fraction" of the whole; a graph pays it once for the sequence. AMD calls the
          overhead of a direct HIP launch negligible and the layers a framework adds significant, and PyTorch's graphs skip
          exactly those layers. In the explorer, set L above d and compare one stream with the graph.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch22-facts.md D7, A3, B2', // i18n-ignore: citation
  },
  'ch22/22.3/q-decode-kernels': {
    kind: 'numeric',
    id: 'ch22/22.3/q-decode-kernels',
    prompt: (
      <Translated
        vi={
          <>
            Với mỗi token, các kernel chiếm 9 phần thời gian và phần việc host (lấy mẫu, khởi chạy) chiếm 1 phần. Các kernel
            nhanh lên 3 lần và không có gì chồng lấp. Số token mỗi giây tăng lên bao nhiêu lần?
          </>
        }
      >
        <>
          Per token, kernels take 9 parts of the time and host work (sampling, launches) takes 1 part. The kernels become 3
          times faster and nothing overlaps. By what factor do tokens per second rise?
        </>
      </Translated>
    ),
    answer: 2.5,
    tol: 0.01,
    unit: '×',
    explain: (
      <Translated
        vi={
          <p>
            Trước: 9 + 1 = 10 phần. Sau: 9/3 + 1 = 4 phần. 10 / 4 = {fmt(speedup(0.9, 3))}. Phần host trước là một phần mười của
            mỗi token và giờ là một phần tư, nên thêm một lần 3× nữa cho các kernel chỉ mua thêm được 4 / 2 = 2×, và không tốc độ
            kernel nào vượt được {times(ceiling(0.9))} tính chung.
          </p>
        }
      >
        <p>
          Before: 9 + 1 = 10 parts. After: 9/3 + 1 = 4 parts. 10 / 4 = {fmt(speedup(0.9, 3))}. The host part was a tenth of
          each token and is now a quarter of it, so a second 3× on the kernels buys only 4 / 2 = 2× more, and no kernel speed
          gets past {times(ceiling(0.9))} overall.
        </p>
      </Translated>
    ),
    source: 'the method of ch. 22 §22.3, pp. 537–538; the host sampler is ch01-facts.md D2',
    vi: { source: "phương pháp của ch. 22 §22.3, pp. 537–538; bộ lấy mẫu trên host là ch01-facts.md D2" },
  },
  'ch22/22.3/q-host-share': {
    kind: 'numeric',
    id: 'ch22/22.3/q-host-share',
    prompt: (
      <Translated
        vi={
          <>
            Host và device chồng lấp, và phần của device được giấu hoàn toàn sau phần của host. Phần host có thể chiếm tối đa bao
            nhiêu trong thời gian chạy ban đầu nếu ứng dụng phải đạt ít nhất 20×?
          </>
        }
      >
        <>
          Host and device overlap, and the device's part is completely hidden behind the host's. What is the largest share
          of the original run time the host part can have if the application must reach at least 20×?
        </>
      </Translated>
    ),
    answer: 5,
    tol: 0.001,
    unit: '%',
    explain: (
      <Translated
        vi={
          <p>
            Khi device được giấu đi, thời gian chạy chỉ còn phần host, nên mức tăng tốc là 1 / (tỉ lệ host). 1 / h ≥ 20 nghĩa là
            h ≤ 5%. Ví dụ của sách nằm đúng trên đường đó, đó là lý do nó đạt {times(overlapped(BOOK.p, BOOK.s))} và không hơn.
            Làm kernel nhanh hơn không thể dịch chuyển nó.
          </p>
        }
      >
        <p>
          With the device hidden, the run time is the host part alone, so the speed-up is 1 / (host share). 1 / h ≥ 20 means
          h ≤ 5%. The book's example sits exactly on that line, which is why it reaches {times(overlapped(BOOK.p, BOOK.s))} and
          no more. Making the kernel faster cannot move it.
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.3, p. 538',
  },
};

// ---------------------------------------------------------------- the section
export function S223() {
  const pPlain = speedup(P_OVERLAP.p, P_OVERLAP.s);
  const pOver = overlapped(P_OVERLAP.p, P_OVERLAP.s);
  return (
    <>
      <Translated
        vi={
          <p>
            Mọi chương cho đến giờ đều làm một kernel nhanh hơn. Một chương trình hiếm khi chỉ là một kernel. Nó là nhiều mô-đun
            chuyển dữ liệu cho nhau, và chúng khác nhau rất xa về lượng việc phải làm. Mục này áp định luật Amdahl của chương 1
            vào bức tranh đó: bạn quyết định từng mô-đun một xem cái gì lên GPU, và những mô-đun bạn để lại sẽ đặt giới hạn cho
            những gì người dùng thấy.
          </p>
        }
      >
        <p>
          Every chapter so far has made one kernel faster. A program is rarely one kernel. It is several modules that hand
          data to each other, and they differ enormously in how much work they do. This section applies chapter 1's
          Amdahl's law to that picture: you decide module by module what goes on the GPU, and the modules you leave behind
          set the limit on what users see.
        </p>
      </Translated>
      <Source section="22.3" pages="537" />

      <h2><Translated vi={"Một ứng dụng gồm nhiều mô-đun"}>An application is several modules</Translated></h2>
      <Translated
        vi={
          <p>
            Ví dụ của sách là một mô phỏng động lực học phân tử, tiến từng bước thời gian một. Mỗi bước dựng một danh sách lân
            cận, dùng để tính lực không liên kết; tính nhiều loại lực trên mọi nguyên tử; rồi cập nhật vị trí và vận tốc. Mỗi
            loại lực có phương pháp riêng: dao động, quay và không liên kết. Lực không liên kết liên quan đến tương tác giữa nhiều
            nguyên tử và tốn hơn hẳn hai loại kia, nên các mô-đun thường được viết thành những lượt riêng trên dữ liệu lực, và mỗi
            lượt là một quyết định riêng.
          </p>
        }
      >
        <p>
          The book's example is a molecular-dynamics simulation, advanced one time step at a time. Each step builds a
          neighbour list, which feeds the non-bonded forces; computes several kinds of force on every atom; and then updates
          positions and velocities. Each kind of force has its own method: vibrational, rotational and non-bonded. The
          non-bonded forces involve interactions among many atoms and cost far more than the other two, so the modules tend
          to be written as separate passes over the force data, and each pass is a separate decision.
        </p>
      </Translated>
      <Source section="22.3" pages="537" figs={tr('Fig. 22.2', 'Hình 22.2')} />
      <Translated
        vi={
          <p>
            Kết quả mà sách cho là hợp lý: một kernel cho lực không liên kết, trong khi lực dao động và lực quay ở lại trên host.
            Bước cập nhật cũng có thể ở lại trên host, kết hợp lực của host với lực của device và tính vị trí và vận tốc mới.
          </p>
        }
      >
        <p>
          The outcome the book calls plausible: a kernel for the non-bonded forces, while the vibrational and rotational
          forces stay on the host. The update may stay on the host too, combining the host's forces with the device's and
          computing the new positions and velocities.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Trang sách nói kernel tính lực non-bonding "for all the grid points", và host tính lực của nó "for the grid points".
              Ví dụ này nói về nguyên tử; hãy đọc là "atoms". Cùng đoạn đó dùng xen kẽ "non-bonded" và "non-bonding" cho cùng một
              loại lực.
            </p>
          }
        >
          <p>
            The page says the kernel computes the non-bonding forces "for all the grid points", and the host its forces "for
            the grid points". The example is about atoms; read "atoms". The same paragraph alternates "non-bonded" and
            "non-bonding" for one kind of force.
          </p>
        </Translated>
      </Aside>

      <h2><Translated vi={"Các con số của sách"}>The book's numbers</Translated></h2>
      <Translated
        vi={
          <p>
            Giả sử lực không liên kết chiếm 95% thời gian chạy tuần tự ban đầu, và kernel làm chúng nhanh hơn 100 lần. 5% còn lại
            ở trên host với tốc độ cũ. Thời gian chạy mới là 5% + 95%/100 = 5.95% thời gian cũ, và mức tăng tốc là 1 / 5.95% ={' '}
            {times(speedup(BOOK.p, BOOK.s))}. Trang sách in 17×.
          </p>
        }
      >
        <p>
          Say the non-bonded forces take 95% of the original sequential run time, and the kernel makes them 100 times
          faster. The other 5% stays on the host at its old speed. The new run time is 5% + 95%/100 = 5.95% of the old one,
          and the speed-up is 1 / 5.95% = {times(speedup(BOOK.p, BOOK.s))}. The page prints 17×.
        </p>
      </Translated>
      <Source section="22.3" pages="537–538" />
      <Translated
        vi={
          <p>
            Sau đó sách cho host và device làm việc cùng lúc. Trong khi kernel tính lực không liên kết, host tính các lực còn
            lại. 0.95% của kernel nằm gọn trong 5% của host và không còn được tính, và mức tăng tốc trở thành 1 / 5% ={' '}
            {times(overlapped(BOOK.p, BOOK.s))}. Đó cũng là trần. Chừng nào 5% công việc còn chạy trên host với tốc độ cũ, không
            kernel nào đưa ứng dụng vượt 20×, dù nhanh đến đâu. Kernel được làm nhanh hơn 100 lần rồi được giấu hoàn toàn, và ứng
            dụng chỉ giữ được một phần năm số đó.
          </p>
        }
      >
        <p>
          Then the book lets host and device work at the same time. While the kernel computes non-bonded forces, the host
          computes the others. The kernel's 0.95% fits inside the host's 5% and stops counting, and the speed-up becomes
          1 / 5% = {times(overlapped(BOOK.p, BOOK.s))}. That is also the ceiling. While 5% of the work runs on the host at
          its old speed, no kernel gets the application past 20×, however fast it is. The kernel was made 100 times faster
          and then hidden completely, and the application kept a fifth of that.
        </p>
      </Translated>
      <Predict
        id="ch22/22.3/p-overlap"
        prompt={
          <Translated
            vi={
              <>
                Một chương trình khác: 90% thời gian của nó được làm nhanh hơn 50 lần trên GPU và 10% còn lại ở trên host. Host và
                device không chồng lấp. Mức tăng tốc của ứng dụng là bao nhiêu?
              </>
            }
          >
            <>
              A different program: 90% of its time is made 50 times faster on the GPU and the other 10% stays on the host.
              Host and device do not overlap. What is the application's speed-up?
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
                  1 / (10% + 90%/50) = 1 / (10% + 1.8%) = <b>{times(pPlain)}</b>.{' '}
                  {Math.abs(v - pPlain) < 0.1
                    ? 'Đúng như bạn đã trả lời.'
                    : Math.abs(v - pOver) < 0.2
                      ? '10 là con số khi chồng lấp. Không chồng lấp thì 1.8% của GPU cộng thêm vào 10% của host.'
                      : v >= 40
                        ? '50 là mức tăng tốc của một phần, không phải của chương trình: 10% bị để lại vẫn được tính nguyên vẹn.'
                        : 'Hãy cộng hai phần còn lại trước, rồi lấy nghịch đảo.'}
                </p>
                <p>
                  Cho chúng chồng lấp thì 1.8% của GPU nằm gọn trong 10% của host: {times(pOver)}, cũng chính là trần
                  1 / 10% = {times(ceiling(P_OVERLAP.p))}. Một kernel nhanh hơn sẽ không thay đổi con số nào trong hai con số này.
                </p>
              </>
            }
          >
            <>
              <p>
                1 / (10% + 90%/50) = 1 / (10% + 1.8%) = <b>{times(pPlain)}</b>.{' '}
                {Math.abs(v - pPlain) < 0.1
                  ? 'That is what you had.'
                  : Math.abs(v - pOver) < 0.2
                    ? "10 is the overlapped figure. Without overlap the GPU's 1.8% adds to the host's 10%."
                    : v >= 40
                      ? "50 is the speed-up of the part, not of the program: the 10% left alone counts at full size."
                      : 'Add the two remaining parts first, then take the reciprocal.'}
              </p>
              <p>
                Let them overlap and the GPU's 1.8% hides inside the host's 10%: {times(pOver)}, which is also the ceiling
                1 / 10% = {times(ceiling(P_OVERLAP.p))}. A faster kernel would change neither number.
              </p>
            </>
          </Translated>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Trong trình khám phá, mỗi mô-đun của bước thời gian là một ô cho biết tỉ lệ của nó trong thời gian ban đầu, nơi nó
            chạy, nó nhanh hơn bao nhiêu trên GPU, và nó có bị buộc với device hay không. Một mô-đun bị buộc thì cấp dữ liệu cho
            device hoặc chờ kết quả của nó trong cùng bước, như các mũi tên của Hình 22.2 cho thấy: kernel lực không liên kết cần
            danh sách lân cận, và bước cập nhật cần cả hai loại lực. Một mô-đun bị buộc không thể chồng lấp với device. Sách chỉ
            đưa cách chia 95/5; cách chia 5% cho ba mô-đun còn lại, và mức 4× chúng sẽ đạt trên GPU, là của khóa học này.
          </p>
        }
      >
        <p>
          In the explorer, each module of the time step is a box with its share of the original time, where it runs, how
          much faster it is on the GPU, and whether it is chained to the device. A chained module feeds the device or waits
          for its result within the step, as the arrows of Fig. 22.2 say: the non-bonded kernel needs the neighbour list,
          and the update needs both kinds of force. A chained module cannot overlap with the device. The book gives only the
          95/5 split; how the 5% divides between the other three modules, and the 4× they would get on a GPU, are ours.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Như trong mô hình của chương 1: các tỉ lệ cố định, mô-đun trên host giữ tốc độ cũ, và chuyển một mô-đun không tốn sao
              chép hay khởi chạy nào. Khi bật chồng lấp, thời gian chạy là thời gian host bị buộc cộng với phần dài hơn giữa thời
              gian host còn lại và thời gian GPU. Sách giả định GPU được giấu hoàn toàn; ở đây đúng là vậy, vì 0.95% của kernel nằm
              gọn trong 2% của lực dao động và lực quay.
            </p>
          }
        >
          <p>
            As in chapter 1's model: shares are fixed, host modules keep their old speed, and moving a module costs no copies
            and no launches. With overlap on, the run time is the chained host time plus the longer of the other host time
            and the GPU time. The book assumes the GPU hides completely; here it does, because the kernel's 0.95% fits
            inside the 2% of vibrational and rotational forces.
          </p>
        </Translated>
      </Simplification>
      <ModulesBench />
      <Translated vi={<p>Ba điều cần tìm, mỗi điều cách thiết lập của sách một hoặc hai thay đổi:</p>}>
        <p>Three things to find, each one or two changes away from the book's setting:</p>
      </Translated>
      <ul>
        <Translated
          vi={
            <li>Đẩy mức tăng tốc của lực không liên kết từ 100× lên 1,000×. Ứng dụng được lợi bao nhiêu, và vì sao ít như vậy?</li>
          }
        >
          <li>Push the non-bonded speed-up from 100× to 1,000×. How much does the application gain, and why so little?</li>
        </Translated>
        <Translated
          vi={
            <li>Bật chồng lấp, rồi chuyển lực dao động và lực quay lên GPU. Vì sao chồng lấp không còn đem lại gì thêm?</li>
          }
        >
          <li>Turn overlap on, then move the vibrational and rotational forces to the GPU. Why does overlap stop adding anything?</li>
        </Translated>
        <Translated
          vi={
            <li>Tìm một cách bố trí có trần trên 20×. Nó đã phải chuyển những gì?</li>
          }
        >
          <li>Find a placement whose ceiling is above 20×. What did it have to move?</li>
        </Translated>
      </ul>
      <Translated
        vi={
          <p>
            Quy luật đằng sau cả ba: khi mô-đun lớn đã nhanh, các mô-đun nhỏ sẽ quyết định. Chồng lấp cùng lắm chỉ giấu được thời
            gian của device sau thời gian của host; nó không bao giờ rút ngắn thời gian của chính host. Chỉ có chuyển phần việc
            host sang device, hoặc làm nó nhanh hơn ngay tại chỗ, mới nâng được trần.
          </p>
        }
      >
        <p>
          The pattern behind all three: once the big module is fast, the small ones decide. Overlap can at best hide the
          device's time behind the host's; it never shortens the host's own. Only moving host work to the device, or making
          it faster where it is, raises the ceiling.
        </p>
      </Translated>

      <h2><Translated vi={"Nhiều hoạt động nhỏ"}>Many small activities</Translated></h2>
      <Translated
        vi={
          <p>
            Sách rút ra bài học rộng hơn từ 5% đó. Trong một ứng dụng thực tế, phần còn lại đó hiếm khi là một mô-đun. Nó là
            nhiều hoạt động nhỏ, không cái nào đáng một kernel riêng, và thời gian cộng dồn của chúng trở thành giới hạn cho mức
            tăng tốc mà người dùng thấy.
          </p>
        }
      >
        <p>
          The book draws the wider lesson from the 5%. In a real application that remainder is rarely one module. It is many
          small activities, none worth a kernel of its own, and their accumulated time becomes the limit on the speed-up
          users see.
        </p>
      </Translated>
      <Source section="22.3" pages="537" />
      <Translated
        vi={
          <p>
            Câu trả lời của sách là song song ở mức tác vụ: khi tập dữ liệu đủ lớn, hãy chạy các hoạt động nhỏ song song với
            nhau. Một host nhiều lõi (core) có thể chạy chúng trên các lõi khác nhau. Hoặc GPU có thể chạy nhiều kernel nhỏ cùng
            lúc, mỗi hoạt động một kernel: mỗi kernel một mình có thể dùng device kém hiệu quả và chẳng hơn CPU bao nhiêu, nhưng
            gộp lại chúng có thể lấp đầy nó. CUDA biểu diễn điều này bằng stream, mà{' '}
            chương 23 trình bày ở <a href={href('ch23', '23.3')}>§23.3</a> (pp. 553–563).
          </p>
        }
      >
        <p>
          Its answer is task-level parallelism: when the data set is large enough, run the small activities in parallel with
          each other. A multi-core host can run them on different cores. Or the GPU can run several small kernels at once,
          one per activity: each alone may use the device poorly and gain little over the CPU, but together they can fill it.
          CUDA expresses this with streams, which{' '}
          chapter 23 covers in <a href={href('ch23', '23.3')}>§23.3</a> (pp. 553–563).
        </p>
      </Translated>
      <Source section="22.3" pages="538" />
      <Translated
        vi={
          <p>
            Việc này có lợi hay không phụ thuộc vào hai điều sách để ngầm: mỗi kernel để rảnh bao nhiêu phần device, và host phát
            lệnh các kernel nhanh đến đâu. Trình khám phá có cả hai. Có k kernel độc lập, mỗi kernel lấp một phần f của device
            trong thời gian d, và host tốn L cho mỗi lần khởi chạy. Thời gian tính theo đơn vị d. "Theo cặp" làm cho cứ mỗi kernel
            thứ hai đọc đầu ra của kernel ngay trước nó.
          </p>
        }
      >
        <p>
          Whether it pays depends on two things the book leaves implicit: how much of the device each kernel leaves idle, and
          how fast the host can issue the kernels. The explorer has both. There are k independent kernels, each filling a
          fraction f of the device for a time d, and the host spends L on each launch. Times are in units of d. "Pairs" makes
          every second kernel read the output of the one before it.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Các kernel chạy song song nhau chừng nào tổng phần lấp đầy của chúng không quá 100%, nếu không chúng phải chờ; câu
              "may run concurrently" của CUDA được mô hình hóa thành "có chạy đồng thời". Một lần khởi chạy, và việc encode một
              dispatch Metal, tốn cùng một L cố định. Graph được capture từ một stream: một L, rồi các kernel của nó nối tiếp nhau.
              Một hàng đợi phần cứng dùng chung bởi nhiều stream chạy các kernel theo thứ tự khởi chạy (giả định của khóa học này).
              Command buffer của Metal bắt đầu khi được commit, sau dispatch cuối cùng. Không có tranh chấp bộ nhớ, không sao chép,
              rào chắn không tốn gì. Thời gian là tương đối, không phải thời gian trên thiết bị.
            </p>
          }
        >
          <p>
            Kernels run side by side while their fills add up to at most 100%, otherwise they wait; CUDA's "may run
            concurrently" is modelled as "does". A launch, and the encoding of a Metal dispatch, cost the same fixed L. The
            graph is captured from one stream: one L, then its kernels back to back. A hardware queue shared by several
            streams runs its kernels in launch order (our assumption). The Metal command buffer starts at its commit, after
            the last dispatch. No memory contention, no copies, no cost for a barrier. Times are relative, not device times.
          </p>
        </Translated>
      </Simplification>
      <ManyKernelsBench />
      <Translated vi={<p>Hãy thử những điều sau trước khi đọc tiếp:</p>}>
        <p>Try these before reading on:</p>
      </Translated>
      <ul>
        <Translated
          vi={
            <li>Đặt f thành 100%. Chế độ nào vẫn còn hơn một stream chút nào?</li>
          }
        >
          <li>Set f to 100%. Which modes still gain anything over one stream?</li>
        </Translated>
        <Translated
          vi={
            <li>Đặt f thành 10%, chọn mỗi kernel một stream, rồi chuyển sang 4 hàng đợi. Chồng lấp dừng lại ở đâu?</li>
          }
        >
          <li>Set f to 10%, pick a stream each, then switch to 4 queues. Where does the overlap stop?</li>
        </Translated>
        <Translated
          vi={
            <li>Tăng L lên trên d. Chế độ nào thắng, và vì sao thêm stream không giúp gì?</li>
          }
        >
          <li>Raise L above d. Which mode wins, and why do more streams not help?</li>
        </Translated>
        <Translated
          vi={
            <li>Chọn theo cặp và encoder của Metal. So sánh encode theo thứ tự phát với bên tạo dữ liệu trước.</li>
          }
        >
          <li>Choose pairs and the Metal encoder. Compare encoding as issued with producers first.</li>
        </Translated>
      </ul>
      <Ask q={questionsS223['ch22/22.3/q-when-concurrent']} />
      <Ask q={questionsS223['ch22/22.3/q-hip-queues']} />

      <h2><Translated vi={"Nhiều kernel cùng lúc, trên từng backend"}>Several kernels at once, on each backend</Translated></h2>
      <Translated
        vi={
          <p>
            Sách chỉ nói CUDA hỗ trợ điều này bằng stream. Mỗi backend hiện nay có cách riêng để cho các kernel độc lập chồng
            lấp, giới hạn riêng, và mặc định riêng ngăn điều đó. Không backend nào hứa hẹn chồng lấp: hướng dẫn của CUDA nói công
            việc trên các stream khác nhau có thể chạy đồng thời khi giữa chúng không có phụ thuộc và có đủ tài nguyên rảnh.
          </p>
        }
      >
        <p>
          The book says only that CUDA supports this with streams. Each backend today has its own way to let independent
          kernels overlap, its own limit, and its own default that prevents it. None of them promises overlap: CUDA's guide
          says work on different streams may run concurrently when there are no dependencies between them and enough free
          resources.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch22-facts A2, B1, C1; infra-facts C12; ch02-facts C9)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              Nguồn: CUDA Programming Guide và Best Practices Guide 13.5, tài liệu HIP ở ROCm 10.2, và tài liệu Metal của Apple
              cùng các header của macOS 27.0 SDK.
            </p>
          }
        >
          <p>
            Sources: the CUDA Programming Guide and Best Practices Guide 13.5, the HIP documentation at ROCm 10.2, and
            Apple's Metal documentation with the macOS 27.0 SDK headers.
          </p>
        </Translated>
      </Today>
      <table className="data wide" data-backends>
        <thead>
          <tr>
            <th />
            <th><Translated vi="Để cho các kernel độc lập chồng lấp">To let independent kernels overlap</Translated></th>
            <th><Translated vi="Giới hạn và điều kiện">Limit and condition</Translated></th>
            <th><Translated vi="Điều ngăn nó theo mặc định">What prevents it by default</Translated></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><b>CUDA</b></td>
            <Translated
              vi={
                <td>
                  Khởi chạy chúng trên các stream không mặc định. <code>concurrentKernels</code> trong <code>cudaDeviceProp</code> cho
                  biết device có làm được không; <code>python3 labs/pmpp.py doctor</code> in giá trị này ra.
                </td>
              }
            >
              <td>
                Launch them on non-default streams. <code>concurrentKernels</code> in <code>cudaDeviceProp</code> says whether
                the device can; <code>python3 labs/pmpp.py doctor</code> prints it.
              </td>
            </Translated>
            <Translated
              vi={
                <td>
                  Tối đa 128 lưới thường trú trên mỗi device, với mọi compute capability mà hướng dẫn liệt kê (7.5 đến 12.x). Chồng lấp
                  cần không có phụ thuộc event, không có đồng bộ hóa ngầm và đủ tài nguyên.
                </td>
              }
            >
              <td>
                Up to 128 resident grids per device, for every compute capability the guide lists (7.5 to 12.x). Overlap
                needs no event dependency, no implicit synchronisation and enough resources.
              </td>
            </Translated>
            <Translated
              vi={
                <td>
                  Default stream kiểu cũ. Một kernel khởi chạy ở đó chờ mọi công việc trước đó trên device, và không gì trên bất kỳ
                  stream nào bắt đầu cho đến khi nó xong. Các stream tạo với <code>cudaStreamNonBlocking</code> được miễn.
                </td>
              }
            >
              <td>
                The legacy default stream. A kernel launched there waits for all earlier work on the device, and nothing
                starts in any stream until it finishes. Streams made with <code>cudaStreamNonBlocking</code> are exempt.
              </td>
            </Translated>
          </tr>
          <tr>
            <td><b>HIP</b></td>
            <td><Translated vi={<>Giống vậy: stream, và cùng trường <code>concurrentKernels</code>.</>}>The same: streams, and the same <code>concurrentKernels</code> field.</Translated></td>
            <Translated
              vi={
                <td>
                  Tối đa <code>GPU_MAX_HW_QUEUES</code> hàng đợi phần cứng cho mỗi device trong mỗi tiến trình, mặc định 4; các stream
                  thêm vào dùng chung chúng theo vòng tròn. AMD: chạy đồng thời "is most effective when the kernels do not fully utilize
                  the GPU's resources".
                </td>
              }
            >
              <td>
                At most <code>GPU_MAX_HW_QUEUES</code> hardware queues per device per process, 4 by default; further streams
                share them round-robin. AMD: concurrency "is most effective when the kernels do not fully utilize the GPU's
                resources".
              </td>
            </Translated>
            <Translated
              vi={
                <td>
                  Default stream, mà AMD gọi là "a blocking operation"; <code>hipStreamNonBlocking</code> tạo một stream không chặn. Việc
                  nó chặn có giống hệt stream kiểu cũ của CUDA hay không là unverified.
                </td>
              }
            >
              <td>
                The default stream, which AMD calls "a blocking operation"; <code>hipStreamNonBlocking</code> makes one that is
                not. Whether it blocks exactly as CUDA's legacy stream does is unverified.
              </td>
            </Translated>
          </tr>
          <tr>
            <td><b>Metal</b></td>
            <Translated
              vi={
                <td>
                  Một compute encoder với dispatch type concurrent, có <code>memoryBarrier(scope:)</code> trước mọi dispatch đọc đầu ra
                  của dispatch khác. Một rào chắn chờ mọi dispatch được encode trước nó.
                </td>
              }
            >
              <td>
                One compute encoder of dispatch type concurrent, with <code>memoryBarrier(scope:)</code> before any dispatch
                that reads another's output. A barrier waits for every dispatch encoded before it.
              </td>
            </Translated>
            <Translated
              vi={
                <td>
                  Apple không ghi nhận chồng lấp giữa các command queue. Với các kiểu của Metal 4 (<code>MTL4</code>), mọi tài nguyên
                  đều không được theo dõi, nên bạn tự đặt mọi rào chắn.
                </td>
              }
            >
              <td>
                Apple documents no overlap between command queues. With the Metal 4 (<code>MTL4</code>) types every resource
                is untracked, so you place every barrier yourself.
              </td>
            </Translated>
            <td><Translated vi="Encoder mà lời gọi thường trả về: các dispatch của nó chạy lần lượt từng cái.">The encoder the plain call returns: its dispatches run one after another.</Translated></td>
          </tr>
        </tbody>
      </table>
      <Ask q={questionsS223['ch22/22.3/q-default-stream']} />
      <Ask q={questionsS223['ch22/22.3/q-metal-overlap']} />

      <h2><Translated vi={"Lần khởi chạy cũng là một hoạt động nhỏ"}>The launch is a small activity too</Translated></h2>
      <Translated
        vi={
          <p>
            Có một hoạt động nhỏ mà sách không gọi tên, đi kèm mọi kernel: lần khởi chạy. Thiết lập và phát lệnh một kernel tốn
            thời gian của host. Sau một kernel dài nó biến mất; trước một kernel ngắn thì không. Trong trình khám phá, khi L lớn
            hơn d thì device dành phần lớn thời gian chờ host, dù bạn cho nó bao nhiêu stream.
          </p>
        }
      >
        <p>
          One small activity the book does not name comes with every kernel: the launch. Setting up and issuing a kernel
          costs host time. Behind a long kernel it disappears; in front of a short one it does not. In the explorer, once L
          is larger than d the device spends most of its time waiting for the host, however many streams you give it.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>Nguồn đọc ngày 2026-11-20: notes/today/ch22-facts.md A3, B2, C2 và D7.</p>
          }
        >
          <p>Sources read 2026-11-20: notes/today/ch22-facts.md A3, B2, C2 and D7.</p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Hướng dẫn của NVIDIA nói thẳng: với một kernel có thời gian thực thi ngắn, chi phí khởi chạy "can be a significant
            fraction of the overall end-to-end execution time". Cách chữa của nó là CUDA graph. Bạn ghi lại một chuỗi khởi chạy
            một lần, bằng cách capture một stream hoặc dựng graph bằng API của nó, khởi tạo nó một lần, rồi khởi chạy cả chuỗi như
            một đơn vị bao nhiêu lần tùy ý, trả chi phí thiết lập theo mỗi graph thay vì mỗi kernel. HIP graph có cùng hình dạng
            với tiền tố <code>hip</code>. AMD nhìn chi phí này theo cách khác: không đáng kể với một kernel khởi chạy trực tiếp qua
            HIP, đáng kể khi một framework thêm các lớp chuyển hướng trên đường đi. Metal có indirect command buffer, mà Apple mô
            tả là cách để encode lệnh một lần, dùng lại chúng, và giảm chi phí phụ trên CPU. Gọi chúng là bản tương ứng của graph
            trong Metal là phép so sánh của khóa học này: một indirect command buffer là một danh sách lệnh không có cạnh phụ
            thuộc, và Apple không đưa ra con số nào về lượng tiết kiệm được.
          </p>
        }
      >
        <p>
          NVIDIA's guide says it plainly: for a kernel with a short execution time, the launch overhead "can be a significant
          fraction of the overall end-to-end execution time". Its remedy is a CUDA graph. You record a sequence of launches
          once, by capturing a stream or by building the graph with its API, instantiate it once, and then launch the whole
          sequence as one unit as often as you need, paying the setup per graph rather than per kernel. HIP graphs have the
          same shape with the <code>hip</code> prefix. AMD frames the cost differently: negligible for a kernel launched
          directly through HIP, significant when a framework adds layers of redirection on the way. Metal has indirect
          command buffers, which Apple describes as a way to encode commands once, reuse them, and reduce CPU overhead.
          Calling them Metal's counterpart of graphs is our analogy: an indirect command buffer is a list of commands with no
          dependency edges, and Apple gives no figure for what it saves.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Mã thực tế dựa vào điều này. Các bản build llama.cpp dùng CUDA graph theo mặc định (thư viện ggml độc lập thì không)
            và cả HIP graph. <code>torch.compile</code> của PyTorch có một chế độ, "reduce-overhead", làm việc này bằng CUDA
            graph. Đặt L lớn hơn d trong trình khám phá và so sánh một stream với graph.
          </p>
        }
      >
        <p>
          Real code relies on this. llama.cpp builds use CUDA graphs by default (the standalone ggml library does not) and
          HIP graphs as well. PyTorch's <code>torch.compile</code> has a mode, "reduce-overhead", that does it with CUDA
          graphs. Set L above d in the explorer and compare one stream with the graph.
        </p>
      </Translated>
      <Ask q={questionsS223['ch22/22.3/q-graphs']} />

      <h2><Translated vi={"Một bước decode là một ứng dụng"}>A decode step is an application</Translated></h2>
      <Translated
        vi={
          <p>
            Sinh văn bản là cùng bức tranh đó ở quy mô nhỏ hơn. Mỗi bước chạy các phép chiếu và attention, một chuỗi kernel nhỏ
            theo từng phần tử với mỗi kernel một lần khởi chạy, một bộ lấy mẫu biến logits thành một token, và mã host biến token
            trở lại thành văn bản. Trong llama.cpp, chuỗi lấy mẫu mặc định chạy trên host trên một mảng token ứng viên; một nhánh
            thử nghiệm chuyển việc lấy mẫu vào graph của backend. Bộ lấy mẫu trên host là một phần còn lại Amdahl thuộc loại đặc
            biệt. Nó không co lại khi các kernel nhanh hơn, và bước kế tiếp không thể bắt đầu cho đến khi nó xong, nên chồng lấp
            không thể giấu nó.
          </p>
        }
      >
        <p>
          Generating text is the same picture at a smaller scale. Each step runs the projections and attention, a string of
          small elementwise kernels with a launch for each, a sampler that turns logits into a token, and host code that
          turns the token back into text. In llama.cpp the default sampler chain runs on the host over an array of candidate
          tokens; an experimental path moves sampling into the backend's graph. A sampler on the host is an Amdahl remainder
          of a particular kind. It does not shrink when the kernels get faster, and the next step cannot start until it has
          finished, so overlap cannot hide it.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Của khóa học này, không phải của sách. Bộ lấy mẫu của llama.cpp: notes/today/ch01-facts.md D2, đọc ngày 2026-11-20.
              Các tỉ lệ trong thiết lập sẵn decode là bịa ra để minh họa, không phải đo được.
            </p>
          }
        >
          <p>
            Ours, not the book's. llama.cpp's sampler: notes/today/ch01-facts.md D2, read 2026-11-20. The decode preset's
            shares are invented for illustration, not measured.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Chọn "Một bước decode (của khóa học này)" trong trình khám phá mô-đun ở trên. Làm cả hai loại kernel nhanh hơn 10 lần
            và đọc trần. Bật chồng lấp và xem nó giấu được gì và không giấu được gì. Rồi chuyển việc lấy mẫu lên GPU.
          </p>
        }
      >
        <p>
          Pick "A decode step (ours)" in the module explorer above. Make both kinds of kernel 10 times faster and read the
          ceiling. Turn overlap on and see what it hides and what it cannot. Then move sampling to the GPU.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Đó là lý do khóa học này đặt phần cuối của bước decode lên device: argmax ở mốc dự án M4 (
            <a href={href('ch10', 'labs')}>lab 10.3</a>) và lấy mẫu ở <a href={href('x1')}>module X1</a>. Khi đó thứ phải về tới
            host mỗi bước chỉ là một token id. Nếu bước kế tiếp cũng đọc id đó từ bộ nhớ device, host không bao giờ phải chờ nó,
            và phần việc còn lại trên host, khởi chạy và văn bản, chạy trong khi device làm việc.
          </p>
        }
      >
        <p>
          That is why this course puts the end of the decode step on the device: argmax in milestone M4 (
          <a href={href('ch10', 'labs')}>lab 10.3</a>) and sampling in <a href={href('x1')}>module X1</a>. What has to reach
          the host each step is then a token id. If the next step also reads that id from device memory, the host never
          waits for it, and the work left on the host, launches and text, runs while the device works.
        </p>
      </Translated>
      <Ask q={questionsS223['ch22/22.3/q-decode-kernels']} />

      <h2><Translated vi={"Các backend thực tế quyết định chồng lấp theo từng kernel"}>Real backends decide overlap kernel by kernel</Translated></h2>
      <Translated
        vi={
          <p>
            Trên Metal, cả llama.cpp lẫn MLX đều encode các kernel của mô hình vào những compute encoder đồng thời và chỉ thêm rào
            chắn buffer ở chỗ cần. Quy tắc của llama.cpp, như mã nguồn của nó phát biểu: một node có thể chạy cùng các node được
            encode trước nó nếu nó không ghi gì mà chúng đã đọc hoặc ghi và không đọc gì mà chúng đã ghi; nếu không, một rào chắn
            được chèn vào và việc theo dõi bắt đầu lại. MLX đặt một cờ khi một buffer đầu ra từng là đầu vào của nhóm dispatch
            trước đó, và chèn rào chắn khi đó. Đó chính là câu hỏi của <a href={href('ch22', '22.2')}>mục 22.2</a>, ai ghi và ai
            đọc, được hỏi cho cả kernel thay vì cho luồng.
          </p>
        }
      >
        <p>
          On Metal, llama.cpp and MLX both encode a model's kernels into concurrent compute encoders and add a buffer barrier
          only where one is needed. llama.cpp's rule, as its source states it: a node may run alongside the nodes encoded
          before it if it writes nothing they read or wrote and reads nothing they wrote; otherwise a barrier goes in and the
          tracking starts over. MLX sets a flag when an output buffer was an input of the previous group of dispatches, and
          inserts the barrier then. That is the question of <a href={href('ch22', '22.2')}>section 22.2</a>, who writes and
          who reads, asked of whole kernels instead of threads.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 on the default branches (ch22-facts D8)', '2026-11-20 trên các nhánh mặc định (ch22-facts D8)')}>
        <Translated
          vi={
            <p>
              llama.cpp:{' '}
              <a href="https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-metal" target="_blank" rel="noreferrer">
                ggml/src/ggml-metal
              </a>
              ; chạy đồng thời được bật trừ khi đặt <code>GGML_METAL_CONCURRENCY_DISABLE</code>. MLX:{' '}
              <a href="https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal" target="_blank" rel="noreferrer">
                mlx/backend/metal
              </a>
              . Cả hai thay đổi thường xuyên: hãy đọc thư mục, không phải số dòng.
            </p>
          }
        >
          <p>
            llama.cpp:{' '}
            <a href="https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-metal" target="_blank" rel="noreferrer">
              ggml/src/ggml-metal
            </a>
            ; concurrency is on unless <code>GGML_METAL_CONCURRENCY_DISABLE</code> is set. MLX:{' '}
            <a href="https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal" target="_blank" rel="noreferrer">
              mlx/backend/metal
            </a>
            . Both change often: read the directory, not a line number.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Khi đọc một trong hai backend, hãy nhớ chế độ theo cặp của trình khám phá. Một rào chắn chờ mọi thứ được encode trước
            nó, nên thứ tự encode các node quyết định mỗi rào chắn giữ lại bao nhiêu công việc.
          </p>
        }
      >
        <p>
          When you read either backend, keep the explorer's pairs in mind. A barrier waits for everything encoded before it,
          so the order the nodes are encoded in decides how much work each barrier holds back.
        </p>
      </Translated>

      <h2><Translated vi={"Kiểm tra"}>Check</Translated></h2>
      <Translated vi={<p>Thêm một câu hỏi về phép tính của mục này, câu cần nhớ khi ai đó trích dẫn mức tăng tốc của một kernel.</p>}>
        <p>One more question on the section's arithmetic, the one to remember when someone quotes a kernel's speed-up.</p>
      </Translated>
      <Ask q={questionsS223['ch22/22.3/q-host-share']} />
    </>
  );
}
