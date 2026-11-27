import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { benchRows, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  CUDA_QUEUES,
  DEFAULTS,
  HALO_BOOK,
  HIP_QUEUES,
  LANE_NAME,
  MAX_CHUNKS,
  MAX_STREAMS,
  ONE_ENGINE,
  PRESETS,
  START,
  SNIPPET,
  bestChunks,
  haloLanes,
  parseLabRows,
  simulate,
  stepAt,
  stepOp,
  sweep,
  type ClosedForm,
  type LabParts,
  type LabStaged,
  type PipeConfig,
  type Schedule,
  type View,
} from '../../sim/chunkpipe';
import { PinnedScene, PipeLanes, fmtN, lanesHeight } from './PinnedScene';

const LAB = 'ch23/lab1_overlap';
const CHECKED = '2026-11-20';
const STATUS_CU = 'emulator-checked and front-end-checked; unverified on CUDA and HIP hardware';
const STATUS_METAL = 'verified on Metal';

// ---------------------------------------------------------------- Fig. 23.13, redrawn
function HaloBench() {
  const [x, setX] = useState(HALO_BOOK.exchange);
  const t = { ...HALO_BOOK, exchange: x };
  const plain = haloLanes(t, false);
  const over = haloLanes(t, true);
  const span = Math.max(plain.total, 1);
  const W = 720;
  const L = 112;
  const X = (v: number) => L + (v / span) * (W - L - 44);
  const SHORT: Record<string, string> = {
    'halo exchange (top)': tr('top halo', 'biên trên'), 'halo exchange (bottom)': tr('bottom halo', 'biên dưới'), 'reduce L2 norm': tr('reduce', 'rút gọn'),
  };
  const diagram = (h: ReturnType<typeof haloLanes>, y0: number, title: string, key: string) => {
    let gpuRow = 0;
    return (
      <g data-halo={key} data-total={h.total}>
        <text x={8} y={y0 - 8} fontSize="12" fontWeight="600">{title}</text>
        <text x={L - 8} y={y0 + 16} fontSize="11" textAnchor="end">GPU</text>
        <text x={L - 8} y={y0 + 62} fontSize="11" textAnchor="end"><Translated vi="Mạng">Network</Translated></text>
        {h.bars.map((b, i) => {
          const sub = b.lane === 'gpu' ? gpuRow++ : 0;
          const y = b.lane === 'gpu' ? y0 + (sub === 0 ? 0 : 12 + sub * 9) : y0 + 48;
          const hgt = b.lane === 'gpu' ? (sub === 0 ? 20 : 8) : 22;
          const wide = X(b.to) - X(b.from);
          const text = wide > b.label.length * 5.6 ? b.label : SHORT[b.label] ?? b.label;
          return (
            <g key={i} data-bar={b.label}>
              <rect x={X(b.from)} y={y} width={Math.max(1, wide)} height={hgt} rx="2" fill="var(--ink)"
                fillOpacity={b.label.startsWith('reduce') ? 0.85 : b.label.includes('top') ? 0.55 : b.label.includes('bottom') ? 0.35 : 0.7} />
              {hgt >= 20 && wide > text.length * 5.6 ? (
                <text x={(X(b.from) + X(b.to)) / 2} y={y + hgt / 2 + 4} fontSize="10" textAnchor="middle"
                  style={{ fill: b.label.includes('bottom') ? 'var(--ink)' : 'var(--page)' }}>{tr(text, ({
                    'compute all points': 'tính mọi điểm',
                    'compute internal': 'tính bên trong',
                    'compute top': 'tính hàng trên',
                    'compute bottom': 'tính hàng dưới',
                    'halo exchange (top)': 'trao đổi vùng biên (trên)',
                    'halo exchange (bottom)': 'trao đổi vùng biên (dưới)',
                    'reduce L2 norm': 'rút gọn chuẩn L2',
                  } as Record<string, string>)[text] ?? text)}</text>
              ) : null}
              {sub === 1 ? (
                <text x={X(b.to) + 5} y={y + 13} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="hàng trên và hàng dưới">top and bottom rows</Translated></text>
              ) : null}
            </g>
          );
        })}
        <text x={X(h.total) + 5} y={y0 + 64} fontSize="11" fontWeight="700">{fmtN(h.total)}</text>
      </g>
    );
  };
  return (
    <Bench
      title={tr('Two modes, then one', 'Hai chế độ, rồi một')}
      task={tr(
        'Drag the exchange time up. At what point does the overlapped iteration start to grow?',
        'Kéo thời gian trao đổi lên. Đến lúc nào thì vòng lặp có chồng lấp bắt đầu dài ra?',
      )}
      foot={tr(
        "Our redrawing of Fig. 23.13 in the book's proportions: each boundary row takes a quarter of the interior's time, as the figure draws it. Unitless. Model: sim/chunkpipe.ts, haloLanes.",
        'Bản vẽ lại Fig. 23.13 của khóa học này theo tỉ lệ của sách: mỗi hàng rìa tốn một phần tư thời gian của phần bên trong, như hình vẽ thể hiện. Không có đơn vị. Mô hình: sim/chunkpipe.ts, haloLanes.',
      )}
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field" style={{ flexDirection: 'row', alignItems: 'center', gap: '0.5rem' }}>
          <span><Translated vi={<>Mỗi lần trao đổi vùng biên tốn <b data-out="halo-x">{fmtN(x)}</b></>}>Each halo exchange takes <b data-out="halo-x">{fmtN(x)}</b></Translated></span>
          <input type="range" min="0.5" max="8" step="0.5" value={x} aria-label={tr('Time of each halo exchange', 'Thời gian của mỗi lần trao đổi vùng biên')} onChange={(e) => setX(Number(e.target.value))} />
        </label>
        <span className="muted"><Translated vi={<>tính phần bên trong tốn {HALO_BOOK.internal}, mỗi hàng rìa {HALO_BOOK.boundary}, phép rút gọn chuẩn {HALO_BOOK.reduce}</>}>computing the interior takes {HALO_BOOK.internal}, each boundary row {HALO_BOOK.boundary}, the norm reduction {HALO_BOOK.reduce}</Translated></span>
      </div>
      <div className="panel" style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${W} 220`} role="img" aria-label={tr('One stencil iteration without and with overlap', 'Một vòng lặp stencil không và có chồng lấp')} style={{ width: '100%', minWidth: 560, display: 'block' }}>
          {diagram(plain, 26, tr('Without overlap: compute everything, then communicate', 'Không chồng lấp: tính mọi thứ, rồi trao đổi'), 'plain')}
          {diagram(over, 134, tr('With overlap: boundary rows first, exchange while the interior computes', 'Có chồng lấp: hàng rìa trước, trao đổi trong khi phần bên trong đang tính'), 'overlap')}
        </svg>
      </div>
      <p style={{ margin: '0.7rem 0 0' }} data-out="halo-verdict">
        <Translated
          vi={
            <>
              Không chồng lấp, vòng lặp tốn <b>{fmtN(plain.total)}</b>; có chồng lấp, <b>{fmtN(over.total)}</b>.{' '}
              {over.hidden
                ? 'Cả hai lần trao đổi kết thúc trước khi phần bên trong xong: việc giao tiếp được giấu đi.'
                : `Các lần trao đổi chạy quá phần bên trong ${fmtN(over.exposed)}: chừng đó giao tiếp không còn được giấu.`}{' '}
              Dù thế nào, phép rút gọn chuẩn cũng chờ mọi thứ.
            </>
          }
        >
          Without overlap the iteration takes <b>{fmtN(plain.total)}</b>; with it, <b>{fmtN(over.total)}</b>.{' '}
          {over.hidden
            ? 'Both exchanges end before the interior does: the communication is hidden.'
            : `The exchanges run ${fmtN(over.exposed)} past the interior: that much communication is no longer hidden.`}{' '}
          The norm reduction waits for everything either way.
        </Translated>
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- the reader's device
interface DeviceData {
  backend: string;
  view: View;
  device: string;
  parts: LabParts;
  staged: LabStaged[];
}
function useDevice(): DeviceData[] {
  const { reports } = useReports();
  return realBackends(reports).flatMap((b) => {
    const view: View = b === 'metal' ? 'metal' : 'cuda';
    const rows = benchRows(reports[b], LAB) as unknown as Record<string, unknown>[];
    const got = parseLabRows(rows, view, b);
    return got.parts ? [{ backend: b, view, device: String(reports[b].device?.name ?? b), parts: got.parts, staged: got.staged }] : [];
  });
}
const deviceConfig = (d: DeviceData): PipeConfig => ({
  ...DEFAULTS, a: d.parts.a, b: d.parts.b, c: d.parts.c, view: d.view, engines: d.parts.engines ?? 2, n: 16, streams: 4,
});
const sameParts = (cfg: PipeConfig, d: DeviceData) =>
  cfg.view === d.view && Math.abs(cfg.a - d.parts.a) < 1e-9 && Math.abs(cfg.b - d.parts.b) < 1e-9 && Math.abs(cfg.c - d.parts.c) < 1e-9;

function DeviceCompare({ cfg, d }: { cfg: PipeConfig; d: DeviceData }) {
  const ns = [1, 2, 4, 8, 16, 32, 64];
  const model = sweep(cfg, ns);
  const points = d.staged.filter((r) => r.streams === cfg.streams && (d.view === 'metal' || (r.order === cfg.order && r.host === cfg.host)));
  const W = 520;
  const H = 190;
  const pad = { l: 46, r: 14, t: 14, b: 34 };
  const ymax = Math.max(...model.map((m) => m.total), ...points.map((p) => p.units), 1) * 1.08;
  const X = (n: number) => pad.l + (Math.log2(n) / 6) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - (v / ymax) * (H - pad.t - pad.b);
  const k = d.view === 'metal' ? 'K' : 'S';
  return (
    <div className="panel" data-device={d.backend} style={{ marginTop: '0.8rem' }}>
      <Translated
        vi={
          <p style={{ margin: '0 0 0.4rem' }}>
            <b>{d.device}</b> ({d.backend}): các phần đo được của lab 23.1 quyết định các tổng, phần lớn nhất được quy về 24
            đơn vị (1 đơn vị = {fmtN(d.parts.unit * 1000)} ms). Đường: mô hình tại {k} = {cfg.streams}
            {d.view === 'metal' ? '' : `, ${cfg.order === 'depth' ? 'từng chunk một' : 'mọi copy-in trước'}, ${cfg.host === 'pinned' ? 'ghim' : 'phân trang'}`}. Chấm: những gì lab
            đo được với cùng thiết lập.
          </p>
        }
      >
        <p style={{ margin: '0 0 0.4rem' }}>
          <b>{d.device}</b> ({d.backend}): lab 23.1's measured parts set the totals, the largest scaled to 24 units (1 unit ={' '}
          {fmtN(d.parts.unit * 1000)} ms). Line: the model at {k} = {cfg.streams}
          {d.view === 'metal' ? '' : `, ${cfg.order === 'depth' ? 'chunk by chunk' : 'all copy-ins first'}, ${cfg.host}`}. Dots: what the
          lab measured with the same settings.
        </p>
      </Translated>
      {points.length ? (
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Model and measured totals against the number of chunks', 'Tổng theo mô hình và tổng đo được theo số chunk')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
          <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} stroke="var(--rule)" />
          <line x1={pad.l} y1={pad.t} x2={pad.l} y2={H - pad.b} stroke="var(--rule)" />
          {ns.map((n) => (
            <text key={n} x={X(n)} y={H - pad.b + 14} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{n}</text>
          ))}
          <text x={(W + pad.l) / 2} y={H - 4} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}><Translated vi="số chunk">chunks</Translated></text>
          <text x={8} y={pad.t - 2} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="đơn vị">units</Translated></text>
          {[0, 0.5, 1].map((f) => {
            const v = Math.round(ymax * f * 0.92);
            return (
              <g key={f}>
                <line x1={pad.l - 4} y1={Y(v)} x2={pad.l} y2={Y(v)} stroke="var(--ink-3)" />
                <text x={pad.l - 7} y={Y(v) + 3.5} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{v}</text>
              </g>
            );
          })}
          <polyline points={model.map((m) => `${X(m.n)},${Y(m.total)}`).join(' ')} fill="none" stroke="var(--ink)" strokeWidth="1.8" />
          {points.map((p, i) => (
            <circle key={i} data-measured={p.chunks} data-units={p.units} cx={X(p.chunks)} cy={Y(p.units)} r="4.5" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.8" />
          ))}
        </svg>
      ) : (
        <Translated
          vi={
            <p className="muted" style={{ margin: 0 }} data-device-empty>
              Lab đã đo {k} = {[...new Set(d.staged.map((r) => r.streams))].sort((x, y) => x - y).join(', ')}
              {d.view === 'metal' ? '' : ' (ghim, cả hai thứ tự; và một hàng pageable)'}. Chọn một trong các giá trị đó để xem số đo của nó ở đây.
            </p>
          }
        >
          <p className="muted" style={{ margin: 0 }} data-device-empty>
            The lab measured {k} = {[...new Set(d.staged.map((r) => r.streams))].sort((x, y) => x - y).join(', ')}
            {d.view === 'metal' ? '' : ' (pinned, both orders; and one pageable row)'}. Pick one of those to see its measurements here.
          </p>
        </Translated>
      )}
      <Translated
        vi={
          <p className="muted" style={{ margin: '0.4rem 0 0' }}>
            Số đo từ máy của bạn, không phải giá trị kỳ vọng. Mỗi số đo là cả một lời gọi, gồm cả phần thiết lập; mô hình không
            có chi phí đó trừ khi bạn đặt nó.
          </p>
        }
      >
        <p className="muted" style={{ margin: '0.4rem 0 0' }}>
          Measurements from your machine, not expected values. Each is a whole call, its setup included; the model has no such
          cost unless you set one.
        </p>
      </Translated>
    </div>
  );
}

// ---------------------------------------------------------------- the stepper
function Range({ label, value, min, max, step, onChange, out, disabled, aria }: {
  label: ReactNode; value: number; min: number; max: number; step: number; onChange: (v: number) => void; out: string; disabled?: boolean; aria: string;
}) {
  return (
    <label className="field" style={{ minWidth: '9.5rem' }}>
      <span>{label} <b data-out={out}>{fmtN(value)}</b></span>
      <input type="range" min={min} max={max} step={step} value={value} disabled={disabled} aria-label={aria} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}
function Seg<T extends string | number>({ label, value, options, onChange, disabled, name }: {
  label: string; value: T; options: { v: T; label: string }[]; onChange: (v: T) => void; disabled?: boolean; name: string;
}) {
  return (
    <span className="row" style={{ gap: '0.45rem' }}>
      <span className="muted">{label}</span>
      <span className="seg" role="group" aria-label={label} data-seg={name}>
        {options.map((o) => (
          <button key={String(o.v)} aria-pressed={value === o.v} disabled={disabled} onClick={() => onChange(o.v)} data-v={String(o.v)}>
            {o.label}
          </button>
        ))}
      </span>
    </span>
  );
}

function FormBox({ f, kind, total }: { f: ClosedForm; kind: 'e1' | 'nv'; total: number }) {
  return (
    <div style={{ marginTop: '0.5rem' }} data-form={kind} data-holds={f.holds ? 'yes' : 'no'} data-value={f.value}>
      <div>
        <b>{tr(f.name, ({ 'Pipeline with enough streams (E1)': 'Pipeline với đủ stream (E1)', "NVIDIA's rough estimate (A7)": 'Ước lượng thô của NVIDIA (A7)' } as Record<string, string>)[f.name] ?? f.name)}</b>: {f.formula} = <b data-out={kind + '-value'}>{fmtN(f.value)}</b>{' '}
        {f.holds ? <span className="chip ok"><Translated vi="các điều kiện của nó thỏa">its conditions hold</Translated></span> : <span className="chip"><Translated vi="điều kiện không thỏa">conditions not met</Translated></span>}
      </div>
      <div className="muted" style={{ fontSize: '0.76rem' }}>
        {tr(
          f.conditions.map((c) => (c.holds ? '✓ ' : '✗ ') + c.text).join('; '),
          f.conditions
            .map(
              (c) =>
                (c.holds ? '✓ ' : '✗ ') +
                (({
                  'a copy engine for each direction': 'mỗi chiều một copy engine',
                  'only one direction is copied': 'chỉ sao chép một chiều',
                  'one copy engine for both directions': 'một copy engine cho cả hai chiều',
                  'pinned host memory': 'bộ nhớ host được ghim',
                  'pageable host memory': 'bộ nhớ host pageable',
                  'equal chunks': 'các chunk bằng nhau',
                  'unequal chunks': 'các chunk không bằng nhau',
                  'no two chunks share a stream or queue': 'không có hai chunk nào dùng chung stream hay hàng đợi',
                  'each stream and queue holds one chunk': 'mỗi stream và hàng đợi giữ một chunk',
                  'issued breadth first with chunks sharing a stream or queue': 'phát lệnh theo chiều rộng với các chunk dùng chung stream hay hàng đợi',
                  "nothing is copied back, as in NVIDIA's example": 'không sao chép gì về, như ví dụ của NVIDIA',
                  'the result is copied back too': 'kết quả cũng được sao chép về',
                  'no per-operation cost': 'không có chi phí mỗi thao tác',
                  'a per-operation cost': 'có chi phí mỗi thao tác',
                } as Record<string, string>)[c.text] ??
                  c.text.replace(/^chunks sharing a stream or queue are (\S+) apart, and /, 'các chunk dùng chung stream hay hàng đợi cách nhau $1, và ')),
            )
            .join('; '),
        )}.{' '}
        {f.holds ? tr('The schedule agrees.', 'Lịch chạy khớp.') : f.matches ? tr('It happens to agree here anyway.', 'Ở đây nó tình cờ vẫn khớp.') : tr(`The schedule gives ${fmtN(total)} instead.`, `Thay vào đó, lịch chạy cho ra ${fmtN(total)}.`)}
      </div>
    </div>
  );
}

function Stepper({ cfg, setCfg, devices }: { cfg: PipeConfig; setCfg: (c: PipeConfig) => void; devices: DeviceData[] }) {
  const cu = useLabFile(SNIPPET.cuda);
  const mm = useLabFile(SNIPPET.metal);
  const locale = useLocale();
  const s: Schedule = useMemo(() => simulate(cfg), [cfg, locale]);
  const best = useMemo(() => bestChunks(cfg), [cfg]);
  const [t, setT] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const metal = cfg.view === 'metal';
  const span = Math.max(s.total, s.serial, 1e-6);
  const k = t == null ? null : stepAt(s, t);
  const step = k == null ? null : s.steps[k];
  const curOp = step ? stepOp(step) : null;
  const op = curOp != null ? s.ops[curOp] : null;
  const atStep = step != null && t != null && Math.abs(step.t - t) < 1e-9;

  useEffect(() => {
    setT(null);
    setPlaying(false);
  }, [cfg]);
  const next = () => {
    if (t == null) return setT(s.steps.length ? s.steps[0].t : 0);
    const nx = s.steps.find((x) => x.t > t + 1e-9);
    setT(nx ? nx.t : s.total);
  };
  const back = () => {
    if (t == null) return;
    const prev = [...s.steps].reverse().find((x) => x.t < t - 1e-9);
    setT(prev ? prev.t : null);
  };
  const finished = t != null && t >= s.total - 1e-9 && (k == null || k === s.steps.length - 1);
  useEffect(() => {
    if (!playing) return;
    if (finished) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(next, 650);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, t, finished]);

  const set = (o: Partial<PipeConfig>) => setCfg({ ...cfg, ...o });
  const narration: ReactNode = t == null ? (
    <Translated vi={<>Chưa có gì chạy. {s.ops.length} thao tác đang chờ được xếp hàng{metal ? ' bởi vòng lặp CPU' : ' bởi luồng host'}. Bấm Bước.</>}>
      <>Nothing has run yet. {s.ops.length} operations wait to be queued{metal ? ' by the CPU loop' : ' by the host thread'}. Press Step.</>
    </Translated>
  ) : step ? (
    <>
      <b>t = {fmtN(t)}{atStep ? '' : tr(` (the last event was at ${fmtN(step.t)})`, ` (sự kiện gần nhất ở ${fmtN(step.t)})`)}.</b>{' '}
      {step.events.map((e, i) => (
        <span key={i} data-ev={e.kind}>
          {e.kind === 'wait' || e.kind === 'block' ? <i>{e.text}</i> : e.text}{' '}
        </span>
      ))}
      {finished ? <b><Translated vi={<> Xong toàn bộ ở {fmtN(s.total)}.</>}> All done at {fmtN(s.total)}.</Translated></b> : null}
    </>
  ) : (
    <Translated vi={<>t = {fmtN(t)}. Chưa có gì xảy ra.</>}>
      <>t = {fmtN(t)}. Nothing has happened yet.</>
    </Translated>
  );

  // The code beside the timeline: the part of the snippet that issues work.
  const slice = (src: string | null, from: number, to: number) => (src ? src.split('\n').slice(from - 1, to).join('\n') : '');
  const hl = op ? op.lines : [];
  const codeTabs = metal
    ? [{ label: tr('Metal host', 'Mã host Metal'), code: slice(mm, 21, 46), status: STATUS_METAL, highlight: hl }]
    : [{ label: 'CUDA', code: slice(cu, 34, 59), status: 'emulator-checked, front-end-checked', highlight: hl }];

  const devicesHere = devices.filter((d) => d.view === cfg.view || devices.length === 1);
  const shownDevice = devices.find((d) => sameParts(cfg, d));
  const H = lanesHeight(s) + 24;
  const scroller = useRef<HTMLDivElement>(null);

  return (
    <Bench
      title={tr('Copy, compute, copy back', 'Sao chép, tính, sao chép về')}
      task={tr(
        'Step through the starting case first. Then find a setting that beats it, and one that loses all the overlap.',
        'Chạy từng bước trường hợp khởi đầu trước. Rồi tìm một thiết lập thắng nó, và một thiết lập mất hết chồng lấp.',
      )}
      foot={
        <Translated
          vi={
            <>
              Mỗi hộp là một thao tác, ghi chunk của nó và tô theo stream của nó; hộp có viền là hộp vừa bắt đầu, và các hộp
              cùng stream có viền đậm hơn. Gạch chéo: host bị chặn hoặc đang chờ. Chấm: thời gian rảnh của làn bận nhất (bong
              bóng). Thời gian không có đơn vị. Mô hình: sim/chunkpipe.ts.
            </>
          }
        >
          <>
            Each box is one operation, labelled with its chunk and shaded by its stream; the outlined box is the one that just
            started, and boxes of the same stream have a firmer outline. Hatched: the host blocked or waiting. Dotted: idle time of
            the busiest lane (the bubble). Times are unitless. Model: sim/chunkpipe.ts.
          </>
        </Translated>
      }
    >
      <div data-stepper data-total={s.total} data-view={cfg.view}>
        <div className="row" style={{ marginBottom: '0.6rem', alignItems: 'flex-end' }}>
          <Range label={tr('Chunks n', 'Số chunk n')} value={cfg.n} min={1} max={MAX_CHUNKS} step={1} out="n" aria={tr('Number of chunks', 'Số chunk')} onChange={(v) => set({ n: v })} />
          <Range label={metal ? tr('Buffers in flight K', 'Số buffer đang dùng K') : tr('Streams S', 'Số stream S')} value={cfg.streams} min={1} max={MAX_STREAMS} step={1}
            out="streams" // i18n-ignore: data key
            aria={metal ? tr('Buffers in flight', 'Số buffer đang dùng') : tr('Number of streams', 'Số stream')} onChange={(v) => set({ streams: v })} />
          <Range label={metal ? tr('Fill (copy-in)', 'Nạp (copy-in)') : tr('Copy-in', 'Copy-in')} value={cfg.a} min={0} max={32} step={0.5} out="a" aria={tr('Total copy-in time', 'Tổng thời gian copy-in')} onChange={(v) => set({ a: v })} />
          <Range label={tr('Kernel', 'Kernel')} value={cfg.b} min={0} max={32} step={0.5} out="b" aria={tr('Total kernel time', 'Tổng thời gian kernel')} onChange={(v) => set({ b: v })} />
          <Range label={metal ? tr('Drain (copy-out)', 'Rút (copy-out)') : tr('Copy-out', 'Copy-out')} value={cfg.c} min={0} max={32} step={0.5} out="c" aria={tr('Total copy-out time', 'Tổng thời gian copy-out')} onChange={(v) => set({ c: v })} />
          <span className="muted" data-out="chunk-size">
            <Translated vi={<>mỗi chunk {cfg.items == null ? `1/${cfg.n}` : 'khoảng 1/' + cfg.n} dữ liệu; chunk j vào {metal ? 'buffer' : 'stream'} j mod {cfg.streams}</>}>
              each chunk {cfg.items == null ? `1/${cfg.n}` : 'about 1/' + cfg.n} of the data; chunk j goes to {metal ? 'buffer' : 'stream'} j mod {cfg.streams}
            </Translated>
          </span>
        </div>
        <div className="row" style={{ marginBottom: '0.6rem' }}>
          <Seg
            name="view" // i18n-ignore: data key
            label={tr('View', 'Chế độ xem')} value={cfg.view} onChange={(v) => set({ view: v })} options={[{ v: 'cuda', label: tr('CUDA / HIP', 'CUDA / HIP') }, { v: 'metal', label: 'Metal' }]} />
          <Seg
            name="engines" // i18n-ignore: data key
            label={tr('Copy engines', 'Copy engine')} value={cfg.engines} disabled={metal} onChange={(v) => set({ engines: v })}
            options={[{ v: 2, label: tr('two', 'hai') }, { v: 1, label: tr('one, shared', 'một, dùng chung') }]} />
          <Seg
            name="order" // i18n-ignore: data key
            label={tr('Issue order', 'Thứ tự phát lệnh')} value={cfg.order} disabled={metal} onChange={(v) => set({ order: v })}
            options={[{ v: 'depth', label: tr('chunk by chunk', 'từng chunk một') }, { v: 'breadth', label: tr('all copy-ins first', 'mọi copy-in trước') }]} />
          <Seg
            name="host" // i18n-ignore: data key
            label={tr('Host memory', 'Bộ nhớ host')} value={cfg.host} disabled={metal} onChange={(v) => set({ host: v })}
            options={[{ v: 'pinned', label: tr('pinned', 'ghim') }, { v: 'pageable', label: tr('pageable', 'pageable') }]} />
          <Seg
            name="queues" // i18n-ignore: data key
            label={tr('Hardware queues', 'Hàng đợi phần cứng')} value={isFinite(cfg.queues) ? String(cfg.queues) : 'inf'} disabled={metal}
            onChange={(v) => set({ queues: v === 'inf' ? Infinity : Number(v) })}
            options={[{ v: 'inf', label: tr('unlimited', 'không giới hạn') }, { v: String(CUDA_QUEUES), label: tr(`${CUDA_QUEUES} (CUDA)`, `${CUDA_QUEUES} (CUDA)`) }, { v: String(HIP_QUEUES), label: tr(`${HIP_QUEUES} (HIP)`, `${HIP_QUEUES} (HIP)`) }]} />
          <Range label={<Translated vi={<>Chi phí mỗi thao tác <span className="muted">(không nhà cung cấp nào công bố con số này)</span></>}>
            <>Per-operation cost <span className="muted">(no vendor publishes this figure)</span></>
          </Translated>} value={cfg.overhead} min={0} max={2} step={0.05}
            out="overhead" // i18n-ignore: data key
            aria={tr('Per-operation cost', 'Chi phí mỗi thao tác')} onChange={(v) => set({ overhead: v })} />
        </div>
        {metal ? (
          <Translated
            vi={
              <p className="muted" style={{ margin: '0 0 0.6rem' }}>
                Chế độ xem Metal: không áp dụng copy engine, thứ tự phát lệnh, bộ nhớ ghim hay hàng đợi phần cứng. CPU rút dữ
                liệu khỏi một buffer, nạp lại nó và commit; GPU chạy các command buffer theo thứ tự commit.
              </p>
            }
          >
            <p className="muted" style={{ margin: '0 0 0.6rem' }}>
              Metal view: no copy engine, issue order, pinned memory or hardware queue applies. The CPU drains a buffer, refills it
              and commits; the GPU runs command buffers in commit order.
            </p>
          </Translated>
        ) : null}
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <span className="muted"><Translated vi="Thiết lập sẵn:">Presets:</Translated></span>
          {PRESETS.map((p) => (
            <button key={p.id} className="quiet" data-preset={p.id} onClick={() => setCfg({ ...p.cfg, view: cfg.view })}>{tr(p.label, ({
              'best-practices': 'ví dụ Best Practices',
              equal: 'các giai đoạn bằng nhau',
              'copy-bound': 'bị giới hạn bởi sao chép',
              'compute-bound': 'bị giới hạn bởi tính toán',
              'p-pipeline': 'trường hợp của phần dự đoán',
            } as Record<string, string>)[p.id] ?? p.label)}</button>
          ))}
          {devicesHere.map((d) => (
            <button key={'dev-' + d.backend} className="quiet" data-preset={'device-' + d.backend} onClick={() => setCfg(deviceConfig(d))}>
              <Translated vi={<>thiết bị của bạn (lab 23.1{devices.length > 1 ? `, ${d.backend}` : ''})</>}>
                your device (lab 23.1{devices.length > 1 ? `, ${d.backend}` : ''})
              </Translated>
            </button>
          ))}
        </div>

        <div className="row" style={{ marginBottom: '0.6rem' }}>
          <button onClick={() => { setPlaying(false); setT(null); }} disabled={t == null}><Translated vi="Làm lại">Restart</Translated></button>
          <button onClick={() => { setPlaying(false); back(); }} disabled={t == null}><Translated vi="Lùi">Back</Translated></button>
          <button className="primary" onClick={() => { setPlaying(false); next(); }} disabled={finished}><Translated vi="Bước">Step</Translated></button>
          <button onClick={() => setPlaying(!playing)} disabled={finished}>{playing ? tr('Pause', 'Tạm dừng') : tr('Play', 'Chạy')}</button>
          <button onClick={() => { setPlaying(false); setT(s.total); }} disabled={finished}><Translated vi="Chạy đến cuối">Run to the end</Translated></button>
          <input type="range" min={0} max={1000} step={1} value={t == null ? 0 : Math.round((t / span) * 1000)} aria-label={tr('Time', 'Thời gian')}
            style={{ flex: '1 1 10rem' }} onChange={(e) => { setPlaying(false); setT((Number(e.target.value) / 1000) * span); }} />
          <span className="muted" aria-live="polite" data-out="clock"><Translated vi={<>t = {t == null ? '·' : fmtN(t)} trên {fmtN(s.total)}</>}>t = {t == null ? '·' : fmtN(t)} of {fmtN(s.total)}</Translated></span>
        </div>

        <div className="narrate" role="status" data-narrate style={{ marginBottom: '0.8rem' }}>{narration}</div>

        <div className="panel" style={{ overflowX: 'auto', marginBottom: '0.8rem', padding: '0.5rem' }} ref={scroller}>
          <svg viewBox={`0 0 940 ${H}`} role="img" aria-label={tr('Timeline of the chunked pipeline, one lane per resource', 'Dòng thời gian của pipeline chia chunk, mỗi tài nguyên một làn')}
            style={{ width: '100%', minWidth: 640, display: 'block' }} data-timeline>
            <PipeLanes s={s} t={t ?? -1} x={0} y={18} w={940} span={span} current={curOp} id="cp-stepper" />
          </svg>
        </div>

        <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 7fr) minmax(0, 5fr)' }}>
          <div>
            {(metal ? mm : cu) ? (
              <Code tabs={codeTabs} firstLine={metal ? 21 : 34}
                caption={<Translated vi={<>Phần của <code>labs/{metal ? SNIPPET.metal : SNIPPET.cuda}</code> xếp hàng công việc; các dòng được đánh dấu đã phát lệnh cho thao tác vừa bắt đầu.</>}>
                  <>The part of <code>labs/{metal ? SNIPPET.metal : SNIPPET.cuda}</code> that queues the work; the marked lines issued the operation that just started.</>
                </Translated>} />
            ) : null}
          </div>
          <div className="panel" data-readouts>
            <table className="threads wrap">
              <tbody>
                {([
                  [tr('Total', 'Tổng'), <b>{fmtN(s.total)}</b>, 'total'],
                  [tr(`In one piece, a + b + c${cfg.overhead > 0 ? ' + costs' : ''}`, `Gộp một khối, a + b + c${cfg.overhead > 0 ? ' + chi phí' : ''}`), fmtN(s.serial), 'serial'],
                  [tr('Speed-up', 'Tăng tốc'), `${fmtN(s.speedup)}×`, 'speedup'],
                  [tr('Busiest lane', 'Làn bận nhất'), tr(`${LANE_NAME[s.bottleneck]} (${fmtN(s.busy[s.bottleneck] ?? 0)} busy)`, `${({
                    host: 'Luồng host',
                    h2d: 'Engine sao chép vào',
                    compute: 'Tính toán',
                    d2h: 'Engine sao chép ra',
                    copy: 'Engine sao chép (cả hai chiều)',
                    cpu: 'CPU: rút ra, rồi nạp vào',
                    gpu: 'GPU',
                  } as Record<string, string>)[s.bottleneck] ?? LANE_NAME[s.bottleneck]} (${fmtN(s.busy[s.bottleneck] ?? 0)} bận)`), 'bottleneck'],
                  [tr('Its idle time, the bubble', 'Thời gian rảnh của nó, tức bong bóng'), fmtN(s.bubble), 'bubble'],
                  [tr('Fastest chunk count, 1 to 64', 'Số chunk nhanh nhất, từ 1 đến 64'), best.n === 1 ? tr(`n = 1 (${fmtN(best.total)}): cutting does not help here`, `n = 1 (${fmtN(best.total)}): cắt nhỏ không giúp gì ở đây`) : `n = ${best.n} (${fmtN(best.total)})`, 'best-n'],
                ] as [string, ReactNode, string][]).map(([k, v, id]) => (
                  <tr key={id}>
                    <td>{k}</td>
                    <td data-out={id}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {s.e1 ? <FormBox f={s.e1} kind="e1" total={s.total} /> : null}
            {s.nvidia && cfg.c === 0 ? <FormBox f={s.nvidia} kind="nv" total={s.total} /> : null}
            {s.lowerBound != null ? (
              <Translated
                vi={
                  <p style={{ margin: '0.5rem 0 0' }} data-out="lower-bound">
                    Ở đây không có dạng đóng. CPU làm mọi lần nạp và rút, GPU làm mọi kernel, nên tổng ít nhất là n ·
                    max(nạp + rút, kernel) mỗi chunk = <b>{fmtN(s.lowerBound)}</b>.
                  </p>
                }
              >
                <p style={{ margin: '0.5rem 0 0' }} data-out="lower-bound">
                  No closed form here. The CPU does every fill and drain and the GPU every kernel, so the total is at least n ·
                  max(fill + drain, kernel) per chunk = <b>{fmtN(s.lowerBound)}</b>.
                </p>
              </Translated>
            ) : null}
          </div>
        </div>
        {shownDevice ? <DeviceCompare cfg={cfg} d={shownDevice} /> : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
/** q-twostreams: three equal stages, 12 chunks, from the stepper's model. */
export const TWO_STREAMS = (streams: number) => simulate({ ...DEFAULTS, a: 12, b: 12, c: 12, n: 12, streams });
const X1 = (v: number) => v.toFixed(1);
/** The false dependency: the "equal stages" preset with 16 chunks in 16 streams, breadth first. */
export const FALSE_DEP: PipeConfig = { ...PRESETS.find((p) => p.id === 'equal')!.cfg, n: 16, streams: 16, order: 'breadth', queues: CUDA_QUEUES };
export const FALSE_DEP_FREE = simulate({ ...FALSE_DEP, queues: Infinity }).total;
export const FALSE_DEP_8 = simulate(FALSE_DEP).total;

export const questionsS233: Record<string, Question> = {
  'ch23/23.3/q-onestream': {
    kind: 'choice',
    id: 'ch23/23.3/q-onestream',
    prompt: (
      <Translated vi={<>Bạn chia một công việc thành 8 chunk nhưng xếp hàng tất cả vào một stream, trên một GPU có hai copy engine. Bạn nên kỳ vọng tăng tốc bao nhiêu so với công việc không chia?</>}>
        <>You split a job into 8 chunks but queue all of them in one stream, on a GPU with two copy engines. What speed-up over the unsplit job should you expect?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Khoảng 1×, hoặc tệ hơn một chút khi tính chi phí cố định của mỗi chunk.">About 1×, or a little worse once each chunk's fixed cost counts.</Translated>, correct: true },
      { label: <Translated vi="Khoảng 3×: mỗi làn một việc, cho copy-in, kernel và copy-out.">About 3×: one lane each for the copy-in, the kernel and the copy-out.</Translated>, why: <Translated vi="Đó là điều ba làn bận rộn có thể đem lại; một stream không bao giờ cho hai thao tác của nó chạy cùng lúc.">That is what three busy lanes could give; one stream never lets two of its operations run at once.</Translated> },
      { label: <Translated vi="Khoảng 8×: mỗi chunk là một phần tám công việc, nên mỗi chunk xong sớm hơn.">About 8×: each chunk is an eighth of the work, so each finishes sooner.</Translated>, why: <Translated vi="Cắt nhỏ công việc không làm tổng thể nhanh hơn. Chỉ chồng lấp mới làm được, và một stream thì không có chồng lấp.">Cutting the work does not make the whole faster. Only overlap can, and one stream has none.</Translated> },
      { label: <Translated vi="Tùy vào occupancy (mức chiếm dụng) của kernel, thứ quyết định có bao nhiêu việc sao chép vừa chạy bên cạnh nó.">It depends on the kernel's occupancy, which sets how much copying fits beside it.</Translated>, why: <Translated vi="Occupancy quyết định mỗi kernel chạy nhanh thế nào, không quyết định một lần sao chép có chạy bên dưới nó được không. Trong một stream thì không được.">Occupancy decides how fast each kernel runs, not whether a copy can run under it. In one stream it cannot.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một stream chạy các thao tác của nó lần lượt, theo thứ tự chúng được đưa vào (pp. 555–556). Copy-in của chunk 1 chờ
            copy-out của chunk 0, nên chỉ có một thao tác chạy tại một thời điểm. Trong trình chạy từng bước, đặt S bằng 1 và
            đổi n tùy ý: tổng vẫn ở mức thời gian gộp một khối, và tăng lên nếu có chi phí mỗi thao tác.
          </p>
        }
      >
        <p>
          A stream runs its operations one after another, in the order they were inserted (pp. 555–556). Chunk 1's copy-in waits
          for chunk 0's copy-out, so only one operation ever runs. In the stepper, set S to 1 and move n anywhere: the total
          stays at the one-piece time, and grows with a per-operation cost.
        </p>
      </Translated>
    ),
    source: 'ch. 23 §23.3, pp. 555–556; the stepper (sim/chunkpipe.ts)',
    vi: { source: 'ch. 23 §23.3, pp. 555–556; trình chạy từng bước (sim/chunkpipe.ts)' },
  },
  'ch23/23.3/q-pageable': {
    kind: 'choice',
    id: 'ch23/23.3/q-pageable',
    prompt: (
      <Translated
        vi={
          <>
            Vòng lặp chia chunk của bạn không nhanh hơn vòng lặp tuần tự, và một bản trace cho thấy mỗi{' '}
            <code>cudaMemcpyAsync</code> chỉ trả về sau khi lần sao chép của nó xong. Nguyên nhân và cách sửa?
          </>
        }
      >
        Your staged loop is no faster than the serial one, and a trace shows each <code>cudaMemcpyAsync</code> returning only
        after its copy has finished. Cause and fix?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi={<>Buffer host đến từ <code>malloc</code>, nên mỗi lần sao chép chạy đồng bộ. Hãy cấp phát chúng bằng <code>cudaMallocHost</code>.</>}>
            <>The host buffers came from <code>malloc</code>, so each copy ran synchronously. Allocate them with <code>cudaMallocHost</code>.</>
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi={<>Các stream được tạo với <code>cudaStreamNonBlocking</code>, khiến mỗi lần sao chép của chúng chờ host.</>}>
            <>The streams were created with <code>cudaStreamNonBlocking</code>, which makes each of their copies wait for the host.</>
          </Translated>
        ),
        why: <Translated vi="Cờ đó chỉ ngăn một stream đồng bộ hóa với default stream kiểu cũ (legacy). Nếu có gì, nó giúp chồng lấp.">That flag only stops a stream from synchronising with the legacy default stream. It helps overlap, if anything.</Translated>,
      },
      {
        label: (
          <Translated vi={<><code>cudaMemcpyAsync</code> cần <code>cudaMemcpyDefault</code> làm chiều sao chép, nếu không nó quay về lần sao chép đồng bộ.</>}>
            <><code>cudaMemcpyAsync</code> needs <code>cudaMemcpyDefault</code> as its direction, or it falls back to a synchronous copy.</>
          </Translated>
        ),
        why: <Translated vi="Đối số chiều sao chép không liên quan gì đến lúc lời gọi trả về.">The direction argument has nothing to do with when the call returns.</Translated>,
      },
      {
        label: <Translated vi="Lần khởi chạy kernel cần tham số thứ tư bằng 1, nếu không các lần sao chép quanh nó đi vào default stream.">The kernel launch needs a fourth parameter of 1, or the copies around it go to the default stream.</Translated>,
        why: <Translated vi="Tham số thứ tư là một stream, không phải một con số, và triệu chứng nằm ở các lần sao chép.">The fourth parameter is a stream, not a number, and the symptom is in the copies.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Từ bộ nhớ phân trang, runtime trung chuyển lần sao chép qua một buffer được ghim, và host phải tham gia (pp. 562–563).
            Tài liệu tham khảo của NVIDIA: lần sao chép như vậy "will revert to a synchronous behavior which will not overlap with
            other work" (A6). Header của AMD: "the memory copy will be performed synchronously" (B4); ở đó cấp phát ghim là{' '}
            <code>hipHostMalloc</code>. Công tắc pageable của trình chạy từng bước cho thấy kết quả.
          </p>
        }
      >
        <p>
          From pageable memory the runtime stages the copy through a pinned buffer, and the host has to take part (pp. 562–563).
          NVIDIA's reference: such a copy "will revert to a synchronous behavior which will not overlap with other work" (A6).
          AMD's header: "the memory copy will be performed synchronously" (B4); there the pinned allocation is{' '}
          <code>hipHostMalloc</code>. The stepper's pageable switch shows the result.
        </p>
      </Translated>
    ),
    source: 'ch. 23 §23.3, pp. 562–563; notes/today/ch23-facts.md A6, B4', // i18n-ignore: citation only
  },
  'ch23/23.3/q-bottleneck': {
    kind: 'numeric',
    id: 'ch23/23.3/q-bottleneck',
    prompt: (
      <Translated
        vi={
          <>
            Tổng thời gian của một công việc: copy-in 8 ms, kernel 12 ms, copy-out 4 ms. Bạn cắt nó thành 4 chunk bằng nhau,
            mỗi chunk một stream, trên một GPU có hai copy engine, và chi phí mỗi chunk không đáng kể. Nó tốn bao lâu, tính
            bằng ms?
          </>
        }
      >
        A job's totals: copy-in 8 ms, kernel 12 ms, copy-out 4 ms. You cut it into 4 equal chunks, one stream per chunk, on a
        GPU with two copy engines, and the per-chunk cost is negligible. How long does it take, in ms?
      </Translated>
    ),
    answer: 15,
    unit: 'ms',
    explain: (
      <Translated
        vi={
          <p>
            Mỗi chunk: 2, 3 và 1 ms. Chunk đầu tiên cần cả ba, 6 ms. Sau đó làn kernel, làn chậm nhất, xong thêm một chunk mỗi
            3 ms: 6 + 3 × 3 = 15 ms, so với 24 ms khi gộp một khối. Đó là a + b + c + (n − 1)·max(a, b, c) với thời gian mỗi
            chunk. Các con số được đặt ra cho bài tập; số của thiết bị bạn đến từ lab 23.1.
          </p>
        }
      >
        <p>
          Per chunk: 2, 3 and 1 ms. The first chunk needs all three, 6 ms. After that the kernel lane, the slowest, finishes one
          more chunk every 3 ms: 6 + 3 × 3 = 15 ms, against 24 ms in one piece. That is a + b + c + (n − 1)·max(a, b, c) for
          per-chunk times. The numbers are made up for the exercise; your device's come from lab 23.1.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch23-facts.md E1 (our arithmetic, after Best Practices 10.1.2)',
    vi: { source: 'notes/today/ch23-facts.md E1 (phép tính của khóa học này, theo Best Practices 10.1.2)' },
  },
  'ch23/23.3/q-twostreams': {
    kind: 'choice',
    id: 'ch23/23.3/q-twostreams',
    prompt: (
      <Translated
        vi={
          <>
            Ba giai đoạn bằng nhau, 12 chunk. Trình chạy từng bước cho khoảng {X1(TWO_STREAMS(2).speedup)}× với 2 stream và
            khoảng {X1(TWO_STREAMS(3).speedup)}× với 3, mức cao nhất mà bất kỳ số stream nào cho được ở đây. Vì sao stream thứ
            hai không đủ để giữ cả ba làn bận rộn?
          </>
        }
      >
        Three equal stages, 12 chunks. The stepper gives about {X1(TWO_STREAMS(2).speedup)}× with 2 streams and about{' '}
        {X1(TWO_STREAMS(3).speedup)}× with 3, the most any stream count gives here. Why is a second stream not enough to keep
        all three lanes busy?
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Chunk i + 2 dùng chung stream với chunk i, nên copy-in của nó chờ copy-out của chunk i.">Chunk i + 2 shares a stream with chunk i, so its copy-in waits for chunk i's copy-out.</Translated>,
        correct: true,
      },
      { label: <Translated vi="GPU chạy tối đa hai stream cùng lúc, nên stream thứ ba sẽ phải chờ một chỗ trống.">The GPU runs at most two streams at once, so a third stream would wait for a free slot.</Translated>, why: <Translated vi="Không có gì giới hạn ở hai. Giới hạn hàng đợi phần cứng là 8 và 4, và chúng chỉ có tác dụng khi có nhiều stream hơn.">Nothing limits it to two. The hardware-queue limits are 8 and 4, and they bind only with more streams.</Translated> },
      { label: <Translated vi="Stream thứ hai dùng chung copy engine với stream thứ nhất, nên các copy-in của chúng phải luân phiên.">The second stream shares the first one's copy engine, so their copy-ins take turns.</Translated>, why: <Translated vi="Mọi stream đều dùng chung các engine; với hai engine, mỗi chiều có engine riêng.">Every stream shares the engines; with two engines, each direction has its own.</Translated> },
      { label: <Translated vi="Một stream chỉ giữ được một kernel mỗi lúc, nên mỗi kernel phải chờ kernel trước nó.">A stream can hold only one kernel at a time, so each kernel waits for the one before.</Translated>, why: <Translated vi="Một stream giữ được bao nhiêu thao tác cũng được, theo thứ tự.">A stream holds any number of operations, in order.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một stream chạy theo thứ tự. Với 2 stream, chunk 2 nằm trong stream 0 sau copy-out của chunk 0, nên nó không thể
            bắt đầu cho đến khi chunk 0 đi hết đường; giữ ba giai đoạn bận rộn cần ba chunk đang chạy, mỗi stream một chunk. Khi
            đó mỗi cặp chunk tốn 3 đơn vị: 19 cho 12 chunk, so với 36 khi gộp một khối và 14 với 3 stream. Quy tắc kinh nghiệm
            chúng tôi dùng: S × giai đoạn chậm nhất mỗi chunk phải phủ được a + b + c của cả một chunk, ở đây 3 × 1 ≥ 3.
          </p>
        }
      >
        <p>
          A stream runs in order. With 2 streams, chunk 2 sits in stream 0 behind chunk 0's copy-out, so it cannot start until
          chunk 0 is all the way through; keeping three stages busy takes three chunks in flight, one per stream. Each pair of chunks then takes 3 units: 19 for 12 chunks, against 36 in one piece and 14 with 3 streams. The
          rule of thumb we use: S × the slowest per-chunk stage must cover a whole chunk's a + b + c, here 3 × 1 ≥ 3.
        </p>
      </Translated>
    ),
    source: 'our model (sim/chunkpipe.ts tests: n = 12 with 1, 1, 1 per chunk gives 19 with 2 streams, 14 with 3)',
    vi: { source: 'mô hình của khóa học này (các test của sim/chunkpipe.ts: n = 12 với 1, 1, 1 mỗi chunk cho 19 với 2 stream, 14 với 3)' },
  },
  'ch23/23.3/q-overhead': {
    kind: 'choice',
    id: 'ch23/23.3/q-overhead',
    prompt: (
      <Translated vi={<>Tăng từ 16 lên 256 chunk làm công việc chia chunk của bạn chậm hơn. Lý do khả dĩ nhất là gì?</>}>
        <>Going from 16 to 256 chunks made your staged job slower. What is the most likely reason?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Mỗi chunk thêm một chi phí cố định vào mỗi lần sao chép và mỗi kernel, còn bong bóng mà nó thu nhỏ vốn đã nhỏ.">Each chunk adds a fixed cost to every copy and kernel, and the bubble it shrinks was small.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Các lần sao chép nhỏ hơn dùng được tỉ lệ đường truyền cao hơn, làm kernel thiếu băng thông bộ nhớ.">Smaller copies reach a higher share of the link, which starves the kernels of memory bandwidth.</Translated>, why: <Translated vi="Không tài liệu nào được kiểm tra ở đây nói vậy, và mô hình vẫn chậm đi mà không cần hiệu ứng nào như thế.">Nothing in the documentation checked here says so, and the model slows down without any such effect.</Translated> },
      { label: <Translated vi="256 chunk vượt quá giới hạn được ghi trong tài liệu của một stream là 64 thao tác xếp hàng, nên host phải chờ chỗ trống.">256 chunks exceed a stream's documented limit of 64 queued operations, so the host waits for room.</Translated>, why: <Translated vi="Không có giới hạn nào như vậy trong tài liệu được kiểm tra cho chương này. Con số 64 bạn có thể nhớ là mặc định của Metal cho số command buffer chưa hoàn tất trên một hàng đợi (C1).">No such limit appears in the documentation checked for this chapter. The 64 you may remember is Metal's default for uncompleted command buffers on one queue (C1).</Translated> },
      { label: <Translated vi="Kernel nhỏ hơn có occupancy cao hơn, làm chậm các lần sao chép chạy bên cạnh.">Smaller kernels have higher occupancy, which slows the copies running beside them.</Translated>, why: <Translated vi="Một lưới nhỏ có khi còn không lấp đầy GPU; occupancy không tăng chỉ vì kernel ngắn hơn.">A small grid may not even fill the GPU; occupancy does not rise because a kernel is shorter.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Với tổng cố định, bong bóng co lại theo 1/n, trong khi chi phí cố định mỗi thao tác tăng theo n (E3: không nhà cung
            cấp nào công bố nó). Trong trình chạy từng bước, đặt một chi phí và theo dõi chỉ số "số chunk nhanh nhất". Một
            nguyên nhân thứ hai có thể cộng thêm: CUDA ánh xạ các stream lên 8 hàng đợi công việc theo mặc định, và khi các
            stream không liên quan dùng chung một hàng đợi, "a false dependency is created" (A8).
          </p>
        }
      >
        <p>
          With fixed totals the bubble shrinks as 1/n, while the fixed cost per operation grows as n (E3: no vendor publishes
          it). In the stepper, set a cost and watch the "fastest chunk count" read-out. A second cause can add to it: CUDA maps
          streams onto 8 work queues by default, and when unrelated streams share one, "a false dependency is created" (A8).
        </p>
      </Translated>
    ),
    source: 'notes/today/ch23-facts.md E3, A8; the stepper',
    vi: { source: 'notes/today/ch23-facts.md E3, A8; trình chạy từng bước' },
  },
  'ch23/23.3/q-hip-engines': {
    kind: 'choice',
    id: 'ch23/23.3/q-hip-engines',
    prompt: (
      <Translated vi={<>Trên ROCm 10.2, chương trình của bạn đọc được <code>asyncEngineCount</code> = 8. Điều đó cho bạn biết gì về việc các lần sao chép hai chiều chồng lấp nhau?</>}>
        <>On ROCm 10.2 your program reads <code>asyncEngineCount</code> = 8. What does that tell you about copies in both directions overlapping?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Không gì cả: ROCm đặt nó bằng một hằng số runtime, 8, trên mọi thiết bị. Hãy đo thay vì đọc.">Nothing: ROCm sets it to a runtime constant, 8, on every device. Measure instead.</Translated>, correct: true },
      { label: <Translated vi="Tám lần sao chép có thể chạy cùng lúc, bốn mỗi chiều, nếu chúng ở các stream khác nhau.">Eight copies can run at once, four in each direction, if they are in different streams.</Translated>, why: <Translated vi="Giá trị đó giống nhau trên mọi thiết bị AMD, bất kể phần cứng.">The value is the same on every AMD device, whatever its hardware.</Translated> },
      { label: <Translated vi="Các lần sao chép chồng lấp theo cả hai chiều, như trên CUDA khi trường này bằng 2.">Copies overlap in both directions, as they do on CUDA when the field reads 2.</Translated>, why: <Translated vi="Cách đọc đó là của CUDA (A7). Trên ROCm trường này là một hằng số (B3).">That reading is CUDA's (A7). On ROCm the field is a constant (B3).</Translated> },
      { label: <Translated vi="GPU có tám DMA engine, và mỗi stream nhận một engine cho các lần sao chép của nó.">The GPU has eight DMA engines, and each stream gets one of them for its copies.</Translated>, why: <Translated vi="Trường này không nói gì như vậy; nó được đặt từ một hằng số trong runtime.">Nothing in the field says so; it is set from a constant in the runtime.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong runtime ROCm, thuộc tính này được đặt từ <code>kMaxAsyncQueues</code>, một hằng số 8 (B3). Hướng dẫn của AMD chỉ
            nói rằng giá trị lớn hơn 0 nghĩa là thiết bị hỗ trợ truyền đồng thời. Lab 23.1 đo xem các lần sao chép của bạn có chồng
            lấp hay không.
          </p>
        }
      >
        <p>
          In the ROCm runtime the property is set from <code>kMaxAsyncQueues</code>, a constant 8 (B3). AMD's guide says only that
          a value above zero means the device supports concurrent transfers. Lab 23.1 measures whether your copies overlap.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch23-facts.md B3', // i18n-ignore: citation only
  },
  'ch23/23.3/q-metal': {
    kind: 'choice',
    id: 'ch23/23.3/q-metal',
    prompt: (
      <Translated vi={<>Khi chuyển vòng lặp chia chunk sang Metal trên Apple silicon, thứ gì đóng vai copy-in?</>}>
        <>Porting the staged loop to Metal on Apple silicon, what plays the role of the copy-in?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="CPU ghi chunk i + 1 vào một trong K buffer dùng chung trong khi GPU tính chunk i từ một buffer khác.">The CPU writing chunk i + 1 into one of K shared buffers while the GPU computes chunk i from another.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Một blit pass trên một copy engine riêng, chạy bên cạnh compute pass như copy-in của CUDA.">A blit pass on a dedicated copy engine, which runs beside the compute pass the way a CUDA copy-in does.</Translated>, why: <Translated vi="Apple mô tả blit là một loại pass, không phải phần cứng sao chép riêng (C4), và với bộ nhớ dùng chung thì không có gì để sao chép.">Apple documents a blit as a kind of pass, not as separate copy hardware (C4), and with shared memory there is nothing to copy.</Translated> },
      { label: <Translated vi="cudaMemcpyAsync, qua một lớp tương thích ánh xạ lời gọi CUDA sang Metal.">cudaMemcpyAsync, through a compatibility layer that maps CUDA calls onto Metal.</Translated>, why: <Translated vi="Không có lớp nào như vậy.">There is no such layer.</Translated> },
      { label: <Translated vi="Một buffer private mà GPU nạp từ bộ nhớ host trước mỗi compute pass.">A private buffer that the GPU fills from host memory before each compute pass.</Translated>, why: <Translated vi="Vẫn phải có thứ gì đó đặt đầu vào vào đó. Một buffer dùng chung do CPU ghi thì không cần sao chép gì (C4).">Something still has to put the input there. A shared buffer the CPU writes needs no copy at all (C4).</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một buffer dùng chung chính là bộ nhớ của GPU (C4). Cách chồng lấp được Apple ghi trong tài liệu là nhiều bản thể
            buffer đang dùng, với quy tắc "finalize all CPU writes to one buffer instance before you commit a command buffer
            that references it" (C5). Với K = 1 không có gì chồng lấp, và ghi vào một buffer mà command buffer của nó chưa hoàn
            tất là race condition (tranh chấp dữ liệu). Trong chế độ xem Metal của trình chạy từng bước, đặt K bằng 1, rồi 2.
          </p>
        }
      >
        <p>
          A shared buffer is the GPU's memory (C4). Apple's documented overlap is several buffer instances in flight, with the rule
          to "finalize all CPU writes to one buffer instance before you commit a command buffer that references it" (C5). With K
          = 1 nothing overlaps, and writing into a buffer whose command buffer has not completed is a race. In the stepper's
          Metal view, set K to 1, then 2.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch23-facts.md C4, C5', // i18n-ignore: citation only
  },
};

// ---------------------------------------------------------------- the page
export function S233() {
  const cu = useLabFile(SNIPPET.cuda);
  const mm = useLabFile(SNIPPET.metal);
  const metalKernel = useLabFile(SNIPPET.metalKernel);
  const devices = useDevice();
  const [cfg, setCfg] = useState<PipeConfig>(START);
  const stepperRef = useRef<HTMLDivElement>(null);
  const load = (c: PipeConfig) => {
    setCfg(c);
    stepperRef.current?.scrollIntoView({ block: 'start' });
  };
  const pPipe = PRESETS.find((p) => p.id === 'p-pipeline')!.cfg;
  const sPipe = simulate(pPipe);
  const sOne = simulate(ONE_ENGINE);
  const sOneB = simulate({ ...ONE_ENGINE, order: 'breadth' });

  return (
    <>
      <Translated
        vi={
          <p>
            Vòng lặp của phần 23.2 có một khuyết điểm mà sách chỉ ra ngay: tại bất kỳ thời điểm nào chỉ một phần cỗ máy làm
            việc. Trong khi mọi GPU tính, mạng rảnh; trong khi các GPU trao đổi hàng vùng biên (halo), các GPU phần lớn ngồi
            không. Trang này lấy cách chữa của sách cùng ba công cụ nó cần, stream, lần sao chép bất đồng bộ và bộ nhớ ghim (pinned
            memory), và áp dụng chúng ở nơi bạn sẽ dùng đến đầu tiên: trên một GPU, giấu các lần sao chép giữa host và device sau
            một kernel.
          </p>
        }
      >
        <p>
          The loop of section 23.2 has a flaw the book points out at once: at any moment only part of the machine works. While
          every GPU computes, the network is idle; while the GPUs exchange halo rows, the GPUs sit mostly idle. This page takes
          the book's cure and the three tools it needs, streams, asynchronous copies and pinned memory, and applies them where
          you will use them first: on one GPU, hiding the copies between host and device behind a kernel.
        </p>
      </Translated>
      <Source section="23.3" pages="553–554" />

      <h2>
        <Translated vi="Hai chế độ, mỗi chế độ để một nửa phần cứng rảnh">Two modes, each leaving half the hardware idle</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Cách sửa của sách có hai giai đoạn (Fig. 23.12). Đầu tiên, mỗi GPU chỉ tính các hàng rìa của nó, những hàng mà các
            lân cận sẽ đọc làm vùng biên. Rồi nó làm hai việc cùng lúc: gửi các hàng đó, và tính phần còn lại của phần chia.
            Nếu việc gửi tốn ít thời gian hơn phần bên trong, phép trao đổi không tốn thêm gì, điều này cần đủ hàng bên trong
            cho mỗi GPU. Có một bước không thể chồng lấp: phép rút gọn chuẩn, vì nó cần mọi điểm.
          </p>
        }
      >
        <p>
          The book's fix has two stages (Fig. 23.12). First, each GPU computes only its boundary rows, the ones its neighbours
          will read as halos. Then it does two things at once: it sends those rows, and it computes the rest of its partition.
          If the sending takes less time than the interior, the exchange costs nothing extra, which needs enough interior rows
          per GPU. One step cannot be overlapped: the reduction of the norm, which needs every point.
        </p>
      </Translated>
      <Source section="23.3" pages="553–555" figs="Figs. 23.12, 23.13" />
      <Simplification>
        <Translated
          vi={
            <p>
              Trong trình khám phá bên dưới, các hàng rìa được tính bên cạnh phần bên trong mà không làm nó chậm đi, như Fig.
              23.13 vẽ. Phép trao đổi phía dưới bắt đầu sau phép phía trên vì <code>MPI_Sendrecv</code> chặn host (p. 562).
            </p>
          }
        >
          <p>
            In the explorer below, boundary rows are computed beside the interior without slowing it, as Fig. 23.13 draws
            them. The bottom exchange starts after the top one because <code>MPI_Sendrecv</code> blocks the host (p. 562).
          </p>
        </Translated>
      </Simplification>
      <HaloBench />
      <Translated
        vi={
          <p>
            Giờ thu nhỏ bức tranh về một GPU. Một công việc sao chép đầu vào lên device, chạy một kernel rồi sao chép kết quả
            về cũng có cùng khuyết điểm. Trong lúc kernel chạy, phần cứng sao chép ngồi không; trong lúc sao chép, các SM ngồi
            không. Cách chữa cũng vậy: cắt công việc thành nhiều mảnh, và chuyển một mảnh trong khi mảnh khác đang tính. Phần
            còn lại của trang xây dựng điều đó, từng công cụ một.
          </p>
        }
      >
        <p>
          Now shrink the picture to one GPU. A job that copies its input to the device, runs a kernel and copies the result back
          has the same flaw. During the kernel the copy hardware sits idle; during the copies the SMs do. The cure is the same
          too: cut the work into pieces, and move one piece while another computes. The rest of the page builds that, one tool
          at a time.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chồng lấp lưu lượng mạng. Phiên bản một GPU, sao chép qua đường kết nối host-device, là của NVIDIA (Best
              Practices Guide 10.1.2, ch23-facts A7); mô hình và các đoạn mã ở đây là của khóa học này.
            </p>
          }
        >
          <p>
            The book overlaps network traffic. The one-GPU version, copies over the host-device link, is NVIDIA's (Best
            Practices Guide 10.1.2, ch23-facts A7); the model and the snippets here are ours.
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Stream: những hàng đợi có thứ tự, có thể chạy song song">Streams: ordered queues that may run side by side</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một stream là một hàng đợi các thao tác GPU: các lần khởi chạy kernel, các lần sao chép và vài loại khác. Bên trong một
            stream, chúng chạy theo thứ tự bạn đưa vào. Giữa hai stream thì không có thứ tự nào cả, và GPU có thể chạy công
            việc của chúng cùng lúc.
          </p>
        }
      >
        <p>
          A stream is a queue of GPU operations: kernel launches, copies and a few others. Inside one stream they run in the
          order you inserted them. Between two streams there is no order at all, and the GPU may run their work at the same time.
        </p>
      </Translated>
      <Source section="23.3" pages="555–556" />
      <Translated
        vi={
          <p>
            Bạn chọn stream bằng tham số khởi chạy thứ tư. Hai tham số đầu, grid và block, là của chương 2; tham số thứ ba,
            kích thước bộ nhớ chia sẻ động, là của chương 5; tham số thứ tư chỉ định stream:{' '}
            <code>kernel&lt;&lt;&lt;grid, block, 0, stream&gt;&gt;&gt;(...)</code>. Bỏ nó đi thì lần khởi chạy vào default
            stream.
          </p>
        }
      >
        <p>
          You choose the stream with the fourth launch parameter. The first two, grid and block, are chapter 2's; the third, the
          dynamic shared memory size, is chapter 5's; the fourth names the stream:{' '}
          <code>kernel&lt;&lt;&lt;grid, block, 0, stream&gt;&gt;&gt;(...)</code>. Leave it out and the launch goes into the
          default stream.
        </p>
      </Translated>
      <Source section="23.3" pages="556" />
      <Translated
        vi={
          <p>
            Và lần khởi chạy là bất đồng bộ: khi lời gọi trả về, kernel đã được đưa vào stream của nó. Nó có thể chưa xong,
            thậm chí chưa bắt đầu; nó chạy khi mọi thứ trước nó trong stream đó đã xong. Bạn đã thấy điều này cho một stream
            trên <a href={href('ch02', '2.2')}>dòng thời gian của chương 2</a>. Với nhiều stream, host có thể xếp việc cho tất
            cả chúng rồi làm việc khác.
          </p>
        }
      >
        <p>
          And the launch is asynchronous: when the call returns, the kernel has been inserted into its stream. It may not have
          finished, or even started; it runs once everything ahead of it in that stream is done. You saw this for one stream on{' '}
          <a href={href('ch02', '2.2')}>chapter 2's timeline</a>. With several streams, the host can queue work for all of them
          and get on with something else.
        </p>
      </Translated>
      <Today checked={`${CHECKED} (ch23-facts A3, A9; ch22-facts A2)`}>{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Default stream kiểu cũ (legacy) "synchronizes with all other streams" trừ các stream non-blocking. Tránh nó bằng{' '}
              <code>cudaStreamNonBlocking</code> hoặc <code>--default-stream per-thread</code>. Các kernel chỉ chồng lấp trên
              các stream không phải default, khi không có phụ thuộc và còn tài nguyên trống.
            </p>
          }
        >
          <p>
            The legacy default stream "synchronizes with all other streams" except non-blocking ones. Avoid it with{' '}
            <code>cudaStreamNonBlocking</code> or <code>--default-stream per-thread</code>. Kernels overlap only on non-default
            streams, with no dependency and free resources.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Chương 22 đã gửi bạn đến đây để học stream như một cách chạy các tác vụ nhỏ song song (
            <a href={href('ch22', '22.3')}>§22.3</a>). Phần còn lại của trang này dùng cùng công cụ đó cho các lần sao chép.
          </p>
        }
      >
        <p>
          Chapter 22 sent you here for streams as a way to run small tasks side by side (<a href={href('ch22', '22.3')}>§22.3</a>).
          The rest of this page uses the same tool on copies.
        </p>
      </Translated>
      <Ask q={questionsS233['ch23/23.3/q-onestream']} />

      <h2>
        <Translated vi="Lần sao chép bất đồng bộ cần bộ nhớ ghim">Asynchronous copies need pinned memory</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lần sao chép cũng có dạng bất đồng bộ: <code>cudaMemcpyAsync</code> nhận một stream và trả về ngay. Có một điểm vướng, và
            sách giải thích nó từ dưới lên.
          </p>
        }
      >
        <p>
          Copies have an asynchronous form too: <code>cudaMemcpyAsync</code> takes a stream and returns at once. There is a catch,
          and the book explains it from the bottom up.
        </p>
      </Translated>
      <Source section="23.3" pages="560, 562–563" />
      <Translated
        vi={
          <p>
            Tiến trình của bạn thấy một không gian địa chỉ ảo lớn, liền mạch, được cắt thành các trang. Hệ điều hành ánh xạ mỗi
            trang lên một khung bộ nhớ vật lý, và nó có thể di chuyển một trang, hoặc ghi nó ra đĩa, bất cứ lúc nào: đó là cách
            nó giữ các tiến trình tách biệt và phục vụ nhiều bộ nhớ hơn lượng nó có. Bộ nhớ từ <code>malloc</code> là như vậy.
            Nó là bộ nhớ phân trang (pageable).
          </p>
        }
      >
        <p>
          Your process sees a large, contiguous virtual address space, cut into pages. The operating system maps each page onto a
          frame of physical memory, and it may move a page, or write it out to disk, at any time: that is how it keeps processes
          apart and serves more memory than it has. Memory from <code>malloc</code> is like that. It is pageable.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Phần cứng sao chép không thấy địa chỉ ảo. Một DMA engine chuyển dữ liệu theo địa chỉ vật lý, không cần CPU. Nếu hệ
            điều hành di chuyển một trang giữa chừng một lần sao chép, engine sẽ đọc hoặc ghi vào một khung (frame) nay đang chứa thứ khác.
            Vì vậy runtime không trao bộ nhớ phân trang cho engine. Thay vào đó nó sao chép hai bước: từ buffer của bạn vào một
            buffer trung chuyển mà hệ điều hành sẽ không di chuyển, rồi bằng DMA sang device, và ngược lại cho chiều về. Lần sao chép
            thêm đó tốn thời gian, và vì CPU làm nó, lời gọi không thể trả về cho đến khi nó xong.
          </p>
        }
      >
        <p>
          The copy hardware does not see virtual addresses. A DMA engine moves data by physical address, without the CPU. If the
          operating system moved a page in the middle of a copy, the engine would read or write a frame that now holds something
          else. So the runtime does not hand pageable memory to the engine. It copies in two steps instead: from your buffer into
          a staging buffer the operating system will not move, then by DMA to the device, and the reverse for the way back. The
          extra copy costs time, and since the CPU does it, the call cannot return until it is finished.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Bộ nhớ mà hệ điều hành sẽ không di chuyển được gọi là bộ nhớ ghim, hay page-locked.{' '}
            <code>cudaMallocHost(&amp;h, bytes)</code> cấp phát nó. Engine đọc nó trực tiếp, và luồng host được tự do ngay khi
            lần sao chép được xếp hàng.
          </p>
        }
      >
        <p>
          Memory the operating system will not move is called pinned, or page-locked. <code>cudaMallocHost(&amp;h, bytes)</code>{' '}
          allocates it. The engine reads it directly, and the host thread is free as soon as the copy is queued.
        </p>
      </Translated>
      <Source section="23.3" pages="562–563" />
      <Simplification>
        <Translated
          vi={
            <p>
              Cảnh bên dưới trung chuyển lần sao chép thành từng mảnh qua hai buffer và trả về sau mảnh cuối. NVIDIA không ghi trong
              tài liệu cả các buffer lẫn chính xác lúc lời gọi trả về ("might be synchronous", A6).
            </p>
          }
        >
          <p>
            The scene below stages the copy in pieces through two buffers and returns after the last piece. NVIDIA documents
            neither the buffers nor exactly when the call returns ("might be synchronous", A6).
          </p>
        </Translated>
      </Simplification>
      <PinnedScene />
      <Translated
        vi={
          <p>
            Một hiệu chỉnh cho sách. Sách nói <code>cudaMemcpyAsync</code> <em>đòi hỏi</em> bộ nhớ ghim, và Best Practices
            Guide của NVIDIA cũng dùng chữ đó. Tài liệu tham khảo runtime chính xác hơn: từ bộ nhớ phân trang lời gọi vẫn hoạt
            động, nhưng nó "will revert to a synchronous behavior which will not overlap with other work". Lỗi này im lặng: kết
            quả đúng, không chồng lấp.
          </p>
        }
      >
        <p>
          One correction to the book. It says <code>cudaMemcpyAsync</code> <em>requires</em> pinned memory, and NVIDIA's Best
          Practices Guide uses the same word. The runtime reference is more precise: from pageable memory the call still works,
          but it "will revert to a synchronous behavior which will not overlap with other work". The failure is silent: right
          results, no overlap.
        </p>
      </Translated>
      <Today checked={`${CHECKED} (ch23-facts A6)`}>{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Trên các hệ thống mà GPU dùng bảng trang của host (HMM hoặc ATS), NVIDIA lưu ý rằng <code>cudaMallocHost</code>{' '}
              "may not page-lock the allocated memory".
            </p>
          }
        >
          <p>
            On systems where the GPU uses the host's page tables (HMM or ATS), NVIDIA notes that <code>cudaMallocHost</code> "may
            not page-lock the allocated memory".
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            <a href={href('appC', 'C.2')}>Phụ lục C, §C.2</a> đưa bộ nhớ ghim đi thêm một bước: bộ nhớ host mà GPU đọc tại chỗ,
            và <code>pin_memory</code> của PyTorch làm gì.
          </p>
        }
      >
        <p>
          <a href={href('appC', 'C.2')}>Appendix C, §C.2</a> takes pinned memory one step further: host memory the GPU reads in
          place, and what PyTorch's <code>pin_memory</code> does.
        </p>
      </Translated>
      <Ask q={questionsS233['ch23/23.3/q-pageable']} />

      <h2>
        <Translated vi="Cắt công việc thành chunk">Cut the job into chunks</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Giờ các mảnh ghép khớp với nhau. Chia dữ liệu thành n chunk. Với mỗi chunk, xếp ba thao tác vào một stream: sao chép
            đầu vào của nó lên, chạy kernel trên nó, sao chép kết quả về. Rải các chunk ra S stream. Trong một stream, ba bước
            của một chunk giữ đúng thứ tự; giữa các stream, lần sao chép của một chunk có thể chạy trong khi kernel của chunk khác
            chạy. Đây là vòng lặp viết theo hai cách, phát lần lượt ba thao tác của từng chunk (theo chiều sâu) hoặc mọi copy-in
            trước (theo chiều rộng), và cùng công việc đó trên Metal.
          </p>
        }
      >
        <p>
          Now the pieces fit together. Split the data into n chunks. For each chunk, queue three operations in one stream: copy
          its input in, run the kernel on it, copy its result out. Spread the chunks over S streams. Inside a stream a chunk's
          three steps stay in order; across streams, one chunk's copy can run while another chunk's kernel does. Here is the loop
          written two ways, issuing each chunk's three operations in turn (depth first) or every copy-in first (breadth first),
          and the same job on Metal.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Vòng lặp chia chunk là của NVIDIA (Best Practices 10.1.2, A7); các tệp này là của khóa học này. Chúng chia dữ liệu
              như lab 23.1 làm, nên những gì bạn đọc ở đây là những gì bạn sẽ đo ở đó.
            </p>
          }
        >
          <p>
            The staged loop is NVIDIA's (Best Practices 10.1.2, A7); these files are ours. They split the data as lab 23.1 does,
            so what you read here is what you will measure there.
          </p>
        </Translated>
      </Beyond>
      {cu && mm && metalKernel ? (
        <Code
          tabs={[
            { label: 'CUDA', code: cu, status: STATUS_CU },
            { label: tr('Metal host', 'Mã host Metal'), code: mm, status: STATUS_METAL },
            { label: tr('Metal kernel', 'Kernel Metal'), code: metalKernel, status: STATUS_METAL },
          ]}
          caption={
            <Translated
              vi={
                <>
                  Từ <code>labs/ch23/snippets/</code>. Header HIP của khóa học ánh xạ mọi lời gọi CUDA trong tệp CUDA; chưa có bản
                  build HIP nào của nó được chạy. Các phép kiểm tra cho ra những trạng thái này nằm trong{' '}
                  <code>labs/ch23/snippets/check.py</code>.
                </>
              }
            >
              <>
                From <code>labs/ch23/snippets/</code>. The course's HIP header maps every CUDA call in the CUDA file; no HIP
                build of it has run yet. The checks that give these statuses are in <code>labs/ch23/snippets/check.py</code>.
              </>
            </Translated>
          }
        />
      ) : null}
      <Predict
        id="ch23/23.3/p-pipeline"
        prompt={
          <Translated
            vi={
              <>
                8 chunk bằng nhau. Với mỗi chunk, copy-in tốn 1, kernel 2, copy-out 1 (đơn vị, không phải mili giây). Mỗi chunk
                một stream, hai copy engine. Làm gộp một khối thì tốn 32. Công việc chạy theo pipeline tốn bao lâu?
              </>
            }
          >
            <>
              8 equal chunks. For each chunk the copy-in takes 1, the kernel 2, the copy-out 1 (units, not milliseconds). One
              stream per chunk, two copy engines. Done in one piece it takes 32. How long does the pipelined job take?
            </>
          </Translated>
        }
        unit={tr('units', 'đơn vị')}
      >
        {(v) => (
          <>
            <Translated
              vi={
                <p>
                  Nạp trường hợp này vào trình chạy từng bước và đọc tổng của nó. Mô hình cho <b>{fmtN(sPipe.total)}</b>
                  {v === sPipe.total ? ', đúng như bạn nói' : ''}: chunk đầu tiên cần 1 + 2 + 1 = 4 để đi hết, và sau đó làn
                  kernel, làn chậm nhất, xong thêm một chunk mỗi 2, nên 4 + 7 × 2.
                </p>
              }
            >
              <p>
                Load the case into the stepper and read its total. The model gives <b>{fmtN(sPipe.total)}</b>
                {v === sPipe.total ? ', as you said' : ''}: the first chunk needs 1 + 2 + 1 = 4 to get through, and after that
                the kernel lane, the slowest, finishes one more chunk every 2, so 4 + 7 × 2.
              </p>
            </Translated>
            <button onClick={() => load(pPipe)} data-load="p-pipeline"><Translated vi="Nạp vào trình chạy từng bước">Load it into the stepper</Translated></button>
          </>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Trình chạy từng bước chạy vòng lặp này trên một mô hình các làn của GPU. Nó bắt đầu với một trường hợp riêng, không
            phải trường hợp bạn vừa dự đoán; hãy chạy từng bước qua nó trước.
          </p>
        }
      >
        <p>
          The stepper runs this loop on a model of the GPU's lanes. It starts on a case of its own, not the one you just
          predicted; step through it first.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi lúc một kernel. Mỗi làn phục vụ các thao tác của nó theo thứ tự phát lệnh, điều NVIDIA không ghi trong tài
              liệu cho trường hợp một copy engine (E2). Một lần sao chép từ bộ nhớ phân trang chờ stream của nó, rồi giữ host (trường hợp xấu
              nhất được phép). Các stream được ánh xạ xoay vòng lên các hàng đợi theo thứ tự. Không có cache, không chia sẻ băng
              thông. Không có đơn vị.
            </p>
          }
        >
          <p>
            One kernel at a time. Each lane serves its operations in issue order, which NVIDIA does not document for one copy
            engine (E2). A pageable copy waits for its stream, then holds the host (the worst case allowed). Streams map
            round-robin onto in-order queues. No caches, no shared bandwidth. Unitless.
          </p>
        </Translated>
      </Simplification>
      <div ref={stepperRef} style={{ scrollMarginTop: '1rem' }} className="wide">
        <Stepper cfg={cfg} setCfg={setCfg} devices={devices} />
      </div>
      <Translated vi={<p>Những điều cần tìm trong trình chạy từng bước:</p>}>
        <p>Things to find in the stepper:</p>
      </Translated>
      <Translated
        vi={
          <ol>
            <li>Đặt S bằng 1. Dù bạn cắt bao nhiêu chunk, không có gì chồng lấp. Vì sao?</li>
            <li>
              Nạp "các giai đoạn bằng nhau" và thử S = 2, rồi S = 3. Xem copy-in nào phải chờ, và phần thuyết
              minh nói nó chờ gì.
            </li>
            <li>Đặt chi phí mỗi thao tác là 0.25 và đổi n từ 1 đến 64. Số chunk nhanh nhất ở đâu, và vì sao nó tồn tại?</li>
            <li>Chuyển bộ nhớ host sang pageable. Giờ làn nào bận, và chuyện gì xảy ra với chồng lấp?</li>
          </ol>
        }
      >
        <ol>
          <li>Set S to 1. However many chunks you cut, nothing overlaps. Why not?</li>
          <li>
            Load "equal stages" and try S = 2, then S = 3. Watch which copy-in waits, and what the narration says it waits for.
          </li>
          <li>Set a per-operation cost of 0.25 and move n from 1 to 64. Where is the fastest chunk count, and why does it exist?</li>
          <li>Switch the host memory to pageable. Which lane is busy now, and what happened to the overlap?</li>
        </ol>
      </Translated>
      <Translated
        vi={
          <p>
            Tổng mà trình chạy từng bước báo là a + b + c + (n − 1)·max(a, b, c) với thời gian mỗi chunk, miễn là mỗi chiều có
            một copy engine và đủ stream: chunk đầu tiên đi qua cả ba làn, rồi làn chậm nhất xong một chunk mỗi bước. Thời
            gian rảnh ở hai đầu, tức bong bóng, là a + b + c − max(a, b, c) mỗi chunk; chunk nhỏ hơn thu nhỏ nó. Dạng đóng đó, và
            điều kiện S nhân giai đoạn chậm nhất phải phủ được cả một chunk, là của khóa học này (ch23-facts E1), và các test
            của mô hình kiểm tra chúng với lịch chạy.
          </p>
        }
      >
        <p>
          The total the stepper reports is a + b + c + (n − 1)·max(a, b, c) for per-chunk times, as long as there is a copy engine
          per direction and enough streams: the first chunk passes through all three lanes, and then the slowest lane finishes one
          chunk per step. The idle time at the two ends, the bubble, is a + b + c − max(a, b, c) per chunk; smaller chunks shrink
          it. That closed form, and the condition that S times the slowest stage covers a whole chunk, are ours (ch23-facts
          E1), and the model's tests check them against the schedule.
        </p>
      </Translated>
      <Today checked={`${CHECKED} (ch23-facts A7)`}>{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              NVIDIA in vòng lặp này kèm một ước lượng thô: khi thời gian kernel tE vượt thời gian truyền tT, khoảng tE +
              tT/nStreams khi chia chunk, so với tE + tT khi làm một lần. Ví dụ của họ chỉ sao chép một chiều; thiết lập sẵn
              "ví dụ Best Practices" tái hiện nó.
            </p>
          }
        >
          <p>
            NVIDIA prints this loop with a rough estimate: when the kernel time tE exceeds the transfer time tT, about tE +
            tT/nStreams staged, against tE + tT in one go. Its example copies one way only; the "Best Practices example" preset
            reproduces it.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Có bao nhiêu lần sao chép chạy cùng lúc phụ thuộc vào số copy engine. Một GPU có một copy engine có thể chạy một lần
            truyền bất đồng bộ bên cạnh các kernel; có hai thì mỗi chiều một lần cùng lúc. CUDA báo con số này là{' '}
            <code>asyncEngineCount</code>.
          </p>
        }
      >
        <p>
          How many copies can move at once depends on copy engines. A GPU with one copy engine can run one asynchronous transfer
          alongside kernels; with two, one in each direction at the same time. CUDA reports the number as{' '}
          <code>asyncEngineCount</code>.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Copy engine không có trong sách; Best Practices Guide 10.1.2 của NVIDIA mô tả chúng (ch23-facts A7).</p>}>
          <p>Copy engines are not in the book; NVIDIA's Best Practices Guide 10.1.2 describes them (ch23-facts A7).</p>
        </Translated>
      </Beyond>
      <Predict
        id="ch23/23.3/p-oneengine"
        prompt={
          <Translated
            vi={
              <>
                Trước khi bạn chuyển trình chạy từng bước sang một copy engine: 4 chunk gồm 1, 2, 1; mỗi chunk một stream; một
                copy engine dùng chung cho cả hai chiều; các thao tác phát lệnh từng chunk một. Làm gộp một khối thì tốn 16. Giờ
                tốn bao lâu?
              </>
            }
          >
            <>
              Before you switch the stepper to one copy engine: 4 chunks of 1, 2, 1; one stream each; one copy engine shared by
              both directions; operations issued chunk by chunk. Done in one piece it takes 16. How long now?
            </>
          </Translated>
        }
        unit={tr('units', 'đơn vị')}
      >
        {() => (
          <>
            <Translated
              vi={
                <p>
                  <b>{fmtN(sOne.total)}</b>: không lợi gì cả. Copy-out của chunk 0 nằm trước copy-in của chunk 1 trong hàng đợi
                  của engine duy nhất, và nó không thể bắt đầu trước khi kernel 0 xong, nên copy-in của chunk 1 cũng phải chờ.
                  Phát mọi copy-in trước thì cùng công việc đó tốn <b>{fmtN(sOneB.total)}</b>.
                </p>
              }
            >
              <p>
                <b>{fmtN(sOne.total)}</b>: no gain at all. Chunk 0's copy-out sits ahead of chunk 1's copy-in in the one engine's
                queue, and it cannot start before kernel 0 has finished, so chunk 1's copy-in waits as well. Issue every copy-in
                first and the same work takes <b>{fmtN(sOneB.total)}</b>.
              </p>
            </Translated>
            <div className="row">
              <button onClick={() => load(ONE_ENGINE)} data-load="one-depth"><Translated vi="Nạp: từng chunk một">Load: chunk by chunk</Translated></button>
              <button onClick={() => load({ ...ONE_ENGINE, order: 'breadth' })} data-load="one-breadth"><Translated vi="Nạp: mọi copy-in trước">Load: all copy-ins first</Translated></button>
            </div>
          </>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Vậy với một engine, thứ tự phát lệnh quyết định kết quả. Điều đó dựa trên một giả định mà mô hình nêu rõ: engine
            phục vụ các lần sao chép theo thứ tự chúng được phát. NVIDIA không ghi thứ tự nào cho trường hợp này, nên hãy coi nó là
            một giả thuyết cần kiểm tra trên một thiết bị có <code>asyncEngineCount</code> bằng 1. (Các trường hợp một engine là
            phép tính của khóa học này, ch23-facts E2.)
          </p>
        }
      >
        <p>
          So with one engine the issue order decides the result. That rests on an assumption the model states: the engine serves
          copies in the order they were issued. NVIDIA documents no order for this case, so treat it as a hypothesis to test on a
          device whose <code>asyncEngineCount</code> is 1. (The one-engine cases are our arithmetic, ch23-facts E2.)
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Stream cũng không miễn phí. CUDA ánh xạ chúng lên các hàng đợi công việc phần cứng, mặc định 8 (
            <code>CUDA_DEVICE_MAX_CONNECTIONS</code>, 1 đến 32). Khi hai stream không liên quan rơi vào cùng một hàng đợi, "a
            false dependency is created which can lead to GPU work serialization". HIP mặc định dùng tối đa 4 hàng đợi phần
            cứng mỗi thiết bị và chia chúng xoay vòng. Để thấy điều đó, nạp "các giai đoạn bằng nhau", đặt 16 chunk và 16 stream, phát mọi
            copy-in trước, và chuyển số hàng đợi từ không giới hạn sang 8. Giờ copy-in nào phải chờ, và chờ gì?{' '}
            <button className="quiet" onClick={() => load(FALSE_DEP)} data-load="false-dep">Nạp trường hợp đó</button>
          </p>
        }
      >
        <p>
          Streams are not free either. CUDA maps them onto hardware work queues, 8 by default (<code>CUDA_DEVICE_MAX_CONNECTIONS</code>,
          1 to 32). When two unrelated streams land in one queue, "a false dependency is created which can lead to GPU work
          serialization". HIP uses at most 4 hardware queues per device by default and shares them round-robin. To see it,
          load "equal stages", set 16 chunks and 16 streams, issue all copy-ins first, and switch the queues from unlimited
          to 8. Which copy-in waits now, and for what?{' '}
          <button className="quiet" onClick={() => load(FALSE_DEP)} data-load="false-dep">Load that case</button>
        </p>
      </Translated>
      <Today checked={`${CHECKED} (ch23-facts A8; ch22-facts B1)`}>{/* i18n-ignore: date and citation */}
        <Translated vi={<p>Cả hai giới hạn đều không có trong sách. Giới hạn của CUDA là một biến môi trường; của HIP là <code>GPU_MAX_HW_QUEUES</code>.</p>}>
          <p>Neither limit is in the book. CUDA's is an environment variable; HIP's is <code>GPU_MAX_HW_QUEUES</code>.</p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Và mỗi thao tác có một chi phí cố định, để thiết lập một lần sao chép hay khởi chạy một kernel, mà không nhà cung cấp nào
            công bố. Cắt quá nhỏ thì chi phí vượt quá bong bóng bạn đang thu nhỏ. Vì lý do đó, chi phí trong trình chạy từng bước
            là một núm vặn, không phải một con số; lab 23.1 tìm số chunk tốt nhất cho thiết bị của bạn bằng cách đo.
          </p>
        }
      >
        <p>
          And each operation has a fixed cost, to set up a copy or launch a kernel, that no vendor publishes. Cut too fine and the
          costs outgrow the bubble you were shrinking. The stepper's cost is a dial for that reason, not a figure; lab 23.1 finds
          your device's best chunk count by measuring.
        </p>
      </Translated>
      <Ask q={questionsS233['ch23/23.3/q-bottleneck']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS233['ch23/23.3/q-twostreams']} />
      <Ask q={questionsS233['ch23/23.3/q-overhead']} />
      <Translated
        vi={
          <p>
            <a href={href('ch23', 'labs')}>Lab 23.1, Giấu các lần sao chép sau kernel</a>, chạy vòng lặp này trên GPU của bạn với buffer ghim và buffer phân trang, cả hai thứ tự phát lệnh và vài số stream. Khi bạn đã chạy nó, trình chạy từng bước cung cấp các
            phần đo được trên thiết bị của bạn làm một thiết lập sẵn và vẽ những gì lab đo được bên cạnh mô hình.
          </p>
        }
      >
        <p>
          <a href={href('ch23', 'labs')}>Lab 23.1, Hide the copies behind the kernel</a>, runs this loop on your GPU with pinned
          and pageable buffers, both issue orders and several stream counts. Once you have run it, the stepper offers your device's
          measured parts as a preset and plots what the lab measured beside the model.
        </p>
      </Translated>

      <h2>
        <Translated vi="Trên HIP và Metal">On HIP and Metal</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trên HIP, cùng mã nguồn đi qua header của khóa học, header này ánh xạ mọi lời gọi mà trang này dùng (cho đến nay
            unverified trên phần cứng AMD). Hai lần đổi tên không chỉ là đổi tiền tố: <code>cudaMallocHost</code> thành{' '}
            <code>hipHostMalloc</code>, và <code>cudaFreeHost</code> thành <code>hipHostFree</code>. Cái bẫy giống như trên
            CUDA: header của AMD nói rằng nếu một buffer không được ghim, "the memory copy will be performed synchronously". Một
            khác biệt cần biết trước khi đo: ROCm đặt <code>asyncEngineCount</code> bằng hằng số 8 trên mọi thiết bị, nên nó
            không đếm phần cứng sao chép nào. Công tắc engine của trình chạy từng bước là một ý tưởng của CUDA; trên AMD, hãy đo.
          </p>
        }
      >
        <p>
          On HIP the same source goes through the course's header, which maps every call this page uses (unverified on AMD
          hardware so far). Two renames are not a prefix swap: <code>cudaMallocHost</code> becomes <code>hipHostMalloc</code>, and{' '}
          <code>cudaFreeHost</code> becomes <code>hipHostFree</code>. The trap is the same as on CUDA: AMD's header says
          that if a buffer is not pinned, "the memory copy will be performed synchronously". One difference to know before you
          measure: ROCm sets <code>asyncEngineCount</code> to the constant 8 on every device, so it counts no copy hardware. The
          stepper's engine switch is a CUDA idea; on AMD, measure.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ dùng CUDA. Các thông tin về HIP và Metal ở đây được kiểm tra với nguồn của nhà cung cấp trong
              notes/today/ch23-facts.md (B3, B4, C1, C4, C5) và infra-facts.md (B8).
            </p>
          }
        >
          <p>
            The book uses CUDA only. HIP and Metal facts here are checked against vendor sources in notes/today/ch23-facts.md (B3,
            B4, C1, C4, C5) and infra-facts.md (B8).
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS233['ch23/23.3/q-hip-engines']} />
      <Translated
        vi={
          <p>
            Trên Apple silicon không có gì để sao chép: một buffer dùng chung chính là bộ nhớ của GPU. Thứ còn lại của pipeline
            là việc của CPU, ghi đầu vào của chunk tiếp theo vào một buffer và đọc đầu ra của một chunk đã xong. Cách Apple ghi
            trong tài liệu để chồng lấp điều đó với GPU là nhiều bản thể của một buffer đang dùng: CPU nạp một bản trong khi GPU
            đọc bản khác, một semaphore ngăn CPU chạy vượt lên, và một completion handler giải phóng một bản thể khi command
            buffer của nó xong. Quy tắc giữ cho nó đúng: "finalize all CPU writes to one buffer instance before you commit a
            command buffer that references it".
          </p>
        }
      >
        <p>
          On Apple silicon there is nothing to copy: a shared buffer is the GPU's memory. What is left of the pipeline is CPU
          work, writing the next chunk's input into a buffer and reading a finished chunk's output. Apple's documented way to
          overlap that with the GPU is several instances of a buffer in flight: the CPU fills one while the GPU reads another, a
          semaphore stops the CPU from running ahead, and a completion handler releases an instance when its command buffer is
          done. The rule that keeps it correct: "finalize all CPU writes to one buffer instance before you commit a command buffer
          that references it".
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Chuyển trình chạy từng bước sang chế độ xem Metal. Các làn trở thành "CPU: rút ra, rồi nạp vào" và
            "GPU", và S trở thành K, số buffer đang dùng. Với K = 1 không có gì chồng lấp; với 2, các lần nạp và rút của CPU có
            thể chạy bên dưới các kernel của GPU. Tệp Metal của chúng tôi chờ command buffer cuối của một buffer bằng{' '}
            <code>waitUntilCompleted</code> thay vì semaphore và completion handler của Apple; cách nào thì CPU cũng chỉ bị chặn
            khi buffer nó cần vẫn đang được dùng.
          </p>
        }
      >
        <p>
          Switch the stepper to its Metal view. The lanes become "CPU: drain, then fill" and "GPU", and S becomes K, the buffers
          in flight. With K = 1 nothing overlaps; with 2 the CPU's fills and drains can run under the GPU's kernels. Our Metal
          file waits for a buffer's last command buffer with <code>waitUntilCompleted</code> instead of Apple's semaphore and
          completion handler; either way the CPU blocks only when the buffer it needs is still in use.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Hãy để ý chế độ xem Metal đã mất gì: một làn. Một luồng CPU làm mọi lần nạp và mọi lần rút, nên chồng lấp tối đa chỉ
            có thể giấu làn ngắn hơn trong hai làn, nạp + rút trên CPU và kernel trên GPU. Chỉ số hiển thị cho giới hạn, n ·
            max(nạp + rút, kernel) mỗi chunk. Trừ khi kernel dài, làn của CPU là làn dài hơn: nó chuyển dữ liệu hai lần. Khi hai
            làn bằng nhau, trường hợp tốt nhất, chồng lấp giảm tối đa một nửa thời gian tuần tự; lab 23.1 đo xem một máy Mac đạt
            gần đến đâu.
          </p>
        }
      >
        <p>
          Notice what the Metal view has lost: a lane. One CPU thread does every fill and every drain, so the most overlap can
          do is hide the shorter of two lanes, fill + drain on the CPU and the kernel on the GPU. The read-out gives the bound,
          n · max(fill + drain, kernel) per chunk. Unless the kernel is long, the CPU&rsquo;s lane is the longer one: it moves
          the data twice. With the two equal, the best case, overlap at most halves the serial time; lab 23.1 measures how close
          a Mac gets.
        </p>
      </Translated>
      <Today checked={`${CHECKED} (ch23-facts C1, C4)`}>{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Hai command queue "can" được lập lịch đồng thời: chữ của Apple, không phải một lời hứa. Không có copy engine nào
              được ghi trong tài liệu. Một hàng đợi từ <code>newCommandQueue</code> cho phép tối đa 64 command buffer chưa hoàn
              tất; <code>newCommandQueueWithMaxCommandBufferCount:</code> thay đổi điều đó. Ở đây K luôn thấp hơn nhiều.
            </p>
          }
        >
          <p>
            Two command queues "can" be scheduled concurrently: Apple's word, not a promise. No copy engine is documented. A queue
            from <code>newCommandQueue</code> allows at most 64 uncompleted command buffers;{' '}
            <code>newCommandQueueWithMaxCommandBufferCount:</code> changes that. K stays far below it here.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS233['ch23/23.3/q-metal']} />

      <h2>
        <Translated vi="Nơi bạn sẽ gặp lại điều này">Where you will meet this again</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Song song pipeline (pipeline parallelism) trong các máy chủ suy luận chính là pipeline này với các nhóm lớp làm các giai đoạn: trong khi
            một giai đoạn làm việc trên một batch, activation của batch trước chuyển sang giai đoạn tiếp theo. Chế độ nhiều GPU
            mặc định của llama.cpp, <code>--split-mode layer</code>, được mô tả là "pipelined": bộ lập lịch của nó giữ tối đa 4
            bản sao của mỗi đầu vào, ghi một event sau mỗi lần chia và bắt lần chia tiếp theo chờ nó. Tài liệu pipelining của
            PyTorch gọi khoảng rảnh lúc đầu là "pipeline startup bubble", phần nạp (fill) bạn đã xem trong trình chạy từng bước.
            vLLM dùng song song pipeline giữa các node.
          </p>
        }
      >
        <p>
          Pipeline parallelism in inference servers is this pipeline with groups of layers as the stages: while one stage works
          on a batch, the activations of the previous batch move on to the next stage. llama.cpp's default multi-GPU mode,{' '}
          <code>--split-mode layer</code>, is described as "pipelined": its scheduler keeps up to 4 copies of each input, records
          an event after each split and makes the next one wait on it. PyTorch's pipelining documentation calls the idle start
          the "pipeline startup bubble", the fill you watched in the stepper. vLLM uses pipeline parallelism across nodes.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Việc đối chiếu này là của khóa học này (ch23-facts D1, D5, D7). Trong llama.cpp, backend CUDA hiện thực việc ghi
              và chờ đó bằng <code>cudaEventRecord</code> và <code>cudaStreamWaitEvent</code>: các công cụ của trang{' '}
              <a href={href('ch23', '23.4')}>23.3–23.5</a>.
            </p>
          }
        >
          <p>
            The mapping is ours (ch23-facts D1, D5, D7). In llama.cpp, the CUDA backend implements that record and wait as{' '}
            <code>cudaEventRecord</code> and <code>cudaStreamWaitEvent</code>: the tools of page{' '}
            <a href={href('ch23', '23.4')}>23.3–23.5</a>.
          </p>
        </Translated>
      </Beyond>
    </>
  );
}
