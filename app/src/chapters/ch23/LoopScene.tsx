import { useId, type ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { DEFAULTS, simulate, type Lane, type LoopResult, type Seg, type Version } from '../../sim/haloloop';
import { Translated, tr } from '../../lib/i18n';

// The timeline drawing shared by the narrated scene "One iteration with three streams" and the
// explorer "Who waits" on page 23.3–23.5 (id 23.4). Everything it draws comes from src/sim/haloloop.ts.
// The scene's five beats (content/ch23/narration/loop-N.txt) draw the model at DEFAULTS: four
// on Fig. 23.14, and one that sets its totals beside the NCCL and NVSHMEM loops'. The chapter is
// condensed, so those two loops get no beat of their own.

export const X0 = 124;
export const X1 = 748;
export const VIEW_W = 760;
const LANE_H = 22;
const ROW = 30;
const SLOT_H = 4;
const SLOTS_BLOCK = 8 * (SLOT_H + 1) + 22;
const AXIS = 24;

export function laneName(l: Lane, v: Version): string {
  if (l === 'internal' && v === 'mpi') return tr('default stream', 'stream mặc định');
  if (l === 'internal' && v === 'nvshmem') return tr('stream', 'stream');
  return tr(
    { host: 'host', top: 'top stream', bottom: 'bottom stream', internal: 'internal stream', linkTop: 'top exchange', linkBottom: 'bottom exchange' }[l],
    { host: 'host', top: 'stream trên', bottom: 'stream dưới', internal: 'stream bên trong', linkTop: 'trao đổi phía trên', linkBottom: 'trao đổi phía dưới' }[l],
  );
}

export function timelineHeight(nLanes: number, slots: boolean): number {
  return nLanes * ROW + (slots ? SLOTS_BLOCK : 0) + AXIS;
}

// A function, not a constant: the labels are read in the selected language at render time.
const SHORT: () => [RegExp, string][] = () => [
  [/cudaStreamSynchronize\(topStream\)/, tr('sync top', 'đồng bộ trên')],
  [/cudaStreamSynchronize\(bottomStream\)/, tr('sync bottom', 'đồng bộ dưới')],
  [/cudaStreamSynchronize\(internalStream\)/, tr('sync internal', 'đồng bộ bên trong')],
  [/cudaStreamSynchronize\(stream\)/, tr('sync stream', 'đồng bộ stream')],
  [/cudaDeviceSynchronize/, tr('sync device', 'đồng bộ device')],
  [/MPI_Sendrecv \(top\)/, tr('Sendrecv top', 'Sendrecv trên')],
  [/MPI_Sendrecv \(bottom\)/, tr('Sendrecv bottom', 'Sendrecv dưới')],
  [/MPI_Allreduce/, 'Allreduce'],
  [/cudaMemcpy/, tr('copy (waits)', 'sao chép (chờ)')],
];

export function segLabel(s: Seg, v: Version): string {
  if (s.lane === 'host') {
    if (s.kind === 'call') return s.line;
    for (const [re, name] of SHORT()) if (re.test(s.label)) return name;
    return s.label;
  }
  switch (s.kind) {
    case 'kernel':
      return s.part === 'whole'
        ? v === 'nvshmem'
          ? tr('kernel: all rows, puts', 'kernel: mọi hàng, puts')
          : tr('kernel: all rows', 'kernel: mọi hàng')
        : tr(`${s.part} kernel`, `kernel ${s.part ? { top: 'trên', bottom: 'dưới', internal: 'bên trong' }[s.part] : s.part}`);
    case 'pending':
      return tr('waits for slots', 'chờ khe trống');
    case 'wait':
      return tr(`wait ${s.event}`, `chờ ${s.event}`);
    case 'exchange':
      if (s.lane === 'top' || s.lane === 'bottom') return 'ncclSend + ncclRecv'; // i18n-ignore: API call names
      return v === 'nccl' ? 'NCCL' : v === 'nvshmem' ? 'puts' : 'MPI_Sendrecv';
    default:
      return s.kind;
  }
}

const TOKEN: Record<string, string> = { memset: 'memset', record: 'record', wait: 'wait', kernel: 'launch', copy: 'copy', exchange: 'nccl', barrier: 'barrier' };
const fit = (text: string, w: number, size = 10) => text.length * size * 0.56 + 6 <= w;
export const r2 = (x: number) => (Math.round(x * 100) / 100).toFixed(2);

export interface TimelineProps {
  r: LoopResult;
  /** Top of the first lane. */
  y: number;
  t0?: number;
  t1: number;
  /** Draw only what has happened by this time. */
  cursor?: number;
  iters?: number[];
  /** Lane rows to draw, in order. Default: every lane of the version. */
  rows?: Lane[];
  /** Lanes whose contents are drawn (the other rows stay empty). Default: every row. */
  content?: Lane[];
  /** 'calls': the host's short calls only, not its waits. */
  host?: 'all' | 'calls';
  slots?: boolean;
  /** The insertion view: queued operations at the moment they were queued, nothing executed. */
  insertions?: boolean;
  /** Dotted lines from each event record to the waits on it. */
  arrows?: boolean;
  /** Bracket the wait between the bottom rows' end and the bottom exchange's start. */
  gap?: boolean;
  violations?: boolean;
  tick?: number;
}

/** A timeline as an svg group: lanes, segments, optional block slots, axis. */
export function Timeline(p: TimelineProps) {
  const uid = useId().replace(/:/g, '');
  const { r } = p;
  const v = r.input.version;
  const lanes = (p.rows ?? r.lanes).filter((l) => r.lanes.includes(l));
  const t0 = p.t0 ?? 0;
  const span = p.t1 - t0;
  const cur = p.cursor ?? Infinity;
  const x = (t: number) => X0 + ((Math.min(Math.max(t, t0), p.t1) - t0) / span) * (X1 - X0);
  const laneY = (l: Lane) => p.y + lanes.indexOf(l) * ROW;
  const shown = (l: Lane) => lanes.includes(l) && (!p.content || p.content.includes(l));
  const inIter = (i: number) => !p.iters || p.iters.includes(i);
  const hatch = `hatch-${uid}`; // i18n-ignore: svg pattern id
  const out: ReactNode[] = [];

  // lane rows
  lanes.forEach((l) => {
    const y = laneY(l);
    out.push(
      <g key={'lane-' + l}>
        <rect x={X0} y={y} width={X1 - X0} height={LANE_H} fill="var(--cell)" stroke="var(--rule-soft)" />
        <text x={6} y={y + LANE_H / 2 + 4} fontSize="11" style={{ fill: l === 'linkTop' || l === 'linkBottom' ? 'var(--ink-2)' : 'var(--ink)' }}>
          {laneName(l, v)}
        </text>
      </g>,
    );
  });

  if (p.insertions) {
    // Host calls as numbered slots, and what each one queued, where it was queued.
    for (const s of r.segs) {
      if (s.lane !== 'host' || !inIter(s.iter) || s.start >= Math.min(cur, p.t1)) continue;
      const a = x(s.start);
      const b = x(Math.min(s.end, cur));
      out.push(
        <g key={`ins-h-${s.iter}-${s.start}`} data-seg="call" data-lane="host" data-line={s.line} data-start={r2(s.start)}>
          <rect x={a + 1} y={laneY('host') + 3} width={Math.max(1, b - a - 2)} height={LANE_H - 6} rx="2" fill="var(--ink)" />
          {fit(s.line, b - a - 2, 10) ? (
            <text x={(a + b) / 2} y={laneY('host') + LANE_H / 2 + 3.5} fontSize="10" textAnchor="middle" style={{ fill: 'var(--page)' }}>{s.line.split('–')[0]}</text>
          ) : null}
        </g>,
      );
    }
    const pxPerCall = (x(t0 + 0.02) - x(t0));
    for (const q of r.insertions) {
      if (!inIter(q.iter) || q.at > cur + 1e-9 || q.at > p.t1 + 1e-9 || !shown(q.lane)) continue;
      const w = Math.min(46, pxPerCall - 4);
      const b = x(q.at);
      const y = laneY(q.lane);
      out.push(
        <g key={`ins-${q.iter}-${q.line}`} data-token={q.kind} data-lane={q.lane} data-at={r2(q.at)}>
          <line x1={b - w / 2 - 2} x2={b - w / 2 - 2} y1={laneY('host') + LANE_H - 2} y2={y + 2} stroke="var(--ink-3)" strokeDasharray="2 2" />
          <rect x={b - w - 2} y={y + 3} width={w} height={LANE_H - 6} rx="3" fill="var(--page)" stroke="var(--ink-2)" strokeDasharray="3 2" />
          <text x={b - w / 2 - 2} y={y + LANE_H / 2 + 3.5} fontSize="9.5" textAnchor="middle">{TOKEN[q.kind]}</text>
        </g>,
      );
    }
  } else {
    for (const s of r.segs) {
      if (!inIter(s.iter) || !shown(s.lane) || s.start > cur || s.start > p.t1) continue;
      if (s.lane === 'host' && p.host === 'calls' && s.kind === 'blocked') continue;
      const a = x(s.start);
      const b = x(Math.min(s.end, cur));
      const y = laneY(s.lane);
      const w = Math.max(b - a, s.kind === 'call' ? 1 : 1.5);
      const label = segLabel(s, v);
      let body: ReactNode;
      let ink = 'var(--ink)';
      switch (s.kind) {
        case 'call':
          body = <rect x={a} y={y + 5} width={w} height={LANE_H - 10} fill="var(--ink)" />;
          break;
        case 'blocked':
          body = <rect x={a} y={y + 2} width={w} height={LANE_H - 4} fill={`url(#${hatch})`} stroke="var(--ink-2)" />;
          break;
        case 'kernel':
          body = <rect x={a} y={y + 2} width={w} height={LANE_H - 4} rx="2" fill="var(--ink-2)" />;
          ink = 'var(--page)';
          break;
        case 'pending':
          body = <rect x={a} y={y + 3} width={w} height={LANE_H - 6} rx="2" fill="none" stroke="var(--ink-3)" strokeDasharray="4 3" />;
          ink = 'var(--ink-2)';
          break;
        case 'wait':
        case 'barrier':
          body = <rect x={a} y={y + 3} width={w} height={LANE_H - 6} rx="2" fill="none" stroke={s.kind === 'barrier' ? 'var(--ink)' : 'var(--ink-3)'} strokeDasharray="1.5 2" />;
          ink = 'var(--ink-2)';
          break;
        case 'memset':
          body = <rect x={a} y={y + 2} width={w} height={LANE_H - 4} fill="var(--global)" />;
          ink = 'var(--page)';
          break;
        default: // copy, exchange: data in global memory on the move
          body = <rect x={a} y={y + 2} width={w} height={LANE_H - 4} rx="2" fill="var(--global-wash)" stroke="var(--global)" />;
      }
      out.push(
        <g key={`seg-${s.lane}-${s.iter}-${s.kind}-${s.start}`} data-seg={s.kind} data-lane={s.lane} data-iter={s.iter} data-line={s.line}
          data-start={r2(s.start)} data-end={r2(s.end)}>
          {body}
          {s.kind !== 'call' && fit(label, b - a) ? (
            <text x={a + 4} y={y + LANE_H / 2 + 3.5} fontSize="10" style={{ fill: ink }}>{label}</text>
          ) : null}
          <title>
            {tr(
              `${s.lane === 'host' ? 'line ' + s.line + ': ' + s.label : label}, ${r2(s.start)} to ${r2(s.end)}`,
              `${s.lane === 'host' ? 'dòng ' + s.line + ': ' + s.label.replace(/^MPI_Sendrecv \(top\)$/, 'MPI_Sendrecv (trên)').replace(/^MPI_Sendrecv \(bottom\)$/, 'MPI_Sendrecv (dưới)') : label}, từ ${r2(s.start)} đến ${r2(s.end)}`,
            )}
          </title>
        </g>,
      );
    }
    // event records
    for (const m of r.marks) {
      if (!inIter(m.iter) || !shown(m.lane) || m.at > cur || m.at > p.t1) continue;
      const cx = x(m.at);
      const cy = laneY(m.lane) + LANE_H / 2;
      out.push(
        <path key={`mark-${m.lane}-${m.iter}-${m.event}`} d={`M ${cx} ${cy - 5} L ${cx + 4} ${cy} L ${cx} ${cy + 5} L ${cx - 4} ${cy} Z`}
          fill="var(--page)" stroke="var(--ink)" data-mark={m.event} data-at={r2(m.at)}>
          <title>{tr(`cudaEventRecord(${m.event}), line ${m.line}: reached at ${r2(m.at)}`, `cudaEventRecord(${m.event}), dòng ${m.line}: đạt tới lúc ${r2(m.at)}`)}</title>
        </path>,
      );
    }
    if (p.arrows)
      for (const w of r.waits) {
        if (!inIter(w.iter) || w.from === null || !w.fromLane || w.end > cur || !shown(w.lane)) continue;
        out.push(
          <line key={`arrow-${w.lane}-${w.iter}-${w.event}`} x1={x(w.from)} y1={laneY(w.fromLane) + LANE_H / 2} x2={x(w.end)} y2={laneY(w.lane) + LANE_H / 2}
            stroke="var(--ink)" strokeWidth="1.2" strokeDasharray="2 2.5" data-arrow={w.event} data-from={r2(w.from)} data-to={r2(w.end)} />,
        );
      }
    if (p.slots) {
      const sy = p.y + lanes.length * ROW + 4;
      out.push(
        <text key="slots-label" x={6} y={sy + 16} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
          <Translated vi="khe cho các khối">block slots</Translated>
        </text>,
      );
      for (const b of r.blocks) {
        if (!inIter(b.iter) || b.start > cur || b.start > p.t1) continue;
        const a = x(b.start);
        const e = x(Math.min(b.end, cur));
        out.push(
          <rect key={`blk-${b.iter}-${b.slot}-${b.start}`} x={a + 0.5} y={sy + b.slot * (SLOT_H + 1)} width={Math.max(1, e - a - 1)} height={SLOT_H}
            fill={b.part === 'internal' ? 'var(--ink-3)' : 'var(--ink)'} data-block={b.part} />,
        );
      }
    }
    if (p.gap)
      for (const it of r.iters) {
        if (!inIter(it.iter) || !shown('linkBottom') || it.exBottom.start > cur || it.bottomGap < 0.005) continue;
        const a = x(it.bottom.end);
        const b = x(it.exBottom.start);
        const y = laneY('linkBottom') + LANE_H / 2;
        out.push(
          <g key={`gap-${it.iter}`} data-gap={r2(it.bottomGap)}>
            <line x1={a} x2={b} y1={y} y2={y} stroke="var(--ink)" strokeDasharray="4 3" />
            <line x1={a} x2={a} y1={y - 6} y2={y + 6} stroke="var(--ink)" />
            <line x1={b} x2={b} y1={y - 6} y2={y + 6} stroke="var(--ink)" />
            {b - a > 70 ? (
              <text x={(a + b) / 2} y={y - 4} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink)' }}>
                <Translated vi={<>đã sẵn sàng, chờ {r2(it.bottomGap)}</>}>ready, waiting {r2(it.bottomGap)}</Translated>
              </text>
            ) : null}
          </g>,
        );
      }
    if (p.violations)
      r.violations.forEach((f, k) => {
        if (!inIter(f.iter) || f.at > cur || !lanes.includes(f.lane)) return;
        const a = x(f.at);
        const b = x(Math.min(f.until, cur));
        const y = laneY(f.lane);
        out.push(
          <g key={`fail-${k}`} data-fail={f.kind}>
            <rect x={a} y={y - 1} width={Math.max(2, b - a)} height={LANE_H + 2} fill="var(--fail-wash)" stroke="var(--fail)" strokeWidth="1.4" />
            <circle cx={a} cy={y} r="6" fill="var(--fail)" />
            <text x={a} y={y + 3.5} fontSize="9.5" fontWeight="700" textAnchor="middle" style={{ fill: 'var(--page)' }}>!</text>
          </g>,
        );
      });
    // iteration boundaries
    for (const it of r.iters)
      if (it.iter > 1 && inIter(it.iter) && it.hostStart < Math.min(cur, p.t1)) {
        const bx = x(it.hostStart);
        out.push(
          <g key={`iter-${it.iter}`}>
            <line x1={bx} x2={bx} y1={p.y - 4} y2={p.y + lanes.length * ROW - 6} stroke="var(--ink-2)" strokeDasharray="6 4" />
            <text x={bx + 4} y={p.y - 6} fontSize="10" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>lần lặp {it.iter}</>}>iteration {it.iter}</Translated></text>
          </g>,
        );
      }
  }

  // axis
  const ay = p.y + lanes.length * ROW + (p.slots && !p.insertions ? SLOTS_BLOCK : 0);
  const tick = p.tick ?? (span <= 0.5 ? 0.02 : span <= 12 ? 1 : span <= 24 ? 2 : 5);
  const ticks: number[] = [];
  for (let k = Math.ceil(t0 / tick - 1e-9); k * tick <= p.t1 + 1e-9; k++) ticks.push(k * tick);
  out.push(
    <g key="axis">
      <line x1={X0} x2={X1} y1={ay} y2={ay} stroke="var(--rule)" />
      {ticks.map((t, k) => (
        <g key={k}>
          <line x1={x(t)} x2={x(t)} y1={ay} y2={ay + 4} stroke="var(--ink-3)" />
          {span > 0.5 || k % 2 === 0 ? (
            <text x={x(t)} y={ay + 15} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>
              {span <= 0.5 ? t.toFixed(2) : String(Math.round(t * 100) / 100)}
            </text>
          ) : null}
        </g>
      ))}
      <text x={6} y={ay + 15} fontSize="10" style={{ fill: 'var(--ink-3)' }}><Translated vi="thời gian (đơn vị)">time (units)</Translated></text>
    </g>,
  );

  return (
    <g data-timeline={v} data-cursor={Number.isFinite(cur) ? r2(cur) : ''}>
      <defs>
        <pattern id={hatch} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="5" height="5" fill="var(--page)" />
          <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" strokeWidth="1.6" />
        </pattern>
      </defs>
      {out}
    </g>
  );
}

// ---------------------------------------------------------------- the scene

// Getters, not stored runs: each run carries host-call labels, which must be read in the selected language.
export const SCENE_RUNS: Record<'streams' | 'nccl' | 'nvshmem', LoopResult> = {
  get streams() { return simulate({ ...DEFAULTS, version: 'streams' }); },
  get nccl() { return simulate({ ...DEFAULTS, version: 'nccl' }); },
  get nvshmem() { return simulate({ ...DEFAULTS, version: 'nvshmem' }); },
};
const S = SCENE_RUNS.streams;
const S1 = S.iters[0];
const T1 = 10; // every version's first iteration fits
const INS_T1 = 0.26;
const FIRST_WAIT = S.segs.find((s) => s.lane === 'host' && s.iter === 1 && s.kind === 'blocked')!.start;
/** The insertion beat: lines 04 to 22 of Fig. 23.14, everything queued before the host's first wait. */
export const SCENE_QUEUED = S.insertions.filter((q) => q.iter === 1 && q.at <= FIRST_WAIT + 1e-9);

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const ALL_STREAMS: Lane[] = ['top', 'bottom', 'internal'];

function Frame({ beat, headline, sub, children, h }: { beat: number; headline: string; sub?: string; children: ReactNode; h: number }) {
  return (
    <svg viewBox={`0 0 ${VIEW_W} ${h}`} role="img" aria-label={tr("One iteration of the book's loop, drawn against time", 'Một lần lặp của vòng lặp trong sách, vẽ theo thời gian')} data-beat={beat}
      style={{ width: '100%', minWidth: 680, display: 'block' }}>
      <text x={8} y={20} fontSize="14" fontWeight="600" data-headline="">{headline}</text>
      {sub ? <text x={8} y={37} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>{sub}</text> : null}
      {children}
    </svg>
  );
}

const Y = 56;
const H6 = Y + timelineHeight(6, true) + 4;
// A function, not a constant: read in the selected language at render time.
const SUB = () =>
  tr(
    `MPI with streams (Fig. 23.14), iteration 1: internal kernel ${DEFAULTS.internal}, boundary kernels ${DEFAULTS.boundary}, exchanges ${DEFAULTS.exTop} each (units)`,
    `MPI với stream (Hình 23.14), lần lặp 1: kernel bên trong ${DEFAULTS.internal}, các kernel biên ${DEFAULTS.boundary}, mỗi trao đổi ${DEFAULTS.exTop} (đơn vị)`,
  );

function Totals({ p }: { p: number }) {
  const rows: [string, LoopResult][] = [
    [tr('MPI with streams', 'MPI với stream'), SCENE_RUNS.streams],
    ['NCCL', SCENE_RUNS.nccl],
    ['NVSHMEM', SCENE_RUNS.nvshmem],
  ];
  const max = 10;
  const w = (v: number) => (v / max) * 420 * ramp(p, 0.05, 0.5);
  return (
    <Frame
      beat={5}
      headline={tr("The book's three loops: only who waits has changed", 'Ba vòng lặp của sách: chỉ có ai phải chờ là thay đổi')}
      sub={tr(
        'time per iteration and host time spent blocked, from the model at the same durations',
        'thời gian mỗi lần lặp và thời gian host bị chặn, từ mô hình với cùng các khoảng thời gian',
      )}
      h={H6}
    >
      {rows.map(([name, r], k) => {
        const y = 70 + k * 74;
        return (
          <g key={name} data-total={r.input.version} data-iter-time={r2(r.iterTime)} data-blocked={r2(r.hostBlocked)}>
            <text x={8} y={y + 14} fontSize="12" fontWeight="600">{name}</text>
            <text x={8} y={y + 30} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{r.hidden ? tr('exchanges hidden', 'trao đổi được che giấu') : tr('exchanges not hidden', 'trao đổi không được che giấu')}</text>
            <rect x={190} y={y} width={w(r.iterTime)} height={18} fill="var(--ink-2)" />
            <text x={196 + w(r.iterTime)} y={y + 13} fontSize="10.5" opacity={ramp(p, 0.45, 0.55)}><Translated vi={<>{r2(r.iterTime)} mỗi lần lặp</>}>{r2(r.iterTime)} per iteration</Translated></text>
            <rect x={190} y={y + 24} width={w(r.hostBlocked)} height={18} fill="url(#tot-hatch)" stroke="var(--ink-2)" />
            <text x={196 + w(r.hostBlocked)} y={y + 37} fontSize="10.5" opacity={ramp(p, 0.45, 0.55)}><Translated vi={<>{r2(r.hostBlocked)} host bị chặn</>}>{r2(r.hostBlocked)} host blocked</Translated></text>
          </g>
        );
      })}
      <defs>
        <pattern id="tot-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="5" height="5" fill="var(--page)" />
          <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" strokeWidth="1.6" />
        </pattern>
      </defs>
      <text x={8} y={H6 - 14} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
        {p > 0.6
          ? tr(
              'In all three the host waits for most of the iteration: it needs the norm. What changed is that the exchanges no longer wait for the host.',
              'Trong cả ba, host chờ gần hết lần lặp: nó cần giá trị norm. Điều đã thay đổi là các trao đổi không còn phải chờ host.',
            )
          : ''}
      </text>
    </Frame>
  );
}

const beats: Beat[] = [
  {
    clip: 'loop-1',
    get label() {
      return tr('Insertion', 'Đưa vào hàng đợi');
    },
    draw: (p) => {
      const cur = ramp(p, 0.05, 0.75) * FIRST_WAIT;
      const n = SCENE_QUEUED.filter((q) => q.at <= cur + 1e-9).length;
      return (
        <Frame beat={1} h={Y + timelineHeight(4, false) + 4} headline={tr(
            `The host queues ${n} of ${SCENE_QUEUED.length} operations: ${r2(Math.max(0, ...SCENE_QUEUED.filter((q) => q.at <= cur + 1e-9).map((q) => q.at)))} time units so far`,
            `Host đã đưa ${n} trên ${SCENE_QUEUED.length} thao tác vào hàng đợi: đến giờ là ${r2(Math.max(0, ...SCENE_QUEUED.filter((q) => q.at <= cur + 1e-9).map((q) => q.at)))} đơn vị thời gian`,
          )}
          sub={tr(
            'lines 04 to 22 of Fig. 23.14, zoomed in: each call returns once its operation is queued',
            'các dòng 04 đến 22 của Hình 23.14, phóng to: mỗi lời gọi trả về ngay khi thao tác của nó đã vào hàng đợi',
          )}>
          <Timeline r={SCENE_RUNS.streams} y={Y} t1={INS_T1} cursor={cur} iters={[1]} insertions rows={['host', ...ALL_STREAMS]} />
        </Frame>
      );
    },
  },
  {
    clip: 'loop-2',
    get label() {
      return tr('Execution', 'Thực thi');
    },
    draw: (p) => {
      const cur = ramp(p, 0.04, 0.92) * S1.internal.end;
      return (
        <Frame beat={2} h={H6} headline={cur < S1.top.end ? tr('The reset, then both boundary kernels', 'Bước đặt lại, rồi đến cả hai kernel biên') : tr(`Boundary kernels done by ${r2(S1.bottom.end)}; the internal kernel runs to ${r2(S1.internal.end)}`, `Các kernel biên xong lúc ${r2(S1.bottom.end)}; kernel bên trong chạy đến ${r2(S1.internal.end)}`)} sub={SUB()}>
          <Timeline r={SCENE_RUNS.streams} y={Y} t1={T1} cursor={cur} iters={[1]} host="calls" content={['host', ...ALL_STREAMS]} slots />{/* i18n-ignore: prop value */}
        </Frame>
      );
    },
  },
  {
    clip: 'loop-3',
    get label() {
      return tr('Events join', 'Event hợp nhánh');
    },
    draw: (p) => {
      const cur = S1.internal.end * 0.55 + ramp(p, 0.1, 0.9) * (S1.copy.end - S1.internal.end * 0.55);
      return (
        <Frame beat={3} h={H6} headline={cur < S1.copy.start ? tr('computeTop and computeBottom are reached; the copy waits', 'Đã đạt tới computeTop và computeBottom; phép sao chép phải chờ') : tr(`The copy runs at ${r2(S1.copy.start)}, after both events and its own kernel`, `Phép sao chép chạy lúc ${r2(S1.copy.start)}, sau cả hai event và kernel của chính nó`)} sub={SUB()}>
          <Timeline r={SCENE_RUNS.streams} y={Y} t1={T1} cursor={cur} iters={[1]} host="calls" content={['host', ...ALL_STREAMS]} slots arrows />{/* i18n-ignore: prop value */}
        </Frame>
      );
    },
  },
  {
    clip: 'loop-4',
    get label() {
      return tr('The host waits', 'Host chờ');
    },
    draw: (p) => {
      const cur = ramp(p, 0.04, 0.9) * S1.hostEnd;
      return (
        <Frame beat={4} h={H6} headline={cur < S1.exBottom.start ? tr(`The host sits in the top exchange until ${r2(S1.exTop.end)}`, `Host nằm chờ trong trao đổi phía trên đến ${r2(S1.exTop.end)}`) : tr(`Bottom rows ready at ${r2(S1.bottom.end)}, sent at ${r2(S1.exBottom.start)}: ${r2(S1.bottomGap)} lost`, `Các hàng dưới sẵn sàng lúc ${r2(S1.bottom.end)}, được gửi lúc ${r2(S1.exBottom.start)}: mất ${r2(S1.bottomGap)}`)} sub={SUB()}>
          <Timeline r={SCENE_RUNS.streams} y={Y} t1={T1} cursor={cur} iters={[1]} slots gap />
        </Frame>
      );
    },
  },
  {
    clip: 'loop-5',
    get label() {
      return tr('Totals', 'Tổng');
    },
    draw: (p) => <Totals p={p} />,
  },
];

export function LoopScene() {
  return <Scene chapter="ch23" title={tr('One iteration with three streams', 'Một lần lặp với ba stream')} beats={beats} />;
}
