import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { LANE_NAME, pinnedScene, type Lane, type Op, type Schedule } from '../../sim/chunkpipe';
import { Translated, tr } from '../../lib/i18n';

// ---------------------------------------------------------------- lanes, shared with the stepper
// Draws a schedule from sim/chunkpipe.ts as horizontal lanes, up to time t. Operations are
// ink, shaded by stream (colour is kept for memory tiers and failure); a running operation
// is filled up to t inside its dashed outline. Every number comes from the schedule.
const SHADES = [0.88, 0.62, 0.42, 0.26];
const short = (op: Op, s: Schedule) => {
  const both = s.cfg.view === 'metal' || s.cfg.engines === 1;
  if (!both) return `${op.chunk}`;
  const word = s.cfg.view === 'metal' ? { in: 'fill', out: 'drain', kernel: '' }[op.kind] : { in: 'in', out: 'out', kernel: '' }[op.kind];
  return word ? `${word} ${op.chunk}` : `${op.chunk}`;
};

export function PipeLanes({
  s, t, x, y, w, span, laneH = 30, gap = 10, labelW = 148, current = null, bubbles = true, id, axis = true,
}: {
  s: Schedule; t: number; x: number; y: number; w: number; span: number; laneH?: number; gap?: number; labelW?: number;
  current?: number | null; bubbles?: boolean; id: string; axis?: boolean;
}) {
  const X = (v: number) => x + labelW + (Math.max(0, v) / span) * (w - labelW - 8);
  const rowY = (l: Lane) => y + s.lanes.indexOf(l) * (laneH + gap);
  const curStream = current != null ? s.ops[current]?.stream : null;
  const every = span > 40 ? 10 : span > 16 ? 4 : 2;
  const out: ReactNode[] = [];
  out.push(
    <defs key="defs">
      <pattern id={id + '-hatch'} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <line x1="0" y1="0" x2="0" y2="6" stroke="var(--ink-3)" strokeWidth="1.5" />
      </pattern>
      <pattern id={id + '-idle'} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)">
        <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" strokeWidth="0.9" />
      </pattern>
    </defs>,
  );
  for (const l of s.lanes) {
    const yy = rowY(l);
    out.push(
      <g key={'lane-' + l} data-lane={l}>
        <text x={x + labelW - 10} y={yy + laneH / 2 + 4} fontSize="11.5" fontWeight={l === s.bottleneck ? 700 : 500} textAnchor="end">
          {tr(
            LANE_NAME[l],
            {
              host: 'Luồng host',
              h2d: 'Copy engine vào',
              compute: 'Tính toán',
              d2h: 'Copy engine ra',
              copy: 'Copy engine (cả hai chiều)',
              cpu: 'CPU: rút ra, rồi nạp vào',
              gpu: 'GPU',
            }[l],
          )}
        </text>
        <rect x={X(0)} y={yy} width={X(span) - X(0)} height={laneH} fill="none" stroke="var(--rule-soft)" />
      </g>,
    );
  }
  // the bubble: idle time of the bottleneck lane, as far as time t
  if (bubbles)
    s.bubbles.forEach((b, i) => {
      if (b.from >= t) return;
      const yy = rowY(s.bottleneck);
      const to = Math.min(b.to, t);
      out.push(
        <rect key={'bub' + i} data-bubble={`${b.from}-${b.to}`} x={X(b.from)} y={yy + 1} width={Math.max(0, X(to) - X(b.from))} height={laneH - 2}
          fill={`url(#${id}-idle)`} stroke="var(--ink-3)" strokeDasharray="2 2" />,
      );
    });
  // host spans: pageable copies (CUDA) or waits for a slot (Metal)
  s.host.forEach((h, i) => {
    if (h.from > t + 1e-9 || h.to - h.from < 1e-9) return;
    const yy = rowY(h.lane);
    const to = Math.min(h.to, t);
    const hatched = h.kind !== 'staging';
    out.push(
      <g key={'h' + i} data-host={h.kind} data-from={h.from} data-to={h.to}>
        <rect x={X(h.from)} y={yy + 3} width={Math.max(1, X(to) - X(h.from))} height={laneH - 6} rx="2"
          fill={hatched ? `url(#${id}-hatch)` : 'var(--ink-2)'} stroke={hatched ? 'var(--ink-3)' : 'none'} strokeDasharray={hatched ? '3 2' : undefined} />
        {X(to) - X(h.from) > 54 ? (
          <text x={(X(h.from) + X(to)) / 2} y={yy + laneH / 2 + 4} fontSize="10" textAnchor="middle" fontWeight="600"
            style={{ fill: hatched ? 'var(--ink)' : 'var(--page)' }}>
            {h.kind === 'blocked' ? tr('blocked', 'bị chặn') : h.kind === 'staging' ? tr('staging', 'trung chuyển') : tr('waiting', 'đang chờ')}
          </text>
        ) : null}
      </g>,
    );
  });
  // CUDA, pinned: the host queues everything at time 0
  if (s.cfg.view === 'cuda' && s.cfg.host === 'pinned' && s.ops.length && t >= 0) {
    const yy = rowY('host');
    out.push(
      <g key="issued" data-issued={s.ops.length}>
        <rect x={X(0)} y={yy + 4} width={4} height={laneH - 8} fill="var(--ink)" />
        <text x={X(0) + 9} y={yy + laneH / 2 + 4} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
          <Translated vi={<>đã đưa cả {s.ops.length} thao tác vào hàng đợi lúc 0, rồi rảnh</>}>
            queued all {s.ops.length} operations at 0, then free
          </Translated>
        </text>
      </g>,
    );
  }
  // operations
  for (const op of s.ops) {
    if (op.start > t + 1e-9) continue;
    const yy = rowY(op.lane);
    const x0 = X(op.start);
    const x1 = X(op.end);
    const xt = X(Math.min(op.end, t));
    const running = op.end > t + 1e-9;
    const shade = SHADES[op.stream % SHADES.length];
    const isCur = current === op.id;
    const sameStream = curStream != null && op.stream === curStream;
    const label = short(op, s);
    out.push(
      <g key={'op' + op.id} data-op={op.id} data-kind={op.kind} data-chunk={op.chunk} data-stream={op.stream} data-lane={op.lane}
        data-start={op.start} data-end={op.end} data-state={running ? 'running' : 'done'}>
        <rect x={x0} y={yy + 2} width={Math.max(1, xt - x0)} height={laneH - 4} fill="var(--ink)" fillOpacity={shade} />
        <rect x={x0} y={yy + 2} width={Math.max(1, x1 - x0)} height={laneH - 4} fill="none"
          stroke="var(--ink)" strokeWidth={isCur ? 2.6 : sameStream ? 1.4 : 0.8} strokeDasharray={running ? '4 2' : undefined} />
        {x1 - x0 > 9 + label.length * 5.6 ? (
          <text x={(x0 + x1) / 2} y={yy + laneH / 2 + 4} fontSize="10.5" textAnchor="middle" fontWeight={isCur ? 700 : 500}
            style={{ fill: !running && shade >= 0.6 ? 'var(--page)' : 'var(--ink)' }}>{label}</text>
        ) : null}
      </g>,
    );
  }
  // axis, the serial time, and now
  const bottom = y + s.lanes.length * (laneH + gap) - gap;
  if (axis) {
    out.push(<line key="axis" x1={X(0)} y1={bottom + 8} x2={X(span)} y2={bottom + 8} stroke="var(--rule)" />);
    for (let v = 0; v <= span + 1e-9; v += every)
      out.push(
        <g key={'tick' + v}>
          <line x1={X(v)} y1={bottom + 8} x2={X(v)} y2={bottom + 12} stroke="var(--ink-3)" />
          <text x={X(v)} y={bottom + 23} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{v}</text>
        </g>,
      );
  }
  if (s.serial <= span + 1e-9 && s.serial > s.total + 1e-9)
    out.push(
      <g key="serial" data-serial={s.serial}>
        <line x1={X(s.serial)} y1={y - 6} x2={X(s.serial)} y2={bottom + 8} stroke="var(--ink-3)" strokeDasharray="5 4" />
        <text x={X(s.serial) - 4} y={y - 9} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>không chia nhỏ: {fmtN(s.serial)}</>}>in one piece: {fmtN(s.serial)}</Translated></text>
      </g>,
    );
  if (t > 1e-9 || s.ops.length === 0)
    out.push(
      <g key="now" data-now={t}>
        <line x1={X(Math.min(t, span))} y1={y - 4} x2={X(Math.min(t, span))} y2={bottom + 8} stroke="var(--ink)" strokeDasharray="2 3" />
      </g>,
    );
  return <g data-total={s.total}>{out}</g>;
}
export const fmtN = (v: number) => String(Number(v.toFixed(2)));
/** Height PipeLanes draws for a schedule, axis included. */
export const lanesHeight = (s: Schedule, laneH = 30, gap = 10) => s.lanes.length * (laneH + gap) - gap + 26;

// ---------------------------------------------------------------- the scene
// "Why an asynchronous copy needs pinned memory" (pp. 562-563). Six beats, one narration
// clip each (content/ch23/narration/pinned-N.txt). Pages, frames and the staging plan come
// from pinnedScene(); parts 4 and 5 are the stepper's four-chunk case from the same model.
const M = pinnedScene();
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const W = 720;
const PX = 40; // pages row
const PW = 64;
const FW = 50; // frames
const FX = 40;

function Pages({ moving, dma }: { moving: number; dma?: { k: number; broken: boolean } }) {
  const frameOf = (page: number) => (page === M.moved.page && moving > 0 ? null : M.frames[page]);
  const pageX = (i: number) => PX + i * (PW + 8);
  const frameX = (f: number) => FX + f * (FW + 4.5);
  const used = new Set(M.frames);
  return (
    <g>
      <text x={PX} y={26} fontSize="12" fontWeight="600"><Translated vi={<>Một buffer từ malloc: {M.pages} trang ảo, nối tiếp nhau</>}>A buffer from malloc: {M.pages} virtual pages, one after another</Translated></text>
      {Array.from({ length: M.pages }, (_, i) => (
        <g key={'p' + i} data-page={i}>
          <rect x={pageX(i)} y={36} width={PW} height={34} rx="3" fill="var(--cell)" stroke="var(--ink)"
            strokeWidth={dma && dma.k === i ? 2.4 : 1} />
          <text x={pageX(i) + PW / 2} y={57} fontSize="11.5" textAnchor="middle"><Translated vi={<>trang {i}</>}>page {i}</Translated></text>
        </g>
      ))}
      <text x={FX} y={172} fontSize="12" fontWeight="600"><Translated vi="Bộ nhớ vật lý: các khung (frame)">Physical memory: frames</Translated></text>
      {Array.from({ length: M.frameCount }, (_, f) => {
        // mine: holds one of the buffer's pages; free: holds nothing; otherwise another process's
        const left = f === M.frames[M.moved.page] && moving >= 1;
        const arrived = f === M.moved.to && moving >= 1;
        const mine = (used.has(f) && !left) || arrived;
        const free = !mine && (left || f === M.moved.to);
        const broken = dma?.broken && f === M.frames[dma.k];
        return (
          <g key={'f' + f} data-frame={f} data-state={broken ? 'reassigned' : mine ? 'mine' : free ? 'free' : 'other'}>
            <rect x={frameX(f)} y={180} width={FW} height={34} rx="3"
              fill={broken ? 'var(--fail-wash)' : mine || free ? 'var(--cell)' : 'url(#pin-hatch)'}
              stroke={broken ? 'var(--fail)' : free ? 'var(--ink-3)' : 'var(--rule)'} strokeWidth={broken ? 2 : 1}
              strokeDasharray={free && !broken ? '3 2' : undefined} />
            <text x={frameX(f) + FW / 2} y={201} fontSize="10.5" textAnchor="middle" style={{ fill: mine ? 'var(--ink)' : 'var(--ink-3)' }}>
              {broken ? tr('not yours', 'không phải của bạn') : free ? tr('free', 'trống') : tr(`frame ${f}`, `khung ${f}`)}
            </text>
          </g>
        );
      })}
      {Array.from({ length: M.pages }, (_, i) => {
        const f = frameOf(i);
        const sx = pageX(i) + PW / 2;
        if (i === M.moved.page && moving > 0) {
          const from = frameX(M.frames[i]) + FW / 2;
          const to = frameX(M.moved.to) + FW / 2;
          const ex = from + (to - from) * clamp(moving);
          return <line key={'l' + i} x1={sx} y1={70} x2={ex} y2={180} stroke="var(--ink)" strokeWidth="1.8" strokeDasharray="4 3" data-moving={i} />;
        }
        return f == null ? null : <line key={'l' + i} x1={sx} y1={70} x2={frameX(f) + FW / 2} y2={180} stroke="var(--ink-3)" strokeWidth="1" />;
      })}
      <text x={W - 20} y={236} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi="gạch chéo: khung do tiến trình khác giữ; nét đứt: trống">hatched: frames other processes hold; dashed: free</Translated></text>
    </g>
  );
}

function Stage({ children, label }: { children: ReactNode; label: string }) {
  return (
    <svg viewBox={`0 0 ${W} 300`} role="img" aria-label={label} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <defs>
        <pattern id="pin-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="6" stroke="var(--ink-3)" strokeWidth="1.2" />
        </pattern>
      </defs>
      {children}
    </svg>
  );
}

function Headline({ children }: { children: ReactNode }) {
  return <text x={20} y={288} fontSize="13" fontWeight="600" data-headline>{children}</text>;
}

function StagingBeat({ p }: { p: number }) {
  const plan = M.plan;
  const span = plan.returnsAt;
  const t = ramp(p, 0.05, 0.92) * span;
  const L = 150;
  const X = (v: number) => L + (v / span) * (W - L - 30);
  const laneY = { host: 140, dma: 190 };
  const cur = plan.pieces.filter((x) => x.hostFrom <= t);
  return (
    <g data-staging={plan.pieces.length} data-returns={plan.returnsAt}>
      <text x={20} y={26} fontSize="12" fontWeight="600">
        <Translated
          vi={
            <>
              Sao chép {M.pages} trang pageable: {plan.pieces.length} phần qua {2} buffer trung chuyển được ghim, mỗi vùng {M.staging} trang
            </>
          }
        >
          Copying {M.pages} pageable pages: {plan.pieces.length} pieces through {2} pinned staging buffers of {M.staging} pages
        </Translated>
      </text>
      {[0, 1].map((b) => {
        const holding = plan.pieces.filter((x) => x.buffer === b && x.hostFrom <= t && x.dmaTo > t).pop();
        return (
          <g key={b} data-buffer={b}>
            <rect x={L + b * 190} y={52} width={170} height={36} rx="3" fill={holding ? 'var(--ink)' : 'var(--cell)'} fillOpacity={holding ? 0.18 : 1} stroke="var(--ink)" />
            <text x={L + b * 190 + 85} y={74} fontSize="11" textAnchor="middle">
              <Translated vi={<>buffer trung chuyển {b} (được ghim){holding ? `: phần ${holding.k}` : ''}</>}>
                staging buffer {b} (pinned){holding ? `: piece ${holding.k}` : ''}
              </Translated>
            </text>
          </g>
        );
      })}
      <text x={L - 10} y={laneY.host + 19} fontSize="11.5" textAnchor="end"><Translated vi="Luồng host">Host thread</Translated></text>
      <text x={L - 10} y={laneY.dma + 19} fontSize="11.5" textAnchor="end"><Translated vi="Copy engine">Copy engine</Translated></text>
      {/* the host is held for the whole call */}
      <rect x={X(0)} y={laneY.host + 2} width={Math.max(0, X(Math.min(t, span)) - X(0))} height={26} fill="url(#pin-hatch)" stroke="var(--ink-3)" strokeDasharray="3 2" />
      {cur.map((x) => (
        <g key={x.k} data-piece={x.k}>
          <rect x={X(x.hostFrom)} y={laneY.host + 2} width={X(Math.min(x.hostTo, t)) - X(x.hostFrom)} height={26} fill="var(--ink-2)" />
          {x.hostTo <= t ? <text x={(X(x.hostFrom) + X(x.hostTo)) / 2} y={laneY.host + 19} fontSize="10" textAnchor="middle" style={{ fill: 'var(--page)' }}><Translated vi={<>chép {x.k}</>}>copy {x.k}</Translated></text> : null}
          {x.dmaFrom <= t ? (
            <>
              <rect x={X(x.dmaFrom)} y={laneY.dma + 2} width={X(Math.min(x.dmaTo, t)) - X(x.dmaFrom)} height={26} fill="var(--ink)" fillOpacity={0.8} />
              {x.dmaTo <= t ? <text x={(X(x.dmaFrom) + X(x.dmaTo)) / 2} y={laneY.dma + 19} fontSize="10" textAnchor="middle" style={{ fill: 'var(--page)' }}><Translated vi={<>chuyển {x.k}</>}>move {x.k}</Translated></text> : null}
            </>
          ) : null}
        </g>
      ))}
      <line x1={X(0)} y1={232} x2={X(span)} y2={232} stroke="var(--rule)" />
      {Array.from({ length: Math.floor(span) + 1 }, (_, v) => (
        <text key={v} x={X(v)} y={246} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{v}</text>
      ))}
      {t >= span - 1e-9 ? (
        <g data-returned>
          <line x1={X(span)} y1={laneY.host - 8} x2={X(span)} y2={laneY.dma + 30} stroke="var(--ink)" strokeWidth="1.6" />
          <text x={X(span) - 4} y={laneY.host - 12} fontSize="11" fontWeight="700" textAnchor="end"><Translated vi={<>lời gọi trả về lúc {span}</>}>the call returns at {span}</Translated></text>
        </g>
      ) : null}
      <text x={20} y={268} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi={<>Từ bộ nhớ ghim, engine sẽ đọc thẳng các trang: {plan.pinnedTime} cho cùng lượng byte, và host rảnh ngay lập tức.</>}>
          From pinned memory the engine would read the pages directly: {plan.pinnedTime} for the same bytes, and the host free at once.
        </Translated>
      </text>
    </g>
  );
}

function TimelineBeat({ p, which }: { p: number; which: 'pageable' | 'pinned' }) {
  const s = which === 'pageable' ? M.pageable : M.pinned;
  const span = M.pageable.total;
  const t = ramp(p, 0.04, 0.85) * s.total;
  return (
    <g data-timeline={which} data-total={s.total}>
      <text x={20} y={22} fontSize="12" fontWeight="600">
        <Translated
          vi={
            <>
              Bốn chunk gồm copy-in 1, kernel 2, copy-out 1, mỗi chunk một stream, buffer host {which === 'pageable' ? 'pageable' : 'được ghim'}
            </>
          }
        >
          Four chunks of copy-in 1, kernel 2, copy-out 1, a stream each, {which} host buffers
        </Translated>
      </text>
      <PipeLanes s={s} t={t} x={10} y={44} w={W - 20} span={span} laneH={28} gap={9} id={'pin-' + which} bubbles={false} />
      {which === 'pinned' && p > 0.86 ? (
        <g data-compare>
          <text x={W - 20} y={250} fontSize="12" fontWeight="700" textAnchor="end">
            <Translated vi={<>pageable {M.pageable.total}, ghim {M.pinned.total}</>}>
              pageable {M.pageable.total}, pinned {M.pinned.total}
            </Translated>
          </text>
        </g>
      ) : null}
    </g>
  );
}

function TodayBeat({ p }: { p: number }) {
  const cols = [
    {
      head: 'CUDA',
      lines: [
        tr('pinned: cudaMallocHost', 'ghim: cudaMallocHost'),
        tr('from malloc: legal, but', 'từ malloc: hợp lệ, nhưng'),
        tr('"will revert to a synchronous', '"sẽ quay về hành vi'),
        tr('behavior" (A6)', 'đồng bộ" (A6)'),
      ],
    },
    {
      head: 'HIP',
      lines: [
        tr('pinned: hipHostMalloc', 'ghim: hipHostMalloc'),
        tr('not pinned: "the memory', 'không ghim: "phép sao chép'),
        tr('copy will be performed', 'bộ nhớ sẽ được thực hiện'),
        tr('synchronously" (B4)', 'một cách đồng bộ" (B4)'),
      ],
    },
    {
      head: 'Apple silicon', // i18n-ignore: product name, also the column key
      lines: [
        tr('a shared buffer is the', 'một buffer shared chính là'),
        tr("GPU's memory: nothing to", 'bộ nhớ của GPU: không có gì'),
        tr('copy, nothing to pin;', 'để sao chép, không có gì để ghim;'),
        tr('only the order matters (C4)', 'chỉ thứ tự là quan trọng (C4)'),
      ],
    },
  ];
  return (
    <g>
      <text x={20} y={30} fontSize="12" fontWeight="600"><Translated vi="Ngày nay: cùng quy tắc, hoặc không cần sao chép gì cả">Today: the same rule, or no copy at all</Translated></text>
      {cols.map((c, i) => (
        <g key={c.head} opacity={ramp(p, 0.08 + i * 0.27, 0.2 + i * 0.27)} data-col={c.head}>
          <rect x={20 + i * 230} y={50} width={214} height={150} rx="4" fill="var(--cell)" stroke="var(--rule)" />
          <text x={34 + i * 230} y={76} fontSize="13" fontWeight="700">{c.head}</text>
          {c.lines.map((l, k) => (
            <text key={k} x={34 + i * 230} y={102 + k * 21} fontSize="11.5">{l}</text>
          ))}
        </g>
      ))}
    </g>
  );
}

const beats: Beat[] = [
  {
    clip: 'pinned-1',
    get label() {
      return tr('Pages', 'Trang');
    },
    draw: (p) => (
      <Stage label={tr('Virtual pages mapped to physical frames, and the operating system moving one', 'Các trang ảo được ánh xạ vào khung vật lý, và hệ điều hành di chuyển một trang')}>
        <Pages moving={ramp(p, 0.5, 0.8)} />
        <Headline>{p < 0.5
            ? tr('pageable: each page sits in some frame, for now', 'pageable: mỗi trang nằm trong một khung nào đó, tạm thời thôi')
            : tr(`the OS moves page ${M.moved.page} to frame ${M.moved.to}, at any time`, `hệ điều hành chuyển trang ${M.moved.page} sang khung ${M.moved.to}, vào bất cứ lúc nào`)}</Headline>
      </Stage>
    ),
  },
  {
    clip: 'pinned-2',
    get label() {
      return tr('DMA', 'DMA');
    },
    draw: (p) => {
      const k = Math.min(M.pages - 1, Math.floor(ramp(p, 0.05, 0.55) * M.pages));
      const broken = p > 0.6;
      return (
        <Stage label={tr('The copy engine reads physical frames directly', 'Copy engine đọc trực tiếp các khung vật lý')}>
          <Pages moving={0} dma={{ k: broken ? 5 : k, broken }} />
          <rect x={560} y={100} width={140} height={40} rx="4" fill="var(--ink)" />
          <text x={630} y={124} fontSize="12" textAnchor="middle" style={{ fill: 'var(--page)' }}><Translated vi="copy engine (DMA)">copy engine (DMA)</Translated></text>
          <Headline>{broken
              ? tr('a frame reassigned mid-copy: the engine would read data that is not yours', 'một khung bị gán lại giữa lúc sao chép: engine sẽ đọc dữ liệu không phải của bạn')
              : tr(`the engine reads frame ${M.frames[k]} (page ${k}) by its physical address`, `engine đọc khung ${M.frames[k]} (trang ${k}) theo địa chỉ vật lý của nó`)}</Headline>
        </Stage>
      );
    },
  },
  {
    clip: 'pinned-3',
    get label() {
      return tr('Staging', 'Trung chuyển');
    },
    draw: (p) => (
      <Stage label={tr('A pageable copy staged through pinned buffers', 'Một phép sao chép từ bộ nhớ phân trang (pageable) đi qua các buffer được ghim')}>
        <StagingBeat p={p} />
        <Headline>{tr(`the host takes part in every piece and is held until ${M.plan.returnsAt}`, `host tham gia vào mọi phần và bị giữ đến ${M.plan.returnsAt}`)}</Headline>
      </Stage>
    ),
  },
  {
    clip: 'pinned-4',
    get label() {
      return tr('Pageable', 'Pageable');
    },
    draw: (p) => (
      <Stage label={tr('The four-chunk pipeline with pageable buffers', 'Pipeline bốn chunk với buffer pageable')}>
        <TimelineBeat p={p} which="pageable" />{/* i18n-ignore: schedule key */}
        <Headline>
          {p > 0.86
            ? tr(`${M.pageable.total}: the same as doing it in one piece (${M.pageable.serial})`, `${M.pageable.total}: bằng với làm không chia nhỏ (${M.pageable.serial})`)
            : tr('each copy holds the host, so the next operation is queued late', 'mỗi phép sao chép giữ host, nên thao tác kế tiếp vào hàng đợi muộn')}</Headline>
      </Stage>
    ),
  },
  {
    clip: 'pinned-5',
    get label() {
      return tr('Pinned', 'Ghim');
    },
    draw: (p) => (
      <Stage label={tr('The four-chunk pipeline with pinned buffers', 'Pipeline bốn chunk với buffer được ghim')}>
        <TimelineBeat p={p} which="pinned" />{/* i18n-ignore: schedule key */}
        <Headline>
          {p > 0.86
            ? tr(`${M.pinned.total} instead of ${M.pageable.total}`, `${M.pinned.total} thay vì ${M.pageable.total}`)
            : tr("every call returns at once; chunk 1's copy runs under kernel 0", 'mọi lời gọi trả về ngay; phép sao chép của chunk 1 chạy chồng lấp bên dưới kernel 0')}</Headline>
      </Stage>
    ),
  },
  {
    clip: 'pinned-6',
    get label() {
      return tr('Today', 'Ngày nay');
    },
    draw: (p) => (
      <Stage label={tr('Pinned memory on CUDA, HIP and Apple silicon', 'Bộ nhớ ghim (pinned memory) trên CUDA, HIP và Apple silicon')}>
        <TodayBeat p={p} />
        <Headline>
          <Translated vi="một phép sao chép async từ bộ nhớ phân trang là hợp lệ nhưng âm thầm chạy đồng bộ">a pageable async copy is legal and silently synchronous</Translated>
        </Headline>
      </Stage>
    ),
  },
];

export function PinnedScene() {
  return <Scene chapter="ch23" title={tr('Why an asynchronous copy needs pinned memory', 'Vì sao phép sao chép bất đồng bộ cần bộ nhớ ghim (pinned memory)')} beats={beats} />;
}
