import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { tr } from '../../lib/i18n';
import { roles, speedup, timeline, type Timeline } from '../../sim/gemmpipe';

// "Loading the next tile while computing this one": the two-lane timelines of
// Figs. 15.12 and 15.13 for four tile pairs, rearranging themselves. Every box and
// barrier is drawn from src/sim/gemmpipe.ts (timeline); nothing is placed by hand.
// Six beats, one narration clip each (content/ch15/narration/pipeline-N.txt).
// Lanes is also the drawing of section 15.8's Gantt explorer.

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const units = (v: number) => tr(`${Number(v.toFixed(2))} ${v === 1 ? 'unit' : 'units'}`, `${Number(v.toFixed(2))} đơn vị`);

export interface LanesProps {
  tl: Timeline;
  /** Where time 0 is, and the top of the drawing. */
  x: number;
  y: number;
  /** Pixels per time unit. */
  scale: number;
  title?: ReactNode;
  /** Draw only what has happened by this time. */
  upTo?: number;
  /** Barrier labels: none, only the true ones, or all. */
  deps?: 'none' | 'true' | 'all';
  /** Opacity of the barriers that guard only reuse (the false dependences). */
  falseOpacity?: number;
  /** Write the buffer under each box. */
  bufs?: boolean;
  dim?: boolean;
  /** Index of a barrier to draw heavier. */
  hiBarrier?: number;
}

/** Height of one Lanes drawing, with room for barrier labels. */
export const LANES_H = 92;

/** One schedule as two lanes, memory hardware above compute hardware. */
export function Lanes({ tl, x, y, scale, title, upTo = Infinity, deps = 'none', falseOpacity = 1, bufs = false, dim = false, hiBarrier }: LanesProps) {
  const X = (t: number) => x + t * scale;
  const memY = y + 10;
  const compY = y + 42;
  const H = 28;
  const done = Math.min(upTo, tl.total);
  return (
    <g data-scheme={tl.scheme} data-total={tl.total} data-n={tl.n} opacity={dim ? 0.45 : 1}>
      {title ? (
        <text x={x} y={y} fontSize="12.5" fontWeight="600" style={{ fill: 'var(--ink)' }}>{title}</text>
      ) : null}
      <text x={x - 8} y={memY + 18} fontSize="11" textAnchor="end" style={{ fill: 'var(--global)' }}>{tr('memory', 'bộ nhớ')}</text>
      <text x={x - 8} y={compY + 18} fontSize="11" textAnchor="end" style={{ fill: 'var(--reg)' }}>{tr('compute', 'tính toán')}</text>
      <line x1={x} x2={X(tl.total)} y1={memY + H + 2} y2={memY + H + 2} stroke="var(--rule)" />
      <line x1={x} x2={X(tl.total)} y1={compY + H + 2} y2={compY + H + 2} stroke="var(--rule)" />
      {tl.boxes.map((b, i) => {
        if (b.start >= upTo) return null;
        const full = (b.end - b.start) * scale;
        const w = Math.max(0, Math.min(upTo, b.end) - b.start) * scale;
        const top = b.lane === 'mem' ? memY : compY;
        const mem = b.lane === 'mem';
        const label = mem
          ? full >= 84 ? tr(`Load A${b.tile} B${b.tile}`, `Nạp A${b.tile} B${b.tile}`) : full >= 40 ? `A${b.tile} B${b.tile}` : `${b.tile}`
          : full >= 84 ? tr(`Compute C${b.tile}`, `Tính C${b.tile}`) : full >= 22 ? `C${b.tile}` : `${b.tile}`;
        const show = w >= full - 0.5;
        return (
          <g key={i}>
            <rect data-lane={b.lane} data-tile={b.tile} data-start={b.start} data-end={b.end} data-buf={b.buf}
              x={X(b.start) + 0.5} y={top} width={Math.max(0, w - 1)} height={H} rx="2"
              fill={mem ? 'var(--global-wash)' : 'var(--reg-wash)'} stroke={mem ? 'var(--global)' : 'var(--reg)'} />
            {show ? (
              <text x={X(b.start) + full / 2} y={top + (bufs ? 12 : 18)} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink)' }}>
                {label}
              </text>
            ) : null}
            {show && bufs ? (
              <text x={X(b.start) + full / 2} y={top + 24} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--shared)' }}>
                {full >= 60 ? tr(`buffer ${b.buf}`, `buffer ${b.buf}`) : `b${b.buf}`}
              </text>
            ) : null}
          </g>
        );
      })}
      {tl.barriers.map((bar, i) => {
        if (bar.start > upTo) return null;
        const onlyReuse = bar.reuse && !bar.arrival;
        const op = onlyReuse ? falseOpacity : 1;
        if (op <= 0.01) return null;
        const hi = hiBarrier === i;
        const label = deps === 'all' || (deps === 'true' && bar.arrival) ? (bar.arrival ? tr('true', 'thật') : tr('false', 'giả')) : null;
        return (
          <g key={'b' + i} opacity={op} data-barrier={i} data-arrival={bar.arrival} data-reuse={bar.reuse}>
            {bar.end > bar.start ? (
              <rect x={X(bar.start)} y={memY - 4} width={(bar.end - bar.start) * scale} height={compY + H + 8 - memY} fill="var(--rule)" />
            ) : null}
            <line x1={X(bar.start)} x2={X(bar.start)} y1={memY - 4} y2={compY + H + 4}
              stroke="var(--ink)" strokeWidth={hi ? 3 : 1.6} strokeDasharray={hi ? undefined : '4 3'} />
            {label ? (
              <text x={X(bar.start)} y={compY + H + 18} fontSize="10" textAnchor="middle" fontWeight={hi ? 700 : 400} style={{ fill: 'var(--ink-2)' }}>
                {label}
              </text>
            ) : null}
          </g>
        );
      })}
      {done >= tl.total ? (
        <text x={X(tl.total) + 8} y={compY + 2} fontSize="12" fontWeight="600" style={{ fill: 'var(--ink)' }}>{units(tl.total)}</text>
      ) : null}
    </g>
  );
}

// ---------------------------------------------------------------- the scene
const N = 4;
const X0 = 92;
const W = 560; // pixels for the longest timeline of a beat
const base = timeline('two-barriers', N, 1, 1);
const dbl = timeline('double-buffered', N, 1, 1);
const pipe = timeline('pipelined', N, 1, 1);
const slowBase = timeline('two-barriers', N, 3, 1);
const slowPipe = timeline('pipelined', N, 3, 1);
const S1 = W / base.total;
const S5 = W / pipe.total; // one timeline in part 5: drawn larger so each iteration has room for its labels
const S6 = W / slowBase.total;
const pct = (v: number) => `${Math.round(v * 100)}%`;

function Stage({ beat, children, headline, sub }: { beat: number; children: ReactNode; headline: ReactNode; sub?: ReactNode }) {
  return (
    <svg data-beat={beat} viewBox="0 0 720 330" role="img" aria-label={tr('Memory and compute lanes of one block over time', 'Làn bộ nhớ và làn tính toán của một khối theo thời gian')} style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      <text x={20} y={24} fontSize="13.5" fontWeight="600" style={{ fill: 'var(--ink)' }}>{headline}</text>
      {children}
      {sub ? <text x={20} y={318} fontSize="12" style={{ fill: 'var(--ink-2)' }}>{sub}</text> : null}
    </svg>
  );
}

function rolesRow(tl: Timeline, y: number, scale: number, upTo: number) {
  return tl.spans
    .filter((s) => s.label.startsWith('iteration') && s.start <= upTo)
    .map((s) => {
      const i = Number(s.label.split(' ')[1]);
      const r = roles(i);
      const cx = X0 + ((s.start + s.end) / 2) * scale;
      return (
        <g key={s.label} data-iter={i} data-curr={r.curr} data-next={r.next}>
          <text x={cx} y={y} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink)' }}>{tr(s.label, `vòng lặp ${i}`)}</text>
          <text x={cx} y={y + 15} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--shared)' }}>curr {r.curr}, next {r.next}{/* i18n-ignore: curr and next are the kernel's buffer variables */}</text>
        </g>
      );
    });
}

const beats: Beat[] = [
  {
    clip: 'pipeline-1',
    get label() {
      return tr('One block alone', 'Một khối đơn độc');
    },
    draw: (p) => (
      <Stage beat={1} headline={tr('one block per SM: load, wait, compute, wait', 'mỗi SM một khối: nạp, chờ, tính, chờ')}
        sub={p > 0.9 ? tr(`memory busy ${pct(base.memUtil)} of the time, compute busy ${pct(base.compUtil)}`, `bộ nhớ bận ${pct(base.memUtil)} thời gian, tính toán bận ${pct(base.compUtil)}`) : undefined}>
        <Lanes tl={base} x={X0} y={70} scale={S1} upTo={ramp(p, 0.04, 0.88) * base.total} title={tr('one buffer, two barriers per iteration', 'một buffer, hai rào chắn mỗi vòng lặp')} />
      </Stage>
    ),
  },
  {
    clip: 'pipeline-2',
    get label() {
      return tr('Two kinds of barrier', 'Hai loại rào chắn');
    },
    draw: (p) => (
      <Stage beat={2} headline={p < 0.45 ? tr('after the loads: a true dependence', 'sau các lần nạp: một phụ thuộc thật') : tr('after the compute: a false dependence', 'sau phần tính: một phụ thuộc giả')}
        sub={p < 0.45 ? tr('the data has to arrive before anyone computes with it', 'dữ liệu phải đến nơi trước khi có ai tính toán với nó') : tr('the next load only waits so as not to overwrite tiles still being read', 'lần nạp kế tiếp chỉ chờ để khỏi ghi đè lên các ô còn đang được đọc')}>
        <Lanes tl={base} x={X0} y={70} scale={S1} deps={p < 0.45 ? 'true' : 'all'} title={tr('one buffer, two barriers per iteration', 'một buffer, hai rào chắn mỗi vòng lặp')} />
      </Stage>
    ),
  },
  {
    clip: 'pipeline-3',
    get label() {
      return tr('Two buffers', 'Hai buffer');
    },
    draw: (p) =>
      p < 0.4 ? (
        <Stage beat={3} headline={tr('a second buffer for the next pair', 'một buffer thứ hai cho cặp kế tiếp')} sub={tr('the barriers that guarded only the reuse of one buffer go', 'các rào chắn chỉ để canh việc dùng lại một buffer biến mất')}>
          <Lanes tl={base} x={X0} y={70} scale={S1} deps="all" /* i18n-ignore: prop value */ falseOpacity={1 - ramp(p, 0.08, 0.36)} title={tr('one buffer, two barriers per iteration', 'một buffer, hai rào chắn mỗi vòng lặp')} />
        </Stage>
      ) : (
        <Stage beat={3} headline={tr('two buffers: one barrier per iteration, still taking turns', 'hai buffer: một rào chắn mỗi vòng lặp, vẫn thay phiên nhau')}
          sub={tr(`still ${units(dbl.total)}: the load of pair i + 1 sits in the next iteration, after compute i`, `vẫn ${units(dbl.total)}: lần nạp cặp i + 1 nằm ở vòng lặp sau, sau phần tính i`)}>
          <Lanes tl={dbl} x={X0} y={70} scale={S1} deps="true" /* i18n-ignore: prop value */ bufs title={tr('two buffers, not rotated', 'hai buffer, chưa xoay vòng lặp')} />
        </Stage>
      ),
  },
  {
    clip: 'pipeline-4',
    get label() {
      return tr('Rotate the loop', 'Xoay vòng lặp');
    },
    draw: (p) => {
      const t = ramp(p, 0.08, 0.85) * pipe.total;
      return (
        <Stage beat={4} headline={tr('iteration i computes pair i and loads pair i + 1', 'vòng lặp i tính cặp i và nạp cặp i + 1')}
          sub={p > 0.86 ? tr(`${units(pipe.total)} instead of ${units(base.total)}: a speed-up of ${speedup(N, 1, 1)}`, `${units(pipe.total)} thay vì ${units(base.total)}: tăng tốc ${speedup(N, 1, 1)} lần`) : undefined}>
          <Lanes tl={base} x={X0} y={70} scale={S1} dim title={tr('before: one buffer, two barriers', 'trước: một buffer, hai rào chắn')} />
          <Lanes tl={pipe} x={X0} y={190} scale={S1} upTo={t} bufs title={tr('two buffers, pipelined (prologue, three iterations, one compute after the loop)', 'hai buffer, pipeline (phần mở đầu, ba vòng lặp, một lần tính sau vòng lặp)')} />
        </Stage>
      );
    },
  },
  {
    clip: 'pipeline-5',
    get label() {
      return tr('curr and next', 'curr và next');
    },
    draw: (p) => {
      const hi = Math.min(pipe.barriers.length - 1, Math.floor(ramp(p, 0.15, 0.95) * pipe.barriers.length));
      const bar = pipe.barriers[hi];
      const text =
        hi === 0
          ? tr('barrier after the prologue: tile 0 has arrived in buffer 0', 'rào chắn sau phần mở đầu: ô 0 đã đến buffer 0')
          : bar.reuse
            ? tr(`barrier ${hi}: tile ${hi} has arrived, and buffer ${(hi - 1) & 1} is free for tile ${hi + 1}`, `rào chắn ${hi}: ô ${hi} đã đến, và buffer ${(hi - 1) & 1} rảnh cho ô ${hi + 1}`)
            : tr(`barrier ${hi}: tile ${hi} has arrived; no tile is left to load`, `rào chắn ${hi}: ô ${hi} đã đến; không còn ô nào để nạp`);
      return (
        <Stage beat={5} headline={tr('the buffers swap roles every iteration', 'hai buffer đổi vai trò sau mỗi vòng lặp')} sub={text}>
          <Lanes tl={pipe} x={X0} y={70} scale={S5} bufs deps="true" /* i18n-ignore: prop value */ hiBarrier={hi} title={tr('two buffers, pipelined', 'hai buffer, pipeline')} />
          {rolesRow(pipe, 200, S5, pipe.total)}
        </Stage>
      );
    },
  },
  {
    clip: 'pipeline-6',
    get label() {
      return tr('Slow loads', 'Nạp chậm');
    },
    draw: (p) => (
      <Stage beat={6} headline={tr('loads three times longer than the compute', 'nạp lâu gấp ba lần phần tính')}
        sub={p > 0.5 ? tr(`${units(slowPipe.total)} instead of ${units(slowBase.total)}: a speed-up of ${speedup(N, 3, 1).toFixed(2)}; the memory lane sets the pace`, `${units(slowPipe.total)} thay vì ${units(slowBase.total)}: tăng tốc ${speedup(N, 3, 1).toFixed(2)} lần; làn bộ nhớ quyết định nhịp độ`) : undefined}>
        <Lanes tl={slowBase} x={X0} y={70} scale={S6} title={tr('one buffer, two barriers', 'một buffer, hai rào chắn')} />
        <Lanes tl={slowPipe} x={X0} y={190} scale={S6} upTo={ramp(p, 0.1, 0.5) * slowPipe.total} title={tr('two buffers, pipelined', 'hai buffer, pipeline')} />
      </Stage>
    ),
  },
];

export function PipelineScene() {
  return <Scene chapter="ch15" title={tr('Loading the next tile while computing this one', 'Nạp ô kế tiếp trong khi tính ô hiện tại')} beats={beats} />;
}
