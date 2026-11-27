import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { FAULT_PAGE, faultStory, type Ev, type Frame, type PageView } from '../../sim/managedmem';

// "One GPU page fault" (section C.8): one managed page through a GPU touch and a later
// CPU touch, under fault and migrate. Every state drawn is a frame of faultStory() in
// src/sim/managedmem.ts: four pages written by the CPU; a kernel reads page 2 (a fault);
// a second kernel reads it again (local); the host synchronises; the CPU writes page 2
// (the reverse fault). Five beats, one narration clip each
// (content/appC/narration/page-fault-N.txt).

const run = faultStory();
const [, , gpuTouch, again, sync, cpuTouch] = run.steps;
const frames = (from: number, to: number) => run.frames.slice(from, to);
const beforeGpu = run.frames[gpuTouch.start - 1];

// The frames each beat walks through, in order.
const BEAT_FRAMES: Frame[][] = [
  frames(gpuTouch.start, gpuTouch.start + 1), // fault
  frames(gpuTouch.start + 1, gpuTouch.start + 2), // unmap from the CPU
  frames(gpuTouch.start + 2, gpuTouch.start + 3), // the move
  [...frames(gpuTouch.start + 3, gpuTouch.end), ...frames(again.start, again.end)], // map, replay, a later local load
  [...frames(sync.start, sync.end), ...frames(cpuTouch.start, cpuTouch.end)], // wait, then the reverse
];
const startOf = (b: number) => (b === 0 ? beforeGpu : BEAT_FRAMES[b - 1][BEAT_FRAMES[b - 1].length - 1]);

const W = 720;
const H = 330;
const BOX = { cpu: { x: 20, y: 70, w: 250, h: 150 }, gpu: { x: 450, y: 70, w: 250, h: 150 } };
const PW = 44;
const PH = 30;
const slot = (side: 'cpu' | 'gpu', i: number) => ({ x: BOX[side].x + 18 + i * 56, y: BOX[side].y + 56 });
const lerp = (a: number, b: number, f: number) => a + (b - a) * f;
const clamp = (v: number) => Math.max(0, Math.min(1, v));

const WHAT: Partial<Record<Ev['e'], (ev: Ev) => string>> = {
  fault: (ev) =>
    tr(
      `${ev.proc === 'gpu' ? 'GPU' : 'CPU'} access to p${ev.page}: no mapping, a page fault; the access waits`,
      `${ev.proc === 'gpu' ? 'GPU' : 'CPU'} truy cập p${ev.page}: không có ánh xạ, một lỗi trang; lần truy cập phải chờ`,
    ),
  unmap: (ev) =>
    tr(
      `the ${ev.proc === 'gpu' ? 'GPU' : 'CPU'} loses its mapping to p${ev.page}`,
      `${ev.proc === 'gpu' ? 'GPU' : 'CPU'} mất ánh xạ tới p${ev.page}`,
    ),
  move: (ev) =>
    tr(
      `p${ev.page} crosses the interconnect (${ev.dir === 'h2d' ? 'to GPU memory' : 'to CPU memory'})`,
      `p${ev.page} đi qua đường kết nối (${ev.dir === 'h2d' ? 'tới bộ nhớ GPU' : 'tới bộ nhớ CPU'})`,
    ),
  map: (ev) =>
    tr(
      `the ${ev.proc === 'gpu' ? 'GPU' : 'CPU'} gains a mapping to p${ev.page}`,
      `${ev.proc === 'gpu' ? 'GPU' : 'CPU'} có được ánh xạ tới p${ev.page}`,
    ),
  replay: (ev) => tr(`the faulting access replays and completes`, `lần truy cập gây lỗi được chạy lại và hoàn tất`),
  local: (ev) =>
    tr(
      `a later ${ev.proc === 'gpu' ? 'kernel' : 'CPU'} load from p${ev.page}: local, no fault`,
      `một lần nạp sau đó của ${ev.proc === 'gpu' ? 'kernel' : 'CPU'} từ p${ev.page}: cục bộ, không lỗi trang`,
    ),
  wait: () => tr('the host synchronises', 'host đồng bộ hóa'),
};

function Mapping({ x, y, who, on }: { x: number; y: number; who: string; on: boolean }) {
  return (
    <g data-map={who.toLowerCase()} data-on={on ? 'yes' : 'no'}>
      <rect x={x} y={y} width={112} height={24} rx="3" fill={on ? 'var(--ok-wash)' : 'var(--page)'} stroke={on ? 'var(--ok)' : 'var(--rule)'} />
      <text x={x + 56} y={y + 16} fontSize="11.5" textAnchor="middle" style={{ fill: on ? 'var(--ok)' : 'var(--ink-3)' }}>
        <Translated vi={<>ánh xạ {who}: {on ? 'có' : 'không'}</>}>
          {who} mapping: {on ? 'yes' : 'none'}
        </Translated>
      </text>
    </g>
  );
}

function Stage({ frame, next, frac, beat, note }: { frame: Frame; next: Frame | null; frac: number; beat: number; note?: string }) {
  const pages = frame.view.m.pages;
  const pg: PageView = pages[FAULT_PAGE];
  const ev = frame.ev;
  const flying = next && next.ev.e === 'move' ? next.ev : null;
  // From the fault until the replay, the access that faulted has not completed.
  const waiting = !!flying || ev.e === 'fault' || ev.e === 'unmap' || ev.e === 'move' || ev.e === 'map';
  const c = frame.counters;
  const desc = flying
    ? WHAT.move!(flying)
    : WHAT[ev.e]?.(ev) ?? (beat === 1
        ? tr(
            `a kernel is about to load from p${FAULT_PAGE}, which lives in CPU memory`,
            `một kernel sắp nạp từ p${FAULT_PAGE}, trang đang nằm trong bộ nhớ CPU`,
          )
        : '');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('One managed page through a GPU page fault', 'Một trang được quản lý đi qua một lỗi trang GPU')} style={{ width: '100%', maxWidth: W, display: 'block' }}
      data-beat={beat} data-event={flying ? 'move' : ev.e} data-proc={(flying ?? ev).proc ?? ''}
      data-loc={flying ? 'moving' : pg.loc} data-cpu-map={pg.cpuMap ? 'yes' : 'no'} data-gpu-map={pg.gpuMap ? 'yes' : 'no'}
      data-h2d={c.h2d} data-d2h={c.d2h} data-gpu-faults={c.gpuFaults} data-cpu-faults={c.cpuFaults}>
      <text x={20} y={26} fontSize="14" fontWeight="620" data-headline="">{desc}</text>
      {(['cpu', 'gpu'] as const).map((side) => (
        <g key={side}>
          <rect x={BOX[side].x} y={BOX[side].y} width={BOX[side].w} height={BOX[side].h} rx="5"
            fill={side === 'gpu' ? 'var(--global-wash)' : 'var(--page)'} fillOpacity={side === 'gpu' ? 0.45 : 1}
            stroke={side === 'gpu' ? 'var(--global)' : 'var(--ink-3)'} />
          <text x={BOX[side].x + 12} y={BOX[side].y + 24} fontSize="12.5" fontWeight="620"
            style={{ fill: side === 'gpu' ? 'var(--global)' : 'var(--ink)' }}>{side === 'cpu' ? tr('CPU memory', 'Bộ nhớ CPU') : tr('GPU memory', 'Bộ nhớ GPU')}</text>
          {pages.map((_, i) => {
            const s = slot(side, i);
            return <rect key={i} x={s.x} y={s.y} width={PW} height={PH} rx="3" fill="none" stroke="var(--rule)" strokeDasharray="3 3" />;
          })}
        </g>
      ))}
      <line x1={BOX.cpu.x + BOX.cpu.w} y1={145} x2={BOX.gpu.x} y2={145} stroke="var(--ink-3)" strokeWidth="6" strokeLinecap="round" opacity="0.35" />{/* i18n-ignore: SVG attribute */}
      <text x={360} y={164} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{tr('interconnect', 'đường kết nối')}</text>
      {pages.map((p, i) => {
        let pos: { x: number; y: number } | null = null;
        let side: 'cpu' | 'gpu' = 'cpu';
        if (flying && i === FAULT_PAGE) {
          const from = slot(flying.dir === 'h2d' ? 'cpu' : 'gpu', i);
          const to = slot(flying.dir === 'h2d' ? 'gpu' : 'cpu', i);
          pos = { x: lerp(from.x, to.x, frac), y: lerp(from.y, to.y, frac) - Math.sin(frac * Math.PI) * 26 };
          side = frac > 0.5 ? (flying.dir === 'h2d' ? 'gpu' : 'cpu') : flying.dir === 'h2d' ? 'cpu' : 'gpu';
        } else if (p.loc === 'cpu' || p.loc === 'gpu') {
          pos = slot(p.loc, i);
          side = p.loc;
        }
        if (!pos) return null;
        const focus = i === FAULT_PAGE;
        return (
          <g key={i} opacity={focus ? 1 : 0.45}>
            <rect x={pos.x} y={pos.y} width={PW} height={PH} rx="3" fill={side === 'gpu' ? 'var(--global-wash)' : 'var(--cell)'}
              stroke={side === 'gpu' ? 'var(--global)' : 'var(--ink-2)'} strokeWidth={focus ? 2.2 : 1} />
            <text x={pos.x + PW / 2} y={pos.y + PH / 2 + 4} fontSize="11" textAnchor="middle">p{i}</text>
          </g>
        );
      })}
      {/* the mappings of the page in focus */}
      <Mapping x={BOX.cpu.x + 12} y={BOX.cpu.y + BOX.cpu.h - 36} who="CPU" on={pg.cpuMap} />
      <Mapping x={BOX.gpu.x + 12} y={BOX.gpu.y + BOX.gpu.h - 36} who="GPU" on={pg.gpuMap} />
      {/* the access that faulted: a kernel load in beats 1 to 4, a CPU store in beat 5 */}
      {waiting || ev.e === 'replay' || ev.e === 'local' ? (
        <g data-access={waiting ? 'waiting' : 'done'}>
          <rect x={beat < 5 ? 560 : 140} y={42} width={130} height={22} rx="11"
            fill={waiting ? 'var(--fail-wash)' : 'var(--ok-wash)'} stroke={waiting ? 'var(--fail)' : 'var(--ok)'} />
          <text x={(beat < 5 ? 560 : 140) + 65} y={57} fontSize="11" textAnchor="middle" style={{ fill: waiting ? 'var(--fail)' : 'var(--ok)' }}>
            {beat < 5 ? tr('kernel load', 'kernel nạp') : tr('CPU store', 'CPU ghi')}: {waiting ? tr('waiting', 'đang chờ') : tr('done', 'xong')}
          </text>
        </g>
      ) : null}
      {(['to GPU', 'to CPU', 'GPU faults', 'CPU faults'] as const).map((label, k) => ( // i18n-ignore: data keys; the text drawn is translated below
        <g key={label} data-counter={label} transform={`translate(${30 + k * 120}, 262)`}>
          <text fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi={k === 0 ? 'tới GPU' : k === 1 ? 'tới CPU' : k === 2 ? 'lỗi trang GPU' : 'lỗi trang CPU'}>{label}</Translated></text>
          <text y={20} fontSize="17" fontWeight="650" data-value="">{[c.h2d, c.d2h, c.gpuFaults, c.cpuFaults][k]}</text>
        </g>
      ))}
      {note ? <text x={20} y={320} fontSize="12" data-note="">{note}</text> : null}
    </svg>
  );
}


function drawBeat(b: number, p: number, note?: (done: boolean) => string | undefined) {
  const fs = BEAT_FRAMES[b];
  const n = fs.length;
  const q = clamp((p - 0.06) / 0.84);
  const t = q * n;
  const k = Math.min(n, Math.floor(t));
  const frame = k === 0 ? startOf(b) : fs[k - 1];
  const next = k < n ? fs[k] : null;
  // The move beat: the page travels during the whole beat.
  if (b === 2) {
    const f = clamp((p - 0.1) / 0.75);
    return <Stage beat={3} frame={f >= 1 ? fs[0] : startOf(2)} next={f >= 1 ? null : fs[0]} frac={f} note={note?.(f >= 1)} />;
  }
  return <Stage beat={b + 1} frame={frame} next={next} frac={t - k} note={note?.(k >= n)} />;
}

const beats: Beat[] = [
  {
    clip: 'page-fault-1',
    get label() {
      return tr('The fault', 'Lỗi trang');
    },
    draw: (p) => drawBeat(0, p),
  },
  {
    clip: 'page-fault-2',
    get label() {
      return tr('Unmap', 'Gỡ ánh xạ');
    },
    draw: (p) =>
      drawBeat(1, p, (d) =>
        d ? tr('Neither processor holds a mapping while the page moves.', 'Không bộ xử lý nào giữ ánh xạ trong lúc trang di chuyển.') : undefined,
      ),
  },
  {
    clip: 'page-fault-3',
    get label() {
      return tr('The move', 'Di chuyển');
    },
    draw: (p) =>
      drawBeat(2, p, (d) =>
        d ? tr('NVIDIA: the work is proportional to the page size.', 'NVIDIA: khối lượng công việc tỉ lệ với kích thước trang.') : undefined,
      ),
  },
  {
    clip: 'page-fault-4',
    get label() {
      return tr('Map and replay', 'Ánh xạ và chạy lại');
    },
    draw: (p) =>
      drawBeat(3, p, (d) =>
        d ? tr('One fault paid once; later loads are local.', 'Một lỗi trang chỉ trả giá một lần; các lần nạp sau là cục bộ.') : undefined,
      ),
  },
  {
    clip: 'page-fault-5',
    get label() {
      return tr('In reverse', 'Chiều ngược lại');
    },
    draw: (p) =>
      drawBeat(4, p, (d) =>
        d
          ? tr('Touch it from both sides in turn and it moves every time.', 'Lần lượt chạm vào nó từ cả hai phía thì lần nào nó cũng di chuyển.')
          : undefined,
      ),
  },
];

export function PageFaultScene() {
  return <Scene chapter="appC" title={tr('One GPU page fault', 'Một lỗi trang GPU')} beats={beats} />;
}
