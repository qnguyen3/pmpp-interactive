import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { moved, runFig, zeroCounters, type Counters, type Ev, type Frame, type Run, type View } from '../../sim/managedmem';

// "Two things called unified memory" (section C.6): Fig. C.1's managed sortfile run by
// src/sim/managedmem.ts in three modes, limited (bulk at launch), full (fault and
// migrate) and one memory (Apple silicon). Every page position and counter below is a
// frame of those runs; nothing is drawn by hand. Six beats, one narration clip each
// (content/appC/narration/two-unified-N.txt).

const N = 8;
const bulk = runFig('managed', 'bulk');
const full = runFig('managed', 'migrate');
const one = runFig('managed', 'one');
const explicit = runFig('explicit', 'migrate');
const line = (role: string) => full.lines.findIndex((l) => l.role === role);

/** The frames of some steps of a run, and the view just before them. */
function slice(run: Run, stepIdx: number[]): { before: View; frames: Frame[]; startCounters: Counters } {
  const first = run.steps[stepIdx[0]];
  const before = first.start > 0 ? run.frames[first.start - 1].view : stepIdx[0] > 0 ? run.steps[stepIdx[0] - 1].view : run.initial;
  const startCounters = stepIdx[0] > 0 ? run.steps[stepIdx[0] - 1].counters : zeroCounters();
  const frames = stepIdx.flatMap((s) => run.frames.slice(run.steps[s].start, run.steps[s].end));
  return { before, frames, startCounters };
}

const clamp = (v: number) => Math.max(0, Math.min(1, v));

/** Where things stand at fraction p of a slice: the view, the counters, and a page in flight. */
function at(sl: ReturnType<typeof slice>, p: number, lead = 0.08, tail = 0.12) {
  const n = sl.frames.length;
  const q = clamp((p - lead) / (1 - lead - tail));
  const t = q * n;
  const k = Math.min(n, Math.floor(t));
  const view = k === 0 ? sl.before : sl.frames[k - 1].view;
  const counters = k === 0 ? sl.startCounters : sl.frames[k - 1].counters;
  const active = k === 0 ? false : sl.frames[k - 1].active;
  const next = k < n ? sl.frames[k] : null;
  const flying = next && next.ev.e === 'move' ? { page: next.ev.page, dir: next.ev.dir!, frac: t - k } : null;
  const ev = k > 0 ? sl.frames[k - 1].ev : null;
  return { view, counters, active, flying, ev, done: k >= n };
}

// Geometry.
const W = 720;
const H = 330;
const BOX = { cpu: { x: 20, y: 64, w: 250, h: 170 }, gpu: { x: 450, y: 64, w: 250, h: 170 } };
const PW = 44;
const PH = 30;
const slot = (side: 'cpu' | 'gpu', i: number) => {
  const b = BOX[side];
  return { x: b.x + 22 + (i % 4) * 54, y: b.y + 52 + Math.floor(i / 4) * 46 };
};
const lerp = (a: number, b: number, f: number) => a + (b - a) * f;

function PageBox({ x, y, i, kind, hot }: { x: number; y: number; i: number; kind: 'cpu' | 'gpu' | 'one'; hot?: boolean }) {
  const fill = kind === 'gpu' ? 'var(--global-wash)' : 'var(--cell)';
  const stroke = kind === 'gpu' ? 'var(--global)' : 'var(--ink-2)';
  return (
    <g>
      <rect x={x} y={y} width={PW} height={PH} rx="3" fill={fill} stroke={stroke} strokeWidth={hot ? 2.4 : 1.2} />
      <text x={x + PW / 2} y={y + PH / 2 + 4} fontSize="11" textAnchor="middle">p{i}</text>
    </g>
  );
}

function Counts({ c, x, y }: { c: Counters; x: number; y: number }) {
  const items: [string, number][] = [
    ['to GPU', c.h2d], // i18n-ignore: data key; the text drawn is translated below
    ['to CPU', c.d2h], // i18n-ignore: data key; the text drawn is translated below
    ['GPU faults', c.gpuFaults], // i18n-ignore: data key; the text drawn is translated below
    ['CPU faults', c.cpuFaults], // i18n-ignore: data key; the text drawn is translated below
  ];
  return (
    <g>
      {items.map(([label, v], i) => (
        <g key={label} data-counter={label} transform={`translate(${x + i * 120}, ${y})`}>
          <text fontSize="11" style={{ fill: 'var(--ink-2)' }}>
            <Translated vi={i === 0 ? 'tới GPU' : i === 1 ? 'tới CPU' : i === 2 ? 'lỗi trang GPU' : 'lỗi trang CPU'}>{label}</Translated>
          </text>
          <text y={20} fontSize="17" fontWeight="650" data-value="">{v}</text>
        </g>
      ))}
    </g>
  );
}

function TwoBoxes({
  view,
  counters,
  flying,
  headline,
  note,
  locked,
  beat,
}: {
  view: View;
  counters: Counters;
  flying: { page: number; dir: 'h2d' | 'd2h'; frac: number } | null;
  headline: ReactNode;
  note?: ReactNode;
  locked?: boolean;
  beat: number;
}) {
  const pages = view.data?.pages ?? [];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('CPU memory and GPU memory with the pages of the buffer', 'Bộ nhớ CPU và bộ nhớ GPU cùng các trang của buffer')} style={{ width: '100%', maxWidth: W, display: 'block' }}
      data-beat={beat} data-h2d={counters.h2d} data-d2h={counters.d2h}>
      <text x={20} y={26} fontSize="14" fontWeight="620" data-headline="">{headline}</text>
      {(['cpu', 'gpu'] as const).map((side) => (
        <g key={side}>
          <rect x={BOX[side].x} y={BOX[side].y} width={BOX[side].w} height={BOX[side].h} rx="5"
            fill={side === 'gpu' ? 'var(--global-wash)' : 'var(--page)'} fillOpacity={side === 'gpu' ? 0.45 : 1}
            stroke={side === 'gpu' ? 'var(--global)' : 'var(--ink-3)'} />
          <text x={BOX[side].x + 12} y={BOX[side].y + 24} fontSize="12.5" fontWeight="620"
            style={{ fill: side === 'gpu' ? 'var(--global)' : 'var(--ink)' }}>{side === 'cpu' ? tr('CPU memory', 'Bộ nhớ CPU') : tr('GPU memory', 'Bộ nhớ GPU')}</text>
          {Array.from({ length: N }, (_, i) => {
            const s = slot(side, i);
            return <rect key={i} x={s.x} y={s.y} width={PW} height={PH} rx="3" fill="none" stroke="var(--rule)" strokeDasharray="3 3" />;
          })}
        </g>
      ))}
      {/* the link */}
      <line x1={BOX.cpu.x + BOX.cpu.w} y1={150} x2={BOX.gpu.x} y2={150} stroke="var(--ink-3)" strokeWidth="6" strokeLinecap="round" opacity="0.35" />{/* i18n-ignore: SVG attribute */}
      <text x={360} y={228} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{tr('interconnect', 'đường kết nối')}</text>
      {locked ? (
        <g data-locked="">
          <rect x={BOX.cpu.x} y={BOX.cpu.y} width={BOX.cpu.w} height={BOX.cpu.h} rx="5" fill="var(--fail-wash)" stroke="var(--fail)" strokeDasharray="5 3" />
          <text x={BOX.cpu.x + BOX.cpu.w / 2} y={BOX.cpu.y + BOX.cpu.h - 28} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--fail)' }}>
            <Translated vi="CPU: không được chạm vào managed memory">CPU: hands off managed memory</Translated>
          </text>
          <text x={BOX.cpu.x + BOX.cpu.w / 2} y={BOX.cpu.y + BOX.cpu.h - 12} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--fail)' }}>
            <Translated vi="cho đến khi đồng bộ hóa">until the synchronise</Translated>
          </text>
        </g>
      ) : null}
      {pages.map((pg, i) => {
        if (flying && flying.page === i) {
          const from = slot(flying.dir === 'h2d' ? 'cpu' : 'gpu', i);
          const to = slot(flying.dir === 'h2d' ? 'gpu' : 'cpu', i);
          return (
            <g key={i} data-page={i} data-where="moving">
              <PageBox x={lerp(from.x, to.x, flying.frac)} y={lerp(from.y, to.y, flying.frac)} i={i} kind={flying.frac > 0.5 ? (flying.dir === 'h2d' ? 'gpu' : 'cpu') : flying.dir === 'h2d' ? 'cpu' : 'gpu'} hot />
            </g>
          );
        }
        if (pg.loc === 'cpu' || pg.loc === 'gpu') {
          const s = slot(pg.loc, i);
          return <g key={i} data-page={i} data-where={pg.loc}><PageBox x={s.x} y={s.y} i={i} kind={pg.loc} /></g>;
        }
        return <g key={i} data-page={i} data-where={pg.loc} />;
      })}
      <Counts c={counters} x={30} y={266} />
      {note ? <text x={20} y={322} fontSize="12" data-note="">{note}</text> : null}
    </svg>
  );
}

function OneBox({ view, waited, headline, note }: { view: View; waited: boolean; headline: ReactNode; note?: ReactNode }) {
  const pages = view.data?.pages ?? [];
  const box = { x: 120, y: 92, w: 480, h: 130 };
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('One memory shared by the CPU and the GPU', 'Một bộ nhớ dùng chung cho CPU và GPU')} style={{ width: '100%', maxWidth: W, display: 'block' }}
      data-beat={5} data-h2d={0} data-d2h={0} data-waited={waited ? 'yes' : 'no'}>
      <text x={20} y={26} fontSize="14" fontWeight="620" data-headline="">{headline}</text>
      {(['CPU', 'GPU'] as const).map((who, k) => (
        <g key={who}>
          <rect x={k === 0 ? 150 : 450} y={44} width={120} height={28} rx="4" fill="var(--page)" stroke="var(--ink-3)" />
          <text x={k === 0 ? 210 : 510} y={63} fontSize="12" textAnchor="middle" fontWeight="600">{who}</text>
          <line x1={k === 0 ? 210 : 510} y1={72} x2={k === 0 ? 260 : 460} y2={box.y} stroke="var(--ink-3)" />
        </g>
      ))}
      <rect x={box.x} y={box.y} width={box.w} height={box.h} rx="5" fill="var(--page)" stroke="var(--ink-3)" />
      <text x={box.x + 12} y={box.y + 22} fontSize="12.5" fontWeight="620"><Translated vi="một bộ nhớ hệ thống, buffer dùng chung">one system memory, shared buffer</Translated></text>
      {pages.map((pg, i) => {
        const x = box.x + 30 + (i % 8) * 54;
        const y = box.y + 50;
        return (
          <g key={i} data-page={i} data-where={pg.loc}>
            {pg.loc === 'one' ? <PageBox x={x} y={y} i={i} kind="one" /> : <rect x={x} y={y} width={PW} height={PH} rx="3" fill="none" stroke="var(--rule)" strokeDasharray="3 3" />}
          </g>
        );
      })}
      <Counts c={{ h2d: 0, d2h: 0, gpuFaults: 0, cpuFaults: 0, remote: 0, invalidations: 0, evictions: 0, copiedInMemory: 0 }} x={30} y={266} />
      {note ? <text x={20} y={322} fontSize="12" data-note="">{note}</text> : null}
    </svg>
  );
}

const fread = line('read-file');
const sortLine = line('sort');
const syncLine = line('sync');
const useLine = line('use');

const s1 = slice(full.run, [full.stepOf[fread]]);
const s2 = slice(bulk.run, [bulk.stepOf[sortLine]]);
const s3 = slice(full.run, [full.stepOf[sortLine]]);
const s4 = slice(full.run, [full.stepOf[syncLine], full.stepOf[useLine]]);
const s5 = slice(one.run, [one.stepOf[fread], one.stepOf[sortLine], one.stepOf[syncLine]]);

/** What the last event did to its page, in a few words. */
function evText(ev: Ev | null, flying: { page: number } | null, who: 'gpu' | 'cpu'): string {
  if (flying) return tr(`p${flying.page} crosses the interconnect`, `p${flying.page} đi qua đường kết nối`);
  if (!ev || ev.page < 0) return '';
  const other = who === 'gpu' ? 'CPU' : 'GPU';
  switch (ev.e) {
    case 'fault': return tr(`p${ev.page} faults: no ${who.toUpperCase()} mapping`, `p${ev.page} gây lỗi trang: không có ánh xạ ${who.toUpperCase()}`);
    case 'unmap': return tr(`p${ev.page}: the ${other} loses its mapping`, `p${ev.page}: ${other} mất ánh xạ`);
    case 'move': return tr(`p${ev.page} has crossed`, `p${ev.page} đã đi qua`);
    case 'map': return tr(`p${ev.page}: the ${who.toUpperCase()} gains a mapping`, `p${ev.page}: ${who.toUpperCase()} có được ánh xạ`);
    case 'replay': return tr(`p${ev.page}: the access replays and completes`, `p${ev.page}: lần truy cập được chạy lại và hoàn tất`);
    default: return '';
  }
}

const beats: Beat[] = [
  {
    clip: 'two-unified-1',
    get label() {
      return tr('First touch', 'Lần chạm đầu tiên');
    },
    draw: (p) => {
      const f = at(s1, p);
      const placed = f.view.data.pages.filter((x) => x.loc === 'cpu').length;
      return (
        <TwoBoxes beat={1} view={f.view} counters={f.counters} flying={null}
          headline={
            placed === 0
              ? tr('cudaMallocManaged: one pointer, no page placed yet', 'cudaMallocManaged: một con trỏ, chưa trang nào được đặt')
              : tr(`fread writes the file: ${placed} of ${N} pages placed in CPU memory`, `fread ghi tệp: ${placed} trên ${N} trang được đặt trong bộ nhớ CPU`)
          }
          note={f.done ? tr('Each page sits where it was first touched.', 'Mỗi trang nằm ở nơi nó được chạm vào lần đầu.') : undefined} />
      );
    },
  },
  {
    clip: 'two-unified-2',
    get label() {
      return tr('Limited mode', 'Chế độ giới hạn');
    },
    draw: (p) => {
      const f = at(s2, p);
      return (
        <TwoBoxes beat={2} view={f.view} counters={f.counters} flying={f.flying} locked={p > 0.08}
          headline={tr(
            `limited: at the launch, every managed page moves (${f.counters.h2d} of ${N})`,
            `giới hạn: lúc khởi chạy, mọi trang được quản lý đều di chuyển (${f.counters.h2d} trên ${N})`,
          )}
          note={
            f.done
              ? tr('CUDA 6 on Kepler and Maxwell; Windows, WSL and some Tegra systems today.', 'CUDA 6 trên Kepler và Maxwell; ngày nay là Windows, WSL và một số hệ thống Tegra.')
              : undefined
          } />
      );
    },
  },
  {
    clip: 'two-unified-3',
    get label() {
      return tr('Full mode', 'Chế độ đầy đủ');
    },
    draw: (p) => {
      const f = at(s3, p);
      return (
        <TwoBoxes beat={3} view={f.view} counters={f.counters} flying={f.flying}
          headline={
            f.counters.gpuFaults === 0
              ? tr('full: the kernel starts, nothing has moved', 'đầy đủ: kernel bắt đầu, chưa có gì di chuyển')
              : tr(`full: ${evText(f.ev, f.flying, 'gpu')}`, `đầy đủ: ${evText(f.ev, f.flying, 'gpu')}`)
          }
          note={
            f.done
              ? tr('One fault per page the kernel touches; an untouched page would stay put.', 'Mỗi trang kernel chạm vào gây một lỗi trang; trang không bị chạm sẽ nằm yên.')
              : undefined
          } />
      );
    },
  },
  {
    clip: 'two-unified-4',
    get label() {
      return tr('Back to the CPU', 'Trở về CPU');
    },
    draw: (p) => {
      const f = at(s4, p);
      const back = f.counters.d2h;
      return (
        <TwoBoxes beat={4} view={f.view} counters={f.counters} flying={f.flying}
          headline={
            f.counters.cpuFaults === 0
              ? tr('cudaDeviceSynchronize, then use_data reads on the CPU', 'cudaDeviceSynchronize, rồi use_data đọc trên CPU')
              : tr(`use_data: ${evText(f.ev, f.flying, 'cpu')} (${back} of ${N} back)`, `use_data: ${evText(f.ev, f.flying, 'cpu')} (${back} trên ${N} đã về)`)
          }
          note={
            f.done
              ? tr(
                  `${moved(full.run.counters)} pages crossed in all. Two explicit copies move ${moved(explicit.run.counters)}.`,
                  `Tổng cộng ${moved(full.run.counters)} trang đã đi qua. Hai lần sao chép tường minh di chuyển ${moved(explicit.run.counters)}.`,
                )
              : undefined
          } />
      );
    },
  },
  {
    clip: 'two-unified-5',
    get label() {
      return tr('One memory', 'Một bộ nhớ');
    },
    draw: (p) => {
      const f = at(s5, p);
      const waited = !!f.done;
      return (
        <OneBox view={f.view} waited={waited}
          headline={tr('Apple silicon: the CPU writes, the GPU reads in place', 'Apple silicon: CPU ghi, GPU đọc tại chỗ')}
          note={
            waited
              ? tr(
                  'Nothing moved. Still owed: wait for the command buffer before use_data.',
                  'Không có gì di chuyển. Vẫn còn phải làm: chờ command buffer xong trước use_data.',
                )
              : undefined
          } />
      );
    },
  },
  {
    clip: 'two-unified-6',
    get label() {
      return tr('Side by side', 'Đặt cạnh nhau');
    },
    draw: (p) => {
      const cols: { name: string; sub: string; n: number; owe: string }[] = [
        {
          name: 'CUDA, limited', // i18n-ignore: data key; the text drawn is translated below
          sub: tr('two memories, bulk at launch', 'hai bộ nhớ, chuyển hàng loạt lúc khởi chạy'),
          n: moved(bulk.run.counters),
          owe: tr('ordering, and no CPU touch while a kernel runs', 'thứ tự, và CPU không được chạm vào khi kernel đang chạy'),
        },
        {
          name: 'CUDA, full', // i18n-ignore: data key; the text drawn is translated below
          sub: tr('two memories, fault and migrate', 'hai bộ nhớ, lỗi trang rồi di chuyển'),
          n: moved(full.run.counters),
          owe: tr('ordering', 'thứ tự'),
        },
        {
          name: 'Apple silicon', // i18n-ignore: data key; the text drawn is translated below
          sub: tr('one memory, shared buffer', 'một bộ nhớ, buffer dùng chung'),
          n: moved(one.run.counters),
          owe: tr('ordering', 'thứ tự'),
        },
      ];
      return (
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Pages moved in three modes', 'Số trang di chuyển trong ba chế độ')} style={{ width: '100%', maxWidth: W, display: 'block' }} data-beat={6}>
          <text x={20} y={26} fontSize="14" fontWeight="620" data-headline=""><Translated vi="một từ, hai cơ chế">one word, two mechanisms</Translated></text>
          {cols.map((c, k) => {
            const x = 20 + k * 233;
            const o = clamp((p - k * 0.15) / 0.25);
            return (
              <g key={c.name} opacity={o} data-mode={c.name} data-moved={c.n}>
                <rect x={x} y={50} width={215} height={230} rx="5" fill="var(--page)" stroke="var(--ink-3)" />
                <text x={x + 14} y={76} fontSize="13" fontWeight="650">
                  <Translated vi={k === 0 ? 'CUDA, giới hạn' : k === 1 ? 'CUDA, đầy đủ' : 'Apple silicon'}>{c.name}</Translated>
                </text>
                <text x={x + 14} y={95} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{c.sub}</text>
                <text x={x + 14} y={150} fontSize="34" fontWeight="650">{c.n}</text>
                <text x={x + 14} y={172} fontSize="11.5"><Translated vi="trang đã đi qua đường kết nối">pages crossed the link</Translated></text>
                <text x={x + 14} y={214} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi="vẫn còn phải làm:">still owed:</Translated></text>
                <foreignObject x={x + 12} y={220} width={195} height={56}>
                  <div style={{ fontFamily: 'var(--sans)', fontSize: 11.5, lineHeight: 1.3, color: 'var(--ink)' }}>{c.owe}</div>
                </foreignObject>
              </g>
            );
          })}
          <text x={20} y={316} fontSize="12" opacity={clamp((p - 0.6) / 0.2)} data-note="">
            <Translated vi="Unified Memory của CUDA di chuyển trang giữa hai bộ nhớ; unified memory của Apple là một bộ nhớ duy nhất.">
              CUDA's Unified Memory migrates pages between two memories; Apple's unified memory is one memory.
            </Translated>
          </text>
        </svg>
      );
    },
  },
];

export function TwoUnifiedScene() {
  return <Scene chapter="appC" title={tr('Two things called unified memory', 'Hai thứ cùng được gọi là unified memory')} beats={beats} />;
}
