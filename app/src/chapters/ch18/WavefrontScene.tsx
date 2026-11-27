import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import {
  MAZE,
  MAZE_BLOCKED,
  MAZE_GOAL,
  MAZE_H,
  MAZE_ROOT,
  MAZE_W,
  backtrace,
  depth,
  levels,
  parentChoices,
  runBFS,
} from '../../sim/bfs';
import { Translated, tr } from '../../lib/i18n';

// "The wavefront": BFS on a 15 × 15 wiring grid (our own layout in the spirit of Fig. 18.5),
// first as levels, then as the push kernel's threads, then as the frontier kernel's, then as
// launches, then as the walk back. Every number and every cell drawn comes from
// src/sim/bfs.ts. Five beats, one narration clip each (content/ch18/narration/wavefront-N.txt).
const LV = levels(MAZE, MAZE_ROOT);
const D = depth(LV); // 30
const SIZES = Array.from({ length: D + 1 }, (_, L) => LV.filter((x) => x === L).length);
// One thread per cell, without rounding up to whole blocks (blockSize 1).
const PUSH = runBFS(MAZE, { variant: 'push', root: MAZE_ROOT, blockSize: 1, warpWidth: 1 });
const FRONT = runBFS(MAZE, { variant: 'frontier', root: MAZE_ROOT, blockSize: 1, warpWidth: 1 });
const CALLS = PUSH.calls.length; // D + 1
const PATH = backtrace(MAZE, MAZE_ROOT, MAZE_GOAL)!;
const CHOICES = parentChoices(MAZE, MAZE_ROOT);
const N = MAZE.n;
const M = MAZE.edges.length;

const C = 21;
const GX = 14;
const GY = 30;
const RX = 352;
const cx = (v: number) => GX + MAZE.cell![v].c * C + C / 2;
const cy = (v: number) => GY + MAZE.cell![v].r * C + C / 2;
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const sum = (a: number[]) => a.reduce((s, x) => s + x, 0);

type CellLook = 'free' | 'visited' | 'wave';
interface Frame {
  beat: number;
  look: (v: number) => CellLook;
  /** Threads: working (on the last wavefront) or idle. */
  dot?: (v: number) => 'work' | 'idle' | null;
  number?: (v: number) => string | null;
  edgesAt?: number | null;
  path?: number[];
  ties?: Set<number>;
  panel: ReactNode;
  data: Record<string, string | number>;
}

function Grid(f: Frame) {
  const blocked = MAZE_BLOCKED.map(([r, c]) => (
    <rect key={`b${r},${c}`} x={GX + c * C} y={GY + r * C} width={C - 1} height={C - 1} rx="1.5" fill="var(--ink)" />
  ));
  return (
    <g>
      {MAZE.cell!.map(({ r, c }, v) => {
        const k = f.look(v);
        return (
          <rect key={v} x={GX + c * C} y={GY + r * C} width={C - 1} height={C - 1} rx="1.5"
            fill={k === 'wave' ? 'var(--global)' : k === 'visited' ? 'var(--global-wash)' : 'var(--cell)'}
            stroke={k === 'free' ? 'var(--rule)' : 'var(--global)'} strokeWidth={k === 'free' ? 0.8 : 0.6} />
        );
      })}
      {blocked}
      {f.edgesAt != null
        ? [[-1, 0], [1, 0], [0, -1], [0, 1]].map(([dr, dc], i) => (
            <line key={i} x1={cx(f.edgesAt!)} y1={cy(f.edgesAt!)} x2={cx(f.edgesAt!) + dc * C} y2={cy(f.edgesAt!) + dr * C} stroke="var(--ink)" strokeWidth="2" />
          ))
        : null}
      {f.number
        ? MAZE.cell!.map((_, v) => {
            const t = f.number!(v);
            return t ? (
              <text key={'n' + v} x={cx(v)} y={cy(v) + 3} fontSize="8.5" textAnchor="middle"
                style={{ fill: f.look(v) === 'wave' ? 'var(--page)' : 'var(--ink)' }}>{t}</text>
            ) : null;
          })
        : null}
      {f.dot
        ? MAZE.cell!.map((_, v) => {
            const k = f.dot!(v);
            return k ? (
              <circle key={'d' + v} cx={cx(v)} cy={cy(v)} r={k === 'work' ? 3.4 : 1.9} data-dot={k}
                fill={k === 'work' ? 'var(--ink)' : 'var(--ink-3)'} opacity={k === 'work' ? 1 : 0.7} />
            ) : null;
          })
        : null}
      {f.path && f.path.length > 1 ? (
        <polyline points={f.path.map((v) => `${cx(v)},${cy(v)}`).join(' ')} fill="none" stroke="var(--ink)" strokeWidth="3.2" strokeLinejoin="round" strokeLinecap="round" /> // i18n-ignore: SVG stroke style values
      ) : null}
      {f.ties ? [...f.ties].map((v) => <circle key={'t' + v} cx={cx(v)} cy={cy(v)} r={7.5} fill="none" stroke="var(--ink)" strokeWidth="1.3" strokeDasharray="2 2" data-tie={v} />) : null}
      {[MAZE_ROOT, MAZE_GOAL].map((v, i) => (
        <g key={'term' + i}>
          <circle cx={cx(v)} cy={cy(v)} r={6} fill="var(--ink)" stroke="var(--page)" strokeWidth="1.5" />
          <text x={cx(v)} y={cy(v) + 3} fontSize="8" fontWeight="700" textAnchor="middle" style={{ fill: 'var(--page)' }}>{i ? 'T' : 'S'}</text>
        </g>
      ))}
    </g>
  );
}

function Stage(f: Frame) {
  return (
    <svg viewBox="0 0 720 370" role="img" aria-label={tr('Breadth-first search on a wiring grid', 'Tìm kiếm theo chiều rộng trên một lưới đi dây')} style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      <g data-beat={f.beat} {...Object.fromEntries(Object.entries(f.data).map(([k, v]) => ['data-' + k, String(v)]))}>
        <text x={GX} y={18} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
          <Translated vi={<>lưới đi dây {MAZE_W} × {MAZE_H}: {N} ô trống, {M} cạnh (mỗi bước tính cả hai chiều); S là nguồn, T là đích</>}>
            {MAZE_W} × {MAZE_H} wiring grid: {N} free blocks, {M} edges (each step counted both ways); S source, T target
          </Translated>
        </text>
        <Grid {...f} />
        <g transform={`translate(${RX}, 0)`}>{f.panel}</g>
      </g>
    </svg>
  );
}

const Line = ({ y, children, bold = false, data }: { y: number; children: ReactNode; bold?: boolean; data?: string }) => (
  <text x={0} y={y} fontSize={bold ? 13 : 12} fontWeight={bold ? 650 : 400} data-line={data}>{children}</text>
);

/** Bars of cells per level, up to level k. */
function LevelBars({ upto, y }: { upto: number; y: number }) {
  const h = 6;
  return (
    <g>
      <text x={0} y={y - 96} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi="số ô mỗi mức">cells per level</Translated></text>
      {SIZES.map((s, L) =>
        L <= upto ? <rect key={L} x={L * 11} y={y - s * h} width={9} height={s * h} fill={L === upto ? 'var(--global)' : 'var(--global-wash)'} stroke="var(--global)" strokeWidth="0.6" /> : null,
      )}
      <line x1={0} y1={y + 0.5} x2={(D + 1) * 11} y2={y + 0.5} stroke="var(--rule)" />
      <text x={0} y={y + 14} fontSize="10" style={{ fill: 'var(--ink-2)' }}><Translated vi="mức 0">level 0</Translated></text>
      <text x={D * 11 - 8} y={y + 14} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{D}</text>
    </g>
  );
}

const visitedUpTo = (k: number) => (v: number): CellLook => (LV[v] < k ? 'visited' : LV[v] === k ? 'wave' : 'free');
const callOf = (p: number) => 1 + Math.min(D, Math.floor(ramp(p, 0.04, 0.94) * (D + 1)));

const beats: Beat[] = [
  {
    clip: 'wavefront-1',
    get label() { return tr('Blocks and wires', 'Ô và dây'); },
    draw: (p) => {
      const k = p < 0.24 ? 0 : Math.min(D, Math.floor(ramp(p, 0.24, 0.96) * (D + 1)));
      return (
        <Stage
          beat={1}
          look={visitedUpTo(k)}
          number={(v) => (LV[v] === k && k > 0 ? String(k) : null)}
          edgesAt={p < 0.24 ? MAZE_ROOT : null}
          data={{ level: k, wave: SIZES[k] }}
          panel={
            <>
              <Line y={40} bold>{p < 0.24 ? tr('a free block is a vertex, a wire step an edge', 'một ô trống là một đỉnh, một bước dây là một cạnh') : tr(`level ${k}: ${SIZES[k]} block${SIZES[k] === 1 ? '' : 's'}`, `mức ${k}: ${SIZES[k]} ô`)}</Line>
              <Line y={60}>{k > 0 ? tr(`the wavefront, labelled ${k} by the search`, `mặt sóng, được phép tìm kiếm gán nhãn ${k}`) : tr('the search starts at S, level 0', 'phép tìm kiếm bắt đầu tại S, mức 0')}</Line>
              <LevelBars upto={k} y={210} />
            </>
          }
        />
      );
    },
  },
  {
    clip: 'wavefront-2',
    get label() { return tr('A thread per cell', 'Một luồng cho mỗi ô'); },
    draw: (p) => {
      const c = callOf(p);
      const cur = PUSH.calls[c - 1];
      const done = PUSH.calls.slice(0, c);
      const working = cur.pastCheck;
      return (
        <Stage
          beat={2}
          look={visitedUpTo(c)}
          dot={(v) => (LV[v] === c - 1 ? 'work' : 'idle')}
          data={{ call: c, launched: cur.items, working, idle: cur.items - working, total: sum(done.map((x) => x.items)) }}
          panel={
            <>
              <Line y={40} bold><Translated vi={<>push, lần gọi {c} trên {CALLS}</>}>push, call {c} of {CALLS}</Translated></Line>
              <Line y={62}><Translated vi={<>{cur.items} luồng được khởi chạy, mỗi ô trống một luồng</>}>{cur.items} threads launched, one per free block</Translated></Line>
              <Line y={82}><Translated vi={<>{working} luồng làm việc: nằm trên mặt sóng vừa rồi</>}>{working} working: on the last wavefront</Translated></Line>
              <Line y={102}><Translated vi={<>{cur.items - working} luồng rảnh: kiểm tra ô của mình rồi dừng</>}>{cur.items - working} idle: check their block and stop</Translated></Line>
              <Line y={136} bold><Translated vi={<>đến giờ: {sum(done.map((x) => x.items))} luồng đã khởi chạy</>}>so far: {sum(done.map((x) => x.items))} threads launched</Translated></Line>
              <Line y={156}><Translated vi={<>{sum(done.map((x) => x.pastCheck))} luồng trong số đó có việc</>}>{sum(done.map((x) => x.pastCheck))} of them had work</Translated></Line>
              <circle cx={6} cy={190} r={3.4} fill="var(--ink)" />
              <text x={16} y={194} fontSize="11"><Translated vi="một luồng có việc">a thread with work</Translated></text>
              <circle cx={6} cy={210} r={1.9} fill="var(--ink-3)" opacity={0.7} />
              <text x={16} y={214} fontSize="11"><Translated vi="một luồng dừng ngay">a thread that stops at once</Translated></text>
            </>
          }
        />
      );
    },
  },
  {
    clip: 'wavefront-3',
    get label() { return tr('A thread per frontier cell', 'Một luồng cho mỗi ô của frontier'); },
    draw: (p) => {
      const c = callOf(p);
      const cur = FRONT.calls[c - 1];
      const done = FRONT.calls.slice(0, c);
      const pushDone = PUSH.calls.slice(0, c);
      const fw = sum(done.map((x) => x.vertexWork + x.edgeWork));
      const pw = sum(pushDone.map((x) => x.vertexWork + x.edgeWork));
      const scale = 330 / PUSH.totals.work;
      return (
        <Stage
          beat={3}
          look={visitedUpTo(c)}
          dot={(v) => (LV[v] === c - 1 ? 'work' : null)}
          data={{ call: c, threads: cur.items, 'threads-total': sum(done.map((x) => x.items)), 'edges-total': sum(done.map((x) => x.edgeWork)), work: fw, 'push-work': pw }}
          panel={
            <>
              <Line y={40} bold><Translated vi={<>frontier, lần gọi {c} trên {CALLS}</>}>frontier, call {c} of {CALLS}</Translated></Line>
              <Line y={62}><Translated vi={<>{cur.items} luồng, mỗi ô của mặt sóng vừa rồi một luồng</>}>{cur.items} thread{cur.items === 1 ? '' : 's'}, one per block of the last wavefront</Translated></Line>
              <Line y={82}><Translated vi={<>{cur.edgeWork} cạnh được đọc</>}>{cur.edgeWork} edges read</Translated></Line>
              <Line y={116} bold><Translated vi={<>đến giờ: {sum(done.map((x) => x.items))} luồng, {sum(done.map((x) => x.edgeWork))} cạnh</>}>so far: {sum(done.map((x) => x.items))} threads, {sum(done.map((x) => x.edgeWork))} edges</Translated></Line>
              <text x={0} y={152} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi="khối lượng việc đến giờ: số luồng + số cạnh đã đọc">work so far: threads + edges read</Translated></text>
              <rect x={0} y={160} width={pw * scale} height={16} fill="var(--ink-3)" />
              <text x={pw * scale + 6} y={173} fontSize="11.5"><Translated vi={<>push {pw}</>}>push {pw}</Translated></text>
              <rect x={0} y={184} width={fw * scale} height={16} fill="var(--ink)" />
              <text x={fw * scale + 6} y={197} fontSize="11.5"><Translated vi={<>frontier {fw}</>}>frontier {fw}</Translated></text>
              {c === CALLS ? <Line y={232} bold data="nm">{fw} = n + m = {N} + {M}</Line> : null}
            </>
          }
        />
      );
    },
  },
  {
    clip: 'wavefront-4',
    get label() { return tr('A launch per level', 'Một lần khởi chạy cho mỗi mức'); },
    draw: (p) => {
      const k = Math.min(CALLS, Math.floor(ramp(p, 0.04, 0.9) * (CALLS + 1)));
      const largest = Math.max(...SIZES);
      return (
        <Stage
          beat={4}
          look={k ? visitedUpTo(Math.min(D, k)) : () => 'free'}
          data={{ launches: k, largest }}
          panel={
            <>
              <Line y={40} bold><Translated vi={<>{k} lần khởi chạy, {k} lần đọc lại</>}>{k} launch{k === 1 ? '' : 'es'}, {k} read-back{k === 1 ? '' : 's'}</Translated></Line>
              <text x={0} y={70} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi="khởi chạy">launch</Translated></text>
              {Array.from({ length: CALLS }, (_, i) => (
                <rect key={'l' + i} x={i * 11} y={76} width={8} height={18} rx="1" fill={i < k ? 'var(--ink)' : 'none'} stroke="var(--ink-3)" strokeWidth="0.8" />
              ))}
              <text x={0} y={116} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi="đọc lại kích thước frontier, quyết định có chạy tiếp không">read back the frontier size, decide whether to go on</Translated></text>
              {Array.from({ length: CALLS }, (_, i) => (
                <rect key={'r' + i} x={i * 11} y={122} width={8} height={18} rx="1" fill={i < k ? 'var(--ink-2)' : 'none'} stroke="var(--ink-3)" strokeWidth="0.8" />
              ))}
              <Line y={176}><Translated vi={<>mặt sóng lớn nhất: {largest} trên {N} ô</>}>largest wavefront: {largest} of {N} blocks</Translated></Line>
              <Line y={196}><Translated vi={<>mỗi lần khởi chạy có một chi phí cố định, dù nó làm ít việc đến đâu</>}>each launch has a fixed cost, however little it does</Translated></Line>
              {k === CALLS ? <Line y={230} bold><Translated vi={<>{CALLS} mức: {CALLS} lần chi phí cố định</>}>{CALLS} levels: {CALLS} fixed costs</Translated></Line> : null}
            </>
          }
        />
      );
    },
  },
  {
    clip: 'wavefront-5',
    get label() { return tr('Walking back', 'Đi ngược về'); },
    draw: (p) => {
      const shown = Math.max(1, Math.ceil(ramp(p, 0.06, 0.8) * PATH.length));
      const part = PATH.slice(PATH.length - shown);
      const ties = new Set(part.filter((v) => CHOICES[v] > 1));
      return (
        <Stage
          beat={5}
          look={(v) => (LV[v] <= D ? 'visited' : 'free')}
          number={(v) => (part.includes(v) ? String(LV[v]) : null)}
          path={part}
          ties={ties}
          data={{ shown, 'path-length': PATH.length, ties: ties.size, target: LV[MAZE_GOAL] }}
          panel={
            <>
              <Line y={40} bold><Translated vi={<>đích T ở mức {LV[MAZE_GOAL]}</>}>target T at level {LV[MAZE_GOAL]}</Translated></Line>
              <Line y={62}><Translated vi={<>bước sang một ô kề thấp hơn một mức, cho tới S</>}>step to a neighbour one level lower, until S</Translated></Line>
              <Line y={96} bold><Translated vi={<>đã đi {shown} trên {PATH.length} ô</>}>{shown} of {PATH.length} blocks walked</Translated></Line>
              <Line y={118}><Translated vi={<>{ties.size} chỗ hòa đến giờ: hai ô kề cùng thấp hơn một mức</>}>{ties.size} tie{ties.size === 1 ? '' : 's'} so far: two neighbours one level lower</Translated></Line>
              <circle cx={7} cy={146} r={7.5} fill="none" stroke="var(--ink)" strokeWidth="1.3" strokeDasharray="2 2" />
              <text x={20} y={150} fontSize="11"><Translated vi="một chỗ hòa: quy tắc cố định chọn ô có số nhỏ hơn">a tie: the fixed rule takes the lower-numbered</Translated></text>
              {shown === PATH.length ? <Line y={186} bold><Translated vi={<>một đường dây ngắn nhất: {PATH.length - 1} bước</>}>a shortest wire: {PATH.length - 1} steps</Translated></Line> : null}
            </>
          }
        />
      );
    },
  },
];

export function WavefrontScene() {
  return <Scene chapter="ch18" title={tr('The wavefront', 'Mặt sóng')} beats={beats} />;
}
