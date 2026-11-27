import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { counts, coverage, loopView, makeData, mapping, stepKernel, threads, working } from '../../sim/grid1d';

// "Where the loop went": the sequential loop of Fig. 2.4 turning into the grid of
// Fig. 2.9 and the kernel of Fig. 2.10. 10 elements, blocks of 4. Every number
// drawn comes from src/sim/grid1d.ts, the model behind the section's stepper and
// explorer. Six beats, one narration clip each
// (content/ch02/narration/loop-to-grid-N.txt).
export const SCENE_N = 10;
export const SCENE_B = 4;
/** The three threads whose index arithmetic beat 4 works out: (block, thread). */
export const SCENE_WORKED: readonly (readonly [number, number])[] = [[0, 1], [1, 2], [2, 3]];

const N = SCENE_N;
const B = SCENE_B;
const cfg = { n: N, blockDim: B };
const data = makeData(N);
const trips = loopView(N, data);
const grid = threads(N, B);
const cnt = counts(cfg);
const idle = coverage(cfg).idleThreads.map((t) => t.g);
const idleSet = new Set(idle);
const done = stepKernel(cfg, 3, data);
// Metal with dispatchThreads: a grid of exactly n threads (lab 2.1's launch.mm).
const exact = counts({ n: N, blockDim: N });
const one = mapping('one');

// The source lines the drawing shows, built from the model's own expressions.
const BODY = 'C[i] = A[i] + B[i];';
export const SCENE_LINES = {
  loopInit: 'int i = 0',
  loopTest: one.guards[0],
  loopStep: '++i',
  body: BODY,
  index: `int i = ${one.firstExpr};`,
  guard: `if (${one.guards[0]})`, // i18n-ignore: C source line
  metalIndex: 'uint i [[thread_position_in_grid]]', // i18n-ignore: Metal source line
} as const;

// Geometry. Thread box g and array cell g share an x, so the grid sits over the data.
const VW = 760;
const X0 = 64;
const CW = 50;
const GAP = 22;
const GY = 178;
const TH = 42;
const AY = 254;
const RH = 26;
const MONO = { fontVariationSettings: "'MONO' 1, 'CASL' 0" } as const; // i18n-ignore: CSS value
const xOf = (i: number, gap: number) => X0 + i * CW + Math.floor(i / B) * GAP * gap;

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const ease = (v: number) => v * v * (3 - 2 * v);

type Part = 'plain' | 'gone' | 'kept';

interface Box {
  /** 0 to 1: how far the box has travelled from the loop to its place. */
  arrived: number;
  opacity: number;
  /** Show the block and thread index above the value. */
  builtins: number;
  /** The thread's private i, or null while it has none. */
  i: number | null;
  /** Failed the guard. */
  off?: boolean;
  /** One of the threads the loop had no trip for. */
  spare?: boolean;
  marked?: boolean;
}

interface Frame {
  /** 0: threads and elements side by side. 1: spread into blocks. */
  gap: number;
  loop: { opacity: number; init: Part; test: Part; step: Part };
  /** Beat 1: the single thread and the trip it is on. */
  cursor?: { i: number; stored: number } | null;
  boxes: Box[];
  blocks: number;
  /** A single frame around a grid of exactly n threads. */
  exactGrid?: number;
  kernel: { title: string; lines: { text: string; on: number; bold?: boolean }[] } | null;
  strip?: string[];
  hot?: number | null;
  /** Elements of C that hold a sum. */
  stored: Set<number>;
  headline?: string;
}

function Stage(f: Frame) {
  const part = (text: string, state: Part) => (
    <tspan
      fontWeight={state === 'kept' ? 700 : 400}
      style={{ fill: state === 'gone' ? 'var(--ink-3)' : 'var(--ink)', textDecoration: state === 'gone' ? 'line-through' : undefined }}
    >
      {text}
    </tspan>
  );
  // On a narrow screen the drawing keeps a legible size and scrolls sideways.
  return (
    <div style={{ overflowX: 'auto' }}>
    <svg viewBox={`0 0 ${VW} 372`} role="img" aria-label={tr('Animation: a loop becomes a grid of threads', 'Hoạt hình: một vòng lặp trở thành một lưới luồng')} style={{ width: '100%', minWidth: 600, maxWidth: VW, display: 'block' }}>
      {/* the loop */}
      <g opacity={f.loop.opacity}>
        <rect x={10} y={8} width={350} height={82} rx="3" fill="var(--cell)" stroke="var(--rule)" />
        <text x={20} y={25} fontSize="11" fontWeight="600" style={{ fill: 'var(--ink-2)' }}>{tr('the sequential loop: one thread', 'vòng lặp tuần tự: một luồng')}</text>
        <text x={20} y={47} fontSize="12" style={MONO}>{/* i18n-ignore: C source line */}
          for ({part(SCENE_LINES.loopInit, f.loop.init)}; {part(SCENE_LINES.loopTest, f.loop.test)}; {part(SCENE_LINES.loopStep, f.loop.step)})
        </text>
        <text x={48} y={66} fontSize="12" style={MONO}>{SCENE_LINES.body}</text>
      </g>

      {/* the kernel */}
      {f.kernel ? (
        <g>
          <rect x={372} y={8} width={378} height={82} rx="3" fill="var(--cell)" stroke="var(--rule)" />
          <text x={382} y={25} fontSize="11" fontWeight="600" style={{ fill: 'var(--ink-2)' }}>{f.kernel.title}</text>
          {f.kernel.lines.map((ln, k) =>
            ln.on > 0 ? (
              <text key={k} x={382} y={44 + k * 18} fontSize="11.5" opacity={ln.on} fontWeight={ln.bold ? 700 : 400} style={{ ...MONO, whiteSpace: 'pre' }}>{ln.text}</text>
            ) : null,
          )}
        </g>
      ) : null}

      {/* what is being worked out */}
      {(f.strip ?? []).map((s, k) => (
        <text key={k} x={X0} y={110 + k * 16} fontSize="12" style={MONO}>{s}</text>
      ))}

      {/* block frames */}
      {f.blocks > 0
        ? Array.from({ length: cnt.blocks }, (_, b) => (
            <g key={b} opacity={f.blocks}>
              <rect x={xOf(b * B, f.gap) - 5} y={GY - 22} width={B * CW + 6} height={TH + 30} rx="3" fill="none" stroke="var(--ink-2)" />
              <text x={xOf(b * B, f.gap)} y={GY - 8} fontSize="11" fontWeight="600"><Translated vi={<>khối {b}</>}>block {b}</Translated></text>
            </g>
          ))
        : null}
      {f.exactGrid ? (
        <g opacity={f.exactGrid}>
          <rect x={xOf(0, f.gap) - 5} y={GY - 22} width={N * CW + 6} height={TH + 30} rx="3" fill="none" stroke="var(--ink-2)" />
          <text x={xOf(0, f.gap)} y={GY - 8} fontSize="11" fontWeight="600"><Translated vi={<>một lưới đúng {exact.launched} luồng</>}>a grid of exactly {exact.launched} threads</Translated></text>
        </g>
      ) : null}

      {/* the one thread of the loop */}
      {f.cursor ? (
        <g>
          <rect x={xOf(f.cursor.i, 0)} y={GY} width={CW - 4} height={TH} rx="2" fill="var(--reg-wash)" stroke="var(--reg)" strokeWidth="1.6" />
          <text x={xOf(f.cursor.i, 0) + (CW - 4) / 2} y={GY + 16} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{tr('thread', 'luồng')}</text>
          <text x={xOf(f.cursor.i, 0) + (CW - 4) / 2} y={GY + 33} fontSize="12.5" textAnchor="middle" fontWeight="600" style={{ fill: 'var(--reg)' }}>i = {f.cursor.i}</text>
        </g>
      ) : null}

      {/* threads */}
      {f.boxes.map((bx, g) => {
        if (bx.opacity <= 0) return null;
        const t = grid[g];
        const e = ease(bx.arrived);
        const x = 150 + (xOf(g, f.gap) - 150) * e;
        const y = 34 + (GY - 34) * e;
        const stroke = bx.off || bx.spare ? 'var(--ink-3)' : 'var(--reg)';
        return (
          <g key={g} opacity={bx.opacity}>
            <rect x={x} y={y} width={CW - 4} height={TH} rx="2"
              fill={bx.off || (bx.spare && bx.i == null) ? 'var(--cell)' : 'var(--reg-wash)'}
              stroke={stroke} strokeWidth={bx.marked ? 2.4 : 1} strokeDasharray={bx.off || (bx.spare && bx.i == null) ? '4 3' : undefined} />
            {bx.builtins > 0 ? (
              <text x={x + (CW - 4) / 2} y={y + 15} fontSize="10" textAnchor="middle" opacity={bx.builtins} style={{ fill: 'var(--ink-2)' }}>
                b{t.block} t{t.thread}
              </text>
            ) : null}
            {bx.i != null ? (
              <text x={x + (CW - 4) / 2} y={y + (bx.builtins > 0 ? 33 : 27)} fontSize="12.5" textAnchor="middle" fontWeight="600"
                style={{ fill: bx.off ? 'var(--ink-3)' : 'var(--reg)', textDecoration: bx.off ? 'line-through' : undefined }}>
                i = {bx.i}
              </text>
            ) : null}
          </g>
        );
      })}

      {/* the arrays */}
      {(['A', 'B', 'C'] as const).map((name, r) => (
        <g key={name}>
          <text x={22} y={AY + r * RH + 16} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}>{name}</text>
          {Array.from({ length: N }, (_, i) => {
            const hot = name !== 'C' && f.hot === i;
            const has = name === 'C' ? f.stored.has(i) : true;
            const v = name === 'A' ? data.A[i] : name === 'B' ? data.B[i] : trips[i].c;
            return (
              <g key={i}>
                <rect x={xOf(i, f.gap)} y={AY + r * RH} width={CW - 4} height={RH - 4} rx="2"
                  fill={hot ? 'var(--global)' : name === 'C' && has ? 'var(--global-wash)' : 'var(--cell)'}
                  stroke={hot || (name === 'C' && has) ? 'var(--global)' : 'var(--rule)'} />
                {has ? (
                  <text x={xOf(i, f.gap) + (CW - 4) / 2} y={AY + r * RH + 15.5} fontSize="11.5" textAnchor="middle"
                    style={{ fill: hot ? 'var(--page)' : 'var(--ink)' }}>{v}</text>
                ) : null}
              </g>
            );
          })}
        </g>
      ))}
      <text x={X0} y={AY + 3 * RH + 10} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>các phần tử 0 đến {N - 1}</>}>elements 0 to {N - 1}</Translated></text>
      {f.headline ? <text x={X0} y={362} fontSize="13" fontWeight="600">{f.headline}</text> : null}
    </svg>
    </div>
  );
}

const noBoxes: Box[] = grid.map(() => ({ arrived: 1, opacity: 0, builtins: 0, i: null }));
const all = new Set(done.C.flatMap((v, i) => (v == null ? [] : [i])));
const empty = new Set<number>();
const plainLoop = { opacity: 1, init: 'plain', test: 'plain', step: 'plain' } as const;
const kernelLines = (index: number, guard: number, body: number, boldGuard = false) => ({
  title: tr('the kernel: run by every thread', 'kernel: mọi luồng đều chạy nó'),
  lines: [
    { text: SCENE_LINES.index, on: index },
    { text: SCENE_LINES.guard, on: guard, bold: boldGuard },
    { text: '    ' + SCENE_LINES.body, on: body },
  ],
});
const workedLine = ([b, t]: readonly [number, number]) => tr(`block ${b}, thread ${t}:  ${working(b, t, B)}`, `khối ${b}, luồng ${t}:  ${working(b, t, B)}`);

const beats: Beat[] = [
  {
    clip: 'loop-to-grid-1',
    get label() {
      return tr('The loop', 'Vòng lặp');
    },
    draw: (p) => {
      // The loop starts once it has been introduced; each trip loads, then stores.
      const pos = ramp(p, 0.16, 0.94) * N;
      const k = Math.min(N - 1, Math.floor(pos));
      const within = pos - k;
      const stored = p >= 0.94 ? N : k + (within > 0.55 ? 1 : 0);
      return (
        <Stage gap={0} loop={plainLoop} cursor={{ i: k, stored }} boxes={noBoxes} blocks={0} kernel={null}
          hot={p < 0.94 && p >= 0.16 && within <= 0.55 ? k : null} stored={new Set(trips.slice(0, stored).map((t) => t.i))}
          headline={stored >= N ? tr(`${N} trips, one after another, all by one thread`, `${N} lượt lặp, lần lượt từng lượt, tất cả do một luồng`) : tr(`trip ${k + 1} of ${N}: C[${k}] = ${trips[k].a} + ${trips[k].b} = ${trips[k].c}`, `lượt ${k + 1} trên ${N}: C[${k}] = ${trips[k].a} + ${trips[k].b} = ${trips[k].c}`)} />
      );
    },
  },
  {
    clip: 'loop-to-grid-2',
    get label() {
      return tr('Trips become threads', 'Lượt lặp thành luồng');
    },
    draw: (p) => {
      const boxes = grid.map((t, g): Box => {
        if (g >= N) return { arrived: 1, opacity: 0, builtins: 0, i: null };
        const start = 0.3 + g * 0.045;
        const a = ramp(p, start, start + 0.2);
        return { arrived: a, opacity: a > 0 ? 1 : 0, builtins: 0, i: t.i };
      });
      // Only the n trips travel; the two threads the blocks add appear in part 3.
      const out = boxes.filter((b, g) => g < N && b.arrived >= 1).length;
      return (
        <Stage gap={0} loop={{ ...plainLoop, opacity: 1 - 0.55 * ramp(p, 0.3, 0.9) }} boxes={boxes} blocks={0} kernel={null} stored={p < 0.3 ? all : empty}
          headline={p < 0.3 ? tr('no trip reads what another trip writes', 'không lượt lặp nào đọc thứ mà lượt khác ghi') : tr(`${out} of ${N} trips are now threads, each with its own i`, `${out} trên ${N} lượt lặp giờ là luồng, mỗi luồng có i riêng`)} />
      );
    },
  },
  {
    clip: 'loop-to-grid-3',
    get label() {
      return tr('Blocks', 'Khối');
    },
    draw: (p) => {
      const spare = ramp(p, 0.68, 0.82);
      const boxes = grid.map((t, g): Box =>
        g < N ? { arrived: 1, opacity: 1, builtins: 0, i: t.i } : { arrived: 1, opacity: spare, builtins: 0, i: null, spare: true },
      );
      return (
        <Stage gap={ease(ramp(p, 0.3, 0.5))} loop={{ ...plainLoop, opacity: 0.45 }} boxes={boxes} blocks={ramp(p, 0.42, 0.58)} kernel={null} stored={empty}
          headline={
            p < 0.3 ? tr('threads come in blocks of one size', 'luồng đến theo từng khối cùng một kích thước')
              : p < 0.66 ? tr(`${N} threads, ${B} to a block: ${cnt.blocks} blocks`, `${N} luồng, ${B} luồng một khối: ${cnt.blocks} khối`)
                : tr(`${cnt.blocks} blocks of ${B} is ${cnt.launched} threads: ${cnt.idle} more than the loop had trips`, `${cnt.blocks} khối, mỗi khối ${B}, là ${cnt.launched} luồng: nhiều hơn số lượt lặp của vòng lặp ${cnt.idle} luồng`)
          } />
      );
    },
  },
  {
    clip: 'loop-to-grid-4',
    get label() {
      return tr('Two built-in numbers', 'Hai số có sẵn');
    },
    draw: (p) => {
      const at = [0.58, 0.72, 0.87];
      const shown = SCENE_WORKED.filter((_, k) => p >= at[k]);
      const isWorked = (g: number) => shown.some(([b, t]) => b * B + t === g);
      const everyone = p >= 0.965;
      const boxes = grid.map((t, g): Box => ({
        arrived: 1, opacity: 1, builtins: ramp(p, 0.12, 0.28), i: everyone || isWorked(g) ? t.i : null, spare: g >= N, marked: !everyone && isWorked(g),
      }));
      return (
        <Stage gap={1} loop={{ ...plainLoop, opacity: 0.45 }} boxes={boxes} blocks={1} kernel={kernelLines(ramp(p, 0.4, 0.5), 0, 0)} stored={empty}
          strip={shown.map(workedLine)}
          headline={
            p < 0.4 ? tr('each thread is given two numbers: its block (b) and its place in the block (t)', 'mỗi luồng được cấp hai số: khối của nó (b) và vị trí của nó trong khối (t)')
              : !everyone ? tr(`i = block index × ${B} + thread index`, `i = chỉ số khối × ${B} + chỉ số luồng`)
                : tr('the same arithmetic in every thread, a different i in each', 'cùng một phép tính ở mọi luồng, mỗi luồng ra một i khác')
          } />
      );
    },
  },
  {
    clip: 'loop-to-grid-5',
    get label() {
      return tr('The guard', 'Phép kiểm tra');
    },
    draw: (p) => {
      const gone: Part = p >= 0.16 ? 'gone' : 'plain';
      const guarded = p >= 0.76;
      const boxes = grid.map((t, g): Box => ({ arrived: 1, opacity: 1, builtins: 1, i: t.i, spare: g >= N, off: guarded && idleSet.has(g) }));
      return (
        <Stage gap={1} loop={{ opacity: 1, init: gone, test: p >= 0.4 ? 'kept' : 'plain', step: gone }} boxes={boxes} blocks={1}
          kernel={kernelLines(1, ramp(p, 0.5, 0.6), ramp(p, 0.86, 0.94), p >= 0.5 && p < 0.86)} stored={p >= 0.92 ? all : empty}
          strip={guarded ? idle.map((g) => tr(`thread i = ${grid[g].i}:  ${grid[g].i} < ${N} is false`, `luồng i = ${grid[g].i}:  ${grid[g].i} < ${N} là sai`)) : []}
          headline={
            p < 0.4 ? tr('the start and the increment became the two built-in numbers', 'giá trị khởi đầu và bước tăng đã trở thành hai số có sẵn')
              : !guarded ? tr(`the test ${SCENE_LINES.loopTest} moves into the kernel`, `phép kiểm tra ${SCENE_LINES.loopTest} chuyển vào trong kernel`)
                : tr(`threads ${idle.map((g) => grid[g].i).join(' and ')} fail the guard and do nothing`, `các luồng ${idle.map((g) => grid[g].i).join(' và ')} không qua phép kiểm tra và không làm gì`)
          } />
      );
    },
  },
  {
    clip: 'loop-to-grid-6',
    get label() {
      return tr('On Metal', 'Trên Metal');
    },
    draw: (p) => {
      const leave = 1 - ramp(p, 0.5, 0.66);
      const boxes = grid.map((t, g): Box =>
        g < N ? { arrived: 1, opacity: 1, builtins: 1 - ramp(p, 0.1, 0.24), i: t.i } : { arrived: 1, opacity: leave, builtins: 1 - ramp(p, 0.1, 0.24), i: t.i, spare: true, off: true },
      );
      return (
        <Stage gap={1 - ease(ramp(p, 0.56, 0.76))} loop={{ opacity: 0.45, init: 'gone', test: 'gone', step: 'gone' }} boxes={boxes}
          blocks={1 - ramp(p, 0.5, 0.62)} exactGrid={ramp(p, 0.74, 0.86)} stored={all}
          kernel={{
            title: tr('the same kernel in Metal Shading Language', 'cùng kernel đó bằng Metal Shading Language'),
            lines: [
              { text: SCENE_LINES.metalIndex, on: ramp(p, 0.14, 0.26) },
              { text: '    // an argument: the runtime fills it in', on: ramp(p, 0.2, 0.32) },
              { text: SCENE_LINES.body, on: ramp(p, 0.26, 0.38) },
            ],
          }}
          headline={
            p < 0.5 ? tr('each thread is handed its position in the grid', 'mỗi luồng được trao vị trí của nó trong lưới')
              : tr(`${exact.launched} threads for ${N} elements: ${exact.idle} spare, nothing to guard against`, `${exact.launched} luồng cho ${N} phần tử: ${exact.idle} luồng dư, không có gì phải kiểm tra`)
          } />
      );
    },
  },
];

export function LoopToGridScene() {
  return <Scene chapter="ch02" title={tr('Where the loop went', 'Vòng lặp đã đi đâu')} beats={beats} />;
}
