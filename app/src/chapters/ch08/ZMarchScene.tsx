import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { fmt } from '../../lib/reports';
import { BOOK_BLOCK_LIMIT, cubeAI, sharedBytes, slabAI, threadsCube, threadsSlab } from '../../sim/stencilslab';
import { CORRECT, localZ, makeProblem, planeTimeline, type PlaneFrame, type ZConfig } from '../../sim/zmarch';

// "Marching through z": the book's picture (Fig. 8.9), a 6-wide input tile whose block
// marches through 4 output planes, drawn from the same model the stepper runs. The
// block is the interior block of a 10^3 grid, so no load or output touches the edge.
// Every plane number, active thread and counter below comes from zmarch.planeTimeline;
// the comparison figures come from stencilslab. Six beats, one narration clip each
// (content/ch08/narration/z-march-N.txt).
const T = 6;
const base = { N: 10, inTile: T, ...CORRECT };
const prob = makeProblem(base.N);
const PLANES: ZConfig = { ...base, kernel: 'planes' };
const REGS: ZConfig = { ...base, kernel: 'registers' };
const TL = planeTimeline(PLANES, prob);
const TLR = planeTimeline(REGS, prob);
const iStart = TL.iStart;
const L = (z: number | null) => (z === null ? null : localZ(iStart, z));
const F = TL.frames;
const last = F[F.length - 1];
const activeCount = F[2].active.filter(Boolean).length;
// Where each block thread's output lands, in the order the threads compute.
const activeList = F[2].active.flatMap((on, t) => (on ? [t] : []));

// ---------------------------------------------------------------- geometry
const C = 11; // cell of an input or buffer plane
const SX = (n: number) => 120 + n * 84; // left edge of input plane n (0..5)
const IY = 40; // input planes
const BY = 154; // buffers
const OY = 270; // output planes
const RX = 618; // right column
const BAR_SCALE = 300; // beat 1: drawing units for 1,024 threads, the dashed limit line
const BAR_CAP = 470; // beat 1: longest bar drawn; a longer one is broken

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

type Tone = 'shared' | 'reg';
interface Box {
  name: string;
  /** Input plane it holds (local numbering, may be fractional while sliding). */
  at: number;
  tone: Tone;
  /** 0..1: how many of its cells are filled. */
  fill?: number;
  stale?: boolean;
  /** Vertical offset from the buffer row (used when registers peel away). */
  dy?: number;
  opacity?: number;
  /** Cells to mark (ty*T + tx). */
  hot?: number[];
  /** No line to the input plane (a box that is leaving shared memory). */
  noLink?: boolean;
}
interface Frame {
  dots?: number; // beat 1: one thread per point, opacity
  block?: number; // opacity of the flattened block of threads
  active?: boolean[];
  hotThread?: number | null;
  inShared?: number[]; // input planes currently held on chip (outlined)
  boxes?: Box[];
  barrier?: string | null;
  outputs?: Record<number, number>; // local output plane -> fraction of its 16 cells filled
  counters?: [string, string][];
  bars?: number;
  compare?: number;
  headline?: ReactNode;
  arrows?: number;
  /** Beat 6: opacity of the per-thread registers drawn on the block. */
  regs?: number;
}

function Plane({ x, y, fill, stroke, wash, hot, dots, n = T, cell = C }: {
  x: number; y: number; fill: string; stroke: string; wash?: (r: number, c: number) => boolean; hot?: number[]; dots?: number; n?: number; cell?: number;
}) {
  const cells: ReactNode[] = [];
  for (let r = 0; r < n; r++)
    for (let c = 0; c < n; c++) {
      const isHot = hot?.includes(r * n + c);
      cells.push(
        <rect key={r * n + c} x={x + c * cell} y={y + r * cell} width={cell - 1.5} height={cell - 1.5} rx="1"
          fill={isHot ? stroke : wash?.(r, c) ? fill : 'var(--cell)'} stroke={stroke} strokeWidth="0.6" />,
      );
      if (dots)
        cells.push(<circle key={'d' + (r * n + c)} cx={x + c * cell + (cell - 1.5) / 2} cy={y + r * cell + (cell - 1.5) / 2} r="1.6" fill="var(--ink)" opacity={dots} />);
    }
  return <g>{cells}</g>;
}

function Stage(f: Frame) {
  const halo = (r: number, c: number) => r === 0 || c === 0 || r === T - 1 || c === T - 1;
  return (
    <svg viewBox="0 0 720 372" role="img" aria-label={tr('Animation of a block of threads marching through z', 'Hoạt cảnh một khối luồng tiến dọc theo z')} style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      {/* input tile, plane by plane */}
      <text x={SX(0)} y={14} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}><Translated vi="ô đầu vào trong bộ nhớ toàn cục, mỗi ô vuông là một mặt phẳng x-y (z tăng dần sang phải)">input tile in global memory, one x-y plane per square (z increases to the right)</Translated></text>
      {Array.from({ length: T }, (_, n) => (
        <g key={n}>
          <text x={SX(n) + 33} y={IY - 6} fontSize="10.5" textAnchor="middle"><Translated vi={<>mặt phẳng {n}</>}>plane {n}</Translated></text>
          <Plane x={SX(n)} y={IY} fill="var(--global-wash)" stroke="var(--global)" wash={(r, c) => !halo(r, c)} dots={f.dots} />
          {f.inShared?.includes(n) ? <rect x={SX(n) - 3} y={IY - 3} width={T * C + 4.5} height={T * C + 4.5} fill="none" stroke="var(--shared)" strokeWidth="2" /> : null}
        </g>
      ))}

      {/* beat 1: threads needed by one thread per point */}
      {f.bars != null ? (
        <g>
          {[
            [tr('this tile, one thread per point', 'ô này, một luồng cho mỗi điểm'), threadsCube(T)],
            [tr('a 32-wide cube', 'một khối lập phương rộng 32'), threadsCube(32)],
          ].map(([label, v], n) => {
            // To scale, BAR_SCALE units per 1,024 threads. A bar that would run past BAR_CAP is
            // cut short and drawn with a break, and its label says it is not to scale.
            const full = (Number(v) / BOOK_BLOCK_LIMIT) * BAR_SCALE;
            const broken = full > BAR_CAP;
            const w = Math.min(BAR_CAP, full) * f.bars!;
            const y = BY + 6 + n * 46;
            const cut = SX(0) + BAR_CAP * 0.82;
            return (
              <g key={n} data-bar={Number(v)} data-broken={broken ? 1 : 0}>
                <text x={SX(0)} y={y} fontSize="11.5">
                  <Translated vi={<>{label}: {Number(v).toLocaleString('en-US')} luồng{broken ? ' (không theo tỉ lệ)' : ''}</>}>
                    {label}: {Number(v).toLocaleString('en-US')} threads{broken ? ' (not to scale)' : ''}
                  </Translated>
                </text>
                <rect x={SX(0)} y={y + 7} width={w} height={14} fill={Number(v) > BOOK_BLOCK_LIMIT ? 'var(--fail)' : 'var(--ink-2)'} />
                {broken && SX(0) + w > cut + 10 ? (
                  <g data-mark="break">
                    <polygon points={`${cut},${y + 3} ${cut + 9},${y + 3} ${cut + 3},${y + 25} ${cut - 6},${y + 25}`} fill="var(--page)" />
                    <line x1={cut} y1={y + 3} x2={cut - 6} y2={y + 25} stroke="var(--ink)" strokeWidth="1.2" />
                    <line x1={cut + 9} y1={y + 3} x2={cut + 3} y2={y + 25} stroke="var(--ink)" strokeWidth="1.2" />
                  </g>
                ) : null}
                {Number(v) > BOOK_BLOCK_LIMIT && f.bars! > 0.9 ? (
                  <text x={SX(0) + w + 6} y={y + 19} fontSize="11" style={{ fill: 'var(--fail)' }}><Translated vi={<>gấp {Number(v) / BOOK_BLOCK_LIMIT} lần giới hạn</>}>{Number(v) / BOOK_BLOCK_LIMIT} times the limit</Translated></text>
                ) : null}
              </g>
            );
          })}
          <line x1={SX(0) + BAR_SCALE} x2={SX(0) + BAR_SCALE} y1={BY - 8} y2={BY + 82} stroke="var(--ink)" strokeDasharray="3 3" />
          <text x={SX(0) + BAR_SCALE + 4} y={BY - 10} fontSize="11"><Translated vi={<>tối đa {BOOK_BLOCK_LIMIT.toLocaleString('en-US')} luồng trong một khối</>}>at most {BOOK_BLOCK_LIMIT.toLocaleString('en-US')} threads in a block</Translated></text>
        </g>
      ) : null}

      {/* the block: one plane of threads */}
      {f.block ? (
        <g opacity={f.block}>
          <text x={12} y={BY - 22} fontSize="11.5" fontWeight="600"><Translated vi="khối">the block</Translated></text>
          <text x={12} y={BY - 8} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>{threadsSlab(T)} luồng</>}>{threadsSlab(T)} threads</Translated></text>
          {Array.from({ length: T * T }, (_, t) => {
            const r = Math.floor(t / T), c = t % T;
            const on = f.active ? f.active[t] : !halo(r, c);
            const hot = f.hotThread === t;
            return (
              <g key={t} data-thread={t} data-active={on ? 1 : 0}>
                {f.regs ? (
                  <rect x={18 + c * 13 - 6} y={BY + 6 + r * 13 - 6} width={12} height={12} rx="2" fill="var(--reg-wash)" stroke="var(--reg)" strokeWidth="1.2" opacity={f.regs} />
                ) : null}
                <circle cx={18 + c * 13} cy={BY + 6 + r * 13} r={hot ? 5 : 4}
                  fill={hot ? 'var(--ok)' : on ? 'var(--ink)' : 'var(--page)'} stroke="var(--ink)" strokeWidth="1.1" />
              </g>
            );
          })}
          {f.regs && f.regs > 0.5 ? (
            <>
              <text x={12} y={BY + 92} fontSize="10" style={{ fill: 'var(--reg)' }}><Translated vi="mỗi luồng giữ">each thread keeps</Translated></text>
              <text x={12} y={BY + 105} fontSize="10" style={{ fill: 'var(--reg)' }}>inPrev, inCurr, inNext{/* i18n-ignore: identifiers */}</text>
              <text x={12} y={BY + 118} fontSize="10" style={{ fill: 'var(--reg)' }}><Translated vi="trong thanh ghi">in registers</Translated></text>
            </>
          ) : (
            <>
              <text x={12} y={BY + 92} fontSize="10" style={{ fill: 'var(--ink-2)' }}><Translated vi="đặc: tính toán">filled: computes</Translated></text>
              <text x={12} y={BY + 105} fontSize="10" style={{ fill: 'var(--ink-2)' }}><Translated vi="rỗng: chỉ nạp">hollow: only loads</Translated></text>
            </>
          )}
          {f.barrier ? (
            <g>
              <rect x={8} y={BY - 2} width={92} height={T * 13 + 2} rx="3" fill="var(--ink)" opacity="0.18" />
              <rect x={8} y={BY + T * 13 + 2} width={92} height={4} fill="var(--ink)" />
            </g>
          ) : null}
        </g>
      ) : null}

      {/* the window: shared-memory planes (and registers) */}
      {(f.boxes ?? []).map((b) => {
        const x = SX(b.at);
        const y = BY + (b.dy ?? 0);
        const col = b.tone === 'reg' ? 'var(--reg)' : 'var(--shared)';
        const wash = b.tone === 'reg' ? 'var(--reg-wash)' : 'var(--shared-wash)';
        const filled = Math.round((b.fill ?? 1) * T * T);
        return (
          <g key={b.name} opacity={(b.opacity ?? 1) * (b.stale ? 0.55 : 1)} data-box={b.name} data-at={b.at} data-tone={b.tone} data-stale={b.stale ? 1 : 0}>
            {b.noLink ? null : <line x1={x + 33} x2={x + 33} y1={IY + T * C + 4} y2={y - 4} stroke={col} strokeWidth="1.2" strokeDasharray={b.stale ? '2 3' : undefined} />}
            <Plane x={x} y={y} fill={wash} stroke={col} wash={(r, c) => r * T + c < filled} hot={b.hot} />
            <text x={x + 33} y={y + T * C + 13} fontSize="10.5" textAnchor="middle" fontWeight="600" style={{ fill: col }}>{b.name}</text>
            {b.stale ? <text x={x + 33} y={y + T * C + 25} fontSize="9.5" textAnchor="middle"><Translated vi="trống cho lần nạp tới">free for the next load</Translated></text> : null}
          </g>
        );
      })}
      {f.arrows ? (
        <text x={SX(0)} y={BY - 31} fontSize="11" opacity={f.arrows} style={{ paintOrder: 'stroke', stroke: 'var(--page)', strokeWidth: 5 }}><Translated vi="mỗi luồng chép điểm của chính nó: hiện tại → trước, kế tiếp → hiện tại">each thread copies its own point: current → previous, next → current</Translated></text>
      ) : null}
      {f.barrier ? (
        <g>
          <rect x={SX(0) - 8} y={BY - 26} width={SX(5) + 74 - SX(0)} height={4} fill="var(--ink)" />
          <text x={SX(0)} y={BY - 31} fontSize="11.5" fontWeight="600" style={{ paintOrder: 'stroke', stroke: 'var(--page)', strokeWidth: 5 }}>{f.barrier}</text>
        </g>
      ) : null}

      {/* outputs */}
      {f.outputs ? (
        <g>
          <text x={12} y={OY + 26} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--ok)' }}><Translated vi="đầu ra">output</Translated></text>
          {[1, 2, 3, 4].map((n) => {
            const frac = f.outputs![n] ?? 0;
            const done = Math.round(frac * activeCount);
            return (
              <g key={n} data-output={n} data-done={done}>
                <text x={SX(n) + 33} y={OY - 5} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>mặt phẳng đầu ra {n}</>}>output plane {n}</Translated></text>
                {Array.from({ length: (T - 2) * (T - 2) }, (_, q) => {
                  const r = Math.floor(q / (T - 2)), c = q % (T - 2);
                  return (
                    <rect key={q} x={SX(n) + 11 + c * C} y={OY + r * C} width={C - 1.5} height={C - 1.5} rx="1"
                      fill={q < done ? 'var(--ok)' : 'var(--cell)'} stroke={q < done ? 'var(--ok)' : 'var(--rule)'} strokeWidth="0.6" />
                  );
                })}
              </g>
            );
          })}
        </g>
      ) : null}

      {/* counters */}
      {f.counters ? (
        <g>
          {f.counters.map(([k, v], n) => (
            <g key={k} data-counter={k} data-value={v}>
              <text x={RX} y={BY + n * 34} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>{k}</text>
              <text x={RX} y={BY + 15 + n * 34} fontSize="14" fontWeight="650">{v}</text>
            </g>
          ))}
        </g>
      ) : null}

      {/* beat 5: the same slab at full width */}
      {f.compare ? (
        <g opacity={f.compare}>
          <rect x={SX(0) - 6} y={BY - 4} width={322} height={74} rx="4" fill="var(--page)" stroke="var(--ink)" />
          <text x={SX(0) + 4} y={BY + 15} fontSize="12" fontWeight="650"><Translated vi={<>tiến theo z ở độ rộng 32: {threadsSlab(32).toLocaleString('en-US')} luồng, {fmt(slabAI(32, 30))} FLOP/B</>}>march at width 32: {threadsSlab(32).toLocaleString('en-US')} threads, {fmt(slabAI(32, 30))} FLOP/B</Translated></text>
          <text x={SX(0) + 4} y={BY + 36} fontSize="12"><Translated vi={<>khối lập phương rộng 8: {threadsCube(8)} luồng, {fmt(cubeAI(8), 2)} FLOP/B</>}>8-wide cube: {threadsCube(8)} threads, {fmt(cubeAI(8), 2)} FLOP/B</Translated></text>
          <text x={SX(0) + 4} y={BY + 56} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="cả hai theo cách đếm số lần nạp và lưu của sách (p. 192)">both by the book's count of loads and stores (p. 192)</Translated></text>
        </g>
      ) : null}

      {f.headline ? <text x={12} y={364} fontSize="13" fontWeight="600">{f.headline}</text> : null}
    </svg>
  );
}

// ---------------------------------------------------------------- frames from the model
const counters = (fr: PlaneFrame, partStores?: number): [string, string][] => [
  [tr('planes loaded', 'mặt phẳng đã nạp'), String(fr.planesLoaded)],
  [tr('elements loaded', 'phần tử đã nạp'), String(fr.loads)],
  [tr('outputs stored', 'đầu ra đã lưu'), String(partStores ?? fr.stores)],
];
const outputsOf = (fr: PlaneFrame) => Object.fromEntries(fr.outputPlanes.map((z) => [L(z)!, 1]));
// Where each part of the window sits, by frame index. After a shift, inNext_s still holds
// a copy of the new current plane that nobody reads; it is drawn empty, under the plane
// the next iteration loads into it (the next frame's hold.next).
const nextAt = (k: number) => (F[k].label === 'shifted' ? (F[k + 1] ? L(F[k + 1].hold.next) : null) : L(F[k].hold.next));
const windowOf = (k: number, before = k, t = 1): Box[] => {
  const fr = F[k];
  const pos = (now: number | null, then: number | null) => lerp(then ?? now ?? 0, now ?? then ?? 0, t);
  const boxes: Box[] = [
    { name: 'inPrev_s', at: pos(L(fr.hold.prev), L(F[before].hold.prev)), tone: 'shared' }, // i18n-ignore: identifier
    { name: 'inCurr_s', at: pos(L(fr.hold.curr), L(F[before].hold.curr)), tone: 'shared' }, // i18n-ignore: identifier
  ];
  const nx = nextAt(k);
  if (nx !== null) {
    const stale = fr.label === 'shifted';
    boxes.push({ name: 'inNext_s', at: pos(nx, nextAt(before) ?? nx), tone: 'shared', stale, fill: stale ? 0 : 1 }); // i18n-ignore: identifier
  }
  return boxes;
};
const onChip = (fr: PlaneFrame) => [...new Set([fr.hold.prev, fr.hold.curr, fr.label === 'shifted' ? null : fr.hold.next].filter((z) => z !== null).map((z) => L(z)!))];

// One step of the march at fraction t of its way through the frames (1, 2, 3 for iteration n).
function marchFrame(n: number, t: number): Frame {
  const kl = 1 + 3 * n; // loaded
  const kc = 2 + 3 * n; // computed
  const ks = 3 + 3 * n; // shifted
  const kb = 3 * n; // the frame before this iteration
  const computed = F[kc];
  if (t < 1 / 3) {
    const u = t * 3;
    return {
      block: 1, active: computed.active, inShared: onChip(F[kl]), boxes: windowOf(kl, kb, Math.min(1, u * 2)),
      outputs: outputsOf(F[kb]), counters: counters(u > 0.5 ? F[kl] : F[kb]), barrier: u > 0.6 ? tr('barrier 1', 'rào chắn 1') : null,
    };
  }
  if (t < 2 / 3)
    return { block: 1, active: computed.active, inShared: onChip(F[kl]), boxes: windowOf(kc), outputs: outputsOf(computed), counters: counters(computed) };
  const u = (t - 2 / 3) * 3;
  return {
    block: 1, active: computed.active, inShared: onChip(u < 0.3 ? computed : F[ks]), boxes: u < 0.3 ? windowOf(kc) : windowOf(ks, kc, ramp(u, 0.3, 0.9)),
    outputs: outputsOf(computed), counters: counters(F[ks]), barrier: u < 0.3 ? tr('barrier 2', 'rào chắn 2') : null,
  };
}

const crossOf = (t: number) => {
  const r = Math.floor(t / T), c = t % T;
  return [t, (r - 1) * T + c, (r + 1) * T + c, r * T + c - 1, r * T + c + 1];
};

const beats: Beat[] = [
  {
    clip: 'z-march-1',
    get label() { return tr('Too many threads', 'Quá nhiều luồng'); },
    draw: (p) => (
      <Stage dots={ramp(p, 0.05, 0.3)} bars={ramp(p, 0.3, 0.75)}
        headline={p < 0.5 ? tr(`one thread per point: ${threadsCube(T)} threads for this tile`, `một luồng cho mỗi điểm: ${threadsCube(T)} luồng cho ô này`) : tr(`a 32-wide cube would need ${threadsCube(32).toLocaleString('en-US')}`, `một khối lập phương rộng 32 sẽ cần ${threadsCube(32).toLocaleString('en-US')}`)} />
    ),
  },
  {
    clip: 'z-march-2',
    get label() { return tr('One plane', 'Một mặt phẳng'); },
    draw: (p) => {
      const f0 = F[0];
      const prevFill = ramp(p, 0.35, 0.6);
      const currFill = ramp(p, 0.65, 0.9);
      const boxes: Box[] = [];
      if (p >= 0.35) boxes.push({ name: 'inPrev_s', at: L(f0.hold.prev)!, tone: 'shared', fill: prevFill }); // i18n-ignore: identifier
      if (p >= 0.65) boxes.push({ name: 'inCurr_s', at: L(f0.hold.curr)!, tone: 'shared', fill: currFill }); // i18n-ignore: identifier
      const loadedSoFar = Math.min(f0.planesLoaded, (prevFill >= 1 ? 1 : 0) + (currFill >= 1 ? 1 : 0));
      return (
        <Stage block={ramp(p, 0.02, 0.3)} boxes={boxes} inShared={boxes.filter((b) => (b.fill ?? 0) >= 1).map((b) => b.at)}
          counters={[[tr('planes loaded', 'mặt phẳng đã nạp'), String(loadedSoFar)], [tr('threads', 'luồng'), String(threadsSlab(T))], [tr('of which compute', 'trong đó tính toán'), String(activeCount)]]}
          headline={p < 0.35 ? tr(`${threadsSlab(T)} threads: one x-y plane of the tile`, `${threadsSlab(T)} luồng: một mặt phẳng x-y của ô`) : tr('before the loop: two planes loaded', 'trước vòng lặp: đã nạp hai mặt phẳng')} />
      );
    },
  },
  {
    clip: 'z-march-3',
    get label() { return tr('Load and compute', 'Nạp và tính'); },
    draw: (p) => {
      const loaded = F[1];
      const computed = F[2];
      const nextFill = ramp(p, 0.04, 0.24);
      const boxes = windowOf(1).map((b) => (b.name === 'inNext_s' ? { ...b, fill: nextFill } : b));
      const progress = ramp(p, 0.42, 0.95);
      const k = Math.min(activeList.length - 1, Math.floor(progress * activeList.length));
      const computing = p >= 0.42 && p < 0.97;
      const thread = activeList[k];
      if (computing)
        for (const b of boxes)
          b.hot = b.name === 'inCurr_s' ? crossOf(thread) : [thread];
      const stores = computing ? Math.round(progress * computed.stores) : p >= 0.97 ? computed.stores : 0;
      return (
        <Stage block={1} active={computed.active} hotThread={computing ? thread : null} boxes={boxes}
          inShared={onChip(loaded)} barrier={p >= 0.26 && p < 0.42 ? tr('barrier 1: wait until the whole plane is in', 'rào chắn 1: chờ đến khi cả mặt phẳng đã vào') : null}
          outputs={{ [L(computed.outputPlanes[0])!]: stores / computed.stores }}
          counters={counters(nextFill >= 1 ? loaded : F[0], stores)}
          headline={p < 0.42 ? tr('load one new plane, then wait', 'nạp một mặt phẳng mới, rồi chờ') : tr(`${activeCount} threads: five values from the current plane, one before, one after`, `${activeCount} luồng: năm giá trị từ mặt phẳng hiện tại, một từ mặt phẳng trước, một từ mặt phẳng sau`)} />
      );
    },
  },
  {
    clip: 'z-march-4',
    get label() { return tr('Roll', 'Cuộn'); },
    draw: (p) => {
      const computed = F[2];
      const shifted = F[3];
      const slide = ramp(p, 0.3, 0.7);
      return (
        <Stage block={1} active={computed.active} boxes={slide > 0 ? windowOf(3, 2, slide) : windowOf(2)} inShared={onChip(slide >= 1 ? shifted : computed)}
          barrier={p < 0.3 ? tr('barrier 2: no overwriting until everyone has read', 'rào chắn 2: không ghi đè cho đến khi mọi luồng đã đọc') : null} arrows={ramp(p, 0.3, 0.4) * (1 - ramp(p, 0.75, 0.85))}
          outputs={outputsOf(computed)} counters={counters(shifted)}
          headline={p < 0.62 ? tr('barrier 2, then shift the window by one plane', 'rào chắn 2, rồi dịch cửa sổ đi một mặt phẳng') : tr('barrier 1: true dependence. barrier 2: false dependence', 'rào chắn 1: phụ thuộc thật. rào chắn 2: phụ thuộc giả')} />
      );
    },
  },
  {
    clip: 'z-march-5',
    get label() { return tr('March on', 'Tiến tiếp'); },
    draw: (p) => {
      const rest = (F.length - 4) / 3; // iterations after the first
      if (p < 0.72) {
        const x = (p / 0.72) * rest;
        const n = Math.min(rest - 1, Math.floor(x));
        return <Stage {...marchFrame(n + 1, x - n)} headline={tr(`iteration ${n + 2}: output plane ${L(F[2 + 3 * (n + 1)].i)}`, `vòng lặp ${n + 2}: mặt phẳng đầu ra ${L(F[2 + 3 * (n + 1)].i)}`)} />;
      }
      return (
        <Stage block={1} active={last.active} boxes={windowOf(F.length - 1)} inShared={onChip(last)} outputs={outputsOf(last)}
          counters={counters(last)} compare={ramp(p, 0.76, 0.86)}
          headline={tr(`${last.planesLoaded} input planes loaded for ${last.outputPlanes.length} output planes`, `đã nạp ${last.planesLoaded} mặt phẳng đầu vào cho ${last.outputPlanes.length} mặt phẳng đầu ra`)} />
      );
    },
  },
  {
    clip: 'z-march-6',
    get label() { return tr('Registers', 'Thanh ghi'); },
    draw: (p) => {
      const at = TLR.frames[5]; // registers kernel, iteration 1, after the compute
      const same = F[5]; // the planes kernel at the same point
      const peel = ramp(p, 0.3, 0.75);
      const away = (z: number | null): Box['at'] => lerp(L(z)!, -1.1, peel);
      const boxes: Box[] = [
        { name: 'inPrev_s', at: away(at.hold.prev), tone: 'shared', opacity: 1 - peel, noLink: peel > 0 }, // i18n-ignore: identifier
        { name: 'inCurr_s', at: L(at.sharedCurr)!, tone: 'shared' }, // i18n-ignore: identifier
        { name: 'inNext_s', at: away(at.hold.next), tone: 'shared', opacity: 1 - peel, noLink: peel > 0 }, // i18n-ignore: identifier
      ];
      return (
        <Stage block={1} active={at.active} boxes={boxes} regs={ramp(p, 0.55, 0.8)}
          inShared={peel >= 1 ? [L(at.sharedCurr)!] : onChip(same)} outputs={outputsOf(at)}
          counters={[
            [tr('loaded, registers', 'đã nạp, thanh ghi'), String(at.loads)],
            [tr('loaded, 3 planes', 'đã nạp, 3 mặt phẳng'), String(same.loads)],
            [tr('shared planes', 'mặt phẳng chia sẻ'), peel >= 1 ? '1' : '3'],
            [tr('bytes at t = 32', 'số byte ở t = 32'), sharedBytes(peel >= 1 ? 'registers' : 'planes', 32).toLocaleString('en-US')],
          ]}
          headline={p < 0.3 ? tr('previous and next are read by one thread only', 'giá trị trước và kế tiếp chỉ do một luồng đọc') : tr('into registers: shared memory keeps only the current plane', 'vào thanh ghi: bộ nhớ chia sẻ chỉ giữ mặt phẳng hiện tại')} />
      );
    },
  },
];

export function ZMarchScene() {
  return <Scene chapter="ch08" title={tr('Marching through z', 'Tiến dọc theo z')} beats={beats} />;
}
