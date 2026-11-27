import { useId, type ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { SCENE, acceptMasks, binCentre, diverges, sceneCase, type Outcome } from '../../sim/cutoff';
import { Translated, tr } from '../../lib/i18n';

// "A block walks its neighbourhood": one block of the binned cutoff kernel (p. 526) on the
// scene case of src/sim/cutoff.ts (a 9 × 9 box of 4 × 4-point bins, 160 atoms, capacity 4,
// a super circle of 2 bin widths). Every drawing comes from blockWalk; nothing is typed in.
// Six beats, one narration clip each (content/ch21/narration/binned-block-N.txt).
const { world, binning, walk } = sceneCase();
const [BX, BY] = SCENE.block;
const S = world.side;
const CAP = SCENE.capacity;
const LIST = walk.list;
/**
 * The block's lanes as one group. The scene's block has 16 threads (4 × 4 points), fewer than
 * a warp or wavefront of 32 or 64 lanes (ch05-facts B5), so the whole block sits in one group
 * whatever the backend's width; no width is assumed here.
 */
const GROUP = walk.points.length;
/** Beat 3's atom: the first staged real atom that splits the warp, and a dummy slot of the same entry. */
const DIV = (() => {
  for (const e of LIST) {
    const k = e.slots.findIndex((s, j) => s >= 0 && diverges(acceptMasks(e, j, GROUP)[0]));
    const d = e.slots.findIndex((s) => s < 0);
    if (k >= 0 && d >= 0) return { e, k, d };
  }
  return { e: LIST[0], k: 0, d: 0 };
})();
const OVERFLOW = new Set(binning.overflow);

const BP = 34; // px per bin
const OX = 16;
const OY = 40;
const RX = 360; // right-hand panel
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const gx = (x: number) => OX + ((x + 0.5) / S) * BP;
const gy = (y: number) => OY + ((y + 0.5) / S) * BP;
const thousands = (v: number) => v.toLocaleString('en-US');

interface Frame {
  headline: ReactNode;
  /** List entries lit so far (beat 1), or all. */
  lit: number;
  /** The entry being processed, or null. */
  entry?: number | null;
  /** Slot whose test is shown on the lanes, or null. */
  slot?: number | null;
  staged?: boolean;
  /** Show the block's threads and what they make of the staged atoms. */
  lanes?: boolean;
  barrier?: string | null;
  /** Running counters (tested, accepted, dummy tests). */
  counters?: { tested: number; accepted: number; dummy: number; k: number } | null;
  overflow?: boolean;
  timeline?: number;
  tally?: boolean;
}

function Stage(f: Frame) {
  const hatch = useId();
  const e = f.entry != null ? LIST[f.entry] : null;
  const slots = e ? e.slots : [];
  const outcome: Outcome[] | null = e && f.slot != null ? e.outcome[f.slot] : null;
  // Per lane, for the whole entry: did it accept anything (beat 4's lane picture).
  const anyAccept = e && f.slot == null ? walk.points.map((_, t) => e.outcome.some((row) => row[t] === 'accept')) : null;
  const [cx, cy] = binCentre(world, BX, BY);
  return (
    <svg viewBox="0 0 720 372" role="img" aria-label={tr('One block walking its list of neighbour bins', 'Một khối đi qua danh sách các ngăn lân cận của nó')}
      style={{ width: '100%', maxWidth: 720, display: 'block' }}
      data-lit={f.lit} data-entry={f.entry ?? ''} data-slot={f.slot ?? ''}
      data-tested={f.counters?.tested ?? ''} data-accepted={f.counters?.accepted ?? ''} data-dummy={f.counters?.dummy ?? ''}>
      <defs>
        <pattern id={hatch} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" strokeWidth="1" />
        </pattern>
      </defs>
      <text x={16} y={24} fontSize="14" fontWeight="600" data-headline="">{f.headline}</text>
      {/* bins */}
      {Array.from({ length: world.binsX * world.binsY }, (_, k) => {
        const x = k % world.binsX;
        const y = Math.floor(k / world.binsX);
        const li = LIST.findIndex((q) => q.bx === x && q.by === y);
        const lit = li >= 0 && li < f.lit;
        const cur = e && e.bx === x && e.by === y;
        return (
          <rect key={k} x={OX + x * BP} y={OY + y * BP} width={BP} height={BP}
            fill={lit ? 'var(--shared)' : 'var(--cell)'} fillOpacity={lit ? (LIST[li].rel.cover === 'full' ? 0.32 : 0.14) : 1}
            stroke={cur ? 'var(--ink)' : 'var(--rule)'} strokeWidth={cur ? 2.4 : 0.6} />
        );
      })}
      <rect x={OX + BX * BP} y={OY + BY * BP} width={BP} height={BP} fill="none" stroke="var(--ink)" strokeWidth="2" />
      <circle cx={gx(cx)} cy={gy(cy)} r={SCENE.R * BP} fill="none" stroke="var(--ink)" strokeWidth="1.3" strokeDasharray="6 4" />
      {walk.points.map(([i, j], t) => <circle key={t} cx={gx(i)} cy={gy(j)} r={1.5} fill="var(--ink)" />)}
      {world.atoms.map((a, k) => {
        const staged = f.staged && e && e.slots.includes(k);
        const over = f.overflow && OVERFLOW.has(k);
        return (
          <circle key={k} cx={gx(a.x)} cy={gy(a.y)} r={staged || over ? 3.6 : 2}
            fill={over ? 'var(--fail)' : 'var(--global)'} fillOpacity={staged || over ? 1 : 0.6}
            stroke={staged ? 'var(--ink)' : over ? 'var(--fail)' : 'none'} strokeWidth={over ? 2 : 1} data-overflow={over ? '' : undefined} />
        );
      })}

      {/* shared memory: the staged bin */}
      {e && f.staged ? (
        <g data-staged={slots.join(',')}>
          <text x={RX} y={60} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--shared)' }}>
            <Translated vi={<>bộ nhớ chia sẻ: ngăn ({e.bx}, {e.by}), độ lệch ({e.rel.dx}, {e.rel.dy}), {CAP} ô chứa</>}>
              shared memory: bin ({e.bx}, {e.by}), offset ({e.rel.dx}, {e.rel.dy}), {CAP} slots
            </Translated>
          </text>
          {slots.map((s, k) => (
            <g key={k}>
              <rect x={RX + k * 84} y={68} width={78} height={30} rx="3"
                fill={s < 0 ? `url(#${hatch})` : f.slot === k ? 'var(--shared)' : 'var(--shared-wash)'} stroke="var(--shared)" strokeWidth={f.slot === k ? 2.2 : 1} />
              <text x={RX + k * 84 + 39} y={87} fontSize="11" textAnchor="middle" style={{ fill: f.slot === k && s >= 0 ? 'var(--page)' : 'var(--ink)' }}>
                {s < 0 ? tr('dummy, q 0', 'giả, q 0') : tr(`atom ${s}`, `nguyên tử ${s}`)}
              </text>
            </g>
          ))}
        </g>
      ) : null}
      {f.barrier ? (
        <g>
          <rect x={RX} y={108} width={330} height={22} rx="3" fill="var(--ink)" />
          <text x={RX + 165} y={123} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--page)' }}>{f.barrier}</text>
        </g>
      ) : null}
      {/* the block's 16 threads */}
      {e && f.staged && f.lanes ? (
        <g>
          <text x={RX} y={152} fontSize="11.5" fontWeight="600">
            <Translated vi={<>{walk.points.length} luồng của khối</>}>the block's {walk.points.length} threads</Translated>{outcome ? (slots[f.slot!] < 0 ? tr(': a dummy, tested by every lane', ': một nguyên tử giả, mọi lane đều kiểm tra') : tr(`: test atom ${slots[f.slot!]}`, `: kiểm tra nguyên tử ${slots[f.slot!]}`)) : tr(': accepted anything from this bin?', ': có nhận gì từ ngăn này không?')}
          </text>
          {walk.points.map((_, t) => {
            const o = outcome ? outcome[t] : anyAccept![t] ? 'accept' : 'reject';
            return (
              <g key={t} data-lane={t} data-outcome={o}>
                <rect x={RX + (t % S) * 34} y={160 + Math.floor(t / S) * 30} width={30} height={26} rx="3"
                  fill={o === 'dummy' ? `url(#${hatch})` : o === 'accept' ? 'var(--ok)' : 'var(--cell)'} stroke={o === 'accept' ? 'var(--ok)' : 'var(--rule)'} />
                <text x={RX + (t % S) * 34 + 15} y={177 + Math.floor(t / S) * 30} fontSize="9.5" textAnchor="middle"
                  style={{ fill: o === 'accept' ? 'var(--page)' : 'var(--ink-2)' }}>{t}</text>
              </g>
            );
          })}
          {outcome && slots[f.slot!] >= 0 ? (
            <text x={RX + S * 34 + 12} y={200} fontSize="11.5" style={{ fill: diverges(outcome.map((o) => o === 'accept')) ? 'var(--fail)' : 'var(--ink-2)' }}>
              <Translated vi={<>{outcome.filter((o) => o === 'accept').length} nhận, {outcome.filter((o) => o === 'reject').length} loại</>}>
                {outcome.filter((o) => o === 'accept').length} accept, {outcome.filter((o) => o === 'reject').length} reject
              </Translated>
            </text>
          ) : null}
        </g>
      ) : null}
      {/* counters */}
      {f.counters ? (
        <g data-counters="">
          <text x={RX} y={300} fontSize="12"><Translated vi={<>mục {f.counters.k + 1} trên {LIST.length} của danh sách</>}>list entry {f.counters.k + 1} of {LIST.length}</Translated></text>
          <text x={RX} y={318} fontSize="12"><Translated vi={<>số phép kiểm tra đến giờ: <tspan fontWeight="650">{thousands(f.counters.tested)}</tspan> ({thousands(f.counters.dummy)} trên nguyên tử giả)</>}>tests so far: <tspan fontWeight="650">{thousands(f.counters.tested)}</tspan> ({thousands(f.counters.dummy)} on dummies)</Translated></text>
          <text x={RX} y={336} fontSize="12"><Translated vi={<>đã nhận đến giờ: <tspan fontWeight="650">{thousands(f.counters.accepted)}</tspan></>}>accepted so far: <tspan fontWeight="650">{thousands(f.counters.accepted)}</tspan></Translated></text>
        </g>
      ) : null}
      {f.tally ? (
        <g data-tally="">
          <rect x={RX - 6} y={140} width={354} height={130} fill="var(--page)" />
          <text x={RX} y={162} fontSize="13" fontWeight="650"><Translated vi={<>khối này, {walk.points.length} luồng, {LIST.length} ngăn</>}>this block, {walk.points.length} threads, {LIST.length} bins</Translated></text>
          <text x={RX} y={186} fontSize="12.5"><Translated vi={<>phép kiểm tra: {thousands(walk.tested)} ({thousands(walk.dummyTests)} trên nguyên tử giả)</>}>tests: {thousands(walk.tested)} ({thousands(walk.dummyTests)} on dummy atoms)</Translated></text>
          <text x={RX} y={208} fontSize="12.5"><Translated vi={<>đã nhận: {thousands(walk.accepted)} ({Math.round((walk.accepted / walk.tested) * 100)} phần trăm)</>}>accepted: {thousands(walk.accepted)} ({Math.round((walk.accepted / walk.tested) * 100)} percent)</Translated></text>
          <text x={RX} y={236} fontSize="12.5" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>tổng trực tiếp, cùng các điểm: {walk.points.length} × {world.atoms.length} = {thousands(walk.directPairs)} cặp</>}>direct summation, same points: {walk.points.length} × {world.atoms.length} = {thousands(walk.directPairs)} pairs</Translated></text>
          <text x={RX} y={256} fontSize="11.5" style={{ fill: 'var(--ink-3)' }}><Translated vi="(con số đó tăng theo hộp; danh sách thì không)">(that grows with the box; the list does not)</Translated></text>
        </g>
      ) : null}
      {f.timeline != null ? (
        <g opacity={f.timeline} data-timeline="">
          <text x={RX} y={160} fontSize="12" fontWeight="600"><Translated vi="một thể tích">one volume</Translated></text>
          <rect x={RX} y={168} width={200} height={20} fill="var(--global-wash)" stroke="var(--global)" />
          <text x={RX + 100} y={182} fontSize="10.5" textAnchor="middle"><Translated vi="kernel (thiết bị)">kernel (device)</Translated></text>
          <rect x={RX + 204} y={168} width={96} height={20} fill="var(--fail-wash)" stroke="var(--fail)" />
          <text x={RX + 252} y={182} fontSize="10.5" textAnchor="middle"><Translated vi="host: phần tràn">host: overflow</Translated></text>
          <text x={RX} y={220} fontSize="12" fontWeight="600"><Translated vi="các thể tích con">sub-volumes</Translated></text>
          {[0, 1, 2].map((k) => (
            <g key={k}>
              <rect x={RX + k * 92} y={228} width={88} height={20} fill="var(--global-wash)" stroke="var(--global)" />
              <text x={RX + k * 92 + 44} y={242} fontSize="10.5" textAnchor="middle"><Translated vi={<>kernel {k + 1}</>}>kernel {k + 1}</Translated></text>
              <rect x={RX + (k + 1) * 92} y={252} width={70} height={20} fill="var(--fail-wash)" stroke="var(--fail)" />
              <text x={RX + (k + 1) * 92 + 35} y={266} fontSize="10" textAnchor="middle"><Translated vi={<>host {k + 1}</>}>host {k + 1}</Translated></text>
            </g>
          ))}
          <text x={RX} y={300} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>{binning.overflow.length} nguyên tử tràn trên {world.atoms.length}; sức chứa {CAP}, ngăn lớn nhất {Math.max(...binning.occupancy)}</>}>{binning.overflow.length} overflow atoms of {world.atoms.length}; capacity {CAP}, largest bin {Math.max(...binning.occupancy)}</Translated></text>
        </g>
      ) : null}
    </svg>
  );
}

const countersAt = (k: number) => ({ tested: LIST[k].cumTested, accepted: LIST[k].cumAccepted, dummy: LIST.slice(0, k + 1).reduce((s, e) => s + e.dummyTests, 0), k });

const beats: Beat[] = [
  {
    clip: 'binned-block-1',
    get label() { return tr('The list', 'Danh sách'); },
    draw: (p) => {
      const lit = Math.round(ramp(p, 0.15, 0.8) * LIST.length);
      return <Stage lit={lit} headline={lit < LIST.length ? tr(`bins the super circle touches: ${lit}`, `số ngăn mà vòng tròn lớn chạm tới: ${lit}`) : tr(`${LIST.length} bins: a list of relative offsets, made once on the host`, `${LIST.length} ngăn: một danh sách độ lệch tương đối, lập một lần trên host`)} />;
    },
  },
  {
    clip: 'binned-block-2',
    get label() { return tr('Stage a bin', 'Nạp một ngăn'); },
    draw: (p) => (
      <Stage lit={LIST.length} entry={0} staged={p > 0.2} barrier={p > 0.6 ? tr('barrier: wait until the whole bin is in shared memory', 'rào chắn: chờ đến khi cả ngăn đã nằm trong bộ nhớ chia sẻ') : null}
        headline={tr(`offset (${LIST[0].rel.dx}, ${LIST[0].rel.dy}): the block loads the bin together`, `độ lệch (${LIST[0].rel.dx}, ${LIST[0].rel.dy}): cả khối cùng nạp ngăn`)} />
    ),
  },
  {
    clip: 'binned-block-3',
    get label() { return tr('Test and diverge', 'Kiểm tra và phân kỳ'); },
    draw: (p) => {
      const dummy = p >= 0.62;
      const e = DIV.e;
      return (
        <Stage lit={LIST.length} entry={e.k} staged lanes slot={dummy ? DIV.d : DIV.k}
          counters={null}
          headline={dummy ? tr('a dummy atom: every lane tests it, none gains anything', 'một nguyên tử giả: mọi lane đều kiểm tra nó, không lane nào được gì') : tr(`atom ${e.slots[DIV.k]}: some lanes accept, some reject; the warp diverges`, `nguyên tử ${e.slots[DIV.k]}: vài lane nhận, vài lane loại; warp phân kỳ`)} />
      );
    },
  },
  {
    clip: 'binned-block-4',
    get label() { return tr('Down the list', 'Đi hết danh sách'); },
    draw: (p) => {
      const k = Math.min(LIST.length - 1, 1 + Math.floor(ramp(p, 0.08, 0.95) * (LIST.length - 1)));
      const local = (ramp(p, 0.08, 0.95) * (LIST.length - 1)) % 1;
      return (
        <Stage lit={LIST.length} entry={k} staged lanes counters={countersAt(k)}
          barrier={local > 0.75 ? tr('barrier: no overwriting until every thread has read', 'rào chắn: không ghi đè cho đến khi mọi luồng đã đọc xong') : null}
          headline={tr(`entry ${k + 1}: offset (${LIST[k].rel.dx}, ${LIST[k].rel.dy}), ${LIST[k].accepted} accepted of ${LIST[k].tested} tests`, `mục ${k + 1}: độ lệch (${LIST[k].rel.dx}, ${LIST[k].rel.dy}), ${LIST[k].accepted} được nhận trên ${LIST[k].tested} phép kiểm tra`)} />
      );
    },
  },
  {
    clip: 'binned-block-5',
    get label() { return tr('Tally', 'Tổng kết'); },
    draw: () => <Stage lit={LIST.length} tally counters={countersAt(LIST.length - 1)} headline={tr('the end of the list', 'cuối danh sách')} />,
  },
  {
    clip: 'binned-block-6',
    get label() { return tr('Overflow', 'Phần tràn'); },
    draw: (p) => (
      <Stage lit={LIST.length} overflow timeline={ramp(p, 0.3, 0.6)}
        headline={tr(`${binning.overflow.length} atoms did not fit their bins: the host adds them afterwards`, `${binning.overflow.length} nguyên tử không vừa ngăn của chúng: host cộng chúng vào sau`)} />
    ),
  },
];

export function BinnedBlockScene() {
  return <Scene chapter="ch21" title={tr('A block walks its neighbourhood', 'Một khối đi qua vùng lân cận của nó')} beats={beats} />;
}
