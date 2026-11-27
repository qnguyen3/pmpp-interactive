import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { SCENE, sceneKeep, serveOneByOne } from '../../sim/filter';
import { binary, lanesOf, vote } from '../../sim/warpvote';
import { tr } from '../../lib/i18n';

// "One vote, one atomic" (section 12.3): an 8-lane toy warp appends its kept keys first
// the way Fig. 12.2 does (one atomic per key, served in an order nobody chose, twice, to
// show the order change), then the way Fig. 12.3 does (a vote, a leader, one atomic, a
// broadcast, a rank per lane). Beats 1 to 3 are drawn from src/sim/filter.ts, beats 4 to 6
// from src/sim/warpvote.ts, the models behind sections 12.1 to 12.3's interactives. Six
// beats, one narration clip each (content/ch12/narration/one-vote-N.txt).

const KEEP = sceneKeep();
const LANES = SCENE.keys.length;
const RUNS = SCENE.seeds.map((s) => serveOneByOne(KEEP, SCENE.counter, s));
const VOTE = vote({ width: 8, pass: KEEP, counter: SCENE.counter, form: 'book' });
const G = VOTE.groups[0];
const PASSING = VOTE.passing;

const W = 720;
const H = 384;
const X0 = 78; // first lane
const STEP = 66;
const CW = 58; // cell width
const LANE_Y = 92;
const CELL_H = 40;
const MASK_Y = 46;
const CTR_X = 628;
const CTR_Y = 178;
const OUT_Y = 262;
const OUT2_Y = 324;

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const lx = (lane: number) => X0 + lane * STEP;
const sx = (slot: number) => X0 + (slot - SCENE.counter) * STEP;

interface Frame {
  /** How far the keep test has run: lanes 0 .. tested - 1 show their verdict. */
  tested?: number;
  /** Write "keep" or "drop" under the tested lanes (beats 1 to 3). */
  verdicts?: boolean;
  counter: number;
  counterNote?: string;
  /** Arrows from lanes to the counter, with an opacity each. */
  toCounter?: { lane: number; o: number; label?: string }[];
  /** Lanes waiting in the counter's queue. */
  queue?: number[];
  /** Output rows: slot -> lane whose key is there (null: empty). */
  out?: (number | null)[];
  out2?: (number | null)[];
  outLabel?: string;
  out2Label?: string;
  /** Mask bits shown above the lanes (0..1 opacity) and the leader ring. */
  mask?: number;
  leader?: number;
  popc?: number;
  /** Shuffle arrows from the leader, 0..1. */
  shuffle?: number;
  /** Per-lane rank labels, 0..1. */
  ranks?: number;
  /** Arrows from lanes to their slots in the first output row. */
  toSlots?: { lane: number; slot: number; o: number }[];
  headline: string;
}

function Arrow({ x1, y1, x2, y2, o = 1, dash }: { x1: number; y1: number; x2: number; y2: number; o?: number; dash?: string }) {
  return <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--ink-2)" strokeWidth="1.4" markerEnd="url(#ov-head)" opacity={o} strokeDasharray={dash} />;
}

function OutRow({ y, row, label, name }: { y: number; row: (number | null)[]; label: string; name: string }) {
  return (
    <g data-out={name} data-values={row.map((l) => (l == null ? '-' : String(SCENE.keys[l]))).join(' ')}>
      <text x={8} y={y + 24} fontSize="11.5" style={{ fill: 'var(--global)' }}>{label}</text>
      {row.map((lane, k) => (
        <g key={k}>
          <rect x={sx(SCENE.counter + k)} y={y} width={CW} height={34} rx="3"
            fill={lane == null ? 'var(--cell)' : 'var(--global-wash)'} stroke="var(--global)" />
          {lane != null ? (
            <text x={sx(SCENE.counter + k) + CW / 2} y={y + 22} fontSize="13" textAnchor="middle">
              {SCENE.keys[lane]}
              <tspan fontSize="9.5" style={{ fill: 'var(--ink-2)' }}> L{lane}</tspan>
            </text>
          ) : null}
          <text x={sx(SCENE.counter + k) + CW / 2} y={y + 47} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>
            {SCENE.counter + k}
          </text>
        </g>
      ))}
    </g>
  );
}

function Stage(f: Frame) {
  const tested = f.tested ?? LANES;
  const verdicts = f.verdicts ?? true;
  const empty = PASSING.map(() => null) as (number | null)[];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('A warp of eight lanes appending its kept keys', 'Một warp tám lane nối thêm các khóa được giữ lại của nó')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <defs>
        <marker id="ov-head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">{/* i18n-ignore: SVG attribute */}
          <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--ink-2)' }} />
        </marker>
      </defs>
      {/* mask bits above the lanes */}
      {f.mask ? (
        <g opacity={f.mask} data-mask={binary(G.mask, 8)}>
          <text x={8} y={MASK_Y + 17} fontSize="11.5" style={{ fill: 'var(--reg)' }}>{tr('mask', 'mặt nạ')}</text>
          {KEEP.map((k, l) => (
            <g key={l}>
              <rect x={lx(l) + CW / 2 - 13} y={MASK_Y} width={26} height={24} rx="2" fill={k ? 'var(--reg)' : 'var(--cell)'} stroke="var(--reg)" />
              <text x={lx(l) + CW / 2} y={MASK_Y + 17} fontSize="13" textAnchor="middle" style={{ fill: k ? 'var(--page)' : 'var(--ink)' }}>{k ? 1 : 0}</text>
            </g>
          ))}
          <text x={X0} y={MASK_Y - 9} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>
            {tr('written lane 7 first: ', 'viết lane 7 trước: ')}{binary(G.mask, 8)}{f.popc != null ? tr(`   ·   ones in it: ${f.popc}`, `   ·   số bit 1 trong đó: ${f.popc}`) : ''}
          </text>
        </g>
      ) : null}
      {/* lanes */}
      <text x={8} y={LANE_Y + 25} fontSize="11.5" style={{ fill: 'var(--reg)' }}>{tr('keys', 'khóa')}</text>
      {SCENE.keys.map((key, l) => {
        const done = l < tested;
        const pass = KEEP[l];
        const dim = done && !pass;
        const leader = f.leader === l;
        return (
          <g key={l} data-lane={l} data-pass={done ? (pass ? 1 : 0) : ''}>
            <text x={lx(l) + CW / 2} y={LANE_Y - 6} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{tr(`lane ${l}`, `lane ${l}`)}</text>
            <rect x={lx(l)} y={LANE_Y} width={CW} height={CELL_H} rx="3"
              fill={done && pass ? 'var(--reg-wash)' : 'var(--cell)'} stroke={leader ? 'var(--ink)' : dim ? 'var(--rule)' : 'var(--reg)'}
              strokeWidth={leader ? 2.6 : 1} opacity={dim ? 0.55 : 1} />
            <text x={lx(l) + CW / 2} y={LANE_Y + 26} fontSize="15" textAnchor="middle" opacity={dim ? 0.55 : 1}>{key}</text>
            {done && verdicts ? (
              <text x={lx(l) + CW / 2} y={LANE_Y + CELL_H + 16} fontSize="10.5" textAnchor="middle" style={{ fill: pass ? 'var(--ink)' : 'var(--ink-3)' }}>
                {pass ? tr('keep', 'giữ') : tr('drop', 'bỏ')}
              </text>
            ) : null}
            {leader ? (
              <text x={lx(l) + CW / 2} y={LANE_Y + CELL_H + 16} fontSize="10.5" fontWeight="650" textAnchor="middle">{tr('leader', 'trưởng nhóm')}</text>
            ) : null}
          </g>
        );
      })}
      {/* rank labels */}
      {f.ranks
        ? VOTE.lanes.filter((L) => L.writes).map((L) => (
            <g key={L.lane} opacity={f.ranks} data-rank-lane={L.lane} data-rank={L.rank}>
              <text x={lx(L.lane) + CW / 2} y={LANE_Y + CELL_H + 34} fontSize="10" textAnchor="middle" style={{ fill: 'var(--reg)' }}>
                {lanesOf(L.and!, 8).length ? tr('below: ', 'phía dưới: ') + lanesOf(L.and!, 8).join(', ') : tr('none below', 'không có ai phía dưới')}
              </text>
              <text x={lx(L.lane) + CW / 2} y={LANE_Y + CELL_H + 48} fontSize="11" fontWeight="600" textAnchor="middle" style={{ fill: 'var(--reg)' }}>
                {tr('rank', 'hạng')} {L.rank}
              </text>
            </g>
          ))
        : null}
      {/* counter */}
      <g data-counter={f.counter}>
        <text x={CTR_X} y={CTR_Y - 8} fontSize="11.5" style={{ fill: 'var(--global)' }}>{tr('counter', 'bộ đếm')}</text>
        <rect x={CTR_X} y={CTR_Y} width={70} height={40} rx="3" fill="var(--global-wash)" stroke="var(--global)" />
        <text x={CTR_X + 35} y={CTR_Y + 26} fontSize="17" fontWeight="650" textAnchor="middle" data-value="">{f.counter}</text>
        {f.counterNote ? <text x={CTR_X + 72} y={CTR_Y + 58} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{f.counterNote}</text> : null}
        {f.queue && f.queue.length ? (
          <text x={CTR_X + 72} y={CTR_Y + 58} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }} data-queue={f.queue.join(' ')}>
            {tr('waiting: ', 'đang chờ: ')}{f.queue.join(', ')}
          </text>
        ) : null}
      </g>
      {f.toCounter?.map((a, k) => (
        <g key={'c' + k}>
          <g opacity={a.o}>
            <Arrow x1={lx(a.lane) + CW / 2} y1={LANE_Y + CELL_H + 22} x2={CTR_X - 2} y2={CTR_Y + 20} />
          </g>
          {a.label ? <text x={CTR_X + 72} y={CTR_Y + 76 + k * 14} fontSize="10.5" textAnchor="end" data-served={a.label}>{a.label}</text> : null}
        </g>
      ))}
      {/* shuffle from the leader */}
      {f.shuffle && f.leader != null
        ? PASSING.filter((l) => l !== f.leader).map((l) => {
            const x1 = lx(f.leader!) + CW / 2;
            const x2 = lx(l) + CW / 2;
            const y0 = LANE_Y + CELL_H + 22;
            const dip = y0 + 18 + Math.abs(l - f.leader!) * 9;
            return (
              <g key={'s' + l} opacity={f.shuffle} data-shuffle={l}>
                <path d={`M${x1},${y0} C${x1},${dip} ${x2},${dip} ${x2},${LANE_Y + CELL_H + 4}`} fill="none" stroke="var(--reg)" strokeWidth="1.4" markerEnd="url(#ov-head)" />
                <text x={x2 + 8} y={LANE_Y + CELL_H + 18} fontSize="11" fontWeight="600" style={{ fill: 'var(--reg)' }}>{G.start}</text>
              </g>
            );
          })
        : null}
      {f.toSlots?.map((a) => (
        <Arrow key={'o' + a.lane} x1={lx(a.lane) + CW / 2} y1={LANE_Y + CELL_H + 54} x2={sx(a.slot) + CW / 2} y2={OUT_Y - 4} o={a.o} />
      ))}
      <OutRow y={OUT_Y} row={f.out ?? empty} label={f.outLabel ?? tr('output', 'đầu ra')} name="1" />
      {f.out2 ? <OutRow y={OUT2_Y} row={f.out2} label={f.out2Label ?? tr('run 2', 'lần chạy 2')} name="2" /> : null}
      <text x={8} y={18} fontSize="13.5" fontWeight="620" data-headline="">{f.headline}</text>
    </svg>
  );
}

/** Fig. 12.2 serving run r up to time p: how many served, the output row, the queue. */
function serving(r: number, p: number) {
  const run = RUNS[r];
  const served = run.filter((_, k) => p >= 0.16 + k * 0.19).length;
  const out: (number | null)[] = PASSING.map(() => null);
  run.slice(0, served).forEach((s) => (out[s.slot - SCENE.counter] = s.lane));
  return { run, served, out, queue: run.slice(served).map((s) => s.lane) };
}

const outOf = (r: number) => {
  const out: (number | null)[] = PASSING.map(() => null);
  RUNS[r].forEach((s) => (out[s.slot - SCENE.counter] = s.lane));
  return out;
};

const beats: Beat[] = [
  {
    clip: 'one-vote-1',
    get label() { return tr('Eight lanes', 'Tám lane'); },
    draw: (p) => {
      const tested = Math.min(LANES, Math.floor(ramp(p, 0.08, 0.7) * (LANES + 0.999)));
      return (
        <Stage tested={tested} counter={SCENE.counter}
          headline={tested < LANES ? tr(`each lane tests its key: keep it if key < ${SCENE.threshold}`, `mỗi lane kiểm tra khóa của nó: giữ lại nếu key < ${SCENE.threshold}`) : tr(`lanes ${PASSING.join(', ')} pass: each needs one slot`, `các lane ${PASSING.join(', ')} vượt qua: mỗi lane cần một ô`)} />
      );
    },
  },
  {
    clip: 'one-vote-2',
    get label() { return tr('One atomic per key', 'Một atomic cho mỗi khóa'); },
    draw: (p) => {
      const s = serving(0, p);
      return (
        <Stage counter={SCENE.counter + s.served} queue={s.queue} out={s.out} outLabel={tr('run 1', 'lần chạy 1')}
          toCounter={s.run.map((x, k) => ({ lane: x.lane, o: k < s.served ? 0.25 : ramp(p, 0, 0.12), label: k < s.served ? tr(`lane ${x.lane} got ${x.slot}`, `lane ${x.lane} nhận ${x.slot}`) : undefined }))}
          headline={s.served < PASSING.length ? tr(`${PASSING.length} atomics on one counter, served one at a time`, `${PASSING.length} atomic trên một bộ đếm, được phục vụ lần lượt từng cái`) : tr(`${PASSING.length} atomics: lane ${s.run.map((x) => x.lane).join(', lane ')} got ${SCENE.counter} to ${SCENE.counter + PASSING.length - 1}`, `${PASSING.length} atomic: lane ${s.run.map((x) => x.lane).join(', lane ')} nhận ${SCENE.counter} đến ${SCENE.counter + PASSING.length - 1}`)} />
      );
    },
  },
  {
    clip: 'one-vote-3',
    get label() { return tr('Run it again', 'Chạy lại lần nữa'); },
    draw: (p) => {
      const s = serving(1, p);
      return (
        <Stage counter={SCENE.counter + s.served} queue={s.queue} out={outOf(0)} outLabel={tr('run 1', 'lần chạy 1')} out2={s.out} out2Label={tr('run 2', 'lần chạy 2')}
          toCounter={s.run.map((x, k) => ({ lane: x.lane, o: k < s.served ? 0.25 : ramp(p, 0, 0.12), label: k < s.served ? tr(`lane ${x.lane} got ${x.slot}`, `lane ${x.lane} nhận ${x.slot}`) : undefined }))}
          headline={s.served < PASSING.length ? tr('same warp, same keys, run again', 'cùng warp, cùng khóa, chạy lại') : tr('same keys, another order: the filter is unstable', 'cùng khóa, thứ tự khác: bộ lọc không ổn định')} />
      );
    },
  },
  {
    clip: 'one-vote-4',
    get label() { return tr('The vote', 'Bỏ phiếu'); },
    draw: (p) => (
      <Stage verdicts={false} counter={SCENE.counter} mask={ramp(p, 0.1, 0.35)} leader={p >= 0.5 ? G.leader : undefined} popc={p >= 0.75 ? G.count : undefined}
        headline={p < 0.5 ? tr('the passing lanes vote: one bit per lane', 'các lane vượt qua bỏ phiếu: mỗi lane một bit') : p < 0.75 ? tr(`lowest set bit: lane ${G.leader} leads`, `bit 1 thấp nhất: lane ${G.leader} làm trưởng nhóm`) : tr(`leader lane ${G.leader}; the mask has ${G.count} ones`, `trưởng nhóm là lane ${G.leader}; mặt nạ có ${G.count} bit 1`)} />
    ),
  },
  {
    clip: 'one-vote-5',
    get label() { return tr('One atomic', 'Một atomic'); },
    draw: (p) => {
      const done = p >= 0.32;
      return (
        <Stage verdicts={false} counter={done ? G.atomic!.after : G.atomic!.before} mask={1} leader={G.leader} popc={G.count}
          counterNote={done ? tr(`lane ${G.leader} added ${G.count}, got ${G.start}`, `lane ${G.leader} cộng ${G.count}, nhận ${G.start}`) : undefined}
          toCounter={[{ lane: G.leader, o: ramp(p, 0.05, 0.2) * (1 - ramp(p, 0.5, 0.58)) }]} shuffle={ramp(p, 0.58, 0.75)}
          headline={!done ? tr(`only lane ${G.leader} touches the counter`, `chỉ lane ${G.leader} chạm vào bộ đếm`) : p < 0.55 ? tr(`one atomic: ${G.atomic!.before} becomes ${G.atomic!.after}`, `một atomic: ${G.atomic!.before} thành ${G.atomic!.after}`) : tr(`a shuffle hands ${G.start} to the other passing lanes`, `một lệnh shuffle chuyển ${G.start} cho các lane vượt qua còn lại`)} />
      );
    },
  },
  {
    clip: 'one-vote-6',
    get label() { return tr('Ranks and slots', 'Hạng và ô'); },
    draw: (p) => {
      const writers = VOTE.lanes.filter((L) => L.writes);
      const landed = writers.filter((_, k) => p >= 0.45 + k * 0.1).length;
      const out: (number | null)[] = PASSING.map(() => null);
      writers.slice(0, landed).forEach((L) => (out[L.slot! - SCENE.counter] = L.lane));
      return (
        <Stage verdicts={false} counter={G.atomic!.after} mask={1} leader={G.leader} popc={G.count} ranks={ramp(p, 0.08, 0.3)} out={out}
          toSlots={writers.map((L, k) => ({ lane: L.lane, slot: L.slot!, o: ramp(p, 0.4 + k * 0.1, 0.47 + k * 0.1) }))}
          headline={landed < writers.length ? tr('rank = passing lanes below me', 'hạng = số lane vượt qua nằm dưới tôi') : tr(`slots ${SCENE.counter} to ${SCENE.counter + writers.length - 1}, in lane order, after ${VOTE.atomics} atomic instead of ${writers.length}`, `các ô ${SCENE.counter} đến ${SCENE.counter + writers.length - 1}, theo thứ tự lane, sau ${VOTE.atomics} atomic thay vì ${writers.length}`)} />
      );
    },
  },
];

export function OneVoteScene(): ReactNode {
  return <Scene chapter="ch12" title={tr('One vote, one atomic', 'Một lần bỏ phiếu, một atomic')} beats={beats} />;
}
