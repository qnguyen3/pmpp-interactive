import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { SceneReadout, useNarrow } from './sceneText';
import {
  FIG45,
  TAIL_EXAMPLE,
  capacity,
  completionOrder,
  profile,
  schedule,
  shuffledOrder,
  tail,
  unevenDurations,
  usage,
  waveSizes,
  waves,
  type Schedule,
} from '../../sim/blocksched';
import { Translated, tr } from '../../lib/i18n';

// "One grid, two devices": the book's Fig. 4.5 and its tail-effect example (pp. 73-74)
// in time. Six beats, one narration clip each (content/ch04/narration/waves-N.txt).
// Every block position, count and percentage drawn comes from sim/blocksched.ts.
const W = 720;
const H = 348;
/** The strip at the bottom that holds the headline and the line under it. */
const FOOT = 44;
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

const SMALL = schedule({ blocks: FIG45.blocks, ...FIG45.small });
const LARGE = schedule({ blocks: FIG45.blocks, ...FIG45.large });
export const SHUFFLE = shuffledOrder(FIG45.blocks, 3);
const SHUFFLED = schedule({ blocks: FIG45.blocks, ...FIG45.large, order: SHUFFLE });

const CAP = capacity(TAIL_EXAMPLE.units, TAIL_EXAMPLE.slotsPerUnit);
const TAILED = schedule(TAIL_EXAMPLE);
export const EVEN_BLOCKS = Math.ceil(waves(TAIL_EXAMPLE.blocks, CAP)) * CAP; // the next multiple of the capacity
const EVENED = schedule({ ...TAIL_EXAMPLE, blocks: EVEN_BLOCKS });
const UNEVEN = schedule({ ...TAIL_EXAMPLE, blocks: EVEN_BLOCKS, durations: unevenDurations(EVEN_BLOCKS) });
const unevenUse = (wavesWanted: number) =>
  usage(schedule({ ...TAIL_EXAMPLE, blocks: wavesWanted * CAP, durations: unevenDurations(wavesWanted * CAP) })).utilisation;
export const UNEVEN_ONE = unevenUse(1);
export const UNEVEN_TEN = unevenUse(10);
const pct = (v: number) => Math.round(v * 100);

/** A block as a box with its number, optionally with its threads and a barrier drawn inside. */
function Block({ x, y, w = 34, h = 24, n, on = false, inside = false, faded = false }: {
  x: number; y: number; w?: number; h?: number; n: number; on?: boolean; inside?: boolean; faded?: boolean;
}) {
  return (
    <g opacity={faded ? 0.35 : 1} data-block={n}>
      <rect x={x} y={y} width={w} height={h} rx="3" fill={on ? 'var(--ink-2)' : 'var(--cell)'} stroke="var(--ink)" strokeWidth="1.2" />
      {inside ? (
        <>
          {[0, 1, 2, 3].map((k) => (
            <path key={k} d={[18, 60].map((top) => `M ${x + 14 + k * ((w - 28) / 3)} ${y + top} q 4 5 0 10 q -4 5 0 10 q 4 5 0 10`).join(' ')}
              fill="none" stroke="var(--ink-2)" strokeWidth="1" />
          ))}
          <line x1={x + 5} x2={x + w - 5} y1={y + h / 2 + 2} y2={y + h / 2 + 2} stroke="var(--ink)" strokeWidth="2" />
          <text x={x + w / 2} y={y + 11} fontSize="10" textAnchor="middle"><Translated vi={<>khối {n}</>}>block {n}</Translated></text>
        </>
      ) : (
        <text x={x + w / 2} y={y + h / 2 + 4} fontSize="11.5" textAnchor="middle" style={{ fill: on ? 'var(--page)' : 'var(--ink)' }}>{n}</text>
      )}
    </g>
  );
}

/** The runtime's list: the blocks that have not started at time t, next first. */
function Pending({ x, y, s, t, shown = 1 }: { x: number; y: number; s: Schedule; t: number; shown?: number }) {
  const waiting = s.spec.order.filter((b) => s.placements[b].start > t);
  const visible = Math.ceil(waiting.length * shown);
  return (
    <g data-part="pending">
      <text x={x} y={y - 8} fontSize="11.5" fontWeight="600"><Translated vi="chưa bắt đầu">not started</Translated></text>
      {waiting.slice(0, visible).map((b, i) => <Block key={b} x={x + (i % 2) * 38} y={y + Math.floor(i / 2) * 28} n={b} />)}
      {waiting.length === 0 ? <text x={x} y={y + 14} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi="không còn khối nào">nothing left</Translated></text> : null}
    </g>
  );
}

/**
 * A device as in Fig. 4.5: its slots on top, and below them, time running down the
 * page, every block that has started by time t in the slot and at the time it ran.
 */
function DeviceRun({ x, y, s, t, name, dim = false }: { x: number; y: number; s: Schedule; t: number; name: string; dim?: boolean }) {
  const cap = s.capacity;
  const cw = 40;
  const rowH = 30;
  const col = (unit: number, slot: number) => slot * s.spec.units + unit;
  return (
    <g opacity={dim ? 0.5 : 1} data-part="device" data-capacity={cap}>
      <text x={x} y={y - 8} fontSize="11.5" fontWeight="600">{name}</text>
      <rect x={x - 5} y={y} width={cap * cw + 8} height={34} rx="4" fill="var(--bench)" stroke="var(--ink-3)" />
      {Array.from({ length: cap }, (_, k) => (
        <rect key={k} x={x + k * cw} y={y + 5} width={cw - 4} height={24} rx="3" fill="none" stroke="var(--ink-3)" strokeDasharray="3 3" />
      ))}
      <g data-part="running">
        {s.placements.filter((p) => p.start <= t && t < p.end && t < s.makespan).map((p) => (
          <Block key={'r' + p.block} x={x + col(p.unit, p.slot) * cw} y={y + 5} w={cw - 4} n={p.block} on />
        ))}
      </g>
      <line x1={x - 14} x2={x - 14} y1={y + 46} y2={y + 46 + s.makespan * rowH} stroke="var(--ink-3)" />
      <text x={x - 18} y={y + 58} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi="thời gian">time</Translated></text>
      <g data-part="history">
        {s.placements.filter((p) => p.start <= t).map((p) => (
          <Block key={p.block} x={x + col(p.unit, p.slot) * cw} y={y + 46 + p.start * rowH} w={cw - 4} n={p.block} on={t < p.end} />
        ))}
      </g>
    </g>
  );
}

function Frame({ children, headline, sub }: { children: ReactNode; headline?: ReactNode; sub?: ReactNode }) {
  const narrow = useNarrow();
  // On a narrow screen the headline and the line under it are shown as text below the drawing (sceneText.tsx).
  return (
    <>
      <svg viewBox={`0 0 ${W} ${narrow ? H - FOOT : H}`} role="img" aria-label={tr('Animation of a grid of blocks draining through a device', 'Hoạt hình một lưới các khối lần lượt chạy hết qua một thiết bị')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
        {children}
        {!narrow && headline ? <text x={24} y={H - 30} fontSize="13.5" fontWeight="600">{headline}</text> : null}
        {!narrow && sub ? <text x={24} y={H - 12} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>{sub}</text> : null}
      </svg>
      {narrow ? <SceneReadout headline={headline} notes={sub ? [sub] : []} /> : null}
    </>
  );
}

/** 132 SMs of 2 slots as a field of cells; a slot is filled while a block runs in it. */
function Field({ x, y, s, t }: { x: number; y: number; s: Schedule; t: number }) {
  const cols = 22;
  const cw = 15;
  const ch = 11;
  const busy = new Set(s.placements.filter((p) => p.start <= t && t < p.end).map((p) => p.unit * 8 + p.slot));
  const cells: ReactNode[] = [];
  for (let u = 0; u < s.spec.units; u++)
    for (let k = 0; k < s.spec.slotsPerUnit; k++) {
      const on = busy.has(u * 8 + k);
      cells.push(
        <rect key={u * 8 + k} x={x + (u % cols) * cw} y={y + Math.floor(u / cols) * (s.spec.slotsPerUnit * ch + 5) + k * ch}
          width={cw - 2} height={ch - 1.5} rx="1.5" fill={on ? 'var(--ink-2)' : 'var(--cell)'} stroke={on ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth="0.7" />,
      );
    }
  return <g data-part="field" data-busy={busy.size} data-slots={cells.length}>{cells}</g>;
}
const busyAt = (s: Schedule, t: number) => s.placements.filter((p) => p.start <= t && t < p.end).length;

/** Blocks in progress over time, as columns under a capacity line. */
function Profile({ x, y, w, h, s, upto = Infinity, label }: { x: number; y: number; w: number; h: number; s: Schedule; upto?: number; label: string }) {
  const span = s.makespan;
  const sx = (t: number) => x + (t / span) * w;
  return (
    <g>
      <text x={x} y={y - 8} fontSize="11.5" fontWeight="600">{label}</text>
      {profile(s).filter((g) => g.from < upto).map((g) => (
        <rect key={g.from} x={sx(g.from)} y={y + h - (g.resident / s.capacity) * h} width={Math.max(0.5, sx(Math.min(g.to, upto)) - sx(g.from) - 0.6)}
          height={(g.resident / s.capacity) * h} fill="var(--ink-2)" />
      ))}
      <line x1={x} x2={x + w} y1={y} y2={y} stroke="var(--ink)" strokeDasharray="4 3" />
      <line x1={x} x2={x + w} y1={y + h} y2={y + h} stroke="var(--ink-3)" />
      <text x={x + w + 6} y={y + 4} fontSize="10" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>cả {s.capacity} ô</>}>all {s.capacity} slots</Translated></text>
      <text x={x + w} y={y + h + 13} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi="thời gian">time</Translated></text>
    </g>
  );
}

const sizes = waveSizes(TAIL_EXAMPLE.blocks, CAP);
const theTail = tail(TAIL_EXAMPLE.blocks, CAP);

const beats: Beat[] = [
  {
    clip: 'waves-1',
    get label() { return tr('The list', 'Danh sách'); },
    draw: (p) => (
      <Frame headline={tr(`${SMALL.spec.blocks} blocks have not started`, `${SMALL.spec.blocks} khối chưa bắt đầu`)} sub={tr('each needs a free slot on an SM for all of its threads', 'mỗi khối cần một ô trống trên một SM cho mọi luồng của nó')}>
        <Pending x={40} y={48} s={SMALL} t={-1} shown={ramp(p, -0.15, 0.45)} />
        <g opacity={ramp(p, 0.6, 0.8)}>
          <text x={300} y={40} fontSize="11.5" fontWeight="600"><Translated vi="một thiết bị">a device</Translated></text>
          <rect x={295} y={48} width={300} height={60} rx="4" fill="var(--bench)" stroke="var(--ink-3)" />
          <text x={445} y={84} fontSize="13" textAnchor="middle"><Translated vi="bao nhiêu ô?">how many slots?</Translated></text>
        </g>
      </Frame>
    ),
  },
  {
    clip: 'waves-2',
    get label() { return tr('Two slots', 'Hai ô'); },
    draw: (p) => {
      const t = ramp(p, 0.1, 0.9) * SMALL.makespan;
      const wave = Math.min(SMALL.makespan, Math.floor(t) + 1);
      const done = t >= SMALL.makespan;
      return (
        <Frame
          headline={done ? tr(`${SMALL.spec.blocks} blocks, ${SMALL.capacity} at a time: ${waves(SMALL.spec.blocks, SMALL.capacity)} waves`, `${SMALL.spec.blocks} khối, mỗi lần ${SMALL.capacity}: ${waves(SMALL.spec.blocks, SMALL.capacity)} đợt`) : tr(`wave ${wave} of ${waves(SMALL.spec.blocks, SMALL.capacity)}`, `đợt ${wave} trên ${waves(SMALL.spec.blocks, SMALL.capacity)}`)}
          sub={tr('a block starts when a slot is free', 'một khối bắt đầu khi có ô trống')}
        >
          <Pending x={40} y={48} s={SMALL} t={p < 0.1 ? -1 : t} />
          <DeviceRun x={300} y={48} s={SMALL} t={p < 0.1 ? -1 : t} name={tr(`a small device: ${SMALL.capacity} slots`, `một thiết bị nhỏ: ${SMALL.capacity} ô`)} />
        </Frame>
      );
    },
  },
  {
    clip: 'waves-3',
    get label() { return tr('Four slots', 'Bốn ô'); },
    draw: (p) => {
      const t = ramp(p, 0.12, 0.6) * LARGE.makespan;
      const done = t >= LARGE.makespan;
      return (
        <Frame
          headline={done ? tr(`the same grid: ${waves(8, SMALL.capacity)} waves there, ${waves(8, LARGE.capacity)} waves here`, `cùng một lưới: ${waves(8, SMALL.capacity)} đợt ở kia, ${waves(8, LARGE.capacity)} đợt ở đây`) : tr(`wave ${Math.min(LARGE.makespan, Math.floor(t) + 1)} of ${waves(8, LARGE.capacity)}`, `đợt ${Math.min(LARGE.makespan, Math.floor(t) + 1)} trên ${waves(8, LARGE.capacity)}`)}
          sub={done ? tr('the kernel and the launch are unchanged', 'kernel và lần khởi chạy không đổi') : tr('the same list, 4 blocks at a time', 'cùng danh sách đó, mỗi lần 4 khối')}
        >
          <DeviceRun x={60} y={48} s={SMALL} t={SMALL.makespan} name={tr(`${SMALL.capacity} slots`, `${SMALL.capacity} ô`)} dim />
          <Pending x={250} y={48} s={LARGE} t={p < 0.12 ? -1 : t} />
          <DeviceRun x={420} y={48} s={LARGE} t={p < 0.12 ? -1 : t} name={tr(`a larger device: ${LARGE.capacity} slots`, `một thiết bị lớn hơn: ${LARGE.capacity} ô`)} />
        </Frame>
      );
    },
  },
  {
    clip: 'waves-4',
    get label() { return tr('Any order', 'Thứ tự bất kỳ'); },
    draw: (p) => {
      const shuffling = p >= 0.5;
      const t = ramp(p, 0.58, 0.9) * SHUFFLED.makespan;
      const finished = new Set(shuffling ? completionOrder(SHUFFLED).filter((b) => SHUFFLED.placements[b].end <= t) : []);
      return (
        <Frame
          headline={shuffling ? tr(`handed out as ${SHUFFLE.join(' ')}: ${finished.size} of ${SHUFFLE.length} outputs written`, `phát theo thứ tự ${SHUFFLE.join(' ')}: đã ghi ${finished.size} trên ${SHUFFLE.length} phần đầu ra`) : tr('a barrier holds the threads of one block', 'một rào chắn (barrier) giữ các luồng của một khối')}
          sub={shuffling ? (finished.size === SHUFFLE.length ? tr('the same outputs as before, whatever the order', 'cùng đầu ra như trước, dù thứ tự thế nào') : tr('each block writes its own part of the output', 'mỗi khối ghi phần đầu ra của riêng nó')) : tr('it stops at the edge of the block: nothing ties one block to another', 'nó dừng ở biên của khối: không có gì ràng buộc khối này với khối khác')}
        >
          {!shuffling ? (
            Array.from({ length: FIG45.blocks }, (_, b) => (
              <g key={b} opacity={ramp(p, 0.02 + b * 0.02, 0.1 + b * 0.02)}>
                <Block x={40 + (b % 4) * 164} y={30 + Math.floor(b / 4) * 128} w={140} h={104} n={b} inside />
              </g>
            ))
          ) : (
            <>
              <Pending x={40} y={48} s={SHUFFLED} t={p < 0.58 ? -1 : t} />
              <DeviceRun x={200} y={48} s={SHUFFLED} t={p < 0.58 ? -1 : t} name={tr(`${SHUFFLED.capacity} slots, the list shuffled`, `${SHUFFLED.capacity} ô, danh sách bị xáo trộn`)} />
              <text x={440} y={40} fontSize="11.5" fontWeight="600"><Translated vi="đầu ra, mỗi khối một phần">output, one part per block</Translated></text>
              {Array.from({ length: FIG45.blocks }, (_, b) => (
                <g key={b} data-part="output" data-written={finished.has(b) ? '1' : '0'}>
                  <rect x={440 + (b % 4) * 62} y={48 + Math.floor(b / 4) * 40} width={56} height={32} rx="3"
                    fill={finished.has(b) ? 'var(--ok-wash)' : 'var(--cell)'} stroke={finished.has(b) ? 'var(--ok)' : 'var(--ink-3)'} />
                  <text x={468 + (b % 4) * 62} y={68 + Math.floor(b / 4) * 40} fontSize="10.5" textAnchor="middle"
                    style={{ fill: finished.has(b) ? 'var(--ink)' : 'var(--ink-3)' }}>{finished.has(b) ? tr(`part ${b}`, `phần ${b}`) : '…'}</text>
                </g>
              ))}
            </>
          )}
        </Frame>
      );
    },
  },
  {
    clip: 'waves-5',
    get label() { return tr('The tail', 'Phần đuôi'); },
    draw: (p) => {
      const t = Math.min(TAILED.makespan - 0.001, ramp(p, 0.12, 0.8) * TAILED.makespan);
      const wave = Math.floor(t);
      const busy = busyAt(TAILED, t);
      return (
        <Frame
          headline={tr(`wave ${wave + 1} of ${sizes.length}: ${busy} of ${CAP} slots busy`, `đợt ${wave + 1} trên ${sizes.length}: ${busy} trên ${CAP} ô đang bận`)}
          sub={p > 0.8 ? tr(`${TAIL_EXAMPLE.blocks} blocks at ${CAP} at once is ${waves(TAIL_EXAMPLE.blocks, CAP)} waves: the last leaves ${theTail.idle} slots idle`, `${TAIL_EXAMPLE.blocks} khối, mỗi lần ${CAP}, là ${waves(TAIL_EXAMPLE.blocks, CAP)} đợt: đợt cuối để ${theTail.idle} ô ngồi không`) : tr(`${TAIL_EXAMPLE.blocks} blocks, ${CAP} at once`, `${TAIL_EXAMPLE.blocks} khối, mỗi lần ${CAP}`)}
        >
          <text x={24} y={30} fontSize="11.5" fontWeight="600"><Translated vi={<>{TAIL_EXAMPLE.units} SM, mỗi SM {TAIL_EXAMPLE.slotsPerUnit} ô</>}>{TAIL_EXAMPLE.units} SMs with {TAIL_EXAMPLE.slotsPerUnit} slots each</Translated></text>
          <Field x={24} y={42} s={TAILED} t={t} />
          <text x={400} y={30} fontSize="11.5" fontWeight="600"><Translated vi="số khối trong mỗi đợt">blocks in each wave</Translated></text>
          {sizes.map((n, i) => (
            <g key={i} opacity={i <= wave ? 1 : 0.25}>
              <text x={400} y={62 + i * 44} fontSize="11"><Translated vi={<>đợt {i + 1}</>}>wave {i + 1}</Translated></text>
              <rect x={452} y={48 + i * 44} width={200} height={20} fill="none" stroke="var(--ink-3)" strokeDasharray="3 3" />
              <rect x={452} y={48 + i * 44} width={200 * (n / CAP)} height={20} fill="var(--ink-2)" />
              <text x={660} y={62 + i * 44} fontSize="11">{n}</text>
            </g>
          ))}
        </Frame>
      );
    },
  },
  {
    clip: 'waves-6',
    get label() { return tr('Two ways out', 'Hai lối thoát'); },
    draw: (p) => {
      if (p < 0.42) {
        const t = Math.min(EVENED.makespan - 0.001, ramp(p, 0.08, 0.38) * EVENED.makespan);
        return (
          <Frame headline={tr(`${EVEN_BLOCKS} blocks = ${waves(EVEN_BLOCKS, CAP)} × ${CAP}: wave ${Math.floor(t) + 1}, ${busyAt(EVENED, t)} of ${CAP} slots busy`, `${EVEN_BLOCKS} khối = ${waves(EVEN_BLOCKS, CAP)} × ${CAP}: đợt ${Math.floor(t) + 1}, ${busyAt(EVENED, t)} trên ${CAP} ô đang bận`)} sub={tr('a multiple of what runs at once leaves no tail, when blocks take equal time', 'một bội số của lượng chạy cùng lúc không để lại đuôi, khi các khối tốn thời gian như nhau')}>
            <text x={24} y={30} fontSize="11.5" fontWeight="600"><Translated vi={<>{TAIL_EXAMPLE.units} SM, mỗi SM {TAIL_EXAMPLE.slotsPerUnit} ô</>}>{TAIL_EXAMPLE.units} SMs with {TAIL_EXAMPLE.slotsPerUnit} slots each</Translated></text>
            <Field x={24} y={42} s={EVENED} t={t} />
            <Profile x={400} y={52} w={220} h={120} s={EVENED} upto={t + 0.001 > EVENED.makespan - 0.01 ? Infinity : Math.floor(t) + 1} label={tr('blocks in progress', 'số khối đang chạy')} />
          </Frame>
        );
      }
      const t = ramp(p, 0.46, 0.8) * UNEVEN.makespan;
      const bars = ramp(p, 0.72, 0.9);
      return (
        <Frame
          headline={bars > 0 ? tr(`uneven blocks: ${pct(UNEVEN_ONE)}% of the capacity used in 1 wave, ${pct(UNEVEN_TEN)}% in 10`, `khối không đều: dùng ${pct(UNEVEN_ONE)}% sức chứa với 1 đợt, ${pct(UNEVEN_TEN)}% với 10 đợt`) : tr(`uneven block times: ${busyAt(UNEVEN, t)} of ${CAP} slots busy`, `thời gian khối không đều: ${busyAt(UNEVEN, t)} trên ${CAP} ô đang bận`)}
          sub={tr('slots free up one at a time, and each takes the next block at once', 'các ô lần lượt rảnh từng ô một, và mỗi ô nhận ngay khối kế tiếp')}
        >
          <text x={24} y={30} fontSize="11.5" fontWeight="600"><Translated vi={<>cũng {EVEN_BLOCKS} khối đó, giờ mỗi khối tốn từ 1 đến 4 đơn vị thời gian</>}>the same {EVEN_BLOCKS} blocks, now taking 1 to 4 time units each</Translated></text>
          <Field x={24} y={42} s={UNEVEN} t={t} />
          <Profile x={400} y={52} w={220} h={90} s={UNEVEN} upto={t} label={tr('blocks in progress', 'số khối đang chạy')} />
          <g opacity={bars}>
            <text x={400} y={186} fontSize="11"><Translated vi="1 đợt khối không đều">1 wave of uneven blocks</Translated></text>
            <rect x={400} y={192} width={220 * UNEVEN_ONE * bars} height={12} fill="var(--ink-2)" />
            <text x={404 + 220 * UNEVEN_ONE * bars} y={202} fontSize="11">{pct(UNEVEN_ONE * bars)}%</text>
            <text x={400} y={226} fontSize="11"><Translated vi="10 đợt khối không đều">10 waves of uneven blocks</Translated></text>
            <rect x={400} y={232} width={220 * UNEVEN_TEN * bars} height={12} fill="var(--ink-2)" />
            <text x={404 + 220 * UNEVEN_TEN * bars} y={242} fontSize="11">{pct(UNEVEN_TEN * bars)}%</text>
          </g>
        </Frame>
      );
    },
  },
];

export function WavesScene() {
  return <Scene chapter="ch04" title={tr('One grid, two devices', 'Một lưới, hai thiết bị')} beats={beats} />;
}
