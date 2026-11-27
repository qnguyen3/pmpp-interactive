import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { SceneReadout, useNarrow } from './sceneText';
import {
  PRESETS, busyFraction, runs, simulate, simulateSwitching, unitShares, utilisation, warpsNeeded, type Cell, type Timeline,
  type UnitCell,
} from '../../sim/latency';
import { BOOK, occupancy } from '../../sim/occupancy';
import { Translated, tr } from '../../lib/i18n';

// "Hiding a wait": section 4.6 in time. Six beats, one narration clip each
// (content/ch04/narration/latency-N.txt). Every cell, count and percentage drawn
// comes from sim/latency.ts, and the register file of beat 5 from sim/occupancy.ts.
const W = 720;
const H = 348;
/** The strip at the bottom that holds the headline and the line under it. */
const FOOT = 44;
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const pct = (v: number) => Math.round(v * 1000) / 10;

// The scene's own small case: 2 cycles of work, then a 6-cycle wait. Illustrative
// numbers, chosen so that the warps needed (4) fit on the screen.
export const SCENE = { work: 2, latency: 6, slots: 1 } as const;
export const CYCLES = 24;
export const NEEDED = warpsNeeded(SCENE.work, SCENE.latency);
const line = (warps: number) => simulate({ warps, ...SCENE }, CYCLES);
const ONE = line(1);
const THREE = line(3);
const ENOUGH = line(NEEDED);
/** The price of a switch on the CPU side of beat 4, in cycles. Illustrative: the book gives no figure. */
export const SWITCH_COST = 2;
const FREE = simulateSwitching({ warps: NEEDED, ...SCENE }, CYCLES, 0);
const COSTLY = simulateSwitching({ warps: NEEDED, ...SCENE }, CYCLES, SWITCH_COST);

// Beat 5: the book's example register file, blocks of two warps, at two register counts.
const REG_BLOCK = 2 * (BOOK.width as number);
export const REGS_A = 32;
export const REGS_B = 64;
const FITS_A = occupancy(BOOK, { threads: REG_BLOCK, regs: REGS_A, shared: 0 });
const FITS_B = occupancy(BOOK, { threads: REG_BLOCK, regs: REGS_B, shared: 0 });
const UNITS = FITS_A.warpSlots as number; // the register file drawn as this many equal cells
const REGS_PER_UNIT = (BOOK.regs as number) / UNITS;

// Beat 6: the explorer's two presets, at these numbers of resident warps.
export const LADDER = [1, 2, 4, 8, 16];
const SIDE_CYCLES = 64;
const side = (preset: { work: number; latency: number }) =>
  LADDER.map((warps) => ({
    warps,
    tl: simulate({ warps, ...preset, slots: 1 }, SIDE_CYCLES),
    use: utilisation({ warps, ...preset, slots: 1 }),
  }));
const HEAVY = side(PRESETS.computeHeavy);
const LOADY = side(PRESETS.loadHeavy);

const FILL: Record<Cell, string> = { exec: 'var(--ink)', wait: 'var(--global-wash)', ready: 'var(--rule)' };
const STROKE: Record<Cell, string> = { exec: 'var(--ink)', wait: 'var(--global)', ready: 'var(--ink-3)' };

/** `data` becomes data-* attributes on the drawing, so that the browser check can read what was drawn. */
function Frame({ children, headline, sub, data = {} }: { children: ReactNode; headline?: ReactNode; sub?: ReactNode; data?: Record<string, string | number> }) {
  const narrow = useNarrow();
  const attrs = Object.fromEntries(Object.entries(data).map(([k, v]) => ['data-' + k, String(v)]));
  // On a narrow screen the headline and the line under it are shown as text below the drawing (sceneText.tsx).
  return (
    <>
      <svg viewBox={`0 0 ${W} ${narrow ? H - FOOT : H}`} role="img" aria-label={tr('Animation of warps taking turns on one execution unit', 'Hoạt hình các warp lần lượt dùng một đơn vị thực thi')} style={{ width: '100%', maxWidth: W, display: 'block' }} {...attrs}>
        {children}
        {!narrow && headline ? <text x={24} y={H - 30} fontSize="13.5" fontWeight="600">{headline}</text> : null}
        {!narrow && sub ? <text x={24} y={H - 12} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>{sub}</text> : null}
      </svg>
      {narrow ? <SceneReadout headline={headline} notes={sub ? [sub] : []} /> : null}
    </>
  );
}

function Legend({ x, y, ready = true }: { x: number; y: number; ready?: boolean }) {
  const items: [Cell, string][] = [['exec', tr('executing', 'đang thực thi')], ['wait', tr('waiting on memory', 'đang chờ bộ nhớ')]];
  if (ready) items.push(['ready', tr('ready, not chosen', 'sẵn sàng, chưa được chọn')]);
  return (
    <g>
      {items.map(([c, label], i) => (
        <g key={c}>
          <rect x={x + i * 150} y={y - 9} width={14} height={10} rx="1.5" fill={FILL[c]} stroke={STROKE[c]} strokeWidth="0.8" />
          <text x={x + i * 150 + 20} y={y} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>{label}</text>
        </g>
      ))}
    </g>
  );
}

/** One row per warp and the execution unit below them, drawn up to cycle `upto`. */
function Lanes({ x, y, tl, upto, cw, rh, labels = true, unit = true }: {
  x: number; y: number; tl: Timeline; upto: number; cw: number; rh: number; labels?: boolean; unit?: boolean;
}) {
  const shown = Math.max(0, Math.min(tl.cycles, Math.floor(upto)));
  const R = tl.rows.length;
  const uy = y + R * rh + 12;
  return (
    <g>
      {tl.rows.map((row, w) => (
        <g key={w}>
          {labels ? <text x={x - 8} y={y + w * rh + rh / 2 + 3} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>warp {w}</>}>warp {w}</Translated></text> : null}
          <rect x={x} y={y + w * rh} width={tl.cycles * cw} height={rh - 3} fill="none" stroke="var(--rule-soft)" />
          {runs(row).filter((r) => r.start < shown).map((r) => {
            const len = Math.min(r.length, shown - r.start);
            return (
              <rect key={r.start} x={x + r.start * cw + 0.5} y={y + w * rh} width={Math.max(0.5, len * cw - 1)} height={rh - 3}
                rx={Math.min(2, rh / 4)} fill={FILL[r.state]} stroke={STROKE[r.state]} strokeWidth={rh > 12 ? 0.9 : 0.4} />
            );
          })}
        </g>
      ))}
      {unit ? (
        <g>
          {labels ? <text x={x - 8} y={uy + rh / 2 + 2} fontSize="10.5" textAnchor="end" fontWeight="600"><Translated vi="đơn vị">unit</Translated></text> : null}
          {tl.chosen.slice(0, shown).map((pick, t) => (
            <rect key={t} x={x + t * cw + 0.5} y={uy} width={Math.max(0.5, cw - 1)} height={rh - 3} rx={Math.min(2, rh / 4)}
              fill={pick.length ? 'var(--ink)' : 'none'} stroke={pick.length ? 'var(--ink)' : 'var(--ink-3)'}
              strokeDasharray={pick.length ? undefined : '2 2'} strokeWidth="0.9" />
          ))}
          {rh > 14 && cw > 14
            ? tl.chosen.slice(0, shown).map((pick, t) =>
                pick.length ? (
                  <text key={'n' + t} x={x + t * cw + cw / 2} y={uy + (rh - 3) / 2 + 3.5} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--page)' }}>{pick[0]}</text>
                ) : null,
              )
            : null}
        </g>
      ) : null}
      <line x1={x} x2={x + tl.cycles * cw} y1={uy + (unit ? rh : -12) + 4} y2={uy + (unit ? rh : -12) + 4} stroke="var(--ink-3)" />
      <text x={x + tl.cycles * cw} y={uy + (unit ? rh : -12) + 16} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi="chu kỳ">cycles</Translated></text>
    </g>
  );
}

/** Cycles drawn so far and how many of them the unit issued in. */
const busySoFar = (tl: Timeline, upto: number) => {
  const shown = Math.max(0, Math.min(tl.cycles, Math.floor(upto)));
  return { busy: Math.round(busyFraction(tl, 0, shown) * shown), shown };
};
/** The headline of beats 1 to 3: what the unit has done so far. */
const busyLine = (busy: number, shown: number, prefix = tr('execution unit busy for', 'đơn vị thực thi bận')) =>
  shown === 0
    ? tr('cycle 0: nothing issued yet', 'chu kỳ 0: chưa phát lệnh nào')
    : tr(`${prefix} ${busy} of ${shown} cycles: ${pct(busy / shown)}%`, `${prefix} ${busy} trên ${shown} chu kỳ: ${pct(busy / shown)}%`);
/** The first wait of a warp: where it starts and where its data is back. */
const firstWait = (tl: Timeline, w: number) => runs(tl.rows[w]).find((r) => r.state === 'wait')!;

const LX = 96; // left edge of the wide timelines
const CW = 24;

/** A warp's resident state: its program counter and its registers, drawn in the register colour. */
function WarpState({ x, y, n, on }: { x: number; y: number; n: number; on: boolean }) {
  return (
    <g>
      <rect x={x} y={y} width={132} height={30} rx="3" fill={on ? 'var(--reg)' : 'var(--reg-wash)'} stroke="var(--reg)" />
      <text x={x + 8} y={y + 19} fontSize="10.5" style={{ fill: on ? 'var(--page)' : 'var(--ink)' }}><Translated vi={<>warp {n}: PC, thanh ghi</>}>warp {n}: PC, registers</Translated></text>
    </g>
  );
}

function UnitStrip({ x, y, unit, upto, cw }: { x: number; y: number; unit: UnitCell[]; upto: number; cw: number }) {
  const shown = Math.max(0, Math.min(unit.length, Math.floor(upto)));
  return (
    <g>
      {unit.slice(0, shown).map((u, t) => (
        <g key={t}>
          <rect x={x + t * cw + 0.5} y={y} width={cw - 1} height={20} rx="2"
            fill={u.kind === 'exec' ? 'var(--ink)' : u.kind === 'switch' ? 'var(--rule)' : 'none'}
            stroke={u.kind === 'exec' ? 'var(--ink)' : 'var(--ink-3)'} strokeDasharray={u.kind === 'idle' ? '2 2' : undefined} strokeWidth="0.9" />
          {u.kind === 'exec' ? (
            <text x={x + t * cw + cw / 2} y={y + 14} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--page)' }}>{u.warp}</text>
          ) : null}
        </g>
      ))}
    </g>
  );
}

/** Two kernels of beat 6: one small timeline and its utilisation. */
function Side({ x, title, sub, rung }: { x: number; title: string; sub: string; rung: { warps: number; tl: Timeline; use: number } }) {
  const cw = 4.8;
  const rh = 9;
  return (
    <g>
      <text x={x} y={26} fontSize="12.5" fontWeight="600">{title}</text>
      <text x={x} y={42} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>{sub}</text>
      <Lanes x={x} y={56} tl={rung.tl} upto={SIDE_CYCLES} cw={cw} rh={rh} labels={false} />
      <text x={x} y={262} fontSize="12" fontWeight="600">
        <Translated vi={<>{rung.warps} warp: đơn vị bận {pct(rung.use)}%</>}>
          {rung.warps} {rung.warps === 1 ? 'warp' : 'warps'}: unit busy {pct(rung.use)}%
        </Translated>
      </text>
      <rect x={x} y={270} width={SIDE_CYCLES * cw} height={9} fill="none" stroke="var(--ink-3)" />
      <rect x={x} y={270} width={SIDE_CYCLES * cw * rung.use} height={9} fill="var(--ink)" />
    </g>
  );
}

const beats: Beat[] = [
  {
    clip: 'latency-1',
    get label() { return tr('One warp', 'Một warp'); },
    draw: (p) => {
      const upto = ramp(p, 0.08, 0.9) * CYCLES;
      const { busy, shown } = busySoFar(ONE, upto);
      const wait = firstWait(ONE, 0);
      return (
        <Frame
          headline={busyLine(busy, shown)}
          sub={tr(`${SCENE.work} cycles of work, then a ${SCENE.latency}-cycle wait on global memory`, `${SCENE.work} chu kỳ làm việc, rồi ${SCENE.latency} chu kỳ chờ bộ nhớ toàn cục (global memory)`)}
          data={{ beat: 1, warps: 1, shown, busy }}
        >
          <Legend x={LX} y={34} ready={false} />
          <Lanes x={LX} y={84} tl={ONE} upto={upto} cw={CW} rh={30} />
          {upto > wait.start ? <text x={LX + wait.start * CW} y={76} fontSize="10.5" style={{ fill: 'var(--global)' }}><Translated vi="đã gửi lệnh nạp">load sent</Translated></text> : null}
          {upto > wait.start + wait.length ? (
            <text x={LX + (wait.start + wait.length) * CW} y={76} fontSize="10.5" style={{ fill: 'var(--global)' }}><Translated vi="dữ liệu về">data back</Translated></text>
          ) : null}
        </Frame>
      );
    },
  },
  {
    clip: 'latency-2',
    get label() { return tr('Three warps', 'Ba warp'); },
    draw: (p) => {
      const upto = ramp(p, 0.08, 0.9) * CYCLES;
      const { busy, shown } = busySoFar(THREE, upto);
      return (
        <Frame headline={busyLine(busy, shown)} sub={tr('when one warp waits, the scheduler issues from another that is ready', 'khi một warp chờ, bộ lập lịch phát lệnh từ một warp khác đang sẵn sàng')} data={{ beat: 2, warps: 3, shown, busy }}>
          <Legend x={LX} y={34} />
          <Lanes x={LX} y={62} tl={THREE} upto={upto} cw={CW} rh={30} />
        </Frame>
      );
    },
  },
  {
    clip: 'latency-3',
    get label() { return tr('Enough warps', 'Đủ warp'); },
    draw: (p) => {
      const upto = ramp(p, 0.06, 0.8) * CYCLES;
      const { busy, shown } = busySoFar(ENOUGH, upto);
      const last = firstWait(ENOUGH, NEEDED - 1);
      const first = firstWait(ENOUGH, 0);
      const meet = first.start + first.length; // the cycle warp 0's data is back
      return (
        <Frame
          headline={busyLine(busy, shown, tr(`${NEEDED} warps: busy for`, `${NEEDED} warp: bận`))}
          sub={p > 0.55 ? tr('latency tolerance: no warp got faster; the wait is hidden behind other warps', 'khả năng chịu đựng độ trễ (latency tolerance): không warp nào nhanh hơn; thời gian chờ được che sau các warp khác') : tr(`as warp ${NEEDED - 1} sends its request, warp 0's data arrives`, `đúng lúc warp ${NEEDED - 1} gửi yêu cầu, dữ liệu của warp 0 về tới`)}
          data={{ beat: 3, warps: NEEDED, shown, busy, meet }}
        >
          <Legend x={LX} y={34} />
          <Lanes x={LX} y={56} tl={ENOUGH} upto={upto} cw={CW} rh={30} />
          {upto >= meet && last.start === meet ? (
            <g>
              <line x1={LX + meet * CW} x2={LX + meet * CW} y1={50} y2={56 + NEEDED * 30 + 42} stroke="var(--ink)" strokeDasharray="4 3" />
              <text x={LX + meet * CW + 5} y={50} fontSize="10.5"><Translated vi={<>chu kỳ {meet}</>}>cycle {meet}</Translated></text>
            </g>
          ) : null}
        </Frame>
      );
    },
  },
  {
    clip: 'latency-4',
    get label() { return tr('A free switch', 'Chuyển đổi miễn phí'); },
    draw: (p) => {
      const upto = ramp(p, 0.1, 0.92) * CYCLES;
      const t = Math.max(0, Math.min(CYCLES - 1, Math.floor(upto)));
      const gpu = FREE[t];
      const cpu = COSTLY[t];
      const shares = unitShares(COSTLY, 0, CYCLES);
      const cw = 12.6;
      // A switch of SWITCH_COST cycles: the first half saves the old state, the second loads the new one.
      let into = 0; // how many cycles of the current switch have been spent, this one included
      while (t - into >= 0 && COSTLY[t - into].kind === 'switch') into += 1;
      const loading = into > SWITCH_COST / 2;
      const execSoFar = (unit: UnitCell[]) => unit.slice(0, Math.floor(upto)).filter((u) => u.kind === 'exec').length;
      return (
        <Frame
          headline={tr(`the SM changed warps for free; the other design spent ${pct(shares.switch)}% of its cycles saving and loading`, `SM đổi warp không tốn gì; thiết kế kia mất ${pct(shares.switch)}% số chu kỳ để lưu và nạp`)}
          sub={tr(`the same ${NEEDED} warps and the same waits on both sides; a ${SWITCH_COST}-cycle switch is an illustrative price`, `cùng ${NEEDED} warp và cùng những lần chờ ở cả hai bên; mức giá ${SWITCH_COST} chu kỳ cho mỗi lần chuyển chỉ để minh họa`)}
          data={{ beat: 4, shown: Math.floor(upto), 'gpu-exec': execSoFar(FREE), 'cpu-exec': execSoFar(COSTLY), 'switch-share': pct(shares.switch) }}
        >
          <text x={30} y={26} fontSize="12.5" fontWeight="600"><Translated vi="một SM: trạng thái của mọi warp nằm yên trên chip">an SM: every warp's state stays on the chip</Translated></text>
          {Array.from({ length: NEEDED }, (_, w) => <WarpState key={w} x={30} y={38 + w * 36} n={w} on={gpu.kind === 'exec' && gpu.warp === w} />)}
          <text x={176} y={38 + (gpu.warp ?? 0) * 36 + 19} fontSize="11"><Translated vi="◀ đang phát lệnh từ đây">◀ issuing from here</Translated></text>
          <text x={30} y={204} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="đơn vị thực thi">execution unit</Translated></text>
          <UnitStrip x={30} y={210} unit={FREE} upto={upto} cw={cw} />

          <text x={384} y={26} fontSize="12.5" fontWeight="600"><Translated vi="một lõi CPU: mỗi lúc chỉ trạng thái của một luồng">a CPU core: one thread's state at a time</Translated></text>
          <rect x={384} y={38} width={156} height={30} rx="3" fill={cpu.kind === 'exec' ? 'var(--reg)' : 'var(--reg-wash)'} stroke="var(--reg)" />
          <text x={392} y={57} fontSize="10.5" style={{ fill: cpu.kind === 'exec' ? 'var(--page)' : 'var(--ink)' }}>
            {cpu.kind === 'switch' ? tr('PC, registers: in transit', 'PC, thanh ghi: đang di chuyển') : tr(`thread ${cpu.warp ?? 0}: PC, registers`, `luồng ${cpu.warp ?? 0}: PC, thanh ghi`)}
          </text>
          <rect x={384} y={96} width={300} height={84} rx="3" fill="var(--global-wash)" stroke="var(--global)" />
          <text x={392} y={112} fontSize="10.5" style={{ fill: 'var(--global)' }}><Translated vi="bộ nhớ: trạng thái của các luồng không đang chạy">memory: the state of the threads that are not running</Translated></text>
          {Array.from({ length: NEEDED }, (_, w) => (
            <g key={w} opacity={cpu.warp === w && cpu.kind === 'exec' ? 0.25 : 1}>
              <rect x={392 + w * 72} y={124} width={64} height={44} rx="2" fill="var(--cell)" stroke="var(--ink-3)" />
              <text x={392 + w * 72 + 32} y={143} fontSize="10" textAnchor="middle"><Translated vi={<>luồng {w}</>}>thread {w}</Translated></text>
              <text x={392 + w * 72 + 32} y={158} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}><Translated vi="trạng thái đã lưu">saved state</Translated></text>
            </g>
          ))}
          {cpu.kind === 'switch' ? (
            <text x={552} y={57} fontSize="11" fontWeight="600">{loading ? tr('load ↑ the next thread', 'nạp ↑ luồng kế tiếp') : tr('save ↓ the old thread', 'lưu ↓ luồng cũ')}</text>
          ) : null}
          <text x={384} y={204} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="đơn vị thực thi (xám: đang lưu và nạp)">execution unit (grey: saving and loading)</Translated></text>
          <UnitStrip x={384} y={210} unit={COSTLY} upto={upto} cw={cw} />
        </Frame>
      );
    },
  },
  {
    clip: 'latency-5',
    get label() { return tr('The price', 'Cái giá'); },
    draw: (p) => {
      const second = p >= 0.55;
      const fits = second ? FITS_B : FITS_A;
      const regs = second ? REGS_B : REGS_A;
      const grow = second ? ramp(p, 0.6, 0.85) : ramp(p, 0.08, 0.45);
      const warps = Math.round(fits.residentWarps * grow);
      const perWarp = regs * fits.width; // registers one warp holds
      const span = perWarp / REGS_PER_UNIT; // cells of the drawing one warp takes
      const cols = 16;
      const cell = 38;
      const X = 56;
      const Y = 70;
      const cells: ReactNode[] = [];
      for (let u = 0; u < UNITS; u++)
        cells.push(<rect key={'u' + u} x={X + (u % cols) * cell} y={Y + Math.floor(u / cols) * cell} width={cell - 3} height={cell - 3} rx="2" fill="var(--cell)" stroke="var(--rule)" />);
      for (let w = 0; w < warps; w++) {
        const u = w * span;
        cells.push(
          <g key={'w' + w}>
            <rect x={X + (u % cols) * cell} y={Y + Math.floor(u / cols) * cell} width={span * cell - 3} height={cell - 3} rx="2" fill="var(--reg-wash)" stroke="var(--reg)" />
            <text x={X + (u % cols) * cell + (span * cell - 3) / 2} y={Y + Math.floor(u / cols) * cell + cell / 2 + 2} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--reg)' }}>{w}</text>
          </g>,
        );
      }
      return (
        <Frame
          headline={tr(`${regs} registers per thread: ${warps} of ${fits.warpSlots} warp slots filled, ${(warps * perWarp).toLocaleString('en-US')} of ${(BOOK.regs as number).toLocaleString('en-US')} registers`, `${regs} thanh ghi mỗi luồng: đã lấp ${warps} trên ${fits.warpSlots} chỗ warp, ${(warps * perWarp).toLocaleString('en-US')} trên ${(BOOK.regs as number).toLocaleString('en-US')} thanh ghi`)}
          sub={second ? tr(`twice the registers per thread: ${FITS_B.residentWarps} warps fit where ${FITS_A.residentWarps} did`, `gấp đôi số thanh ghi mỗi luồng: chỉ vừa ${FITS_B.residentWarps} warp ở chỗ trước đây vừa ${FITS_A.residentWarps}`) : tr("the book's example SM: the register file is shared out among the resident warps", 'SM trong ví dụ của sách: tệp thanh ghi được chia cho các warp thường trú')}
          data={{ beat: 5, regs, warps, fit: fits.residentWarps, slots: fits.warpSlots ?? 0 }}
        >
          <text x={X} y={40} fontSize="12.5" fontWeight="600"><Translated vi="tệp thanh ghi của một SM (ví dụ của sách)">the register file of one SM (the book's example)</Translated></text>
          <text x={X} y={56} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
            <Translated vi={<>một ô = {REGS_PER_UNIT.toLocaleString('en-US')} thanh ghi; một hộp đánh số là phần của một warp thường trú</>}>
              one cell = {REGS_PER_UNIT.toLocaleString('en-US')} registers; a numbered box is one resident warp's share
            </Translated>
          </text>
          {cells}
        </Frame>
      );
    },
  },
  {
    clip: 'latency-6',
    get label() { return tr('Two kernels', 'Hai kernel'); },
    draw: (p) => {
      const i = Math.min(LADDER.length - 1, Math.floor(ramp(p, 0.1, 0.8) * LADDER.length));
      return (
        <Frame
          headline={tr(`busy with ${warpsNeeded(PRESETS.computeHeavy.work, PRESETS.computeHeavy.latency)} warps on the left; ${warpsNeeded(PRESETS.loadHeavy.work, PRESETS.loadHeavy.latency)} needed on the right`, `bên trái bận kín với ${warpsNeeded(PRESETS.computeHeavy.work, PRESETS.computeHeavy.latency)} warp; bên phải cần ${warpsNeeded(PRESETS.loadHeavy.work, PRESETS.loadHeavy.latency)}`)}
          sub={tr('token-by-token decoding looks like the right-hand kernel: little arithmetic per load', 'decode từng token một giống kernel bên phải: ít phép tính cho mỗi lần nạp')}
          data={{ beat: 6, rung: LADDER[i], left: pct(HEAVY[i].use), right: pct(LOADY[i].use) }}
        >
          <Side x={30} title={tr('long compute between loads', 'tính lâu giữa các lần nạp')} sub={tr(`${PRESETS.computeHeavy.work} cycles of work, then a ${PRESETS.computeHeavy.latency}-cycle wait`, `${PRESETS.computeHeavy.work} chu kỳ làm việc, rồi ${PRESETS.computeHeavy.latency} chu kỳ chờ`)} rung={HEAVY[i]} />
          <Side x={384} title={tr('a load every other instruction', 'cứ hai lệnh lại có một lệnh nạp')} sub={tr(`${PRESETS.loadHeavy.work} cycles of work, then a ${PRESETS.loadHeavy.latency}-cycle wait`, `${PRESETS.loadHeavy.work} chu kỳ làm việc, rồi ${PRESETS.loadHeavy.latency} chu kỳ chờ`)} rung={LOADY[i]} />
        </Frame>
      );
    },
  },
];

export function LatencyScene() {
  return <Scene chapter="ch04" title={tr('Hiding a wait', 'Che giấu thời gian chờ')} beats={beats} />;
}
