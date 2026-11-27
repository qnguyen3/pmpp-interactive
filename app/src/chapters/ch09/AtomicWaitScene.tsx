import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import {
  SCENE,
  SKEWED4,
  UNIFORM4,
  longestChain,
  sceneBins,
  sceneLoads,
  sceneLockstep,
  sceneOneBin,
  scenePrivatised,
  type Bar,
  type Timeline,
} from '../../sim/atomicrate';
import { Translated, tr } from '../../lib/i18n';

// "Why one busy bin sets the pace" (section 9.3). Six beats, one narration clip each
// (content/ch09/narration/atomic-wait-N.txt). Every bar's start and end comes from
// sim/atomicrate.ts timeline(): DRAM at the book's 200-cycle example, the cache,
// shared-memory and issue figures illustrative (SCENE).
const W = 720;
const H = 360;
const X0 = 128; // start of the time axis
const X1 = 704;
const TMAX = 3400;
const Y0 = 44;
const ROW = 23;
const x = (t: number) => X0 + (Math.min(t, TMAX) / TMAX) * (X1 - X0);
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const fmtInt = (n: number) => n.toLocaleString('en-US');

const LOADS = sceneLoads();
const ONE = sceneOneBin();
const LOCK = sceneLockstep();
const UNI = sceneBins(UNIFORM4);
const SKEW = sceneBins(SKEWED4);
const SKEW_CACHE = sceneBins(SKEWED4, SCENE.cache);
const PRIV = scenePrivatised();

// Row labels: the regexes below match atomicrate.ts's two row names, 'thread N' and 'block N commit',
// and give the Vietnamese; the English name stays the bar's identity.
type Tone = 'global' | 'shared';

function Bars({ tl, now, tone = (_b: Bar) => 'global' as Tone, label, valuesAt }: {
  tl: Timeline;
  now: number;
  tone?: (b: Bar) => Tone;
  label?: (b: Bar, i: number) => string;
  /** Put every value label at this time instead of after its own bar. */
  valuesAt?: number;
}) {
  return (
    <g>
      {tl.bars.map((b, i) => {
        const y = Y0 + i * ROW;
        const t = tone(b);
        const shown = now >= b.issue;
        const readTo = Math.min(now, b.readEnd);
        const storeFrom = b.storeStart ?? b.readEnd;
        const storeTo = Math.min(now, b.end);
        const done = now >= b.end;
        return (
          <g key={i} data-row={i} data-start={b.start} data-end={b.end}>
            <text x={X0 - 8} y={y + 14} fontSize="11" textAnchor="end">{label ? label(b, i) : tr(b.row, b.row.replace(/^thread (\d+)$/, 'luồng $1').replace(/^block (\d+) commit$/, 'khối $1 ghi về'))}</text>
            <line x1={X0} x2={X1} y1={y + ROW - 2} y2={y + ROW - 2} stroke="var(--rule-soft)" />
            {shown && b.start > b.issue ? (
              <line x1={x(b.issue)} x2={x(Math.min(now, b.start))} y1={y + 10} y2={y + 10} stroke="var(--ink-3)" strokeDasharray="3 3" />
            ) : null}
            {shown && now > b.start ? (
              <rect x={x(b.start)} y={y + 3} width={Math.max(0.8, x(readTo) - x(b.start))} height={14} fill={`var(--${t})`} />
            ) : null}
            {b.kind !== 'load' && now > storeFrom ? (
              <rect x={x(storeFrom)} y={y + 3} width={Math.max(0.8, x(storeTo) - x(storeFrom))} height={14}
                fill={`var(--${t}-wash)`} stroke={`var(--${t})`} strokeWidth="1" />
            ) : null}
            {done && b.kind !== 'load' ? (
              <text x={x(valuesAt ?? b.end) + 7} y={y + 14} fontSize="10" style={{ fill: b.kind === 'plain' ? 'var(--fail)' : 'var(--ink-2)' }}>
                {b.read}→{b.wrote}
              </text>
            ) : null}
          </g>
        );
      })}
    </g>
  );
}

function Axis({ rows }: { rows: number }) {
  const y = Y0 + rows * ROW + 6;
  return (
    <g>
      <line x1={X0} x2={X1} y1={y} y2={y} stroke="var(--ink-2)" />
      {[0, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200].map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={y} y2={y + 4} stroke="var(--ink-2)" />
          <text x={x(t)} y={y + 16} fontSize="10" textAnchor="middle">{t}</text>
        </g>
      ))}
      <text x={X1} y={y + 30} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{tr('clock cycles', 'chu kỳ xung nhịp')}</text>
    </g>
  );
}

function Finish({ t, rows, show, label, ghost = false }: { t: number; rows: number; show: number; label: string; ghost?: boolean }) {
  return (
    <g opacity={show} data-finish={ghost ? undefined : t} data-ghost={ghost ? t : undefined}>
      <line x1={x(t)} x2={x(t)} y1={Y0 - 6} y2={Y0 + rows * ROW + 6} stroke={ghost ? 'var(--ink-3)' : 'var(--ink)'} strokeWidth={ghost ? 1 : 1.6} strokeDasharray={ghost ? '2 4' : '5 3'} />
      <text x={Math.min(x(t) + 4, X1 - 4)} y={Y0 - 10} fontSize="10.5" textAnchor={x(t) > X1 - 140 ? 'end' : 'start'} style={{ fill: ghost ? 'var(--ink-3)' : 'var(--ink)' }}>
        {label}
      </text>
    </g>
  );
}

/** The current time; hidden once the finish line `end` has been reached. */
function Now({ t, rows, end }: { t: number; rows: number; end: number }) {
  if (t <= 0 || t > TMAX || t >= end) return null;
  return <line x1={x(t)} x2={x(t)} y1={Y0 - 4} y2={Y0 + rows * ROW + 4} stroke="var(--ink-2)" strokeWidth="0.8" />;
}

function Value({ v, lost, total }: { v: number; lost?: number; total?: number }) {
  return (
    <g data-bin-value={v} data-lost={lost ?? 0}>
      <rect x={X1 - 150} y={H - 46} width={150} height={38} rx="3" fill="var(--global-wash)" stroke="var(--global)" />
      <text x={X1 - 140} y={H - 31} fontSize="10.5" style={{ fill: 'var(--global)' }}>{tr('the bin, global memory', 'ô đếm, bộ nhớ toàn cục')}</text>
      <text x={X1 - 140} y={H - 14} fontSize="14" fontWeight="600">
        {v}
        {total != null ? <tspan fontSize="11" fontWeight="400"><Translated vi={<> trên {total} lần cập nhật</>}> of {total} updates</Translated></tspan> : null}
      </text>
      {lost ? (
        <text x={X1 - 156} y={H - 14} fontSize="12" fontWeight="600" textAnchor="end" style={{ fill: 'var(--fail)' }}>
          <Translated vi={<>mất {lost} lần cập nhật</>}>
            {lost} updates lost
          </Translated>
        </text>
      ) : null}
    </g>
  );
}

function Stage({ title, children, foot }: { title: ReactNode; children: ReactNode; foot?: ReactNode }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Timeline of memory requests', 'Dòng thời gian của các yêu cầu bộ nhớ')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <text x={10} y={18} fontSize="12.5" fontWeight="600">{title}</text>
      {children}
      <text x={10} y={H - 6} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>
        {foot ?? tr(`memory access ${SCENE.dram} cycles (the book's example); one issue slot ${SCENE.issue} cycles (illustrative)`, `truy cập bộ nhớ ${SCENE.dram} chu kỳ (ví dụ của sách); một khe phát lệnh ${SCENE.issue} chu kỳ (minh họa)`)}
      </text>
    </svg>
  );
}

/** Bin values reached by `now`: completed atomics only. */
const valueAt = (tl: Timeline, address: string, now: number) =>
  tl.bars.filter((b) => b.address === address && b.kind === 'atomic' && b.end <= now).reduce((m, b) => Math.max(m, b.wrote ?? 0), 0);

const beats: Beat[] = [
  {
    clip: 'atomic-wait-1',
    get label() {
      return tr('Loads overlap', 'Các lần nạp chồng lên nhau');
    },
    draw: (p) => {
      const now = ramp(p, 0.05, 0.75) * LOADS.end * 1.08;
      return (
        <Stage title={tr('8 threads load from 8 different addresses', '8 luồng nạp từ 8 địa chỉ khác nhau')}>
          <Axis rows={8} />
          <Bars tl={LOADS} now={now} />
          <Now t={now} rows={8} end={LOADS.end} />
          <Finish t={LOADS.end} rows={8} show={now >= LOADS.end ? 1 : 0} label={tr(`all done at ${fmtInt(LOADS.end)} cycles`, `xong hết lúc ${fmtInt(LOADS.end)} chu kỳ`)} />
          <text x={X0} y={H - 22} fontSize="12" opacity={ramp(p, 0.78, 0.9)}>
            {tr('many requests in flight at once: the latencies overlap', 'nhiều yêu cầu cùng đang bay: các độ trễ chồng lên nhau')}
          </text>
        </Stage>
      );
    },
  },
  {
    clip: 'atomic-wait-2',
    get label() {
      return tr('One address', 'Một địa chỉ');
    },
    draw: (p) => {
      const now = ramp(p, 0.05, 0.92) * ONE.end * 1.01;
      return (
        <Stage title={tr('the same 8 threads: an atomic add on one bin', 'cùng 8 luồng đó: một phép cộng nguyên tử trên một ô đếm')}>
          <Axis rows={8} />
          <Finish t={LOADS.end} rows={8} show={1} label={tr('loads', 'các lần nạp')} ghost />
          <Bars tl={ONE} now={now} />
          <Now t={now} rows={8} end={ONE.end} />
          <Finish t={ONE.end} rows={8} show={now >= ONE.end ? 1 : 0} label={tr(`done at ${fmtInt(ONE.end)} = 8 × ${2 * SCENE.dram} cycles`, `xong lúc ${fmtInt(ONE.end)} = 8 × ${2 * SCENE.dram} chu kỳ`)} />
          <Value v={valueAt(ONE, 'bin', now)} />
          <text x={X0} y={H - 22} fontSize="11" opacity={ramp(p, 0.1, 0.2)}>
            <Translated vi={<>mỗi thanh: nạp ({SCENE.dram}), rồi ghi ({SCENE.dram}); nét đứt: đang chờ ô đếm</>}>
              each bar: load ({SCENE.dram}), then store ({SCENE.dram}); dashes: waiting for the bin
            </Translated>
          </text>
        </Stage>
      );
    },
  },
  {
    clip: 'atomic-wait-3',
    get label() {
      return tr('Without the atomic', 'Không có atomic');
    },
    draw: (p) => {
      const now = ramp(p, 0.05, 0.7) * LOCK.end * 1.08;
      const done = now >= LOCK.end;
      return (
        <Stage title={tr('no atomic: a warp of 8 lanes reads, adds and writes in step', 'không atomic: một warp 8 lane đọc, cộng và ghi đồng bộ')}>
          <Axis rows={8} />
          <Bars tl={LOCK} now={now} valuesAt={LOCK.end} />
          <Now t={now} rows={8} end={LOCK.end} />
          <Finish t={LOCK.end} rows={8} show={done ? 1 : 0} label={tr(`done at ${fmtInt(LOCK.end)} cycles: fast`, `xong lúc ${fmtInt(LOCK.end)} chu kỳ: nhanh`)} />
          <Value v={done ? LOCK.memory.bin : 0} lost={done ? LOCK.lost : 0} total={done ? LOCK.increments : undefined} />
          <text x={X0} y={H - 22} fontSize="11" opacity={ramp(p, 0.1, 0.2)}>
            {tr('every lane reads 0 and writes 1', 'mọi lane đều đọc 0 và ghi 1')}
          </text>
        </Stage>
      );
    },
  },
  {
    clip: 'atomic-wait-4',
    get label() {
      return tr('Many bins', 'Nhiều ô đếm');
    },
    draw: (p) => {
      const skewed = p >= 0.45;
      const tl = skewed ? SKEW : UNI;
      const bins = skewed ? SKEWED4 : UNIFORM4;
      const now = (skewed ? ramp(p, 0.47, 0.9) : ramp(p, 0.04, 0.38)) * tl.end * 1.03;
      return (
        <Stage title={skewed ? tr(`skewed: ${longestChain(SKEWED4)} of the 8 updates go to bin 0`, `lệch: ${longestChain(SKEWED4)} trong 8 lần cập nhật rơi vào ô 0`) : tr('uniform: 8 updates spread over 4 bins', 'đều: 8 lần cập nhật trải trên 4 ô')}>
          <Axis rows={8} />
          {skewed ? <Finish t={UNI.end} rows={8} show={1} label={tr('uniform', 'đều')} ghost /> : null}
          <Finish t={ONE.end} rows={8} show={1} label={tr('one bin', 'một ô')} ghost />
          <Bars tl={tl} now={now} label={(_b, i) => tr(`thread ${i} → bin ${bins[i]}`, `luồng ${i} → ô ${bins[i]}`)} />
          <Now t={now} rows={8} end={tl.end} />
          <Finish t={tl.end} rows={8} show={now >= tl.end ? 1 : 0}
            label={tr(`done at ${fmtInt(tl.end)}`, `xong lúc ${fmtInt(tl.end)}`)} />
          <text x={X0} y={H - 22} fontSize="11.5" opacity={skewed ? ramp(p, 0.88, 0.95) : ramp(p, 0.3, 0.38)}>
            {skewed
              ? tr("the busiest bin's chain sets the finish; bins 1 to 3 finish early and sit idle", 'chuỗi của ô bận nhất quyết định lúc kết thúc; các ô 1 đến 3 xong sớm và ngồi không')
              : tr(`four chains of ${longestChain(UNIFORM4)}, side by side`, `bốn chuỗi dài ${longestChain(UNIFORM4)}, chạy song song`)}
          </text>
        </Stage>
      );
    },
  },
  {
    clip: 'atomic-wait-5',
    get label() {
      return tr('Shorter links', 'Mắt xích ngắn hơn');
    },
    draw: (p) => {
      const now = ramp(p, 0.08, 0.6) * SKEW_CACHE.end * 1.1;
      return (
        <Stage
          title={tr('the same skewed chains, with the atomics done in the last-level cache', 'cùng các chuỗi lệch đó, với các atomic thực hiện trong bộ nhớ đệm cấp cuối')}
          foot={tr(`cache access ${SCENE.cache} cycles and issue slot ${SCENE.issue} cycles: illustrative; the book says only "tens of cycles"`, `truy cập bộ nhớ đệm ${SCENE.cache} chu kỳ và khe phát lệnh ${SCENE.issue} chu kỳ: minh họa; sách chỉ nói "hàng chục chu kỳ"`)}
        >
          <Axis rows={8} />
          <Finish t={SKEW.end} rows={8} show={1} label={tr('in DRAM', 'trong DRAM')} ghost />
          <Bars tl={SKEW_CACHE} now={now} label={(_b, i) => tr(`thread ${i} → bin ${SKEWED4[i]}`, `luồng ${i} → ô ${SKEWED4[i]}`)} />
          <Now t={now} rows={8} end={SKEW_CACHE.end} />
          <Finish t={SKEW_CACHE.end} rows={8} show={now >= SKEW_CACHE.end ? 1 : 0} label={tr(`done at ${fmtInt(SKEW_CACHE.end)}`, `xong lúc ${fmtInt(SKEW_CACHE.end)}`)} />
          <text x={X0} y={H - 22} fontSize="11.5" opacity={ramp(p, 0.62, 0.75)}>
            <Translated vi={<>mỗi mắt xích ngắn hơn {(2 * SCENE.dram) / (2 * SCENE.cache)} lần; ô bận nhất vẫn quyết định nhịp độ</>}>
              every link {(2 * SCENE.dram) / (2 * SCENE.cache)} times shorter; the busiest bin still sets the pace
            </Translated>
          </text>
        </Stage>
      );
    },
  },
  {
    clip: 'atomic-wait-6',
    get label() {
      return tr('Next: privatise', 'Tiếp theo: riêng tư hóa');
    },
    draw: (p) => {
      const now = ramp(p, 0.05, 0.75) * PRIV.end * 1.06;
      const rows = PRIV.bars.length;
      return (
        <Stage
          title={tr('2 blocks of 4: count in a private copy, then commit once per block', '2 khối 4 luồng: đếm trong một bản sao riêng, rồi ghi về một lần mỗi khối')}
          foot={tr(`shared memory ${SCENE.shared} cycles per access (illustrative; the book: "a few cycles"); commits ${SCENE.dram} cycles (the book's DRAM)`, `bộ nhớ chia sẻ ${SCENE.shared} chu kỳ mỗi lần truy cập (minh họa; sách: "vài chu kỳ"); ghi về ${SCENE.dram} chu kỳ (DRAM của sách)`)}
        >
          <Axis rows={rows} />
          <Finish t={ONE.end} rows={rows} show={1} label={tr('all 8 on the global bin', 'cả 8 trên ô toàn cục')} ghost />
          <Bars tl={PRIV} now={now} tone={(b) => (b.address.startsWith('private') ? 'shared' : 'global')}
            label={(b) => (b.address.startsWith('private') ? tr(`${b.row} → copy ${b.address.slice(-1)}`, `${b.row.replace(/^thread (\d+)$/, 'luồng $1').replace(/^block (\d+) commit$/, 'khối $1 ghi về')} → bản sao ${b.address.slice(-1)}`) : tr(b.row, b.row.replace(/^thread (\d+)$/, 'luồng $1').replace(/^block (\d+) commit$/, 'khối $1 ghi về')))} />
          <Now t={now} rows={rows} end={PRIV.end} />
          <Finish t={PRIV.end} rows={rows} show={now >= PRIV.end ? 1 : 0} label={tr(`done at ${fmtInt(PRIV.end)}`, `xong lúc ${fmtInt(PRIV.end)}`)} />
          <Value v={valueAt(PRIV, 'bin', now)} />
          <text x={X0} y={H - 22} fontSize="11.5" opacity={ramp(p, 0.78, 0.9)}>
            {tr('one short commit per block replaces the long global chain', 'một lần ghi về ngắn cho mỗi khối thay cho chuỗi toàn cục dài')}
          </text>
        </Stage>
      );
    },
  },
];

export function AtomicWaitScene() {
  return <Scene chapter="ch09" title={tr('Why one busy bin sets the pace', 'Vì sao một ô đếm bận rộn quyết định nhịp độ')} beats={beats} />;
}
