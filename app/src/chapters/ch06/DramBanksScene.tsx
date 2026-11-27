import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { banksNeeded, busBusy, busGaps, channelTimeline, inFlight, utilisationBound, type ChannelTimeline } from '../../sim/dram';
import { Translated, tr } from '../../lib/i18n';

// "Why one bank is not enough": section 6.2's Fig. 6.8 argument in time. Six beats, one
// narration clip each (content/ch06/narration/dram-banks-N.txt). Every bar, gap and
// percentage drawn comes from sim/dram.ts channelTimeline.
const W = 720;
const H = 360;
const X0 = 70; // left edge of the time axis
const PW = W - X0 - 16; // plot width
const TOP = 30;
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const pct = (v: number) => Math.round(v * 1000) / 10;

// The book's illustration (p. 134): latency 20 times the burst, a 16 GB/s channel.
export const R = 20;
export const CHANNEL_GBS = 16;
export const FULL = banksNeeded(R); // 21
/** Time drawn in beats 1 to 4, in burst times: three trips of one bank. */
export const SPAN = 3 * (R + 1) + 1;
/** Beat 5: the request that is sent to the bank the previous one is still using. */
export const CONFLICT_AT = 30;
const SPAN_CONFLICT = 84;
/** Beat 6: the moment at which requests in flight are counted. */
export const COUNT_AT = 50.5;

const ONE = channelTimeline({ R, banks: 1 });
const TWO = channelTimeline({ R, banks: 2 });
const byBanks = Array.from({ length: FULL }, (_, i) => channelTimeline({ R, banks: i + 1 }));
const CONFLICT = channelTimeline({ R, banks: FULL, forced: [CONFLICT_AT] });
const FULL_TL = byBanks[FULL - 1];

const LAT = { fill: 'var(--global-wash)', stroke: 'var(--global)' };
const BURST = { fill: 'var(--global)', stroke: 'var(--global)' };

export interface ChannelDrawingProps {
  tl: ChannelTimeline;
  /** Height of the drawing in its own units; the scene keeps it fixed so parts do not jump. */
  height?: number;
  /** Below this width (px) the drawing stops shrinking and its container scrolls instead. */
  minWidth?: number;
  label?: string;
  /** Time drawn, in burst times. */
  span: number;
  /** Only what happened before this time is drawn. */
  until?: number;
  /** Fixed row height, so rows can appear one by one without the drawing jumping. */
  rowH?: number;
  /** Request numbers drawn in the failure colour. */
  bad?: number[];
  /** Bus gaps between these times are marked. */
  gapsIn?: [number, number];
  /** A vertical line at this time. */
  mark?: number;
  headline?: ReactNode;
  sub?: ReactNode;
  extra?: ReactNode;
  data?: Record<string, string | number>;
}

/** One channel in time: a row per bank, the bus underneath. Also drawn by the explorer in section 6.2. */
export function ChannelDrawing({
  tl, span, until = span, rowH, bad = [], gapsIn, mark, headline, sub, extra, data = {}, height, minWidth,
  label = tr("Animation of DRAM banks sharing one channel's bus", 'Hoạt cảnh các bank DRAM dùng chung bus của một kênh'),
}: ChannelDrawingProps) {
  const nb = tl.config.banks;
  const rh = rowH ?? Math.max(3, Math.min(30, Math.floor(196 / nb)));
  const X = (t: number) => X0 + (Math.min(t, span) / span) * PW;
  const busY = TOP + nb * rh + 14;
  const busH = 20;
  const end = Math.min(until, span);
  const clip = (s: number, e: number) => [Math.max(0, s), Math.min(e, end)] as const;
  const labelled = nb <= 4 ? tl.banks.map((_, b) => b) : [0, nb - 1];
  const attrs = Object.fromEntries(Object.entries({ banks: nb, r: tl.config.R, ...data }).map(([k, v]) => ['data-' + k, String(v)]));
  const gaps = gapsIn ? busGaps(tl, gapsIn[0], Math.min(gapsIn[1], end)) : [];
  const HH = height ?? busY + busH + (gaps.length ? 40 : 24) + (headline ? 44 : 0);
  return (
    <svg viewBox={`0 0 ${W} ${HH}`} role="img" aria-label={label} style={{ width: '100%', maxWidth: W, minWidth, display: 'block' }} {...attrs}>
      {/* legend */}
      <rect x={X0} y={8} width={14} height={9} fill={LAT.fill} stroke={LAT.stroke} strokeWidth="0.8" />
      <text x={X0 + 20} y={16} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi="bank bận truy cập các ô nhớ của nó (độ trễ)">bank busy reaching its cells (latency)</Translated></text>
      <rect x={X0 + 250} y={8} width={14} height={9} fill={BURST.fill} />
      <text x={X0 + 270} y={16} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi="burst trên bus">burst on the bus</Translated></text>
      {/* banks */}
      {tl.banks.map((segs, b) => (
        <g key={b} data-bank={b}>
          <rect x={X0} y={TOP + b * rh} width={PW} height={Math.max(1, rh - 2)} fill="none" stroke="var(--rule-soft)" />
          {segs.map((s) => {
            const [a, e] = clip(s.start, s.end);
            if (e <= a) return null;
            const isBad = bad.includes(s.k);
            const st = s.kind === 'burst' ? BURST : LAT;
            return (
              <rect key={s.kind + s.k} x={X(a)} y={TOP + b * rh} width={Math.max(0.8, X(e) - X(a))} height={Math.max(1, rh - 2)}
                fill={isBad ? (s.kind === 'burst' ? 'var(--fail)' : 'var(--fail-wash)') : st.fill}
                stroke={isBad ? 'var(--fail)' : st.stroke} strokeWidth={rh >= 8 ? 0.7 : 0} />
            );
          })}
        </g>
      ))}
      {labelled.map((b) => (
        <text key={b} x={X0 - 8} y={TOP + b * rh + Math.max(rh / 2 + 3, 8)} fontSize={rh >= 16 ? 11.5 : 9.5} textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
          <Translated vi={<>bank {b}</>}>bank {b}</Translated>
        </text>
      ))}
      {nb > 4 ? (
        <text x={X0 - 8} y={TOP + (nb * rh) / 2 + 4} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>⋮</text>
      ) : null}
      {/* the bus */}
      <text x={X0 - 8} y={busY + 14} fontSize="11.5" fontWeight="600" textAnchor="end"><Translated vi="bus">bus</Translated></text>
      <rect x={X0} y={busY} width={PW} height={busH} fill="none" stroke="var(--ink-3)" strokeDasharray="3 3" strokeWidth="0.8" />
      {tl.bus.map((s) => {
        const [a, e] = clip(s.start, s.end);
        if (e <= a) return null;
        return <rect key={s.k} x={X(a)} y={busY} width={Math.max(0.8, X(e) - X(a))} height={busH} fill={bad.includes(s.k) ? 'var(--fail)' : BURST.fill} />;
      })}
      {gaps.map(([a, e]) => (
        <g key={a}>
          <rect x={X(a)} y={busY - 3} width={X(e) - X(a)} height={busH + 6} fill="var(--fail-wash)" stroke="var(--fail)" strokeDasharray="4 2" />
          <text x={(X(a) + X(e)) / 2} y={busY + busH + 16} fontSize="11" textAnchor="middle" style={{ fill: 'var(--fail)' }}>
            <Translated vi={<>bus rảnh trong {Math.floor(e - a)} thời gian burst</>}>bus idle for {Math.floor(e - a)} burst times</Translated>
          </text>
        </g>
      ))}
      {/* time axis */}
      <text x={X0} y={busY + busH + (gaps.length ? 32 : 16)} fontSize="10" style={{ fill: 'var(--ink-2)' }}><Translated vi="thời điểm 0">time 0</Translated></text>
      <text x={X0 + PW} y={busY + busH + (gaps.length ? 32 : 16)} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi={<>{span} thời gian burst</>}>{span} burst times</Translated>
      </text>
      {until < span ? <line x1={X(until)} x2={X(until)} y1={TOP - 4} y2={busY + busH + 4} stroke="var(--ink)" strokeWidth="1" /> : null}
      {mark != null ? (
        <g>
          <line x1={X(mark)} x2={X(mark)} y1={TOP - 4} y2={busY + busH + 4} stroke="var(--ink)" strokeWidth="1.6" />
          <text x={X(mark) + 5} y={TOP - 6} fontSize="10.5" fontWeight="600"><Translated vi="bây giờ">now</Translated></text>
        </g>
      ) : null}
      {extra}
      {headline ? <text x={24} y={HH - 30} fontSize="13.5" fontWeight="600">{headline}</text> : null}
      {sub ? <text x={24} y={HH - 11} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>{sub}</text> : null}
    </svg>
  );
}

/** A horizontal gauge in the empty lower half of beats 1 to 3: a share of the bus's time. */
function Meter({ y, label, fraction, value }: { y: number; label: string; fraction: number; value: string }) {
  const w = 300;
  return (
    <g data-meter={label}>
      <text x={X0} y={y - 6} fontSize="11.5">{label}</text>
      <rect x={X0} y={y} width={w} height={13} fill="none" stroke="var(--global)" />
      <rect x={X0} y={y} width={w * clamp(fraction)} height={13} fill="var(--global)" />
      <text x={X0 + w + 10} y={y + 11} fontSize="11.5" fontWeight="600">{value}</text>
    </g>
  );
}

/** Beat 2's comparison: what the bus could move, what one bank lets it deliver. */
function Delivered({ grow }: { grow: number }) {
  const got = ONE.utilisation * CHANNEL_GBS;
  return (
    <g opacity={grow > 0 ? 1 : 0}>
      <Meter y={238} label={tr(`the bus could move ${CHANNEL_GBS} GB/s; one bank lets it deliver`, `bus có thể chuyển ${CHANNEL_GBS} GB/s; một bank chỉ cho nó cung cấp`)} fraction={(got / CHANNEL_GBS) * grow}
        value={`${(Math.round(got * 100) / 100).toFixed(2)} GB/s`} />
    </g>
  );
}

const beats: Beat[] = [
  {
    clip: 'dram-banks-1',
    get label() {
      return tr('One bank', 'Một bank');
    },
    draw: (p) => {
      const until = p * SPAN;
      const busy = Math.round(busBusy(ONE, 0, Math.max(until, 1e-9)) * until);
      return (
        <ChannelDrawing height={H} tl={ONE} span={SPAN} until={until} rowH={30}
          data={{ beat: 1, until: Math.round(until * 10) / 10, busy }}
          extra={<Meter y={176} label={tr('bus busy so far', 'thời gian bus bận đến giờ')} fraction={until > 0 ? busy / until : 0} value={until >= 1 ? tr(`${busy} of ${Math.floor(until)} burst times`, `${busy} trên ${Math.floor(until)} thời gian burst`) : ''} />}
          headline={until < R ? tr('the bank is reaching into its cell array; nothing on the bus yet', 'bank đang truy cập vào mảng ô nhớ của nó; chưa có gì trên bus') : tr(`the bus has carried data for ${busy} of ${Math.floor(until)} burst times`, `bus đã chở dữ liệu trong ${busy} trên ${Math.floor(until)} thời gian burst`)}
          sub={tr(`one access: ${R} burst times of latency, then 1 of burst`, `một lần truy cập: ${R} thời gian burst cho độ trễ, rồi 1 cho burst`)} />
      );
    },
  },
  {
    clip: 'dram-banks-2',
    get label() {
      return tr('Mostly idle', 'Phần lớn thời gian rảnh');
    },
    draw: (p) => (
      <ChannelDrawing height={H} tl={ONE} span={SPAN} rowH={30}
        data={{ beat: 2, util: pct(ONE.utilisation), gb: Math.round(ONE.utilisation * CHANNEL_GBS * 100) / 100 }}
        extra={
          <>
            <Meter y={176} label={tr('bus busy, one bank', 'bus bận, một bank')} fraction={ONE.utilisation} value={tr(`1 of ${R + 1}: ${pct(ONE.utilisation)}%`, `1 trên ${R + 1}: ${pct(ONE.utilisation)}%`)} />
            <Delivered grow={ramp(p, 0.4, 0.8)} />
          </>
        }
        headline={tr(`one bank: bus busy 1 of every ${R + 1} burst times, ${pct(ONE.utilisation)}%`, `một bank: bus bận 1 trên mỗi ${R + 1} thời gian burst, ${pct(ONE.utilisation)}%`)}
        sub={p > 0.4 ? tr(`${CHANNEL_GBS} GB/s × ${pct(ONE.utilisation)}% ≈ ${(Math.round(ONE.utilisation * CHANNEL_GBS * 100) / 100).toFixed(2)} GB/s delivered`, `${CHANNEL_GBS} GB/s × ${pct(ONE.utilisation)}% ≈ ${(Math.round(ONE.utilisation * CHANNEL_GBS * 100) / 100).toFixed(2)} GB/s được cung cấp`) : tr('latency to burst 20 : 1, the book\'s illustration (p. 134)', 'độ trễ so với burst là 20 : 1, minh họa của sách (p. 134)')} />
    ),
  },
  {
    clip: 'dram-banks-3',
    get label() {
      return tr('A second bank', 'Một bank thứ hai');
    },
    draw: (p) => {
      const until = Math.max(0.001, p) * SPAN;
      return (
        <ChannelDrawing height={H} tl={TWO} span={SPAN} until={until} rowH={30}
          data={{ beat: 3, util: pct(TWO.utilisation) }}
          extra={
            <>
              <Meter y={176} label={tr('bus busy, one bank', 'bus bận, một bank')} fraction={ONE.utilisation} value={`${pct(ONE.utilisation)}%`} />
              <Meter y={226} label={tr('bus busy, two banks', 'bus bận, hai bank')} fraction={TWO.utilisation * ramp(p, 0.35, 0.7)} value={p >= 0.35 ? `${pct(TWO.utilisation)}%` : ''} />
            </>
          }
          headline={until > R + 2 ? tr(`two banks: bursts go out back to back, bus busy ${pct(TWO.utilisation)}%`, `hai bank: các burst nối tiếp sát nhau, bus bận ${pct(TWO.utilisation)}%`) : tr('bank 1 starts while bank 0 is still waiting', 'bank 1 bắt đầu trong khi bank 0 vẫn đang chờ')}
          sub={tr(`twice one bank's ${pct(ONE.utilisation)}%: the latencies overlap`, `gấp đôi ${pct(ONE.utilisation)}% của một bank: các độ trễ chồng lên nhau`)} />
      );
    },
  },
  {
    clip: 'dram-banks-4',
    get label() {
      return tr('R + 1 banks', 'R + 1 bank');
    },
    draw: (p) => {
      const nb = 1 + Math.round(ramp(p, 0.05, 0.8) * (FULL - 1));
      const tl = byBanks[nb - 1];
      return (
        <ChannelDrawing height={H} tl={tl} span={SPAN} rowH={9.4}
          data={{ beat: 4, util: pct(tl.utilisation), bound: pct(utilisationBound(R, nb)) }}
          headline={nb === FULL ? tr(`${FULL} banks, R + 1: the bursts tile the bus, busy ${pct(tl.utilisation)}%`, `${FULL} bank, R + 1: các burst lấp kín bus, bận ${pct(tl.utilisation)}%`) : tr(`${nb} ${nb === 1 ? 'bank' : 'banks'}: bus busy ${nb} of every ${R + 1} burst times, ${pct(tl.utilisation)}%`, `${nb} bank: bus bận ${nb} trên mỗi ${R + 1} thời gian burst, ${pct(tl.utilisation)}%`)}
          sub={tr('after the first latency, which no number of banks can hide', 'sau độ trễ đầu tiên, thứ mà không số bank nào che giấu được')} />
      );
    },
  },
  {
    clip: 'dram-banks-5',
    get label() {
      return tr('A conflict', 'Một xung đột');
    },
    draw: (p) => {
      const until = 30 + p * (SPAN_CONFLICT - 30);
      const prev = CONFLICT.accesses[CONFLICT_AT - 1];
      const hit = CONFLICT.accesses[CONFLICT_AT];
      const gap = busGaps(CONFLICT, R, Math.min(until, SPAN_CONFLICT)).reduce((s, [a, e]) => s + e - a, 0);
      return (
        <ChannelDrawing height={H} tl={CONFLICT} span={SPAN_CONFLICT} until={until} rowH={9.4} bad={[CONFLICT_AT - 1, CONFLICT_AT]} gapsIn={[R, SPAN_CONFLICT]}
          data={{ beat: 5, gap, bank: hit.bank, waited: hit.start - prev.start - 1 }}
          headline={until < hit.start ? tr(`requests ${CONFLICT_AT - 1} and ${CONFLICT_AT} both want bank ${hit.bank}`, `yêu cầu ${CONFLICT_AT - 1} và ${CONFLICT_AT} cùng cần bank ${hit.bank}`) : tr(`bank conflict: request ${CONFLICT_AT} waited for bank ${hit.bank} until time ${hit.start}`, `xung đột bank: yêu cầu ${CONFLICT_AT} phải chờ bank ${hit.bank} đến thời điểm ${hit.start}`)}
          sub={tr(`it could start only when request ${CONFLICT_AT - 1}'s burst ended, at ${prev.burst + 1}; every later request queues behind it`, `nó chỉ có thể bắt đầu khi burst của yêu cầu ${CONFLICT_AT - 1} kết thúc, tại ${prev.burst + 1}; mọi yêu cầu sau đó xếp hàng phía sau nó`)} />
      );
    },
  },
  {
    clip: 'dram-banks-6',
    get label() {
      return tr('In flight', 'Đang chờ phục vụ');
    },
    draw: (p) => {
      const n = inFlight(FULL_TL, COUNT_AT);
      const shown = ramp(p, 0.1, 0.5);
      return (
        <ChannelDrawing height={H} tl={FULL_TL} span={SPAN} rowH={9.4} mark={COUNT_AT}
          data={{ beat: 6, inflight: n, util: pct(FULL_TL.utilisation) }}
          extra={
            <g opacity={shown}>
              <rect x={X0 + (COUNT_AT / SPAN) * PW - 196} y={TOP + 92} width={188} height={40} rx={3} fill="var(--page)" stroke="var(--rule)" />
              <text x={X0 + (COUNT_AT / SPAN) * PW - 186} y={TOP + 109} fontSize="12" fontWeight="600"><Translated vi={<>{n} yêu cầu đang chờ phục vụ</>}>{n} requests in flight</Translated></text>
              <text x={X0 + (COUNT_AT / SPAN) * PW - 186} y={TOP + 125} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi="mỗi bank một yêu cầu, ở mọi thời điểm">one in every bank, at every moment</Translated></text>
            </g>
          }
          headline={p < 0.55 ? tr(`a full bus: ${n} requests in flight at once`, `một bus đầy: ${n} yêu cầu cùng lúc đang chờ phục vụ`) : tr('each one is a load some warp issued: more warps, or more bytes per load', 'mỗi yêu cầu là một lần nạp do một warp nào đó phát ra: thêm warp, hoặc thêm byte mỗi lần nạp')}
          sub={p < 0.55 ? tr(`R + 1 = ${FULL}, the same count as the banks it takes`, `R + 1 = ${FULL}, bằng đúng số bank cần có`) : tr('occupancy (this section) and vector loads (section 6.3) are the two ways to keep them coming', 'occupancy (mục này) và nạp dạng vector (mục 6.3) là hai cách giữ cho chúng đến liên tục')} />
      );
    },
  },
];

export function DramBanksScene() {
  return <Scene chapter="ch06" title={tr('Why one bank is not enough', 'Vì sao một bank là không đủ')} beats={beats} />;
}
