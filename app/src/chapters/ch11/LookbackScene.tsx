import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import {
  SCENE_G,
  SCENE_R,
  flagsAt,
  phaseAt,
  rows,
  sceneRuns,
  simulate,
  type Flag,
  type LookbackRun,
  type Phase,
} from '../../sim/lookback';

// "Waiting for the block before you" (section 11.9): six blocks on a device that holds three,
// doing an inter-block scan. Six beats, one narration clip each
// (content/ch11/narration/lookback-N.txt). Every timeline, flag and count drawn comes from
// sim/lookback.ts runs with fixed seeds (sceneRuns).
export const SCENE = sceneRuns();
/** The same claimed-index single lookback with 32 blocks, all resident: the chain the narration names. */
export const SCENE_WIDE = simulate({ G: 32, R: 32, order: 'index', seed: 6, index: 'counter', scheme: 'single', progress: 'started' });

const W = 720;
const H = 340;
const LX = 8; // row labels
const TX = 150; // timeline start
const TW = 400; // timeline width
const FX = 580; // flags column
const TOP = 46;
const RH = 28;
const T_MAX = 24;
const X = (t: number) => TX + (Math.min(t, T_MAX) / T_MAX) * TW;
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

const FILL: Record<Phase, string> = {
  local: 'var(--ink-3)',
  look: 'var(--ink)',
  spin: 'url(#lb-spin)',
  barrier: 'url(#lb-spin)',
  scan: 'var(--ink)',
  finish: 'var(--ink-2)',
  starved: 'none',
};

function Defs() {
  return (
    <defs>
      <pattern id="lb-spin" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="6" height="6" fill="var(--global-wash)" />
        <line x1="0" y1="0" x2="0" y2="6" stroke="var(--global)" strokeWidth="2" />
      </pattern>
    </defs>
  );
}

function Legend({ y }: { y: number }) {
  const items: [string, ReactNode][] = [
    [tr('scanning its segment', 'đang scan đoạn của mình'), <rect key="a" width="14" height="10" fill="var(--ink-3)" opacity="0.6" />],
    [tr('spinning on a flag', 'quay chờ trên một cờ'), <rect key="b" width="14" height="10" fill="url(#lb-spin)" />],
    [tr('looking back, adding', 'nhìn lại, cộng dồn'), <rect key="c" width="14" height="10" fill="var(--ink)" />],
    [tr('adding and storing', 'cộng và ghi'), <rect key="d" width="14" height="10" fill="var(--ink-2)" />],
    [tr('resident, not run', 'thường trú, không được chạy'), <rect key="e" width="14" height="10" fill="none" stroke="var(--fail)" strokeDasharray="3 2" />],
  ];
  // Five items in one row from the left edge: about 5.3 px a character at this size.
  let x = LX;
  return (
    <g>
      {items.map(([label, sw]) => {
        const g = (
          <g key={label} transform={`translate(${x} ${y})`}>
            {sw}
            <text x={18} y={9} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{label}</text>
          </g>
        );
        x += 18 + label.length * 5.3 + 22;
        return g;
      })}
    </g>
  );
}

interface StageProps {
  run: LookbackRun;
  /** Time shown, in rounds. */
  t: number;
  headline?: ReactNode;
  lines?: ReactNode[];
  /** Draw the chain of scanned sums (dependency links) up to time t. */
  chain?: boolean;
  /** Highlight the message passing between block 0 and block 1 (part 2). */
  message?: number;
  banner?: { text: string; tone: 'fail' | 'ok' } | null;
}

function Stage({ run, t, headline, lines = [], chain, message = 0, banner }: StageProps) {
  const rs = rows(run);
  const flags = flagsAt(run, Math.floor(t));
  const counter = run.config.index === 'counter';
  const resident = run.blocks.filter((b) => b.start != null && b.start <= t && (b.end == null || b.end > t)).length;
  const rowY = (hw: number) => TOP + rs.findIndex((b) => b.hw === hw) * RH;
  const rowOfIdx = (idx: number) => rs.find((b) => b.idx === idx);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Blocks of a single-pass scan waiting for each other', 'Các khối của một scan một lượt chờ lẫn nhau')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <Defs />
      <text x={LX} y={16} fontSize="12" fontWeight="600">
        <Translated
          vi={
            <>
              {run.config.G} khối, {run.config.R} chỗ; chỉ số {counter ? 'giành từ một bộ đếm' : 'lấy từ blockIdx.x'}; lookback {run.config.scheme === 'decoupled' ? 'tách rời (decoupled)' : 'đơn'}
            </>
          }
        >
          {run.config.G} blocks, {run.config.R} slots; index {counter ? 'claimed from a counter' : 'from blockIdx.x'}; {run.config.scheme === 'decoupled' ? 'decoupled' : 'single'} lookback
        </Translated>
      </text>
      <text x={TX} y={34} fontSize="10.5" style={{ fill: 'var(--ink-2)' }} data-resident={resident}>
        <Translated vi={<>vòng {Math.floor(t)} · thường trú {resident} trên {run.config.R}</>}>
          round {Math.floor(t)} · resident {resident} of {run.config.R}
        </Translated>
      </text>
      <text x={FX} y={34} fontSize="10.5" fontWeight="600" style={{ fill: 'var(--global)' }}><Translated vi="cờ (bộ nhớ toàn cục)">flags (global memory)</Translated></text>
      {[0, 6, 12, 18, 24].map((r) => (
        <g key={r}>
          <line x1={X(r)} x2={X(r)} y1={TOP - 4} y2={TOP + rs.length * RH - 4} stroke="var(--rule-soft)" />
          <text x={X(r)} y={TOP + rs.length * RH + 8} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{r}</text>
        </g>
      ))}
      {rs.map((b, k) => {
        const y = TOP + k * RH;
        const now = phaseAt(b, Math.floor(t));
        return (
          <g key={b.hw} data-row={b.hw} data-idx={b.idx ?? ''} data-phase={now}>
            <text x={LX} y={y + 15} fontSize="11">
              {!counter ? tr(`block ${b.hw}${b.start == null || b.start > t ? ' (queued)' : ''}`, `khối ${b.hw}${b.start == null || b.start > t ? ' (đang chờ)' : ''}`) : b.start != null && b.start <= t ? tr(`index ${b.idx} · blockIdx ${b.hw}`, `chỉ số ${b.idx} · blockIdx ${b.hw}`) : tr(`blockIdx ${b.hw} (queued)`, `blockIdx ${b.hw} (đang chờ)`)}
            </text>
            {b.start == null || b.start > t ? (
              <rect x={TX} y={y + 4} width={TW} height={RH - 10} rx="2" fill="none" stroke="var(--rule)" strokeDasharray="3 3" />
            ) : null}
            {b.spans.filter((s) => s.from < t).map((s, i) => {
              const x0 = X(s.from);
              const x1 = X(Math.min(s.to, t));
              return s.phase === 'starved' ? (
                <rect key={i} x={x0} y={y + 4} width={Math.max(0.5, x1 - x0)} height={RH - 10} fill="none" stroke="var(--fail)" strokeDasharray="3 2" />
              ) : (
                <rect key={i} x={x0} y={y + 4} width={Math.max(0.5, x1 - x0)} height={RH - 10}
                  fill={FILL[s.phase]} opacity={s.phase === 'local' ? 0.6 : 1} />
              );
            })}
            {b.end != null && b.end <= t ? (
              <text x={X(b.end) + 4} y={y + 15} fontSize="10.5" style={{ fill: 'var(--ok)' }}><Translated vi="xong">done</Translated></text>
            ) : null}
          </g>
        );
      })}
      {/* the flags, one per index */}
      {flags.map((f: Flag, i) => {
        const y = TOP + i * RH;
        return (
          <g key={i} data-flag={i} data-state={f}>
            <rect x={FX} y={y + 4} width={18} height={RH - 10} rx="2" fill={f === 'inclusive' ? 'var(--global)' : f === 'own' ? 'var(--global-wash)' : 'var(--cell)'} stroke="var(--global)" />
            <text x={FX + 24} y={y + 15} fontSize="10" style={{ fill: 'var(--ink-2)' }}>
              {i}: {f === 'inclusive' ? tr('prefix ready', 'tiền tố sẵn sàng') : f === 'own' ? tr('own total', 'tổng riêng') : '—'}
            </text>
          </g>
        );
      })}
      {/* chain links: from the block whose scanned sum was used to the block that used it */}
      {chain
        ? rs.flatMap((b) => {
            if (b.dep == null || b.idx == null) return [];
            const pub = run.events.find((e) => e.kind === 'inclusive' && e.idx === b.dep);
            const mine = run.events.find((e) => e.kind === 'inclusive' && e.idx === b.idx);
            const from = rowOfIdx(b.dep);
            if (!pub || !mine || !from || mine.t > t) return [];
            return [
              <line key={'c' + b.hw} data-link={`${b.dep}-${b.idx}`} x1={X(pub.t)} y1={rowY(from.hw) + RH - 6} x2={X(mine.t - 0.5)} y2={rowY(b.hw) + 4}
                stroke="var(--global)" strokeWidth="1.8" markerEnd="none" />,
            ];
          })
        : null}
      {message > 0 ? (
        <g opacity={message}>
          {(() => {
            const b0 = rowOfIdx(0)!;
            const b1 = rowOfIdx(1)!;
            const pub = run.events.find((e) => e.kind === 'inclusive' && e.idx === 0)!;
            return (
              <g data-message="0-1">
                <line x1={X(pub.t)} y1={rowY(b0.hw) + RH / 2} x2={FX - 4} y2={TOP + RH / 2} stroke="var(--global)" strokeDasharray="3 2" />
                <line x1={FX - 4} y1={TOP + RH / 2} x2={X(pub.t + 0.5)} y2={rowY(b1.hw) + RH / 2} stroke="var(--global)" strokeWidth="1.8" />
                <text x={X(pub.t) + 6} y={rowY(b0.hw) - 1} fontSize="10"><Translated vi="ghi tổng, rồi giải phóng cờ (release)">store sum, then release the flag</Translated></text>
                <text x={X(pub.t + 1) + 6} y={rowY(b1.hw) + RH + 6} fontSize="10"><Translated vi="nhận cờ (acquire), rồi đọc tổng">acquire the flag, then read the sum</Translated></text>
              </g>
            );
          })()}
        </g>
      ) : null}
      <line x1={X(t)} x2={X(t)} y1={TOP - 8} y2={TOP + rs.length * RH - 2} stroke="var(--ink)" strokeWidth="1.2" />
      <Legend y={TOP + rs.length * RH + 16} />
      {banner ? (
        <g data-banner={banner.tone}>
          <rect x={TX} y={TOP + rs.length * RH + 34} width={TW} height={24} rx="3" fill={banner.tone === 'fail' ? 'var(--fail)' : 'var(--ok)'} />
          <text x={TX + TW / 2} y={TOP + rs.length * RH + 50} fontSize="12" textAnchor="middle" fontWeight="600" style={{ fill: 'var(--page)' }}>{banner.text}</text>
        </g>
      ) : null}
      {headline ? <text x={LX} y={H - 40} fontSize="13" fontWeight="600" data-headline>{headline}</text> : null}
      {lines.map((l, i) => (
        <text key={i} x={LX} y={H - 22 + i * 15} fontSize="11" style={{ fill: 'var(--ink-2)' }} data-line={i}>{l}</text>
      ))}
    </svg>
  );
}

const S = SCENE.single;
const firstPublish = S.events.find((e) => e.kind === 'inclusive' && e.idx === 0)!.t;
const b1 = S.blocks.find((b) => b.idx === 1)!;
const b1Read = b1.spans.find((s) => s.phase === 'look')!.from;

const beats: Beat[] = [
  {
    clip: 'lookback-1',
    get label() {
      return tr('Six blocks, three slots', 'Sáu khối, ba chỗ');
    },
    draw: (p) => (
      <Stage run={S} t={ramp(p, 0.05, 0.95) * (firstPublish - 2)}
        headline={tr(`${SCENE_G} blocks, ${SCENE_R} slots: the first ${SCENE_R} start, the rest queue`, `${SCENE_G} khối, ${SCENE_R} chỗ: ${SCENE_R} khối đầu bắt đầu, số còn lại xếp hàng chờ`)}
        lines={[tr('Each block scans its segment; then it needs the total of every segment before it.', 'Mỗi khối scan đoạn của mình; sau đó nó cần tổng của mọi đoạn đứng trước nó.')]} />
    ),
  },
  {
    clip: 'lookback-2',
    get label() {
      return tr('One flag', 'Một cờ');
    },
    draw: (p) => {
      const t = firstPublish - 2 + ramp(p, 0.05, 0.85) * (b1Read + 2 - (firstPublish - 2));
      return (
        <Stage run={S} t={t} message={t >= firstPublish ? 1 : 0}
          headline={tr("indices 0, 1, 2 claimed in start order; index 1 spins on index 0's flag", 'các chỉ số 0, 1, 2 được giành theo thứ tự bắt đầu; chỉ số 1 quay chờ trên cờ của chỉ số 0')}
          lines={[tr(`index 1 spins ${b1.wait} round${b1.wait === 1 ? '' : 's'}; it reads the sum only after the flag is set`, `chỉ số 1 quay chờ ${b1.wait} vòng; nó chỉ đọc tổng sau khi cờ được đặt`)]} />
      );
    },
  },
  {
    clip: 'lookback-3',
    get label() {
      return tr('The chain', 'Chuỗi');
    },
    draw: (p) => {
      const t = b1Read + 2 + ramp(p, 0.02, 0.7) * (S.time - b1Read - 2);
      return (
        <Stage run={S} t={t} chain
          headline={tr(`a chain of ${S.criticalPath} links for ${SCENE_G} blocks`, `một chuỗi ${S.criticalPath} mắt xích cho ${SCENE_G} khối`)}
          lines={[p > 0.72 ? tr(`with ${SCENE_WIDE.config.G} blocks all resident: ${SCENE_WIDE.criticalPath} links, finished after ${SCENE_WIDE.time} rounds`, `với ${SCENE_WIDE.config.G} khối đều thường trú: ${SCENE_WIDE.criticalPath} mắt xích, xong sau ${SCENE_WIDE.time} vòng`) : tr(`finished after ${S.time} rounds; ${S.waitTotal} rounds spent spinning in all`, `xong sau ${S.time} vòng; tổng cộng ${S.waitTotal} vòng quay chờ`)]} />
      );
    },
  },
  {
    clip: 'lookback-4',
    get label() {
      return tr('Without the counter', 'Không có bộ đếm');
    },
    draw: (p) => {
      const D = SCENE.deadlock;
      return (
        <Stage run={D} t={ramp(p, 0.05, 0.75) * (D.time + 1)}
          banner={p > 0.78 ? { text: tr('deadlock: every resident block spins, no slot can free', 'deadlock: mọi khối thường trú đều quay chờ, không chỗ nào được giải phóng'), tone: 'fail' } : null}
          headline={tr(`the scheduler starts blocks ${D.dispatch.slice(0, SCENE_R).join(', ')} first`, `bộ lập lịch khởi động các khối ${D.dispatch.slice(0, SCENE_R).join(', ')} trước`)}
          lines={[D.stuck.map((s) => tr(`${s.idx} waits on ${s.on}${s.onStarted ? '' : ' (not started)'}`, `${s.idx} chờ ${s.on}${s.onStarted ? '' : ' (chưa bắt đầu)'}`)).join(' · ')]} />
      );
    },
  },
  {
    clip: 'lookback-5',
    get label() {
      return tr('Decoupled', 'Decoupled');
    },
    draw: (p) => {
      const D = SCENE.decoupled;
      return (
        <Stage run={D} t={ramp(p, 0.03, 0.75) * D.time} chain
          headline={tr(`spinning: ${S.waitTotal} rounds single, ${D.waitTotal} decoupled; ${D.repeated} extra additions`, `quay chờ: ${S.waitTotal} vòng với lookback đơn, ${D.waitTotal} với decoupled; thêm ${D.repeated} phép cộng`)}
          lines={[tr(`chain ${S.criticalPath} links single, ${D.criticalPath} decoupled; finished after ${S.time} rounds single, ${D.time} decoupled`, `chuỗi ${S.criticalPath} mắt xích với lookback đơn, ${D.criticalPath} với decoupled; xong sau ${S.time} vòng với lookback đơn, ${D.time} với decoupled`)]} />
      );
    },
  },
  {
    clip: 'lookback-6',
    get label() {
      return tr('Who promises progress', 'Ai bảo đảm tiến triển');
    },
    draw: (p) => {
      const M = SCENE.metal;
      return (
        <Stage run={M} t={ramp(p, 0.05, 0.6) * (M.time + 1)}
          banner={p > 0.62 ? { text: tr('no progress guarantee: the waiter runs, the block it waits for does not', 'không bảo đảm tiến triển: khối chờ thì chạy, khối nó chờ thì không'), tone: 'fail' } : null}
          headline={tr('CUDA documents progress for a started block; AMD and Metal do not', 'CUDA có tài liệu bảo đảm tiến triển cho khối đã bắt đầu; AMD và Metal thì không')}
          lines={[tr('CUDA: claimed index is safe · HIP: unverified · Metal: unsafe, use separate dispatches', 'CUDA: chỉ số giành từ bộ đếm là an toàn · HIP: unverified · Metal: không an toàn, hãy dùng các dispatch riêng')]} />
      );
    },
  },
];

export function LookbackScene() {
  return <Scene chapter="ch11" title={tr('Waiting for the block before you', 'Chờ khối đứng trước bạn')} beats={beats} />;
}
