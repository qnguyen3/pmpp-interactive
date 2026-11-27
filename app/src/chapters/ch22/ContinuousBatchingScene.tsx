import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { M8_DEFAULT, precisionFor, relativeStep, stepBStar } from '../../sim/batching';
import {
  SCENE_NAMES,
  SCENE_REQUESTS,
  SCENE_SHORT_MAX,
  SCENE_SLOTS,
  makeRequests,
  serve,
  slotsAt,
  waitingAt,
  type ServeResult,
} from '../../sim/serving';

// "Static and continuous batching" (section 22.5). Frames come from src/sim/serving.ts:
// six requests, four slots, each policy run once. Step times come from src/sim/batching.ts
// (lab 22.2's step, float32, the book's example threshold of 20 FLOP/B) in relative units,
// one unit being a step at batch 1, so no device timing appears. Five beats, one narration
// clip each (content/ch22/narration/continuous-batching-N.txt).

const PREC = precisionFor(4);
const T = 20;
const base = relativeStep(M8_DEFAULT, 1, PREC, 1, T).step;
const steps: number[] = [];
/** Step time with `a` requests generating, in batch-1 steps. */
export const sceneStepTime = (a: number) => (steps[a] ??= relativeStep(M8_DEFAULT, a, PREC, 1, T).step / base);

export const STATIC = serve({ requests: SCENE_REQUESTS, slots: SCENE_SLOTS, policy: 'static', stepTime: sceneStepTime, shortMax: SCENE_SHORT_MAX });
export const CONTINUOUS = serve({ requests: SCENE_REQUESTS, slots: SCENE_SLOTS, policy: 'continuous', stepTime: sceneStepTime, shortMax: SCENE_SHORT_MAX });

/** Beat 5: the slot limit under heavy load (every request waiting from the start), continuous batching. */
export const LIMITS = [1, 2, 4, 8, 16, 32, 64, 128, 256];
const HEAVY = makeRequests({ count: 600, rate: 1e6, mix: 'short', seed: 1 });
export const KNOB = LIMITS.map((slots) => {
  const r = serve({ requests: HEAVY, slots, policy: 'continuous', stepTime: sceneStepTime });
  return { slots, tokens: r.tokensPerSecond, perToken: r.meanTimePerToken };
});
export const KNOB_BSTAR = stepBStar(M8_DEFAULT, PREC, { peak: T, bw: 1 })!;

const W = 720;
const H = 340;
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const f2 = (v: number) => (Math.round(v * 100) / 100).toString();

function Frame({ children, headline, policy }: { children: ReactNode; headline: string; policy?: string }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Slots of a batch over decode steps', 'Các khe của một batch qua các bước decode')} style={{ width: '100%', maxWidth: W, display: 'block' }}
      data-policy={policy ?? ''}>
      <text x={12} y={24} fontSize="15" fontWeight="600" data-headline="">{headline}</text>
      {children}
    </svg>
  );
}

/**
 * The slot-by-step chart of a run, columns 0..upTo-1 drawn. A generating request shows its
 * letter; a finished one still holding its slot (static) is grey; a short request that has
 * been returned to its user gets a tick on the last cell it occupied.
 */
function SlotChart({ run, upTo, x, y, cw, rh, cols, title }: {
  run: ServeResult; upTo: number; x: number; y: number; cw: number; rh: number; cols: number; title?: string;
}) {
  const shown = Math.min(upTo, run.steps.length);
  return (
    <g data-chart={run.policy} data-shown={shown}>
      {title ? <text x={x} y={y - 22} fontSize="12" fontWeight="600">{title}</text> : null}
      {Array.from({ length: cols }, (_, k) => (
        <text key={k} x={x + k * cw + cw / 2} y={y - 6} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{k}</text>
      ))}
      {Array.from({ length: run.slots }, (_, s) => (
        <g key={s}>
          <text x={x - 8} y={y + s * rh + rh / 2 + 4} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>khe {s + 1}</>}>slot {s + 1}</Translated></text>
          {Array.from({ length: cols }, (_, k) => (
            <rect key={k} x={x + k * cw} y={y + s * rh} width={cw - 2} height={rh - 3} fill="none" stroke="var(--rule-soft)" />
          ))}
        </g>
      ))}
      {Array.from({ length: shown }, (_, k) =>
        slotsAt(run, k).map((c, s) =>
          c ? (
            <g key={`${k}-${s}`} data-cell={`${s},${k}`} data-id={SCENE_NAMES[c.id]} data-state={c.state}>
              <rect x={x + k * cw} y={y + s * rh} width={cw - 2} height={rh - 3} rx={2}
                fill={c.state === 'gen' ? 'var(--cell)' : 'var(--rule)'} stroke={c.state === 'gen' ? 'var(--ink-2)' : 'none'} />
              <text x={x + k * cw + cw / 2 - 1} y={y + s * rh + rh / 2 + 2} fontSize={Math.min(12, rh * 0.45)} textAnchor="middle"
                style={{ fill: c.state === 'gen' ? 'var(--ink)' : 'var(--ink-3)' }}>
                {c.state === 'gen' ? SCENE_NAMES[c.id] : tr('idle', 'rảnh')}
              </text>
            </g>
          ) : null,
        ),
      )}
      {run.served
        .filter((r) => r.length <= SCENE_SHORT_MAX && r.returnStep <= shown)
        .map((r) => (
          <g key={r.id} data-returned={SCENE_NAMES[r.id]}>
            <rect x={x + (r.returnStep - 1) * cw} y={y + r.slot * rh} width={cw - 2} height={rh - 3} rx={2} fill="none" stroke="var(--ok)" strokeWidth={2} />
            <text x={x + r.returnStep * cw - 5} y={y + r.slot * rh + Math.min(11, rh * 0.5)} fontSize={Math.min(11, rh * 0.5)} textAnchor="end" style={{ fill: 'var(--ok)' }}>✓</text>
          </g>
        ))}
    </g>
  );
}

function Queue({ run, k, x, y }: { run: ServeResult; k: number; x: number; y: number }) {
  const waiting = k < run.steps.length ? waitingAt(run, k) : [];
  return (
    <g data-queue={waiting.map((i) => SCENE_NAMES[i]).join(' ')}>
      <text x={x} y={y} fontSize="11.5"><Translated vi="đang chờ:">waiting:</Translated></text>
      {waiting.map((i, j) => (
        <g key={i}>
          <rect x={x + 58 + j * 30} y={y - 14} width={24} height={20} rx={2} fill="var(--cell)" stroke="var(--ink-2)" />
          <text x={x + 70 + j * 30} y={y} fontSize="11.5" textAnchor="middle">{SCENE_NAMES[i]}</text>
        </g>
      ))}
      {!waiting.length ? <text x={x + 58} y={y} fontSize="11.5" style={{ fill: 'var(--ink-3)' }}><Translated vi="không ai">nobody</Translated></text> : null}
    </g>
  );
}

function Counters({ run, k, x, y }: { run: ServeResult; k: number; x: number; y: number }) {
  const st = run.steps[Math.min(k, run.steps.length - 1)];
  const idle = run.slots - st.active;
  return (
    <g data-tokens-step={st.active} data-idle={idle}>
      <text x={x} y={y} fontSize="12">
        <Translated vi={<>bước {Math.min(k, run.steps.length - 1)}: tạo {st.active} token, {idle} khe rảnh</>}>
          step {Math.min(k, run.steps.length - 1)}: {st.active} {st.active === 1 ? 'token' : 'tokens'} made, {idle} {idle === 1 ? 'slot' : 'slots'} idle
        </Translated>
      </text>
    </g>
  );
}

const CW = 40;
const CX = 90;
const CY = 74;
const RH = 34;

function Policy({ run, p, label }: { run: ServeResult; p: number; label: string }) {
  const k = Math.min(run.steps.length - 1, Math.floor(ramp(p, 0.04, 0.9) * run.steps.length));
  return (
    <Frame headline={label} policy={run.policy}>
      <SlotChart run={run} upTo={k + 1} x={CX} y={CY} cw={CW} rh={RH} cols={12} />
      <Queue run={run} k={k} x={12} y={242} />
      <g>
        <rect x={430} y={230} width={16} height={14} rx={2} fill="none" stroke="var(--ok)" strokeWidth={2} />
        <text x={452} y={241} fontSize="11"><Translated vi="một yêu cầu ngắn đã trả về cho người dùng">a short request returned to its user</Translated></text>
      </g>
      <Counters run={run} k={k} x={12} y={276} />
      <text x={12} y={308} fontSize="12" data-steps-total={k + 1 >= run.steps.length ? run.steps.length : ''}>
        {k + 1 >= run.steps.length ? tr(`all 6 requests done after ${run.steps.length} steps, ${run.tokens} tokens`, `cả 6 yêu cầu xong sau ${run.steps.length} bước, ${run.tokens} token`) : ''}
      </text>
    </Frame>
  );
}

function knobPlot(p: number) {
  const X0 = 80, Y0 = 70, PW = 250, PH = 180;
  const n = Math.max(1, Math.round(ramp(p, 0.05, 0.75) * LIMITS.length));
  const X = (s: number) => X0 + (Math.log2(s) / 8) * PW;
  const plot = (dx: number, vals: number[], title: string, key: string) => {
    const lo = Math.min(...vals) / 1.3;
    const hi = Math.max(...vals) * 1.3;
    const Y = (v: number) => Y0 + PH - ((Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * PH;
    return (
      <g data-knob={key} data-n={n}>
        <text x={X0 + dx} y={Y0 - 10} fontSize="12" fontWeight="600">{title}</text>
        <rect x={X0 + dx} y={Y0} width={PW} height={PH} fill="none" stroke="var(--rule)" />
        {LIMITS.map((s) => (
          <text key={s} x={X(s) + dx} y={Y0 + PH + 14} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{s}</text>
        ))}
        <text x={X0 + dx + PW / 2} y={Y0 + PH + 30} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}><Translated vi="giới hạn khe">slot limit</Translated></text>
        <line x1={X(KNOB_BSTAR) + dx} x2={X(KNOB_BSTAR) + dx} y1={Y0} y2={Y0 + PH} stroke="var(--ink)" strokeDasharray="3 3" />
        <text x={X(KNOB_BSTAR) + dx + 4} y={Y0 + PH - 8} fontSize="10.5"><Translated vi="điểm gãy">knee</Translated></text>
        <polyline points={vals.slice(0, n).map((v, i) => `${X(LIMITS[i]) + dx},${Y(v)}`).join(' ')} fill="none" stroke="var(--ink)" strokeWidth="2" />
        {vals.slice(0, n).map((v, i) => <circle key={i} cx={X(LIMITS[i]) + dx} cy={Y(v)} r={3} fill="var(--ink)" />)}
      </g>
    );
  };
  return (
    <>
      {plot(0, KNOB.map((k) => k.tokens), tr('tokens per batch-1 step', 'token mỗi bước ở batch 1'), 'tokens')}
      {plot(350, KNOB.map((k) => k.perToken), tr("each user's time per token", 'thời gian mỗi token của từng người dùng'), 'perToken')}
    </>
  );
}

const shortFinish = (r: ServeResult) => r.meanFinishShort ?? NaN;

const beats: Beat[] = [
  {
    clip: 'continuous-batching-1',
    get label() { return tr('Six requests', 'Sáu yêu cầu'); },
    draw: (p) => (
      <Frame headline={tr('6 requests, each wanting its own number of tokens', '6 yêu cầu, mỗi yêu cầu cần một số token riêng')}>
        {SCENE_REQUESTS.map((r, i) => (
          <g key={r.id} opacity={0.3 + 0.7 * ramp(p, 0.02 + i * 0.07, 0.08 + i * 0.07)} data-request={SCENE_NAMES[r.id]} data-length={r.length}>
            <text x={12} y={70 + i * 24} fontSize="12"><Translated vi={<>{SCENE_NAMES[r.id]}: đến ở bước {r.arrival}</>}>{SCENE_NAMES[r.id]}: arrives at step {r.arrival}</Translated></text>
            {Array.from({ length: r.length }, (_, t) => (
              <rect key={t} x={170 + t * 18} y={58 + i * 24} width={14} height={14} rx={2} fill="var(--cell)" stroke="var(--ink-2)" />
            ))}
            <text x={170 + r.length * 18 + 4} y={70 + i * 24} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>{r.length} token</>}>{r.length} tokens</Translated></text>
          </g>
        ))}
        <g>
          <SlotChart run={CONTINUOUS} upTo={p > 0.7 ? 1 : 0} x={CX} y={240} cw={CW} rh={20} cols={12} />
          <text x={CX + 12 * CW + 8} y={262} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi="mỗi cột một bước">one column per step</Translated></text>
        </g>
      </Frame>
    ),
  },
  {
    clip: 'continuous-batching-2',
    get label() { return tr('Static, 4 slots', 'Tĩnh, 4 khe'); },
    draw: (p) => <Policy run={STATIC} p={p} label={tr('static batching, 4 slots: the batch ends when its longest request does', 'batching tĩnh, 4 khe: batch kết thúc khi yêu cầu dài nhất của nó kết thúc')} />,
  },
  {
    clip: 'continuous-batching-3',
    get label() { return tr('Continuous', 'Liên tục'); },
    draw: (p) => <Policy run={CONTINUOUS} p={p} label={tr('continuous batching: leave and join at every step', 'batching liên tục: rời đi và gia nhập ở mỗi bước')} />,
  },
  {
    clip: 'continuous-batching-4',
    get label() { return tr('The tally', 'Bảng tổng kết'); },
    draw: (p) => (
      <Frame headline={tr('the same 6 requests, the same 4 slots', 'cùng 6 yêu cầu, cùng 4 khe')}>
        <SlotChart run={STATIC} upTo={99} x={CX} y={62} cw={26} rh={20} cols={12} />
        <text x={CX + 12 * 26 + 10} y={90} fontSize="12" fontWeight="600"><Translated vi="tĩnh">static</Translated></text>
        <SlotChart run={CONTINUOUS} upTo={99} x={CX} y={176} cw={26} rh={20} cols={12} />
        <text x={CX + 12 * 26 + 10} y={204} fontSize="12" fontWeight="600"><Translated vi="liên tục">continuous</Translated></text>
        <g opacity={ramp(p, 0.15, 0.3)} data-tally=""
          data-static-steps={STATIC.steps.length} data-continuous-steps={CONTINUOUS.steps.length}
          data-static-tps={f2(STATIC.tokensPerSecond)} data-continuous-tps={f2(CONTINUOUS.tokensPerSecond)}
          data-static-short={f2(shortFinish(STATIC))} data-continuous-short={f2(shortFinish(CONTINUOUS))}>
          <text x={500} y={74} fontSize="11.5"><Translated vi={<>{STATIC.steps.length} bước, {f2(STATIC.tokensPerSecond)} token mỗi đơn vị</>}>{STATIC.steps.length} steps, {f2(STATIC.tokensPerSecond)} tokens per unit</Translated></text>
          <text x={500} y={90} fontSize="11.5"><Translated vi={<>yêu cầu ngắn trả về sau trung bình {f2(shortFinish(STATIC))}</>}>short ones back after {f2(shortFinish(STATIC))} on average</Translated></text>
          <text x={500} y={188} fontSize="11.5"><Translated vi={<>{CONTINUOUS.steps.length} bước, {f2(CONTINUOUS.tokensPerSecond)} token mỗi đơn vị</>}>{CONTINUOUS.steps.length} steps, {f2(CONTINUOUS.tokensPerSecond)} tokens per unit</Translated></text>
          <text x={500} y={204} fontSize="11.5"><Translated vi={<>yêu cầu ngắn trả về sau trung bình {f2(shortFinish(CONTINUOUS))}</>}>short ones back after {f2(shortFinish(CONTINUOUS))} on average</Translated></text>
        </g>
        <text x={12} y={300} fontSize="12" opacity={ramp(p, 0.55, 0.7)}>
          <Translated vi={<>thời gian mỗi token của từng người dùng: một bước ở batch mà server đang chạy (ở đây là {f2(CONTINUOUS.meanTimePerToken)})</>}>
            time per token for each user: one step at the batch the server is running ({f2(CONTINUOUS.meanTimePerToken)} here)
          </Translated>
        </text>
        <text x={12} y={322} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
          <Translated vi="đơn vị: một bước decode ở batch 1, theo mô hình batching tại ngưỡng trong ví dụ của sách">
            unit: one decode step at batch 1, from the batching model at the book's example threshold
          </Translated>
        </text>
      </Frame>
    ),
  },
  {
    clip: 'continuous-batching-5',
    get label() { return tr('The knob', 'Núm vặn'); },
    draw: (p) => (
      <Frame headline={tr("the slot limit sets the batch: throughput against each user's wait", 'giới hạn khe quyết định batch: thông lượng đổi lấy thời gian chờ của từng người dùng')}>
        {knobPlot(p)}
        <text x={12} y={318} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
          <Translated vi={<>batching liên tục, 600 yêu cầu cùng chờ, thời gian bước theo mô hình batching (điểm gãy tại B* = {KNOB_BSTAR.toFixed(1)})</>}>
            continuous batching, 600 requests all waiting, step times from the batching model (knee at B* = {KNOB_BSTAR.toFixed(1)})
          </Translated>
        </text>
      </Frame>
    ),
  },
];

export function ContinuousBatchingScene() {
  return <Scene chapter="ch22" title={tr('Static and continuous batching', 'Batching tĩnh và liên tục')} beats={beats} />;
}
