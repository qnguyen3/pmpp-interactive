import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import {
  FALLBACK_WEIGHTS,
  GE_GT,
  SCENE_CFG,
  SCENE_U,
  SCENE_WEIGHTS,
  TOP_U,
  cdfIn,
  fallbackDraw,
  inverseCdf,
  sceneBoundary,
  sceneDraw,
} from '../../sim/draw';
import { f32short } from '../../sim/philox';
import { Translated, tr } from '../../lib/i18n';

// "Finding u in the scan" (section X1.8): one block of 8 threads (display warps of 4, 2
// survivors per thread, chunks of 16) draws from 40 survivors at T = 3 with u = 0.8. Six
// beats, one narration clip each (content/x1/narration/find-u-N.txt). Every sum, flag, ballot
// and token drawn comes from src/sim/draw.ts (drawKernel, sceneBoundary, fallbackDraw),
// pinned in draw.test.ts.
const K = sceneDraw();
const N = SCENE_WEIGHTS.length;
const CHUNK = SCENE_CFG.threads * SCENE_CFG.vec;
const BND = sceneBoundary()!;
const FB = fallbackDraw();

const W = 720;
const H = 340;
const X0 = 20;
const BW = 680;
const MONO = { fontVariationSettings: "'MONO' 1" } as const; // i18n-ignore: CSS font variation value
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const short = (v: number) => f32short(v);

/** Bar geometry in double precision (drawing only); the numbers printed come from the kernel trace. */
function geometry(w: number[]) {
  const total = w.reduce((s, x) => s + x, 0);
  const edges: number[] = [0];
  for (const x of w) edges.push(edges[edges.length - 1] + x);
  return { total, xs: edges.map((e) => X0 + (e / total) * BW) };
}
const G = geometry(SCENE_WEIGHTS);

function Frame({ beat, headline, children }: { beat: number; headline: string; children: ReactNode }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={headline} data-beat={beat} style={{ display: 'block' }}>
      <text x={14} y={22} fontSize="13" fontWeight="600" data-headline="" style={{ fill: 'var(--ink)' }}>{headline}</text>
      {children}
    </svg>
  );
}

/** The row as one stacked bar; `state(i)` picks each survivor's look. */
function Bar({ y, state, target = true, chunks = false }: { y: number; state?: (i: number) => 'win' | 'skip' | 'scan' | 'plain'; target?: boolean; chunks?: boolean }) {
  const tx = X0 + (K.target / K.total) * BW;
  return (
    <g>
      {SCENE_WEIGHTS.map((_, i) => {
        const st = state ? state(i) : 'plain';
        return (
          <rect key={i} data-survivor={i} data-state={st} x={G.xs[i]} y={y} width={Math.max(0.5, G.xs[i + 1] - G.xs[i])} height={26}
            fill={st === 'win' ? 'var(--ink)' : st === 'skip' ? 'var(--rule)' : i % 2 ? 'var(--global-wash)' : 'var(--cell)'}
            stroke={st === 'win' ? 'var(--ink)' : 'var(--global)'} strokeWidth={0.6} opacity={st === 'skip' ? 0.6 : 1} />
        );
      })}
      {[0, 1, 2, 5].map((i) => (
        <text key={i} x={(G.xs[i] + G.xs[i + 1]) / 2} y={y + 17} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink)' }}>{i}</text>
      ))}
      {chunks
        ? Array.from({ length: Math.ceil(N / CHUNK) }, (_, c) => {
            const a = G.xs[c * CHUNK];
            const b = G.xs[Math.min(N, (c + 1) * CHUNK)];
            return (
              <g key={c}>
                <line x1={a} x2={a} y1={y - 6} y2={y + 34} stroke="var(--ink)" strokeWidth={1.4} />
                <text x={b - a < 120 ? b : a + 3} textAnchor={b - a < 120 ? 'end' : 'start'} y={y + 46 + (c % 2) * 14} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
                  <Translated vi={<>đoạn {c}: #{c * CHUNK} đến #{Math.min(N, (c + 1) * CHUNK) - 1}</>}>
                    chunk {c}: #{c * CHUNK} to #{Math.min(N, (c + 1) * CHUNK) - 1}
                  </Translated>
                </text>
              </g>
            );
          })
        : null}
      {target ? (
        <g data-target={K.target}>
          <line x1={tx} x2={tx} y1={y - 14} y2={y + 32} stroke="var(--ink)" strokeWidth={2} strokeDasharray="4 2" />
          <text x={tx} y={y - 18} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink)' }}><Translated vi={<>đích = u × tổng = {short(K.target)}</>}>target = u × total = {short(K.target)}</Translated></text>
        </g>
      ) : null}
    </g>
  );
}

const crossing = K.chunks.find((c) => c.crosses)!;

function LaneGrid({ y, upto, flags }: { y: number; upto: number; flags: boolean }) {
  return (
    <g data-part="lanes">
      <text x={14} y={y - 8} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi={<>đoạn {crossing.c}: tổng gộp trước nó {short(crossing.aggBefore)}; mỗi lane: phần mang sang + các tổng bao gồm của riêng nó</>}>
          chunk {crossing.c}: aggregate before it {short(crossing.aggBefore)}; each lane: carry + its own inclusive sums
        </Translated>
      </text>
      {crossing.lanes.map((l) => {
        const x = 14 + l.thread * 87;
        const on = l.thread < upto;
        return (
          <g key={l.thread} opacity={on ? 1 : 0.25} data-lane={l.thread} data-warp={l.warp}>
            <text x={x + 40} y={y + 10} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
              <Translated vi={<>warp {l.warp}, lane {l.lane}</>}>warp {l.warp}, lane {l.lane}</Translated>
            </text>
            {l.cum.map((c, j) => {
              const i = l.first + j;
              const f = flags && l.flags[j];
              return (
                <g key={j} data-index={i} data-cum={c} data-flag={flags ? (l.flags[j] ? 1 : 0) : ''}>
                  <rect x={x} y={y + 16 + j * 34} width={80} height={30} rx="3" fill={f ? 'var(--ok-wash)' : 'var(--cell)'} stroke={f ? 'var(--ok)' : 'var(--rule)'} />
                  <text x={x + 5} y={y + 28 + j * 34} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>#{i}{flags ? (l.flags[j] ? tr('  > target', '  > đích') : tr('  ≤ target', '  ≤ đích')) : ''}</text>
                  <text x={x + 5} y={y + 41 + j * 34} fontSize="11" style={{ ...MONO, fill: 'var(--ink)' }}>{short(c)}</text>
                </g>
              );
            })}
          </g>
        );
      })}
    </g>
  );
}

const ballotBits = (b: number, width: number) =>
  Array.from({ length: width }, (_, l) => ((b >>> l) & 1 ? '1' : '0')).reverse().join('');

const beats: Beat[] = [
  {
    clip: 'find-u-1',
    get label() { return tr('The target', 'Đích'); },
    draw: (p) => (
      <Frame beat={1} headline={tr(`${N} survivors after temperature, total ${short(K.total)}; u = ${SCENE_U}`, `${N} token còn lại sau temperature, tổng ${short(K.total)}; u = ${SCENE_U}`)}>
        <Bar y={90} target={p > 0.35} state={(i) => (p > 0.7 && i === K.index ? 'win' : 'plain')} />
        <text x={14} y={160} fontSize="12" opacity={ramp(p, 0.35, 0.5)} style={{ fill: 'var(--ink)' }}>
          <Translated vi="độ rộng của token còn lại i là trọng số w_i = exp((l_i − l_0) / T) của nó; độ dài thanh là tổng">
            survivor i's width is its weight w_i = exp((l_i − l_0) / T); the bar's length is the total
          </Translated>
        </text>
        <text x={14} y={186} fontSize="12" opacity={ramp(p, 0.7, 0.85)} style={{ fill: 'var(--ink)' }} data-token={K.index}>
          <Translated vi={<>token được chọn: token còn lại đầu tiên có tổng tích lũy lớn hơn hẳn đích: #{K.index}</>}>
            the token: the first survivor whose running sum is strictly greater than the target: #{K.index}
          </Translated>
        </text>
      </Frame>
    ),
  },
  {
    clip: 'find-u-2',
    get label() { return tr('Chunks', 'Các đoạn'); },
    draw: (p) => {
      const shown = K.chunks.filter((_, c) => p > 0.2 + c * 0.35);
      return (
        <Frame beat={2} headline={tr(`walk the row ${CHUNK} survivors at a time; add each chunk to the aggregate`, `đi qua hàng, mỗi lần ${CHUNK} token còn lại; cộng mỗi đoạn vào tổng gộp`)}>
          <Bar y={90} chunks state={(i) => {
            const c = Math.floor(i / CHUNK);
            const ch = shown.find((x) => x.c === c);
            return ch && !ch.crosses ? 'skip' : 'plain';
          }} />
          {shown.map((ch) => (
            <text key={ch.c} x={14} y={190 + ch.c * 24} fontSize="12" data-chunk={ch.c} data-crosses={ch.crosses ? 1 : 0} data-agg={ch.aggAfter}
              style={{ ...MONO, fill: 'var(--ink)' }}>
              <Translated vi={<>đoạn {ch.c}: tổng {short(ch.sum)}, tổng gộp {short(ch.aggBefore)} → {short(ch.aggAfter)}</>}>
                chunk {ch.c}: sum {short(ch.sum)}, aggregate {short(ch.aggBefore)} → {short(ch.aggAfter)}
              </Translated>
              {ch.crosses ? tr(` > ${short(K.target)}: it crosses, scan it`, ` > ${short(K.target)}: nó vượt qua, scan nó`) : tr(` ≤ ${short(K.target)}: skip, no scan`, ` ≤ ${short(K.target)}: bỏ qua, không scan`)}
            </text>
          ))}
        </Frame>
      );
    },
  },
  {
    clip: 'find-u-3',
    get label() { return tr('Scan and flag', 'Scan và đánh cờ'); },
    draw: (p) => {
      const upto = Math.ceil(ramp(p, 0.05, 0.55) * SCENE_CFG.threads);
      return (
        <Frame beat={3} headline={tr(`scan chunk ${crossing.c}; a lane flags a survivor whose sum is greater than ${short(K.target)}`, `scan đoạn ${crossing.c}; một lane đánh cờ token còn lại có tổng lớn hơn ${short(K.target)}`)}>
          <LaneGrid y={70} upto={upto} flags={p > 0.6} />
          <text x={14} y={210} fontSize="12" opacity={ramp(p, 0.65, 0.8)} style={{ fill: 'var(--ink)' }}>
            {crossing.lanes.reduce((s, l) => s + l.flags.filter(Boolean).length, 0)}<Translated vi=" token còn lại được đánh cờ; token đầu tiên trong số đó là token được chọn"> survivors flagged; the first of them is the token</Translated>
          </text>
        </Frame>
      );
    },
  },
  {
    clip: 'find-u-4',
    get label() { return tr('The vote', 'Bỏ phiếu'); },
    draw: (p) => {
      const nW = crossing.ballots.length;
      return (
        <Frame beat={4} headline={tr('one ballot per warp; the lowest flagged lane of the lowest flagged warp writes the token', 'một ballot cho mỗi warp; lane có cờ thấp nhất của warp có cờ thấp nhất ghi token')}>
          <LaneGrid y={70} upto={SCENE_CFG.threads} flags />
          {crossing.ballots.map((b, k) => (
            <g key={k} opacity={ramp(p, 0.1 + k * 0.2, 0.25 + k * 0.2)} data-ballot={k} data-bits={b} data-first={crossing.warpFirst[k]}>
              <text x={14} y={220 + k * 22} fontSize="12" style={{ ...MONO, fill: 'var(--ink)' }}>
                <Translated vi={<>ballot của warp {k} (lane 0 ở bên phải): {ballotBits(b, SCENE_CFG.warp)} → lane có cờ thấp nhất {crossing.warpFirst[k] < 0 ? 'không có' : crossing.warpFirst[k]}</>}>
                  warp {k} ballot (lane 0 on the right): {ballotBits(b, SCENE_CFG.warp)} → lowest flagged lane {crossing.warpFirst[k] < 0 ? 'none' : crossing.warpFirst[k]}
                </Translated>
              </text>
            </g>
          ))}
          <text x={14} y={226 + nW * 22} fontSize="12.5" fontWeight="700" opacity={ramp(p, 0.65, 0.8)} data-winner={K.index} style={{ fill: 'var(--ok)' }}>
            <Translated vi={<>warp {K.winner!.warp}, lane {K.winner!.lane} (luồng {K.winner!.thread}) ghi token còn lại #{K.index}</>}>
              warp {K.winner!.warp}, lane {K.winner!.lane} (thread {K.winner!.thread}) writes survivor #{K.index}
            </Translated>
          </text>
        </Frame>
      );
    },
  },
  {
    clip: 'find-u-5',
    get label() { return tr('Near a boundary', 'Gần một ranh giới'); },
    draw: (p) => {
      const L = cdfIn(SCENE_WEIGHTS, 'f32-loop', SCENE_CFG);
      const i = BND.boundary;
      return (
        <Frame beat={5} headline={tr(`u = ${short(BND.u)}: two orders of addition, one target, neighbouring tokens`, `u = ${short(BND.u)}: hai thứ tự cộng, một đích, hai token liền kề`)}>
          <text x={14} y={70} fontSize="12" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>tổng bao gồm tại token còn lại #{i}, ranh giới giữa #{i} và #{i + 1}</>}>the inclusive sum at survivor #{i}, the boundary between #{i} and #{i + 1}</Translated></text>
          <g opacity={ramp(p, 0.1, 0.3)} data-order="loop" data-cum={BND.loop.cum} data-index={BND.loop.index}>
            <text x={14} y={104} fontSize="12.5" style={{ ...MONO, fill: 'var(--ink)' }}>
              <Translated vi={<>trái sang phải:   tổng {short(BND.loop.cum)}, đích {short(BND.loop.target)} → không lớn hơn → token #{BND.loop.index}</>}>
                left to right:    sum {short(BND.loop.cum)}, target {short(BND.loop.target)} → not greater → token #{BND.loop.index}
              </Translated>
            </text>
          </g>
          <g opacity={ramp(p, 0.35, 0.55)} data-order="kernel" data-cum={BND.kernel.cum} data-index={BND.kernel.index}>
            <text x={14} y={132} fontSize="12.5" style={{ ...MONO, fill: 'var(--ink)' }}>
              <Translated vi={<>thứ tự của kernel: tổng {short(BND.kernel.cum)}, đích {short(BND.kernel.target)} → lớn hơn → token #{BND.kernel.index}</>}>
                kernel's order:   sum {short(BND.kernel.cum)}, target {short(BND.kernel.target)} → greater → token #{BND.kernel.index}
              </Translated>
            </text>
          </g>
          <text x={14} y={176} fontSize="12" opacity={ramp(p, 0.6, 0.75)} style={{ fill: 'var(--ink)' }}>
            <Translated vi={<>hai tổng khác nhau ở bit cuối; tổng của hàng ở double là {G.total.toPrecision(9)} và tổng float32 là {short(L.total)}</>}>
              the two sums differ in the last bit; the row's total in double is {G.total.toPrecision(9)} and float32 total {short(L.total)}
            </Translated>
          </text>
          <text x={14} y={204} fontSize="12" fontWeight="600" opacity={ramp(p, 0.78, 0.92)} style={{ fill: 'var(--ink)' }}>
            <Translated vi={<>một bài kiểm tra giữa các backend chấp nhận #{i} hoặc #{i + 1} cho một lần rút sát ranh giới đến vậy</>}>
              a cross-backend test accepts either #{i} or #{i + 1} for a draw this close to the boundary
            </Translated>
          </text>
        </Frame>
      );
    },
  },
  {
    clip: 'find-u-6',
    get label() { return tr('The two ends', 'Hai đầu mút'); },
    draw: (p) => {
      const gt = inverseCdf(GE_GT.weights, GE_GT.u, { rule: 'gt' });
      const ge = inverseCdf(GE_GT.weights, GE_GT.u, { rule: 'ge' });
      const { cum } = cdfIn(FALLBACK_WEIGHTS, 'f32-kernel', SCENE_CFG);
      return (
        <Frame beat={6} headline={tr('u = 0 never picks a zero weight; nothing above the target falls back', 'u = 0 không bao giờ chọn trọng số 0; không token nào vượt đích thì dùng phương án dự phòng')}>
          <g opacity={ramp(p, 0.05, 0.2)} data-end="zero" data-gt={gt.index} data-ge={ge.index}>
            <text x={14} y={66} fontSize="12" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>trọng số [0, 0.5, 0.5], u = 0, nên đích là 0; các tổng bao gồm 0, 0.5, 1</>}>weights [0, 0.5, 0.5], u = 0, so the target is 0; inclusive sums 0, 0.5, 1</Translated></text>
            <text x={14} y={92} fontSize="12.5" style={{ ...MONO, fill: 'var(--ink)' }}><Translated vi={<>lớn hơn hẳn 0:     token còn lại #{gt.index} (trọng số 0.5)</>}>strictly greater than 0: survivor #{gt.index} (weight 0.5)</Translated></text>
            <text x={14} y={114} fontSize="12.5" style={{ ...MONO, fill: 'var(--fail)' }}><Translated vi={<>lớn hơn hoặc bằng: token còn lại #{ge.index} (trọng số 0)</>}>greater or equal:   survivor #{ge.index} (weight 0)</Translated></text>
          </g>
          <g opacity={ramp(p, 0.45, 0.6)} data-end="fallback" data-index={FB.index} data-fallback={FB.fallback ? 1 : 0}>
            <text x={14} y={168} fontSize="12" style={{ fill: 'var(--ink-2)' }}>
              <Translated vi={<>{FALLBACK_WEIGHTS.length} token còn lại, u lớn nhất của M5 = 1 − 2⁻²⁴: tổng {short(FB.total)} (tổng theo bước nhảy, rồi cây warp)</>}>
                {FALLBACK_WEIGHTS.length} survivors, M5's largest u = 1 − 2⁻²⁴: total {short(FB.total)} (strided sums, then warp trees)
              </Translated>
            </text>
            <text x={14} y={194} fontSize="12.5" style={{ ...MONO, fill: 'var(--ink)' }}>
              <Translated vi={<>đích {short(FB.target)}; tổng cuối của scan {short(cum[cum.length - 1])}</>}>
                target {short(FB.target)}; the scan's last sum {short(cum[cum.length - 1])}
              </Translated>
              {cum[cum.length - 1] === FB.target ? tr(', equal, so not greater', ', bằng nhau, nên không lớn hơn') : tr(', below it', ', nhỏ hơn nó')}<Translated vi=": không token nào đạt">: nothing qualifies</Translated>
            </text>
            <text x={14} y={220} fontSize="12.5" fontWeight="700" opacity={ramp(p, 0.7, 0.85)} style={{ fill: 'var(--ok)' }}>
              <Translated vi={<>dự phòng: token còn lại cuối cùng có trọng số dương, #{FB.index}</>}>
                fallback: the last survivor with positive weight, #{FB.index}
              </Translated>
            </text>
            <text x={14} y={246} fontSize="11.5" opacity={ramp(p, 0.75, 0.9)} style={{ fill: 'var(--ink-2)' }}>
              <Translated vi={<>u = {short(TOP_U)}; tổng và scan cộng cùng các trọng số theo những thứ tự khác nhau</>}>
                u = {short(TOP_U)}; the total and the scan add the same weights in different orders
              </Translated>
            </text>
          </g>
        </Frame>
      );
    },
  },
];

export function FindUScene() {
  return <Scene chapter="x1" title={tr('Finding u in the scan', 'Tìm u trong scan')} beats={beats} />;
}
