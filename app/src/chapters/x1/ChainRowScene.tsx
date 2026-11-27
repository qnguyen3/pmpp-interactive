import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { tr } from '../../lib/i18n';
import { KAT_WORD, SCENE_ROW, SCENE_SETTINGS, rankDesc, sceneChain } from '../../sim/chain';

// "One row through the chain": the scene row of src/sim/chain.ts through M5's order, one
// beat per stage. Every value drawn comes from sceneChain()'s snapshots; nothing is typed
// in by hand. Clips: content/x1/narration/chain-row-N.txt.
const R = sceneChain();
const N = SCENE_ROW.logits.length;
const S = SCENE_SETTINGS;
const snap = (stage: string) => R.snapshots.find((s) => s.stage === stage)!;
const RAW = SCENE_ROW.logits;
const PEN = snap('penalties').logits;
const COUNTS = RAW.map((_, i) => SCENE_ROW.context.filter((t) => t === i).length);
const idsOf = (a: boolean[]) => a.flatMap((x, i) => (x ? [i] : []));
const TOPK = idsOf(snap('top-k').alive);
const TOPK_SORTED = rankDesc(PEN, snap('top-k').alive);
const TOPK_PROBS = snap('top-k').probs;
const TOPP_ALIVE = snap('top-p').alive;
const MINP = snap('min-p');
const MINP_SORTED = rankDesc(PEN, TOPP_ALIVE);
const FINAL = R.draw;
/** The tie at the top-k cut: the first dropped token whose logit equals the last kept one. */
const CUT = (() => {
  const all = rankDesc(PEN, PEN.map(() => true));
  const last = all[S.k - 1];
  const next = all[S.k];
  return PEN[last] === PEN[next] ? { kept: Math.min(last, next), dropped: Math.max(last, next), value: PEN[last] } : null;
})();

const W = 720;
const H = 330;
const X0 = 48;
const SLOT = 52;
const BW = 34;
const ZERO = 168; // y of logit 0
const LS = 28; // px per logit unit
const PBASE = 236; // y of probability 0
const PS = 170; // px per probability 1
const STRIP_Y = 268;
const STRIP_X0 = 48;
const STRIP_W = 620;

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const slotX = (slot: number) => X0 + slot * SLOT;
const f2 = (v: number) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2));
const pc = (v: number) => `${Math.round(v * 100)}%`;

function Frame({ headline, sub, children }: { headline: string; sub?: string; children: ReactNode }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('One row of logits through the sampler chain', 'Một hàng logits đi qua chuỗi sampler')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <text x={X0 - 8} y={22} fontSize="14" fontWeight="650" data-headline>{headline}</text>
      {sub ? <text x={X0 - 8} y={40} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>{sub}</text> : null}
      {children}
    </svg>
  );
}

/** One logit bar with its value, its token id and (optionally) its count. */
function LogitBar({ id, x, v, opacity = 1, strong = false, count, dim = false }: {
  id: number; x: number; v: number; opacity?: number; strong?: boolean; count?: number; dim?: boolean;
}) {
  const h = Math.abs(v) * LS;
  const y = v >= 0 ? ZERO - h : ZERO;
  return (
    <g opacity={opacity} data-token={id}>
      <rect x={x} y={y} width={BW} height={Math.max(h, 1)} rx="2"
        fill={dim ? 'var(--cell)' : 'var(--ink-3)'} fillOpacity={dim ? 1 : 0.55}
        stroke={strong ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth={strong ? 2 : 1} />
      <text x={x + BW / 2} y={v >= 0 ? y - 4 : y + h + 12} fontSize="10.5" textAnchor="middle" data-value={f2(v)}>{f2(v)}</text>
      <text x={x + BW / 2} y={258} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{id}</text>
      {count != null && count > 0 ? (
        <text x={x + BW / 2} y={274} fontSize="10.5" textAnchor="middle" fontWeight="650" data-count={count}>{tr(`used ${count}`, `dùng ${count}`)}</text>
      ) : null}
    </g>
  );
}

function ProbBar({ id, x, p, opacity = 1, strong = false, dim = false, label }: {
  id: number; x: number; p: number; opacity?: number; strong?: boolean; dim?: boolean; label?: string;
}) {
  const h = p * PS;
  return (
    <g opacity={opacity} data-token={id}>
      <rect x={x} y={PBASE - h} width={BW} height={Math.max(h, 1)} rx="2"
        fill={dim ? 'var(--cell)' : strong ? 'var(--ink)' : 'var(--ink-3)'} fillOpacity={dim ? 1 : strong ? 0.85 : 0.55}
        stroke={strong ? 'var(--ink)' : 'var(--ink-3)'} />
      <text x={x + BW / 2} y={PBASE - h - 4} fontSize="10.5" textAnchor="middle">{label ?? pc(p)}</text>
      <text x={x + BW / 2} y={PBASE + 14} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{id}</text>
    </g>
  );
}

const ZeroLine = () => (
  <g>
    <line x1={X0 - 6} y1={ZERO} x2={slotX(N) - 6} y2={ZERO} stroke="var(--rule)" />
    <text x={X0 - 10} y={ZERO + 4} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>0</text>
  </g>
);
const ProbAxis = () => <line x1={X0 - 6} y1={PBASE} x2={slotX(N) - 6} y2={PBASE} stroke="var(--rule)" />;

/** Survivors laid end to end from 0 to 1, as the draw sees them. */
function Strip({ ids, probs, opacity = 1, chosen, u, pLine, marker }: {
  ids: number[]; probs: number[]; opacity?: number; chosen?: number; u?: number; pLine?: number; marker?: number;
}) {
  let at = 0;
  const sx = (m: number) => STRIP_X0 + m * STRIP_W;
  return (
    <g opacity={opacity} data-part="strip">
      {ids.map((id, j) => {
        const x = sx(at);
        const w = probs[j] * STRIP_W;
        at += probs[j];
        return (
          <g key={id} data-segment={id}>
            <rect x={x} y={STRIP_Y} width={Math.max(0, w)} height={22}
              fill={chosen === id ? 'var(--ink)' : 'var(--ink-3)'} fillOpacity={chosen === id ? 0.85 : 0.3} stroke="var(--ink-2)" strokeWidth="0.8" />
            {w > 22 ? <text x={x + w / 2} y={STRIP_Y + 15} fontSize="11" textAnchor="middle" style={{ fill: chosen === id ? 'var(--page)' : 'var(--ink)' }}>{id}</text> : null}
          </g>
        );
      })}
      <text x={sx(0)} y={STRIP_Y + 36} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>0</text>
      <text x={sx(1)} y={STRIP_Y + 36} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>1</text>
      {pLine != null ? (
        <g data-pline={pLine}>
          <line x1={sx(pLine)} y1={STRIP_Y - 6} x2={sx(pLine)} y2={STRIP_Y + 28} stroke="var(--ink)" strokeWidth="2" />
          <text x={sx(pLine)} y={STRIP_Y + 40} fontSize="11" textAnchor="middle" fontWeight="650">p = {pLine}</text>
        </g>
      ) : null}
      {marker != null ? <path d={`M ${sx(marker)} ${STRIP_Y - 2} l -5 -9 l 10 0 z`} fill="var(--ink)" data-marker={marker.toFixed(4)} /> : null}
      {u != null ? (
        <g data-u={u.toFixed(4)}>
          <line x1={sx(u)} y1={STRIP_Y - 6} x2={sx(u)} y2={STRIP_Y + 30} stroke="var(--ink)" strokeWidth="2" strokeDasharray="4 2" />
          <text x={sx(u)} y={STRIP_Y + 44} fontSize="11" textAnchor="middle" fontWeight="650">u = {u.toFixed(3)}</text>
        </g>
      ) : null}
    </g>
  );
}

// ---------------------------------------------------------------- beats
function penalties(p: number) {
  const t = ramp(p, 0.2, 0.7);
  const rawTop = RAW.indexOf(Math.max(...RAW));
  const penTop = rankDesc(PEN, PEN.map(() => true));
  return (
    <Frame
      headline={p < 0.2 ? tr(`${N} logits, and how often each token is in the context`, `${N} logits, và số lần mỗi token có trong ngữ cảnh`) : tr(`penalties: repetition ${S.repeat}, frequency ${S.freq}, presence ${S.presence}`, `penalty: lặp lại ${S.repeat}, tần suất ${S.freq}, hiện diện ${S.presence}`)}
      sub={p < 0.75 ? tr('positive logits are divided, negative ones multiplied, then count × frequency + presence is subtracted', 'logit dương bị chia, logit âm bị nhân, rồi trừ đi count × frequency + presence')
        : tr(`token ${rawTop} was first; now token ${penTop[0]} is first and token ${rawTop} is ${penTop.indexOf(rawTop) === 1 ? 'second' : 'lower'}`, `token ${rawTop} từng đứng đầu; giờ token ${penTop[0]} đứng đầu và token ${rawTop} ${penTop.indexOf(rawTop) === 1 ? 'đứng thứ hai' : 'xuống thấp hơn'}`)}>
      <ZeroLine />
      {RAW.map((v, i) => (
        <LogitBar key={i} id={i} x={slotX(i)} v={lerp(v, PEN[i], COUNTS[i] > 0 ? t : 0)} count={COUNTS[i]} strong={COUNTS[i] > 0} />
      ))}
    </Frame>
  );
}

function topK(p: number) {
  const fade = ramp(p, 0.25, 0.6);
  return (
    <Frame headline={tr(`top-k, k = ${S.k}: ${TOPK.length} stay lit, ${N - TOPK.length} fade where they stand`, `top-k, k = ${S.k}: ${TOPK.length} vẫn sáng, ${N - TOPK.length} mờ đi tại chỗ`)}
      sub={CUT && p > 0.55 ? tr(`tokens ${CUT.kept} and ${CUT.dropped} tie at the cut (${f2(CUT.value)}): the lower id, ${CUT.kept}, stays`, `token ${CUT.kept} và ${CUT.dropped} hòa nhau tại điểm cắt (${f2(CUT.value)}): id nhỏ hơn, ${CUT.kept}, được giữ`) : tr('a selection finds the cut; nothing is sorted', 'một phép chọn tìm điểm cắt; không có gì được sắp xếp')}>
      <ZeroLine />
      {PEN.map((v, i) => {
        const kept = TOPK.includes(i);
        return <LogitBar key={i} id={i} x={slotX(i)} v={v} opacity={kept ? 1 : 1 - 0.7 * fade} dim={!kept && fade > 0.5} strong={CUT != null && p > 0.55 && (i === CUT.kept || i === CUT.dropped)} />;
      })}
    </Frame>
  );
}

function sortSoftmax(p: number) {
  const move = ramp(p, 0.08, 0.45);
  const morph = ramp(p, 0.5, 0.85);
  const sum = TOPK_SORTED.reduce((s, i) => s + TOPK_PROBS[i], 0);
  return (
    <Frame headline={morph < 1 ? tr(`sort the ${TOPK.length} survivors, descending`, `sắp xếp ${TOPK.length} token còn lại, giảm dần`) : tr(`softmax over ${TOPK.length} values: they add up to ${sum.toFixed(2)}`, `softmax trên ${TOPK.length} giá trị: tổng của chúng là ${sum.toFixed(2)}`)}
      sub={tr(`only ${TOPK.length} values sorted and normalised, not the whole vocabulary`, `chỉ ${TOPK.length} giá trị được sắp xếp và chuẩn hóa, không phải cả bộ từ vựng`)}>
      {morph > 0 ? <ProbAxis /> : <ZeroLine />}
      {TOPK_SORTED.map((id, slot) => {
        const x = lerp(slotX(id), slotX(slot + 1), move);
        if (morph === 0) return <LogitBar key={id} id={id} x={x} v={PEN[id]} />;
        return <ProbBar key={id} id={id} x={x} p={TOPK_PROBS[id] * morph} label={morph < 1 ? '' : pc(TOPK_PROBS[id])} />;
      })}
    </Frame>
  );
}

function topPBeat(p: number) {
  const ids = TOPK_SORTED;
  const probs = ids.map((i) => TOPK_PROBS[i]);
  const before = probs.map((_, j) => probs.slice(0, j).reduce((a, b) => a + b, 0));
  const reached = Math.min(ids.length, Math.floor(ramp(p, 0.12, 0.85) * (ids.length + 0.999)));
  const markerAt = reached > 0 ? before[Math.min(reached, ids.length) - 1] : 0;
  return (
    <Frame headline={tr(`top-p at ${S.p}: keep a token while the mass ranked before it is below ${S.p}`, `top-p tại ${S.p}: giữ một token khi khối xác suất xếp hạng trước nó còn dưới ${S.p}`)}
      sub={reached > 0 ? tr(`token ${ids[reached - 1]}: mass before ${before[reached - 1].toFixed(2)} ${before[reached - 1] < S.p ? `< ${S.p}, kept` : `≥ ${S.p}, dropped`}`, `token ${ids[reached - 1]}: khối xác suất trước nó ${before[reached - 1].toFixed(2)} ${before[reached - 1] < S.p ? `< ${S.p}, giữ` : `≥ ${S.p}, loại`}`) : tr('the marker carries the mass ranked before each token', 'con trỏ mang khối xác suất xếp hạng trước mỗi token')}>
      <ProbAxis />
      {ids.map((id, slot) => {
        const decided = slot < reached;
        const kept = TOPP_ALIVE[id];
        return (
          <g key={id}>
            <ProbBar id={id} x={slotX(slot + 1)} p={probs[slot]} dim={decided && !kept} opacity={decided && !kept ? 0.45 : 1} />
            {decided ? (
              <text x={slotX(slot + 1) + BW / 2} y={PBASE + 28} fontSize="10" textAnchor="middle" data-before={before[slot].toFixed(2)}
                style={{ fill: kept ? 'var(--ink)' : 'var(--ink-3)' }}>{before[slot].toFixed(2)}</text>
            ) : null}
          </g>
        );
      })}
      <text x={slotX(0) + BW} y={PBASE + 28} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{tr('before:', 'trước:')}</text>
      <Strip ids={ids} probs={probs} pLine={S.p} marker={markerAt} />
    </Frame>
  );
}

function minPBeat(p: number) {
  const ids = MINP_SORTED;
  const top = PEN[ids[0]];
  const th = MINP.threshold!;
  const lineV = lerp(top, th, ramp(p, 0.15, 0.55));
  const decided = p > 0.62;
  const yLine = ZERO - lineV * LS;
  return (
    <Frame headline={tr(`min-p at ${S.minP}: keep a logit at or above max + ln ${S.minP} = ${f2(top)} − ${f2(-Math.log(S.minP))} = ${f2(th)}`, `min-p tại ${S.minP}: giữ logit bằng hoặc trên max + ln ${S.minP} = ${f2(top)} − ${f2(-Math.log(S.minP))} = ${f2(th)}`)}
      sub={tr(`the same test as probability ≥ ${S.minP} × the top probability, without a softmax`, `cùng phép kiểm tra như xác suất ≥ ${S.minP} × xác suất lớn nhất, mà không cần softmax`)}>
      <ZeroLine />
      {ids.map((id, slot) => {
        const kept = MINP.alive[id];
        return <LogitBar key={id} id={id} x={slotX(slot + 1)} v={PEN[id]} dim={decided && !kept} opacity={decided && !kept ? 0.45 : 1} strong={decided && !kept} />;
      })}
      <g data-threshold={th.toFixed(4)}>
        <line x1={X0 - 6} y1={yLine} x2={slotX(ids.length + 1) + 8} y2={yLine} stroke="var(--ink)" strokeWidth="2" strokeDasharray="6 3" />
        <text x={slotX(ids.length + 1) + 12} y={yLine + 4} fontSize="11" fontWeight="650">{f2(lineV)}</text>
      </g>
      {decided ? (
        <text x={slotX(ids.length + 2)} y={ZERO + 50} fontSize="12">
          {ids.filter((i) => !MINP.alive[i]).map((i) => tr(`token ${i} (${f2(PEN[i])}) falls`, `token ${i} (${f2(PEN[i])}) bị loại`)).join(', ')}; {idsOf(MINP.alive).length} {tr('left', 'còn lại')}
        </text>
      ) : null}
    </Frame>
  );
}

function drawBeat(p: number) {
  const ids = FINAL.order;
  const t1 = ids.map((i) => MINP.probs[i]);
  const tT = FINAL.probs;
  const morph = ramp(p, 0.05, 0.35);
  const now = ids.map((_, j) => lerp(t1[j], tT[j], morph));
  const strip = ramp(p, 0.35, 0.5);
  const showU = p > 0.58;
  const chosen = p > 0.74 ? FINAL.token : undefined;
  return (
    <Frame headline={chosen != null ? tr(`u = ${FINAL.u.toFixed(3)} lands in token ${FINAL.token}: token ${FINAL.token} is written out`, `u = ${FINAL.u.toFixed(3)} rơi vào token ${FINAL.token}: token ${FINAL.token} được ghi ra`) : tr(`temperature ${S.T}: the ${ids.length} survivors reweighted`, `temperature ${S.T}: ${ids.length} token còn lại được đánh lại trọng số`)}
      sub={showU ? tr(`u from Philox4x32-10, counter 0, key 0 (seed 0, row 0, step 0): word ${KAT_WORD.toString(16)}, top 24 bits`, `u từ Philox4x32-10, counter 0, key 0 (seed 0, hàng 0, bước 0): word ${KAT_WORD.toString(16)}, 24 bit cao`) : tr(`token ${ids[0]}: ${pc(t1[0])} at T = 1, ${pc(tT[0])} at T = ${S.T}`, `token ${ids[0]}: ${pc(t1[0])} tại T = 1, ${pc(tT[0])} tại T = ${S.T}`)}>
      <ProbAxis />
      {ids.map((id, slot) => (
        <ProbBar key={id} id={id} x={slotX(slot + 1)} p={now[slot]} strong={chosen === id} />
      ))}
      {strip > 0 ? <Strip ids={ids} probs={tT} opacity={strip} chosen={chosen} u={showU ? FINAL.u : undefined} /> : null}
    </Frame>
  );
}

const beats: Beat[] = [
  { clip: 'chain-row-1', get label() { return tr('Penalties', 'Penalty'); }, draw: penalties },
  { clip: 'chain-row-2', get label() { return tr('Top-k', 'Top-k'); }, draw: topK },
  { clip: 'chain-row-3', get label() { return tr('Sort, softmax', 'Sắp xếp, softmax'); }, draw: sortSoftmax },
  { clip: 'chain-row-4', get label() { return tr('Top-p', 'Top-p'); }, draw: topPBeat },
  { clip: 'chain-row-5', get label() { return tr('Min-p', 'Min-p'); }, draw: minPBeat },
  { clip: 'chain-row-6', get label() { return tr('Draw', 'Rút'); }, draw: drawBeat },
];

export function ChainRowScene() {
  return <Scene chapter="x1" title={tr('One row through the chain', 'Một hàng đi qua chuỗi')} beats={beats} />;
}
