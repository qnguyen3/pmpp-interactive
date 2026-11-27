import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { alignTrace, runningSum, type Step } from '../../sim/fpadd';
import { FP16, FP32, nextDown, nextUp, show, ulp } from '../../sim/minifloat';
import { Translated, tr } from '../../lib/i18n';

// "The disappearing addend": 1000 + 0.3 in FP16, first as one addition (align, add, round),
// then as a loop that stalls at 1024. Every number drawn comes from fpadd.alignTrace and
// fpadd.runningSum (FP16, round to nearest even). Six beats, one narration clip each
// (content/appA/narration/disappearing-addend-N.txt).

const TRACE = alignTrace(FP16, 1000, 0.3, 'rne');
const P = TRACE.p; // 11 significant bits
const COMPARE = TRACE.steps[0] as Extract<Step, { kind: 'compare' }>;
const SHIFTS = TRACE.steps.filter((s): s is Extract<Step, { kind: 'shift' }> => s.kind === 'shift');
const ADD = TRACE.steps.find((s): s is Extract<Step, { kind: 'add' }> => s.kind === 'add')!;
const ROUND = TRACE.steps.find((s): s is Extract<Step, { kind: 'round' }> => s.kind === 'round')!;
const K = 400;
const RUN16 = runningSum(FP16, 1000, 0.3, K);
const RUN32 = runningSum(FP32, 1000, 0.3, K);
const STALL = RUN16.stallAt ?? K;

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

// ---------------------------------------------------------------- the bit view (beats 1 to 3)
const PITCH = 24;
const CELL = 21;
const WX = 168; // window's left edge
const EDGE = WX + P * PITCH; // window's right edge
const GAP = 6; // tail cells sit this far right of the window's edge
const TX = EDGE + GAP; // first tail cell

function Bits({ y, bits, x0, label, sub, exp, tailFrom, dim }: {
  y: number; bits: string; x0: number; label: string; sub?: string; exp: string; tailFrom?: number; dim?: boolean;
}) {
  return (
    <g opacity={dim ? 0.35 : 1}>
      <text x={14} y={y + 15} fontSize="14" fontWeight="600">{label}</text>
      {sub ? <text x={14} y={y + 31} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{sub}</text> : null}
      <text x={WX - 12} y={y + 15} fontSize="12" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{exp}</text>
      {[...bits].map((b, i) => {
        const out = tailFrom !== undefined && i >= tailFrom;
        const x = x0 + i * PITCH + (out ? GAP : 0);
        return (
          <g key={i}>
            <rect x={x} y={y} width={CELL} height={CELL} rx="2" fill={out ? 'var(--page)' : 'var(--cell)'}
              stroke={out ? 'var(--ink-3)' : 'var(--rule)'} strokeDasharray={out ? '3 2' : undefined} />
            <text x={x + CELL / 2} y={y + 15} fontSize="12.5" textAnchor="middle" style={{ fill: out ? 'var(--ink-2)' : 'var(--ink)' }}>{b}</text>
          </g>
        );
      })}
    </g>
  );
}

function Window({ h }: { h: number }) {
  return (
    <g>
      <rect x={WX - 4} y={58} width={P * PITCH + 2.5} height={h} rx="4" fill="none" stroke="var(--ink)" strokeWidth="1.6" />
      <text x={WX - 4} y={52} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>{tr('the 11-bit window (leading 1 first)', 'cửa sổ 11 bit (bit 1 dẫn đầu trước)')}</text>
      <text x={TX + 6} y={52} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>{tr('bits that left it', 'các bit đã rời khỏi nó')}</text>
    </g>
  );
}

function GRS({ g, r, s, n, y }: { g: number; r: number; s: number; n: number; y: number }) {
  if (n === 0) return null;
  const lbl = (i: number, t: string) => (
    <text x={TX + i * PITCH + CELL / 2} y={y} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{t}</text>
  );
  return (
    <g data-g={g} data-r={r} data-s={s}>
      {lbl(0, `G ${g}`)}
      {n > 1 ? lbl(1, `R ${r}`) : null}
      {n > 2 ? (
        <>
          <line x1={TX + 2 * PITCH} x2={TX + (n - 1) * PITCH + CELL} y1={y - 12} y2={y - 12} stroke="var(--ink-3)" />
          <text x={TX + 2 * PITCH + ((n - 2) * PITCH) / 2} y={y} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
            <Translated vi={<>sticky = OR của các bit này = {s}</>}>sticky = OR of these = {s}</Translated>
          </text>
        </>
      ) : null}
    </g>
  );
}

function BitStage({ children, headline, beat, data }: { children: ReactNode; headline: ReactNode; beat: number; data?: Record<string, string | number> }) {
  return (
    <svg viewBox="0 0 720 360" role="img" aria-label={tr('Two FP16 numbers being aligned and added', 'Hai số FP16 đang được căn chỉnh và cộng')} data-beat={beat}
      {...Object.fromEntries(Object.entries(data ?? {}).map(([k, v]) => ['data-' + k, String(v)]))}
      style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      <text x={14} y={26} fontSize="14" fontWeight="600" data-headline>{headline}</text>
      {children}
    </svg>
  );
}

const BIG_Y = 70;
const SMALL_Y = 120;
const SUM_Y = 196;

function beatTwoNumbers(p: number) {
  const show2 = ramp(p, 0.25, 0.45);
  return (
    <BitStage beat={1} headline={tr('1000 + 0.3, both in FP16', '1000 + 0.3, cả hai đều ở FP16')} data={{ diff: COMPARE.diff }}>
      <Window h={92} />
      <Bits y={BIG_Y} bits={COMPARE.big.sig} x0={WX} label="1000" sub={tr('sign +', 'dấu +')} exp={`× 2^${COMPARE.big.exp}`} />
      <g opacity={show2}>
        <Bits y={SMALL_Y} bits={COMPARE.small.sig} x0={WX} label="0.3" sub={tr(`stored as ${show(TRACE.b, 12)}`, `lưu thành ${show(TRACE.b, 12)}`)} exp={`× 2^${COMPARE.small.exp}`} />
      </g>
      <g opacity={ramp(p, 0.6, 0.8)}>
        <text x={14} y={210} fontSize="13">
          <Translated vi={<>số mũ {COMPARE.big.exp} và {COMPARE.small.exp}: cách nhau {COMPARE.diff}.</>}>
            exponents {COMPARE.big.exp} and {COMPARE.small.exp}: {COMPARE.diff} apart.
          </Translated>
        </text>
        <text x={14} y={232} fontSize="13" style={{ fill: 'var(--ink-2)' }}>
          <Translated vi="Các bit của 0.3 có giá trị nhỏ hơn chính các bit đó của 1000, nên không thể cộng chúng tại chỗ.">
            The bits of 0.3 are worth less than the same bits of 1000, so they cannot be added in place.
          </Translated>
        </text>
      </g>
    </BitStage>
  );
}

function beatAlign(p: number) {
  const off = Math.min(COMPARE.diff, ramp(p, 0.08, 0.85) * COMPARE.diff);
  const k = Math.floor(off + 1e-9);
  const st = k > 0 ? SHIFTS[k - 1] : null;
  // Draw 0.3's bits at their shifted place; those right of the window's edge are the tail.
  const x0 = WX + off * PITCH;
  const tailFrom = Math.max(0, P - k);
  return (
    <BitStage beat={2} headline={tr(`shift ${k} of ${COMPARE.diff}: exponent ${COMPARE.small.exp + k}`, `dịch ${k} trên ${COMPARE.diff}: số mũ ${COMPARE.small.exp + k}`)} data={{ k, exp: COMPARE.small.exp + k, window: st?.window ?? COMPARE.small.sig, tail: st?.tail ?? '' }}>
      <Window h={92} />
      <Bits y={BIG_Y} bits={COMPARE.big.sig} x0={WX} label="1000" exp={`× 2^${COMPARE.big.exp}`} />
      <Bits y={SMALL_Y} bits={COMPARE.small.sig} x0={x0} label="0.3" exp={`× 2^${COMPARE.small.exp + k}`} tailFrom={tailFrom} />
      {st ? <GRS g={st.g} r={st.r} s={st.s} n={k} y={SMALL_Y + 42} /> : null}
      {k === COMPARE.diff ? (
        <text x={14} y={226} fontSize="13" opacity={ramp(p, 0.86, 0.95)}>
          <Translated vi="Giờ mọi bit của 0.3 đều nằm ngoài cửa sổ. Chỉ còn các bit guard, round và sticky nhớ đến nó.">
            Every bit of 0.3 is now outside the window. Only the guard, round and sticky bits remember it.
          </Translated>
        </text>
      ) : null}
    </BitStage>
  );
}

function NumberLine({ y, lo, hi, marks, exact, chosen, k }: {
  y: number; lo: number; hi: number; marks: number[]; exact?: number; chosen?: number; k?: string;
}) {
  const X0 = 60;
  const X1 = 680;
  const sx = (v: number) => X0 + ((v - lo) / (hi - lo)) * (X1 - X0);
  return (
    <g data-line={k}>
      <line x1={X0} x2={X1} y1={y} y2={y} stroke="var(--ink-2)" />
      {marks.map((v) => (
        <g key={v}>
          <line x1={sx(v)} x2={sx(v)} y1={y - 7} y2={y + 7} stroke={v === chosen ? 'var(--ok)' : 'var(--ink)'} strokeWidth={v === chosen ? 3 : 1.4} />
          <text x={sx(v)} y={y + 22} fontSize="11.5" textAnchor="middle" style={{ fill: v === chosen ? 'var(--ok)' : 'var(--ink)' }} fontWeight={v === chosen ? 700 : 400}>{show(v)}</text>
        </g>
      ))}
      {exact !== undefined ? (
        <g>
          <circle cx={sx(exact)} cy={y} r={4.5} fill="var(--ink)" />
          <text x={sx(exact)} y={y - 12} fontSize="11.5" textAnchor="middle"><Translated vi={<>chính xác {show(exact, 10)}</>}>exact {show(exact, 10)}</Translated></text>
        </g>
      ) : null}
    </g>
  );
}

function beatRound(p: number) {
  const sumIn = ramp(p, 0.05, 0.25);
  const lineIn = ramp(p, 0.3, 0.5);
  const picked = p >= 0.62;
  return (
    <BitStage beat={3} headline={picked ? tr(`nearest even: ${show(ROUND.result)}, about ${ROUND.errorUlps.toFixed(1)} ULP off`, `chẵn gần nhất: ${show(ROUND.result)}, lệch khoảng ${ROUND.errorUlps.toFixed(1)} ULP`) : tr('add exactly, then round', 'cộng chính xác, rồi làm tròn')}
      data={{ result: ROUND.result, below: ROUND.below, above: ROUND.above, err: ROUND.errorUlps.toFixed(4), exact: ADD.exact }}>
      <Window h={92} />
      <Bits y={BIG_Y} bits={COMPARE.big.sig} x0={WX} label="1000" exp={`× 2^${COMPARE.big.exp}`} dim={sumIn > 0.5} />
      <Bits y={SMALL_Y} bits={COMPARE.small.sig} x0={WX + COMPARE.diff * PITCH} label="0.3" exp={`× 2^${COMPARE.big.exp}`} tailFrom={0} dim={sumIn > 0.5} />
      <g opacity={sumIn}>
        <line x1={WX - 5} x2={TX + P * PITCH} y1={SUM_Y - 14} y2={SUM_Y - 14} stroke="var(--ink-2)" />
        <Bits y={SUM_Y} bits={ADD.window + ADD.tail} x0={WX} label={tr('sum', 'tổng')} sub={tr('exact', 'chính xác')} exp={`× 2^${COMPARE.big.exp}`} tailFrom={P} />
      </g>
      <g opacity={lineIn}>
        <NumberLine y={290} lo={999.8} hi={1000.7} marks={[ROUND.below, ROUND.above]} exact={ADD.exact} chosen={picked ? ROUND.result : undefined} k="round" />{/* i18n-ignore: data key */}
      </g>
      {picked ? (
        <text x={14} y={346} fontSize="12.5" style={{ fill: 'var(--ink-2)' }}>
          <Translated vi={<>Tổng chính xác nằm ở {((ADD.exact - ROUND.below) / (ROUND.above - ROUND.below)).toFixed(2)} quãng đường từ {show(ROUND.below)} đến {show(ROUND.above)}: gần {show(ROUND.above)} hơn.</>}>
            The exact sum lies {((ADD.exact - ROUND.below) / (ROUND.above - ROUND.below)).toFixed(2)} of the way from {show(ROUND.below)} to {show(ROUND.above)}: nearer {show(ROUND.above)}.
          </Translated>
        </text>
      ) : null}
    </BitStage>
  );
}

// ---------------------------------------------------------------- the loop (beats 4 to 6)
interface View { k0: number; k1: number; y0: number; y1: number }
const V4: View = { k0: 0, k1: 64, y0: 998, y1: 1036 };
const V5: View = { k0: 0, k1: K, y0: 995, y1: 1130 };
const PX0 = 70;
const PX1 = 690;
const PY0 = 48;
const PY1 = 236;

function Plot({ v, upto, show32, beat, headline, extra, lineFor }: {
  v: View; upto: number; show32?: number; beat: number; headline: ReactNode; extra?: ReactNode; lineFor?: number;
}) {
  const sx = (k: number) => PX0 + ((k - v.k0) / (v.k1 - v.k0)) * (PX1 - PX0);
  const sy = (y: number) => PY1 - ((y - v.y0) / (v.y1 - v.y0)) * (PY1 - PY0);
  const n = Math.max(0, Math.min(Math.floor(upto), K));
  const pts = (arr: number[]) => arr.slice(0, n + 1).map((y, k) => `${sx(k).toFixed(1)},${sy(Math.min(y, v.y1 + 4)).toFixed(1)}`).join(' ');
  const ticksY = [1000, 1024, 1048, 1072, 1096, 1120].filter((t) => t >= v.y0 && t <= v.y1);
  const stepK = v.k1 > 100 ? 100 : 16;
  const ticksX = Array.from({ length: Math.floor(v.k1 / stepK) + 1 }, (_, i) => i * stepK);
  const acc = RUN16.acc[n];
  const exact = RUN16.exact[n];
  return (
    <svg viewBox="0 0 720 360" role="img" aria-label={tr('Running sum of 0.3 in FP16 against the exact sum', 'Tổng cộng dồn của 0.3 ở FP16 so với tổng chính xác')} data-beat={beat}
      data-k={n} data-acc={acc} data-exact={exact} data-spacing={ulp(FP16, acc)} data-stall={STALL}
      style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      <text x={14} y={26} fontSize="14" fontWeight="600" data-headline>{headline}</text>
      <defs>
        <clipPath id="addend-plot"><rect x={PX0} y={PY0 - 4} width={PX1 - PX0} height={PY1 - PY0 + 8} /></clipPath>
      </defs>
      <line x1={PX0} x2={PX1} y1={PY1} y2={PY1} stroke="var(--ink-2)" />
      <line x1={PX0} x2={PX0} y1={PY0} y2={PY1} stroke="var(--ink-2)" />
      {ticksY.map((t) => (
        <g key={t}>
          <line x1={PX0 - 4} x2={PX1} y1={sy(t)} y2={sy(t)} stroke="var(--rule-soft)" />
          <text x={PX0 - 8} y={sy(t) + 4} fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{t}</text>
        </g>
      ))}
      {ticksX.map((t) => (
        <text key={t} x={sx(t)} y={PY1 + 16} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{t}</text>
      ))}
      <text x={PX1} y={PY1 + 32} fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{tr('additions of 0.3', 'số lần cộng 0.3')}</text>
      <g clipPath="url(#addend-plot)">
        <polyline points={pts(RUN16.exact)} fill="none" stroke="var(--ink-2)" strokeWidth="1.6" strokeDasharray="5 4" data-curve="exact" />
        {show32 ? <polyline points={pts(RUN32.acc)} fill="none" stroke="var(--ok)" strokeWidth="2" opacity={show32} data-curve="fp32" /> : null}
        <polyline points={pts(RUN16.acc.slice(0, STALL))} fill="none" stroke="var(--ink)" strokeWidth="2.2" />
        {n >= STALL ? (
          <polyline points={pts(RUN16.acc).split(' ').slice(STALL - 1).join(' ')} fill="none" stroke="var(--fail)" strokeWidth="2.4" data-curve="fp16" />
        ) : (
          <polyline points="" fill="none" stroke="var(--ink)" data-curve="fp16" />
        )}
      </g>
      <g>
        <text x={PX0 + 8} y={PY0 + 4} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>{tr('exact (dashed)', 'chính xác (nét đứt)')}</text>
        <text x={PX0 + 8} y={PY0 + 20} fontSize="11.5" style={{ fill: n >= STALL ? 'var(--fail)' : 'var(--ink)' }}><Translated vi={<>tổng cộng dồn FP16: {show(acc)}</>}>FP16 running sum: {show(acc)}</Translated></text>
        {show32 ? <text x={PX0 + 8} y={PY0 + 36} fontSize="11.5" opacity={show32} style={{ fill: 'var(--ok)' }}><Translated vi={<>tổng cộng dồn FP32: {show(RUN32.acc[n], 7)}</>}>FP32 running sum: {show(RUN32.acc[n], 7)}</Translated></text> : null}
      </g>
      {lineFor !== undefined ? <Spacing at={lineFor} /> : null}
      {extra}
    </svg>
  );
}

/** The FP16 values around the accumulator after k additions, and where + 0.3 lands. */
function Spacing({ at }: { at: number }) {
  const k = Math.max(0, Math.min(K, at));
  const acc = RUN16.acc[k];
  const next = RUN16.acc[Math.min(K, k + 1)];
  const target = acc + RUN16.addend;
  const scale = 110; // pixels per unit
  const cx = 360;
  const sx = (v: number) => cx + (v - acc) * scale;
  const ticks: number[] = [acc];
  for (let v = acc, i = 0; i < 6; i++) { const u = nextUp(FP16, v); if (u === null) break; ticks.push(u); v = u; }
  for (let v = acc, i = 0; i < 6; i++) { const d = nextDown(FP16, v); if (d === null) break; ticks.push(d); v = d; }
  const Y = 300;
  return (
    <g data-spacing-line data-acc={acc} data-next={next}>
      <line x1={40} x2={680} y1={Y} y2={Y} stroke="var(--ink-2)" />
      {ticks.filter((v) => sx(v) > 40 && sx(v) < 680).map((v) => (
        <g key={v}>
          <line x1={sx(v)} x2={sx(v)} y1={Y - 6} y2={Y + 6} stroke="var(--ink)" strokeWidth={v === acc ? 2.6 : 1.2} />
          <text x={sx(v)} y={Y + 36} fontSize="10.5" textAnchor="middle" style={{ fill: v === acc ? 'var(--ink)' : 'var(--ink-2)' }}>{show(v)}</text>
        </g>
      ))}
      <path d={`M ${sx(acc)} ${Y - 10} Q ${(sx(acc) + sx(target)) / 2} ${Y - 34} ${sx(target)} ${Y - 10}`} fill="none" stroke="var(--ink)" markerEnd="url(#addend-arrow)" />
      <circle cx={sx(target)} cy={Y} r={3.5} fill="var(--ink)" />
      <path d={`M ${sx(target)} ${Y + 6} Q ${(sx(target) + sx(next)) / 2} ${Y + 22} ${sx(next)} ${Y + 6}`} fill="none"
        stroke={next === acc ? 'var(--fail)' : 'var(--ok)'} strokeWidth="1.6" />
      <defs>
        <marker id="addend-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">{/* i18n-ignore: SVG attribute value */}
          <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--ink)" />
        </marker>
      </defs>
      <text x={40} y={Y - 30} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi={<>khoảng cách ở đây: {show(ulp(FP16, acc))} · + 0.3 rơi vào {show(target, 8)} · làm tròn thành {show(next)}</>}>
          spacing here: {show(ulp(FP16, acc))} · + 0.3 lands at {show(target, 8)} · rounds to {show(next)}
        </Translated>
      </text>
    </g>
  );
}

function beatClimb(p: number) {
  const upto = ramp(p, 0.05, 0.92) * V4.k1;
  const k = Math.floor(upto);
  const crossed = RUN16.acc[k] >= 1024;
  return (
    <Plot v={V4} upto={upto} beat={4} lineFor={Math.min(k, STALL - 2)}
      headline={crossed ? tr('at 1024 the exponent goes up by one: the spacing doubles to 1', 'tại 1024 số mũ tăng thêm một: khoảng cách gấp đôi thành 1') : tr('below 1024 the spacing is 0.5: every + 0.3 rounds up to + 0.5', 'dưới 1024 khoảng cách là 0.5: mỗi + 0.3 làm tròn lên thành + 0.5')} />
  );
}

function beatStuck(p: number) {
  const z = ramp(p, 0, 0.2);
  const v: View = { k0: 0, k1: lerp(V4.k1, V5.k1, z), y0: lerp(V4.y0, V5.y0, z), y1: lerp(V4.y1, V5.y1, z) };
  const upto = lerp(V4.k1, K, ramp(p, 0.15, 0.9));
  const k = Math.floor(upto);
  return (
    <Plot v={v} upto={upto} beat={5} lineFor={k}
      headline={tr(`stuck at ${show(RUN16.acc[k])} after ${STALL - 1} additions; the exact sum is ${show(RUN16.exact[k], 6)}`, `kẹt ở ${show(RUN16.acc[k])} sau ${STALL - 1} lần cộng; tổng chính xác là ${show(RUN16.exact[k], 6)}`)} />
  );
}

function beatWaysOut(p: number) {
  const items = [
    tr('add the small values to each other first (a tree)', 'cộng các giá trị nhỏ với nhau trước (một cây)'),
    tr('carry what is lost in a second variable (Kahan, section A.6)', 'giữ phần bị mất trong một biến thứ hai (Kahan, mục A.6)'),
    tr('accumulate in FP32: a 24-bit window', 'tích lũy ở FP32: một cửa sổ 24 bit'),
  ];
  return (
    <Plot v={V5} upto={K} beat={6} show32={ramp(p, 0.62, 0.8)}
      headline={tr('three ways out', 'ba lối thoát')}
      extra={
        <g>
          {items.map((t, i) => (
            <text key={i} x={PX0} y={290 + i * 24} fontSize="13" opacity={ramp(p, 0.08 + i * 0.27, 0.18 + i * 0.27)} data-way={i + 1}>
              {i + 1}. {t}
            </text>
          ))}
        </g>
      }
    />
  );
}

const beats: Beat[] = [
  { clip: 'disappearing-addend-1', get label() { return tr('Two numbers', 'Hai số'); }, draw: beatTwoNumbers },
  { clip: 'disappearing-addend-2', get label() { return tr('Align', 'Căn chỉnh'); }, draw: beatAlign },
  { clip: 'disappearing-addend-3', get label() { return tr('Round', 'Làm tròn'); }, draw: beatRound },
  { clip: 'disappearing-addend-4', get label() { return tr('Keep adding', 'Cộng tiếp'); }, draw: beatClimb },
  { clip: 'disappearing-addend-5', get label() { return tr('Stuck', 'Bị kẹt'); }, draw: beatStuck },
  { clip: 'disappearing-addend-6', get label() { return tr('Ways out', 'Lối thoát'); }, draw: beatWaysOut },
];

export function DisappearingAddendScene() {
  return <Scene chapter="appA" title={tr('The disappearing addend', 'Số hạng biến mất')} beats={beats} />;
}
