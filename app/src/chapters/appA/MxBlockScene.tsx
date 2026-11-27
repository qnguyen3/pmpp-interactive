import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { E2M1, binade, emax } from '../../sim/minifloat';
import { SCENE_BLOCK, blockBytes, elemGrid, maxTree, quantizeBlock, sceneValues, type QuantOpts } from '../../sim/mxblock';
import { Translated, tr } from '../../lib/i18n';

// "Quantising one block": one MXFP4 block of 32 weights through the OCP rule, every
// drawing computed from src/sim/mxblock.ts (quantizeBlock, maxTree) on a fixed block
// (SCENE_BLOCK) so the scripts in content/appA/narration/mx-block-N.txt can quote numbers.
const MXFP4: QuantOpts = { elem: 'E2M1', k: 32, scale: 'ocp-floor' };
const V = sceneValues();
const Q32 = quantizeBlock(V, MXFP4);
const Q16 = quantizeBlock(V, { ...MXFP4, k: 16 });
const TREE = maxTree(V);
const B = Q32.blocks[0];
const X = B.scale;
const AT = SCENE_BLOCK.at;
const GRID = elemGrid(E2M1);
const REST = Math.max(...V.filter((_, i) => i !== AT).map(Math.abs));

const W = 720;
const H = 330;
const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp01((p - a) / (b - a));
const ease = (t: number) => t * t * (3 - 2 * t);
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const fmt = (v: number, d = 2) => (Math.abs(v) < 1e-12 ? '0' : Number(v.toFixed(d)).toString().replace('-', '−'));

// ---------------------------------------------------------------- the number line
const U0 = -1.5;
const U1 = 13;
const LX0 = 40;
const LX1 = 690;
const ux = (u: number) => LX0 + ((u - U0) / (U1 - U0)) * (LX1 - LX0);
const jitter = (i: number) => (((i * 5) % 7) - 3) * 8;

function Line({ y, children }: { y: number; children?: ReactNode }) {
  const ticks = [0, 2, 4, 6, 8, 10, 12];
  return (
    <g>
      <line x1={LX0} x2={LX1} y1={y + 34} y2={y + 34} stroke="var(--rule)" />
      {ticks.map((t) => (
        <g key={t}>
          <line x1={ux(t)} x2={ux(t)} y1={y + 30} y2={y + 38} stroke="var(--ink-3)" />
          <text x={ux(t)} y={y + 52} fontSize="11" textAnchor="middle">{t}</text>
        </g>
      ))}
      {children}
    </g>
  );
}

interface Dot {
  u: number;
  i: number;
  faded?: boolean;
  hot?: boolean;
  lost?: boolean;
}
function Dots({ y, dots }: { y: number; dots: Dot[] }) {
  return (
    <g>
      {dots.map((d) => (
        <circle key={d.i} cx={ux(d.u)} cy={y + jitter(d.i)} r={d.hot ? 6 : 4.5} opacity={d.faded ? 0.22 : 1}
          fill={d.lost ? 'var(--fail)' : d.hot ? 'var(--global)' : 'var(--global-wash)'}
          stroke={d.lost ? 'var(--fail)' : 'var(--global)'} strokeWidth="1.3" />
      ))}
    </g>
  );
}

function Headline({ text }: { text: string }) {
  return <text x={LX0} y={H - 14} fontSize="13.5" fontWeight="600">{text}</text>;
}

function Frame({ beat, data, children }: { beat: number; data: Record<string, string | number>; children: ReactNode }) {
  const attrs = Object.fromEntries(Object.entries(data).map(([k, v]) => ['data-' + k.replace(/[A-Z]/g, (c) => '-' + c.toLowerCase()), String(v)]));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('One MXFP4 block being quantised', 'Một khối MXFP4 đang được lượng tử hóa')} data-beat={beat} {...attrs}
      style={{ width: '100%', maxWidth: W, display: 'block' }}>
      {children}
    </svg>
  );
}

// ---------------------------------------------------------------- the beats
function beat1(p: number) {
  const shown = 32;
  const dots = V.map((v, i) => ({ u: v, i, hot: i === AT && p > 0.35 }));
  return (
    <Frame beat={1} data={{ shown, outlier: V[AT], rest: fmt(REST) }}>
      <text x={LX0} y={40} fontSize="12" style={{ fill: 'var(--ink-2)' }}>{tr('32 weights of one block, on a number line (weights in global memory)', '32 trọng số của một khối, trên một trục số (trọng số nằm trong bộ nhớ toàn cục (global memory))')}</text>
      <Line y={150}><Dots y={150} dots={dots} /></Line>
      {p > 0.35 ? <text x={ux(V[AT])} y={110} fontSize="12" textAnchor="middle"><Translated vi={<>phần tử {AT}: {fmt(V[AT])}</>}>element {AT}: {fmt(V[AT])}</Translated></text> : null}
      <Headline text={p > 0.35 ? tr(`one outlier at ${fmt(V[AT])}; the other 31 lie within ±${fmt(REST)}`, `một giá trị ngoại lai ở ${fmt(V[AT])}; 31 giá trị còn lại nằm trong ±${fmt(REST)}`) : tr(`${shown} weights, one block`, `${shown} trọng số, một khối`)} />
    </Frame>
  );
}

function beat2(p: number) {
  const round = Math.min(TREE.length - 1, Math.floor(ramp(p, 0.05, 0.85) * TREE.length));
  const alive = new Set(TREE[round]);
  const TX0 = 72;
  const slot = 18.5;
  return (
    <Frame beat={2} data={{ round, survivors: TREE[round].length, winner: TREE[TREE.length - 1][0] }}>
      <text x={LX0} y={18} fontSize="12" style={{ fill: 'var(--ink-2)' }}>{tr('a max reduction over |v|: neighbours compare, the larger goes on', 'một phép rút gọn (reduction) max trên |v|: các cặp kề nhau so sánh, giá trị lớn hơn đi tiếp')}</text>
      {TREE.slice(0, round + 1).map((lvl, l) => {
        const span = 32 / lvl.length;
        return (
          <g key={l}>
            <text x={TX0 - 10} y={36 + l * 14} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>{lvl.length}</text>
            {lvl.map((idx, j) => (
              <rect key={j} x={TX0 + (j + 0.5) * span * slot - 6} y={28 + l * 14} width={12} height={10} rx="2"
                fill={idx === AT ? 'var(--global)' : 'var(--global-wash)'} stroke="var(--global)" strokeWidth="0.8" />
            ))}
          </g>
        );
      })}
      <Line y={190}><Dots y={190} dots={V.map((v, i) => ({ u: v, i, faded: !alive.has(i), hot: i === AT && round === TREE.length - 1 }))} /></Line>
      <Headline text={round === 0 ? tr('32 candidates', '32 ứng viên') : round < TREE.length - 1 ? tr(`round ${round}: ${TREE[round].length} left`, `vòng ${round}: còn ${TREE[round].length}`) : tr(`after ${round} rounds: ${fmt(B.amax)}, element ${B.amaxAt}`, `sau ${round} vòng: ${fmt(B.amax)}, phần tử ${B.amaxAt}`)} />
    </Frame>
  );
}

function beat3(p: number) {
  const k = binade(B.amax);
  const lines = [
    tr(`largest magnitude: ${fmt(B.amax)}`, `độ lớn lớn nhất: ${fmt(B.amax)}`),
    tr(`log2 ${fmt(B.amax)} = ${Math.log2(B.amax).toFixed(3)}, rounded down: ${k}`, `log2 ${fmt(B.amax)} = ${Math.log2(B.amax).toFixed(3)}, làm tròn xuống: ${k}`),
    tr(`largest exponent of E2M1: ${emax(E2M1)}  (6 = 1.5 × 2^${emax(E2M1)})`, `số mũ lớn nhất của E2M1: ${emax(E2M1)}  (6 = 1.5 × 2^${emax(E2M1)})`),
    tr(`shared exponent: ${k} − ${emax(E2M1)} = ${B.scaleExp}`, `số mũ dùng chung: ${k} − ${emax(E2M1)} = ${B.scaleExp}`),
    tr(`scale X = 2^${B.scaleExp} = ${fmt(X)}`, `hệ số tỉ lệ X = 2^${B.scaleExp} = ${fmt(X)}`),
    tr(`stored as UE8M0: ${B.scaleExp} + 127 = ${B.scaleCode}`, `lưu dưới dạng UE8M0: ${B.scaleExp} + 127 = ${B.scaleCode}`),
  ];
  const n = Math.min(lines.length, Math.floor(ramp(p, 0.02, 0.8) * (lines.length + 0.99)));
  const bits = B.scaleCode.toString(2).padStart(8, '0');
  return (
    <Frame beat={3} data={{ lines: n, exp: B.scaleExp!, scale: X, code: B.scaleCode }}>
      <text x={LX0} y={36} fontSize="12" style={{ fill: 'var(--ink-2)' }}>{tr('the OCP rule (MX v1.1 §6.3)', 'quy tắc OCP (MX v1.1 §6.3)')}</text>
      {lines.slice(0, n).map((t, i) => (
        <text key={i} x={LX0} y={70 + i * 30} fontSize="15" fontWeight={i === 4 ? 650 : 450}>{t}</text>
      ))}
      {n >= lines.length ? (
        <g>
          <text x={500} y={226} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{tr('one byte per block', 'một byte mỗi khối')}</text>
          {bits.split('').map((b, i) => (
            <g key={i}>
              <rect x={500 + i * 22} y={234} width={20} height={26} rx="2" fill="var(--cell)" stroke="var(--ink)" />
              <text x={510 + i * 22} y={252} fontSize="13" textAnchor="middle">{b}</text>
            </g>
          ))}
        </g>
      ) : null}
      <Headline text={n >= 5 ? tr(`X = ${fmt(X)}: every weight in the block is stored as a multiple of it`, `X = ${fmt(X)}: mọi trọng số trong khối được lưu dưới dạng bội của nó`) : tr('choosing the shared scale', 'chọn hệ số tỉ lệ dùng chung')} />
    </Frame>
  );
}

function beat4(p: number) {
  const t1 = ease(ramp(p, 0.04, 0.32));
  const g = ramp(p, 0.3, 0.48);
  const t2 = ease(ramp(p, 0.55, 0.85));
  const done = p >= 0.86;
  const elemOf = (i: number) => Q32.decoded[i] / X;
  const dots = V.map((v, i) => {
    const u = t2 > 0 ? lerp(Q32.scaled[i], elemOf(i), t2) : lerp(v, Q32.scaled[i], t1);
    return { u, i, lost: done && Q32.decoded[i] === 0, hot: i === AT };
  });
  const gridLines = [...GRID, ...GRID.slice(1).map((x) => -x)].filter((x) => x >= U0 && x <= U1);
  return (
    <Frame beat={4} data={{ divided: t1.toFixed(2), snapped: t2.toFixed(2), zeroed: done ? Q32.zeroed : 0, top: Q32.scaled[AT] }}>
      <text x={LX0} y={40} fontSize="12" style={{ fill: 'var(--ink-2)' }}>{t1 > 0 ? tr(`each weight divided by X = ${fmt(X)}`, `mỗi trọng số chia cho X = ${fmt(X)}`) : tr('the weights', 'các trọng số')}</text>
      {g > 0 ? (
        <g opacity={g}>
          {gridLines.map((x) => (
            <g key={x}>
              <line x1={ux(x)} x2={ux(x)} y1={96} y2={184} stroke="var(--ink)" strokeWidth={x === 0 ? 1.6 : 1} strokeDasharray={x === 0 ? undefined : '3 3'} />
              {x >= 0 ? <text x={ux(x)} y={90} fontSize="10.5" textAnchor="middle">{fmt(x, 1)}</text> : null}
            </g>
          ))}
          <text x={ux(6) + 8} y={104} fontSize="10.5">{tr('largest E2M1 value', 'giá trị E2M1 lớn nhất')}</text>
        </g>
      ) : null}
      <Line y={150}><Dots y={150} dots={dots} /></Line>
      <Headline
        text={
          done ? tr(`${Q32.zeroed} of 31 small weights snap to 0; the outlier lands on ${fmt(Q32.scaled[AT])}`, `${Q32.zeroed} trên 31 trọng số nhỏ bị kéo về 0; giá trị ngoại lai rơi vào ${fmt(Q32.scaled[AT])}`)
            : t2 > 0 ? tr('each one snaps to its nearest grid point', 'mỗi giá trị bị kéo về điểm lưới gần nhất')
              : g > 0 ? tr('the E2M1 grid: 0, 0.5, 1, 1.5, 2, 3, 4, 6', 'lưới E2M1: 0, 0.5, 1, 1.5, 2, 3, 4, 6')
                : tr(`dividing by ${fmt(X)}`, `chia cho ${fmt(X)}`)
        }
      />
    </Frame>
  );
}

function beat5(p: number) {
  const codeBytes = Array.from({ length: 16 }, (_, j) => (Q32.codes[2 * j] & 0xf) | ((Q32.codes[2 * j + 1] & 0xf) << 4));
  const mx = blockBytes(4, 32, 1);
  const bf = blockBytes(16, 32, 0);
  const shownMx = Math.min(mx, Math.ceil(ramp(p, 0.02, 0.4) * mx));
  const shownBf = Math.min(bf, Math.ceil(ramp(p, 0.45, 0.8) * bf));
  const hex = (b: number) => b.toString(16).padStart(2, '0');
  return (
    <Frame beat={5} data={{ mx, bf, shownMx, shownBf, scaleByte: hex(B.scaleCode) }}>
      <text x={LX0} y={36} fontSize="12" style={{ fill: 'var(--ink-2)' }}>{tr('MXFP4: 16 bytes of codes (two 4-bit codes per byte) and one scale byte', 'MXFP4: 16 byte mã (hai mã 4 bit mỗi byte) và một byte hệ số tỉ lệ')}</text>
      {Array.from({ length: shownMx }, (_, j) => {
        const isScale = j === 16;
        return (
          <g key={j}>
            <rect x={LX0 + j * 37 + (isScale ? 8 : 0)} y={48} width={33} height={30} rx="2"
              fill={isScale ? 'var(--ink)' : 'var(--cell)'} stroke="var(--ink)" />
            <text x={LX0 + j * 37 + 16.5 + (isScale ? 8 : 0)} y={68} fontSize="11.5" textAnchor="middle"
              style={{ fill: isScale ? 'var(--page)' : 'var(--ink)' }}>{isScale ? hex(B.scaleCode) : hex(codeBytes[j])}</text>
          </g>
        );
      })}
      {shownMx === mx ? <text x={LX0} y={100} fontSize="13"><Translated vi={<>{mx} byte cho 32 trọng số: {(mx * 8) / 32} bit mỗi trọng số</>}>{mx} bytes for 32 weights: {(mx * 8) / 32} bits per weight</Translated></text> : null}
      {shownBf > 0 ? (
        <g>
          <text x={LX0} y={146} fontSize="12" style={{ fill: 'var(--ink-2)' }}>{tr('the same 32 weights in BF16', 'cũng 32 trọng số đó ở BF16')}</text>
          {Array.from({ length: shownBf }, (_, j) => (
            <rect key={j} x={LX0 + (j % 32) * 20} y={156 + Math.floor(j / 32) * 22} width={18} height={18} rx="2" fill="var(--global-wash)" stroke="var(--global)" strokeWidth="0.8" />
          ))}
          {shownBf === bf ? <text x={LX0} y={222} fontSize="13"><Translated vi={<>{bf} byte: 16 bit mỗi trọng số</>}>{bf} bytes: 16 bits per weight</Translated></text> : null}
        </g>
      ) : null}
      <Headline text={shownBf === bf ? tr(`${mx} bytes against ${bf}: ${fmt(bf / mx, 2)} times fewer bytes to read`, `${mx} byte so với ${bf}: số byte phải đọc ít hơn ${fmt(bf / mx, 2)} lần`) : tr(`${shownMx} bytes`, `${shownMx} byte`)} />
    </Frame>
  );
}

function beat6(p: number) {
  const split = p >= 0.5;
  const q = split ? Q16 : Q32;
  const CX0 = 50;
  const cw = 20;
  const Y0 = 170;
  const ys = 130;
  const top = 64;
  const bottom = 280;
  const vy = (v: number) => Math.min(bottom, Math.max(top, Y0 - v * ys));
  const lost = q.zeroed;
  return (
    <Frame beat={6} data={{ k: q.opts.k, zeroed: lost, blocks: q.blocks.length, exps: q.blocks.map((b) => b.scaleExp).join(',') }}>
      <text x={LX0} y={30} fontSize="12" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi="mỗi trọng số (vòng tròn rỗng) và giá trị giải mã lại được (chấm); thanh nối giữa chúng là sai số">
          each weight (open circle) and what decodes back (dot); the bar between them is the error
        </Translated>
      </text>
      <line x1={CX0 - 10} x2={CX0 + 32 * cw} y1={Y0} y2={Y0} stroke="var(--rule)" />
      <text x={CX0 - 14} y={Y0 + 4} fontSize="10" textAnchor="end">0</text>
      <text x={CX0 - 14} y={Y0 - 0.5 * ys + 4} fontSize="10" textAnchor="end">0.5</text>
      <text x={CX0 - 14} y={Y0 + 0.5 * ys + 4} fontSize="10" textAnchor="end">−0.5</text>
      {V.map((v, i) => {
        const x = CX0 + i * cw + cw / 2;
        const d = q.decoded[i];
        const z = v !== 0 && d === 0;
        return (
          <g key={i}>
            <line x1={x} x2={x} y1={vy(v)} y2={vy(d)} stroke={z ? 'var(--fail)' : 'var(--ink-3)'} strokeWidth="2" />
            <circle cx={x} cy={vy(v)} r="4" fill="none" stroke="var(--global)" strokeWidth="1.3" />
            <circle cx={x} cy={vy(d)} r="3.2" fill={z ? 'var(--fail)' : 'var(--global)'} />
          </g>
        );
      })}
      <text x={CX0 + AT * cw + cw / 2} y={top - 8} fontSize="10.5" textAnchor="middle"><Translated vi={<>{fmt(V[AT])} → {fmt(q.decoded[AT])}, vượt khỏi thang đo</>}>{fmt(V[AT])} → {fmt(q.decoded[AT])}, off the scale</Translated></text>
      {q.blocks.map((b, j) => (
        <g key={j}>
          {j > 0 ? <line x1={CX0 + b.start * cw} x2={CX0 + b.start * cw} y1={48} y2={286} stroke="var(--ink)" strokeWidth="2" /> : null}
          <text x={CX0 + b.start * cw + 4} y={300} fontSize="11"><Translated vi={<>khối {j + 1}: X = 2^{b.scaleExp}, mất {b.zeroed}</>}>block {j + 1}: X = 2^{b.scaleExp}, {b.zeroed} lost</Translated></text>
        </g>
      ))}
      <Headline
        text={split
          ? tr(`two blocks of 16: the half without the outlier loses ${q.blocks[1].zeroed}`, `hai khối 16: nửa không có giá trị ngoại lai mất ${q.blocks[1].zeroed}`)
          : tr(`one block of 32: ${lost} of 31 small weights decode to 0`, `một khối 32: ${lost} trên 31 trọng số nhỏ giải mã thành 0`)}
      />
    </Frame>
  );
}

const beats: Beat[] = [
  { clip: 'mx-block-1', get label() { return tr('Thirty-two weights', 'Ba mươi hai trọng số'); }, draw: beat1 },
  { clip: 'mx-block-2', get label() { return tr('Find the largest', 'Tìm giá trị lớn nhất'); }, draw: beat2 },
  { clip: 'mx-block-3', get label() { return tr('Choose the scale', 'Chọn hệ số tỉ lệ'); }, draw: beat3 },
  { clip: 'mx-block-4', get label() { return tr('Divide and snap', 'Chia và kéo về lưới'); }, draw: beat4 },
  { clip: 'mx-block-5', get label() { return tr('Store', 'Lưu'); }, draw: beat5 },
  { clip: 'mx-block-6', get label() { return tr('Decode', 'Giải mã'); }, draw: beat6 },
];

export function MxBlockScene() {
  return <Scene chapter="appA" title={tr('Quantising one block', 'Lượng tử hóa một khối')} beats={beats} />;
}
