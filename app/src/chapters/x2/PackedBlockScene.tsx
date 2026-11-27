import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import {
  MIRROR_AT,
  OUTLIER_AT,
  SCENE_BLOCK,
  blockAddress,
  blockDotQ4_0,
  fp16FromBits,
  isAligned,
  quantTrace,
  quantizeQ4_0,
  sceneActivations,
  unpackTrace,
} from '../../sim/quant';
import { Translated, tr } from '../../lib/i18n';

// "One block, packed, shipped and unpacked": the seeded block the X2.2 stepper and the X2.3
// explorers use, through llama.cpp's Q4_0 reference rule (quant.ts). Six beats, one script
// each (content/x2/narration/packed-block-N.txt). Every number drawn comes from quant.ts.
export const SCENE_WORD = 2; // the 16-bit word beat 5 unpacks: bytes 4 and 5, weights 4, 5, 20, 21
const W = 720;
const H = 340;
const block = quantizeQ4_0(SCENE_BLOCK);
const trace = quantTrace('q4_0', SCENE_BLOCK);
const y = sceneActivations();
const word = unpackTrace('metal16', block.bytes.slice(2), SCENE_WORD, y);
const dot = blockDotQ4_0(block, y, SCENE_BLOCK);
const d16 = fp16FromBits(block.dBits);
const R = 0.9;
const X = (v: number) => 40 + ((v + R) / (2 * R)) * (W - 80);
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const MASKS = ['0x000F', '0x0F00', '0x00F0', '0xF000'];
const hex2 = (b: number) => b.toString(16).toUpperCase().padStart(2, '0');

/** Export for the browser check: what each beat must show. */
export const SCENE_FACTS = {
  d: block.d,
  d16,
  codes: block.codes,
  bytes: block.bytes,
  word: word[0].word,
  masked: word.slice(1, 5).map((s) => s.out[0].value),
  folded: dot.folded,
  dequantised: dot.dequantised,
  exact: dot.exact,
};

function Head({ children }: { children: ReactNode }) {
  return <text x={20} y={326} fontSize="13" fontWeight="600">{children}</text>;
}

function Line({ p, showScale }: { p: number; showScale: boolean }) {
  const shown = Math.ceil(ramp(p, 0, 0.35) * 32);
  const hot = p > 0.4;
  return (
    <g>
      <line x1={40} x2={W - 40} y1={120} y2={120} stroke="var(--rule)" />
      <line x1={X(0)} x2={X(0)} y1={112} y2={128} stroke="var(--ink-3)" />
      <text x={X(0)} y={142} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>0</text>
      {SCENE_BLOCK.slice(0, shown).map((v, i) => (
        <circle key={i} cx={X(v)} cy={100 - (i % 4) * 9} r={i === OUTLIER_AT && hot ? 6 : 3.5}
          fill={i === OUTLIER_AT && hot ? 'var(--ink)' : 'var(--page)'} stroke="var(--ink)" />
      ))}
      {hot ? <text x={X(SCENE_BLOCK[OUTLIER_AT])} y={60} fontSize="11" textAnchor="middle"><Translated vi={<>trọng số {OUTLIER_AT}: {SCENE_BLOCK[OUTLIER_AT].toFixed(2)}</>}>weight {OUTLIER_AT}: {SCENE_BLOCK[OUTLIER_AT].toFixed(2)}</Translated></text> : null}
      {showScale ? (
        <g opacity={ramp(p, 0.55, 0.75)}>
          <text x={20} y={200} fontSize="14">d = {SCENE_BLOCK[OUTLIER_AT].toFixed(1)} / −8 = {block.d.toPrecision(3)}</text>
          <text x={20} y={224} fontSize="14" style={{ fill: 'var(--global)' }}><Translated vi={<>lưu dưới dạng FP16: {d16.toPrecision(6)}</>}>stored as FP16: {d16.toPrecision(6)}</Translated></text>
        </g>
      ) : null}
    </g>
  );
}

function Codes({ p }: { p: number }) {
  const n = Math.ceil(ramp(p, 0.05, 0.7) * 32);
  const LX = (q: number) => 60 + q * 40;
  return (
    <g>
      {Array.from({ length: 16 }, (_, q) => (
        <g key={q}>
          <line x1={LX(q)} x2={LX(q)} y1={200} y2={214} stroke="var(--global)" strokeWidth={1.6} />
          <text x={LX(q)} y={230} fontSize="10" textAnchor="middle">{q}</text>
        </g>
      ))}
      <text x={LX(16)} y={230} fontSize="10" textAnchor="middle" style={{ fill: 'var(--fail)' }}>16?</text>
      <text x={20} y={250} fontSize="11" style={{ fill: 'var(--global)' }}><Translated vi="mã: giá trị = (mã − 8) × d">codes: value = (code − 8) × d</Translated></text>
      {SCENE_BLOCK.map((v, i) => {
        const done = i < n;
        const bad = trace.clipped[i] && p > 0.75;
        const q = done ? (bad ? trace.codes[i] : trace.rounded[i]) : null;
        return (
          <g key={i}>
            <circle cx={X(v)} cy={70 - (i % 4) * 9} r={3.5} fill={trace.clipped[i] && done ? 'var(--fail)' : 'var(--page)'} stroke="var(--ink)" />
            {q != null ? (
              <line x1={X(v)} y1={74 - (i % 4) * 9} x2={LX(q)} y2={198} stroke={trace.clipped[i] ? 'var(--fail)' : 'var(--ink-3)'} strokeWidth={trace.clipped[i] ? 1.8 : 0.6} />
            ) : null}
          </g>
        );
      })}
      {n > MIRROR_AT ? (
        <text x={W - 20} y={24} fontSize="11" textAnchor="end" style={{ fill: 'var(--fail)' }}>
          <Translated vi={<>trọng số {MIRROR_AT}: 0.8 × 10 + 8.5 = 16.5 → 16 → {p > 0.75 ? tr('clipped to 15', 'bị chặn về 15') : '?'}</>}>
            weight {MIRROR_AT}: 0.8 × 10 + 8.5 = 16.5 → 16 → {p > 0.75 ? tr('clipped to 15', 'bị chặn về 15') : '?'}
          </Translated>
        </text>
      ) : null}
    </g>
  );
}

function Pack({ p }: { p: number }) {
  const t = ramp(p, 0.15, 0.7);
  const C = 40;
  const x0 = 40;
  return (
    <g>
      <text x={x0} y={30} fontSize="11"><Translated vi="mã của các trọng số 0 đến 15 (nửa thấp) và 16 đến 31 (nửa cao)">codes of weights 0 to 15 (low halves) and 16 to 31 (high halves)</Translated></text>
      {block.codes.map((c, i) => {
        const j = i % 16;
        const hi = i >= 16;
        const sx = x0 + j * C;
        const sy = hi ? 74 : 44;
        const yy = sy + (160 - sy) * t;
        const wd = C - 4 - 18 * t;
        const xx = hi ? sx : sx + 18 * t;
        const share = j === OUTLIER_AT;
        return (
          <g key={i}>
            <rect x={xx} y={yy} width={wd} height={24} rx={2} fill={share ? 'var(--global-wash)' : 'var(--cell)'} stroke={share ? 'var(--global)' : 'var(--rule)'} />
            <text x={xx + wd / 2} y={yy + 16} fontSize="11" textAnchor="middle">{c.toString(16).toUpperCase()}</text>
          </g>
        );
      })}
      {t > 0.99 ? (
        <g>
          {block.bytes.slice(2).map((b, j) => (
            <text key={j} x={x0 + j * C + 18} y={202} fontSize="10" textAnchor="middle" style={{ fill: 'var(--global)' }}>0x{hex2(b)}</text>
          ))}
          <text x={x0} y={222} fontSize="10.5"><Translated vi={<>byte j = mã j | mã (j + 16) &lt;&lt; 4; byte {OUTLIER_AT} chứa các trọng số {OUTLIER_AT} và {MIRROR_AT}</>}>byte j = code j | code (j + 16) &lt;&lt; 4; byte {OUTLIER_AT} holds weights {OUTLIER_AT} and {MIRROR_AT}</Translated></text>
        </g>
      ) : null}
      <g opacity={ramp(p, 0.72, 0.9)}>
        <rect x={x0} y={248} width={18 * 4} height={16} fill="var(--global)" />
        <text x={x0 + 18 * 4 + 8} y={261} fontSize="11"><Translated vi="Q4_0: hệ số tỉ lệ 2 byte + 16 byte = 18 byte">Q4_0: 2-byte scale + 16 bytes = 18 bytes</Translated></text>
        <rect x={x0} y={274} width={128 * 4} height={16} fill="var(--global-wash)" stroke="var(--global)" />
        <text x={x0 + 8} y={287} fontSize="11"><Translated vi="float32: 32 × 4 = 128 byte">float32: 32 × 4 = 128 bytes</Translated></text>
      </g>
    </g>
  );
}

function Strip({ p }: { p: number }) {
  const B = 9; // pixels per byte
  const x0 = 24;
  const blocks = [0, 1, 2, 3];
  const loads = ramp(p, 0.35, 0.6);
  const halves = ramp(p, 0.7, 0.9);
  return (
    <g>
      <text x={x0} y={40} fontSize="11"><Translated vi="bốn khối nối tiếp nhau, mỗi khối 18 byte; vạch chia mỗi 4 byte">four blocks end to end, 18 bytes each; ticks every 4 bytes</Translated></text>
      {Array.from({ length: 73 }, (_, a) => (a % 4 === 0 ? <line key={a} x1={x0 + a * B} x2={x0 + a * B} y1={56} y2={110} stroke="var(--rule)" /> : null))}
      {blocks.map((i) => {
        const a = blockAddress(i);
        const ok = isAligned(a.qs, 4);
        return (
          <g key={i} data-block={i}>
            <rect x={x0 + a.d * B} y={66} width={2 * B} height={30} fill="var(--global)" />
            <rect x={x0 + a.qs * B} y={66} width={16 * B} height={30} fill="var(--global-wash)" stroke="var(--global)" />
            <text x={x0 + a.start * B + 9 * B} y={86} fontSize="10" textAnchor="middle"><Translated vi={<>khối {i}</>}>block {i}</Translated></text>
            <text x={x0 + a.qs * B} y={126} fontSize="10" textAnchor="middle" style={{ fill: ok ? 'var(--ok)' : 'var(--fail)' }}>{a.qs}</text>
            {loads > 0 ? (
              <g opacity={loads}>
                {[0, 1, 2, 3].map((k) => (
                  <rect key={k} x={x0 + (a.qs + 4 * k) * B + 1} y={138} width={4 * B - 2} height={14} fill={ok ? 'var(--ok-wash)' : 'var(--fail-wash)'} stroke={ok ? 'var(--ok)' : 'var(--fail)'} />
                ))}
              </g>
            ) : null}
            {halves > 0 ? (
              <g opacity={halves}>
                {Array.from({ length: 8 }, (_, k) => (
                  <rect key={k} x={x0 + (a.qs + 2 * k) * B + 1} y={178} width={2 * B - 2} height={14} fill="var(--ok-wash)" stroke="var(--ok)" />
                ))}
              </g>
            ) : null}
          </g>
        );
      })}
      {loads > 0 ? <text x={x0} y={166} fontSize="11" opacity={loads}><Translated vi="các lần nạp 4 byte của mã: chỉ căn chỉnh khi mã bắt đầu tại một bội số của 4">4-byte loads of the codes: aligned only where the codes start on a multiple of 4</Translated></text> : null}
      {halves > 0 ? <text x={x0} y={208} fontSize="11" opacity={halves}><Translated vi="các lần nạp 16 bit: luôn căn chỉnh, vì mọi khối đều bắt đầu tại một địa chỉ chẵn">16-bit loads: always aligned, because every block starts on an even address</Translated></text> : null}
    </g>
  );
}

function Word({ p }: { p: number }) {
  const u = word[0].word;
  const k = Math.min(4, Math.floor(ramp(p, 0.1, 0.85) * 4.999));
  const BX = (bit: number) => 40 + (15 - bit) * 30;
  const fields = word[0].fields;
  const div = [1, 256, 16, 4096];
  return (
    <g>
      <text x={40} y={34} fontSize="11"><Translated vi={<>word 16 bit {SCENE_WORD} = các byte {2 * SCENE_WORD} và {2 * SCENE_WORD + 1}: 0x{u.toString(16).toUpperCase().padStart(4, '0')}</>}>16-bit word {SCENE_WORD} = bytes {2 * SCENE_WORD} and {2 * SCENE_WORD + 1}: 0x{u.toString(16).toUpperCase().padStart(4, '0')}</Translated></text>
      {Array.from({ length: 16 }, (_, bit) => {
        const f = fields.find((x) => bit >= x.lo && bit <= x.hi)!;
        const active = k > 0 && fields.indexOf(f) < k && fields.indexOf(f) === k - 1;
        return (
          <g key={bit}>
            <rect x={BX(bit)} y={48} width={28} height={28} rx={2} fill={active ? 'var(--reg)' : 'var(--cell)'} stroke={active ? 'var(--reg)' : 'var(--rule)'} />
            <text x={BX(bit) + 14} y={67} fontSize="12" textAnchor="middle" style={{ fill: active ? 'var(--page)' : 'var(--ink)' }}>{(u >> bit) & 1}</text>
            <text x={BX(bit) + 14} y={90} fontSize="8.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{bit}</text>
          </g>
        );
      })}
      {fields.map((f, i) => (
        <text key={i} x={BX(f.hi) + 60} y={110} fontSize="10.5" textAnchor="middle">w{f.weight} = {f.value}</text>
      ))}
      {word.slice(1, 5).map((s, i) =>
        i < k ? (
          <text key={i} x={40} y={140 + i * 26} fontSize="12.5" data-masked={s.out[0].value}>
            <tspan style={{ fill: 'var(--reg)' }}>u &amp; {MASKS[i]}</tspan>{/* i18n-ignore: a code expression */}
            {` = ${s.out[0].value} = ${fields[i].value} × ${div[i]};   × y[${fields[i].weight}] / ${div[i]}  →  ${word[5].out[i].value.toPrecision(4)}`}
          </text>
        ) : null,
      )}
    </g>
  );
}

function Sum({ p }: { p: number }) {
  const a = ramp(p, 0.05, 0.3);
  const b = ramp(p, 0.4, 0.6);
  const c = ramp(p, 0.65, 0.85);
  return (
    <g fontSize="14">
      <g opacity={a}>
        <text x={40} y={60}>Σ code × y = {dot.sumQY.toFixed(3)}</text>
        <text x={260} y={60}>Σ y = {dot.sumY.toFixed(3)}</text>
        <text x={440} y={60}>d = {d16.toPrecision(6)}</text>
        <text x={40} y={96}>d × ({dot.sumQY.toFixed(3)}) − 8 × d × ({dot.sumY.toFixed(3)}) = <tspan fontWeight="650" style={{ fill: 'var(--reg)' }}>{dot.folded.toFixed(4)}</tspan></text>
      </g>
      <g opacity={b}>
        <text x={40} y={140}><Translated vi={<>tích vô hướng với các trọng số đã giải lượng tử hóa: <tspan fontWeight="650">{dot.dequantised.toFixed(4)}</tspan>  (như nhau, sai khác do làm tròn)</>}>dot with the dequantised weights: <tspan fontWeight="650">{dot.dequantised.toFixed(4)}</tspan>  (the same, to rounding)</Translated></text>
      </g>
      <g opacity={c}>
        <text x={40} y={184}><Translated vi={<>tích vô hướng với các trọng số float32: <tspan fontWeight="650">{dot.exact.toFixed(4)}</tspan></>}>dot with the float32 weights: <tspan fontWeight="650">{dot.exact.toFixed(4)}</tspan></Translated></text>
        <text x={40} y={212} style={{ fill: 'var(--fail)' }}><Translated vi={<>chênh lệch {(dot.dequantised - dot.exact).toFixed(4)}: sai số lượng tử hóa, không phải lỗi của kernel</>}>difference {(dot.dequantised - dot.exact).toFixed(4)}: the quantisation error, not the kernel's</Translated></text>
      </g>
    </g>
  );
}

const svg = (body: ReactNode, head: ReactNode) => (
  <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('One Q4_0 block, quantised, packed, stored and unpacked', 'Một khối Q4_0: lượng tử hóa, đóng gói, lưu trữ và giải nén')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
    {body}
    <Head>{head}</Head>
  </svg>
);

const beats: Beat[] = [
  { clip: 'packed-block-1', get label() { return tr('The scale', 'Hệ số tỉ lệ'); }, draw: (p) => svg(<Line p={p} showScale />, tr('find the signed extreme; d = extreme / −8', 'tìm giá trị cực trị có dấu; d = cực trị / −8')) },
  { clip: 'packed-block-2', get label() { return tr('The codes', 'Các mã'); }, draw: (p) => svg(<Codes p={p} />, tr('add 8.5, truncate, clip to 15', 'cộng 8.5, cắt phần lẻ, chặn về 15')) },
  { clip: 'packed-block-3', get label() { return tr('Two per byte', 'Hai mã mỗi byte'); }, draw: (p) => svg(<Pack p={p} />, tr('byte j: weight j low, weight j + 16 high', 'byte j: trọng số j ở nửa thấp, trọng số j + 16 ở nửa cao')) },
  { clip: 'packed-block-4', get label() { return tr('In memory', 'Trong bộ nhớ'); }, draw: (p) => svg(<Strip p={p} />, tr('every other block is 2 bytes off a 4-byte boundary', 'cứ cách một khối lại có một khối lệch 2 byte so với ranh giới 4 byte')) },
  { clip: 'packed-block-5', get label() { return tr('Unpack a word', 'Giải nén một word'); }, draw: (p) => svg(<Word p={p} />, tr('four masks, no shifts, activations pre-divided', 'bốn mặt nạ, không dịch bit, activation được chia trước')) },
  { clip: 'packed-block-6', get label() { return tr('The block sum', 'Tổng của khối'); }, draw: (p) => svg(<Sum p={p} />, tr('the 8 comes off once per block', 'số 8 được trừ đi một lần cho mỗi khối')) },
];

export function PackedBlockScene() {
  return <Scene chapter="x2" title={tr('One block, packed, shipped and unpacked', 'Một khối: đóng gói, vận chuyển và giải nén')} beats={beats} />;
}
