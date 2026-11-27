import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { INT32_MAX, SCENE_LANES, byteOf, codesOf, hex, integerSceneRow, qsWord, show, tracePath } from '../../sim/qdot';
import { Translated, tr } from '../../lib/i18n';

// "The integer path, one block": llama.cpp's CUDA and HIP decode order for one Q4_0 block
// (notes/today/x2-facts.md D3), drawn from src/sim/qdot.ts on the block quant.ts quantises.
// Five beats, one narration clip each (content/x2/narration/integer-path-N.txt).
const row = integerSceneRow();
const c = row.case0;
const run = tracePath('C', c);
const q8 = run.q8!;
const codes = codesOf(c.block);
const dp = run.steps.filter((s) => s.kind === 'dp4a');
const corr = run.steps.find((s) => s.kind === 'correct')!;
const amaxAt = c.x.findIndex((v) => Math.abs(v) === q8.amax);

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const W = 720;
const H = 330;
const X0 = 30;
const CW = 20.5; // one activation column

function Frame({ children, head }: { children: ReactNode; head: ReactNode }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Animation of the integer path through one block', 'Hoạt ảnh đường số nguyên đi qua một khối')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <text x={X0} y={22} fontSize="13" fontWeight="600" data-head="">{head}</text>
      {children}
    </svg>
  );
}

/** Beat 1: the activations, their max, the scale, the bytes and s. */
function Activations({ p }: { p: number }) {
  const showMax = ramp(p, 0.15, 0.3);
  const coded = Math.floor(ramp(p, 0.35, 0.8) * 32);
  const showS = p > 0.82;
  const mid = 120;
  const k = 38 / q8.amax;
  return (
    <Frame head={tr("1. Quantise the block's 32 activations to int8 (Q8_1)", '1. Lượng tử hóa 32 activation của khối về int8 (Q8_1)')}>
      <line x1={X0} x2={X0 + 32 * CW} y1={mid} y2={mid} stroke="var(--rule)" />
      {c.x.map((v, j) => {
        const h = Math.abs(v) * k;
        const hot = showMax > 0 && j === amaxAt;
        return (
          <g key={j}>
            <rect x={X0 + j * CW + 3} y={v >= 0 ? mid - h : mid} width={CW - 6} height={Math.max(1, h)}
              fill={hot ? 'var(--global)' : 'var(--global-wash)'} stroke="var(--global)" strokeWidth="0.8" />
            {j < coded ? (
              <text x={X0 + j * CW + CW / 2} y={196} fontSize="10" textAnchor="middle" data-q8={j} style={{ fill: 'var(--reg)' }}>{q8.qs[j]}</text>
            ) : null}
          </g>
        );
      })}
      <text x={X0} y={58} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi="x, số thực (mỗi activation một cột)">x, float (one bar per activation)</Translated></text>
      <text x={X0} y={180} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi="q = roundf(x / d), một byte có dấu">q = roundf(x / d), a signed byte</Translated></text>
      <g opacity={showMax}>
        <text x={X0} y={236} fontSize="12.5" data-amax="">
          <Translated vi={<>max của |x| trên warp: amax = {show(q8.amax, 5)} (activation {amaxAt}); d = amax / 127 = {show(q8.dFloat, 5)}, lưu dưới dạng FP16 là {show(q8.d, 6)}</>}>
            warp max of |x|: amax = {show(q8.amax, 5)} (activation {amaxAt}); d = amax / 127 = {show(q8.dFloat, 5)}, stored in FP16 as {show(q8.d, 6)}
          </Translated>
        </text>
      </g>
      {showS ? (
        <text x={X0} y={262} fontSize="12.5" data-s="">
          <Translated vi={<>tổng các byte trên warp: {q8.sum}; s = d · {q8.sum} = {show(q8.s, 6)} ở FP16, giữ lại cho bước hiệu chỉnh</>}>
            warp sum of the bytes: {q8.sum}; s = d · {q8.sum} = {show(q8.s, 6)} in FP16, kept for the correction
          </Translated>
        </text>
      ) : null}
      <text x={X0} y={292} fontSize="11" style={{ fill: 'var(--ink-2)' }} data-shared="">
        <tspan x={X0}><Translated vi="Một lần cho mỗi token, mỗi khối.">Once per token per block.</Translated></tspan>
        <tspan x={X0} dy={15}><Translated vi="Mọi hàng trọng số đọc các activation này đều dùng chung nó.">Every weight row that reads these activations shares it.</Translated></tspan>
      </text>
    </Frame>
  );
}

/** Beat 2: one 32-bit word of weight codes, masked and shifted. */
function Word({ p }: { p: number }) {
  const v = qsWord(c.block, 0);
  const lowOn = ramp(p, 0.2, 0.4);
  const highOn = ramp(p, 0.55, 0.75);
  const bx = 120;
  const BW = 120;
  return (
    <Frame head={tr('2. One 32-bit word of weight codes: four bytes, eight nibbles', '2. Một word 32 bit chứa mã trọng số: bốn byte, tám nibble')}>
      <text x={X0} y={62} fontSize="12"><Translated vi={<>qs word 0 = {hex(v)} (các byte 0 đến 3 của khối)</>}>qs word 0 = {hex(v)} (bytes 0 to 3 of the block)</Translated></text>
      {[0, 1, 2, 3].map((k) => {
        const byte = byteOf(v, k) & 255;
        const x = bx + (3 - k) * BW;
        return (
          <g key={k}>
            <text x={x + BW / 2 - 2} y={88} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>byte {k}</>}>byte {k}</Translated></text>
            <rect x={x} y={96} width={BW / 2 - 4} height={30} fill={highOn ? 'var(--global)' : 'var(--global-wash)'} stroke="var(--global)" />
            <rect x={x + BW / 2} y={96} width={BW / 2 - 4} height={30} fill={lowOn && !highOn ? 'var(--global)' : 'var(--global-wash)'} stroke="var(--global)" />
            <text x={x + BW / 4 - 2} y={116} fontSize="12" textAnchor="middle" style={{ fill: highOn ? 'var(--page)' : 'var(--ink)' }}>{byte >> 4}</text>
            <text x={x + (3 * BW) / 4 - 2} y={116} fontSize="12" textAnchor="middle" style={{ fill: lowOn && !highOn ? 'var(--page)' : 'var(--ink)' }}>{byte & 15}</text>
            <text x={x + BW / 4 - 2} y={142} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>w{k + 16}</text>
            <text x={x + (3 * BW) / 4 - 2} y={142} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>w{k}</text>
          </g>
        );
      })}
      <g opacity={lowOn}>
        <text x={X0} y={186} fontSize="12.5" data-low="">
          <Translated vi={<>v &amp; 0x0F0F0F0F = {hex(v & 0x0f0f0f0f)}: trọng số 0 đến 3, mã {[0, 1, 2, 3].map((j) => codes[j]).join(', ')}</>}>
            v &amp; 0x0F0F0F0F = {hex(v & 0x0f0f0f0f)}: weights 0 to 3, codes {[0, 1, 2, 3].map((j) => codes[j]).join(', ')}
          </Translated>
        </text>
      </g>
      <g opacity={highOn}>
        <text x={X0} y={212} fontSize="12.5" data-high="">
          <Translated vi={<>(v &gt;&gt; 4) &amp; 0x0F0F0F0F = {hex((v >> 4) & 0x0f0f0f0f)}: trọng số 16 đến 19, mã {[16, 17, 18, 19].map((j) => codes[j]).join(', ')}</>}>
            (v &gt;&gt; 4) &amp; 0x0F0F0F0F = {hex((v >> 4) & 0x0f0f0f0f)}: weights 16 to 19, codes {[16, 17, 18, 19].map((j) => codes[j]).join(', ')}
          </Translated>
        </text>
      </g>
      <g opacity={ramp(p, 0.8, 0.9)}>
        <text x={X0} y={252} fontSize="12"><Translated vi="Các mã vẫn không dấu, từ 0 đến 15: ở đây không có gì trừ đi 8.">The codes stay unsigned, 0 to 15: nothing subtracts 8 here.</Translated></text>
        <text x={X0} y={272} fontSize="12"><Translated vi="Trọng số j và trọng số j + 16 dùng chung một byte, nên một mặt nạ và một lần dịch bit cho ra tám mã.">Weight j and weight j + 16 share a byte, so one mask and one shift give eight codes.</Translated></text>
      </g>
    </Frame>
  );
}

/** Beat 3: eight dp4a, the running integer against the int32 range (log scale). */
function Dp4a({ p }: { p: number }) {
  const k = Math.min(8, Math.floor(ramp(p, 0.05, 0.85) * 8) + 1);
  const cur = dp[k - 1];
  const gx = 470;
  const gw = 220;
  const logX = (v: number) => gx + (Math.log10(Math.max(1, Math.abs(v))) / Math.log10(INT32_MAX)) * gw;
  return (
    <Frame head={tr(`3. Eight four-way byte dot products; this is number ${k}`, `3. Tám tích vô hướng byte bốn chiều; đây là phép thứ ${k}`)}>
      {dp.map((s, i) => (
        <g key={i} opacity={i < k ? 1 : 0.25}>
          <text x={X0} y={58 + i * 24} fontSize="11.5" data-dp={i + 1}>
            dp4a {i + 1}: [{s.lanes!.map((l) => l.wv).join(', ')}] · [{s.lanes!.map((l) => l.xv).join(', ')}] → sumi = {i < k ? show(s.acc) : '…'}
          </text>
        </g>
      ))}
      <text x={gx} y={58} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="|sumi| so với phạm vi int32 (thang log)">|sumi| against the int32 range (log scale)</Translated></text>
      <line x1={gx} x2={gx + gw} y1={92} y2={92} stroke="var(--rule)" strokeWidth="6" />
      {[0, 3, 6, 9].map((e) => (
        <g key={e}>
          <line x1={gx + (e / Math.log10(INT32_MAX)) * gw} x2={gx + (e / Math.log10(INT32_MAX)) * gw} y1={84} y2={100} stroke="var(--ink-3)" />
          <text x={gx + (e / Math.log10(INT32_MAX)) * gw} y={114} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{e === 0 ? '1' : `10^${e}`}</text>
        </g>
      ))}
      <line x1={gx + gw} x2={gx + gw} y1={78} y2={106} stroke="var(--fail)" strokeWidth="2" />
      <text x={gx + gw} y={128} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--fail)' }}><Translated vi="max int32">int32 max</Translated></text>
      <circle cx={logX(cur.acc)} cy={92} r={6} fill="var(--reg)" />
      <text x={gx} y={152} fontSize="12" data-sumi=""><Translated vi={<>sumi = {show(cur.acc)} (một thanh ghi, chính xác)</>}>sumi = {show(cur.acc)} (a register, exact)</Translated></text>
      <text x={gx} y={176} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi="tối đa 60,960 với mọi khối Q4_0">at most 60,960 for any Q4_0 block</Translated></text>
      <text x={X0} y={290} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi="Nibble không dấu nhân byte có dấu: các tích là số nguyên chính xác, và tổng của chúng cũng vậy.">Unsigned nibbles times signed bytes: the products are exact integers, and so is their sum.</Translated></text>
    </Frame>
  );
}

/** Beat 4: the correction and the rescale, once per block. */
function Correct({ p }: { p: number }) {
  const v = corr.vals!;
  return (
    <Frame head={tr('4. Correct and rescale, once per block', '4. Hiệu chỉnh và nhân lại tỉ lệ, một lần cho mỗi khối')}>
      <text x={X0} y={70} fontSize="13" opacity={ramp(p, 0.05, 0.2)}>sumi · d8 = {show(v.sumi)} · {show(v.d8)} = {show(v.scaled)}</text>
      <text x={X0} y={102} fontSize="13" opacity={ramp(p, 0.25, 0.4)}>8 · s = 8 · {show(v.s)} = {show(v.off)}   <Translated vi="(loại bỏ số 8 mà mỗi nibble không dấu mang theo)">(removes the 8 every unsigned nibble carried)</Translated></text>
      <text x={X0} y={134} fontSize="13" opacity={ramp(p, 0.45, 0.6)}><Translated vi={<>hiệu = {show(corr.acc)}</>}>difference = {show(corr.acc)}</Translated></text>
      <text x={X0} y={174} fontSize="14" fontWeight="600" opacity={ramp(p, 0.65, 0.8)} data-result="">
        <Translated vi={<>kết quả khối = d4 · {show(corr.acc)} = {show(c.block.d)} · {show(corr.acc)} = {show(run.result)}</>}>
          block result = d4 · {show(corr.acc)} = {show(c.block.d)} · {show(corr.acc)} = {show(run.result)}
        </Translated>
      </text>
      <text x={X0} y={214} fontSize="12" opacity={ramp(p, 0.8, 0.9)}>
        <Translated vi={<>Ba phép nhân số thực và một phép trừ cho 32 trọng số. Đường số thực (B) cho {show(row.b[0])}.</>}>
          Three float multiplies and one subtraction for 32 weights. The float path (B) gave {show(row.b[0])}.
        </Translated>
      </text>
    </Frame>
  );
}

/** Beat 5: across the row, lanes then the warp; compared with the float path. */
function Row({ p }: { p: number }) {
  const shown = Math.floor(ramp(p, 0.05, 0.45) * row.c.length + 0.0001);
  const lanesOn = p > 0.45;
  const outOn = p > 0.65;
  const bw = 80;
  return (
    <Frame head={tr('5. Across the row: each lane adds its blocks in float, then the warp reduces', '5. Dọc theo hàng: mỗi lane cộng các khối của nó bằng số thực, rồi warp rút gọn')}>
      {row.c.map((v, i) => (
        <g key={i} opacity={i < shown ? 1 : 0.2}>
          <rect x={X0 + i * bw} y={40} width={bw - 6} height={50} fill="var(--cell)" stroke="var(--rule)" />
          <text x={X0 + i * bw + 5} y={54} fontSize="10" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>khối {i}</>}>block {i}</Translated></text>
          <text x={X0 + i * bw + 5} y={69} fontSize="11.5">{show(v, 5)}</text>
          <text x={X0 + i * bw + 5} y={84} fontSize="10" style={{ fill: 'var(--reg)' }}><Translated vi={<>→ lane {i % SCENE_LANES}</>}>→ lane {i % SCENE_LANES}</Translated></text>
        </g>
      ))}
      {lanesOn
        ? row.lanes.map((v, l) => (
            <g key={l}>
              <rect x={X0 + l * 160} y={110} width={150} height={30} fill="var(--reg-wash)" stroke="var(--reg)" />
              <text x={X0 + l * 160 + 6} y={130} fontSize="11.5"><Translated vi={<>lane {l}: {show(v, 6)}</>}>lane {l}: {show(v, 6)}</Translated></text>
            </g>
          ))
        : null}
      {outOn ? (
        <g>
          <text x={X0} y={182} fontSize="13" fontWeight="600" data-outc=""><Translated vi={<>đường số nguyên (C): {show(row.outC, 6)}</>}>integer path (C): {show(row.outC, 6)}</Translated></text>
          <text x={X0} y={206} fontSize="13" data-outb=""><Translated vi={<>đường số thực (B), cùng thứ tự: {show(row.outB, 6)}</>}>float path (B), same order: {show(row.outB, 6)}</Translated></text>
          <text x={X0} y={230} fontSize="12" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>chính xác, với các trọng số đã giải lượng tử hóa: {show(row.exact, 7)}</>}>exact, with the dequantised weights: {show(row.exact, 7)}</Translated></text>
          <text x={X0} y={262} fontSize="12.5" opacity={ramp(p, 0.75, 0.85)}>
            <Translated vi={<>chênh lệch {show(Math.abs(row.outC - row.outB), 3)}: do làm tròn activation về int8, ở đây là {((100 * Math.abs(row.outC - row.outB)) / Math.abs(row.outB)).toFixed(2)}%</>}>
              gap {show(Math.abs(row.outC - row.outB), 3)}: the activations' rounding to int8, {((100 * Math.abs(row.outC - row.outB)) / Math.abs(row.outB)).toFixed(2)}% here
            </Translated>
          </text>
        </g>
      ) : null}
    </Frame>
  );
}

const beats: Beat[] = [
  { clip: 'integer-path-1', get label() { return tr('Activations to int8', 'Activation về int8'); }, draw: (p) => <Activations p={p} /> },
  { clip: 'integer-path-2', get label() { return tr('Mask the codes', 'Lọc mã bằng mặt nạ'); }, draw: (p) => <Word p={p} /> },
  { clip: 'integer-path-3', get label() { return tr('Byte dot products', 'Tích vô hướng byte'); }, draw: (p) => <Dp4a p={p} /> },
  { clip: 'integer-path-4', get label() { return tr('Correct, rescale', 'Hiệu chỉnh, nhân lại tỉ lệ'); }, draw: (p) => <Correct p={p} /> },
  { clip: 'integer-path-5', get label() { return tr('Across the row', 'Dọc theo hàng'); }, draw: (p) => <Row p={p} /> },
];

export function IntegerPathScene() {
  return <Scene chapter="x2" title={tr('The integer path, one block', 'Đường số nguyên, một khối')} beats={beats} />;
}
