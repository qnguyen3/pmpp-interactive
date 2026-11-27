import { Scene, type Beat } from '../../components/Scene';
import { BOUND_Q8Q8, FP16_MAX, INT16_MAX, MAX_Q8_PRODUCT, narrowScene, sci, show, type Inner } from '../../sim/qdot';
import { Translated, tr as trText } from '../../lib/i18n';

// "Where the sum stops fitting": the worst-case Q8_0 × Q8_1 block added one product at a time
// into four accumulators, then a 4,096-wide row summed across its 128 blocks in half and in
// float. Every number is from src/sim/qdot.ts (innerSum, accumulate), FP16 rounding from
// fpsum.ts and minifloat.ts. Five beats (content/x2/narration/narrow-accumulator-N.txt).
const S = narrowScene();
const COLS: { id: Inner; name: string; limit: number; limitName: string }[] = [
  { id: 'int16', name: 'int16', limit: INT16_MAX, limitName: '32,767' }, // i18n-ignore: a type name
  { id: 'fp16', name: 'FP16', limit: FP16_MAX, limitName: '65,504' },
  { id: 'int32', name: 'int32', limit: 2 ** 31 - 1, limitName: '2,147,483,647' }, // i18n-ignore: a type name
  { id: 'fp32', name: 'FP32', limit: 3.4e38, get limitName() { return trText('about 3.4e38', 'khoảng 3.4e38'); } },
];
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const W = 720;
const H = 340;
const TOP = 66;
const BOT = 280;
const YMAX = 560000; // a little above 516,128
const y = (v: number) => BOT - (Math.min(Math.abs(v), YMAX) / YMAX) * (BOT - TOP);

function Columns({ k, focus, head }: { k: number; focus?: Inner[]; head: string }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={trText('Animation of four accumulators adding the same products', 'Hoạt ảnh bốn bộ cộng dồn cùng cộng những tích giống nhau')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <text x={20} y={22} fontSize="13" fontWeight="600" data-head="">{head}</text>
      <text x={20} y={40} fontSize="11.5" style={{ fill: 'var(--ink-2)' }} data-k={k}>
        <Translated vi={<>số tích đã cộng: {k} trên 32, mỗi tích 127 × 127 = {MAX_Q8_PRODUCT.toLocaleString('en-US')}; tổng chính xác {(k * MAX_Q8_PRODUCT).toLocaleString('en-US')}</>}>
          products added: {k} of 32, each 127 × 127 = {MAX_Q8_PRODUCT.toLocaleString('en-US')}; exact sum {(k * MAX_Q8_PRODUCT).toLocaleString('en-US')}
        </Translated>
      </text>
      <line x1={70} x2={700} y1={y(BOUND_Q8Q8)} y2={y(BOUND_Q8Q8)} stroke="var(--ink-3)" strokeDasharray="4 3" />
      <text x={20} y={y(BOUND_Q8Q8) - 2} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="cận">bound</Translated></text>
      <text x={20} y={y(BOUND_Q8Q8) + 10} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>516,128</text>
      {COLS.map((col, i) => {
        const tr = S.traces[col.id];
        const v = k === 0 ? 0 : tr.values[k - 1];
        const exact = k * MAX_Q8_PRODUCT;
        const wrapped = tr.first.wrap !== undefined && k >= tr.first.wrap;
        const inf = tr.first.inf !== undefined && k >= tr.first.inf;
        const dim = focus && !focus.includes(col.id) ? 0.35 : 1;
        const cx = 90 + i * 160;
        const bad = wrapped || inf;
        const status = k === 0 ? '' : inf ? 'infinity' : wrapped ? 'wrapped' : v === exact ? 'exact' : 'rounded';
        return (
          <g key={col.id} opacity={dim} data-col={col.id} data-value={String(v)} data-status={status}>
            <text x={cx + 40} y={BOT + 20} fontSize="12.5" fontWeight="600" textAnchor="middle">{col.name}</text>
            <rect x={cx} y={TOP} width={80} height={BOT - TOP} fill="var(--cell)" stroke="var(--rule)" />
            {col.limit < YMAX ? (
              <g>
                <line x1={cx - 6} x2={cx + 86} y1={y(col.limit)} y2={y(col.limit)} stroke="var(--fail)" strokeWidth="1.5" />
                <text x={cx + 84} y={y(col.limit) - 4} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--fail)' }}><Translated vi={<>max {col.limitName}</>}>max {col.limitName}</Translated></text>
              </g>
            ) : (
              <text x={cx + 40} y={TOP - 5} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>max {col.limitName}</>}>max {col.limitName}</Translated></text>
            )}
            {inf ? (
              <rect x={cx + 8} y={TOP} width={64} height={BOT - TOP} fill="var(--fail-wash)" stroke="var(--fail)" />
            ) : (
              <rect x={cx + 8} y={y(v)} width={64} height={BOT - y(v)}
                fill={bad ? 'var(--fail-wash)' : 'var(--reg-wash)'} stroke={bad ? 'var(--fail)' : 'var(--reg)'} />
            )}
            <text x={cx + 40} y={BOT + 38} fontSize="12" textAnchor="middle" style={{ fill: bad ? 'var(--fail)' : 'var(--ink)' }}>
              {k === 0 ? '0' : show(v)}
            </text>
            <text x={cx + 40} y={BOT + 54} fontSize="10.5" textAnchor="middle" style={{ fill: bad ? 'var(--fail)' : status === 'exact' ? 'var(--ok)' : 'var(--ink-2)' }}>
              <Translated vi={status === 'infinity' ? 'vô cực' : status === 'wrapped' ? 'bị quay vòng' : status === 'exact' ? 'chính xác' : status === 'rounded' ? 'bị làm tròn' : ''}>{status}</Translated>
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** Beat 5: the absolute error of the outer sum after each of 128 blocks, half against float (log scale). */
function Drift({ p }: { p: number }) {
  const n = Math.max(1, Math.round(ramp(p, 0.05, 0.75) * 128));
  const gx = 70;
  const gw = 600;
  const gy = 60;
  const gh = 200;
  const LO = -9;
  const HI = -1;
  const yl = (e: number) => gy + gh - ((Math.log10(Math.max(e, 10 ** LO)) - LO) / (HI - LO)) * gh;
  const path = (arr: number[]) => arr.slice(0, n).map((e, i) => `${i ? 'L' : 'M'}${gx + (i / 127) * gw},${yl(e)}`).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={trText('Error of a half and a float accumulator across a row', 'Sai số của một bộ cộng dồn half và một bộ cộng dồn float dọc theo một hàng')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <text x={20} y={22} fontSize="13" fontWeight="600" data-head=""><Translated vi="Một hàng rộng 4,096: 128 khối, int32 bên trong mỗi khối, tổng bên ngoài bằng half hoặc float">A 4,096-wide row: 128 blocks, int32 inside each, the outer sum in half or float</Translated></text>
      <text x={20} y={42} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="sai số tuyệt đối của tổng tích lũy sau mỗi khối (thang log), dữ liệu thực tế">absolute error of the running sum after each block (log scale), realistic data</Translated></text>
      <rect x={gx} y={gy} width={gw} height={gh} fill="var(--cell)" stroke="var(--rule)" />
      {[-9, -7, -5, -3, -1].map((e) => (
        <g key={e}>
          <line x1={gx} x2={gx + gw} y1={yl(10 ** e)} y2={yl(10 ** e)} stroke="var(--rule-soft)" />
          <text x={gx - 6} y={yl(10 ** e) + 3} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>1e{e}</text>
        </g>
      ))}
      <path d={path(S.outerHalf)} fill="none" stroke="var(--fail)" strokeWidth="2" />
      <path d={path(S.outerFloat)} fill="none" stroke="var(--reg)" strokeWidth="2" />
      <text x={gx + gw + 4} y={yl(S.outerHalf[n - 1]) + 4} fontSize="10.5" style={{ fill: 'var(--fail)' }}><Translated vi="half">half</Translated></text>
      <text x={gx + gw + 4} y={yl(S.outerFloat[n - 1]) + 4} fontSize="10.5" style={{ fill: 'var(--reg)' }}><Translated vi="float">float</Translated></text>
      <text x={gx} y={gy + gh + 18} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="khối 1">block 1</Translated></text>
      <text x={gx + gw} y={gy + gh + 18} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi="khối 128">block 128</Translated></text>
      <text x={20} y={306} fontSize="12" data-drift="">
        <Translated vi={<>sau {n} khối: half lệch {sci(S.outerHalf[n - 1])}, float lệch {sci(S.outerFloat[n - 1])}</>}>
          after {n} blocks: half off by {sci(S.outerHalf[n - 1])}, float off by {sci(S.outerFloat[n - 1])}
        </Translated>
      </text>
      <text x={20} y={328} fontSize="13" fontWeight="600" opacity={ramp(p, 0.8, 0.9)}>
        <Translated vi="Số nguyên bên trong một khối, float giữa các khối, không bao giờ dùng half cho một tổng.">
          Integer inside a block, float across blocks, never half for a sum.
        </Translated>
      </text>
    </svg>
  );
}

const beats: Beat[] = [
  { clip: 'narrow-accumulator-1', get label() { return trText('The worst case', 'Trường hợp xấu nhất'); }, draw: (p) => <Columns k={Math.round(ramp(p, 0.4, 0.9) * 2)} head={trText('The worst-case Q8_0 × Q8_1 block, one product at a time', 'Khối Q8_0 × Q8_1 trong trường hợp xấu nhất, từng tích một')} /> },
  { clip: 'narrow-accumulator-2', get label() { return trText('int16 wraps', 'int16 quay vòng'); }, draw: (p) => <Columns k={2 + Math.round(ramp(p, 0.2, 0.6) * 2)} focus={['int16']} head={trText('The 16-bit integer passes 32,767 on the third product', 'Số nguyên 16 bit vượt quá 32,767 ở tích thứ ba')} /> },
  { clip: 'narrow-accumulator-3', get label() { return trText('FP16 overflows', 'FP16 bị tràn'); }, draw: (p) => <Columns k={4 + Math.round(ramp(p, 0.3, 0.7) * 2)} focus={['fp16']} head={trText('FP16 rounds from the first product and overflows on the fifth', 'FP16 làm tròn ngay từ tích đầu tiên và bị tràn ở tích thứ năm')} /> },
  { clip: 'narrow-accumulator-4', get label() { return trText('Wide enough', 'Đủ rộng'); }, draw: (p) => <Columns k={6 + Math.round(ramp(p, 0.05, 0.7) * 26)} focus={['int32', 'fp32']} head={trText('int32 and float carry the block to 516,128, exactly', 'int32 và float đưa khối đến 516,128, chính xác')} /> },
  { clip: 'narrow-accumulator-5', get label() { return trText('Across the row', 'Dọc theo hàng'); }, draw: (p) => <Drift p={p} /> },
];

export function NarrowAccumulatorScene() {
  return <Scene chapter="x2" title={trText('Where the sum stops fitting', 'Nơi tổng không còn vừa')} beats={beats} />;
}
