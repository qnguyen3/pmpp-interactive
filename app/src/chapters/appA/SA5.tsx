import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';
import { E2M1, E2M3, E3M2, E4M3, E5M2, maxFinite, minSubnormal, showBits, type Format } from '../../sim/minifloat';
import {
  ELEMENTS,
  ELEMENT_IDS,
  SCALE_RULES,
  WEIGHT_FORMATS,
  WEIGHT_FORMAT_IDS,
  bitsPerWeight,
  convert,
  decodeCeiling,
  elemGrid,
  namedFormat,
  opsPerByte,
  quantizeBlock,
  sampleBlock,
  weightBytes,
  type ElemId,
  type ScaleRule,
  type WeightFormat,
} from '../../sim/mxblock';
import { MxBlockScene } from './MxBlockScene';

const minus = (s: string) => s.replace(/-/g, '−');
const show = (v: number, d = 4) =>
  Number.isNaN(v) ? 'NaN' : !Number.isFinite(v) ? (v > 0 ? 'inf' : '−inf') : v === 0 ? (Object.is(v, -0) ? '−0' : '0') : minus(String(Number(v.toPrecision(d))));

// ---------------------------------------------------------------- decode ceiling
interface Hw {
  key: string;
  label: string;
  gbs: number;
  note: string;
}
// The book's example device (ch. 5, p. 94): used only as the book's example.
const BOOK_HW: Hw = {
  key: 'book',
  get label() {
    return tr("The book's example (H100, 3.35 TB/s)", 'Ví dụ của sách (H100, 3.35 TB/s)');
  },
  gbs: 3350,
  get note() {
    return tr('The bandwidth the book quotes for its example GPU in chapter 5 (p. 94).', 'Băng thông sách nêu cho GPU ví dụ của nó ở chương 5 (p. 94).');
  },
};

const P_MIN = 0.5e9;
const P_MAX = 70e9;
const paramsAt = (s: number) => Number((P_MIN * (P_MAX / P_MIN) ** (s / 1000)).toPrecision(2));
const sliderFor = (p: number) => Math.round((1000 * Math.log(p / P_MIN)) / Math.log(P_MAX / P_MIN));

function DecodeCeilingBench() {
  const { reports } = useReports();
  const measured: Hw[] = useMemo(
    () =>
      realBackends(reports).flatMap((b) => {
        const r = benchRows(reports[b], 'ch05/lab1_roofline').find((x) => x.name === 'roofline');
        const v = num(r?.bandwidth_gb_per_s);
        return v
          ? [
              {
                key: b,
                get label() {
                  return tr(`Yours: ${String(reports[b].device.name ?? b)} (${b}), ${fmt(v)} GB/s`, `Của bạn: ${String(reports[b].device.name ?? b)} (${b}), ${fmt(v)} GB/s`);
                },
                gbs: v,
                get note() {
                  return tr(
                    `Measured by lab 5.1 on ${reports[b].generated.slice(0, 10)}: a copy kernel, not the vendor's peak.`,
                    `Đo bằng lab 5.1 ngày ${reports[b].generated.slice(0, 10)}: một kernel sao chép, không phải đỉnh của nhà cung cấp.`,
                  );
                },
              },
            ]
          : [];
      }),
    [reports],
  );
  const options = [BOOK_HW, ...measured];
  const [hwKey, setHwKey] = useState('book');
  const [custom, setCustom] = useState(500);
  const [s, setS] = useState(sliderFor(7e9));
  const [sel, setSel] = useState<WeightFormat>('mxfp4');
  const params = paramsAt(s);
  const hw: Hw = hwKey === 'custom' ? { key: 'custom', label: tr('Custom', 'Tùy chỉnh'), gbs: custom, note: tr('A bandwidth you typed.', 'Một băng thông bạn đã nhập.') } : (options.find((o) => o.key === hwKey) ?? BOOK_HW);
  const rows = WEIGHT_FORMAT_IDS.map((f) => {
    const bits = bitsPerWeight(f);
    return { f, bits, bytes: weightBytes(params, bits), ai: opsPerByte(bits), ceil: decodeCeiling(params, bits, hw.gbs) };
  });
  const top = Math.max(...rows.map((r) => r.ceil));
  const bf16 = rows.find((r) => r.f === 'bf16')!;
  const cur = rows.find((r) => r.f === sel)!;
  return (
    <Bench
      title={tr('Decode ceiling by weight format', 'Trần decode theo định dạng trọng số')}
      task={tr(
        'Pick a model size and a device, then compare the formats. What sets the ratio between two rows?',
        'Chọn một kích thước mô hình và một thiết bị, rồi so sánh các định dạng. Điều gì quyết định tỉ số giữa hai hàng?',
      )}
      foot={
        <Translated vi={<>Trần = băng thông ÷ số byte trọng số mỗi token, ở batch 1. {hw.note} Không con số thiết bị nào ở đây là giá trị kỳ vọng cho thiết bị của bạn.</>}>
          <>Ceiling = bandwidth ÷ bytes of weights per token, at batch 1. {hw.note} No device figure here is an expected value for yours.</>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          <Translated vi={<>Tham số: {fmt(params / 1e9)} tỷ</>}>
            <>Parameters: {fmt(params / 1e9)} billion</>
          </Translated>
          <input type="range" min="0" max="1000" value={s} aria-label={tr('parameters', 'tham số')} onChange={(e) => setS(+e.target.value)} />
        </label>
        <label className="field">
          <Translated vi="Băng thông thiết bị">Device bandwidth</Translated>
          <select value={hwKey} onChange={(e) => setHwKey(e.target.value)} aria-label={tr('device', 'thiết bị')}>
            {options.map((o) => (
              <option key={o.key} value={o.key}>{o.label}</option>
            ))}
            <option value="custom">{tr('Custom', 'Tùy chỉnh')}</option>
          </select>
        </label>
        {hwKey === 'custom' ? (
          <label className="field">
            GB/s
            <input type="number" min="1" value={custom} onChange={(e) => setCustom(Math.max(1, +e.target.value || 1))} />
          </label>
        ) : null}
        <label className="field">
          <Translated vi="Định dạng trọng số">Weight format</Translated>
          <select value={sel} onChange={(e) => setSel(e.target.value as WeightFormat)} aria-label={tr('weight format', 'định dạng trọng số')}>
            {WEIGHT_FORMAT_IDS.map((f) => (
              <option key={f} value={f}>{WEIGHT_FORMATS[f].label}</option>
            ))}
          </select>
        </label>
      </div>
      <p className="ceiling-head" style={{ margin: '0 0 0.2rem', fontSize: '1.45rem', fontWeight: 600 }} data-ceiling={cur.ceil}>
        <Translated vi={<>tối đa {fmt(cur.ceil)} token mỗi giây</>}>
          <>{fmt(cur.ceil)} tokens per second, at most</>
        </Translated>
      </p>
      <Translated
        vi={
          <p className="muted" style={{ margin: '0 0 0.8rem' }}>
            {WEIGHT_FORMATS[sel].label}: {fmt(cur.bits)} bit mỗi trọng số{WEIGHT_FORMATS[sel].extra ? ` (${WEIGHT_FORMATS[sel].extra})` : ''}, đọc {fmt(cur.bytes / 1e9)} GB
            trọng số mỗi token, {fmt(cur.ai)} FLOP/B, ở {fmt(hw.gbs)} GB/s.
          </p>
        }
      >
        <p className="muted" style={{ margin: '0 0 0.8rem' }}>
          {WEIGHT_FORMATS[sel].label}: {fmt(cur.bits)} bits per weight{WEIGHT_FORMATS[sel].extra ? ` (${WEIGHT_FORMATS[sel].extra})` : ''}, {fmt(cur.bytes / 1e9)} GB
          of weights read per token, {fmt(cur.ai)} FLOP/B, at {fmt(hw.gbs)} GB/s.
        </p>
      </Translated>
      <table className="data ceiling-table" style={{ margin: 0 }}>
        <thead>
          <tr>
            <th>
              <Translated vi="Định dạng">Format</Translated>
            </th>
            <th className="num">
              <Translated vi="Bit mỗi trọng số">Bits per weight</Translated>
            </th>
            <th className="num">
              <Translated vi="Trọng số, GB">Weights, GB</Translated>
            </th>
            <th className="num">FLOP/B</th>
            <th className="num">
              <Translated vi="Trần, token/s">Ceiling, tokens/s</Translated>
            </th>
            <th className="num">× BF16</th>{/* i18n-ignore: a ratio label */}
            <th style={{ width: '22%' }}></th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.f} data-format={r.f} style={r.f === sel ? { background: 'var(--rule-soft)' } : undefined}>
              <td>{WEIGHT_FORMATS[r.f].label}</td>
              <td className="num">{fmt(r.bits)}</td>
              <td className="num">{fmt(r.bytes / 1e9)}</td>
              <td className="num">{fmt(r.ai)}</td>
              <td className="num">{fmt(r.ceil)}</td>
              <td className="num">{fmt(r.ceil / bf16.ceil)}</td>
              <td>
                <span style={{ display: 'block', height: '0.7rem', width: `${(100 * r.ceil) / top}%`, background: 'var(--ink-3)', borderRadius: 2 }} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- near the top of E4M3
const TOP_CASES: { x: number; f: Format }[] = [
  { x: 448, f: E4M3 },
  { x: 460, f: E4M3 },
  { x: 464, f: E4M3 },
  { x: 470, f: E4M3 },
  { x: 57344, f: E5M2 },
  { x: 61440, f: E5M2 },
];

const big = (v: number) => (Number.isFinite(v) ? v.toLocaleString('en-US') : show(v));

function TopReveal({ v }: { v: number }) {
  return (
    <>
      <Translated
        vi={
          <p>
            <b>448</b>, và ở đây chế độ không bão hòa cũng cho 448. Gần đỉnh của E4M3, các giá trị cách nhau 32 (số mũ 8, ba
            bit mantissa: 2<sup>8−3</sup>), nên 460 nằm giữa 448 và một giá trị 480 giả định và bị làm tròn xuống trước khi
            bất kỳ quy tắc tràn nào được áp dụng.{' '}
            {v === 448 ? 'Đúng như bạn đã tính.' : v === 460 ? 'Bản thân 460 không phải là một giá trị E4M3.' : v > 448 ? 'Không có gì lớn hơn 448 tồn tại trong E4M3.' : 'Hãy làm tròn về giá trị gần nhất; 448 là gần nhất.'}
          </p>
        }
      >
        <p>
          <b>448</b>, and here non-saturating mode gives 448 too. Near the top of E4M3 the values are 32 apart (exponent 8,
          three mantissa bits: 2<sup>8−3</sup>), so 460 sits between 448 and a would-be 480 and rounds down before any
          overflow rule applies.{' '}
          {v === 448 ? 'That is what you had.' : v === 460 ? '460 itself is not an E4M3 value.' : v > 448 ? 'Nothing above 448 exists in E4M3.' : 'Round to the nearest value; 448 is the closest.'}
        </p>
      </Translated>
      <table className="data" style={{ margin: '0.4rem 0' }} data-reveal="e4m3">
        <thead>
          <tr>
            <th>
              <Translated vi="Đầu vào">Input</Translated>
            </th>
            <th>
              <Translated vi="Sang">To</Translated>
            </th>
            <th>
              <Translated vi="Bão hòa">Saturating</Translated>
            </th>
            <th>
              <Translated vi="Không bão hòa">Non-saturating</Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          {TOP_CASES.map(({ x, f }) => {
            const s = convert(x, f, 'saturate');
            const n = convert(x, f, 'ieee');
            return (
              <tr key={x} data-input={x}>
                <td className="num">{x.toLocaleString('en-US')}</td>
                <td>{f.name.replace('FP8 ', '')}</td>
                <td data-sat={s.value}>{big(s.value)} <span className="muted mono">{showBits(f, s.code)}</span></td>
                <td data-nosat={String(n.value)} className={Number.isFinite(n.value) ? '' : 't-fail'}>
                  {big(n.value)} <span className="muted mono">{n.code >= 0 ? showBits(f, n.code) : ''}</span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Vượt quá 464, điểm chính giữa, 470 được làm tròn lên 480, giá trị mà E4M3 không có: chế độ bão hòa kẹp về 448,
            chế độ không bão hòa cho NaN vì E4M3 không có vô cực. E5M2 hoạt động như IEEE: 61,440 nằm chính giữa giá trị lớn
            nhất của nó, 57,344, và lũy thừa kế tiếp của hai, trường hợp hòa được làm tròn lên, và nó tràn thành vô cực.
          </p>
        }
      >
        <p>
          Past 464, the halfway point, 470 rounds up to 480, which E4M3 does not have: saturating mode clamps to 448,
          non-saturating mode gives NaN because E4M3 has no infinity. E5M2 behaves like IEEE: 61,440 is halfway between
          its largest value, 57,344, and the next power of two, the tie rounds up, and it overflows to infinity.
        </p>
      </Translated>
    </>
  );
}

// ---------------------------------------------------------------- one MX block
const OUTLIER_AT = 7;
const ELEM_LABEL: Record<ElemId, string> = { E2M1: 'FP4 E2M1', E2M3: 'FP6 E2M3', E3M2: 'FP6 E3M2', E4M3: 'FP8 E4M3', E5M2: 'FP8 E5M2' }; // i18n-ignore: format names

function MxBlockBench() {
  const [seed, setSeed] = useState(1);
  const [outlier, setOutlier] = useState(0);
  const [elem, setElem] = useState<ElemId>('E2M1');
  const [k, setK] = useState(32);
  const [rule, setRule] = useState<ScaleRule>('ocp-floor');
  const values = useMemo(() => sampleBlock(seed, 32, 0.25, outlier, OUTLIER_AT), [seed, outlier]);
  const q = useMemo(() => quantizeBlock(values, { elem, k, scale: rule }), [values, elem, k, rule]);
  const f = ELEMENTS[elem];
  const named = namedFormat({ elem, k, scale: rule });
  const nonzero = values.filter((v) => v !== 0).length;

  // Log2 magnitude axis, with a lane for zero underneath.
  const W = 720;
  const H = 300;
  const CX0 = 64;
  const cw = 20;
  const L0 = -12;
  const L1 = 7;
  const yTop = 18;
  const yBot = 236;
  const zeroY = 262;
  const ly = (a: number) => {
    if (a === 0) return zeroY;
    const l = Math.max(L0, Math.min(L1, Math.log2(a)));
    return yBot - ((l - L0) / (L1 - L0)) * (yBot - yTop);
  };
  const smallGrid = elemGrid(f).length <= 32;
  return (
    <Bench
      title={tr('One MX block', 'Một khối MX')}
      task={tr(
        'Inject an outlier, then try to win back the small weights: smaller blocks, another scale rule, wider elements.',
        'Chèn một giá trị ngoại lai, rồi thử giành lại các trọng số nhỏ: khối nhỏ hơn, quy tắc tỉ lệ khác, phần tử rộng hơn.',
      )}
      foot={
        <Translated
          vi={
            <>
              Độ lớn trên thang log; dấu được lưu chính xác và không vẽ ra. Dải xám là những gì khối biểu diễn được (tỉ lệ của
              nó nhân với giá trị nhỏ nhất và lớn nhất của định dạng phần tử); mọi thứ dưới đường nét đứt được làm tròn về 0.
              32 trọng số lấy từ một phân phối chuẩn, σ = 0.25.
            </>
          }
        >
          <>
            Magnitudes on a log scale; signs are stored exactly and not drawn. The grey band is what the block can represent
            (its scale times the element format's smallest and largest values); anything below the dashed line rounds to 0.
            32 weights drawn from a normal distribution, σ = 0.25.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi={<>Giá trị ngoại lai ở phần tử {OUTLIER_AT}: {outlier === 0 ? 'không có' : outlier}</>}>
            <>Outlier at element {OUTLIER_AT}: {outlier === 0 ? 'none' : outlier}</>
          </Translated>
          <input type="range" min="0" max="100" value={outlier} aria-label={tr('outlier', 'giá trị ngoại lai')} onChange={(e) => setOutlier(+e.target.value)} />
        </label>
        <button onClick={() => setSeed((x) => x + 1)}><Translated vi="Trọng số mới">New weights</Translated></button>
        <label className="field">
          <Translated vi="Định dạng phần tử">Element format</Translated>
          <select value={elem} onChange={(e) => setElem(e.target.value as ElemId)} aria-label={tr('element format', 'định dạng phần tử')}>
            {ELEMENT_IDS.map((id) => (
              <option key={id} value={id}>{ELEM_LABEL[id]}</option>
            ))}
          </select>
        </label>
        <label className="field">
          <Translated vi="Số phần tử mỗi tỉ lệ">Elements per scale</Translated>
          <span className="seg">
            {[32, 16].map((n) => (
              <button key={n} aria-pressed={k === n} onClick={() => setK(n)}>{n}</button>
            ))}
          </span>
        </label>
        <label className="field">
          <Translated vi="Quy tắc tỉ lệ">Scale rule</Translated>
          <select value={rule} onChange={(e) => setRule(e.target.value as ScaleRule)} aria-label={tr('scale rule', 'quy tắc tỉ lệ')}>
            {(Object.keys(SCALE_RULES) as ScaleRule[]).map((r) => (
              <option key={r} value={r}>{SCALE_RULES[r]}</option>
            ))}
          </select>
        </label>
      </div>
      {/* On a phone the drawing keeps a readable width and scrolls sideways in its own frame. */}
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Weights and their decoded values', 'Các trọng số và giá trị giải mã của chúng')} style={{ width: '100%', minWidth: 560, maxWidth: W, display: 'block' }}
          data-zeroed={q.zeroed} data-clamped={q.clamped} data-k={k} data-elem={elem} data-rule={rule} data-seed={seed} data-outlier={outlier}
          data-scales={q.blocks.map((b) => b.scale).join(',')}>
          {[-12, -8, -4, 0, 4].map((e) => (
            <g key={e}>
              <line x1={CX0 - 6} x2={CX0 + 32 * cw} y1={ly(2 ** e)} y2={ly(2 ** e)} stroke="var(--rule-soft)" />
              <text x={CX0 - 10} y={ly(2 ** e) + 4} fontSize="11" textAnchor="end">{e === 0 ? '1' : `2^${e}`}</text>
            </g>
          ))}
          <text x={CX0 - 10} y={zeroY + 4} fontSize="11" textAnchor="end">0</text>
          <line x1={CX0 - 6} x2={CX0 + 32 * cw} y1={zeroY} y2={zeroY} stroke="var(--rule)" />
          {q.blocks.map((b, j) => {
            const x0 = CX0 + b.start * cw;
            const x1 = x0 + b.size * cw;
            const hi = ly(b.scale * maxFinite(f));
            const lo = ly(b.scale * minSubnormal(f)!);
            const half = ly((b.scale * minSubnormal(f)!) / 2);
            return (
              <g key={j}>
                <rect x={x0} y={hi} width={x1 - x0} height={Math.max(0, lo - hi)} fill="var(--rule-soft)" />
                {smallGrid
                  ? elemGrid(f).slice(1).map((g) => (
                      <line key={g} x1={x0} x2={x1} y1={ly(b.scale * g)} y2={ly(b.scale * g)} stroke="var(--rule)" strokeWidth="0.7" />
                    ))
                  : null}
                <line x1={x0} x2={x1} y1={half} y2={half} stroke="var(--fail)" strokeDasharray="4 3" />
                {j > 0 ? <line x1={x0} x2={x0} y1={yTop - 8} y2={zeroY + 10} stroke="var(--ink)" strokeWidth="2" /> : null}
                <text x={x0 + 4} y={yTop - 4} fontSize="11">X = {b.scaleExp != null ? `2^${b.scaleExp}` : show(b.scale, 4)}</text>
              </g>
            );
          })}
          {values.map((v, i) => {
            const x = CX0 + i * cw + cw / 2;
            const d = q.decoded[i];
            const lost = v !== 0 && d === 0;
            const clamped = Math.abs(q.scaled[i]) > maxFinite(f);
            const bad = lost || clamped;
            return (
              <g key={i} data-i={i} data-lost={lost ? 1 : 0}>
                <line x1={x} x2={x} y1={ly(Math.abs(v))} y2={ly(Math.abs(d))} stroke={bad ? 'var(--fail)' : 'var(--ink-3)'} strokeWidth="1.6" />
                <circle cx={x} cy={ly(Math.abs(v))} r="4" fill="none" stroke="var(--ink-2)" strokeWidth="1.3" />
                <circle cx={x} cy={ly(Math.abs(d))} r="3" fill={bad ? 'var(--fail)' : 'var(--ink)'} />
              </g>
            );
          })}
          <text x={CX0} y={H - 6} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{tr('element 0 … 31: open circle = weight, dot = what decodes back', 'phần tử 0 … 31: vòng tròn rỗng = trọng số, chấm = giá trị giải mã ra')}</text>
        </svg>
      </div>
      <div className="cols" style={{ marginTop: '0.8rem' }}>
        <div className="panel">
          <p style={{ margin: '0 0 0.5rem' }}>
            <b>{named ?? tr('Not a documented format', 'Không phải định dạng có tài liệu')}</b>
            <span className="muted"><Translated vi={<> · {fmt(q.bitsPerWeight)} bit mỗi trọng số</>}><> · {fmt(q.bitsPerWeight)} bits per weight</></Translated></span>
          </p>
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr>
                <th>
                  <Translated vi="Khối">Block</Translated>
                </th>
                <th className="num">max |v|</th>{/* i18n-ignore: a formula */}
                <th>
                  <Translated vi="Tỉ lệ X">Scale X</Translated>
                </th>
                <th className="num">
                  <Translated vi="Byte">Byte</Translated>
                </th>
                <th className="num">
                  <Translated vi="Về 0">To 0</Translated>
                </th>
                <th className="num">
                  <Translated vi="Bị kẹp">Clamped</Translated>
                </th>
              </tr>
            </thead>
            <tbody>
              {q.blocks.map((b, j) => (
                <tr key={j} data-block={j}>
                  <td>{j + 1}</td>
                  <td className="num">{show(b.amax, 3)}</td>
                  <td>{b.scaleExp != null ? <>2<sup>{b.scaleExp}</sup> = {show(b.scale, 4)}</> : show(b.scale, 4)}</td>
                  <td className="num mono">{b.scaleCode.toString(16).padStart(2, '0')}</td>
                  <td className={'num' + (b.zeroed ? ' t-fail' : '')}>{b.zeroed}</td>
                  <td className={'num' + (b.clamped ? ' t-fail' : '')}>{b.clamped}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <p style={{ margin: 0 }} data-summary>
            <span className={q.zeroed ? 'chip fail' : 'chip ok'}><Translated vi={<>{q.zeroed} trên {nonzero} bị mất về 0</>}><>{q.zeroed} of {nonzero} lost to 0</></Translated></span>{' '}
            {q.clamped ? <span className="chip fail"><Translated vi={<>{q.clamped} bị kẹp</>}><>{q.clamped} clamped</></Translated></span> : null}
          </p>
          <Translated
            vi={
              <p className="muted" style={{ margin: '0.5rem 0 0' }}>
                Sai số RMS bằng {show(100 * q.rmsRel, 3)}% RMS của khối; {show(100 * q.rmsRelRest, 3)}% với 31 trọng số không
                phải lớn nhất.
              </p>
            }
          >
            <p className="muted" style={{ margin: '0.5rem 0 0' }}>
              RMS error {show(100 * q.rmsRel, 3)}% of the block's RMS; {show(100 * q.rmsRelRest, 3)}% for the 31 weights other than the
              largest.
            </p>
          </Translated>
        </div>
      </div>
      <details style={{ marginTop: '0.8rem' }}>
        <summary>
          <Translated vi="Cả 32 phần tử">All 32 elements</Translated>
        </summary>
        <table className="data" style={{ margin: '0.4rem 0 0' }} data-elements>
          <thead>
            <tr>
              <th className="num">i</th>
              <th className="num">
                <Translated vi="trọng số v">weight v</Translated>
              </th>
              <th className="num">v / X</th>
              <th>
                <Translated vi="mã được lưu">stored code</Translated>
              </th>
              <th className="num">
                <Translated vi="giải mã thành">decodes to</Translated>
              </th>
              <th className="num">
                <Translated vi="sai số">error</Translated>
              </th>
            </tr>
          </thead>
          <tbody>
            {values.map((v, i) => (
              <tr key={i}>
                <td className="num">{i}</td>
                <td className="num">{show(v, 4)}</td>
                <td className="num">{show(q.scaled[i], 4)}</td>
                <td className="mono">{showBits(f, q.codes[i])}</td>
                <td className={'num' + (v !== 0 && q.decoded[i] === 0 ? ' t-fail' : '')}>{show(q.decoded[i], 4)}</td>
                <td className="num">{show(q.err[i], 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSA5: Record<string, Question> = {
  'appA/A.5/q-h100': {
    kind: 'choice',
    id: 'appA/A.5/q-h100',
    prompt: (
      <Translated
        vi={
          <>
            Sách đặt 3,958 TFLOPS FP8 trên tensor core của H100 cạnh 989 TFLOPS mà sách gọi là độ chính xác đơn trên tensor
            core. Theo chính bảng của NVIDIA, con số 989 là gì?
          </>
        }
      >
        <>
          The book sets the H100's 3,958 TFLOPS for FP8 on tensor cores against 989 TFLOPS it calls single precision on
          tensor cores. What is the 989, by NVIDIA's own table?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Thông lượng ma trận IEEE FP32 trên tensor core, dạng dày (dense).">IEEE FP32 matrix throughput on tensor cores, dense.</Translated>,
        why: (
          <Translated vi="Hàng của NVIDIA là TF32, và nó mang chú thích 'with sparsity' (với độ thưa).">
            {"NVIDIA's row is TF32, and it carries the 'with sparsity' footnote."}
          </Translated>
        ),
      },
      { label: <Translated vi="Thông lượng tensor core TF32 với độ thưa (sparsity).">TF32 tensor-core throughput with sparsity.</Translated>, correct: true },
      {
        label: <Translated vi="Thông lượng FP32 của các lõi thông thường, không dùng tensor core.">FP32 throughput of the ordinary cores, no tensor cores.</Translated>,
        why: (
          <Translated vi="NVIDIA liệt kê con số đó là 67 TFLOPS; 989 là một hàng của tensor core.">
            NVIDIA lists that as 67 TFLOPS; 989 is a tensor-core row.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Thông lượng tensor core FP16 với độ thưa.">FP16 tensor-core throughput with sparsity.</Translated>,
        why: <Translated vi="FP16 và BF16 là hàng 1,979.">FP16 and BF16 are the 1,979 row.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Cả hai con số của sách đều là các hàng "with sparsity" của NVIDIA, và 989 là TF32: phạm vi số mũ của FP32 với ít
            nhất 10 bit mantissa, một định dạng đầu vào của tensor core chứ không phải độ chính xác đơn IEEE. Tỉ lệ bốn trên
            một đúng giữa hai hàng thưa, nên ý của sách vẫn đứng vững: ít bit hơn cho mỗi giá trị, nhiều phép toán hơn mỗi
            giây.
          </p>
        }
      >
        <p>
          Both of the book's numbers are NVIDIA's "with sparsity" rows, and 989 is TF32: FP32's exponent range with at
          least 10 mantissa bits, a tensor-core input format rather than IEEE single precision. The four-to-one ratio holds
          between the two sparse rows, so the book's point survives: fewer bits per value, more operations per second.
        </p>
      </Translated>
    ),
    source: 'book §A.5, p. 593; notes/today/appA-facts.md A11 and ch05-facts.md A8',
    vi: { source: 'sách §A.5, p. 593; notes/today/appA-facts.md A11 và ch05-facts.md A8' },
  },
  'appA/A.5/q-ceiling': {
    kind: 'numeric',
    id: 'appA/A.5/q-ceiling',
    prompt: (
      <Translated
        vi={
          <>
            Một định dạng không có trong bảng lưu mã 4-bit cộng một tỉ lệ FP16 và một điểm không (zero point) FP16 cho mỗi 32
            trọng số. Trọng số của một mô hình chuyển sang định dạng đó từ BF16. Theo lập luận về băng thông ở trên, trần
            decode ở batch 1 tăng lên bao nhiêu lần?
          </>
        }
      >
        <>
          A format not in the table stores 4-bit codes plus one FP16 scale and one FP16 zero point for every 32 weights. A
          model's weights move to it from BF16. By what factor does the batch-1 decode ceiling rise, by the bandwidth
          argument above?
        </>
      </Translated>
    ),
    answer: 3.2,
    tol: 0.02,
    unit: '×',
    explain: (
      <Translated
        vi={
          <p>
            Số bit mỗi trọng số: 4 + (16 + 16) / 32 = 5. Trần bằng băng thông chia cho số byte mỗi token, nên nó tỉ lệ với 1
            / số bit mỗi trọng số: 16 / 5 = 3.2. Riêng mã 4-bit sẽ hứa hẹn 4; hai số 16-bit cho mỗi 32 trọng số tốn trọn một
            bit mỗi trọng số, gấp bốn lần một phần tư bit của MXFP4. Một bước decode thực còn đọc KV cache và activation,
            vốn không co lại theo trọng số, nên mức lợi đo được nhỏ hơn.
          </p>
        }
      >
        <p>
          Bits per weight: 4 + (16 + 16) / 32 = 5. The ceiling is bandwidth divided by bytes per token, so it scales with 1
          / bits per weight: 16 / 5 = 3.2. The 4-bit codes alone would promise 4; two 16-bit numbers per 32 weights cost a
          whole bit each, four times MXFP4's quarter bit. A real decode step also reads the KV cache and activations, which
          do not shrink with the weights, so the measured gain is smaller.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.1 method; our arithmetic (the format is a hypothetical one)',
    vi: { source: 'phương pháp ch. 5 §5.1; phép tính của khóa học này (định dạng này là giả định)' },
  },
  'appA/A.5/q-outlier': {
    kind: 'choice',
    id: 'appA/A.5/q-outlier',
    prompt: (
      <Translated vi={<>Một khối MXFP4 chứa một trọng số bằng 100 và 31 trọng số gần 0.3. Theo quy tắc tỉ lệ của OCP, các giá trị 0.3 giải mã thành gì?</>}>
        <>An MXFP4 block holds one weight of 100 and 31 weights near 0.3. Under the OCP scale rule, what do the 0.3s decode to?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="0: chia cho tỉ lệ, 0.3 nhỏ hơn một nửa bước nhỏ nhất.">0: divided by the scale, 0.3 is under half the smallest step.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Khoảng 0.3, vì mỗi phần tử E2M1 giữ số mũ riêng của nó.">About 0.3, since each E2M1 element keeps an exponent of its own.</Translated>,
        why: (
          <Translated vi="Một phần tử có 2 bit mũ; phạm vi đến từ một tỉ lệ dùng chung duy nhất.">
            An element has 2 exponent bits; the range comes from the one shared scale.
          </Translated>
        ),
      },
      {
        label: <Translated vi="8: bước nhỏ nhất của E2M1, 0.5, nhân với tỉ lệ khối 16.">{'8: E2M1\'s smallest step, 0.5, times the block scale of 16.'}</Translated>,
        why: <Translated vi="0.3 / 16 ≈ 0.019 gần 0 hơn nhiều so với 0.5.">0.3 / 16 ≈ 0.019 is far closer to 0 than to 0.5.</Translated>,
      },
      {
        label: <Translated vi="96: chúng bị kẹp về giá trị lớn nhất nhân với tỉ lệ.">96: they clamp to the largest value times the scale.</Translated>,
        why: (
          <Translated vi="Chỉ các phần tử lớn hơn giá trị lớn nhất mới bị kẹp; các phần tử này rất nhỏ.">
            Only elements above the largest value clamp; these are tiny.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            floor(log2 100) = 6; trừ số mũ lớn nhất của E2M1 là 2, được tỉ lệ 2<sup>4</sup> = 16. Khi đó 100 / 16 = 6.25 bị kẹp
            về 6 (giải mã thành 96), và 0.3 / 16 ≈ 0.019 nhỏ hơn 0.25, một nửa bước nhỏ nhất của E2M1, nên nó được làm tròn
            về 0. Một giá trị ngoại lai đã xóa 31 trọng số. Các khối 16, hoặc tỉ lệ mịn hơn, khoanh vùng nó lại: hãy thử cả hai
            trong trình khám phá.
          </p>
        }
      >
        <p>
          floor(log2 100) = 6; minus E2M1's largest exponent, 2, gives a scale of 2<sup>4</sup> = 16. Then 100 / 16 = 6.25
          clamps to 6 (decoding to 96), and 0.3 / 16 ≈ 0.019 is below 0.25, half of E2M1's smallest step, so it rounds to
          0. One outlier erased 31 weights. Blocks of 16, or finer scales, confine it: try both in the explorer.
        </p>
      </Translated>
    ),
    source: 'MX v1.1 §6.3 rule (notes/today/appA-facts.md A4); our arithmetic',
    vi: { source: 'quy tắc MX v1.1 §6.3 (notes/today/appA-facts.md A4); phép tính của khóa học này' },
  },
  'appA/A.5/q-mxfp4-bits': {
    kind: 'numeric',
    id: 'appA/A.5/q-mxfp4-bits',
    prompt: (
      <Translated vi={<>Giả sử MXFP4 dùng chung mỗi tỉ lệ UE8M0 cho 64 trọng số thay vì 32. Nó sẽ tốn bao nhiêu bit mỗi trọng số?</>}>
        <>Suppose MXFP4 shared each UE8M0 scale across 64 weights instead of 32. How many bits per weight would it take?</>
      </Translated>
    ),
    answer: 4.125,
    tol: 0.001,
    unit: 'bits',
    explain: (
      <Translated
        vi={
          <p>
            64 phần tử 4 bit cộng một tỉ lệ 8-bit: 264 bit cho 64 trọng số, 4.125 bit mỗi trọng số. Định dạng thực với 32
            tốn 4.25, đó là <code>block_mxfp4</code> của llama.cpp: 17 byte cho mỗi 32 trọng số. Khối lớn hơn tiết kiệm được
            một phần tám bit, và để một giá trị ngoại lai xóa số trọng số nhiều gấp đôi (câu hỏi ở trên).
          </p>
        }
      >
        <p>
          64 elements of 4 bits plus one 8-bit scale: 264 bits for 64 weights, 4.125 per weight. The real format's 32 cost
          4.25, which is llama.cpp's <code>block_mxfp4</code>: 17 bytes per 32 weights. Bigger blocks save an eighth of a
          bit, and let one outlier erase twice as many weights (the question above).
        </p>
      </Translated>
    ),
    source: 'MX v1.1 Table 1; llama.cpp ggml-common.h (notes/today/appA-facts.md A4, D3)', // i18n-ignore: a citation
    vi: { unit: 'bit' },
  },
  'appA/A.5/q-fnuz': {
    kind: 'choice',
    id: 'appA/A.5/q-fnuz',
    prompt: (
      <Translated vi={<>Các byte OCP E4M3 được trao, chưa chuyển đổi, cho mã đọc chúng như FNUZ E4M3 của CDNA3. Chuyện gì xảy ra?</>}>
        <>OCP E4M3 bytes are handed, unconverted, to code that reads them as CDNA3's FNUZ E4M3. What happens?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Các giá trị hữu hạn giảm một nửa, 0x80 thành NaN, và byte NaN của OCP thành ±240.">
            Finite values halve, 0x80 turns into NaN, and the OCP NaN byte becomes ±240.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không có gì thay đổi: cả hai định dạng đều có 1 bit dấu, 4 bit mũ và 3 bit mantissa.">
            Nothing changes: both formats have 1 sign bit, 4 exponent bits and 3 mantissa bits.
          </Translated>
        ),
        why: (
          <Translated vi="Cùng độ rộng trường, nhưng bias khác và các giá trị đặc biệt khác.">
            Same field widths, different bias and different special values.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các giá trị hữu hạn tăng gấp đôi, vì hai bias số mũ chênh nhau một.">
            Finite values double, because the two exponent biases differ by one.
          </Translated>
        ),
        why: (
          <Translated vi="FNUZ có bias 8, hơn OCP một: cùng các bit nghĩa là một giá trị nhỏ hơn.">
            FNUZ has bias 8, one more than OCP: the same bits mean a smaller value.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ các giá trị lớn hơn 240 thay đổi: chúng bão hòa về 240, giá trị lớn nhất của FNUZ.">
            Only values above 240 change: they saturate to 240, the FNUZ maximum.
          </Translated>
        ),
        why: (
          <Translated vi="Bias khác nhau, nên mọi giá trị khác 0 đều thay đổi.">
            The bias differs, so every non-zero value changes.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            FNUZ E4M3 có bias 8 trong khi OCP có 7, nên cùng các bit mũ biểu thị một lũy thừa của hai nhỏ hơn một bậc: mọi
            giá trị hữu hạn giảm một nửa. FNUZ không có số 0 âm và dùng 1000 0000 làm NaN, còn mẫu NaN S.1111.111 của OCP là
            một số thông thường trong FNUZ, 1.875 × 2<sup>7</sup> = 240. Đó là lý do HIP có hai kiểu riêng{' '}
            <code>__hip_fp8_e4m3</code> và <code>__hip_fp8_e4m3_fnuz</code>, còn PyTorch có hai kiểu riêng{' '}
            <code>float8_e4m3fn</code> và <code>float8_e4m3fnuz</code>.
          </p>
        }
      >
        <p>
          FNUZ E4M3 has bias 8 where OCP has 7, so the same exponent bits stand for one power of two less: every finite
          value halves. FNUZ has no negative zero and uses 1000 0000 as its NaN, while OCP's NaN pattern S.1111.111 is an
          ordinary number in FNUZ, 1.875 × 2<sup>7</sup> = 240. That is why HIP has separate <code>__hip_fp8_e4m3</code> and{' '}
          <code>__hip_fp8_e4m3_fnuz</code> types and PyTorch separate <code>float8_e4m3fn</code> and <code>float8_e4m3fnuz</code>.
        </p>
      </Translated>
    ),
    source: 'notes/today/appA-facts.md B2 (bias from HIP\'s header), D1; our arithmetic',
    vi: { source: 'notes/today/appA-facts.md B2 (bias lấy từ header của HIP), D1; phép tính của khóa học này' },
  },
};

// ---------------------------------------------------------------- the section
const FORMAT_ROWS: { f: Format; label: string; sem: string; bias: number; inf: string; nan: string }[] = [
  {
    f: E4M3,
    label: 'FP8 E4M3', // i18n-ignore: a format name
    sem: '1-4-3',
    bias: 7,
    get inf() {
      return tr('none', 'không có');
    },
    nan: 'S.1111.111',
  },
  { f: E5M2, label: 'FP8 E5M2', sem: '1-5-2', bias: 15, inf: 'S.11111.00', nan: 'S.11111.{01, 10, 11}' }, // i18n-ignore: a format name and bit patterns
  {
    f: E3M2,
    label: 'FP6 E3M2', // i18n-ignore: a format name
    sem: '1-3-2',
    bias: 3,
    get inf() {
      return tr('none', 'không có');
    },
    get nan() {
      return tr('none', 'không có');
    },
  },
  {
    f: E2M3,
    label: 'FP6 E2M3', // i18n-ignore: a format name
    sem: '1-2-3',
    bias: 1,
    get inf() {
      return tr('none', 'không có');
    },
    get nan() {
      return tr('none', 'không có');
    },
  },
  {
    f: E2M1,
    label: 'FP4 E2M1', // i18n-ignore: a format name
    sem: '1-2-1',
    bias: 1,
    get inf() {
      return tr('none', 'không có');
    },
    get nan() {
      return tr('none', 'không có');
    },
  },
];

export function SA5() {
  const mx = useLabFile('appA/snippets/mx_quantize.h');
  return (
    <>
      <Translated
        vi={
          <p>
            Sách dành cho các định dạng độ chính xác thấp hai đoạn văn. Đoạn đầu nói vì sao chúng tồn tại. Một số ứng dụng
            vốn làm việc trên các kiểu dữ liệu nhỏ, rộng 4, 6, 8 hoặc 16 bit: sách nêu tên ảnh y tế, viễn thám, thiên văn vô
            tuyến và phân tích địa chấn, và gần đây các LLM cũng đã được điều chỉnh cho các kiểu dấu phẩy động nhỏ. Dữ liệu như
            vậy chỉ cần vài bit có nghĩa, nhưng vẫn được lợi từ phạm vi, thứ mà bit mũ mua được (§A.2). Đó là lý do dùng các
            định dạng dấu phẩy động hẹp hơn độ chính xác đơn thay vì các số nguyên nhỏ.
          </p>
        }
      >
        <p>
          The book gives lower-precision formats two paragraphs. The first says why they exist. Some applications work on
          small data types by nature, 4, 6, 8 or 16 bits wide: the book names medical imaging, remote sensing, radio
          astronomy and seismic analysis, and LLMs have more recently been adapted to small floating-point types too. Such data needs only a few
          significant bits, but it still benefits from range, which is what exponent bits buy (§A.2). That is the case for
          floating-point formats narrower than single precision rather than small integers.
        </p>
      </Translated>
      <Source section="A.5" pages="592–593" />
      <Translated
        vi={
          <p>
            Đoạn thứ hai nói về tốc độ. Một định dạng nhỏ hơn di chuyển ít byte hơn cho mỗi phép toán số học, điều này nâng tỉ
            lệ tính toán trên bộ nhớ mà chương 5 dùng để đặt một kernel lên roofline, và GPU ngày nay tính trực tiếp bằng dấu
            phẩy động 16, 8 và 4 bit. Ví dụ của sách là H100: 3,958 TFLOPS FP8 trên tensor core so với 989 TFLOPS độ chính xác
            đơn, gấp bốn lần số phép toán mỗi giây.
          </p>
        }
      >
        <p>
          The second paragraph is about speed. A smaller format moves fewer bytes for each arithmetic operation, which
          raises the compute-to-memory ratio chapter 5 used to place a kernel on the roofline, and GPUs now compute in
          16-, 8- and 4-bit floating point directly. The book's example is the H100: 3,958 TFLOPS for FP8 on its tensor
          cores against 989 TFLOPS for single precision, four times as many operations per second.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/appA-facts.md A10, A11; ch05-facts.md A8)">{/* i18n-ignore: a date and a citation */}
        <Translated
          vi={
            <p>
              Đã đối chiếu với bảng H100 của NVIDIA: các hàng của nó được ghi nhãn khác với cách diễn đạt của sách (câu hỏi bên
              dưới hỏi khác ra sao). Tensor core FP8 bắt đầu từ compute capability 8.9; FP6 và FP4 từ 10.0.
            </p>
          }
        >
          <p>
            Checked against NVIDIA's H100 table: its rows are labelled differently from the book's wording (the question
            below asks how). FP8 tensor cores start at compute capability 8.9; FP6 and FP4 at 10.0.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsSA5['appA/A.5/q-h100']} />

      <h2>
        <Translated vi="Ít bit hơn mua được gì cho một bước decode">What fewer bits buy a decode step</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương 5 đã đếm bước decode ở batch 1: mỗi trọng số được đọc một lần và dùng cho một phép nhân và một phép cộng.
            Không gì dùng lại một trọng số, nên bước này bị giới hạn bởi bộ nhớ, và trần của nó là băng thông chia cho số byte
            trọng số đọc mỗi token. Giảm một nửa số bit mỗi trọng số và trần tăng gấp đôi, dù không phép toán nào nhanh hơn.
            Đó là lập luận băng thông cho trọng số lượng tử hóa.
          </p>
        }
      >
        <p>
          Chapter 5 counted the decode step at batch 1: every weight is read once and used for one multiply and one add.
          Nothing reuses a weight, so the step is bound by memory, and its ceiling is the bandwidth divided by the bytes of
          weights read per token. Halve the bits per weight and the ceiling doubles, though no arithmetic got faster. That
          is the bandwidth argument for quantised weights.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              §A.5 của sách dừng ở "ít byte hơn cho mỗi phép toán". Đây là phương pháp của chương 5 (
              <a href={href('ch05', '5.1')}>§5.1</a>) áp dụng cho các định dạng bên dưới; bản thân các định dạng khối đến từ OCP
              và NVIDIA.
            </p>
          }
        >
          <p>
            The book's §A.5 stops at "fewer bytes per operation". This is chapter 5's method (<a href={href('ch05', '5.1')}>§5.1</a>)
            applied to the formats below; the block formats themselves come from OCP and NVIDIA.
          </p>
        </Translated>
      </Beyond>
      <DecodeCeilingBench />
      <Translated
        vi={
          <p>
            Hãy đọc bảng từ trên xuống. Trần của mỗi hàng bằng trần của hàng BF16 nhân 16 chia cho số bit mỗi trọng số của
            nó, bất kể kích thước mô hình hay thiết bị. Các định dạng khối trả giá cho các tỉ lệ của chúng: một tỉ lệ 8-bit
            cho mỗi 32 trọng số thêm một phần tư bit, một tỉ lệ cho mỗi 16 thêm nửa bit. Hãy chọn thiết bị của bạn nếu bạn đã
            chạy lab 5.1, và cả bảng sẽ dịch chuyển cùng nhau.
          </p>
        }
      >
        <p>
          Read the table down. Every row's ceiling is the BF16 row's times 16 over its bits per weight, whatever the model
          size or the device. Block formats pay for their scales: one 8-bit scale per 32 weights adds a quarter of a bit,
          one per 16 adds half a bit. Select your own device if you have run lab 5.1, and the whole table moves together.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Chỉ tính trọng số: không có KV cache, activation hay chi phí khởi chạy; batch 1; dùng toàn bộ băng thông; bỏ qua
              tỉ lệ theo tensor. Decode thực đạt được một phần các trần này, và KV cache không co lại theo trọng số.
            </p>
          }
        >
          <p>
            Weights only: no KV cache, activations or launch overhead; batch 1; the full bandwidth used; per-tensor scales
            ignored. Real decode reaches a fraction of these ceilings, and the KV cache does not shrink with the weights.
          </p>
        </Translated>
      </Simplification>
      <Ask q={questionsSA5['appA/A.5/q-ceiling']} />

      <h2>
        <Translated vi="Các bố cục mà sách bỏ qua">The layouts the book leaves out</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách không nêu bố cục bit nào cho bất cứ thứ gì dưới 16 bit. Các định nghĩa mà phần cứng và framework ngày nay dùng
            chung đến từ Open Compute Project: đặc tả OFP8 cho hai định dạng 8-bit, và đặc tả MX cho các định dạng 6 và 4 bit
            và cho kiểu tỉ lệ. Mọi điều trong §A.1 đến §A.3 đều áp dụng cho chúng, với một điểm khác biệt: các định dạng hẹp
            bỏ vô cực, và đôi khi cả NaN, để giữ nhiều giá trị hữu hạn hơn.
          </p>
        }
      >
        <p>
          The book names no bit layout for anything below 16 bits. The definitions that hardware and frameworks now share
          come from the Open Compute Project: the OFP8 specification for the two 8-bit formats, and the MX specification
          for the 6- and 4-bit ones and for the scale type. Everything in §A.1 to §A.3 applies to them, with one twist: the
          narrow formats give up infinities, and sometimes NaN, to keep more finite values.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Từ OCP OFP8 r1.1 và MX v1.1 (notes/today/appA-facts.md A3, A4). Sách dẫn MX v1.0 (2023); v1.1 đổi tên kiểu tỉ lệ
              thành UE8M0.
            </p>
          }
        >
          <p>
            From OCP OFP8 r1.1 and MX v1.1 (notes/today/appA-facts.md A3, A4). The book cites MX v1.0 (2023); v1.1 renamed
            the scale type UE8M0.
          </p>
        </Translated>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-formats>
        <thead>
          <tr>
            <th>
              <Translated vi="Định dạng">Format</Translated>
            </th>
            <th>
              <Translated vi="Bit S-E-M">S-E-M bits</Translated>
            </th>
            <th className="num">
              <Translated vi="Bias">Bias</Translated>
            </th>
            <th className="num">
              <Translated vi="Lớn nhất">Largest</Translated>
            </th>
            <th className="num">
              <Translated vi="Dương nhỏ nhất">Smallest positive</Translated>
            </th>
            <th>
              <Translated vi="Vô cực">Infinity</Translated>
            </th>
            <th>NaN</th>
          </tr>
        </thead>
        <tbody>
          {FORMAT_ROWS.map((r) => (
            <tr key={r.label}>
              <td>{r.label}</td>
              <td className="mono">{r.sem}</td>
              <td className="num">{r.bias}</td>
              <td className="num">{maxFinite(r.f).toLocaleString('en-US')}</td>
              <td className="num">
                2<sup>{Math.log2(minSubnormal(r.f)!)}</sup>
              </td>
              <td className="mono">{r.inf}</td>
              <td className="mono">{r.nan}</td>
            </tr>
          ))}
          <tr>
            <td>
              <Translated vi="UE8M0 (tỉ lệ)">UE8M0 (scales)</Translated>
            </td>
            <td className="mono">0-8-0</td>
            <td className="num">127</td>
            <td className="num">
              2<sup>127</sup>
            </td>
            <td className="num">
              2<sup>−127</sup>
            </td>
            <td className="mono">
              <Translated vi="không có">none</Translated>
            </td>
            <td className="mono">
              <Translated vi="1111 1111; cũng không có số 0">1111 1111; no zero either</Translated>
            </td>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            E2M1 chỉ có tám độ lớn: {elemGrid(E2M1).join(', ')}. Riêng nó thì vô dụng với trọng số. Nó trở nên hữu ích khi một
            khối phần tử dùng chung một tỉ lệ, đó là chủ đề của phần tiếp theo trên trang này. E5M2 giữ các giá trị đặc biệt
            của IEEE và chính là FP16 bị cắt đi tám bit mantissa; E4M3 dùng số mũ cao nhất cho các số, chỉ giữ một mẫu NaN, và
            đạt 448 trong khi một bố cục kiểu IEEE sẽ dừng ở 240.
          </p>
        }
      >
        <p>
          E2M1 has just eight magnitudes: {elemGrid(E2M1).join(', ')}. On its own that is useless for weights. It becomes
          useful when a block of elements shares a scale, which is the subject of the next part of this page. E5M2 keeps
          IEEE's special values and is FP16 with eight mantissa bits cut off; E4M3 spends its top exponent on numbers,
          keeping a single NaN pattern, and reaches 448 where an IEEE-style layout would stop at 240.
        </p>
      </Translated>

      <h3>
        <Translated vi="Chuyển đổi sang chúng">Converting into them</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Phép chuyển đổi làm tròn về gần nhất, hòa thì về số chẵn, quy tắc của §A.4. Chuyện gì xảy ra khi vượt quá giá trị
            lớn nhất là một lựa chọn riêng: chế độ bão hòa kẹp về giá trị đó; chế độ không bão hòa cho NaN với E4M3, vốn không
            có vô cực, và cho vô cực với E5M2. Làm tròn diễn ra trước, nên một giá trị vượt đỉnh một chút có thể được làm tròn
            xuống đỉnh trước khi quy tắc nào được áp dụng.
          </p>
        }
      >
        <p>
          Conversion rounds to nearest, ties to even, the rule of §A.4. What happens past the largest value is a separate
          choice: saturating mode clamps to it; non-saturating mode gives NaN for E4M3, which has no infinity, and infinity
          for E5M2. Rounding happens first, so a value a little past the top may round down to it before either rule
          applies.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/appA-facts.md A3, A5, B2, C4)">{/* i18n-ignore: a date and a citation */}
        <Translated
          vi={
            <p>
              Các mặc định khác nhau. <code>__nv_fp8_e4m3(float)</code> của CUDA và <code>__hip_fp8_e4m3</code> của HIP bão
              hòa. <code>pack</code> của Metal bão hòa với E4M3 và E2M1 nhưng không với E5M2. Các trường hợp 464 và 61,440 nằm
              trong Appendix A mới của OFP8 r1.1.
            </p>
          }
        >
          <p>
            Defaults differ. CUDA's <code>__nv_fp8_e4m3(float)</code> and HIP's <code>__hip_fp8_e4m3</code> saturate. Metal's{' '}
            <code>pack</code> saturates for E4M3 and E2M1 but not for E5M2. The 464 and 61,440 cases are in OFP8 r1.1's new
            Appendix A.
          </p>
        </Translated>
      </Today>
      <Predict
        id="appA/A.5/p-e4m3"
        prompt={
          <Translated vi={<>460 được chuyển sang E4M3 ở chế độ bão hòa. Giá trị nào được lưu?</>}>
            <>460 is converted to E4M3 in saturating mode. What value is stored?</>
          </Translated>
        }
      >
        {(v) => <TopReveal v={v} />}
      </Predict>

      <h2>
        <Translated vi="Một tỉ lệ cho một khối trọng số">One scale for a block of weights</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Tỉ lệ theo khối lưu một nhóm k phần tử P<sub>i</sub> và một tỉ lệ dùng chung X, và giá trị của phần tử i là X ·
            P<sub>i</sub>. Trong các định dạng MX của OCP, k là 32 và X là một lũy thừa của hai, lưu trong một byte UE8M0:
            MXFP8 có phần tử E4M3 hoặc E5M2, MXFP6 có E3M2 hoặc E2M3, MXFP4 có E2M1. Tỉ lệ cho khối phạm vi; các phần tử cho nó
            vài bit có nghĩa.
          </p>
        }
      >
        <p>
          Block scaling stores a group of k elements P<sub>i</sub> and one shared scale X, and the value of element i is X
          · P<sub>i</sub>. In OCP's MX formats k is 32 and X is a power of two, stored as one UE8M0 byte: MXFP8 has E4M3 or
          E5M2 elements, MXFP6 E3M2 or E2M3, MXFP4 E2M1. The scale gives the block its range; the elements give it a few
          significant bits.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Từ MX v1.1 §§5–6 (notes/today/appA-facts.md A4). Các định dạng khối số nguyên (MXINT8) cũng tồn tại và được bỏ qua ở đây.</p>}>
          <p>From MX v1.1 §§5–6 (notes/today/appA-facts.md A4). Integer block formats (MXINT8) exist too and are left out here.</p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            OCP gợi ý chọn X từ độ lớn lớn nhất trong khối: số mũ dùng chung là floor(log2 max|V|) trừ số mũ lớn nhất của
            định dạng phần tử, và phần tử nào sau đó rơi lên trên giá trị lớn nhất của định dạng phần tử thì bị kẹp về giá trị
            đó. cuBLAS thì làm tròn số mũ lên, điều mà theo lời họ "khác với cách làm tròn mà OCP MXFP gợi ý": phần tử lớn
            nhất khi đó không bao giờ bị kẹp, với cái giá là đôi khi mất một binade cho các phần tử còn lại.
          </p>
        }
      >
        <p>
          OCP suggests choosing X from the largest magnitude in the block: the shared exponent is floor(log2 max|V|) minus
          the largest exponent of the element format, and any element that then lands above the element format's largest
          value is clamped to it. cuBLAS rounds the exponent up instead, which, in its own words, "differs from the OCP
          MXFP suggested rounding scheme": the largest element then never clamps, at the cost of one binade for the rest
          some of the time.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/appA-facts.md A5, D2, D3)">{/* i18n-ignore: a date and a citation */}
        <Translated
          vi={
            <p>
              Các phép GEMM FP4 của NVIDIA trong cuBLAS dùng một bố cục khác: 16 phần tử mỗi tỉ lệ, tỉ lệ là một số float 8-bit
              (UE4M3) thay vì một lũy thừa của hai, cộng thêm một tỉ lệ cho mỗi tensor. Các trang của NVIDIA không gọi nó là
              NVFP4; MLX và llama.cpp thì có.
            </p>
          }
        >
          <p>
            NVIDIA's FP4 GEMMs in cuBLAS use another layout: 16 elements per scale, the scale an 8-bit float (UE4M3) instead
            of a power of two, plus one scale per tensor. NVIDIA's pages do not call it NVFP4; MLX and llama.cpp do.
          </p>
        </Translated>
      </Today>
      <MxBlockBench />
      <Translated
        vi={
          <p>
            Bắt đầu không có giá trị ngoại lai và với E2M1: một vài trọng số nhỏ nhất đã bị làm tròn về 0. Giờ hãy tăng giá
            trị ngoại lai. Độ lớn của nó quyết định tỉ lệ cho cả 32, và các trọng số nhỏ lần lượt rơi xuống dưới đường nét
            đứt. Rồi giành lại chúng theo ba cách: 16 phần tử mỗi tỉ lệ, để giá trị ngoại lai chỉ làm hỏng nửa của nó; phần tử
            E4M3, có thêm bit mũ vươn xa xuống dưới tỉ lệ; hoặc quy tắc tỉ lệ E4M3, khớp tỉ lệ với khối mịn hơn một lũy thừa
            của hai. Hãy so sánh quy tắc OCP và cuBLAS xem cái nào bị kẹp.
          </p>
        }
      >
        <p>
          Start with no outlier and E2M1: a few of the smallest weights already round to 0. Now raise the outlier. Its
          magnitude sets the scale for all 32, and the small weights fall below the dashed line one after another. Then
          win them back three ways: 16 elements per scale, so the outlier only spoils its own half; E4M3 elements, whose
          extra exponent bits reach far below the scale; or the E4M3 scale rule, which fits the scale to the block more
          finely than a power of two. Compare the OCP and cuBLAS rules for which one clamps.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một khối riêng lẻ, trọng số phân phối chuẩn. Tỉ lệ E4M3 được mô hình hóa như một giá trị E4M3 dương với tỉ lệ
              toàn tensor đặt bằng 1 (mã hóa chính xác của UE4M3 không có trong các nguồn chúng tôi đã kiểm tra). Một khối toàn
              số 0 lấy tỉ lệ nhỏ nhất, đó là lựa chọn của chúng tôi; dù sao nó cũng giải mã thành các số 0.
            </p>
          }
        >
          <p>
            One block in isolation, normally distributed weights. The E4M3 scale is modelled as a positive E4M3 value with
            the tensor-wide scale set to 1 (UE4M3's exact encoding is not in the sources we checked). An all-zero block takes
            the smallest scale, our choice; it decodes to zeros either way.
          </p>
        </Translated>
      </Simplification>
      <Ask q={questionsSA5['appA/A.5/q-outlier']} />
      <Ask q={questionsSA5['appA/A.5/q-mxfp4-bits']} />

      <h3>
        <Translated vi="Lượng tử hóa một khối, từng bước">Quantising one block, step by step</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Quy trình ngắn, và một quyết định sớm lan tỏa qua toàn bộ: tìm độ lớn lớn nhất, suy ra tỉ lệ, chia, làm tròn lên
            lưới, kẹp, lưu. Hãy xem nó một lần trên một khối có giá trị ngoại lai.
          </p>
        }
      >
        <p>
          The pipeline is short, and one early decision ripples through it: find the largest magnitude, derive the scale,
          divide, round onto the grid, clamp, store. Watch it once on a block with an outlier.
        </p>
      </Translated>
      <MxBlockScene />
      <Translated vi={<p>Cùng quy trình đó bằng C++ trên host, hình dạng mà một bản tham chiếu CPU cho một kernel lượng tử hóa có:</p>}>
        <p>The same pipeline in host C++, the shape a CPU reference for a quantisation kernel takes:</p>
      </Translated>
      <Code
        tabs={[{ label: tr('C++ (host)', 'C++ (host)'), code: mx ?? '// loading labs/appA/snippets/mx_quantize.h' }]}
        caption={
          <Translated
            vi={
              <>
                labs/appA/snippets/mx_quantize.h, của khóa học này. Các unit test biên dịch nó, chạy nó trên khối của cảnh và
                mười bốn khối khác, và so từng byte tỉ lệ, mã và số float giải mã với mô hình đứng sau trình khám phá, từng bit.
              </>
            }
          >
            <>
              labs/appA/snippets/mx_quantize.h, ours. The unit tests compile it, run it on the scene's block and fourteen
              others, and compare every scale byte, code and decoded float with the model behind the explorer, bit for bit.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Thêm một thuộc tính của các định dạng nối về phụ lục này. MX v1.1 nói rằng "độ chính xác bên trong của tích vô hướng
            và thứ tự các phép toán do cài đặt quyết định". Hai kernel tuân thủ có thể cộng 32 tích của một khối theo thứ tự
            khác nhau và chỉ khớp nhau trong phạm vi sai số làm tròn: ý của §<a href={href('appA', 'A.6')}>A.6</a>, ở quy mô
            một khối.
          </p>
        }
      >
        <p>
          One more property of the formats connects back to this appendix. MX v1.1 states that "the internal precision of
          the dot product and order of operations is implementation-defined". Two conforming kernels may sum the 32
          products of a block in different orders and agree only to within rounding: §
          <a href={href('appA', 'A.6')}>A.6</a>'s point, at the scale of a block.
        </p>
      </Translated>

      <h2>
        <Translated vi="Ai có gì, hôm nay">Who has what, today</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Không điều nào ở đây có trong sách; tất cả đều có trong tài liệu hiện hành của các nhà cung cấp. Tóm tắt: CUDA và
            HIP có các kiểu vô hướng cho các định dạng nhỏ để bạn chuyển đổi qua lại, tensor core và các khối ma trận tính
            toán với chúng, còn Metal chỉ có chúng dưới dạng lưu trữ đóng gói và kiểu phần tử của tensor.
          </p>
        }
      >
        <p>
          None of this is in the book; all of it is in current vendor documentation. The short version: CUDA and HIP have
          scalar types for the small formats that you convert to and from, tensor cores and matrix units compute with them,
          and Metal has them only as packed storage and tensor element types.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/appA-facts.md A5, A10, B2, C4, D1 to D3)', '2026-11-20 (notes/today/appA-facts.md A5, A10, B2, C4, D1 đến D3)')}>
        <Translated
          vi={
            <p>
              Các kiểu nhỏ của CUDA hỗ trợ các truy vấn như <code>numeric_limits</code>; NVIDIA không liệt kê toán tử số học
              nào cho chúng. Tables 2.15 và 2.16 của Apple có ba lỗi in (hãy dùng bảng của OCP), và Apple không nói họ GPU nào
              hỗ trợ các định dạng này.
            </p>
          }
        >
          <p>
            CUDA's small types support queries such as <code>numeric_limits</code>; NVIDIA lists no arithmetic operators
            for them. Apple's Tables 2.15 and 2.16 contain three misprints (use OCP's tables), and Apple does not say which
            GPU families support these formats.
          </p>
        </Translated>
      </Today>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr>
            <th></th>
            <th>FP8</th>
            <th>FP6, FP4</th>{/* i18n-ignore: format names */}
            <th>
              <Translated vi="Tỉ lệ theo khối">Block scales</Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>CUDA</td>
            <Translated
              vi={
                <td>
                  <code>__nv_fp8_e4m3</code>, <code>__nv_fp8_e5m2</code> (<code>cuda_fp8.h</code>); tensor core từ compute
                  capability 8.9
                </td>
              }
            >
              <td>
                <code>__nv_fp8_e4m3</code>, <code>__nv_fp8_e5m2</code> (<code>cuda_fp8.h</code>); tensor cores from compute
                capability 8.9
              </td>
            </Translated>
            <Translated
              vi={
                <td>
                  <code>__nv_fp6_e2m3</code>, <code>__nv_fp6_e3m2</code>, <code>__nv_fp4_e2m1</code> (<code>cuda_fp6.h</code>,{' '}
                  <code>cuda_fp4.h</code>); tensor core từ 10.0
                </td>
              }
            >
              <td>
                <code>__nv_fp6_e2m3</code>, <code>__nv_fp6_e3m2</code>, <code>__nv_fp4_e2m1</code> (<code>cuda_fp6.h</code>,{' '}
                <code>cuda_fp4.h</code>); tensor cores from 10.0
              </td>
            </Translated>
            <Translated
              vi={
                <td>
                  <code>__nv_fp8_e8m0</code>; PTX <code>mma</code> với tỉ lệ theo khối; cuBLAS UE8M0 32 phần tử cho FP8, UE4M3
                  16 phần tử cho FP4
                </td>
              }
            >
              <td>
                <code>__nv_fp8_e8m0</code>; PTX <code>mma</code> with block scaling; cuBLAS 32-element UE8M0 for FP8, 16-element
                UE4M3 for FP4
              </td>
            </Translated>
          </tr>
          <tr>
            <td>HIP</td>
            <Translated
              vi={
                <td>
                  <code>__hip_fp8_e4m3</code>, <code>_e5m2</code> (OCP; CDNA4, RDNA4); <code>_fnuz</code> (CDNA3: bias cao hơn
                  một, không có vô cực, không có −0, NaN 0x80)
                </td>
              }
            >
              <td>
                <code>__hip_fp8_e4m3</code>, <code>_e5m2</code> (OCP; CDNA4, RDNA4); <code>_fnuz</code> (CDNA3: bias one
                higher, no infinity, no −0, NaN 0x80)
              </td>
            </Translated>
            <td>
              <Translated vi="chỉ CDNA4">CDNA4 only</Translated>
            </td>
            <Translated
              vi={
                <td>
                  các API chuyển đổi <code>__amd_scale_t</code> (E8M0), phần cứng trên gfx950
                </td>
              }
            >
              <td>
                <code>__amd_scale_t</code> (E8M0) conversion APIs, hardware on gfx950
              </td>
            </Translated>
          </tr>
          <tr>
            <td>Metal 4.1</td>{/* i18n-ignore: a product name and version */}
            <Translated
              vi={
                <td>
                  <code>metal_fp8_e4m3_format</code>, <code>metal_fp8_e5m2_format</code>: lưu trữ đóng gói và kiểu phần tử của
                  tensor; không có phép tính FP8 vô hướng
                </td>
              }
            >
              <td>
                <code>metal_fp8_e4m3_format</code>, <code>metal_fp8_e5m2_format</code>: packed storage and tensor element types;
                no scalar FP8 arithmetic
              </td>
            </Translated>
            <Translated
              vi={
                <td>
                  <code>metal_fp4_e2m1_format</code>; không có FP6
                </td>
              }
            >
              <td>
                <code>metal_fp4_e2m1_format</code>; no FP6
              </td>
            </Translated>
            <Translated
              vi={
                <td>
                  <code>tensor_blockwise</code> với <code>metal_fp8_ue8m0_format</code>, khối 32
                </td>
              }
            >
              <td>
                <code>tensor_blockwise</code> with <code>metal_fp8_ue8m0_format</code>, block 32
              </td>
            </Translated>
          </tr>
          <tr>
            <td>
              <Translated vi="Framework">Frameworks</Translated>
            </td>
            <Translated
              vi={
                <td>
                  các dtype "shell" của PyTorch như <code>float8_e4m3fn</code> ("f": chỉ hữu hạn, không có vô cực; "n": mã hóa
                  NaN khác với IEEE; "uz": chỉ có số 0 không dấu)
                </td>
              }
            >
              <td>
                PyTorch's "shell" dtypes such as <code>float8_e4m3fn</code> ("f": finite only, no infinity; "n": NaN encodings
                differ from IEEE; "uz": unsigned zero only)
              </td>
            </Translated>
            <Translated
              vi={
                <td>
                  các chế độ <code>quantize</code> của MLX: <code>mxfp4</code>, <code>mxfp8</code>, <code>nvfp4</code>
                </td>
              }
            >
              <td>
                MLX <code>quantize</code> modes <code>mxfp4</code>, <code>mxfp8</code>, <code>nvfp4</code>
              </td>
            </Translated>
            <Translated
              vi={
                <td>
                  <code>block_mxfp4</code> của llama.cpp: 17 byte cho mỗi 32 trọng số, bảng E2M1 của nó được lưu nhân đôi để vừa{' '}
                  <code>int8_t</code>
                </td>
              }
            >
              <td>
                llama.cpp <code>block_mxfp4</code>: 17 bytes per 32 weights, its E2M1 table stored doubled to fit{' '}
                <code>int8_t</code>
              </td>
            </Translated>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Hàng FNUZ là cái bẫy cho bất kỳ ai đọc mã AMD và NVIDIA song song: cùng độ rộng trường, khác con số.
          </p>
        }
      >
        <p>
          The FNUZ row is the trap for anyone reading AMD and NVIDIA code side by side: same field widths, different
          numbers.
        </p>
      </Translated>
      <Ask q={questionsSA5['appA/A.5/q-fnuz']} />

      <h2>
        <Translated vi="Tiếp theo sẽ đi đến đâu">Where this goes next</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lượng tử hóa số nguyên, mã int8 hoặc 4-bit với một tỉ lệ và một điểm không, hoàn toàn không có trong sách, và cũng
            không có kernel nào tính toán với bất kỳ định dạng hẹp nào. Module <a href={href('x2')}>X2</a> xây các kernel số
            nguyên: các phép GEMV lượng tử hóa có vòng lặp trong giải mã các khối mã dùng chung một tỉ lệ, trên các bố cục Q8_0
            và Q4_0 của llama.cpp, và mốc dự án M9, bước decode lượng tử hóa, đo xem bao nhiêu phần của tỉ lệ byte đứng sau các
            trần của trình khám phá đầu tiên biến thành tốc độ. Nó không có kernel FP8 hay MX nào.
          </p>
        }
      >
        <p>
          Integer quantisation, int8 or 4-bit codes with a scale and a zero point, is not in the book at all, and neither is
          a kernel that computes with any narrow format. Module <a href={href('x2')}>X2</a> builds integer ones: quantised
          GEMVs whose inner loops decode blocks of codes that share a scale, on llama.cpp's Q8_0 and Q4_0 layouts, and
          milestone M9, the quantised decode step, which measures how much of the byte ratio behind the first explorer's
          ceilings turns into speed. It has no FP8 or MX kernel.
        </p>
      </Translated>
    </>
  );
}
