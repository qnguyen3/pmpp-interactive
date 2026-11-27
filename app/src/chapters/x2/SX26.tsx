import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { Translated, tr as trText } from '../../lib/i18n';
import {
  BOUND_Q4_SIGNED,
  BOUND_Q4_UNSIGNED,
  BOUND_Q8Q8,
  FRAGMENTS,
  INNER,
  INNER_LIMIT,
  LADDER_LENGTHS,
  OUTER,
  PAIRING,
  accumulate,
  int32Terms,
  parseFragments,
  runFragment,
  sci,
  show,
  type DataSet,
  type FragmentId,
  type FragmentRun,
  type Inner,
  type Outer,
  type Pairing,
} from '../../sim/qdot';
import { NarrowAccumulatorScene } from './NarrowAccumulatorScene';

// Section X2.6: how wide each level of a quantised dot product's sum must be. Every number is
// from src/sim/qdot.ts (bounds, innerSum, accumulate, runFragment), pinned in qdot.test.ts.
// Module X2 is ours; the anchor is appendix A, §A.6 (and §A.2 for FP16's range).
const FACTS = 'notes/today/x2-facts.md';
const CHECKED = `2026-11-20 (${FACTS}`;

// ---------------------------------------------------------------- the ladder
function Ladder() {
  const [inner, setInner] = useState<Inner>('int16');
  const [outer, setOuter] = useState<Outer>('fp32');
  const [data, setData] = useState<DataSet>('worst');
  const [k, setK] = useState(7); // 4,096
  const [pairing, setPairing] = useState<Pairing>('q8q8');
  const [step, setStep] = useState<1 | 4>(1);
  const n = LADDER_LENGTHS[k];
  const r = useMemo(() => accumulate({ inner, outer, data, n, pairing, step }), [inner, outer, data, n, pairing, step]);
  const tr = r.block0;
  const lim = INNER_LIMIT[inner];
  const isInt = inner === 'int16' || inner === 'int32';
  // block 0's running sum, |value| on a log axis from 1 to 1e10
  const gx = 56, gy = 16, gw = 600, gh = 170;
  const LO = 0, HI = 10;
  const yl = (v: number) => gy + gh - ((Math.log10(Math.max(1, Math.abs(v))) - LO) / (HI - LO)) * gh;
  const xs = (i: number) => gx + ((i + 0.5) / tr.values.length) * gw;
  const bad = (i: number) =>
    (tr.first.wrap !== undefined && i + 1 >= tr.first.wrap) || (tr.first.inf !== undefined && i + 1 >= tr.first.inf);
  const status = r.overflowed ? 'inf' : r.wrapped ? 'wrapped' : r.rounded ? 'rounded' : 'exact';
  return (
    <Bench
      title={trText('The accumulator ladder', 'Thang bộ tích lũy')}
      task={trText(
        'Start with int16 on the worst case, then climb: int32, FP16, FP32. Then put FP16 on the outer sum and drag the row length.',
        'Bắt đầu với int16 ở trường hợp xấu nhất, rồi leo lên: int32, FP16, FP32. Sau đó đặt FP16 cho tổng ngoài và kéo độ dài hàng.',
      )}
      foot={
        <Translated
          vi={
            <>
              Một hàng gồm n trọng số và n activation, mã hóa theo khối 32 như llama.cpp làm: mỗi khối một tổng số nguyên của
              32 tích (bộ tích lũy trong), chuyển thành số thực bằng các hệ số tỉ lệ của khối, và các khối được cộng lại (bộ
              tích lũy ngoài). Thực tế: trọng số và activation phân phối chuẩn tắc sinh từ seed. Trường hợp xấu nhất: mọi mã ở
              giá trị cực trị (127, hoặc 15 với các nibble không dấu), với các hệ số tỉ lệ thực tế. Giá trị tham chiếu là cùng
              công thức đó ở binary64 trên các tổng số nguyên chính xác.
            </>
          }
        >
        <>
          One row of n weights and n activations, coded in blocks of 32 as llama.cpp does: per block an integer sum of
          32 products (the inner accumulator), turned into a float with the block's scales, and the blocks added up (the
          outer accumulator). Realistic: seeded standard-normal weights and activations. Worst case: every code at its
          extreme (127, or 15 for the unsigned nibbles), with the realistic scales. The reference is the same formula
          in binary64 on exact integer sums.
        </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <label className="field">
          <Translated vi="Bộ tích lũy trong">Inner accumulator</Translated>
          <select aria-label={trText('Inner accumulator', 'Bộ tích lũy trong')} value={inner} onChange={(e) => setInner(e.target.value as Inner)}>
            {INNER.map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Bộ tích lũy ngoài">Outer accumulator</Translated>
          <select aria-label={trText('Outer accumulator', 'Bộ tích lũy ngoài')} value={outer} onChange={(e) => setOuter(e.target.value as Outer)}>
            {OUTER.map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Cặp ghép">Pairing</Translated>
          <select aria-label={trText('Pairing', 'Cặp ghép')} value={pairing} onChange={(e) => setPairing(e.target.value as Pairing)}>
            {PAIRING.map((t) => <option key={t.id} value={t.id}>{trText(t.label, t.id === 'q8q8' ? 'trọng số Q8_0 × activation Q8_1' : 'nibble Q4_0 (0..15) × activation Q8_1')}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Bước số nguyên">Integer step</Translated>
          <select aria-label={trText('Integer step', 'Bước số nguyên')} value={step} onChange={(e) => setStep(+e.target.value as 1 | 4)} disabled={!isInt}>
            <option value={1}>{trText('one product at a time', 'mỗi lần một tích')}</option>
            <option value={4}>{trText('four at a time (dp4a)', 'mỗi lần bốn tích (dp4a)')}</option>
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={trText('Data', 'Dữ liệu')}>
          {(['realistic', 'worst'] as DataSet[]).map((d) => (
            <button key={d} aria-pressed={data === d} onClick={() => setData(d)}>{d === 'worst' ? trText('worst case', 'trường hợp xấu nhất') : trText('realistic', 'thực tế')}</button>
          ))}
        </span>
        <label className="field" style={{ minWidth: '14rem' }}>
          <Translated vi={<>Độ dài hàng: {n.toLocaleString('en-US')} ({n / 32} khối)</>}>Row length: {n.toLocaleString('en-US')} ({n / 32} blocks)</Translated>
          <input type="range" aria-label={trText('Row length', 'Độ dài hàng')} min={0} max={LADDER_LENGTHS.length - 1} value={k} onChange={(e) => setK(+e.target.value)} />
        </label>
      </div>
      <svg viewBox="0 0 700 240" role="img" aria-label={trText("The first block's running inner sum against the accumulator's limits", 'Tổng trong đang chạy của khối đầu tiên so với giới hạn của bộ tích lũy')} style={{ width: '100%', maxWidth: 700, display: 'block', minWidth: 540 }}>
        <rect x={gx} y={gy} width={gw} height={gh} fill="var(--cell)" stroke="var(--rule)" />
        {[0, 2, 4, 6, 8, 10].map((e) => (
          <g key={e}>
            <line x1={gx} x2={gx + gw} y1={yl(10 ** e)} y2={yl(10 ** e)} stroke="var(--rule-soft)" />
            <text x={gx - 6} y={yl(10 ** e) + 3} fontSize="10" textAnchor="end">{e === 0 ? '1' : `1e${e}`}</text>
          </g>
        ))}
        {lim.max < 1e10 ? (
          <g>
            <line x1={gx} x2={gx + gw} y1={yl(lim.max)} y2={yl(lim.max)} stroke="var(--fail)" strokeWidth="1.5" />
            <text x={gx + gw - 4} y={yl(lim.max) - 4} fontSize="10" textAnchor="end" style={{ fill: 'var(--fail)' }}><Translated vi={<>{INNER.find((t) => t.id === inner)!.label} lớn nhất: {show(lim.max)}</>}>largest {INNER.find((t) => t.id === inner)!.label}: {show(lim.max)}</Translated></text>
          </g>
        ) : null}
        {!isInt && lim.exactTo < 1e10 ? (
          <g>
            <line x1={gx} x2={gx + gw} y1={yl(lim.exactTo)} y2={yl(lim.exactTo)} stroke="var(--ink-3)" strokeDasharray="4 3" />
            <text x={gx + 4} y={yl(lim.exactTo) - 4} fontSize="10" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>mọi số nguyên đều chính xác đến {show(lim.exactTo)}</>}>every integer exact up to {show(lim.exactTo)}</Translated></text>
          </g>
        ) : null}
        <path d={tr.exact.map((v, i) => `${i ? 'L' : 'M'}${xs(i)},${yl(v)}`).join(' ')} fill="none" stroke="var(--ink-3)" strokeWidth="1.5" />
        {tr.values.map((v, i) => (
          <g key={i}>
            {Number.isFinite(v) ? (
              <circle cx={xs(i)} cy={yl(v)} r={3.2} fill={bad(i) ? 'var(--fail)' : v === tr.exact[i] ? 'var(--reg)' : 'var(--page)'}
                stroke={bad(i) ? 'var(--fail)' : 'var(--reg)'} />
            ) : (
              <text x={xs(i)} y={gy + 10} fontSize="10" textAnchor="middle" style={{ fill: 'var(--fail)' }}>∞</text>
            )}
          </g>
        ))}
        <text x={gx} y={gy + gh + 16} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
          <Translated vi={<>tổng đang chạy của khối 1, từng bước ({tr.values.length} bước, mỗi bước {tr.step} tích).</>}>
          block 1's running sum, step by step ({tr.values.length} steps of {tr.step} product{tr.step > 1 ? 's' : ''}).
          </Translated>
        </text>
        <text x={gx} y={gy + gh + 30} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
          <Translated vi="Đường xám: tổng chính xác. Chấm đặc: chính xác; chấm rỗng: đã làm tròn; đỏ: bị quay vòng hoặc tràn.">
          Grey line: exact sum. Filled dot: exact; open: rounded; red: wrapped or overflowed.
          </Translated>
        </text>
        <text x={gx} y={gy + gh + 48} fontSize="11" data-block0={String(tr.values[tr.values.length - 1])}>
          <Translated
            vi={
              <>
                khối 1 kết thúc ở {show(tr.values[tr.values.length - 1])} (chính xác {show(tr.exact[tr.exact.length - 1])})
                {tr.first.wrap ? `; quay vòng ở bước ${tr.first.wrap}` : ''}
                {tr.first.inf ? `; vô cực ở bước ${tr.first.inf}` : ''}
                {!tr.first.inf && tr.first.inexact ? `; lần làm tròn đầu tiên ở bước ${tr.first.inexact}` : ''}
              </>
            }
          >
          block 1 ends at {show(tr.values[tr.values.length - 1])} (exact {show(tr.exact[tr.exact.length - 1])})
          {tr.first.wrap ? `; wrapped at step ${tr.first.wrap}` : ''}
          {tr.first.inf ? `; infinity at step ${tr.first.inf}` : ''}
          {!tr.first.inf && tr.first.inexact ? `; first rounding at step ${tr.first.inexact}` : ''}
          </Translated>
        </text>
      </svg>
      <div className="row" style={{ marginTop: '0.6rem' }} data-ladder="" data-status={status} data-relerr={String(r.relErr)}>
        <span>
          <Translated vi={<>Tổng trong trên {n / 32} khối: {r.wrapped} bị quay vòng, {r.overflowed} bị tràn, {r.rounded} bị làm tròn.</>}>
          Inner sums over {n / 32} blocks: {r.wrapped} wrapped, {r.overflowed} overflowed, {r.rounded} rounded.
          </Translated>
        </span>
        <span>
          <Translated vi={<>Kết quả hàng {show(r.result, 8)} so với {show(r.reference, 8)}: sai số tương đối</>}>Row result {show(r.result, 8)} against {show(r.reference, 8)}: relative error</Translated>{' '}
          <b className={Number.isFinite(r.relErr) && r.relErr < 1e-6 ? 't-ok' : 't-fail'}>{Number.isFinite(r.relErr) ? sci(r.relErr) : trText('overflow', 'tràn')}</b>
          {Number.isFinite(r.absErr) ? <Translated vi={<> (tuyệt đối {sci(r.absErr)})</>}><> (absolute {sci(r.absErr)})</></Translated> : null}
        </span>
        <span className={'chip ' + (status === 'exact' ? 'ok' : status === 'rounded' ? '' : 'fail')} data-chip="">{status === 'exact' ? trText('inner sums exact', 'tổng trong chính xác') : status === 'rounded' ? trText('inner sums rounded', 'tổng trong bị làm tròn') : status === 'wrapped' ? trText('inner sums wrapped', 'tổng trong bị quay vòng') : trText('inner sums overflowed', 'tổng trong bị tràn')}</span>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the bug spot
/** The four fragments are a lesson snippet, labs/x2/snippets/accumulators.md, tied to FRAGMENT_MODEL by qdot.snippets.test.ts. */
const FRAG_FILE = 'x2/snippets/accumulators.md';

function reason(r: FragmentRun, base: FragmentRun): string {
  const ratio = r.realistic / base.realistic;
  switch (r.id) {
    case 'metal-half-dot':
      return trText(
        `dot(half4, half4) returns half, so every four products are summed and rounded in half. On the realistic row its error is ${sci(r.realistic)}, about ${Math.round(ratio)} times the integer-then-float fragment's ${sci(base.realistic)}. It does not overflow on this worst case (four products of 889), but four products past 65,504 would. Converting both operands to float4 before dot costs nothing.`,
        `dot(half4, half4) trả về half, nên cứ bốn tích lại được cộng và làm tròn ở half. Trên hàng thực tế, sai số của nó là ${sci(r.realistic)}, khoảng ${Math.round(ratio)} lần mức ${sci(base.realistic)} của đoạn số nguyên rồi số thực. Nó không tràn ở trường hợp xấu nhất này (bốn tích 889), nhưng bốn tích vượt 65,504 thì sẽ tràn. Chuyển cả hai toán hạng sang float4 trước khi dot không tốn gì.`,
      );
    case 'cuda-hfma2':
      return trText(
        `Each half lane accumulates half the row. On the realistic row the error is ${sci(r.realistic)}, about ${Math.round(ratio).toLocaleString('en-US')} times the safe fragment's; on the worst-case row a lane passes 65,504 at product ${r.worst.at} and the result is infinity.`,
        `Mỗi lane half tích lũy một nửa hàng. Trên hàng thực tế, sai số là ${sci(r.realistic)}, khoảng ${Math.round(ratio).toLocaleString('en-US')} lần của đoạn an toàn; trên hàng xấu nhất, một lane vượt 65,504 ở tích thứ ${r.worst.at} và kết quả là vô cực.`,
      );
    case 'short-sum':
      return trText(
        `On the realistic row it gives exactly what fragment 4 gives (${sci(r.realistic)}): the tests you would write first pass. On the worst-case block (codes 15 against 127, at most 60,960) the short wraps on dp4a step ${r.worst.at} of 8, and the row comes out as ${show(r.worst.result)} instead of ${show(r.worst.reference)}.`,
        `Trên hàng thực tế, nó cho đúng kết quả của đoạn 4 (${sci(r.realistic)}): những bài kiểm tra bạn viết đầu tiên đều qua. Trên khối xấu nhất (mã 15 với 127, tối đa 60,960), biến short quay vòng ở bước dp4a thứ ${r.worst.at} trên 8, và hàng ra ${show(r.worst.result)} thay vì ${show(r.worst.reference)}.`,
      );
    case 'int-then-float':
      return trText(
        `A block sum is at most 60,960, which an int32 holds with room to spare, and float holds each block's result and the row's: the worst-case row is exactly ${show(r.worst.result)}, and the realistic error is ${sci(r.realistic)}, float rounding.`,
        `Tổng của một khối tối đa là 60,960, một int32 chứa thoải mái, và float chứa được kết quả của từng khối lẫn của hàng: hàng xấu nhất cho đúng ${show(r.worst.result)}, và sai số thực tế là ${sci(r.realistic)}, do làm tròn số thực.`,
      );
  }
}

function BugSpot() {
  const runs = useMemo(() => Object.fromEntries(FRAGMENTS.map((id) => [id, runFragment(id)])) as Record<FragmentId, FragmentRun>, []);
  const md = useLabFile(FRAG_FILE);
  const texts = useMemo(() => (md && !md.startsWith('// could not load') ? parseFragments(md) : []), [md]);
  const [marks, setMarks] = useState<Partial<Record<FragmentId, boolean>>>({});
  const [checked, setChecked] = useState(false);
  const all = FRAGMENTS.every((id) => marks[id] !== undefined);
  const right = FRAGMENTS.filter((id) => marks[id] === runs[id].safe).length;
  return (
    <Bench
      title={trText('Safe or unsafe?', 'An toàn hay không an toàn?')}
      task={trText(
        'Mark each fragment, then check. Every verdict comes with what the model measured.',
        'Đánh dấu từng đoạn, rồi kiểm tra. Mỗi kết luận đi kèm với những gì mô hình đo được.',
      )}
      foot={
        <Translated
          vi={
            <>
              Các đoạn mã từ <code>labs/{FRAG_FILE}</code>: chúng chỉ cho thấy bộ tích lũy, không gì khác. Các đoạn 2 đến 4 là
              emulator-checked và front-end-checked (một unit test chạy chúng, đúng như in ra, qua{' '}
              <code>labs/x2/snippets/check/accumulators_check.cu</code> và khớp với kết quả của mô hình), unverified trên
              CUDA; đoạn 1, viết bằng MSL, không được biên dịch (unverified). q, x4, w2, x2, v và u là các mã và activation
              của khối; dot4 là phép tính của dp4a. Phép tính của chúng được mô hình hóa trong qdot.ts trên một hàng rộng
              4,096: dữ liệu thực tế sinh từ seed, và một trường hợp xấu nhất gồm mã 15 với 127 cùng hệ số tỉ lệ đơn vị (nên các
              đoạn số thực thấy trọng số 7 và activation 127). Phép dot của half4 được mô hình hóa như một chuỗi phép
              nhân-cộng half; Metal không ghi lại thứ tự của nó.
            </>
          }
        >
        <>
          Fragments from <code>labs/{FRAG_FILE}</code>: they show the accumulator, nothing else. Fragments 2 to 4 are
          emulator-checked and front-end-checked (a unit test runs them, as printed, through{' '}
          <code>labs/x2/snippets/check/accumulators_check.cu</code> and matches the model's results), unverified on
          CUDA; fragment 1, in MSL, is not compiled (unverified). q, x4, w2, x2, v and u are the
          block's codes and activations; dot4 is dp4a's arithmetic. Their arithmetic is modelled in qdot.ts on a
          4,096-wide row: realistic seeded data, and a worst case of codes 15 against 127 with unit scales (so the float
          fragments see weights of 7 and activations of 127). The dot of half4 is modelled as a chain of half
          multiply-adds; Metal does not document its order.
        </>
        </Translated>
      }
    >
      <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(19rem, 1fr))' }}>
        {FRAGMENTS.map((id) => {
          const f = texts.find((t) => t.id === id);
          const r = runs[id];
          const ok = marks[id] === r.safe;
          return (
            <div key={id} className="panel" data-frag={id}>
              <div style={{ fontWeight: 600, marginBottom: '0.3rem' }}>
                {f ? f.title : trText(`Fragment ${FRAGMENTS.indexOf(id) + 1}`, `Đoạn ${FRAGMENTS.indexOf(id) + 1}`)} {f ? <span className="muted">({f.lang})</span> : null}
              </div>
              {f ? (
                <pre className="mono" style={{ fontSize: '0.74rem', margin: '0 0 0.5rem', overflowX: 'auto' }} data-frag-code="">{f.code}</pre>
              ) : (
                <p className="muted" style={{ margin: '0 0 0.5rem' }}>{md ? trText(`Could not read labs/${FRAG_FILE}.`, `Không đọc được labs/${FRAG_FILE}.`) : trText('Loading the fragment.', 'Đang tải đoạn mã.')}</p>
              )}
              <div className="row">
                {[true, false].map((safe) => (
                  <label key={String(safe)} className="check">
                    <input type="radio" name={'frag-' + id} checked={marks[id] === safe}
                      onChange={() => { setMarks({ ...marks, [id]: safe }); setChecked(false); }} />
                    {safe ? trText('safe', 'an toàn') : trText('unsafe', 'không an toàn')}
                  </label>
                ))}
              </div>
              {checked ? (
                <p style={{ margin: '0.5rem 0 0' }} data-verdict={r.safe ? 'safe' : 'unsafe'}>
                  <span className={'chip ' + (ok ? 'ok' : 'fail')}>{ok ? trText('right', 'đúng') : trText('not quite', 'chưa đúng')}: {r.safe ? trText('safe', 'an toàn') : trText('unsafe', 'không an toàn')}</span>{' '}
                  {reason(r, runs['int-then-float'])}
                </p>
              ) : null}
            </div>
          );
        })}
      </div>
      <div className="row" style={{ marginTop: '0.8rem' }}>
        <button className="primary" onClick={() => setChecked(true)} disabled={!all}><Translated vi="Kiểm tra">Check</Translated></button>
        {checked ? <span data-score={right}><Translated vi={<>Đúng {right} trên 4.</>}>{right} of 4 right.</Translated></span> : <span className="muted"><Translated vi="Hãy đánh dấu cả bốn trước.">Mark all four first.</Translated></span>}
      </div>
    </Bench>
  );
}

function BoundReveal({ predicted }: { predicted: number }) {
  return (
    <>
      <Translated
        vi={
        <p data-bound-reveal="">
          Đó là <b>{BOUND_Q8Q8.toLocaleString('en-US')}</b>: 32 tích, mỗi tích tối đa 127 × 127 = 16,129
          {predicted === BOUND_Q8Q8 ? ', đúng con số của bạn' : ''}. Nó có vừa FP16 không? Không, vì hai lẽ. Giá trị hữu hạn lớn
          nhất của FP16 là 65,504, nên tổng bị tràn; và FP16 chỉ giữ chính xác mọi số nguyên đến 2,048, nên nó đã làm tròn từ lâu
          trước đó. Một int32 chứa nó với dư địa gấp bốn nghìn lần, và một float chứa nó chính xác, vì nó nhỏ hơn 2<sup>24</sup>.
        </p>
        }
      >
      <p data-bound-reveal="">
        It is <b>{BOUND_Q8Q8.toLocaleString('en-US')}</b>: 32 products, each at most 127 × 127 = 16,129
        {predicted === BOUND_Q8Q8 ? ', exactly your number' : ''}. Does it fit in FP16? No, twice over. FP16's largest
        finite value is 65,504, so the sum overflows; and FP16 holds every integer only up to 2,048, so it was
        already rounding long before that. An int32 holds it with four thousand times room to spare, and a float holds
        it exactly, because it is below 2<sup>24</sup>.
      </p>
      </Translated>
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSX26: Record<string, Question> = {
  'x2/X2.6/q-q8-bound': {
    kind: 'numeric',
    id: 'x2/X2.6/q-q8-bound',
    prompt: <Translated vi={<>Độ lớn lớn nhất mà tổng số nguyên của một khối Q8_0 × Q8_1 (32 mã int8 nhân 32 mã int8, mỗi mã trong −127..127) có thể đạt tới là bao nhiêu?</>}><>What is the largest magnitude the integer sum of one Q8_0 × Q8_1 block (32 int8 codes times 32 int8 codes, each in −127..127) can reach?</></Translated>,
    answer: 516128,
    explain: <Translated vi={<p>32 × 127 × 127 = 516,128. Vượt 65,504 của FP16; nằm sâu trong phạm vi int32; nhỏ hơn 2<sup>24</sup>, nên chính xác ở float.</p>}><p>32 × 127 × 127 = 516,128. Past FP16's 65,504; far inside int32; below 2<sup>24</sup>, so exact in float.</p></Translated>,
    source: `${FACTS} E3 (our arithmetic)`,
    vi: { source: `${FACTS} E3 (phép tính của chúng tôi)` },
  },
  'x2/X2.6/q-int32-terms': {
    kind: 'numeric',
    id: 'x2/X2.6/q-int32-terms',
    prompt: (
      <Translated
        vi={
          <>
            Một bộ tích lũy int32 cộng các tích int8 trường hợp xấu nhất, mỗi tích 127 × 127 = 16,129. Nó cộng được bao nhiêu
            tích trước khi tích tiếp theo làm tràn?
          </>
        }
      >
      <>
        An int32 accumulator adds worst-case int8 products, each 127 × 127 = 16,129. How many can it add before the
        next one overflows?
      </>
      </Translated>
    ),
    answer: int32Terms(16129),
    explain: (
      <Translated
        vi={
        <p>
          (2<sup>31</sup> − 1) / 16,129 = 133,144.6, nên 133,144 tích vừa và tích thứ 133,145 làm tràn. Không kernel thật nào đến
          gần mức đó, và không phải vì hàng ngắn: mỗi khối 32 có hệ số tỉ lệ riêng, nên tổng số nguyên được nhân hệ số sang số
          thực sau tối đa 32 tích.
        </p>
        }
      >
      <p>
        (2<sup>31</sup> − 1) / 16,129 = 133,144.6, so 133,144 products fit and the 133,145th overflows. No real kernel
        gets near it, and not because rows are short: every block of 32 has its own scales, so the integer sum is
        rescaled into float after 32 products at most.
      </p>
      </Translated>
    ),
    source: `${FACTS} E3 ("about 133,144 worst-case terms"; the exact count is ours)`,
    vi: { source: `${FACTS} E3 ("about 133,144 worst-case terms"; con số chính xác là của chúng tôi)` },
  },
  'x2/X2.6/q-nibble-int16': {
    kind: 'choice',
    id: 'x2/X2.6/q-nibble-int16',
    prompt: <Translated vi={<>Một kernel cộng các nibble không dấu (0..15) của một khối Q4_0 nhân với activation int8, 32 tích, vào một <code>short</code>, mỗi lần bốn tích bằng dp4a. Như vậy có an toàn không?</>}><>A kernel sums one Q4_0 block's unsigned nibbles (0..15) times int8 activations, 32 products, in a <code>short</code>, four at a time with dp4a. Is that safe?</></Translated>,
    options: [
      { label: <Translated vi={`Không: tổng có thể đạt ${BOUND_Q4_UNSIGNED.toLocaleString('en-US')}, vượt 32,767.`}>{`No: the sum can reach ${BOUND_Q4_UNSIGNED.toLocaleString('en-US')}, past 32,767.`}</Translated>, correct: true },
      { label: <Translated vi="Có: 15 × 127 = 1,905 mỗi tích, thấp hơn nhiều so với 32,767.">{'Yes: 15 × 127 = 1,905 per product, far below 32,767.'}</Translated>, why: <Translated vi="Một tích thì vậy; 32 tích thì không.">{'One product is; 32 of them are not.'}</Translated> },
      { label: <Translated vi="Có, miễn là các mã được đưa vào dạng có dấu, là −8..7.">{'Yes, as long as the codes go in signed, as −8..7.'}</Translated>, why: <Translated vi={`Mã có dấu đạt ${BOUND_Q4_SIGNED.toLocaleString('en-US')}, vừa với dư 255. Đó là sự trùng hợp của 32 × 8 × 127, không phải một khoảng an toàn để dựa vào, và llama.cpp đưa vào các mã không dấu.`}>{`Signed codes reach ${BOUND_Q4_SIGNED.toLocaleString('en-US')}, which fits with 255 to spare. That is a coincidence of 32 × 8 × 127, not a margin to build on, and llama.cpp feeds the unsigned codes.`}</Translated> },
      { label: <Translated vi="Không, nhưng chỉ vì dp4a trả về 32 bit.">{'No, but only because dp4a returns 32 bits.'}</Translated>, why: <Translated vi="Kết quả 32 bit của dp4a không có vấn đề; lưu nó vào 16 bit mới là điều làm mất phần trên.">{'dp4a’s 32-bit result is fine; storing it into 16 bits is what loses the top.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
        <p>
          Mỗi dp4a cộng tối đa 4 × 15 × 127 = 7,620: 30,480 sau bốn bước, 38,100 sau năm bước. Thang ở trên cho thấy điều này
          với cặp ghép Q4_0, trường hợp xấu nhất và bốn tích mỗi bước. Trên dữ liệu thực tế nó vẫn qua, và đó là điều khiến nó
          thành một lỗi cài sẵn tốt: lab X2.3 có lỗi này.
        </p>
        }
      >
      <p>
        Each dp4a adds up to 4 × 15 × 127 = 7,620: 30,480 after four steps, 38,100 after five. The ladder above shows it with the Q4_0
        pairing, the worst case and four products per step. On realistic data it passes, which is what makes it a
        good planted defect: lab X2.3 has it.
      </p>
      </Translated>
    ),
    source: `${FACTS} E3; lab X2.3`, // i18n-ignore: a citation
  },
  'x2/X2.6/q-half-prescale': {
    kind: 'choice',
    id: 'x2/X2.6/q-half-prescale',
    prompt: <Translated vi={<>Để tiết kiệm thanh ghi, một kernel Metal lưu các activation đã chia trước (x / 16, x / 256, x / 4096) ở half. Một activation 1e−4 gặp số chia 4096. Điều gì xảy ra?</>}><>To save registers, a Metal kernel stores its pre-divided activations (x / 16, x / 256, x / 4096) in half. An activation of 1e−4 meets the 4096 divisor. What happens?</></Translated>,
    options: [
      { label: <Translated vi="Nó làm tròn về 0, vì nhỏ hơn một nửa số subnormal FP16 nhỏ nhất.">{'It rounds to 0, below half the smallest FP16 subnormal.'}</Translated>, correct: true },
      { label: <Translated vi="Không có gì: chia cho một lũy thừa của hai là chính xác.">{'Nothing: dividing by a power of two is exact.'}</Translated>, why: <Translated vi="Chỉ chính xác khi kết quả còn biểu diễn được. FP16 hết số mũ ở 2^−24.">{'Exact only while the result is representable. FP16 runs out of exponent at 2^−24.'}</Translated> },
      { label: <Translated vi="Nó trở thành một số subnormal FP16 chỉ còn vài bit phần định trị.">{'It becomes an FP16 subnormal with only a few significand bits.'}</Translated>, why: <Translated vi="Nó nhỏ hơn một nửa số subnormal nhỏ nhất, nên không còn gì để làm tròn tới ngoài 0.">{'It is below half the smallest subnormal, so there is nothing left to round to but 0.'}</Translated> },
      { label: <Translated vi="Nó tràn khi nibble đã mask nhân nó trở lại.">{'It overflows once the masked nibble scales it back up.'}</Translated>, why: <Translated vi="Giá trị nhỏ thì underflow; giá trị lớn mới tràn.">{'Small values underflow; it is the large ones that overflow.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
        <p>
          1e−4 / 4096 khoảng 2.4e−8. Giá trị dương nhỏ nhất của FP16 là 2<sup>−24</sup> ≈ 5.96e−8, và mọi thứ nhỏ hơn một nửa
          của nó đều làm tròn về 0, nên phần đóng góp của trọng số đó biến mất. Ở float, cùng thương số đó là một số chuẩn hóa
          (normal) bình thường: hãy giữ activation đã chia trước ở float. Mẹo chia trước chính xác ở float32 (mục X2.5) chính vì float còn dư
          phạm vi số mũ; half thì không.
        </p>
        }
      >
      <p>
        1e−4 / 4096 is about 2.4e−8. FP16's smallest positive value is 2<sup>−24</sup> ≈ 5.96e−8, and anything below
        half of it rounds to 0, so that weight's contribution vanishes. In float the same quotient is an ordinary normal
        number: keep pre-divided activations in float. The pre-division trick is exact in float32 (section X2.5)
        precisely because float has exponent range to spare; half does not.
      </p>
      </Translated>
    ),
    source: `notes/today/appA-facts.md A2 (FP16 smallest denormal 2^-24); ${FACTS} E4`,
    vi: { source: `notes/today/appA-facts.md A2 (số denormal nhỏ nhất của FP16 là 2^-24); ${FACTS} E4` },
  },
  'x2/X2.6/q-metal-dot': {
    kind: 'choice',
    id: 'x2/X2.6/q-metal-dot',
    prompt: <Translated vi={<>Một kernel Metal giải lượng tử hóa bốn trọng số thành <code>half4 w</code>, nạp <code>half4 x</code>, và cộng <code>dot(w, x)</code> vào một float. Phép dot tích lũy ở kiểu gì, và cách sửa là gì?</>}><>A Metal kernel dequantises four weights to <code>half4 w</code>, loads <code>half4 x</code>, and adds <code>dot(w, x)</code> to a float. What does the dot accumulate in, and what is the fix?</></Translated>,
    options: [
      { label: <Translated vi="Ở half; chuyển cả hai sang float4 trước khi dot.">{'In half; convert both to float4 before dot.'}</Translated>, correct: true },
      { label: <Translated vi="Ở float, vì kết quả được cộng vào một float.">{'In float, because the result is added to a float.'}</Translated>, why: <Translated vi="Việc chuyển sang float xảy ra sau khi phép dot đã được tính ở half.">{'The conversion to float happens after the dot has already been computed in half.'}</Translated> },
      { label: <Translated vi="Ở float: Metal luôn mở rộng phép tích vô hướng.">{'In float: Metal always widens dot products.'}</Translated>, why: <Translated vi="Đặc tả nói phép dot của một halfn có kiểu vô hướng half.">{'The spec says the dot of a halfn has the scalar type half.'}</Translated> },
      { label: <Translated vi="Ở half; cho bộ tích lũy của hàng cũng là half.">{'In half; make the row accumulator half too.'}</Translated>, why: <Translated vi="Như vậy còn tệ hơn: đoạn 2 của phần tìm lỗi.">{'That makes it worse: fragment 2 of the bug spot.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
        <p>
          Đặc tả MSL định nghĩa <code>dot</code> cho floatn và halfn, trả về kiểu vô hướng của các đối số, nên phép dot của một
          halfn là half. <code>mul_mv</code> của llama.cpp viết <code>dot(float4(xb4[i]), float4(yl4[i]))</code>.
        </p>
        }
      >
      <p>
        The MSL specification defines <code>dot</code> for floatn and halfn, returning the scalar type of its arguments,
        so the dot of a halfn is half.
        llama.cpp's <code>mul_mv</code> writes <code>dot(float4(xb4[i]), float4(yl4[i]))</code>.
      </p>
      </Translated>
    ),
    source: `${FACTS} C1`,
  },
  'x2/X2.6/q-order': {
    kind: 'choice',
    id: 'x2/X2.6/q-order',
    prompt: <Translated vi={<>Kernel của lab X2.2 cho đầu ra khác nhau ở các chữ số cuối giữa hai backend. Đó có phải là lỗi không?</>}><>Lab X2.2's kernel gives outputs that differ in the last digits between two backends. Is that a bug?</></Translated>,
    options: [
      { label: <Translated vi="Không hẳn: các tổng số thực có thể chạy theo thứ tự khác nhau.">{'Not necessarily: the float sums may run in different orders.'}</Translated>, correct: true },
      { label: <Translated vi="Có: số học số nguyên phải cho kết quả giống hệt nhau.">{'Yes: integer arithmetic should make results identical.'}</Translated>, why: <Translated vi="Chỉ các tổng bên trong một khối là số nguyên; mọi thứ giữa các khối là số thực, cộng theo thứ tự mà mỗi kernel tự chọn.">{'Only the sums inside a block are integers; everything across blocks is float, added in whatever order each kernel chose.'}</Translated> },
      { label: <Translated vi="Có lẽ: một backend có thể đang tích lũy ở half ở đâu đó.">{'Probably: one backend is likely accumulating in half somewhere.'}</Translated>, why: <Translated vi="Có thể, nhưng thứ tự cộng số thực khác nhau đã đủ giải thích chênh lệch ở chữ số cuối.">{'Possible, but different orders of float additions are enough to explain a last-digit difference.'}</Translated> },
      { label: <Translated vi="Không: số học số thực trên GPU thay đổi giữa các lần chạy, nên cứ bỏ qua.">{'No: GPU float arithmetic varies from run to run, so ignore it.'}</Translated>, why: <Translated vi="Một kernel cho trước trên một thiết bị cho trước thường lặp lại được; khác biệt đến từ thứ tự các phép cộng, vốn là thuộc tính của mã.">{'A given kernel on a given device is usually repeatable; the difference comes from the order of the additions, which is a property of the code.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
        <p>
          Ý chính của phụ lục A, từ §A.6: phép cộng số thực không có tính kết hợp, nên thứ tự quyết định các bit thấp của một tổng
          dài. Một warp 32 lane so với một số lượng lane khác, một cách chia hàng khác, hay một cây thay cho một chuỗi đều sắp xếp lại nó.
          Một bài kiểm tra nên cho phép một sai số tỉ lệ với tổng độ lớn của các số hạng, và nên thất bại với bất cứ thứ gì lớn
          hơn.
        </p>
        }
      >
      <p>
        Appendix A's point, from §A.6: float addition is not associative, so order decides the low bits of a long sum. A 32-lane warp and a different lane
        count, a different split of the row, or a tree instead of a chain all reorder it. A test should allow an error
        scaled by the sum of the magnitudes of the terms, and should fail on anything larger.
      </p>
      </Translated>
    ),
    source: 'book appendix A §A.6, pp. 593–594',
    vi: { source: 'sách, phụ lục A §A.6, pp. 593–594' },
  },
};

// ---------------------------------------------------------------- the page
export function SX26() {
  return (
    <>
      <Translated
        vi={
        <p>
          Phụ lục A nêu một điểm về các tổng dài: thứ tự và độ rộng của chúng quyết định bao nhiêu phần của đáp án còn sót lại.
          Một bộ tích lũy half chỉ giữ chính xác số nguyên đến 2,048 và tràn ở 65,504; lab A.1 đã đo điều đó tác động thế nào lên
          một tổng dài trên chính thiết bị của bạn. Một phép tích vô hướng lượng tử hóa là một tổng ở hai cấp, bên trong mỗi khối
          và giữa các khối, và mỗi cấp cần câu trả lời riêng cho câu hỏi "rộng bao nhiêu?".
        </p>
        }
      >
      <p>
        Appendix A made one point about long sums: their order and their width decide how much of the answer
        survives. A half accumulator holds integers exactly only to 2,048 and overflows at 65,504; lab A.1 measured
        what that does to a long sum on your own device. A quantised dot product is a sum on two levels, inside each
        block and across the blocks, and each level needs its own answer to "how wide?".
      </p>
      </Translated>
      <Source section="A.6" pages="593–594" />
      <Beyond title={trText("Ours, not the book's", 'Của khóa học này, không phải của sách')}>
        <Translated
          vi={
          <p>
            Phụ lục không có định dạng số nguyên. Các cận ở đây là phép tính của chúng tôi; việc các kernel tích lũy ở kiểu gì đã
            được kiểm tra trong lượt kiểm tra dữ kiện ngày 2026-11-20 ({FACTS} E3, C1, D3 đến D5). Giới hạn của FP16: phụ lục A,
            §A.2, và tệp dữ kiện của nó, A2. Lab A.1 nằm trên <a href={href('appA', 'labs')}>trang lab của phụ lục</a>.
          </p>
          }
        >
        <p>
          The appendix has no integer formats. The bounds here are our arithmetic; what the kernels accumulate in was
          checked in the facts pass on 2026-11-20 ({FACTS} E3, C1, D3 to D5). FP16's limits: appendix A, §A.2, and
          its facts file, A2. Lab A.1 is on the <a href={href('appA', 'labs')}>appendix's labs page</a>.
        </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Tìm cận cho mọi tổng riêng phần">Bound every partial sum</Translated></h2>
      <Translated
        vi={
        <p>
          Bắt đầu bên trong một khối. Với trọng số Q8_0 và activation Q8_1, mỗi tích có độ lớn tối đa 127 × 127, và một khối có
          32 tích như vậy. Trước khi đọc tiếp, hãy tính tổng lớn nhất.
        </p>
        }
      >
      <p>
        Start inside a block. With Q8_0 weights and Q8_1 activations, every product is at most 127 × 127 in magnitude,
        and a block has 32 of them. Before you read on, work out the largest sum.
      </p>
      </Translated>
      <Predict id="x2/X2.6/p-q8-bound" prompt={<Translated vi={<>Độ lớn lớn nhất của tổng số nguyên của một khối Q8_0 × Q8_1?</>}><>The largest magnitude of one Q8_0 × Q8_1 block's integer sum?</></Translated>}>
        {(v) => <BoundReveal predicted={v} />}
      </Predict>
      <Translated
        vi={
        <p>
          Trường hợp 4 bit phụ thuộc vào cách đưa mã vào. Kernel CUDA của llama.cpp đưa vào các nibble không dấu, 0 đến 15, và
          hiệu chỉnh cho số 8 sau: tối đa 15 × 127 × 32 = 60,960 mỗi khối, nhiều hơn mức một số nguyên 16 bit chứa được. Mã có dấu,
          −8 đến 7, đạt 8 × 127 × 32 = 32,512, vừa một số nguyên 16 bit với dư 255. Và một int32 sẽ nhận được 133,144 tích int8
          trường hợp xấu nhất trước khi tràn. Không kernel nào cộng nhiều như thế mà không nhân lại hệ số, vì mỗi khối có hệ số
          tỉ lệ riêng.
        </p>
        }
      >
      <p>
        The 4-bit case depends on how the codes go in. llama.cpp's CUDA kernel feeds the unsigned nibbles, 0 to 15,
        and corrects for the 8 later: up to 15 × 127 × 32 = 60,960 per block, more than a 16-bit integer holds. Signed
        codes, −8 to 7, reach 8 × 127 × 32 = 32,512, which fits a 16-bit integer by 255. And an int32 would take
        133,144 worst-case int8 products before overflowing. No kernel adds that many without rescaling, because every
        block has its own scale.
      </p>
      </Translated>
      <Ask q={questionsSX26['x2/X2.6/q-q8-bound']} label={trText('Check the bound', 'Kiểm tra cận')} />
      <Ask q={questionsSX26['x2/X2.6/q-int32-terms']} label={trText('Work it out', 'Tính ra')} />
      <Ask q={questionsSX26['x2/X2.6/q-nibble-int16']} />

      <h2><Translated vi="Xem các lựa chọn hẹp thất bại">Watch the narrow choices fail</Translated></h2>
      <Translated
        vi={
        <p>
          Thang cộng cả một hàng ở cả hai cấp với các bộ tích lũy bạn chọn. Biểu đồ cho thấy tổng trong của khối đầu tiên theo từng
          bước, trên thang log, so với giá trị lớn nhất mà kiểu trong chứa được. Bên dưới là điều đã xảy ra với mọi khối và với
          hàng.
        </p>
        }
      >
      <p>
        The ladder sums a whole row on both levels with the accumulators you choose. The plot shows the first block's
        inner sum step by step, on a log scale, against the largest value the inner type holds. Below it is what
        happened to every block and to the row.
      </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
          <p>
            Trường hợp xấu nhất là được dựng lên: dữ liệu thực tế hiếm khi đến gần các cận. Nhưng một kernel phải đúng với mọi đầu
            vào nó có thể nhận, và trường hợp xấu nhất là một trong số đó. Mô hình không có lane và không đo thời gian.
          </p>
          }
        >
        <p>
          The worst case is constructed: realistic data rarely gets near the bounds. But a kernel has to be right for
          every input it can be given, and the worst case is one of them. The model has no lanes and no timing.
        </p>
        </Translated>
      </Simplification>
      <Ladder />
      <Translated
        vi={
        <p>
          Bốn điều cần tìm. int16 quay vòng bên trong một khối xấu nhất, và với mã Q8 nó cũng quay vòng ở vài khối thực tế. FP16
          làm tròn ngay từ tích đầu tiên (16,129 không phải là một số half) và đạt vô cực ở tích thứ năm. Với int32 bên trong,
          một tổng ngoài FP16 trôi dần khi bạn kéo hàng dài ra: sai số tuyệt đối của nó tăng theo hàng. int32 rồi FP32 giữ ở mức
          làm tròn số thực ở mọi độ dài.
        </p>
        }
      >
      <p>
        Four things to find. int16 wraps inside one worst-case block, and with Q8 codes it wraps on a few realistic
        blocks too. FP16 rounds from the very first product (16,129 is not a half) and reaches infinity on the fifth.
        With int32 inside, an FP16 outer sum drifts as you drag the row longer: its absolute error grows with the row.
        int32 then FP32 stays at float rounding at every length.
      </p>
      </Translated>
      <Translated
        vi={
        <p>
          Hoạt ảnh cộng một khối xấu nhất vào cả bốn kiểu cạnh nhau, rồi cộng cả một hàng qua các khối của nó ở half và ở float.
        </p>
        }
      >
      <p>
        The animation adds one worst-case block into all four types side by side, then sums a whole row across its
        blocks in half and in float.
      </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
          <p>
            Mọi giá trị đến từ <code>qdot.ts</code>: mỗi lần một tích, theo thứ tự, mỗi phép cộng làm tròn theo cách kiểu của nó
            làm tròn. Không có lane, không có thứ tự khác, không gì được đo thời gian.
          </p>
          }
        >
        <p>
          Every value comes from <code>qdot.ts</code>: one product at a time, in order, each add rounded as its type
          rounds. No lanes, no other orders, nothing timed.
        </p>
        </Translated>
      </Simplification>
      <NarrowAccumulatorScene />

      <h2><Translated vi="Quy tắc các kernel thật tuân theo">The rule the real kernels follow</Translated></h2>
      <Translated
        vi={
        <p>
          Số nguyên bên trong một khối, số thực giữa các khối. Decode CUDA của llama.cpp giữ tổng dp4a của khối trong một{' '}
          <code>int</code>, biến mỗi khối thành một số thực, và cộng các khối vào các tổng riêng phần số thực trước phép rút gọn
          trên warp. Trên đường số thực, các bộ tích lũy cũng là số thực: các kernel Metal của llama.cpp cộng ở float, và kernel
          ma trận-vector lượng tử hóa của MLX khai báo kiểu bộ tích lũy là float bất kể kiểu đầu vào.
        </p>
        }
      >
      <p>
        Integer inside a block, float across blocks. llama.cpp's CUDA decode keeps the block's dp4a sum in an{' '}
        <code>int</code>, turns each block into a float, and adds the blocks into float partial sums before the warp
        reduction. On the float path the accumulators are float too: llama.cpp's Metal kernels sum in float, and MLX's
        quantised matrix-vector kernel declares its accumulator type as float whatever the input type.
      </p>
      </Translated>
      <Today checked={`${CHECKED}, D3, D4, D5)`}>
        <Translated
          vi={
          <p>
            llama.cpp: <code>int sumi</code> cho mỗi khối và <code>float tmp[..]</code> giữa các khối trong{' '}
            <code>mul_mat_vec_q</code>. MLX: <code>typedef float U;</code> trong <code>qmv_fast_impl</code>.
          </p>
          }
        >
        <p>
          llama.cpp: <code>int sumi</code> per block and <code>float tmp[..]</code> across blocks in{' '}
          <code>mul_mat_vec_q</code>. MLX: <code>typedef float U;</code> in <code>qmv_fast_impl</code>.
        </p>
        </Translated>
      </Today>
      <Translated
        vi={
        <p>
          Cái bẫy nằm ở đâu tùy vào backend. Trên Metal, <code>dot</code> của hai vector half4 trả về một half, nên một kernel giải
          lượng tử hóa về half rồi gọi nó có một bộ tích lũy half mà không nói ra; chuyển sang float4 trước, như llama.cpp làm, sẽ
          loại bỏ nó. CUDA hoàn toàn không có phép tích vô hướng half, và một chuỗi{' '}
          <code>__hfma2</code> tích lũy ở half. HIP có một phép tích lũy ở float,{' '}
          <code>amd_mixed_dot(__half2, __half2, float, bool)</code>, từ một header không có tài liệu.
        </p>
        }
      >
      <p>
        Where the trap is depends on the backend. On Metal, <code>dot</code> of two half4 vectors returns a half, so a
        kernel that dequantises to half and calls it has a half accumulator without saying so; converting to float4
        first, as llama.cpp does, removes it. CUDA has no half dot product at all, and a chain of{' '}
        <code>__hfma2</code> accumulates in half. HIP has one that accumulates in float,{' '}
        <code>amd_mixed_dot(__half2, __half2, float, bool)</code>, from an undocumented header.
      </p>
      </Translated>
      <Today checked={`${CHECKED}, C1, A3, B4)`}>
        <Translated
          vi={
          <p>
            MSL 4.1, mục 6.9: với đối số halfn, dot trả về half. Trang half2 của CUDA Math API không có hàm dot.{' '}
            <code>amd_hip_fp16.h</code> của HIP định nghĩa phép tích lũy ở float.
          </p>
          }
        >
        <p>
          MSL 4.1, section 6.9: for halfn arguments, dot returns half. The CUDA Math API's half2 page has no dot
          function. HIP's <code>amd_hip_fp16.h</code> defines the float-accumulating one.
        </p>
        </Translated>
      </Today>
      <Ask q={questionsSX26['x2/X2.6/q-metal-dot']} />
      <Translated
        vi={
        <p>
          ggml, thư viện tensor của llama.cpp, tiến thêm một bước và biến bộ tích lũy thành một thuộc tính của đồ thị: một phép toán
          có thể được đánh dấu là cần tích lũy F32, hoặc cho phép BF16 hay F16. Đó chính là lựa chọn bạn vừa làm với thang, được ghi
          lại ở nơi mọi backend đều đọc được.
        </p>
        }
      >
      <p>
        ggml, llama.cpp's tensor library, goes one step further and makes the accumulator a property of the graph: an
        operation can be marked as needing F32 accumulation, or allowing BF16 or F16. It is the same choice you just
        made with the ladder, written down where every backend can read it.
      </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/appA-facts.md, D3)">{/* i18n-ignore: a citation */}
        <Translated
          vi={
          <p>
            <code>ggml_prec_set_acc</code> với <code>GGML_PREC_F32</code>, <code>_BF16</code> hoặc <code>_F16</code>, trong{' '}
            <code>ggml.h</code>; llama.cpp yêu cầu F32 cho flash attention.
          </p>
          }
        >
        <p>
          <code>ggml_prec_set_acc</code> with <code>GGML_PREC_F32</code>, <code>_BF16</code> or <code>_F16</code>, in{' '}
          <code>ggml.h</code>; llama.cpp asks for F32 on flash attention.
        </p>
        </Translated>
      </Today>

      <h2><Translated vi="Tìm bộ tích lũy hẹp">Spot the narrow accumulator</Translated></h2>
      <Translated
        vi={
        <p>
          Bốn đoạn của vòng lặp trong, theo cách bạn có thể gặp chúng trong một kernel. Mỗi đoạn tích lũy ở đâu đó; hãy quyết định
          đoạn nào an toàn với mọi đầu vào.
        </p>
        }
      >
      <p>
        Four fragments of the inner loop, the way you might find them in a kernel. Each one accumulates somewhere;
        decide which are safe for every input.
      </p>
      </Translated>
      <BugSpot />

      <h2><Translated vi="Activation đã chia trước giữ ở float">Pre-divided activations stay float</Translated></h2>
      <Translated
        vi={
        <p>
          Đường số thực của mục X2.5 chia activation cho 16, 256 và 4096 để các nibble đã mask không cần dịch bit. Điều đó chính xác
          ở float32 vì float còn dư phạm vi số mũ. Half thì không: giá trị dương nhỏ nhất của nó khoảng 6e−8, và một activation
          nhỏ chia cho 4096 có thể rơi xuống dưới mức đó.
        </p>
        }
      >
      <p>
        Section X2.5's float path divides activations by 16, 256 and 4096 so that masked nibbles need no shift. That is
        exact in float32 because float has exponent range to spare. Half does not: its smallest positive value is
        about 6e−8, and a small activation divided by 4096 can fall below it.
      </p>
      </Translated>
      <Ask q={questionsSX26['x2/X2.6/q-half-prescale']} />

      <h2><Translated vi="Nó xuất hiện tiếp ở đâu">Where it shows up next</Translated></h2>
      <Translated
        vi={
        <p>
          Lab X2.3 (<a href={href('x2', 'labs')}>trang lab</a>) giấu bốn lỗi trong một GEMV đường số nguyên. Một trong số đó chính
          là mục này: một tổng khối hẹp hơn một kiểu, qua được trên dữ liệu thực tế và thất bại ở trường hợp xấu nhất. Và khi
          đầu ra của nó khác với của một backend khác ở các chữ số cuối, câu hỏi bên dưới là câu nên hỏi trước tiên.
        </p>
        }
      >
      <p>
        Lab X2.3 (<a href={href('x2', 'labs')}>labs page</a>) hides four defects in an integer-path GEMV. One of them is
        exactly this section: a block sum one type too narrow, which passes on realistic data and fails on the worst
        case. And when its outputs differ from another backend's in the last digits, the question below is the one to
        ask first.
      </p>
      </Translated>
      <Ask q={questionsSX26['x2/X2.6/q-order']} />
    </>
  );
}
