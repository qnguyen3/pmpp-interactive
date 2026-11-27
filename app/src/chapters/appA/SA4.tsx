import { useEffect, useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';
import {
  ALIGN_PRESETS,
  BF16_PRESETS,
  MODE_LABEL,
  MODES,
  alignTrace,
  bf16FromBits,
  bitsFromF32,
  parseOperand,
  type AlignTrace,
  type Step,
} from '../../sim/fpadd';
import { BF16, FP16, FP32, TOY5, show, type Format, type Mode } from '../../sim/minifloat';
import { DisappearingAddendScene } from './DisappearingAddendScene';

const FACTS = 'notes/today/appA-facts.md';
const CHECKED = (items: string) => `2026-11-20 (${FACTS}, ${items})`; // i18n-ignore: a date and a citation

// ---------------------------------------------------------------- the alignment stepper
const STEP_FORMATS: Format[] = [TOY5, FP16, BF16, FP32];
const FORMAT_LABEL: Record<string, string> = {
  get toy5() {
    return tr("the book's 5-bit", '5-bit của sách');
  },
  fp16: 'FP16',
  bf16: 'BF16',
  fp32: 'FP32',
};

const dotted = (w: string, carry = false) => (carry ? `${w.slice(0, 2)}.${w.slice(2)}` : `${w[0]}.${w.slice(1)}`);
const ulpsText = (e: number, belowOne = true) =>
  !Number.isFinite(e) ? '∞' : e >= 0.9995 && e < 1 + 1e-12 && belowOne && e !== 0 ? tr('just under 1', 'gần bằng 1') : String(Number(e.toPrecision(4)));

function Tail({ bits }: { bits: string }) {
  if (!bits) return <span className="muted">{tr('(none)', '(không có)')}</span>;
  const shown = bits.length > 30 ? bits.slice(0, 28) : bits;
  return (
    <span className="mono" data-tail={bits}>
      {[...shown].map((b, i) => (
        <span key={i} style={i < 2 ? { textDecoration: 'underline', textUnderlineOffset: '3px' } : { color: 'var(--ink-2)' }}>{b}</span>
      ))}
      {bits.length > 30 ? <span className="muted"><Translated vi={<>… (thêm {bits.length - 28})</>}><>… ({bits.length - 28} more)</></Translated></span> : null}
    </span>
  );
}

function GRSChips({ g, r, s }: { g: number; r: number; s: number }) {
  return (
    <span className="mono" data-grs={`${g}${r}${s}`}>
      <Translated vi={<>bit guard {g} · bit round {r} · bit sticky {s}</>}>
        <>guard {g} · round {r} · sticky {s}</>
      </Translated>
    </span>
  );
}

function describe(st: Step, t: AlignTrace): string {
  switch (st.kind) {
    case 'compare': {
      const same = st.diff === 0;
      return (
        tr(
          `Compare the exponents: ${show(st.big.value)} is 1.${st.big.sig.slice(1)} × 2^${st.big.exp} and ${show(st.small.value)} is 1.${st.small.sig.slice(1)} × 2^${st.small.exp}`,
          `So sánh các số mũ: ${show(st.big.value)} là 1.${st.big.sig.slice(1)} × 2^${st.big.exp} và ${show(st.small.value)} là 1.${st.small.sig.slice(1)} × 2^${st.small.exp}`,
        ) +
        (st.small.stored === 'subnormal' ? tr(' (stored as a denormal, written here with its leading 1)', ' (được lưu dưới dạng denormal, ở đây viết kèm bit 1 dẫn đầu)') : '') +
        (same
          ? tr('. The exponents are equal: nothing to align.', '. Các số mũ bằng nhau: không cần căn chỉnh gì.')
          : tr(
              `. They differ by ${st.diff}, so the smaller significand moves ${st.diff} place${st.diff > 1 ? 's' : ''} right.`,
              `. Chúng chênh nhau ${st.diff}, nên phần định trị nhỏ hơn dịch sang phải ${st.diff} vị trí.`,
            )) +
        (st.subtract ? tr(' The signs differ, so the magnitudes are subtracted.', ' Hai dấu khác nhau, nên các độ lớn được trừ cho nhau.') : '')
      );
    }
    case 'shift':
      return st.jump
        ? tr(
            `The remaining ${st.k - (t.p + 2)} places only push zeros past the round bit, so they are done at once: ${st.k} places in all, exponent ${st.exp}. The sticky bit stays ${st.s}.`,
            `${st.k - (t.p + 2)} vị trí còn lại chỉ đẩy các số 0 qua bit round, nên chúng được làm một lần: tổng cộng ${st.k} vị trí, số mũ ${st.exp}. Bit sticky vẫn là ${st.s}.`,
          )
        : tr(`Shift ${st.k}: one place right, exponent up to ${st.exp}. `, `Lần dịch ${st.k}: sang phải một vị trí, số mũ tăng lên ${st.exp}. `) +
            (st.k === 1
              ? tr('The bit that leaves the window becomes the guard bit.', 'Bit rời khỏi cửa sổ trở thành bit guard.')
              : st.k === 2
                ? tr('The guard bit moves to the round position and a new guard bit arrives.', 'Bit guard chuyển sang vị trí round và một bit guard mới đến.')
                : tr('Everything past the round bit is folded into sticky: it is 1 if any of those bits is 1.', 'Mọi thứ sau bit round được gộp vào sticky: nó bằng 1 nếu bất kỳ bit nào trong số đó bằng 1.'));
    case 'add':
      return tr(
        `${st.subtract ? 'Subtract' : 'Add'} exactly. The exact result is ${st.exactText}${st.carry ? ', and it carried into a new leading bit' : ''}. It needs more bits than the window holds unless the part below the window is all zeros.`,
        `${st.subtract ? 'Trừ' : 'Cộng'} chính xác. Kết quả chính xác là ${st.exactText}${st.carry ? ', và nó nhớ sang một bit dẫn đầu mới' : ''}. Nó cần nhiều bit hơn số bit cửa sổ chứa được, trừ khi phần dưới cửa sổ toàn là số 0.`,
      );
    case 'normalise':
      return st.subnormal
        ? tr(
            `Normalise: the result is below the smallest normal value, so it keeps the smallest exponent, ${st.exp}, and stays denormal.`,
            `Chuẩn hóa: kết quả nhỏ hơn giá trị chuẩn nhỏ nhất, nên nó giữ số mũ nhỏ nhất, ${st.exp}, và vẫn là denormal.`,
          )
        : st.moved > 0
          ? tr(
              `Normalise: the carry makes the result 1x.xx, so it shifts right one place and the exponent goes up to ${st.exp}. One more bit leaves the window.`,
              `Chuẩn hóa: phép nhớ làm kết quả thành 1x.xx, nên nó dịch sang phải một vị trí và số mũ tăng lên ${st.exp}. Thêm một bit rời khỏi cửa sổ.`,
            )
          : st.moved < 0
            ? tr(
                `Normalise: the subtraction cancelled the leading bit${st.moved < -1 ? 's' : ''}. Shift left ${-st.moved} place${st.moved < -1 ? 's' : ''}; the exponent drops to ${st.exp}.`,
                `Chuẩn hóa: phép trừ đã triệt tiêu (các) bit dẫn đầu. Dịch sang trái ${-st.moved} vị trí; số mũ giảm xuống ${st.exp}.`,
              )
            : tr(`Normalise: the result already has the form 1.xx × 2^${st.exp}.`, `Chuẩn hóa: kết quả đã có dạng 1.xx × 2^${st.exp}.`);
    case 'round': {
      const half = st.reason === 'tie-even' || st.reason === 'tie-odd';
      const lead =
        st.reason === 'exact'
          ? tr('Nothing was cut off: the result is exact.', 'Không có gì bị cắt bỏ: kết quả là chính xác.')
          : st.reason === 'below-half'
            ? tr('Guard 0: less than half a unit was cut off, so to nearest keeps the window as it is.', 'Guard 0: phần bị cắt nhỏ hơn nửa đơn vị, nên làm tròn về gần nhất giữ nguyên cửa sổ.')
            : st.reason === 'above-half'
              ? tr('Guard 1 and something after it: more than half a unit was cut off, so to nearest rounds up.', 'Guard 1 và có gì đó sau nó: phần bị cắt lớn hơn nửa đơn vị, nên làm tròn về gần nhất làm tròn lên.')
              : half
                ? tr(
                    `Guard 1, round 0, sticky 0: exactly half a unit, a tie. The window's last bit is ${st.last}, so nearest-even ${st.reason === 'tie-even' ? 'keeps it (already even)' : 'rounds up to make it even'}.`,
                    `Guard 1, round 0, sticky 0: đúng nửa đơn vị, một trường hợp hòa. Bit cuối của cửa sổ là ${st.last}, nên làm tròn về số chẵn gần nhất ${st.reason === 'tie-even' ? 'giữ nguyên nó (đã chẵn)' : 'làm tròn lên để nó thành chẵn'}.`,
                  )
                : st.reason === 'truncate'
                  ? tr('This mode points toward zero here, so the bits below the window are simply dropped.', 'Ở đây chế độ này hướng về phía 0, nên các bit dưới cửa sổ đơn giản là bị bỏ đi.')
                  : tr('This mode points away from zero here, and something was cut off, so it rounds up.', 'Ở đây chế độ này hướng ra xa 0, và có phần bị cắt bỏ, nên nó làm tròn lên.');
      const err = st.reason === 'exact' ? '' : tr(` Error: ${ulpsText(st.errorUlps, st.errorBelowOne)} ULP.`, ` Sai số: ${ulpsText(st.errorUlps, st.errorBelowOne)} ULP.`);
      const over = st.overflow
        ? tr(
            ` That is past the largest ${t.f.id === 'toy5' ? 'value of the format, which has no infinity, so it stays at the largest value' : 'finite value'}: the format stores ${show(st.result)}.`,
            ` Giá trị đó vượt quá ${t.f.id === 'toy5' ? 'giá trị lớn nhất của định dạng, vốn không có vô cực, nên nó dừng ở giá trị lớn nhất' : 'giá trị hữu hạn lớn nhất'}: định dạng lưu ${show(st.result)}.`,
          )
        : '';
      return tr(`${lead} Result: ${show(st.result, 10)}.${err}${over}`, `${lead} Kết quả: ${show(st.result, 10)}.${err}${over}`);
    }
    case 'zero-operand':
      return tr(`One operand is zero, so nothing needs aligning: the result is ${show(st.result, 10)}.`, `Một toán hạng bằng 0, nên không cần căn chỉnh gì: kết quả là ${show(st.result, 10)}.`);
    case 'exact-zero':
      return tr(`The magnitudes cancel exactly. IEEE gives +0 here (−0 when rounding down): ${show(st.result)}.`, `Các độ lớn triệt tiêu nhau hoàn toàn. Ở đây IEEE cho +0 (−0 khi làm tròn xuống): ${show(st.result)}.`);
    case 'special':
      return st.why === 'nan-in' ? tr('A NaN goes in, a NaN comes out.', 'NaN vào thì NaN ra.') : st.why === 'inf-minus-inf' ? tr('∞ − ∞ is undefined: NaN.', '∞ − ∞ không xác định: NaN.') : tr(`An infinite operand: the result is ${show(st.result)}.`, `Một toán hạng vô cực: kết quả là ${show(st.result)}.`);
  }
}

function RoundLine({ st, exact }: { st: Extract<Step, { kind: 'round' }>; exact: number }) {
  const lo = Math.min(st.below, st.above);
  const hi = Math.max(st.below, st.above);
  const u = hi - lo;
  const a = lo - 0.35 * u;
  const b = hi + 0.35 * u;
  // A narrow canvas, so the labels stay readable when a phone scales it down.
  const sx = (v: number) => 20 + ((v - a) / (b - a)) * 400;
  const chosen = st.pick === 'above' ? st.above : st.below;
  return (
    <svg viewBox="0 0 440 74" style={{ width: '100%', maxWidth: 460, display: 'block' }} role="img" aria-label={tr('The exact result between its two neighbours', 'Kết quả chính xác nằm giữa hai giá trị lân cận')} data-chosen={chosen}>
      <line x1={10} x2={430} y1={36} y2={36} stroke="var(--ink-2)" />
      {[st.below, st.above].map((v) => (
        <g key={v}>
          <line x1={sx(v)} x2={sx(v)} y1={26} y2={46} stroke={v === chosen ? 'var(--ok)' : 'var(--ink)'} strokeWidth={v === chosen ? 3 : 1.4} />
          <text x={sx(v)} y={64} fontSize="14" textAnchor="middle" style={{ fill: v === chosen ? 'var(--ok)' : 'var(--ink)' }} fontWeight={v === chosen ? 700 : 400}>
            {show(v, 10)}
          </text>
        </g>
      ))}
      {Number.isFinite(exact) && exact >= a && exact <= b ? (
        <g>
          <circle cx={sx(exact)} cy={36} r={4.5} fill="var(--ink)" />
          <text x={sx(exact)} y={18} fontSize="12.5" textAnchor="middle">{tr('exact', 'chính xác')}</text>
        </g>
      ) : null}
    </svg>
  );
}

function Stepper() {
  const [fmtId, setFmtId] = useState('toy5');
  const [aText, setAText] = useState('2');
  const [bText, setBText] = useState('0.25');
  const [mode, setMode] = useState<Mode>('rne');
  const [n, setN] = useState(0);
  const f = STEP_FORMATS.find((x) => x.id === fmtId)!;
  const a = parseOperand(aText);
  const b = parseOperand(bText);
  const trace = useMemo(() => (a === null || b === null ? null : alignTrace(f, a, b, mode)), [f, a, b, mode]);
  // Changing the mode keeps your place: it only changes the last step.
  useEffect(() => setN(0), [fmtId, aText, bText]);
  const total = trace?.steps.length ?? 0;
  const at = Math.min(n, total);
  const steps = trace?.steps.slice(0, at) ?? [];
  const cur = at > 0 ? trace!.steps[at - 1] : undefined;
  const compare = steps.find((s): s is Extract<Step, { kind: 'compare' }> => s.kind === 'compare');
  const shift = [...steps].reverse().find((s): s is Extract<Step, { kind: 'shift' }> => s.kind === 'shift');
  const add = steps.find((s): s is Extract<Step, { kind: 'add' }> => s.kind === 'add');
  const norm = steps.find((s): s is Extract<Step, { kind: 'normalise' }> => s.kind === 'normalise');
  const rnd = steps.find((s): s is Extract<Step, { kind: 'round' }> => s.kind === 'round');
  const firstAfterShifts = trace ? trace.steps.findIndex((s) => s.kind === 'add') + 1 : 0;

  const preset = (id: string) => {
    const p = ALIGN_PRESETS.find((x) => x.id === id)!;
    setFmtId(p.fmt.id);
    setAText(p.aText);
    setBText(p.bText);
  };

  return (
    <Bench
      title={tr('Alignment and rounding', 'Căn chỉnh và làm tròn')}
      task={tr(
        'Step through one addition. Then change the rounding mode at the last step and watch which neighbour wins.',
        'Chạy từng bước một phép cộng. Rồi đổi chế độ làm tròn ở bước cuối và xem giá trị lân cận nào thắng.',
      )}
      foot={
        <Translated
          vi={
            <>
              Mỗi bước đến từ tổng chính xác của hai giá trị được lưu; kết quả được kiểm tra với một phép làm tròn độc lập
              của cùng tổng đó ở cả bốn chế độ. Gạch chân: các bit guard và round.
            </>
          }
        >
          <>
            Every step comes from the exact sum of the two stored values; the result is checked against an independent
            rounding of the same sum in all four modes. Underlined: the guard and round bits.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label={tr('Format', 'Định dạng')}>
          {STEP_FORMATS.map((x) => (
            <button key={x.id} aria-pressed={x.id === fmtId} onClick={() => setFmtId(x.id)}>{FORMAT_LABEL[x.id]}</button>
          ))}
        </span>
        <label className="field">
          a
          <input type="text" value={aText} onChange={(e) => setAText(e.target.value)} aria-label={tr('operand a', 'toán hạng a')} style={{ width: '8rem' }} />
        </label>
        <label className="field">
          b
          <input type="text" value={bText} onChange={(e) => setBText(e.target.value)} aria-label={tr('operand b', 'toán hạng b')} style={{ width: '8rem' }} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="muted"><Translated vi="Mẫu có sẵn:">Presets:</Translated></span>
        {ALIGN_PRESETS.map((p) => (
          <button key={p.id} className="quiet" onClick={() => preset(p.id)} data-preset={p.id}>{p.label}</button>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="muted"><Translated vi="Làm tròn:">Rounding:</Translated></span>
        <span className="seg" role="group" aria-label={tr('Rounding mode', 'Chế độ làm tròn')}>
          {MODES.map((m) => (
            <button key={m} aria-pressed={m === mode} onClick={() => setMode(m)}>{MODE_LABEL[m]}</button>
          ))}
        </span>
      </div>
      {trace ? (
        <>
          <Translated
            vi={
              <p className="muted" style={{ margin: '0 0 0.6rem' }}>
                Lưu trong {f.id === 'toy5' ? 'định dạng 5-bit của sách' : f.name}: a = {show(trace.a, 12)}, b = {show(trace.b, 12)}
                {trace.a !== a || trace.b !== b ? ' (giá trị bạn nhập đã được làm tròn về giá trị gần nhất mà định dạng chứa được)' : ''}.
              </p>
            }
          >
            <p className="muted" style={{ margin: '0 0 0.6rem' }}>
              Stored in {f.id === 'toy5' ? "the book's 5-bit format" : f.name}: a = {show(trace.a, 12)}, b = {show(trace.b, 12)}
              {trace.a !== a || trace.b !== b ? ' (your input was rounded to the nearest value the format holds)' : ''}.
            </p>
          </Translated>
          <div className="row" style={{ marginBottom: '0.8rem' }}>
            <button onClick={() => setN(0)} disabled={at === 0}><Translated vi="Làm lại từ đầu">Restart</Translated></button>
            <button onClick={() => setN(at - 1)} disabled={at === 0}><Translated vi="Lùi lại">Back</Translated></button>
            <button className="primary" onClick={() => setN(at + 1)} disabled={at >= total}><Translated vi="Bước tiếp">Step</Translated></button>
            <button onClick={() => setN(Math.max(at, firstAfterShifts - 1))} disabled={at >= firstAfterShifts - 1}><Translated vi="Hoàn tất các lần dịch">Finish the shifts</Translated></button>
            <button onClick={() => setN(total)} disabled={at >= total}><Translated vi="Chạy đến cuối">Run to the end</Translated></button>
            <span className="muted" aria-live="polite"><Translated vi={<>bước {at} trên {total}</>}><>step {at} of {total}</></Translated></span>
          </div>
          <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }}>
            {cur ? describe(cur, trace) : <Translated vi={<>Chưa có gì xảy ra. Hãy bấm Bước tiếp.</>}><>Nothing has happened yet. Press Step.</></Translated>}
          </div>
          {compare ? (
            // A 24-bit window is wider than a phone: the table scrolls on its own, the rest of the bench does not.
            <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
            <table className="data" style={{ background: 'var(--page)', margin: '0 0 0.6rem' }} aria-label={tr('The operands and the sum', 'Các toán hạng và tổng')}>
              <thead>
                <tr><th style={{ paddingLeft: '0.6rem' }}></th><th><Translated vi="dấu">sign</Translated></th><th><Translated vi="phần định trị (cửa sổ)">significand (the window)</Translated></th><th><Translated vi="các bit đã rời cửa sổ">bits that left the window</Translated></th><th><Translated vi="tỉ lệ">scale</Translated></th></tr>
              </thead>
              <tbody>
                <tr data-row="big">
                  <td style={{ paddingLeft: '0.6rem' }}>{show(compare.big.value, 10)}</td>
                  <td className="mono">{compare.big.sign ? '−' : '+'}</td>
                  <td className="mono">{dotted(compare.big.sig)}</td>
                  <td></td>
                  <td className="mono nowrap">× 2^{compare.big.exp}</td>
                </tr>
                <tr data-row="small">
                  <td style={{ paddingLeft: '0.6rem' }}>{show(compare.small.value, 10)}</td>
                  <td className="mono">{compare.small.sign ? '−' : '+'}</td>
                  <td className="mono" data-window={shift ? shift.window : compare.small.sig}>{dotted(shift ? shift.window : compare.small.sig)}</td>
                  <td>{shift ? <><Tail bits={shift.tail} /><br /><GRSChips g={shift.g} r={shift.r} s={shift.s} /></> : null}</td>
                  <td className="mono nowrap">× 2^{shift ? shift.exp : compare.small.exp}</td>
                </tr>
                {add ? (
                  <tr data-row="sum">
                    <td style={{ paddingLeft: '0.6rem' }}><Translated vi={<>{add.subtract ? 'hiệu' : 'tổng'} chính xác</>}><>exact {add.subtract ? 'difference' : 'sum'}</></Translated></td>
                    <td className="mono">{compare.big.sign ? '−' : '+'}</td>
                    <td className="mono">{dotted(add.window, add.carry)}</td>
                    <td><Tail bits={add.tail} /></td>
                    <td className="mono nowrap">× 2^{compare.big.exp}</td>
                  </tr>
                ) : null}
                {norm ? (
                  <tr data-row="normalised">
                    <td style={{ paddingLeft: '0.6rem' }}><Translated vi="đã chuẩn hóa">normalised</Translated></td>
                    <td className="mono">{compare.big.sign ? '−' : '+'}</td>
                    <td className="mono">{norm.subnormal ? `0.${norm.window.slice(1)}` : dotted(norm.window)}</td>
                    <td><Tail bits={norm.tail} />{norm.tail ? <><br /><GRSChips g={norm.g} r={norm.r} s={norm.s} /></> : null}</td>
                    <td className="mono nowrap">× 2^{norm.exp}</td>
                  </tr>
                ) : null}
              </tbody>
            </table>
            </div>
          ) : null}
          {rnd && add ? (
            <div data-result={rnd.result} data-err={rnd.errorUlps} data-reason={rnd.reason}>
              <RoundLine st={rnd} exact={add.exact} />
              <p style={{ margin: '0.3rem 0 0' }}>
                <span className={'chip ' + (rnd.overflow && !Number.isFinite(rnd.result) ? 'fail' : 'ok')}>
                  {MODE_LABEL[mode]}: {show(rnd.result, 10)}
                </span>{' '}
                <Translated vi={<>chính xác {add.exactText}; sai số {ulpsText(rnd.errorUlps, rnd.errorBelowOne)} ULP{rnd.carryOut ? '; làm tròn lên đã nhớ sang phần số mũ' : ''}.</>}>
                  <>exact {add.exactText}; error {ulpsText(rnd.errorUlps, rnd.errorBelowOne)} ULP{rnd.carryOut ? '; rounding up carried into the exponent' : ''}.</>
                </Translated>
              </p>
            </div>
          ) : null}
        </>
      ) : (
        <p className="t-fail"><Translated vi="Hãy nhập hai số (ví dụ 1000, -0.3, 2^-11 hoặc 0.75×2^-23).">Type two numbers (for example 1000, -0.3, 2^-11 or 0.75×2^-23).</Translated></p>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- your device (lab A.1's probe)
const LAB = 'appA/lab1_sumerror';
const yes = (v: unknown) => v === true || v === 1 || v === 'true' || v === 'yes';

function YourDevice() {
  const { reports, loading } = useReports();
  const found = Object.keys(reports).flatMap((b) => {
    const rows = [...benchRows(reports[b], LAB), ...benchRows(reports[b], LAB, 'work')];
    const probe = rows.find((r) => r.name === 'probe');
    return probe ? [{ b, device: String(reports[b].device?.name ?? b), probe: probe as Record<string, unknown> }] : [];
  });
  return (
    <Bench
      title={tr('Your device', 'Thiết bị của bạn')}
      task={tr("What lab A.1's probe found on the machines you have run it on.", 'Những gì probe của lab A.1 tìm thấy trên các máy bạn đã chạy nó.')}
      foot={
        <Translated vi={<>Probe đọc đầu vào từ một buffer, nên trình biên dịch không thể tính sẵn các phép toán lúc build.</>}>
          <>The probe reads its inputs from a buffer, so the compiler cannot fold the arithmetic away at build time.</>
        </Translated>
      }
    >
      {loading ? (
        <p className="muted"><Translated vi="Đang tải kết quả…">Loading results…</Translated></p>
      ) : !found.length ? (
        <Translated
          vi={
            <p data-empty>
              Máy này chưa có lần chạy nào. Hãy chạy <code>python3 labs/pmpp.py run appA/lab1 --bench</code> rồi tải lại trang
              này: probe cộng 1 với ba phần tư ULP bằng float, chia đôi số float chuẩn nhỏ nhất, và tính a·b + c ở chỗ việc hợp
              nhất lộ ra.
            </p>
          }
        >
          <p data-empty>
            No run on this machine yet. Run <code>python3 labs/pmpp.py run appA/lab1 --bench</code> and reload this page:
            the probe adds 1 and three quarters of an ULP in float, halves the smallest normal float, and computes a·b + c
            where fusing shows.
          </p>
        </Translated>
      ) : (
        <table className="data" style={{ background: 'var(--page)', margin: 0 }}>
          <thead>
            <tr><th style={{ paddingLeft: '0.6rem' }}><Translated vi="backend">backend</Translated></th><th><Translated vi="thiết bị">device</Translated></th><th><Translated vi="1 + 0.75 ULP bằng float">1 + 0.75 ULP in float</Translated></th><th><Translated vi="denormal FP32">FP32 denormals</Translated></th><th>a·b + c</th></tr>
          </thead>
          <tbody>
            {found.map(({ b, device, probe }) => (
              <tr key={b} data-backend={b}>
                <td style={{ paddingLeft: '0.6rem' }}>{b}</td>
                <td>{b === 'emu' || b === 'race' ? <Translated vi={<>CPU (trình giả lập): phép tính của host, không phải của GPU</>}><>the CPU (emulator): the host's arithmetic, not a GPU's</></Translated> : device}</td>
                <td>{probe.rounding === 'toward-zero' ? tr('rounds down to 1: toward zero', 'làm tròn xuống 1: về phía 0') : probe.rounding === 'nearest' ? tr('rounds up: not toward zero', 'làm tròn lên: không về phía 0') : String(probe.rounding ?? tr('not reported', 'chưa có báo cáo'))}</td>
                <td>{probe.keeps_denormals === undefined ? tr('not reported', 'chưa có báo cáo') : yes(probe.keeps_denormals) ? tr('kept', 'giữ lại') : tr('flushed to zero', 'bị ép về 0')}</td>
                <td>{probe.fuses_mul_add === undefined ? tr('not reported', 'chưa có báo cáo') : yes(probe.fuses_mul_add) ? tr('fused (one rounding)', 'gộp (một lần làm tròn)') : tr('two roundings', 'hai lần làm tròn')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- FP32 to BF16 on the bits
const hex = (v: number, w: number) => '0x' + (v >>> 0).toString(16).padStart(w, '0');
const bits32 = (u: number) => (u >>> 0).toString(2).padStart(32, '0');
const groups = (b: string, cuts: number[]) => {
  const out: string[] = [];
  let i = 0;
  for (const c of cuts) {
    out.push(b.slice(i, i + c));
    i += c;
  }
  return out.join(' ');
};

function parseF32(t: string): number | null {
  const s = t.trim();
  if (/^0x[0-9a-f]{1,8}$/i.test(s)) return Number.parseInt(s.slice(2), 16) >>> 0;
  if (/^-?nan$/i.test(s)) return 0x7fc00000;
  if (/^-?(inf|infinity)$/i.test(s)) return s.startsWith('-') ? 0xff800000 : 0x7f800000;
  if (s === '' || !/^[-+]?[\d.]+(e[-+]?\d+)?$/i.test(s)) return null;
  return bitsFromF32(Number(s));
}

function Bf16Explorer() {
  const [text, setText] = useState('3.14159274');
  const u = parseF32(text);
  const c = u === null ? null : bf16FromBits(u);
  const err = (e: number) => (Number.isNaN(e) ? 'n/a' : Number.isFinite(e) ? String(Number(e.toPrecision(3))) : '∞');
  return (
    <Bench
      title={tr('FP32 to BF16 on the bits', 'Từ FP32 sang BF16 trên từng bit')}
      task={tr(
        'Try the two ties first: same rule, different outcome. Then the carry, the largest float, and the NaN.',
        'Thử hai trường hợp hòa trước: cùng quy tắc, kết quả khác nhau. Rồi đến phép nhớ, số float lớn nhất, và NaN.',
      )}
      foot={
        <Translated vi={<>BF16 là nửa trên của một FP32: cùng dấu và số mũ, 7 trong 23 bit mantissa.</>}>
          <>BF16 is the top half of an FP32: same sign and exponent, 7 of the 23 mantissa bits.</>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          <Translated vi="một giá trị FP32 (một số, hoặc các bit dạng 0x…)">an FP32 value (a number, or bits as 0x…)</Translated>
          <input type="text" value={text} onChange={(e) => setText(e.target.value)} aria-label={tr('FP32 value', 'giá trị FP32')} style={{ width: '14rem' }} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        {BF16_PRESETS.map((p) => (
          <button key={p.id} className="quiet" data-preset={p.id} onClick={() => setText(hex(p.bits, 8))}>{p.label}</button>
        ))}
      </div>
      {c ? (
        <div data-rounded={c.rounded} data-truncated={c.truncated} data-carry={c.carryIntoExponent ? 1 : 0} data-tie={c.tie ? 1 : 0}>
          <table className="data" style={{ background: 'var(--page)', margin: '0 0 0.6rem' }}>
            <tbody>
              <tr>
                <td style={{ paddingLeft: '0.6rem' }}><Translated vi="các bit FP32">FP32 bits</Translated></td>
                <td className="mono">
                  {groups(bits32(c.input).slice(0, 16), [1, 8, 7])} <span className="muted">| {bits32(c.input).slice(16)}</span>
                </td>
                <td className="mono">{hex(c.input, 8)} = {show(c.x, 9)}</td>
              </tr>
              <tr>
                <td style={{ paddingLeft: '0.6rem' }}><Translated vi="phần giữ, phần cắt">kept, cut</Translated></td>
                <td className="mono">{hex(c.kept, 4)}, {hex(c.cut, 4)}</td>
                <td>{c.nan ? tr('a NaN: the rounding is skipped', 'một NaN: bỏ qua làm tròn') : c.tie ? tr('the cut half is exactly 0x8000: a tie', 'nửa bị cắt đúng bằng 0x8000: một trường hợp hòa') : c.cut > 0x8000 ? tr('more than half a unit cut', 'cắt hơn nửa đơn vị') : c.cut === 0 ? tr('nothing cut', 'không cắt gì') : tr('less than half a unit cut', 'cắt ít hơn nửa đơn vị')}</td>
              </tr>
              {!c.nan ? (
                <tr>
                  <td style={{ paddingLeft: '0.6rem' }}><Translated vi="cộng 0x7fff + lsb">add 0x7fff + lsb</Translated></td>
                  <td className="mono">{hex(c.input, 8)} + {hex(c.increment, 4)} = {hex(c.sum, 8)}</td>
                  <Translated vi={<td>lsb (bit cuối của nửa được giữ) = {c.lsb}{c.roundedUp ? '; tổng nhớ sang nửa được giữ' : '; không nhớ sang nửa được giữ'}{c.carryIntoExponent ? ', và tiếp tục sang phần số mũ' : ''}</td>}>
                    <td>lsb (the kept half's last bit) = {c.lsb}{c.roundedUp ? '; the sum carries into the kept half' : '; no carry into the kept half'}{c.carryIntoExponent ? ', and on into the exponent' : ''}</td>
                  </Translated>
                </tr>
              ) : null}
              <tr>
                <td style={{ paddingLeft: '0.6rem' }}><Translated vi="làm tròn (llama.cpp)">rounded (llama.cpp)</Translated></td>
                <td className="mono">{groups((c.rounded >>> 0).toString(2).padStart(16, '0'), [1, 8, 7])} = {hex(c.rounded, 4)}</td>
                <td>
                  <b>{show(c.roundedValue, 9)}</b>
                  {c.nan ? tr(' (quiet NaN: bit 6 of the half, the top mantissa bit, forced on)', ' (NaN quiet: bit 6 của nửa này, bit mantissa cao nhất, bị ép bật)') : tr(`, error ${err(c.roundedErr)} BF16 ULP`, `, sai số ${err(c.roundedErr)} ULP BF16`)}
                </td>
              </tr>
              <tr>
                <td style={{ paddingLeft: '0.6rem' }}><Translated vi="cắt bớt">truncated</Translated></td>
                <td className="mono">{groups((c.truncated >>> 0).toString(2).padStart(16, '0'), [1, 8, 7])} = {hex(c.truncated, 4)}</td>
                <td>
                  {show(c.truncatedValue, 9)}
                  {c.nan ? (Number.isFinite(c.truncatedValue) || Number.isNaN(c.truncatedValue) ? '' : <span className="t-fail">{tr(' : the NaN became infinity', ' : NaN đã thành vô cực')}</span>) : tr(`, error ${err(c.truncatedErr)} BF16 ULP`, `, sai số ${err(c.truncatedErr)} ULP BF16`)}
                </td>
              </tr>
            </tbody>
          </table>
          {!c.nan && !Number.isFinite(c.roundedValue) && Number.isFinite(c.x) ? (
            <p className="t-fail" style={{ margin: 0 }}><Translated vi="Làm tròn về gần nhất, giá trị này vượt quá giá trị lớn nhất của BF16 cộng nửa đơn vị: nó thành vô cực.">Rounded to nearest, this value is past BF16's largest value plus half a unit: it becomes infinity.</Translated></p>
          ) : null}
        </div>
      ) : (
        <p className="t-fail"><Translated vi="Hãy nhập một số, hoặc 32 bit ở dạng hex như 0x3f808000.">Type a number, or 32 bits in hex such as 0x3f808000.</Translated></p>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSA4: Record<string, Question> = {
  'appA/A.4/q-tie': {
    kind: 'choice',
    id: 'appA/A.4/q-tie',
    prompt: (
      <Translated
        vi={
          <>
            Một bộ tích lũy FP16 đang giữ 2048, và một vòng lặp liên tục cộng thêm 1 với chế độ làm tròn về số chẵn gần nhất.
            Sau phép cộng tiếp theo nó giữ giá trị nào, và vì sao nó không bao giờ thay đổi nữa?
          </>
        }
      >
        <>
          An FP16 accumulator holds 2048, and a loop keeps adding 1 under round to nearest even. What does it hold after
          the next addition, and why does it never move again?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="2048. 2049 là trường hợp hòa giữa 2048 và 2050, và làm tròn về số chẵn gần nhất lần nào cũng chọn 2048.">
            2048. 2049 is a tie between 2048 and 2050, and nearest-even picks 2048 every time.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="2050. Trường hợp hòa làm tròn lên giá trị lân cận lớn hơn, và từ đó tổng tiếp tục tăng mỗi lần 2.">
            2050. A tie rounds up to the larger neighbour, and from there the sum keeps climbing by 2.
          </Translated>
        ),
        why: (
          <Translated vi="Ở chế độ này các trường hợp hòa không làm tròn lên; chúng đi về giá trị lân cận có bit mantissa cuối bằng 0.">
            Ties do not round up under this mode; they go to the neighbour whose last mantissa bit is 0.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="2049. FP16 chứa được mọi số nguyên đến giá trị lớn nhất của nó, 65,504, nên cộng 1 là chính xác.">
            2049. FP16 holds every integer up to its largest value, 65,504, so adding 1 is exact.
          </Translated>
        ),
        why: (
          <Translated vi="Trên 2048, khoảng cách giữa các giá trị FP16 là 2: 11 bit có nghĩa phủ từ 2048 đến 4094 với bước 2.">
            Above 2048 the spacing of FP16 values is 2: 11 significant bits cover 2048 to 4094 in steps of 2.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Vô cực: 2049 cần nhiều hơn 11 bit có nghĩa mà FP16 có, nên nó bị tràn.">
            Infinity: 2049 needs more than the 11 significant bits FP16 has, so it overflows.
          </Translated>
        ),
        why: (
          <Translated vi="FP16 đạt tới 65,504. Ở đây không có gì tràn; tổng ngừng tăng từ lâu trước đó.">
            FP16 reaches 65,504. Nothing overflows here; the sum stops growing long before.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Giữa 2048 và 4096, các giá trị FP16 cách nhau 2, nên 2049 nằm đúng giữa 2048 và 2050. Làm tròn về số chẵn gần
            nhất chọn giá trị lân cận có mantissa chẵn, tức là 2048 (mantissa toàn số 0). Phép cộng tiếp theo bắt đầu từ cùng
            2048 và kết thúc như vậy. Làm tròn về phía 0 cũng cho 2048; làm tròn lên cho 2050, rồi 2052, và cứ thế. Hãy kiểm
            tra trong trình chạy từng bước với mẫu có sẵn "2048 + 1 in FP16" và cả bốn chế độ.
          </p>
        }
      >
        <p>
          Between 2048 and 4096 FP16 values are 2 apart, so 2049 lies exactly halfway between 2048 and 2050. Nearest-even
          picks the neighbour with an even mantissa, which is 2048 (mantissa all zeros). The next addition starts from the
          same 2048 and ends the same way. Toward zero also gives 2048; rounding up gives 2050, and then 2052, and so on.
          Check it in the stepper with the preset "2048 + 1 in FP16" and all four modes.
        </p>
      </Translated>
    ),
    source: 'book §A.4, pp. 591–592 (alignment and rounding); the tie rule is IEEE 754\'s, appA-facts A9',
    vi: { source: 'sách §A.4, pp. 591–592 (căn chỉnh và làm tròn); quy tắc hòa là của IEEE 754, appA-facts A9' },
  },
  'appA/A.4/q-ulp-reference': {
    kind: 'choice',
    id: 'appA/A.4/q-ulp-reference',
    prompt: (
      <Translated
        vi={
          <>
            NVIDIA liệt kê phép chia float với "sai số ULP tối đa bằng không". Sách nói làm tròn đúng nghĩa là tối đa 0.5 ULP.
            Đặc tả của Metal cho <code>x / y</code> float dưới fast math là "&lt;= 2.5 ulp". Những cái nào trong số này là
            làm tròn đúng?
          </>
        }
      >
        <>
          NVIDIA lists float division with "a maximum ULP error of zero". The book says correct rounding means at most 0.5
          ULP. Metal's specification gives float <code>x / y</code> under fast math as "&lt;= 2.5 ulp". Which of these are
          correctly rounded?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Của NVIDIA và của sách: cùng một bảo đảm, đo từ các điểm tham chiếu khác nhau. Phép chia fast math của Metal thì không.">
            {"NVIDIA's and the book's: the same guarantee, measured from different references. Metal's fast-math division is not."}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chỉ của sách: 0 ULP là không thể, nên NVIDIA hẳn muốn nói điều gì đó yếu hơn.">
            {"Only the book's: 0 ULP is impossible, so NVIDIA must mean something weaker."}
          </Translated>
        ),
        why: (
          <Translated vi="NVIDIA đếm từ kết quả đã làm tròn đúng, không phải từ kết quả chính xác. 0 ULP tính từ giá trị đã làm tròn đúng là 0.5 ULP tính từ giá trị chính xác.">
            NVIDIA counts from the correctly rounded result, not from the exact one. Zero ULP from the correctly rounded value is 0.5 ULP from the exact value.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Cả ba: mỗi cái là một cận trên của cùng một sai số, và 2.5 ulp nằm trong dung sai thông thường.">
            All three: each is a bound on the same error, and 2.5 ulp is within the usual tolerance.
          </Translated>
        ),
        why: (
          <Translated vi="Làm tròn đúng nghĩa là giá trị biểu diễn được gần nhất, cách kết quả chính xác tối đa nửa đơn vị. 2.5 ulp cho phép kết quả cách xa hai giá trị.">
            Correctly rounded means the nearest representable value, which is at most half a unit from the exact result. 2.5 ulp allows the result to be two values away.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không cái nào: phép chia trên GPU được xây từ một phép xấp xỉ, nên không cái nào trong ba cái được làm tròn đúng.">
            None: GPU division is built from an approximation, so none of the three is correctly rounded.
          </Translated>
        ),
        why: (
          <Translated vi="NVIDIA bảo đảm điều đó cho +, −, ×, / và 1/x với các cờ mặc định của họ, và Metal bảo đảm cho +, − và × ở mọi chế độ.">
            NVIDIA guarantees it for +, −, ×, / and 1/x under its default flags, and Metal for +, − and × in every mode.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trước khi so sánh con số của hai nhà cung cấp, hãy tìm xem mỗi bên đếm từ đâu. NVIDIA: "hiệu tính bằng ULP giữa
            giá trị hàm trả về và một kết quả đã làm tròn đúng". Metal: "kết quả chính xác vô hạn". Sách: kết quả chính xác.
            Phép chia float của Metal chỉ được làm tròn đúng khi tắt fast math. AMD thì ghi trong tài liệu các khác biệt đo
            được so với thư viện C++ (<code>__fadd_rn</code>: 0 trên [−1000, 1000]), và không nói gì về các toán tử thông
            thường.
          </p>
        }
      >
        <p>
          Before comparing two vendors' numbers, find what each counts from. NVIDIA: "the difference in ULPs between the
          value returned by the function and a correctly rounded result". Metal: "the infinitely precise result". The book:
          the exact result. Metal's float division is correctly rounded only with fast math off. AMD documents
          measured differences against the C++ library instead (<code>__fadd_rn</code>: 0 over [−1000, 1000]), and nothing
          about the plain operators.
        </p>
      </Translated>
    ),
    source: `book §A.4, p. 592; ${FACTS} A8 (NVIDIA 5.6.6, 5.6.7), B4 (AMD) and C2 (MSL spec 8.4)`,
    vi: { source: `sách §A.4, p. 592; ${FACTS} A8 (NVIDIA 5.6.6, 5.6.7), B4 (AMD) và C2 (đặc tả MSL 8.4)` },
  },
  'appA/A.4/q-ex3': {
    kind: 'choice',
    id: 'appA/A.4/q-ex3',
    prompt: (
      <Translated vi={<>Bài tập 3: một bộ cộng chỉ có thể làm tròn về phía 0 (cắt bớt), và giữ đủ bit để làm tròn là sai số duy nhất của nó. Sai số lớn nhất của nó là bao nhiêu?</>}>
        <>Exercise 3: an adder can only round toward zero (truncate), and keeps enough bits that rounding is its only error. What is its largest error?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Gần bằng 1 ULP.">Just under 1 ULP.</Translated>, correct: true },
      {
        label: <Translated vi="0.5 ULP, như mọi phép làm tròn đúng khác.">0.5 ULP, like any other correct rounding.</Translated>,
        why: (
          <Translated vi="Đó là làm tròn về gần nhất: giá trị chính xác cách giá trị lân cận gần hơn tối đa nửa khoảng cách. Cắt bớt luôn lấy giá trị lân cận về phía 0, dù nó xa đến đâu.">
            That is to nearest: the farthest the exact value can be from the nearer neighbour is half the gap. Truncation always takes the neighbour toward zero, however far away it is.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Đúng 1 ULP.">Exactly 1 ULP.</Translated>,
        why: (
          <Translated vi="Nếu cắt đi trọn một đơn vị, giá trị chính xác sẽ là số biểu diễn được kế tiếp, và cắt bớt sẽ trả về nó nguyên vẹn.">
            If a whole unit were cut off, the exact value would be the next representable number, and truncation would return it unchanged.
          </Translated>
        ),
      },
      {
        label: '2 ULP.', // i18n-ignore: a number and a unit
        why: (
          <Translated vi="Cắt bớt chọn giữa hai giá trị lân cận của giá trị chính xác; nó không thể rơi cách xa hai giá trị.">
            Truncation chooses between the two neighbours of the exact value; it cannot land two values away.
          </Translated>
        ),
      },
    ],
    fixedOrder: true,
    explain: (
      <Translated
        vi={
          <p>
            Cắt bớt bỏ mọi thứ dưới bit cuối được giữ. Phần bị bỏ nhỏ hơn một đơn vị của vị trí đó và có thể tiến gần một
            đơn vị đến mức các bit thêm cho phép: hãy thử 1 + 0.75 ULP, hoặc bất kỳ tổng nào có các bit guard, round và sticky
            đều bằng 1, trong trình chạy từng bước với "toward zero". Phần cứng thực làm điều này theo yêu cầu:{' '}
            <code>__fadd_rz</code> của CUDA chính là bộ cộng này (trên AMD nó không có sẵn nếu thiếu một macro).
          </p>
        }
      >
        <p>
          Truncation drops everything below the last kept bit. What is dropped is less than one unit of that place and can
          come as close to one unit as the extra bits allow: try 1 + 0.75 ULP, or any sum whose guard, round and sticky bits
          are all 1, in the stepper with "toward zero". Real hardware does this on demand: CUDA's <code>__fadd_rz</code>
          is this adder (it is not available on AMD without a macro).
        </p>
      </Translated>
    ),
    source: `book §A.9 Exercise 3, p. 599; our answer (notes/appA.md); ${FACTS} A9, B4`,
    vi: { source: `sách §A.9 Bài tập 3, p. 599; lời giải của khóa học này (notes/appA.md); ${FACTS} A9, B4` },
  },
  'appA/A.4/q-ex5': {
    kind: 'numeric',
    id: 'appA/A.4/q-ex5',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 5: một khối <code>sin()</code> tạo thêm hai bit mantissa chính xác mỗi chu kỳ xung nhịp và chạy trong 9
            chu kỳ; các bit mantissa còn lại được đặt bằng 0. Bit 1 ẩn dẫn đầu cũng phải được tạo ra. Sai số lớn nhất, tính
            bằng ULP, của kết quả độ chính xác đơn của nó là bao nhiêu?
          </>
        }
      >
        <>
          Exercise 5: a <code>sin()</code> unit produces two more accurate mantissa bits per clock cycle and runs for 9
          cycles; the remaining mantissa bits are set to 0. The hidden leading 1 must be produced too. What is the largest
          error, in ULPs, of its single-precision result?
        </>
      </Translated>
    ),
    answer: 63,
    unit: 'ULP',
    explain: (
      <Translated
        vi={
          <p>
            9 chu kỳ × 2 bit = 18 bit chính xác. Phần định trị độ chính xác đơn có 24 bit, tính cả bit ẩn, nên 6 bit thấp
            nhất bị ép bằng 0. Trong trường hợp xấu nhất cả 6 bit lẽ ra phải là 1: 2⁶ − 1 = 63 đơn vị ở vị trí cuối (nói
            chặt chẽ, gần bằng 64 nếu đo từ giá trị chính xác).
          </p>
        }
      >
        <p>
          9 cycles × 2 bits = 18 accurate bits. A single-precision significand has 24 bits, the hidden one included, so
          the lowest 6 are forced to 0. In the worst case all 6 should have been 1: 2⁶ − 1 = 63 units of the last place
          (strictly, just under 64 measured from the exact value).
        </p>
      </Translated>
    ),
    source: 'book §A.9 Exercise 5, p. 600; our answer (notes/appA.md)',
    vi: { source: 'sách §A.9 Bài tập 5, p. 600; lời giải của khóa học này (notes/appA.md)' },
  },
  'appA/A.4/q-fma': {
    kind: 'choice',
    id: 'appA/A.4/q-fma',
    prompt: (
      <Translated
        vi={
          <>
            Kernel CUDA của bạn tính <code>a*b + c</code>. Bản tham chiếu CPU của bạn, build với mặc định của Clang, tính{' '}
            <code>float p = a*b; float r = p + c;</code> trên cùng các số float. Hai kết quả khác nhau ở bit cuối, và không có
            gì khác trong kernel sai. Nguyên nhân có khả năng nhất là gì?
          </>
        }
      >
        <>
          Your CUDA kernel computes <code>a*b + c</code>. Your CPU reference, built with Clang's defaults, computes{' '}
          <code>float p = a*b; float r = p + c;</code> on the same floats. The two differ in the last bit, and nothing else
          in the kernel is wrong. What is the most likely cause?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="nvcc đã gộp phép nhân và phép cộng thành một FMA, làm tròn một lần; bản tham chiếu làm tròn tích trước.">
            nvcc fused the multiply and add into one FMA, rounding once; the reference rounds the product first.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="GPU mặc định làm tròn mọi phép toán float về phía 0, còn CPU làm tròn về số chẵn gần nhất.">
            The GPU rounds every float operation toward zero by default, and the CPU rounds to nearest even.
          </Translated>
        ),
        why: (
          <Translated vi="CUDA mặc định làm tròn về số chẵn gần nhất; các chế độ khác cần các intrinsic _rz, _ru, _rd.">
            CUDA rounds to nearest even by default; other modes need the _rz, _ru, _rd intrinsics.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="GPU đã ép một tích denormal về 0, trong khi bản tham chiếu CPU giữ lại nó.">
            The GPU flushed a denormal product to zero, which the CPU reference keeps.
          </Translated>
        ),
        why: (
          <Translated vi="Chỉ có thể xảy ra với -ftz=true hoặc --use_fast_math, vốn mặc định tắt, và chỉ khi một giá trị rất nhỏ.">
            Possible only with -ftz=true or --use_fast_math, which are off by default, and only if a value is tiny.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một race condition (tranh chấp dữ liệu) giữa các luồng ghi các đầu ra cạnh nhau.">
            A race condition between threads that write neighbouring outputs.
          </Translated>
        ),
        why: (
          <Translated vi="Một race cho kết quả sai hoặc thay đổi, không phải một khác biệt một bit nhất quán trong một biểu thức không có dữ liệu dùng chung.">
            A race gives wrong or varying results, not a consistent one-bit difference in an expression with no shared data.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              <code>--fmad</code> của nvcc mặc định là true, nên kernel làm tròn một lần. Bản tham chiếu của bạn lưu tích vào{' '}
              <code>p</code> trước, và Clang trên host chỉ gộp trong một câu lệnh, nên nó làm tròn hai lần. (HIP-Clang
              mặc định gộp xuyên các câu lệnh, nên cách tách như vậy sẽ không ngăn được nó ở đó.) Nếu bản tham chiếu là
              một câu lệnh duy nhất <code>a*b + c</code>, một host có lệnh FMA cũng có thể gộp nó, và hai bên có thể khớp
              nhau. Không kết quả nào sai: kết quả gộp thực ra chính xác hơn. Hãy so sánh với một dung sai, hoặc, khi bạn
              cần khớp từng bit, hãy viết <code>__fadd_rn(__fmul_rn(a, b), c)</code>: hai intrinsic đó không bao giờ bị gộp
              thành một FMA.
            </p>
            <p>Probe của lab A.1 báo cáo trình biên dịch của chính bạn đã làm gì.</p>
          </>
        }
      >
        <>
          <p>
            nvcc's <code>--fmad</code> is true by default, so the kernel rounds once. Your reference stores the product in{' '}
            <code>p</code> first, and Clang on the host fuses only within one statement, so it rounds twice. (HIP-Clang
            fuses across statements by default, so the same split would not stop it there.) Had the reference been the one
            statement <code>a*b + c</code>, a host with FMA instructions could fuse it too, and the two might agree. Neither
            result is wrong: the fused one is in fact the more accurate. Compare with a tolerance, or, where you need
            bit-for-bit agreement, write <code>__fadd_rn(__fmul_rn(a, b), c)</code>: those two intrinsics are never merged
            into an FMA.
          </p>
          <p>The probe of lab A.1 reports what your own compiler did.</p>
        </>
      </Translated>
    ),
    source: `${FACTS} A9 (NVIDIA 5.6.1.5, 5.6.9.1, nvcc --fmad) and B5 (Clang -ffp-contract)`,
    vi: { source: `${FACTS} A9 (NVIDIA 5.6.1.5, 5.6.9.1, nvcc --fmad) và B5 (Clang -ffp-contract)` },
  },
};

// ---------------------------------------------------------------- the page
export function SA4() {
  const bf16 = useLabFile('appA/snippets/bf16_round.h');
  return (
    <>
      <Translated
        vi={
          <p>
            Mục A.2 kết thúc với độ chính xác của định dạng (precision): giá trị gần nhất của định dạng có thể gần đến đâu
            với một số bạn muốn lưu. Đó là một thuộc tính của định dạng, do số bit mantissa quyết định. <b>Độ chính xác của
            phép toán (accuracy)</b> là một thuộc tính của phép toán: sai số lớn nhất mà một phép cộng, một phép chia hay một
            phép exp đưa vào kết quả của nó. Một định dạng có thể có precision cao mà một phép toán trên nó lại kém chính xác.
            Mục này nói về nguồn gốc của sai số đó và nó được phép lớn đến đâu.
          </p>
        }
      >
        <p>
          Section A.2 ended with precision: how close the format's nearest value can be to a number you want to store. It
          is a property of the format, set by its mantissa bits. <b>Accuracy</b> is a property of an operation: the largest
          error an addition, a division or an exp introduces into its result. A format can be precise and an operation on
          it inaccurate. This section is about where that error comes from and how big it is allowed to be.
        </p>
      </Translated>
      <Source section="A.4" pages="591" />

      <h2>
        <Translated vi="Vì sao một kết quả cần làm tròn">Why a result needs rounding</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Nguồn sai số chính rất đơn giản: kết quả chính xác của một phép toán thường có nhiều bit có nghĩa hơn số bit định
            dạng chứa được. Tích của hai phần định trị 24-bit có tới 48 bit. Một phép cộng trước hết phải đưa hai toán hạng về
            cùng số mũ. Bộ cộng dịch phần định trị của toán hạng nhỏ hơn sang phải mỗi lần một vị trí và tăng số mũ của nó
            thêm một mỗi lần, cho đến khi các số mũ khớp nhau. Các bit bị đẩy ra khỏi cuối cửa sổ không biến mất khỏi tổng
            chính xác, nên tổng có thể cần nhiều bit hơn cửa sổ có.
          </p>
        }
      >
        <p>
          The main source of error is simple: the exact result of an operation often has more significant bits than the
          format can hold. A product of two 24-bit significands has up to 48. An addition first has to bring the two
          operands to the same exponent. The adder shifts the smaller operand's significand right one place at a time and
          raises its exponent by one each time, until the exponents match. Bits pushed off the end of the window do not
          vanish from the exact sum, so the sum can need more bits than the window has.
        </p>
      </Translated>
      <Source section="A.4" pages="591–592" />
      <Translated
        vi={
          <p>
            Ví dụ của sách dùng định dạng 5-bit từ mục A.2: cộng 0.25 vào 2.0. Trình chạy từng bước dưới đây bắt đầu từ đó.
            Nó tính tổng chính xác và trình bày theo cách một bộ cộng xử lý, giữ thêm ba bit ngoài cửa sổ: guard, round và
            sticky. Trước khi chạy từng bước, hãy chốt kết quả.
          </p>
        }
      >
        <p>
          The book's example uses its 5-bit format from section A.2: add 0.25 to 2.0. The stepper below starts there. It
          computes the exact sum and shows it the way an adder works on it, keeping three bits beyond the window: guard,
          round and sticky. Before you step through it, commit to the result.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Không mô phỏng pipeline hay độ rộng bên trong của bộ cộng thực nào. Một bộ cộng chỉ cần các bit guard, round và
              sticky để làm tròn đúng, đó là lý do ba bit này được hiển thị. Không có fused multiply-add, không ép về 0.
            </p>
          }
        >
          <p>
            No real adder's pipeline or internal width. An adder needs only the guard, round and sticky bits to round
            correctly, which is why those three are shown. No fused multiply-add, no flush-to-zero.
          </p>
        </Translated>
      </Simplification>
      <Predict
        id="appA/A.4/p-book-sum"
        prompt={
          <Translated vi={<>2.0 + 0.25 = 2.25 chính xác, nhưng định dạng 5-bit không có 2.25: các giá trị gần nó là 2.0 và 2.5. Bộ cộng sẽ trả về giá trị nào?</>}>
            <>2.0 + 0.25 = 2.25 exactly, but the 5-bit format has no 2.25: its values near it are 2.0 and 2.5. Which will the adder return?</>
          </Translated>
        }
      >
        {(v) => (
          <Translated
            vi={
              <p>
                {v === 2 ? 'Đúng, nhưng vì một lý do sách không nêu. ' : v === 2.5 ? 'Một phỏng đoán hợp lý: cả hai giá trị lân cận đều gần như nhau. ' : 'Nó phải là một trong hai giá trị lân cận. '}
                2.25 nằm đúng chính giữa: đáp án nào cũng lệch 0.5 ULP, và sách dừng ở đó. IEEE phá thế hòa bằng cách chọn giá
                trị lân cận có bit mantissa cuối bằng 0, tức là 2.0 (mantissa 00; 2.5 là 01). Hãy chạy từng bước bên dưới.
              </p>
            }
          >
            <p>
              {v === 2 ? 'Right, but for a reason the book does not give. ' : v === 2.5 ? 'A reasonable guess: both neighbours are equally close. ' : 'It has to be one of the two neighbours. '}
              2.25 is exactly halfway: either answer is 0.5 ULP off, and the book stops there. IEEE breaks the tie toward the
              neighbour whose last mantissa bit is 0, which is 2.0 (mantissa 00; 2.5 is 01). Step through it below.
            </p>
          </Translated>
        )}
      </Predict>
      <Stepper />
      <Translated
        vi={
          <p>
            Trong ví dụ của sách, 0.25 là một denormal trong định dạng 5-bit: được lưu là 0.01 × 2⁰. Trình chạy từng bước viết
            nó kèm bit 1 dẫn đầu, thành 1.00 × 2⁻², đó là cách sách đếm ba lần dịch. Sau ba lần đó, bit 1 của 0.25 nằm ở vị
            trí guard: đúng nửa đơn vị của vị trí cuối, một trường hợp hòa.
          </p>
        }
      >
        <p>
          In the book's example, 0.25 is a denormal in the 5-bit format: stored as 0.01 × 2⁰. The stepper writes it with its
          leading 1, as 1.00 × 2⁻², which is how the book counts its three shifts. After them, 0.25's 1 sits in the guard
          position: exactly half a unit of the last place, a tie.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Quy tắc phá hòa là do khóa học này bổ sung. Sách chỉ nói 2.0 và 2.5 đều cách 0.5 ULP. Làm tròn về gần nhất, hòa
              thì về số chẵn, là mặc định của IEEE và của CUDA (appA-facts A9).
            </p>
          }
        >
          <p>
            The tie rule is ours to add. The book says only that 2.0 and 2.5 are both 0.5 ULP away. Round to nearest, ties
            to even, is IEEE's default and CUDA's (appA-facts A9).
          </p>
        </Translated>
      </Aside>
      <Ask q={questionsSA4['appA/A.4/q-tie']} />

      <h2>
        <Translated vi="ULP, và lý tưởng 0.5 ULP">ULP, and the 0.5 ULP ideal</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Đơn vị sai số của sách là <b>ULP</b>, một đơn vị ở vị trí cuối (unit in the last place): giá trị vị trí của bit
            mantissa cuối, cũng là khoảng cách giữa hai số biểu diễn được liền kề. Trong ví dụ, sai số là 0.001 × 2¹, bằng một
            nửa giá trị 0.01 × 2¹ của vị trí cuối: 0.5 ULP. Một kết quả chính xác không biểu diễn được nằm giữa hai giá trị
            lân cận, nên điều tốt nhất mọi phần cứng làm được là chọn giá trị gần hơn, cách tối đa nửa ULP. Sách nói các thiết
            bị CUDA đạt được điều đó.
          </p>
        }
      >
        <p>
          The book's unit for error is the <b>ULP</b>, a unit in the last place: the place value of the last mantissa bit,
          which is also the gap between two neighbouring representable numbers. In the example the error is 0.001 × 2¹,
          half of the last place's 0.01 × 2¹: 0.5 ULP. An exact result that is not representable lies between two
          neighbours, so the best any hardware can do is the nearer one, at most half a ULP away. The book says that is
          what CUDA devices achieve.
        </p>
      </Translated>
      <Source section="A.4" pages="592" />
      <Ask q={questionsSA4['appA/A.4/q-ulp-reference']} />

      <h2>
        <Translated vi="Bốn chế độ làm tròn">Four rounding modes</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách cho thấy một cách làm tròn. IEEE 754 định nghĩa bốn: về gần nhất với hòa thì về số chẵn, về phía 0, về phía
            +∞ (lên) và về phía −∞ (xuống). Đặt trình chạy từng bước về mẫu FP32 "1 + 0.75 ULP" và chạy đến cuối, rồi đổi chế
            độ: về gần nhất và lên cho 1 + 2⁻²³, về phía 0 và xuống cho 1. Một phép cộng đó phân biệt được hai trong các chế
            độ, và đó là việc probe của lab A.1 dùng nó.
          </p>
        }
      >
        <p>
          The book shows one rounding. IEEE 754 defines four: to nearest with ties to even, toward zero, toward +∞ (up)
          and toward −∞ (down). Set the stepper to the FP32 preset "1 + 0.75 ULP" and run it to the end, then switch modes:
          nearest and up give 1 + 2⁻²³, toward zero and down give 1. That one addition tells two of the modes apart, which
          is what lab A.1's probe uses it for.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Các chế độ làm tròn ngoài ví dụ duy nhất của sách, từ appA-facts A9, B4 và C2.</p>}>
          <p>Rounding modes beyond the book's single example, from appA-facts A9, B4 and C2.</p>
        </Translated>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th><Translated vi="Backend">Backend</Translated></th><th><Translated vi="Mặc định">Default</Translated></th><th><Translated vi="Các chế độ khác">Other modes</Translated></th></tr>
        </thead>
        <tbody>
          <tr>
            <td>CUDA</td>
            <td><Translated vi="về gần nhất, hòa thì về số chẵn">to nearest, ties to even</Translated></td>
            <Translated vi={<td>chọn theo từng phép toán bằng intrinsic: <code>__fadd_rz</code>, <code>__fmul_ru</code>, <code>__fdiv_rd</code> và các hàm tương tự (<code>_rn _rz _ru _rd</code>), mỗi hàm "sai số ULP tối đa bằng 0"; không có chế độ nào bạn đặt được cho cả một kernel</td>}>
              <td>chosen per operation by intrinsic: <code>__fadd_rz</code>, <code>__fmul_ru</code>, <code>__fdiv_rd</code> and so on (<code>_rn _rz _ru _rd</code>), each "maximum ULP error of 0"; no mode you can set for a whole kernel</td>
            </Translated>
          </tr>
          <tr>
            <td>HIP</td>
            <td><Translated vi="về gần nhất, hòa thì về số chẵn">to nearest, ties to even</Translated></td>
            <Translated vi={<td>"Mặc định trên GPU AMD chỉ hỗ trợ chế độ làm tròn về số chẵn gần nhất"; các intrinsic <code>_rz _ru _rd</code> chỉ tồn tại nếu <code>OCML_BASIC_ROUNDED_OPERATIONS</code> được định nghĩa</td>}>
              <td>"Only the nearest-even rounding mode is supported by default on AMD GPUs"; the <code>_rz _ru _rd</code> intrinsics exist only if <code>OCML_BASIC_ROUNDED_OPERATIONS</code> is defined</td>
            </Translated>
          </tr>
          <tr>
            <td>Metal</td>
            <Translated vi={<td>"Có thể hỗ trợ chế độ làm tròn hòa về số chẵn hoặc làm tròn về phía 0"</td>}>
              <td>"Either round ties to even or round toward zero rounding mode may be supported"</td>
            </Translated>
            <Translated vi={<td>không có chế độ nào bạn chọn được; Apple không nói một GPU cụ thể dùng chế độ nào</td>}>
              <td>none you can choose; Apple does not say which mode a given GPU uses</td>
            </Translated>
          </tr>
        </tbody>
      </table>
      <YourDevice />

      <h2>
        <Translated vi="Một bộ cộng cắt bớt">A truncating adder</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bài tập 3 hỏi về một bộ cộng chỉ có thể làm tròn về phía 0. Đặt trình chạy từng bước về "toward zero" và thử các
            mẫu có sẵn: mọi thứ nằm dưới cửa sổ đều bị bỏ đi, dù nó gần một đơn vị trọn vẹn đến đâu.
          </p>
        }
      >
        <p>
          Exercise 3 asks about an adder that can only round toward zero. Set the stepper to "toward zero" and try the
          presets: whatever lies below the window is dropped, however close it is to a whole unit.
        </p>
      </Translated>
      <Ask q={questionsSA4['appA/A.4/q-ex3']} />

      <h2>
        <Translated vi="Các khối tính xấp xỉ">Units that approximate</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Phép chia và các hàm siêu việt thường được xây từ các phép xấp xỉ đa thức. Với quá ít số hạng, kết quả có thể lệch
            hơn nửa ULP. Ví dụ của sách: một phép nghịch đảo có kết quả chính xác là 1.00 × 2¹ lại trả về 1.10 × 2¹, lệch
            0.10, gấp đôi giá trị 0.01 của vị trí cuối: 2 ULP. Phép nghịch đảo phần cứng của một số thiết bị đời đầu kém chính
            xác đến vậy; sách nói các thiết bị CUDA gần đây tuân thủ IEEE.
          </p>
        }
      >
        <p>
          Division and transcendental functions are usually built from polynomial approximations. With too few terms the
          result can be off by more than half a ULP. The book's example: an inversion whose exact result is 1.00 × 2¹
          returns 1.10 × 2¹, which is off by 0.10, twice the last place's 0.01: 2 ULP. Some early devices' hardware
          inversion was this inaccurate; the book says recent CUDA devices are IEEE-conformant.
        </p>
      </Translated>
      <Source section="A.4" pages="592" />
      <Translated
        vi={
          <p>
            Phép chia kém chính xác ngày nay vẫn còn, nếu bạn chọn nó. Trong CUDA, <code>-prec-div=false</code>, được{' '}
            <code>--use_fast_math</code> ngầm bật, biến <code>x / y</code> thành <code>__fdividef</code>, trong phạm vi 2
            ULP, và <code>1/x</code> thành một phép nghịch đảo 1-ULP; mặc định là phép chia làm tròn đúng.
          </p>
        }
      >
        <p>
          The inaccurate division is still available today, by choice. In CUDA, <code>-prec-div=false</code>, which{' '}
          <code>--use_fast_math</code> implies, turns <code>x / y</code> into <code>__fdividef</code>, within 2 ULP, and{' '}
          <code>1/x</code> into a 1-ULP reciprocal; the default is the correctly rounded division.
        </p>
      </Translated>
      <Today checked={CHECKED('A8, A14')}>
        <Translated
          vi={
            <p>
              "Các khối hàm đặc biệt" (special function units) của sách (p. 599) không xuất hiện trong văn bản hiện tại của
              NVIDIA, nơi nói rằng các intrinsic "ánh xạ sang ít lệnh gốc hơn".
            </p>
          }
        >
          <p>
            The book's "special function units" (p. 599) do not appear in NVIDIA's current text, which says intrinsics "map
            to fewer native instructions".
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Hàm mũ cũng có cùng sự đánh đổi. <code>expf</code> của CUDA nằm trong 2 ULP còn <code>__expf</code> trong 2 +
            ⌊|1.173 x|⌋; <code>exp</code> của Metal nằm trong 4 ulp còn <code>fast::exp</code> trong 3 + ⌊|2x|⌋. Cả hai hàm
            mũ nhanh đều tệ hơn khi |x| tăng, thêm một lý do để softmax giữ đối số của exp không quá 0 (mục{' '}
            <a href={href('appA', 'A.3')}>A.3</a>).
          </p>
        }
      >
        <p>
          Exponentials make the same trade. CUDA's <code>expf</code> is within 2 ULP and <code>__expf</code> within 2 +
          ⌊|1.173 x|⌋; Metal's <code>exp</code> is within 4 ulp and <code>fast::exp</code> within 3 + ⌊|2x|⌋. Both fast
          exponentials get worse as |x| grows, which is one more reason softmax keeps the argument of exp at or below 0
          (section <a href={href('appA', 'A.3')}>A.3</a>).
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch10-facts.md, A11, C4)">{/* i18n-ignore: a date and a citation */}
        <Translated
          vi={
            <p>
              Các cận lấy từ hướng dẫn của NVIDIA (mục 5.6.7.2 và 5.6.9.2) và đặc tả Metal Shading Language (Tables 8.1 và
              8.2).
            </p>
          }
        >
          <p>
            Bounds from NVIDIA's guide (sections 5.6.7.2 and 5.6.9.2) and the Metal Shading Language specification
            (Tables 8.1 and 8.2).
          </p>
        </Translated>
      </Today>
      <Ask q={questionsSA4['appA/A.4/q-ex5']} label={tr('Work it out', 'Tự tính')} />

      <h2>
        <Translated vi="Một lần làm tròn thay vì hai">One rounding instead of two</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một phép <b>fused multiply-add</b> (nhân-cộng gộp) tính <code>a*b + c</code> với một lần làm tròn duy nhất ở
            cuối, trong khi một phép nhân theo sau là một phép cộng làm tròn hai lần. Sách không nhắc đến nó, nhưng mọi trình
            biên dịch GPU đều tạo ra nó theo mặc định, và đó là lý do phổ biến nhất khiến một kernel đúng và một bản tham chiếu
            CPU đúng khác nhau ở bit cuối.
          </p>
        }
      >
        <p>
          A <b>fused multiply-add</b> computes <code>a*b + c</code> with a single rounding at the end, where a multiply
          followed by an add rounds twice. The book does not mention it, but every GPU compiler produces it by default, and
          it is the most common reason a correct kernel and a correct CPU reference disagree in the last bit.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Mặc định: nvcc <code>--fmad=true</code>; HIP-Clang <code>-ffp-contract=fast-honor-pragmas</code> (gộp xuyên
              các câu lệnh); Metal cho phép gộp (contraction); Clang trên host <code>-ffp-contract=on</code> (trong một câu
              lệnh). <code>__fadd_rn</code> và <code>__fmul_rn</code> không bao giờ bị gộp (appA-facts A9, B5).
            </p>
          }
        >
          <p>
            Defaults: nvcc <code>--fmad=true</code>; HIP-Clang <code>-ffp-contract=fast-honor-pragmas</code> (fuses across
            statements); Metal allows contraction; host Clang <code>-ffp-contract=on</code> (within one statement).{' '}
            <code>__fadd_rn</code> and <code>__fmul_rn</code> are never fused (appA-facts A9, B5).
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsSA4['appA/A.4/q-fma']} />

      <h2>
        <Translated vi="Từ FP32 sang BF16 trên từng bit">FP32 to BF16 on the bits</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Làm tròn cũng là thứ bạn sẽ tự viết. Chuyển trọng số hoặc activation FP32 sang BF16 là trường hợp thường ngày:
            BF16 là 16 bit trên của một FP32, nên cắt bớt chỉ là một phép dịch. llama.cpp thì làm tròn, trên các bit số
            nguyên: nó cộng <code>0x7fff</code> với bit cuối của nửa được giữ rồi dịch. Nếu nửa bị cắt nhỏ hơn 0x8000 thì
            không có nhớ; lớn hơn thì nhớ sang nửa được giữ; đúng 0x8000, một trường hợp hòa, chỉ nhớ khi bit cuối của nửa
            được giữ là 1, nên bit cuối của kết quả thành chẵn. Đó là làm tròn về số chẵn gần nhất bằng hai phép cộng số
            nguyên. NaN được xử lý trước: bit mantissa cao nhất của nó bị ép bật, nên một NaN có payload chỉ nằm trong nửa
            bị cắt không thể biến thành vô cực.
          </p>
        }
      >
        <p>
          Rounding is something you will also write yourself. Converting FP32 weights or activations to BF16 is the
          everyday case: BF16 is the top 16 bits of an FP32, so truncation is one shift. llama.cpp rounds instead, on the
          integer bits: it adds <code>0x7fff</code> plus the kept half's last bit and then shifts. If the cut half is below
          0x8000 nothing carries; above it, it carries into the kept half; exactly 0x8000, a tie, carries only when the
          kept half's last bit is 1, so the result's last bit ends even. That is round to nearest even in two integer
          additions. A NaN is handled first: its top mantissa bit is forced on, so a NaN whose payload lived only in the
          cut half cannot turn into infinity.
        </p>
      </Translated>
      <Today checked={CHECKED('D3, C2, C3')}>
        <Translated
          vi={
            <p>
              llama.cpp (<code>ggml-impl.h</code>): "Số float sẽ được làm tròn về số chẵn gần nhất, và NaN sẽ là quiet. Số
              subnormal không bị ép về 0". Phép chuyển float sang bfloat của chính Metal cũng làm tròn hòa về số chẵn; Metal
              4.1 thêm <code>-fmetal-rtz-fp-conversion</code> để các phép chuyển float sang float làm tròn về phía 0 thay vào
              đó.
            </p>
          }
        >
          <p>
            llama.cpp (<code>ggml-impl.h</code>): "Floats shall round to nearest even, and NANs shall be quiet. Subnormals
            aren't flushed to zero". Metal's own float-to-bfloat conversion also rounds ties to even; Metal 4.1 adds{' '}
            <code>-fmetal-rtz-fp-conversion</code> to make float-to-float conversions round toward zero instead.
          </p>
        </Translated>
      </Today>
      <Code
        tabs={[{ label: 'C++', code: bf16 ?? '// loading labs/appA/snippets/bf16_round.h' }]}
        highlight={[16]}
        caption={
          <Translated
            vi={
              <>
                labs/appA/snippets/bf16_round.h. Dòng 16 là phép làm tròn của llama.cpp; phép kiểm tra NaN phía trên nó và{' '}
                <code>memcpy</code> là của khóa học này. Các unit test biên dịch và chạy nó trên CPU và so từng kết quả, từng
                bit, với trình khám phá bên dưới.
              </>
            }
          >
            <>
              labs/appA/snippets/bf16_round.h. Line 16 is llama.cpp's rounding; the NaN test above it and the{' '}
              <code>memcpy</code> are ours. The unit tests compile and run it on the CPU and compare every result, bit for
              bit, with the explorer below.
            </>
          </Translated>
        }
      />
      <Bf16Explorer />
      <Translated
        vi={
          <p>
            Cắt bớt không chỉ kém chính xác hơn, nó còn bị lệch: mọi giá trị đều dịch về phía 0, trung bình nửa đơn vị BF16,
            nên sai số của một tổng gồm nhiều giá trị bị cắt bớt tăng theo một hướng. Làm tròn về gần nhất sai về cả hai phía
            và không có độ trôi như vậy. Đó là thứ phép cộng số nguyên thêm vào mua được.
          </p>
        }
      >
        <p>
          Truncation is not only less accurate, it is biased: every value moves toward zero, by half a BF16 unit on
          average, so the error of a sum of many truncated values grows in one direction. Rounding to nearest errs both ways
          and has no such drift. That is what the extra integer addition buys.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Một cách làm tròn thứ năm: các API chuyển đổi của AMD cho gfx950 có <b>làm tròn ngẫu nhiên (stochastic
              rounding)</b>; một API như vậy "sẽ yêu cầu một đầu vào làm seed" (appA-facts B2). Theo nghĩa thông thường, do
              khóa học này nêu, nó làm tròn lên hoặc xuống một cách ngẫu nhiên, thường xuyên hơn về phía giá trị lân cận gần
              hơn, nên các sai số triệt tiêu nhau về trung bình. Sách không đề cập đến nó.
            </p>
          }
        >
          <p>
            A fifth way to round: AMD's conversion APIs for gfx950 offer <b>stochastic rounding</b>; such an API "will
            expect an input as seed" (appA-facts B2). In the usual meaning, which is ours to state, it rounds up or down at
            random, more often toward the nearer neighbour, so the errors cancel on average. The book does not cover it.
          </p>
        </Translated>
      </Aside>

      <h2>
        <Translated vi="Số hạng biến mất">The disappearing addend</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một phép cộng mất tối đa nửa ULP. Một vòng lặp gồm các phép cộng như vậy có thể mất toàn bộ những gì nó cộng vào.
            Hãy xem một bộ tích lũy FP16 nhận 0.3 lần này đến lần khác, bắt đầu từ 1000.
          </p>
        }
      >
        <p>
          One addition loses at most half a ULP. A loop of them can lose everything it adds. Watch an FP16 accumulator
          receive 0.3 again and again, starting from 1000.
        </p>
      </Translated>
      <DisappearingAddendScene />
      <Translated
        vi={
          <p>
            Thứ tự các phép cộng và độ rộng của bộ tích lũy là chủ đề của mục <a href={href('appA', 'A.6')}>A.6</a>, và lab
            A.1 đo cả hai trên GPU của bạn (<a href={href('appA', 'labs')}>các lab</a>).
          </p>
        }
      >
        <p>
          The order of additions and the width of the accumulator are the subject of section{' '}
          <a href={href('appA', 'A.6')}>A.6</a>, and lab A.1 measures both on your GPU (<a href={href('appA', 'labs')}>labs</a>).
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mô hình của cảnh này: 0.3 trước hết được lưu thành giá trị FP16 gần nhất, và mỗi phép cộng làm tròn một lần về
              số chẵn gần nhất, không có fused multiply-add và không có trình biên dịch nào xen vào. Đường FP32 dùng cùng mô
              hình đó.
            </p>
          }
        >
          <p>
            The scene's model: 0.3 is first stored as the nearest FP16 value, and every addition rounds once to nearest
            even, with no fused multiply-add and no compiler in between. The FP32 line uses the same model.
          </p>
        </Translated>
      </Simplification>
      <Today checked={CHECKED('D1, D2')}>
        <Translated
          vi={
            <p>
              Đây là lý do các framework chọn bộ tích lũy. PyTorch tích lũy half và bfloat16 bằng float trên mọi thiết bị.{' '}
              <code>softmax</code> mặc định của MLX trên half giữ tổng của nó ở half; <code>precise=True</code> thì tích lũy
              bằng float32, vì nếu không nó "có thể mất độ chính xác trên các trục rút gọn dài".
            </p>
          }
        >
          <p>
            This is why frameworks choose the accumulator. PyTorch accumulates half and bfloat16 in float on every device.
            MLX's default <code>softmax</code> on half keeps its sum in half; <code>precise=True</code> accumulates in
            float32 instead, because otherwise it "can lose precision over long reduction axes".
          </p>
        </Translated>
      </Today>
    </>
  );
}
