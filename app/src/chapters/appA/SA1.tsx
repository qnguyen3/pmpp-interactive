import { useState, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  BF16, E2M1, E4M3, E5M2, FP16, FP32, TOY6,
  anatomy, bitString, bitsOfValue, classify, encode, excessBias, excessCode, excessTable, fields, flipBit,
  maxFinite, minNormal, nextDown, nextUp, precisionBits, show, showBits, totalBits, twosComplement, ulp,
  type Anatomy, type Format,
} from '../../sim/minifloat';

// ---------------------------------------------------------------- small pieces
const minus = (n: number | string) => String(n).replace(/-/g, '−');
const Pow = ({ k }: { k: number | string }) => (
  <>
    2<sup>{minus(k)}</sup>
  </>
);
const hex = (f: Format, bits: number) => '0x' + (bits >>> 0).toString(16).toUpperCase().padStart(Math.ceil(totalBits(f) / 4), '0');
/** Parse what a reader types: decimals, exponents, inf and nan. Returns null for nothing usable. */
function parseValue(t: string): number | null {
  const s = t.trim().toLowerCase().replace(/[−–]/g, '-').replace(/[,_\s]/g, '');
  if (s === '') return null;
  if (s === 'inf' || s === '+inf' || s === 'infinity' || s === '+infinity') return Infinity;
  if (s === '-inf' || s === '-infinity') return -Infinity;
  if (s === 'nan') return NaN;
  const v = Number(s);
  return Number.isNaN(v) ? null : v;
}

// ---------------------------------------------------------------- explorer: excess codes (Fig. A.1)
type Coding = 'excess' | 'twos';
function ExcessBench() {
  const [e, setE] = useState(3);
  const bias = excessBias(e);
  const [a, setA] = useState(-2);
  const [b, setB] = useState(1);
  const [pred, setPred] = useState<Partial<Record<Coding, 'A' | 'B'>>>({});
  const clamp = (v: number) => Math.max(-bias, Math.min(bias, v));
  const A = clamp(a);
  const B = clamp(b);
  const rows = excessTable(e);
  const code = (k: Coding, v: number) => (k === 'excess' ? excessCode(e, v) : twosComplement(e, v));
  const exps: number[] = [];
  for (let v = -bias; v <= bias; v++) exps.push(v);
  const byTwos = rows.slice().sort((x, y) => x.twos - y.twos);
  const setWidth = (w: number) => {
    setE(w);
    setPred({});
  };

  const verdict = (k: Coding) => {
    const ca = code(k, A);
    const cb = code(k, B);
    const larger = ca > cb ? 'A' : 'B';
    const sameOrder = (ca > cb) === (A > B);
    const p = pred[k];
    return (
      <div data-coding={k} data-same-order={String(sameOrder)} style={{ marginBottom: '0.7rem' }}>
        <p style={{ margin: '0 0 0.35rem' }}>
          <b>{k === 'excess' ? tr(`Excess-${bias} codes`, `Mã excess-${bias}`) : tr("Two's-complement codes", 'Mã bù hai')}.</b>{' '}
          <Translated vi="Đọc như số nguyên không dấu, mã nào lớn hơn?">Read as unsigned integers, which code is larger?</Translated>
        </p>
        <div className="row" style={{ marginBottom: '0.35rem' }}>
          {(['A', 'B'] as const).map((w) => (
            <button key={w} aria-pressed={p === w} disabled={A === B} onClick={() => setPred({ ...pred, [k]: w })}>
              <Translated vi={<>mã của {w}</>}>{w}'s code</Translated>
            </button>
          ))}
        </div>
        {p && A !== B ? (
          <p style={{ margin: 0 }} data-part="reveal">
            <Translated
              vi={
                <>
                  A = {minus(A)} được lưu thành <span className="mono">{bitString(ca, e)}</span> ({ca}); B = {minus(B)} thành{' '}
                  <span className="mono">{bitString(cb, e)}</span> ({cb}). Mã của {larger} lớn hơn.{' '}
                </>
              }
            >
              A = {minus(A)} is stored as <span className="mono">{bitString(ca, e)}</span> ({ca}); B = {minus(B)} as{' '}
              <span className="mono">{bitString(cb, e)}</span> ({cb}). {larger}'s code is larger.{' '}
            </Translated>
            {p === larger ? <span className="chip ok"><Translated vi={<>bạn chọn {p}: đúng</>}>you said {p}: right</Translated></span> : <span className="chip fail"><Translated vi={<>bạn chọn {p}: sai</>}>you said {p}: wrong</Translated></span>}{' '}
            {sameOrder ? tr('The codes are in the same order as the exponents.', 'Các mã có cùng thứ tự với các số mũ.') : tr('The codes are in the opposite order to the exponents.', 'Các mã có thứ tự ngược với các số mũ.')}
          </p>
        ) : null}
      </div>
    );
  };

  return (
    <Bench
      title={tr('Excess codes', 'Mã excess')}
      task={tr('Pick two exponents, say which code is larger as an unsigned integer, then change the width.', 'Chọn hai số mũ, nói mã nào lớn hơn khi đọc như số nguyên không dấu, rồi đổi độ rộng.')}
      foot={<Translated vi={<>Fig. A.1 cho mọi độ rộng từ 2 đến 5 bit. Mã excess toàn bit 1 là mẫu được dành riêng; cột bù hai của nó cho thấy các bit nó dùng chung với số bù hai âm nhất.</>}><>Fig. A.1 for any width from 2 to 5 bits. The all-ones excess code is the reserved pattern; its two's-complement column shows the bits it shares with the most negative two's-complement number.</></Translated>}
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          <Translated vi={<>Số bit số mũ e = {e}, bias {bias}</>}>Exponent bits e = {e}, bias {bias}</Translated>
          <input type="range" min="2" max="5" value={e} aria-label={tr('exponent bits', 'số bit số mũ')} onChange={(ev) => setWidth(+ev.target.value)} />
        </label>
        <label className="field">
          <Translated vi="Số mũ A">Exponent A</Translated>
          <select value={A} aria-label={tr('exponent A', 'số mũ A')} onChange={(ev) => { setA(+ev.target.value); setPred({}); }}>
            {exps.map((v) => <option key={v} value={v}>{minus(v)}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Số mũ B">Exponent B</Translated>
          <select value={B} aria-label={tr('exponent B', 'số mũ B')} onChange={(ev) => { setB(+ev.target.value); setPred({}); }}>
            {exps.map((v) => <option key={v} value={v}>{minus(v)}</option>)}
          </select>
        </label>
      </div>
      <div className="cols">
        <div className="panel">
          {A === B ? <p style={{ margin: '0 0 0.7rem' }}><Translated vi="Hãy chọn hai số mũ khác nhau.">Pick two different exponents.</Translated></p> : null}
          {verdict('excess')}
          {verdict('twos')}
          <p className="muted" style={{ margin: 0 }} data-part="twos-order">
            <Translated
              vi={
                <>
                  Sắp theo mã bù hai, các số mũ chạy theo thứ tự{' '}
                  {byTwos.map((r) => (r.value === null ? '(dành riêng)' : minus(r.value))).join(', ')}: mọi số âm đứng sau mọi số dương.
                </>
              }
            >
              Sorted by two's-complement code, the exponents run{' '}
              {byTwos.map((r) => (r.value === null ? '(reserved)' : minus(r.value))).join(', ')}: every negative after every positive.
            </Translated>
          </p>
        </div>
        <div className="panel">
          <table className="data" style={{ margin: 0 }} aria-label={tr('Excess codes sorted by excess code', 'Các mã excess sắp theo mã excess')}>
            <thead>
              <tr>
                <th><Translated vi="Bù hai">Two's complement</Translated></th>
                <th className="num"><Translated vi="Số mũ">Exponent</Translated></th>
                <th><Translated vi={<>Excess-{bias}</>}>Excess-{bias}</Translated></th>{/* i18n-ignore: code name */}
                <th><Translated vi="Đã chọn">Picked</Translated></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const tag = r.value === null ? '' : [r.value === A ? 'A' : '', r.value === B ? 'B' : ''].filter(Boolean).join(', ');
                return (
                  <tr key={r.excess} data-excess={r.excess} style={tag ? { fontWeight: 650, background: 'var(--rule-soft)' } : undefined}>
                    <td className="mono">{bitString(r.twos, e)}</td>
                    <td className="num">{r.value === null ? tr('reserved', 'dành riêng') : minus(r.value)}</td>
                    <td className="mono">{bitString(r.excess, e)}</td>
                    <td>{tag}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: bits and value (Eq. A.1)
const BIT_FORMATS: Format[] = [TOY6, FP16, BF16, FP32, E4M3, E5M2, E2M1];
const SHORT: Record<string, string> = {
  get toy6() { return tr("Book's 6-bit", '6 bit của sách'); },
  fp16: 'FP16',
  bf16: 'BF16',
  fp32: 'FP32',
  e4m3: 'E4M3',
  e5m2: 'E5M2',
  e2m1: 'E2M1',
};

function classWords(a: Anatomy): string {
  switch (a.cls) {
    case 'normal':
      return tr('normal', 'normal');
    case 'subnormal':
      return tr('subnormal (denormal)', 'subnormal (denormal)');
    case 'zero':
      return Object.is(a.value, -0) ? tr('zero (negative zero)', 'số không (số không âm)') : tr('zero', 'số không');
    case 'inf':
      return tr('infinity', 'vô cực');
    case 'qnan':
      return tr('NaN (quiet)', 'NaN (quiet)');
    case 'snan':
      return tr('NaN (signaling)', 'NaN (signaling)');
    case 'nan':
      return 'NaN';
    case 'reserved':
      return tr('reserved pattern', 'mẫu dành riêng');
  }
}

function Formula({ f, a }: { f: Format; a: Anatomy }) {
  const mBits = bitString(a.M, f.m);
  if (a.rule === 'special') {
    if (a.cls === 'reserved')
      return <Translated vi={<>Sách để dành mã số mũ này (p. 585). IEEE dùng nó cho vô cực và NaN (§A.3).</>}><>The book sets this exponent code aside (p. 585). IEEE spends it on infinity and NaN (§A.3).</></Translated>;
    if (a.cls === 'inf') return <Translated vi={<>Số mũ toàn bit 1, phần định trị bằng không: (−1)<sup>{a.s}</sup> × vô cực (§A.3).</>}><>Exponent all ones, mantissa zero: (−1)<sup>{a.s}</sup> × infinity (§A.3).</></Translated>;
    if (f.id === 'e4m3') return <Translated vi={<>S.1111.111 là mẫu NaN duy nhất của E4M3; nó không có vô cực.</>}><>S.1111.111 is E4M3's only NaN pattern; it has no infinity.</></Translated>;
    return <Translated vi={<>Số mũ toàn bit 1, phần định trị khác không: NaN (§A.3).</>}><>Exponent all ones, mantissa not zero: NaN (§A.3).</></Translated>;
  }
  if (a.rule === 'flushed-pattern') return <Translated vi={<>Mã số mũ 0 khi dùng underflow đột ngột: số không.</>}><>Exponent code 0 under abrupt underflow: zero.</></Translated>;
  const sig = (a.s ? '−' : '') + show(a.significand!, 9);
  if (a.rule === 'denormal')
    return (
      <span className="mono" data-part="formula">
        (−1)<sup>{a.s}</sup> × 0.{mBits}<sub>B</sub> × 2<sup>1 − {f.bias}</sup> = {sig} × <Pow k={a.scale!} /> = {show(a.value, 9)}
      </span>
    );
  return (
    <span className="mono" data-part="formula">
      (−1)<sup>{a.s}</sup> × 1.{mBits}<sub>B</sub> × 2<sup>{a.E} − {f.bias}</sup> = {sig} × <Pow k={a.scale!} /> = {show(a.value, 9)}
    </span>
  );
}
function BitsBench() {
  const [fid, setFid] = useState('toy6');
  const f = BIT_FORMATS.find((g) => g.id === fid)!;
  const [bits, setBits] = useState(() => encode(TOY6, 0.5));
  const [text, setText] = useState('');
  const [typed, setTyped] = useState<number | null>(null);
  const [note, setNote] = useState<ReactNode>(null);
  const a = anatomy(f, bits);
  const n = totalBits(f);
  const x = fields(f, bits);

  const choose = (g: Format) => {
    if (g.id === f.id) return;
    if (totalBits(g) === n) {
      setNote(
        <>
          <Translated vi={<>Vẫn {n} bit như trước. Đọc theo {f.name}, chúng là <b>{show(a.value)}</b>.</>}>
            Same {n} bits as before. Read as {f.name} they were <b>{show(a.value)}</b>.
          </Translated>
        </>,
      );
      setTyped(null);
      setText('');
    } else {
      const v = typed ?? a.value;
      const nb = encode(g, v);
      setBits(nb >= 0 ? nb : 0);
      setNote(Number.isFinite(v) || Number.isNaN(v) ? <Translated vi={<>Đã mã hóa lại {show(v)} thành giá trị {g.name} gần nhất.</>}><>Re-encoded {show(v)} as the nearest {g.name} value.</></Translated> : null);
    }
    setFid(g.id);
  };
  const flip = (i: number) => {
    setBits(flipBit(bits, i));
    setTyped(null);
    setText('');
    setNote(null);
  };
  const onType = (t: string) => {
    setText(t);
    const v = parseValue(t);
    setNote(null);
    if (v === null) return;
    const nb = encode(f, v);
    setTyped(v);
    if (nb >= 0) setBits(nb);
  };
  const go = (to: number | null) => {
    if (to === null) return;
    setBits(bitsOfValue(f, to));
    setTyped(null);
    setText('');
    setNote(null);
  };
  const up = Number.isNaN(a.value) ? null : nextUp(f, a.value);
  const down = Number.isNaN(a.value) ? null : nextDown(f, a.value);
  const c = typed !== null ? classify(f, typed) : null;

  // Bit indices of each field, most significant first.
  const idx = (lo: number, w: number) => Array.from({ length: w }, (_, k) => lo + w - 1 - k);
  const groups = [
    ...(f.signed ? [{ key: 'S', label: 'S', idx: idx(f.e + f.m, 1) }] : []),
    { key: 'E', label: tr(`E: exponent, ${f.e} bits`, `E: số mũ, ${f.e} bit`), idx: idx(f.m, f.e) },
    { key: 'M', label: tr(`M: mantissa, ${f.m} bit${f.m === 1 ? '' : 's'}`, `M: phần định trị, ${f.m} bit`), idx: idx(0, f.m) },
  ];
  const bitOf = (i: number) => Math.floor((bits >>> 0) / 2 ** i) % 2;

  return (
    <Bench
      title={tr('Bits and value', 'Bit và giá trị')}
      task={tr('Flip bits, or type a number. Then switch between FP16 and BF16 and watch the same bits change meaning.', 'Lật bit, hoặc gõ một số. Rồi chuyển giữa FP16 và BF16 và xem cùng những bit đó đổi nghĩa.')}
      foot={<Translated vi={<>Định nghĩa: {f.source}. Giá trị được tính chính xác và hiển thị đến 9 chữ số có nghĩa.</>}><>Definitions: {f.source}. Values are computed exactly and shown to 9 significant digits.</></Translated>}
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <span className="seg" role="group" aria-label={tr('Format', 'Định dạng')} style={{ flexWrap: 'wrap', rowGap: 4 }}>
          {BIT_FORMATS.map((g) => (
            <button key={g.id} aria-pressed={g.id === fid} onClick={() => choose(g)}>
              {SHORT[g.id]}
            </button>
          ))}
        </span>
        {f.notInBook ? (
          <span className="chip" style={{ border: '1px solid var(--rule)' }} data-part="badge">
            <Translated vi="không có trong sách">not in the book</Translated>
          </span>
        ) : null}
      </div>

      <div className="row" style={{ alignItems: 'flex-start', gap: '0.5rem 0.9rem', marginBottom: '0.9rem' }} data-part="bits">
        {groups.map((g) => (
          <div key={g.key} data-field={g.key} style={{ display: 'inline-flex', flexDirection: 'column', gap: 3, maxWidth: '100%' }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
              {g.idx.map((i) => (
                <button
                  key={i}
                  data-bit={i}
                  aria-label={tr(`bit ${i}, ${g.key}`, `bit ${i}, ${g.key}`)}
                  aria-pressed={bitOf(i) === 1}
                  onClick={() => flip(i)}
                  className="mono"
                  style={{ width: '1.45rem', padding: '0.3rem 0', fontSize: '0.8rem', borderColor: 'var(--ink-3)' }}
                >
                  {bitOf(i)}
                </button>
              ))}
            </div>
            <div aria-hidden="true" style={{ height: 6, border: '1px solid var(--ink-3)', borderTop: 0 }} />
            <div className="muted" style={{ fontSize: '0.74rem', textAlign: 'center' }}>
              {g.label}
            </div>
          </div>
        ))}
      </div>

      <div className="cols">
        <div className="panel">
          <table className="data" style={{ margin: 0 }} aria-label={tr('How the bits become a value', 'Các bit trở thành một giá trị thế nào')}>
            <tbody>
              <tr>
                <th><Translated vi="Mẫu bit">Pattern</Translated></th>
                <td>
                  <span className="mono" data-part="pattern">{showBits(f, bits)}</span> <span className="muted">({hex(f, bits)})</span>
                </td>
              </tr>
              <tr>
                <th><Translated vi="Số mũ">Exponent</Translated></th>
                <td data-part="exponent">
                  <Translated vi={<>E = <span className="mono">{x.eBits}</span> = {x.E}, bias {f.bias}.</>}>
                    E = <span className="mono">{x.eBits}</span> = {x.E}, bias {f.bias}.
                  </Translated>{' '}
                  {a.rule === 'normal' || a.rule === 'nozero-low' ? (
                    <Translated vi={<>Lũy thừa của hai: E − bias = <b data-part="unbiased">{minus(a.scale!)}</b>.</>}>
                      <>
                        Power of two: E − bias = <b data-part="unbiased">{minus(a.scale!)}</b>.
                      </>
                    </Translated>
                  ) : a.rule === 'denormal' ? (
                    <Translated vi={<>E = 0: lũy thừa được cố định ở 1 − bias = <b data-part="unbiased">{minus(a.scale!)}</b> (§A.2).</>}>
                      <>
                        E = 0: the power is fixed at 1 − bias = <b data-part="unbiased">{minus(a.scale!)}</b> (§A.2).
                      </>
                    </Translated>
                  ) : (
                    <Translated vi={<>Toàn bit 1: một mẫu đặc biệt.</>}><>All ones: a special pattern.</></Translated>
                  )}
                </td>
              </tr>
              <tr>
                <th><Translated vi="Phần có nghĩa">Significand</Translated></th>
                <td data-part="significand">
                  {a.hidden === null ? (
                    '–'
                  ) : (
                    <>
                      <Translated vi={<>bit ẩn <b data-part="hidden">{a.hidden}</b> (không được lưu), rồi đến M:</>}>
                        hidden bit <b data-part="hidden">{a.hidden}</b> (not stored), then M:
                      </Translated>{' '}
                      <span className="mono">
                        {a.hidden}.{bitString(a.M, f.m)}
                        <sub>B</sub>
                      </span>{' '}
                      = {show(a.significand!, 9)}
                    </>
                  )}
                </td>
              </tr>
              <tr>
                <th><Translated vi="Công thức">Formula</Translated></th>
                <td>
                  <Formula f={f} a={a} />
                </td>
              </tr>
              <tr>
                <th><Translated vi="Giá trị">Value</Translated></th>
                <td>
                  <b style={{ fontSize: '1.15rem' }} data-part="value">{show(a.value, 9)}</b>{' '}
                  <span className="muted" data-part="class">{classWords(a)}</span>
                </td>
              </tr>
            </tbody>
          </table>
          {note ? <p style={{ margin: '0.6rem 0 0' }} data-part="note">{note}</p> : null}
        </div>

        <div className="panel">
          <label className="field" style={{ marginBottom: '0.6rem' }}>
            <Translated vi="Gõ một giá trị (thập phân, 1e-5, inf, nan)">Type a value (decimal, 1e-5, inf, nan)</Translated>
            <input type="text" inputMode="decimal" value={text} aria-label={tr('value to encode', 'giá trị cần mã hóa')} onChange={(ev) => onType(ev.target.value)} />
          </label>
          {c ? (
            <p style={{ margin: '0 0 0.7rem' }} data-part="typed">
              {c.status === 'inf' && Number.isFinite(c.input) ? (
                <Translated vi={<>{show(c.input)} vượt quá giá trị lớn nhất của {f.name}, {show(maxFinite(f))}: nó tràn thành <b>{show(c.value)}</b>.</>}>
                  <>
                    {show(c.input)} is beyond {f.name}'s largest value, {show(maxFinite(f))}: it overflows to <b>{show(c.value)}</b>.
                  </>
                </Translated>
              ) : c.status === 'nan' && Number.isFinite(c.input) ? (
                <Translated
                  vi={
                    <>
                      {show(c.input)} vượt quá giá trị lớn nhất của {f.name}, {show(maxFinite(f))}, và {f.name} không có vô cực:
                      phép chuyển đổi không bão hòa cho ra <b>NaN</b>.
                    </>
                  }
                >
                  <>
                    {show(c.input)} is beyond {f.name}'s largest value, {show(maxFinite(f))}, and {f.name} has no infinity: the
                    non-saturating conversion gives <b>NaN</b>.
                  </>
                </Translated>
              ) : c.status === 'saturated' ? (
                <Translated
                  vi={
                    <>
                      {show(c.input)} vượt quá giá trị lớn nhất của {f.name}; vì không có vô cực hay NaN để trả về, nó bị kẹp về{' '}
                      <b>{show(c.value)}</b>.
                    </>
                  }
                >
                  <>
                    {show(c.input)} is beyond {f.name}'s largest value; with no infinity or NaN to give, it is clamped to{' '}
                    <b>{show(c.value)}</b>.
                  </>
                </Translated>
              ) : Number.isFinite(c.input) ? (
                <>
                  <Translated vi={<>Giá trị {f.name} gần nhất (làm tròn về gần nhất, hòa thì về chẵn): <b>{show(c.value, 9)}</b>.</>}>
                    Nearest {f.name} value (round to nearest, ties to even): <b>{show(c.value, 9)}</b>.
                  </Translated>{' '}
                  {c.exact ? (
                    <Translated vi={<>Chính xác: không có sai số làm tròn.</>}><>Exact: no rounding error.</></Translated>
                  ) : (
                    <Translated vi={<>Sai số {show(c.error, 3)}, bằng {show(Math.abs(c.error) / ulp(f, c.input), 3)} khoảng cách giữa hai giá trị kề nhau ở đó.</>}>
                      <>
                        Error {show(c.error, 3)}, which is {show(Math.abs(c.error) / ulp(f, c.input), 3)} of the spacing between neighbouring values there.
                      </>
                    </Translated>
                  )}
                </>
              ) : (
                <Translated vi={<>Được lưu thành {show(c.value)}.</>}><>Stored as {show(c.value)}.</></Translated>
              )}
            </p>
          ) : null}
          <div className="row" style={{ marginBottom: '0.8rem' }}>
            <button onClick={() => go(down)} disabled={down === null}>
              <Translated vi="Giá trị kế tiếp phía dưới">Next value down</Translated>
            </button>
            <button onClick={() => go(up)} disabled={up === null}>
              <Translated vi="Giá trị kế tiếp phía trên">Next value up</Translated>
            </button>
          </div>
          <p className="muted" style={{ margin: 0 }} data-part="facts">
            <Translated vi={<>{f.name}: {f.e} bit số mũ cho</>}>{f.name}: {f.e} exponent bits give</Translated>{' '}
            {f.subnormals === 'nozero' ? (
              <Translated
                vi={
                  <>
                    các giá trị từ {show(minNormal(f))} đến {show(maxFinite(f))}, và không có số không: §A.1 đọc E = {bitString(0, f.e)}{' '}
                    bằng cùng công thức như mọi mã khác (§A.2 trình bày các cách sửa).
                  </>
                }
              >
                <>
                  values from {show(minNormal(f))} to {show(maxFinite(f))}, and no zero: §A.1 reads E = {bitString(0, f.e)} with
                  the same formula as every other code (§A.2 shows the fixes).
                </>
              </Translated>
            ) : (
              <Translated vi={<>các giá trị normal từ {show(minNormal(f))} đến {show(maxFinite(f))}.</>}>
                <>
                  normal values from {show(minNormal(f))} to {show(maxFinite(f))}.
                </>
              </Translated>
            )}{' '}
            <Translated
              vi={
                <>
                  {f.m} bit phần định trị cộng với bit ẩn cho {precisionBits(f)} bit có nghĩa: {2 ** f.m} giá trị giữa mỗi lũy thừa
                  của hai và lũy thừa kế tiếp.
                </>
              }
            >
              {f.m} mantissa
              bits plus the hidden one give {precisionBits(f)} significant bits: {2 ** f.m} values between each power of two and
              the next.
            </Translated>
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the dtype table
function DtypeTable() {
  const rows = [
    { f: FP32, torch: 'torch.float32' },
    { f: FP16, torch: 'torch.float16' },
    { f: BF16, torch: 'torch.bfloat16' },
  ];
  return (
    <table className="data" aria-label={tr('The three formats behind float32, float16 and bfloat16', 'Ba định dạng đứng sau float32, float16 và bfloat16')}>
      <thead>
        <tr>
          <th><Translated vi="Định dạng">Format</Translated></th>
          <th><Translated vi="dtype của PyTorch">PyTorch dtype</Translated></th>
          <th>S-E-M</th>
          <th className="num">{tr('Bias', 'Bias')}</th>
          <th className="num"><Translated vi="Số bit có nghĩa">Significant bits</Translated></th>
          <th className="num"><Translated vi="Hữu hạn lớn nhất">Largest finite</Translated></th>
          <th className="num"><Translated vi="Normal nhỏ nhất">Smallest normal</Translated></th>
        </tr>
      </thead>
      <tbody>
        {rows.map(({ f, torch }) => (
          <tr key={f.id} data-format={f.id}>
            <td>{f.name}</td>
            <td>
              <code>{torch}</code>
            </td>
            <td>
              1-{f.e}-{f.m}
            </td>
            <td className="num">{f.bias}</td>
            <td className="num">{precisionBits(f)}</td>
            <td className="num">{show(maxFinite(f), 4)}</td>
            <td className="num">
              <Pow k={Math.log2(minNormal(f))} />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSA1: Record<string, Question> = {
  'appA/A.1/q-encode': {
    kind: 'choice',
    id: 'appA/A.1/q-encode',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Trong định dạng 6 bit của sách (1 bit dấu, 3 bit số mũ theo excess-3, 2 bit phần định trị), mẫu nào lưu 0.75?
            </>
          }
        >
          In the book's 6-bit format (1 sign bit, 3 exponent bits in excess-3, 2 mantissa bits), which pattern stores 0.75?
        </Translated>
      </>
    ),
    options: [
      { label: <span className="mono">001010</span>, correct: true },
      { label: <span className="mono">001100</span>, why: <Translated vi="Đó là 1.0: mã số mũ 011 là số mũ 0 theo excess-3, và M = 00.">That is 1.0: exponent code 011 is exponent 0 in excess-3, and M = 00.</Translated> },
      { label: <span className="mono">010010</span>, why: <Translated vi="Đó là 3.0: mã số mũ 100 là số mũ +1, nên là 1.10B × 2^1.">That is 3.0: exponent code 100 is exponent +1, so 1.10B × 2^1.</Translated> },
      { label: <span className="mono">000110</span>, why: <Translated vi="Đó là 0.375: mã số mũ 001 là số mũ −2.">That is 0.375: exponent code 001 is exponent −2.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            0.75 là 0.11 ở hệ nhị phân, chuẩn hóa thành 1.1B × 2<sup>−1</sup>. Dấu 0. Số mũ −1, cộng bias 3, là 2: mã 010.
            Phần có nghĩa 1.10B bỏ đi bit 1 dẫn đầu, còn M = 10. Gộp lại: 0 010 10.
          </p>
        }
      >
        <p>
          0.75 is 0.11 in binary, which normalised is 1.1B × 2<sup>−1</sup>. Sign 0. Exponent −1, plus the bias 3, is 2:
          code 010. The significand 1.10B loses its leading 1, leaving M = 10. Together: 0 010 10.
        </p>
      </Translated>
    ),
    source: '§A.1, p. 586 (the method of the worked example for 0.5)',
    vi: { source: '§A.1, p. 586 (phương pháp của ví dụ mẫu cho 0.5)' },
  },
  'appA/A.1/q-max-half': {
    kind: 'numeric',
    id: 'appA/A.1/q-max-half',
    prompt: (
      <Translated vi={<>FP16 có 1 bit dấu, 5 bit số mũ với bias 15, và 10 bit phần định trị. Giá trị hữu hạn lớn nhất của nó là bao nhiêu?</>}>
        <>FP16 has 1 sign bit, 5 exponent bits with bias 15, and 10 mantissa bits. What is its largest finite value?</>
      </Translated>
    ),
    answer: 65504,
    tol: 0,
    explain: (
      <Translated
        vi={
          <p>
            Mã số mũ toàn bit 1, 31, được dành riêng, nên mã lớn nhất dùng được là 30, và 30 − 15 = 15. Phần định trị lớn nhất là
            toàn bit 1, nên phần có nghĩa là 2 − 2<sup>−10</sup>. (2 − 2<sup>−10</sup>) × 2<sup>15</sup> = 65536 − 32 = 65504. Một
            điểm attention hay một activation lớn hơn thế không còn chỗ nào để đi trong FP16; §A.2 cho thấy điều gì xảy ra với nó.
          </p>
        }
      >
        <p>
          The all-ones exponent code, 31, is reserved, so the largest usable code is 30, and 30 − 15 = 15. The largest
          mantissa is all ones, so the significand is 2 − 2<sup>−10</sup>. (2 − 2<sup>−10</sup>) × 2<sup>15</sup> = 65536 −
          32 = 65504. An attention score or an activation above that has nowhere to go in FP16; §A.2 shows what happens to it.
        </p>
      </Translated>
    ),
    source: 'our arithmetic on the §A.1 formula; NVIDIA guide Table 51 prints the same value (appA-facts A2)',
    vi: { source: 'phép tính của chúng tôi trên công thức §A.1; NVIDIA guide Table 51 in cùng giá trị (appA-facts A2)' },
  },
  'appA/A.1/q-unsigned-compare': {
    kind: 'choice',
    id: 'appA/A.1/q-unsigned-compare',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Một thuật toán sắp xếp đọc các bit của các giá trị FP32 không âm như số nguyên không dấu 32 bit và sắp xếp các số nguyên
              đó. Việc này có đặt các số float vào đúng thứ tự không?
            </>
          }
        >
          A sort reads the bits of non-negative FP32 values as 32-bit unsigned integers and sorts those integers. Does that put
          the floats in the right order?
        </Translated>
      </>
    ),
    options: [
      {
        label: <Translated vi="Có: bit dấu của tất cả đều là 0, và số mũ mã hóa excess nằm phía trên phần định trị.">Yes: the sign bit is 0 for all of them, and the excess-coded exponent sits above the mantissa.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Không: số mũ được lưu với bias 127, nên thứ tự số nguyên và thứ tự giá trị khác nhau.">No: the exponent is stored with a bias of 127, so the integer order and the value order differ.</Translated>,
        why: <Translated vi="Chính bias làm nó chạy được: các mã excess tăng theo các số mũ mà chúng đại diện.">The bias is what makes it work: excess codes rise with the exponents they stand for.</Translated>,
      },
      {
        label: <Translated vi="Không, nếu có số subnormal: không có bit ẩn, phần định trị của chúng không còn sắp được thứ tự nữa.">Not if some are subnormal: without the hidden bit, their mantissas no longer order them.</Translated>,
        why: <Translated vi="Các số subnormal có mã số mũ 0, thấp hơn mọi giá trị normal, và phần định trị sắp thứ tự chúng với nhau. Vẫn đúng.">Subnormals have exponent code 0, below every normal value, and their mantissas order them among themselves. Still correct.</Translated>,
      },
      {
        label: <Translated vi="Trong cùng một số mũ thì có; các giá trị có số mũ khác nhau vẫn có thể ra sai thứ tự.">Within one exponent, yes; values with different exponents can still come out in the wrong order.</Translated>,
        why: <Translated vi="Trường số mũ có trọng số cao hơn trường phần định trị, nên các số mũ khác nhau được so sánh trước, và đó là thứ tự đúng.">The exponent field is more significant than the mantissa field, so different exponents are compared first, which is the right order.</Translated>,
      },
    ],
    explain: (
      <>
        <Translated
          vi={
            <p>
              Với bit dấu 0, số mũ lớn hơn cho số nguyên lớn hơn (các mã excess có thứ tự), và khi số mũ bằng nhau thì phần định trị
              lớn hơn cho số nguyên lớn hơn. Vậy thứ tự số nguyên chính là thứ tự giá trị.
            </p>
          }
        >
          <p>
            With the sign bit 0, a larger exponent gives a larger integer (excess codes are ordered), and for equal exponents
            a larger mantissa does. So integer order is value order.
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              Với giá trị âm thì nó hỏng. Dấu là một bit riêng, nên mọi giá trị âm được đọc thành số nguyên lớn hơn mọi giá trị dương,
              và giữa các số âm, độ lớn lớn hơn lại đọc thành lớn hơn. −0.0 cũng vậy, vì bit dấu của nó được bật. Phép biến đổi khóa
              float của chương 14 sửa cả hai (<a href={href('ch14', '14.4')}>§14.4</a>).
            </p>
          }
        >
          <p>
            With negative values it fails. The sign is a separate bit, so every negative value reads as a larger integer than
            every positive one, and among negatives a larger magnitude reads larger. The same goes for −0.0, whose sign bit is
            set. Chapter 14's float-key transform fixes both (<a href={href('ch14', '14.4')}>§14.4</a>).
          </p>
        </Translated>
      </>
    ),
    source: '§A.1, p. 585 (excess codes and the unsigned comparator); the bridge to §14.4 is ours',
    vi: { source: '§A.1, p. 585 (mã excess và bộ so sánh không dấu); phần nối sang §14.4 là của khóa học này' },
  },
  'appA/A.1/q-bf16-top': {
    kind: 'choice',
    id: 'appA/A.1/q-bf16-top',
    prompt: (
      <Translated vi={<>Bạn giữ 16 bit cao của một giá trị FP32 và bỏ 16 bit thấp. Bạn có được gì?</>}>
        <>You keep the top 16 bits of an FP32 value and drop the low 16. What do you have?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Một giá trị BF16 làm tròn về phía không: lệch tới gần một bước.">A BF16 value rounded toward zero: off by up to almost one step.</Translated>, correct: true },
      {
        label: <Translated vi="Một giá trị BF16 làm tròn về gần nhất, hòa thì về chẵn: lệch nhiều nhất nửa bước BF16.">A BF16 value rounded to nearest, ties to even: off by at most half a BF16 step.</Translated>,
        why: <Translated vi="Bỏ bit không bao giờ làm tròn lên. FP32 1.0078124 giữ lại 1.0, dù 1.0078125 gần hơn.">Dropping bits never rounds up. FP32 1.0078124 keeps 1.0, though 1.0078125 is nearer.</Translated>,
      },
      {
        label: <Translated vi="Một giá trị FP16: dấu, số mũ và phần đầu của phần định trị, như mọi số float 16 bit.">An FP16 value: the sign, the exponent and the top of the mantissa, like any 16-bit float.</Translated>,
        why: <Translated vi="FP16 có 5 bit số mũ. 16 bit cao của FP32 giữ cả 8 bit số mũ của nó: đó là bố cục BF16.">FP16 has 5 exponent bits. The top 16 bits of FP32 keep all 8 of its exponent bits: that is the BF16 layout.</Translated>,
      },
      {
        label: <Translated vi="Không có gì có nghĩa: vết cắt 16 bit rơi vào giữa trường số mũ 8 bit.">Nothing meaningful: the 16-bit cut falls in the middle of the 8-bit exponent field.</Translated>,
        why: <Translated vi="Dấu và 8 bit số mũ là 9 bit cao nhất, nên chúng còn nguyên vẹn.">The sign and the 8 exponent bits are the top 9 bits, so they survive whole.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            BF16 là bit dấu của FP32, 8 bit số mũ của nó và 7 bit cao trong 23 bit phần định trị. Bỏ 16 bit thấp luôn đẩy giá trị về
            phía không, một lượng bất kỳ tới ngay dưới một bước BF16. llama.cpp thì chuyển đổi bằng làm tròn về chẵn gần nhất, bằng
            cách cộng 0x7FFF cùng bit thấp nhất được giữ trước khi dịch (<a href={href('appA', 'A.4')}>§A.4</a> giải thích việc làm
            tròn), và nó buộc NaN giữ trạng thái quiet: một NaN có mọi bit được bật đều nằm ở nửa thấp nếu không sẽ bị cắt thành vô
            cực.
          </p>
        }
      >
        <p>
          BF16 is FP32's sign, its 8 exponent bits and the top 7 of its 23 mantissa bits. Discarding the low 16 bits always
          moves the value toward zero, by anything up to just under one BF16 step. llama.cpp converts with round to nearest
          even instead, by adding 0x7FFF plus the lowest kept bit before the shift (<a href={href('appA', 'A.4')}>§A.4</a>{' '}
          explains rounding), and it forces NaNs to stay quiet: a NaN whose set bits are all in the low half would otherwise
          truncate to infinity.
        </p>
      </Translated>
    ),
    source: 'appA-facts A2 (BF16 layout) and D3 (llama.cpp ggml-impl.h); our arithmetic',
    vi: { source: 'appA-facts A2 (bố cục BF16) và D3 (llama.cpp ggml-impl.h); phép tính của chúng tôi' },
  },
};

// ---------------------------------------------------------------- the section
export function SA1() {
  return (
    <>
      <Translated
        vi={
          <p>
            Một tensor float32 trong PyTorch là một dãy các mẫu 32 bit, còn một tensor float16 hay bfloat16 là một dãy các mẫu
            16 bit. Mọi kernel bạn viết cho chúng hoặc đọc các bit đó như những con số, hoặc, với một phép sắp xếp hay chuyển
            đổi, làm việc trực tiếp trên các bit. Mục này nói về ý nghĩa của các bit.
          </p>
        }
      >
        <p>
          A float32 tensor in PyTorch is a run of 32-bit patterns, and a float16 or bfloat16 tensor a run of 16-bit ones.
          Every kernel you write for them either reads those bits as numbers or, for a sort or a conversion, works on the bits
          directly. This section is what the bits mean.
        </p>
      </Translated>

      <h2><Translated vi="Ba trường và một công thức">Three fields and one formula</Translated></h2>
      <Translated
        vi={
          <p>
            Phần cứng chia mỗi mẫu thành ba trường: một bit dấu S, một trường số mũ E và một trường phần định trị (mantissa) M.
            Bỏ qua vài mẫu dành riêng mà §A.3 đề cập, các trường này chỉ định một giá trị:
          </p>
        }
      >
        <p>
          The hardware splits each pattern into three fields: a sign bit S, an exponent field E and a mantissa field M.
          Leaving aside a few reserved patterns that §A.3 covers, the fields name one value:
        </p>
      </Translated>
      <p className="mono" style={{ textAlign: 'center', fontSize: '1.05rem' }}>
        value = (−1)<sup>S</sup> · 1.M · 2<sup>E − bias</sup>{/* i18n-ignore: equation */}
      </p>
      <Source section="A.1" pages="583–584" figs="Eq. (A.1)" />
      <Translated
        vi={
          <p>
            S chọn dấu: 0 là dương, 1 là âm. E, sau khi trừ đi một độ lệch cố định gọi là bias, chọn một lũy thừa của hai. M chọn
            một điểm giữa lũy thừa của hai đó và lũy thừa kế tiếp. Cách lưu E và M không hề tùy tiện, và cả hai lựa chọn đều lộ
            ra trong kernel: bit ẩn quyết định bạn có bao nhiêu bit có nghĩa, còn số mũ có bias là thứ cho phép một phép sắp xếp
            trên GPU coi số float như số nguyên.
          </p>
        }
      >
        <p>
          S picks the sign: 0 for positive, 1 for negative. E, after a fixed offset called the bias is taken off, picks a power
          of two. M picks a point between that power of two and the next one. How E and M are stored is not arbitrary, and
          both choices show up in kernels: the hidden bit sets how many significant bits you get, and the biased exponent is
          what lets a GPU sort treat floats as integers.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 against the IEEE SA pages (notes/today/appA-facts.md, A1)', '2026-11-20, đối chiếu với các trang IEEE SA (notes/today/appA-facts.md, A1)')}>
        <Translated
          vi={
            <p>
              IEEE 754-2019, bản sửa đổi mà sách trích dẫn, vẫn là chuẩn đang có hiệu lực. Một dự án sửa đổi và một bản thảo
              riêng cho các định dạng học máy (P3109) đang mở; chưa cái nào là chuẩn.
            </p>
          }
        >
          <p>
            IEEE 754-2019, the revision the book cites, is still the active standard. A revision project and a separate draft
            for machine-learning formats (P3109) are open; neither is a standard yet.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Bit bạn không lưu">The bit you do not store</Translated></h2>
      <Translated
        vi={
          <p>
            0.5 có thể viết ở hệ nhị phân là 0.1B × 2<sup>0</sup>, là 10.0B × 2<sup>−2</sup> hoặc là 1.0B × 2<sup>−1</sup>. Định
            dạng chỉ cho phép dạng cuối, với đúng một chữ số 1 đứng trước dấu chấm nhị phân; sách gọi các số ở dạng đó là đã chuẩn
            hóa. Hai điều suy ra từ đó. Mỗi giá trị có đúng một mẫu, nên không có mã hóa nào bị phí. Và vì chữ số trước dấu chấm
            luôn là 1, nó không bao giờ được lưu: một trường phần định trị m bit chỉ chứa các chữ số sau dấu chấm, và phần có
            nghĩa có m + 1 bit. FP16 lưu 10 bit phần định trị và có 11 bit có nghĩa; FP32 lưu 23 và có 24.
          </p>
        }
      >
        <p>
          0.5 can be written in binary as 0.1B × 2<sup>0</sup>, as 10.0B × 2<sup>−2</sup> or as 1.0B × 2<sup>−1</sup>. The
          format allows only the last shape, with exactly one 1 before the binary point; the book calls numbers in that shape
          normalised. Two things follow. Every value has one pattern, so there is no wasted encoding. And since the digit
          before the point is always 1, it is never stored: an m-bit mantissa field holds only the digits after the point, and
          the significand has m + 1 bits. FP16 stores 10 mantissa bits and has 11 significant bits; FP32 stores 23 and has 24.
        </p>
      </Translated>
      <Source section="A.1" pages="584" />

      <h2><Translated vi="Mã hóa excess">Excess encoding</Translated></h2>
      <Translated
        vi={
          <p>
            Số mũ phải có dấu. Bù hai, cách mã hóa số nguyên mà bạn đã biết, là lựa chọn hiển nhiên. IEEE dùng mã excess thay
            vào đó: cộng một bias cố định 2<sup>e−1</sup> − 1 cho một trường e bit, và lưu tổng như một số không dấu. Với 3 bit
            số mũ, bias là 3, nên số mũ −1 được lưu thành 010 và +1 thành 100.
          </p>
        }
      >
        <p>
          The exponent has to be signed. Two's complement, the integer encoding you already know, is the obvious choice. IEEE
          uses an excess code instead: add a fixed bias of 2<sup>e−1</sup> − 1 for an e-bit field, and store the sum as an
          unsigned number. With 3 exponent bits the bias is 3, so the exponent −1 is stored as 010 and +1 as 100.
        </p>
      </Translated>
      <Source section="A.1" pages="585" figs="Fig. A.1" />
      <Translated
        vi={
          <p>
            Lý do là thứ tự. Mã excess tăng theo các số mũ mà chúng đại diện, nên mạch rẻ nhất có thể, một bộ so sánh không dấu,
            so sánh trực tiếp được hai số mũ. Mã bù hai thì không tăng như vậy: đọc như số không dấu, mọi số mũ âm trông lớn hơn
            mọi số mũ dương. Hãy kiểm tra cả hai khẳng định trước khi đọc tiếp.
          </p>
        }
      >
        <p>
          The reason is order. Excess codes rise with the exponents they stand for, so the cheapest circuit there is, an
          unsigned comparator, can compare two exponents directly. Two's-complement codes do not rise: read as unsigned
          numbers, every negative exponent looks larger than every positive one. Check both claims before reading on.
        </p>
      </Translated>
      <ExcessBench />
      <Translated
        vi={
          <p>
            Bảng còn thừa một mã. Số không cộng với số lượng bằng nhau các số mũ dương và âm là một số lẻ, còn 2<sup>e</sup> mã
            là một số chẵn, nên bên nào lấy mã cuối cùng sẽ nhiều hơn bên kia một. Sách để dành mã toàn bit 1; IEEE dùng nó cho
            vô cực và NaN, là nội dung của §A.3.
          </p>
        }
      >
        <p>
          The table has one code left over. Zero plus equally many positive and negative exponents is an odd count, and 2
          <sup>e</sup> codes is an even number, so whichever side took the last code would be one larger than the other. The
          book reserves the all-ones code; IEEE uses it for infinity and NaN, which is §A.3.
        </p>
      </Translated>
      <Source section="A.1" pages="585" />

      <h3><Translated vi="Số float sắp xếp như số nguyên">Floats that sort as integers</Translated></h3>
      <Translated
        vi={
          <p>
            Thứ tự này vẫn giữ ở quy mô cả mẫu bit. Với một số float không âm, bit dấu là 0, trường số mũ nằm trên trường phần
            định trị, và cả hai đều là mã không dấu có thứ tự. Vậy các mẫu bit của số float không âm, đọc như số nguyên không
            dấu, có cùng thứ tự với các giá trị. Radix sort trên GPU, thứ sắp xếp logits cho lấy mẫu top-k và top-p, dựa vào điều
            này để coi số float là khóa số nguyên.
          </p>
        }
      >
        <p>
          The ordering survives at the scale of the whole pattern. For a non-negative float the sign bit is 0, the exponent
          field sits above the mantissa field, and both are ordered unsigned codes. So the bit patterns of non-negative floats,
          read as unsigned integers, are in the same order as the values. GPU radix sorts, which sort logits for top-k and
          top-p sampling, depend on this to treat floats as integer keys.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách dừng ở việc so sánh số mũ. Việc sắp xếp cả số float như số nguyên, và cách sửa cho giá trị âm, nằm ở chương 14
              (<a href={href('ch14', '14.4')}>§14.4</a>), theo quy tắc khóa float được ghi trong tài liệu của CUB.
            </p>
          }
        >
          <p>
            The book stops at comparing exponents. Sorting whole floats as integers, and the fix for negative values, are in
            chapter 14 (<a href={href('ch14', '14.4')}>§14.4</a>), from CUB's documented float-key rule.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsSA1['appA/A.1/q-unsigned-compare']} />

      <h2><Translated vi="Mã hóa một số bằng tay">Encoding a number by hand</Translated></h2>
      <Translated
        vi={
          <p>
            Ví dụ xuyên suốt của sách là một định dạng 6 bit: 1 bit dấu, 3 bit số mũ theo excess-3, 2 bit phần định trị. Lấy 0.5.
            Nó dương, nên S = 0. Chuẩn hóa, 0.5 = 1.0B × 2<sup>−1</sup>, nên số mũ là −1, được lưu thành −1 + 3 = 2, tức 010. Phần
            có nghĩa 1.00B bỏ đi bit 1 dẫn đầu, còn M = 00. Mẫu bit là 0 010 00.
          </p>
        }
      >
        <p>
          The book's running example is a 6-bit format: 1 sign bit, 3 exponent bits in excess-3, 2 mantissa bits. Take 0.5. It
          is positive, so S = 0. Normalised, 0.5 = 1.0B × 2<sup>−1</sup>, so the exponent is −1, stored as −1 + 3 = 2, which is
          010. The significand 1.00B loses its leading 1, leaving M = 00. The pattern is 0 010 00.
        </p>
      </Translated>
      <Source section="A.1" pages="586" />
      <Ask q={questionsSA1['appA/A.1/q-encode']} />
      <Translated
        vi={
          <p>
            Không có gì trong phương pháp này phụ thuộc vào độ rộng. Với một trường số mũ n bit, bias là 2<sup>n−1</sup> − 1, và
            một mẫu đã chuẩn hóa nghĩa là (−1)<sup>S</sup> · 1.M · 2<sup>E − (2<sup>n−1</sup> − 1)</sup>. FP32 là n = 8, bias 127;
            FP16 là n = 5, bias 15.
          </p>
        }
      >
        <p>
          Nothing in the method depends on the widths. With an n-bit exponent field the bias is 2<sup>n−1</sup> − 1, and a
          normalised pattern means (−1)<sup>S</sup> · 1.M · 2<sup>E − (2<sup>n−1</sup> − 1)</sup>. FP32 is n = 8, bias 127;
          FP16 is n = 5, bias 15.
        </p>
      </Translated>
      <Source section="A.1" pages="586" />

      <h2><Translated vi="Bit và giá trị">Bits and value</Translated></h2>
      <Translated
        vi={
          <p>
            Giờ đi theo chiều ngược lại, từ bit sang giá trị, trong định dạng của sách và trong những định dạng bạn dùng. Nhấn vào
            bit để lật chúng, hoặc gõ một số để xem mẫu bit lưu nó.
          </p>
        }
      >
        <p>
          Now go the other way, from bits to a value, in the book's format and in the ones you use. Flip bits by clicking them,
          or type a number to see the pattern that stores it.
        </p>
      </Translated>
      <BitsBench />
      <Translated
        vi={
          <p>
            Những điều cần tìm. Trong FP16, lật từng bit số mũ thì giá trị nhảy một lũy thừa của hai; lật các bit phần định trị
            thì nó di chuyển giữa một lũy thừa của hai và lũy thừa kế tiếp. Gõ 1 trong FP16, rồi chuyển sang BF16: trình khám phá
            giữ nguyên 16 bit đó, và giờ chúng mang nghĩa khác, vì hai định dạng cắt các trường ở những chỗ khác nhau. Gõ 0.1 trong
            từng định dạng và so sánh sai số: không định dạng nào lưu được nó chính xác.
          </p>
        }
      >
        <p>
          Things to find. In FP16, flip each exponent bit and the value jumps by a power of two; flip mantissa bits and it
          moves between one power of two and the next. Type 1 in FP16, then switch to BF16: the explorer keeps the same 16
          bits, and they now mean something else, because the two formats cut the fields in different places. Type 0.1 in
          each format and compare the errors: none of them can store it exactly.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Giá trị là chính xác, nhưng phần hiển thị làm tròn chúng đến 9 chữ số có nghĩa. Trình khám phá chỉ mô hình hóa cách
              mã hóa; payload của NaN và cách phần cứng xử lý chúng nằm ở §A.3.
            </p>
          }
        >
          <p>
            Values are exact, but the display rounds them to 9 significant digits. The explorer models encodings only; NaN
            payloads and how hardware treats them are §A.3.
          </p>
        </Translated>
      </Simplification>

      <h2><Translated vi="Các định dạng đứng sau dtype của bạn">The formats behind your dtypes</Translated></h2>
      <Translated vi={<p>Ba dtype dấu phẩy động bạn dùng nhiều nhất là Eq. (A.1) với các độ rộng trường khác nhau:</p>}>
        <p>The three floating-point dtypes you use most are Eq. (A.1) with different field widths:</p>
      </Translated>
      <DtypeTable />
      <Translated
        vi={
          <p>
            BF16 là nửa trên của FP32: cùng bit dấu, cùng 8 bit số mũ và do đó cùng dải giá trị, với phần định trị cắt còn 7 bit,
            8 bit có nghĩa. FP16 chuyển ba trong số các bit số mũ đó sang phần định trị: thêm 3 bit có nghĩa và dải giá trị hẹp
            hơn nhiều. Đánh đổi nào hợp với một tensor tùy thuộc vào việc các giá trị của nó cần dải giá trị hay độ phân giải; §A.2
            làm rõ điều đó.
          </p>
        }
      >
        <p>
          BF16 is the top half of FP32: the same sign bit, the same 8 exponent bits and so the same range, with the mantissa
          cut to 7 bits, 8 significant. FP16 moves three of those exponent bits into the mantissa: 3 more significant bits and
          far less range. Which trade suits a tensor depends on whether its values need range or resolution; §A.2 makes that
          concrete.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ nêu độ chính xác đơn và kép. FP16 và BF16 lấy từ hướng dẫn của NVIDIA (Tables 50 và 51), đặc tả FP8 của
              OCP và bảng dtype của PyTorch (appA-facts A2). BF16 không phải là một định dạng IEEE.
            </p>
          }
        >
          <p>
            The book gives only single and double precision. FP16 and BF16 come from NVIDIA's guide (Tables 50 and 51), OCP's
            FP8 specification and PyTorch's dtype table (appA-facts A2). BF16 is not an IEEE format.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsSA1['appA/A.1/q-max-half']} />
      <Ask q={questionsSA1['appA/A.1/q-bf16-top']} />

      <h2><Translated vi="Tên của chúng trên mỗi backend">Their names on each backend</Translated></h2>
      <Translated
        vi={
          <p>
            Trong mã kernel, các định dạng này có tên khác nhau trên mỗi backend, và chính những khác biệt đó là chỗ một tệp .cu
            duy nhất thôi biên dịch được cho HIP mà không cần sửa. Một tệp include <code>&lt;cuda_fp16.h&gt;</code> cần một{' '}
            <code>#if</code> để include <code>&lt;hip/hip_fp16.h&gt;</code> thay vào đó; một tệp dùng tên{' '}
            <code>__nv_bfloat16</code> cũng cần một typedef, đúng như llama.cpp viết trong header HIP của nó:{' '}
            <code>typedef __hip_bfloat16 nv_bfloat16;</code>. Để hỏi một định dạng về các giới hạn của nó trong mã CUDA, NVIDIA
            chỉ đến <code>cuda::std::numeric_limits</code> trong <code>&lt;cuda/std/limits&gt;</code>.
          </p>
        }
      >
        <p>
          In kernel code these formats have different names on each backend, and the differences are where a single .cu file
          stops compiling for HIP unchanged. A file that includes <code>&lt;cuda_fp16.h&gt;</code> needs an <code>#if</code>{' '}
          to include <code>&lt;hip/hip_fp16.h&gt;</code> instead; one that names <code>__nv_bfloat16</code> also needs a
          typedef, which is what llama.cpp writes in its HIP header: <code>typedef __hip_bfloat16 nv_bfloat16;</code>. To ask
          a format for its limits in CUDA code, NVIDIA points to <code>cuda::std::numeric_limits</code> in{' '}
          <code>&lt;cuda/std/limits&gt;</code>.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 against the CUDA 13.5 guide, ROCm 10.2 HIP docs, the MSL 4.1 specification and llama.cpp master (appA-facts A2, A16, B1, C1, D3)',
          '2026-11-20, đối chiếu với hướng dẫn CUDA 13.5, tài liệu HIP của ROCm 10.2, đặc tả MSL 4.1 và llama.cpp master (appA-facts A2, A16, B1, C1, D3)',
        )}
      >
        <Translated vi={<p>Không có trong sách. Mỗi tên dưới đây lấy từ tài liệu hiện hành của nhà cung cấp.</p>}>
          <p>Not in the book. Each name below is from the vendor's current documentation.</p>
        </Translated>
      </Today>
      <table className="data" aria-label={tr('Half, bfloat16 and double on each backend', 'Half, bfloat16 và double trên mỗi backend')}>
        <thead>
          <tr>
            <th>{tr('Backend', 'Backend')}</th>
            <th>FP16</th>
            <th>BF16</th>
            <th><Translated vi="double 64 bit">64-bit double</Translated></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>CUDA</td>
            <td>
              <code>__half</code>, <code>&lt;cuda_fp16.h&gt;</code>
            </td>
            <td>
              <Translated vi={<><code>__nv_bfloat16</code>, <code>&lt;cuda_bf16.h&gt;</code>; compute capability 8.0 trở lên</>}>
                <code>__nv_bfloat16</code>, <code>&lt;cuda_bf16.h&gt;</code>; compute capability 8.0 or higher
              </Translated>
            </td>
            <td>
              <code>double</code>
            </td>
          </tr>
          <tr>
            <td>HIP</td>
            <td>
              <code>__half</code>, <code>&lt;hip/hip_fp16.h&gt;</code>
            </td>
            <td>
              <code>__hip_bfloat16</code>, <code>&lt;hip/hip_bf16.h&gt;</code>
            </td>
            <td>
              <code>double</code>
            </td>
          </tr>
          <tr>
            <td>Metal</td>
            <td>
              <Translated vi={<><code>half</code>, có sẵn (lưu trữ IEEE binary16)</>}>
                <code>half</code>, built in (IEEE binary16 storage)
              </Translated>
            </td>
            <td>
              <Translated vi={<><code>bfloat</code>, Metal 3.1 và Apple6 trở lên</>}>
                <code>bfloat</code>, Metal 3.1 and Apple6 or later
              </Translated>
            </td>
            <td><Translated vi={<>không có: Metal không có <code>double</code></>}>none: Metal has no <code>double</code></Translated></td>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Trình khám phá cũng chứa ba định dạng mà sách chỉ nhắc theo kích thước, là dấu phẩy động 8 bit và 4 bit: FP8 E4M3 và
            E5M2, và FP4 E2M1. Chúng là cùng công thức với ít bit hơn và quy tắc khác cho mã số mũ cao nhất;{' '}
            <a href={href('appA', 'A.5')}>§A.5</a> giải thích chúng.
          </p>
        }
      >
        <p>
          The explorer also holds three formats the book mentions only by size, as 8-bit and 4-bit floating point: FP8 E4M3
          and E5M2, and FP4 E2M1. They are the same formula with fewer bits and different rules for the top exponent code;{' '}
          <a href={href('appA', 'A.5')}>§A.5</a> explains them.
        </p>
      </Translated>
    </>
  );
}
