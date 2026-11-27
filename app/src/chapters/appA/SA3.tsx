import { Fragment, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { Translated, tr as trText, useLocale } from '../../lib/i18n';
import {
  CLASSIFIER_FORMATS,
  OUTCOMES,
  OUTCOME_LABEL,
  classifierPresets,
  figA8Agrees,
  figA8Row,
  patternCounts,
  specialTable,
  type Outcome,
} from '../../sim/fpadd';
import { BF16, FP16, FP32, anatomy, bitsOfValue, flipBit, maxFinite, minNormal, show, totalBits, type Cls, type Format } from '../../sim/minifloat';
import { SOFTMAX_FORMATS, SOFTMAX_PRESETS, overflowThreshold, softmaxTrace, type Event, type Quantity } from '../../sim/softmaxnum';

const FACTS = 'notes/today/appA-facts.md';
const CHECKED = (items: string) => `2026-11-20 (${FACTS}, ${items})`; // i18n-ignore: a date and a citation

// ---------------------------------------------------------------- the classifier (Fig. A.8)
const CLASS_LABEL: Record<Cls, string> = {
  get zero() {
    return trText('zero', 'số 0');
  },
  get subnormal() {
    return trText('denormal', 'số denormal');
  },
  get normal() {
    return trText('normal number', 'số chuẩn');
  },
  get inf() {
    return trText('infinity', 'vô cực');
  },
  get qnan() {
    return trText('NaN, quiet', 'NaN, quiet (im lặng)');
  },
  get snan() {
    return trText('NaN, signalling', 'NaN, signalling (báo hiệu)');
  },
  nan: 'NaN',
  get reserved() {
    return trText('reserved', 'dành riêng');
  },
};
const FIG_ROWS: { key: 'nan' | 'inf' | 'denorm' | 'zero'; e: string; m: string; meaning: string }[] = [
  { key: 'nan', e: '11…1', m: '≠ 0', meaning: 'NaN' },
  { key: 'inf', e: '11…1', m: '= 0', meaning: '(−1)^S × ∞' },
  {
    key: 'denorm',
    e: '00…0',
    m: '≠ 0',
    get meaning() {
      return trText('denormalised', 'phi chuẩn hóa');
    },
  },
  { key: 'zero', e: '00…0', m: '= 0', meaning: '0' },
];
const hexOf = (f: Format, b: number) => '0x' + (b >>> 0).toString(16).padStart(Math.ceil(totalBits(f) / 4), '0');
const fmtCount = (n: number) => n.toLocaleString('en-US');

function Classifier() {
  const [fid, setFid] = useState('fp32');
  const f = CLASSIFIER_FORMATS.find((x) => x.id === fid)!;
  const [bits, setBitsRaw] = useState(() => bitsOfValue(FP32, 1));
  const [hexText, setHexText] = useState(hexOf(FP32, bitsOfValue(FP32, 1)));
  const n = totalBits(f);
  const setBits = (b: number, g: Format = f) => {
    setBitsRaw(b >>> 0);
    setHexText(hexOf(g, b));
  };
  const a = anatomy(f, bits);
  const row = figA8Row(f, bits);
  const agrees = figA8Agrees(f, bits);
  const counts = useMemo(() => patternCounts(f), [f]);
  const presets = classifierPresets(f);
  const nan = a.cls === 'qnan' || a.cls === 'snan' || a.cls === 'nan';

  const bitButton = (i: number) => {
    const on = Math.floor(bits / 2 ** i) % 2 === 1;
    return (
      <button key={i} aria-pressed={on} data-bit={i} onClick={() => setBits(flipBit(bits, i))} title={trText(`bit ${i}`, `bit ${i}`)}
        style={{ padding: '0.15rem 0', width: '1.45rem', fontFamily: 'var(--sans)', fontVariantNumeric: 'tabular-nums' }}>
        {on ? 1 : 0}
      </button>
    );
  };
  const range = (hi: number, lo: number) => Array.from({ length: hi - lo + 1 }, (_, k) => hi - k);
  const field = (label: string, idx: number[]) => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
      <span className="muted" style={{ fontSize: '0.8rem' }}>{label}</span>
      <span style={{ display: 'flex', flexWrap: 'wrap', gap: '2px' }}>{idx.map(bitButton)}</span>
    </div>
  );

  return (
    <Bench
      title={trText('Classify a bit pattern', 'Phân loại một mẫu bit')}
      task={trText(
        "Press each preset in FP32, then switch to FP8 E4M3 and press 'largest finite'.",
        "Bấm từng mẫu có sẵn trong FP32, rồi chuyển sang FP8 E4M3 và bấm 'largest finite'.",
      )}
      foot={
        <Translated vi={<>Các định dạng lấy từ chuẩn IEEE (FP32, FP16), các bảng của NVIDIA (BF16) và các đặc tả OCP (FP8, FP4). Chỉ FP32 có trong sách.</>}>
          <>Formats from the IEEE standard (FP32, FP16), NVIDIA's tables (BF16) and the OCP specifications (FP8, FP4). Only FP32 is in the book.</>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label={trText('Format', 'Định dạng')} style={{ flexWrap: 'wrap', rowGap: '4px' }}>
          {CLASSIFIER_FORMATS.map((x) => (
            <button key={x.id} aria-pressed={x.id === fid} onClick={() => { setFid(x.id); setBits(bitsOfValue(x, 1), x); }}>
              {x.name.replace('FP8 ', '').replace('FP4 ', '')}
            </button>
          ))}
        </span>
        <label className="field">
          <Translated vi="các bit ở dạng hex">bits in hex</Translated>
          <input type="text" value={hexText} aria-label={trText('bits in hex', 'các bit ở dạng hex')} style={{ width: '9rem' }}
            onChange={(e) => {
              const t = e.target.value.trim();
              setHexText(e.target.value);
              const m = /^(0x)?([0-9a-f]+)$/i.exec(t);
              if (m) {
                const v = Number.parseInt(m[2], 16);
                if (v < 2 ** n) setBitsRaw(v >>> 0);
              }
            }} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', gap: '0.4rem 0.5rem' }}>
        {presets.map((p) => (
          <button key={p.id} className="quiet" data-preset={p.id} disabled={p.bits === null} title={p.reason ?? ''}
            onClick={() => p.bits !== null && setBits(p.bits)}>
            {p.label}
          </button>
        ))}
      </div>
      {presets.some((p) => p.bits === null) ? (
        <ul className="muted" style={{ margin: '0 0 0.8rem', paddingLeft: '1.2rem' }} data-missing>
          {presets.filter((p) => p.bits === null).map((p) => <li key={p.id}><Translated vi={<>Không có {p.label}: {p.reason}.</>}><>No {p.label}: {p.reason}.</></Translated></li>)}
        </ul>
      ) : null}
      <div className="row" style={{ alignItems: 'flex-start', gap: '0.8rem 1rem', marginBottom: '0.9rem' }}>
        {f.signed ? field('S', [n - 1]) : null}
        {field(trText(`exponent (${f.e} bits, bias ${f.bias})`, `số mũ (${f.e} bit, bias ${f.bias})`), range(f.e + f.m - 1, f.m))}
        {f.m > 0 ? field(trText(`mantissa (${f.m} bits)`, `mantissa (${f.m} bit)`), range(f.m - 1, 0)) : null}
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
        <div className="panel" data-cls={a.cls} data-value={String(a.value)}>
          <p style={{ margin: '0 0 0.4rem' }}>
            <b data-class>{CLASS_LABEL[a.cls]}</b>
          </p>
          <p className="mono" style={{ margin: '0 0 0.4rem' }}>
            S = {a.s}, E = {a.E}{a.E === 2 ** f.e - 1 ? trText(' (all ones)', ' (toàn bit 1)') : a.E === 0 ? trText(' (all zeros)', ' (toàn bit 0)') : ''}, M = {a.M}
          </p>
          <p style={{ margin: 0 }} data-shown>
            {a.rule === 'special'
              ? nan
                ? a.cls === 'qnan'
                  ? trText('Top mantissa bit set: quiet. Arithmetic passes it on without a fuss.', 'Bit mantissa cao nhất bằng 1: quiet. Phép toán số học chuyển nó đi tiếp mà không ồn ào gì.')
                  : a.cls === 'snan'
                    ? trText(
                        'Top mantissa bit clear: signalling, by the IEEE convention. On a CPU, using it can raise an exception; on a GPU it cannot (see below).',
                        'Bit mantissa cao nhất bằng 0: signalling, theo quy ước IEEE. Trên CPU, dùng nó có thể gây ra một ngoại lệ; trên GPU thì không (xem bên dưới).',
                      )
                    : trText(`${f.name} does not divide its NaN patterns into quiet and signalling.`, `${f.name} không chia các mẫu NaN của nó thành quiet và signalling.`)
                : trText(`value ${show(a.value)}`, `giá trị ${show(a.value)}`)
              : a.rule === 'denormal'
                ? <Translated vi={<>giá trị 0.M × 2<sup>{1 - f.bias}</sup> = {show(a.value, 9)}</>}><>value 0.M × 2<sup>{1 - f.bias}</sup> = {show(a.value, 9)}</></Translated>
                : <Translated vi={<>giá trị 1.M × 2<sup>{a.E - f.bias}</sup> = {show(a.value, 9)}</>}><>value 1.M × 2<sup>{a.E - f.bias}</sup> = {show(a.value, 9)}</></Translated>}
          </p>
          {!agrees ? (
            <p className="t-fail" style={{ margin: '0.5rem 0 0' }} data-disagree>
              <Translated
                vi={
                  <>
                    Fig. A.8 sẽ gọi mẫu này là {row === 'nan' ? 'NaN' : row === 'inf' ? 'vô cực' : row === 'denorm' ? 'denormal' : 'số 0'}. Trong {f.name} nó là {a.cls === 'normal' ? `số ${show(a.value)}` : CLASS_LABEL[a.cls]}.
                  </>
                }
              >
                <>
                  Fig. A.8 would call this pattern {row === 'nan' ? 'NaN' : row === 'inf' ? 'infinity' : row}. In {f.name} it is {a.cls === 'normal' ? `the number ${show(a.value)}` : CLASS_LABEL[a.cls]}.
                </>
              </Translated>
            </p>
          ) : null}
        </div>
        <table className="data" style={{ margin: 0 }} aria-label="Fig. A.8">
          <thead>
            <tr><th><Translated vi="số mũ">exponent</Translated></th><th><Translated vi="mantissa">mantissa</Translated></th><th><Translated vi="ý nghĩa (Fig. A.8)">meaning (Fig. A.8)</Translated></th></tr>
          </thead>
          <tbody>
            {FIG_ROWS.map((r) => (
              <tr key={r.key} data-fig={r.key} style={r.key === row ? { background: 'var(--rule-soft)', fontWeight: 600 } : undefined}>
                <td className="mono">{r.e}</td>
                <td>{r.m}</td>
                <td>{r.meaning}</td>
              </tr>
            ))}
            <tr data-fig="other" style={row === null ? { background: 'var(--rule-soft)', fontWeight: 600 } : undefined}>
              <td colSpan={2}><Translated vi="mọi trường hợp khác">anything else</Translated></td>
              <td><Translated vi="một số chuẩn">a normal number</Translated></td>
            </tr>
          </tbody>
        </table>
      </div>
      <table className="data" style={{ background: 'var(--page)', margin: '1rem 0 0', maxWidth: '26rem' }} aria-label={trText('How the patterns are spent', 'Các mẫu bit được dùng ra sao')} data-counts={JSON.stringify(counts)}>
        <thead>
          <tr><th style={{ paddingLeft: '0.6rem' }}><Translated vi={<>{f.name}: {fmtCount(counts.total)} mẫu bit</>}><>{f.name}: {fmtCount(counts.total)} patterns</></Translated></th><th className="num"><Translated vi="bao nhiêu">how many</Translated></th></tr>
        </thead>
        <tbody>
          {([[trText('zero', 'số 0'), counts.zero], [trText('denormal', 'số denormal'), counts.subnormal], [trText('normal', 'số chuẩn'), counts.normal], [trText('infinity', 'vô cực'), counts.inf]] as [string, number][]).map(([k, v]) => (
            <tr key={k}><td style={{ paddingLeft: '0.6rem' }}>{k}</td><td className="num">{fmtCount(v)}</td></tr>
          ))}
          <tr>
            <td style={{ paddingLeft: '0.6rem' }}>NaN{counts.qnan + counts.snan ? <span className="muted"><Translated vi={<> ({fmtCount(counts.qnan)} quiet, {fmtCount(counts.snan)} signalling)</>}><> ({fmtCount(counts.qnan)} quiet, {fmtCount(counts.snan)} signalling)</></Translated></span> : null}</td>
            <td className="num" data-nan-count={counts.qnan + counts.snan + counts.nan}>{fmtCount(counts.qnan + counts.snan + counts.nan)}</td>
          </tr>
          <tr><td style={{ paddingLeft: '0.6rem' }}><Translated vi="giá trị hữu hạn lớn nhất">largest finite value</Translated></td><td className="num">{show(maxFinite(f), 6)}</td></tr>
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- what comes out?
function WhatComesOut() {
  const locale = useLocale();
  const rows = useMemo(() => specialTable(), [locale]);
  const [picks, setPicks] = useState<Record<string, Outcome | ''>>({});
  const [shown, setShown] = useState(false);
  const right = rows.filter((r) => picks[r.id] === r.outcome).length;
  const picked = rows.filter((r) => picks[r.id]).length;
  return (
    <Bench
      title={trText('What comes out?', 'Kết quả là gì?')}
      task={trText(
        'Pick an answer for every row, then reveal. Each row is evaluated in FP32 unless it says otherwise.',
        'Chọn một đáp án cho mỗi hàng, rồi bấm hiện kết quả. Mỗi hàng được tính bằng FP32 trừ khi có ghi khác.',
      )}
      foot={
        <Translated vi={<>Mọi kết quả được tính bằng mô hình float của khóa học, theo IEEE 754, còn fmaxf theo quy tắc trong tài liệu của CUDA.</>}>
          <>Every result is computed by the course's float model, which follows IEEE 754, and fmaxf by CUDA's documented rule.</>
        </Translated>
      }
    >
      <table className="data" style={{ background: 'var(--page)', margin: '0 0 0.8rem' }}>
        <thead>
          <tr><th style={{ paddingLeft: '0.6rem' }}><Translated vi="biểu thức">expression</Translated></th><th><Translated vi="đáp án của bạn">your answer</Translated></th><th>{shown ? trText('result', 'kết quả') : ''}</th></tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const p = picks[r.id] ?? '';
            const ok = p === r.outcome;
            const joined = shown ? { borderBottom: 'none' } : undefined;
            return (
              <Fragment key={r.id}>
                <tr data-row={r.id} data-outcome={r.outcome}>
                  <td className="mono" style={{ paddingLeft: '0.6rem', ...joined }}>{r.expr}</td>
                  <td style={joined}>
                    <select value={p} aria-label={trText('your answer for ' + r.expr, 'đáp án của bạn cho ' + r.expr)} style={{ maxWidth: '10rem' }}
                      onChange={(e) => setPicks({ ...picks, [r.id]: e.target.value as Outcome })}>
                      <option value="">{trText('choose', 'chọn')}</option>
                      {OUTCOMES.map((o) => <option key={o} value={o}>{OUTCOME_LABEL[o]}</option>)}
                    </select>
                  </td>
                  <td className="nowrap" style={joined} data-result>
                    {shown ? (
                      <span className={'chip ' + (p ? (ok ? 'ok' : 'fail') : '')} data-mark={p ? (ok ? 'right' : 'wrong') : 'none'}>
                        {typeof r.value === 'boolean' ? String(r.value) : show(r.value)}
                      </span>
                    ) : null}
                  </td>
                </tr>
                {shown ? (
                  <tr data-rule={r.id}>
                    <td colSpan={3} style={{ paddingLeft: '0.6rem', paddingTop: 0 }}>
                      {r.rule} <span className="muted">({r.source})</span>
                    </td>
                  </tr>
                ) : null}
              </Fragment>
            );
          })}
        </tbody>
      </table>
      <div className="row">
        <button className="primary" onClick={() => setShown(true)} disabled={picked === 0 || shown}><Translated vi="Hiện kết quả">Reveal</Translated></button>
        {shown ? <span data-score={right}><Translated vi={<>Bạn đúng {right} trên {rows.length}.</>}><>You got {right} of {rows.length}.</></Translated></span> : <span className="muted"><Translated vi={<>đã trả lời {picked} trên {rows.length}</>}><>{picked} of {rows.length} answered</></Translated></span>}
        {shown ? <button onClick={() => { setPicks({}); setShown(false); }}><Translated vi="Làm lại">Start again</Translated></button> : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- softmax overflow
const COLS: { q: Quantity | 'x'; label: string }[] = [
  {
    q: 'x',
    get label() {
      return trText('logit x', 'logit x');
    },
  },
  {
    q: 'z',
    get label() {
      return trText('z = x × scale', 'z = x × hệ số');
    },
  },
  {
    q: 'd',
    get label() {
      return trText('exp argument', 'đối số của exp');
    },
  },
  {
    q: 't',
    get label() {
      return trText('term exp(·)', 'số hạng exp(·)');
    },
  },
  {
    q: 's',
    get label() {
      return trText('running sum', 'tổng đang chạy');
    },
  },
  {
    q: 'p',
    get label() {
      return trText('probability', 'xác suất');
    },
  },
];

function causeText(e: Event, f: Format, row: { d: number }): string {
  const max = show(maxFinite(f), 6);
  switch (e.cause) {
    case 'scaled-overflow':
      return trText(
        `Logit ${e.index + 1} times the scale is past ${f.name}'s largest value, ${max}: it saturates to infinity.`,
        `Logit ${e.index + 1} nhân hệ số vượt quá giá trị lớn nhất của ${f.name}, ${max}: nó bão hòa thành vô cực.`,
      );
    case 'inf-minus-inf':
      return trText(`The maximum is infinite, and so is logit ${e.index + 1}: ∞ − ∞ is NaN.`, `Giá trị lớn nhất là vô cực, logit ${e.index + 1} cũng vậy: ∞ − ∞ là NaN.`);
    case 'exp-overflow':
      return trText(
        `exp(${show(row.d, 6)}) is past ${max}: an exp argument above about ${overflowThreshold(f).toFixed(2)} overflows ${f.name}.`,
        `exp(${show(row.d, 6)}) vượt quá ${max}: một đối số exp lớn hơn khoảng ${overflowThreshold(f).toFixed(2)} sẽ làm tràn ${f.name}.`,
      );
    case 'sum-overflow':
      return trText(`Every term is finite, but the running sum passes ${max} at element ${e.index + 1}.`, `Mọi số hạng đều hữu hạn, nhưng tổng đang chạy vượt quá ${max} ở phần tử ${e.index + 1}.`);
    case 'inf-over-inf':
      return trText(`An infinite term divided by an infinite sum: ∞ / ∞ is NaN.`, `Một số hạng vô cực chia cho một tổng vô cực: ∞ / ∞ là NaN.`);
    case 'nan-propagated':
      return trText('A NaN from an earlier step passes straight through.', 'Một NaN từ bước trước đi thẳng qua.');
  }
}

const SCALE_MIN = -2;
const SCALE_MAX = 4;

function SoftmaxOverflow() {
  const [texts, setTexts] = useState<string[]>(SOFTMAX_PRESETS[0].logits.map(String));
  const [lg, setLg] = useState(0); // log2 of the scale
  const [fid, setFid] = useState('fp32');
  const [subtract, setSubtract] = useState(false);
  const f = SOFTMAX_FORMATS.find((x) => x.id === fid)!;
  const scale = 2 ** lg;
  const xs = texts.map((t) => Number(t.trim().replace('−', '-')));
  const valid = texts.every((t) => t.trim() !== '') && xs.every((v) => Number.isFinite(v));
  const tr = useMemo(() => (valid ? softmaxTrace(xs, scale, f, subtract) : null), [texts, scale, f, subtract, valid]); // eslint-disable-line react-hooks/exhaustive-deps
  const mark = (i: number, q: Quantity) =>
    tr?.firstInf && tr.firstInf.index === i && tr.firstInf.quantity === q ? 'inf' : tr?.firstNan && tr.firstNan.index === i && tr.firstNan.quantity === q ? 'nan' : '';
  const allZero = !!tr && tr.rows.every((r) => r.p === 0);
  return (
    <Bench
      title={trText('Softmax overflow', 'Tràn số trong softmax')}
      task={trText(
        "Raise the scale in FP16 until something breaks. Then tick 'subtract the maximum'. Then try 'eight equal logits of 10' in FP16.",
        "Tăng hệ số trong FP16 cho đến khi có gì đó hỏng. Rồi đánh dấu 'trừ giá trị lớn nhất'. Rồi thử 'eight equal logits of 10' trong FP16.",
      )}
      foot={
        <Translated vi={<>Mỗi giá trị được làm tròn về định dạng đã chọn ngay khi được tính. Tổng chạy từ trái sang phải, mỗi lần một phép cộng.</>}>
          <>Each value is rounded to the chosen format as it is computed. The sum runs left to right, one addition at a time.</>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label={trText('Format', 'Định dạng')}>
          {SOFTMAX_FORMATS.map((x) => <button key={x.id} aria-pressed={x.id === fid} onClick={() => setFid(x.id)}>{x.name}</button>)}
        </span>
        <label className="field" style={{ minWidth: '14rem' }}>
          <span data-scale={scale}><Translated vi={<>hệ số (1/nhiệt độ): {Number(scale.toPrecision(3))}</>}><>scale (1/temperature): {Number(scale.toPrecision(3))}</></Translated></span>
          <input type="range" min={SCALE_MIN} max={SCALE_MAX} step={1 / 16} value={lg} aria-label={trText('scale', 'hệ số')} onChange={(e) => setLg(Number(e.target.value))} />
        </label>
        <label className="check">
          <input type="checkbox" checked={subtract} onChange={(e) => setSubtract(e.target.checked)} /> <Translated vi="trừ giá trị lớn nhất">subtract the maximum</Translated>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem', gap: '0.4rem 0.5rem' }}>
        {SOFTMAX_PRESETS.map((p) => (
          <button key={p.id} className="quiet" data-preset={p.id} onClick={() => setTexts(p.logits.map(String))}>{p.label}</button>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '0.9rem', gap: '0.4rem' }}>
        {texts.map((t, i) => (
          <input key={i} type="text" value={t} aria-label={trText(`logit ${i + 1}`, `logit ${i + 1}`)} style={{ width: '4.6rem' }}
            onChange={(e) => setTexts(texts.map((v, k) => (k === i ? e.target.value : v)))} />
        ))}
      </div>
      {tr ? (
        <>
          <table className="data" style={{ background: 'var(--page)', margin: '0 0 0.7rem' }} data-first-inf={tr.firstInf ? `${tr.firstInf.index}:${tr.firstInf.quantity}` : ''}
            data-first-nan={tr.firstNan ? `${tr.firstNan.index}:${tr.firstNan.quantity}` : ''}>
            <thead>
              <tr><th style={{ paddingLeft: '0.6rem' }}>i</th>{COLS.map((c) => <th key={c.q} className="num">{c.q === 'd' && !subtract ? trText('exp argument (= z)', 'đối số exp (= z)') : c.q === 'd' ? trText('exp argument z − max', 'đối số exp z − max') : c.label}</th>)}</tr>
            </thead>
            <tbody>
              {tr.rows.map((r, i) => (
                <tr key={i} data-i={i}>
                  <td style={{ paddingLeft: '0.6rem' }}>{i + 1}</td>
                  {COLS.map((c) => {
                    const v = r[c.q as keyof typeof r] as number;
                    const m = c.q === 'x' ? '' : mark(i, c.q as Quantity);
                    const bad = !Number.isFinite(v);
                    return (
                      <td key={c.q} className="num" data-q={c.q} data-v={String(v)} data-mark={m}
                        style={m ? { background: 'var(--fail-wash)', color: 'var(--fail)', fontWeight: 700 } : bad ? { color: 'var(--fail)' } : c.q === 't' && r.underflow ? { color: 'var(--ink-3)' } : undefined}>
                        {show(v, 6)}
                        {c.q === 't' && r.underflow ? <span className="muted">{trText(' (underflow)', ' (underflow)')}</span> : null}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          <div className={'narrate' + (tr.ok ? '' : ' bad')} role="status" data-ok={tr.ok ? 1 : 0}>
            {tr.ok ? (
              <>
                <span className="chip ok"><Translated vi="ổn">fine</Translated></span>{' '}
                <Translated vi={<>Mọi xác suất đều hữu hạn và tổng của chúng là {show(tr.sumP, 8)}.</>}><>Every probability is finite and they sum to {show(tr.sumP, 8)}.</></Translated>
                {tr.rows.some((r) => r.underflow) ? trText(' The terms marked underflow were too small for the format and became 0: harmless here, they are tiny next to the largest term.', ' Các số hạng đánh dấu underflow quá nhỏ với định dạng và thành 0: ở đây vô hại, chúng rất nhỏ so với số hạng lớn nhất.') : ''}
              </>
            ) : (
              <>
                {tr.firstInf ? <div><b className="t-fail"><Translated vi="Vô cực đầu tiên:">First infinity:</Translated></b> {causeText(tr.firstInf, f, tr.rows[tr.firstInf.index])}</div> : null}
                {tr.firstNan ? <div><b className="t-fail"><Translated vi="NaN đầu tiên:">First NaN:</Translated></b> {causeText(tr.firstNan, f, tr.rows[tr.firstNan.index])}{' '}<Translated vi="Từ đó nó lan ra, và không có gì báo lỗi.">From there it spreads, and nothing raises an error.</Translated></div> : null}
                {!tr.firstNan && allZero ? <div><b className="t-fail"><Translated vi="Không có NaN, mà vẫn sai:">No NaN, and still wrong:</Translated></b>{' '}<Translated vi="mọi xác suất là một số hạng hữu hạn chia cho một tổng vô cực, tức là 0. Tổng của chúng là 0, không phải 1.">every probability is a finite term divided by an infinite sum, which is 0. They sum to 0, not 1.</Translated></div> : null}
              </>
            )}
          </div>
          <p className="muted" style={{ margin: '0.6rem 0 0' }}>
            <Translated vi={<>Trong {f.name}, exp tràn với các đối số lớn hơn khoảng {overflowThreshold(f).toFixed(2)}.</>}>
              <>In {f.name}, exp overflows for arguments above about {overflowThreshold(f).toFixed(2)}.</>
            </Translated>{' '}
            {subtract ? trText('With the maximum subtracted, every argument is at most 0, every term at most 1, and the sum at least 1.', 'Khi đã trừ giá trị lớn nhất, mọi đối số đều không quá 0, mọi số hạng không quá 1, và tổng ít nhất là 1.') : ''}
          </p>
        </>
      ) : (
        <p className="t-fail"><Translated vi="Mỗi logit phải là một số.">Every logit must be a number.</Translated></p>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSA3: Record<string, Question> = {
  'appA/A.3/q-intervals': {
    kind: 'numeric',
    id: 'appA/A.3/q-intervals',
    prompt: (
      <Translated vi={<>Độ chính xác kép có 11 bit mũ trong khi độ chính xác đơn có 8. Số mã số mũ, và vì thế số khoảng của các số biểu diễn được, của double gấp bao nhiêu lần?</>}>
        <>Double precision has 11 exponent bits where single has 8. How many times as many exponent codes, and so intervals of representable numbers, does double have?</>
      </Translated>
    ),
    answer: 8,
    unit: 'times',
    explain: (
      <Translated
        vi={
          <p>
            2<sup>11</sup> / 2<sup>8</sup> = 2,048 / 256 = 8. Mỗi bit mũ thêm vào nhân đôi số mã, nên ba bit thêm vào nhân nó
            với 2<sup>3</sup>. Sách in "2<sup>(2<sup>3</sup>)</sup> = 256 lần", đó không phải là điều ba bit thêm vào mang lại.
          </p>
        }
      >
        <p>
          2<sup>11</sup> / 2<sup>8</sup> = 2,048 / 256 = 8. Each extra exponent bit doubles the number of codes, so three extra
          bits multiply it by 2<sup>3</sup>. The book prints "2<sup>(2<sup>3</sup>)</sup> = 256 times", which is not what
          three extra bits give.
        </p>
      </Translated>
    ),
    source: 'book §A.3, p. 590 (the printed 256 is an apparent erratum, notes/appA.md)',
    vi: { unit: 'lần', source: 'sách §A.3, p. 590 (số 256 được in có vẻ là một lỗi in, notes/appA.md)' },
  },
  'appA/A.3/q-nan-count': {
    kind: 'numeric',
    id: 'appA/A.3/q-nan-count',
    prompt: (
      <Translated vi={<>Trong 2<sup>32</sup> mẫu bit FP32, có bao nhiêu mẫu là NaN, tính cả hai dấu và cả hai loại?</>}>
        <>How many of the 2<sup>32</sup> FP32 bit patterns are NaN, counting both signs and both kinds?</>
      </Translated>
    ),
    answer: 16777214,
    unit: 'patterns',
    explain: (
      <Translated
        vi={
          <>
            <p>
              Một NaN có số mũ toàn bit 1 và mantissa bất kỳ trừ toàn bit 0 (trường hợp đó là vô cực): 2<sup>23</sup> − 1 mẫu
              cho mỗi dấu, tổng cộng 2 × (2<sup>23</sup> − 1) = 2<sup>24</sup> − 2 = 16,777,214. NVIDIA viết cùng con số đó là
              2<sup>(mantissa + 1)</sup> − 2.
            </p>
            <p>
              Con số 2<sup>22</sup> − 1 của sách chỉ đếm một loại cho một dấu: các NaN signalling, có bit mantissa cao nhất bằng
              0 và 22 bit còn lại không được toàn là 0. Các NaN quiet có 2<sup>22</sup> mẫu cho mỗi dấu. Bảng của bộ phân loại
              cho thấy sự phân chia này.
            </p>
          </>
        }
      >
        <>
          <p>
            A NaN has the exponent all ones and any mantissa except all zeros (that one is infinity): 2<sup>23</sup> − 1
            patterns per sign, 2 × (2<sup>23</sup> − 1) = 2<sup>24</sup> − 2 = 16,777,214 in all. NVIDIA writes the same as
            2<sup>(mantissa + 1)</sup> − 2.
          </p>
          <p>
            The book's 2<sup>22</sup> − 1 counts one kind for one sign: the signalling NaNs, whose top mantissa bit is 0 and
            whose other 22 bits must not all be 0. The quiet ones number 2<sup>22</sup> per sign. The classifier's table shows
            the split.
          </p>
        </>
      </Translated>
    ),
    source: `book §A.3, p. 591; ${FACTS} A7 and E1`,
    vi: { unit: 'mẫu bit', source: `sách §A.3, p. 591; ${FACTS} A7 và E1` },
  },
  'appA/A.3/q-softmax-fp16': {
    kind: 'numeric',
    id: 'appA/A.3/q-softmax-fp16',
    prompt: (
      <Translated vi={<>Một softmax tính mỗi số hạng exp(x) bằng FP16 và lưu nó trong FP16, không trừ giá trị lớn nhất. Logit nguyên x nhỏ nhất làm số hạng của nó tràn là bao nhiêu?</>}>
        <>A softmax computes each term exp(x) in FP16 and stores it in FP16, without subtracting the maximum. What is the smallest integer logit x whose term overflows?</>
      </Translated>
    ),
    answer: 12,
    explain: (
      <Translated
        vi={
          <p>
            e<sup>11</sup> ≈ 59,874, FP16 chứa được (thành 59,872). e<sup>12</sup> ≈ 162,755, vượt xa giá trị lớn nhất của FP16
            là 65,504: nó thành vô cực. Ranh giới là ln(65,520) ≈ 11.09, trong đó 65,520 là giá trị lớn nhất cộng nửa bước.
            Với FP32 và BF16, ranh giới vào khoảng 88.72.
          </p>
        }
      >
        <p>
          e<sup>11</sup> ≈ 59,874, which FP16 holds (as 59,872). e<sup>12</sup> ≈ 162,755, far past FP16's largest value,
          65,504: it becomes infinity. The boundary is ln(65,520) ≈ 11.09, where 65,520 is the largest value plus half a
          step. In FP32 and BF16 the boundary is about 88.72.
        </p>
      </Translated>
    ),
    source: 'our arithmetic on the FP16 format (appA-facts A2); checked by app/src/sim/softmaxnum.test.ts',
    vi: { source: 'phép tính của khóa học này trên định dạng FP16 (appA-facts A2); được kiểm tra bởi app/src/sim/softmaxnum.test.ts' },
  },
  'appA/A.3/q-nan-source': {
    kind: 'choice',
    id: 'appA/A.3/q-nan-source',
    prompt: (
      <Translated vi={<>Đầu ra của một kernel softmax chứa NaN, dù mọi logit đầu vào đều là số hữu hạn. Nguyên nhân có khả năng nhất là gì?</>}>
        <>A softmax kernel's output contains NaN, although every input logit was a finite number. What is the most likely cause?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Một phép exp tràn thành vô cực, và phép chia cho ra ∞ / ∞.">An exp overflowed to infinity, and the division gave ∞ / ∞.</Translated>,
        correct: true,
      },
      {
        label: (
          <Translated vi="Một số hạng rất nhỏ bị ép về 0 như một denormal, nên xác suất của nó thành 0 chia cho tổng.">
            A tiny term was flushed to zero as a denormal, so its probability came out as 0 divided by the sum.
          </Translated>
        ),
        why: (
          <Translated vi="Ép về 0 biến một giá trị rất nhỏ thành 0, và 0 chia cho một tổng khác 0 là xác suất 0, không phải NaN.">
            Flushing turns a tiny value into 0, and 0 over a nonzero sum is a 0 probability, not a NaN.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một trong các logit là số âm, và exp của một đối số âm không có giá trị xác định.">
            One of the logits was negative, and exp of a negative argument has no defined value.
          </Translated>
        ),
        why: (
          <Translated vi="exp của một số âm là một số dương nhỏ. Một số rất âm sẽ underflow về 0, điều này vô hại chừng nào tổng còn khác 0.">
            exp of a negative number is a small positive number. A very negative one underflows to 0, which is harmless while the sum stays nonzero.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Sai số làm tròn chồng chất qua tổng dài các số hạng exp cho đến khi nó thành NaN.">
            Rounding errors piled up over the long sum of exp terms until it became NaN.
          </Translated>
        ),
        why: (
          <Translated vi="Sai số làm tròn làm một tổng hơi sai; chúng không bao giờ biến nó thành NaN.">
            Rounding errors make a sum slightly wrong; they never make it NaN.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đầu vào hữu hạn chỉ tạo ra NaN qua một trong bốn phép toán không xác định của sách. Trong softmax, các ứng viên là
            ∞ / ∞ (một số hạng bị tràn chia cho một tổng bị tràn) và ∞ − ∞ (một logit bị tràn trừ một giá trị lớn nhất bị
            tràn). Cả hai đều bắt đầu bằng một lần tràn, và không có gì trên GPU báo cáo nó. Trường hợp thứ ba cần mọi số hạng
            đều underflow: khi đó tổng là 0 và mỗi xác suất là 0 / 0. Trừ giá trị lớn nhất của hàng giữ mọi đối số exp không
            quá 0, loại bỏ việc tràn, và làm số hạng của chính giá trị lớn nhất bằng đúng 1, nên tổng không thể là 0 nữa.
          </p>
        }
      >
        <p>
          Finite inputs produce a NaN only through one of the book's four undefined operations. In a softmax the candidates
          are ∞ / ∞ (an overflowed term over an overflowed sum) and ∞ − ∞ (an overflowed logit minus an overflowed maximum).
          Both start with an overflow, and nothing on a GPU reports it. A third needs every term to underflow: then the sum
          is 0 and each probability is 0 / 0. Subtracting the row's maximum keeps every exp argument at or below 0, which
          removes the overflow, and makes the maximum's own term exactly 1, so the sum can no longer be 0.
        </p>
      </Translated>
    ),
    source: 'book §A.3, pp. 590–591; the softmax reading is ours',
    vi: { source: 'sách §A.3, pp. 590–591; cách đọc cho softmax là của khóa học này' },
  },
  'appA/A.3/q-metal-nan': {
    kind: 'choice',
    id: 'appA/A.3/q-metal-nan',
    prompt: (
      <Translated vi={<>Một kernel Metal biên dịch với các tùy chọn mặc định kiểm tra kết quả bằng <code>isnan</code> để bắt một softmax bị hỏng. Nó có dựa vào phép kiểm tra đó được không?</>}>
        <>A Metal kernel compiled with the default options checks its result with <code>isnan</code> to catch a broken softmax. Can it rely on that check?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không. Dưới fast math, việc xử lý NaN và INF là không xác định, và Apple mô tả chế độ mặc định theo ba cách khác nhau.">
            No. Under fast math NaN and INF handling is undefined, and Apple documents the default mode three ways.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Có. isnan là một hàm thư viện kiểm tra các bit, nên không chế độ toán nào thay đổi được câu trả lời của nó.">
            Yes. isnan is a library function that tests the bits, so no math mode can change its answer.
          </Translated>
        ),
        why: (
          <Translated vi="Fast math cho phép trình biên dịch giả định không hề có NaN, và khi đó isnan có thể bị rút gọn thành false.">
            Fast math lets the compiler assume there are no NaNs at all, and then isnan can be folded to false.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Có, vì Metal hỗ trợ NaN signalling, chúng dừng kernel ngay tại chỗ chúng được tạo ra.">
            Yes, because Metal supports signalling NaNs, which stop the kernel where they are made.
          </Translated>
        ),
        why: (
          <Translated vi={'Trong Metal, "Signaling NaNs are not supported" (NaN signalling không được hỗ trợ), và các ngoại lệ dấu phẩy động bị tắt.'}>
            {'"Signaling NaNs are not supported" in Metal, and floating-point exceptions are disabled.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không, vì Metal hoàn toàn không có NaN: 0 / 0 trong một kernel cho ra 0.">
            No, because Metal has no NaN at all: a 0 / 0 in a kernel gives 0.
          </Translated>
        ),
        why: (
          <Translated vi="Metal hỗ trợ NaN cho float, half và bfloat, khi fast math bị tắt.">
            Metal supports NaN for float, half and bfloat, with fast math disabled.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Tài liệu web của Apple nói mặc định trên Apple silicon là relaxed (tôn trọng INF và NaN), đặc tả trình biên dịch
            của họ nói là fast, còn header của SDK nói tùy thuộc phiên bản ngôn ngữ. Hoặc chọn rõ <code>safe</code> hay{' '}
            <code>relaxed</code>, hoặc giữ các giá trị hữu hạn để không có gì phải bắt: lab 10.3 khởi đầu giá trị lớn nhất bằng{' '}
            <code>-FLT_MAX</code> thay vì <code>-INFINITY</code> vì lý do này.
          </p>
        }
      >
        <p>
          Apple's web documentation says the default on Apple silicon is relaxed (which honours INF and NaN), its compiler
          specification says fast, and the SDK header says it depends on the language version. Either select{' '}
          <code>safe</code> or <code>relaxed</code> explicitly, or keep the values finite so there is nothing to catch: lab
          10.3 starts its maximum at <code>-FLT_MAX</code> instead of <code>-INFINITY</code> for this reason.
        </p>
      </Translated>
    ),
    source: `${FACTS} C2; ch05-facts C7; ch10-facts C3`, // i18n-ignore: a citation of facts items
  },
};

// ---------------------------------------------------------------- the page
export function SA3() {
  const formats = [FP32, FP16, BF16];
  return (
    <>
      <Translated
        vi={
          <p>
            IEEE 754 cố định độ rộng các trường trong các định dạng của nó và để riêng một số mẫu bit cho những thứ không
            phải số thông thường. Mục này nói về cả hai: độ chính xác đơn và độ chính xác kép mang bao nhiêu độ chính xác, và
            những mẫu bit nào nghĩa là số 0, denormal, vô cực và NaN. Biết điều thứ hai là cách bạn đọc đầu ra của một kernel
            khi nó gặp trục trặc.
          </p>
        }
      >
        <p>
          IEEE 754 fixes the field widths of its formats and sets aside some bit patterns for things that are not ordinary
          numbers. This section is about both: how much precision single and double precision carry, and which patterns
          mean zero, denormal, infinity and NaN. Knowing the second is how you read a kernel's output when it goes wrong.
        </p>
      </Translated>

      <h2>
        <Translated vi="Đơn và kép">Single and double</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Độ chính xác đơn chia 32 bit của nó thành 1 bit dấu, 8 bit mũ và 23 bit mantissa. Độ chính xác kép chia 64 bit
            thành 1, 11 và 52. 29 bit mantissa thêm vào là nguồn gốc độ chính xác của double: mỗi bit giảm một nửa sai số biểu
            diễn lớn nhất, nên sai số của double bằng 2<sup>−29</sup> sai số của single. 3 bit mũ thêm vào là nguồn gốc phạm
            vi của nó: double đạt khoảng 10<sup>308</sup>, so với khoảng 10<sup>80</sup> nguyên tử trong vũ trụ.
          </p>
        }
      >
        <p>
          Single precision spends its 32 bits as 1 sign bit, 8 exponent bits and 23 mantissa bits. Double precision spends
          64 as 1, 11 and 52. The 29 extra mantissa bits are where double's precision comes from: each one halves the
          largest representation error, so double's is 2<sup>−29</sup> of single's. The 3 extra exponent bits are where its
          range comes from: double reaches about 10<sup>308</sup>, against about 10<sup>80</sup> atoms in the universe.
        </p>
      </Translated>
      <Source section="A.3" pages="590" />
      <Translated
        vi={
          <p>
            <b>Trong PDF</b>, các bit mũ thêm vào nhân số khoảng lên "2<sup>(2<sup>3</sup>)</sup> = 256 lần". Đó không phải là
            điều ba bit thêm vào mang lại. Hãy tính ra thừa số đúng.
          </p>
        }
      >
        <p>
          <b>In the PDF</b>, the extra exponent bits multiply the number of intervals "by 2<sup>(2<sup>3</sup>)</sup> = 256
          times". That is not what three extra bits give. Work out the right factor.
        </p>
      </Translated>
      <Ask q={questionsSA3['appA/A.3/q-intervals']} label={trText('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Các định dạng bạn sẽ dùng nhiều nhất cho suy luận thì hẹp hơn. FP16 và BF16 đều là 16 bit, chia theo cách khác
            nhau: FP16 giữ nhiều mantissa hơn, BF16 giữ phần số mũ của FP32. Lựa chọn đó quyết định cái nào tràn trước, như
            softmax bên dưới sẽ cho thấy.
          </p>
        }
      >
        <p>
          The formats you will use most for inference are narrower. FP16 and BF16 are both 16 bits, split differently: FP16
          keeps more mantissa, BF16 keeps FP32's exponent. That choice decides which of them overflows first, as the softmax
          below will show.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>FP16 và BF16 không có trong sách. Các trường và giới hạn của chúng lấy từ Table 51 của NVIDIA và các định nghĩa của OCP và Clang (appA-facts A2).</p>}>
          <p>FP16 and BF16 are not in the book. Their fields and limits are from NVIDIA's Table 51 and the OCP and Clang definitions (appA-facts A2).</p>
        </Translated>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th><Translated vi="định dạng">format</Translated></th><th className="num"><Translated vi="dấu">sign</Translated></th><th className="num"><Translated vi="số mũ">exponent</Translated></th><th className="num"><Translated vi="mantissa">mantissa</Translated></th><th className="num"><Translated vi="giá trị lớn nhất">largest value</Translated></th><th className="num"><Translated vi="số chuẩn nhỏ nhất">smallest normal</Translated></th></tr>
        </thead>
        <tbody>
          {formats.map((g) => (
            <tr key={g.id}>
              <td>{g.name}{g.notInBook ? <span className="muted">{trText(' (not in the book)', ' (không có trong sách)')}</span> : null}</td>
              <td className="num">1</td>
              <td className="num">{g.e}</td>
              <td className="num">{g.m}</td>
              <td className="num">{show(maxFinite(g), 5)}</td>
              <td className="num">2<sup>{Math.log2(minNormal(g))}</sup></td>
            </tr>
          ))}
          <tr>
            <td>FP64 (double)</td>{/* i18n-ignore: a format name */}
            <td className="num">1</td>
            <td className="num">11</td>
            <td className="num">52</td>
            <td className="num"><Translated vi="khoảng 1.8e308">about 1.8e308</Translated></td>
            <td className="num">2<sup>−1022</sup></td>
          </tr>
        </tbody>
      </table>

      <h2>
        <Translated vi="Bốn mẫu bit đặc biệt">Four special patterns</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hai mã số mũ không được dùng cho số thông thường: toàn bit 1 và toàn bit 0. Kết hợp với việc mantissa có bằng 0
            hay không, chúng tạo thành bốn nhóm (Fig. A.8):
          </p>
        }
      >
        <p>
          Two exponent codes are not used for ordinary numbers: all ones and all zeros. Together with whether the mantissa
          is zero, they make four groups (Fig. A.8):
        </p>
      </Translated>
      <Source section="A.3" pages="590" figs="Fig. A.8" />
      <Translated
        vi={
          <ul>
            <li>số mũ toàn bit 1, mantissa 0: vô cực, mang dấu của bit dấu;</li>
            <li>số mũ toàn bit 1, mantissa khác 0: NaN, "not a number" (không phải một số);</li>
            <li>số mũ toàn bit 0, mantissa 0: số 0 (có hai số, +0 và −0);</li>
            <li>số mũ toàn bit 0, mantissa khác 0: một denormal, đọc là 0.M thay vì 1.M (mục <a href={href('appA', 'A.2')}>A.2</a>).</li>
          </ul>
        }
      >
        <ul>
          <li>exponent all ones, mantissa 0: infinity, with the sign bit's sign;</li>
          <li>exponent all ones, mantissa not 0: NaN, "not a number";</li>
          <li>exponent all zeros, mantissa 0: zero (there are two, +0 and −0);</li>
          <li>exponent all zeros, mantissa not 0: a denormal, read as 0.M instead of 1.M (section <a href={href('appA', 'A.2')}>A.2</a>).</li>
        </ul>
      </Translated>
      <Ask q={questionsSA3['appA/A.3/q-nan-count']} label={trText('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Mọi mẫu bit khác là một số chuẩn. Hãy dùng bộ phân loại để tìm từng nhóm, rồi thử các định dạng hẹp. Không phải
            định dạng nào cũng theo Fig. A.8.
          </p>
        }
      >
        <p>
          Every other pattern is a normal number. Use the classifier to find each group, then try the narrow formats. They
          do not all follow Fig. A.8.
        </p>
      </Translated>
      <Classifier />
      <Translated
        vi={
          <p>
            FP8 E4M3 bỏ vô cực và chỉ giữ một mẫu NaN cho mỗi dấu, S.1111.111. Khi đó mã số mũ cao nhất của nó chứa thêm bảy
            số, lên đến 448. Theo quy tắc của Fig. A.8, mã cao nhất chỉ chứa vô cực và NaN, và giá trị E4M3 lớn nhất sẽ là
            1.875 × 2<sup>7</sup> = 240. E5M2 theo bố cục IEEE. FP4 E2M1 không có cả vô cực lẫn NaN: cả 16 mẫu bit của nó đều
            là số. Định dạng càng hẹp, nó càng ít có khả năng dành chỗ cho các giá trị đặc biệt.
          </p>
        }
      >
        <p>
          FP8 E4M3 gives up infinity and keeps one NaN pattern per sign, S.1111.111. Its top exponent code then holds seven
          more numbers, up to 448. With Fig. A.8's rule, the top code would hold only infinity and NaN, and the largest E4M3
          value would be 1.875 × 2<sup>7</sup> = 240. E5M2 follows the IEEE layout. FP4 E2M1 has neither infinity nor NaN:
          all 16 of its patterns are numbers. The narrower the format, the less it can afford to spend on special values.
        </p>
      </Translated>
      <Today checked={CHECKED('A3, A4')}>
        <Translated
          vi={
            <p>
              Đặc tả FP8 của OCP: E4M3 "không biểu diễn vô cực và chỉ dùng hai mẫu bit cho NaN ... để tăng emax lên 8 và nhờ
              đó tăng phạm vi động thêm một binade." Trong FP4 và FP6, "Không có mã nào được dành riêng cho Inf hay NaN". Con
              số 240 là phép tính của khóa học này.
            </p>
          }
        >
          <p>
            OCP's FP8 specification: E4M3 "does not represent infinities and uses only two bit patterns for NaN ... in order
            to increase emax to 8 and thus to increase the dynamic range by one binade." In FP4 and FP6 "No encodings are
            reserved for Inf or NaN". The 240 is our arithmetic.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Vô cực">Infinity</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một kết quả quá lớn với định dạng, chẳng hạn một số lớn chia cho một số rất nhỏ, không quay vòng. Nó
            <b> bão hòa</b> ở +∞ hoặc −∞ và giữ nguyên ở đó qua các phép tính sau, trừ việc mọi số biểu diễn được chia cho ±∞
            đều bằng 0. Số học số nguyên làm điều khác: một kết quả ngoài phạm vi sẽ quay vòng, hoặc là không xác định.
          </p>
        }
      >
        <p>
          A result too large for the format, such as a large number divided by a very small one, does not wrap around. It
          <b> saturates</b> at +∞ or −∞ and stays there through later arithmetic, except that any representable number
          divided by ±∞ is 0. Integer arithmetic does something else: a result out of range wraps around, or is undefined.
        </p>
      </Translated>
      <Source section="A.3" pages="590" />
      <Translated
        vi={
          <p>
            Rắc rối bắt đầu khi một vô cực đi sang số nguyên. NVIDIA gọi một phép chuyển float sang số nguyên mà kết quả không
            vừa là hành vi không xác định, kể cả NaN và vô cực. Điều đó quan trọng ngay khi bạn lượng tử hóa: một kernel biến
            một activation thành mã int8 bằng một phép ép kiểu sẽ có hành vi không xác định nếu activation đó đã thành vô cực.
          </p>
        }
      >
        <p>
          The trouble starts when an infinity crosses into integers. NVIDIA calls a float-to-integer conversion whose result
          does not fit undefined behaviour, NaN and infinity included. That matters as soon as you quantise: a kernel that
          turns an activation into an int8 code with a cast has undefined behaviour if the activation has already become
          infinity.
        </p>
      </Translated>
      <Today checked={CHECKED('A7')}>
        <Translated
          vi={
            <p>
              NVIDIA: "các phép toán tràn khỏi phạm vi biểu diễn được cho kết quả +Infinity hoặc -Infinity", và "Chuyển một
              giá trị dấu phẩy động sang một kiểu số nguyên mà kết quả không biểu diễn được là hành vi không xác định. Điều này
              cũng bao gồm NaN và vô cực."
            </p>
          }
        >
          <p>
            NVIDIA: "operations that overflow the representable range result in +Infinity or -Infinity", and "Converting a
            floating-point value to an integer type for which the result is not representable is undefined behavior. This
            also includes NaN and infinity."
          </p>
        </Translated>
      </Today>

      <h2>NaN</h2>
      <Translated
        vi={
          <p>
            NaN là kết quả của một phép toán không có giá trị xác định. Sách liệt kê bốn: 0/0, 0 × ∞, ∞/∞ và ∞ − ∞. NaN cũng là
            một dấu hiệu tiện lợi cho dữ liệu chưa từng được khởi tạo. Có hai loại, phân biệt bằng bit mantissa cao nhất. Khi
            nó bằng 0, đó là NaN <b>signalling</b>: dùng nó làm toán hạng sẽ gây ra một ngoại lệ, đó là cách bạn bắt được một
            lần đọc dữ liệu chưa khởi tạo. Khi nó bằng 1, đó là NaN <b>quiet</b>: phép tính trên nó trả về một NaN quiet khác,
            một cách âm thầm, và nó được in ra là "NaN" để bạn tìm thấy sau đó.
          </p>
        }
      >
        <p>
          A NaN is the result of an operation with no defined value. The book lists four: 0/0, 0 × ∞, ∞/∞ and ∞ − ∞. NaNs are
          also a convenient marker for data that was never initialised. There are two kinds, told apart by the top mantissa
          bit. With it clear, a <b>signalling</b> NaN: using one as an operand raises an exception, which is how you catch a
          read of uninitialised data. With it set, a <b>quiet</b> NaN: arithmetic on it returns another quiet NaN, silently,
          and it is printed as "NaN" for you to find afterwards.
        </p>
      </Translated>
      <Source section="A.3" pages="590–591" />
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi hàng được tính một lần bằng mô hình IEEE 754 của khóa học: làm tròn về số chẵn gần nhất, giữ denormal, không
              fast math. Dưới fast math của Metal, việc xử lý NaN và vô cực là không xác định, nên một kernel ở đó có thể không
              khớp.
            </p>
          }
        >
          <p>
            Each row is evaluated once by the course's IEEE 754 model: round to nearest even, denormals kept, no fast math.
            Under Metal's fast math the handling of NaN and infinity is undefined, so a kernel there may not match.
          </p>
        </Translated>
      </Simplification>
      <WhatComesOut />

      <h2>
        <Translated vi="GPU không báo hiệu">GPUs do not signal</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách nói phần cứng GPU không hỗ trợ NaN signalling: gây ra ngoại lệ một cách chính xác giữa hàng nghìn luồng là
            việc khó. Sách lưu ý rằng các bit mantissa dư của một NaN có thể mang nguyên nhân của vấn đề, và đếm được
            2<sup>22</sup> − 1 mẫu NaN trong độ chính xác đơn.
          </p>
        }
      >
        <p>
          The book says that GPU hardware does not support signalling NaNs: raising exceptions precisely in the middle of
          thousands of threads is hard. It notes that the spare mantissa bits of a NaN could carry the cause of the problem,
          and counts 2<sup>22</sup> − 1 NaN patterns in single precision.
        </p>
      </Translated>
      <Source section="A.3" pages="591" />
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th><Translated vi="Backend">Backend</Translated></th><th><Translated vi="NaN signalling">Signalling NaN</Translated></th><th><Translated vi="Ngoại lệ">Exceptions</Translated></th></tr>
        </thead>
        <tbody>
          <tr>
            <td>CUDA</td>
            <Translated vi={<td>"dù các mã hóa signaling NaN SNaN được hỗ trợ, chúng không báo hiệu và được xử lý như quiet"</td>}>
              <td>"although signaling NaN SNaN encodings are supported, they are not signaling and are handled as quiet"</td>
            </Translated>
            <Translated vi={<td>"mọi phép toán hoạt động như thể các ngoại lệ IEEE-754 luôn bị che"</td>}>
              <td>"all operations behave as if IEEE-754 exceptions are always masked"</td>
            </Translated>
          </tr>
          <tr>
            <td>HIP</td>
            <Translated vi={<td>ở chế độ IEEE, trên GFX9 đến GFX11, các đầu vào NaN signalling được chuyển thành quiet và chuyển tiếp (tài liệu AMDGPU của LLVM; tài liệu riêng của HIP không nói gì)</td>}>
              <td>in IEEE mode, on GFX9 to GFX11, signalling NaN inputs are quieted and passed on (LLVM's AMDGPU documentation; HIP's own docs say nothing)</td>
            </Translated>
            <Translated vi={<td>mặc định của Clang bỏ qua các ngoại lệ dấu phẩy động</td>}>
              <td>Clang's default ignores floating-point exceptions</td>
            </Translated>
          </tr>
          <tr>
            <td>Metal</td>
            <Translated vi={<td>"Signaling NaNs are not supported." (NaN signalling không được hỗ trợ.)</td>}>
              <td>"Signaling NaNs are not supported."</td>
            </Translated>
            <Translated vi={<td>"Floating-point exceptions are disabled in Metal." (Ngoại lệ dấu phẩy động bị tắt trong Metal.)</td>}>
              <td>"Floating-point exceptions are disabled in Metal."</td>
            </Translated>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Vì vậy trên mọi GPU bạn sẽ lập trình, một NaN di chuyển âm thầm từ chỗ nó được tạo ra đến đầu ra. Bạn cũng không
            thể trông vào các bit dư của nó để biết nó đến từ đâu: NVIDIA cảnh báo rằng "Các phép toán dấu phẩy động có thể
            thay đổi mẫu bit của payload của NaN đầu vào."
          </p>
        }
      >
        <p>
          So on every GPU you will program, a NaN travels silently from where it is made to the output. Nor can you count on
          its spare bits to say where it came from: NVIDIA warns that "Floating-point operations may alter the bit patterns
          of input NaN payloads."
        </p>
      </Translated>
      <Today checked={CHECKED('A7, B6, C2, E1')}>
        <Translated
          vi={
            <p>
              Con số 2<sup>22</sup> − 1 của sách là số NaN signalling của một dấu. Tất cả NaN FP32 cộng lại là
              2<sup>24</sup> − 2; NVIDIA viết 2<sup>(mantissa + 1)</sup> − 2.
            </p>
          }
        >
          <p>
            The book's 2<sup>22</sup> − 1 is the number of signalling NaNs of one sign. All FP32 NaNs together number
            2<sup>24</sup> − 2; NVIDIA writes 2<sup>(mantissa + 1)</sup> − 2.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Vô cực đến từ đâu: softmax">Where infinities come from: softmax</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trong một kernel suy luận, nguồn vô cực kinh điển là hàm mũ trong softmax. exp(x) vượt giá trị lớn nhất của FP16 ở
            x ≈ 11.09, và của FP32 và BF16 ở khoảng 88.72. Không gì ngăn một logit lớn đến vậy, và một nhiệt độ thấp (hệ số
            lớn hơn 1) còn nhân nó lên thêm. Một số hạng bị tràn làm mẫu số thành vô cực, xác suất của số hạng bị tràn thành
            ∞/∞ = NaN, và mọi xác suất khác thành 0. Không gì báo cáo điều đó.
          </p>
        }
      >
        <p>
          In an inference kernel the classic source of infinity is the exponential in softmax. exp(x) passes FP16's largest
          value at x ≈ 11.09, and FP32's and BF16's at about 88.72. Nothing stops a logit from being that large, and a low
          temperature (a scale above 1) multiplies it further. One overflowed term makes the denominator infinite, the overflowed
          term's probability ∞/∞ = NaN, and every other probability 0. Nothing reports it.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách không hề nhắc đến softmax. Các mảnh ghép là của sách (bão hòa, p. 590; ∞/∞ là NaN, p. 591; không báo hiệu,
              p. 591); việc ghép chúng lại là của khóa học này.
            </p>
          }
        >
          <p>
            The book never mentions softmax. The pieces are its own (saturation, p. 590; ∞/∞ is NaN, p. 591; no signalling,
            p. 591); putting them together is ours.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Cách sửa chính là cách lab 10.3 đã dùng cho mốc dự án M4: trừ giá trị lớn nhất của hàng trước hàm mũ. Khi đó mọi
            đối số đều không quá 0, mọi số hạng không quá 1, và mẫu số ít nhất là 1, vì số hạng lớn nhất là exp(0) = 1. Về mặt
            toán học không có gì thay đổi, vì thừa số e<sup>−max</sup> triệt tiêu giữa tử số và mẫu số.
          </p>
        }
      >
        <p>
          The fix is the one lab 10.3 already uses for milestone M4: subtract the row's maximum before the exponential. Then
          every argument is at most 0, every term at most 1, and the denominator at least 1, since the largest term is
          exp(0) = 1. Mathematically nothing changes, because the factor e<sup>−max</sup> cancels between numerator and
          denominator.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              exp được tính chính xác và làm tròn một lần; các hàm exp thực tế lệch vài ULP. Tổng chạy từ trái sang phải. Làm
              tròn IEEE về gần nhất, giữ denormal. Dưới fast math của Metal, hành vi của INF và NaN là không xác định, nên ở đó
              kết quả không giống bảng này.
            </p>
          }
        >
          <p>
            exp is computed exactly and rounded once; real exp functions are a few ULPs off. The sum runs left to right.
            IEEE rounding to nearest, denormals kept. Under Metal's fast math the behaviour of INF and NaN is undefined, so
            there the outcome is not this table's.
          </p>
        </Translated>
      </Simplification>
      <SoftmaxOverflow />
      <Translated
        vi={
          <p>
            Hai điều cần để ý. Các logit rất âm sẽ underflow: số hạng của chúng thành 0, điều này vô hại bên cạnh một số hạng
            bằng 1. Và với tám logit bằng 10 trong FP16, không số hạng nào tràn (mỗi số khoảng 22,000), nhưng tổng thì tràn ở
            số thứ ba: mọi xác suất thành 0 mà không hề có NaN.
          </p>
        }
      >
        <p>
          Two things to notice. Very negative logits underflow: their terms become 0, which is harmless next to a term of 1.
          And with eight logits of 10 in FP16, no term overflows (each is about 22,000), yet the sum does at the third one:
          every probability becomes 0 with no NaN at all.
        </p>
      </Translated>
      <Ask q={questionsSA3['appA/A.3/q-softmax-fp16']} label={trText('Work it out', 'Tự tính')} />
      <Ask q={questionsSA3['appA/A.3/q-nan-source']} />
      <Translated
        vi={
          <p>
            Kernel Metal của lab 10.3 đưa ra thêm một lựa chọn vì cùng lý do: giá trị lớn nhất của nó bắt đầu ở{' '}
            <code>-FLT_MAX</code>, không phải <code>-INFINITY</code>. Và một phép rút gọn max viết bằng <code>fmaxf</code> bỏ qua
            một logit NaN thay vì chuyển nó đi tiếp, nên NaN có thể biến mất khỏi giá trị lớn nhất rồi tái xuất hiện trong tổng
            (<a href={href('ch10', 'labs')}>các lab của chương 10</a>).
          </p>
        }
      >
        <p>
          Lab 10.3's Metal kernel makes one more choice for the same reason: its maximum starts at <code>-FLT_MAX</code>,
          not <code>-INFINITY</code>. And a max reduction written with <code>fmaxf</code> drops a NaN logit instead of
          passing it on, so the NaN can disappear from the maximum and resurface in the sum (<a href={href('ch10', 'labs')}>chapter 10's labs</a>).
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch10-facts C3, A10; appA-facts C2)">{/* i18n-ignore: a date and a citation */}
        <Translated
          vi={
            <p>
              Metal: "Nếu fast math được bật, hành vi xử lý NaN hoặc INF (làm đầu vào hay đầu ra) là không xác định."{' '}
              <code>fmaxf</code> của CUDA "Coi các đối số NaN là dữ liệu bị thiếu"; <code>fmax</code> của Metal cũng làm vậy.
            </p>
          }
        >
          <p>
            Metal: "If fast math is enabled the behavior of handling NaN or INF (as inputs or outputs) is undefined." CUDA's{' '}
            <code>fmaxf</code> "Treats NaN arguments as missing data"; Metal's <code>fmax</code> does the same.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsSA3['appA/A.3/q-metal-nan']} />
      <Translated
        vi={
          <p>
            Mục <a href={href('appA', 'A.4')}>A.4</a> chuyển từ các giá trị đặc biệt sang các giá trị thông thường: kết quả của
            một phép cộng có thể cách đáp án chính xác bao xa.
          </p>
        }
      >
        <p>
          Section <a href={href('appA', 'A.4')}>A.4</a> turns from special values to ordinary ones: how far the result of
          an addition can be from the exact answer.
        </p>
      </Translated>
    </>
  );
}
