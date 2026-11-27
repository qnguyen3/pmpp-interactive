import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  ALL_PRESETS,
  BOOK_ORDERS,
  FLIP,
  ORDER_LABEL,
  RULES,
  RULE_FACT,
  RULE_LABEL,
  RULE_WHO,
  SAUSAGE_INCHES,
  TOKEN_COUNTS,
  band,
  exclusive,
  exclusiveIn,
  f32,
  f32hex,
  f32short,
  inclusive,
  offsets,
  sequentialAdditions,
  toExclusive,
  toInclusive,
  topP,
  ulpSigned,
  type Arith,
  type Order,
} from '../../sim/scan';

/** A caption above a group of buttons (a <label> would forward clicks on the caption to the first button). */
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const WRAP: CSSProperties = { flexWrap: 'wrap' };

// ---------------------------------------------------------------- cut the sausage
const SIZE_PRESETS = [
  {
    id: 'book',
    get label() {
      return tr("the book's eight orders", 'tám phần gọi món của sách');
    },
    values: BOOK_ORDERS,
  },
  {
    id: 'tokens',
    get label() {
      return tr('token counts of six sequences', 'số token của sáu chuỗi');
    },
    values: TOKEN_COUNTS,
  },
];
function parseSizes(s: string): number[] | null {
  const parts = s.split(/[\s,]+/).filter(Boolean);
  if (parts.length < 1 || parts.length > 12) return null;
  const v = parts.map(Number);
  return v.every((x) => Number.isInteger(x) && x >= 0 && x <= 99) ? v : null;
}

function SausageExplorer() {
  const [values, setValues] = useState<number[]>(BOOK_ORDERS);
  const [text, setText] = useState(BOOK_ORDERS.join(' '));
  const [mode, setMode] = useState<'inclusive' | 'exclusive'>('inclusive');
  const [convert, setConvert] = useState(false);
  const inc = useMemo(() => inclusive(values), [values]);
  const exc = useMemo(() => exclusive(values), [values]);
  const off = useMemo(() => offsets(values), [values]);
  const isBook = values.join(' ') === BOOK_ORDERS.join(' ');
  const isTokens = values.join(' ') === TOKEN_COUNTS.join(' ');
  const bad = parseSizes(text) === null;
  const N = values.length;
  const length = isBook ? SAUSAGE_INCHES : Math.max(1, off.total);
  const W = 640;
  const X0 = 20;
  const scale = (W - 2 * X0) / length;
  const pos = (v: number) => X0 + v * scale;
  const marks = mode === 'inclusive' ? inc : exc;
  // Equal cut points (an order of 0) share one mark, labelled with how many cuts it is.
  const times = marks.map((m) => marks.filter((x) => x === m).length);
  const firstOf = marks.map((m, i) => marks.indexOf(m) === i);
  const unit = isBook ? tr('inches', 'inch') : isTokens ? tr('slots', 'ô') : tr('units', 'đơn vị');
  const derived = mode === 'inclusive' ? toExclusive(inc) : toInclusive(exc, values);

  return (
    <Bench
      title={tr('Cut the sausage', 'Cắt khúc xúc xích')}
      task={tr('Switch between inclusive and exclusive. Then change the orders and watch which cuts move.', 'Chuyển qua lại giữa bao gồm và loại trừ. Rồi đổi các phần gọi món và xem những vết cắt nào dịch chuyển.')}
      foot={
        <Translated
          vi={
            <>
              Mọi điểm cắt đều có được từ scan trong một lần, nên mọi khúc có thể cắt cùng lúc, theo thứ tự bất kỳ. Ở đây các
              scan là chính xác: kích thước là số nguyên.
            </>
          }
        >
          <>
            Every cut point comes from the scan in one go, so all pieces can be cut at once, in any order. The scans are
            exact here: sizes are whole numbers.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Scan">Scan</Translated>
          <span className="seg" role="group" aria-label={tr('Scan', 'Scan')}>
            <button aria-pressed={mode === 'inclusive'} onClick={() => setMode('inclusive')}><Translated vi="bao gồm: nơi các khúc kết thúc">inclusive: where pieces end</Translated></button>
            <button aria-pressed={mode === 'exclusive'} onClick={() => setMode('exclusive')}><Translated vi="loại trừ: nơi các khúc bắt đầu">exclusive: where pieces start</Translated></button>
          </span>
        </span>
        <label className="check">
          <input type="checkbox" checked={convert} onChange={(e) => setConvert(e.target.checked)} /> <Translated vi="hiện phép chuyển đổi">show the conversion</Translated>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ flex: '1 1 16rem' }}>
          <Translated vi="Kích thước, theo thứ tự (1 đến 12 số nguyên)">Sizes, in order (1 to 12 whole numbers)</Translated>
          <input
            type="text"
            value={text}
            aria-label={tr('Sizes', 'Kích thước')}
            aria-invalid={bad}
            style={{ width: '100%', fontVariantNumeric: 'tabular-nums', borderColor: bad ? 'var(--fail)' : undefined }}
            onChange={(e) => {
              setText(e.target.value);
              const v = parseSizes(e.target.value);
              if (v) setValues(v);
            }}
          />
        </label>
        <span className="seg" role="group" aria-label={tr('Presets', 'Bộ giá trị có sẵn')} style={WRAP}>
          {SIZE_PRESETS.map((p) => (
            <button key={p.id} onClick={() => { setValues(p.values); setText(p.values.join(' ')); }}>{p.label}</button>
          ))}
        </span>
      </div>
      {bad ? <p className="t-fail" style={{ marginTop: 0 }}><Translated vi="Hãy gõ 1 đến 12 số nguyên từ 0 đến 99. Trình khám phá giữ danh sách hợp lệ gần nhất.">Type 1 to 12 whole numbers from 0 to 99. The explorer keeps the last list that parsed.</Translated></p> : null}

      <svg viewBox={`0 0 ${W} 112`} role="img" aria-label={tr('The pieces laid end to end, with the cut points', 'Các khúc đặt nối đuôi nhau, cùng các điểm cắt')} data-part="sausage" data-mode={mode}
        style={{ width: '100%', maxWidth: W, minWidth: 520, display: 'block' }}>
        {isBook ? (
          <g data-part="left">
            <rect x={pos(off.total)} y={56} width={pos(SAUSAGE_INCHES) - pos(off.total)} height={24} rx={11} fill="none" stroke="var(--ink-3)" strokeDasharray="4 3" />
            <text x={(pos(off.total) + pos(SAUSAGE_INCHES)) / 2} y={72} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
              <Translated vi={<>còn lại {SAUSAGE_INCHES - off.total} inch</>}>{SAUSAGE_INCHES - off.total} inches left</Translated>
            </text>
          </g>
        ) : null}
        {values.map((v, i) =>
          v > 0 ? (
            <g key={i} data-part="piece" data-index={i} data-start={exc[i]} data-end={inc[i]}>
              <rect x={pos(exc[i])} y={56} width={v * scale} height={24} fill={i % 2 ? 'var(--rule-soft)' : 'var(--cell)'} stroke="var(--ink-2)" />
              {v * scale > 16 ? <text x={pos(exc[i]) + (v * scale) / 2} y={72} fontSize="11" textAnchor="middle">{i}</text> : null}
            </g>
          ) : null,
        )}
        {marks.map((m, i) => (
          <g key={i} data-part="cut" data-index={i} data-at={m}>
            <line x1={pos(m)} y1={48} x2={pos(m)} y2={86} stroke="var(--ink)" strokeWidth={1.6} />
            {firstOf[i] ? (
              <text x={pos(m)} y={42} fontSize="11" textAnchor="middle" fontWeight="600">{times[i] > 1 ? `${m} ×${times[i]}` : m}</text>
            ) : null}
          </g>
        ))}
        <text x={X0} y={104} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
          {mode === 'inclusive' ? tr('Cut at the end of every piece: the inclusive scan.', 'Cắt ở cuối mỗi khúc: scan bao gồm.') : tr('Mark the start of every piece: the exclusive scan.', 'Đánh dấu đầu mỗi khúc: scan loại trừ.')}
        </text>
      </svg>

      <div className="panel" style={{ overflowX: 'auto', marginTop: '0.6rem' }}>
        <table className="data" style={{ margin: 0 }} data-part="scan-table">
          <tbody>
            <tr>
              <th>i</th>
              {values.map((_, i) => <td key={i} className="num muted">{i}</td>)}
            </tr>
            <tr data-row="input">
              <th><Translated vi="kích thước">size</Translated></th>
              {values.map((v, i) => <td key={i} className="num">{v}</td>)}
            </tr>
            <tr data-row="inclusive" style={{ fontWeight: mode === 'inclusive' ? 650 : 400 }}>
              <th><Translated vi="bao gồm">inclusive</Translated></th>
              {inc.map((v, i) => <td key={i} className="num">{v}</td>)}
            </tr>
            <tr data-row="exclusive" style={{ fontWeight: mode === 'exclusive' ? 650 : 400 }}>
              <th><Translated vi="loại trừ">exclusive</Translated></th>
              {exc.map((v, i) => <td key={i} className="num">{v}</td>)}
            </tr>
          </tbody>
        </table>
      </div>

      {convert ? (
        <div className="panel" style={{ marginTop: '0.6rem' }} data-part="conversion" data-from={mode}>
          {mode === 'inclusive' ? (
            <Translated
              vi={
                <p style={{ marginTop: 0 }}>
                  Từ bao gồm sang loại trừ: dịch mọi giá trị sang phải một ô và đặt phần tử đơn vị, 0, vào ô 0.
                  Giá trị bao gồm cuối cùng, {inc[N - 1]}, rơi ra ngoài.
                </p>
              }
            >
              <p style={{ marginTop: 0 }}>
                From inclusive to exclusive: shift every value one place to the right and put the identity, 0, in slot 0.
                The last inclusive value, {inc[N - 1]}, falls off the end.
              </p>
            </Translated>
          ) : (
            <Translated
              vi={
                <p style={{ marginTop: 0 }}>
                  Từ loại trừ sang bao gồm: dịch mọi giá trị sang trái một ô, và điền ô cuối bằng giá trị loại trừ cuối cùng cộng
                  kích thước cuối cùng: {exc[N - 1]} + {values[N - 1]} = {exc[N - 1] + values[N - 1]}.
                </p>
              }
            >
              <p style={{ marginTop: 0 }}>
                From exclusive to inclusive: shift every value one place to the left, and fill the last slot with the last
                exclusive value plus the last size: {exc[N - 1]} + {values[N - 1]} = {exc[N - 1] + values[N - 1]}.
              </p>
            </Translated>
          )}
          <svg viewBox={`0 0 ${40 + N * 44} 92`} role="img" aria-label={tr('The conversion as a one-place shift', 'Phép chuyển đổi như một lần dịch một ô')} style={{ width: '100%', maxWidth: 40 + N * 44, display: 'block' }}>
            {(mode === 'inclusive' ? [inc, derived] : [exc, derived]).map((row, r) =>
              row.map((v, i) => (
                <g key={`${r}-${i}`} data-part={r ? 'derived' : 'source'} data-index={i} data-value={v}>
                  <rect x={30 + i * 44} y={4 + r * 56} width={38} height={26} rx={2} fill="var(--cell)"
                    stroke={(mode === 'inclusive' && r === 1 && i === 0) || (mode === 'exclusive' && r === 1 && i === N - 1) ? 'var(--ink)' : 'var(--rule)'}
                    strokeWidth={(mode === 'inclusive' && r === 1 && i === 0) || (mode === 'exclusive' && r === 1 && i === N - 1) ? 2 : 1} />
                  <text x={49 + i * 44} y={21 + r * 56} fontSize="12" textAnchor="middle">{v}</text>
                </g>
              )),
            )}
            {Array.from({ length: N - 1 }, (_, i) => {
              const from = mode === 'inclusive' ? i : i + 1;
              const to = mode === 'inclusive' ? i + 1 : i;
              return <line key={i} x1={49 + from * 44} y1={31} x2={49 + to * 44} y2={59} stroke="var(--ink-3)" />;
            })}
            <text x={4} y={21} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{mode === 'inclusive' ? 'inc' : 'exc'}</text>
            <text x={4} y={77} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{mode === 'inclusive' ? 'exc' : 'inc'}</text>
          </svg>
        </div>
      ) : null}

      <p style={{ marginBottom: 0 }} data-readout="sausage">
        {isTokens ? (
          <Translated
            vi={
              <>
                Chuỗi <i>k</i> chiếm các ô từ exclusive[<i>k</i>] đến inclusive[<i>k</i>] − 1: chuỗi 4 bắt đầu ở ô{' '}
                <b>{exc[4]}</b>. Buffer đóng gói cần <b>{off.total}</b> ô, tức giá trị bao gồm cuối cùng. Chuỗi 2 không có
                token nào, nên nó bắt đầu và kết thúc đúng nơi chuỗi 3 bắt đầu.
              </>
            }
          >
            <>
              Sequence <i>k</i> occupies slots exclusive[<i>k</i>] up to inclusive[<i>k</i>] − 1: sequence 4 starts at slot{' '}
              <b>{exc[4]}</b>. The packed buffer needs <b>{off.total}</b> slots, the last inclusive value. Sequence 2 has no
              tokens, so it starts and ends where sequence 3 starts.
            </>
          </Translated>
        ) : (
          <Translated
            vi={
              <>
                Khúc <i>i</i> chạy từ exclusive[<i>i</i>] đến inclusive[<i>i</i>]. Tổng cộng <b>{off.total}</b> {unit}
                {isBook ? <> của khúc xúc xích {SAUSAGE_INCHES} inch được chia ra và còn lại {SAUSAGE_INCHES - off.total}</> : null}.
                Vòng lặp tính một trong hai hàng này làm {sequentialAdditions(N)} phép cộng.
              </>
            }
          >
            <>
              Piece <i>i</i> runs from exclusive[<i>i</i>] to inclusive[<i>i</i>]. In all, <b>{off.total}</b> {unit}
              {isBook ? <> of the {SAUSAGE_INCHES}-inch sausage are served and {SAUSAGE_INCHES - off.total} are left</> : null}.
              The loop that computes either row does {sequentialAdditions(N)} additions.
            </>
          </Translated>
        )}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- where the nucleus ends
function NucleusExplorer() {
  const [presetId, setPresetId] = useState('peaked');
  const preset = ALL_PRESETS.find((p) => p.id === presetId)!;
  const [p, setP] = useState(preset.p);
  const [float, setFloat] = useState(false);
  const [order, setOrder] = useState<Order>('loop');
  const probs = preset.probs;
  const arith: Arith = float ? { kind: 'f32', order } : { kind: 'exact' };
  const results = useMemo(() => RULES.map((r) => topP(probs, p, r, arith)), [probs, p, float, order]); // eslint-disable-line react-hooks/exhaustive-deps
  const base = results[0];
  const n = probs.length;
  const crossing = base.before.findIndex((b, i) => b < (float ? f32(p) : p) && base.upto[i] >= (float ? f32(p) : p));
  const loopB = useMemo(() => exclusiveIn(probs, 'loop'), [probs]);
  const ksB = useMemo(() => exclusiveIn(probs, 'kogge-stone'), [probs]);
  const bd = useMemo(() => band(probs, p), [probs, p]);
  const pf = f32(p);
  const disagree = loopB.map((v, i) => v < pf !== ksB[i] < pf);
  const W = 640;
  const X0 = 16;
  const sx = (v: number) => X0 + v * (W - 2 * X0);
  const pick = (id: string) => {
    const pr = ALL_PRESETS.find((x) => x.id === id)!;
    setPresetId(id);
    setP(pr.p);
    if (id === 'flip') setFloat(true);
  };
  const fmt = (v: number) => (float ? f32short(v) : String(Number(v.toFixed(6))));
  /** Float steps from p, written out only when they are few enough to mean something. */
  const steps = (u: number) => (Math.abs(u) <= 999 ? String(u) : u < 0 ? tr('far below', 'thấp hơn rất xa') : tr('far above', 'cao hơn rất xa'));
  const total = n ? (float ? f32(base.before[n - 1] + f32(probs[n - 1])) : base.upto[n - 1]) : 0;

  return (
    <Bench
      title={tr('Where the nucleus ends', 'Nucleus kết thúc ở đâu')}
      task={tr('Move p until the crossing token changes. Then compare the four rules, and switch to float32.', 'Dịch p cho đến khi token vượt ngưỡng thay đổi. Rồi so sánh bốn quy tắc, và chuyển sang float32.')}
      foot={
        <Translated
          vi={
            <>
              Khúc của mỗi token chạy từ khối xác suất đứng trước nó đến khối xác suất tính đến và bao gồm nó. Quy tắc của các
              dự án giữ một token khi khúc của nó bắt đầu trước p. Chế độ chính xác dùng số học thập phân trên các xác suất như
              đã viết; float32 làm tròn mỗi phép cộng, theo thứ tự bạn chọn.
            </>
          }
        >
          <>
            Each token's piece runs from the mass before it to the mass up to and including it. A token is kept by the
            projects' rule when its piece starts before p. Exact mode uses decimal arithmetic on the probabilities as
            written; float32 rounds every addition, in the order you choose.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Các xác suất, sắp từ token cao nhất trở xuống">Probabilities, sorted from the top token down</Translated>
          <span className="seg" role="group" aria-label={tr('Rows', 'Các hàng')} style={WRAP}>
            {ALL_PRESETS.map((pr) => (
              <button key={pr.id} aria-pressed={presetId === pr.id} onClick={() => pick(pr.id)}>{pr.label}</button>
            ))}
          </span>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ flex: '1 1 14rem' }}>
          p = {p.toFixed(2)}
          <input type="range" min="1" max="99" step="1" value={Math.round(p * 100)} aria-label="p"
            onChange={(e) => setP(Number(e.target.value) / 100)} style={{ width: '100%' }} />
        </label>
        <span style={FIELD}>
          <Translated vi="Số học">Arithmetic</Translated>
          <span className="seg" role="group" aria-label={tr('Arithmetic', 'Số học')}>
            <button aria-pressed={!float} onClick={() => setFloat(false)}><Translated vi="chính xác">exact</Translated></button>
            <button aria-pressed={float} onClick={() => setFloat(true)}><Translated vi="float32">float32</Translated></button>
          </span>
        </span>
        {float ? (
          <span style={FIELD}>
            <Translated vi="Thứ tự các phép cộng">Order of the additions</Translated>
            <span className="seg" role="group" aria-label={tr('Order', 'Thứ tự')}>
              {(['loop', 'kogge-stone'] as Order[]).map((o) => (
                <button key={o} aria-pressed={order === o} onClick={() => setOrder(o)}>{ORDER_LABEL[o]}</button>
              ))}
            </span>
          </span>
        ) : null}
      </div>

      <svg viewBox={`0 0 ${W} 86`} role="img" aria-label={tr("The tokens' pieces of the unit interval, and p", 'Các khúc của token trên đoạn đơn vị, và p')} data-part="mass"
        style={{ width: '100%', maxWidth: W, minWidth: 520, display: 'block' }}>
        {probs.map((v, i) =>
          v > 0 ? (
            <g key={i} data-part="token" data-index={i}>
              <rect x={sx(base.before[i])} y={30} width={Math.max(0, sx(base.upto[i]) - sx(base.before[i]))} height={24}
                fill={base.keep[i] ? 'var(--ink-3)' : 'var(--cell)'} fillOpacity={base.keep[i] ? 0.35 : 1}
                stroke={i === crossing ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth={i === crossing ? 2.2 : 1} />
              {sx(base.upto[i]) - sx(base.before[i]) > 18 ? (
                <text x={(sx(base.before[i]) + sx(base.upto[i])) / 2} y={46} fontSize="11" textAnchor="middle">{i}</text>
              ) : null}
            </g>
          ) : null,
        )}
        <line x1={sx(p)} y1={18} x2={sx(p)} y2={64} stroke="var(--ink)" strokeWidth={2} />
        <text x={sx(p)} y={13} fontSize="12" textAnchor="middle" fontWeight="650">p = {p.toFixed(2)}</text>
        <text x={X0} y={80} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>0</text>
        <text x={W - X0} y={80} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>1</text>
        <text x={W / 2} y={80} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
          {crossing >= 0 ? tr(`token ${crossing} crosses p`, `token ${crossing} vượt qua p`) : tr('no token crosses p: p falls exactly on a cut', 'không token nào vượt qua p: p rơi đúng vào một vết cắt')}
        </text>
      </svg>

      <div className="panel" style={{ overflowX: 'auto', marginTop: '0.6rem' }}>
        <table className="data" style={{ margin: 0 }} data-part="masses">
          <tbody>
            <tr><th><Translated vi="token">token</Translated></th>{probs.map((_, i) => <td key={i} className="num muted">{i}</td>)}</tr>
            <tr><th><Translated vi="xác suất">probability</Translated></th>{probs.map((v, i) => <td key={i} className="num">{fmt(float ? f32(v) : v)}</td>)}</tr>
            <tr data-row="before"><th><Translated vi="khối trước (loại trừ)">mass before (exclusive)</Translated></th>{base.before.map((v, i) => <td key={i} className="num">{fmt(v)}</td>)}</tr>
            <tr data-row="upto"><th><Translated vi="khối tính đến (bao gồm)">mass up to (inclusive)</Translated></th>{base.upto.map((v, i) => <td key={i} className="num">{fmt(v)}</td>)}</tr>
          </tbody>
        </table>
      </div>

      <div style={{ overflowX: 'auto', marginTop: '0.8rem' }}>
      <table className="data" style={{ margin: 0, background: 'var(--page)', minWidth: '36rem' }} data-part="rules">
        <thead>
          <tr><th style={{ paddingLeft: '0.6rem' }}><Translated vi="Quy tắc">Rule</Translated></th><th><Translated vi="Ai tính theo cách này">Who computes it this way</Translated></th><th><Translated vi="Token được giữ">Tokens kept</Translated></th><th className="num"><Translated vi="Số lượng">Count</Translated></th></tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <tr key={r.rule} data-rule={r.rule} data-kept={r.kept}>
              <td style={{ paddingLeft: '0.6rem' }}>{RULE_LABEL[r.rule]}</td>
              <td className="muted">{RULE_WHO[r.rule]} <span className="nowrap">({RULE_FACT[r.rule]})</span></td>
              <td>
                <span style={{ display: 'inline-flex', gap: 3, flexWrap: 'wrap' }}>
                  {r.keep.map((k, i) => (
                    <span key={i} data-token={i} data-keep={k ? '1' : '0'} className="mono"
                      style={{ display: 'inline-block', minWidth: '1.5rem', textAlign: 'center', padding: '0 0.2rem', borderRadius: 3,
                        border: `1px ${k ? 'solid' : 'dashed'} var(--ink-3)`, background: k ? 'var(--ink)' : 'transparent', color: k ? 'var(--page)' : 'var(--ink-3)' }}>
                      {i}
                    </span>
                  ))}
                </span>
              </td>
              <td className="num"><b>{r.kept}</b></td>
            </tr>
          ))}
        </tbody>
      </table>
      </div>

      {float ? (
        <div className="panel" style={{ overflowX: 'auto', marginTop: '0.8rem' }}>
          <Translated
            vi={
              <p style={{ marginTop: 0 }}>
                Khối xác suất trước mỗi token trong float32, cộng theo hai thứ tự, so với p = {f32short(pf)} ({f32hex(pf)}). Cột
                cuối là điều một bài kiểm thử có thể đòi hỏi ở quy tắc của các dự án: bên trong dải làm tròn thì câu trả lời nào
                cũng đúng.
              </p>
            }
          >
            <p style={{ marginTop: 0 }}>
              The mass before each token in float32, summed in two orders, against p = {f32short(pf)} ({f32hex(pf)}). The
              last column is what a test may demand of the projects' rule: inside the rounding band either answer is
              right.
            </p>
          </Translated>
          <table className="data" style={{ margin: 0 }} data-part="float">
            <thead>
              <tr>
                <th><Translated vi="token">token</Translated></th><th><Translated vi="thứ tự vòng lặp">loop order</Translated></th><th className="num"><Translated vi="số bước từ p">steps from p</Translated></th><th><Translated vi="giữ?">keeps?</Translated></th>
                <th><Translated vi="thứ tự Kogge-Stone">Kogge-Stone order</Translated></th><th className="num"><Translated vi="số bước từ p">steps from p</Translated></th><th><Translated vi="giữ?">keeps?</Translated></th><th><Translated vi="bài kiểm thử chấp nhận">a test accepts</Translated></th>
              </tr>
            </thead>
            <tbody>
              {probs.map((_, i) => (
                <tr key={i} data-token={i} data-disagree={disagree[i] ? '1' : '0'} style={disagree[i] ? { fontWeight: 650 } : undefined}>
                  <td className="num">{i}</td>
                  <td className="mono">{f32short(loopB[i])} <span className="muted">{f32hex(loopB[i])}</span></td>
                  <td className="num">{steps(ulpSigned(loopB[i], pf))}</td>
                  <td>{loopB[i] < pf ? tr('keep', 'giữ') : tr('drop', 'bỏ')}</td>
                  <td className="mono">{f32short(ksB[i])} <span className="muted">{f32hex(ksB[i])}</span></td>
                  <td className="num">{steps(ulpSigned(ksB[i], pf))}</td>
                  <td>{ksB[i] < pf ? tr('keep', 'giữ') : tr('drop', 'bỏ')}</td>
                  <td data-verdict={bd.verdict[i]}>{bd.verdict[i]}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p style={{ marginBottom: 0 }} data-readout="orders">
            {disagree.some(Boolean) ? (
              <Translated
                vi={
                  <>
                    Hai thứ tự bất đồng ở token {disagree.indexOf(true)}: khối trước nó cách p{' '}
                    {ulpSigned(loopB[disagree.indexOf(true)], pf)} bước theo thứ tự vòng lặp và{' '}
                    {ulpSigned(ksB[disagree.indexOf(true)], pf)} bước theo thứ tự Kogge-Stone, nên một bên giữ nó còn bên kia bỏ nó.
                    Tính chính xác, khối trước nó là {topP(probs, p, 'exclusive').before[disagree.indexOf(true)]}: đúng bằng p.
                  </>
                }
              >
                <>
                  The two orders disagree on token {disagree.indexOf(true)}: its mass before is{' '}
                  {ulpSigned(loopB[disagree.indexOf(true)], pf)} steps from p in loop order and{' '}
                  {ulpSigned(ksB[disagree.indexOf(true)], pf)} in Kogge-Stone order, so one keeps it and the other drops it.
                  Exactly, the mass before it is {topP(probs, p, 'exclusive').before[disagree.indexOf(true)]}: precisely p.
                </>
              </Translated>
            ) : (
              <Translated vi={<>Ở đây hai thứ tự đồng ý ở mọi token. Dải làm tròn là ±{bd.bound.toExponential(1)}.</>}>The two orders agree on every token here. The rounding band is ±{bd.bound.toExponential(1)}.</Translated>
            )}
          </p>
        </div>
      ) : null}
      <p style={{ marginBottom: 0, marginTop: '0.6rem' }} data-readout="nucleus">
        <Translated
          vi={
            <>
              Tại p = {p.toFixed(2)}, quy tắc của các dự án giữ <b>{results[0].kept}</b> trên {n} token; bản lệch một giữ{' '}
              <b>{results[1].kept}</b>
              {results[1].kept === 0 ? ', không giữ token nào' : ''}.
              {results[2].kept > results[0].kept ? <> mlx-lm giữ <b>{results[2].kept}</b>, vì nó giữ mọi token đồng hạng với token ở biên.</> : null}
              {results[2].kept < results[0].kept ? <> mlx-lm giữ <b>{results[2].kept}</b>: nó so sánh với p nhân tổng mà nó tính được, {fmt(total)}, khiến ngưỡng của nó nằm ngay dưới p.</> : null}
              {results[3].kept !== results[0].kept ? <> Dạng tăng dần giữ <b>{results[3].kept}</b>: nó so sánh một tổng khác, với 1 − p, và phép làm tròn của nó rơi khác đi.</> : null}
            </>
          }
        >
          At p = {p.toFixed(2)} the projects' rule keeps <b>{results[0].kept}</b> of {n} tokens; the off-by-one keeps{' '}
          <b>{results[1].kept}</b>
          {results[1].kept === 0 ? ', none at all' : ''}.
          {results[2].kept > results[0].kept ? <> mlx-lm keeps <b>{results[2].kept}</b>, because it keeps every token tied with the boundary token.</> : null}
          {results[2].kept < results[0].kept ? <> mlx-lm keeps <b>{results[2].kept}</b>: it compares with p times the total it computed, {fmt(total)}, which puts its threshold just below p.</> : null}
          {results[3].kept !== results[0].kept ? <> The ascending form keeps <b>{results[3].kept}</b>: it compares a different sum, with 1 − p, and its rounding lands differently.</> : null}
        </Translated>
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS111: Record<string, Question> = {
  'ch11/11.1/q-rule': {
    kind: 'numeric',
    id: 'ch11/11.1/q-rule',
    prompt: (
      <Translated
        vi={
          <>
            Các xác suất đã sắp 0.4, 0.3, 0.2, 0.1 và p = 0.6. Quy tắc "giữ một token khi tổng bao gồm tính đến nó còn dưới
            p" giữ bao nhiêu token?
          </>
        }
      >
        <>
          Sorted probabilities 0.4, 0.3, 0.2, 0.1 and p = 0.6. How many tokens does the rule "keep a token while the
          inclusive sum up to it is below p" keep?
        </>
      </Translated>
    ),
    answer: 1,
    unit: 'tokens',
    vi: { unit: 'token', source: 'ch. 11 §11.1, pp. 252–253; các dự án: notes/today/ch11-facts.md D2–D4' },
    explain: (
      <Translated
        vi={
          <p>
            Các tổng bao gồm là 0.4, 0.7, 0.9, 1.0, và chỉ 0.4 nằm dưới 0.6. Các dự án giữ 2: khối trước token 1 là 0.4,
            dưới p, nên nó được giữ dù xác suất của chính nó đẩy tổng vượt quá p. Giữ token vượt qua p chính là mục đích của
            nucleus; so sánh tổng bao gồm thì lại bỏ nó. Đó là lỗi lệch một giữa hai loại scan.
          </p>
        }
      >
        <p>
          The inclusive sums are 0.4, 0.7, 0.9, 1.0, and only 0.4 is below 0.6. The projects keep 2: token 1's mass before
          it is 0.4, below p, so it is kept although its own probability carries the total past p. Keeping the token that
          crosses p is the whole point of the nucleus; comparing the inclusive sum drops it. That is the off-by-one
          between the two scans.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.1, pp. 252–253; projects: notes/today/ch11-facts.md D2–D4',
  },
  'ch11/11.1/q-total': {
    kind: 'choice',
    id: 'ch11/11.1/q-total',
    prompt: (
      <Translated vi={<>E là scan loại trừ của một mảng x gồm N phần tử. Tổng của cả N phần tử của x là:</>}>
        <>E is the exclusive scan of an array x of N elements. The sum of all N elements of x is:</>
      </Translated>
    ),
    options: [
      { label: 'E[N−1]', why: <Translated vi="E[N−1] dừng trước x[N−1].">E[N−1] stops before x[N−1].</Translated> },
      { label: 'E[N−1] + x[N−1]', correct: true },
      { label: 'E[0] + x[N−1]', why: <Translated vi="E[0] là phần tử đơn vị, 0.">E[0] is the identity, 0.</Translated> },
      { label: 'x[N−1]', why: <Translated vi="Đó chỉ là phần tử cuối cùng.">That is only the last element.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Scan loại trừ dừng thiếu một đầu vào ở mọi vị trí, kể cả vị trí cuối. Cộng lại đầu vào cuối cho ra tổng, cũng là
            cách p. 253 điền ô cuối khi chuyển một scan loại trừ thành scan bao gồm. Trong cấp phát, đó là kích thước buffer
            bạn cần.
          </p>
        }
      >
        <p>
          The exclusive scan stops one input short at every position, including the last. Adding the last input back gives
          the total, which is also how p. 253 fills the last slot when it converts an exclusive scan into an inclusive one.
          In allocation it is the size of the buffer you need.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.1, p. 253',
  },
  'ch11/11.1/q-offsets': {
    kind: 'numeric',
    id: 'ch11/11.1/q-offsets',
    prompt: (
      <Translated
        vi={
          <>
            Sáu chuỗi gồm 5, 3, 0, 7, 2 và 4 token được đóng gói vào một buffer, theo thứ tự này. Chuỗi 4 (đếm từ 0) bắt đầu ở
            ô nào?
          </>
        }
      >
        <>
          Six sequences of 5, 3, 0, 7, 2 and 4 tokens are packed into one buffer, in this order. At which slot does sequence 4
          (counting from 0) start?
        </>
      </Translated>
    ),
    answer: 15,
    explain: (
      <Translated
        vi={
          <p>
            Scan loại trừ của các số đếm là 0, 5, 8, 8, 15, 17: chuỗi 4 bắt đầu ở 15. Buffer cần 17 + 4 = 21 ô. Offset bắt
            đầu chính là scan loại trừ, đó là lý do sách gọi dạng loại trừ là dạng mà cấp phát bộ nhớ cần.
          </p>
        }
      >
        <p>
          The exclusive scan of the counts is 0, 5, 8, 8, 15, 17: sequence 4 starts at 15. The buffer needs 17 + 4 = 21
          slots. Start offsets are exclusive scans, which is why the book calls the exclusive form the one memory
          allocation needs.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.1, p. 253 (the use for token counts is ours)',
    vi: { source: 'ch. 11 §11.1, p. 253 (ứng dụng cho số token là của khóa học này)' },
  },
  'ch11/11.1/q-float-boundary': {
    kind: 'choice',
    id: 'ch11/11.1/q-float-boundary',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel GPU và một vòng lặp CPU tính scan loại trừ của cùng một hàng float32 theo các thứ tự khác nhau. Với một
            token, khối trước nó theo CPU đúng bằng p. Mặt nạ giữ của GPU có thể nói gì về token đó?
          </>
        }
      >
        <>
          A GPU kernel and a CPU loop compute the exclusive scan of the same float32 row in different orders. For one
          token, the CPU's mass before it is exactly p. What may the GPU's keep mask say about that token?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Phải bỏ nó, như CPU: cả hai tính cùng một khối chính xác.">It must drop it, like the CPU: both compute the same exact mass.</Translated>,
        why: (
          <Translated vi="Không bên nào tính khối chính xác: mỗi bên làm tròn tổng float của riêng mình, và tổng của GPU có thể rơi ngay dưới p, nên nó giữ token.">
            Neither computes the exact mass: each rounds its own float sum, and the GPU's may land just below p, so it keeps the token.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Phải giữ nó: một cây cộng các giá trị nhỏ trước, nên tổng của nó rơi thấp hơn.">It must keep it: a tree adds the small values first, so its sum lands lower.</Translated>,
        why: (
          <Translated vi="Nó cũng có thể làm tròn thành p hoặc cao hơn. Thứ tự các phép cộng không quyết định chiều làm tròn.">
            It may as well round to p or above. The order of the additions does not fix the direction of the rounding.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Cả hai: bên trong một dải do cận làm tròn đặt ra, bài kiểm thử phải chấp nhận cả hai.">Either: inside a band set by the rounding bound, a test must accept both.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Không cái nào: kernel sai nếu các tổng khác nhau dù chỉ một chút.">Neither: the kernel is wrong if the sums differ at all.</Translated>,
        why: (
          <Translated vi="Phép cộng float không có tính kết hợp. Hai thứ tự khác nhau ở các bit cuối là điều được dự liệu, không phải lỗi.">
            Float addition is not associative. Two orders differing in the last bits is expected, not a bug.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một tổng của n số float có thể xê dịch khoảng n đơn vị ở vị trí cuối, bất kể thứ tự. Vì vậy bài kiểm thử một kernel
            top-p phải chấp nhận cả hai quyết định cho những token có khối chính xác đứng trước nằm trong khoảng cách đó so với
            p, và phải dùng các hàng giá trị biểu diễn được chính xác (0.5, 0.25, 0.125, ...) để kiểm thử chính quy tắc biên,
            nơi mọi thứ tự cho ra cùng các bit.
          </p>
        }
      >
        <p>
          A sum of n floats can move by about n units in the last place, whatever the order. A test of a top-p kernel must
          therefore accept either decision for tokens whose exact mass before them lies within that distance of p, and
          must use rows of exactly representable values (0.5, 0.25, 0.125, ...) to test the boundary rule itself, where
          every order gives the same bits.
        </p>
      </Translated>
    ),
    source: 'ours, from ch10-facts A9 (float addition is not associative) and ch11-facts D4',
    vi: { source: 'của khóa học này, từ ch10-facts A9 (phép cộng float không có tính kết hợp) và ch11-facts D4' },
  },
  'ch11/11.1/q-cpu-ref': {
    kind: 'choice',
    id: 'ch11/11.1/q-cpu-ref',
    prompt: (
      <Translated
        vi={
          <>
            Vì sao các lab tính tham chiếu CPU bằng một vòng lặp tường minh từ trái sang phải thay vì bằng{' '}
            <code>std::inclusive_scan</code>?
          </>
        }
      >
        <>
          Why do the labs compute their CPU reference with an explicit left-to-right loop rather than with{' '}
          <code>std::inclusive_scan</code>?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Vòng lặp nhanh hơn trên CPU với các độ dài hàng mà lab kiểm thử.">The loop is faster on the CPU for the row lengths the labs test.</Translated>,
        why: <Translated vi="Tốc độ không phải là mục đích của một tham chiếu.">Speed is not the point of a reference.</Translated>,
      },
      {
        label: (
          <Translated vi="Chuẩn để ngỏ thứ tự của std::inclusive_scan, nên kết quả float có thể thay đổi.">
            The standard leaves std::inclusive_scan's order open, so float results may vary.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="std::inclusive_scan bỏ qua phần tử ở mỗi vị trí: thực ra nó là scan loại trừ.">
            std::inclusive_scan skips the element at each position: it is really the exclusive scan.
          </Translated>
        ),
        why: (
          <Translated vi="Nó tính scan bao gồm; std::exclusive_scan mới là cái còn lại.">
            It computes the inclusive scan; std::exclusive_scan is the other one.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="std::inclusive_scan chỉ có từ C++20, mới hơn chuẩn ngôn ngữ của các lab.">
            std::inclusive_scan arrived only in C++20, newer than the labs' language standard.
          </Translated>
        ),
        why: <Translated vi="Cả hai đều có trong <numeric> từ C++17.">{'Both are in <numeric> since C++17.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            cppreference viết về cả hai hàm: "The result is non-deterministic if the add_op is not associative (such as
            floating-point addition)." (Kết quả không tất định nếu add_op không có tính kết hợp, như phép cộng dấu phẩy động.)
            Vòng lặp của Fig. 11.1 có một thứ tự cố định, nên lần nào cũng cho cùng các bit. Kết quả GPU được so với nó theo
            một dung sai với dữ liệu float, và chỉ so từng bit với dữ liệu mà mọi tổng riêng phần đều chính xác theo mọi thứ
            tự, như toàn số 1 hoặc các số nguyên nhỏ.
          </p>
        }
      >
        <p>
          cppreference says of both functions: "The result is non-deterministic if the add_op is not associative (such as
          floating-point addition)." The loop of Fig. 11.1 has one fixed order, so it gives the same bits every time. The
          GPU result is compared with it under a tolerance for float data, and bit for bit only for data whose every
          partial sum is exact in any order, such as all ones or small integers.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.1, p. 254; notes/today/ch11-facts.md A8', // i18n-ignore: citation
  },
};

// ---------------------------------------------------------------- the section
export function S111() {
  const seq = useLabFile('ch11/snippets/s111_sequential.h');
  const d = ALL_PRESETS.find((x) => x.id === 'dyadic')!;
  const crossing = topP(d.probs, d.p, 'exclusive');
  return (
    <>
      <Translated
        vi={
          <p>
            Một <b>scan bao gồm</b> (inclusive scan) nhận một toán tử hai ngôi ⊕ và một mảng x₀, x₁, …, x<sub>N−1</sub>, rồi
            trả về x₀, x₀ ⊕ x₁, …, x₀ ⊕ x₁ ⊕ … ⊕ x<sub>N−1</sub>: đầu ra <i>i</i> kết hợp mọi đầu vào tính đến và bao gồm đầu
            vào <i>i</i>. Với phép cộng, nó còn được gọi là <b>tổng tiền tố (prefix sum)</b>, ví dụ xuyên suốt của sách: 3 1 7
            0 4 1 6 3 scan thành 3 4 11 11 15 16 22 25.
          </p>
        }
      >
        <p>
          An <b>inclusive scan</b> takes a binary operator ⊕ and an array x₀, x₁, …, x<sub>N−1</sub>, and returns x₀, x₀ ⊕
          x₁, …, x₀ ⊕ x₁ ⊕ … ⊕ x<sub>N−1</sub>: output <i>i</i> combines every input up to and including input <i>i</i>.
          With addition it is also called a <b>prefix sum</b>, the book's running example: 3 1 7 0 4 1 6 3 scans to 3 4 11
          11 15 16 22 25.
        </p>
      </Translated>
      <Source section="11.1" pages="252" />
      <Translated
        vi={
          <p>
            Hình ảnh của sách là một khúc xúc xích dài 40 inch và tám người gọi 3, 1, 7, 0, 4, 1, 6 và 3 inch. Phục vụ lần lượt
            từng người thì mỗi vết cắt phải chờ vết trước nó. Nhưng scan bao gồm của các phần gọi cho ra mọi điểm cắt cùng lúc:
            3, 4, 11, 11, 15, 16, 22, 25 inch tính từ đầu khúc. Khi đã biết hết, các vết cắt có thể làm theo thứ tự bất kỳ,
            hoặc làm tất cả cùng lúc. Điều này đúng với mọi thứ được chia thành các khúc liền nhau: bánh mì, chỗ cắm trại, một
            khối bộ nhớ.
          </p>
        }
      >
        <p>
          The book's picture is a 40-inch sausage and eight people who order 3, 1, 7, 0, 4, 1, 6 and 3 inches. Served one
          after another, each cut waits for the one before it. But the inclusive scan of the orders gives every cut point at
          once: 3, 4, 11, 11, 15, 16, 22, 25 inches from the end. With all of them known, the cuts can be made in any order,
          or all together. The same is true of anything handed out in contiguous pieces: bread, campground space, a block
          of memory.
        </p>
      </Translated>
      <Source section="11.1" pages="252–253" />
      <Translated
        vi={
          <p>
            Một <b>scan loại trừ</b> (exclusive scan) bỏ đi đầu vào ở mỗi vị trí: nó trả về <b>phần tử đơn vị</b> của toán tử
            (0 với phép cộng, giá trị không làm thay đổi gì), rồi x₀, rồi x₀ ⊕ x₁, và cứ thế đến x<sub>N−2</sub>. Với khúc xúc
            xích, đó là 0 3 4 11 11 15 16 22: nơi mỗi khúc <em>bắt đầu</em>. Đó là thứ cấp phát bộ nhớ cần, vì bên gọi nhận lại
            một con trỏ tới đầu khối của nó.
          </p>
        }
      >
        <p>
          An <b>exclusive scan</b> leaves out the input at each position: it returns the operator's <b>identity</b> (0 for
          addition, the value that changes nothing), then x₀, then x₀ ⊕ x₁, and so on up to x<sub>N−2</sub>. For the
          sausage that is 0 3 4 11 11 15 16 22: where each piece <em>starts</em>. That is what memory allocation needs,
          since a caller gets back a pointer to the start of its block.
        </p>
      </Translated>
      <Source section="11.1" pages="253" />
      <SausageExplorer />
      <Translated
        vi={
          <p>
            Hai loại scan lệch nhau một ô. Dịch một scan bao gồm sang phải một ô và đặt phần tử đơn vị lên đầu, bạn có scan loại
            trừ. Dịch một scan loại trừ sang trái, điền ô cuối bằng giá trị loại trừ cuối cộng đầu vào cuối, và bạn quay lại
            được. Vì chuyển đổi rẻ như vậy, từ đây sách chỉ cài đặt scan bao gồm.
          </p>
        }
      >
        <p>
          The two scans differ by one place. Shift an inclusive scan right by one and put the identity first, and you have
          the exclusive scan. Shift an exclusive scan left, and fill the last slot with the last exclusive value plus the
          last input, and you are back. Because converting is this cheap, the book implements only inclusive scans from
          here on.
        </p>
      </Translated>
      <Source section="11.1" pages="253" />
      <Ask q={questionsS111['ch11/11.1/q-total']} />
      <Ask q={questionsS111['ch11/11.1/q-offsets']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Vòng lặp mà mọi lab dùng để đối chiếu">The loop every lab checks against</Translated></h2>
      <Translated
        vi={
          <p>
            Chạy tuần tự, scan bao gồm là một vòng lặp: chép đầu vào đầu tiên, rồi mỗi đầu ra là đầu ra trước cộng đầu vào kế
            tiếp. N phần tử cần N − 1 phép cộng, ít nhất mà mọi scan có thể làm. Vòng lặp này cũng là tham chiếu CPU của mọi
            lab trong chương, vì một lý do mà đoạn sau sẽ nói.
          </p>
        }
      >
        <p>
          Sequentially, an inclusive scan is one loop: copy the first input, then each output is the previous output plus
          the next input. N elements take N − 1 additions, the least any scan can do. This loop is also the CPU reference
          of every lab in the chapter, for a reason the next paragraph gives.
        </p>
      </Translated>
      <Source section="11.1" pages="253–254" figs="Fig. 11.1" />
      <Code
        tabs={[{ label: 'C++', code: seq ?? '// loading labs/ch11/snippets/s111_sequential.h' }]}
        highlight={[11, 12, 13]}
        caption={
          <Translated
            vi={
              <>
                Fig. 11.1, thêm <code>const</code> cho đầu vào và một phép kiểm tra mảng rỗng, cùng với scan loại trừ và hai phép
                chuyển đổi (labs/ch11/snippets/s111_sequential.h). Đây là C++ phía host, nên các trạng thái GPU của khóa học không
                áp dụng: các unit test biên dịch nó, chạy trên CPU và so từng số float với các trình khám phá trên trang này, từng
                bit một.
              </>
            }
          >
            <>
              Fig. 11.1, with <code>const</code> on the input and a guard for an empty array, plus the exclusive scan and
              the two conversions (labs/ch11/snippets/s111_sequential.h). This is host C++, so the course's GPU statuses do
              not apply: the unit tests compile it, run it on the CPU and compare every float with the explorers on this
              page, bit for bit.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Scan phổ biến đến mức các thư viện đều có. C++17 có <code>std::inclusive_scan</code> và{' '}
            <code>std::exclusive_scan</code>; trên GPU NVIDIA, Thrust và CUB có các bản scan được tinh chỉnh cho phần cứng.
            Sách kỳ vọng bạn dùng chúng trong thực tế, và coi scan là phương tiện để dạy tối ưu hóa.
          </p>
        }
      >
        <p>
          Scan is common enough that libraries have it. C++17 has <code>std::inclusive_scan</code> and{' '}
          <code>std::exclusive_scan</code>; on NVIDIA GPUs, Thrust and CUB have scans tuned for the hardware. The book
          expects you to use them in practice, and treats scan as a vehicle for teaching optimisation.
        </p>
      </Translated>
      <Source section="11.1" pages="254" />
      <Translated
        vi={
          <p>
            Hãy đọc tài liệu của chúng để tìm một từ: deterministic (tất định). Mọi thư viện này, và cả rocPRIM của AMD, đều nói
            rằng một scan dấu phẩy động có thể cho các bit khác nhau giữa các lần chạy, vì nó có thể cộng theo thứ tự khác. CUB
            có một chế độ cố định thứ tự cho phép cộng float, còn rocPRIM có một hàm tất định riêng. Vì vậy tham chiếu của các
            lab là vòng lặp tường minh, có thứ tự không bao giờ đổi.
          </p>
        }
      >
        <p>
          Read their documentation for one word: deterministic. Every one of these libraries, and AMD's rocPRIM, says that a
          floating-point scan may give different bits from run to run, because it may add in a different order. CUB has a
          mode that fixes the order for float addition, and rocPRIM a separate deterministic function. So the labs' reference
          is the explicit loop, whose order never changes.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch11-facts A8)', '2026-11-20 (ch11-facts A8)')}>
        <Translated
          vi={
            <p>
              cppreference: "The result is non-deterministic if the add_op is not associative (such as floating-point
              addition)." (Kết quả không tất định nếu add_op không có tính kết hợp.) CUB <code>DeviceScan</code> có{' '}
              <code>run_to_run</code> cho <code>plus</code> trên float; rocPRIM có <code>deterministic_inclusive_scan</code>.
            </p>
          }
        >
          <p>
            cppreference: "The result is non-deterministic if the add_op is not associative (such as floating-point
            addition)." CUB <code>DeviceScan</code> offers <code>run_to_run</code> for float <code>plus</code>; rocPRIM has{' '}
            <code>deterministic_inclusive_scan</code>.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS111['ch11/11.1/q-cpu-ref']} />

      <h2><Translated vi="Token vượt qua p">The token that crosses p</Translated></h2>
      <Translated
        vi={
          <p>
            Lấy mẫu top-p sắp các xác suất của token kế tiếp theo thứ tự giảm dần và giữ tập nhỏ nhất ở đầu có tổng đạt tới p.
            Scan bao gồm tại token <i>i</i> là khối xác suất tính đến và bao gồm nó; scan loại trừ là khối xác suất xếp trước
            nó. llama.cpp (trên host và trong đồ thị backend), vLLM, transformers và mlx-lm đều giữ token <i>i</i> đúng khi khối
            trước nó nhỏ hơn p. Vậy token mà xác suất của chính nó đẩy tổng vượt quá p vẫn được giữ. Viết "giữ khi tổng bao gồm
            còn dưới p" là bạn bỏ mất nó: lỗi lệch một giữa hai loại scan.
          </p>
        }
      >
        <p>
          Top-p sampling sorts the probabilities of the next token in descending order and keeps the smallest set at the
          top whose total reaches p. The inclusive scan at token <i>i</i> is the mass up to and including it; the exclusive
          scan is the mass ranked before it. llama.cpp (on the host and in its backend graph), vLLM, transformers and mlx-lm
          all keep token <i>i</i> exactly when the mass before it is below p. So the token whose own probability carries the
          total past p is kept. Write "keep while the inclusive sum is below p" and you drop it: the off-by-one between
          the two scans.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách không nối scan với lấy mẫu. Các quy tắc ở đây được đọc từ mã nguồn của các dự án ngày 2026-11-20 (ch11-facts
              D2–D4); việc chúng đều giữ cùng một tập là suy luận của chúng tôi.
            </p>
          }
        >
          <p>
            The book does not connect scan to sampling. The rules here are read from the projects' source on 2026-11-20
            (ch11-facts D2–D4); that they all keep the same set is our derivation.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Chúng đến được tập đó theo ba cách khác nhau. llama.cpp scan bao gồm trên hàng giảm dần và thêm một token sau vết
            cắt. transformers và nhánh PyTorch của vLLM sắp tăng dần và bỏ token khi tổng tăng dần còn tối đa 1 − p. mlx-lm scan
            ngược, loại trừ, so với p nhân tổng đã tính, rồi giữ mọi token đồng hạng với token ở biên. Các trường hợp đồng hạng
            là nơi chúng khác nhau: llama.cpp cắt giữa các token đồng hạng ở bất cứ đâu phép sắp đặt chúng, còn mlx-lm giữ hết.
          </p>
        }
      >
        <p>
          They reach that set three different ways. llama.cpp scans the descending row inclusively and adds one token after
          the cut. transformers and vLLM's PyTorch path sort ascending and drop tokens while the ascending sum is at most 1 −
          p. mlx-lm scans in reverse, exclusively, compares with p times the computed total, and then keeps every token tied
          with the boundary token. Ties are where they part: llama.cpp cuts between tied tokens wherever the sort put them,
          mlx-lm keeps them all.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Các xác suất được cho sẵn và đã sắp; không có bước lấy mẫu. Float32 được mô phỏng bằng cách làm tròn sau mỗi phép
              cộng. Phép so sánh trong không gian log của mlx-lm được làm trên các xác suất. 1 − p của transformers được tạo ở
              double và làm tròn sang float32 một lần, đây là giả định của chúng tôi.
            </p>
          }
        >
          <p>
            The probabilities are given and already sorted; there is no sampling step. Float32 is imitated by rounding after
            every addition. mlx-lm's comparison in log space is done on the probabilities. transformers' 1 − p is formed in
            double and rounded to float32 once, which is our assumption.
          </p>
        </Translated>
      </Simplification>
      <Predict
        id="ch11/11.1/p-crossing"
        prompt={
          <Translated
            vi={
              <>
                Các xác suất đã sắp 0.5, 0.25, 0.125, 0.125 và p = 0.75. llama.cpp giữ bao nhiêu token? Chốt một con số, rồi chọn
                hàng đó trong trình khám phá bên dưới.
              </>
            }
          >
            <>
              Sorted probabilities 0.5, 0.25, 0.125, 0.125 and p = 0.75. How many tokens does llama.cpp keep? Commit a
              number, then pick that row in the explorer below.
            </>
          </Translated>
        }
        unit={tr('tokens', 'token')}
      >
        {(v) => (
          <Translated
            vi={
              <p data-readout="crossing">
                Nó giữ <b>{crossing.kept}</b>. Các khối trước các token là {crossing.before.join(', ')}. Khối bao gồm của token 1
                đúng bằng 0.75 = p, và nó được giữ, vì khối trước nó, 0.5, nhỏ hơn p. Khối trước token 2 là 0.75, không nhỏ hơn p,
                nên nó bị bỏ. Mọi giá trị ở đây đều là lũy thừa của hai, nên float32 cho cùng câu trả lời theo mọi thứ tự, và
                transformers, vLLM, mlx-lm đều đồng ý.{' '}
                {v === crossing.kept ? 'Bạn đã đúng.' : v === 1 ? 'Bạn đã áp dụng quy tắc bao gồm, tức lỗi lệch một.' : ''}
              </p>
            }
          >
            <p data-readout="crossing">
              It keeps <b>{crossing.kept}</b>. The masses before the tokens are {crossing.before.join(', ')}. Token 1's
              inclusive mass is exactly 0.75 = p, and it is kept, because the mass before it, 0.5, is below p. Token 2's mass
              before it is 0.75, not below p, so it is dropped. Every value here is a power of two, so float32 gives the same
              answer in any order, and transformers, vLLM and mlx-lm agree.{' '}
              {v === crossing.kept ? 'You had it.' : v === 1 ? 'You applied the inclusive rule, the off-by-one.' : ''}
            </p>
          </Translated>
        )}
      </Predict>
      <NucleusExplorer />
      <Translated
        vi={
          <p>
            Bạn hẳn đã rút ra ba điều. Mọi dự án đều giữ token vượt ngưỡng còn bản lệch một thì bỏ nó; trên hàng có đỉnh nhọn với
            p nhỏ, nó thậm chí có thể không giữ gì. Các trường hợp đồng hạng quyết định biên khác đi trong mlx-lm. Và ở float32,
            trên hàng mà hai thứ tự bất đồng, khối chính xác trước token {FLIP.token} đúng bằng p ={' '}
            {FLIP.p}: tổng của vòng lặp rơi thấp hơn p một số float và giữ nó, tổng Kogge-Stone rơi đúng p và bỏ nó.
          </p>
        }
      >
        <p>
          Three things should have come out of it. Every project keeps the crossing token and the off-by-one drops it; on
          the peaked row with a small p it can even keep nothing. Ties decide the boundary differently in mlx-lm. And in
          float32, on the row where the two orders disagree, the exact mass before token {FLIP.token} is precisely p ={' '}
          {FLIP.p}: the loop's sum lands one float below p and keeps it, the Kogge-Stone sum lands on p and drops it.
        </p>
      </Translated>
      <Ask q={questionsS111['ch11/11.1/q-rule']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Hình ảnh khúc xúc xích cũng chính là phép lấy mẫu. Scan bao gồm của các xác suất cắt đoạn từ 0 đến 1 thành mỗi token
            một khúc, rộng bằng xác suất của nó. Rút một số ngẫu nhiên phân bố đều, tìm khúc mà nó rơi vào, và đó là token được
            lấy mẫu. Module X1 xây dựng phép tìm kiếm đó trong{' '}
            <a href={href('x1', 'X1.8')}>X1.8, Phép rút: CDF ngược trên một scan</a>.
          </p>
        }
      >
        <p>
          The sausage picture is also sampling itself. The inclusive scan of the probabilities cuts the interval from 0 to 1
          into one piece per token, as wide as its probability. Draw a uniform random number, find the piece it lands in,
          and that is the sampled token. Module X1 builds that search in{' '}
          <a href={href('x1', 'X1.8')}>X1.8, The draw: inverse CDF over a scan</a>.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Module X1 (các kernel lấy mẫu) là của khóa học này, không phải của sách.
            </p>
          }
        >
          <p>
            Module X1 (sampling kernels) is ours, not the book's.
          </p>
        </Translated>
      </Aside>

      <h2><Translated vi="Dấu phẩy động ở biên">Floating point at the boundary</Translated></h2>
      <Translated
        vi={
          <p>
            Một scan song song cộng cùng các xác suất theo thứ tự khác với vòng lặp, và phép cộng float không có tính kết hợp,
            nên các bit cuối của khối tích lũy khác nhau. Phần lớn thời gian điều đó không thay đổi gì. Nhưng một token có khối
            trước nó nằm trong khoảng làm tròn quanh p có thể rơi về bên nào cũng được, như trình khám phá đã cho thấy. Vì vậy
            bài kiểm thử một kernel top-p cần một dải quanh p, bên trong đó nó chấp nhận cả hai quyết định, cùng một quy tắc
            đồng hạng được nêu rõ; để kiểm thử chính quy tắc, nó dùng các hàng lũy thừa của hai, mà mọi thứ tự đều cộng chính
            xác. Lab 11.6 nêu cả hai.
          </p>
        }
      >
        <p>
          A parallel scan adds the same probabilities in a different order from the loop, and float addition is not
          associative, so the last bits of the cumulative mass differ. Most of the time that changes nothing. But a token
          whose mass before it sits within rounding distance of p can land on either side, as the explorer showed. A test
          of a top-p kernel therefore needs a band around p, inside which it accepts either decision, and a stated tie
          rule; to test the rule itself it uses rows of powers of two, which every order adds exactly. Lab 11.6 states
          both.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách không bàn đến điều này. Tính không kết hợp: ch10-facts A9. Dải trong trình khám phá là γ<sub>n−1</sub> nhân
              tổng của hàng, một cận chuẩn cho mọi thứ tự của n − 1 phép cộng.
            </p>
          }
        >
          <p>
            The book does not discuss this. Non-associativity: ch10-facts A9. The band in the explorer is γ<sub>n−1</sub>
            times the row's sum, a standard bound on any order of n − 1 additions.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS111['ch11/11.1/q-float-boundary']} />
      <Translated
        vi={
          <p>
            Tiếp theo: cách tính một scan với nhiều luồng cùng lúc, ở mục <a href={href('ch11', '11.2')}>11.2</a>.
          </p>
        }
      >
        <p>
          Next: how to compute a scan with many threads at once, in section <a href={href('ch11', '11.2')}>11.2</a>.
        </p>
      </Translated>
    </>
  );
}
