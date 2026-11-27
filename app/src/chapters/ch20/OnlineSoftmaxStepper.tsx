import { useEffect, useMemo, useState } from 'react';
import { Bench } from '../../components/notes';
import {
  FORM_LABEL,
  INIT_LABEL,
  ORDER_LABEL,
  PRESET_LABEL,
  ROW_N,
  V12,
  presetScores,
  runOnline,
  type Form,
  type InitMax,
  type PresetId,
  type TileOrder,
  type TileStep,
} from '../../sim/onlinesoftmax';
import { Translated, tr } from '../../lib/i18n';

// Section 20.5's central interactive: one query row merged tile by tile (Eqs. 20.3–20.6).
// Every number comes from src/sim/onlinesoftmax.ts.

/** A float as the stepper prints it: 4 significant digits, the two start values by name. */
export function fv(x: number | null | undefined): string {
  if (x == null) return '—';
  if (Number.isNaN(x)) return 'NaN';
  if (x === -Infinity) return '−∞';
  if (x === Infinity) return '∞';
  if (x <= -1e37) return '−FLT_MAX/2';
  if (x !== 0 && Math.abs(x) < 1e-3) return x.toExponential(1).replace('-', '−').replace('e', '×10^');
  return Number(x.toPrecision(4)).toString().replace('-', '−');
}
const fvec = (v: number[] | null) => (v ? '(' + v.map(fv).join(', ') + ')' : '—');

const WIDTHS = [1, 2, 3, 4, 6, 12];
const PRESETS: PresetId[] = ['demo', 'random', 'late', 'offset'];
const ORDERS: TileOrder[] = ['ltr', 'rtl', 'shuffled'];
const INITS: InitMax[] = ['neg-inf', 'sentinel'];
const FORMS: Form[] = ['deferred', 'normalised'];

function Select<T extends string>({ label, value, options, labels, onChange }: {
  label: string; value: T; options: T[]; labels: Record<T, string>; onChange: (v: T) => void;
}) {
  return (
    <label className="field">
      {label}
      <select value={value} onChange={(e) => onChange(e.target.value as T)} aria-label={label}>
        {options.map((o) => <option key={o} value={o}>{labels[o]}</option>)}
      </select>
    </label>
  );
}

function verdict(s: TileStep | undefined) {
  if (!s) return null;
  if (s.nan) return <span className="chip fail" data-check="nan">NaN</span>;
  if (s.diff == null) return <span className="muted"><Translated vi="chưa thấy gì">nothing seen yet</Translated></span>;
  return <span className={'chip ' + (s.diff < 1e-5 ? 'ok' : 'fail')} data-check={s.diff < 1e-5 ? 'ok' : 'off'}>{s.diff < 1e-5 ? tr('matches', 'khớp') : tr('differs', 'khác')} ({fv(s.diff)})</span>;
}

export function OnlineSoftmaxStepper() {
  const [preset, setPreset] = useState<PresetId>('demo');
  const [bc, setBc] = useState(4);
  const [row, setRow] = useState(ROW_N - 1);
  const [order, setOrder] = useState<TileOrder>('ltr');
  const [init, setInit] = useState<InitMax>('neg-inf');
  const [form, setForm] = useState<Form>('deferred');
  const [k, setK] = useState(0);

  const scores = useMemo(() => presetScores(preset), [preset]);
  const trace = useMemo(() => runOnline({ scores, V: V12, bc, row, order, init, form }), [scores, bc, row, order, init, form]);
  useEffect(() => setK(0), [preset, bc, row, order, init, form]);

  const total = trace.steps.length;
  const cur = k > 0 ? trace.steps[k - 1] : undefined;
  const mergedTiles = new Set(trace.steps.slice(0, k).map((s) => s.tile));
  const nTiles = Math.ceil(ROW_N / bc);
  const done = k >= total;
  const startM = trace.start.m;
  const formLabel = form === 'deferred' ? 'Õ' : 'o';

  return (
    <Bench
      title={tr('Online softmax, one tile at a time', 'Online softmax, từng ô một')}
      task={
        <Translated vi="Trước mỗi lần Bước, hãy nói giá trị lớn nhất đang chạy có tăng không và hệ số nhân vào D cũ sẽ là bao nhiêu.">
          Before each Step, say whether the running max will rise and what the factor on the old D will be.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Một hàng truy vấn gồm {ROW_N} điểm số (đã được co giãn) và một V cỡ {ROW_N} × 3. Các cột vượt quá chỉ số hàng bị
              che mặt nạ theo chỉ số, như trong các lab. Cột kiểm tra so ước lượng hiện tại với attention chính xác, ở độ chính
              xác double, trên các cột nhìn thấy được đã trộn đến giờ.
            </>
          }
        >
          <>
            One query row of {ROW_N} scores (already scaled) and a {ROW_N} × 3 V. Columns past the row index are masked by
            index, as the labs do. The check column compares the current estimate with exact attention, in double
            precision, over the visible columns merged so far.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          <Translated vi="Độ rộng ô B_c">Tile width B_c</Translated>
          <span className="seg" role="group" aria-label={tr('Tile width', 'Độ rộng ô')}>
            {WIDTHS.map((w) => (
              <button key={w} aria-pressed={bc === w} onClick={() => setBc(w)}>{w}</button>
            ))}
          </span>
        </label>
        <label className="field">
          <Translated vi={<>Hàng nhân quả r = {row}</>}>Causal row r = {row}</Translated>
          <input type="range" min={0} max={ROW_N - 1} value={row} aria-label={tr('Causal row', 'Hàng nhân quả')} onChange={(e) => setRow(+e.target.value)} />
        </label>
        <Select label={tr('Data', 'Dữ liệu')} value={preset} options={PRESETS} labels={PRESET_LABEL} onChange={setPreset} />
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <Select label={tr('Tile order', 'Thứ tự ô')} value={order} options={ORDERS} labels={ORDER_LABEL} onChange={setOrder} />
        <Select label={tr('Running max starts at', 'Giá trị lớn nhất đang chạy bắt đầu từ')} value={init} options={INITS} labels={INIT_LABEL} onChange={setInit} />
        <Select label={tr('Form', 'Dạng')} value={form} options={FORMS} labels={FORM_LABEL} onChange={setForm} />
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setK(0)} disabled={k === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => setK(k - 1)} disabled={k === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => setK(k + 1)} disabled={done}><Translated vi="Bước">Step</Translated></button>
        <span className="muted" aria-live="polite" data-progress><Translated vi={<>đã trộn {k} / {total} ô</>}>tile {k} of {total} merged</Translated></span>
      </div>

      <div className="mx-label">
        <Translated vi={<>Hàng (các ô gồm {bc} phần tử, trộn theo thứ tự {trace.steps.map((s) => s.tile).join(', ')})</>}>
          The row (tiles of {bc}, merged in the order {trace.steps.map((s) => s.tile).join(', ')})
        </Translated>
      </div>
      <div className="mx" data-row style={{ display: 'flex', flexWrap: 'wrap', marginBottom: '0.3rem' }}>
        {scores.map((s, j) => {
          const t = Math.floor(j / bc);
          const vis = j <= row;
          const isCur = cur?.tile === t;
          const merged = mergedTiles.has(t);
          return (
            <span
              key={j}
              className={vis ? '' : 'void'}
              data-col={j}
              data-state={isCur ? 'current' : merged ? 'merged' : 'waiting'}
              title={vis ? tr(`column ${j}: ${s}`, `cột ${j}: ${s}`) : tr(`column ${j}: masked (past row ${row})`, `cột ${j}: bị che (vượt quá hàng ${row})`)}
              style={{
                background: isCur ? 'var(--ink)' : merged && vis ? 'var(--rule-soft)' : undefined,
                color: isCur && vis ? 'var(--page)' : undefined,
                borderLeft: j % bc === 0 ? '3px solid var(--ink-2)' : undefined,
                width: '2.9rem',
              }}
            >
              {vis ? fv(s) : '×'}
            </span>
          );
        })}
      </div>
      <Translated
        vi={
          <p className="muted" style={{ margin: '0 0 0.8rem' }}>
            × đánh dấu một cột bị che. Cạnh trái đậm đánh dấu ranh giới ô; ô được tô sáng là ô vừa trộn, các ô màu xám đã được trộn trước đó.
            {preset === 'offset' ? <> e<sup>điểm số lớn nhất</sup> trong float32 là {fv(trace.naiveExpOfMax)}: nếu không trừ đi giá trị lớn nhất, các hàm mũ bị tràn số.</> : null}
          </p>
        }
      >
        <p className="muted" style={{ margin: '0 0 0.8rem' }}>
          × marks a masked column. Thick left edges mark tile boundaries; the highlighted tile is the one just merged, greyed ones were merged before.
          {preset === 'offset' ? <> e<sup>largest score</sup> in float32 is {fv(trace.naiveExpOfMax)}: without subtracting the max, the exponentials overflow.</> : null}
        </p>
      </Translated>

      <div className={'narrate' + (cur?.nan ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }} data-narrate>
        {!cur ? (
          <>
            <Translated vi={<>Chưa trộn gì: m = {fv(startM)}, D = 0, {formLabel} = (0, 0, 0). Hàng có {nTiles} ô. Bấm Bước.</>}>
              Nothing merged yet: m = {fv(startM)}, D = 0, {formLabel} = (0, 0, 0). The row has {nTiles} tile{nTiles > 1 ? 's' : ''}. Press Step.
            </Translated>
          </>
        ) : cur.skipped ? (
          <>
            <Translated vi={<>Ô {cur.tile} không có cột nào nhìn thấy được. Eq. 20.5 sẽ là 0/0 với nó, nên dạng chuẩn hóa bỏ qua nó; không có gì thay đổi.</>}>
              Tile {cur.tile} has no visible column. Eq. 20.5 would be 0/0 for it, so the normalised form skips it; nothing
              changes.
            </Translated>
          </>
        ) : (
          <Translated
            vi={
              <>
                Ô {cur.tile} (các cột {cur.cols[0]}–{cur.cols[cur.cols.length - 1]}
                {cur.visible.length < cur.cols.length ? `, ${cur.visible.length ? cur.visible.length + ' cột nhìn thấy' : 'không cột nào nhìn thấy'}` : ''}):{' '}
                {cur.visible.length ? <>giá trị lớn nhất riêng của nó m<sub>B</sub> = {fv(cur.mB)}. </> : <>không có cột nhìn thấy được, nên giá trị lớn nhất của nó là giá trị khởi đầu {fv(cur.mB)}. </>}
                {cur.rose ? (
                  <>Giá trị lớn nhất đang chạy tăng từ {fv(cur.mOld)} lên {fv(cur.mNew)}, nên mọi thứ giữ đến giờ co lại theo e<sup>{fv(cur.mOld)} − {fv(cur.mNew)}</sup> = {fv(cur.aFactor)}.</>
                ) : Number.isNaN(cur.aFactor) ? (
                  <>Giá trị lớn nhất đang chạy vẫn là {fv(cur.mOld)}, và hệ số co giãn e<sup>{fv(cur.mOld)} − ({fv(cur.mNew)})</sup> là NaN: (−∞) − (−∞) không có giá trị, và {formLabel} trở thành NaN vĩnh viễn.</>
                ) : (
                  <>Giá trị lớn nhất đang chạy vẫn là {fv(cur.mNew)}: hệ số nhân vào phần đã giữ là {fv(cur.aFactor)}, và các số hạng của ô được cộng vào.</>
                )}
              </>
            }
          >
              Tile {cur.tile} (columns {cur.cols[0]}–{cur.cols[cur.cols.length - 1]}
              {cur.visible.length < cur.cols.length ? `, ${cur.visible.length ? cur.visible.length + ' visible' : 'none visible'}` : ''}):{' '}
              {cur.visible.length ? <>its own max m<sub>B</sub> = {fv(cur.mB)}. </> : <>no visible column, so its max is the start value {fv(cur.mB)}. </>}
              {cur.rose ? (
                <>The running max rises from {fv(cur.mOld)} to {fv(cur.mNew)}, so everything kept so far shrinks by e<sup>{fv(cur.mOld)} − {fv(cur.mNew)}</sup> = {fv(cur.aFactor)}.</>
              ) : Number.isNaN(cur.aFactor) ? (
                <>The running max stays {fv(cur.mOld)}, and the rescale factor e<sup>{fv(cur.mOld)} − ({fv(cur.mNew)})</sup> is NaN: (−∞) − (−∞) has no value, and {formLabel} becomes NaN for good.</>
              ) : (
                <>The running max stays {fv(cur.mNew)}: the factor on what is kept is {fv(cur.aFactor)}, and the tile's terms add on.</>
              )}
          </Translated>
        )}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 5fr) minmax(0, 7fr)' }}>
        <table className="data" data-detail style={{ margin: 0, background: 'var(--page)' }}>
          <caption className="muted" style={{ textAlign: 'left', captionSide: 'top', padding: '0 0 0.3rem' }}>
            <Translated vi="Bước này, theo các ký hiệu của Eq. 20.4 (A = phần đã trộn đến giờ, B = ô này)">
              This step, in the terms of Eq. 20.4 (A = merged so far, B = this tile)
            </Translated>
          </caption>
          <tbody>
            <tr><td><Translated vi={<>m<sub>A</sub>, giá trị lớn nhất đang chạy trước đó</>}>m<sub>A</sub>, the running max before</Translated></td><td data-v="mA">{fv(cur?.mOld)}</td></tr>
            <tr><td><Translated vi={<>m<sub>B</sub>, giá trị lớn nhất của ô</>}>m<sub>B</sub>, the tile's max</Translated></td><td data-v="mB">{fv(cur?.mB)}</td></tr>
            <tr><td>m<sub>A∪B</sub> = max(m<sub>A</sub>, m<sub>B</sub>)</td><td data-v="mAB">{fv(cur?.mNew)}</td></tr>
            <tr><td><Translated vi={<>Hệ số của số hạng A e<sup>m<sub>A</sub> − m<sub>A∪B</sub></sup> (nhân vào D<sub>A</sub> và {formLabel})</>}>A term factor e<sup>m<sub>A</sub> − m<sub>A∪B</sub></sup> (on D<sub>A</sub> and {formLabel})</Translated></td><td data-v="aFactor"><b>{fv(cur?.aFactor)}</b></td></tr>
            <tr><td><Translated vi={<>Hệ số của số hạng B e<sup>m<sub>B</sub> − m<sub>A∪B</sub></sup> (nhân vào D<sub>B</sub>)</>}>B term factor e<sup>m<sub>B</sub> − m<sub>A∪B</sub></sup> (on D<sub>B</sub>)</Translated></td><td data-v="bFactor">{cur && cur.bFactor == null ? tr('no visible column', 'không có cột nhìn thấy được') : fv(cur?.bFactor)}</td></tr>
            <tr><td><Translated vi={<>D<sub>B</sub>, Eq. 20.3 trên ô</>}>D<sub>B</sub>, Eq. 20.3 over the tile</Translated></td><td data-v="DB">{fv(cur?.DB)}</td></tr>
            <tr><td><Translated vi={<>Số hạng B theo cách mã cộng nó: Σ e<sup>l − m<sub>A∪B</sub></sup></>}>B term as the code adds it: Σ e<sup>l − m<sub>A∪B</sub></sup></Translated></td><td data-v="bTerm">{fv(cur?.bTerm)}</td></tr>
            <tr><td><Translated vi="D sau khi trộn">D after the merge</Translated></td><td data-v="D"><b>{fv(cur ? cur.D : 0)}</b></td></tr>
            <tr><td><Translated vi={<>{formLabel} sau khi trộn</>}>{formLabel} after the merge</Translated></td><td data-v="O">{fvec(cur ? cur.O : [0, 0, 0])}</td></tr>
          </tbody>
        </table>
        <div style={{ minWidth: 0, overflowX: 'auto' }}>
          <table className="data" data-history style={{ margin: 0, background: 'var(--page)' }}>
            <thead>
              <tr>
                <th><Translated vi="bước">step</Translated></th>
                <th><Translated vi="ô">tile</Translated></th>
                <th>m</th>
                <th>D</th>
                <th><Translated vi={<>ước lượng {form === 'deferred' ? 'Õ/D' : 'o'}</>}>estimate {form === 'deferred' ? 'Õ/D' : 'o'}</Translated></th>
                <th><Translated vi="chính xác đến giờ">exact so far</Translated></th>
                <th><Translated vi="kiểm tra">check</Translated></th>
              </tr>
            </thead>
            <tbody>
              {trace.steps.slice(0, k).map((s) => (
                <tr key={s.k} data-step={s.k}>
                  <td>{s.k}</td>
                  <td>{s.tile}</td>
                  <td>{fv(s.mNew)}</td>
                  <td>{fv(s.D)}</td>
                  <td>{fvec(s.out)}</td>
                  <td>{fvec(s.ref)}</td>
                  <td>{verdict(s)}</td>
                </tr>
              ))}
              {k === 0 ? <tr><td colSpan={7} className="muted"><Translated vi="Chưa trộn ô nào.">No tile merged yet.</Translated></td></tr> : null}
            </tbody>
          </table>
          {done ? (
            <p style={{ margin: '0.7rem 0 0' }} data-final>
              {trace.nan ? (
                <span className="chip fail"><Translated vi="Kết quả là NaN.">The result is NaN.</Translated></span>
              ) : trace.diff < 1e-5 ? (
                <span className="chip ok"><Translated vi={<>Bằng attention chính xác trên các cột 0–{row}: chênh lệch float32 {fv(trace.diff)}.</>}>Equals exact attention over columns 0–{row}: float32 difference {fv(trace.diff)}.</Translated></span>
              ) : (
                <span className="chip fail"><Translated vi={<>Khác attention chính xác {fv(trace.diff)}.</>}>Differs from exact attention by {fv(trace.diff)}.</Translated></span>
              )}
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}
