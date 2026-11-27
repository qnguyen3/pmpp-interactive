import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { fmt } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  BEFORE_AFTER,
  LAB116_RTOL,
  MINP_ROW,
  THRESHOLD_Q,
  VOCAB_PRESETS,
  filterBeforeSort,
  minPTestsF32,
  minPThreshold,
  minPView,
  plantedRow,
  rowView,
  type Placement,
} from '../../sim/nucleus';

// Section X1.6: min-p as a filter. The explorers draw what src/sim/nucleus.ts computes
// (minPView, minPTestsF32, filterBeforeSort), pinned in nucleus.test.ts.

const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const n0 = (v: number) => v.toLocaleString('en-US');
const f3 = (v: number) => (isFinite(v) ? v.toFixed(3) : '−∞');
const MINPS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5];

// ---------------------------------------------------------------- before or after temperature
function TempExplorer() {
  const [minP, setMinP] = useState(THRESHOLD_Q.minP);
  const [T, setT] = useState(1);
  const [placement, setPlacement] = useState<Placement>('before');
  const [p, setP] = useState(0.95);
  const [plant, setPlant] = useState(false);
  const planted = useMemo(() => plantedRow(minP), [minP]);
  const row = plant ? planted.logits : MINP_ROW;
  const v = minPView(row, minP, T, p);
  const f32 = plant ? minPTestsF32(planted.logits, minP) : null;
  const thr = placement === 'before' ? v.thrBefore : v.thrAfter;
  const W = 600;
  const H = 210;
  const X0 = 44;
  const lo = -3;
  const hi = 5.5;
  const sy = (x: number) => H - 26 - ((Math.max(lo, Math.min(hi, x)) - lo) / (hi - lo)) * (H - 46);
  const bw = (W - X0 - 10) / row.length;
  return (
    <Bench
      title={tr('Min-p before or after temperature', 'Min-p trước hay sau temperature')}
      task={tr('Leave min-p at 0.05 and move T both ways, under each placement. Then plant a logit at the threshold.', 'Giữ min-p ở 0.05 và kéo T theo cả hai chiều, với từng vị trí. Rồi cài một logit đúng tại ngưỡng.')}
      foot={
        <Translated
          vi={
            <>
              Các thanh là logits của hàng, lớn nhất trước; thanh tô đậm là token sống sót qua min-p ở vị trí bạn chọn. Cả hai
              ngưỡng được vẽ theo đơn vị của các logit chưa chia tỉ lệ này: trước temperature là l_max + ln p_base, sau temperature là
              l_max + T ln p_base. Một chấm đánh dấu các token top-p giữ ở temperature này.
            </>
          }
        >
        <>
          Bars are the row's logits, largest first; filled bars survive min-p under the placement you chose. Both
          thresholds are drawn in the units of these unscaled logits: before temperature l_max + ln p_base, after
          temperature l_max + T ln p_base. A dot marks the tokens top-p keeps at this temperature.
        </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="min-p (p_base)">min-p (p_base)</Translated>
          <span className="seg" role="group" aria-label={tr('min-p', 'min-p')}>
            {MINPS.map((m) => (
              <button key={m} aria-pressed={minP === m} onClick={() => setMinP(m)}>{m}</button>
            ))}
          </span>
        </span>
        <label className="field" style={{ flex: '1 1 12rem' }}>
          <Translated vi={<>temperature T = {T.toFixed(2)}</>}>
            temperature T = {T.toFixed(2)}
          </Translated>
          <input type="range" min={10} max={200} step={5} value={Math.round(T * 100)} aria-label={tr('temperature', 'temperature')} onChange={(e) => setT(Number(e.target.value) / 100)} style={{ width: '100%' }} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Min-p chạy">Min-p runs</Translated>
          <span className="seg" role="group" aria-label={tr('Placement', 'Vị trí')}>
            <button aria-pressed={placement === 'before'} onClick={() => setPlacement('before')}><Translated vi="trước temperature (llama.cpp, mlx-lm)">before temperature (llama.cpp, mlx-lm)</Translated></button>
            <button aria-pressed={placement === 'after'} onClick={() => setPlacement('after')}><Translated vi="sau (bài báo, transformers)">after (the paper, transformers)</Translated></button>
          </span>
        </span>
        <label className="field" style={{ flex: '1 1 8rem' }}>
          top-p p = {p.toFixed(2)}
          <input type="range" min={50} max={100} step={1} value={Math.round(p * 100)} aria-label={tr('top-p', 'top-p')} onChange={(e) => setP(Number(e.target.value) / 100)} style={{ width: '100%' }} />
        </label>
        <label className="check">
          <input type="checkbox" checked={plant} onChange={(e) => setPlant(e.target.checked)} /> <Translated vi="cài một logit đúng tại ngưỡng float">plant a logit at the float threshold</Translated>
        </label>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr("The row's logits and min-p's thresholds", 'Logits của hàng và các ngưỡng của min-p')} data-part="bars"
          style={{ width: '100%', maxWidth: W, minWidth: 460, display: 'block', background: 'var(--page)', border: '1px solid var(--rule)', borderRadius: 4 }}>
          {[-2, 0, 2, 4].map((g) => (
            <g key={g}>
              <line x1={X0} y1={sy(g)} x2={W - 10} y2={sy(g)} stroke="var(--rule-soft)" />
              <text x={X0 - 6} y={sy(g) + 4} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{g}</text>
            </g>
          ))}
          {v.tokens.map((t) => {
            const keep = placement === 'before' ? t.before : t.after;
            const x = X0 + t.index * bw + 2;
            const isPlanted = plant && t.index === planted.planted;
            return (
              <g key={t.index} data-token={t.index} data-keep={keep ? '1' : '0'} data-topp={t.topP ? '1' : '0'}>
                <rect x={x} y={Math.min(sy(t.logit), sy(0))} width={bw - 4} height={Math.abs(sy(t.logit) - sy(0))}
                  fill={keep ? 'var(--ink-3)' : 'var(--cell)'} fillOpacity={keep ? 0.45 : 1}
                  stroke={isPlanted ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth={isPlanted ? 2 : 1} strokeDasharray={keep ? undefined : '3 2'} />
                {t.topP ? <circle cx={x + (bw - 4) / 2} cy={20} r={3} fill="var(--ink)" /> : null}
                <text x={x + (bw - 4) / 2} y={H - 10} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{t.logit.toFixed(1)}</text>
              </g>
            );
          })}
          <line x1={X0} y1={sy(v.thrBefore)} x2={W - 10} y2={sy(v.thrBefore)} stroke="var(--ink)"
            strokeWidth={placement === 'before' ? 2 : 1} strokeDasharray={placement === 'before' ? undefined : '5 4'} opacity={placement === 'before' ? 1 : 0.5} />
          <line x1={X0} y1={sy(v.thrAfter)} x2={W - 10} y2={sy(v.thrAfter)} stroke="var(--ink)"
            strokeWidth={placement === 'after' ? 2 : 1} strokeDasharray={placement === 'after' ? undefined : '5 4'} opacity={placement === 'after' ? 1 : 0.5} />
          {/* The higher threshold is labelled above its line, the lower one below, so the labels never overlap. */}
          <text x={W - 12} y={v.thrBefore >= v.thrAfter ? sy(v.thrBefore) - 4 : sy(v.thrBefore) + 12} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink)' }}><Translated vi={<>trước: {f3(v.thrBefore)}</>}>before: {f3(v.thrBefore)}</Translated></text>
          <text x={W - 12} y={v.thrBefore >= v.thrAfter ? sy(v.thrAfter) + 12 : sy(v.thrAfter) - 4} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink)' }}><Translated vi={<>sau: {f3(v.thrAfter)}</>}>after: {f3(v.thrAfter)}</Translated></text>
          <text x={X0} y={12} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>● được top-p giữ ở T = {T.toFixed(2)}</>}>● kept by top-p at T = {T.toFixed(2)}</Translated></text>
        </svg>
      </div>
      <table className="data" style={{ margin: '0.8rem 0 0', background: 'var(--page)' }} data-part="kept">
        <tbody>
          <tr data-row="before" data-kept={v.keptBefore}>
            <td style={{ paddingLeft: '0.6rem' }}><Translated vi="min-p trước temperature (llama.cpp, mlx-lm)">min-p before temperature (llama.cpp, mlx-lm)</Translated></td>
            <td className="num"><Translated vi={<>ngưỡng {f3(v.thrBefore)}</>}>threshold {f3(v.thrBefore)}</Translated></td>
            <td className="num"><Translated vi={<><b>{v.keptBefore}</b> được giữ</>}><b>{v.keptBefore}</b> kept</Translated></td>
          </tr>
          <tr data-row="after" data-kept={v.keptAfter}>
            <td style={{ paddingLeft: '0.6rem' }}><Translated vi="min-p sau temperature (bài báo, transformers)">min-p after temperature (the paper, transformers)</Translated></td>
            <td className="num"><Translated vi={<>ngưỡng {f3(v.thrAfter)} ({f3(v.thrAfter / T)} trên các logit đã chia tỉ lệ)</>}>threshold {f3(v.thrAfter)} ({f3(v.thrAfter / T)} on the scaled logits)</Translated></td>
            <td className="num"><Translated vi={<><b>{v.keptAfter}</b> được giữ</>}><b>{v.keptAfter}</b> kept</Translated></td>
          </tr>
          <tr data-row="topp" data-kept={v.keptTopP}>
            <td style={{ paddingLeft: '0.6rem' }}><Translated vi={<>top-p {p.toFixed(2)} ở T = {T.toFixed(2)}, để so sánh</>}>top-p {p.toFixed(2)} at T = {T.toFixed(2)}, for comparison</Translated></td>
            <td />
            <td className="num"><Translated vi={<><b>{v.keptTopP}</b> được giữ</>}><b>{v.keptTopP}</b> kept</Translated></td>
          </tr>
        </tbody>
      </table>
      <p style={{ margin: '0.7rem 0 0' }} data-readout="placement" data-threshold={thr}>
        <Translated
          vi={
            <>
              {T < 1 && v.keptAfter < v.keptBefore
                ? `Ở T = ${T.toFixed(2)}, min-p sau temperature chặt hơn: ${v.keptAfter} token so với ${v.keptBefore}.`
                : T > 1 && v.keptAfter > v.keptBefore
                  ? `Ở T = ${T.toFixed(2)}, min-p sau temperature lỏng hơn: ${v.keptAfter} token so với ${v.keptBefore}.`
                  : `Ở T = ${T.toFixed(2)}, cả hai vị trí đều giữ ${v.keptBefore}.`}{' '}
              Trước temperature, ngưỡng hoàn toàn không dịch chuyển theo T.
            </>
          }
        >
        {T < 1 && v.keptAfter < v.keptBefore
          ? `At T = ${T.toFixed(2)} min-p after temperature is stricter: ${v.keptAfter} tokens against ${v.keptBefore}.`
          : T > 1 && v.keptAfter > v.keptBefore
            ? `At T = ${T.toFixed(2)} min-p after temperature is looser: ${v.keptAfter} tokens against ${v.keptBefore}.`
            : `At T = ${T.toFixed(2)} both placements keep ${v.keptBefore}.`}{' '}
        Before temperature, the threshold does not move with T at all.
        </Translated>
      </p>
      {f32 ? (
        <p style={{ margin: '0.5rem 0 0' }} data-readout="planted" data-logit={f32.logit[planted.planted] ? '1' : '0'} data-prob={f32.prob[planted.planted] ? '1' : '0'}>
          <Translated
            vi={
              <>
                Cài tại T = 1: logit {planted.logits[planted.planted].toPrecision(9)}, đúng bằng ngưỡng float của llama.cpp{' '}
                <code>max + logf(p)</code>. Phép kiểm tra trên logit {f32.logit[planted.planted] ? 'giữ' : 'loại'} nó; phép kiểm tra
                trên xác suất tính bằng float32 {f32.prob[planted.planted] ? 'giữ' : 'loại'} nó
                {f32.logit[planted.planted] !== f32.prob[planted.planted] ? <b>: hai phép kiểm tra lệch nhau một lần làm tròn</b> : ', nên ở đây chúng khớp nhau'}.
              </>
            }
          >
          Planted at T = 1: logit {planted.logits[planted.planted].toPrecision(9)}, exactly llama.cpp's float threshold{' '}
          <code>max + logf(p)</code>. The logit test {f32.logit[planted.planted] ? 'keeps' : 'drops'} it; the probability test
          computed in float32 {f32.prob[planted.planted] ? 'keeps' : 'drops'} it
          {f32.logit[planted.planted] !== f32.prob[planted.planted] ? <b>: the two tests disagree by one rounding</b> : ', so they agree here'}.
          </Translated>
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- filter before you sort
function FilterExplorer() {
  const [V, setV] = useState(151936);
  const [s, setS] = useState(1);
  const [seed, setSeed] = useState(1);
  const [minP, setMinP] = useState(0.05);
  const [p, setP] = useState(0.95);
  const row = useMemo(() => rowView(V, s, seed), [V, s, seed]);
  const f = filterBeforeSort(row, p, minP);
  const ids = (x: Uint32Array) => {
    const head = Array.from(x.slice(0, 8)).join(', ');
    return x.length > 8 ? `${head}, … (${n0(x.length)})` : head;
  };
  const routes = [
    { id: 'A', label: tr('A. top-p then min-p over the whole sorted row (llama.cpp\'s order)', 'A. top-p rồi min-p trên cả hàng đã sắp xếp (thứ tự của llama.cpp)'), sorted: V, kept: f.routeA },
    { id: 'B', label: tr('B. min-p as a compaction, sort only the survivors, top-p with full-row probabilities', 'B. min-p như một phép nén gọn (compaction), chỉ sắp xếp các token sống sót, top-p với xác suất của cả hàng'), sorted: f.survivors, kept: f.routeB },
    { id: 'C', label: tr('C. as B, but top-p renormalises over the survivors', 'C. như B, nhưng top-p chuẩn hóa lại trên các token sống sót'), sorted: f.survivors, kept: f.routeC },
  ];
  return (
    <Bench
      title={tr('Filter before you sort', 'Lọc trước khi sắp xếp')}
      task={tr('Check that A and B agree, then see how few values B sorts. Set min-p to 0 and to a flat row; compare C.', 'Kiểm tra rằng A và B khớp nhau, rồi xem B sắp xếp ít giá trị đến mức nào. Đặt min-p về 0 và chọn một hàng phẳng; so sánh C.')}
      foot={
        <Translated
          vi={
            <>
              A và B giữ token theo thứ tự lớn nhất trước, hòa thì theo id token. Xác suất trong A và B là của cả hàng: cùng giá trị
              lớn nhất của hàng và cùng mẫu số softmax. C thì chia cho tổng của các token sống sót.
            </>
          }
        >
        <>
          A and B keep tokens largest first, ties by token id. Probabilities in A and B are the full row's: the same row max
          and the same softmax denominator. C divides by the survivors' sum instead.
        </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Bộ từ vựng">Vocabulary</Translated>
          <span className="seg" role="group" aria-label={tr('Vocabulary', 'Bộ từ vựng')}>
            {VOCAB_PRESETS.map((x) => (
              <button key={x} aria-pressed={V === x} onClick={() => setV(x)}>{n0(x)}</button>
            ))}
          </span>
        </span>
        <label className="field" style={{ flex: '1 1 12rem' }}>
          <Translated vi={<>hình dạng s = {s.toFixed(2)} ({s < 0.8 ? 'phẳng' : s < 1.4 ? 'ở giữa' : 'nhọn'})</>}>
            shape s = {s.toFixed(2)} ({s < 0.8 ? 'flat' : s < 1.4 ? 'in between' : 'peaked'})
          </Translated>
          <input type="range" min={0} max={3} step={0.05} value={s} aria-label={tr('shape', 'hình dạng')} onChange={(e) => setS(Number(e.target.value))} style={{ width: '100%' }} />
        </label>
        <button onClick={() => setSeed(seed + 1)}>{tr('another row', 'hàng khác')}</button>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="min-p">min-p</Translated>
          <span className="seg" role="group" aria-label={tr('min-p', 'min-p')}>
            {[0, ...MINPS.slice(0, 5)].map((m) => (
              <button key={m} aria-pressed={minP === m} onClick={() => setMinP(m)}>{m === 0 ? tr('off', 'tắt') : m}</button>
            ))}
          </span>
        </span>
        <label className="field" style={{ flex: '1 1 12rem' }}>
          top-p p = {p.toFixed(2)}
          <input type="range" min={50} max={100} step={1} value={Math.round(p * 100)} aria-label={tr('top-p', 'top-p')} onChange={(e) => setP(Number(e.target.value) / 100)} style={{ width: '100%' }} />
        </label>
      </div>
      <p style={{ marginTop: 0 }} data-readout="survivors" data-survivors={f.survivors}>
        <Translated
          vi={
            <>
              Min-p {minP === 0 ? 'tắt' : minP} để lại <b>{n0(f.survivors)}</b> trên {n0(V)} token
              {minP > 0 ? <> (ngưỡng {f3(f.threshold)}, giá trị lớn nhất của hàng {f3(row.lmax)} cộng ln {minP})</> : null}. Đường B sắp xếp{' '}
              {n0(f.survivors)} giá trị thay vì {n0(V)}
              {f.survivors < V ? `: ít hơn ${fmt(V / f.survivors)} lần` : ': không tiết kiệm được gì'}.
            </>
          }
        >
        Min-p {minP === 0 ? 'off' : minP} leaves <b>{n0(f.survivors)}</b> of {n0(V)} tokens
        {minP > 0 ? <> (threshold {f3(f.threshold)}, the row max {f3(row.lmax)} plus ln {minP})</> : null}. Route B sorts{' '}
        {n0(f.survivors)} values instead of {n0(V)}
        {f.survivors < V ? `: ${fmt(V / f.survivors)} times fewer` : ': no saving at all'}.
        </Translated>
      </p>
      <div style={{ height: 12, background: 'var(--cell)', border: '1px solid var(--rule)', borderRadius: 2, marginBottom: '0.8rem' }} aria-hidden="true">
        <div style={{ height: '100%', width: `${Math.max(0.4, (f.survivors / V) * 100)}%`, background: 'var(--global)' }} />
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0, background: 'var(--page)', minWidth: '38rem' }} data-part="routes">
          <thead>
            <tr><th style={{ paddingLeft: '0.6rem' }}><Translated vi="Đường">Route</Translated></th><th className="num"><Translated vi="Số giá trị được sắp xếp">Values sorted</Translated></th><th className="num"><Translated vi="Được giữ">Kept</Translated></th><th><Translated vi="Id token được giữ, lớn nhất trước">Token ids kept, largest first</Translated></th></tr>
          </thead>
          <tbody>
            {routes.map((r) => (
              <tr key={r.id} data-route={r.id} data-kept={r.kept.length} data-ids={Array.from(r.kept.slice(0, 50)).join(',')}>
                <td style={{ paddingLeft: '0.6rem', whiteSpace: 'normal' }}>{r.label}</td>
                <td className="num">{n0(r.sorted)}</td>
                <td className="num"><b>{n0(r.kept.length)}</b></td>
                <td className="mono" style={{ fontSize: '0.75rem' }}>{ids(r.kept)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ margin: '0.7rem 0 0' }} data-readout="verdict" data-identical={f.identical ? '1' : '0'} data-c-differs={f.cDiffers ? '1' : '0'}>
        <Translated
          vi={
            <>
              {f.identical ? <span className="chip ok">A và B giữ các tập giống hệt nhau</span> : <span className="chip fail">A và B khác nhau</span>}{' '}
              tiền tố của top-p a = {n0(f.a)}, tiền tố của min-p b = {n0(f.b)}, được giữ = min(a, b) = {n0(f.routeA.length)}.{' '}
              {f.cDiffers
                ? `C giữ ${n0(f.routeC.length)}: chuẩn hóa lại trên các token sống sót làm mọi xác suất phình lên, nên top-p dừng sớm hơn.`
                : 'Ở đây C khớp.'}
            </>
          }
        >
        {f.identical ? <span className="chip ok">A and B keep identical sets</span> : <span className="chip fail">A and B differ</span>}{' '}
        top-p's prefix a = {n0(f.a)}, min-p's prefix b = {n0(f.b)}, kept = min(a, b) = {n0(f.routeA.length)}.{' '}
        {f.cDiffers
          ? `C keeps ${n0(f.routeC.length)}: renormalising over the survivors inflates every probability, so top-p stops sooner.`
          : 'C agrees here.'}
        </Translated>
      </p>
      <p className="muted" style={{ margin: '0.4rem 0 0' }} data-readout="band" data-band={f.band}>
        <Translated
          vi={
            <>
              {n0(f.band)} token có quyết định top-p nằm trong dải của lab 11.6 (khối xác suất trước nó cách p trong phạm vi{' '}
              {LAB116_RTOL}, tương đối): một kernel float32 có thể quyết định theo hướng nào cũng được.{' '}
              {f.band === 0
                ? ''
                : f.bandInMinP === 0
                  ? 'Không token nào trong số đó nằm trong tiền tố của min-p, nên ở đây chúng không thể thay đổi tập được giữ.'
                  : `${n0(f.bandInMinP)} token trong số đó nằm trong tiền tố của min-p, nơi quyết định có thể thay đổi tập được giữ.`}
            </>
          }
        >
        {n0(f.band)} {f.band === 1 ? 'token has' : 'tokens have'} a top-p decision inside lab 11.6's band (mass before within{' '}
        {LAB116_RTOL} of p, relative): a float32 kernel may decide {f.band === 1 ? 'it' : 'them'} either way.{' '}
        {f.band === 0
          ? ''
          : f.bandInMinP === 0
            ? 'None of them is inside min-p\'s prefix, so here they cannot change what is kept.'
            : `${n0(f.bandInMinP)} of them ${f.bandInMinP === 1 ? 'is' : 'are'} inside min-p's prefix, where the decision can change what is kept.`}
        </Translated>
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const BA = minPView(BEFORE_AFTER.logits, BEFORE_AFTER.minP, BEFORE_AFTER.T, 1);

export const questionsSX16: Record<string, Question> = {
  'x1/X1.6/q-before-after': {
    kind: 'numeric',
    id: 'x1/X1.6/q-before-after',
    prompt: (
      <Translated
        vi={
          <>
            Logits [3.0, 2.0, 1.0], min-p 0.1, T = 0.5. llama.cpp (min-p trước temperature) giữ cả {BA.keptBefore}. transformers
            (min-p sau temperature) giữ bao nhiêu?
          </>
        }
      >
      <>
        Logits [3.0, 2.0, 1.0], min-p 0.1, T = 0.5. llama.cpp (min-p before temperature) keeps all {BA.keptBefore}. How many
        does transformers (min-p after temperature) keep?
      </>
      </Translated>
    ),
    answer: BA.keptAfter,
    unit: 'tokens',
    explain: (
      <Translated
        vi={
          <p>
            Sau temperature, các logit là [6, 4, 2] và ngưỡng là 6 + ln 0.1 = {f3(BA.thrAfter / BEFORE_AFTER.T)}: 6 và 4 sống sót,
            2 thì không. Trước temperature, ngưỡng là 3 + ln 0.1 = {f3(BA.thrBefore)} và cả ba đều sống sót. Temperature thấp làm
            min-p sau temperature chặt hơn.
          </p>
        }
      >
      <p>
        After temperature the logits are [6, 4, 2] and the threshold is 6 + ln 0.1 = {f3(BA.thrAfter / BEFORE_AFTER.T)}: 6 and
        4 survive, 2 does not. Before temperature the threshold is 3 + ln 0.1 = {f3(BA.thrBefore)} and all three survive. A
        low temperature makes after-temperature min-p stricter.
      </p>
      </Translated>
    ),
    source: 'notes/today/x1-facts.md D1, D4, D9', // i18n-ignore: citation
    vi: { unit: 'token' },
  },
  'x1/X1.6/q-prefix': {
    kind: 'choice',
    id: 'x1/X1.6/q-prefix',
    prompt: (
      <Translated
        vi={
          <>
            Trên một hàng đã sắp xếp lớn nhất trước, top-p giữ một tiền tố dài a và min-p một tiền tố dài b. Theo thứ tự của
            llama.cpp (top-p, rồi min-p), bao nhiêu token sống sót qua cả hai?
          </>
        }
      >
      <>
        On a row sorted largest first, top-p keeps a prefix of length a and min-p a prefix of length b. In llama.cpp's
        order (top-p, then min-p), how many tokens survive both?
      </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="min(a, b), và ít nhất là 1: top-p giữ giá trị lớn nhất, nên ngưỡng của min-p không dịch chuyển">min(a, b), and at least 1: top-p keeps the max, so min-p's threshold does not move</Translated>, correct: true },
      { label: <Translated vi="a + b, vì mỗi bộ lọc cộng thêm các token nó giữ">a + b, since each filter adds the tokens it keeps</Translated>, why: <Translated vi="Cả hai đều là tiền tố của cùng một hàng; sống sót qua cả hai nghĩa là nằm trong tiền tố ngắn hơn.">Both are prefixes of the same row; surviving both means being in the shorter one.</Translated> },
      { label: <Translated vi="Tùy bộ lọc nào chạy trước, vì top-p chuẩn hóa lại hàng và điều đó dịch chuyển ngưỡng của min-p">It depends on which filter runs first, because top-p renormalises the row and that moves min-p's threshold</Translated>, why: <Translated vi="Theo thứ tự của llama.cpp, ngưỡng của min-p dùng logit lớn nhất, thứ top-p không bao giờ loại bỏ; không có phép chuẩn hóa lại nào đi vào phép kiểm tra của nó.">In llama.cpp's order min-p's threshold uses the max logit, which top-p never removes; no renormalisation enters its test.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mọi phép cắt bớt trên một hàng đã sắp xếp đều là một tiền tố. Ngưỡng của min-p chỉ phụ thuộc vào logit lớn nhất, thứ
            top-p giữ lại, nên tiền tố của min-p là như nhau dù top-p có chạy trước hay không. Chạy min-p trước rồi top-p trên
            các xác suất đã chuẩn hóa lại sẽ là một quy tắc khác (đường C trong trình khám phá).
          </p>
        }
      >
      <p>
        Every truncation of a sorted row is a prefix. Min-p's threshold depends only on the largest logit, which top-p
        keeps, so min-p's prefix is the same whether top-p ran first or not. Running min-p first and then top-p over
        renormalised probabilities would be a different rule (route C in the explorer).
      </p>
      </Translated>
    ),
    source: 'ours; notes/today/x1-facts.md D1, ch12-facts D3',
    vi: { source: 'của khóa học này; notes/today/x1-facts.md D1, ch12-facts D3' },
  },
  'x1/X1.6/q-equivalent': {
    kind: 'choice',
    id: 'x1/X1.6/q-equivalent',
    prompt: (
      <Translated
        vi={
          <>
            Top-k đang tắt. Bạn chạy min-p trước như một phép nén gọn ổn định, chỉ sắp xếp các token sống sót và áp dụng top-p
            với xác suất của cả hàng. Cách đó có giữ đúng các token như top-p rồi min-p của llama.cpp trên cả hàng không?
          </>
        }
      >
      <>
        Top-k is off. You run min-p first as a stable compaction, sort only the survivors and apply top-p using full-row
        probabilities. Does that keep the same tokens as llama.cpp's top-p then min-p over the whole row?
      </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Có, trong số học chính xác: mọi token xếp hạng trước một token sống sót cũng sống sót, nên khối xác suất trước nó không đổi">Yes, in exact arithmetic: every token ranked before a survivor also survives, so the mass before it is unchanged</Translated>, correct: true },
      { label: <Translated vi="Không: top-p cần cả hàng đã sắp xếp, nên một hàng đã lọc phải cho ra một nucleus khác">No: top-p needs the whole sorted row, so a filtered row must give a different nucleus</Translated>, why: <Translated vi="Top-p cần khối xác suất xếp hạng trước mỗi token được giữ, và toàn bộ khối đó thuộc về các token sống sót.">Top-p needs the mass ranked before each kept token, and all of that mass belongs to survivors.</Translated> },
      { label: <Translated vi="Chỉ khi p lớn hơn min-p, để top-p là bộ lọc lỏng hơn trong hai bộ lọc">Only when p is larger than min-p, so that top-p is the looser of the two filters</Translated>, why: <Translated vi="Hai con số này không so sánh được với nhau; lập luận đúng với mọi p và min-p.">The two numbers are not comparable; the argument holds for any p and min-p.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một token xếp hạng trước một token sống sót có logit lớn hơn hoặc bằng, nên nó cũng vượt qua ngưỡng của min-p. Khối
            xác suất trước mỗi token sống sót, tính với mẫu số của cả hàng, vì vậy là cùng một tổng trên cùng các token. Tiền tố
            được giữ là min(a, b) theo cả hai cách. Trong float, hai đường vẫn có thể khác nhau với các token nằm trong dải làm
            tròn.
          </p>
        }
      >
      <p>
        A token ranked before a survivor has a logit at least as large, so it clears min-p's threshold too. The mass before
        each survivor, computed with the full row's denominator, is then the same sum over the same tokens. The kept
        prefix is min(a, b) both ways. In floats, the two routes can still differ for tokens inside the rounding band.
      </p>
      </Translated>
    ),
    source: 'ours',
    vi: { source: 'của khóa học này' },
  },
  'x1/X1.6/q-mask': {
    kind: 'choice',
    id: 'x1/X1.6/q-mask',
    prompt: <Translated vi={<>Vì sao các sampler backend thử nghiệm của llama.cpp và mlx-lm đặt các token bị loại thành âm vô cùng thay vì nén gọn chúng thành một danh sách?</>}><>Why do llama.cpp's experimental backend samplers and mlx-lm set losing tokens to minus infinity instead of compacting them into a list?</></Translated>,
    options: [
      { label: <Translated vi="Một compute graph cần hình dạng tĩnh, còn số token sống sót phụ thuộc vào dữ liệu">A compute graph needs static shapes, and the number of survivors depends on the data</Translated>, correct: true },
      { label: <Translated vi="Che (mask) di chuyển ít byte hơn nén gọn, nên nó là lựa chọn nhanh hơn trên mọi backend">Masking moves fewer bytes than compaction, so it is the faster choice on every backend</Translated>, why: <Translated vi="Một mask vẫn đọc và ghi cả hàng; tốc độ không phải lý do được nêu.">A mask still reads and writes the whole row; speed is not the stated reason.</Translated> },
      { label: <Translated vi="Không thể viết phép nén gọn trên Metal, vốn không có primitive lọc ổn định">Compaction cannot be written on Metal, which has no stable filter primitive</Translated>, why: <Translated vi="Lab 12.4 nén gọn trên Metal; MPSGraph cũng có nonZeroIndices.">Lab 12.4 compacts on Metal; MPSGraph also has nonZeroIndices.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            MLX ghi rõ hỗ trợ hạn chế cho các phép toán có hình dạng đầu ra phụ thuộc vào dữ liệu đầu vào. Top-k backend của
            llama.cpp có thu nhỏ hàng, nhưng theo một k cố định đã biết khi dựng graph. Một kernel viết tay có thể trả về một
            danh sách và một số đếm; định kích thước bước tiếp theo từ số đếm đó cần một lần đọc từ host, hoặc một giới hạn trên
            kèm phương án dự phòng.
          </p>
        }
      >
      <p>
        MLX documents limited support for operations whose output shape depends on the input data. llama.cpp's backend
        top-k does shrink the row, but by a fixed k known when the graph is built. A hand-written kernel can return a list
        and a count; sizing the next step from that count needs a host read, or a cap with a fallback.
      </p>
      </Translated>
    ),
    source: 'notes/today/ch12-facts.md D3, D6', // i18n-ignore: citation
  },
  'x1/X1.6/q-space': {
    kind: 'choice',
    id: 'x1/X1.6/q-space',
    prompt: <Translated vi={<>Vì sao phép kiểm tra trong không gian logit l ≥ l_max + ln p_base là phép một kernel nên dùng?</>}><>Why is the logit-space test l ≥ l_max + ln p_base the one a kernel should use?</></Translated>,
    options: [
      { label: <Translated vi="Nó chỉ cần giá trị lớn nhất của hàng và một phép so sánh cho mỗi token, không có exp và không có softmax">It needs only the row max and one compare per token, with no exp and no softmax</Translated>, correct: true },
      { label: <Translated vi="Nó cho một tập token khác, được hiệu chỉnh tốt hơn so với phép kiểm tra trên xác suất">It gives a different, better-calibrated set of tokens than the test on probabilities</Translated>, why: <Translated vi="Trong số học chính xác đó là cùng một tập; chỉ tại ngưỡng, phép làm tròn mới có thể khiến chúng khác nhau.">In exact arithmetic it is the same set; only at the threshold can rounding make them differ.</Translated> },
      { label: <Translated vi="Xác suất bị underflow về 0 với phần lớn một hàng 150,000 token, nên phép kiểm tra trên chúng không xác định">Probabilities underflow to zero for most of a 150,000-token row, so their test is undefined</Translated>, why: <Translated vi="Các xác suất nhỏ vẫn so sánh đúng; underflow không phải lý do.">Small probabilities compare correctly; underflow is not the reason.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Chia p ≥ p_base · p_max cho mẫu số softmax chung rồi lấy log: mẫu số triệt tiêu. Hai phép kiểm tra chọn cùng một tập
            trong số học chính xác nhưng có thể lệch nhau một lần làm tròn đúng tại ngưỡng, nên một lab phải nói rõ nó kiểm tra
            phép nào.
          </p>
        }
      >
      <p>
        Divide p ≥ p_base · p_max by the common softmax denominator and take logs: the denominator cancels. The two tests
        pick the same set in exact arithmetic but can differ by one rounding exactly at the threshold, so a lab must say
        which one it tests.
      </p>
      </Translated>
    ),
    source: 'ours; notes/today/x1-facts.md D9, ch12-facts D3',
    vi: { source: 'của khóa học này; notes/today/x1-facts.md D9, ch12-facts D3' },
  },
};

// ---------------------------------------------------------------- the section
export function SX16() {
  const thr = minPThreshold(THRESHOLD_Q.lmax, THRESHOLD_Q.minP);
  return (
    <>
      <Translated
        vi={
          <p>
            Min-p giữ một token khi xác suất của nó ít nhất bằng một tỉ lệ p_base của xác suất token đứng đầu. Nó thích nghi với
            hình dạng của hàng: một hàng tự tin giữ ít token, một hàng phẳng giữ nhiều. Trang này biến nó thành một phép so
            sánh duy nhất, chạy nó theo hai cách bằng các lab bạn đã có, và cho thấy nó có thể thu nhỏ phép sắp xếp mà top-p cần.
          </p>
        }
      >
      <p>
        Min-p keeps a token when its probability is at least a fraction p_base of the top token's. It adapts to the shape
        of the row: a confident row keeps few tokens, a flat one many. This page turns it into a single comparison, runs it
        two ways with labs you have, and shows that it can shrink the sort that top-p needs.
      </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Min-p không có trong sách. Bộ lọc ổn định là §12.5 (pp. 297–298) và giá trị lớn nhất của hàng là phép rút gọn
              (reduction) của chương 10.
            </p>
          }
        >
        <p>
          Min-p is not in the book. The stable filter is §12.5 (pp. 297–298) and the row max is chapter 10's reduction.
        </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Một phép so sánh, không softmax">One comparison, no softmax</Translated></h2>
      <Translated
        vi={
          <p>
            Nguyen et al. giữ mọi token có xác suất ít nhất là p_base × p_max. Mọi xác suất trong một hàng có cùng mẫu số
            softmax, nên hãy chia nó đi và lấy log: giữ một token khi logit của nó ít nhất là l_max + ln p_base. Việc đó cần giá
            trị lớn nhất của hàng và một phép so sánh cho mỗi token, không exp và không tổng. Min-p phía host của llama.cpp
            chính là phép kiểm tra này, một bộ lọc ổn định tuần tự trên logits. transformers kiểm tra xác suất sau softmax và
            giữ token nằm đúng tại ngưỡng; mlx-lm kiểm tra log-xác suất.
          </p>
        }
      >
      <p>
        Nguyen et al. keep every token with probability at least p_base × p_max. Every probability in a row has the same
        softmax denominator, so divide it out and take logs: keep a token when its logit is at least l_max + ln p_base. That
        needs the row's max and one compare per token, with no exp and no sum. llama.cpp's host min-p is exactly this test,
        a sequential stable filter on logits. transformers tests probabilities after a softmax and keeps a token exactly at
        the threshold; mlx-lm tests log-probabilities.
      </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x1-facts.md D4, D8, D9; ch12-facts D3)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              llama.cpp: <code>const float min_logit = max_logit + logf(ctx-&gt;p);</code> rồi giữ{' '}
              <code>logit &gt;= min_logit</code> vào một danh sách mới.
            </p>
          }
        >
        <p>
          llama.cpp: <code>const float min_logit = max_logit + logf(ctx-&gt;p);</code> then keeps{' '}
          <code>logit &gt;= min_logit</code> into a new list.
        </p>
        </Translated>
      </Today>
      <Predict id="x1/X1.6/p-threshold" prompt={<Translated vi={<>Logit lớn nhất là 5.0 và min-p là 0.05. Ngưỡng logit là bao nhiêu?</>}><>The largest logit is 5.0 and min-p is 0.05. What is the logit threshold?</></Translated>}>
        {(v) => (
          <p data-reveal="threshold">
            <Translated
              vi={
                <>
                  5 + ln 0.05 = <b>{thr.toFixed(3)}</b>
                  {Math.abs(v - thr) < 0.01 ? ', đúng như bạn nói' : ''}. Mọi token có logit từ 2.004 trở lên đều sống sót; trình
                  khám phá bên dưới mở ra với hàng đó.
                </>
              }
            >
            5 + ln 0.05 = <b>{thr.toFixed(3)}</b>
            {Math.abs(v - thr) < 0.01 ? ', as you said' : ''}. Every token with a logit of 2.004 or more survives; the explorer
            below opens on that row.
            </Translated>
          </p>
        )}
      </Predict>
      <Ask q={questionsSX16['x1/X1.6/q-space']} />

      <h2><Translated vi="Trước hay sau temperature">Before or after temperature</Translated></h2>
      <Translated
        vi={
          <p>
            Vị trí của min-p trong chuỗi thay đổi những gì nó giữ. Bài báo và transformers áp dụng nó sau temperature, tương ứng
            với ngưỡng l_max + T ln p_base trên các logit chưa chia tỉ lệ. Chuỗi mặc định của llama.cpp và mlx-lm áp dụng nó
            trước temperature, ở T = 1, nên temperature không bao giờ dịch chuyển nó.
          </p>
        }
      >
      <p>
        Where min-p sits in the chain changes what it keeps. The paper and transformers apply it after temperature, which in
        the unscaled logits is a threshold of l_max + T ln p_base. llama.cpp's default chain and mlx-lm apply it before
        temperature, at T = 1, so temperature never moves it.
      </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x1-facts.md D1, D4, D8, D9)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Bài báo: "After applying temperature scaling, the scaled threshold pscaled is computed" (sau khi áp dụng chia tỉ lệ
              theo temperature, ngưỡng đã chia tỉ lệ pscaled được tính). Chú thích trong transformers nói điều tương tự. Chuỗi mặc
              định của llama.cpp kết thúc bằng temperature; mlx-lm áp dụng nó cuối cùng.
            </p>
          }
        >
        <p>
          The paper: "After applying temperature scaling, the scaled threshold pscaled is computed". transformers' comment
          says the same. llama.cpp's default chain ends with temperature; mlx-lm applies it last.
        </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Trình khám phá mở ra với một hàng 16 token có logit lớn nhất là 5.0, với min-p 0.05: đúng ngưỡng bạn vừa tính. Cả hai
            ngưỡng đều được vẽ; vị trí bạn chọn quyết định ngưỡng nào cắt.
          </p>
        }
      >
      <p>
        The explorer opens on a 16-token row whose largest logit is 5.0, with min-p 0.05: the threshold you just
        computed. Both thresholds are drawn; the placement you pick decides which one cuts.
      </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một hàng cố định 16 token; số học ở double, trừ phép kiểm tra logit được cài, vốn giả lập float32 với làm tròn về
              gần nhất và cộng theo thứ tự. Kernel thật cộng theo các thứ tự khác.
            </p>
          }
        >
        <p>
          A fixed 16-token row; arithmetic in double, except the planted-logit check, which emulates float32 with round to
          nearest and sums in order. Real kernels sum in other orders.
        </p>
        </Translated>
      </Simplification>
      <TempExplorer />
      <Translated
        vi={
          <p>
            Min-p sau temperature chặt hơn khi T dưới 1 và lỏng hơn khi T trên 1, vì T nhân với ln p_base, một số âm. Min-p trước
            temperature hoàn toàn không dịch chuyển theo T.
          </p>
        }
      >
      <p>
        After-temperature min-p is stricter for T below 1 and looser above it, because T multiplies ln p_base, a negative
        number. Before-temperature min-p does not move with T at all.
      </p>
      </Translated>
      <Ask q={questionsSX16['x1/X1.6/q-before-after']} label={tr('Work it out', 'Tự tính ra')} />

      <h2><Translated vi="Hai kernel cho một quy tắc">Two kernels for one rule</Translated></h2>
      <Translated
        vi={
          <p>
            Trên một hàng chưa sắp xếp, min-p là giá trị lớn nhất của hàng từ lab 10.3, một ngưỡng cho mỗi hàng (ln p_base tính
            một lần trên host, bằng float, như llama.cpp làm), và <code>launchCompactCandidates</code> của lab 12.4: vòng lặp
            phía host của llama.cpp dưới dạng một bộ lọc ổn định trên GPU, ghi ra một danh sách các cặp (cột, logit) và một số
            đếm cho mỗi hàng.
          </p>
        }
      >
      <p>
        On an unsorted row, min-p is lab 10.3's row max, one threshold per row (ln p_base computed once on the host, in
        float, as llama.cpp does), and lab 12.4's <code>launchCompactCandidates</code>: llama.cpp's host loop as a GPU
        stable filter, which writes a list of (column, logit) pairs and a count per row.
      </p>
      </Translated>
      <Source section="12.5" pages="297–298" figs={tr('Figs. 12.7, 12.8', 'Hình 12.7, 12.8')} />
      <Translated
        vi={
          <p>
            Trên một hàng đã sắp xếp, thứ M5 có sau lab 14.3, các token sống sót là một tiền tố, nên min-p là một phép đếm: số
            logit đã sắp xếp ít nhất bằng logit đầu tiên cộng ln p_base. Lab X1.2 viết nó. Top-p là một tiền tố của cùng hàng,
            nên sau cả hai, các token sống sót là tiền tố ngắn hơn, và không bao giờ ít hơn một token.
          </p>
        }
      >
      <p>
        On a sorted row, which is what M5 has after lab 14.3, the survivors are a prefix, so min-p is a count: the number of
        sorted logits at least the first one plus ln p_base. Lab X1.2 writes it. Top-p is a prefix of the same row, so after
        both the survivors are the shorter prefix, and never fewer than one token.
      </p>
      </Translated>
      <Ask q={questionsSX16['x1/X1.6/q-prefix']} />

      <h2><Translated vi="Lọc trước khi sắp xếp">Filter before you sort</Translated></h2>
      <Translated
        vi={
          <p>
            Khi top-k tắt, một top-p trên toàn bộ từ vựng sẽ sắp xếp toàn bộ từ vựng. Nhưng min-p thường giữ rất ít token, nên
            hãy chạy nó trước: nén gọn các token sống sót bằng lab 12.4, chỉ sắp xếp chúng, và tính khối xác suất của top-p bằng
            xác suất của cả hàng. Mọi token xếp hạng trước một token sống sót cũng sống sót qua min-p, nên khối xác suất trước
            mỗi token sống sót giống như trên cả hàng. Trong số học chính xác, tập được giữ chính xác là top-p rồi min-p của
            llama.cpp.
          </p>
        }
      >
      <p>
        With top-k off, a top-p over the whole vocabulary sorts the whole vocabulary. But min-p usually keeps very few
        tokens, so run it first: compact the survivors with lab 12.4, sort only them, and compute top-p's mass with the
        full row's probabilities. Every token ranked before a survivor survives min-p too, so the mass before each survivor
        is the same as over the whole row. In exact arithmetic the kept set is exactly llama.cpp's top-p then min-p.
      </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Lọc trước khi sắp xếp là thiết kế của chúng tôi, không phải của một dự án nào; trình khám phá kiểm tra nó trên các hàng tổng hợp.</p>}>
        <p>Filtering before sorting is our design, not a project's; the explorer checks it on synthetic rows.</p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Trình khám phá chạy ba đường cạnh nhau trên cả một hàng tổng hợp và đếm số giá trị mỗi đường sắp xếp. Đường C là lối
            tắt hấp dẫn: min-p trước, rồi một top-p chuẩn hóa lại trên phần còn lại.
          </p>
        }
      >
      <p>
        The explorer runs the three routes side by side on a whole synthetic row and counts what each one sorts. Route C is
        the tempting shortcut: min-p first, then a top-p that renormalises over what is left.
      </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Các hàng tổng hợp (logits kiểu Zipf có seed, như ở X1.5), không phải của một mô hình. Số học ở double; số đếm trong
              dải cho biết nơi float32 có thể quyết định khác đi.
            </p>
          }
        >
        <p>
          Synthetic rows (seeded Zipf-like logits, as in X1.5), not a model's. Arithmetic in double; the band count says
          where float32 could decide differently.
        </p>
        </Translated>
      </Simplification>
      <FilterExplorer />
      <Translated
        vi={
          <p>
            Vướng mắc nằm ở số đếm. Bao nhiêu token sống sót phụ thuộc vào dữ liệu, nên kích thước của phép sắp xếp tiếp theo
            không được biết khi công việc được lên kế hoạch. Một graph với hình dạng tĩnh không thể biểu diễn điều đó, và đó là lý
            do các sampler backend của llama.cpp và MLX che các token bị loại bằng âm vô cùng thay vì nén gọn. Một kernel có thể
            trả về một danh sách và một số đếm, nhưng định kích thước phép sắp xếp theo sau cần hoặc một lần host đọc các số
            đếm, hoặc một giới hạn trên kèm phương án dự phòng. M5 tránh vấn đề này bằng cách chạy top-k trước.
          </p>
        }
      >
      <p>
        The catch is the count. How many tokens survive depends on the data, so the size of the next sort is not known
        when the work is planned. A graph with static shapes cannot express it, which is why llama.cpp's backend samplers
        and MLX mask losers with minus infinity instead of compacting. A kernel can return a list and a count, but sizing
        the sort that follows needs either a host read of the counts or a cap with a fallback. M5 sidesteps it by running
        top-k first.
      </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch12-facts.md D3, D6)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              MLX: "MLX has limited support for operations for which output shapes are dependent on input data." (MLX hỗ trợ hạn
              chế cho các phép toán có hình dạng đầu ra phụ thuộc vào dữ liệu đầu vào.) Min-p và top-p backend của llama.cpp:
              "Apply -INFINITY bias for masked-out tokens" (áp bias -INFINITY cho các token bị che).
            </p>
          }
        >
        <p>
          MLX: "MLX has limited support for operations for which output shapes are dependent on input data." llama.cpp's
          backend min-p and top-p: "Apply -INFINITY bias for masked-out tokens".
        </p>
        </Translated>
      </Today>
      <Ask q={questionsSX16['x1/X1.6/q-equivalent']} />
      <Ask q={questionsSX16['x1/X1.6/q-mask']} />

      <h2><Translated vi="Số float tại ngưỡng">Floats at the threshold</Translated></h2>
      <Translated
        vi={
          <p>
            Phép kiểm tra trên logit và phép kiểm tra trên xác suất khớp nhau trong số học chính xác, nhưng không phải lúc nào
            cũng khớp ngay tại ngưỡng, nơi một lần làm tròn quyết định. Đánh dấu "cài một logit đúng tại ngưỡng float" trong
            trình khám phá đầu tiên: một logit đặt đúng bằng <code>max + logf(p)</code> của llama.cpp vượt qua phép kiểm tra
            trên logit, còn một softmax float32 có thể đặt xác suất của nó thấp hơn p_base × p_max một lần làm tròn. Lab X1.2
            cài một logit như vậy trong các bài test của nó, và đó là lý do hợp đồng của nó phải nói rõ nó nói tới phép kiểm tra
            nào trong hai phép.
          </p>
        }
      >
      <p>
        The logit test and the probability test agree in exact arithmetic, but not always at the threshold itself, where
        one rounding decides. Tick "plant a logit at the float threshold" in the first explorer: a logit set exactly to
        llama.cpp's <code>max + logf(p)</code> passes the logit test, and a float32 softmax can put its probability one
        rounding below p_base × p_max. Lab X1.2 plants such a logit in its tests, which is why its contract has to say which
        of the two tests it means.
      </p>
      </Translated>
      <Translated
        vi={
          <p>
            Tiếp theo, cần rút một token từ các token sống sót: <a href={href('x1', 'X1.7')}>các số ngẫu nhiên mà mọi backend
            đều thống nhất</a>, rồi đến chính phép rút.
          </p>
        }
      >
      <p>
        Next, the survivors need one token drawn from them: <a href={href('x1', 'X1.7')}>random numbers every backend agrees
        on</a>, then the draw itself.
      </p>
      </Translated>
    </>
  );
}
