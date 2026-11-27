import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, NoteGroup, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  CDF_LABEL,
  HUNT_ORDER_LABEL,
  MISSING_LABEL,
  P_DRAW,
  SAMPLER_LABEL,
  SAMPLER_PROBS,
  STEP_ROWS,
  atRiskBound,
  chooseIn,
  compareSamplers,
  disagreement,
  drawKernel,
  flipPoints,
  huntDraw,
  inverseCdf,
  kernelTotal,
  marginOf,
  perturb,
  sampleLogits,
  sampleWith,
  uniformsFor,
  weightsFrom,
  type CdfOrder,
  type Chunk,
  type HuntOrder,
  type KernelCfg,
  type Missing,
  type RaceQ,
  type Sampler,
} from '../../sim/draw';
import { f32short, m5U } from '../../sim/philox';
import { FindUScene } from './FindUScene';

// Section X1.8: the final draw. Everything the interactives show is computed by
// src/sim/draw.ts (drawKernel, huntDraw, cdfIn / chooseIn / disagreement, compareSamplers),
// pinned in draw.test.ts, which also runs labs/x1/snippets/draw_search.cu on the course's
// emulator (with each barrier removed in turn) and compares its tokens with the model.

const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const MONO: CSSProperties = { fontFamily: 'var(--sans)', fontVariationSettings: "'MONO' 1" }; // i18n-ignore: CSS value
const loaded = (t: string | null) => (t && !t.startsWith('// could not load') ? t : null);
const s = (v: number) => f32short(v);
const f32 = Math.fround;
const TOP = 1 - 2 ** -24;

// ---------------------------------------------------------------- the stepper
type View = { kind: 'total' } | { kind: 'chunk'; c: number } | { kind: 'flags'; c: number } | { kind: 'vote'; c: number } | { kind: 'done' };

function viewsOf(chunks: Chunk[]): View[] {
  const v: View[] = [{ kind: 'total' }];
  for (const ch of chunks) {
    v.push({ kind: 'chunk', c: ch.c });
    if (ch.crosses) v.push({ kind: 'flags', c: ch.c }, { kind: 'vote', c: ch.c });
  }
  v.push({ kind: 'done' });
  return v;
}
/** Runs of consecutive threads in one warp that read the same thing. */
function groupReads(reads: { thread: number; warp: number; value: number; ok: boolean; what: string }[]) {
  const out: { from: number; to: number; warp: number; value: number; ok: boolean; what: string }[] = [];
  for (const r of reads) {
    const g = out[out.length - 1];
    if (g && g.warp === r.warp && g.to === r.thread - 1 && g.what === r.what && Object.is(g.value, r.value)) g.to = r.thread;
    else out.push({ from: r.thread, to: r.thread, warp: r.warp, value: r.value, ok: r.ok, what: r.what });
  }
  return out;
}
const bits = (b: number, width: number) => Array.from({ length: width }, (_, l) => (Math.floor(b / 2 ** l) % 2 ? '1' : '0')).reverse().join('');

function FindUStepper() {
  const [rowId, setRowId] = useState('long');
  const row0 = STEP_ROWS.find((r) => r.id === rowId) ?? STEP_ROWS[1];
  const [n, setN] = useState(row0.n);
  const [T, setT] = useState(row0.T);
  const [uText, setUText] = useState(String(row0.u));
  const [fromPhilox, setFromPhilox] = useState(false);
  const [pSeed, setPSeed] = useState(7);
  const [pStream, setPStream] = useState(3);
  const [pStep, setPStep] = useState(0);
  const [vec, setVec] = useState(4);
  const [warp, setWarp] = useState(8);
  const [missing, setMissing] = useState<Missing>('none');
  const [order, setOrder] = useState<HuntOrder>('in-order');
  const [at, setAt] = useState(0);

  const uParsed = Number(uText);
  const uOk = uText.trim() !== '' && isFinite(uParsed) && uParsed >= 0 && uParsed < 1;
  const u = fromPhilox ? m5U(BigInt(pSeed), BigInt(pStream), BigInt(pStep)) : uOk ? f32(Math.min(uParsed, TOP)) : 0;
  const cfg: KernelCfg = useMemo(() => ({ threads: 16, warp, vec }), [warp, vec]);
  const weights = useMemo(() => weightsFrom(sampleLogits(n), T), [n, T]);
  const locale = useLocale();
  const run = useMemo(() => {
    const tot = kernelTotal(weights, cfg);
    const correct = drawKernel(weights, u, cfg);
    if (missing === 'none') return { tot, correct, search: correct, targets: new Array(16).fill(correct.target) as number[], hunt: null };
    const h = huntDraw(weights, u, cfg, missing, order);
    return { tot, correct, search: h.search, targets: h.targets, hunt: h.hunt };
  }, [weights, u, cfg, missing, order, locale]);
  const views = useMemo(() => viewsOf(run.search.chunks), [run]);
  const total = views.length;
  const pos = Math.min(at, total);
  const cur = pos > 0 ? views[pos - 1] : undefined;
  const reset = () => setAt(0);

  const pickRow = (id: string) => {
    const r = STEP_ROWS.find((x) => x.id === id)!;
    setRowId(id);
    setN(r.n);
    setT(r.T);
    setUText(String(r.u));
    setFromPhilox(false);
    reset();
  };

  const chunkOf = (c: number) => run.search.chunks.find((x) => x.c === c)!;
  const lastChunk = run.search.chunks[run.search.chunks.length - 1];
  const curChunk = cur && cur.kind !== 'total' && cur.kind !== 'done' ? chunkOf(cur.c) : cur?.kind === 'done' && lastChunk?.crosses ? lastChunk : undefined;
  const flagsShown = cur?.kind === 'flags' || cur?.kind === 'vote' || (cur?.kind === 'done' && !!lastChunk?.crosses);
  const nW = Math.ceil(cfg.threads / cfg.warp);
  const uniformTarget = run.targets.every((t) => t === run.targets[0]);
  const targetTxt = uniformTarget ? s(run.targets[0]) : tr('differs by thread', 'khác nhau theo luồng');

  const narrate = (() => {
    if (!cur)
      return tr(
        `One block of 16 threads owns this row of ${n} survivors. Chunks of 16 × ${vec} = ${16 * vec} survivors: thread t holds survivors t·${vec} to t·${vec} + ${vec - 1} of each chunk. Press Step.`,
        `Một khối 16 luồng phụ trách hàng gồm ${n} token còn lại này. Các đoạn gồm 16 × ${vec} = ${16 * vec} token còn lại: luồng t giữ các token còn lại từ t·${vec} đến t·${vec} + ${vec - 1} của mỗi đoạn. Bấm Bước.`,
      );
    if (cur.kind === 'total')
      return tr(
        `Phase 1: each thread adds survivors t, t + 16, t + 32, …; each warp of ${warp} adds its threads' sums with a shuffle tree; lane 0 writes the warp sum to scratch_s; warp 0 adds those and thread 0 writes the total, ${s(run.tot.total)}, to scratch_s[0]. Every thread reads it and computes target = u × total = ${targetTxt}.${run.hunt ? ` With ${MISSING_LABEL[missing]}, ${run.hunt.wrong} of 16 threads read something else (below).` : ''}`,
        `Giai đoạn 1: mỗi luồng cộng các token còn lại t, t + 16, t + 32, …; mỗi warp gồm ${warp} luồng cộng tổng của các luồng của nó bằng một cây shuffle; lane 0 ghi tổng của warp vào scratch_s; warp 0 cộng các tổng đó và luồng 0 ghi tổng, ${s(run.tot.total)}, vào scratch_s[0]. Mọi luồng đọc nó và tính đích = u × total = ${targetTxt}.${run.hunt ? ` Khi ${({ none: 'cả hai rào chắn còn nguyên', written: 'bỏ rào chắn A (sau khi ghi tổng)', read: 'bỏ rào chắn B (sau khi đọc tổng)' } as Record<Missing, string>)[missing]}, ${run.hunt.wrong} trong 16 luồng đọc phải giá trị khác (bên dưới).` : ''}`,
      );
    if (cur.kind === 'chunk') {
      const ch = chunkOf(cur.c);
      return tr(
        `Chunk ${ch.c}, survivors ${ch.base} to ${ch.end - 1}: each warp scans its threads' totals, the last lane writes the warp total to scratch_s, and every thread adds the warp totals: chunk sum ${s(ch.sum)}. Aggregate ${s(ch.aggBefore)} + ${s(ch.sum)} = ${s(ch.aggAfter)} ${ch.crosses ? `is greater than the target: this chunk crosses.` : `is not greater than the target: skip it, no flags.`}`,
        `Đoạn ${ch.c}, các token còn lại từ ${ch.base} đến ${ch.end - 1}: mỗi warp scan tổng của các luồng của nó, lane cuối ghi tổng của warp vào scratch_s, và mọi luồng cộng các tổng warp: tổng đoạn ${s(ch.sum)}. Tổng gộp ${s(ch.aggBefore)} + ${s(ch.sum)} = ${s(ch.aggAfter)} ${ch.crosses ? `lớn hơn đích: đoạn này vượt qua đích.` : `không lớn hơn đích: bỏ qua nó, không đánh cờ.`}`,
      );
    }
    if (cur.kind === 'flags') {
      const ch = chunkOf(cur.c);
      const nf = ch.lanes.reduce((a, l) => a + l.flags.filter(Boolean).length, 0);
      return tr(
        `Each thread adds the aggregate, the warps before it and the lanes before it to its own inclusive sums, then flags each survivor whose sum is greater than its target. ${nf} survivors are flagged; each thread keeps its first.`,
        `Mỗi luồng cộng tổng gộp, các warp trước nó và các lane trước nó vào các tổng bao gồm của riêng nó, rồi đánh cờ mỗi token còn lại có tổng lớn hơn đích của nó. ${nf} token còn lại được đánh cờ; mỗi luồng giữ cờ đầu tiên của nó.`,
      );
    }
    if (cur.kind === 'vote') {
      const ch = chunkOf(cur.c);
      return tr(
        `Each warp ballots "I have a flag": ${ch.ballots.map((b, k) => `warp ${k} ${bits(b, cfg.warp)}`).join(', ')}. The lowest set bit is the warp's first flagged lane; an atomicMin over the warps' candidates keeps the lowest: survivor ${ch.winner}.`,
        `Mỗi warp bỏ phiếu ballot "tôi có cờ": ${ch.ballots.map((b, k) => `warp ${k} ${bits(b, cfg.warp)}`).join(', ')}. Bit bật thấp nhất là lane có cờ đầu tiên của warp; một atomicMin trên các ứng viên của các warp giữ lại cái thấp nhất: token còn lại ${ch.winner}.`,
      );
    }
    const sr = run.search;
    const base = sr.fallback
      ? tr(
          `No sum was greater than the target: the fallback takes the last survivor with positive weight, ${sr.index}.`,
          `Không tổng nào lớn hơn đích: phương án dự phòng lấy token còn lại cuối cùng có trọng số dương, ${sr.index}.`,
        )
      : tr(`The block writes survivor ${sr.index}.`, `Khối ghi ra token còn lại ${sr.index}.`);
    return run.hunt
      ? tr(
          `${base} With both barriers it writes ${run.correct.index}: ${sr.index === run.correct.index ? 'the same token, this time.' : 'a different token.'}`,
          `${base} Với cả hai rào chắn, nó ghi ra ${run.correct.index}: ${sr.index === run.correct.index ? 'lần này là cùng một token.' : 'một token khác.'}`,
        )
      : base;
  })();

  return (
    <Bench
      title={tr('Find u', 'Tìm u')}
      task={tr(
        'Step through the 200-survivor row; then remove barrier A, then barrier B, under each order.',
        'Chạy từng bước qua hàng 200 token còn lại; rồi bỏ rào chắn A, rồi bỏ rào chắn B, dưới mỗi thứ tự.',
      )}
      foot={
        <Translated
          vi={
            <>
              Một khối 16 luồng; độ rộng warp chỉ là lựa chọn hiển thị. Các tổng float32 theo đúng thứ tự của kernel. Các thứ
              tự là hai lịch chạy cố định của trình giả lập của khóa học và một lịch mà GPU có thể tạo ra, không phải thời gian
              trên GPU.
            </>
          }
        >
          <>
            One block of 16 threads; the warp width is a display choice. Float32 sums in the kernel's own order. The orders are
            the course emulator's two fixed schedules and one a GPU may produce, not GPU timing.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Hàng">Row</Translated>
          <span className="seg" role="group" aria-label={tr('Row', 'Hàng')}>
            {STEP_ROWS.map((r) => (
              <button key={r.id} aria-pressed={rowId === r.id && n === r.n && T === r.T && !fromPhilox && uText === String(r.u)} onClick={() => pickRow(r.id)}>{tr(r.label, ({ short: '40 token còn lại', long: '200 token còn lại, T = 2.5', hunt: '120 token còn lại, T = 2' } as Record<string, string>)[r.id] ?? r.label)}</button>
            ))}
          </span>
        </span>
        <label className="field" style={{ flex: '1 1 10rem' }}>
          <Translated vi={<>số token còn lại n = {n}</>}>
            <>survivors n = {n}</>
          </Translated>
          <input type="range" min={1} max={200} value={n} aria-label="n" onChange={(e) => { setN(Number(e.target.value)); reset(); }} style={{ width: '100%' }} />
        </label>
        <label className="field" style={{ flex: '1 1 10rem' }}>
          <Translated vi={<>nhiệt độ T = {T.toFixed(2)}</>}>
            <>temperature T = {T.toFixed(2)}</>
          </Translated>
          <input type="range" min={0.3} max={3} step={0.05} value={T} aria-label="T" onChange={(e) => { setT(Number(e.target.value)); reset(); }} style={{ width: '100%' }} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="check">
          <input type="checkbox" checked={fromPhilox} aria-label={tr('u from Philox', 'u từ Philox')} onChange={(e) => { setFromPhilox(e.target.checked); reset(); }} />{' '}
          <Translated vi="u từ Philox (X1.7)">u from Philox (X1.7)</Translated>
        </label>
        {fromPhilox ? (
          <>
            {([['seed', pSeed, setPSeed], ['stream', pStream, setPStream], ['step', pStep, setPStep]] as const).map(([name, v, set]) => (
              <label key={name} className="field">
                {name}
                <input type="number" min={0} step={1} value={v} aria-label={'philox ' + name} onChange={(e) => { set(Math.max(0, Math.floor(Number(e.target.value) || 0))); reset(); }} style={{ width: '6rem' }} />
              </label>
            ))}
          </>
        ) : (
          <label className="field">
            <Translated vi="u trong [0, 1)">u in [0, 1)</Translated>
            <input type="number" min={0} max={0.99999994} step="any" value={uText} aria-label="u" onChange={(e) => { setUText(e.target.value); reset(); }} />
          </label>
        )}
        <span className="muted" data-u={u}>u = {s(u)}</span>
        <span style={FIELD}>
          <Translated vi="Token còn lại mỗi luồng">Survivors per thread</Translated>
          <span className="seg" role="group" aria-label={tr('Survivors per thread', 'Token còn lại mỗi luồng')}>
            {[1, 2, 4, 8].map((v) => (
              <button key={v} aria-pressed={vec === v} onClick={() => { setVec(v); reset(); }}>{v}</button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          <Translated vi="Độ rộng warp (hiển thị)">Warp width (display)</Translated>
          <span className="seg" role="group" aria-label={tr('Warp width', 'Độ rộng warp')}>
            {[4, 8, 16].map((v) => (
              <button key={v} aria-pressed={warp === v} onClick={() => { setWarp(v); reset(); }}>{v}</button>
            ))}
          </span>
        </span>
      </div>
      {!fromPhilox && !uOk ? <p className="t-fail" style={{ margin: '0 0 0.6rem' }}><Translated vi="u phải là một số từ 0 tới 1, không tính 1.">u must be a number from 0 up to, not including, 1.</Translated></p> : null}
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Các rào chắn quanh lệnh đọc tổng">Barriers around the read of the total</Translated>
          <span className="seg" role="group" aria-label={tr('Barriers', 'Rào chắn')}>
            {(['none', 'written', 'read'] as Missing[]).map((m) => (
              <button key={m} aria-pressed={missing === m} onClick={() => { setMissing(m); reset(); }}>
                {m === 'none' ? tr('both in place', 'giữ cả hai') : m === 'written' ? tr('remove A', 'bỏ A') : tr('remove B', 'bỏ B')}
              </button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          <Translated vi="Thứ tự">Order</Translated>
          <span className="seg" role="group" aria-label={tr('Order', 'Thứ tự')}>
            {(['in-order', 'reversed', 'warp-ahead'] as HuntOrder[]).map((o) => (
              <button key={o} aria-pressed={order === o} disabled={missing === 'none'} onClick={() => { setOrder(o); reset(); }}>{tr(HUNT_ORDER_LABEL[o], ({ 'in-order': 'trình giả lập, theo thứ tự', reversed: 'trình giả lập, đảo ngược', 'warp-ahead': 'warp 0 chạy trước (GPU có thể làm vậy)' } as Record<HuntOrder, string>)[o])}</button>
            ))}
          </span>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setAt(0)} disabled={pos === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => setAt(pos - 1)} disabled={pos === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => setAt(pos + 1)} disabled={pos >= total}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => setAt(total)} disabled={pos >= total}><Translated vi="Chạy đến cuối">Run to the end</Translated></button>
        <span className="muted" aria-live="polite" data-step={pos}><Translated vi={<>bước {pos} trên {total}</>}><>step {pos} of {total}</></Translated></span>
      </div>
      <div className={'narrate' + (cur?.kind === 'done' && run.hunt && run.search.index !== run.correct.index ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }}>
        {narrate}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 3fr)' }}>
        <div>
          <div className="panel" data-part="phase1" style={{ marginBottom: '0.8rem' }}>
            <div className="mx-label t-shared"><Translated vi="Giai đoạn 1: tổng (scratch_s)">Phase 1: the total (scratch_s)</Translated></div>
            {cur ? (
              <>
                <div style={{ fontSize: '0.78rem' }}>
                  <Translated vi="tổng các warp:">warp sums:</Translated> {run.tot.warpSums.map((v, k) => <span key={k} style={MONO}>{k ? ', ' : ''}{s(v)}</span>)}
                </div>
                <div>
                  <Translated vi="tổng">total</Translated> <b data-total={run.tot.total}>{s(run.tot.total)}</b>; <Translated vi="đích">target</Translated> <b data-target={uniformTarget ? run.targets[0] : ''}>{targetTxt}</b>
                </div>
              </>
            ) : (
              <div className="muted"><Translated vi="chưa tính">not computed yet</Translated></div>
            )}
            {cur && run.hunt ? (
              <div style={{ marginTop: '0.5rem' }} data-part="hunt" data-wrong={run.hunt.wrong}>
                <div style={{ fontSize: '0.8rem' }}>
                  <Translated
                    vi={
                      <>
                        <b className={run.hunt.wrong ? 't-fail' : 't-ok'}>{run.hunt.wrong} trong 16</b> luồng đọc phải giá trị khác với tổng ({({ 'in-order': 'trình giả lập, theo thứ tự', reversed: 'trình giả lập, đảo ngược', 'warp-ahead': 'warp 0 chạy trước (GPU có thể làm vậy)' } as Record<HuntOrder, string>)[order]}).
                      </>
                    }
                  >
                    <>
                      <b className={run.hunt.wrong ? 't-fail' : 't-ok'}>{run.hunt.wrong} of 16</b> threads read something other than the total ({HUNT_ORDER_LABEL[order]}).
                    </>
                  </Translated>
                </div>
                <table className="threads">
                  <thead><tr><th><Translated vi="luồng">threads</Translated></th><th><Translated vi="warp">warp</Translated></th><th><Translated vi="đọc scratch_s[0]">read scratch_s[0]</Translated></th><th><Translated vi="đích">target</Translated></th></tr></thead>
                  <tbody>
                    {groupReads(run.hunt.reads).map((g) => (
                      <tr key={g.from} data-read={g.from} data-to={g.to} data-ok={g.ok ? 1 : 0}>
                        <td>{g.from === g.to ? g.from : tr(`${g.from} to ${g.to}`, `${g.from} đến ${g.to}`)}</td>
                        <td>{g.warp}</td>
                        <td className={g.ok ? '' : 't-fail'} style={{ whiteSpace: 'normal' }}>{isNaN(g.value) ? '—' : s(g.value)} ({g.what})</td>
                        <td style={MONO}>{s(run.targets[g.from])}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
          </div>
          <div className="panel" data-part="chunks">
            <div className="mx-label"><Translated vi="Các đoạn">Chunks</Translated></div>
            <div data-part="chunks-scroll" style={{ overflowX: 'auto', maxWidth: '100%' }}>
            <table className="threads">
              <thead><tr><th><Translated vi="đoạn">chunk</Translated></th><th><Translated vi="token còn lại">survivors</Translated></th><th><Translated vi="tổng">sum</Translated></th><th><Translated vi="tổng gộp">aggregate</Translated></th><th /></tr></thead>
              <tbody>
                {run.search.chunks.map((ch) => {
                  const seen = views.slice(0, pos).some((v) => v.kind === 'chunk' && v.c === ch.c);
                  return (
                    <tr key={ch.c} data-chunk={ch.c} data-crosses={seen ? (ch.crosses ? 1 : 0) : ''} className={curChunk?.c === ch.c ? 'cur' : ''} style={{ opacity: seen ? (ch.crosses ? 1 : 0.55) : 0.3 }}>
                      <td>{ch.c}</td>
                      <td>{ch.base}–{ch.end - 1}</td>
                      <td style={MONO}>{seen ? s(ch.sum) : ''}</td>
                      <td style={MONO}>{seen ? `${s(ch.aggBefore)} → ${s(ch.aggAfter)}` : ''}</td>
                      <td>{seen ? (ch.crosses ? <b><Translated vi="vượt qua">crosses</Translated></b> : tr('skipped', 'bỏ qua')) : ''}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            </div>
            {run.search.chunks.length * 16 * vec < n ? <div className="muted" style={{ fontSize: '0.75rem' }}><Translated vi="các đoạn sau không bao giờ được đọc">later chunks are never read</Translated></div> : null}
          </div>
        </div>

        <div style={{ overflowX: 'auto' }}>
          {curChunk ? (
            <table className="threads" data-part="lanes" style={{ minWidth: '26rem' }}>
              <thead>
                <tr><th><Translated vi="luồng">thread</Translated></th><th><Translated vi="token còn lại: trọng số → tổng">survivors: weight → sum</Translated></th><th><Translated vi="phía trước">before</Translated></th>{run.hunt ? <th><Translated vi="đích của nó">its target</Translated></th> : null}<th><Translated vi="cờ đầu tiên">first flag</Translated></th></tr>
              </thead>
              <tbody>
                {Array.from({ length: nW }, (_, k) => {
                  const ls = curChunk.lanes.filter((l) => l.warp === k);
                  return [
                    <tr key={'w' + k}>
                      <td colSpan={run.hunt ? 5 : 4} className="muted" style={{ paddingTop: '0.4rem' }}>
                        <Translated vi={<>warp {k}</>}><>warp {k}</></Translated>
                        {cur?.kind === 'vote' || cur?.kind === 'done' ? (
                          <span data-ballot={k} data-bits={curChunk.ballots[k]}>
                            <Translated
                              vi={
                                <>
                                  {' '}· ballot <span style={MONO}>{bits(curChunk.ballots[k], cfg.warp)}</span> · lane có cờ đầu tiên {curChunk.warpFirst[k] < 0 ? 'không có' : curChunk.warpFirst[k]}
                                </>
                              }
                            >
                              <>
                                {' '}· ballot <span style={MONO}>{bits(curChunk.ballots[k], cfg.warp)}</span> · first flagged lane {curChunk.warpFirst[k] < 0 ? 'none' : curChunk.warpFirst[k]}
                              </>
                            </Translated>
                          </span>
                        ) : null}
                      </td>
                    </tr>,
                    ...ls.map((l) => {
                      const win = (cur?.kind === 'vote' || cur?.kind === 'done') && run.search.winner && run.search.winner.thread === l.thread && run.search.winner.chunk === curChunk.c;
                      return (
                        <tr key={l.thread} data-thread={l.thread} data-hit={flagsShown && l.hit >= 0 ? l.first + l.hit : ''} className={win ? 'cur' : ''}>
                          <td>{l.thread}</td>
                          <td style={{ whiteSpace: 'normal' }}>
                            {l.w.map((w, j) => {
                              const i = l.first + j;
                              if (i >= n) return null;
                              const f = flagsShown && l.flags[j];
                              return (
                                <span key={j} data-i={i} data-flag={flagsShown ? (l.flags[j] ? 1 : 0) : ''}
                                  style={{ ...MONO, display: 'inline-block', marginRight: 4, padding: '0 0.25rem', borderRadius: 3, fontSize: '0.72rem',
                                    border: `1px solid ${f ? 'var(--ok)' : 'var(--rule)'}`, background: f ? 'var(--ok-wash)' : 'transparent' }}>
                                  #{i} {s(w)}{flagsShown && l.cum.length ? ` → ${s(l.cum[j])}` : ''}
                                </span>
                              );
                            })}
                          </td>
                          <td style={MONO}>{s(l.before)}</td>
                          {run.hunt ? <td style={MONO} className={run.targets[l.thread] === run.correct.target ? '' : 't-fail'}>{s(run.targets[l.thread])}</td> : null}
                          <td>{flagsShown ? (l.hit >= 0 ? <b>#{l.first + l.hit}</b> : '—') : ''}</td>
                        </tr>
                      );
                    }),
                  ];
                })}
              </tbody>
            </table>
          ) : (
            <p className="muted" style={{ margin: 0 }}><Translated vi="Các lane của đoạn hiện tại hiện ở đây.">The lanes of the current chunk appear here.</Translated></p>
          )}
        </div>
      </div>
      {cur?.kind === 'done' ? (
        <p style={{ margin: '0.8rem 0 0' }} data-result={run.search.index} data-correct={run.correct.index} data-fallback={run.search.fallback ? 1 : 0}>
          <Translated
            vi={
              <>
                Token: <b className={run.hunt && run.search.index !== run.correct.index ? 't-fail' : 't-ok'}>token còn lại {run.search.index}</b>
                {run.search.fallback ? ' (phương án dự phòng)' : ''}. {run.search.chunks.filter((c) => !c.crosses).length} đoạn bị bỏ qua, một đoạn được scan.
                {run.hunt ? ` Với cả hai rào chắn: token còn lại ${run.correct.index}.` : ` Bản tham chiếu double trên CPU rút ra token còn lại ${inverseCdf(weights, u).index}.`}
              </>
            }
          >
            <>
              Token: <b className={run.hunt && run.search.index !== run.correct.index ? 't-fail' : 't-ok'}>survivor {run.search.index}</b>
              {run.search.fallback ? ' (fallback)' : ''}. {run.search.chunks.filter((c) => !c.crosses).length} chunk(s) skipped, one scanned.
              {run.hunt ? ` With both barriers: survivor ${run.correct.index}.` : ` The double-precision CPU reference draws survivor ${inverseCdf(weights, u).index}.`}
            </>
          </Translated>
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- near a boundary
const BROW = weightsFrom(sampleLogits(40), 1);
const EPS: { label: string; v: number }[] = [
  {
    get label() {
      return tr('none', 'không');
    },
    v: 0,
  },
  { label: '2⁻²²', v: 2 ** -22 },
  { label: '2⁻²⁰', v: 2 ** -20 },
  { label: '2⁻¹⁸', v: 2 ** -18 },
];
const ORDERS: CdfOrder[] = ['double', 'f32-loop', 'f32-kernel'];
const K_RANGE = 12;

function BoundaryExplorer() {
  const [b, setB] = useState(20);
  const [eps, setEps] = useState(0);
  const [k, setK] = useState(0);
  const pert = useMemo(() => (eps ? perturb(BROW, eps) : null), [eps]);
  const flip = useMemo(() => flipPoints(BROW, 'double'), []);
  const c = Math.round(flip[b] * 2 ** 24);
  const ks = Array.from({ length: 2 * K_RANGE + 1 }, (_, i) => i - K_RANGE);
  const rows: { id: string; label: string; pick: (u: number) => number }[] = [
    ...ORDERS.map((o) => ({ id: o, label: tr(CDF_LABEL[o], ({ double: 'double, từ trái sang phải (bản tham chiếu CPU)', 'f32-loop': 'float32, từ trái sang phải', 'f32-kernel': 'float32, theo thứ tự của kernel (luồng, warp, khối)' } as Record<CdfOrder, string>)[o]), pick: (u: number) => chooseIn(BROW, u, o) })),
    ...(pert ? [{ id: 'perturbed', label: tr("float32, the kernel's order, weights perturbed (another backend's exp)", 'float32, theo thứ tự của kernel, trọng số bị nhiễu (exp của một backend khác)'), pick: (u: number) => chooseIn(pert, u, 'f32-kernel') }] : []),
  ];
  const grid = rows.map((r) => ks.map((d) => r.pick((c + d) / 2 ** 24)));
  const ref = grid[0];
  const u = (c + k) / 2 ** 24;
  const dis = useMemo(
    () => disagreement([{ weights: BROW, order: 'double' }, { weights: BROW, order: 'f32-loop' }, { weights: BROW, order: 'f32-kernel' }, ...(pert ? [{ weights: pert, order: 'f32-kernel' as CdfOrder }] : [])]),
    [pert],
  );
  const col = ks.indexOf(k);
  return (
    <Bench
      title={tr('Near a boundary', 'Gần một ranh giới')}
      task={tr(
        'Drag u across boundary 20, then switch on a perturbation of 2⁻²⁰ and watch the band widen.',
        'Kéo u qua ranh giới 20, rồi bật nhiễu 2⁻²⁰ và xem dải rộng ra.',
      )}
      foot={
        <Translated
          vi={<>u dịch theo bước 2⁻²⁴, đúng các bước mà u của M5 có thể nhận. Đỏ: một lựa chọn khác với bản tham chiếu double trên CPU.</>}
        >
          <>u moves in steps of 2⁻²⁴, the steps M5's u can take. Red: a choice that differs from the double-precision CPU reference.</>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ flex: '1 1 12rem' }}>
          <Translated vi={<>ranh giới giữa token còn lại {b} và {b + 1}</>}>
            <>boundary between survivors {b} and {b + 1}</>
          </Translated>
          <input type="range" min={0} max={38} value={b} aria-label={tr('boundary', 'ranh giới')} onChange={(e) => { setB(Number(e.target.value)); setK(0); }} style={{ width: '100%' }} />
        </label>
        <span style={FIELD}>
          <Translated vi="Nhiễu tương đối của các trọng số">Relative perturbation of the weights</Translated>
          <span className="seg" role="group" aria-label={tr('Perturbation', 'Nhiễu')}>
            {EPS.map((e) => (
              <button key={e.label} aria-pressed={eps === e.v} onClick={() => setEps(e.v)}>{e.label}</button>
            ))}
          </span>
        </span>
      </div>
      <label className="field" style={{ display: 'flex', marginBottom: '0.7rem' }}>
        <Translated vi={<>u = ranh giới {k >= 0 ? '+' : '−'} {Math.abs(k)} × 2⁻²⁴</>}>
          <>u = boundary {k >= 0 ? '+' : '−'} {Math.abs(k)} × 2⁻²⁴</>
        </Translated>
        <input type="range" min={-K_RANGE} max={K_RANGE} value={k} aria-label={tr('u offset', 'độ lệch của u')} onChange={(e) => setK(Number(e.target.value))} style={{ width: '100%' }} />
      </label>
      <div style={{ overflowX: 'auto' }}>
        <table className="threads" data-part="grid" style={{ minWidth: '40rem' }}>
          <thead>
            <tr>
              <th><Translated vi="thứ tự">order</Translated></th>
              {ks.map((d) => <th key={d} style={{ textAlign: 'center', fontWeight: d === k ? 800 : 400 }}>{d === 0 ? '0' : d % 4 === 0 ? (d > 0 ? '+' + d : d) : ''}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, ri) => (
              <tr key={r.id} data-order={r.id} data-tokens={grid[ri].join(',')}>
                <td style={{ whiteSpace: 'normal', minWidth: '9rem', maxWidth: '15rem' }}>{r.label}</td>
                {grid[ri].map((t, ci) => (
                  <td key={ci} style={{ textAlign: 'center', padding: '0.2rem 0', minWidth: '1.5rem',
                    background: t !== ref[ci] ? 'var(--fail-wash)' : ci === col ? 'var(--rule-soft)' : undefined,
                    color: t !== ref[ci] ? 'var(--fail)' : undefined, fontWeight: ci === col ? 700 : 400 }}>{t}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ margin: '0.7rem 0 0' }} data-part="readout" data-margin={marginOf(BROW, u)} data-choices={rows.map((r, i) => grid[i][col]).join(',')}>
        <Translated
          vi={
            <>
              u = {u.toPrecision(9)}: {rows.map((r, i) => `${r.id === 'perturbed' ? 'bị nhiễu' : r.id} → ${grid[i][col]}`).join(', ')}. Lượt rút cách ranh giới gần nhất{' '}
              {marginOf(BROW, u).toExponential(2)} của tổng.
            </>
          }
        >
          <>
            u = {u.toPrecision(9)}: {rows.map((r, i) => `${r.id === 'perturbed' ? 'perturbed' : r.id} → ${grid[i][col]}`).join(', ')}. The draw is{' '}
            {marginOf(BROW, u).toExponential(2)} of the total from the nearest boundary.
          </>
        </Translated>
      </p>
      <p style={{ margin: '0.4rem 0 0' }} data-part="risk" data-measure={dis.measure}>
        <Translated
          vi={
            <>
              Trên cả 39 ranh giới của hàng này, các thứ tự được hiển thị bất đồng trên <b>{(dis.measure * 1e6).toFixed(1)} phần triệu</b> giá trị của u.
              Cận 2 × (n − 1) × m cho {(atRiskBound(40, 2 ** -22) * 1e6).toFixed(1)} phần triệu ở biên m bằng 2⁻²²
              và {(atRiskBound(40, 2 ** -20) * 1e6).toFixed(1)} ở 2⁻²⁰.
            </>
          }
        >
          <>
            Over all 39 boundaries of this row, the orders shown disagree on <b>{(dis.measure * 1e6).toFixed(1)} per million</b> of u.
            The bound 2 × (n − 1) × m gives {(atRiskBound(40, 2 ** -22) * 1e6).toFixed(1)} per million at a margin m of 2⁻²²
            and {(atRiskBound(40, 2 ** -20) * 1e6).toFixed(1)} at 2⁻²⁰.
          </>
        </Translated>
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- three samplers
const DRAWS = [1, 10, 100, 1000, 10000, 100000];
const METHODS: Sampler[] = ['inverse-cdf', 'race', 'gumbel'];

function SamplersExplorer() {
  const [draws, setDraws] = useState(1000);
  const [seed, setSeed] = useState(5);
  const [logU, setLogU] = useState(false);
  const q: RaceQ = logU ? 'log-u' : 'one-minus';
  const cmp = useMemo(() => compareSamplers(SAMPLER_PROBS, draws, seed, 0, q), [draws, seed, q]);
  const first = useMemo(
    () => Array.from({ length: 8 }, (_, d) => {
      const us = uniformsFor(seed, 0, d, SAMPLER_PROBS.length);
      return { d, u0: us[0], picks: METHODS.map((m) => sampleWith(SAMPLER_PROBS, us, m, q)) };
    }),
    [seed, q],
  );
  return (
    <Bench
      title={tr('Same bits, three samplers', 'Cùng các bit, ba sampler')}
      task={tr(
        'Run 10 draws, then 100,000: the columns converge to p, the per-draw agreement does not.',
        'Chạy 10 lượt rút, rồi 100,000: các cột hội tụ về p, còn mức trùng khớp theo từng lượt rút thì không.',
      )}
      foot={
        <Translated
          vi={<>Độ chính xác double. Mỗi lượt rút lấy 6 số ngẫu nhiên đều từ Philox (seed, stream 0, step = lượt rút), theo phép chuyển đổi của M5; số đều 0 là u của M5.</>}
        >
          <>Double precision. Each draw takes 6 uniforms from Philox (seed, stream 0, step = draw), M5's conversion; uniform 0 is M5's u.</>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Số lượt rút">Draws</Translated>
          <span className="seg" role="group" aria-label={tr('Draws', 'Số lượt rút')}>
            {DRAWS.map((d) => (
              <button key={d} aria-pressed={draws === d} onClick={() => setDraws(d)}>{d.toLocaleString('en-US')}</button>
            ))}
          </span>
        </span>
        <label className="field">
          <Translated vi="seed">seed</Translated>
          <input type="number" min={0} step={1} value={seed} aria-label={tr('sampler seed', 'seed của sampler')} onChange={(e) => setSeed(Math.max(0, Math.floor(Number(e.target.value) || 0)))} style={{ width: '6rem' }} />
        </label>
        <label className="check">
          <input type="checkbox" checked={logU} aria-label={tr('race with q = -ln u', 'cuộc đua với q = -ln u')} onChange={(e) => setLogU(e.target.checked)} />{' '}
          <Translated vi="cuộc đua với q = −ln u thay vì −ln(1 − u)">race with q = −ln u instead of −ln(1 − u)</Translated>
        </label>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0, background: 'var(--page)', minWidth: '34rem' }} data-part="freq">
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}><Translated vi="token">token</Translated></th>
              <th className="num">p</th>
              {METHODS.map((m) => <th key={m} className="num">{m === 'inverse-cdf' ? tr('inverse CDF', 'CDF ngược') : m === 'race' ? 'argmax(p / q)' : 'argmax(l + g)'}</th>)}
            </tr>
          </thead>
          <tbody>
            {SAMPLER_PROBS.map((p, i) => (
              <tr key={i} data-token={i}>
                <td style={{ paddingLeft: '0.6rem' }}>{i}</td>
                <td className="num">{p.toFixed(2)}</td>
                {METHODS.map((m) => (
                  <td key={m} className="num" data-method={m} data-count={cmp.counts[m][i]}>{(cmp.counts[m][i] / draws).toFixed(draws >= 1000 ? 3 : 2)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ margin: '0.7rem 0 0' }} data-part="agree" data-inv-race={cmp.agree.invRace} data-inv-gumbel={cmp.agree.invGumbel} data-race-gumbel={cmp.agree.raceGumbel}>
        <Translated
          vi={
            <>
              Cùng token trên cùng lượt rút: CDF ngược và cuộc đua {(cmp.agree.invRace * 100).toFixed(1)}%, CDF ngược và Gumbel{' '}
              {(cmp.agree.invGumbel * 100).toFixed(1)}%, cuộc đua và Gumbel {(cmp.agree.raceGumbel * 100).toFixed(1)}%.
            </>
          }
        >
          <>
            Same token on the same draw: inverse CDF and race {(cmp.agree.invRace * 100).toFixed(1)}%, inverse CDF and Gumbel{' '}
            {(cmp.agree.invGumbel * 100).toFixed(1)}%, race and Gumbel {(cmp.agree.raceGumbel * 100).toFixed(1)}%.
          </>
        </Translated>
      </p>
      <div style={{ overflowX: 'auto', marginTop: '0.6rem' }}>
        <table className="threads" data-part="first" style={{ minWidth: '28rem' }}>
          <thead><tr><th><Translated vi="lượt rút">draw</Translated></th><th><Translated vi="u (số đều 0)">u (uniform 0)</Translated></th>{METHODS.map((m) => <th key={m}><Translated vi={({ 'inverse-cdf': 'CDF ngược', race: 'cuộc đua mũ argmax(p / q)', gumbel: 'Gumbel-max argmax(l + g)' } as Record<Sampler, string>)[m]}>{SAMPLER_LABEL[m].split(' (')[0]}</Translated></th>)}</tr></thead>
          <tbody>
            {first.map((r) => (
              <tr key={r.d} data-draw={r.d} data-picks={r.picks.join(',')}>
                <td>{r.d}</td>
                <td style={MONO}>{s(r.u0)}</td>
                {r.picks.map((t, i) => <td key={i}><b>{t}</b></td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSX18: Record<string, Question> = {
  'x1/X1.8/q-ge-gt': {
    kind: 'choice',
    id: 'x1/X1.8/q-ge-gt',
    prompt: (
      <Translated
        vi={
          <>
            Các ứng viên theo thứ tự từ vựng có xác suất [0, 0.5, 0.5] và u = 0 chính xác (có thể xảy ra với phép chuyển đổi của M5).
            Quy tắc "tổng bao gồm đầu tiên ≥ u × total" chọn chỉ số nào, và quy tắc "lớn hơn hẳn" chọn chỉ số nào?
          </>
        }
      >
        <>
          Candidates in vocabulary order with probabilities [0, 0.5, 0.5] and u = 0 exactly (possible with M5's conversion).
          Which index does "first inclusive sum ≥ u × total" pick, and which does "strictly greater"?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="≥ chọn chỉ số 0, một token có xác suất bằng 0; > chọn chỉ số 1">
            {'≥ picks index 0, a zero-probability token; > picks index 1'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Cả hai chọn chỉ số 1: cả hai quy tắc đều bỏ qua token có xác suất 0">
            Both pick index 1: both rules skip a token whose probability is 0
          </Translated>
        ),
        why: (
          <Translated vi="Tổng bao gồm đầu tiên là 0, và 0 ≥ 0 đúng, nên ≥ dừng ở chỉ số 0.">
            The first inclusive sum is 0, and 0 ≥ 0 holds, so ≥ stops at index 0.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Cả hai chọn chỉ số 0">Both pick index 0</Translated>,
        why: (
          <Translated vi="0 > 0 là sai, nên quy tắc lớn hơn hẳn đi tiếp tới chỉ số 1, nơi tổng là 0.5.">
            {'0 > 0 is false, so the strict rule moves on to index 1, where the sum is 0.5.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đích 0 × 1 = 0; các tổng bao gồm 0, 0.5, 1. "≥" dừng ở cái đầu tiên, một token mà mô hình gán xác suất 0. "&gt;"
            không bao giờ dừng ở token trọng số 0, vì cộng thêm 0 không thể làm một tổng vượt quá đích mà tổng ngay trước nó
            chưa vượt. M5 dùng "&gt;".
          </p>
        }
      >
        <p>
          Target 0 × 1 = 0; inclusive sums 0, 0.5, 1. "≥" stops at the first, a token the model gave probability 0. "&gt;"
          never stops on a zero-weight token, because adding 0 cannot make a sum exceed a target the sum before it did not
          exceed. M5 uses "&gt;".
        </p>
      </Translated>
    ),
    source: 'ours; notes/today/x1-facts.md A2, E2',
    vi: { source: 'của khóa học này; notes/today/x1-facts.md A2, E2' },
  },
  'x1/X1.8/q-u1': {
    kind: 'choice',
    id: 'x1/X1.8/q-u1',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel rút u bằng <code>curand_uniform</code>, hàm có thể trả về đúng 1.0, rồi tìm tổng bao gồm đầu tiên
            ≥ u, nhưng tổng float của hàng là 0.99999994. Chuyện gì xảy ra?
          </>
        }
      >
        <>
          A kernel draws u with <code>curand_uniform</code>, which can return exactly 1.0, and searches for the first
          inclusive sum ≥ u, but the row's float sum is 0.99999994. What happens?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không chỉ số nào thỏa; không có phương án dự phòng thì kernel ghi ra một token không hợp lệ hoặc đọc vượt quá hàng">
            No index qualifies; without a fallback the kernel writes an invalid token or reads past the row
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Token cuối được chọn tự động, vì một phép tìm chạy quá cuối hàng sẽ dừng ở phần tử cuối">
            The last token is chosen automatically, because a search that runs off the end of the row stops on the last entry
          </Translated>
        ),
        why: (
          <Translated vi="Chỉ khi có người viết phương án dự phòng đó. Bản thân phép tìm không tìm thấy gì.">
            Only if someone wrote that fallback. The search itself finds nothing.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không có gì, vì các xác suất cộng lại bằng 1">Nothing, because probabilities sum to 1</Translated>
        ),
        why: (
          <Translated vi="Chúng cộng lại bằng 1 trong số học chính xác; tổng float ở đây là 1 − 2⁻²⁴.">
            They sum to 1 in exact arithmetic; the float sum here is 1 − 2⁻²⁴.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Kernel của FlashInfer nói rõ điều này trong một chú thích: "this would happen when u is very close to 1 and the sum of probabilities
            is smaller than u. In this case, we use the last valid index". Phép tìm nhị phân của PyTorch cũng có một chốt chặn: "this actually
            can happen when the random number is 1".
          </p>
        }
      >
        <p>
          FlashInfer's kernel says so in a comment: "this would happen when u is very close to 1 and the sum of probabilities
          is smaller than u. In this case, we use the last valid index". PyTorch's binary search has a guard too: "this actually
          can happen when the random number is 1".
        </p>
      </Translated>
    ),
    source: 'notes/today/x1-facts.md A2, A6, D5 (checked 2026-11-20)',
    vi: { source: 'notes/today/x1-facts.md A2, A6, D5 (kiểm tra 2026-11-20)' },
  },
  'x1/X1.8/q-multinomial': {
    kind: 'choice',
    id: 'x1/X1.8/q-multinomial',
    prompt: (
      <Translated
        vi={<>Ngày nay <code>torch.multinomial(probs, 1)</code>, lệnh gọi mà <code>generate</code> của transformers dùng, tính gì?</>}
      >
        <>What does <code>torch.multinomial(probs, 1)</code>, the call transformers' <code>generate</code> makes, compute today?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Một cuộc đua mũ: q ~ Exp(1) cho mỗi phần tử, rồi argmax(p / q), trên CPU, CUDA, MPS và XPU">
            An exponential race: q ~ Exp(1) per entry, then argmax(p / q), on CPU, CUDA, MPS and XPU
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Một phép tìm nhị phân trên tổng tích lũy của probs: CDF ngược">
            A binary search of the cumulative sum of probs: the inverse CDF
          </Translated>
        ),
        why: (
          <Translated vi="Nhánh đó chỉ dùng khi có hoàn lại và nhiều hơn một mẫu.">
            That path is used only with replacement and more than one sample.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chính argmax của probs: với một mẫu và không hoàn lại, multinomial lùi về greedy">
            The argmax of probs itself: with one sample and no replacement, multinomial falls back to greedy
          </Translated>
        ),
        why: (
          <Translated vi="Không có nhánh lùi về greedy. Nó chia mỗi p cho một lượt rút Exp(1) mới trước argmax, nên bất kỳ token nào có p > 0 đều có thể thắng.">
            {'There is no greedy fallback. It divides each p by a fresh Exp(1) draw before the argmax, so any token with p > 0 can win.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>Distributions.cpp</code>: "Fast-path for no replacement or if only one sample is drawn", <code>q.exponential_(1)</code>,
            <code>p / q</code>, <code>argmax</code>. Cùng phân phối với CDF ngược, nhưng khác token từ cùng các bit.
          </p>
        }
      >
        <p>
          <code>Distributions.cpp</code>: "Fast-path for no replacement or if only one sample is drawn", <code>q.exponential_(1)</code>,
          <code>p / q</code>, <code>argmax</code>. Same distribution as the inverse CDF, different token from the same bits.
        </p>
      </Translated>
    ),
    source: 'notes/today/x1-facts.md D4, D5 (checked 2026-11-20)',
    vi: { source: 'notes/today/x1-facts.md D4, D5 (kiểm tra 2026-11-20)' },
  },
  'x1/X1.8/q-mlx-batch': {
    kind: 'choice',
    id: 'x1/X1.8/q-mlx-batch',
    prompt: (
      <Translated
        vi={<>mlx-lm rút một hàng bằng <code>mx.random.categorical</code>. Theo cách chúng tôi đọc mã nguồn của MLX, điều gì thay đổi khi batch tăng từ 1 lên 2 hàng với cùng một khóa?</>}
      >
        <>mlx-lm draws a row with <code>mx.random.categorical</code>. By our reading of MLX's source, what changes when the batch grows from 1 to 2 rows with the same key?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Thuật toán: CDF ngược ở batch 1, Gumbel-max từ 2 hàng trở lên, nên cùng một khóa có thể cho một token khác cho một hàng">
            The algorithm: inverse CDF at batch 1, Gumbel-max from 2 rows, so one key can give a different token for a row
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chỉ tốc độ: categorical chạy cùng một thuật toán ở mọi kích thước batch, nên mỗi hàng giữ token nó đã rút">
            Nothing but the speed: categorical runs the same algorithm at every batch size, so each row keeps the token it drew
          </Translated>
        ),
        why: (
          <Translated vi="`categorical_impl` chỉ đi theo nhánh CDF ngược cho một phân phối duy nhất có sample shape.">
            `categorical_impl` takes its inverse-CDF path only for a single distribution with a sample shape.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Khóa được tách theo từng hàng, nên mỗi hàng nhận một stream mới nhưng vẫn cùng thuật toán">
            The key is split per row, so each row gets a fresh stream but the same algorithm
          </Translated>
        ),
        why: (
          <Translated vi="Sự chuyển đổi nằm ở thuật toán, không chỉ ở khóa.">
            The switch is in the algorithm, not only in the key.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>categorical_inverse_cdf</code> chạy khi logits là một phân phối và có sample shape; nếu không thì
            <code> argmax(logits + gumbel)</code>. Với logits dạng (batch, từ vựng), batch 1 thỏa còn batch 2 thì không (theo
            cách chúng tôi đọc).
          </p>
        }
      >
        <p>
          <code>categorical_inverse_cdf</code> runs when the logits are one distribution and a sample shape is given; otherwise
          <code> argmax(logits + gumbel)</code>. With (batch, vocabulary) logits, batch 1 qualifies and batch 2 does not (our
          reading).
        </p>
      </Translated>
    ),
    source: 'notes/today/x1-facts.md D7, D8 (our reading of the source, checked 2026-11-20)',
    vi: { source: 'notes/today/x1-facts.md D7, D8 (cách chúng tôi đọc mã nguồn, kiểm tra 2026-11-20)' },
  },
  'x1/X1.8/q-risk': {
    kind: 'numeric',
    id: 'x1/X1.8/q-risk',
    prompt: (
      <Translated
        vi={
          <>
            u phân phối đều trên [0, 1). Một lượt rút chỉ có thể đổi kết quả giữa các backend nếu u × total rơi vào trong biên 2⁻²⁰ × total
            quanh một trong 39 ranh giới bên trong của 40 token còn lại. Nhiều nhất bao nhiêu lượt rút trên một triệu có rủi ro?
          </>
        }
      >
        <>
          u is uniform on [0, 1). A draw can flip between backends only if u × total lands within a margin of 2⁻²⁰ × total of
          one of the 39 interior boundaries of 40 survivors. At most how many draws per million are at risk?
        </>
      </Translated>
    ),
    answer: 74,
    tol: 0.02,
    unit: 'per million',
    explain: (
      <Translated vi={<p>Mỗi ranh giới có một dải rộng 2 × 2⁻²⁰ trên trục u: 2 × 39 × 2⁻²⁰ × 10⁶ = 74.4.</p>}>
        <p>Each boundary has a band of width 2 × 2⁻²⁰ on the u axis: 2 × 39 × 2⁻²⁰ × 10⁶ = 74.4.</p>
      </Translated>
    ),
    source: 'ours; src/sim/draw.ts atRiskBound',
    vi: { unit: 'trên một triệu', source: 'của khóa học này; src/sim/draw.ts atRiskBound' },
  },
  'x1/X1.8/q-device-u': {
    kind: 'choice',
    id: 'x1/X1.8/q-device-u',
    prompt: (
      <Translated
        vi={<><code>dist</code> trên backend của llama.cpp tải lên ở mỗi bước một u do mt19937 của host rút. Việc tính u trên device từ (seed, stream, step) mang lại gì?</>}
      >
        <>llama.cpp's backend <code>dist</code> uploads a u drawn by the host's mt19937 every step. What does deriving u on the device from (seed, stream, step) buy?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không còn lần tải lên từ host ở mỗi bước, và mọi backend tái tạo cùng một u từ cùng các đầu vào">
            No per-step upload from the host, and every backend reproduces the same u from the same inputs
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Tính ngẫu nhiên tốt hơn mt19937, vốn có đầu ra 32 bit quá thô cho bộ từ vựng 150,000">
            Better randomness than mt19937, whose 32-bit outputs are too coarse for a vocabulary of 150,000
          </Translated>
        ),
        why: (
          <Translated vi="Cả hai đều vượt qua các bộ kiểm thử chuẩn, và 32 bit cho hơn bốn tỷ giá trị, mịn hơn rất nhiều so với 150,000 token. Chất lượng không phải là khác biệt.">
            Both pass the standard test batteries, and 32 bits give over four billion values, far finer than 150,000 tokens. Quality is not the difference.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không gì cả: lần tải lên chỉ vài byte mỗi hàng, và nó chồng lấp với kernel kế tiếp">
            Nothing: the upload is a few bytes per row, and it overlaps with the next kernel
          </Translated>
        ),
        why: (
          <Translated vi="Số byte thì ít, nhưng đó là một bước trên host cho mỗi token, và stream của host trói mọi backend vào host.">
            The bytes are few, but it is a host step per token, and the host stream ties every backend to the host.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lựa chọn của llama.cpp có lý do riêng: nhánh host và nhánh backend dùng chung một stream mt19937, nên chuyển nhánh vẫn giữ
            nguyên các số. M5 đánh đổi điều đó để lấy một lượt rút được tính ngay nơi logits đang nằm.
          </p>
        }
      >
        <p>
          llama.cpp's choice has its own reason: host and backend paths consume one mt19937 stream, so switching paths keeps
          the numbers. M5 trades that for a draw computed where the logits already are.
        </p>
      </Translated>
    ),
    source: 'notes/today/x1-facts.md D3 (checked 2026-11-20)',
    vi: { source: 'notes/today/x1-facts.md D3 (kiểm tra 2026-11-20)' },
  },
  'x1/X1.8/q-barrier': {
    kind: 'choice',
    id: 'x1/X1.8/q-barrier',
    prompt: (
      <Translated
        vi={
          <>
            Trong kernel rút, mọi luồng đọc tổng từ <code>scratch_s[0]</code> (giữa rào chắn A và B), rồi vòng lặp theo
            đoạn ghi các tổng warp vào cùng mảng đó. Bỏ rào chắn B. Backend nào của khóa học chắc chắn
            báo lỗi này?
          </>
        }
      >
        <>
          In the draw kernel every thread reads the total from <code>scratch_s[0]</code> (between barriers A and B), and the
          chunk loop then writes warp totals into the same array. Remove barrier B. Which of the course's backends is sure to
          report it?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Backend race: nó chỉ ra lệnh đọc tổng và lệnh ghi một tổng warp; emu qua được ở mọi thứ tự">
            The race backend: it names the read of the total and the write of a warp total; emu passes in every order
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Backend emu, dưới lịch chạy theo thứ tự, lịch chạy luồng 0 trước">
            The emu backend, under its in-order schedule, which runs thread 0 first
          </Translated>
        ),
        why: (
          <Translated vi="Mọi luồng đọc tổng trước lần shuffle đầu tiên, và không luồng nào ghi một tổng warp cho đến khi cả warp của nó đã qua phép scan: ở cả hai thứ tự cố định, mọi lệnh đọc đều đến trước.">
            Every thread reads the total before its first shuffle, and no thread writes a warp total until its whole warp is past the scan: in both fixed orders all reads come first.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một GPU, ở mọi lần chạy, vì các warp của nó thật sự chạy đồng thời">
            A GPU, on every run, since its warps really do run concurrently
          </Translated>
        ),
        why: (
          <Translated vi="Chỉ khi một warp chạy vượt lên qua phép scan của nó trước khi warp khác kịp đọc tổng; nó có thể qua được mọi lần chạy.">
            Only when one warp runs ahead through its scan before another has read the total; it may pass every run.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Rào chắn A là cái mà emu bắt được: thiếu nó, các warp khác đọc phải tổng từng phần của warp 0. Rào chắn B bảo vệ một
            lệnh ghi-sau-đọc giữa các warp mà các lịch chạy của trình giả lập không bao giờ đảo thứ tự, nên chỉ backend race, vốn kiểm tra
            quan hệ happens-before thay vì kết quả, mới thấy nó. Được kiểm tra bởi <code>draw.test.ts</code> trên đoạn mã.
          </p>
        }
      >
        <p>
          Barrier A is the one emu catches: without it, the other warps read warp 0's partial sum. Barrier B guards a
          write-after-read across warps that the emulator's schedules never reorder, so only the race backend, which checks
          happens-before rather than outcomes, sees it. Checked by <code>draw.test.ts</code> on the snippet.
        </p>
      </Translated>
    ),
    source: 'ours; labs/README.md (emu and race); src/sim/draw.test.ts',
    vi: { source: 'của khóa học này; labs/README.md (emu và race); src/sim/draw.test.ts' },
  },
};

// ---------------------------------------------------------------- the section
export function SX18() {
  const cu = loaded(useLabFile('x1/snippets/draw_search.cu'));
  const barrierLines = cu ? cu.split('\n').flatMap((l, i) => (/barrier [AB]:/.test(l) ? [i + 1] : [])) : [];
  return (
    <>
      <Translated
        vi={
          <p>
            Mọi giai đoạn tới giờ đều thu hẹp hàng. Giờ phải chọn ra một token còn lại, với xác suất tỉ lệ với trọng số
            của nó, dùng u từ <a href={href('x1', 'X1.7')}>trang trước</a>. Xếp các trọng số nối đuôi nhau trên một đường thẳng,
            đặt một dấu tại u nhân tổng của chúng, và lấy token còn lại nằm dưới dấu đó.
          </p>
        }
      >
        <p>
          Every stage so far has narrowed the row. Now one survivor must be picked, with probability proportional to its
          weight, using the u from the <a href={href('x1', 'X1.7')}>previous page</a>. Lay the weights end to end on a line,
          put a mark at u times their total, and take the survivor under the mark.
        </p>
      </Translated>
      <NoteGroup>
        <Beyond>
          <Translated
            vi={
              <p>
                Phép rút là của khóa học này. Nó là "cây xúc xích" của chương 11 với u làm con dao: một phép scan bao gồm cho ra các điểm
                cắt. Phép scan theo khối là §11.4, tổng là §10.8, phép bỏ phiếu là §12.3.
              </p>
            }
          >
            <p>
              The draw is ours. It is chapter 11's sausage with u as the knife: an inclusive scan gives the cut points. The block
              scan is §11.4, the total §10.8, the vote §12.3.
            </p>
          </Translated>
        </Beyond>
        <Source section="11.1" pages="252–254" />
      </NoteGroup>

      <h2><Translated vi="CDF ngược (inverse CDF)">The inverse CDF</Translated></h2>
      <Translated
        vi={
          <p>
            Các token còn lại đến theo thứ tự đã sắp xếp, nên l₀ là logit lớn nhất. Trọng số của chúng là w_i = exp((l_i − l₀) / T), với
            nhiệt độ được gộp vào ở đây, như ở trang X1.2. Một phép rút gọn (reduction) theo khối cho ra tổng. Đích là u × total, và
            token là i đầu tiên có tổng bao gồm w₀ + … + w_i lớn hơn hẳn đích.
          </p>
        }
      >
        <p>
          The survivors arrive sorted, so l₀ is the largest logit. Their weights are w_i = exp((l_i − l₀) / T), with
          temperature folded in here, as on the X1.2 page. A block reduction gives the total. The target is u × total, and the
          token is the first i whose inclusive sum w₀ + … + w_i is strictly greater than the target.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Nhân u với tổng, thay vì so u với các tổng xác suất, tránh được giả định rằng các trọng số cộng lại đúng bằng
            1: trong float chúng không bao giờ như vậy. Sampler trên host của llama.cpp cũng làm như thế.
          </p>
        }
      >
        <p>
          Scaling u by the total, rather than comparing u with sums of probabilities, avoids assuming the weights sum to exactly
          1: they never do in float. llama.cpp's host sampler does the same.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x1-facts.md D3)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              llama.cpp <code>dist</code>: <code>const double sum_tgt = sum_cum*rnd;</code>, rồi ứng viên đầu tiên có{' '}
              <code>sum_run &gt;= sum_tgt</code>. Nhánh backend của nó so các xác suất tích lũy với u, theo kiểu lớn hơn hẳn.
            </p>
          }
        >
          <p>
            llama.cpp <code>dist</code>: <code>const double sum_tgt = sum_cum*rnd;</code>, then the first candidate with{' '}
            <code>sum_run &gt;= sum_tgt</code>. Its backend path compares cumulative probabilities with u, strictly.
          </p>
        </Translated>
      </Today>
      <Predict
        id="x1/X1.8/p-draw"
        prompt={
          <Translated vi={<>Bốn token còn lại có trọng số bằng nhau 1, 1, 1, 1 và u = 0.6. M5 rút chỉ số nào (đếm từ 0)?</>}>
            <>Four survivors with equal weights 1, 1, 1, 1 and u = 0.6. Which index does M5 draw (counting from 0)?</>
          </Translated>
        }
        unit={tr('index', 'chỉ số')}
      >
        {(v) => {
          const d = inverseCdf(P_DRAW.weights, P_DRAW.u);
          return (
            <p data-reveal={d.index}>
              <Translated
                vi={
                  <>
                    <b>{d.index}.</b> Đích 0.6 × 4 = {d.target.toFixed(1)}; các tổng bao gồm {d.inclusive.join(', ')}; tổng đầu tiên lớn
                    hơn hẳn là 3, ở chỉ số {d.index}. {v === d.index ? 'Đúng.' : v === 3 ? 'Chỉ số 3 là nơi tổng bằng 4; tổng 3 ở chỉ số 2 đã vượt 2.4.' : ''}
                  </>
                }
              >
                <>
                  <b>{d.index}.</b> Target 0.6 × 4 = {d.target.toFixed(1)}; inclusive sums {d.inclusive.join(', ')}; the first strictly
                  greater is 3, at index {d.index}. {v === d.index ? 'Right.' : v === 3 ? 'Index 3 is where the sum is 4; the sum 3 at index 2 already exceeds 2.4.' : ''}
                </>
              </Translated>
            </p>
          );
        }}
      </Predict>

      <h2><Translated vi="Cả hai đầu của khoảng">Both ends of the interval</Translated></h2>
      <Translated
        vi={
          <p>
            u của M5 nằm trong [0, 1 − 2⁻²⁴] và có thể đúng bằng 0. Với "lớn hơn hẳn", một token trọng số 0 không bao giờ được chọn, kể cả
            khi u = 0: tổng bao gồm của nó bằng tổng ngay trước, vốn không vượt đích. Với "lớn hơn hoặc bằng" thì nó có thể
            bị chọn. Host của llama.cpp dùng ≥ với u × total, còn backend của nó dùng &gt; với u; chúng khác nhau khi hòa chính xác.
          </p>
        }
      >
        <p>
          M5's u lies in [0, 1 − 2⁻²⁴] and can be exactly 0. With "strictly greater", a token of weight 0 is never chosen, even
          at u = 0: its inclusive sum equals the one before it, which did not exceed the target. With "greater or equal" it can
          be. llama.cpp's host uses ≥ with u × total, its backend &gt; with u; they differ on an exact tie.
        </p>
      </Translated>
      <Ask q={questionsSX18['x1/X1.8/q-ge-gt']} />
      <Translated
        vi={
          <p>
            Đầu kia gây rắc rối cho phép chuyển đổi sang (0, 1]. <code>curand_uniform</code> có thể trả về đúng 1.0, và một tổng float của
            các xác suất có thể làm tròn xuống ngay dưới 1, nên không gì thỏa. u của M5 không bao giờ chạm 1, nhưng nó vẫn giữ một phương án dự phòng:
            tổng của kernel và phép scan của nó cộng cùng các trọng số theo thứ tự khác nhau, và tại u = 1 − 2⁻²⁴ tổng scan cuối cùng có thể
            kết thúc ở mức bằng hoặc dưới u × total. Phương án dự phòng là token còn lại cuối cùng có trọng số dương.
          </p>
        }
      >
        <p>
          The other end bites a (0, 1] conversion. <code>curand_uniform</code> can return exactly 1.0, and a float sum of
          probabilities can round to just under 1, so nothing qualifies. M5's u never reaches 1, but it keeps a fallback anyway:
          the kernel's total and its scan add the same weights in different orders, and at u = 1 − 2⁻²⁴ the last scanned sum can
          end at or below u × total. The fallback is the last survivor with a positive weight.
        </p>
      </Translated>
      <Ask q={questionsSX18['x1/X1.8/q-u1']} />

      <h2><Translated vi="Một khối cho mỗi hàng">One block per row</Translated></h2>
      <Translated
        vi={
          <p>
            Kernel giao cho mỗi hàng một khối. Giai đoạn 1 là phép rút gọn hai tầng của chương 10: mỗi luồng cộng một lát theo bước nhảy (stride),
            mỗi warp cộng các luồng của nó bằng một cây shuffle, và warp 0 cộng các tổng warp. Luồng 0 ghi tổng vào bộ nhớ chia sẻ
            (shared memory); sau rào chắn (barrier) A mọi luồng đọc nó và lập đích của mình; rào chắn B cho phép dùng lại mảng.
          </p>
        }
      >
        <p>
          The kernel gives each row one block. Phase 1 is chapter 10's two-stage reduction: each thread sums a strided slice,
          each warp sums its threads with a shuffle tree, and warp 0 sums the warp sums. Thread 0 writes the total to shared
          memory; after barrier A every thread reads it and forms its target; barrier B lets the array be reused.
        </p>
      </Translated>
      <Source section="10.8" pages="241–244" />
      <Translated
        vi={
          <p>
            Giai đoạn 2 đi qua hàng theo các đoạn gồm blockDim × 4 token còn lại, bốn token còn lại liên tiếp cho mỗi luồng. Với mỗi đoạn,
            khối tính tổng của đoạn (mỗi warp scan tổng của các luồng của nó bằng shuffle; lane cuối ghi tổng của warp
            vào bộ nhớ chia sẻ; mọi luồng cộng các tổng đó). Một đoạn để tổng gộp đang chạy ở mức bằng hoặc dưới đích thì bị
            bỏ qua. Đoạn vượt qua đích là đoạn duy nhất được scan đến cuối: mỗi luồng cộng tổng gộp và mọi thứ
            trước nó vào các tổng bao gồm của riêng nó và đánh cờ mọi token còn lại có tổng vượt đích.
          </p>
        }
      >
        <p>
          Phase 2 walks the row in chunks of blockDim × 4 survivors, four consecutive survivors per thread. For each chunk the
          block computes the chunk's sum (each warp scans its threads' totals with shuffles; the last lane writes the warp total
          to shared memory; every thread adds those). A chunk that leaves the running aggregate at or below the target is
          skipped. The chunk that crosses it is the only one scanned to the end: each thread adds the aggregate and everything
          before it to its own inclusive sums and flags every survivor whose sum exceeds the target.
        </p>
      </Translated>
      <Source section="11.4" pages="261–265" />
      <Translated
        vi={
          <p>
            Rồi đến phép bỏ phiếu. Một ballot cho mỗi warp một mặt nạ các lane đang giữ cờ; bit bật thấp nhất của nó là lane có cờ
            đầu tiên của warp; một <code>atomicMin</code> trên một chỉ số chung giữ lại cái thấp nhất giữa các warp. Đó là phép bầu
            leader của chương 12, chọn cái thấp nhất thay vì bất kỳ cái nào.
          </p>
        }
      >
        <p>
          Then the vote. A ballot gives each warp a mask of the lanes holding a flag; its lowest set bit is the warp's first
          flagged lane; an <code>atomicMin</code> on a shared index keeps the lowest across warps. That is chapter 12's leader
          election, choosing the lowest instead of any.
        </p>
      </Translated>
      <NoteGroup>
        <Source section="12.3" pages="291–295" />
        <Today checked="2026-11-20 (notes/today/x1-facts.md A6)">{/* i18n-ignore: citation */}
          <Translated
            vi={
              <p>
                <code>SamplingFromProbKernel</code> của FlashInfer có hình dạng này: một khối cho mỗi hàng, các đoạn gồm{' '}
                <code>BLOCK_THREADS * VEC_SIZE</code>, một đoạn chỉ được scan nếu <code>aggregate + aggregate_local &gt; u</code>, rồi{' '}
                <code>BlockScan</code> và lane đầu tiên vượt qua u.
              </p>
            }
          >
            <p>
              FlashInfer's <code>SamplingFromProbKernel</code> has this shape: one block per row, chunks of{' '}
              <code>BLOCK_THREADS * VEC_SIZE</code>, a chunk scanned only if <code>aggregate + aggregate_local &gt; u</code>, then{' '}
              <code>BlockScan</code> and the first lane past u.
            </p>
          </Translated>
        </Today>
      </NoteGroup>
      <Translated
        vi={
          <p>
            Trình chạy từng bước chạy kernel này trên một hàng. Hãy chạy từng bước qua hàng 200 token còn lại trước: hai đoạn bị bỏ qua chỉ nhờ
            tổng của chúng và chỉ đoạn thứ ba được scan. Rồi dùng công tắc rào chắn để bỏ một trong hai rào chắn quanh
            lệnh đọc tổng, và chọn thứ tự chạy của các luồng.
          </p>
        }
      >
        <p>
          The stepper runs this kernel on one row. Step through the 200-survivor row first: two chunks are skipped on their
          sums alone and only the third is scanned. Then use the barrier switch to remove one of the two barriers around the
          read of the total, and pick the order in which the threads run.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một khối 16 luồng với độ rộng warp do bạn chọn để hiển thị; float32 được mô phỏng bằng cách làm tròn sau mỗi phép cộng. Các
              thứ tự là lịch chạy theo thứ tự và đảo ngược của trình giả lập của khóa học (mỗi luồng chạy tới rào chắn hoặc shuffle kế tiếp),
              cộng thêm một thứ tự mà GPU có thể tạo ra. Không mô phỏng thời gian.
            </p>
          }
        >
          <p>
            One block of 16 threads with a warp width you choose for display; float32 imitated by rounding after every add. The
            orders are the course emulator's in-order and reversed schedules (each thread runs to its next barrier or shuffle),
            plus one a GPU may produce. No timing.
          </p>
        </Translated>
      </Simplification>
      <FindUStepper />
      <Translated
        vi={
          <p>
            Hai rào chắn kẹp lệnh đọc tổng, và chúng hỏng theo cách khác nhau. Bỏ A thì các luồng của warp khác đọc
            scratch_s[0] trước khi luồng 0 ghi tổng vào đó; chúng thấy tổng từng phần của warp 0, nhắm quá thấp, và trên
            hàng 120 token còn lại khối ghi ra sai token ở cả hai thứ tự của trình giả lập. Bỏ B thì cả hai thứ tự của trình giả lập vẫn
            qua: mọi luồng đọc tổng trước lần shuffle đầu tiên, và không ai ghi một tổng warp cho đến khi cả một warp đã
            qua phép scan của nó. Chỉ một warp chạy vượt lên mới ghi đè lên tổng dưới chân các warp khác, điều mà GPU có thể làm hoặc không.
          </p>
        }
      >
        <p>
          Two barriers bracket the read of the total, and they fail differently. Remove A and the threads of other warps read
          scratch_s[0] before thread 0 has written the total there; they find warp 0's partial sum, aim too low, and on the
          120-survivor row the block writes the wrong token in both emulator orders. Remove B and both emulator orders still
          pass: every thread reads the total before its first shuffle, and nobody writes a warp total until a whole warp is
          through its scan. Only a warp that runs ahead overwrites the total under the others, which a GPU may or may not do.
        </p>
      </Translated>
      <Ask q={questionsSX18['x1/X1.8/q-barrier']} />
      <Translated
        vi={
          <p>
            Hoạt hình bên dưới đi qua trọn một lượt rút với cả hai rào chắn còn nguyên: đích, một đoạn bị bỏ qua, phép scan của
            đoạn vượt qua và phép bỏ phiếu, rồi một lượt rút do việc làm tròn quyết định và hai đầu của khoảng.
          </p>
        }
      >
        <p>
          The animation below walks one whole draw with both barriers in place: the target, a skipped chunk, the scan of
          the crossing chunk and the vote, then a draw that rounding decides and the two ends of the interval.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một khối 8 luồng trong các warp hiển thị gồm 4 luồng, hai token còn lại mỗi luồng, các đoạn gồm 16, rút từ 40 token còn lại
              với T = 3 và u = 0.8; float32 được mô phỏng như trong trình chạy từng bước. Không mô phỏng thời gian.
            </p>
          }
        >
          <p>
            One block of 8 threads in display warps of 4, two survivors per thread, chunks of 16, drawing from 40 survivors
            at T = 3 with u = 0.8; float32 imitated as in the stepper. No timing.
          </p>
        </Translated>
      </Simplification>
      <FindUScene />

      <h2><Translated vi="Khớp chính xác giữa các backend">Exact agreement across backends</Translated></h2>
      <Translated
        vi={
          <p>
            u giống hệt nhau đến từng bit ở mọi nơi. Các tổng tích lũy thì không. <code>exp</code> khác nhau theo backend trong phạm vi sai số
            được công bố, một tổng float phụ thuộc vào thứ tự các phép cộng, CUDA mặc định gộp phép nhân-cộng, còn Metal có thể làm tròn
            về phía 0 và mặc định dùng fast math. Một lượt rút có đích nằm trong vài bước float quanh một ranh giới có thể chọn
            token kề bên trên một backend khác.
          </p>
        }
      >
        <p>
          u is bit-identical everywhere. The cumulative sums are not. <code>exp</code> differs by backend within its documented
          error, a float sum depends on its order of additions, CUDA contracts multiply-adds by default, and Metal may round
          toward zero and defaults to fast math. A draw whose target lies within a few float steps of a boundary can pick the
          neighbouring token on another backend.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x1-facts.md C5; ch10-facts A9, A11, B4, C4; ch05-facts C7)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              MSL: "Either round ties to even or round toward zero rounding mode may be supported." Phép cộng dấu phẩy động không có tính
              kết hợp, theo chính lời của NVIDIA.
            </p>
          }
        >
          <p>
            MSL: "Either round ties to even or round toward zero rounding mode may be supported." Floating-point addition is not
            associative, in NVIDIA's own words.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Vì vậy phép kiểm thử so token giữa các backend phân loại trước rồi mới so. Trên CPU, bằng double, nó tính biên của
            mỗi lượt rút, tức khoảng cách từ u × total tới ranh giới gần nhất. Nó chỉ đòi khớp chính xác với các lượt rút dứt khoát
            và chấp nhận token kề bên nào cũng được với phần còn lại. Trình khám phá cho thấy "phần còn lại" rộng đến đâu: bốn mươi token còn lại, ba thứ tự
            cộng, và một nhiễu trên các trọng số đóng vai <code>exp</code> của một backend khác.
          </p>
        }
      >
        <p>
          So the test that compares tokens across backends classifies before it compares. On the CPU, in double, it computes each
          draw's margin, the distance from u × total to the nearest boundary. It demands exact agreement only for decisive draws
          and accepts either neighbour for the rest. The explorer shows how wide "the rest" is: forty survivors, three orders
          of summation, and a perturbation of the weights standing in for another backend's <code>exp</code>.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Các tổng được làm tròn bằng Math.fround theo các thứ tự đã nêu. Khác biệt <code>exp</code> giữa các backend không được mô phỏng, ngoại trừ
              qua nhiễu bạn đặt, một thay đổi tương đối trên mỗi trọng số được rút từ Philox.
            </p>
          }
        >
          <p>
            Sums are rounded with Math.fround in the stated orders. Backend <code>exp</code> differences are not modelled except
            as the perturbation you set, a relative change per weight drawn from Philox.
          </p>
        </Translated>
      </Simplification>
      <BoundaryExplorer />
      <Ask q={questionsSX18['x1/X1.8/q-risk']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Cùng phân phối, khác token">Same distribution, different tokens</Translated></h2>
      <Translated
        vi={
          <p>
            CDF ngược không phải là sampler chính xác duy nhất. <code>torch.multinomial(probs, 1)</code>, lệnh mà{' '}
            <code>generate</code> của transformers gọi, chạy một cuộc đua mũ: rút q ~ Exp(1) cho mỗi token và lấy argmax(p / q). Hàm{' '}
            <code>categorical</code> của MLX dùng CDF ngược cho một phân phối có sample shape và Gumbel-max, argmax(l + g),
            trong các trường hợp khác. llama.cpp, FlashInfer và MLX ở batch 1 dùng CDF ngược.
          </p>
        }
      >
        <p>
          The inverse CDF is not the only exact sampler. <code>torch.multinomial(probs, 1)</code>, which transformers'{' '}
          <code>generate</code> calls, runs an exponential race: draw q ~ Exp(1) per token and take argmax(p / q). MLX's{' '}
          <code>categorical</code> uses an inverse CDF for one distribution with a sample shape and Gumbel-max, argmax(l + g),
          otherwise. llama.cpp, FlashInfer and MLX at batch 1 use the inverse CDF.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x1-facts.md D4, D5, D7)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              PyTorch: "s = argmax( p / q ) where q ~ Exp(1)", trên CPU, CUDA, MPS và XPU. Nhánh CDF ngược của MLX có chú thích "O(N
              + M) in memory, where the Gumbel-max trick needs O(N * M)".
            </p>
          }
        >
          <p>
            PyTorch: "s = argmax( p / q ) where q ~ Exp(1)", on CPU, CUDA, MPS and XPU. MLX's inverse-CDF path is commented "O(N
            + M) in memory, where the Gumbel-max trick needs O(N * M)".
          </p>
        </Translated>
      </Today>
      <Ask q={questionsSX18['x1/X1.8/q-multinomial']} />
      <Translated
        vi={
          <p>
            Trình khám phá đưa cùng các số đều từ Philox cho cả ba phương pháp: số đều 0 cho CDF ngược, mỗi token một số đều
            cho cuộc đua và cho Gumbel-max.
          </p>
        }
      >
        <p>
          The explorer feeds the same Philox uniforms to all three methods: uniform 0 to the inverse CDF, one uniform per token
          to the race and to Gumbel-max.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Độ chính xác double; q = −ln(1 − u) và g = −ln(−ln u) từ các số đều của M5. Hàm exponential của PyTorch được tính khác nhau
              trên mỗi thiết bị; điều đó không được mô phỏng.
            </p>
          }
        >
          <p>
            Double precision; q = −ln(1 − u) and g = −ln(−ln u) from M5's uniforms. PyTorch's exponential is computed differently
            on each device; that is not modelled.
          </p>
        </Translated>
      </Simplification>
      <SamplersExplorer />
      <Translated
        vi={
          <p>
            Cả ba hội tụ về cùng các tần suất; xét từng lượt rút thì chúng trùng nhau ít hơn nhiều. Đây là thêm một lý do khiến "cùng
            seed, cùng token" không bao giờ đúng giữa các framework. Giờ hãy tích vào ô: với q = −ln u, cuộc đua chọn đúng token của Gumbel-max ở
            mọi lượt rút, vì argmax(p / q) = argmax(ln p − ln q), và −ln q = −ln(−ln u) là một lượt rút Gumbel. Đó là đồng nhất thức
            mà chú thích của PyTorch dùng để suy ra cuộc đua; cách mỗi bên biến các bit thành q hay g quyết định bạn nhận token nào.
          </p>
        }
      >
        <p>
          All three converge to the same frequencies; draw by draw they agree far less often. This is one more reason "same
          seed, same token" never holds across frameworks. Now tick the box: with q = −ln u the race picks Gumbel-max's token on
          every draw, because argmax(p / q) = argmax(ln p − ln q), and −ln q = −ln(−ln u) is a Gumbel draw. That is the identity
          PyTorch's comment derives the race from; how each one turns bits into q or g decides which token you get.
        </p>
      </Translated>
      <Ask q={questionsSX18['x1/X1.8/q-mlx-batch']} />

      <h2><Translated vi="Ai đang dùng nó">Who ships it</Translated></h2>
      <Translated
        vi={
          <p>
            Kernel của FlashInfer, thứ vLLM có thể gọi trên CUDA, chính là thiết kế này trong môi trường production, với{' '}
            <code>curand_init(seed, row, offset)</code> cho u. Nhánh backend thử nghiệm của llama.cpp dựng phép rút từ các phép toán
            đồ thị: softmax, một cumsum bao gồm, một hàm bậc tại u, và đếm số phần tử phía trên nó. u của nó được rút trên host và
            tải lên. Theo cách chúng tôi đọc, một số double ngay dưới 1 bị làm tròn thành 1.0f trong lần tải lên đó khiến nó chọn ứng viên cuối,
            có thể là một ứng viên đã bị che, khoảng một lần trong 2²⁵ lượt rút.
          </p>
        }
      >
        <p>
          FlashInfer's kernel, which vLLM can call on CUDA, is this design in production, with{' '}
          <code>curand_init(seed, row, offset)</code> for u. llama.cpp's experimental backend path builds the draw from graph
          ops: softmax, an inclusive cumsum, a step at u, and a count of entries above it. Its u is drawn on the host and
          uploaded. By our reading, a double just below 1 that rounds to 1.0f in that upload makes it pick the last candidate,
          which can be a masked one, about once in 2²⁵ draws.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x1-facts.md A6, D1, D3)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              llama.cpp: <code>ggml_soft_max</code> → <code>ggml_cumsum</code> → <code>ggml_step</code> → <code>ggml_sum</code>;
              "We sample in double precision and cast to float". Lấy mẫu trên backend mặc định tắt khi sinh văn bản.
            </p>
          }
        >
          <p>
            llama.cpp: <code>ggml_soft_max</code> → <code>ggml_cumsum</code> → <code>ggml_step</code> → <code>ggml_sum</code>;
            "We sample in double precision and cast to float". Backend sampling is off by default for generation.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsSX18['x1/X1.8/q-device-u']} />

      <h2><Translated vi="Trên HIP và Metal">On HIP and Metal</Translated></h2>
      <Translated
        vi={
          <p>
            Phép bỏ phiếu là nơi các backend khác nhau. Ballot của HIP là 64 bit trên mọi target, nên mặt nạ là kiểu 64 bit và
            lane đầu tiên đến từ <code>__ffsll</code>; <code>__ffs</code> chỉ thấy 32 lane thấp. Trên Metal,{' '}
            <code>simd_ballot</code> trả về một phiếu 64 bit có các bit cao không xác định trên một SIMD group hẹp hơn; lane có cờ đầu tiên
            là <code>ctz</code> của phiếu, giá trị không bao giờ chạm tới các bit đó khi có lane nào đã bỏ phiếu.{' '}
            <code>simd_prefix_inclusive_sum</code> scan một SIMD group trong một lệnh gọi, theo một thứ tự mà đặc tả không nêu.
          </p>
        }
      >
        <p>
          The vote is where the backends differ. HIP's ballot is 64 bits on every target, so the mask is a 64-bit type and the
          first lane comes from <code>__ffsll</code>; <code>__ffs</code> would see only the low 32 lanes. On Metal,{' '}
          <code>simd_ballot</code> returns a 64-bit vote whose top bits are undefined on a narrower SIMD-group; the first flagged
          lane is <code>ctz</code> of the vote, which never reaches those bits when some lane voted.{' '}
          <code>simd_prefix_inclusive_sum</code> scans a SIMD-group in one call, in an order the spec does not state.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch12-facts.md B1, C2; ch11-facts.md C1, C5; infra-facts B3)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              <code>ctz(x)</code>: "If x is 0, returns the size in bits of the type of x". Một threadgroup cho mỗi hàng nghĩa là không
              threadgroup nào phải chờ threadgroup khác; kiểu chờ đó (một look-back) là điều Metal không ghi là an toàn.
            </p>
          }
        >
          <p>
            <code>ctz(x)</code>: "If x is 0, returns the size in bits of the type of x". One threadgroup per row means no
            threadgroup ever waits for another; such waiting (a look-back) is what Metal does not document as safe.
          </p>
        </Translated>
      </Today>
      {cu ? (
        <Code
          tabs={[
            {
              label: tr('CUDA and HIP', 'CUDA và HIP'),
              code: cu,
              status: tr(
                'emulator-checked and front-end-checked; unverified on CUDA and HIP hardware',
                'emulator-checked và front-end-checked; unverified trên phần cứng CUDA và HIP',
              ),
              highlight: barrierLines,
            },
          ]}
          caption={
            <Translated
              vi={
                <>
                  <code>labs/x1/snippets/draw_search.cu</code>, vòng lặp theo đoạn và phép bỏ phiếu đúng như bản tham chiếu của lab X1.3; rào chắn A
                  và B được tô sáng. <code>draw.test.ts</code> chạy nó trên trình giả lập, lần lượt bỏ từng rào chắn, và
                  so các token của nó với mô hình.
                </>
              }
            >
              <>
                <code>labs/x1/snippets/draw_search.cu</code>, the chunk loop and the vote as lab X1.3's reference has them; barriers A
                and B are highlighted. <code>draw.test.ts</code> runs it on the emulator, with each barrier removed in turn, and
                compares its tokens with the model.
              </>
            </Translated>
          }
        />
      ) : (
        <Translated
          vi={<p className="muted">Đoạn mã (<code>labs/x1/snippets/draw_search.cu</code>) sẽ hiện ở đây khi đã tải xong.</p>}
        >
          <p className="muted">The snippet (<code>labs/x1/snippets/draw_search.cu</code>) appears here once loaded.</p>
        </Translated>
      )}
      <Translated
        vi={
          <p>
            Lab X1.3 trên <a href={href('x1', 'labs')}>trang lab</a> lắp ráp toàn bộ sampler, mốc dự án M5, và kiểm thử nó
            giữa các backend bằng quy tắc biên. Lab X1.4 cài sẵn các lỗi kiểu này trong một bộ lọc gộp để bạn tìm.
          </p>
        }
      >
        <p>
          Lab X1.3 on the <a href={href('x1', 'labs')}>labs page</a> assembles the whole sampler, milestone M5, and tests it
          across backends with the margin rule. Lab X1.4 plants defects of this kind in a fused filter for you to find.
        </p>
      </Translated>
    </>
  );
}
