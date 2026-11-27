import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Bench, TierLegend } from '../../components/notes';
import {
  DEFECTS,
  LANE_LABEL,
  OUTLINE,
  TOY,
  WARP_LABEL,
  exactRow,
  fmtv,
  makeProblem,
  panels,
  rowOk,
  runBlock,
  runKernel,
  withDefects,
  type DefectId,
  type KConfig,
  type KStep,
  type LaneSchedule,
  type Op,
  type WarpSchedule,
} from '../../sim/flashkernel';
import { FlashBlockView } from './FlashBlockView';
import { Translated, tr, useLocale } from '../../lib/i18n';

// "Step one block": the flash kernel of Figs. 20.8–20.16 as lab 20.2 corrects it, at a toy
// size, with its barriers and the book's slips switchable. Every value comes from
// src/sim/flashkernel.ts.

const LANES: LaneSchedule[] = ['lockstep', 'apart'];
const WARPS: WarpSchedule[] = ['together', 'ahead'];
const SCHEDULES = LANES.flatMap((l) => WARPS.map((w) => [l, w] as const));

/** Which outline line a step belongs to. */
function outlineIndex(s: KStep | undefined): number {
  if (!s) return -1;
  const op: Op = s.op === 'release_syncthreads' ? 'syncthreads' : s.op === 'release_syncwarp' ? 'syncwarp' : s.op;
  if (op === 'syncthreads') return s.barrier === 'end' ? 8 : 2;
  return OUTLINE.findIndex((o) => o.op === op);
}

function Outline({ cur, cfg }: { cur: KStep | undefined; cfg: KConfig }) {
  const hl = outlineIndex(cur);
  return (
    <ol className="mono" style={{ listStyle: 'none', padding: '0.5rem 0.6rem', margin: 0, background: 'var(--cell)', border: '1px solid var(--rule)', borderRadius: 4, fontSize: '0.74rem', lineHeight: 1.7 }} data-outline>
      <li className="muted">for (i = blockIdx.x; i &lt; T_r; i += gridDim.x)</li>
      {OUTLINE.map((o, k) => {
        const removed = (o.op === 'syncwarp' && !cfg.syncwarp) || (k === 8 && !cfg.barrierAfter);
        const header =
          k === 1 ? <li className="muted" style={{ paddingLeft: '1.2em' }}>for (j : tiles{/* i18n-ignore: pseudocode */}{cfg.order === 'rtl' ? tr(', right to left', ', từ phải sang trái') : ''})</li> :
          k === 3 ? <li className="muted" style={{ paddingLeft: '2.4em' }}>for (ii = 0; ii &lt; B_r_warp; ii++)</li> : null;
        return (
          <FragmentLine key={k} header={header}>
            <li
              data-line={k}
              className={hl === k ? 'cur' : ''}
              style={{
                paddingLeft: `${1.2 * (o.depth + 1)}em`,
                background: hl === k ? 'var(--ink)' : undefined,
                color: hl === k ? 'var(--page)' : removed ? 'var(--fail)' : undefined,
                textDecoration: removed ? 'line-through' : undefined,
                borderRadius: 2,
              }}
            >
              {o.text}
              {removed ? tr(' (removed)', ' (đã bỏ)') : ''}
              {k === 4 && cfg.returnsNewMax ? <span style={{ color: hl === k ? undefined : 'var(--fail)' }}> <Translated vi="trả về giá trị lớn nhất mới">returns the new max</Translated></span> : null}
            </li>
          </FragmentLine>
        );
      })}
    </ol>
  );
}
function FragmentLine({ header, children }: { header: ReactNode; children: ReactNode }) {
  return (
    <>
      {header}
      {children}
    </>
  );
}

export function FlashStepper() {
  const [n, setN] = useState(12);
  const [brWarp, setBrWarp] = useState(2);
  const [bc, setBc] = useState(4);
  const [panel, setPanel] = useState(1);
  const [lanes, setLanes] = useState<LaneSchedule>('lockstep');
  const [warps, setWarps] = useState<WarpSchedule>('together');
  const [defects, setDefects] = useState<DefectId[]>([]);
  const [at, setAt] = useState(0);
  const [playing, setPlaying] = useState(false);

  const cfg = useMemo(() => withDefects({ ...TOY, n, brWarp, bc }, defects), [n, brWarp, bc, defects]);
  const prob = useMemo(() => makeProblem(n, TOY.d), [n]);
  const nPanels = panels(cfg);
  const p = Math.min(panel, nPanels - 1);
  const locale = useLocale();
  const trace = useMemo(() => runBlock(cfg, prob, p, lanes, warps), [cfg, prob, p, lanes, warps, locale]);
  const total = trace.steps.length;
  const k = Math.min(at, total);
  const cur = k > 0 ? trace.steps[k - 1] : undefined;
  const snap = trace.snaps[k];
  const firstBad = trace.violations.length ? trace.violations[0].step + 1 : null;
  const BR = cfg.warps * cfg.brWarp;

  useEffect(() => { setAt(0); setPlaying(false); }, [cfg, p, lanes, warps]);
  useEffect(() => {
    if (!playing) return;
    if (k >= total) return setPlaying(false);
    const id = window.setTimeout(() => setAt((x) => x + 1), 380);
    return () => window.clearTimeout(id);
  }, [playing, k, total]);

  const jump = (pred: (s: KStep) => boolean) => {
    for (let i = k; i < total; i++) if (pred(trace.steps[i])) return setAt(i + 1);
    setAt(total);
  };
  // Just past the first load of the next tile; once no tile is left, to the end.
  const nextTile = () => {
    const t = cur?.tile;
    for (let i = k; i < total; i++) {
      const s = trace.steps[i];
      if (s.op === 'load_KT_and_V' && s.tile !== t) return setAt(i + 1);
      if (s.op === 'store_O') break;
    }
    setAt(total);
  };
  const toggle = (id: DefectId) => setDefects((d) => (d.includes(id) ? d.filter((x) => x !== id) : [...d, id]));

  const storedRows = Object.keys(snap.stored).map(Number);
  const allOk = storedRows.length > 0 && storedRows.every((r) => rowOk(prob, r, snap.stored[r]));
  const finished = k >= total;

  return (
    <Bench
      title={tr('Step one block', 'Chạy từng bước một khối')}
      task={
        <Translated vi="Tìm nơi mỗi rào chắn có tác dụng: bật từng lỗi một và thử cả hai cách lập lịch.">
          Find where each barrier matters: switch on one defect at a time and try both schedules.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Một cấu hình đồ chơi: d = {cfg.d}, warp gồm {cfg.lanes} lane (không phải 32, để một warp vừa màn hình), {cfg.warps} warp,
              B_r = {BR}, B_c = {cfg.bc}. Di chuột lên một ô để xem giá trị của nó. <TierLegend />
            </>
          }
        >
          <>
            A toy configuration: d = {cfg.d}, warps of {cfg.lanes} lanes (not 32, so a warp fits on screen), {cfg.warps} warps,
            B_r = {BR}, B_c = {cfg.bc}. Hover a cell for its value. <TierLegend />
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <label className="field">
          <Translated vi="Số hàng N">Rows N</Translated>
          <select value={n} onChange={(e) => { setN(+e.target.value); setPanel(1); }} aria-label={tr('Rows N', 'Số hàng N')}>
            {[8, 10, 12].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className="field">{/* i18n-ignore: variable name */}
          B_r_warp
          <select value={brWarp} onChange={(e) => { setBrWarp(+e.target.value); setPanel(0); }} aria-label="B_r_warp">{/* i18n-ignore: variable name */}
            {[1, 2].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className="field">
          B_c
          <select value={bc} onChange={(e) => setBc(+e.target.value)} aria-label="B_c">
            {[2, 4, 6].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Khối (panel Q)">Block (Q panel)</Translated>
          <select value={p} onChange={(e) => setPanel(+e.target.value)} aria-label={tr('Block', 'Khối')}>
            {Array.from({ length: nPanels }, (_, i) => (
              <option key={i} value={i}>i = {i}: {tr('rows', 'hàng')} {i * BR} {tr('to', 'đến')} {Math.min(n, (i + 1) * BR) - 1}</option>
            ))}
          </select>
        </label>
        <label className="field">
          <Translated vi="Lane">Lanes</Translated>
          <select value={lanes} onChange={(e) => setLanes(e.target.value as LaneSchedule)} aria-label={tr('Lanes', 'Lane')}>
            {LANES.map((l) => <option key={l} value={l}>{LANE_LABEL[l]}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Warp">Warps</Translated>
          <select value={warps} onChange={(e) => setWarps(e.target.value as WarpSchedule)} aria-label={tr('Warps', 'Warp')}>
            {WARPS.map((w) => <option key={w} value={w}>{WARP_LABEL[w]}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }} data-defects>
        {DEFECTS.map((d) => (
          <label key={d.id} className="check" data-defect={d.id}>
            <input type="checkbox" checked={defects.includes(d.id)} onChange={() => toggle(d.id)} /> {d.label}
          </label>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setAt(0); }} disabled={k === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => { setPlaying(false); setAt(k - 1); }} disabled={k === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => { setPlaying(false); setAt(k + 1); }} disabled={finished}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => { setPlaying(false); jump((s) => s.op === 'release_syncthreads' || s.op === 'release_syncwarp'); }} disabled={finished}><Translated vi="Chạy tới rào chắn kế tiếp">Run to next barrier</Translated></button>
        <button onClick={() => { setPlaying(false); nextTile(); }} disabled={finished}><Translated vi="Ô kế tiếp">Next tile</Translated></button>
        <button onClick={() => setPlaying(!playing)} disabled={finished}>{playing ? tr('Pause', 'Tạm dừng') : tr('Play', 'Chạy')}</button>
        {firstBad ? <button onClick={() => { setPlaying(false); setAt(firstBad); }}><Translated vi="Tới lỗi đầu tiên">Go to the first failure</Translated></button> : null}
        <span className="muted" aria-live="polite" data-progress><Translated vi={<>bước {k} / {total}</>}>step {k} of {total}</Translated></span>
      </div>

      <div className={'narrate' + (cur?.violation ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }} data-narrate>
        {cur ? (
          <>
            {cur.text}
            {cur.violation ? <><br /><b className="t-fail"><Translated vi="Lỗi: ">Failure: </Translated>{cur.violation.message}.</b></> : null}
          </>
        ) : (
          <>
            <Translated
              vi={
                <>
                  Chưa có gì chạy. Khối {p} sở hữu các hàng {p * BR} đến {(p + 1) * BR - 1} của Q và O
                  {(p + 1) * BR > n ? ` (các hàng từ ${n} trở đi vượt quá N)` : ''}:{' '}
                  {Array.from({ length: cfg.warps }, (_, w) => `warp ${w} hàng ${p * BR + w * cfg.brWarp}${cfg.brWarp > 1 ? `–${p * BR + w * cfg.brWarp + cfg.brWarp - 1}` : ''}`).join(', ')}. Bấm Bước.
                </>
              }
            >
              Nothing has run yet. Block {p} owns rows {p * BR} to {(p + 1) * BR - 1} of Q and O
              {(p + 1) * BR > n ? ` (rows from ${n} on are past N)` : ''}:{' '}
              {Array.from({ length: cfg.warps }, (_, w) => `warp ${w} rows ${p * BR + w * cfg.brWarp}${cfg.brWarp > 1 ? `–${p * BR + w * cfg.brWarp + cfg.brWarp - 1}` : ''}`).join(', ')}. Press Step.
            </Translated>
          </>
        )}
      </div>

      <div style={{ overflowX: 'auto', marginBottom: '0.8rem' }}>
        <div style={{ minWidth: 560 }}>
          <FlashBlockView trace={trace} prob={prob} at={k} />
        </div>
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 4fr) minmax(0, 8fr)' }}>
        <div style={{ minWidth: 0 }}>
          <Outline cur={cur} cfg={cfg} />
          <Translated
            vi={
              <p className="muted" style={{ fontSize: '0.74rem', margin: '0.5rem 0 0' }}>
                Cấu trúc kernel của lab 20.2. Khi bật một lỗi, rào chắn bị bỏ sẽ bị gạch ngang.
              </p>
            }
          >
            <p className="muted" style={{ fontSize: '0.74rem', margin: '0.5rem 0 0' }}>
              The structure of lab 20.2's kernel. With a defect on, a removed barrier is struck through.
            </p>
          </Translated>
        </div>
        <div style={{ minWidth: 0, overflowX: 'auto' }}>
          <table className="data" data-rows style={{ margin: 0, background: 'var(--page)' }}>
            <thead>
              <tr>
                <th><Translated vi="hàng">row</Translated></th>
                <th>m</th>
                <th>D</th>
                <th><Translated vi="các ô đã trộn">tiles merged</Translated></th>
                <th><Translated vi="Õ/D so với giá trị chính xác trên các ô đó">Õ/D vs exact over those tiles</Translated></th>
                <th><Translated vi="đã ghi">stored</Translated></th>
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: BR }, (_, lr) => {
                const row = p * BR + lr;
                const w = Math.floor(lr / cfg.brWarp);
                const ii = lr % cfg.brWarp;
                const per = snap.merged[lr];
                const same = per.every((t) => t.length === per[0].length && t.every((x, i) => x === per[0][i]));
                const reg0 = snap.regs[w * cfg.lanes];
                const est = Array.from({ length: cfg.d }, (_, dd) => {
                  const r = snap.regs[w * cfg.lanes + (dd % cfg.lanes)];
                  return r.O[ii][Math.floor(dd / cfg.lanes)] / r.D[ii];
                });
                const cols = per[0].flatMap((t) => Array.from({ length: cfg.bc }, (_, q) => t * cfg.bc + q));
                const ex = row < n ? exactRow(prob, row, cols) : null;
                const err = ex ? Math.max(...est.map((x, i) => Math.abs(x - ex[i]))) : null;
                const st = snap.stored[row];
                return (
                  <tr key={lr} data-row={row}>
                    <td>{row}{row >= n ? tr(' (past N)', ' (quá N)') : ''}</td>
                    <td className="t-reg">{fmtv(reg0.m[ii])}</td>
                    <td className="t-reg">{fmtv(reg0.D[ii])}</td>
                    <td>{same ? (per[0].length ? per[0].join(', ') : tr('none', 'không có')) : per.map((t, L) => `lane ${L}: ${t.join(',') || '–'}`).join('; ')}</td>
                    <td>
                      {row >= n ? <span className="muted"><Translated vi="không ghi">not stored</Translated></span>
                        : !same ? <span className="muted"><Translated vi="các lane chưa đồng mức">lanes not level yet</Translated></span>
                        : ex == null ? <span className="muted"><Translated vi="chưa thấy gì">nothing visible yet</Translated></span>
                        : Number.isNaN(err!) ? <span className="chip fail" data-live="nan">NaN</span>
                        : <span className={'chip ' + (err! < 1e-9 ? 'ok' : 'fail')} data-live={err! < 1e-9 ? 'ok' : 'off'}>{err! < 1e-9 ? tr('equal', 'bằng nhau') : tr(`off by ${fmtv(err!)}`, `lệch ${fmtv(err!)}`)}</span>}
                    </td>
                    <td data-stored={st ? (rowOk(prob, row, st) ? 'ok' : 'wrong') : 'no'}>
                      {st ? (rowOk(prob, row, st) ? <span className="chip ok"><Translated vi="O và L đúng">O and L correct</Translated></span> : <span className="chip fail"><Translated vi="sai">wrong</Translated></span>) : <span className="muted"><Translated vi="chưa">not yet</Translated></span>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
      {finished ? (
        <p style={{ margin: '0.8rem 0 0' }} data-block-result={allOk ? 'ok' : 'wrong'}>
          {allOk ? <span className="chip ok"><Translated vi="Các hàng của khối này khớp với attention chính xác.">This block's rows match exact attention.</Translated></span> : <span className="chip fail"><Translated vi="Kết quả của khối này sai.">This block's results are wrong.</Translated></span>}{' '}
          {trace.violations.length ? tr(`${trace.violations.length} bad reads of shared memory.`, `${trace.violations.length} lần đọc sai bộ nhớ chia sẻ.`) : tr('No bad read of shared memory.', 'Không có lần đọc sai nào từ bộ nhớ chia sẻ.')}
        </p>
      ) : null}
    </Bench>
  );
}

/** Every defect against every schedule, for the whole 12-row kernel. */
export function FlashOutcomes() {
  const prob = useMemo(() => makeProblem(12, TOY.d), []);
  const rows: { label: string; on: DefectId[] }[] = [{ label: tr('corrected kernel (lab 20.2)', 'kernel đã sửa (lab 20.2)'), on: [] }, ...DEFECTS.map((d) => ({ label: d.label, on: [d.id] }))];
  return (
    <Bench
      title={tr('Every schedule at once', 'Mọi cách lập lịch cùng lúc')}
      task={
        <Translated vi="Trước khi đọc bảng: lỗi nào sai dưới mọi cách lập lịch, và lỗi nào chỉ sai dưới một cách?">
          Before reading the table: which defect is wrong under every schedule, and which only under one?
        </Translated>
      }
      foot={
        <Translated vi="Mỗi ô chạy cả ba khối của kernel đồ chơi (N = 12) trong mô hình ở trên và so O và L với attention chính xác.">
          Each cell runs all three blocks of the toy kernel (N = 12) in the model above and compares O and L with exact attention.
        </Translated>
      }
    >
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-outcomes>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}>Kernel{/* i18n-ignore: glossary keeps kernel */}</th>
              {SCHEDULES.map(([l, w]) => <th key={l + w}>{LANE_LABEL[l]}, {WARP_LABEL[w]}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.label} data-kernel={r.on[0] ?? 'correct'}>
                <td style={{ paddingLeft: '0.6rem' }}>{r.label}</td>
                {SCHEDULES.map(([l, w]) => {
                  const res = runKernel(withDefects(TOY, r.on), prob, l, w);
                  return (
                    <td key={l + w} data-outcome={res.correct ? 'correct' : 'wrong'}>
                      {res.correct ? <span className="chip ok"><Translated vi="đúng">correct</Translated></span> : <span className="chip fail"><Translated vi={<>{res.wrongRows.length} trên 12 hàng sai</>}>{res.wrongRows.length} of 12 rows wrong</Translated></span>}
                      {res.violations.length ? <div className="muted"><Translated vi={<>{res.violations.length} lần đọc sai</>}>{res.violations.length} bad reads</Translated></div> : null}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}
