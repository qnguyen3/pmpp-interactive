import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { solutionTabStatus } from '../../components/LabKit';
import { Translated, tr as trText } from '../../lib/i18n';
import { benchRows, fmt, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BISECT_PASSES,
  LAB_X11,
  STEP_PRESETS,
  costs,
  countStatus,
  hex,
  measuredPairs,
  prefixHex,
  selectTrace,
  statusAfter,
  trafficRatio,
  warpsOf,
  type GatherLane,
  type SelectTrace,
} from '../../sim/topkselect';
import { VOCAB_PRESETS } from '../../sim/nucleus';
import { RadixSelectScene } from './RadixSelectScene';

// Section X1.4: top-k as an MSD radix select. Everything the interactives show is computed by
// src/sim/topkselect.ts (selectTrace, costs, measuredPairs), pinned in topkselect.test.ts.

const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const MONO: CSSProperties = { fontFamily: 'var(--sans)', fontVariationSettings: "'MONO' 1" }; // i18n-ignore: CSS
const binHex = (b: number) => b.toString(16).toUpperCase().padStart(2, '0');
/** The shortest decimal that reads back as the same float32. */
const showLogit = (v: number) => {
  for (let p = 1; p <= 9; p++) {
    const s = v.toPrecision(p);
    if (Math.fround(Number(s)) === v) return String(Number(s));
  }
  return String(v);
};
const loaded = (t: string | null) => (t && !t.startsWith('// could not load') ? t : null);

/** Hex of a ballot: lane 0 is bit 0, the rightmost digit. */
function ballotHex(lanes: boolean[]): string {
  let s = '';
  for (let d = Math.ceil(lanes.length / 4) - 1; d >= 0; d--) {
    let v = 0;
    for (let j = 0; j < 4; j++) if (lanes[d * 4 + j]) v |= 1 << j;
    s += v.toString(16).toUpperCase();
  }
  return '0x' + s;
}

// ---------------------------------------------------------------- the stepper
type View =
  | { kind: 'count'; pass: number; round: number }
  | { kind: 'walk'; pass: number }
  | { kind: 'gather'; round: number }
  | { kind: 'done' };

function viewsOf(tr: SelectTrace): View[] {
  const R = Math.ceil(tr.n / tr.blockThreads);
  const v: View[] = [];
  tr.passes.forEach((p) => {
    for (let r = 0; r < R; r++) v.push({ kind: 'count', pass: p.index, round: r });
    v.push({ kind: 'walk', pass: p.index });
  });
  for (let r = 0; r < R; r++) v.push({ kind: 'gather', round: r });
  v.push({ kind: 'done' });
  return v;
}

function describe(tr: SelectTrace, v: View | undefined): string {
  const T = tr.blockThreads;
  if (!v)
    return trText(
      `Nothing has run yet. One block of ${T} threads owns this row of ${tr.n} logits and must find the ${tr.k} largest. Thread t reads columns t, t + ${T}, t + ${2 * T} and t + ${3 * T}. Press Step.`,
      `Chưa có gì chạy. Một khối ${T} luồng sở hữu hàng ${tr.n} logits này và phải tìm ${tr.k} giá trị lớn nhất. Luồng t đọc các cột t, t + ${T}, t + ${2 * T} và t + ${3 * T}. Bấm Bước.`,
    );
  if (v.kind === 'count') {
    const p = tr.passes[v.pass];
    const rd = p.rounds[v.round];
    const present = rd.lanes.filter((l) => l.col != null).length;
    const active = rd.lanes.filter((l) => l.match).length;
    const div = rd.warps.filter((w) => w.diverged).length;
    const who =
      v.pass === 0
        ? trText(
            `No bits are decided yet, so every lane's key matches and all ${active} add 1`,
            `Chưa bit nào được quyết định, nên khóa của mọi lane đều khớp và cả ${active} lane cộng 1`,
          )
        : trText(
            `${active} of ${present} lanes hold keys that start with the prefix ${prefixHex(p.prefixIn, p.maskIn)} and add 1`,
            `${active} trên ${present} lane giữ khóa bắt đầu bằng tiền tố ${prefixHex(p.prefixIn, p.maskIn)} và cộng 1`,
          );
    return trText(
      `Pass ${v.pass + 1}, round ${v.round + 1}: thread t reads column t + ${v.round * T}. ${who} to the bin of byte ${v.pass + 1} of their key, an atomic on the shared-memory histogram${active < present ? `; the other ${present - active} skip the branch` : ''}. ${div ? `${div} ${div === 1 ? 'warp splits' : 'warps split'} at the prefix test.` : 'No warp splits at the test.'}`,
      `Lượt ${v.pass + 1}, vòng ${v.round + 1}: luồng t đọc cột t + ${v.round * T}. ${who} vào bin của byte ${v.pass + 1} trong khóa của chúng, một atomic trên histogram ở bộ nhớ chia sẻ${active < present ? `; ${present - active} lane còn lại bỏ qua nhánh` : ''}. ${div ? `${div} warp bị tách ở phép kiểm tra tiền tố.` : 'Không warp nào bị tách ở phép kiểm tra.'}`,
    );
  }
  if (v.kind === 'walk') {
    const p = tr.passes[v.pass];
    const steps = p.walk.steps
      .map((s) =>
        trText(`bin ${binHex(s.bin)} holds ${s.count} (running ${s.running})`, `bin ${binHex(s.bin)} chứa ${s.count} (tích lũy ${s.running})`),
      )
      .join(', ');
    let s = trText(
      `Walk from bin FF down: ${steps}. The running count reaches the ${p.needIn} still needed in bin ${binHex(p.bin)}: the ${p.above} keys above it are in, the ${p.below} below it are out, and its ${p.inBin} keys are undecided, ${p.needOut} of them still needed.`,
      `Đi từ bin FF xuống: ${steps}. Số đếm tích lũy đạt ${p.needIn} khóa còn cần ở bin ${binHex(p.bin)}: ${p.above} khóa phía trên nó được chọn, ${p.below} khóa phía dưới bị loại, và ${p.inBin} khóa của nó chưa được quyết định, còn cần ${p.needOut} khóa trong số đó.`,
    );
    if (p.early)
      s += trText(
        ` That is the whole bin, so the set is decided after ${v.pass + 1} ${v.pass ? 'passes' : 'pass'}: the select stops here.`,
        ` Đó là toàn bộ bin, nên tập được quyết định sau ${v.pass + 1} lượt: phép chọn dừng ở đây.`,
      );
    else if (p.shift === 0)
      s += trText(
        ` After the fourth byte the prefix is a whole key: K* = ${hex(tr.prefix)}. ${tr.greater} keys lie above it, and ${tr.need} of the ${tr.equal} equal to it are needed.`,
        ` Sau byte thứ tư, tiền tố là cả một khóa: K* = ${hex(tr.prefix)}. ${tr.greater} khóa nằm trên nó, và cần ${tr.need} trong số ${tr.equal} khóa bằng nó.`,
      );
    else s += trText(` Pass ${v.pass + 2} refines only that bin.`, ` Lượt ${v.pass + 2} chỉ tinh chỉnh bin đó.`);
    return s;
  }
  if (v.kind === 'gather') {
    const g = tr.gather[v.round];
    const own = g.lanes.filter((l) => l.col != null);
    const n = (c: string) => own.filter((l) => l.cls === c).length;
    const ties = own.filter((l) => l.cls === 'tie');
    const tieText = ties.length
      ? trText(
          ` Ties get ranks ${ties.map((l) => l.tieRank).join(', ')} (ties before them in column order); a tie is kept while its rank is below the budget of ${tr.need}.`,
          ` Các khóa hòa nhận hạng ${ties.map((l) => l.tieRank).join(', ')} (số khóa hòa đứng trước chúng theo thứ tự cột); một khóa hòa được giữ khi hạng của nó nhỏ hơn ngân sách ${tr.need}.`,
        )
      : '';
    return trText(
      `Gather, round ${v.round + 1}: ${n('above')} keys above the threshold, ${n('tie')} equal to it, ${n('out')} below.${tieText} A kept lane's slot is the ${g.keptBefore} kept in earlier rounds plus the kept lanes before it in this round: a ballot and a popcount. This round writes ${g.keptAfter - g.keptBefore}.`,
      `Gom, vòng ${v.round + 1}: ${n('above')} khóa trên ngưỡng, ${n('tie')} khóa bằng ngưỡng, ${n('out')} khóa dưới.${tieText} Vị trí của một lane được giữ là ${g.keptBefore} khóa đã giữ ở các vòng trước cộng số lane được giữ đứng trước nó trong vòng này: một ballot và một popcount. Vòng này ghi ${g.keptAfter - g.keptBefore}.`,
    );
  }
  return trText(
    `Done: exactly ${tr.k} (column, logit) pairs, in ascending column order. That is the order of the columns, not of the values; a sampler that needs them by value sorts these ${tr.k}, not the row.`,
    `Xong: đúng ${tr.k} cặp (cột, logit), theo thứ tự cột tăng dần. Đó là thứ tự của các cột, không phải của các giá trị; một sampler cần chúng theo giá trị sẽ sắp xếp ${tr.k} cặp này, không phải cả hàng.`,
  );
}

function RadixStepper() {
  const [presetId, setPresetId] = useState('scene');
  const preset = STEP_PRESETS.find((p) => p.id === presetId)!;
  const [k, setK] = useState(preset.k);
  const [kText, setKText] = useState(String(preset.k));
  const [warp, setWarp] = useState(32);
  const [at, setAt] = useState(0);
  const tr = useMemo(() => selectTrace(preset.logits, k, 16, warp), [preset, k, warp]);
  const views = useMemo(() => viewsOf(tr), [tr]);
  const total = views.length;
  const cur = at > 0 ? views[Math.min(at, total) - 1] : undefined;
  useEffect(() => setAt((a) => Math.min(a, views.length)), [views.length]);

  const pick = (id: string) => {
    const p = STEP_PRESETS.find((x) => x.id === id)!;
    setPresetId(id);
    setK(p.k);
    setKText(String(p.k));
    setAt(0);
  };
  const setKFrom = (text: string) => {
    setKText(text);
    const v = Math.round(Number(text));
    if (text.trim() !== '' && v >= 1 && v <= tr.n) {
      setK(v);
      setAt(0);
    }
  };
  const nextPhase = () => {
    for (let i = at; i < total; i++) if (views[i].kind !== 'count' && views[i].kind !== 'gather') return setAt(i + 1);
    setAt(total);
  };

  const T = tr.blockThreads;
  const R = Math.ceil(tr.n / T);
  const passesDone = !cur ? 0 : cur.kind === 'count' ? cur.pass : cur.kind === 'walk' ? cur.pass + 1 : tr.passes.length;
  const pass = cur && (cur.kind === 'count' || cur.kind === 'walk') ? tr.passes[cur.pass] : undefined;
  const byte = cur?.kind === 'count' || cur?.kind === 'walk' ? cur.pass : null;
  const histNow = !cur || !pass ? null : cur.kind === 'count' ? pass.rounds[cur.round].histAfter : pass.hist;
  const gatherUpto = cur?.kind === 'gather' ? cur.round : cur?.kind === 'done' ? R - 1 : -1;
  const lanesByCol = new Map<number, GatherLane>();
  tr.gather.slice(0, gatherUpto + 1).forEach((g) => g.lanes.forEach((l) => l.col != null && lanesByCol.set(l.col, l)));
  const counts = countStatus(tr, passesDone);
  const groups = warpsOf(T, warp);

  const cellState = (col: number): string => {
    const r = Math.floor(col / T);
    if (cur?.kind === 'count' && r === cur.round) return pass!.rounds[r].lanes[col % T].match ? 'active' : 'idle';
    const g = lanesByCol.get(col);
    if (g) return g.keep ? 'kept' : g.cls === 'tie' ? 'dropped' : 'out';
    return statusAfter(tr, passesDone, col);
  };
  const cellStyle = (st: string, current: boolean): CSSProperties => ({
    padding: '0.1rem 0.3rem', // i18n-ignore: CSS
    lineHeight: 1.25,
    borderRadius: 3,
    border: `1px solid ${st === 'in' || st === 'kept' ? 'var(--ok)' : st === 'active' ? 'var(--ink)' : current ? 'var(--ink-3)' : 'var(--rule)'}`, // i18n-ignore: CSS
    background: st === 'active' ? 'var(--ink)' : st === 'in' || st === 'kept' ? 'var(--ok-wash)' : st === 'idle' ? 'var(--rule)' : 'var(--cell)',
    color: st === 'active' ? 'var(--page)' : 'var(--ink)',
    opacity: st === 'out' || st === 'dropped' ? 0.38 : st === 'idle' ? 0.6 : 1,
    textDecoration: st === 'dropped' ? 'line-through' : undefined,
    whiteSpace: 'nowrap',
  });
  const curRound = cur?.kind === 'count' || cur?.kind === 'gather' ? cur.round : null;

  return (
    <Bench
      title={trText('Radix select', 'Radix select')}
      task={
        <Translated vi="Đi từng bước qua lượt 1, rồi nạp các preset có chung các bit cao, và các trường hợp hòa.">
          Step through pass 1, then load the presets that share top bits, and the ties.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Một khối 16 luồng, 64 logits. Mỗi ô hiện cột, logit và khóa 32-bit của nó dạng hex; byte mà lượt hiện tại đếm
              được in đậm. Xanh: đã quyết định chọn. Mờ: đã quyết định loại. Đặc: một lane đang làm atomic ở vòng này. Xám:
              một lane bỏ qua nó. Các warp được nhóm theo độ rộng bạn chọn; các lane không có luồng bị bỏ ra.
            </>
          }
        >
          <>
            One block of 16 threads, 64 logits. Each cell shows the column, the logit and its 32-bit key in hex; the byte the
            current pass counts is in bold. Green: decided in. Faded: decided out. Solid: a lane doing the atomic this round.
            Grey: a lane that skips it. Warps are grouped at the width you pick; lanes with no thread are left out.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Hàng">Row</Translated>
          <span className="seg" role="group" aria-label={trText('Row', 'Hàng')} style={{ flexWrap: 'wrap' }}>
            {STEP_PRESETS.map((p) => (
              <button key={p.id} aria-pressed={presetId === p.id} onClick={() => pick(p.id)}>{p.label}</button>
            ))}
          </span>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi={<>k (1 đến {tr.n})</>}>k (1 to {tr.n})</Translated>
          <input type="number" min={1} max={tr.n} value={kText} aria-label="k" onChange={(e) => setKFrom(e.target.value)} />
        </label>
        <span style={FIELD}>
          <Translated vi="Độ rộng warp (hiển thị)">Warp width (display)</Translated>
          <span className="seg" role="group" aria-label={trText('Warp width', 'Độ rộng warp')}>
            {[8, 32, 64].map((w) => (
              <button key={w} aria-pressed={warp === w} onClick={() => setWarp(w)}>{w}</button>
            ))}
          </span>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setAt(0)} disabled={at === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => setAt(at - 1)} disabled={at === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => setAt(at + 1)} disabled={at >= total}><Translated vi="Bước">Step</Translated></button>
        <button onClick={nextPhase} disabled={at >= total}><Translated vi="Tới lần đi kế tiếp">To the next walk</Translated></button>
        <button onClick={() => setAt(total)} disabled={at >= total}><Translated vi="Chạy đến cuối">Run to the end</Translated></button>
        <span className="muted" aria-live="polite" data-step={at}>
          <Translated vi={<>bước {at} trên {total}</>}>step {at} of {total}</Translated>
        </span>
      </div>
      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }}>{describe(tr, cur)}</div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 2fr)' }}>
        <div style={{ overflowX: 'auto' }}>
          <table className="threads" aria-label={trText('Threads and the columns they read', 'Các luồng và các cột chúng đọc')} style={{ minWidth: '27rem' }}>
            <thead>
              <tr>
                <th style={{ width: '3rem' }}><Translated vi="luồng">thread</Translated></th>
                {Array.from({ length: R }, (_, r) => (
                  <th key={r} style={curRound === r ? { color: 'var(--ink)' } : { color: 'var(--ink-2)' }}>
                    {r === 0 ? trText('column t', 'cột t') : `t + ${r * T}`}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {groups.map((g) => {
                const threads = g.threads.filter((t): t is number => t != null);
                const rd = cur?.kind === 'count' ? pass!.rounds[cur.round].warps[g.warp] : undefined;
                return [
                  <tr key={'w' + g.warp} data-warp={g.warp} data-diverged={rd?.diverged ? '1' : '0'}>
                    <td colSpan={R + 1} className="muted" style={{ paddingTop: '0.4rem', whiteSpace: 'normal' }}>
                      <Translated
                        vi={
                          <>
                            warp {g.warp}: lane 0 đến {warp - 1}
                            {threads.length < warp ? `, chỉ có luồng ở lane 0 đến ${threads.length - 1}` : ''}
                            {rd ? (
                              <>
                                {' '}· vòng này {rd.active} trên {rd.present} lane hoạt động
                                {rd.diverged ? <b> · bị tách ở phép kiểm tra tiền tố</b> : null}
                              </>
                            ) : null}
                          </>
                        }
                      >
                        warp {g.warp}: lanes 0 to {warp - 1}
                        {threads.length < warp ? `, threads in lanes 0 to ${threads.length - 1} only` : ''}
                        {rd ? (
                          <>
                            {' '}· this round {rd.active} of {rd.present} lanes active
                            {rd.diverged ? <b> · splits at the prefix test</b> : null}
                          </>
                        ) : null}
                      </Translated>
                    </td>
                  </tr>,
                  ...threads.map((t) => (
                    <tr key={t}>
                      <td>{t}</td>
                      {Array.from({ length: R }, (_, r) => {
                        const col = r * T + t;
                        if (col >= tr.n) return <td key={r} />;
                        const st = cellState(col);
                        const g2 = lanesByCol.get(col);
                        const bytes = hex(tr.keys[col]).slice(2).match(/../g)!;
                        return (
                          <td key={r} data-col={col} data-state={st}>
                            <div style={cellStyle(st, curRound === r)}>
                              <span style={{ opacity: 0.6, fontSize: '0.68rem' }}>c{col} </span>
                              {showLogit(tr.logits[col])}
                              <br />
                              <span style={{ ...MONO, fontSize: '0.72rem' }}>
                                {bytes.map((b, i) => (
                                  <span key={i} style={{ fontWeight: byte === i ? 800 : 400, opacity: byte != null && i > byte ? 0.45 : 1 }}>{b}</span>
                                ))}
                              </span>
                              {g2?.keep ? <b className="t-ok" data-slot-of={col}> → {g2.slot}</b> : null}
                              {g2 && g2.cls === 'tie' ? <span className="muted"> <Translated vi={<>hòa {g2.tieRank}</>}>tie {g2.tieRank}</Translated></span> : null}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  )),
                ];
              })}
            </tbody>
          </table>
        </div>

        <div>
          <div className="panel" style={{ marginBottom: '0.8rem' }} data-part="status">
            <div>
              <Translated vi="Tiền tố đến giờ:">Prefix so far:</Translated>{' '}
              <b style={MONO} data-prefix="">{passesDone ? prefixHex(tr.passes[passesDone - 1].prefixOut, tr.passes[passesDone - 1].maskOut) : '0x········'}</b>
            </div>
            <div>
              <Translated vi="Còn cần từ bin chưa quyết định:">Still needed from the undecided bin:</Translated>{' '}
              <b data-need="">{passesDone ? tr.passes[passesDone - 1].needOut : tr.k}</b>
            </div>
            <div className="muted" data-counts={`${counts.in},${counts.out},${counts.open}`}>
              <Translated vi={<>đã chọn {counts.in}, đã loại {counts.out}, chưa quyết định {counts.open}</>}>
                decided in {counts.in}, out {counts.out}, undecided {counts.open}
              </Translated>
            </div>
            {cur?.kind === 'gather' || cur?.kind === 'done' ? (
              <div>
                <Translated
                  vi={
                    <>
                      Ngưỡng: {tr.exact ? <>K* = <b style={MONO}>{hex(tr.prefix)}</b></> : <>các khóa có {(tr.passes.length * 8)} bit cao là <b style={MONO}>{prefixHex(tr.prefix, tr.mask)}</b></>}
                      ; ngân sách hòa <b data-budget="">{tr.need}</b> trên {tr.equal}
                    </>
                  }
                >
                  Threshold: {tr.exact ? <>K* = <b style={MONO}>{hex(tr.prefix)}</b></> : <>keys whose top {(tr.passes.length * 8)} bits are <b style={MONO}>{prefixHex(tr.prefix, tr.mask)}</b></>}
                  ; tie budget <b data-budget="">{tr.need}</b> of {tr.equal}
                </Translated>
              </div>
            ) : null}
          </div>

          {histNow ? (
            <div className="panel" data-part="hist">
              <div className="mx-label t-shared">
                <Translated vi={<>Histogram của byte {pass!.index + 1} (bộ nhớ chia sẻ){cur?.kind === 'count' ? `, sau vòng ${cur.round + 1}` : ''}</>}>
                  Histogram of byte {pass!.index + 1} (shared memory){cur?.kind === 'count' ? `, after round ${cur.round + 1}` : ''}
                </Translated>
              </div>
              <div className="muted" style={{ fontSize: '0.75rem', marginBottom: '0.3rem' }}>
                <Translated vi="256 bin; chỉ hiện các bin không rỗng, cao nhất trước">256 bins; the nonempty ones, highest first</Translated>
              </div>
              <table className="threads">
                <tbody>
                  {histNow
                    .map((c, b) => [b, c] as const)
                    .filter(([, c]) => c > 0)
                    .reverse()
                    .map(([b, c]) => {
                      const w = cur?.kind === 'walk' ? pass!.walk.steps.find((s) => s.bin === b) : undefined;
                      const mark = cur?.kind === 'walk' ? (w ? (b === pass!.bin ? 'crossing' : 'in') : 'out') : '';
                      return (
                        <tr key={b} data-bin={b} data-count={c} data-mark={mark} style={{ opacity: mark === 'out' ? 0.45 : 1 }}>
                          <td style={MONO}><Translated vi={<>bin {binHex(b)}</>}>bin {binHex(b)}</Translated></td>
                          <td style={{ width: '45%' }}>
                            <span style={{ display: 'inline-block', height: 9, width: `${(c / Math.max(...histNow)) * 100}%`, minWidth: 2, background: b === pass!.bin && cur?.kind === 'walk' ? 'var(--ink)' : 'var(--shared)' }} />
                          </td>
                          <td className="num">{c}</td>
                          <td>
                            {w ? (
                              <Translated vi={<>tích lũy {w.running}{mark === 'crossing' ? <b> ← khóa thứ k</b> : ', được chọn'}</>}>
                                <>running {w.running}{mark === 'crossing' ? <b> ← k-th key</b> : ', in'}</>
                              </Translated>
                            ) : mark === 'out' ? trText('out', 'loại') : ''}
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
          ) : null}

          {cur?.kind === 'gather' ? (
            <div className="panel" data-part="ballots">
              <div className="mx-label">
                <Translated vi={<>Các ballot, vòng {cur.round + 1} (lane 0 là bit 0)</>}>Ballots, round {cur.round + 1} (lane 0 is bit 0)</Translated>
              </div>
              {tr.gather[cur.round].warps.map((w) => (
                <div key={w.warp} data-ballot-warp={w.warp} style={{ fontSize: '0.78rem' }}>
                  <Translated
                    vi={
                      <>
                        warp {w.warp}: hòa <span style={MONO} data-ties={ballotHex(w.tieBallot)}>{ballotHex(w.tieBallot)}</span>, giữ{' '}
                        <span style={MONO} data-kept={ballotHex(w.keepBallot)}>{ballotHex(w.keepBallot)}</span> (popcount {w.keepCount})
                      </>
                    }
                  >
                    warp {w.warp}: ties <span style={MONO} data-ties={ballotHex(w.tieBallot)}>{ballotHex(w.tieBallot)}</span>, kept{' '}
                    <span style={MONO} data-kept={ballotHex(w.keepBallot)}>{ballotHex(w.keepBallot)}</span> (popcount {w.keepCount})
                  </Translated>
                </div>
              ))}
              <div className="muted" style={{ fontSize: '0.75rem', marginTop: '0.3rem' }}>
                <Translated vi={<>số hòa trước vòng này {tr.gather[cur.round].tiesBefore}, số đã giữ trước vòng này {tr.gather[cur.round].keptBefore}</>}>
                  ties before this round {tr.gather[cur.round].tiesBefore}, kept before this round {tr.gather[cur.round].keptBefore}
                </Translated>
              </div>
            </div>
          ) : null}

          {cur?.kind === 'gather' || cur?.kind === 'done' ? (
            <div className="panel" style={{ marginTop: '0.8rem' }} data-part="out">
              <div className="mx-label t-global">
                <Translated vi={<>Đầu ra: {tr.k} cặp (cột, logit)</>}>Output: {tr.k} (column, logit) pairs</Translated>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
                {Array.from({ length: tr.k }, (_, s) => {
                  const o = tr.gather.slice(0, gatherUpto + 1).flatMap((g) => g.lanes).find((l) => l.slot === s);
                  return (
                    <span key={s} data-slot={s} data-col={o?.col ?? ''} className="mono"
                      style={{ minWidth: '2.2rem', textAlign: 'center', padding: '0 0.2rem', borderRadius: 3, fontSize: '0.75rem',
                        border: `1px ${o ? 'solid var(--ok)' : 'dashed var(--rule)'}`, background: o ? 'var(--ok-wash)' : 'transparent' }}>
                      {o ? `c${o.col}` : s}
                    </span>
                  );
                })}
              </div>
            </div>
          ) : null}
        </div>
      </div>
      {cur?.kind === 'done' ? (
        <p style={{ margin: '0.8rem 0 0' }} data-passes={tr.passes.length}>
          <Translated
            vi={
              <>
                <span className="chip ok">{tr.passes.length} lượt và một lần gom</span>{' '}
                {tr.passes.reduce((s, p) => s + p.divergedWarps, 0)} lượt-vòng warp bị tách ở phép kiểm tra tiền tố với độ rộng
                warp này.
              </>
            }
          >
            <span className="chip ok">{tr.passes.length} {tr.passes.length === 1 ? 'pass' : 'passes'} and one gather</span>{' '}
            {tr.passes.reduce((s, p) => s + p.divergedWarps, 0)} warp-rounds split at the prefix test at this warp width.
          </Translated>
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- cost explorer
const MB = (b: number) => (b >= 1e9 ? `${fmt(b / 1e9)} GB` : b >= 1e6 ? `${fmt(b / 1e6)} MB` : `${fmt(b / 1e3)} kB`); // i18n-ignore: units
const BATCHES = [1, 8, 64, 256];

function useMeasured() {
  const { reports } = useReports();
  return realBackends(reports).flatMap((b) =>
    measuredPairs(benchRows(reports[b], LAB_X11)).map((m) => ({ ...m, backend: b, device: String(reports[b].device.name ?? b) })),
  );
}

function CostExplorer() {
  const [V, setV] = useState(151936);
  const [kPos, setKPos] = useState(Math.round((Math.log(40) / Math.log(151936)) * 1000));
  const [batch, setBatch] = useState(64);
  const k = Math.max(1, Math.min(V, Math.round(Math.exp((kPos / 1000) * Math.log(V)))));
  const rows = costs(V, k, batch);
  const ratio = trafficRatio(V, k);
  const max = Math.max(...rows.map((r) => r.perRow));
  const measured = useMeasured();
  const { reports } = useReports();
  const sms = realBackends(reports).flatMap((b) => {
    const n = Number(reports[b].device.multiProcessorCount);
    return isFinite(n) && n > 0 ? [{ b, n, name: String(reports[b].device.name ?? b) }] : [];
  });
  return (
    <Bench
      title={trText('Passes and bytes against k and V', 'Số lượt và số byte theo k và V')}
      task={
        <Translated vi="Kéo k từ 1 lên hàng nghìn và theo dõi số byte của phép chọn; rồi đặt batch về 1.">
          Move k from 1 to thousands and watch the select's bytes; then set the batch to 1.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Số byte một thiết kế phải di chuyển cho mỗi hàng, được đếm chứ không đo; 1 MB = 10⁶ byte. Phép sắp xếp là của
              lab 14.3: bốn lượt 8 bit, mỗi lượt đọc khóa 4 byte để đếm và đọc rồi ghi cặp (khóa, chỉ số) 8 byte để phân tán.
              Phép chọn và phép chia đôi cộng thêm 8 byte cho mỗi cặp (cột, logit) được giữ.
            </>
          }
        >
          <>
            Bytes a design must move per row, counted, not measured; 1 MB = 10⁶ bytes. The sort is lab 14.3's: four 8-bit
            passes, each reading the 4-byte key to count and reading and writing the 8-byte (key, index) pair to scatter. The
            select and bisection add 8 bytes per kept (column, logit) pair.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi={<>Bộ từ vựng V = {V.toLocaleString('en-US')}</>}>Vocabulary V = {V.toLocaleString('en-US')}</Translated>
          <span className="seg" role="group" aria-label={trText('Vocabulary', 'Bộ từ vựng')}>
            {VOCAB_PRESETS.map((v) => (
              <button key={v} aria-pressed={V === v} onClick={() => setV(v)}>{v.toLocaleString('en-US')}</button>
            ))}
          </span>
        </span>
        <label className="field" style={{ flex: '1 1 12rem' }}>
          <Translated vi="hoặc bất kỳ V nào tới 262,144">or any V up to 262,144</Translated>
          <input type="range" min={1024} max={262144} step={1} value={V} aria-label="V" onChange={(e) => setV(Number(e.target.value))} style={{ width: '100%' }} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ flex: '1 1 14rem' }}>
          <Translated vi={<>k = {k.toLocaleString('en-US')} (thang log, 1 đến V)</>}>k = {k.toLocaleString('en-US')} (log scale, 1 to V)</Translated>
          <input type="range" min={0} max={1000} step={1} value={kPos} aria-label="k" onChange={(e) => setKPos(Number(e.target.value))} style={{ width: '100%' }} />
        </label>
        <span style={FIELD}>
          <Translated vi="Batch (số hàng)">Batch (rows)</Translated>
          <span className="seg" role="group" aria-label={trText('Batch', 'Batch')}>
            {BATCHES.map((b) => (
              <button key={b} aria-pressed={batch === b} onClick={() => setBatch(b)}>{b}</button>
            ))}
          </span>
        </span>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0, background: 'var(--page)', minWidth: '40rem' }} data-part="costs">
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}><Translated vi="Thiết kế">Design</Translated></th>
              <th className="num"><Translated vi="Số lượt">Passes</Translated></th>
              <th className="num"><Translated vi="Byte mỗi khóa">Bytes per key</Translated></th>
              <th className="num"><Translated vi="Mỗi hàng">Per row</Translated></th>
              <th style={{ width: '24%' }} />
              <th className="num"><Translated vi="Cả batch">Whole batch</Translated></th>
              <th className="num"><Translated vi="Số khối">Blocks</Translated></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} data-design={r.id} data-per-row={r.perRow} data-passes={r.passes}>
                <td style={{ paddingLeft: '0.6rem' }}>{r.label}</td>
                <td className="num">{r.passes}</td>
                <td className="num">{r.perKey}</td>
                <td className="num">{MB(r.perRow)}</td>
                <td><span style={{ display: 'inline-block', height: 10, width: `${(r.perRow / max) * 100}%`, background: 'var(--global)' }} /></td>
                <td className="num">{MB(r.total)}</td>
                <td className="num">{r.blocks}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ margin: '0.8rem 0 0' }} data-readout="ratio" data-ratio={ratio}>
        <Translated
          vi={
            <>
              Phép sắp xếp di chuyển <b>{fmt(ratio)}×</b> số byte của phép chọn ở k = {k.toLocaleString('en-US')}. Đầu ra
              của phép chọn là {MB(rows[1].outBytes)} trên tổng {MB(rows[1].perRow)} của nó cho mỗi hàng
              {rows[1].outBytes / rows[1].perRow < 0.01 ? ', dưới 1 phần trăm' : ''}. Phép chia đôi đọc hàng{' '}
              {BISECT_PASSES + 1} lần.
            </>
          }
        >
          The sort moves <b>{fmt(ratio)}×</b> the select's bytes at k = {k.toLocaleString('en-US')}. The select's output is{' '}
          {MB(rows[1].outBytes)} of its {MB(rows[1].perRow)} per row
          {rows[1].outBytes / rows[1].perRow < 0.01 ? ', under 1 percent' : ''}. Bisection reads the row {BISECT_PASSES + 1} times.
        </Translated>
      </p>
      <p style={{ margin: '0.5rem 0 0' }} data-readout="blocks">
        {batch === 1
          ? trText(
              'Batch 1: one block does all the work for every design here, on one multiprocessor; the rest of the device waits.',
              'Batch 1: một khối làm toàn bộ việc cho mọi thiết kế ở đây, trên một multiprocessor; phần còn lại của thiết bị ngồi chờ.',
            )
          : trText(`Batch ${batch}: ${batch} blocks, one per row.`, `Batch ${batch}: ${batch} khối, mỗi hàng một khối.`)}{' '}
        {sms.length
          ? sms.map((s) => trText(`${s.name} reports ${s.n} multiprocessors.`, `${s.name} báo cáo ${s.n} multiprocessor.`)).join(' ')
          : null}
      </p>
      {measured.length ? (
        <div className="panel" style={{ marginTop: '0.8rem', overflowX: 'auto' }} data-part="measured">
          <Translated vi={<p style={{ marginTop: 0 }}>Lab X1.1 trên máy này, bên cạnh tỉ lệ lưu lượng mà mô hình cho cùng hình dạng:</p>}>
            <p style={{ marginTop: 0 }}>Lab X1.1 on this machine, beside the traffic ratio the model gives for the same shape:</p>
          </Translated>
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr>
                <th><Translated vi="Backend">Backend</Translated></th>
                <th className="num"><Translated vi="Số hàng">Rows</Translated></th>
                <th className="num">V</th>
                <th className="num">k</th>
                <th className="num"><Translated vi="Chọn">Select</Translated></th>
                <th className="num"><Translated vi="Sắp xếp">Sort</Translated></th>
                <th className="num"><Translated vi="Tỉ lệ đo được">Measured ratio</Translated></th>
                <th className="num"><Translated vi="Tỉ lệ lưu lượng">Traffic ratio</Translated></th>
              </tr>
            </thead>
            <tbody>
              {measured.map((m, i) => (
                <tr key={i} data-measured={m.backend}>
                  <td>{m.device} ({m.backend})</td>
                  <td className="num">{m.rows}</td>
                  <td className="num">{m.cols.toLocaleString('en-US')}</td>
                  <td className="num">{isFinite(m.k) ? m.k : '?'}</td>
                  <td className="num">{fmt(m.select * 1e3)} ms</td>
                  <td className="num">{fmt(m.sort * 1e3)} ms</td>
                  <td className="num"><b>{fmt(m.speedup)}×</b></td>
                  <td className="num">{isFinite(m.k) ? `${fmt(trafficRatio(m.cols, m.k))}×` : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="muted" style={{ margin: '0.6rem 0 0' }} data-part="measured-none">
          <Translated
            vi={
              <>
                Chưa có benchmark của lab X1.1 trên máy này. Chạy{' '}
                <code>python3 labs/pmpp.py run x1/lab1 --solution --bench</code> rồi tải lại: thời gian đo được sẽ hiện ở
                đây bên cạnh mô hình.
              </>
            }
          >
            No lab X1.1 benchmark on this machine yet. Run <code>python3 labs/pmpp.py run x1/lab1 --solution --bench</code> and
            reload: the measured times appear here beside the model.
          </Translated>
        </p>
      )}
    </Bench>
  );
}

function MeasuredSelect({ predicted }: { predicted: number }) {
  const measured = useMeasured();
  if (!measured.length)
    return (
      <Translated
        vi={
          <p>
            Chưa có số đo nào trên máy này. Chạy <code>python3 labs/pmpp.py run x1/lab1 --solution --bench</code> rồi quay
            lại: ô này sẽ so sánh dự đoán của bạn với những gì thiết bị của bạn đo được.
          </p>
        }
      >
        <p>
          No measurement on this machine yet. Run <code>python3 labs/pmpp.py run x1/lab1 --solution --bench</code> and come
          back: this box will compare your prediction with what your own device measured.
        </p>
      </Translated>
    );
  const exact = measured.filter((m) => m.rows === 64 && m.cols === 151936 && m.k === 40);
  const show = exact.length ? exact : measured;
  return (
    <>
      {!exact.length ? (
        <Translated vi={<p>Lab không ghi đúng cấu hình 64 × 151,936 ở k = 40; đây là những gì nó đã ghi.</p>}>
          <p>The lab did not record exactly 64 × 151,936 at k = 40; here is what it did record.</p>
        </Translated>
      ) : null}
      {show.map((m, i) => (
        <p key={i} data-reveal={m.backend}>
          <Translated
            vi={
              <>
                Trên <b>{m.device}</b> ({m.backend}), {m.rows} hàng gồm {m.cols.toLocaleString('en-US')}
                {isFinite(m.k) ? `, k = ${m.k}` : ''}: phép chọn mất {fmt(m.select * 1e3)} ms và phép sắp xếp{' '}
                {fmt(m.sort * 1e3)} ms, <b>{fmt(m.speedup)}×</b>.{' '}
              </>
            }
          >
            On <b>{m.device}</b> ({m.backend}), {m.rows} rows of {m.cols.toLocaleString('en-US')}
            {isFinite(m.k) ? `, k = ${m.k}` : ''}: the select took {fmt(m.select * 1e3)} ms and the sort {fmt(m.sort * 1e3)} ms,{' '}
            <b>{fmt(m.speedup)}×</b>.{' '}
          </Translated>
          {predicted > m.speedup * 1.5
            ? trText(
                'You expected more. The traffic ratio is about 4; time follows traffic only where both kernels are limited by memory, and the sort may have had the cache on its side.',
                'Bạn kỳ vọng nhiều hơn. Tỉ lệ lưu lượng khoảng 4; thời gian chỉ bám theo lưu lượng khi cả hai kernel đều bị giới hạn bởi bộ nhớ, và phép sắp xếp có thể đã được cache trợ giúp.',
              )
            : predicted < m.speedup / 1.5
              ? trText(
                  'You expected less. The count of bytes is a floor on time, not a forecast: launches, barriers and atomics also cost.',
                  'Bạn kỳ vọng ít hơn. Số byte là mức sàn của thời gian, không phải một dự báo: các lần khởi chạy, rào chắn và atomic cũng tốn kém.',
                )
              : trText('That is close to what you predicted.', 'Kết quả đó gần với điều bạn dự đoán.')}
        </p>
      ))}
      <Translated
        vi={
          <p className="muted">
            Đây là số đo từ máy của bạn, không phải giá trị kỳ vọng; một thiết bị khác có thể khác rất nhiều.
          </p>
        }
      >
        <p className="muted">These are measurements from your machine, not expected values; another device can differ widely.</p>
      </Translated>
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSX14: Record<string, Question> = {
  'x1/X1.4/q-passes': {
    kind: 'numeric',
    id: 'x1/X1.4/q-passes',
    prompt: (
      <Translated vi={<>Một radix select với chữ số 8 bit trên khóa 32 bit: nó cần tối đa bao nhiêu lượt histogram để biết chính xác khóa thứ k?</>}>
        <>A radix select with 8-bit digits over 32-bit keys: at most how many histogram passes does it need to know the k-th key exactly?</>
      </Translated>
    ),
    answer: 4,
    unit: 'passes',
    explain: (
      <Translated
        vi={
          <p>
            Mỗi lượt cố định thêm một byte của tiền tố, và khóa có bốn byte. Sau lượt thứ tư, bin chưa quyết định chứa các
            khóa bằng nhau ở mọi bit: chính khóa thứ k và các khóa hòa với nó. Nó có thể dừng sớm hơn khi số khóa còn cần đúng
            bằng cả bin chứa điểm cắt.
          </p>
        }
      >
        <p>
          Each pass fixes one more byte of the prefix, and the key has four. After the fourth the undecided bin holds keys
          equal in every bit: the k-th key itself and its ties. It can stop sooner when the k still needed equals the whole
          crossing bin.
        </p>
      </Translated>
    ),
    source: 'ours; the MSD idea is ch. 14 §14.9, p. 346',
    vi: { unit: 'lượt', source: 'của khóa học này; ý tưởng MSD là ch. 14 §14.9, p. 346' },
  },
  'x1/X1.4/q-bisect': {
    kind: 'numeric',
    id: 'x1/X1.4/q-bisect',
    prompt: (
      <Translated
        vi={
          <>
            Phép chia đôi ngưỡng trên cùng các khóa 32 bit cố định một bit mỗi lượt đếm: "có ít nhất k khóa lớn hơn hoặc bằng
            ứng viên này không?" Cần bao nhiêu lượt để có chính xác khóa thứ k?
          </>
        }
      >
        Threshold bisection over the same 32-bit keys fixes one bit per counting pass: "are there at least k keys at or
        above this candidate?" How many passes for the exact k-th key?
      </Translated>
    ),
    answer: 32,
    unit: 'passes',
    explain: (
      <Translated
        vi={
          <p>
            Một bit mỗi lượt, 32 bit: 32 lượt, gấp tám lần bốn lượt của radix select, và mỗi lượt đọc cả hàng. Một lượt đếm
            256 bin cùng lúc học được 8 bit; một lượt đếm một ngưỡng học được 1. Histogram tốn bộ nhớ chia sẻ và atomic, và
            đổi lại giảm số lần đọc đi tám lần.
          </p>
        }
      >
        <p>
          One bit per pass, 32 bits: 32 passes, eight times the radix select's four, and each reads the whole row. A pass
          that counts 256 bins at once learns 8 bits; a pass that counts one threshold learns 1. The histogram costs shared
          memory and atomics, and buys an eightfold cut in reads.
        </p>
      </Translated>
    ),
    source: 'ours',
    vi: { unit: 'lượt', source: 'của khóa học này' },
  },
  'x1/X1.4/q-bytes-ratio': {
    kind: 'numeric',
    id: 'x1/X1.4/q-bytes-ratio',
    prompt: (
      <Translated
        vi={
          <>
            Đếm byte trên mỗi khóa như trang này: một lượt sắp xếp LSD đọc khóa 4 byte một lần để đếm và đọc rồi ghi cặp
            (khóa, chỉ số) 8 byte một lần để phân tán, bốn lượt; phép chọn đọc khóa bốn lần để làm histogram và một lần để
            gom. Số byte của sắp xếp chia cho số byte của phép chọn, bỏ qua k cặp đầu ra?
          </>
        }
      >
        Count bytes per key as this page does: an LSD sort pass reads the 4-byte key once to count and reads and writes the
        8-byte (key, index) pair once to scatter, four passes; the select reads the key four times to histogram and once
        to gather. Sort bytes divided by select bytes, ignoring the k output pairs?
      </Translated>
    ),
    answer: 4,
    explain: (
      <Translated
        vi={
          <p>
            Sắp xếp: 4 × (4 + 8 + 8) = 80 byte mỗi khóa. Chọn: 5 × 4 = 20. Tỉ lệ là 4, và nó hầu như không đổi theo k: k cặp
            đầu ra cộng thêm 8k byte vào một hàng V khóa.
          </p>
        }
      >
        <p>
          Sort: 4 × (4 + 8 + 8) = 80 bytes per key. Select: 5 × 4 = 20. The ratio is 4, and it barely moves with k: the k
          output pairs add 8k bytes to a row of V keys.
        </p>
      </Translated>
    ),
    source: 'ours; sort traffic as src/sim/radixcost.ts counts lab 14.3',
    vi: { source: 'của khóa học này; lưu lượng sắp xếp theo cách src/sim/radixcost.ts đếm lab 14.3' },
  },
  'x1/X1.4/q-hf-ties': {
    kind: 'choice',
    id: 'x1/X1.4/q-hf-ties',
    prompt: (
      <Translated
        vi={
          <>
            Logits [6, 5, 5, 4, 1] và k = 2. <code>TopKLogitsWarper</code> của transformers giữ bao nhiêu token, và phép chọn
            của M5 giữ bao nhiêu?
          </>
        }
      >
        Logits [6, 5, 5, 4, 1] and k = 2. How many tokens does transformers' <code>TopKLogitsWarper</code> keep, and how
        many does M5's select?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="transformers giữ 3, vì cả hai số 5 hòa với giá trị thứ k; M5 giữ đúng 2, token 0 và 1">
            transformers 3, as both 5s tie the k-th value; M5 exactly 2, tokens 0 and 1
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Cả hai giữ 2: top-k nghĩa là đúng k token, và cả hai cắt trường hợp hòa theo thứ tự cột">
            Both keep 2: top-k means exactly k tokens, and both cut the tie by column order
          </Translated>
        ),
        why: (
          <Translated vi="transformers chỉ loại các token nhỏ hơn hẳn giá trị thứ k, nên một token hòa với nó vẫn sống sót.">
            transformers removes only tokens strictly below the k-th value, so a tie with it survives.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Cả hai giữ 3: hai số 5 hòa với giá trị thứ k, và cả hai giữ khóa hòa tại điểm cắt">
            Both keep 3: the two 5s tie with the k-th value, and both keep a tie at the cut
          </Translated>
        ),
        why: (
          <Translated vi="Phép chọn của M5 có một ngân sách hòa: nó giữ đúng k, cắt các khóa hòa theo cột.">
            M5's select has a tie budget: it keeps exactly k, cutting ties by column.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            transformers tính giá trị thứ k bằng <code>torch.topk</code> và loại các token có{' '}
            <code>scores &lt; kth</code>: mọi token bằng giá trị thứ k đều ở lại, nên có thể còn nhiều hơn k. Phép chọn giữ
            một khóa lớn hơn 5 rồi 1 trong 2 khóa hòa, cột thấp hơn: token 0 và 1. Radix select CUDA của llama.cpp và{' '}
            <code>kernel_top_k</code> Metal cắt khóa hòa theo cùng cách.
          </p>
        }
      >
        <p>
          transformers computes the k-th value with <code>torch.topk</code> and removes tokens with{' '}
          <code>scores &lt; kth</code>: every token equal to the k-th value stays, so more than k can remain. The select
          keeps the one key above 5 and then 1 of the 2 ties, the lower column: tokens 0 and 1. llama.cpp's CUDA radix select
          and Metal <code>kernel_top_k</code> cut ties the same way.
        </p>
      </Translated>
    ),
    source: 'notes/today/x1-facts.md D4; ch14-facts D2, ch12-facts D1, D2 (checked 2026-11-20)',
    vi: { source: 'notes/today/x1-facts.md D4; ch14-facts D2, ch12-facts D1, D2 (kiểm tra 2026-11-20)' },
  },
  'x1/X1.4/q-k1': {
    kind: 'choice',
    id: 'x1/X1.4/q-k1',
    prompt: (
      <Translated vi={<>Với k = 1 và quy tắc hòa của M5, top-k có giống greedy argmax không?</>}>
        <>With k = 1 and M5's tie rule, is top-k the same as greedy argmax?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Có: ứng viên sống sót duy nhất là logit lớn nhất, cột thấp nhất khi hòa, đúng quy tắc argmax của lab 10.3">
            Yes: the one survivor is the largest logit, lowest column on a tie, which is lab 10.3's argmax rule
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không: với k = 1, top-k vẫn lấy mẫu giữa các token hòa ở logit lớn nhất">
            No: with k = 1, top-k still samples among the tokens tied for the largest logit
          </Translated>
        ),
        why: (
          <Translated vi="Ngân sách hòa là 1, nên chỉ cột hòa thấp nhất sống sót; không còn gì để lấy mẫu.">
            The tie budget is 1, so only the lowest tied column survives; there is nothing left to sample among.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ khi temperature là 1, vì temperature thay đổi logit nào lớn nhất">
            Only if the temperature is 1, since temperature changes which logit is largest
          </Translated>
        ),
        why: (
          <Translated vi="Temperature nhân mọi logit với cùng một hệ số dương và không thay đổi logit nào lớn nhất.">
            Temperature scales every logit by the same positive factor and does not change which one is largest.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một ứng viên sống sót, nên bước rút chỉ có một ứng viên. Bộ dựng chuỗi của llama.cpp biết điều này: trong một số
            cấu hình nó thay bước rút cuối bằng sampler greedy khi top-k là 1 (x1-facts D1).
          </p>
        }
      >
        <p>
          One survivor, so the draw has one candidate. llama.cpp's chain builder knows this: in some configurations it
          replaces the final draw with its greedy sampler when top-k is 1 (x1-facts D1).
        </p>
      </Translated>
    ),
    source: 'ours; lab 10.3; notes/today/x1-facts.md D1',
    vi: { source: 'của khóa học này; lab 10.3; notes/today/x1-facts.md D1' },
  },
  'x1/X1.4/q-integer': {
    kind: 'choice',
    id: 'x1/X1.4/q-integer',
    prompt: (
      <Translated vi={<>Vì sao phép chọn so sánh khóa số nguyên thay vì float, khi chế độ toán mặc định của Metal không tuân thủ IEEE chặt chẽ?</>}>
        <>Why does the select compare integer keys instead of floats, given that Metal's default math mode is not IEEE-strict?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Phép toán số nguyên chính xác dưới mọi chế độ toán, nên mọi backend chọn cùng một tập">
            Integer operations are exact under every math mode, so every backend selects the same set
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="So sánh số nguyên nhanh hơn so sánh float trên mọi GPU, điều quan trọng trong một lượt qua 150,000 khóa">
            Integer compares are faster than float compares on every GPU, which matters in a pass over 150,000 keys
          </Translated>
        ),
        why: (
          <Translated vi="Tốc độ không phải lý do; điểm mấu chốt là các quy tắc fast-math không áp dụng cho số nguyên.">
            Speed is not the reason; the point is that fast-math rules do not apply to integers.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một float không thể làm chỉ số cho bin histogram, nên dù sao các bit cũng phải được chuyển đổi cho atomic">
            A float cannot index a histogram bin, so the bits have to be converted for the atomic anyway
          </Translated>
        ),
        why: (
          <Translated vi="Điều đó giải thích chữ số, không giải thích vì sao các phép so sánh với tiền tố cũng là số nguyên.">
            That explains the digit, not why the comparisons with the prefix are integer too.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            So sánh float cũng chính xác với các giá trị hữu hạn, nhưng các chế độ fast và relaxed của Metal cho phép "no
            signed zeros", và fast math để hành vi của NaN và vô cùng là không xác định. Sau{' '}
            <code>as_type&lt;uint&gt;</code>, không còn gì để các quy tắc đó viết lại, và khóa làm -0.0 bằng +0.0 ngay từ
            cách xây dựng.
          </p>
        }
      >
        <p>
          A float comparison is exact for finite values too, but Metal's fast and relaxed modes allow "no signed zeros",
          and fast math leaves NaN and infinity behaviour undefined. After <code>as_type&lt;uint&gt;</code>, nothing is
          left for those rules to rewrite, and the key makes -0.0 equal to +0.0 by construction.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch14-facts.md C3, A3, E1; ch05-facts C7', // i18n-ignore: citation
  },
  'x1/X1.4/q-batch1': {
    kind: 'choice',
    id: 'x1/X1.4/q-batch1',
    prompt: (
      <Translated vi={<>Ở batch 1, điều gì giới hạn mọi top-k một khối cho mỗi hàng, dù là chọn hay sắp xếp?</>}>
        <>At batch 1, what limits every one-block-per-row top-k, select or sort alike?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Một khối dùng một multiprocessor, và phần còn lại của thiết bị ngồi không">
            One block uses one multiprocessor, and the rest of the device idles
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Kích thước bộ nhớ chia sẻ">The size of shared memory</Translated>,
        why: (
          <Translated vi="Một histogram 256 bin là 1 kB; bộ nhớ chia sẻ không phải thứ bị cạn.">
            A 256-bin histogram is 1 kB; shared memory is not what runs out.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Số bin histogram: 256 bin nghĩa là bốn lượt qua mỗi hàng">
            The number of histogram bins: 256 bins mean four passes over every row
          </Translated>
        ),
        why: (
          <Translated vi="Nhiều bin hơn sẽ nghĩa là ít lượt hơn, nhưng vẫn một khối.">
            More bins would mean fewer passes, but still one block.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>torch.topk</code> của PyTorch có một nhánh nhiều khối, <code>mbtopk</code>, với chữ số 8 bit (ch14-facts
            D5). Chia một hàng nghĩa là nhiều khối đếm phần của mình, và các số đếm phải được gộp trước lần đi: thêm một bước
            mỗi lượt (cách chúng tôi hiểu sự đánh đổi). Không được dựng ở đây.
          </p>
        }
      >
        <p>
          PyTorch's <code>torch.topk</code> has a multi-block path, <code>mbtopk</code>, with 8-bit digits (ch14-facts
          D5). Splitting a row means several blocks count their shares, and the counts must be combined before the walk:
          one more step per pass (our reading of the trade-off). Not built here.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch14-facts.md D5; ch. 9 §9.4, pp. 211–213', // i18n-ignore: citation
  },
};

// ---------------------------------------------------------------- the section
export function SX14() {
  const { reports } = useReports();
  const st = { cu: solutionTabStatus(reports, LAB_X11, 'kernel.cu'), metal: solutionTabStatus(reports, LAB_X11, 'kernel.metal') };
  const cu = loaded(useLabFile('x1/lab1_topk/solution/kernel.cu'));
  const metal = loaded(useLabFile('x1/lab1_topk/solution/kernel.metal'));
  return (
    <>
      <Translated
        vi={
          <p>
            Một sampler top-k cần tập k logits lớn nhất, không cần thứ tự của 150,000 logits còn lại. Lab 14.3 sắp xếp mọi
            hàng, rồi top-k là một nhát cắt: đúng, nhưng nó trả lời một câu hỏi lớn hơn câu được hỏi. Bốn lượt 8 bit của nó
            mỗi lượt đọc và ghi mọi khóa và mọi chỉ số. Trang này tìm k giá trị lớn nhất mà không sắp xếp, bằng cách thu hẹp
            dần về khóa thứ k, mỗi lần một byte.
          </p>
        }
      >
        <p>
          A top-k sampler needs the set of the k largest logits, not the order of the other 150,000. Lab 14.3 sorts every
          row, and then top-k is a cut: correct, but it answers a bigger question than the one asked. Its four 8-bit passes
          each read and write every key and every index. This page finds the k largest without sorting, by narrowing in on
          the k-th key one byte at a time.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách không mô tả thuật toán chọn nào. MSD radix sort là §14.9 (p. 346); trang này áp ý tưởng của nó vào phép
              chọn, như llama.cpp và PyTorch làm (notes/today/ch14-facts.md D2, D5).
            </p>
          }
        >
          <p>
            The book describes no selection algorithm. MSD radix sort is §14.9 (p. 346); this page applies its idea to
            selection, as llama.cpp and PyTorch do (notes/today/ch14-facts.md D2, D5).
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Bạn đã gọi thứ này rồi: <code>torch.topk(logits, 40)</code> trên một tensor CUDA chạy một radix select kiểu này,
            với chữ số 2 bit trong một khối hoặc chữ số 8 bit trên nhiều khối, không phải một phép sắp xếp.
          </p>
        }
      >
        <p>
          You have called this already: <code>torch.topk(logits, 40)</code> on a CUDA tensor runs a radix select of this
          kind, with 2-bit digits in one block or 8-bit digits across several, not a sort.
        </p>
      </Translated>

      <h2>
        <Translated vi="Khóa, không phải float">Keys, not floats</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mỗi logit trở thành khóa 32 bit của lab 14.3: bật bit dấu của một float không âm, đảo mọi bit của một float âm,
            và trước hết ánh xạ -0.0 thành +0.0. Đọc như số nguyên không dấu, các khóa có cùng thứ tự với logits. Từ đây trở
            đi phép chọn không làm phép tính float nào, chỉ có dịch bit, mặt nạ và so sánh số nguyên. Điều đó chính xác trên
            mọi backend, kể cả fast math của Metal.
          </p>
        }
      >
        <p>
          Every logit becomes lab 14.3's 32-bit key: set the sign bit of a non-negative float, invert every bit of a
          negative one, and map -0.0 to +0.0 first. Read as unsigned integers, the keys are in the order of the logits.
          From here on the select does no float arithmetic at all, only shifts, masks and integer compares. That is exact on
          every backend, Metal's fast math included.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md A3, C3, E1)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              CUB ghi trong tài liệu phép biến đổi này và bước -0.0. Các chế độ fast và relaxed của Metal đều cho phép "no
              signed zeros", nên bước -0.0 phải được làm trên <code>uint</code> sau <code>as_type</code>, không bao giờ như
              một phép kiểm tra float.
            </p>
          }
        >
          <p>
            CUB documents this transform and the -0.0 step. Metal's fast and relaxed modes both allow "no signed zeros", so
            the -0.0 step has to be done on the <code>uint</code> after <code>as_type</code>, never as a float test.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsSX14['x1/X1.4/q-integer']} />

      <h2>
        <Translated vi="Thu hẹp dần, mỗi lần một byte">Narrowing in, one byte at a time</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Byte cao nhất của một khóa chia hàng thành tối đa 256 bin. Một khối đếm số khóa rơi vào mỗi bin: một histogram
            trong bộ nhớ chia sẻ (shared memory), được điền bằng atomic, đúng là histogram được riêng tư hóa (privatization)
            của chương 9. Đi qua các bin từ trên xuống, cộng dồn các số đếm, cho đến khi số đếm tích lũy đạt k. Mọi khóa trong
            một bin phía trên bin đó được chọn. Mọi khóa phía dưới bị loại. Chỉ bin chứa điểm cắt là chưa quyết định, và byte
            của nó trở thành byte đầu tiên của tiền tố.
          </p>
        }
      >
        <p>
          The top byte of a key splits the row into at most 256 bins. A block counts how many keys fall in each bin: one
          histogram in shared memory, filled with atomics, exactly chapter 9's privatised histogram. Walk the bins from the
          top, adding counts, until the running count reaches k. Every key in a bin above that one is in. Every key below is
          out. Only the crossing bin is undecided, and its byte becomes the first byte of the prefix.
        </p>
      </Translated>
      <Source section="9.4" pages="211–214" figs="Fig. 9.10" />
      <Translated
        vi={
          <p>
            Lượt tiếp theo làm như vậy với byte thứ hai, chỉ đếm các khóa có byte cao bằng tiền tố, và tìm k khóa còn cần sau
            các khóa đã được chọn. Sau tối đa bốn lượt, tiền tố chính là khóa thứ k, gọi là K*. Khi đó bạn biết có bao nhiêu
            khóa lớn hơn hẳn K* và còn cần bao nhiêu khóa trong số các khóa bằng K*.
          </p>
        }
      >
        <p>
          The next pass does the same with the second byte, counting only keys whose top byte equals the prefix, and looking
          for the k still needed after the keys already in. After at most four passes the prefix is the k-th key itself,
          call it K*. You then know how many keys lie strictly above K* and how many of those equal to K* are still needed.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Trình chạy từng bước chạy một khối 16 luồng trên 64 khóa; các hàng thật dài hơn và các khối rộng hơn. Warp là
              một cách nhóm bạn chọn để hiển thị: nó cho thấy lane nào đi vào nhánh, không cho thấy một warp bị tách tốn bao
              nhiêu. Không có thời gian, không có cache.
            </p>
          }
        >
          <p>
            The stepper runs one block of 16 threads over 64 keys; real rows are longer and blocks wider. Warps are a
            grouping you choose for display: it shows which lanes take the branch, not what a split warp costs. No timing,
            no caches.
          </p>
        </Translated>
      </Simplification>
      <Ask q={questionsSX14['x1/X1.4/q-passes']} label={trText('Work it out', 'Tự tính')} />
      <RadixStepper />
      <Translated
        vi={
          <p>
            Ba điều cần tìm trong trình chạy từng bước. Ở lượt 1 mọi lane đều tham gia; từ lượt 2 trở đi, chỉ các lane giữ
            khóa trong bin chưa quyết định mới tham gia, và các warp bị tách ở phép kiểm tra đó. Nạp "top 24 bits shared":
            các lượt 1 đến 3 mỗi lượt chỉ tìm thấy một bin và không quyết định gì, và toàn bộ công việc diễn ra ở lượt 4. Nạp
            "peaked" với k = 3: ba khóa của bin cao nhất đúng là ba khóa cần, và phép chọn dừng sau một lượt.
          </p>
        }
      >
        <p>
          Three things to find in the stepper. In pass 1 every lane takes part; from pass 2 on, only lanes holding a key in
          the undecided bin do, and warps split at that test. Load "top 24 bits shared": passes 1 to 3 each find a single bin
          and decide nothing, and all the work happens in pass 4. Load "peaked" with k = 3: the three keys of the top bin are
          exactly the three needed, and the select stops after one pass.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Hoạt ảnh bên dưới chạy cùng mô hình trên hàng riêng của nó: 64 khóa, k = 12, một khối 16 luồng trong một warp.
              Không có thời gian, không có cache.
            </p>
          }
        >
          <p>
            The animation below runs the same model on its own row: 64 keys, k = 12, one block of 16 threads in one warp. No
            timing, no caches.
          </p>
        </Translated>
      </Simplification>
      <RadixSelectScene />

      <h2>
        <Translated vi="Ghi câu trả lời">Writing the answer</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Thêm một lượt ghi đúng k cặp (cột, logit). Theo các vòng bằng kích thước khối, mỗi luồng phân loại khóa của nó:
            trên K* (giữ), bằng K* (hòa), hoặc dưới (loại). Một khóa hòa được giữ khi số khóa hòa đứng trước nó theo thứ tự
            cột nhỏ hơn ngân sách. Vị trí của một khóa được giữ là số khóa được giữ đứng trước nó: các khóa được giữ ở những
            vòng trước, cộng một ballot trên các lane của vòng và một popcount của các lane phía dưới. Đó là phép xếp hạng cấp
            warp của chương 12 bên trong bộ lọc ổn định của chương 12, có thêm ngân sách hòa.
          </p>
        }
      >
        <p>
          One more pass writes exactly k (column, logit) pairs. In rounds of the block size, each thread classifies its key:
          above K* (kept), equal to K* (a tie), or below (dropped). A tie is kept while the number of ties before it in
          column order is below the budget. A kept key's slot is the number of kept keys before it: the kept keys of earlier
          rounds, plus a ballot of the round's lanes and a popcount of the lanes below. That is chapter 12's warp-level rank
          inside chapter 12's stable filter, with a tie budget added.
        </p>
      </Translated>
      <Source section="12.3, 12.5" pages="292–293, 297–298" figs="Fig. 12.3" />
      <Translated
        vi={
          <p>
            Đầu ra theo thứ tự cột tăng dần, không theo thứ tự giá trị, và các khóa hòa thuộc về các cột thấp nhất. Đó là quy
            tắc của radix select CUDA của llama.cpp và <code>kernel_top_k</code> Metal của nó. Hugging Face transformers giữ
            mọi token hòa với giá trị thứ k, nên nó có thể giữ nhiều hơn k.
          </p>
        }
      >
        <p>
          The output is in ascending column order, not value order, and ties go to the lowest columns. That is the rule of
          llama.cpp's CUDA radix select and its Metal <code>kernel_top_k</code>. Hugging Face transformers keeps every token
          tied with the k-th value, so it can keep more than k.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md D2, ch12-facts D1, D2, x1-facts D4)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              llama.cpp, <code>top-k.cu</code> (trích nguyên văn): "the ties fill the last rank slots in column order, the
              lowest columns win." Các khóa thắng phía trên ngưỡng của nó lấy vị trí bằng atomic, nên thứ tự của chúng thay
              đổi giữa các lần chạy; chỉ M5 đòi thứ tự cột từ đầu đến cuối.
            </p>
          }
        >
          <p>
            llama.cpp, <code>top-k.cu</code>: "the ties fill the last rank slots in column order, the lowest columns win."
            Its winners above the threshold take slots with an atomic, so their order varies from run to run; only M5
            requires column order throughout.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsSX14['x1/X1.4/q-hf-ties']} />
      <Ask q={questionsSX14['x1/X1.4/q-k1']} />

      <h2>
        <Translated vi="Nó tiết kiệm được gì">What it saves</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Đếm byte trên mỗi khóa. Một lượt LSD radix sort đọc khóa 4 byte để đếm, rồi đọc và ghi cặp (khóa, chỉ số) 8 byte
            để phân tán: 20 byte, bốn lượt, 80. Phép chọn đọc khóa bốn lần để làm histogram và một lần để gom: 20, cộng 8 byte
            cho mỗi cặp được giữ. Phép chia đôi ngưỡng, kiểm tra một bit của khóa thứ k mỗi lượt đếm, cần 32 lần đọc để đếm
            và cùng lần gom đó.
          </p>
        }
      >
        <p>
          Count bytes per key. An LSD radix sort pass reads the 4-byte key to count it, then reads and writes the 8-byte
          (key, index) pair to scatter it: 20 bytes, four passes, 80. The select reads the key four times to histogram and
          once to gather: 20, plus 8 bytes for each kept pair. Threshold bisection, which tests one bit of the k-th key per
          counting pass, needs 32 counting reads and the same gather.
        </p>
      </Translated>
      <Ask q={questionsSX14['x1/X1.4/q-bytes-ratio']} label={trText('Work it out', 'Tự tính')} />
      <CostExplorer />
      <Translated
        vi={
          <p>
            Lưu lượng của phép chọn hầu như không phụ thuộc vào k; của phép sắp xếp thì hoàn toàn không. Và ở batch 1, cả ba
            thiết kế đều chạy một khối, nên các multiprocessor khác của thiết bị không có gì để làm.
          </p>
        }
      >
        <p>
          The select's traffic barely depends on k; the sort's not at all. And at batch 1, all three designs run one block,
          so the device's other multiprocessors have nothing to do.
        </p>
      </Translated>
      <Ask q={questionsSX14['x1/X1.4/q-bisect']} label={trText('Work it out', 'Tự tính')} />
      <Ask q={questionsSX14['x1/X1.4/q-batch1']} />
      <Predict
        id="x1/X1.4/p-select-speedup"
        prompt={
          <Translated
            vi={
              <>
                Lab X1.1 chạy cả hai trên 64 hàng gồm 151,936 logits với k = 40. Phép chọn sẽ nhanh hơn phép sắp xếp toàn bộ
                bao nhiêu lần trên thiết bị <em>của bạn</em>? Chốt một con số trước khi chạy.
              </>
            }
          >
            Lab X1.1 runs both on 64 rows of 151,936 logits with k = 40. How many times faster will its select be than the
            full sort on <em>your</em> device? Commit a number before you run it.
          </Translated>
        }
        unit={trText('× faster', '× nhanh hơn')}
      >
        {(v) => <MeasuredSelect predicted={v} />}
      </Predict>
      <Translated
        vi={
          <p>
            <a href={href('x1', 'labs')}>Trang lab</a> có lab X1.1, phép chọn được đo so với phép sắp xếp của lab 14.3 trên
            mọi backend.
          </p>
        }
      >
        <p>
          The <a href={href('x1', 'labs')}>labs page</a> has lab X1.1, the select measured against lab 14.3's sort on every
          backend.
        </p>
      </Translated>

      <h2>
        <Translated vi="Ai cung cấp nó">Who ships it</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mọi thư viện bạn có thể gọi đều có một top-k, và hầu hết là phép chọn. Các bảo đảm của chúng khác nhau nhiều hơn
            tên gọi gợi ý.
          </p>
        }
      >
        <p>
          Every library you are likely to call has a top-k, and most of them are selections. Their guarantees differ more
          than their names suggest.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md A7, B4, C1, D2, D4, D5, D7; ch09-facts D4)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              <code>torch.topk</code>: radix select, chữ số 2 bit trong một khối, 8 bit trong <code>mbtopk</code>. llama.cpp:{' '}
              <code>top_k_radix_cuda</code> với chữ số 8 bit và histogram chia sẻ theo từng khối; Metal{' '}
              <code>kernel_top_k</code> với một histogram threadgroup 256 bin. MLX <code>topk</code>: hiện là một phép sắp xếp
              toàn bộ trên Metal. Top-k/top-p mặc định của vLLM: một phép tìm pivot, không sắp xếp (chưa đọc chính bài báo
              Qrita).
            </p>
          }
        >
          <p>
            <code>torch.topk</code>: radix select, 2-bit digits in one block, 8-bit in <code>mbtopk</code>. llama.cpp:{' '}
            <code>top_k_radix_cuda</code> with 8-bit digits and per-block shared histograms; Metal{' '}
            <code>kernel_top_k</code> with a 256-bin threadgroup histogram. MLX <code>topk</code>: a full sort on Metal today.
            vLLM's default top-k/top-p: a pivot search, no sort (the Qrita paper itself was not read).
          </p>
        </Translated>
      </Today>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-table="libraries">
        <thead>
          <tr>
            <th><Translated vi="Lời gọi thư viện">Library call</Translated></th>
            <th><Translated vi="Backend">Backend</Translated></th>
            <th><Translated vi="Điều nó hứa">What it promises</Translated></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>cub::DeviceTopK</code></td>
            <td>CUDA</td>
            <td>
              <Translated vi="K phần tử, không có thứ tự, có thể không tất định; các phần tử hòa ở vị trí thứ K không được xác định (cấu hình duy nhất được chấp nhận hiện nay)">
                K items, unordered, possibly non-deterministic; tied items at the K-th position unspecified (the only configuration accepted today)
              </Translated>
            </td>
          </tr>
          <tr>
            <td><code>cub::DeviceBatchedTopK</code></td>
            <td><Translated vi="CUDA 13.5 trở lên">CUDA 13.5 and later</Translated></td>
            <td>
              <Translated vi="Theo từng đoạn; một đoạn bị giới hạn ở 2^21 phần tử">Per segment; a segment is capped at 2^21 items</Translated>
            </td>
          </tr>
          <tr>
            <td><code>rocprim::topk</code>, <code>segmented_topk</code></td>
            <td>HIP</td>
            <td>
              <Translated vi={'Không có wrapper hipCUB; đầu ra theo đoạn có thứ tự và tất định thì "not supported yet"'}>
                No hipCUB wrapper; ordered and deterministic segmented output "not supported yet"
              </Translated>
            </td>
          </tr>
          <tr>
            <td>MPSGraph{/* i18n-ignore: product name */} <code>topK</code>; <code>MPSMatrixFindTopK</code></td>
            <td>Metal (macOS 14; 10.13.4){/* i18n-ignore: product versions */}</td>
            <td>
              <Translated vi={<>Không nêu thứ tự khi hòa; <code>MPSMatrixFindTopK</code> nhận k không quá 16</>}>
                No stated tie order; <code>MPSMatrixFindTopK</code> takes k of 16 or less
              </Translated>
            </td>
          </tr>
          <tr>
            <td>MLX <code>topk</code></td>
            <td>Metal</td>
            <td>
              <Translated vi={'"not necessarily in sorted order"; hiện chạy một phép merge sort toàn bộ'}>
                "not necessarily in sorted order"; runs a full merge sort today
              </Translated>
            </td>
          </tr>
        </tbody>
      </table>

      <h2>
        <Translated vi="Trên HIP và Metal">On HIP and Metal</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Histogram như nhau trên cả ba: một mảng 256 phần tử trong bộ nhớ chia sẻ với các phép tăng atomic. Trên Metal nó
            là một mảng <code>threadgroup atomic_uint</code> được cập nhật bằng <code>atomic_fetch_add_explicit</code>. Ballot
            của bước gom là chỗ chúng khác nhau. Ballot của CUDA là 32 bit. Của HIP là 64 bit trên mọi target, nên mặt nạ
            cần kiểu 64 bit và <code>__popcll</code>, không phải <code>__popc</code>. <code>simd_ballot</code> của Metal trả
            về một <code>simd_vote</code> 64 bit mà các bit cao không xác định trên một SIMD group hẹp hơn: xếp hạng bằng{' '}
            <code>popcount(mask &amp; ((1ul &lt;&lt; lane) - 1))</code>, không bao giờ đọc các bit đó, và đếm bằng{' '}
            <code>simd_sum</code>.
          </p>
        }
      >
        <p>
          The histogram is the same on all three: a 256-entry array in shared memory with atomic increments. On Metal it is
          a <code>threadgroup atomic_uint</code> array updated with <code>atomic_fetch_add_explicit</code>. The gather's
          ballot is where they differ. CUDA's ballot is 32 bits. HIP's is 64 bits on every target, so the mask needs a
          64-bit type and <code>__popcll</code>, not <code>__popc</code>. Metal's <code>simd_ballot</code> returns a 64-bit{' '}
          <code>simd_vote</code> whose top bits are undefined on a narrower SIMD-group: rank with{' '}
          <code>popcount(mask &amp; ((1ul &lt;&lt; lane) - 1))</code>, which never reads them, and count with{' '}
          <code>simd_sum</code>.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch09-facts.md C1; infra-facts B3; ch12-facts B1, C2)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              Một mặt nạ 32 bit truyền cho các hàm <code>_sync</code> của HIP làm hỏng một <code>static_assert</code>; một
              mặt nạ 64 bit truyền cho <code>__popc</code> bị cắt âm thầm còn 32 bit.
            </p>
          }
        >
          <p>
            A 32-bit mask passed to HIP's <code>_sync</code> functions fails a <code>static_assert</code>; a 64-bit mask
            passed to <code>__popc</code> is silently cut to 32 bits.
          </p>
        </Translated>
      </Today>
      {cu && metal ? (
        <Code
          tabs={[
            { label: trText('CUDA and HIP', 'CUDA và HIP'), code: cu, status: st.cu },
            { label: 'Metal', code: metal, status: st.metal },
          ]}
          caption={
            <Translated
              vi={
                <>
                  Lời giải tham chiếu (solution) của lab X1.1, nạp từ <code>labs/x1/lab1_topk/solution/</code>. Hãy tự viết
                  bản của bạn trong lab trước.
                </>
              }
            >
              <>Lab X1.1's reference solution, loaded from <code>labs/x1/lab1_topk/solution/</code>. Write your own in the lab first.</>
            </Translated>
          }
        />
      ) : (
        <p className="muted">
          <Translated
            vi={
              <>
                Các kernel của lab X1.1 (<code>labs/x1/lab1_topk/solution/</code>) sẽ hiện ở đây khi lab có trên máy này.
              </>
            }
          >
            Lab X1.1's kernels (<code>labs/x1/lab1_topk/solution/</code>) are shown here once the lab is on this machine.
          </Translated>
        </p>
      )}
      <Translated
        vi={
          <p>
            Phép chọn giao lại k ứng viên sống sót theo thứ tự cột. <a href={href('x1', 'X1.5')}>Trang tiếp theo</a> chỉ sắp
            xếp k ứng viên đó và cắt top-p trên chúng.
          </p>
        }
      >
        <p>
          The select hands over its k survivors in column order. The <a href={href('x1', 'X1.5')}>next page</a> sorts just
          those k and cuts top-p over them.
        </p>
      </Translated>
    </>
  );
}
