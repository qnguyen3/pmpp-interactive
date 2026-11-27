import { useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { defineLabs, solutionTabStatus } from '../../components/LabKit';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { benchRows, fmt, labStatus, num, realBackends, useReports, type BenchRow, type Reports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  blockHistogram,
  classifyTaps,
  filterDim,
  footprint,
  gridDims,
  intensity,
  makeProblem,
  readsPerOutput,
  runCached,
  warpSplits,
  type TapKind,
} from '../../sim/convcached';
import {
  imageDims,
  lineBytes,
  linesPerTileRow,
  summarise,
  type CacheConfig,
  type Placement,
} from '../../sim/halocache';
import { HaloFromCacheScene } from './HaloFromCacheScene';

// Section 7.6 (book pp. 178 to 179): tiled convolution that keeps only the tile interior in
// shared memory and reads the halo through the cache. Models: sim/convcached.ts (Fig. 7.15
// tap by tap) and sim/halocache.ts (our toy L2).

const [LAB72] = defineLabs('ch07', [{ num: '7.2', dir: 'lab2_tiledconv', name: 'tiled', vi: { name: 'phân ô' } }]);
const SOL = 'ch07/lab2_tiledconv/solution';

/** Identifier colours for the convolution kernels (each section group keeps its own copy). */
const CONV_TIERS: Record<string, Tier> = {
  N: 'global', P: 'global',
  F: 'const', F_c: 'const',
  N_s: 'shared',
  Pvalue: 'reg', row: 'reg', col: 'reg', tx: 'reg', ty: 'reg', fRow: 'reg', fCol: 'reg', inRow: 'reg', inCol: 'reg',
  tileRow: 'reg', tileCol: 'reg', sRow: 'reg', sCol: 'reg',
};

/** The lines after `// @part <name>` up to the next part marker; the whole file if there is no marker. */
function part(src: string | null, name: string): string | null {
  if (src == null) return null;
  const lines = src.split('\n');
  const start = lines.findIndex((l) => l.trim() === `// @part ${name}`);
  if (start < 0) return src;
  let end = lines.findIndex((l, i) => i > start && /^\s*\/\/ @part /.test(l));
  if (end < 0) end = lines.length;
  return lines.slice(start + 1, end).join('\n').replace(/\s+$/, '');
}
const failed = (src: string | null) => src != null && src.startsWith('// could not load');

/** Real GPUs only: realBackends() keeps the emulator's race-detection flavour. */
const gpuBackends = (reports: Reports) =>
  realBackends(reports).filter((b) => b !== 'race' && !/emulator/i.test(String(reports[b].device?.name ?? '')));

const pct = (x: number, d = 1) => (100 * x).toFixed(d) + '%';
const n = (x: number) => x.toLocaleString('en-US');

// ---------------------------------------------------------------- explorer 1: where each tap comes from

const TILES = [4, 8, 16, 32];
const KIND_FILL: Record<TapKind, string> = { shared: 'var(--shared)', global: 'var(--global)', ghost: 'transparent' };

function TapExplorer({ warpChoices }: { warpChoices: { label: string; value: number }[] }) {
  const [T, setT] = useState(32);
  const [r, setR] = useState(2);
  const [width, setWidth] = useState(3 * 32 + 5);
  const [height, setHeight] = useState(3 * 32 + 3);
  const [bx, setBx] = useState(1);
  const [by, setBy] = useState(1);
  const [tx, setTx] = useState(0);
  const [ty, setTy] = useState(0);
  const [warp, setWarp] = useState(warpChoices[0]?.value ?? 32);
  const [printed, setPrinted] = useState(false);

  const { gx, gy } = gridDims(T, width, height);
  const cbx = Math.min(bx, gx - 1);
  const cby = Math.min(by, gy - 1);
  const spec = useMemo(() => ({ tile: T, r, width, height, bx: cbx, by: cby }), [T, r, width, height, cbx, cby]);
  const thread = classifyTaps({ ...spec, tx: Math.min(tx, T - 1), ty: Math.min(ty, T - 1) });
  const hist = useMemo(() => blockHistogram(spec), [spec]);
  const fp = useMemo(() => footprint(spec), [spec]);
  const warps = useMemo(() => warpSplits(T, r, warp), [T, r, warp]);
  const prob = useMemo(() => makeProblem(width, height, r), [width, height, r]);
  const run = useMemo(() => runCached(prob, { tile: T, asPrinted: printed }), [prob, T, printed]);
  const m = filterDim(r);

  const setTile = (t: number) => {
    setT(t);
    setWidth(3 * t + 5);
    setHeight(3 * t + 3);
    setBx(1);
    setBy(1);
    setTx(0);
    setTy(0);
  };
  const preset = (which: 'interior' | 'edge' | 'corner' | 'overhang') => {
    const [x, y] = which === 'interior' ? [1, 1] : which === 'edge' ? [1, 0] : which === 'corner' ? [0, 0] : [gx - 1, gy - 1];
    setBx(Math.max(0, Math.min(gx - 1, x)));
    setBy(Math.max(0, Math.min(gy - 1, y)));
  };

  // Footprint drawing: (T + 2r) cells a side, the tile in the middle.
  const S = fp.size;
  const c = Math.max(6, Math.min(26, Math.floor(432 / S)));
  const inImg = (i: number, j: number) => {
    const row = cby * T - r + i;
    const col = cbx * T - r + j;
    return row >= 0 && row < height && col >= 0 && col < width;
  };
  const inTile = (i: number, j: number) => i >= r && i < r + T && j >= r && j < r + T;
  const patch = new Map<number, TapKind>();
  for (const p of thread.taps) patch.set((thread.ty + p.fRow) * S + (thread.tx + p.fCol), p.kind);
  const printedAt = new Map<number, 'past' | 'outside' | 'wrong'>();
  if (printed)
    for (const p of thread.taps)
      if (p.printed) printedAt.set((p.printed[0] + r) * S + (p.printed[1] + r), p.outsideArray ? 'outside' : p.pastTile ? 'past' : 'wrong');
  const mine = { shared: 0, global: 0, ghost: 0 };
  for (const p of thread.taps) mine[p.kind]++;
  const tWrong = thread.taps.filter((p) => p.wrongElement).length;
  const tPast = thread.taps.filter((p) => p.pastTile).length;
  const tOut = thread.taps.filter((p) => p.outsideArray).length;
  const guardRows = thread.taps.filter((p) => p.fRow === 0);

  const sel = (label: string, value: number, options: number[], on: (v: number) => void, fmtOpt: (v: number) => string = String) => (
    <label className="field">
      {label}
      <select value={value} onChange={(e) => on(+e.target.value)}>
        {options.map((o) => <option key={o} value={o}>{fmtOpt(o)}</option>)}
      </select>
    </label>
  );

  return (
    <Bench
      title={tr('Where each tap comes from', 'Mỗi tap đến từ đâu')}
      task={tr(
        'Find a thread whose taps all stay in shared memory, then switch line 21 to the printed version.',
        'Tìm một luồng có mọi tap đều nằm trong bộ nhớ chia sẻ, rồi chuyển dòng 21 sang phiên bản như được in.',
      )}
      foot={
        <Translated
          vi={
            <>
              Được tính bởi bộ phân loại trong <code>sim/convcached.ts</code>, vốn theo Hình 7.15 với dòng 21 đã sửa, hoặc
              như được in khi bạn yêu cầu. Màu: <span className="t-shared">bộ nhớ chia sẻ</span>,{' '}
              <span className="t-global">bộ nhớ toàn cục</span>, nét đứt cho ô ma; viền đỏ là những gì dòng 21 như được in đọc.
            </>
          }
        >
          <>
            Computed by the classifier in <code>sim/convcached.ts</code>, which follows Fig. 7.15 with line 21 corrected,
            or as printed when you ask. Colours: <span className="t-shared">shared memory</span>,{' '}
            <span className="t-global">global memory</span>, dashed for ghost cells; red outlines are what printed line 21 reads.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        {sel('TILE_DIM', T, TILES, setTile)}
        {sel('FILTER_RADIUS', r, [1, 2, 3], (v) => setR(v))}
        <label className="field">
          <Translated vi="Chiều rộng ảnh">Image width</Translated>
          <input type="number" min={1} max={200} value={width} aria-label={tr('image width', 'chiều rộng ảnh')}
            onChange={(e) => setWidth(Math.max(1, Math.min(200, Math.floor(+e.target.value || 1))))} />
        </label>
        <label className="field">
          <Translated vi="Chiều cao ảnh">Image height</Translated>
          <input type="number" min={1} max={200} value={height} aria-label={tr('image height', 'chiều cao ảnh')}
            onChange={(e) => setHeight(Math.max(1, Math.min(200, Math.floor(+e.target.value || 1))))} />
        </label>
        {sel(tr('Warp width', 'Độ rộng warp'), warp, warpChoices.map((w) => w.value), setWarp, (v) => warpChoices.find((w) => w.value === v)?.label ?? String(v))}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('Block presets', 'Khối đặt sẵn')}>
          {(['interior', 'edge', 'corner', 'overhang'] as const).map((w) => (
            <button key={w} onClick={() => preset(w)}>
              <Translated vi={w === 'overhang' ? 'khối cuối' : w === 'interior' ? 'bên trong' : w === 'edge' ? 'cạnh' : 'góc'}>
                {w === 'overhang' ? 'last block' : w}
              </Translated>
            </button>
          ))}
        </span>
        {sel('blockIdx.x', cbx, Array.from({ length: gx }, (_, i) => i), setBx)}
        {sel('blockIdx.y', cby, Array.from({ length: gy }, (_, i) => i), setBy)}
        {sel('threadIdx.x', Math.min(tx, T - 1), Array.from({ length: T }, (_, i) => i), setTx)}
        {sel('threadIdx.y', Math.min(ty, T - 1), Array.from({ length: T }, (_, i) => i), setTy)}
        <label className="check">
          <input type="checkbox" checked={printed} onChange={(e) => setPrinted(e.target.checked)} /> <Translated vi="dòng 21 như được in">line 21 as printed</Translated>
        </label>
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
        <div>
          <div className="mx-label">
            <Translated vi={<>Khối ({cbx}, {cby}): ô {T} × {T} của nó và vòng rộng {r} bao quanh. Nhấn vào một luồng.</>}>
              Block ({cbx}, {cby}): its {T} × {T} tile and the {r}-wide ring around it. Click a thread.
            </Translated>
          </div>
          <svg viewBox={`0 0 ${S * c + 2} ${S * c + 2}`} style={{ width: '100%', maxWidth: S * c + 2, display: 'block' }}
            role="img" aria-label={tr("The block's footprint with the selected thread's taps", 'Vùng phủ của khối cùng các tap của luồng đang chọn')}>
            {Array.from({ length: S * S }, (_, k) => {
              const i = Math.floor(k / S);
              const j = k % S;
              const tile = inTile(i, j);
              const img = inImg(i, j);
              const kind = patch.get(k);
              const fill = kind ? KIND_FILL[kind] : tile ? (img ? 'var(--shared-wash)' : 'var(--cell)') : img ? 'var(--global-wash)' : 'transparent';
              const dashed = !img;
              return (
                <rect key={k} x={1 + j * c + 0.5} y={1 + i * c + 0.5} width={c - 1} height={c - 1} fill={fill}
                  stroke={dashed ? 'var(--ink-3)' : tile ? 'var(--shared)' : 'var(--global)'} strokeWidth={0.5}
                  strokeDasharray={dashed ? '2 2' : undefined} opacity={kind || tile || img ? 1 : 0.7}
                  style={{ cursor: tile ? 'pointer' : 'default' }} data-kind={kind ?? undefined}
                  onClick={tile ? () => { setTx(j - r); setTy(i - r); } : undefined} />
              );
            })}
            {/* ghost taps of the thread: a cross */}
            {thread.taps.filter((p) => p.kind === 'ghost').map((p) => {
              const x = 1 + (thread.tx + p.fCol) * c;
              const y = 1 + (thread.ty + p.fRow) * c;
              return <path key={`g${p.fRow},${p.fCol}`} d={`M${x + 2} ${y + 2}L${x + c - 2} ${y + c - 2}M${x + c - 2} ${y + 2}L${x + 2} ${y + c - 2}`} stroke="var(--ink-3)" strokeWidth="1" />;
            })}
            {/* the tile's edge */}
            <rect x={1 + r * c} y={1 + r * c} width={T * c} height={T * c} fill="none" stroke="var(--shared)" strokeWidth="1.6" />
            {/* the selected thread */}
            <rect x={1 + (thread.tx + r) * c - 1} y={1 + (thread.ty + r) * c - 1} width={c + 2} height={c + 2} fill="none" stroke="var(--ink)" strokeWidth="2" />
            {/* what printed line 21 reads */}
            {[...printedAt].map(([k, how]) => (
              <rect key={'p' + k} x={1 + (k % S) * c + 1.5} y={1 + Math.floor(k / S) * c + 1.5} width={c - 3} height={c - 3}
                fill={how === 'outside' ? 'var(--fail-wash)' : 'none'} stroke="var(--fail)" strokeWidth="1.6"
                strokeDasharray={how === 'wrong' ? undefined : '3 1.5'} data-printed={how} />
            ))}
          </svg>
        </div>

        <div>
          <div className="panel" role="status" data-thread={`${thread.tx},${thread.ty}`}>
            {thread.active ? (
              <>
                <p style={{ margin: '0 0 0.4rem' }}>
                  <Translated
                    vi={
                      <>
                        Luồng ({thread.tx}, {thread.ty}) của khối ({cbx}, {cby}) tính <code>P[{thread.row}][{thread.col}]</code>.
                        {m * m} tap của nó: <span className="chip shared" data-mine-shared={mine.shared}>{mine.shared} chia sẻ</span>{' '}
                        <span className="chip global" data-mine-global={mine.global}>{mine.global} toàn cục</span>{' '}
                        <span className="chip" style={{ border: '1px dashed var(--ink-3)' }}>{mine.ghost} ô ma</span>
                      </>
                    }
                  >
                    Thread ({thread.tx}, {thread.ty}) of block ({cbx}, {cby}) computes <code>P[{thread.row}][{thread.col}]</code>.
                    Its {m * m} taps: <span className="chip shared" data-mine-shared={mine.shared}>{mine.shared} shared</span>{' '}
                    <span className="chip global" data-mine-global={mine.global}>{mine.global} global</span>{' '}
                    <span className="chip" style={{ border: '1px dashed var(--ink-3)' }}>{mine.ghost} ghost</span>
                  </Translated>
                </p>
                {printed ? (
                  <p style={{ margin: '0 0 0.4rem' }} data-thread-printed={`${tWrong},${tPast},${tOut}`}>
                    <Translated
                      vi={
                        <>
                          Như được in, dòng 21 đưa <b className="t-fail">{tWrong}</b> trong {mine.shared} tap chia sẻ của nó tới một
                          phần tử khác: <code>N_s[ty+fRow][tx+fCol]</code> lệch {r} hàng xuống dưới và {r} cột sang phải so với ô đúng.{' '}
                          {tPast ? <>{tPast} trong số đó đánh chỉ số vượt quá ô{tOut ? <>, và {tOut} rơi hẳn ra ngoài <code>N_s</code></> : null}.</> : 'Không tap nào vượt quá ô.'}
                        </>
                      }
                    >
                      As printed, line 21 sends <b className="t-fail">{tWrong}</b> of its {mine.shared} shared taps to another
                      element: <code>N_s[ty+fRow][tx+fCol]</code> is {r} rows and {r} columns down and right of the right slot.{' '}
                      {tPast ? <>{tPast} of them index past the tile{tOut ? <>, and {tOut} land outside <code>N_s</code> altogether</> : null}.</> : 'None runs past the tile.'}
                    </Translated>
                  </p>
                ) : null}
              </>
            ) : (
              <p style={{ margin: 0 }}>
                <Translated
                  vi={
                    <>
                      Luồng ({thread.tx}, {thread.ty}) của khối này ánh xạ tới <code>P[{thread.row}][{thread.col}]</code>, nằm ngoài
                      ảnh {width} × {height}. Nó nạp một số không ô ma vào <code>N_s</code>, chờ ở rào chắn, và không tính gì.
                    </>
                  }
                >
                  Thread ({thread.tx}, {thread.ty}) of this block maps to <code>P[{thread.row}][{thread.col}]</code>, outside the
                  {' '}{width} × {height} image. It loads a ghost zero into <code>N_s</code>, waits at the barrier, and computes nothing.
                </Translated>
              </p>
            )}
          </div>
          {thread.active ? (
            <table className="threads" aria-label={tr('The printed guard, x half', 'Điều kiện chặn như được in, nửa theo x')} style={{ marginTop: '0.6rem' }}>
              <thead>
                <tr>
                  <th>fCol</th>
                  <th>tx−R+fCol</th>
                  <th><Translated vi="dạng không dấu">as unsigned</Translated></th>
                  <th>&gt;= 0</th>
                  <th>&lt; TILE_DIM</th>{/* i18n-ignore: code expression */}
                </tr>
              </thead>
              <tbody>
                {guardRows.map((p) => (
                  <tr key={p.fCol} data-fcol={p.fCol}>
                    <td>{p.fCol}</td>
                    <td>{p.gx.signed}</td>
                    <td data-u32={p.gx.u32} className={p.gx.signed < 0 ? 't-fail' : ''}>{p.gx.u32}</td>
                    <td>{String(p.gx.geZero)}</td>
                    <td>{String(p.gx.ltTile)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
        </div>
      </div>

      <div className="cols" style={{ marginTop: '1rem', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
        <div className="panel" data-hist={`${hist.shared},${hist.global},${hist.ghost},${hist.taps}`}>
          <div className="mx-label"><Translated vi="Toàn bộ khối">The whole block</Translated></div>
          <p style={{ margin: '0 0 0.4rem' }}>
            <Translated
              vi={
                <>
                  {hist.active} trên {hist.threads} luồng tính toán; {n(hist.taps)} tap:{' '}
                  <span className="chip shared">{n(hist.shared)} chia sẻ</span>{' '}
                  <span className="chip global">{n(hist.global)} toàn cục</span>{' '}
                  <span className="chip" style={{ border: '1px dashed var(--ink-3)' }}>{n(hist.ghost)} ô ma</span>
                </>
              }
            >
              {hist.active} of {hist.threads} threads compute; {n(hist.taps)} taps:{' '}
              <span className="chip shared">{n(hist.shared)} shared</span>{' '}
              <span className="chip global">{n(hist.global)} global</span>{' '}
              <span className="chip" style={{ border: '1px dashed var(--ink-3)' }}>{n(hist.ghost)} ghost</span>
            </Translated>
          </p>
          <svg viewBox="0 0 300 12" style={{ width: '100%', display: 'block', margin: '0.2rem 0 0.5rem' }} aria-hidden="true">
            <rect x="0" y="0" width={hist.taps ? (300 * hist.shared) / hist.taps : 0} height="12" fill="var(--shared)" />
            <rect x={hist.taps ? (300 * hist.shared) / hist.taps : 0} y="0" width={hist.taps ? (300 * hist.global) / hist.taps : 0} height="12" fill="var(--global)" />
          </svg>
          <p style={{ margin: 0 }}>
            <Translated
              vi={
                <>
                  {hist.taps ? <>{pct(hist.global / hist.taps)} số tap rời khỏi bộ nhớ chia sẻ. </> : null}
                  Chúng đọc {hist.distinctGlobal} phần tử vùng biên khác nhau. Số lần đọc <code>N</code> mỗi đầu ra nếu không có
                  bộ nhớ đệm nào giúp:{' '}
                  <b data-reads-all={hist.readsPerOutputAllTaps.toFixed(2)}>{hist.readsPerOutputAllTaps.toFixed(2)}</b>; nếu khối
                  nạp mỗi phần tử vùng biên một lần: <b>{hist.readsPerOutputPerBlock.toFixed(2)}</b>.
                  {hist.zeroInTile ? <> {n(hist.zeroInTile)} tap chia sẻ đọc số không ô ma, vì ô này chìa ra ngoài ảnh.</> : null}
                </>
              }
            >
              {hist.taps ? <>{pct(hist.global / hist.taps)} of the taps leave shared memory. </> : null}
              They read {hist.distinctGlobal} different halo elements. Reads of <code>N</code> per output if no cache helps at all:{' '}
              <b data-reads-all={hist.readsPerOutputAllTaps.toFixed(2)}>{hist.readsPerOutputAllTaps.toFixed(2)}</b>; if the
              block fetches each halo element once: <b>{hist.readsPerOutputPerBlock.toFixed(2)}</b>.
              {hist.zeroInTile ? <> {n(hist.zeroInTile)} shared taps read ghost zeros, because this tile overhangs the image.</> : null}
            </Translated>
          </p>
          {printed ? (
            <p style={{ margin: '0.5rem 0 0' }} data-block-printed={`${hist.printedWrong},${hist.printedPastTile},${hist.printedOutside}`}>
              <Translated
                vi={
                  <>
                    Như được in: <b className="t-fail">{n(hist.printedWrong)}</b> tap chia sẻ đọc sai phần tử,{' '}
                    {n(hist.printedPastTile)} đánh chỉ số vượt quá ô, {n(hist.printedOutside)} rơi ra ngoài <code>N_s</code>.
                  </>
                }
              >
                As printed: <b className="t-fail">{n(hist.printedWrong)}</b> shared taps read the wrong element,{' '}
                {n(hist.printedPastTile)} index past the tile, {n(hist.printedOutside)} fall outside <code>N_s</code>.
              </Translated>
            </p>
          ) : null}
        </div>
        <div className="panel" data-warps={`${warps.diverging},${warps.warps.length}`}>
          <div className="mx-label">
            <Translated vi={<>Các warp {warp} luồng tại phép kiểm tra trong-ô (dòng 17 đến 20)</>}>
              Warps of {warp} threads at the in-tile test (lines 17 to 20)
            </Translated>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, margin: '0.3rem 0 0.5rem' }}>
            {warps.warps.map((w) => (
              <span key={w.warp} title={tr(`warp ${w.warp}: threads ${w.first} to ${w.last}; lanes disagree on ${w.splitTaps} of ${m * m} taps`, `warp ${w.warp}: luồng ${w.first} đến ${w.last}; các lane bất đồng ở ${w.splitTaps} trên ${m * m} tap`)}
                style={{
                  minWidth: '1.7rem', textAlign: 'center', fontSize: '0.72rem', padding: '0.1rem 0.15rem', borderRadius: 2,
                  border: '1px solid ' + (w.diverges ? 'var(--ink)' : 'var(--rule)'), background: w.diverges ? 'var(--rule-soft)' : 'var(--cell)',
                }}>
                {w.splitTaps}
              </span>
            ))}
          </div>
          <p style={{ margin: 0 }}>
            <Translated
              vi={
                <>
                  <b>{warps.diverging} trên {warps.warps.length}</b> warp có các lane bất đồng ở ít nhất một tap. Mỗi ô là một
                  warp; con số là số tap trong {m * m} tap tách nó ra. Phép kiểm tra chỉ phụ thuộc vào <code>threadIdx</code>, nên
                  điều này giống nhau ở mọi khối.
                </>
              }
            >
              <b>{warps.diverging} of {warps.warps.length}</b> warps have lanes that disagree on at least one tap. Each box is a
              warp; the number is how many of its {m * m} taps split it. The test depends only on <code>threadIdx</code>, so this
              is the same for every block.
            </Translated>
          </p>
        </div>
      </div>

      <p style={{ margin: '0.9rem 0 0' }} data-run={`${run.wrong.length},${width * height},${run.pastTileReads},${run.outsideReads}`}>
        {run.correct ? (
          <span className="chip ok">
            <Translated vi={<>Toàn bộ kernel, chạy trên ảnh {width} × {height} này: mọi đầu ra đều khớp với giá trị tham chiếu.</>}>
              The whole kernel, run on this {width} × {height} image: every output matches the reference.
            </Translated>
          </span>
        ) : (
          <Translated
            vi={
              <>
                <span className="chip fail">
                  Toàn bộ kernel như được in: {n(run.wrong.length)} trên {n(width * height)} đầu ra sai
                </span>{' '}
                {n(run.pastTileReads)} lần đọc vượt quá ô sang hàng kế tiếp của <code>N_s</code>; {n(run.outsideReads)} rơi ra
                ngoài nó (hành vi không xác định; mô hình trả về giá trị rác).
              </>
            }
          >
            <>
              <span className="chip fail">
                The whole kernel as printed: {n(run.wrong.length)} of {n(width * height)} outputs wrong
              </span>{' '}
              {n(run.pastTileReads)} reads ran past the tile into the next row of <code>N_s</code>; {n(run.outsideReads)} fell
              outside it (undefined behaviour; the model returns garbage).
            </>
          </Translated>
        )}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer 2: will the halo be in the cache?

/** The L2 size your CUDA or HIP device reported to the doctor, where it did (0 means not reported, ch07-facts B5). */
function YourL2() {
  const { reports } = useReports();
  const sizes = gpuBackends(reports).flatMap((b) => {
    const v = Number(reports[b].device?.l2CacheSize);
    return isFinite(v) && v > 0 ? [tr(`${fmt(v / (1024 * 1024))} MB on ${String(reports[b].device.name ?? b)} (${b})`, `${fmt(v / (1024 * 1024))} MB trên ${String(reports[b].device.name ?? b)} (${b})`)] : [];
  });
  return sizes.length ? (
    <span data-your-l2>
      <Translated vi={<>L2 của bạn, như thiết bị của bạn đã báo: {sizes.join('; ')}.</>}>
        Your L2, as your device reported it: {sizes.join('; ')}.
      </Translated>
    </span>
  ) : (
    <span data-your-l2="">
      <Translated
        vi={
          <>
            Báo cáo thiết bị của bạn không có kích thước L2 (Apple không ghi; trên CUDA đó là thuộc tính thiết bị{' '}
            <code>l2CacheSize</code>).
          </>
        }
      >
        Your device report has no L2 size (Apple documents none; on CUDA it is the device property{' '}
        <code>l2CacheSize</code>).
      </Translated>
    </span>
  );
}

const CAPS = [0, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, Infinity];
const PLACEMENT_LABEL: Record<Placement, string> = {
  get 'in-order'() {
    return tr('in order (each wave split into runs)', 'theo thứ tự (mỗi đợt chia thành các dải liên tiếp)');
  },
  get 'round-robin'() {
    return tr('round-robin', 'xoay vòng');
  },
  get random() {
    return tr('random (seeded)', 'ngẫu nhiên (có seed)');
  },
};

function CacheExplorer() {
  const [tilesX, setTilesX] = useState(8);
  const [tilesY, setTilesY] = useState(8);
  const [tile, setTile] = useState(32);
  const [r, setR] = useState(2);
  const [lineFloats, setLineFloats] = useState(32);
  const [capIdx, setCapIdx] = useState(CAPS.indexOf(1024));
  const [resident, setResident] = useState(8);
  const [slices, setSlices] = useState(1);
  const [placement, setPlacement] = useState<Placement>('in-order');
  const [seed, setSeed] = useState(1);
  const capacity = CAPS[capIdx];
  const cfg: CacheConfig = { tilesX, tilesY, tile, r, lineFloats, capacity, resident, slices, placement, seed };
  const s = useMemo(
    () => summarise({ tilesX, tilesY, tile, r, lineFloats, capacity, resident, slices, placement, seed }),
    [tilesX, tilesY, tile, r, lineFloats, capacity, resident, slices, placement, seed],
  );
  const { width, height } = imageDims(cfg);
  const rowLines = linesPerTileRow(cfg);
  const kb = (lines: number) => (lines * lineBytes(lineFloats)) / 1024;

  const cellPx = Math.max(14, Math.min(36, Math.floor(340 / Math.max(tilesX, tilesY))));
  const bars: { id: keyof typeof s.bytes; label: ReactNode }[] = [
    { id: 'ideal', label: tr('the ideal: every line from DRAM once', 'lý tưởng: mỗi line từ DRAM một lần') },
    { id: 'run', label: <b><Translated vi="lần chạy này (Hình 7.15 qua bộ nhớ đệm này)">this run (Fig. 7.15 through this cache)</Translated></b> },
    { id: 'allMiss', label: tr('Fig. 7.15, no reuse between blocks (all miss)', 'Hình 7.15, không tái sử dụng giữa các khối (trượt hết)') },
    { id: 'fig712', label: tr("Fig. 7.12, no reuse between blocks (the book's count)", 'Hình 7.12, không tái sử dụng giữa các khối (cách đếm của sách)') },
    { id: 'fig712Cached', label: tr('Fig. 7.12 through this same cache', 'Hình 7.12 qua cùng bộ nhớ đệm này') },
  ];
  const max = Math.max(...bars.map((b) => (isFinite(s.bytes[b.id]) ? s.bytes[b.id] : 0)));
  const sel = (label: string, value: number, options: number[], on: (v: number) => void, fmtOpt: (v: number) => string = String) => (
    <label className="field">
      {label}
      <select value={value} onChange={(e) => on(+e.target.value)}>
        {options.map((o) => <option key={o} value={o}>{fmtOpt(o)}</option>)}
      </select>
    </label>
  );

  return (
    <Bench
      title={tr('Will the halo be in the cache?', 'Vùng biên có nằm trong bộ nhớ đệm không?')}
      task={tr(
        'Shrink the cache until traffic climbs; then restore it and split it across dies with random placement.',
        'Thu nhỏ bộ nhớ đệm cho đến khi lưu lượng tăng vọt; rồi khôi phục nó và chia nó qua các die với cách đặt ngẫu nhiên.',
      )}
      foot={
        <Translated
          vi={
            <>
              Mô hình đồ chơi của khóa học này (<code>sim/halocache.ts</code>), không phải của sách và không phải của hãng
              nào; hãy đọc những gì nó bỏ qua bên cạnh trình khám phá này. Kích thước line và dung lượng là do bạn chọn; không
              có số liệu của hãng nào được cài sẵn.{' '}
              <YourL2 />
            </>
          }
        >
          <>
            Our toy model (<code>sim/halocache.ts</code>), not the book's and not any vendor's; read what it leaves out
            beside this explorer. Line size and capacity are yours to choose; no vendor's figures are built in.{' '}
            <YourL2 />
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        {sel(tr('Image width, tiles', 'Chiều rộng ảnh, theo ô'), tilesX, [2, 3, 4, 6, 8, 12, 16], setTilesX)}
        {sel(tr('height, tiles', 'chiều cao, theo ô'), tilesY, [2, 3, 4, 6, 8, 12, 16], setTilesY)}
        {sel('TILE_DIM', tile, [8, 16, 32], setTile)}
        {sel('FILTER_RADIUS', r, [1, 2, 3], setR)}
        {sel(tr('Line size', 'Kích thước line'), lineFloats, [8, 16, 32], setLineFloats, (v) => tr(`${lineBytes(v)} bytes`, `${lineBytes(v)} byte`))}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field" style={{ minWidth: '15rem' }}>
          <span>
            <Translated vi="Dung lượng bộ nhớ đệm:">Cache capacity:</Translated>{' '}
            <b data-capacity={isFinite(capacity) ? capacity : 'unbounded'}>
              {isFinite(capacity) ? tr(`${n(capacity)} lines (${fmt(kb(capacity))} KB, ${fmt(capacity / rowLines, 2)} rows of tiles)`, `${n(capacity)} line (${fmt(kb(capacity))} KB, ${fmt(capacity / rowLines, 2)} hàng ô)`) : tr('unbounded', 'không giới hạn')}
            </b>
          </span>
          <input type="range" min={0} max={CAPS.length - 1} value={capIdx} aria-label={tr('cache capacity', 'dung lượng bộ nhớ đệm')} onChange={(e) => setCapIdx(+e.target.value)} />
        </label>
        {sel(tr('Blocks resident at once', 'Số khối thường trú cùng lúc'), resident, [1, 2, 4, 8, 16, 32, 64], setResident)}
        {sel(tr('L2 slices', 'Số lát L2'), slices, [1, 2, 4, 8], setSlices, (v) => (v === 1 ? tr('1 (one L2 for all)', '1 (một L2 cho tất cả)') : tr(`${v} (one per die)`, `${v} (mỗi die một lát)`)))}
        <label className="field">
          <Translated vi="Cách đặt lên các lát">Placement on slices</Translated>
          <select value={placement} disabled={slices === 1} onChange={(e) => setPlacement(e.target.value as Placement)}>
            {(Object.keys(PLACEMENT_LABEL) as Placement[]).map((p) => <option key={p} value={p}>{PLACEMENT_LABEL[p]}</option>)}
          </select>
        </label>
        {placement === 'random' && slices > 1 ? (
          <label className="field">
            <Translated vi="Seed (hạt giống ngẫu nhiên)">Seed</Translated>
            <input type="number" min={1} max={999} value={seed} aria-label={tr('seed', 'seed ngẫu nhiên')} style={{ width: '5rem' }}
              onChange={(e) => setSeed(Math.max(1, Math.floor(+e.target.value || 1)))} />
          </label>
        ) : null}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 3fr)' }}>
        <div>
          <div className="mx-label">
            <Translated vi={<>Các line vùng biên trúng cache, theo từng khối ({width} × {height} float)</>}>
              Halo lines that hit, per block ({width} × {height} floats)
            </Translated>
          </div>
          <div style={{ display: 'inline-grid', gridTemplateColumns: `repeat(${s.run.gx}, ${cellPx}px)`, gap: 2 }} data-heat={s.run.blocks.length}>
            {s.run.blocks.map((b) => {
              const f = b.haloHitFraction;
              return (
                <span key={b.block} title={tr(`block (${b.bx}, ${b.by}), slice ${b.slice}: ${b.haloHits} of ${b.haloLines} halo lines hit`, `khối (${b.bx}, ${b.by}), lát ${b.slice}: ${b.haloHits} trên ${b.haloLines} line vùng biên trúng cache`)}
                  data-block-hit={f == null ? '' : f.toFixed(3)}
                  style={{
                    width: cellPx, height: cellPx, display: 'grid', placeItems: 'center', position: 'relative',
                    fontSize: cellPx >= 26 ? '0.66rem' : '0.5rem', fontVariantNumeric: 'tabular-nums',
                    background: f == null ? 'var(--cell)' : `color-mix(in srgb, var(--ok) ${Math.round(f * 85)}%, var(--cell))`,
                    border: '1px solid var(--rule)', borderRadius: 2, color: 'var(--ink)',
                  }}>
                  {f == null ? '–' : cellPx >= 22 ? Math.round(f * 100) : ''}
                  {slices > 1 && cellPx >= 22 ? (
                    <sub style={{ position: 'absolute', right: 1, bottom: 0, fontSize: '0.5rem', color: 'var(--ink-3)' }}>{b.slice}</sub>
                  ) : null}
                </span>
              );
            })}
          </div>
          <p className="muted" style={{ margin: '0.4rem 0 0' }}>
            <Translated
              vi={
                <>
                  Phần trăm số line vùng biên của mỗi khối được tìm thấy trong L2 của nó{slices > 1 ? '; con số nhỏ là lát mà khối chạy trên đó' : ''}.
                  Trên mọi khối: <b data-halo-hit={s.haloHitFraction == null ? '' : s.haloHitFraction.toFixed(3)}>
                    {s.haloHitFraction == null ? 'không có vùng biên' : pct(s.haloHitFraction, 0)}</b>.
                </>
              }
            >
              Percent of each block's halo lines found in its L2{slices > 1 ? '; the small number is the slice it ran on' : ''}.
              Over all blocks: <b data-halo-hit={s.haloHitFraction == null ? '' : s.haloHitFraction.toFixed(3)}>
                {s.haloHitFraction == null ? 'no halo' : pct(s.haloHitFraction, 0)}</b>.
            </Translated>
          </p>
        </div>
        <div>
          <div className="mx-label">
            <Translated vi={<>Số byte DRAM mỗi đầu ra (các lần nạp và lệnh ghi), và FLOP trên mỗi byte tại m = {2 * r + 1}</>}>
              DRAM bytes per output (loads and the store), and FLOP per byte at m = {2 * r + 1}
            </Translated>
          </div>
          {bars.map((b) => {
            const v = s.bytes[b.id];
            return (
              <div key={b.id} data-bar={b.id} data-bytes={isFinite(v) ? v.toFixed(4) : ''} data-lines={isFinite(s.lines[b.id]) ? s.lines[b.id] : ''}
                style={{ margin: '0 0 0.45rem' }}>
                <div style={{ fontSize: '0.78rem' }}>{b.label}</div>
                {isFinite(v) ? (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span style={{ display: 'inline-block', height: 12, width: `${(70 * v) / max}%`, background: b.id === 'run' ? 'var(--global)' : 'var(--global-wash)', border: '1px solid var(--global)' }} />
                    <span style={{ whiteSpace: 'nowrap', fontVariantNumeric: 'tabular-nums' }}>
                      {fmt(v)} B, {fmt(s.intensity[b.id])} FLOP/B
                    </span>
                  </div>
                ) : (
                  <span className="muted"><Translated vi={<>không còn ô đầu ra (T − 2r &lt; 1)</>}>no output tile left (T − 2r &lt; 1)</Translated></span>
                )}
              </div>
            );
          })}
          <p className="muted" style={{ margin: '0.3rem 0 0' }}>
            <Translated
              vi={
                <>
                  Lần chạy này lấy <b data-misses={s.lines.run}>{n(s.lines.run)}</b> line {lineBytes(lineFloats)} byte từ DRAM:
                  mức lý tưởng là {n(s.lines.ideal)}, không tái sử dụng giữa các khối là {n(s.lines.allMiss)}.
                </>
              }
            >
              This run fetched <b data-misses={s.lines.run}>{n(s.lines.run)}</b> lines of {lineBytes(lineFloats)} bytes from DRAM:
              the ideal is {n(s.lines.ideal)}, no reuse between blocks {n(s.lines.allMiss)}.
            </Translated>
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- measured on the reader's machine

function labRows(reports: Reports, backend: string): BenchRow[] {
  const s = benchRows(reports[backend], LAB72.id, 'solution');
  return s.length ? s : benchRows(reports[backend], LAB72.id, 'work');
}

function Measured() {
  const { reports, loading } = useReports();
  const sets = gpuBackends(reports).flatMap((b) => {
    const rs = labRows(reports, b);
    const cells = rs
      .filter((x) => x.name === 'tiled_cached')
      .flatMap((c) => {
        const h = rs.find((x) => x.name === 'tiled_halo' && num(x.tile) === num(c.tile) && num(x.radius) === num(c.radius));
        const gc = num(c.gflops);
        const gh = num(h?.gflops);
        const tile = num(c.tile);
        const radius = num(c.radius);
        return gc && gh && tile && radius ? [{ tile, radius, gc, gh }] : [];
      });
    return cells.length ? [{ b, device: String(reports[b].device.name ?? b), cells }] : [];
  });
  if (loading)
    return (
      <p className="muted">
        <Translated vi="Đang đọc báo cáo.">Reading reports.</Translated>
      </p>
    );
  if (!sets.length)
    return (
      <Translated
        vi={
          <p data-measured="none">
            Máy này chưa có số đo nào của lab 7.2. Hãy chạy <code>python3 labs/pmpp.py run 7.2 --solution --bench</code>,
            rồi tải lại trang này: đoạn này sẽ cho thấy, với mỗi kích thước ô và bán kính, tốc độ của kernel dùng bộ nhớ đệm
            so với kernel phân ô có vùng biên trên GPU của bạn.
          </p>
        }
      >
        <p data-measured="none">
          No measurement of lab 7.2 on this machine yet. Run <code>python3 labs/pmpp.py run 7.2 --solution --bench</code>,
          then reload this page: this paragraph will show, for each tile and radius, how the cached kernel's speed
          compares with the halo-tiled kernel's on your GPU.
        </p>
      </Translated>
    );
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }} data-measured={sets.reduce((a, s) => a + s.cells.length, 0)}>
      {sets.map((set) => {
        const tiles = [...new Set(set.cells.map((c) => c.tile))].sort((a, b) => a - b);
        const radii = [...new Set(set.cells.map((c) => c.radius))].sort((a, b) => a - b);
        return (
          <table key={set.b} className="data" style={{ maxWidth: '40rem' }}>
            <caption style={{ textAlign: 'left', fontWeight: 600, paddingBottom: '0.3rem' }}>
              <Translated vi={<>{set.device} ({set.b}): GFLOP/s của Hình 7.15 chia cho của Hình 7.12, cả hai từ lab 7.2</>}>
                {set.device} ({set.b}): Fig. 7.15's GFLOP/s over Fig. 7.12's, both from lab 7.2
              </Translated>
            </caption>
            <thead>
              <tr>
                <th><Translated vi="bán kính">radius</Translated></th>
                {tiles.map((t) => <th key={t} className="num">TILE_DIM {t}</th>)}
              </tr>
            </thead>
            <tbody>
              {radii.map((r) => (
                <tr key={r}>
                  <td>{r}</td>
                  {tiles.map((t) => {
                    const c = set.cells.find((x) => x.tile === t && x.radius === r);
                    return (
                      <td key={t} className="num" data-tile={t} data-radius={r} data-ratio={c ? (c.gc / c.gh).toFixed(3) : ''}
                        title={c ? tr(`Fig. 7.15: ${fmt(c.gc)} GFLOP/s, Fig. 7.12: ${fmt(c.gh)} GFLOP/s`, `Hình 7.15: ${fmt(c.gc)} GFLOP/s, Hình 7.12: ${fmt(c.gh)} GFLOP/s`) : tr('not measured', 'chưa đo')}>
                        {c ? (c.gc >= c.gh ? <b>{fmt(c.gc / c.gh, 2)}×</b> : <>{fmt(c.gc / c.gh, 2)}×</>) : <span className="muted">–</span>}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        );
      })}
      <p className="status-line" style={{ margin: '-0.8rem 0 1.4rem' }}>
        <Translated
          vi={
            <>
              Số đo từ các lần chạy lab 7.2 trên máy này, không phải giá trị kỳ vọng (rê chuột lên một ô để xem hai con số
              GFLOP/s). Trên 1 (in đậm), kernel dùng bộ nhớ đệm nhanh hơn ở thiết lập đó; dưới 1, việc nạp trước vùng biên
              thắng. Chỉ đo thời gian thì không cho biết lý do; tỉ lệ trúng L2 của một profiler sẽ giúp ích.
            </>
          }
        >
          Measurements from this machine's lab 7.2 runs, not expected values (hover a cell for the two GFLOP/s figures).
          Above 1 (in bold), the cached kernel was faster at that setting; below 1, staging the halo won. Timing alone does not say
          why; a profiler's L2 hit rate helps.
        </Translated>
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- questions

const B32 = readsPerOutput(32, 2);
const H32 = blockHistogram({ tile: 32, r: 2, width: 96, height: 96, bx: 1, by: 1 });

export const questionsS76: Record<string, Question> = {
  'ch07/7.6/q-printed': {
    kind: 'choice',
    id: 'ch07/7.6/q-printed',
    prompt: (
      <Translated
        vi={
          <>
            Dòng 21 như được in của Hình 7.15 đọc <code>N_s[threadIdx.y+fRow][threadIdx.x+fCol]</code>, dưới một điều kiện
            chặn kiểm tra <code>threadIdx.x-FILTER_RADIUS+fCol</code> và <code>threadIdx.y-FILTER_RADIUS+fRow</code>. Với{' '}
            <code>FILTER_RADIUS = 2</code>, một luồng ở bên trong đọc gì ở một tap vượt qua điều kiện chặn?
          </>
        }
      >
        <>
          Printed line 21 of Fig. 7.15 reads <code>N_s[threadIdx.y+fRow][threadIdx.x+fCol]</code>, under a guard that tests{' '}
          <code>threadIdx.x-FILTER_RADIUS+fCol</code> and <code>threadIdx.y-FILTER_RADIUS+fRow</code>. With{' '}
          <code>FILTER_RADIUS = 2</code>, what does an interior thread read on a tap that passes the guard?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Đúng phần tử: điều kiện chặn và chỉ số gọi cùng một ô, viết theo hai cách.">
            The right element: the guard and the index name the same slot, written two ways.
          </Translated>
        ),
        why: (
          <Translated vi="Hãy so sánh chúng: điều kiện chặn trừ FILTER_RADIUS, còn chỉ số thì không.">
            Compare them: the guard subtracts FILTER_RADIUS, the index does not.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một phần tử nằm dưới 2 hàng và sang phải 2 cột so với phần tử nó cần, ở mọi tap như vậy.">
            An element 2 rows down and 2 columns right of the one it needs, on every such tap.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Một phần tử nằm trên 2 hàng và sang trái 2 cột so với phần tử nó cần, ở mọi tap như vậy.">
            An element 2 rows up and 2 columns left of the one it needs, on every such tap.
          </Translated>
        ),
        why: (
          <Translated vi="Chỉ số được in lớn hơn chỉ số đúng, không phải nhỏ hơn.">
            The printed index is larger than the right one, not smaller.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Đúng phần tử, trừ các luồng gần cạnh ô, vốn có tap chạy vượt quá ô.">
            {'The right element, except for threads near the tile\'s edge, whose taps run past the tile.'}
          </Translated>
        ),
        why: (
          <Translated vi="Độ lệch giống nhau cho mọi luồng; chỉ việc chạy vượt quá ô là giới hạn ở vùng cạnh.">
            The shift is the same for every thread; only running past the tile is limited to the edge.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Tap cần <code>N_s[ty-R+fRow][tx-R+fCol]</code>, chính là thứ điều kiện chặn kiểm tra. Chỉ số được in cộng thêm R
            vào cả hai tọa độ. Mọi lần đọc bộ nhớ chia sẻ đều bị lệch, nên mọi đầu ra đều sai; và vì điều kiện chặn cho phép{' '}
            <code>tx-R+fCol</code> tới <code>TILE_DIM-1</code>, cột được in với tới <code>TILE_DIM+R-1</code>. Trong một mảng
            thứ tự hàng (row-major), điều đó đọc sang hàng kế tiếp, và từ các hàng cuối thì nó ra khỏi <code>N_s</code>. Trong
            một khối 32 × 32 bên trong với bán kính 2, mô hình đếm được {n(H32.printedPastTile)} lần đọc vượt quá ô,{' '}
            {n(H32.printedOutside)} trong số đó nằm ngoài mảng.
          </p>
        }
      >
        <p>
          The tap needs <code>N_s[ty-R+fRow][tx-R+fCol]</code>, which is what the guard checks. The printed index adds R to
          both coordinates. Every shared read is shifted, so every output is wrong; and since the guard allows{' '}
          <code>tx-R+fCol</code> up to <code>TILE_DIM-1</code>, the printed column reaches <code>TILE_DIM+R-1</code>. In a
          row-major array that reads the next row, and from the last rows it leaves <code>N_s</code>. In a 32 × 32 interior
          block at radius 2 the model counts {n(H32.printedPastTile)} reads past the tile, {n(H32.printedOutside)} of them
          outside the array.
        </p>
      </Translated>
    ),
    source: 'notes/ch07.md, Fig. 7.15 and "Unclear in the PDF"; facts ch07 A8; counts from sim/convcached.ts',
    vi: { source: 'notes/ch07.md, Hình 7.15 và "Unclear in the PDF"; facts ch07 A8; số đếm từ sim/convcached.ts' },
  },
  'ch07/7.6/q-unsigned': {
    kind: 'choice',
    id: 'ch07/7.6/q-unsigned',
    prompt: (
      <Translated
        vi={
          <>
            <code>threadIdx.x = 0</code>, <code>FILTER_RADIUS = 2</code>, <code>fCol = 0</code>. Trong CUDA C++,{' '}
            <code>threadIdx.x - FILTER_RADIUS + fCol &gt;= 0</code> cho kết quả gì?
          </>
        }
      >
        <>
          <code>threadIdx.x = 0</code>, <code>FILTER_RADIUS = 2</code>, <code>fCol = 0</code>. In CUDA C++, what is{' '}
          <code>threadIdx.x - FILTER_RADIUS + fCol &gt;= 0</code>?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="false: giá trị là −2, nhỏ hơn 0.">false: the value is −2, which is less than 0.</Translated>,
        why: (
          <Translated vi="Đó là điều được dự định. threadIdx.x là số không dấu, và cả biểu thức cũng vậy.">
            That is what was meant. threadIdx.x is unsigned, and so is the whole expression.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="true: biểu thức là số không dấu và quay vòng thành 4294967294.">
            true: the expression is unsigned and wraps to 4294967294.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không biên dịch được: bạn không thể so sánh một giá trị không dấu với 0.">
            It does not compile: you cannot compare an unsigned value with 0.
          </Translated>
        ),
        why: (
          <Translated vi="Nó biên dịch được; clang chỉ cảnh báo rằng phép so sánh luôn đúng.">
            It compiles; clang only warns that the comparison is always true.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Hành vi không xác định: phép trừ tràn xuống dưới 0.">
            Undefined behaviour: the subtraction overflows below zero.
          </Translated>
        ),
        why: (
          <Translated vi="Theo định nghĩa, phép tính không dấu quay vòng theo modulo 2^32; chỉ tràn số có dấu mới là không xác định.">
            Unsigned arithmetic wraps modulo 2^32 by definition; only signed overflow is undefined.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trộn <code>unsigned int</code> với <code>int</code> sẽ chuyển <code>int</code> thành không dấu, nên 0 − 2 quay
            vòng thành 2^32 − 2 = 4294967294, và <code>&gt;= 0</code> đúng với mọi giá trị không dấu. Nửa còn lại,{' '}
            <code>&lt; TILE_DIM</code>, sai với giá trị đã quay vòng, nên điều kiện chặn vẫn loại bỏ tap: nó chạy đúng nhờ
            may mắn. Hãy sao chép chỉ số vào một <code>int</code> trước, như sách đã làm cho <code>row</code> và{' '}
            <code>col</code>, và như lab 7.2 làm. Các thuộc tính vị trí luồng của Metal cũng là số không dấu, nên cũng cần
            cẩn thận như vậy ở đó.
          </p>
        }
      >
        <p>
          Mixing <code>unsigned int</code> with <code>int</code> converts the <code>int</code> to unsigned, so 0 − 2 wraps to
          2^32 − 2 = 4294967294, and <code>&gt;= 0</code> holds for every unsigned value. The other half,{' '}
          <code>&lt; TILE_DIM</code>, is false for the wrapped value, so the guard still rejects the tap: it works by
          accident. Copy the index to an <code>int</code> first, as the book already does for <code>row</code> and{' '}
          <code>col</code>, and as lab 7.2 does. Metal's thread-position attributes are unsigned too, so the same care applies
          there.
        </p>
      </Translated>
    ),
    source: 'facts ch07 A8 (compiled as host C++ with clang; not with nvcc); ch03 A2 and C6',
    vi: { source: 'facts ch07 A8 (biên dịch dưới dạng C++ host bằng clang; không phải nvcc); ch03 A2 và C6' },
  },
  'ch07/7.6/q-warps': {
    kind: 'numeric',
    id: 'ch07/7.6/q-warps',
    prompt: (
      <Translated
        vi={
          <>
            Hình 7.15 với <code>TILE_DIM = 32</code>, bán kính 2, và warp 32 luồng (mỗi warp một hàng của khối). Trong bao
            nhiêu warp trên 32 warp của khối, các lane bất đồng tại phép kiểm tra trong-ô ở ít nhất một tap?
          </>
        }
      >
        <>
          Fig. 7.15 with <code>TILE_DIM = 32</code>, radius 2, and warps of 32 threads (one row of the block each). In how
          many of the block's 32 warps do lanes disagree at the in-tile test on at least one tap?
        </>
      </Translated>
    ),
    answer: warpSplits(32, 2, 32).diverging,
    unit: 'warps',
    explain: (
      <Translated
        vi={
          <p>
            Tất cả. Mỗi hàng đều có lane 0 và 1 gần cạnh trái và lane 30 và 31 gần cạnh phải: ở các tap có{' '}
            <code>fCol ≠ 2</code>, cột của chúng ra khỏi ô trong khi cột của các lane ở giữa vẫn ở bên trong. Hai hàng đầu và
            hai hàng cuối còn có các tap mà hàng ra khỏi ô, nhưng những tap đó đưa cả warp đi cùng một hướng. Mỗi warp bị tách
            ở 12 đến 20 trong 25 tap của nó. Ngược lại, trong Hình 7.12, phép kiểm tra khiến vòng ngoài ngồi không nằm ngoài
            vòng lặp bộ lọc, nên một warp chỉ bị tách ở đó một lần.
          </p>
        }
      >
        <p>
          All of them. Every row has lanes 0 and 1 near the left edge and 30 and 31 near the right edge: on taps with{' '}
          <code>fCol ≠ 2</code> their columns leave the tile while the middle lanes' columns stay inside. The first and last
          two rows also have taps whose row leaves the tile, but those send the whole warp the same way. Each warp is split
          on 12 to 20 of its 25 taps. In Fig. 7.12, by contrast, the test that idles the outer ring sits outside the filter
          loop, so a warp splits there once.
        </p>
      </Translated>
    ),
    source: 'our analysis with sim/convcached.ts; the claim it tests is on book p. 179',
    vi: { unit: 'warp', source: 'phân tích của khóa học này với sim/convcached.ts; nhận định mà nó kiểm tra nằm ở tr. 179 của sách' },
  },
  'ch07/7.6/q-miss': {
    kind: 'numeric',
    id: 'ch07/7.6/q-miss',
    prompt: (
      <Translated
        vi={
          <>
            Một khối bên trong, ô 32 × 32, bộ lọc 5 × 5. Giả sử không có bộ nhớ đệm nào giúp: mỗi tap toàn cục là một lần
            đọc DRAM riêng, kể cả khi một luồng khác của khối vừa đọc cùng phần tử đó. Khối thực hiện bao nhiêu lần đọc{' '}
            <code>N</code> từ DRAM cho mỗi đầu ra, tính cả các lần nạp ô?
          </>
        }
      >
        <>
          An interior block, 32 × 32 tile, 5 × 5 filter. Suppose no cache helps at all: every global tap is its own read of
          DRAM, even when another thread of the block has just read the same element. How many reads of <code>N</code> from
          DRAM per output does the block make, counting the tile loads?
        </>
      </Translated>
    ),
    answer: Math.round(B32.allTaps * 100) / 100,
    tol: 0.02,
    unit: 'reads per output',
    explain: (
      <Translated
        vi={
          <>
            <p>
              (1024 lần nạp + {n(H32.global)} tap vùng biên) / 1024 = {B32.allTaps.toFixed(2)}, cho phép đạt{' '}
              {intensity(B32.allTaps, 2).toFixed(2)} FLOP/B. Hình 7.12 với ô đầu vào rộng 32 cần 1024 / 784 ={' '}
              {B32.fig712.toFixed(2)} lần đọc (5.42 FLOP/B), và mức lý tưởng là 1 (6.25 FLOP/B).
            </p>
            <p>
              Giữa hai cực: nếu khối nạp mỗi phần tử trong {H32.distinctGlobal} phần tử vùng biên phân biệt của nó một lần,
              nó sẽ cần {B32.perBlock.toFixed(2)} lần đọc mỗi đầu ra, đúng bằng công thức của mục 7.5 tại t = 32 (5.52 FLOP/B
              trong Hình 7.14). Để vùng biên cho bộ nhớ đệm lo chỉ có lợi nếu có bộ nhớ đệm nào đó làm được việc.
            </p>
          </>
        }
      >
        <>
          <p>
            (1024 loads + {n(H32.global)} halo taps) / 1024 = {B32.allTaps.toFixed(2)}, which allows{' '}
            {intensity(B32.allTaps, 2).toFixed(2)} FLOP/B. Fig. 7.12 with a 32-wide input tile needs 1024 / 784 ={' '}
            {B32.fig712.toFixed(2)} reads (5.42 FLOP/B), and the ideal is 1 (6.25 FLOP/B).
          </p>
          <p>
            Between the two extremes: if the block fetched each of its {H32.distinctGlobal} distinct halo elements once, it
            would need {B32.perBlock.toFixed(2)} reads per output, which is the section 7.5 formula at t = 32 (5.52 FLOP/B in
            Fig. 7.14). Leaving the halo to the cache pays only if some cache delivers.
          </p>
        </>
      </Translated>
    ),
    source: 'our analysis with sim/convcached.ts; book pp. 167, 176 to 178',
    vi: { unit: 'lần đọc mỗi đầu ra', source: 'phân tích của khóa học này với sim/convcached.ts; sách tr. 167, 176 đến 178' },
  },
  'ch07/7.6/q-xcd': {
    kind: 'choice',
    id: 'ch07/7.6/q-xcd',
    prompt: (
      <Translated
        vi={
          <>
            Một GPU có L2 chia theo từng die, và việc đặt khối lên các compute unit không tất định. Tiền đề của mục 7.6 rằng
            vùng biên của một khối có lẽ đang nằm trong L2 sẽ ra sao?
          </>
        }
      >
        <>
          A GPU whose L2 is split per die, and whose placement of blocks on compute units is not deterministic. What becomes
          of section 7.6's premise that a block's halo is probably in L2?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không có gì thay đổi: L2 của mỗi die giữ một bản sao của mọi line, nên trúng ở một L2 là trúng ở tất cả.">
            {'Nothing changes: every die\'s L2 keeps a copy of every line, so a hit in one is a hit in all.'}
          </Translated>
        ),
        why: (
          <Translated vi="Không có gì trong tài liệu của AMD nói vậy: L2 của mỗi die phục vụ các compute unit của die đó. Một lần trúng chỉ tính trong L2 mà khối này đọc từ đó.">
            {'Nothing in AMD\'s documents says so: each die\'s L2 serves the compute units of that die. A hit counts only in the L2 this block reads from.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các khối lân cận có thể chạy trên các die khác nhau, nên một line vùng biên có thể nằm trong L2 của die khác: hãy chờ đợi nhiều lần trượt hơn.">
            {'Neighbours may run on different dies, so a halo line can sit in another die\'s L2: expect more misses.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Kernel tính ra kết quả sai, vì một line vùng biên được nạp trên die khác có thể đã cũ.">
            The kernel computes wrong results, because a halo line loaded on another die may be stale.
          </Translated>
        ),
        why: (
          <Translated vi="N chỉ được đọc, không bao giờ được ghi, nên không bản sao nào có thể cũ: mọi lần đọc đều trả về đúng giá trị. Điều thay đổi là nó đến từ đâu và tốn bao nhiêu.">
            N is only read, never written, so no copy can be stale: every read returns the right value. What changes is
            where it comes from and what it costs.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các lần đọc vùng biên được chuyển tiếp tới L2 của die kia và được phục vụ ở đó nhanh như một lần trúng cục bộ.">
            {'The halo reads are forwarded to the other die\'s L2 and served there as fast as a local hit.'}
          </Translated>
        ),
        why: (
          <Translated vi="Không có gì trong tệp facts nói vậy. AMD mô tả L2 là được dùng chung bởi các compute unit của một die.">
            Nothing in the facts file says so. AMD describes the L2 as shared by the compute units of one die.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trên MI300X của AMD, mỗi die (XCD) có L2 riêng, và tầng duy nhất được cả GPU dùng chung là cache cấp cuối. AMD nói
            compute unit nào chạy một workgroup là không dự đoán được, và có thể thay đổi từ lần khởi chạy này sang lần khác.
            Vì vậy các khối lân cận có thể chỉ dùng chung tầng cuối đó, nếu có: hãy chờ đợi nhiều lần trượt hơn mức sách giả
            định, và hãy đo. Trong trình khám phá ở trên, đặt L2 thành 8 lát và cách đặt thành ngẫu nhiên.
          </p>
        }
      >
        <p>
          On AMD's MI300X each die (XCD) has its own L2, and the only level shared by the whole GPU is the last-level
          cache. AMD says which compute unit runs a workgroup is not predictable, and can change from one launch to the next.
          So neighbouring blocks may share only that last level, if anything: expect more misses than the book assumes, and
          measure. In the explorer above, set the L2 to 8 slices and placement to random.
        </p>
      </Translated>
    ),
    source: 'facts ch07 B5', // i18n-ignore: citation only
  },
};

// ---------------------------------------------------------------- the section

export function S76() {
  const { reports } = useReports();
  const cu = useLabFile(SOL + '/kernel.cu');
  const metal = useLabFile(SOL + '/kernel.metal');
  const [showKernel, setShowKernel] = useState(false);
  // The reader's own lab 7.2 passing anywhere means part B is repaired: the listing is no longer a spoiler.
  const repaired = Object.keys(reports).some((b) => labStatus(reports[b], LAB72.id, 'work') === 'pass');
  // The labs page's words (LabKit's solutionTabStatus), so a failed run reads as failed here too.
  const st = (f: 'kernel.cu' | 'kernel.metal') => solutionTabStatus(reports, LAB72, f);
  const warpChoices = useMemo(() => {
    const base = [
      {
        get label() {
          return tr('32 (NVIDIA)', '32 (của NVIDIA)');
        },
        value: 32,
      },
      {
        get label() {
          return tr('64 (some AMD GPUs)', '64 (một số GPU AMD)');
        },
        value: 64,
      },
    ];
    for (const b of gpuBackends(reports)) {
      const w = Number(reports[b].device?.warpSize);
      if (w > 0 && !base.some((x) => x.value === w))
        base.push({
          get label() {
            return tr(`${w} (your ${b} device)`, `${w} (thiết bị ${b} của bạn)`);
          },
          value: w,
        });
    }
    return base;
  }, [reports]);
  const fig = H32;

  return (
    <>
      <Translated
        vi={
          <p>
            Mục 7.5 đã nạp toàn bộ ô đầu vào của mỗi khối, cả vùng biên (halo), vào bộ nhớ chia sẻ (shared memory), và đã
            trả giá cho điều đó: một khối lớn hơn ô đầu ra của nó, những luồng (thread) chỉ để nạp, và một ô đầu ra rộng 28
            không phải lũy thừa của hai. Mục 7.6 hỏi điều gì xảy ra nếu bạn hoàn toàn không nạp trước vùng biên.
          </p>
        }
      >
        <p>
          Section 7.5 loaded each block's whole input tile, halo included, into shared memory, and paid for it: a block
          larger than its output tile, threads that only load, and an output tile of 28 that is not a power of two. Section
          7.6 asks what happens if you do not stage the halo at all.
        </p>
      </Translated>
      <Source section="7.5" pages="177" />
      <Translated
        vi={
          <p>
            Lập luận rất ngắn. Các ô vùng biên của một khối là các ô bên trong của các khối lân cận. Một khối lân cận chạy
            gần đây đã nạp chúng từ DRAM, nên theo lời sách, có &quot;một xác suất đáng kể&quot; rằng chúng vẫn còn trong bộ
            nhớ đệm (cache) L2 khi khối này đọc chúng. Nếu vậy, một lần đọc vùng biên thẳng từ bộ nhớ toàn cục (global
            memory) không tốn lưu lượng DRAM, và bộ nhớ chia sẻ chỉ cần chứa bản thân ô.
          </p>
        }
      >
        <p>
          The argument is short. A block's halo cells are interior cells of its neighbours. A neighbour that ran recently
          loaded them from DRAM, so there is, in the book's words, "a significant probability" that they are still in the
          L2 cache when this block reads them. If so, a halo read straight from global memory costs no DRAM traffic, and
          shared memory need only hold the tile itself.
        </p>
      </Translated>
      <Source section="7.6" pages="178" />

      <h2>
        <Translated vi="Điều gì thay đổi">What changes</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Ba điều theo sau. Khối, ô chia sẻ và ô đầu ra đều có thể có cạnh <code>TILE_DIM</code>, và <code>TILE_DIM</code>{' '}
            có thể là một lũy thừa của hai như 32. Việc nạp trở nên đơn giản: mỗi luồng nạp phần tử tại vị trí đầu ra của chính
            nó, và kiểm tra duy nhất còn lại là liệu một ô ở cạnh phải hoặc cạnh dưới có chìa ra ngoài ảnh hay không. Việc tính
            trở nên khó hơn, vì mỗi tap giờ có ba trường hợp thay vì một:
          </p>
        }
      >
        <p>
          Three things follow. The block, the shared tile and the output tile can all be <code>TILE_DIM</code> on a side,
          and <code>TILE_DIM</code> can be a power of two such as 32. Loading becomes simple: each thread loads the element
          at its own output position, and the only check left is whether a tile at the right or bottom edge overhangs the
          image. Computing becomes harder, because each tap now has three cases instead of one:
        </p>
      </Translated>
      <Source section="7.6" pages="178–179" />
      <Translated
        vi={
          <ol>
            <li>Tap nằm trong ô của khối này: đọc nó từ bộ nhớ chia sẻ.</li>
            <li>Nó nằm ngoài ô nhưng trong ảnh: đó là một ô vùng biên; đọc nó từ bộ nhớ toàn cục và hy vọng bộ nhớ đệm có nó.</li>
            <li>Nó nằm ngoài ảnh: đó là một ô ma; bỏ qua nó, tương đương với cộng thêm số không.</li>
          </ol>
        }
      >
        <ol>
          <li>The tap lies inside this block's tile: read it from shared memory.</li>
          <li>It lies outside the tile but inside the image: it is a halo cell; read it from global memory and hope a cache has it.</li>
          <li>It lies outside the image: it is a ghost cell; skip it, which is the same as adding zero.</li>
        </ol>
      </Translated>

      <h2>
        <Translated vi="Kernel">The kernel</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lab 7.2 cài đặt nó, và phần B của lab đưa bạn một bản có hai lỗi cài sẵn để sửa, nên đoạn mã tham chiếu chờ cho
            đến khi bạn yêu cầu. Khi mở nó, hãy đọc theo màu: <code>N</code> màu hổ phách bên trong vòng lặp bộ lọc chính là
            lần đọc vùng biên mà cả mục này nói về.
          </p>
        }
      >
        <p>
          Lab 7.2 implements it, and its part B hands you a copy with two planted defects to repair, so the reference
          listing waits until you ask for it. When you open it, read it by colour: the amber <code>N</code> inside the
          filter loop is the halo read the whole section is about.
        </p>
      </Translated>
      {!showKernel && !repaired ? (
        <p className="ui" data-listing="gated">
          <Translated
            vi={
              <>
                Kernel đã sửa là đáp án cho phần B của lab 7.2 trên <a href={href('ch07', 'labs')}>trang lab</a>. Nó tự hiện
                ra khi lab 7.2 của chính bạn vượt qua trên máy này.{' '}
              </>
            }
          >
            <>
              The repaired kernel is the answer to part B of lab 7.2 on the <a href={href('ch07', 'labs')}>labs page</a>. It
              appears by itself once your own lab 7.2 passes on this machine.{' '}
            </>
          </Translated>
          <button className="quiet" onClick={() => setShowKernel(true)}>
            <Translated vi="Hiện kernel đã sửa ngay">Show the repaired kernel now</Translated>
          </button>
        </p>
      ) : cu && metal ? (
        failed(cu) ? (
          <p className="muted" data-listing="missing">
            <Translated
              vi={
                <>
                  Đoạn mã được tải từ <code>labs/ch07/lab2_tiledconv/solution/</code>, thư mục mà bản sao kho mã này không
                  có.
                </>
              }
            >
              The listing loads from <code>labs/ch07/lab2_tiledconv/solution/</code>, which this copy of the repository
              does not have.
            </Translated>
          </p>
        ) : (
          <Code
            tabs={[
              { label: 'CUDA', code: part(cu, 'cached')!, tiers: CONV_TIERS, status: st('kernel.cu') },
              ...(failed(metal) ? [] : [{ label: tr('Metal kernel', 'Kernel Metal'), code: part(metal, 'cached')!, tiers: CONV_TIERS, status: st('kernel.metal') }]),
            ]}
            caption={
              <Translated
                vi={
                  <>
                    Kernel vùng biên qua bộ nhớ đệm trong lời giải tham chiếu của lab 7.2 (phần được đánh dấu{' '}
                    <code>// @part cached</code> trong <code>labs/ch07/lab2_tiledconv/solution/</code>). Phần B của lab đưa
                    bạn một bản hỏng để sửa. Bản build HIP được thiết lập để nhận cùng tệp; nó chưa được biên dịch hay chạy
                    trên phần cứng AMD (unverified).
                  </>
                }
              >
                <>
                  The cached-halo kernel of lab 7.2's reference solution (the part marked <code>// @part cached</code> in{' '}
                  <code>labs/ch07/lab2_tiledconv/solution/</code>). Part B of the lab gives you a broken copy to repair. The HIP
                  build is set up to take the same file; it has not been compiled or run on AMD hardware (unverified).
                </>
              </Translated>
            }
          />
        )
      ) : null}

      <h3>
        <Translated vi="Đoạn mã được in sai ở đâu">What the printed listing gets wrong</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Hình 7.15 như được in có một lỗi thật và một lỗi tự triệt tiêu. Lỗi thật nằm ở dòng 21: lần đọc bộ nhớ chia sẻ là{' '}
            <code>N_s[threadIdx.y+fRow][threadIdx.x+fCol]</code>, trong khi điều kiện chặn ngay phía trên kiểm tra{' '}
            <code>threadIdx.x-FILTER_RADIUS+fCol</code>. Chỉ số thiếu <code>-FILTER_RADIUS</code> mà chính điều kiện chặn
            của nó dùng.
          </p>
        }
      >
        <p>
          Fig. 7.15 as printed has one real error and one that cancels out. The real one is line 21: the shared-memory
          read is <code>N_s[threadIdx.y+fRow][threadIdx.x+fCol]</code>, while the guard just above it tests{' '}
          <code>threadIdx.x-FILTER_RADIUS+fCol</code>. The index lacks the <code>-FILTER_RADIUS</code> its own guard uses.
        </p>
      </Translated>
      <Source section="7.6" pages="178" figs={tr('Fig. 7.15', 'Hình 7.15')} />
      <Ask q={questionsS76['ch07/7.6/q-printed']} />
      <Translated
        vi={
          <p>
            Lỗi thứ hai nằm trong điều kiện chặn đó. <code>threadIdx.x</code> là số không dấu, nên cả biểu thức là không dấu
            và nửa <code>&gt;= 0</code> của nó luôn đúng. Nửa <code>&lt; TILE_DIM</code> tình cờ loại bỏ các giá trị đã quay
            vòng, nên điều kiện chặn vẫn làm đúng điều được dự định, nhờ may mắn. Cách sửa là cách sách đã áp dụng cho{' '}
            <code>row</code> và <code>col</code>: sao chép chỉ số vào một <code>int</code> trước khi trừ.
          </p>
        }
      >
        <p>
          The second is in that guard. <code>threadIdx.x</code> is unsigned, so the whole expression is unsigned and its{' '}
          <code>&gt;= 0</code> half is always true. The <code>&lt; TILE_DIM</code> half happens to reject the wrapped values,
          so the guard still does what was meant, by accident. The fix is the one the book already applies to{' '}
          <code>row</code> and <code>col</code>: copy the index into an <code>int</code> before subtracting.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Được kiểm tra dưới dạng C++ host với Apple clang (facts ch07 A8): biểu thức có kiểu <code>unsigned</code>, và{' '}
              <code>-Wtautological-unsigned-zero-compare</code> báo rằng phép so sánh luôn đúng. Chưa biên dịch bằng nvcc.
            </p>
          }
        >
          <p>
            Checked as host C++ with Apple clang (facts ch07 A8): the expression has type <code>unsigned</code>, and{' '}
            <code>-Wtautological-unsigned-zero-compare</code> reports that the comparison is always true. Not compiled with
            nvcc.
          </p>
        </Translated>
      </Aside>
      <Ask q={questionsS76['ch07/7.6/q-unsigned']} />
      <Translated
        vi={
          <p>
            Ba sơ suất nhỏ hơn thì dễ sửa: số dòng được in bị hỏng (24 đến 27 xuất hiện hai lần), bộ lọc được khai báo là{' '}
            <code>F_c</code> nhưng dùng là <code>F</code>, và <code>N_s</code> được khai báo không có kiểu, nên đoạn mã không
            biên dịch được như được in. Lời giải tham chiếu của lab 7.2 không có vấn đề nào trong số này.
          </p>
        }
      >
        <p>
          Three smaller slips are easy to fix: the printed line numbers are corrupted (24 to 27 appear twice), the filter is
          declared <code>F_c</code> and used as <code>F</code>, and <code>N_s</code> is declared without a type, so the
          listing does not compile as printed. Lab 7.2&rsquo;s reference solution has none of these problems.
        </p>
      </Translated>

      <h2>
        <Translated vi="Mỗi tap đến từ đâu">Where each tap comes from</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trước khi mở trình khám phá, hãy lấy một khối bên trong với ô 32 × 32 và bộ lọc 5 × 5. 1,024 luồng của nó mỗi
            luồng thực hiện 25 tap. Các tap gần giữa ô nằm trong bộ nhớ chia sẻ; các tap của luồng gần cạnh ô với ra vòng bao
            quanh nó.
          </p>
        }
      >
        <p>
          Before you open the explorer, take an interior block with a 32 × 32 tile and a 5 × 5 filter. Its 1,024 threads make
          25 taps each. Taps near the middle of the tile stay in shared memory; taps of threads near the tile's edge reach
          into the ring around it.
        </p>
      </Translated>
      <Predict
        id="ch07/7.6/p-global-fraction"
        prompt={
          <Translated vi={<>Bao nhiêu phần trăm trong {n(fig.taps)} tap của khối rời bộ nhớ chia sẻ và đọc bộ nhớ toàn cục?</>}>
            <>What percentage of the block's {n(fig.taps)} taps leave shared memory and read global memory?</>
          </Translated>
        }
        unit={tr('% of taps', '% số tap')}
      >
        {(v) => {
          const truth = (100 * fig.global) / fig.taps;
          return (
            <Translated
              vi={
                <>
                  <p data-reveal={truth.toFixed(1)}>
                    Mô hình đếm được <b>{n(fig.global)}</b> trên {n(fig.taps)}: <b>{truth.toFixed(1)}%</b>.{' '}
                    {Math.abs(v - truth) <= 0.05 * truth ? 'Trong vòng 5% so với dự đoán của bạn.' : v > truth ? 'Ít hơn bạn nghĩ.' : 'Nhiều hơn bạn nghĩ.'}
                  </p>
                  <p>
                    Hãy đếm từng trục một. Theo x, luồng 0 giữ 3 trong 5 tap cột của nó bên trong ô, luồng 1 giữ 4, các luồng 2
                    đến 29 giữ cả 5, và các luồng 30 và 31 đối xứng với hai luồng đầu: tổng cộng 154. Theo y cũng vậy, và một tap
                    chỉ nằm bên trong khi cả hai tọa độ đều nằm bên trong, nên 154 × 154 = 23,716 tap đọc bộ nhớ chia sẻ và
                    1,884 tap còn lại thì không. Phần lớn luồng không bao giờ rời ô; 1,884 tap đến từ một khung rộng hai luồng.
                  </p>
                </>
              }
            >
              <>
                <p data-reveal={truth.toFixed(1)}>
                  The model counts <b>{n(fig.global)}</b> of {n(fig.taps)}: <b>{truth.toFixed(1)}%</b>.{' '}
                  {Math.abs(v - truth) <= 0.05 * truth ? 'Within 5% of your prediction.' : v > truth ? 'Fewer than you expected.' : 'More than you expected.'}
                </p>
                <p>
                  Count one axis at a time. Along x, thread 0 keeps 3 of its 5 column taps inside the tile, thread 1 keeps 4,
                  threads 2 to 29 keep all 5, and threads 30 and 31 mirror the first two: 154 in all. The same holds along y,
                  and a tap stays inside only if both coordinates do, so 154 × 154 = 23,716 taps read shared memory and the
                  other 1,884 do not. Most threads never leave the tile; the 1,884 come from a frame two threads wide.
                </p>
              </>
            </Translated>
          );
        }}
      </Predict>
      <Translated
        vi={
          <p>
            Trình khám phá chạy bộ phân loại trên bất kỳ khối nào của bất kỳ ảnh nào. Chọn một khối và nhấn vào một luồng: các
            tap của nó sáng lên theo màu của bộ nhớ mà chúng đọc, và các bảng bên dưới đếm cùng thứ đó cho toàn khối và cho
            từng warp. Công tắc ở cuối hàng thứ hai thay dòng 21 bằng phiên bản như được in.
          </p>
        }
      >
        <p>
          The explorer runs the classifier on any block of any image. Pick a block and click a thread: its taps light up in
          the colour of the memory they read, and the panels below count the same thing for the whole block and for each
          warp. The switch at the end of the second row swaps in line 21 as printed.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Nó đếm tap và tính chỉ số, không có thời gian và không có bộ nhớ đệm. Một tap &quot;toàn cục&quot; là một lần
              đọc rời khỏi SM, bất kể thứ gì trả lời nó. Warp là các dãy id luồng liên tiếp, x chạy nhanh nhất.
            </p>
          }
        >
          <p>
            It counts taps and evaluates indices, with no timing and no caches. A "global" tap is a read that leaves the SM,
            whatever answers it. Warps are runs of consecutive thread ids, x fastest.
          </p>
        </Translated>
      </Simplification>
      <TapExplorer warpChoices={warpChoices} />
      <Translated vi={<p>Những điều nên thử:</p>}>
        <p>Things to try:</p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>Luồng (0, 0), rồi luồng (2, 2): luồng đầu gửi 16 tap của nó tới bộ nhớ toàn cục, luồng sau không gửi tap nào.</li>
            <li>Khối ở góc: các tap vùng biên rơi ra ngoài ảnh trở thành ô ma, và kernel bỏ qua chúng.</li>
            <li>Khối cuối cùng với ảnh mặc định có kích thước lẻ: ô của nó chìa ra ngoài ảnh, nên một số tap chia sẻ đọc các số không mà pha nạp đã ghi.</li>
            <li>Ô 4 với bán kính 3: vòng rộng hơn ô, và phần lớn tap rời khỏi bộ nhớ chia sẻ.</li>
            <li>Dòng 21 như được in: mọi tap chia sẻ dịch xuống dưới và sang phải một khoảng bằng bán kính, và một số rơi ra ngoài ô.</li>
          </ul>
        }
      >
        <ul>
          <li>Thread (0, 0), then thread (2, 2): the first sends 16 of its taps to global memory, the second none.</li>
          <li>The corner block: halo taps that fall outside the image become ghosts, and the kernel skips them.</li>
          <li>The last block with the default ragged image: its tile overhangs the image, so some shared taps read the zeros the load phase stored.</li>
          <li>Tile 4 with radius 3: the ring is wider than the tile, and most taps leave shared memory.</li>
          <li>Line 21 as printed: every shared tap moves down and right by the radius, and some land past the tile.</li>
        </ul>
      </Translated>

      <h2>
        <Translated vi="Ưu điểm tinh tế, và cái giá của nó">The subtle advantage, and its price</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách chỉ ra một &quot;ưu điểm tinh tế&quot;: khối, ô đầu vào và ô đầu ra giờ có cùng kích thước, có thể là một
            lũy thừa của hai. Sách nói thêm rằng kernel của Hình 7.12, với các kích thước không bằng nhau, &quot;có lẽ&quot;
            có nhiều phân kỳ bộ nhớ và phân kỳ điều khiển hơn. Điểm kém hiệu quả thứ nhất của mục 7.5 cũng biến mất: ở đây
            không luồng nào chỉ để nạp.
          </p>
        }
      >
        <p>
          The book points out a "subtle advantage": the block, the input tile and the output tile now have the same size,
          which can be a power of two. It adds that the Fig. 7.12 kernel, with unequal sizes, "likely" has more memory
          divergence and control divergence. Section 7.5's first inefficiency is gone too: no thread here only loads.
        </p>
      </Translated>
      <Source section="7.6" pages="179" />
      <Translated
        vi={
          <p>
            Giờ hãy nhìn bảng warp của trình khám phá. Phép kiểm tra trong-ô nằm bên trong vòng lặp bộ lọc và phụ thuộc vào
            vị trí của luồng, và mọi hàng của ô đều có các lane gần cạnh trái và cạnh phải. Vì vậy với ô 32 và warp 32, mọi
            warp đều bị tách tại phép kiểm tra đó ở phần lớn tap của nó. Hình 7.12 trả giá phân kỳ một lần mỗi warp, bên ngoài
            vòng lặp, và để một số lane ngồi không; Hình 7.15 giữ mọi lane bận rộn và bị tách bên trong vòng lặp.
          </p>
        }
      >
        <p>
          Now look at the warp panel of the explorer. The in-tile test sits inside the filter loop and depends on the
          thread's position, and every row of the tile has lanes near its left and right edges. So with tile 32 and warps of
          32, every warp is split at that test on most of its taps. Fig. 7.12 pays its divergence once per warp, outside the
          loop, and leaves some lanes idle; Fig. 7.15 keeps every lane busy and splits inside the loop.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Các con số theo từng warp là của khóa học này, từ mô hình. Sách không so sánh phân kỳ của hai kernel ngoài chữ
              &quot;có lẽ&quot;. Cái nào tốn hơn tùy thuộc vào thiết bị; lab 7.2 đo thời gian cả hai.
            </p>
          }
        >
          <p>
            The per-warp counts are ours, from the model. The book does not compare the two kernels' divergence beyond
            "likely". Which one costs more depends on the device; lab 7.2 times both.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS76['ch07/7.6/q-warps']} />

      <h2>
        <Translated vi="Bao nhiêu lưu lượng DRAM?">How much DRAM traffic?</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách không đưa ra cường độ tính toán cho kernel này, và không thể: nó phụ thuộc vào bộ nhớ đệm. Thứ đếm được là
            các cận. Lấy lại khối 32 × 32 bên trong với bán kính 2 và đếm số lần đọc <code>N</code> từ DRAM mỗi đầu ra, với một
            lệnh ghi 4 byte mỗi đầu ra và 2m² FLOP mỗi đầu ra như ở mục 7.5.
          </p>
        }
      >
        <p>
          The book gives no arithmetic intensity for this kernel, and cannot: it depends on the cache. What can be counted
          are bounds. Take the interior 32 × 32 block at radius 2 again and count reads of <code>N</code> from DRAM per
          output, with one 4-byte store per output and 2m² FLOP per output as in section 7.5.
        </p>
      </Translated>
      <Source section="7.5" pages="176–177" figs={tr('Fig. 7.14', 'Hình 7.14')} />
      <table className="data wide" style={{ maxWidth: '50rem' }} data-bounds={[B32.ideal, B32.perBlock, B32.allTaps, B32.fig712].map((x) => x.toFixed(4)).join(',')}>
        <thead>
          <tr>
            <th><Translated vi="Bộ nhớ đệm bắt được gì">What a cache catches</Translated></th>
            <th className="num"><Translated vi="Số lần đọc N mỗi đầu ra">Reads of N per output</Translated></th>
            <th className="num"><Translated vi="FLOP/B, m = 5">FLOP/B, m = 5</Translated></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="Mọi lần đọc vùng biên đều trúng L2 (hy vọng của sách)">Every halo read hits in L2 (the book's hope)</Translated></td>
            <td className="num">{B32.ideal.toFixed(2)}</td>
            <td className="num"><Translated vi={<>{intensity(B32.ideal, 2).toFixed(2)}, mức lý tưởng m²/4</>}>{intensity(B32.ideal, 2).toFixed(2)}, the ideal m²/4</Translated></td>
          </tr>
          <tr>
            <td>
              <Translated vi={<>Mỗi khối nạp mỗi phần tử trong {H32.distinctGlobal} phần tử vùng biên của nó một lần; không có gì được chia sẻ giữa các khối</>}>
                Each block fetches each of its {H32.distinctGlobal} halo elements once; nothing is shared between blocks
              </Translated>
            </td>
            <td className="num">{B32.perBlock.toFixed(2)}</td>
            <td className="num">{intensity(B32.perBlock, 2).toFixed(2)}</td>
          </tr>
          <tr>
            <td>
              <Translated vi={<>Không gì cả: mỗi tap trong {n(H32.global)} tap toàn cục là một lần đọc DRAM riêng</>}>
                Nothing: each of the {n(H32.global)} global taps is its own DRAM read
              </Translated>
            </td>
            <td className="num">{B32.allTaps.toFixed(2)}</td>
            <td className="num">{intensity(B32.allTaps, 2).toFixed(2)}</td>
          </tr>
          <tr>
            <td>
              <Translated vi="Để so sánh: Hình 7.12 với ô đầu vào rộng 32, theo cách đếm của sách">
                For comparison: Fig. 7.12 with a 32-wide input tile, by the book's count
              </Translated>
            </td>
            <td className="num">{B32.fig712.toFixed(2)}</td>
            <td className="num">{intensity(B32.fig712, 2).toFixed(2)}</td>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Hàng giữa đáng nhìn lại lần nữa. Đó là công thức của mục 7.5 tại t = 32, con số 5.52 của Hình 7.14: khi mỗi khối
            nạp vùng phủ của chính nó một lần, kernel này di chuyển lượng dữ liệu mà Hình 7.12 sẽ di chuyển với ô đầu ra rộng
            32, điều mà Hình 7.12 không thể có vì khối của nó là ô đầu vào. Đó là mặt lưu lượng của &quot;ưu điểm tinh
            tế&quot;. Hàng cuối là cái giá nếu không có gì bắt được ngay cả các lần đọc lặp lại của chính khối.
          </p>
        }
      >
        <p>
          The middle row deserves a second look. It is the section 7.5 formula at t = 32, the 5.52 of Fig. 7.14: when each
          block fetches its own footprint once, this kernel moves the data Fig. 7.12 would move with a 32-wide output tile,
          which Fig. 7.12 cannot have because its block is the input tile. That is the traffic side of the "subtle
          advantage". The bottom row is the price if nothing catches even the block's own repeated reads.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách không đưa ra con số nào cho Hình 7.15. Các cận này và phép so sánh với Hình 7.14 là của khóa học này, được
              tính bởi <code>sim/convcached.ts</code>.
            </p>
          }
        >
          <p>
            The book gives no number for Fig. 7.15. These bounds and the comparison with Fig. 7.14 are ours, computed by{' '}
            <code>sim/convcached.ts</code>.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS76['ch07/7.6/q-miss']} label={tr('Work it out', 'Tự tính')} />

      <h3>
        <Translated vi="Vùng biên có nằm trong bộ nhớ đệm không?">Will the halo be in the cache?</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Một lần đọc vùng biên có trúng hay không phụ thuộc vào thời gian: những khối nào đã chạy trước khối này, từ đó
            chúng đã chạm vào bao nhiêu line khác, và chúng có dùng cùng bộ nhớ đệm hay không. Điều đó cần một mô hình bộ nhớ
            đệm, và sách không đưa ra mô hình nào. Hoạt ảnh và trình khám phá bên dưới chạy mô hình của khóa học này. Hoạt ảnh
            theo dõi hai khối lân cận, rồi một khối thứ ba, qua một bộ nhớ đệm cố ý nhỏ, mỗi lần một khối. Trình khám phá chạy
            toàn bộ một ảnh và để bạn điều khiển kích thước bộ nhớ đệm, số khối chạy cùng lúc và cách chia L2.
          </p>
        }
      >
        <p>
          Whether a halo read hits depends on time: which blocks ran before this one, how many other lines they touched
          since, and whether they used the same cache. That takes a model of a cache, and the book offers none. The
          animation and the explorer below run ours. The animation follows two neighbouring blocks, then a third, through a
          deliberately small cache, one block at a time. The explorer runs a whole image and puts the cache size, the number
          of blocks running at once and the split of the L2 under your control.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              <b>Một mô hình đồ chơi, và được ghi rõ như vậy.</b> Các khối chạy theo đợt, theo thứ tự chỉ số, nạp ô của chúng
              rồi đọc vùng biên; một khối lấy mỗi line nhiều nhất một lần. L2 là một hoặc nhiều bộ nhớ đệm LRU kết hợp toàn
              phần. Không được mô hình hóa: cơ chế thay thế và băm thực tế, sector, bộ lập lịch thực tế, thời gian, lưu lượng
              khác, bất kỳ kích thước cache thực nào.
            </p>
          }
        >
          <p>
            <b>A toy, labelled as such.</b> Blocks run in waves, in index order, loading their tile and then reading their
            halo; a block fetches each line at most once. The L2 is one or more fully associative LRU caches. Not modelled:
            real replacement and hashing, sectors, the real scheduler, timing, other traffic, any real cache size.
          </p>
        </Translated>
      </Simplification>
      <HaloFromCacheScene />
      <CacheExplorer />
      <Translated
        vi={
          <p>
            Ba điều trong trình khám phá rất dễ bỏ lỡ. Thứ nhất, một lần đọc vùng biên bị trượt không phải lúc nào cũng là lưu
            lượng thêm: nếu nó mang vào một line mà khối sở hữu nạp ngay sau đó, khối sở hữu sẽ trúng thay vào đó. Điều quan
            trọng là mỗi line đến từ DRAM bao nhiêu lần, và đó là thứ các thanh đo. Thứ hai, trình khám phá đếm cả cache line,
            nên thanh không-tái-sử-dụng của nó cho kernel này cao hơn hàng giữa của bảng: một vùng biên rộng hai cột vẫn tốn
            nguyên một line mỗi hàng. Đặt kích thước line là 32 byte và hai thanh không-tái-sử-dụng sẽ xấp xỉ bằng nhau. Thứ
            ba, cùng bộ nhớ đệm đó phục vụ Hình 7.12: các ô đầu vào lân cận của nó chồng lên nhau 2r, và trong mô hình nó đạt
            mức lý tưởng ở dung lượng nhỏ hơn kernel này. Cách đếm của sách cho Hình 7.12 giả định không tái sử dụng giữa các
            khối, giống như thanh trượt-hết.
          </p>
        }
      >
        <p>
          Three things in the explorer are easy to miss. First, a halo read that misses is not always extra traffic: if it
          brings in a line that its owner block loads a moment later, the owner hits instead. What counts is how many times
          each line comes from DRAM, and that is what the bars measure. Second, the explorer counts whole cache lines, so its
          no-reuse bar for this kernel is higher than the table's middle row: a halo two columns wide still costs a full line
          per row. Set the line size to 32 bytes and the two no-reuse bars come out about equal. Third, the same cache serves
          Fig. 7.12: its neighbouring input tiles overlap by 2r, and in the model it reaches the ideal at a smaller capacity
          than this kernel does. The book's count for Fig. 7.12 assumes no reuse between blocks, just like the all-miss bar.
        </p>
      </Translated>
      <Ask q={questionsS76['ch07/7.6/q-xcd']} />
      <Translated
        vi={
          <p>
            GPU của chính bạn làm gì là chuyện phải đo, và có hai cách đo. Lab 7.2 đo thời gian kernel này so với kernel phân
            ô có vùng biên ở nhiều kích thước ô và bán kính, và trang lab yêu cầu bạn dự đoán tỉ lệ trước khi chạy; bảng sau
            bảng kế tiếp cho thấy những gì máy này đã ghi lại. Một profiler có thể cho thấy chính bộ nhớ đệm, dưới dạng tỉ lệ
            yêu cầu L2 trúng. Đây là những gì từng hãng ghi trong tài liệu về bộ nhớ đệm mà mục này dựa vào, và công cụ của họ
            để đo tỉ lệ trúng.
          </p>
        }
      >
        <p>
          What your own GPU does is a measurement, and there are two ways to take it. Lab 7.2 times this kernel against the
          halo-tiled one at several tile sizes and radii, and the labs page asks you to predict the ratio before you run
          it; the table after the next one shows what this machine has recorded. A profiler can show the cache itself, as
          the share of L2 requests that hit. Here is what each vendor documents about the cache this section relies on,
          and its tool for the hit rate.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 (notes/today/ch07-facts.md A5, A6, A9 for NVIDIA; B4, B5 for AMD; C4 for Apple)',
          '2026-11-20 (notes/today/ch07-facts.md A5, A6, A9 cho NVIDIA; B4, B5 cho AMD; C4 cho Apple)',
        )}
      >
        <Translated vi={<p>Hàng của Apple đến từ một bài nói ở hội nghị, không phải từ tài liệu tham chiếu.</p>}>
          <p>Apple&rsquo;s row comes from a conference talk, not from reference documentation.</p>
        </Translated>
      </Today>
      <table className="data" data-table="vendor-caches">
        <thead>
          <tr>
            <th><Translated vi="Hãng">Vendor</Translated></th>
            <th><Translated vi="Bộ nhớ đệm phía sau một lần đọc vùng biên">The cache behind a halo read</Translated></th>
            <th><Translated vi="Lấy tỉ lệ trúng từ">Hit rate from</Translated></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>NVIDIA</td>
            <td>
              <Translated
                vi={
                  <>
                    một L2 cho mọi SM, kích thước trong <code>l2CacheSize</code>; các lần nạp <code>const __restrict__</code>{' '}
                    trở thành <code>ld.global.nc</code>
                  </>
                }
              >
                one L2 for all SMs, size in <code>l2CacheSize</code>; <code>const __restrict__</code> loads become{' '}
                <code>ld.global.nc</code>
              </Translated>
            </td>
            <td>
              <Translated vi="Nsight Compute; kém tin cậy hơn với các kernel rất ngắn">Nsight Compute; less reliable for very short kernels</Translated>
            </td>
          </tr>
          <tr>
            <td>AMD</td>
            <td>
              <Translated
                vi={
                  <>
                    mỗi die một L2; trên MI300X, mỗi XCD một L2, nằm dưới một cache cấp cuối dùng chung cho toàn GPU; việc đặt
                    workgroup không tất định; <code>__ldg</code> là một lần nạp thông thường
                  </>
                }
              >
                an L2 per die; on MI300X, one per XCD, under a last-level cache shared by the whole GPU; workgroup placement is not deterministic;{' '}
                <code>__ldg</code> is a plain load
              </Translated>
            </td>
            <td>
              <Translated vi="L2 Hit Rate của rocprofiler-compute, vốn tính một hit-on-miss là một lần trúng">
                rocprofiler-compute&rsquo;s L2 Hit Rate, which counts a hit-on-miss as a hit
              </Translated>
            </td>
          </tr>
          <tr>
            <td>
              <Translated vi="Apple">Apple</Translated>
            </td>
            <td>
              <Translated vi="một cache cấp cuối cho mọi lõi GPU; không nêu kích thước">one last-level cache for all GPU cores; no size given</Translated>
            </td>
            <td>
              <Translated vi={<>các bộ đếm &ldquo;Last Level Cache&rdquo; của Xcode</>}>Xcode&rsquo;s &ldquo;Last Level Cache&rdquo; counters</Translated>
            </td>
          </tr>
        </tbody>
      </table>
      <Measured />

      <h2>
        <Translated vi="Trong các backend thực tế">In real backends</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Các kernel của mục 7.5 và 7.6 đánh dấu hai đầu của một dải: nạp trước toàn bộ vùng biên, hoặc không nạp chút nào
            và tin vào bộ nhớ đệm. Phép tích chập depthwise của MLX trên Metal chọn đầu thứ nhất. Nó nạp trước toàn bộ ô đầu
            vào cùng vùng biên vào bộ nhớ threadgroup, ghi số không cho các ô ma, và các luồng của nó lặp để nạp, giống Bài tập
            11 của sách. Nó truyền kích thước kernel dưới dạng một function constant và định kích thước mảng threadgroup cho
            kernel lớn nhất mà nó chấp nhận (một function constant không thể định kích thước cho mảng đó). Đầu kia, một kernel
            đọc mọi tap từ bộ nhớ toàn cục, là phép tích chập CUDA trực tiếp của llama.cpp ở mục{' '}
            <a href={href('ch07', '7.2')}>7.2</a>.
          </p>
        }
      >
        <p>
          The kernels of sections 7.5 and 7.6 mark the ends of a spectrum: stage the whole halo, or stage none of it and
          trust the cache. MLX&rsquo;s depthwise convolution on Metal takes the first end. It stages the whole input tile
          with its halo in threadgroup memory, writing zeros for ghost cells, and its threads loop to load it, like the
          book&rsquo;s Exercise 11. It passes the kernel size as a function constant and sizes its threadgroup array for
          the largest kernel it accepts (a function constant cannot size one). The other end, a kernel that reads
          every tap from global memory, is llama.cpp&rsquo;s direct CUDA convolution of section{' '}
          <a href={href('ch07', '7.2')}>7.2</a>.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch07-facts.md, C3, D2)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              MLX <code>depthwise_conv_2d</code> trong <code>mlx/backend/metal/kernels/conv.metal</code>. Các phép tích chập
              tổng quát thì chuyển sang phép nhân ma trận (chương 19).
            </p>
          }
        >
          <p>
            MLX <code>depthwise_conv_2d</code> in <code>mlx/backend/metal/kernels/conv.metal</code>. General convolutions
            go to matrix multiplication instead (chapter 19).
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Tiếp theo: <a href={href('ch07', 'labs')}>trang lab</a> cho lab 7.2, và phần{' '}
            <a href={href('ch07', 'wrap')}>tóm tắt</a> cho những gì cần mang theo từ chương này.
          </p>
        }
      >
        <p>
          Next: the <a href={href('ch07', 'labs')}>labs page</a> for lab 7.2, and the{' '}
          <a href={href('ch07', 'wrap')}>summary</a> for what to carry forward from the chapter.
        </p>
      </Translated>
    </>
  );
}
