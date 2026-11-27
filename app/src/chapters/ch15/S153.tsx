import { useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { defineLabs } from '../../components/LabKit';
import { cuStatus, metalStatus } from './labStatus';
import { Aside, Bench, Beyond, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BOOK,
  SECTOR,
  WARP,
  gridDims,
  lastTileZeroColumns,
  launch2D,
  loaderMap,
  loadsPerThreadPerTilePair,
  owns,
  slotOf,
  tileLoadSummary,
  tilePairs,
  tileWindow,
  warpLoad,
  type LoadMode,
  type Ownership,
  type WarpLoad,
} from '../../sim/gemmmap';

// ---------------------------------------------------------------- the lab files this section shows
const LABS = defineLabs('ch15', [
  { num: '15.1', dir: 'lab1_coarse', name: 'coarsened', vi: { name: 'gộp việc cho luồng' } },
]);
const [LAB1] = LABS;
const SOL = 'ch15/lab1_coarse/solution/';

/** Identifiers coloured by the memory they live in, as the diagrams colour them. */
export const GEMM15_TIERS: Record<string, Tier> = {
  A: 'global', B: 'global', C: 'global', T: 'global',
  A_s: 'shared', B_s: 'shared', T_s: 'shared',
  C_r: 'reg', a_r: 'reg', b_r: 'reg',
  bRow: 'reg', bCol: 'reg', tRow: 'reg', tCol: 'reg', ty: 'reg', tx: 'reg', row: 'reg', col: 'reg', i: 'reg', tile: 'reg',
};

/**
 * One top-level function of a lab file, located by name: from the line that defines it
 * to its closing brace in column 0. The lesson shows a slice of the tested file with the
 * file's own line numbers; nothing is retyped.
 */
export function functionOf(text: string, name: string): { code: string; firstLine: number } | null {
  const lines = text.split('\n');
  const def = new RegExp(`\\b${name}\\s*\\(`);
  const start = lines.findIndex((l) => def.test(l) && !/^\s/.test(l) && !/^\s*(\/\/|\*)/.test(l) && !/;\s*$/.test(l));
  if (start < 0) return null;
  const end = lines.findIndex((l, i) => i > start && /^}/.test(l));
  return end < 0 ? null : { code: lines.slice(start, end + 1).join('\n'), firstLine: start + 1 };
}

function LabSlice({ fn, caption }: { fn: string; caption: ReactNode }) {
  const { reports } = useReports();
  const cu = useLabFile(SOL + 'kernel.cu');
  if (!cu) return null;
  const s = functionOf(cu, fn);
  return (
    <Code
      tabs={[{ label: tr(`${fn}, lab 15.1`, `${fn}, lab 15.1`), code: s ? s.code : cu, tiers: GEMM15_TIERS, status: cuStatus(reports, LAB1) }]}
      firstLine={s ? s.firstLine : 1}
      caption={caption}
    />
  );
}

function Listing() {
  const { reports } = useReports();
  const cu = useLabFile(SOL + 'kernel.cu');
  const metal = useLabFile(SOL + 'kernel.metal');
  const mm = useLabFile(SOL + 'launch.mm');
  if (!cu || !metal || !mm) return null;
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: cu, tiers: GEMM15_TIERS, status: cuStatus(reports, LAB1) },
        { label: tr('Metal kernel', 'Kernel Metal'), code: metal, tiers: GEMM15_TIERS, status: metalStatus(reports, LAB1) },
        { label: tr('Metal host', 'Host Metal'), code: mm, tiers: {}, status: metalStatus(reports, LAB1) },
      ]}
      caption={
        <Translated
          vi={
            <>
              Lời giải tham chiếu (solution) của lab 15.1, nạp từ <code>labs/ch15/lab1_coarse/solution/</code>; các trạng thái
              lấy từ báo cáo của máy này. Hãy thử <a href={href('ch15', 'labs')}>lab</a> trước: bạn sẽ học được nhiều hơn từ
              lỗi trong phiên bản của chính bạn so với đọc bản này. Cùng tệp CUDA đó là mã nguồn HIP, thông qua{' '}
              <code>pmpp/vendors/hip.h</code> của khóa học. Điều đó đúng với kernel này vì nó chỉ dùng luồng, khối, bộ nhớ
              chia sẻ và rào chắn; intrinsic của warp sẽ cần một lane mask 64-bit trên HIP, và lệnh ma trận cần các builtin
              khác (mục 15.9). Chưa máy HIP nào build nó (unverified).
            </>
          }
        >
          <>
            The reference solution of lab 15.1, loaded from <code>labs/ch15/lab1_coarse/solution/</code>; the statuses
            come from this machine's reports. Try the <a href={href('ch15', 'labs')}>lab</a> first: you will learn more
            from your own version's mistakes than from reading this one. The same CUDA file is the HIP source, through
            the course's <code>pmpp/vendors/hip.h</code>. That holds for this kernel because it uses only threads,
            blocks, shared memory and barriers; warp intrinsics would need a 64-bit lane mask on HIP, and matrix
            instructions different builtins (section 15.9). No HIP machine has built it yet (unverified).
          </>
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- explorer: who owns what
const clampInt = (v: number, lo: number, hi: number) => (Number.isFinite(v) ? Math.max(lo, Math.min(hi, Math.round(v))) : lo);

function NumField({ label, value, min, max, onChange, width = '5.5rem' }: { label: ReactNode; value: number; min: number; max: number; onChange: (v: number) => void; width?: string }) {
  return (
    <label className="field">
      {label}
      <input type="number" min={min} max={max} value={value} style={{ width }} onChange={(e) => onChange(clampInt(Number(e.target.value), min, max))} />
    </label>
  );
}

/** Fig. 15.1's layout: A bottom left, B top right, C bottom right; the block's panels and the thread's patch. */
function GlobalView({ M, N, o }: { M: number; N: number; o: Ownership }) {
  const S = 220 / Math.max(M, N);
  const kW = 70;
  const ax = 14, ty0 = 34, bY = ty0, cy = ty0 + kW + 12, cx = ax + kW + 12;
  const w = (v: number) => Math.max(1.5, v * S);
  const cW = w(N), cH = w(M);
  const r0 = Math.min(o.bRow, M), r1 = Math.min(o.bRow + BOOK.bM, M);
  const c0 = Math.min(o.bCol, N), c1 = Math.min(o.bCol + BOOK.bN, N);
  const px = cx + o.cols[0] * S, py = cy + o.rows[0] * S;
  const inside = o.writes > 0;
  return (
    <svg viewBox={`0 0 ${cx + 232} ${cy + Math.max(cH, 20) + 30}`} style={{ width: '100%', maxWidth: 340, display: 'block' }} role="img" aria-label={tr("The block's panels in global memory", 'Các dải của khối trong bộ nhớ toàn cục')} data-view="global">
      <text x={ax} y={14} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}>
        <Translated vi="bộ nhớ toàn cục">global memory</Translated>
      </text>
      {/* B, K x N */}
      <rect x={cx} y={bY} width={cW} height={kW} fill="var(--cell)" stroke="var(--rule)" />
      <rect x={cx + c0 * S} y={bY} width={Math.max(1.5, (c1 - c0) * S)} height={kW} fill="var(--global-wash)" stroke="var(--global)" data-panel="B" />
      <text x={cx + cW + 4} y={bY + 12} fontSize="10.5">B</text>
      <text x={cx - 4} y={bY + kW / 2 + 3} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>K</text>
      {/* A, M x K */}
      <rect x={ax} y={cy} width={kW} height={cH} fill="var(--cell)" stroke="var(--rule)" />
      <rect x={ax} y={cy + r0 * S} width={kW} height={Math.max(1.5, (r1 - r0) * S)} fill="var(--global-wash)" stroke="var(--global)" data-panel="A" />
      <text x={ax} y={cy - 4} fontSize="10.5">A</text>
      <text x={ax + kW / 2} y={cy - 4} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>K</text>
      {/* C, M x N */}
      <rect x={cx} y={cy} width={cW} height={cH} fill="var(--cell)" stroke="var(--rule)" />
      <rect x={cx + c0 * S} y={cy + r0 * S} width={Math.max(1.5, (c1 - c0) * S)} height={Math.max(1.5, (r1 - r0) * S)} fill="none" stroke="var(--ink)" strokeWidth="1.2" />
      {inside ? (
        <rect x={px - 1} y={py - 1} width={Math.max(4, BOOK.tN * S) + 2} height={Math.max(4, BOOK.tM * S) + 2} fill="var(--reg)" stroke="var(--reg)" data-patch="global" />
      ) : null}
      <text x={cx + cW + 4} y={cy + 12} fontSize="10.5">C</text>
      <text x={cx} y={cy + Math.max(cH, 20) + 16} fontSize="10" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi={<>{M} × {N}; viền đậm: khối ({o.bx}, {o.by})</>}>
          {M} × {N}; outlined: block ({o.bx}, {o.by})
        </Translated>
      </text>
    </svg>
  );
}

/** Fig. 15.2's layout at one block: A_s, B_s and the 128 x 128 tile of C as 16 x 16 patches. */
function BlockView({ M, N, o, mode2d }: { M: number; N: number; o: Ownership; mode2d: ReturnType<typeof launch2D> | null }) {
  const s = 2; // pixels per element
  const ax = 22, by0 = 30, cx = ax + BOOK.bK * s + 10, cy = by0 + BOOK.bK * s + 10;
  const tile = BOOK.bM * s;
  const per = BOOK.bN / BOOK.tN;
  const rowsIn = Math.max(0, Math.min(BOOK.bM, M - o.bRow));
  const colsIn = Math.max(0, Math.min(BOOK.bN, N - o.bCol));
  const cells: ReactNode[] = [];
  for (let p = 0; p < per * per; p++) {
    const r = Math.floor(p / per), c = p % per;
    let fill = 'none', stroke = 'none', dash: string | undefined;
    if (mode2d) {
      if (mode2d.writers[p]) fill = p === o.t ? 'var(--reg)' : 'var(--reg-wash)';
      else { fill = 'var(--fail-wash)'; }
    } else if (p === o.t) fill = o.writes > 0 ? 'var(--reg)' : 'none';
    if (!mode2d && p === o.t && o.writes === 0) { stroke = 'var(--reg)'; dash = '2 2'; }
    cells.push(
      <rect key={p} x={cx + c * BOOK.tN * s} y={cy + r * BOOK.tM * s} width={BOOK.tN * s} height={BOOK.tM * s}
        fill={fill} stroke={stroke} strokeDasharray={dash} data-p={p} data-state={mode2d ? (mode2d.writers[p] ? 'written' : 'never') : p === o.t ? 'mine' : ''} />,
    );
  }
  const aSlots: ReactNode[] = [];
  const bSlots: ReactNode[] = [];
  if (mode2d) {
    mode2d.aLoaded.forEach((ok, i) => {
      const r = Math.floor(i / BOOK.bK), c = i % BOOK.bK;
      aSlots.push(<rect key={i} x={ax + c * s} y={cy + r * s} width={s} height={s} fill={ok ? 'var(--shared)' : 'var(--fail-wash)'} />);
    });
    mode2d.bLoaded.forEach((ok, i) => {
      const r = Math.floor(i / BOOK.bN), c = i % BOOK.bN;
      bSlots.push(<rect key={i} x={cx + c * s} y={by0 + r * s} width={s} height={s} fill={ok ? 'var(--shared)' : 'var(--fail-wash)'} />);
    });
  }
  return (
    <svg viewBox={`0 0 ${cx + tile + 34} ${cy + tile + 34}`} style={{ width: '100%', maxWidth: 380, display: 'block' }} role="img" aria-label={tr("One block's tiles", 'Các ô của một khối')} data-view="block">
      <text x={ax} y={14} fontSize="11.5" fontWeight="600">
        <Translated vi={<>khối ({o.bx}, {o.by})</>}>
          block ({o.bx}, {o.by})
        </Translated>
      </text>
      {/* B_s */}
      <rect x={cx} y={by0} width={tile} height={BOOK.bK * s} fill="var(--shared-wash)" stroke="var(--shared)" />
      {bSlots}
      {!mode2d ? <rect x={cx + o.tCol * s} y={by0} width={BOOK.tN * s} height={BOOK.bK * s} fill="var(--shared)" data-strip="B_s" /> : null}
      <text x={cx + tile + 4} y={by0 + 12} fontSize="10" style={{ fill: 'var(--shared)' }}>B_s</text>
      {/* A_s */}
      <rect x={ax} y={cy} width={BOOK.bK * s} height={tile} fill="var(--shared-wash)" stroke="var(--shared)" />
      {aSlots}
      {!mode2d ? <rect x={ax} y={cy + o.tRow * s} width={BOOK.bK * s} height={BOOK.tM * s} fill="var(--shared)" data-strip="A_s" /> : null}
      <text x={ax - 2} y={cy - 4} fontSize="10" style={{ fill: 'var(--shared)' }}>A_s</text>
      {/* C tile */}
      <rect x={cx} y={cy} width={tile} height={tile} fill="var(--cell)" stroke="var(--rule)" />
      {rowsIn < BOOK.bM || colsIn < BOOK.bN ? (
        <path d={`M${cx + colsIn * s},${cy} H${cx + tile} V${cy + tile} H${cx} V${cy + rowsIn * s} H${cx + colsIn * s} Z`} fill="var(--rule-soft)" />
      ) : null}
      {cells}
      {Array.from({ length: per - 1 }, (_, k) => (
        <g key={k} stroke="var(--rule-soft)">
          <line x1={cx + (k + 1) * BOOK.tN * s} y1={cy} x2={cx + (k + 1) * BOOK.tN * s} y2={cy + tile} />
          <line x1={cx} y1={cy + (k + 1) * BOOK.tM * s} x2={cx + tile} y2={cy + (k + 1) * BOOK.tM * s} />
        </g>
      ))}
      <rect x={cx} y={cy} width={tile} height={tile} fill="none" stroke="var(--ink)" strokeWidth="1" />
      {rowsIn < BOOK.bM || colsIn < BOOK.bN ? (
        <rect x={cx} y={cy} width={colsIn * s} height={rowsIn * s} fill="none" stroke="var(--ink-2)" strokeDasharray="3 2" />
      ) : null}
      <text x={cx} y={cy + tile + 14} fontSize="10" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi="Ô C, 128 × 128: 16 × 16 mảng 8 × 8">C tile, 128 × 128: 16 × 16 patches of 8 × 8</Translated>
      </text>
      <text x={cx} y={cy + tile + 27} fontSize="10" style={{ fill: 'var(--ink-2)' }}>
        {rowsIn < BOOK.bM || colsIn < BOOK.bN
          ? tr(`grey: outside C (${rowsIn} rows, ${colsIn} columns inside)`, `xám: ngoài C (${rowsIn} hàng, ${colsIn} cột ở trong)`)
          : ''}
      </text>
    </svg>
  );
}

function OwnerExplorer() {
  const [M, setM] = useState(1024);
  const [N, setN] = useState(1024);
  const [bx, setBx] = useState(0);
  const [by, setBy] = useState(0);
  const [t, setT] = useState(0);
  const [twoD, setTwoD] = useState(false);
  const [x, setX] = useState(0);
  const [y, setY] = useState(0);
  const g = gridDims(M, N);
  const bxc = Math.min(bx, g.x - 1);
  const byc = Math.min(by, g.y - 1);
  const tEff = twoD ? x : t;
  const o = owns(M, N, bxc, byc, tEff);
  const l2 = useMemo(() => launch2D(16, 16), []);
  const per = o.tilesPerBlockX;
  const twoDFacts = twoD ? (
    <ul className="ui" style={{ margin: '0.4rem 0 0', paddingLeft: '1.1rem' }} data-twod>
      <Translated
        vi={
          <li>
            <b>{l2.patchesWritten}</b> trên 256 mảng được ghi, hàng trên cùng, mỗi mảng bởi <b>{l2.writers[0]}</b> luồng.
          </li>
        }
      >
        <li>
          <b>{l2.patchesWritten}</b> of 256 patches written, the top row, each by <b>{l2.writers[0]}</b> threads.
        </li>
      </Translated>
      <Translated
        vi={
          <li>
            Hàng 8 đến 127 của ô khối: <b className="t-fail">không bao giờ được ghi</b> ({l2.rowsNeverWritten} hàng).
          </li>
        }
      >
        <li>
          Rows 8 to 127 of the block tile: <b className="t-fail">never written</b> ({l2.rowsNeverWritten} rows).
        </li>
      </Translated>
      <Translated
        vi={
          <li>
            <code>loadTile</code> lấp <b>{l2.aLoadedCount}</b> trên 1024 ô nhớ của <code>A_s</code> và <b>{l2.bLoadedCount}</b>{' '}
            trên 1024 của <code>B_s</code>; phần còn lại chứa bất cứ thứ gì đã có ở đó.
          </li>
        }
      >
        <li>
          <code>loadTile</code> fills <b>{l2.aLoadedCount}</b> of 1024 slots of <code>A_s</code> and <b>{l2.bLoadedCount}</b> of 1024 of{' '}
          <code>B_s</code>; the rest hold whatever was there.
        </li>
      </Translated>
      <Translated
        vi={
          <li>
            Số mảng có mọi đầu vào đều được nạp: <b className={l2.correctPatches ? '' : 't-fail'}>{l2.correctPatches}</b>.
          </li>
        }
      >
        <li>
          Patches whose inputs were all loaded: <b className={l2.correctPatches ? '' : 't-fail'}>{l2.correctPatches}</b>.
        </li>
      </Translated>
    </ul>
  ) : null;
  return (
    <Bench
      title={tr('Who owns what', 'Ai sở hữu gì')}
      task={
        <Translated vi="Chọn một khối và một luồng, rồi đọc ra nó sở hữu gì, nó đọc gì từ bộ nhớ chia sẻ, và khối của nó lần lượt duyệt qua những dải nào.">
          Pick a block and a thread, and read off what it owns, what it reads from shared memory, and which panels its block streams through.
        </Translated>
      }
      foot={
        <>
          <Translated
            vi={
              <>
                Vẽ từ chính phép tính chỉ số của kernel (Fig. 15.3 dòng 05–13 và 36–39) với kích thước của sách: 128 × 128
                mỗi khối, 8 × 8 mỗi luồng, 256 luồng.
              </>
            }
          >
            Drawn from the kernel's own index arithmetic (Fig. 15.3 lines 05–13 and 36–39) with the book's sizes: 128 × 128
            per block, 8 × 8 per thread, 256 threads.
          </Translated>{' '}
          <TierLegend />
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <NumField label={tr('M (rows of C)', 'M (số hàng của C)')} value={M} min={1} max={16384} onChange={setM} />
        <NumField label={tr('N (columns of C)', 'N (số cột của C)')} value={N} min={1} max={16384} onChange={setN} />
        <label className="field">
          {tr('blockIdx (x, y)', 'blockIdx (x, y)')}
          <select value={`${bxc},${byc}`} aria-label={tr('block', 'khối')} onChange={(e) => { const [a, b] = e.target.value.split(',').map(Number); setBx(a); setBy(b); }}>
            {Array.from({ length: Math.min(g.blocks, 4096) }, (_, k) => [k % g.x, Math.floor(k / g.x)]).map(([a, b]) => (
              <option key={`${a},${b}`} value={`${a},${b}`}>({a}, {b})</option>
            ))}
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={twoD} onChange={(e) => setTwoD(e.target.checked)} />{' '}
          <Translated vi="khởi chạy dim3(16, 16) thay vào đó">launch dim3(16, 16) instead</Translated>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        {twoD ? (
          <>
            <label className="field">
              threadIdx.x <input type="range" min={0} max={15} value={x} aria-label="threadIdx.x" onChange={(e) => setX(+e.target.value)} />
              <input type="number" min={0} max={15} value={x} style={{ width: '4rem' }} aria-label={tr('threadIdx.x value', 'giá trị threadIdx.x')} onChange={(e) => setX(clampInt(+e.target.value, 0, 15))} />
            </label>
            <label className="field">
              threadIdx.y <input type="range" min={0} max={15} value={y} aria-label="threadIdx.y" onChange={(e) => setY(+e.target.value)} />
              <input type="number" min={0} max={15} value={y} style={{ width: '4rem' }} aria-label={tr('threadIdx.y value', 'giá trị threadIdx.y')} onChange={(e) => setY(clampInt(+e.target.value, 0, 15))} />
            </label>
          </>
        ) : (
          <label className="field">
            threadIdx.x <input type="range" min={0} max={255} value={t} aria-label="threadIdx.x" style={{ width: '16rem' }} onChange={(e) => setT(+e.target.value)} />
            <input type="number" min={0} max={255} value={t} style={{ width: '4.5rem' }} aria-label={tr('threadIdx.x value', 'giá trị threadIdx.x')} onChange={(e) => setT(clampInt(+e.target.value, 0, 255))} />
          </label>
        )}
      </div>

      <div className="narrate" role="status" data-readout style={{ marginBottom: '0.9rem' }}>
        {twoD ? (
          <Translated
            vi={
              <p style={{ margin: '0 0 0.4rem' }}>
                Luồng (x {x}, y {y}) của một khối 16 × 16. Kernel không bao giờ đọc <code>threadIdx.y</code>, nên luồng này
                làm đúng những gì luồng {x} của khối dự định làm, và 15 luồng khác có x = {x} cũng vậy. Các luồng 16 đến 255
                của khối dự định không ai đóng vai.
              </p>
            }
          >
            <p style={{ margin: '0 0 0.4rem' }}>
              Thread (x {x}, y {y}) of a 16 × 16 block. The kernel never reads <code>threadIdx.y</code>, so this thread does
              exactly what thread {x} of the intended block does, and so do the 15 other threads with x = {x}. Threads 16 to
              255 of the intended block are played by nobody.
            </p>
          </Translated>
        ) : null}
        <div className="mono" style={{ fontSize: '0.8rem', lineHeight: 1.6 }}>
          <div>bRow = blockIdx.y · bM = {byc} · 128 = <b data-v="bRow">{o.bRow}</b>; bCol = blockIdx.x · bN = {bxc} · 128 = <b data-v="bCol">{o.bCol}</b></div>
          <div>ty = {tEff} / (bN / tN) = {tEff} / {per} = <b data-v="ty">{o.ty}</b>; tx = {tEff} % {per} = <b data-v="tx">{o.tx}</b></div>
          <div>tRow = ty · tM = <b data-v="tRow">{o.tRow}</b>; tCol = tx · tN = <b data-v="tCol">{o.tCol}</b></div>
        </div>
        <p style={{ margin: '0.45rem 0 0' }} data-owns>
          <Translated
            vi={
              <>
                <span className="t-reg">Thanh ghi:</span> C hàng <b>{o.rows[0]}</b> đến {o.rows[1]}, cột <b>{o.cols[0]}</b> đến {o.cols[1]}
                {o.writes === 64
                  ? ' (64 phần tử).'
                  : o.writes === 0
                    ? '. Mảng đó nằm ngoài C: maxRow hoặc maxCol bằng 0, và writeTile không ghi gì.'
                    : `; chỉ ${Math.min(o.maxRow, 8)} × ${Math.min(o.maxCol, 8)} = ${o.writes} nằm trong C, và writeTile chỉ ghi những phần tử đó.`}{' '}
                <span className="t-shared">Chia sẻ:</span> hàng {o.aRows[0]} đến {o.aRows[1]} của <code>A_s</code> (cả 8 cột), cột{' '}
                {o.bCols[0]} đến {o.bCols[1]} của <code>B_s</code> (cả 8 hàng).{' '}
                <span className="t-global">Toàn cục:</span> khối duyệt qua hàng {o.bRow} đến {Math.min(o.bRow + 127, M - 1)} của A và cột{' '}
                {o.bCol} đến {Math.min(o.bCol + 127, N - 1)} của B, mỗi lần 8.
              </>
            }
          >
            <span className="t-reg">Registers:</span> C rows <b>{o.rows[0]}</b> to {o.rows[1]}, columns <b>{o.cols[0]}</b> to {o.cols[1]}
            {o.writes === 64 ? ' (64 elements).' : o.writes === 0 ? '. That patch lies outside C: maxRow or maxCol is 0, and writeTile stores nothing.' : `; only ${Math.min(o.maxRow, 8)} × ${Math.min(o.maxCol, 8)} = ${o.writes} lie inside C, and writeTile stores just those.`}{' '}
            <span className="t-shared">Shared:</span> rows {o.aRows[0]} to {o.aRows[1]} of <code>A_s</code> (all 8 columns), columns{' '}
            {o.bCols[0]} to {o.bCols[1]} of <code>B_s</code> (all 8 rows).{' '}
            <span className="t-global">Global:</span> the block streams rows {o.bRow} to {Math.min(o.bRow + 127, M - 1)} of A and columns{' '}
            {o.bCol} to {Math.min(o.bCol + 127, N - 1)} of B, 8 at a time.
          </Translated>
        </p>
        {twoDFacts}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(15rem, 1fr))' }}>
        <GlobalView M={M} N={N} o={o} />
        <BlockView M={M} N={N} o={o} mode2d={twoD ? l2 : null} />
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- stepper: loadTile one sub-tile at a time
const PRESETS = [
  {
    key: 'aligned',
    get label() {
      return tr('Aligned: M = N = K = 1024', 'Căn chỉnh: M = N = K = 1024');
    },
    M: 1024, N: 1024, K: 1024, by: 0, tile: 0,
  },
  { key: 'k1001', label: 'K = 1001', M: 1024, N: 1024, K: 1001, by: 0, tile: 0 },
  {
    key: 'edge',
    get label() {
      return tr('Edge block: M = 130, K = 9', 'Khối ở mép: M = 130, K = 9');
    },
    M: 130, N: 256, K: 9, by: 1, tile: 1,
  },
] as const;

/** One warp's addresses as runs of whole 32-byte sectors, one line per run. */
function MemoryStrip({ wl, which }: { wl: WarpLoad; which: 'A' | 'B' }) {
  const slot = 12, gapS = 8, perLine = 8;
  const lines: { label: string; sectors: number[]; start: number; end: number }[] = [];
  for (const [start, end] of wl.runs) {
    const first = Math.floor(start / SECTOR), last = Math.floor((end - 1) / SECTOR);
    const secs = Array.from({ length: last - first + 1 }, (_, k) => first + k);
    const lanes = wl.lanes.filter((l) => l.branch !== 'zero' && l.byte >= start && l.byte < end);
    const r = lanes[0]?.gRow ?? 0;
    const cs = lanes.flatMap((l) => Array.from({ length: l.inside || l.vec }, (_, j) => l.gCol + j));
    for (let k = 0; k < secs.length; k += perLine)
      lines.push({ label: k ? '' : tr(`${which}[${r}][${Math.min(...cs)}…${Math.max(...cs)}]  byte ${start}`, `${which}[${r}][${Math.min(...cs)}…${Math.max(...cs)}]  từ byte ${start}`), sectors: secs.slice(k, k + perLine), start, end });
  }
  const lineH = 36, labelW = 0;
  const W = labelW + perLine * (8 * slot + gapS);
  const H = Math.max(1, lines.length) * lineH + 8;
  if (!lines.length)
    return (
      <Translated vi={<p className="muted" data-strip-empty>Không lane nào của warp này đọc bộ nhớ: tất cả đều đi vào nhánh điền số 0.</p>}>
        <p className="muted" data-strip-empty>No lane of this warp reads memory: every one takes the zero-fill branch.</p>
      </Translated>
    );
  return (
    <div style={{ overflowX: 'auto' }}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Global addresses of the selected warp', 'Địa chỉ toàn cục của warp đã chọn')} data-strip>
        {lines.map((ln, li) => (
          <g key={li} transform={`translate(0, ${li * lineH})`}>
            {ln.label ? <text x={0} y={10} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>{ln.label}</text> : null}
            {ln.sectors.map((s, k) => (
              <g key={s} transform={`translate(${labelW + k * (8 * slot + gapS)}, 15)`} data-sector={s}>
                {Array.from({ length: 8 }, (_, j) => {
                  const b = s * SECTOR + 4 * j;
                  const used = b >= ln.start && b < ln.end;
                  return <rect key={j} x={j * slot} y={0} width={slot - 1} height={15} fill={used ? 'var(--global)' : 'var(--cell)'} stroke="var(--global)" strokeWidth="0.6" />;
                })}
              </g>
            ))}
          </g>
        ))}
      </svg>
    </div>
  );
}

function LoadTileStepper() {
  const [which, setWhich] = useState<'A' | 'B'>('A');
  const [sub, setSub] = useState(0);
  const [warp, setWarp] = useState(0);
  const [mode, setMode] = useState<LoadMode>('scalar');
  const [p, setP] = useState({ M: 1024, N: 1024, K: 1024 });
  const [bx, setBx] = useState(0);
  const [by, setBy] = useState(0);
  const [tile, setTile] = useState(0);
  const g = gridDims(p.M, p.N);
  const pairs = tilePairs(p.K);
  const bxc = Math.min(bx, g.x - 1), byc = Math.min(by, g.y - 1), tc = Math.min(tile, pairs - 1);
  const win = tileWindow(which, p, bxc, byc, tc, BOOK, mode);
  const plan = win.plan;
  const sc = Math.min(sub, plan.numSubtiles - 1);
  const map = useMemo(() => loaderMap(plan), [plan.height, plan.width, plan.mode]); // eslint-disable-line react-hooks/exhaustive-deps
  const wl = warpLoad(which, p, bxc, byc, tc, sc, warp, BOOK, mode);
  const sum = tileLoadSummary(which, p, bxc, byc, tc, BOOK, mode);
  const inside = (r: number, c: number) => r < win.maxRow && c < win.maxCol;
  const warpOf = (t: number) => Math.floor(t / WARP);
  const preset = (k: (typeof PRESETS)[number]) => {
    setP({ M: k.M, N: k.N, K: k.K });
    setBy(k.by);
    setBx(0);
    setTile(k.tile);
    setSub(0);
  };
  const isA = which === 'A';

  // Overview of the whole tile.
  const ow = isA ? 10 : 4, oh = isA ? 2.5 : 10;
  const overview = (
    <svg viewBox={`0 0 ${plan.width * ow + 60} ${plan.height * oh + 18}`} style={{ width: isA ? 140 : '100%', maxWidth: isA ? 140 : 580, display: 'block' }} role="img" aria-label={tr(`The whole ${which} tile`, `Toàn bộ ô ${which}`)} data-overview>
      <text x={0} y={10} fontSize="9.5" style={{ fill: 'var(--shared)' }}>{isA ? 'A_s' : 'B_s'}, {plan.height} × {plan.width}</text>
      {map.thread.map((t, i) => {
        const r = Math.floor(i / plan.width), c = i % plan.width;
        const s = map.sub[i];
        const mine = s === sc && warpOf(t) === warp;
        const zero = !inside(r, c);
        const fill = mine ? 'var(--global)' : s < sc ? 'var(--shared-wash)' : s === sc ? 'var(--shared)' : 'var(--cell)';
        return <rect key={i} x={c * ow} y={14 + r * oh} width={ow} height={oh} fill={fill} opacity={zero && !mine ? 0.35 : 1} />;
      })}
      <rect x={0} y={14} width={plan.width * ow} height={plan.height * oh} fill="none" stroke="var(--shared)" />
      {Array.from({ length: plan.numSubtiles }, (_, k) => (isA && plan.mode === 'scalar' ? (
        <text key={k} x={plan.width * ow + 4} y={14 + (k + 0.5) * plan.rowsPerSubTile * oh + 3} fontSize="9" style={{ fill: k === sc ? 'var(--ink)' : 'var(--ink-3)' }}>
          <Translated vi={<>ô con {k}</>}>sub-tile {k}</Translated>
        </text>
      ) : null))}
    </svg>
  );

  // Detail: a window of the tile with the loading thread written in each cell.
  const firstRow = wl.lanes[0]?.row ?? sc * plan.rowsPerSubTile;
  const winRows = isA ? 32 : 2;
  const r0 = Math.floor(firstRow / winRows) * winRows;
  const cw = isA ? 30 : 23, chh = 16, segCols = 32;
  const segs = isA ? 1 : Math.ceil(plan.width / segCols);
  const detailCells: ReactNode[] = [];
  for (let r = r0; r < Math.min(plan.height, r0 + winRows); r++)
    for (let c = 0; c < plan.width; c += plan.vec) {
      const i = r * plan.width + c;
      const t = map.thread[i];
      if (t < 0 || map.sub[i] !== sc) continue;
      const seg = isA ? 0 : Math.floor(c / segCols);
      const lx = 70 + (isA ? c : c % segCols) * cw;
      const ly = 16 + (isA ? r - r0 : seg * (winRows + 0.6) + (r - r0)) * chh;
      const mine = warpOf(t) === warp;
      const zero = !inside(r, c);
      detailCells.push(
        <g key={i} data-cell={`${r},${c}`} data-t={t} data-zero={zero ? 1 : 0}>
          <rect x={lx + 0.5} y={ly + 0.5} width={cw * plan.vec - 1} height={chh - 1} rx="1.5"
            fill={mine ? (zero ? 'var(--global-wash)' : 'var(--global)') : zero ? 'var(--cell)' : 'var(--shared-wash)'}
            stroke={mine ? 'var(--global)' : zero ? 'var(--ink-3)' : 'var(--shared)'} strokeWidth="0.7" strokeDasharray={zero ? '2 1.5' : undefined} />
          <text x={lx + (cw * plan.vec) / 2} y={ly + 11.5} fontSize="9.5" textAnchor="middle" style={{ fill: mine && !zero ? 'var(--page)' : zero ? 'var(--ink-3)' : 'var(--ink)' }}>{t}</text>
        </g>,
      );
    }
  const rowLabels: ReactNode[] = [];
  for (let r = r0; r < Math.min(plan.height, r0 + winRows); r++)
    for (let seg = 0; seg < segs; seg++) {
      const ly = 16 + (isA ? r - r0 : seg * (winRows + 0.6) + (r - r0)) * chh;
      rowLabels.push(
        <text key={`${r}-${seg}`} x={0} y={ly + 11.5} fontSize="9" style={{ fill: 'var(--ink-3)' }}>
          {isA ? tr(`row ${r}`, `hàng ${r}`) : tr(`row ${r}, ${seg * segCols}–${seg * segCols + segCols - 1}`, `hàng ${r}, ${seg * segCols}–${seg * segCols + segCols - 1}`)}
        </text>,
      );
    }
  const dW = 70 + (isA ? plan.width : segCols) * cw + 4;
  const dH = 22 + (isA ? winRows : segs * (winRows + 0.6)) * chh;
  const detail = (
    <div style={{ overflowX: 'auto' }}>
      <svg width={dW} height={dH} viewBox={`0 0 ${dW} ${dH}`} role="img" aria-label={tr('Which thread loads each element', 'Luồng nào nạp mỗi phần tử')} data-detail>
        <text x={0} y={10} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>
          {plan.mode === 'scalar'
            ? tr(`sub-tile ${sc}: the thread that loads each element`, `ô con ${sc}: luồng nạp mỗi phần tử`)
            : tr(`rows ${r0}–${r0 + winRows - 1}: one float4 per thread`, `hàng ${r0}–${r0 + winRows - 1}: một float4 mỗi luồng`)}
        </text>
        {rowLabels}
        {detailCells}
      </svg>
    </div>
  );

  const loads = wl.lanes.filter((l) => l.branch === 'load');
  const rows = [...new Set(wl.lanes.map((l) => l.row))];
  const misaligned = wl.lanes.some((l) => l.misaligned && l.branch !== 'zero');

  return (
    <Bench
      title={tr('loadTile, one sub-tile at a time', 'loadTile, từng ô con một')}
      task={
        <Translated vi="Duyệt qua các ô con của mỗi ô, chọn một warp, và so sánh các địa chỉ mà một lần nạp A và một lần nạp B chạm tới.">
          Step through the sub-tiles of each tile, pick a warp, and compare the addresses an A load and a B load touch.
        </Translated>
      }
      foot={
        <>
          <Translated
            vi={
              <>
                Sector là đơn vị truy cập bộ nhớ toàn cục 32 byte của NVIDIA, đếm theo mỗi lệnh warp với ma trận bắt đầu ở biên
                256 byte. Một warp là 32 luồng, theo giả định của sách.
              </>
            }
          >
            Sectors are NVIDIA's 32-byte unit of global-memory access, counted per warp instruction with the matrix starting
            on a 256-byte boundary. A warp is 32 threads, the book's assumption.
          </Translated>{' '}
          <TierLegend tiers={['shared', 'global']} />
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label={tr('Tile', 'Ô')}>
          <button aria-pressed={isA} onClick={() => { setWhich('A'); setSub(0); }}>
            <Translated vi="Ô A (128 × 8)">A tile (128 × 8)</Translated>
          </button>
          <button aria-pressed={!isA} onClick={() => { setWhich('B'); setSub(0); }}>
            <Translated vi="Ô B (8 × 128)">B tile (8 × 128)</Translated>
          </button>
        </span>
        <span className="seg" role="group" aria-label={tr('Presets', 'Cấu hình mẫu')}>
          {PRESETS.map((k) => (
            <button key={k.key} aria-pressed={p.M === k.M && p.N === k.N && p.K === k.K && byc === k.by && tc === k.tile} onClick={() => preset(k)}>{k.label}</button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <NumField label="M" value={p.M} min={1} max={16384} onChange={(v) => setP({ ...p, M: v })} width="5rem" />
        <NumField label="N" value={p.N} min={1} max={16384} onChange={(v) => setP({ ...p, N: v })} width="5rem" />
        <NumField label="K" value={p.K} min={1} max={16384} onChange={(v) => setP({ ...p, K: v })} width="5rem" />
        <label className="field">
          {tr('blockIdx (x, y)', 'blockIdx (x, y)')}
          <select value={`${bxc},${byc}`} aria-label={tr('block', 'khối')} onChange={(e) => { const [a, b] = e.target.value.split(',').map(Number); setBx(a); setBy(b); }}>
            {Array.from({ length: Math.min(g.blocks, 4096) }, (_, k) => [k % g.x, Math.floor(k / g.x)]).map(([a, b]) => (
              <option key={`${a},${b}`} value={`${a},${b}`}>({a}, {b})</option>
            ))}
          </select>
        </label>
        <NumField label={tr(`tile pair (0–${pairs - 1})`, `cặp ô (0–${pairs - 1})`)} value={tc} min={0} max={pairs - 1} onChange={setTile} width="4.5rem" />
        <label className="check">
          <input type="checkbox" checked={mode === 'float4'} onChange={(e) => { setMode(e.target.checked ? 'float4' : 'scalar'); setSub(0); }} />{' '}
          <Translated vi="nạp vector (Bài tập 1)">vector loads (Exercise 1)</Translated>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setSub(Math.max(0, sc - 1))} disabled={sc === 0}>
          <Translated vi="Lùi">Back</Translated>
        </button>
        <button className="primary" onClick={() => setSub(Math.min(plan.numSubtiles - 1, sc + 1))} disabled={sc >= plan.numSubtiles - 1}>
          <Translated vi="Bước">Step</Translated>
        </button>
        <span className="muted" aria-live="polite" data-sub>
          <Translated
            vi={
              <>
                ô con {sc} trên {plan.numSubtiles} (
                {plan.numSubtiles === 1 ? 'một lệnh nạp mỗi luồng' : `mỗi ô con ${plan.rowsPerSubTile} hàng`})
              </>
            }
          >
            sub-tile {sc} of {plan.numSubtiles} ({plan.numSubtiles === 1 ? 'one load instruction per thread' : `${plan.rowsPerSubTile} rows each`})
          </Translated>
        </span>
        <span className="seg" role="group" aria-label={tr('Warp', 'Warp')}>
          {Array.from({ length: BOOK.threads / WARP }, (_, k) => (
            <button key={k} aria-pressed={k === warp} onClick={() => setWarp(k)}>
              <Translated vi={<>warp {k}</>}>warp {k}</Translated>
            </button>
          ))}
        </span>
      </div>

      <div className={'narrate' + (misaligned ? ' bad' : '')} role="status" data-warp-readout style={{ marginBottom: '0.9rem' }}>
        <Translated
          vi={
            <>
              Warp {warp} (luồng {warp * WARP} đến {warp * WARP + WARP - 1}), ô con {sc}: {plan.vec === 4 ? 'mỗi luồng một float4, ' : ''}
              {rows.length > 1
                ? `các hàng ô ${Math.min(...rows)} đến ${Math.max(...rows)}, tức các hàng ${win.row0 + Math.min(...rows)} đến ${win.row0 + Math.max(...rows)} của ${which}`
                : `hàng ô ${rows[0]}, tức hàng ${win.row0 + rows[0]} của ${which}`}
              .{' '}
            </>
          }
        >
          Warp {warp} (threads {warp * WARP} to {warp * WARP + WARP - 1}), sub-tile {sc}: {plan.vec === 4 ? 'one float4 each, ' : ''}
          {rows.length > 1 ? `tile rows ${Math.min(...rows)} to ${Math.max(...rows)}, so ${which} rows ${win.row0 + Math.min(...rows)} to ${win.row0 + Math.max(...rows)}` : `tile row ${rows[0]}, so ${which} row ${win.row0 + rows[0]}`}.{' '}
        </Translated>
        {loads.length || wl.lanes.some((l) => l.branch === 'partial') ? (
          <Translated
            vi={
              <>
                Nó chạm tới <b data-v="sectors">{wl.sectors.length}</b> sector trong{' '}
                <b data-v="runs">{wl.runs.length}</b> {wl.runs.length === 1 ? 'đoạn' : 'đoạn riêng biệt'} địa chỉ
                {isA && wl.runs.length > 1 ? `, cách nhau một hàng của A (K · 4 = ${p.K * 4} byte)` : ''}.{' '}
              </>
            }
          >
            <>
              It touches <b data-v="sectors">{wl.sectors.length}</b> sector{wl.sectors.length === 1 ? '' : 's'} in{' '}
              <b data-v="runs">{wl.runs.length}</b> {wl.runs.length === 1 ? 'run' : 'separate runs'} of addresses
              {isA && wl.runs.length > 1 ? `, one row of A apart (K · 4 = ${p.K * 4} bytes)` : ''}.{' '}
            </>
          </Translated>
        ) : null}
        <Translated
          vi={
            <>
              <span data-v="zero">{wl.zeroLanes}</span> trên {wl.lanes.length} lane đi vào nhánh điền số 0
              {wl.diverges ? <b data-v="diverges">: warp phân kỳ (divergence) ở phép kiểm tra biên.</b> : wl.zeroLanes ? ', tất cả: không có phân kỳ.' : '.'}
            </>
          }
        >
          <span data-v="zero">{wl.zeroLanes}</span> of {wl.lanes.length} lanes take the zero-fill branch
          {wl.diverges ? <b data-v="diverges">: the warp diverges at the bounds check.</b> : wl.zeroLanes ? ', all of them: no divergence.' : '.'}
        </Translated>
        {misaligned ? (
          <>
            <br />
            <Translated
              vi={
                <b className="t-fail">
                  K = {p.K} không phải bội số của 4, nên các địa chỉ float4 này không căn chỉnh 16 byte: một lệnh nạp float4 từ
                  chúng phá vỡ quy tắc căn chỉnh của kiểu.
                </b>
              }
            >
              <b className="t-fail">K = {p.K} is not a multiple of 4, so these float4 addresses are not 16-byte aligned: a float4 load from them breaks the type's alignment rule.</b>
            </Translated>
          </>
        ) : null}
      </div>

      <div className="row" style={{ alignItems: 'flex-start', gap: '1.2rem' }}>
        {overview}
        {detail}
      </div>
      <div style={{ marginTop: '0.9rem' }}>
        <div className="mx-label t-global">
          <Translated vi={<>địa chỉ của warp {warp} trong bộ nhớ toàn cục, trọn các sector 32 byte (tô đậm: byte nó đọc)</>}>
            warp {warp}'s addresses in global memory, whole 32-byte sectors (filled: bytes it reads)
          </Translated>
        </div>
        <MemoryStrip wl={wl} which={which} />
      </div>
      <p style={{ margin: '0.8rem 0 0' }} data-summary>
        <Translated
          vi={
            <>
              Toàn bộ ô {which} của khối ({bxc}, {byc}), cặp ô {tc}: <b>{sum.instructions}</b> lệnh warp, <b data-v="total-sectors">{sum.sectors}</b> sector,{' '}
              <b data-v="divergent">{sum.divergent}</b> lệnh phân kỳ ở phép kiểm tra biên, {sum.zeroFilled} phần tử được điền số 0.
            </>
          }
        >
          Whole {which} tile of block ({bxc}, {byc}), tile pair {tc}: <b>{sum.instructions}</b> warp instructions, <b data-v="total-sectors">{sum.sectors}</b> sectors,{' '}
          <b data-v="divergent">{sum.divergent}</b> diverging at the bounds check, {sum.zeroFilled} elements zero-filled.
        </Translated>
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const W77 = slotOf(tileWindow('A', { M: 1024, N: 1024, K: 1024 }, 0, 0, 0).plan, 77, 2)!;
const AL = warpLoad('A', { M: 1024, N: 1024, K: 1024 }, 0, 0, 0, 0, 0);
const BL = warpLoad('B', { M: 1024, N: 1024, K: 1024 }, 0, 0, 0, 0, 0);
const L2 = launch2D(16, 16);
const DECODE = { big: gridDims(1, 4096).blocks, small: gridDims(1, 4096, { ...BOOK, bM: 16, bN: 16 }).blocks };

export const questionsS153: Record<string, Question> = {
  'ch15/15.3/q-loadtile-row': {
    kind: 'numeric',
    id: 'ch15/15.3/q-loadtile-row',
    prompt: (
      <Translated vi={<>Ô A có kích thước 128 × 8 và khối có 256 luồng. Luồng 77 nạp hàng nào của ô trong ô con 2?</>}>
        <>
          The A tile is 128 × 8 and the block has 256 threads. Which row of the tile does thread 77 load in sub-tile 2?
        </>
      </Translated>
    ),
    answer: W77.row,
    explain: (
      <Translated
        vi={
          <p>
            <code>rowsPerSubTile = 256 / 8 = 32</code>, nên ô con 2 bắt đầu ở hàng 64. Trong đó, luồng 77 lấy hàng{' '}
            <code>77 / 8 = 9</code> và cột <code>77 % 8 = {W77.col}</code>: hàng 64 + 9 = {W77.row}. Tám luồng liên tiếp lấp
            một hàng rộng 8, đó là lý do một warp 32 luồng phủ bốn hàng.
          </p>
        }
      >
        <p>
          <code>rowsPerSubTile = 256 / 8 = 32</code>, so sub-tile 2 starts at row 64. Within it, thread 77 takes row{' '}
          <code>77 / 8 = 9</code> and column <code>77 % 8 = {W77.col}</code>: row 64 + 9 = {W77.row}. Eight consecutive
          threads fill one 8-wide row, which is why a warp of 32 covers four rows.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.3, Fig. 15.5, p. 355',
  },
  'ch15/15.3/q-sectors': {
    kind: 'choice',
    id: 'ch15/15.3/q-sectors',
    prompt: (
      <Translated
        vi={
          <>
            Một warp thực thi một lệnh nạp của <code>loadTile</code> cho ô A (128 × 8), và một lệnh khác cho ô B (8 × 128). K
            và N là bội số của 8 và các ma trận bắt đầu ở biên 256 byte. Đếm theo sector 32 byte của NVIDIA, câu nào đúng?
          </>
        }
      >
        <>
          One warp executes one load instruction of <code>loadTile</code> for the A tile (128 × 8), and another for the B
          tile (8 × 128). K and N are multiples of 8 and the matrices start on a 256-byte boundary. Counting NVIDIA's
          32-byte sectors, which is right?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Cả hai chạm tới bốn sector. Của B là một đoạn liên tiếp 128 byte; của A là bốn đoạn 32 byte trên bốn hàng, cách nhau K float.">
            {'Both touch four sectors. B\'s are one contiguous 128-byte run; A\'s are four 32-byte runs in four rows, K floats apart.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="A chạm tới 32 sector, mỗi lane một sector, vì các lane của nó đi dọc một cột.">
            {'A touches 32 sectors, one per lane, because its lanes walk down a column.'}
          </Translated>
        ),
        why: (
          <Translated vi="Tám lane liên tiếp lấp một hàng của ô: một đoạn 32 byte, không phải đi dọc cột.">
            {'Eight consecutive lanes fill one row of the tile: a 32-byte run, not a column walk.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="B chạm tới 32 sector, vì một hàng của B dài N float.">
            {'B touches 32 sectors, because a row of B is N floats long.'}
          </Translated>
        ),
        why: (
          <Translated vi="Warp đọc 32 float liên tiếp của một hàng: 128 byte, bốn sector.">
            {'The warp reads 32 consecutive floats of one row: 128 bytes, four sectors.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Mỗi lệnh chạm tới một sector: một lệnh warp là một giao dịch.">
            {'Each touches one sector: a warp instruction is one transaction.'}
          </Translated>
        ),
        why: (
          <Translated vi="Đơn vị là sector 32 byte, và 32 float là 128 byte.">
            {'The unit is the 32-byte sector, and 32 floats are 128 bytes.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mô hình cho {AL.sectors.length} sector trong {AL.runs.length} đoạn với A và {BL.sectors.length} trong{' '}
            {BL.runs.length} đoạn với B. Cả hai đều dùng mọi byte chúng nạp. Khác biệt nằm ở số chỗ mà địa chỉ của warp đi tới:
            bốn hàng của A, một hàng của B. Hãy cho K lẻ trong trình chạy từng bước và mỗi hàng của A sẽ vắt ngang một biên
            sector.
          </p>
        }
      >
        <p>
          The model gives {AL.sectors.length} sectors in {AL.runs.length} runs for A and {BL.sectors.length} in{' '}
          {BL.runs.length} run for B. Both use every byte they fetch. The difference is in how many places the warp's
          addresses go: four rows of A, one row of B. Make K odd in the stepper and each of A's rows straddles a sector
          boundary.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.3, Fig. 15.5, pp. 355–356; sectors: notes/today/ch06-facts.md A2; counts: our model (src/sim/gemmmap.ts)',
    vi: {
      source: 'ch. 15 §15.3, Fig. 15.5, pp. 355–356; sector: notes/today/ch06-facts.md A2; số đếm: mô hình của chúng tôi (src/sim/gemmmap.ts)',
    },
  },
  'ch15/15.3/q-2d-block': {
    kind: 'choice',
    id: 'ch15/15.3/q-2d-block',
    prompt: (
      <Translated
        vi={
          <>
            Phần văn của sách mô tả khối là 16 × 16 luồng, nên bạn khởi chạy kernel với <code>dim3 block(16, 16)</code>. Điều
            gì xảy ra?
          </>
        }
      >
        <>
          The book's prose describes the block as 16 × 16 threads, so you launch the kernel with{' '}
          <code>dim3 block(16, 16)</code>. What happens?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Chạy như cũ: đằng nào cũng 256 luồng, và kernel chỉ phụ thuộc vào số luồng.">
            {'It works the same: 256 threads either way, and the kernel only depends on how many there are.'}
          </Translated>
        ),
        why: (
          <Translated vi="Số lượng như nhau; kernel chỉ đọc threadIdx.x, giờ chạy từ 0 đến 15.">
            {'The count is the same; the kernel only ever reads threadIdx.x, which now runs from 0 to 15.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Kết quả sai: threadIdx.x dừng ở 15, nên ty luôn bằng 0; chỉ hàng mảng trên cùng được ghi, mỗi mảng bởi 16 luồng, và các ô chỉ được nạp một phần.">
            {'Wrong results: threadIdx.x stops at 15, so ty is always 0; only the top row of patches is written, 16 threads to each, and the tiles are only partly loaded.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Nó không khởi chạy được: kernel giả định một khối một chiều NUM_THREADS_PER_BLOCK, và runtime từ chối 16 × 16.">
            {'It fails to launch: the kernel assumes a one-dimensional block of NUM_THREADS_PER_BLOCK, and the runtime rejects 16 × 16.'}
          </Translated>
        ),
        why: (
          <Translated vi="Một khối 16 × 16 là lần khởi chạy hợp lệ; không có gì kiểm tra hình dạng so với hằng số.">
            {'A 16 × 16 block is a legal launch; nothing checks the shape against the constant.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Kết quả đúng nhưng chậm hơn, vì 16 luồng tính mỗi mảng và ghi các giá trị giống hệt nhau.">
            {'The results are right but slower, because 16 threads compute each patch and write identical values.'}
          </Translated>
        ),
        why: (
          <Translated vi="Phần lớn ô không bao giờ được ghi, và những gì được ghi được tính từ các ô nhớ không ai nạp.">
            {'Most of the tile is never written, and what is written is computed from slots nobody loaded.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Fig. 15.3 suy ra cả hai tọa độ từ <code>threadIdx.x</code>. Trong một khối 16 × 16, giá trị đó là 0 đến 15, nên{' '}
            <code>ty = threadIdx.x / 16</code> luôn bằng 0. Mô hình: {L2.patchesWritten} mảng được ghi, mỗi mảng bởi{' '}
            {L2.writers[0]} luồng; {L2.rowsNeverWritten} hàng không bao giờ được ghi; {L2.aLoadedCount} trên 1024 ô nhớ của
            mỗi ô được nạp; {L2.correctPatches} mảng có đủ mọi đầu vào. Mã cần một khối một chiều 256 luồng; "16 × 16" là cách
            sắp xếp logic của các ô, không phải của các luồng.
          </p>
        }
      >
        <p>
          Fig. 15.3 derives both coordinates from <code>threadIdx.x</code>. In a 16 × 16 block that is 0 to 15, so{' '}
          <code>ty = threadIdx.x / 16</code> is always 0. The model: {L2.patchesWritten} patches written, each by{' '}
          {L2.writers[0]} threads; {L2.rowsNeverWritten} rows never written; {L2.aLoadedCount} of 1024 slots of each
          tile loaded; {L2.correctPatches} patches with all their inputs. The code needs a one-dimensional block of 256;
          "16 × 16" is the logical arrangement of the tiles, not of the threads.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.3, pp. 352, 354; notes/ch15.md "Unclear in the PDF" item 7',
    vi: { source: 'ch. 15 §15.3, pp. 352, 354; notes/ch15.md "Unclear in the PDF" mục 7' },
  },
  'ch15/15.3/q-tail': {
    kind: 'numeric',
    id: 'ch15/15.3/q-tail',
    prompt: (
      <Translated vi={<>K = 1001. Trong cặp ô cuối cùng, loadTile điền số 0 vào bao nhiêu trong 8 cột của ô A?</>}>
        <>K = 1001. In the last tile pair, how many of the A tile's 8 columns does loadTile fill with zeros?</>
      </Translated>
    ),
    answer: lastTileZeroColumns(1001),
    explain: (
      <Translated
        vi={
          <p>
            Vòng lặp chạy <code>(K + bK - 1) / bK = 1008 / 8 = {tilePairs(1001)}</code> lần. Ô cuối phủ k = 1000 đến 1007, và
            chỉ k = 1000 tồn tại: một cột thật, {lastTileZeroColumns(1001)} cột điền số 0. Số 0 không đóng góp gì vào một tổng
            các tích, đó là lý do <code>mm</code> không cần kiểm tra biên.
          </p>
        }
      >
        <p>
          The loop runs <code>(K + bK - 1) / bK = 1008 / 8 = {tilePairs(1001)}</code> times. The last tile covers k = 1000
          to 1007, and only k = 1000 exists: one real column, {lastTileZeroColumns(1001)} zero-filled. Zeros add nothing to
          a sum of products, which is why <code>mm</code> needs no bounds checks.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.3, Fig. 15.3 line 20 and Fig. 15.5, pp. 354–356',
    vi: { source: 'ch. 15 §15.3, Fig. 15.3 dòng 20 và Fig. 15.5, pp. 354–356' },
  },
  'ch15/15.3/q-unroll': {
    kind: 'choice',
    id: 'ch15/15.3/q-unroll',
    prompt: (
      <Translated
        vi={
          <>
            Bạn xóa <code>#pragma unroll</code> trước vòng lặp cột của <code>mm</code>. Điều gì có thể thay đổi?
          </>
        }
      >
        <>
          You delete the <code>#pragma unroll</code> in front of <code>mm</code>'s column loop. What can change?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Chỉ tốc độ: khi không unroll, vòng lặp đánh chỉ số C_r bằng một biến, nên C_r có thể bị đặt vào bộ nhớ cục bộ.">
            {'Only speed: rolled, the loop indexes C_r with a variable, so C_r may be placed in local memory.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Kernel không còn biên dịch được: C_r được định nằm trong thanh ghi, và một mảng thanh ghi không thể được đánh chỉ số bằng biến vòng lặp.">
            {'The kernel no longer compiles: C_r is meant to live in registers, and a register array cannot be indexed by a loop variable.'}
          </Translated>
        ),
        why: (
          <Translated vi="Đánh chỉ số một mảng cục bộ bằng một biến là C++ bình thường. Mảng nằm ở đâu là quyết định của trình biên dịch.">
            {'Indexing a local array with a variable is ordinary C++. Where the array lives is the compiler\'s decision.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Kết quả trở nên sai: khi C_r rời thanh ghi, nó nằm trong bộ nhớ mà các luồng của một khối dùng chung.">
            {'The results become wrong: once C_r leaves the registers it sits in memory the threads of a block share.'}
          </Translated>
        ),
        why: (
          <Translated vi="Bộ nhớ cục bộ là riêng của mỗi luồng, như thanh ghi. Chỉ tốc độ của nó khác.">
            {'Local memory is private to each thread, like registers. Only its speed differs.'}
          </Translated>
        ),
      },
      {
        label: <Translated vi="Không gì cả: pragma chỉ để trang trí.">{'Nothing: the pragma is decoration.'}</Translated>,
        why: (
          <Translated vi="NVIDIA có nói các vòng lặp nhỏ với số lần lặp biết trước được unroll mặc định, nên vòng này rất có thể đã được unroll. Nhưng bạn không thể biết nếu không nhìn vào.">
            {'NVIDIA does say small loops with a known trip count are unrolled by default, so this one may well be. But you cannot know without looking.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            NVIDIA công bố rằng trình biên dịch mặc định unroll các vòng lặp nhỏ có số lần lặp biết trước và pragma yêu cầu
            unroll hoàn toàn. <code>C_r</code> có ở lại thanh ghi hay không thì thấy được trong báo cáo của trình biên dịch (
            <code>--resource-usage</code>, hoặc <code>--ptxas-options=-v</code>, in bộ nhớ cục bộ dưới dạng{' '}
            <code>lmem</code>) và trong <code>numRegs</code> từ <code>cudaFuncGetAttributes</code>, không phải trong kết quả.
            Trình giả lập chạy các luồng trên CPU, nên nó không thể cho bạn biết; Metal cũng không, vì nó không công bố báo cáo
            thanh ghi nào.
          </p>
        }
      >
        <p>
          NVIDIA documents that the compiler unrolls small loops with a known trip count by default and that the pragma
          asks for a full unroll. Whether <code>C_r</code> stayed in registers is visible in the compiler's report (
          <code>--resource-usage</code>, or <code>--ptxas-options=-v</code>, which prints local memory as{' '}
          <code>lmem</code>) and in <code>numRegs</code> from <code>cudaFuncGetAttributes</code>, not in the results. The emulator runs threads on a CPU, so it cannot tell
          you either way; neither can Metal, which documents no register report.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.3, pp. 353–354; notes/today/infra-facts.md A7, ch06-facts.md A13, ch15-facts.md A4 and C3',
    vi: { source: 'ch. 15 §15.3, pp. 353–354; notes/today/infra-facts.md A7, ch06-facts.md A13, ch15-facts.md A4 và C3' },
  },
  'ch15/15.3/q-decode-blocks': {
    kind: 'choice',
    id: 'ch15/15.3/q-decode-blocks',
    prompt: (
      <Translated
        vi={
          <>
            Lớp benchmark của lab 5.3: 1024 đầu vào thành 4096 đầu ra, ở batch 1 (M = 1, K = 1024, N = 4096). Kernel 128 × 128
            khởi chạy bao nhiêu khối, và kernel 16 × 16 của lab 5.3 đã khởi chạy bao nhiêu?
          </>
        }
      >
        <>
          Lab 5.3's benchmark layer: 1024 inputs to 4096 outputs, at batch 1 (M = 1, K = 1024, N = 4096). How many blocks
          does the 128 × 128 kernel launch, and how many did lab 5.3's 16 × 16 kernel?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi={`${DECODE.big} so với ${DECODE.small}, và mỗi khối 128 × 128 tính 127 hàng số 0.`}>
            {`${DECODE.big} against ${DECODE.small}, and each 128 × 128 block computes 127 rows of zeros.`}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi={`${DECODE.big} so với ${DECODE.big}: lưới chỉ phụ thuộc vào N, và cả hai phủ 4096 cột trong một hàng khối.`}>
            {`${DECODE.big} against ${DECODE.big}: the grid depends only on N, and both cover 4096 columns in one row of blocks.`}
          </Translated>
        ),
        why: (
          <Translated vi="Cả hai lưới đều cao một khối ở đây, nhưng một ô khối rộng 16 cần gấp tám lần số khối trên 4096 cột.">
            {'Both grids are one block tall here, but a 16-wide block tile needs eight times as many blocks across 4096 columns.'}
          </Translated>
        ),
      },
      {
        label: <Translated vi="1 so với 16: M = 1 vừa trong một ô khối.">{'1 against 16: M = 1 fits inside one block tile.'}</Translated>,
        why: (
          <Translated vi="Lưới là ceil(N / bN) nhân ceil(M / bM).">
            {'The grid is ceil(N / bN) by ceil(M / bM).'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi={`${DECODE.small} so với 4096: một khối cho mỗi 16 đầu ra, và một khối cho mỗi đầu ra.`}>
            {`${DECODE.small} against 4096: one block per 16 outputs, and one per output.`}
          </Translated>
        ),
        why: (
          <Translated vi="Đếm số ô khối: 4096 / 128 theo chiều ngang, 1 theo chiều dọc.">
            {'Count block tiles: 4096 / 128 across, 1 down.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>ceil(4096 / 128) × ceil(1 / 128) = {DECODE.big} × 1</code>, so với <code>4096 / 16 = {DECODE.small}</code>.
            Với một hàng thật, ô A được điền số 0 khiến 127 trên 128 hàng của mỗi khối là việc thừa, và vài chục khối không
            thể lấp đầy một GPU lớn. Mục 15.2 đã chỉ ra lý do trong phép tính: với m = 1, cường độ tính toán ở dưới 0.5 FLOP/B
            bất kể ô nào. Một lớp batch 1 cần một kernel nhân ma trận-vector; llama.cpp, chẳng hạn, chuyển sang kernel ma
            trận-vector cho tới 8 hàng.
          </p>
        }
      >
        <p>
          <code>ceil(4096 / 128) × ceil(1 / 128) = {DECODE.big} × 1</code>, against <code>4096 / 16 = {DECODE.small}</code>.
          With one real row, the zero-filled A tile makes 127 of each block's 128 rows wasted work, and a few dozen
          blocks cannot fill a large GPU. Section 15.2 showed the reason in the arithmetic: with m = 1 the intensity stays
          under 0.5 FLOP/B whatever the tile. A batch-1 layer wants a matrix-vector kernel; llama.cpp, for one, switches to
          matrix-vector kernels up to 8 rows.
        </p>
      </Translated>
    ),
    source: 'our arithmetic with Fig. 15.3\'s grid; §15.2 p. 351; notes/today/ch15-facts.md D1 and E1',
    vi: { source: 'phép tính của chúng tôi với lưới của Fig. 15.3; §15.2 p. 351; notes/today/ch15-facts.md D1 và E1' },
  },
  'ch15/15.3/q-vec-loads': {
    kind: 'numeric',
    id: 'ch15/15.3/q-vec-loads',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 1: <code>loadTile</code> được viết lại để nạp một <code>float4</code> mỗi lần truy cập. Mỗi luồng phát bao
            nhiêu lệnh nạp cho mỗi cặp ô (phần của nó trong một ô A và một ô B)?
          </>
        }
      >
        <>
          Exercise 1: <code>loadTile</code> rewritten to load one <code>float4</code> per access. How many load instructions
          does each thread issue per tile pair (its share of one A tile and one B tile)?
        </>
      </Translated>
    ),
    answer: loadsPerThreadPerTilePair(BOOK, 'float4'),
    explain: (
      <Translated
        vi={
          <p>
            Mỗi ô chứa 1024 float, 4 cho mỗi luồng: một <code>float4</code> từ ô A và một từ ô B, tổng cộng{' '}
            {loadsPerThreadPerTilePair(BOOK, 'float4')}, thay vì {loadsPerThreadPerTilePair(BOOK, 'scalar')} lệnh nạp vô
            hướng. Mỗi warp khi đó chuyển 512 byte mỗi lệnh, 16 sector. Điểm vướng là căn chỉnh: địa chỉ phải là bội số của 16
            byte, nên K và N phải là bội số của 4, nếu không kernel cần một nhánh vô hướng.
          </p>
        }
      >
        <p>
          Each tile holds 1024 floats, 4 per thread: one <code>float4</code> from the A tile and one from the B tile,{' '}
          {loadsPerThreadPerTilePair(BOOK, 'float4')} in all, instead of {loadsPerThreadPerTilePair(BOOK, 'scalar')} scalar
          loads. Each warp then moves 512 bytes per instruction, 16 sectors. The catch is alignment: the address must be a
          multiple of 16 bytes, so K and N must be multiples of 4, or the kernel needs a scalar path.
        </p>
      </Translated>
    ),
    source: 'ch. 15 Exercise 1, p. 370 (our sketch, notes/ch15.md); alignment: notes/today/infra-facts.md A6, ch15-facts.md E1',
    vi: {
      source: 'ch. 15 Bài tập 1, p. 370 (phác thảo của chúng tôi, notes/ch15.md); căn chỉnh: notes/today/infra-facts.md A6, ch15-facts.md E1',
    },
  },
};

// ---------------------------------------------------------------- the section
export function S153() {
  const t17 = owns(1024, 1024, 2, 1, 17);
  const t255 = owns(1024, 1024, 2, 1, 255);
  return (
    <>
      <Translated
        vi={
          <p>
            Mục 15.2 kết thúc bằng một con số: một khối sở hữu ô 128 × 128 của C thực hiện 32 FLOP trên mỗi byte nó nạp, gấp
            bốn lần một ô 32 × 32. Rắc rối là một khối không thể có 16,384 luồng. Khối ở đây có 256, nên mỗi luồng tính 64 đầu
            ra, và mỗi luồng nạp vài phần tử của mỗi ô đầu vào thay vì một. Đó là <b>gộp việc cho luồng (thread coarsening)</b>,
            kỹ thuật của chương 6 và của stencil ở chương 8, và thứ nó loại bỏ cũng giống như ở đó: các khối khác nhau nạp cùng
            những giá trị đầu vào. Bốn khối mà mỗi khối tính một phần rộng 32 của một dải hàng đều nạp cùng ô A; một khối tính
            phần rộng 128 chỉ nạp nó một lần.
          </p>
        }
      >
        <p>
          Section 15.2 ended with a number: a block that owns a 128 × 128 tile of C does 32 FLOP per byte it loads, four
          times what a 32 × 32 tile does. The trouble is that a block cannot have 16,384 threads. The block here has 256,
          so each thread computes 64 outputs, and each thread loads several elements of every input tile instead of one.
          That is <b>thread coarsening</b>, the technique of chapter 6 and of the stencil in chapter 8, and what it removes is
          the same as there: different blocks loading the same input values. Four blocks that each computed a 32-wide piece
          of a row band all loaded the same A tile; one block that computes the 128-wide piece loads it once.
        </p>
      </Translated>
      <Source section="15.3" pages="352" />

      <h2>
        <Translated vi="Một thứ bậc các ô">A hierarchy of tiles</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách đặt tên cho các tầng. <b>Ô đầu ra cấp khối</b> là phần một khối tính: bM × bN = 128 × 128 ở đây.{' '}
            <b>Ô đầu ra cấp luồng</b> là phần của một luồng trong đó: tM × tN = 8 × 8. Ở trên và ở giữa các tầng này, cách chia
            tương tự có thể làm cho một warp (mục 15.5 dùng tầng đó), cho cả một GPU, và cho các node của một cụm máy. Mục này
            chỉ dùng khối và luồng.
          </p>
        }
      >
        <p>
          The book names the levels. The <b>block-level output tile</b> is what one block computes: bM × bN = 128 × 128
          here. The <b>thread-level output tile</b> is one thread's share of it: tM × tN = 8 × 8. Above and between these,
          the same split can be made for a warp (section 15.5 uses that level), for a whole GPU, and for the nodes of a
          cluster. This section uses only the block and the thread.
        </p>
      </Translated>
      <Source section="15.3" pages="352" figs="Fig. 15.2" />
      <Translated
        vi={
          <p>
            Kernel suy ra từ đó. Mỗi khối tìm gốc của ô của nó trong C, mỗi luồng tìm mảng của nó bên trong ô đó, đặt 64 bộ
            tích lũy về 0, rồi khối đi dọc chiều K theo bước bK = 8: nạp một ô A (128 × 8) và một ô B (8 × 128) vào bộ nhớ chia
            sẻ (shared memory), chờ, để mọi luồng cộng phần của mảng mình, lại chờ. Cuối cùng mỗi luồng ghi 64 giá trị của nó.
            Bốn hàm trợ giúp nhỏ làm việc này: <code>clear</code>, <code>loadTile</code>, <code>mm</code> và{' '}
            <code>writeTile</code>.
          </p>
        }
      >
        <p>
          The kernel follows from that. Each block finds its tile's origin in C, each thread finds its patch inside it,
          zeroes 64 accumulators, and then the block walks along the K dimension in steps of bK = 8: load an A tile
          (128 × 8) and a B tile (8 × 128) into shared memory, wait, let every thread add its patch's share, wait again.
          At the end each thread writes its 64 values. Four small helpers do the work: <code>clear</code>,{' '}
          <code>loadTile</code>, <code>mm</code> and <code>writeTile</code>.
        </p>
      </Translated>
      <Source section="15.3" pages="352–354" figs="Figs. 15.3, 15.4" />
      <Listing />
      <Translated vi={<p>Những chỗ tệp của lab khác trang sách đều là cố ý:</p>}>
        <p>Where the lab's file differs from the page, it is on purpose:</p>
      </Translated>
      <ul>
        <Translated
          vi={
            <li>
              Nó được khởi chạy với <b>một khối một chiều 256 luồng</b>. Phần văn nói về một khối 16 × 16 luồng, nhưng mã chỉ
              đọc <code>threadIdx.x</code>; "16 × 16" là cách sắp xếp 256 mảng. Trình khám phá bên dưới cho thấy một lần khởi
              chạy hai chiều làm gì.
            </li>
          }
        >
          <li>
            It is launched with a <b>one-dimensional block of 256 threads</b>. The prose speaks of a block of 16 × 16
            threads, but the code reads only <code>threadIdx.x</code>; "16 × 16" is how the 256 patches are arranged.
            The explorer below shows what a two-dimensional launch does.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Các ô chia sẻ được khai báo một lần ở đầu kernel, nơi thấy rõ mọi vòng lặp tái sử dụng cùng hai mảng. Sách khai
              báo chúng bên trong vòng lặp ô và không bình luận gì.
            </li>
          }
        >
          <li>
            The shared tiles are declared once at the top of the kernel, where it is plain that every iteration reuses
            the same two arrays. The book declares them inside the tile loop and does not comment on it.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Các kích thước theo thứ tự của lab 5.3, (M, K, N), để các chỗ gọi mốc dự án từ chương 5 vẫn chạy. Kernel của sách
              nhận (M, N, K). Kích thước là <code>int</code>, <code>A</code> và <code>B</code> là <code>const</code>, và các
              hàm trợ giúp là <code>static</code>.
            </li>
          }
        >
          <li>
            The sizes come in lab 5.3's order, (M, K, N), so that the milestone's call sites from chapter 5 still work.
            The book's kernel takes (M, N, K). Sizes are <code>int</code>, <code>A</code> and <code>B</code> are{' '}
            <code>const</code>, and the helpers are <code>static</code>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <code>__launch_bounds__(256)</code> là của lab, không phải của sách. Nó quay lại ở bên dưới và ở mục 15.7.
            </li>
          }
        >
          <li>
            <code>__launch_bounds__(256)</code> is the lab's, not the book's. It comes back below and in section 15.7.
          </li>
        </Translated>
      </ul>
      <Aside>
        <Translated
          vi={
            <p>
              Sách bỏ qua phần khởi chạy. Nó suy ra từ cách đánh chỉ số: một lưới ceil(N / 128) × ceil(M / 128) khối, x chạy
              theo các cột của C, và 256 luồng trong một chiều (notes, "Unclear in the PDF" mục 7 và 16).
            </p>
          }
        >
          <p>
            The book leaves the launch out. It follows from the indexing: a grid of ceil(N / 128) × ceil(M / 128) blocks,
            x across the columns of C, and 256 threads in one dimension (notes, "Unclear in the PDF" items 7 and 16).
          </p>
        </Translated>
      </Aside>

      <h2>
        <Translated vi="Ai sở hữu gì">Who owns what</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hai dòng định vị một khối: <code>bRow = blockIdx.y·bM</code> và <code>bCol = blockIdx.x·bN</code>. Bốn dòng nữa
            định vị một luồng bên trong nó. Có bN / tN = 16 mảng theo chiều ngang của một ô khối, nên luồng t nằm ở hàng mảng{' '}
            <code>ty = t / 16</code> và cột mảng <code>tx = t % 16</code>, và mảng của nó bắt đầu ở <code>tRow = ty·tM</code>,{' '}
            <code>tCol = tx·tN</code>. Luồng 0 sở hữu mảng trên cùng bên trái, luồng 15 mảng trên cùng bên phải, luồng 16 mảng
            đầu tiên của hàng thứ hai: cách đánh số của Fig. 15.2.
          </p>
        }
      >
        <p>
          Two lines place a block: <code>bRow = blockIdx.y·bM</code> and <code>bCol = blockIdx.x·bN</code>. Four more
          place a thread inside it. There are bN / tN = 16 patches across a block tile, so thread t sits in patch row{' '}
          <code>ty = t / 16</code> and patch column <code>tx = t % 16</code>, and its patch starts at{' '}
          <code>tRow = ty·tM</code>, <code>tCol = tx·tN</code>. Thread 0 owns the top-left patch, thread 15 the top-right,
          thread 16 the first patch of the second row: the numbering of Fig. 15.2.
        </p>
      </Translated>
      <Source section="15.3" pages="353–354" figs={tr('Figs. 15.2, 15.3 lines 05–13', 'Figs. 15.2, 15.3 dòng 05–13')} />
      <Predict
        id="ch15/15.3/p-thread17"
        prompt={
          <Translated vi={<>Trong khối (x 2, y 1), hàng đầu tiên của C mà luồng 17 sở hữu là hàng nào?</>}>
            <>In block (x 2, y 1), what is the first row of C that thread 17 owns?</>
          </Translated>
        }
        unit={tr('row', 'hàng')}
      >
        {(v) => (
          <Translated
            vi={
              <p>
                <b>Hàng {t17.rows[0]}.</b> Khối bắt đầu ở hàng 1 · 128 = {t17.bRow}; luồng 17 ở hàng mảng 17 / 16 = 1, nên
                thấp hơn 8 hàng. Các cột của nó là {t17.cols[0]} đến {t17.cols[1]}: 2 · 128 = 256, cộng 8 cho cột mảng 1.{' '}
                {v === t17.rows[0]
                  ? 'Bạn đã đúng.'
                  : v === 8
                    ? '8 là vị trí của nó bên trong ô khối; hãy cộng thêm gốc của khối.'
                    : v === 17
                      ? 'Số thứ tự luồng đếm theo mảng, và mỗi mảng cao 8 hàng; 16 mảng làm thành một hàng mảng.'
                      : 'Kiểm tra lại trong trình khám phá: khối (2, 1), luồng 17.'}
              </p>
            }
          >
            <p>
              <b>Row {t17.rows[0]}.</b> The block starts at row 1 · 128 = {t17.bRow}; thread 17 is in patch row 17 / 16 = 1,
              so 8 rows further down. Its columns are {t17.cols[0]} to {t17.cols[1]}: 2 · 128 = 256, plus 8 for patch
              column 1. {v === t17.rows[0] ? 'You had it.' : v === 8 ? '8 is where it sits inside the block tile; add the block\'s origin.' : v === 17 ? 'Thread numbers count patches, and each patch is 8 rows tall; 16 patches make one row of patches.' : 'Check it in the explorer: block (2, 1), thread 17.'}
            </p>
          </Translated>
        )}
      </Predict>
      <Predict
        id="ch15/15.3/p-thread255"
        prompt={
          <Translated vi={<>Cùng khối đó. Cột cuối cùng của C mà luồng 255 sở hữu là cột nào?</>}>
            <>Same block. What is the last column of C that thread 255 owns?</>
          </Translated>
        }
        unit={tr('column', 'cột')}
      >
        {(v) => (
          <Translated
            vi={
              <p>
                <b>Cột {t255.cols[1]}.</b> Luồng 255 là mảng (15, 15), mảng dưới cùng bên phải: cột {t255.cols[0]} đến{' '}
                {t255.cols[1]} và hàng {t255.rows[0]} đến {t255.rows[1]}. 128 cột của khối kết thúc ở 256 + 127 = 383.{' '}
                {v === t255.cols[1] ? 'Bạn đã đúng.' : 'Kiểm tra lại trong trình khám phá.'}
              </p>
            }
          >
            <p>
              <b>Column {t255.cols[1]}.</b> Thread 255 is patch (15, 15), the bottom-right one: columns {t255.cols[0]} to{' '}
              {t255.cols[1]} and rows {t255.rows[0]} to {t255.rows[1]}. The block's 128 columns end at 256 + 127 = 383.{' '}
              {v === t255.cols[1] ? 'You had it.' : 'Check it in the explorer.'}
            </p>
          </Translated>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Trình khám phá tính sáu dòng đó, và các biên của <code>writeTile</code>, cho bất kỳ khối và luồng nào. Kiểm tra hai
            dự đoán của bạn trong đó, rồi thử các mép.
          </p>
        }
      >
        <p>
          The explorer evaluates those six lines, and the bounds of <code>writeTile</code>, for any block and thread. Check
          your two predictions in it, then try the edges.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Chỉ có phép tính chỉ số: không có thời gian, và khung nhìn toàn cục đúng tỉ lệ trừ việc mảng của luồng được vẽ
              rộng ít nhất vài pixel. K không ảnh hưởng đến quyền sở hữu, nên A và B được vẽ rộng K bất kể K là bao nhiêu.
            </p>
          }
        >
          <p>
            Index arithmetic only: no timing, and the global view is to scale except that the thread's patch is drawn at
            least a few pixels wide. K does not affect ownership, so A and B are drawn K wide whatever it is.
          </p>
        </Translated>
      </Simplification>
      <OwnerExplorer />
      <Translated
        vi={
          <p>
            Hai điều cần để ý khi bạn di chuyển luồng. 64 đầu ra của một mảng chỉ cần 8 hàng của <code>A_s</code> và 8 cột của{' '}
            <code>B_s</code>: những lát nhỏ riêng của luồng trong các ô của khối. Và 16 luồng của một hàng mảng đều đọc cùng 8
            hàng của <code>A_s</code>; 16 luồng của một cột mảng đều đọc cùng 8 cột của <code>B_s</code>. Việc đọc chung đó là
            thứ mà một lần nạp duy nhất của khối cho mỗi ô đem lại. Giờ hãy đặt M thành 130 và chọn khối y 1: phần lớn các mảng
            của nó nằm ngoài C, và <code>maxRow</code> chặn các lần ghi của chúng.
          </p>
        }
      >
        <p>
          Two things to notice as you move the thread. A patch's 64 outputs need only 8 rows of <code>A_s</code> and 8
          columns of <code>B_s</code>: the thread's own small slices of the block's tiles. And the 16 threads of one patch
          row all read the same 8 rows of <code>A_s</code>; the 16 threads of a patch column all read the same 8 columns of{' '}
          <code>B_s</code>. That shared reading is what the block's single load of each tile pays for. Now set M to 130 and
          pick block y 1: most of its patches lie outside C, and <code>maxRow</code> stops their writes.
        </p>
      </Translated>
      <Ask q={questionsS153['ch15/15.3/q-2d-block']} />

      <h2>
        <Translated vi="Giữ 64 bộ tích lũy trong thanh ghi">Keeping 64 accumulators in registers</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mỗi luồng khai báo <code>float C_r[8][8]</code>. Một mảng cục bộ như vậy mặc định được đặt trong bộ nhớ, thứ mà{' '}
            <a href={href('ch06', '6.6')}>chương 6</a> gọi là bộ nhớ cục bộ (local memory): riêng của luồng, nhưng về vật lý
            nằm trong bộ nhớ device. Trình biên dịch chỉ chuyển nó vào thanh ghi (register) khi mọi truy cập dùng chỉ số nó
            biết lúc biên dịch. Vì vậy mọi vòng lặp chạm tới <code>C_r</code> đều được unroll hoàn toàn, và các hàm trợ giúp
            chứa những vòng lặp đó bị buộc inline vào kernel, để biên vòng lặp và mảng là hằng số ở nơi unroll diễn ra.
          </p>
        }
      >
        <p>
          Each thread declares <code>float C_r[8][8]</code>. A local array like that is placed in memory by default, which{' '}
          <a href={href('ch06', '6.6')}>chapter 6</a> calls local memory: private to the thread, but physically in device
          memory. The compiler moves it into
          registers only when every access uses an index it knows at compile time. So every loop that touches{' '}
          <code>C_r</code> is fully unrolled, and the helpers that contain those loops are forced inline into the kernel, so
          that the loop bounds and the array are constants where the unrolling happens.
        </p>
      </Translated>
      <Source section="15.3" pages="353–354" figs="Fig. 15.4" />
      <Translated
        vi={
          <p>
            Với 64 bộ tích lũy mỗi luồng, đây là toàn bộ thiết kế: nếu <code>C_r</code> rơi vào bộ nhớ cục bộ, mỗi phép
            nhân-cộng phải đọc và ghi nó ở đó. Không gì trong kết quả cho thấy điều đó; chỉ báo cáo của trình biên dịch cho
            thấy.
          </p>
        }
      >
        <p>
          With 64 accumulators per thread this is the whole design: if <code>C_r</code> lands in local memory, every
          multiply-add has to read and write it there. Nothing in the results would show it; only the compiler's report
          does.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 (notes/today/infra-facts.md A7, B7; ch15-facts.md A4, C3, C4)',
          '2026-11-20 (notes/today/infra-facts.md, mục A7, B7; ch15-facts.md, mục A4, C3, C4)',
        )}
      >
        <Translated
          vi={
            <p>
              CUDA và HIP viết cả hai cách điều khiển giống nhau. NVIDIA công bố rằng <code>#pragma unroll</code> không có số
              đếm sẽ unroll hoàn toàn khi số lần lặp là hằng số (AMD gọi nó là gợi ý), và cả hai đều nói{' '}
              <code>__forceinline__</code> buộc inline. MSL không công bố cái nào. Pragma vòng lặp của Clang và macro{' '}
              <code>METAL_FUNC</code> trong header của Apple (<code>inline __attribute__((__always_inline__))</code>) chạy được
              trong thực tế; không cái nào có trong đặc tả.
            </p>
          }
        >
          <p>
            CUDA and HIP spell both controls the same. NVIDIA documents that <code>#pragma unroll</code> with no count
            unrolls completely when the trip count is constant (AMD calls it a hint), and both say{' '}
            <code>__forceinline__</code> forces inlining. MSL documents neither. The
            Clang loop pragma and Apple's header macro <code>METAL_FUNC</code> (<code>inline __attribute__((__always_inline__))</code>)
            work in practice; neither is in the specification.
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              <code>C_r</code> rốt cuộc nằm ở đâu thì thấy được trên CUDA và HIP trong báo cáo tài nguyên của trình biên dịch (
              <code>--resource-usage</code>) hoặc trong <code>numRegs</code> từ truy vấn thuộc tính hàm; Metal hoàn toàn không
              công bố báo cáo thanh ghi nào.
            </p>
          }
        >
          <p>
            Where <code>C_r</code> ended up is visible on CUDA and HIP in the compiler's resource report (
            <code>--resource-usage</code>) or in <code>numRegs</code> from the function-attributes query; Metal documents
            no register report at all.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS153['ch15/15.3/q-unroll']} />

      <h2>
        <Translated vi="Nạp một ô với ít luồng hơn số phần tử">Loading a tile with fewer threads than elements</Translated>
      </h2>
      <Translated
        vi={
          <p>
            <code>loadTile</code> chép một ô từ bộ nhớ toàn cục (global memory) vào bộ nhớ chia sẻ. Nó nhận điểm bắt đầu của
            ô, <b>chiều đầu (leading dimension)</b> của nó (khoảng cách tính bằng phần tử từ một hàng sang hàng kế tiếp: với
            một ô bên trong ma trận thứ tự hàng (row-major), đó là độ rộng của cả ma trận, K với A và N với B), số hàng và cột
            còn lại bên trong ma trận, và đích đến với chiều đầu riêng của nó.
          </p>
        }
      >
        <p>
          <code>loadTile</code> copies one tile from global memory into shared memory. It takes the tile's start, its{' '}
          <b>leading dimension</b> (the distance in elements from one row to the next: for a tile inside a row-major
          matrix, the width of the whole matrix, K for A and N for B), how many rows and columns remain inside the matrix,
          and the destination with its own leading dimension.
        </p>
      </Translated>
      <Source section="15.3" pages="355" figs={tr('Fig. 15.5, footnote 1', 'Fig. 15.5, chú thích 1')} />
      <Translated
        vi={
          <p>
            Một ô A 128 × 8 có 1024 phần tử và khối có 256 luồng, nên mỗi luồng nạp bốn. Ô được cắt thành các ô con có số phần
            tử bằng số luồng: mỗi ô con <code>rowsPerSubTile = 256 / width</code> hàng. Với ô A là 32 hàng; với ô B 8 × 128, 2
            hàng. Đằng nào cũng là bốn ô con, lần lượt từng cái, và trong mỗi ô con luồng t lấy hàng <code>t / width</code>,
            cột <code>t % width</code> của ô con. Một phần tử nằm ngoài ma trận được ghi là 0.0f, để <code>mm</code> có thể
            chạy mà không cần một phép kiểm tra biên nào.
          </p>
        }
      >
        <p>
          A 128 × 8 A tile has 1024 elements and the block 256 threads, so each thread loads four. The tile is cut into
          sub-tiles of as many elements as there are threads: <code>rowsPerSubTile = 256 / width</code> rows each. For the
          A tile that is 32 rows; for the 8 × 128 B tile, 2 rows. Either way, four sub-tiles, one after the other, and in
          each one thread t takes row <code>t / width</code>, column <code>t % width</code> of the sub-tile. An element
          that lies outside the matrix is written as 0.0f, so that <code>mm</code> can run without a single bounds check.
        </p>
      </Translated>
      <Source section="15.3" pages="355–356" />
      <LabSlice
        fn="loadTile"
        caption={
          <Translated vi={<>Từ lời giải tham chiếu của lab 15.1; số dòng là của tệp.</>}>
            <>From lab 15.1's reference solution; the line numbers are the file's.</>
          </Translated>
        }
      />
      <Ask q={questionsS153['ch15/15.3/q-loadtile-row']} label={tr('Work it out', 'Tính ra')} />
      <Translated
        vi={
          <p>
            Trình chạy từng bước chạy bốn ô con đó cho cả hai ô. Chọn một warp và nhìn dải bên dưới các ô: nó cho thấy các địa
            chỉ toàn cục mà một lệnh của warp đó yêu cầu, dưới dạng trọn các sector 32 byte. Sau đó thử các cấu hình mẫu.
          </p>
        }
      >
        <p>
          The stepper runs those four sub-tiles for either tile. Pick a warp and look at the strip under the tiles: it
          shows the global addresses that warp's one instruction asks for, as whole 32-byte sectors. Then try the presets.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi lần một lệnh warp, đếm theo sector 32 byte của NVIDIA; không có cache, không có thời gian, và không gộp giữa
              các warp. Wavefront của AMD có thể rộng 64 và Apple không công bố kích thước sector, nên trên các nền đó con số là
              một mô hình, không phải một dự đoán.
            </p>
          }
        >
          <p>
            One warp instruction at a time, counted in NVIDIA's 32-byte sectors; no caches, no timing, and no merging
            between warps. AMD wavefronts can be 64 wide and Apple documents no sector size, so on those the count is a
            model, not a prediction.
          </p>
        </Translated>
      </Simplification>
      <LoadTileStepper />
      <Translated
        vi={
          <p>
            Với K và N là bội số của 8, cả hai ô tốn bốn sector mỗi lệnh warp và dùng mọi byte. Chúng khác nhau về hình dạng:
            một warp B đọc một đoạn 128 byte, một warp A bốn đoạn 32 byte trên bốn hàng của A. Với K = 1001, một hàng của A
            không còn bắt đầu ở biên sector, và một warp A chạm tới bảy hoặc tám sector thay vì bốn. Phép kiểm tra biên chỉ tách
            một warp ở nơi một ô vắt qua mép ma trận: hàng hoặc cột khối cuối cùng, hoặc cặp ô cuối cùng khi K không phải bội số
            của 8.
          </p>
        }
      >
        <p>
          With K and N multiples of 8, both tiles cost four sectors per warp instruction and use every byte. They differ
          in shape: a B warp reads one run of 128 bytes, an A warp four runs of 32 bytes in four rows of A. With K = 1001 an
          A row no longer starts on a sector boundary, and an A warp touches seven or eight sectors instead of four. The bounds check only splits a warp where
          a tile crosses the edge of the matrix: the last row or column of blocks, or the last tile pair when K is not a
          multiple of 8.
        </p>
      </Translated>
      <Ask q={questionsS153['ch15/15.3/q-sectors']} />
      <Ask q={questionsS153['ch15/15.3/q-tail']} label={tr('Work it out', 'Tính ra')} />
      <Translated
        vi={
          <p>
            Sách để lại một cải tiến làm Bài tập 1: nạp bốn phần tử bằng một lệnh <code>float4</code>. Đánh dấu "nạp vector"
            trong trình chạy từng bước để thấy cách bố trí: một ô con mỗi ô, và mỗi warp hoặc 16 hàng của A (hai float4 mỗi
            hàng) hoặc trọn một hàng 128 float của B. Lab 15.4 xây nó.
          </p>
        }
      >
        <p>
          The book leaves one improvement as Exercise 1: load four elements with one <code>float4</code> instruction. Tick
          "vector loads" in the stepper to see the layout: one sub-tile per tile, and per warp either 16 rows of A (two
          float4s a row) or one whole 128-float row of B. Lab 15.4 builds it.
        </p>
      </Translated>
      <Source pages="356, 370" figs={tr('Exercise 1', 'Bài tập 1')} />
      <Ask q={questionsS153['ch15/15.3/q-vec-loads']} label={tr('Work it out', 'Tính ra')} />

      <h2>
        <Translated vi="Nhân các ô, và ghi kết quả">Multiplying the tiles, and writing the result</Translated>
      </h2>
      <Translated
        vi={
          <p>
            64 đầu ra của một luồng chỉ cần một lát tM × bK của ô A (8 hàng của nó) và một lát bK × tN của ô B (8 cột của nó).
            Sách gọi chúng là <b>ô đầu vào cấp luồng</b>. Vì chúng là trọn hàng và trọn cột, phép nhân cấp luồng có cùng chiều
            trong với của khối: bK. Kernel truyền cho <code>mm</code> hai con trỏ vào bộ nhớ chia sẻ,{' '}
            <code>&amp;A_s[tRow*bK]</code> và <code>&amp;B_s[tCol]</code>, cùng chiều đầu của chúng, bK và bN; các chỉ số bên
            trong <code>mm</code> là tương đối so với các lát của luồng, nhưng mọi lần đọc vẫn đi tới các mảng của khối trong bộ
            nhớ chia sẻ.
          </p>
        }
      >
        <p>
          A thread's 64 outputs need only a tM × bK slice of the A tile (its 8 rows) and a bK × tN slice of the B tile (its
          8 columns). The book calls these the <b>thread-level input tiles</b>. Because they are whole rows and whole
          columns, the thread-level multiplication has the same inner dimension as the block's: bK. The kernel passes{' '}
          <code>mm</code> two pointers into shared memory, <code>&amp;A_s[tRow*bK]</code> and{' '}
          <code>&amp;B_s[tCol]</code>, and their leading dimensions, bK and bN; indices inside <code>mm</code> are relative
          to the thread's slices, but every read still goes to the block's arrays in shared memory.
        </p>
      </Translated>
      <Source section="15.3" pages="356–357" figs="Fig. 15.6" />
      <LabSlice
        fn="mm"
        caption={
          <Translated vi={<>Phiên bản đầu tiên của <code>mm</code> (Fig. 15.6), từ lab 15.1. Mục 15.4 sắp xếp lại các vòng lặp của nó.</>}>
            <>The first version of <code>mm</code> (Fig. 15.6), from lab 15.1. Section 15.4 reorders its loops.</>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            <code>writeTile</code> ghi 64 giá trị, lại được unroll, bỏ qua mọi phần tử vượt quá <code>maxRow</code> hoặc{' '}
            <code>maxCol</code>. Hãy để ý một lệnh ghi của warp làm gì: luồng t ghi phần tử (row, col) của mảng của nó, và mảng
            của luồng kế bên bắt đầu xa hơn 8 cột. Các địa chỉ của warp cách nhau 8 float. Mục 15.5 nói về điều đó.
          </p>
        }
      >
        <p>
          <code>writeTile</code> stores the 64 values, unrolled again, skipping any element beyond <code>maxRow</code> or{' '}
          <code>maxCol</code>. Note what one store instruction of a warp does: thread t writes element (row, col) of its own
          patch, and its neighbour's patch starts 8 columns further on. The warp's addresses are 8 floats apart. Section
          15.5 is about that.
        </p>
      </Translated>
      <Source section="15.3" pages="357" figs="Fig. 15.7" />
      <LabSlice
        fn="writeTile"
        caption={
          <Translated vi={<>Từ lab 15.1. Kernel tính <code>c</code>, <code>maxRow</code> và <code>maxCol</code> trước lời gọi.</>}>
            <>From lab 15.1. The kernel computes <code>c</code>, <code>maxRow</code> and <code>maxCol</code> before the call.</>
          </Translated>
        }
      />

      <h2>
        <Translated vi="Các thư viện chọn gì">What libraries choose</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Ô khối là tham số đầu tiên của mọi thư viện GEMM. CUTLASS gọi nó là <code>ThreadblockShape</code> và xây chính thứ
            bậc này bên dưới nó. Các GEMM viết cho Metal chọn ô nhỏ hơn của sách: MLX của Apple tạo các hình dạng như 64 × 64 ×
            16 với 64 đến 128 luồng và chọn giữa chúng theo từng thiết bị, còn kernel Metal của llama.cpp dùng ô đầu ra 64 × 32
            với bước K là 32. Apple không công bố gì có thể giải thích lựa chọn đó, nên hãy coi đây là sự thật về thiết kế,
            không phải quy tắc.
          </p>
        }
      >
        <p>
          The block tile is the first parameter of every GEMM library. CUTLASS calls it <code>ThreadblockShape</code> and
          builds exactly this hierarchy beneath it. GEMMs written for Metal pick smaller tiles than the book's: Apple's MLX
          instantiates shapes such as 64 × 64 × 16 with 64 to 128 threads and chooses among them per device, and
          llama.cpp's Metal kernel uses a 64 × 32 output tile with a K step of 32. Apple documents nothing that would explain the
          choice, so take these as design facts, not as rules.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch15-facts.md A12, D2, D3)', '2026-11-20 (notes/today/ch15-facts.md, mục A12, D2, D3)')}>
        <Translated
          vi={
            <p>
              CUTLASS: ô threadblock <code>ThreadblockShape::&#123;kM, kN, kK&#125;</code>, rồi đến ô warp và ô luồng. MLX steel
              GEMM: <code>(bm, bn, bk)</code> được tạo thành (64, 64, 16), (64, 32, 32), (32, 32, 16) và các hình dạng khác,{' '}
              <code>tgp_size = WM·WN·32</code>, chọn theo loại thiết bị. llama.cpp <code>kernel_mul_mm</code>:{' '}
              <code>NR0 = 64</code>, <code>NR1 = 32</code>, <code>NK = 32</code>.
            </p>
          }
        >
          <p>
            CUTLASS: threadblock tile <code>ThreadblockShape::&#123;kM, kN, kK&#125;</code>, then warp and thread tiles.
            MLX steel GEMM: <code>(bm, bn, bk)</code> instantiated as (64, 64, 16), (64, 32, 32), (32, 32, 16) and
            others, <code>tgp_size = WM·WN·32</code>, picked by device class. llama.cpp <code>kernel_mul_mm</code>: <code>NR0 = 64</code>, <code>NR1 = 32</code>,{' '}
            <code>NK = 32</code>.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS153['ch15/15.3/q-decode-blocks']} />
      <Translated
        vi={
          <p>
            Với HIP, cùng tệp <code>.cu</code> là mã nguồn, thông qua header của khóa học, điều này chạy được ở đây vì kernel
            chỉ dùng luồng, khối, bộ nhớ chia sẻ và rào chắn (chưa bản build HIP nào chạy: unverified). Một thuộc tính trong đó
            quan trọng ở HIP hơn ở CUDA: <code>__launch_bounds__(256)</code>, cho trình biên dịch biết khối sẽ không bao giờ lớn
            hơn. Mục 15.7 giải thích vì sao một kernel với 64 bộ tích lũy mỗi luồng cần nó.
          </p>
        }
      >
        <p>
          For HIP the same <code>.cu</code> is the source, through the course's header, which works here because the
          kernel uses only threads, blocks, shared memory and barriers (no HIP build has run yet: unverified). One
          attribute in it matters more there than on CUDA: <code>__launch_bounds__(256)</code>, which tells the compiler the block will never be larger.
          Section 15.7 explains why a kernel with 64 accumulators per thread wants it.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ dùng CUDA. Không có launch bounds, HIP biên dịch cho khối lớn nhất mà thiết bị cho phép, điều này giới
              hạn số thanh ghi mỗi luồng (notes/today/ch15-facts.md B3).
            </p>
          }
        >
          <p>
            The book is CUDA-only. Without launch bounds, HIP compiles for the largest block the device allows, which
            limits registers per thread (notes/today/ch15-facts.md B3).
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Đến lượt bạn">Your turn</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lab 15.1 yêu cầu bạn viết kernel này trong <code>labs/ch15/lab1_coarse/work/</code>, sau một hàm khởi chạy với các
            đối số của lab 5.3, <code>launchGemmCoarse(A, B, C, M, K, N)</code>, và đo hiệu năng (benchmark) nó. Trước khi chạy
            benchmark, hãy chốt một dự đoán trên <a href={href('ch15', 'labs')}>trang lab</a>. Mô hình của mục 15.2 cho ô
            128 × 128 32 FLOP/B, gấp tám lần ô 16 × 16 của lab 5.3. Nhưng mô hình chỉ đếm lưu lượng tới DRAM; trên một GPU mà
            bộ nhớ đệm (cache) đã hấp thụ phần lớn các lần đọc lặp lại của kernel 16 × 16, mức tăng đo được có thể nhỏ hơn
            nhiều.
          </p>
        }
      >
        <p>
          Lab 15.1 has you write this kernel in <code>labs/ch15/lab1_coarse/work/</code>, behind a launch function with
          lab 5.3's arguments, <code>launchGemmCoarse(A, B, C, M, K, N)</code>, and benchmark it. Before you run the
          benchmark, commit to a prediction on the <a href={href('ch15', 'labs')}>labs page</a>. Section 15.2's model
          gives the 128 × 128 tile 32 FLOP/B, eight times what lab 5.3's 16 × 16 tile gets. But the model counts only
          traffic to DRAM; on a GPU whose caches already absorbed much of the 16 × 16 kernel's repeated reads, the measured
          gain can be much smaller.
        </p>
      </Translated>
    </>
  );
}
