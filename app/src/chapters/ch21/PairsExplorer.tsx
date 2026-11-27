import { useEffect, useId, useMemo, useRef, useState, type MouseEvent as RMouseEvent, type PointerEvent as RPointerEvent, type ReactNode } from 'react';
import { Bench } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { num, useReports, type Reports } from '../../lib/reports';
import { SEGMENT_MODELS, type SegmentModelId } from '../../sim/coalesce';
import {
  BOOK_OPS_F4,
  KERNELS,
  KERNEL_ORDER,
  arithmetic,
  f32text,
  mapKernel,
  movedAtom,
  opsCoarsened,
  opsGather,
  seededAtoms,
  writeStatements,
  type Atom,
  type KernelId,
  type MapResult,
  type Ownership,
  type ThreadInfo,
} from '../../sim/dcs';
import {
  acceptMasks,
  acceptedShare,
  binAtoms,
  binCentre,
  binOf,
  blockWalk,
  capacityFor,
  counts,
  diverges,
  makeWorld,
  maxOccupancy,
  nearFar,
  runGrid,
  superRadius,
  worldH,
  worldW,
  type RadiusMode,
} from '../../sim/cutoff';

// "Who writes where": one explorer with three views, on the models in src/sim/dcs.ts and
// src/sim/cutoff.ts. Section 21.2 uses the mapping view (twice: scatter and gather, then the
// coarsened kernels) and the write-statement view; section 21.5 uses the cutoff view.
// Nothing here computes kernel logic of its own: every number comes from the models.

export type ExplorerView = 'mapping' | 'writes' | 'cutoff';

const thousands = (v: number) => v.toLocaleString('en-US');
const pct = (v: number, digits = 0) => `${(v * 100).toFixed(digits)} %`;
const plural = (n: number, one: string, many = one + 's') => `${thousands(n)} ${n === 1 ? one : many}`;
/** Table cells that may wrap (the shared .threads style keeps cells on one line). */
const WRAP = { whiteSpace: 'normal' as const, verticalAlign: 'top' as const };
/** Below this width a drawing scrolls sideways inside the bench instead of shrinking past legibility. */
const MIN_DRAW = 520;

function Field({ label, children }: { label: ReactNode; children: ReactNode }) {
  return (
    <label className="field">
      {label}
      {children}
    </label>
  );
}

// ================================================================ view 1: who writes where

const KERNEL_SHORT: Record<KernelId, string> = {
  get scatter() {
    return tr('scatter (21.5)', 'scatter (21.5)');
  },
  get gather() {
    return tr('gather (21.6)', 'gather (21.6)');
  },
  get printed() {
    return tr('coarsened, as printed (21.8)', 'đã gộp việc, như bản in (21.8)');
  },
  get adjacent() {
    return tr('coarsened, adjacent (21.8 fixed)', 'đã gộp việc, liền kề (21.8 đã sửa)');
  },
  get strided() {
    return tr('coarsened, stride (21.10)', 'đã gộp việc, bước nhảy (21.10)');
  },
};

/**
 * The mapping view splits threads into warps of 32 lanes: NVIDIA's warp width, which the
 * book assumes throughout (ch05-facts B5 gives 64 on AMD Instinct, 32 on Radeon). It affects
 * only the warp and lane labels and the constant-address count; the S212 Simplification says so.
 */
const BOOK_WARP = 32;

const threadName = (t: ThreadInfo) => tr(`block (${t.bx}, ${t.by}), thread (${t.tx}, ${t.ty})`, `khối (${t.bx}, ${t.by}), luồng (${t.tx}, ${t.ty})`);

/** The thread's index arithmetic, with its numbers. */
function indexText(r: MapResult, t: ThreadInfo): string {
  const { kernel, blockX: B, blockY: By, W } = r.config;
  const F = r.config.F;
  const pts = t.points.map((p) => `(${p % W}, ${Math.floor(p / W)})`);
  const list = pts.length > 6 ? tr(`${pts.slice(0, 5).join(', ')}, … ${pts.length} points`, `${pts.slice(0, 5).join(', ')}, … ${pts.length} điểm`) : pts.join(', ');
  const jText = `j = blockIdx.y·blockDim.y + threadIdx.y = ${t.by}·${By} + ${t.ty} = ${t.j}`;
  const out = t.outside
    ? tr(
        ` ${plural(t.outside, 'write')} would fall outside the slice (a bound check stops ${t.outside === 1 ? 'it' : 'them'}).`,
        ` ${thousands(t.outside)} lần ghi sẽ rơi ra ngoài lát (một phép kiểm tra biên chặn ${t.outside === 1 ? 'nó' : 'chúng'}).`,
      )
    : '';
  switch (kernel) {
    case 'scatter':
      return t.atom! < 0
        ? tr(
            `n = (blockIdx.x·blockDim.x + threadIdx.x)·4 = (${t.bx}·${B} + ${t.tx})·4 = ${(t.bx * B + t.tx) * 4}: past the chunk's last atom, so the bound check returns and this thread does nothing.`,
            `n = (blockIdx.x·blockDim.x + threadIdx.x)·4 = (${t.bx}·${B} + ${t.tx})·4 = ${(t.bx * B + t.tx) * 4}: vượt quá nguyên tử cuối cùng của chunk, nên phép kiểm tra biên trả về và luồng này không làm gì.`,
          )
        : tr(
            `n = (blockIdx.x·blockDim.x + threadIdx.x)·4 = (${t.bx}·${B} + ${t.tx})·4 = ${t.atom! * 4}: atom ${t.atom}. It reads that one atom and adds to every one of the ${W * r.config.H} points.`,
            `n = (blockIdx.x·blockDim.x + threadIdx.x)·4 = (${t.bx}·${B} + ${t.tx})·4 = ${t.atom! * 4}: nguyên tử ${t.atom}. Nó đọc đúng nguyên tử đó và cộng vào từng điểm trong ${W * r.config.H} điểm.`,
          );
    case 'gather':
      return t.points.length
        ? tr(
            `i = blockIdx.x·blockDim.x + threadIdx.x = ${t.bx}·${B} + ${t.tx} = ${t.i}; ${jText}. It reads every atom and writes point ${list} once.`,
            `i = blockIdx.x·blockDim.x + threadIdx.x = ${t.bx}·${B} + ${t.tx} = ${t.i}; ${jText}. Nó đọc mọi nguyên tử và ghi điểm ${list} một lần.`,
          )
        : tr(
            `i = ${t.bx}·${B} + ${t.tx} = ${t.i}, j = ${t.j}: outside the slice. The bound check returns (the book pads the slice instead, Fig. 21.9).`,
            `i = ${t.bx}·${B} + ${t.tx} = ${t.i}, j = ${t.j}: nằm ngoài lát. Phép kiểm tra biên trả về (sách thì đệm lát thay vào đó, Fig. 21.9).`,
          );
    case 'printed':
      return tr(
        `i = blockIdx.x·blockDim.x·${F} + threadIdx.x = ${t.bx}·${B}·${F} + ${t.tx} = ${t.i} (threadIdx.x not scaled); ${jText}. It writes i, i+1, …: ${list || 'nothing'}.${out}`,
        `i = blockIdx.x·blockDim.x·${F} + threadIdx.x = ${t.bx}·${B}·${F} + ${t.tx} = ${t.i} (threadIdx.x không được nhân hệ số); ${jText}. Nó ghi i, i+1, …: ${list || 'không gì cả'}.${out}`,
      );
    case 'adjacent':
      return tr(
        `i = (blockIdx.x·blockDim.x + threadIdx.x)·${F} = (${t.bx}·${B} + ${t.tx})·${F} = ${t.i}; ${jText}. It writes i to i+${F - 1}: ${list || 'nothing'}.${out}`,
        `i = (blockIdx.x·blockDim.x + threadIdx.x)·${F} = (${t.bx}·${B} + ${t.tx})·${F} = ${t.i}; ${jText}. Nó ghi từ i đến i+${F - 1}: ${list || 'không gì cả'}.${out}`,
      );
    case 'strided':
      return tr(
        `i = blockIdx.x·blockDim.x·${F} + threadIdx.x = ${t.bx}·${B}·${F} + ${t.tx} = ${t.i}; ${jText}. It writes i, i+${B}, …: ${list || 'nothing'}.${out}`,
        `i = blockIdx.x·blockDim.x·${F} + threadIdx.x = ${t.bx}·${B}·${F} + ${t.tx} = ${t.i}; ${jText}. Nó ghi i, i+${B}, …: ${list || 'không gì cả'}.${out}`,
      );
  }
}

function MappingView({ initialKernel, kernels }: { initialKernel: KernelId; kernels: KernelId[] }) {
  const hatch = useId();
  const svgRef = useRef<SVGSVGElement>(null);
  const [kernel, setKernel] = useState<KernelId>(initialKernel);
  const [W, setW] = useState(16);
  const [H, setH] = useState(4);
  const [nAtoms, setNAtoms] = useState(5);
  const [seed, setSeed] = useState(1);
  const [moved, setMoved] = useState<Record<number, Atom>>({});
  const [blockX, setBlockX] = useState(4);
  const [blockY, setBlockY] = useState(2);
  const [F, setF] = useState(4);
  const [atomic, setAtomic] = useState(true);
  const [point, setPoint] = useState(3);
  const [thread, setThread] = useState<number | null>(null);
  const [drag, setDrag] = useState<{ k: number; moved: boolean } | null>(null);

  const atoms = useMemo(() => seededAtoms(nAtoms, W, H, seed).map((a, k) => moved[k] ?? a), [nAtoms, W, H, seed, moved]);
  const res = useMemo(() => mapKernel({ kernel, W, H, atoms, blockX, blockY, F, atomic, warpWidth: BOOK_WARP }), [kernel, W, H, atoms, blockX, blockY, F, atomic]);
  const info = KERNELS[kernel];
  const p = Math.min(point, W * H - 1);
  const writers = res.writers[p];
  const tSel = thread != null && thread < res.threads.length ? res.threads[thread] : null;
  const reset = () => {
    setMoved({});
    setThread(null);
  };

  // Geometry: one cell per point, atoms drawn at their (x, y), labels outside the slice.
  const cs = Math.max(14, Math.min(34, Math.floor(640 / W)));
  const ox = 26;
  const oy = 22;
  const VW = ox + W * cs + 8;
  const VH = oy + H * cs + 8;
  const toSvg = (e: RPointerEvent) => {
    const svg = svgRef.current;
    const m = svg?.getScreenCTM();
    if (!svg || !m) return null;
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse());
    return { x: (pt.x - ox) / cs - 0.5, y: (pt.y - oy) / cs - 0.5 };
  };
  const onMove = (e: RPointerEvent) => {
    if (!drag) return;
    const q = toSvg(e);
    if (!q) return;
    const x = Math.max(-0.4, Math.min(W - 0.6, q.x));
    const y = Math.max(-0.4, Math.min(H - 0.6, q.y));
    setMoved((m) => ({ ...m, [drag.k]: movedAtom(atoms[drag.k], x, y) }));
    if (!drag.moved) setDrag({ ...drag, moved: true });
  };
  const pickPoint = (q: number) => {
    setPoint(q);
    setThread(res.writers[q][0] ?? null);
  };
  const selectedAtom = kernel === 'scatter' && tSel ? tSel.atom : undefined;
  const shade = (q: number) => {
    const w = res.writers[q];
    if (kernel === 'scatter' || w.length !== 1) return 0;
    return 0.05 + 0.075 * (res.threads[w[0]].id % 4);
  };
  const maxW = Math.max(...res.writers.map((w) => w.length));
  const minW = Math.min(...res.writers.map((w) => w.length));
  const tolerance = 1e-5 * atoms.reduce((s, a) => s + Math.abs(a.q / Math.hypot(p % W - a.x, Math.floor(p / W) - a.y, a.z)), 0);
  const diff = res.potentials[p] - res.direct[p];

  return (
    <div data-view="mapping" data-kernel={kernel} data-verdict={res.verdict} data-conflicts={res.conflicts}
      data-unwritten={res.unwritten} data-const={res.constAddrsPerWarp} data-writers-max={maxW} data-writers-min={minW}
      data-correct={res.correct ? 'yes' : 'no'}>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label={tr('Kernel', 'Kernel')} style={{ flexWrap: 'wrap', rowGap: 4 }}>
          {kernels.map((k) => (
            <button key={k} aria-pressed={k === kernel} onClick={() => { setKernel(k); setThread(null); }} title={tr(`${KERNELS[k].label} (${KERNELS[k].fig})`, { scatter: 'scatter (Fig. 21.5)', gather: 'gather (Fig. 21.6)', printed: 'đã gộp việc, dòng 05 như bản in (Fig. 21.8 như bản in)', adjacent: 'đã gộp việc, liền kề (đã sửa) (Fig. 21.8 đã sửa)', strided: 'đã gộp việc, bước nhảy blockDim.x (Fig. 21.10)' }[k])}>
              {KERNEL_SHORT[k]}
            </button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <Field label={tr(`Slice width ${W}`, `Bề rộng lát ${W}`)}>
          <input type="range" min={8} max={40} value={W} aria-label={tr('slice width', 'bề rộng lát')} onChange={(e) => { setW(+e.target.value); reset(); }} />
        </Field>
        <Field label={tr(`Height ${H}`, `Chiều cao ${H}`)}>
          <input type="range" min={2} max={8} value={H} aria-label={tr('slice height', 'chiều cao lát')} onChange={(e) => { setH(+e.target.value); reset(); }} />
        </Field>
        <Field label={tr(`Atoms ${nAtoms}`, `Nguyên tử ${nAtoms}`)}>
          <input type="range" min={1} max={12} value={nAtoms} aria-label={tr('atoms', 'nguyên tử')} onChange={(e) => { setNAtoms(+e.target.value); reset(); }} />
        </Field>
        <button onClick={() => { setSeed(seed + 1); reset(); }}>
          <Translated vi="Nguyên tử ngẫu nhiên mới">New random atoms</Translated>
        </button>
        <Field label="blockDim.x">
          <select value={blockX} onChange={(e) => { setBlockX(+e.target.value); setThread(null); }}>
            {[4, 8, 16, 32].map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </Field>
        {kernel !== 'scatter' ? (
          <Field label="blockDim.y">
            <select value={blockY} onChange={(e) => { setBlockY(+e.target.value); setThread(null); }}>
              {[1, 2, 4].map((b) => <option key={b} value={b}>{b}</option>)}
            </select>
          </Field>
        ) : null}
        {info.coarsened ? (
          <Field label={tr('Coarsening factor', 'Hệ số gộp việc')}>
            <select value={F} onChange={(e) => { setF(+e.target.value); setThread(null); }}>
              {[2, 4, 8].map((b) => <option key={b} value={b}>{b}</option>)}
            </select>
          </Field>
        ) : null}
        {kernel === 'scatter' ? (
          <label className="check">
            <input type="checkbox" checked={!atomic} onChange={(e) => setAtomic(!e.target.checked)} />{' '}
            <Translated vi="+= thường thay cho atomicAdd">plain += instead of atomicAdd</Translated>
          </label>
        ) : null}
      </div>

      <p className="ui" style={{ margin: '0 0 0.6rem' }} data-badges="">
        {res.verdict === 'one-writer' ? <span className="chip ok"><Translated vi="mỗi điểm một người ghi: không cần phép toán nguyên tử">one writer per point: no atomic needed</Translated></span>
          : res.verdict === 'atomic' ? <span className="chip" style={{ border: '1px solid var(--ink)' }}><Translated vi={<>cần atomicAdd: mỗi điểm có {maxW} người ghi</>}>atomicAdd needed: every point has {maxW} writers</Translated></span>
          : <span className="chip fail"><Translated vi={<>data race: {thousands(res.conflicts)} điểm có nhiều người ghi với một phép += thường</>}>data race: {plural(res.conflicts, 'point')} with several writers and a plain +=</Translated></span>}{' '}
        {res.unwritten ? <span className="chip fail"><Translated vi={<>{thousands(res.unwritten)} điểm không bao giờ được ghi</>}>{plural(res.unwritten, 'point')} never written</Translated></span> : null}{' '}
        {res.correct ? <span className="chip ok"><Translated vi="kết quả bằng tổng trực tiếp">result equals the direct sum</Translated></span> : <span className="chip fail"><Translated vi={<>kết quả sai tại {thousands(res.wrong)} điểm</>}>result wrong at {plural(res.wrong, 'point')}</Translated></span>}
      </p>

      <div style={{ overflowX: 'auto' }}>
        <svg ref={svgRef} viewBox={`0 0 ${VW} ${VH}`} role="img" aria-label={tr('Grid points coloured by the thread that writes them, with the atoms above', 'Các điểm lưới được tô màu theo luồng ghi chúng, với các nguyên tử ở trên')}
          style={{ width: '100%', maxWidth: VW * 1.2, minWidth: Math.min(MIN_DRAW, VW), display: 'block', touchAction: 'none' }}
          onPointerMove={onMove} onPointerUp={() => setDrag(null)} onPointerCancel={() => setDrag(null)}>
          <defs>
            <pattern id={hatch} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1="0" y1="0" x2="0" y2="6" stroke="var(--fail)" strokeWidth="1.4" />
            </pattern>
          </defs>
          {Array.from({ length: W }, (_, i) => (i % (W > 24 ? 4 : 2) === 0 ? (
            <text key={i} x={ox + i * cs + cs / 2} y={oy - 7} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{i}</text>
          ) : null))}
          {Array.from({ length: H }, (_, j) => (
            <text key={j} x={ox - 6} y={oy + j * cs + cs / 2 + 3} fontSize="9" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>{j}</text>
          ))}
          {res.writers.map((w, q) => {
            const i = q % W;
            const j = Math.floor(q / W);
            const n = w.length;
            const race = n > 1 && res.verdict === 'race';
            const mine = tSel && kernel !== 'scatter' && tSel.points.includes(q);
            return (
              <g key={q} data-point={q} data-writers={n} onClick={() => pickPoint(q)} style={{ cursor: 'pointer' }}>
                <rect x={ox + i * cs + 1} y={oy + j * cs + 1} width={cs - 2} height={cs - 2} rx="2"
                  fill={n === 0 ? `url(#${hatch})` : race ? 'var(--fail-wash)' : 'var(--ink)'}
                  fillOpacity={n === 0 || race ? 1 : shade(q)}
                  stroke={race ? 'var(--fail)' : 'var(--rule)'} strokeWidth={race ? 1.4 : 1} />
                {mine ? <rect x={ox + i * cs + 3} y={oy + j * cs + 3} width={cs - 6} height={cs - 6} rx="2" fill="none" stroke="var(--ink)" strokeDasharray="3 2" /> : null}
                {cs >= 18 && n !== 1 ? (
                  <text x={ox + i * cs + cs / 2} y={oy + j * cs + cs / 2 + 3.5} fontSize="9.5" textAnchor="middle"
                    style={{ fill: race || n === 0 ? 'var(--fail)' : 'var(--ink-2)', fontWeight: race ? 650 : 400 }}>×{n}</text>
                ) : null}
                {cs >= 22 && n === 1 && kernel !== 'scatter' ? (
                  <text x={ox + i * cs + cs / 2} y={oy + j * cs + cs / 2 + 3.5} fontSize="8.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
                    t{res.threads[w[0]].tx}
                  </text>
                ) : null}
              </g>
            );
          })}
          <rect x={ox + (p % W) * cs} y={oy + Math.floor(p / W) * cs} width={cs} height={cs} fill="none" stroke="var(--ink)" strokeWidth="2.4" pointerEvents="none" /* i18n-ignore: SVG attribute values */ />
          {atoms.map((a, k) => (
            <g key={k} data-atom={k}
              onPointerDown={(e) => { e.stopPropagation(); (e.currentTarget.ownerSVGElement ?? e.currentTarget).setPointerCapture?.(e.pointerId); setDrag({ k, moved: false }); }}
              onPointerUp={() => {
                if (drag && !drag.moved && kernel === 'scatter') setThread(res.threads.find((t) => t.atom === k)?.id ?? null);
                setDrag(null);
              }}
              style={{ cursor: 'grab' }}>
              <circle cx={ox + (a.x + 0.5) * cs} cy={oy + (a.y + 0.5) * cs} r={Math.max(5, cs * 0.3)}
                fill="var(--const)" fillOpacity={selectedAtom === k ? 1 : 0.78}
                stroke={selectedAtom === k ? 'var(--ink)' : 'var(--page)'} strokeWidth={selectedAtom === k ? 2.2 : 1.2} />
              <text x={ox + (a.x + 0.5) * cs} y={oy + (a.y + 0.5) * cs + 3.2} fontSize="9" textAnchor="middle" style={{ fill: 'var(--page)', pointerEvents: 'none' }}>
                {k}
              </text>
            </g>
          ))}
        </svg>
      </div>

      <div className="cols" style={{ marginTop: '0.8rem' }}>
        <div className="ui" style={{ fontSize: '0.86rem' }}>
          <table className="threads" aria-label={tr('What this mapping does', 'Cách ánh xạ này làm gì')}>
            <tbody>
              <tr>
                <td style={WRAP}><Translated vi="khởi chạy">launch</Translated></td>
                <td style={WRAP}>
                  <Translated vi={<>{res.launch.gridX} × {res.launch.gridY} khối {res.launch.blockX} × {res.launch.blockY}: {thousands(res.launch.threads)} luồng{res.idle ? `, ${res.idle} luồng không có việc gì` : ''}</>}>
                    {res.launch.gridX} × {res.launch.gridY} blocks of {res.launch.blockX} × {res.launch.blockY}: {plural(res.launch.threads, 'thread')}{res.idle ? `, ${res.idle} with nothing to do` : ''}
                  </Translated>
                </td>
              </tr>
              <tr><td style={WRAP}><Translated vi="số người ghi mỗi điểm">writers per point</Translated></td><td style={WRAP} data-readout="writers">{minW === maxW ? minW : tr(`${minW} to ${maxW}`, `${minW} đến ${maxW}`)}</td></tr>
              <tr>
                <td style={WRAP}><Translated vi={<>số địa chỉ hằng mỗi lệnh của một warp {BOOK_WARP} lane</>}>constant addresses per instruction of a {BOOK_WARP}-lane warp</Translated></td>
                <td style={WRAP} data-readout="const">{res.constAddrsPerWarp} {kernel === 'scatter' ? tr('(one per lane: each thread reads its own atom)', '(mỗi lane một địa chỉ: mỗi luồng đọc nguyên tử của riêng nó)') : tr('(every lane reads the same atom)', '(mọi lane đọc cùng một nguyên tử)')}</td>
              </tr>
              <tr><td style={WRAP}><Translated vi="số cặp mỗi luồng có việc">pairs per busy thread</Translated></td><td style={WRAP}>{thousands(res.pairsPerThread)}</td></tr>
            </tbody>
          </table>
        </div>
        <div className="ui" style={{ fontSize: '0.86rem' }} data-selected-point={p}>
          <Translated
            vi={
              <p style={{ margin: '0 0 0.4rem' }}>
                <b>Điểm ({p % W}, {Math.floor(p / W)})</b>: {writers.length === 0 ? 'không luồng nào ghi nó' : `được ${thousands(writers.length)} luồng ghi`}.{' '}
                Kết quả của kernel <span className="mono" data-readout="result">{f32text(res.potentials[p])}</span>; tổng trực tiếp float64{' '}
                <span className="mono" data-readout="direct">{res.direct[p].toPrecision(9)}</span>
                {Math.abs(diff) <= tolerance ? <> (chênh lệch {diff.toExponential(1)}: làm tròn float32)</> : <b className="t-fail"> (sai)</b>}.
              </p>
            }
          >
            <p style={{ margin: '0 0 0.4rem' }}>
              <b>Point ({p % W}, {Math.floor(p / W)})</b>: {writers.length === 0 ? 'no thread writes it' : `written by ${plural(writers.length, 'thread')}`}.{' '}
              Kernel's result <span className="mono" data-readout="result">{f32text(res.potentials[p])}</span>; float64 direct sum{' '}
              <span className="mono" data-readout="direct">{res.direct[p].toPrecision(9)}</span>
              {Math.abs(diff) <= tolerance ? <> (difference {diff.toExponential(1)}: float32 rounding)</> : <b className="t-fail"> (wrong)</b>}.
            </p>
          </Translated>
          {writers.length ? (
            <div className="row" style={{ gap: '0.3rem', marginBottom: '0.4rem' }}>
              {writers.slice(0, 12).map((id) => (
                <button key={id} className="quiet" aria-pressed={thread === id} onClick={() => setThread(id)} style={{ fontSize: '0.78rem', padding: '0.15rem 0.45rem' }}>
                  {kernel === 'scatter' ? tr(`atom ${res.threads[id].atom}`, `nguyên tử ${res.threads[id].atom}`) : `(${res.threads[id].bx},${res.threads[id].by}|${res.threads[id].tx},${res.threads[id].ty})`}
                </button>
              ))}
              {writers.length > 12 ? <span className="muted">…</span> : null}
            </div>
          ) : null}
          <p className="narrate" style={{ margin: 0, minHeight: '2.6em' }} data-thread-text="">
            {tSel ? <><b>{threadName(tSel)}</b> <Translated vi={<>(thuộc warp {tSel.warp}, lane {tSel.lane}):</>}>(warp {tSel.warp}, lane {tSel.lane}):</Translated> {indexText(res, tSel)}</> : kernel === 'scatter'
              ? tr('Click an atom to see its thread, or a point to list the threads that write it.', 'Nhấp vào một nguyên tử để xem luồng của nó, hoặc vào một điểm để liệt kê các luồng ghi nó.')
              : tr('Click a point to list the threads that write it, then pick one to see its index arithmetic.', 'Nhấp vào một điểm để liệt kê các luồng ghi nó, rồi chọn một luồng để xem phép tính chỉ số của nó.')}
          </p>
        </div>
      </div>
      {info.coarsened ? <OpsPanel F0={F} /> : null}
    </div>
  );
}

/** Operations per atom: F threads of Fig. 21.6 against one coarsened thread, counted from the listings. */
function OpsPanel({ F0 }: { F0: number }) {
  const [F, setF] = useState(F0);
  useEffect(() => setF(F0), [F0]);
  const g = opsGather(F);
  const c = opsCoarsened(F);
  const at4 = F === 4;
  const rows: { name: string; g: ReactNode; c: ReactNode; bg: ReactNode; bc: ReactNode; key: string }[] = [
    { key: 'loads', name: tr('constant-memory loads', 'lần nạp bộ nhớ hằng'), g: g.loads, c: c.loads, bg: BOOK_OPS_F4.gather.loads, bc: BOOK_OPS_F4.coarsened.loads },
    { key: 'sub', name: tr('subtractions', 'phép trừ'), g: g.sub, c: c.sub, bg: BOOK_OPS_F4.gather.sub, bc: BOOK_OPS_F4.coarsened.sub },
    { key: 'add', name: tr('additions', 'phép cộng'), g: g.add, c: c.add, bg: BOOK_OPS_F4.gather.add, bc: BOOK_OPS_F4.coarsened.add },
    { key: 'mul', name: tr('multiplications', 'phép nhân'), g: g.mul, c: <>{c.mul} <span className="muted"><Translated vi={<>({c.mulUnhoisted} theo đúng mã viết)</>}>({c.mulUnhoisted} as written)</Translated></span></>, bg: BOOK_OPS_F4.gather.mul, bc: BOOK_OPS_F4.coarsened.mul },
    { key: 'div', name: tr('divisions', 'phép chia'), g: g.div, c: c.div, bg: BOOK_OPS_F4.gather.div, bc: BOOK_OPS_F4.coarsened.div },
    { key: 'sqrt', name: tr('square roots', 'căn bậc hai'), g: g.sqrt, c: c.sqrt, bg: tr('not counted', 'không đếm'), bc: tr('not counted', 'không đếm') },
    { key: 'branch', name: tr('loop branches', 'nhánh vòng lặp'), g: g.branch, c: c.branch, bg: '–', bc: '–' },
    { key: 'total', name: tr('arithmetic, without roots', 'phép tính, không kể căn'), g: arithmetic(g), c: <>{arithmetic(c, true)} <span className="muted"><Translated vi={<>({arithmetic(c)} theo đúng mã viết)</>}>({arithmetic(c)} as written)</Translated></span></>, bg: BOOK_OPS_F4.gather.total, bc: BOOK_OPS_F4.coarsened.total },
  ];
  return (
    <div className="panel" style={{ marginTop: '1rem' }} data-ops={F}>
      <div className="row" style={{ marginBottom: '0.5rem' }}>
        <b className="ui">
          <Translated vi={<>Khối lượng việc mỗi nguyên tử cho {F} điểm</>}>Work per atom for {F} {F === 1 ? 'point' : 'points'}</Translated>
        </b>
        <Field label={`F = ${F}`}>
          <input type="range" min={1} max={8} value={F} aria-label={tr('points per coarsened thread', 'số điểm mỗi luồng đã gộp việc')} onChange={(e) => setF(+e.target.value)} />
        </Field>
      </div>
      <table className="data" style={{ margin: 0 }}>
        <thead>
          <tr>
            <th>
              <Translated vi="mỗi nguyên tử">per atom</Translated>
            </th>
            <th className="num">
              <Translated vi={<>{F} luồng của Fig. 21.6</>}>{F} threads of Fig. 21.6</Translated>
            </th>
            <th className="num">
              <Translated vi="một luồng đã gộp việc">one coarsened thread</Translated>
            </th>
            <th className="num" style={{ opacity: at4 ? 1 : 0.5 }}>
              <Translated vi="sách, Fig. 21.6 ×4">book, Fig. 21.6 ×4</Translated>
            </th>
            <th className="num" style={{ opacity: at4 ? 1 : 0.5 }}>
              <Translated vi="sách, Fig. 21.8">book, Fig. 21.8</Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key} data-op={r.key}>
              <td>{r.name}</td>
              <td className="num" data-g="">{r.g}</td>
              <td className="num" data-c="">{r.c}</td>
              <td className="num" style={{ opacity: at4 ? 1 : 0.5 }}>{r.bg}</td>
              <td className="num" style={{ opacity: at4 ? 1 : 0.5 }}>{r.bc}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <Translated
        vi={
          <p className="ui muted" style={{ margin: '0.5rem 0 0', fontSize: '0.8rem' }}>
            Đếm từ các đoạn mã: "hoisted" (đưa ra ngoài vòng lặp) coi các tích m·gridspacing là được tính một lần, bên ngoài vòng
            lặp nguyên tử. Các cột của sách là các con số in trong sách cho bốn điểm (p. 521); chúng bị làm mờ khi F khác 4.
          </p>
        }
      >
        <p className="ui muted" style={{ margin: '0.5rem 0 0', fontSize: '0.8rem' }}>
          Counted from the listings: "hoisted" treats the m·gridspacing products as computed once, outside the atom loop.
          The book's columns are its printed figures for four points (p. 521); they are greyed when F is not 4.
        </p>
      </Translated>
    </div>
  );
}

// ================================================================ view 2: write statements

/** The SIMD width a lab recorded on this Mac: lab 21.1's group_width if it has run, else any lab's. */
export function appleWidth(reports: Reports): { width: number; source: string } | null {
  const run = reports.metal;
  if (!run) return null;
  const rows = run.results.flatMap((r) => r.bench.map((b) => ({ lab: r.lab, b })));
  const mine = rows.find((x) => x.lab === 'ch21/lab1_dcs' && num(x.b.group_width));
  if (mine) return { width: num(mine.b.group_width)!, source: tr('lab 21.1 on this Mac', 'lab 21.1 trên máy Mac này') };
  for (const x of rows) {
    const w = num(x.b.simd_width) ?? (x.b.name === 'simd_width' ? num(x.b.width) : undefined) ?? num(x.b.threadExecutionWidth);
    if (w)
      return {
        width: w,
        source: tr(
          `lab ${x.lab.replace(/^ch0?(\d+)\/lab(\d+).*/, '$1.$2')} on this Mac (lab 21.1 has not run yet)`,
          `lab ${x.lab.replace(/^ch0?(\d+)\/lab(\d+).*/, '$1.$2')} trên máy Mac này (lab 21.1 chưa chạy)`,
        ),
      };
  }
  return null;
}

type Group = '32' | '64' | 'apple';
const OWNERSHIP: { id: Ownership; label: string }[] = [
  {
    id: 'adjacent',
    get label() {
      return tr('adjacent (Fig. 21.8, corrected)', 'liền kề (Fig. 21.8, đã sửa)');
    },
  },
  {
    id: 'strided',
    get label() {
      return tr('blockDim.x stride (Fig. 21.10)', 'bước nhảy blockDim.x (Fig. 21.10)');
    },
  },
  {
    id: 'float4',
    get label() {
      return tr('one float4 store (Exercise 4)', 'một lần lưu float4 (Bài tập 4)');
    },
  },
];

function WritesView() {
  const { reports } = useReports();
  const apple = appleWidth(reports);
  const [own, setOwn] = useState<Ownership>('adjacent');
  const [group, setGroup] = useState<Group>('32');
  const [appleChoice, setAppleChoice] = useState(32);
  const [blockX, setBlockX] = useState(128);
  const [blockY, setBlockY] = useState(1);
  const [seg, setSeg] = useState<SegmentModelId>('nvidia');
  const [appleSeg, setAppleSeg] = useState(32);
  const [F, setF] = useState(4);
  const [m, setM] = useState(0);
  const [playing, setPlaying] = useState(false);

  const warpWidth = group === 'apple' ? (apple?.width ?? appleChoice) : Number(group);
  const segmentBytes = SEGMENT_MODELS[seg].segmentBytes ?? appleSeg;
  const unit = seg === 'nvidia' ? 'sector' : seg === 'metal' ? 'segment' : 'line'; // i18n-ignore: unit keys; the Vietnamese readouts map them
  const Fe = own === 'float4' && F % 4 !== 0 ? 4 : F;
  const res = useMemo(
    () => writeStatements({ ownership: own, F: Fe, warpWidth, blockX, blockY, rowPitch: 1024, segmentBytes }),
    [own, Fe, warpWidth, blockX, blockY, segmentBytes],
  );
  const n = res.statements.length;
  const at = Math.min(m, n - 1);
  const cur = res.statements[at];
  const soFar = res.statements.slice(0, at + 1).reduce((s, x) => s + x.access.transactions, 0);
  useEffect(() => setM(0), [own, Fe, warpWidth, blockX, blockY, segmentBytes]);
  useEffect(() => {
    if (!playing) return;
    if (at >= n - 1) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setM((x) => x + 1), 900);
    return () => window.clearTimeout(id);
  }, [playing, at, n]);

  // One strip per slice row the warp touches; each covers every segment any statement touches there.
  const rowsTouched = [...new Set(res.statements.flatMap((s) => s.lanes.map((l) => l.ty)))].sort((a, b) => a - b);
  const bpl = cur.bytesPerLane;
  const strips = rowsTouched.map((ty) => {
    const all = res.statements.flatMap((s) => s.lanes.filter((l) => l.ty === ty).map((l) => l.byte));
    const lo = Math.floor(Math.min(...all) / segmentBytes) * segmentBytes;
    const hi = Math.ceil((Math.max(...all) + bpl) / segmentBytes) * segmentBytes;
    return { ty, lo, hi };
  });
  const span = Math.max(...strips.map((s) => s.hi - s.lo));
  const px = Math.min(1.6, 660 / span);
  const X0 = 6;
  const pitch = 54;
  const VW = Math.max(360, Math.ceil(X0 * 2 + span * px));
  const VH = 6 + strips.length * pitch;
  const curSegs = new Set(cur.access.segments);
  const lanes = cur.lanes;
  const step = lanes.length > 1 ? lanes[1].element - lanes[0].element : 0;

  return (
    <div data-view="writes" data-statement={at} data-statements={n} data-transactions={cur.access.transactions}
      data-sofar={soFar} data-total={res.transactions} data-runs={cur.runs} data-width={warpWidth} data-segment={segmentBytes}>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label={tr('Ownership', 'Quyền sở hữu')} style={{ flexWrap: 'wrap', rowGap: 4 }}>
          {OWNERSHIP.map((o) => (
            <button key={o.id} aria-pressed={o.id === own} onClick={() => setOwn(o.id)}>{o.label}</button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <Field label={tr('Group width', 'Độ rộng nhóm')}>
          <select value={group} onChange={(e) => setGroup(e.target.value as Group)}>
            <option value="32">{tr('32 lanes (NVIDIA; AMD Radeon)', '32 lane (NVIDIA; AMD Radeon)')}</option>
            <option value="64">{tr('64 lanes (AMD Instinct)', '64 lane (AMD Instinct)')}</option>
            <option value="apple">
              {tr('Apple: ', 'Apple: ')}
              {apple ? tr(`${apple.width} lanes, from ${apple.source}`, `${apple.width} lane, từ ${apple.source}`) : tr('not recorded here yet', 'chưa được ghi lại ở đây')}
            </option>
          </select>
        </Field>
        {group === 'apple' && !apple ? (
          <Field label={tr('SIMD width (your choice)', 'Độ rộng SIMD (bạn chọn)')}>
            <select value={appleChoice} onChange={(e) => setAppleChoice(+e.target.value)}>
              {[16, 32, 64].map((w) => <option key={w} value={w}>{w}</option>)}
            </select>
          </Field>
        ) : null}
        <Field label="blockDim.x">
          <select value={blockX} onChange={(e) => setBlockX(+e.target.value)}>
            {[16, 32, 64, 128].map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </Field>
        <Field label="blockDim.y">
          <select value={blockY} onChange={(e) => setBlockY(+e.target.value)}>
            {[1, 2, 4].map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </Field>
        <Field label={tr('Coarsening factor', 'Hệ số gộp việc')}>
          <select value={Fe} onChange={(e) => setF(+e.target.value)}>
            {[2, 4, 8].map((b) => <option key={b} value={b} disabled={own === 'float4' && b % 4 !== 0}>{b}</option>)}
          </select>
        </Field>
        <Field label={tr('Memory unit', 'Đơn vị bộ nhớ')}>
          <select value={seg} onChange={(e) => setSeg(e.target.value as SegmentModelId)}>
            <option value="nvidia">{tr('NVIDIA: 32-byte sectors', 'NVIDIA: sector 32 byte')}</option>
            <option value="amd64">{tr('AMD: 64-byte lines', 'AMD: dòng 64 byte')}</option>
            <option value="amd128">{tr('AMD: 128-byte lines', 'AMD: dòng 128 byte')}</option>
            <option value="metal">{tr('Apple: not documented', 'Apple: không được ghi nhận')}</option>
          </select>
        </Field>
        {seg === 'metal' ? (
          <Field label={tr('Segment (your choice)', 'Đoạn (bạn chọn)')}>
            <select value={appleSeg} onChange={(e) => setAppleSeg(+e.target.value)}>
              {[32, 64, 128].map((b) => <option key={b} value={b}>{b}{tr(' bytes', ' byte')}</option>)}
            </select>
          </Field>
        ) : null}
      </div>
      {group === 'apple' && !apple ? (
        <Translated
          vi={
            <p className="ui muted" style={{ margin: '0 0 0.6rem', fontSize: '0.82rem' }}>
              Apple không ghi nhận độ rộng SIMD cố định nào: kernel đọc nó từ pipeline của mình (threadExecutionWidth). Hãy chạy
              lab 21.1 trên máy Mac này và giá trị nó ghi lại sẽ thay cho lựa chọn của bạn.
            </p>
          }
        >
          <p className="ui muted" style={{ margin: '0 0 0.6rem', fontSize: '0.82rem' }}>
            Apple documents no fixed SIMD width: a kernel reads it from its pipeline (threadExecutionWidth). Run lab 21.1 on
            this Mac and the value it records replaces your choice.
          </p>
        </Translated>
      ) : null}

      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <button onClick={() => { setPlaying(false); setM(0); }} disabled={at === 0}>
          <Translated vi="Bắt đầu lại">Restart</Translated>
        </button>
        <button onClick={() => { setPlaying(false); setM(at - 1); }} disabled={at === 0}>
          <Translated vi="Lùi">Back</Translated>
        </button>
        <button className="primary" onClick={() => { setPlaying(false); setM(at + 1); }} disabled={at >= n - 1}>
          <Translated vi="Câu lệnh kế tiếp">Next statement</Translated>
        </button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= n - 1}>{playing ? tr('Pause', 'Tạm dừng') : tr('Play', 'Phát')}</button>
        <span className="muted" aria-live="polite">
          <Translated vi={<>{own === 'float4' ? 'lần lưu' : 'câu lệnh'} {at + 1} trên {n}</>}>
            {own === 'float4' ? 'store' : 'statement'} {at + 1} of {n}
          </Translated>
        </span>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${VW} ${VH}`} role="img" aria-label={tr('Bytes of the slice one warp writes in one statement', 'Các byte của lát mà một warp ghi trong một câu lệnh')}
          style={{ width: '100%', maxWidth: VW * 1.2, minWidth: Math.min(MIN_DRAW, VW), display: 'block' }}>
          {strips.map((s, r) => {
            const y = 4 + r * pitch;
            const segs: number[] = [];
            for (let b = s.lo; b < s.hi; b += segmentBytes) segs.push(b / segmentBytes);
            return (
              <g key={s.ty} data-strip={s.ty}>
                <text x={X0} y={y + 10} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
                  <Translated vi={<>hàng {s.ty} của khối <tspan style={{ fill: 'var(--ink-3)' }}>byte {thousands(s.lo)} đến {thousands(s.hi - 1)}</tspan></>}>
                    row {s.ty} of the block <tspan style={{ fill: 'var(--ink-3)' }}>bytes {thousands(s.lo)} to {thousands(s.hi - 1)}</tspan>
                  </Translated>
                </text>
                {segs.map((g) => (
                  <rect key={g} x={X0 + (g * segmentBytes - s.lo) * px} y={y + 15} width={Math.max(1, segmentBytes * px - 1)} height={22}
                    fill={curSegs.has(g) ? 'var(--global-wash)' : 'none'} stroke={curSegs.has(g) ? 'var(--global)' : 'var(--rule)'} strokeWidth="0.8"
                    data-seg={curSegs.has(g) ? 'on' : 'off'} />
                ))}
                {res.statements.slice(0, at).flatMap((st) => st.lanes.filter((l) => l.ty === s.ty).map((l) => (
                  <rect key={`${st.m}-${l.lane}`} x={X0 + (l.byte - s.lo) * px} y={y + 19} width={Math.max(0.8, st.bytesPerLane * px - 0.3)} height={14}
                    fill="var(--global)" fillOpacity="0.25" />
                )))}
                {lanes.filter((l) => l.ty === s.ty).map((l) => {
                  const w = Math.max(1, bpl * px - 0.3);
                  return (
                    <g key={l.lane}>
                      <rect x={X0 + (l.byte - s.lo) * px} y={y + 19} width={w} height={14} fill="var(--global)">
                        <title>
                          <Translated vi={<>lane {l.lane}: phần tử {l.element}, byte {thousands(l.byte)} đến {thousands(l.byte + bpl - 1)}</>}>
                            lane {l.lane}: element {l.element}, bytes {thousands(l.byte)} to {thousands(l.byte + bpl - 1)}
                          </Translated>
                        </title>
                      </rect>
                      {w >= 12 ? <text x={X0 + (l.byte - s.lo) * px + w / 2} y={y + 29} fontSize="7.5" textAnchor="middle" style={{ fill: 'var(--page)' }}>{l.lane}</text> : null}
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>
      </div>

      <Translated
        vi={
          <p className="ui" style={{ margin: '0.6rem 0 0.3rem' }} data-readout="statement">
            {own === 'float4' ? `Lần lưu ${at + 1}` : `Câu lệnh ${at + 1}`}: lane l ghi phần tử {lanes[0].element}
            {step ? <> + {step}·l</> : null}
            {rowsTouched.length > 1 ? <> (các lane vượt quá blockDim.x tiếp tục ở hàng kế tiếp)</> : null}, mỗi lane {bpl} byte.{' '}
            <b>{thousands(cur.access.transactions)} {({ sector: 'sector', segment: 'đoạn', line: 'dòng' })[unit]}</b>, dùng {thousands(cur.access.bytesNeeded)} trên {thousands(cur.access.bytesFetched)} byte
            ({pct(cur.access.efficiency)}), {thousands(cur.runs)} dải byte liên tục.
          </p>
        }
      >
        <p className="ui" style={{ margin: '0.6rem 0 0.3rem' }} data-readout="statement">
          {own === 'float4' ? `Store ${at + 1}` : `Statement ${at + 1}`}: lane l writes element {lanes[0].element}
          {step ? <> + {step}·l</> : null}
          {rowsTouched.length > 1 ? <> (lanes past blockDim.x continue on the next row)</> : null}, {bpl} bytes each.{' '}
          <b>{plural(cur.access.transactions, unit)}</b>, {thousands(cur.access.bytesNeeded)} of {thousands(cur.access.bytesFetched)} bytes
          used ({pct(cur.access.efficiency)}), {plural(cur.runs, 'run')} of contiguous bytes.
        </p>
      </Translated>
      <Translated
        vi={
          <p className="ui muted" style={{ margin: 0 }} data-readout="total">
            Đến giờ: {thousands(soFar)} {({ sector: 'sector', segment: 'đoạn', line: 'dòng' })[unit]} qua {at + 1} trên {n} {own === 'float4' ? 'lần lưu' : 'câu lệnh'}. Tất cả:{' '}
            {thousands(res.transactions)} {({ sector: 'sector', segment: 'đoạn', line: 'dòng' })[unit]}, {thousands(res.bytesNeeded)} byte được ghi, {thousands(res.bytesFetched)} byte được di chuyển.
            {own === 'float4' && !res.aligned ? <b className="t-fail"> Lệch căn: một float4 cần mọi hàng bắt đầu tại một ranh giới 16 byte.</b> : null}
          </p>
        }
      >
        <p className="ui muted" style={{ margin: 0 }} data-readout="total">
          So far: {plural(soFar, unit)} over {at + 1} of {n} {own === 'float4' ? 'stores' : 'statements'}. All of them:{' '}
          {plural(res.transactions, unit)}, {thousands(res.bytesNeeded)} bytes written, {thousands(res.bytesFetched)} moved.
          {own === 'float4' && !res.aligned ? <b className="t-fail"> Misaligned: a float4 needs every row to start on a 16-byte boundary.</b> : null}
        </p>
      </Translated>
    </div>
  );
}

// ================================================================ view 3: cutoff and bins

const CUT_3D = [8, 12, 16];
const BIN_3D = [2, 4, 8];

function CutoffView() {
  const hatch = useId();
  const svgRef = useRef<SVGSVGElement>(null);
  const [side, setSide] = useState(4);
  const [bins, setBins] = useState(8);
  const [density, setDensity] = useState(2);
  const [seed, setSeed] = useState(1);
  const [c, setC] = useState(1.3);
  const [mode, setMode] = useState<RadiusMode>('half');
  const [cap, setCap] = useState<number | null>(null);
  const [block, setBlock] = useState<[number, number]>([3, 3]);
  const [thread, setThread] = useState(5);
  /** The staged atom shown on the lanes; null until the reader picks one: then the first atom that splits a warp. */
  const [pick, setPick] = useState<{ entry: number; slot: number } | null>(null);
  const [abs, setAbs] = useState(false);
  const [prev, setPrev] = useState<{ points: number; atoms: number; direct: number; tests: number; perPoint: number } | null>(null);
  const [c3, setC3] = useState(12);
  const [b3, setB3] = useState(4);
  const { reports } = useReports();
  const apple = appleWidth(reports);
  const [group, setGroup] = useState<Group>('32');
  const [appleChoice, setAppleChoice] = useState(32);
  /** Lanes per warp, wavefront or SIMD-group: the reader's choice, never a fixed 32. */
  const width = group === 'apple' ? (apple?.width ?? appleChoice) : Number(group);
  const groupName = group === '64' ? 'wavefront' : group === 'apple' ? 'SIMD-group' : 'warp'; // i18n-ignore: the glossary keeps these names

  const nAtoms = Math.round(density * bins * bins);
  const world = useMemo(() => makeWorld(bins, bins, side, nAtoms, seed), [bins, side, nAtoms, seed]);
  const maxOcc = maxOccupancy(world);
  const capAuto = useMemo(() => capacityFor(world, 0.03), [world]);
  const capacity = Math.max(1, Math.min(maxOcc || 1, cap ?? capAuto));
  const binning = useMemo(() => binAtoms(world, capacity), [world, capacity]);
  const cutoff = c * side;
  const R = superRadius(cutoff, side, 2, mode) / side;
  const bx = Math.min(block[0], bins - 1);
  const by = Math.min(block[1], bins - 1);
  const walk = useMemo(() => blockWalk(world, binning, cutoff, R, bx, by), [world, binning, cutoff, R, bx, by]);
  const grid = useMemo(() => runGrid(world, binning, cutoff, R, false), [world, binning, cutoff, R]);
  const cnt = counts(R, 2);
  const t = Math.min(thread, walk.points.length - 1);
  const [pi, pj] = walk.points[t];
  const nf = nearFar(world, pi, pj, cutoff);
  const inBox = walk.list.filter((e) => e.inBox);
  const auto = useMemo(() => {
    for (const x of walk.list)
      for (let k = 0; k < x.slots.length; k++) if (x.slots[k] >= 0 && acceptMasks(x, k, width).some(diverges)) return { entry: x.k, slot: k };
    const first = walk.list.find((x) => x.inBox);
    return { entry: first ? first.k : 0, slot: 0 };
  }, [walk, width]);
  /** Over the whole list of this block: staged real atoms, and how many of them split at least one group. */
  const split = useMemo(() => {
    let real = 0;
    let splitting = 0;
    for (const x of walk.list)
      for (let k = 0; k < x.slots.length; k++) {
        if (x.slots[k] < 0) continue;
        real++;
        if (acceptMasks(x, k, width).some(diverges)) splitting++;
      }
    return { real, splitting };
  }, [walk, width]);
  const chosen = pick ?? auto;
  const e = walk.list[Math.min(chosen.entry, walk.list.length - 1)];
  const sl = Math.min(chosen.slot, Math.max(0, e.slots.length - 1));
  const masks = e.slots.length ? acceptMasks(e, sl, width) : [];
  const threadStats = walk.list.reduce((s, x) => {
    for (const row of x.outcome) {
      const o = row[t];
      s.scanned++;
      if (o === 'accept') s.accepted++;
      if (o === 'dummy') s.dummy++;
    }
    return s;
  }, { scanned: 0, accepted: 0, dummy: 0 });
  const overflowSet = new Set(binning.overflow);
  const acceptedByThread = new Set<number>();
  for (const x of walk.list) x.slots.forEach((s, k) => { if (s >= 0 && x.outcome[k][t] === 'accept') acceptedByThread.add(s); });
  const listBins = new Set(inBox.map((x) => `${x.bx},${x.by}`));
  const points = worldW(world) * worldH(world);
  const stats = { points, atoms: world.atoms.length, direct: grid.directPairs, tests: grid.tests, perPoint: grid.tests / points };

  // Drawing: the box, bins, the block's points, atoms, the super circle and the thread's cutoff circle.
  const BOX = 380;
  const bp = BOX / bins; // px per bin
  const gp = bp / side; // px per grid unit
  const O = 8;
  const toPx = (x: number) => O + (x + 0.5) * gp;
  const [cx, cy] = binCentre(world, bx, by);
  const onClick = (ev: RMouseEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    const m = svg?.getScreenCTM();
    if (!svg || !m) return;
    const q = new DOMPoint(ev.clientX, ev.clientY).matrixTransform(m.inverse());
    const gx = (q.x - O) / gp - 0.5;
    const gy = (q.y - O) / gp - 0.5;
    const [nbx, nby] = binOf(world, gx, gy);
    if (nbx === bx && nby === by) {
      const tx = Math.max(0, Math.min(side - 1, Math.round(gx) - bx * side));
      const ty = Math.max(0, Math.min(side - 1, Math.round(gy) - by * side));
      setThread(ty * side + tx);
    } else {
      setBlock([nbx, nby]);
      setPick(null);
    }
  };
  const share3 = acceptedShare(c3, b3, 3, 'half');
  const n3h = counts(superRadius(c3, b3, 3, 'half') / b3, 3);
  const n3f = counts(superRadius(c3, b3, 3, 'full') / b3, 3);

  return (
    <div data-view="cutoff" data-list={cnt.total} data-full={cnt.full} data-partial={cnt.partial} data-radius={R.toFixed(3)}
      data-capacity={capacity} data-overflow={binning.overflow.length} data-dummy={binning.dummy} data-tests={grid.tests}
      data-direct={grid.directPairs} data-thread-scanned={threadStats.scanned} data-thread-accepted={threadStats.accepted}
      data-bins={bins} data-atoms={world.atoms.length}>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <Field label={tr(`Cutoff ${c.toFixed(2)} bin widths`, `Bán kính cắt ${c.toFixed(2)} bề rộng ngăn`)}>
          <input type="range" min={0.5} max={3} step={0.05} value={c} aria-label={tr('cutoff in bin widths', 'bán kính cắt tính bằng bề rộng ngăn')} onChange={(ev) => setC(+ev.target.value)} />
        </Field>
        <button onClick={() => { setC(Number((2 - Math.SQRT2 / 2).toFixed(4))); setMode('half'); }}>
          <Translated vi="R = 2 bề rộng ngăn">R = 2 bin widths</Translated>
        </button>
        <Field label={tr('Super-circle radius', 'Bán kính vòng tròn lớn')}>
          <select value={mode} onChange={(ev) => setMode(ev.target.value as RadiusMode)}>
            <option value="half">{tr('cutoff + half the diagonal (text, Fig. 21.14)', 'bán kính cắt + nửa đường chéo (văn bản, Fig. 21.14)')}</option>
            <option value="full">{tr("cutoff + the whole diagonal (Fig. 21.13's label)", 'bán kính cắt + cả đường chéo (nhãn của Fig. 21.13)')}</option>
          </select>
        </Field>
        <Field label={tr('Bin side', 'Cạnh ngăn')}>
          <select value={side} onChange={(ev) => { setSide(+ev.target.value); setThread(0); setPrev(null); }}>
            <option value={4}>{tr('4 grid spacings (16 threads a block)', '4 khoảng cách lưới (16 luồng mỗi khối)')}</option>
            <option value={8}>{tr('8 grid spacings (64 threads a block)', '8 khoảng cách lưới (64 luồng mỗi khối)')}</option>
          </select>
        </Field>
        <Field label={tr(`Atoms per bin ${density} (${thousands(nAtoms)} atoms)`, `Nguyên tử mỗi ngăn ${density} (${thousands(nAtoms)} nguyên tử)`)}>
          <input type="range" min={0.5} max={4} step={0.25} value={density} aria-label={tr('atoms per bin', 'nguyên tử mỗi ngăn')} onChange={(ev) => { setDensity(+ev.target.value); setCap(null); setPrev(null); }} />
        </Field>
        <button onClick={() => { setSeed(seed + 1); setCap(null); }}>
          <Translated vi="Nguyên tử ngẫu nhiên mới">New random atoms</Translated>
        </button>
        {bins < 16 ? (
          <button onClick={() => { setPrev(stats); setBins(bins * 2); setCap(capacity); }}>
            <Translated vi="Nhân đôi hộp">Double the box</Translated>
          </button>
        ) : (
          <button onClick={() => { setPrev(null); setBins(bins / 2); setCap(capacity); setBlock([Math.min(bx, bins / 2 - 1), Math.min(by, bins / 2 - 1)]); }}>
            <Translated vi={<>Quay lại {bins / 2} × {bins / 2} ngăn</>}>Back to {bins / 2} × {bins / 2} bins</Translated>
          </button>
        )}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 5fr) minmax(0, 4fr)' }}>
        <div style={{ overflowX: 'auto' }}>
          <svg ref={svgRef} viewBox={`0 0 ${BOX + 2 * O} ${BOX + 2 * O}`} role="img" aria-label={tr("The box of bins, the chosen block's neighbourhood and the atoms", 'Hộp các ngăn, vùng lân cận của khối đã chọn và các nguyên tử')}
            style={{ width: '100%', maxWidth: 470, minWidth: 300, display: 'block', cursor: 'pointer' }} onClick={onClick}>
            <defs>
              <pattern id={hatch} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" strokeWidth="1" />
              </pattern>
            </defs>
            {Array.from({ length: bins * bins }, (_, k) => {
              const x = k % bins;
              const y = Math.floor(k / bins);
              const ent = walk.list.find((q) => q.bx === x && q.by === y);
              const isBlock = x === bx && y === by;
              const isEntry = e.inBox && e.bx === x && e.by === y;
              return (
                <rect key={k} x={O + x * bp} y={O + y * bp} width={bp} height={bp}
                  fill={ent ? 'var(--shared)' : 'var(--cell)'} fillOpacity={ent ? (ent.rel.cover === 'full' ? 0.32 : 0.13) : 1}
                  stroke={isEntry ? 'var(--ink)' : 'var(--rule)'} strokeWidth={isEntry ? 2 : isBlock ? 1.6 : 0.6}
                  data-bin={`${x},${y}`} data-cover={ent ? ent.rel.cover : 'none'} />
              );
            })}
            <rect x={O + bx * bp} y={O + by * bp} width={bp} height={bp} fill="none" stroke="var(--ink)" strokeWidth="2.4" pointerEvents="none" /* i18n-ignore: SVG attribute values */ />
            <circle cx={toPx(cx)} cy={toPx(cy)} r={R * bp} fill="none" stroke="var(--ink)" strokeWidth="1.4" strokeDasharray="6 4" pointerEvents="none" data-super="" /* i18n-ignore: SVG attribute values */ />
            <circle cx={toPx(pi)} cy={toPx(pj)} r={cutoff * gp} fill="none" stroke="var(--ink-2)" strokeWidth="1" pointerEvents="none" /* i18n-ignore: SVG attribute values */ />
            {walk.points.map(([i, j], k) => (
              <circle key={k} cx={toPx(i)} cy={toPx(j)} r={k === t ? 3.4 : 1.4} fill="var(--ink)" pointerEvents="none" /* i18n-ignore: SVG attribute values */ />
            ))}
            {world.atoms.map((a, k) => {
              const home = binOf(world, a.x, a.y).join(',');
              const scanned = listBins.has(home) && !overflowSet.has(k);
              const ok = acceptedByThread.has(k);
              return (
                <circle key={k} cx={toPx(a.x)} cy={toPx(a.y)} r={ok ? 3.2 : scanned ? 2.3 : 1.8}
                  fill={scanned ? 'var(--global)' : 'var(--ink-3)'} fillOpacity={ok || scanned ? 1 : 0.55}
                  stroke={overflowSet.has(k) ? 'var(--fail)' : ok ? 'var(--ink)' : 'none'} strokeWidth={overflowSet.has(k) ? 1.4 : 1}
                  pointerEvents="none" /* i18n-ignore: SVG attribute values */ />
              );
            })}
          </svg>
          <Translated
            vi={
              <p className="ui muted" style={{ fontSize: '0.78rem', margin: '0.4rem 0 0' }}>
                Nét đứt: vòng tròn lớn quanh khối đã chọn (có viền). Nét mảnh: bán kính cắt quanh điểm của luồng đã chọn. Các ngăn
                đậm hơn nằm trọn trong vòng tròn lớn, các ngăn nhạt hơn nằm một phần. Chấm lớn: các nguyên tử mà luồng nhận; vòng
                đỏ: các nguyên tử trong danh sách tràn. Nhấp vào một ngăn để chọn khối của nó, vào một điểm trong khối đã chọn để
                chọn luồng của nó.
              </p>
            }
          >
            <p className="ui muted" style={{ fontSize: '0.78rem', margin: '0.4rem 0 0' }}>
              Dashed: the super circle around the chosen block (outlined). Thin: the cutoff around the chosen thread's point.
              Darker bins are fully inside the super circle, lighter ones partly. Large dots: atoms the thread accepts; red
              rings: atoms on the overflow list. Click a bin to choose its block, a point inside the chosen block to choose
              its thread.
            </p>
          </Translated>
        </div>

        <div className="ui" style={{ fontSize: '0.86rem' }}>
          <Translated
            vi={
              <p style={{ margin: '0 0 0.5rem' }} data-readout="list">
                R = {c.toFixed(2)} + {mode === 'half' ? '0.71' : '1.41'} = <b>{R.toFixed(2)} bề rộng ngăn</b>. Danh sách có{' '}
                <b>{cnt.total} ngăn</b>: {cnt.full} nằm trọn bên trong, {cnt.partial} nằm một phần.
                {inBox.length < cnt.total ? <> Với khối này, {cnt.total - inBox.length} ngăn trong số đó nằm ngoài hộp và bị bỏ qua.</> : null}
              </p>
            }
          >
            <p style={{ margin: '0 0 0.5rem' }} data-readout="list">
              R = {c.toFixed(2)} + {mode === 'half' ? '0.71' : '1.41'} = <b>{R.toFixed(2)} bin widths</b>. The list has{' '}
              <b>{cnt.total} bins</b>: {cnt.full} fully inside, {cnt.partial} partly.
              {inBox.length < cnt.total ? <> For this block {cnt.total - inBox.length} of them lie outside the box and are skipped.</> : null}
            </p>
          </Translated>
          <div className="row" style={{ gap: '0.25rem', marginBottom: '0.3rem' }} role="group" aria-label={tr('The list', 'Danh sách')}>
            {walk.list.map((x) => (
              <button key={x.k} className="quiet" aria-pressed={x.k === e.k} disabled={!x.inBox}
                onClick={() => setPick({ entry: x.k, slot: 0 })} style={{ fontSize: '0.72rem', padding: '0.1rem 0.35rem' }}>
                {abs ? `(${x.bx}, ${x.by})` : `(${x.rel.dx}, ${x.rel.dy})`}
              </button>
            ))}
          </div>
          <label className="check" style={{ marginBottom: '0.6rem' }}>
            <input type="checkbox" checked={abs} onChange={(ev) => setAbs(ev.target.checked)} />{' '}
            <Translated vi={<>tọa độ tuyệt đối cho khối ({bx}, {by})</>}>absolute coordinates for block ({bx}, {by})</Translated>
          </label>
          <table className="threads" aria-label={tr('Counts', 'Số đếm')}>
            <tbody>
              <tr>
                <td style={WRAP}><Translated vi={<>khối này ({thousands(walk.points.length)} luồng)</>}>this block ({plural(walk.points.length, 'thread')})</Translated></td>
                <td style={WRAP} data-readout="block">
                  <Translated vi={<>{thousands(walk.tested)} phép kiểm tra ({thousands(walk.dummyTests)} trên nguyên tử giả), {thousands(walk.accepted)} được nhận ({pct(walk.tested ? walk.accepted / walk.tested : 0)})</>}>
                    {thousands(walk.tested)} tests ({thousands(walk.dummyTests)} on dummies), {thousands(walk.accepted)} accepted ({pct(walk.tested ? walk.accepted / walk.tested : 0)})
                  </Translated>
                </td>
              </tr>
              <tr>
                <td style={WRAP}><Translated vi={<>luồng tại điểm ({pi}, {pj})</>}>thread at point ({pi}, {pj})</Translated></td>
                <td style={WRAP} data-readout="thread">
                  <Translated vi={<>{threadStats.scanned} đã quét ({threadStats.dummy} nguyên tử giả), {threadStats.accepted} được nhận</>}>
                    {threadStats.scanned} scanned ({threadStats.dummy} dummies), {threadStats.accepted} accepted
                  </Translated>
                </td>
              </tr>
              <tr>
                <td style={WRAP}><Translated vi="cả lưới, trực tiếp">whole grid, direct</Translated></td>
                <td style={WRAP} data-readout="direct">
                  <Translated vi={<>{thousands(points)} điểm × {thousands(world.atoms.length)} nguyên tử = {thousands(grid.directPairs)} cặp</>}>
                    {thousands(points)} points × {thousands(world.atoms.length)} atoms = {thousands(grid.directPairs)} pairs
                  </Translated>
                </td>
              </tr>
              <tr>
                <td style={WRAP}><Translated vi="cả lưới, phân ngăn">whole grid, binned</Translated></td>
                <td style={WRAP} data-readout="binned">
                  <Translated vi={<>{thousands(grid.tests)} phép kiểm tra ({pct(grid.tests / grid.directPairs)} so với trực tiếp), {thousands(grid.realTests)} trên nguyên tử thật, {thousands(grid.accepted)} được nhận</>}>
                    {thousands(grid.tests)} tests ({pct(grid.tests / grid.directPairs)} of direct), {thousands(grid.realTests)} on real atoms, {thousands(grid.accepted)} accepted
                  </Translated>
                </td>
              </tr>
              {prev ? (
                <tr data-readout="doubled">
                  <td style={WRAP}><Translated vi="so với hộp trước đó (cùng sức chứa)">against the box before (same capacity)</Translated></td>
                  <td style={WRAP}>
                    <Translated vi={<>điểm ×{(points / prev.points).toFixed(0)}, nguyên tử ×{(world.atoms.length / prev.atoms).toFixed(0)}, cặp trực tiếp ×{(grid.directPairs / prev.direct).toFixed(0)}, phép kiểm tra phân ngăn ×{(grid.tests / prev.tests).toFixed(1)} (phép kiểm tra mỗi điểm ×{(stats.perPoint / prev.perPoint).toFixed(2)})</>}>
                      points ×{(points / prev.points).toFixed(0)}, atoms ×{(world.atoms.length / prev.atoms).toFixed(0)}, direct pairs ×{(grid.directPairs / prev.direct).toFixed(0)}, binned tests ×{(grid.tests / prev.tests).toFixed(1)} (tests per point ×{(stats.perPoint / prev.perPoint).toFixed(2)})
                    </Translated>
                  </td>
                </tr>
              ) : null}
              <tr>
                <td style={WRAP}><Translated vi="trường xa bị bỏ ở đây">far field dropped here</Translated></td>
                <td style={WRAP} data-readout="far">
                  <Translated vi={<>giữ {nf.near.toFixed(3)} từ {nf.nearAtoms} nguyên tử; bỏ {nf.far.toFixed(3)}; trực tiếp {nf.direct.toFixed(3)}</>}>
                    kept {nf.near.toFixed(3)} from {nf.nearAtoms} atoms; dropped {nf.far.toFixed(3)}; direct {nf.direct.toFixed(3)}
                  </Translated>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="cols" style={{ marginTop: '0.9rem' }}>
        <div className="panel" data-mask="" data-group-width={width} data-split={split.splitting} data-real={split.real}>
          <b className="ui">
            <Translated vi="Một nguyên tử đã nạp, trên các lane của khối">One staged atom, across the block's lanes</Translated>
          </b>
          <div className="row" style={{ margin: '0.4rem 0', alignItems: 'flex-end' }}>
            <Field label={tr('Group width', 'Độ rộng nhóm')}>
              <select value={group} onChange={(ev) => setGroup(ev.target.value as Group)}>
                <option value="32">{tr('32 lanes (NVIDIA; AMD Radeon)', '32 lane (NVIDIA; AMD Radeon)')}</option>
                <option value="64">{tr('64 lanes (AMD Instinct)', '64 lane (AMD Instinct)')}</option>
                <option value="apple">
                  {tr('Apple: ', 'Apple: ')}
                  {apple ? tr(`${apple.width} lanes, from ${apple.source}`, `${apple.width} lane, từ ${apple.source}`) : tr('not recorded here yet', 'chưa được ghi lại ở đây')}
                </option>
              </select>
            </Field>
            {group === 'apple' && !apple ? (
              <Field label={tr('SIMD width (your choice)', 'Độ rộng SIMD (bạn chọn)')}>
                <select value={appleChoice} onChange={(ev) => setAppleChoice(+ev.target.value)}>
                  {[16, 32, 64].map((w) => <option key={w} value={w}>{w}</option>)}
                </select>
              </Field>
            ) : null}
          </div>
          <Translated
            vi={
              <p className="ui muted" style={{ margin: '0.3rem 0 0.4rem', fontSize: '0.8rem' }}>
                Mục ({e.rel.dx}, {e.rel.dy}) của danh sách, ngăn ({e.bx}, {e.by}). Chọn một ô chứa: mọi luồng kiểm tra nguyên tử
                trong đó với bán kính cắt quanh điểm của riêng nó.
              </p>
            }
          >
            <p className="ui muted" style={{ margin: '0.3rem 0 0.4rem', fontSize: '0.8rem' }}>
              Entry ({e.rel.dx}, {e.rel.dy}) of the list, bin ({e.bx}, {e.by}). Pick a slot: every thread tests the atom in it
              against the cutoff around its own point.
            </p>
          </Translated>
          <div className="seg" role="group" aria-label={tr('Slot', 'Ô chứa')} style={{ marginBottom: '0.5rem', flexWrap: 'wrap' }}>
            {e.slots.map((s, k) => (
              <button key={k} aria-pressed={k === sl} onClick={() => setPick({ entry: e.k, slot: k })} style={{ fontSize: '0.78rem' }}>
                {s < 0 ? tr(`slot ${k}: dummy`, `ô ${k}: giả`) : tr(`slot ${k}: atom ${s}`, `ô ${k}: nguyên tử ${s}`)}
              </button>
            ))}
          </div>
          {masks.map((mk, w) => (
            <div key={w} className="row" style={{ gap: 2, marginBottom: 4 }} data-warp={w} data-diverges={diverges(mk) ? 'yes' : 'no'}>
              <span className="muted" style={{ width: '5.4rem', fontSize: '0.78rem' }}>{groupName} {w}</span>
              {mk.map((on, l) => (
                <svg key={l} width="11" height="14" aria-hidden="true">
                  <rect x="0.5" y="0.5" width="10" height="13" rx="1.5"
                    fill={e.slots[sl] < 0 ? `url(#${hatch})` : on ? 'var(--ok)' : 'var(--cell)'} stroke={on ? 'var(--ok)' : 'var(--rule)'} />
                </svg>
              ))}
              <span style={{ marginLeft: 6 }}>
                {e.slots[sl] < 0 ? <span className="chip"><Translated vi="giả: được kiểm tra, cộng 0">dummy: tested, adds 0</Translated></span>
                  : diverges(mk) ? <span className="chip fail"><Translated vi={<>phân kỳ: {mk.filter(Boolean).length} nhận, {mk.length - mk.filter(Boolean).length} loại</>}>diverges: {mk.filter(Boolean).length} accept, {mk.length - mk.filter(Boolean).length} reject</Translated></span>
                  : <span className="chip ok"><Translated vi={<>không phân kỳ: tất cả {mk.some(Boolean) ? 'nhận' : 'loại'}</>}>no divergence: all {mk.some(Boolean) ? 'accept' : 'reject'}</Translated></span>}
              </span>
            </div>
          ))}
          <Translated
            vi={
              <p className="ui" style={{ margin: '0.5rem 0 0', fontSize: '0.84rem' }} data-readout="split">
                Trên toàn bộ danh sách của khối này: {split.splitting} trên {thousands(split.real)} nguyên tử thật đã nạp chia tách ít
                nhất một {groupName} {width} lane; các nguyên tử còn lại được mọi lane nhận hoặc loại như nhau.
              </p>
            }
          >
            <p className="ui" style={{ margin: '0.5rem 0 0', fontSize: '0.84rem' }} data-readout="split">
              Over this block's whole list: {split.splitting} of {plural(split.real, 'staged real atom')} split at least one{' '}
              {groupName} of {width} lanes; the others are accepted or rejected by every lane alike.
            </p>
          </Translated>
        </div>
        <div className="panel" data-capacity-panel="">
          <b className="ui">
            <Translated vi="Sức chứa ngăn">Bin capacity</Translated>
          </b>
          <div className="row" style={{ margin: '0.3rem 0' }}>
            <Field label={tr(`Capacity ${plural(capacity, 'atom')} (largest bin holds ${maxOcc})`, `Sức chứa ${thousands(capacity)} nguyên tử (ngăn lớn nhất chứa ${maxOcc})`)}>
              <input type="range" min={1} max={Math.max(1, maxOcc)} value={capacity} aria-label={tr('bin capacity', 'sức chứa ngăn')} onChange={(ev) => setCap(+ev.target.value)} />
            </Field>
          </div>
          <Translated
            vi={
              <p className="ui" style={{ margin: 0 }} data-readout="capacity">
                Ô giả: {thousands(binning.dummy)} trên {thousands(bins * bins * capacity)} ({pct(binning.dummy / (bins * bins * capacity))}).
                Phần tràn: {thousands(binning.overflow.length)} nguyên tử ({pct(binning.overflow.length / Math.max(1, world.atoms.length), 1)}), được cộng trên host
                sau đó: {binning.overflow.length <= 0.03 * world.atoms.length ? 'dưới' : 'trên'} mức 3 % của sách.
              </p>
            }
          >
            <p className="ui" style={{ margin: 0 }} data-readout="capacity">
              Dummy slots: {thousands(binning.dummy)} of {thousands(bins * bins * capacity)} ({pct(binning.dummy / (bins * bins * capacity))}).
              Overflow: {plural(binning.overflow.length, 'atom')} ({pct(binning.overflow.length / Math.max(1, world.atoms.length), 1)}), summed on the host
              afterwards: {binning.overflow.length <= 0.03 * world.atoms.length ? 'under' : 'over'} the book's 3 %.
            </p>
          </Translated>
        </div>
      </div>

      <div className="panel" style={{ marginTop: '0.9rem' }} data-book3d={n3h.total}>
        <b className="ui">
          <Translated vi="3D, trường hợp của sách (p. 525)">3D, the book's case (p. 525)</Translated>
        </b>
        <div className="row" style={{ margin: '0.4rem 0' }}>
          <Field label={tr('Cutoff', 'Bán kính cắt')}>
            <select value={c3} onChange={(ev) => setC3(+ev.target.value)}>
              {CUT_3D.map((v) => <option key={v} value={v}>{v} Å</option>)}
            </select>
          </Field>
          <Field label={tr('Bin side', 'Cạnh ngăn')}>
            <select value={b3} onChange={(ev) => setB3(+ev.target.value)}>
              {BIN_3D.map((v) => <option key={v} value={v}>{v} Å</option>)}
            </select>
          </Field>
        </div>
        <Translated
          vi={
            <p className="ui" style={{ margin: 0 }} data-readout="book3d">
              Bán kính cắt + nửa đường chéo ngăn: <b>{thousands(n3h.total)} ngăn</b> ({thousands(n3h.full)} nằm trọn bên trong). Với cả
              đường chéo của Fig. 21.13: {thousands(n3f.total)}. Ở mật độ đều, khoảng <b>{pct(share3)}</b> số nguyên tử mà một luồng
              kiểm tra qua được phép kiểm tra bán kính cắt.
            </p>
          }
        >
          <p className="ui" style={{ margin: 0 }} data-readout="book3d">
            Cutoff + half the bin diagonal: <b>{thousands(n3h.total)} bins</b> ({thousands(n3h.full)} fully inside). With Fig. 21.13's
            whole diagonal: {thousands(n3f.total)}. At uniform density about <b>{pct(share3)}</b> of the atoms a thread tests
            pass its cutoff test.
          </p>
        </Translated>
      </div>
    </div>
  );
}

// ================================================================ the explorer

const VIEW_LABEL: Record<ExplorerView, string> = {
  get mapping() {
    return tr('Who writes where', 'Ai ghi vào đâu');
  },
  get writes() {
    return tr('Write statements', 'Các câu lệnh ghi');
  },
  get cutoff() {
    return tr('Cutoff and bins', 'Bán kính cắt và ngăn');
  },
};

export function PairsExplorer({
  title,
  task,
  views,
  initialKernel = 'scatter',
  kernels = KERNEL_ORDER,
  foot,
}: {
  title: string;
  task?: ReactNode;
  views: ExplorerView[];
  initialKernel?: KernelId;
  /** The kernels the mapping view offers (all five by default). */
  kernels?: KernelId[];
  foot?: ReactNode;
}) {
  const [view, setView] = useState<ExplorerView>(views[0]);
  return (
    <Bench title={title} task={task} foot={foot}>
      {views.length > 1 ? (
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <span className="seg" role="group" aria-label={tr('View', 'Chế độ xem')}>
            {views.map((v) => <button key={v} aria-pressed={v === view} onClick={() => setView(v)}>{VIEW_LABEL[v]}</button>)}
          </span>
        </div>
      ) : null}
      {view === 'mapping' ? <MappingView initialKernel={initialKernel} kernels={kernels} /> :view === 'writes' ? <WritesView /> : <CutoffView />}
    </Bench>
  );
}
