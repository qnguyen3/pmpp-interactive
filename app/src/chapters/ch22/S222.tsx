import { useEffect, useMemo, useState, type MouseEvent, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { benchRows, num, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';
import { SEGMENT_MODELS, runs as byteRuns, warpAccess } from '../../sim/coalesce';
import {
  CRITERION_LABEL,
  DECOMP_LABEL,
  INTERLEAVE_LABEL,
  MAPPING_LABEL,
  NVIDIA_WARP,
  PRESETS,
  SPLITK_TINY,
  TOUR,
  decomposeState,
  describeSKStep,
  describeWorker,
  gradeCard,
  mappingsOf,
  ownerOfWeight,
  splitChoices,
  splitKOrder,
  splitKRun,
  splitKTraffic,
  warp0Load,
  warpOps,
  warpPlace,
  type Criterion,
  type Decomp,
  type Decomposition,
  type ExplorerState,
  type Interleave,
  type Kind,
  type MatvecProblem,
  type SplitKConfig,
} from '../../sim/decompose';
import { GatherScatterScene } from './GatherScatterScene';

const SECTOR = SEGMENT_MODELS.nvidia.segmentBytes!; // NVIDIA's 32-byte unit (ch06-facts A2)
const fmt = (v: number) => v.toLocaleString('en-US');
const fmt2 = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(2));

// ---------------------------------------------------------------- the lesson's kernels
// Copies of labs/ch22/snippets/scatter_conv_plain.cu and gather_conv.cu, which are
// front-end-checked and run on both emulators by S222.snippets.test.ts. That test also
// fails if these copies drift from the files.
const SCATTER = `// Input-centric 1-D convolution (scatter): thread i owns in[i]
// and adds its contribution to every output that uses it.
// Wrong on purpose: several threads add into the same out[o],
// and the += is a plain read, add and write, not an atomic.
// out[o] = sum over t of filter[t] * in[o + t - r]
__global__ void convScatterPlain(const float* in, const float* filter,
                                 float* out, int n, int r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
    for (int t = 0; t <= 2 * r; ++t) {
        int o = i - t + r;            // out[o] reads in[i] with tap t
        if (o >= 0 && o < n) out[o] += filter[t] * v;
    }
}

void launchConvScatterPlain(const float* in, const float* filter,
                            float* out, int n, int r) {
    cudaMemset(out, 0, n * sizeof(float));  // the threads add into it
    convScatterPlain<<<(n + 255) / 256, 256>>>(in, filter, out, n, r);
}`;

const GATHER = `// Output-centric 1-D convolution (gather): thread o owns out[o],
// adds up the inputs it needs in a register, and writes once.
// No other thread writes out[o].
// out[o] = sum over t of filter[t] * in[o + t - r]
__global__ void convGather(const float* in, const float* filter,
                           float* out, int n, int r) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n) return;
    float acc = 0.0f;
    for (int t = 0; t <= 2 * r; ++t) {
        int i = o + t - r;            // the input tap t reads
        if (i >= 0 && i < n) acc += filter[t] * in[i];
    }
    out[o] = acc;
}

void launchConvGather(const float* in, const float* filter,
                      float* out, int n, int r) {
    convGather<<<(n + 255) / 256, 256>>>(in, filter, out, n, r);
}`;

const CONV_TIERS: Record<string, Tier> = {
  in: 'global', filter: 'global', out: 'global',
  i: 'reg', o: 'reg', t: 'reg', v: 'reg', acc: 'reg', n: 'reg', r: 'reg',
};

/** Code in a table cell: may wrap at its spaces, never inside an identifier. */
const C = ({ children }: { children: ReactNode }) => (
  <code style={{ whiteSpace: 'normal', overflowWrap: 'normal', wordBreak: 'normal' }}>{children}</code>
);

// ---------------------------------------------------------------- explorer: warp widths
const LAB61 = 'ch06/lab1_coalescing';
interface WidthChoice {
  key: string;
  width: number;
  label: string;
}
/** The documented widths, plus any the reader's devices reported (lab 6.1's simd_width row, else warpSize). */
function useWidthChoices(): WidthChoice[] {
  const { reports } = useReports();
  const fixed: WidthChoice[] = [
    { key: 'nv', width: NVIDIA_WARP, label: tr(`${NVIDIA_WARP} (NVIDIA; AMD gfx10 and later)`, `${NVIDIA_WARP} (NVIDIA; AMD gfx10 trở về sau)`) },
    { key: 'amd', width: 64, label: '64 (AMD gfx9, Instinct)' }, // i18n-ignore: product names
  ];
  const devices = realBackends(reports).flatMap((b) => {
    const rows = benchRows(reports[b], LAB61, 'solution');
    const w = num(rows.find((r) => r.name === 'simd_width')?.width) ?? num(Number(reports[b].device?.warpSize));
    return w ? [{ key: 'dev:' + b, width: w, label: tr(`${w} (your ${String(reports[b].device.name ?? b)}, ${b})`, `${w} (${String(reports[b].device.name ?? b)} của bạn, ${b})`) }] : [];
  });
  return [...fixed, ...devices];
}

// ---------------------------------------------------------------- explorer: drawings
const DW = 720;

/** Inputs on top, workers in the middle, outputs below; arrows from the decomposition. */
function Bipartite({ d, sel, onSel }: { d: Decomposition; sel: number | null; onSel: (w: number) => void }) {
  const p = d.problem;
  const L = 74;
  const span = DW - L - 14;
  const pitch = (n: number) => Math.min(46, span / n);
  const xs = (n: number) => (k: number) => L + (span - n * pitch(n)) / 2 + k * pitch(n) + pitch(n) / 2;
  const inX = xs(p.nIn);
  const outX = xs(p.nOut);
  const output = d.mapping === 'output';
  const wX = (w: number) => (output ? outX(w) : inX(w));
  const cIn = Math.min(30, pitch(p.nIn) - 4);
  const cOut = Math.min(30, pitch(p.nOut) - 4);
  const rW = Math.max(4, Math.min(11, Math.min(output ? pitch(p.nOut) : pitch(p.nIn)) / 2 - 2));
  const IY = 26, WY = 132, OY = 222, H = 286;
  const label = (i: number): string => {
    if (p.kind === 'hist') return String(p.values[i]);
    if (p.kind === 'spmv') return String(p.rowOf[i]);
    return String(i);
  };
  const dim = (w: number) => (sel != null && sel !== w ? 0.18 : 1);
  const selW = sel != null ? d.workers[sel] : null;
  return (
    <svg viewBox={`0 0 ${DW} ${H}`} role="img" aria-label={tr('Threads, the inputs they read and the outputs they write', 'Các luồng, đầu vào chúng đọc và đầu ra chúng ghi')} style={{ width: '100%', minWidth: 560, display: 'block' }}>
      <text x={8} y={IY + cIn / 2 + 4} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}><Translated vi="vào">in</Translated></text>
      <text x={8} y={IY + cIn / 2 + 18} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>
        {p.kind === 'hist' ? tr('value (bin)', 'giá trị (bin)') : p.kind === 'spmv' ? tr('non-zero, row', 'khác không, hàng') : tr('index', 'chỉ số')}
      </text>
      <text x={8} y={WY + 4} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="luồng">threads</Translated></text>
      <text x={8} y={OY + cOut / 2 + 4} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}><Translated vi="ra">out</Translated></text>
      {/* reads */}
      {d.workers.map((w) => (
        <g key={'r' + w.id} opacity={dim(w.id)}>
          {w.useful.map((i) => (
            <line key={i} x1={inX(i)} y1={IY + cIn} x2={wX(w.id)} y2={WY - rW} stroke="var(--ink-3)" strokeWidth={sel === w.id ? 1.8 : 1} />
          ))}
        </g>
      ))}
      {selW?.inspectsAll
        ? Array.from({ length: p.nIn }, (_, i) =>
            selW.useful.includes(i) ? null : (
              <line key={'x' + i} x1={inX(i)} y1={IY + cIn} x2={wX(selW.id)} y2={WY - rW} stroke="var(--ink-3)" strokeDasharray="2 3" strokeWidth="1" />
            ),
          )
        : null}
      {/* writes */}
      {d.workers.map((w) => (
        <g key={'w' + w.id} opacity={dim(w.id)}>
          {w.writes.map((x) => {
            const bad = d.writers[x.o] > 1;
            return (
              <line key={x.o} x1={wX(w.id)} y1={WY + rW} x2={outX(x.o)} y2={OY - 2} stroke={bad ? 'var(--fail)' : 'var(--ink)'}
                strokeWidth={sel === w.id ? 2.2 : bad ? 1.3 : 1.5} />
            );
          })}
        </g>
      ))}
      <g data-row="in">
        {Array.from({ length: p.nIn }, (_, i) => {
          const used = selW?.useful.includes(i);
          return (
            <g key={i}>
              <rect x={inX(i) - cIn / 2} y={IY} width={cIn} height={cIn} rx="2" fill={used ? 'var(--global)' : 'var(--global-wash)'} stroke="var(--global)" />
              {cIn >= 14 ? (
                <text x={inX(i)} y={IY + cIn / 2 + 4} fontSize="11" textAnchor="middle" style={{ fill: used ? 'var(--page)' : 'var(--ink)' }}>{label(i)}</text>
              ) : null}
            </g>
          );
        })}
      </g>
      <g data-row="workers">
        {d.workers.map((w) => (
          <g key={w.id} data-worker={w.id} role="button" aria-label={tr(`thread ${w.id}`, `luồng ${w.id}`)} style={{ cursor: 'pointer' }} onClick={() => onSel(w.id)}>
            <circle cx={wX(w.id)} cy={WY} r={rW + 3} fill="transparent" />
            <circle cx={wX(w.id)} cy={WY} r={rW} fill={sel === w.id ? 'var(--ink)' : 'var(--page)'} stroke="var(--ink)" strokeWidth="1.3" />
          </g>
        ))}
      </g>
      <g data-row="out" data-writers={d.writers.join(' ')}>
        {d.writers.map((n, o) => {
          const bad = n > 1;
          return (
            <g key={o}>
              <rect x={outX(o) - cOut / 2} y={OY} width={cOut} height={cOut} rx="2" fill={bad ? 'var(--fail-wash)' : 'var(--cell)'} stroke={bad ? 'var(--fail)' : 'var(--rule)'} strokeWidth="1.3" />
              {cOut >= 14 ? <text x={outX(o)} y={OY + cOut / 2 + 4} fontSize="11" textAnchor="middle">{o}</text> : null}
              <text x={outX(o)} y={OY + cOut + 14} fontSize="10.5" fontWeight="600" textAnchor="middle" style={{ fill: bad ? 'var(--fail)' : 'var(--ink-2)' }}>
                {bad ? `${n}×` : '1'}
              </text>
            </g>
          );
        })}
      </g>
      <text x={8} y={OY + cOut + 14} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="số luồng ghi">writers</Translated></text>
    </svg>
  );
}

/** W as N rows of K weights, x above, y at the right. */
function GridDrawing({ d, sel, onSel }: { d: Decomposition; sel: number | null; onSel: (w: number) => void }) {
  const p = d.problem as MatvecProblem;
  const { N, K } = p;
  const L = 34, GX = L, GW = 500, XY = 22, XH = 12, GY = 52;
  const rowH = Math.max(5, Math.min(32, 250 / N));
  const GH = rowH * N;
  const colW = GW / K;
  const YX = GX + GW + 92;
  const H = GY + GH + 40;
  const S = d.opts.S;
  const loaded = warp0Load(p, d.mapping, d.opts).map((a) => a / 4);
  // Contiguous runs of warp 0's elements, [row, first column, last column], drawn as one bar each.
  const loadRuns: [number, number, number][] = [];
  for (const e of [...loaded].sort((a, b) => a - b)) {
    const row = Math.floor(e / K);
    const col = e % K;
    const last = loadRuns[loadRuns.length - 1];
    if (last && last[0] === row && last[2] === col - 1) last[2] = col;
    else loadRuns.push([row, col, col]);
  }
  const region = (w: number): { x: number; y: number; w: number; h: number } => {
    const wk = d.workers[w];
    if (d.mapping === 'input') return { x: GX + w * colW, y: GY, w: Math.max(2, colW), h: GH };
    const row = wk.writes[0].o;
    const first = wk.useful[0] ?? 0;
    return { x: GX + first * colW, y: GY + row * rowH, w: wk.useful.length * colW, h: rowH };
  };
  const arrows = d.workers.reduce((s, w) => s + w.writes.length, 0) <= 64;
  const click = (e: MouseEvent<SVGRectElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const col = Math.min(K - 1, Math.max(0, Math.floor(((e.clientX - box.left) / box.width) * K)));
    const row = Math.min(N - 1, Math.max(0, Math.floor(((e.clientY - box.top) / box.height) * N)));
    onSel(ownerOfWeight(p, d.mapping, d.opts, row, col));
  };
  const yC = (j: number) => GY + j * rowH + rowH / 2;
  const writeLine = (w: number, o: number, strong: boolean) => {
    const r = region(w);
    const x1 = d.mapping === 'input' ? r.x + r.w / 2 : r.x + r.w;
    // S warps of one row leave at different heights so that their arrows visibly converge on y.
    const y1 = d.mapping === 'input' ? GY - 2 : d.mapping === 'splitk' ? r.y + (r.h * ((w % S) + 1)) / (S + 1) : r.y + r.h / 2;
    const bad = d.writers[o] > 1;
    return (
      <line key={`${w}-${o}`} x1={x1} y1={y1} x2={YX - 2} y2={yC(o)} stroke={bad ? 'var(--fail)' : 'var(--ink)'}
        strokeWidth={strong ? 2 : 1} opacity={sel != null && !strong ? 0.25 : 0.9} />
    );
  };
  return (
    <svg viewBox={`0 0 ${DW} ${H}`} role="img" aria-label={tr('The weight matrix, the vector x and the outputs y', 'Ma trận trọng số, vector x và các đầu ra y')} style={{ width: '100%', minWidth: 560, display: 'block' }}>
      <text x={4} y={XY + XH - 2} fontSize="11" fontWeight="600" style={{ fill: 'var(--global)' }}>x</text>
      <rect x={GX} y={XY} width={GW} height={XH} fill="var(--global-wash)" stroke="var(--global)" />
      <text x={GX + GW + 6} y={XY + XH - 2} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>K = {K}</text>
      <text x={4} y={GY + 12} fontSize="11" fontWeight="600" style={{ fill: 'var(--global)' }}>W</text>
      <rect x={GX} y={GY} width={GW} height={GH} fill="var(--global-wash)" stroke="var(--global)" />
      {/* row lines and chunk dividers */}
      {N <= 32 ? Array.from({ length: N - 1 }, (_, j) => (
        <line key={'r' + j} x1={GX} x2={GX + GW} y1={GY + (j + 1) * rowH} y2={GY + (j + 1) * rowH} stroke="var(--global)" strokeWidth="0.5" opacity="0.6" />
      )) : null}
      {d.mapping === 'splitk'
        ? Array.from({ length: S - 1 }, (_, c) => (
            <line key={'c' + c} x1={GX + d.workers[c + 1].useful[0] * colW} x2={GX + d.workers[c + 1].useful[0] * colW} y1={GY} y2={GY + GH} stroke="var(--ink)" strokeWidth="1.2" />
          ))
        : null}
      {sel != null && d.workers[sel] ? (() => {
        const r = region(sel);
        return <rect data-selected-region x={r.x} y={r.y} width={r.w} height={r.h} fill="var(--global)" opacity="0.55" />;
      })() : null}
      {/* warp 0's first load */}
      <g data-loaded={loaded.length}>
        {loadRuns.map(([row, c0, c1]) => (
          <rect key={`${row}-${c0}`} x={GX + c0 * colW} y={GY + row * rowH} width={Math.max(1.5, (c1 - c0 + 1) * colW)} height={rowH} fill="var(--ink)" />
        ))}
      </g>
      <rect x={GX} y={GY} width={GW} height={GH} fill="transparent" style={{ cursor: 'pointer' }} onClick={click} data-grid />
      {/* writes */}
      {arrows
        ? d.workers.flatMap((w) => w.writes.map((x) => writeLine(w.id, x.o, sel === w.id)))
        : sel != null && d.workers[sel] ? d.workers[sel].writes.map((x) => writeLine(sel, x.o, true)) : null}
      <g data-row="out" data-writers={d.writers.join(' ')}>
        {d.writers.map((n, j) => {
          const bad = n > 1;
          return (
            <g key={j}>
              <rect x={YX} y={GY + j * rowH + (rowH > 8 ? 1 : 0)} width={24} height={rowH > 8 ? rowH - 2 : rowH} rx="1.5"
                fill={bad ? 'var(--fail-wash)' : 'var(--cell)'} stroke={bad ? 'var(--fail)' : 'var(--rule)'} />
              {rowH >= 10 || N <= 16 ? (
                <text x={YX + 30} y={yC(j) + 4} fontSize="10" fontWeight="600" style={{ fill: bad ? 'var(--fail)' : 'var(--ink-2)' }}>{bad ? `${fmt(n)}×` : '1'}</text>
              ) : null}
            </g>
          );
        })}
      </g>
      <text x={YX} y={GY - 6} fontSize="11" fontWeight="600" style={{ fill: 'var(--global)' }}>y</text>
      <text x={YX + 30} y={GY - 6} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="số luồng ghi">writers</Translated></text>
      <rect x={GX} y={GY + GH + 14} width={10} height={10} fill="var(--ink)" />
      <text x={GX + 16} y={GY + GH + 23} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="lần nạp W đầu tiên của warp 0">warp 0's first load of W</Translated></text>
      <rect x={GX + 190} y={GY + GH + 14} width={10} height={10} fill="var(--global)" opacity="0.55" />
      <text x={GX + 206} y={GY + GH + 23} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>những gì {d.mapping === 'warp' || d.mapping === 'splitk' ? 'warp' : 'luồng'} được chọn đọc (bấm vào W để chọn)</>}>what the selected {d.mapping === 'warp' || d.mapping === 'splitk' ? 'warp' : 'thread'} reads (click W to choose)</Translated></text>
    </svg>
  );
}

// ---------------------------------------------------------------- explorer
const KIND_LABEL: Record<Kind, string> = {
  get conv() { return tr('1-D convolution', 'tích chập 1 chiều'); },
  get matvec() { return tr('decode GEMV, y = W x', 'GEMV lúc decode, y = W x'); },
  get hist() { return tr('histogram', 'histogram'); },
  get spmv() { return tr('sparse matrix × vector', 'ma trận thưa × vector'); },
};

function Slider({ label, value, min, max, step = 1, on, show }: {
  label: string; value: number; min: number; max: number; step?: number; on: (v: number) => void; show?: string;
}) {
  return (
    <label className="field">
      {label}: {show ?? value}
      <input type="range" min={min} max={max} step={step} value={value} aria-label={label} onChange={(e) => on(Number(e.target.value))} />
    </label>
  );
}

function Explorer() {
  const [s, setS] = useState<ExplorerState>(PRESETS[0].state);
  const [preset, setPreset] = useState<string | null>(PRESETS[0].id);
  const [sel, setSel] = useState<number | null>(null);
  const [widthKey, setWidthKey] = useState('nv');
  const widths = useWidthChoices();
  const d = useMemo(() => decomposeState(s), [s]);
  const m = d.metrics;
  const set = (next: ExplorerState) => {
    setS(next);
    setPreset(null);
    setSel(null);
  };
  const pick = (id: string) => {
    const p = PRESETS.find((x) => x.id === id)!;
    setS(p.state);
    setPreset(id);
    setSel(null);
    setWidthKey('nv');
  };
  const mv = s.matvec;
  const choices = splitChoices(mv.K, mv.warpWidth);
  const access = s.kind === 'matvec' ? warpAccess(warp0Load(d.problem as MatvecProblem, d.mapping, d.opts), 4, SECTOR) : null;
  const accessRuns = s.kind === 'matvec' ? byteRuns(warp0Load(d.problem as MatvecProblem, d.mapping, d.opts), 4) : null;
  const unit = d.mapping === 'warp' || d.mapping === 'splitk' ? 'warp' : 'thread';
  const selected = sel != null && sel < d.workers.length ? sel : null;
  const row = (crit: string, what: ReactNode, key: string, value: ReactNode, bad = false) => (
    <tr>
      <td className="muted">{crit}</td>
      <td>{what}</td>
      <td className="num" data-metric={key} style={bad ? { color: 'var(--fail)', fontWeight: 600 } : undefined}>{value}</td>
    </tr>
  );
  return (
    <Bench
      title={tr('Who writes, who conflicts', 'Ai ghi, ai xung đột')}
      task={tr('Pick a computation and a cut. Which criteria does each cut win?', 'Chọn một phép tính và một cách cắt. Mỗi cách cắt thắng ở tiêu chí nào?')}
      foot={
        <Translated
          vi={
            <>
              Đỏ: một đầu ra được ghi bởi nhiều hơn một {unit === 'warp' ? 'warp' : 'luồng'}, cần một phép toán nguyên tử
              (atomic) cho mỗi lần cập nhật. Mũi tên vào là đọc, mũi tên ra là ghi. Cân bằng tải đếm việc hữu ích trên mỗi
              luồng (một lane, với các cách cắt theo warp). <TierLegend tiers={['reg', 'global']} />
            </>
          }
        >
          <>
            Red: an output written by more than one {unit}, which needs an atomic for every update. Arrows in are reads, arrows out
            are writes. Load balance counts useful work per thread (a lane, for the warp cuts). <TierLegend tiers={['reg', 'global']} />
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="muted"><Translated vi="Thiết lập sẵn">Presets</Translated></span>
        <span className="seg" role="group" aria-label={tr('Presets', 'Thiết lập sẵn')} style={{ flexWrap: 'wrap' }}>
          {PRESETS.map((p) => (
            <button key={p.id} aria-pressed={preset === p.id} onClick={() => pick(p.id)}><Translated vi={({ 'conv-scatter': 'Tích chập, kiểu scatter', 'hist-bins': 'Histogram, mỗi bin một luồng', 'spmv-coo': 'Ma trận thưa lệch, COO', 'gemv-narrow': 'GEMV lúc decode hẹp, split-K' } as Record<string, string>)[p.id]}>{p.label}</Translated></button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label={tr('Computation', 'Phép tính')} style={{ flexWrap: 'wrap' }}>
          {(Object.keys(KIND_LABEL) as Kind[]).map((k) => (
            <button key={k} aria-pressed={s.kind === k} onClick={() => set({ ...s, kind: k, mapping: mappingsOf(k).includes(s.mapping) ? s.mapping : 'output' })}>{KIND_LABEL[k]}</button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label={tr('Decomposition', 'Cách phân rã')} style={{ flexWrap: 'wrap' }}>
          {mappingsOf(s.kind).map((mp) => (
            <button key={mp} aria-pressed={d.mapping === mp} onClick={() => set({ ...s, mapping: mp })}><Translated vi={({ 'conv:output': 'lấy đầu ra làm trung tâm (gather)', 'conv:input': 'lấy đầu vào làm trung tâm (scatter)', 'hist:output': 'lấy đầu ra làm trung tâm: mỗi bin một luồng', 'hist:input': 'lấy đầu vào làm trung tâm: mỗi đầu vào một luồng', 'spmv:output': 'lấy đầu ra làm trung tâm: mỗi hàng một luồng (CSR)', 'spmv:input': 'lấy đầu vào làm trung tâm: mỗi phần tử khác không một luồng (COO)', 'matvec:input': 'lấy đầu vào làm trung tâm: mỗi đầu vào một luồng', 'matvec:output': 'mỗi đầu ra một luồng', 'matvec:warp': 'mỗi đầu ra một warp', 'matvec:splitk': 'chia chiều đầu vào: S warp cho mỗi đầu ra' } as Record<string, string>)[`${s.kind}:${mp}`]}>{MAPPING_LABEL[s.kind][mp]}</Translated></button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        {s.kind === 'conv' ? (
          <>
            <Slider label={tr('Inputs n', 'Số đầu vào n')} value={s.conv.n} min={6} max={16} on={(v) => set({ ...s, conv: { ...s.conv, n: v } })} />
            <label className="field">
              <Translated vi="Độ rộng bộ lọc">Filter width</Translated>
              <select value={s.conv.w} onChange={(e) => set({ ...s, conv: { ...s.conv, w: Number(e.target.value) } })}>
                {[3, 5, 7].map((w) => <option key={w} value={w}>{w}</option>)}
              </select>
            </label>
          </>
        ) : null}
        {s.kind === 'hist' ? (
          <>
            <Slider label={tr('Inputs n', 'Số đầu vào n')} value={s.hist.n} min={8} max={24} step={4} on={(v) => set({ ...s, hist: { ...s.hist, n: v } })} />
            <Slider label={tr('Bins', 'Số bin')} value={s.hist.bins} min={2} max={8} on={(v) => set({ ...s, hist: { ...s.hist, bins: v } })} />
            <Slider label={tr('Skew to bin 0', 'Lệch về bin 0')} value={s.hist.skew} min={0} max={1} step={0.25} show={`${Math.round(s.hist.skew * 100)}%`} on={(v) => set({ ...s, hist: { ...s.hist, skew: v } })} />
          </>
        ) : null}
        {s.kind === 'spmv' ? (
          <>
            <Slider label={tr('Rows', 'Số hàng')} value={s.spmv.rows} min={4} max={8} on={(v) => set({ ...s, spmv: { ...s.spmv, rows: v } })} />
            <Slider label={tr('Extra non-zeros in row 0', 'Số phần tử khác không thêm ở hàng 0')} value={s.spmv.heavy} min={0} max={12} step={2} on={(v) => set({ ...s, spmv: { ...s.spmv, heavy: v } })} />
          </>
        ) : null}
        {s.kind === 'matvec' ? (
          <>
            <label className="field">
              <Translated vi="Số đầu ra N">Outputs N</Translated>
              <select value={mv.N} onChange={(e) => set({ ...s, matvec: { ...mv, N: Number(e.target.value) } })}>
                {[1, 2, 4, 8, 16, 32].map((v) => <option key={v} value={v}>{v}</option>)}
              </select>
            </label>
            <label className="field">
              <Translated vi="Số đầu vào K">Inputs K</Translated>
              <select value={mv.K} onChange={(e) => {
                const K = Number(e.target.value);
                const ok = splitChoices(K, mv.warpWidth);
                set({ ...s, matvec: { ...mv, K, S: ok.includes(mv.S) ? mv.S : ok[ok.length - 1] } });
              }}>
                {[64, 128, 256, 512].map((v) => <option key={v} value={v}>{v}</option>)}
              </select>
            </label>
            <label className="field">
              <Translated vi="Độ rộng warp">Warp width</Translated>
              <select value={widthKey} onChange={(e) => {
                const c = widths.find((x) => x.key === e.target.value)!;
                setWidthKey(c.key);
                const ok = splitChoices(mv.K, c.width);
                set({ ...s, matvec: { ...mv, warpWidth: c.width, S: ok.includes(mv.S) ? mv.S : ok[ok.length - 1] } });
              }}>
                {widths.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
              </select>
            </label>
            {d.mapping === 'splitk' ? (
              <label className="field">
                <Translated vi="Số đoạn S">Chunks S</Translated>
                <select value={mv.S} onChange={(e) => set({ ...s, matvec: { ...mv, S: Number(e.target.value) } })}>
                  {choices.map((v) => <option key={v} value={v}>{v}</option>)}
                </select>
              </label>
            ) : null}
          </>
        ) : null}
        <label className="field">
          <Translated vi={<>Hiện {unit === 'warp' ? 'warp' : 'luồng'}</>}>Show {unit}</Translated>
          <input type="number" min={0} max={d.workers.length - 1} value={selected ?? ''} placeholder={tr('none', 'không')} aria-label={tr(`Show ${unit}`, `Hiện ${unit === 'warp' ? 'warp' : 'luồng'}`)}
            onChange={(e) => setSel(e.target.value === '' ? null : Math.max(0, Math.min(d.workers.length - 1, Number(e.target.value))))} style={{ width: '5.5rem' }} />
        </label>
      </div>

      <div style={{ maxWidth: 880, margin: '0 auto 0.8rem', overflowX: 'auto' }}>
        {s.kind === 'matvec' ? <GridDrawing d={d} sel={selected} onSel={setSel} /> : <Bipartite d={d} sel={selected} onSel={setSel} />}
      </div>
      <div className="cols">
        <div className="narrate" role="status" data-mapping-text>
          {selected != null
            ? describeWorker(d, selected)
            : tr(
                `Click a ${unit}${s.kind === 'matvec' ? ' (a cell of W)' : ''} to see its index mapping. ${fmt(m.workers)} ${unit}s in all.`,
                `Bấm vào một ${unit === 'warp' ? 'warp' : 'luồng'}${s.kind === 'matvec' ? ' (một ô của W)' : ''} để xem cách ánh xạ chỉ số của nó. Tổng cộng ${fmt(m.workers)} ${unit === 'warp' ? 'warp' : 'luồng'}.`,
              )}
        </div>
        <div>
          <table className="data" style={{ background: 'var(--page)', margin: 0 }} aria-label={tr("The book's criteria for this cut", 'Các tiêu chí của sách cho cách cắt này')}>
            <thead>
              <tr><th style={{ paddingLeft: '0.5rem' }}><Translated vi="Tiêu chí (p. 533)">Criterion (p. 533)</Translated></th><th><Translated vi="Thước đo">Measure</Translated></th><th className="num"><Translated vi="Giá trị">Value</Translated></th></tr>
            </thead>
            <tbody>
              {row(tr('Parallelism', 'Mức song song'), tr('threads launched', 'số luồng khởi chạy'), 'threads', fmt(m.threads))}
              {row(tr('Gather or scatter', 'Gather hay scatter'), tr('outputs that need atomics', 'số đầu ra cần atomic'), 'outputs-atomic', tr(`${fmt(m.outputsNeedingAtomics)} of ${fmt(d.problem.nOut)}`, `${fmt(m.outputsNeedingAtomics)} trên ${fmt(d.problem.nOut)}`), m.outputsNeedingAtomics > 0)}
              {row('', tr('atomic updates', 'số lần cập nhật atomic'), 'atomics', fmt(m.atomicUpdates), m.atomicUpdates > 0)}
              {row('', tr('most writers on one output', 'số luồng ghi nhiều nhất trên một đầu ra'), 'worst', fmt(m.worstWriters), m.worstWriters > 1)}
              {row('', <Translated vi={<>số phép cộng trong <span className="t-reg">thanh ghi</span></>}><>additions in <span className="t-reg">registers</span></></Translated>, 'registers', fmt(m.registerAccumulations))}
              {row(tr('Ease of mapping', 'Độ dễ ánh xạ'), tr(`inputs the busiest ${unit} inspects`, `số đầu vào mà ${unit === 'warp' ? 'warp' : 'luồng'} bận nhất xem xét`), 'inspected-max', fmt(m.inspectedMax))}
              {row('', tr(`inputs inspected in all, for ${fmt(m.contributions)} contributions`, `tổng số đầu vào được xem xét, cho ${fmt(m.contributions)} đóng góp`), 'inspected', fmt(m.inspectedTotal), m.inspectedTotal > m.contributions)}
              {row(tr('Load balance', 'Cân bằng tải'), tr('useful work per thread, max ÷ mean', 'việc hữu ích mỗi luồng, max ÷ trung bình'), 'imbalance', fmt2(m.imbalance), m.imbalance > 1.5)}
              {access
                ? row(tr('Coalescing (ch. 6)', 'Hợp nhất truy cập (ch. 6)'), <Translated vi={<>số sector trong lần nạp <span className="t-global">W</span> đầu tiên của warp 0 (đơn vị {SECTOR} byte)</>}><>sectors in warp 0's first load of <span className="t-global">W</span> ({SECTOR}-byte unit)</></Translated>, 'sectors',
                    tr(`${access.transactions}${accessRuns === 1 ? ', one run' : ''}`, `${access.transactions}${accessRuns === 1 ? ', một đoạn liền' : ''}`), access.transactions > access.bytesNeeded / SECTOR + 1)
                : null}
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- classification
type Answer = { decomp?: Decomp; crit?: Criterion; checked?: boolean };

function SortTheTour() {
  const [k, setK] = useState(0);
  const [ans, setAns] = useState<Record<string, Answer>>({});
  const row = TOUR[k];
  const a = ans[row.id] ?? {};
  const grade = a.checked && a.decomp && a.crit ? gradeCard(row, a.decomp, a.crit) : null;
  const put = (x: Answer) => setAns({ ...ans, [row.id]: { ...a, ...x, checked: false } });
  const verdictOf = (id: string): 'right' | 'partly' | 'wrong' | '' => {
    const r = TOUR.find((t) => t.id === id)!;
    const x = ans[id];
    if (!x?.checked || !x.decomp || !x.crit) return '';
    const g = gradeCard(r, x.decomp, x.crit);
    return g.decomp && g.criterion ? 'right' : g.decomp || g.criterion ? 'partly' : 'wrong';
  };
  const done = TOUR.filter((t) => verdictOf(t.id));
  const right = done.filter((t) => verdictOf(t.id) === 'right').length;
  const v = verdictOf(row.id);
  return (
    <Bench
      title={tr("Sort the book's tour", 'Phân loại chuyến tham quan của sách')}
      task={tr('For each card: which decomposition, and what decided it?', 'Với mỗi thẻ: cách phân rã nào, và điều gì quyết định nó?')}
      foot={
        <Translated vi={<>Mười bảy thẻ từ pp. 533–536 và bốn thẻ của khóa học này, có đánh dấu. Một thẻ có thể đúng với nhiều tiêu chí; phần phản hồi sẽ nói là những tiêu chí nào. Câu trả lời của bạn ở đây không được ghi lại.</>}>
          <>Seventeen cards from pp. 533–536 and four of ours, marked. Several criteria can be right for one card; the feedback says which. Your answers here are not recorded.</>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', gap: '0.3rem' }} aria-label={tr('Cards', 'Các thẻ')}>
        {TOUR.map((t, i) => {
          const vv = verdictOf(t.id);
          return (
            <button key={t.id} aria-pressed={i === k} onClick={() => setK(i)} title={t.name} data-dot={t.id} data-verdict={vv}
              style={{ padding: '0.1rem 0.45rem', minWidth: '2rem', borderColor: vv === 'right' ? 'var(--ok)' : vv ? 'var(--fail)' : undefined, color: vv === 'right' ? 'var(--ok)' : vv ? 'var(--fail)' : undefined }}>
              {i + 1}
            </button>
          );
        })}
      </div>
      <div className="panel" data-card={row.id} data-verdict={v}>
        <p style={{ margin: '0 0 0.6rem', fontSize: '1rem' }}>
          <b>{row.name}</b>{' '}
          <span className="muted">{row.ours ? '' : tr(`chapter ${row.chapter}`, `chương ${row.chapter}`)}</span>{' '}
          {row.ours ? <span className="chip global"><Translated vi="của khóa học này, không có trong sách">ours, not in the book</Translated></span> : null}
        </p>
        <div className="row" style={{ marginBottom: '0.6rem' }}>
          <span className="muted" style={{ minWidth: '7.5rem' }}><Translated vi="Cách phân rã">Decomposition</Translated></span>
          <span className="seg" role="group" aria-label={tr('Decomposition', 'Cách phân rã')} style={{ flexWrap: 'wrap' }}>
            {(Object.keys(DECOMP_LABEL) as Decomp[]).map((x) => (
              <button key={x} aria-pressed={a.decomp === x} onClick={() => put({ decomp: x })}><Translated vi={({ output: 'lấy đầu ra làm trung tâm', input: 'lấy đầu vào làm trung tâm', either: 'cách nào cũng như nhau' } as Record<Decomp, string>)[x]}>{DECOMP_LABEL[x]}</Translated></button>
            ))}
          </span>
        </div>
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <span className="muted" style={{ minWidth: '7.5rem' }}><Translated vi="Điều quyết định">What decided it</Translated></span>
          <span className="seg" role="group" aria-label={tr('What decided it', 'Điều quyết định')} style={{ flexWrap: 'wrap' }}>
            {(Object.keys(CRITERION_LABEL) as Criterion[]).map((x) => (
              <button key={x} aria-pressed={a.crit === x} onClick={() => put({ crit: x })}><Translated vi={({ atomics: 'tránh atomic', parallelism: 'lộ ra song song', mapping: 'dễ tìm đầu vào', balance: 'cân bằng tải', none: 'cách nào cũng không cần atomic' } as Record<Criterion, string>)[x]}>{CRITERION_LABEL[x]}</Translated></button>
            ))}
          </span>
        </div>
        <div className="row">
          <button className="primary" disabled={!a.decomp || !a.crit} onClick={() => setAns({ ...ans, [row.id]: { ...a, checked: true } })}><Translated vi="Kiểm tra">Check</Translated></button>
          <button onClick={() => setK((k + TOUR.length - 1) % TOUR.length)}><Translated vi="Thẻ trước">Previous card</Translated></button>
          <button onClick={() => setK((k + 1) % TOUR.length)}><Translated vi="Thẻ sau">Next card</Translated></button>
          <span className="muted" aria-live="polite" data-score={right}><Translated vi={<>đã kiểm tra {done.length} trên {TOUR.length}, {right} đúng hoàn toàn</>}>{done.length} of {TOUR.length} checked, {right} fully right</Translated></span>
        </div>
        {grade ? (
          <div className="narrate" style={{ marginTop: '0.8rem' }} data-feedback>
            <p style={{ margin: '0 0 0.4rem' }}>
              <span className={'chip ' + (grade.decomp ? 'ok' : 'fail')}>{grade.decomp ? tr('decomposition right', 'cách phân rã đúng') : tr(`it is ${DECOMP_LABEL[row.decomp]}`, `đó là ${({ output: 'lấy đầu ra làm trung tâm', input: 'lấy đầu vào làm trung tâm', either: 'cách nào cũng như nhau' } as Record<Decomp, string>)[row.decomp]}`)}</span>{' '}
              <span className={'chip ' + (grade.criterion ? 'ok' : 'fail')}>
                {grade.criterion ? tr('criterion right', 'tiêu chí đúng') : tr(`decided by: ${row.accept.map((c) => CRITERION_LABEL[c]).join(', or ')}`, `quyết định bởi: ${row.accept.map((c) => ({ atomics: 'tránh atomic', parallelism: 'lộ ra song song', mapping: 'dễ tìm đầu vào', balance: 'cân bằng tải', none: 'cách nào cũng không cần atomic' } as Record<Criterion, string>)[c]).join(', hoặc ')}`)}
              </span>
            </p>
            <p style={{ margin: 0 }}>
              {row.why} <span className="muted">{row.ours ? tr(`By the book's criteria, p. ${row.pages}.`, `Theo các tiêu chí của sách, p. ${row.pages}.`) : tr(`Book pp. ${row.pages}.`, `Sách pp. ${row.pages}.`)}</span>
            </p>
          </div>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- split-K stepper
const SK_S = [1, 2, 4];
const SK_ORDERS: Interleave[] = ['sequential', 'round-robin', 'reversed'];

function SplitKStepper() {
  const [S, setS] = useState(2);
  const [atomics, setAtomics] = useState(true);
  const [order, setOrder] = useState<Interleave>('sequential');
  const [lane, setLane] = useState(1);
  const [n, setN] = useState(0);
  const [playing, setPlaying] = useState(false);
  const cfg: SplitKConfig = useMemo(() => ({ ...SPLITK_TINY, S, atomics }), [S, atomics]);
  const run = useMemo(() => splitKRun(cfg, splitKOrder(cfg, order)), [cfg, order]);
  const total = run.steps.length;
  const at = Math.min(n, total);
  const snap = run.snaps[at];
  const cur = at > 0 ? run.steps[at - 1] : undefined;
  const firstLost = run.steps.findIndex((s) => s.lost);
  const traffic = splitKTraffic(cfg);
  const { batch, K, N, warpWidth: ww } = cfg;

  useEffect(() => setN(0), [S, atomics, order]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) return setPlaying(false);
    const id = window.setTimeout(() => setN((x) => x + 1), 380);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);
  const nextCommit = () => {
    for (let i = at; i < total; i++) if (['store', 'atomic', 'read', 'write'].includes(run.steps[i].op.kind)) return setN(i + 1);
    setN(total);
  };

  // Drawing.
  const CWD = 30, X0 = 70, XY = 30, WY = XY + batch * CWD + 26, YX = X0 + K * CWD + 40;
  const cw = cur ? cur.warp : -1;
  const place = cw >= 0 ? warpPlace(cfg, cw) : null;
  const loadK = cur?.ks ?? [];
  const YY = XY + 20; // first row of Y
  const YP = 54; // pitch of Y's rows
  const H = Math.max(WY + N * CWD + 20 + batch * 26, YY + batch * YP + 8 + Math.ceil((N * S) / 4) * 22) + 6;
  const wv = (j: number, k: number) => run.data.W[j * K + k];
  const xv = (b: number, k: number) => run.data.X[b * K + k];
  const VW = YX + N * (CWD + 16) + 20;
  return (
    <Bench
      title={tr('Split-K, warp by warp', 'Split-K, từng warp một')}
      task={tr(
        'Step with S = 1, 2 and 4. Then commit with a plain += and run all warps together.',
        'Chạy từng bước với S = 1, 2 và 4. Sau đó ghi kết quả bằng một phép += thường và chạy mọi warp cùng lúc.',
      )}
      foot={
        <Translated
          vi={
            <>
              Batch {batch}, {K} đầu vào, {N} đầu ra, một warp đồ chơi gồm {ww} lane. Các giá trị là số nguyên nhỏ nên mọi tổng đều chính xác. <TierLegend tiers={['reg', 'global']} />
            </>
          }
        >
          <>
            Batch {batch}, {K} inputs, {N} outputs, a toy warp of {ww} lanes. Values are small integers so every sum is exact. <TierLegend tiers={['reg', 'global']} />
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          <Translated vi="Số đoạn S">Chunks S</Translated>
          <span className="seg" role="group" aria-label={tr('Chunks S', 'Số đoạn S')} style={{ flexWrap: 'wrap' }}>
            {SK_S.map((v) => <button key={v} aria-pressed={S === v} onClick={() => setS(v)}>{v}</button>)}
          </span>
        </label>
        <label className="field">
          <Translated vi="Lane 0 ghi kết quả bằng">Lane 0 commits with</Translated>
          <span className="seg" role="group" aria-label={tr('Commit', 'Ghi kết quả')} style={{ flexWrap: 'wrap' }}>
            <button aria-pressed={atomics} onClick={() => setAtomics(true)}><Translated vi="một phép cộng atomic">an atomic add</Translated></button>
            <button aria-pressed={!atomics} onClick={() => setAtomics(false)}><Translated vi="một phép += thường">a plain +=</Translated></button>
          </span>
        </label>
        <label className="field">
          <Translated vi="Thứ tự warp">Warp order</Translated>
          <select value={order} onChange={(e) => setOrder(e.target.value as Interleave)} aria-label={tr('Warp order', 'Thứ tự warp')} style={{ maxWidth: '100%' }}>
            {SK_ORDERS.map((o) => <option key={o} value={o}>{tr(INTERLEAVE_LABEL[o], ({ sequential: 'mỗi lần một warp', 'round-robin': 'mọi warp cùng lúc, lần lượt', reversed: 'mỗi lần một warp, warp cuối trước' } as Record<Interleave, string>)[o])}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Lane cần theo dõi">Lane to follow</Translated>
          <select value={lane} onChange={(e) => setLane(Number(e.target.value))} aria-label={tr('Lane to follow', 'Lane cần theo dõi')}>
            {Array.from({ length: ww }, (_, l) => <option key={l} value={l}>{l}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <button onClick={() => { setPlaying(false); setN(0); }} disabled={at === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => { setPlaying(false); setN(at - 1); }} disabled={at === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => { setPlaying(false); setN(at + 1); }} disabled={at >= total}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => { setPlaying(false); nextCommit(); }} disabled={at >= total}><Translated vi="Chạy đến lần ghi kết quả kế tiếp">Run to the next commit</Translated></button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? tr('Pause', 'Tạm dừng') : tr('Play', 'Chạy')}</button>
        {firstLost >= 0 ? <button onClick={() => { setPlaying(false); setN(firstLost + 1); }}><Translated vi="Đến lần cập nhật bị mất đầu tiên">Go to the first lost update</Translated></button> : null}
        <span className="muted" aria-live="polite" data-step={at}><Translated vi={<>bước {at} trên {total}</>}>step {at} of {total}</Translated></span>
      </div>
      <div className={'narrate' + (cur?.lost ? ' bad' : '')} role="status" style={{ marginBottom: '0.8rem' }} data-sk-text>
        {cur ? describeSKStep(run, at - 1, lane) : (
          <Translated
            vi={
              <>
                Chưa có gì chạy. {N * S} warp:{' '}
                {S === 1 ? <>warp g sở hữu đầu ra g và đọc cả {K} đầu vào của hàng g.</> : <>warp g sở hữu đầu ra g ÷ {S} và đoạn g mod {S} của {K} đầu vào, như trong <code>gemvSplitKKernel</code> của lab 22.1.</>}{' '}
                Nhấn Bước.
              </>
            }
          >
            <>
              Nothing has run yet. {N * S} warps:{' '}
              {S === 1 ? <>warp g owns output g and reads all {K} inputs of row g.</> : <>warp g owns output g ÷ {S} and chunk g mod {S} of the {K} inputs, as in lab 22.1's <code>gemvSplitKKernel</code>.</>}{' '}
              Press Step.
            </>
          </Translated>
        )}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${VW} ${H}`} role="img" aria-label={tr("x, W, the current warp's lanes and Y", 'x, W, các lane của warp hiện tại và Y')} style={{ width: '100%', minWidth: 560, display: 'block' }}>
          {/* x rows */}
          {Array.from({ length: batch }, (_, b) => (
            <g key={'x' + b}>
              <text x={X0 - 8} y={XY + b * CWD + 19} fontSize="12" textAnchor="end" style={{ fill: 'var(--global)' }}>x[{b}]</text>
              {Array.from({ length: K }, (_, k) => {
                const hot = cur && (cur.op.kind === 'load' || cur.op.kind === 'fma') && loadK.includes(k);
                return (
                  <g key={k}>
                    <rect x={X0 + k * CWD} y={XY + b * CWD} width={CWD - 2} height={CWD - 2} rx="2" fill={hot ? 'var(--global)' : 'var(--global-wash)'} stroke="var(--global)" strokeWidth="0.8" />
                    <text x={X0 + k * CWD + CWD / 2 - 1} y={XY + b * CWD + 19} fontSize="11" textAnchor="middle" style={{ fill: hot ? 'var(--page)' : 'var(--ink)' }}>{xv(b, k)}</text>
                  </g>
                );
              })}
            </g>
          ))}
          {Array.from({ length: K }, (_, k) => (
            <text key={'k' + k} x={X0 + k * CWD + CWD / 2 - 1} y={XY - 6} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{k}</text>
          ))}
          {/* W */}
          {Array.from({ length: N }, (_, j) => (
            <g key={'w' + j}>
              <text x={X0 - 8} y={WY + j * CWD + 19} fontSize="12" textAnchor="end" style={{ fill: 'var(--global)' }}>W[{j}]</text>
              {Array.from({ length: K }, (_, k) => {
                const mine = place && place.j === j && k >= place.start && k < place.start + place.len;
                const hot = cur && (cur.op.kind === 'load' || cur.op.kind === 'fma') && place?.j === j && loadK.includes(k);
                const isLane = hot && loadK[lane] === k;
                return (
                  <g key={k}>
                    <rect x={X0 + k * CWD} y={WY + j * CWD} width={CWD - 2} height={CWD - 2} rx="2"
                      fill={hot ? 'var(--global)' : mine ? 'var(--global-wash)' : 'var(--cell)'} stroke={isLane ? 'var(--ink)' : 'var(--global)'} strokeWidth={isLane ? 2.2 : 0.8} />
                    <text x={X0 + k * CWD + CWD / 2 - 1} y={WY + j * CWD + 19} fontSize="11" textAnchor="middle" style={{ fill: hot ? 'var(--page)' : 'var(--ink)' }}>{wv(j, k)}</text>
                  </g>
                );
              })}
            </g>
          ))}
          {/* chunk dividers */}
          {Array.from({ length: S - 1 }, (_, c) => {
            const x = X0 + ((c + 1) * K / S) * CWD - 1;
            return <line key={'d' + c} x1={x} x2={x} y1={WY - 6} y2={WY + N * CWD + 4} stroke="var(--ink)" strokeWidth="2" strokeDasharray="4 3" />;
          })}
          {/* the current warp's lanes */}
          <text x={X0 - 8} y={WY + N * CWD + 30} fontSize="11.5" textAnchor="end" style={{ fill: 'var(--reg)' }}>{cw >= 0 ? tr(`warp ${cw}`, `warp ${cw}`) : tr('lanes', 'các lane')}</text>
          {Array.from({ length: batch }, (_, b) => (
            <g key={'acc' + b} data-acc={b}>
              <text x={X0 + ww * 44 + 10} y={WY + N * CWD + 34 + b * 26} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>tổng tích lũy cho x[{b}]</>}>running sum for x[{b}]</Translated></text>
              {Array.from({ length: ww }, (_, l) => (
                <g key={l}>
                  <rect x={X0 + l * 44} y={WY + N * CWD + 16 + b * 26} width={40} height={22} rx="2" fill="var(--reg-wash)" stroke={l === lane ? 'var(--ink)' : 'var(--reg)'} strokeWidth={l === lane ? 1.8 : 1} />
                  <text x={X0 + l * 44 + 20} y={WY + N * CWD + 31 + b * 26} fontSize="11" textAnchor="middle" style={{ fill: 'var(--reg)' }}>{cw >= 0 ? snap.acc[cw][b][l] : 0}</text>
                </g>
              ))}
            </g>
          ))}
          {Array.from({ length: ww }, (_, l) => (
            <text key={'ln' + l} x={X0 + l * 44 + 20} y={WY + N * CWD + 12} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}><Translated vi={<>lane {l}</>}>lane {l}</Translated></text>
          ))}
          {/* Y */}
          <text x={YX} y={XY - 6} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}><Translated vi="Y (bộ nhớ toàn cục)">Y (global memory)</Translated></text>
          {Array.from({ length: N }, (_, j) => (
            <text key={'yj' + j} x={YX + j * (CWD + 16) + (CWD + 10) / 2} y={XY + 14} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>j {j}</text>
          ))}
          {Array.from({ length: batch }, (_, b) => (
            <g key={'y' + b}>
              {Array.from({ length: N }, (_, j) => {
                const cell = b * N + j;
                const touched = cur?.touched?.includes(cell);
                const complete = snap.parts[cell].length === S && snap.Y[cell] === run.data.ref[cell];
                const bad = at >= total && snap.Y[cell] !== run.data.ref[cell];
                const x = YX + j * (CWD + 16);
                const y = YY + b * YP;
                return (
                  <g key={j} data-y={`${b},${j}`} data-value={snap.Y[cell]} data-parts={snap.parts[cell].join(',')}>
                    <rect x={x} y={y} width={CWD + 10} height={CWD} rx="2" fill={bad ? 'var(--fail-wash)' : complete ? 'var(--ok-wash)' : 'var(--cell)'}
                      stroke={bad ? 'var(--fail)' : touched ? 'var(--ink)' : complete ? 'var(--ok)' : 'var(--global)'} strokeWidth={touched ? 2.2 : 1} />
                    <text x={x + (CWD + 10) / 2} y={y + 20} fontSize="12" textAnchor="middle">{snap.Y[cell]}</text>
                    <text x={x + (CWD + 10) / 2} y={y + CWD + 12} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
                      {snap.parts[cell].length ? 'c ' + snap.parts[cell].join(',') : '·'}
                    </text>
                  </g>
                );
              })}
              <text x={YX - 6} y={YY + b * YP + 20} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>b {b}</text>
            </g>
          ))}
          {/* warp progress */}
          <g>
            {Array.from({ length: N * S }, (_, w) => {
              const total = warpOps(cfg, w).length;
              const done = snap.done[w];
              const x = YX + (w % 4) * 34;
              const y = YY + batch * YP + 8 + Math.floor(w / 4) * 22;
              return (
                <g key={w}>
                  <rect x={x} y={y} width={30} height={16} rx="2" fill={done === total ? 'var(--rule-soft)' : 'var(--page)'} stroke={w === cw ? 'var(--ink)' : 'var(--rule)'} strokeWidth={w === cw ? 2 : 1} />
                  <text x={x + 15} y={y + 12} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>w{w}</text>
                </g>
              );
            })}
          </g>
        </svg>
      </div>
      <div className="row" style={{ marginTop: '0.6rem' }}>
        <span data-sk-threads={traffic.threads}><Translated vi={<>{traffic.threads} luồng</>}>{traffic.threads} threads</Translated></span>
        <span data-sk-atomics={traffic.atomics}>
          {S === 1
            ? tr('no shared outputs: plain stores', 'không có đầu ra dùng chung: ghi thường')
            : atomics
              ? tr(`${traffic.atomics} atomic updates`, `${traffic.atomics} lần cập nhật atomic`)
              : <b className="t-fail"><Translated vi={<>{traffic.atomics} lần đọc-cộng-ghi không được bảo vệ trên các đầu ra dùng chung</>}>{traffic.atomics} unprotected read-add-writes on shared outputs</Translated></b>}
        </span>
        <span className="muted"><Translated vi={<>W đọc: {traffic.wBytes} byte; x đọc: {traffic.xBytes} byte, bất kể S là bao nhiêu</>}>W read: {traffic.wBytes} bytes; x read: {traffic.xBytes} bytes, whatever S is</Translated></span>
        {at >= total ? (
          run.correct
            ? <span className="chip ok" data-result="right"><Translated vi="Y khớp với kết quả tham chiếu">Y matches the reference</Translated></span>
            : <span className="chip fail" data-result="wrong"><Translated vi={<>Y sai: mất {run.lostCount} đóng góp</>}>Y is wrong: {run.lostCount} contributions lost</Translated></span>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS222: Record<string, Question> = {
  'ch22/22.2/q-conv-writers': {
    kind: 'numeric',
    id: 'ch22/22.2/q-conv-writers',
    prompt: (
      <Translated
        vi={
          <>
            Một phép tích chập 1 chiều trên 20 đầu vào với bộ lọc rộng 7, được đệm số không để có 20 đầu ra, được phân rã theo
            kiểu lấy đầu vào làm trung tâm: mỗi luồng sở hữu một đầu vào và cộng các đóng góp của nó vào các đầu ra. Tổng cộng
            nó thực hiện bao nhiêu lần cập nhật atomic? (Trình khám phá dừng ở 16 đầu vào; hãy dùng nó để kiểm tra cách làm của
            bạn, không phải đáp án.)
          </>
        }
      >
        <>
          A 1-D convolution of 20 inputs with a 7-wide filter, zero-padded so that there are 20 outputs, is decomposed
          input-centric: each thread owns one input and adds its contributions into the outputs. How many atomic updates
          does it make in all? (The explorer stops at 16 inputs; use it to check your method, not the answer.)
        </>
      </Translated>
    ),
    answer: 128,
    explain: (
      <Translated
        vi={
          <p>
            Đếm các đóng góp theo đầu ra. 14 đầu ra bên trong mỗi đầu ra nhận 7, tức 98. Gần mỗi biên, phần đệm bỏ đi 3, 2 và 1
            đóng góp, nên 3 đầu ra ở biên mỗi bên nhận 4, 5 và 6: 15 mỗi bên, tổng cộng 30. Vậy là 128 đóng góp. Mỗi đầu ra có
            ít nhất 4 luồng ghi, nên cả 128 đều là atomic. Với cách lấy đầu ra làm trung tâm, mỗi đầu ra có một luồng ghi, cộng
            các tích trong một thanh ghi (register) và ghi một lần: 20 lần ghi thường, không có atomic. Thiết lập sẵn tích chập
            của trình khám phá, 12 đầu vào và rộng 5, được đếm cùng cách: 54.
          </p>
        }
      >
        <p>
          Count the contributions by output. The 14 inner outputs each receive 7, which is 98. Near each edge the padding
          removes 3, 2 and 1 of them, so the 3 edge outputs on each side receive 4, 5 and 6: 15 per side, 30 in all.
          That is 128 contributions. Every output has at least 4 writers, so every one of the 128 is an atomic.
          Output-centric, each output has one writer that adds its products in a register and stores once: 20 plain stores,
          no atomics. The explorer&rsquo;s convolution preset, 12 inputs and 5 wide, is the same count: 54.
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.2, pp. 533–534; our arithmetic (sim/decompose.ts)',
    vi: { source: "ch. 22 §22.2, pp. 533–534; phép tính của khóa học này (sim/decompose.ts)" },
  },
  'ch22/22.2/q-hist-inspect': {
    kind: 'numeric',
    id: 'ch22/22.2/q-hist-inspect',
    prompt: (
      <Translated
        vi={
          <>
            Một histogram gồm 1,000,000 đầu vào vào 256 bin được phân rã theo kiểu lấy đầu ra làm trung tâm: mỗi bin một luồng.
            Nó đọc đầu vào nhiều gấp bao nhiêu lần so với phiên bản lấy đầu vào làm trung tâm?
          </>
        }
      >
        <>
          A histogram of 1,000,000 inputs into 256 bins is decomposed output-centric: one thread per bin. How many times as
          many input reads does it make as the input-centric version?
        </>
      </Translated>
    ),
    answer: 256,
    unit: '×',
    explain: (
      <Translated
        vi={
          <p>
            Luồng của một bin không thể biết đầu vào nào rơi vào bin của nó nếu không đọc chúng, nên mỗi luồng trong 256 luồng
            đọc cả 1,000,000: 256 triệu lần đọc so với 1 triệu. Mức song song giảm từ một triệu luồng xuống 256. Và phản bác thứ
            ba của sách vẫn đứng vững ngay cả khi việc tra cứu là miễn phí: các bin nhận số lượng khác nhau, nên các luồng mất
            cân bằng tải.
          </p>
        }
      >
        <p>
          A bin's thread cannot know which inputs fall in its bin without reading them, so each of the 256 threads reads all
          1,000,000: 256 million reads against 1 million. Parallelism falls from a million threads to 256. And the book's
          third objection survives even a free lookup: the bins receive different counts, so the threads are out of balance.
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.2, p. 534',
  },
  'ch22/22.2/q-merge': {
    kind: 'choice',
    id: 'ch22/22.2/q-merge',
    prompt: (
      <Translated vi={<>Phép trộn (merge) của sách lấy đầu ra làm trung tâm, và lý do không phải là atomic. Vậy lý do là gì?</>}>
        <>The book's merge is output-centric, and atomics are not the reason. What is?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Cân bằng tải: cắt theo đầu ra cho mọi luồng cùng số phần tử; cắt theo một đầu vào thì không."}>Load balance: cutting the output gives every thread the same number of elements; cutting one input does not.</Translated>,
        correct: true,
      },
      { label: <Translated vi={"Lấy đầu ra làm trung tâm tránh được các atomic mà một phép trộn lấy đầu vào làm trung tâm sẽ cần ở bất cứ đâu hai đầu vào rơi vào cùng một đầu ra."}>Output-centric avoids the atomics an input-centric merge would need wherever two inputs land in the same output.</Translated>, why: <Translated vi={"Một phép trộn lấy đầu vào làm trung tâm cũng ghi mỗi đầu ra một lần: mỗi phần tử có đúng một đích."}>An input-centric merge writes each output once too: every element has exactly one destination.</Translated> },
      { label: <Translated vi={"Một luồng lấy đầu vào làm trung tâm không thể biết đầu vào của nó đi đâu nếu không tìm kiếm, trong khi luồng lấy đầu ra làm trung tâm biết đoạn của nó."}>An input-centric thread could not find where its inputs go without a search, while an output-centric one knows its range.</Translated>, why: <Translated vi={"Cả hai phiên bản đều tìm kiếm; ý của sách là rốt cuộc mỗi luồng nhận bao nhiêu việc."}>Both versions search; the book's point is how much work each thread ends up with.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Cân bằng tải khi gộp việc cho luồng (thread coarsening). Một luồng sở hữu một đoạn của đầu ra làm cùng lượng việc
            như mọi luồng khác. Một luồng sở hữu một đoạn của một đầu vào sẽ tìm thấy đoạn tương ứng trong đầu vào kia, có thể
            ngắn hoặc dài, nên các luồng trộn số phần tử rất khác nhau và phân kỳ.
          </p>
        }
      >
        <p>
          Load balance under thread coarsening. A thread that owns a chunk of the output does the same amount of work as
          every other. A thread that owns a chunk of one input finds a counterpart range in the other input that can be
          short or long, so threads merge very different numbers of elements and diverge.
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.2, p. 535; ch. 13', // i18n-ignore: citation
  },
  'ch22/22.2/q-bfs': {
    kind: 'choice',
    id: 'ch22/22.2/q-bfs',
    prompt: (
      <Translated
        vi={
          <>
            Trong BFS kiểu push lấy đỉnh làm trung tâm, nhiều luồng có thể ghi cấp của cùng một đỉnh kề cùng lúc, không dùng
            atomic. Vì sao điều đó được phép?
          </>
        }
      >
        <>
          In vertex-centric push BFS, several threads may write the same neighbour's level at the same time, with no
          atomic. Why is that allowed?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Mọi luồng ghi đều ghi cùng một giá trị, cấp kế tiếp, nên việc cập nhật là lũy đẳng.">Every writer stores the same value, the next level, so the update is idempotent.</Translated>, correct: true },
      { label: <Translated vi={"CUDA tuần tự hóa các lần ghi thường vào một địa chỉ, nên các luồng ghi đồng thời lần lượt ghi và không lần ghi nào bị mất."}>CUDA serialises plain stores to one address, so concurrent writers take turns and none of them is lost.</Translated>, why: <Translated vi={"Lần ghi thường của luồng nào còn lại là không xác định (ch09-facts A3). Ở đây điều đó không quan trọng chỉ vì chúng đều ghi cùng một giá trị."}>Which thread's plain store survives is undefined (ch09-facts A3). It does not matter here only because they all store the same value.</Translated> },
      { label: <Translated vi={"Mỗi cấp chỉ được ghi một lần trong mỗi lần khởi chạy kernel."}>A level is written only once per kernel launch.</Translated>, why: <Translated vi={"Trong một lần khởi chạy, nhiều luồng có thể cùng tới một đỉnh kề chưa thăm và ghi nó."}>Several threads in one launch can reach the same unvisited neighbour and write it.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Khi atomic không còn là vấn đề, sách nói điều quyết định giữa push và pull là mức song song mỗi cách khai thác được
            và cân bằng tải mà nó mang lại, và cả hai đều phụ thuộc vào đồ thị.
          </p>
        }
      >
        <p>
          With atomics out of the question, the book says what decides push against pull is the parallelism each extracts
          and the load balance it gives, and both depend on the graph.
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.2, p. 536; ch. 18', // i18n-ignore: citation
  },
  'ch22/22.2/q-coo': {
    kind: 'choice',
    id: 'ch22/22.2/q-coo',
    prompt: (
      <Translated
        vi={
          <>
            Một ma trận thưa có một hàng chứa một nửa số phần tử khác không của nó. Vì sao kernel lấy đầu vào làm trung tâm của
            COO, với một atomic cho mỗi phần tử khác không, có thể thắng cách mỗi hàng một luồng của CSR?
          </>
        }
      >
        <>
          A sparse matrix has one row that holds half its non-zeros. Why might COO's input-centric kernel, with an atomic per
          non-zero, beat CSR's thread per row?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="CSR giao cho một luồng nửa khối lượng việc; COO rải đều các phần tử khác không cho các luồng và trả giá bằng atomic để có sự cân bằng đó.">CSR gives one thread half the work; COO spreads the non-zeros evenly over threads and pays atomics for that balance.</Translated>, correct: true },
      { label: <Translated vi={"Với ma trận lệch như thế này, COO dùng ít bộ nhớ hơn CSR, nên các luồng của nó di chuyển ít byte hơn cho mỗi phần tử khác không."}>COO uses less memory than CSR for a matrix this skewed, so its threads move fewer bytes per non-zero.</Translated>, why: <Translated vi={"COO lưu một chỉ số hàng cho mỗi phần tử khác không, nên nó không phải định dạng nhỏ hơn. Và bộ nhớ không phải sự đánh đổi sách mô tả; đó là cân bằng tải và mức song song đổi lấy atomic."}>COO stores a row index for every non-zero, so it is not the smaller format. And memory is not the trade-off the book describes; it is balance and parallelism against atomics.</Translated> },
      { label: <Translated vi={"Mỗi phép cộng atomic rẻ hơn phép cộng trong thanh ghi mà nó thay thế, nên COO làm ít phép tính hơn CSR cho mỗi phần tử khác không."}>Each atomic addition costs less than the register addition it replaces, so COO does less arithmetic work per non-zero than CSR.</Translated>, why: <Translated vi={"Không tài liệu nào ghi nhận điều đó, và sách nói điều ngược lại (p. 533)."}>Nothing documents that, and the book says the opposite (p. 533).</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong thiết lập sẵn ma trận lệch của trình khám phá, luồng bận nhất của CSR làm gấp 3 lần mức việc trung bình; các
            luồng của COO mỗi luồng làm một phần tử khác không và trả 20 lần cập nhật atomic cho điều đó.
          </p>
        }
      >
        <p>
          In the explorer's skewed preset, CSR's busiest thread does 3 times the mean work; COO's threads do one non-zero
          each and pay 20 atomic updates for it.
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.2, pp. 535–536; ch. 17', // i18n-ignore: citation
  },
  'ch22/22.2/q-plain-scatter': {
    kind: 'choice',
    id: 'ch22/22.2/q-plain-scatter',
    prompt: (
      <Translated
        vi={
          <>
            Kernel lấy đầu vào làm trung tâm ở trên vượt qua mọi bài kiểm tra trên backend <code>emu</code>, trong cả ba thứ tự
            luồng. Nó có đúng không?
          </>
        }
      >
        <>
          The input-centric kernel above passes every test on the <code>emu</code> backend, in all three thread orders. Is it
          correct?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Không. Hai luồng có thể cùng đọc out[o], cộng và ghi lại, làm mất một lần cập nhật; emu không bao giờ ngắt một luồng ở đó, nhưng race báo dòng +=."}>No. Two threads can each read out[o], add and write back, losing an update; emu never interrupts a thread there, but race reports the += line.</Translated>,
        correct: true,
      },
      { label: <Translated vi={"Có: emu chạy các luồng theo ba thứ tự khác nhau và so mọi đầu ra với một kết quả tham chiếu trên CPU, nên bất kỳ lần cập nhật bị mất nào cũng sẽ lộ ra."}>Yes: emu runs the threads in three different orders and compares every output with a CPU reference, so any lost update would have shown up.</Translated>, why: <Translated vi={"Mỗi luồng giả lập chạy đến điểm đồng bộ hóa kế tiếp của nó trước khi luồng tiếp theo bắt đầu, nên một phép += không bao giờ bị ngắt (labs/README.md)."}>Each emulated thread runs to its next synchronisation point before the next one starts, so a += is never interrupted (labs/README.md).</Translated> },
      { label: <Translated vi={"Có: các đầu ra là float, và một phép += trên float được biên dịch thành một lần ghi duy nhất, nên phép cộng của hai luồng không thể xung đột."}>Yes: the outputs are floats, and a float += compiles to a single store, so two threads’ additions cannot conflict.</Translated>, why: <Translated vi={"Một phép += là một lần nạp, một phép cộng và một lần ghi với mọi kiểu, và hai luồng làm vậy trên một địa chỉ có thể làm mất một lần cập nhật bất kể giá trị lưu ở đó là gì."}>A += is a load, an add and a store for any type, and two threads doing that to one address can lose an update whatever is stored there.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Unit test của trang này build chính file này trên cả hai trình giả lập (emulator). Trên <code>emu</code> nó cho đáp án
            đúng ở mọi thứ tự. Trên <code>race</code>, ThreadSanitizer báo dòng <code>+=</code> tranh chấp với chính nó, và các đầu
            ra có thể sai. Phiên bản <code>atomicAdd</code> và phiên bản gather đúng trên cả hai, không có báo cáo nào.
          </p>
        }
      >
        <p>
          This page's unit test builds this exact file on both emulators. On <code>emu</code> it gives the right answer in
          every order. On <code>race</code>, ThreadSanitizer reports the <code>+=</code> line racing with itself, and outputs
          can come out wrong. The <code>atomicAdd</code> version and the gather version are right on both, with no report.
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.2, p. 533; ch. 9 §9.2; labs/README.md', // i18n-ignore: citation
  },
  'ch22/22.2/q-hip-atomics': {
    kind: 'choice',
    id: 'ch22/22.2/q-hip-atomics',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel nhân ma trận với vector kiểu split-K cộng các tổng riêng phần bằng <code>atomicAdd</code> trên{' '}
            <code>float</code>. Nó chạy tốt trên GPU NVIDIA và chậm trên GPU AMD. Khác biệt nào đã được tài liệu hóa là một ứng
            viên?
          </>
        }
      >
        <>
          A split-K matrix-vector kernel adds its partial sums with <code>atomicAdd</code> on <code>float</code>. It runs well
          on an NVIDIA GPU and slowly on an AMD one. Which documented difference is a candidate?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Theo mặc định HIP biên dịch atomicAdd trên float theo cách an toàn, có thể thành một vòng lặp CAS, trừ khi build với -munsafe-fp-atomics.">HIP compiles float atomicAdd safely by default, possibly as a CAS loop, unless built with -munsafe-fp-atomics.</Translated>, correct: true },
      { label: <Translated vi={"HIP không có atomicAdd trên float, nên lời gọi lùi về một khóa bao quanh một lần đọc, cộng và ghi thường."}>HIP has no float atomicAdd, so the call falls back to a lock around a plain read, add and write.</Translated>, why: <Translated vi={"HIP có nó, với cùng cách viết (infra-facts B1)."}>HIP has it, with the same spelling (infra-facts B1).</Translated> },
      { label: <Translated vi={"GPU AMD tuần tự hóa mọi atomic trên toàn thiết bị, bất kể địa chỉ, nên tranh chấp tốn hơn trên NVIDIA."}>AMD GPUs serialise every atomic device-wide, whatever its address, so contention costs more than on NVIDIA.</Translated>, why: <Translated vi={"Không tài liệu nào ghi nhận điều đó. Atomic trên một địa chỉ bị tuần tự hóa trên mọi backend; AMD ghi rằng các atomic toàn cục của nó nằm ở L2 (ch09-facts B2)."}>Nothing documents that. Atomics on one address serialise on every backend; AMD documents its global atomics in the L2 (ch09-facts B2).</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Tài liệu của AMD: một <code>atomicAdd</code> trên float "may generate a hardware instruction that is faster than
            emulating the atomic operation using an atomic compare-and-swap (CAS) loop", và "by default the compiler does not
            generate unsafe instructions". Đó là một ứng viên, không phải một chẩn đoán: hãy đo khi có và không có cờ này. Bản
            build HIP của khóa học không truyền cờ đó.
          </p>
        }
      >
        <p>
          AMD's documentation: a float <code>atomicAdd</code> "may generate a hardware instruction that is faster than
          emulating the atomic operation using an atomic compare-and-swap (CAS) loop", and "by default the compiler does not
          generate unsafe instructions". It is a candidate, not a diagnosis: measure with and without the flag. The course's
          HIP build does not pass it.
        </p>
      </Translated>
    ),
    source: 'notes/today/infra-facts.md B1, checked 2026-11-20',
    vi: { source: "notes/today/infra-facts.md B1, kiểm tra ngày 2026-11-20" },
  },
  'ch22/22.2/q-metal-max': {
    kind: 'choice',
    id: 'ch22/22.2/q-metal-max',
    prompt: (
      <Translated
        vi={
          <>
            Bạn đang port sang Metal một scatter giữ giá trị lớn nhất tích lũy bằng <code>atomicMax</code> trên float. Nó biên
            dịch được cho HIP. Bạn làm gì?
          </>
        }
      >
        <>
          You are porting to Metal a scatter that keeps a running maximum with <code>atomicMax</code> on floats. It compiles
          for HIP. What do you do?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Cấu trúc lại: một gather với mỗi đầu ra một chủ sở hữu, hoặc khóa không dấu giữ nguyên thứ tự với atomic max trên atomic_uint. Trong các phép fetch-and-modify, MSL chỉ ghi nhận add và sub cho atomic_float."}>Restructure it: a gather with one owner per output, or order-preserving unsigned keys with atomic max on atomic_uint. Of the fetch-and-modify operations, MSL documents only add and sub for atomic_float.</Translated>,
        correct: true,
      },
      { label: <Translated vi={"Gọi atomic_fetch_max_explicit trên một device atomic_float*, bản tương ứng của atomicMax trên float của HIP trong MSL."}>Call atomic_fetch_max_explicit on a device atomic_float*, MSL’s counterpart of HIP’s float atomicMax.</Translated>, why: <Translated vi={"MSL liệt kê max và min cho atomic_int và atomic_uint; với atomic_float nó chỉ ghi nhận add và sub (infra-facts C1)."}>MSL lists max and min for atomic_int and atomic_uint; for atomic_float it documents add and sub only (infra-facts C1).</Translated> },
      { label: <Translated vi={"Metal hoàn toàn không có atomic, nên không thể port một scatter; hãy giữ kernel trên CUDA và HIP."}>Metal has no atomics at all, so a scatter cannot be ported; keep the kernel on CUDA and HIP.</Translated>, why: <Translated vi={"Metal có atomic_int, atomic_uint và atomic_float, và fetch-and-add dùng được trên cả ba (infra-facts C1)."}>Metal has atomic_int, atomic_uint and atomic_float, and fetch-and-add works on all three (infra-facts C1).</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            HIP chấp nhận <code>atomicMax(float*, float)</code>; <code>atomicMax</code> kiểu cũ của CUDA chỉ nhận kiểu số nguyên
            và công thức mà hướng dẫn của nó đưa ra cho float là một vòng lặp compare-and-swap (infra-facts A1, B1). Khóa giữ
            nguyên thứ tự là phép biến đổi mà mục 14.4 dùng để sắp xếp logits; áp dụng nó cho atomic max là của khóa học này.
            Một hướng thứ ba, cũng của khóa học này, là một vòng lặp compare-exchange trên một device <code>atomic_float</code>,
            điều mà MSL có ghi nhận (infra-facts C1).
          </p>
        }
      >
        <p>
          HIP accepts <code>atomicMax(float*, float)</code>; CUDA's legacy <code>atomicMax</code> takes integer types only
          and its guide's recipe for a float is a compare-and-swap loop (infra-facts A1, B1). The order-preserving key is the
          transform section 14.4 uses to sort logits; applying it to an atomic max is ours. A third route, also ours, is a
          compare-exchange loop on a device <code>atomic_float</code>, which MSL documents (infra-facts C1).
        </p>
      </Translated>
    ),
    source: 'notes/today/infra-facts.md A1, B1, C1 and its backend table, checked 2026-11-20',
    vi: { source: "notes/today/infra-facts.md A1, B1, C1 và bảng backend của nó, kiểm tra ngày 2026-11-20" },
  },
  'ch22/22.2/q-determinism': {
    kind: 'choice',
    id: 'ch22/22.2/q-determinism',
    prompt: (
      <Translated
        vi={
          <>
            Logits từ kernel split-K của bạn khác nhau ở các bit cuối giữa các lần chạy, và thỉnh thoảng greedy decoding chọn một
            token khác. Điều này thuộc tiêu chí nào trong các tiêu chí của sách để chọn cách phân rã (p. 533)?
          </>
        }
      >
        <>
          Your split-K kernel's logits differ in the last bits from run to run, and now and then greedy decoding picks a
          different token. Which of the book's criteria for choosing a decomposition (p. 533) does this fall under?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Các tiêu chí khác tùy ứng dụng.">Others that depend on the application.</Translated>, correct: true },
      { label: <Translated vi={"Mức song song mà cách phân rã lộ ra."}>The amount of parallelism the decomposition exposes.</Translated>, why: <Translated vi={"S thay đổi mức song song, không thay đổi các bit đầu ra."}>S changes the parallelism, not which bits come out.</Translated> },
      { label: <Translated vi={"Một luồng xác định các đầu vào cấp cho đầu ra của nó dễ đến đâu."}>How easily a thread identifies the inputs that feed its outputs.</Translated>, why: <Translated vi={"Mỗi warp tìm thấy đoạn của nó trực tiếp."}>Every warp finds its chunk directly.</Translated> },
      { label: <Translated vi={"Cân bằng tải mà cách phân rã tạo ra."}>The load balance the decomposition induces.</Translated>, why: <Translated vi={"Các đoạn bằng nhau thì cân bằng. Sự khác biệt đến từ thứ tự các phép cộng atomic được thực hiện."}>Equal chunks are balanced. The variation comes from the order in which the atomic adds land.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Atomic cộng theo bất kỳ thứ tự nào chúng đến, và phép cộng dấu phẩy động không có tính kết hợp (phụ lục A, §A.6).
            Cách mỗi đầu ra một warp cộng theo cùng thứ tự mỗi lần chạy và cho cùng các bit. Điều đó có quan trọng hay không phụ
            thuộc vào ứng dụng; với decode tái lập được thì có.
          </p>
        }
      >
        <p>
          Atomics add in whatever order they arrive, and floating-point addition is not associative (appendix A, §A.6). A
          warp per output adds in the same order every run and gives the same bits. Whether that matters depends on the
          application; for reproducible decoding it does.
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.2, p. 533; appendix A §A.6, pp. 593–594; the scenario is ours',
    vi: { source: "ch. 22 §22.2, p. 533; phụ lục A §A.6, pp. 593–594; tình huống này là của khóa học này" },
  },
};

// ---------------------------------------------------------------- the section
export function S222() {
  return (
    <>
      <Translated
        vi={
          <p>
            Bạn đã chọn được thuật toán. Còn một quyết định trước khi viết kernel, và nó định hình mọi thứ sau đó: một luồng
            (thread) sở hữu cái gì? Sách diễn đạt điều này là cắt bài toán thành các bài toán con có thể chạy an toàn cùng lúc.
            Tìm ra tính song song trong một bài toán lớn thường đơn giản về nguyên tắc nhưng khó trong thực tế; phần khó là chọn
            đơn vị công việc mà mỗi luồng, hoặc mỗi nhóm luồng, đảm nhận.
          </p>
        }
      >
        <p>
          You have chosen an algorithm. One decision remains before you write the kernel, and it shapes everything after it:
          what does one thread own? The book frames this as cutting the problem into sub-problems that can safely run at the
          same time. Finding the parallelism in a large problem is usually simple in principle and hard in practice; the
          hard part is choosing the unit of work each thread, or group of threads, takes on.
        </p>
      </Translated>
      <Source section="22.2" pages="532" />

      <h2><Translated vi={"Hai cách cắt"}>Two ways to cut</Translated></h2>
      <Translated
        vi={
          <p>
            Có hai câu trả lời tự nhiên. Trong cách phân rã <b>lấy đầu ra làm trung tâm</b> (output-centric), mỗi luồng sở hữu
            một hoặc nhiều đầu ra và tính chúng từ các đầu vào cần thiết: toàn bộ đầu vào với phép cộng Coulomb trực tiếp ở
            chương 21, một tập con với phép nhân ma trận hoặc phương pháp ngưỡng cắt có xếp bin. Trong cách phân rã <b>lấy đầu
            vào làm trung tâm</b> (input-centric), mỗi luồng sở hữu một hoặc nhiều đầu vào và áp phần đóng góp của chúng lên các
            đầu ra: lên một đầu ra với histogram, lên một tập con với một phương pháp ngưỡng cắt giả định lấy nguyên tử làm trung
            tâm, lên mọi đầu ra với phép cộng trực tiếp lấy nguyên tử làm trung tâm.
          </p>
        }
      >
        <p>
          There are two natural answers. In an <b>output-centric</b> decomposition each thread owns one or more outputs and
          computes them from the inputs they need: all of the inputs for direct Coulomb summation in chapter 21, a subset for
          matrix multiplication or the binned cutoff method. In an <b>input-centric</b> decomposition each thread owns one or
          more inputs and applies their contributions to the outputs: to one output for a histogram, to a subset for a
          hypothetical atom-centric cutoff method, to every output for atom-centric direct summation.
        </p>
      </Translated>
      <Source section="22.2" pages="532" figs={tr('Fig. 22.1', 'Hình 22.1')} />
      <Translated
        vi={
          <p>
            Đôi khi sự phân biệt biến mất. Trong odd-even sort, mỗi luồng đọc một cặp và ghi chính cặp đó; đầu vào và đầu ra
            tương ứng trực tiếp đến mức cái nhãn không còn quan trọng. Trong mọi trường hợp, hai cách cắt tính ra cùng một kết
            quả, và chúng có thể chạy rất khác nhau trên cùng phần cứng. Trình khám phá ở dưới vẽ cả hai cho bốn phép tính.
          </p>
        }
      >
        <p>
          Sometimes the distinction disappears. In odd-even sort each thread reads a pair and writes the same pair; inputs and
          outputs correspond so directly that the label does not matter. In every case the two cuts compute the same result,
          and they can run very differently on the same hardware. The explorer further down draws both for four computations.
        </p>
      </Translated>
      <Source pages="533" />

      <h2><Translated vi={"Gather và scatter"}>Gather and scatter</Translated></h2>
      <Translated
        vi={
          <p>
            Một luồng lấy đầu ra làm trung tâm thường <b>gather</b>: nó kéo tác động của nhiều đầu vào về một đầu ra. Sách đưa
            ra hai lý do điều này hợp với GPU. Tổng tích lũy nằm trong một <span className="t-reg">thanh ghi (register)</span>{' '}
            riêng cho đến cuối. Và các luồng lân cận cần những đầu vào chồng lấn nhau, mà chúng có thể chia sẻ qua bộ nhớ đệm
            (cache) hoặc{' '}
            <span className="t-shared">bộ nhớ chia sẻ (shared memory)</span> thay vì mỗi luồng tự nạp từ bộ nhớ toàn cục.
          </p>
        }
      >
        <p>
          An output-centric thread usually <b>gathers</b>: it pulls the effect of many inputs into one output. The book
          gives two reasons that suits a GPU. The running total lives in a private <span className="t-reg">register</span>{' '}
          until the end. And neighbouring threads need overlapping inputs, which they can share through the caches or{' '}
          <span className="t-shared">shared memory</span> instead of each fetching them from global memory.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Một luồng lấy đầu vào làm trung tâm thường <b>scatter</b>: nó đẩy tác động của một đầu vào ra nhiều đầu ra. Giờ thì
            nhiều luồng có thể cập nhật một đầu ra cùng lúc. Các đầu ra phải nằm trong bộ nhớ mà tất cả chúng đều ghi được, và
            mọi lần cập nhật một đầu ra dùng chung phải là một phép đọc-sửa-ghi nguyên tử (atomic), nếu không các lần cập nhật sẽ
            bị mất. Kết luận của sách là các atomic đó chậm hơn đáng kể so với các phép cộng trong thanh ghi mà chúng thay thế.
          </p>
        }
      >
        <p>
          An input-centric thread usually <b>scatters</b>: it pushes one input's effect out to several outputs. Now several
          threads may update one output at the same time. The outputs must live in memory all of them can write, and every
          update to a shared output must be an atomic read-modify-write, or updates are lost. The book's verdict is that
          those atomics are significantly slower than the register additions they replace.
        </p>
      </Translated>
      <Source pages="533" />
      <Today checked="2026-11-20 (notes/today/ch22-facts.md A1)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              Cơ chế này có tài liệu: NVIDIA nói một lần truy cập thanh ghi "consumes zero extra clock cycles per instruction"
              (Best Practices Guide 10.2.7), các atomic trên một địa chỉ bị tuần tự hóa, và atomic phạm vi device chạy ở L2
              (ch09-facts A4). Không backend nào ghi lại atomic chậm hơn phép cộng trong thanh ghi bao nhiêu, nên khóa học này
              không đưa ra tỉ lệ nào. Các lab của chương 9 đo atomic khi có tranh chấp; lab 22.1 đo toàn bộ sự đánh đổi (thêm warp
              đổi lấy S atomic cho mỗi đầu ra) trên một GEMV lúc decode, không riêng chi phí của một atomic.
            </p>
          }
        >
          <p>
            The mechanism is documented: NVIDIA says a register access "consumes zero extra clock cycles per instruction"
            (Best Practices Guide 10.2.7), atomics on one address are serialised, and device-scope atomics run in the L2
            (ch09-facts A4). No backend documents how much slower an atomic is than a register add, so this course gives no
            ratio. Chapter 9's labs measure contended atomics; lab 22.1 measures the whole trade (more warps against S
            atomics per output) on a decode GEMV, not an atomic's cost alone.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Một bức hình tĩnh về scatter cho thấy các mũi tên hội tụ. Điều nó không thể cho thấy là thời gian: hàng đợi hình
            thành khi nhiều lần cập nhật cùng đến một địa chỉ. Hoạt ảnh vẽ ra hàng đợi đó.
          </p>
        }
      >
        <p>
          A still picture of a scatter shows the arrows converging. What it cannot show is time: the queue that forms when
          several updates arrive at one address. The animation draws that queue.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một nhịp là một lần cập nhật vào một địa chỉ. Mọi lần cập nhật được phát ra cùng lúc, và các địa chỉ khác nhau được
              phục vụ song song. Phép cộng trong thanh ghi, bộ nhớ đệm và độ dài thực của một nhịp không được mô hình hóa; không
              backend nào ghi lại độ dài đó.
            </p>
          }
        >
          <p>
            One tick is one update to one address. Every update is issued at once, and different addresses are served in
            parallel. Register additions, caches and the real length of a tick are not modelled; no backend documents that
            length.
          </p>
        </Translated>
      </Simplification>
      <GatherScatterScene />

      <h2><Translated vi={"Ai ghi, ai xung đột"}>Who writes, who conflicts</Translated></h2>
      <Translated
        vi={
          <p>
            Gather hay scatter chưa phải toàn bộ quyết định. Sách liệt kê thêm các tiêu chí: cách phân rã lộ ra bao nhiêu song
            song, một luồng xác định các đầu vào đóng góp vào đầu ra của nó dễ đến đâu, cân bằng tải mà nó tạo ra, và các tiêu
            chí khác tùy ứng dụng. Chương 21 là nơi duy nhất sách cài đặt cả hai cách cắt và so sánh chúng; ở mọi nơi khác lựa
            chọn được đưa ra mà không nói.
          </p>
        }
      >
        <p>
          Gather against scatter is not the whole decision. The book lists more criteria: how much parallelism the
          decomposition exposes, how easily a thread can identify the inputs that contribute to its outputs, the load balance
          it induces, and others that depend on the application. Chapter 21 is the one place the book implements both cuts
          and compares them; everywhere else the choice was made silently.
        </p>
      </Translated>
      <Source pages="533" />
      <Translated
        vi={
          <p>
            Trình khám phá làm lựa chọn đó trở nên tường minh. Chọn một phép tính và một cách cắt, rồi bấm vào một luồng để xem
            cách ánh xạ chỉ số của nó. Thử từng thiết lập sẵn, rồi mỗi lần thay đổi một thứ:
          </p>
        }
      >
        <p>
          The explorer makes the choice explicit. Pick a computation and a cut, then click a thread to see its index mapping.
          Try each preset, then change one thing at a time:
        </p>
      </Translated>
      <ul>
        <Translated
          vi={
            <li>Tích chập: chuyển qua lại giữa hai cách cắt. Có tiêu chí nào mà cách lấy đầu vào làm trung tâm thắng không?</li>
          }
        >
          <li>Convolution: switch between the two cuts. Is there any criterion on which input-centric wins?</li>
        </Translated>
        <Translated
          vi={
            <li>Histogram, mỗi bin một luồng: mỗi luồng xem xét bao nhiêu đầu vào? Rồi tăng độ lệch lên.</li>
          }
        >
          <li>Histogram, a thread per bin: how many inputs does each thread inspect? Then raise the skew.</li>
        </Translated>
        <Translated
          vi={
            <li>Ma trận thưa lệch: so sánh độ cân bằng của CSR với của COO, và COO trả giá gì cho sự cân bằng đó.</li>
          }
        >
          <li>Skewed sparse matrix: compare CSR's balance with COO's, and what COO pays for its balance.</li>
        </Translated>
        <Translated
          vi={
            <li>
              GEMV lúc decode: với 2 đầu ra, cách cắt nào lộ ra nhiều hơn vài warp, và với giá nào? Rồi đặt 32 đầu ra và so sánh
              số sector mà warp 0 chạm tới khi mỗi đầu ra một luồng và khi mỗi đầu ra một warp.
            </li>
          }
        >
          <li>
            Decode GEMV: with 2 outputs, which cut exposes more than a couple of warps, and at what price? Then set 32
            outputs and compare the sectors warp 0 touches with a thread per output and with a warp per output.
          </li>
        </Translated>
      </ul>
      <Simplification>
        <Translated
          vi={
            <p>
              Một luồng (hoặc warp) cộng mọi đóng góp của nó cho một đầu ra trong một thanh ghi và ghi một lần. Không có gì được
              đo thời gian: số atomic cho biết bao nhiêu lần cập nhật bị tuần tự hóa, không phải chúng tốn bao lâu. Không có bộ nhớ
              đệm, nên một đầu vào được hai luồng đọc thì tính hai lần. Số sector là của lần nạp W đầu tiên của warp 0, một lệnh,
              theo đơn vị 32 byte của NVIDIA (ch06-facts A2); dòng cache của AMD là 64 hoặc 128 byte và Apple không ghi lại con số
              nào.
            </p>
          }
        >
          <p>
            A thread (or warp) adds all its contributions to one output in a register and writes once. Nothing is timed: the
            atomic counts say how many updates serialise, not how long they take. No caches, so an input read by two threads
            counts twice. The sector count is warp 0's first load of W, one instruction, in NVIDIA's 32-byte unit
            (ch06-facts A2); AMD's lines are 64 or 128 bytes and Apple documents none.
          </p>
        </Translated>
      </Simplification>
      <Explorer />
      <Ask q={questionsS222['ch22/22.2/q-conv-writers']} label={tr('Work it out', 'Hãy tính ra')} />
      <Ask q={questionsS222['ch22/22.2/q-hist-inspect']} label={tr('Work it out', 'Hãy tính ra')} />

      <h2><Translated vi={"Chuyến tham quan của sách"}>The book's tour</Translated></h2>
      <Translated
        vi={
          <p>
            Sau đó sách đi lại qua mọi mẫu hình nó đã dạy và gọi tên cách phân rã mà mỗi mẫu hình đã dùng. Phần lớn lấy đầu ra
            làm trung tâm vì lý do hiển nhiên. Những thẻ đáng học nhất là những thẻ mà atomic không quyết định gì cả. Với mỗi
            thẻ, chọn cách phân rã rồi chọn điều đã quyết định nó.
          </p>
        }
      >
        <p>
          The book then walks back through every pattern it has taught and names the decomposition each one used. Most are
          output-centric for the obvious reason. The instructive cards are the ones where atomics decide nothing at all. For
          each card, choose the decomposition and then what decided it.
        </p>
      </Translated>
      <Source pages="533–536" />
      <SortTheTour />
      <Translated
        vi={
          <p>
            Hai điều cần giữ lại từ chuyến tham quan. Các phép tính thực tế trộn lẫn chiến lược: phép lọc ổn định lấy đầu vào
            làm trung tâm, nhưng scan bên trong nó lại lấy đầu ra làm trung tâm, và định dạng lai ELL-COO kết hợp một phần lấy
            đầu ra làm trung tâm (ELL) với một phần lấy đầu vào làm trung tâm (COO). Và với dữ liệu thưa, sách kết luận rằng cách
            phân rã tốt nhất phụ thuộc vào chính tập dữ liệu đầu vào.
          </p>
        }
      >
        <p>
          Two things to keep from the tour. Real computations mix strategies: the stable filter is input-centric, yet the
          scan inside it is output-centric, and the hybrid ELL-COO format combines an output-centric part (ELL) with an
          input-centric one (COO). And for sparse data the book concludes that the best decomposition depends on the input
          dataset itself.
        </p>
      </Translated>
      <Source pages="534–536" />
      <Aside>
        <Translated
          vi={
            <p>
              Hai chỗ nhầm trong bản in của chuyến tham quan. Trang 534 dẫn "scan (Chapter 12)"; scan là chương 11, như chính
              chương này nói ở p. 530. Trang 536 nói bản đồ tĩnh điện ưu tiên lấy đầu ra làm trung tâm "because of the benefit over
              the gather pattern of the scatter pattern"; phần còn lại của đoạn cho thấy ý là lợi thế của gather so với scatter.
            </p>
          }
        >
          <p>
            Two slips in the printed tour. Page 534 cites "scan (Chapter 12)"; scan is chapter 11, as the same chapter says
            on p. 530. Page 536 says the electrostatic map favours output-centric "because of the benefit over the gather
            pattern of the scatter pattern"; the rest of the paragraph shows it means the benefit of gather over scatter.
          </p>
        </Translated>
      </Aside>
      <Ask q={questionsS222['ch22/22.2/q-merge']} />
      <Ask q={questionsS222['ch22/22.2/q-bfs']} />
      <Ask q={questionsS222['ch22/22.2/q-coo']} />

      <h2><Translated vi={"Split-K: atomic xuất hiện cùng warp thứ hai"}>Split-K: the atomic arrives with the second warp</Translated></h2>
      <Translated
        vi={
          <p>
            Giờ đến một cách phân rã bạn sẽ tự viết. Trong lúc decode, mỗi token mới nhân vector activation của nó với mọi ma
            trận trọng số: y = W x, một phép nhân ma trận với vector (mục 22.5 giải thích vì sao decode có dạng này). Cách cắt
            lấy đầu ra làm trung tâm giao mỗi đầu ra một warp. Các lane của nó cùng bước dọc theo hàng j của W, nên mỗi bước đọc
            một đoạn liền, và lane 0 ghi tổng của warp. Không có atomic. Nhưng một ma trận có ít đầu ra thì cho ít warp.
          </p>
        }
      >
        <p>
          Now a decomposition you will write. During decoding, each new token multiplies its activation vector by every
          weight matrix: y = W x, a matrix-vector product (section 22.5 says why decoding has this shape). The output-centric
          cut gives a warp to each output. Its lanes stride along row j of W together, so each step reads one contiguous run,
          and lane 0 stores the warp's total. No atomics. But a matrix with few outputs gives few warps.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Split-K cắt cả chiều đầu vào. S warp dùng chung một đầu ra, mỗi warp nhận một đoạn của hàng, nhờ vậy số warp nhân
            lên S lần. Nó cũng mang scatter quay lại: giờ có S warp cùng cộng vào mỗi phần tử của Y.
          </p>
        }
      >
        <p>
          Split-K cuts the input dimension as well. S warps share an output, each taking one chunk of the row, which
          multiplies the warps by S. It also brings the scatter back: S warps now add into each element of Y.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              GEMV lúc decode và split-K là của khóa học này; chương 22 không in mã nào. Cách ánh xạ là của{' '}
              <code>gemvSplitKKernel</code> trong lab 22.1: warp g sở hữu đầu ra g ÷ splits và đoạn g mod splits.
            </p>
          }
        >
          <p>
            The decode GEMV and split-K are ours; chapter 22 prints no code. The mapping is lab 22.1's{' '}
            <code>gemvSplitKKernel</code>: warp g owns output g ÷ splits and chunk g mod splits.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Trình chạy từng bước chạy một trường hợp rất nhỏ với một warp đồ chơi gồm 4 lane để bạn thấy được từng lane: một
            batch 2 hàng, 16 đầu vào, 3 đầu ra. Chạy qua từng bước với S = 1, rồi 2, rồi 4, và xem chỗ ghi kết quả thay đổi ở
            đâu. Sau đó chuyển cách ghi kết quả sang một phép <code>+=</code> thường và chạy mọi warp cùng lúc.
          </p>
        }
      >
        <p>
          The stepper runs a tiny case with a toy warp of 4 lanes so you can see every one: a batch of 2 rows, 16 inputs, 3
          outputs. Step through it with S = 1, then 2, then 4, and watch where the commit changes. Then switch the commit to
          a plain <code>+=</code> and run all warps together.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Các lane của một warp tiến cùng nhau, mỗi bước một phép toán, và các warp xen kẽ theo thứ tự bạn chọn. Không đo thời
              gian, không có bộ nhớ đệm. Một warp thật rộng hơn: hãy đọc <code>warpSize</code>, hoặc độ rộng thực thi của pipeline
              trên Metal. Hai phép <code>+=</code> thường tranh chấp nhau thường làm mất một lần cập nhật như minh họa, nhưng về mặt
              hình thức đó là một race condition (tranh chấp dữ liệu) với hành vi không xác định (ch09-facts A3).
            </p>
          }
        >
          <p>
            The lanes of a warp advance together, one operation per step, and warps interleave in the order you choose. No
            timing, no caches. A real warp is wider: read <code>warpSize</code>, or the pipeline's execution width on Metal.
            Two racing plain <code>+=</code> typically lose an update as shown, but formally it is a data race with undefined
            behaviour (ch09-facts A3).
          </p>
        </Translated>
      </Simplification>
      <SplitKStepper />
      <Translated
        vi={
          <p>
            Dù S là bao nhiêu, các warp cộng lại đọc mỗi trọng số một lần và mỗi phần tử của x một lần cho mỗi đầu ra; số byte
            dưới hình vẽ không đổi. Thứ S thay đổi là bao nhiêu luồng chia nhau công việc và các đầu ra nhận bao nhiêu lần cập
            nhật atomic, đúng là sự đánh đổi mà lab 22.1 đo trên thiết bị của bạn. Xem{' '}
            <a href={href('ch22', 'labs')}>trang lab</a>.
          </p>
        }
      >
        <p>
          Whatever S is, the warps between them read every weight once and every element of x once per output; the byte
          counts under the drawing do not move. What S changes is how many threads share the work and how many atomic updates
          the outputs take, which is exactly the trade lab 22.1 measures on your device. See the{' '}
          <a href={href('ch22', 'labs')}>labs page</a>.
        </p>
      </Translated>

      <h2><Translated vi={"Scatter này qua được trên trình giả lập"}>This scatter passes on the emulator</Translated></h2>
      <Translated
        vi={
          <p>
            Đây là phép tích chập theo cả hai cách, dưới dạng kernel. Đọc kỹ scatter, rồi trả lời câu hỏi bên dưới nó.
          </p>
        }
      >
        <p>
          Here is the convolution both ways, as kernels. Read the scatter carefully, then answer the question below it.
        </p>
      </Translated>
      <div className="wide" style={{ display: 'grid', gap: '0 1rem', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 30rem), 1fr))' }}>
        <Code tabs={[{ label: tr('scatter: a thread per input', 'scatter: mỗi đầu vào một luồng'), code: SCATTER, tiers: CONV_TIERS, status: tr('emulator-checked and front-end-checked; wrong on purpose', 'emulator-checked và front-end-checked; cố ý sai') }]} />
        <Code tabs={[{ label: tr('gather: a thread per output', 'gather: mỗi đầu ra một luồng'), code: GATHER, tiers: CONV_TIERS, status: tr('emulator-checked and front-end-checked; unverified on GPUs', 'emulator-checked và front-end-checked; unverified trên GPU') }]} />
      </div>
      <Ask q={questionsS222['ch22/22.2/q-plain-scatter']} />
      <Translated
        vi={
          <p>
            Lab 22.1 giấu cùng lỗi đó trong kernel split-K của nó. Hãy chạy nó trên backend <code>race</code> trước khi đi tìm.
          </p>
        }
      >
        <p>
          Lab 22.1 hides the same mistake in its split-K kernel. Run it on the <code>race</code> backend before you go looking
          for it.
        </p>
      </Translated>

      <h2><Translated vi={"Theo cách nói của PyTorch"}>In PyTorch terms</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn đã dùng cả hai hướng rồi. <code>torch.gather</code> đánh chỉ số theo đầu ra:{' '}
            <code>out[i] = src[index[i]]</code>, nên mỗi đầu ra biết đúng một đầu vào nó lấy. <code>Tensor.scatter_add_</code>{' '}
            đánh chỉ số theo đầu vào: <code>out[index[i]] += src[i]</code>, nên mỗi đầu vào biết nó đi đâu, và hai giá trị i có
            cùng chỉ số gặp nhau ở một đầu ra. Đó chính là xung đột của mục này, chỉ từ định nghĩa: bất kỳ{' '}
            <code>scatter_add_</code> song song nào mà hai i dùng chung một chỉ số đều phải bằng cách nào đó làm cho phép{' '}
            <code>+=</code> đó an toàn.
          </p>
        }
      >
        <p>
          You have used both directions already. <code>torch.gather</code> is output-indexed:{' '}
          <code>out[i] = src[index[i]]</code>, so each output knows the one input it takes. <code>Tensor.scatter_add_</code>{' '}
          is input-indexed: <code>out[index[i]] += src[i]</code>, so each input knows where it goes, and two values of i with
          the same index meet at one output. That is this section's conflict, from the definitions alone: any parallel{' '}
          <code>scatter_add_</code> in which two i share an index has to make that <code>+=</code> safe somehow.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Phép liên hệ này là của khóa học này, từ định nghĩa của hai phép toán. Nó không nói gì về cách PyTorch cài đặt
              phép nào trên backend nào.
            </p>
          }
        >
          <p>
            The bridge is ours, from the two operations' definitions. It says nothing about how PyTorch implements either on
            any backend.
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi={"Trên HIP và Metal"}>On HIP and Metal</Translated></h2>
      <Translated
        vi={
          <p>
            Một gather không cần điều nào dưới đây. Chi phí của một scatter, và cả cách viết nó, thay đổi theo backend. Bảng này
            dành cho hai scatter mà mục này đã gặp: một tổng float và một giá trị lớn nhất float vào các đầu ra dùng chung.
          </p>
        }
      >
        <p>
          A gather needs none of the following. A scatter's cost, and even its spelling, change by backend. The table is
          for the two scatters this section has met: a float sum and a float maximum into shared outputs.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Không có trong sách. Lấy từ notes/today/infra-facts.md (A1, B1, C1 và bảng backend của nó) và ch22-facts.md, kiểm
              tra ngày 2026-11-20. Không có gì ở đây được chạy trên phần cứng AMD.
            </p>
          }
        >
          <p>
            Not in the book. From notes/today/infra-facts.md (A1, B1, C1 and its backend table) and ch22-facts.md, checked
            2026-11-20. Nothing here was run on AMD hardware.
          </p>
        </Translated>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th><Translated vi="Scatter">Scatter</Translated></th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="Cộng float vào một đầu ra dùng chung">Float add into a shared output</Translated></td>
            <td><C>atomicAdd(float*, float)</C></td>
            <td><Translated vi={<>Cùng lời gọi; được biên dịch an toàn, có thể thành một vòng lặp compare-and-swap, trừ khi build với <C>-munsafe-fp-atomics</C></>}>Same call; compiled safely, possibly as a compare-and-swap loop, unless built with <C>-munsafe-fp-atomics</C></Translated></td>
            <td><Translated vi={<><C>atomic_fetch_add_explicit</C> trên một buffer khai báo <C>device atomic_float*</C>; chỉ add và sub; bộ nhớ threadgroup từ Metal 4.1</>}><C>atomic_fetch_add_explicit</C> on a buffer declared <C>device atomic_float*</C>; add and sub only; threadgroup memory from Metal 4.1</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Max float vào một đầu ra dùng chung">Float max into a shared output</Translated></td>
            <td><Translated vi="Không có overload cho float; một vòng lặp compare-and-swap">No float overload; a compare-and-swap loop</Translated></td>
            <td><Translated vi={<><C>atomicMax(float*, float)</C> có tồn tại (CUDA từ chối nó, nên không port ngược lại được)</>}><C>atomicMax(float*, float)</C> exists (CUDA rejects it, so not portable back)</Translated></td>
            <td><Translated vi="Không có tài liệu">None documented</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Nơi atomic của device chạy">Where device atomics run</Translated></td>
            <td><Translated vi="Tuần tự hóa ở L2 (ch09-facts A4)">Serialised in the L2 (ch09-facts A4)</Translated></td>
            <td><Translated vi="L2 cho bộ nhớ toàn cục (ch09-facts B2)">L2 for global memory (ch09-facts B2)</Translated></td>
            <td><Translated vi="Không có tài liệu (ch09-facts C4)">Not documented (ch09-facts C4)</Translated></td>
          </tr>
        </tbody>
      </table>
      <Ask q={questionsS222['ch22/22.2/q-hip-atomics']} />
      <Ask q={questionsS222['ch22/22.2/q-metal-max']} />

      <h2><Translated vi={"Thứ tự của một scatter"}>The order of a scatter</Translated></h2>
      <Translated
        vi={
          <p>
            Thêm một hệ quả của scatter mà chương 22 của sách không nêu. Atomic trên float cộng theo bất kỳ thứ tự nào chúng đến,
            và phép cộng dấu phẩy động không có tính kết hợp: phụ lục A tính một ví dụ trong đó hai thứ tự của cùng bốn số cho
            2.0 và 2.5. Vì vậy kết quả của một scatter có thể thay đổi ở các bit cuối giữa các lần chạy, trong khi một gather cộng
            theo thứ tự cố định cho cùng các bit mỗi lần. Theo cách nói của sách, điều này thuộc tiêu chí thứ tư, "các tiêu chí
            khác tùy ứng dụng".
          </p>
        }
      >
        <p>
          One more consequence of scattering, which the book's chapter 22 does not raise. Float atomics add in whatever order
          they arrive, and floating-point addition is not associative: appendix A works an example in which two orders of
          the same four numbers give 2.0 and 2.5. So a scatter's result can change in its last bits from run to run, while a
          gather that adds in a fixed order gives the same bits every time. In the book's terms this belongs to the fourth
          criterion, "others that depend on the application".
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Của khóa học này, dựa trên phụ lục A §A.6 (pp. 593–594). Chính tài liệu của PyTorch liệt kê{' '}
              <code>Tensor.scatter_add_()</code> trên một tensor CUDA trong số các phép toán thường không tất định của nó
              (notes/today/ch21-facts.md D1, kiểm tra ngày 2026-11-20).
            </p>
          }
        >
          <p>
            Ours, built on appendix A §A.6 (pp. 593–594). PyTorch's own documentation lists{' '}
            <code>Tensor.scatter_add_()</code> on a CUDA tensor among its normally nondeterministic operations
            (notes/today/ch21-facts.md D1, checked 2026-11-20).
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS222['ch22/22.2/q-determinism']} />
    </>
  );
}
