import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type BenchRow, type Reports } from '../../lib/reports';
import { Translated, tr as trText } from '../../lib/i18n';
import { href } from '../../lib/router';
import { strideCase, tileAccess } from '../../sim/banks';
import {
  EXPRESSIONS,
  SEGMENT_MODELS,
  exercise1,
  exprAccess,
  strideEfficiency,
  tileLoad,
  warpAccess,
  type AccessParams,
  type Arrangement,
  type Ex1Verdict,
  type ExprId,
  type Layout,
  type SegmentModelId,
} from '../../sim/coalesce';
import { CoalesceScene } from './CoalesceScene';

// The kernel of the book's exercise 1 (pp. 154–155), one source line per numbered
// line. A copy of labs/ch06/snippets/s61_foo_kernel.cu, which ./verify.sh
// front-end-checks; snippets-s61.test.ts fails if the two drift apart.
const FOO = `__global__ void foo_kernel(float* a, float* b, float* c, float* d, float* e) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ float a_s[256];
    __shared__ float bc_s[4*256];
    a_s[threadIdx.x] = a[i];
    for(unsigned int j = 0; j < 4; ++j) {
        bc_s[j*256 + threadIdx.x] = b[j*blockDim.x*gridDim.x + i] + c[i*4 + j];
    }
    __syncthreads();
    d[i + 8] = a_s[threadIdx.x];
    e[i*8] = bc_s[threadIdx.x*4];
}`;
const FOO_TIERS = {
  a: 'global', b: 'global', c: 'global', d: 'global', e: 'global', a_s: 'shared', bc_s: 'shared', i: 'reg', j: 'reg',
} as const;

const LAB61 = 'ch06/lab1_coalescing';
const pct = (x: number) => `${fmt(x * 100, 3)} %`;
const bytes = (n: number) => n.toLocaleString('en-US');

// ---------------------------------------------------------------- reports

/** Backends that are real GPUs: realBackends() drops the emulator but not its race-detection flavour. */
export const gpuBackends = (reports: Reports) =>
  realBackends(reports).filter((b) => b !== 'race' && !/emulator/i.test(String(reports[b].device?.name ?? '')));

/** Bench rows of a lab on one backend: the reference solution's if present, else the reader's own. */
function labRows(reports: Reports, backend: string, lab: string): BenchRow[] {
  const s = benchRows(reports[backend], lab, 'solution');
  return s.length ? s : benchRows(reports[backend], lab, 'work');
}

/** Warp widths the reader's devices reported: lab 6.1's simd_width row, else the device's warpSize. */
function useDeviceWidths(): { backend: string; device: string; width: number }[] {
  const { reports } = useReports();
  return gpuBackends(reports).flatMap((b) => {
    const row = labRows(reports, b, LAB61).find((r) => r.name === 'simd_width');
    const w = num(row?.width) ?? num(Number(reports[b].device.warpSize));
    return w ? [{ backend: b, device: String(reports[b].device.name ?? b), width: w }] : [];
  });
}

// ---------------------------------------------------------------- drawing

/** Below this width the explorer's drawings scroll sideways inside the bench instead of shrinking past legibility. */
const NARROW_MIN = 600;

/** One warp's lanes, arrows to their bytes, and the aligned segments those bytes fall in. */
function SegmentStrip({
  lanes,
  warpWidth,
  elemBytes,
  seg,
  unit,
}: {
  lanes: { lane: number; byte: number }[];
  warpWidth: number;
  elemBytes: number;
  seg: number;
  unit: string;
}) {
  const a = warpAccess(lanes.map((l) => l.byte), elemBytes, seg);
  const W = 700;
  const lw = Math.min(20, (W - 40) / warpWidth);
  const lx = (l: number) => 20 + l * lw;
  const n = a.segments.length;
  const gaps = a.segments.filter((s, i) => i > 0 && s !== a.segments[i - 1] + 1).length;
  const bw = Math.min(64, (W - 40 - gaps * 10) / Math.max(1, n));
  const xs: number[] = [];
  let x = 0;
  a.segments.forEach((s, i) => {
    if (i > 0 && s !== a.segments[i - 1] + 1) x += 10;
    xs.push(x);
    x += bw;
  });
  const off = (W - x) / 2;
  const boxX = (i: number) => off + xs[i];
  const posOf = (byte: number) => {
    const i = a.segments.indexOf(Math.floor(byte / seg));
    return boxX(i) + ((byte % seg) / seg) * bw;
  };
  const active = new Set(lanes.map((l) => l.lane));
  const SY = 104;
  const servedBy = (s: number) => lanes.filter((l) => Math.floor(l.byte / seg) <= s && Math.floor((l.byte + elemBytes - 1) / seg) >= s).map((l) => l.lane);
  return (
    <div style={{ overflowX: 'auto' }}>
    <svg viewBox={`0 0 ${W} 166`} role="img" aria-label={trText('Lanes, their addresses and the memory segments they touch', 'Các lane, địa chỉ của chúng và các đoạn bộ nhớ chúng chạm vào')} style={{ width: '100%', minWidth: NARROW_MIN, display: 'block' }} data-segments={n}>
      <text x={20} y={12} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="các lane của một warp">lanes of one warp</Translated></text>
      {Array.from({ length: warpWidth }, (_, l) => (
        <g key={l} opacity={active.has(l) ? 1 : 0.25}>
          <rect x={lx(l)} y={18} width={lw - 2} height={16} rx="2" fill="var(--cell)" stroke="var(--ink-3)" />
          {lw >= 14 ? <text x={lx(l) + (lw - 2) / 2} y={29.5} fontSize="8.5" textAnchor="middle">{l}</text> : null}
        </g>
      ))}
      {lanes.map((l) => (
        <line key={l.lane} x1={lx(l.lane) + (lw - 2) / 2} y1={35} x2={posOf(l.byte) + 0.5} y2={SY - 1} stroke="var(--global)" strokeWidth="0.8" opacity="0.75" />
      ))}
      {a.segments.map((s, i) => {
        const lo = s * seg;
        return (
          <g key={s} data-segment={s}>
            <rect x={boxX(i)} y={SY} width={bw} height={20} fill="var(--global-wash)" stroke="var(--global)" strokeWidth="1" />
            {lanes
              .filter((l) => l.byte < lo + seg && l.byte + elemBytes > lo)
              .map((l) => {
                const a0 = Math.max(l.byte, lo);
                const a1 = Math.min(l.byte + elemBytes, lo + seg);
                return <rect key={l.lane} x={boxX(i) + ((a0 - lo) / seg) * bw} y={SY} width={((a1 - a0) / seg) * bw} height={20} fill="var(--global)" />;
              })}
            {i > 0 && s !== a.segments[i - 1] + 1 ? (
              <text x={boxX(i) - 5} y={SY + 14} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>⋯</text>
            ) : null}
            {n <= 8 ? (
              <text x={boxX(i) + bw / 2} y={SY + 34} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
                {(() => {
                  const ls = servedBy(s);
                  return ls.length === 1 ? `lane ${ls[0]}` : ls.length === warpWidth || (ls.length > 1 && ls.every((v, j) => j === 0 || v === ls[j - 1] + 1)) ? `lanes ${ls[0]}–${ls[ls.length - 1]}` : trText(`${ls.length} lanes`, `${ls.length} lane`); // i18n-ignore: lane numbers
                })()}
              </text>
            ) : null}
          </g>
        );
      })}
      <text x={20} y={SY + 52} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
        <Translated
          vi={
            <>
              {n} {({ sector: 'sector', line: 'dòng', segment: 'đoạn' } as Record<string, string>)[unit] ?? unit} {seg} byte đã
              căn lề, byte {bytes(a.segments[0] * seg)} đến {bytes((a.segments[n - 1] + 1) * seg - 1)}
              {gaps ? ' (khoảng trống không theo tỉ lệ)' : ''}; tô đậm: các byte warp yêu cầu
            </>
          }
        >
          {n} aligned {seg}-byte {unit}{n === 1 ? '' : 's'}, bytes {bytes(a.segments[0] * seg)} to {bytes((a.segments[n - 1] + 1) * seg - 1)}
          {gaps ? ' (gaps not to scale)' : ''}; solid: bytes the warp asked for
        </Translated>
      </text>
    </svg>
    </div>
  );
}

/** A window onto the logical matrix around the cells the warp reads. */
function MatrixWindow({ name, Width, cells, colMajor }: { name: string; Width: number; cells: { r: number; c: number }[]; colMajor: boolean }) {
  const rs = cells.map((x) => x.r);
  const cs = cells.map((x) => x.c);
  const nR = Math.min(Width, 6);
  const nC = Math.min(Width, Math.max(...cs) - Math.min(...cs) + 1 + 4, 68);
  const r0 = Math.max(0, Math.min(Math.min(...rs) - 2, Width - nR));
  const c0 = Math.max(0, Math.min(Math.min(...cs) - 2, Width - nC));
  const cell = Math.min(16, 640 / nC);
  const hot = new Set(cells.map((x) => `${x.r},${x.c}`));
  return (
    <div style={{ overflowX: 'auto' }}>
    <svg viewBox={`0 0 700 ${nR * cell + 46}`} role="img" aria-label={trText(`Logical view of ${name}`, `Góc nhìn logic của ${name}`)} style={{ width: '100%', minWidth: NARROW_MIN, display: 'block' }}>
      <text x={20} y={12} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
        <Translated
          vi={
            <>
              {name}, hàng {r0}–{r0 + nR - 1}, cột {c0}–{c0 + nC - 1} trên {Width}; lưu{' '}
              {colMajor ? 'theo từng cột (địa chỉ chạy dọc xuống)' : 'theo từng hàng (địa chỉ chạy ngang)'}
            </>
          }
        >
          {name}, rows {r0}–{r0 + nR - 1}, columns {c0}–{c0 + nC - 1} of {Width}; stored {colMajor ? 'column by column (addresses run down)' : 'row by row (addresses run across)'}
        </Translated>
      </text>
      {Array.from({ length: nR * nC }, (_, i) => {
        const r = r0 + Math.floor(i / nC);
        const c = c0 + (i % nC);
        const on = hot.has(`${r},${c}`);
        return (
          <rect key={i} x={20 + (c - c0) * cell} y={20 + (r - r0) * cell} width={cell - 1} height={cell - 1}
            fill={on ? 'var(--global)' : 'var(--cell)'} stroke={on ? 'var(--global)' : 'var(--rule)'} strokeWidth="0.6" />
        );
      })}
      <text x={20} y={nR * cell + 36} fontSize="10" style={{ fill: 'var(--ink-3)' }}><Translated vi="tô đậm: các phần tử warp này đọc ở bước này">solid: the elements this warp reads at this step</Translated></text>
    </svg>
    </div>
  );
}

// ---------------------------------------------------------------- the explorer

const EXPR_ORDER: { id: ExprId; label: string }[] = [
  {
    id: 'n-row',
    get label() {
      return trText('N[k*Width + col], N row-major (Fig. 6.2)', 'N[k*Width + col], N thứ tự hàng (Hình 6.2)');
    },
  },
  {
    id: 'n-col',
    get label() {
      return trText('N[col*Width + k], N column-major (Fig. 6.3)', 'N[col*Width + k], N thứ tự cột (Hình 6.3)');
    },
  },
  {
    id: 'm-row',
    get label() {
      return trText('M[row*Width + k] (the M read of both figures)', 'M[row*Width + k] (lần đọc M của cả hai hình)');
    },
  },
  { id: 'stride', label: 'in[i*stride]' },
  { id: 'offset', label: 'a[i + offset]' },
  {
    id: 'ex1-a',
    get label() {
      return trText('Exercise 1(a): a[i]', 'Bài tập 1(a): a[i]');
    },
  },
  {
    id: 'ex1-c',
    get label() {
      return trText('Exercise 1(c): b[j*blockDim.x*gridDim.x + i]', 'Bài tập 1(c): b[j*blockDim.x*gridDim.x + i]');
    },
  },
  {
    id: 'ex1-d',
    get label() {
      return trText('Exercise 1(d): c[i*4 + j]', 'Bài tập 1(d): c[i*4 + j]');
    },
  },
  {
    id: 'ex1-g',
    get label() {
      return trText('Exercise 1(g): d[i + 8]', 'Bài tập 1(g): d[i + 8]');
    },
  },
  {
    id: 'ex1-i',
    get label() {
      return trText('Exercise 1(i): e[i*8]', 'Bài tập 1(i): e[i*8]');
    },
  },
];

interface ExplorerState {
  expr: ExprId;
  Width: number;
  k: number;
  bdx: number; // 0 = at least as wide as the warp
  stride: number;
  offset: number;
  j: number;
  warp: number;
  elem: number;
  model: SegmentModelId;
  custom: number; // segment size chosen for Metal
}
const START: ExplorerState = { expr: 'n-row', Width: 1024, k: 0, bdx: 0, stride: 1, offset: 0, j: 0, warp: 32, elem: 4, model: 'nvidia', custom: 64 };
const PRESETS: { label: string; s: Partial<ExplorerState> }[] = [
  {
    get label() {
      return trText('Fig. 6.2: row-major N', 'Hình 6.2: N thứ tự hàng');
    },
    s: { expr: 'n-row' },
  },
  {
    get label() {
      return trText('Fig. 6.3: column-major N', 'Hình 6.3: N thứ tự cột');
    },
    s: { expr: 'n-col' },
  },
  {
    get label() {
      return trText('The M read', 'Lần đọc M');
    },
    s: { expr: 'm-row' },
  },
  {
    get label() {
      return trText('One float off', 'Lệch một float');
    },
    s: { expr: 'offset', offset: 1 },
  },
  {
    get label() {
      return trText('Stride 2', 'Bước nhảy 2');
    },
    s: { expr: 'stride', stride: 2 },
  },
];

function CoalescingExplorer() {
  const [s, setS] = useState<ExplorerState>(START);
  const set = (o: Partial<ExplorerState>) => setS((x) => ({ ...x, ...o }));
  const widths = useDeviceWidths();
  const info = EXPRESSIONS[s.expr];
  const model = SEGMENT_MODELS[s.model];
  const seg = model.segmentBytes ?? s.custom;
  const elem = info.matrix ? 4 : s.elem;
  const params = (k: number): AccessParams => ({
    Width: s.Width, k, row: 0, stride: s.stride, offset: s.offset, warpWidth: s.warp, lane0: 0,
    blockDimX: s.bdx || undefined, j: s.j, gridThreads: 1024,
  });
  const a = exprAccess(s.expr, params(s.k), seg, elem);
  const lines = model.id === 'nvidia' ? warpAccess(a.lanes.map((l) => l.byte), elem, 128).transactions : null;
  const total = useMemo(() => {
    if (!info.matrix) return null;
    let f = 0;
    let n = 0;
    for (let k = 0; k <= s.k; k++) {
      const x = exprAccess(s.expr, params(k), seg, elem);
      f += x.bytesFetched;
      n += x.bytesNeeded;
    }
    return { fetched: f, needed: n };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s, seg, elem]);
  const cells = info.matrix ? a.lanes.map((l) => ({ r: l.r!, c: l.c! })) : [];

  return (
    <Bench
      title={trText('Coalescing explorer', 'Trình khám phá hợp nhất truy cập bộ nhớ')}
      task={trText(
        'Predict the count for each preset, then check it. Change one thing at a time.',
        'Dự đoán con số cho mỗi thiết lập sẵn, rồi kiểm tra. Mỗi lần chỉ thay đổi một thứ.',
      )}
      foot={
        <Translated
          vi={
            <>
              Đếm một lệnh nạp của một warp: số đoạn đã căn lề khác nhau mà các lane của nó chạm vào, theo đơn vị mà nhà
              cung cấp được chọn công bố ({model.facts}). Mảng bắt đầu ở một biên 256 byte. Không có bộ nhớ đệm, không đo
              thời gian.
            </>
          }
        >
          <>
            Counts one load instruction of one warp: the distinct aligned segments its lanes touch, in the unit the
            chosen vendor documents ({model.facts}). Arrays start on a 256-byte boundary. No caches, no timing.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="muted">
          <Translated vi="Thiết lập sẵn:">Presets:</Translated>
        </span>
        {PRESETS.map((p) => (
          <button key={p.label} className="quiet" onClick={() => setS({ ...START, warp: s.warp, model: s.model, custom: s.custom, ...p.s })}>
            {p.label}
          </button>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Truy cập">Access</Translated>
          <select value={s.expr} onChange={(e) => set({ expr: e.target.value as ExprId, k: 0 })} style={{ maxWidth: '100%' }}>
            {EXPR_ORDER.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
          </select>
        </label>
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <select value={s.warp} onChange={(e) => set({ warp: Number(e.target.value) })} style={{ maxWidth: '100%' }}>
            <option value={32}>{trText('32 lanes (NVIDIA; AMD RDNA)', '32 lane (NVIDIA; AMD RDNA)')}</option>
            <option value={64}>{trText('64 lanes (AMD CDNA)', '64 lane (AMD CDNA)')}</option>
            {widths.filter((w) => w.width !== 32 && w.width !== 64).map((w) => (
              <option key={w.backend} value={w.width}>
                <Translated vi={<>{w.width} lane (thiết bị của bạn: {w.device})</>}>
                  <>{w.width} lanes (your device: {w.device})</>
                </Translated>
              </option>
            ))}
          </select>
        </label>
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Mô hình đoạn">Segment model</Translated>
          <select value={s.model} onChange={(e) => set({ model: e.target.value as SegmentModelId })} style={{ maxWidth: '100%' }}>
            {Object.values(SEGMENT_MODELS).map((m) => (
              <option key={m.id} value={m.id}>
                <Translated
                  vi={
                    {
                      nvidia: 'NVIDIA: sector 32 byte trong dòng 128 byte',
                      amd64: 'AMD: dòng cache 64 byte',
                      amd128: 'AMD: dòng cache 128 byte',
                      metal: 'Apple: không có tài liệu',
                    }[m.id]
                  }
                >
                  {m.label}
                </Translated>
              </option>
            ))}
          </select>
        </label>
        {model.segmentBytes == null ? (
          <label className="field">
            <Translated vi="Lựa chọn của bạn (không phải sự thật)">Your choice (not a fact)</Translated>
            <select value={s.custom} onChange={(e) => set({ custom: Number(e.target.value) })}>
              {[32, 64, 128].map((v) => (
                <option key={v} value={v}>
                  <Translated vi={<>{v} byte</>}>
                    <>{v} bytes</>
                  </Translated>
                </option>
              ))}
            </select>
          </label>
        ) : null}
      </div>
      <p className="muted" style={{ margin: '0 0 0.7rem', fontSize: '0.84rem' }} data-device-widths={widths.map((w) => w.width).join(',')}>
        {widths.length
          ? (
            <Translated
              vi={<>Thiết bị của bạn đã báo cáo: {widths.map((w) => `${w.device} (${w.backend}), ${w.width} lane`).join('; ')}.</>}
            >
              <>Your device{widths.length > 1 ? 's' : ''} reported: {widths.map((w) => `${w.device} (${w.backend}), ${w.width} lanes`).join('; ')}.</>
            </Translated>
          )
          : (
            <Translated vi={<>Các thiết bị của bạn chưa báo cáo độ rộng warp nào; lab 6.1 ghi lại nó.</>}>
              <>No warp width reported by your devices yet; lab 6.1 records it.</>
            </Translated>
          )}
      </p>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        {info.matrix ? (
          <>
            <label className="field">
              Width = {s.Width}
              <input type="range" min={4} max={2048} value={s.Width} aria-label={trText('Width', 'Width')}
                onChange={(e) => { const W = Number(e.target.value); set({ Width: W, k: Math.min(s.k, W - 1) }); }} />
            </label>
            <label className="field">
              <Translated vi="Độ rộng khối (blockDim.x)">Block width (blockDim.x)</Translated>
              <select value={s.bdx} onChange={(e) => set({ bdx: Number(e.target.value) })}>
                <option value={0}>{trText('as wide as the warp, or wider', 'rộng bằng warp, hoặc rộng hơn')}</option>
                <option value={16}>16</option>
                <option value={8}>8</option>
              </select>
            </label>
            <span className="row" style={{ gap: '0.4rem' }}>
              <button onClick={() => set({ k: 0 })} disabled={s.k === 0}>k = 0</button>
              <button onClick={() => set({ k: s.k - 1 })} disabled={s.k === 0}>k − 1</button>
              <button className="primary" onClick={() => set({ k: s.k + 1 })} disabled={s.k >= s.Width - 1}>k + 1</button>
              <span className="muted" aria-live="polite">k = {s.k}</span>
            </span>
          </>
        ) : null}
        {s.expr === 'stride' ? (
          <label className="field">
            <Translated vi={<>stride = {s.stride} phần tử</>}>
              <>stride = {s.stride} element{s.stride === 1 ? '' : 's'}</>
            </Translated>
            <input type="range" min={1} max={64} value={s.stride} aria-label={trText('stride', 'stride')} onChange={(e) => set({ stride: Number(e.target.value) })} />
          </label>
        ) : null}
        {s.expr === 'offset' ? (
          <label className="field">
            <Translated vi={<>offset = {s.offset} phần tử</>}>
              <>offset = {s.offset} element{s.offset === 1 ? '' : 's'}</>
            </Translated>
            <input type="range" min={0} max={32} value={s.offset} aria-label={trText('offset', 'offset')} onChange={(e) => set({ offset: Number(e.target.value) })} />
          </label>
        ) : null}
        {s.expr === 'ex1-c' || s.expr === 'ex1-d' ? (
          <label className="field">
            j
            <select value={s.j} onChange={(e) => set({ j: Number(e.target.value) })}>
              {[0, 1, 2, 3].map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
        ) : null}
        {!info.matrix ? (
          <label className="field">
            <Translated vi="Kích thước phần tử">Element size</Translated>
            <select value={s.elem} onChange={(e) => set({ elem: Number(e.target.value) })}>
              <option value={4}>{trText('4 bytes (float)', '4 byte (float)')}</option>
              <option value={16}>{trText('16 bytes (float4)', '16 byte (float4)')}</option>
            </select>
          </label>
        ) : null}
      </div>

      <p className="narrate" style={{ margin: '0 0 0.8rem' }}>
        <code>{info.code}</code>
        {info.matrix
          ? trText(
              ` with Width = ${s.Width}, k = ${s.k}: lane 0 reads element ${a.lanes[0]?.element ?? '–'}, lane 1 reads ${a.lanes[1]?.element ?? '–'}${a.lanes.length > 1 ? `, ${a.lanes[1].element - a.lanes[0].element} element${Math.abs(a.lanes[1].element - a.lanes[0].element) === 1 ? '' : 's'} further on` : ''}.`,
              ` với Width = ${s.Width}, k = ${s.k}: lane 0 đọc phần tử ${a.lanes[0]?.element ?? '–'}, lane 1 đọc ${a.lanes[1]?.element ?? '–'}${a.lanes.length > 1 ? `, xa thêm ${a.lanes[1].element - a.lanes[0].element} phần tử` : ''}.`,
            )
          : trText(
              `: lane 0 reads element ${a.lanes[0].element}, lane 1 reads ${a.lanes[1].element}, lane ${a.lanes.length - 1} reads ${a.lanes[a.lanes.length - 1].element}.`,
              `: lane 0 đọc phần tử ${a.lanes[0].element}, lane 1 đọc ${a.lanes[1].element}, lane ${a.lanes.length - 1} đọc ${a.lanes[a.lanes.length - 1].element}.`,
            )}
        {a.lanes.length < s.warp
          ? trText(
              ` Only ${a.lanes.length} lanes are active: the kernel's guard idles the lanes past column ${s.Width - 1}.`,
              ` Chỉ ${a.lanes.length} lane hoạt động: phép kiểm tra biên của kernel để các lane vượt quá cột ${s.Width - 1} ngồi không.`,
            )
          : ''}
      </p>

      {info.matrix ? <MatrixWindow name={s.expr === 'm-row' ? 'M' : 'N'} Width={s.Width} cells={cells} colMajor={s.expr === 'n-col'} /> : null}
      <SegmentStrip lanes={a.lanes} warpWidth={s.warp} elemBytes={elem} seg={seg} unit={model.unit} />

      <div className="row" style={{ marginTop: '0.6rem' }} data-readout
        data-transactions={a.transactions} data-fetched={a.bytesFetched} data-needed={a.bytesNeeded} data-efficiency={a.efficiency} data-runs={a.runs}>
        <span>
          <Translated
            vi={
              <>
                <b>{a.transactions}</b> {({ sector: 'sector', line: 'dòng', segment: 'đoạn' } as Record<string, string>)[model.unit] ?? model.unit}{' '}
                {seg} byte{model.segmentBytes == null ? ' (lựa chọn của bạn)' : ''}
              </>
            }
          >
            <><b>{a.transactions}</b> {model.unit}{a.transactions === 1 ? '' : 's'} of {seg} bytes{model.segmentBytes == null ? ' (your choice)' : ''}</>
          </Translated>
        </span>
        {lines != null ? (
          <span>
            <Translated vi={<>trong <b>{lines}</b> dòng 128 byte</>}>
              <>in <b>{lines}</b> 128-byte line{lines === 1 ? '' : 's'}</>
            </Translated>
          </span>
        ) : null}
        <span>
          <Translated vi={<>đã nạp <b>{bytes(a.bytesFetched)}</b> byte</>}>
            <>fetched <b>{bytes(a.bytesFetched)}</b> bytes</>
          </Translated>
        </span>
        <span>
          <Translated vi={<>cần <b>{bytes(a.bytesNeeded)}</b></>}>
            <>needed <b>{bytes(a.bytesNeeded)}</b></>
          </Translated>
        </span>
        <span className={'chip ' + (a.efficiency === 1 ? 'ok' : a.efficiency <= 0.25 ? 'fail' : 'global')}>
          <Translated vi={<>{pct(a.efficiency)} số byte đã nạp được dùng</>}>
            <>{pct(a.efficiency)} of fetched bytes used</>
          </Translated>
        </span>
        <span className="muted">
          {a.runs === 1 ? trText('one run of consecutive bytes', 'một dải byte liên tiếp') : trText(`${a.runs} separate runs`, `${a.runs} dải riêng biệt`)}
        </span>
      </div>
      {total && s.k > 0 ? (
        <p className="muted" style={{ margin: '0.5rem 0 0' }} data-total-fetched={total.fetched} data-total-needed={total.needed}>
          <Translated
            vi={
              <>
                Qua k = 0 đến {s.k}, đếm lại mỗi bước từ đầu: đã nạp {bytes(total.fetched)} byte cho {bytes(total.needed)} byte cần
                {total.fetched !== total.needed ? ` (gấp ${fmt(total.fetched / total.needed)} lần)` : ''}.
              </>
            }
          >
            <>
              Over k = 0 to {s.k}, counting each step afresh: fetched {bytes(total.fetched)} bytes for {bytes(total.needed)} needed
              {total.fetched !== total.needed ? ` (${fmt(total.fetched / total.needed)} times as much)` : ''}.
            </>
          </Translated>
        </p>
      ) : null}
      {model.id === 'metal' ? (
        <p className="muted" style={{ margin: '0.5rem 0 0' }}>
          <Translated vi="Apple không công bố quy tắc hợp nhất truy cập, kích thước dòng hay kích thước sector nào (facts C1). Con số ở trên dùng kích thước bạn chọn; lab 6.1 là cách để biết một máy Mac làm gì.">
            Apple documents no coalescing rule, line size or sector size (facts C1). The count above uses the size you picked; lab 6.1 is how to find out what a Mac does.
          </Translated>
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- corner-turning stepper

const TILES = [4, 8, 16, 32];

function CornerTurnStepper() {
  const [layout, setLayout] = useState<Layout>('col');
  const [arr, setArr] = useState<Arrangement>('ch5');
  const [T, setT] = useState(4);
  const [warpW, setWarpW] = useState(4);
  const [Width, setWidth] = useState(16);
  const [bx, setBx] = useState(1);
  const [ph, setPh] = useState(2);
  const [w, setW] = useState(0);
  const [tx, setTx] = useState(3);
  const [ty, setTy] = useState(1);

  const widthsOk = [16, 64, 1024].filter((v) => v >= 2 * T);
  const W = widthsOk.includes(Width) ? Width : widthsOk[0];
  const nb = Math.min(W / T, 8);
  const BX = Math.min(bx, nb - 1);
  const PH = Math.min(ph, nb - 1);
  const ww = T === 4 ? warpW : Math.max(32, warpW);
  const load = tileLoad({ layout, arrangement: arr, T, bx: BX, ph: PH, Width: W, warpWidth: ww });
  const nw = load.warps.length;
  const cw = load.warps[Math.min(w, nw - 1)];
  const TX = Math.min(tx, T - 1);
  const TY = Math.min(ty, T - 1);
  const sel = load.threads.find((t) => t.tx === TX && t.ty === TY)!;
  const nb1 = load.threads.find((t) => t.tx === Math.max(0, TX - 1) && t.ty === TY)!;
  const inWarp = new Set(cw.threads.map((t) => `${t.sharedRow},${t.sharedCol}`));
  const cell = Math.min(26, 360 / T);
  const store =
    ww >= 32
      ? tileAccess({ T, pitch: T, orientation: arr === 'turned' ? 'x-first' : 'y-first', ty0: cw.threads[0].ty, tx0: cw.threads[0].tx, warpWidth: cw.threads.length })
      : null;
  const turned = arr === 'turned';
  const rowExpr = turned ? 'ph·T + tx' : 'ph·T + ty'; // i18n-ignore: index expressions
  const colExpr = turned ? 'bx·T + ty' : 'bx·T + tx'; // i18n-ignore: index expressions

  return (
    <Bench
      title={trText('Corner-turning stepper', 'Trình chạy từng bước xoay góc')}
      task={trText(
        "Step through the warps of one block's N-tile load in both arrangements. Which one gives each warp one run of addresses?",
        'Đi qua từng warp trong phép nạp ô N của một khối ở cả hai cách sắp xếp. Cách nào cho mỗi warp một dải địa chỉ duy nhất?',
      )}
      foot={
        <Translated
          vi={
            <>
              Ô N của một khối trong một pha của kernel phân ô. Luồng (tx, ty) là lane (ty·T + tx) mod độ rộng warp. Sector
              là đơn vị 32 byte của NVIDIA. Warp bốn luồng chỉ có trong hình của sách; warp thật có 32 hoặc 64 lane.
            </>
          }
        >
          <>
            The N tile of one block in one phase of the tiled kernel. Thread (tx, ty) is lane (ty·T + tx) mod the warp width.
            Sectors are NVIDIA's 32 bytes. The four-thread warp exists only in the book's figure; real warps are 32 or 64 lanes.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span className="seg" role="group" aria-label={trText('Arrangement', 'Cách sắp xếp')}>
          <button aria-pressed={!turned} onClick={() => setArr('ch5')}>
            <Translated vi="(a) ánh xạ của chương 5">(a) chapter 5's mapping</Translated>
          </button>
          <button aria-pressed={turned} onClick={() => setArr('turned')}>
            <Translated vi="(b) đã xoay góc">(b) corner turned</Translated>
          </button>
        </span>
        <label className="field">
          <Translated vi="N được lưu">N is stored</Translated>
          <select value={layout} onChange={(e) => setLayout(e.target.value as Layout)}>
            <option value="col">{trText('column by column', 'theo từng cột')}</option>
            <option value="row">{trText('row by row', 'theo từng hàng')}</option>
          </select>
        </label>
        <label className="field">
          <Translated vi="Độ rộng ô T">Tile width T</Translated>
          <select value={T} onChange={(e) => { const v = Number(e.target.value); setT(v); setW(0); if (v !== 4 && warpW < 32) setWarpW(32); }}>
            {TILES.map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <select value={ww} onChange={(e) => { setWarpW(Number(e.target.value)); setW(0); }}>
            {T === 4 ? <option value={4}>{trText("4 (the figure's toy)", '4 (bản minh họa của hình)')}</option> : null}
            <option value={32}>32</option>
            <option value={64}>64</option>
          </select>
        </label>
        <label className="field">
          <Translated vi="Width">Width</Translated>
          <select value={W} onChange={(e) => setWidth(Number(e.target.value))}>
            {widthsOk.map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className="field">
          bx
          <select value={BX} onChange={(e) => setBx(Number(e.target.value))}>
            {Array.from({ length: nb }, (_, i) => <option key={i} value={i}>{i}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="pha">phase</Translated>
          <select value={PH} onChange={(e) => setPh(Number(e.target.value))}>
            {Array.from({ length: nb }, (_, i) => <option key={i} value={i}>{i}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setW(Math.max(0, cw.warp - 1))} disabled={cw.warp === 0}>
          <Translated vi="Warp trước">Previous warp</Translated>
        </button>
        <button className="primary" onClick={() => setW(Math.min(nw - 1, cw.warp + 1))} disabled={cw.warp >= nw - 1}>
          <Translated vi="Warp sau">Next warp</Translated>
        </button>
        <span className="muted" aria-live="polite">
          <Translated vi={<>warp {cw.warp + 1} trên {nw}</>}>
            <>warp {cw.warp + 1} of {nw}</>
          </Translated>
        </span>
        <label className="field">
          <Translated vi="Luồng tx">Thread tx</Translated>
          <select value={TX} onChange={(e) => setTx(Number(e.target.value))}>
            {Array.from({ length: T }, (_, i) => <option key={i} value={i}>{i}</option>)}
          </select>
        </label>
        <label className="field">
          ty
          <select value={TY} onChange={(e) => setTy(Number(e.target.value))}>
            {Array.from({ length: T }, (_, i) => <option key={i} value={i}>{i}</option>)}
          </select>
        </label>
      </div>

      <p className="narrate" style={{ margin: '0 0 0.8rem' }} data-thread-element={sel.element}>
        <Translated
          vi={
            <>
              Luồng (tx {TX}, ty {TY}) nạp N({rowExpr}, {colExpr}) = N({PH}·{T} + {turned ? TX : TY}, {BX}·{T} + {turned ? TY : TX}) = N({sel.r}, {sel.c}).{' '}
            </>
          }
        >
          <>
            Thread (tx {TX}, ty {TY}) loads N({rowExpr}, {colExpr}) = N({PH}·{T} + {turned ? TX : TY}, {BX}·{T} + {turned ? TY : TX}) = N({sel.r}, {sel.c}).{' '}
          </>
        </Translated>
        {layout === 'col'
          ? (
            <Translated vi={<>Theo từng cột, tức là column·Width + row = {sel.c}·{W} + {sel.r} = <b>{sel.element}</b>.</>}>
              <>Column by column, that is column·Width + row = {sel.c}·{W} + {sel.r} = <b>{sel.element}</b>.</>
            </Translated>
          )
          : (
            <Translated vi={<>Theo từng hàng, tức là row·Width + column = {sel.r}·{W} + {sel.c} = <b>{sel.element}</b>.</>}>
              <>Row by row, that is row·Width + column = {sel.r}·{W} + {sel.c} = <b>{sel.element}</b>.</>
            </Translated>
          )}
        {TX > 0 ? (
          <Translated vi={<> Luồng láng giềng tx {TX - 1} nạp {nb1.element}, cách {Math.abs(sel.element - nb1.element)} phần tử.</>}>
            <> Its neighbour tx {TX - 1} loads {nb1.element}, {Math.abs(sel.element - nb1.element)} element{Math.abs(sel.element - nb1.element) === 1 ? '' : 's'} away.</>
          </Translated>
        ) : null}{' '}
        <Translated
          vi={<>Nó ghi giá trị vào Nds[{sel.sharedRow}][{sel.sharedCol}], tức là Nds[{turned ? 'tx' : 'ty'}][{turned ? 'ty' : 'tx'}].</>}
        >
          <>It stores the value at Nds[{sel.sharedRow}][{sel.sharedCol}], which is Nds[{turned ? 'tx' : 'ty'}][{turned ? 'ty' : 'tx'}].</>
        </Translated>
      </p>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 3fr)' }}>
        <div>
          <div className="mx-label t-global">
            <Translated vi={<>ô N (hàng {PH * T}–{PH * T + T - 1}, cột {BX * T}–{BX * T + T - 1})</>}>
              <>the N tile (rows {PH * T}–{PH * T + T - 1}, columns {BX * T}–{BX * T + T - 1})</>
            </Translated>
          </div>
          <svg viewBox={`0 0 ${T * cell + 2} ${T * cell + 2}`} style={{ width: '100%', maxWidth: T * cell + 2, display: 'block' }} role="img" aria-label={trText("The tile, with the current warp's elements marked", 'Ô, với các phần tử của warp hiện tại được đánh dấu')}>
            {[...load.threads.filter((t) => t !== sel), sel].map((t) => {
              const me = t === sel;
              const on = inWarp.has(`${t.sharedRow},${t.sharedCol}`);
              return (
                <g key={t.t}>
                  <rect x={1 + t.sharedCol * cell} y={1 + t.sharedRow * cell} width={cell - 1.5} height={cell - 1.5}
                    fill={on ? 'var(--global)' : 'var(--global-wash)'} stroke={me ? 'var(--ink)' : 'var(--rule)'} strokeWidth={me ? 2.4 : 0.5} />
                  {T === 4 ? (
                    <text x={1 + t.sharedCol * cell + cell / 2} y={1 + t.sharedRow * cell + cell / 2 + 3} fontSize="8" textAnchor="middle"
                      fontWeight={me ? 700 : 400} style={{ fill: on ? 'var(--page)' : 'var(--ink)' }}>{t.element}</text>
                  ) : null}
                </g>
              );
            })}
          </svg>
          <p className="muted" style={{ fontSize: '0.8rem', margin: '0.3rem 0 0' }}>
            <Translated
              vi={
                <>
                  Tô màu: các phần tử warp này nạp. Viền đậm: phần tử của luồng được chọn.{' '}
                  {T === 4 ? 'Các con số là chỉ số phẳng vào N.' : ''}
                </>
              }
            >
              <>
                Shaded: elements this warp loads. Outlined: the selected thread's element. {T === 4 ? 'Numbers are flat indices into N.' : ''}
              </>
            </Translated>
          </p>
        </div>
        <div>
          <div className="row" data-readout data-transactions={cw.access.transactions} data-efficiency={cw.access.efficiency} data-runs={cw.runs}>
            <span>
              <Translated vi={<><b>{cw.access.transactions}</b> sector cho warp này</>}>
                <><b>{cw.access.transactions}</b> sector{cw.access.transactions === 1 ? '' : 's'} for this warp</>
              </Translated>
            </span>
            <span className={'chip ' + (cw.access.efficiency === 1 ? 'ok' : cw.access.efficiency <= 0.25 ? 'fail' : 'global')}>
              <Translated vi={<>{pct(cw.access.efficiency)} được dùng</>}>
                <>{pct(cw.access.efficiency)} used</>
              </Translated>
            </span>
            <span className="muted">
              {cw.runs === 1 ? trText('one run of consecutive addresses', 'một dải địa chỉ liên tiếp') : trText(`${cw.runs} separate runs`, `${cw.runs} dải riêng biệt`)}
            </span>
          </div>
          {store ? (
            <p className="muted" style={{ margin: '0.6rem 0 0' }} data-store-passes={store.passes}>
              <Translated
                vi={
                  <>
                    Phép ghi vào ô chia sẻ (<code>{turned ? 'Nds[tx][ty]' : 'Nds[ty][tx]'}</code>, {T} từ mỗi hàng) chạm vào{' '}
                    {store.banksUsed} bank:{' '}
                    {store.passes === 1 ? 'một lượt, không xung đột' : <b className="t-fail">{store.passes} lượt theo quy tắc bank của NVIDIA</b>}.{' '}
                    {turned && store.passes > 1 ? <>Mục <a href={href('ch06', '6.4')}>6.4</a> nói về điều này.</> : null}
                  </>
                }
              >
                <>
                  The store into the shared tile (<code>{turned ? 'Nds[tx][ty]' : 'Nds[ty][tx]'}</code>, {T} words per row) touches{' '}
                  {store.banksUsed} bank{store.banksUsed === 1 ? '' : 's'}:{' '}
                  {store.passes === 1 ? 'one pass, no conflict' : <b className="t-fail">{store.passes} passes under NVIDIA's bank rule</b>}.{' '}
                  {turned && store.passes > 1 ? <>Section <a href={href('ch06', '6.4')}>6.4</a> is about this.</> : null}
                </>
              </Translated>
            </p>
          ) : (
            <p className="muted" style={{ margin: '0.6rem 0 0' }}>
              <Translated vi="Với warp bốn luồng thì không có số bank nào để hiển thị; hãy chọn độ rộng warp 32 để xem phép ghi vào bộ nhớ chia sẻ.">
                With a four-thread warp there is no bank count to show; pick a warp width of 32 to see the shared-memory store.
              </Translated>
            </p>
          )}
        </div>
      </div>
      <SegmentStrip lanes={cw.threads.map((t) => ({ lane: t.lane, byte: t.byte }))} warpWidth={ww} elemBytes={4} seg={32} unit="sector" /* i18n-ignore: unit key, translated where shown */ />
    </Bench>
  );
}

// ---------------------------------------------------------------- Exercise 1 drill

const VERDICTS: Ex1Verdict[] = ['coalesced', 'uncoalesced', 'not applicable']; // i18n-ignore: verdict keys, translated where shown

function Exercise1Drill() {
  const ex = exercise1();
  const [pick, setPick] = useState<Record<string, Ex1Verdict | undefined>>({});
  const [checked, setChecked] = useState(false);
  const right = ex.filter((x) => pick[x.id] === x.verdict).length;
  const h = strideCase(4);
  return (
    <Bench
      title={trText('Exercise 1 drill', 'Luyện tập bài tập 1')}
      task={trText('Classify all nine accesses, then check.', 'Phân loại cả chín truy cập, rồi kiểm tra.')}
      foot={trText(
        "Sector counts are for a 32-lane warp whose lane 0 has i = 0, in NVIDIA's 32-byte unit, with blockDim.x·gridDim.x = 1024. Your answers here are not recorded.",
        'Số sector tính cho một warp 32 lane có lane 0 với i = 0, theo đơn vị 32 byte của NVIDIA, với blockDim.x·gridDim.x = 1024. Câu trả lời của bạn ở đây không được ghi lại.',
      )}
    >
      <table className="data" style={{ background: 'var(--page)', margin: 0 }}>
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}>
              <Translated vi="Truy cập">Access</Translated>
            </th>
            <th>
              <Translated vi="Dòng">Line</Translated>
            </th>
            <th>
              <Translated vi="Phán đoán của bạn">Your call</Translated>
            </th>
            {checked ? (
              <th>
                <Translated vi="Mô hình">Model</Translated>
              </th>
            ) : null}
          </tr>
        </thead>
        <tbody>
          {ex.map((x) => (
            <tr key={x.id} data-access={x.id} data-result={checked ? (pick[x.id] === x.verdict ? 'right' : 'wrong') : ''}>
              <td style={{ paddingLeft: '0.6rem' }}>({x.id}) <code>{x.code}</code></td>
              <td>{String(x.line).padStart(2, '0')}</td>
              <td>
                <span className="seg" role="group" aria-label={trText(`Access (${x.id})`, `Truy cập (${x.id})`)}>
                  {VERDICTS.map((v) => (
                    <button key={v} aria-pressed={pick[x.id] === v} onClick={() => { setPick({ ...pick, [x.id]: v }); setChecked(false); }}>
                      <Translated vi={{ coalesced: 'được hợp nhất', uncoalesced: 'không được hợp nhất', 'not applicable': 'không áp dụng' }[v]}>{v}</Translated>
                    </button>
                  ))}
                </span>
              </td>
              {checked ? (
                <td>
                  <span className={'chip ' + (pick[x.id] === x.verdict ? 'ok' : 'fail')}>
                    <Translated vi={{ coalesced: 'được hợp nhất', uncoalesced: 'không được hợp nhất', 'not applicable': 'không áp dụng' }[x.verdict]}>{x.verdict}</Translated>
                  </span>{' '}
                  <span className="muted">
                    {x.space === 'shared'
                      ? x.sharedStride === 4
                        ? trText(
                            `shared memory needs no coalescing; section 6.4 will count this one as ${h.passes}-way on ${h.banksUsed} banks`,
                            `bộ nhớ chia sẻ không cần hợp nhất truy cập; mục 6.4 sẽ đếm truy cập này là xung đột ${h.passes} chiều trên ${h.banksUsed} bank`,
                          )
                        : trText('shared memory needs no coalescing', 'bộ nhớ chia sẻ không cần hợp nhất truy cập')
                      : trText(
                          `${x.transactions} sectors, ${pct(x.efficiency!)} used${x.runs === 1 ? '' : `, ${x.runs} runs`}`,
                          `${x.transactions} sector, ${pct(x.efficiency!)} được dùng${x.runs === 1 ? '' : `, ${x.runs} dải`}`,
                        )}
                  </span>
                </td>
              ) : null}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="row" style={{ marginTop: '0.8rem' }}>
        <button className="primary" onClick={() => setChecked(true)} disabled={Object.keys(pick).length < ex.length}>
          <Translated vi="Kiểm tra câu trả lời">Check my answers</Translated>
        </button>
        {checked ? (
          <span aria-live="polite" data-score={right}>
            <Translated vi={<>Đúng {right} trên {ex.length}.</>}>
              <>{right} of {ex.length} right.</>
            </Translated>
          </span>
        ) : (
          <span className="muted">
            <Translated vi={<>Đã trả lời {Object.keys(pick).length} trên {ex.length}</>}>
              <>{Object.keys(pick).length} of {ex.length} answered</>
            </Translated>
          </span>
        )}
      </div>
      {checked ? (
        <Translated
          vi={
            <p style={{ margin: '0.7rem 0 0' }}>
              Quy tắc đứng sau mọi dòng truy cập bộ nhớ toàn cục: chỉ số có phải là cùng một biểu thức cộng <code>i</code> cho
              mọi lane, với hệ số 1 trên <code>i</code> không? Một số hạng không phụ thuộc vào lane (
              <code>j·blockDim.x·gridDim.x</code>, <code>+ 8</code>) dịch cả dải và tốn nhiều nhất một đoạn thêm khi nó phá vỡ
              việc căn lề. Một hệ số trên <code>i</code> (<code>i*4</code>, <code>i*8</code>) kéo các lane xa nhau ra.
            </p>
          }
        >
          <p style={{ margin: '0.7rem 0 0' }}>
            The rule behind every global line: is the index the same expression plus <code>i</code> for every lane, with a
            coefficient of 1 on <code>i</code>? A term that does not depend on the lane (<code>j·blockDim.x·gridDim.x</code>,{' '}
            <code>+ 8</code>) shifts the whole run and costs at most one extra segment when it breaks alignment. A coefficient
            on <code>i</code> (<code>i*4</code>, <code>i*8</code>) spreads the lanes apart.
          </p>
        </Translated>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- your device

const STRIDES = [1, 2, 4, 8, 16, 32, 64];
const SEGS = [32, 64, 128];

function YourDevice61() {
  const { reports, loading } = useReports();
  const found = gpuBackends(reports)
    .map((b) => ({ b, device: String(reports[b].device.name ?? b), rows: labRows(reports, b, LAB61) }))
    .filter((x) => x.rows.length);
  const W = 680, H = 230, L = 50, R = 170, TOP = 14, BOT = 34;
  const xOf = (s: number) => L + (Math.log2(s) / 6) * (W - L - R);
  const yOf = (v: number) => TOP + (1 - Math.min(1.1, v) / 1.1) * (H - TOP - BOT);
  const measured = found.map((f) => {
    const st = f.rows.filter((r) => r.name === 'strided' && num(r.stride) != null).sort((p, q) => Number(p.stride) - Number(q.stride));
    const base = num(st.find((r) => r.stride === 1)?.gb_per_s);
    return {
      ...f,
      points: st.map((r) => ({ s: Number(r.stride), v: num(r.relative_to_stride1) ?? (base ? (num(r.gb_per_s) ?? 0) / base : NaN) })).filter((p) => isFinite(p.v)),
      copy: num(f.rows.find((r) => r.name === 'copy')?.gb_per_s),
      readRows: f.rows.find((r) => r.name === 'transpose' && (r.variant === 'read_rows' || r.kernel === 'read_rows')),
      writeRows: f.rows.find((r) => r.name === 'transpose' && (r.variant === 'write_rows' || r.kernel === 'write_rows')),
    };
  });
  const dash: Record<number, string> = { 32: '6 3', 64: '2 3', 128: '10 3 2 3' };
  return (
    <Bench
      title={trText('Your device: coalescing', 'Thiết bị của bạn: hợp nhất truy cập bộ nhớ')}
      task={trText(
        'Where do your measurements sit against the three segment models?',
        'Các phép đo của bạn nằm ở đâu so với ba mô hình đoạn?',
      )}
      foot={trText(
        "Model curves: fraction of fetched bytes a warp of 32 floats uses at each stride, from the same model as the explorer. Measured: lab 6.1's useful bandwidth relative to stride 1, which also includes the (coalesced) write and whatever the caches did. They are not the same quantity; compare shapes, not values.",
        'Đường mô hình: tỉ lệ số byte đã nạp mà một warp 32 float dùng ở mỗi bước nhảy, từ cùng mô hình với trình khám phá. Đo được: băng thông hữu ích của lab 6.1 so với bước nhảy 1, vốn còn gồm cả phép ghi (được hợp nhất) và bất cứ điều gì bộ nhớ đệm đã làm. Chúng không phải cùng một đại lượng; hãy so hình dạng, không so giá trị.',
      )}
    >
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={trText('Efficiency against stride: models and your measurements', 'Hiệu suất theo bước nhảy: các mô hình và phép đo của bạn')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line x1={L} x2={W - R} y1={yOf(v)} y2={yOf(v)} stroke="var(--rule-soft)" />
            <text x={L - 6} y={yOf(v) + 3} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{v}</text>
          </g>
        ))}
        {STRIDES.map((s) => (
          <text key={s} x={xOf(s)} y={H - BOT + 16} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{s}</text>
        ))}
        <text x={(L + W - R) / 2} y={H - 4} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}><Translated vi="bước nhảy (float)">stride (floats)</Translated></text>
        {SEGS.map((g) => {
          const pts = STRIDES.map((s) => `${xOf(s)},${yOf(strideEfficiency(s, g))}`).join(' ');
          return (
            <g key={g} data-model={g}>
              <polyline points={pts} fill="none" stroke="var(--ink-3)" strokeWidth="1.4" strokeDasharray={dash[g]} />
              <line x1={W - R + 8} x2={W - R + 34} y1={TOP + 10 + SEGS.indexOf(g) * 16} y2={TOP + 10 + SEGS.indexOf(g) * 16} stroke="var(--ink-3)" strokeWidth="1.4" strokeDasharray={dash[g]} />
              <text x={W - R + 38} y={TOP + 14 + SEGS.indexOf(g) * 16} fontSize="10" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>mô hình, {g} B</>}><>model, {g} B</></Translated></text>
            </g>
          );
        })}
        {measured.map((m) => (
          <g key={m.b} data-measured={m.b}>
            <polyline points={m.points.map((p) => `${xOf(p.s)},${yOf(p.v)}`).join(' ')} fill="none" stroke="var(--global)" strokeWidth="2" />
            {m.points.map((p) => <circle key={p.s} cx={xOf(p.s)} cy={yOf(p.v)} r="3.2" fill="var(--global)" />)}
            <line x1={W - R + 8} x2={W - R + 34} y1={TOP + 62 + measured.indexOf(m) * 16} y2={TOP + 62 + measured.indexOf(m) * 16} stroke="var(--global)" strokeWidth="2" />
            <text x={W - R + 38} y={TOP + 66 + measured.indexOf(m) * 16} fontSize="10" style={{ fill: 'var(--global)' }}>{m.device}</text>
          </g>
        ))}
      </svg>
      {loading ? null : !measured.length ? (
        <p data-no-measurement>
          <Translated
            vi={
              <>
                Chưa có phép đo nào. Chạy <code>python3 labs/pmpp.py run 6.1 --solution --bench</code> trên một máy có GPU
                (hoặc <code>run 6.1 --bench</code> khi các kernel của chính bạn đã vượt qua) rồi tải lại: dải quét bước nhảy
                của bạn sẽ hiện cạnh các đường mô hình, và hai phép chuyển vị ngây thơ xuất hiện bên dưới.
              </>
            }
          >
            <>
              No measurement yet. Run <code>python3 labs/pmpp.py run 6.1 --solution --bench</code> on a machine with a GPU
              (or <code>run 6.1 --bench</code> once your own kernels pass) and reload: your stride sweep joins the model
              curves, and the two naive transposes appear below.
            </>
          </Translated>
        </p>
      ) : (
        <table className="data" style={{ background: 'var(--page)', margin: '0.6rem 0 0' }}>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}>
                <Translated vi="Thiết bị">Device</Translated>
              </th>
              <th className="num">
                <Translated vi="sao chép GB/s">copy GB/s</Translated>
              </th>
              <th className="num">
                <Translated vi="chuyển vị, ghi theo bước nhảy">transpose, strided stores</Translated>
              </th>
              <th className="num">
                <Translated vi="chuyển vị, nạp theo bước nhảy">transpose, strided loads</Translated>
              </th>
              <th className="num">
                <Translated vi="thời gian, ghi theo bước nhảy ÷ nạp theo bước nhảy">time, strided stores ÷ strided loads</Translated>
              </th>
            </tr>
          </thead>
          <tbody>
            {measured.map((m) => {
              const tr = num(m.readRows?.seconds);
              const tw = num(m.writeRows?.seconds);
              return (
                <tr key={m.b} data-device={m.b}>
                  <td style={{ paddingLeft: '0.6rem' }}>{m.device} ({m.b})</td>
                  <td className="num">{m.copy != null ? fmt(m.copy) : '–'}</td>
                  <td className="num">{num(m.readRows?.gb_per_s) != null ? `${fmt(num(m.readRows?.gb_per_s)!)} GB/s` : '–'}</td>
                  <td className="num">{num(m.writeRows?.gb_per_s) != null ? `${fmt(num(m.writeRows?.gb_per_s)!)} GB/s` : '–'}</td>
                  <td className="num">{tr && tw ? `${fmt(tr / tw)}×` : '–'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </Bench>
  );
}

function TransposeReveal({ predicted }: { predicted: number }) {
  const { reports, loading } = useReports();
  const found = gpuBackends(reports).flatMap((b) => {
    const rows = labRows(reports, b, LAB61);
    const tr = num(rows.find((r) => r.name === 'transpose' && (r.variant === 'read_rows' || r.kernel === 'read_rows'))?.seconds);
    const tw = num(rows.find((r) => r.name === 'transpose' && (r.variant === 'write_rows' || r.kernel === 'write_rows'))?.seconds);
    return tr && tw ? [{ b, device: String(reports[b].device.name ?? b), ratio: tr / tw }] : [];
  });
  if (loading) return null;
  if (!found.length)
    return (
      <p data-no-measurement>
        <Translated
          vi={
            <>
              Máy này chưa có phép đo nào. Chạy <code>python3 labs/pmpp.py run 6.1 --solution --bench</code> rồi tải lại: ô
              này sẽ so dự đoán của bạn với thời gian của hai phép chuyển vị.
            </>
          }
        >
          <>
            No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 6.1 --solution --bench</code> and reload:
            this box will compare your prediction with the two transposes' times.
          </>
        </Translated>
      </p>
    );
  return (
    <>
      {found.map((f) => (
        <p key={f.b} data-ratio={f.ratio}>
          <Translated
            vi={
              <>
                Trên <b>{f.device}</b> ({f.b}) phép chuyển vị ghi theo bước nhảy mất <b>{fmt(f.ratio)}×</b> thời gian của phép
                chuyển vị nạp theo bước nhảy.{' '}
                {Math.abs(Math.log(predicted / f.ratio)) < Math.log(1.25)
                  ? 'Con số đó gần với dự đoán của bạn.'
                  : predicted > f.ratio
                    ? 'Bạn đã nghĩ phép ghi theo bước nhảy tốn hơn mức nó tốn ở đây.'
                    : 'Bạn đã nghĩ phép ghi theo bước nhảy tốn ít hơn mức nó tốn ở đây.'}
              </>
            }
          >
            <>
              On <b>{f.device}</b> ({f.b}) the transpose with strided stores took <b>{fmt(f.ratio)}×</b> the time of the one with
              strided loads. {Math.abs(Math.log(predicted / f.ratio)) < Math.log(1.25) ? 'That is close to what you predicted.' : predicted > f.ratio ? 'You expected strided stores to cost more than they did here.' : 'You expected strided stores to cost less than they did here.'}
            </>
          </Translated>
        </p>
      ))}
      <p className="muted">
        <Translated vi="Một phép đo từ máy của bạn, không phải giá trị kỳ vọng. Bộ nhớ đệm, việc gộp ghi và loại bộ nhớ đều góp phần, và một thiết bị khác có thể lệch theo cả hai chiều.">
          A measurement from your machine, not an expected value. Caches, write combining and the memory type all play a part, and another device can differ in either direction.
        </Translated>
      </p>
    </>
  );
}

// ---------------------------------------------------------------- questions

export const questionsS61: Record<string, Question> = {
  'ch06/6.1/q-align': {
    kind: 'numeric',
    id: 'ch06/6.1/q-align',
    prompt: (
      <Translated
        vi={
          <>
            Một warp 32 lane đọc 32 float liên tiếp, nhưng float đầu tiên bắt đầu lệch một float sau một biên 32 byte. Phép
            nạp chạm vào bao nhiêu sector 32 byte?
          </>
        }
      >
        <>
          A 32-lane warp reads 32 consecutive floats, but the first one starts one float past a 32-byte boundary. How
          many 32-byte sectors does the load touch?
        </>
      </Translated>
    ),
    answer: 5,
    explain: (
      <Translated
        vi={
          <p>
            Warp cần các byte 4 đến 131 của vùng đã căn lề. Các sector phủ 0–31, 32–63, 64–95, 96–127 và 128–159: bốn byte
            cuối tràn sang sector thứ năm. Hướng dẫn của NVIDIA cho cùng con số với một warp truy cập tuần tự nhưng lệch lề.
            Bộ nhớ từ <code>cudaMalloc</code> bắt đầu ở một biên 256 byte, nên việc lệch lề đến từ chỉ số, không phải từ vùng
            cấp phát.
          </p>
        }
      >
        <p>
          The warp needs bytes 4 to 131 of the aligned region. Sectors cover 0–31, 32–63, 64–95, 96–127 and 128–159: the
          last four bytes spill into a fifth. NVIDIA's guide gives the same count for a sequential but misaligned warp.
          Memory from <code>cudaMalloc</code> starts on a 256-byte boundary, so the misalignment comes from the index, not
          the allocation.
        </p>
      </Translated>
    ),
    source: 'book §6.1 p. 126; notes/today/ch06-facts.md A3',
    vi: { source: 'sách §6.1 p. 126; notes/today/ch06-facts.md A3' },
  },
  'ch06/6.1/q-stride2': {
    kind: 'numeric',
    id: 'ch06/6.1/q-stride2',
    prompt: (
      <Translated
        vi={
          <>
            Lane <code>i</code> của một warp 32 lane đọc <code>in[2*i]</code> (float, điểm bắt đầu đã căn lề). Warp dùng bao
            nhiêu phần trăm số byte đã nạp, tính theo sector 32 byte của NVIDIA?
          </>
        }
      >
        <>
          Lane <code>i</code> of a 32-lane warp reads <code>in[2*i]</code> (floats, aligned start). What percentage of the
          bytes fetched does the warp use, in NVIDIA's 32-byte sectors?
        </>
      </Translated>
    ),
    answer: 50,
    unit: '%',
    explain: (
      <Translated
        vi={
          <p>
            32 lane trải trên 256 byte, tám sector, và dùng cách một float trong mỗi sector: 128 trên 256 byte. Hướng dẫn của
            NVIDIA nêu 50 % cho bước nhảy 2. Đẩy bước nhảy lên 8 float hoặc hơn thì mỗi lane có một sector riêng: dùng 4 byte
            trên 32, 12.5 %, trường hợp tệ nhất.
          </p>
        }
      >
        <p>
          The 32 lanes span 256 bytes, eight sectors, and use every other float in each: 128 of 256 bytes. NVIDIA's guide
          states 50 % for stride 2. Push the stride to 8 floats or more and each lane has a sector to itself: 4 bytes used
          of 32, 12.5 %, the worst case.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch06-facts.md A2', // i18n-ignore: file citation
  },
  'ch06/6.1/q-mread': {
    kind: 'choice',
    id: 'ch06/6.1/q-mread',
    prompt: (
      <Translated
        vi={
          <>
            Trong kernel matmul cơ bản với các khối rộng ít nhất 32 luồng, mọi lane của một warp đọc cùng một{' '}
            <code>M[row*Width + k]</code>. Đó có phải là vấn đề hợp nhất truy cập bộ nhớ không?
          </>
        }
      >
        <>
          In the basic matmul kernel with blocks at least 32 threads wide, every lane of a warp reads the same{' '}
          <code>M[row*Width + k]</code>. Is that a coalescing problem?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Có: 32 lane phát 32 yêu cầu, mỗi lane một, dù chỉ ở một địa chỉ.">
            Yes: 32 lanes issue 32 requests, one per lane, even at a single address.
          </Translated>
        ),
        why: (
          <Translated vi="Các lane dùng chung một địa chỉ. Phần cứng đếm số đoạn bị chạm, không đếm số lane.">
            The lanes share one address. The hardware counts the segments touched, not the lanes.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không: một sector phục vụ cả warp, vốn chỉ cần 4 byte.">
            No: one sector serves the whole warp, which needs only 4 bytes.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Có: các địa chỉ không liên tiếp, nên phần cứng không gộp chúng được.">
            Yes: the addresses are not consecutive, so the hardware cannot merge them.
          </Translated>
        ),
        why: (
          <Translated vi="Chúng giống hệt nhau, đó là trường hợp rẻ nhất: một đoạn.">
            They are identical, which is the cheapest case of all: one segment.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Sách không phân tích lần đọc M trong mục này, vì nó không phải vấn đề. Với các khối rộng 16 luồng, một warp phủ
            hai hàng luồng, nên nó đọc hai phần tử khác nhau của M: hai địa chỉ, hai sector. Đặt độ rộng khối của trình khám
            phá là 16 để thấy điều đó.
          </p>
        }
      >
        <p>
          The book does not analyse the M read in this section, because it is not a problem. With blocks 16 threads wide a
          warp covers two rows of threads, so it reads two different elements of M: two addresses, two sectors. Set the
          explorer's block width to 16 to see it.
        </p>
      </Translated>
    ),
    source: 'book §6.1 pp. 127–128, Figs. 6.2–6.3; our analysis with NVIDIA\'s rule (facts A2)',
    vi: { source: 'sách §6.1 pp. 127–128, Figs. 6.2–6.3; phân tích của chúng tôi với quy tắc của NVIDIA (facts A2)' },
  },
  'ch06/6.1/q-linear': {
    kind: 'choice',
    id: 'ch06/6.1/q-linear',
    prompt: (
      <Translated
        vi={
          <>
            <code>W</code> được lưu như cách <code>nn.Linear</code> lưu nó, hình dạng (out, in), thứ tự hàng. Mỗi luồng sở
            hữu một đặc trưng đầu ra <code>o</code> của một hàng batch và, trong vòng lặp theo <code>k</code>, đọc{' '}
            <code>W[o*in + k]</code>. Các lane kề nhau có <code>o</code> kề nhau. Lần đọc có được hợp nhất không?
          </>
        }
      >
        <>
          <code>W</code> is stored as <code>nn.Linear</code> stores it, shape (out, in), row-major. Each thread owns one
          output feature <code>o</code> of one batch row and, in its loop over <code>k</code>, reads{' '}
          <code>W[o*in + k]</code>. Neighbouring lanes have neighbouring <code>o</code>. Is the read coalesced?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Có: mỗi luồng đọc các địa chỉ liên tiếp khi k tăng.">
            Yes: each thread reads consecutive addresses as k advances.
          </Translated>
        ),
        why: (
          <Translated vi="Hợp nhất truy cập nói về các lane của một lệnh, không phải về một luồng theo thời gian. Ở một k cố định các lane cách xa nhau.">
            Coalescing is about the lanes of one instruction, not about one thread over time. At a fixed k the lanes are far apart.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ khi in là bội của 32, để mọi hàng của W bắt đầu ở một biên sector.">
            Only when in is a multiple of 32, so that every row of W starts on a sector boundary.
          </Translated>
        ),
        why: (
          <Translated vi="Một bội của 32 giữ các hàng được căn lề, nhưng các lane vẫn cách nhau in phần tử.">
            A multiple of 32 keeps rows aligned, but the lanes are still in elements apart.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không: ở mọi k các lane cách nhau in phần tử. Đây là Hình 6.3.">
            No: at every k the lanes are in elements apart. This is Fig. 6.3.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một luồng đi dọc theo hàng của nó là điều một lập trình viên CPU nhìn vào, và ở đây đó là góc nhìn sai. Hai cách
            sửa: lưu trọng số đã chuyển vị một lần khi mô hình được nạp (một thay đổi bố cục), hoặc xoay góc ô của{' '}
            <code>W</code> khi nó được nạp, đó là lab 6.4.
          </p>
        }
      >
        <p>
          One thread walking along its row is what a CPU programmer looks at, and it is the wrong view here. Two fixes: store
          the transposed weight once when the model loads (a layout change), or corner-turn the tile of <code>W</code> as
          it is loaded, which is lab 6.4.
        </p>
      </Translated>
    ),
    source: 'book §6.1 pp. 128–129; notes/today/ch03-facts.md D6 for nn.Linear\'s layout',
    vi: { source: 'sách §6.1 pp. 128–129; notes/today/ch03-facts.md D6 cho bố cục của nn.Linear' },
  },
  'ch06/6.1/q-cornerturn': {
    kind: 'numeric',
    id: 'ch06/6.1/q-cornerturn',
    prompt: (
      <Translated
        vi={
          <>
            Bản minh họa của Hình 6.4: N có kích thước 16 × 16 và được lưu theo từng cột (phần tử (r, c) ở{' '}
            <code>c*16 + r</code>), <code>T = 4</code>, khối <code>bx = 1</code>, pha <code>ph = 2</code>. Với xoay góc, luồng{' '}
            <code>tx = 3, ty = 1</code> nạp chỉ số phẳng nào?
          </>
        }
      >
        <>
          The toy of Fig. 6.4: N is 16 × 16 and stored column by column (element (r, c) at <code>c*16 + r</code>),{' '}
          <code>T = 4</code>, block <code>bx = 1</code>, phase <code>ph = 2</code>. With corner turning, which flat index
          does thread <code>tx = 3, ty = 1</code> load?
        </>
      </Translated>
    ),
    answer: 91,
    explain: (
      <Translated
        vi={
          <p>
            Khi đã xoay, luồng lấy hàng ph·T + tx = 2·4 + 3 = 11 và cột bx·T + ty = 1·4 + 1 = 5: chỉ số 5·16 + 11 = 91. Luồng
            láng giềng tx = 2 nạp 90, cách một phần tử. Theo ánh xạ của chương 5, cùng hai luồng đó nạp hàng ph·T + ty = 9 và
            các cột 7 và 6: chỉ số 121 và 105, cách nhau mười sáu.
          </p>
        }
      >
        <p>
          Turned, the thread takes row ph·T + tx = 2·4 + 3 = 11 and column bx·T + ty = 1·4 + 1 = 5: index 5·16 + 11 = 91.
          Its neighbour tx = 2 loads 90, one element away. Under chapter 5's mapping the same two threads load rows
          ph·T + ty = 9 and columns 7 and 6: indices 121 and 105, sixteen apart.
        </p>
      </Translated>
    ),
    source: 'book §6.1 pp. 129–130, Fig. 6.4',
    vi: { source: 'sách §6.1 pp. 129–130, Fig. 6.4' },
  },
  'ch06/6.1/q-ex3': {
    kind: 'choice',
    id: 'ch06/6.1/q-ex3',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 3: trong matmul phân ô của chương 5, những độ rộng ô vuông nào tránh được truy cập không hợp nhất vào bộ
            nhớ toàn cục?
          </>
        }
      >
        <>
          Exercise 3: in chapter 5's tiled matmul, which square tile widths avoid uncoalesced accesses to global memory?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Độ rộng nào cũng được: mọi hàng ô đều được đọc liên tiếp.">Any width: every tile row is read consecutively.</Translated>,
        why: (
          <Translated vi="Một ô rộng 4 float lấp đầy một nửa mỗi sector nó chạm vào. Đặt T = 4 với N theo từng hàng trong trình chạy từng bước.">
            A tile 4 floats wide fills half of each sector it touches. Set T = 4 with N row by row in the stepper.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Còn tùy cách hiểu. Nếu phép nạp của mỗi warp phải là một dải liên tiếp, chỉ 32. Nếu điều quan trọng là số byte đã nạp bị lãng phí trong các sector 32 byte của NVIDIA, thì mọi bội của 8 float với các hàng đã căn lề đều không lãng phí gì, kể cả 8 và 16.">
            It depends on the reading. If each warp's load must be one consecutive run, only 32. If what counts is fetched bytes wasted in NVIDIA's 32-byte sectors, any multiple of 8 floats with aligned rows wastes nothing, 8 and 16 included.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chỉ 16: nửa warp mỗi hàng ô, nên mỗi warp đọc hai hàng trọn vẹn.">
            Only 16: half a warp per tile row, so each warp reads two whole rows.
          </Translated>
        ),
        why: (
          <Translated vi="Hai hàng là hai dải, nên cách hiểu chặt chẽ đòi 32; theo cách hiểu sector thì 8 không lãng phí nhiều byte hơn 16. Không có gì làm 16 nổi bật.">
            Two rows are two runs, so the strict reading wants 32; under the sector reading 8 wastes no more bytes than 16. Nothing singles out 16.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Chính nhận xét của sách rằng các ô xoay góc thực tế rộng bằng một warp (p. 130) ủng hộ cách hiểu thứ nhất; với
              giới hạn 1024 luồng, 32 cũng là ô vuông lớn nhất. Đơn vị được công bố của NVIDIA ủng hộ cách hiểu thứ hai (facts
              A2): một hàng 16 float là hai sector trọn vẹn. Mở rộng điều đó cho 8 là phép tính của chúng tôi từ quy tắc 32
              byte.
            </p>
            <p>Trong trình chạy từng bước, lưu N theo từng hàng, giữ ánh xạ của chương 5 và thử T = 8, 16 và 32.</p>
          </>
        }
      >
        <>
          <p>
            The book's own remark that real corner-turning tiles are as wide as a warp (p. 130) supports the first
            reading; with the 1024-thread limit, 32 is also the largest square tile. NVIDIA's documented unit supports the
            second (facts A2): a 16-float row is two whole sectors. Extending that to 8 is our arithmetic from the 32-byte
            rule.
          </p>
          <p>In the stepper, store N row by row, keep chapter 5's mapping and try T = 8, 16 and 32.</p>
        </>
      </Translated>
    ),
    source: 'book p. 155 (exercise 3), p. 130; notes/today/ch06-facts.md A2',
    vi: { source: 'sách p. 155 (bài tập 3), p. 130; notes/today/ch06-facts.md A2' },
  },
  'ch06/6.1/q-store': {
    kind: 'choice',
    id: 'ch06/6.1/q-store',
    prompt: (
      <Translated
        vi={
          <>
            Phép chuyển vị ngây thơ của bạn đọc các hàng của đầu vào và ghi các cột của đầu ra, và nó chạy thấp hơn nhiều so
            với băng thông của kernel sao chép. Trước khi viết lại, điều gì cho bạn biết liệu các phép ghi theo bước nhảy có
            phải là thứ tốn kém nhất?
          </>
        }
      >
        <>
          Your naive transpose reads rows of the input and writes columns of the output, and it runs far below your copy
          kernel&rsquo;s bandwidth. Before you rewrite it, what tells you whether the strided stores are what costs the
          most?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Đo thời gian bản đối xứng, đọc cột và ghi hàng: cùng số byte với bước nhảy chuyển sang phép nạp.">
            Time the mirror image, which reads columns and writes rows: the same bytes with the stride moved to the loads.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không cần gì: sách nói phép ghi không hợp nhất gây hại hơn phép nạp, nên hãy sửa phép ghi trước.">
            Nothing: the book says uncoalesced stores hurt more than loads, so fix the stores first.
          </Translated>
        ),
        why: (
          <Translated vi="Sách nói chúng có thể gây hại hơn (p. 130). NVIDIA gắn chi phí thêm với ECC trên bộ nhớ GDDR và nói HBM2 trả ít hơn nhiều, nên câu trả lời phụ thuộc vào thiết bị.">
            The book says they can hurt more (p. 130). NVIDIA ties the extra cost to ECC on GDDR memory and says HBM2 pays much less, so the answer depends on the device.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Đếm số lệnh ghi so với số lệnh nạp: bên nào có nhiều lệnh hơn thì tốn hơn.">
            Count store instructions against load instructions: the side with more of them costs more.
          </Translated>
        ),
        why: (
          <Translated vi="Cả hai kernel phát một lệnh nạp và một lệnh ghi mỗi phần tử. Điều khác nhau là mỗi phép ghi toàn warp chạm bao nhiêu sector, không phải bao nhiêu lệnh được chạy.">
            Both kernels issue one load and one store per element. What differs is how many sectors each warp-wide store touches, not how many instructions run.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chạy nó trên trình giả lập và đọc xem báo cáo đánh dấu những phép ghi theo bước nhảy nào.">
            Run it on the emulator and read which strided stores its report flags.
          </Translated>
        ),
        why: (
          <Translated vi="Các trình giả lập kiểm tra tính đúng đắn. Chúng không mô hình hóa thời gian, bộ nhớ đệm hay hợp nhất truy cập, nên chi phí cần một thiết bị thật.">
            The emulators check correctness. They model no time, caches or coalescing, so cost needs a real device.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lab 6.1 chạy đúng cặp đó. Những gì các nhà cung cấp công bố không đồng đều. Với bộ nhớ GDDR bật ECC, NVIDIA viết:
            &ldquo;Scattered accesses increase ECC memory transfer overhead, especially when writing data to global
            memory.&rdquo; (truy cập rải rác làm tăng chi phí truyền ECC, nhất là khi ghi vào bộ nhớ toàn cục). Họ nói ECC
            của HBM2 tốn ít hơn nhiều, và Nsight Compute đánh dấu một mẫu ghi &ldquo;significantly exceeding expected sector
            count&rdquo; (vượt đáng kể số sector kỳ vọng). Thao tác đọc-sửa-ghi mà sách nêu tên không được mô tả trong trang
            nào chúng tôi đã đọc. Dù bên nào tốn hơn trên thiết bị của bạn, xoay góc qua bộ nhớ chia sẻ cho phép cả phép nạp
            lẫn phép ghi chạy dọc theo hàng.
          </p>
        }
      >
        <p>
          Lab 6.1 runs exactly that pair. What the vendors document is uneven. For GDDR memory with ECC on, NVIDIA says:
          &ldquo;Scattered accesses increase ECC memory transfer overhead, especially when writing data to global
          memory.&rdquo; It says HBM2&rsquo;s ECC costs much less, and Nsight Compute flags a store pattern
          &ldquo;significantly exceeding expected sector count&rdquo;. The read-modify-write the book names is described
          in none of the pages we read. Whichever side costs more on your device, corner turning through shared memory
          lets both the loads and the stores run along rows.
        </p>
      </Translated>
    ),
    source: 'book §6.1 p. 130; notes/today/ch06-facts.md A5; labs/README.md for what the emulators model',
    vi: { source: 'sách §6.1 p. 130; notes/today/ch06-facts.md A5; labs/README.md về những gì trình giả lập mô hình hóa' },
  },
  'ch06/6.1/q-ex1c': {
    kind: 'choice',
    id: 'ch06/6.1/q-ex1c',
    prompt: (
      <Translated vi={<>Bài tập 1(c): <code>b[j*blockDim.x*gridDim.x + i]</code> ở dòng 07, với một <code>j</code> cố định. Có được hợp nhất không?</>}>
        <>Exercise 1(c): <code>b[j*blockDim.x*gridDim.x + i]</code> on line 07, for a fixed <code>j</code>. Coalesced?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không được hợp nhất, vì phép nhân với blockDim.x·gridDim.x.">
            Uncoalesced, because of the multiplication by blockDim.x·gridDim.x.
          </Translated>
        ),
        why: (
          <Translated vi="Tích đó không liên quan đến lane. Chỉ một hệ số trên i mới kéo các lane xa nhau ra.">
            The product does not involve the lane. Only a coefficient on i spreads the lanes apart.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Được hợp nhất: số hạng j như nhau với mọi lane.">Coalesced: the j term is the same for every lane.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Hợp nhất truy cập không áp dụng.">Coalescing does not apply.</Translated>,
        why: <Translated vi="b là một con trỏ vào bộ nhớ toàn cục.">b is a pointer into global memory.</Translated>,
      },
    ],
    explain: (
      <Translated vi={<p>Với mỗi j, warp đọc 32 float liên tiếp bắt đầu từ j·blockDim.x·gridDim.x. Theo đơn vị của NVIDIA: bốn sector, đều được dùng hết.</p>}>
        <p>For each j the warp reads 32 consecutive floats starting at j·blockDim.x·gridDim.x. In NVIDIA's unit: four sectors, all used.</p>
      </Translated>
    ),
    source: 'book p. 155 (exercise 1); notes/ch06.md',
    vi: { source: 'sách p. 155 (bài tập 1); notes/ch06.md' },
  },
  'ch06/6.1/q-ex1d': {
    kind: 'choice',
    id: 'ch06/6.1/q-ex1d',
    prompt: (
      <Translated vi={<>Bài tập 1(d): <code>c[i*4 + j]</code> ở dòng 07. Có được hợp nhất không?</>}>
        <>Exercise 1(d): <code>c[i*4 + j]</code> on line 07. Coalesced?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Không được hợp nhất: các lane kề nhau cách nhau 4 float.">Uncoalesced: neighbouring lanes are 4 floats apart.</Translated>,
        correct: true,
      },
      {
        label: (
          <Translated vi="Được hợp nhất, vì j như nhau với mọi lane và vòng lặp đọc mọi byte.">
            Coalesced, because j is the same for every lane and the loop reads every byte.
          </Translated>
        ),
        why: (
          <Translated vi="j thì đúng vậy, nhưng i được nhân với 4. Qua cả vòng lặp mọi byte đều được đọc, nhưng hợp nhất truy cập được xét theo từng lệnh một.">
            j is, but i is multiplied by 4. Over the loop every byte is read, but coalescing is judged one instruction at a time.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Hợp nhất truy cập không áp dụng cho một chỉ số chứa biến vòng lặp.">
            Coalescing does not apply to an index that contains a loop variable.
          </Translated>
        ),
        why: (
          <Translated vi="c nằm trong bộ nhớ toàn cục; điều quyết định là chỉ số thay đổi thế nào từ lane này sang lane khác.">
            c is in global memory; what decides is how the index changes from lane to lane.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một lệnh: 16 sector cho 128 byte hữu ích. Qua bốn giá trị của j, vòng lặp đúng là đọc mọi byte nó đã nạp, đó là lý
            do một bộ nhớ đệm có thể cứu nó; mô hình đếm lại mỗi lệnh từ đầu. Toàn bộ bài luyện: a được hợp nhất, b không áp
            dụng, c được hợp nhất, d không được hợp nhất, e, f và h không áp dụng, g được hợp nhất, i không được hợp nhất.
          </p>
        }
      >
        <p>
          One instruction: 16 sectors for 128 useful bytes. Over the four values of j the loop does read every byte it
          fetched, which is why a cache might rescue it; the model counts each instruction afresh. The full drill:
          a coalesced, b not applicable, c coalesced, d uncoalesced, e, f and h not applicable, g coalesced, i uncoalesced.
        </p>
      </Translated>
    ),
    source: 'book p. 155 (exercise 1); notes/ch06.md',
    vi: { source: 'sách p. 155 (bài tập 1); notes/ch06.md' },
  },
  'ch06/6.1/q-ex1g': {
    kind: 'choice',
    id: 'ch06/6.1/q-ex1g',
    prompt: (
      <Translated vi={<>Bài tập 1(g): <code>d[i + 8]</code> ở dòng 10. Có được hợp nhất không?</>}>
        <>Exercise 1(g): <code>d[i + 8]</code> on line 10. Coalesced?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không được hợp nhất, vì độ lệch 8 làm lệch lề địa chỉ của mọi lane.">
            Uncoalesced, because the offset of 8 misaligns every lane's address.
          </Translated>
        ),
        why: (
          <Translated vi="Mọi lane có cùng độ lệch, nên dải vẫn liên tiếp; tệ nhất nó vắt qua thêm một đoạn.">
            Every lane gets the same offset, so the run stays consecutive; at worst it straddles one more segment.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Hợp nhất truy cập không áp dụng: dòng 10 là phép ghi, không phải phép nạp.">
            Coalescing does not apply: line 10 is a store, not a load.
          </Translated>
        ),
        why: (
          <Translated vi="d nằm trong bộ nhớ toàn cục, và phép ghi được hợp nhất theo cùng quy tắc (p. 130).">
            d is in global memory, and stores coalesce by the same rule (p. 130).
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Được hợp nhất; độ lệch có thể đẩy dải vắt qua thêm một đoạn.">
            Coalesced; the offset can shift the run across one more segment.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Tám float đúng bằng một sector 32 byte, nên theo đơn vị của NVIDIA không có gì thêm bị nạp: bốn sector. Với dòng
            128 byte, cùng dải đó vắt qua hai dòng thay vì một. Hãy thử với mô hình đoạn của trình khám phá.
          </p>
        }
      >
        <p>
          Eight floats are exactly one 32-byte sector, so in NVIDIA's unit nothing extra is fetched: four sectors. In
          128-byte lines the same run straddles two lines instead of one. Try it with the explorer's segment model.
        </p>
      </Translated>
    ),
    source: 'book p. 155 (exercise 1), p. 126 on alignment; facts A2, A3',
    vi: { source: 'sách p. 155 (bài tập 1), p. 126 về căn lề; facts A2, A3' },
  },
};

// ---------------------------------------------------------------- the section

export function S61() {
  const nCol = exprAccess('n-col', { Width: 1024, k: 0, row: 0, stride: 1, offset: 0, warpWidth: 32, lane0: 0 }, 32);
  const nRow = exprAccess('n-row', { Width: 1024, k: 0, row: 0, stride: 1, offset: 0, warpWidth: 32, lane0: 0 }, 32);
  const amd = exprAccess('n-row', { Width: 1024, k: 0, row: 0, stride: 1, offset: 0, warpWidth: 64, lane0: 0 }, 128);
  return (
    <>
      <Translated
        vi={
          <p>
            Chương 5 cắt giảm lượng dữ liệu một kernel nạp từ bộ nhớ toàn cục (global memory). Mục này nói về việc mỗi lần
            nạp diễn ra hiệu quả đến đâu. Hai điều đó đi cùng nhau: phân ô (tiling) quyết định bao nhiêu byte đi qua bus bộ
            nhớ, hợp nhất truy cập bộ nhớ (coalescing) quyết định chúng cần bao nhiêu chuyến.
          </p>
        }
      >
        <p>
          Chapter 5 cut how much data a kernel fetches from global memory. This section is about how efficiently each
          fetch happens. The two work together: tiling decides how many bytes cross the memory bus, coalescing decides
          how many trips they take.
        </p>
      </Translated>
      <Source section="6.1" pages="124" />

      <h2>
        <Translated vi="Vì sao DRAM muốn các phần tử kề nhau">Why DRAM wants neighbours</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bộ nhớ toàn cục là DRAM. Mỗi bit là một tụ điện nhỏ xíu. Đọc nó nghĩa là để lượng điện tích nhỏ đó kéo một dây
            dài, đường bit (bit line), dùng chung với khoảng một nghìn ô khác, đủ xa để một bộ khuếch đại cảm biến quyết định
            điện tích có ở đó hay không.
          </p>
        }
      >
        <p>
          Global memory is DRAM. Each bit is a tiny capacitor. Reading it means letting that small charge pull a long wire,
          the bit line, shared by about a thousand other cells, far enough for a sense amplifier to decide whether the
          charge was there.
        </p>
      </Translated>
      <Source section="6.1" pages="124–125" figs={trText('sidebar “Why are DRAMs so slow?”', 'khung bên “Why are DRAMs so slow?”')} />
      <Translated
        vi={
          <p>
            Sách đặt con số này ở hàng chục nano giây, so với chu kỳ xung nhịp chưa tới một nano giây. Tụ điện lớn hơn sẽ
            nhanh hơn, nhưng mỗi thế hệ lại làm chúng nhỏ đi để nhét thêm bit vào một chip, nên, theo lập luận của sách, độ
            trễ đã không giảm xuống.
          </p>
        }
      >
        <p>
          The book puts this at tens of nanoseconds, against clock cycles well under one nanosecond. Bigger capacitors
          would be faster, but every generation makes them smaller to fit more bits on a chip, so, the book argues, the
          latency has not come down.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md, A1)" /* i18n-ignore: date and facts-file reference */>
        <Translated
          vi={
            <p>
              Không trang nào của nhà cung cấp mà chúng tôi đã đọc đưa ra độ trễ DRAM tính bằng nano giây hay một xu hướng.
              Hướng dẫn của NVIDIA đặt truy cập bộ nhớ toàn cục không qua cache ở mức “hundreds of clock cycles” (hàng trăm
              chu kỳ xung nhịp). Xu hướng là nhận định của sách.
            </p>
          }
        >
          <p>
            No vendor page we read gives a DRAM latency in nanoseconds or a trend. NVIDIA's guide puts uncached global
            accesses at “hundreds of clock cycles”. The trend is the book's claim.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Vì vậy DRAM được xây dựng để song song thay vào đó. Đọc một vị trí sẽ cảm biến cả một dải các vị trí láng giềng
            cùng lúc, nhiều bộ khuếch đại làm việc cạnh nhau, rồi dải đó được chuyển nhanh tới bộ xử lý. Dải đó là một{' '}
            <b>burst</b>. Một chương trình dùng trọn các burst khai thác bộ nhớ hiệu quả hơn nhiều so với một chương trình lấy
            vài byte từ mỗi nơi trong rất nhiều nơi rải rác.
          </p>
        }
      >
        <p>
          So DRAM is built for parallelism instead. Reading one location senses a whole run of its neighbours at once, many
          amplifiers working side by side, and the run then goes out to the processor quickly. That run is a{' '}
          <b>burst</b>. A program that uses whole bursts gets far more out of the memory than one that takes a few bytes from
          each of many scattered places.
        </p>
      </Translated>
      <Source pages="125" />

      <h2>
        <Translated vi="Bộ nhớ đệm di chuyển trọn từng dòng">Caches move whole lines</Translated>
      </h2>
      <Translated
        vi={
          <p>
            GPU cũng có bộ nhớ đệm (cache) L1 và L2, làm từ SRAM. Sách để chúng ở hậu cảnh vì chúng không phải phần nổi bật
            của mô hình lập trình CUDA, nhưng chúng định hình những gì bạn đo được.
          </p>
        }
      >
        <p>
          GPUs also have L1 and L2 caches, built from SRAM. The book has kept them in the background because they are not
          a prominent part of the CUDA programming model, but they shape what you measure.
        </p>
      </Translated>
      <Source section="6.1" pages="125–126" />
      <Translated
        vi={
          <p>
            Dữ liệu di chuyển giữa chúng theo các <b>dòng cache</b> (cache line) kích thước cố định, nên ngay cả khi dữ liệu
            đã nằm trong cache, một warp dùng mọi byte của các dòng nó chạm vào sẽ di chuyển ít dòng hơn một warp không làm
            vậy.
          </p>
        }
      >
        <p>
          Data moves between them in fixed-size <b>cache lines</b>, so even after data is cached, a warp that uses every
          byte of the lines it touches moves fewer lines than one that does not.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md, A4, B1, C1)" /* i18n-ignore: date and facts-file reference */>
        <Translated
          vi={
            <p>
              NVIDIA: dòng L1 và L2 dài 128 byte, gồm bốn sector 32 byte, và trên compute capability 6.0 trở về sau đơn vị
              truy cập là sector 32 byte bất kể một lần nạp có được cache ở L1 hay không. AMD: dòng 64 và/hoặc 128 byte. Apple
              không công bố kích thước dòng hay sector.
            </p>
          }
        >
          <p>
            NVIDIA: L1 and L2 lines are 128 bytes, made of four 32-byte sectors, and on compute capability 6.0 and later
            the unit of access is the 32-byte sector whether or not a load is cached in L1. AMD: 64- and/or 128-byte lines. Apple documents no line or sector
            size.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Quy tắc">The rule</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mọi lane của một warp thực thi cùng một lệnh nạp cùng lúc, mỗi lane với địa chỉ riêng. Phần cứng xem xét các địa
            chỉ cùng nhau. Nếu chúng rơi vào các vị trí liên tiếp trong cùng một burst hay dòng cache, nó gộp chúng thành một
            truy cập hợp nhất, và lệnh nạp được gọi là <b>được hợp nhất</b> (coalesced). Lane 0 đọc vị trí X, lane 1 đọc X + 1,
            lane 2 đọc X + 2 là trường hợp tốt nhất. Căn lề là điều kiện thứ hai: khi dải bắt đầu ở đầu một burst hay dòng, nó
            cần số giao dịch tối thiểu; khi không, cùng những byte đó vắt qua thêm một đơn vị.
          </p>
        }
      >
        <p>
          All lanes of a warp execute the same load at the same time, each with its own address. The hardware looks at the
          addresses together. If they fall on consecutive locations within the same burst or cache line, it merges them
          into one consolidated access, and the load is <b>coalesced</b>. Lane 0 reading location X, lane 1 reading X + 1,
          lane 2 reading X + 2 is the best case. Alignment is a second condition: when the run starts at the beginning of a
          burst or line, it needs the minimum number of transactions; when it does not, the same bytes straddle one more.
        </p>
      </Translated>
      <Source section="6.1" pages="126" />
      <Translated
        vi={
          <p>
            Sách nói về burst và dòng mà không đưa ra kích thước. Trình khám phá bên dưới đếm theo đơn vị được công bố của
            từng nhà cung cấp và cho biết nó đang dùng đơn vị nào.
          </p>
        }
      >
        <p>
          The book speaks of bursts and lines without giving a size. The explorer below counts in each vendor's documented
          unit and says which one it is using.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md, A2, A3)" /* i18n-ignore: date and facts-file reference */>
        <Translated
          vi={
            <p>
              NVIDIA phát biểu quy tắc theo giao dịch 32 byte. 32 float liên tiếp: 4. Cùng dải đó lệch khỏi biên 32 byte: 5.
              Bước nhảy 2: dùng một nửa mỗi byte đã nạp. Các lane cách nhau từ 32 byte trở lên: mỗi lane một giao dịch, 12.5 %.
              Thứ tự lane liên tiếp “merely the typical way” (chỉ là cách thường gặp): mọi hoán vị trong cùng các sector đó
              đều tốn như nhau.
            </p>
          }
        >
          <p>
            NVIDIA states the rule in 32-byte transactions. 32 consecutive floats: 4. The same run off a 32-byte boundary:
            5. Stride 2: half of each fetched byte used. Lanes 32 bytes or more apart: one transaction each, 12.5 %.
            Consecutive lane order is “merely the typical way”: any permutation within the same sectors costs the same.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Đếm số giao dịch">Count the transactions</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Quay lại phép nhân ma trận cơ bản của chương 3. Trong vòng lặp k mỗi luồng đọc <code>M[row*Width + k]</code> và{' '}
            <code>N[k*Width + col]</code>. Trong một warp, <code>row</code>, <code>k</code> và <code>Width</code> như nhau với
            mọi lane còn <code>col</code> tăng một từ lane này sang lane kế, nên các địa chỉ của N liên tiếp nhau: ở mỗi k,
            warp đọc một đoạn của một hàng của N. Sách nói thêm rằng mọi kernel cho tới giờ đều đã hợp nhất theo cách này mà
            không cần cố gắng.
          </p>
        }
      >
        <p>
          Back to the basic matrix multiplication of chapter 3. In the k loop each thread reads{' '}
          <code>M[row*Width + k]</code> and <code>N[k*Width + col]</code>. Within a warp, <code>row</code>,{' '}
          <code>k</code> and <code>Width</code> are the same for every lane and <code>col</code> goes up by one from lane
          to lane, so the N addresses are consecutive: at each k the warp reads a piece of one row of N. The book adds that
          every kernel so far has coalesced this way without trying.
        </p>
      </Translated>
      <Source section="6.1" pages="127–128" figs={trText('Fig. 6.2', 'Hình 6.2')} />
      <Translated
        vi={
          <p>
            Giờ giả sử N được lưu theo từng cột, nên lần đọc trở thành <code>N[col*Width + k]</code>. Các lane kề nhau cách
            nhau Width phần tử, hàng trăm hoặc hàng nghìn với ma trận thật.
          </p>
        }
      >
        <p>
          Now suppose N is stored column by column, so the read becomes <code>N[col*Width + k]</code>. Neighbouring lanes
          are Width elements apart, hundreds or thousands for real matrices.
        </p>
      </Translated>
      <Source pages="128–129" figs={trText('Fig. 6.3', 'Hình 6.3')} />
      <Predict
        id="ch06/6.1/p-sectors"
        prompt={
          <Translated
            vi={
              <>
                Width = 1024, N lưu theo từng cột, một warp 32 lane thực thi <code>N[col*Width + k]</code> với một k. Nó chạm
                vào bao nhiêu sector 32 byte?
              </>
            }
          >
            <>
              Width = 1024, N stored column by column, a 32-lane warp executes <code>N[col*Width + k]</code> for one k. How
              many 32-byte sectors does it touch?
            </>
          </Translated>
        }
        unit={trText('sectors', 'sector')}
      >
        {(v) => (
          <Translated
            vi={
              <p>
                Mô hình cho ra <b>{nCol.transactions}</b>: các lane cách nhau {bytes(nCol.lanes[1].byte - nCol.lanes[0].byte)}{' '}
                byte, nên mỗi lane cần một sector riêng, và chỉ 4 byte trong 32 được dùng. Lần đọc thứ tự hàng chạm vào{' '}
                {nRow.transactions}. {v === nCol.transactions ? 'Bạn đã đúng.' : 'Hãy kiểm tra với thiết lập sẵn thứ tự cột bên dưới.'}
              </p>
            }
          >
            <p>
              The model says <b>{nCol.transactions}</b>: the lanes are {bytes(nCol.lanes[1].byte - nCol.lanes[0].byte)} bytes
              apart, so each needs a sector of its own, and 4 bytes of its 32 are used. The row-major read touches{' '}
              {nRow.transactions}. {v === nCol.transactions ? 'You had it.' : 'Check it with the column-major preset below.'}
            </p>
          </Translated>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Dùng các thiết lập sẵn trước, rồi mỗi lần thay đổi một điều khiển. Tăng k từng bước và quan sát các tổng dưới phần
            kết quả: mô hình đếm lại mỗi lệnh từ đầu.
          </p>
        }
      >
        <p>
          Use the presets first, then change one control at a time. Step k and watch the totals under the readout: the
          model counts every instruction afresh.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một warp, một lệnh: số đoạn đã căn lề khác nhau mà các lane của nó chạm vào. Không có bộ nhớ đệm giữa các lệnh,
              không đo thời gian. Kích thước đoạn là đơn vị được công bố của từng nhà cung cấp; với Metal, nơi không có đơn vị
              nào được công bố, kích thước là một lựa chọn do bạn đưa ra.
            </p>
          }
        >
          <p>
            One warp, one instruction: the distinct aligned segments its lanes touch. No caches between instructions, no
            timing. Segment sizes are each vendor's documented unit; for Metal, where none is documented, the size is a
            choice you make.
          </p>
        </Translated>
      </Simplification>
      <CoalescingExplorer />
      <Ask q={questionsS61['ch06/6.1/q-align']} label={trText('Work it out', 'Tự tính')} />
      <Ask q={questionsS61['ch06/6.1/q-stride2']} label={trText('Work it out', 'Tự tính')} />
      <Ask q={questionsS61['ch06/6.1/q-mread']} />

      <h2>
        <Translated vi="Quan sát một lệnh">Watch one instruction</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Cùng một warp ở cùng một k, trong cả hai bố cục, rồi qua cả vòng lặp k. Điều mà một bức hình tĩnh làm phẳng đi là
            việc này xảy ra với một lệnh, lặp đi lặp lại.
          </p>
        }
      >
        <p>
          The same warp at the same k, in both layouts, and then over the k loop. What a still picture flattens is that this
          happens to one instruction, again and again.
        </p>
      </Translated>
      <CoalesceScene />

      <h2>
        <Translated vi="Các lần đọc thứ tự cột đến từ đâu">Where column-major reads come from</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trong mã C, một lần đọc thứ tự cột (column-major) thường đến từ một phép chuyển vị. Bạn cần một ma trận và chuyển
            vị của nó, lưu cả hai thì phí bộ nhớ, nên bạn lưu một cái và đọc cái kia bằng cách hoán đổi chỉ số hàng và cột,
            tức là đọc cái đã lưu như thể nó theo thứ tự cột.
          </p>
        }
      >
        <p>
          In C code, a column-major read usually comes from a transpose. You need a matrix and its transpose, storing both
          wastes memory, so you store one and read the other by swapping the row and column indices, which is the same as
          reading the stored one as column-major.
        </p>
      </Translated>
      <Source pages="128" />
      <Translated
        vi={
          <p>
            Bạn gặp điều này hằng ngày. <code>nn.Linear</code> lưu trọng số dạng (out_features, in_features) và tính{' '}
            <code>y = x·Wᵀ + b</code>. Viết một kernel giao cho mỗi luồng một đặc trưng đầu ra <code>o</code> của một hàng
            batch, và vòng lặp k của nó đọc <code>W[o*in + k]</code>: dọc theo một hàng của W, trông ổn với một luồng. Nhưng
            các lane của một warp có <code>o</code> kề nhau, nên ở mọi k chúng cách nhau <code>in</code> phần tử.
          </p>
        }
      >
        <p>
          You meet this every day. <code>nn.Linear</code> stores its weight as (out_features, in_features) and computes{' '}
          <code>y = x·Wᵀ + b</code>. Write a kernel that gives each thread one output feature <code>o</code> of one batch
          row, and its k loop reads <code>W[o*in + k]</code>: along a row of W, which looks fine for one thread. But the
          lanes of a warp have neighbouring <code>o</code>, so at every k they are <code>in</code> elements apart.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Bố cục trọng số của <code>nn.Linear</code> lấy từ tài liệu và mã nguồn PyTorch (notes/today/ch03-facts.md D6).
              Kernel một-luồng-mỗi-đặc-trưng là của chúng tôi, để minh họa.
            </p>
          }
        >
          <p>
            The layout of <code>nn.Linear</code>'s weight comes from PyTorch's documentation and source
            (notes/today/ch03-facts.md D6). The one-thread-per-feature kernel is ours, for illustration.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS61['ch06/6.1/q-linear']} />

      <h2>
        <Translated vi="Ba lối thoát">Three ways out</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Khi truy cập tự nhiên không được hợp nhất, sách đưa ra ba cách khắc phục. Thay đổi luồng nào xử lý dữ liệu nào.
            Thay đổi bố cục dữ liệu, chẳng hạn lưu bản chuyển vị một lần. Hoặc đưa dữ liệu vào bộ nhớ chia sẻ (shared memory)
            theo thứ tự được hợp nhất và thực hiện truy cập khó khăn ở đó, nơi thứ tự tốn ít chi phí. Mục 6.8 quay lại hai
            cách đầu; cách thứ ba là xoay góc (corner turning).
          </p>
        }
      >
        <p>
          When the natural access does not coalesce, the book offers three remedies. Change which thread handles which
          data. Change the layout of the data, for example by storing the transposed copy once. Or bring the data into
          shared memory in a coalesced order and do the awkward access there, where order costs little. Section 6.8 returns
          to the first two; the third is corner turning.
        </p>
      </Translated>
      <Source pages="129" />

      <h2>
        <Translated vi="Xoay góc">Corner turning</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lấy kernel phân ô của chương 5 với M thứ tự hàng và N thứ tự cột. Ô M được nạp như trước. Với ô N, ánh xạ của
            chương 5 để một hàng luồng đọc một hàng của ô, và lưu trữ thứ tự cột làm rải rác một hàng. Hoán đổi vai trò của{' '}
            <code>threadIdx.x</code> và <code>threadIdx.y</code> trong phép nạp N thì các luồng liên tiếp sẽ đọc dọc mép trái
            của ô thay vào đó: một cột, thứ mà lưu trữ thứ tự cột giữ liền mạch.
          </p>
        }
      >
        <p>
          Take the tiled kernel of chapter 5 with M row-major and N column-major. The M tile loads as before. For the N
          tile, chapter 5's mapping has a row of threads read a row of the tile, and column-major storage scatters a row.
          Swap the roles of <code>threadIdx.x</code> and <code>threadIdx.y</code> in the N load and consecutive threads
          read down the tile's left edge instead: a column, which column-major storage keeps contiguous.
        </p>
      </Translated>
      <Source section="6.1" pages="129–130" figs={trText('Fig. 6.4', 'Hình 6.4')} />
      <Translated
        vi={
          <p>
            Hình vẽ dùng bốn luồng và các ô 4 × 4 cho gọn. Trong kernel thực tế, theo sách, ô rộng bằng một warp, nên 32 luồng
            đầu tiên của một khối đọc 32 phần tử liên tiếp dọc theo mép trái.
          </p>
        }
      >
        <p>
          The figure uses four threads and 4 × 4 tiles to stay small. In real kernels, the book says, the tile is as wide
          as a warp, so the first 32 threads of a block read 32 consecutive elements down the left edge.
        </p>
      </Translated>
      <Source pages="130" />
      <Translated
        vi={
          <p>
            Khi ô đã nằm trong bộ nhớ chia sẻ, nó có thể được đọc theo thứ tự bất kỳ với rất ít thiệt hại: bộ nhớ chia sẻ là
            SRAM và không có quy tắc hợp nhất truy cập. Nó có một quy tắc riêng, mà trình chạy từng bước gợi ý. Trong cách sắp
            xếp (b), mỗi luồng ghi hàng riêng của nó trong ô chia sẻ, nên warp ghi một cột, và{' '}
            <a href={href('ch06', '6.4')}>mục 6.4</a> nói về cái giá của điều đó.
          </p>
        }
      >
        <p>
          Once the tile is in shared memory it can be read in any order with little penalty: shared memory is SRAM and has
          no coalescing rule. It has a rule of its own, which the stepper hints at. In arrangement (b) each thread writes
          its own row of the shared tile, so the warp writes a column, and{' '}
          <a href={href('ch06', '6.4')}>section 6.4</a> is about what that costs.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md, A6)" /* i18n-ignore: date and facts-file reference */>
        <Translated
          vi={
            <p>
              Best Practices Guide của NVIDIA dạy cùng hai bước đó trong ví dụ C = AAᵀ: một warp đọc một hàng của A vào một cột
              của một ô chia sẻ, rồi ô được chèn đệm. Tài liệu đó cũng nói rằng, ngoài xung đột bank, bộ nhớ chia sẻ không bị
              thiệt hại gì khi truy cập không tuần tự hoặc không căn lề.
            </p>
          }
        >
          <p>
            NVIDIA's Best Practices Guide teaches the same two steps in its C = AAᵀ example: a warp reads a row of A into a
            column of a shared tile, then the tile is padded. It also says that, apart from bank conflicts, shared memory
            has no penalty for non-sequential or unaligned access.
          </p>
        </Translated>
      </Today>
      <CornerTurnStepper />
      <Ask q={questionsS61['ch06/6.1/q-cornerturn']} label={trText('Work it out', 'Tự tính')} />
      <Ask q={questionsS61['ch06/6.1/q-ex3']} />

      <h2>
        <Translated vi="Phép ghi cũng tính">Stores count too</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hợp nhất truy cập cũng áp dụng cho phép ghi, và sách nói phép ghi không hợp nhất có thể gây hại hơn phép nạp không
            hợp nhất: chúng trở thành các lần ghi một phần cần một thao tác đọc-sửa-ghi, tốn thêm băng thông trên các GPU bật
            bộ nhớ sửa lỗi (ECC).
          </p>
        }
      >
        <p>
          Coalescing applies to stores as well, and the book says uncoalesced stores can hurt more than uncoalesced loads:
          they become partial writes that need a read-modify-write, which takes more bandwidth on GPUs with
          error-correcting memory (ECC) enabled.
        </p>
      </Translated>
      <Source pages="130" />
      <Today checked="2026-11-20 (notes/today/ch06-facts.md, A5)" /* i18n-ignore: date and facts-file reference */>
        <Translated
          vi={
            <p>
              NVIDIA: với ECC trên bộ nhớ GDDR, truy cập rải rác, nhất là phép ghi, thêm chi phí; ECC của HBM2 tốn ít hơn
              nhiều. Không nguồn nào chúng tôi đọc mô tả thao tác đọc-sửa-ghi, nên hãy coi nó là unverified.
            </p>
          }
        >
          <p>
            NVIDIA: with ECC on GDDR memory, scattered accesses, writes especially, add overhead; HBM2&rsquo;s ECC costs
            far less. No source we read describes the read-modify-write, so treat it as unverified.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Lab 6.1 có hai phép chuyển vị ngây thơ chỉ khác nhau ở bên nào theo bước nhảy: một bản đọc hàng và ghi cột, bản
            kia đọc cột và ghi hàng. Hãy chốt một con số trước khi chạy.
          </p>
        }
      >
        <p>
          Lab 6.1 has two naive transposes that differ only in which side is strided: one reads rows and writes columns,
          the other reads columns and writes rows. Commit to a number before you run it.
        </p>
      </Translated>
      <Predict
        id="ch06/6.1/p-transpose"
        prompt={
          <Translated
            vi={
              <>
                Trên thiết bị của bạn, thời gian của phép chuyển vị ghi theo bước nhảy chia cho thời gian của phép chuyển vị
                nạp theo bước nhảy. Lớn hơn 1 nghĩa là phép ghi theo bước nhảy tốn hơn.
              </>
            }
          >
            <>
              On your device, the time of the transpose with strided stores divided by the time of the one with strided
              loads. Above 1 means strided stores cost more.
            </>
          </Translated>
        }
        unit="×"
      >
        {(v) => <TransposeReveal predicted={v} />}
      </Predict>
      <Ask q={questionsS61['ch06/6.1/q-store']} />

      <h2>
        <Translated vi="Một warp là một nhóm đi chung xe tốt">A warp is a good carpool</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lợi ích là ít lưu lượng hơn: nhiều truy cập được gộp thành một yêu cầu, điều đòi hỏi chúng cần dữ liệu cùng lúc
            và ở các địa chỉ kề nhau.
          </p>
        }
      >
        <p>
          The benefit is less traffic: several accesses merged into one request, which needs them at the same time and at
          neighbouring addresses.
        </p>
      </Translated>
      <Source pages="130–132" figs={trText('Figs. 6.5–6.6', 'Hình 6.5–6.6')} />
      <Translated
        vi={
          <p>
            Hình ảnh của sách là đi chung xe. Yêu cầu là xe cộ và dữ liệu là hành khách; khi tốc độ phát yêu cầu vượt quá
            khả năng chuyên chở của hệ thống DRAM, đường tắc và các đơn vị số học phải chờ. Đi chung xe cần những người có
            cùng lịch trình, và các lane của một warp là những người đi chung lý tưởng: chúng phát lệnh nạp cùng một thời
            điểm.
          </p>
        }
      >
        <p>
          The book's picture is carpooling. Requests are vehicles and data are passengers; when the request rate passes
          what the DRAM system can carry, the road jams and the arithmetic units wait. Carpooling needs people on the same
          schedule, and the lanes of a warp are ideal carpoolers: they issue the load at the same instant.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Bộ nhớ đệm của CPU cũng được định hình bởi burst của DRAM: một dòng cache CPU ứng với một phần của burst, một
              burst, hoặc vài burst, và mã dùng mọi byte của các dòng nó chạm vào cũng chạy nhanh hơn nhiều ở đó. Các kỹ thuật
              của chương này áp dụng được sang đó. (Sách p. 132.)
            </p>
          }
        >
          <p>
            CPU caches are shaped by DRAM bursts too: a CPU cache line maps to part of a burst, one, or several, and code
            that uses every byte of the lines it touches runs much faster there as well. The techniques of this chapter
            carry over. (Book p. 132.)
          </p>
        </Translated>
      </Aside>

      <h2>
        <Translated vi="Luyện tập: bài tập 1">Drill: exercise 1</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bài tập đầu tiên của sách là một kernel với chín truy cập bộ nhớ. Hãy phân loại từng cái trước khi kiểm tra. Số
            dòng là số dòng của sách.
          </p>
        }
      >
        <p>
          The book's first exercise is a kernel with nine memory accesses. Classify each one before you check. Line
          numbers are the book's.
        </p>
      </Translated>
      <Source section="6.11" pages="154–155" />
      <Code
        tabs={[{ label: 'foo_kernel', code: FOO, tiers: FOO_TIERS, status: 'front-end-checked; not run' }]} // i18n-ignore: kernel name and verification status
        caption={
          <Translated
            vi={
              <>
                Kernel của Bài tập 1 đúng như in, từ <code>labs/ch06/snippets/s61_foo_kernel.cu</code>. Nó là bài luyện tập,
                không phải một kernel hữu ích.
              </>
            }
          >
            <>Exercise 1's kernel as printed, from <code>labs/ch06/snippets/s61_foo_kernel.cu</code>. It is a drill, not a useful kernel.</>
          </Translated>
        }
      />
      <Exercise1Drill />
      <Ask q={questionsS61['ch06/6.1/q-ex1c']} />
      <Ask q={questionsS61['ch06/6.1/q-ex1d']} />
      <Ask q={questionsS61['ch06/6.1/q-ex1g']} />

      <h2>
        <Translated vi="Trên các backend khác">On other backends</Translated>
      </h2>
      <Translated
        vi={
          <p>
            AMD mô tả cùng cơ chế dựa trên burst. Phần cứng hợp nhất truy cập của họ nhóm các địa chỉ của một wavefront thành
            ít yêu cầu dòng cache nhất, với dòng 64 hoặc 128 byte. Wavefront có 64 lane trên CDNA (Instinct) và 32 trên RDNA
            (Radeon), nên trên CDNA 64 float liên tiếp là {amd.bytesNeeded} byte: {amd.transactions} dòng 128 byte, không phải
            một. Đặt độ rộng warp của trình khám phá là 64 để thấy điều đó.
          </p>
        }
      >
        <p>
          AMD documents the same burst-based mechanism. Its coalescing hardware groups a wavefront's addresses into the
          fewest cache-line requests, with 64- or 128-byte lines. Wavefronts are 64 lanes on CDNA (Instinct) and 32 on RDNA
          (Radeon), so on CDNA 64 consecutive floats are {amd.bytesNeeded} bytes: {amd.transactions} lines of 128 bytes, not
          one. Set the explorer's warp width to 64 to see it.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md B1; ch05-facts.md B5)" /* i18n-ignore: date and facts-file reference */>
        <Translated
          vi={
            <p>
              AMD: “the hardware can combine all 64 thread requests into as few as 4-8 cache line requests (depending on data
              size and alignment)” (phần cứng có thể gộp cả 64 yêu cầu của luồng thành chỉ 4-8 yêu cầu dòng cache, tùy kích
              thước dữ liệu và căn lề), và các địa chỉ cách xa nhau có thể cắt băng thông hiệu dụng “by up to 16x or more”
              (tới 16 lần hoặc hơn).
            </p>
          }
        >
          <p>
            AMD: “the hardware can combine all 64 thread requests into as few as 4-8 cache line requests (depending on data
            size and alignment)”, and widely separated addresses can cut effective bandwidth “by up to 16x or more”.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Apple không công bố quy tắc hợp nhất truy cập, kích thước dòng hay kích thước sector nào. Lời khuyên của họ mang
            tính định tính: sắp xếp dữ liệu cho khớp với mẫu truy cập, và định hình threadgroup cho khớp với bố cục. Trên máy
            Mac, lab 6.1 là cách duy nhất để biết. Dù backend nào, đừng viết cứng độ rộng warp vào mã: hãy đọc{' '}
            <code>warpSize</code> trên CUDA và HIP, và <code>threadExecutionWidth</code> của pipeline trên Metal.
          </p>
        }
      >
        <p>
          Apple documents no coalescing rule, no line size and no sector size. Its advice is qualitative: arrange the data
          to fit the access pattern, and shape threadgroups to match the layout. On a Mac, lab 6.1 is the only way to
          know. Whatever the backend, do not write the warp width into your code: read <code>warpSize</code> on CUDA and
          HIP, and the pipeline's <code>threadExecutionWidth</code> on Metal.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md C1)" /* i18n-ignore: date and facts-file reference */>
        <Translated
          vi={
            <p>
              Bài nói WWDC22 của Apple về tính toán: các yêu cầu từ một SIMD group được gói vào các dòng cache, và phần lớn
              một dòng có thể không được dùng; “Re-arrange the data to fit the access pattern better” (sắp xếp lại dữ liệu cho
              khớp mẫu truy cập hơn). Không có con số nào được đưa ra.
            </p>
          }
        >
          <p>
            Apple's WWDC22 talk on compute: requests from a SIMD-group are packed into cache lines, and most of a line can
            go unused; “Re-arrange the data to fit the access pattern better”. No numbers are given.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Thiết bị của bạn">Your device</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lab 6.1 quét bước nhảy của một lần đọc từ 1 đến 64 float và đo thời gian hai phép chuyển vị so với một phép sao
            chép thuần. Trang lab có chi tiết: <a href={href('ch06', 'labs')}>lab</a>.
          </p>
        }
      >
        <p>
          Lab 6.1 sweeps the stride of a read from 1 to 64 floats and times the two transposes against a plain copy. The
          lab page has the details: <a href={href('ch06', 'labs')}>labs</a>.
        </p>
      </Translated>
      <YourDevice61 />
    </>
  );
}
