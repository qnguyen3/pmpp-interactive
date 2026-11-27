import { useMemo, useState, type CSSProperties, type KeyboardEvent, type MouseEvent, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { useProgress } from '../../lib/progress';
import { href } from '../../lib/router';
import {
  BOOK,
  CHANNELS,
  GRAY_WEIGHTS,
  STYLE_LABEL,
  blockBands,
  blockInfo,
  bookLabel,
  elementAt,
  fmtFloat,
  grayDetail,
  indexingCoverage,
  latticeOf,
  layout,
  offsetOf,
  regions,
  stepsGrayscale,
  stridesFor,
  threadAt,
  type Dims,
  type GrayOp,
  type GrayVariant,
  type Indexing,
  type Launch,
  type RegionId,
  type Style,
  type XY,
} from '../../sim/mapping';
import { GridOverPictureScene } from './GridOverPictureScene';

// ---------------------------------------------------------------- code shown in this section
// Copies of labs/ch03/snippets/*. S32.snippets.test.ts fails if a copy drifts from its file.

/** labs/ch03/snippets/grayscale_kernel.cu: the corrected kernel and its launch. */
export const GRAY_CU = `#define CHANNELS 3   // bytes per colour pixel: r, g, b

// One thread per pixel. Pin holds three bytes for each pixel, Pout one.
__global__ void colorToGrayscaleConversion(unsigned char* Pout,
                                           unsigned char* Pin,
                                           int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        // Offset in the flat gray picture: skip the rows above.
        int grayOffset = row * width + col;
        // The colour picture: the same walk, CHANNELS bytes a pixel.
        int rgbOffset = grayOffset * CHANNELS;
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];
        // The cast covers the whole sum. It drops the fraction.
        Pout[grayOffset] =
            (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// n rows (y), m columns (x): x is sized from the width, y from the height.
void launchColorToGrayscale(unsigned char* Pin_d, unsigned char* Pout_d,
                            int m, int n) {
    dim3 dimGrid((m + 15) / 16, (n + 15) / 16, 1);
    dim3 dimBlock(16, 16, 1);
    colorToGrayscaleConversion<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, m, n);
}`;

/** labs/ch03/snippets/grayscale_kernel.metal */
export const GRAY_METAL = `#include <metal_stdlib>
using namespace metal;

#define CHANNELS 3   // bytes per colour pixel: r, g, b

// The same kernel in Metal Shading Language. gid is the thread's position in the
// whole grid: .x is the column, .y the row. Nothing has to be multiplied out.
kernel void color_to_grayscale(device const uchar* Pin  [[buffer(0)]],
                               device uchar*       Pout [[buffer(1)]],
                               constant uint2&     size [[buffer(2)]],   // (width, height)
                               uint2 gid [[thread_position_in_grid]]) {
    uint col = gid.x;
    uint row = gid.y;
    // Needed when the launch is dispatchThreadgroups. With dispatchThreads over
    // (width, height) it is never false.
    if (col < size.x && row < size.y) {
        uint grayOffset = row * size.x + col;
        uint rgbOffset = grayOffset * CHANNELS;
        uchar r = Pin[rgbOffset];
        uchar g = Pin[rgbOffset + 1];
        uchar b = Pin[rgbOffset + 2];
        Pout[grayOffset] = (uchar)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}`;

/** labs/ch03/snippets/grayscale_as_printed.cu, after its marker line: Fig. 3.4's code lines, comments reworded. */
export const PRINTED = `// Input bytes are unsigned chars, 0 to 255
// A pixel is 3 chars in a row, one per channel (RGB)
__global__
void colortoGrayscaleConvertion(unsigned char * Pout,
                unsigned char * Pin, int width, int height) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        // 1D offset into the gray picture
        int grayOffset = row*width + col;
        // The colour picture is CHANNELS times
        // as wide, counted in bytes
        int rgbOffset = grayOffset*CHANNELS;
        unsigned char r = Pin[rgbOffset    ]; // red
        unsigned char g = Pin[rgbOffset + 1]; // green
        unsigned char b = Pin[rgbOffset + 2]; // blue
        // Weight the channels, add them up
        // and store one byte
        Pout[grayOffset] = (unsigned char) 0.21f*r+0.71f*g+0.07f*b;
    }
}`;

/** The launch on book p. 50, as printed (a short quote; it is half of one of the defects). */
export const PRINTED_HOST = [
  'dim3 dimGrid(ceil(m/16.0), ceil(n/16.0), 1);', // i18n-ignore: printed code line
  'dim3 dimBlock(16, 16, 1);', // i18n-ignore: printed code line
  'colorToGrayscaleConversion<<<dimGrid,dimBlock>>>(Pin_d, Pout_d, m, n);',
];

/** Which line of GRAY_CU each step of the model's stepper is on. */
export const LINE: Record<GrayOp, number> = {
  col: 7, row: 8, guard: 9, grayOffset: 11, rgbOffset: 13, r: 14, g: 15, b: 16, sum: 19, cast: 19, store: 18, done: 21,
};

const TIERS: Record<string, Tier> = {
  Pin: 'global', Pout: 'global', Pin_d: 'global', Pout_d: 'global',
  col: 'reg', row: 'reg', grayOffset: 'reg', rgbOffset: 'reg', r: 'reg', g: 'reg', b: 'reg', gid: 'reg',
};

// ---------------------------------------------------------------- small pieces

const mono: CSSProperties = { margin: '0.18rem 0', overflowWrap: 'anywhere' }; // i18n-ignore: CSS value
const Reg = ({ children }: { children: ReactNode }) => <b className="t-reg">{children}</b>;
const Glob = ({ children }: { children: ReactNode }) => <b className="t-global">{children}</b>;
/** Inline code that may wrap, for narrow panels. */
const Cw = ({ children }: { children: ReactNode }) => (
  <code style={{ whiteSpace: 'normal', overflowWrap: 'anywhere' }}>{children}</code>
);
const onKey = (fn: () => void) => (e: KeyboardEvent) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fn();
  }
};

function NumField({ label, value, min, max, onChange }: {
  label: string; value: number; min: number; max: number; onChange: (v: number) => void;
}) {
  const [text, setText] = useState(String(value));
  const [last, setLast] = useState(value);
  if (last !== value) {
    setLast(value);
    setText(String(value));
  }
  return (
    <label className="field">
      {label}
      <input type="number" min={min} max={max} step="1" value={text} style={{ width: '5.2rem' }}
        onChange={(e) => {
          setText(e.target.value);
          const v = Math.round(Number(e.target.value));
          if (e.target.value.trim() !== '' && v >= min && v <= max) {
            setLast(v);
            onChange(v);
          }
        }} />
    </label>
  );
}

// ---------------------------------------------------------------- the thread map

const PRESETS: { id: string; label: string; height: number; width: number; blockW: number; blockH: number }[] = [
  { id: 'book', get label() { return tr("the book's 62 × 76, blocks 16 × 16", 'ảnh 62 × 76 của sách, khối 16 × 16'); }, height: 62, width: 76, blockW: 16, blockH: 16 },
  { id: 'small', get label() { return tr('small: 10 × 13, blocks 4 × 4', 'nhỏ: 10 × 13, khối 4 × 4'); }, height: 10, width: 13, blockW: 4, blockH: 4 },
  { id: 'ex3', get label() { return tr('exercise 3: 150 × 300, blocks 16 wide by 32 high', 'bài tập 3: 150 × 300, khối rộng 16, cao 32'); }, height: 150, width: 300, blockW: 16, blockH: 32 },
  { id: 'large', get label() { return tr('1500 × 2000, blocks 16 × 16', '1500 × 2000, khối 16 × 16'); }, height: 1500, width: 2000, blockW: 16, blockH: 16 },
];
const BLOCK_SHAPES: [number, number][] = [[4, 4], [8, 8], [16, 16], [32, 32], [16, 32]];
const STYLES: Style[] = ['blocks', 'threadgroups', 'threads'];
/** Past this many cells along a side, a thread is too small to click and the map draws blocks instead. */
const THREAD_LEVEL_MAX = 128;

const NAMES: Record<Style, { block: string; thread: string; dim: string; unit: string; Unit: string }> = {
  blocks: { block: 'blockIdx', thread: 'threadIdx', dim: 'blockDim', get unit() { return tr('block', 'khối'); }, get Unit() { return tr('Block', 'Khối'); } },
  threadgroups: {
    block: 'threadgroup_position_in_grid', thread: 'thread_position_in_threadgroup', dim: 'threads_per_threadgroup',
    get unit() { return tr('threadgroup', 'threadgroup'); }, get Unit() { return tr('Threadgroup', 'Threadgroup'); },
  },
  threads: {
    block: 'threadgroup_position_in_grid', thread: 'thread_position_in_threadgroup', dim: 'threads_per_threadgroup',
    get unit() { return tr('threadgroup', 'threadgroup'); }, get Unit() { return tr('Threadgroup', 'Threadgroup'); },
  },
};

const clampSel = (l: Launch, p: XY): XY => {
  const L = layout(l);
  return { x: Math.max(0, Math.min(p.x, L.threadsX - 1)), y: Math.max(0, Math.min(p.y, L.threadsY - 1)) };
};

interface Win {
  x0: number;
  y0: number;
  cols: number;
  rows: number;
}

/** A window onto the lattice of threads: one cell per thread, filled where the thread has a pixel. */
function Lattice({ id, launch, win, cell, sel, onPick, hot, hover, onHover, label }: {
  id: string; launch: Launch; win: Win; cell: number; sel: XY; onPick: (p: XY) => void;
  hot?: RegionId | null; hover?: XY | null; onHover?: (b: XY | null) => void; label: string;
}) {
  const L = layout(launch);
  const ML = 54, MT = 24, MR = 8, MB = 8;
  const W = win.cols * cell, H = win.rows * cell;
  const VBW = ML + W + MR, VBH = MT + H + MB;
  const gap = cell >= 6 ? 1.5 : 1;
  const inWin = sel.x >= win.x0 && sel.x < win.x0 + win.cols && sel.y >= win.y0 && sel.y < win.y0 + win.rows;
  const at = (e: MouseEvent<SVGSVGElement>): XY | null => {
    const r = e.currentTarget.getBoundingClientRect();
    const k = VBW / r.width;
    const x = win.x0 + Math.floor(((e.clientX - r.left) * k - ML) / cell);
    const y = win.y0 + Math.floor(((e.clientY - r.top) * k - MT) / cell);
    const ok = x >= win.x0 && y >= win.y0 && x < Math.min(L.threadsX, win.x0 + win.cols) && y < Math.min(L.threadsY, win.y0 + win.rows);
    return ok ? { x, y } : null;
  };
  const xs: number[] = [];
  for (let x = 0; x < L.threadsX; x += launch.blockW) xs.push(x);
  xs.push(L.threadsX);
  const ys: number[] = [];
  for (let y = 0; y < L.threadsY; y += launch.blockH) ys.push(y);
  ys.push(L.threadsY);
  const hotBlocks: XY[] = [];
  if (hot && L.blocks <= 4096)
    for (let by = 0; by < L.gridY; by++)
      for (let bx = 0; bx < L.gridX; bx++) if (blockInfo(launch, bx, by).region === hot) hotBlocks.push({ x: bx, y: by });
  const hb = hover ? blockInfo(launch, hover.x, hover.y) : null;
  const sx = ML + (sel.x - win.x0) * cell, sy = MT + (sel.y - win.y0) * cell;
  const key = (e: KeyboardEvent<SVGSVGElement>) => {
    const d: Record<string, [number, number]> = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    const m = d[e.key];
    if (!m) return;
    e.preventDefault();
    const sx2 = e.shiftKey ? launch.blockW : 1, sy2 = e.shiftKey ? launch.blockH : 1;
    onPick({
      x: Math.max(win.x0, Math.min(Math.min(L.threadsX, win.x0 + win.cols) - 1, sel.x + m[0] * sx2)),
      y: Math.max(win.y0, Math.min(Math.min(L.threadsY, win.y0 + win.rows) - 1, sel.y + m[1] * sy2)),
    });
  };
  return (
    <svg viewBox={`0 0 ${VBW} ${VBH}`} role="application" aria-label={label} tabIndex={0}
      data-lattice={id} data-ml={ML} data-mt={MT} data-cell={cell} data-x0={win.x0} data-y0={win.y0} data-vbw={VBW}
      style={{ width: VBW, maxWidth: '100%', height: 'auto', display: 'block', cursor: 'pointer' }}
      onClick={(e) => { const p = at(e); if (p) onPick(p); }}
      onMouseMove={(e) => {
        if (!onHover) return;
        const p = at(e);
        const b = p ? { x: Math.floor(p.x / launch.blockW), y: Math.floor(p.y / launch.blockH) } : null;
        if (b?.x !== hover?.x || b?.y !== hover?.y) onHover(b);
      }}
      onMouseLeave={() => onHover?.(null)}
      onKeyDown={key}>
      <defs>
        <pattern id={`${id}-f`} width={cell} height={cell} patternUnits="userSpaceOnUse">
          <rect width={cell - gap} height={cell - gap} fill="var(--ink-3)" />
        </pattern>
        <pattern id={`${id}-h`} width={cell} height={cell} patternUnits="userSpaceOnUse">
          <rect x="0.5" y="0.5" width={cell - gap - 1} height={cell - gap - 1} fill="none" stroke="var(--ink-3)" strokeWidth="1" />
        </pattern>
      </defs>
      <svg x={ML} y={MT} width={W} height={H} viewBox={`${win.x0 * cell} ${win.y0 * cell} ${W} ${H}`}>
        <rect x={win.x0 * cell} y={win.y0 * cell} width={W} height={H} fill="var(--page)" />
        {/* every thread, hollow; then the ones that have a pixel, filled */}
        <rect width={L.threadsX * cell} height={L.threadsY * cell} fill={`url(#${id}-h)`} />
        <rect width={L.coveredCols * cell} height={L.coveredRows * cell} fill="var(--page)" />
        <rect width={L.coveredCols * cell} height={L.coveredRows * cell} fill={`url(#${id}-f)`} data-area="working" />
        {/* pixels no thread reaches */}
        {L.unreached > 0 ? (
          <g data-area="unreached">
            {L.coveredCols < launch.width ? (
              <rect x={L.coveredCols * cell} y={0} width={(launch.width - L.coveredCols) * cell} height={launch.height * cell}
                fill="var(--fail-wash)" stroke="var(--fail)" strokeDasharray="4 3" />
            ) : null}
            {L.coveredRows < launch.height ? (
              <rect x={0} y={L.coveredRows * cell} width={L.coveredCols * cell} height={(launch.height - L.coveredRows) * cell}
                fill="var(--fail-wash)" stroke="var(--fail)" strokeDasharray="4 3" />
            ) : null}
          </g>
        ) : null}
        {hotBlocks.map((b) => {
          const i = blockInfo(launch, b.x, b.y);
          return <rect key={`${b.x},${b.y}`} x={i.x0 * cell} y={i.y0 * cell} width={i.w * cell} height={i.h * cell} fill="var(--ink)" fillOpacity="0.2" />;
        })}
        {/* block borders */}
        <g stroke="var(--ink)" strokeWidth={cell >= 6 ? 1.4 : 1}>
          {xs.map((x) => <line key={'x' + x} x1={x * cell} x2={x * cell} y1={0} y2={L.threadsY * cell} />)}
          {ys.map((y) => <line key={'y' + y} y1={y * cell} y2={y * cell} x1={0} x2={L.threadsX * cell} />)}
        </g>
        {hb ? <rect x={hb.x0 * cell} y={hb.y0 * cell} width={hb.w * cell} height={hb.h * cell} fill="none" stroke="var(--ink)" strokeWidth="3" /> : null}
      </svg>
      {inWin ? (
        <g data-sel={`${sel.x},${sel.y}`}>
          <path d={`M${ML - 4} ${sy + cell / 2} H${sx} M${sx + cell / 2} ${MT - 4} V${sy}`} stroke="var(--reg)" strokeWidth="1" fill="none" />
          <rect x={sx} y={sy} width={cell - gap} height={cell - gap} fill="var(--reg)" />
          <rect x={sx - 3} y={sy - 3} width={cell - gap + 6} height={cell - gap + 6} fill="none" stroke="var(--reg)" strokeWidth="2" />
          <text x={ML - 7} y={sy + cell / 2 + 4} fontSize="11" textAnchor="end" style={{ fill: 'var(--reg)' }}><Translated vi={<>hàng {sel.y}</>}>row {sel.y}</Translated></text>
          <text x={Math.max(ML + 14, Math.min(ML + W - 14, sx + cell / 2))} y={MT - 8} fontSize="11" textAnchor="middle" style={{ fill: 'var(--reg)' }}><Translated vi={<>cột {sel.x}</>}>col {sel.x}</Translated></text>
        </g>
      ) : null}
    </svg>
  );
}

/** The same launch drawn one cell per block, for pictures too large to draw thread by thread. */
function BlockLattice({ id, launch, cell, selBlock, onPick, hot, hover, onHover, label }: {
  id: string; launch: Launch; cell: number; selBlock: XY; onPick: (b: XY) => void;
  hot?: RegionId | null; hover: XY | null; onHover: (b: XY | null) => void; label: string;
}) {
  const L = layout(launch);
  const { xs, ys } = blockBands(launch);
  const extX = Math.max(L.gridX, Math.ceil(launch.width / launch.blockW));
  const extY = Math.max(L.gridY, Math.ceil(launch.height / launch.blockH));
  const M = 6;
  const VBW = extX * cell + 2 * M, VBH = extY * cell + 2 * M;
  const gap = cell >= 8 ? 2 : 1;
  const at = (e: MouseEvent<SVGSVGElement>): XY | null => {
    const r = e.currentTarget.getBoundingClientRect();
    const k = VBW / r.width;
    const x = Math.floor(((e.clientX - r.left) * k - M) / cell);
    const y = Math.floor(((e.clientY - r.top) * k - M) / cell);
    return x >= 0 && y >= 0 && x < L.gridX && y < L.gridY ? { x, y } : null;
  };
  const key = (e: KeyboardEvent<SVGSVGElement>) => {
    const d: Record<string, [number, number]> = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    const m = d[e.key];
    if (!m) return;
    e.preventDefault();
    onPick({ x: Math.max(0, Math.min(L.gridX - 1, selBlock.x + m[0])), y: Math.max(0, Math.min(L.gridY - 1, selBlock.y + m[1])) });
  };
  const fx = cell / launch.blockW, fy = cell / launch.blockH; // pixels of drawing per thread
  return (
    <svg viewBox={`0 0 ${VBW} ${VBH}`} role="application" aria-label={label} tabIndex={0}
      data-lattice={id} data-ml={M} data-mt={M} data-cell={cell} data-x0={0} data-y0={0} data-vbw={VBW}
      style={{ width: VBW, maxWidth: '100%', height: 'auto', display: 'block', cursor: 'pointer' }}
      onClick={(e) => { const p = at(e); if (p) onPick(p); }}
      onMouseMove={(e) => { const p = at(e); if (p?.x !== hover?.x || p?.y !== hover?.y) onHover(p); }}
      onMouseLeave={() => onHover(null)}
      onKeyDown={key}>
      <defs>
        <pattern id={`${id}-f`} x={M} y={M} width={cell} height={cell} patternUnits="userSpaceOnUse">
          <rect width={cell - gap} height={cell - gap} fill="var(--ink-2)" />
        </pattern>
        <pattern id={`${id}-h`} x={M} y={M} width={cell} height={cell} patternUnits="userSpaceOnUse">
          <rect x="0.5" y="0.5" width={cell - gap - 1} height={cell - gap - 1} fill="none" stroke="var(--ink-3)" strokeWidth="1" />
        </pattern>
      </defs>
      <rect x={M} y={M} width={extX * cell} height={extY * cell} fill="var(--page)" />
      {xs.flatMap((bx) => ys.map((by) => {
        const b = blockInfo(launch, bx.first, by.first);
        const idle = b.working === 0;
        return (
          <rect key={`${bx.first},${by.first}`} data-band={b.region}
            x={M + bx.first * cell} y={M + by.first * cell} width={bx.count * cell} height={by.count * cell}
            fill={`url(#${id}-${idle ? 'h' : 'f'})`}
            opacity={idle ? 1 : hot && hot !== b.region ? 0.25 : 0.35 + 0.65 * (b.working / (launch.blockW * launch.blockH))} />
        );
      }))}
      {L.unreached > 0 ? (
        <g data-area="unreached">
          {L.coveredCols < launch.width ? (
            <rect x={M + L.coveredCols * fx} y={M} width={(launch.width - L.coveredCols) * fx} height={launch.height * fy}
              fill="var(--fail-wash)" stroke="var(--fail)" strokeDasharray="4 3" />
          ) : null}
          {L.coveredRows < launch.height ? (
            <rect x={M} y={M + L.coveredRows * fy} width={L.coveredCols * fx} height={(launch.height - L.coveredRows) * fy}
              fill="var(--fail-wash)" stroke="var(--fail)" strokeDasharray="4 3" />
          ) : null}
        </g>
      ) : null}
      {hover ? <rect x={M + hover.x * cell - 1} y={M + hover.y * cell - 1} width={cell + 1} height={cell + 1} fill="none" stroke="var(--ink)" strokeWidth="2" /> : null}
      <rect data-sel-block={`${selBlock.x},${selBlock.y}`} x={M + selBlock.x * cell - 2} y={M + selBlock.y * cell - 2} width={cell + 3} height={cell + 3}
        fill="none" stroke="var(--reg)" strokeWidth="2.5" />
    </svg>
  );
}

/** A few slots of a flat array around the ones a thread touches. */
function Strip({ name, size, hot, labels, around = 2 }: { name: ReactNode; size: number; hot: number[]; labels?: string[]; around?: number }) {
  const lo = Math.max(0, hot[0] - around);
  const hi = Math.min(size - 1, hot[hot.length - 1] + around);
  const idx = Array.from({ length: hi - lo + 1 }, (_, i) => lo + i);
  return (
    <div style={{ marginTop: '0.6rem' }}>
      <div className="mx-label t-global">{name}</div>
      <div className="mx global" style={{ gridAutoFlow: 'column', justifyContent: 'start' }}>
        {lo > 0 ? <span className="void">…</span> : null}
        {idx.map((i) => {
          const k = hot.indexOf(i);
          return (
            <span key={i} className={k >= 0 ? 'hot' : ''} style={{ padding: '0 0.3rem' }}>
              {i}
              {k >= 0 && labels ? <sub style={{ color: 'inherit' }}>{labels[k]}</sub> : null}
            </span>
          );
        })}
        {hi < size - 1 ? <span className="void">…</span> : null}
      </div>
    </div>
  );
}

const REGION_NAME: Record<RegionId, string> = {
  get '1'() { return tr('inside', 'bên trong'); },
  get '2'() { return tr('right edge', 'mép phải'); },
  get '3'() { return tr('bottom edge', 'mép dưới'); },
  get '4'() { return tr('corner', 'góc'); },
  get idle() { return tr('outside', 'bên ngoài'); },
};

function ThreadMap({ launch, setLaunch, sel, setSel }: {
  launch: Launch; setLaunch: (l: Launch) => void; sel: XY; setSel: (p: XY) => void;
}) {
  const [hover, setHover] = useState<XY | null>(null);
  const [hot, setHot] = useState<RegionId | null>(null);
  const L = useMemo(() => layout(launch), [launch]);
  const regs = useMemo(() => regions(launch), [launch]);
  const t = threadAt(launch, sel.x, sel.y);
  const nm = NAMES[launch.style];
  const metal = launch.style !== 'blocks';
  const extX = Math.max(L.threadsX, launch.width), extY = Math.max(L.threadsY, launch.height);
  const threadLevel = extX <= THREAD_LEVEL_MAX && extY <= THREAD_LEVEL_MAX;
  const cell = Math.max(4, Math.min(26, Math.floor(520 / extX)));
  const extBX = Math.max(L.gridX, Math.ceil(launch.width / launch.blockW));
  const blockCell = Math.max(4, Math.min(36, Math.floor(540 / extBX)));
  const preset = PRESETS.find((p) => p.height === launch.height && p.width === launch.width && p.blockW === launch.blockW && p.blockH === launch.blockH);
  const change = (patch: Partial<Launch>) => { setHover(null); setHot(null); setLaunch({ ...launch, ...patch }); };
  const shownBlock = hover ?? t.blockIdx;
  const b = blockInfo(launch, shownBlock.x, shownBlock.y);
  const forX = launch.swapped ? launch.height : launch.width;
  const forY = launch.swapped ? launch.width : launch.height;
  const firstSurplus: XY | null =
    L.threadsX > L.coveredCols ? { x: L.coveredCols, y: 0 } : L.threadsY > L.coveredRows ? { x: 0, y: L.coveredRows } : null;
  const bookThread = L.gridY > 1 ? latticeOf(launch, 0, 1, 0, 0) : null;
  const total = regs.reduce((s, r) => s + r.total, 0);

  return (
    <Bench
      title={tr('The thread map', 'Bản đồ luồng')}
      task={
        <Translated vi="Bấm vào một luồng, hoặc đưa tiêu điểm vào bản đồ và dùng các phím mũi tên (Shift di chuyển mỗi lần một khối).">
          Click a thread, or focus the map and use the arrow keys (Shift moves a block at a time).
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Ô đặc là luồng có điểm ảnh, ô rỗng là luồng không có. Không qua phép kiểm tra không phải là lỗi, nên ô
              rỗng không tô đỏ; màu đỏ chỉ đánh dấu những điểm ảnh không luồng nào tới được. <TierLegend tiers={['reg', 'global']} />
            </>
          }
        >
          <>
            Filled squares are threads with a pixel, hollow ones threads without. A failed guard is not an error, so
            hollow is not red; red marks only pixels that no thread reaches. <TierLegend tiers={['reg', 'global']} />
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi="Ảnh và khối">Picture and blocks</Translated>
          <select aria-label={tr('Preset', 'Cấu hình mẫu')} value={preset?.id ?? 'custom'}
            onChange={(e) => {
              const p = PRESETS.find((x) => x.id === e.target.value);
              if (p) change({ height: p.height, width: p.width, blockW: p.blockW, blockH: p.blockH });
            }}>
            {PRESETS.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
            {preset ? null : <option value="custom">{tr('your own sizes', 'kích thước của bạn')}</option>}
          </select>
        </label>
        <NumField label={tr('Height (rows)', 'Chiều cao (hàng)')} value={launch.height} min={1} max={2000} onChange={(v) => change({ height: v })} />
        <NumField label={tr('Width (columns)', 'Chiều rộng (cột)')} value={launch.width} min={1} max={2000} onChange={(v) => change({ width: v })} />
        <label className="field">
          <Translated vi="Khối, x nhân y">Block, x by y</Translated>
          <select aria-label={tr('Block size', 'Kích thước khối')} value={`${launch.blockW}x${launch.blockH}`}
            onChange={(e) => { const [w, h] = e.target.value.split('x').map(Number); change({ blockW: w, blockH: h }); }}>
            {BLOCK_SHAPES.map(([w, h]) => <option key={`${w}x${h}`} value={`${w}x${h}`}>{w} × {h}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Kiểu khởi chạy">Launch style</Translated>
          <select aria-label={tr('Launch style', 'Kiểu khởi chạy')} value={launch.style} onChange={(e) => change({ style: e.target.value as Style })}>
            {STYLES.map((s) => <option key={s} value={s}>{tr(STYLE_LABEL[s], s === 'blocks' ? 'CUDA hoặc HIP: nguyên khối' : s === 'threadgroups' ? 'Metal: dispatchThreadgroups' : 'Metal: dispatchThreads')}</option>)}
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={!!launch.swapped} onChange={(e) => change({ swapped: e.target.checked })} /> <Translated vi="đổi chỗ hai chiều của lưới (một lỗi)">swap the grid extents (a defect)</Translated>
        </label>
      </div>

      <div className="row" role="status" style={{ marginBottom: '0.8rem', gap: '0.4rem 1.4rem' }}>
        <span data-total="blocks"><Translated vi={<>{nm.unit}: <b>{L.gridX} ngang × {L.gridY} dọc = {L.blocks}</b></>}>{nm.unit}s: <b>{L.gridX} across × {L.gridY} down = {L.blocks}</b></Translated></span>
        <span data-total="threads"><Translated vi={<>luồng: <b>{L.threadsX} ngang × {L.threadsY} dọc = {L.threads}</b></>}>threads: <b>{L.threadsX} across × {L.threadsY} down = {L.threads}</b></Translated></span>
        <span data-total="pixels"><Translated vi={<>điểm ảnh: <b>{launch.width} ngang × {launch.height} dọc = {L.pixels}</b></>}>pixels: <b>{launch.width} across × {launch.height} down = {L.pixels}</b></Translated></span>
        <span data-total="surplus"><Translated vi={<>luồng không có điểm ảnh: <b>{L.surplus}</b></>}>threads with no pixel: <b>{L.surplus}</b></Translated></span>
        {L.unreached > 0 ? <span className="chip fail" data-total="unreached"><Translated vi={<>{L.unreached} điểm ảnh không bao giờ được ghi</>}>{L.unreached} pixels never written</Translated></span> : null}
      </div>
      <p className="muted" style={{ margin: '0 0 0.8rem' }} data-k="launch">
        {launch.style === 'threads' ? (
          <Translated
            vi={
              <>
                Lưới được cho theo số luồng: ({forX}, {forY}). Metal cắt nó thành các nhóm {launch.blockW} × {launch.blockH}{' '}
                và làm các nhóm ở mép xa nhỏ hơn.
              </>
            }
          >
            <>
              The grid is given in threads: ({forX}, {forY}). Metal cuts it into groups of {launch.blockW} × {launch.blockH}{' '}
              and makes the groups on the far edges smaller.
            </>
          </Translated>
        ) : (
          <Translated
            vi={
              <>
                x: {forX} / {launch.blockW} làm tròn lên = {L.gridX} {nm.unit}, tính từ {launch.swapped ? 'chiều cao' : 'chiều rộng'}.{' '}
                y: {forY} / {launch.blockH} làm tròn lên = {L.gridY}, tính từ {launch.swapped ? 'chiều rộng' : 'chiều cao'}.
              </>
            }
          >
            <>
              x: {forX} / {launch.blockW} rounded up = {L.gridX} {nm.unit}s, sized from the {launch.swapped ? 'height' : 'width'}.{' '}
              y: {forY} / {launch.blockH} rounded up = {L.gridY}, sized from the {launch.swapped ? 'width' : 'height'}.
            </>
          </Translated>
        )}
        {launch.swapped ? <> <b className="t-fail"><Translated vi="Hai chiều bị đặt ngược.">The two are the wrong way round.</Translated></b></> : null}
        {launch.style === 'threads' && L.surplus > 0 ? (
          <> <b className="t-fail"><Translated vi={<>Không có phép kiểm tra, {L.surplus} luồng nằm ngoài ảnh sẽ ghi sai chỗ hoặc ghi quá cuối mảng.</>}>With no guard, the {L.surplus} threads outside the picture would write to the wrong place or past the end of the array.</Translated></b></>
        ) : null}
      </p>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 11fr) minmax(0, 9fr)' }}>
        <div>
          {threadLevel ? (
            <Lattice id="map" launch={launch} win={{ x0: 0, y0: 0, cols: extX, rows: extY }} cell={cell} sel={sel} onPick={setSel}
              hot={hot} hover={hover} onHover={setHover}
              label={tr(`Threads of the launch: ${L.threadsX} across and ${L.threadsY} down. Arrow keys move the selected thread.`, `Các luồng của lần khởi chạy: ${L.threadsX} ngang và ${L.threadsY} dọc. Phím mũi tên di chuyển luồng đang chọn.`)} />
          ) : (
            <>
              <Translated
                vi={
                  <p className="muted" style={{ margin: '0 0 0.5rem' }}>
                    Quá nhiều luồng để vẽ từng cái: mỗi ô ở đây là trọn một {nm.unit}. Chọn một {nm.unit}, rồi một
                    luồng bên trong nó.
                  </p>
                }
              >
                <p className="muted" style={{ margin: '0 0 0.5rem' }}>
                  Too many threads to draw one by one: each square here is a whole {nm.unit}. Pick a {nm.unit}, then a
                  thread inside it.
                </p>
              </Translated>
              <BlockLattice id="bmap" launch={launch} cell={blockCell} selBlock={t.blockIdx} hot={hot} hover={hover} onHover={setHover}
                onPick={(bk) => setSel(clampSel(launch, latticeOf(launch, bk.x, bk.y, t.threadIdx.x, t.threadIdx.y)))}
                label={tr(`${nm.Unit}s of the launch: ${L.gridX} across and ${L.gridY} down. Arrow keys move the selected ${nm.unit}.`, `Các ${nm.unit} của lần khởi chạy: ${L.gridX} ngang và ${L.gridY} dọc. Phím mũi tên di chuyển ${nm.unit} đang chọn.`)} />
              <div className="mx-label" style={{ marginTop: '0.8rem' }}>
                <Translated vi={<>{metal ? `Threadgroup tại (${t.blockIdx.x}, ${t.blockIdx.y})` : `Khối ${bookLabel(t.blockIdx)}`}, từng luồng một</>}>
                  {metal ? `Threadgroup at (${t.blockIdx.x}, ${t.blockIdx.y})` : `Block ${bookLabel(t.blockIdx)}`}, thread by thread
                </Translated>
              </div>
              <Lattice id="inset" launch={launch} sel={sel} onPick={setSel}
                cell={Math.max(5, Math.min(18, Math.floor(288 / launch.blockW), Math.floor(320 / launch.blockH)))}
                win={{ x0: t.blockIdx.x * launch.blockW, y0: t.blockIdx.y * launch.blockH, cols: launch.blockW, rows: launch.blockH }}
                label={tr(`Threads of the selected ${nm.unit}. Arrow keys move the selected thread.`, `Các luồng của ${nm.unit} đang chọn. Phím mũi tên di chuyển luồng đang chọn.`)} />
            </>
          )}
          <div className="row" style={{ marginTop: '0.7rem', gap: '0.5rem' }}>
            <span className="muted"><Translated vi="Đi tới">Go to</Translated></span>
            <button data-jump="book" disabled={!bookThread} onClick={() => bookThread && setSel(bookThread)}><Translated vi="Luồng (0,0) của Khối (1,0)">Thread (0,0) of Block (1,0)</Translated></button>
            <button data-jump="surplus" disabled={!firstSurplus} onClick={() => firstSurplus && setSel(firstSurplus)}><Translated vi="một luồng không có điểm ảnh">a thread with no pixel</Translated></button>
            <button data-jump="last" onClick={() => setSel({ x: L.threadsX - 1, y: L.threadsY - 1 })}><Translated vi="luồng cuối cùng">the last thread</Translated></button>
          </div>
        </div>

        <div>
          <div className="panel" data-panel="thread" aria-live="polite">
            <div className="mx-label" data-k="title">
              {metal
                ? <Translated vi={<>Luồng tại vị trí lưới (x {t.col}, y {t.row})</>}>Thread at grid position (x {t.col}, y {t.row})</Translated>
                : <Translated vi={<>Luồng {bookLabel(t.threadIdx)} của Khối {bookLabel(t.blockIdx)}</>}>Thread {bookLabel(t.threadIdx)} of Block {bookLabel(t.blockIdx)}</Translated>}
            </div>
            <div className="muted" style={{ marginBottom: '0.5rem' }}>
              {metal
                ? <Translated vi={<>Metal viết vị trí với x trước. Sách sẽ gọi đây là Luồng {bookLabel(t.threadIdx)} của Khối {bookLabel(t.blockIdx)}.</>}>Metal writes positions x first. The book would call this Thread {bookLabel(t.threadIdx)} of Block {bookLabel(t.blockIdx)}.</Translated>
                : <Translated vi="Nhãn của sách đặt y trước. Các trường bên dưới theo thứ tự x, y.">The book's labels put y first. The fields below are in x, y order.</Translated>}
            </div>
            <div className="mono" style={mono} data-k="blockIdx">{nm.block} = (x <Reg>{t.blockIdx.x}</Reg>, y <Reg>{t.blockIdx.y}</Reg>)</div>
            <div className="mono" style={mono} data-k="threadIdx">{nm.thread} = (x <Reg>{t.threadIdx.x}</Reg>, y <Reg>{t.threadIdx.y}</Reg>)</div>
            <div className="mono" style={mono} data-k="blockDim">
              {launch.style === 'threads' ? tr('dispatch size', 'kích thước dispatch') : nm.dim} = (x {t.blockDim.x}, y {t.blockDim.y})
              {launch.style === 'threads' && (t.groupSize.x !== t.blockDim.x || t.groupSize.y !== t.blockDim.y)
                ? <Translated vi={<>; nhóm ở mép này là ({t.groupSize.x}, {t.groupSize.y})</>}>; this edge group is ({t.groupSize.x}, {t.groupSize.y})</Translated> : null}
            </div>
            <hr style={{ border: 0, borderTop: '1px solid var(--rule)', margin: '0.5rem 0' }} />
            {metal ? (
              <>
                <div className="mono" style={mono} data-k="gid"><Translated vi={<>thread_position_in_grid = (<Reg>{t.col}</Reg>, <Reg>{t.row}</Reg>), được trao cho kernel</>}>thread_position_in_grid = (<Reg>{t.col}</Reg>, <Reg>{t.row}</Reg>), handed to the kernel</Translated></div>
                <div className="mono" style={mono} data-k="row">row = gid.y = <Reg>{t.row}</Reg>{launch.style === 'threadgroups' ? <> (= {t.blockIdx.y}*{t.blockDim.y} + {t.threadIdx.y})</> : null}</div>
                <div className="mono" style={mono} data-k="col">col = gid.x = <Reg>{t.col}</Reg>{launch.style === 'threadgroups' ? <> (= {t.blockIdx.x}*{t.blockDim.x} + {t.threadIdx.x})</> : null}</div>
              </>
            ) : (
              <>
                <div className="mono" style={mono} data-k="row">row = blockIdx.y*blockDim.y + threadIdx.y = {t.blockIdx.y}*{t.blockDim.y} + {t.threadIdx.y} = <Reg>{t.row}</Reg></div>
                <div className="mono" style={mono} data-k="col">col = blockIdx.x*blockDim.x + threadIdx.x = {t.blockIdx.x}*{t.blockDim.x} + {t.threadIdx.x} = <Reg>{t.col}</Reg></div>
              </>
            )}
            <div className="mono" style={mono} data-k="guard">
              {'col < width && row < height: '}
              <Translated vi={<>{t.col} {'<'} {launch.width} là {String(t.colOk)}</>}>{t.col} {'<'} {launch.width} is {String(t.colOk)}</Translated>
              {t.failing === 'col' ? <Translated vi={<>, nên <b>luồng dừng lại</b> (phép kiểm tra hàng không được tính)</>}>, so <b>the thread stops</b> (the row test is not evaluated)</Translated> : (
                <Translated vi={<>; {t.row} {'<'} {launch.height} là {String(t.rowOk)}{t.works ? <>, nên <b>luồng làm việc</b></> : <>, nên <b>luồng dừng lại</b></>}</>}>; {t.row} {'<'} {launch.height} is {String(t.rowOk)}{t.works ? <>, so <b>the thread works</b></> : <>, so <b>the thread stops</b></>}</Translated>
              )}
            </div>
            {launch.style === 'threads' && L.surplus === 0 ? (
              <div className="muted" data-k="noguard"><Translated vi="Không luồng nào của lần khởi chạy này có thể trượt phép kiểm tra, nên kernel có thể bỏ nó đi.">No thread of this launch can fail the test, so the kernel may leave it out.</Translated></div>
            ) : null}
            {t.works ? (
              <>
                <div className="mono" style={mono} data-k="gray">grayOffset = row*width + col = {t.row}*{launch.width} + {t.col} = <Glob>{t.grayOffset}</Glob></div>
                <div className="mono" style={mono} data-k="rgb">rgbOffset = grayOffset*{CHANNELS} = <Glob>{t.rgbOffset}</Glob></div>
                <Strip name={tr('Pout, one byte per pixel', 'Pout, một byte mỗi điểm ảnh')} size={L.pixels} hot={[t.grayOffset!]} />
                <Strip name={tr('Pin, three bytes per pixel', 'Pin, ba byte mỗi điểm ảnh')} size={L.pixels * CHANNELS} hot={[t.rgbOffset!, t.rgbOffset! + 1, t.rgbOffset! + 2]} labels={['r', 'g', 'b']} around={1} />
              </>
            ) : (
              <div className="muted" data-k="nomem" style={{ marginTop: '0.4rem' }}><Translated vi="Không có offset: luồng này không bao giờ tới các dòng tính chúng, và không chạm vào bộ nhớ.">No offsets: this thread never reaches the lines that compute them, and touches no memory.</Translated></div>
            )}
          </div>

          <div className="panel" data-panel="block" style={{ marginTop: '0.8rem' }}>
            <div data-k="block">
              <Translated
                vi={
                  <>
                    <b>{metal ? `${nm.Unit} (x ${b.bx}, y ${b.by})` : `Khối ${bookLabel({ x: b.bx, y: b.by })}`}</b>
                    {hover ? ' (dưới con trỏ)' : ' (của luồng đang chọn)'}: <b>{b.working} trên {b.launched}</b> luồng làm việc
                    {b.working ? <>, {b.workW} ngang × {b.workH} dọc</> : null}.
                  </>
                }
              >
                <b>{metal ? `${nm.Unit} (x ${b.bx}, y ${b.by})` : `Block ${bookLabel({ x: b.bx, y: b.by })}`}</b>
                {hover ? ' (under the pointer)' : " (the selected thread's)"}: <b>{b.working} of {b.launched}</b> threads work
                {b.working ? <>, {b.workW} across × {b.workH} down</> : null}.
              </Translated>
            </div>
            <table className="threads" aria-label={tr('Kinds of block', 'Các loại khối')} style={{ marginTop: '0.5rem' }}>
              <thead>
                <tr><th><Translated vi="loại">kind</Translated></th><th><Translated vi={<>số {nm.unit}</>}>{nm.unit}s</Translated></th><th><Translated vi="làm việc, mỗi khối">working, each</Translated></th><th><Translated vi="tích">product</Translated></th></tr>
              </thead>
              <tbody onMouseLeave={() => setHot(null)}>
                {regs.map((r) => (
                  <tr key={r.region} data-region={r.region} className={b.region === r.region ? 'cur' : ''} onMouseEnter={() => setHot(r.region)}>
                    <td>{r.region === 'idle' ? '' : r.region + ': '}{REGION_NAME[r.region]}</td>
                    <td>{r.blocks}</td>
                    <td>{r.working}{r.launched !== r.working ? tr(` of ${r.launched}`, ` trên ${r.launched}`) : ''}</td>
                    <td>{r.total}</td>
                  </tr>
                ))}
                <tr data-region="sum">
                  <td colSpan={3}><Translated vi="tổng số luồng làm việc">working threads in all</Translated></td>
                  <td><b>{total}</b></td>
                </tr>
              </tbody>
            </table>
            <div className="muted" data-k="sum" style={{ marginTop: '0.4rem' }}>
              {total === L.pixels
                ? <Translated vi={<>{total} = {launch.height} × {launch.width}: mỗi điểm ảnh có đúng một luồng.</>}>{total} = {launch.height} × {launch.width}: every pixel has exactly one thread.</Translated>
                : <span className="t-fail"><Translated vi={<>{total} thiếu {L.unreached} điểm ảnh so với {launch.height} × {launch.width} = {L.pixels}.</>}>{total} is short of {launch.height} × {launch.width} = {L.pixels} by {L.unreached} pixels.</Translated></span>}
              {regs.some((r) => r.region === 'idle') ? <Translated vi=" Các khối thuộc loại cuối không làm việc gì cả."> Blocks of the last kind do no work at all.</Translated> : null}
            </div>
          </div>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the kernel stepper

const REG_ORDER = ['col', 'row', 'grayOffset', 'rgbOffset', 'r', 'g', 'b', 'sum', 'byte'];
const KERNEL_FIRST = 3, KERNEL_LAST = 21;

function KernelStepper({ launch, sel, setSel }: { launch: Launch; sel: XY; setSel: (p: XY) => void }) {
  const [n, setN] = useState(0);
  const locale = useLocale();
  const steps = useMemo(() => stepsGrayscale(launch, sel.x, sel.y), [launch, sel.x, sel.y, locale]);
  const L = layout(launch);
  const t = threadAt(launch, sel.x, sel.y);
  const at = Math.min(n, steps.length);
  const cur = at > 0 ? steps[at - 1] : undefined;
  const regs = cur?.regs ?? {};
  const read = new Map(steps.slice(0, at).filter((s) => s.mem && !s.mem.write).map((s) => [s.mem!.index, s.mem!.value]));
  const stored = steps.slice(0, at).find((s) => s.mem?.write)?.mem;
  const pick = (p: XY | null) => { if (p) { setSel(p); setN(0); } };
  const bookThread = L.gridY > 1 ? latticeOf(launch, 0, 1, 0, 0) : null;
  const right: XY | null = L.threadsX > L.coveredCols ? { x: L.coveredCols, y: Math.min(5, L.threadsY - 1) } : null;
  const below: XY | null = L.threadsY > L.coveredRows ? { x: Math.min(3, L.threadsX - 1), y: L.coveredRows } : null;
  const body = GRAY_CU.split('\n').slice(KERNEL_FIRST - 1, KERNEL_LAST).join('\n');
  return (
    <Bench
      title={tr('Step this thread through the kernel', 'Chạy từng bước luồng này qua kernel')}
      task={
        <Translated vi="Chạy từng bước luồng của sách tới lệnh ghi của nó. Rồi chạy từng bước một luồng không có điểm ảnh và xem nó dừng ở đâu.">
          Step the book's thread to its store. Then step a thread with no pixel and see where it stops.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Luồng ở đây là luồng đang chọn trong bản đồ luồng; đổi nó ở đây thì bên kia cũng đổi. Giá trị điểm ảnh lấy
              từ một ảnh tổng hợp nhỏ. <TierLegend tiers={['reg', 'global']} />
            </>
          }
        >
          <>
            The thread is the one selected in the thread map; changing it here changes it there. Pixel values come from a
            small synthetic picture. <TierLegend tiers={['reg', 'global']} />
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', gap: '0.5rem' }}>
        <span data-k="who">
          <Translated
            vi={
              <>
                <b>Luồng {bookLabel(t.threadIdx)} của Khối {bookLabel(t.blockIdx)}</b>, ảnh {launch.height} × {launch.width}, khối {launch.blockW} × {launch.blockH}.
              </>
            }
          >
            <b>Thread {bookLabel(t.threadIdx)} of Block {bookLabel(t.blockIdx)}</b>, picture {launch.height} × {launch.width}, blocks {launch.blockW} × {launch.blockH}.
          </Translated>
        </span>
        <span className="muted"><Translated vi="Dùng">Use</Translated></span>
        <button data-pick="book" disabled={!bookThread} onClick={() => pick(bookThread)}><Translated vi="Luồng (0,0) của Khối (1,0)">Thread (0,0) of Block (1,0)</Translated></button>
        <button data-pick="right" disabled={!right} onClick={() => pick(right)}><Translated vi="luồng ngay sau mép phải">one past the right edge</Translated></button>
        <button data-pick="below" disabled={!below} onClick={() => pick(below)}><Translated vi="luồng ngay dưới mép dưới">one below the bottom edge</Translated></button>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setN(0)} disabled={at === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => setN(at - 1)} disabled={at === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => setN(at + 1)} disabled={at >= steps.length}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => setN(steps.length)} disabled={at >= steps.length}><Translated vi="Chạy đến cuối">Run to the end</Translated></button>
        <span className="muted" aria-live="polite"><Translated vi={<>bước {at} trên {steps.length}</>}>step {at} of {steps.length}</Translated></span>
      </div>
      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }}>
        {cur ? cur.text : <Translated vi={<>Chưa có gì chạy. Mỗi luồng của lưới bắt đầu ở dòng đầu tiên của kernel với <code>blockIdx</code> và <code>threadIdx</code> riêng của nó. Bấm Bước.</>}>Nothing has run yet. Every thread of the grid starts at the first line of the kernel with its own <code>blockIdx</code> and <code>threadIdx</code>. Press Step.</Translated>}
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 7fr) minmax(0, 5fr)' }}>
        <div>
          <Code tabs={[{ label: tr('kernel', 'kernel'), code: body, tiers: TIERS }]} firstLine={KERNEL_FIRST} highlight={cur ? [LINE[cur.op]] : []} />
        </div>
        <div>
          <div className="mx-label t-reg"><Translated vi="Thanh ghi của luồng này">Registers of this thread</Translated></div>
          <table className="threads" aria-label={tr('Registers of this thread', 'Thanh ghi của luồng này')}>
            <tbody>
              {REG_ORDER.map((name) => (
                <tr key={name} data-reg={name} className={cur?.set === name ? 'cur' : ''}>
                  <td>{name === 'sum' ? tr('the float sum', 'tổng float') : name === 'byte' ? tr('after the cast', 'sau phép ép kiểu') : name}</td>
                  <td className="t-reg">{regs[name] === undefined ? '·' : name === 'sum' ? fmtFloat(regs[name]) : regs[name]}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {t.works ? (
            <div data-k="mem">
              <div className="mx-label t-global" style={{ marginTop: '0.8rem' }}><Translated vi="Pin (bộ nhớ toàn cục): ba byte của điểm ảnh này">Pin (global memory): the three bytes of this pixel</Translated></div>
              <div className="mx global" style={{ gridTemplateColumns: 'repeat(3, auto)' }}>
                {[0, 1, 2].map((k) => {
                  const i = t.rgbOffset! + k;
                  return <small key={'i' + k} className="muted" style={{ textAlign: 'center', padding: '0 0.3rem' }}>[{i}]</small>;
                })}
                {[0, 1, 2].map((k) => {
                  const i = t.rgbOffset! + k;
                  const hotNow = cur?.mem && !cur.mem.write && cur.mem.index === i;
                  return <span key={k} data-pin={i} className={hotNow ? 'hot' : read.has(i) ? 'on' : ''}>{pixelValue(steps, i)}</span>;
                })}
              </div>
              <div className="mx-label t-global" style={{ marginTop: '0.8rem' }}><Translated vi="Pout (bộ nhớ toàn cục): phần tử duy nhất của luồng này">Pout (global memory): this thread's one element</Translated></div>
              <div className="mx global" style={{ gridTemplateColumns: 'auto' }}>
                <small className="muted" style={{ textAlign: 'center' }}>[{t.grayOffset}]</small>
                <span data-pout={t.grayOffset} className={cur?.mem?.write ? 'hot' : stored ? 'on' : 'void'}>{stored ? stored.value : ''}</span>
              </div>
            </div>
          ) : (
            <p className="muted" data-k="mem" style={{ margin: '0.8rem 0 0' }}>
              <Translated vi="Luồng này không đọc cũng không ghi gì. Khối của nó vẫn phải được khởi chạy trọn vẹn.">
                This thread reads and writes nothing. Its block still had to be launched whole.
              </Translated>
            </p>
          )}
        </div>
      </div>
    </Bench>
  );
}

/** The byte the synthetic picture holds at a Pin index, taken from the trace so the component computes nothing. */
function pixelValue(steps: ReturnType<typeof stepsGrayscale>, index: number): number | string {
  return steps.find((s) => s.mem && !s.mem.write && s.mem.index === index)?.mem!.value ?? '';
}

// ---------------------------------------------------------------- the flattening explorer

const KIND_LABEL: Record<Indexing, string> = {
  get row() { return tr('row-major', 'thứ tự hàng'); },
  get col() { return tr('column-major', 'thứ tự cột'); },
  get wrong() { return tr('wrong stride (a defect)', 'sai bước nhảy (một lỗi)'); },
};

function FlatteningExplorer() {
  const [height, setHeight] = useState(3);
  const [width, setWidth] = useState(5);
  const [depth, setDepth] = useState(1);
  const [kind, setKind] = useState<Indexing>('row');
  const [showStrides, setShowStrides] = useState(false);
  const [pick, setPick] = useState({ plane: 0, row: 1, col: 2 });
  const d: Dims = { depth: kind === 'col' ? 1 : depth, height, width };
  const s = { plane: Math.min(pick.plane, d.depth - 1), row: Math.min(pick.row, height - 1), col: Math.min(pick.col, width - 1) };
  const off = offsetOf(d, kind, s.plane, s.row, s.col);
  const cov = useMemo(() => indexingCoverage({ depth: kind === 'col' ? 1 : depth, height, width }, kind), [depth, height, width, kind]);
  const st = stridesFor(d, kind);
  const three = d.depth > 1;
  const lab = (e: { plane: number; row: number; col: number }) => (three ? `${e.plane},${e.row},${e.col}` : `${e.row},${e.col}`);
  // Where one step along each dimension lands, for the stride markers.
  const next: { slot: number; text: string }[] = [];
  if (showStrides && kind !== 'wrong') {
    if (s.col + 1 < width) next.push({ slot: off + st[2], text: 'col+1' }); // i18n-ignore: index expression
    if (s.row + 1 < height) next.push({ slot: off + st[1], text: 'row+1' }); // i18n-ignore: index expression
    if (s.plane + 1 < d.depth) next.push({ slot: off + st[0], text: 'plane+1' }); // i18n-ignore: index expression
  }
  const firstAt = (slot: number) => {
    for (let plane = 0; plane < d.depth; plane++)
      for (let row = 0; row < height; row++)
        for (let col = 0; col < width; col++) if (offsetOf(d, kind, plane, row, col) === slot) return { plane, row, col };
    return null;
  };
  const pickSlot = (slot: number) => {
    const e = kind === 'wrong' ? firstAt(slot) : elementAt(d, kind, slot);
    if (e) setPick(e);
  };
  const formula =
    kind === 'row'
      ? three
        ? `plane*height*width + row*width + col = ${s.plane}*${height}*${width} + ${s.row}*${width} + ${s.col}`
        : `row*width + col = ${s.row}*${width} + ${s.col}`
      : kind === 'col'
        ? `col*height + row = ${s.col}*${height} + ${s.row}`
        : three
          ? `plane*height*width + row*height + col = ${s.plane}*${height}*${width} + ${s.row}*${height} + ${s.col}`
          : `row*height + col = ${s.row}*${height} + ${s.col}`;
  const shape = three ? [d.depth, height, width] : [height, width];
  const strideTuple = three ? st : st.slice(1);
  const names = three ? ['plane', 'row', 'col'] : ['row', 'col'];
  const idxs = three ? [s.plane, s.row, s.col] : [s.row, s.col];
  const slotBox: CSSProperties = { display: 'grid', justifyItems: 'center', gap: 1 };

  return (
    <Bench
      title={tr('Flatten an array', 'Làm phẳng một mảng')}
      task={tr('Click an element to see its slot, or a slot to get the indices back.', 'Bấm vào một phần tử để xem ô nhớ của nó, hoặc vào một ô nhớ để lấy lại các chỉ số.')}
      foot={tr("Slots are numbered from 0. Each element is one slot wide here; in memory, multiply the slot by the element's size in bytes.", 'Các ô nhớ được đánh số từ 0. Ở đây mỗi phần tử rộng một ô; trong bộ nhớ, hãy nhân số ô với kích thước phần tử tính bằng byte.')}
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi={<>Chiều cao (hàng): {height}</>}>Height (rows): {height}</Translated>
          <input type="range" min="1" max="6" value={height} aria-label={tr('Height', 'Chiều cao')} onChange={(e) => setHeight(+e.target.value)} />
        </label>
        <label className="field">
          <Translated vi={<>Chiều rộng (cột): {width}</>}>Width (columns): {width}</Translated>
          <input type="range" min="1" max="8" value={width} aria-label={tr('Width', 'Chiều rộng')} onChange={(e) => setWidth(+e.target.value)} />
        </label>
        <label className="field">
          <Translated vi={<>Chiều sâu (mặt): {d.depth}</>}>Depth (planes): {d.depth}</Translated>
          <input type="range" min="1" max="3" value={d.depth} aria-label={tr('Depth', 'Chiều sâu')} disabled={kind === 'col'} onChange={(e) => setDepth(+e.target.value)} />
        </label>
        <span className="seg" role="group" aria-label={tr('Indexing', 'Cách đánh chỉ số')}>
          {(['row', 'col', 'wrong'] as Indexing[]).map((k) => (
            <button key={k} aria-pressed={kind === k} onClick={() => setKind(k)}>{KIND_LABEL[k]}</button>
          ))}
        </span>
        <label className="check">
          <input type="checkbox" checked={showStrides} onChange={(e) => setShowStrides(e.target.checked)} /> <Translated vi="hiện các bước nhảy">show the strides</Translated>
        </label>
      </div>

      <div className="row" style={{ alignItems: 'flex-start', gap: '1.4rem', marginBottom: '0.9rem' }}>
        {Array.from({ length: d.depth }, (_, plane) => (
          <div key={plane}>
            <div className="mx-label">{three ? tr(`plane ${plane}`, `mặt ${plane}`) : `M, ${height} × ${width}`}</div>
            <div className="mx global" data-grid={plane} style={{ gridTemplateColumns: `repeat(${width}, auto)` }}>
              {Array.from({ length: height * width }, (_, i) => {
                const e = { plane, row: Math.floor(i / width), col: i % width };
                const on = e.plane === s.plane && e.row === s.row && e.col === s.col;
                return (
                  <span key={i} role="button" tabIndex={0} aria-pressed={on} data-el={`${e.plane},${e.row},${e.col}`} className={on ? 'hot' : ''}
                    style={{ cursor: 'pointer' }} onClick={() => setPick(e)} onKeyDown={onKey(() => setPick(e))}>
                    {lab(e)}
                  </span>
                );
              })}
            </div>
          </div>
        ))}
        {kind === 'col' ? (
          <div>
            <div className="mx-label"><Translated vi={<>chuyển vị của M, {width} × {height}</>}>the transpose of M, {width} × {height}</Translated></div>
            <div className="mx global" data-grid="transpose" style={{ gridTemplateColumns: `repeat(${height}, auto)` }}>
              {Array.from({ length: height * width }, (_, i) => {
                // row i of the transpose is column i of M
                const e = { plane: 0, row: i % height, col: Math.floor(i / height) };
                const on = e.row === s.row && e.col === s.col;
                return <span key={i} className={on ? 'hot' : ''}>{lab(e)}</span>;
              })}
            </div>
            <div className="muted" style={{ maxWidth: '16rem', marginTop: '0.3rem' }}><Translated vi="Đọc ma trận chuyển vị theo từng hàng: đó chính là dải bên dưới.">Read the transpose row by row: it is the strip below.</Translated></div>
          </div>
        ) : null}
      </div>

      <div className="narrate" role="status" style={{ marginBottom: '0.9rem', minHeight: 0 }}>
        <span className="mono" data-k="formula">
          <Translated vi={<>phần tử [{lab(s)}] nằm ở ô {formula} = <Glob>{off}</Glob></>}>
            element [{lab(s)}] is at slot {formula} = <Glob>{off}</Glob>
          </Translated>
        </span>
        {kind === 'wrong' ? <> <span className="t-fail"><Translated vi={<>Bước nhảy phải là chiều rộng, {width}, chứ không phải chiều cao, {height}.</>}>The stride should be the width, {width}, not the height, {height}.</Translated></span></> : null}
        {showStrides ? (
          <div className="mono" data-k="strides" style={{ marginTop: '0.3rem' }}>
            <Translated
              vi={
                <>
                  {kind === 'wrong' ? 'các bước nhảy mà lỗi này dùng' : kind === 'col' ? 'bước nhảy' : 'x.stride()'} với shape ({shape.join(', ')}) = ({strideTuple.join(', ')});
                  ô = {idxs.map((v, i) => `${v}*${strideTuple[i]}`).join(' + ')} = {off}.{' '}
                </>
              }
            >
              {kind === 'wrong' ? 'the strides this defect uses' : kind === 'col' ? 'strides' : 'x.stride()'} for shape ({shape.join(', ')}) = ({strideTuple.join(', ')});
              slot = {idxs.map((v, i) => `${v}*${strideTuple[i]}`).join(' + ')} = {off}.{' '}
            </Translated>
            <span className="muted">
              <Translated vi={<>Một bước dọc theo {names.join(', ')} dịch đi {strideTuple.join(', ')} ô.</>}>
                One step along {names.join(', ')} moves {strideTuple.join(', ')} slots.
              </Translated>
              {kind === 'col' ? tr(' Not in decreasing order, so PyTorch would not call a tensor with these strides contiguous.', ' Không theo thứ tự giảm dần, nên PyTorch sẽ không gọi một tensor có các bước nhảy này là liền kề (contiguous).') : ''}
            </span>
          </div>
        ) : null}
      </div>

      <div className="mx-label t-global"><Translated vi={<>mảng phẳng, {cov.size} ô</>}>the flat array, {cov.size} slots</Translated></div>
      <div data-strip style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 2px', alignItems: 'flex-start' }}>
        {cov.hits.map((h, i) => {
          const e = kind === 'wrong' ? null : elementAt(d, kind, i);
          const nx = next.find((x) => x.slot === i);
          const here = i === off;
          // With the wrong stride a slot's look says how often it is written, so the selection is an outline.
          const cls = kind === 'wrong' ? (h === 0 ? 'void' : h > 1 ? 'bad' : '') : here ? 'hot' : nx ? 'on' : '';
          return (
            <div key={i} style={slotBox}>
              <small className="muted" style={{ fontSize: '0.64rem' }}>{i}</small>
              <div className="mx global">
                <span role="button" tabIndex={0} data-slot={i} data-hits={h} data-here={here ? '' : undefined} className={cls}
                  style={{ cursor: 'pointer', boxShadow: here && kind === 'wrong' ? '0 0 0 2px var(--ink)' : undefined }}
                  onClick={() => pickSlot(i)} onKeyDown={onKey(() => pickSlot(i))}>
                  {e ? lab(e) : h === 0 ? '' : `×${h}`}
                </span>
              </div>
              <small style={{ fontSize: '0.62rem', minHeight: '0.9rem' }}>{nx ? nx.text : ''}</small>
            </div>
          );
        })}
        {cov.outOfBounds.map((o) => (
          <div key={`o${o.plane},${o.row},${o.col}`} style={slotBox}>
            <small className="t-fail" style={{ fontSize: '0.64rem' }}>{o.offset}</small>
            <div className="mx global">
              <span data-oob={o.offset} className="bad" style={{ borderStyle: 'dashed' }}>{lab(o)}</span>
            </div>
            <small style={{ fontSize: '0.62rem', minHeight: '0.9rem' }} />
          </div>
        ))}
      </div>
      {kind === 'wrong' ? (
        <p data-k="coverage" style={{ margin: '0.8rem 0 0' }}>
          {cov.collisions.length + cov.holes.length + cov.outOfBounds.length === 0 ? (
            <span className="chip ok"><Translated vi="mỗi ô được ghi đúng một lần">every slot written exactly once</Translated></span>
          ) : (
            <>
              {cov.collisions.length ? <><span className="chip fail"><Translated vi={<>{cov.collisions.length} ô bị ghi hai lần trở lên</>}>{cov.collisions.length} slots written twice or more</Translated></span>{' '}</> : null}
              {cov.holes.length ? <><span className="chip fail"><Translated vi={<>{cov.holes.length} ô không bao giờ được ghi</>}>{cov.holes.length} slots never written</Translated></span>{' '}</> : null}
              {cov.outOfBounds.length ? <span className="chip fail"><Translated vi={<>{cov.outOfBounds.length} lần ghi quá cuối mảng</>}>{cov.outOfBounds.length} writes past the end of the array</Translated></span> : null}
            </>
          )}{' '}
          {height === width
            ? tr('Height equals width, so the wrong stride is the right number and the defect cannot be seen.', 'Chiều cao bằng chiều rộng, nên bước nhảy sai lại đúng là con số cần, và không thể thấy lỗi.')
            : tr('Each slot shows how many elements land on it; dashed red slots do not exist.', 'Mỗi ô cho biết bao nhiêu phần tử rơi vào nó; các ô đỏ viền đứt nét không tồn tại.')}
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the erratum hunt

type Finding = 'cast' | 'constants' | 'args' | 'spelling';
export interface Spot {
  id: string;
  /** 'k' for the kernel listing, 'h' for the host call. */
  where: 'k' | 'h';
  /** 1-based line in that listing. */
  line: number;
  /** The exact text of the spot on that line. */
  text: string;
  finding?: Finding;
  /** Why a spot that is not a defect is right. */
  fine?: string;
}

export const SPOTS: Spot[] = [
  { id: 'name', where: 'k', line: 4, text: 'colortoGrayscaleConvertion', finding: 'spelling' },
  { id: 'params', where: 'k', line: 4, text: 'unsigned char * Pout', finding: 'args' }, // i18n-ignore: code
  { id: 'col', where: 'k', line: 6, text: 'blockIdx.x*blockDim.x + threadIdx.x', get fine() { return tr('Columns run along x, so col comes from the x fields.', 'Cột chạy dọc theo x, nên col lấy từ các trường x.'); } }, // i18n-ignore: code
  { id: 'row', where: 'k', line: 7, text: 'blockIdx.y*blockDim.y + threadIdx.y', get fine() { return tr('Rows run along y, so row comes from the y fields.', 'Hàng chạy dọc theo y, nên row lấy từ các trường y.'); } }, // i18n-ignore: code
  { id: 'guard', where: 'k', line: 8, text: 'col < width && row < height', get fine() { return tr('Both tests are needed and both comparisons are strict: valid columns are 0 to width - 1.', 'Cả hai phép kiểm tra đều cần, và cả hai phép so sánh đều là so sánh chặt: các cột hợp lệ là từ 0 đến width - 1.'); } },
  { id: 'gray', where: 'k', line: 10, text: 'row*width + col', get fine() { return tr('Each row above contributes width elements. This is row-major order.', 'Mỗi hàng phía trên góp width phần tử. Đây là thứ tự hàng (row-major).'); } }, // i18n-ignore: code
  { id: 'rgb', where: 'k', line: 13, text: 'grayOffset*CHANNELS', get fine() { return tr('Every pixel before this one took CHANNELS bytes.', 'Mỗi điểm ảnh trước điểm này chiếm CHANNELS byte.'); } }, // i18n-ignore: code
  { id: 'blue', where: 'k', line: 16, text: 'rgbOffset + 2', get fine() { return tr('Blue is the third of the three bytes.', 'Xanh lam là byte thứ ba trong ba byte.'); } }, // i18n-ignore: code
  { id: 'cast', where: 'k', line: 19, text: '(unsigned char)', finding: 'cast' }, // i18n-ignore: code
  { id: 'consts', where: 'k', line: 19, text: '0.21f*r+0.71f*g+0.07f*b', finding: 'constants' },
  { id: 'grid', where: 'h', line: 1, text: 'ceil(m/16.0), ceil(n/16.0)', get fine() { return tr('x is sized from the width m and y from the height n, as it must be.', 'x được tính từ chiều rộng m và y từ chiều cao n, đúng như phải thế.'); } }, // i18n-ignore: code
  { id: 'block', where: 'h', line: 2, text: '16, 16, 1', get fine() { return tr('16 × 16 × 1 is 256 threads, a legal block.', '16 × 16 × 1 là 256 luồng, một khối hợp lệ.'); } },
  { id: 'hname', where: 'h', line: 3, text: 'colorToGrayscaleConversion', finding: 'spelling' },
  { id: 'args', where: 'h', line: 3, text: 'Pin_d, Pout_d, m, n', finding: 'args' }, // i18n-ignore: code
];

const FINDINGS: { id: Finding; title: string; body: ReactNode; scored: boolean }[] = [
  {
    id: 'cast', scored: true, get title() { return tr('The cast covers one constant, not the sum (line 19)', 'Phép ép kiểu chỉ phủ một hằng số, không phải cả tổng (dòng 19)'); },
    body: (
      <Translated
        vi={
          <>
            Phép ép kiểu áp dụng cho biểu thức ngay sau nó, còn <Cw>*</Cw> có độ ưu tiên thấp hơn. Dòng này được phân tích thành{' '}
            <Cw>((unsigned char)0.21f)*r + 0.71f*g + 0.07f*b</Cw>. <Cw>(unsigned char)0.21f</Cw> bằng 0, nên màu đỏ bị
            nhân với 0 và giá trị được lưu là phần nguyên bị cắt của <Cw>0.71f*g + 0.07f*b</Cw>. Nó biên dịch
            mà không có cảnh báo nào.
          </>
        }
      >
        <>
          A cast applies to the expression right after it, and <Cw>*</Cw> binds less tightly. The line parses as{' '}
          <Cw>((unsigned char)0.21f)*r + 0.71f*g + 0.07f*b</Cw>. <Cw>(unsigned char)0.21f</Cw> is 0, so red is
          multiplied by 0 and what is stored is the truncated value of <Cw>0.71f*g + 0.07f*b</Cw>. It compiles
          without a warning.
        </>
      </Translated>
    ),
  },
  {
    id: 'constants', scored: true, get title() { return tr('The constants are not the ones in the text (line 19)', 'Các hằng số không phải là những hằng số trong phần chữ (dòng 19)'); },
    body: (
      <Translated
        vi={
          <>
            Phần chữ của sách ở p. 52 cho 0.299, 0.587 và 0.114. Mã in có 0.21, 0.71 và 0.07, cộng lại được
            0.99, nên kể cả khi đã sửa phép ép kiểu, một điểm ảnh trắng vẫn ra 252.
          </>
        }
      >
        <>
          The book's text on p. 52 gives 0.299, 0.587 and 0.114. The listing has 0.21, 0.71 and 0.07, which add up to
          0.99, so even with the cast fixed a white pixel comes out as 252.
        </>
      </Translated>
    ),
  },
  {
    id: 'args', scored: true, get title() { return tr('The call passes input and output the wrong way round', 'Lời gọi truyền đầu vào và đầu ra ngược nhau'); },
    body: (
      <Translated
        vi={
          <>
            Kernel được khai báo là <Cw>(Pout, Pin, width, height)</Cw>; lời gọi ở p. 50 truyền{' '}
            <Cw>(Pin_d, Pout_d, m, n)</Cw>. Như bản in, kernel sẽ đọc từ buffer đầu ra và ghi vào chính
            ảnh mà lẽ ra nó phải đọc.
          </>
        }
      >
        <>
          The kernel is declared <Cw>(Pout, Pin, width, height)</Cw>; the call on p. 50 passes{' '}
          <Cw>(Pin_d, Pout_d, m, n)</Cw>. As printed, the kernel would read from the output buffer and write into the
          picture it was meant to read.
        </>
      </Translated>
    ),
  },
  {
    id: 'spelling', scored: false, get title() { return tr('The name is spelt two ways', 'Tên được viết theo hai cách'); },
    body: (
      <Translated
        vi={
          <>
            <Cw>colortoGrayscaleConvertion</Cw> trong mã in, <Cw>colorToGrayscaleConversion</Cw> trong lời gọi, phần
            chữ và chú thích hình. Đây là lỗi duy nhất trong bốn lỗi mà trình biên dịch sẽ từ chối.
          </>
        }
      >
        <>
          <Cw>colortoGrayscaleConvertion</Cw> in the listing, <Cw>colorToGrayscaleConversion</Cw> in the call, the
          text and the caption. This is the only one of the four a compiler would refuse.
        </>
      </Translated>
    ),
  },
];

function Listing({ title, lines, where, spots, marked, toggle, shown, firstLine = 1 }: {
  title: string; lines: string[]; where: 'k' | 'h'; spots: Spot[]; marked: Set<string>; toggle: (id: string) => void;
  /** After a check: marked spots are coloured right or wrong; with 'all', missed defects are outlined too. */
  shown: 'none' | 'marks' | 'all'; firstLine?: number;
}) {
  return (
    <div style={{ marginBottom: '0.9rem' }}>
      <div className="mx-label">{title}</div>
      <div className="mono" data-listing={where}
        style={{ background: 'var(--cell)', border: '1px solid var(--rule)', borderRadius: 4, padding: '0.6rem 0.8rem 0.6rem 0', overflowX: 'auto', fontSize: '0.8rem', lineHeight: 1.75, whiteSpace: 'pre' }}>
        {lines.map((text, i) => {
          const n = i + 1;
          const here = spots.filter((s) => s.where === where && s.line === n).sort((a, b) => text.indexOf(a.text) - text.indexOf(b.text));
          const out: ReactNode[] = [];
          let pos = 0;
          for (const s of here) {
            const at = text.indexOf(s.text, pos);
            if (at < 0) continue;
            out.push(text.slice(pos, at));
            const on = marked.has(s.id);
            const state = shown === 'none' ? (on ? 'marked' : 'idle') : on ? (s.finding ? 'right' : 'wrong') : shown === 'all' && s.finding ? 'missed' : 'idle';
            out.push(
              <span key={s.id} role="button" tabIndex={0} aria-pressed={on} data-spot={s.id} data-state={state}
                onClick={() => toggle(s.id)} onKeyDown={onKey(() => toggle(s.id))}
                style={{
                  cursor: 'pointer', borderRadius: 3, padding: '0.1em 0.15em',
                  borderBottom: state === 'idle' ? '1px dashed var(--ink-3)' : undefined,
                  outline: state === 'marked' ? '2px solid var(--ink)' : state === 'missed' ? '2px dashed var(--fail)' : state === 'right' ? '2px solid var(--ok)' : state === 'wrong' ? '2px solid var(--fail)' : undefined,
                  background: state === 'marked' ? 'var(--rule)' : state === 'right' ? 'var(--ok-wash)' : state === 'wrong' ? 'var(--fail-wash)' : undefined,
                }}>
                {s.text}
              </span>,
            );
            pos = at + s.text.length;
          }
          out.push(text.slice(pos));
          return (
            <div key={i}>
              <span className="muted" style={{ display: 'inline-block', width: '2.6rem', paddingRight: '0.9rem', textAlign: 'right', userSelect: 'none' }}>
                {String(n + firstLine - 1).padStart(2, '0')}
              </span>
              {out}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ErratumHunt() {
  const [marked, setMarked] = useState<Set<string>>(new Set());
  const [checked, setChecked] = useState(false);
  const [answers, setAnswers] = useState(false);
  const toggle = (id: string) => {
    const m = new Set(marked);
    if (m.has(id)) m.delete(id);
    else m.add(id);
    setMarked(m);
    setChecked(false);
  };
  const hit = (f: Finding) => SPOTS.some((s) => s.finding === f && marked.has(s.id));
  const scored = FINDINGS.filter((f) => f.scored);
  const found = scored.filter((f) => hit(f.id));
  const alarms = SPOTS.filter((s) => !s.finding && marked.has(s.id));
  const shown = answers ? 'all' : checked ? 'marks' : 'none';
  return (
    <Bench
      title={tr('Find what is wrong in Fig. 3.4', 'Tìm chỗ sai trong Fig. 3.4')}
      task={tr('Mark every underlined spot you believe is wrong, in the kernel and in the call, then check.', 'Đánh dấu mọi chỗ gạch chân mà bạn cho là sai, trong kernel và trong lời gọi, rồi kiểm tra.')}
      foot={tr('The code lines are as printed on pp. 50 and 53; the comments are reworded. Fourteen spots can be marked; most of them are right.', 'Các dòng mã giữ đúng như in ở pp. 50 và 53; phần chú thích đã được viết lại. Có mười bốn chỗ có thể đánh dấu; phần lớn trong số đó là đúng.')}
    >
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 1fr)' }}>
        <div>
          <Listing title={tr('The kernel, as printed', 'Kernel, như bản in')} lines={PRINTED.split('\n')} where="k" spots={SPOTS} marked={marked} toggle={toggle} shown={shown} />
          <Listing title={tr('The launch, as printed (n rows, m columns)', 'Lệnh khởi chạy, như bản in (n hàng, m cột)')} lines={PRINTED_HOST} where="h" spots={SPOTS} marked={marked} toggle={toggle} shown={shown} />
        </div>
        <div>
          <div className="row" style={{ marginBottom: '0.7rem' }}>
            <button className="primary" disabled={marked.size === 0} onClick={() => setChecked(true)}><Translated vi="Kiểm tra các đánh dấu">Check my marks</Translated></button>
            <button onClick={() => { setMarked(new Set()); setChecked(false); setAnswers(false); }} disabled={marked.size === 0 && !answers}><Translated vi="Xóa">Clear</Translated></button>
            <button onClick={() => { setChecked(true); setAnswers(true); }} disabled={answers}><Translated vi="Hiện đáp án">Show the answers</Translated></button>
            <span className="muted"><Translated vi={<>đã đánh dấu {marked.size}</>}>{marked.size} marked</Translated></span>
          </div>
          {checked ? (
            <div className="panel" role="status" data-k="hunt" data-found={found.length}>
              <p style={{ margin: '0 0 0.5rem' }}>
                <b><Translated vi={<>Đã tìm thấy {found.length} trên 3 lỗi.</>}>{found.length} of 3 defects found.</Translated></b>{' '}
                {found.length < 3 && !answers ? tr('Keep looking, or show the answers.', 'Tìm tiếp, hoặc hiện đáp án.') : ''}
              </p>
              {FINDINGS.filter((f) => answers || hit(f.id)).map((f) => (
                <p key={f.id} data-finding={f.id} style={{ margin: '0 0 0.6rem' }}>
                  <span className={'chip ' + (hit(f.id) ? 'ok' : 'fail')}>{hit(f.id) ? tr('found', 'đã tìm thấy') : tr('missed', 'bỏ sót')}</span>{' '}
                  <b>{f.title}.</b>{f.scored ? '' : tr(' (Not scored.)', ' (Không tính điểm.)')} {f.body}
                </p>
              ))}
              {alarms.map((s) => (
                <p key={s.id} data-alarm={s.id} style={{ margin: '0 0 0.6rem' }}>
                  <span className="chip fail"><Translated vi="không phải lỗi">not a defect</Translated></span> <Cw>{s.text}</Cw>: {s.fine}
                </p>
              ))}
            </div>
          ) : (
            <Translated
              vi={
                <p className="muted" style={{ margin: 0 }}>
                  Ba lỗi làm thay đổi những gì chương trình làm. Một lỗi nữa là viết nhầm một cái tên. Chỉ cần đối chiếu các
                  dòng với nhau là đủ; bạn không cần chạy gì cả.
                </p>
              }
            >
              <p className="muted" style={{ margin: 0 }}>
                Three defects change what the program does. One more is a slip in a name. Reading the lines against each
                other is enough; you do not need to run anything.
              </p>
            </Translated>
          )}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the pixel tester

const VARIANTS: { id: GrayVariant; name: string; expr: (w: [number, number, number]) => string }[] = [
  { id: 'printed', get name() { return tr('Line 19 as printed', 'Dòng 19 như bản in'); }, expr: (w) => `(unsigned char) ${w[0]}f*r + ${w[1]}f*g + ${w[2]}f*b` }, // i18n-ignore: code expression
  { id: 'printedParen', get name() { return tr('Printed constants, sum in parentheses', 'Hằng số như bản in, tổng trong ngoặc'); }, expr: (w) => `(unsigned char)(${w[0]}f*r + ${w[1]}f*g + ${w[2]}f*b)` }, // i18n-ignore: code expression
  { id: 'corrected', get name() { return tr('Corrected line', 'Dòng đã sửa'); }, expr: (w) => `(unsigned char)(${w[0]}f*r + ${w[1]}f*g + ${w[2]}f*b)` }, // i18n-ignore: code expression
];
const COLOURS: { label: string; rgb: [number, number, number] }[] = [
  { label: '(200, 100, 50)', rgb: [200, 100, 50] },
  { get label() { return tr('pure red', 'đỏ thuần'); }, rgb: [255, 0, 0] },
  { get label() { return tr('white', 'trắng'); }, rgb: [255, 255, 255] },
  { get label() { return tr('pure green', 'lục thuần'); }, rgb: [0, 255, 0] },
  { get label() { return tr('pure blue', 'lam thuần'); }, rgb: [0, 0, 255] },
];
const swatch = (css: string): CSSProperties => ({
  display: 'inline-block', width: '1.6rem', height: '1.1rem', border: '1px solid var(--rule)', borderRadius: 2, background: css, verticalAlign: 'middle', // i18n-ignore: CSS value
});

function PixelTester() {
  const [rgb, setRgb] = useState<[number, number, number]>([200, 100, 50]);
  const set = (k: number, v: number) => setRgb(rgb.map((x, i) => (i === k ? v : x)) as [number, number, number]);
  const res = VARIANTS.map((v) => ({ ...v, d: grayDetail(rgb[0], rgb[1], rgb[2], v.id) }));
  const right = res[2].d.byte;
  return (
    <Bench
      title={tr('What each version stores', 'Mỗi phiên bản lưu gì')}
      task={tr('Try pure red, then white. Which version gets which one wrong, and why?', 'Thử đỏ thuần, rồi trắng. Phiên bản nào sai ở màu nào, và vì sao?')}
      foot={tr('Sums are computed in single precision, one rounded operation at a time, left to right. The swatches show the input colour and each stored grey.', 'Các tổng được tính với độ chính xác đơn, mỗi lần một phép toán có làm tròn, từ trái sang phải. Các ô màu cho thấy màu đầu vào và từng mức xám được lưu.')}
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        {(['r', 'g', 'b'] as const).map((ch, k) => (
          <label key={ch} className="field">
            {ch} = {rgb[k]}
            <input type="range" min="0" max="255" value={rgb[k]} aria-label={ch} onChange={(e) => set(k, +e.target.value)} />
          </label>
        ))}
        <span className="seg" role="group" aria-label={tr('Colours', 'Màu')}>
          {COLOURS.map((c) => (
            <button key={c.label} aria-pressed={c.rgb.every((v, i) => v === rgb[i])} onClick={() => setRgb(c.rgb)}>{c.label}</button>
          ))}
        </span>
        <span><Translated vi="đầu vào">input</Translated> <i style={swatch(`rgb(${rgb.join(',')})`)} data-swatch="in" /></span>
      </div>
      <table className="data" style={{ margin: 0, background: 'var(--page)' }} aria-label={tr('Stored byte by version', 'Byte được lưu theo phiên bản')}>
        <thead>
          <tr><th style={{ paddingLeft: '0.6rem' }}><Translated vi="Phiên bản">Version</Translated></th><th><Translated vi="C++ tính gì">What C++ evaluates</Translated></th><th className="num"><Translated vi="Tổng float">Float sum</Translated></th><th className="num"><Translated vi="Byte được lưu">Byte stored</Translated></th><th /></tr>
        </thead>
        <tbody>
          {res.map((v) => (
            <tr key={v.id} data-variant={v.id}>
              <td style={{ paddingLeft: '0.6rem' }}>{v.name}</td>
              <td className="mono" style={{ overflowWrap: 'anywhere' }}>
                {v.id === 'printed'
                  ? <>0*{rgb[0]} + {GRAY_WEIGHTS[v.id][1]}f*{rgb[1]} + {GRAY_WEIGHTS[v.id][2]}f*{rgb[2]}</>
                  : <>{GRAY_WEIGHTS[v.id][0]}f*{rgb[0]} + {GRAY_WEIGHTS[v.id][1]}f*{rgb[1]} + {GRAY_WEIGHTS[v.id][2]}f*{rgb[2]}</>}
                <div className="muted">{v.expr(GRAY_WEIGHTS[v.id])}</div>
              </td>
              <td className="num">{fmtFloat(v.d.sum)}</td>
              <td className="num" data-byte={v.d.byte}>
                <b className={v.d.byte === right ? '' : 't-fail'}>{v.d.byte}</b>
              </td>
              <td><i style={swatch(`rgb(${v.d.byte},${v.d.byte},${v.d.byte})`)} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions

const L0 = layout(BOOK);
const SWAPPED = layout({ ...BOOK, swapped: true });
const BOOK_ONE = latticeOf(BOOK, 0, 1, 0, 0);
const T0 = threadAt(BOOK, BOOK_ONE.x, BOOK_ONE.y);

export const questionsS32: Record<string, Question> = {
  'ch03/3.2/q-swapped': {
    kind: 'numeric',
    id: 'ch03/3.2/q-swapped',
    prompt: (
      <Translated
        vi={
          <>
            Ảnh có n = 62 hàng và m = 76 cột. Ai đó viết lệnh khởi chạy thành{' '}
            <code>dim3 dimGrid(ceil(n/16.0), ceil(m/16.0), 1)</code> với các khối 16 × 16 và giữ nguyên kernel.
            Có bao nhiêu điểm ảnh không bao giờ được ghi?
          </>
        }
      >
        <>
          The picture has n = 62 rows and m = 76 columns. Someone writes the launch as{' '}
          <code>dim3 dimGrid(ceil(n/16.0), ceil(m/16.0), 1)</code> with 16 × 16 blocks and leaves the kernel unchanged.
          How many pixels are never written?
        </>
      </Translated>
    ),
    answer: 744,
    unit: 'pixels',
    explain: (
      <Translated
        vi={
          <>
            <p>
              Chiều x của lưới giờ lấy từ chiều cao: ceil(62/16) = 4 khối, 64 cột luồng. Chiều y lấy từ chiều rộng:
              5 khối, 80 hàng luồng. Cả 62 hàng đều được tới, nhưng các cột 64 đến 75 không có luồng nào:{' '}
              {BOOK.width - SWAPPED.coveredCols} cột × {BOOK.height} hàng = {SWAPPED.unreached} điểm ảnh giữ nguyên bất cứ
              thứ gì buffer đang chứa.
            </p>
            <p>
              Lệnh khởi chạy còn tạo ra một hàng 4 khối, các hàng luồng 64 đến 79, trong đó không luồng nào làm việc. Và
              trên một ảnh vuông, hai lệnh khởi chạy giống hệt nhau, nên một phép thử với đầu vào vuông không thể bắt được lỗi này.
            </p>
          </>
        }
      >
        <>
          <p>
            The grid's x extent now comes from the height: ceil(62/16) = 4 blocks, 64 thread columns. Its y extent comes
            from the width: 5 blocks, 80 thread rows. All 62 rows are reached, but columns 64 to 75 have no thread:{' '}
            {BOOK.width - SWAPPED.coveredCols} columns × {BOOK.height} rows = {SWAPPED.unreached} pixels keep whatever the
            buffer held.
          </p>
          <p>
            The launch also creates a row of 4 blocks, thread rows 64 to 79, in which no thread works. And on a square
            picture the two launches are identical, so a test on a square input cannot catch this.
          </p>
        </>
      </Translated>
    ),
    source: 'ch. 3 §3.2, pp. 49–50',
    vi: { unit: 'điểm ảnh' },
  },
  'ch03/3.2/q-ex3d': {
    kind: 'numeric',
    id: 'ch03/3.2/q-ex3d',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 3 của sách. Một kernel tính <code>row</code> và <code>col</code> như trên và bảo vệ câu lệnh duy nhất của nó
            bằng <code>if (row &lt; M &amp;&amp; col &lt; N)</code>. Host đặt M = 150, N = 300,{' '}
            <code>dim3 bd(16, 32)</code> và <code>dim3 gd((N - 1)/16 + 1, (M - 1)/32 + 1)</code>. Có bao nhiêu luồng thực thi
            câu lệnh được bảo vệ đó?
          </>
        }
      >
        <>
          Book exercise 3. A kernel computes <code>row</code> and <code>col</code> as above and guards its one statement
          with <code>if (row &lt; M &amp;&amp; col &lt; N)</code>. The host sets M = 150, N = 300,{' '}
          <code>dim3 bd(16, 32)</code> and <code>dim3 gd((N - 1)/16 + 1, (M - 1)/32 + 1)</code>. How many threads execute
          the guarded statement?
        </>
      </Translated>
    ),
    answer: 45000,
    unit: 'threads',
    explain: (
      <Translated
        vi={
          <p>
            Lưới là (19, 5): 95 khối, mỗi khối 16 × 32 = 512 luồng, tổng cộng 48,640, trải trên 304 cột và 160 hàng. Chỉ
            những luồng có row &lt; 150 và col &lt; 300 mới qua: 150 × 300 = 45,000, mỗi phần tử một luồng. 3,640 luồng còn lại
            dừng ở phép kiểm tra. Chọn cấu hình mẫu "bài tập 3" trong bản đồ luồng để thấy cùng những con số này.
          </p>
        }
      >
        <p>
          The grid is (19, 5): 95 blocks of 16 × 32 = 512 threads, 48,640 in all, spanning 304 columns and 160 rows. Only
          threads with row &lt; 150 and col &lt; 300 pass: 150 × 300 = 45,000, one per element. The other 3,640 stop at
          the guard. Choose the "exercise 3" preset in the thread map to see the same numbers.
        </p>
      </Translated>
    ),
    source: 'ch. 3 exercise 3, p. 65',
    vi: { unit: 'luồng', source: 'ch. 3 bài tập 3, p. 65' },
  },
  'ch03/3.2/q-ex4a': {
    kind: 'numeric',
    id: 'ch03/3.2/q-ex4a',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 4 của sách. Một ma trận rộng 400 và cao 500 được lưu trong một mảng phẳng theo <b>thứ tự hàng</b> (row-major). Chỉ số
            của phần tử ở hàng 20, cột 10 là bao nhiêu?
          </>
        }
      >
        <>
          Book exercise 4. A matrix 400 wide and 500 high is stored in a flat array in <b>row-major</b> order. What is the
          index of the element at row 20, column 10?
        </>
      </Translated>
    ),
    answer: 8010,
    explain: (
      <Translated vi={<p>Hai mươi hàng đầy đủ, mỗi hàng 400, đứng trước: 20·400 + 10 = 8010. Chiều cao không xuất hiện.</p>}>
        <p>Twenty full rows of 400 come first: 20·400 + 10 = 8010. The height does not appear.</p>
      </Translated>
    ),
    source: 'ch. 3 exercise 4, p. 65; §3.2, p. 51',
    vi: { source: 'ch. 3 bài tập 4, p. 65; §3.2, p. 51' },
  },
  'ch03/3.2/q-ex4b': {
    kind: 'numeric',
    id: 'ch03/3.2/q-ex4b',
    prompt: (
      <Translated
        vi={
          <>
            Cùng ma trận đó (rộng 400, cao 500), giờ được lưu theo <b>thứ tự cột</b> (column-major). Chỉ số của phần tử
            ở hàng 20, cột 10 là bao nhiêu?
          </>
        }
      >
        <>
          The same matrix (400 wide, 500 high), now stored in <b>column-major</b> order. What is the index of the element
          at row 20, column 10?
        </>
      </Translated>
    ),
    answer: 5020,
    explain: (
      <Translated
        vi={
          <p>
            Mười cột đầy đủ, mỗi cột 500, đứng trước: 10·500 + 20 = 5020. Giờ thì chiều rộng không xuất hiện. Đây là macro mà cuBLAS
            đưa cho người gọi từ C, <code>IDX2C(i,j,ld) = j*ld + i</code>, với chiều dẫn đầu (leading dimension) <code>ld</code> bằng
            số hàng.
          </p>
        }
      >
        <p>
          Ten full columns of 500 come first: 10·500 + 20 = 5020. Now the width does not appear. This is the macro cuBLAS
          gives C callers, <code>IDX2C(i,j,ld) = j*ld + i</code>, with the leading dimension <code>ld</code> equal to the
          number of rows.
        </p>
      </Translated>
    ),
    source: 'ch. 3 exercise 4, p. 65; §3.2, pp. 51–52; cuBLAS data layout (notes/today/ch03-facts.md, A7)',
    vi: { source: 'ch. 3 bài tập 4, p. 65; §3.2, pp. 51–52; bố cục dữ liệu của cuBLAS (notes/today/ch03-facts.md, A7)' },
  },
  'ch03/3.2/q-ex5-stride': {
    kind: 'numeric',
    id: 'ch03/3.2/q-ex5-stride',
    prompt: (
      <Translated
        vi={
          <>
            Một tensor liền kề <code>x</code> có shape (300, 500, 400): chiều sâu, chiều cao, chiều rộng. <code>x.stride()</code> là
            (200000, 400, 1). Offset phẳng của <code>x[5, 20, 10]</code> là bao nhiêu?
          </>
        }
      >
        <>
          A contiguous tensor <code>x</code> has shape (300, 500, 400): depth, height, width. <code>x.stride()</code> is
          (200000, 400, 1). What is the flat offset of <code>x[5, 20, 10]</code>?
        </>
      </Translated>
    ),
    answer: 1008010,
    explain: (
      <Translated
        vi={
          <p>
            Chỉ số nhân bước nhảy, cộng lại: 5·200,000 + 20·400 + 10·1 = 1,008,010. Đó là công thức{' '}
            <code>plane*m*n + row*m + col</code> của sách với m = 400 cột và n = 500 hàng, và đó là bài tập 5 của sách (x = 10,
            y = 20, z = 5) nói theo cách của PyTorch.
          </p>
        }
      >
        <p>
          Index times stride, summed: 5·200,000 + 20·400 + 10·1 = 1,008,010. It is the book's{' '}
          <code>plane*m*n + row*m + col</code> with m = 400 columns and n = 500 rows, and it is book exercise 5 (x = 10,
          y = 20, z = 5) in PyTorch's words.
        </p>
      </Translated>
    ),
    source: 'ch. 3 §3.2, p. 55; exercise 5, p. 65; torch.Tensor.stride (notes/today/ch03-facts.md, D6)',
    vi: { source: 'ch. 3 §3.2, p. 55; bài tập 5, p. 65; torch.Tensor.stride (notes/today/ch03-facts.md, D6)' },
  },
  'ch03/3.2/q-precedence': {
    kind: 'choice',
    id: 'ch03/3.2/q-precedence',
    prompt: (
      <Translated
        vi={
          <>
            Tìm lỗi. Dòng này biên dịch không có cảnh báo, với <code>r</code>, <code>g</code>, <code>b</code> có
            kiểu <code>unsigned char</code>. Nó lưu gì?
            <br />
            <code>Pout[grayOffset] = (unsigned char) 0.21f*r + 0.71f*g + 0.07f*b;</code>
          </>
        }
      >
        <>
          Find the defect. This line compiles without a warning, with <code>r</code>, <code>g</code>, <code>b</code> of
          type <code>unsigned char</code>. What does it store?
          <br />
          <code>Pout[grayOffset] = (unsigned char) 0.21f*r + 0.71f*g + 0.07f*b;</code>
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Độ sáng mong muốn, 0.21·r + 0.71·g + 0.07·b, bị cắt phần lẻ.">The intended luminance, 0.21·r + 0.71·g + 0.07·b, truncated.</Translated>,
        why: <Translated vi="Điều đó cần cả tổng nằm trong cặp ngoặc của phép ép kiểu.">That needs the whole sum inside the parentheses of the cast.</Translated>,
      },
      { label: <Translated vi="Giá trị bị cắt phần lẻ của 0.71·g + 0.07·b. Màu đỏ không đóng góp gì.">The truncated value of 0.71·g + 0.07·b. Red contributes nothing.</Translated>, correct: true },
      {
        label: <Translated vi="0 cho mọi điểm ảnh, bất kể màu gì.">0 for every pixel, whatever its colour.</Translated>,
        why: <Translated vi="Chỉ tích đầu tiên bằng 0. Hai tích còn lại được tính bằng float và vẫn còn.">Only the first product is 0. The other two are computed in float and survive.</Translated>,
      },
      {
        label: <Translated vi="Một giá trị có thể vượt 255 và quay vòng.">A value that can exceed 255 and wrap around.</Translated>,
        why: <Translated vi="Hai trọng số còn lại cộng lại là 0.78, nên giá trị luôn dưới 199.">The two remaining weights sum to 0.78, so the value stays below 199.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Phép ép kiểu chỉ gắn với <code>0.21f</code> và biến nó thành số nguyên 0, nên số hạng đầu là 0·r. Tổng của
            hai số hạng còn lại là float, và phép gán vào một <code>unsigned char</code> cắt bỏ phần lẻ. Đỏ thuần được lưu
            thành 0 thay vì 53 như dự định; (200, 100, 50) được lưu thành 74 thay vì 116.
          </p>
        }
      >
        <p>
          The cast binds to <code>0.21f</code> alone and turns it into the integer 0, so the first term is 0·r. The sum of
          the other two is a float, and the assignment to an <code>unsigned char</code> truncates it. Pure red is stored
          as 0 where 53 was intended; (200, 100, 50) is stored as 74 instead of 116.
        </p>
      </Translated>
    ),
    source: 'ch. 3 Fig. 3.4 line 19, p. 53; C++ grammar and a host compile (notes/today/ch03-facts.md, A9)',
    vi: { source: 'ch. 3 Fig. 3.4 dòng 19, p. 53; ngữ pháp C++ và một lần biên dịch trên host (notes/today/ch03-facts.md, A9)' },
  },
  'ch03/3.2/q-uchar3': {
    kind: 'choice',
    id: 'ch03/3.2/q-uchar3',
    prompt: (
      <Translated
        vi={
          <>
            Một bản chuyển sang Metal gắn ảnh màu, ba byte mỗi điểm ảnh, dưới dạng <code>device const uchar3* Pin</code> và đọc
            một điểm ảnh bằng <code>Pin[grayOffset]</code>. Điều gì xảy ra?
          </>
        }
      >
        <>
          A Metal port binds the colour picture, three bytes per pixel, as <code>device const uchar3* Pin</code> and reads
          a pixel as <code>Pin[grayOffset]</code>. What happens?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Nó chạy đúng: một uchar3 là một điểm ảnh ba byte, nên Pin[i] là điểm ảnh i.">It works: one uchar3 is one three-byte pixel, so Pin[i] is pixel i.</Translated>,
        why: <Translated vi="Trong Metal Shading Language, uchar3 có kích thước 4 và căn lề 4.">In Metal Shading Language uchar3 has size 4 and alignment 4.</Translated>,
      },
      { label: <Translated vi="Mỗi phần tử là 4 byte, nên từ điểm ảnh thứ hai trở đi nó đọc sai byte.">Each element is 4 bytes, so from the second pixel on it reads the wrong bytes.</Translated>, correct: true },
      {
        label: <Translated vi="Nó không biên dịch: một buffer device không thể chứa kiểu vector.">It does not compile: a device buffer cannot hold a vector type.</Translated>,
        why: <Translated vi="Buffer có thể chứa kiểu vector. Vấn đề nằm ở kích thước của chúng.">Buffers may hold vector types. The trouble is their size.</Translated>,
      },
      {
        label: <Translated vi="Nó đọc đúng các byte nhưng theo thứ tự b, g, r.">It reads the right bytes in the order b, g, r.</Translated>,
        why: <Translated vi="Không có gì bị đảo thứ tự. Bước giữa các phần tử mới là chỗ sai.">Nothing is reordered. The step between elements is wrong.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>Pin[i]</code> trên một <code>uchar3*</code> nằm ở byte 4·i, còn dữ liệu có một điểm ảnh cứ mỗi 3 byte. Điểm ảnh 0
            đúng, điểm ảnh 1 bắt đầu trễ một byte, và từ đó càng lúc càng tệ hơn. Hãy giữ phép tính của sách trên một{' '}
            <code>uchar*</code> (<code>rgbOffset = grayOffset*3</code>), hoặc khai báo buffer là{' '}
            <code>packed_uchar3</code>, có kích thước 3 byte với căn lề 1.
          </p>
        }
      >
        <p>
          <code>Pin[i]</code> on a <code>uchar3*</code> is at byte 4·i, and the data has a pixel every 3 bytes. Pixel 0 is
          right, pixel 1 starts one byte late, and it gets worse from there. Keep the book's arithmetic on a{' '}
          <code>uchar*</code> (<code>rgbOffset = grayOffset*3</code>), or declare the buffer{' '}
          <code>packed_uchar3</code>, which is 3 bytes with alignment 1.
        </p>
      </Translated>
    ),
    source: 'ours; Metal Shading Language specification, tables 2.3 and 2.4 (notes/today/ch03-facts.md, C5)',
    vi: { source: 'của khóa học này; đặc tả Metal Shading Language, bảng 2.3 và 2.4 (notes/today/ch03-facts.md, C5)' },
  },
  'ch03/3.2/q-noguard': {
    kind: 'choice',
    id: 'ch03/3.2/q-noguard',
    prompt: (
      <Translated
        vi={
          <>
            Cùng một thân kernel Metal, dùng <code>thread_position_in_grid</code>, được khởi chạy trên ảnh 62 × 76
            hai lần: một lần với <code>dispatchThreads</code> trên lưới (76, 62), và một lần với{' '}
            <code>dispatchThreadgroups</code> trên (5, 4) nhóm 16 × 16. Lần khởi chạy nào cần phép kiểm tra phạm vi?
          </>
        }
      >
        <>
          The same Metal kernel body, using <code>thread_position_in_grid</code>, is launched over the 62 × 76 picture
          twice: once with <code>dispatchThreads</code> over a grid of (76, 62), and once with{' '}
          <code>dispatchThreadgroups</code> over (5, 4) groups of 16 × 16. Which launch needs the range test?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Chỉ dispatchThreads.">Only dispatchThreads.</Translated>,
        why: <Translated vi="Lời gọi đó tạo đúng số luồng bạn yêu cầu.">That call creates exactly the threads you ask for.</Translated>,
      },
      { label: <Translated vi="Chỉ dispatchThreadgroups.">Only dispatchThreadgroups.</Translated>, correct: true },
      {
        label: <Translated vi="Cả hai.">Both.</Translated>,
        why: <Translated vi="Một trong hai tạo đúng 76 × 62 luồng.">One of them creates exactly 76 × 62 threads.</Translated>,
      },
      {
        label: <Translated vi="Không cái nào: Metal bỏ qua các lần ghi ra ngoài buffer.">Neither: Metal discards writes outside a buffer.</Translated>,
        why: (
          <Translated vi="Không trang nào chúng tôi đã kiểm tra nói vậy. Apple bảo bạn thoát sớm khi lưới lớn hơn dữ liệu.">
            Nothing in the pages we checked says so. Apple tells you to exit early when the grid is larger than the data.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>dispatchThreadgroups</code> tạo 5 × 4 nhóm trọn vẹn: {L0.threads} luồng cho {L0.pixels} điểm ảnh, nên{' '}
            {L0.surplus} luồng nằm ngoài ảnh và phải bị chặn lại. <code>dispatchThreads</code> trên (76, 62)
            tạo đúng {L0.pixels} luồng và làm các nhóm ở mép nhỏ hơn. Tài liệu của Apple nói không cần
            kiểm tra vị trí khi dùng nó. Lưu ý thứ tự: kích thước trong Metal ghi chiều rộng trước.
          </p>
        }
      >
        <p>
          <code>dispatchThreadgroups</code> creates 5 × 4 whole groups: {L0.threads} threads for {L0.pixels} pixels, so{' '}
          {L0.surplus} threads are outside the picture and must be stopped. <code>dispatchThreads</code> over (76, 62)
          creates exactly {L0.pixels} threads and makes the groups on the edges smaller. Apple's documentation says no
          position check is needed with it. Note the order: Metal sizes are width first.
        </p>
      </Translated>
    ),
    source: 'ours; Apple, "Calculating threadgroup and grid sizes" (notes/today/ch03-facts.md, C3)',
    vi: { source: 'của khóa học này; Apple, "Calculating threadgroup and grid sizes" (notes/today/ch03-facts.md, C3)' },
  },
};

// ---------------------------------------------------------------- the section

export function S32() {
  const [launch, setLaunch] = useState<Launch>(BOOK);
  const [selRaw, setSel] = useState<XY>({ x: 37, y: 21 });
  const sel = clampSel(launch, selRaw);
  const { progress } = useProgress();
  const predictedRed = typeof progress.answers['ch03/3.2/p-red']?.value === 'number';
  const showBookThread = () => {
    setLaunch(BOOK);
    setSel(BOOK_ONE);
    document.querySelector(tr('section.bench[aria-label="The thread map"]', 'section.bench[aria-label="Bản đồ luồng"]'))?.scrollIntoView({ block: 'start' });
  };

  return (
    <>
      <Translated
        vi={
          <p>
            Mục 3.1 đã cho mỗi luồng tọa độ của nó: chỉ số khối của nó và chỉ số riêng của nó trong khối, mỗi cái theo
            x và y. Mục này biến chúng thành hai thứ một kernel cần: luồng sở hữu phần tử nào của dữ liệu,
            và phần tử đó nằm ở đâu trong bộ nhớ. Cái đầu cần hai phép nhân-cộng, cái sau thêm một phép nữa. Gần như mọi
            kernel bạn sẽ viết cho suy luận đều mở đầu bằng những dòng này.
          </p>
        }
      >
        <p>
          Section 3.1 gave every thread its coordinates: its block's index and its own index within the block, each along
          x and y. This section turns them into the two things a kernel needs, which element of the data the thread owns
          and where that element is in memory. The first takes two multiply-adds, the second one more. Nearly every
          kernel you will write for inference opens with these lines.
        </p>
      </Translated>
      <Source section="3.2" pages="48–55" />

      <h2><Translated vi="Mỗi luồng một điểm ảnh">One thread per pixel</Translated></h2>
      <Translated
        vi={
          <p>
            Ví dụ của sách là một ảnh 62 hàng và 76 cột, được chuyển từ màu sang xám bởi mỗi điểm ảnh một
            luồng. Sách viết kích thước với chiều dọc trước, nên "62 × 76" là 62 theo chiều dọc và 76 theo chiều ngang, đúng thứ tự
            các chỉ số trong <code>P[y][x]</code>. Kích thước khởi chạy thì ngược lại, x trước. Chính sách gọi đây là một
            nguồn gây nhầm lẫn, và một trong các câu hỏi bên dưới được dựng trên điều này.
          </p>
        }
      >
        <p>
          The book's example is a picture of 62 rows and 76 columns, converted from colour to grey by one thread per
          pixel. The book writes sizes with the vertical dimension first, so "62 × 76" is 62 down and 76 across, the order
          of the subscripts in <code>P[y][x]</code>. Launch sizes go the other way, x first. The book itself calls this a
          source of confusion, and one of the questions below is built on it.
        </p>
      </Translated>
      <Source section="3.2" pages="48–49" figs={tr('Fig. 3.2 and footnote 1', 'Fig. 3.2 và chú thích 1')} />
      <Translated
        vi={
          <p>
            Mỗi khối là 16 × 16 luồng. Phủ 76 cột cần 5 khối theo x và phủ 62 hàng cần 4 khối theo y: 20
            khối. Trước khi xem điều đó diễn ra, hãy chốt hai con số.
          </p>
        }
      >
        <p>
          Blocks are 16 × 16 threads. Covering 76 columns takes 5 blocks along x and covering 62 rows takes 4 along y: 20
          blocks. Before you watch that happen, commit to two numbers.
        </p>
      </Translated>
      <Predict
        id="ch03/3.2/p-surplus"
        prompt={
          <Translated vi={<>Lệnh khởi chạy tạo 20 khối 256 luồng cho một ảnh 62 × 76 điểm ảnh. Có bao nhiêu luồng không có điểm ảnh?</>}>
            <>The launch creates 20 blocks of 256 threads for a picture of 62 × 76 pixels. How many threads have no pixel?</>
          </Translated>
        }
        unit={tr('threads', 'luồng')}
      >
        {(v) => (
          <Translated
            vi={
              <>
                <p>
                  {L0.threads} luồng cho {L0.pixels} điểm ảnh còn dư <b>{L0.surplus}</b>.{' '}
                  {v === L0.surplus ? 'Đúng như bạn dự đoán.' : `Bạn lệch ${Math.abs(v - L0.surplus)}.`}
                </p>
                <p>
                  Chúng không nằm rải rác: tất cả đều ở {L0.threadsX - BOOK.width} cột luồng cuối hoặc
                  {' '}{L0.threadsY - BOOK.height} hàng luồng cuối.
                </p>
              </>
            }
          >
            <>
              <p>
                {L0.threads} threads for {L0.pixels} pixels leaves <b>{L0.surplus}</b>.{' '}
                {v === L0.surplus ? 'That is what you predicted.' : `You were off by ${Math.abs(v - L0.surplus)}.`}
              </p>
              <p>
                They are not scattered: every one of them is in the last {L0.threadsX - BOOK.width} thread columns or the
                last {L0.threadsY - BOOK.height} thread rows.
              </p>
            </>
          </Translated>
        )}
      </Predict>
      <Predict
        id="ch03/3.2/p-offsets"
        prompt={
          <Translated
            vi={
              <>
                Đầu ra xám là một mảng phẳng, một byte mỗi điểm ảnh, với các hàng của ảnh được lưu nối tiếp nhau.
                Nhãn của mục 3.1 là (y, x). Luồng (0,0) của Khối (1,0) ghi vào phần tử nào của mảng đó?
              </>
            }
          >
            <>
              The grey output is one flat array, one byte per pixel, with the picture's rows stored one after another.
              Section 3.1's labels are (y, x). Which element of that array does Thread (0,0) of Block (1,0) write?
            </>
          </Translated>
        }
      >
        {(v) => (
          <Translated
            vi={
              <>
                <p>
                  Khối (1,0) ở dưới một khối và không lệch ngang khối nào, nên luồng nằm ở hàng {T0.blockIdx.y}·{BOOK.blockH} + 0 ={' '}
                  {T0.row}, cột {T0.col}. {T0.row} hàng đầy đủ, mỗi hàng {BOOK.width}, đứng trước nó: phần tử{' '}
                  <b>{T0.grayOffset}</b>. {v === T0.grayOffset ? 'Đúng như bạn dự đoán.' : `Bạn đã trả lời ${v}.`}
                </p>
                <p>
                  Giờ hãy nhẩm xem byte đỏ của điểm ảnh đó nằm ở đâu trong đầu vào màu, nơi chứa ba byte
                  mỗi điểm ảnh. Phần 5 của hoạt hình đi tới đó, và bản đồ luồng cho thấy cả hai con số.
                </p>
                <p>
                  <button onClick={showBookThread}>Chọn luồng này trên bản đồ</button>
                </p>
              </>
            }
          >
            <>
              <p>
                Block (1,0) is one block down and none across, so the thread is in row {T0.blockIdx.y}·{BOOK.blockH} + 0 ={' '}
                {T0.row}, column {T0.col}. {T0.row} full rows of {BOOK.width} come before it: element{' '}
                <b>{T0.grayOffset}</b>. {v === T0.grayOffset ? 'That is what you predicted.' : `You said ${v}.`}
              </p>
              <p>
                Now work out, in your head, where that pixel's red byte is in the colour input, which holds three bytes
                per pixel. Part 5 of the animation gets there, and the thread map shows both numbers.
              </p>
              <p>
                <button onClick={showBookThread}>Select this thread in the map</button>
              </p>
            </>
          </Translated>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Hoạt hình đặt lần khởi chạy của sách lên ảnh từng khối một, theo một luồng tới điểm ảnh của nó và vào
            đầu ra phẳng, rồi kết thúc bằng việc xếp các khối thành từng loại.
          </p>
        }
      >
        <p>
          The animation lays the book's launch over the picture block by block, follows one thread to its pixel and into
          the flat output, and ends by sorting the blocks into kinds.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Hoạt hình chỉ cho thấy quyền sở hữu: luồng nào nhận điểm ảnh nào, và điểm ảnh đó được lưu ở đâu. Các khối
              thực ra không đến lần lượt từng cái theo thứ tự, và không có gì trong đó nói về tốc độ.
            </p>
          }
        >
          <p>
            The animation shows ownership only: which thread gets which pixel, and where that pixel is stored. Blocks do
            not really arrive one at a time in order, and nothing in it is about speed.
          </p>
        </Translated>
      </Simplification>
      <GridOverPictureScene />

      <h2><Translated vi="Phép ánh xạ mà mọi kernel về sau dùng lại">The mapping every later kernel reuses</Translated></h2>
      <Translated
        vi={
          <p>
            Hàng và cột của một luồng trong dữ liệu là vị trí của nó trong toàn bộ lưới luồng:{' '}
            <code>row = blockIdx.y*blockDim.y + threadIdx.y</code> và <code>col = blockIdx.x*blockDim.x + threadIdx.x</code>.
            Đọc mỗi công thức là: số khối đứng trước khối của tôi, nhân số luồng trong một khối, cộng vị trí của tôi trong khối của mình. Các trường y
            cho hàng và các trường x cho cột. Với Luồng (0,0) của Khối (1,0), đó là hàng 1·16 + 0 = 16 và
            cột 0·16 + 0 = 0, nên nó xử lý điểm ảnh (16, 0).
          </p>
        }
      >
        <p>
          A thread's row and column in the data are its position in the whole grid of threads:{' '}
          <code>row = blockIdx.y*blockDim.y + threadIdx.y</code> and <code>col = blockIdx.x*blockDim.x + threadIdx.x</code>.
          Read each as: the blocks before mine, times the threads in a block, plus my place in my own block. The y fields
          give the row and the x fields give the column. For Thread (0,0) of Block (1,0) that is row 1·16 + 0 = 16 and
          column 0·16 + 0 = 0, so it handles pixel (16, 0).
        </p>
      </Translated>
      <Source section="3.2" pages="49–50" />
      <Translated
        vi={
          <p>
            Bản đồ bên dưới chính là lần khởi chạy đó. Mỗi ô nhỏ là một luồng và các đường đậm là biên khối. Chọn một
            luồng và bảng bên cạnh sẽ thay các con số của nó vào công thức, cho tới tận các byte nó sẽ chạm vào. Tìm xem
            mỗi con số trong công thức đến từ đâu. Rồi tìm những khối nào chứa luồng không có điểm ảnh, và có bao nhiêu loại
            khối.
          </p>
        }
      >
        <p>
          The map below is that launch. Each small square is a thread and the heavy lines are block borders. Select a
          thread and the panel substitutes its numbers into the formulas, down to the bytes it would touch. Find where
          each number in a formula comes from. Then find which blocks contain threads with no pixel, and how many kinds
          of block there are.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Bản đồ chỉ cho thấy luồng nào sở hữu điểm ảnh nào, ngoài ra không gì khác: không có thời gian, không có warp, không có chi phí cho một luồng
              không làm gì. Các nhóm nhỏ hơn ở mép của kiểu <code>dispatchThreads</code> trong Metal theo mô tả của
              Apple và chưa được đo trên thiết bị.
            </p>
          }
        >
          <p>
            The map shows which thread owns which pixel and nothing else: no timing, no warps, no cost for a thread that
            does nothing. The smaller edge groups of the Metal <code>dispatchThreads</code> style follow Apple's
            description and were not measured on a device.
          </p>
        </Translated>
      </Simplification>
      <ThreadMap launch={launch} setLaunch={setLaunch} sel={sel} setSel={setSel} />

      <h2><Translated vi="Luồng dư và phép kiểm tra biên">Surplus threads and the guard</Translated></h2>
      <Translated
        vi={
          <p>
            Một lưới gồm một số nguyên các khối và mọi khối có cùng hình dạng, nên lần khởi chạy làm tròn lên: 5 × 16 = 80
            cột luồng cho 76 cột điểm ảnh, 4 × 16 = 64 hàng luồng cho 62 hàng điểm ảnh. {L0.surplus} luồng
            nằm ngoài ảnh không được làm gì, nên kernel kiểm tra cả hai chỉ số trước khi chạm vào bộ nhớ:{' '}
            <code>{'if (col < width && row < height)'}</code>. Đó là dạng hai chiều của phép kiểm tra trong phép cộng vector
            ở chương 2, nơi bốn khối 256 luồng để lại 24 luồng không có phần tử nào của một vector 1000 phần tử.
          </p>
        }
      >
        <p>
          A grid is a whole number of blocks and every block has the same shape, so the launch rounds up: 5 × 16 = 80
          thread columns for 76 pixel columns, 4 × 16 = 64 thread rows for 62 pixel rows. The {L0.surplus} threads
          outside the picture must do nothing, so the kernel tests both indices before it touches memory:{' '}
          <code>{'if (col < width && row < height)'}</code>. It is the two-dimensional form of the test in chapter 2's
          vector addition, where four blocks of 256 threads left 24 with no element of a 1000-element vector.
        </p>
      </Translated>
      <Source section="3.2" pages="50" />
      <Translated
        vi={
          <p>
            Cùng phép kiểm tra đó hoàn tất lập luận rằng mọi điểm ảnh đều được xử lý. Theo x có{' '}
            <code>blockDim.x*gridDim.x</code> luồng, và <code>col</code> nhận mọi số nguyên từ 0 đến nhỏ hơn số đó
            một đơn vị. Nếu số đó ít nhất bằng chiều rộng, và điều tương tự đúng theo y, thì mọi điểm ảnh đều có một luồng. Phép kiểm tra
            loại bỏ các luồng vượt ra ngoài ảnh, và không có hai luồng nào chung một điểm ảnh vì không có hai luồng nào chung một (row, col).
          </p>
        }
      >
        <p>
          The same test completes the argument that every pixel is handled. Along x there are{' '}
          <code>blockDim.x*gridDim.x</code> threads, and <code>col</code> takes every integer from 0 up to one less than
          that. If that count is at least the width, and the same holds along y, every pixel has a thread. The guard
          removes the threads beyond the picture, and no two threads share a pixel because no two share a (row, col).
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, A5)" /* i18n-ignore: date and citation */>
        <Translated
          vi={
            <p>
              Hướng dẫn lập trình của NVIDIA ghi lại cùng mẫu này: làm tròn lưới lên và để các luồng ngoài phạm vi thoát ra. Nó
              thêm một quy tắc mà sách không nêu: luồng rỗi bên trong một khối tốn ít, nhưng nên tránh những khối mà không luồng nào
              làm việc. Một lưới được làm tròn lên từ đúng các chiều không bao giờ chứa khối như vậy.
            </p>
          }
        >
          <p>
            NVIDIA's programming guide documents the same pattern: round the grid up and let out-of-range threads exit. It
            adds a rule the book does not state: idle threads inside a block cost little, but blocks in which no thread
            works should be avoided. A grid rounded up from the right dimensions never contains one.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Lần khởi chạy phải tính chiều x của lưới từ chiều rộng và chiều y từ chiều cao. Với n hàng và m
            cột, sách viết <code>dim3 dimGrid(ceil(m/16.0), ceil(n/16.0), 1)</code>. Không có gì cảnh báo bạn nếu hai chiều
            bị đặt ngược. Đánh dấu "đổi chỗ hai chiều của lưới" trên bản đồ và xem điều gì thay đổi, rồi trả lời.
          </p>
        }
      >
        <p>
          The launch has to size the grid's x extent from the width and its y extent from the height. With n rows and m
          columns the book writes <code>dim3 dimGrid(ceil(m/16.0), ceil(n/16.0), 1)</code>. Nothing warns you if the two
          are the wrong way round. Tick "swap the grid extents" in the map and look at what changes, then answer.
        </p>
      </Translated>
      <Source section="3.2" pages="50, 52–53" />
      <Ask q={questionsS32['ch03/3.2/q-swapped']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS32['ch03/3.2/q-ex3d']} label={tr('Book exercise 3d', 'Bài tập 3d của sách')} />

      <h2><Translated vi="Vì sao làm phẳng, và thứ tự hàng (row-major) nghĩa là gì">Why flatten, and what row-major means</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn sẽ muốn viết <code>Pin_d[row][col]</code>. C++ chỉ đánh chỉ số mảng hai chiều theo cách đó khi
            nó biết số cột lúc biên dịch, và với một buffer được cấp phát lúc chạy thì nó không biết. Vì vậy
            buffer là một chiều và bạn tự tính offset. Không mất gì cả. Bộ nhớ là một dãy phẳng các
            byte có địa chỉ, nên mọi mảng đa chiều đều được làm phẳng ở đâu đó: với mảng tĩnh thì trình biên dịch làm
            việc đó, với mảng động thì bạn làm.
          </p>
        }
      >
        <p>
          You would like to write <code>Pin_d[row][col]</code>. C++ can index a two-dimensional array that way only when
          it knows the number of columns at compile time, and for a buffer allocated at run time it does not. So the
          buffer is one-dimensional and you compute the offset. Nothing is lost by this. Memory is a flat sequence of
          addressed bytes, so every multidimensional array is flattened somewhere: for a static array the compiler does
          it, for a dynamic one you do.
        </p>
      </Translated>
      <Source section="3.2" pages="50–51" figs={tr('sidebar "Memory Space"', 'khung "Memory Space"')} />
      <Translated
        vi={
          <p>
            C và C++ làm phẳng theo <b>thứ tự hàng</b> (row-major): toàn bộ hàng 0, rồi toàn bộ hàng 1, và cứ thế. Phần tử (j, i) của một
            mảng rộng <code>Width</code> cột nằm ở <code>j*Width + i</code>, trong đó số hạng đầu bỏ qua j hàng
            phía trên. Trong ma trận 4 × 4, M[2][1] là phần tử 2·4 + 1 = 9. Quy ước còn lại, <b>thứ tự cột</b> (column-major), lưu
            trọn từng cột nối tiếp nhau, và FORTRAN dùng nó. Lưu một ma trận theo thứ tự cột chính là lưu
            ma trận chuyển vị của nó theo thứ tự hàng, đó là lý do các thư viện viết cho người gọi từ FORTRAN bảo người gọi từ C chuyển vị đầu vào của mình.
          </p>
        }
      >
        <p>
          C and C++ flatten in <b>row-major</b> order: all of row 0, then all of row 1, and so on. Element (j, i) of an
          array <code>Width</code> columns wide is at <code>j*Width + i</code>, where the first term skips the j rows
          above. In a 4 × 4 matrix, M[2][1] is element 2·4 + 1 = 9. The other convention, <b>column-major</b>, stores
          whole columns one after another, and FORTRAN uses it. Column-major storage of a matrix is the row-major storage
          of its transpose, which is why libraries written for FORTRAN callers tell C callers to transpose their inputs.
        </p>
      </Translated>
      <Source section="3.2" pages="51–52" figs="Fig. 3.3" />
      <Translated
        vi={
          <p>
            Hãy kiểm tra ba điều trong trình khám phá. Một bước dọc theo một chiều luôn là cùng một bước nhảy trong mảng phẳng,
            dù bạn bắt đầu ở đâu. Thứ tự cột đúng là ma trận chuyển vị đọc theo từng hàng. Và lỗi kinh điển, dùng
            chiều cao ở chỗ của chiều rộng, không thể thấy được trên một mảng vuông.
          </p>
        }
      >
        <p>
          Check three things in the explorer. A step along one dimension is always the same jump in the flat array,
          wherever you start. Column-major really is the transpose read row by row. And the classic defect, using the
          height where the width belongs, cannot be seen on a square array.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Tối đa 144 phần tử, mỗi phần tử vẽ thành một ô. Offset thật cũng tính bằng phần tử; trình biên dịch nhân với
              kích thước phần tử khi biến <code>P[i]</code> thành một địa chỉ.
            </p>
          }
        >
          <p>
            At most 144 elements, each drawn as one slot. Real offsets are in elements too; the compiler multiplies by the
            element size when it turns <code>P[i]</code> into an address.
          </p>
        </Translated>
      </Simplification>
      <FlatteningExplorer />
      <Translated
        vi={
          <p>
            Sách nói lập trình viên "cần" tự tuyến tính hóa bằng tay trong CUDA C++ hiện nay. Câu đó đã cũ. Thói quen mà nó
            mô tả thì chưa: offset viết tay là thứ bạn sẽ gặp trong mọi backend bạn đọc.
          </p>
        }
      >
        <p>
          The book says programmers "need to" linearise by hand in current CUDA C++. That sentence has aged. The habit it
          describes has not: hand-written offsets are what you will find in every backend you read.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, A6)" /* i18n-ignore: date and citation */>
        <Translated
          vi={
            <p>
              nvcc giờ chấp nhận C++ tới C++23, và libcu++ của NVIDIA có <code>cuda::std::mdspan</code>, nhờ đó một
              kernel có thể viết <code>m(row, col)</code> trên một buffer phẳng (<code>operator()</code>, vì{' '}
              <code>operator[]</code> bị tắt). Chưa kiểm chứng: bản toolkit nào đưa nó vào đầu tiên. Chính hướng dẫn của NVIDIA,
              llama.cpp, MLX và PyTorch đều vẫn làm phẳng bằng tay, và các lab ở đây cũng vậy.
            </p>
          }
        >
          <p>
            nvcc now accepts C++ up to C++23, and NVIDIA's libcu++ has <code>cuda::std::mdspan</code>, with which a
            kernel can write <code>m(row, col)</code> over a flat buffer (<code>operator()</code>, because{' '}
            <code>operator[]</code> is disabled). Unverified: which toolkit release first shipped it. NVIDIA's own guide,
            llama.cpp, MLX and PyTorch all still flatten by hand, and so do the labs here.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS32['ch03/3.2/q-ex4a']} label={tr('Book exercise 4a', 'Bài tập 4a của sách')} />
      <Ask q={questionsS32['ch03/3.2/q-ex4b']} label={tr('Book exercise 4b', 'Bài tập 4b của sách')} />
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, A7, B6, C7)" /* i18n-ignore: date and citation */>
        <Translated
          vi={
            <p>
              Lời cảnh báo về thứ tự cột áp dụng cho chính BLAS của các hãng. cuBLAS và rocBLAS lưu ma trận theo thứ tự cột,
              và tài liệu của cuBLAS đưa cho người gọi từ C macro <code>IDX2C(i,j,ld) = j*ld + i</code>. Phía Apple,{' '}
              <code>MPSMatrix</code> theo thứ tự hàng, còn các kiểu nhỏ <code>floatNxM</code> của ngôn ngữ shading thì theo
              thứ tự cột.
            </p>
          }
        >
          <p>
            The column-major warning applies to the vendors' own BLAS. cuBLAS and rocBLAS store matrices column-major,
            and cuBLAS's documentation gives C callers the macro <code>IDX2C(i,j,ld) = j*ld + i</code>. On Apple's side{' '}
            <code>MPSMatrix</code> is row-major, while the shading language's small <code>floatNxM</code> types are
            column-major.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Điều stride() vẫn luôn nói với bạn">What stride() has been telling you</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn đã dùng bố cục này nhiều năm. Một tensor PyTorch là một buffer phẳng cộng một bộ các bước nhảy (stride), và{' '}
            <code>x.stride()</code> là bước nhảy trong buffer đó cho một bước dọc theo mỗi chiều. Một tensor liền kề có
            shape (2, 5) có stride (5, 1): một bước xuống là 5 phần tử và một bước ngang là 1, tức là{' '}
            <code>row*Width + col</code> với chiều rộng 5. Tên PyTorch đặt cho bố cục này, <code>contiguous_format</code>,
            được định nghĩa bằng các bước nhảy theo thứ tự giảm dần. Đánh dấu "hiện các bước nhảy" trong trình khám phá ở trên là bạn đang nhìn thấy
            bộ số đó.
          </p>
        }
      >
        <p>
          You have used this layout for years. A PyTorch tensor is a flat buffer plus a tuple of strides, and{' '}
          <code>x.stride()</code> is the jump in that buffer for one step along each dimension. A contiguous tensor of
          shape (2, 5) has stride (5, 1): one step down is 5 elements and one step across is 1, which is{' '}
          <code>row*Width + col</code> with a width of 5. PyTorch's name for the layout, <code>contiguous_format</code>,
          is defined by strides in decreasing order. Tick "show the strides" in the explorer above and you are looking at
          that tuple.
        </p>
      </Translated>
      <Beyond title={tr('Bridge: PyTorch', 'Cầu nối: PyTorch')}>
        <Translated
          vi={
            <p>
              Của khóa học này, không phải của sách. Lấy từ tài liệu PyTorch 2.15 về <code>Tensor.stride</code> và{' '}
              <code>torch.memory_format</code>, đọc ngày 2026-11-20 (notes/today/ch03-facts.md, D6).
            </p>
          }
        >
          <p>
            Ours, not the book's. From the PyTorch 2.15 documentation of <code>Tensor.stride</code> and{' '}
            <code>torch.memory_format</code>, read on 2026-11-20 (notes/today/ch03-facts.md, D6).
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Kernel chuyển ảnh xám">The grayscale kernel</Translated></h2>
      <Translated
        vi={
          <p>
            Giờ là toàn bộ kernel. Một điểm ảnh màu là ba byte liên tiếp, đỏ, lục và lam, còn một điểm ảnh xám là một
            byte. Luồng tính offset xám của nó như trên, rồi nhân ba để tìm điểm ảnh của nó trong đầu vào
            màu, vì mỗi điểm ảnh trước nó chiếm ba byte: <code>rgbOffset = grayOffset*CHANNELS</code>. Nó đọc
            ba byte, lập một tổng có trọng số và lưu một byte. Phần chữ của sách cho các trọng số: L = 0.299·r +
            0.587·g + 0.114·b.
          </p>
        }
      >
        <p>
          Now the whole kernel. A colour pixel is three consecutive bytes, red, green and blue, and a grey pixel is one
          byte. The thread computes its grey offset as above, then multiplies by three to find its pixel in the colour
          input, because every pixel before it took three bytes: <code>rgbOffset = grayOffset*CHANNELS</code>. It reads
          the three bytes, forms a weighted sum and stores one byte. The book's text gives the weights: L = 0.299·r +
          0.587·g + 0.114·b.
        </p>
      </Translated>
      <Source section="3.2" pages="52–54" figs="Fig. 3.4" />
      <Code
        tabs={[
          { label: tr('CUDA and HIP', 'CUDA và HIP'), code: GRAY_CU, tiers: TIERS, status: tr('front-end-checked; emulator-checked (python3 labs/ch03/snippets/check.py); unverified on CUDA or HIP hardware', 'front-end-checked; emulator-checked (python3 labs/ch03/snippets/check.py); unverified trên phần cứng CUDA hoặc HIP') },
          { label: tr('Metal kernel', 'Kernel Metal'), code: GRAY_METAL, tiers: TIERS, status: tr('unverified: compiles with xcrun metal, not run', 'unverified: biên dịch được với xcrun metal, chưa chạy') },
        ]}
        caption={
          <Translated
            vi={
              <>
                <code>labs/ch03/snippets/grayscale_kernel.cu</code> và <code>.metal</code>. Đây là kernel đã sửa: nó
                khác trang sách ở phép ép kiểu, các hằng số, thứ tự đối số của lời gọi và một lỗi chính tả, những điều mà
                bạn sẽ tự tìm ra ở bên dưới. Lệnh khởi chạy cũng làm tròn lên bằng số nguyên ở chỗ sách viết{' '}
                <code>ceil(m/16.0)</code>; hướng dẫn của NVIDIA dùng dạng số nguyên. Lab 3.1 cho bạn viết và chạy kernel này,
                đổi tên thành <code>colorToGrayscaleKernel</code> và nhận đầu vào trước, giống như lệnh khởi chạy của nó.
              </>
            }
          >
            <>
              <code>labs/ch03/snippets/grayscale_kernel.cu</code> and <code>.metal</code>. This is a corrected kernel: it
              differs from the page in the cast, the constants, the argument order of the call and one spelling, which
              you will find for yourself below. The launch also rounds up with integers where the book writes{' '}
              <code>ceil(m/16.0)</code>; NVIDIA's guide uses the integer form. Lab 3.1 has you write and run this kernel,
              renamed <code>colorToGrayscaleKernel</code> and taking the input first, as its launch does.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Sách tính các con số cho Luồng (0,0) của Khối (1,0): nó ghi <code>Pout[16*76 + 0] = Pout[1216]</code>{' '}
            và đọc ba byte bắt đầu tại <code>Pin[16*76*3] = Pin[3648]</code>. Bạn đã dự đoán con số đầu tiên
            trước hoạt hình. Hãy chạy từng dòng luồng đó qua kernel, rồi chạy từng bước một luồng không có điểm ảnh.
          </p>
        }
      >
        <p>
          The book works the numbers for Thread (0,0) of Block (1,0): it writes <code>Pout[16*76 + 0] = Pout[1216]</code>{' '}
          and reads the three bytes that start at <code>Pin[16*76*3] = Pin[3648]</code>. You predicted the first of these
          before the animation. Step that thread through the kernel line by line, then step a thread with no pixel.
        </p>
      </Translated>
      <Source section="3.2" pages="53–54" />
      <Translated
        vi={
          <p>
            Hãy đọc kernel theo màu khi bạn chạy từng bước. Giá trị màu tím nằm trong thanh ghi (register) riêng của luồng và dùng lại
            không tốn gì. Màu hổ phách là bộ nhớ toàn cục (global memory): kernel này chạm vào nó đúng bốn lần mỗi điểm ảnh, ba lần đọc và một lần ghi,
            và mọi thứ khác là phép tính trên thanh ghi.
          </p>
        }
      >
        <p>
          Read the kernel by colour as you step. Violet values live in the thread's own registers and cost nothing to
          reuse. Amber is global memory: this kernel touches it exactly four times per pixel, three reads and one write,
          and everything else is arithmetic on registers.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi lần một luồng, mỗi bước một dòng mã, trên một ảnh tổng hợp. Luồng thật chạy hàng nghìn cái
              cùng lúc. Tổng float được tính mỗi lần một phép toán độ chính xác đơn có làm tròn, từ trái sang phải; trình biên dịch
              GPU có thể gộp hoặc đổi thứ tự các phép toán đó.
            </p>
          }
        >
          <p>
            One thread at a time, one source line per step, on a synthetic picture. Real threads run in their thousands
            at once. The float sum is computed one rounded single-precision operation at a time, left to right; a GPU
            compiler may fuse or reorder those operations.
          </p>
        </Translated>
      </Simplification>
      <KernelStepper key={`${launch.height},${launch.width},${launch.blockW},${launch.blockH},${launch.style},${!!launch.swapped}`}
        launch={launch} sel={sel} setSel={setSel} />

      <h2><Translated vi="Tìm lỗi trong mã in">Find the errors in the printed listing</Translated></h2>
      <Translated
        vi={
          <p>
            Fig. 3.4 như bản in không phải là kernel ở trên. Một trong các khác biệt là một lỗi biên dịch sạch sẽ mà cho ra
            một ảnh sai. Bên dưới là các dòng mã của hình và lệnh khởi chạy ở p. 50, như bản in. Đánh dấu mọi chỗ bạn
            cho là sai, rồi kiểm tra. Có ba lỗi cần tìm và một chỗ sơ suất chỉ là chính tả.
          </p>
        }
      >
        <p>
          Fig. 3.4 as printed is not the kernel above. One of the differences is a defect that compiles cleanly and gives
          a wrong picture. Below are the figure's code lines and the launch from p. 50, as printed. Mark every spot you
          believe is wrong, then check. There are three defects to find and one slip that is only a spelling.
        </p>
      </Translated>
      <Source section="3.2" pages="50, 53" figs="Fig. 3.4" />
      <ErratumHunt />
      <Translated
        vi={
          <p>
            Phép ép kiểu là lỗi cần nhớ, vì C++ không giúp gì bạn với nó. Một phép ép kiểu áp dụng cho biểu thức đơn
            ngay sau nó, còn phép nhân có độ ưu tiên thấp hơn, nên dòng in có nghĩa là{' '}
            <code>((unsigned char)0.21f)*r + 0.71f*g + 0.07f*b</code>. Trước khi mở trình thử, hãy chốt xem điều đó làm gì
            với một điểm ảnh đỏ.
          </p>
        }
      >
        <p>
          The cast is the one to remember, because C++ gives you no help with it. A cast applies to the single expression
          that follows it, and multiplication binds less tightly, so the printed line means{' '}
          <code>((unsigned char)0.21f)*r + 0.71f*g + 0.07f*b</code>. Before you open the tester, commit to what that does
          to a red pixel.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, A9, C6)" /* i18n-ignore: date and citation */>
        <Translated
          vi={
            <p>
              Đã kiểm tra theo ngữ pháp C++ và bằng cách biên dịch cả hai dạng trên host với mọi cảnh báo được bật: không có chẩn đoán nào.
              Metal Shading Language dùng ngữ pháp C++, nên dòng in cũng là cùng lỗi đó ở đó.
            </p>
          }
        >
          <p>
            Checked against the C++ grammar and by compiling both forms on a host with all warnings on: no diagnostic.
            Metal Shading Language uses the C++ grammar, so the printed line is the same defect there.
          </p>
        </Translated>
      </Today>
      <Predict
        id="ch03/3.2/p-red"
        prompt={
          <Translated vi={<>Một điểm ảnh đỏ thuần là (r, g, b) = (255, 0, 0). Dòng 19 như bản in lưu byte nào cho nó?</>}>
            <>A pure red pixel is (r, g, b) = (255, 0, 0). What byte does line 19 as printed store for it?</>
          </Translated>
        }
      >
        {(v) => (
          <Translated
            vi={
              <p>
                Nó lưu <b>{grayDetail(255, 0, 0, 'printed').byte}</b>
                {v === grayDetail(255, 0, 0, 'printed').byte ? ', đúng như bạn dự đoán' : `, không phải ${v}`}. Với tổng trong
                ngoặc, các hằng số như bản in sẽ cho {grayDetail(255, 0, 0, 'printedParen').byte}, còn dòng đã sửa
                cho {grayDetail(255, 0, 0, 'corrected').byte}. Một ảnh đỏ trở thành màu đen. Trình thử bên dưới giờ
                đã mở.
              </p>
            }
          >
            <p>
              It stores <b>{grayDetail(255, 0, 0, 'printed').byte}</b>
              {v === grayDetail(255, 0, 0, 'printed').byte ? ', as you predicted' : `, not ${v}`}. With the sum in
              parentheses the printed constants would give {grayDetail(255, 0, 0, 'printedParen').byte}, and the corrected
              line gives {grayDetail(255, 0, 0, 'corrected').byte}. A red picture comes out black. The tester below is
              now open.
            </p>
          </Translated>
        )}
      </Predict>
      <Simplification>
        <Translated
          vi={
            <p>
              Trình thử tính từng phiên bản trên CPU với độ chính xác đơn, mỗi lần một phép toán có làm tròn, từ trái
              sang phải. Nó tái tạo các giá trị biên dịch trên host được ghi trong tệp facts. Một trình biên dịch GPU gộp hoặc
              đổi thứ tự các phép toán có thể lệch một mức xám.
            </p>
          }
        >
          <p>
            The tester evaluates each version on the CPU in single precision, one rounded operation at a time, left to
            right. It reproduces the host-compiled values recorded in the facts file. A GPU compiler that fuses or
            reorders the operations can differ by one grey level.
          </p>
        </Translated>
      </Simplification>
      {predictedRed ? (
        <PixelTester />
      ) : (
        <p className="muted" data-k="tester-closed"><Translated vi="Trình thử điểm ảnh sẽ mở khi bạn đã chốt một dự đoán ở trên.">The pixel tester opens once you have committed a prediction above.</Translated></p>
      )}
      <Translated
        vi={
          <p>
            Dòng đã sửa là{' '}
            <code>{'Pout[grayOffset] = (unsigned char)(0.299f*r + 0.587f*g + 0.114f*b);'}</code>. Nó khác trang sách
            ở cặp ngoặc và ở các hằng số, vốn là những hằng số mà chính phần chữ của sách nêu. Chúng cộng lại bằng 1, nên màu trắng
            vẫn là 255. Phép chuyển sang byte bỏ phần lẻ, nên giá trị được lưu có thể thay đổi một mức xám tùy
            thứ tự cộng ba tích, đó là lý do lab 3.1 so sánh với tham chiếu CPU của nó với
            dung sai 1.
          </p>
        }
      >
        <p>
          The corrected line is{' '}
          <code>{'Pout[grayOffset] = (unsigned char)(0.299f*r + 0.587f*g + 0.114f*b);'}</code>. It differs from the page
          in the parentheses and in the constants, which are the ones the book's own text states. They sum to 1, so white
          stays 255. The conversion to a byte drops the fraction, so the stored value can change by one grey level with
          the order in which the three products are added, which is why lab 3.1 compares against its CPU reference with a
          tolerance of 1.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, A8, A9)" /* i18n-ignore: date and citation */>
        <Translated
          vi={
            <p>
              0.299, 0.587, 0.114 là các trọng số luma của ITU-R BT.601. Các giá trị 0.21, 0.71, 0.07 của mã in không khớp với cả
              BT.601 lẫn BT.709 (0.2126, 0.7152, 0.0722). Trong một phép thử trên host, đổi thứ tự các phép cộng làm thay đổi byte ở 40
              trên 140,608 màu được lấy mẫu, mỗi màu lệch một mức.
            </p>
          }
        >
          <p>
            0.299, 0.587, 0.114 are the luma weights of ITU-R BT.601. The listing's 0.21, 0.71, 0.07 match neither
            BT.601 nor BT.709 (0.2126, 0.7152, 0.0722). In a host test, reordering the additions changed the byte for 40
            of 140,608 sampled colours, by one level each.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS32['ch03/3.2/q-precedence']} label={tr('Find the defect', 'Tìm lỗi')} />

      <h2><Translated vi="Bốn loại khối">Four kinds of block</Translated></h2>
      <Translated
        vi={
          <p>
            Quay lại bản đồ, chọn ảnh của sách, và di con trỏ qua các khối. Phép kiểm tra xếp 20
            khối thành bốn loại. Mười hai khối nằm trọn trong ảnh và cả 256 luồng của chúng đều làm việc. Ba khối ở
            mép phải mất 4 cột cuối, còn 12 × 16 = 192. Bốn khối dọc đáy mất 2 hàng cuối,
            còn 16 × 14 = 224. Khối ở góc mất cả hai, còn 12 × 14 = 168. Và 12·256 + 3·192 + 4·224 + 168 ={' '}
            {L0.pixels}, đúng số điểm ảnh: lại là lập luận về độ phủ, dưới dạng số học.
          </p>
        }
      >
        <p>
          Go back to the map, choose the book's picture, and move the pointer over the blocks. The guard sorts the 20
          blocks into four kinds. Twelve lie wholly inside the picture and all 256 of their threads work. The three on the
          right edge lose their last 4 columns, leaving 12 × 16 = 192. The four along the bottom lose their last 2 rows,
          leaving 16 × 14 = 224. The corner block loses both, leaving 12 × 14 = 168. And 12·256 + 3·192 + 4·224 + 168 ={' '}
          {L0.pixels}, the number of pixels: the coverage argument again, as arithmetic.
        </p>
      </Translated>
      <Source section="3.2" pages="54–55" figs="Fig. 3.5" />
      <Translated
        vi={
          <p>
            Chương 4 quay lại đúng bốn vùng này khi nó đếm các warp mà các luồng của chúng cho kết quả khác nhau tại
            phép kiểm tra.
          </p>
        }
      >
        <p>
          Chapter 4 comes back to exactly these four regions when it counts the warps whose threads disagree at the
          guard.
        </p>
      </Translated>

      <h2><Translated vi="Ba chiều">Three dimensions</Translated></h2>
      <Translated
        vi={
          <p>
            Chiều thứ ba thêm một dòng nữa cho mỗi loại. Luồng nhận một chỉ số mặt,{' '}
            <code>plane = blockIdx.z*blockDim.z + threadIdx.z</code>, và các mặt được đặt nối tiếp nhau trong bộ nhớ.
            Với n hàng và m cột trong một mặt, phần tử là <code>P[plane*m*n + row*m + col]</code>, và phép kiểm tra
            xét cả ba chỉ số. Đổi tên các chiều và đây chính là tensor bạn đánh chỉ số hằng ngày: [batch, seq, hidden]
            là plane, row, col. Đặt chiều sâu lớn hơn 1 trong trình khám phá làm phẳng để xem các mặt xếp hàng.
          </p>
        }
      >
        <p>
          A third dimension adds one more line of each kind. The thread gets a plane index,{' '}
          <code>plane = blockIdx.z*blockDim.z + threadIdx.z</code>, and planes are placed one after another in memory.
          With n rows and m columns in a plane, the element is <code>P[plane*m*n + row*m + col]</code>, and the guard
          tests all three indices. Rename the dimensions and this is the tensor you index every day: [batch, seq, hidden]
          is plane, row, col. Set the depth above 1 in the flattening explorer to watch the planes line up.
        </p>
      </Translated>
      <Source section="3.2" pages="55" />
      <Ask q={questionsS32['ch03/3.2/q-ex5-stride']} label={tr('Book exercise 5, as PyTorch states it', 'Bài tập 5 của sách, theo cách PyTorch phát biểu')} />
      <Translated
        vi={
          <p>
            Trên HIP không có gì trong mục này thay đổi. Từ khóa <code>__global__</code>, bốn biến dựng sẵn, câu
            lệnh khởi chạy và phép tính chỉ số là cùng một mã nguồn, miễn là bạn dùng các thành viên{' '}
            <code>.x</code>, <code>.y</code>, <code>.z</code> và không sao chép trọn một biến dựng sẵn vào một{' '}
            <code>uint3</code>.
          </p>
        }
      >
        <p>
          On HIP nothing in this section changes. The <code>__global__</code> qualifier, the four built-in variables, the
          launch statement and the index arithmetic are the same source text, as long as you use the{' '}
          <code>.x</code>, <code>.y</code>, <code>.z</code> members and do not copy a whole built-in into a{' '}
          <code>uint3</code>.
        </p>
      </Translated>
      <Beyond title={tr('Not in the book: HIP', 'Không có trong sách: HIP')}>
        <Translated
          vi={
            <p>
              Đã kiểm tra theo tài liệu và header HIP của AMD ngày 2026-11-20 (notes/today/ch03-facts.md, B2, B5); chưa
              biên dịch trên phần cứng AMD, và nhận xét về <code>uint3</code> là suy luận từ header của AMD. Có một
              điều trên trang ngôn ngữ của AMD không nên sao chép: công thức "global unique
              identifier" 3D của nó dùng chiều rộng và diện tích của khối làm bước nhảy, nên nó chỉ duy nhất khi lưới rộng đúng một
              khối. Bước nhảy cho dữ liệu là chiều rộng của dữ liệu, như ở trên.
            </p>
          }
        >
          <p>
            Checked against AMD's HIP documentation and headers on 2026-11-20 (notes/today/ch03-facts.md, B2, B5); not
            compiled on AMD hardware, and the remark about <code>uint3</code> is an inference from AMD's header. One
            thing on AMD's language page should not be copied: its 3D "global unique
            identifier" formula uses the block's width and area as strides, so it is unique only when the grid is one
            block wide. The stride for data is the data's width, as above.
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Cùng phép ánh xạ trên Metal">The same mapping on Metal</Translated></h2>
      <Translated
        vi={
          <p>
            Metal trao cho kernel tọa độ của nó dưới dạng tham số có thuộc tính thay vì biến toàn cục, và một trong số đó
            đã là đáp án. <code>thread_position_in_grid</code> là vị trí của luồng trong toàn bộ lưới:{' '}
            <code>.x</code> là cột, <code>.y</code> là hàng, và gốc tọa độ ở góc trên bên trái. Tài liệu của Apple
            in công thức CUDA làm định nghĩa của nó: vị trí threadgroup nhân số luồng mỗi threadgroup, cộng vị trí trong
            threadgroup. Đổi kiểu khởi chạy của bản đồ để thấy mọi cái tên thay đổi còn các con số vẫn giữ nguyên.
          </p>
        }
      >
        <p>
          Metal hands a kernel its coordinates as attributed parameters instead of global variables, and one of them is
          already the answer. <code>thread_position_in_grid</code> is the thread's position in the whole grid:{' '}
          <code>.x</code> is the column, <code>.y</code> the row, and the origin is the top left. Apple's documentation
          prints the CUDA formula as its definition: threadgroup position times threads per threadgroup, plus position in
          the threadgroup. Change the map's launch style to see every name change and the numbers stay put.
        </p>
      </Translated>
      <Beyond title={tr('Not in the book: Metal', 'Không có trong sách: Metal')}>
        <Translated
          vi={
            <p>
              Sách chỉ dùng CUDA. Phần này là của khóa học này, đã kiểm tra ngày 2026-11-20 theo tài liệu của Apple, đặc tả Metal
              Shading Language và mã nguồn của ba dự án (notes/today/ch03-facts.md, C1, C3, C5, C6,
              D3, D4, D5). Kernel Metal ở trên biên dịch được ở đây và chưa được chạy; lab 3.1 chạy một kernel như vậy.
            </p>
          }
        >
          <p>
            The book uses CUDA only. This part is ours, checked on 2026-11-20 against Apple's documentation, the Metal
            Shading Language specification and the three projects' sources (notes/today/ch03-facts.md, C1, C3, C5, C6,
            D3, D4, D5). The Metal kernel shown above compiles here and has not been run; lab 3.1 runs one.
          </p>
        </Translated>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th><Translated vi="Trong kernel">In the kernel</Translated></th><th><Translated vi="CUDA và HIP">CUDA and HIP</Translated></th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr><td><Translated vi="Vị trí của khối">Block's position</Translated></td><td><code>blockIdx</code></td><td><code>[[threadgroup_position_in_grid]]</code></td></tr>
          <tr><td><Translated vi="Vị trí của luồng trong khối">Thread's position in its block</Translated></td><td><code>threadIdx</code></td><td><code>[[thread_position_in_threadgroup]]</code></td></tr>
          <tr><td><Translated vi="Hình dạng khối">Block shape</Translated></td><td><code>blockDim</code></td><td><code>[[threads_per_threadgroup]]</code></td></tr>
          <tr><td><Translated vi="Hình dạng lưới tính theo khối">Grid shape in blocks</Translated></td><td><code>gridDim</code></td><td><code>[[threadgroups_per_grid]]</code></td></tr>
          <tr><td><Translated vi="Hàng và cột trong dữ liệu">Row and column in the data</Translated></td><td><Translated vi="tính từ ba thứ ở trên">computed from the three above</Translated></td><td><Translated vi={<><code>[[thread_position_in_grid]]</code>, được cho sẵn</>}><code>[[thread_position_in_grid]]</code>, given</Translated></td></tr>
          <tr><td><Translated vi="Ảnh dạng byte">Byte picture</Translated></td><td><code>unsigned char*</code></td><td><Translated vi={<><code>device uchar*</code> với <code>[[buffer(n)]]</code></>}><code>device uchar*</code> with <code>[[buffer(n)]]</code></Translated></td></tr>
          <tr><td><Translated vi="Chiều rộng và chiều cao">Width and height</Translated></td><td><Translated vi="truyền theo giá trị trong lệnh khởi chạy">passed by value in the launch</Translated></td><td><Translated vi={<><code>constant T&amp;</code>, được host điền bằng <code>setBytes</code></>}><code>constant T&amp;</code>, filled by the host with <code>setBytes</code></Translated></td></tr>
          <tr><td><Translated vi="Hằng số float">Float literal</Translated></td><td><code>0.299f</code></td><td><Translated vi={<><code>0.299f</code>; không có <code>double</code></>}><code>0.299f</code>; there is no <code>double</code></Translated></td></tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Lệnh khởi chạy là nơi Metal cho một lựa chọn mà CUDA của sách không có. <code>dispatchThreadgroups</code> nhận một
            số lượng threadgroup, như một lưới CUDA: bạn làm tròn lên và bạn kiểm tra biên. <code>dispatchThreads</code> nhận
            kích thước lưới tính bằng luồng. Khi đó Metal làm các threadgroup dọc các mép nhỏ hơn, tạo đúng số
            luồng bạn yêu cầu, và tài liệu của Apple nói không cần kiểm tra vị trí. Bốn loại khối
            chỉ tồn tại với lời gọi thứ nhất. Cả hai đều được dùng thật; danh sách ở cuối trang này cho biết ở đâu.
          </p>
        }
      >
        <p>
          The launch is where Metal offers a choice the book's CUDA does not. <code>dispatchThreadgroups</code> takes a
          number of threadgroups, like a CUDA grid: you round up and you guard. <code>dispatchThreads</code> takes the
          size of the grid in threads. Metal then makes the threadgroups along the edges smaller, creates exactly the
          threads you asked for, and Apple's documentation says the position check is not needed. The four kinds of block
          exist only under the first call. Both are in real use; the list at the end of this page says where.
        </p>
      </Translated>
      <Ask q={questionsS32['ch03/3.2/q-noguard']} />
      <Translated
        vi={
          <p>
            Bên trong kernel, có ba điều cần cẩn thận. Một ảnh dạng byte là một buffer thường, <code>device uchar*</code>, đánh chỉ số
            y hệt như trong CUDA, nhưng đừng dùng <code>uchar3</code> để đọc một điểm ảnh: trong Metal Shading Language nó
            chiếm 4 byte. Hãy giữ phép tính <code>rgbOffset</code> trên một con trỏ <code>uchar</code>, hoặc dùng{' '}
            <code>packed_uchar3</code>, có kích thước 3. Không có <code>double</code>, nên hãy viết <code>0.299f</code> với hậu
            tố của nó. Và các giá trị vô hướng nhỏ như chiều rộng và chiều cao đến dưới dạng tham chiếu <code>constant T&amp;</code> mà
            host điền bằng <code>setBytes</code>.
          </p>
        }
      >
        <p>
          Inside the kernel, three things need care. A byte picture is a plain buffer, <code>device uchar*</code>, indexed
          exactly as in CUDA, but do not reach for <code>uchar3</code> to read a pixel: in Metal Shading Language it
          occupies 4 bytes. Keep the <code>rgbOffset</code> arithmetic on a <code>uchar</code> pointer, or use{' '}
          <code>packed_uchar3</code>, which is 3. There is no <code>double</code>, so write <code>0.299f</code> with its
          suffix. And small scalars such as the width and height arrive as <code>constant T&amp;</code> references that
          the host fills with <code>setBytes</code>.
        </p>
      </Translated>
      <Ask q={questionsS32['ch03/3.2/q-uchar3']} />
      <Translated
        vi={
          <p>
            Giờ bạn đã có phép ánh xạ, phép kiểm tra biên và offset phẳng. <a href={href('ch03', '3.3')}>Mục 3.3</a> giữ cả
            ba và cho mỗi luồng một vòng lặp. Trên <a href={href('ch03', 'labs')}>trang lab</a>, lab 3.1 cho bạn viết
            kernel này và chạy nó trên GPU của chính bạn, còn lab 3.5 giấu bốn lỗi chỉ số để bạn tìm.
          </p>
        }
      >
        <p>
          You now have the mapping, the guard and the flat offset. <a href={href('ch03', '3.3')}>Section 3.3</a> keeps all
          three and gives each thread a loop. On the <a href={href('ch03', 'labs')}>labs page</a>, lab 3.1 has you write
          this kernel and run it on your own GPU, and lab 3.5 hides four index defects for you to find.
        </p>
      </Translated>

      <h2><Translated vi="Trong các backend thực tế">In real backends</Translated></h2>
      <Translated
        vi={
          <p>
            Những lựa chọn của trang này xuất hiện ở đâu trong mã bạn sẽ đọc sau này. Không phần nào trong đó cần cho các lab.
          </p>
        }
      >
        <p>
          Where the choices of this page turn up in code you will read later. None of it is needed for the labs.
        </p>
      </Translated>
      <ul data-list="backends">
        <Translated
          vi={
            <li>
              <b>Phép toán theo từng phần tử.</b> Các kernel CUDA của PyTorch bên dưới <code>a + b</code> hay <code>torch.relu</code>{' '}
              khởi chạy một lưới một chiều bất kể shape của tensor, và mỗi luồng chỉ tính chỉ số phẳng.
              Hướng dẫn của NVIDIA: "Việc dùng khối luồng và lưới đa chiều chỉ nhằm cho tiện và không
              ảnh hưởng đến hiệu năng." Một lưới 2D cho bạn <code>row</code> và <code>col</code> mà không cần phép chia; khi
              phép toán không quan tâm một phần tử nằm ở hàng nào, hãy làm phẳng trước.
            </li>
          }
        >
          <li>
            <b>Elementwise operations.</b> PyTorch's CUDA kernels under <code>a + b</code> or <code>torch.relu</code>{' '}
            launch a one-dimensional grid whatever the tensor's shape, and each thread computes flat indices only.
            NVIDIA's guide: "The use of multi-dimensional thread blocks and grids is for convenience only and does not
            affect performance." A 2D grid hands you <code>row</code> and <code>col</code> without a division; when the
            operation does not care which row an element is in, flatten first.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Hai kiểu khởi chạy của Metal.</b> Backend Metal của llama.cpp và matmul MPS của PyTorch gọi{' '}
              <code>dispatchThreadgroups</code>, nên chúng làm tròn lên và kiểm tra biên. MLX gọi <code>dispatchThreads</code> và
              tự làm phẳng vị trí thành <code>index.x + grid_dim.x * index.y</code>, tức là{' '}
              <code>row*width + col</code> với chiều rộng của chính lưới làm bước nhảy.
            </li>
          }
        >
          <li>
            <b>The two Metal launches.</b> llama.cpp's Metal backend and PyTorch's MPS matmul call{' '}
            <code>dispatchThreadgroups</code>, so they round up and guard. MLX calls <code>dispatchThreads</code> and
            flattens the position itself as <code>index.x + grid_dim.x * index.y</code>, which is{' '}
            <code>row*width + col</code> with the grid's own width as the stride.
          </li>
        </Translated>
      </ul>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, A2, C3, D3, D4, D5, D6)" /* i18n-ignore: date and citation */>
        <Translated
          vi={
            <p>
              Đọc trong mã nguồn của các dự án vào ngày kiểm tra. Câu trích dẫn lấy từ hướng dẫn lập trình của NVIDIA.
            </p>
          }
        >
          <p>
            Read in the projects' sources on the check date. The quotation is from NVIDIA's programming guide.
          </p>
        </Translated>
      </Today>
    </>
  );
}
