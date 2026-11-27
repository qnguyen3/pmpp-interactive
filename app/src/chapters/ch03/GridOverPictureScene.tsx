import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { BOOK, CHANNELS, blockInfo, bookLabel, latticeOf, layout, regions, threadAt } from '../../sim/mapping';

// "A grid over a picture": the book's 62 x 76 example (pp. 49-55) built up in six
// beats, one narration clip each (content/ch03/narration/grid-over-picture-N.txt).
// Every count, position and offset drawn here is computed by app/src/sim/mapping.ts.
const L = layout(BOOK);
const C = 4; // one thread is a 4 x 4 cell
const LX = 44, LY = 48; // lattice origin, multiples of C so the cell patterns line up
const PW = BOOK.width * C, PH = BOOK.height * C; // the picture
const TW = L.threadsX * C, TH = L.threadsY * C; // the threads
const BW = BOOK.blockW * C, BH = BOOK.blockH * C;
const RX = 396; // right-hand text column
const SX = 44, SW = 664; // the flat strips
const SY = 330, SH = 14; // Pout
const SY2 = 390; // Pin
const ROW_W = SW / BOOK.height; // one picture row inside the strip

const ONE = latticeOf(BOOK, 0, 1, 0, 0); // Thread (0,0) of Block (1,0): labels are (y, x)
const T = threadAt(BOOK, ONE.x, ONE.y);
const REG = regions(BOOK);
const BLOCKS = Array.from({ length: L.blocks }, (_, i) => blockInfo(BOOK, i % L.gridX, Math.floor(i / L.gridX)));
const RIGHT = L.threadsX - BOOK.width; // surplus thread columns
const BELOW = L.threadsY - BOOK.height; // surplus thread rows

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const ease = (q: number) => q * q * (3 - 2 * q);
const mix = (a: number, b: number, q: number) => a + (b - a) * q;

interface Line {
  text: ReactNode;
  /** 0 to 1. */
  show: number;
  tone?: 'reg' | 'global' | 'muted' | 'strong';
  mono?: boolean;
}

interface Frame {
  pic: number;
  dims?: number;
  /** Blocks laid so far. */
  laid?: number;
  /** How faded the block outlines are (0 = full). */
  quiet?: number;
  overhang?: number;
  /** How far the surplus threads are switched off. */
  off?: number;
  thread?: number;
  unroll?: number;
  mark?: number;
  pin?: number;
  shade?: number;
  lines: Line[];
}

const toneFill = (t?: Line['tone']) =>
  t === 'reg' ? 'var(--reg)' : t === 'global' ? 'var(--global)' : t === 'muted' ? 'var(--ink-2)' : 'var(--ink)';

function Stage(f: Frame) {
  const laid = f.laid ?? 0;
  const u = f.unroll ?? 0;
  // Row k leaves the picture at its own moment and takes a fifth of the unroll to arrive.
  const q = (k: number) => ease(clamp((u - (k / BOOK.height) * 0.8) / 0.2));
  const gone = u > 0 ? Array.from({ length: BOOK.height }, (_, k) => q(k)) : [];
  const markX = SX + T.row * ROW_W;
  return (
    <svg viewBox="0 0 720 416" role="img" aria-label={tr('Animation: blocks of threads laid over a 62 by 76 picture', 'Hoạt hình: các khối luồng đặt phủ lên một bức ảnh 62 × 76')}
      style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      <defs>
        <pattern id="gop-pix" width={C} height={C} patternUnits="userSpaceOnUse">
          <rect width={C - 1} height={C - 1} fill="var(--ink-3)" />
        </pattern>
        <pattern id="gop-thr" width={C} height={C} patternUnits="userSpaceOnUse">
          <rect x="0.4" y="0.4" width={C - 1.8} height={C - 1.8} fill="none" stroke="var(--ink-3)" strokeWidth="0.7" />
        </pattern>
      </defs>

      {/* threads of the blocks laid so far; outside the picture they stay hollow */}
      <g opacity={1 - 0.78 * (f.off ?? 0)}>
        {BLOCKS.slice(0, laid).map((b) => (
          <rect key={`t${b.bx},${b.by}`} x={LX + b.x0 * C} y={LY + b.y0 * C} width={BW} height={BH} fill="url(#gop-thr)" />
        ))}
      </g>
      {/* the picture */}
      <rect x={LX} y={LY} width={PW} height={PH} fill="var(--page)" />
      <rect x={LX} y={LY} width={PW} height={PH} fill="url(#gop-pix)" opacity={f.pic} />
      {/* rows that have left for the strip */}
      {gone.map((g, k) => (g > 0 ? <rect key={k} x={LX} y={LY + k * C} width={PW} height={C} fill="var(--page)" opacity={0.72 * g} /> : null))}

      {/* region shading and per-block counts */}
      {f.shade
        ? BLOCKS.map((b) => (
            <g key={`s${b.bx},${b.by}`} opacity={f.shade}>
              <rect x={LX + b.x0 * C} y={LY + b.y0 * C} width={BW} height={BH} fill="var(--ink)" fillOpacity={0.06 + 0.3 * (b.working / b.launched)} />
              <rect x={LX + b.x0 * C + BW / 2 - 17} y={LY + b.y0 * C + BH / 2 - 10} width="34" height="19" rx="3" fill="var(--page)" />
              <text x={LX + b.x0 * C + BW / 2} y={LY + b.y0 * C + BH / 2 + 4} fontSize="12" fontWeight="650" textAnchor="middle">{b.working}</text>
            </g>
          ))
        : null}

      {/* block borders */}
      <g opacity={1 - 0.6 * (f.quiet ?? 0)}>
        {BLOCKS.slice(0, laid).map((b) => (
          <rect key={`b${b.bx},${b.by}`} x={LX + b.x0 * C} y={LY + b.y0 * C} width={BW} height={BH} fill="none" stroke="var(--ink)" strokeWidth="1.3" />
        ))}
      </g>

      {/* the picture's size */}
      {f.dims ? (
        <g opacity={f.dims} fontSize="11.5">
          <text x={LX + PW / 2} y={LY - 26} textAnchor="middle"><Translated vi={<>{BOOK.width} cột</>}>{BOOK.width} columns</Translated></text>
          <path d={`M${LX} ${LY - 20} h${PW} M${LX} ${LY - 24} v8 M${LX + PW} ${LY - 24} v8`} stroke="var(--ink-2)" fill="none" />
          <text x={LX - 24} y={LY + PH / 2} textAnchor="middle" transform={`rotate(-90 ${LX - 24} ${LY + PH / 2})`}><Translated vi={<>{BOOK.height} hàng</>}>{BOOK.height} rows</Translated></text>
          <path d={`M${LX - 16} ${LY} v${PH} M${LX - 20} ${LY} h8 M${LX - 20} ${LY + PH} h8`} stroke="var(--ink-2)" fill="none" />
        </g>
      ) : null}

      {/* the overhang */}
      {f.overhang ? (
        <g opacity={f.overhang} fill="none" stroke="var(--ink)" strokeWidth="1.8" strokeDasharray="5 3">
          <rect x={LX + PW} y={LY} width={TW - PW} height={TH} />
          <rect x={LX} y={LY + PH} width={TW} height={TH - PH} />
        </g>
      ) : null}

      {/* one thread */}
      {f.thread ? (
        <g opacity={f.thread}>
          <rect x={LX + T.blockIdx.x * BW} y={LY + T.blockIdx.y * BH} width={BW} height={BH} fill="none" stroke="var(--ink)" strokeWidth="2.6" />
          <path d={`M${LX - 6} ${LY + T.row * C + C / 2} H${LX + T.col * C}`} stroke="var(--reg)" strokeWidth="1.4" />
          <circle cx={LX + T.col * C + C / 2} cy={LY + T.row * C + C / 2} r="7" fill="none" stroke="var(--reg)" strokeWidth="2" />
          <rect x={LX + T.col * C} y={LY + T.row * C} width={C} height={C} fill="var(--reg)" />
          <text x={LX + T.col * C + 14} y={LY + T.row * C - 5} fontSize="11.5" fontWeight="650" style={{ fill: 'var(--reg)' }}>
            <Translated vi={<>hàng {T.row}, cột {T.col}</>}>row {T.row}, col {T.col}</Translated>
          </text>
        </g>
      ) : null}

      {/* the flat output: rows in flight, then at rest */}
      {u > 0 ? (
        <g>
          <text x={SX} y={SY - 6} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}>
            <Translated vi={<>Pout, một mảng phẳng gồm {L.pixels} byte</>}>Pout, one flat array of {L.pixels} bytes</Translated>
          </text>
          <rect x={SX} y={SY} width={SW} height={SH} fill="none" stroke="var(--rule)" />
          {gone.map((g, k) => {
            if (g <= 0) return null;
            const hot = k === T.row;
            return (
              <rect key={k}
                x={mix(LX, SX + k * ROW_W, g)} y={mix(LY + k * C, SY, g)}
                width={mix(PW, ROW_W, g)} height={mix(C, SH, g)}
                fill={hot ? 'var(--global)' : 'var(--global-wash)'} stroke="var(--global)" strokeWidth={hot ? 1 : 0.5} />
            );
          })}
        </g>
      ) : null}
      {f.mark ? (
        <g opacity={f.mark}>
          <path d={`M${markX} ${SY - 3} v${SH + 6}`} stroke="var(--ink)" strokeWidth="1.6" />
          <text x={markX + 5} y={SY + SH + 13} fontSize="11.5" fontWeight="650">
            <Translated vi={<>hàng {T.row} bắt đầu tại {T.row} × {BOOK.width} = {T.grayOffset}</>}>row {T.row} starts at {T.row} × {BOOK.width} = {T.grayOffset}</Translated>
          </text>
        </g>
      ) : null}
      {f.pin ? (
        <g opacity={f.pin}>
          <text x={SX} y={SY2 - 6} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}>
            <Translated vi={<>Pin, {CHANNELS} byte mỗi điểm ảnh, {L.pixels * CHANNELS} byte</>}>Pin, {CHANNELS} bytes per pixel, {L.pixels * CHANNELS} bytes</Translated>
          </text>
          <rect x={SX} y={SY2} width={SW} height={SH} fill="var(--global-wash)" stroke="var(--global)" strokeWidth="0.6" />
          <path d={`M${markX} ${SY2 - 3} v${SH + 6}`} stroke="var(--ink)" strokeWidth="1.6" />
          {['r', 'g', 'b'].map((ch, k) => (
            <g key={ch}>
              <rect x={markX + 6 + k * 46} y={SY2 - 1} width="44" height={SH + 2} rx="2" fill={k === 0 ? 'var(--global)' : 'var(--page)'} stroke="var(--global)" />
              <text x={markX + 28 + k * 46} y={SY2 + 11} fontSize="10.5" textAnchor="middle" style={{ fill: k === 0 ? 'var(--page)' : 'var(--ink)' }}>
                {ch} {T.rgbOffset! + k}
              </text>
            </g>
          ))}
          <text x={markX + 6 + 3 * 46 + 6} y={SY2 + 11} fontSize="11.5" fontWeight="650">
            {CHANNELS} × {T.grayOffset} = {T.rgbOffset}
          </text>
        </g>
      ) : null}

      {/* the running text */}
      {f.lines.map((ln, i) =>
        ln.show > 0 ? (
          <text key={i} x={RX} y={LY + 14 + i * 21} fontSize={ln.mono ? 12 : 13} opacity={ln.show}
            fontWeight={ln.tone === 'strong' ? 650 : 450}
            style={{ fill: toneFill(ln.tone), fontVariationSettings: ln.mono ? "'MONO' 1" : undefined, whiteSpace: 'pre' }}>
            {ln.text}
          </text>
        ) : null,
      )}
    </svg>
  );
}

const all = L.blocks;
const sumLines = (p: number, a: number, b: number): Line[] => [
  { text: tr(`Thread ${bookLabel(T.threadIdx)} of Block ${bookLabel(T.blockIdx)}`, `Luồng ${bookLabel(T.threadIdx)} của Khối ${bookLabel(T.blockIdx)}`), show: 1, tone: 'strong' },
  { text: tr('labels are (y, x): one block down, none across', 'nhãn ghi (y, x): xuống một khối, không sang ngang khối nào'), show: 1, tone: 'muted' },
  { text: 'row = blockIdx.y × blockDim.y + threadIdx.y', show: ramp(p, a, a + 0.1), mono: true },
  { text: `    = ${T.blockIdx.y} × ${T.blockDim.y} + ${T.threadIdx.y} = ${T.row}`, show: ramp(p, a + 0.08, a + 0.18), tone: 'reg', mono: true },
  { text: 'col = blockIdx.x × blockDim.x + threadIdx.x', show: ramp(p, b, b + 0.1), mono: true },
  { text: `    = ${T.blockIdx.x} × ${T.blockDim.x} + ${T.threadIdx.x} = ${T.col}`, show: ramp(p, b + 0.08, b + 0.18), tone: 'reg', mono: true },
];

const beats: Beat[] = [
  {
    clip: 'grid-over-picture-1',
    get label() {
      return tr('The picture', 'Bức ảnh');
    },
    draw: (p) => (
      <Stage pic={mix(0.35, 1, ramp(p, 0, 0.2))} dims={ramp(p, 0.1, 0.3)}
        lines={[
          { text: tr(`${BOOK.height} rows × ${BOOK.width} columns`, `${BOOK.height} hàng × ${BOOK.width} cột`), show: ramp(p, 0.15, 0.3) },
          { text: tr(`= ${L.pixels} pixels`, `= ${L.pixels} điểm ảnh`), show: ramp(p, 0.36, 0.48), tone: 'strong' },
          { text: tr('one thread for each pixel', 'mỗi điểm ảnh một luồng'), show: ramp(p, 0.52, 0.64) },
          { text: tr('threads come in blocks', 'các luồng đi theo khối'), show: ramp(p, 0.68, 0.8), tone: 'muted' },
        ]} />
    ),
  },
  {
    clip: 'grid-over-picture-2',
    get label() {
      return tr('Blocks', 'Các khối');
    },
    draw: (p) => {
      const n = Math.ceil(ramp(p, 0.14, 0.74) * all);
      const firstRow = Math.min(n, L.gridX);
      return (
        <Stage pic={1} dims={1} laid={n}
          lines={[
            { text: tr(`blocks of ${BOOK.blockW} × ${BOOK.blockH} = ${BOOK.blockW * BOOK.blockH} threads`, `khối ${BOOK.blockW} × ${BOOK.blockH} = ${BOOK.blockW * BOOK.blockH} luồng`), show: 1 },
            { text: tr(`across: ${firstRow} × ${BOOK.blockW} = ${firstRow * BOOK.blockW}${firstRow === L.gridX ? ` covers ${BOOK.width}` : ''}`, `ngang: ${firstRow} × ${BOOK.blockW} = ${firstRow * BOOK.blockW}${firstRow === L.gridX ? ` phủ ${BOOK.width}` : ''}`), show: n > 0 ? 1 : 0 },
            { text: tr(`down: ${Math.ceil(n / L.gridX)} × ${BOOK.blockH} = ${Math.ceil(n / L.gridX) * BOOK.blockH}${n === all ? ` covers ${BOOK.height}` : ''}`, `dọc: ${Math.ceil(n / L.gridX)} × ${BOOK.blockH} = ${Math.ceil(n / L.gridX) * BOOK.blockH}${n === all ? ` phủ ${BOOK.height}` : ''}`), show: n > 0 ? 1 : 0 },
            { text: '', show: 0 },
            { text: tr(`blocks: ${n}`, `số khối: ${n}`), show: 1, tone: 'strong' },
            { text: tr(`threads: ${n} × ${BOOK.blockW * BOOK.blockH} = ${n * BOOK.blockW * BOOK.blockH}`, `số luồng: ${n} × ${BOOK.blockW * BOOK.blockH} = ${n * BOOK.blockW * BOOK.blockH}`), show: 1, tone: 'strong' },
            { text: tr(`pixels: ${L.pixels}`, `số điểm ảnh: ${L.pixels}`), show: ramp(p, 0.8, 0.9), tone: 'muted' },
          ]} />
      );
    },
  },
  {
    clip: 'grid-over-picture-3',
    get label() {
      return tr('The overhang', 'Phần thừa ra');
    },
    draw: (p) => (
      <Stage pic={1} dims={1} laid={all} overhang={ramp(p, 0.04, 0.18)} off={ramp(p, 0.72, 0.92)}
        lines={[
          { text: tr(`right: ${RIGHT} columns × ${L.threadsY} rows = ${RIGHT * L.threadsY}`, `bên phải: ${RIGHT} cột × ${L.threadsY} hàng = ${RIGHT * L.threadsY}`), show: ramp(p, 0.12, 0.24) },
          { text: tr(`bottom: ${BELOW} rows × ${L.threadsX} columns = ${BELOW * L.threadsX}`, `bên dưới: ${BELOW} hàng × ${L.threadsX} cột = ${BELOW * L.threadsX}`), show: ramp(p, 0.3, 0.42) },
          { text: tr(`the corner counted twice: ${RIGHT} × ${BELOW} = ${RIGHT * BELOW}`, `góc bị đếm hai lần: ${RIGHT} × ${BELOW} = ${RIGHT * BELOW}`), show: ramp(p, 0.42, 0.52), tone: 'muted' },
          { text: tr(`${L.surplus} threads with no pixel`, `${L.surplus} luồng không có điểm ảnh nào`), show: ramp(p, 0.5, 0.6), tone: 'strong' },
          { text: '', show: 0 },
          { text: 'if (col < width && row < height)', show: ramp(p, 0.66, 0.78), mono: true },
          { text: tr(`false for those ${L.surplus}: they stop`, `sai với ${L.surplus} luồng đó: chúng dừng lại`), show: ramp(p, 0.8, 0.92) },
        ]} />
    ),
  },
  {
    clip: 'grid-over-picture-4',
    get label() {
      return tr('One thread', 'Một luồng');
    },
    draw: (p) => <Stage pic={1} dims={1} laid={all} off={1} quiet={0.6} thread={ramp(p, 0.04, 0.16)} lines={sumLines(p, 0.42, 0.74)} />,
  },
  {
    clip: 'grid-over-picture-5',
    get label() {
      return tr('Into one strip', 'Trải thành một dải');
    },
    draw: (p) => (
      <Stage pic={1} laid={all} off={1} quiet={1} thread={1} unroll={ramp(p, 0.06, 0.5)} mark={ramp(p, 0.5, 0.58)} pin={ramp(p, 0.72, 0.82)}
        lines={[
          { text: tr(`Thread ${bookLabel(T.threadIdx)} of Block ${bookLabel(T.blockIdx)}: row ${T.row}, col ${T.col}`, `Luồng ${bookLabel(T.threadIdx)} của Khối ${bookLabel(T.blockIdx)}: hàng ${T.row}, cột ${T.col}`), show: 1, tone: 'strong' },
          { text: tr('rows go into memory one after another', 'các hàng nằm nối tiếp nhau trong bộ nhớ'), show: ramp(p, 0.06, 0.16) },
          { text: '', show: 0 },
          { text: 'grayOffset = row × width + col', show: ramp(p, 0.44, 0.54), mono: true },
          { text: `           = ${T.row} × ${BOOK.width} + ${T.col} = ${T.grayOffset}`, show: ramp(p, 0.52, 0.62), tone: 'global', mono: true },
          { text: `rgbOffset = grayOffset × ${CHANNELS}`, show: ramp(p, 0.72, 0.82), mono: true },
          { text: `          = ${T.rgbOffset}`, show: ramp(p, 0.8, 0.9), tone: 'global', mono: true },
        ]} />
    ),
  },
  {
    clip: 'grid-over-picture-6',
    get label() {
      return tr('Four kinds of block', 'Bốn loại khối');
    },
    draw: (p) => (
      <Stage pic={1} dims={1} laid={all} shade={ramp(p, 0.03, 0.14)}
        lines={[
          { text: tr('threads that work, block by block', 'số luồng làm việc, theo từng khối'), show: 1, tone: 'strong' },
          ...REG.map((r, i): Line => ({
            text: `${String(r.blocks).padStart(2, ' ')} × ${r.working} = ${r.total}`, show: ramp(p, 0.14 + i * 0.14, 0.22 + i * 0.14), mono: true,
          })),
          { text: tr(`sum ${REG.reduce((s, r) => s + r.total, 0)} = ${BOOK.height} × ${BOOK.width}`, `tổng ${REG.reduce((s, r) => s + r.total, 0)} = ${BOOK.height} × ${BOOK.width}`), show: ramp(p, 0.68, 0.76), tone: 'strong', mono: true },
          { text: tr('every pixel exactly once', 'mỗi điểm ảnh đúng một lần'), show: ramp(p, 0.74, 0.82) },
          { text: tr('chapter 4 returns to these four kinds', 'chương 4 quay lại bốn loại này'), show: ramp(p, 0.86, 0.94), tone: 'muted' },
        ]} />
    ),
  },
];

export function GridOverPictureScene() {
  return <Scene chapter="ch03" title={tr('A grid over a picture', 'Một lưới phủ lên bức ảnh')} beats={beats} />;
}
