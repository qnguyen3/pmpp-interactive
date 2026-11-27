import { useId, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, type Tier } from '../../components/notes';
import { solutionTabStatus } from '../../components/LabKit';
import { Translated, tr } from '../../lib/i18n';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  CLAUSES,
  CLAUSE_TEXT,
  CORRECT,
  DEFECTS,
  DEFECT_LABEL,
  ORDERS,
  ORDER_LABEL,
  area,
  blurImage,
  challenge,
  countMap,
  numbered,
  patch,
  span,
  spot,
  tally,
  textured,
  trace,
  uniform,
  withDefect,
  wrongSpan,
  type BadAccess,
  type BlurOptions,
  type Clause,
  type Defect,
  type Order,
  type Picture,
  type Pos,
} from '../../sim/blur';
import { Hatch, InPlaceBlurScene, PixelCells, grey, show, type CellLook } from './InPlaceBlurScene';

// The kernel of Fig. 3.8 as this course compiles it: labs/ch03/snippets/blur_kernel.cu.
// S33.snippets.test.ts fails if this copy and that file drift apart. Shown with the
// #define as line 0, so that every other line carries the number the book prints.
const KERNEL = `#define BLUR_SIZE 1   // patch radius: the patch is 2*BLUR_SIZE+1 pixels on a side
__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (col < w && row < h) {
        int pixVal = 0;
        int pixels = 0;

        // Average the (2*BLUR_SIZE+1) x (2*BLUR_SIZE+1) patch around (row, col)
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                // Keep only the positions that are inside the picture
                if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                    pixVal += in[curRow*w + curCol];
                    ++pixels;   // how many values went into the sum
                }
            }
        }
        // Store the average of the values that were kept
        out[row*w + col] = (unsigned char)((float)pixVal/pixels);
    }
}`;
const BLUR_TIERS: Record<string, Tier> = {
  in: 'global', out: 'global',
  col: 'reg', row: 'reg', pixVal: 'reg', pixels: 'reg', blurRow: 'reg', blurCol: 'reg', curRow: 'reg', curCol: 'reg',
};
// Lines 10 to 22 for the stepper, with the common indentation removed.
const LOOP = KERNEL.split('\n').slice(10, 23).map((l) => l.slice(8)).join('\n');

const at = (p: Pos) => `(${p.row}, ${p.col})`;
const clampInt = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;

// ---------------------------------------------------------------- loop stepper for one thread
const STEP_PIC = numbered(7, 6);
const PRESETS: { label: string; row: number; col: number }[] = [
  { get label() { return tr('1: top-left corner', '1: góc trên bên trái'); }, row: 0, col: 0 },
  { get label() { return tr('2: left edge', '2: mép trái'); }, row: 3, col: 0 },
  { get label() { return tr('3: top edge', '3: mép trên'); }, row: 0, col: 3 },
  { get label() { return tr('4: right edge', '4: mép phải'); }, row: 2, col: 6 },
  { get label() { return tr('5: bottom-right corner', '5: góc dưới bên phải'); }, row: 5, col: 6 },
  { get label() { return tr('interior', 'bên trong'); }, row: 2, col: 3 },
];

function LoopStepper() {
  const hatch = useId();
  const [k, setK] = useState(0);
  const [n, setN] = useState(0);
  const { w, h } = STEP_PIC;
  const { row, col } = PRESETS[k];
  const t = useMemo(() => trace(STEP_PIC, row, col, 1), [row, col]);
  const total = t.steps.length + 1; // nine loop iterations and the store
  const cur = n >= 1 && n <= t.steps.length ? t.steps[n - 1] : undefined;
  const done = n === total;
  const last = n === 0 ? undefined : t.steps[Math.min(n, t.steps.length) - 1];
  const pixVal = last?.pixVal ?? 0;
  const pixels = last?.pixels ?? 0;
  const visited = new Map(t.steps.slice(0, Math.min(n, t.steps.length)).map((s) => [s.curRow + ',' + s.curCol, s]));

  const look = (r: number, cc: number, inside: boolean): CellLook | null => {
    const key = r + ',' + cc;
    const inPatch = Math.abs(r - row) <= 1 && Math.abs(cc - col) <= 1;
    if (!inside && !inPatch) return null;
    const text = inside ? String(STEP_PIC.data[r * w + cc]) : undefined;
    if (cur && cur.curRow === r && cur.curCol === cc)
      return cur.inside
        ? { text, fill: 'var(--global)', ink: 'var(--page)', stroke: 'var(--global)', strokeWidth: 3, bold: true }
        : { hatch: true, stroke: 'var(--ink)', strokeWidth: 3 };
    const v = visited.get(key);
    if (v) return v.inside ? { text, fill: 'var(--global-wash)', stroke: 'var(--global)' } : { hatch: true, dash: true, stroke: 'var(--ink-3)' };
    if (r === row && cc === col) return { text, stroke: 'var(--ink)', strokeWidth: 3 };
    if (inPatch) return inside ? { text, dash: true, stroke: 'var(--ink-3)' } : { dash: true, stroke: 'var(--ink-3)' };
    return { text };
  };

  let say: ReactNode;
  if (n === 0)
    say = (
      <Translated
        vi={
          <>
            Chưa có gì chạy. Luồng này sở hữu điểm ảnh đầu ra {at({ row, col })} của một ảnh cao {h} hàng và rộng {w}
            cột, và bắt đầu với pixVal = 0 và pixels = 0. Bấm Bước.
          </>
        }
      >
        <>
          Nothing has run yet. This thread owns output pixel {at({ row, col })} of a picture {h} rows high and {w} columns
          wide, and starts with pixVal = 0 and pixels = 0. Press Step.
        </>
      </Translated>
    );
  else if (cur) {
    const where = `blurRow = ${cur.blurRow}, blurCol = ${cur.blurCol}: curRow = ${row} + ${cur.blurRow} = ${cur.curRow}, curCol = ${col} + ${cur.blurCol} = ${cur.curCol}. `;
    const skipped = cur.failed ? CLAUSES.length - 1 - CLAUSES.indexOf(cur.failed) : 0;
    say = cur.failed ? (
      <Translated
        vi={
          <>
            {where}
            <b>{CLAUSE_TEXT[cur.failed]} là false</b>, nên vị trí {at({ row: cur.curRow, col: cur.curCol })} bị bỏ qua
            {skipped ? ` và ${skipped === 1 ? 'mệnh đề' : `${skipped} mệnh đề`} sau nó không bao giờ được tính` : ''}.
            pixVal và pixels giữ nguyên ở {cur.pixVal} và {cur.pixels}.
          </>
        }
      >
        <>
          {where}
          <b>{CLAUSE_TEXT[cur.failed]} is false</b>, so position {at({ row: cur.curRow, col: cur.curCol })} is skipped
          {skipped ? ` and the ${skipped === 1 ? 'clause' : `${skipped} clauses`} after it ${skipped === 1 ? 'is' : 'are'} never evaluated` : ''}.
          pixVal and pixels stay at {cur.pixVal} and {cur.pixels}.
        </>
      </Translated>
    ) : (
      <Translated
        vi={
          <>
            {where}
            Cả bốn mệnh đề đều đúng. pixVal += in[{cur.curRow}·{w} + {cur.curCol}] = in[{cur.index}] = {cur.value}, cho ra{' '}
            {cur.pixVal}; pixels = {cur.pixels}.
          </>
        }
      >
        <>
          {where}
          All four clauses hold. pixVal += in[{cur.curRow}·{w} + {cur.curCol}] = in[{cur.index}] = {cur.value}, giving{' '}
          {cur.pixVal}; pixels = {cur.pixels}.
        </>
      </Translated>
    );
  } else
    say = (
      <Translated
        vi={
          <>
            Các vòng lặp đã xong: {t.pixels} trên 9 vị trí được giữ và {t.rejected} bị bỏ qua. Dòng 22: out[{row}·{w} +{' '}
            {col}] = out[{t.outIndex}] = (unsigned char)((float){t.pixVal}/{t.pixels}) = (unsigned char){' '}
            {Number.isInteger(t.quotient) ? t.quotient.toFixed(1) : t.quotient.toFixed(3)} = <b>{t.byte}</b>.
            {Number.isInteger(t.quotient) ? '' : ' Phép ép kiểu bỏ phần lẻ; nó không làm tròn.'}
          </>
        }
      >
        <>
          The loops are finished: {t.pixels} of 9 positions were kept and {t.rejected} skipped. Line 22: out[{row}·{w} +{' '}
          {col}] = out[{t.outIndex}] = (unsigned char)((float){t.pixVal}/{t.pixels}) = (unsigned char){' '}
          {Number.isInteger(t.quotient) ? t.quotient.toFixed(1) : t.quotient.toFixed(3)} = <b>{t.byte}</b>.
          {Number.isInteger(t.quotient) ? '' : ' The cast drops the fraction; it does not round.'}
        </>
      </Translated>
    );

  const hl = cur ? [12, 13, 15, ...(cur.inside ? [16, 17] : [])] : done ? [22] : [];
  return (
    <Bench
      title={tr('Step through one thread', 'Chạy từng bước một luồng')}
      task={tr('Pick a case, say how many positions will be kept, then step through the nine iterations.', 'Chọn một trường hợp, nói xem bao nhiêu vị trí sẽ được giữ, rồi chạy từng bước qua chín lần lặp.')}
      foot={tr('Hatched positions are outside the picture. The thick outline is the output pixel; amber cells have been read from in.', 'Các vị trí gạch chéo nằm ngoài ảnh. Viền dày là điểm ảnh đầu ra; các ô màu hổ phách đã được đọc từ in.')}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Điểm ảnh đầu ra">Output pixel</Translated>
          <select aria-label={tr('Case', 'Trường hợp')} value={k} onChange={(e) => { setK(+e.target.value); setN(0); }}>
            {PRESETS.map((p, i) => <option key={i} value={i}>{p.label} ({p.row}, {p.col})</option>)}
          </select>
        </label>
        <button onClick={() => setN(0)} disabled={n === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => setN(n - 1)} disabled={n === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => setN(n + 1)} disabled={done}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => setN(total)} disabled={done}><Translated vi="Chạy đến cuối">Run to the end</Translated></button>
        <span className="muted" aria-live="polite"><Translated vi={<>bước {n} trên {total}</>}>step {n} of {total}</Translated></span>
      </div>
      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }}>{say}</div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 13fr) minmax(0, 7fr)' }}>
        <div>
          <Code tabs={[{ label: tr('lines 10 to 22', 'dòng 10 đến 22'), code: LOOP, tiers: BLUR_TIERS }]} highlight={hl} firstLine={10} />
          <table className="threads" aria-label={tr('The inner test', 'Phép kiểm tra bên trong')}>
            <thead>
              <tr><th><Translated vi="mệnh đề của dòng 15">clause of line 15</Translated></th><th><Translated vi="lần lặp này">this iteration</Translated></th></tr>
            </thead>
            <tbody>
              {CLAUSES.map((cl, i) => {
                const v = cur ? cur.clauses[i] : undefined;
                return (
                  <tr key={cl} className={v === false ? 'cur' : ''} data-clause={cl} data-value={v === undefined ? '' : String(v)}>
                    <td>{CLAUSE_TEXT[cl]}</td>
                    <td>{v === undefined ? '' : v === null ? <span className="muted"><Translated vi="không được tính">not evaluated</Translated></span> : v ? 'true' : tr('false: skip', 'false: bỏ qua')}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div>
          <svg viewBox={`0 0 ${(w + 2) * 34} ${(h + 2) * 34}`} role="img" aria-label={tr(`Picture with the patch of output pixel ${at({ row, col })}`, `Ảnh với vùng của điểm ảnh đầu ra ${at({ row, col })}`)}
            style={{ width: '100%', maxWidth: (w + 2) * 34, display: 'block' }}>
            <Hatch id={hatch} />
            <PixelCells w={w} h={h} pad={1} cell={34} look={look} hatchId={hatch} />
          </svg>
          <table className="threads" aria-label={tr('Registers of this thread', 'Thanh ghi của luồng này')} style={{ marginTop: '0.6rem' }}>
            <thead>
              <tr><th>blurRow</th><th>blurCol</th><th>curRow</th><th>curCol</th><th>pixVal</th><th>pixels{/* i18n-ignore: variable name */}</th></tr>
            </thead>
            <tbody>
              <tr className="t-reg">
                <td>{cur ? cur.blurRow : ''}</td>
                <td>{cur ? cur.blurCol : ''}</td>
                <td>{cur ? cur.curRow : ''}</td>
                <td>{cur ? cur.curCol : ''}</td>
                <td data-reg="pixVal">{pixVal}</td>
                <td data-reg="pixels">{pixels}</td>
              </tr>
            </tbody>
          </table>
          {done ? (
            <p style={{ margin: '0.8rem 0 0' }}>
              <span className="chip ok">out[{/* i18n-ignore: code */}{t.outIndex}] = {t.byte}</span> <Translated vi={<>từ {t.pixVal}/{t.pixels}</>}>from {t.pixVal}/{t.pixels}</Translated>
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- boundary explorer
function BoundaryExplorer() {
  const hatch = useId();
  const [w, setW] = useState(8);
  const [h, setH] = useState(6);
  const [r, setR] = useState(1);
  const [sel, setSel] = useState<Pos>({ row: 2, col: 3 });
  const [map, setMap] = useState(false);
  const [drag, setDrag] = useState(false);
  const svg = useRef<SVGSVGElement>(null);

  const row = clampInt(sel.row, 0, h - 1);
  const col = clampInt(sel.col, 0, w - 1);
  const pic = useMemo(() => numbered(w, h), [w, h]);
  const t = useMemo(() => trace(pic, row, col, r), [pic, row, col, r]);
  const counts = useMemo(() => countMap(w, h, r), [w, h, r]);
  const A = area(r);
  const pad = r;
  const cell = 30;
  const kept = new Set(patch(row, col, w, h, r).filter((p) => p.inside).map((p) => p.curRow + ',' + p.curCol));
  const distinct = [...new Set(counts)].sort((a, b) => b - a);
  const reads = counts.reduce((a, b) => a + b, 0);

  const pickFrom = (e: PointerEvent<SVGSVGElement>) => {
    const b = svg.current!.getBoundingClientRect();
    setSel({
      row: clampInt(Math.floor(((e.clientY - b.top) / b.height) * (h + 2 * pad)) - pad, 0, h - 1),
      col: clampInt(Math.floor(((e.clientX - b.left) / b.width) * (w + 2 * pad)) - pad, 0, w - 1),
    });
  };
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const d: Record<string, [number, number]> = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
    const m = d[e.key];
    if (!m) return;
    e.preventDefault();
    setSel({ row: clampInt(row + m[0], 0, h - 1), col: clampInt(col + m[1], 0, w - 1) });
  };

  const look = (rr: number, cc: number, inside: boolean): CellLook | null => {
    const inPatch = Math.abs(rr - row) <= r && Math.abs(cc - col) <= r;
    if (!inside) return inPatch ? { hatch: true, dash: true, stroke: 'var(--ink-3)' } : null;
    const centre = rr === row && cc === col;
    const stroke = centre ? 'var(--ink)' : kept.has(rr + ',' + cc) ? 'var(--global)' : undefined;
    const strokeWidth = centre ? 3 : kept.has(rr + ',' + cc) ? 2 : 1;
    if (map) {
      const n = counts[rr * w + cc];
      const pct = Math.round(6 + 62 * (n / A));
      return { text: String(n), fill: `color-mix(in srgb, var(--ink) ${pct}%, var(--cell))`, ink: pct > 42 ? 'var(--page)' : 'var(--ink)', stroke, strokeWidth, bold: centre };
    }
    return { text: String(pic.data[rr * w + cc]), fill: kept.has(rr + ',' + cc) ? 'var(--global-wash)' : undefined, stroke, strokeWidth, bold: centre };
  };

  const rows = span(row, h, r);
  const cols = span(col, w, r);
  return (
    <Bench
      title={tr('Boundary explorer', 'Trình khám phá biên')}
      task={tr('Drag the output pixel to a corner, then raise BLUR_SIZE. Before each move, say how many values will be summed.', 'Kéo điểm ảnh đầu ra vào một góc, rồi tăng BLUR_SIZE. Trước mỗi lần di chuyển, hãy nói bao nhiêu giá trị sẽ được cộng.')}
      foot={tr('Drag, click, or focus the picture and use the arrow keys. Hatched positions are outside the picture and are skipped by the inner test.', 'Kéo, bấm, hoặc đưa tiêu điểm vào ảnh và dùng các phím mũi tên. Các vị trí gạch chéo nằm ngoài ảnh và bị phép kiểm tra bên trong bỏ qua.')}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi={<>Chiều cao h: {h}</>}>Height h: {h}</Translated>
          <input type="range" min="3" max="12" value={h} aria-label={tr('Height', 'Chiều cao')} onChange={(e) => setH(+e.target.value)} />
        </label>
        <label className="field">
          <Translated vi={<>Chiều rộng w: {w}</>}>Width w: {w}</Translated>
          <input type="range" min="3" max="12" value={w} aria-label={tr('Width', 'Chiều rộng')} onChange={(e) => setW(+e.target.value)} />
        </label>
        <label className="field">
          <Translated vi={<>BLUR_SIZE: {r} (vùng {2 * r + 1} × {2 * r + 1})</>}>BLUR_SIZE: {r} (patch {2 * r + 1} × {2 * r + 1})</Translated>
          <input type="range" min="1" max="3" value={r} aria-label="BLUR_SIZE" onChange={(e) => setR(+e.target.value)} />
        </label>
        <label className="check">
          <input type="checkbox" checked={map} onChange={(e) => setMap(e.target.checked)} /> <Translated vi="bản đồ số đếm">count map</Translated>
        </label>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 2fr)' }}>
        <div tabIndex={0} role="group" aria-label={tr('Picture: arrow keys move the output pixel', 'Ảnh: phím mũi tên di chuyển điểm ảnh đầu ra')} onKeyDown={onKey} data-explorer-grid>
          <svg ref={svg} viewBox={`0 0 ${(w + 2 * pad) * cell} ${(h + 2 * pad) * cell}`} role="img"
            aria-label={tr(`A ${h} by ${w} picture with the patch of output pixel ${at({ row, col })}`, `Một ảnh ${h} × ${w} với vùng của điểm ảnh đầu ra ${at({ row, col })}`)}
            style={{ width: '100%', maxWidth: (w + 2 * pad) * cell * 1.15, display: 'block', touchAction: 'none', cursor: 'pointer' }}
            onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); setDrag(true); pickFrom(e); }}
            onPointerMove={(e) => { if (drag) pickFrom(e); }}
            onPointerUp={() => setDrag(false)}
            onPointerCancel={() => setDrag(false)}>
            <Hatch id={hatch} />
            <PixelCells w={w} h={h} pad={pad} cell={cell} look={look} hatchId={hatch} />
          </svg>
        </div>
        <div className="panel" aria-live="polite" data-explorer-panel>
          <table className="threads" aria-label={tr('What the thread for this pixel does', 'Luồng của điểm ảnh này làm gì')}>
            <tbody>
              <tr><td><Translated vi="điểm ảnh đầu ra (row, col)">output pixel (row, col)</Translated></td><td data-k="pixel">{at({ row, col })}</td></tr>
              <tr>
                <td><Translated vi="các hàng được giữ">rows kept</Translated></td>
                <td data-k="rows"><Translated vi={<>{Math.max(0, row - r)} đến {Math.min(h - 1, row + r)}: {rows} trên {2 * r + 1}</>}>{Math.max(0, row - r)} to {Math.min(h - 1, row + r)}: {rows} of {2 * r + 1}</Translated></td>
              </tr>
              <tr>
                <td><Translated vi="các cột được giữ">columns kept</Translated></td>
                <td data-k="cols"><Translated vi={<>{Math.max(0, col - r)} đến {Math.min(w - 1, col + r)}: {cols} trên {2 * r + 1}</>}>{Math.max(0, col - r)} to {Math.min(w - 1, col + r)}: {cols} of {2 * r + 1}</Translated></td>
              </tr>
              <tr className="cur">
                <td><Translated vi="số giá trị được cộng">values summed</Translated></td>
                <td data-k="count"><Translated vi={<>{t.pixels} trên {A}</>}>{t.pixels} of {A}</Translated></td>
              </tr>
              <tr><td><Translated vi="dưới dạng tích">as a product</Translated></td><td data-k="product"><Translated vi={<>{rows} hàng × {cols} cột = {rows * cols}</>}>{rows} rows × {cols} columns = {rows * cols}</Translated></td></tr>
              <tr><td className="t-reg">pixVal</td><td data-k="sum" className="t-reg">{t.pixVal}</td></tr>
              <tr><td className="t-reg">pixels{/* i18n-ignore: variable name */}</td><td className="t-reg">{t.pixels}</td></tr>
              <tr>
                <td>(float)pixVal/pixels</td>
                <td data-k="quotient">{Number.isInteger(t.quotient) ? t.quotient : t.quotient.toFixed(3)}</td>
              </tr>
              <tr><td className="t-global"><Translated vi="byte được lưu">stored byte</Translated></td><td data-k="byte" className="t-global">{t.byte}</td></tr>
            </tbody>
          </table>
          <p style={{ margin: '0.8rem 0 0' }} data-k="summary">
            <Translated vi="Trên toàn ảnh:">Over the whole picture:</Translated>{' '}
            {distinct.map((n, i) => (
              <span key={n}>
                {i ? ', ' : ''}
                <Translated vi={<>{counts.filter((v) => v === n).length} luồng cộng {n}</>}>
                  {plural(counts.filter((v) => v === n).length, 'thread sums', 'threads sum')} {n}
                </Translated>
              </span>
            ))}
            <Translated vi={<>. Tổng cộng là {reads} lần đọc; sẽ là {w * h * A} nếu mọi luồng đều đọc trọn một vùng.</>}>
              . That is {reads} reads in all; {w * h * A} if every thread read a full patch.
            </Translated>
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- break the kernel
const BW = 6;
const BH = 7;
type Kind = 'texture' | 'spot' | 'grey';
const KIND_LABEL: Record<Kind, string> = {
  get texture() { return tr('slope with texture', 'dốc có vân'); },
  get spot() { return tr('one bright pixel', 'một điểm ảnh sáng'); },
  get grey() { return tr('uniform grey', 'xám đều'); },
};
const makePic = (kind: Kind, w = BW, h = BH): Picture => (kind === 'texture' ? textured(w, h) : kind === 'spot' ? spot(w, h) : uniform(w, h));

function Pic({ title, w, h, values, wrong, tier = true }: { title: string; w: number; h: number; values: number[]; wrong?: Set<number>; tier?: boolean }) {
  return (
    <div>
      <div className={'mx-label' + (tier ? ' t-global' : '')}>{title}</div>
      <svg viewBox={`0 0 ${w * 32} ${h * 32}`} role="img" aria-label={title} style={{ width: '100%', maxWidth: w * 34, display: 'block' }}>
        <PixelCells w={w} h={h} cell={32}
          look={(r, cc) => {
            const i = r * w + cc;
            const bad = wrong?.has(i);
            return { text: show(values[i]), ...grey(values[i]), stroke: bad ? 'var(--fail)' : undefined, strokeWidth: bad ? 3 : 1, bold: bad };
          }} />
      </svg>
    </div>
  );
}

function example(a: BadAccess, n: number): string {
  const who = tr(`thread ${at(a.thread)}`, `luồng ${at(a.thread)}`);
  if (a.op === 'read')
    return a.kind === 'wrapped'
      ? tr(`${who} read in[${a.index}]: it wanted position ${at(a.wanted)}, which is not in the picture, and got pixel ${at(a.got!)}.`, `${who} đọc in[${a.index}]: nó muốn vị trí ${at(a.wanted)}, vốn không nằm trong ảnh, và nhận được điểm ảnh ${at(a.got!)}.`)
      : tr(`${who} read in[${a.index}] for position ${at(a.wanted)}; the buffer has elements 0 to ${n - 1}.`, `${who} đọc in[${a.index}] cho vị trí ${at(a.wanted)}; buffer có các phần tử từ 0 đến ${n - 1}.`);
  return a.kind === 'wrapped'
    ? tr(`${who}, which owns no pixel, stored to out[${a.index}]: that is pixel ${at(a.got!)}, which belongs to another thread.`, `${who}, vốn không sở hữu điểm ảnh nào, đã ghi vào out[${a.index}]: đó là điểm ảnh ${at(a.got!)}, thuộc về một luồng khác.`)
    : tr(`${who}, which owns no pixel, stored to out[${a.index}]; the buffer has elements 0 to ${n - 1}.`, `${who}, vốn không sở hữu điểm ảnh nào, đã ghi vào out[${a.index}]; buffer có các phần tử từ 0 đến ${n - 1}.`);
}

function BreakTheKernel() {
  const [kind, setKind] = useState<Kind>('texture');
  const [o, setO] = useState<BlurOptions>(CORRECT);
  const pic = useMemo(() => makePic(kind), [kind]);
  const r = useMemo(() => blurImage(pic, o), [pic, o]);
  const byOrder = useMemo(() => ORDERS.map((order) => blurImage(pic, { ...o, order })), [pic, o]);
  const sameUnderAll = new Set(byOrder.map((x) => x.out.map(show).join())).size === 1;
  const tl = tally(r.accesses);
  const n = pic.w * pic.h;
  const first = (op: BadAccess['op'], k: BadAccess['kind']) => r.accesses.find((a) => a.op === op && a.kind === k);
  const untouched = JSON.stringify(o) === JSON.stringify(CORRECT);
  const setClause = (cl: Clause, on: boolean) => setO({ ...o, clauses: { ...o.clauses, [cl]: on } });
  const kinds: { n: number; cls: string; text: string; ex?: BadAccess }[] = [
    { n: tl.readWrapped, cls: 'read-wrapped', text: tr(`${plural(tl.readWrapped, 'read', 'reads')} fetched a wrong element (wrapped into another row)`, `${tl.readWrapped} lần đọc lấy sai phần tử (tràn sang một hàng khác)`), ex: first('read', 'wrapped') },
    { n: tl.readOutside, cls: 'read-outside', text: tr(`${plural(tl.readOutside, 'read', 'reads')} outside the buffer`, `${tl.readOutside} lần đọc ra ngoài buffer`), ex: first('read', 'outside') },
    { n: tl.writeWrapped, cls: 'write-wrapped', text: tr(`${plural(tl.writeWrapped, 'store', 'stores')} landed on another thread's pixel (wrapped into another row)`, `${tl.writeWrapped} lần ghi rơi vào điểm ảnh của luồng khác (tràn sang một hàng khác)`), ex: first('write', 'wrapped') },
    { n: tl.writeOutside, cls: 'write-outside', text: tr(`${plural(tl.writeOutside, 'store', 'stores')} outside the buffer`, `${tl.writeOutside} lần ghi ra ngoài buffer`), ex: first('write', 'outside') },
  ];
  return (
    <Bench
      title={tr('Break the kernel', 'Làm hỏng kernel')}
      task={tr('Untick one piece at a time. Before you look at the result: which pixels will be wrong, and will anything leave the buffer?', 'Bỏ chọn từng phần một. Trước khi nhìn kết quả: những điểm ảnh nào sẽ sai, và có gì ra ngoài buffer không?')}
      foot={
        <Translated
          vi={
            <>
              Một ảnh {BH} × {BW} ({BH} hàng, mỗi hàng {BW}) với các khối 4 × 4: {r.gridW} × {r.gridH} khối, {r.threads} luồng, trong đó {r.surplus}
              {' '}luồng dư. Sắc độ được kéo giãn để các giá trị thấp vẫn nhìn thấy được; các con số mới là dữ liệu. Dấu hỏi
              là một điểm ảnh không có giá trị xác định.
            </>
          }
        >
          <>
            A {BH} by {BW} picture ({BH} rows of {BW}) under 4 by 4 blocks: {r.gridW} by {r.gridH} blocks, {r.threads} threads, {r.surplus} of
            them surplus. Shades are stretched so that low values stay visible; the numbers are the data. A question mark
            is a pixel with no defined value.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          <Translated vi="Ảnh">Picture</Translated>
          <select aria-label={tr('Picture', 'Ảnh')} value={kind} onChange={(e) => setKind(e.target.value as Kind)}>
            {(Object.keys(KIND_LABEL) as Kind[]).map((kk) => <option key={kk} value={kk}>{KIND_LABEL[kk]}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Thứ tự luồng trong một khối">Thread order inside a block</Translated>
          <select aria-label={tr('Thread order', 'Thứ tự luồng')} value={o.order} onChange={(e) => setO({ ...o, order: e.target.value as Order })}>
            {ORDERS.map((x) => <option key={x} value={x}>{tr(ORDER_LABEL[x], x === 'forward' ? 'theo thứ tự' : x === 'reversed' ? 'đảo ngược' : 'xáo trộn')}</option>)}
          </select>
        </label>
        <button onClick={() => setO(CORRECT)} disabled={untouched}><Translated vi="Khôi phục kernel đúng">Restore the correct kernel</Translated></button>
      </div>
      <div className="row" style={{ marginBottom: '0.5rem', gap: '0.4rem 1.1rem' }} role="group" aria-label={tr('Pieces of the kernel', 'Các phần của kernel')}>
        <label className="check">
          <input type="checkbox" checked={o.outerGuard} onChange={(e) => setO({ ...o, outerGuard: e.target.checked })} />
          <span><Translated vi="dòng 05:">line 05:</Translated> <code>{'col < w && row < h'}</code></span>
        </label>
        {CLAUSES.map((cl) => (
          <label className="check" key={cl}>
            <input type="checkbox" checked={o.clauses[cl]} onChange={(e) => setClause(cl, e.target.checked)} />
            <span><Translated vi="dòng 15:">line 15:</Translated> <code>{CLAUSE_TEXT[cl]}</code></span>
          </label>
        ))}
        <label className="check">
          <input type="checkbox" checked={o.divisor === 'count'} onChange={(e) => setO({ ...o, divisor: e.target.checked ? 'count' : 'area' })} />
          <span><Translated vi={<>dòng 22: chia cho <code>pixels</code> (khi bỏ chọn: chia cho diện tích vùng, 9)</>}>line 22: divide by <code>pixels</code> (unticked: by the patch area, 9)</Translated></span>
        </label>
        <label className="check">
          <input type="checkbox" checked={!o.inPlace} onChange={(e) => setO({ ...o, inPlace: !e.target.checked })} />
          <span><Translated vi={<>một buffer <code>out</code> riêng (khi bỏ chọn: <code>out</code> chính là <code>in</code>)</>}>a separate <code>out</code> buffer (unticked: <code>out</code> is <code>in</code>)</Translated></span>
        </label>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', marginTop: '0.9rem' }}>
        <Pic title={tr('in, as the host filled it', 'in, như host đã điền')} w={pic.w} h={pic.h} values={pic.data} />
        <Pic title={o.inPlace ? tr('the one buffer after the run', 'buffer duy nhất sau khi chạy') : 'out'} w={pic.w} h={pic.h} values={r.out} wrong={new Set(r.wrong)} />
        <Pic title={tr('reference (the correct kernel)', 'tham chiếu (kernel đúng)')} w={pic.w} h={pic.h} values={r.reference} tier={false} />
      </div>
      <div className="panel" style={{ marginTop: '0.9rem' }} data-verdict={r.wrong.length ? 'wrong' : 'correct'}>
        <p style={{ margin: '0 0 0.5rem' }}>
          {r.wrong.length ? (
            <span className="chip fail"><Translated vi={<>{r.wrong.length} trên {n} điểm ảnh sai</>}>{r.wrong.length} of {n} pixels wrong</Translated></span>
          ) : (
            <span className="chip ok"><Translated vi="out khớp với tham chiếu">out matches the reference</Translated></span>
          )}{' '}
          {r.accesses.length === 0 && r.zeroDivisions.length === 0 && r.overwrittenReads === 0
            ? r.wrong.length
              ? tr('Every access was in bounds and no thread read what another had stored: only the arithmetic is wrong.', 'Mọi truy cập đều trong biên và không luồng nào đọc thứ luồng khác đã ghi: chỉ có phép tính là sai.')
              : tr('Every access was in bounds.', 'Mọi truy cập đều trong biên.')
            : null}
        </p>
        <ul style={{ margin: 0, paddingLeft: '1.2em' }}>
          {kinds.filter((x) => x.n > 0).map((x) => (
            <li key={x.cls} data-access={x.cls} data-n={x.n}>
              <b className="t-fail">{x.text}.</b> {x.ex ? tr('For example, ', 'Ví dụ, ') + example(x.ex, n) : ''}
            </li>
          ))}
          {r.zeroDivisions.length ? (
            <li data-access="zero-div" data-n={r.zeroDivisions.length}>
              <Translated
                vi={
                  <>
                    <b className="t-fail">{r.zeroDivisions.length} luồng đã chia 0 cho 0.</b> Ví dụ, luồng{' '}
                    {at(r.zeroDivisions[0])} không tìm thấy vị trí nào trong ảnh, nên pixVal và pixels đều bằng 0.
                  </>
                }
              >
                <b className="t-fail">{plural(r.zeroDivisions.length, 'thread', 'threads')} divided 0 by 0.</b> For example, thread{' '}
                {at(r.zeroDivisions[0])} found no position inside the picture, so pixVal and pixels were both 0.
              </Translated>
            </li>
          ) : null}
          {r.overwrittenReads ? (
            <li data-access="overwritten" data-n={r.overwrittenReads}>
              <Translated
                vi={
                  <>
                    <b className="t-fail">{r.overwrittenReads} lần đọc gặp một phần tử mà luồng khác đã ghi rồi.</b>{' '}
                    Tất cả đều nằm trong buffer.
                  </>
                }
              >
                <b className="t-fail">{plural(r.overwrittenReads, 'read', 'reads')} found an element that another thread had already stored.</b>{' '}
                All of them were inside the buffer.
              </Translated>
            </li>
          ) : null}
        </ul>
        <p style={{ margin: '0.6rem 0 0' }} data-orders={sameUnderAll ? 'same' : 'differ'}>
          <Translated vi="Với mỗi thứ tự luồng:">Under each thread order:</Translated>{' '}
          {byOrder.map((x, i) => (
            <span key={ORDERS[i]} data-order={ORDERS[i]} data-wrong={x.wrong.length}>
              {i ? ', ' : ''}
              <Translated vi={<>{tr(ORDER_LABEL[ORDERS[i]], ORDERS[i] === 'forward' ? 'theo thứ tự' : ORDERS[i] === 'reversed' ? 'đảo ngược' : 'xáo trộn')} {x.wrong.length} sai</>}>
                {ORDER_LABEL[ORDERS[i]]} {x.wrong.length} wrong
              </Translated>
            </span>
          ))}
          . {sameUnderAll ? tr('The three outputs are identical.', 'Ba đầu ra giống hệt nhau.') : <b><Translated vi="Ba đầu ra khác nhau.">The three outputs differ from each other.</Translated></b>}
        </p>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- one buffer, three orders
function ThreeOrders() {
  const [kind, setKind] = useState<Kind>('spot');
  const [inPlace, setInPlace] = useState(true);
  const pic = useMemo(() => makePic(kind, 6, 6), [kind]);
  const runs = useMemo(
    () => ORDERS.map((order) => blurImage(pic, { ...CORRECT, blockW: 6, blockH: 6, inPlace, order })),
    [pic, inPlace],
  );
  const same = new Set(runs.map((x) => x.out.join())).size === 1;
  return (
    <Bench
      title={tr('One buffer, three thread orders', 'Một buffer, ba thứ tự luồng')}
      task={tr('The kernel is correct apart from the buffer. Untick the box: what do you expect the three pictures to do?', 'Kernel đúng, chỉ trừ buffer. Bỏ chọn ô: bạn nghĩ ba ảnh sẽ thế nào?')}
      foot={tr('A 6 by 6 picture covered by one block of 6 by 6 threads, the configuration of the animation above.', 'Một ảnh 6 × 6 được phủ bởi một khối 6 × 6 luồng, đúng cấu hình của hoạt hình ở trên.')}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Ảnh">Picture</Translated>
          <select aria-label={tr('Picture', 'Ảnh')} value={kind} onChange={(e) => setKind(e.target.value as Kind)}>
            {(['spot', 'texture'] as Kind[]).map((kk) => <option key={kk} value={kk}>{KIND_LABEL[kk]}</option>)}
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={inPlace} onChange={(e) => setInPlace(e.target.checked)} /> <span><Translated vi={<><code>out</code> là cùng buffer với <code>in</code></>}><code>out</code> is the same buffer as <code>in</code></Translated></span>
        </label>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'repeat(4, minmax(0, 1fr))' }}>
        <Pic title={tr('reference', 'tham chiếu')} w={6} h={6} values={runs[0].reference} tier={false} />
        {runs.map((x, i) => (
          <div key={ORDERS[i]} data-order={ORDERS[i]} data-wrong={x.wrong.length}>
            <Pic title={tr(`threads ${ORDER_LABEL[ORDERS[i]]}`, `luồng ${ORDERS[i] === 'forward' ? 'theo thứ tự' : ORDERS[i] === 'reversed' ? 'đảo ngược' : 'xáo trộn'}`)} w={6} h={6} values={x.out} wrong={new Set(x.wrong)} />
            <p style={{ margin: '0.4rem 0 0' }}>
              {x.wrong.length ? <span className="chip fail"><Translated vi={<>{x.wrong.length} trên 36 sai</>}>{x.wrong.length} of 36 wrong</Translated></span> : <span className="chip ok"><Translated vi="khớp">matches</Translated></span>}
            </p>
          </div>
        ))}
      </div>
      <p style={{ margin: '0.8rem 0 0' }} data-same={same ? 'yes' : 'no'}>
        {same
          ? tr('The three orders give the same picture: no thread reads anything that another thread writes.', 'Ba thứ tự cho cùng một ảnh: không luồng nào đọc thứ mà luồng khác ghi.')
          : tr('Three orders, three different pictures, and not one access outside the buffer.', 'Ba thứ tự, ba ảnh khác nhau, và không một truy cập nào ra ngoài buffer.')}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- name the defect
const WHY: Record<Exclude<Defect, 'outer'>, string> = {
  get rowLo() {
    return tr('Only the top row is wrong. Its threads read the row above the picture, which lies before the start of the buffer (the model reads 0 there, so the values are too low).', 'Chỉ hàng trên cùng sai. Các luồng của nó đọc hàng phía trên ảnh, vốn nằm trước đầu buffer (mô hình đọc 0 ở đó, nên các giá trị quá thấp).');
  },
  get rowHi() {
    return tr('Only the bottom row is wrong. Its threads read a row below the picture, past the end of the buffer.', 'Chỉ hàng dưới cùng sai. Các luồng của nó đọc một hàng bên dưới ảnh, quá cuối buffer.');
  },
  get colLo() {
    return tr('Only the left-hand column is wrong. in[curRow*w - 1] is the last pixel of the row above, so each of those threads averaged in values from the far side of the picture.', 'Chỉ cột bên trái sai. in[curRow*w - 1] là điểm ảnh cuối của hàng phía trên, nên mỗi luồng đó đã lấy trung bình cả những giá trị từ phía bên kia của ảnh.');
  },
  get colHi() {
    return tr('Only the right-hand column is wrong. in[curRow*w + w] is the first pixel of the next row.', 'Chỉ cột bên phải sai. in[curRow*w + w] là điểm ảnh đầu tiên của hàng kế tiếp.');
  },
  get area() {
    return tr('The whole border is too low and the interior is right: a border thread summed 4 or 6 values and divided by 9.', 'Toàn bộ viền quá thấp còn bên trong thì đúng: một luồng ở viền cộng 4 hoặc 6 giá trị rồi chia cho 9.');
  },
  get inPlace() {
    return tr('Wrong all over, with no pattern tied to one edge: threads read neighbours that had already been replaced by averages.', 'Sai khắp nơi, không có quy luật gắn với một mép nào: các luồng đọc những điểm lân cận đã bị thay bằng giá trị trung bình.');
  },
};

/** "4", "5 and 6", "1, 2, 3, 5 and 6". */
const listOf = (xs: number[]) => (xs.length < 2 ? xs.join('') : tr(`${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`, `${xs.slice(0, -1).join(', ')} và ${xs[xs.length - 1]}`));

/**
 * The explanation for a missing line 05, built from the pixels the model marks wrong, so that
 * the rows it names are always the rows in the picture beside it.
 */
function whyOuter(wrong: number[], w: number): string {
  const { rows, cols, spared } = wrongSpan(wrong, w);
  if (!rows.length) return tr('Nothing is wrong in this picture: every surplus store landed outside the buffer or was overwritten later.', 'Không có gì sai trong ảnh này: mọi lần ghi dư đều rơi ra ngoài buffer hoặc bị ghi đè sau đó.');
  const extra = Array.from({ length: cols.length }, (_, k) => w + k);
  return tr(
    `${cols.length === 1 ? 'Column' : 'Columns'} ${listOf(cols)} ${cols.length === 1 ? 'is' : 'are'} wrong in rows ${listOf(rows)}. ` +
    `The surplus threads at col ${listOf(extra)} stored to out[row*${w} + col], the start of the next row. ` +
    (spared.length
      ? `${spared.length === 1 ? 'Row' : 'Rows'} ${listOf(spared)} ${spared.length === 1 ? 'survives' : 'survive'} only because ${spared.length === 1 ? 'its' : 'their'} own block runs after the block that wrote over ${spared.length === 1 ? 'it' : 'them'}. `
      : '') +
    'A question mark is a thread that found no valid position and divided 0 by 0.',
    `${cols.length === 1 ? 'Cột' : 'Các cột'} ${listOf(cols)} sai ở các hàng ${listOf(rows)}. ` +
      `Các luồng dư ở col ${listOf(extra)} đã ghi vào out[row*${w} + col], tức đầu hàng kế tiếp. ` +
      (spared.length
        ? `${spared.length === 1 ? 'Hàng' : 'Các hàng'} ${listOf(spared)} còn đúng chỉ vì khối của chính ${spared.length === 1 ? 'nó' : 'chúng'} chạy sau khối đã ghi đè lên ${spared.length === 1 ? 'nó' : 'chúng'}. `
        : '') +
      'Dấu hỏi là một luồng không tìm thấy vị trí hợp lệ nào và đã chia 0 cho 0.',
  );
}
const why = (d: Defect, wrong: number[]) => (d === 'outer' ? whyOuter(wrong, BW) : WHY[d]);

function NameTheDefect() {
  const [seed, setSeed] = useState(1);
  const [picks, setPicks] = useState<(Defect | '')[]>(['', '', '']);
  const [checked, setChecked] = useState(false);
  const pic = useMemo(() => textured(BW, BH), []);
  const cases = useMemo(() => challenge(seed).map((d) => ({ d, r: blurImage(pic, withDefect(d)) })), [seed, pic]);
  const right = cases.filter((x, i) => picks[i] === x.d).length;
  return (
    <Bench
      title={tr('Name the defect', 'Gọi tên lỗi')}
      task={tr('Each picture came from the kernel with exactly one thing wrong. Where the wrong pixels are tells you which.', 'Mỗi ảnh đến từ kernel có đúng một chỗ sai. Vị trí của các điểm ảnh sai cho bạn biết đó là chỗ nào.')}
      foot={tr(`The same ${BH} by ${BW} picture (${BH} rows of ${BW}), 4 by 4 blocks and in-order threads as the bench above. Red outlines mark pixels that differ from the reference.`, `Cùng ảnh ${BH} × ${BW} (${BH} hàng, mỗi hàng ${BW}), khối 4 × 4 và luồng theo thứ tự như bảng tương tác ở trên. Viền đỏ đánh dấu các điểm ảnh khác với tham chiếu.`)}
    >
      <div className="cols" style={{ gridTemplateColumns: 'repeat(4, minmax(0, 1fr))' }}>
        <Pic title={tr('reference', 'tham chiếu')} w={BW} h={BH} values={cases[0].r.reference} tier={false} />
        {cases.map((x, i) => (
          <div key={seed + '-' + i} data-case={i}>
            <Pic title={tr(`picture ${i + 1}`, `ảnh ${i + 1}`)} w={BW} h={BH} values={x.r.out} wrong={new Set(x.r.wrong)} />
            <label className="field" style={{ marginTop: '0.5rem', display: 'flex' }}>
              <Translated vi="Chỗ nào sai?">What is wrong?</Translated>
              <select aria-label={tr(`Defect behind picture ${i + 1}`, `Lỗi đằng sau ảnh ${i + 1}`)} value={picks[i]} style={{ width: '100%' }}
                onChange={(e) => { setPicks(picks.map((p, j) => (j === i ? (e.target.value as Defect | '') : p))); setChecked(false); }}>
                <option value="">{tr('choose', 'chọn')}</option>
                {DEFECTS.map((d) => <option key={d} value={d}>{tr(DEFECT_LABEL[d], d === 'rowLo' ? 'thiếu curRow >= 0' : d === 'rowHi' ? 'thiếu curRow < h' : d === 'colLo' ? 'thiếu curCol >= 0' : d === 'colHi' ? 'thiếu curCol < w' : d === 'area' ? 'số chia là diện tích vùng' : d === 'outer' ? 'thiếu phép kiểm tra ngoài (dòng 05)' : 'out là cùng buffer với in')}</option>)}
              </select>
            </label>
            {checked ? (
              <p style={{ margin: '0.5rem 0 0' }}>
                {picks[i] === x.d ? <span className="chip ok"><Translated vi="đúng">right</Translated></span> : <span className="chip fail"><Translated vi={<>thực ra là: {tr(DEFECT_LABEL[x.d], x.d === 'rowLo' ? 'thiếu curRow >= 0' : x.d === 'rowHi' ? 'thiếu curRow < h' : x.d === 'colLo' ? 'thiếu curCol >= 0' : x.d === 'colHi' ? 'thiếu curCol < w' : x.d === 'area' ? 'số chia là diện tích vùng' : x.d === 'outer' ? 'thiếu phép kiểm tra ngoài (dòng 05)' : 'out là cùng buffer với in')}</>}>it was: {DEFECT_LABEL[x.d]}</Translated></span>}{' '}
                {why(x.d, x.r.wrong)}
              </p>
            ) : null}
          </div>
        ))}
      </div>
      <div className="row" style={{ marginTop: '0.9rem' }}>
        <button className="primary" disabled={picks.some((p) => p === '')} onClick={() => setChecked(true)}><Translated vi="Kiểm tra">Check</Translated></button>
        <button onClick={() => { setSeed(seed + 1); setPicks(['', '', '']); setChecked(false); }}><Translated vi="Bộ mới">New set</Translated></button>
        {checked ? <span aria-live="polite" data-score={right}><Translated vi={<>Gọi tên đúng {right} trên 3.</>}>{right} of 3 named correctly.</Translated></span> : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS33: Record<string, Question> = {
  'ch03/3.3/q-corner7': {
    kind: 'numeric',
    id: 'ch03/3.3/q-corner7',
    prompt: (
      <Translated
        vi={
          <>
            <code>BLUR_SIZE = 3</code> trên một ảnh 100 × 100. Luồng của điểm ảnh đầu ra (0, 0) lấy trung bình bao nhiêu
            giá trị?
          </>
        }
      >
        <>
          <code>BLUR_SIZE = 3</code> on a 100 by 100 picture. How many values does the thread for output pixel (0, 0)
          average?
        </>
      </Translated>
    ),
    answer: 16,
    explain: (
      <Translated
        vi={
          <p>
            Các vòng lặp yêu cầu các hàng −3 đến 3 và các cột −3 đến 3, 49 vị trí. Các hàng −3 đến −1 và các cột −3 đến −1 không
            tồn tại, còn lại các hàng 0 đến 3 và các cột 0 đến 3: 4 × 4 = 16. Số đếm luôn là một thừa số theo hàng nhân một thừa số
            theo cột, nên bạn không bao giờ cần liệt kê cả vùng.
          </p>
        }
      >
        <p>
          The loops ask for rows −3 to 3 and columns −3 to 3, 49 positions. Rows −3 to −1 and columns −3 to −1 do not
          exist, which leaves rows 0 to 3 and columns 0 to 3: 4 × 4 = 16. The count is always a row term times a column
          term, so you never need to enumerate the patch.
        </p>
      </Translated>
    ),
    source: 'ch. 3 §3.3, pp. 58–59, applied to a 7 by 7 patch',
    vi: { source: 'ch. 3 §3.3, pp. 58–59, áp dụng cho một vùng 7 × 7' },
  },
  'ch03/3.3/q-edge': {
    kind: 'numeric',
    id: 'ch03/3.3/q-edge',
    prompt: (
      <Translated
        vi={
          <>
            <code>BLUR_SIZE = 2</code>, và ảnh rộng 8 cột, cao 6 hàng. Luồng của điểm ảnh đầu ra (hàng 0, cột 5)
            lấy trung bình bao nhiêu giá trị?
          </>
        }
      >
        <>
          <code>BLUR_SIZE = 2</code>, and the picture is 8 columns wide and 6 rows high. How many values does the thread
          for output pixel (row 0, col 5) average?
        </>
      </Translated>
    ),
    answer: 15,
    explain: (
      <Translated
        vi={
          <p>
            Các hàng −2 đến 2 được yêu cầu và các hàng 0 đến 2 tồn tại: 3 hàng. Các cột 3 đến 7 được yêu cầu và ảnh có
            các cột 0 đến 7, nên cả 5 đều tồn tại. 3 × 5 = 15. Phía cột là phía cần kiểm tra hai lần: cột 7 là cột cuối
            cùng, nên vùng chạm vào mép phải mà không vượt qua nó.
          </p>
        }
      >
        <p>
          Rows −2 to 2 are requested and rows 0 to 2 exist: 3 rows. Columns 3 to 7 are requested and the picture has
          columns 0 to 7, so all 5 exist. 3 × 5 = 15. The column side is the one to check twice: column 7 is the last
          one, so the patch touches the right-hand edge without crossing it.
        </p>
      </Translated>
    ),
    source: 'ch. 3 §3.3, pp. 58–59',
  },
  'ch03/3.3/q-area': {
    kind: 'choice',
    id: 'ch03/3.3/q-area',
    prompt: (
      <Translated
        vi={
          <>
            Ai đó đơn giản hóa dòng cuối để chia cho <code>(2*BLUR_SIZE+1)*(2*BLUR_SIZE+1)</code> thay vì{' '}
            <code>pixels</code>. Rồi kernel làm mờ một ảnh xám đều. Kết quả ra sao?
          </>
        }
      >
        <>
          Someone simplifies the last line to divide by <code>(2*BLUR_SIZE+1)*(2*BLUR_SIZE+1)</code> instead of{' '}
          <code>pixels</code>. The kernel then blurs a uniformly grey picture. What comes out?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Mọi điểm ảnh đều tối hơn, bên trong cũng tối như viền.">Every pixel is darker, the interior as much as the border.</Translated>,
        why: <Translated vi="Một luồng bên trong đúng là đã cộng 9 giá trị, nên chia cho 9 là đúng với nó.">An interior thread did sum 9 values, so dividing by 9 is right for it.</Translated>,
      },
      { label: <Translated vi="Bên trong không đổi; viền tối hơn, và các góc tối nhất.">The interior is unchanged; the border is darker, and the corners are darkest.</Translated>, correct: true },
      {
        label: <Translated vi="Ảnh không đổi, vì mọi giá trị đều bằng nhau.">The picture is unchanged, because all the values are equal.</Translated>,
        why: <Translated vi="Các giá trị bằng nhau làm tổng ở viền bằng 4 hoặc 6 lần mức xám, rồi tổng đó bị chia cho 9.">Equal values make the sum 4 or 6 times the grey level at the border, and that is then divided by 9.</Translated>,
      },
      {
        label: <Translated vi="Rác ở viền, do đọc ra ngoài ảnh.">Garbage at the border, from reads outside the picture.</Translated>,
        why: <Translated vi="Phép kiểm tra bên trong vẫn nguyên, nên không có gì ngoài ảnh bị đọc. Chỉ số chia thay đổi.">The inner test is untouched, so nothing outside the picture is read. Only the divisor changed.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Với <code>BLUR_SIZE</code> bằng 1 và mức xám g, một luồng bên trong lưu 9g/9 = g, một luồng ở mép lưu 6g/9 và một
            luồng ở góc lưu 4g/9. Mọi truy cập đều trong biên và không luồng nào đọc thứ luồng khác ghi, nên không công cụ theo dõi
            bộ nhớ nào nhận ra. Đó là lý do số chia là bộ đếm chứ không phải diện tích vùng.
          </p>
        }
      >
        <p>
          With <code>BLUR_SIZE</code> 1 and grey level g, an interior thread stores 9g/9 = g, an edge thread 6g/9 and a
          corner thread 4g/9. Every access is in bounds and no thread reads what another writes, so no tool that watches
          memory will notice. That is why the divisor is the counter and not the patch area.
        </p>
      </Translated>
    ),
    source: 'ch. 3 §3.3, p. 59',
  },
  'ch03/3.3/q-twoguards': {
    kind: 'choice',
    id: 'ch03/3.3/q-twoguards',
    prompt: (
      <Translated
        vi={
          <>
            Phép kiểm tra ở dòng 05, <code>{'col < w && row < h'}</code>, bị bỏ đi. Ảnh có 62 hàng × 76 cột, các
            khối là 16 × 16, và <code>BLUR_SIZE</code> bằng 1. Điều gì hỏng?
          </>
        }
      >
        <>
          The test on line 05, <code>{'col < w && row < h'}</code>, is removed. The picture is 62 rows by 76 columns, the
          blocks are 16 by 16, and <code>BLUR_SIZE</code> is 1. What goes wrong?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không có gì. Phép kiểm tra bên trong ở dòng 15 loại mọi vị trí ngoài ảnh, cho cả luồng dư, nên chúng không gây hại.">
            Nothing. The inner test on line 15 rejects every position outside the picture, for surplus threads too, so they do no harm.
          </Translated>
        ),
        why: (
          <Translated vi="Đúng là nó làm vậy, cho các lần đọc. Lệnh ghi ở dòng 22 không nằm trong phép kiểm tra đó.">
            It does, for the reads. The store on line 22 is not inside that test.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="408 luồng dư cũng ghi, mỗi luồng vào out[row*w + col]: lên điểm ảnh của luồng khác hoặc quá cuối out.">
            The 408 surplus threads also store, each at out[row*w + col]: on another thread's pixel or past the end of out.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Mỗi luồng trong 408 luồng dư không tìm thấy vị trí hợp lệ nào, chia cho 0, và lần khởi chạy dừng với một lỗi.">
            Each of the 408 surplus threads finds no valid position, divides by zero, and the launch stops with an error.
          </Translated>
        ),
        why: (
          <Translated vi="Một số luồng tìm thấy vị trí hợp lệ và chia cho 1, 2 hoặc 3. Số còn lại chia 0 cho 0 bằng dấu phẩy động, điều đó không làm dừng gì cả.">
            Some of them find valid positions and divide by 1, 2 or 3. The others divide 0 by 0 in floating point, which stops nothing.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Một lỗi biên dịch, vì row và col được dùng mà không kiểm tra.">A compile error, because row and col are used unchecked.</Translated>,
        why: <Translated vi="Trình biên dịch không hề biết các buffer lớn đến đâu.">The compiler has no idea how large the buffers are.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
            Lưới có 64 × 80 luồng cho 62 × 76 điểm ảnh: 5120 − 4712 = 408 luồng dư. Phép kiểm tra bên trong bảo vệ
            các lần đọc của mọi luồng, dư hay không. Nó không bảo vệ lệnh ghi.
            </p>
            <p>
            139 luồng dư nằm sát ảnh và vẫn tìm thấy 1 đến 3 vị trí hợp lệ; chúng lưu một giá trị trung bình thật
            vào sai chỗ. 269 luồng còn lại không tìm thấy vị trí nào: pixVal và pixels đều bằng 0, phép chia float cho ra
            not-a-number (NaN), và chuyển giá trị đó sang byte là hành vi không xác định. Còn về chỗ các lần ghi rơi vào: 244 lần rơi vào các cột 0 đến
            3 của hàng kế tiếp, đè lên các điểm ảnh mà luồng khác sở hữu, và 164 lần rơi quá cuối <code>out</code>.
            </p>
          </>
        }
      >
        <>
          <p>
            The grid is 64 by 80 threads for 62 by 76 pixels: 5120 − 4712 = 408 surplus threads. The inner test protects
            the reads of every thread, surplus or not. It does not protect the store.
          </p>
          <p>
            139 of the surplus threads sit next to the picture and still find 1 to 3 valid positions; they store a real
            average in the wrong place. The other 269 find none: pixVal and pixels are both 0, the float division gives
            not-a-number, and converting that to a byte is undefined. As for where the stores go: 244 land on columns 0 to
            3 of the next row, on top of pixels that other threads own, and 164 land past the end of <code>out</code>.
          </p>
        </>
      </Translated>
    ),
    source: 'our analysis of Fig. 3.8 (ch. 3 §3.3, p. 57); the counts come from the tested model',
    vi: { source: 'phân tích của chúng tôi về Fig. 3.8 (ch. 3 §3.3, p. 57); các số đếm lấy từ mô hình đã kiểm thử' },
  },
  'ch03/3.3/q-inplace': {
    kind: 'choice',
    id: 'ch03/3.3/q-inplace',
    prompt: (
      <Translated
        vi={
          <>
            Host truyền cùng một buffer làm <code>in</code> và làm <code>out</code>. Phát biểu nào đúng?
          </>
        }
      >
        <>
          The host passes the same buffer as <code>in</code> and as <code>out</code>. Which statement holds?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Kết quả vẫn đúng: mỗi luồng chỉ ghi điểm ảnh của riêng nó, và không có hai luồng nào ghi cùng một điểm ảnh.">The result is still correct: each thread writes only its own pixel, and no two threads write the same one.</Translated>,
        why: <Translated vi="Mỗi luồng ghi một điểm ảnh, nhưng đọc tới chín, và tám trong số đó giờ là điểm ảnh mà luồng khác ghi.">Each thread writes one pixel, but it reads up to nine, and eight of those are now pixels that other threads write.</Translated>,
      },
      {
        label: <Translated vi="Kết quả sai, nhưng mọi lần chạy đều cho cùng một ảnh sai, cố định bởi thứ tự các khối.">The result is wrong, but every run produces the same wrong picture, fixed by the order of the blocks.</Translated>,
        why: <Translated vi="Sẽ đúng như vậy nếu các luồng chạy theo một thứ tự cố định. Không có gì cố định thứ tự đó.">It would be if the threads ran in a fixed order. Nothing fixes it.</Translated>,
      },
      { label: <Translated vi="Nó phụ thuộc vào thứ tự các luồng: một số luồng đọc những điểm lân cận đã bị ghi đè.">It depends on the order of the threads: some read neighbours already overwritten.</Translated>, correct: true },
      {
        label: <Translated vi="Kernel không biên dịch được, vì in và out là cùng một con trỏ.">The kernel fails to compile, because in and out are the same pointer.</Translated>,
        why: <Translated vi="Hai tham số con trỏ có thể giữ cùng một địa chỉ; trình biên dịch không kiểm tra.">Two pointer parameters may hold the same address; the compiler does not check.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Với hai buffer, mỗi luồng ghi một phần tử mà không ai đọc, và đọc các phần tử mà không ai ghi, nên
            thứ tự không thể ảnh hưởng. Với một buffer, việc một luồng thấy giá trị gốc của điểm lân cận hay giá trị trung bình của nó
            phụ thuộc vào luồng nào trong hai luồng chạy trước. Hoạt hình cho thấy hai thứ tự cho ra hai ảnh sai. Một GPU chạy các
            luồng cùng lúc, theo một thứ tự mà bạn không chọn được và cũng không lặp lại được.
          </p>
        }
      >
        <p>
          With two buffers each thread writes one element that nobody reads, and reads elements that nobody writes, so
          the order cannot matter. With one buffer, whether a thread sees its neighbour's original value or its average
          depends on which of the two ran first. The animation shows two orders giving two wrong pictures. A GPU runs the
          threads at the same time, in an order you neither choose nor can repeat.
        </p>
      </Translated>
    ),
    source: 'ours; built on the one-to-one mapping of threads to output elements (ch. 3 §3.3, p. 56, and §3.4, p. 60)',
    vi: { source: 'của khóa học này; dựa trên ánh xạ một-một giữa luồng và phần tử đầu ra (ch. 3 §3.3, p. 56, và §3.4, p. 60)' },
  },
  'ch03/3.3/q-maskedmean': {
    kind: 'choice',
    id: 'ch03/3.3/q-maskedmean',
    prompt: (
      <Translated vi={<>Phép tính nào trong framework có cùng dạng với các dòng 10 đến 22 của kernel làm mờ?</>}>
        <>Which framework computation has the same shape as lines 10 to 22 of the blur kernel?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="softmax trên chiều cuối của các điểm số">softmax over the last dimension of the scores</Translated>,
        why: <Translated vi="Nó chuẩn hóa bằng một tổng các hàm mũ, không phải bằng số vị trí được cho phép.">It normalises by a sum of exponentials, not by a count of the positions that were allowed in.</Translated>,
      },
      { label: <Translated vi="mean pooling trên các token không bị che (mask) của một batch có đệm (padding)">mean pooling over the unmasked tokens of a padded batch</Translated>, correct: true },
      {
        label: <Translated vi="chuyển vị từng lát [seq, hidden]">a transpose of each [seq, hidden] slice</Translated>,
        why: <Translated vi="Đó thuần là ánh xạ chỉ số, một phần tử vào và một ra: dạng của kernel chuyển ảnh xám.">That is pure index mapping, one element in and one out: the grayscale kernel's shape.</Translated>,
      },
      {
        label: <Translated vi="phép cộng residual của hai hidden state">a residual add of two hidden states</Translated>,
        why: <Translated vi="Mỗi đầu vào một phần tử vào và một ra, không có vòng lặp và không có phép kiểm tra.">One element in from each input and one out, with no loop and no test.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Masked mean pooling cộng các vector token mà mask cho phép và chia cho số lượng của chúng. Đó đúng là
            vòng lặp của phép làm mờ: tích lũy ở nơi phép kiểm tra qua, đếm số lần qua, chia cho số đếm. Mask đệm (padding mask)
            đóng vai phép kiểm tra mép, và chia cho độ dài đã đệm thay vì số đếm là cùng một sai lầm như
            chia cho diện tích vùng.
          </p>
        }
      >
        <p>
          Masked mean pooling sums the token vectors that the mask allows and divides by how many there were. That is the
          blur's loop exactly: accumulate where a test passes, count the passes, divide by the count. The padding mask
          plays the part of the edge test, and dividing by the padded length instead of the count is the same mistake as
          dividing by the patch area.
        </p>
      </Translated>
    ),
    source: 'ours',
    vi: { source: 'của khóa học này' },
  },
};

// ---------------------------------------------------------------- the page
const missing = (s: string | null) => !s || s.startsWith('// could not load');
const LAB2 = 'ch03/lab2_blur';

export function S33() {
  const cu = useLabFile(LAB2 + '/solution/kernel.cu');
  const metal = useLabFile(LAB2 + '/solution/kernel.metal');
  const mm = useLabFile(LAB2 + '/solution/launch.mm');
  // The status of the reference solution in the labs page's words, read from this machine's lab runs.
  const { reports } = useReports();
  const cudaStatus = solutionTabStatus(reports, LAB2, 'kernel.cu');
  const metalStatus = solutionTabStatus(reports, LAB2, 'kernel.metal');
  return (
    <>
      <Translated
        vi={
          <p>
            Kernel chuyển ảnh xám ở mục trước làm vài phép tính số học trên một phần tử. Phần lớn các kernel đáng
            viết làm nhiều hơn: mỗi luồng đọc một vùng lân cận, một hàng, một cửa sổ token. Làm mờ ảnh là ví dụ nhỏ nhất
            của điều đó. Mỗi điểm ảnh đầu ra là trung bình của các điểm ảnh đầu vào quanh nó (chính điều này khử nhiễu và
            làm dịu ảnh), nên giờ một luồng chứa một vòng lặp.
          </p>
        }
      >
        <p>
          The grayscale kernel of the last section did a few arithmetic operations on one element. Most kernels worth
          writing do more: each thread reads a neighbourhood, a row, a window of tokens. Image blur is the smallest
          example of that. Each output pixel is an average of the input pixels around it (which is what removes noise and
          softens a picture), so a thread now contains a loop.
        </p>
      </Translated>
      <Source section="3.3" pages="55–56" figs="Figs. 3.6, 3.7" />
      <Translated
        vi={
          <p>
            Phiên bản của sách là phiên bản đơn giản nhất có thể: điểm ảnh đầu ra là trung bình không trọng số của một vùng vuông
            của đầu vào, tâm đặt tại cùng vị trí, tính cả tâm. Với vùng 3 × 3, điểm ảnh đầu ra (25, 50) là
            trung bình của chín điểm ảnh đầu vào ở các hàng 24 đến 26 và các cột 49 đến 51. Cho chín vị trí các trọng số
            khác nhau là bạn có một phép tích chập (convolution), chủ đề của chương 7.
          </p>
        }
      >
        <p>
          The book's version is the plainest one possible: the output pixel is the unweighted average of a square patch
          of the input centred on the same position, the centre included. With a 3 by 3 patch, output pixel (25, 50) is
          the average of the nine input pixels in rows 24 to 26 and columns 49 to 51. Give the nine positions different
          weights and you have a convolution, which is chapter 7.
        </p>
      </Translated>
      <Source section="3.3" pages="56" />

      <h2><Translated vi="Phép ánh xạ vẫn thế; giờ luồng có vòng lặp">The mapping is the same; the thread now loops</Translated></h2>
      <Translated
        vi={
          <p>
            Cách các luồng tìm việc không thay đổi. Mỗi điểm ảnh đầu ra một luồng, <code>col</code> lấy
            từ các chỉ số x và <code>row</code> từ các chỉ số y, và cùng phép kiểm tra đó loại các luồng dư của
            một lưới đã được làm tròn lên thành số nguyên khối. Cái mới nằm bên trong phép kiểm tra đó: hai vòng lặp lồng nhau đi qua
            vùng, một tổng tích lũy, và một bộ đếm.
          </p>
        }
      >
        <p>
          Nothing changes in how threads find their work. There is one thread per output pixel, <code>col</code> comes
          from the x indices and <code>row</code> from the y indices, and the same test turns away the surplus threads of
          a grid that was rounded up to whole blocks. What is new sits inside that test: two nested loops that walk the
          patch, a running sum, and a counter.
        </p>
      </Translated>
      <Source section="3.3" pages="56, 58" figs="Fig. 3.8" />
      <Code
        tabs={[{ label: 'CUDA', code: KERNEL, tiers: BLUR_TIERS, status: tr('front-end-checked; emulator-checked (python3 labs/ch03/snippets/check.py); unverified on CUDA or HIP hardware', 'front-end-checked; emulator-checked (python3 labs/ch03/snippets/check.py); unverified trên phần cứng CUDA hoặc HIP') }]}
        firstLine={0}
        highlight={[5, 15, 22]}
        caption={
          <Translated
            vi={
              <>
                Kernel của Fig. 3.8, lưu tại <code>labs/ch03/snippets/blur_kernel.cu</code>. Các dòng 1 đến 24 mang
                số dòng của sách; mã in của sách không có dòng 08, chính là dòng trống ở đây. Hai khác biệt so với
                trang sách: hằng số được định nghĩa ở dòng 0 (sách định nghĩa nó ở chỗ khác), và chú thích ở dòng 09 ghi
                đúng kích thước vùng, trong khi chú thích của sách ghi "BLUR_SIZE x BLUR_SIZE box". Tên màu hổ phách nằm trong bộ nhớ toàn cục,
                tên màu tím nằm trong thanh ghi.
              </>
            }
          >
            <>
              The kernel of Fig. 3.8, stored as <code>labs/ch03/snippets/blur_kernel.cu</code>. Lines 1 to 24 carry the
              book's numbers; the book's listing has no line 08, which is the blank line here. Two differences from the
              page: the constant is defined on line 0 (the book defines it elsewhere), and the comment on line 09 gives
              the real patch size, where the book's says "BLUR_SIZE x BLUR_SIZE box". Amber names live in global memory,
              violet ones in registers.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            <code>BLUR_SIZE</code> là bán kính của vùng: số điểm ảnh ở mỗi phía của tâm. Vùng có cạnh
            <code>2*BLUR_SIZE+1</code>, nên 1 cho 3 × 3 và 3 cho 7 × 7. Các vòng lặp cho{' '}
            <code>blurRow</code> và <code>blurCol</code> chạy từ <code>-BLUR_SIZE</code> đến <code>BLUR_SIZE</code>, và các dòng
            12 và 13 biến các độ lệch đó thành một vị trí trong ảnh. Với luồng của điểm ảnh đầu ra (25, 50), lượt đầu
            của vòng lặp ngoài có <code>curRow</code> = 24 và vòng lặp trong đi qua các cột 49, 50, 51; lượt thứ hai
            làm hàng 25, lượt thứ ba hàng 26.
          </p>
        }
      >
        <p>
          <code>BLUR_SIZE</code> is the radius of the patch: the number of pixels on each side of the centre. The patch
          is <code>2*BLUR_SIZE+1</code> on a side, so 1 gives 3 by 3 and 3 gives 7 by 7. The loops run{' '}
          <code>blurRow</code> and <code>blurCol</code> from <code>-BLUR_SIZE</code> to <code>BLUR_SIZE</code>, and lines
          12 and 13 turn those offsets into a position in the picture. For the thread of output pixel (25, 50) the first
          pass of the outer loop has <code>curRow</code> = 24 and the inner loop visits columns 49, 50, 51; the second
          pass does row 25, the third row 26.
        </p>
      </Translated>
      <Source section="3.3" pages="58" />

      <h2><Translated vi="Mép ảnh">The edge</Translated></h2>
      <Translated
        vi={
          <p>
            Lần theo dấu đó suôn sẻ vì điểm ảnh (25, 50) ở giữa ảnh. Thay vào đó hãy lấy luồng của điểm ảnh đầu ra (0, 0).
            Các vòng lặp của nó tạo ra cùng chín độ lệch, và một số vị trí chúng trỏ tới không tồn tại.
          </p>
        }
      >
        <p>
          That trace works because pixel (25, 50) is in the middle of the picture. Take the thread of output pixel (0, 0)
          instead. Its loops produce the same nine offsets, and some of the positions they point at do not exist.
        </p>
      </Translated>
      <Source section="3.3" pages="58" />
      <Predict
        id="ch03/3.3/p-corner"
        prompt={
          <Translated
            vi={
              <>
                <code>BLUR_SIZE</code> bằng 1. Trong chín vị trí (<code>curRow</code>, <code>curCol</code>) mà
                luồng của điểm ảnh đầu ra (0, 0) tính, bao nhiêu vị trí qua được phép kiểm tra ở dòng 15?
              </>
            }
          >
            <>
              <code>BLUR_SIZE</code> is 1. Of the nine (<code>curRow</code>, <code>curCol</code>) positions that the
              thread for output pixel (0, 0) computes, how many pass the test on line 15?
            </>
          </Translated>
        }
        unit={tr('positions', 'vị trí')}
      >
        {(v) => (
          <>
            <p>
              <Translated vi={<>{v === 4 ? 'Đúng, 4.' : 'Đáp án là 4.'} Chín cặp là</>}>
                {v === 4 ? 'Yes, 4.' : 'It is 4.'} The nine pairs are
              </Translated>{' '}
              {patch(0, 0, 76, 62, 1).map((p, i) => (
                <span key={i} className="nowrap" style={p.inside ? { fontWeight: 650 } : { color: 'var(--ink-3)' }}>
                  ({p.curRow}, {p.curCol}){i < 8 ? ', ' : ''}
                </span>
              ))}
              <Translated vi={<>. Năm cặp chứa −1 và bị bỏ qua; bốn cặp in đậm được cộng, và <code>pixels</code> kết thúc ở 4.</>}>
                . Five contain a −1 and are skipped; the four in bold are summed, and <code>pixels</code> ends at 4.
              </Translated>
            </p>
            <Translated vi={<p>Chạy từng bước bên dưới và xem mệnh đề nào loại từng vị trí.</p>}>
              <p>Step through it below and watch which clause turns each position away.</p>
            </Translated>
          </>
        )}
      </Predict>
      <LoopStepper />
      <Translated
        vi={
          <p>
            Sáu trường hợp trong trình chạy từng bước là năm vùng mà sách vẽ ở viền, cộng một điểm ảnh bên trong. Một
            luồng ở góc cộng 4 giá trị, một luồng ở chỗ khác trên mép cộng 6, một luồng bên trong cộng 9. Vì
            con số thay đổi, kernel không thể chia cho một hằng số: nó đếm những gì đã cộng, trong <code>pixels</code>, và
            chia cho số đó.
          </p>
        }
      >
        <p>
          The six cases in the stepper are the five patches the book draws at the borders, plus one interior pixel. A
          corner thread sums 4 values, a thread elsewhere on an edge sums 6, an interior thread sums 9. Because the
          number varies, the kernel cannot divide by a constant: it counts what it summed, in <code>pixels</code>, and
          divides by that.
        </p>
      </Translated>
      <Source section="3.3" pages="58–59" figs="Fig. 3.9" />
      <Translated
        vi={
          <p>
            Trình khám phá cho bạn tự di chuyển điểm ảnh đầu ra và thay đổi ảnh cũng như vùng. Hai điều cần tìm:
            một luồng ở góc lấy trung bình những gì khi <code>BLUR_SIZE</code> bằng 3, và vì sao bạn không bao giờ cần liệt kê cả vùng
            để có số đếm.
          </p>
        }
      >
        <p>
          The explorer lets you move the output pixel yourself and change the picture and the patch. Two things to find:
          what a corner thread averages when <code>BLUR_SIZE</code> is 3, and why you never need to enumerate the patch
          to get the count.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một kênh màu, ảnh tối đa 12 × 12, không có thời gian. Các giá trị chỉ có một chữ số để bạn có thể tự kiểm tra một tổng
              bằng tay; một ảnh thật chứa các byte từ 0 đến 255.
            </p>
          }
        >
          <p>
            One channel, a picture of at most 12 by 12, no timing. Values are single digits so that you can check a sum
            by hand; a real picture holds bytes from 0 to 255.
          </p>
        </Translated>
      </Simplification>
      <BoundaryExplorer />
      <Translated
        vi={
          <p>
            Bật bản đồ số đếm và ảnh được tô màu theo lượng việc mỗi luồng làm. Các luồng dọc
            viền chạy cùng các vòng lặp như mọi luồng khác nhưng đi đường tắt ở phần lớn các lần lặp. Các luồng của cùng một
            kernel làm lượng việc khác nhau là chủ đề của chương 4, chương sẽ quay lại những bức ảnh như thế này.
          </p>
        }
      >
        <p>
          Switch on the count map and the picture is coloured by how much work each thread does. The threads along the
          border run the same loops as everyone else but take the short way through most iterations. Threads of one
          kernel doing different amounts of work is the subject of chapter 4, which returns to pictures like this one.
        </p>
      </Translated>
      <Ask q={questionsS33['ch03/3.3/q-corner7']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS33['ch03/3.3/q-edge']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Hai phép kiểm tra, hai kiểu hỏng khác nhau">Two guards, two different failures</Translated></h2>
      <Translated
        vi={
          <p>
            Kernel có hai phép kiểm tra trông giống nhau nhưng bảo vệ những thứ khác nhau. Dòng 05 nói về luồng: một lưới gồm
            các khối trọn vẹn có nhiều luồng hơn số điểm ảnh của ảnh, và một luồng dư không được ghi gì cả. Dòng
            15 nói về một vị trí trong vùng của một luồng: một luồng có sở hữu điểm ảnh cũng không được đọc những vị trí
            không nằm trong ảnh.
          </p>
        }
      >
        <p>
          The kernel has two tests that look alike and protect different things. Line 05 is about the thread: a grid of
          whole blocks has more threads than the picture has pixels, and a surplus thread must not store anything. Line
          15 is about one position of one thread's patch: a thread that does own a pixel must not read positions that
          are not in the picture.
        </p>
      </Translated>
      <Source section="3.3" pages="57–59" figs="Fig. 3.8" />
      <Translated
        vi={
          <p>
            Bỏ một trong hai đi thì kernel vẫn biên dịch và vẫn chạy. Kết quả bạn nhận được tùy vào mệnh đề nào
            bị thiếu, và trường hợp đáng học nhất là một mệnh đề về cột. Ảnh được lưu từng hàng nối tiếp trong một mảng
            phẳng, nên <code>in[curRow*w + w]</code>, quá mép phải một cột, là một địa chỉ hoàn toàn hợp lệ: nó
            là điểm ảnh đầu tiên của hàng kế tiếp. Lần đọc thành công và trả về sai điểm ảnh.
          </p>
        }
      >
        <p>
          Take either away and the kernel still compiles and still runs. What you get depends on which clause went
          missing, and the most instructive case is a column clause. The picture is stored row after row in one flat
          array, so <code>in[curRow*w + w]</code>, one column past the right-hand edge, is a perfectly valid address: it
          is the first pixel of the next row. The read succeeds and returns the wrong pixel.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách nêu hai phép kiểm tra dùng để làm gì. Mỗi phép kiểm tra gây ra điều gì khi bị thiếu, theo từng mệnh đề, là phân tích
              của chúng tôi, chạy trên mô hình bên dưới.
            </p>
          }
        >
          <p>
            The book states what the two tests are for. What each one does when it is missing, clause by clause, is our
            analysis, run on the model below.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Bảng tương tác chạy toàn bộ kernel trên một ảnh nhỏ với bất kỳ phần nào của nó bị bỏ đi. Nó so sánh kết quả
            với kết quả của kernel đúng và phân loại mọi truy cập đi tới chỗ không nên tới.
          </p>
        }
      >
        <p>
          The bench runs the whole kernel over a small picture with any of its pieces taken out. It compares the result
          with the correct kernel's and classifies every access that went somewhere it should not.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Các luồng chạy từng cái một, mỗi cái đến khi xong, hết khối này đến khối khác, như trong trình giả lập (emulator) CPU của các lab. Ở đây một lần đọc
              ngoài buffer trả về 0 và một lần ghi ra ngoài bị bỏ; trên thiết bị thật cả hai đều không xác định. 0/0 được
              hiển thị là dấu hỏi.
            </p>
          }
        >
          <p>
            Threads run one at a time, each to completion, block after block, as in the labs' CPU emulator. A read
            outside the buffer returns 0 here and a store outside it is dropped; on a device both are undefined. 0/0 is
            shown as a question mark.
          </p>
        </Translated>
      </Simplification>
      <BreakTheKernel />
      <Translated
        vi={
          <p>
            Có ba điều đáng tìm trong bảng tương tác. Bỏ chọn <code>{'curCol < w'}</code>: cột bên phải
            sai, 17 lần đọc lấy một điểm ảnh từ hàng khác, và chỉ 2 lần nhắm xuống dưới hàng cuối mới ra ngoài buffer. Một
            công cụ kiểm tra địa chỉ, như bản build AddressSanitizer của trình giả lập của các lab, có thể báo 2 lần đó và
            không nói được gì về 17 lần kia. Giờ hãy chuyển ảnh sang
            xám đều: gần như mọi hư hại biến mất, vì một điểm ảnh sai mà tình cờ giữ đúng giá trị thì
            không thay đổi gì. Một ảnh thử có thể che giấu lỗi này. Rồi khôi phục kernel và bỏ chọn số chia: mọi
            truy cập đều trong biên, bên trong đúng, và viền quá tối.
          </p>
        }
      >
        <p>
          Three things are worth finding in the bench. Untick <code>{'curCol < w'}</code>: the right-hand column is
          wrong, 17 reads fetched a pixel from another row, and only the 2 aimed below the last row left the buffer. A
          tool that checks addresses, such as the AddressSanitizer build of the labs' emulator, can report those 2 and
          has nothing to say about the other 17. Now switch the picture to
          uniform grey: almost all of the damage disappears, because a wrong pixel that happens to hold the right value
          changes nothing. A test picture can hide this defect. Then restore the kernel and untick the divisor: every
          access is in bounds, the interior is right, and the border is too dark.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Bỏ chọn dòng 05 cho thấy kiểu hỏng còn lại. Các luồng dư chạy các vòng lặp, phép kiểm tra bên trong vẫn giữ các lần đọc
            của chúng trong ảnh, rồi mỗi luồng đều ghi. Một luồng ở <code>col</code> = 6 của một ảnh rộng 6
            ghi vào <code>out[row*6 + 6]</code>, tức điểm ảnh 0 của hàng kế tiếp. Giờ có hai luồng ghi vào phần tử đó,
            và luồng ghi sau cùng thắng. Trong bảng tương tác, hàng 4 ra đúng chỉ vì lý do đó: khối của chính nó chạy
            sau khối có các luồng dư đã ghi lên nó.
          </p>
        }
      >
        <p>
          Unticking line 05 shows the other failure. The surplus threads run the loops, the inner test still keeps their
          reads inside the picture, and then each of them stores. A thread at <code>col</code> = 6 of a picture 6 wide
          stores to <code>out[row*6 + 6]</code>, which is pixel 0 of the next row. Two threads now store to that element,
          and the one that stores last wins. In the bench, row 4 comes out right for that reason only: its own block runs
          after the block whose surplus threads wrote on it.
        </p>
      </Translated>
      <Ask q={questionsS33['ch03/3.3/q-twoguards']} />
      <Ask q={questionsS33['ch03/3.3/q-area']} />
      <NameTheDefect />

      <h2><Translated vi="Số học có dấu">Signed arithmetic</Translated></h2>
      <Translated
        vi={
          <p>
            Phép kiểm tra <code>{'curRow >= 0'}</code> chỉ có thể sai nếu <code>curRow</code> có thể âm. Các biến
            dựng sẵn <code>blockIdx</code>, <code>blockDim</code> và <code>threadIdx</code> là không dấu. Sách lưu{' '}
            <code>row</code> và <code>col</code> trong <code>int</code>, và cũng khai báo <code>curRow</code> và{' '}
            <code>curCol</code> là <code>int</code>, để 0 + (−1) là −1 và phép kiểm tra ở dòng 15 có đúng nghĩa của nó.
          </p>
        }
      >
        <p>
          The test <code>{'curRow >= 0'}</code> can only fail if <code>curRow</code> can be negative. The built-in
          variables <code>blockIdx</code>, <code>blockDim</code> and <code>threadIdx</code> are unsigned. The book stores{' '}
          <code>row</code> and <code>col</code> in <code>int</code>, and declares <code>curRow</code> and{' '}
          <code>curCol</code> as <code>int</code> too, so that 0 + (−1) is −1 and the test on line 15 means what it says.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách dùng <code>int</code> mà không giải thích. Kiểu của các biến dựng sẵn lấy từ hướng dẫn của NVIDIA
              (notes/today/ch03-facts.md, A2); lập luận về dòng 15 là của chúng tôi.
            </p>
          }
        >
          <p>
            The book uses <code>int</code> without comment. The types of the built-ins are from NVIDIA's guide
            (notes/today/ch03-facts.md, A2); the reasoning about line 15 is ours.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Nếu bạn giữ mọi thứ không dấu thì sao? Trong số học không dấu, 0 + (−1) là 4294967295. Khi đó mệnh đề{' '}
            <code>{'curRow >= 0'}</code> luôn đúng và không làm gì, và chính <code>{'curRow < h'}</code> mới
            loại vị trí đó, vì 4294967295 không nhỏ hơn bất kỳ chiều cao ảnh nào. Chúng tôi đã chạy biến thể đó trên
            trình giả lập của khóa học, đối chiếu với kernel ở trên, trên sáu kích thước ảnh: cùng đầu ra. Vậy kernel không dấu tình cờ
            chạy đúng, với hai mệnh đề là mã chết và những phép so sánh giữa giá trị có dấu và không dấu mà
            trình biên dịch đã cảnh báo khi bật cảnh báo. Hãy viết có dấu, để mã nguồn nói đúng điều bạn muốn.
          </p>
        }
      >
        <p>
          What if you kept everything unsigned? In unsigned arithmetic 0 + (−1) is 4294967295. The clause{' '}
          <code>{'curRow >= 0'}</code> is then always true and does nothing, and it is <code>{'curRow < h'}</code> that
          rejects the position, because 4294967295 is not less than any picture height. We ran that variant on the
          course's emulator against the kernel above, on six picture sizes: same output. So the unsigned kernel happens
          to work, with two clauses that are dead code and comparisons between signed and unsigned values that the
          compiler flagged once warnings were switched on. Write it signed, so that the source says what you mean.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              emulator-checked: <code>labs/ch03/snippets/blur_unsigned.cu</code>, được chạy và biên dịch với{' '}
              <code>-Wall -Wextra</code> bởi <code>python3 labs/ch03/snippets/check.py</code>. Không có gì trong đoạn này
              được chạy trên GPU.
            </p>
          }
        >
          <p>
            Emulator-checked: <code>labs/ch03/snippets/blur_unsigned.cu</code>, run and compiled with{' '}
            <code>-Wall -Wextra</code> by <code>python3 labs/ch03/snippets/check.py</code>. Nothing in this paragraph
            was run on a GPU.
          </p>
        </Translated>
      </Aside>

      <h2><Translated vi="Phép ép kiểu">The cast</Translated></h2>
      <Translated
        vi={
          <p>
            Dòng 22 tính bằng <code>float</code> và lưu một byte: <code>(unsigned char)((float)pixVal/pixels)</code>.
            Cặp ngoặc quan trọng (mã in chuyển ảnh xám của mục trước đã đặt sai chúng); ở đây cả thương được
            chuyển đổi. C++ chuyển một số float sang số nguyên bằng cách bỏ phần lẻ, nên 19/4 = 4.75 được lưu thành 4, không phải 5.
            Góc dưới bên phải của trình chạy từng bước cho thấy đúng điều đó.
          </p>
        }
      >
        <p>
          Line 22 computes in <code>float</code> and stores a byte: <code>(unsigned char)((float)pixVal/pixels)</code>.
          The parentheses matter (the grayscale listing of the last section got them wrong); here the whole quotient is
          converted. C++ converts a float to an integer by dropping the fraction, so 19/4 = 4.75 is stored as 4, not 5.
          The stepper's bottom-right corner shows exactly that.
        </p>
      </Translated>
      <Source section="3.3" pages="57" figs={tr('Fig. 3.8, line 22', 'Fig. 3.8, dòng 22')} />
      <Translated
        vi={
          <p>
            Cùng quy tắc đó nói rằng chuyển đổi một giá trị mà kiểu đích không chứa được là không xác định. Điều đó không thể xảy ra trong
            kernel đúng: trung bình của các byte nằm giữa 0 và 255. Nó là điều xảy ra với một luồng dư chia 0
            cho 0 khi dòng 05 đã bị bỏ.
          </p>
        }
      >
        <p>
          The same rule says that converting a value the destination cannot hold is undefined. That cannot happen in the
          correct kernel: an average of bytes is between 0 and 255. It is what happens to a surplus thread that divides 0
          by 0 once line 05 is gone.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Việc cắt phần lẻ và trường hợp ngoài phạm vi không xác định là quy tắc C++ [conv.fpint] (notes/today/ch03-facts.md, A9).
              Ngôn ngữ shading của Metal cũng làm tròn về 0 và không bão hòa (C6), và mặc định không tuân thủ IEEE nghiêm ngặt
              (ch05-facts C7). Một thương tính ra thấp hơn một chút sẽ tụt xuống một mức xám khi bị cắt phần lẻ, nên
              lab 3.2 so sánh với tham chiếu CPU của nó trong phạm vi một mức.
            </p>
          }
        >
          <p>
            Truncation and the undefined out-of-range case are the C++ rule [conv.fpint] (notes/today/ch03-facts.md, A9).
            Metal's shading language also rounds toward zero and does not saturate (C6), and it is not strict IEEE by
            default (ch05-facts C7). A quotient computed a hair low lands one grey level down once it is truncated, so
            lab 3.2 compares with its CPU reference to within one level.
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Vì sao cần hai buffer">Why two buffers</Translated></h2>
      <Translated
        vi={
          <p>
            Kernel nhận hai con trỏ, <code>in</code> và <code>out</code>, và rất dễ bị cám dỗ tiết kiệm bộ nhớ bằng cách
            làm mờ ảnh tại chỗ. Hãy xem điều đó gây ra gì trước khi đọc tiếp.
          </p>
        }
      >
        <p>
          The kernel takes two pointers, <code>in</code> and <code>out</code>, and it is tempting to save memory by
          blurring a picture in place. Watch what that does before reading on.
        </p>
      </Translated>
      <InPlaceBlurScene />
      <Translated
        vi={
          <p>
            Với hai buffer, mỗi luồng ghi một phần tử mà không luồng nào đọc, và đọc các phần tử mà không luồng nào
            ghi. Các luồng không chia sẻ thứ gì thay đổi, nên chúng có thể chạy theo bất kỳ thứ tự nào, hoặc cùng lúc, và kết quả
            vẫn thế. Truyền một buffer cho cả hai thì mỗi luồng đọc tới tám phần tử mà luồng khác ghi. Việc
            nó thấy giá trị gốc của điểm lân cận hay giá trị trung bình phụ thuộc vào ai tới đó trước.
          </p>
        }
      >
        <p>
          With two buffers, every thread writes one element that no thread reads, and reads elements that no thread
          writes. The threads share nothing that changes, so they can run in any order, or all at once, and the result is
          the same. Pass one buffer for both and each thread reads up to eight elements that other threads write. Whether
          it sees a neighbour's original value or its average depends on who got there first.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách không bàn đến làm mờ tại chỗ. Phần này là của khóa học này; nó dựa trên ánh xạ một-một giữa
              luồng và phần tử đầu ra (pp. 56, 60).
            </p>
          }
        >
          <p>
            The book does not discuss blurring in place. This section is ours; it rests on the one-to-one mapping of
            threads to output elements (pp. 56, 60).
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Bảng tương tác tiếp theo đặt ba thứ tự luồng cạnh nhau, và bạn tự chọn dùng một hay hai buffer.
          </p>
        }
      >
        <p>
          The next bench puts three thread orders side by side, with the choice of one buffer or two under your control.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Hoạt hình và bảng tương tác bên dưới chạy các luồng từng cái một theo một thứ tự cố định. Như vậy đủ để cho thấy
              kết quả phụ thuộc vào thứ tự. Một GPU chạy chúng cùng lúc, và ở đó kết quả không phải là một trong
              những ảnh này mà là bất cứ thứ gì cách đan xen tạo ra.
            </p>
          }
        >
          <p>
            The animation and the bench below run threads one at a time in a fixed order. That is enough to show that
            the result depends on the order. A GPU runs them at the same time, and there the outcome is not one of
            these pictures but whatever the interleaving produces.
          </p>
        </Translated>
      </Simplification>
      <ThreeOrders />
      <Translated
        vi={
          <p>
            Đây là race condition (tranh chấp dữ liệu) của chương: một kernel trong đó một luồng đọc thứ luồng khác ghi, mà không có gì quy định
            điều nào xảy ra trước, thì không có kết quả xác định. Cũng hãy để ý điều đã không xảy ra. Không địa chỉ nào ra ngoài biên, nên một
            công cụ kiểm tra biên không có gì để báo. Cả ba kernel của chương này đều tránh vấn đề theo cùng một cách, bằng cách
            tách riêng đầu vào và đầu ra. Chương 5 là nơi các luồng chia sẻ bộ nhớ mà chúng cũng ghi vào, một cách có chủ đích, và
            là nơi các rào chắn (barrier) quyết định ai đi trước.
          </p>
        }
      >
        <p>
          This is the chapter's race: a kernel in which one thread reads what another thread writes, with nothing to say
          which happens first, has no defined result. Notice also what did not happen. No address was out of bounds, so a
          bounds checker has nothing to report. All three kernels of this chapter avoid the problem the same way, by
          keeping inputs and outputs apart. Chapter 5 is where threads share memory that they also write, on purpose, and
          where barriers decide who goes first.
        </p>
      </Translated>
      <Ask q={questionsS33['ch03/3.3/q-inplace']} />

      <h2><Translated vi="Cùng vòng lặp đó trong một framework: masked mean pooling">The same loop in a framework: masked mean pooling</Translated></h2>
      <Translated
        vi={
          <p>
            Các dòng 10 đến 22 là một mẫu bạn đã viết nhiều lần trong framework: tích lũy trên các vị trí mà phép kiểm tra
            cho qua, đếm chúng, chia cho số đếm. Embedding câu thường được tạo theo cách đó. Với hidden state{' '}
            <code>x</code> có shape [batch, seq, hidden] và padding mask <code>m</code> có shape [batch, seq],
            vector sau pooling là <code>(x * m[..., None]).sum(1) / m.sum(1, keepdim=True)</code>. Mask là phép kiểm tra mép,{' '}
            <code>m.sum</code> là <code>pixels</code>, và nếu chia cho <code>seq</code> thì embedding
            của mọi câu ngắn sẽ bị thu nhỏ, giống như chia cho diện tích vùng làm tối viền.
          </p>
        }
      >
        <p>
          Lines 10 to 22 are a pattern you have written many times in a framework: accumulate over the positions a test
          lets in, count them, divide by the count. Sentence embeddings are often made that way. With hidden states{' '}
          <code>x</code> of shape [batch, seq, hidden] and a padding mask <code>m</code> of shape [batch, seq], the
          pooled vector is <code>(x * m[..., None]).sum(1) / m.sum(1, keepdim=True)</code>. The mask is the edge test,{' '}
          <code>m.sum</code> is <code>pixels</code>, and dividing by <code>seq</code> instead would shrink the embedding
          of every short sentence, the way dividing by the patch area darkens the border.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Cầu nối này là của khóa học này. Một causal attention mask cũng là cấu trúc như vậy, một phép kiểm tra bên trong vòng lặp của từng luồng
              quyết định đầu vào nào được tính; chương 20 xây dựng kernel đó.
            </p>
          }
        >
          <p>
            The bridge is ours. A causal attention mask is the same construct, a test inside the per-thread loop that
            decides which inputs count; chapter 20 builds that kernel.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS33['ch03/3.3/q-maskedmean']} />

      <h2><Translated vi="Cùng kernel trên HIP và Metal">The same kernel on HIP and Metal</Translated></h2>
      <Translated
        vi={
          <p>
            Trên HIP mã nguồn giống hệt: từ khóa, các biến dựng sẵn, lệnh khởi chạy và phép tính chỉ số được
            viết cùng một cách, và các lab biên dịch một tệp <code>.cu</code> cho cả hai. Metal thay đổi ba điều, và
            hai trong số đó là những điểm đã nêu ở trên.
          </p>
        }
      >
        <p>
          On HIP the source is identical: the qualifier, the built-in variables, the launch and the index arithmetic are
          spelled the same way, and the labs compile one <code>.cu</code> file for both. Metal changes three things, and
          two of them are the points made above.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Đã kiểm tra theo tài liệu của các hãng: notes/today/ch03-facts.md B5 (HIP), C1, C3, C6 (Metal). Bản build HIP
              chưa được biên dịch trên phần cứng AMD.
            </p>
          }
        >
          <p>
            Checked against vendor documentation: notes/today/ch03-facts.md B5 (HIP), C1, C3, C6 (Metal). The HIP
            build has not been compiled on AMD hardware.
          </p>
        </Translated>
      </Beyond>
      <ul>
        <Translated
          vi={
            <li>
              Một kernel Metal có thể xin trực tiếp vị trí của nó, dưới dạng một tham số được đánh dấu{' '}
              <code>[[thread_position_in_grid]]</code>; <code>.x</code> là cột và <code>.y</code> là hàng. Tham số
              đó chỉ có thể có kiểu không dấu, nên hãy chuyển nó sang <code>int</code> trước khi cộng một độ lệch có dấu.
            </li>
          }
        >
          <li>
            A Metal kernel can ask for its position directly, as a parameter marked{' '}
            <code>[[thread_position_in_grid]]</code>; <code>.x</code> is the column and <code>.y</code> the row. That
            parameter may only have an unsigned type, so convert it to <code>int</code> before adding a signed offset.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Phép ép kiểu được viết là <code>(uchar)</code> và cắt về phía 0 như trong C++, không bão hòa.
            </li>
          }
        >
          <li>
            The cast is written <code>(uchar)</code> and truncates toward zero like the C++ one, with no saturation.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Bạn có cần dòng 05 hay không tùy vào lệnh khởi chạy. Lab 3.2 dùng <code>dispatchThreadgroups</code>, nhận một
              số lượng nhóm trọn vẹn như một lưới CUDA, nên lưới bị thừa ra ngoài và phép kiểm tra ngoài vẫn ở lại.{' '}
              <code>dispatchThreads</code> của Metal nhận đúng số luồng và, trên các thiết bị hỗ trợ, làm các nhóm ở mép
              nhỏ hơn; khi đó không có luồng dư nào cần loại. Phép kiểm tra bên trong thì cần trong cả hai trường hợp.
            </li>
          }
        >
          <li>
            Whether you need line 05 depends on the launch. Lab 3.2 uses <code>dispatchThreadgroups</code>, which takes a
            number of whole groups as a CUDA grid does, so the grid overhangs and the outer test stays. Metal's{' '}
            <code>dispatchThreads</code> takes the exact number of threads and, on devices that support it, makes the edge
            groups smaller; then there are no surplus threads to turn away. The inner test is needed either way.
          </li>
        </Translated>
      </ul>
      {!missing(cu) && !missing(metal) && !missing(mm) ? (
        <Code
          tabs={[
            { label: tr('CUDA (lab 3.2)', 'CUDA (lab 3.2)'), code: cu!, tiers: BLUR_TIERS, status: cudaStatus },
            { label: tr('Metal kernel', 'Kernel Metal'), code: metal!, tiers: BLUR_TIERS, status: metalStatus },
            { label: tr('Metal host', 'Host Metal'), code: mm!, tiers: {}, status: metalStatus },
          ]}
          caption={
            <Translated
              vi={
                <>
                  Lời giải tham chiếu (solution) của lab 3.2, nạp từ <code>labs/ch03/lab2_blur/solution/</code>. Nó khác
                  mã in ở trên ba điểm: bán kính là một đối số của kernel, để benchmark có thể thay đổi nó;{' '}
                  <code>in</code> là <code>const</code>; và nó có kèm lệnh khởi chạy, thứ sách không in. Trạng
                  thái bên cạnh các tab được đọc từ các lần chạy lab đã ghi trên máy này. Hãy tự viết bản của bạn trong lab trước khi
                  xem bản này.
                </>
              }
            >
              <>
                The reference solution of lab 3.2, loaded from <code>labs/ch03/lab2_blur/solution/</code>. It differs
                from the listing above in three ways: the radius is a kernel argument, so the benchmark can vary it;{' '}
                <code>in</code> is <code>const</code>; and it includes a launch, which the book does not print. The
                status beside the tabs is read from this machine's recorded lab runs. Write your own in the lab before
                studying this one.
              </>
            </Translated>
          }
        />
      ) : null}
      <Translated
        vi={
          <p>
            <a href={href('ch03', 'labs')}>Lab 3.2</a> cho bạn viết kernel này cho một ảnh thật và kiểm tra nó với một
            tham chiếu CPU. Lab 3.5 đưa bạn một kernel có bốn lỗi chỉ số đúng loại bạn vừa tạo ra trong bảng tương tác,
            và yêu cầu bạn tìm chúng từ đầu ra. Tiếp theo, <a href={href('ch03', '3.4')}>mục 3.4</a> thay
            vùng bằng một hàng và một cột: phép nhân ma trận.
          </p>
        }
      >
        <p>
          <a href={href('ch03', 'labs')}>Lab 3.2</a> has you write this kernel for a real picture and check it against a
          CPU reference. Lab 3.5 hands you one kernel with four index defects of the kind you just produced in the bench,
          and asks you to find them from their output. Next, <a href={href('ch03', '3.4')}>section 3.4</a> replaces the
          patch by a row and a column: matrix multiplication.
        </p>
      </Translated>
    </>
  );
}
