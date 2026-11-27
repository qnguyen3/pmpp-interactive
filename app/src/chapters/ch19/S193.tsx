import { Fragment, useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile, type CodeTab } from '../../components/Code';
import { defineLabs, solutionTabStatus } from '../../components/LabKit';
import { Beyond, Bench, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  CORRECTED,
  HUNT_SHAPES,
  NVIDIA_BANKS,
  gridOf,
  huntShape,
  multiples,
  phasesOf,
  runKernel,
  warpView,
  type HuntId,
  type KShape,
  type Run,
} from '../../sim/implicitgemm';
import {
  BOOK_F,
  BOOK_SHAPE,
  BOOK_X,
  MiB,
  PRESETS,
  bToX,
  ckk,
  convAsGemm,
  copiesPerPixel,
  dotTerms,
  expansion,
  gemmShape,
  limitRatio,
  makeData,
  outDims,
  pixels,
  preset,
  sizes,
  unfold,
  valid,
  xToB,
  type BSource,
  type ConvShape,
  type FlagId,
  type PresetId,
} from '../../sim/unfold';
import { ImplicitTilesScene } from './ImplicitTilesScene';
import { UnfoldScene } from './UnfoldScene';

// ---------------------------------------------------------------- the lab files this section shows

const LABS = defineLabs('ch19', [{ num: '19.1', dir: 'lab1_convgemm', name: 'convgemm', vi: { name: 'convgemm' } }]);
const [LAB1] = LABS;
const LAB_DIR = 'ch19/lab1_convgemm/';
/** Fig. 19.11, corrected, as lab 19.1 provides it. */
export const PROVIDED = { cu: LAB_DIR + 'provided/implicit_gemm.cu', metal: LAB_DIR + 'provided/implicit_gemm.metal' };
/**
 * The two pieces of the reference solution shown beside q-roles. The lab marks each with
 * `// [lesson:<name>]` and `// [/lesson:<name>]`: unroll in kernel.cu and kernel.metal,
 * gemm-path in kernel.cu and launch.mm.
 */
export const SOLUTION_FILES = ['kernel.cu', 'kernel.metal', 'launch.mm'] as const;
export const REGIONS = ['unroll', 'gemm-path'] as const;

/** Identifiers coloured by the memory they live in, as the diagrams colour them. */
export const CONV_TIERS: Record<string, Tier> = {
  X: 'global', F: 'global', Y: 'global', Xn: 'global', Bt: 'global',
  Fds: 'shared', Bds: 'shared',
  Pvalue: 'reg', Row: 'reg', Col: 'reg', tx: 'reg', ty: 'reg', bx: 'reg', by: 'reg', bz: 'reg', u: 'reg', v: 'reg', ph: 'reg', k: 'reg',
  fc: 'reg', c: 'reg', r: 'reg', h: 'reg', w: 'reg', i: 'reg', P: 'reg', CKK: 'reg', H_out: 'reg', W_out: 'reg',
};

/** The lines strictly between `// [lesson:name]` and `// [/lesson:name]`, with the file's line numbers, or null. */
export function region(src: string | null, name: string): { code: string; firstLine: number } | null {
  if (!src || src.startsWith('// could not load')) return null;
  const lines = src.split('\n');
  const a = lines.findIndex((l) => l.trim() === `// [lesson:${name}]`);
  const b = a < 0 ? -1 : lines.findIndex((l, i) => i > a && l.trim() === `// [/lesson:${name}]`);
  if (a < 0 || b < 0) return null;
  return { code: lines.slice(a + 1, b).join('\n'), firstLine: a + 2 };
}

/** Status of lab 19.1's files, in the labs page's words (LabKit's solutionTabStatus). */
const cuStatus = (reports: Reports): string => solutionTabStatus(reports, LAB1, 'kernel.cu');
const metalStatus = (reports: Reports): string => solutionTabStatus(reports, LAB1, 'kernel.metal');

// ---------------------------------------------------------------- small helpers

const big = (n: number) => n.toLocaleString('en-US');
const r2 = (x: number) => (Math.round(x * 100) / 100).toString();
const bytesText = (b: number) => (b >= 1024 * MiB ? `${(b / (1024 * MiB)).toFixed(2)} GiB` : b >= MiB ? `${(b / MiB).toFixed(1)} MiB` : `${(b / 1024).toFixed(1)} KiB`); // i18n-ignore: byte units
const pct = (x: number) => `${Math.round(x * 100)}%`;

const FLAG_TEXT: Record<FlagId, ReactNode> = {
  int32: <Translated vi={<>nhiều hơn 2<sup>31</sup> − 1 phần tử: chỉ số <code>int</code> 32-bit vào B bị tràn</>}><>more than 2<sup>31</sup> − 1 elements: a 32-bit <code>int</code> index into B overflows</></Translated>,
  cudnn: <Translated vi={<>ít nhất 2·10<sup>9</sup> phần tử: cuDNN ghi trong tài liệu giới hạn "2 Giga-elements" (2 tỉ phần tử) cho mỗi tensor (A1)</>}><>at least 2·10<sup>9</sup> elements: cuDNN documents a cap of "2 Giga-elements" per tensor (A1)</></Translated>,
  metal: <Translated vi={<>hơn 256 MB: vượt mức <code>maxBufferLength</code> nhỏ nhất mà Metal bảo đảm (C2). Hãy kiểm tra giá trị trên thiết bị của bạn; MLX mở rộng theo từng dải hàng khi B không vừa (D3)</>}><>more than 256 MB: above the smallest <code>maxBufferLength</code> Metal guarantees (C2). Check your own device's value; MLX unfolds in row tiles when B will not fit (D3)</></Translated>,
};

function Slider({ label, value, min, max, onChange }: { label: string; value: number; min: number; max: number; onChange: (v: number) => void }) {
  return (
    <label className="field" style={{ minWidth: '8.5rem' }}>
      <span>
        {label} = <b>{value}</b>
      </span>
      <input type="range" min={min} max={max} value={value} aria-label={label} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

const cell = { minWidth: '1.55rem', height: '1.45rem' };

// ---------------------------------------------------------------- unfold explorer

function SmallDrawing({ s, X, F, hover, setHover, pick, setPick, heat }: {
  s: ConvShape; X: number[]; F: number[];
  hover: { m: number; v: number } | null; setHover: (h: { m: number; v: number }) => void;
  pick: { c: number; h: number; w: number } | null; setPick: (p: { c: number; h: number; w: number }) => void;
  heat: boolean;
}) {
  const P = pixels(s);
  const R = ckk(s);
  const { Wout } = outDims(s);
  const B = unfold(s, X);
  const Y = convAsGemm(s, X, F);
  const terms = hover ? dotTerms(s, X, F, hover.m, hover.v) : [];
  const patch = new Set(terms.map((t) => t.src.flat));
  const copiedBy = pick ? xToB(s, pick.c, pick.h, pick.w) : [];
  const copied = new Set(copiedBy.map((e) => e.u * P + e.v));
  const copies = copiesPerPixel(s);
  const maxCopies = Math.max(...copies);
  const outline = { outline: '2px solid var(--ink)', outlineOffset: '-2px' }; // i18n-ignore: CSS value
  const key = (e: React.KeyboardEvent, f: () => void) => (e.key === 'Enter' || e.key === ' ') && f();
  return (
    <>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', marginBottom: '0.8rem' }}>
        <div>
          <div className="mx-label t-global"><Translated vi={<>X: {s.C} bản đồ đầu vào, {s.H} × {s.W}</>}>X: {s.C} input map{s.C > 1 ? 's' : ''}, {s.H} × {s.W}</Translated></div>
          <div className="row" style={{ gap: '0.8rem', alignItems: 'flex-start' }} data-part="x">
            {Array.from({ length: s.C }, (_, c) => (
              <div key={c}>
                <div className="muted" style={{ fontSize: '0.75rem' }}><Translated vi={<>bản đồ {c}</>}>map {c}</Translated></div>
                <div className="mx global" style={{ gridTemplateColumns: `repeat(${s.W}, auto)` }}>
                  {Array.from({ length: s.H * s.W }, (_, i) => {
                    const h = Math.floor(i / s.W);
                    const w = i % s.W;
                    const flat = c * s.H * s.W + i;
                    const on = pick && pick.c === c && pick.h === h && pick.w === w;
                    return (
                      <span key={i} role="button" tabIndex={0} data-x={`${c},${h},${w}`} className={patch.has(flat) ? 'hot' : ''}
                        style={{ ...cell, cursor: 'pointer', ...(on ? outline : {}) }}
                        title={tr(`X[n, ${c}, ${h}, ${w}]: copied ${copies[i]} time${copies[i] === 1 ? '' : 's'}`, `X[n, ${c}, ${h}, ${w}]: được sao chép ${copies[i]} lần`)}
                        onClick={() => setPick({ c, h, w })} onKeyDown={(e) => key(e, () => setPick({ c, h, w }))}>
                        {heat ? <i aria-hidden style={{ position: 'absolute', inset: 0, background: 'var(--global)', opacity: (0.08 + 0.32 * copies[i] / maxCopies) * (patch.has(flat) ? 0 : 1), borderRadius: 2 }} /> : null}
                        <span style={{ position: 'relative' }}>{X[flat]}</span>
                        {heat ? <sub>×{copies[i]}</sub> : null}
                      </span>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
        <div>
          <div className="mx-label t-global"><Translated vi={<>F: {s.M} × {R} (mảng bộ lọc, không sao chép)</>}>F: {s.M} × {R} (the filter array, no copy)</Translated></div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${R}, auto)` }} data-part="f">
            {F.map((v, i) => (
              <span key={i} className={hover && Math.floor(i / R) === hover.m ? 'on' : ''}
                style={{ ...cell, minWidth: '1.3rem', ...(i % (s.K * s.K) === 0 && i % R ? { marginLeft: 4 } : {}) }}>{v}</span>
            ))}
          </div>
        </div>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
        <div>
          <div className="mx-label t-global"><Translated vi={<>B: {R} × {P} (mỗi pixel đầu ra một cột)</>}>B: {R} × {P} (one column per output pixel)</Translated></div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${P}, auto)` }} data-part="b">
            {B.map((val, i) => {
              const u = Math.floor(i / P);
              const v = i % P;
              return (
                <span key={i} data-b={`${u},${v}`} className={hover && v === hover.v ? 'on' : ''}
                  style={{ ...cell, ...(copied.has(i) ? outline : {}), ...(u % (s.K * s.K) === 0 && u ? { marginTop: 4 } : {}) }}>{val}</span>
              );
            })}
          </div>
        </div>
        <div>
          <div className="mx-label t-global">Y = F · B: {s.M} × {P}</div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${P}, auto)` }} data-part="y">
            {Y.map((val, i) => {
              const m = Math.floor(i / P);
              const v = i % P;
              const on = hover && hover.m === m && hover.v === v;
              return (
                <span key={i} role="button" tabIndex={0} data-y={`${m},${v}`} className={on ? 'hot' : ''} style={{ ...cell, cursor: 'pointer' }}
                  onMouseEnter={() => setHover({ m, v })} onFocus={() => setHover({ m, v })} onClick={() => setHover({ m, v })}>{val}</span>
              );
            })}
          </div>
          <p className="muted" style={{ fontSize: '0.8rem', margin: '0.4rem 0 0' }}>
            <Translated vi={<>Hàng m của Y là bản đồ đầu ra m, từng hàng một ({outDims(s).Hout} × {Wout}).</>}>Row m of Y is output map m, row by row ({outDims(s).Hout} × {Wout}).</Translated>
          </p>
        </div>
      </div>
      <div className="narrate" role="status" style={{ marginTop: '0.8rem' }} data-readout="dot">
        {hover ? (
          <>
            <Translated vi={<>Y[n, {hover.m}, {Math.floor(hover.v / Wout)}, {hover.v % Wout}] = hàng {hover.m} của F · cột {hover.v} của B ={' '}{terms.map((t) => `${t.f}·${t.x}`).join(' + ')} = <b>{Y[hover.m * P + hover.v]}</b></>}>
            Y[n, {hover.m}, {Math.floor(hover.v / Wout)}, {hover.v % Wout}] = row {hover.m} of F · column {hover.v} of B ={' '}
            {terms.map((t) => `${t.f}·${t.x}`).join(' + ')} = <b>{Y[hover.m * P + hover.v]}</b>
            </Translated>
          </>
        ) : (
          tr('Hover over or click a cell of Y.', 'Di chuột lên hoặc nhấp vào một ô của Y.')
        )}
        <br />
        {pick ? (
          <span data-readout="copies">
            <Translated vi={<>X[n, {pick.c}, {pick.h}, {pick.w}] được sao chép vào <b>{copiedBy.length}</b> phần tử của B:{' '}{copiedBy.map((e) => `B[${e.u}, ${e.v}]`).join(', ')} (được viền).</>}>
            X[n, {pick.c}, {pick.h}, {pick.w}] is copied into <b>{copiedBy.length}</b> entr{copiedBy.length === 1 ? 'y' : 'ies'} of B:{' '}
            {copiedBy.map((e) => `B[${e.u}, ${e.v}]`).join(', ')} (outlined).
            </Translated>
          </span>
        ) : (
          <span className="muted"><Translated vi="Nhấp vào một pixel của X để viền mọi phần tử của B sao chép nó.">Click a pixel of X to outline every entry of B that copies it.</Translated></span>
        )}
      </div>
    </>
  );
}

function ShapeDiagram({ s }: { s: ConvShape }) {
  const R = ckk(s);
  const P = pixels(s);
  const M = s.M;
  const Wd = 600;
  const Hd = 250;
  const k = Math.min((Wd - 60) / (R + P), (Hd - 40) / (R + M));
  const px = (n: number) => Math.max(1.5, n * k);
  const ox = 40;
  const oy = 20;
  const z = sizes(s);
  const xBar = Math.max(1.5, (Wd - 200) / Math.max(1, expansion(s)));
  return (
    <svg viewBox={`0 0 ${Wd} ${Hd + 70}`} role="img" aria-label={tr("The three matrices drawn to scale", 'Ba ma trận được vẽ theo đúng tỉ lệ')} data-layout="narrow" style={{ width: '100%', maxWidth: Wd, display: 'block' }}>
      <rect x={ox + px(R)} y={oy} width={px(P)} height={px(R)} fill="var(--global-wash)" stroke="var(--global)" />
      <rect x={ox} y={oy + px(R)} width={px(R)} height={px(M)} fill="var(--global-wash)" stroke="var(--global)" />
      <rect x={ox + px(R)} y={oy + px(R)} width={px(P)} height={px(M)} fill="var(--cell)" stroke="var(--global)" />
      <text x={ox + px(R) + 4} y={oy + 12} fontSize="11" style={{ fill: 'var(--ink)' }}>B: {big(R)} × {big(P)}</text>
      <text x={ox} y={oy + px(R) + px(M) + 14} fontSize="11" style={{ fill: 'var(--ink)' }}>F: {big(M)} × {big(R)}</text>
      <text x={ox + px(R) + 4} y={oy + px(R) + px(M) + 14} fontSize="11" style={{ fill: 'var(--ink)' }}>Y: {big(M)} × {big(P)}</text>
      <text x={ox} y={Hd + 14} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi="số phần tử mỗi mẫu, theo đúng tỉ lệ">elements per sample, to scale</Translated></text>
      <rect x={ox + 60} y={Hd + 22} width={Wd - 200} height={12} fill="var(--global)" />
      <text x={ox} y={Hd + 32} fontSize="11" style={{ fill: 'var(--ink)' }}>B</text>
      <text x={ox + 66 + Wd - 200} y={Hd + 32} fontSize="11" style={{ fill: 'var(--ink)' }}>{big(z.bSample)}</text>
      <rect x={ox + 60} y={Hd + 42} width={xBar} height={12} fill="var(--global-wash)" stroke="var(--global)" />
      <text x={ox} y={Hd + 52} fontSize="11" style={{ fill: 'var(--ink)' }}>X</text>
      <text x={ox + 66 + xBar} y={Hd + 52} fontSize="11" style={{ fill: 'var(--ink)' }}>{big(z.xSample)}</text>
    </svg>
  );
}

const isBook = (s: ConvShape) => s.C === 3 && s.H === 3 && s.W === 3 && s.K === 2 && s.M === 2 && (s.stride ?? 1) === 1;

function UnfoldExplorer() {
  const [presetId, setPresetId] = useState<PresetId | 'custom'>('book');
  const [shape, setShape] = useState<ConvShape>(BOOK_SHAPE);
  const [patchEmbed, setPatchEmbed] = useState(false);
  const [hover, setHover] = useState<{ m: number; v: number } | null>({ m: 0, v: 0 });
  const [pick, setPick] = useState<{ c: number; h: number; w: number } | null>(null);
  const [heat, setHeat] = useState(true);
  const s: ConvShape = { ...shape, stride: patchEmbed ? shape.K : 1 };
  const P = pixels(s);
  const R = ckk(s);
  const small = R <= 18 && P <= 16 && s.M <= 4 && s.C <= 3 && s.H <= 6 && s.W <= 6;
  const data = useMemo(
    () => (isBook(s) ? { X: BOOK_X, F: BOOK_F } : { X: makeData(s.C * s.H * s.W, 77 + s.H, 0, 3), F: makeData(s.M * R, 91 + s.C, 0, 2) }),
    [s.C, s.H, s.W, s.K, s.M, s.stride, R], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const z = sizes(s);
  const ratio = expansion(s);
  const set = (k: keyof ConvShape, v: number) => {
    const next = { ...shape, [k]: v };
    next.K = Math.min(next.K, next.H, next.W);
    setShape(next);
    setPresetId('custom');
    setHover({ m: 0, v: 0 });
    setPick(null);
  };
  const choose = (id: PresetId) => {
    setPresetId(id);
    setShape(preset(id).shape);
    setHover({ m: 0, v: 0 });
    setPick(null);
  };
  const { Hout, Wout } = outDims(s);
  return (
    <Bench
      title={tr("Unfold explorer", 'Trình khám phá phép mở rộng (unfold)')}
      task={tr("Find out what makes B big: change K, the map size, C and M, and watch the ratio.", 'Tìm xem điều gì làm B lớn: thay đổi K, kích thước bản đồ, C và M, và quan sát tỉ số.')}
      foot={
        <>
          <Translated vi={<>Hình vẽ xuất hiện với các lớp nhỏ (ví dụ của sách là mặc định); các lớp lớn hơn được vẽ theo đúng tỉ lệ. Các giá trị khác với của sách là những số nguyên nhỏ được tạo ra. <TierLegend tiers={['global']} /></>}>
          The drawings appear for small layers (the book's example is the default); larger layers are drawn to scale. Values other
          than the book's are small made-up integers. <TierLegend tiers={['global']} />
          </Translated>
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Lớp">Layer</Translated>
          <select value={presetId} onChange={(e) => e.target.value !== 'custom' && choose(e.target.value as PresetId)} aria-label={tr("Layer", 'Lớp')}>
            {PRESETS.map((p) => (
              <option key={p.id} value={p.id}>{tr(p.label, p.id === 'book' ? 'ví dụ của sách' : p.id === 'early' ? 'lớp đầu của lab' : p.id === 'middle' ? 'lớp giữa của lab' : p.id === 'late' ? 'lớp cuối của lab' : p.id === 'pointwise' ? 'lớp 1 × 1 của lab' : p.label)}</option>
            ))}
            <option value="custom">{tr('your own (sliders)', 'của bạn (thanh trượt)')}</option>
          </select>
        </label>
        <Slider label="C" value={shape.C} min={1} max={512} onChange={(v) => set('C', v)} />
        <Slider label="H" value={shape.H} min={2} max={512} onChange={(v) => set('H', v)} />
        <Slider label="W" value={shape.W} min={2} max={512} onChange={(v) => set('W', v)} />
        <Slider label="K" value={shape.K} min={1} max={Math.min(11, shape.H, shape.W)} onChange={(v) => set('K', v)} />
        <Slider label="M" value={shape.M} min={1} max={512} onChange={(v) => set('M', v)} />
        <Slider label="N" value={shape.N} min={1} max={64} onChange={(v) => set('N', v)} />
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="check">
          <input type="checkbox" checked={heat} onChange={(e) => setHeat(e.target.checked)} /> <Translated vi="số bản sao mỗi pixel đầu vào (hình vẽ nhỏ)">copies per input pixel (small drawings)</Translated>
        </label>
        <label className="check">
          <input type="checkbox" checked={patchEmbed} onChange={(e) => { setPatchEmbed(e.target.checked); setHover({ m: 0, v: 0 }); setPick(null); }} />{' '}
          <span className="chip const"><Translated vi="Ngoài sách">Beyond</Translated></span> <Translated vi="stride = K (một patch embedding)">stride = K (a patch embedding)</Translated>
        </label>
      </div>

      {!valid(s) ? (
        <p className="t-fail"><Translated vi="K phải nằm vừa trong bản đồ.">K must fit inside the map.</Translated></p>
      ) : small ? (
        <SmallDrawing s={s} X={data.X} F={data.F} hover={hover} setHover={setHover} pick={pick} setPick={setPick} heat={heat} />
      ) : (
        <ShapeDiagram s={s} />
      )}

      <table className="data" style={{ background: 'var(--page)', margin: '0.9rem 0 0' }} data-panel="unfold">
        <tbody>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}><Translated vi="Hình dạng (một mẫu)">Shapes (one sample)</Translated></th>
            <td>
              <Translated vi={<>bản đồ đầu ra {Hout} × {Wout}; F {big(s.M)} × {big(R)}; B {big(R)} × {big(P)}; Y {big(s.M)} × {big(P)}</>}>
              output maps {Hout} × {Wout}; F {big(s.M)} × {big(R)}; B {big(R)} × {big(P)}; Y {big(s.M)} × {big(P)}
              </Translated>
            </td>
          </tr>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}><Translated vi="Số phần tử mỗi mẫu">Elements per sample</Translated></th>
            <td data-value="elements">X {big(z.xSample)}; B {big(z.bSample)}</td>
          </tr>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}><Translated vi="B trên mỗi phần tử của X (Eq. 19.4)">B per element of X (Eq. 19.4)</Translated></th>
            <td>
              <b data-value="ratio">{r2(ratio)}</b>{' '}<Translated vi={<>so với K² = {limitRatio(s)}</>}>against K² = {limitRatio(s)}</Translated>
              {patchEmbed ? <span className="muted"><Translated vi={<> (stride {s.stride}: các cửa sổ không chồng lấn)</>}> (stride {s.stride}: windows do not overlap)</Translated></span> : null}
            </td>
          </tr>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}><Translated vi="B ở FP32">B in FP32</Translated></th>
            <td data-value="bytes">
              <Translated vi={<>{bytesText(z.bBytesSample)} mỗi mẫu; {bytesText(z.bBytesBatch)} cho batch {s.N} mẫu (X: {bytesText(z.xBytesBatch)})</>}>
              {bytesText(z.bBytesSample)} per sample; {bytesText(z.bBytesBatch)} for the batch of {s.N} (X: {bytesText(z.xBytesBatch)})
              </Translated>
            </td>
          </tr>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}><Translated vi="Giới hạn">Limits</Translated></th>
            <td data-value="flags" data-flags={z.flags.map((f) => `${f.id}:${f.what}`).join(' ')}>
              {z.flags.length === 0 ? (
                <span className="chip ok"><Translated vi="chưa chạm giới hạn nào">none reached</Translated></span>
              ) : (
                z.flags.map((f) => (
                  <div key={f.id + f.what}>
                    <span className="chip fail">{f.what === 'sample' ? tr('one sample', 'một mẫu') : tr(`the batch of ${s.N}`, `batch ${s.N} mẫu`)}</span> {FLAG_TEXT[f.id]}.
                  </div>
                ))
              )}
            </td>
          </tr>
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- where does B[u, v] come from?

const TRAINER_SHAPES: { label: string; s: ConvShape; X: number[] }[] = [
  { get label() { return tr("the book's example: C = 3, 3 × 3, K = 2", 'ví dụ của sách: C = 3, 3 × 3, K = 2'); }, s: BOOK_SHAPE, X: BOOK_X },
  { label: 'C = 3, 4 × 4, K = 2', s: { C: 3, H: 4, W: 4, K: 2, M: 1, N: 1 }, X: makeData(48, 1931, 0, 9) },
];

type StepId = 'c' | 'p' | 'q' | 'oh' | 'ow' | 'flat';
interface Step {
  id: StepId;
  ask: string;
  answer: (b: BSource) => number;
  work: (s: ConvShape, b: BSource) => string;
}
const STEPS: Step[] = [
  { id: 'c', get ask() { return tr('Channel: u / K²', 'Kênh: u / K²'); }, answer: (b) => b.c, work: (s, b) => `${b.u} / ${s.K * s.K} = ${b.c}` },
  { id: 'p', get ask() { return tr('Row inside the patch: (u % K²) / K', 'Hàng bên trong vùng (patch): (u % K²) / K'); }, answer: (b) => b.p, work: (s, b) => `(${b.u} % ${s.K * s.K}) / ${s.K} = ${b.u % (s.K * s.K)} / ${s.K} = ${b.p}` },
  { id: 'q', get ask() { return tr('Column inside the patch: (u % K²) % K', 'Cột bên trong vùng: (u % K²) % K'); }, answer: (b) => b.q, work: (s, b) => `(${b.u} % ${s.K * s.K}) % ${s.K} = ${b.u % (s.K * s.K)} % ${s.K} = ${b.q}` },
  { id: 'oh', get ask() { return tr('Patch corner, row: v / W_out', 'Góc của vùng, hàng: v / W_out'); }, answer: (b) => b.oh, work: (s, b) => `${b.v} / ${outDims(s).Wout} = ${b.oh}` },
  { id: 'ow', get ask() { return tr('Patch corner, column: v % W_out', 'Góc của vùng, cột: v % W_out'); }, answer: (b) => b.ow, work: (s, b) => `${b.v} % ${outDims(s).Wout} = ${b.ow}` },
  {
    id: 'flat',
    get ask() { return tr("Flat index into the sample's X: c·H·W + (corner row + patch row)·W + (corner column + patch column)", 'Chỉ số phẳng vào X của mẫu: c·H·W + (hàng góc + hàng trong vùng)·W + (cột góc + cột trong vùng)'); },
    answer: (b) => b.flat,
    work: (s, b) => `${b.c}·${s.H * s.W} + (${b.oh} + ${b.p})·${s.W} + (${b.ow} + ${b.q}) = ${b.flat}`,
  },
];

function TrainerDrawing({ s, X, u, v, shown, reverse }: {
  s: ConvShape; X: number[]; u: number | null; v: number | null; shown: Set<StepId>;
  reverse: { pick: { c: number; h: number; w: number } | null; reveal: boolean; onPick: (c: number, h: number, w: number) => void } | null;
}) {
  const P = pixels(s);
  const R = ckk(s);
  const cx = 20;
  const cxs = 24;
  const mapX = (c: number) => 16 + c * (s.W * cxs + 14);
  const mapY = 30;
  const bc = 20;
  const bx0 = mapX(s.C) + 24;
  const by0 = 30;
  const b = u != null && v != null ? bToX(s, u, v) : null;
  const B = unfold(s, X);
  const copies = reverse?.pick && reverse.reveal ? xToB(s, reverse.pick.c, reverse.pick.h, reverse.pick.w) : [];
  const xCentre = (c: number, h: number, w: number) => ({ x: mapX(c) + w * cxs + cxs / 2, y: mapY + h * cxs + cxs / 2 });
  const bCentre = (uu: number, vv: number) => ({ x: bx0 + vv * bc + bc / 2, y: by0 + uu * bc + bc / 2 });
  const width = bx0 + P * bc + 20;
  const height = Math.max(by0 + R * bc, mapY + s.H * cxs + 90) + 10;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={tr("B and the input maps it copies", 'B và các bản đồ đầu vào mà nó sao chép')} data-layout="narrow" style={{ width: '100%', maxWidth: width * 1.25, display: 'block' }}>
      <defs>
        <marker id="ch19-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">{/* i18n-ignore: SVG attribute */}
          <path d="M0,0 L10,5 L0,10 z" fill="var(--ink)" />
        </marker>
      </defs>
      <text x={16} y={16} fontSize="11" fontWeight="600" style={{ fill: 'var(--global)' }}><Translated vi="X (một mẫu)">X (one sample)</Translated></text>
      {Array.from({ length: s.C }, (_, c) => (
        <g key={c} data-map={c}>
          <text x={mapX(c)} y={mapY - 4} fontSize="10" fontWeight={b && shown.has('c') && b.c === c ? 700 : 400} style={{ fill: b && shown.has('c') && b.c === c ? 'var(--ink)' : 'var(--ink-2)' }}>
            <Translated vi={<>bản đồ {c}</>}>map {c}</Translated>
          </text>
          {Array.from({ length: s.H * s.W }, (_, i) => {
            const h = Math.floor(i / s.W);
            const w = i % s.W;
            const target = b && shown.has('flat') && b.c === c && b.h === h && b.w === w;
            const picked = reverse?.pick && reverse.pick.c === c && reverse.pick.h === h && reverse.pick.w === w;
            const inMap = b && shown.has('c') && b.c === c;
            return (
              <g key={i} style={{ cursor: reverse ? 'pointer' : undefined }} data-xcell={`${c},${h},${w}`}
                onClick={reverse ? () => reverse.onPick(c, h, w) : undefined}>
                <rect x={mapX(c) + w * cxs + 1} y={mapY + h * cxs + 1} width={cxs - 2} height={cxs - 2} rx="2"
                  fill={target || picked ? 'var(--global)' : inMap ? 'var(--global-wash)' : 'var(--cell)'} stroke={target || picked || inMap ? 'var(--global)' : 'var(--rule)'} />
                <text x={mapX(c) + w * cxs + cxs / 2} y={mapY + h * cxs + cxs / 2 + 4} fontSize="11.5" textAnchor="middle"
                  style={{ fill: target || picked ? 'var(--page)' : 'var(--ink)' }}>{X[c * s.H * s.W + i]}</text>
              </g>
            );
          })}
          {b && shown.has('c') && shown.has('oh') && shown.has('ow') && b.c === c ? (
            <rect x={mapX(c) + b.w0 * cxs - 1} y={mapY + b.h0 * cxs - 1} width={s.K * cxs + 2} height={s.K * cxs + 2} fill="none" stroke="var(--ink)" strokeWidth="2" strokeDasharray="4 2" />
          ) : null}
        </g>
      ))}
      {b && (shown.has('p') || shown.has('q')) ? (
        <g>
          <text x={16} y={mapY + s.H * cxs + 24} fontSize="10" style={{ fill: 'var(--ink-2)' }}><Translated vi="độ lệch trong vùng">offset in the patch</Translated></text>
          {Array.from({ length: s.K * s.K }, (_, i) => {
            const p = Math.floor(i / s.K);
            const q = i % s.K;
            const hit = (!shown.has('p') || p === b.p) && (!shown.has('q') || q === b.q);
            return <rect key={i} x={16 + q * cx} y={mapY + s.H * cxs + 30 + p * cx} width={cx - 2} height={cx - 2} rx="2" fill={hit ? 'var(--global-wash)' : 'var(--cell)'} stroke={hit ? 'var(--global)' : 'var(--rule)'} />;
          })}
          {shown.has('p') && shown.has('q') ? (
            <text x={16 + s.K * cx + 8} y={mapY + s.H * cxs + 44} fontSize="10" style={{ fill: 'var(--ink)' }}>({b.p}, {b.q})</text>
          ) : null}
        </g>
      ) : null}

      <text x={bx0} y={16} fontSize="11" fontWeight="600" style={{ fill: 'var(--global)' }}>B: {R} × {P}</text>
      {B.map((val, i) => {
        const uu = Math.floor(i / P);
        const vv = i % P;
        const asked = uu === u && vv === v;
        const copy = copies.some((e) => e.u === uu && e.v === vv);
        return (
          <g key={i} data-bcell={`${uu},${vv}`}>
            <rect x={bx0 + vv * bc + 1} y={by0 + uu * bc + 1} width={bc - 2} height={bc - 2} rx="1.5"
              fill={asked || copy ? 'var(--global)' : 'var(--cell)'} stroke={asked || copy ? 'var(--global)' : 'var(--rule)'} />
            <text x={bx0 + vv * bc + bc / 2} y={by0 + uu * bc + bc / 2 + 4} fontSize="10.5" textAnchor="middle" style={{ fill: asked || copy ? 'var(--page)' : 'var(--ink-2)' }}>{val}</text>
          </g>
        );
      })}
      {Array.from({ length: s.C - 1 }, (_, c) => (
        <line key={c} x1={bx0 - 2} x2={bx0 + P * bc + 2} y1={by0 + (c + 1) * s.K * s.K * bc} y2={by0 + (c + 1) * s.K * s.K * bc} stroke="var(--ink-3)" />
      ))}
      {b && shown.has('flat') ? (() => {
        const from = bCentre(b.u, b.v);
        const to = xCentre(b.c, b.h, b.w);
        return <line x1={from.x - bc / 2} y1={from.y} x2={to.x + cxs / 2} y2={to.y} stroke="var(--ink)" strokeWidth="1.6" markerEnd="url(#ch19-arrow)" />;
      })() : null}
      {reverse?.pick && reverse.reveal
        ? copies.map((e, i) => {
            const from = xCentre(reverse.pick!.c, reverse.pick!.h, reverse.pick!.w);
            const to = bCentre(e.u, e.v);
            return <line key={i} x1={from.x + cxs / 2} y1={from.y} x2={to.x - bc / 2} y2={to.y} stroke="var(--ink)" strokeWidth="1.2" markerEnd="url(#ch19-arrow)" opacity="0.8" />;
          })
        : null}
    </svg>
  );
}

function IndexTrainer() {
  const [which, setWhich] = useState(0);
  const [mode, setMode] = useState<'forward' | 'reverse'>('forward');
  const [u, setU] = useState(10);
  const [v, setV] = useState(2);
  const [step, setStep] = useState(0);
  const [given, setGiven] = useState<Partial<Record<StepId, { value: number; ok: boolean }>>>({});
  const [text, setText] = useState('');
  const [pick, setPick] = useState<{ c: number; h: number; w: number } | null>(null);
  const [guess, setGuess] = useState('');
  const [reveal, setReveal] = useState(false);
  const { s, X } = TRAINER_SHAPES[which];
  const P = pixels(s);
  const R = ckk(s);
  const b = bToX(s, Math.min(u, R - 1), Math.min(v, P - 1));
  const shown = new Set(STEPS.slice(0, step).map((x) => x.id));
  const restart = (nu: number, nv: number) => {
    setU(nu);
    setV(nv);
    setStep(0);
    setGiven({});
    setText('');
  };
  const check = () => {
    if (text.trim() === '' || step >= STEPS.length) return;
    const st = STEPS[step];
    const value = Number(text);
    setGiven((g) => ({ ...g, [st.id]: { value, ok: value === st.answer(b) } }));
    setStep(step + 1);
    setText('');
  };
  const copies = pick ? xToB(s, pick.c, pick.h, pick.w) : [];
  return (
    <Bench title={tr("Where does B[u, v] come from?", 'B[u, v] đến từ đâu?')} task={tr("Work Eq. 19.5 one piece at a time; each answer is drawn as you give it.", 'Giải Eq. 19.5 từng phần một; mỗi câu trả lời được vẽ ra ngay khi bạn đưa ra.')}>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Lớp">Layer</Translated>
          <select value={which} onChange={(e) => { setWhich(+e.target.value); restart(+e.target.value === 0 ? 10 : 7, +e.target.value === 0 ? 2 : 5); setPick(null); setReveal(false); }} aria-label={tr("Trainer layer", 'Lớp của trình luyện')}>
            {TRAINER_SHAPES.map((t, i) => <option key={i} value={i}>{t.label}</option>)}
          </select>
        </label>
        <span className="seg" role="group" aria-label={tr("Direction", 'Chiều')}>
          <button aria-pressed={mode === 'forward'} onClick={() => setMode('forward')}><Translated vi="B[u, v] sang X">B[u, v] to X</Translated></button>
          <button aria-pressed={mode === 'reverse'} onClick={() => setMode('reverse')}><Translated vi="X sang mọi B[u, v]">X to every B[u, v]</Translated></button>
        </span>
      </div>

      {mode === 'forward' ? (
        <>
          <div className="row" style={{ marginBottom: '0.8rem' }}>
            <label className="field">
              <Translated vi={<>u (0 đến {R - 1})</>}>u (0 to {R - 1})</Translated>
              <input type="number" min={0} max={R - 1} value={u} aria-label="u" onChange={(e) => restart(Math.max(0, Math.min(R - 1, +e.target.value || 0)), v)} />
            </label>
            <label className="field">
              <Translated vi={<>v (0 đến {P - 1})</>}>v (0 to {P - 1})</Translated>
              <input type="number" min={0} max={P - 1} value={v} aria-label="v" onChange={(e) => restart(u, Math.max(0, Math.min(P - 1, +e.target.value || 0)))} />
            </label>
            <button onClick={() => restart(Math.floor(Math.random() * R), Math.floor(Math.random() * P))}><Translated vi="Phần tử ngẫu nhiên">Random element</Translated></button>
            <span className="muted">K = {s.K}, W_out = {outDims(s).Wout}, H × W = {s.H} × {s.W}</span>
          </div>
          <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 6fr) minmax(0, 5fr)' }}>
            <TrainerDrawing s={s} X={X} u={b.u} v={b.v} shown={shown} reverse={null} />
            <div>
              <ol style={{ margin: 0, paddingLeft: '1.2rem' }} data-steps>
                {STEPS.map((st, i) => {
                  const g = given[st.id];
                  return (
                    <li key={st.id} style={{ marginBottom: '0.45rem', opacity: i > step ? 0.45 : 1 }} data-step={st.id}>
                      <div>{st.ask}</div>
                      {g ? (
                        <div className={g.ok ? 't-ok' : 't-fail'}>
                          {g.ok ? tr('Right: ', 'Đúng: ') : tr(`You said ${g.value}. `, `Bạn trả lời ${g.value}. `)}
                          <code>{st.work(s, b)}</code>
                        </div>
                      ) : i === step ? (
                        <div className="row" style={{ gap: '0.4rem' }}>
                          <input type="number" value={text} aria-label={tr("your answer", 'câu trả lời của bạn')} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && check()} style={{ width: '6rem' }} />
                          <button className="primary" onClick={check} disabled={text.trim() === ''}><Translated vi="Kiểm tra">Check</Translated></button>
                        </div>
                      ) : null}
                    </li>
                  );
                })}
              </ol>
              {step >= STEPS.length ? (
                <p style={{ margin: '0.5rem 0 0' }} data-result>
                  B[{b.u}, {b.v}] ← X[n, {b.c}, {b.h}, {b.w}] = <b>{X[b.flat]}</b>.{' '}
                  {Object.values(given).every((g) => g?.ok) ? <span className="chip ok"><Translated vi="cả sáu đều đúng">all six right</Translated></span> : <span className="chip fail"><Translated vi={<>{Object.values(given).filter((g) => !g?.ok).length} chỗ cần xem lại</>}>{Object.values(given).filter((g) => !g?.ok).length} to look at again</Translated></span>}
                </p>
              ) : null}
            </div>
          </div>
        </>
      ) : (
        <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 6fr) minmax(0, 5fr)' }}>
          <TrainerDrawing s={s} X={X} u={null} v={null} shown={new Set()} reverse={{ pick, reveal, onPick: (c, h, w) => { setPick({ c, h, w }); setReveal(false); setGuess(''); } }} />
          <div>
            {pick ? (
              <>
                <p style={{ marginTop: 0 }}>
                  <Translated vi={<>Bao nhiêu phần tử của B sao chép X[n, {pick.c}, {pick.h}, {pick.w}]?</>}>How many entries of B copy X[n, {pick.c}, {pick.h}, {pick.w}]?</Translated>
                </p>
                <div className="row" style={{ gap: '0.4rem' }}>
                  <input type="number" value={guess} aria-label={tr("your count", 'số đếm của bạn')} onChange={(e) => setGuess(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && guess.trim() && setReveal(true)} style={{ width: '6rem' }} />
                  <button className="primary" disabled={guess.trim() === ''} onClick={() => setReveal(true)}><Translated vi="Kiểm tra">Check</Translated></button>
                </div>
                {reveal ? (
                  <p data-reverse-result>
                    <span className={Number(guess) === copies.length ? 't-ok' : 't-fail'}>{Number(guess) === copies.length ? tr('Right', 'Đúng') : tr(`It is ${copies.length}`, `Đáp án là ${copies.length}`)}</span>:{' '}
                    <Translated vi={<>{copies.map((e) => `B[${e.u}, ${e.v}] (góc ${Math.floor(e.v / outDims(s).Wout)}, ${e.v % outDims(s).Wout}; độ lệch ${e.p}, ${e.q})`).join('; ')}. Mỗi bản sao là một vùng chứa pixel đó, nên số bản sao là số cửa sổ K × K phủ lên nó.</>}>
                    {copies.map((e) => `B[${e.u}, ${e.v}] (corner ${Math.floor(e.v / outDims(s).Wout)}, ${e.v % outDims(s).Wout}; offset ${e.p}, ${e.q})`).join('; ')}. Each copy is one patch that
                    contains the pixel, so the count is the number of K × K windows covering it.
                    </Translated>
                  </p>
                ) : null}
              </>
            ) : (
              <p className="muted" style={{ marginTop: 0 }}><Translated vi="Nhấp vào một pixel của X.">Click a pixel of X.</Translated></p>
            )}
          </div>
        </div>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- Fig. 19.11 as printed and corrected

function YGrid({ s, status, Y }: { s: KShape; status: string[]; Y: number[] }) {
  const { Hout, Wout } = outDims(s);
  const P = Hout * Wout;
  const cols = Math.min(s.M, 8);
  const rows = Math.ceil(s.M / cols);
  const gap = 8;
  const cs = Math.max(2, Math.min(26, Math.floor((600 - (cols - 1) * gap) / (cols * Wout))));
  const mw = Wout * cs;
  const mh = Hout * cs;
  const width = cols * mw + (cols - 1) * gap;
  const height = rows * (mh + 16) + 4;
  const fill = (st: string) => (st === 'ok' ? 'var(--ok-wash)' : st === 'clobbered' ? 'var(--fail-wash)' : 'var(--fail)');
  return (
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={tr("Every output, right or wrong", 'Mọi đầu ra, đúng hay sai')} data-layout="narrow" style={{ width: '100%', maxWidth: Math.max(width, 160), display: 'block' }}>
      {Array.from({ length: s.M }, (_, m) => {
        const ox = (m % cols) * (mw + gap);
        const oy = Math.floor(m / cols) * (mh + 16) + 12;
        return (
          <g key={m}>
            <text x={ox} y={oy - 3} fontSize="9" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>bản đồ {m}</>}>map {m}</Translated></text>
            {Array.from({ length: P }, (_, i) => {
              const st = status[m * P + i];
              const x = ox + (i % Wout) * cs;
              const y = oy + Math.floor(i / Wout) * cs;
              return (
                <g key={i}>
                  <rect x={x} y={y} width={cs - (cs > 4 ? 1 : 0)} height={cs - (cs > 4 ? 1 : 0)} fill={fill(st)} stroke={st === 'clobbered' && cs > 4 ? 'var(--fail)' : 'none'} />
                  {cs >= 18 ? (
                    <text x={x + cs / 2} y={y + cs / 2 + 4} fontSize="11" textAnchor="middle" style={{ fill: st === 'wrong' ? 'var(--page)' : 'var(--ink)' }}>
                      {Number.isNaN(Y[m * P + i]) ? '?' : Y[m * P + i]}
                    </text>
                  ) : null}
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

const COMBOS = [
  { ceil: false, bounds: false, get label() { return tr('as printed', 'như được in'); } },
  { ceil: true, bounds: false, get label() { return tr('ceiling only', 'chỉ làm tròn lên'); } },
  { ceil: false, bounds: true, get label() { return tr('bounds checks only', 'chỉ kiểm tra biên'); } },
  { ceil: true, bounds: true, get label() { return tr('both (the lab)', 'cả hai (bản của lab)'); } },
];

function ErratumHunt() {
  const [id, setId] = useState<HuntId>('c1');
  const [ceil, setCeil] = useState(false);
  const [bounds, setBounds] = useState(false);
  const [bug, setBug] = useState(false);
  const h = huntShape(id);
  const s = h.shape;
  const r = useMemo(() => runKernel(s, { ceil, bounds, bug }), [s, ceil, bounds, bug]);
  const table = useMemo(() => HUNT_SHAPES.map((x) => COMBOS.map((c) => runKernel(x.shape, { ceil: c.ceil, bounds: c.bounds, bug: false }))), []);
  const mult = multiples(s);
  const needed = Math.ceil(ckk(s) / s.T);
  const clob = r.status.filter((x) => x === 'clobbered').length;
  const yes = (b: boolean) => (b ? <span className="t-ok"><Translated vi="có">yes</Translated></span> : <span className="t-fail"><Translated vi="không">no</Translated></span>);
  return (
    <Bench
      title={tr("Fig. 19.11 as printed and corrected", 'Hình 19.11 như được in và đã sửa')}
      task={tr("Find the shapes the printed kernel gets wrong, and the smallest fix for each.", 'Tìm các hình dạng mà kernel được in tính sai, và cách sửa nhỏ nhất cho từng hình dạng.')}
      foot={<Translated vi={<>Xanh: đầu ra bằng kết quả của vòng lặp lồng nhau. Đỏ: một giá trị sai. Đỏ nhạt có viền: một ô mà một luồng nằm ngoài phạm vi của Y cũng đã ghi vào.</>}><>Green: the output equals the loop nest's. Red: a wrong value. Pale red with an outline: a slot that a thread outside Y's range also stored to.</></Translated>}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Hình dạng">Shape</Translated>
          <select value={id} onChange={(e) => setId(e.target.value as HuntId)} aria-label={tr("Shape", 'Hình dạng')}>
            {HUNT_SHAPES.map((x) => <option key={x.id} value={x.id}>{tr(x.label, x.id === 'book' ? 'ví dụ của sách, T = 2' : x.label)}</option>)}
          </select>
        </label>
        <span className="seg" role="group" aria-label={tr("Phase count", 'Số pha')}>
          <button aria-pressed={!ceil} onClick={() => setCeil(false)}><Translated vi="số pha: (C·K·K) / T, như được in">phases: (C·K·K) / T, as printed</Translated></button>
          <button aria-pressed={ceil} onClick={() => setCeil(true)}><Translated vi="số pha: làm tròn lên">phases: ceiling</Translated></button>
        </span>
        <span className="seg" role="group" aria-label={tr("Bounds checks", 'Kiểm tra biên')}>
          <button aria-pressed={!bounds} onClick={() => setBounds(false)}><Translated vi="không kiểm tra biên">no bounds checks</Translated></button>
          <button aria-pressed={bounds} onClick={() => setBounds(true)}><Translated vi="có kiểm tra biên">bounds checks</Translated></button>
        </span>
        <label className="check">
          <input type="checkbox" checked={bug} onChange={(e) => setBug(e.target.checked)} /> <Translated vi="lỗi cài sẵn: W thay cho W − K + 1 trong phép gom (gather)">planted bug: W in place of W − K + 1 in the gather</Translated>
        </label>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 2fr)' }}>
        <div>
          <div className="mx-label"><Translated vi={<>Y: {s.M} bản đồ {outDims(s).Hout} × {outDims(s).Wout}, lưới {r.grid.x} × {r.grid.y} khối {s.T} × {s.T}</>}>Y: {s.M} maps of {outDims(s).Hout} × {outDims(s).Wout}, grid {r.grid.x} × {r.grid.y} blocks of {s.T} × {s.T}</Translated></div>
          <YGrid s={s} status={r.status} Y={r.Y} />
        </div>
        <table className="data" style={{ margin: 0 }} data-panel="hunt">
          <tbody>
            <tr><th><Translated vi="Số pha đã chạy">Phases run</Translated></th><td data-value="phases"><Translated vi={<>{r.phases} (C·K·K = {ckk(s)} cần {needed})</>}>{r.phases} (C·K·K = {ckk(s)} needs {needed})</Translated></td></tr>
            <tr><th><Translated vi="Số hạng mỗi đầu ra">Terms per output</Translated></th><td data-value="terms"><Translated vi={<>{r.termsUsed} trên {r.terms}{r.extraTerms ? `, cộng thêm ${r.extraTerms} vượt quá cuối (${bounds ? 'điền 0' : 'giá trị rác'})` : ''}</>}>{r.termsUsed} of {r.terms}{r.extraTerms ? `, plus ${r.extraTerms} past the end (${bounds ? 'zero-filled' : 'garbage'})` : ''}</Translated></td></tr>
            <tr><th><Translated vi="Số đầu ra sai">Outputs wrong</Translated></th><td data-value="wrong" data-wrong={r.wrong}><Translated vi={<>{big(r.wrong)} trên {big(r.status.length)}{clob ? ` (${big(clob)} bị một lần ghi đi lạc đè lên)` : ''}</>}>{big(r.wrong)} of {big(r.status.length)}{clob ? ` (${big(clob)} hit by a stray store)` : ''}</Translated></td></tr>
            <tr><th><Translated vi="Số lần đọc ngoài X">Reads outside X</Translated></th><td data-value="oobx">{big(r.oobX)}{r.oobX ? tr(': AddressSanitizer stops on the first in the emulator', ': AddressSanitizer dừng ở lần đầu tiên trong trình giả lập') : ''}</td></tr>
            <tr><th><Translated vi="Số lần đọc ngoài F">Reads outside F</Translated></th><td>{big(r.oobF)}</td></tr>
            <tr><th><Translated vi="Số lần ghi vượt quá cuối Y">Stores past the end of Y</Translated></th><td>{big(r.outsideWrites)}</td></tr>
            <tr><th><Translated vi="Bội số của T">Multiples of T</Translated></th><td>M {yes(mult.M)}, H_out·W_out {yes(mult.P)}, C·K·K {yes(mult.CKK)}</td></tr>{/* i18n-ignore: dimension names */}
            <tr>
              <th><Translated vi="Kết luận">Verdict</Translated></th>
              <td data-verdict={r.correct ? 'correct' : 'wrong'}>{r.correct ? <span className="chip ok"><Translated vi="khớp với vòng lặp lồng nhau">matches the loop nest</Translated></span> : <span className="chip fail"><Translated vi="sai">wrong</Translated></span>}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <h5 style={{ margin: '1.1rem 0 0.4rem', fontFamily: 'var(--sans)' }}><Translated vi="Mọi hình dạng với mọi cách sửa (không có lỗi cài sẵn)">Every shape against every fix (no planted bug)</Translated></h5>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0, minWidth: '36rem' }} data-table="outcomes">
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}><Translated vi="Hình dạng">Shape</Translated></th>
              {COMBOS.map((c) => <th key={c.label}>{c.label}</th>)}
            </tr>
          </thead>
          <tbody>
            {HUNT_SHAPES.map((x, i) => (
              <tr key={x.id}>
                <td style={{ paddingLeft: '0.6rem' }}>{tr(x.label, x.id === 'book' ? 'ví dụ của sách, T = 2' : x.label)}</td>
                {table[i].map((res, j) => (
                  <td key={j} data-outcome={res.correct ? 'correct' : 'wrong'}>
                    {res.correct ? <span className="chip ok"><Translated vi="đúng">correct</Translated></span> : <span className="chip fail"><Translated vi={<>{big(res.wrong)} sai</>}>{big(res.wrong)} wrong</Translated></span>}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- one warp's accesses (Exercise 4)

function LaneStrip({ label, addrs, runs }: { label: string; addrs: (number | null)[]; runs: Run[] }) {
  const runOf = (lane: number) => runs.findIndex((r) => lane >= r.firstLane && lane <= r.lastLane);
  return (
    <div style={{ marginBottom: '0.7rem' }}>
      <div className="mx-label t-global">
        <Translated vi={<>{label}: {runs.length} dải ({runs.map((r) => r.length).join(' + ')} số float)</>}>
        {label}: {runs.length} run{runs.length === 1 ? '' : 's'} ({runs.map((r) => r.length).join(' + ')} floats)
        </Translated>
      </div>
      <div className="mx global" style={{ display: 'flex', flexWrap: 'wrap', rowGap: 4 }}>
        {addrs.map((a, lane) => {
          const ri = runOf(lane);
          const prevLane = lane > 0 ? runOf(lane - 1) : ri;
          const brk = a !== null && lane > 0 && ri !== prevLane && ri >= 0;
          const prev = brk ? runs[ri - 1] : null;
          return (
            <Fragment key={lane}>
              {brk && prev ? (
                <span aria-hidden data-jump style={{ border: 'none', background: 'transparent', minWidth: 'auto', padding: '0 0.15rem', color: 'var(--ink-2)', fontSize: '0.68rem' }}>
                  {a - (prev.start + prev.length - 1) > 0 ? `+${a - (prev.start + prev.length - 1)}` : `${a - (prev.start + prev.length - 1)}`}
                </span>
              ) : null}
              <span className={a === null ? 'void' : ri % 2 ? 'on' : ''} title={tr(`lane ${lane}`, 'lane ${lane}')} data-lane={lane} style={{ minWidth: '1.9rem', height: '1.5rem', fontSize: '0.68rem' }}>
                {a === null ? '·' : a}
              </span>
            </Fragment>
          );
        })}
      </div>
    </div>
  );
}

function BankStrip({ label, perBank }: { label: string; perBank: { bank: number; lanes: number[]; words: number[] }[] }) {
  return (
    <div style={{ marginBottom: '0.6rem' }}>
      <div className="mx-label t-shared">{label}</div>
      <div className="mx shared" style={{ gridTemplateColumns: 'repeat(16, auto)' }}>
        {perBank.map((b) => (
          <span key={b.bank} className={b.lanes.length ? 'on' : ''} title={tr(`bank ${b.bank}: ${b.lanes.length} lanes, ${b.words.length} word(s)`, `bank ${b.bank}: ${b.lanes.length} lane, ${b.words.length} word`)} style={{ minWidth: '1.5rem', height: '1.4rem', fontSize: '0.66rem' }}>
            {b.lanes.length ? `${b.lanes.length}` : ''}
            <sub>{b.bank}</sub>
          </span>
        ))}
      </div>
    </div>
  );
}

function WarpExplorer() {
  const [id, setId] = useState<HuntId>('c3');
  const [bx, setBx] = useState(0);
  const [ph, setPh] = useState(0);
  const [warp, setWarp] = useState(0);
  const [k, setK] = useState(0);
  const s = huntShape(id).shape;
  const g = gridOf(s);
  const nph = phasesOf(s, CORRECTED);
  const bxx = Math.min(bx, g.x - 1);
  const phh = Math.min(ph, nph - 1);
  const w = warpView(s, bxx, 0, phh, warp, k);
  const nb = w.banks;
  const rowsOfWarp = [...new Set(w.lanes.map((l) => l.ty))];
  const bankRows: [string, typeof nb.fdsWrite, string][] = [
    ['Fds[ty][tx] = …', nb.fdsWrite, tr('store', 'ghi')],
    ['Bds[ty][tx] = …', nb.bdsWrite, tr('store', 'ghi')],
    [`Fds[ty][${k}]`, nb.fdsRead, tr('read', 'đọc')],
    [`Bds[${k}][tx]`, nb.bdsRead, tr('read', 'đọc')],
  ];
  return (
    <Bench
      title={tr("One warp's accesses in Fig. 19.11", 'Các truy cập của một warp trong Hình 19.11')}
      task={tr("For Exercise 4: are the global accesses coalesced, and do the shared accesses conflict?", 'Cho Bài tập 4: các truy cập toàn cục có được hợp nhất không, và các truy cập bộ nhớ chia sẻ có xung đột không?')}
      foot={<Translated vi={<>Địa chỉ là chỉ số phần tử (số float) vào F, X và Y; các lane trong cùng một dải được tô cùng màu, và con số giữa các dải là bước nhảy. Số bank nằm dưới mỗi ô. <TierLegend tiers={['shared', 'global']} /></>}><>Addresses are element indices (floats) into F, X and Y; lanes of one run are shaded alike, and the number between runs is the jump. Bank numbers are under each cell. <TierLegend tiers={['shared', 'global']} /></></Translated>}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Hình dạng (T = 16)">Shape (T = 16)</Translated>
          <select value={id} onChange={(e) => { setId(e.target.value as HuntId); setBx(0); setPh(0); }} aria-label={tr("Warp shape", 'Hình dạng của warp')}>
            {HUNT_SHAPES.filter((x) => x.shape.T === 16).map((x) => <option key={x.id} value={x.id}>{tr(x.label, x.id === 'book' ? 'ví dụ của sách, T = 2' : x.label)}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Cột khối bx">Block column bx</Translated>
          <select value={bxx} onChange={(e) => setBx(+e.target.value)} aria-label={tr("Block column", 'Cột khối')}>
            {Array.from({ length: g.x }, (_, i) => <Translated key={i} vi={<option value={i}>{i} (pixel {i * 16} đến {i * 16 + 15})</option>}><option key={i} value={i}>{i} (pixels {i * 16} to {i * 16 + 15})</option></Translated>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Pha">Phase</Translated>
          <select value={phh} onChange={(e) => setPh(+e.target.value)} aria-label={tr("Phase", 'Pha')}>
            {Array.from({ length: nph }, (_, i) => <Translated key={i} vi={<option value={i}>{i} (u = {i * 16} đến {i * 16 + 15})</option>}><option key={i} value={i}>{i} (u = {i * 16} to {i * 16 + 15})</option></Translated>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Warp của khối">Warp of the block</Translated>
          <select value={warp} onChange={(e) => setWarp(+e.target.value)} aria-label={tr("Warp", 'Warp')}>
            {Array.from({ length: 8 }, (_, i) => <Translated key={i} vi={<option value={i}>{i} (ty {2 * i} và {2 * i + 1})</option>}><option key={i} value={i}>{i} (ty {2 * i} and {2 * i + 1})</option></Translated>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Bước trong k">Inner step k</Translated>
          <select value={k} onChange={(e) => setK(+e.target.value)} aria-label={tr("Inner step", 'Bước trong')}>
            {Array.from({ length: 16 }, (_, i) => <option key={i} value={i}>{i}</option>)}
          </select>
        </label>
      </div>
      <p className="muted" style={{ marginTop: 0 }}>
        <Translated vi={<>Khối ({bxx}, 0), pha {phh}: các luồng của warp này có ty = {rowsOfWarp.join(' và ')}, nên chúng gom hàng u = {rowsOfWarp.map((t) => phh * 16 + t).join(' và ')} của B; W_out = {outDims(s).Wout}, K = {s.K}.</>}>
        Block ({bxx}, 0), phase {phh}: this warp's threads have ty = {rowsOfWarp.join(' and ')}, so they gather row u = {rowsOfWarp.map((t) => phh * 16 + t).join(' and ')} of B; W_out = {outDims(s).Wout}, K = {s.K}.
        </Translated>
      </p>
      <LaneStrip label={tr("F load, line 23: F[Row·C·K·K + ph·T + tx]", 'Nạp F, dòng 23: F[Row·C·K·K + ph·T + tx]')} addrs={w.lanes.map((l) => l.f)} runs={w.fRuns} />
      <LaneStrip label={tr("X gather, line 28 (Eq. 19.5)", 'Gom X, dòng 28 (Eq. 19.5)')} addrs={w.lanes.map((l) => l.x)} runs={w.xRuns} />
      <LaneStrip label={tr("Y store, line 37: Y[Row·P + Col]", 'Ghi Y, dòng 37: Y[Row·P + Col]')} addrs={w.lanes.map((l) => l.y)} runs={w.yRuns} />
      <p data-readout="gather" style={{ margin: '0 0 0.9rem' }}>
        <Translated vi={<>Phép gom: {w.xRunsByTy.map((x) => `ty ${x.ty} trong ${x.runs.length} dải`).join(', ')}{w.xJumps.length ? <>; bước nhảy giữa các dải là {[...new Set(w.xJumps)].join(' và ')} (K = {s.K}), tại chỗ v sang hàng đầu ra kế tiếp.</> : '; không có chỗ xuống hàng nào bên trong warp này.'}</>}>
        The gather: {w.xRunsByTy.map((x) => `ty ${x.ty} in ${x.runs.length} run${x.runs.length === 1 ? '' : 's'}`).join(', ')}
        {w.xJumps.length ? <>; the jumps between runs are {[...new Set(w.xJumps)].join(' and ')} (K = {s.K}), where v reaches the next output row.</> : '; no row wrap inside this warp.'}
        </Translated>
      </p>
      <table className="data" style={{ background: 'var(--page)', margin: '0 0 0.8rem' }} data-table="banks">
        <thead>
          <tr><th style={{ paddingLeft: '0.6rem' }}><Translated vi="Truy cập bộ nhớ chia sẻ">Shared access</Translated></th><th><Translated vi="số word khác nhau">distinct words</Translated></th><th><Translated vi="số bank được dùng">banks used</Translated></th><th><Translated vi={<>số lượt (mô hình {NVIDIA_BANKS} bank)</>}>passes ({NVIDIA_BANKS}-bank model)</Translated></th></tr>
        </thead>
        <tbody>
          {bankRows.map(([name, b, kind]) => (
            <tr key={name}>
              <td style={{ paddingLeft: '0.6rem' }}><code>{name}</code> ({kind})</td>
              <td>{new Set(b.lanes.map((l) => l.word)).size}</td>
              <td>{b.banksUsed}</td>
              <td data-passes={b.passes}>{b.passes === 1 ? <span className="chip ok"><Translated vi="1: không xung đột">1: no conflict</Translated></span> : <span className="chip fail"><Translated vi={<>xung đột {b.passes} chiều</>}>{b.passes}-way</Translated></span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="cols">
        <BankStrip label={tr(`Fds[ty][${k}]: lanes per bank`, `Fds[ty][${k}]: số lane mỗi bank`)} perBank={nb.fdsRead.perBank} />
        <BankStrip label={tr(`Bds[${k}][tx]: lanes per bank`, `Bds[${k}][tx]: số lane mỗi bank`)} perBank={nb.bdsRead.perBank} />
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- which layers suit a GEMM

const STRIP: PresetId[] = ['c1', 'c3', 'c5', 'early', 'middle', 'late', 'pointwise'];

function ShapesExplorer() {
  const [N, setN] = useState(1);
  const [whole, setWhole] = useState(false);
  return (
    <Bench
      title={tr("Which layers suit a GEMM", 'Những lớp nào hợp với GEMM')}
      task={tr("Which layers leave most of a 16 × 16 tile empty, and what does batching change?", 'Những lớp nào để trống phần lớn một ô 16 × 16, và batching thay đổi điều gì?')}
      foot={tr("Loads are what one thread requests from global memory per output: no caches. Tiles are 16 × 16 blocks of Y.", 'Số lần nạp là những gì một luồng yêu cầu từ bộ nhớ toàn cục cho mỗi đầu ra: không có cache. Các ô là các khối 16 × 16 của Y.')}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <Slider label={tr("N (batch)", 'N (batch)')} value={N} min={1} max={64} onChange={setN} />
        <span className="seg" role="group" aria-label={tr("GEMMs", 'GEMM')}>
          <button aria-pressed={!whole} onClick={() => setWhole(false)}><Translated vi="mỗi mẫu một GEMM">one GEMM per sample</Translated></button>
          <button aria-pressed={whole} onClick={() => setWhole(true)}><Translated vi="cả batch trong một GEMM (cần một phép hoán vị)">whole batch in one GEMM (needs a permute)</Translated></button>
        </span>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0, minWidth: '44rem' }} data-table="shapes">
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}><Translated vi="Lớp">Layer</Translated></th>
              <th><Translated vi="GEMM: M × C·K·K × số cột">GEMM: M × C·K·K × columns</Translated></th>
              <th className="num"><Translated vi="số khối mỗi GEMM">blocks per GEMM</Translated></th>
              <th className="num"><Translated vi="số khối cho cả batch">blocks for the batch</Translated></th>
              <th><Translated vi="tỉ lệ ô là phần đệm">slots that are padding</Translated></th>
              <th className="num"><Translated vi="số pha">phases</Translated></th>
              <th className="num"><Translated vi="số lần nạp mỗi đầu ra: trực tiếp / phân ô">loads per output: direct / tiled</Translated></th>
            </tr>
          </thead>
          <tbody>
            {STRIP.map((pid) => {
              const p = preset(pid);
              const g = gemmShape({ ...p.shape, N }, { wholeBatch: whole });
              return (
                <tr key={pid} data-layer={pid}>
                  <td style={{ paddingLeft: '0.6rem' }}>{tr(p.label, p.id === 'book' ? 'ví dụ của sách' : p.id === 'early' ? 'lớp đầu của lab' : p.id === 'middle' ? 'lớp giữa của lab' : p.id === 'late' ? 'lớp cuối của lab' : p.id === 'pointwise' ? 'lớp 1 × 1 của lab' : p.label)}<div className="muted" style={{ fontSize: '0.72rem' }}>C {p.shape.C}, {p.shape.H} × {p.shape.W}, K {p.shape.K}, M {p.shape.M}</div></td>
                  <td>{big(g.rows)} × {big(g.inner)} × {big(g.cols)}{g.needsPermute ? <div className="muted" style={{ fontSize: '0.72rem' }}><Translated vi="ra dạng M × N × P">comes out M × N × P</Translated></div> : null}</td>
                  <td className="num">{big(g.tilesPerGemm)}</td>
                  <td className="num" data-value="tiles">{big(g.tilesTotal)}</td>
                  <td data-value="padding" data-padding={g.padding.toFixed(4)}>
                    <span style={{ display: 'inline-block', width: 60, height: 9, background: 'var(--rule-soft)', marginRight: 6, verticalAlign: 'middle' }}>
                      <span style={{ display: 'block', width: `${g.padding * 100}%`, height: '100%', background: 'var(--ink-3)' }} />
                    </span>
                    {pct(g.padding)}
                  </td>
                  <td className="num">{g.phases}</td>
                  <td className="num">{big(g.loadsDirect)} / {g.loadsImplicit}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- code from the lab

function ProvidedKernel() {
  const { reports } = useReports();
  const cu = useLabFile(PROVIDED.cu);
  const metal = useLabFile(PROVIDED.metal);
  if (!cu || !metal) return null;
  return (
    <Code
      tabs={[
        { label: tr('CUDA (also HIP, emulator)', 'CUDA (cũng cho HIP, trình giả lập)'), code: cu, tiers: CONV_TIERS, status: tr('tested with lab 19.1: ', 'được kiểm tra bằng lab 19.1: ') + cuStatus(reports) },
        { label: 'Metal', code: metal, tiers: CONV_TIERS, status: tr('tested with lab 19.1: ', 'được kiểm tra bằng lab 19.1: ') + metalStatus(reports) },
      ]}
      caption={
        <>
          <Translated vi={<>Hình 19.11, đã sửa, như lab 19.1 cung cấp: <code>labs/{PROVIDED.cu}</code> và <code>.metal</code>. Bạn không viết kernel này; lab chạy nó cạnh các kernel của bạn và kiểm tra nó với vòng lặp lồng nhau trên CPU.</>}>
          Fig. 19.11, corrected, as lab 19.1 provides it: <code>labs/{PROVIDED.cu}</code> and <code>.metal</code>. You do not write this
          one; the lab runs it beside your kernels and checks it against the CPU loop nest.
          </Translated>
        </>
      }
    />
  );
}

function SolutionRegions() {
  const { reports } = useReports();
  const files = {
    'kernel.cu': useLabFile(LAB_DIR + 'solution/kernel.cu'),
    'kernel.metal': useLabFile(LAB_DIR + 'solution/kernel.metal'),
    'launch.mm': useLabFile(LAB_DIR + 'solution/launch.mm'),
  };
  const [open, setOpen] = useState(false);
  const tabs: CodeTab[] = [];
  for (const name of REGIONS)
    for (const f of SOLUTION_FILES) {
      const r = region(files[f], name);
      if (!r) continue;
      tabs.push({
        label: `${name}, ${f}`,
        code: r.code,
        tiers: f === 'launch.mm' ? {} : CONV_TIERS,
        status: f === 'kernel.cu' ? cuStatus(reports) : metalStatus(reports),
      });
    }
  const loading = Object.values(files).some((x) => x === null);
  return (
    <div className="wide" style={{ maxWidth: '58rem' }} data-part="solution-regions">
      {!open ? (
        <p>
          <button onClick={() => setOpen(true)}><Translated vi="Hiện phần unroll và lời gọi GEMM của lời giải tham chiếu">Show the reference solution's unroll and GEMM call</Translated></button>
        </p>
      ) : loading ? null : tabs.length ? (
        <Code
          tabs={tabs}
          caption={
            <>
              <Translated vi={<>Hai phần của lời giải tham chiếu (solution) của lab 19.1, cắt từ <code>labs/{LAB_DIR}solution/</code> tại các dấu <code>[lesson:…]</code>{' '}của chúng: kernel ghi Bᵀ, và mã host giao F và Bᵀ cho GEMM của lab 6.4, từng mẫu một.</>}>
              Two pieces of lab 19.1's reference solution, cut from <code>labs/{LAB_DIR}solution/</code> at their <code>[lesson:…]</code>{' '}
              markers: the kernel that writes Bᵀ, and the host code that hands F and Bᵀ to lab 6.4's GEMM, one sample at a time.
              </Translated>
            </>
          }
        />
      ) : (
        <p className="muted"><Translated vi="Các tệp lời giải của lab không có các dấu [lesson:unroll] và [lesson:gemm-path] như mong đợi.">The lab's solution files do not carry the expected [lesson:unroll] and [lesson:gemm-path] markers.</Translated></p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- questions

export const questionsS193: Record<string, Question> = {
  'ch19/19.3/q-belems': {
    kind: 'numeric',
    id: 'ch19/19.3/q-belems',
    prompt: <Translated vi={<>LeNet C3: 6 bản đồ đầu vào 14 × 14, K = 5, 16 bản đồ đầu ra. B chứa bao nhiêu phần tử cho một mẫu?</>}><>LeNet C3: 6 input maps of 14 × 14, K = 5, 16 output maps. How many elements does B hold for one sample?</></Translated>,
    answer: 15000,
    explain: (
      <Translated vi={<p>B có C·K·K = 6·25 = 150 hàng và H_out·W_out = 10·10 = 100 cột: 15,000 phần tử, được tạo từ 1,176 giá trị đầu vào (gấp 12.76 lần). 16 bản đồ đầu ra không tham gia: chúng chỉ quyết định chiều cao của F và Y.</p>}>
      <p>
        B has C·K·K = 6·25 = 150 rows and H_out·W_out = 10·10 = 100 columns: 15,000 entries, built from 1,176 input values (12.76 times as
        many). The 16 output maps do not enter: they set the height of F and Y only.
      </p>
      </Translated>
    ),
    source: 'ch. 19 §19.3, pp. 465–467; LeNet sizes p. 458',
    vi: { source: 'ch. 19 §19.3, pp. 465–467; kích thước LeNet p. 458' },
  },
  'ch19/19.3/q-noM': {
    kind: 'choice',
    id: 'ch19/19.3/q-noM',
    prompt: <Translated vi={<>Kích thước của B là C·K·K × H_out·W_out. Vì sao số bản đồ đầu ra M không xuất hiện trong đó?</>}><>B's size is C·K·K × H_out·W_out. Why does the number of output maps M not appear in it?</></Translated>,
    options: [
      { label: <Translated vi="Cả M bản đồ đầu ra dùng chung một B; M chỉ quyết định chiều cao của F và của Y.">All M output maps use the same B; M sets only the height of F and of Y.</Translated>, correct: true },
      { label: <Translated vi="M được gộp vào C·K·K, vốn đã đếm mọi bộ lọc trong tập bộ lọc.">M is folded into C·K·K, which already counts every filter in the bank.</Translated>, why: <Translated vi="C·K·K đếm số hạng của một đầu ra: số bản đồ đầu vào nhân kích thước bộ lọc. M không tham gia vào đó.">C·K·K counts the terms of one output: input maps times filter size. M does not enter it.</Translated> },
      { label: <Translated vi="B được tạo một lần cho mỗi bản đồ đầu ra, nên kích thước của nó tính cho mỗi bản đồ rồi nhân với M.">B is built once per output map, so its size is given per map and multiplied by M.</Translated>, why: <Translated vi="Một B phục vụ cả M bản đồ. Tạo nó cho từng bản đồ sẽ nhân chi phí lên M lần.">One B serves all M maps. Building it per map would multiply the cost by M.</Translated> },
      { label: <Translated vi="M bằng 1 trong ví dụ của sách, nên công thức có thể bỏ nó đi.">M is 1 in the book’s worked example, so the formula can leave it out.</Translated>, why: <Translated vi="Ví dụ có M = 2 và LeNet C3 có M = 16.">The example has M = 2 and LeNet C3 has M = 16.</Translated> },
    ],
    explain: (
      <Translated vi={<p>Một cột của B là các vùng đầu vào của một pixel đầu ra. Mỗi hàng của F (trọng số của một bản đồ đầu ra) được nhân với đúng cột đó, nên M bản đồ dùng chung B và chỉ khác nhau ở hàng F của chúng (p. 467).</p>}>
      <p>
        A column of B is one output pixel's input patches. Each row of F (one output map's weights) is multiplied by that same column, so
        the M maps share B and differ only in their row of F (p. 467).
      </p>
      </Translated>
    ),
    source: 'ch. 19 §19.3, p. 467',
  },
  'ch19/19.3/q-map': {
    kind: 'numeric',
    id: 'ch19/19.3/q-map',
    prompt: (
      <Translated vi={<>C = 3, bản đồ đầu vào 4 × 4, K = 2, nên W_out = 3. B[7, 5] sao chép phần tử phẳng nào của X của mẫu (bố cục C × H × W, thứ tự hàng)?</>}>
      <>
        C = 3, input maps of 4 × 4, K = 2, so W_out = 3. B[7, 5] copies which flat element of the sample's X (laid out C × H × W, row-major)?
      </>
      </Translated>
    ),
    answer: 27,
    explain: (
      <Translated vi={<p>Kênh là 7 / 4 = 1; độ lệch 7 % 4 = 3 là hàng 1, cột 1 của vùng. Góc của vùng là (5 / 3, 5 % 3) = (1, 2). Vậy phần tử là X[n, 1, 1 + 1, 1 + 2] = X[n, 1, 2, 3], chỉ số phẳng 1·16 + 2·4 + 3 = 27.</p>}>
      <p>
        The channel is 7 / 4 = 1; the offset 7 % 4 = 3 is row 1, column 1 of the patch. The patch corner is (5 / 3, 5 % 3) = (1, 2). So the
        element is X[n, 1, 1 + 1, 1 + 2] = X[n, 1, 2, 3], flat index 1·16 + 2·4 + 3 = 27.
      </p>
      </Translated>
    ),
    source: 'Eq. 19.5, ch. 19 §19.3, pp. 469–470',
  },
  'ch19/19.3/q-copies': {
    kind: 'numeric',
    id: 'ch19/19.3/q-copies',
    prompt: <Translated vi={<>Cùng lớp đó (C = 3, bản đồ 4 × 4, K = 2). Bao nhiêu phần tử của B sao chép X[n, 1, 1, 2]?</>}><>Same layer (C = 3, 4 × 4 maps, K = 2). How many entries of B copy X[n, 1, 1, 2]?</></Translated>,
    answer: 4,
    explain: (
      <Translated vi={<p>Mỗi cửa sổ 2 × 2 phủ lên pixel đó cho một bản: các cửa sổ có góc (0, 1), (0, 2), (1, 1) và (1, 2). Cả bốn đều tồn tại trong một đầu ra 3 × 3, nên pixel được sao chép bốn lần, như một pixel bên trong luôn vậy khi K = 2.</p>}>
      <p>
        One for each 2 × 2 window that covers the pixel: windows with corners (0, 1), (0, 2), (1, 1) and (1, 2). All four exist in a 3 × 3
        output, so the pixel is copied four times, as an interior pixel always is when K = 2.
      </p>
      </Translated>
    ),
    source: 'Eq. 19.5 run backwards; ch. 19 §19.3, pp. 466–467',
    vi: { source: 'Eq. 19.5 chạy ngược; ch. 19 §19.3, pp. 466–467' },
  },
  'ch19/19.3/q-phases': {
    kind: 'numeric',
    id: 'ch19/19.3/q-phases',
    prompt: (
      <Translated vi={<>LeNet C3 với TILE_WIDTH = 16 và vòng lặp pha như được in, <code>ph &lt; (C*K*K)/TILE_WIDTH</code>. Bao nhiêu trong 150 số hạng của mỗi đầu ra bị bỏ sót?</>}>
      <>
        LeNet C3 with TILE_WIDTH = 16 and the phase loop as printed, <code>ph &lt; (C*K*K)/TILE_WIDTH</code>. How many of each output's 150
        terms are dropped?
      </>
      </Translated>
    ),
    answer: 6,
    explain: (
      <Translated vi={<p>150 / 16 bị cắt còn 9 pha, phủ u = 0 đến 143: 144 số hạng. 6 số hạng cuối (u = 144 đến 149, phần cuối vùng của bản đồ đầu vào 5) không bao giờ được cộng. Làm tròn lên, 10 pha, sẽ phủ chúng, và kiểm tra biên điền 0 cho u = 150 đến 159.</p>}>
      <p>
        150 / 16 truncates to 9 phases, which cover u = 0 to 143: 144 terms. The last 6 (u = 144 to 149, the end of input map 5's patch) are
        never added. The ceiling, 10 phases, covers them, and the bounds check zero-fills u = 150 to 159.
      </p>
      </Translated>
    ),
    source: 'Fig. 19.11, line 19, p. 471',
    vi: { source: 'Fig. 19.11, dòng 19, p. 471' },
  },
  'ch19/19.3/q-loads': {
    kind: 'numeric',
    id: 'ch19/19.3/q-loads',
    prompt: <Translated vi={<>Hình 19.11 đã sửa trên LeNet C3 với T = 16: một luồng yêu cầu bao nhiêu lần nạp toàn cục cho mỗi đầu ra (F cộng X)?</>}><>The corrected Fig. 19.11 on LeNet C3 with T = 16: how many global loads does one thread request per output (F plus X)?</></Translated>,
    answer: 20,
    explain: (
      <Translated vi={<p>Mỗi pha một phần tử F và một phần tử X, và ceil(150 / 16) = 10 pha: 20. Kernel trực tiếp của mục 19.2 yêu cầu mỗi loại một lần cho mỗi số hạng, 2·150 = 300. Các giá trị khác mà một luồng cần đến từ bộ nhớ chia sẻ, do các luồng lân cận đặt vào đó.</p>}>
      <p>
        One F element and one X element per phase, and ceil(150 / 16) = 10 phases: 20. The direct kernel of section 19.2 requests one of each per
        term, 2·150 = 300. The other loads a thread needs come from shared memory, put there by its neighbours.
      </p>
      </Translated>
    ),
    source: 'derived from Figs. 19.7 and 19.11, pp. 462 and 471',
    vi: { source: 'suy ra từ Hình 19.7 và 19.11, pp. 462 và 471' },
  },
  'ch19/19.3/q-roles': {
    kind: 'choice',
    id: 'ch19/19.3/q-roles',
    prompt: (
      <Translated vi={<>Hàm <code>launchLinear(X, W, Y, batch, in, out)</code> của dự án xuyên suốt tính Y = X·Wᵀ với W được lưu dạng (out, in). Để có Y = F·B của một mẫu dưới dạng M × P (NCHW) mà không sao chép F, bạn truyền gì?</>}>
      <>
        The project's <code>launchLinear(X, W, Y, batch, in, out)</code> computes Y = X·Wᵀ with W stored (out, in). To get one sample's
        Y = F·B as M × P (NCHW) without copying F, what do you pass?
      </>
      </Translated>
    ),
    options: [
      { label: <Translated vi={<>X = F; W = Bᵀ, lưu dạng P × C·K·K (mỗi pixel một hàng); batch = M, in = C·K·K, out = P.</>}><>X = F; W = Bᵀ, stored P × C·K·K (a row per pixel); batch = M, in = C·K·K, out = P.</></Translated>, correct: true },
      {
        label: <Translated vi={<>X = Bᵀ, lưu dạng P × C·K·K; W = F; batch = P, in = C·K·K, out = M.</>}><>X = Bᵀ, stored P × C·K·K; W = F; batch = P, in = C·K·K, out = M.</></Translated>,
        why: <Translated vi="Cách đó tính Bᵀ·Fᵀ = (F·B)ᵀ: P × M mỗi mẫu, M kênh của mỗi pixel nằm cạnh nhau (kênh nằm cuối), không phải NCHW.">That computes Bᵀ·Fᵀ = (F·B)ᵀ: P × M per sample, each pixel's M channels together (channels last), not NCHW.</Translated>,
      },
      {
        label: <Translated vi={<>X = F; W = B, lưu dạng C·K·K × P; batch = M, in = C·K·K, out = P.</>}><>X = F; W = B, stored C·K·K × P; batch = M, in = C·K·K, out = P.</></Translated>,
        why: <Translated vi="launchLinear đọc W từng hàng theo dạng (out, in): hàng o phải là các đầu vào của đầu ra o. B lưu dạng C·K·K × P đặt chúng dọc theo một cột.">launchLinear reads W row by row as (out, in): row o must be output o's inputs. B stored C·K·K × P has them down a column.</Translated>,
      },
      { label: <Translated vi={<>X = B; W = F; batch = C·K·K, in = P, out = M.</>}><>X = B; W = F; batch = C·K·K, in = P, out = M.</></Translated>, why: <Translated vi="Chiều trong của tích phải là C·K·K, độ dài của mỗi tích vô hướng.">The inner dimension of the product must be C·K·K, the length of each dot product.</Translated> },
    ],
    explain: (
      <Translated vi={<p>Với X = F, hàng m của kết quả là hàng m của F nhân với mọi hàng của W: Y[m, v] = Σᵤ F[m, u]·Bᵀ[v, u], bản đồ đầu ra m theo thứ tự pixel, tức là Y[n, m, :, :]. F được truyền vào nguyên như khi lưu. Việc của kernel unroll là ghi Bᵀ, mỗi pixel đầu ra một hàng C·K·K.</p>}>
      <p>
        With X = F, row m of the result is F's row m against every row of W: Y[m, v] = Σᵤ F[m, u]·Bᵀ[v, u], output map m in pixel order,
        which is Y[n, m, :, :]. F goes in as stored. The unroll kernel's job is to write Bᵀ, one row of C·K·K per output pixel.
      </p>
      </Translated>
    ),
    source: 'labs/ch06/lab4_linear (launchLinear); ch. 19 §19.3, p. 466; ch19-facts A4, D2', // i18n-ignore: citation
  },
  'ch19/19.3/q-wholebatch': {
    kind: 'choice',
    id: 'ch19/19.3/q-wholebatch',
    prompt: <Translated vi={<>Bạn unroll cả N mẫu vào một ma trận C·K·K × N·P và chạy một GEMM duy nhất với F. Bố cục nào được tạo ra, và cái giá là gì?</>}><>You unroll all N samples into one C·K·K × N·P matrix and run a single GEMM with F. What layout comes out, and what does it cost?</></Translated>,
    options: [
      { label: <Translated vi="M × N × P (CNHW): muốn có NCHW cần một phép hoán vị, thêm một lượt đi qua Y.">M × N × P (CNHW): reaching NCHW takes a permute, one more pass over Y.</Translated>, correct: true },
      { label: <Translated vi="N × M × P (NCHW), bố cục lớp kế tiếp đọc, không tốn thêm chi phí.">N × M × P (NCHW), the layout the next layer reads, at no extra cost.</Translated>, why: <Translated vi="Hàng m của tích chứa bản đồ đầu ra m của lần lượt từng mẫu: chỉ số mẫu nằm bên trong chỉ số bản đồ.">Row m of the product holds output map m for every sample in turn: the sample index sits inside the map index.</Translated> },
      { label: <Translated vi="N·P × M (kênh nằm cuối), không tốn thêm chi phí: các bản đồ của mỗi pixel nằm cạnh nhau.">N·P × M (channels last), at no extra cost: each pixel’s maps sit together.</Translated>, why: <Translated vi="Đó là kết quả của Bᵀ·Fᵀ. F·B giữ M làm chỉ số hàng.">That is what Bᵀ·Fᵀ would give. F·B keeps M as the row index.</Translated> },
      { label: <Translated vi="Giống như khi làm từng mẫu, nhưng B không còn vừa bất kỳ GPU nào.">The same as per sample, but B no longer fits any GPU.</Translated>, why: <Translated vi="B lớn lên N lần, điều có thể đáng kể (xem trình khám phá), nhưng câu hỏi là về bố cục.">B grows N-fold, which can matter (see the explorer), but the question is the layout.</Translated> },
    ],
    explain: (
      <Translated vi={<p>Y = F·B đặt các bản đồ đầu ra lên các hàng và các cột của B, nay là pixel của mọi mẫu nối tiếp nhau, lên các cột: M × N × P. Đồ thị của llama.cpp làm vậy rồi trả giá bằng một lần sao chép (<code>ggml_cont</code> của một phép hoán vị); đường dự phòng CUDA của PyTorch thì lặp qua từng mẫu, mỗi mẫu một GEMM (ch19-facts D2, D4, E2). Mỗi mẫu một GEMM là điều sách vẽ (p. 464).</p>}>
      <p>
        Y = F·B puts output maps on rows and B's columns, now every sample's pixels one after another, on columns: M × N × P. llama.cpp's
        graph does this and then pays for a copy (<code>ggml_cont</code> of a permute); PyTorch's CUDA fallback loops over samples instead,
        one GEMM each (ch19-facts D2, D4, E2). One GEMM per sample is what the book draws (p. 464).
      </p>
      </Translated>
    ),
    source: 'ch19-facts E2, D2, D4; ch. 19 §19.3, p. 464', // i18n-ignore: citation
  },
  'ch19/19.3/q-thin': {
    kind: 'choice',
    id: 'ch19/19.3/q-thin',
    prompt: <Translated vi={<>LeNet C5 ở batch 1 cho B kích thước 400 × 1. Vì sao cách diễn đạt GEMM gần như không mang lại gì ở đây?</>}><>LeNet C5 at batch 1 gives a B of 400 × 1. Why does the GEMM formulation buy almost nothing here?</></Translated>,
    options: [
      {
        label: <Translated vi="Mỗi bản đồ chỉ một pixel khiến nó thành một phép nhân ma trận với vector: mỗi phần tử F chỉ được dùng một lần.">One pixel per map makes it a matrix-vector product: each F element is used once.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Tỉ số mở rộng là K² = 25 với bộ lọc 5 × 5, nên B quá lớn để đáng xây dựng.">The expansion ratio is K² = 25 for a 5 × 5 filter, so B is too large to be worth building.</Translated>, why: <Translated vi="Với một pixel đầu ra duy nhất, không có gì chồng lấn: B là 400 giá trị từ 400 đầu vào, tỉ số 1.">With a single output pixel nothing overlaps: B is 400 values from 400 inputs, ratio 1.</Translated> },
      { label: <Translated vi="1920 bộ lọc của C5 không vừa bộ nhớ chia sẻ, nên các ô không chứa được chúng.">C5’s 1920 filters do not fit in shared memory, so the tiles cannot hold them.</Translated>, why: <Translated vi="Kernel phân ô chỉ giữ một ô 16 × 16 của F, bất kể kích thước bộ lọc.">The tiled kernel only ever holds a 16 × 16 tile of F, whatever the filter size.</Translated> },
      { label: <Translated vi="M = 120 không phải bội số của 16, nên hàng ô cuối cùng một nửa là phần đệm.">M = 120 is not a multiple of 16, so the last row of tiles is half padding.</Translated>, why: <Translated vi="Điều đó tốn 8 hàng đệm trên 128, một mất mát nhỏ. Các cột bị thiếu tốn hơn nhiều.">That costs 8 padding rows out of 128, a small loss. The missing columns cost far more.</Translated> },
    ],
    explain: (
      <Translated vi={<p>Sự dùng lại trong một GEMM phân ô đến từ việc mỗi phần tử F gặp nhiều cột của B. Chỉ với một cột thì không có sự dùng lại nào để phân ô, và kernel bị giới hạn bởi việc đọc F, y như trọng số của một LLM ở batch 1 trong lúc decode (mục 5.1). Batching thêm cột: N mẫu cho N cột.</p>}>
      <p>
        Reuse in a tiled GEMM comes from each F element meeting many columns of B. With one column there is no reuse to tile for, and the kernel
        is limited by reading F, exactly like an LLM's weights at batch 1 during decoding (section 5.1). Batching adds columns: N samples give N
        columns.
      </p>
      </Translated>
    ),
    source: 'ch. 19 §19.3, p. 472; derived shapes for LeNet C5',
    vi: { source: 'ch. 19 §19.3, p. 472; hình dạng suy ra cho LeNet C5' },
  },
  'ch19/19.3/q-vit': {
    kind: 'numeric',
    id: 'ch19/19.3/q-vit',
    prompt: (
      <Translated vi={<>Ngoài sách: một patch embedding thị giác tích chập ảnh 3 × 224 × 224 với K = 16 ở stride 16, nên các cửa sổ không chồng lấn. Một biến thể dùng stride 8, nên các cửa sổ kề nhau chồng lấn một nửa. Tỉ số mở rộng của B (số phần tử của B trên mỗi phần tử của X) với biến thể đó là bao nhiêu?</>}>
      <>
        Beyond the book: a vision patch embedding convolves a 3 × 224 × 224 image with K = 16 at stride 16, so its windows
        do not overlap. A variant uses stride 8, so neighbouring windows overlap by half. What is B&rsquo;s expansion ratio
        (elements of B per element of X) for the variant?
      </>
      </Translated>
    ),
    answer: Number(expansion({ C: 3, H: 224, W: 224, K: 16, M: 1, N: 1, stride: 8 }).toFixed(2)), // 3.7194, shown as 3.72
    tol: 0.02,
    explain: (
      <Translated vi={<p>(224 − 16) / 8 + 1 = 27 cửa sổ mỗi chiều, tổng cộng 729, mỗi cửa sổ có 3·256 = 768 giá trị: 559,872 phần tử cho 150,528 đầu vào, khoảng 3.72. Giảm một nửa stride làm số bản sao tăng khoảng bốn lần, (K / stride)² = 4, ít hơn một chút vì các cửa sổ ở biên không có cửa sổ lân cận bên ngoài. Ở stride 16, tỉ số đúng bằng 1: B là một phép xáo trộn lại của X, và một patch embedding là một phép unroll theo sau bởi một lớp tuyến tính (ch19-facts E1, D2).</p>}>
      <p>
        (224 − 16) / 8 + 1 = 27 windows each way, 729 in all, each with 3·256 = 768 values: 559,872 entries for 150,528
        inputs, about 3.72. Halving the stride roughly quadruples the copies, (K / stride)² = 4, a little less because the
        windows at the edges have no neighbour beyond them. At stride 16 the ratio is exactly 1: B is a reshuffle of X,
        and a patch embedding is an unroll followed by a linear layer (ch19-facts E1, D2).
      </p>
      </Translated>
    ),
    source: 'ours (Beyond); ch19-facts E1, D2',
    vi: { source: 'của khóa học này (ngoài sách); ch19-facts E1, D2' },
  },
  'ch19/19.3/q-divmod': {
    kind: 'choice',
    id: 'ch19/19.3/q-divmod',
    prompt: <Translated vi={<>Kernel tích chập implicit-GEMM của llama.cpp biến chiều rộng và chiều cao bộ lọc thành tham số template cho bộ lọc 3 × 3 và 1 × 1. Vì sao?</>}><>llama.cpp's implicit-GEMM convolution makes the filter width and height template parameters for 3 × 3 and 1 × 1 filters. Why?</></Translated>,
    options: [
      {
        label: <Translated vi="Một số chia biết trước lúc biên dịch giúp phép gom khỏi phải làm một phép chia tổng quát, tới 20 lệnh trên NVIDIA, cho mỗi phần tử.">A divisor known at compile time spares the gather a general division, up to 20 instructions on NVIDIA, per element.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Tham số template cho phép trình biên dịch chọn các fragment tensor-core khớp với kích thước bộ lọc.">Template parameters let the compiler pick tensor-core fragments that match the filter size.</Translated>, why: <Translated vi="Các fragment ma trận được chọn ở chỗ khác; tham số template chỉ cố định các số chia.">The matrix fragments are chosen elsewhere; the template parameters only fix the divisors.</Translated> },
      { label: <Translated vi="Nó tiết kiệm các thanh ghi lẽ ra giữ K, nhờ đó tăng occupancy.">It saves the registers that would hold K, which raises occupancy.</Translated>, why: <Translated vi="Một hai thanh ghi không phải là chi phí đáng kể ở đây; phép chia và phép lấy dư cho mỗi phần tử được gom mới là chi phí.">A register or two is not the cost at stake; the divide and remainder per gathered element are.</Translated> },
      { label: <Translated vi="HIP không thể chia cho một biến, nên bản dựng HIP cần cố định các số chia.">HIP cannot divide by a variable, so the HIP build needs the divisors fixed.</Translated>, why: <Translated vi="Nó chia được; AMD chỉ nói phép chia tốn nhiều chu kỳ hơn phép nhân rất nhiều.">It can; AMD only says division takes many more cycles than multiplication.</Translated> },
    ],
    explain: (
      <Translated vi={<p>Eq. 19.5 chia u cho K² và cho K, và chia v cho W_out với mỗi phần tử được gom. NVIDIA ghi trong tài liệu rằng phép chia và phép lấy dư số nguyên biên dịch ra tới 20 lệnh, và nói trình biên dịch tự chuyển phép chia cho một hằng là lũy thừa của hai thành phép dịch bit (A8). Template biến K thành hằng với 3 × 3 và 1 × 1; các kích thước khác dùng một phép chia nhanh được tính trước (D1).</p>}>
      <p>
        Eq. 19.5 divides u by K² and by K and v by W_out for every gathered element. NVIDIA documents integer division and modulo as compiling
        to up to 20 instructions, and says the compiler converts a division by a literal power of two into a shift itself (A8). The template
        makes K a literal for 3 × 3 and 1 × 1; other sizes use a precomputed fast division (D1).
      </p>
      </Translated>
    ),
    source: 'ch19-facts A8, B5, D1', // i18n-ignore: citation
  },
  'ch19/19.3/q-ex4': {
    kind: 'numeric',
    id: 'ch19/19.3/q-ex4',
    prompt: (
      <Translated vi={<>Bài tập 4. LeNet C3 (W_out = 10), T = 16, cột khối 0. 16 luồng có cùng một giá trị ty gom v = 0 đến 15 tại một u. Đó là bao nhiêu dải địa chỉ liên tiếp?</>}>
      <>
        Exercise 4. LeNet C3 (W_out = 10), T = 16, block column 0. The 16 threads with one value of ty gather v = 0 to 15 at one u. How many
        runs of consecutive addresses is that?
      </>
      </Translated>
    ),
    answer: 2,
    explain: (
      <Translated vi={<p>v = 0 đến 9 nằm trên hàng đầu ra 0 và đọc 10 pixel đầu vào liên tiếp; v = 10 đến 15 nằm trên hàng 1, và địa chỉ nhảy K = 5 (hàng đầu vào dài hơn hàng đầu ra K − 1), rồi lại chạy liên tiếp. Các truy cập Fds và Bds không có xung đột bank (bank conflict) trên mô hình 32 bank của NVIDIA, nơi các luồng đọc cùng một word không xung đột (ch06-facts A12).</p>}>
      <p>
        v = 0 to 9 lie on output row 0 and read 10 consecutive input pixels; v = 10 to 15 lie on row 1, and the address jumps by K = 5 (the
        input row is K − 1 longer than the output row), then runs again. The Fds and Bds accesses have no bank conflicts on NVIDIA's 32-bank
        model, where threads reading the same word do not conflict (ch06-facts A12).
      </p>
      </Translated>
    ),
    source: 'Exercise 4, p. 475; ours (warp model); ch06-facts A12, infra-facts A10',
    vi: { source: 'Exercise 4, p. 475; của khóa học này (mô hình warp); ch06-facts A12, infra-facts A10' },
  },
};

// ---------------------------------------------------------------- the section

export function S193() {
  return (
    <>
      <Translated vi={<p>Kernel trực tiếp của <a href={href('ch19', '19.2')}>mục 19.2</a> giao cho mỗi khối một ô của một bản đồ đầu ra. Mọi bản đồ đầu ra đều đọc mọi bản đồ đầu vào, nên các pixel đầu vào mà một khối nạp chính là những pixel mà các khối cho các bản đồ đầu ra khác cần. Nhưng bộ nhớ chia sẻ thuộc về một khối, và trong thiết kế đó các khối có thể dùng lại một ô không bao giờ thấy nó. Muốn có sự dùng lại đó cần một cách khác để chia công việc (p. 463).</p>}>
      <p>
        The direct kernel of <a href={href('ch19', '19.2')}>section 19.2</a> gives each block a tile of one output map. Every output map
        reads every input map, so the input pixels one block loads are the same ones the blocks for the other output maps need. But shared
        memory belongs to one block, and in that design the blocks that could reuse a tile never see it. Getting that reuse takes a different
        way of cutting up the work (p. 463).
      </p>
      </Translated>
      <Source section="19.3" pages="463" figs={tr("Fig. 19.7", 'Hình 19.7')} />

      <h2><Translated vi="Một bảng các tích vô hướng">A table of dot products</Translated></h2>
      <Translated vi={<p>Mỗi pixel đầu ra là một tích vô hướng: C·K·K giá trị đầu vào với C·K·K trọng số. Một tích ma trận là một bảng các tích vô hướng, các hàng của toán hạng thứ nhất với các cột của toán hạng thứ hai. Vậy hãy đặt trọng số của mỗi bản đồ đầu ra vào một hàng, các vùng đầu vào của mỗi pixel đầu ra vào một cột, và cả lớp là một phép nhân ma trận, Y = F · B.</p>}>
      <p>
        Each output pixel is a dot product: C·K·K input values against C·K·K weights. A matrix product is a table of dot products, rows of
        the first operand against columns of the second. So put each output map's weights in a row, each output pixel's input patches in a
        column, and the whole layer is one matrix multiplication, Y = F · B.
      </p>
      </Translated>
      <Source section="19.3" pages="463–464" />
      <Translated vi={<p>Khi đó phân ô (tiling) của chương 5 và 15 dùng lại cả đầu vào lẫn bộ lọc, hoặc một thư viện GEMM (sách nêu tên cuBLAS) làm việc này. Sách ghi công ý tưởng này cho Chellapilla và các cộng sự, tài liệu tham khảo số 8 của sách (pp. 463–464).</p>}>
      <p>
        The tiling of chapters 5 and 15 then reuses both the inputs and the filters, or a GEMM library (the book names cuBLAS) does the
        work. The book credits the idea to Chellapilla and colleagues, its reference 8 (pp. 463–464).
      </p>
      </Translated>
      <Today checked="2026-11-20 (ch19-facts A10)">{/* i18n-ignore: date and fact citation */}
        <Translated vi={<p>Chú thích 3,{' '}<a href="https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/">bài viết năm 2015 của Pete Warden</a>, vẫn còn trên mạng.</p>}>
        <p>
          Footnote 3,{' '}
          <a href="https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/">Pete Warden's 2015 post</a>, is still
          online.
        </p>
        </Translated>
      </Today>

      <h2><Translated vi="F vốn đã là một ma trận">F is already a matrix</Translated></h2>
      <Translated vi={<p>Hàng m của F chứa mọi bộ lọc của bản đồ đầu ra m, mỗi bộ lọc đọc từng hàng, bản đồ đầu vào 0 trước. Tức là M hàng, mỗi hàng C·K·K trọng số, và đó đúng là cách mảng bộ lọc M × C × K × K vốn đã nằm trong bộ nhớ: cùng một con trỏ, đọc như một ma trận có các hàng dài C·K·K. Trong ví dụ của sách, hai hàng là <code>1 1 2 2 | 1 1 1 1 | 0 1 1 0</code> và <code>1 0 0 1 | 2 1 2 1 | 1 2 2 0</code>{' '}(pp. 464–465).</p>}>
      <p>
        Row m of F holds every filter of output map m, each read row by row, input map 0 first. That is M rows of C·K·K weights, and it is
        exactly how the M × C × K × K filter array already sits in memory: the same pointer, read as a matrix whose rows are C·K·K long.
        In the book's example the two rows are <code>1 1 2 2 | 1 1 1 1 | 0 1 1 0</code> and <code>1 0 0 1 | 2 1 2 1 | 1 2 2 0</code>{' '}
        (pp. 464–465).
      </p>
      </Translated>
      <Source section="19.3" pages="464–465" figs={tr("Fig. 19.8", 'Hình 19.8')} />
      <Translated vi={<p>Không có gì bị sao chép. Theo cách nói của PyTorch, <code>conv.weight.view(M, -1)</code> chính là F này.</p>}>
      <p>
        Nothing is copied. In PyTorch terms, <code>conv.weight.view(M, -1)</code> is this F.
      </p>
      </Translated>
      <Today checked="2026-11-20 (ch19-facts D4)">{/* i18n-ignore: date and fact citation */}
        <Translated vi={<p>PyTorch ghi trong tài liệu trọng số có dạng (out_channels, in_channels / groups, kH, kW): với một group, đó là bố cục của sách.</p>}>
        <p>
          PyTorch documents the weight as (out_channels, in_channels / groups, kH, kW): with one group, the book's layout.
        </p>
        </Translated>
      </Today>

      <h2><Translated vi="B: mỗi pixel đầu ra một cột">B: one column per output pixel</Translated></h2>
      <Translated vi={<p>Cột v của B thuộc về pixel đầu ra v, đếm theo từng hàng. Nó chứa vùng K × K của pixel đó từ bản đồ đầu vào 0, rồi từ bản đồ 1, và cứ thế, mỗi vùng đọc từng hàng: cùng thứ tự với một hàng của F, nên một hàng và một cột khớp nhau từng số hạng. B có kích thước C·K·K × H_out·W_out, 12 × 4 trong ví dụ (pp. 465–466).</p>}>
      <p>
        Column v of B belongs to output pixel v, counted row by row. It holds that pixel's K × K patch from input map 0, then from map 1, and
        so on, each patch read row by row: the same order as a row of F, so a row and a column line up term by term. B is C·K·K × H_out·W_out,
        12 × 4 in the example (pp. 465–466).
      </p>
      </Translated>
      <Source section="19.3" pages="465–466" figs={tr("Fig. 19.8", 'Hình 19.8')} />
      <Translated vi={<p>Vì các cột đi theo các pixel đầu ra theo thứ tự hàng, hàng m của tích là bản đồ đầu ra m, xếp từng hàng một. Y = F · B đã là Y[n, m, :, :], ở bố cục mà lớp kế tiếp đọc, và không cần xáo trộn gì sau đó (p. 466). Sách vẽ từng mẫu một: mỗi mẫu có B riêng và tất cả dùng chung F (p. 464).</p>}>
      <p>
        Because the columns follow the output pixels in row-major order, row m of the product is output map m, laid out row by row. Y = F · B
        is already Y[n, m, :, :], in the layout the next layer reads, and nothing is reshuffled afterwards (p. 466). The book draws one sample
        at a time: each sample has its own B and all of them share F (p. 464).
      </p>
      </Translated>
      <Source section="19.3" pages="464, 466" />
      <Predict
        id="ch19/19.3/p-ratio-c1"
        prompt={
          <Translated vi={<>LeNet C1 (đầu vào 1 × 32 × 32, K = 5): B có số phần tử gấp bao nhiêu lần X? Hãy chốt một con số ngay bây giờ; cả hoạt cảnh lẫn trình khám phá bên dưới đều quay lại câu hỏi này.</>}>
          <>
            LeNet C1 (1 × 32 × 32 input, K = 5): how many times as many elements as X does B have? Commit a number now; the animation
            and the explorer below both come back to it.
          </>
          </Translated>
        }
        unit="×"
      >
        {(v) => (
          <Translated vi={<p>Đó là 25·784 / 1024 = <b>19.14</b>: 25 giá trị mỗi pixel đầu ra, 28 × 28 pixel đầu ra, 32 × 32 đầu vào.{' '}{Math.abs(v - 19.14) < 1 ? 'Gần đúng.' : v > 25 ? 'Nó không thể vượt quá K² = 25.' : 'K² = 25 là mức trần; một bản đồ 32 × 32 đã là lớn so với một bộ lọc 5 × 5.'}</p>}>
          <p>
            It is 25·784 / 1024 = <b>19.14</b>: 25 values per output pixel, 28 × 28 output pixels, 32 × 32 inputs.{' '}
            {Math.abs(v - 19.14) < 1 ? 'Close.' : v > 25 ? 'It cannot pass K² = 25.' : 'K² = 25 is the ceiling; a 32 × 32 map is already large next to a 5 × 5 filter.'}
          </p>
          </Translated>
        )}
      </Predict>
      <Translated vi={<p>Hoạt cảnh xây dựng F và B cho ví dụ của sách, nhân chúng, rồi phóng to ví dụ lên.</p>}>
      <p>
        The animation builds F and B for the book's example, multiplies them, and then scales the example up.
      </p>
      </Translated>
      <Simplification>
        <Translated vi={<p>Một mẫu, stride 1, không đệm (padding). B được điền từng cột một để bạn theo dõi; một phép unroll thật ghi mọi cột cùng lúc. Đếm theo giá trị, không theo byte hay thời gian.</p>}>
        <p>
          One sample, stride 1, no padding. B fills one column at a time so you can follow it; a real unroll writes every column at once.
          Values are counted, not bytes or time.
        </p>
        </Translated>
      </Simplification>
      <UnfoldScene />

      <h2><Translated vi="Cái giá của việc mở rộng (unfold)">What unfolding costs</Translated></h2>
      <Translated vi={<p>Trong ví dụ, B có 48 phần tử được tạo từ 27 giá trị đầu vào, gấp 16/9 lần. Sự trùng lặp đến từ chồng lấn: các pixel đầu ra kề nhau đọc các vùng chồng lấn nhau, nên một pixel ở góc của bản đồ đầu vào rơi vào một cột, một pixel ở cạnh vào hai cột, và pixel ở giữa vào bốn cột (pp. 466–467).</p>}>
      <p>
        In the example B has 48 entries made from 27 input values, 16/9 as many. The duplication comes from overlap: neighbouring output
        pixels read overlapping patches, so a corner pixel of an input map lands in one column, an edge pixel in two, and the centre pixel in
        four (pp. 466–467).
      </p>
      </Translated>
      <Source section="19.3" pages="466–467" />
      <Translated vi={<p>Tổng quát, tỉ số là Eq. 19.4, C·K·K·H_out·W_out chia cho C·H·W, tức K²·H_out·W_out / (H·W). C bị triệt tiêu, và khi các bản đồ lớn lên thì H_out·W_out tiến tới H·W, nên tỉ số tiến tới K². Với K = 3, nó là 7.91 ở 32 × 32 và 8.84 ở 224 × 224 (tính toán của khóa học này); con số ước lượng của sách là 20 hoặc hơn khi bộ lọc là 5 × 5 và bản đồ rộng hàng trăm pixel (p. 467).</p>}>
      <p>
        In general the ratio is Eq. 19.4, C·K·K·H_out·W_out over C·H·W, which is K²·H_out·W_out / (H·W). C cancels, and as the maps grow
        H_out·W_out approaches H·W, so the ratio approaches K². For K = 3 it is 7.91 at 32 × 32 and 8.84 at 224 × 224 (our arithmetic); the
        book's rough figure is 20 or more once filters are 5 × 5 and maps are hundreds of pixels wide (p. 467).
      </p>
      </Translated>
      <Source section="19.3" pages="467" figs={tr("Eq. 19.4", 'Eq. 19.4')} />
      <Translated vi={<p>Kích thước không phải là chi phí duy nhất. Việc ghi B ra là một lượt riêng qua bộ nhớ toàn cục, và GEMM sau đó lại đọc nó vào: chính lưu lượng mà cách diễn đạt này lẽ ra phải tiết kiệm (p. 467).</p>}>
      <p>
        Size is not the only cost. Writing B out is a pass over global memory of its own, and the GEMM then reads it back: the traffic the
        formulation was supposed to save (p. 467).
      </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch19-facts.md, E2, C2, D3, A1)">{/* i18n-ignore: date and fact citation */}
        <Translated vi={<p>B của một mẫu cho một lớp 64 kênh 224 × 224 với K = 3 là 108 MiB ở FP32, và một batch 8 mẫu là 866 MiB (tính toán của khóa học này). Metal chỉ bảo đảm một buffer có thể chứa 256 MB (mức tối thiểu được ghi trong tài liệu của <code>maxBufferLength</code>), và đường mở rộng tường minh của MLX mở rộng theo từng dải hàng vì lý do đó. cuDNN giới hạn một tensor ở "2 Giga-elements" (2 tỉ phần tử).</p>}>
        <p>
          One sample's B for a 64-channel 224 × 224 layer with K = 3 is 108 MiB in FP32, and a batch of 8 is 866 MiB (our arithmetic). Metal
          guarantees only that one buffer can hold 256 MB (the documented minimum of <code>maxBufferLength</code>), and MLX's explicit path
          unfolds in row tiles for that reason. cuDNN caps a tensor at "2 Giga-elements".
        </p>
        </Translated>
      </Today>
      <Translated vi={<p>Trình khám phá bên dưới vẽ ví dụ của sách và cho bạn di chuột lên bất kỳ đầu ra nào để xem hàng F, cột B và các vùng của nó, hoặc nhấp vào một pixel đầu vào để xem mọi phần tử của B sao chép nó. Đổi lớp thì hình vẽ chuyển sang các hình dạng theo đúng tỉ lệ, cùng với các con số, tỉ số và các giới hạn nêu trên.</p>}>
      <p>
        The explorer below draws the book's example and lets you hover over any output to see its row of F, its column of B and its patches,
        or click an input pixel to see every entry of B that copies it. Change the layer and the drawing switches to the shapes to scale, with
        the counts, the ratio and the limits above.
      </p>
      </Translated>
      <Simplification>
        <Translated vi={<p>Chỉ FP32; stride 1 và không đệm trừ khi bật công tắc ngoài sách. Số byte chỉ tính B, không tính đầu ra của GEMM.</p>}>
        <p>FP32 only; stride 1 and no padding unless the Beyond toggle is on. Byte counts are B alone, not the GEMM's output.</p>
        </Translated>
      </Simplification>
      <UnfoldExplorer />
      <Ask q={questionsS193['ch19/19.3/q-noM']} />
      <Ask q={questionsS193['ch19/19.3/q-belems']} label={tr("Work it out", 'Tự tính')} />

      <h2><Translated vi="Mở rộng mà không lưu B">Unfolding without storing B</Translated></h2>
      <Translated vi={<p>Các hệ thống thực tế không xây B trong bộ nhớ toàn cục. Một phép nhân ma trận phân ô vốn đã có một bước, trong mỗi pha, ở đó mỗi luồng nạp một phần tử của mỗi toán hạng vào bộ nhớ chia sẻ. Chỉ cần đổi bước nạp B: thay vì đọc một B đã lưu, mỗi luồng tính xem phần tử B của nó sao chép phần tử nào của X, rồi đọc phần tử đó. X vẫn nằm nguyên trong bộ nhớ toàn cục; B tồn tại từng ô một, trong bộ nhớ chia sẻ. Sách nói B được hiện thực hóa khi cần (pp. 467–468).</p>}>
      <p>
        Practical systems do not build B in global memory. A tiled matrix multiplication already has a step, in every phase, where each thread
        loads one element of each operand into shared memory. Change only the B load: instead of reading a stored B, each thread works out
        which element of X its element of B copies, and reads that. X stays in global memory as it is; B exists one tile at a time, in shared
        memory. The book says B is materialised on demand (pp. 467–468).
      </p>
      </Translated>
      <Source section="19.3" pages="467–468" />
      <Translated vi={<p>Hình 19.9 cho thấy điều này với các ô 2 × 2. Y là 2 × 4, nên hai khối bốn luồng phủ nó. Ở pha 0, khối bên trái nạp B[0, 0] từ X[n, 0, 0, 0], B[0, 1] và B[1, 0] đều từ X[n, 0, 0, 1], và B[1, 1] từ X[n, 0, 0, 2]; ở pha 1 nó nạp hàng giữa của bản đồ đầu vào 0. Sự trùng lặp vẫn xảy ra, nhưng chỉ trong bộ nhớ chia sẻ (p. 468).</p>}>
      <p>
        Fig. 19.9 shows it with 2 × 2 tiles. Y is 2 × 4, so two blocks of four threads cover it. In phase 0 the left block loads B[0, 0] from
        X[n, 0, 0, 0], B[0, 1] and B[1, 0] both from X[n, 0, 0, 1], and B[1, 1] from X[n, 0, 0, 2]; in phase 1 it loads the middle row of input
        map 0. The duplication still happens, but only in shared memory (p. 468).
      </p>
      </Translated>
      <Source section="19.3" pages="468" figs={tr("Fig. 19.9", 'Hình 19.9')} />
      <Translated vi={<p>Một chỗ cần sửa trên trang sách. Phần chữ liệt kê các đầu ra của khối đó là bốn pixel của bản đồ đầu ra 0. Khung đỏ của hình, và các ô B mà khối dùng, lại cho thấy khác: nó tính 14 và 20 từ bản đồ 0, và 12 và 24 từ bản đồ 1, hai pixel của mỗi bản đồ. Nửa trái của ma trận Y là hai cột của nó, và các cột là các pixel (p. 468).</p>}>
      <p>
        One correction to the page. The text lists that block's outputs as four pixels of output map 0. The figure's red box, and the B tiles
        the block uses, say otherwise: it computes 14 and 20 from map 0 and 12 and 24 from map 1, two pixels of each map. The left half of the
        Y matrix is two of its columns, and the columns are pixels (p. 468).
      </p>
      </Translated>
      <Source pages="468" />
      <Translated vi={<p>Hoạt cảnh chạy khối bên trái đó qua cả sáu pha của ví dụ của sách, với các ô 2 × 2.</p>}>
      <p>
        The animation runs that left block through all six phases of the book's example, with 2 × 2 tiles.
      </p>
      </Translated>
      <Simplification>
        <Translated vi={<p>Một khối của một mẫu; khối kia làm cùng công việc trên các cột của nó. Số lần đọc được đếm theo giá trị, không theo giao dịch bộ nhớ. Không có cache, không có thời gian.</p>}>
        <p>
          One block of one sample; the other block does the same work on its own columns. Reads are counted as values, not memory
          transactions. No caches, no timing.
        </p>
        </Translated>
      </Simplification>
      <ImplicitTilesScene />

      <h2><Translated vi={<>Eq. 19.5: mỗi phần tử của B đến từ đâu</>}>Eq. 19.5: where each element of B comes from</Translated></h2>
      <Translated vi={<p>Phép gom cần một công thức từ (u, v) tới một phần tử của X. Bắt đầu với cột: v đánh số các pixel đầu ra theo từng hàng, nên pixel nằm ở hàng v / W_out, cột v % W_out, và đó cũng là góc trên bên trái của vùng của nó trong mọi bản đồ đầu vào. Rồi đến hàng: u chạy qua C vùng, mỗi vùng K² giá trị, nên kênh là u / K², và vị trí bên trong vùng là u % K², tức hàng (u % K²) / K và cột (u % K²) % K của vùng. Cộng góc của vùng với độ lệch (pp. 469–470):</p>}>
      <p>
        The gather needs a formula from (u, v) to an element of X. Start with the column: v numbers output pixels row by row, so the pixel is at
        row v / W_out, column v % W_out, and that is also the top-left corner of its patch in every input map. Then the row: u runs through C
        patches of K² values, so the channel is u / K², and the place inside the patch is u % K², which is row (u % K²) / K and column
        (u % K²) % K of the patch. Add patch corner and offset (pp. 469–470):
      </p>
      </Translated>
      <Source section="19.3" pages="469–470" figs={tr("Eq. 19.5", 'Eq. 19.5')} />
      <p data-eq="19.5" style={{ overflowX: 'auto' }}>
        <code style={{ whiteSpace: 'pre', display: 'inline-block', lineHeight: 1.6 }}>
          {'B[u, v] ← X[n, u / K²,\n              (u % K²) / K + v / W_out,\n              (u % K²) % K + v % W_out]'}
        </code>
      </p>
      <Translated vi={<p>Làm trên Hình 19.10: B[10, 2] với K = 2 và W_out = 2. Kênh là 10 / 4 = 2; độ lệch 10 % 4 = 2 là hàng 1, cột 0 của vùng; góc là (2 / 2, 2 % 2) = (1, 0). Vậy phần tử là X[n, 2, 2, 0], có giá trị 3. Trang sách nói "u is equal to 2" (u bằng 2) cho phần tử này: u là 10, còn 2 là kênh mà bước đầu tiên cho ra (p. 469).</p>}>
      <p>
        Worked on Fig. 19.10: B[10, 2] with K = 2 and W_out = 2. The channel is 10 / 4 = 2; the offset 10 % 4 = 2 is row 1, column 0 of the
        patch; the corner is (2 / 2, 2 % 2) = (1, 0). So the element is X[n, 2, 2, 0], whose value is 3. The page says "u is equal to 2" for
        this element: u is 10, and the 2 is the channel that the first step gives (p. 469).
      </p>
      </Translated>
      <Source pages="469–470" figs={tr("Fig. 19.10", 'Hình 19.10')} />
      <Translated vi={<p>Hai cặp chia và lấy dư, mỗi cặp cho một chỉ số. Hãy làm với bất kỳ phần tử nào trong trình luyện, hoặc đảo ngược lại và đếm số phần tử sao chép một pixel.</p>}>
      <p>
        Two divide-and-remainder pairs, one for each index. Work through any element in the trainer, or turn it round and count the entries that
        copy a pixel.
      </p>
      </Translated>
      <Simplification>
        <Translated vi={<p>Stride 1, không đệm. Chỉ số mẫu n được bỏ đi: nó cộng cùng một độ lệch, n·C·H·W, vào mọi phần tử.</p>}>
        <p>Stride 1, no padding. The sample index n is left out: it adds the same offset, n·C·H·W, to every element.</p>
        </Translated>
      </Simplification>
      <IndexTrainer />
      <Ask q={questionsS193['ch19/19.3/q-map']} label={tr("Work it out", 'Tự tính')} />
      <Ask q={questionsS193['ch19/19.3/q-copies']} label={tr("Work it out", 'Tự tính')} />

      <h2><Translated vi="Kernel">The kernel</Translated></h2>
      <Translated vi={<p>Hình 19.11 là kernel phân ô của chương 5 với ba thay đổi: các kích thước của lớp thay cho <code>Width</code>, chỉ số z của khối là mẫu, và bước nạp B trở thành phép gom theo Eq. 19.5 (pp. 470–471). Đây là nó như lab 19.1 cung cấp, đã sửa và đã kiểm tra. Hãy đọc theo màu: hổ phách là bộ nhớ toàn cục, xanh mòng két là bộ nhớ chia sẻ, tím là thanh ghi.</p>}>
      <p>
        Fig. 19.11 is chapter 5's tiled kernel with three changes: the layer's sizes replace <code>Width</code>, the block's z index is the
        sample, and the B load becomes the Eq. 19.5 gather (pp. 470–471). Here it is as lab 19.1 provides it, corrected and tested. Read it by
        colour: amber is global memory, teal is shared memory, purple is registers.
      </p>
      </Translated>
      <Source section="19.3" pages="470–471" figs={tr("Fig. 19.11", 'Hình 19.11')} />
      <ProvidedKernel />
      <p><Translated vi="Những chỗ nó khác trang sách, và vì sao:">Where it differs from the page, and why:</Translated></p>
      <Translated vi={<ul><li><b>Số pha.</b> Dòng 19 được in là <code>for (int ph = 0; ph &lt; (C*K*K)/TILE_WIDTH; ++ph)</code>. Phép chia nguyên cắt bớt phần lẻ, nên khi C·K·K không phải bội số của ô thì pha cuối, pha không đầy đủ, không bao giờ chạy và các số hạng của nó bị bỏ sót. Kernel của lab chạy ceil(C·K·K / TILE_WIDTH) pha.</li><li><b>Kiểm tra biên.</b> Sách cố ý bỏ chúng (p. 470). Lab điền 0 cho bước nạp F và phép gom X bên ngoài ma trận và bảo vệ bước ghi, nhờ đó mọi M, H_out·W_out và C·K·K đều chạy đúng.</li><li><b>Tên.</b> Trang sách viết W_out thành <code>W-K+1</code> bên trong phép gom; lab đặt tên <code>H_out</code> và <code>W_out</code>.</li><li><b>Lần khởi chạy.</b> Trang sách chỉ nói trục z của lưới là batch (p. 470). Theo kernel của chương 5, lab khởi chạy ceil(P / T) ô pixel theo x, ceil(M / T) ô bản đồ theo y và N mẫu theo z. Lần khởi chạy đó là suy luận của khóa học này.</li></ul>}>
      <ul>
        <li>
          <b>The phase count.</b> Printed line 19 is <code>for (int ph = 0; ph &lt; (C*K*K)/TILE_WIDTH; ++ph)</code>. Integer division
          truncates, so when C·K·K is not a multiple of the tile the last, partial phase never runs and its terms are dropped. The lab's kernel
          runs ceil(C·K·K / TILE_WIDTH) phases.
        </li>
        <li>
          <b>Bounds checks.</b> The book leaves them out on purpose (p. 470). The lab zero-fills the F load and the X gather outside the matrix
          and guards the store, which makes any M, H_out·W_out and C·K·K work.
        </li>
        <li>
          <b>Names.</b> The page writes W_out as <code>W-K+1</code> inside the gather; the lab names <code>H_out</code> and <code>W_out</code>.
        </li>
        <li>
          <b>The launch.</b> The page states only that grid z is the batch (p. 470). Following chapter 5's kernel, the lab launches
          ceil(P / T) pixel tiles in x, ceil(M / T) map tiles in y and N samples in z. That launch is our derivation.
        </li>
      </ul>
      </Translated>
      <p><Translated vi="Dòng 28 được in là nơi Eq. 19.5 nằm, dưới dạng một chỉ số phẳng (ở đây được ngắt tại các dấu +):">Printed line 28 is where Eq. 19.5 lives, as one flat index (broken here at its + signs):</Translated></p>
      <Source pages="471" figs={tr("Fig. 19.11, lines 19 and 28", 'Hình 19.11, dòng 19 và 28')} />
      <p data-line="28" style={{ overflowX: 'auto' }}>
        <code style={{ whiteSpace: 'pre', display: 'inline-block', lineHeight: 1.6 }}>
          {'Bds[ty][tx]=X[bz*(C*H*W)\n  +(u/(K*K))*(H*W)\n  +((u%(K*K))/K + v/(W-K+1))*W\n  +((u%(K*K))%K + (v%(W-K+1)))];'}
        </code>
      </p>
      <Translated vi={<p>Số hạng hàng được nhân với chiều rộng đầu vào W, trong khi v được chia cho chiều rộng đầu ra W − K + 1. Hai chiều rộng này dễ nhầm, và dùng W cho cả hai là lỗi cài sẵn trong phần săn lỗi bên dưới.</p>}>
      <p>
        The row term is multiplied by the input width W, while v is divided by the output width W − K + 1. The two widths are easy to mix up,
        and using W for both is the planted bug in the hunt below.
      </p>
      </Translated>
      <Translated vi={<p>Tệp CUDA được thiết lập để dựng cho HIP mà không cần sửa qua <code>pmpp/vendors/hip.h</code> (có trong tài liệu, ch19-facts B1; nó chưa được biên dịch hay chạy trên phần cứng AMD ở đây), vì nó chỉ dùng mảng chia sẻ, rào chắn, chỉ số khối và luồng, và cú pháp khởi chạy ba dấu ngoặc nhọn; một kernel có intrinsic warp <code>_sync</code>, cooperative groups, header bán chính xác (half-precision), functor của CUB, intrinsic của lõi ma trận hoặc <code>__dp4a</code> (HIP không có) thì không như vậy. Trên Metal, Fds và Bds trở thành các mảng <code>threadgroup</code>; tệp <code>implicit_gemm.metal</code> của lab 19.1 cho thấy phần còn lại.</p>}>
      <p>
        The CUDA file is set up to build for HIP unchanged through <code>pmpp/vendors/hip.h</code> (documented, ch19-facts B1; it has
        not been compiled or run on AMD hardware here), because it uses only shared arrays, barriers, block and thread indices and the
        triple-chevron launch; a kernel with warp <code>_sync</code> intrinsics, cooperative groups, half-precision headers, CUB
        functors, matrix-core intrinsics or <code>__dp4a</code> (HIP has none) would not. On Metal, Fds and Bds become <code>threadgroup</code> arrays; lab
        19.1's <code>implicit_gemm.metal</code> shows the rest.
      </p>
      </Translated>
      <Today checked="2026-11-20 (ch19-facts B1, C1)">{/* i18n-ignore: date and fact citation */}
        <Translated vi={<p>Metal viết cả hai rào chắn là <code>threadgroup_barrier(mem_flags::mem_threadgroup)</code>.</p>}>
        <p>
          Metal spells both barriers <code>threadgroup_barrier(mem_flags::mem_threadgroup)</code>.
        </p>
        </Translated>
      </Today>
      <Translated vi={<p>Phần săn lỗi chạy các giá trị của kernel từng khối một trên năm hình dạng. Hãy bắt đầu với LeNet C1 như được in, rồi tìm thay đổi nhỏ nhất làm cho từng hình dạng đúng.</p>}>
      <p>
        The hunt runs the kernel's values block by block on five shapes. Start with LeNet C1 as printed, then find the smallest change that
        makes each shape right.
      </p>
      </Translated>
      <Simplification>
        <Translated vi={<p>Chỉ có giá trị, không có thời gian; một mẫu. Một lần đọc bên ngoài X hoặc F được đếm, không được thực thi: giá trị của nó không xác định. Một lần ghi từ một luồng nằm ngoài phạm vi của Y mà rơi vào ô của một đầu ra khác sẽ đánh dấu ô đó là sai, vì trên GPU, lần ghi nào đến sau cùng là không xác định. Lần khởi chạy, thứ sách không trình bày, ở đây luôn được làm tròn lên; nếu cũng cắt bớt, LeNet C1 sẽ không khởi chạy khối nào cả.</p>}>
        <p>
          Values only, no timing; one sample. A read outside X or F is counted, not executed: its value is undefined. A store from a thread
          outside Y's range that lands on another output's slot marks that slot wrong, because on a GPU which store lands last is not defined.
          The launch, which the book does not show, is always rounded up here; with truncation as well, LeNet C1 would launch no blocks at all.
        </p>
        </Translated>
      </Simplification>
      <Predict
        id="ch19/19.3/p-terms-c1"
        prompt={<Translated vi={<>LeNet C1 có C·K·K = 25. Với TILE_WIDTH = 16 và vòng lặp pha như được in, bao nhiêu trong 25 số hạng của mỗi đầu ra được cộng?</>}><>LeNet C1 has C·K·K = 25. With TILE_WIDTH = 16 and the phase loop as printed, how many of the 25 terms of each output get added?</></Translated>}
        unit={tr("terms", 'số hạng')}
      >
        {(v) => (
          <Translated vi={<p><b>16.</b> 25 / 16 bị cắt còn 1 pha gồm 16 số hạng; 9 số hạng còn lại bị bỏ sót, nên mọi đầu ra đều sai. {v === 16 ? 'Đúng như bạn nói.' : ''}</p>}>
          <p>
            <b>16.</b> 25 / 16 truncates to 1 phase of 16 terms; the other 9 are dropped, so every output is wrong. {v === 16 ? 'As you said.' : ''}
          </p>
          </Translated>
        )}
      </Predict>
      <ErratumHunt />
      <Ask q={questionsS193['ch19/19.3/q-phases']} label={tr("Work it out", 'Tự tính')} />

      <h2><Translated vi="Các ô mang lại gì">What the tiles buy</Translated></h2>
      <Translated vi={<p>Một khối phủ T hàng của Y (T bản đồ đầu ra) và T cột (T pixel đầu ra). Mỗi ô B nó gom từ X được dùng bởi cả T bản đồ đầu ra trong khối, và mỗi ô F bởi cả T pixel: sự dùng lại giữa các bản đồ đầu ra mà kernel trực tiếp không có được. Với mỗi đầu ra, kernel trực tiếp yêu cầu 2·C·K² lần nạp toàn cục, mỗi số hạng một X và một F. Hãy đếm số lần nạp của kernel phân ô từ đoạn mã ở trên: một luồng nạp gì từ bộ nhớ toàn cục trong một pha, và có bao nhiêu pha?</p>}>
      <p>
        A block covers T rows of Y (T output maps) and T columns (T output pixels). Each B tile it gathers from X is used by all T output maps
        in the block, and each F tile by all T pixels: the reuse across output maps that the direct kernel could not get. Per output, the
        direct kernel requests 2·C·K² global loads, one X and one F per term. Count the tiled kernel's from the listing above: what does one
        thread load from global memory in a phase, and how many phases are there?
      </p>
      </Translated>
      <Source section="19.3" pages="462, 471" figs={tr("Figs. 19.7, 19.11", 'Hình 19.7, 19.11')} />
      <Ask q={questionsS193['ch19/19.3/q-loads']} label={tr("Work it out", 'Tự tính')} />

      <h2><Translated vi="Cái giá: phép chia nguyên trong mỗi lần gom">The price: integer division in every gather</Translated></h2>
      <Translated vi={<p>Mỗi phần tử được gom tốn các phép chia và lấy dư cho K², cho K và cho W_out, trong khi một lần nạp GEMM thông thường tốn một phép nhân và một phép cộng. Sách, khi mô tả phiên bản của cùng ý tưởng này trong cuDNN, gọi nó là phép tính chỉ số bổ sung (p. 474). Nó tốn bao nhiêu tùy vào phần cứng, và các nhà sản xuất nói những điều khác nhau.</p>}>
      <p>
        Each gathered element costs divides and remainders by K², by K and by W_out, where a plain GEMM load costs a multiply and an add. The
        book, describing cuDNN's version of the same idea, calls it additional indexing arithmetic (p. 474). How much it costs depends on the
        hardware, and the vendors say different things.
      </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch19-facts.md, A8, B5, C6, A7, D1)">{/* i18n-ignore: date and fact citation */}
        <Translated vi={<p>NVIDIA: phép chia và lấy dư số nguyên "compile to up to 20 instructions" (biên dịch ra tới 20 lệnh), và trình biên dịch biến chúng thành phép dịch bit khi số chia là một hằng lũy thừa của hai. AMD: phép chia "requires many more hardware cycles than multiplication" (cần nhiều chu kỳ phần cứng hơn phép nhân rất nhiều). Phát biểu duy nhất của Apple là một bài nói năm 2016: phần lớn GPU không có lệnh chia số nguyên, nên hãy chia cho hằng hoặc function constant. CUTLASS tính trước các độ lệch con trỏ và dùng một divmod nhanh; llama.cpp biến kích thước bộ lọc thành tham số template cho 3 × 3 và 1 × 1 và dùng một phép chia nhanh được tính trước cho các trường hợp khác.</p>}>
        <p>
          NVIDIA: integer division and modulo "compile to up to 20 instructions", and the compiler turns them into shifts when the divisor is a
          literal power of two. AMD: division "requires many more hardware cycles than multiplication". Apple's only statement is a 2016 talk:
          most GPUs have no integer divide instruction, so divide by literals or function constants. CUTLASS precomputes pointer deltas and
          uses a fast divmod; llama.cpp makes the filter size a template parameter for 3 × 3 and 1 × 1 and uses a precomputed fast division
          otherwise.
        </p>
        </Translated>
      </Today>
      <Ask q={questionsS193['ch19/19.3/q-divmod']} />

      <h2><Translated vi="Bài tập 4: hợp nhất truy cập và xung đột bank">Exercise 4: coalescing and bank conflicts</Translated></h2>
      <Translated vi={<p>Bài tập 4 hỏi các truy cập toàn cục của Hình 19.11 có được hợp nhất (coalescing) không và các truy cập bộ nhớ chia sẻ của nó có xung đột không. Với các ô 16 × 16, một warp là hai hàng luồng: hai giá trị của ty, và tx từ 0 đến 15. Bước nạp F và bước ghi Y dịch một phần tử cho mỗi tx. Phép gom cũng vậy, trừ chỗ v sang hàng đầu ra kế tiếp: ở đó địa chỉ nhảy K, vì một hàng đầu vào dài hơn một hàng đầu ra K − 1.</p>}>
      <p>
        Exercise 4 asks whether Fig. 19.11's global accesses are coalesced and whether its shared-memory accesses conflict. With 16 × 16
        tiles a warp is two rows of threads: two values of ty, and tx from 0 to 15. The F load and the Y store move by one element per tx. The
        gather does too, except where v reaches the next output row: there the address jumps by K, because an input row is K − 1 longer than an
        output row.
      </p>
      </Translated>
      <Source section="19.6" pages="475" figs={tr("Exercise 4", 'Bài tập 4')} />
      <Translated vi={<p>Trong bộ nhớ chia sẻ, các lần ghi đi tới 32 word liên tiếp, <code>Fds[ty][k]</code> được đọc tại hai word cách nhau 16, và{' '}<code>Bds[k][tx]</code> tại 16 word liên tiếp, mỗi word bởi hai luồng (phân tích của khóa học này).</p>}>
      <p>
        In shared memory the stores go to 32 consecutive words, <code>Fds[ty][k]</code> is read at two words 16 apart, and{' '}
        <code>Bds[k][tx]</code> at 16 consecutive words, each by two threads (the analysis is ours).
      </p>
      </Translated>
      <Today checked="2026-11-20 (ch06-facts A12; infra-facts A10, B10, C13)">{/* i18n-ignore: date and fact citation */}
        <Translated vi={<p>NVIDIA: 32 bank, mỗi bank 4 byte; các luồng đọc cùng một word không xung đột. AMD: 32 hoặc 64 bank. Metal không công bố.</p>}>
        <p>
          NVIDIA: 32 banks of 4 bytes; threads reading the same word do not conflict. AMD: 32 or 64 banks. Metal documents none.
        </p>
        </Translated>
      </Today>
      <p><Translated vi="Trình khám phá trải ra các địa chỉ mà một warp chạm tới trong mỗi truy cập; hãy chọn hình dạng, khối, pha, warp và bước trong.">The explorer lays out the addresses one warp touches in each access; choose the shape, the block, the phase, the warp and the inner step.</Translated></p>
      <Simplification>
        <Translated vi={<p>Độ rộng warp và mô hình bank mà NVIDIA công bố. Các dải được đếm theo số float, không theo giao dịch bộ nhớ, và không có cache.</p>}>
        <p>
          NVIDIA's documented warp width and bank model. Runs are counted in floats, not memory transactions, and there are no caches.
        </p>
        </Translated>
      </Simplification>
      <WarpExplorer />
      <Ask q={questionsS193['ch19/19.3/q-ex4']} label={tr("Work it out", 'Tự tính')} />

      <h2><Translated vi="Khi nào dạng GEMM đáng giá">When the GEMM form pays off</Translated></h2>
      <Translated vi={<p>Phép nhân ma trận được tinh chỉnh trên mọi nền tảng, và một phép nhân phân ô thực hiện nhiều phép toán cho mỗi byte nó nạp. Nhưng sự dùng lại trên mỗi phần tử được nạp tăng theo chiều rộng ô, nên các ma trận nhỏ, vốn buộc phải dùng ô nhỏ hoặc phần lớn trống, mất gần hết lợi ích đó, và với các phép nhân nhỏ thì bản thân lần khởi chạy trở thành một chi phí đáng kể. Cách tiếp cận này đáng giá khi các ma trận lớn (p. 472).</p>}>
      <p>
        Matrix multiplication is tuned on every platform, and a tiled one does many operations per byte it loads. But the reuse per loaded
        element grows with the tile width, so small matrices, which force small or mostly empty tiles, lose most of it, and for small
        multiplications the launch itself becomes a noticeable cost. The approach pays when the matrices are large (p. 472).
      </p>
      </Translated>
      <Source section="19.3" pages="472" />
      <Translated vi={<p>Các mạng tích chập thường thuận lợi. F là M × C·K·K và B là C·K·K × H_out·W_out: ngoài M, mọi chiều đều là tích của các kích thước của lớp. Các lớp đầu có ít kênh và bản đồ lớn, các lớp cuối có nhiều kênh và bản đồ nhỏ, nên C·H_out·W_out vẫn lớn suốt cả mạng (p. 472). Tuy vậy không phải lúc nào cũng thế. LeNet C5 biến 16 bản đồ 5 × 5 thành 120 bản đồ một pixel: B là 400 × 1. Ở batch 1 đó là một phép nhân ma trận với vector, cùng vấn đề GEMM mỏng như một LLM sinh một token ở batch 1 (<a href={href('ch05', '5.1')}>mục 5.1</a>): mỗi trọng số chỉ được dùng một lần.</p>}>
      <p>
        Convolutional networks tend to cooperate. F is M × C·K·K and B is C·K·K × H_out·W_out: apart from M, every dimension is a product of
        the layer's sizes. Early layers have few channels and large maps, late layers many channels and small maps, so C·H_out·W_out stays
        large through the network (p. 472). Not always, though. LeNet C5 turns 16 maps of 5 × 5 into 120 maps of one pixel: B is 400 × 1. At
        batch 1 that is a matrix-vector product, the same thin-GEMM problem as an LLM generating one token at batch 1 (
        <a href={href('ch05', '5.1')}>section 5.1</a>): every weight is used once.
      </p>
      </Translated>
      <Source pages="458, 472" />
      <Translated vi={<p>Hãy so sánh các lớp bên dưới, rồi cho LeNet C5 một batch: trước tiên mỗi mẫu một GEMM, rồi cả batch trong một GEMM. Chỉ cách thứ hai mới thêm cột cho một GEMM, và nó trả giá bằng một phép hoán vị sau đó (câu hỏi thứ hai sau trình khám phá).</p>}>
      <p>
        Compare the layers below, then give LeNet C5 a batch: first one GEMM per sample, then the whole batch in one GEMM. Only the second
        adds columns to a GEMM, and it pays for them with a permute afterwards (the second question after the explorer).
      </p>
      </Translated>
      <Simplification>
        <Translated vi={<p>Chỉ có lưu lượng được yêu cầu, không có cache. Bốn trong năm lớp của lab có ở đây (đầu, giữa, cuối, 1 × 1); lớp thứ năm là hàng C3 ở N = 64. Bốn lớp này là của khóa học này, không phải của sách.</p>}>
        <p>
          Requested traffic only, no caches. Four of the lab's five layers are here (early, middle, late, 1 × 1); its fifth is the C3 row
          at N = 64. The four are ours, not the book's.
        </p>
        </Translated>
      </Simplification>
      <ShapesExplorer />
      <Translated vi={<p>Mã production mã hóa cùng lập luận đó thành các điều kiện bạn đọc được: một kích thước mà dưới đó dùng kernel trực tiếp, một trường hợp đặc biệt cho bộ lọc 1 × 1, và một cách tách một phép rút gọn dài để một đầu ra nhỏ vẫn lấp đầy GPU.</p>}>
      <p>
        Production code encodes the same argument in conditions you can read: a size below which a direct kernel is used instead, a
        special case for 1 × 1 filters, and a way to split a long reduction so that a small output still fills the GPU.
      </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch19-facts.md, D1, D4, D3)">{/* i18n-ignore: date and fact citation */}
        <Translated vi={<p>llama.cpp (CUDA và HIP) gửi các lớp có C·K·K dưới 64 và ít hơn 512 pixel đầu ra tới một kernel trực tiếp, gửi các tích chập 1 × 1 stride 1 không đệm thẳng tới cuBLAS trên đầu vào, và tách các phép rút gọn dài để các bản đồ nhỏ vẫn lấp đầy GPU. Đường dự phòng CUDA của PyTorch mở rộng từng mẫu một và bỏ qua bước mở rộng cho 1 × 1. MLX trên Metal chọn Winograd, implicit GEMM hoặc một phép mở rộng tường minh theo ô.</p>}>
        <p>
          llama.cpp (CUDA and HIP) sends layers with C·K·K below 64 and fewer than 512 output pixels to a direct kernel, sends unpadded
          stride-1 1 × 1 convolutions straight to cuBLAS on the input, and splits long reductions so small maps still fill the GPU. PyTorch's CUDA fallback
          unfolds one sample at a time and skips the unfold for 1 × 1. MLX on Metal picks Winograd, implicit GEMM or a tiled explicit unfold.
        </p>
        </Translated>
      </Today>
      <Ask q={questionsS193['ch19/19.3/q-thin']} />
      <Ask q={questionsS193['ch19/19.3/q-wholebatch']} />

      <h2><Translated vi="Hai trường hợp đặc biệt">Two special cases</Translated></h2>
      <Translated vi={<p>Với K = 1, Eq. 19.5 trở thành B[u, v] ← X[n, u, v / W_out, v % W_out]: B chính là X, nhìn như C × H·W, và tỉ số là 1. Một tích chập 1 × 1 là một lớp tuyến tính áp dụng tại mọi pixel, chính là điều các phép chiếu của một LLM làm tại mọi token (quan sát của khóa học này, từ ghi chú).</p>}>
      <p>
        With K = 1, Eq. 19.5 becomes B[u, v] ← X[n, u, v / W_out, v % W_out]: B is X itself, viewed as C × H·W, and the ratio is 1. A 1 × 1
        convolution is a linear layer applied at every pixel, which is what an LLM's projections do at every token (our observation, from the
        notes).
      </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Patch embedding của một bộ mã hóa thị giác là một tích chập có stride bằng K, không đệm; các bộ mã hóa đa phương thức của llama.cpp dựng đúng như vậy (ch19-facts D2). Các cửa sổ không chồng lấn thì không trùng lặp gì: tỉ số là 1 khi K chia hết kích thước bản đồ (E1). Hãy thử công tắc stride = K trong trình khám phá.</p>}>
        <p>
          A vision encoder's patch embedding is a convolution whose stride equals K, with no padding; llama.cpp's multimodal encoders build
          exactly that (ch19-facts D2). Windows that do not overlap duplicate nothing: the ratio is 1 when K divides the map (E1). Try the
          stride = K toggle in the explorer.
        </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS193['ch19/19.3/q-vit']} label={tr("Work it out", 'Tự tính')} />

      <h2><Translated vi="Về GEMM của dự án xuyên suốt">On the project's GEMM</Translated></h2>
      <Translated vi={<p>Lab 19.1 tính lớp này theo hai cách: trực tiếp, và như một phép unroll theo sau bởi lớp tuyến tính từ lab 6.4,{' '}<code>launchLinear(X, W, Y, batch, in, out)</code>, hàm tính Y = X·Wᵀ với W lưu dạng (out, in). Khớp lời gọi đó với Y = F · B cần suy nghĩ một chút.</p>}>
      <p>
        Lab 19.1 computes the layer two ways: directly, and as an unroll followed by the linear layer from lab 6.4,{' '}
        <code>launchLinear(X, W, Y, batch, in, out)</code>, which computes Y = X·Wᵀ with W stored (out, in). Matching that call to Y = F · B
        takes a moment's thought.
      </p>
      </Translated>
      <Ask q={questionsS193['ch19/19.3/q-roles']} />
      <Translated vi={<p>Vì vậy phép unroll ghi Bᵀ, mỗi pixel đầu ra một hàng C·K·K giá trị, và F được truyền vào nguyên như khi lưu. Hướng mỗi pixel một hàng đó cũng là thứ mà <code>cudnnIm2Col</code> của cuDNN và <code>ggml_im2col</code> của llama.cpp tạo ra.</p>}>
      <p>
        The unroll therefore writes Bᵀ, one row of C·K·K values per output pixel, and F goes in as stored. That row-per-pixel orientation is
        also what cuDNN's <code>cudnnIm2Col</code> and llama.cpp's <code>ggml_im2col</code> produce.
      </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch19-facts.md, A4, D2)">{/* i18n-ignore: date and fact citation */}
        <Translated vi={<p><code>cudnnIm2Col</code> (đã ngừng khuyến nghị từ cuDNN 9.0) dựng một ma trận có chiều cao batch·y_height·y_width; <code>ggml_im2col</code>{' '}trả về [N, OH, OW, IC·KH·KW]. Cả hai đều có mỗi pixel đầu ra một hàng.</p>}>
        <p>
          <code>cudnnIm2Col</code> (deprecated in cuDNN 9.0) builds a matrix of height batch·y_height·y_width; <code>ggml_im2col</code>{' '}
          returns [N, OH, OW, IC·KH·KW]. Both have one row per output pixel.
        </p>
        </Translated>
      </Today>
      <Translated vi={<p>Hãy thử trong lab trước: <a href={href('ch19', 'labs')}>trang lab</a> có lab 19.1 và phần dự đoán của nó. Hai phần của lời giải tham chiếu ở đây để xem sau.</p>}>
      <p>
        Try it in the lab first: the <a href={href('ch19', 'labs')}>labs page</a> has lab 19.1 and its prediction. The reference solution's two
        pieces are here for afterwards.
      </p>
      </Translated>
      <SolutionRegions />
      <Translated vi={<p>Nhìn lại bước nạp ô một lần cuối. Đó là chỗ duy nhất Hình 19.11 khác một GEMM phân ô thông thường, và không có gì trong vòng lặp tích vô hướng bên trong phải thay đổi.</p>}>
      <p>
        One last look at the tile load. It is the one place where Fig. 19.11 differs from a plain tiled GEMM, and nothing in the inner product
        loop had to change.
      </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Cũng chính chỗ đó là nơi một GEMM lượng tử hóa giải nén và giải lượng tử hóa trọng số của nó trong lúc nạp một ô. Mô-đun{' '}<a href={href('x2')}>X2, Kernel lượng tử hóa</a>, xây dựng dựa trên điều đó.</p>}>
        <p>
          The same slot is where a quantised GEMM unpacks and dequantises its weights while loading a tile. Module{' '}
          <a href={href('x2')}>X2, Quantised kernels</a>, builds on it.
        </p>
        </Translated>
      </Beyond>
    </>
  );
}
