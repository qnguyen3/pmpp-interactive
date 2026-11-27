import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { BOOK_F, BOOK_X, UNFOLD_SCENE, bToX } from '../../sim/unfold';

// "Unfolding the input into a matrix": the book's example (Figs. 19.2 and 19.8),
// every number from sim/unfold.ts. Six beats, one narration clip each
// (content/ch19/narration/unfold-N.txt).
const S = UNFOLD_SCENE;
const s = S.shape;
const C_X = 26; // input cell
const XX = (c: number) => 20 + c * 96; // input map origins
const XY = 44;
const C_F = 22; // filter and matrix cell
const FX = 20; // F matrix origin
const FY = 178;
const BX = 336; // B origin
const BY = 44;
const C_Y = 30;
const YX = 486; // Y matrix origin
const YY = 44;

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const ease = (t: number) => t * t * (3 - 2 * t);

/** Centre of input pixel (c, h, w). */
const xAt = (c: number, h: number, w: number) => ({ x: XX(c) + w * C_X + C_X / 2, y: XY + h * C_X + C_X / 2 });
/** Centre of B[u, v]. */
const bAt = (u: number, v: number) => ({ x: BX + v * C_F + C_F / 2, y: BY + u * C_F + C_F / 2 });
/** Centre of F[m][c][p][q] drawn as six separate filters (beat 1) and as the 2 × 12 matrix (beat 2 on). */
const filterAt = (m: number, c: number, p: number, q: number) => ({ x: XX(c) + 16 + q * C_F + C_F / 2, y: FY - 26 + m * 60 + p * C_F + C_F / 2 });
const fAt = (m: number, k: number) => ({ x: FX + k * C_F + C_F / 2, y: FY + m * C_F + C_F / 2 });

function Cell({ x, y, size, value, hot, wash, dashed, fade = 1, tier = 'global' }: {
  x: number; y: number; size: number; value?: number | null; hot?: boolean; wash?: boolean; dashed?: boolean; fade?: number; tier?: 'global' | 'reg';
}) {
  return (
    <g opacity={fade}>
      <rect x={x - size / 2 + 1} y={y - size / 2 + 1} width={size - 2} height={size - 2} rx="2"
        fill={hot ? `var(--${tier})` : wash ? `var(--${tier}-wash)` : 'var(--cell)'}
        stroke={hot || wash ? `var(--${tier})` : 'var(--rule)'} strokeDasharray={dashed ? '3 2' : undefined} />
      {value != null ? (
        <text x={x} y={y + 4} fontSize="12" textAnchor="middle" style={{ fill: hot ? 'var(--page)' : 'var(--ink)' }}>{value}</text>
      ) : null}
    </g>
  );
}

function Label({ x, y, children, tier = 'global', anchor = 'start' }: { x: number; y: number; children: ReactNode; tier?: string; anchor?: 'start' | 'middle' | 'end' }) {
  return <text x={x} y={y} fontSize="12" fontWeight="600" textAnchor={anchor} style={{ fill: `var(--${tier})` }}>{children}</text>;
}

interface XOpts {
  hot?: (c: number, h: number, w: number) => boolean;
  copies?: number[];
}
function InputMaps({ hot, copies }: XOpts) {
  return (
    <g>
      <Label x={20} y={16}><Translated vi="X: 3 map đầu vào, 3 × 3 (bộ nhớ toàn cục)">X: 3 input maps, 3 × 3 (global memory)</Translated></Label>
      {[0, 1, 2].map((c) => (
        <g key={c}>
          <text x={XX(c)} y={XY - 6} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>map {c}</>}>map {c}</Translated></text>
          {Array.from({ length: 9 }, (_, i) => {
            const h = Math.floor(i / 3);
            const w = i % 3;
            const { x, y } = xAt(c, h, w);
            return (
              <g key={i}>
                <Cell x={x} y={y} size={C_X} value={BOOK_X[c * 9 + i]} hot={hot?.(c, h, w)} />
              </g>
            );
          })}
        </g>
      ))}
      {copies ? <Counters copies={copies} /> : null}
    </g>
  );
}

/** Copies of each input pixel so far, one small grid under each map's column (the same counts for every map). */
function Counters({ copies }: { copies: number[] }) {
  const cs = 20;
  const top = 266;
  return (
    <g>
      <text x={20} y={top - 8} fontSize="11" fontWeight="600" style={{ fill: 'var(--ink)' }}><Translated vi="số bản sao của mỗi pixel đầu vào trong B đến giờ">copies of each input pixel in B so far</Translated></text>
      {[0, 1, 2].map((c) => (
        <g key={c}>
          {copies.map((n, i) => {
            const x = XX(c) + (i % 3) * cs;
            const y = top + Math.floor(i / 3) * cs;
            return (
              <g key={i}>
                <rect x={x + 1} y={y + 1} width={cs - 2} height={cs - 2} rx="2" fill={n > 1 ? 'var(--global-wash)' : 'var(--cell)'} stroke={n > 0 ? 'var(--global)' : 'var(--rule)'} />
                <text x={x + cs / 2} y={y + cs / 2 + 4} fontSize="11" fontWeight={n > 1 ? 650 : 400} textAnchor="middle" style={{ fill: 'var(--ink)' }}>{n}</text>
              </g>
            );
          })}
        </g>
      ))}
    </g>
  );
}

/** The filters, moving from six 2 × 2 squares (t = 0) to the 2 × 12 matrix F (t = 1). */
function Filters({ t, hot, wash }: { t: number; hot?: (m: number, k: number) => boolean; wash?: (m: number, k: number) => boolean }) {
  const cells: ReactNode[] = [];
  for (let m = 0; m < 2; m++)
    for (let k = 0; k < 12; k++) {
      const c = Math.floor(k / 4);
      const p = Math.floor((k % 4) / 2);
      const q = k % 2;
      const a = filterAt(m, c, p, q);
      const b = fAt(m, k);
      const e = ease(t);
      cells.push(<Cell key={m * 12 + k} x={lerp(a.x, b.x, e)} y={lerp(a.y, b.y, e)} size={C_F} value={BOOK_F[m * 12 + k]} hot={hot?.(m, k)} wash={wash?.(m, k)} />);
    }
  return (
    <g>
      <Label x={20} y={FY - 40 + (t > 0.5 ? 12 : 0)}>{t > 0.5 ? tr('F: 2 × 12, the same array read row by row', 'F: 2 × 12, cùng mảng đó đọc theo từng hàng') : tr('F: 6 filters, 2 × 2 (one row per output map)', 'F: 6 bộ lọc, 2 × 2 (mỗi map đầu ra một hàng)')}</Label>
      {cells}
      {t >= 1
        ? [4, 8].map((k) => <line key={k} x1={FX + k * C_F} x2={FX + k * C_F} y1={FY - 2} y2={FY + 2 * C_F + 2} stroke="var(--ink)" strokeWidth="1.6" />)
        : null}
      {t >= 1 ? (
        <text x={FX} y={FY + 2 * C_F + 16} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
          <Translated vi="hàng m: các bộ lọc của map đầu ra m cho map đầu vào 0, 1, 2">row m: output map m's filters for input maps 0, 1, 2</Translated>
        </text>
      ) : null}
    </g>
  );
}

/** B, with the first `cols` columns filled, column `moving` in flight (fraction t), and highlights. */
function Unfolded({ cols, moving, t = 0, movingRows, hot, outline }: {
  cols: number; moving?: number; t?: number; movingRows?: number[]; hot?: (u: number, v: number) => boolean; outline?: boolean;
}) {
  const cells: ReactNode[] = [];
  const flying: ReactNode[] = [];
  for (let u = 0; u < S.rows; u++)
    for (let v = 0; v < S.cols; v++) {
      const { x, y } = bAt(u, v);
      const filled = v < cols || (v === moving && !movingRows) || (v === moving && movingRows && movingRows[u] === 1);
      const inFlight = v === moving && (!movingRows || movingRows[u] === 0.5);
      if (inFlight && t < 1) {
        const src = bToX(s, u, v);
        const a = xAt(src.c, src.h, src.w);
        const e = ease(t);
        cells.push(<Cell key={`e${u},${v}`} x={x} y={y} size={C_F} dashed />);
        flying.push(<Cell key={`f${u},${v}`} x={lerp(a.x, x, e)} y={lerp(a.y, y, e)} size={C_F} value={S.B[u * S.cols + v]} hot />);
      } else
        cells.push(<Cell key={`${u},${v}`} x={x} y={y} size={C_F} value={filled || (inFlight && t >= 1) ? S.B[u * S.cols + v] : null} dashed={!(filled || inFlight)} hot={hot?.(u, v)} />);
    }
  return (
    <g>
      <Label x={BX} y={16}>B: 12 × 4</Label>
      <text x={BX} y={30} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi="mỗi pixel đầu ra một cột">one column per output pixel</Translated></text>
      {cells}
      {[4, 8].map((u) => <line key={u} x1={BX - 2} x2={BX + 4 * C_F + 2} y1={BY + u * C_F} y2={BY + u * C_F} stroke="var(--ink)" strokeWidth="1.6" />)}
      {[0, 1, 2].map((c) => (
        <text key={c} x={BX + 4 * C_F + 6} y={BY + c * 4 * C_F + 2 * C_F + 4} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>map {c}</>}>map {c}</Translated></text>
      ))}
      {outline ? <rect x={BX - 3} y={BY - 3} width={4 * C_F + 6} height={12 * C_F + 6} fill="none" stroke="var(--ink)" strokeWidth="1.5" /> : null}
      {flying}
    </g>
  );
}

function YMatrix({ shown, hot }: { shown: number; hot?: number }) {
  return (
    <g>
      <Label x={YX} y={16}>Y = F · B: 2 × 4</Label>
      {S.Y.map((v, i) => {
        const r = Math.floor(i / 4);
        const c = i % 4;
        return <Cell key={i} x={YX + c * C_Y + C_Y / 2} y={YY + r * C_Y + C_Y / 2} size={C_Y} value={i < shown ? v : null} hot={hot === i} dashed={i >= shown} />;
      })}
    </g>
  );
}

function YMaps({ fade = 1, hot }: { fade?: number; hot?: number }) {
  return (
    <g opacity={fade}>
      <Label x={YX} y={YY + 2 * C_Y + 36}><Translated vi="các map đầu ra">the output maps</Translated></Label>
      {[0, 1].map((m) => (
        <g key={m}>
          <text x={YX + m * 92} y={YY + 2 * C_Y + 54} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>map {m}</>}>map {m}</Translated></text>
          {[0, 1, 2, 3].map((k) => (
            <Cell key={k} x={YX + m * 92 + (k % 2) * C_Y + C_Y / 2} y={YY + 2 * C_Y + 60 + Math.floor(k / 2) * C_Y + C_Y / 2} size={C_Y}
              value={S.Y[m * 4 + k]} hot={hot === m * 4 + k} />
          ))}
        </g>
      ))}
    </g>
  );
}

function Headline({ children }: { children: ReactNode }) {
  return <text x={20} y={356} fontSize="13" fontWeight="600" style={{ fill: 'var(--ink)' }}>{children}</text>;
}

function Stage({ children }: { children: ReactNode }) {
  return (
    <svg viewBox="0 0 720 368" role="img" aria-label={tr('Animation: unfolding the input into a matrix', 'Hoạt ảnh: unfold đầu vào thành một ma trận')} style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      {children}
    </svg>
  );
}

const patch0 = (c: number, h: number, w: number, only?: number) => (only === undefined || c === only) && h < 2 && w < 2;
const col0 = Array.from({ length: 12 }, (_, u) => bToX(s, u, 0));

const beats: Beat[] = [
  {
    clip: 'unfold-1',
    get label() {
      return tr('One output', 'Một đầu ra');
    },
    draw: (p) => {
      const ch = p < 0.15 ? -1 : Math.min(2, Math.floor((p - 0.15) / 0.22));
      const done = p >= 0.81;
      const sums = S.perChannel.slice(0, ch + 1);
      return (
        <Stage>
          <InputMaps hot={(c, h, w) => ch >= 0 && !done && patch0(c, h, w, ch)} />
          <Filters t={0} hot={(m, k) => m === 0 && ch >= 0 && !done && Math.floor(k / 4) === ch} wash={(m) => m === 0} />
          <YMaps hot={0} fade={1} />
          <text x={YX} y={244} fontSize="12" style={{ fill: 'var(--ink-2)' }}><Translated vi="map đầu ra 0, pixel (0, 0):">output map 0, pixel (0, 0):</Translated></text>
          {sums.map((v, c) => (
            <text key={c} x={YX} y={262 + c * 17} fontSize="12" style={{ fill: 'var(--ink)' }}><Translated vi={<>vùng của map {c} · bộ lọc = {v}</>}>map {c} patch · filter = {v}</Translated></text>
          ))}
          {done ? <text x={YX} y={318} fontSize="13" fontWeight="600" style={{ fill: 'var(--reg)' }}>{S.perChannel.join(' + ')} = {S.Y[0]}</text> : null}
          <Headline><Translated vi="một pixel đầu ra = tổng của một vùng trên mỗi map đầu vào nhân với bộ lọc của nó">one output pixel = one patch of every input map times its filter, summed</Translated></Headline>
        </Stage>
      );
    },
  },
  {
    clip: 'unfold-2',
    get label() {
      return tr('F is already a matrix', 'F đã là một ma trận');
    },
    draw: (p) => (
      <Stage>
        <InputMaps />
        <Filters t={ramp(p, 0.12, 0.62)} wash={(m) => p > 0.7 && m === 0} />
        <Headline>{p < 0.62 ? tr('M × C × K × K, row-major …', 'M × C × K × K, thứ tự hàng (row-major) …') : tr('… is already M × (C·K·K): no data moves', '… đã là M × (C·K·K): không dữ liệu nào phải di chuyển')}</Headline>
      </Stage>
    ),
  },
  {
    clip: 'unfold-3',
    get label() {
      return tr('Column 0', 'Cột 0');
    },
    draw: (p) => {
      const c = Math.min(2, Math.floor(ramp(p, 0.08, 0.92) * 3));
      const t = ramp(p, 0.08 + c * 0.28, 0.08 + c * 0.28 + 0.22);
      const rows = Array.from({ length: 12 }, (_, u) => (Math.floor(u / 4) < c ? 1 : Math.floor(u / 4) === c ? 0.5 : 0));
      return (
        <Stage>
          <InputMaps hot={(cc, h, w) => p >= 0.08 && patch0(cc, h, w, c)} />
          <Filters t={1} wash={(m, k) => m === 0 && Math.floor(k / 4) <= c && p >= 0.08} />
          <Unfolded cols={0} moving={p >= 0.08 ? 0 : undefined} t={t} movingRows={p >= 0.08 ? rows : rows.map(() => 0)} />
          <YMatrix shown={0} />
          <Headline>{tr(`column 0 = pixel 0's patches, map 0, then map 1, then map 2: ${col0.filter((x) => x.c <= c).length} of 12 entries`, `cột 0 = các vùng của pixel 0, map 0, rồi map 1, rồi map 2: ${col0.filter((x) => x.c <= c).length} trên 12 phần tử`)}</Headline>
        </Stage>
      );
    },
  },
  {
    clip: 'unfold-4',
    get label() {
      return tr('All four columns', 'Cả bốn cột');
    },
    draw: (p) => {
      const j = 1 + Math.min(2, Math.floor(ramp(p, 0.05, 0.86) * 3));
      const t = ramp(p, 0.05 + (j - 1) * 0.27, 0.05 + (j - 1) * 0.27 + 0.2);
      const doneCols = t >= 1 ? j + 1 : j;
      const counts = S.copiesAfter[doneCols];
      const vv = j;
      const src = Array.from({ length: 12 }, (_, u) => bToX(s, u, vv));
      return (
        <Stage>
          <InputMaps hot={(c, h, w) => t < 1 && src.some((x) => x.c === c && x.h === h && x.w === w)} copies={counts} />
          <Filters t={1} />
          <Unfolded cols={j} moving={j} t={t} />
          <YMatrix shown={0} />
          <Headline>
            {p < 0.88
              ? tr(`column ${vv}: pixel ${vv}'s patches. Counters show copies of each input pixel so far.`, `cột ${vv}: các vùng của pixel ${vv}. Bộ đếm cho thấy số bản sao của mỗi pixel đầu vào đến giờ.`)
              : tr(`${S.entries} entries from ${S.inputs} values: corners 1, edges 2, centre 4`, `${S.entries} phần tử từ ${S.inputs} giá trị: góc 1, cạnh 2, giữa 4`)}
          </Headline>
        </Stage>
      );
    },
  },
  {
    clip: 'unfold-5',
    get label() {
      return tr('F times B', 'F nhân B');
    },
    draw: (p) => {
      const first = p < 0.45;
      const shown = first ? (p > 0.32 ? 1 : 0) : Math.min(8, 1 + Math.floor(ramp(p, 0.45, 0.8) * 7.99));
      return (
        <Stage>
          <InputMaps />
          <Filters t={1} hot={(m) => first && m === 0} />
          <Unfolded cols={4} hot={(_, v) => first && v === 0} />
          <YMatrix shown={shown} hot={first ? 0 : undefined} />
          <YMaps fade={ramp(p, 0.82, 0.95)} />
          {first ? (
            <text x={YX} y={YY + 2 * C_Y + 20} fontSize="12" style={{ fill: 'var(--reg)' }}><Translated vi={<>hàng 0 · cột 0 = 12 tích = {S.Y[0]}</>}>row 0 · column 0 = 12 products = {S.Y[0]}</Translated></text>
          ) : null}
          <Headline>{p < 0.82 ? tr('every entry of Y is a row of F times a column of B', 'mỗi phần tử của Y là một hàng của F nhân một cột của B') : tr('row m of Y is output map m, already row-major: no reshuffle', 'hàng m của Y là map đầu ra m, đã theo thứ tự hàng: không cần sắp xếp lại')}</Headline>
        </Stage>
      );
    },
  },
  {
    clip: 'unfold-6',
    get label() {
      return tr('Scale up', 'Phóng to quy mô');
    },
    draw: (p) => {
      const x0 = 70;
      const x1 = 470;
      const y0 = 296;
      const y1 = 40;
      const maxSide = 64;
      const maxR = 26;
      const X = (side: number) => x0 + ((side - 3) / (maxSide - 3)) * (x1 - x0);
      const Y = (r: number) => y0 - (r / maxR) * (y0 - y1);
      const upTo = 3 + ramp(p, 0.05, 0.6) * (maxSide - 3);
      return (
        <Stage>
          <Label x={20} y={20} tier="ink"><Translated vi="số phần tử của B trên mỗi phần tử của X (Eq. 19.4), với map vuông cạnh H">elements of B per element of X (Eq. 19.4), for square maps of side H</Translated></Label>
          <line x1={x0} x2={x1} y1={y0} y2={y0} stroke="var(--ink-3)" />
          <line x1={x0} x2={x0} y1={y0} y2={y1} stroke="var(--ink-3)" />
          {[0, 5, 10, 15, 20, 25].map((r) => (
            <text key={r} x={x0 - 8} y={Y(r) + 4} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{r}</text>
          ))}
          {[3, 16, 32, 48, 64].map((h) => (
            <text key={h} x={X(h)} y={y0 + 16} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{h}</text>
          ))}
          <text x={(x0 + x1) / 2} y={y0 + 30} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}><Translated vi="cạnh H của map đầu vào">input map side H</Translated></text>
          {S.curves.map(({ K, points }) => {
            const pts = points.filter((q) => !Number.isNaN(q.ratio) && q.side <= upTo);
            return (
              <g key={K}>
                <line x1={x0} x2={x1} y1={Y(K * K)} y2={Y(K * K)} stroke="var(--ink-3)" strokeDasharray="4 3" />
                <text x={x1 + 6} y={Y(K * K) + 4} fontSize="11" style={{ fill: 'var(--global)' }}><Translated vi={<>K = {K}: giới hạn {K * K}</>}>K = {K}: limit {K * K}</Translated></text>
                <polyline fill="none" stroke="var(--global)" strokeWidth="2" points={pts.map((q) => `${X(q.side)},${Y(q.ratio)}`).join(' ')} />
              </g>
            );
          })}
          <circle cx={X(3)} cy={Y(16 / 9)} r="4" fill="var(--ink)" />
          <text x={X(3) + 10} y={Y(16 / 9) + 12} fontSize="11" style={{ fill: 'var(--ink)' }}><Translated vi={<>ví dụ của sách: {(16 / 9).toFixed(2)}</>}>the book's example: {(16 / 9).toFixed(2)}</Translated></text>
          {upTo >= 32 ? (
            <>
              <circle cx={X(32)} cy={Y(S.c1Ratio)} r="4" fill="var(--ink)" />
              <text x={X(32) + 8} y={Y(S.c1Ratio) + 16} fontSize="11" style={{ fill: 'var(--ink)' }}>LeNet C1 (K = 5, 32 × 32): {S.c1Ratio.toFixed(2)}</text>
            </>
          ) : null}
          {p > 0.68 ? (
            <g>
              <text x={592} y={150} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}><Translated vi="B tường minh tốn">explicit B costs</Translated></text>
              <text x={592} y={170} fontSize="12" style={{ fill: 'var(--ink)' }}><Translated vi="đọc X">read X</Translated></text>
              <text x={592} y={188} fontSize="12" style={{ fill: 'var(--ink)' }}><Translated vi="ghi B">write B</Translated></text>
              <text x={592} y={206} fontSize="12" style={{ fill: 'var(--ink)' }}><Translated vi="đọc lại B">read B back</Translated></text>
              <text x={592} y={226} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi="thêm một lượt qua">one more pass over</Translated></text>
              <text x={592} y={240} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi="bộ nhớ toàn cục">global memory</Translated></text>
            </g>
          ) : null}
          <Headline><Translated vi="tỉ lệ tiến tới K², bất kể C và M là bao nhiêu">the ratio approaches K², whatever C and M are</Translated></Headline>
        </Stage>
      );
    },
  },
];

export function UnfoldScene() {
  return <Scene chapter="ch19" title={tr('Unfolding the input into a matrix', 'Unfold đầu vào thành một ma trận')} beats={beats} />;
}
