import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { CORRECT, blurImage, patch, spot, type BlurOptions, type BlurResult } from '../../sim/blur';

// ---------------------------------------------------------------- drawing a small picture
// Shared by this scene and by the interactives of section 3.3.
export interface CellLook {
  text?: string;
  fill?: string;
  ink?: string;
  stroke?: string;
  strokeWidth?: number;
  dash?: boolean;
  /** Diagonal hatching: a position that is not part of the picture. */
  hatch?: boolean;
  bold?: boolean;
}

/**
 * The shade of a pixel value. A picture has to look the same in the light and the dark
 * theme, so these greys are computed from the data and are the one place where a colour
 * is not a theme variable. Low values are stretched so that a faint smudge stays visible;
 * the number printed in the cell is the data.
 */
export function grey(v: number): { fill: string; ink: string } {
  if (Number.isNaN(v)) return { fill: 'var(--fail-wash)', ink: 'var(--fail)' };
  const g = Math.round(255 * Math.pow(Math.max(0, Math.min(255, v)) / 255, 0.6));
  return { fill: `rgb(${g},${g},${g})`, ink: g > 128 ? 'rgb(20,20,20)' : 'rgb(242,242,242)' };
}
export const show = (v: number) => (Number.isNaN(v) ? '?' : String(v));

/** Cells of a w by h picture, plus `pad` cells of surroundings on every side, as an SVG group. */
export function PixelCells({
  x = 0, y = 0, w, h, pad = 0, cell = 30, look, hatchId, onCell,
}: {
  x?: number; y?: number; w: number; h: number; pad?: number; cell?: number;
  look: (row: number, col: number, inside: boolean) => CellLook | null;
  hatchId?: string;
  onCell?: (row: number, col: number) => void;
}) {
  const cells: ReactNode[] = [];
  const over: ReactNode[] = [];
  for (let row = -pad; row < h + pad; ++row)
    for (let col = -pad; col < w + pad; ++col) {
      const inside = row >= 0 && row < h && col >= 0 && col < w;
      const k = look(row, col, inside);
      if (!k) continue;
      const cx = x + (col + pad) * cell;
      const cy = y + (row + pad) * cell;
      const emphasised = (k.strokeWidth ?? 1) > 1;
      const box = (
        <g key={row + ',' + col} data-rc={row + ',' + col} onClick={onCell ? () => onCell(row, col) : undefined}>
          <rect x={cx + 1} y={cy + 1} width={cell - 2} height={cell - 2} rx="2"
            fill={k.fill ?? 'var(--cell)'} stroke={k.stroke ?? 'var(--rule)'} strokeWidth={k.strokeWidth ?? 1}
            strokeDasharray={k.dash ? '3 3' : undefined} />
          {k.hatch && hatchId ? <rect x={cx + 1} y={cy + 1} width={cell - 2} height={cell - 2} rx="2" fill={`url(#${hatchId})`} /> : null}
          {k.text != null ? (
            <text x={cx + cell / 2} y={cy + cell / 2 + cell * 0.13} fontSize={cell * 0.36} textAnchor="middle"
              fontWeight={k.bold ? 700 : 400} style={{ fill: k.ink ?? 'var(--ink)', fontVariantNumeric: 'tabular-nums' }}>{k.text}</text>
          ) : null}
        </g>
      );
      (emphasised ? over : cells).push(box); // thick outlines are drawn last so neighbours do not cover them
    }
  return <g>{cells}{over}</g>;
}

export function Hatch({ id }: { id: string }) {
  return (
    <defs>
      <pattern id={id} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <line x1="0" y1="0" x2="0" y2="6" stroke="var(--ink-3)" strokeWidth="1.4" />
      </pattern>
    </defs>
  );
}

// ---------------------------------------------------------------- the scene
// "Why the blur writes somewhere else": a 6x6 picture with one bright pixel, blurred
// into a second buffer and then in place, in two thread orders. Every picture below is
// computed by app/src/sim/blur.ts; nothing is drawn by hand.
const W = 6;
const H = 6;
const N = W * H;
const C = 36;
const LX = 12;   // left picture
const RX = 258;  // right picture
const GY = 46;
const TX = 496;  // text column
const pic = spot(W, H);
const base: BlurOptions = { ...CORRECT, blockW: W, blockH: H };
const states = (over: Partial<BlurOptions>) =>
  Array.from({ length: N + 1 }, (_, limit) => blurImage(pic, { ...base, ...over, limit }));
const separate = states({});
const forward = states({ inPlace: true });
const reversed = states({ inPlace: true, order: 'reversed' });
const ref = separate[N].reference;

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const wrongSoFar = (s: BlurResult) => s.out.filter((v, i) => s.written[i] && v !== ref[i]).length;
const inPatch = (t: { row: number; col: number } | undefined, row: number, col: number) =>
  !!t && patch(t.row, t.col, W, H, 1).some((q) => q.inside && q.curRow === row && q.curCol === col);

interface Grid {
  x: number;
  title: string;
  tier?: boolean;
  look: (row: number, col: number) => CellLook;
}
function Stage({ grids, lines, rule }: { grids: Grid[]; lines: { text: string; tone?: 'fail' | 'ok' | 'muted'; bold?: boolean }[]; rule?: number }) {
  return (
    <svg viewBox="0 0 720 330" role="img" aria-label={tr('Animation of a blur kernel writing to a second buffer and then in place', 'Hoạt hình một kernel làm mờ ghi vào buffer thứ hai, rồi ghi tại chỗ')}
      style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      {grids.map((g) => (
        <g key={g.title}>
          <text x={g.x} y={GY - 12} fontSize="12" fontWeight="600" style={{ fill: g.tier === false ? 'var(--ink-2)' : 'var(--global)' }}>{g.title}</text>
          <PixelCells x={g.x} y={GY} w={W} h={H} cell={C} look={(r, c) => g.look(r, c)} />
        </g>
      ))}
      {lines.map((l, i) => (
        <text key={i} x={TX} y={GY + 10 + i * 19} fontSize="12" fontWeight={l.bold ? 650 : 400}
          style={{ fill: l.tone === 'fail' ? 'var(--fail)' : l.tone === 'ok' ? 'var(--ok)' : l.tone === 'muted' ? 'var(--ink-2)' : 'var(--ink)' }}>{l.text}</text>
      ))}
      {rule ? (
        <g opacity={rule}>
          <text x={LX} y={GY + H * C + 30} fontSize="13.5" fontWeight="600"><Translated vi="Mỗi luồng chỉ ghi phần tử của riêng nó,">Each thread writes only its own element,</Translated></text>
          <text x={LX} y={GY + H * C + 50} fontSize="13.5" fontWeight="600"><Translated vi="và chỉ đọc dữ liệu mà không luồng nào ghi.">and reads only data that no thread writes.</Translated></text>
        </g>
      ) : null}
    </svg>
  );
}

/** A buffer as it stands in state s, with the reads of thread t outlined and wrong stored values marked. */
const buffer = (s: BlurResult, opts: { reader?: { row: number; col: number }; mark?: boolean; blankUnwritten?: boolean; hot?: number } = {}) =>
  (row: number, col: number): CellLook => {
    const i = row * W + col;
    if (opts.blankUnwritten && !s.written[i]) return { dash: true };
    const v = s.out[i];
    const bad = opts.mark && s.written[i] && v !== ref[i];
    const read = inPatch(opts.reader, row, col);
    return {
      text: show(v), ...grey(v),
      stroke: bad ? 'var(--fail)' : read ? 'var(--global)' : i === opts.hot ? 'var(--ink)' : undefined,
      strokeWidth: bad || read || i === opts.hot ? 3 : 1,
    };
  };
const input = (reader?: { row: number; col: number }) => (row: number, col: number): CellLook => {
  const v = pic.data[row * W + col];
  const read = inPatch(reader, row, col);
  return { text: show(v), ...grey(v), stroke: read ? 'var(--global)' : undefined, strokeWidth: read ? 3 : 1 };
};
const plain = (values: number[], mark = false) => (row: number, col: number): CellLook => {
  const v = values[row * W + col];
  const bad = mark && v !== ref[row * W + col];
  return { text: show(v), ...grey(v), stroke: bad ? 'var(--fail)' : undefined, strokeWidth: bad ? 3 : 1 };
};
const at = (t: { row: number; col: number }) => `(${t.row}, ${t.col})`;

const beats: Beat[] = [
  {
    clip: 'in-place-blur-1',
    get label() {
      return tr('Two buffers', 'Hai buffer');
    },
    draw: (p) => {
      const k = Math.min(N, Math.floor(ramp(p, 0.06, 0.78) * (N + 1)));
      const s = separate[k];
      const t = k < N ? separate[N].ran[k] : undefined;
      return (
        <Stage
          grids={[
            { x: LX, title: tr('in: every thread reads here', 'in: mọi luồng đọc ở đây'), look: input(t) },
            { x: RX, title: tr('out: every thread stores one element', 'out: mỗi luồng ghi một phần tử'), look: buffer(s, { blankUnwritten: true, hot: k > 0 ? separate[N].ran[k - 1].row * W + separate[N].ran[k - 1].col : undefined }) },
          ]}
          lines={[
            { text: tr(`${k} of ${N} threads done`, `${k} trên ${N} luồng đã xong`), bold: true },
            { text: t ? tr(`thread ${at(t)} reads its patch`, `luồng ${at(t)} đọc vùng lân cận của nó`) : tr('all threads done', 'mọi luồng đã xong'), tone: 'muted' },
            ...(k === N ? [{ text: tr(`${N - separate[N].wrong.length} of ${N} match the reference`, `${N - separate[N].wrong.length} trên ${N} khớp với kết quả tham chiếu`), tone: 'ok' as const, bold: true }] : []),
          ]}
        />
      );
    },
  },
  {
    clip: 'in-place-blur-2',
    get label() {
      return tr('One buffer', 'Một buffer');
    },
    draw: (p) => {
      const k = Math.min(8, Math.floor(ramp(p, 0.3, 0.6) * 9));
      const s = forward[k];
      const next = p >= 0.74 ? forward[N].ran[8] : undefined; // thread (1,2), the neighbour
      return (
        <Stage
          grids={[
            { x: LX, title: tr('one buffer: in and out', 'một buffer: vừa in vừa out'), look: buffer(s, { mark: true, reader: next }) },
            { x: RX, title: tr('what it should become', 'kết quả đúng phải là'), tier: false, look: plain(ref) },
          ]}
          lines={[
            { text: tr(`${k} of ${N} threads done`, `${k} trên ${N} luồng đã xong`), bold: true },
            ...(k >= 7 ? [{ text: tr('7 threads stored a 0 over a 0', '7 luồng ghi 0 đè lên 0'), tone: 'muted' as const }] : []),
            ...(k >= 8 ? [{ text: tr(`thread (1, 1) stores ${forward[8].out[7]}`, `luồng (1, 1) ghi ${forward[8].out[7]}`), tone: 'fail' as const, bold: true }] : []),
            ...(next ? [{ text: tr(`thread ${at(next)} is next:`, `luồng ${at(next)} chạy tiếp theo:`) }, { text: tr(`it reads ${forward[8].out[7]} where in held ${pic.data[7]}`, `nó đọc ${forward[8].out[7]} ở chỗ in từng là ${pic.data[7]}`) }] : []),
          ]}
        />
      );
    },
  },
  {
    clip: 'in-place-blur-3',
    get label() {
      return tr('In order', 'Theo thứ tự');
    },
    draw: (p) => {
      const k = 8 + Math.min(N - 8, Math.floor(ramp(p, 0.04, 0.62) * (N - 7)));
      const s = forward[k];
      const done = k === N;
      return (
        <Stage
          grids={[
            { x: LX, title: tr('one buffer, threads in order', 'một buffer, các luồng theo thứ tự'), look: buffer(s, { mark: true, reader: done ? undefined : forward[N].ran[k] }) },
            { x: RX, title: tr('the reference', 'kết quả tham chiếu'), tier: false, look: plain(ref) },
          ]}
          lines={[
            { text: tr(`${k} of ${N} threads done`, `${k} trên ${N} luồng đã xong`), bold: true },
            { text: tr(`wrong so far: ${wrongSoFar(s)}`, `số chỗ sai đến giờ: ${wrongSoFar(s)}`), tone: 'fail', bold: true },
            ...(done
              ? [
                  { text: tr(`${s.wrong.length} of ${N} pixels are wrong`, `${s.wrong.length} trên ${N} điểm ảnh bị sai`) },
                  { text: tr('accesses outside the buffer: ', 'số lần truy cập ra ngoài buffer: ') + s.accesses.length, tone: 'muted' as const },
                ]
              : []),
          ]}
        />
      );
    },
  },
  {
    clip: 'in-place-blur-4',
    get label() {
      return tr('Reversed', 'Đảo ngược');
    },
    draw: (p) => {
      const k = Math.min(N, Math.floor(ramp(p, 0.14, 0.62) * (N + 1)));
      const s = reversed[k];
      const done = k === N;
      const differ = forward[N].out.filter((v, i) => v !== reversed[N].out[i]).length;
      return (
        <Stage
          grids={[
            { x: LX, title: tr('same kernel, threads reversed', 'cùng kernel, các luồng theo thứ tự ngược'), look: buffer(s, { mark: true, reader: done ? undefined : reversed[N].ran[k] }) },
            { x: RX, title: tr('the run before, threads in order', 'lần chạy trước, các luồng theo thứ tự'), look: plain(forward[N].out, true) },
          ]}
          lines={[
            { text: tr(`${k} of ${N} threads done`, `${k} trên ${N} luồng đã xong`), bold: true },
            { text: tr(`wrong so far: ${wrongSoFar(s)}`, `số chỗ sai đến giờ: ${wrongSoFar(s)}`), tone: 'fail', bold: true },
            ...(done ? [{ text: tr(`the two runs differ in ${differ} pixels`, `hai lần chạy khác nhau ở ${differ} điểm ảnh`) }, { text: tr('same code, same input', 'cùng mã, cùng đầu vào'), tone: 'muted' as const }] : []),
          ]}
        />
      );
    },
  },
  {
    clip: 'in-place-blur-5',
    get label() {
      return tr('The rule', 'Quy tắc');
    },
    draw: (p) => (
      <Stage
        grids={[
          { x: LX, title: tr('in: read by many, written by nobody', 'in: nhiều luồng đọc, không luồng nào ghi'), look: input() },
          { x: RX, title: tr('out: each element written by one thread', 'out: mỗi phần tử do một luồng ghi'), look: plain(separate[N].out) },
        ]}
        lines={[
          { text: tr('grayscale, blur, matmul:', 'grayscale, làm mờ, matmul:'), bold: true },
          { text: tr('all three keep the rule', 'cả ba đều giữ quy tắc này'), tone: 'muted' },
          ...(p > 0.62 ? [{ text: tr('chapter 5: shared memory', 'chương 5: bộ nhớ chia sẻ (shared memory)') }, { text: tr('that threads also write,', 'mà các luồng cũng ghi vào,'), tone: 'muted' as const }, { text: tr('and the barriers it needs', 'và các rào chắn (barrier) nó cần'), tone: 'muted' as const }] : []),
        ]}
        rule={ramp(p, 0.12, 0.3)}
      />
    ),
  },
];

export function InPlaceBlurScene() {
  return <Scene chapter="ch03" title={tr('Why the blur writes somewhere else', 'Vì sao phép làm mờ ghi sang chỗ khác')} beats={beats} />;
}
