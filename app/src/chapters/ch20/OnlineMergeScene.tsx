import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import {
  DEMO_SCORES,
  V12,
  exactOver,
  mergeStates,
  partialState,
  runOnline,
  type TileStep,
} from '../../sim/onlinesoftmax';
import { fv } from './OnlineSoftmaxStepper';
import { Translated, tr } from '../../lib/i18n';

// "Merging softmax one tile at a time" (section 20.5): the scene row of the stepper, tiles
// of 4, merged left to right in the code's deferred form. Every number drawn comes from
// src/sim/onlinesoftmax.ts; the scripts are content/ch20/narration/online-merge-N.txt and
// S205.narration.test.ts ties their numbers to these.
export const SCENE = runOnline({ scores: DEMO_SCORES, V: V12, bc: 4, row: 11, order: 'ltr', init: 'neg-inf', form: 'deferred' });
const ALL = Array.from({ length: 12 }, (_, j) => j);
const EXACT = exactOver(DEMO_SCORES, V12, ALL)!;
export const SPLIT = {
  A: partialState(DEMO_SCORES, V12, ALL.slice(0, 8)),
  B: partialState(DEMO_SCORES, V12, ALL.slice(8)),
};
export const MERGED = mergeStates(SPLIT.A, SPLIT.B);

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

// Geometry
const Y0 = 118; // score baseline
const SY = 26; // px per score unit
const BW = 22; // bar width
const barX = (j: number) => 34 + j * (BW + 6) + Math.floor(j / 4) * 16;
const DX = 486; // D bar origin
const DS = 62; // px per unit of D
const OX = 590; // Õ bars zero line
const OS = 70; // px per unit of Õ

interface Frame {
  lit?: Set<number>; // highlighted score columns
  done?: Set<number>; // merged columns
  m?: number | null;
  mFrom?: number; // the max line spans these columns
  mTo?: number;
  D?: number | null;
  O?: number[] | null;
  out?: number[] | null;
  exact?: boolean;
  factor?: { value: number; opacity: number } | null;
  bracket?: number; // opacity of the "all twelve" bracket
  expRow?: (number | null)[] | null; // e^(l − m) under the current tile
  expTile?: number;
  headline?: ReactNode;
  split?: number; // opacity of the two-block picture
}

function Stage(f: Frame) {
  const mY = f.m == null ? null : Y0 - f.m * SY;
  return (
    <svg viewBox="0 0 720 330" role="img" aria-label={tr('Animation of the online softmax merging tiles', 'Hoạt hình online softmax trộn các ô')} style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      <text x={34} y={18} fontSize="12" fontWeight="600"><Translated vi="một hàng điểm số, chia thành các ô 4 phần tử">one row of scores, in tiles of 4</Translated></text>
      <line x1={26} x2={430} y1={Y0} y2={Y0} stroke="var(--rule)" />
      {DEMO_SCORES.map((s, j) => {
        const h = s * SY;
        const lit = f.lit?.has(j);
        const done = f.done?.has(j);
        return (
          <g key={j} opacity={f.split ? 1 - 0.75 * f.split : 1}>
            <rect x={barX(j)} y={h >= 0 ? Y0 - h : Y0} width={BW} height={Math.abs(h)} rx="2"
              fill={lit ? 'var(--ink)' : done ? 'var(--ink-3)' : 'var(--cell)'} stroke="var(--ink-2)" />
            <text x={barX(j) + BW / 2} y={h >= 0 ? Y0 + 13 : Y0 - 5} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{fv(s)}</text>
          </g>
        );
      })}
      <text x={26} y={Y0 + 72} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>V</text>
      {V12.map((row, j) =>
        row.map((v, c) => (
          <rect key={`${j}-${c}`} x={barX(j) + 1} y={Y0 + 56 + c * 9} width={BW - 2} height={8} rx="1"
            fill="var(--ink-2)" fillOpacity={0.12 + 0.6 * Math.abs(v)} opacity={f.split ? 1 - 0.75 * f.split : 1}>
            <title>{`V[${j}][${c}] = ${v}`}</title>
          </rect>
        )),
      )}
      {[0, 1, 2].map((t) => (
        <text key={t} x={barX(t * 4) + 2 * (BW + 6) - 3} y={Y0 + 100} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>ô {t + 1}</>}>tile {t + 1}</Translated></text>
      ))}
      {f.bracket ? (
        <g opacity={f.bracket}>
          <path d={`M ${barX(0)} 32 v -6 H ${barX(11) + BW} v 6`} fill="none" stroke="var(--ink)" strokeWidth="1.4" />
          <text x={(barX(0) + barX(11) + BW) / 2} y={22} fontSize="11" textAnchor="middle"><Translated vi="giá trị lớn nhất và tổng của cả mười hai, trước khi có bất kỳ xác suất nào">max and sum of all twelve, before any probability</Translated></text>
        </g>
      ) : null}
      {mY != null && f.mFrom != null && f.mTo != null && Number.isFinite(mY) ? (
        <g>
          <line x1={barX(f.mFrom) - 3} x2={barX(f.mTo) + BW + 3} y1={mY} y2={mY} stroke="var(--ink)" strokeDasharray="4 3" strokeWidth="1.5" />
          <text x={barX(f.mTo) + BW + 6} y={mY + 4} fontSize="11" fontWeight="600">m = {fv(f.m!)}</text>
        </g>
      ) : null}
      {f.expRow && f.expTile != null ? (
        <g>
          <text x={26} y={Y0 + 44} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>e^(l − m)</text>
          {f.expRow.map((e, i) => (
            <text key={i} x={barX(f.expTile! * 4 + i) + BW / 2} y={Y0 + 44} fontSize="9.5" textAnchor="middle">{e == null ? '' : fv(e)}</text>
          ))}
        </g>
      ) : null}

      {/* running state; in the last beat it gives way to the two-block picture */}
      <g opacity={f.split ? 1 - f.split : 1}>
      <text x={DX - 20} y={38} fontSize="12" fontWeight="600"><Translated vi="trạng thái đang chạy">running state</Translated></text>
      <text x={DX - 20} y={64} fontSize="11.5">D</text>
      <rect x={DX} y={54} width={Math.max(0, (f.D ?? 0) * DS)} height={14} fill="var(--ink-3)" />
      <text x={DX + Math.max(0, (f.D ?? 0) * DS) + 6} y={65} fontSize="11">{f.D == null ? '0' : fv(f.D)}</text>
      <text x={DX - 20} y={98} fontSize="11.5">{f.out ? 'o = Õ / D' : 'Õ'}</text>
      <line x1={OX} x2={OX} y1={84} y2={150} stroke="var(--rule)" />
      {[0, 1, 2].map((c) => {
        const v = (f.out ?? f.O ?? [0, 0, 0])[c];
        const w = v * OS;
        return (
          <g key={c}>
            <rect x={w >= 0 ? OX : OX + w} y={88 + c * 20} width={Math.abs(w)} height={13} fill="var(--ink-2)" />
            {f.exact ? (
              <rect x={EXACT[c] >= 0 ? OX : OX + EXACT[c] * OS} y={86 + c * 20} width={Math.abs(EXACT[c] * OS)} height={17} fill="none" stroke="var(--ok)" strokeWidth="1.6" strokeDasharray="3 2" />
            ) : null}
            <text x={w >= 0 ? OX + w + 5 : OX + 5} y={98 + c * 20} fontSize="10.5">{fv(v)}</text>
          </g>
        );
      })}
      {f.factor ? (
        <g opacity={f.factor.opacity}>
          <rect x={DX - 20} y={166} width={232} height={26} rx="3" fill="var(--ink)" />
          <text x={DX + 96} y={183} fontSize="12" textAnchor="middle" style={{ fill: 'var(--page)' }}>× e^(m_old − m_new) = {fv(f.factor.value)}</text>
        </g>
      ) : null}
      {f.exact ? (
        <text x={DX - 20} y={214} fontSize="11" style={{ fill: 'var(--ok)' }}><Translated vi={<>nét đứt: chính xác (float64); lệch {fv(SCENE.diff)}</>}>dashed: exact (float64); off by {fv(SCENE.diff)}</Translated></text>
      ) : null}
      </g>

      {f.split ? (
        <g opacity={f.split}>
          {(
            [
              [tr('block 1: columns 0 to 7', 'khối 1: cột 0 đến 7'), SPLIT.A, 30],
              [tr('block 2: columns 8 to 11', 'khối 2: cột 8 đến 11'), SPLIT.B, 102],
            ] as const
          ).map(([label, s, y]) => (
            <g key={label}>
              <rect x={466} y={y} width={244} height={58} rx="4" fill="var(--page)" stroke="var(--ink-2)" />
              <text x={476} y={y + 18} fontSize="11.5" fontWeight="600">{label}</text>
              <text x={476} y={y + 36} fontSize="11">m = {fv(s.m)}, D = {fv(s.D)}</text>
              <text x={476} y={y + 51} fontSize="11">Õ = ({s.O.map(fv).join(', ')})</text>
            </g>
          ))}
          <path d="M 588 160 V 184" fill="none" stroke="var(--ink-2)" />
          <text x={596} y={177} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="Eq. 20.4, và 20.6 trên Õ">Eq. 20.4, and 20.6 on Õ</Translated></text>
          <rect x={466} y={184} width={244} height={58} rx="4" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.5" />
          <text x={476} y={202} fontSize="11.5" fontWeight="600"><Translated vi="đã trộn">merged</Translated></text>
          <text x={476} y={220} fontSize="11">m = {fv(MERGED.m)}, D = {fv(MERGED.D)}</text>
          <text x={476} y={235} fontSize="11">Õ / D = ({MERGED.O.map((x) => fv(x / MERGED.D)).join(', ')})</text>
        </g>
      ) : null}

      {f.headline ? <text x={26} y={318} fontSize="13" fontWeight="600">{f.headline}</text> : null}
    </svg>
  );
}

const cols = (t: number) => [0, 1, 2, 3].map((i) => t * 4 + i);
const upTo = (t: number) => new Set(Array.from({ length: (t + 1) * 4 }, (_, j) => j));
const before = (k: number): { D: number; O: number[] } => (k === 0 ? { D: 0, O: [0, 0, 0] } : { D: SCENE.steps[k - 1].D, O: SCENE.steps[k - 1].O });

/** A merge as an animation: rescale what is kept, then add the tile's terms. */
function mergeFrame(k: number, p: number, headline: ReactNode): Frame {
  const s: TileStep = SCENE.steps[k];
  const prev = before(k);
  const shrink = ramp(p, 0.15, 0.45);
  const grow = ramp(p, 0.55, 0.85);
  const keptD = prev.D * lerp(1, s.aFactor, shrink);
  const keptO = prev.O.map((o) => o * lerp(1, s.aFactor, shrink));
  const D = keptD + (s.D - prev.D * s.aFactor) * grow;
  const O = keptO.map((o, c) => o + (s.O[c] - prev.O[c] * s.aFactor) * grow);
  return {
    lit: new Set(cols(s.tile)),
    done: k > 0 ? upTo(k - 1) : undefined,
    m: p < 0.12 && k > 0 ? s.mOld : s.mNew,
    mFrom: 0,
    mTo: k * 4 + 3,
    D,
    O,
    factor: k > 0 ? { value: s.aFactor, opacity: ramp(p, 0.08, 0.2) * (1 - ramp(p, 0.9, 1)) } : null,
    expRow: p > 0.5 ? s.p : null,
    expTile: s.tile,
    headline,
  };
}

const beats: Beat[] = [
  {
    clip: 'online-merge-1',
    get label() {
      return tr('The row', 'Hàng');
    },
    draw: (p) => <Stage bracket={ramp(p, 0.2, 0.45)} headline={tr('an ordinary softmax needs the whole row first', 'một softmax thông thường cần cả hàng trước đã')} />,
  },
  {
    clip: 'online-merge-2',
    get label() {
      return tr('Tile 1', 'Ô 1');
    },
    draw: (p) => <Stage {...mergeFrame(0, p, tr(`tile 1: its own max ${fv(SCENE.steps[0].mB)}, nothing kept yet to rescale`, `ô 1: giá trị lớn nhất riêng ${fv(SCENE.steps[0].mB)}, chưa giữ gì để co giãn lại`))} />,
  },
  {
    clip: 'online-merge-3',
    get label() {
      return tr('A larger max', 'Giá trị lớn nhất lớn hơn');
    },
    draw: (p) => <Stage {...mergeFrame(1, p, p < 0.5 ? tr(`the max rises to ${fv(SCENE.steps[1].mNew)}: D and Õ shrink first`, `giá trị lớn nhất tăng lên ${fv(SCENE.steps[1].mNew)}: D và Õ co lại trước`) : tr('then tile 2’s terms add on', 'rồi các số hạng của ô 2 được cộng vào'))} />,
  },
  {
    clip: 'online-merge-4',
    get label() {
      return tr('Factor 1', 'Hệ số 1');
    },
    draw: (p) => <Stage {...mergeFrame(2, p, tr(`tile 3’s max ${fv(SCENE.steps[2].mB)} is lower: the factor is exactly 1`, `giá trị lớn nhất của ô 3 là ${fv(SCENE.steps[2].mB)}, thấp hơn: hệ số đúng bằng 1`))} />,
  },
  {
    clip: 'online-merge-5',
    get label() {
      return tr('One division', 'Một phép chia');
    },
    draw: (p) => {
      const t = ramp(p, 0.15, 0.5);
      const last = SCENE.steps[2];
      return (
        <Stage done={upTo(2)} m={last.mNew} mFrom={0} mTo={11} D={last.D}
          O={t < 1 ? last.O.map((o, c) => lerp(o, last.out[c], t)) : null} out={t >= 1 ? last.out : null}
          exact={p > 0.55} headline={tr('divide once at the end; it matches exact attention', 'chia một lần ở cuối; kết quả khớp với attention chính xác')} />
      );
    },
  },
  {
    clip: 'online-merge-6',
    get label() {
      return tr('Why it matters', 'Vì sao điều này quan trọng');
    },
    draw: (p) => (
      <Stage done={upTo(2)} m={SCENE.steps[2].mNew} mFrom={0} mTo={11} D={SCENE.steps[2].D} out={SCENE.final} split={ramp(p, 0.25, 0.5)} headline={p < 0.25 ? tr('no row of N scores ever existed', 'chưa từng có một hàng N điểm số nào tồn tại') : tr('two numbers per row let separate blocks merge', 'hai con số mỗi hàng cho phép các khối riêng rẽ trộn với nhau')} />
    ),
  },
];

export function OnlineMergeScene() {
  return <Scene chapter="ch20" title={tr('Merging softmax one tile at a time', 'Trộn softmax từng ô một')} beats={beats} />;
}
