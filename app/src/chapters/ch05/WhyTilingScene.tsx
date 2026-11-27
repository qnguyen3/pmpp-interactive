import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { makeProblem } from '../../sim/tiled';
import { tr } from '../../lib/i18n';

// "Why tiling works": block (0,0) of a 4x4 multiplication with 2x2 tiles, drawn
// from the same small-integer problem the stepper uses. Six beats, one narration
// clip each (content/ch05/narration/why-tiling-N.txt).
const W = 4;
const T = 2;
const prob = makeProblem(W);
const C = 36; // cell size
const MX = 24, MY = 190; // M origin
const NX = 196, NY = 22; // N origin
const PX = 196, PY = 190; // P origin
const SX = 392; // shared tiles and counters

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

interface Frame {
  hotM?: Set<number>;
  hotN?: Set<number>;
  washM?: Set<number>;
  washN?: Set<number>;
  times?: number; // opacity of the "×2" overlay
  tiles?: number; // opacity of tile outlines
  shared?: number; // opacity of the shared tiles
  mds?: (number | null)[];
  nds?: (number | null)[];
  hotShared?: boolean;
  barrier?: string | null;
  sums?: (number | null)[];
  pOut?: boolean;
  headline?: ReactNode;
  bars?: number; // 0..1 growth of the comparison bars
}

function Grid({ x, y, values, name, hot, wash, show = true }: {
  x: number; y: number; values: (number | null)[]; name: string; hot?: Set<number>; wash?: Set<number>; show?: boolean;
}) {
  return (
    <g>
      <text x={x} y={y - 6} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}>{name}</text>
      {values.map((v, i) => {
        const r = Math.floor(i / W);
        const c = i % W;
        const isHot = hot?.has(i);
        const isWash = wash?.has(i);
        return (
          <g key={i}>
            <rect x={x + c * C} y={y + r * C} width={C - 2} height={C - 2} rx="2"
              fill={isHot ? 'var(--global)' : isWash ? 'var(--global-wash)' : 'var(--cell)'}
              stroke={isHot || isWash ? 'var(--global)' : 'var(--rule)'} />
            {show && v != null ? (
              <text x={x + c * C + C / 2 - 1} y={y + r * C + C / 2 + 3} fontSize="12" textAnchor="middle"
                style={{ fill: isHot ? 'var(--page)' : 'var(--ink)' }}>{v}</text>
            ) : null}
          </g>
        );
      })}
    </g>
  );
}

function Tile({ x, y, values, name, hot }: { x: number; y: number; values: (number | null)[]; name: string; hot?: boolean }) {
  return (
    <g>
      <text x={x} y={y - 6} fontSize="12" fontWeight="600" style={{ fill: 'var(--shared)' }}>{name}</text>
      {values.map((v, i) => (
        <g key={i}>
          <rect x={x + (i % T) * C} y={y + Math.floor(i / T) * C} width={C - 2} height={C - 2} rx="2"
            fill={v == null ? 'var(--cell)' : hot ? 'var(--shared)' : 'var(--shared-wash)'} stroke="var(--shared)" />
          {v != null ? (
            <text x={x + (i % T) * C + C / 2 - 1} y={y + Math.floor(i / T) * C + C / 2 + 3} fontSize="12" textAnchor="middle"
              style={{ fill: hot ? 'var(--page)' : 'var(--ink)' }}>{v}</text>
          ) : null}
        </g>
      ))}
    </g>
  );
}

function Stage(f: Frame) {
  const pVals = prob.ref.map((v, i) => (f.pOut && Math.floor(i / W) < T && i % W < T ? v : null));
  return (
    <svg viewBox="0 0 720 348" role="img" aria-label={tr('Animation of one block multiplying matrices', 'Hoạt hình một khối nhân hai ma trận')} style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      <Grid x={MX} y={MY} values={prob.M} name="M" hot={f.hotM} wash={f.washM} />
      <Grid x={NX} y={NY} values={prob.N} name="N" hot={f.hotN} wash={f.washN} />
      <Grid x={PX} y={PY} values={pVals} name="P = M × N" wash={new Set([0, 1, 4, 5])} />
      <text x={PX + 2 * C + 6} y={PY + C} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{tr("this block's", '4 luồng')}</text>
      <text x={PX + 2 * C + 6} y={PY + C + 14} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{tr('4 threads', 'của khối này')}</text>
      {/* "fetched twice" overlay */}
      {f.times
        ? [...(f.washM ?? [])].map((i) => (
            <text key={'m' + i} x={MX + (i % W) * C + C - 6} y={MY + Math.floor(i / W) * C + 11} fontSize="9" textAnchor="end"
              opacity={f.times} style={{ fill: 'var(--fail)' }}>×2</text>
          )).concat([...(f.washN ?? [])].map((i) => (
            <text key={'n' + i} x={NX + (i % W) * C + C - 6} y={NY + Math.floor(i / W) * C + 11} fontSize="9" textAnchor="end"
              opacity={f.times} style={{ fill: 'var(--fail)' }}>×2</text>
          )))
        : null}
      {/* tile outlines */}
      {f.tiles
        ? [0, 1].map((k) => (
            <g key={k} opacity={f.tiles}>
              <rect x={MX + k * T * C - 2} y={MY - 2} width={T * C + 2} height={T * C + 2} fill="none" stroke="var(--ink)" strokeWidth="2.2" />
              <rect x={NX - 2} y={NY + k * T * C - 2} width={T * C + 2} height={T * C + 2} fill="none" stroke="var(--ink)" strokeWidth="2.2" />
            </g>
          ))
        : null}
      {/* shared memory */}
      {f.shared ? (
        <g opacity={f.shared}>
          <Tile x={SX} y={40} values={f.mds ?? [null, null, null, null]} name={tr('Mds (shared)', 'Mds (chia sẻ)')} hot={f.hotShared} />
          <Tile x={SX + 120} y={40} values={f.nds ?? [null, null, null, null]} name={tr('Nds (shared)', 'Nds (chia sẻ)')} hot={f.hotShared} />
        </g>
      ) : null}
      {f.barrier ? (
        <g>
          <rect x={SX} y={128} width={308} height={26} rx="3" fill="var(--ink)" />
          <text x={SX + 154} y={145} fontSize="12" textAnchor="middle" style={{ fill: 'var(--page)' }}>{f.barrier}</text>
        </g>
      ) : null}
      {f.sums ? (
        <g>
          <text x={SX} y={182} fontSize="12" fontWeight="600" style={{ fill: 'var(--reg)' }}>{tr('running sums (one register per thread)', 'tổng đang cộng dồn (mỗi luồng một thanh ghi)')}</text>
          {f.sums.map((s, i) => (
            <g key={i}>
              <rect x={SX + i * 74} y={190} width={68} height={30} rx="2" fill="var(--reg-wash)" stroke="var(--reg)" />
              <text x={SX + i * 74 + 34} y={210} fontSize="12.5" textAnchor="middle">{s == null ? '0' : s}</text>
            </g>
          ))}
        </g>
      ) : null}
      {f.bars != null ? (
        <g>
          <text x={SX} y={250} fontSize="11.5">{tr('fetched once per thread', 'mỗi luồng nạp một lần')}</text>
          <rect x={SX} y={256} width={300 * f.bars} height={14} fill="var(--global)" />
          <text x={SX + 300 * f.bars + 6} y={267} fontSize="11.5">{Math.round(32 * f.bars)}</text>
          <text x={SX} y={292} fontSize="11.5">{tr('fetched once per block', 'mỗi khối nạp một lần')}</text>
          <rect x={SX} y={298} width={150 * f.bars} height={14} fill="var(--shared)" />
          <text x={SX + 150 * f.bars + 6} y={309} fontSize="11.5">{Math.round(16 * f.bars)}</text>
        </g>
      ) : null}
      {f.headline ? (
        <text x={SX} y={338} fontSize="13" fontWeight="600">{f.headline}</text>
      ) : null}
    </svg>
  );
}

const idx = (r: number, c: number) => r * W + c;
const ROWS = new Set([0, 1, 2, 3, 4, 5, 6, 7]); // rows 0 and 1 of M
const COLS = new Set([0, 1, 4, 5, 8, 9, 12, 13]); // columns 0 and 1 of N
const mTile = (ph: number) => [idx(0, ph * T), idx(0, ph * T + 1), idx(1, ph * T), idx(1, ph * T + 1)];
const nTile = (ph: number) => [idx(ph * T, 0), idx(ph * T, 1), idx(ph * T + 1, 0), idx(ph * T + 1, 1)];
const partial = (upto: number) =>
  [0, 1, 2, 3].map((t) => {
    const tx = t % T;
    const ty = Math.floor(t / T);
    let s = 0;
    for (let k = 0; k < upto * T; k++) s += prob.M[idx(ty, k)] * prob.N[idx(k, tx)];
    return s;
  });

function phase(ph: number, p: number, last: boolean): Frame {
  const loading = p < 0.36;
  const waiting = p >= 0.36 && p < 0.52;
  const filled = Math.ceil(ramp(p, 0.04, 0.34) * 4);
  // A slot shows this phase's value once loaded; until then, what the previous phase left there.
  const vals = (src: number[], tileOf: (ph: number) => number[]) =>
    tileOf(ph).map((c, i) => (p >= 0.36 || i < filled ? src[c] : ph > 0 ? src[tileOf(ph - 1)[i]] : null));
  return {
    tiles: 1,
    shared: 1,
    washM: new Set(mTile(ph)),
    washN: new Set(nTile(ph)),
    hotM: loading ? new Set(mTile(ph).slice(0, filled)) : undefined,
    hotN: loading ? new Set(nTile(ph).slice(0, filled)) : undefined,
    mds: vals(prob.M, mTile),
    nds: vals(prob.N, nTile),
    hotShared: p >= 0.52 && p < 0.86,
    barrier: waiting ? tr('barrier: every thread waits for all the loads', 'rào chắn: mọi luồng chờ tất cả các lần nạp')
      : p >= 0.86
        ? last
          ? tr('last phase done: each thread stores its sum', 'xong giai đoạn cuối: mỗi luồng ghi tổng của nó')
          : tr('barrier: no overwriting until all have read', 'rào chắn: không ghi đè cho đến khi tất cả đã đọc')
        : null,
    sums: p >= 0.6 ? partial(ph + 1) : partial(ph),
    pOut: last && p >= 0.9,
  };
}

const beats: Beat[] = [
  {
    clip: 'why-tiling-1',
    get label() {
      return tr('Four threads', 'Bốn luồng');
    },
    draw: (p) => {
      const k = Math.min(W - 1, Math.floor(p * 4.4));
      const seenM = new Set<number>();
      const seenN = new Set<number>();
      for (let j = 0; j < k; j++) {
        [0, 1].forEach((r) => seenM.add(idx(r, j)));
        [0, 1].forEach((c) => seenN.add(idx(j, c)));
      }
      return (
        <Stage hotM={new Set([idx(0, k), idx(1, k)])} hotN={new Set([idx(k, 0), idx(k, 1)])} washM={seenM} washN={seenN}
          headline={tr(
            `step k = ${k}: ${8 * (k + 1)} fetches from global memory so far`,
            `bước k = ${k}: đến giờ đã có ${8 * (k + 1)} lần nạp từ bộ nhớ toàn cục`,
          )} />
      );
    },
  },
  {
    clip: 'why-tiling-2',
    get label() {
      return tr('Count the fetches', 'Đếm số lần nạp');
    },
    draw: (p) => (
      <Stage washM={ROWS} washN={COLS} times={ramp(p, 0.1, 0.3)} bars={undefined}
        headline={
          p < 0.55
            ? tr('32 fetches, 16 different elements: each one fetched twice', '32 lần nạp, 16 phần tử khác nhau: mỗi phần tử bị nạp hai lần')
            : tr('a block 32 threads wide would fetch each element 32 times', 'một khối rộng 32 luồng sẽ nạp mỗi phần tử 32 lần')
        } />
    ),
  },
  {
    clip: 'why-tiling-3',
    get label() {
      return tr('The idea', 'Ý tưởng');
    },
    draw: (p) => (
      <Stage washM={ROWS} washN={COLS} tiles={ramp(p, 0.45, 0.75)} shared={ramp(p, 0.1, 0.4)}
        headline={tr('fetch once, share inside the block, one tile at a time', 'nạp một lần, chia sẻ trong khối, mỗi lần một ô')} />
    ),
  },
  {
    clip: 'why-tiling-4',
    get label() {
      return tr('Phase 0', 'Giai đoạn 0');
    },
    draw: (p) => <Stage {...phase(0, p, false)} headline={tr('phase 0', 'giai đoạn 0')} />,
  },
  {
    clip: 'why-tiling-5',
    get label() {
      return tr('Phase 1', 'Giai đoạn 1');
    },
    draw: (p) => (
      <Stage {...phase(1, p, true)} headline={tr('phase 1: the same eight slots, the next tile', 'giai đoạn 1: vẫn tám ô nhớ đó, ô (tile) tiếp theo')} />
    ),
  },
  {
    clip: 'why-tiling-6',
    get label() {
      return tr('What changed', 'Điều gì đã thay đổi');
    },
    draw: (p) => (
      <Stage tiles={1} shared={1} mds={mTile(1).map((c) => prob.M[c])} nds={nTile(1).map((c) => prob.N[c])} sums={partial(2)} pOut
        bars={ramp(p, 0.12, 0.45)} headline={
          p > 0.5
            ? tr('tiles T wide: T times less traffic', 'ô rộng T: lưu lượng giảm T lần')
            : tr('same answer, half the fetches', 'cùng kết quả, một nửa số lần nạp')
        } />
    ),
  },
];

export function WhyTilingScene() {
  return <Scene chapter="ch05" title={tr('Why tiling works', 'Vì sao phân ô hiệu quả')} beats={beats} />;
}
