import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { intensity, regionBlocks, regionCounts, tilePair } from '../../sim/gemmreuse';

// "One wide block instead of four narrow ones" (section 15.2, pp. 351–352). A 128 x 128
// corner of C, first as sixteen 32 x 32 blocks, then as one 128 x 128 block. Every count
// drawn comes from src/sim/gemmreuse.ts. Five beats, one narration clip each
// (content/ch15/narration/reuse-N.txt).

const REGION = 128;
const NARROW = 32;
const K_STEP = 8; // one step along the inner dimension: the book's bK
const S = 1.5; // px per element of C
const CX = 84;
const CY = 112;
const SIDE = REGION * S; // 192
const STRIP = 30; // drawn thickness of the A and B strips (k is not to scale)
const RX = 330; // right-hand panel

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const fmtInt = (v: number) => v.toLocaleString('en-US', { maximumFractionDigits: 0 });
const per = REGION / NARROW;
const BLOCKS = regionBlocks(REGION, NARROW);

interface Frame {
  /** Blocks that have fetched so far (row-major), or 'merged' for the single wide block. */
  upto?: number;
  merged?: number; // 0..1: how far the narrow outlines have faded into one block
  /** The block lighting up right now. */
  active?: number;
  /** Emphasise one A band (row) or one B band (column). */
  rowBand?: number;
  colBand?: number;
  headline: ReactNode;
  right?: ReactNode;
}

function Corner({ f }: { f: Frame }) {
  const act = f.active != null ? BLOCKS[f.active] : undefined;
  const t = NARROW * S;
  return (
    <g>
      <text x={CX} y={CY - STRIP - 12} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}>
        {(f.merged ?? 0) >= 1 ? tr('B: one band of 128 columns', 'B: một dải 128 cột') : tr('B: four bands of 32 columns', 'B: bốn dải 32 cột')}
      </text>
      <text x={CX - STRIP - 12} y={CY + SIDE / 2} fontSize="12" fontWeight="600" textAnchor="middle"
        transform={`rotate(-90 ${CX - STRIP - 12} ${CY + SIDE / 2})`} style={{ fill: 'var(--global)' }}>
        {(f.merged ?? 0) >= 1 ? tr('A: one band of 128 rows', 'A: một dải 128 hàng') : tr('A: four bands of 32 rows', 'A: bốn dải 32 hàng')}
      </text>
      {/* B bands above C, A bands left of C */}
      {Array.from({ length: per }, (_, i) => {
        const hotB = act?.bBand === i || f.colBand === i;
        const hotA = act?.aBand === i || f.rowBand === i;
        return (
          <g key={i}>
            <rect x={CX + i * t} y={CY - STRIP - 4} width={t - 2} height={STRIP} rx="2"
              fill={hotB ? 'var(--global)' : 'var(--global-wash)'} stroke="var(--global)" data-band={'B' + i} data-hot={hotB ? 1 : 0} />
            <rect x={CX - STRIP - 4} y={CY + i * t} width={STRIP} height={t - 2} rx="2"
              fill={hotA ? 'var(--global)' : 'var(--global-wash)'} stroke="var(--global)" data-band={'A' + i} data-hot={hotA ? 1 : 0} />
          </g>
        );
      })}
      {/* C: the corner of the output, in registers */}
      <rect x={CX} y={CY} width={SIDE - 2} height={SIDE - 2} fill="var(--reg-wash)" stroke="var(--reg)" />
      {BLOCKS.map((b) => {
        const done = f.upto != null && b.index < f.upto;
        const hot = f.active === b.index;
        return (
          <rect key={b.index} x={CX + b.col * t + 2} y={CY + b.row * t + 2} width={t - 6} height={t - 6} rx="2"
            fill={hot ? 'var(--reg)' : done ? 'var(--reg-wash)' : 'none'}
            stroke="var(--reg)" strokeDasharray={done || hot ? undefined : '3 3'}
            opacity={1 - (f.merged ?? 0)} data-block={b.index} />
        );
      })}
      {f.merged ? (
        <rect x={CX - 1} y={CY - 1} width={SIDE} height={SIDE} rx="3" fill="none" stroke="var(--reg)" strokeWidth="3" opacity={f.merged} data-merged="1" />
      ) : null}
      <text x={CX + SIDE / 2} y={CY + SIDE / 2 + 4} fontSize="12" textAnchor="middle" opacity={f.merged ? f.merged : 0}>{tr('one 128 × 128 block', 'một khối 128 × 128')}</text>
    </g>
  );
}

function Counters({ upto, merged }: { upto: number; merged: boolean }) {
  const c = merged ? regionCounts(REGION, REGION, K_STEP) : regionCounts(REGION, NARROW, K_STEP, upto);
  const narrowAll = regionCounts(REGION, NARROW, K_STEP);
  const complete = merged || upto >= BLOCKS.length;
  const W = 330;
  const bar = (v: number) => (W * v) / narrowAll.requested;
  const box = (label: string, counts: number[], y: number, key: string) => (
    <g>
      <text x={RX} y={y} fontSize="12">{label}</text>
      {counts.map((n, i) => (
        <g key={i} data-counter={key + i}>
          <rect x={RX + i * (merged ? 120 : 60)} y={y + 6} width={merged ? 112 : 52} height={28} rx="2" fill="var(--global-wash)" stroke="var(--global)" />
          <text x={RX + i * (merged ? 120 : 60) + (merged ? 56 : 26)} y={y + 25} fontSize="13" textAnchor="middle" data-value>{n}</text>
        </g>
      ))}
    </g>
  );
  return (
    <g>
      {box(merged ? tr('Fetches of the A band (all 128 rows)', 'Số lần nạp dải A (cả 128 hàng)') : tr('Fetches of each band of A', 'Số lần nạp mỗi dải của A'), c.aBand, 30, 'A')}
      {box(merged ? tr('Fetches of the B band (all 128 columns)', 'Số lần nạp dải B (cả 128 cột)') : tr('Fetches of each band of B', 'Số lần nạp mỗi dải của B'), c.bBand, 92, 'B')}
      <text x={RX} y={162} fontSize="12"><Translated vi={<>Số byte được yêu cầu cho một bước k = {K_STEP}</>}>Bytes requested for one step of k = {K_STEP}</Translated></text>
      <rect x={RX} y={170} width={Math.max(1, bar(c.requested))} height={16} fill="var(--global)" />
      <text x={RX + Math.max(1, bar(c.requested)) + 6} y={183} fontSize="12" data-requested={c.requested}>{fmtInt(c.requested)}</text>
      <text x={RX} y={210} fontSize="12">{tr('Bytes the data needs (each band once)', 'Số byte dữ liệu thực sự cần (mỗi dải một lần)')}</text>
      <rect x={RX} y={218} width={bar(c.needed)} height={16} fill="var(--shared)" />
      <text x={RX + bar(c.needed) + 6} y={231} fontSize="12" data-needed={c.needed}>{fmtInt(c.needed)}</text>
      <text x={RX} y={262} fontSize="12.5" fontWeight="600" data-ratio={c.needed ? c.requested / c.needed : 0}>
        {complete ? tr(`requested ÷ needed = ${+(c.requested / c.needed).toFixed(2)}`, `yêu cầu ÷ cần = ${+(c.requested / c.needed).toFixed(2)}`) : tr(`${upto} of ${BLOCKS.length} blocks ${upto === 1 ? 'has' : 'have'} fetched`, `${upto} trên ${BLOCKS.length} khối đã nạp`)}
      </text>
    </g>
  );
}

function Stage({ f }: { f: Frame }) {
  return (
    <svg viewBox="0 0 720 340" role="img" aria-label={tr('Blocks of C fetching bands of A and B', 'Các khối của C nạp các dải của A và B')} style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      <Corner f={f} />
      {f.right}
      <text x={20} y={330} fontSize="13" fontWeight="600" data-headline>{f.headline}</text>
    </svg>
  );
}

// ---------------------------------------------------------------- beats 4 and 5: the ratio
function RatioPanel({ m, n, k, note }: { m: number; n: number; k: number; note?: ReactNode }) {
  const t = tilePair(m, n, k);
  const max = tilePair(128, 128, 32);
  const W = 330;
  return (
    <g data-m={m} data-n={n} data-k={k}>
      <text x={RX} y={30} fontSize="12"><Translated vi={<>Một cặp ô: m = {m}, n = {fmtInt(n)}, k = {k}</>}>One tile pair: m = {m}, n = {fmtInt(n)}, k = {k}</Translated></text>
      <text x={RX} y={62} fontSize="12">{tr('operations, 2·m·n·k', 'số phép toán, 2·m·n·k')}</text>
      <rect x={RX} y={70} width={Math.max(1, (W * t.ops) / max.ops)} height={16} fill="var(--reg)" />
      <text x={RX + Math.min(W, Math.max(1, (W * t.ops) / max.ops)) + 6} y={83} fontSize="12" data-ops={t.ops}>{fmtInt(t.ops)}</text>
      <text x={RX} y={112} fontSize="12">{tr('bytes, 4·k·(m + n)', 'số byte, 4·k·(m + n)')}</text>
      <rect x={RX} y={120} width={Math.min(W, Math.max(1, (W * t.traffic) / max.traffic))} height={16} fill="var(--global)" />
      <text x={RX + Math.min(W, Math.max(1, (W * t.traffic) / max.traffic)) + 6} y={133} fontSize="12" data-bytes={t.traffic}>{fmtInt(t.traffic)}</text>
      <text x={RX} y={180} fontSize="13">{tr('operations ÷ bytes', 'phép toán ÷ byte')}</text>
      {/* Truncated, never rounded up: 0.49988 must not read as 0.500. */}
      <text x={RX} y={214} fontSize="28" fontWeight="650" data-intensity={t.intensity}>
        {t.intensity >= 10 ? t.intensity.toFixed(0) : (Math.floor(t.intensity * 1e4) / 1e4).toFixed(4)} FLOP/B
      </text>
      {note ? <text x={RX} y={246} fontSize="12" style={{ fill: 'var(--ink-2)' }}>{note}</text> : null}
    </g>
  );
}

function Tile({ m, n, k, wideNote }: { m: number; n: number; k: number; wideNote?: boolean }) {
  const h = Math.max(2, m * S);
  const w = SIDE;
  const th = Math.max(4, k * 1.1); // strip thickness grows with k (not to scale)
  return (
    <g>
      <rect x={CX - th - 4} y={CY} width={th} height={h} fill="var(--shared-wash)" stroke="var(--shared)" />
      <rect x={CX} y={CY - th - 4} width={w - 2} height={th} fill="var(--shared-wash)" stroke="var(--shared)" />
      <rect x={CX} y={CY} width={w - 2} height={h} fill="var(--reg-wash)" stroke="var(--reg)" />
      <text x={CX - th - 6} y={CY - th - 10} fontSize="12" style={{ fill: 'var(--shared)' }}>{tr('tiles of A and B, k deep', 'các ô của A và B, sâu k')}</text>
      <text x={CX} y={CY + h + 18} fontSize="12" style={{ fill: 'var(--reg)' }}><Translated vi={<>ô đầu ra, {m} × {fmtInt(n)}</>}>output tile, {m} × {fmtInt(n)}</Translated></text>
      {wideNote ? <text x={CX} y={CY + h + 33} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{tr('width not to scale', 'chiều rộng không theo tỉ lệ')}</text> : null}
    </g>
  );
}

const beats: Beat[] = [
  {
    clip: 'reuse-1',
    get label() {
      return tr('Along a row', 'Dọc một hàng');
    },
    draw: (p) => {
      const upto = Math.max(1, Math.min(per, Math.floor(p * (per + 0.6)) + 1));
      return (
        <Stage f={{
          upto, active: upto - 1, rowBand: 0,
          headline: tr(`${upto} of the 4 blocks in this row ${upto === 1 ? 'has' : 'have'} fetched the same band of A`, `${upto} trên 4 khối của hàng này đã nạp cùng một dải của A`),
          right: <Counters upto={upto} merged={false} />,
        }} />
      );
    },
  },
  {
    clip: 'reuse-2',
    get label() {
      return tr('Down a column', 'Xuống một cột');
    },
    draw: (p) => {
      const upto = Math.max(per + 1, Math.min(BLOCKS.length, per + Math.floor(ramp(p, 0, 0.7) * (BLOCKS.length - per + 0.5)) + 1));
      return (
        <Stage f={{
          upto, active: upto < BLOCKS.length ? upto - 1 : undefined, colBand: 0,
          headline: upto < BLOCKS.length ? tr('every row repeats it, and so does every column for B', 'hàng nào cũng lặp lại điều đó, và cột nào cũng vậy với B') : tr('16 blocks: every band fetched 4 times', '16 khối: mỗi dải bị nạp 4 lần'),
          right: <Counters upto={upto} merged={false} />,
        }} />
      );
    },
  },
  {
    clip: 'reuse-3',
    get label() {
      return tr('One wide block', 'Một khối rộng');
    },
    draw: (p) => {
      const m = ramp(p, 0.08, 0.4);
      const merged = m >= 1;
      return (
        <Stage f={{
          upto: BLOCKS.length, merged: m,
          headline: merged ? tr('each band fetched once: a quarter of the bytes', 'mỗi dải chỉ nạp một lần: một phần tư số byte') : tr('merging 16 narrow blocks into one', 'gộp 16 khối hẹp thành một'),
          right: <Counters upto={BLOCKS.length} merged={merged} />,
        }} />
      );
    },
  },
  {
    clip: 'reuse-4',
    get label() {
      return tr('k cancels', 'k triệt tiêu');
    },
    draw: (p) => {
      const k = Math.round(8 + 24 * ramp(p, 0.15, 0.85));
      return (
        <svg viewBox="0 0 720 340" role="img" aria-label={tr('Operations and bytes of one tile pair as k grows', 'Số phép toán và số byte của một cặp ô khi k tăng')} style={{ width: '100%', maxWidth: 720, display: 'block' }}>
          <Tile m={128} n={128} k={k} />
          <RatioPanel m={128} n={128} k={k} note={tr('both bars grow by the same factor', 'cả hai thanh tăng cùng một hệ số')} />
          <text x={20} y={330} fontSize="13" fontWeight="600" data-headline><Translated vi={<>k = {k}: tỉ số không đổi</>}>k = {k}: the ratio does not move</Translated></text>
        </svg>
      );
    },
  },
  {
    clip: 'reuse-5',
    get label() {
      return tr('One row', 'Một hàng');
    },
    draw: (p) => {
      // First half: the tile height falls from 128 to 1. Second half: the width grows from 128 to 4096.
      const a = ramp(p, 0.05, 0.5);
      const b = ramp(p, 0.55, 0.95);
      const m = Math.max(1, Math.round(128 * Math.pow(1 / 128, a)));
      const n = Math.round(128 * Math.pow(32, b));
      return (
        <svg viewBox="0 0 720 340" role="img" aria-label={tr('The ratio of a one-row output tile', 'Tỉ số của một ô đầu ra chỉ có một hàng')} style={{ width: '100%', maxWidth: 720, display: 'block' }}>
          <Tile m={m} n={n} k={8} wideNote={n > 128} />
          <RatioPanel m={m} n={n} k={8} note={m === 1 && intensity(1, n) < 0.5 ? tr('below 0.5 for any width (our derivation)', 'dưới 0.5 với mọi chiều rộng (suy luận của khóa học này)') : undefined} />
          <text x={20} y={330} fontSize="13" fontWeight="600" data-headline>
            {m === 1 ? tr('one row of output: the whole band of B for one row', 'một hàng đầu ra: cả dải B cho chỉ một hàng') : tr('the tile loses rows', 'ô mất dần hàng')}
          </text>
        </svg>
      );
    },
  },
];

export function ReuseScene() {
  return <Scene chapter="ch15" title={tr('One wide block instead of four narrow ones', 'Một khối rộng thay cho bốn khối hẹp')} beats={beats} />;
}
