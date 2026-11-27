import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { tileLoad } from '../../sim/coalesce';
import { tileAccess, type BankResult } from '../../sim/banks';
import { Translated, tr } from '../../lib/i18n';

// "Turn the corner, then pad": the corner-turned tile load of section 6.1 fixes the
// global reads and creates a 32-way bank conflict on the shared-memory store; one
// padding column removes it (book pp. 129–130, 139–141). Global sectors come from
// src/sim/coalesce.ts (tileLoad), bank passes from src/sim/banks.ts (tileAccess).
// Six beats, one narration clip each (content/ch06/narration/turn-and-pad-N.txt).
const T = 32;
const WIDTH = 1024;
const load = (arrangement: 'ch5' | 'turned') =>
  tileLoad({ layout: 'col', arrangement, T, bx: 0, ph: 0, Width: WIDTH, warpWidth: 32 }).warps[0];
const store = (pitch: number, warpWidth = 32, banks = 32) => tileAccess({ T, pitch, orientation: 'x-first', ty0: 0, warpWidth, banks });

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

const C = 4; // cell size of the tiles
const NX = 24, NY = 56; // N tile
const SX = 520, SY = 56; // shared tile
const BY = 236; // bank header

function NTile({ arrangement }: { arrangement: 'ch5' | 'turned' | null }) {
  const w = arrangement ? load(arrangement) : null;
  const on = new Set(w?.threads.map((t) => `${t.sharedRow},${t.sharedCol}`) ?? []);
  return (
    <g data-arrangement={arrangement ?? ''} data-sectors={w ? w.access.transactions : ''}>
      <text x={NX} y={NY - 10} fontSize="11" style={{ fill: 'var(--global)' }}><Translated vi="ô của N (theo từng cột)">tile of N (column by column)</Translated></text>
      {Array.from({ length: T * T }, (_, i) => {
        const r = Math.floor(i / T);
        const c = i % T;
        return <rect key={i} x={NX + c * C} y={NY + r * C} width={C - 0.6} height={C - 0.6} fill={on.has(`${r},${c}`) ? 'var(--global)' : 'var(--global-wash)'} />;
      })}
      {w ? (
        <g>
          <text x={NX + T * C + 18} y={NY + 14} fontSize="11.5"><Translated vi={<>một warp: {w.access.transactions} sector</>}>one warp: {w.access.transactions} sectors</Translated></text>
          {w.access.segments.map((seg, i) => {
            const used = w.threads.filter((t) => Math.floor(t.byte / 32) === seg).length * 4 / 32;
            const bw = arrangement === 'turned' ? 18 : 7;
            const x = NX + T * C + 18 + i * (bw + 2);
            return (
              <g key={seg}>
                <rect x={x} y={NY + 24} width={bw} height={16} fill="var(--global-wash)" />
                <rect x={x} y={NY + 24} width={bw * used} height={16} fill="var(--global)" />
                <rect x={x + 0.4} y={NY + 24.4} width={bw - 0.8} height={15.2} fill="none" stroke="var(--ink-2)" strokeWidth="0.7" />
              </g>
            );
          })}
        </g>
      ) : null}
    </g>
  );
}

function SharedTile({ pitch, column, padHot = false }: { pitch: number; column: boolean; padHot?: boolean }) {
  return (
    <g data-pitch={pitch}>
      <text x={SX} y={SY - 10} fontSize="11" style={{ fill: 'var(--shared)' }}><Translated vi={<>ô trong bộ nhớ chia sẻ, {T} × {pitch} số float</>}>shared tile, {T} × {pitch} floats</Translated></text>
      {Array.from({ length: T * pitch }, (_, i) => {
        const r = Math.floor(i / pitch);
        const c = i % pitch;
        const pad = c >= T;
        return (
          <rect key={i} x={SX + c * C} y={SY + r * C} width={C - 0.6} height={C - 0.6}
            fill={pad ? (padHot ? 'var(--fail-wash)' : 'transparent') : column && c === 0 ? 'var(--shared)' : 'var(--shared-wash)'}
            stroke={pad ? (padHot ? 'var(--fail)' : 'var(--rule)') : 'none'} strokeWidth="0.5" />
        );
      })}
    </g>
  );
}

/** Lanes dropped into bank columns, one dot per lane, stacked by the pass that serves it. */
function Banks({ r, banks, served, label }: { r: BankResult; banks: number; served: number; label: ReactNode }) {
  const cw = 640 / banks;
  const x0 = 40;
  // pass index of each lane: its word's position in the bank's queue
  const passOf = new Map<number, number>();
  for (const b of r.perBank) for (const l of b.lanes) passOf.set(l, b.words.indexOf(r.lanes[l].word));
  return (
    <g data-passes={r.passes} data-banks-used={r.banksUsed} data-banks={banks} data-served={served}>
      <text x={4} y={BY - 14} fontSize="11" style={{ fill: 'var(--shared)' }}>{label}</text>
      {Array.from({ length: banks }, (_, b) => (
        <g key={b}>
          <rect x={x0 + b * cw} y={BY - 6} width={cw - 1} height={4} fill={r.perBank[b].lanes.length ? 'var(--shared)' : 'var(--rule)'} />
          {banks <= 32 || b % 8 === 0 ? <text x={x0 + b * cw + cw / 2} y={BY + 8} fontSize="7.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{b}</text> : null}
        </g>
      ))}
      {r.lanes.map((l) => {
        const pass = passOf.get(l.lane) ?? 0;
        const gone = pass < served;
        return (
          <circle key={l.lane} cx={x0 + l.bank * cw + cw / 2} cy={BY + 16 + pass * 3.8}
            r={Math.min(1.9, cw / 2.6)} fill={r.perBank[l.bank].words.length > 1 ? 'var(--fail)' : 'var(--shared)'} opacity={gone ? 0.18 : 1} />
        );
      })}
    </g>
  );
}

interface Frame {
  arrangement: 'ch5' | 'turned' | null;
  pitch?: number;
  column?: boolean;
  padHot?: boolean;
  bank?: { r: BankResult; banks: number; served: number; label: ReactNode };
  headline: ReactNode;
  lines?: ReactNode[];
  /** Where the lines start (default: beside the N tile). */
  linesAt?: { x: number; y: number };
}

function Stage(f: Frame) {
  return (
    <svg viewBox="0 0 720 380" role="img" aria-label={tr('Animation of a corner-turned tile load and the bank conflict it causes', 'Hoạt cảnh một lần nạp ô đã xoay góc và xung đột bank mà nó gây ra')} style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      <text x={4} y={16} fontSize="13" fontWeight="600" data-headline>{f.headline}</text>
      <NTile arrangement={f.arrangement} />
      {f.pitch ? <SharedTile pitch={f.pitch} column={!!f.column} padHot={f.padHot} /> : null}
      {f.bank ? <Banks {...f.bank} /> : null}
      {f.lines
        ? f.lines.map((t, i) => (
            <text key={i} x={f.linesAt?.x ?? NX + T * C + 18} y={(f.linesAt?.y ?? NY + 74) + i * 18} fontSize="11.5" data-line>{t}</text>
          ))
        : null}
    </svg>
  );
}

const served = (p: number, a: number, b: number, passes: number) => Math.floor(ramp(p, a, b) * passes);

const beats: Beat[] = [
  {
    clip: 'turn-and-pad-1',
    get label() {
      return tr('Top edge', 'Cạnh trên');
    },
    draw: (p) => (
      <Stage arrangement={p < 0.2 ? null : 'ch5'}
        headline={p < 0.2 ? tr('a 32 × 32 tile of N, loaded with chapter 5\'s mapping', 'một ô 32 × 32 của N, nạp theo cách ánh xạ của chương 5') : tr(`one warp reads the top edge: ${load('ch5').access.transactions} sectors, one per lane`, `một warp đọc cạnh trên: ${load('ch5').access.transactions} sector, mỗi lane một sector`)} />
    ),
  },
  {
    clip: 'turn-and-pad-2',
    get label() {
      return tr('Turned', 'Đã xoay góc');
    },
    draw: (p) => (
      <Stage arrangement="turned" pitch={T} column={p > 0.55} /* i18n-ignore: arrangement key */
        headline={p < 0.55 ? tr(`corner turned: the warp reads the left edge, ${load('turned').access.transactions} sectors`, `đã xoay góc: warp đọc cạnh trái, ${load('turned').access.transactions} sector`) : tr('thread x writes row x of the shared tile: the warp writes a column', 'luồng x ghi hàng x của ô trong bộ nhớ chia sẻ: warp ghi một cột')} />
    ),
  },
  {
    clip: 'turn-and-pad-3',
    get label() {
      return tr('32 passes', '32 lượt');
    },
    draw: (p) => {
      const r = store(T);
      const s = served(p, 0.35, 0.95, r.passes);
      return (
        <Stage arrangement="turned" pitch={T} column /* i18n-ignore: arrangement key */
          bank={{ r, banks: 32, served: s, label: tr(`32 banks: row length ${T} words, every lane in bank ${r.lanes[0].bank}`, `32 bank: độ dài hàng ${T} word, mọi lane đều ở bank ${r.lanes[0].bank}`) }}
          headline={p < 0.35 ? tr('every word of the column is a multiple of 32: one bank', 'mọi word của cột đều là bội của 32: một bank') : tr(`pass ${Math.max(1, Math.min(r.passes, s + 1))} of ${r.passes}; ${32 - r.banksUsed} banks idle`, `lượt ${Math.max(1, Math.min(r.passes, s + 1))} trên ${r.passes}; ${32 - r.banksUsed} bank rảnh`)} />
      );
    },
  },
  {
    clip: 'turn-and-pad-4',
    get label() {
      return tr('One column more', 'Thêm một cột');
    },
    draw: (p) => {
      const r = store(T + 1);
      return (
        <Stage arrangement="turned" pitch={T + 1} column /* i18n-ignore: arrangement key */
          bank={{ r, banks: 32, served: 0, label: tr(`32 banks: row length ${T + 1} words, lane i in bank i`, `32 bank: độ dài hàng ${T + 1} word, lane i ở bank i`) }}
          headline={p < 0.3 ? tr('pad each row to 33 words', 'đệm mỗi hàng thành 33 word') : tr(`${r.banksUsed} banks, ${r.passes} pass`, `${r.banksUsed} bank, ${r.passes} lượt`)} />
      );
    },
  },
  {
    clip: 'turn-and-pad-5',
    get label() {
      return tr('The price', 'Cái giá');
    },
    draw: (p) => (
      <Stage arrangement="turned" pitch={T + 1} column padHot /* i18n-ignore: arrangement key */
        headline={tr('the price of the padding', 'cái giá của phần đệm')} linesAt={{ x: 40, y: BY }}
        lines={[
          tr(`unused: ${T} floats × 4 bytes = ${T * 4} bytes per tile`, `không dùng: ${T} số float × 4 byte = ${T * 4} byte mỗi ô`),
          ...(p > 0.3 ? [tr('row i at 32 words: i × 32 = i << 5, one shift', 'hàng i với 32 word: i × 32 = i << 5, một phép dịch')] : []),
          ...(p > 0.45 ? [tr('row i at 33 words: i × 33 = (i << 5) + i, a shift and an add', 'hàng i với 33 word: i × 33 = (i << 5) + i, một phép dịch và một phép cộng')] : []),
          ...(p > 0.65 ? [tr('the book prints i >> 5 + i, which C reads as i >> (5 + i)', 'sách in i >> 5 + i, mà C đọc thành i >> (5 + i)')] : []),
        ]} />
    ),
  },
  {
    clip: 'turn-and-pad-6',
    get label() {
      return tr('Other GPUs', 'Các GPU khác');
    },
    draw: (p) => {
      const padded = p > 0.4;
      const r = store(padded ? T + 1 : T, 64, 64);
      return (
        <Stage arrangement="turned" pitch={padded ? T + 1 : T} column /* i18n-ignore: arrangement key */
          bank={{ r, banks: 64, served: 0, label: tr(`64 banks, 64 lanes (AMD CDNA 4), row length ${padded ? T + 1 : T}: ${r.passes} passes by NVIDIA's arithmetic`, `64 bank, 64 lane (AMD CDNA 4), độ dài hàng ${padded ? T + 1 : T}: ${r.passes} lượt theo cách tính của NVIDIA`) }}
          headline={p < 0.7 ? tr('the same store with 64 banks and 64 lanes', 'cùng lần ghi đó với 64 bank và 64 lane') : tr('AMD serves 16 lanes per cycle; Apple documents no banks: measure (lab 6.3)', 'AMD phục vụ 16 lane mỗi chu kỳ; Apple không công bố gì về bank: hãy đo (lab 6.3)')}
          lines={p > 0.7 ? [tr('NVIDIA: 32 banks, documented rule', 'NVIDIA: 32 bank, quy tắc có tài liệu'), tr('AMD: 16 lanes per cycle, counts differ', 'AMD: 16 lane mỗi chu kỳ, số đếm khác đi'), tr('Apple: nothing documented', 'Apple: không có tài liệu nào')] : undefined} />
      );
    },
  },
];

export function TurnAndPadScene() {
  return <Scene chapter="ch06" title={tr('Turn the corner, then pad', 'Xoay góc, rồi đệm')} beats={beats} />;
}
