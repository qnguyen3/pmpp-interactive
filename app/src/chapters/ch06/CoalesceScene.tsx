import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { exprAccess, tileLoad, type AccessParams, type ExprId } from '../../sim/coalesce';
import { Translated, tr } from '../../lib/i18n';

// "One instruction, thirty-two addresses": one warp executing the N load of the
// basic matmul kernel (book Figs. 6.2 and 6.3) with NVIDIA's 32-byte sectors, then a
// preview of corner turning (Fig. 6.4). Every address, sector and counter is computed
// by src/sim/coalesce.ts. Six beats, one narration clip each
// (content/ch06/narration/coalesce-N.txt).
const WIDTH = 1024;
const LANES = 32;
const SECTOR = 32;
const access = (expr: ExprId, k: number) => {
  const p: AccessParams = { Width: WIDTH, k, row: 0, stride: 1, offset: 0, warpWidth: LANES, lane0: 0 };
  return exprAccess(expr, p, SECTOR);
};
const turnLoad = (arrangement: 'ch5' | 'turned') =>
  tileLoad({ layout: 'col', arrangement, T: 32, bx: 0, ph: 0, Width: WIDTH, warpWidth: LANES }).warps[0];

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const kAt = (p: number) => Math.min(7, Math.floor(p * 8.6));

// Geometry.
const LX = (l: number) => 40 + l * 20; // left edge of lane l's box
const LY = 34;
const SY = 150; // strip
const SW = 18; // one 32-byte sector, drawn the same width everywhere
const ROW_X0 = LX(16) - 1 - 2 * SW; // four adjacent sectors centred under the warp

interface Frame {
  layout: 'row' | 'col';
  k: number;
  lanesShown: number; // how many lanes have computed their address
  sectors: number; // 0..1, how far the grouping has gone
  counters?: { fetched: number; needed: number; max: number; steps: number };
  headline: ReactNode;
  note?: ReactNode;
}

function Strip({ f }: { f: Frame }) {
  const a = access(f.layout === 'row' ? 'n-row' : 'n-col', f.k);
  const base = a.segments[0] * SECTOR;
  // Where lane l's float is drawn, and the box of the sector it lands in.
  const target = (l: number) => {
    const byte = a.lanes[l].byte;
    if (f.layout === 'row') return ROW_X0 + ((byte - base) / SECTOR) * SW + SW / 16;
    return LX(l) + ((byte % SECTOR) / SECTOR) * SW + SW / 16;
  };
  const boxX = (i: number) => (f.layout === 'row' ? ROW_X0 + i * SW : LX(i));
  const usedFrac = (seg: number) => {
    const lo = seg * SECTOR;
    return a.lanes.filter((x) => x.byte >= lo && x.byte < lo + SECTOR).length * 4 / SECTOR;
  };
  return (
    <g data-layout={f.layout} data-k={f.k} data-transactions={f.sectors > 0.5 ? a.transactions : ''}>
      {/* lanes */}
      <text x={4} y={LY + 13} fontSize="10" style={{ fill: 'var(--ink-2)' }}><Translated vi="các lane">lanes</Translated></text>
      {Array.from({ length: LANES }, (_, l) => (
        <g key={l} opacity={l < f.lanesShown ? 1 : 0.25}>
          <rect x={LX(l)} y={LY} width={18} height={18} rx="2" fill="var(--cell)" stroke="var(--ink-3)" />
          <text x={LX(l) + 9} y={LY + 12.5} fontSize="8" textAnchor="middle">{l}</text>
        </g>
      ))}
      {/* one arrow per lane that has its address */}
      {Array.from({ length: Math.min(f.lanesShown, LANES) }, (_, l) => (
        <line key={l} x1={LX(l) + 9} y1={LY + 19} x2={target(l)} y2={SY - 2} stroke="var(--global)" strokeWidth="0.9" opacity="0.8" />
      ))}
      {/* the strip: before grouping, a plain run of bytes; after, the sectors */}
      {f.layout === 'row' && f.sectors === 0 ? (
        <rect x={ROW_X0} y={SY} width={4 * SW} height={22} fill="var(--global-wash)" stroke="var(--global)" strokeWidth="0.8" />
      ) : null}
      {f.sectors > 0
        ? a.segments.map((seg, i) => (
            <g key={seg} data-sector={seg} opacity={ramp(f.sectors, i / a.segments.length / 2, i / a.segments.length / 2 + 0.5)}>
              <rect x={boxX(i)} y={SY} width={SW} height={22} fill="var(--global-wash)" />
              {f.layout === 'row' ? (
                <rect x={boxX(i)} y={SY} width={SW * usedFrac(seg)} height={22} fill="var(--global)" />
              ) : (
                <rect x={LX(i) + ((a.lanes[i].byte % SECTOR) / SECTOR) * SW} y={SY - 4} width={SW * usedFrac(seg)} height={30} fill="var(--global)" />
              )}
              <rect x={boxX(i) + 0.5} y={SY + 0.5} width={SW - 1} height={21} fill="none" stroke="var(--ink-2)" strokeWidth="0.8" />
            </g>
          ))
        : null}
      {f.layout === 'row' && f.sectors > 0.7 ? (
        <g>
          <path d={`M${ROW_X0} ${SY + 28} v5 h${4 * SW} v-5`} fill="none" stroke="var(--ink-2)" />
          <text x={ROW_X0 + 2 * SW} y={SY + 46} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}><Translated vi="một dòng 128 byte = 4 sector">one 128-byte line = 4 sectors</Translated></text>
        </g>
      ) : null}
      {f.layout === 'col' && f.sectors > 0.7 ? (
        <text x={LX(0)} y={SY + 40} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
          <Translated vi={<>mỗi lane một sector 32 byte; hai lane kề nhau cách nhau {(a.lanes[1].byte - a.lanes[0].byte).toLocaleString('en-US')} byte (khoảng cách không vẽ theo tỉ lệ)</>}>
            one 32-byte sector per lane; neighbours are {(a.lanes[1].byte - a.lanes[0].byte).toLocaleString('en-US')} bytes apart (gaps not drawn to scale)
          </Translated>
        </text>
      ) : null}
      <text x={4} y={SY + 15} fontSize="10" style={{ fill: 'var(--ink-2)' }}>N</text>
      <text x={LX(0)} y={SY - 8} fontSize="10" style={{ fill: 'var(--ink-2)' }}>
        {f.layout === 'row' ? tr(`N row by row: bytes ${base.toLocaleString('en-US')} to ${(base + 127).toLocaleString('en-US')}`, `N theo từng hàng: byte ${base.toLocaleString('en-US')} đến ${(base + 127).toLocaleString('en-US')}`) : tr('N column by column', 'N theo từng cột')}
      </text>
    </g>
  );
}

function Counters({ c, transactions }: { c: NonNullable<Frame['counters']>; transactions: number }) {
  const W = 470;
  const bar = (v: number) => Math.max(1, (W * v) / c.max);
  return (
    <g>
      <g data-counter="transactions this step">
        <text x={40} y={238} fontSize="11.5"><Translated vi="giao dịch ở bước này">transactions this step</Translated></text>
        <text x={210} y={238} fontSize="11.5" fontWeight="650" data-value>{transactions}</text>
      </g>
      <g data-counter="bytes fetched">
        <text x={40} y={268} fontSize="11.5"><Translated vi="số byte đã nạp">bytes fetched</Translated></text>
        <rect x={210} y={258} width={bar(c.fetched)} height={14} fill="var(--global)" />
        <text x={216 + bar(c.fetched)} y={269} fontSize="11.5" data-value>{c.fetched.toLocaleString('en-US')}</text>
      </g>
      <g data-counter="bytes needed">
        <text x={40} y={296} fontSize="11.5"><Translated vi="số byte warp cần">bytes the warp needed</Translated></text>
        <rect x={210} y={286} width={bar(c.needed)} height={14} fill="var(--global-wash)" stroke="var(--global)" />
        <text x={216 + bar(c.needed)} y={297} fontSize="11.5" data-value>{c.needed.toLocaleString('en-US')}</text>
      </g>
      <text x={40} y={322} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi={<>tổng qua {c.steps} bước của k, mỗi lệnh được tính lại từ đầu (không có cache)</>}>
          totals over {c.steps} step{c.steps === 1 ? '' : 's'} of k, counting every instruction afresh (no cache)
        </Translated>
      </text>
    </g>
  );
}

function Stage(f: Frame) {
  const a = access(f.layout === 'row' ? 'n-row' : 'n-col', f.k);
  return (
    <svg viewBox="0 0 720 360" role="img" aria-label={tr("Animation of one warp's load and the memory transactions it needs", 'Hoạt cảnh một lần nạp của một warp và các giao dịch bộ nhớ nó cần')} style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      <text x={4} y={16} fontSize="13" fontWeight="600" data-headline>{f.headline}</text>
      <Strip f={f} />
      {f.counters ? <Counters c={f.counters} transactions={f.sectors > 0.5 ? a.transactions : 0} /> : null}
      {f.note ? <text x={40} y={348} fontSize="11" style={{ fill: 'var(--ink-2)' }} data-note>{f.note}</text> : null}
    </svg>
  );
}

/** Totals over steps 0..k, every instruction counted afresh. */
const totals = (layout: 'row' | 'col', k: number, max: number) => {
  let fetched = 0;
  let needed = 0;
  for (let j = 0; j <= k; j++) {
    const a = access(layout === 'row' ? 'n-row' : 'n-col', j);
    fetched += a.bytesFetched;
    needed += a.bytesNeeded;
  }
  return { fetched, needed, max, steps: k + 1 };
};

// The corner-turning preview: a 32 × 32 tile of column-major N, its two arrangements.
function TurnPreview({ p }: { p: number }) {
  const turned = p >= 0.42;
  const w = turnLoad(turned ? 'turned' : 'ch5');
  const C = 4; // cell size
  const TX = 40, TY = 70; // tile origin
  const loaded = new Set(w.threads.map((t) => `${t.sharedRow},${t.sharedCol}`));
  const shared = ramp(p, 0.68, 0.8);
  const sx = 536; // shared tile origin
  return (
    <svg viewBox="0 0 720 360" role="img" aria-label={tr('Corner turning: which edge of the tile one warp loads', 'Xoay góc (corner turning): một warp nạp cạnh nào của ô')} style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      <text x={4} y={16} fontSize="13" fontWeight="600" data-headline>
        {turned ? tr('(b) corner turned: the warp reads the left edge', '(b) đã xoay góc: warp đọc cạnh trái') : tr('(a) chapter 5\'s mapping: the warp reads the top edge', '(a) cách ánh xạ của chương 5: warp đọc cạnh trên')}
      </text>
      <text x={TX} y={TY - 10} fontSize="11" style={{ fill: 'var(--global)' }}><Translated vi="ô 32 × 32 của N, lưu theo từng cột">32 × 32 tile of N, stored column by column</Translated></text>
      {Array.from({ length: 32 * 32 }, (_, i) => {
        const r = Math.floor(i / 32);
        const c = i % 32;
        const on = loaded.has(`${r},${c}`);
        return <rect key={i} x={TX + c * C} y={TY + r * C} width={C - 0.6} height={C - 0.6} fill={on ? 'var(--global)' : 'var(--global-wash)'} />;
      })}
      <text x={TX} y={TY + 32 * C + 16} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="địa chỉ chạy dọc xuống mỗi cột">addresses run down each column</Translated></text>
      {/* the warp's sectors */}
      <g data-arrangement={turned ? 'turned' : 'ch5'} data-transactions={w.access.transactions}>
        <text x={230} y={TY + 30} fontSize="11.5"><Translated vi={<>lần nạp của một warp: {w.access.transactions} sector, dùng {+(w.access.efficiency * 100).toFixed(1)} % số byte đã nạp</>}>one warp's load: {w.access.transactions} sectors, {+(w.access.efficiency * 100).toFixed(1)} % of fetched bytes used</Translated></text>
        {w.access.segments.map((seg, i) => {
          const x = turned ? 300 + i * SW : 230 + i * 10;
          const used = w.threads.filter((t) => Math.floor(t.byte / SECTOR) === seg).length * 4 / SECTOR;
          return (
            <g key={seg}>
              <rect x={x} y={TY + 40} width={turned ? SW : 8} height={22} fill="var(--global-wash)" />
              <rect x={x} y={TY + 40} width={(turned ? SW : 8) * used} height={22} fill="var(--global)" />
              <rect x={x + 0.4} y={TY + 40.4} width={(turned ? SW : 8) - 0.8} height={21.2} fill="none" stroke="var(--ink-2)" strokeWidth="0.7" />
            </g>
          );
        })}
      </g>
      {/* turned inside shared memory */}
      <g opacity={shared}>
        <text x={sx} y={TY - 10} fontSize="11" style={{ fill: 'var(--shared)' }}><Translated vi="ô trong bộ nhớ chia sẻ">shared tile</Translated></text>
        {Array.from({ length: 32 * 32 }, (_, i) => {
          const r = Math.floor(i / 32);
          const c = i % 32;
          return <rect key={i} x={sx + c * C} y={TY + r * C} width={C - 0.6} height={C - 0.6} fill={turned && c === 0 ? 'var(--shared)' : 'var(--shared-wash)'} />;
        })}
        <path d={`M${TX + 32 * C + 14} ${TY + 64} C 300 200, 470 200, ${sx - 10} ${TY + 64}`} fill="none" stroke="var(--shared)" strokeWidth="1.4" />
        <text x={300} y={TY + 150} fontSize="11" style={{ fill: 'var(--shared)' }}><Translated vi="mỗi lane ghi hàng của riêng nó: warp lấp đầy một cột">each lane writes its own row: the warp fills a column</Translated></text>
        <text x={300} y={TY + 166} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi="bộ nhớ chia sẻ không có quy tắc hợp nhất truy cập; mục 6.4 nói về cái giá của nó">shared memory has no coalescing rule; section 6.4 has its cost</Translated></text>
      </g>
    </svg>
  );
}

const beats: Beat[] = [
  {
    clip: 'coalesce-1',
    get label() {
      return tr('Addresses', 'Địa chỉ');
    },
    draw: (p) => {
      const lanes = Math.round(ramp(p, 0.05, 0.8) * LANES);
      const a = access('n-row', 0);
      return (
        <Stage layout="row" k={0} lanesShown={lanes} sectors={0} /* i18n-ignore: layout key */
          headline={tr('k = 0: each lane computes k·Width + col', 'k = 0: mỗi lane tính k·Width + col')}
          note={lanes > 2 ? tr(`lane 0 → N[${a.lanes[0].element}], lane 1 → N[${a.lanes[1].element}], … lane 31 → N[${a.lanes[31].element}]: side by side`, `lane 0 → N[${a.lanes[0].element}], lane 1 → N[${a.lanes[1].element}], … lane 31 → N[${a.lanes[31].element}]: nằm cạnh nhau`) : undefined} />
      );
    },
  },
  {
    clip: 'coalesce-2',
    get label() {
      return tr('Four sectors', 'Bốn sector');
    },
    draw: (p) => (
      <Stage layout="row" k={0} lanesShown={LANES} sectors={ramp(p, 0.1, 0.6)} /* i18n-ignore: layout key */
        counters={p > 0.6 ? totals('row', 0, 1024) : undefined}
        headline={p < 0.6 ? tr('the hardware serves the warp in aligned 32-byte sectors (NVIDIA)', 'phần cứng phục vụ warp theo các sector 32 byte được căn chỉnh (NVIDIA)') : tr('4 transactions serve 32 lanes; every fetched byte is used', '4 giao dịch phục vụ 32 lane; mọi byte đã nạp đều được dùng')} />
    ),
  },
  {
    clip: 'coalesce-3',
    get label() {
      return tr('k advances', 'k tăng dần');
    },
    draw: (p) => {
      const k = kAt(p);
      return (
        <Stage layout="row" k={k} lanesShown={LANES} sectors={1} counters={totals('row', k, 1024)} /* i18n-ignore: layout key */
          headline={k === 0 ? tr('k = 0: 128 bytes from row 0 of N', 'k = 0: 128 byte từ hàng 0 của N') : tr(`k = ${k}: 128 bytes from the next row of N, ${(4 * WIDTH).toLocaleString('en-US')} bytes further on`, `k = ${k}: 128 byte từ hàng kế tiếp của N, xa thêm ${(4 * WIDTH).toLocaleString('en-US')} byte`)} />
      );
    },
  },
  {
    clip: 'coalesce-4',
    get label() {
      return tr('Column by column', 'Theo từng cột');
    },
    draw: (p) => (
      <Stage layout="col" k={0} lanesShown={Math.round(ramp(p, 0.05, 0.4) * LANES)} sectors={ramp(p, 0.45, 0.8)} /* i18n-ignore: layout key */
        counters={p > 0.8 ? totals('col', 0, 8192) : undefined}
        headline={p < 0.8 ? tr('N stored column by column: the same lanes, Width elements apart', 'N lưu theo từng cột: vẫn các lane đó, cách nhau Width phần tử') : tr('32 transactions: 32 bytes fetched for every 4 used', '32 giao dịch: nạp 32 byte cho mỗi 4 byte được dùng')} />
    ),
  },
  {
    clip: 'coalesce-5',
    get label() {
      return tr('Eight times', 'Gấp tám lần');
    },
    draw: (p) => {
      const k = kAt(p);
      const c = totals('col', k, 8192);
      return (
        <Stage layout="col" k={k} lanesShown={LANES} sectors={1} counters={c} /* i18n-ignore: layout key */
          headline={tr(`k = ${k}: fetched ${c.fetched / c.needed} times what the warp needed`, `k = ${k}: đã nạp gấp ${c.fetched / c.needed} lần lượng warp cần`)}
          note={p > 0.45 ? tr('the rest of each sector holds what that lane reads at the next steps of k: a cache might keep it', 'phần còn lại của mỗi sector chứa thứ mà lane đó đọc ở các bước k tiếp theo: một cache có thể giữ lại nó') : undefined} />
      );
    },
  },
  {
    clip: 'coalesce-6',
    get label() {
      return tr('Turn the corner', 'Xoay góc');
    },
    draw: (p) => <TurnPreview p={p} />,
  },
];

export function CoalesceScene() {
  return <Scene chapter="ch06" title={tr('One instruction, thirty-two addresses', 'Một lệnh, ba mươi hai địa chỉ')} beats={beats} />;
}
