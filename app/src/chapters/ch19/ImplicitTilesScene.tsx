import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { IMPLICIT_SCENE, type PhaseRecord } from '../../sim/implicitgemm';
import { BOOK_F, BOOK_X } from '../../sim/unfold';

// "Implicit unfolding, one tile at a time": Fig. 19.11 (corrected) on the book's
// example with T = 2, block (0, 0), every value from sim/implicitgemm.ts. Six
// beats, one narration clip each (content/ch19/narration/implicit-tiles-N.txt).
const SC = IMPLICIT_SCENE;
const blk = SC.block;
const T = SC.shape.T;
const C_X = 26;
const XX = (c: number) => 20 + c * 96;
const XY = 44;
const C_F = 22;
const FX = 20;
const FY = 178;
const BX = 336;
const BY = 44;
const C_Y = 30;
const YX = 486;
const YY = 44;
const SH_X = 486; // shared tiles
const SH_Y = 160;
const RG_Y = 262; // registers

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

type Tier = 'global' | 'shared' | 'reg';
function Cell({ x, y, size, value, hot, wash, dashed, tier = 'global' }: {
  x: number; y: number; size: number; value?: number | null; hot?: boolean; wash?: boolean; dashed?: boolean; tier?: Tier;
}) {
  return (
    <g>
      <rect x={x + 1} y={y + 1} width={size - 2} height={size - 2} rx="2"
        fill={hot ? `var(--${tier})` : wash ? `var(--${tier}-wash)` : dashed ? 'none' : 'var(--cell)'}
        stroke={hot || wash || tier !== 'global' ? `var(--${tier})` : 'var(--rule)'} strokeDasharray={dashed ? '3 2' : undefined} />
      {value != null ? (
        <text x={x + size / 2} y={y + size / 2 + 4} fontSize="12" textAnchor="middle" style={{ fill: hot ? 'var(--page)' : 'var(--ink)' }}>{value}</text>
      ) : null}
    </g>
  );
}
const Label = ({ x, y, children, tier = 'global' }: { x: number; y: number; children: ReactNode; tier?: string }) => (
  <text x={x} y={y} fontSize="12" fontWeight="600" style={{ fill: `var(--${tier})` }}>{children}</text>
);

interface Frame {
  ph: number | null; // phase whose tiles are drawn
  fLoaded?: number; // F-tile slots loaded so far (0..4)
  bLoaded?: number; // B-tile slots gathered so far (0..4)
  lines?: boolean; // draw the gather lines
  hotShared?: boolean;
  barrier?: string | null;
  sums?: number[] | null;
  stored?: boolean;
  headline: ReactNode;
  twice?: boolean; // mark the X pixel that fills two slots
}

const xAt = (c: number, h: number, w: number) => ({ x: XX(c) + w * C_X, y: XY + h * C_X });
const prev = (ph: number): PhaseRecord | null => (ph > 0 ? blk.phases[ph - 1] : null);

function Stage(f: Frame) {
  const P = f.ph != null ? blk.phases[f.ph] : null;
  const fl = f.fLoaded ?? 4;
  const bl = f.bLoaded ?? 4;
  // What a shared slot holds: this phase's value once loaded, until then the last phase's.
  const fds = Array.from({ length: 4 }, (_, i) => (P && i < fl ? P.Fds[i] : prev(f.ph ?? 0) && f.ph != null ? prev(f.ph)!.Fds[i] : null));
  const bds = Array.from({ length: 4 }, (_, i) => (P && i < bl ? P.Bds[i] : prev(f.ph ?? 0) && f.ph != null ? prev(f.ph)!.Bds[i] : null));
  const gathered = P ? P.b.slice(0, bl) : [];
  const hits = new Map<number, number>();
  for (const g of gathered) hits.set(g.idx!, (hits.get(g.idx!) ?? 0) + 1);
  const fCols = P ? [P.ph * T, P.ph * T + 1] : [];
  return (
    <svg viewBox="0 0 720 372" role="img" aria-label={tr('Animation: implicit unfolding, one tile at a time', 'Hoạt ảnh: unfold ngầm, mỗi lần một ô')} style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      <Label x={20} y={16}><Translated vi="X: giữ nguyên (bộ nhớ toàn cục)">X: stays as it is (global memory)</Translated></Label>
      {[0, 1, 2].map((c) => (
        <g key={c}>
          <text x={XX(c)} y={XY - 6} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>map {c}</>}>map {c}</Translated></text>
          {Array.from({ length: 9 }, (_, i) => {
            const flat = c * 9 + i;
            const { x, y } = xAt(c, Math.floor(i / 3), i % 3);
            const n = hits.get(flat) ?? 0;
            return (
              <g key={i}>
                <Cell x={x} y={y} size={C_X} value={BOOK_X[flat]} hot={n > 0} />
                {f.twice && n > 1 ? <rect x={x - 1} y={y - 1} width={C_X + 2} height={C_X + 2} rx="3" fill="none" stroke="var(--ink)" strokeWidth="2.5" /> : null}
              </g>
            );
          })}
        </g>
      ))}

      <Label x={FX} y={FY - 12}><Translated vi="F: 2 × 12 (bộ nhớ toàn cục)">F: 2 × 12 (global memory)</Translated></Label>
      {BOOK_F.map((v, i) => {
        const m = Math.floor(i / 12);
        const k = i % 12;
        const inTile = fCols.includes(k);
        return <Cell key={i} x={FX + k * C_F} y={FY + m * C_F} size={C_F} value={v} wash={inTile} hot={inTile && P != null && m * 2 + (k - fCols[0]) < fl && fl < 4} />;
      })}
      {P ? <rect x={FX + fCols[0] * C_F - 2} y={FY - 2} width={2 * C_F + 4} height={2 * C_F + 4} fill="none" stroke="var(--ink)" strokeWidth="2" /> : null}

      <Label x={BX} y={16}><Translated vi="B: không bao giờ được lưu">B: never stored</Translated></Label>
      <text x={BX} y={30} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi="12 × 4, chỉ là đường viền">12 × 4, an outline only</Translated></text>
      {Array.from({ length: 48 }, (_, i) => (
        <Cell key={i} x={BX + (i % 4) * C_F} y={BY + Math.floor(i / 4) * C_F} size={C_F} dashed />
      ))}
      {P ? <rect x={BX - 2} y={BY + P.ph * T * C_F - 2} width={2 * C_F + 4} height={2 * C_F + 4} fill="none" stroke="var(--ink)" strokeWidth="2" /> : null}
      {[4, 8].map((u) => <line key={u} x1={BX - 2} x2={BX + 4 * C_F + 2} y1={BY + u * C_F} y2={BY + u * C_F} stroke="var(--ink-3)" />)}

      <Label x={YX} y={16}><Translated vi="Y (bộ nhớ toàn cục)">Y (global memory)</Translated></Label>
      {SC.finals.length
        ? [0, 1, 2, 3, 4, 5, 6, 7].map((i) => {
            const r = Math.floor(i / 4);
            const c = i % 4;
            const mine = c < 2;
            const v = mine && f.stored ? SC.finals[r * 2 + c] : null;
            return <Cell key={i} x={YX + c * C_Y} y={YY + r * C_Y} size={C_Y} value={v} wash={mine && !f.stored} dashed={!mine} />;
          })
        : null}
      <rect x={YX - 2} y={YY - 2} width={2 * C_Y + 4} height={2 * C_Y + 4} fill="none" stroke="var(--ink)" strokeWidth="2" />
      <text x={YX} y={YY + 2 * C_Y + 16} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi="khối này: map 0 và 1, pixel 0 và 1">this block: maps 0 and 1, pixels 0 and 1</Translated></text>

      <Label x={SH_X} y={SH_Y - 8} tier="shared">Fds</Label>
      <Label x={SH_X + 86} y={SH_Y - 8} tier="shared"><Translated vi="Bds (chia sẻ, khối này)">Bds (shared, this block)</Translated></Label>
      {fds.map((v, i) => <Cell key={'f' + i} x={SH_X + (i % 2) * C_Y} y={SH_Y + Math.floor(i / 2) * C_Y} size={C_Y} value={v} tier="shared" hot={f.hotShared} wash={!f.hotShared && v != null} />)}
      {bds.map((v, i) => <Cell key={'b' + i} x={SH_X + 86 + (i % 2) * C_Y} y={SH_Y + Math.floor(i / 2) * C_Y} size={C_Y} value={v} tier="shared" hot={f.hotShared} wash={!f.hotShared && v != null} />)}

      {f.lines && P
        ? gathered.map((g, i) => {
            const a = xAt(g.c, g.h, g.w);
            return (
              <line key={i} x1={a.x + C_X / 2} y1={a.y + C_X / 2} x2={SH_X + 86 + (i % 2) * C_Y + C_Y / 2} y2={SH_Y + Math.floor(i / 2) * C_Y + C_Y / 2}
                stroke="var(--shared)" strokeWidth="1.6" strokeDasharray="5 3" />
            );
          })
        : null}

      {f.barrier ? (
        <g>
          <rect x={336} y={316} width={370} height={24} rx="3" fill="var(--ink)" />
          <text x={521} y={332} fontSize="12" textAnchor="middle" style={{ fill: 'var(--page)' }}>{f.barrier}</text>
        </g>
      ) : null}

      {f.sums ? (
        <g>
          <Label x={SH_X} y={RG_Y - 8} tier="reg"><Translated vi="tổng tích lũy (thanh ghi)">running sums (registers)</Translated></Label>
          {f.sums.map((v, i) => (
            <g key={i}>
              <rect x={SH_X + i * 54} y={RG_Y} width={48} height={28} rx="2" fill="var(--reg-wash)" stroke="var(--reg)" />
              <text x={SH_X + i * 54 + 24} y={RG_Y + 19} fontSize="12.5" textAnchor="middle" style={{ fill: 'var(--ink)' }}>{v}</text>
            </g>
          ))}
        </g>
      ) : null}

      <text x={20} y={364} fontSize="13" fontWeight="600" style={{ fill: 'var(--ink)' }}>{f.headline}</text>
    </svg>
  );
}

const zero = [0, 0, 0, 0];
const sumsAfter = (ph: number) => (ph < 0 ? zero : blk.phases[ph].sums);

/** One phase in a beat: [barrier 2 of the previous phase], loads, barrier 1, products. */
function phaseFrame(ph: number, p: number, opts: { lead?: boolean } = {}): Omit<Frame, 'headline'> {
  const a = opts.lead ? 0.18 : 0; // time spent on the previous phase's second barrier
  if (p < a) return { ph: ph - 1, barrier: tr('barrier 2: nobody overwrites a tile still being read', 'rào chắn 2: không ai ghi đè một ô vẫn đang được đọc'), sums: sumsAfter(ph - 1), hotShared: false };
  const fLoaded = Math.min(4, Math.floor(ramp(p, a + 0.02, a + 0.2) * 4.99));
  const bLoaded = Math.min(4, Math.floor(ramp(p, a + 0.22, a + 0.5) * 4.99));
  const computing = p >= a + 0.68;
  return {
    ph,
    fLoaded,
    bLoaded,
    lines: p < a + 0.66,
    twice: true,
    barrier: p >= a + 0.52 && p < a + 0.66 ? tr('barrier 1: every slot written before anyone reads', 'rào chắn 1: mọi ô nhớ được ghi trước khi ai đó đọc') : null,
    hotShared: computing,
    sums: computing ? sumsAfter(ph) : sumsAfter(ph - 1),
  };
}

const beats: Beat[] = [
  {
    clip: 'implicit-tiles-1',
    get label() {
      return tr('The block', 'Khối');
    },
    draw: () => <Stage ph={null} sums={zero} headline={tr('block (0, 0) owns 14, 20 (map 0) and 12, 24 (map 1); B is only an outline', 'khối (0, 0) sở hữu 14, 20 (map 0) và 12, 24 (map 1); B chỉ là đường viền')} />,
  },
  {
    clip: 'implicit-tiles-2',
    get label() {
      return tr('Phase 0 loads', 'Pha 0 nạp');
    },
    draw: (p) => {
      const fLoaded = Math.min(4, Math.floor(ramp(p, 0.05, 0.35) * 4.99));
      const bLoaded = Math.min(4, Math.floor(ramp(p, 0.4, 0.85) * 4.99));
      return (
        <Stage ph={0} fLoaded={fLoaded} bLoaded={bLoaded} lines twice sums={zero}
          headline={bLoaded < 4 ? tr('F tile: as in any tiled GEMM. B tile: gathered from X with Eq. 19.5', 'Ô F: như mọi GEMM phân ô. Ô B: gom từ X theo Eq. 19.5') : tr('3 input pixels fill 4 slots: X[n, 0, 0, 1] (outlined) lands in two', '3 pixel đầu vào lấp 4 ô nhớ: X[n, 0, 0, 1] (có viền) nằm ở hai ô')} />
      );
    },
  },
  {
    clip: 'implicit-tiles-3',
    get label() {
      return tr('Barrier, then multiply', 'Rào chắn, rồi nhân');
    },
    draw: (p) => {
      const computing = p >= 0.35;
      return (
        <Stage ph={0} lines={false} barrier={computing ? null : tr('barrier 1: every slot written before anyone reads', 'rào chắn 1: mọi ô nhớ được ghi trước khi ai đó đọc')} hotShared={computing}
          sums={computing ? sumsAfter(0) : zero}
          headline={computing ? tr('each thread: 2 products from shared memory into its register', 'mỗi luồng: 2 tích từ bộ nhớ chia sẻ vào thanh ghi của nó') : tr('wait until the whole block has loaded', 'chờ đến khi cả khối nạp xong')} />
      );
    },
  },
  {
    clip: 'implicit-tiles-4',
    get label() {
      return tr('Phase 1', 'Pha 1');
    },
    draw: (p) => <Stage {...phaseFrame(1, p, { lead: true })} headline={tr('phase 1: F columns 2–3, B from the middle row of map 0, the same slots', 'pha 1: cột 2–3 của F, B từ hàng giữa của map 0, cùng những ô nhớ đó')} />,
  },
  {
    clip: 'implicit-tiles-5',
    get label() {
      return tr('Phases 2 to 5', 'Pha 2 đến 5');
    },
    draw: (p) => {
      const ph = 2 + Math.min(3, Math.floor(ramp(p, 0, 0.84) * 4));
      const q = ramp(p, (ph - 2) * 0.21, (ph - 1) * 0.21);
      const fr = phaseFrame(ph, q, { lead: true });
      const done = p >= 0.86;
      return (
        <Stage {...fr} {...(done ? { ph: 5, sums: sumsAfter(5), stored: true, barrier: null, lines: false, hotShared: false } : {})}
          headline={done ? tr(`after phase 5: ${sumsAfter(5).join(', ')}, stored to Y`, `sau pha 5: ${sumsAfter(5).join(', ')}, ghi vào Y`) : tr(`phase ${ph}: map ${blk.phases[ph].b[0].c} of the input`, `pha ${ph}: map ${blk.phases[ph].b[0].c} của đầu vào`)} />
      );
    },
  },
  {
    clip: 'implicit-tiles-6',
    get label() {
      return tr('The count', 'Phép đếm');
    },
    draw: (p) => {
      const t = SC.traffic;
      const grow = ramp(p, 0.08, 0.6);
      const scale = 3.2; // px per access
      const bar = (y: number, n: number, label: string, tier: Tier) => (
        <g key={label}>
          <text x={40} y={y - 6} fontSize="12" style={{ fill: 'var(--ink)' }}>{label}</text>
          <rect x={40} y={y} width={n * scale * grow} height={16} fill={`var(--${tier})`} />
          <text x={46 + n * scale * grow} y={y + 13} fontSize="12" style={{ fill: 'var(--ink)' }}>{Math.round(n * grow)}</text>
        </g>
      );
      return (
        <svg viewBox="0 0 720 372" role="img" aria-label={tr('Global accesses of the three methods', 'Số truy cập bộ nhớ toàn cục của ba phương pháp')} style={{ width: '100%', maxWidth: 720, display: 'block' }}>
          <text x={40} y={30} fontSize="13" fontWeight="600" style={{ fill: 'var(--ink)' }}><Translated vi="truy cập X và B trong bộ nhớ toàn cục, cả ví dụ nhỏ (2 khối)">accesses to X and B in global memory, the whole toy (2 blocks)</Translated></text>
          <text x={40} y={66} fontSize="12" fontWeight="600" style={{ fill: 'var(--ink-2)' }}><Translated vi="kernel trực tiếp (Fig. 19.7)">direct kernel (Fig. 19.7)</Translated></text>
          {bar(90, t.direct.x, tr('reads of X', 'lần đọc X'), 'global')}
          <text x={40} y={140} fontSize="12" fontWeight="600" style={{ fill: 'var(--ink-2)' }}><Translated vi="unroll tường minh, rồi một GEMM phân ô">explicit unrolling, then a tiled GEMM</Translated></text>
          {bar(164, t.explicit.unfoldReadsX, tr('reads of X to build B', 'lần đọc X để dựng B'), 'global')}
          {bar(200, t.explicit.writesB, tr('writes of B', 'lần ghi B'), 'global')}
          {bar(236, t.explicit.gemmReadsB, tr('reads of B by the GEMM', 'lần GEMM đọc B'), 'global')}
          <text x={40} y={286} fontSize="12" fontWeight="600" style={{ fill: 'var(--ink-2)' }}><Translated vi="unroll ngầm (Fig. 19.11)">implicit unrolling (Fig. 19.11)</Translated></text>
          {bar(310, t.implicit.gathersX, tr('gathers from X; copies only in shared memory', 'lần gom từ X; bản sao chỉ nằm trong bộ nhớ chia sẻ'), 'global')}
          {p > 0.7 ? (
            <text x={430} y={330} fontSize="12" fontWeight="600" style={{ fill: 'var(--ink)' }}>
              <Translated vi="cái giá: một phép chia và một phép lấy dư cho mỗi lần gom">the price: a divide and a remainder per gather</Translated>
            </text>
          ) : null}
          <text x={430} y={90} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>F: {t.direct.f} lần đọc với kernel trực tiếp,</>}>F: {t.direct.f} reads direct,</Translated></text>
          <text x={430} y={105} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>{t.implicit.f} với ô 2 × 2</>}>{t.implicit.f} with 2 × 2 tiles</Translated></text>
        </svg>
      );
    },
  },
];

export function ImplicitTilesScene() {
  return <Scene chapter="ch19" title={tr('Implicit unfolding, one tile at a time', 'Unfold ngầm, mỗi lần một ô')} beats={beats} />;
}
