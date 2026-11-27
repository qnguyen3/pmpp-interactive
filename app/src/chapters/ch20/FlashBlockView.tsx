import type { ReactNode } from 'react';
import { br as brOf, fmtv, type BlockTrace, type KProblem, type KStep, type Snapshot } from '../../sim/flashkernel';
import { Translated, tr } from '../../lib/i18n';

// One block of the flash kernel drawn from a flashkernel.ts snapshot: global arrays in amber,
// shared tiles in teal, registers in purple, as the memory-tier colours are used everywhere
// in the course. The section 20.5 kernel stepper and the scene "One block's journey" both
// draw with it, so the picture is computed from the same tested model in both.

const G = 7; // global cell size
const short = (x: number) => {
  if (Number.isNaN(x)) return '?';
  if (x <= -1e37) return '−F/2';
  if (x === -Infinity) return '−∞';
  return fmtv(Math.round(x * 100) / 100);
};

function hits(step: KStep | undefined, array: string, write?: boolean): Set<number> {
  const out = new Set<number>();
  for (const t of step?.touch ?? []) if (t.array === array && (write === undefined || t.write === write)) t.cells.forEach((c) => out.add(c));
  return out;
}

export function FlashBlockView({
  trace,
  prob,
  at,
  caption,
  showNumbers = true,
}: {
  trace: BlockTrace;
  prob: KProblem;
  /** Number of steps executed (the snapshot index). */
  at: number;
  caption?: ReactNode;
  showNumbers?: boolean;
}) {
  const c = trace.config;
  const BR = brOf(c);
  const snap: Snapshot = trace.snaps[Math.min(at, trace.snaps.length - 1)];
  const step: KStep | undefined = at > 0 ? trace.steps[at - 1] : undefined;
  const bad = !!step?.violation;
  const panelRows = new Set(Array.from({ length: BR }, (_, i) => trace.panel * BR + i));
  const curTile = step?.tile ?? null;
  const n = c.n;
  const d = c.d;

  const gx = [14, 82, 150, 218];
  const gy = 40;
  const globalGrid = (x: number, name: string, arr: 'Q' | 'K' | 'V' | 'O') => {
    const lit = hits(step, arr);
    return (
      <g key={name}>
        <text x={x} y={gy - 8} fontSize="11" fontWeight="600" style={{ fill: 'var(--global)' }}>{name}</text>
        {Array.from({ length: n * d }, (_, i) => {
          const r = Math.floor(i / d);
          const col = i % d;
          const inPanel = (arr === 'Q' || arr === 'O') && panelRows.has(r);
          const inTile = (arr === 'K' || arr === 'V') && curTile != null && r >= curTile * c.bc && r < (curTile + 1) * c.bc;
          const stored = arr === 'O' && snap.stored[r];
          const fill = lit.has(i) ? 'var(--global)' : stored ? 'var(--global)' : inPanel || inTile ? 'var(--global-wash)' : 'var(--cell)';
          return (
            <rect key={i} x={x + col * G} y={gy + r * G} width={G - 1} height={G - 1}
              fill={fill} opacity={stored && !lit.has(i) ? 0.55 : 1} stroke={inPanel || inTile ? 'var(--global)' : 'var(--rule)'} strokeWidth="0.5" />
          );
        })}
      </g>
    );
  };

  // shared
  const KX = 300, KY = 40, KC = 10;
  const SX = 362, SY = 40, SW = 42, SH = 22;
  const VX = 362 + c.bc * SW + 16, VY = 40, VC = 10;
  const ktW = hits(step, 'KT', true), ktR = hits(step, 'KT', false);
  const sW = hits(step, 'S', true), sR = hits(step, 'S', false);
  const vW = hits(step, 'Vj', true), vR = hits(step, 'Vj', false);
  const staleTile = (t: number) => curTile != null && t >= 0 && t !== curTile;

  // registers
  const RY = 40 + Math.max(n * G, d * KC, BR * SH) + 44;
  const rowsShown = Array.from({ length: BR }, (_, lr) => lr);
  const W = c.lanes;
  const execRows = new Set<number>();
  if (step && step.warp >= 0 && step.ii != null) execRows.add(step.warp * c.brWarp + step.ii);
  else if (step && step.warp >= 0) for (let ii = 0; ii < c.brWarp; ii++) execRows.add(step.warp * c.brWarp + ii);
  const height = RY + BR * 22 + 34;

  return (
    <svg viewBox={`0 0 720 ${height}`} role="img" aria-label={tr('One block of the flash kernel: global, shared and register storage', 'Một khối của kernel flash: bộ nhớ toàn cục, bộ nhớ chia sẻ và thanh ghi')} style={{ width: '100%', maxWidth: 720, display: 'block' }} data-block-view>
      {globalGrid(gx[0], tr('Q (global)', 'Q (toàn cục)'), 'Q')}
      {globalGrid(gx[1], tr('K (global)', 'K (toàn cục)'), 'K')}
      {globalGrid(gx[2], tr('V (global)', 'V (toàn cục)'), 'V')}
      {globalGrid(gx[3], tr('O (global)', 'O (toàn cục)'), 'O')}

      <text x={KX} y={KY - 8} fontSize="11" fontWeight="600" style={{ fill: 'var(--shared)' }}>KT_j</text>
      {snap.KT.map((v, i) => {
        const dd = Math.floor(i / c.bc);
        const jj = i % c.bc;
        const t = snap.KTtile[i];
        const hot = ktW.has(i) || ktR.has(i);
        return (
          <rect key={i} x={KX + jj * KC} y={KY + dd * KC} width={KC - 1} height={KC - 1}
            fill={hot ? 'var(--shared)' : t >= 0 ? 'var(--shared-wash)' : 'var(--cell)'}
            stroke={staleTile(t) && ktR.has(i) ? 'var(--fail)' : 'var(--shared)'} strokeWidth={staleTile(t) && ktR.has(i) ? 1.6 : 0.6}
            data-kt={i} data-tile={t}>
            <title>{tr(`KT_j[${dd}][${jj}] = ${short(v)}${t >= 0 ? ` (tile ${t})` : ''}`, `KT_j[${dd}][${jj}] = ${short(v)}${t >= 0 ? ` (ô ${t})` : ''}`)}</title>
          </rect>
        );
      })}

      <text x={SX} y={SY - 8} fontSize="11" fontWeight="600" style={{ fill: 'var(--shared)' }}><Translated vi="S_i, bị ghi đè bằng P">S_i, overwritten with P</Translated></text>
      {snap.S.map((v, i) => {
        const lr = Math.floor(i / c.bc);
        const jj = i % c.bc;
        const kind = snap.Skind[i];
        const t = snap.Stile[i];
        const hot = sW.has(i) || sR.has(i);
        const stale = sR.has(i) && bad && step?.op === 'compute_O' && !(kind === 'P' && t === curTile);
        return (
          <g key={i} data-s={i} data-kind={kind} data-tile={t}>
            <rect x={SX + jj * SW} y={SY + lr * SH} width={SW - 2} height={SH - 2} rx="2"
              fill={stale ? 'var(--fail-wash)' : hot ? 'var(--shared)' : kind === 'none' ? 'var(--cell)' : 'var(--shared-wash)'}
              stroke={stale ? 'var(--fail)' : 'var(--shared)'} strokeWidth={stale ? 1.6 : 0.8} />
            {showNumbers && kind !== 'none' ? (
              <text x={SX + jj * SW + SW / 2 - 1} y={SY + lr * SH + 14} fontSize="9.5" textAnchor="middle"
                style={{ fill: stale ? 'var(--fail)' : hot ? 'var(--page)' : 'var(--ink)' }}>
                {kind === 'P' ? 'P ' : ''}{short(v)}
              </text>
            ) : null}
          </g>
        );
      })}
      {rowsShown.map((lr) => (
        <text key={lr} x={SX - 4} y={SY + lr * SH + 14} fontSize="9" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>{trace.panel * BR + lr}</text>
      ))}

      <text x={VX} y={VY - 8} fontSize="11" fontWeight="600" style={{ fill: 'var(--shared)' }}>V_j</text>
      {snap.Vj.map((_, i) => {
        const jj = Math.floor(i / d);
        const dd = i % d;
        const t = snap.Vtile[i];
        const hot = vW.has(i) || vR.has(i);
        return (
          <rect key={i} x={VX + dd * VC} y={VY + jj * VC} width={VC - 1} height={VC - 1}
            fill={hot ? 'var(--shared)' : t >= 0 ? 'var(--shared-wash)' : 'var(--cell)'}
            stroke={staleTile(t) && vR.has(i) ? 'var(--fail)' : 'var(--shared)'} strokeWidth={staleTile(t) && vR.has(i) ? 1.6 : 0.6} />
        );
      })}
      {curTile != null ? (
        <text x={KX} y={Math.max(KY + d * KC, SY + BR * SH) + 16} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
          <Translated vi={<>đang xử lý ô {curTile} (các hàng K và V từ {curTile * c.bc} đến {Math.min(n, (curTile + 1) * c.bc) - 1})</>}>
            working on tile {curTile} (K and V rows {curTile * c.bc} to {Math.min(n, (curTile + 1) * c.bc) - 1})
          </Translated>
        </text>
      ) : null}

      <text x={14} y={RY - 8} fontSize="11" fontWeight="600" style={{ fill: 'var(--reg)' }}>
        <Translated vi="thanh ghi, theo từng hàng của panel (bản sao m và D của lane 0; Õ / D gom từ các lane)">
          registers, per row of the panel (lane 0's copy of m and D; Õ / D gathered from the lanes)
        </Translated>
      </text>
      {rowsShown.map((lr) => {
        const w = Math.floor(lr / c.brWarp);
        const ii = lr % c.brWarp;
        const reg = snap.regs[w * W];
        const y = RY + lr * 22;
        const row = trace.panel * BR + lr;
        const on = execRows.has(lr);
        return (
          <g key={lr} data-reg-row={row}>
            {on ? <rect x={10} y={y - 2} width={700} height={20} rx="3" fill="var(--rule-soft)" /> : null}
            <text x={14} y={y + 12} fontSize="10.5">
              <Translated vi={<>hàng {row}{row >= n ? ' (quá N)' : ''}, warp {w}</>}>
                row {row}{row >= n ? ' (past N)' : ''}, warp {w}
              </Translated>
            </text>
            <text x={128} y={y + 12} fontSize="10.5" style={{ fill: 'var(--reg)' }}>m {fmtv(reg.m[ii])}</text>
            <text x={236} y={y + 12} fontSize="10.5" style={{ fill: 'var(--reg)' }}>D {fmtv(reg.D[ii])}</text>
            {Array.from({ length: d }, (_, dd) => {
              const r = snap.regs[w * W + (dd % W)];
              const o = r.O[ii][Math.floor(dd / W)];
              const broken = Number.isNaN(o) || Number.isNaN(r.D[ii]);
              // Before the first visible column O and D are both 0: nothing to show yet, not a NaN.
              const shown = r.D[ii] > 0 || broken;
              const v = broken ? NaN : shown ? o / r.D[ii] : 0;
              const exact = row < n ? prob.O[row * d + dd] : 0;
              return (
                <g key={dd}>
                  <rect x={318 + dd * 40} y={y} width={38} height={16} rx="2"
                    fill={broken ? 'var(--fail-wash)' : shown ? 'var(--reg-wash)' : 'var(--cell)'} stroke={broken ? 'var(--fail)' : 'var(--reg)'} strokeWidth="0.6" />
                  {shown ? (
                    <text x={318 + dd * 40 + 19} y={y + 12} fontSize="9" textAnchor="middle" style={{ fill: broken ? 'var(--fail)' : 'var(--ink)' }}>{short(v)}</text>
                  ) : null}
                  <title>{tr(`Õ/D column ${dd}: ${short(v)}; exact over the whole row: ${short(exact)}`, `Õ/D cột ${dd}: ${short(v)}; giá trị chính xác trên cả hàng: ${short(exact)}`)}</title>
                </g>
              );
            })}
          </g>
        );
      })}
      {caption ? <text x={14} y={height - 10} fontSize="12" fontWeight="600">{caption}</text> : null}
    </svg>
  );
}
