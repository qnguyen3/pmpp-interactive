import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { SCENE, sceneData, type Store } from '../../sim/compactstore';
import { Translated, tr } from '../../lib/i18n';

// "Gather, then write together": the book's sixteen keys, two blocks of eight threads, warps
// of four lanes, 16-byte toy segments (four keys). Every number drawn comes from sceneData()
// in src/sim/compactstore.ts, which compactstore.test.ts pins together with the narration
// scripts (content/ch12/narration/stage-write-N.txt).
const D = sceneData();
const KEYS = D.keep.length;
const W = 720;
const H = 440;
const CELL = 38;
const GAP = 26; // between the two blocks
const IN_X0 = (W - (KEYS * CELL + GAP)) / 2;
const inX = (i: number) => IN_X0 + i * CELL + (i >= SCENE.blockDim ? GAP : 0) + CELL / 2;
const IN_Y = 52;
const SH_Y = 156;
const OUT_CELL = 46;
const OUT_X0 = (W - D.direct.totals.kept * OUT_CELL) / 2;
const outX = (j: number) => OUT_X0 + j * OUT_CELL + OUT_CELL / 2;
const OUT_Y = 262;
const SLOTS_PER_SEG = SCENE.segmentBytes / 4;
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const keptIdx = D.keep.flatMap((k, i) => (k ? [i] : []));
/** Block-local offset of a kept key (its slot in the block's staging array). */
const local = (i: number, per: number) => D.offsets[i] - D.offsets[Math.floor(i / per) * per];

interface Frame {
  offsets?: number; // how many kept keys show their offset
  direct?: number; // direct stores drawn
  toShared?: number; // kept keys placed in shared memory
  barrier?: boolean;
  staged?: number; // staged (copy-out) stores drawn
  coarse?: boolean; // one block of eight threads over all sixteen keys
  faint?: boolean;
  tally?: number; // 0..1
  note?: ReactNode;
  data?: Record<string, string | number>;
}

function Arrow({ x1, y1, x2, y2, tier }: { x1: number; y1: number; x2: number; y2: number; tier: 'global' | 'shared' }) {
  return <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={`var(--${tier})`} strokeWidth={1.4} markerEnd={`url(#sw-head-${tier})`} />;
}

function Stage(f: Frame) {
  const per = f.coarse ? KEYS : SCENE.blockDim;
  const blocks = KEYS / per;
  const stores: Store[] = f.direct != null ? D.direct.stores.slice(0, f.direct) : f.staged != null ? (f.coarse ? D.coarse : D.staged).stores.slice(0, f.staged) : [];
  const written = new Set(stores.flatMap((s) => s.slots));
  const touched = new Set(stores.flatMap((s) => s.segments));
  const placed = new Set(keptIdx.slice(0, f.toShared ?? 0));
  const showShared = f.toShared != null || f.staged != null;
  const attrs = Object.fromEntries(Object.entries(f.data ?? {}).map(([k, v]) => [`data-${k}`, v])); // i18n-ignore: attribute name
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Animation: the stores of a direct scatter, then of a staged one', 'Hoạt cảnh: các lần ghi của một phép rải trực tiếp, rồi của một phép rải có gom tạm')} style={{ width: '100%', maxWidth: W, display: 'block' }} {...attrs}>
      <defs>
        {(['global', 'shared'] as const).map((t) => (
          <marker key={t} id={`sw-head-${t}`} viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto">{/* i18n-ignore: SVG attribute */}
            <path d="M0,0 L6,3 L0,6 z" fill={`var(--${t})`} />
          </marker>
        ))}
      </defs>
      <g opacity={f.faint ? 0.45 : 1}>
        {/* blocks and warps */}
        {Array.from({ length: blocks }, (_, b) => {
          const x1 = inX(b * per) - CELL / 2;
          const x2 = inX(b * per + per - 1) + CELL / 2;
          return (
            <g key={b}>
              <text x={(x1 + x2) / 2} y={14} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
                <Translated vi={<>khối {b}: {SCENE.blockDim} luồng{f.coarse ? `, mỗi luồng ${SCENE.coarse} khóa` : ''}</>}>
                  block {b}: {SCENE.blockDim} threads{f.coarse ? tr(`, ${SCENE.coarse} keys each`, `, mỗi luồng ${SCENE.coarse} khóa`) : ''}
                </Translated>
              </text>
              {Array.from({ length: per / SCENE.warp / (f.coarse ? SCENE.coarse : 1) }, (_, w) => {
                const span = SCENE.warp * (f.coarse ? SCENE.coarse : 1);
                const a = inX(b * per + w * span) - CELL / 2 + 2;
                const z = inX(b * per + w * span + span - 1) + CELL / 2 - 2;
                return (
                  <g key={w}>
                    <line x1={a} y1={IN_Y - 8} x2={z} y2={IN_Y - 8} stroke="var(--ink-3)" />
                    <text x={(a + z) / 2} y={IN_Y - 12} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{tr(`warp ${w}`, `warp ${w}`)}</text>
                  </g>
                );
              })}
            </g>
          );
        })}
        {/* the keys */}
        {D.keep.map((k, i) => (
          <g key={i} data-key={i}>
            <rect x={inX(i) - CELL / 2 + 1} y={IN_Y} width={CELL - 2} height={24} rx="2"
              fill={k ? 'var(--global-wash)' : 'var(--cell)'} stroke={k ? 'var(--global)' : 'var(--rule)'} />
            <text x={inX(i)} y={IN_Y + 16} fontSize="11" textAnchor="middle" style={{ fill: k ? 'var(--ink)' : 'var(--ink-3)' }}>k{i}</text>
            {k && keptIdx.indexOf(i) < (f.offsets ?? 0) ? (
              <text x={inX(i)} y={IN_Y + 40} fontSize="10.5" textAnchor="middle" data-offset={D.offsets[i]} style={{ fill: 'var(--global)' }}>
                ↓{D.offsets[i]}
              </text>
            ) : null}
          </g>
        ))}

        {/* shared memory, per block */}
        {showShared ? (
          <g>
            {Array.from({ length: blocks }, (_, b) => {
              const x1 = inX(b * per) - CELL / 2;
              return (
                <g key={b} data-shared={b}>
                  <text x={x1} y={SH_Y - 8} fontSize="10.5" fontWeight="600" style={{ fill: 'var(--shared)' }}>
                    <Translated vi={<>mảng gom tạm của khối {b} (bộ nhớ chia sẻ)</>}>block {b}'s staging array (shared memory)</Translated>
                  </text>
                  {Array.from({ length: per }, (_, s) => (
                    <rect key={s} x={inX(b * per + s) - CELL / 2 + 1} y={SH_Y} width={CELL - 2} height={24} rx="2" fill="var(--cell)" stroke="var(--shared)" strokeOpacity={0.5} />
                  ))}
                </g>
              );
            })}
            {keptIdx.filter((i) => placed.has(i)).map((i) => {
              const b = Math.floor(i / per);
              const slot = b * per + local(i, per);
              return (
                <g key={i} data-staged={i}>
                  <rect x={inX(slot) - CELL / 2 + 1} y={SH_Y} width={CELL - 2} height={24} rx="2" fill="var(--shared-wash)" stroke="var(--shared)" />
                  <text x={inX(slot)} y={SH_Y + 16} fontSize="11" textAnchor="middle">k{i}</text>
                  {(f.staged ?? 0) === 0 ? <Arrow x1={inX(i)} y1={IN_Y + 46} x2={inX(slot)} y2={SH_Y - 2} tier="shared" /> : null}
                </g>
              );
            })}
            {f.barrier ? (
              <g data-barrier="1">
                <rect x={IN_X0} y={SH_Y + 32} width={KEYS * CELL + GAP} height={18} rx="3" fill="var(--ink)" />
                <text x={W / 2} y={SH_Y + 45} fontSize="11" textAnchor="middle" style={{ fill: 'var(--page)' }}>{tr('barrier: the whole block has staged its keys', 'rào chắn: cả khối đã gom tạm xong các khóa của mình')}</text>
              </g>
            ) : null}
          </g>
        ) : null}

        {/* the output, with segment outlines */}
        <text x={OUT_X0 - 10} y={OUT_Y + 12} fontSize="11" fontWeight="600" textAnchor="end" style={{ fill: 'var(--global)' }}>{tr('output', 'đầu ra')}</text>
        <text x={OUT_X0 - 10} y={OUT_Y + 25} fontSize="10" textAnchor="end" style={{ fill: 'var(--global)' }}>{tr('(global)', '(toàn cục)')}</text>
        {Array.from({ length: Math.ceil(D.direct.totals.kept / SLOTS_PER_SEG) }, (_, g) => (
          <rect key={g} data-segment={g} data-touched={touched.has(g) ? 1 : 0}
            x={OUT_X0 + g * SLOTS_PER_SEG * OUT_CELL - 3} y={OUT_Y - 4} width={SLOTS_PER_SEG * OUT_CELL + 6} height={36} rx="4"
            fill="none" stroke={touched.has(g) ? 'var(--ink)' : 'var(--rule)'} strokeWidth={touched.has(g) ? 1.6 : 1} strokeDasharray={touched.has(g) ? undefined : '4 3'} />
        ))}
        {Array.from({ length: D.direct.totals.kept }, (_, j) => {
          const key = keptIdx[j];
          return (
            <g key={j} data-out={j} data-written={written.has(j) ? 1 : 0}>
              <rect x={outX(j) - OUT_CELL / 2 + 2} y={OUT_Y} width={OUT_CELL - 4} height={28} rx="2"
                fill={written.has(j) ? 'var(--global-wash)' : 'var(--cell)'} stroke="var(--global)" strokeOpacity={written.has(j) ? 1 : 0.4} />
              <text x={outX(j)} y={OUT_Y + 18} fontSize="11.5" textAnchor="middle">{written.has(j) ? `k${key}` : ''}</text>
              <text x={outX(j)} y={OUT_Y + 42} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{j}</text>
            </g>
          );
        })}
        <text x={OUT_X0 + D.direct.totals.kept * OUT_CELL + 10} y={OUT_Y + 18} fontSize="9.5" style={{ fill: 'var(--ink-3)' }}>
          {tr('segment =', 'đoạn =')} {SCENE.segmentBytes} {tr('bytes', 'byte')}
        </text>

        {/* the stores: arrows and a bracket each */}
        {stores.map((s, k) => {
          const lo = Math.min(...s.slots);
          const hi = Math.max(...s.slots);
          const fromY = f.direct != null ? IN_Y + 46 : SH_Y + 26;
          const by = OUT_Y + 52 + (k % 2) * 24;
          return (
            <g key={s.id} data-store={s.id}>
              {s.slots.map((slot, m) => {
                const src = f.direct != null
                  ? inX(s.block * SCENE.blockDim + s.warp * SCENE.warp + s.lanes[m])
                  : inX(s.block * per + (slot - (f.coarse ? D.coarse : D.staged).runs[s.block].start));
                return <Arrow key={slot} x1={src} y1={fromY} x2={outX(slot)} y2={OUT_Y - 6} tier="global" />;
              })}
              <line x1={outX(lo) - OUT_CELL / 2 + 4} y1={by} x2={outX(hi) + OUT_CELL / 2 - 4} y2={by} stroke="var(--ink)" strokeWidth={2} />
              <text x={(outX(lo) + outX(hi)) / 2} y={by + 11} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
                <Translated vi={<>lần ghi {k + 1}: khối {s.block}, warp {s.warp}, {s.lanes.length} lane</>}>
                  store {k + 1}: block {s.block}, warp {s.warp}, {s.lanes.length} lane{s.lanes.length === 1 ? '' : 's'}
                </Translated>
              </text>
            </g>
          );
        })}
      </g>

      {f.tally != null ? (
        <g data-tally="1">
          {[
            { label: tr('direct (Fig. 12.8)', 'trực tiếp (Fig. 12.8)'), t: D.direct.totals, y: 372 },
            { label: tr('staged (Fig. 12.9)', 'gom tạm (Fig. 12.9)'), t: D.staged.totals, y: 396 },
          ].map((r) => (
            <g key={r.y} data-row={r.label}>
              <text x={150} y={r.y + 11} fontSize="11.5" textAnchor="end">{r.label}</text>
              <rect x={160} y={r.y} width={(r.t.stores / D.direct.totals.stores) * 150 * f.tally!} height={14} fill="var(--global)" />
              <text x={160 + (r.t.stores / D.direct.totals.stores) * 150 * f.tally! + 6} y={r.y + 11} fontSize="11.5" fontWeight="600">
                {Math.round(r.t.stores * f.tally!)} {tr('stores', 'lần ghi')}
              </text>
              <text x={420} y={r.y + 11} fontSize="11.5">
                <Translated vi={<>{r.t.segments} đoạn, {r.t.bytesFetched} byte được di chuyển cho {r.t.bytesUseful} byte hữu ích</>}>
                  {r.t.segments} segments, {r.t.bytesFetched} bytes moved for {r.t.bytesUseful} useful
                </Translated>
              </text>
            </g>
          ))}
        </g>
      ) : null}
      {f.note ? (
        <text x={W / 2} y={H - 8} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{f.note}</text>
      ) : null}
    </svg>
  );
}

const beats: Beat[] = [
  {
    clip: 'stage-write-1',
    get label() { return tr('Sixteen keys', 'Mười sáu khóa'); },
    draw: (p) => {
      const shown = Math.round(ramp(p, 0.3, 0.9) * keptIdx.length);
      return <Stage offsets={shown} note={tr(`${keptIdx.length} kept keys; each one's offset is the number of kept keys before it`, `${keptIdx.length} khóa được giữ; độ lệch của mỗi khóa là số khóa được giữ đứng trước nó`)} data={{ offsets: shown }} />;
    },
  },
  {
    clip: 'stage-write-2',
    get label() { return tr('Direct scatter', 'Rải trực tiếp'); },
    draw: (p) => {
      const n = Math.round(ramp(p, 0.1, 0.85) * D.direct.stores.length);
      const segs = D.direct.stores.slice(0, n).reduce((a, s) => a + s.segments.length, 0);
      return (
        <Stage offsets={keptIdx.length} direct={n}
          note={tr(`store instructions so far: ${n}; segments touched: ${segs}`, `số lệnh ghi đến giờ: ${n}; số đoạn bị chạm: ${segs}`)}
          data={{ stores: n, segments: segs }} />
      );
    },
  },
  {
    clip: 'stage-write-3',
    get label() { return tr('Stage, then copy out', 'Gom tạm, rồi chép ra'); },
    draw: (p) => {
      const placed = Math.round(ramp(p, 0.05, 0.4) * keptIdx.length);
      const n = Math.round(ramp(p, 0.6, 0.95) * D.staged.stores.length);
      const segs = D.staged.stores.slice(0, n).reduce((a, s) => a + s.segments.length, 0);
      return (
        <Stage offsets={keptIdx.length} toShared={placed} barrier={p >= 0.42 && p < 0.6} staged={n}
          note={n ? tr(`store instructions so far: ${n}; segments touched: ${segs}`, `số lệnh ghi đến giờ: ${n}; số đoạn bị chạm: ${segs}`) : tr('each kept key goes to its block-local offset', 'mỗi khóa được giữ đi tới độ lệch cục bộ trong khối của nó')}
          data={{ stores: n, segments: segs, placed }} />
      );
    },
  },
  {
    clip: 'stage-write-4',
    get label() { return tr('The tally', 'Kết quả đếm'); },
    draw: (p) => (
      <Stage offsets={keptIdx.length} toShared={keptIdx.length} staged={D.staged.stores.length} faint tally={ramp(p, 0.05, 0.5)}
        data={{ before: D.direct.totals.stores, after: D.staged.totals.stores }} />
    ),
  },
  {
    clip: 'stage-write-5',
    get label() { return tr('Coarsen by two', 'Gộp việc gấp đôi'); },
    draw: (p) => {
      const placed = Math.round(ramp(p, 0.05, 0.4) * keptIdx.length);
      const n = Math.round(ramp(p, 0.55, 0.9) * D.coarse.stores.length);
      return (
        <Stage offsets={keptIdx.length} coarse toShared={placed} staged={n}
          note={tr(`${D.coarse.totals.chunks} chunk of ${D.coarse.totals.meanChunk} keys instead of ${D.staged.totals.chunks} of ${D.staged.totals.meanChunk}; one block-level scan serves ${D.coarse.totals.keysPerBlock} keys`, `${D.coarse.totals.chunks} đoạn gồm ${D.coarse.totals.meanChunk} khóa thay vì ${D.staged.totals.chunks} đoạn gồm ${D.staged.totals.meanChunk}; một phép scan cấp khối phục vụ ${D.coarse.totals.keysPerBlock} khóa`)}
          data={{ stores: n, chunks: D.coarse.totals.chunks, keys: D.coarse.totals.keysPerBlock }} />
      );
    },
  },
  {
    clip: 'stage-write-6',
    get label() { return tr('When staging pays', 'Khi nào gom tạm có lợi'); },
    draw: () => (
      <Stage offsets={keptIdx.length} direct={D.direct.stores.length}
        note={tr(`each block keeps ${D.staged.runs[0].length} keys for ${SCENE.blockDim} threads: CUB would scatter directly here`, `mỗi khối giữ ${D.staged.runs[0].length} khóa cho ${SCENE.blockDim} luồng: ở đây CUB sẽ rải trực tiếp`)}
        data={{ kept: D.staged.runs[0].length, threads: SCENE.blockDim }} />
    ),
  },
];

export function StageWriteScene() {
  return <Scene chapter="ch12" title={tr('Gather, then write together', 'Gom lại, rồi ghi cùng nhau')} beats={beats} />;
}
