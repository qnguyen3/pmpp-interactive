import type { ReactNode, SVGProps } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import {
  BM,
  BN,
  coarseThread,
  remapGeometry,
  remapThread,
  rowStrips,
  storeInstruction,
  type StorePattern,
  type StoreResult,
  type Width,
} from '../../sim/gemmremap';

// "From a block tile to one lane's four sub-tiles" (book §15.5, Fig. 15.10, pp. 360–361):
// the bad store of §15.3, the re-partition level by level, the same store replayed, and
// the 64-wide version. Every rectangle, address, sector and count is computed by
// src/sim/gemmremap.ts. Six beats, one narration clip each
// (content/ch15/narration/remap-N.txt).
const N = 1024; // columns of C: a multiple of 8, so rows start on a sector boundary
const SECTOR = 32; // NVIDIA's unit (ch06-facts A2)
const WARP = 6; // the warp Fig. 15.10 shades
const WAVE = 3; // its counterpart among four 64-wide wavefronts: bottom right
const LANE = 5; // the lane whose four squares the scene follows

const store = (pattern: StorePattern, W: Width, warp: number) =>
  storeInstruction({ pattern, W, N, warp, row: 0, col: 0, segmentBytes: SECTOR });
const SCALAR = store('scalar8x8', 32, WARP);
const FLOAT4 = store('float4-8x8', 32, WARP);
const REMAP = store('remap4x4', 32, WARP);
const WIDE = store('remap4x4', 64, WAVE);
const G32 = remapGeometry(32);
const G64 = remapGeometry(64);
const W6 = remapThread(WARP * 32, 32);
const L5 = remapThread(WARP * 32 + LANE, 32);

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const thousands = (v: number) => v.toLocaleString('en-US');

// ---------------------------------------------------------------- geometry
const TX = 20; // tile panel
const TY = 48;
const TS = 280;
const MX = 336; // memory panel
const MSTRIP = 392;
const PX_BYTE = 2.6 / 4; // 2.6 px per 4-byte word

interface View {
  row0: number;
  col0: number;
  size: number;
}
const FULL: View = { row0: 0, col0: 0, size: BM };
// Warp 6's 64 x 32 tile, centred in the panel.
const ZOOM: View = { row0: W6.wRow, col0: W6.wCol + G32.warpCols / 2 - G32.warpRows / 2, size: G32.warpRows };
const lerp = (a: View, b: View, z: number): View => ({
  row0: a.row0 + (b.row0 - a.row0) * z,
  col0: a.col0 + (b.col0 - a.col0) * z,
  size: a.size + (b.size - a.size) * z,
});

function at(v: View) {
  const s = TS / v.size;
  return {
    s,
    x: (col: number) => TX + (col - v.col0) * s,
    y: (row: number) => TY + (row - v.row0) * s,
  };
}

/** A rectangle of the block tile, rows r0.., columns c0.., in the current view. */
function Box({ v, r0, c0, rows, cols, ...rest }: { v: View; r0: number; c0: number; rows: number; cols: number } & SVGProps<SVGRectElement>) {
  const m = at(v);
  return <rect x={m.x(c0)} y={m.y(r0)} width={cols * m.s} height={rows * m.s} {...rest} />;
}

function Tile({ v, children, patches, warpTiles, waveTiles }: { v: View; children?: ReactNode; patches?: number; warpTiles?: number; waveTiles?: number }) {
  const m = at(v);
  const lines = (step: number, op: number, dash?: string) =>
    op > 0 ? (
      <g opacity={op}>
        {Array.from({ length: BM / step - 1 }, (_, k) => (k + 1) * step).map((r) => (
          <line key={'h' + r} x1={m.x(0)} x2={m.x(BN)} y1={m.y(r)} y2={m.y(r)} stroke="var(--ink-3)" strokeWidth="0.6" strokeDasharray={dash} />
        ))}
        {Array.from({ length: BN / step - 1 }, (_, k) => (k + 1) * step).map((c) => (
          <line key={'v' + c} y1={m.y(0)} y2={m.y(BM)} x1={m.x(c)} x2={m.x(c)} stroke="var(--ink-3)" strokeWidth="0.6" strokeDasharray={dash} />
        ))}
      </g>
    ) : null;
  return (
    <g>
      <defs>
        <clipPath id="remap-clip">
          <rect x={TX} y={TY} width={TS} height={TS} />
        </clipPath>
      </defs>
      <g clipPath="url(#remap-clip)">
        <rect x={m.x(0)} y={m.y(0)} width={BN * m.s} height={BM * m.s} fill="var(--cell)" stroke="var(--rule)" />
        {lines(8, patches ?? 0)}
        {children}
        {warpTiles ? (
          <g opacity={warpTiles}>
            {Array.from({ length: G32.warps }, (_, w) => {
              const t = remapThread(w * 32, 32);
              return (
                <g key={w} data-warp-tile={w}>
                  <Box v={v} r0={t.wRow} c0={t.wCol} rows={G32.warpRows} cols={G32.warpCols} fill="none" stroke="var(--ink)" strokeWidth="1.4" />
                  {m.s < 3 ? (
                    <text x={m.x(t.wCol + G32.warpCols / 2)} y={m.y(t.wRow + 6)} fontSize="11" textAnchor="middle"
                      stroke="var(--page)" strokeWidth="3" paintOrder="stroke" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>warp {w}</>}>warp {w}</Translated></text>
                  ) : null}
                </g>
              );
            })}
          </g>
        ) : null}
        {waveTiles ? (
          <g opacity={waveTiles}>
            {Array.from({ length: G64.warps }, (_, w) => {
              const t = remapThread(w * 64, 64);
              return (
                <g key={w} data-wave-tile={w}>
                  <Box v={v} r0={t.wRow} c0={t.wCol} rows={G64.warpRows} cols={G64.warpCols} fill="none" stroke="var(--ink)" strokeWidth="1.4" />
                  <text x={m.x(t.wCol + G64.warpCols / 2)} y={m.y(t.wRow + G64.warpRows / 2) + 4} fontSize="11" textAnchor="middle"
                    stroke="var(--page)" strokeWidth="3" paintOrder="stroke" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>wavefront {w}</>}>wavefront {w}</Translated></text>
                </g>
              );
            })}
          </g>
        ) : null}
      </g>
      <rect x={TX} y={TY} width={TS} height={TS} fill="none" stroke="var(--rule)" />
    </g>
  );
}

/** The elements one store instruction writes, drawn on the tile. */
function Written({ v, res, upto }: { v: View; res: StoreResult; upto: number }) {
  const m = at(v);
  const w = res.bytesPerLane / 4;
  return (
    <g data-written={res.lanes.length}>
      {res.lanes.slice(0, upto).map((l) =>
        m.s * w < 6 ? (
          <circle key={l.lane} cx={m.x(l.col + w / 2)} cy={m.y(l.row + 0.5)} r={w > 1 ? 3.2 : 2.6} fill="var(--global)" />
        ) : (
          <rect key={l.lane} x={m.x(l.col)} y={m.y(l.row)} width={w * m.s} height={Math.max(2, m.s)} fill="var(--global)" />
        ),
      )}
    </g>
  );
}

/** One instruction's bytes, one strip of memory per matrix row, cut into sectors. */
function Memory({ res, shown, counter, W }: { res: StoreResult; shown: number; counter: boolean; W: Width }) {
  const strips = rowStrips(res, SECTOR);
  const pitch = strips.length > 4 ? 28 : 40;
  return (
    <g data-sectors={res.transactions} data-useful={res.efficiency} data-rows={strips.length} data-width={W}>
      <text x={MX} y={TY + 4} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}>{tr('C in global memory: what this one instruction writes', 'C trong bộ nhớ toàn cục: những gì lệnh duy nhất này ghi')}</text>
      <text x={MX} y={TY + 20} fontSize="10" style={{ fill: 'var(--ink-3)' }}>{tr('one box per 32-byte sector touched; one strip per matrix row', 'mỗi sector 32 byte được chạm tới là một hộp; mỗi hàng ma trận là một dải')}</text>
      <g opacity={shown}>
        {strips.map((s, k) => {
          const y = TY + 40 + k * pitch;
          return (
            <g key={s.row} data-strip-row={s.row} data-strip-sectors={s.seg1 - s.seg0}>
              <text x={MX} y={y + 11} fontSize="10" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>hàng {s.row}</>}>row {s.row}</Translated></text>
              {s.used.map((u, j) => (
                <rect key={j} x={MSTRIP + j * SECTOR * PX_BYTE} y={y} width={SECTOR * PX_BYTE - 1} height={14}
                  fill="var(--global-wash)" stroke="var(--global)" strokeWidth="0.6" />
              ))}
              {s.pieces.map((pc) => (
                <rect key={pc.lane} x={MSTRIP + (pc.lo - s.seg0 * SECTOR) * PX_BYTE} y={y + 2} width={(pc.hi - pc.lo) * PX_BYTE - 0.6} height={10} fill="var(--global)" />
              ))}
            </g>
          );
        })}
      </g>
      {counter ? (
        <text x={MX} y={TY + TS - 2} fontSize="13" fontWeight="600" data-counter="">
          <Translated
            vi={
              <>
                {res.transactions} sector · dùng {thousands(res.bytesNeeded)} trên {thousands(res.bytesFetched)} byte · {Math.round(res.efficiency * 1000) / 10} %
              </>
            }
          >
            {res.transactions} sectors · {thousands(res.bytesNeeded)} of {thousands(res.bytesFetched)} bytes used · {Math.round(res.efficiency * 1000) / 10} %
          </Translated>
        </text>
      ) : null}
    </g>
  );
}

function Stage({ beat, headline, children }: { beat: number; headline: ReactNode; children: ReactNode }) {
  return (
    <svg viewBox="0 0 760 352" role="img" data-beat={beat} aria-label={tr('Animation of the output tile being remapped for coalesced stores', 'Hoạt hình ô đầu ra được ánh xạ lại để các lần ghi được hợp nhất')} style={{ width: '100%', maxWidth: 760, display: 'block' }}>
      <text x={TX} y={22} fontSize="13" fontWeight="600" data-headline="">{headline}</text>
      {children}
      <text x={TX} y={TY + TS + 18} fontSize="10" style={{ fill: 'var(--ink-3)' }}>
        <Translated vi={<>C rộng {thousands(N)} float và bắt đầu ở một ranh giới 256 byte; khối (0, 0); sector 32 byte của NVIDIA.</>}>
          C is {thousands(N)} floats wide and starts on a 256-byte boundary; block (0, 0); NVIDIA's 32-byte sectors.
        </Translated>
      </text>
    </svg>
  );
}

/** The patches of the warp's 32 threads in the §15.3 layout. */
function CoarseWarp({ v, warp }: { v: View; warp: number }) {
  return (
    <g>
      {Array.from({ length: 32 }, (_, l) => {
        const c = coarseThread(warp * 32 + l);
        return <Box key={l} v={v} r0={c.tRow} c0={c.tCol} rows={8} cols={8} fill="var(--reg-wash)" stroke="none" />;
      })}
    </g>
  );
}

/** The lane grid of one warp or wavefront: every lane's 4 x 4 square in every quadrant. */
function LaneSquares({ v, W, warp, upto, quads, labels, mark }: { v: View; W: Width; warp: number; upto: number; quads: number; labels?: boolean; mark?: number }) {
  const m = at(v);
  return (
    <g>
      {Array.from({ length: Math.min(upto, W) }, (_, lane) => {
        const t = remapThread(warp * W + lane, W);
        return t.subTiles.slice(0, quads).map((s) => (
          <g key={lane + '-' + s.qr + s.qc} data-lane-square={`${lane}:${s.row},${s.col}`}>
            <Box v={v} r0={s.row} c0={s.col} rows={4} cols={4}
              fill={lane === mark ? 'var(--reg)' : 'var(--reg-wash)'} stroke="var(--reg)" strokeWidth="0.5" />
            {labels ? (
              <text x={m.x(s.col + 2)} y={m.y(s.row + 2) + 3} fontSize="7.5" textAnchor="middle"
                style={{ fill: lane === mark ? 'var(--page)' : 'var(--ink-2)' }}>{lane}</text>
            ) : null}
          </g>
        ));
      })}
    </g>
  );
}

function Quadrants({ v, W, warp, op }: { v: View; W: Width; warp: number; op: number }) {
  const g = remapGeometry(W);
  const t = remapThread(warp * W, W);
  return (
    <g opacity={op}>
      {[0, 1].flatMap((qr) => [0, 1].map((qc) => (
        <Box key={`${qr}${qc}`} v={v} r0={t.wRow + qr * g.QR} c0={t.wCol + qc * g.QC} rows={g.QR} cols={g.QC}
          fill="none" stroke="var(--ink-2)" strokeWidth="1" strokeDasharray="4 3" />
      )))}
    </g>
  );
}

function InfoLines({ lines, y0 = TY + 44 }: { lines: ReactNode[]; y0?: number }) {
  return (
    <g>
      {lines.map((l, k) => (
        <text key={k} x={MX} y={y0 + k * 22} fontSize="12">{l}</text>
      ))}
    </g>
  );
}

const beats: Beat[] = [
  {
    clip: 'remap-1',
    get label() {
      return tr('8 × 8, scalar', '8 × 8, vô hướng');
    },
    draw: (p) => (
      <Stage beat={1} headline={tr('§15.3: each thread owns an 8 × 8 patch; warp 6 stores one float per lane', '§15.3: mỗi luồng sở hữu một mảng 8 × 8; warp 6 ghi một float mỗi lane')}>
        <Tile v={FULL} patches={1}>
          <CoarseWarp v={FULL} warp={WARP} />
          <Written v={FULL} res={SCALAR} upto={Math.floor(ramp(p, 0.1, 0.5) * 32)} />
        </Tile>
        <Memory res={SCALAR} shown={ramp(p, 0.5, 0.72)} counter={p > 0.74} W={32} />
      </Stage>
    ),
  },
  {
    clip: 'remap-2',
    get label() {
      return tr('float4', 'float4');
    },
    draw: (p) => (
      <Stage beat={2} headline={tr('Exercise 2: a float4 store per lane, a quarter of the instructions', 'Bài tập 2: mỗi lane một lần ghi float4, chỉ còn một phần tư số lệnh')}>
        <Tile v={FULL} patches={1}>
          <CoarseWarp v={FULL} warp={WARP} />
          <Written v={FULL} res={FLOAT4} upto={Math.floor(ramp(p, 0.1, 0.45) * 32)} />
        </Tile>
        <Memory res={FLOAT4} shown={ramp(p, 0.45, 0.7)} counter={p > 0.72} W={32} />
      </Stage>
    ),
  },
  {
    clip: 'remap-3',
    get label() {
      return tr('Warp tiles', 'Ô của warp');
    },
    draw: (p) => {
      const v = lerp(FULL, ZOOM, ramp(p, 0.58, 0.95));
      return (
        <Stage beat={3} headline={tr(`8 warps as ${G32.WR} × ${G32.WC} warp tiles of ${G32.warpRows} × ${G32.warpCols}`, `8 warp thành ${G32.WR} × ${G32.WC} ô warp cỡ ${G32.warpRows} × ${G32.warpCols}`)}>
          <Tile v={v} patches={1 - ramp(p, 0.05, 0.3)} warpTiles={ramp(p, 0.05, 0.3)}>
            <g opacity={ramp(p, 0.3, 0.5)}>
              <Box v={v} r0={W6.wRow} c0={W6.wCol} rows={G32.warpRows} cols={G32.warpCols} fill="var(--reg-wash)" />
            </g>
          </Tile>
          <InfoLines lines={[
            tr(`block tile ${BM} × ${BN}, ${G32.warps} warps of 32 lanes`, `ô của khối ${BM} × ${BN}, ${G32.warps} warp, mỗi warp 32 lane`),
            tr(`warp tile ${G32.warpRows} rows × ${G32.warpCols} columns`, `ô warp ${G32.warpRows} hàng × ${G32.warpCols} cột`),
            tr(`warp 6 starts at row ${W6.wRow}, column ${W6.wCol}`, `warp 6 bắt đầu ở hàng ${W6.wRow}, cột ${W6.wCol}`),
            tr('same 8 × 8 outputs per thread as before', 'vẫn 8 × 8 đầu ra mỗi luồng như trước'),
          ]} />
        </Stage>
      );
    },
  },
  {
    clip: 'remap-4',
    get label() {
      return tr('Quadrants', 'Góc phần tư');
    },
    draw: (p) => {
      const upto = p < 0.2 ? 0 : Math.ceil(ramp(p, 0.2, 0.55) * 32);
      const quads = p < 0.58 ? 1 : 1 + Math.floor(ramp(p, 0.58, 0.82) * 3);
      return (
        <Stage beat={4} headline={tr(`4 quadrants of ${G32.QR} × ${G32.QC}; lanes 8 × ${G32.LC}; one 4 × 4 square per lane per quadrant`, `4 góc phần tư cỡ ${G32.QR} × ${G32.QC}; lane xếp 8 × ${G32.LC}; mỗi lane một ô vuông 4 × 4 trong mỗi góc phần tư`)}>
          <Tile v={ZOOM} warpTiles={1}>
            <Box v={ZOOM} r0={W6.wRow} c0={W6.wCol} rows={G32.warpRows} cols={G32.warpCols} fill="var(--page)" />
            <LaneSquares v={ZOOM} W={32} warp={WARP} upto={upto} quads={quads} labels mark={p > 0.84 ? LANE : undefined} />
            <Quadrants v={ZOOM} W={32} warp={WARP} op={ramp(p, 0, 0.18)} />
          </Tile>
          <g opacity={ramp(p, 0.84, 0.95)}>
            <InfoLines lines={[
              tr(`lane ${LANE}: lane row ${L5.lRow / 4}, lane column ${L5.lCol / 4}`, `lane ${LANE}: hàng lane ${L5.lRow / 4}, cột lane ${L5.lCol / 4}`),
              ...L5.subTiles.map((s) => tr(`quadrant (${s.qr}, ${s.qc}): rows ${s.row}–${s.row + 3}, columns ${s.col}–${s.col + 3}`, `góc phần tư (${s.qr}, ${s.qc}): hàng ${s.row}–${s.row + 3}, cột ${s.col}–${s.col + 3}`)),
              tr('4 squares of 16 = a logical 8 × 8 tile: 64 registers', '4 ô vuông 16 phần tử = một ô logic 8 × 8: 64 thanh ghi'),
            ]} />
          </g>
        </Stage>
      );
    },
  },
  {
    clip: 'remap-5',
    get label() {
      return tr('The store again', 'Lại lần ghi đó');
    },
    draw: (p) => {
      const rowsShown = Math.ceil(ramp(p, 0.05, 0.42) * 8);
      return (
        <Stage beat={5} headline={tr(`§15.5: each lane row of ${G32.LC} writes ${G32.laneRowBytes} contiguous bytes`, `§15.5: mỗi hàng ${G32.LC} lane ghi ${G32.laneRowBytes} byte liên tiếp`)}>
          <Tile v={ZOOM} warpTiles={1}>
            <Box v={ZOOM} r0={W6.wRow} c0={W6.wCol} rows={G32.warpRows} cols={G32.warpCols} fill="var(--page)" />
            <LaneSquares v={ZOOM} W={32} warp={WARP} upto={32} quads={4} />
            <Quadrants v={ZOOM} W={32} warp={WARP} op={1} />
            <Written v={ZOOM} res={REMAP} upto={rowsShown * G32.LC} />
          </Tile>
          <Memory res={REMAP} shown={ramp(p, 0.42, 0.66)} counter={p > 0.68} W={32} />
        </Stage>
      );
    },
  },
  {
    clip: 'remap-6',
    get label() {
      return tr('64 lanes', '64 lane');
    },
    draw: (p) => {
      const v = lerp(ZOOM, FULL, ramp(p, 0, 0.22));
      const wave = remapThread(WAVE * 64, 64);
      return (
        <Stage beat={6} headline={tr(`64 lanes: ${G64.warps} wavefronts as ${G64.WR} × ${G64.WC} tiles of ${G64.warpRows} × ${G64.warpCols}; lane rows of ${G64.LC} write ${G64.laneRowBytes} bytes`, `64 lane: ${G64.warps} wavefront thành ${G64.WR} × ${G64.WC} ô cỡ ${G64.warpRows} × ${G64.warpCols}; mỗi hàng ${G64.LC} lane ghi ${G64.laneRowBytes} byte`)}>
          <Tile v={v} warpTiles={1 - ramp(p, 0.15, 0.3)} waveTiles={ramp(p, 0.22, 0.38)}>
            <g opacity={ramp(p, 0.3, 0.45)}>
              <Box v={v} r0={wave.wRow} c0={wave.wCol} rows={G64.warpRows} cols={G64.warpCols} fill="var(--page)" />
              <LaneSquares v={v} W={64} warp={WAVE} upto={64} quads={4} />
              <Quadrants v={v} W={64} warp={WAVE} op={1} />
            </g>
            <Written v={v} res={WIDE} upto={Math.ceil(ramp(p, 0.45, 0.62) * 8) * G64.LC} />
          </Tile>
          <Memory res={WIDE} shown={ramp(p, 0.62, 0.8)} counter={p > 0.82} W={64} />
        </Stage>
      );
    },
  },
];

export function RemapScene() {
  return <Scene chapter="ch15" title={tr("From a block tile to one lane's four sub-tiles", 'Từ ô của khối đến bốn ô con của một lane')} beats={beats} />;
}
