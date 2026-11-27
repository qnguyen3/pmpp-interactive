import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import {
  CORRECT,
  SWAP_ORDER,
  TOO_LARGE,
  coopLoop,
  gridGraph,
  gridSummary,
  hostStrip,
  residency,
  simulate,
  strideAssignment,
  treeSummary,
} from '../../sim/gridbfs';
import { Translated, tr } from '../../lib/i18n';

// "One launch for every level" (section 18.6, book §18.7 pp. 445-448). Every drawing is
// computed from src/sim/gridbfs.ts, the model behind the section's "Three grid barriers"
// stepper and "Does it fit?" tab: frontier sizes from its formulas, the host loop as a strip
// in units of vertex work, the stride assignment, the residency model and the swap's event
// order from a simulated run. Time is drawn in levels and launches, never in seconds. Six
// beats, one narration clip each (content/ch18/narration/one-launch-N.txt).

/** The round trip, drawn as the work of this many vertices: an assumption for the picture, the same in every strip. */
export const SCENE_GAP = 10;
const ROAD_W = 16;
const ROAD = gridSummary(ROAD_W);
const SOCIAL = treeSummary(8, 4);
const ROAD_STRIP = hostStrip(ROAD.sizes, SCENE_GAP);
const SOCIAL_STRIP = hostStrip(SOCIAL.sizes, SCENE_GAP);
const COOP = coopLoop(ROAD.sizes, 3);
const BLOCKS = 3;
const THREADS = 2;
const G = BLOCKS * THREADS;
/** A level of the road grid with more elements than threads, so that some threads take two or three. */
export const STRIDE_LEVEL = 13;
const STRIDE = strideAssignment(ROAD.sizes[STRIDE_LEVEL], G);
const FIT = residency(2, 2, 6);
/** One correct level of a small search, for the swap's order of events. */
const RUN = simulate(CORRECT, 'forward', gridGraph(4), 2, 2);
export const SWAP_LEVEL = 3;
const LEVEL_EVENTS = RUN.history.filter((e) => e.level === SWAP_LEVEL);

export const SCENE_DATA = {
  road: ROAD, social: SOCIAL, roadStrip: ROAD_STRIP, socialStrip: SOCIAL_STRIP, coop: COOP,
  stride: STRIDE, fit: FIT, levelEvents: LEVEL_EVENTS,
};

const W = 720;
const H = 340;
const X0 = 24;
const X1 = 696;
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

function Svg({ label, children }: { label: string; children: ReactNode }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      {children}
    </svg>
  );
}
const T = ({ x, y, children, size = 12, anchor = 'start', weight, muted }: {
  x: number; y: number; children: ReactNode; size?: number; anchor?: 'start' | 'middle' | 'end'; weight?: number; muted?: boolean;
}) => (
  <text x={x} y={y} fontSize={size} textAnchor={anchor} fontWeight={weight} style={{ fill: muted ? 'var(--ink-2)' : 'var(--ink)' }}>{children}</text>
);

/** A host-loop strip: kernel bars and dashed round trips, revealed up to `upto` (0..1 of its length). */
function Strip({ y, strip, upto, label, h = 22 }: { y: number; strip: ReturnType<typeof hostStrip>; upto: number; label: string; h?: number }) {
  const scale = (X1 - X0) / strip.total;
  const end = upto * strip.total;
  return (
    <g>
      <T x={X0} y={y - 7} size={12} weight={600}>{label}</T>
      <rect x={X0} y={y} width={X1 - X0} height={h} fill="none" stroke="var(--rule-soft)" />
      {strip.parts.filter((p) => p.from < end).map((p, i) => {
        const w = Math.max(p.kind === 'kernel' ? 0.8 : 0.5, Math.min(p.len, end - p.from) * scale);
        return p.kind === 'kernel'
          ? <rect key={i} data-part="kernel" x={X0 + p.from * scale} y={y} width={w} height={h} fill="var(--ink-2)" />
          : <rect key={i} data-part="gap" x={X0 + p.from * scale} y={y + 3} width={w} height={h - 6} fill="none" stroke="var(--ink-3)" strokeDasharray="2 2" />;
      })}
    </g>
  );
}

/** The 16 x 16 road grid, coloured up to `level`, with the frontier at `level` marked. */
function RoadGrid({ x, y, level }: { x: number; y: number; level: number }) {
  const c = 8.5;
  const cells: ReactNode[] = [];
  for (let r = 0; r < ROAD_W; r++)
    for (let q = 0; q < ROAD_W; q++) {
      const l = r + q;
      cells.push(
        <rect key={r * ROAD_W + q} x={x + q * c} y={y + r * c} width={c - 1} height={c - 1}
          fill={l === level ? 'var(--ink)' : l < level ? 'var(--ink-3)' : 'var(--cell)'} stroke="var(--rule-soft)" strokeWidth="0.5" />,
      );
    }
  return <g>{cells}</g>;
}

function hostBeat(p: number): ReactNode {
  const upto = ramp(p, 0.05, 0.95);
  const kernelsShown = ROAD_STRIP.parts.filter((q) => q.kind === 'kernel' && q.from <= upto * ROAD_STRIP.total);
  const level = Math.max(0, kernelsShown.length - 1);
  const size = ROAD.sizes[level];
  return (
    <Svg label={tr('The per-level host loop on a road-like grid', 'Vòng lặp host theo từng mức trên một lưới giống mạng đường')}>
      <RoadGrid x={X0} y={28} level={level} />
      <T x={190} y={44} size={13} weight={600}>{tr('Host loop, one launch per level', 'Vòng lặp host, một lần khởi chạy cho mỗi mức')}</T>
      <T x={190} y={68}>{tr(`launch ${kernelsShown.length} of ${ROAD.launches}: a frontier of ${size} ${size === 1 ? 'vertex' : 'vertices'}`, `lần khởi chạy ${kernelsShown.length} trên ${ROAD.launches}: một frontier gồm ${size} đỉnh`)}</T>
      <T x={190} y={88} muted>{tr(`launches ${kernelsShown.length}, read-backs of the size ${kernelsShown.length}`, `số lần khởi chạy ${kernelsShown.length}, số lần đọc lại kích thước ${kernelsShown.length}`)}</T>
      <T x={190} y={112} muted>{tr(`a ${ROAD_W} × ${ROAD_W} grid from a corner: ${ROAD.levels} levels, never more than ${ROAD.maxFrontier} vertices in one`, `một lưới ${ROAD_W} × ${ROAD_W} tính từ một góc: ${ROAD.levels} mức, không mức nào quá ${ROAD.maxFrontier} đỉnh`)}</T>
      <Strip y={196} strip={ROAD_STRIP} upto={upto} label={tr('each bar a launch, as long as its frontier; each dashed box a round trip', 'mỗi thanh là một lần khởi chạy, dài bằng frontier của nó; mỗi ô nét đứt là một vòng khứ hồi')} />
      <T x={X0} y={246} muted>{tr(`round trips: ${Math.round(ROAD_STRIP.gapShare * 100)}% of the strip`, `vòng khứ hồi: ${Math.round(ROAD_STRIP.gapShare * 100)}% của dải`)}</T>
      <T x={X0} y={298} size={11} muted>{tr(`A round trip is drawn as the work of ${SCENE_GAP} vertices. That is an assumption for the picture;`, `Một vòng khứ hồi được vẽ bằng khối lượng việc của ${SCENE_GAP} đỉnh. Đó là một giả định cho hình vẽ;`)}</T>
      <T x={X0} y={314} size={11} muted>{tr("the lab's floor row measures what one costs on your device.", 'hàng floor của lab đo một vòng tốn bao nhiêu trên thiết bị của bạn.')}</T>
    </Svg>
  );
}

function socialBeat(p: number): ReactNode {
  const upto = ramp(p, 0.08, 0.85);
  return (
    <Svg label={tr('The same host loop for a graph whose levels grow fast', 'Cùng vòng lặp host cho một đồ thị có các mức tăng nhanh')}>
      <Strip y={52} strip={ROAD_STRIP} upto={1} label={tr(`a road-like grid: ${ROAD.launches} launches`, `một lưới giống mạng đường: ${ROAD.launches} lần khởi chạy`)} h={18} />
      <T x={X0} y={92} muted>{tr(`round trips: ${Math.round(ROAD_STRIP.gapShare * 100)}% of the strip`, `vòng khứ hồi: ${Math.round(ROAD_STRIP.gapShare * 100)}% của dải`)}</T>
      <Strip y={152} strip={SOCIAL_STRIP} upto={upto} label={tr(`a graph that fans out 8 ways per hop: ${SOCIAL.launches} launches, frontiers up to ${SOCIAL.maxFrontier.toLocaleString('en-US')}`, `một đồ thị tỏa ra 8 hướng mỗi bước: ${SOCIAL.launches} lần khởi chạy, frontier lên tới ${SOCIAL.maxFrontier.toLocaleString('en-US')}`)} />
      <T x={X0} y={200} muted>{upto > 0.98 ? tr(`round trips: ${(SOCIAL_STRIP.gapShare * 100).toFixed(1)}% of the strip`, `vòng khứ hồi: ${(SOCIAL_STRIP.gapShare * 100).toFixed(1)}% của dải`) : ''}</T>
      <T x={X0} y={252} size={13} weight={600}>{p > 0.6 ? tr('Big frontiers: launch overhead is negligible. Small ones: it may dominate (p. 445).', 'Frontier lớn: chi phí khởi chạy không đáng kể. Frontier nhỏ: nó có thể chiếm phần lớn (p. 445).') : ''}</T>
      <T x={X0} y={298} size={11} muted>{tr(`Both strips are drawn to the same rule: a bar per launch as long as its frontier, a round trip as ${SCENE_GAP} vertices of work.`, `Cả hai dải được vẽ theo cùng một quy tắc: mỗi lần khởi chạy một thanh dài bằng frontier của nó, một vòng khứ hồi bằng khối lượng việc của ${SCENE_GAP} đỉnh.`)}</T>
    </Svg>
  );
}

function strideBeat(p: number): ReactNode {
  const n = ROAD.sizes[STRIDE_LEVEL];
  const shown = Math.ceil(ramp(p, 0.05, 0.5) * n);
  const cw = 44;
  return (
    <Svg label={tr('One cooperative launch, the frontier dealt out by stride', 'Một lần khởi chạy cooperative, frontier được chia theo bước nhảy (stride)')}>
      <T x={X0} y={26} size={13} weight={600}>{tr(`One launch: ${BLOCKS} resident blocks of ${THREADS} threads, ${G} threads for every level`, `Một lần khởi chạy: ${BLOCKS} khối thường trú, mỗi khối ${THREADS} luồng, ${G} luồng cho mọi mức`)}</T>
      <T x={X0} y={46} muted>{tr(`level ${STRIDE_LEVEL} of the grid has ${n} elements: thread r takes r, r + ${G}, r + ${2 * G}, ...`, `mức ${STRIDE_LEVEL} của lưới có ${n} phần tử: luồng r nhận r, r + ${G}, r + ${2 * G}, ...`)}</T>
      {STRIDE.map((els, r) => (
        <g key={r}>
          <T x={X0} y={78 + r * 22} size={11} muted>{tr(`thread ${r} (block ${Math.floor(r / THREADS)})`, `luồng ${r} (khối ${Math.floor(r / THREADS)})`)}</T>
          {els.map((i, k) => (
            <g key={i} opacity={i < shown ? 1 : 0.12}>
              <rect data-element={i} data-owner={r} x={150 + k * (cw + 4)} y={64 + r * 22} width={cw} height={18} rx={2} fill="var(--cell)" stroke="var(--ink-2)" />
              <T x={150 + k * (cw + 4) + cw / 2} y={77 + r * 22} size={11} anchor="middle">{i}</T>{/* i18n-ignore: SVG text-anchor value */}
            </g>
          ))}
        </g>
      ))}
      <Strip y={232} strip={ROAD_STRIP} upto={1} label={tr(`host loop: ${ROAD.launches} launches, ${ROAD.launches} read-backs`, `vòng lặp host: ${ROAD.launches} lần khởi chạy, ${ROAD.launches} lần đọc lại`)} h={16} />
      <g opacity={ramp(p, 0.5, 0.7)}>
        <T x={X0} y={284} size={12} weight={600}>{tr(`one cooperative launch: no read-backs, ${COOP.gridBarriers} grid barriers (3 per level)`, `một lần khởi chạy cooperative: không đọc lại, ${COOP.gridBarriers} rào chắn lưới (3 mỗi mức)`)}</T>
        <rect x={X0} y={292} width={X1 - X0} height={16} fill="none" stroke="var(--rule-soft)" />
        {(() => {
          const total = ROAD.sizes.reduce((a, b) => a + b, 0);
          const scale = (X1 - X0) / total;
          let x = X0;
          return ROAD.sizes.map((s, k) => {
            const r = <g key={k}><rect x={x} y={292} width={Math.max(0.8, s * scale - 1)} height={16} fill="var(--ink-2)" /><line x1={x + s * scale - 0.5} x2={x + s * scale - 0.5} y1={288} y2={312} stroke="var(--ink)" strokeWidth="1" /></g>;
            x += s * scale;
            return r;
          });
        })()}
      </g>
      <T x={X0} y={330} size={11} muted>{p > 0.75 ? tr('The CPU is free, and data can stay in shared memory or registers from one level to the next (p. 448).', 'CPU được rảnh, và dữ liệu có thể ở lại trong bộ nhớ chia sẻ (shared memory) hoặc thanh ghi (register) từ mức này sang mức sau (p. 448).') : ''}</T>
    </Svg>
  );
}

function fitBeat(p: number): ReactNode {
  const arrive = ramp(p, 0.08, 0.35);
  const frozen = p > 0.4;
  const refused = ramp(p, 0.62, 0.75);
  const slotW = 70;
  return (
    <Svg label={tr('A grid with more blocks than resident slots', 'Một lưới có nhiều khối hơn số chỗ thường trú')}>
      <T x={X0} y={26} size={13} weight={600}>{tr(`${FIT.grid} blocks, ${FIT.slots} resident slots (2 SMs × 2 blocks each)`, `${FIT.grid} khối, ${FIT.slots} chỗ thường trú (2 SM × 2 khối mỗi SM)`)}</T>
      {[0, 1].map((sm) => (
        <g key={sm}>
          <rect x={X0 + sm * 180} y={44} width={168} height={92} rx={4} fill="none" stroke="var(--ink-2)" />
          <T x={X0 + 8 + sm * 180} y={60} size={11} muted>{`SM ${sm}`}</T>{/* i18n-ignore: hardware unit name and index */}
          {[0, 1].map((s) => {
            const b = sm * 2 + s;
            return (
              <g key={s} opacity={b < FIT.resident ? 1 : 0}>
                <rect data-resident={b} x={X0 + 10 + sm * 180 + s * (slotW + 8)} y={68 + (1 - arrive) * 0} width={slotW} height={28} rx={3} fill="var(--cell)" stroke="var(--ink)" />
                <T x={X0 + 10 + sm * 180 + s * (slotW + 8) + slotW / 2} y={86} size={11} anchor="middle">{tr(`block ${b}`, `khối ${b}`)}</T>{/* i18n-ignore: SVG text-anchor value */}
              </g>
            );
          })}
          <line x1={X0 + 6 + sm * 180} x2={X0 + 162 + sm * 180} y1={112} y2={112} stroke="var(--ink)" strokeWidth="2" opacity={arrive} />
        </g>
      ))}
      <T x={X0} y={154} muted>{arrive > 0.99 ? tr('grid barrier: blocks 0 to 3 have arrived and wait for 4 and 5', 'rào chắn lưới: các khối 0 đến 3 đã tới và chờ khối 4 và 5') : tr('blocks 0 to 3 run to the grid barrier', 'các khối 0 đến 3 chạy tới rào chắn lưới')}</T>
      {Array.from({ length: FIT.pending }, (_, k) => FIT.resident + k).map((b, k) => (
        <g key={b}>
          <rect data-pending={b} x={420 + k * 84} y={68} width={slotW} height={28} rx={3} fill="none" stroke="var(--ink-3)" strokeDasharray="3 3" />
          <T x={420 + k * 84 + slotW / 2} y={86} size={11} anchor="middle" muted>{tr(`block ${b}`, `khối ${b}`)}</T>{/* i18n-ignore: SVG text-anchor value */}
        </g>
      ))}
      <T x={420} y={118} size={11} muted>{tr('waiting for a slot, which no block frees', 'chờ một chỗ trống mà không khối nào giải phóng')}</T>
      {frozen ? <T x={420} y={140} size={12} weight={600}>{refused < 0.5 ? tr('never completes', 'không bao giờ hoàn tất') : ''}</T> : null}
      <g opacity={refused}>
        <rect x={X0} y={176} width={X1 - X0} height={112} rx={4} fill="var(--fail-wash)" stroke="var(--fail)" />
        <T x={X0 + 12} y={198} size={12} weight={600}>{tr('Under a cooperative launch the same grid is refused at launch:', 'Với khởi chạy cooperative, cùng lưới đó bị từ chối ngay lúc khởi chạy:')}</T>
        <T x={X0 + 12} y={220} size={12}>{tr(`CUDA  cudaLaunchCooperativeKernel returns ${TOO_LARGE.cuda}`, `CUDA  cudaLaunchCooperativeKernel trả về ${TOO_LARGE.cuda}`)}</T>
        <T x={X0 + 12} y={240} size={12}>{tr(`HIP   hipLaunchCooperativeKernel returns ${TOO_LARGE.hip}`, `HIP   hipLaunchCooperativeKernel trả về ${TOO_LARGE.hip}`)}</T>
        <T x={X0 + 12} y={260} size={12}>{tr('Metal has no cooperative launch and no grid barrier', 'Metal không có khởi chạy cooperative và không có rào chắn lưới')}</T>
        <T x={X0 + 12} y={280} size={11} muted>{tr(`Size it from the occupancy query: SMs × blocks per SM = ${FIT.slots} blocks.`, `Định kích thước từ truy vấn occupancy (mức chiếm dụng): số SM × số khối mỗi SM = ${FIT.slots} khối.`)}</T>
      </g>
      <T x={X0} y={318} size={11} muted>{tr("Ordinary launch: the book's argument (p. 445). What grid.sync() does there is not documented.", 'Khởi chạy thông thường: lập luận của sách (p. 445). grid.sync() làm gì trong trường hợp đó thì không được tài liệu hóa.')}</T>
    </Svg>
  );
}

function swapBeat(p: number): ReactNode {
  const rows = SWAP_ORDER.length;
  const at = Math.min(rows - 1, Math.floor(ramp(p, 0.02, 0.7) * rows));
  const ex3 = ramp(p, 0.74, 0.86);
  const appends = LEVEL_EVENTS.filter((e) => e.kind === 'append').length;
  const reads = LEVEL_EVENTS.filter((e) => e.kind === 'read').length;
  return (
    <Svg label={tr('The three grid barriers of the frontier swap', 'Ba rào chắn lưới của bước hoán đổi frontier')}>
      <T x={X0} y={24} size={13} weight={600}>{tr('The end of each level (Fig. 18.17, lines 65 to 74)', 'Cuối mỗi mức (Fig. 18.17, các dòng 65 đến 74)')}</T>
      {SWAP_ORDER.map((r, k) => {
        const y = 40 + k * 32;
        const sync = r.guard !== null;
        return (
          <g key={k} opacity={k <= at ? 1 : 0.25} data-swap-row={k}>
            <rect x={X0} y={y} width={300} height={26} rx={3} fill={sync ? 'var(--ink)' : 'var(--cell)'} stroke="var(--ink-2)" />
            <text x={X0 + 10} y={y + 17} fontSize="12" style={{ fill: sync ? 'var(--page)' : 'var(--ink)' }}><Translated vi={['mỗi luồng nối thêm các đỉnh mới của nó', 'grid.sync() 1', 'mỗi luồng đọc *numCurrFrontier', 'grid.sync() 2', 'luồng 0 của lưới đặt nó về 0', 'grid.sync() 3', 'mỗi luồng hoán đổi hai con trỏ của nó'][k]}>{r.what}</Translated></text>
            {sync ? <T x={X0 + 312} y={y + 17} size={12} muted={k !== at}><Translated vi={[null, 'không ai đọc kích thước trước khi mọi lần nối thêm xong', null, 'không ai đọc sau khi đặt lại', null, 'không lần nối thêm nào bắt đầu trước khi đặt lại', null][k]}>{r.guard}</Translated></T> : null}
          </g>
        );
      })}
      <T x={X0} y={284} size={11} muted>{tr('In a run of the model (4 × 4 grid, 4 threads),', 'Trong một lần chạy mô hình (lưới 4 × 4, 4 luồng),')}</T>
      <T x={X0} y={300} size={11} muted>{tr(`level ${SWAP_LEVEL}: ${appends} appends, then ${reads} reads, then the reset.`, `mức ${SWAP_LEVEL}: ${appends} lần nối thêm, rồi ${reads} lần đọc, rồi đặt lại.`)}</T>
      <g opacity={ex3}>
        <rect x={430} y={226} width={266} height={96} rx={4} fill="var(--page)" stroke="var(--ink-2)" />
        <T x={442} y={246} size={12} weight={600}>{tr('Exercise 3: a counter per level', 'Bài tập 3: một bộ đếm cho mỗi mức')}</T>
        <T x={442} y={266} size={11}>{tr('append to numFrontier[L]', 'nối thêm vào numFrontier[L]')}</T>
        <T x={442} y={284} size={11}>{tr('grid.sync(): the only barrier', 'grid.sync(): rào chắn duy nhất')}</T>
        <T x={442} y={302} size={11}>{tr('read numFrontier[L], swap. Nothing is reset.', 'đọc numFrontier[L], hoán đổi. Không có gì bị đặt lại.')}</T>
      </g>
    </Svg>
  );
}

function backendsBeat(p: number): ReactNode {
  const show = (a: number) => ramp(p, a, a + 0.12);
  const col = (x: number, title: string, lines: string[], o: number) => (
    <g opacity={o}>
      <rect x={x} y={34} width={214} height={150} rx={4} fill="var(--page)" stroke="var(--ink-2)" />
      <T x={x + 10} y={54} size={13} weight={600}>{title}</T>
      {lines.map((l, k) => <T key={k} x={x + 10} y={78 + k * 20} size={11}>{l}</T>)}
    </g>
  );
  const upto = ramp(p, 0.55, 0.95);
  const total = ROAD.sizes.reduce((a, b) => a + b, 0) + ROAD.sizes.length * 2;
  const scale = (X1 - X0) / total;
  let x = X0;
  return (
    <Svg label={tr('The same idea on HIP and Metal', 'Cùng ý tưởng trên HIP và Metal')}>
      {col(X0, 'CUDA', ['<cooperative_groups.h>', 'cudaLaunchCooperativeKernel', tr('  4 arguments compile', '  4 đối số vẫn biên dịch được'), 'grid.sync()', tr('ranks are 64-bit', 'rank là 64-bit')], show(0))}
      {col(253, 'HIP', ['<hip/hip_cooperative_groups.h>', 'hipLaunchCooperativeKernel', tr('  all 6 arguments', '  đủ cả 6 đối số'), tr('grid.sync(), same name', 'grid.sync(), cùng tên'), tr('ranks are 32-bit', 'rank là 32-bit')], show(0.12))}
      {col(482, 'Metal', [tr('no grid barrier', 'không có rào chắn lưới'), tr('no cooperative launch', 'không có khởi chạy cooperative'), tr('one dispatch per level', 'một dispatch cho mỗi mức'), tr('serial encoder: in order', 'encoder tuần tự: đúng thứ tự'), tr('size from a GPU buffer', 'kích thước lấy từ một buffer GPU')], show(0.26))}
      <g opacity={ramp(p, 0.5, 0.6)}>
        <T x={X0} y={214} size={12} weight={600}>{tr("Metal: a dispatch per level, each sized by an indirect dispatch from the previous level's count", 'Metal: một dispatch cho mỗi mức, mỗi dispatch được định kích thước bằng một indirect dispatch từ số đếm của mức trước')}</T>
        {ROAD.sizes.map((s, k) => {
          const x0 = x;
          x += s * scale + 2 * scale;
          return x0 < X0 + upto * (X1 - X0)
            ? <rect key={k} data-dispatch={k} x={x0} y={226} width={Math.max(0.8, s * scale)} height={18} fill="var(--ink-2)" />
            : null;
        })}
        <T x={X0} y={266} size={11} muted>{tr('No CPU read between dispatches. A threadgroup count of 0 is not documented: write at least 1 and return early.', 'CPU không đọc gì giữa các dispatch. Số threadgroup bằng 0 không được tài liệu hóa: hãy ghi ít nhất 1 và return sớm.')}</T>
      </g>
      <T x={X0} y={312} size={12} weight={600}>{p > 0.85 ? tr('Everything survives except the single launch itself.', 'Mọi thứ đều giữ được, trừ chính lần khởi chạy duy nhất.') : ''}</T>
    </Svg>
  );
}

const beats: Beat[] = [
  { clip: 'one-launch-1', get label() { return tr('The host loop', 'Vòng lặp host'); }, draw: hostBeat },
  { clip: 'one-launch-2', get label() { return tr('Big frontiers', 'Frontier lớn'); }, draw: socialBeat },
  { clip: 'one-launch-3', get label() { return tr('One grid, every level', 'Một lưới, mọi mức'); }, draw: strideBeat },
  { clip: 'one-launch-4', get label() { return tr('Why it must all fit', 'Vì sao tất cả phải vừa'); }, draw: fitBeat },
  { clip: 'one-launch-5', get label() { return tr('Three barriers, three jobs', 'Ba rào chắn, ba nhiệm vụ'); }, draw: swapBeat },
  { clip: 'one-launch-6', get label() { return tr('On HIP and Metal', 'Trên HIP và Metal'); }, draw: backendsBeat },
];

export function OneLaunchScene() {
  return <Scene chapter="ch18" title={tr('One launch for every level', 'Một lần khởi chạy cho mọi mức')} beats={beats} />;
}
