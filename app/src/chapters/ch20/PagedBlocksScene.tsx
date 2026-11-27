import { Scene, type Beat } from '../../components/Scene';
import { firstReuse, isScattered, peakLive, simulate, type PagedRun, type SideFrame } from '../../sim/paged';
import { Translated, tr } from '../../lib/i18n';

// "Paging the KV cache" (section 20.7). A small stream from src/sim/paged.ts: a pool of
// 256 token slots, blocks of 32 tokens, max_seq 128, 10 requests, seed 12. Every drawing is
// a frame of that simulation, and src/sim/paged.test.ts pins what the narration says about
// this run. Five beats, one narration clip each (content/ch20/narration/paged-blocks-N.txt).
// The two pool drawings are exported for the stepper in section 20.7.

export const SCENE_CFG = { pool: 256, block: 32, maxSeq: 128, requests: 10, seed: 12 };
export const SCENE_RUN = simulate(SCENE_CFG);
/** The first frame in which, on the paged side, a request finishes and a waiting one takes its blocks. */
export const REUSE_FRAME = firstReuse(SCENE_RUN);

const W = 720;
const H = 340;
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

/** Contiguous slots of max_seq tokens: the used part solid, the rest of the reservation washed. */
export function ContiguousPool({ run, side, x, y, w, h }: { run: PagedRun; side: SideFrame; x: number; y: number; w: number; h: number }) {
  const rowH = Math.min(22, h / Math.max(1, run.slots));
  const maxSeq = run.cfg.maxSeq;
  const leftover = run.cfg.pool - run.slots * maxSeq;
  return (
    <g data-side="contiguous" data-reserved={side.reserved} data-used={side.used} data-wasted={side.wasted} data-live={side.live.length}>
      {Array.from({ length: run.slots }, (_, s) => {
        const r = side.live.find((q) => q.slot === s);
        const yy = y + s * rowH;
        if (!r)
          return <rect key={s} x={x} y={yy} width={w} height={rowH - 3} fill="none" stroke="var(--rule)" strokeDasharray="3 3" />;
        return (
          <g key={s}>
            <rect x={x} y={yy} width={w} height={rowH - 3} fill="var(--global-wash)" stroke="var(--global)" strokeWidth="0.8" />
            <rect x={x} y={yy} width={(r.len / maxSeq) * w} height={rowH - 3} fill="var(--global)" />
            {rowH > 11 ? (
              <text x={x + w - 4} y={yy + rowH - 7} fontSize={Math.min(10.5, rowH - 5)} textAnchor="end"><Translated vi={<>r{r.id}: dùng {r.len} / {maxSeq}</>}>r{r.id}: {r.len} of {maxSeq} used</Translated></text>
            ) : null}
          </g>
        );
      })}
      {leftover > 0 ? (
        <text x={x} y={y + run.slots * rowH + 10} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>còn thừa {leftover} token: quá ít cho một slot nữa</>}>{leftover} tokens left over: too few for another slot</Translated></text>
      ) : null}
    </g>
  );
}

/** Paged blocks: each block shows its owner and how full it is; free blocks are dashed. */
export function PagedPool({ run, side, x, y, w, h, hot }: { run: PagedRun; side: SideFrame; x: number; y: number; w: number; h: number; hot?: number }) {
  const B = run.cfg.block;
  const nb = run.blocks;
  const cols = Math.min(16, nb);
  const rows = Math.ceil(nb / cols);
  const cw = w / cols;
  const ch = Math.min(28, h / rows);
  const owner = new Map<number, { id: number; fill: number }>();
  for (const r of side.live) r.blocks!.forEach((b, i) => owner.set(b, { id: r.id, fill: Math.min(B, r.len - i * B) / B }));
  return (
    <g data-side="paged" data-reserved={side.reserved} data-used={side.used} data-wasted={side.wasted} data-live={side.live.length} data-free={side.free.length}>
      {Array.from({ length: nb }, (_, b) => {
        const bx = x + (b % cols) * cw;
        const by = y + Math.floor(b / cols) * ch;
        const o = owner.get(b);
        if (!o)
          return <rect key={b} x={bx + 1} y={by + 1} width={cw - 2} height={ch - 2} fill="none" stroke="var(--rule)" strokeDasharray="3 3" />;
        const isHot = hot != null && o.id === hot;
        return (
          <g key={b} data-block={b} data-owner={o.id}>
            <rect x={bx + 1} y={by + 1} width={cw - 2} height={ch - 2} fill="var(--global-wash)" stroke={isHot ? 'var(--ink)' : 'var(--global)'} strokeWidth={isHot ? 2 : 0.8} />
            <rect x={bx + 1} y={by + 1} width={(cw - 2) * o.fill} height={ch - 2} fill="var(--global)" />
            {ch > 12 && cw > 18 ? (
              <text x={bx + cw / 2} y={by + ch / 2 + 3.5} fontSize={Math.min(10, ch - 6)} textAnchor="middle"
                style={{ fill: o.fill > 0.55 ? 'var(--page)' : 'var(--ink)' }}>r{o.id}</text>
            ) : null}
          </g>
        );
      })}
    </g>
  );
}

function Counters({ side, x, y, label }: { side: SideFrame; x: number; y: number; label: string }) {
  return (
    <text x={x} y={y} fontSize="10.5">
      <tspan fontWeight="600">{label}</tspan>
      <tspan>{tr(`  reserved ${side.reserved}, used ${side.used}, wasted ${side.wasted} · running ${side.live.length}, waiting ${side.waiting.length}, done ${side.done}`, `  giữ chỗ ${side.reserved}, dùng ${side.used}, lãng phí ${side.wasted} · đang chạy ${side.live.length}, đang chờ ${side.waiting.length}, xong ${side.done}`)}</tspan>
    </text>
  );
}

function Stage({ f, headline, show, hot, tables }: { f: number; headline: string; show: 'contiguous' | 'paged' | 'both'; hot?: number; tables?: boolean }) {
  const frame = SCENE_RUN.frames[Math.min(f, SCENE_RUN.frames.length - 1)];
  const dimC = show === 'paged' ? 0.3 : 1;
  const dimP = show === 'contiguous' ? 0.3 : 1;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Contiguous and paged KV-cache pools', 'Vùng KV cache liên tục và vùng KV cache phân trang')} style={{ width: '100%', maxWidth: W, display: 'block' }} data-frame={frame.step}>
      <text x={12} y={24} fontSize="14.5" fontWeight="600" data-headline="">{headline}</text>
      <text x={W - 12} y={24} fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>bước {frame.step}</>}>step {frame.step}</Translated></text>
      <g opacity={dimC}>
        <text x={12} y={52} fontSize="12" fontWeight="600"><Translated vi="liên tục: mỗi yêu cầu một slot max_seq">contiguous: a max_seq slot per request</Translated></text>
        <ContiguousPool run={SCENE_RUN} side={frame.contiguous} x={12} y={60} w={420} h={50} />
        <Counters side={frame.contiguous} x={12} y={124} label={tr('contiguous', 'liên tục')} />
      </g>
      <g opacity={dimP}>
        <text x={12} y={156} fontSize="12" fontWeight="600"><Translated vi={<>phân trang: các khối {SCENE_CFG.block} token lấy từ danh sách trống</>}>paged: blocks of {SCENE_CFG.block} tokens from a free list</Translated></text>
        <PagedPool run={SCENE_RUN} side={frame.paged} x={12} y={164} w={420} h={60} hot={hot} />
        <Counters side={frame.paged} x={12} y={242} label={tr('paged', 'phân trang')} />
      </g>
      {tables ? (
        <g data-tables="">
          <text x={460} y={156} fontSize="12" fontWeight="600"><Translated vi="bảng khối">block tables</Translated></text>
          {frame.paged.live.slice(0, 9).map((r, i) => (
            <text key={r.id} x={460} y={174 + i * 15} fontSize="11" fontWeight={r.id === hot ? 700 : 400}>
              r{r.id}: [{r.blocks!.join(', ')}]
            </text>
          ))}
        </g>
      ) : null}
      <text x={460} y={52} fontSize="12" fontWeight="600"><Translated vi="hàng đợi">queue</Translated></text>
      <text x={460} y={70} fontSize="11"><Translated vi={<>liên tục: {frame.contiguous.waiting.map((id) => 'r' + id).join(' ') || 'trống'}</>}>contiguous: {frame.contiguous.waiting.map((id) => 'r' + id).join(' ') || 'empty'}</Translated></text>
      <text x={460} y={86} fontSize="11"><Translated vi={<>phân trang: {frame.paged.waiting.map((id) => 'r' + id).join(' ') || 'trống'}</>}>paged: {frame.paged.waiting.map((id) => 'r' + id).join(' ') || 'empty'}</Translated></text>
      {frame.paged.events.length || frame.contiguous.events.length ? (
        <text x={12} y={300} fontSize="11.5" data-events="">
          <Translated vi={<>bước này, phân trang: {frame.paged.events.map((e) => e.replace(/ swapped back in$/, ' được đưa vào lại').replace(/ swapped out$/, ' bị đẩy ra').replace(/ admitted$/, ' được nhận').replace(/ finished$/, ' đã xong')).join(', ') || 'không có gì'}; liên tục: {frame.contiguous.events.map((e) => e.replace(/ admitted$/, ' được nhận').replace(/ finished$/, ' đã xong')).join(', ') || 'không có gì'}</>}>
            this step, paged: {frame.paged.events.join(', ') || 'nothing'}; contiguous: {frame.contiguous.events.join(', ') || 'nothing'}
          </Translated>
        </text>
      ) : null}
    </svg>
  );
}

/** The last beat: wasted token slots over the whole run, both sides, from every frame. */
function Summary({ p }: { p: number }) {
  const frames = SCENE_RUN.frames;
  const end = Math.max(2, Math.round(ramp(p, 0.05, 0.7) * (frames.length - 1)));
  const x0 = 60;
  const x1 = 690;
  const y0 = 226;
  const y1 = 56;
  const X = (t: number) => x0 + (t / (frames.length - 1)) * (x1 - x0);
  const Y = (v: number) => y0 - (v / SCENE_CFG.pool) * (y0 - y1);
  const line = (pick: (f: (typeof frames)[number]) => number) =>
    frames.slice(0, end + 1).map((f, t) => `${X(t).toFixed(1)},${Y(pick(f)).toFixed(1)}`).join(' ');
  const peak = peakLive(SCENE_RUN);
  const { contiguous: fc, paged: fp } = SCENE_RUN.finishedAt;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Wasted cache slots over time, contiguous and paged', 'Số slot cache lãng phí theo thời gian, liên tục và phân trang')} style={{ width: '100%', maxWidth: W, display: 'block' }}
      data-finished-contiguous={fc ?? ''} data-finished-paged={fp ?? ''} data-peak-contiguous={peak.contiguous} data-peak-paged={peak.paged}>
      <text x={12} y={24} fontSize="14.5" fontWeight="600" data-headline=""><Translated vi="Slot lãng phí trong suốt lượt chạy: giữ chỗ so với khối">Wasted slots over the run: reservations against blocks</Translated></text>
      <rect x={x0} y={y1} width={x1 - x0} height={y0 - y1} fill="var(--page)" stroke="var(--rule)" />
      {[0, 0.5, 1].map((k) => (
        <g key={k}>
          <line x1={x0} x2={x1} y1={Y(k * SCENE_CFG.pool)} y2={Y(k * SCENE_CFG.pool)} stroke="var(--rule-soft)" />
          <text x={x0 - 6} y={Y(k * SCENE_CFG.pool) + 4} fontSize="10" textAnchor="end">{k * SCENE_CFG.pool}</text>
        </g>
      ))}
      <text x={(x0 + x1) / 2} y={y0 + 18} fontSize="11" textAnchor="middle"><Translated vi="bước">step</Translated></text>
      <text x={x0 - 40} y={y1 - 10} fontSize="11"><Translated vi="slot lãng phí">wasted slots</Translated></text>
      <polyline points={line((f) => f.contiguous.wasted)} fill="none" stroke="var(--global)" strokeWidth="2.2" />
      <polyline points={line((f) => f.paged.wasted)} fill="none" stroke="var(--ink)" strokeWidth="2.2" strokeDasharray="5 3" />
      <line x1={x0} x2={x0 + 28} y1={y0 + 40} y2={y0 + 40} stroke="var(--global)" strokeWidth="2.2" />
      <text x={x0 + 36} y={y0 + 44} fontSize="11"><Translated vi="liên tục: giữ chỗ theo max_seq, nên phần lớn mỗi slot bị lãng phí">contiguous: reserved at max_seq, so most of each slot is wasted</Translated></text>
      <line x1={x0} x2={x0 + 28} y1={y0 + 58} y2={y0 + 58} stroke="var(--ink)" strokeWidth="2.2" strokeDasharray="5 3" />
      <text x={x0 + 36} y={y0 + 62} fontSize="11"><Translated vi={<>phân trang: không bao giờ quá {SCENE_CFG.block - 1} slot cho mỗi yêu cầu đang chạy</>}>paged: never more than {SCENE_CFG.block - 1} slots per running request</Translated></text>
      <text x={12} y={318} fontSize="11.5">
        <Translated vi={<>cả {SCENE_CFG.requests} yêu cầu đã xong: liên tục sau {fc} bước, phân trang sau {fp}; tối đa {peak.contiguous} yêu cầu chạy cùng lúc so với {peak.paged}</>}>
          all {SCENE_CFG.requests} requests done: contiguous after {fc} steps, paged after {fp}; at most {peak.contiguous} running at once against {peak.paged}
        </Translated>
      </text>
    </svg>
  );
}

const reuse = REUSE_FRAME;
/** The request to outline in a frame: the first live one whose blocks are scattered, else the first. */
const hotIn = (f: number) => {
  const live = SCENE_RUN.frames[f].paged.live;
  return (live.find((q) => isScattered(q.blocks!)) ?? live[0])?.id;
};
const GROW_END = Math.min(SCENE_RUN.frames.length - 1, reuse + 26);
const beats: Beat[] = [
  {
    clip: 'paged-blocks-1',
    get label() {
      return tr('Contiguous', 'Liên tục');
    },
    draw: (p) => <Stage f={Math.round(ramp(p, 0.05, 0.9) * (reuse + 2))} headline={tr('Contiguous: each request reserves max_seq on arrival', 'Liên tục: mỗi yêu cầu giữ chỗ max_seq ngay khi đến')} show="contiguous" />, // i18n-ignore: show is a mode value
  },
  {
    clip: 'paged-blocks-2',
    get label() {
      return tr('Blocks', 'Khối');
    },
    draw: (p) => {
      const f = Math.max(1, Math.round(ramp(p, 0.1, 0.8) * 3));
      return <Stage f={f} headline={tr('Paged: fixed-size blocks and a block table per request', 'Phân trang: các khối kích thước cố định và một bảng khối cho mỗi yêu cầu')} show="paged" hot={hotIn(f)} tables />; // i18n-ignore: show is a mode value
    },
  },
  {
    clip: 'paged-blocks-3',
    get label() {
      return tr('Finish and reuse', 'Xong và tái sử dụng');
    },
    draw: (p) => <Stage f={p < 0.35 ? reuse - 1 : reuse} headline={p < 0.35 ? tr('A request is about to finish…', 'Một yêu cầu sắp xong…') : tr('…its block returns to the free list and a waiting request starts', '…khối của nó trở về danh sách trống và một yêu cầu đang chờ bắt đầu')} show="both" tables />, // i18n-ignore: show is a mode value
  },
  {
    clip: 'paged-blocks-4',
    get label() {
      return tr('Growing', 'Lớn dần');
    },
    draw: (p) => {
      const f = reuse + 1 + Math.round(ramp(p, 0.05, 0.95) * (GROW_END - reuse - 1));
      return <Stage f={f} headline={tr('New blocks come from the free list; the tables keep the order', 'Khối mới lấy từ danh sách trống; các bảng giữ thứ tự')} show="paged" hot={hotIn(f)} tables />; // i18n-ignore: show is a mode value
    },
  },
  { clip: 'paged-blocks-5', get label() { return tr('Side by side', 'Đặt cạnh nhau'); }, draw: (p) => <Summary p={p} /> },
];

export function PagedBlocksScene() {
  return <Scene chapter="ch20" title={tr('Paging the KV cache', 'Phân trang KV cache')} beats={beats} />;
}
