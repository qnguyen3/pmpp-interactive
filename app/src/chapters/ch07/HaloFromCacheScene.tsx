import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { blockHistogram } from '../../sim/convcached';
import {
  SCENE_A,
  SCENE_B,
  SCENE_C,
  SCENE_ONE_L2,
  SCENE_SPLIT_L2,
  eventsOf,
  imageDims,
  lineBytes,
  sceneFacts,
  simulate,
  type CacheConfig,
  type CacheEvent,
  type CacheRun,
} from '../../sim/halocache';

// "Where the halo comes from" (section 7.6). Every frame is drawn from our toy cache model
// (sim/halocache.ts: the two scene runs) and the tap classifier (sim/convcached.ts); the
// numbers the narration says are pinned by halocache.test.ts. Five beats, one clip each
// (content/ch07/narration/halo-from-cache-N.txt).

const F = sceneFacts();
const CFG = SCENE_ONE_L2;
const T = CFG.tile;
const R = CFG.r;
const { width: W, height: H } = imageDims(CFG);
const LINES_PER_ROW = W / CFG.lineFloats;
const B_TAPS = blockHistogram({ tile: T, r: R, width: W, height: H, bx: 1, by: 0 });

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

// Cache contents after k accesses, per run, remembered (each beat asks for the same few dozen moments).
const memo = new Map<string, number[][]>();
function contentsAt(cfg: CacheConfig, k: number): number[][] {
  const key = cfg.slices + ':' + k;
  let c = memo.get(key);
  if (!c) {
    c = simulate(cfg, { stopAfter: k }).contents;
    memo.set(key, c);
  }
  return c;
}

/** Event index reached at progress p through events [s, e), holding the end for the last 15%. */
const at = (p: number, s: number, e: number) => s + Math.round(clamp(p / 0.85) * (e - s));
const range = (run: CacheRun, block: number) => {
  const ix = eventsOf(run, block);
  return [ix[0], ix[ix.length - 1] + 1] as const;
};
const tally = (evs: CacheEvent[], keep?: (e: CacheEvent) => boolean) => {
  const xs = keep ? evs.filter(keep) : evs;
  return { hits: xs.filter((e) => e.hit).length, of: xs.length };
};

interface MapProps {
  x: number;
  y: number;
  cell: number;
  /** Lines resident in this cache now. */
  inL2: Set<number>;
  /** Accesses to show as just made: line -> hit or miss (the latest wins). */
  marks: Map<number, boolean>;
  /** Tiles to outline, with a label. */
  tiles?: { bx: number; by: number; label: string; tone: 'shared' | 'ink' }[];
  title?: string;
  /** Lines to underline (e.g. A's lines in B's halo). */
  ring?: Set<number>;
}

function LineMap({ x, y, cell, inL2, marks, tiles = [], title, ring }: MapProps) {
  const lw = CFG.lineFloats * cell;
  const rows: ReactNode[] = [];
  for (let row = 0; row < H; row++)
    for (let k = 0; k < LINES_PER_ROW; k++) {
      const line = row * LINES_PER_ROW + k;
      const mark = marks.get(line);
      const fill = mark === true ? 'var(--ok-wash)' : mark === false ? 'var(--fail-wash)' : inL2.has(line) ? 'var(--global-wash)' : 'var(--cell)';
      const stroke = mark === true ? 'var(--ok)' : mark === false ? 'var(--fail)' : inL2.has(line) ? 'var(--global)' : 'var(--rule)';
      rows.push(
        <rect key={line} x={x + k * lw + 0.5} y={y + row * cell + 0.5} width={lw - 1} height={cell - 1}
          fill={fill} stroke={stroke} strokeWidth={mark === undefined ? 0.6 : 1.4}
          data-line={line} data-state={mark === true ? 'hit' : mark === false ? 'miss' : inL2.has(line) ? 'l2' : 'none'} />,
      );
      if (ring?.has(line))
        rows.push(<line key={'u' + line} x1={x + k * lw + 1} x2={x + (k + 1) * lw - 1} y1={y + (row + 1) * cell - 0.5} y2={y + (row + 1) * cell - 0.5}
          stroke="var(--ink)" strokeWidth="1.2" strokeDasharray="2 1.5" />);
    }
  return (
    <g>
      {title ? <text x={x} y={y - 8} fontSize="11.5" fontWeight="600">{title}</text> : null}
      {rows}
      {/* tile grid */}
      {Array.from({ length: W / T + 1 }, (_, i) => (
        <line key={'v' + i} x1={x + i * T * cell} x2={x + i * T * cell} y1={y} y2={y + H * cell} stroke="var(--ink-3)" strokeWidth="0.8" />
      ))}
      {Array.from({ length: H / T + 1 }, (_, i) => (
        <line key={'h' + i} x1={x} x2={x + W * cell} y1={y + i * T * cell} y2={y + i * T * cell} stroke="var(--ink-3)" strokeWidth="0.8" />
      ))}
      {tiles.map((t) => (
        <g key={t.label}>
          <rect x={x + t.bx * T * cell - 1} y={y + t.by * T * cell - 1} width={T * cell + 2} height={T * cell + 2} fill="none"
            stroke={t.tone === 'shared' ? 'var(--shared)' : 'var(--ink)'} strokeWidth="2.4" />
          <rect x={x + t.bx * T * cell + (T * cell) / 2 - 8} y={y + t.by * T * cell + (T * cell) / 2 - 8} width={16} height={16} rx={2}
            fill="var(--page)" opacity={0.9} />
          <text x={x + t.bx * T * cell + (T * cell) / 2} y={y + t.by * T * cell + (T * cell) / 2 + 5} fontSize="14" fontWeight="700"
            textAnchor="middle" style={{ fill: t.tone === 'shared' ? 'var(--shared)' : 'var(--ink)' }}>{t.label}</text>
        </g>
      ))}
    </g>
  );
}

function Counter({ y, name, label, hits, of, word = tr('found in L2', 'có trong L2') }: { y: number; name: string; label: string; hits: number; of: number; word?: string }) {
  return (
    <g data-counter={name} data-hits={hits} data-of={of}>
      <text x={432} y={y} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>{label}</text>
      <text x={432} y={y + 16} fontSize="13" fontWeight="650">{hits} {tr('of', 'trên')} {of} {word}</text>
    </g>
  );
}

function Legend({ y }: { y: number }) {
  const items: [string, string, string][] = [
    ['var(--global-wash)', 'var(--global)', tr('line in L2', 'dòng có trong L2')],
    ['var(--fail-wash)', 'var(--fail)', tr('just read: miss, from DRAM', 'vừa đọc: trượt, lấy từ DRAM')],
    ['var(--ok-wash)', 'var(--ok)', tr('just read: hit in L2', 'vừa đọc: trúng ở L2')],
  ];
  return (
    <g>
      {items.map(([fill, stroke, label], i) => (
        <g key={label}>
          <rect x={432} y={y + i * 17 - 9} width={22} height={9} fill={fill} stroke={stroke} />
          <text x={460} y={y + i * 17} fontSize="11">{label}</text>
        </g>
      ))}
      <text x={432} y={y + 3 * 17} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>một thanh = một dòng cache = một hàng của ô gồm {CFG.lineFloats} số float</>}>one bar = one cache line = one tile row of {CFG.lineFloats} floats</Translated></text>
    </g>
  );
}

function Frame({ beat, headline, children, note }: { beat: number; headline: string; children: ReactNode; note?: string }) {
  return (
    <svg viewBox="0 0 720 330" role="img" aria-label={tr('Our toy cache model: which halo reads hit in L2', 'Mô hình cache đồ chơi của chúng tôi: những lần đọc vùng biên (halo) nào trúng ở L2')} data-beat={beat}
      style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      <text x={16} y={20} fontSize="13" fontWeight="650" data-headline>{headline}</text>
      {children}
      {note ? <text x={16} y={318} fontSize="11" style={{ fill: 'var(--ink-2)' }} data-note>{note}</text> : null}
    </svg>
  );
}

const MAP = { x: 16, y: 48, cell: 6 };
const NOTE = () => tr(`Our toy model, not a measurement: a ${CFG.capacity}-line LRU cache, one block at a time, image ${W} × ${H} floats, radius ${R}.`, `Mô hình đồ chơi của chúng tôi, không phải số đo: một cache LRU ${CFG.capacity} dòng, mỗi lần một khối, ảnh ${W} × ${H} số float, bán kính ${R}.`);

function marksOf(run: CacheRun, from: number, to: number, keep?: (e: CacheEvent) => boolean) {
  const m = new Map<number, boolean>();
  for (let i = from; i < to; i++) {
    const e = run.events[i];
    if (!keep || keep(e)) m.set(e.line, e.hit);
  }
  return m;
}

const aSet = new Set(F.aLines);
const aBottom = new Set(F.aBottom);

// ---------------------------------------------------------------- beats

function beatA(p: number) {
  const run = F.one;
  const [s, e] = range(run, SCENE_A);
  const k = at(p, s, e);
  const done = run.events.slice(s, k);
  const loads = tally(done.filter((x) => x.phase === 'load'));
  const halo = tally(done.filter((x) => x.phase === 'halo'));
  return (
    <Frame beat={1} headline={k - s <= 8 ? tr('Block A loads its own tile: every line comes from DRAM, through L2', 'Khối A nạp ô của chính nó: mọi dòng đều đến từ DRAM, qua L2') : tr('Then A reads its halo from global memory: those lines land in L2 too', 'Rồi A đọc vùng biên của nó từ bộ nhớ toàn cục: các dòng đó cũng nằm lại trong L2')} note={NOTE()}>
      <LineMap {...MAP} inL2={new Set(contentsAt(CFG, k)[0])} marks={marksOf(run, s, k)} tiles={[{ bx: 0, by: 0, label: 'A', tone: 'shared' }]} title={tr('the image, one bar per cache line', 'ảnh, mỗi dòng cache một thanh')} />
      <Counter y={66} name="a-load" /* i18n-ignore: data key */ label={tr("A's own lines (into shared memory)", 'các dòng của chính A (vào bộ nhớ chia sẻ)')} hits={loads.hits} of={loads.of} />
      <Counter y={108} name="a-halo" /* i18n-ignore: data key */ label={tr("A's halo lines (read from global)", 'các dòng vùng biên của A (đọc từ bộ nhớ toàn cục)')} hits={halo.hits} of={halo.of} />
      <Legend y={170} />
    </Frame>
  );
}

function beatB(p: number) {
  const run = F.one;
  const [s, e] = range(run, SCENE_B);
  const k = at(p, s, e);
  const done = run.events.slice(s, k);
  const own = tally(done.filter((x) => x.phase === 'load'));
  const left = tally(done.filter((x) => x.phase === 'halo' && aSet.has(x.line)));
  const other = tally(done.filter((x) => x.phase === 'halo' && !aSet.has(x.line)));
  return (
    <Frame beat={2} headline={left.of === F.bLeft ? tr(`B's left halo is A's interior: ${left.hits} of ${left.of} lines found in L2`, `Vùng biên trái của B là phần trong của A: ${left.hits} trên ${left.of} dòng có trong L2`) : tr('Block B, next door, loads its own tile', 'Khối B, ngay bên cạnh, nạp ô của chính nó')} note={NOTE()}>
      <LineMap {...MAP} inL2={new Set(contentsAt(CFG, k)[0])} marks={marksOf(run, s, k)} ring={aSet}
        tiles={[{ bx: 0, by: 0, label: 'A', tone: 'ink' }, { bx: 1, by: 0, label: 'B', tone: 'shared' }]} title={tr('the image, one bar per cache line', 'ảnh, mỗi dòng cache một thanh')} />
      <Counter y={66} name="b-load" /* i18n-ignore: data key */ label={tr("B's own lines", 'các dòng của chính B')} hits={own.hits} of={own.of} />
      <Counter y={108} name="b-left" /* i18n-ignore: data key */ label={tr("B's halo reads of A's lines (dashed)", 'các lần B đọc vùng biên vào dòng của A (nét đứt)')} hits={left.hits} of={left.of} />
      <Counter y={150} name="b-other" /* i18n-ignore: data key */ label={tr("B's other halo lines", 'các dòng vùng biên khác của B')} hits={other.hits} of={other.of} />
      <text x={432} y={200} fontSize="11" style={{ fill: 'var(--ink-2)' }} data-taps={`${B_TAPS.shared},${B_TAPS.global},${B_TAPS.ghost}`}>
        <tspan x={432}><Translated vi={<>{B_TAPS.threads} luồng của B thực hiện {B_TAPS.taps.toLocaleString('en-US')} tap:</>}>B's {B_TAPS.threads} threads make {B_TAPS.taps.toLocaleString('en-US')} taps:</Translated></tspan>
        <tspan x={432} dy={14}><Translated vi={<>{B_TAPS.shared.toLocaleString('en-US')} chia sẻ, {B_TAPS.global} toàn cục, {B_TAPS.ghost} ô ma</>}>{B_TAPS.shared.toLocaleString('en-US')} shared, {B_TAPS.global} global, {B_TAPS.ghost} ghost</Translated></tspan>
      </text>
      <Legend y={250} />
    </Frame>
  );
}

function beatC(p: number) {
  const run = F.one;
  const [, eB] = range(run, SCENE_B);
  const [sC, eC] = range(run, SCENE_C);
  const between = p < 0.35;
  const k = between ? eB + Math.round(ramp(p, 0, 0.3) * (sC - eB)) : at(ramp(p, 0.35, 1), sC, eC);
  const cur = run.events[Math.max(0, k - 1)];
  const done = between ? [] : run.events.slice(sC, k);
  const onA = tally(done.filter((x) => x.phase === 'halo' && aBottom.has(x.line)));
  const halo = tally(done.filter((x) => x.phase === 'halo'));
  const runningBlock = between ? run.blocks[cur.block] : run.blocks[SCENE_C];
  return (
    <Frame beat={3}
      headline={between ? tr(`Blocks ${SCENE_B + 1} to ${SCENE_C - 1} run; the ${CFG.capacity}-line cache turns over`, `Các khối ${SCENE_B + 1} đến ${SCENE_C - 1} chạy; cache ${CFG.capacity} dòng được thay mới hoàn toàn`)
        : onA.of === F.cOnABottom ? tr(`C reads A's bottom rows as halo: ${onA.hits} of ${onA.of} lines still in L2`, `C đọc các hàng dưới cùng của A làm vùng biên: ${onA.hits} trên ${onA.of} dòng vẫn còn trong L2`) : tr('C, the first block of the next row', 'C, khối đầu tiên của hàng kế tiếp')}
      note={NOTE()}>
      <LineMap {...MAP} inL2={new Set(contentsAt(CFG, k)[0])} marks={between ? new Map() : marksOf(run, sC, k)} ring={aBottom}
        tiles={[{ bx: 0, by: 0, label: 'A', tone: 'ink' },
          ...(between ? [{ bx: runningBlock.bx, by: runningBlock.by, label: '·', tone: 'shared' as const }] : [{ bx: 0, by: 1, label: 'C', tone: 'shared' as const }])]}
        title={tr('the image, one bar per cache line', 'ảnh, mỗi dòng cache một thanh')} />
      <g data-counter="since-b" data-hits={between ? cur.block - SCENE_B : SCENE_C - SCENE_B - 1} data-of={F.evictedBeforeC}>
        <text x={432} y={66} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="số khối đã chạy kể từ B">blocks run since B</Translated></text>
        <text x={432} y={82} fontSize="13" fontWeight="650">{between ? Math.max(0, cur.block - SCENE_B) : SCENE_C - SCENE_B - 1}</text>
        <text x={432} y={104} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="số dòng bị loại khỏi cache giữa B và C">lines evicted between B and C</Translated></text>
        <text x={432} y={120} fontSize="13" fontWeight="650">{F.evictedBeforeC}</text>
      </g>
      <Counter y={150} name="c-on-a" /* i18n-ignore: data key */ label={tr("C's reads of A's bottom rows (dashed)", 'các lần C đọc các hàng dưới cùng của A (nét đứt)')} hits={onA.hits} of={onA.of} />
      <Counter y={192} name="c-halo" /* i18n-ignore: data key */ label={tr("all of C's halo lines", 'tất cả các dòng vùng biên của C')} hits={halo.hits} of={halo.of} />
      <Legend y={250} />
    </Frame>
  );
}

function beatSplit(p: number) {
  const run = F.split;
  const [sA] = range(run, SCENE_A);
  const [, eB] = range(run, SCENE_B);
  const k = at(p, sA, eB);
  const done = run.events.slice(sA, k);
  const left = tally(done.filter((x) => x.block === SCENE_B && x.phase === 'halo' && aSet.has(x.line)));
  const c = contentsAt(SCENE_SPLIT_L2, k);
  const cell = 4.5;
  const die = (s: number, x: number) => {
    const tiles = [
      ...(F.sliceA === s ? [{ bx: 0, by: 0, label: 'A', tone: 'shared' as const }] : []),
      ...(F.sliceB === s ? [{ bx: 1, by: 0, label: 'B', tone: 'shared' as const }] : []),
    ];
    return (
      <LineMap x={x} y={64} cell={cell} inL2={new Set(c[s])} marks={marksOf(run, sA, k, (e) => e.slice === s)} tiles={tiles}
        ring={s === F.sliceB ? aSet : undefined} title={tr(`L2 of die ${s} (${SCENE_SPLIT_L2.capacity / SCENE_SPLIT_L2.slices} lines)`, `L2 của die ${s} (${SCENE_SPLIT_L2.capacity / SCENE_SPLIT_L2.slices} dòng)`)} />
    );
  };
  return (
    <Frame beat={4} headline={left.of === F.splitBLeft ? tr(`Split L2: B's reads of A's lines, ${left.hits} of ${left.of} in its die's L2`, `L2 tách đôi: các lần B đọc dòng của A, ${left.hits} trên ${left.of} có trong L2 của die của nó`) : tr(`Split L2: A runs on die ${F.sliceA}, B on die ${F.sliceB}`, `L2 tách đôi: A chạy trên die ${F.sliceA}, B trên die ${F.sliceB}`)}
      note={tr(`Our toy model: two dies, each with half of the ${SCENE_SPLIT_L2.capacity} lines; blocks dealt to dies at random (seed ${SCENE_SPLIT_L2.seed}).`, `Mô hình đồ chơi của chúng tôi: hai die, mỗi die giữ một nửa trong ${SCENE_SPLIT_L2.capacity} dòng; các khối được chia ngẫu nhiên cho các die (seed ${SCENE_SPLIT_L2.seed}).`)}>
      <g data-dies={`${F.sliceA},${F.sliceB}`}>
        {die(0, 16)}
        {die(1, 368)}
      </g>
      <Counter y={252} name="split-b-left" /* i18n-ignore: data key */ label={tr("B's halo reads of A's lines (dashed)", 'các lần B đọc vùng biên vào dòng của A (nét đứt)')} hits={left.hits} of={left.of} word={tr("found in B's L2", 'có trong L2 của B')} />
      <text x={16} y={258} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>
        <tspan x={16}><Translated vi={<>A đã nạp các dòng của nó vào L2 của die {F.sliceA}.</>}>A loaded its lines into die {F.sliceA}'s L2.</Translated></tspan>
        <tspan x={16} dy={15}><Translated vi={<>B chạy trên die {F.sliceB} và chỉ tìm ở đó.</>}>B runs on die {F.sliceB} and looks only there.</Translated></tspan>
      </text>
    </Frame>
  );
}

function beatBars(p: number) {
  const b = F.bytes;
  const bars: [string, string, number][] = [
    ['ideal', tr('all hits: each line from DRAM once (the ideal)', 'trúng hết: mỗi dòng lấy từ DRAM một lần (lý tưởng)'), b.ideal],
    ['one', tr('the model, one shared L2 (part 3)', 'mô hình, một L2 dùng chung (phần 3)'), b.one],
    ['split', tr('the model, L2 split per die (part 4)', 'mô hình, L2 tách theo từng die (phần 4)'), b.split],
    ['none', tr('all misses: no reuse between blocks', 'trượt hết: không tái sử dụng giữa các khối'), b.noReuse],
  ];
  const max = Math.max(...bars.map((x) => x[2]));
  const g = ramp(p, 0.05, 0.5);
  return (
    <Frame beat={5} headline={tr('DRAM bytes per output: which bar does your GPU give you?', 'Số byte DRAM mỗi đầu ra: GPU của bạn cho bạn thanh nào?')} note={tr(`Bytes per output: cache lines fetched × ${lineBytes(CFG.lineFloats)} bytes, plus one 4-byte store, over the outputs. Our toy model.`, `Số byte mỗi đầu ra: số dòng cache đã nạp × ${lineBytes(CFG.lineFloats)} byte, cộng một lần ghi 4 byte, chia cho số đầu ra. Mô hình đồ chơi của chúng tôi.`)}>
      {bars.map(([id, label, v], i) => (
        <g key={id} data-bar={id} data-bytes={v}>
          <text x={16} y={64 + i * 52} fontSize="12">{label}</text>
          <rect x={16} y={72 + i * 52} width={Math.max(1, (520 * v * g) / max)} height={18} fill="var(--global)" />
          <text x={16 + (520 * v * g) / max + 8} y={86 + i * 52} fontSize="12.5" fontWeight="650">{g > 0.98 ? `${v.toFixed(v % 1 ? 1 : 0)} B` : ''}</text>
        </g>
      ))}
      <text x={16} y={290} fontSize="12" style={{ fill: 'var(--ink-2)' }} opacity={ramp(p, 0.55, 0.75)}>
        <Translated vi={<>Sách: một &ldquo;xác suất đáng kể&rdquo; là trúng. Tỷ lệ trúng L2 của một profiler, hoặc số đo thời gian của lab 7.2, sẽ cho bạn biết là thanh nào.</>}>
          The book: a "significant probability" of hits. A profiler's L2 hit rate, or lab 7.2's timings, tell you which.
        </Translated>
      </text>
    </Frame>
  );
}

const beats: Beat[] = [
  { clip: 'halo-from-cache-1', get label() { return tr('Block A', 'Khối A'); }, draw: beatA },
  { clip: 'halo-from-cache-2', get label() { return tr('Next door', 'Ngay bên cạnh'); }, draw: beatB },
  { clip: 'halo-from-cache-3', get label() { return tr('A row later', 'Một hàng sau'); }, draw: beatC },
  { clip: 'halo-from-cache-4', get label() { return tr('Split L2', 'L2 tách đôi'); }, draw: beatSplit },
  { clip: 'halo-from-cache-5', get label() { return tr('Which bar', 'Thanh nào'); }, draw: beatBars },
];

export function HaloFromCacheScene() {
  return <Scene chapter="ch07" title={tr('Where the halo comes from', 'Vùng biên (halo) đến từ đâu')} beats={beats} />;
}
