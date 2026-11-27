import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { PRESETS, bookTiled, exactTiled, tag, type Item, type TileIteration } from '../../sim/mergetile';

// "Loaded twice, used once": the book's tiled merge (Figs. 13.10 to 13.13) on one block
// with a tile of 8 and two threads, the worst case that forces a full tile of each input,
// and the exact-load design of Exercise 3 that the libraries ship. Every number drawn or
// said comes from src/sim/mergetile.ts. Five beats, one narration clip each
// (content/ch13/narration/tile-waste-N.txt).

const UNEVEN = PRESETS.uneven;
const WORST = PRESETS.allFromA;
const RUN = bookTiled(UNEVEN.A, UNEVEN.B, 1, UNEVEN.tile, 'corrected', UNEVEN.threads);
const IT = RUN.blocks[0].iterations;
const RUN_W = bookTiled(WORST.A, WORST.B, 1, WORST.tile, 'corrected', WORST.threads);
const IT_W = RUN_W.blocks[0].iterations;
const EXACT = exactTiled(UNEVEN.A, UNEVEN.B, UNEVEN.tile);
const TILE = UNEVEN.tile;

const W = 720;
const GW = 19; // a global-memory cell
const SW = 30; // a shared-memory cell
const AX = 44;
const GY = 46; // global rows
const SY = 128; // shared rows
const CY = 212; // C row
const ASX = 104;
const BSX = 404;

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

type GState = '' | 'load' | 'again' | 'done';
type SState = '' | 'loaded' | 'used' | 'unused';

interface Frame {
  beat: number;
  A: readonly number[];
  B: readonly number[];
  /** State of each global element. */
  gA: GState[];
  gB: GState[];
  /** Shared tiles: values and states. In the exact view one tile holds A's part, then B's. */
  sA: { v: number; s: SState }[];
  sB: { v: number; s: SState }[];
  exact?: boolean;
  /** Outputs written so far. */
  C: (Item | null)[];
  arrows?: number;
  threads?: string;
  counter?: string;
  headline: string;
  data: Record<string, string | number>;
}

const bx = (A: readonly number[]) => AX + A.length * GW + 40;

/**
 * Global states for iteration n of the book's scheme: consumed by earlier iterations,
 * loaded now for the first time, or loaded now although the previous iteration had
 * already loaded it (and not used it).
 */
function globalStates(its: TileIteration[], n: number, len: number, which: 'A' | 'B'): GState[] {
  const it = its[n];
  const start = which === 'A' ? it.aStart : it.bStart;
  const loaded = which === 'A' ? it.aLoaded : it.bLoaded;
  const prev = n > 0 ? its[n - 1] : null;
  const prevEnd = prev ? (which === 'A' ? prev.aStart + prev.aLoaded : prev.bStart + prev.bLoaded) : 0;
  return Array.from({ length: len }, (_, g) => {
    if (g < start) return 'done';
    if (g < start + loaded) return g < prevEnd ? 'again' : 'load';
    return '';
  });
}

function tileStates(keys: readonly number[], it: TileIteration, which: 'A' | 'B', used: boolean): { v: number; s: SState }[] {
  const start = which === 'A' ? it.aStart : it.bStart;
  const loaded = which === 'A' ? it.aLoaded : it.bLoaded;
  const u = which === 'A' ? it.aUsed : it.bUsed;
  return Array.from({ length: loaded }, (_, s) => ({ v: keys[start + s], s: !used ? 'loaded' : s < u ? 'used' : 'unused' }));
}

/** Totals of the book scheme after iterations 0..n (loaded counts whole iterations; used only once merged). */
function totals(its: TileIteration[], n: number, merged: boolean) {
  const loaded = its.slice(0, n + 1).reduce((s, it) => s + it.aLoaded + it.bLoaded, 0);
  const used = its.slice(0, merged ? n + 1 : n).reduce((s, it) => s + it.cOut, 0);
  return { loaded, used };
}

function outputs(run: typeof RUN, upto: number): (Item | null)[] {
  return run.output.map((x, k) => (k < upto ? x : null));
}

const split = (it: TileIteration) =>
  it.threads.map((t, q) => tr(`thread ${q}: ${t.a} of A, ${t.b} of B`, `luồng ${q}: ${t.a} của A, ${t.b} của B`)).join(';  ');

/** One iteration of the book's scheme, as a frame: loads first, then (if `merged`) what the merge used. */
function bookFrame(beat: number, run: typeof RUN, A: readonly number[], B: readonly number[], n: number, merged: boolean, cProgress: number, headline: string): Frame {
  const its = run.blocks[0].iterations;
  const it = its[n];
  const t = totals(its, n, merged);
  const cUpto = it.cStart + Math.round(it.cOut * (merged ? cProgress : 0));
  const gA = globalStates(its, n, A.length, 'A');
  const gB = globalStates(its, n, B.length, 'B');
  const again = [...gA, ...gB].filter((s) => s === 'again').length;
  return {
    beat,
    A,
    B,
    gA,
    gB,
    sA: tileStates(A, it, 'A', merged),
    sB: tileStates(B, it, 'B', merged),
    C: outputs(run, cUpto),
    arrows: merged ? 0.25 : 1,
    threads: merged ? split(it) : undefined,
    counter: tr(`loaded so far: ${t.loaded}   used so far: ${t.used}`, `đã nạp: ${t.loaded}   đã dùng: ${t.used}`),
    headline,
    data: {
      scheme: 'book',
      it: n,
      phase: merged ? 'use' : 'load',
      aloaded: it.aLoaded,
      bloaded: it.bLoaded,
      aused: merged ? it.aUsed : '',
      bused: merged ? it.bUsed : '',
      again,
      loaded: t.loaded,
      used: t.used,
    },
  };
}

function Cells({ x, y, w, h, vals, states, name, kind }: {
  x: number; y: number; w: number; h: number; vals: readonly number[]; states: string[]; name: string; kind: 'g' | 's';
}) {
  return (
    <g>
      {vals.map((v, p) => {
        const s = states[p] ?? '';
        const fill = s === 'load' || s === 'again' || s === 'loaded' ? 'var(--shared-wash)' : s === 'used' ? 'var(--ok-wash)' : s === 'done' ? 'var(--global-wash)' : 'var(--cell)';
        const stroke = s === 'again' || s === 'unused' ? 'var(--fail)' : s === 'used' ? 'var(--ok)' : s === 'load' || s === 'loaded' ? 'var(--shared)' : 'var(--rule)';
        return (
          <g key={p} data-cell={`${name}${p}`} data-state={s}>
            <rect x={x + p * w} y={y} width={w - 2} height={h} rx="2" fill={fill} stroke={stroke}
              strokeWidth={s && s !== 'done' ? 1.6 : 1} strokeDasharray={s === 'again' || s === 'unused' ? '3 2' : undefined} />
            <text x={x + p * w + (w - 2) / 2} y={y + h / 2 + 4} fontSize={kind === 'g' ? 9.5 : 12} textAnchor="middle"
              style={{ fill: s === 'done' ? 'var(--ink-3)' : 'var(--ink)' }}>{v}</text>
          </g>
        );
      })}
    </g>
  );
}

function Stage(f: Frame) {
  const BX = bx(f.A);
  const sAx = ASX;
  const sBx = f.exact ? ASX + f.sA.length * SW : BSX;
  const cw = Math.min(GW, (W - 80) / Math.max(1, f.C.length));
  const cx0 = (W - f.C.length * cw) / 2;
  const loadedA = f.gA.map((s, g) => (s === 'load' || s === 'again' ? g : -1)).filter((g) => g >= 0);
  const loadedB = f.gB.map((s, g) => (s === 'load' || s === 'again' ? g : -1)).filter((g) => g >= 0);
  const mid = (xs: number[], x0: number) => (xs.length ? x0 + ((xs[0] + xs[xs.length - 1] + 1) / 2) * GW : null);
  const mA = mid(loadedA, AX);
  const mB = mid(loadedB, BX);
  return (
    <svg viewBox={`0 0 ${W} 330`} role="img" aria-label={tr('A tiled merge: what one block loads and what it uses', 'Một phép trộn phân ô: một khối nạp gì và dùng gì')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <g data-beat={f.beat} {...Object.fromEntries(Object.entries(f.data).map(([k, v]) => [`data-${k}` /* i18n-ignore: data attribute name */, String(v)]))}>
        <text x={8} y={GY - 16} fontSize="11" style={{ fill: 'var(--global)' }} fontWeight="600"><Translated vi="bộ nhớ toàn cục (global memory)">global memory</Translated></text>
        <text x={AX - 8} y={GY + 15} fontSize="13" fontWeight="650" textAnchor="end" style={{ fill: 'var(--global)' }}>A</text>
        <Cells x={AX} y={GY} w={GW} h={22} vals={f.A} states={f.gA} name="A" kind="g" />
        <text x={BX - 8} y={GY + 15} fontSize="13" fontWeight="650" textAnchor="end" style={{ fill: 'var(--global)' }}>B</text>
        <Cells x={BX} y={GY} w={GW} h={22} vals={f.B} states={f.gB} name="B" kind="g" />

        {f.arrows && mA != null ? (
          <line x1={mA} y1={GY + 26} x2={sAx + (f.sA.length * SW) / 2} y2={SY - 6} stroke="var(--shared)" strokeWidth="1.6" opacity={f.arrows} />
        ) : null}
        {f.arrows && mB != null ? (
          <line x1={mB} y1={GY + 26} x2={sBx + (f.sB.length * SW) / 2} y2={SY - 6} stroke="var(--shared)" strokeWidth="1.6" opacity={f.arrows} />
        ) : null}

        <text x={8} y={SY - 12} fontSize="11" style={{ fill: 'var(--shared)' }} fontWeight="600"><Translated vi="bộ nhớ chia sẻ (shared memory)">shared memory</Translated></text>
        <text x={sAx - 8} y={SY + 20} fontSize="12.5" fontWeight="650" textAnchor="end" style={{ fill: 'var(--shared)' }}>{f.exact ? tr('tile', 'ô') : 'A_S'}</text>
        {/* empty slots of the tiles, so a partly loaded tile shows */}
        {Array.from({ length: f.exact ? TILE : TILE }, (_, s) => (
          <rect key={'ea' + s} x={sAx + s * SW} y={SY} width={SW - 2} height={28} rx="2" fill="none" stroke="var(--rule)" strokeDasharray="2 3" opacity={f.exact ? 1 : 1} />
        ))}
        <Cells x={sAx} y={SY} w={SW} h={28} vals={f.sA.map((c) => c.v)} states={f.sA.map((c) => c.s)} name="AS" kind="s" />
        {!f.exact ? (
          <>
            <text x={BSX - 8} y={SY + 20} fontSize="12.5" fontWeight="650" textAnchor="end" style={{ fill: 'var(--shared)' }}>B_S</text>
            {Array.from({ length: TILE }, (_, s) => (
              <rect key={'eb' + s} x={BSX + s * SW} y={SY} width={SW - 2} height={28} rx="2" fill="none" stroke="var(--rule)" strokeDasharray="2 3" />
            ))}
          </>
        ) : null}
        <Cells x={sBx} y={SY} w={SW} h={28} vals={f.sB.map((c) => c.v)} states={f.sB.map((c) => c.s)} name="BS" kind="s" />
        {f.exact && f.sA.length + f.sB.length ? (
          <g>
            <text x={sAx + (f.sA.length * SW) / 2} y={SY + 44} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{f.sA.length ? tr(`from A`, `từ A`) : ''}</text>
            <text x={sBx + (f.sB.length * SW) / 2} y={SY + 44} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{f.sB.length ? tr(`from B`, `từ B`) : ''}</text>
          </g>
        ) : null}
        {f.threads ? (
          <text x={W / 2} y={SY + 60} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }} data-threads>{f.threads}</text>
        ) : null}

        <text x={8} y={CY - 10} fontSize="11" style={{ fill: 'var(--global)' }} fontWeight="600"><Translated vi="C, bộ nhớ toàn cục">C, global memory</Translated></text>
        <g data-order={f.C.map((x) => (x ? tag(x) : '·')).join(' ')}>
          {f.C.map((x, k) => (
            <g key={k}>
              <rect x={cx0 + k * cw} y={CY} width={cw - 2} height={22} rx="2" fill={x ? 'var(--global-wash)' : 'var(--cell)'} stroke={x ? 'var(--global)' : 'var(--rule)'} />
              {x ? (
                <>
                  <text x={cx0 + k * cw + (cw - 2) / 2} y={CY + 15} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink)' }}>{x.key}</text>
                  <text x={cx0 + k * cw + (cw - 2) / 2} y={CY + 33} fontSize="8" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{tag(x)}</text>
                </>
              ) : null}
            </g>
          ))}
        </g>
        {f.counter ? (
          <text x={W / 2} y={CY + 60} fontSize="12.5" fontWeight="600" textAnchor="middle" data-counter>{f.counter}</text>
        ) : null}
        <text x={W / 2} y={318} fontSize="13.5" fontWeight="600" textAnchor="middle" data-headline>{f.headline}</text>
      </g>
    </svg>
  );
}

/** The exact-load scheme on the first tile: search, load exactly, merge. */
function exactFrame(p: number): Frame {
  const t = EXACT.tiles[0];
  const searched = p >= 0.15;
  const loaded = p >= 0.35;
  const merged = p >= 0.55;
  const done = p >= 0.8;
  const [a0, a1] = t.aRange;
  const [b0, b1] = t.bRange;
  const gA: GState[] = UNEVEN.A.map((_, g) => (loaded && g >= a0 && g < a1 ? 'load' : ''));
  const gB: GState[] = UNEVEN.B.map((_, g) => (loaded && g >= b0 && g < b1 ? 'load' : ''));
  const sA = loaded ? UNEVEN.A.slice(a0, a1).map((v) => ({ v, s: (merged ? 'used' : 'loaded') as SState })) : [];
  const sB = loaded ? UNEVEN.B.slice(b0, b1).map((v) => ({ v, s: (merged ? 'used' : 'loaded') as SState })) : [];
  const C = EXACT.output.map((x, k) => (merged && k < t.cRange[1] ? x : null));
  return {
    beat: 5,
    A: UNEVEN.A,
    B: UNEVEN.B,
    gA,
    gB,
    sA,
    sB,
    exact: true,
    C,
    arrows: loaded && !merged ? 1 : 0,
    threads: searched ? tr(`search on global memory first: the first ${t.cRange[1]} outputs take ${a1 - a0} of A and ${b1 - b0} of B`, `tìm trên bộ nhớ toàn cục trước: ${t.cRange[1]} đầu ra đầu tiên lấy ${a1 - a0} của A và ${b1 - b0} của B`) : undefined,
    counter: done
      ? tr(`whole merge: loaded ${EXACT.totals.loaded}, used ${EXACT.totals.used}; one search per tile, ${EXACT.totals.extraSearchReads} extra reads in all`, `cả phép trộn: nạp ${EXACT.totals.loaded}, dùng ${EXACT.totals.used}; mỗi ô một lần tìm, tổng cộng thêm ${EXACT.totals.extraSearchReads} lần đọc`)
      : loaded
        ? tr(`loaded so far: ${a1 - a0 + b1 - b0}   used so far: ${merged ? t.used : 0}`, `đã nạp: ${a1 - a0 + b1 - b0}   đã dùng: ${merged ? t.used : 0}`)
        : undefined,
    headline: done ? tr('every loaded element is used: what CUB, rocPRIM and MLX do', 'mọi phần tử đã nạp đều được dùng: cách CUB, rocPRIM và MLX làm') : tr('search first, then load exactly what the tile will use', 'tìm trước, rồi nạp đúng những gì ô sẽ dùng'),
    data: {
      scheme: 'exact',
      it: 0,
      phase: merged ? 'use' : loaded ? 'load' : 'search',
      aloaded: loaded ? a1 - a0 : '',
      bloaded: loaded ? b1 - b0 : '',
      loaded: done ? EXACT.totals.loaded : loaded ? a1 - a0 + b1 - b0 : 0,
      used: done ? EXACT.totals.used : merged ? t.used : 0,
      extra: done ? EXACT.totals.extraSearchReads : '',
    },
  };
}

const beats: Beat[] = [
  {
    clip: 'tile-waste-1',
    get label() {
      return tr('Load a tile', 'Nạp một ô');
    },
    draw: (p) => {
      if (p < 0.35) {
        const f = bookFrame(1, RUN, UNEVEN.A, UNEVEN.B, 0, false, 0, tr(`one block, the next ${IT[0].cOut} outputs: how many come from A, how many from B?`, `một khối, ${IT[0].cOut} đầu ra tiếp theo: bao nhiêu đến từ A, bao nhiêu từ B?`));
        return <Stage {...f} gA={f.gA.map(() => '')} gB={f.gB.map(() => '')} sA={[]} sB={[]} arrows={0} counter={undefined}
          data={{ ...f.data, phase: 'before', aloaded: 0, bloaded: 0, loaded: 0 }} />;
      }
      const f = bookFrame(1, RUN, UNEVEN.A, UNEVEN.B, 0, false, 0, tr(`it cannot know yet, so it loads ${IT[0].aLoaded} of A and ${IT[0].bLoaded} of B`, `nó chưa thể biết, nên nó nạp ${IT[0].aLoaded} của A và ${IT[0].bLoaded} của B`));
      const shown = Math.round(TILE * ramp(p, 0.35, 0.7));
      return <Stage {...f} sA={f.sA.slice(0, shown)} sB={f.sB.slice(0, shown)} />;
    },
  },
  {
    clip: 'tile-waste-2',
    get label() {
      return tr('Use what the merge needs', 'Dùng những gì phép trộn cần');
    },
    draw: (p) => {
      const merged = p >= 0.3;
      const wasted = IT[0].aLoaded + IT[0].bLoaded - IT[0].cOut;
      return (
        <Stage {...bookFrame(2, RUN, UNEVEN.A, UNEVEN.B, 0, merged, ramp(p, 0.3, 0.6),
          !merged ? tr('each thread searches the tile for its own 4 outputs', 'mỗi luồng tìm trong ô cho 4 đầu ra của riêng nó')
            : p < 0.7 ? tr(`used: ${IT[0].aUsed} of A and ${IT[0].bUsed} of B`, `đã dùng: ${IT[0].aUsed} của A và ${IT[0].bUsed} của B`)
              : tr(`${wasted} of the ${IT[0].aLoaded + IT[0].bLoaded} loaded elements were never used`, `${wasted} trong ${IT[0].aLoaded + IT[0].bLoaded} phần tử đã nạp chưa bao giờ được dùng`))} />
      );
    },
  },
  {
    clip: 'tile-waste-3',
    get label() {
      return tr('Load again', 'Nạp lại');
    },
    draw: (p) => {
      const merged = p >= 0.5;
      const again = IT[0].aLoaded - IT[0].aUsed + (IT[0].bLoaded - IT[0].bUsed);
      return (
        <Stage {...bookFrame(3, RUN, UNEVEN.A, UNEVEN.B, 1, merged, ramp(p, 0.5, 0.7),
          !merged ? tr(`the next tiles start at A[${IT[1].aStart}] and B[${IT[1].bStart}]: ${again} elements are loaded a second time`, `các ô tiếp theo bắt đầu ở A[${IT[1].aStart}] và B[${IT[1].bStart}]: ${again} phần tử bị nạp lần thứ hai`)
            : p < 0.8 ? tr(`this time: ${IT[1].aUsed} of A and ${IT[1].bUsed} of B`, `lần này: ${IT[1].aUsed} của A và ${IT[1].bUsed} của B`)
              : tr(`the whole merge: ${RUN.totals.loaded} elements loaded for ${RUN.totals.used} outputs`, `cả phép trộn: nạp ${RUN.totals.loaded} phần tử cho ${RUN.totals.used} đầu ra`))} />
      );
    },
  },
  {
    clip: 'tile-waste-4',
    get label() {
      return tr('The worst case', 'Trường hợp xấu nhất');
    },
    draw: (p) => {
      const n = p < 0.5 ? 0 : 1;
      const local = n === 0 ? ramp(p, 0.15, 0.4) : ramp(p, 0.6, 0.85);
      const merged = local > 0;
      const it = IT_W[n];
      return (
        <Stage {...bookFrame(4, RUN_W, WORST.A, WORST.B, n, merged, local,
          !merged ? tr(`another merge: tile ${n} loads ${it.aLoaded} of A and ${it.bLoaded} of B`, `một phép trộn khác: ô ${n} nạp ${it.aLoaded} của A và ${it.bLoaded} của B`)
            : tr(`all ${it.cOut} outputs from A: a smaller A tile could not have written them`, `cả ${it.cOut} đầu ra đều từ A: một ô A nhỏ hơn đã không thể ghi chúng`))} />
      );
    },
  },
  {
    clip: 'tile-waste-5',
    get label() {
      return tr('Search first', 'Tìm trước');
    },
    draw: (p) => <Stage {...exactFrame(p)} />,
  },
];

export function TileWasteScene() {
  return <Scene chapter="ch13" title={tr('Loaded twice, used once', 'Nạp hai lần, dùng một lần')} beats={beats} />;
}
