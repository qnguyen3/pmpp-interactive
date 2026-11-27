import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import {
  activeThreads,
  cellAt,
  launchPlan,
  launches,
  loadsOf,
  producerOf,
  slotCell,
  stepsPerTile,
  tileOutline,
  type Geometry,
  type Tile,
} from '../../sim/wavetile';

// "Shearing a tile" (book §16.7, pp. 389–394, Figs. 16.13, 16.14, 16.17). Tile (1, 1) of a
// 9 x 9 interior cut into 3 x 3 tiles, square and then sheared, and the whole table at the
// end. Every cell, step, input and count drawn comes from src/sim/wavetile.ts. Five beats,
// one narration clip each (content/ch16/narration/shear-N.txt).

const SQ: Geometry = { Lseq: 9, W: 3, scheme: 'square' };
const SH: Geometry = { Lseq: 9, W: 3, scheme: 'sheared' };
const TILE: Tile = { row: 1, col: 1 };
const W = SQ.W;

const VW = 720;
const VH = 336;
const C = 44; // a table cell in beats 1 to 4
const X0 = 56; // x of table column 1
const Y0 = 58; // y of table row 3 (the row above the tile)
const R0 = 3; // first table row drawn
const COLS = [1, 2, 3, 4, 5, 6, 7];
const ROWS = [3, 4, 5, 6];
const cx = (q: number) => X0 + (q - 1) * C;
const cy = (r: number) => Y0 + (r - R0) * C;
const RX = 420; // right-hand panel

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const stepOf = (p: number, n: number) => Math.min(n - 1, Math.floor(ramp(p, 0.05, 0.93) * n));

const BUSY_SQ = Array.from({ length: stepsPerTile(SQ) }, (_, k) => activeThreads(SQ, TILE, k).length);
const BUSY_SH = Array.from({ length: stepsPerTile(SH) }, (_, k) => activeThreads(SH, TILE, k).length);

function Frame({ beat, data, headline, children }: { beat: number; data: Record<string, string | number>; headline: string; children: ReactNode }) {
  return (
    <svg viewBox={`0 0 ${VW} ${VH}`} role="img" aria-label={tr('Animation: shearing a tile of the Smith-Waterman table', 'Hoạt hình: trượt (shear) một ô của bảng Smith-Waterman')} style={{ width: '100%', maxWidth: VW, display: 'block' }}>
      <defs>
        <marker id="shear-arrow-s" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">{/* i18n-ignore: SVG attribute value */}
          <path d="M0,0 L8,4 L0,8 z" style={{ fill: 'var(--shared)' }} />
        </marker>
        <marker id="shear-arrow-g" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">{/* i18n-ignore: SVG attribute value */}
          <path d="M0,0 L8,4 L0,8 z" style={{ fill: 'var(--global)' }} />
        </marker>
      </defs>
      <g data-beat={beat} {...Object.fromEntries(Object.entries(data).map(([k, v]) => [`data-${k}`, String(v)]))}>{/* i18n-ignore: data attribute name */}
        {children}
        <text x={VW / 2} y={VH - 10} fontSize="13.5" fontWeight="600" textAnchor="middle" data-headline>{headline}</text>
      </g>
    </svg>
  );
}

/** The faint table around the tile: rows 3 to 6, columns 1 to 7, with their numbers. */
function TableGrid() {
  return (
    <g>
      {COLS.map((q) => (
        <text key={'c' + q} x={cx(q) + C / 2} y={Y0 - 8} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{tr(`q ${q}`, `q ${q}`)}</text>
      ))}
      {ROWS.map((r) => (
        <text key={'r' + r} x={X0 - 8} y={cy(r) + C / 2 + 4} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>{tr(`r ${r}`, `r ${r}`)}</text>
      ))}
      {ROWS.flatMap((r) =>
        COLS.map((q) => (
          <rect key={`${r},${q}`} x={cx(q)} y={cy(r)} width={C - 2} height={C - 2} rx="2" fill="var(--cell)" stroke="var(--rule-soft)" />
        )),
      )}
    </g>
  );
}

type CellLook = 'idle' | 'done' | 'hot' | 'focus' | 'input';
function TileCell({ x, y, label, look, size = C }: { x: number; y: number; label: string; look: CellLook; size?: number }) {
  const fill = look === 'hot' || look === 'focus' ? 'var(--shared)' : look === 'done' || look === 'input' ? 'var(--shared-wash)' : 'var(--cell)';
  return (
    <g>
      <rect x={x} y={y} width={size - 2} height={size - 2} rx="2" fill={fill} stroke="var(--shared)" strokeWidth={look === 'input' || look === 'focus' ? 2.2 : 1.2} />
      <text x={x + (size - 2) / 2} y={y + size / 2 + 3} fontSize="13" textAnchor="middle" fontWeight={look === 'hot' || look === 'focus' ? 650 : 450}
        style={{ fill: look === 'hot' || look === 'focus' ? 'var(--page)' : 'var(--ink)' }}>{label}</text>
    </g>
  );
}

/** Busy and idle threads per step, one column of W lanes per step. */
function BusyBars({ busy, upto, title }: { busy: number[]; upto: number; title: string }) {
  const lane = 18;
  const gap = 46;
  return (
    <g data-bars={busy.slice(0, upto + 1).join(',')}>
      <text x={RX} y={Y0 + 4} fontSize="12" fontWeight="600">{title}</text>
      {busy.map((b, k) => (
        <g key={k} opacity={k <= upto ? 1 : 0.25}>
          {Array.from({ length: W }, (_, i) => {
            const on = i < b;
            return (
              <rect key={i} x={RX + k * gap} y={Y0 + 20 + (W - 1 - i) * (lane + 3)} width={lane * 1.6} height={lane} rx="2"
                fill={on && k <= upto ? 'var(--shared)' : 'none'} stroke={on ? 'var(--shared)' : 'var(--ink-3)'} strokeDasharray={on ? undefined : '3 2'} />
            );
          })}
          <text x={RX + k * gap + lane * 0.8} y={Y0 + 20 + W * (lane + 3) + 12} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{tr(`step ${k}`, `bước ${k}`)}</text>
        </g>
      ))}
      <text x={RX} y={Y0 + 20 + W * (lane + 3) + 34} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{tr('filled: a busy thread; dashed: an idle one', 'tô đậm: luồng đang bận; nét đứt: luồng rảnh')}</text>
    </g>
  );
}

// ---------------------------------------------------------------- beat 1: the square tile

function beat1(p: number) {
  const s = stepOf(p, BUSY_SQ.length);
  const lit = activeThreads(SQ, TILE, s).map((t) => cellAt(SQ, TILE, s, t));
  const busy = BUSY_SQ[s];
  return (
    <Frame beat={1} data={{ step: s, busy, steps: BUSY_SQ.length, lit: lit.map((c) => `${c.r},${c.q}`).join(' ') }}
      headline={tr(`step ${s}: ${busy} busy, ${W - busy} idle; ${W * W} cells take ${BUSY_SQ.length} steps, 0 to ${BUSY_SQ.length - 1}`, `bước ${s}: ${busy} bận, ${W - busy} rảnh; ${W * W} ô cần ${BUSY_SQ.length} bước, từ 0 đến ${BUSY_SQ.length - 1}`)}>
      <TableGrid />
      {Array.from({ length: W * W }, (_, i) => {
        const rt = Math.floor(i / W);
        const qt = i % W;
        const { r, q } = slotCell(SQ, TILE, rt, qt);
        const k = rt + qt; // the step that computes it (square: d_tile = r_tile + q_tile)
        return <TileCell key={i} x={cx(q)} y={cy(r)} label={String(rt)} look={k === s ? 'hot' : k < s ? 'done' : 'idle'} />;
      })}
      <text x={cx(4)} y={cy(7) + 16} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>{tr('tile (1, 1): each cell shows the thread that computes it', 'ô (1, 1): mỗi ô con ghi luồng tính nó')}</text>
      <BusyBars busy={BUSY_SQ} upto={s} title={tr('threads per step, square tile', 'số luồng mỗi bước, ô vuông')} />
    </Frame>
  );
}

// ---------------------------------------------------------------- beat 2: the shear

const DIAGS = Array.from({ length: W }, (_, qt) => [...new Set(Array.from({ length: W }, (_, rt) => { const c = slotCell(SH, TILE, rt, qt); return c.r + c.q; }))]);

function beat2(p: number) {
  const m = ramp(p, 0.1, 0.55);
  const lines = ramp(p, 0.62, 0.8);
  return (
    <Frame beat={2} data={{ shift: m.toFixed(2), diagonals: DIAGS.map((d) => d.join('/')).join(' '), lines: lines > 0 ? 1 : 0 }}
      headline={m < 1 ? tr(`row r_tile slides r_tile cells left: ${Math.round(m * 100)}% of the way`, `hàng r_tile trượt sang trái r_tile ô: đã đi ${Math.round(m * 100)}% quãng đường`) : tr('each tile column now lies on one anti-diagonal of the table', 'giờ mỗi cột của ô nằm trên một đường chéo phụ của bảng')}>
      <TableGrid />
      {Array.from({ length: W * W }, (_, i) => {
        const rt = Math.floor(i / W);
        const qt = i % W;
        const from = slotCell(SQ, TILE, rt, qt);
        const to = slotCell(SH, TILE, rt, qt);
        const x = cx(from.q) + (cx(to.q) - cx(from.q)) * m;
        return <TileCell key={i} x={x} y={cy(from.r)} label={String(rt)} look={lines > 0 ? 'done' : 'idle'} />;
      })}
      {lines > 0
        ? Array.from({ length: W }, (_, qt) => {
            const top = slotCell(SH, TILE, 0, qt);
            const bot = slotCell(SH, TILE, W - 1, qt);
            const x1 = cx(bot.q) + C / 2 - 26;
            const y1 = cy(bot.r) + C / 2 + 26;
            const x2 = cx(top.q) + C / 2 + 24;
            const y2 = cy(top.r) + C / 2 - 24;
            return (
              <g key={qt} opacity={lines}>
                <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--ink-3)" strokeWidth="1.3" strokeDasharray="4 3" />
                <text x={x2 + 3} y={y2 - 3} fontSize="11" fontWeight="600" style={{ fill: 'var(--ink-2)' }}>{qt}</text>
              </g>
            );
          })
        : null}
      <g>
        <text x={RX} y={Y0 + 4} fontSize="12" fontWeight="600">{tr('the shear, factor −1', 'phép trượt, hệ số −1')}</text>
        <text x={RX} y={Y0 + 28} fontSize="12" style={{ fill: 'var(--ink-2)' }}>{tr('square: q = W·tile_col + q_tile + 1', 'vuông: q = W·tile_col + q_tile + 1')}</text>
        <text x={RX} y={Y0 + 48} fontSize="12" style={{ fill: 'var(--ink-2)' }}>{tr('sheared: q = W·tile_col + q_tile − r_tile + 1', 'đã trượt: q = W·tile_col + q_tile − r_tile + 1')}</text>
        {Array.from({ length: W }, (_, rt) => (
          <text key={rt} x={RX} y={Y0 + 82 + rt * 20} fontSize="12">
            <Translated vi={<>hàng {rt} (luồng {rt}): {rt === 0 ? 'đứng yên' : `sang trái ${rt} ô`}</>}>
              row {rt} (thread {rt}): {rt === 0 ? 'stays' : `${rt} cell${rt === 1 ? '' : 's'} left`}
            </Translated>
          </text>
        ))}
        <g opacity={lines}>
          <text x={RX} y={Y0 + 160} fontSize="12" fontWeight="600">{tr('each tile column, one anti-diagonal:', 'mỗi cột của ô, một đường chéo phụ:')}</text>
          {DIAGS.map((dg, qt) => (
            <text key={qt} x={RX} y={Y0 + 182 + qt * 19} fontSize="12" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>cột {qt} của ô: các ô con có r + q = {dg.join(', ')}</>}>tile column {qt}: cells with r + q = {dg.join(', ')}</Translated></text>
          ))}
        </g>
      </g>
    </Frame>
  );
}

// ---------------------------------------------------------------- beat 3: the sheared tile runs

const FOCUS_T = 1; // the thread whose inputs are drawn

function inputsFor(k: number) {
  const c = cellAt(SH, TILE, k, FOCUS_T);
  return loadsOf(SH, c).map((ld) => {
    const pr = producerOf(SH, ld.r, ld.q);
    const same = !!pr && pr.tile.row === TILE.row && pr.tile.col === TILE.col;
    return { ...ld, earlier: same ? k - pr!.dTile : null };
  });
}
const NAME = {
  get n() {
    return tr('north', 'bắc');
  },
  get w() {
    return tr('west', 'tây');
  },
  get nw() {
    return tr('north-west', 'tây bắc');
  },
} as const;

function beat3(p: number) {
  const s = stepOf(p, BUSY_SH.length);
  const lit = activeThreads(SH, TILE, s).map((t) => cellAt(SH, TILE, s, t));
  const focus = cellAt(SH, TILE, s, FOCUS_T);
  const ins = inputsFor(s);
  const arrows = ramp(p, 0.05 + (s / 3) * 0.88 + 0.06, 0.05 + (s / 3) * 0.88 + 0.16);
  return (
    <Frame beat={3} data={{ step: s, busy: lit.length, steps: BUSY_SH.length, lit: lit.map((c) => `${c.r},${c.q}`).join(' '), inputs: ins.map((x) => `${x.name}:${x.from}:${x.earlier ?? 'launch'}`).join(' ') }}
      headline={tr(`step ${s}: all ${lit.length} threads busy; ${W * W} cells take ${BUSY_SH.length} steps, 0 to ${BUSY_SH.length - 1}`, `bước ${s}: cả ${lit.length} luồng đều bận; ${W * W} ô cần ${BUSY_SH.length} bước, từ 0 đến ${BUSY_SH.length - 1}`)}>
      <TableGrid />
      {ins.filter((x) => x.from === 'global').map((x) => (
        <rect key={'g' + x.name} x={cx(x.q)} y={cy(x.r)} width={C - 2} height={C - 2} rx="2" fill="var(--global-wash)" stroke="var(--global)" strokeWidth="1.6" opacity={arrows} />
      ))}
      {Array.from({ length: W * W }, (_, i) => {
        const rt = Math.floor(i / W);
        const qt = i % W;
        const { r, q } = slotCell(SH, TILE, rt, qt);
        const look: CellLook = qt === s ? (rt === FOCUS_T ? 'focus' : 'hot') : qt < s ? (ins.some((x) => x.from === 'tile' && x.r === r && x.q === q) && arrows > 0 ? 'input' : 'done') : 'idle';
        return <TileCell key={i} x={cx(q)} y={cy(r)} label={String(rt)} look={look} />;
      })}
      {ins.map((x) => {
        // From the edge of the input cell to the edge of the focus cell.
        const dx = Math.sign(focus.q - x.q);
        const dy = Math.sign(focus.r - x.r);
        const ax = cx(x.q) + C / 2 - 1;
        const ay = cy(x.r) + C / 2 - 1;
        const bx = cx(focus.q) + C / 2 - 1;
        const by = cy(focus.r) + C / 2 - 1;
        const e = C / 2 - 3;
        return (
          <line key={'a' + x.name} x1={ax + dx * (e - 8)} y1={ay + dy * (e - 8)} x2={bx - dx * e} y2={by - dy * e}
            stroke={x.from === 'tile' ? 'var(--shared)' : 'var(--global)'} strokeWidth="2.4" opacity={arrows}
            markerEnd={`url(#shear-arrow-${x.from === 'tile' ? 's' : 'g'})`} />
        );
      })}
      <BusyBars busy={BUSY_SH} upto={s} title={tr('threads per step, sheared tile', 'số luồng mỗi bước, ô đã trượt')} />
      <g opacity={arrows}>
        <text x={RX} y={Y0 + 164} fontSize="12" fontWeight="600"><Translated vi={<>luồng {FOCUS_T} ở bước {s}, ô ({focus.r}, {focus.q}):</>}>thread {FOCUS_T} at step {s}, cell ({focus.r}, {focus.q}):</Translated></text>
        {ins.map((x, i) => (
          <text key={x.name} x={RX} y={Y0 + 186 + i * 19} fontSize="12" data-input={x.name} style={{ fill: x.from === 'tile' ? 'var(--shared)' : 'var(--global)' }}>
            {NAME[x.name]} ({x.r}, {x.q}): {x.earlier != null ? tr(`this tile, ${x.earlier} step${x.earlier === 1 ? '' : 's'} earlier`, `ô này, sớm hơn ${x.earlier} bước`) : tr('an earlier launch, global memory', 'một lần khởi chạy trước, bộ nhớ toàn cục')}
          </text>
        ))}
      </g>
    </Frame>
  );
}

// ---------------------------------------------------------------- beat 4: stored as a square

const F4 = { rt: 2, k: 2 };
const F4_LOADS = loadsOf(SH, cellAt(SH, TILE, F4.k, F4.rt));
const SQX = 470;
const SQY = 88;
const SC = 52;

function beat4(p: number) {
  const square = ramp(p, 0.08, 0.25);
  const shown = (['n', 'w', 'nw'] as const).filter((_, i) => p >= 0.35 + i * 0.15);
  const focus = slotCell(SH, TILE, F4.rt, F4.k);
  return (
    <Frame beat={4} data={{ focus: `${F4.rt},${F4.k}`, ...Object.fromEntries(F4_LOADS.map((l) => [l.name, l.slot ? l.slot.join(',') : 'global'])), shown: shown.join(' ') }}
      headline={tr('stored column = step: north and west one column left, north-west two', 'cột lưu trữ = bước: bắc và tây lùi một cột, tây bắc lùi hai')}>
      <TableGrid />
      {Array.from({ length: W * W }, (_, i) => {
        const rt = Math.floor(i / W);
        const qt = i % W;
        const { r, q } = slotCell(SH, TILE, rt, qt);
        const isFocus = rt === F4.rt && qt === F4.k;
        const isIn = F4_LOADS.some((l, j) => l.slot && l.slot[0] === rt && l.slot[1] === qt && shown.includes(l.name) && j >= 0);
        return <TileCell key={i} x={cx(q)} y={cy(r)} label={String(rt)} look={isFocus ? 'focus' : isIn ? 'input' : 'done'} />;
      })}
      <text x={cx(focus.q) + C / 2} y={cy(focus.r) + C + 14} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>({focus.r}, {focus.q})</text>
      <g opacity={square}>
        <text x={SQX} y={SQY - 34} fontSize="12" fontWeight="600" style={{ fill: 'var(--shared)' }}>{tr('the tile in shared memory', 'ô trong bộ nhớ chia sẻ')}</text>
        {Array.from({ length: W }, (_, k) => (
          <text key={'h' + k} x={SQX + k * SC + SC / 2 - 1} y={SQY - 10} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{tr(`step ${k}`, `bước ${k}`)}</text>
        ))}
        {Array.from({ length: W }, (_, rt) => (
          <text key={'t' + rt} x={SQX - 8} y={SQY + rt * SC + SC / 2 + 4} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{tr(`thread ${rt}`, `luồng ${rt}`)}</text>
        ))}
        {Array.from({ length: W * W }, (_, i) => {
          const rt = Math.floor(i / W);
          const qt = i % W;
          const isFocus = rt === F4.rt && qt === F4.k;
          const ld = F4_LOADS.find((l) => l.slot && l.slot[0] === rt && l.slot[1] === qt && shown.includes(l.name));
          const { r, q } = slotCell(SH, TILE, rt, qt);
          return (
            <g key={i}>
              <TileCell x={SQX + qt * SC} y={SQY + rt * SC} size={SC} label={`${r},${q}`} look={isFocus ? 'focus' : ld ? 'input' : 'done'} />
              {ld ? <text x={SQX + qt * SC + 4} y={SQY + rt * SC + 12} fontSize="9.5" fontWeight="650" style={{ fill: 'var(--shared)' }}>{ld.name}</text> : null}
            </g>
          );
        })}
        {F4_LOADS.filter((l) => l.slot && shown.includes(l.name)).map((l) => {
          const [rt, qt] = l.slot!;
          const dc = F4.k - qt;
          const dr = F4.rt - rt;
          return (
            <text key={'o' + l.name} x={SQX} y={SQY + W * SC + 22 + ['n', 'w', 'nw'].indexOf(l.name) * 18} fontSize="12" data-offset={l.name}>
              <Translated vi={<>load_{l.name}: {dr ? `r_tile − ${dr}, ` : 'cùng hàng, '}q_tile − {dc}</>}>
                load_{l.name}: {dr ? `r_tile − ${dr}, ` : 'same row, '}q_tile − {dc}
              </Translated>
            </text>
          );
        })}
      </g>
    </Frame>
  );
}

// ---------------------------------------------------------------- beat 5: the whole table

const PC = 19; // a cell in beat 5
const PANELS: { g: Geometry; x: number; readonly label: string }[] = [
  {
    g: SQ,
    x: 44,
    get label() {
      return tr('square tiles', 'ô vuông');
    },
  },
  {
    g: SH,
    x: 392,
    get label() {
      return tr('sheared tiles', 'ô đã trượt');
    },
  },
];
const PY = 46;
const NSQ = launches(SQ);
const NSH = launches(SH);

function Panel({ g, x, label, n }: { g: Geometry; x: number; label: string; n: number }) {
  const L = g.Lseq + 1;
  const total = launches(g);
  const cur = Math.min(n, total - 1);
  const finished = n >= total;
  const plan = launchPlan(g, cur);
  const clip = `shear-clip-${g.scheme}`; // i18n-ignore: element id
  const all = Array.from({ length: total }, (_, d) => launchPlan(g, d));
  const most = Math.max(...all.map((pl) => pl.tiles.length));
  return (
    <g data-panel={g.scheme} data-launch={finished ? total : cur + 1} data-total={total} data-tiles={finished ? 0 : plan.tiles.length} data-most={most}>
      <defs>
        <clipPath id={clip}>
          <rect x={x + PC} y={PY + PC} width={(L - 1) * PC} height={(L - 1) * PC} />
        </clipPath>
      </defs>
      <text x={x} y={PY - 12} fontSize="12.5" fontWeight="600">{label}</text>
      {Array.from({ length: L * L }, (_, i) => {
        const r = Math.floor(i / L);
        const q = i % L;
        const edge = r === 0 || q === 0;
        return <rect key={i} x={x + q * PC} y={PY + r * PC} width={PC - 1} height={PC - 1} fill={edge ? 'none' : 'var(--cell)'} stroke="var(--rule-soft)" strokeDasharray={edge ? '2 2' : undefined} />;
      })}
      <g clipPath={`url(#${clip})`}>
        {all.flatMap((pl) =>
          pl.tiles.map((t) => {
            const pts = tileOutline(g, t).map(([qx, ry]) => `${x + qx * PC},${PY + ry * PC}`).join(' ');
            const state = pl.d < cur || finished ? 'done' : pl.d === cur ? 'hot' : 'wait';
            return (
              <polygon key={`${t.row},${t.col}`} points={pts} data-tile={`${t.row},${t.col}`} data-state={state}
                fill={state === 'hot' ? 'var(--shared)' : state === 'done' ? 'var(--shared-wash)' : 'none'} fillOpacity={state === 'hot' ? 0.75 : 1}
                stroke="var(--ink)" strokeWidth="1.4" />
            );
          }),
        )}
      </g>
      <text x={x} y={PY + L * PC + 22} fontSize="12" data-count>
        {finished ? tr(`done: ${total} launches, at most ${most} tiles in one`, `xong: ${total} lần khởi chạy, nhiều nhất ${most} ô trong một lần`) : tr(`launch ${cur + 1} of ${total}: ${plan.tiles.length} tile${plan.tiles.length === 1 ? '' : 's'}`, `lần khởi chạy ${cur + 1} trên ${total}: ${plan.tiles.length} ô`)}
      </text>
      <g>
        {all.map((pl, d) => (
          <rect key={d} x={x + d * 22} y={PY + L * PC + 34 + (3 - pl.tiles.length) * 9} width={18} height={pl.tiles.length * 9}
            fill={d === cur && !finished ? 'var(--shared)' : d < cur || finished ? 'var(--shared-wash)' : 'none'} stroke="var(--shared)" />
        ))}
        <text x={x + total * 22 + 6} y={PY + L * PC + 58} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>{tr('tiles per launch', 'số ô mỗi lần khởi chạy')}</text>
      </g>
    </g>
  );
}

function beat5(p: number) {
  const n = Math.floor(ramp(p, 0.06, 0.9) * (NSH + 0.999));
  return (
    <Frame beat={5} data={{ n }} headline={n < NSQ ? tr('same table, same tiles, two schedules', 'cùng bảng, cùng các ô, hai lịch chạy') : n < NSH ? tr(`square tiles are done after ${NSQ}; sheared tiles still need ${NSH - n} more`, `ô vuông xong sau ${NSQ} lần; ô đã trượt còn cần thêm ${NSH - n} lần`) : tr(`${NSQ} launches against ${NSH}: the shear's price outside the tile`, `${NSQ} lần khởi chạy so với ${NSH}: cái giá của phép trượt bên ngoài ô`)}>
      {PANELS.map((pn) => <Panel key={pn.g.scheme} {...pn} n={n} />)}
    </Frame>
  );
}

const beats: Beat[] = [
  {
    clip: 'shear-1',
    get label() {
      return tr('A square tile', 'Một ô vuông');
    },
    draw: beat1,
  },
  {
    clip: 'shear-2',
    get label() {
      return tr('Shear it', 'Trượt nó');
    },
    draw: beat2,
  },
  {
    clip: 'shear-3',
    get label() {
      return tr('Every thread busy', 'Mọi luồng đều bận');
    },
    draw: beat3,
  },
  {
    clip: 'shear-4',
    get label() {
      return tr('Stored as a square', 'Lưu dưới dạng hình vuông');
    },
    draw: beat4,
  },
  {
    clip: 'shear-5',
    get label() {
      return tr('The whole table', 'Toàn bộ bảng');
    },
    draw: beat5,
  },
];

export function ShearScene() {
  return <Scene chapter="ch16" title={tr('Shearing a tile', 'Trượt một ô')} beats={beats} />;
}
