import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { BOOK_F, BOOK_LAYER, BOOK_X, directTrace, readsAtStep, readsOfThread, type DirectRead } from '../../sim/cnnlayer';

// "What the direct kernel reads": the book's example (Fig. 19.2) launched as Fig. 19.7 would
// launch it with TILE_WIDTH 2: two blocks of four threads. Every number drawn comes from
// cnnlayer.ts's directTrace. Six beats, one narration clip each
// (content/ch19/narration/direct-reads-N.txt).

const S = BOOK_LAYER;
const TR = directTrace(S, BOOK_X, BOOK_F, 2);
const B0 = TR.blocks[0];
const B1 = TR.blocks[1];
const STEPS = S.C * S.K * S.K; // 12
const MAP = S.H * S.W; // 9

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

// Layout (viewBox 720 x 360).
const GX = 22; // grid of blocks
const GY = 44;
const BW = 92; // block box
const XX = 252; // input maps
const XY = 44;
const XC = 30; // input cell
const FX = 572; // filters
const FY = 44;
const FC = 20; // filter cell
const YX = 252; // outputs
const YY = 208;
const YC = 26;

interface Frame {
  /** Reads done so far, in order: they colour the heat map and the counters. */
  done: DirectRead[];
  /** The reads happening now (lit). */
  now?: DirectRead[];
  /** Opacity of each block. */
  blocks?: [number, number];
  /** Which block is running. */
  active?: number;
  /** Register sums shown for the active block, by thread. */
  regs?: (number | null)[];
  headline?: ReactNode;
  /** Shown under the input maps. */
  note?: ReactNode;
  /** Final totals panel. */
  totals?: number;
  /** Data for checks. */
  data?: Record<string, string | number>;
}

function heat(reads: DirectRead[]) {
  const h = new Array<number>(BOOK_X.length).fill(0);
  for (const r of reads) h[r.xi]++;
  return h;
}

function Stage(f: Frame) {
  const counts = heat(f.done);
  const maxHeat = Math.max(1, ...counts);
  const hotX = new Set((f.now ?? []).map((r) => r.xi));
  const hotF = new Set((f.now ?? []).map((r) => r.fi));
  const stored = new Map<string, number>();
  for (const b of TR.blocks)
    for (let t = 0; t < 4; t++) {
      const rs = readsOfThread(b, t);
      const last = rs[rs.length - 1];
      if (f.done.includes(last)) stored.set(`${b.bx},${last.h},${last.w}`, last.sum);
    }
  const xReads = f.done.length;
  const fReads = f.done.length;
  const distinct = counts.filter((c) => c > 0).length;
  const op = f.blocks ?? [1, 1];
  const nowThreads = new Set((f.now ?? []).map((r) => `${r.bx}:${r.thread}`));
  return (
    <svg
      viewBox="0 0 720 360"
      role="img"
      aria-label={tr('Animation of the reads made by the direct convolution kernel', 'Hoạt ảnh các lần đọc của kernel tích chập trực tiếp')}
      style={{ width: '100%', maxWidth: 720, display: 'block' }}
      data-xreads={xReads}
      data-freads={fReads}
      data-distinct={distinct}
      {...Object.fromEntries(Object.entries(f.data ?? {}).map(([k, v]) => [`data-${k}`, v] /* i18n-ignore: data attribute name */))}
    >
      {/* the grid */}
      <text x={GX} y={GY - 22} fontSize="12" fontWeight="600"><Translated vi="lưới (grid)">the grid</Translated></text>
      <text x={GX} y={GY - 8} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="x = map đầu ra, y = ô, z = mẫu">x = output map, y = tile, z = sample</Translated></text>
      {[0, 1].map((b) => (
        <g key={b} opacity={op[b]}>
          <rect x={GX + b * (BW + 18)} y={GY} width={BW} height={BW} rx="4" fill="var(--page)"
            stroke={f.active === b ? 'var(--ink)' : 'var(--rule)'} strokeWidth={f.active === b ? 2.2 : 1} />
          {[0, 1, 2, 3].map((t) => {
            const cx = GX + b * (BW + 18) + 26 + (t % 2) * 40;
            const cy = GY + 26 + Math.floor(t / 2) * 40;
            const lit = nowThreads.has(`${b}:${t}`);
            return (
              <g key={t}>
                <circle cx={cx} cy={cy} r="14" fill={lit ? 'var(--ink)' : 'var(--cell)'} stroke="var(--ink-3)" />
                <text x={cx} y={cy + 4} fontSize="10" textAnchor="middle" style={{ fill: lit ? 'var(--page)' : 'var(--ink)' }}>{t % 2},{Math.floor(t / 2)}</text>
              </g>
            );
          })}
          <text x={GX + b * (BW + 18) + BW / 2} y={GY + BW + 16} fontSize="11" textAnchor="middle"><Translated vi={<>khối ({b}, 0, 0)</>}>block ({b}, 0, 0)</Translated></text>
          <text x={GX + b * (BW + 18) + BW / 2} y={GY + BW + 30} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>map đầu ra {b}</>}>output map {b}</Translated></text>
        </g>
      ))}

      {/* registers of the running block */}
      {f.regs ? (
        <g>
          <text x={GX} y={YY - 4} fontSize="12" fontWeight="600" style={{ fill: 'var(--reg)' }}>
            <Translated vi={<>thanh ghi của khối {f.active ?? 0}</>}>registers of block {f.active ?? 0}</Translated>
          </text>
          {f.regs.map((s, t) => (
            <g key={t}>
              <rect x={GX + (t % 2) * 100} y={YY + 6 + Math.floor(t / 2) * 34} width={92} height={28} rx="2" fill="var(--reg-wash)" stroke="var(--reg)" />
              <text x={GX + (t % 2) * 100 + 46} y={YY + 25 + Math.floor(t / 2) * 34} fontSize="12" textAnchor="middle">
                <Translated vi={<>acc {s == null ? '0' : s}</>}>acc {s == null ? '0' : s}</Translated>
              </text>
            </g>
          ))}
        </g>
      ) : null}

      {/* input maps with fetch counts */}
      <text x={XX} y={XY - 22} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}><Translated vi="X: ba map đầu vào (bộ nhớ toàn cục)">X: three input maps (global memory)</Translated></text>
      {[0, 1, 2].map((c) => (
        <g key={c}>
          {Array.from({ length: MAP }, (_, i) => {
            const xi = c * MAP + i;
            const x = XX + c * (3 * XC + 14) + (i % 3) * XC;
            const y = XY + Math.floor(i / 3) * XC;
            const hot = hotX.has(xi);
            const n = counts[xi];
            return (
              <g key={i} data-x={xi} data-count={n}>
                <rect x={x} y={y} width={XC - 2} height={XC - 2} rx="2"
                  fill={hot ? 'var(--global)' : n ? 'var(--global-wash)' : 'var(--cell)'}
                  fillOpacity={hot ? 1 : n ? 0.35 + (0.65 * n) / maxHeat : 1}
                  stroke={hot || n ? 'var(--global)' : 'var(--rule)'} />
                <text x={x + XC / 2 - 3} y={y + XC / 2 + 6} fontSize="12" textAnchor="middle" style={{ fill: hot ? 'var(--page)' : 'var(--ink)' }}>
                  {BOOK_X[xi]}
                </text>
                {n ? (
                  <text x={x + XC - 4} y={y + 9} fontSize="8" textAnchor="end" style={{ fill: hot ? 'var(--page)' : 'var(--ink-2)' }}>×{n}</text>
                ) : null}
              </g>
            );
          })}
          <text x={XX + c * (3 * XC + 14)} y={XY + 3 * XC + 12} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>X{c}</text>
        </g>
      ))}
      {f.note ? <text x={XX} y={XY + 3 * XC + 30} fontSize="11.5">{f.note}</text> : null}

      {/* filters */}
      <text x={FX} y={FY - 24} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}><Translated vi="F: sáu bộ lọc">F: six filters</Translated></text>
      {[0, 1].map((m) =>
        [0, 1, 2].map((c) => (
          <g key={`${m}${c}`}>
            {[0, 1, 2, 3].map((i) => {
              const fi = (m * 3 + c) * 4 + i;
              const x = FX + c * (2 * FC + 8) + (i % 2) * FC;
              const y = FY + m * (2 * FC + 22) + Math.floor(i / 2) * FC;
              const hot = hotF.has(fi);
              return (
                <g key={i}>
                  <rect x={x} y={y} width={FC - 2} height={FC - 2} rx="2" fill={hot ? 'var(--global)' : 'var(--cell)'} stroke={hot ? 'var(--global)' : 'var(--rule)'} />
                  <text x={x + FC / 2 - 1} y={y + FC / 2 + 3} fontSize="10" textAnchor="middle" style={{ fill: hot ? 'var(--page)' : 'var(--ink)' }}>{BOOK_F[fi]}</text>
                </g>
              );
            })}
          </g>
        )),
      )}
      {[0, 1].map((m) => (
        <text key={m} x={FX} y={FY + m * (2 * FC + 22) - 5} fontSize="10" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>cho map đầu ra {m}</>}>for output map {m}</Translated></text>
      ))}

      {/* outputs */}
      <text x={YX} y={YY - 4} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}><Translated vi="Y: mỗi luồng ghi một lần">Y: stored once per thread</Translated></text>
      {[0, 1].map((m) => (
        <g key={m}>
          {[0, 1, 2, 3].map((i) => {
            const h = Math.floor(i / 2);
            const w = i % 2;
            const v = stored.get(`${m},${h},${w}`);
            const x = YX + m * (2 * YC + 16) + w * YC;
            const y = YY + 6 + h * YC;
            return (
              <g key={i}>
                <rect x={x} y={y} width={YC - 2} height={YC - 2} rx="2" fill={v != null ? 'var(--global-wash)' : 'var(--cell)'} stroke="var(--rule)" />
                {v != null ? <text x={x + YC / 2 - 1} y={y + YC / 2 + 3} fontSize="11" textAnchor="middle">{v}</text> : null}
              </g>
            );
          })}
          <text x={YX + m * (2 * YC + 16)} y={YY + 6 + 2 * YC + 12} fontSize="10" style={{ fill: 'var(--ink-2)' }}>Y{m}</text>
        </g>
      ))}

      {/* counters */}
      <g>
        <text x={430} y={YY + 18} fontSize="12"><Translated vi="số lần đọc đầu vào">input reads</Translated> <tspan fontWeight="650">{xReads}</tspan></text>
        <text x={430} y={YY + 38} fontSize="12"><Translated vi="số giá trị đầu vào khác nhau">distinct input values</Translated> <tspan fontWeight="650">{distinct}</tspan></text>
        <text x={430} y={YY + 58} fontSize="12"><Translated vi="số lần đọc bộ lọc">filter reads</Translated> <tspan fontWeight="650">{fReads}</tspan></text>
      </g>

      {f.totals != null ? (
        <g opacity={f.totals}>
          <text x={GX} y={300} fontSize="11.5"><Translated vi="giá trị đầu vào đã nạp">input values fetched</Translated></text>
          <rect x={GX + 150} y={290} width={(TR.xReads / TR.xReads) * 380} height={13} fill="var(--global)" />
          <text x={GX + 536} y={301} fontSize="11.5">{TR.xReads}</text>
          <text x={GX} y={322} fontSize="11.5"><Translated vi="giá trị đầu vào khác nhau">distinct input values</Translated></text>
          <rect x={GX + 150} y={312} width={(TR.distinctX / TR.xReads) * 380} height={13} fill="var(--global-wash)" stroke="var(--global)" />
          <text x={GX + 150 + (TR.distinctX / TR.xReads) * 380 + 6} y={323} fontSize="11.5">{TR.distinctX}</text>
        </g>
      ) : null}
      {f.headline ? (
        <text x={GX} y={350} fontSize="13" fontWeight="600">{f.headline}</text>
      ) : null}
    </svg>
  );
}

const thread0 = readsOfThread(B0, 0);
const regsAfter = (block: typeof B0, reads: DirectRead[]) =>
  [0, 1, 2, 3].map((t) => {
    const mine = reads.filter((r) => r.bx === block.bx && r.thread === t);
    return mine.length ? mine[mine.length - 1].sum : null;
  });
/** Block reads in step order (all threads at step 0, then step 1, ...): how a warp of four runs. */
const stepOrder = (b: typeof B0, threads: number[]) =>
  Array.from({ length: STEPS }, (_, k) => readsAtStep(b, k).filter((r) => threads.includes(r.thread))).flat();

const beats: Beat[] = [
  {
    clip: 'direct-reads-1',
    get label() {
      return tr('Two blocks', 'Hai khối');
    },
    draw: (p) => (
      <Stage done={[]} blocks={[ramp(p, 0.05, 0.3), ramp(p, 0.3, 0.55)]}
        headline={tr(`gridDim (${TR.grid.x}, ${TR.grid.y}, ${TR.grid.z}): one block of ${TR.tile} × ${TR.tile} threads per output map`, `gridDim (${TR.grid.x}, ${TR.grid.y}, ${TR.grid.z}): mỗi map đầu ra một khối ${TR.tile} × ${TR.tile} luồng`)}
        data={{ beat: 1, blocks: TR.blocks.length }} />
    ),
  },
  {
    clip: 'direct-reads-2',
    get label() {
      return tr('One thread', 'Một luồng');
    },
    draw: (p) => {
      const k = Math.min(STEPS, Math.floor(ramp(p, 0.08, 0.92) * STEPS + (p >= 0.92 ? 1 : 0)));
      const done = thread0.slice(0, k);
      const now = k > 0 && k <= STEPS ? [thread0[k - 1]] : [];
      const sum = k ? thread0[k - 1].sum : 0;
      return (
        <Stage done={done} now={now} active={0} regs={[sum, null, null, null]}
          headline={k ? tr(`step ${k} of ${STEPS}: c = ${thread0[k - 1].c}, p = ${thread0[k - 1].p}, q = ${thread0[k - 1].q}, acc = ${sum}`, `bước ${k} trên ${STEPS}: c = ${thread0[k - 1].c}, p = ${thread0[k - 1].p}, q = ${thread0[k - 1].q}, acc = ${sum}`) : tr('thread (0, 0) of block 0 computes Y0 (0, 0)', 'luồng (0, 0) của khối 0 tính Y0 (0, 0)')}
          data={{ beat: 2, step: k, sum }} />
      );
    },
  },
  {
    clip: 'direct-reads-3',
    get label() {
      return tr('Four threads', 'Bốn luồng');
    },
    draw: (p) => {
      const rest = stepOrder(B0, [1, 2, 3]);
      const k = Math.min(STEPS, Math.floor(ramp(p, 0.05, 0.85) * STEPS + (p >= 0.85 ? 1 : 0)));
      const done = [...thread0, ...rest.slice(0, 3 * k)];
      const now = k > 0 ? rest.slice(3 * (k - 1), 3 * k) : [];
      return (
        <Stage done={done} now={p < 0.85 ? now : []} active={0} regs={regsAfter(B0, done)}
          note={p >= 0.85 ? tr(`block 0: ${B0.xReads} reads of ${B0.distinctX} values. Centre ×4, edges ×2, corners ×1.`, `khối 0: ${B0.xReads} lần đọc cho ${B0.distinctX} giá trị. Ô giữa ×4, ô cạnh ×2, ô góc ×1.`) : undefined}
          headline={p >= 0.85 ? tr('neighbouring patches overlap, so the same inputs are fetched again', 'các vùng lân cận chồng lên nhau, nên cùng những đầu vào bị nạp lại') : tr(`the other three threads, step ${k} of ${STEPS}`, `ba luồng còn lại, bước ${k} trên ${STEPS}`)}
          data={{ beat: 3, step: k }} />
      );
    },
  },
  {
    clip: 'direct-reads-4',
    get label() {
      return tr('One weight', 'Một trọng số');
    },
    draw: (p) => {
      const k = Math.min(STEPS - 1, Math.floor(p * STEPS));
      const now = readsAtStep(B0, k);
      const addresses = new Set(now.map((r) => r.fi)).size;
      return (
        <Stage done={B0.reads} now={now} active={0} regs={regsAfter(B0, B0.reads)}
          note={tr(`step ${k + 1}: ${now.length} threads, ${addresses} filter address`, `bước ${k + 1}: ${now.length} luồng, ${addresses} địa chỉ bộ lọc`)}
          headline={tr('every step: four input reads, one filter weight shared by all four threads', 'mỗi bước: bốn lần đọc đầu vào, một trọng số bộ lọc dùng chung cho cả bốn luồng')}
          data={{ beat: 4, step: k + 1, faddresses: addresses, fthreads: now.length }} />
      );
    },
  },
  {
    clip: 'direct-reads-5',
    get label() {
      return tr('Block 1', 'Khối 1');
    },
    draw: (p) => {
      const order = stepOrder(B1, [0, 1, 2, 3]);
      const k = Math.min(STEPS, Math.floor(ramp(p, 0.05, 0.85) * STEPS + (p >= 0.85 ? 1 : 0)));
      const done = [...B0.reads, ...order.slice(0, 4 * k)];
      const now = k > 0 && p < 0.85 ? order.slice(4 * (k - 1), 4 * k) : [];
      return (
        <Stage done={done} now={now} active={1} regs={regsAfter(B1, done)}
          note={p >= 0.85 ? tr(`block 1 fetched all ${B1.xReads} again: shared memory is private to a block`, `khối 1 nạp lại cả ${B1.xReads}: bộ nhớ chia sẻ (shared memory) là riêng của từng khối`) : undefined}
          headline={p >= 0.85 ? tr('every other output map repeats the whole input traffic', 'mỗi map đầu ra khác lặp lại toàn bộ lưu lượng đầu vào') : tr(`block 1, output map 1, step ${k} of ${STEPS}`, `khối 1, map đầu ra 1, bước ${k} trên ${STEPS}`)}
          data={{ beat: 5, step: k }} />
      );
    },
  },
  {
    clip: 'direct-reads-6',
    get label() {
      return tr('Totals', 'Tổng cộng');
    },
    draw: (p) => (
      <Stage done={[...B0.reads, ...B1.reads]} totals={ramp(p, 0.05, 0.3)}
        note={tr(`${TR.xReads} input reads and ${TR.fReads} filter reads for ${TR.flops} operations`, `${TR.xReads} lần đọc đầu vào và ${TR.fReads} lần đọc bộ lọc cho ${TR.flops} phép toán`)}
        headline={p > 0.45 ? tr(`requested: ${TR.flops} FLOP / ${TR.loadBytes} bytes = ${TR.ratio} FLOP per byte`, `đã yêu cầu: ${TR.flops} FLOP / ${TR.loadBytes} byte = ${TR.ratio} FLOP mỗi byte`) : tr(`${TR.xReads} input reads for ${TR.distinctX} distinct values`, `${TR.xReads} lần đọc đầu vào cho ${TR.distinctX} giá trị khác nhau`)}
        data={{ beat: 6, ratio: TR.ratio }} />
    ),
  },
];

export function DirectReadsScene() {
  return <Scene chapter="ch19" title={tr('What the direct kernel reads', 'Kernel trực tiếp đọc những gì')} beats={beats} />;
}
