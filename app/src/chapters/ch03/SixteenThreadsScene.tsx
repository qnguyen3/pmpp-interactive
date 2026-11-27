import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr as trText } from '../../lib/i18n';
import { blockOrder, makeProblem, threadFor, trace, type YX } from '../../sim/naivemm';

// "One kernel, sixteen threads": the 4 x 4 multiplication of book Figs. 3.12 and
// 3.13 with BLOCK_WIDTH 2, drawn from the same model as the hand-trace stepper
// (src/sim/naivemm.ts). Six beats, one narration clip each
// (content/ch03/narration/sixteen-threads-N.txt). It stops at the mapping: how
// often each input is fetched is chapter 5's scene.
const W = 4;
const BW = 2;
const prob = makeProblem(W);
const C = 36; // cell of the 2D views
const MX = 24, MY = 200; // M origin
const NX = 200, NY = 30; // N origin
const PX = 200, PY = 200; // P origin
const RX = 384; // right-hand panel
const SC = 22; // cell of the flat strips
/** Left edge of slot i of a strip: a small gap after every row of the matrix. */
const slotX = (i: number) => RX + i * SC + Math.floor(i / W) * 5;

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const same = (a: YX, b: YX) => a.y === b.y && a.x === b.x;

/** Every thread of the grid in launch order, with the element the model says it owns. */
const THREADS = blockOrder(W / BW, W / BW, 'in-order').flatMap((block) =>
  [0, 1, 2, 3].map((t) => {
    const thread = { y: Math.floor(t / BW), x: t % BW };
    return { block, thread, ...threadFor(BW, block, thread) };
  }),
);
const B00: YX = { y: 0, x: 0 };
/** The other three blocks, in a shuffled order taken from the model. */
const LATER = blockOrder(W / BW, W / BW, 'shuffled', 1).filter((b) => !same(b, B00));
const inBlock = (b: YX) => THREADS.filter((t) => same(t.block, b));
const label = (b: YX) => `(${b.y},${b.x})`;

interface Frame {
  /** Opacity of the block outlines on P. */
  blocks?: number;
  /** Beat 1: how many threads have been placed, and whether to name the blocks. */
  dots?: number;
  blockLabels?: boolean;
  activeBlock?: YX;
  hotM?: Set<number>;
  hotN?: Set<number>;
  washM?: Set<number>;
  washN?: Set<number>;
  /** Elements of P whose threads are running. */
  hotP?: Set<number>;
  /** Elements of P that have been stored. */
  stored?: Set<number>;
  /** Opacity of the two flat strips. */
  strips?: number;
  sums?: { label: string; value: number }[];
  lines?: ReactNode[];
  linesY?: number;
  code?: string;
  headline?: ReactNode;
}

function Grid({ x, y, values, name, hot, wash, shown }: {
  x: number; y: number; values: number[]; name: string; hot?: Set<number>; wash?: Set<number>; shown?: Set<number>;
}) {
  return (
    <g>
      <text x={x} y={y - 7} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}>{name}</text>
      {values.map((v, i) => {
        const r = Math.floor(i / W);
        const c = i % W;
        const isHot = hot?.has(i);
        const isWash = wash?.has(i);
        return (
          <g key={i}>
            <rect x={x + c * C} y={y + r * C} width={C - 2} height={C - 2} rx="2"
              fill={isHot ? 'var(--global)' : isWash ? 'var(--global-wash)' : 'var(--cell)'}
              stroke={isHot || isWash ? 'var(--global)' : 'var(--rule)'} />
            {!shown || shown.has(i) ? (
              <text x={x + c * C + C / 2 - 1} y={y + r * C + C / 2 + 3} fontSize="12" textAnchor="middle"
                style={{ fill: isHot ? 'var(--page)' : 'var(--ink)' }}>{v}</text>
            ) : null}
          </g>
        );
      })}
    </g>
  );
}

function Strip({ y, values, name, hot, wash }: { y: number; values: number[]; name: string; hot?: Set<number>; wash?: Set<number> }) {
  return (
    <g>
      <text x={RX} y={y - 7} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}>{name}</text>
      {values.map((v, i) => {
        const isHot = hot?.has(i);
        const isWash = wash?.has(i);
        return (
          <g key={i}>
            <rect x={slotX(i)} y={y} width={SC - 2} height={24} rx="2"
              fill={isHot ? 'var(--global)' : isWash ? 'var(--global-wash)' : 'var(--cell)'}
              stroke={isHot || isWash ? 'var(--global)' : 'var(--rule)'} />
            <text x={slotX(i) + SC / 2 - 1} y={y + 16} fontSize="11" textAnchor="middle"
              style={{ fill: isHot ? 'var(--page)' : 'var(--ink)' }}>{v}</text>
            <text x={slotX(i) + SC / 2 - 1} y={y + 35} fontSize="8.5" textAnchor="middle"
              style={{ fill: isHot ? 'var(--ink)' : 'var(--ink-3)' }} fontWeight={isHot ? 700 : 400}>{i}</text>
          </g>
        );
      })}
    </g>
  );
}

function Stage(f: Frame) {
  const linesY = f.linesY ?? 252;
  return (
    <svg viewBox="0 0 760 372" role="img" aria-label={trText('Animation of sixteen threads multiplying two 4 by 4 matrices', 'Hoạt hình mười sáu luồng nhân hai ma trận 4 × 4')}
      style={{ width: '100%', maxWidth: 760, display: 'block' }}>
      <Grid x={MX} y={MY} values={prob.M} name="M" hot={f.hotM} wash={f.washM} />
      <Grid x={NX} y={NY} values={prob.N} name="N" hot={f.hotN} wash={f.washN} />
      <Grid x={PX} y={PY} values={prob.ref} name="P = M × N" wash={f.hotP} shown={f.stored ?? new Set()} />
      {/* blocks of the grid, drawn on the output they cover */}
      {f.blocks
        ? blockOrder(W / BW, W / BW, 'in-order').map((b) => {
            const on = f.activeBlock && same(f.activeBlock, b);
            return (
              <g key={label(b)} opacity={f.blocks}>
                <rect x={PX + b.x * BW * C - 2} y={PY + b.y * BW * C - 2} width={BW * C + 2} height={BW * C + 2}
                  fill="none" stroke="var(--ink)" strokeWidth={on ? 3.4 : 1.8} />
                {f.blockLabels ? (
                  <g>
                    <rect x={PX + b.x * BW * C + C - 31} y={PY + b.y * BW * C + C - 9} width={60} height={16} rx="3"
                      fill="var(--page)" stroke="var(--rule)" />
                    <text x={PX + b.x * BW * C + C - 1} y={PY + b.y * BW * C + C + 3} fontSize="9.5" textAnchor="middle">
                      <Translated vi={<>Khối {label(b)}</>}>Block {label(b)}</Translated>
                    </text>
                  </g>
                ) : null}
              </g>
            );
          })
        : null}
      {/* one dot per thread, pushed to the outer corner of its cell so the block's name stays readable */}
      {f.dots
        ? THREADS.slice(0, f.dots).map((t, i) => (
            <circle key={i} r={i === f.dots! - 1 ? 6 : 4.5}
              cx={PX + t.col * C + (t.thread.x === 0 ? 11 : C - 13)} cy={PY + t.row * C + (t.thread.y === 0 ? 10 : C - 12)}
              fill={i === f.dots! - 1 ? 'var(--ink)' : 'var(--ink-2)'} />
          ))
        : null}
      {f.strips ? (
        <g opacity={f.strips}>
          <Strip y={48} values={prob.M} name={trText('M as stored: 16 floats, row after row', 'M khi lưu: 16 float, hết hàng này đến hàng khác')} hot={f.hotM} wash={f.washM} />
          <Strip y={118} values={prob.N} name={trText('N as stored', 'N khi lưu')} hot={f.hotN} wash={f.washN} />
        </g>
      ) : null}
      {f.sums ? (
        <g>
          <text x={RX} y={186} fontSize="12" fontWeight="600" style={{ fill: 'var(--reg)' }}>
            {f.sums.length > 1 ? trText('Pvalue: one register per thread', 'Pvalue: mỗi luồng một thanh ghi (register)') : trText('Pvalue: a register of this thread', 'Pvalue: một thanh ghi (register) của luồng này')}
          </text>
          {f.sums.map((s, i) => (
            <g key={i}>
              <rect x={RX + i * 92} y={194} width={86} height={34} rx="2" fill="var(--reg-wash)" stroke="var(--reg)" />
              <text x={RX + i * 92 + 6} y={206} fontSize="8.5" style={{ fill: 'var(--ink-2)' }}>{s.label}</text>
              <text x={RX + i * 92 + 43} y={222} fontSize="13" textAnchor="middle">{s.value}</text>
            </g>
          ))}
        </g>
      ) : null}
      {(f.lines ?? []).map((ln, i) => (
        <text key={i} x={RX} y={linesY + i * 21} fontSize="12.5">{ln}</text>
      ))}
      {f.code ? (
        <text x={RX} y={300} fontSize="11.5" style={{ fontVariationSettings: "'MONO' 1" }}>{f.code}</text>
      ) : null}
      {f.headline ? <text x={RX} y={352} fontSize="13" fontWeight="600">{f.headline}</text> : null}
    </svg>
  );
}

/**
 * The threads of one block part-way through their loops. kf runs from 0 to W:
 * while it is below W, step floor(kf) is the one being executed by every thread.
 */
function marching(block: YX, kf: number): Pick<Frame, 'hotM' | 'hotN' | 'washM' | 'washN' | 'hotP' | 'sums' | 'activeBlock'> & { k: number; done: boolean } {
  const threads = inBlock(block);
  const done = kf >= W;
  const k = Math.min(W - 1, Math.floor(kf));
  const hotM = new Set<number>();
  const hotN = new Set<number>();
  const washM = new Set<number>();
  const washN = new Set<number>();
  const sums: { label: string; value: number }[] = [];
  for (const t of threads) {
    const tr = trace(prob, t.row, t.col);
    for (const s of tr.steps) {
      washM.add(s.mIndex);
      washN.add(s.nIndex);
    }
    if (!done) {
      hotM.add(tr.steps[k].mIndex);
      hotN.add(tr.steps[k].nIndex);
    }
    sums.push({ label: `P[${t.row}][${t.col}]`, value: tr.steps[k].sum });
  }
  return { hotM, hotN, washM, washN, hotP: new Set(threads.map((t) => t.row * W + t.col)), sums, activeBlock: block, k, done };
}
const cellsOfBlock = (b: YX) => inBlock(b).map((t) => t.row * W + t.col);
const ALL_P = new Set(prob.ref.map((_, i) => i));

function beats(loopLine: string): Beat[] {
  return [
    {
      clip: 'sixteen-threads-1',
      label: trText('Sixteen threads', 'Mười sáu luồng'),
      draw: (p) => {
        const n = Math.min(16, Math.floor(ramp(p, 0.24, 0.9) * 16.999));
        const t = n > 0 ? THREADS[n - 1] : null;
        return (
          <Stage blocks={ramp(p, 0.04, 0.18)} blockLabels dots={n} linesY={64}
            lines={
              t
                ? [
                    trText(`Thread ${label(t.thread)} of Block ${label(t.block)}`, `Luồng ${label(t.thread)} của Khối ${label(t.block)}`),
                    `row = ${t.block.y}·${BW} + ${t.thread.y} = ${t.row}`,
                    `col = ${t.block.x}·${BW} + ${t.thread.x} = ${t.col}`,
                    trText(`it owns P[${t.row}][${t.col}]`, `nó sở hữu P[${t.row}][${t.col}]`),
                  ]
                : [trText('P is 4 by 4. BLOCK_WIDTH is 2.', 'P có kích thước 4 × 4. BLOCK_WIDTH là 2.'), trText('4 blocks of 2 × 2 threads.', '4 khối, mỗi khối 2 × 2 luồng.')]
            }
            headline={trText(`${n} of 16 threads placed, one per element of P`, `đã đặt ${n} trên 16 luồng, mỗi phần tử của P một luồng`)} />
        );
      },
    },
    {
      clip: 'sixteen-threads-2',
      label: trText('One thread', 'Một luồng'),
      draw: (p) => {
        const started = p >= 0.28;
        const kf = ramp(p, 0.28, 0.94) * W;
        const tr = trace(prob, 0, 0);
        const k = Math.min(W - 1, Math.floor(kf));
        const s = tr.steps[k];
        const m = new Set(tr.steps.map((x) => x.mIndex));
        const n = new Set(tr.steps.map((x) => x.nIndex));
        return (
          <Stage blocks={1} activeBlock={B00} washM={m} washN={n} hotP={new Set([0])}
            hotM={started && kf < W ? new Set([s.mIndex]) : undefined} hotN={started && kf < W ? new Set([s.nIndex]) : undefined}
            sums={[{ label: 'P[0][0]', value: started ? s.sum : 0 }]} linesY={64}
            lines={
              started
                ? [
                    trText('Thread (0,0) of Block (0,0): row 0 of M, column 0 of N', 'Luồng (0,0) của Khối (0,0): hàng 0 của M, cột 0 của N'),
                    `k = ${k}:  Pvalue += M[0][${k}] × N[${k}][0]`,
                    `${s.sum - s.product} + ${s.mValue} × ${s.nValue} = ${s.sum}`,
                  ]
                : [trText('Thread (0,0) of Block (0,0): row 0 of M, column 0 of N', 'Luồng (0,0) của Khối (0,0): hàng 0 của M, cột 0 của N'), trText('Pvalue starts at 0', 'Pvalue bắt đầu từ 0')]
            }
            headline={kf >= W ? trText(`after 4 steps, Pvalue = ${tr.store!.value}`, `sau 4 bước, Pvalue = ${tr.store!.value}`) : trText('one multiply and one add per step', 'mỗi bước một phép nhân và một phép cộng')} />
        );
      },
    },
    {
      clip: 'sixteen-threads-3',
      label: trText('In memory', 'Trong bộ nhớ'),
      draw: (p) => {
        const started = p >= 0.3;
        const kf = ramp(p, 0.3, 0.9) * W;
        const tr = trace(prob, 0, 0);
        const k = Math.min(W - 1, Math.floor(kf));
        const s = tr.steps[k];
        const seen = tr.steps.filter((x) => started && x.k <= k);
        return (
          <Stage blocks={1} activeBlock={B00} strips={ramp(p, 0.02, 0.16)} hotP={new Set([0])} linesY={200}
            washM={new Set(seen.map((x) => x.mIndex))} washN={new Set(seen.map((x) => x.nIndex))}
            hotM={started && kf < W ? new Set([s.mIndex]) : undefined} hotN={started && kf < W ? new Set([s.nIndex]) : undefined}
            lines={
              started
                ? [`k = ${k}`, `M[row·4 + k] = M[0·4 + ${k}] = M[${s.mIndex}]`, `N[k·4 + col] = N[${k}·4 + 0] = N[${s.nIndex}]`]
                : [trText('The same thread, the same loop.', 'Cùng luồng đó, cùng vòng lặp đó.'), trText('Now watch the index, not the row and column.', 'Giờ hãy nhìn chỉ số, không phải hàng và cột.')]
            }
            headline={
              kf >= W
                ? `M: ${tr.steps.map((x) => x.mIndex).join(', ')}.  N: ${tr.steps.map((x) => x.nIndex).join(', ')}.`
                : trText('M moves 1 slot per step; N moves a whole row', 'M dịch 1 ô mỗi bước; N dịch trọn một hàng')
            } />
        );
      },
    },
    {
      clip: 'sixteen-threads-4',
      label: trText('One block', 'Một khối'),
      draw: (p) => {
        const kf = ramp(p, 0.22, 0.8) * W;
        const f = marching(B00, kf);
        const storing = p >= 0.86;
        return (
          <Stage blocks={1} strips={1} {...f} hotP={storing ? undefined : f.hotP} stored={storing ? new Set(cellsOfBlock(B00)) : undefined}
            lines={[
              trText('Block (0,0): 4 threads at the same k', 'Khối (0,0): 4 luồng ở cùng một k'),
              f.done ? trText('the loop is over: each thread stores its sum', 'vòng lặp kết thúc: mỗi luồng ghi tổng của nó') : trText(`k = ${f.k}: rows 0 and 1 of M, columns 0 and 1 of N`, `k = ${f.k}: hàng 0 và 1 của M, cột 0 và 1 của N`),
            ]}
            headline={storing ? trText('4 elements of P written, one per thread', 'đã ghi 4 phần tử của P, mỗi luồng một phần tử') : trText('2 rows of M and 2 columns of N in use', 'đang dùng 2 hàng của M và 2 cột của N')} />
        );
      },
    },
    {
      clip: 'sixteen-threads-5',
      label: trText('Any order', 'Thứ tự bất kỳ'),
      draw: (p) => {
        const SEG = 0.22;
        const START = 0.1;
        const stored = new Set(cellsOfBlock(B00));
        let running: ReturnType<typeof marching> | null = null;
        const ran: YX[] = [];
        LATER.forEach((b, i) => {
          const q = ramp(p, START + i * SEG, START + (i + 1) * SEG);
          if (q >= 0.8) {
            cellsOfBlock(b).forEach((c) => stored.add(c));
            ran.push(b);
          } else if (q > 0) running = marching(b, (q / 0.8) * W);
        });
        const r = running as ReturnType<typeof marching> | null;
        return (
          <Stage blocks={1} strips={1} stored={stored} hotM={r?.hotM} hotN={r?.hotN} washM={r?.washM} washN={r?.washN}
            hotP={r?.hotP} sums={r?.sums} activeBlock={r?.activeBlock}
            lines={[
              trText(`Block order here: ${LATER.map(label).join(', ')}`, `Thứ tự khối ở đây: ${LATER.map(label).join(', ')}`),
              r ? trText(`Block ${label(r.activeBlock!)} is running, k = ${r.k}`, `Khối ${label(r.activeBlock!)} đang chạy, k = ${r.k}`) : ran.length === LATER.length ? trText('all four blocks have finished', 'cả bốn khối đã xong') : trText('Block (0,0) has finished', 'Khối (0,0) đã xong'),
            ]}
            headline={p > 0.8 ? trText('no thread waited for another', 'không luồng nào phải chờ luồng khác') : trText(`${stored.size} of 16 elements of P stored`, `đã ghi ${stored.size} trên 16 phần tử của P`)} />
        );
      },
    },
    {
      clip: 'sixteen-threads-6',
      label: trText('The kernel', 'Kernel'),
      draw: (p) => (
        <Stage blocks={1} strips={1} stored={ALL_P} code={loopLine}
          lines={[trText('16 threads, each on its own element of P', '16 luồng, mỗi luồng lo một phần tử riêng của P'), trText(`${W} multiply-adds and 1 store per thread`, `mỗi luồng ${W} phép nhân-cộng và 1 lần ghi`)]}
          headline={p > 0.55 ? trText('Chapter 5 asks: how often was each input fetched?', 'Chương 5 hỏi: mỗi đầu vào đã được nạp bao nhiêu lần?') : trText('the whole kernel is this loop, once per thread', 'toàn bộ kernel là vòng lặp này, mỗi luồng chạy một lần')} />
      ),
    },
  ];
}

/** `loopLine` is line 08 of the kernel as the section shows it, so the scene cannot drift from the listing. */
export function SixteenThreadsScene({ loopLine }: { loopLine: string }) {
  return <Scene chapter="ch03" title={trText('One kernel, sixteen threads', 'Một kernel, mười sáu luồng')} beats={beats(loopLine)} />;
}
