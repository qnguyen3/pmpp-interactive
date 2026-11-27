import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { tr, Translated } from '../../lib/i18n';
import { FIG17_1, exercise2Count, idleLanes, matrixRuns, nnz, rowLists, spmvTrace, storage, toCSR, toELL } from '../../sim/sparse';

// "Pad, turn, flatten": CSR's scattered, divergent reads on Fig. 17.1, then the ELL
// construction (pad to the longest row, read the rectangle down its columns, lay it flat)
// and the same four threads reading one contiguous run per iteration (book pp. 409 to 414,
// Figs. 17.8, 17.10, 17.11). Every array, read and idle lane is computed by
// src/sim/sparse.ts (toCSR, toELL, spmvTrace, matrixRuns, storage); the closing numbers
// are the book's 1000 x 1000 example through exercise2Count. Six beats, one narration clip
// each (content/ch17/narration/ell-N.txt).
const M = FIG17_1;
const csr = toCSR(M);
const ell = toELL(M);
const csrTrace = spmvTrace(csr);
const ellTrace = spmvTrace(ell);
const rows = rowLists(M);
const R = M.rows;
const W = ell.width;
const bill = storage(ell, M);
const book = exercise2Count(1000, 10000, 200); // pp. 412 to 413: 1% non-zeros, one row of 200

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const ease = (v: number) => v * v * (3 - 2 * v);

const CW = 40; // array cell
const CH = 30;
const AX = 120; // array origin x
const THREAD_Y = 232;
const threadX = (r: number) => 160 + r * 92;
const ARROW = 'ell-scene-arrow';

function Cell({ x, y, text, pad, hot, dim, w = CW - 3, h = CH - 3, size = 12 }: {
  x: number; y: number; text: string; pad?: boolean; hot?: boolean; dim?: boolean; w?: number; h?: number; size?: number;
}) {
  return (
    <g opacity={dim ? 0.35 : 1}>
      <rect x={x} y={y} width={w} height={h} rx="2"
        fill={hot ? 'var(--global)' : pad ? 'transparent' : 'var(--global-wash)'}
        stroke={pad ? 'var(--ink-3)' : 'var(--global)'} strokeDasharray={pad ? '3 2' : undefined} />
      <text x={x + w / 2} y={y + h / 2 + 4} fontSize={size} textAnchor="middle"
        style={{ fill: hot ? 'var(--page)' : pad ? 'var(--ink-3)' : 'var(--ink)' }}>{text}</text>
    </g>
  );
}

/** A flat value array with index numbers under it. */
function Flat({ y, labels, pads, hot, upto = labels.length, name, bounds }: {
  y: number; labels: string[]; pads: boolean[]; hot?: Set<number>; upto?: number; name: string; bounds?: number[];
}) {
  return (
    <g data-flat={labels.slice(0, upto).join(' ')}>
      <text x={AX - 10} y={y + 19} fontSize="12" textAnchor="end" style={{ fill: 'var(--global)' }}>{name}</text>
      {labels.map((l, i) => (
        <g key={i} opacity={i < upto ? 1 : 0}>
          <Cell x={AX + i * CW} y={y} text={l} pad={pads[i]} hot={hot?.has(i)} />
          <text x={AX + i * CW + (CW - 3) / 2} y={y + CH + 12} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{i}</text>
        </g>
      ))}
      {bounds?.map((b) => (
        <line key={b} x1={AX + b * CW - 1.5} x2={AX + b * CW - 1.5} y1={y - 6} y2={y + CH + 2} stroke="var(--ink-2)" strokeDasharray="2 3" />
      ))}
    </g>
  );
}

/** The four threads, each with an arrow to the slot it reads, or greyed when idle. */
function Threads({ reads, arrayY, show = 1, label, label2 }: { reads: (number | null)[]; arrayY: number; show?: number; label: string; label2?: string }) {
  return (
    <g data-reads={reads.map((i) => (i == null ? '-' : i)).join(' ')} data-idle={reads.filter((i) => i == null).length}>
      {reads.map((i, r) => {
        const x = threadX(r);
        const on = r < Math.ceil(show * R - 1e-9);
        return (
          <g key={r}>
            {i != null && on ? (
              <line x1={x} y1={THREAD_Y - 4} x2={AX + i * CW + (CW - 3) / 2} y2={arrayY + CH + 16}
                stroke="var(--ink)" strokeWidth="1.4" markerEnd={`url(#${ARROW})`} />
            ) : null}
            <rect x={x - 40} y={THREAD_Y} width={80} height={24} rx="3"
              fill={i == null ? 'transparent' : 'var(--page)'} stroke={i == null ? 'var(--ink-3)' : 'var(--ink)'}
              strokeDasharray={i == null ? '3 2' : undefined} />
            <text x={x} y={THREAD_Y + 16} fontSize="11.5" textAnchor="middle" style={{ fill: i == null ? 'var(--ink-3)' : 'var(--ink)' }}>
              {i == null ? tr(`thread ${r}: idle`, `luồng ${r}: rảnh`) : tr(`thread ${r}`, `luồng ${r}`)}
            </text>
          </g>
        );
      })}
      <text x={AX} y={THREAD_Y + 44} fontSize="12" data-line="">{label}</text>
      {label2 ? <text x={AX} y={THREAD_Y + 60} fontSize="12" style={{ fill: 'var(--ink-2)' }} data-line="">{label2}</text> : null}
    </g>
  );
}

function Stage({ children, headline }: { children: ReactNode; headline: string }) {
  return (
    <svg viewBox="0 0 720 340" role="img" aria-label={tr('ELL built from CSR, step by step', 'ELL dựng từ CSR, từng bước')} style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      <defs>
        <marker id={ARROW} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse" /* i18n-ignore: SVG keyword */>
          <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--ink)" />
        </marker>
      </defs>
      {children}
      <text x={24} y={330} fontSize="13" fontWeight="600" data-headline="">{headline}</text>
    </svg>
  );
}

const csrLabels = csr.at.map((e) => e.label);
const csrBounds = csr.rowPtrs.slice(1, -1);
const ellLabels = ell.at.map((e) => (e ? e.label : 'pad'));
const ellPads = ell.at.map((e) => e === null);
const readsAt = (steps: typeof csrTrace.steps, t: number) => steps[t].lanes.map((l) => l.index);
const placesAt = (steps: typeof csrTrace.steps, t: number) => matrixRuns(steps[t], R)[0];

function CsrBeat({ t, show }: { t: number; show: number }) {
  const reads = readsAt(csrTrace.steps, t);
  const idle = idleLanes(csrTrace.steps[t]);
  return (
    <g data-iter={t} data-runs={placesAt(csrTrace.steps, t)}>
      <text x={AX} y={40} fontSize="12" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi="CSR, một luồng mỗi hàng; các vạch đứt là nơi các hàng bắt đầu (rowPtrs)">CSR, one thread per row; dashed lines are where rows begin (rowPtrs)</Translated>
      </text>
      <Flat y={70} name="value" /* i18n-ignore: the array's name in the code */ labels={csrLabels} pads={csrLabels.map(() => false)} bounds={csrBounds}
        hot={new Set(reads.filter((i, r): i is number => i != null && r < Math.ceil(show * R - 1e-9)))} />
      <Threads reads={reads} arrayY={70} show={show}
        label={tr(
          `iteration ${t}: reads ${reads.filter((i) => i != null).map((i) => `value[${i}]`).join(', ')}` +
          ` (${placesAt(csrTrace.steps, t)} separate places)` + (idle ? `; ${idle} of ${R} threads idle` : ''),
          `lần lặp ${t}: đọc ${reads.filter((i) => i != null).map((i) => `value[${i}]`).join(', ')}` +
            ` (${placesAt(csrTrace.steps, t)} chỗ riêng rẽ)` + (idle ? `; ${idle} trên ${R} luồng rảnh` : ''),
        )} />
    </g>
  );
}

const RX = 260; // rectangle origin
const RY = 64;
const RC = 44;

function PadBeat({ p }: { p: number }) {
  const slide = ease(ramp(p, 0.08, 0.45));
  const padIn = ramp(p, 0.5, 0.8);
  const outline = ramp(p, 0.42, 0.6);
  return (
    <g data-pads={bill.pads} data-slots={bill.values}>
      <text x={RX} y={RY - 26} fontSize="12" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi="các phần tử khác 0 của mỗi hàng trượt sang trái; rồi phần đệm lấp đầy hình chữ nhật">each row's non-zeros slide left; then pads fill the rectangle</Translated>
      </text>
      {Array.from({ length: R }, (_, r) => (
        <text key={r} x={RX - 12} y={RY + r * RC + 25} fontSize="11.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>hàng {r}</>}>row {r}</Translated></text>
      ))}
      {/* the matrix's grid, fading as the rectangle forms */}
      {Array.from({ length: R * M.cols }, (_, k) => (
        <rect key={k} x={RX + (k % M.cols) * RC} y={RY + Math.floor(k / M.cols) * RC} width={RC - 4} height={RC - 4} rx="2"
          fill="none" stroke="var(--rule)" opacity={1 - outline * 0.8} />
      ))}
      {rows.map((l, r) => l.map((e, t) => {
        const x = RX + (e.c + (t - e.c) * slide) * RC;
        return <Cell key={`${r},${t}`} x={x} y={RY + r * RC} text={e.label} w={RC - 4} h={RC - 4} />;
      }))}
      {rows.map((l, r) => Array.from({ length: W - l.length }, (_, k) => (
        <g key={`p${r},${k}`} opacity={padIn}>
          <Cell x={RX + (l.length + k) * RC} y={RY + r * RC} text="−1 / 0" pad w={RC - 4} h={RC - 4} size={9.5} />
        </g>
      )))}
      <rect x={RX - 3} y={RY - 3} width={W * RC + 2} height={R * RC + 2} fill="none" stroke="var(--ink)" strokeWidth="2" opacity={outline} />
      <text x={RX + W * RC + 30} y={RY + 40} fontSize="12" opacity={padIn}><Translated vi="phần đệm: cột −1, giá trị 0">pad: column −1, value 0</Translated></text>
      <text x={RX + W * RC + 30} y={RY + 60} fontSize="12" opacity={padIn} data-line="">
        <Translated vi={<>{bill.pads} phần đệm cho {nnz(M)} phần tử khác 0</>}>{bill.pads} pads for {nnz(M)} non-zeros</Translated>
      </text>
    </g>
  );
}

function TurnBeat({ p }: { p: number }) {
  const n = R * W;
  const drop = ramp(p, 0.12, 0.86) * n;
  const upto = Math.floor(drop + 1e-9);
  // The path down the columns, drawn as far as the cells dropped so far.
  const pts: string[] = [];
  for (let t = 0; t < W; t++) for (let r = 0; r < R; r++) pts.push(`${RX + t * RC + RC / 2 - 2},${RY - 8 + r * RC + RC / 2}`);
  const reached = Math.max(1, Math.min(n, Math.ceil(drop)));
  return (
    <g data-upto={upto}>
      {Array.from({ length: R }, (_, r) => (
        <text key={r} x={RX - 12} y={RY - 8 + r * RC + 25} fontSize="11.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>hàng {r}</>}>row {r}</Translated></text>
      ))}
      {Array.from({ length: W }, (_, t) => (
        <text key={t} x={RX + t * RC + RC / 2 - 2} y={RY - 22} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>t = {t}</text>
      ))}
      {Array.from({ length: n }, (_, i) => {
        const t = Math.floor(i / ell.stride);
        const r = i % ell.stride;
        return (
          <Cell key={i} x={RX + t * RC} y={RY - 8 + r * RC} text={ellLabels[i] === 'pad' ? '−1 / 0' : ellLabels[i]} pad={ellPads[i]}
            w={RC - 4} h={RC - 4} dim={i < upto} size={ellPads[i] ? 9.5 : 12} />
        );
      })}
      <polyline points={pts.slice(0, reached).join(' ')} fill="none" stroke="var(--ink)" strokeWidth="1.6" opacity={p > 0.1 ? 0.55 : 0} />
      <text x={RX + W * RC + 30} y={RY + 30} fontSize="12">index = t × stride + row</text>
      <text x={RX + W * RC + 30} y={RY + 50} fontSize="12" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi={<>stride = {ell.stride} (numRows của sách)</>}>stride = {ell.stride} (the book's numRows)</Translated>
      </text>
      <Flat y={236} name="value" /* i18n-ignore: the array's name in the code */ labels={ellLabels} pads={ellPads} upto={upto} bounds={p > 0.9 ? [4, 8] : undefined} />
    </g>
  );
}

function RunBeat({ t, show }: { t: number; show: number }) {
  const reads = readsAt(ellTrace.steps, t);
  const active = reads.filter((i): i is number => i != null);
  const runsNow = placesAt(ellTrace.steps, t);
  const lo = t * ell.stride;
  return (
    <g data-iter={t} data-runs={runsNow}>
      <text x={AX} y={40} fontSize="12" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi="ELL, một luồng mỗi hàng, vòng lặp bị chặn bởi nnzPerRow">ELL, one thread per row, the loop bounded by nnzPerRow</Translated>
      </text>
      <Flat y={70} name="value" /* i18n-ignore: the array's name in the code */ labels={ellLabels} pads={ellPads} hot={new Set(active)} bounds={[4, 8]} />
      <rect x={AX + lo * CW - 4} y={62} width={ell.stride * CW + 5} height={CH + 30} rx="4" fill="none" stroke="var(--ink)" strokeWidth="1.6" />
      <Threads reads={reads} arrayY={70} show={show}
        label={tr(
          `iteration ${t}: reads ${active.map((i) => `value[${i}]`).join(', ')}: ${runsNow} contiguous run`,
          `lần lặp ${t}: đọc ${active.map((i) => `value[${i}]`).join(', ')}: ${runsNow} dải liên tiếp`,
        )}
        label2={tr(
          `CSR in the same iteration: ${placesAt(csrTrace.steps, t)} separate place${placesAt(csrTrace.steps, t) === 1 ? '' : 's'}` +
          (idleLanes(ellTrace.steps[t]) ? `; ${idleLanes(ellTrace.steps[t])} of ${R} threads idle, as in CSR` : '; no thread idle'),
          `CSR trong cùng lần lặp: ${placesAt(csrTrace.steps, t)} chỗ riêng rẽ` +
            (idleLanes(ellTrace.steps[t]) ? `; ${idleLanes(ellTrace.steps[t])} trên ${R} luồng rảnh, như ở CSR` : '; không luồng nào rảnh'),
        )} />
    </g>
  );
}

function BillBeat({ p }: { p: number }) {
  const grow = ease(ramp(p, 0.4, 0.8));
  const BW = 300;
  const bars = [
    { k: 'dense', label: tr('dense: 1,000,000 numbers', 'dày đặc: 1,000,000 số'), frac: 1 },
    { k: 'csr', label: tr(`CSR: ${book.csr.toLocaleString('en-US')}`, `CSR: ${book.csr.toLocaleString('en-US')}`), frac: book.csr / 1e6 },
    { k: 'ell', label: tr(`ELL, width 200: ${book.ell.toLocaleString('en-US')}`, `ELL, độ rộng 200: ${book.ell.toLocaleString('en-US')}`), frac: book.ell / 1e6 },
  ];
  return (
    <g data-slots={bill.values} data-pads={bill.pads}>
      <Flat y={44} name="value" /* i18n-ignore: the array's name in the code */ labels={ellLabels} pads={ellPads} bounds={[4, 8]} />
      <text x={AX} y={108} fontSize="12" data-line="">
        <Translated vi={<>{bill.values} vị trí cho {nnz(M)} phần tử khác 0, và colIdx cũng dài chừng đó:</>}>{bill.values} slots for {nnz(M)} non-zeros, and colIdx is as long:</Translated>
      </text>
      <text x={AX} y={124} fontSize="12" data-line="">
        <Translated vi={<>{bill.total} số tính cả nnzPerRow, so với {bill.dense} cho ma trận dày đặc 4 × 4</>}>{bill.total} numbers with nnzPerRow, against {bill.dense} for the dense 4 × 4</Translated>
      </text>
      <g opacity={ramp(p, 0.3, 0.45)}>
        <text x={AX} y={150} fontSize="12" style={{ fill: 'var(--ink-2)' }}>
          <Translated vi="ví dụ của sách: 1000 × 1000, 1% khác 0, một hàng dài 200 (pp. 412–413)">the book's example: 1000 × 1000, 1% non-zero, one row of 200 (pp. 412–413)</Translated>
        </text>
        {bars.map((b, i) => (
          <g key={b.k} data-bar={b.k} data-frac={b.frac.toFixed(3)}>
            <rect x={AX} y={166 + i * 34} width={BW} height={18} fill="none" stroke="var(--rule)" />
            <rect x={AX} y={166 + i * 34} width={Math.max(1.5, BW * b.frac * grow)} height={18} fill={b.k === 'dense' ? 'var(--ink-3)' : 'var(--global)'} />
            <text x={AX + BW + 10} y={180 + i * 34} fontSize="12">{b.label} ({(b.frac * 100).toFixed(1)}%)</text>
          </g>
        ))}
      </g>
    </g>
  );
}

const beats: Beat[] = [
  {
    clip: 'ell-1',
    get label() {
      return tr('CSR reads', 'CSR đọc');
    },
    draw: (p) => (
      <Stage headline={tr('CSR: neighbouring threads, far-apart addresses', 'CSR: luồng kề nhau, địa chỉ cách xa nhau')}>
        <CsrBeat t={0} show={ramp(p, 0.15, 0.6)} />
      </Stage>
    ),
  },
  {
    clip: 'ell-2',
    get label() {
      return tr('Idle lanes', 'Lane rảnh');
    },
    draw: (p) => {
      const t = p < 0.5 ? 1 : 2;
      return (
        <Stage headline={tr("row length sets each thread's trip count: divergence", 'độ dài hàng quyết định số vòng lặp của mỗi luồng: phân kỳ luồng (divergence)')}>
          <CsrBeat t={t} show={1} />
        </Stage>
      );
    },
  },
  {
    clip: 'ell-3',
    get label() {
      return tr('Pad', 'Đệm');
    },
    draw: (p) => <Stage headline={tr(`pad every row to the longest: ${R} × ${W} slots`, `đệm mọi hàng cho dài bằng hàng dài nhất: ${R} × ${W} vị trí`)}><PadBeat p={p} /></Stage>,
  },
  {
    clip: 'ell-4',
    get label() {
      return tr('Turn', 'Xoay');
    },
    draw: (p) => <Stage headline={tr('read down the columns, lay it flat: the place is the address', 'đọc dọc theo các cột, trải phẳng: vị trí chính là địa chỉ')}><TurnBeat p={p} /></Stage>,
  },
  {
    clip: 'ell-5',
    get label() {
      return tr('Run', 'Chạy');
    },
    draw: (p) => {
      const t = Math.min(W - 1, Math.floor(p * W));
      return <Stage headline={tr('one contiguous run per iteration; the idle lanes stay', 'mỗi lần lặp một dải liên tiếp; các lane rảnh vẫn còn đó')}><RunBeat t={t} show={1} /></Stage>;
    },
  },
  {
    clip: 'ell-6',
    get label() {
      return tr('The bill', 'Cái giá');
    },
    draw: (p) => <Stage headline={tr('one long row sets the width for every row', 'một hàng dài quyết định độ rộng cho mọi hàng')}><BillBeat p={p} /></Stage>,
  },
];

export function EllScene() {
  return <Scene chapter="ch17" title={tr('Pad, turn, flatten', 'Đệm, xoay, trải phẳng')} beats={beats} />;
}
