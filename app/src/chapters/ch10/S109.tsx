import { useEffect, useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { benchRows, num, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  DESIGN_LABEL,
  VOCABS,
  blockSums,
  blockView,
  f32,
  f32hex,
  f32short,
  f64short,
  history,
  logits,
  reference,
  runOf,
  segments,
  threadLoads,
  ulpDistance,
  type Design,
} from '../../sim/multiblock';
import { functionSource, type FunctionSource } from '../../sim/toolchain';
import { CommitOrderScene } from './CommitOrderScene';
import { useLabStatus } from './status';

const LAB = 'ch10/lab1_sum';
const fmtInt = (n: number) => n.toLocaleString('en-US');

// Identifiers coloured by the memory they live in, as everywhere in the course.
const SUM_TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', output_ref: 'global', in: 'global', out: 'global',
  partialSums_s: 'shared', input_s: 'shared', warpSums: 'shared',
  partialSum: 'reg', segment: 'reg', i: 'reg', t: 'reg', c: 'reg', idx: 'reg', sum: 'reg',
};

/** The Metal kernel in a source file whose body contains `needle`. */
function metalKernelWith(src: string, needle: string): FunctionSource | null {
  for (const m of src.matchAll(/\bkernel\s+void\s+(\w+)\s*\(/g)) {
    const f = functionSource(src, m[1]);
    if (f && f.text.includes(needle)) return f;
  }
  return null;
}
/** 1-based lines of `text` that contain any of the needles. */
const linesWith = (text: string, needles: string[]) =>
  text.split('\n').flatMap((l, i) => (needles.some((n) => l.includes(n)) ? [i + 1] : []));

function CommitListing() {
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  const metal = useLabFile(LAB + '/solution/kernel.metal');
  const status = useLabStatus(LAB);
  const k = cu ? functionSource(cu, 'CoarsenedSumReductionKernel') : null;
  const host = cu ? functionSource(cu, 'launchCoarsenedSum') : null;
  const m = metal ? metalKernelWith(metal, 'atomic_fetch_add_explicit') : null;
  const zeroing = host ? linesWith(host.text, ['emset']) : [];
  if (!k)
    return (
      <p className="muted">
        <Translated
          vi={
            <>
              Kernel được nạp từ <code>labs/{LAB}/solution/kernel.cu</code> khi lab 10.1 đã có sẵn.
            </>
          }
        >
          <>
            The kernel is loaded from <code>labs/{LAB}/solution/kernel.cu</code> once lab 10.1 is in place.
          </>
        </Translated>
      </p>
    );
  const tabs = [
    { label: tr('CUDA kernel', 'Kernel CUDA'), code: k.text, tiers: SUM_TIERS, status: status.cu, highlight: linesWith(k.text, ['atomic_ref', 'fetch_add', 'atomicAdd']) },
    ...(host ? [{ label: tr('CUDA host', 'Host CUDA'), code: host.text, tiers: SUM_TIERS, status: status.cu, highlight: zeroing }] : []),
    ...(m ? [{ label: tr('Metal kernel', 'Kernel Metal'), code: m.text, tiers: SUM_TIERS, status: status.metal, highlight: linesWith(m.text, ['atomic_fetch_add_explicit']) }] : []),
  ];
  return (
    <Code
      tabs={tabs}
      caption={
        <Translated
          vi={
            <>
              Lời giải tham chiếu (solution) của lab 10.1: <code>CoarsenedSumReductionKernel</code> (dòng {k.firstLine} đến{' '}
              {k.lastLine} của <code>kernel.cu</code>){host ? <>, hàm host của nó <code>launchCoarsenedSum</code></> : null}
              {m ? <> và kernel Metal commit bằng <code>atomic_fetch_add_explicit</code></> : null}, từ{' '}
              <code>labs/{LAB}/solution/</code>. Các dòng được đánh dấu là lần commit{zeroing.length ? ' và việc đặt tổng về 0' : ''}.
              Với hệ số gộp việc bằng 1, kernel là Fig. 10.18 được làm an toàn cho mọi độ dài; mục 10.10 giải thích hệ số này.
              Hãy thử lab trước khi nghiên cứu đoạn này.
            </>
          }
        >
          <>
            Lab 10.1's reference solution: <code>CoarsenedSumReductionKernel</code> (lines {k.firstLine} to {k.lastLine} of{' '}
            <code>kernel.cu</code>){host ? <>, its host function <code>launchCoarsenedSum</code></> : null}
            {m ? <> and the Metal kernel that commits with <code>atomic_fetch_add_explicit</code></> : null}, from{' '}
            <code>labs/{LAB}/solution/</code>. The marked lines are the commit{zeroing.length ? ' and the zeroing of the total' : ''}. With a coarsening
            factor of 1 the kernel is Fig. 10.18 made safe for any length; section 10.10 explains the factor. Try the lab before
            you study this.
          </>
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- interactive A: segment calculator
const BLOCK_SIZES = [8, 32, 64, 128, 256, 512, 1024];
const COARSE = [1, 2, 4, 8];

function SegmentCalculator() {
  const [N, setN] = useState(1_000_000);
  const [text, setText] = useState('1000000');
  const [block, setBlock] = useState(1024);
  const [coarse, setCoarse] = useState(1);
  const seg = segments(N, block, coarse);
  const [b, setB] = useState(seg.blocks - 1);
  const [t, setT] = useState(0);
  const [guard, setGuard] = useState(true);
  const [k, setK] = useState(0); // loads shown so far

  const bb = Math.min(b, Math.max(0, seg.blocks - 1));
  const tt = Math.min(t, block - 1);
  const view = blockView(N, block, coarse, bb);
  const loads = threadLoads(N, block, coarse, bb, tt);
  const last = blockView(N, block, coarse, seg.blocks - 1);
  useEffect(() => setK(0), [N, block, coarse, bb, tt, guard]);

  const commitN = (s: string) => {
    setText(s);
    const v = Math.floor(Number(s));
    if (isFinite(v) && v >= 1 && v <= 1e9) {
      setN(v);
      setB(segments(v, block, coarse).blocks - 1);
    }
  };
  const setShape = (nb: number, nc: number) => {
    setBlock(nb);
    setCoarse(nc);
    setB(segments(N, nb, nc).blocks - 1);
    setT(0);
  };

  const cur = k > 0 ? loads[k - 1] : null;
  const say = (l: (typeof loads)[number]) =>
    tr(
      `Load ${l.k}: index = segment + t + ${l.k}·blockDim = ${fmtInt(view.start)} + ${tt} + ${l.k}·${block} = ${fmtInt(l.index)}. ` +
        (l.inRange
          ? `That is below N = ${fmtInt(N)}, so the thread adds input[${fmtInt(l.index)}].`
          : guard
            ? `That is past the end (N = ${fmtInt(N)}): the guard makes the thread add 0 instead.`
            : `That is past the end (N = ${fmtInt(N)}) and nothing stops the read: out of bounds.`),
      `Lần nạp ${l.k}: index = segment + t + ${l.k}·blockDim = ${fmtInt(view.start)} + ${tt} + ${l.k}·${block} = ${fmtInt(l.index)}. ` +
        (l.inRange
          ? `Giá trị đó nhỏ hơn N = ${fmtInt(N)}, nên luồng cộng input[${fmtInt(l.index)}].`
          : guard
            ? `Giá trị đó vượt quá cuối mảng (N = ${fmtInt(N)}): phép kiểm tra khiến luồng cộng 0 thay vào đó.`
            : `Giá trị đó vượt quá cuối mảng (N = ${fmtInt(N)}) và không có gì ngăn lần đọc: ngoài biên.`),
    );

  // Drawing: the whole input on top, the selected segment's 2C rows of blockDim slots below.
  const W = 680;
  const X0 = 20;
  const total = seg.blocks * seg.perBlock; // slots up to the end of the last segment
  const sx = (i: number) => X0 + (i / total) * W;
  const ROW = 22;
  const zx = (j: number) => X0 + (j / block) * W;
  const oob = !guard && view.pastEnd > 0;

  return (
    <Bench
      title={tr('Segment calculator', 'Bộ tính đoạn')}
      task={tr(
        'Pick a length that is not a multiple of the segment. Find the last block, then a thread in it with nothing to load.',
        'Chọn một độ dài không phải bội của đoạn. Tìm khối cuối, rồi một luồng trong đó không có gì để nạp.',
      )}
      foot={
        <Translated
          vi={
            <>
              Thanh trên: toàn bộ đầu vào, mỗi khối một đoạn, vẽ theo tỉ lệ tới cuối đoạn cuối cùng. Dưới: đoạn của khối được
              chọn dưới dạng 2·C hàng gồm <code>blockDim</code> ô, mỗi lần nạp một hàng; vạch đánh dấu luồng được chọn. Các ô
              gạch chéo nằm quá N.
            </>
          }
        >
          <>
            Top bar: the whole input, one segment per block, drawn to scale up to the end of the last segment. Bottom: the selected
            block's segment as 2·C rows of <code>blockDim</code> slots, one row per load; the tick marks the selected thread.
            Hatched slots lie past N.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          N
          <input type="number" min={1} value={text} onChange={(e) => commitN(e.target.value)} style={{ width: '8.5rem' }} aria-label="N" />
        </label>
        <label className="field">
          <Translated vi="Số luồng mỗi khối">Threads per block</Translated>
          <select value={block} onChange={(e) => setShape(+e.target.value, coarse)} aria-label={tr('threads per block', 'số luồng mỗi khối')}>
            {BLOCK_SIZES.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Hệ số gộp việc C">Coarsening factor C</Translated>
          <select value={coarse} onChange={(e) => setShape(block, +e.target.value)} aria-label={tr('coarsening factor', 'hệ số gộp việc')}>
            {COARSE.map((s) => <option key={s} value={s}>{s}{s === 1 ? tr(' (this section)', ' (mục này)') : ''}</option>)}
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={guard} onChange={(e) => setGuard(e.target.checked)} />{' '}
          <Translated vi="kiểm tra biên các lần nạp (bài tập 5)">guard loads (exercise 5)</Translated>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field" style={{ minWidth: '16rem' }}>
          <span>
            <Translated
              vi={
                <>
                  Khối <b data-testid="block">{fmtInt(bb)}</b> trên {fmtInt(seg.blocks)} (0 đến {fmtInt(seg.blocks - 1)})
                </>
              }
            >
              Block <b data-testid="block">{fmtInt(bb)}</b> of {fmtInt(seg.blocks)} (0 to {fmtInt(seg.blocks - 1)})
            </Translated>
          </span>
          <input type="range" min={0} max={Math.max(0, seg.blocks - 1)} value={bb} onChange={(e) => setB(+e.target.value)} aria-label={tr('block index', 'chỉ số khối')} />
        </label>
        <label className="field" style={{ minWidth: '16rem' }}>
          <span>
            <Translated
              vi={
                <>
                  Luồng <b data-testid="thread">{tt}</b> trên {block} (0 đến {block - 1})
                </>
              }
            >
              Thread <b data-testid="thread">{tt}</b> of {block} (0 to {block - 1})
            </Translated>
          </span>
          <input type="range" min={0} max={block - 1} value={tt} onChange={(e) => setT(+e.target.value)} aria-label={tr('thread index', 'chỉ số luồng')} />
        </label>
        <button onClick={() => setB(seg.blocks - 1)}>
          <Translated vi="Khối cuối">Last block</Translated>
        </button>
        <button onClick={() => setT(Math.min(block - 1, last.busyThreads))} disabled={bb !== seg.blocks - 1 || last.idleThreads === 0}>
          <Translated vi="Luồng rảnh đầu tiên">First idle thread</Translated>
        </button>
      </div>

      <div style={{ overflowX: 'auto' }}>
      <svg viewBox={`0 0 ${W + 2 * X0} ${100 + 2 * coarse * ROW}`} role="img" aria-label={tr('The input split into segments', 'Đầu vào được chia thành các đoạn')} style={{ width: '100%', minWidth: 540, display: 'block' }}>
        <defs>
          <pattern id="seg-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--ink-3)" strokeWidth="1.6" />
          </pattern>
          <pattern id="seg-hatch-bad" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--fail)" strokeWidth="1.8" />
          </pattern>
        </defs>
        <text x={X0} y={14} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}>
          <Translated vi={<>input (bộ nhớ toàn cục), {fmtInt(N)} phần tử, {fmtInt(seg.blocks)} đoạn</>}>
            input (global memory), {fmtInt(N)} elements, {fmtInt(seg.blocks)} segments
          </Translated>
        </text>
        <rect x={sx(0)} y={22} width={sx(N) - sx(0)} height={20} fill="var(--global-wash)" stroke="var(--global)" />
        {total > N ? <rect x={sx(N)} y={22} width={sx(total) - sx(N)} height={20} fill={oob ? 'url(#seg-hatch-bad)' : 'url(#seg-hatch)'} stroke="var(--rule)" /> : null}
        {seg.blocks <= 128
          ? Array.from({ length: seg.blocks - 1 }, (_, i) => (
              <line key={i} x1={sx((i + 1) * seg.perBlock)} x2={sx((i + 1) * seg.perBlock)} y1={22} y2={42} stroke="var(--global)" strokeWidth="0.8" />
            ))
          : null}
        <rect x={sx(view.start)} y={18} width={Math.max(2, sx(view.end) - sx(view.start))} height={28} fill="none" stroke="var(--ink)" strokeWidth="2" />
        <text x={Math.min(W + X0, sx(view.start))} y={60} fontSize="11.5" textAnchor={sx(view.start) > W * 0.7 ? 'end' : 'start'}>
          <Translated vi={<>khối {fmtInt(bb)}: các phần tử {fmtInt(view.start)} đến {fmtInt(view.end - 1)}</>}>
            block {fmtInt(bb)}: elements {fmtInt(view.start)} to {fmtInt(view.end - 1)}
          </Translated>
        </text>

        <text x={X0} y={88} fontSize="12" fontWeight="600">
          <Translated vi={<>đoạn của khối {fmtInt(bb)}, mỗi lần nạp một hàng (k = 0 .. {2 * coarse - 1})</>}>
            block {fmtInt(bb)}'s segment, one row per load (k = 0 .. {2 * coarse - 1})
          </Translated>
        </text>
        {Array.from({ length: 2 * coarse }, (_, r) => {
          const rowStart = view.start + r * block;
          const realHere = Math.max(0, Math.min(block, N - rowStart));
          const y = 96 + r * ROW;
          return (
            <g key={r} data-row={r}>
              <rect x={zx(0)} y={y} width={zx(realHere) - zx(0)} height={ROW - 6} fill="var(--global-wash)" stroke="var(--global)" />
              {realHere < block ? (
                <rect x={zx(realHere)} y={y} width={zx(block) - zx(realHere)} height={ROW - 6} fill={oob ? 'url(#seg-hatch-bad)' : 'url(#seg-hatch)'} stroke="var(--rule)" />
              ) : null}
              <line x1={zx(tt + 0.5)} x2={zx(tt + 0.5)} y1={y - 2} y2={y + ROW - 4}
                stroke={cur && cur.k === r ? (cur.inRange ? 'var(--ok)' : guard ? 'var(--ink)' : 'var(--fail)') : 'var(--ink)'} strokeWidth={cur && cur.k === r ? 3 : 1.6} />
            </g>
          );
        })}
      </svg>
      </div>

      <div className="row" style={{ margin: '0.6rem 0 0.8rem' }}>
        <button onClick={() => setK(0)} disabled={k === 0}>
          <Translated vi="Bắt đầu lại">Restart</Translated>
        </button>
        <button className="primary" onClick={() => setK(Math.min(loads.length, k + 1))} disabled={k >= loads.length}>
          <Translated vi="Bước">Step</Translated>
        </button>
        <button onClick={() => setK(loads.length)} disabled={k >= loads.length}>
          <Translated vi="Mọi lần nạp">All loads</Translated>
        </button>
        <span className="muted">
          <Translated vi={<>lần nạp {k} trên {loads.length}</>}>
            load {k} of {loads.length}
          </Translated>
        </span>
      </div>
      <div className={'narrate' + (cur && !cur.inRange && !guard ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }}>
        {cur ? say(cur) : (
          <Translated
            vi={
              <>
                Luồng {tt} của khối {fmtInt(bb)} chưa nạp gì. Đoạn của nó bắt đầu ở 2·C·blockDim·b = 2·{coarse}·{block}·{fmtInt(bb)} ={' '}
                {fmtInt(view.start)}. Bấm Bước.
              </>
            }
          >
            <>Thread {tt} of block {fmtInt(bb)} has loaded nothing yet. Its segment starts at 2·C·blockDim·b = 2·{coarse}·{block}·{fmtInt(bb)} = {fmtInt(view.start)}. Press Step.</>
          </Translated>
        )}
      </div>

      <div className="cols">
        <div className="panel">
          <table className="data" style={{ margin: 0 }} aria-label={tr("This thread's loads", 'Các lần nạp của luồng này')}>
            <thead>
              <tr>
                <th><Translated vi="lần nạp k">load k</Translated></th>
                <th>index{/* i18n-ignore: variable name */}</th>
                <th><Translated vi="nhỏ hơn N?">below N?</Translated></th>
                <th><Translated vi="cộng">adds</Translated></th>
              </tr>
            </thead>
            <tbody>
              {loads.map((l) => (
                <tr key={l.k} data-load={l.k} data-index={l.index} data-in={l.inRange ? 1 : 0} style={{ opacity: l.k < k ? 1 : 0.45 }}>
                  <td>{l.k}</td>
                  <td className="mono">{fmtInt(l.index)}</td>
                  <td>{l.inRange ? tr('yes', 'có') : tr('no', 'không')}</td>
                  <td>
                    {l.inRange ? <>input[{fmtInt(l.index)}]{/* i18n-ignore: array expression */}</> : guard ? tr('0 (guarded)', '0 (đã kiểm tra)') : (
                      <span className="t-fail">
                        <Translated vi="ngoài biên">out of bounds</Translated>
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel" data-testid="facts">
          <p style={{ margin: 0 }}>
            <Translated
              vi={
                <>
                  Số khối: ceil({fmtInt(N)} / (2·{coarse}·{block})) = <b data-testid="blocks">{fmtInt(seg.blocks)}</b>.
                </>
              }
            >
              Blocks: ceil({fmtInt(N)} / (2·{coarse}·{block})) = <b data-testid="blocks">{fmtInt(seg.blocks)}</b>.
            </Translated>
          </p>
          <p style={{ margin: '0.4rem 0 0' }}>
            <Translated
              vi={
                <>
                  Khối {fmtInt(bb)} chứa <b data-testid="real">{fmtInt(view.real)}</b> phần tử thật trên {fmtInt(seg.perBlock)};{' '}
                  {view.idleThreads > 0 ? <>{fmtInt(view.idleThreads)} luồng của nó không có gì để nạp.</> : <>mọi luồng đều có việc.</>}
                </>
              }
            >
              Block {fmtInt(bb)} holds <b data-testid="real">{fmtInt(view.real)}</b> real elements of {fmtInt(seg.perBlock)};{' '}
              {view.idleThreads > 0 ? <>{fmtInt(view.idleThreads)} of its threads have nothing to load.</> : <>every thread has work.</>}
            </Translated>
          </p>
          <p style={{ margin: '0.4rem 0 0' }}>
            <Translated
              vi={
                <>
                  Khối cuối chứa <b data-testid="last-real">{fmtInt(seg.lastReal)}</b> phần tử thật
                  {seg.pastEnd > 0 ? <>; {fmtInt(seg.pastEnd)} lần nạp của nó nằm quá N.</> : <>: N là bội của đoạn.</>}
                </>
              }
            >
              The last block holds <b data-testid="last-real">{fmtInt(seg.lastReal)}</b> real elements
              {seg.pastEnd > 0 ? <>; {fmtInt(seg.pastEnd)} of its loads lie past N.</> : <>: N is a multiple of the segment.</>}
            </Translated>
          </p>
          {seg.pastEnd > 0 ? (
            <p style={{ margin: '0.5rem 0 0' }} data-testid="verdict">
              {guard ? (
                <span className="chip ok">
                  <Translated vi="có kiểm tra: các luồng rảnh vẫn tham gia và cộng 0">
                    guarded: idle threads still take part and add 0
                  </Translated>
                </span>
              ) : (
                <span className="chip fail">
                  <Translated vi={<>không kiểm tra: khối cuối đọc {fmtInt(seg.pastEnd)} phần tử quá cuối mảng</>}>
                    no guard: the last block reads {fmtInt(seg.pastEnd)} elements past the end
                  </Translated>
                </span>
              )}
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive B: commit order
const LENGTHS = [4096, 65536, 1 << 20];
const COMMIT_BLOCKS = [128, 256, 512, 1024];
const DATA_SEED = 2026;

function CommitExplorer() {
  const [n, setNRaw] = useState(65536);
  const [block, setBlockRaw] = useState(256);
  const [design, setDesignRaw] = useState<Design>('atomic');
  const [runs, setRuns] = useState(1);
  const values = useMemo(() => logits(n, DATA_SEED), [n]);
  const ref = useMemo(() => reference(values), [values]);
  const sums = useMemo(() => blockSums(values, block), [values, block]);
  const run = runOf(sums, design, runs - 1);
  const hist = useMemo(() => history(sums, design, runs), [sums, design, runs]);
  // A new data set, block size or design starts a new history.
  const setN = (v: number) => { setNRaw(v); setRuns(1); };
  const setBlock = (v: number) => { setBlockRaw(v); setRuns(1); };
  const setDesign = (v: Design) => { setDesignRaw(v); setRuns(1); };

  const B = sums.length;
  const shown = design === 'atomic' ? (B <= 8 ? run.order.map((_, i) => i) : [0, 1, 2, 3, 4, -1, B - 2, B - 1]) : [];
  const nearest = f32(ref);
  const counts = hist.distinct.map((h) => ({ h, n: hist.totals.filter((x) => f32hex(x) === h).length, v: hist.totals.find((x) => f32hex(x) === h)! }));

  return (
    <Bench
      title={tr('Blocks commit in any order', 'Các khối commit theo thứ tự bất kỳ')}
      task={tr(
        'Press Run again several times under each design. Then try 4,096 values with 1,024-thread blocks.',
        'Bấm Chạy lại vài lần với mỗi thiết kế. Rồi thử 4,096 giá trị với các khối 1,024 luồng.',
      )}
      foot={
        <Translated
          vi={
            <>
              Các giá trị float32 giống logit (phân phối chuẩn, trung bình −1, độ lệch 2.5; seed {DATA_SEED}). Mỗi tổng khối được
              tính một lần, theo một cây cố định; chỉ thứ tự các lần commit thay đổi giữa các lần chạy. Tham chiếu là tổng float64
              của chính các giá trị float32 đó.
            </>
          }
        >
          <>
            Logit-like float32 values (normal, mean −1, deviation 2.5; seed {DATA_SEED}). Every block sum is computed once, in a
            fixed tree; only the order of the commits changes from run to run. The reference is the float64 sum of the same
            float32 values.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Số giá trị">Values</Translated>
          <select value={n} onChange={(e) => setN(+e.target.value)} aria-label={tr('values', 'số giá trị')}>
            {LENGTHS.map((s) => <option key={s} value={s}>{fmtInt(s)}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Số luồng mỗi khối">Threads per block</Translated>
          <select value={block} onChange={(e) => setBlock(+e.target.value)} aria-label={tr('threads per block', 'số luồng mỗi khối')}>
            {COMMIT_BLOCKS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <span className="seg" role="group" aria-label={tr('Design', 'Thiết kế')}>
          {(['atomic', 'two-pass'] as Design[]).map((d) => (
            <button key={d} aria-pressed={design === d} onClick={() => setDesign(d)}>
              <Translated vi={{ atomic: 'commit atomic (Fig. 10.18)', 'two-pass': 'các tổng bộ phận, rồi một lượt thứ hai cố định' }[d]}>
                {DESIGN_LABEL[d]}
              </Translated>
            </button>
          ))}
        </span>
        <button className="primary" onClick={() => setRuns(runs + 1)}>
          <Translated vi="Chạy lại">Run again</Translated>
        </button>
        <button onClick={() => setRuns(1)} disabled={runs === 1}>
          <Translated vi="Xóa lịch sử">Clear history</Translated>
        </button>
      </div>

      <p style={{ margin: '0 0 0.7rem' }}>
        <Translated
          vi={
            <>
              {fmtInt(B)} khối {block} luồng, mỗi khối {fmtInt(2 * block)} giá trị.{' '}
              {design === 'atomic'
                ? <>Lần chạy {runs}: các luồng đứng đầu tới tổng theo thứ tự này.</>
                : <>Lần chạy {runs}: mỗi luồng đứng đầu ghi tổng của nó vào <code>partial[blockIdx.x]</code>; sau đó một khối rút gọn {fmtInt(B)} tổng bộ phận theo một cây cố định.</>}
            </>
          }
        >
          {fmtInt(B)} blocks of {block} threads, {fmtInt(2 * block)} values each.{' '}
          {design === 'atomic'
            ? <>Run {runs}: the leaders reach the total in this order.</>
            : <>Run {runs}: each leader writes its sum to <code>partial[blockIdx.x]</code>; one block then reduces the {fmtInt(B)} partials in a fixed tree.</>}
        </Translated>
      </p>

      {design === 'atomic' ? (
        <table className="data" style={{ margin: '0 0 0.8rem', background: 'var(--page)' }} aria-label={tr('Commits of this run', 'Các lần commit của lần chạy này')}>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}><Translated vi="lần commit">commit</Translated></th>
              <th><Translated vi="khối">block</Translated></th>
              <th><Translated vi="tổng khối">block sum</Translated></th>
              <th><Translated vi="tổng sau phép cộng">total after the add</Translated></th>
            </tr>
          </thead>
          <tbody>
            {shown.map((i, j) =>
              i < 0 ? (
                <tr key={'gap' + j}>
                  <td colSpan={4} className="muted" style={{ paddingLeft: '0.6rem' }}>
                    <Translated vi={<>… thêm {fmtInt(B - 7)} lần commit …</>}>… {fmtInt(B - 7)} more commits …</Translated>
                  </td>
                </tr>
              ) : (
                <tr key={i} data-commit={i}>
                  <td style={{ paddingLeft: '0.6rem' }}>{i + 1}</td>
                  <td>{run.order[i]}</td>
                  <td className="mono t-reg">{f32short(sums[run.order[i]])}</td>
                  <td className="mono t-global">{f32short(run.running[i])}</td>
                </tr>
              ),
            )}
          </tbody>
        </table>
      ) : null}

      <div className="cols">
        <div className="panel" data-testid="final">
          <p style={{ margin: 0 }}>
            <Translated
              vi={
                <>
                  Tổng của lần chạy {runs}: <b className="mono" data-testid="total">{f32short(run.total)}</b>{' '}
                  <span className="mono" data-testid="total-hex">{f32hex(run.total)}</span>
                </>
              }
            >
              Total of run {runs}: <b className="mono" data-testid="total">{f32short(run.total)}</b>{' '}
              <span className="mono" data-testid="total-hex">{f32hex(run.total)}</span>
            </Translated>
          </p>
          <p style={{ margin: '0.4rem 0 0' }}>
            <Translated
              vi={
                <>
                  tham chiếu float64: <span className="mono">{f64short(ref)}</span>; float32 gần nhất là{' '}
                  <span className="mono">{f32hex(nearest)}</span>, cách tổng này {ulpDistance(run.total, nearest)} bước float32.
                </>
              }
            >
              float64 reference: <span className="mono">{f64short(ref)}</span>; the nearest float32 is{' '}
              <span className="mono">{f32hex(nearest)}</span>, {ulpDistance(run.total, nearest)} float32 step
              {ulpDistance(run.total, nearest) === 1 ? '' : 's'} from this total.
            </Translated>
          </p>
        </div>
        <div className="panel" data-testid="history" data-runs={runs} data-distinct={hist.distinct.length}>
          <p style={{ margin: 0 }}>
            <Translated
              vi={
                <>
                  <b>{runs}</b> lần chạy tới giờ, <b>{hist.distinct.length}</b> tổng khác nhau:
                </>
              }
            >
              <b>{runs}</b> run{runs === 1 ? '' : 's'} so far, <b>{hist.distinct.length}</b> distinct total{hist.distinct.length === 1 ? '' : 's'}:
            </Translated>
          </p>
          <ul style={{ margin: '0.4rem 0 0', paddingLeft: '1.1rem' }}>
            {counts.map((c) => (
              <li key={c.h} className="mono">
                {c.h} ({f32short(c.v)}) × {c.n}
              </li>
            ))}
          </ul>
        </div>
      </div>
      {runs >= 4 ? (
        <p style={{ margin: '0.8rem 0 0' }} data-testid="verdict">
          {hist.distinct.length > 1 ? (
            <span className="chip fail">
              <Translated vi={<>cùng dữ liệu cho {hist.distinct.length} tổng khác nhau</>}>
                the same data gave {hist.distinct.length} different totals
              </Translated>
            </span>
          ) : design === 'two-pass' ? (
            <span className="chip ok">
              <Translated vi="cùng các bit ở mọi lần chạy">the same bits every run</Translated>
            </span>
          ) : (
            <span className="chip ok">
              <Translated vi="tới giờ cùng các bit ở mọi lần chạy">so far the same bits every run</Translated>
            </span>
          )}{' '}
          {design === 'atomic' && B <= 2
            ? tr(
                'With two blocks the order cannot matter: a + b and b + a round the same way. Float addition is commutative; it is only not associative.',
                'Với hai khối, thứ tự không thể ảnh hưởng: a + b và b + a làm tròn như nhau. Phép cộng float có tính giao hoán; nó chỉ không có tính kết hợp.',
              )
            : null}
        </p>
      ) : null}
    </Bench>
  );
}

function MeasuredDistinct({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const found = Object.keys(reports).flatMap((b) => {
    const row = benchRows(reports[b], LAB).find((r) => r.name === 'determinism');
    const distinct = num(row?.distinct);
    return row && distinct != null ? [{ b, device: String(reports[b].device.name ?? b), distinct, runs: num(row.runs), n: num(row.n) }] : [];
  });
  if (!found.length)
    return (
      <Translated
        vi={
          <p>
            Chưa có phép đo nào trên máy này. Hãy chạy <code>python3 labs/pmpp.py run 10.1 --solution --bench</code> trên một máy
            có GPU, rồi tải lại trang: ô này sẽ cho thấy thiết bị của bạn đã tạo ra bao nhiêu tổng khác nhau.
          </p>
        }
      >
        <p>
          No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 10.1 --solution --bench</code> on a machine
          with a GPU, then reload the page: this box will show how many different totals your device produced.
        </p>
      </Translated>
    );
  return (
    <>
      {found.map((f) => (
        <Translated
          key={f.b}
          vi={
            <p data-backend={f.b} data-distinct={f.distinct}>
              Trên <b>{f.device}</b> ({f.b}): <b>{f.distinct}</b> kết quả khác nhau
              {f.runs ? <> trong {f.runs} lần chạy</> : null}
              {f.n ? <> trên {fmtInt(f.n)} float</> : null}.{' '}
              {f.b === 'emu'
                ? 'Trình giả lập chạy từng luồng một theo một thứ tự cố định, nên các khối của nó luôn commit theo cùng thứ tự: nó chỉ có thể cho ra 1.'
                : f.b === 'race'
                  ? 'Backend race chạy các khối như các luồng của hệ điều hành, nên thứ tự của nó thay đổi theo bộ lập lịch, không theo GPU.'
                  : f.distinct > 1
                    ? predicted <= 1
                      ? 'Bạn đã dự đoán kernel atomic lặp lại được. Không phải vậy: các khối đã kết thúc theo những thứ tự khác nhau.'
                      : 'Lần commit atomic không lặp lại được trên thiết bị này.'
                    : 'Mọi lần chạy đều cho cùng các bit ở đây. Đó không phải một lời hứa: không có gì trong mô hình lập trình cố định thứ tự.'}
            </p>
          }
        >
          <p key={f.b} data-backend={f.b} data-distinct={f.distinct}>
            On <b>{f.device}</b> ({f.b}): <b>{f.distinct}</b> distinct result{f.distinct === 1 ? '' : 's'}
            {f.runs ? <> in {f.runs} runs</> : null}
            {f.n ? <> on {fmtInt(f.n)} floats</> : null}.{' '}
            {f.b === 'emu'
              ? 'The emulator runs threads one at a time in a fixed order, so its blocks always commit in the same order: it can only ever show 1.'
              : f.b === 'race'
                ? 'The race backend runs blocks as operating-system threads, so its order changes with the scheduler, not with a GPU.'
                : f.distinct > 1
                  ? predicted <= 1
                    ? 'You expected the atomic kernel to be repeatable. It is not: the blocks finished in different orders.'
                    : 'The atomic commit is not repeatable on this device.'
                  : 'Every run gave the same bits here. That is not a promise: nothing in the programming model fixes the order.'}
          </p>
        </Translated>
      ))}
      <p className="muted">
        <Translated vi="Đây là các phép đo từ máy của bạn. Một thiết bị, kích thước hay mức tải khác có thể cho con số khác.">
          These are measurements from your machine. Another device, size or load can give a different count.
        </Translated>
      </p>
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS109: Record<string, Question> = {
  'ch10/10.9/q-lastblock': {
    kind: 'numeric',
    id: 'ch10/10.9/q-lastblock',
    prompt: (
      <Translated
        vi={
          <>
            N = 1,000,000 với các khối 1024 luồng, mỗi khối phủ 2048 phần tử. Khối cuối chứa bao nhiêu phần tử thật?
          </>
        }
      >
        <>
          N = 1,000,000 with 1024-thread blocks, each block covering 2048 elements. How many real elements does the last block
          hold?
        </>
      </Translated>
    ),
    answer: 576,
    explain: (
      <Translated
        vi={
          <p>
            ceil(1,000,000 / 2048) = 489 khối. 488 khối đầu phủ 488 × 2048 = 999,424 phần tử, nên khối cuối bắt đầu ở 999,424
            và chứa 1,000,000 − 999,424 = 576. Các luồng 0 đến 575 mỗi luồng nạp một phần tử thật (lần nạp đầu của chúng); các
            luồng 576 đến 1023 không nạp phần tử thật nào và, nhờ phép kiểm tra, cộng 0. Cả 1024 luồng vẫn tham gia các lệnh
            shuffle và rào chắn.
          </p>
        }
      >
        <p>
          ceil(1,000,000 / 2048) = 489 blocks. The first 488 cover 488 × 2048 = 999,424 elements, so the last block starts at
          999,424 and holds 1,000,000 − 999,424 = 576. Threads 0 to 575 load one real element each (their first load); threads
          576 to 1023 load nothing real and, with the guard, add 0. All 1024 still take part in the shuffles and the barrier.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.9, pp. 244–245; exercise 5, p. 250',
    vi: { source: 'ch. 10 §10.9, pp. 244–245; bài tập 5, p. 250' },
  },
  'ch10/10.9/q-twice': {
    kind: 'choice',
    id: 'ch10/10.9/q-twice',
    prompt: (
      <Translated
        vi={
          <>
            Host khởi chạy kernel của Fig. 10.18 hai lần liên tiếp trên cùng đầu vào và cùng buffer <code>output</code>, không
            đụng tới <code>*output</code> ở giữa. Sau lần khởi chạy thứ hai, <code>*output</code> chứa gì?
          </>
        }
      >
        <>
          The host launches the kernel of Fig. 10.18 twice in a row on the same input and the same <code>output</code> buffer,
          without touching <code>*output</code> in between. What does <code>*output</code> hold after the second launch?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Tổng: lần khởi chạy thứ hai ghi đè lần thứ nhất.">The sum: the second launch overwrites the first.</Translated>
        ),
        why: (
          <Translated vi="Không có gì trong kernel ghi tổng; mỗi khối chỉ cộng vào nó.">
            Nothing in the kernel writes the total; every block only adds to it.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Gấp đôi tổng: mỗi lần khởi chạy cộng thêm vào giá trị đang có.">
            Twice the sum: each launch adds onto what is there.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không xác định: hai lần khởi chạy tranh chấp trên cùng buffer.">
            Undefined: two launches race on the same buffer.
          </Translated>
        ),
        why: (
          <Translated vi="Các lần khởi chạy trên một stream chạy lần lượt; lần thứ hai bắt đầu từ thứ lần thứ nhất để lại.">
            Launches on one stream run one after the other; the second starts from what the first left.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Tổng, vì fetch_add với thứ tự relaxed bắt đầu từ 0.">
            The sum, because fetch_add with relaxed order starts from 0.
          </Translated>
        ),
        why: (
          <Translated vi="Một phép cộng atomic bắt đầu từ bất cứ giá trị nào đang có trong bộ nhớ.">
            An atomic add starts from whatever is in memory.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mỗi luồng đứng đầu thực hiện <code>total += blockSum</code>. Lần khởi chạy đầu để lại tổng trong{' '}
            <code>*output</code>; lần thứ hai cộng mọi tổng khối vào đó thêm lần nữa. Host phải đặt <code>*output</code> về phần
            tử đơn vị, 0, trước mỗi lần khởi chạy. Sách không cho thấy mã host của kernel này, nên không nói điều đó; hàm host của
            lab 10.1 làm việc này.
          </p>
        }
      >
        <p>
          Each leader does <code>total += blockSum</code>. The first launch leaves the sum in <code>*output</code>; the second
          adds every block sum onto it again. The host must set <code>*output</code> to the identity, 0, before every launch.
          The book does not show host code for this kernel, so it does not say so; lab 10.1's host function does it.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.9, p. 245 (Fig. 10.18); the zeroing is our inference, recorded in notes/ch10.md',
    vi: { source: 'ch. 10 §10.9, p. 245 (Fig. 10.18); việc đặt về 0 là suy luận của chúng tôi, ghi trong notes/ch10.md' },
  },
  'ch10/10.9/q-max-commit': {
    kind: 'choice',
    id: 'ch10/10.9/q-max-commit',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 4 biến phép tổng thành max. Bạn cần một max float xuyên khối, dựng được với mọi bộ công cụ CUDA gần đây và
            cũng chạy trên Metal. Thiết kế nào?
          </>
        }
      >
        <>
          Exercise 4 turns the sum into a max. You need a float max across blocks that builds with any recent CUDA toolkit
          and also works on Metal. Which design?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi={<><code>atomicMax(output, partialMax)</code> trên một <code>float*</code>.</>}>
            <><code>atomicMax(output, partialMax)</code> on a <code>float*</code>.</>
          </Translated>
        ),
        why: (
          <Translated vi="Overload đó chỉ có trên HIP. atomicMax legacy của CUDA nhận các kiểu số nguyên.">
            That overload exists on HIP only. CUDA's legacy atomicMax takes integer types.
          </Translated>
        ),
      },
      {
        label: (
          <Translated
            vi={
              <>
                <code>cuda::atomic_ref&lt;float&gt;::fetch_max</code> trên CUDA, và <code>atomic_fetch_max_explicit</code> trên một{' '}
                <code>atomic_float</code> trên Metal.
              </>
            }
          >
            <><code>cuda::atomic_ref&lt;float&gt;::fetch_max</code> on CUDA, and <code>atomic_fetch_max_explicit</code> on an <code>atomic_float</code> on Metal.</>
          </Translated>
        ),
        why: (
          <Translated vi="CCCL hiện hành chấp nhận float cho fetch_max, nhưng bộ công cụ đầu tiên làm được điều đó không được ghi trong tài liệu; Metal chỉ liệt kê atomic max cho các kiểu số nguyên.">
            Current CCCL accepts float for fetch_max, but the first toolkit that does is not documented; Metal lists atomic max for integer types only.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Mỗi khối ghi max của nó vào một mảng; một lượt thứ hai, một khối duy nhất, rút gọn mảng đó.">
            Each block writes its maximum to an array; a second, single-block pass reduces it.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi={<>Dùng <code>fetch_add</code> trên các giá trị đã đổi dấu, rồi đổi dấu tổng ở cuối.</>}>
            <>Use <code>fetch_add</code> on the negated values, then negate the total at the end.</>
          </Translated>
        ),
        why: (
          <Translated vi="Tổng của các giá trị đã đổi dấu là âm của tổng, không phải một giá trị max.">
            A sum of negated values is minus the sum, not a maximum.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Thiết kế thứ hai của sách không cần atomic nào, nên nó chạy ở bất cứ đâu một kernel có thể ghi một mảng. Các cách
            dùng atomic đều hỏng ở đâu đó: không có <code>atomicMax</code> float legacy trên CUDA, không rõ bộ công cụ đầu tiên
            hỗ trợ <code>fetch_max</code> trên float, không có atomic max float trên Metal. Một vòng lặp compare-and-swap cũng
            được (trên Metal, trên <code>atomic_float</code>), với cái giá là phải thử lại khi có tranh chấp.
          </p>
        }
      >
        <p>
          The book's second design needs no atomic at all, so it works wherever a kernel can write an array. The atomic
          routes each fail somewhere: no legacy float <code>atomicMax</code> on CUDA, an unknown first toolkit for{' '}
          <code>fetch_max</code> on float, no float atomic max on Metal. A compare-and-swap loop also works (on Metal over{' '}
          <code>atomic_float</code>), at the price of retries under contention.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.9, pp. 245–246, exercise 4 (p. 250); notes/today/ch10-facts.md A7, B3, C2',
    vi: { source: 'ch. 10 §10.9, pp. 245–246, bài tập 4 (p. 250); notes/today/ch10-facts.md A7, B3, C2' },
  },
  'ch10/10.9/q-reproducible': {
    kind: 'choice',
    id: 'ch10/10.9/q-reproducible',
    prompt: (
      <Translated
        vi={
          <>
            Một bài kiểm thử hồi quy so sánh tổng của một tensor logits từng bit một giữa các lần chạy trên cùng GPU và cùng bản
            dựng. Thiết kế nhiều khối nào vượt qua nó?
          </>
        }
      >
        <>
          A regression test compares the sum of a logits tensor bit for bit across runs on the same GPU and build. Which
          multi-block design passes it?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Commit atomic với memory_order_relaxed.">The atomic commit with memory_order_relaxed.</Translated>,
        why: (
          <Translated vi="Thứ tự commit theo thứ tự các khối kết thúc, vốn có thể thay đổi giữa các lần chạy.">
            The commit order follows the order blocks finish, which can change between runs.
          </Translated>
        ),
      },
      {
        label: (
          <Translated
            vi={<>Commit atomic với <code>memory_order_seq_cst</code>, đặt mọi phép cộng vào một thứ tự toàn phần.</>}
          >
            <>The atomic commit with <code>memory_order_seq_cst</code>, which puts all the adds in one total order.</>
          </Translated>
        ),
        why: (
          <Translated vi="Một memory order ràng buộc cách các thao tác bộ nhớ khác được nhìn thấy quanh atomic. Nó không chọn khối nào cộng trước, nên thứ tự toàn phần có thể khác nhau giữa các lần chạy.">
            A memory order constrains how other memory operations are seen around the atomic. It does not choose which block adds first, so the total order can differ from run to run.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các tổng bộ phận theo khối, rồi một lượt thứ hai theo thứ tự cố định.">
            Per-block partials, then a second pass in a fixed order.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Cách nào cũng được, miễn là mọi khối rút gọn đoạn của nó bằng cùng một cây ở mọi lần chạy.">
            Either, as long as every block reduces its segment with the same tree on every run.
          </Translated>
        ),
        why: (
          <Translated vi="Các cây trong khối vốn đã cố định. Thứ thay đổi là thứ tự mà các tổng khối gặp nhau, và việc làm tròn float khiến thứ tự đó lộ ra.">
            The block trees are fixed already. What varies is the order in which the block sums meet, and float rounding makes that order visible.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mỗi tổng khối đều như nhau ở mọi lần chạy: bên trong một khối, thứ tự được mã cố định. Thứ duy nhất thay đổi là thứ
            tự mà các tổng khối gặp nhau. Ghi chúng vào một mảng đánh chỉ số theo khối và rút gọn mảng đó theo một cây cố định,
            thì không có gì phụ thuộc vào thời gian. Đó là cách chế độ mặc định của <code>cub::DeviceReduce</code> được tài liệu
            mô tả.
          </p>
        }
      >
        <p>
          Every block sum is the same on every run: inside a block the order is fixed by the code. What changes is only the
          order in which block sums meet. Write them to an array indexed by block and reduce that array in a fixed tree,
          and nothing depends on timing. That is how <code>cub::DeviceReduce</code>'s default mode is documented to work.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.9, pp. 245–246; notes/today/ch10-facts.md A13', // i18n-ignore: citation only
  },
  'ch10/10.9/q-argmax-pack': {
    kind: 'choice',
    id: 'ch10/10.9/q-argmax-pack',
    prompt: (
      <Translated
        vi={
          <>
            Để commit một argmax xuyên khối bằng một lệnh <code>atomicMax(unsigned long long*, …)</code> 64-bit, bạn đóng gói mỗi
            ứng viên thành một khóa: số float ở 32 bit cao, chỉ số ở 32 bit thấp. Cách đóng gói phải bảo toàn điều gì để khóa
            lớn nhất là giá trị lớn nhất, và khi hòa thì chỉ số nhỏ nhất thắng?
          </>
        }
      >
        <>
          To commit an argmax across blocks with one 64-bit <code>atomicMax(unsigned long long*, …)</code>, you pack each
          candidate into a key: the float in the high 32 bits, the index in the low 32. What must the packing preserve so that
          the largest key is the largest value, with ties going to the lowest index?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Các bit float thô ở nửa cao và chỉ số ở nửa thấp.">
            The raw float bits in the high half and the index in the low half.
          </Translated>
        ),
        why: (
          <Translated vi="Đọc như số không dấu, một số float âm có bit dấu bật, nên nó so sánh lớn hơn mọi số dương, và các giá trị càng âm lại so sánh càng lớn.">
            Read as unsigned, a negative float has its sign bit set, so it compares above every positive one, and more negative values compare larger.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các bit float được biến đổi để giữ thứ tự khi xem như số không dấu ở nửa cao; 0xFFFFFFFF − index ở nửa thấp.">
            Float bits made order-preserving as unsigned in the high half; 0xFFFFFFFF − index in the low half.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chỉ số ở nửa cao, để các trường hợp hòa được giải quyết trước, và các bit float ở nửa thấp.">
            The index in the high half, so that ties are settled first, and the float bits in the low half.
          </Translated>
        ),
        why: (
          <Translated vi="Khi đó chỉ số lớn nhất thắng, bất kể giá trị.">Then the largest index wins, whatever the values.</Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một phép biến đổi float giữ thứ tự ở nửa cao và chỉ số nguyên gốc ở nửa thấp.">
            An order-preserving float transform in the high half and the plain index in the low half.
          </Translated>
        ),
        why: (
          <Translated vi="Giá trị thì đúng; hòa thì sai: với các giá trị bằng nhau, khóa lớn hơn có chỉ số lớn hơn.">
            Values right; ties wrong: on equal values the larger key has the larger index.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Atomic so sánh các khóa như số nguyên không dấu, nên nửa cao phải có thứ tự giống các số float, và nửa thấp phải phá
            hòa theo cách bạn muốn. Phép biến đổi lật bit dấu của một số float không âm và lật mọi bit của một số âm; khi đó
            float lớn hơn cho giá trị không dấu lớn hơn. Lưu <code>0xFFFFFFFF − index</code> khiến chỉ số nhỏ hơn thắng. Cách
            đóng gói này là kỹ thuật của chúng tôi, không phải công thức của hãng nào. Trên Metal, một atomic max 64-bit cần họ
            Apple9 (Apple8 trên macOS) và không trả về gì; ở những nơi khác, cách khả chuyển một lần nữa là hai lượt.
          </p>
        }
      >
        <p>
          The atomic compares keys as unsigned integers, so the high half must order like the floats, and the low half must
          break ties the way you want. The transform flips the sign bit of a non-negative float and every bit of a
          negative one; then larger floats give larger unsigned values. Storing{' '}
          <code>0xFFFFFFFF − index</code> makes the lower index win. The packing is our technique, not a vendor recipe. On
          Metal a 64-bit atomic max needs the Apple9 family (Apple8 on macOS) and returns nothing; elsewhere the portable route
          is again two passes.
        </p>
      </Translated>
    ),
    source: 'ours (not in the book); availability from notes/today/ch10-facts.md A7, C2 and infra-facts C1',
    vi: { source: 'của khóa học này (không có trong sách); tính khả dụng từ notes/today/ch10-facts.md A7, C2 và infra-facts C1' },
  },
};

export function S109() {
  return (
    <>
      <Translated
        vi={
          <p>
            Mọi kernel cho tới giờ đều là một khối. Lý do là rào chắn: <code>__syncthreads()</code> điều phối các luồng của một
            khối và không luồng nào khác, và một khối chứa tối đa 1024 luồng, nên một khối rút gọn tối đa 2048 phần tử.
          </p>
        }
      >
        <p>
          Every kernel so far has been one block. The reason is the barrier: <code>__syncthreads()</code> coordinates the
          threads of one block and no others, and a block holds at most 1024 threads, so one block reduces at most 2048
          elements.
        </p>
      </Translated>
      <Source section="10.9" pages="244" />
      <Translated
        vi={
          <p>
            Cũng không có rào chắn tốt nào xuyên khối. Vì vậy các khối phải làm việc độc lập: mỗi khối rút gọn phần đầu vào của
            riêng mình, không khối nào phải chờ khối khác, và các phần được kết hợp ở cuối. Điều đó đặt ra một câu hỏi mà các
            kernel một khối chưa bao giờ phải trả lời: các phần gặp nhau theo thứ tự nào?
          </p>
        }
      >
        <p>
          Nor is there a good barrier across blocks. So blocks have to work independently: each reduces its own part of the
          input, no block ever waits for another, and the parts are combined at the end. That raises a question the
          one-block kernels never had to answer: in what order do the parts meet?
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch10-facts A8, C2)">{/* i18n-ignore: date and citation only */}
        <Translated
          vi={
            <p>
              CUDA và HIP có <code>sync()</code> trên cả lưới (grid), nhưng chỉ dưới một lần khởi chạy cooperative mà toàn bộ
              lưới vừa với thiết bị. Metal không có.
            </p>
          }
        >
          <p>
            CUDA and HIP have a grid-wide <code>sync()</code>, but only under a cooperative launch whose whole grid fits on
            the device. Metal has none.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Giới hạn đó quan trọng với các phép rút gọn mà bạn quan tâm. Softmax trên hàng đầu ra của một mô hình cộng mỗi mục từ
            vựng một số hạng, và các bộ từ vựng trải từ 50,257 mục (GPT-2) tới 201,088 (gpt-oss-20b). Một hàng dài như vậy cần
            hoặc nhiều khối, là mục này, hoặc một khối trong đó mỗi luồng làm nhiều việc tuần tự, là mục 10.10.
          </p>
        }
      >
        <p>
          That limit matters for the reductions you care about. The softmax over a model's output row sums one term per
          vocabulary entry, and vocabularies run from 50,257 entries (GPT-2) to 201,088 (gpt-oss-20b). A row that long needs
          either many blocks, which is this section, or one block in which every thread does a lot of serial work, which is
          section 10.10.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Cách đặt vấn đề theo softmax là của khóa học này. Kích thước bộ từ vựng từ cấu hình công khai của các mô hình, đọc
              vào 2026-11-20 (notes/today/ch09-facts.md, D7): {VOCABS.map((v) => `${v.name} ${fmtInt(v.size)}`).join(', ')}.
            </p>
          }
        >
          <p>
            The softmax framing is ours. Vocabulary sizes from public model configs, read on 2026-11-20
            (notes/today/ch09-facts.md, D7): {VOCABS.map((v) => `${v.name} ${fmtInt(v.size)}`).join(', ')}.
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Mỗi khối một đoạn">One segment per block</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Cắt đầu vào thành các đoạn <code>2·blockDim.x</code> phần tử. Khối <code>b</code> lấy đoạn bắt đầu ở{' '}
            <code>2·blockDim.x·b</code>: với 1024 luồng, khối 0 bắt đầu ở 0, khối 1 ở 2048, khối 2 ở 4096. Bên trong đoạn của
            mình, một khối chạy bất kỳ phép rút gọn một khối nào bạn đã có; sách dùng lại phép rút gọn warp hai giai đoạn của
            mục 10.8. Điều đó để lại một con số cho mỗi khối. Luồng đứng đầu của khối, luồng 0, cộng nó vào tổng duy nhất bằng
            một atomic:
          </p>
        }
      >
        <p>
          Cut the input into segments of <code>2·blockDim.x</code> elements. Block <code>b</code> takes the segment that starts
          at <code>2·blockDim.x·b</code>: with 1024 threads, block 0 starts at 0, block 1 at 2048, block 2 at 4096. Inside its
          segment, a block runs any of the single-block reductions you already have; the book reuses the two-stage warp
          reduction of section 10.8. That leaves one number per block. Its leader, thread 0, adds it into the single total
          with an atomic:
        </p>
      </Translated>
      <Source section="10.9" pages="244–245" figs={tr('Figs. 10.17, 10.18', 'Hình 10.17, 10.18')} />
      <Translated
        vi={
          <p>
            <code>cuda::atomic_ref&lt;float, cuda::thread_scope_device&gt; output_ref(*output);</code> rồi{' '}
            <code>output_ref.fetch_add(partialSum, cuda::memory_order_relaxed);</code>. Đây là tham chiếu atomic từ chương 9, ở
            phạm vi device vì mọi khối cộng vào cùng một vị trí, với thứ tự relaxed vì không có gì khác được công bố qua phép cộng
            này. Dưới HIP, lab 10.1 viết <code>atomicAdd(output, partialSum)</code> legacy thay vào đó, cũng relaxed và có phạm
            vi toàn thiết bị: ở đó <code>&lt;cuda/atomic&gt;</code> đến từ libhipcxx, thứ mà bản dựng HIP của khóa học chỉ liên
            kết ở nơi ROCm cung cấp nó, nên lab không dựa vào nó.
          </p>
        }
      >
        <p>
          <code>cuda::atomic_ref&lt;float, cuda::thread_scope_device&gt; output_ref(*output);</code> then{' '}
          <code>output_ref.fetch_add(partialSum, cuda::memory_order_relaxed);</code>. This is the atomic reference from chapter
          9, at device scope because every block adds to the same location, with relaxed order because nothing else is
          published through this add. Under HIP, lab 10.1 writes the legacy <code>atomicAdd(output, partialSum)</code>{' '}
          instead, which is also relaxed and device-wide: there <code>&lt;cuda/atomic&gt;</code> comes from libhipcxx, which
          the course's HIP build links only where ROCm provides it, so the lab does not rely on it.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch10-facts A6; infra-facts A1, B1, B2)">{/* i18n-ignore: date and citation only */}
        <Translated
          vi={
            <p>
              <code>output_ref(*output)</code> của sách là đúng: hàm khởi tạo nhận một tham chiếu. Ví dụ riêng của hướng dẫn CUDA
              hiện hành truyền con trỏ. Các phép cộng atomic float vào bộ nhớ toàn cục đẩy các số denormal về không.
            </p>
          }
        >
          <p>
            The book's <code>output_ref(*output)</code> is right: the constructor takes a reference. The current CUDA guide's own
            example passes the pointer. Float atomic adds to global memory flush denormals to zero.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Có hai điều mà đoạn mã để lại cho bạn. Thứ nhất, <code>*output</code> phải bằng 0 trước khi khởi chạy: kernel chỉ cộng
            vào nó, và sách không cho thấy mã host. Thứ hai, thân hàm như khi in không bao giờ dùng tham số <code>N</code>; nó giả
            định N là bội của <code>2·blockDim.x</code>. Bài tập 5 gỡ bỏ giả định đó: khởi chạy{' '}
            <code>ceil(N / (2·blockDim.x))</code> khối và kiểm tra biên mọi lần nạp, để một chỉ số bằng hoặc vượt N đóng góp 0.
            Kernel của lab 10.1 được viết theo cách đó.
          </p>
        }
      >
        <p>
          Two things the listing leaves to you. First, <code>*output</code> must hold 0 before the launch: the kernel only ever
          adds to it, and the book shows no host code. Second, the printed body never uses its parameter <code>N</code>; it
          assumes N is a multiple of <code>2·blockDim.x</code>. Exercise 5 removes that assumption: launch{' '}
          <code>ceil(N / (2·blockDim.x))</code> blocks and guard every load, so that an index at or past N contributes 0. Lab
          10.1's kernel is written that way.
        </p>
      </Translated>
      <Source section="10.9" pages="245, 250" figs={tr('exercise 5', 'bài tập 5')} />
      <CommitListing />

      <Translated
        vi={
          <p>
            Trong bộ tính, mặc định là trường hợp của câu hỏi bên dưới. Chuyển tới khối cuối và chạy từng bước các lần nạp của
            một luồng, rồi bỏ chọn phép kiểm tra.
          </p>
        }
      >
        <p>
          In the calculator, the default is the case from the question below. Move to the last block and step through a
          thread's loads, then untick the guard.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={<p>Chỉ có tính toán chỉ số: không giá trị, không thời gian. Hình vẽ theo tỉ lệ, nên với nhiều đoạn thì biên của chúng hòa vào nhau.</p>}
        >
          <p>Index arithmetic only: no values, no timing. The drawing is to scale, so with many segments their borders merge.</p>
        </Translated>
      </Simplification>
      <SegmentCalculator />
      <Ask q={questionsS109['ch10/10.9/q-lastblock']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Hãy để ý các luồng ngoài phạm vi làm gì. Chúng không return sớm. Chúng không nạp gì, giữ 0, và vẫn tham gia các lệnh
            shuffle của warp và rào chắn, vì những thứ đó đòi mọi lane và mọi luồng của khối phải tới.
          </p>
        }
      >
        <p>
          Notice what the out-of-range threads do. They do not return early. They load nothing, hold 0, and still join the
          warp shuffles and the barrier, because those require every lane and every thread of the block to arrive.
        </p>
      </Translated>
      <Ask q={questionsS109['ch10/10.9/q-twice']} />

      <h2>
        <Translated vi="Các khối kết thúc theo thứ tự bất kỳ">Blocks finish in any order</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một phép cộng atomic đảm bảo hai luồng đứng đầu không bao giờ làm mất cập nhật của nhau. Nó không quyết định luồng nào
            đi trước. Các khối kết thúc theo bất cứ thứ tự nào phần cứng lập lịch, nên các tổng khối gặp tổng chung theo một thứ
            tự không ai kiểm soát. Điều đó chỉ đúng nếu toán tử có tính kết hợp và giao hoán.
          </p>
        }
      >
        <p>
          An atomic add guarantees that two leaders never lose each other's update. It does not decide which leader goes
          first. Blocks finish in whatever order the hardware schedules them, so the block sums meet the total in an order that
          nobody controls. That is only correct if the operator is associative and commutative.
        </p>
      </Translated>
      <Source section="10.9" pages="245" />
      <Translated
        vi={
          <p>
            Phép cộng số nguyên thỏa điều đó. Phép cộng float có tính giao hoán nhưng, sau khi làm tròn, không có tính kết hợp
            (mục 10.2): cộng cùng các tổng khối theo thứ tự khác có thể làm tròn khác đi. Kết quả vẫn là một tổng tốt; chỉ là
            không phải cùng các bit mỗi lần. Hãy xem nó xảy ra, rồi tự thử.
          </p>
        }
      >
        <p>
          Integer addition qualifies. Float addition is commutative but, after rounding, not associative (section 10.2): adding
          the same block sums in another order can round differently. The result is still a good sum; it is just not the same
          bits every time. Watch it happen, then try it yourself.
        </p>
      </Translated>
      <CommitOrderScene />
      <Translated
        vi={
          <p>
            Giờ với nhiều dữ liệu hơn. Mỗi lần bấm Chạy lại commit cùng các tổng khối theo một thứ tự mới. Đếm số tổng khác nhau
            dưới mỗi thiết kế, rồi tìm một kích thước mà tại đó thiết kế atomic không thể bất đồng với chính nó, và giải thích vì
            sao.
          </p>
        }
      >
        <p>
          Now on more data. Each press of Run again commits the same block sums in a new order. Count the distinct totals under
          each design, then find a size at which the atomic design cannot disagree with itself, and say why.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một thứ tự commit ở đây là một phép xáo trộn có seed, không phải mô hình thời gian: nó không nói gì về những thứ tự
              nào GPU tạo ra hay thường xuyên ra sao. Mỗi tổng khối được tính một lần, theo một cây cố định, như trên GPU.
            </p>
          }
        >
          <p>
            A commit order here is a seeded shuffle, not a model of timing: it says nothing about which orders a GPU produces or
            how often. Each block sum is computed once, in a fixed tree, as on a GPU.
          </p>
        </Translated>
      </Simplification>
      <CommitExplorer />
      <Predict
        id="ch10/10.9/p-distinct"
        prompt={
          <Translated
            vi={
              <>
                Lab 10.1 chạy kernel nhiều khối dùng atomic của nó 20 lần trên cùng 2<sup>24</sup> float (ít hơn nơi thiếu bộ
                nhớ). Bạn dự đoán bao nhiêu kết quả khác nhau, tính theo mẫu bit, trên GPU <em>của bạn</em>?
              </>
            }
          >
            <>
              Lab 10.1 runs its atomic multi-block kernel 20 times on the same 2<sup>24</sup> floats (fewer where memory is
              short). How many distinct results, as bit patterns, do you expect on <em>your</em> GPU?
            </>
          </Translated>
        }
        unit={tr('distinct results', 'kết quả khác nhau')}
      >
        {(v) => <MeasuredDistinct predicted={v} />}
      </Predict>

      <h2>
        <Translated vi="Không dùng atomic: một lượt thứ hai">Without atomics: a second pass</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Thiết kế còn lại không commit gì. Mỗi luồng đứng đầu ghi tổng khối của mình vào một mảng, tại chỉ số{' '}
            <code>blockIdx.x</code>. Sau đó một kernel thứ hai với một khối duy nhất rút gọn mảng đó, hoặc host sao chép nó về và
            hoàn tất trên CPU. Sách đưa ra cách này cho các hệ thống thiếu atomic hoặc atomic quá đắt, và lưu ý lợi ích phụ: bạn
            chọn thứ tự của các phép cộng cuối cùng.
          </p>
        }
      >
        <p>
          The other design commits nothing. Each leader writes its block sum to an array, at index <code>blockIdx.x</code>.
          Then a second kernel with a single block reduces that array, or the host copies it back and finishes on the CPU. The
          book offers this for systems where atomics are missing or too expensive, and notes the side benefit: you choose the
          order of the final additions.
        </p>
      </Translated>
      <Source section="10.9" pages="245–246" />
      <Translated
        vi={
          <p>
            Lợi ích phụ đó lại là lợi ích chính khi kết quả không được thay đổi giữa các lần chạy. Mỗi tổng khối vốn đã được mã
            cố định; chỉ thứ tự gặp nhau là thay đổi, và giờ nó cũng được cố định. Phép rút gọn trong thư viện của chính NVIDIA
            lấy điều này làm mặc định, và chỉ cung cấp thiết kế atomic khi được yêu cầu: hai thiết kế của sách là hai chế độ của
            CUB.
          </p>
        }
      >
        <p>
          That side benefit is the main one when results must not change between runs. Every block sum is already fixed by the
          code; only the meeting order varied, and now it is fixed too. NVIDIA's own library reduction makes this its
          default, and offers the atomic design only on request: the book's two designs are CUB's two modes.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch10-facts A13)">{/* i18n-ignore: date and citation only */}
        <Translated
          vi={
            <p>
              <code>cub::DeviceReduce</code>: tất định giữa các lần chạy theo mặc định (một cây cố định, không atomic); atomic
              chỉ ở chế độ <code>not_guaranteed</code>.
            </p>
          }
        >
          <p>
            <code>cub::DeviceReduce</code>: run-to-run deterministic by default (a fixed tree, no atomics); atomic only as{' '}
            <code>not_guaranteed</code>.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS109['ch10/10.9/q-reproducible']} />

      <h2>
        <Translated vi="Max và argmax xuyên khối">Max and argmax across blocks</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bài tập 4 yêu cầu một max thay cho tổng. Bên trong một khối không có gì thay đổi ngoài toán tử. Lần commit là nơi các
            backend rẽ lối, và argmax, bước đứng sau decode greedy, còn khó hơn vì giá trị và chỉ số của nó phải đi cùng nhau.
          </p>
        }
      >
        <p>
          Exercise 4 asks for a max instead of a sum. Inside a block nothing changes but the operator. The commit is where the
          backends part ways, and argmax, the step behind greedy decoding, is harder still because the value and its index have
          to move together.
        </p>
      </Translated>
      <Source section="10.12" pages="250" figs={tr('exercise 4', 'bài tập 4')} />
      <table className="data wide" style={{ maxWidth: '58rem' }} aria-label={tr('Committing across blocks', 'Commit xuyên khối')}>
        <thead>
          <tr>
            <th><Translated vi="Commit">Commit</Translated></th>
            <th>CUDA</th>
            <th>HIP</th>
            <th>Metal</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="Tổng float">Float sum</Translated></td>
            <td>
              <Translated vi={<><code>atomic_ref&lt;float&gt;::fetch_add</code> hoặc <code>atomicAdd(float*)</code></>}>
                <><code>atomic_ref&lt;float&gt;::fetch_add</code> or <code>atomicAdd(float*)</code></>
              </Translated>
            </td>
            <td>
              <Translated vi={<>cùng các lời gọi đó (<code>atomic_ref</code> qua libhipcxx)</>}>
                <>same calls (<code>atomic_ref</code> through libhipcxx)</>
              </Translated>
            </td>
            <td>
              <Translated
                vi={<><code>atomic_fetch_add_explicit</code> trên <code>device atomic_float*</code>, Metal 3, Apple7: mọi máy Mac dùng Apple silicon</>}
              >
                <><code>atomic_fetch_add_explicit</code> on <code>device atomic_float*</code>, Metal 3, Apple7: every Apple silicon Mac</>
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="Max float">Float max</Translated></td>
            <td>
              <Translated
                vi={<>không có <code>atomicMax(float*)</code> legacy; <code>fetch_max</code> trên float trong CCCL hiện hành, không rõ bộ công cụ đầu tiên; hoặc một vòng lặp compare-and-swap</>}
              >
                <>no legacy <code>atomicMax(float*)</code>; <code>fetch_max</code> on float in current CCCL, first toolkit unknown; or a compare-and-swap loop</>
              </Translated>
            </td>
            <td>
              <Translated
                vi={<>có <code>atomicMax(float*)</code> (chỉ HIP); <code>fetch_max</code> của libhipcxx là một vòng lặp compare-and-swap</>}
              >
                <><code>atomicMax(float*)</code> exists (HIP only); libhipcxx's <code>fetch_max</code> is a compare-and-swap loop</>
              </Translated>
            </td>
            <td>
              <Translated
                vi={<>không có atomic max float; một vòng lặp compare-and-swap trên <code>atomic_float</code>, hoặc một lượt thứ hai</>}
              >
                <>no float atomic max; a compare-and-swap loop on <code>atomic_float</code>, or a second pass</>
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="(giá trị, chỉ số) cho argmax">(value, index) for argmax</Translated></td>
            <td>
              <Translated
                vi={<>một khóa 64-bit đóng gói với <code>atomicMax(unsigned long long*)</code>, hoặc một lượt thứ hai</>}
              >
                <>a packed 64-bit key with <code>atomicMax(unsigned long long*)</code>, or a second pass</>
              </Translated>
            </td>
            <td><Translated vi="như trên">same</Translated></td>
            <td>
              <Translated vi="atomic max 64-bit cần Apple9 (Apple8 trên macOS) và không trả về gì; nếu không thì một lượt thứ hai">
                64-bit atomic max needs Apple9 (Apple8 on macOS) and returns nothing; otherwise a second pass
              </Translated>
            </td>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Cách chạy được ở mọi nơi là hai lượt, và đó là cách backend Metal của llama.cpp dùng cho argmax: một lần dispatch ghi
            một cặp (giá trị, chỉ số) bộ phận cho mỗi threadgroup, lần thứ hai rút gọn chúng. Lối thoát còn lại là tránh hẳn việc
            commit xuyên khối, bằng cách giao cho mỗi hàng một khối. Đó là điều mốc dự án của chương làm (lab 10.3), và mục
            10.10 cho thấy cái giá của nó.
          </p>
        }
      >
        <p>
          The route that works everywhere is two passes, and it is the one llama.cpp's Metal backend takes for argmax: one
          dispatch writes a partial (value, index) pair per threadgroup, a second reduces them. The other way out is to avoid
          committing across blocks at all, by giving each row one block. That is what the chapter's milestone does (lab 10.3),
          and section 10.10 shows what it costs.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Bảng và cách làm của llama.cpp là của khóa học này, từ tài liệu của các hãng và mã nguồn dự án đọc vào 2026-11-20
              (notes/today/ch10-facts.md A7, B3, C2, D2; infra-facts C1). Chưa có gì ở đây được chạy trên phần cứng CUDA hay AMD.
            </p>
          }
        >
          <p>
            The table and the llama.cpp route are ours, from vendor documentation and project source read on 2026-11-20
            (notes/today/ch10-facts.md A7, B3, C2, D2; infra-facts C1). Nothing here has run on CUDA or AMD hardware.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS109['ch10/10.9/q-max-commit']} />
      <Ask q={questionsS109['ch10/10.9/q-argmax-pack']} />
      <Translated
        vi={
          <p>
            Lab 10.1 có kernel cho độ dài bất kỳ, phép đếm tính tất định và phép quét hệ số gộp việc; lab 10.3 dựng các hàng
            softmax và argmax. Cả hai đều ở <a href={href('ch10', 'labs')}>trang lab</a>.
          </p>
        }
      >
        <p>
          Lab 10.1 has the any-length kernel, the determinism count and the coarsening sweep; lab 10.3 builds the softmax and
          argmax rows. Both are on the <a href={href('ch10', 'labs')}>labs page</a>.
        </p>
      </Translated>
    </>
  );
}
