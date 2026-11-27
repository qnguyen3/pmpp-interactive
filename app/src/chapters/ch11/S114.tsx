import { useEffect, useMemo, useState, type KeyboardEvent } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { Aside, Bench, Beyond, NoteGroup, Simplification, Source, Today, type Tier } from '../../components/notes';
import { Translated, localized, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { functionSource } from '../../sim/toolchain';
import {
  FIG_11_6,
  HALF_CASES,
  LANE_TEST_CODE,
  PORT_TARGETS,
  PRINTED_FILE,
  halfCase,
  issueLines,
  laneValues,
  portIssues,
  scanScanAdd,
  scorePort,
  warpScanTrace,
  type HalfCase,
  type LaneTest,
  type PortTarget,
  type WarpScanTrace,
} from '../../sim/warpscan';

// What has been checked for the snippet shown here, in the course's fixed vocabulary (the lab's files read theirs from the reports).
const STATUS = {
  printed: 'front-end-checked; emulator-checked at warp width 32; unverified on CUDA hardware; as printed, CUDA only',
} as const;
const LAB = 'ch11/lab1_blockscan/solution';
const missing = (s: string | null) => !s || s.startsWith('// could not load');

const TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', warpSums_s: 'shared',
  val: 'reg', leftVal: 'reg', warpSum: 'reg', stride: 'reg', i: 'reg', full: 'reg', s: 'reg', lane: 'reg', sg: 'reg',
};

const range = (n: number, from = 0) => Array.from({ length: n }, (_, i) => from + i);
const count = (n: number, one: string, many = one + 's') => `${n} ${n === 1 ? one : many}`;
const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
/** "lanes 3, 4 and 5" or "lanes 0 to 30". */
function lanesText(ls: number[], word = 'lane'): string {
  if (!ls.length) return tr('no ' + word + 's', 'không ' + word + ' nào');
  if (ls.length === 1) return `${word} ${ls[0]}`;
  const contiguous = ls.every((l, i) => i === 0 || l === ls[i - 1] + 1);
  if (contiguous && ls.length > 3) return tr(`${word}s ${ls[0]} to ${ls[ls.length - 1]}`, `${word} ${ls[0]} đến ${ls[ls.length - 1]}`);
  return tr(`${word}s ${ls.slice(0, -1).join(', ')} and ${ls[ls.length - 1]}`, `${word} ${ls.slice(0, -1).join(', ')} và ${ls[ls.length - 1]}`);
}


/** True while the viewport is at most `px` wide. */
function useNarrow(px = 600): boolean {
  const query = `(max-width: ${px}px)`; // i18n-ignore: CSS media query
  const [narrow, setNarrow] = useState(() => typeof window !== 'undefined' && window.matchMedia(query).matches);
  useEffect(() => {
    const m = window.matchMedia(query);
    const on = () => setNarrow(m.matches);
    on();
    m.addEventListener('change', on);
    return () => m.removeEventListener('change', on);
  }, [query]);
  return narrow;
}

// ---------------------------------------------------------------- explorer: scan, scan, add
type Unit = 'warp' | 'run' | 'block';
const UNIT: Record<Unit, { readonly seg: string; readonly s1: string; readonly s2: string; readonly s3: string; readonly fig: string; readonly where: string }> = {
  warp: {
    get seg() {
      return tr('warp', 'warp');
    },
    get s1() {
      return tr('every warp scans its own values with shuffles', 'mỗi warp scan các giá trị của chính nó bằng shuffle');
    },
    get s2() {
      return tr('one warp scans the warp totals', 'một warp scan các tổng của warp');
    },
    get s3() {
      return tr('every warp but the first adds the scanned total before it', 'mỗi warp trừ warp đầu tiên cộng tổng đã scan đứng trước nó');
    },
    get fig() {
      return tr('Fig. 11.7, this section', 'Fig. 11.7, phần này');
    },
    get where() {
      return tr('The totals pass through warpSums_s, in shared memory.', 'Các tổng đi qua warpSums_s, trong bộ nhớ chia sẻ.');
    },
  },
  run: {
    get seg() {
      return tr("thread's run", 'đoạn của một luồng');
    },
    get s1() {
      return tr('every thread scans its own run in a loop', 'mỗi luồng scan đoạn của chính nó trong một vòng lặp');
    },
    get s2() {
      return tr('one block scan of the run totals', 'một lần scan khối trên các tổng của đoạn');
    },
    get s3() {
      return tr('every thread but the first adds the scanned total before it', 'mỗi luồng trừ luồng đầu tiên cộng tổng đã scan đứng trước nó');
    },
    get fig() {
      return tr('Fig. 11.11, section 11.6', 'Fig. 11.11, phần 11.6');
    },
    get where() {
      return tr('The totals pass through threadSums, in shared memory.', 'Các tổng đi qua threadSums, trong bộ nhớ chia sẻ.');
    },
  },
  block: {
    get seg() {
      return tr('block', 'khối');
    },
    get s1() {
      return tr('every block scans its own segment', 'mỗi khối scan đoạn của chính nó');
    },
    get s2() {
      return tr('one scan of the block sums', 'một lần scan trên các tổng của khối');
    },
    get s3() {
      return tr('every block but the first adds the scanned sum before it', 'mỗi khối trừ khối đầu tiên cộng tổng đã scan đứng trước nó');
    },
    get fig() {
      return tr('Fig. 11.14, section 11.9', 'Fig. 11.14, phần 11.9');
    },
    get where() {
      return tr('The block sums pass through global memory, between kernels or blocks.', 'Các tổng của khối đi qua bộ nhớ toàn cục, giữa các kernel hoặc các khối.');
    },
  },
};

function ScanScanAddExplorer() {
  const [vals, setVals] = useState<number[]>(FIG_11_6);
  const [seg, setSeg] = useState(4);
  const [unit, setUnit] = useState<Unit>('warp');
  const narrow = useNarrow();
  const r = useMemo(() => scanScanAdd(vals, seg), [vals, seg]);
  const n = vals.length;
  const segs = n / seg;
  const W = narrow ? 344 : 704;
  const pad = 2;
  const g = narrow ? 5 : 10;
  const cw = (W - 2 * pad - (segs - 1) * g) / n;
  const x = (i: number) => pad + i * cw + Math.floor(i / seg) * g;
  const CH = narrow ? 22 : 26;
  const LBL = 15;
  const ROW = CH + LBL + 14;
  const y = (row: number) => row * ROW + LBL;
  const font = narrow ? 9.5 : 11.5;
  const u = UNIT[unit];
  const ends = range(segs).map((k) => k * seg + seg - 1);
  const label = (row: number, text: string) => (
    <text x={pad} y={y(row) - 4} fontSize={narrow ? 9.5 : 11} style={{ fill: 'var(--ink-2)' }}>{text}</text>
  );
  const cell = (i: number, row: number, v: number, kind: 'in' | 'mid' | 'total' | 'final', mark = false) => (
    <g key={`${row}-${i}`}>
      <rect x={x(i) + 0.8} y={y(row)} width={cw - 1.6} height={CH} rx="2"
        fill={kind === 'total' ? 'var(--shared-wash)' : kind === 'final' ? 'var(--reg-wash)' : 'var(--cell)'}
        stroke={mark ? 'var(--ink)' : kind === 'total' ? 'var(--shared)' : kind === 'final' ? 'var(--reg)' : 'var(--rule)'}
        strokeWidth={mark ? 2 : 1} />
      <text x={x(i) + cw / 2} y={y(row) + CH / 2 + font / 2.8} fontSize={font} textAnchor="middle" style={{ fill: 'var(--ink)' }}>{v}</text>
    </g>
  );
  const H = y(3) + CH + 20;
  const setOne = (i: number, text: string) => {
    const v = Math.round(Number(text));
    setVals((old) => old.map((o, k) => (k === i ? (Number.isFinite(v) ? Math.max(-99, Math.min(999, v)) : 0) : o)));
  };
  const input = (i: number) => (
    <input key={i} type="number" step={1} value={vals[i]} aria-label={tr(`input ${i}`, `đầu vào ${i}`)} data-input={i}
      style={{ width: '100%', minWidth: 0, padding: '0.2rem 0.1rem', textAlign: 'center', boxSizing: 'border-box' }}
      onChange={(e) => setOne(i, e.target.value)} />
  );

  return (
    <Bench
      title={tr('Scan, scan, add', 'Scan, scan, cộng')}
      task={
        <Translated vi="Thay đổi một đầu vào bất kỳ và quan sát cả ba giai đoạn. Sau đó đổi kích thước đoạn, và tên gọi của một đoạn.">
          Change any input and watch all three stages. Then switch the segment size, and the name of a segment.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Các hàng đi xuống theo thứ tự: giai đoạn 1, các tổng, giai đoạn 2, giai đoạn 3. Viền đậm đánh dấu một giá trị của
              giai đoạn 2 và chỗ nó xuất hiện lại trong kết quả. Các đường cong nét đứt mang mỗi tổng đã scan đến đoạn mà nó
              được cộng vào.
            </>
          }
        >
          <>
            Rows run down in order: stage 1, the totals, stage 2, stage 3. A thick outline marks a value of stage 2 and the
            place it reappears in the result. Dashed curves carry each scanned total to the segment it is added to.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label={tr('Segment size', 'Kích thước đoạn')}>
          {[2, 4, 8].map((s) => (
            <button key={s} aria-pressed={seg === s} onClick={() => setSeg(s)}><Translated vi={<>đoạn {s} phần tử</>}>segments of {s}</Translated></button>
          ))}
        </span>
        <label className="field">
          <Translated vi="Một đoạn là">A segment is</Translated>
          <select value={unit} aria-label={tr('what a segment is', 'một đoạn là gì')} onChange={(e) => setUnit(e.target.value as Unit)}>
            <option value="warp">{tr('a warp (Fig. 11.7)', 'một warp (Fig. 11.7)')}</option>
            <option value="run">{tr("a thread's run (Fig. 11.11)", 'đoạn của một luồng (Fig. 11.11)')}</option>
            <option value="block">{tr('a block (Fig. 11.14)', 'một khối (Fig. 11.14)')}</option>
          </select>
        </label>
        <button className="quiet" onClick={() => setVals(FIG_11_6)}><Translated vi="Các số của Fig. 11.6">Fig. 11.6's numbers</Translated></button>
        <button className="quiet" onClick={() => setVals(new Array(16).fill(1))}><Translated vi="toàn số 1">all ones</Translated></button>
      </div>
      <div className="muted" style={{ fontSize: '0.8rem', margin: '0 0 0.25rem' }}><Translated vi={<>đầu vào, chia thành các đoạn {seg} phần tử: mỗi {u.seg} một đoạn (gõ để thay đổi)</>}>input, in segments of {seg}: one segment per {u.seg} (type to change)</Translated></div>
      {narrow ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, minmax(0, 1fr))', gap: '3px', marginBottom: '0.6rem' }} aria-label={tr('Inputs', 'Đầu vào')}>
          {range(n).map(input)}
        </div>
      ) : (
        <div style={{ display: 'flex', gap: `${(g / W) * 100}%`, maxWidth: W, marginBottom: '0.6rem', padding: `0 ${(pad / W) * 100}%`, boxSizing: 'border-box' }} aria-label={tr('Inputs', 'Đầu vào')}>
          {range(segs).map((k) => (
            <div key={k} style={{ flex: seg, display: 'grid', gridTemplateColumns: `repeat(${seg}, minmax(0, 1fr))`, gap: '2px' }}>
              {range(seg, k * seg).map(input)}
            </div>
          ))}
        </div>
      )}
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('The three stages of scan-scan-add', 'Ba giai đoạn của scan-scan-cộng')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
        {label(0, tr(`stage 1: ${u.s1}`, `giai đoạn 1: ${u.s1}`))}
        {r.stage1.map((v, i) => cell(i, 0, v, 'mid'))}
        {label(1, tr('each segment\'s total: its last value after stage 1', 'tổng của mỗi đoạn: giá trị cuối của nó sau giai đoạn 1'))}
        {r.totals.map((v, k) => cell(ends[k], 1, v, 'total'))}
        {label(2, tr(`stage 2: ${u.s2}`, `giai đoạn 2: ${u.s2}`))}
        {r.stage2.map((v, k) => cell(ends[k], 2, v, 'total', true))}
        {range(segs - 1).map((k) => {
          const x1 = x(ends[k]) + cw / 2;
          const y1 = y(2) + CH;
          const x2 = x((k + 1) * seg) + (seg * cw) / 2;
          const y2 = y(3) - 1;
          return <path key={'a' + k} d={`M ${x1} ${y1} C ${x1} ${y1 + 12}, ${x2} ${y2 - 14}, ${x2} ${y2}`} fill="none" stroke="var(--shared)" strokeWidth="1.3" strokeDasharray="4 2" />;
        })}
        {r.final.map((v, i) => cell(i, 3, v, 'final', ends.includes(i)))}
        <text x={pad} y={y(3) + CH + 14} fontSize={narrow ? 9.5 : 11} style={{ fill: 'var(--ink-2)' }}>{tr(`stage 3: ${u.s3}`, `giai đoạn 3: ${u.s3}`)}</text>
      </svg>
      <div className="row" style={{ margin: '0.6rem 0 0', display: 'none' }}>
        <span data-row="stage1">{r.stage1.join(' ')}</span>
        <span data-row="totals">{r.totals.join(' ')}</span>
        <span data-row="stage2">{r.stage2.join(' ')}</span>
        <span data-row="final">{r.final.join(' ')}</span>
      </div>
      <Translated
        vi={
          <p className="narrate" role="status" style={{ margin: '0.7rem 0 0' }} data-ssa-says>
            Giai đoạn 2 chứa {r.stage2.join(', ')}; kết quả chứa đúng các số đó ở các vị trí {ends.join(', ')}, cuối các
            đoạn. {segs > 1 ? `Giai đoạn 3 cộng ${r.added.slice(1).join(', ')} vào các đoạn 1 đến ${segs - 1}. ` : ''}
            Số phép cộng: {r.additions.stage1} ở giai đoạn 1, {r.additions.stage2} ở giai đoạn 2, {r.additions.stage3} ở giai
            đoạn 3, so với {n - 1} của vòng lặp. {u.where} Cùng ba giai đoạn này xuất hiện trong {u.fig}.
          </p>
        }
      >
        <p className="narrate" role="status" style={{ margin: '0.7rem 0 0' }} data-ssa-says>
          Stage 2 holds {r.stage2.join(', ')}; the result holds the same numbers at positions {ends.join(', ')}, the ends of the
          segments. {segs > 1 ? `Stage 3 adds ${r.added.slice(1).join(', ')} to segments 1 to ${segs - 1}. ` : ''}
          Additions: {r.additions.stage1} in stage 1, {r.additions.stage2} in stage 2, {r.additions.stage3} in stage 3, against{' '}
          {n - 1} for the loop. {u.where} The same three stages appear in {u.fig}.
        </p>
      </Translated>
    </Bench>
  );
}

// ---------------------------------------------------------------- stepper: shuffle up
type Lanes = 8 | 32;
const TESTS: { key: LaneTest; readonly label: string }[] = [
  {
    key: 'laneIdx',
    get label() {
      return tr('laneIdx() >= stride (the book)', 'laneIdx() >= stride (sách)');
    },
  },
  {
    key: 'threadIdx',
    get label() {
      return tr('threadIdx.x >= stride, in warp 1 of the block', 'threadIdx.x >= stride, trong warp 1 của khối');
    },
  },
  {
    key: 'none',
    get label() {
      return tr('no test', 'không kiểm tra');
    },
  },
];

function stepperTrace(lanes: Lanes, test: LaneTest, data: 'small' | 'ones'): WarpScanTrace {
  const warp = test === 'threadIdx' ? 1 : 0;
  const v = data === 'ones' ? new Array(lanes).fill(1) : laneValues(lanes, warp * lanes);
  return warpScanTrace(v, { test, warp });
}

function stepSays(t: WarpScanTrace, k: number): string {
  if (k === 0) {
    const first = t.test === 'threadIdx' ? tr(` This is warp 1, so threadIdx.x runs from ${t.hw} to ${2 * t.hw - 1} while the lane runs from 0 to ${t.hw - 1}.`, ` Đây là warp 1, nên threadIdx.x chạy từ ${t.hw} đến ${2 * t.hw - 1} trong khi số lane chạy từ 0 đến ${t.hw - 1}.`) : '';
    return tr(`Each lane holds one value in a register.${first} The loop runs for strides ${t.steps.map((s) => s.stride).join(', ')}. Press Step.`, `Mỗi lane giữ một giá trị trong một thanh ghi (register).${first} Vòng lặp chạy với các bước nhảy (stride) ${t.steps.map((s) => s.stride).join(', ')}. Hãy nhấn Bước.`);
  }
  const st = t.steps[k - 1];
  const s = st.stride;
  const moved = st.lanes.filter((l) => l.src !== null);
  const own = st.lanes.filter((l) => l.src === null);
  const doubled = st.lanes.filter((l) => l.doubled);
  let out = tr(`Stride ${s}: every lane calls __shfl_up_sync and receives the value held ${s} lane${s === 1 ? '' : 's'} to its left. ` +
    `${cap(lanesText(moved.map((l) => l.lane)))} receive another lane's value and add it. `, `Bước nhảy ${s}: mọi lane gọi __shfl_up_sync và nhận giá trị do lane cách nó ${s} vị trí về bên trái giữ. ` +
    `${cap(lanesText(moved.map((l) => l.lane)))} nhận giá trị của một lane khác và cộng vào. `);
  out += tr(`${cap(lanesText(own.map((l) => l.lane)))} ${own.length === 1 ? 'has' : 'have'} no lane ${s} to the left, so the shuffle hands back ${own.length === 1 ? 'its' : 'their'} own value. `, `${cap(lanesText(own.map((l) => l.lane)))} không có lane nào cách ${s} vị trí về bên trái, nên shuffle trả lại chính giá trị của ${own.length === 1 ? 'nó' : 'chúng'}. `);
  if (doubled.length)
    out += t.test === 'none'
      ? tr(`With no test, ${doubled.length === 1 ? 'it adds it' : 'they add it'} anyway: ${lanesText(doubled.map((l) => l.lane))} double${doubled.length === 1 ? 's' : ''}.`, `Không có phép kiểm tra, ${doubled.length === 1 ? 'nó' : 'chúng'} vẫn cộng giá trị đó vào: ${lanesText(doubled.map((l) => l.lane))} bị nhân đôi.`)
      : tr(`threadIdx.x is at least ${t.hw} here, never below the stride, so the test lets ${doubled.length === 1 ? 'it' : 'them'} add it: ${lanesText(doubled.map((l) => l.lane))} double${doubled.length === 1 ? 's' : ''}.`, `Ở đây threadIdx.x ít nhất là ${t.hw}, không bao giờ nhỏ hơn bước nhảy, nên phép kiểm tra cho ${doubled.length === 1 ? 'nó' : 'chúng'} cộng giá trị đó vào: ${lanesText(doubled.map((l) => l.lane))} bị nhân đôi.`);
  else out += tr(`The test ${LANE_TEST_CODE[t.test]} stops ${own.length === 1 ? 'it' : 'them'} from adding it.`, `Phép kiểm tra ${LANE_TEST_CODE[t.test]} ngăn ${own.length === 1 ? 'nó' : 'chúng'} cộng giá trị đó vào.`);
  return out;
}

function ShuffleStepper() {
  const [lanes, setLanes] = useState<Lanes>(8);
  const [test, setTest] = useState<LaneTest>('laneIdx');
  const [data, setData] = useState<'small' | 'ones'>('small');
  const [k, setK] = useState(0);
  const narrow = useNarrow();
  const t = useMemo(() => stepperTrace(lanes, test, data), [lanes, test, data]);
  const total = t.steps.length;
  const at = Math.min(k, total);
  useEffect(() => setK(0), [lanes, test, data]);
  const done = at === total;

  const VW = narrow && lanes === 8 ? 360 : 720;
  const LW = narrow && lanes === 8 ? 52 : 70;
  const cw = Math.min((VW - LW - 6) / lanes, 64);
  const X0 = LW + (VW - LW - 6 - cw * lanes) / 2;
  const CH = 24;
  const GAP = 30;
  const rowY = (r: number) => 18 + r * (CH + GAP);
  const rows = at + 1 + (done ? 1 : 0);
  const H = rowY(rows - 1) + CH + 8;
  const font = cw < 24 ? 9 : 11.5;
  const xs = (l: number) => X0 + l * cw;
  const valuesAt = (r: number) => (r === 0 ? t.initial : t.steps[r - 1].lanes.map((l) => l.after));
  const cur = at > 0 ? t.steps[at - 1] : null;
  const wrongSet = new Set(t.wrong);

  return (
    <Bench
      title={tr('Shuffle up, lane by lane', 'Shuffle lên, từng lane một')}
      task={
        <Translated vi="Chạy từng bước vòng lặp của sách trên 8 lane. Sau đó bỏ phép kiểm tra, hoặc viết nó bằng threadIdx.x, và tìm các lane bị sai.">
          Step the book's loop on 8 lanes. Then remove the test, or write it with threadIdx.x, and find the lanes that go wrong.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Các hàng đi xuống theo thời gian: các giá trị ban đầu, rồi giá trị của mỗi lane sau mỗi bước nhảy. Mũi tên liền:
              một giá trị nhận từ lane khác và được cộng vào. Vòng nét đứt: lane nhận lại chính giá trị của nó. Màu đỏ: một
              lane đã cộng giá trị của chính nó vào bản thân. Sau bước nhảy cuối cùng, đáp án đúng được vẽ bên dưới để so
              sánh.
            </>
          }
        >
          <>
            Rows run down in time: the starting values, then each lane's value after each stride. Solid arrows: a value received
            from another lane and added. A dashed loop: the lane got its own value back. Red: a lane that added its own value to
            itself. After the last stride, the right answer is drawn below for comparison.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label={tr('Lanes', 'Số lane')}>
          <button aria-pressed={lanes === 8} onClick={() => setLanes(8)}><Translated vi="8 lane (đồ chơi)">8 lanes (a toy)</Translated></button>
          <button aria-pressed={lanes === 32} onClick={() => setLanes(32)}><Translated vi="32 lane">32 lanes</Translated></button>
        </span>
        <label className="field">
          <Translated vi="Phép kiểm tra tham gia">Participation test</Translated>
          <select value={test} aria-label={tr('participation test', 'phép kiểm tra tham gia')} onChange={(e) => setTest(e.target.value as LaneTest)}>
            {TESTS.map((x) => <option key={x.key} value={x.key}>{x.label}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Giá trị">Values</Translated>
          <select value={data} aria-label={tr('values', 'giá trị')} onChange={(e) => setData(e.target.value as 'small' | 'ones')}>
            <option value="small">{tr('small integers', 'số nguyên nhỏ')}</option>
            <option value="ones">{tr('all ones', 'toàn số 1')}</option>
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <button onClick={() => setK(0)} disabled={at === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => setK(at - 1)} disabled={at === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => setK(at + 1)} disabled={done}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => setK(total)} disabled={done}><Translated vi="Chạy đến cuối">Run to the end</Translated></button>
        <span className="muted" aria-live="polite"><Translated vi={<>bước {at} trên {total}</>}>step {at} of {total}</Translated></span>
      </div>
      <pre className="mono" style={{ margin: '0 0 0.6rem', whiteSpace: 'pre-wrap' }} data-loop>
        {`for (stride = 1; stride < ${lanes}; stride *= 2) {\n    leftVal = __shfl_up_sync(mask, val, stride);\n    ${test === 'none' ? 'val += leftVal;                // no test' : `${LANE_TEST_CODE[test]} val += leftVal;`}\n}`}
      </pre>
      <div className="narrate" role="status" style={{ marginBottom: '0.7rem' }} data-narrate>{stepSays(t, at)}</div>
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${VW} ${H}`} role="img" aria-label={tr("Every lane's value after each stride", 'Giá trị của mọi lane sau mỗi bước nhảy')} data-stepper
          style={{ width: '100%', maxWidth: VW, minWidth: lanes === 32 && narrow ? 640 : 0, display: 'block' }}>
          <text x={4} y={10} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>{test === 'threadIdx' ? 'threadIdx.x' : tr('lane', 'lane')}</text>
          {range(lanes).map((l) => (lanes === 8 || l % 4 === 0 ? (
            <text key={'h' + l} x={xs(l) + cw / 2} y={10} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
              {test === 'threadIdx' ? lanes + l : l}
            </text>
          ) : null))}
          {/* arrows for the step just taken */}
          {cur
            ? cur.lanes.map((l) => {
                const y1 = rowY(at - 1) + CH;
                const y2 = rowY(at);
                if (l.src === null) {
                  const cx = xs(l.lane) + cw / 2;
                  return (
                    <path key={'a' + l.lane} d={`M ${cx - 3} ${y1 + 1} C ${cx - 10} ${y1 + 14}, ${cx + 10} ${y1 + 14}, ${cx + 3} ${y2 - 1}`}
                      fill="none" stroke={l.doubled ? 'var(--fail)' : 'var(--ink-3)'} strokeDasharray="2 2" strokeWidth={l.doubled ? 1.6 : 1} />
                  );
                }
                return (
                  <line key={'a' + l.lane} x1={xs(l.src) + cw / 2} y1={y1 + 1} x2={xs(l.lane) + cw / 2} y2={y2 - 1}
                    stroke="var(--reg)" strokeWidth={lanes === 8 ? 1.6 : 1} opacity={0.8} />
                );
              })
            : null}
          {range(at + 1).map((r) => (
            <g key={'r' + r}>
              <text x={4} y={rowY(r) + CH / 2 + 4} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>{r === 0 ? tr('start', 'đầu') : tr(`stride ${t.steps[r - 1].stride}`, `bước nhảy ${t.steps[r - 1].stride}`)}</text>
              {valuesAt(r).map((v, l) => {
                const st = r > 0 ? t.steps[r - 1].lanes[l] : null;
                const kind = !st ? 'start' : st.doubled ? 'doubled' : st.adds ? 'added' : 'kept';
                return (
                  <g key={l} data-cell={`${r}-${l}`} data-kind={kind} data-value={v}>
                    <rect x={xs(l) + 0.8} y={rowY(r)} width={cw - 1.6} height={CH} rx="2"
                      fill={kind === 'doubled' ? 'var(--fail-wash)' : kind === 'added' ? 'var(--reg-wash)' : 'var(--cell)'}
                      stroke={kind === 'doubled' ? 'var(--fail)' : kind === 'added' ? 'var(--reg)' : 'var(--rule)'}
                      strokeDasharray={kind === 'kept' ? '3 2' : undefined} />
                    <text x={xs(l) + cw / 2} y={rowY(r) + CH / 2 + font / 2.8} fontSize={font} textAnchor="middle" style={{ fill: 'var(--ink)' }}>{v}</text>
                  </g>
                );
              })}
            </g>
          ))}
          {done ? (
            <g data-expected>
              <text x={4} y={rowY(at + 1) + CH / 2 + 4} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="đúng">right</Translated></text>
              {t.expected.map((v, l) => (
                <g key={l}>
                  <rect x={xs(l) + 0.8} y={rowY(at + 1)} width={cw - 1.6} height={CH} rx="2" fill="none"
                    stroke={wrongSet.has(l) ? 'var(--fail)' : 'var(--ok)'} strokeWidth={wrongSet.has(l) ? 1.6 : 1} />
                  <text x={xs(l) + cw / 2} y={rowY(at + 1) + CH / 2 + font / 2.8} fontSize={font} textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{v}</text>
                </g>
              ))}
            </g>
          ) : null}
        </svg>
      </div>
      {done ? (
        <p style={{ margin: '0.7rem 0 0' }} data-result={t.wrong.length ? 'wrong' : 'ok'}>
          {t.wrong.length ? (
            <Translated
              vi={
                <>
                  <span className="chip fail">sai</span> {cap(lanesText(t.wrong))} khác với kết quả scan.{' '}
                  {cap(lanesText(t.doubled))} đã cộng chính giá trị của {t.doubled.length === 1 ? 'nó' : 'chúng'} ít nhất một
                  lần, và mọi lane về sau đọc một trong số đó đều thừa hưởng lỗi.
                  {t.wrong.length === lanes - 1 ? ` Chỉ lane ${lanes - 1} còn đúng.` : ''}
                  {t.test === 'none' ? ` Lane 0 kết thúc ở ${t.final[0]}, gấp ${t.final[0] / t.initial[0]} lần đầu vào của nó.` : ''}
                </>
              }
            >
              <>
                <span className="chip fail">wrong</span> {cap(lanesText(t.wrong))} {t.wrong.length === 1 ? 'differs' : 'differ'} from the
                scan. {cap(lanesText(t.doubled))} added {t.doubled.length === 1 ? 'its' : 'their'} own value at least once, and every lane
                that later read one of them inherited the error.
                {t.wrong.length === lanes - 1 ? ` Only lane ${lanes - 1} survives.` : ''}
                {t.test === 'none' ? ` Lane 0 ends at ${t.final[0]}, ${t.final[0] / t.initial[0]} times its input.` : ''}
              </>
            </Translated>
          ) : (
            <Translated
              vi={
                <>
                  <span className="chip ok">đúng</span> Mọi lane giữ tổng của giá trị của chính nó và mọi giá trị bên trái nó,
                  sau {total} bước nhảy và {t.additions} phép cộng.
                </>
              }
            >
              <>
                <span className="chip ok">correct</span> Every lane holds the sum of its own value and all the values to its left,
                after {count(total, 'stride')} and {t.additions} additions.
              </>
            </Translated>
          )}
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- erratum hunt: Port it
function PortIt() {
  const src = useLabFile(PRINTED_FILE);
  const [target, setTarget] = useState<PortTarget>('hip64');
  const [clicked, setClicked] = useState<number[]>([]);
  const [revealed, setRevealed] = useState(false);
  const [half, setHalf] = useState<HalfCase>('mixed');
  const narrow = useNarrow();
  useEffect(() => {
    setClicked([]);
    setRevealed(false);
  }, [target]);
  if (missing(src)) return null;
  const lines = src!.replace(/\n$/, '').split('\n');
  const first = lines.findIndex((l) => l.startsWith('#define WARP_SIZE')); // skip the file's header comment
  const figs = [
    { name: 'warpScan', fig: '11.8' },
    { name: 'blockScan', fig: '11.9' },
    { name: 'scan_kernel', fig: '11.10' }, // i18n-ignore: function name
  ].map((f) => ({ ...f, at: functionSource(src!, f.name)?.firstLine ?? -1 }));
  // The gutter names where each line comes from: a figure and its line, chapter 10, or added (not printed).
  const gutter = (n: number) => {
    const f = [...figs].reverse().find((x) => x.at > 0 && n >= x.at);
    if (f && n <= functionSource(src!, f.name)!.lastLine) return `${f.fig}:${String(n - f.at + 1).padStart(2, '0')}`;
    const text = lines[n - 1];
    if (/not printed|^#ifndef|^#endif/.test(text)) return tr('added', 'thêm vào');
    if (/^#define WARP_SIZE|warpIdx\(\) \{|laneIdx\(\) \{/.test(text)) return tr('ch. 10', 'ch. 10');
    return '';
  };
  const issues = portIssues(target);
  const issueOf = (n: number) => issues.find((p) => issueLines(p).includes(n));
  const score = scorePort(target, clicked);
  const toggle = (n: number) => {
    if (revealed) return;
    setClicked((c) => (c.includes(n) ? c.filter((x) => x !== n) : [...c, n]));
  };
  const onKey = (e: KeyboardEvent, n: number) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      toggle(n);
    }
  };
  const lineStyle = (n: number) => {
    const on = clicked.includes(n);
    const iss = issueOf(n);
    if (revealed && iss) {
      const hit = score.found.includes(iss.id);
      return { background: hit ? 'var(--ok-wash)' : 'var(--fail-wash)', boxShadow: `inset 3px 0 0 ${hit ? 'var(--ok)' : 'var(--fail)'}` }; // i18n-ignore: CSS value
    }
    if (revealed && on) return { textDecoration: 'line-through', color: 'var(--ink-3)' };
    return on ? { background: 'var(--rule-soft)', boxShadow: 'inset 3px 0 0 var(--ink)' } : {}; // i18n-ignore: CSS value
  };

  // The 64-lane sub-view.
  const v64 = laneValues(64);
  const hc = halfCase(v64, half);
  const per = narrow ? 16 : 32;
  const VW = narrow ? 360 : 720;
  const cw = (VW - 8) / per;
  const RH = 40;
  const svgH = Math.ceil(64 / per) * RH + 6;
  const wrong = new Set(hc.wrong);

  return (
    <Bench
      title={tr('Port it', 'Chuyển mã')}
      task={
        <Translated vi="Chọn một nền tảng đích. Bấm vào mọi dòng bạn sẽ phải sửa, hoặc ít nhất phải kiểm tra, trước khi mã này chạy được ở đó. Rồi nhấn Kiểm tra.">
          Pick a target. Click every line you would have to change, or at least check, before this code runs there. Then press Check.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Fig. 11.8 đến 11.10 của sách và hai hàm phụ trợ từ chương 10, đúng như in (<code>labs/{PRINTED_FILE}</code>,{' '}
              {STATUS.printed}). Những thay đổi về cách viết như <code>__syncthreads()</code> trên Metal không tính: câu hỏi
              là những dòng nào không thể mang sang nguyên văn.
            </>
          }
        >
          <>
            The book's Figs. 11.8 to 11.10 and the two helpers from chapter 10, as printed (<code>labs/{PRINTED_FILE}</code>,{' '}
            {STATUS.printed}). Spelling changes such as <code>__syncthreads()</code> on Metal do not count: the question is which
            lines cannot be carried over word for word.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          <Translated vi="Nền tảng đích">Target</Translated>
          <select value={target} aria-label={tr('target', 'nền tảng đích')} onChange={(e) => setTarget(e.target.value as PortTarget)}>
            {PORT_TARGETS.map((p) => <option key={p.key} value={p.key}>{localized(p.label, ({ hip64: 'HIP trên wavefront rộng 64 (gfx9)', hip32: 'HIP trên wavefront rộng 32 (gfx10 trở về sau)', metal: 'Metal (MSL)' } as Record<PortTarget, string>)[p.key])}</option>)}
          </select>
        </label>
        <button className="primary" onClick={() => setRevealed(true)} disabled={revealed}><Translated vi="Kiểm tra">Check</Translated></button>
        <button onClick={() => { setClicked([]); setRevealed(false); }} disabled={!clicked.length && !revealed}><Translated vi="Xóa">Clear</Translated></button>
        <span className="muted" aria-live="polite"><Translated vi={<>đã đánh dấu {clicked.length} dòng</>}>{count(clicked.length, 'line')} marked</Translated></span>
      </div>
      <pre className="mono" style={{ margin: 0, padding: '0.5rem 0', fontSize: narrow ? '0.7rem' : '0.8rem', lineHeight: 1.55, overflowX: 'auto', background: 'var(--page)', border: '1px solid var(--rule)', borderRadius: 3 }} data-port-listing>
        {lines.slice(first).map((text, i) => {
          const n = first + i + 1;
          const iss = revealed ? issueOf(n) : undefined;
          return (
            <span key={n} role="checkbox" aria-checked={clicked.includes(n)} tabIndex={0} data-line={n}
              data-state={revealed ? (iss ? (score.found.includes(iss.id) ? 'found' : 'missed') : clicked.includes(n) ? 'extra' : '') : clicked.includes(n) ? 'marked' : ''}
              onClick={() => toggle(n)} onKeyDown={(e) => onKey(e, n)}
              style={{ display: 'block', cursor: revealed ? 'default' : 'pointer', padding: '0 0.4rem', whiteSpace: 'pre', ...lineStyle(n) }}>
              <span style={{ display: 'inline-block', width: '6.2em', color: 'var(--ink-3)', userSelect: 'none' }}>{gutter(n)}</span>
              {text || ' '}
            </span>
          );
        })}
      </pre>
      {revealed ? (
        <div style={{ marginTop: '0.8rem' }} data-port-result>
          <Translated
            vi={
              <p className="narrate" style={{ margin: '0 0 0.6rem' }}>
                Bạn tìm được {score.found.length} trên {issues.length}.
                {score.extra.length ? ` ${score.extra.length} dòng bạn đánh dấu vẫn ổn trên nền tảng này khi giữ nguyên (bị gạch ngang).` : ''}
              </p>
            }
          >
            <p className="narrate" style={{ margin: '0 0 0.6rem' }}>
              You found {score.found.length} of {issues.length}.
              {score.extra.length ? ` ${count(score.extra.length, 'marked line')} ${score.extra.length === 1 ? 'is' : 'are'} fine on this target as written (struck through).` : ''}
            </p>
          </Translated>
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              {issues.map((p) => (
                <tr key={p.id} data-issue={p.id}>
                  <td style={{ width: '9.5rem' }}>
                    <span className={'chip' + (p.kind === 'compile error' || p.kind === 'wrong or limited' ? ' fail' : '')}>{tr(p.kind, ({ 'compile error': 'lỗi biên dịch', 'wrong or limited': 'sai hoặc bị giới hạn', undocumented: 'không có trong tài liệu', limit: 'giới hạn', rewrite: 'viết lại' } as Record<string, string>)[p.kind] ?? p.kind)}</span>
                    <div className="muted" style={{ fontSize: '0.78rem', marginTop: '0.2rem' }}>{p.where}</div>
                  </td>
                  <td>
                    <code>{p.text}</code>
                    <div>{p.says}</div>
                    <div className="muted" style={{ fontSize: '0.78rem' }}>{p.facts}</div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      <h4 style={{ margin: '1.2rem 0 0.4rem' }}><Translated vi="Sáu mươi bốn lane, ba cách đếm">Sixty-four lanes, three ways to count</Translated></h4>
      <Translated
        vi={
          <p className="muted" style={{ margin: '0 0 0.6rem' }}>
            Khi mask đã được sửa, Fig. 11.8 chạy trên một wavefront rộng 64. Giới hạn vòng lặp và cách đánh số lane đều có thể
            viết bằng 32 hoặc bằng <code>warpSize</code>. Lane <i>t</i> truyền vào (5<i>t</i> mod 7) + 1.
          </p>
        }
      >
        <p className="muted" style={{ margin: '0 0 0.6rem' }}>
          The mask fixed, Fig. 11.8 runs on a 64-wide wavefront. The loop bound and the lane numbering can each be written
          with 32 or with <code>warpSize</code>. Lane <i>t</i> passes (5<i>t</i> mod 7) + 1.
        </p>
      </Translated>
      <span className="seg" role="group" aria-label={tr('How the code counts', 'Cách mã đếm')} style={{ flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        {HALF_CASES.map((c) => (
          <button key={c.key} aria-pressed={half === c.key} onClick={() => setHalf(c.key)}>{localized(c.label, ({ both32: 'giới hạn vòng lặp 32, lane đếm theo nhóm 32', mixed: 'giới hạn vòng lặp 32, lane đếm từ warpSize (64)', both64: 'warpSize (64) cho cả hai' } as Record<HalfCase, string>)[c.key])}</button>
        ))}
      </span>
      <pre className="mono" style={{ margin: '0.2rem 0 0.5rem', whiteSpace: 'pre-wrap' }}>{HALF_CASES.find((c) => c.key === half)!.code}</pre>
      <svg viewBox={`0 0 ${VW} ${svgH}`} role="img" aria-label={tr('Which of the 64 lanes end right', 'Lane nào trong 64 lane kết thúc đúng')} style={{ width: '100%', maxWidth: VW, display: 'block' }} data-half={half}>
        {range(64).map((l) => {
          const xx = 4 + (l % per) * cw;
          const yy = Math.floor(l / per) * RH + 12;
          const bad = wrong.has(l);
          return (
            <g key={l} data-lane64={l} data-ok={bad ? 'wrong' : 'right'}>
              {l % 8 === 0 ? <text x={xx + cw / 2} y={yy - 3} fontSize="8.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{l}</text> : null}
              <rect x={xx + 0.6} y={yy} width={cw - 1.2} height={22} rx="2" fill={bad ? 'var(--fail-wash)' : 'var(--ok-wash)'} stroke={bad ? 'var(--fail)' : 'var(--ok)'} />
              <text x={xx + cw / 2} y={yy + 14.5} fontSize={cw < 22 ? 7.5 : 9} textAnchor="middle" style={{ fill: 'var(--ink)' }}>{hc.final[l]}</text>
            </g>
          );
        })}
      </svg>
      <p className="narrate" role="status" style={{ margin: '0.6rem 0 0' }} data-half-says>
        {half === 'both32'
          ? tr(`All 64 lanes are right, counted the code's way: each half is scanned as its own warp of 32, in ${hc.steps.length} strides. warpIdx(), the warp totals and NUM_WARPS must then count in 32s too, and each shuffle reaches only half as far as the hardware could.`, `Cả 64 lane đều đúng, theo cách đếm của mã: mỗi nửa được scan như một warp 32 riêng, trong ${hc.steps.length} bước nhảy. Khi đó warpIdx(), các tổng của warp và NUM_WARPS cũng phải đếm theo nhóm 32, và mỗi lần shuffle chỉ với xa được một nửa so với khả năng của phần cứng.`)
          : half === 'mixed'
            ? tr(`${cap(lanesText(hc.wrong))} are wrong. Their test now passes at every stride, but the loop stops at 16, so a lane's sum reaches back only 31 lanes: lane 32 misses lane 0, lane 63 misses lanes 0 to 31. Lanes 0 to 31 are right.`, `${cap(lanesText(hc.wrong))} bị sai. Phép kiểm tra của chúng giờ qua ở mọi bước nhảy, nhưng vòng lặp dừng ở 16, nên tổng của một lane chỉ với ngược lại được 31 lane: lane 32 thiếu lane 0, lane 63 thiếu các lane 0 đến 31. Các lane 0 đến 31 đều đúng.`)
            : tr(`All 64 lanes are right, in ${hc.steps.length} strides: one scan over the whole wavefront. This is what the portable listing below does.`, `Cả 64 lane đều đúng, trong ${hc.steps.length} bước nhảy: một lần scan trên toàn bộ wavefront. Đây là điều mà mã khả chuyển bên dưới làm.`)}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- code
function BlockScanListing() {
  const printed = useLabFile(PRINTED_FILE);
  const cu = useLabFile(LAB + '/kernel.cu');
  const metal = useLabFile(LAB + '/kernel.metal');
  // The labs page's words (LabKit's solutionTabStatus), from this machine's runs.
  const { reports } = useReports();
  const labStatus = { cu: solutionTabStatus(reports, 'ch11/lab1_blockscan', 'kernel.cu'), metal: solutionTabStatus(reports, 'ch11/lab1_blockscan', 'kernel.metal') };
  if (missing(printed)) return null;
  const pick = (src: string | null, names: string[], extra?: RegExp) => {
    if (missing(src)) return null;
    const parts = names.map((n) => functionSource(src!, n)?.text).filter(Boolean) as string[];
    if (parts.length !== names.length) return null;
    const head = extra ? extra.exec(src!)?.[0] : undefined;
    return [...(head ? [head] : []), ...parts].join('\n\n');
  };
  const portable = pick(cu, ['laneIdx', 'warpIdx', 'warpScan', 'blockScan', 'WarpScanKernel'], /#if defined\(__HIPCC__\)[\s\S]*?#endif/);
  const shuffle = pick(metal, ['warp_scan', 'block_scan_shuffle', 'warp_scan_shuffle']);
  const prefix = pick(metal, ['block_scan_prefix', 'warp_scan_prefix']);
  const tabs = [
    { label: tr('as printed (CUDA only)', 'như in trong sách (chỉ CUDA)'), code: printed!, tiers: TIERS, status: STATUS.printed },
    ...(portable ? [{ label: tr('portable (lab 11.1)', 'khả chuyển (lab 11.1)'), code: portable, tiers: TIERS, status: labStatus.cu }] : []),
    ...(shuffle ? [{ label: tr('Metal, shuffle loop (lab 11.1)', 'Metal, vòng lặp shuffle (lab 11.1)'), code: shuffle, tiers: TIERS, status: labStatus.metal }] : []),
    ...(prefix ? [{ label: tr('Metal, simd_prefix_inclusive_sum (lab 11.1)', 'Metal, dùng simd_prefix_inclusive_sum (lab 11.1)'), code: prefix, tiers: TIERS, status: labStatus.metal }] : []),
  ];
  return (
    <Code
      tabs={tabs}
      caption={
        <Translated
          vi={
            <>
              Tab đầu tiên: Fig. 11.8 đến 11.10 đúng như in, cùng các hàm phụ trợ từ chương 10 (<code>labs/{PRINTED_FILE}</code>).
              Các tab còn lại là lời giải tham chiếu (solution) của lab 11.1 (<code>labs/{LAB}/</code>): một kiểu mask cho mỗi
              nhà cung cấp, <code>warpSize</code> thay cho 32, và các tổng của warp được khai báo trong kernel rồi truyền vào.
              Trên Metal, số lane và số SIMD group đến từ các thuộc tính của kernel, và tab cuối cùng làm mỗi lần scan SIMD
              group trong một lời gọi.
              {portable ? null : <> Các tệp của lab chưa có trên máy này, nên chỉ hiện tab đầu tiên.</>}
            </>
          }
        >
          <>
            First tab: Figs. 11.8 to 11.10 as printed, with the helpers from chapter 10 (<code>labs/{PRINTED_FILE}</code>). The
            others are lab 11.1's reference solution (<code>labs/{LAB}/</code>): a mask type per vendor, <code>warpSize</code>{' '}
            in place of 32, and the warp totals declared in the kernel and passed in. On Metal, lane and SIMD-group numbers come
            from kernel attributes, and the last tab does each SIMD-group scan in one call.
            {portable ? null : <> The lab's files are not on this machine yet, so only the first tab is shown.</>}
          </>
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- questions
export const questionsS114: Record<string, Question> = {
  'ch11/11.4/q-sss': {
    kind: 'numeric',
    id: 'ch11/11.4/q-sss',
    prompt: (
      <Translated
        vi={
          <>
            Scan-scan-cộng trên mười sáu giá trị chia thành bốn đoạn: <code>1 2 0 3 | 2 2 1 0 | 4 0 1 1 | 0 3 2 2</code>. Giá
            trị cuối cùng ở vị trí 9 (đếm từ 0) là bao nhiêu?
          </>
        }
      >
        <>
          Scan-scan-add on sixteen values in four segments: <code>1 2 0 3 | 2 2 1 0 | 4 0 1 1 | 0 3 2 2</code>. What is the
          final value at position 9 (counting from 0)?
        </>
      </Translated>
    ),
    answer: 15,
    explain: (
      <Translated
        vi={
          <p>
            Vị trí 9 là phần tử thứ hai của đoạn 2. Giai đoạn 1 scan đoạn đó, 4 0 1 1, thành 4 4 5 6, nên nó chứa 4. Các tổng
            của đoạn là 6, 5, 6 và 7, giai đoạn 2 scan chúng thành 6, 11, 17 và 24. Giai đoạn 3 cộng tổng đã scan của đoạn 1,
            tức 11: 4 + 11 = 15. Một tổng chạy trực tiếp cho cùng kết quả: 1, 3, 3, 6, 8, 10, 11, 11, 15, 15.
          </p>
        }
      >
        <p>
          Position 9 is the second element of segment 2. Stage 1 scans that segment, 4 0 1 1, to 4 4 5 6, so it holds 4. The
          segment totals are 6, 5, 6 and 7, which stage 2 scans to 6, 11, 17 and 24. Stage 3 adds the scanned total of segment
          1, which is 11: 4 + 11 = 15. A direct running sum agrees: 1, 3, 3, 6, 8, 10, 11, 11, 15, 15.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.4, pp. 261–262 (Fig. 11.6); model: src/sim/warpscan.ts',
    vi: { source: 'ch. 11 §11.4, pp. 261–262 (Fig. 11.6); mô hình: src/sim/warpscan.ts' },
  },
  'ch11/11.4/q-below-stride': {
    kind: 'choice',
    id: 'ch11/11.4/q-below-stride',
    prompt: (
      <Translated
        vi={
          <>
            Phép kiểm tra tham gia <code>if(laneIdx() &gt;= stride)</code> bị xóa khỏi Fig. 11.8, nên mọi lane đều chạy{' '}
            <code>val += leftVal</code> ở mọi bước nhảy. Điều gì xảy ra với lane 0 của một warp 32 lane?
          </>
        }
      >
        <>
          The participation test <code>if(laneIdx() &gt;= stride)</code> is deleted from Fig. 11.8, so every lane runs{' '}
          <code>val += leftVal</code> at every stride. What happens to lane 0 of a 32-lane warp?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Nó cộng lại chính giá trị của mình ở mọi bước nhảy và kết thúc ở 32 lần đầu vào.">It adds its own value back at every stride and ends at 32 times its input.</Translated>, correct: true },
      { label: <Translated vi="Không gì cả: lane không có lane nguồn nhận 0, nên nó cộng 0 ở mọi bước nhảy.">Nothing: a lane with no source lane receives 0, so it adds 0 at every stride.</Translated>, why: <Translated vi="Khi đó phép kiểm tra chỉ để làm cảnh. NVIDIA, AMD và Apple đều ghi trong tài liệu điều ngược lại: lane nhận lại chính giá trị của nó.">Then the test would be cosmetic. NVIDIA, AMD and Apple all document the opposite: the lane gets its own value back.</Translated> },
      { label: <Translated vi="Kết quả của nó không xác định, vì nó đọc một lane không tồn tại.">Its result is undefined, because it reads a lane that does not exist.</Translated>, why: <Translated vi={'Trường hợp này đã được định nghĩa: "the lower delta lanes will remain unchanged".'}>The case is defined: "the lower delta lanes will remain unchanged".</Translated> },
      { label: <Translated vi="Lane 0 vẫn ổn; chỉ các lane ở đầu trên của warp bị sai.">Lane 0 is fine; only the lanes at the top of the warp go wrong.</Translated>, why: <Translated vi="Chạy từng bước khi không có phép kiểm tra: các ô đỏ nằm ở đầu dưới của warp.">Step it with no test: the red cells are at the bottom of the warp.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>__shfl_up_sync</code> không quay vòng: một lane có nguồn nằm dưới lane 0 giữ nguyên giá trị của nó. Lane 0
            nằm dưới mọi bước nhảy, nên ở các bước nhảy 1, 2, 4, 8 và 16 nó tự cộng chính mình: 2 mũ 5 là 32. Các lane 1 đến 15
            bị nhân đôi ở các bước nhảy lớn hơn, và các lane phía trên đọc những giá trị đã bị thổi phồng đó; trong mô hình chỉ
            lane 31 kết thúc đúng. Phép kiểm tra là bắt buộc, không phải để làm cảnh. Văn bản của sách không bao giờ nói các
            lane đó nhận được gì; tài liệu của các nhà cung cấp thì có.
          </p>
        }
      >
        <p>
          <code>__shfl_up_sync</code> does not wrap around: a lane whose source would be below lane 0 keeps its own value. Lane 0
          is below every stride, so at strides 1, 2, 4, 8 and 16 it adds itself: 2 to the power 5 is 32. Lanes 1 to 15 double at
          the larger strides, and the lanes above read their inflated values; in the model only lane 31 ends right. The test is
          required, not cosmetic. The book's text never says what those lanes receive; the vendors' documentation does.
        </p>
      </Translated>
    ),
    source: 'CUDA guide 5.4.6 (ch11-facts A3; HIP B3, Metal C1); book Fig. 11.8, p. 264; model: src/sim/warpscan.ts',
    vi: { source: 'CUDA guide 5.4.6 (ch11-facts A3; HIP B3, Metal C1); sách Fig. 11.8, p. 264; mô hình: src/sim/warpscan.ts' },
  },
  'ch11/11.4/q-laneidx': {
    kind: 'choice',
    id: 'ch11/11.4/q-laneidx',
    prompt: (
      <Translated
        vi={
          <>
            Fig. 11.8 được viết với <code>threadIdx.x &gt;= stride</code> thay cho <code>laneIdx() &gt;= stride</code>.
            Những warp nào của một khối 256 luồng cho kết quả scan sai?
          </>
        }
      >
        <>
          Fig. 11.8 is written with <code>threadIdx.x &gt;= stride</code> in place of <code>laneIdx() &gt;= stride</code>.
          Which warps of a 256-thread block get wrong scans?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Chỉ warp 0.">Only warp 0.</Translated>, why: <Translated vi="Trong warp 0, threadIdx.x chính là số lane, nên hai phép kiểm tra là như nhau.">In warp 0, threadIdx.x is the lane number, so the two tests are the same.</Translated> },
      { label: <Translated vi="Các warp 1 đến 7.">Warps 1 to 7.</Translated>, correct: true },
      { label: <Translated vi="Cả tám.">All eight.</Translated>, why: <Translated vi="Hãy xem lại warp 0: threadIdx.x của các lane trong nó là bao nhiêu?">Look at warp 0 again: what is threadIdx.x for its lanes?</Translated> },
      { label: <Translated vi="Không warp nào: shuffle đã biết lane nào có nguồn.">None: the shuffle already knows which lanes have a source.</Translated>, why: <Translated vi="Shuffle trả cho lane không có nguồn chính giá trị của nó; chỉ phép kiểm tra mới ngăn nó cộng giá trị đó vào.">The shuffle hands a lane with no source its own value; only the test stops it adding it.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong warp 1, threadIdx.x chạy từ 32 đến 63, lớn hơn mọi bước nhảy, nên mọi lane đều qua phép kiểm tra. Khi đó các
            lane 0 đến 15 của warp cộng chính giá trị của mình ở những bước nhảy lớn hơn số lane của chúng, và các lane phía
            trên đọc những giá trị bị nhân đôi. Điều tương tự xảy ra với các warp 2 đến 7. Warp 0 không bị ảnh hưởng vì ở đó
            threadIdx.x và số lane trùng nhau, đó là lý do một phép thử chỉ với một warp sẽ không bắt được lỗi. Hãy thử
            "threadIdx.x, trong warp 1" trong trình chạy từng bước.
          </p>
        }
      >
        <p>
          In warp 1 threadIdx.x runs from 32 to 63, larger than every stride, so every lane passes the test. Lanes 0 to 15 of the
          warp then add their own value at the strides above their lane number, and lanes above them read the doubled values.
          The same holds for warps 2 to 7. Warp 0 is unaffected because there threadIdx.x and the lane number coincide, which is
          why a test with a single warp would not catch it. Try "threadIdx.x, in warp 1" in the stepper.
        </p>
      </Translated>
    ),
    source: 'book p. 264 ("local to the warp"); model: src/sim/warpscan.ts, checked on the emulator',
    vi: { source: 'sách p. 264 ("local to the warp"); mô hình: src/sim/warpscan.ts, đã kiểm tra trên trình giả lập' },
  },
  'ch11/11.4/q-barriers': {
    kind: 'numeric',
    id: 'ch11/11.4/q-barriers',
    prompt: (
      <Translated
        vi={
          <>
            Kernel buffer kép của Fig. 11.5 thực thi bao nhiêu lời gọi <code>__syncthreads()</code> cho một lần scan khối
            trên 1,024 phần tử, mỗi luồng một phần tử?
          </>
        }
      >
        <>
          How many <code>__syncthreads()</code> calls does the double-buffered kernel of Fig. 11.5 execute for one block scan
          of 1,024 elements, one per thread?
        </>
      </Translated>
    ),
    answer: 10,
    explain: (
      <Translated
        vi={
          <p>
            Mỗi lần lặp một lời gọi, và vòng lặp chạy với các bước nhảy 1, 2, 4, …, 512: mười lần lặp. Fig. 11.3 cần hai lời
            gọi mỗi lần lặp, tổng cộng 20. Fig. 11.9 cần 2 bất kể kích thước khối, vì phần việc bên trong một warp không cần
            lời gọi nào. Mỗi rào chắn trong số 10 hay 20 rào chắn đó đều chờ cả 1,024 luồng.
          </p>
        }
      >
        <p>
          One per iteration, and the loop runs for strides 1, 2, 4, …, 512: ten iterations. Fig. 11.3 needed two per iteration,
          20 in all. Fig. 11.9 needs 2 whatever the block size, because the work inside a warp needs none. Each of those 10 or 20
          barriers waits for all 1,024 threads.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §§11.2–11.4, pp. 256–265; model: src/sim/warpscan.ts',
    vi: { source: 'ch. 11 §§11.2–11.4, pp. 256–265; mô hình: src/sim/warpscan.ts' },
  },
  'ch11/11.4/q-hip-mask': {
    kind: 'choice',
    id: 'ch11/11.4/q-hip-mask',
    prompt: <Translated vi={<>Fig. 11.8 không biên dịch được dưới HIP. Vì sao, và cách sửa nhỏ nhất nào giữ được một mã nguồn cho cả hai nhà cung cấp?</>}><>Fig. 11.8 fails to compile under HIP. Why, and what is the smallest fix that keeps one source for both vendors?</></Translated>,
    options: [
      { label: <Translated vi="HIP đòi mask 64-bit; hãy cho mask một kiểu riêng theo từng nhà cung cấp.">HIP requires a 64-bit mask; give the mask a type per vendor.</Translated>, correct: true },
      {
        label: <Translated vi="HIP không có __shfl_up_sync; hãy viết lại vòng lặp bằng __shfl_up không có mask.">HIP has no __shfl_up_sync; rewrite the loop with the maskless __shfl_up.</Translated>,
        why: <Translated vi="HIP có hàm này, cùng tên và có tham số mask 64-bit.">HIP has it, with the same name and a 64-bit mask parameter.</Translated>,
      },
      { label: <Translated vi="Chỉ trên wavefront rộng 64, nơi các lane 32 đến 63 thiếu trong mask: hãy viết 0xffffffffffffffff.">Only on 64-wide wavefronts, where lanes 32 to 63 are missing from the mask: write 0xffffffffffffffff.</Translated>, why: <Translated vi="Phép kiểm tra là trên kiểu và áp dụng cho mọi nền tảng đích, kể cả wave32. Trên CUDA hằng số đó không gây lỗi; nó bị cắt ngầm xuống 32 bit, chỉ kèm một cảnh báo, nên một hằng số không thể phục vụ gọn gàng cả hai nhà cung cấp.">The check is on the type and applies on every target, wave32 included. On CUDA the literal does not fail; it is silently truncated to 32 bits with only a warning, so one literal cannot serve both vendors cleanly.</Translated> },
      { label: <Translated vi="Giới hạn vòng lặp WARP_SIZE là 32, sai với wavefront rộng 64.">The loop bound WARP_SIZE is 32, which is wrong for 64-wide wavefronts.</Translated>, why: <Translated vi="Nếu dùng ở mọi chỗ, 32 khiến một wavefront rộng 64 scan thành hai nửa, nửa nào cũng đúng. Và một giới hạn vòng lặp không bao giờ là lỗi biên dịch.">Used everywhere, 32 makes a 64-wide wavefront scan as two halves, each right. And a bound is never a compile error.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Header của HIP đặt <code>static_assert(... sizeof(MaskT) == 8, "The mask must be a 64-bit integer. ...")</code> lên
            mọi hàm <code>_sync</code>, nên hằng số 32-bit bị từ chối trên mọi nền tảng đích. Một kiểu được chọn theo từng nhà
            cung cấp, unsigned trên CUDA và unsigned long long trên HIP, giữ được một tệp duy nhất: lab 10.1 và lab 11.1 gọi
            nó là <code>lane_mask_t</code>; llama.cpp gọi nó là <code>ggml_cuda_thread_mask_t</code>.
          </p>
        }
      >
        <p>
          HIP's header puts <code>static_assert(... sizeof(MaskT) == 8, "The mask must be a 64-bit integer. ...")</code> on every{' '}
          <code>_sync</code> function, so the 32-bit literal is rejected on every target. A type chosen per vendor, unsigned on
          CUDA and unsigned long long on HIP, keeps one file: lab 10.1 and lab 11.1 call it{' '}
          <code>lane_mask_t</code>; llama.cpp calls it <code>ggml_cuda_thread_mask_t</code>.
        </p>
      </Translated>
    ),
    source: 'notes/today/infra-facts.md B3; ch11-facts B1', // i18n-ignore: citation only
  },
  'ch11/11.4/q-half-wave': {
    kind: 'choice',
    id: 'ch11/11.4/q-half-wave',
    prompt: (
      <Translated
        vi={
          <>
            Một bản chuyển sang HIP dùng mask có kiểu và <code>laneIdx() = threadIdx.x % warpSize</code>, nhưng giữ giới hạn
            vòng lặp <code>stride &lt; 32</code>. Trên một wavefront rộng 64, điều gì xảy ra?
          </>
        }
      >
        <>
          A HIP port uses a typed mask and <code>laneIdx() = threadIdx.x % warpSize</code>, but keeps the loop bound{' '}
          <code>stride &lt; 32</code>. On a 64-wide wavefront, what happens?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Cả 64 lane đều đúng.">All 64 lanes are right.</Translated>, why: <Translated vi="Lane 63 cần giá trị của các lane 0 đến 63. Các bước nhảy 1 đến 16 với ngược lại được bao xa?">Lane 63 needs the values of lanes 0 to 63. How far back can strides 1 to 16 reach?</Translated> },
      { label: <Translated vi="Các lane 32 đến 63 bị sai.">Lanes 32 to 63 are wrong.</Translated>, correct: true },
      { label: <Translated vi="Các lane 0 đến 31 bị sai.">Lanes 0 to 31 are wrong.</Translated>, why: <Translated vi="Một lane dưới 32 cần nhiều nhất 31 lane bên trái nó, và các bước nhảy 1 đến 16 với ngược lại được 31.">A lane below 32 needs at most 31 lanes to its left, and strides 1 to 16 reach 31 back.</Translated> },
      { label: <Translated vi="Nó không biên dịch được.">It does not compile.</Translated>, why: <Translated vi="Mask giờ đã có kiểu; mọi thứ còn lại đều là HIP hợp lệ.">The mask is typed now; everything else is legal HIP.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Các bước nhảy 1 đến 16 cộng lại cho tầm với 31 lane. Một lane ở vị trí 32 trở lên, đếm theo <code>warpSize</code>,
            cần nhiều hơn thế, nên nó thiếu các lane ở đầu dưới của wavefront. Đếm mọi thứ theo nhóm 32 sẽ đúng, vì khi đó mỗi
            nửa scan như một warp riêng, nhưng chỉ dùng một nửa tầm với của wavefront và đòi các tổng của warp cũng phải đếm
            theo nhóm 32. Dùng <code>warpSize</code> ở cả hai chỗ là cách sửa. Phần xem phụ của "Chuyển mã" cho thấy cả ba.
          </p>
        }
      >
        <p>
          Strides 1 to 16 add up to a reach of 31 lanes. A lane at position 32 or above, counted from <code>warpSize</code>, needs
          more than that, so it misses the lanes at the bottom of the wavefront. Counting everything in 32s would be right, since
          each half then scans as its own warp, but uses half of the wavefront's reach and requires the warp totals to count in
          32s too. <code>warpSize</code> in both places is the fix. The sub-view of "Port it" shows all three.
        </p>
      </Translated>
    ),
    source: 'ch11-facts B1 (our arithmetic from HIP\'s documented out-of-range rule); model: src/sim/warpscan.ts',
    vi: { source: 'ch11-facts B1 (phép tính của chúng tôi từ quy tắc ngoài phạm vi đã được HIP ghi trong tài liệu); mô hình: src/sim/warpscan.ts' },
  },
  'ch11/11.4/q-metal-width': {
    kind: 'choice',
    id: 'ch11/11.4/q-metal-width',
    prompt: (
      <Translated
        vi={
          <>
            Trên một thiết bị Metal có SIMD group rộng 16, một threadgroup 1,024 luồng chạy thiết kế của Fig. 11.9. Cái gì
            hỏng?
          </>
        }
      >
        <>
          On a Metal device whose SIMD-groups were 16 wide, a 1,024-thread threadgroup runs the design of Fig. 11.9. What
          breaks?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Giai đoạn 2 chỉ scan được 16 trong 64 tổng, nên các SIMD group phía sau cộng những tổng chưa được scan.">Stage 2 can scan only 16 of the 64 totals, so later SIMD-groups add unscanned ones.</Translated>, correct: true },
      {
        label: <Translated vi="Không gì cả: mỗi lần scan SIMD group tự thích ứng với độ rộng nó đọc từ pipeline.">Nothing: each SIMD-group scan adapts to the width it reads from the pipeline.</Translated>,
        why: <Translated vi="Từng lần scan SIMD group thì có. Giai đoạn 2 yêu cầu một SIMD group phủ hết mọi tổng.">Each SIMD-group scan does. Stage 2 asks one SIMD-group to cover every total.</Translated>,
      },
      {
        label: <Translated vi="Nó không biên dịch được: 1,024 luồng cần SIMD group rộng 32, và trình biên dịch biết độ rộng.">It fails to compile: 1,024 threads need SIMD-groups of 32, and the compiler knows the width.</Translated>,
        why: <Translated vi="Độ rộng chỉ được biết khi chạy; trình biên dịch không thể phản đối.">The width is only known when it runs; the compiler cannot object.</Translated>,
      },
      { label: <Translated vi="Chỉ SIMD group cuối cùng bị sai.">Only the last SIMD-group is wrong.</Translated>, why: <Translated vi="Mọi SIMD group từ 17 trở đi đều cộng một tổng mà giai đoạn 2 chưa bao giờ scan.">Every SIMD-group from 17 on adds a total that stage 2 never scanned.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Fig. 11.9 cần (số luồng mỗi threadgroup) ÷ độ rộng ≤ độ rộng, tức độ rộng × độ rộng ≥ số luồng. Ở 1,024 luồng điều
            này đúng với 32 và sai với 16. Apple không ghi độ rộng SIMD group trong tài liệu, và ví dụ định kích thước của chính
            họ vẽ 16. Vì vậy host nên từ chối cấu hình như thế (lab 11.1 kiểm tra nó với <code>threadExecutionWidth</code>),
            hoặc giai đoạn 2 phải lặp. Mọi dự án được đọc cho tệp facts (llama.cpp, MLX, PyTorch) đều đơn giản cố định 32.
          </p>
        }
      >
        <p>
          Fig. 11.9 needs (threads per threadgroup) ÷ width ≤ width, that is width × width ≥ threads. At 1,024 threads it holds
          for 32 and fails for 16. Apple documents no SIMD-group width and its own sizing example draws 16. So the host should
          refuse such a configuration (lab 11.1 checks it against <code>threadExecutionWidth</code>), or stage 2 must loop.
          Every project read for the facts file (llama.cpp, MLX, PyTorch) simply hard-codes 32.
        </p>
      </Translated>
    ),
    source: 'ch11-facts C2; model: src/sim/warpscan.ts',
    vi: { source: 'ch11-facts C2; mô hình: src/sim/warpscan.ts' },
  },
  'ch11/11.4/q-metal-helper': {
    kind: 'choice',
    id: 'ch11/11.4/q-metal-helper',
    prompt: <Translated vi={<>Trong một phiên bản Metal của <code>blockScan</code>, mảng các tổng của SIMD group phải nằm ở đâu?</>}><>Where must the array of SIMD-group totals live in a Metal version of <code>blockScan</code>?</></Translated>,
    options: [
      { label: <Translated vi="Khai báo bên trong hàm phụ trợ là threadgroup float warpSums[32], như Fig. 11.9 làm.">Declared inside the helper as threadgroup float warpSums[32], as Fig. 11.9 does.</Translated>, why: <Translated vi={'MSL từ chối nó: "variables in the threadgroup address space cannot be declared in a non-qualified function".'}>MSL rejects it: "variables in the threadgroup address space cannot be declared in a non-qualified function".</Translated> },
      { label: <Translated vi="Khai báo trong kernel là một mảng threadgroup, và truyền cho hàm phụ trợ dưới dạng threadgroup float*.">Declared in the kernel as a threadgroup array, and passed to the helper as a threadgroup float*.</Translated>, correct: true },
      { label: <Translated vi="Trong một buffer device, vì bộ nhớ threadgroup không thể chia sẻ với hàm phụ trợ.">In a device buffer, since threadgroup memory cannot be shared with a helper.</Translated>, why: <Translated vi="Hàm phụ trợ có thể nhận một con trỏ threadgroup khi được gọi từ kernel. Bộ nhớ device cũng chạy được, nhưng chậm, và không cần thiết.">A helper may take a threadgroup pointer when called from a kernel. Device memory would work, slowly, and is not needed.</Translated> },
      { label: <Translated vi="Trong thanh ghi: simd_prefix_inclusive_sum khiến mảng trở nên thừa.">In registers: simd_prefix_inclusive_sum makes the array unnecessary.</Translated>, why: <Translated vi="Nó scan bên trong một SIMD group. Tổng của các SIMD group khác nhau vẫn cần bộ nhớ mà mọi SIMD group đều thấy được.">It scans within a SIMD-group. The totals of different SIMD-groups still need memory every SIMD-group can see.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            MSL chỉ cho phép biến threadgroup trong hàm kernel (và hàm mesh hoặc object); lỗi của trình biên dịch Metal được
            trích trong tệp facts. Cùng dạng đó, khai báo trong kernel và truyền vào như một con trỏ, là dạng được ghi trong
            tài liệu trên cả ba backend, vì vậy lab 11.1 cũng dùng nó cho CUDA và HIP.
          </p>
        }
      >
        <p>
          MSL allows threadgroup variables only in kernel (and mesh or object) functions; the Metal compiler error is quoted in
          the facts file. The same form, declared in the kernel and passed as a pointer, is the one that is documented on all
          three backends, which is why lab 11.1 uses it for CUDA and HIP too.
        </p>
      </Translated>
    ),
    source: 'ch11-facts C2 (MSL spec §4.4, compiled here); B1 for HIP',
    vi: { source: 'ch11-facts C2 (MSL spec §4.4, đã biên dịch ở đây); B1 cho HIP' },
  },
};

// ---------------------------------------------------------------- the section
export function S114() {
  return (
    <>
      <Translated
        vi={
          <p>
            Hãy đếm xem một phép cộng tốn gì trong kernel buffer kép của <a href={href('ch11', '11.3')}>phần 11.3</a>. Mỗi
            lần lặp có một rào chắn (barrier) mà mọi luồng của khối phải đến được, và mỗi phép cộng thực hiện ba lần truy cập
            bộ nhớ chia sẻ (shared memory): hai lần đọc và một lần ghi. Chỉ cho một phép cộng dấu phẩy động. Kernel dành thời
            gian cho việc đồng bộ hóa và di chuyển dữ liệu, chứ không phải cộng.
          </p>
        }
      >
        <p>
          Count what one addition costs in the double-buffered kernel of <a href={href('ch11', '11.3')}>section 11.3</a>.
          Every iteration has a barrier that all threads of the block must reach, and each addition makes three shared-memory
          accesses: two reads and a write. For one floating-point add. The kernel spends its time synchronising and moving
          data, not adding.
        </p>
      </Translated>
      <Source section="11.4" pages="261" />
      <Translated
        vi={
          <p>
            Chương 10 đã gặp cùng vấn đề này ở phép rút gọn (reduction) và giải quyết nó bằng các primitive mức warp: các luồng
            của một warp có thể trao đổi giá trị qua thanh ghi (register), không cần bộ nhớ chia sẻ và không cần rào chắn khối
            (<a href={href('ch10', '10.7')}>phần 10.7</a>). Scan cũng dùng được chúng, một khi phép scan của khối được cắt
            thành những phần mà một warp tự làm được.
          </p>
        }
      >
        <p>
          Chapter 10 met the same problem in reduction and answered it with warp-level primitives: threads of one warp can
          exchange values through registers, with no shared memory and no block barrier (<a href={href('ch10', '10.7')}>section
          10.7</a>). Scan can use them too, once the block's scan is cut into pieces a warp can do on its own.
        </p>
      </Translated>
      <Source pages="261" />

      <h2><Translated vi="Scan, scan, cộng">Scan, scan, add</Translated></h2>
      <Translated
        vi={
          <p>
            Cách cắt là một phép phân rã ba giai đoạn. Chia đầu vào thành các đoạn. Giai đoạn 1: scan từng đoạn riêng, tất cả
            cùng lúc. Giai đoạn 2: lấy giá trị cuối của mỗi đoạn, tức tổng của nó, và scan các tổng đó. Giai đoạn 3: cộng tổng
            đã scan của đoạn <i>k</i> − 1 vào mọi phần tử của đoạn <i>k</i>. Sau giai đoạn 2, tổng đã scan thứ <i>k</i> là tổng
            của mọi thứ tính đến cuối đoạn <i>k</i>, đúng là phần mà mỗi phần tử của đoạn kế tiếp còn thiếu.
          </p>
        }
      >
        <p>
          The cut is a three-stage decomposition. Split the input into segments. Stage 1: scan every segment on its own, all at
          once. Stage 2: take each segment's last value, which is its total, and scan those totals. Stage 3: add the scanned
          total of segment <i>k</i> − 1 to every element of segment <i>k</i>. After stage 2 the <i>k</i>-th scanned total is
          the sum of everything up to the end of segment <i>k</i>, which is exactly what each element of the next segment is
          missing.
        </p>
      </Translated>
      <Source pages="261–262" figs="Fig. 11.6" />
      <Translated
        vi={
          <p>
            Trình khám phá bắt đầu với mười sáu số của sách. Hãy đổi một đầu vào trong đoạn đầu tiên và xem giá trị nào trong
            các đoạn khác thay đổi, và giai đoạn nào làm chúng thay đổi. Sau đó đổi định nghĩa của một đoạn: các giai đoạn không
            đổi, chỉ có ai làm chúng và các tổng chờ ở đâu là thay đổi.
          </p>
        }
      >
        <p>
          The explorer starts on the book's sixteen numbers. Change one input in the first segment and watch which values in
          the other segments move, and which stage moves them. Then switch what a segment is: the stages do not change, only
          who does them and where the totals wait.
        </p>
      </Translated>
      <ScanScanAddExplorer />
      <Ask q={questionsS114['ch11/11.4/q-sss']} label={tr('Work it out', 'Tự tính ra')} />

      <h2><Translated vi="Một khối, nhiều warp">One block, many warps</Translated></h2>
      <Translated
        vi={
          <p>
            Áp dụng cho một khối, một đoạn là một warp. Mỗi warp scan 32 giá trị của nó; khi đó lane cuối của mỗi warp giữ tổng
            của warp và ghi nó vào một mảng chia sẻ nhỏ, mỗi warp một ô. Một warp scan mảng đó. Mỗi warp trừ warp đầu tiên cộng
            tổng đã scan của warp đứng trước nó vào mọi giá trị của mình. Hai trong các bước này cần cả khối thống nhất: các
            tổng phải được ghi xong hết trước khi warp 0 đọc chúng, và warp 0 phải scan xong chúng trước khi các warp khác đọc.
            Mọi thứ khác diễn ra bên trong một warp.
          </p>
        }
      >
        <p>
          Applied to a block, a segment is a warp. Every warp scans its 32 values; the last lane of each warp then holds the
          warp's total and writes it to a small shared array, one slot per warp. One warp scans that array. Every warp except
          the first adds the scanned total of the warp before it to all of its values. Two of these steps need the whole block
          to agree: the totals must all be written before warp 0 reads them, and warp 0 must have scanned them before the others
          read. Everything else happens inside a warp.
        </p>
      </Translated>
      <Source pages="262–263" figs="Fig. 11.7" />

      <h2><Translated vi="Scan trong warp bằng shuffle">A warp scan with shuffles</Translated></h2>
      <Translated
        vi={
          <p>
            Bên trong một warp, Fig. 11.8 lại chạy Kogge-Stone (<a href={href('ch11', '11.2')}>phần 11.2</a>), với các bước
            nhảy 1, 2, 4, 8 và 16. Giá trị "cách tôi <code>stride</code> vị trí về bên trái" không còn đến từ bộ nhớ chia sẻ:
            mỗi lane yêu cầu nó bằng <code>__shfl_up_sync(0xffffffff, val, stride)</code>, lệnh này đọc thanh ghi của lane cách
            nó <code>stride</code>{' '} vị trí phía dưới. Mọi lane trao đổi trong một lệnh, nên không có bộ nhớ chia sẻ và không có
            rào chắn. Một lane chỉ cộng thứ nó nhận được nếu <code>laneIdx() &gt;= stride</code>. Phép kiểm tra dùng số lane,
            không phải <code>threadIdx.x</code>, vì phép scan là cục bộ trong một warp.
          </p>
        }
      >
        <p>
          Inside a warp, Fig. 11.8 runs Kogge-Stone again (<a href={href('ch11', '11.2')}>section 11.2</a>), strides 1, 2, 4, 8
          and 16. The value "<code>stride</code> positions to my left" no longer comes from shared memory: each lane asks for it
          with <code>__shfl_up_sync(0xffffffff, val, stride)</code>, which reads the register of the lane <code>stride</code>{' '}
          below it. All lanes exchange in one instruction, so there is no shared memory and no barrier. A lane adds what it
          received only if <code>laneIdx() &gt;= stride</code>. The test uses the lane number, not <code>threadIdx.x</code>,
          because the scan is local to one warp.
        </p>
      </Translated>
      <Source pages="263–264" figs="Fig. 11.8" />
      <Translated
        vi={
          <p>
            Một lane nằm dưới bước nhảy nhận được gì? Sách không nói; mã chỉ đơn giản là không dùng nó. Hãy chạy từng bước vòng
            lặp để tìm hiểu, rồi phá phép kiểm tra theo cả hai cách.
          </p>
        }
      >
        <p>
          What does a lane below the stride receive? The book does not say; the code simply does not use it. Step the loop and
          find out, then break the test both ways.
        </p>
      </Translated>
      <NoteGroup>
      <Today checked="2026-11-20 (ch11-facts A3; infra-facts A3, B3, C2)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Cả ba nhà cung cấp đều ghi điều này trong tài liệu: các lane <code>delta</code> phía dưới "remain unchanged",
              tức nhận lại chính giá trị của mình. NVIDIA in cùng vòng lặp này, trên nhóm 8 lane, làm ví dụ của riêng họ, và
              gợi ý dùng <code>cub::WarpScan</code> thay thế.
            </p>
          }
        >
          <p>
            All three vendors document it: the lower <code>delta</code> lanes "remain unchanged", that is, get their own value
            back. NVIDIA prints the same loop, on groups of 8 lanes, as its own example, and suggests{' '}
            <code>cub::WarpScan</code> instead.
          </p>
        </Translated>
      </Today>
      <Simplification>
        <Translated
          vi={
            <p>
              Các lane di chuyển đồng bộ, mỗi bước một bước nhảy, theo quy tắc shuffle đã được ghi trong tài liệu. Không có thời
              gian. Warp 8 lane là đồ chơi: không GPU nào có warp 8 lane.
            </p>
          }
        >
          <p>
            The lanes move in lockstep, one stride per step, by the documented shuffle rule. No timing. The 8-lane warp is a toy:
            no GPU has warps of 8.
          </p>
        </Translated>
      </Simplification>
      </NoteGroup>
      <ShuffleStepper />
      <Ask q={questionsS114['ch11/11.4/q-below-stride']} />
      <Ask q={questionsS114['ch11/11.4/q-laneidx']} />

      <h2><Translated vi="Scan khối, với hai rào chắn">The block scan, with two barriers</Translated></h2>
      <Translated
        vi={
          <p>
            Fig. 11.9 ghép các mảnh lại với nhau. Dòng 03: mỗi warp scan. Dòng 05 đến 08: lane cuối của mỗi warp ghi tổng của
            warp vào <code>warpSums_s[warpIdx()]</code>. Một rào chắn. Dòng 11 đến 17: warp 0 nạp mỗi lane một tổng (các lane
            không có warp tương ứng nạp 0, phần tử đơn vị), scan chúng bằng cùng hàm <code>warpScan</code>, rồi ghi chúng trở
            lại. Rào chắn thứ hai. Dòng 20 đến 22: mỗi warp trừ warp đầu tiên cộng tổng đã scan của warp đứng trước nó. Kernel
            của Fig. 11.10 chỉ nạp, gọi <code>blockScan</code> và ghi ra.
          </p>
        }
      >
        <p>
          Fig. 11.9 puts the pieces together. Line 03: every warp scans. Lines 05 to 08: the last lane of each warp writes the
          warp's total to <code>warpSums_s[warpIdx()]</code>. A barrier. Lines 11 to 17: warp 0 loads one total per lane (lanes
          with no warp behind them load 0, the identity), scans them with the same <code>warpScan</code>, and writes them back.
          A second barrier. Lines 20 to 22: every warp but the first adds the scanned total of the warp before it. The kernel of
          Fig. 11.10 just loads, calls <code>blockScan</code> and stores.
        </p>
      </Translated>
      <Source pages="264–265" figs="Figs. 11.9, 11.10" />
      <BlockScanListing />
      <Translated
        vi={
          <p>
            Hai rào chắn mỗi lần scan khối, so với một rào chắn mỗi lần lặp ở Fig. 11.5, và lưu lượng bộ nhớ chia sẻ duy nhất là
            mảng nhỏ chứa các tổng của warp. Đi kèm là hai quy tắc. Mọi luồng của khối phải gọi <code>blockScan</code>, kể cả
            các luồng vượt quá cuối dữ liệu: nó chứa rào chắn và shuffle với mask đầy đủ, vì vậy Fig. 11.10 nạp 0 cho các luồng
            đó thay vì cho chúng trả về sớm. Và một warp scan mọi tổng của warp, nên một khối có tối đa số warp bằng số lane của
            một warp. Ở 1,024 luồng, đó là 32 warp mỗi warp 32 lane, hoặc 16 wavefront mỗi wavefront 64 lane: điều này đúng với
            cả hai nhà cung cấp.
          </p>
        }
      >
        <p>
          Two barriers per block scan, against one per iteration in Fig. 11.5, and the only shared-memory traffic is the small
          array of warp totals. Two rules come with it. Every thread of the block must call <code>blockScan</code>, including threads past
          the end of the data: it contains barriers and full-mask shuffles, which is why Fig. 11.10 loads 0 for those threads
          instead of returning early. And one warp scans all the warp totals, so a block may have at most as many warps as a
          warp has lanes. At 1,024 threads that is 32 warps of 32 lanes, or 16 wavefronts of 64: it holds on both vendors.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Ở p. 265, câu so sánh hai kernel nhắc Fig. 11.5 hai lần; lần thứ hai phải là Fig. 11.9, kernel có "only two
              occasions". (Notes, mục "Unclear in the PDF".)
            </p>
          }
        >
          <p>
            On p. 265 the sentence contrasting the two kernels names Fig. 11.5 twice; the second should be Fig. 11.9, the one
            with "only two occasions". (Notes, "Unclear in the PDF".)
          </p>
        </Translated>
      </Aside>
      <Ask q={questionsS114['ch11/11.4/q-barriers']} label={tr('Work it out', 'Tự tính ra')} />

      <h2><Translated vi="Chuyển mã sang nền tảng khác">Porting it</Translated></h2>
      <Translated
        vi={
          <p>
            Đến đây, quy tắc của khóa học rằng một tệp <code>.cu</code> phục vụ cả CUDA và HIP không còn đúng. Nó đúng với luồng,
            khối, bộ nhớ chia sẻ, rào chắn, atomic kiểu cũ và các lời gọi runtime. Nó không đúng với các intrinsic warp{' '}
            <code>_sync</code>, cooperative groups và header half-precision, functor của CUB, hay intrinsic matrix-core.
            Fig. 11.8 gặp ngay điều đầu tiên trong số đó.
          </p>
        }
      >
        <p>
          Here the course's rule that one <code>.cu</code> file serves CUDA and HIP stops being true. It holds for threads,
          blocks, shared memory, barriers, legacy atomics and runtime calls. It does not hold for the warp <code>_sync</code>{' '}
          intrinsics, cooperative groups and half-precision headers, CUB functors, or matrix-core intrinsics. Fig. 11.8 meets
          the first of these at once.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ dùng CUDA. Các thông tin về HIP và Metal ở đây lấy từ tài liệu của AMD và Apple (notes/today/ch11-facts.md
              B1, B2, C1, C2; infra-facts.md B3, C2). Chưa có gì được chạy trên phần cứng AMD.
            </p>
          }
        >
          <p>
            The book uses CUDA only. HIP and Metal facts here are from AMD's and Apple's documentation (notes/today/ch11-facts.md
            B1, B2, C1, C2; infra-facts.md B3, C2). Nothing has run on AMD hardware.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Trên HIP, hằng số mask 32-bit không biên dịch được: HIP đòi mask 64-bit trên mọi nền tảng đích. Wavefront rộng 64
            lane trên các thiết bị gfx9 và 32 trên gfx10 trở về sau, nên mã phải đọc <code>warpSize</code>, không bao giờ giả
            định 32. Ở đây mã của sách có một cái bẫy. Với <code>WARP_SIZE</code> 32 dùng ở mọi chỗ, một wavefront rộng 64 hoạt
            động như hai warp 32 và phép scan đúng, với một nửa tầm với. Đổi cách đánh số lane sang <code>warpSize</code> nhưng
            giữ giới hạn vòng lặp ở 32 thì nửa trên bị sai. Và AMD không ghi trong tài liệu việc khai báo một mảng{' '}
            <code>__shared__</code> bên trong một hàm <code>__device__</code>, như Fig. 11.9 làm.
          </p>
        }
      >
        <p>
          On HIP, the 32-bit mask literal does not compile: HIP requires a 64-bit mask on every target. Wavefronts are 64 lanes
          wide on gfx9 devices and 32 on gfx10 and later, so the code must read <code>warpSize</code>, never assume 32. Here the
          book's code has a trap. With <code>WARP_SIZE</code> 32 used everywhere, a 64-wide wavefront behaves as two warps of 32
          and the scan is right, with half the reach. Change the lane numbering to <code>warpSize</code> but keep the loop bound
          at 32 and the upper half goes wrong. And AMD does not document a <code>__shared__</code> array declared inside a{' '}
          <code>__device__</code> function, as Fig. 11.9 does.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Trên Metal, <code>simd_prefix_inclusive_sum(val)</code> làm toàn bộ phép scan warp trong một lời gọi, trên{' '}
            <code>float</code> và <code>half</code>, trên mọi máy Mac Apple silicon (Apple7 trở về sau); Apple không ghi thứ tự
            các phép cộng của nó trong tài liệu, nên các bit cuối có thể khác với vòng lặp. Vòng lặp cũng chuyển được:{' '}
            <code>simd_shuffle_up(val, stride)</code> không có mask và không có độ rộng, và cùng quy tắc cho các lane dưới bước
            nhảy. <code>simd_shuffle_and_fill_up</code> có thể lấp các lane đó bằng phần tử đơn vị và bỏ phép kiểm tra, nhưng
            cần Apple8, nên không chạy trên M1. Hai dòng của Fig. 11.9 không chuyển được: không thể khai báo một mảng threadgroup
            trong hàm phụ trợ (lỗi biên dịch), và SIMD group duy nhất scan các tổng cần một độ rộng mà Apple không ghi trong tài
            liệu.
          </p>
        }
      >
        <p>
          On Metal, <code>simd_prefix_inclusive_sum(val)</code> does the whole warp scan in one call, on <code>float</code> and{' '}
          <code>half</code>, on every Apple silicon Mac (Apple7 and later); Apple does not document the order of its additions,
          so its last bits may differ from the loop's. The loop also ports: <code>simd_shuffle_up(val, stride)</code> has no
          mask and no width, and the same rule for lanes below the stride. <code>simd_shuffle_and_fill_up</code> can fill those
          lanes with the identity and drop the test, but needs Apple8, so not M1. Two lines of Fig. 11.9 do not port: a
          threadgroup array cannot be declared in a helper (a compile error), and the single SIMD-group that scans the totals
          needs a width Apple does not document.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Phần xem phụ chạy Fig. 11.8 trên 64 lane với quy tắc shuffle đã được ghi trong tài liệu và độ rộng mặc định của
              HIP, <code>warpSize</code>. Các lane di chuyển đồng bộ; không có thời gian.
            </p>
          }
        >
          <p>
            The sub-view runs Fig. 11.8 on 64 lanes with the documented shuffle rule and HIP's default width,{' '}
            <code>warpSize</code>. Lanes move in lockstep; no timing.
          </p>
        </Translated>
      </Simplification>
      <PortIt />
      <Ask q={questionsS114['ch11/11.4/q-hip-mask']} />
      <Ask q={questionsS114['ch11/11.4/q-half-wave']} />
      <Ask q={questionsS114['ch11/11.4/q-metal-width']} />
      <Ask q={questionsS114['ch11/11.4/q-metal-helper']} />

      <h2><Translated vi="Hoặc gọi một thư viện">Or call a library</Translated></h2>
      <Translated
        vi={
          <p>
            Trong môi trường production, bạn sẽ hiếm khi tự viết tay phần này. CUDA có <code>cub::WarpScan</code> và{' '}
            <code>cub::BlockScan</code>; tùy chọn <code>BLOCK_SCAN_WARP_SCANS</code> của block scan chính là thiết kế của phần
            này, còn mặc định của nó là "raking reduce-then-scan". Cooperative groups cung cấp <code>cg::inclusive_scan</code>{' '}
            trên một tile. HIP có cùng các tên đó qua hipCUB, và phép scan cooperative-groups của nó trên một tile cỡ wavefront
            dùng các lệnh DPP của AMD, được tăng tốc cho <code>float</code>, từ một header khác (
            <code>cooperative_groups/hip_scan.h</code>), nên một mã nguồn dùng chung cần một <code>#if</code>. Metal không có
            hàm block-scan: các lời gọi SIMD group cộng với bộ nhớ threadgroup, như ở đây.
          </p>
        }
      >
        <p>
          You will rarely write this by hand in production. CUDA has <code>cub::WarpScan</code> and <code>cub::BlockScan</code>;
          the block scan's <code>BLOCK_SCAN_WARP_SCANS</code> option is this section's design, while its default is a "raking
          reduce-then-scan". Cooperative groups offer <code>cg::inclusive_scan</code> on a tile. HIP has the same names through
          hipCUB, and its cooperative-groups scan on a wavefront-sized tile uses AMD's DPP instructions, accelerated for{' '}
          <code>float</code>, from a different header (<code>cooperative_groups/hip_scan.h</code>), so a shared source needs an{' '}
          <code>#if</code>. Metal has no block-scan function: SIMD-group calls plus threadgroup memory, as here.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch11-facts A4, B2)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Tài liệu tham chiếu của CUDA chỉ cho phép <code>cg::inclusive_scan</code> trên <code>coalesced_group</code> và{' '}
              <code>thread_block_tile</code>; ví dụ trong chính hướng dẫn của họ lại truyền cả một khối. Hãy dùng một tile.
            </p>
          }
        >
          <p>
            CUDA's reference allows <code>cg::inclusive_scan</code> only on <code>coalesced_group</code> and{' '}
            <code>thread_block_tile</code>; its guide's own example passes a whole block. Use a tile.
          </p>
        </Translated>
      </Today>
      <div className="wide" style={{ overflowX: 'auto', maxWidth: '58rem' }}>
        <table className="data" style={{ minWidth: '38rem' }} data-backends>
          <thead>
            <tr><th style={{ width: '9rem' }}><Translated vi="Mục">What</Translated></th><th>CUDA</th><th>HIP (ROCm 10.2){/* i18n-ignore: product name */}</th><th>Metal (MSL 4.1){/* i18n-ignore: product name */}</th></tr>
          </thead>
          <tbody>
            <tr><td><Translated vi="Độ rộng cho một lần scan warp">Width for a warp scan</Translated></td><td><Translated vi="32 trên mọi thiết bị được liệt kê">32 on every listed device</Translated></td><td><Translated vi={<>64 trên gfx9, 32 trên gfx10 trở về sau; template hoặc <code>__GFX9__</code> lúc biên dịch</>}>64 on gfx9, 32 on gfx10 and later; template or <code>__GFX9__</code> at compile time</Translated></td><td><Translated vi={<>không có trong tài liệu; đọc <code>[[threads_per_simdgroup]]</code>; các dự án giả định 32</>}>undocumented; read <code>[[threads_per_simdgroup]]</code>; projects assume 32</Translated></td></tr>
            <tr><td><Translated vi="Shuffle lên">Shuffle up</Translated></td><td><Translated vi={<><code>__shfl_up_sync(mask, v, d, width)</code>, mask 32-bit</>}><code>__shfl_up_sync(mask, v, d, width)</code>, 32-bit mask</Translated></td><td><Translated vi="cùng tên; bắt buộc mask 64-bit">same name; 64-bit mask required</Translated></td><td><Translated vi={<><code>simd_shuffle_up(v, d)</code>, không mask, không độ rộng</>}><code>simd_shuffle_up(v, d)</code>, no mask, no width</Translated></td></tr>
            <tr><td><Translated vi="Các lane dưới bước nhảy">Lanes below the stride</Translated></td><td><Translated vi="giữ nguyên giá trị của mình">keep their own value</Translated></td><td><Translated vi="trả về chính giá trị của mình">return their own value</Translated></td><td><Translated vi={<>không đổi; <code>simd_shuffle_and_fill_up</code> lấp bằng phần tử đơn vị (Apple8 trở về sau)</>}>unchanged; <code>simd_shuffle_and_fill_up</code> fills the identity (Apple8 and later)</Translated></td></tr>
            <tr><td><Translated vi="Tổng tiền tố warp trong một lời gọi">One-call warp prefix sum</Translated></td><td><Translated vi={<>không có cho <code>float</code>: dùng vòng lặp, <code>cub::WarpScan</code>, hoặc <code>cg::inclusive_scan</code> trên một tile</>}>none for <code>float</code>: the loop, <code>cub::WarpScan</code>, or <code>cg::inclusive_scan</code> on a tile</Translated></td><td><Translated vi={<><code>cg::inclusive_scan</code> trên một tile wavefront (DPP, tăng tốc cho <code>float</code>); <code>hipcub::WarpScan</code></>}><code>cg::inclusive_scan</code> on a wavefront tile (DPP, <code>float</code> accelerated); <code>hipcub::WarpScan</code></Translated></td><td><Translated vi={<><code>simd_prefix_inclusive_sum</code> / <code>_exclusive_sum</code> trên <code>float</code> và <code>half</code>, Apple7 trở về sau; thứ tự không có trong tài liệu</>}><code>simd_prefix_inclusive_sum</code> / <code>_exclusive_sum</code> on <code>float</code> and <code>half</code>, Apple7 and later; order undocumented</Translated></td></tr>
            <tr><td><Translated vi="Mảng chia sẻ khai báo trong hàm phụ trợ">Shared array declared in a helper</Translated></td><td><Translated vi="được phép">allowed</Translated></td><td><Translated vi="không có trong tài liệu">not documented</Translated></td><td><Translated vi="lỗi biên dịch: khai báo trong kernel, truyền một con trỏ">compile error: declare in the kernel, pass a pointer</Translated></td></tr>
            <tr><td><Translated vi="Block scan từ thư viện">Block scan from a library</Translated></td><td><Translated vi={<><code>cub::BlockScan</code> (mặc định là raking, hoặc warp scan)</>}><code>cub::BlockScan</code> (raking by default, or warp scans)</Translated></td><td><Translated vi={<><code>hipcub::BlockScan</code> dựa trên rocPRIM</>}><code>hipcub::BlockScan</code> over rocPRIM</Translated></td><td><Translated vi="không có">none</Translated></td></tr>
          </tbody>
        </table>
      </div>

      <h2><Translated vi="Nơi bạn gặp điều này: tổng tích lũy trong llama.cpp">Where you meet this: a cumulative sum in llama.cpp</Translated></h2>
      <Translated
        vi={
          <p>
            <code>GGML_OP_CUMSUM</code> của llama.cpp, phép toán đứng sau tổng tích lũy trên một hàng xác suất, chính là thiết
            kế của phần này. Trên CUDA và HIP, <code>warp_prefix_inclusive_sum</code> là vòng lặp của Fig. 11.8 với mask có kiểu
            và độ rộng chọn theo kiến trúc: 64 khi build cho gfx8 và gfx9 của AMD, 32 trong các trường hợp khác. Lane cuối của
            mỗi warp ghi tổng của warp vào bộ nhớ chia sẻ, warp 0 scan các tổng, mỗi luồng cộng phần bù của mình. Trên Metal,
            phép scan warp là một lời gọi <code>simd_prefix_inclusive_sum</code>, dưới một header ghi <code>#define N_SIMDWIDTH 32 // assuming
            SIMD group size is 32</code>. Khi đọc tệp đó, giờ bạn biết dòng nào sẽ hỏng trên một thiết bị rộng 16.
          </p>
        }
      >
        <p>
          llama.cpp's <code>GGML_OP_CUMSUM</code>, the operation behind a cumulative sum over a row of probabilities, is this
          section's design. On CUDA and HIP, <code>warp_prefix_inclusive_sum</code> is Fig. 11.8's loop with a typed mask and a
          width chosen per architecture: 64 when built for AMD's gfx8 and gfx9, 32 otherwise. The last lane of each warp writes
          the warp's total to shared memory, warp 0 scans the totals, every thread adds its offset. On Metal the warp scan is
          one call to <code>simd_prefix_inclusive_sum</code>, under a header that says <code>#define N_SIMDWIDTH 32 // assuming
          SIMD group size is 32</code>. When you read that file, you now know which line would break on a 16-wide device.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              <code>ggml-cuda/cumsum.cu</code>, <code>common.cuh</code> và <code>ggml-metal/kernels/reduce.metal</code> của
              llama.cpp{' '} như được đọc ngày 2026-11-20 (ch11-facts D1).
            </p>
          }
        >
          <p>
            llama.cpp's <code>ggml-cuda/cumsum.cu</code>, <code>common.cuh</code> and <code>ggml-metal/kernels/reduce.metal</code>{' '}
            as read on 2026-11-20 (ch11-facts D1).
          </p>
        </Translated>
      </Beyond>
    </>
  );
}
