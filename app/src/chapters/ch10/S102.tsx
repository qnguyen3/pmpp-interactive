import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  BUILDER_OPS,
  DIST_LABEL,
  FIG103A,
  FIG103B,
  PAIR_RULES,
  SHAPES,
  SHAPE_LABEL,
  builderResult,
  builderTree,
  compareOrders,
  f32hex,
  f32short,
  f64short,
  generate,
  orderIndependent,
  surveyOrders,
  tieRow,
  tournament,
  treeStats,
  winners,
  type BuilderOpId,
  type Dist,
  type PairRule,
  type Shape,
} from '../../sim/reducetree';
import { TreeFoldScene, TreeSvg } from './TreeFoldScene';

import { NoteGroup } from './NoteGroup';
/** A caption above a group of buttons (a <label> would forward clicks on the caption to the first button). */
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
/** Long button groups wrap on a narrow screen instead of running off the bench. */
const WRAP: CSSProperties = { flexWrap: 'wrap' };
const fmt = (x: number) => String(Number(x.toFixed(2)));
const SHAPE_NOTE: Record<Shape, string> = {
  get loop() {
    return tr('Fig. 10.2: one element at a time, from element 0', 'Fig. 10.2: từng phần tử một, bắt đầu từ phần tử 0');
  },
  get neighbours() {
    return tr('Fig. 10.3: adjacent pairs at every level', 'Fig. 10.3: các cặp kề nhau ở mọi tầng');
  },
  get halves() {
    return tr(
      'node j with node j + m/2, the order of section 10.4. The leaves are drawn in the order the tree combines them',
      'nút j với nút j + m/2, thứ tự của mục 10.4. Các lá được vẽ theo thứ tự mà cây kết hợp chúng',
    );
  },
};

function ShapePicker({ shape, set }: { shape: Shape; set: (s: Shape) => void }) {
  return (
    <span style={FIELD}>
      <Translated vi="Thứ tự">Order</Translated>
      <span className="seg" role="group" aria-label={tr('Order', 'Thứ tự')}>
        {SHAPES.map((s) => (
          <button key={s} aria-pressed={shape === s} onClick={() => set(s)}>
            <Translated vi={{ loop: 'vòng lặp', neighbours: 'cặp kề nhau', halves: 'hai nửa' }[s]}>{SHAPE_LABEL[s]}</Translated>
          </button>
        ))}
      </span>
    </span>
  );
}

/** Operations in flight per step, as bars. */
function StepBars({ ops, label, max }: { ops: number[]; label: string; max?: number }) {
  const top = max ?? Math.max(...ops);
  const bw = Math.max(6, Math.min(26, 260 / ops.length));
  const H = 64;
  return (
    <svg viewBox={`0 0 ${ops.length * (bw + 3) + 4} ${H + 16}`} role="img" aria-label={label} data-part="bars" data-ops={ops.join(' ')}
      style={{ width: ops.length * (bw + 3) + 4, maxWidth: '100%', display: 'block', marginTop: '0.4rem' }}>
      {ops.map((k, s) => {
        const h = Math.max(1, (k / top) * H);
        return (
          <g key={s}>
            <rect x={2 + s * (bw + 3)} y={H - h} width={bw} height={h} fill="var(--ink-2)" />
            {ops.length <= 16 ? (
              <text x={2 + s * (bw + 3) + bw / 2} y={H + 12} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{s + 1}</text>
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}

// ---------------------------------------------------------------- interactive 1: the tree builder
const POOL = [...FIG103B, ...FIG103A];

function LeafInput({ value, onValue, k }: { value: number; onValue: (v: number) => void; k: number }) {
  const [text, setText] = useState(String(value));
  useEffect(() => {
    if (Number(text) !== value) setText(String(value));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);
  return (
    <input
      type="number"
      step="1"
      value={text}
      aria-label={tr(`leaf ${k}`, `lá ${k}`)}
      style={{ width: '3.6rem', padding: '0.25rem 0.3rem' }}
      onChange={(e) => {
        setText(e.target.value);
        const v = Number(e.target.value);
        if (e.target.value.trim() !== '' && Number.isFinite(v)) onValue(Math.max(-999, Math.min(999, Math.round(v))));
      }}
    />
  );
}

function TreeBuilder() {
  const [leaves, setLeaves] = useState<number[]>(FIG103B);
  const [op, setOp] = useState<BuilderOpId>('sum');
  const [shape, setShape] = useState<Shape>('neighbours');
  const [logN, setLogN] = useState(10);
  const n = leaves.length;
  const tree = useMemo(() => builderTree(leaves, op, shape), [leaves, op, shape]);
  const loop = useMemo(() => builderResult(leaves, op, 'loop'), [leaves, op]);
  const mine = useMemo(() => builderResult(leaves, op, shape), [leaves, op, shape]);
  const same = mine === loop;
  const stats = treeStats(2 ** logN);
  const resize = (k: number) => setLeaves(Array.from({ length: k }, (_, j) => leaves[j] ?? POOL[j]));
  const dx = n > 8 ? 42 : 56;
  // Rows far enough apart that a leaf's index label clears the first row of operations.
  const dy = shape === 'loop' ? (n > 8 ? 36 : 40) : n > 8 ? 54 : 62;
  const margin = 36;
  const width = 2 * margin + (n - 1) * dx + (shape === 'loop' ? 30 : 0);
  const height = 24 + (tree.levels.length - 1) * dy + 30;
  const peak = Math.max(...tree.opsPerStep);

  return (
    <Bench
      title={tr('Build a reduction tree', 'Dựng một cây rút gọn')}
      task={tr(
        "Edit the leaves, pick an operator and an order. Which operators give the loop's answer in every order?",
        'Sửa các lá, chọn một toán tử và một thứ tự. Những toán tử nào cho đáp án của vòng lặp ở mọi thứ tự?',
      )}
      foot={
        <Translated
          vi={
            <>
              Các số nguyên được kết hợp chính xác, nên mọi sai khác so với vòng lặp chỉ đến từ thứ tự. "join" viết hai toán
              hạng của nó cạnh nhau: nó cho thấy thứ tự mà các lá đã được kết hợp.
            </>
          }
        >
          <>
            Integers are combined exactly, so any disagreement with the loop comes from the order alone. "join" writes its
            two operands side by side: it shows the order in which the leaves were combined.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Số lá">Leaves</Translated>
          <span className="seg" role="group" aria-label={tr('Leaves', 'Số lá')}>
            {[4, 8, 16].map((k) => (
              <button key={k} aria-pressed={n === k} onClick={() => resize(k)}>
                {k}
              </button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          <Translated vi="Toán tử">Operator</Translated>
          <span className="seg" role="group" aria-label={tr('Operator', 'Toán tử')} style={WRAP}>
            {BUILDER_OPS.map((o) => (
              <button key={o.id} aria-pressed={op === o.id} onClick={() => setOp(o.id)}>
                <Translated
                  vi={{ sum: 'tổng', max: 'max', min: 'min', product: 'tích', sub: 'phép trừ', join: 'join' }[o.id]}
                >
                  {o.label}
                </Translated>
              </button>
            ))}
          </span>
        </span>
        <ShapePicker shape={shape} set={setShape} />
        <span style={FIELD}>
          <Translated vi="Ví dụ của sách">Book examples</Translated>
          <span className="seg" role="group" aria-label={tr('Book examples', 'Ví dụ của sách')} style={WRAP}>
            <button onClick={() => { setLeaves(FIG103B); setOp('sum'); setShape('neighbours'); }}>
              <Translated vi="Hình 10.3(b), tổng">Fig. 10.3(b), sum</Translated>
            </button>
            <button onClick={() => { setLeaves(FIG103A); setOp('max'); setShape('neighbours'); }}>
              <Translated vi="Hình 10.3(a), max">Fig. 10.3(a), max</Translated>
            </button>
          </span>
        </span>
      </div>
      <div className="row" role="group" aria-label={tr('Leaf values', 'Giá trị các lá')} style={{ gap: '0.3rem', marginBottom: '0.6rem' }}>
        {leaves.map((v, k) => (
          <LeafInput key={k} k={k} value={v} onValue={(nv) => setLeaves(leaves.map((x, j) => (j === k ? nv : x)))} />
        ))}
      </div>
      <p className="muted" style={{ margin: '0 0 0.4rem' }}>
        <Translated
          vi={
            <>
              {{ loop: 'vòng lặp', neighbours: 'cặp kề nhau', halves: 'hai nửa' }[shape]}: {SHAPE_NOTE[shape]}. Thời gian
              chảy từ trên xuống, mỗi hàng là một bước.
            </>
          }
        >
          {SHAPE_LABEL[shape]}: {SHAPE_NOTE[shape]}. Time flows down, one row per step.
        </Translated>
      </p>
      <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={tr('The reduction tree', 'Cây rút gọn')} data-part="builder-tree"
          style={{ width: '100%', minWidth: Math.min(width, n * 30), maxWidth: width * 1.15, display: 'block' }}>
          <TreeSvg tree={tree} x={margin} y={6} dx={dx} dy={dy} w={n > 8 ? 34 : 40} maxChars={8} name="builder" // i18n-ignore: internal key
            under={(k) => `x${k}`} operandOrder={shape === 'halves'} />
        </svg>
      </div>
      <div className="cols" style={{ marginTop: '0.8rem' }}>
        <div className="panel" aria-live="polite" data-readout="result" data-same={same ? '1' : '0'}>
          <p style={{ marginTop: 0, overflowWrap: 'anywhere' }}>
            <code data-part="expr">{tree.expr}</code>
          </p>
          <p style={{ marginBottom: 0, overflowWrap: 'anywhere' }}>
            <Translated
              vi={
                <>
                  Thứ tự này: <b data-value="mine">{tree.root.value}</b>. Vòng lặp: <b data-value="loop">{loopDisplay(leaves, op)}</b>.{' '}
                  {same ? <span className="chip ok">giống vòng lặp</span> : <span className="chip fail">khác vòng lặp</span>}
                </>
              }
            >
              This order: <b data-value="mine">{tree.root.value}</b>. The loop: <b data-value="loop">{loopDisplay(leaves, op)}</b>.{' '}
              {same ? <span className="chip ok">same as the loop</span> : <span className="chip fail">differs from the loop</span>}
            </Translated>
          </p>
        </div>
        <div className="panel" data-readout="counters">
          <p style={{ margin: 0 }}>
            <Translated
              vi={
                <>
                  Khối lượng công việc (work) <b data-counter="work">{tree.work}</b> phép toán. Độ dài (span){' '}
                  <b data-counter="span">{tree.span}</b> bước. Tối đa <b data-counter="peak">{peak}</b> phép cùng lúc.
                </>
              }
            >
              Work <b data-counter="work">{tree.work}</b> operations. Span <b data-counter="span">{tree.span}</b> steps. At
              most <b data-counter="peak">{peak}</b> at once.
            </Translated>
          </p>
          <StepBars ops={tree.opsPerStep} label={tr('Operations in each step', 'Số phép toán ở mỗi bước')} />
          <p className="muted" style={{ margin: '0.2rem 0 0', fontSize: '0.85em' }}>
            <Translated vi="số phép toán ở mỗi bước">operations in each step</Translated>
          </p>
        </div>
      </div>
      <div className="panel" style={{ marginTop: '1rem' }} data-readout="scale">
        <label className="field" style={{ width: '100%' }}>
          <span>
            <Translated
              vi={
                <>
                  Chỉ các bộ đếm, cho một cây trên N = <b data-stat="n">{stats.n}</b> đầu vào
                </>
              }
            >
              Counters only, for a tree over N = <b data-stat="n">{stats.n}</b> inputs
            </Translated>
          </span>
          <input type="range" min={1} max={10} value={logN} aria-label={tr('N for the counters', 'N cho các bộ đếm')} onChange={(e) => setLogN(Number(e.target.value))} style={{ width: '100%', maxWidth: '24rem' }} />
        </label>
        <p style={{ margin: '0.5rem 0 0' }}>
          <Translated
            vi={
              <>
                Work N − 1 = <b data-stat="work">{stats.work}</b>. Span log₂ N = <b data-stat="span">{stats.span}</b> bước.
                Đỉnh N/2 = <b data-stat="peak">{stats.peak}</b> phép toán cùng lúc. Trung bình (N − 1)/log₂ N ={' '}
                <b data-stat="average">{fmt(stats.average)}</b>.
              </>
            }
          >
            Work N − 1 = <b data-stat="work">{stats.work}</b>. Span log₂ N = <b data-stat="span">{stats.span}</b> steps.
            Peak N/2 = <b data-stat="peak">{stats.peak}</b> operations at once. Average (N − 1)/log₂ N ={' '}
            <b data-stat="average">{fmt(stats.average)}</b>.
          </Translated>
        </p>
        <StepBars ops={stats.opsPerStep} label={tr('Operations in each step for N inputs', 'Số phép toán ở mỗi bước cho N đầu vào')} />
      </div>
    </Bench>
  );
}
const loopDisplay = (leaves: number[], op: BuilderOpId) => builderTree(leaves, op, 'loop').root.value;

// ---------------------------------------------------------------- interactive 2: three orders, one float32 sum
const DISTS: Dist[] = ['uniform', 'softmax', 'wide'];
const trialsFor = (n: number) => (n >= 16384 ? 10 : n >= 4096 ? 30 : 100);
const DEFAULT_ORDERS = compareOrders(generate(1024, 'softmax', 1));
const DEFAULT_SURVEY = surveyOrders(1024, 'softmax', trialsFor(1024));

function OrdersBench() {
  const [logN, setLogN] = useState(10);
  const [dist, setDist] = useState<Dist>('softmax');
  const [seed, setSeed] = useState(1);
  const n = 2 ** logN;
  const values = useMemo(() => generate(n, dist, seed), [n, dist, seed]);
  const r = useMemo(() => compareOrders(values), [values]);
  const trials = trialsFor(n);
  const survey = useMemo(() => surveyOrders(n, dist, trials), [n, dist, trials]);
  return (
    <Bench
      title={tr('Three orders, one float32 sum', 'Ba thứ tự, một tổng float32')}
      task={tr(
        'Change the size, the kind of data and the seed. When do the orders agree, and which is closest to the exact sum?',
        'Thay đổi kích thước, loại dữ liệu và seed. Khi nào các thứ tự đồng ý với nhau, và thứ tự nào gần tổng chính xác nhất?',
      )}
      foot={
        <Translated
          vi={
            <>
              Các giá trị được sinh dưới dạng float32. Mỗi thứ tự cộng chúng bằng float32, làm tròn sau mỗi phép cộng. Tham
              chiếu là chính các giá trị đó cộng bằng float64 với phép tổng bù sai số (compensated summation), chính xác hơn
              nhiều so với bất kỳ thứ tự float32 nào. Khoảng cách ULP đếm số giá trị float32 nằm giữa một tổng và giá trị
              float32 gần tham chiếu nhất.
            </>
          }
        >
          <>
            Values are generated as float32. Each order adds them in float32, rounding after every addition. The reference
            is the same values added in float64 with compensated summation, far more accurate than any float32 order. ULP
            distance counts the float32 values between a sum and the float32 nearest the reference.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <span>
            <Translated
              vi={
                <>
                  N = <b data-value="n">{n}</b> giá trị
                </>
              }
            >
              N = <b data-value="n">{n}</b> values
            </Translated>
          </span>
          <input type="range" min={3} max={16} value={logN} aria-label={tr('Number of values', 'Số giá trị')} onChange={(e) => setLogN(Number(e.target.value))} style={{ width: '14rem' }} />
        </label>
        <span style={FIELD}>
          <Translated vi="Dữ liệu">Data</Translated>
          <span className="seg" role="group" aria-label={tr('Data', 'Dữ liệu')} style={WRAP}>
            {DISTS.map((d) => (
              <button key={d} aria-pressed={dist === d} onClick={() => setDist(d)}>
                <Translated
                  vi={
                    {
                      uniform: 'phân bố đều trong [−1, 1]',
                      softmax: 'các số hạng softmax, một logit lớn',
                      wide: 'hỗn hợp dải rộng',
                    }[d]
                  }
                >
                  {DIST_LABEL[d]}
                </Translated>
              </button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          <Translated vi={<>Seed {seed}</>}>Seed {seed}</Translated>
          <button onClick={() => setSeed(seed + 1)}>
            <Translated vi="Dữ liệu mới">New data</Translated>
          </button>
        </span>
      </div>
      <p className="muted" style={{ marginTop: 0, overflowWrap: 'anywhere' }}>
        <Translated vi={<>Các giá trị đầu: {values.slice(0, 6).map(f32short).join(', ')}, …</>}>
          First values: {values.slice(0, 6).map(f32short).join(', ')}, …
        </Translated>
      </p>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" data-part="orders">
          <thead>
            <tr>
              <th>
                <Translated vi="Thứ tự">Order</Translated>
              </th>
              <th className="num">
                <Translated vi="tổng float32">float32 sum</Translated>
              </th>
              <th className="num">
                <Translated vi="các bit">bits</Translated>
              </th>
              <th className="num">
                <Translated vi="Số ULP cách tham chiếu">ULPs from the reference</Translated>
              </th>
            </tr>
          </thead>
          <tbody>
            {SHAPES.map((s) => (
              <tr key={s} data-order={s} data-hex={f32hex(r.sums[s])} data-ulps={r.ulps[s]}>
                <td>
                  <Translated vi={{ loop: 'vòng lặp', neighbours: 'cặp kề nhau', halves: 'hai nửa' }[s]}>{SHAPE_LABEL[s]}</Translated>
                  {r.closest.includes(s) ? (
                    <span className="chip ok" style={{ marginLeft: 6 }}>
                      <Translated vi="gần nhất">closest</Translated>
                    </span>
                  ) : null}
                </td>
                <td className="num">{f32short(r.sums[s])}</td>
                <td className="num mono">{f32hex(r.sums[s])}</td>
                <td className="num">{r.ulps[s]}</td>
              </tr>
            ))}
            <tr data-order="reference">
              <td>
                <Translated vi="tham chiếu float64">float64 reference</Translated>
              </td>
              <td className="num">{f64short(r.ref)}</td>
              <td className="num muted">
                <Translated vi={<>float32 gần nhất {f32hex(r.ref)}</>}>nearest float32 {f32hex(r.ref)}</Translated>
              </td>
              <td className="num">0</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p aria-live="polite" data-readout="distinct" data-distinct={r.distinct} style={{ marginBottom: '0.4rem' }}>
        <b>{r.distinct === 1 ? tr('One result:', 'Một kết quả:') : tr(`${r.distinct} different results.`, `${r.distinct} kết quả khác nhau.`)}</b>{' '}
        {r.distinct === 1
          ? tr('all three orders agree, bit for bit, on this data.', 'cả ba thứ tự đồng ý với nhau, từng bit một, trên dữ liệu này.')
          : tr('The same numbers, added in different groupings.', 'Cùng những con số, được cộng theo những cách nhóm khác nhau.')}
      </p>
      <p className="muted" style={{ marginTop: 0 }} data-readout="survey">
        <Translated
          vi={
            <>
              Trên {survey.trials} bộ dữ liệu cùng loại và kích thước này (seed 1 đến {survey.trials}): các thứ tự bất đồng ở{' '}
              <b>{survey.disagree}</b> bộ. Gần tham chiếu nhất (hòa thì tính cho mỗi bên): vòng lặp ở {survey.closest.loop},
              cặp kề nhau ở {survey.closest.neighbours}, hai nửa ở {survey.closest.halves}.
            </>
          }
        >
          Over {survey.trials} data sets of this kind and size (seeds 1 to {survey.trials}): the orders disagree in{' '}
          <b>{survey.disagree}</b>. Closest to the reference (ties count for each): the loop in {survey.closest.loop},
          neighbours in {survey.closest.neighbours}, halves in {survey.closest.halves}.
        </Translated>
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive 3: who wins a tie
const ARG_RULES: PairRule[] = ['lower', 'left', 'higher', 'nanfirst'];
const MAX_RULES: PairRule[] = ['fmax', 'cmp'];

function TieBench() {
  const [n, setN] = useState(8);
  const [maxima, setMaxima] = useState<number[]>([1, 4]);
  const [nanOn, setNanOn] = useState(false);
  const [nanAt, setNanAt] = useState(6);
  const [rule, setRule] = useState<PairRule>('left');
  const row = useMemo(() => tieRow(n, maxima, nanOn ? nanAt : null), [n, maxima, nanOn, nanAt]);
  const w = useMemo(() => winners(row, rule), [row, rule]);
  const indep = orderIndependent(row, rule);
  const info = PAIR_RULES[rule];
  const resize = (k: number) => {
    setN(k);
    const m = maxima.filter((x) => x < k);
    setMaxima(m.length === 2 ? m : [1, 4]);
    if (nanAt >= k) setNanAt(k - 2);
  };
  const show = (v: number) => (Number.isNaN(v) ? 'NaN' : f32short(v));
  const radio = (r: PairRule) => (
    <label key={r} className="check" style={{ display: 'flex' }}>
      <input type="radio" name="tie-rule" checked={rule === r} onChange={() => setRule(r)} />{' '}{/* i18n-ignore: form group name */}
      <Translated
        vi={
          {
            lower: 'giá trị lớn hơn; hòa thì lấy chỉ số nhỏ hơn',
            left: 'giá trị lớn hơn; hòa thì giữ toán hạng bên trái',
            higher: 'giá trị lớn hơn; hòa thì lấy chỉ số lớn hơn',
            nanfirst: 'NaN thắng, rồi giá trị lớn hơn; hòa thì lấy chỉ số nhỏ hơn',
            fmax: 'fmaxf (NaN coi như thiếu)',
            cmp: 'a > b ? a : b',
          }[r]
        }
      >
        {PAIR_RULES[r].label}
      </Translated>
    </label>
  );
  return (
    <Bench
      title={tr('Who wins a tie', 'Ai thắng khi hòa')}
      task={tr(
        'Pick a combine rule and compare the winner under the three orders. Then move a maximum, or add a NaN.',
        'Chọn một quy tắc kết hợp và so sánh bên thắng dưới ba thứ tự. Rồi di chuyển một giá trị cực đại, hoặc thêm một NaN.',
      )}
      foot={
        <Translated
          vi={
            <>
              Mỗi thứ tự là một cây trên các cặp (giá trị, chỉ số), kết hợp theo combine(left, right). Bấm vào một ô để chuyển
              giá trị cực đại cũ hơn trong hai giá trị cực đại tới đó.
            </>
          }
        >
          <>
            Each order is one tree over (value, index) pairs, combined as combine(left, right). Click a cell to move the
            older of the two maxima there.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Logits">Logits</Translated>
          <span className="seg" role="group" aria-label={tr('Logits', 'Logits')}>
            {[8, 16].map((k) => (
              <button key={k} aria-pressed={n === k} onClick={() => resize(k)}>
                {k}
              </button>
            ))}
          </span>
        </span>
        <label className="check">
          <input type="checkbox" checked={nanOn} onChange={(e) => setNanOn(e.target.checked)} />{' '}
          <Translated vi="đặt một NaN tại chỉ số">put a NaN at index</Translated>
        </label>
        <select value={nanAt} aria-label={tr('NaN index', 'Chỉ số của NaN')} disabled={!nanOn} onChange={(e) => setNanAt(Number(e.target.value))}>
          {Array.from({ length: n }, (_, k) => (
            <option key={k} value={k}>
              {k}
            </option>
          ))}
        </select>
      </div>
      <div role="group" aria-label={tr('Row of logits', 'Hàng logits')} style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginBottom: '0.9rem' }}>
        {row.map((v, k) => {
          const isMax = maxima.includes(k) && !Number.isNaN(v);
          return (
            <button
              key={k}
              data-cell={k}
              data-max={isMax ? 1 : 0}
              aria-label={tr(`logit ${k}: ${show(v)}`, `logit ${k}: ${show(v)}`)}
              onClick={() => !maxima.includes(k) && setMaxima([maxima[1], k])}
              style={{
                width: 52,
                padding: '0.2rem 0',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                fontVariantNumeric: 'tabular-nums',
                border: isMax ? '2px solid var(--ink)' : '1px solid var(--rule)',
                background: 'var(--cell)',
                color: 'var(--ink)',
                fontWeight: isMax ? 700 : 450,
              }}
            >
              <span>{show(v)}</span>
              <span className="muted" style={{ fontSize: '0.75em' }}>#{k}</span>
            </button>
          );
        })}
      </div>
      <div className="cols">
        <div role="radiogroup" aria-label={tr('Combine rule', 'Quy tắc kết hợp')}>
          <p style={{ margin: '0 0 0.3rem' }} className="muted">
            <Translated vi="argmax: chỉ số có ý nghĩa">argmax: the index matters</Translated>
          </p>
          {ARG_RULES.map(radio)}
          <p style={{ margin: '0.6rem 0 0.3rem' }} className="muted">
            <Translated vi="max thường: chỉ giá trị có ý nghĩa">plain max: only the value matters</Translated>
          </p>
          {MAX_RULES.map(radio)}
        </div>
        <div className="panel" aria-live="polite" data-readout="winners" data-independent={indep ? '1' : '0'}>
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              {SHAPES.map((s) => (
                <tr key={s} data-order={s} data-index={w[s].i} data-value={show(w[s].v)}>
                  <td>
                    <Translated vi={{ loop: 'vòng lặp', neighbours: 'cặp kề nhau', halves: 'hai nửa' }[s]}>{SHAPE_LABEL[s]}</Translated>
                  </td>
                  <td className="num">
                    {info.argmax
                      ? tr(`index ${w[s].i} (value ${show(w[s].v)})`, `chỉ số ${w[s].i} (giá trị ${show(w[s].v)})`)
                      : tr(`value ${show(w[s].v)}`, `giá trị ${show(w[s].v)}`)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p style={{ margin: '0.6rem 0 0' }}>
            {indep ? (
              <span className="chip ok">
                <Translated vi="như nhau ở mọi thứ tự">the same in every order</Translated>
              </span>
            ) : (
              <span className="chip fail">
                <Translated vi="phụ thuộc vào thứ tự">depends on the order</Translated>
              </span>
            )}
          </p>
          <p className="muted" style={{ margin: '0.4rem 0 0' }} data-readout="rule-note">
            <Translated
              vi={
                {
                  lower: 'functor ArgMax của hipCUB, viết bằng các phép so sánh (ch10 facts B5)',
                  left: 'một bước "chỉ thay khi lớn hơn hẳn", dùng bên trong một cây',
                  higher: 'nhất quán, nhưng không phải chỉ số mà PyTorch ghi trong tài liệu',
                  nanfirst: 'GreaterOrNan của PyTorch và ArgMax của MLX (ch10 facts D3, D4)',
                  fmax: 'fmaxf và max(float, float) của CUDA (A10)',
                  cmp: 'một phép so sánh, như nhiều kernel viết tay vẫn làm (A10)',
                }[rule]
              }
            >
              {info.note}
            </Translated>
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const AVG4096 = treeStats(4096);
const VENUES = tournament(8, 2);

export const questionsS102: Record<string, Question> = {
  'ch10/10.2/q-avg-4096': {
    kind: 'numeric',
    id: 'ch10/10.2/q-avg-4096',
    prompt: (
      <Translated
        vi={<>Một cây rút gọn trên 4096 phần tử. Mức song song trung bình của nó là bao nhiêu: trung bình số phép toán mỗi bước?</>}
      >
        <>A reduction tree over 4096 elements. What is its average parallelism: operations per step, on average?</>
      </Translated>
    ),
    answer: AVG4096.average,
    tol: 0.01,
    unit: 'operations per step',
    explain: (
      <Translated
        vi={
          <p>
            4095 phép toán trong log₂ 4096 = 12 bước: {AVG4096.average}. Bước đầu chạy {AVG4096.peak} phép toán cùng lúc. Phần
            cứng có kích thước cho mức đỉnh đó, tính trung bình, chỉ được dùng một phần sáu.
          </p>
        }
      >
        <p>
          4095 operations in log₂ 4096 = 12 steps: {AVG4096.average}. The first step runs {AVG4096.peak} operations at
          once. Hardware sized for that peak is used, on average, at a sixth of it.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.2, pp. 225–226',
    vi: { unit: 'phép toán mỗi bước' },
  },
  'ch10/10.2/q-subtract': {
    kind: 'choice',
    id: 'ch10/10.2/q-subtract',
    prompt: (
      <Translated vi={<>Một cây song song có thể tính x0 − x1 − x2 − … − x7, với đáp án mà vòng lặp từ trái sang phải cho, không?</>}>
        <>Can a parallel tree compute x0 − x1 − x2 − … − x7, with the answer the left-to-right loop gives?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không. Phép trừ không có tính kết hợp, nên không cây nào tính được nó.">
            No. Subtraction is not associative, so no tree can compute it.
          </Translated>
        ),
        why: (
          <Translated vi="Hãy đổi toán tử, không phải đổi cây: cùng đại lượng đó là một tổng.">
            Change the operator, not the tree: the same quantity is a sum.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Có, sau khi viết lại thành x0 + (−x1) + … + (−x7) và cộng bằng cây bất kỳ.">
            Yes, after rewriting it as x0 + (−x1) + … + (−x7) and summing with any tree.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Có, cây phép trừ nào cũng cho cùng đáp án, miễn là các lá giữ nguyên thứ tự.">
            Yes, any tree of subtractions gives the same answer, as long as the leaves stay in order.
          </Translated>
        ),
        why: (
          <Translated vi="Thử trong trình dựng cây: trên các lá của Fig. 10.3(b), cây cặp kề nhau cho −2 còn vòng lặp cho −36.">
            Try it in the builder: on Fig. 10.3(b)'s leaves the neighbours tree gives −2 and the loop −36.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Cây cần một toán tử có tính kết hợp, và phép trừ không phải: (1 − 2) − 3 ≠ 1 − (2 − 3). Nhưng trừ chính là cộng với
            số đối, và phép cộng có tính kết hợp. Đổi dấu mọi phần tử trừ phần tử đầu, rồi cộng theo thứ tự bất kỳ.
          </p>
        }
      >
        <p>
          The tree needs an associative operator, and subtraction is not one: (1 − 2) − 3 ≠ 1 − (2 − 3). But subtracting
          is adding the negation, and addition is associative. Negate every element but the first, then sum in any order.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.2, p. 224',
  },
  'ch10/10.2/q-commute': {
    kind: 'choice',
    id: 'ch10/10.2/q-commute',
    prompt: (
      <Translated vi={<>Thứ tự nào cần một toán tử có tính giao hoán, chứ không chỉ có tính kết hợp?</>}>
        <>Which order needs a commutative operator, not merely an associative one?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="cặp kề nhau: ((x0 x1)(x2 x3))((x4 x5)(x6 x7)), các lá kề nhau được ghép cặp trước">
            neighbours: ((x0 x1)(x2 x3))((x4 x5)(x6 x7)), adjacent leaves paired first
          </Translated>
        ),
        why: (
          <Translated vi="Các lá giữ nguyên trình tự ban đầu. Chỉ có dấu ngoặc là dịch chuyển.">
            The leaves stay in their original sequence. Only the parentheses moved.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="hai nửa: x0 với x4, x1 với x5, rồi các cặp của chúng">
            halves: x0 with x4, x1 with x5, then pairs of those
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="vòng lặp, từng phần tử một">the loop, one element at a time</Translated>,
        why: (
          <Translated vi="Vòng lặp là thứ tự tham chiếu: nó không nhóm lại cũng không sắp xếp lại.">
            The loop is the reference order: it neither regroups nor reorders.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đọc các lá của cây hai nửa từ trái sang phải: x0, x4, x2, x6, x1, x5, x3, x7. Các toán hạng không còn theo trình tự
            ban đầu, nên kết quả bằng kết quả của vòng lặp chỉ khi việc đổi chỗ các toán hạng không thay đổi gì. Toán tử "join"
            trong trình dựng cây cho thấy điều đó: có tính kết hợp nhưng không giao hoán, nó khớp với vòng lặp dưới thứ tự cặp
            kề nhau và không khớp dưới thứ tự hai nửa. Kernel của mục 10.4 dùng thứ tự hai nửa.
          </p>
        }
      >
        <p>
          Read the leaves of the halves tree from left to right: x0, x4, x2, x6, x1, x5, x3, x7. The operands are no
          longer in their original sequence, so the result equals the loop's only if swapping operands changes nothing. The
          "join" operator in the builder shows it: associative but not commutative, it agrees with the loop under
          neighbours and not under halves. Section 10.4's kernel uses the halves order.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.2, p. 225; §10.4, pp. 232–233', // i18n-ignore: citation only
  },
  'ch10/10.2/q-venues': {
    kind: 'numeric',
    id: 'ch10/10.2/q-venues',
    prompt: (
      <Translated
        vi={
          <>
            Một giải đấu loại trực tiếp với 8 đội chỉ có 2 sân, mỗi sân tổ chức được một trận mỗi ngày, và một vòng phải kết
            thúc trước khi vòng sau bắt đầu. Giải mất bao nhiêu ngày?
          </>
        }
      >
        <>
          A knockout tournament with 8 teams has only 2 venues, each able to host one game a day, and a round must finish
          before the next begins. How many days does it take?
        </>
      </Translated>
    ),
    answer: VENUES.days,
    unit: 'days',
    explain: (
      <Translated
        vi={
          <p>
            Tứ kết: 4 trận trên 2 sân, 2 ngày. Bán kết: 1 ngày. Chung kết: 1 ngày. Tổng cộng {VENUES.days}, so với 3 khi có 4
            sân. Chia sẻ một tài nguyên giữa các phép toán làm phép rút gọn dài ra; mục 10.10 cố ý thực hiện đúng sự đánh đổi
            đó.
          </p>
        }
      >
        <p>
          Quarter-finals: 4 games on 2 venues, 2 days. Semi-finals: 1 day. Final: 1 day. Total {VENUES.days}, against 3
          with 4 venues. Sharing a resource between operations lengthens the reduction; section 10.10 makes the same
          trade on purpose.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.2 sidebar, pp. 226–227 (the numbers are ours)',
    vi: { unit: 'ngày', source: 'ch. 10 §10.2 khung bên, pp. 226–227 (các con số là của khóa học này)' },
  },
  'ch10/10.2/q-argmax-rule': {
    kind: 'choice',
    id: 'ch10/10.2/q-argmax-rule',
    prompt: (
      <Translated
        vi={
          <>
            Một argmax trên GPU kết hợp các cặp (giá trị, chỉ số) trong một cây. Quy tắc kết hợp nào khiến đáp án của nó không
            phụ thuộc vào thứ tự mà cây dùng?
          </>
        }
      >
        <>A GPU argmax combines (value, index) pairs in a tree. Which combine rule makes its answer independent of the order the tree uses?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Giá trị lớn hơn thắng; các giá trị bằng nhau thuộc về chỉ số nhỏ hơn.">
            The larger value wins; equal values go to the lower index.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Giá trị lớn hơn thắng; khi bằng nhau thì giữ toán hạng bên trái.">
            The larger value wins; on equal values keep the left operand.
          </Translated>
        ),
        why: (
          <Translated vi="Toán hạng nào nằm bên trái phụ thuộc vào cây. Trình khám phá cho thấy bên thắng dịch chuyển.">
            Which operand is on the left depends on the tree. The explorer shows the winner moving.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Toán hạng bên phải chỉ thay toán hạng bên trái nếu giá trị của nó lớn hơn hẳn.">
            The right operand replaces the left only if its value is strictly greater.
          </Translated>
        ),
        why: (
          <Translated vi="Nói cách khác, đó cũng là quy tắc giữ toán hạng bên trái.">
            The same rule as keeping the left operand, in other words.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ so sánh chỉ số khi cả hai giá trị đều là NaN.">Compare indices only when both values are NaN.</Translated>
        ),
        why: (
          <Translated vi="Cách đó giải quyết một trường hợp góc và để các trường hợp hòa thông thường cho toán hạng nào tình cờ nằm bên trái.">
            That settles a corner and leaves ordinary ties to whichever operand is on the left.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Sắp thứ tự các cặp theo giá trị rồi theo chỉ số nhỏ hơn là một thứ tự toàn phần trên các giá trị hữu hạn, và phần
            tử lớn nhất của một thứ tự toàn phần có tính kết hợp và giao hoán: mọi cây đều cho cùng một cặp, chỉ số cực đại đầu
            tiên. Đó cũng là chỉ số mà PyTorch ghi trong tài liệu cho <code>torch.argmax</code>. Một NaN cần thêm một quy tắc
            riêng.
          </p>
        }
      >
        <p>
          Ordering pairs by value and then by lower index is a total order on finite values, and the maximum of a total
          order is associative and commutative: every tree gives the same pair, the first maximal index. That is also the
          index PyTorch documents for <code>torch.argmax</code>. A NaN needs a rule of its own on top.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.2, pp. 224–225; PyTorch: notes/today/ch10-facts.md D4', // i18n-ignore: citation only
  },
};

// ---------------------------------------------------------------- the section
export function S102() {
  const q = questionsS102;
  return (
    <>
      <Translated
        vi={
          <p>
            Vòng lặp của mục 10.1 kết hợp từng phần tử một, nên N phần tử mất N bước dù bạn có bao nhiêu bộ xử lý. Một phép rút
            gọn song song thì kết hợp theo cặp. Ở bước đầu, các phép toán riêng biệt mỗi phép kết hợp hai đầu vào. Ở bước sau,
            một nửa số đó kết hợp các kết quả này, và cứ thế cho tới khi chỉ còn một giá trị. Vẽ với thời gian chảy từ trên
            xuống và mỗi hàng là một bước, đây là một <b>cây rút gọn</b>. Tám đầu vào cần 4 phép toán, rồi 2, rồi 1.
          </p>
        }
      >
        <p>
          The loop of section 10.1 combines one element at a time, so N elements take N steps however many processors you
          have. A parallel reduction combines pairs instead. In the first step, separate operations each combine two
          inputs. In the next, half as many combine those results, and so on until one value is left. Drawn with time
          flowing downward and one row per step, this is a <b>reduction tree</b>. Eight inputs take 4 operations, then 2,
          then 1.
        </p>
      </Translated>
      <Source section="10.2" pages="223–224" figs={tr('Fig. 10.3', 'Hình 10.3')} />
      <Translated
        vi={
          <p>
            Sách vẽ hai cây: một max trên 3, 1, 7, 0, 4, 1, 6, 3, cho ra 7, và một tổng trên 4, 7, 2, 3, 8, 5, 9, 6, cho ra
            44. Cây thứ hai là cây mà các kernel còn lại của chương tính. Dù có tên như vậy, cây rút gọn không phải là một cấu
            trúc dữ liệu. Các cạnh của nó là những giá trị được trao từ bước này sang bước kế.
          </p>
        }
      >
        <p>
          The book draws two: a max over 3, 1, 7, 0, 4, 1, 6, 3, which gives 7, and a sum over 4, 7, 2, 3, 8, 5, 9, 6,
          which gives 44. The second is the one the rest of the chapter's kernels compute. Despite the name, a reduction
          tree is not a data structure. Its edges are values handed from one step to the next.
        </p>
      </Translated>
      <NoteGroup>
        <Source section="10.2" pages="225, 227" />
        <Simplification>
          <Translated
            vi={
              <p>
                Trình dựng cây dùng số học số nguyên chính xác, nên chỉ thứ tự mới có thể thay đổi kết quả. Mỗi phép toán mất
                một bước; không có gì mô hình hóa luồng, bộ nhớ hay một bước kéo dài bao lâu.
              </p>
            }
          >
            <p>
              The builder uses exact integer arithmetic, so only the order can change a result. Every operation takes one
              step; nothing models threads, memory or how long a step takes.
            </p>
          </Translated>
        </Simplification>
      </NoteGroup>
      <TreeBuilder />
      <Translated
        vi={
          <p>
            Hoạt ảnh chạy vòng lặp và cây trên tám giá trị của sách, rồi xem điều gì thay đổi khi có nhiều giá trị hơn hẳn.
          </p>
        }
      >
        <p>
          The animation runs the loop and the tree on the book's eight values and then looks at what changes when there
          are many more of them.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một bộ cộng cho mỗi phép toán và một bước cho mỗi tầng. Các tổng float được làm tròn về float32 sau mỗi phép
              cộng, không có phép toán gộp (fused). Không có gì mô hình hóa luồng, bộ nhớ hay một bước kéo dài bao lâu.
            </p>
          }
        >
          <p>
            One adder per operation and one step per level. The float sums round to float32 after every addition, with no
            fused operations. Nothing models threads, memory or how long a step takes.
          </p>
        </Translated>
      </Simplification>
      <TreeFoldScene />

      <h2>
        <Translated vi="Cùng khối lượng công việc, độ dài ngắn hơn">Same work, shorter span</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hai con số mô tả một thuật toán song song. <b>Work</b> (khối lượng công việc) của nó là số phép toán nó thực hiện.{' '}
            <b>Span</b> (độ dài) của nó là thời gian nó cần nếu tài nguyên không giới hạn, đếm bằng bước, trong đó một bước là
            một khoảng thời gian mà một số hoặc tất cả các luồng mỗi luồng thực hiện một phép toán. Phép cộng vector của
            chương 2 làm N phép toán trong một bước. Với một thuật toán tuần tự, work và span là cùng một con số.
          </p>
        }
      >
        <p>
          Two numbers describe a parallel algorithm. Its <b>work</b> is how many operations it performs. Its <b>span</b>{' '}
          is how long it takes if resources are unlimited, counted in steps, where a step is a period in which some or all
          threads each perform one operation. The vector addition of chapter 2 does N operations in one step. A sequential
          algorithm's work and span are the same number.
        </p>
      </Translated>
      <Source section="10.2" pages="225–226" />
      <Translated
        vi={
          <p>
            Một cây trên N đầu vào thực hiện N/2 + N/4 + … + 1 = N − 1 phép toán: cùng work với vòng lặp. Span của nó là log₂ N
            bước. Với 1024 đầu vào là 10 bước, với 512 phép toán ở bước đầu và một ở bước cuối. Chia work cho span và bạn có{' '}
            <b>mức song song trung bình</b>: 1023 / 10 = 102.3, so với đỉnh 512. Một cỗ máy đủ lớn cho bước đầu sẽ phần lớn
            ngồi không trong các bước còn lại. Nhu cầu không đồng đều đó là điều khiến cây rút gọn khó chịu với phần cứng song
            song, và phần lớn các kernel của chương này là những cách sống chung với nó.
          </p>
        }
      >
        <p>
          A tree over N inputs does N/2 + N/4 + … + 1 = N − 1 operations: the same work as the loop. Its span is log₂ N
          steps. For 1024 inputs that is 10 steps, with 512 operations in the first and one in the last. Divide the work
          by the span and you get the <b>average parallelism</b>: 1023 / 10 = 102.3, against a peak of 512. A machine big
          enough for the first step sits mostly idle for the rest. That uneven demand is what makes reduction trees
          awkward for parallel hardware, and most of this chapter's kernels are ways of living with it.
        </p>
      </Translated>
      <Source section="10.2" pages="226" />
      <Translated
        vi={
          <p>
            Một cây không làm nhiều việc hơn vòng lặp không phải là điều hiển nhiên. Chương 11 gặp những thuật toán song song
            làm nhiều việc hơn phiên bản tuần tự, và câu hỏi nhiều hơn bao nhiêu thì chấp nhận được.
          </p>
        }
      >
        <p>
          A tree that does no more work than the loop is not a given. Chapter 11 meets parallel algorithms that do more
          work than the sequential version, and the question of how much more is acceptable.
        </p>
      </Translated>
      <Ask q={q['ch10/10.2/q-avg-4096']} />

      <h2>
        <Translated vi="Một giải đấu là một cây max">A tournament is a max tree</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một giải đấu loại trực tiếp là một cây rút gọn có toán tử trả về đội thắng. Tám đội còn bốn đội thắng sau tứ kết,
            hai sau bán kết, một sau chung kết: ba vòng, log₂ 8. 1024 đội sẽ cần 10 vòng. Một cây thì nhanh, và đói tài nguyên:
            ở vòng đầu mỗi trận cần sân và nhân sự riêng. Ban tổ chức nào dùng chung một sân cho hai trận của một vòng sẽ phải
            đấu chúng vào những ngày khác nhau, và giải đấu kéo dài hơn. Mục 10.10 cố ý thực hiện đúng sự đánh đổi đó, từ bỏ
            một phần mức song song để dùng phần cứng tốt hơn.
          </p>
        }
      >
        <p>
          A knockout tournament is a reduction tree whose operator returns the team that wins. Eight teams leave four
          winners after the quarter-finals, two after the semi-finals, one after the final: three rounds, log₂ 8. 1024
          teams would need 10 rounds. A tree is fast, and hungry for resources: in the first
          round every game needs its own stadium and staff. Organisers who share a stadium between two games of a round
          have to play them on different days, and the tournament gets longer. Section 10.10 makes the same trade
          deliberately, giving up some parallelism to use the hardware better.
        </p>
      </Translated>
      <NoteGroup>
        <Source section={tr('10.2 sidebar', '10.2 khung bên')} pages="226–227" figs={tr('Fig. 10.4', 'Hình 10.4')} />
        <Aside>
          <Translated
            vi={
              <p>
                Khung bên nói giải đấu trong Fig. 10.4 "has a span of four steps". Tám đội của nó đấu ba vòng, nên span là ba.
              </p>
            }
          >
            <p>
              The sidebar says the tournament in Fig. 10.4 "has a span of four steps". Its eight teams play three rounds,
              so the span is three.
            </p>
          </Translated>
        </Aside>
      </NoteGroup>
      <Ask q={q['ch10/10.2/q-venues']} />

      <h2>
        <Translated vi="Cây thay đổi thứ tự các phép toán">The tree changes the order of operations</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Vòng lặp và cây không thực hiện cùng những phép toán. Trên ví dụ max, vòng lặp so sánh 3 với 1, rồi bên thắng với
            7, và cứ thế dọc danh sách. Cây so sánh 7 với 0 trước khi kết hợp 3 với 1. Viết ra, vòng lặp tính ((((((3 max 1)
            max 7) max 0) max 4) max 1) max 6) max 3, còn cây tính ((3 max 1) max (7 max 0)) max ((4 max 1) max (6 max 3)). Cả
            hai cho 7 vì max có <b>tính kết hợp</b>: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c) với mọi a, b, c, nên dấu ngoặc đặt ở đâu cũng
            được.
          </p>
        }
      >
        <p>
          The loop and the tree do not perform the same operations. On the max example, the loop compares 3 with 1, then
          the winner with 7, and so on along the list. The tree compares 7 with 0 before it has combined 3 with 1. Written
          out, the loop computes ((((((3 max 1) max 7) max 0) max 4) max 1) max 6) max 3, and the tree computes ((3 max 1)
          max (7 max 0)) max ((4 max 1) max (6 max 3)). Both give 7 because max is <b>associative</b>: (a ⊕ b) ⊕ c = a ⊕
          (b ⊕ c) for every a, b, c, so the parentheses can go anywhere.
        </p>
      </Translated>
      <Source section="10.2" pages="224–225" />
      <Translated
        vi={
          <p>
            Phép cộng số nguyên có tính kết hợp: (1 + 2) + 3 = 1 + (2 + 3). Trong C++ điều đó chỉ đúng khi không có gì tràn
            số, vì một cách nhóm khác có thể tràn ở chỗ cách ban đầu không tràn, và tràn số có dấu là hành vi không xác định.
            Phép trừ số nguyên không có tính kết hợp: (1 − 2) − 3 ≠ 1 − (2 − 3). Nó vẫn có thể được rút gọn song song khi được
            viết lại thành phép cộng các toán hạng đã đổi dấu.
          </p>
        }
      >
        <p>
          Integer addition is associative: (1 + 2) + 3 = 1 + (2 + 3). In C++ that holds only while nothing overflows,
          because a different grouping can overflow where the original did not, and signed overflow is undefined
          behaviour. Integer subtraction is not associative: (1 − 2) − 3 ≠ 1 − (2 − 3). It can still be reduced in
          parallel once it is rewritten as an addition of negated operands.
        </p>
      </Translated>
      <Source section="10.2" pages="224 (footnote 2)" />
      <Ask q={q['ch10/10.2/q-subtract']} />
      <Translated
        vi={
          <p>
            Phép cộng dấu phẩy động có tính kết hợp trong toán học nhưng không trong máy tính. Mỗi phép cộng làm tròn kết quả
            về giá trị biểu diễn được gần nhất, và những kết quả trung gian nào bị làm tròn phụ thuộc vào cách nhóm. Appendix
            A giải một ví dụ trong một định dạng float năm bit: 1 + 1 + 0.25 + 0.25 cộng từ trái sang phải cho 2.0, vì mỗi
            0.25 quá nhỏ để thay đổi 2.0; cộng theo cặp, (1 + 1) + (0.25 + 0.25), nó cho đúng 2.5. Sách lưu ý rằng một ví dụ
            gần đó lại đi theo chiều ngược lại. Các ứng dụng chấp nhận kết quả trong một dung sai vẫn coi phép cộng float là
            có tính kết hợp, và các kernel trong chương này cũng vậy.
          </p>
        }
      >
        <p>
          Floating-point addition is associative in mathematics and not in a computer. Each addition rounds its result to
          the nearest representable value, and which intermediate results get rounded depends on the grouping. Appendix A
          works an example in a five-bit float format: 1 + 1 + 0.25 + 0.25 added left to right gives 2.0, because each
          0.25 is too small to change 2.0; added in pairs, (1 + 1) + (0.25 + 0.25), it gives the exact 2.5. The book
          notes that a nearby example goes the other way. Applications that accept results within a tolerance treat float
          addition as associative anyway, and so do the kernels in this chapter.
        </p>
      </Translated>
      <NoteGroup>
        <Source section="10.2" pages="224" figs="Appendix A §A.6, pp. 593–594" />
        <Today checked="2026-11-20">
          <Translated
            vi={
              <p>
                Hướng dẫn lập trình của NVIDIA giờ có ví dụ ba giá trị của riêng nó (§5.6.1.4), và ví dụ tích vô hướng của nó
                (§5.6.1.6) gọi thứ tự hình cây là "the parallel algorithm".
              </p>
            }
          >
            <p>
              NVIDIA's programming guide now has its own three-value example (§5.6.1.4), and its dot-product example
              (§5.6.1.6) calls the tree-shaped order "the parallel algorithm".
            </p>
          </Translated>
          <p className="muted">notes/today/ch10-facts.md A9{/* i18n-ignore: citation only */}</p>
        </Today>
      </NoteGroup>
      <Translated
        vi={<p>Thứ tự quan trọng đến mức nào với tổng mà softmax dùng để chia? Hãy đoán trước khi xem.</p>}
      >
        <p>How much does the order matter for the sum a softmax divides by? Guess before you look.</p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Làm tròn về gần nhất sau mỗi phép cộng: không phép toán gộp, không đẩy về không, không có trình biên dịch
              kết hợp lại.
            </p>
          }
        >
          <p>
            Rounding to nearest after every addition: no fused operations, no flushing to zero, no reassociation by a
            compiler.
          </p>
        </Translated>
      </Simplification>
      <Predict
        id="ch10/10.2/p-orders"
        prompt={
          <Translated vi="1024 giá trị float32, các số hạng của mẫu số softmax cho một hàng có một logit lớn, được cộng theo ba thứ tự: vòng lặp, cặp kề nhau và hai nửa. Ba thứ tự sẽ cho bao nhiêu kết quả khác nhau: 1, 2 hay 3?">
            1024 float32 values, the terms of a softmax denominator for a row with one large logit, are summed in the three orders: the loop, neighbours and halves. How many different results will the three give: 1, 2 or 3?
          </Translated>
        }
        unit={tr('different results', 'kết quả khác nhau')}
      >
        {(v) => (
          <Translated
            vi={
              <p>
                <b>{DEFAULT_ORDERS.distinct} trên bộ dữ liệu đầu tiên bên dưới.</b>{' '}
                {v === DEFAULT_ORDERS.distinct ? 'Đúng như bạn đã dự đoán. ' : ''}
                {v === 1
                  ? 'Một kết quả có nghĩa là cách nhóm không bao giờ thay đổi một lần làm tròn nào, qua cả nghìn phép cộng. Gần như luôn luôn là có thay đổi.'
                  : v === 3 && DEFAULT_ORDERS.distinct === 2
                    ? 'Ba cũng có xảy ra; trên bộ dữ liệu này hai thứ tự cây đồng ý với nhau.'
                    : ''}{' '}
                Trên {DEFAULT_SURVEY.trials} bộ dữ liệu loại này, ít nhất hai thứ tự bất đồng ở {DEFAULT_SURVEY.disagree} bộ.
                Các sai khác nằm ở những bit cuối. Hãy khám phá khi nào chúng xuất hiện.
              </p>
            }
          >
            <p>
              <b>
                {DEFAULT_ORDERS.distinct} on the first data set below.
              </b>{' '}
              {v === DEFAULT_ORDERS.distinct ? 'That is what you predicted. ' : ''}
              {v === 1
                ? 'One would mean the grouping never changed a rounding, across a thousand additions. It almost always does.'
                : v === 3 && DEFAULT_ORDERS.distinct === 2
                  ? 'Three happens too; on this data set the two tree orders agree with each other.'
                  : ''}{' '}
              Over {DEFAULT_SURVEY.trials} data sets of this kind, at least two orders disagree in {DEFAULT_SURVEY.disagree}.
              The differences sit in the last bits. Explore when they appear.
            </p>
          </Translated>
        )}
      </Predict>
      <OrdersBench />
      <Translated
        vi={
          <p>
            Các sai khác nhỏ và có thật. Không thứ tự nào trong ba nói chung là tổng chính xác. Khi N tăng, các thứ tự cây rơi
            gần tổng chính xác thường xuyên hơn vòng lặp, vì mỗi giá trị chỉ đi qua log₂ N phép cộng thay vì tới N phép, nhưng
            không phải lúc nào cũng vậy. Với softmax, điều này có nghĩa là hai kernel đúng có thể cho ra các xác suất khác nhau
            ở những bit cuối, và một bài kiểm thử so với vòng lặp CPU cần một dung sai, không bao giờ là một phép so sánh bằng.
          </p>
        }
      >
        <p>
          The differences are small and they are real. None of the three orders is the exact sum in general. The tree
          orders land closer to it more often than the loop does as N grows, because each value passes through only log₂ N
          additions instead of up to N, but not always. For softmax this means two correct kernels can produce
          probabilities that differ in the last bits, and a test against a CPU loop needs a tolerance, never an equality.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Cách đặt vấn đề theo softmax và lời giải thích "log₂ N phép cộng cho mỗi giá trị" là của khóa học này.</p>}>
          <p>The softmax framing and the "log₂ N additions per value" explanation are ours.</p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Sắp xếp lại cả các toán hạng">Reordering the operands as well</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mục 10.4 đi thêm một bước. Kernel của nó kết hợp x0 với x4 và x1 với x5, không phải x0 với x1. Điều đó thay đổi
            trình tự các toán hạng, không chỉ dấu ngoặc, nên toán tử còn phải có <b>tính giao hoán</b>: a ⊕ b = b ⊕ a. Max,
            min, phép cộng và phép nhân có tính chất này; phép trừ thì không. Đặt trình dựng cây sang "join", một toán tử có
            tính kết hợp nhưng không giao hoán, và so sánh thứ tự cặp kề nhau với thứ tự hai nửa.
          </p>
        }
      >
        <p>
          Section 10.4 goes one step further. Its kernel combines x0 with x4 and x1 with x5, not x0 with x1. That changes
          the sequence of the operands, not only the parentheses, so the operator must also be <b>commutative</b>: a ⊕ b =
          b ⊕ a. Max, min, addition and multiplication are; subtraction is not. Set the builder to "join", an operator that
          is associative but not commutative, and compare the neighbours order with the halves order.
        </p>
      </Translated>
      <Source section="10.2" pages="225" />
      <Ask q={q['ch10/10.2/q-commute']} />
      <Translated
        vi={
          <p>
            Một lưu ý trong chú thích quan trọng hơn vẻ ngoài của nó. Max và min dấu phẩy động không hoàn toàn giao hoán khi
            một toán hạng là NaN: max(NaN, 1) và max(1, NaN) không nhất thiết khớp nhau. Hàm max nào bạn viết quyết định điều
            gì xảy ra, và một cây được tự do đặt NaN ở bên nào cũng được.
          </p>
        }
      >
        <p>
          One caveat in a footnote matters more than it looks. Floating-point max and min are not exactly commutative when
          one operand is a NaN: max(NaN, 1) and max(1, NaN) need not agree. Which max you write decides what happens, and
          a tree is free to put the NaN on either side.
        </p>
      </Translated>
      <NoteGroup>
        <Source section="10.2" pages="225 (footnote 3)" />
        <Today checked="2026-11-20 (ch10-facts A10)">{/* i18n-ignore: date and citation only */}
          <Translated
            vi={
              <p>
                <code>fmaxf</code> của CUDA coi NaN "as missing data"; <code>a &gt; b ? a : b</code> trả về <code>b</code>{' '}
                nếu một trong hai là NaN.
              </p>
            }
          >
            <p>
              CUDA's <code>fmaxf</code> treats NaN "as missing data"; <code>a &gt; b ? a : b</code> returns <code>b</code>{' '}
              if either is NaN.
            </p>
          </Translated>
        </Today>
      </NoteGroup>

      <h2>
        <Translated vi="Ai thắng khi hòa">Who wins a tie</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Decode greedy chọn chỉ số của logit lớn nhất. GPU tính nó như một cây max trên các cặp (giá trị, chỉ số), nên toán
            tử cặp cũng phải có tính kết hợp và giao hoán. Với các cặp, đó là một ràng buộc thật: hai logit có thể bằng nhau,
            và một quy tắc như "khi hòa, giữ toán hạng bên trái" cho bên thắng phụ thuộc vào toán hạng nào tình cờ được cây đặt
            bên trái. Hãy tìm xem những quy tắc nào sống sót qua mọi thứ tự.
          </p>
        }
      >
        <p>
          Greedy decoding picks the index of the largest logit. A GPU computes it as a max tree over (value, index) pairs,
          so the pair operator must be associative and commutative too. For pairs that is a real constraint: two logits
          can be equal, and a rule such as "on a tie, keep the left operand" gives a winner that depends on which operand
          the tree happens to put on the left. Find out which rules survive every order.
        </p>
      </Translated>
      <NoteGroup>
        <Beyond>
          <Translated vi={<p>Argmax dưới dạng phép rút gọn trên cặp, và các so sánh thư viện bên dưới, là của khóa học này.</p>}>
            <p>Argmax as a pair reduction, and the library comparisons below, are ours.</p>
          </Translated>
        </Beyond>
        <Simplification>
          <Translated
            vi={
              <p>
                Một cây cho mỗi thứ tự, không có warp hay khối. Hành vi của các thư viện nêu trong trình khám phá là cách đọc
                của chúng tôi về tài liệu và mã nguồn của chúng, không phải kiểm thử trên phần cứng.
              </p>
            }
          >
            <p>
              One tree per order, no warps or blocks. Library behaviours named in the explorer are our reading of their
              documentation and source, not tests on hardware.
            </p>
          </Translated>
        </Simplification>
      </NoteGroup>
      <TieBench />
      <Translated
        vi={
          <p>
            Các thư viện không thống nhất. PyTorch ghi trong tài liệu rằng <code>torch.argmax</code> trả về chỉ số cực đại đầu
            tiên, và <code>ArgMax</code> của CUB tìm "the first device-wide maximum". Trong llama.cpp, theo cách chúng tôi đọc
            mã nguồn, bộ lấy mẫu greedy phía host giữ chỉ số nhỏ nhất khi hòa, đáp án của kernel argmax CUDA phụ thuộc vào lane
            nào giữ chỉ số nào, còn kernel Metal lấy chỉ số lớn nhất. Một mô hình, chạy theo ba cách, có thể chọn các token
            khác nhau khi hòa chính xác. NaN là một chính sách thứ hai chồng lên: PyTorch và MLX để NaN thắng,{' '}
            <code>Max</code> của CUB bỏ qua NaN, và argmax CUDA của llama.cpp bỏ qua nó. Hòa chính xác trong các logits float
            là hiếm; dù vậy mốc dự án M4 vẫn nêu rõ quy tắc của nó và kiểm thử bằng các điểm hòa cài sẵn.
          </p>
        }
      >
        <p>
          Libraries do not agree. PyTorch documents that <code>torch.argmax</code> returns the first maximal index, and
          CUB's <code>ArgMax</code> finds "the first device-wide maximum". In llama.cpp, by our reading of the source,
          the host's greedy sampler keeps the lowest index on a tie, the CUDA argmax kernel's answer depends on which lane
          held which index, and the Metal kernel takes the largest index. One model, run three ways, can pick different
          tokens on an exact tie. NaN is a second policy on top: PyTorch and MLX let a NaN win, CUB's <code>Max</code>{' '}
          ignores NaN, and llama.cpp's CUDA argmax skips it. Exact ties in float logits are rare; milestone M4 states its
          rule and tests it with planted ties anyway.
        </p>
      </Translated>
      <Today checked="2026-11-20">
        <Translated
          vi={
            <p>
              PyTorch 2.15: "If there are multiple maximal values then the indices of the first maximal value are returned."
              Hành vi của llama.cpp được đọc từ mã nguồn, không phải quan sát.
            </p>
          }
        >
          <p>
            PyTorch 2.15: "If there are multiple maximal values then the indices of the first maximal value are returned."
            llama.cpp's behaviour is read from source, not observed.
          </p>
        </Translated>
        <p className="muted">ch10-facts A13, D2, D3, D4{/* i18n-ignore: citation only */}</p>
      </Today>
      <Ask q={q['ch10/10.2/q-argmax-rule']} />
      <Translated
        vi={
          <p>
            Với cây, chi phí của nó và hai điều kiện của nó trong tay, <a href={href('ch10', '10.3')}>mục 10.3</a> viết kernel
            đầu tiên.
          </p>
        }
      >
        <p>
          With the tree, its cost and its two conditions in hand, <a href={href('ch10', '10.3')}>section 10.3</a> writes
          the first kernel.
        </p>
      </Translated>
    </>
  );
}
