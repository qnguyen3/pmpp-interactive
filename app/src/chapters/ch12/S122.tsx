import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { Beyond, Bench, Simplification, Source, Today, type Tier } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  FRACTIONS,
  PATTERNS,
  PATTERN_LABEL,
  atomicCounts,
  curve,
  expectedAtomics,
  patternSource,
  type Pattern,
} from '../../sim/filter';
import { functionSource } from '../../sim/toolchain';

// Section 12.2, A simple parallel unstable filter (pp. 290-291). The listing is lab 12.1's
// reference solution; the explorer runs on src/sim/filter.ts. Vendor facts:
// notes/today/ch12-facts.md (A7, B4, D1), ch09-facts.md (A3, B1), infra-facts.md (B1, B2, C1).

const LAB = 'ch12/lab1_unstable/solution/';
/** A lab's reference-solution status, in the labs page's words (LabKit's solutionTabStatus), from this machine's runs. */
function useLabStatus(lab: string) {
  const { reports } = useReports();
  return {
    cu: solutionTabStatus(reports, lab, 'kernel.cu'),
    metal: solutionTabStatus(reports, lab, 'kernel.metal'),
    mm: solutionTabStatus(reports, lab, 'launch.mm'),
  };
}
const TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', outputSize: 'global', outputSize_ref: 'global',
  i: 'reg', val: 'reg', j: 'reg', n: 'reg', threshold: 'reg',
};

const missing = (s: string | null) => !s || s.startsWith('// could not load');
/** The text between "// region: name" and "// endregion" in a lab file, or the named function. */
function labPart(src: string | null, region: string, fn: string): string | null {
  if (missing(src)) return null;
  const m = new RegExp('//\\s*region:\\s*' + region + '\\s*\\n([\\s\\S]*?)\\n\\s*//\\s*endregion').exec(src!);
  if (m) return m[1];
  return functionSource(src!, fn)?.text ?? null;
}

function Listing() {
  const status = useLabStatus('ch12/lab1_unstable');
  const cu = useLabFile(LAB + 'kernel.cu');
  const metal = useLabFile(LAB + 'kernel.metal');
  const k = labPart(cu, 'atomic', 'filterAtomicKernel');
  const m = labPart(metal, 'atomic', 'filter_atomic');
  if (!k) return null;
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: k, tiers: TIERS, status: status.cu },
        ...(m ? [{ label: 'Metal', code: m, tiers: TIERS, status: status.metal }] : []),
      ]}
      caption={
        <Translated
          vi={
            <>
              Fig. 12.2 theo cách lab 12.1 viết, từ <code>labs/{LAB}</code>. Bản build HIP biên dịch tệp CUDA, trong đó
              nhánh <code>__HIPCC__</code> dùng <code>atomicAdd</code> kiểu cũ.
            </>
          }
        >
          Fig. 12.2 as lab 12.1 writes it, from <code>labs/{LAB}</code>. The HIP build compiles the CUDA file, where
          the <code>__HIPCC__</code> branch uses the legacy <code>atomicAdd</code>.
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- explorer
const BLOCKS = [64, 128, 256, 512, 1024];
const pct = (f: number) => (f === 0 ? '0' : f < 0.01 ? (f * 100).toFixed(1) : String(Math.round(f * 1000) / 10)) + tr(' percent', ' phần trăm');
const count = (v: number) => (v >= 10 ? Math.round(v).toLocaleString('en-US') : v.toFixed(1));
const ratio = (a: number, b: number) => (b <= 0 ? '–' : a / b >= 10 ? (a / b).toFixed(1) : (a / b).toFixed(2));

const PW = 580;
const PH = 300;
const PL = 58;
const PR = 118;
const PT = 14;
const PB = 40;
const STYLE: Record<'perKey' | 'perWarp' | 'perBlock', { dash?: string; label: string }> = {
  perKey: {
    get label() {
      return tr('one per kept key', 'một cho mỗi khóa được giữ');
    },
  },
  perWarp: {
    dash: '7 5',
    get label() {
      return tr('one per busy warp', 'một cho mỗi warp có việc');
    },
  },
  perBlock: {
    dash: '2 4',
    get label() {
      return tr('one per block', 'một cho mỗi khối');
    },
  },
};

/** The curves' key, above the plot, so that it stays in view when a narrow screen scrolls the plot sideways. */
function Legend() {
  return (
    <div className="row ui" style={{ gap: '0.3rem 1rem', fontSize: '0.78rem', marginBottom: '0.3rem', color: 'var(--ink-2)' }} data-legend="">
      {(['perKey', 'perWarp', 'perBlock'] as const).map((k) => (
        <span key={k} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }} data-legend-item={k}>
          <svg width="30" height="8" aria-hidden="true">
            <line x1="1" x2="29" y1="4" y2="4" stroke="var(--global)" strokeWidth="2" strokeDasharray={STYLE[k].dash} />
          </svg>
          {STYLE[k].label}
        </span>
      ))}
    </div>
  );
}

function Plot({ pattern, n, warp, block, fraction }: { pattern: Pattern; n: number; warp: number; block: number; fraction: number }) {
  const pts = useMemo(() => curve(pattern, n, warp, block), [pattern, n, warp, block]);
  const max = Math.max(...pts.map((p) => Math.max(p.perKey, p.perWarp, p.perBlock)), 10);
  const top = Math.ceil(Math.log10(max));
  const x = (f: number) => PL + ((Math.log10(f) + 3) / 3) * (PW - PL - PR);
  const y = (v: number) => PT + (1 - Math.log10(Math.max(v, 1)) / top) * (PH - PT - PB);
  const keys = ['perKey', 'perWarp', 'perBlock'] as const;
  const now = fraction > 0 ? (pattern === 'random' ? expectedAtomics(n, fraction, warp, block) : atomicCounts(patternSource(pattern, n, fraction), warp, block)) : null;
  return (
    <svg viewBox={`0 0 ${PW} ${PH}`} role="img" aria-label={tr('Atomics on the public counter against the fraction of keys kept', 'Số atomic trên bộ đếm chung theo tỉ lệ khóa được giữ')}
      style={{ width: '100%', minWidth: 520, display: 'block' }} data-plot={pattern}>
      {Array.from({ length: top + 1 }, (_, k) => (
        <g key={'y' + k}>
          <line x1={PL} x2={PW - PR} y1={y(10 ** k)} y2={y(10 ** k)} stroke="var(--rule-soft)" />
          <text x={PL - 6} y={y(10 ** k) + 4} fontSize="12" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
            {k < 3 ? 10 ** k : k < 6 ? `${10 ** (k - 3)}k` : `${10 ** (k - 6)}M`}
          </text>
        </g>
      ))}
      {[0.001, 0.01, 0.1, 1].map((f) => (
        <g key={'x' + f}>
          <line x1={x(f)} x2={x(f)} y1={PT} y2={PH - PB} stroke="var(--rule-soft)" />
          <text x={x(f)} y={PH - PB + 17} fontSize="12" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{f * 100}%</text>
        </g>
      ))}
      <text x={(PL + PW - PR) / 2} y={PH - 4} fontSize="12" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{tr('fraction of keys kept (log scale)', 'tỉ lệ khóa được giữ (thang log)')}</text>
      <text x={13} y={PT + 4} fontSize="12" style={{ fill: 'var(--ink-2)' }} transform={`rotate(-90 13 ${PT + 4})`} textAnchor="end">{tr('atomics on the counter', 'số atomic trên bộ đếm')}</text>
      {keys.map((k) => (
        <g key={k} data-curve={k}>
          <polyline fill="none" stroke="var(--global)" strokeWidth="2" strokeDasharray={STYLE[k].dash}
            points={pts.map((p) => `${x(p.fraction).toFixed(1)},${y(p[k]).toFixed(1)}`).join(' ')} />
          <text x={PW - PR + 6} y={y(pts[pts.length - 1][k]) + (k === 'perBlock' ? 13 : k === 'perWarp' ? 2 : -6)} fontSize="12">
            {STYLE[k].label}
          </text>
        </g>
      ))}
      {now ? (
        <g data-marker={fraction}>
          <line x1={x(fraction)} x2={x(fraction)} y1={PT} y2={PH - PB} stroke="var(--ink)" strokeWidth="1" strokeDasharray="3 3" />
          {keys.map((k) => (
            <circle key={k} cx={x(fraction)} cy={y(now[k])} r="4" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.6" />
          ))}
        </g>
      ) : null}
    </svg>
  );
}

function OneCounter() {
  const [fi, setFi] = useState(FRACTIONS.indexOf(0.5));
  const [pattern, setPattern] = useState<Pattern>('random');
  const [log2n, setLog2n] = useState(20);
  const [block, setBlock] = useState(256);
  const [warp, setWarp] = useState<32 | 64>(32);
  const f = FRACTIONS[fi];
  const n = 2 ** log2n;
  const c = useMemo(
    () => (pattern === 'random' ? { ...expectedAtomics(n, f, warp, block) } : atomicCounts(patternSource(pattern, n, f), warp, block)),
    [pattern, n, f, warp, block],
  );
  const rows: { key: 'perKey' | 'perWarp' | 'perBlock'; what: string; where: string }[] = [
    { key: 'perKey', what: tr('one per kept key', 'một cho mỗi khóa được giữ'), where: 'Fig. 12.2' },
    { key: 'perWarp', what: tr('one per warp that has a kept key', 'một cho mỗi warp có khóa được giữ'), where: tr('section 12.3', 'mục 12.3') },
    { key: 'perBlock', what: tr('one per block, even one that kept nothing', 'một cho mỗi khối, kể cả khối không giữ gì'), where: 'Fig. 12.6' },
  ];
  return (
    <Bench
      title={tr('One counter, many threads', 'Một bộ đếm, nhiều luồng')}
      task={tr(
        'Start at 50 percent, then drag the fraction towards zero. Then try runs of kept keys.',
        'Bắt đầu ở 50 phần trăm, rồi kéo tỉ lệ về gần 0. Sau đó thử các dải khóa được giữ liền nhau.',
      )}
      foot={
        pattern === 'random'
          ? tr('Independent keys: the numbers are expected values, so they need not be whole.', 'Khóa độc lập: các số là giá trị kỳ vọng, nên không nhất thiết là số nguyên.')
          : tr(
              `One concrete input with this layout${pattern === 'runs' ? ' (runs of 50 keys, each run kept or dropped as a whole)' : ''}; the counts are exact.`,
              `Một đầu vào cụ thể với cách bố trí này${pattern === 'runs' ? ' (các dải 50 khóa, mỗi dải được giữ hoặc bỏ nguyên cả dải)' : ''}; số đếm là chính xác.`,
            )
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ minWidth: '13rem' }}>
          {tr('Keys kept:', 'Khóa được giữ:')} {pct(f)}
          <input type="range" min={0} max={FRACTIONS.length - 1} value={fi} aria-label={tr('fraction kept', 'tỉ lệ được giữ')} onChange={(e) => setFi(+e.target.value)} />
        </label>
        <label className="field">
          {tr('Layout', 'Cách bố trí')}
          <select value={pattern} onChange={(e) => setPattern(e.target.value as Pattern)} aria-label={tr('layout', 'cách bố trí')}>
            {PATTERNS.map((p) => <option key={p} value={p}>{tr(PATTERN_LABEL[p], { random: 'khóa ngẫu nhiên độc lập', runs: 'các dải khóa được giữ liền nhau', spread: 'rải đều (ở 50 phần trăm, cứ cách một khóa)', sorted: 'đã sắp xếp: các khóa được giữ đứng trước' }[p])}</option>)}
          </select>
        </label>
        <label className="field" style={{ minWidth: '11rem' }}>
          <span><Translated vi="Số khóa:">Keys:</Translated> 2<sup>{log2n}</sup> = {n.toLocaleString('en-US')}</span>
          <input type="range" min={10} max={24} value={log2n} aria-label={tr('number of keys', 'số khóa')} onChange={(e) => setLog2n(+e.target.value)} />
        </label>
        <label className="field">
          {tr('Block size', 'Kích thước khối')}
          <select value={block} onChange={(e) => setBlock(+e.target.value)} aria-label={tr('block size', 'kích thước khối')}>
            {BLOCKS.map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </label>
        <span className="field ui" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
          {tr('Warp width', 'Độ rộng warp')}
          <span className="seg" role="group" aria-label={tr('warp width', 'độ rộng warp')}>
            {[32, 64].map((w) => (
              <button key={w} aria-pressed={warp === w} onClick={() => setWarp(w as 32 | 64)}>{w}</button>
            ))}
          </span>
        </span>
      </div>
      <div className="cols plot">
        <div>
          <Legend />
          <Plot pattern={pattern} n={n} warp={warp} block={block} fraction={f} />
          {f === 0 ? <p className="status-line"><Translated vi="0 phần trăm không có chỗ trên trục log; bảng vẫn đếm nó.">0 percent has no place on a log axis; the table still counts it.</Translated></p> : null}
        </div>
        <div>
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr><th><Translated vi="Số atomic trên bộ đếm chung">Atomics on the public counter</Translated></th><th className="num"><Translated vi="số đếm">count</Translated></th><th className="num"><Translated vi="ít hơn">fewer by</Translated></th></tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.key} data-count={r.key} data-value={c[r.key]}>
                  <td>{r.what} <span className="muted">({r.where})</span></td>
                  <td className="num">{count(c[r.key])}</td>
                  <td className="num">{r.key === 'perKey' ? '' : `${ratio(c.perKey, c[r.key])}×`}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="status-line" style={{ marginTop: '0.7rem' }} role="status" data-summary="">
            {c.perKey === 0
              ? tr('Nothing is kept: no atomics per key or per warp, but every block still adds its zero.', 'Không gì được giữ: không có atomic nào cho mỗi khóa hay mỗi warp, nhưng mỗi khối vẫn cộng số 0 của nó.')
              : tr(
                  `Per warp issues ${ratio(c.perKey, c.perWarp)} times fewer atomics than per key here; the most it can ever save is ${warp} times.`,
                  `Ở đây, cách theo warp phát ít hơn ${ratio(c.perKey, c.perWarp)} lần số atomic so với cách theo khóa; mức tiết kiệm tối đa nó có thể đạt là ${warp} lần.`,
                )}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS122: Record<string, Question> = {
  'ch12/12.2/q-stale-counter': {
    kind: 'numeric',
    id: 'ch12/12.2/q-stale-counter',
    prompt: (
      <Translated vi={<>Trước một lần khởi chạy Fig. 12.2, host quên đặt bộ đếm về 0, nên nó vẫn giữ giá trị 5 từ lần gọi trước. Sáu khóa vượt qua. Chỉ số đầu ra cao nhất được ghi là bao nhiêu?</>}>
        Before a launch of Fig. 12.2 the host forgets to zero the counter, which still holds 5 from the previous call.
        Six keys pass. What is the highest output index written?
      </Translated>
    ),
    answer: 10,
    explain: (
      <Translated
        vi={
          <p>
            <code>fetch_add</code> trả ra các giá trị cũ 5, 6, 7, 8, 9 và 10, nên sáu khóa rơi vào{' '}
            <code>output[5]</code> đến <code>output[10]</code> và bộ đếm kết thúc ở 11. Các ô 0 đến 4 giữ dữ liệu cũ, kích thước
            được báo là sai, và một đầu ra có kích thước cho sáu khóa như dự kiến bị ghi tràn. Sách nêu rõ việc đặt về 0 cho
            Fig. 12.2 (p. 291) và mặc định điều đó cho các kernel sau nó.
          </p>
        }
      >
        <p>
          <code>fetch_add</code> hands out the old values 5, 6, 7, 8, 9 and 10, so the six keys land in{' '}
          <code>output[5]</code> to <code>output[10]</code> and the counter ends at 11. Slots 0 to 4 keep stale data,
          the reported size is wrong, and an output sized for the six expected keys is overrun. The book states the
          zeroing for Fig. 12.2 (p. 291) and assumes it for the kernels after it.
        </p>
      </Translated>
    ),
    source: '§12.2, pp. 290–291',
  },
  'ch12/12.2/q-same-twice': {
    kind: 'choice',
    id: 'ch12/12.2/q-same-twice',
    prompt: (
      <Translated vi={<>Fig. 12.2 chạy hai lần trên cùng một đầu vào. Điều gì được bảo đảm giống hệt nhau giữa hai lần chạy?</>}>
        <>Fig. 12.2 runs twice on the same input. What is guaranteed to be identical between the two runs?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Tất cả: đầu vào là như nhau, và các atomic được phục vụ theo thứ tự các luồng được khởi chạy.">
            Everything: the input is the same, and the atomics are served in the order the threads were launched.
          </Translated>
        ),
        why: (
          <Translated vi="Thứ tự thực hiện các atomic trên một địa chỉ không cố định (ch09-facts A3), nên các ô được cấp ra theo một thứ tự khác.">
            The order in which atomics on one address are performed is not fixed (ch09-facts A3), so the slots go out in a different order.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Giá trị cuối của bộ đếm và tập khóa trong output[0, count); không phải thứ tự của chúng.">
            The final counter and the set of keys in output[0, count); not their order.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chỉ số đếm.">
            Only the count.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi khóa được giữ được ghi đúng một lần, vào một ô mà không khóa nào khác nhận, nên tập khóa cũng như nhau.">
            Every kept key is written exactly once, to a slot no other key gets, so the set of keys is the same too.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mỗi khóa được giữ thực hiện đúng một <code>fetch_add(1)</code>, nên bộ đếm luôn kết thúc ở số khóa được giữ, và mỗi
            khóa nhận một ô riêng biệt nhỏ hơn số đó. Khóa nào nhận ô nào do thứ tự phục vụ các atomic quyết định, và thứ tự đó
            không cố định.
          </p>
        }
      >
        <p>
          Each kept key performs exactly one <code>fetch_add(1)</code>, so the counter always ends at the number of kept
          keys, and each gets a distinct slot below it. Which key gets which slot is decided by the order the atomics are
          served in, and that is not fixed.
        </p>
      </Translated>
    ),
    source: '§12.2, p. 291',
  },
  'ch12/12.2/q-lowfraction': {
    kind: 'numeric',
    id: 'ch12/12.2/q-lowfraction',
    prompt: (
      <Translated vi={<>Các khóa độc lập, và 1 phần trăm trong số đó vượt qua. Chuyển từ một atomic cho mỗi khóa được giữ sang một atomic cho mỗi warp 32 lane, trung bình số atomic trên bộ đếm giảm đi bao nhiêu lần? Hai chữ số thập phân.</>}>
        Keys are independent, and 1 percent of them pass. Moving from one atomic per kept key to one per warp of 32
        lanes, by what factor does the number of atomics on the counter fall, on average? Two decimals.
      </Translated>
    ),
    answer: 1.16,
    tol: 0.03,
    unit: '×',
    explain: (
      <Translated
        vi={
          <>
            <p>
              Mỗi warp có số khóa được giữ kỳ vọng là 32 × 0.01 = 0.32. Xác suất một warp có ít nhất một khóa là
              1 − 0.99<sup>32</sup> ≈ 0.275. Tỉ số là 0.32 / 0.275 ≈ 1.16. Ở 50 phần trăm nó là 16, ở 100 phần trăm là 32.
            </p>
            <p>
              Gộp theo warp có lợi khi các warp bận; một atomic cho mỗi khối có lợi ở mọi tỉ lệ. Đây là dự đoán mà lab 12.1 kiểm
              tra trên thiết bị của bạn.
            </p>
          </>
        }
      >
        <>
          <p>
            Per warp, the expected number of kept keys is 32 × 0.01 = 0.32. The chance that a warp has at least one is
            1 − 0.99<sup>32</sup> ≈ 0.275. The ratio is 0.32 / 0.275 ≈ 1.16. At 50 percent it is 16, at 100 percent 32.
          </p>
          <p>
            Aggregating per warp pays when warps are busy; one atomic per block pays at any fraction. This is the
            prediction lab 12.1 tests on your device.
          </p>
        </>
      </Translated>
    ),
    source: 'our arithmetic on §§12.2–12.4, pp. 291–297',
    vi: { source: 'phép tính của khóa học này dựa trên §§12.2–12.4, pp. 291–297' },
  },
};

export function S122() {
  const half = expectedAtomics(2 ** 20, 0.5, 32, 256);
  return (
    <>
      <Translated
        vi={
          <p>
            Bộ lọc không ổn định đơn giản nhất cấp cho mỗi khóa một luồng riêng. Một luồng có khóa vượt qua phép kiểm tra cần
            một ô trong đầu ra, và một bộ đếm trong bộ nhớ toàn cục (global memory) phát các ô: luồng cộng nguyên tử 1 vào bộ đếm
            và dùng giá trị nhận lại làm ô của mình. Đây là kernel đó theo cách lab 12.1 viết.
          </p>
        }
      >
        <p>
          The simplest unstable filter gives every key a thread of its own. A thread whose key passes the test needs one
          slot in the output, and a counter in global memory hands slots out: the thread adds 1 to the counter atomically
          and uses the value it gets back as its slot. Here is that kernel as lab 12.1 writes it.
        </p>
      </Translated>
      <Source section="12.2" pages="290–291" figs="Fig. 12.2" />
      <Listing />
      <Translated
        vi={
          <p>
            Chỗ khác với Fig. 12.2: phép kiểm tra của sách, <code>cond(val)</code>, là một chỗ giữ chỗ mà sách không bao giờ định
            nghĩa; ở đây nó là <code>val &lt; threshold</code>, với ngưỡng được truyền vào để lab có thể chọn bao nhiêu khóa vượt
            qua. Bộ đếm phải bằng 0 trước khi khởi chạy; sách nói vậy cho kernel này (p. 291), và mã host của lab làm việc đó
            trước mỗi lần gọi.
          </p>
        }
      >
        <p>
          Where it differs from Fig. 12.2: the book's test, <code>cond(val)</code>, is a placeholder it never defines;
          here it is <code>val &lt; threshold</code>, with the threshold passed in so the lab can choose how many keys
          pass. The counter must be zero before the launch; the book says so for this kernel (p. 291), and the lab's
          host code does it before every call.
        </p>
      </Translated>

      <h2><Translated vi="Vì sao giá trị cũ là một ô">Why the old value is a slot</Translated></h2>
      <Translated
        vi={
          <p>
            <code>fetch_add</code> trả về giá trị bộ đếm đang giữ ngay trước phép cộng của luồng này. Histogram của chương 9 vứt
            giá trị đó đi; chương 11 dùng nó để đánh số các khối (<a href={href('ch11', '11.9')}>mục 11.9</a>, p. 280); ở đây nó
            trở thành ô của mỗi khóa. Mỗi atomic trên bộ đếm thấy kết quả của mọi atomic thực hiện trước nó, nên không có hai
            atomic nào nhận lại cùng một giá trị: k khóa được giữ nhận k ô khác nhau, 0 đến k − 1, và bộ đếm kết thúc ở k, chính
            là độ dài của đầu ra. Xem <a href={href('ch09', '9.2')}>mục 9.2</a> cho bản thân lời gọi.
          </p>
        }
      >
        <p>
          <code>fetch_add</code> returns the value the counter held just before this thread's addition. Chapter 9's
          histogram threw that value away; chapter 11 used it to number blocks (<a href={href('ch11', '11.9')}>section
          11.9</a>, p. 280); here it becomes each key's slot. Each atomic on the counter sees the results of all those performed
          before it, so no two get the same value back: k kept keys receive k different slots, 0 to k − 1, and the
          counter ends at k, which is the length of the output. See <a href={href('ch09', '9.2')}>section 9.2</a> for
          the call itself.
        </p>
      </Translated>
      <Source section="9.2" pages="208–209" />
      <Ask q={questionsS122['ch12/12.2/q-stale-counter']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Vì sao thứ tự thay đổi">Why the order changes</Translated></h2>
      <Translated
        vi={
          <p>
            Phần cứng có thể thực hiện các atomic trên một địa chỉ theo thứ tự bất kỳ, nên một luồng nhận ô kế tiếp nào đang có
            khi tới lượt nó, bất kể khóa của nó nằm ở đâu trong đầu vào. Chạy kernel hai lần và cùng những khóa đó ra theo một
            thứ tự khác: bộ lọc không ổn định.
          </p>
        }
      >
        <p>
          The hardware may perform the atomics on one address in any order, so a thread receives whatever slot is next
          when its turn comes, regardless of where its key sat in the input. Run the kernel twice and the same keys come
          out in a different order: the filter is unstable.
        </p>
      </Translated>
      <Source section="12.2" pages="291" />
      <Ask q={questionsS122['ch12/12.2/q-same-twice']} />

      <h2><Translated vi="Một bộ đếm cho mọi khóa được giữ">One counter for every kept key</Translated></h2>
      <Translated
        vi={
          <p>
            Tốc độ là vấn đề còn lại. Mọi khóa được giữ trong cả lưới (grid) đều cập nhật cùng một địa chỉ, và các atomic trên
            một địa chỉ bị tuần tự hóa. Mô hình của chương 9 đặt một con số cho điều đó: chỉ một atomic trên một vị trí có thể
            đang diễn ra, và mỗi cái tốn khoảng một độ trễ đọc cộng một độ trễ ghi, nên một vị trí duy nhất hấp thụ khoảng một
            atomic cho mỗi độ trễ đọc-cộng-ghi, bất kể bao nhiêu luồng đang chờ (<a href={href('ch09', '9.3')}>mục 9.3</a>).
          </p>
        }
      >
        <p>
          Speed is the other problem. Every kept key in the whole grid updates the same address, and atomics on one
          address are serialised. Chapter 9's model puts a number on it: only one atomic on a location can be in
          progress, and each takes about a load latency plus a store latency, so a single location absorbs roughly one
          atomic per load-plus-store latency, however many threads are waiting (<a href={href('ch09', '9.3')}>section 9.3</a>).
        </p>
      </Translated>
      <Source section="9.3" pages="209–210" />
      <Translated
        vi={
          <p>
            Một histogram ít ra còn rải atomic của nó ra nhiều bin. Một bộ lọc chỉ có một bộ đếm, và mọi khóa được giữ đều xếp
            hàng chờ nó. Sách đưa ra hai cách chữa, và cả hai đều cắt giảm số atomic tới được bộ đếm chung.{' '}
            <a href={href('ch12', '12.3')}>Mục 12.3</a> cho các lane của một warp thống nhất một tổng và gửi một atomic thay cho
            tất cả; <a href={href('ch12', '12.4')}>mục 12.4</a> cấp cho mỗi khối một bộ đếm riêng trong bộ nhớ chia sẻ (shared
            memory) và gửi một atomic cho mỗi khối. Mỗi cách tiết kiệm bao nhiêu phụ thuộc vào số khóa vượt qua và vị trí của chúng.
          </p>
        }
      >
        <p>
          A histogram at least spreads its atomics over many bins. A filter has one counter, and every kept key queues
          for it. The book offers two remedies, and both cut the number of atomics that reach the public counter.{' '}
          <a href={href('ch12', '12.3')}>Section 12.3</a> lets the lanes of a warp agree on a total and send one atomic
          for all of them; <a href={href('ch12', '12.4')}>section 12.4</a> gives each block a private counter in shared
          memory and sends one atomic per block. How much each saves depends on how many keys pass and where they sit.
        </p>
      </Translated>
      <Source section="12.2" pages="291" />
      <Predict
        id="ch12/12.2/p-atomics"
        prompt={
          <Translated
            vi={
              <>
                2<sup>20</sup> khóa, một nửa được giữ ngẫu nhiên, warp 32 lane. Một atomic cho mỗi warp phát ít hơn bao
                nhiêu lần số atomic so với một atomic cho mỗi khóa được giữ?
              </>
            }
          >
            2<sup>20</sup> keys, half of them kept at random, warps of 32 lanes. How many times fewer atomics does one
            atomic per warp issue than one atomic per kept key?
          </Translated>
        }
        unit={tr('times fewer', 'lần ít hơn')}
      >
        {(v) => (
          <Translated
            vi={
              <p>
                Một cho mỗi khóa được giữ là {count(half.perKey)} atomic. Khi một nửa số khóa được giữ, một warp không có khóa nào
                được giữ chỉ khi cả 32 lane đều bỏ khóa của mình, xác suất 1 trên 2<sup>32</sup>, nên về cơ bản cả{' '}
                {count(half.perWarp)} warp đều gửi một atomic: ít hơn {Math.round(half.perKey / half.perWarp)} lần.{' '}
                {v === 16
                  ? 'Đó đúng là con số của bạn.'
                  : v === 32
                    ? 'Không phải 32: điều đó đòi hỏi cả 32 lane của mọi warp đều giữ khóa của mình. Ở đây trung bình một warp mang 16 khóa được giữ.'
                    : v < 16
                      ? 'Nhiều hơn bạn dự đoán: ở 50 phần trăm gần như mọi warp đều bận, và mỗi warp trung bình mang 16 khóa được giữ.'
                      : 'Ít hơn bạn dự đoán: trung bình một warp mang 16 khóa được giữ, nên một atomic thay cho 16.'}{' '}
                Trình khám phá bên dưới bắt đầu từ thiết lập này.
              </p>
            }
          >
            <p>
              One per kept key is {count(half.perKey)} atomics. With half the keys kept, a warp has no kept key only if
              all 32 lanes drop theirs, a chance of 1 in 2<sup>32</sup>, so essentially all {count(half.perWarp)} warps
              send one: {Math.round(half.perKey / half.perWarp)} times fewer.{' '}
              {v === 16
                ? 'That is your number.'
                : v === 32
                  ? 'Not 32: that would need all 32 lanes of every warp to keep their key. Here a warp carries 16 kept keys on average.'
                  : v < 16
                    ? 'More than you expected: at 50 percent nearly every warp is busy, and each carries 16 kept keys on average.'
                    : 'Fewer than you expected: a warp carries 16 kept keys on average, so one atomic replaces 16.'}{' '}
              The explorer below starts from this setting.
            </p>
          </Translated>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Trình khám phá đếm số atomic tới được bộ đếm chung theo ba cách, cho đầu vào bạn chọn. Bắt đầu ở 50 phần trăm, rồi
            kéo tỉ lệ về gần 0, và quan sát khoảng cách giữa đường liền và đường nét đứt.
          </p>
        }
      >
        <p>
          The explorer counts the atomics that reach the public counter under the three schemes, for the input you
          choose. Start at 50 percent, then drag the fraction towards zero, and watch the gap between the solid and the
          dashed curve.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Số đếm, không phải thời gian: không có bộ nhớ đệm (cache), hàng đợi hay độ trễ. Một trình biên dịch có thể đã gộp các
              atomic của một warp thành một (xem bên dưới), và khi đó phần cứng không bao giờ thấy số đếm của Fig. 12.2.
            </p>
          }
        >
          <p>
            Counts, not time: no caches, queues or latency. A compiler may already merge a warp's atomics into one (see
            below), and then the hardware never sees Fig. 12.2's count.
          </p>
        </Translated>
      </Simplification>
      <OneCounter />
      <Translated
        vi={
          <p>
            Ba điều cần mang theo. Với các khóa độc lập, một atomic cho mỗi warp thắng một atomic cho mỗi khóa tới tối đa bằng độ
            rộng warp, và chỉ khi phần lớn warp có khóa được giữ: 16 lần ở 50 phần trăm, 32 ở 100, và hầu như không hơn gì ở 1
            phần trăm, khi một warp có việc thường chỉ giữ một khóa. Một atomic cho mỗi khối không quan tâm tới tỉ lệ: N chia cho
            kích thước khối, kể cả với các khối không giữ gì. Và cách bố trí có ý nghĩa: với các dải liền nhau hoặc khóa đã sắp
            xếp, các khóa được giữ dùng chung warp, và cách theo warp lấy lại ưu thế ở tỉ lệ thấp.
          </p>
        }
      >
        <p>
          Three things to take away. With independent keys, one atomic per warp beats one per key by up to the warp
          width, and only when most warps have a kept key: 16 times at 50 percent, 32 at 100, and hardly at all at 1
          percent, where a busy warp usually holds a single kept key. One atomic per block ignores the fraction: N
          divided by the block size, even for blocks that kept nothing. And the layout matters: with runs or sorted keys
          the kept keys share warps, and the warp scheme recovers at low fractions.
        </p>
      </Translated>
      <Ask q={questionsS122['ch12/12.2/q-lowfraction']} />
      <Translated
        vi={
          <p>
            Phần cứng của bạn có thấy các số đếm này hay không là một câu hỏi khác. Sách nói các trình biên dịch đã làm việc gộp
            của mục 12.3 (p. 294), và các nhà cung cấp ghi lại điều đó không đồng đều. Lab 12.1 trả lời bằng cách đo: nó đo thời
            gian ba kernel trên thiết bị của bạn ở vài tỉ lệ. Hãy ghi lại cái nào bạn dự đoán sẽ thắng ở 1 phần trăm trước khi
            chạy; <a href={href('ch12', 'labs')}>trang lab</a> có chi tiết.
          </p>
        }
      >
        <p>
          Whether your hardware sees these counts is another question. The book says compilers already do section
          12.3's aggregation (p. 294), and the vendors document that unevenly. Lab 12.1 answers it by measurement: it
          times the three kernels on your device at several fractions. Write down which one you expect to win at 1
          percent before you run it; the <a href={href('ch12', 'labs')}>labs page</a> has the details.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch12-facts.md A7, B4)', '2026-11-20 (notes/today/ch12-facts.md, mục A7, B4)')}>
        <Translated
          vi={
            <p>
              NVIDIA: một bài blog năm 2017, không kèm điều kiện nào. AMD: một pass LLVM upstream, ở <code>-O1</code> trở lên; mặc
              định của ROCm chưa được xác nhận. Apple: không có gì.
            </p>
          }
        >
          <p>
            NVIDIA: a 2017 blog post, with no conditions. AMD: an upstream LLVM pass, at <code>-O1</code> and above;
            ROCm's default not confirmed. Apple: nothing.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Cùng kernel đó trên HIP và Metal">The same kernel on HIP and Metal</Translated></h2>
      <Translated
        vi={
          <p>
            Chuyển tab trong đoạn mã. Trên HIP, cách viết CUDA <code>cuda::atomic_ref</code> có sẵn qua libhipcxx của AMD. Bản
            build HIP của khóa học liên kết <code>libhipcxx::libhipcxx</code> khi CMake tìm thấy nó (nó đi kèm ROCm 10.1 trở về
            sau, infra-facts B2). Không có nó, mã include <code>&lt;cuda/atomic&gt;</code> không build được. Có nó, thì cũng chưa
            máy ROCm nào build mã của khóa học này (unverified). Vì vậy lab dùng <code>atomicAdd</code> kiểu cũ trên HIP, thứ mà
            HIP ghi trong tài liệu cho bộ nhớ chia sẻ và bộ nhớ toàn cục (B1), relaxed và có phạm vi device như lời gọi của sách,
            và không cần thư viện thêm. Trên Metal, bộ đếm phải được khai báo với một kiểu atomic,{' '}
            <code>device atomic_uint*</code>, và lời gọi là{' '}
            <code>atomic_fetch_add_explicit(counter, 1, memory_order_relaxed)</code>; nó trả về giá trị cũ, như trên CUDA.
          </p>
        }
      >
        <p>
          Switch tabs in the listing. On HIP, the CUDA spelling of <code>cuda::atomic_ref</code> exists through AMD's
          libhipcxx. The course's HIP build links <code>libhipcxx::libhipcxx</code> when CMake finds it (it ships with
          ROCm 10.1 and later, infra-facts B2). Without it, code that includes <code>&lt;cuda/atomic&gt;</code> does not
          build. With it, no ROCm machine has built this course's code yet (unverified). So the lab uses the legacy{' '}
          <code>atomicAdd</code> on HIP, which HIP documents for shared and global memory (B1), relaxed and
          device-scoped like the book's call, and which needs no extra library. On Metal,
          the counter must be declared with an atomic type, <code>device atomic_uint*</code>, and the call is{' '}
          <code>atomic_fetch_add_explicit(counter, 1, memory_order_relaxed)</code>; it returns the old value, as on CUDA.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              HIP và Metal không có trong sách. Đã đối chiếu với tài liệu và header của nhà cung cấp: notes/today/
              infra-facts.md B1, B2, C1 và ch09-facts.md B1 (atomicAdd là relaxed và có phạm vi device trên HIP).
            </p>
          }
        >
          <p>
            HIP and Metal are not in the book. Checked against vendor documentation and headers: notes/today/
            infra-facts.md B1, B2, C1 and ch09-facts.md B1 (atomicAdd is relaxed and device-scoped on HIP).
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Kernel này không chỉ là một bài tập trong sách giáo khoa. Top-k CUDA của llama.cpp, nhánh radix mà nó dùng cho nhiều
            hàng cùng lúc, gom những phần tử thắng rõ ràng của mỗi hàng đúng theo cách này:{' '}
            <code>atomicAdd(&amp;state-&gt;greater_count, 1)</code>{' '}
            và một lần ghi tại vị trí trả về, mỗi hàng một bộ đếm. Bộ đếm của mỗi hàng nằm trên một dòng 128 byte riêng, để các
            hàng không tranh nhau một dòng. Vì vậy thứ tự của những phần tử thắng đó trong đầu ra là bất kỳ thứ tự nào mà các
            atomic đã xảy ra.
          </p>
        }
      >
        <p>
          This kernel is not only a textbook exercise. llama.cpp's CUDA top-k, the radix path it uses for many rows at
          once, gathers the clear winners of each row exactly this way: <code>atomicAdd(&amp;state-&gt;greater_count, 1)</code>{' '}
          and a store at the returned position, one counter per row. Each row's counter sits on its own 128-byte line,
          so that rows do not compete for one. The order of those winners in the output is therefore whatever order the
          atomics happened in.
        </p>
      </Translated>
      <Beyond title={tr('Not in the book: llama.cpp', 'Không có trong sách: llama.cpp')}>
        <Translated vi={<p>Từ ggml/src/ggml-cuda/top-k.cu, đọc ngày 2026-11-20 (ch12-facts D1).</p>}>
          <p>From ggml/src/ggml-cuda/top-k.cu, read on 2026-11-20 (ch12-facts D1).</p>
        </Translated>
      </Beyond>
    </>
  );
}
