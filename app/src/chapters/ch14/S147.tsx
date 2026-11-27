import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, realBackends, useReports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  LAB2,
  PROBLEMS,
  R_CHOICES,
  directTraffic,
  fastest,
  measuredLocal,
  passes,
  sweepR,
  tableEntries,
  type Problem,
  type Row,
} from '../../sim/radixcost';
import { BOOK_KEYS, bin, localPass, warpStores, summarise } from '../../sim/radixlocal';

const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const dec = (v: number) => (Math.round(v * 10) / 10).toFixed(1);
const big = (v: number) => v.toLocaleString('en-US');

function Seg<T extends string | number>({ label, value, options, onPick, show, disabled }: {
  label: string; value: T; options: readonly T[]; onPick: (v: T) => void; show?: (v: T) => string; disabled?: (v: T) => boolean;
}) {
  return (
    <span style={FIELD}>
      {label}
      <span className="seg" role="group" aria-label={label}>
        {options.map((o) => (
          <button key={String(o)} aria-pressed={value === o} disabled={disabled?.(o)} onClick={() => onPick(o)}>
            {show ? show(o) : String(o)}
          </button>
        ))}
      </span>
    </span>
  );
}

/** The digit widths lab 14.2's benchmark times at one key per thread (labs/ch14/lab2_multibit/main.cpp). */
const TIMED_R = [1, 2, 4, 6, 8];

/** Library choices of r (ch14-facts A4, B3, D6): tunings, not rules. */
const CHOSEN: Record<number, string> = {
  get 6() {
    return tr('CUB, single tile', 'CUB, một tile');
  },
  get 7() {
    return tr('CUB, multi-pass', 'CUB, nhiều lượt');
  },
  8: 'CUB OneSweep; rocPRIM; PyTorch MPS', // i18n-ignore: library names only
};

// ---------------------------------------------------------------- lab 14.2's measurements
function useMeasured() {
  const { reports } = useReports();
  return realBackends(reports)
    .map((b) => {
      const rows = benchRows(reports[b], LAB2);
      const byR = Object.fromEntries(R_CHOICES.map((r) => [r, measuredLocal(rows, r, 1)]));
      return { b, device: String(reports[b].device.name ?? b), byR };
    })
    .filter((d) => Object.values(d.byR).some((v) => v != null));
}

// ---------------------------------------------------------------- a small line chart over r
const XS = R_CHOICES;
function Chart({ title, series, log, unit, floor, id }: {
  title: string; id: string;
  /** `inTitle`: the title already names the series, so no label is drawn on the line. */
  series: { label: string; values: (number | undefined)[]; dashed?: boolean; tier?: string; inTitle?: boolean }[];
  log?: boolean; unit: string; floor?: { value: number; label: string; left?: boolean };
}) {
  const W = 300, H = 170, L = 44, R = 10, T = 16, B = 28;
  const all = series.flatMap((s) => s.values.filter((v): v is number => v != null && v > 0)).concat(floor ? [floor.value] : []);
  if (!all.length) return null;
  let lo = Math.min(...all), hi = Math.max(...all);
  if (!log) lo = 0;
  if (hi === lo) hi = lo + 1;
  const f = (v: number) => (log ? Math.log10(v) : v);
  const y = (v: number) => T + (H - T - B) * (1 - (f(v) - f(lo)) / (f(hi) - f(lo)));
  const x = (i: number) => L + ((W - L - R) * (i < 8 ? i : 8.6)) / 8.6;
  const ticks = log
    ? Array.from({ length: Math.floor(Math.log10(hi)) - Math.ceil(Math.log10(lo)) + 1 }, (_, k) => 10 ** (Math.ceil(Math.log10(lo)) + k))
    : [lo, (lo + hi) / 2, hi];
  return (
    <figure style={{ margin: 0, flex: '1 1 17rem', minWidth: 0, maxWidth: '24rem' }} data-chart={id}>
      <figcaption className="ui" style={{ fontSize: '0.84rem', marginBottom: 2 }}><b>{title}</b></figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block' }} role="img" aria-label={title}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="var(--rule-soft)" />
            <text x={L - 4} y={y(t) + 3} fontSize="9" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>{fmt(t, 2)}</text>
          </g>
        ))}
        {XS.map((r, i) => (
          <text key={r} x={x(i)} y={H - 12} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{r}</text>
        ))}
        <text x={(L + W) / 2} y={H - 1} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>
          <Translated vi={<>độ rộng chữ số r (bit); {unit}</>}>digit width r (bits); {unit}</Translated>
        </text>
        <line x1={x(7) + 9} x2={x(8) - 9} y1={H - B} y2={H - B} stroke="var(--ink-3)" strokeDasharray="2 2" />
        {floor ? (
          <g>
            <line x1={L} x2={W - R} y1={y(floor.value)} y2={y(floor.value)} stroke="var(--ink)" strokeDasharray="4 3" />
            <text x={floor.left ? L + 3 : W - R} y={y(floor.value) - 3} fontSize="9" textAnchor={floor.left ? 'start' : 'end'} style={{ fill: 'var(--ink-2)' }}>{floor.label}</text>
          </g>
        ) : null}
        {series.map((s) => {
          const pts = s.values.map((v, i) => (v != null && v > 0 ? [x(i), y(v)] : null)).filter((p): p is number[] => !!p);
          const col = `var(--${s.tier ?? 'global'})`; // the keys and the table live in global memory; a measurement is drawn in ink
          return (
            <g key={s.label} data-series={s.label}>
              <polyline points={pts.map((p) => p.join(',')).join(' ')} fill="none" stroke={col} strokeWidth="1.8" strokeDasharray={s.dashed ? '5 3' : undefined} />
              {pts.map((p, k) => <circle key={k} cx={p[0]} cy={p[1]} r="2.6" fill={col} />)}
              {pts.length && !s.inTitle ? (
                <text x={pts[pts.length - 1][0] - 4} y={pts[pts.length - 1][1] - 6} fontSize="9" textAnchor="end" style={{ fill: col }}>{s.label}</text>
              ) : null}
            </g>
          );
        })}
      </svg>
    </figure>
  );
}

// ---------------------------------------------------------------- the explorer
const THREADS = [128, 256, 512, 1024];
type ProblemId = Problem['id'];

function BookExample() {
  const p = localPass(BOOK_KEYS, 0, 2, 4);
  const runs = summarise(warpStores(p, 32, 4, 32));
  const runs1 = summarise(warpStores(localPass(BOOK_KEYS, 0, 1, 4), 32, 4, 32));
  const ones = p.blocks.flatMap((b) => b.counts).filter((c) => c === 1).length;
  const words = ['none', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen'];
  const names = ['00', '01', '10', '11'];
  return (
    <div className="panel" style={{ overflowX: 'auto' }} data-book-example>
      <p style={{ margin: '0 0 0.5rem' }}>
        <Translated vi={<><b>Ví dụ 2 bit của sách</b> (Hình 14.11 và 14.12): mười sáu khóa 4 bit, bốn khối bốn luồng, lượt đầu.</>}>
          <b>The book's 2-bit example</b> (Figs. 14.11 and 14.12): sixteen 4-bit keys, four blocks of four threads, first pass.
        </Translated>
      </p>
      <table className="data" style={{ margin: '0 0 0.6rem', minWidth: '30rem' }}>
        <thead>
          <tr>
            <th><Translated vi="khối">block</Translated></th>
            {p.blocks.map((b) => <th key={b.block}>{b.block}</th>)}
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="sắp cục bộ theo hai bit thấp">sorted locally by the low two bits</Translated></td>
            {p.blocks.map((b) => <td key={b.block} className="nowrap" data-local={b.block}>{b.sorted.map((k) => bin(k, 4)).join(' ')}</td>)}
          </tr>
          {names.map((nm, d) => (
            <tr key={nm}>
              <td><Translated vi={<>#{nm}: số đếm, rồi vị trí bắt đầu</>}>#{nm}: count, then start</Translated></td>
              {p.blocks.map((b) => (
                <td key={b.block} className="num" data-cell={`${d}-${b.block}`}>
                  {b.counts[d]} <span className="muted">→</span> <b>{p.starts[b.block][d]}</b>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted" style={{ margin: 0, fontSize: '0.86rem' }} data-book-runs={runs.meanRuns}>
        <Translated
          vi={
            <>
              {ones} trong mười sáu bucket cục bộ chứa một khóa, nên lần ghi bốn khóa của mỗi khối tạo trung bình{' '}
              {dec(runs.meanRuns)} dải, trong khi lượt 1 bit của Hình 14.8 tạo {dec(runs1.meanRuns)}.
            </>
          }
        >
          {words[ones][0].toUpperCase() + words[ones].slice(1)} of the sixteen local buckets hold one key, so each block's store
          of four keys makes {dec(runs.meanRuns)} runs on average, where the 1-bit pass of Fig. 14.8 made {dec(runs1.meanRuns)}.
        </Translated>
      </p>
    </div>
  );
}

function RadixWidth() {
  const [pid, setPid] = useState<ProblemId>('n20');
  const [bits, setBits] = useState<32 | 16>(32);
  const [threads, setThreads] = useState(256);
  const [book, setBook] = useState(false);
  const problem = PROBLEMS.find((p) => p.id === pid)!;
  const rows: Row[] = useMemo(() => sweepR(problem, bits, threads), [problem, bits, threads]);
  const ref = directTraffic(problem, bits);
  const measured = useMeasured();
  const minPerKey = Math.min(...rows.map((r) => r.traffic.perKey));
  const bestTraffic = rows.find((r) => r.traffic.perKey === minPerKey)!.r;

  return (
    <Bench
      title={tr('Radix width', 'Độ rộng radix')}
      task={
        <Translated vi="Đọc dọc bảng khi r tăng: số lượt và số byte giảm, số dải tăng, bảng lớn dần. Rồi đổi bài toán, độ rộng khóa và kích thước khối.">
          Read down the table as r grows: passes and bytes fall, runs climb, the table grows. Then change the problem, the key width and the block size.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Mô hình: src/sim/radixcost.ts trên src/sim/radixlocal.ts. Một khóa mỗi luồng (mục 14.8 thay đổi điều này). Số
              byte được đếm trên mỗi khóa cho cả phép sắp xếp: mỗi lượt, kernel đếm đọc các khóa và lần scatter đọc và ghi các
              cặp (khóa, giá trị 4 byte), và mỗi mục của bảng được chạm bốn lần; của khóa học này, không phải số đo. Số dải và
              số sector được lấy trung bình qua mọi lượt của một mẫu có seed gồm {rows[0].local.n.toLocaleString('en-US')} khóa
              trong {rows[0].local.blocks} khối, warp 32 lane, sector 32 byte của NVIDIA.
            </>
          }
        >
          <>
            Model: src/sim/radixcost.ts on src/sim/radixlocal.ts. One key per thread (section 14.8 varies it). Bytes are counted
            per key over the whole sort: per pass the count kernel reads the keys and the scatter reads and writes (key, 4-byte
            value) pairs, and each table entry is touched four times; ours, not measured. Runs and sectors are averaged over every
            pass of a seeded sample of {rows[0].local.n.toLocaleString('en-US')} keys in {rows[0].local.blocks} blocks, 32-lane warps,
            NVIDIA's 32-byte sectors.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <Seg label={tr('Problem', 'Bài toán')} value={pid} options={PROBLEMS.map((p) => p.id)} show={(id) => tr(PROBLEMS.find((p) => p.id === id)!.label, id === 'n20' ? '2²⁰ khóa ngẫu nhiên' : id === 'n24' ? '2²⁴ khóa ngẫu nhiên' : id === 'row' ? 'một hàng 151,936 token' : '64 hàng như thế')}
          onPick={(v) => { setPid(v); setBook(false); }} />
        <Seg label={tr('Key width', 'Độ rộng khóa')} value={bits} options={[32, 16] as (32 | 16)[]} show={(b) => (b === 32 ? tr('32 bits', '32 bit') : tr('16 bits (bf16)', '16 bit (bf16)'))} onPick={setBits} />
        <Seg label={tr('Threads per block', 'Số luồng mỗi khối')} value={threads} options={THREADS} onPick={setThreads} />
        <span style={FIELD}>
          <Translated vi="Thiết lập sẵn">Preset</Translated>
          <span className="seg" role="group" aria-label={tr('Preset', 'Thiết lập sẵn')}>
            <button aria-pressed={book} onClick={() => setBook(!book)}><Translated vi="ví dụ 2 bit của sách">the book's 2-bit example</Translated></button>
          </span>
        </span>
      </div>

      {book ? <BookExample /> : null}

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', margin: book ? '1rem 0' : '0 0 1rem' }}>
        <Chart
          id="bytes"
          title={tr('Bytes moved per key, whole sort', 'Số byte di chuyển mỗi khóa, cả phép sắp xếp')}
          unit={tr('log scale', 'thang log')}
          log
          series={[{ label: tr('local design', 'thiết kế cục bộ'), values: rows.map((r) => r.traffic.perKey) }]}
          floor={{ value: ref.perKey, label: tr("14.5's one-bit sort", 'phép sắp xếp một bit của 14.5'), left: true }}
        />
        <Chart
          id="table"
          title={tr('Table entries per key', 'Số mục bảng mỗi khóa')}
          unit={tr('log scale', 'thang log')}
          log
          series={[{ label: tr('table', 'bảng'), values: rows.map((r) => r.tableToKeys) }]}
          floor={{ value: 1, label: tr('as many entries as keys', 'số mục bằng số khóa'), left: true }}
        />
        <Chart
          id="sectors"
          title={tr('Sectors per warp store (sampled)', 'Số sector mỗi lần ghi của warp (lấy mẫu)')}
          unit={tr('32-byte sectors', 'sector 32 byte')}
          series={[{ label: tr('local design', 'thiết kế cục bộ'), values: rows.map((r) => r.local.meanUnits) }]}
          floor={{ value: rows[0].local.meanMin, label: tr('one coalesced store', 'một lần ghi được hợp nhất') }}
        />
        {measured.map((d) => (
          <Chart
            key={d.b}
            id={`measured-${d.b}`}
            title={tr(`Measured on ${d.device} (${d.b}), lab 14.2: local sort, c = 1`, `Đo trên ${d.device} (${d.b}), lab 14.2: sắp xếp cục bộ, c = 1`)}
            unit={tr('milliseconds', 'mili giây')}
            series={[{ label: tr('local, c = 1', 'cục bộ, c = 1'), inTitle: true, values: R_CHOICES.map((r) => (d.byR[r] != null ? d.byR[r]! * 1000 : undefined)), tier: 'ink' }]}
          />
        ))}
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0, minWidth: '46rem', background: 'var(--page)' }} aria-label={tr('Costs by digit width', 'Chi phí theo độ rộng chữ số')}>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.5rem' }}>r</th>
              <th className="num"><Translated vi="số lượt">passes</Translated></th>
              <th className="num"><Translated vi="số lần khởi chạy">launches</Translated></th>
              <th className="num"><Translated vi="byte mỗi khóa">bytes per key</Translated></th>
              <th className="num"><Translated vi="số mục bảng">table entries</Translated></th>
              <th className="num"><Translated vi="bảng ÷ khóa">table ÷ keys</Translated></th>
              <th className="num"><Translated vi="khóa mỗi bucket cục bộ">keys per local bucket</Translated></th>
              <th className="num"><Translated vi="dải mỗi lần ghi">runs per store</Translated></th>
              <th className="num"><Translated vi="sector mỗi lần ghi">sectors per store</Translated></th>
              {measured.map((d) => <th key={d.b} className="num">{d.b}, ms{/* i18n-ignore: backend name and unit */}</th>)}
              <th><Translated vi="được chọn bởi">chosen by</Translated></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.r} data-r={r.r}>
                <td style={{ paddingLeft: '0.5rem' }}><b>{r.r}</b></td>
                <td className="num" data-col="passes">{r.passes}</td>
                <td className="num" data-col="launches">{r.launches}</td>
                <td className="num" data-col="bytes">{r.r === bestTraffic ? <b>{fmt(r.traffic.perKey)}</b> : fmt(r.traffic.perKey)}</td>
                <td className="num" data-col="table">{big(r.tableEntries)}</td>
                <td className="num" data-col="ratio" style={{ color: r.tableToKeys > 1 ? 'var(--fail)' : undefined }}>{fmt(r.tableToKeys)}</td>
                <td className="num" data-col="bucket">{fmt(r.keysPerBucket)}</td>
                <td className="num" data-col="runs">{dec(r.local.meanRuns)}</td>
                <td className="num" data-col="sectors">{dec(r.local.meanUnits)}</td>
                {measured.map((d) => <td key={d.b} className="num" data-measured={`${d.b}-${r.r}`}>{d.byR[r.r] != null ? fmt(d.byR[r.r]! * 1000) : <span className="muted"><Translated vi="không có dòng">no row</Translated></span>}</td>)}
                <td className="muted" style={{ fontSize: '0.84rem' }}>{CHOSEN[r.r] ?? ''}</td>
              </tr>
            ))}
            <tr data-r="ref">
              <td style={{ paddingLeft: '0.5rem' }} className="muted" colSpan={3}><Translated vi="phép sắp xếp một bit của mục 14.5">section 14.5's one-bit sort</Translated></td>
              <td className="num">{fmt(ref.perKey)}</td>
              <td colSpan={6 + measured.length} className="muted">
                <Translated vi="một lần scan qua mọi khóa ở mọi lượt thay cho một bảng">a scan over every key in every pass instead of a table</Translated>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      {!measured.length ? (
        <Translated
          vi={
            <p className="ui" style={{ margin: '0.7rem 0 0' }} data-measured="none">
            Máy này chưa có số đo nào của lab 14.2. Hãy chạy <code>python3 labs/pmpp.py run 14.2 --solution --bench</code>, rồi tải
            lại: một cột và một biểu đồ thời gian đo được sẽ hiện ra cho mỗi thiết bị.
            </p>
          }
        >
          <p className="ui" style={{ margin: '0.7rem 0 0' }} data-measured="none">
            No lab 14.2 measurement on this machine yet. Run <code>python3 labs/pmpp.py run 14.2 --solution --bench</code>, then
            reload: a column and a chart of measured times appear for each device.
          </p>
        </Translated>
      ) : (
        <Translated
          vi={
            <p className="ui" style={{ margin: '0.7rem 0 0' }}>
            Các cột mili giây là số đo của lab 14.2 trên các thiết bị của máy này, ở kích thước bài toán riêng của lab; mọi thứ khác
            là mô hình.
            </p>
          }
        >
          <p className="ui" style={{ margin: '0.7rem 0 0' }}>
            The millisecond columns are measurements of lab 14.2 on this machine's devices, at its own problem size; everything
            else is the model.
          </p>
        </Translated>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const N22 = { rows: 1, rowLength: 2 ** 22 };
const BOOK2 = localPass(BOOK_KEYS, 0, 2, 4);

export const questionsS147: Record<string, Question> = {
  'ch14/14.7/q-passes': {
    kind: 'numeric',
    id: 'ch14/14.7/q-passes',
    prompt: (
      <Translated vi={<>Bạn sắp xếp logit bf16 như các khóa 16 bit với chữ số 6 bit. Phép sắp xếp chạy bao nhiêu lượt?</>}>
        <>You sort bf16 logits as 16-bit keys with a 6-bit digit. How many passes does the sort make?</>
      </Translated>
    ),
    answer: passes(16, 6),
    unit: 'passes',
    vi: { unit: 'lượt', source: 'ch. 14 §14.7, p. 342; khóa float 16 bit: ch14-facts A2' },
    explain: (
      <Translated
        vi={
          <p>
          ceil(16 / 6) = {passes(16, 6)}: các bit 0 đến 5, 6 đến 11, và 12 đến 15. Lượt cuối chỉ có bốn bit để làm việc, nên nó
          dùng 16 trong 64 bucket của mình. Với r = 8, cùng các khóa đó cần {passes(16, 8)} lượt; phép sắp xếp MPS của PyTorch chọn
          8 bit cho khóa 16 bit (ch14-facts D6).
          </p>
        }
      >
        <p>
          ceil(16 / 6) = {passes(16, 6)}: bits 0 to 5, 6 to 11, and 12 to 15. The last pass has only four bits to work on, so
          it uses 16 of its 64 buckets. At r = 8 the same keys take {passes(16, 8)} passes; PyTorch's MPS sort picks 8 bits for
          16-bit keys (ch14-facts D6).
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.7, p. 342; 16-bit float keys: ch14-facts A2',
  },
  'ch14/14.7/q-table': {
    kind: 'numeric',
    id: 'ch14/14.7/q-table',
    prompt: (
      <Translated vi={<>2<sup>22</sup> khóa, 256 luồng mỗi khối, một khóa mỗi luồng, r = 8. Bảng bucket có bao nhiêu mục?</>}>
        <>2<sup>22</sup> keys, 256 threads per block, one key per thread, r = 8. How many entries does the bucket table have?</>
      </Translated>
    ),
    answer: tableEntries(N22, 8, 256, 1),
    unit: 'entries',
    vi: { unit: 'mục' },
    explain: (
      <Translated
        vi={
          <p>
          2<sup>22</sup> / 256 = 16,384 khối, mỗi khối 256 bucket: {big(tableEntries(N22, 8, 256, 1))} mục, đúng bằng số khóa.
          Lúc này lần scan bảng tốn xấp xỉ bằng việc di chuyển các khóa, ở mọi lượt. Đó là chi phí thứ hai của một radix lớn mà
          sách nêu.
          </p>
        }
      >
        <p>
          2<sup>22</sup> / 256 = 16,384 blocks, each with 256 buckets: {big(tableEntries(N22, 8, 256, 1))} entries, exactly
          as many as there are keys. The scan of the table now costs about as much as moving the keys, in every pass. That is
          the book's second cost of a large radix.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.7, pp. 343–344',
  },
  'ch14/14.7/q-2bit': {
    kind: 'numeric',
    id: 'ch14/14.7/q-2bit',
    prompt: (
      <Translated
        vi={
          <>
          Ví dụ 2 bit của sách có các số đếm bucket sau cho các khối 0 đến 3: #00 = 1 1 0 2, #01 = 1 1 2 0, #10 = 1 1 1 1,
          #11 = 1 1 1 1. Bucket 10 của khối 2 bắt đầu ở đâu?
          </>
        }
      >
        <>
          The book's 2-bit example has these bucket counts for blocks 0 to 3: #00 = 1 1 0 2, #01 = 1 1 2 0, #10 = 1 1 1 1,
          #11 = 1 1 1 1. Where does block 2's 10 bucket start?
        </>
      </Translated>
    ),
    answer: BOOK2.starts[2][2],
    explain: (
      <Translated
        vi={
          <p>
          Mọi khóa có chữ số nhỏ hơn đứng trước: 4 khóa ở 00 và 4 khóa ở 01. Rồi đến các khóa 10 của những khối đứng trước khối 2:
          một ở khối 0, một ở khối 1. 4 + 4 + 1 + 1 = {BOOK2.starts[2][2]}, mục thứ ba của hàng 10 trong Hình 14.12.
          </p>
        }
      >
        <p>
          Everything with a smaller digit comes first: 4 keys in 00 and 4 in 01. Then the 10 keys of the blocks before block 2:
          one in block 0, one in block 1. 4 + 4 + 1 + 1 = {BOOK2.starts[2][2]}, the third entry of the row for 10 in Fig. 14.12.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.7, Fig. 14.12, pp. 343–344',
  },
  'ch14/14.7/q-sixteen': {
    kind: 'choice',
    id: 'ch14/14.7/q-sixteen',
    prompt: (
      <Translated vi={<>Sao không dùng r = 16, và sắp khóa 32 bit trong hai lượt?</>}>
        <>Why not use r = 16, and sort 32-bit keys in two passes?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Hai lượt 16 bit không thể ổn định: với 65,536 bucket, phép sắp xếp cục bộ không giữ được các chữ số bằng nhau theo thứ tự đầu vào.">
            Two 16-bit passes cannot be stable: with 65,536 buckets the local sort cannot keep equal digits in input order.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi lượt là một phép phân hoạch ổn định dù độ rộng là bao nhiêu; hai lượt ổn định sắp đúng khóa 32 bit. Vấn đề là chi phí, không phải tính đúng đắn.">
            Each pass is a stable partition whatever its width; two stable passes sort 32-bit keys correctly. The problem is cost, not correctness.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một lượt chỉ sắp được chữ số vừa trong một byte, nên khóa 32 bit cần ít nhất bốn lượt.">
            A pass can only sort a digit that fits in one byte, so 32-bit keys need at least four passes.
          </Translated>
        ),
        why: (
          <Translated vi="Không có gì giới hạn một chữ số trong một byte; 8 bit là một lựa chọn tinh chỉnh. Phép trích chữ số là cùng một biểu thức với mọi r.">
            Nothing limits a digit to a byte; 8 bits is a tuning choice. The digit extraction is the same expression for any r.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="65,536 bucket mỗi khối cho vài trăm khóa: không gì được hợp nhất, bảng lớn hơn dữ liệu, và các số đếm cần 256 KiB bộ nhớ chia sẻ.">
            65,536 buckets per block for a few hundred keys: nothing coalesces, the table outgrows the data, and the counts need 256 KiB of shared memory.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
          Cả hai chi phí của sách đều lên đến cực điểm. Với 256 luồng, mỗi luồng một khóa, 65,536 bucket của một khối chứa trung
          bình khoảng 0.004 khóa, nên mọi lần ghi đều là scatter; và 65,536 số đếm bốn byte là 256 KiB mỗi khối. Hãy so với giới
          hạn bộ nhớ chia sẻ mà <code>python3 labs/pmpp.py doctor</code> báo cho thiết bị của bạn, và đặt r bằng 16 trong trình
          khám phá.
          </p>
        }
      >
        <p>
          Both of the book's costs go to their extreme. With 256 threads and one key each, the 65,536 buckets of a block hold
          about 0.004 keys on average, so every store is a scatter; and 65,536 four-byte counts are 256 KiB per block. Compare
          that with the shared-memory limit <code>python3 labs/pmpp.py doctor</code> reports for your device, and set r to 16 in
          the explorer.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.7, pp. 343–344',
  },
};

// ---------------------------------------------------------------- the section
export function S147() {
  const measured = useMeasured();
  return (
    <>
      <Translated
        vi={
          <p>
          Với một bit mỗi lượt, phép sắp xếp khóa 32 bit chạy 32 lượt, mỗi lượt là một chuyến đi trọn vẹn qua bộ nhớ toàn cục
          (global memory). Chữ số rộng hơn thì ít lượt hơn. Với chữ số r bit, một lượt phân các khóa vào 2<sup>r</sup> bucket, và
          khóa k bit cần ceil(k / r) lượt. Ví dụ của sách sắp các khóa 4 bit với chữ số 2 bit: trước theo hai bit thấp vào các
          bucket 00, 01, 10 và 11, rồi theo hai bit cao, mỗi lượt giữ thứ tự bên trong từng bucket.
          </p>
        }
      >
        <p>
          With one bit per pass, a sort of 32-bit keys makes 32 passes, each a full trip through global memory. A wider digit
          makes fewer. With an r-bit digit a pass distributes keys into 2<sup>r</sup> buckets, and keys of k bits need
          ceil(k / r) passes. The book's example sorts its 4-bit keys with a 2-bit digit: first by the low two bits into the
          buckets 00, 01, 10 and 11, then by the high two bits, each pass keeping the order inside every bucket.
        </p>
      </Translated>
      <Source section="14.7" pages="342" figs={tr('Fig. 14.10', 'Hình 14.10')} />
      <Translated
        vi={
          <p>
          Không gì khác trong thiết kế thay đổi: một luồng mỗi khóa, một phép sắp xếp cục bộ trong bộ nhớ chia sẻ (shared memory),
          rồi ghi theo từng bucket. Phép sắp xếp cục bộ một chữ số r bit là r lần tách một bit liên tiếp, mỗi lần có scan riêng
          trên khối và không cần phối hợp với các khối khác ở giữa. Sau r lần tách, các khóa của khối nằm trong 2<sup>r</sup>{' '}
          bucket cục bộ.
          </p>
        }
      >
        <p>
          Nothing else in the design changes: one thread per key, a local sort in shared memory, then bucket-by-bucket writes. The
          local sort of an r-bit digit is r one-bit splits in a row, each with its own scan over the block and no coordination
          with other blocks in between. After r splits the block's keys sit in 2<sup>r</sup> local buckets.
        </p>
      </Translated>
      <Source section="14.7" pages="342" figs={tr('Fig. 14.11', 'Hình 14.11')} />
      <Translated
        vi={
          <p>
          Bảng có một hàng cho mỗi giá trị chữ số, 2<sup>r</sup> hàng, mỗi hàng một mục cho mỗi khối, và vẫn được bố trí theo
          từng chữ số. Trong ví dụ của sách, bốn hàng sau khi scan cho vị trí bắt đầu 0, 1, 2, 2 với 00; 4, 5, 6, 8 với 01; 8, 9,
          10, 11 với 10; và 12, 13, 14, 15 với 11.
          </p>
        }
      >
        <p>
          The table gets a row per digit value, 2<sup>r</sup> rows, each with one entry per block, and it is still laid out
          digit by digit. In the book's example the four rows scan to starts 0, 1, 2, 2 for 00; 4, 5, 6, 8 for 01; 8, 9, 10, 11
          for 10; and 12, 13, 14, 15 for 11.
        </p>
      </Translated>
      <Source section="14.7" pages="342–343" figs={tr('Fig. 14.12', 'Hình 14.12')} />
      <Ask q={questionsS147['ch14/14.7/q-2bit']} />

      <h2><Translated vi="Chữ số rộng hơn đem lại gì, và tốn gì">What a wider digit buys, and what it costs</Translated></h2>
      <Translated
        vi={
          <p>
          Ít lượt hơn nghĩa là ít lần khởi chạy kernel hơn, ít chuyến đi của các khóa qua bộ nhớ toàn cục hơn, và ít lần scan bảng
          hơn. Đó là toàn bộ lợi ích.
          </p>
        }
      >
        <p>
          Fewer passes mean fewer kernel launches, fewer trips of the keys through global memory, and fewer scans of a table.
          That is the whole benefit.
        </p>
      </Translated>
      <Source section="14.7" pages="343" />
      <Translated
        vi={
          <p>
          Có hai chi phí. Mỗi khối giờ trải các khóa của nó ra nhiều bucket cục bộ hơn, nên mỗi bucket chứa ít khóa hơn và khối ghi
          nhiều dải hơn, ngắn hơn: trong Hình 14.11, phần lớn bucket cục bộ chỉ chứa một khóa. Và bảng có 2<sup>r</sup>{' '}
          mục mỗi khối, nên lần scan giữa các pha lớn lên theo. Sách dừng ở đó. Sách nói radix phải cân bằng hai điều này và không
          đưa ra giá trị nào.
          </p>
        }
      >
        <p>
          The costs are two. Each block now spreads its keys over more local buckets, so each bucket holds fewer keys and the
          block writes more, shorter runs: in Fig. 14.11 most local buckets hold a single key. And the table has 2<sup>r</sup>{' '}
          entries per block, so the scan between the phases grows with it. The book stops there. It says the radix must balance
          the two and gives no value.
        </p>
      </Translated>
      <Source section="14.7" pages="343–344" />
      <Predict
        id="ch14/14.7/p-best-r"
        prompt={
          <Translated vi="Lab 14.2 đo thời gian thiết kế sắp xếp cục bộ của nó ở một khóa mỗi luồng với r = 1, 2, 4, 6 và 8. Giá trị nào trong năm giá trị đó sẽ nhanh nhất trên thiết bị của bạn?">
            Lab 14.2 times its local-sort design at one key per thread for r = 1, 2, 4, 6 and 8. Which of those five will be fastest on your device?
          </Translated>
        }
        unit={tr('bits', 'bit')}
      >
        {(v) =>
          measured.length ? (
            <>
              {measured.map((d) => {
                const best = fastest(TIMED_R.map((r) => ({ at: r, seconds: d.byR[r] })));
                return best ? (
                  <p key={d.b} data-best-r={`${d.b}-${best.at}`}>
                    <Translated
                      vi={
                        <>
                          <b>{d.device}</b> ({d.b}): nhanh nhất ở <b>r = {best.at}</b>, {fmt(best.seconds * 1000)} ms
                          {best.at === v ? ', đúng như bạn nói' : `; bạn đã nói ${v}`}.
                        </>
                      }
                    >
                      <b>{d.device}</b> ({d.b}): fastest at <b>r = {best.at}</b>, {fmt(best.seconds * 1000)} ms
                      {best.at === v ? ', as you said' : `; you said ${v}`}.
                    </Translated>
                  </p>
                ) : null;
              })}
              <p className="muted">
                <Translated vi="Mô hình của trình khám phá không có đồng hồ. Hãy so chỗ số byte của nó ngừng giảm với chỗ thời gian của bạn ngừng giảm.">
                  The explorer's model has no clock. Compare where its bytes stop falling with where your times do.
                </Translated>
              </p>
            </>
          ) : (
            <Translated
              vi={
                <p data-measured="none">
                Máy này chưa có số đo nào. Hãy chạy <code>python3 labs/pmpp.py run 14.2 --solution --bench</code>, rồi tải lại trang
                này. Trong lúc đó, trình khám phá bên dưới cho thấy mỗi lựa chọn tốn bao nhiêu byte và sector.
                </p>
              }
            >
              <p data-measured="none">
                No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 14.2 --solution --bench</code>, then
                reload this page. Meanwhile, the explorer below shows what each choice costs in bytes and sectors.
              </p>
            </Translated>
          )
        }
      </Predict>
      <Translated
        vi={
          <p>
          Trình khám phá đặt các chi phí cạnh nhau cho mọi r từ 1 đến 8, và 16. Các biểu đồ cho thấy ba xu hướng; bảng đưa ra con
          số, với các độ rộng chữ số mà các phép sắp xếp thư viện chọn được đánh dấu ở bên phải.
          </p>
        }
      >
        <p>
          The explorer lays the costs side by side for every r from 1 to 8, and 16. The charts show the three trends; the table
          gives the numbers, with the digit widths that library sorts choose marked on the right.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
            Số byte được đếm từ mô hình, không phải đo: không bộ nhớ đệm (cache), không occupancy (mức chiếm dụng), không thời gian.
            Số dải và sector đến từ các khóa lấy mẫu, cỡ vài khối. Chỉ các cột được đánh dấu là đo mới đến từ thiết bị của bạn.
            </p>
          }
        >
          <p>
            Bytes are counted from the model, not measured: no caches, no occupancy, no time. Runs and sectors come from sampled
            keys, a few blocks' worth. Only the columns marked as measured come from your device.
          </p>
        </Translated>
      </Simplification>
      <RadixWidth />
      <Translated
        vi={
          <p>
          Đọc cột byte từ trên xuống. Số byte mỗi khóa giảm gần như 1/r khi các khóa còn chiếm phần lớn, rồi đi ngang và tăng trở
          lại khi bảng đuổi kịp dữ liệu; ở r = 16 chúng bùng nổ. Trong khi đó số dải mỗi lần ghi tăng dần về phía độ rộng warp, vì
          các bucket cục bộ co lại dưới lượng khóa của một warp (warp của trình khám phá có 32 lane). Giá trị r rẻ nhất về byte
          không phải rẻ nhất về sector, và cả hai đều không phải thời gian. Độ rộng chữ số tốt nhất nằm đâu đó ở giữa, và chỉ thiết
          bị mới cho biết ở đâu.
          </p>
        }
      >
        <p>
          Read the bytes column from the top. Bytes per key fall almost as 1/r while the keys dominate, then flatten and turn up
          once the table catches the data; at r = 16 they explode. Meanwhile the runs per store climb towards the warp width,
          because the local buckets shrink below one warp's worth of keys (the explorer's warps have 32 lanes). The cheapest r in bytes is not the cheapest in sectors, and
          neither is a time. Somewhere in between is the best digit width, and only the device can say where.
        </p>
      </Translated>
      <Today checked="2026-11-20">
        <Translated
          vi={
            <p>
            Các độ rộng chữ số đã tinh chỉnh của CUB cho khóa rộng hơn một byte là 8 bit trong OneSweep, 7 trong đường nhiều lượt và
            6 cho một tile duy nhất (ch14-facts A4). Các cấu hình OneSweep đã tinh chỉnh của rocPRIM dùng 8 trong 374 trên 384 (B3).
            Phép sắp xếp MPS của PyTorch dùng 8 cho khóa 16 và 32 bit; chú thích trong mã nguồn của nó nói điều đó giảm một nửa lưu
            lượng toàn cục so với 4 bit cho cùng lượng việc cục bộ trong khối (D6). <code>BlockRadixSort</code> của hipCUB nhận một
            tham số <code>RADIX_BITS</code> rồi bỏ qua nó (B3). Đây là các tinh chỉnh cho những thiết bị cụ thể, không phải quy tắc.
            </p>
          }
        >
          <p>
            CUB's tuned digit widths for keys wider than a byte are 8 bits in OneSweep, 7 in its multi-pass path and 6 for a
            single tile (ch14-facts A4). rocPRIM's tuned OneSweep configurations use 8 in 374 of 384 (B3). PyTorch's MPS sort uses
            8 for 16- and 32-bit keys; its source comment says that halves global traffic against 4 bits for the same block-local
            work (D6). hipCUB's <code>BlockRadixSort</code> accepts a <code>RADIX_BITS</code> parameter and ignores it (B3).
            These are tunings for particular devices, not rules.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS147['ch14/14.7/q-passes']} />
      <Ask q={questionsS147['ch14/14.7/q-table']} />
      <Ask q={questionsS147['ch14/14.7/q-sixteen']} />

      <h2><Translated vi="Chỗ điều này gặp logit bf16">Where this meets bf16 logits</Translated></h2>
      <Translated
        vi={
          <p>
          Một bộ lấy mẫu giữ logit ở dạng bf16 có thể sắp chúng như khóa 16 bit: phép biến đổi float của mục 14.4 hoạt động trên
          16 bit như trên 32 bit. Với chữ số 8 bit, đó là hai lượt thay vì bốn, mỗi lượt gồm một lần đếm 256 bin theo từng khối
          (histogram được riêng tư hóa (privatization) của <a href={href('ch09', '9.4')}>chương 9</a>), một lần scan bảng và một
          lần scatter. Đặt độ rộng khóa của trình khám phá thành 16 bit và đọc hàng r = 8: một nửa số lượt, và mỗi lượt vẫn ghi
          và scan một bảng 256 mục mỗi khối trong khi mỗi khóa mang ít byte hơn, nên bảng chiếm phần lớn hơn trong những gì còn
          lại.
          </p>
        }
      >
        <p>
          A sampler that holds its logits in bf16 can sort them as 16-bit keys: section 14.4's float transform works on 16
          bits as it does on 32. With 8-bit digits that is two passes instead of four, each a per-block 256-bin count (the
          privatised histogram of <a href={href('ch09', '9.4')}>chapter 9</a>), a scan of the table and a scatter. Set the
          explorer's key width to 16 bits and read the r = 8 row: half the passes, and each pass still writes and scans a
          table of 256 entries per block while every key carries fewer bytes, so the table is a larger share of what is left.
        </p>
      </Translated>
      <Beyond title={tr('Ours', 'Của khóa học này')}>
        <Translated
          vi={
            <p>
            Ví dụ bf16 là của khóa học này; khóa của sách là số nguyên không dấu. CUB liệt kê <code>__nv_bfloat16</code> trong các
            kiểu khóa của radix sort (ch14-facts A2), và phép sắp xếp MPS của PyTorch dùng chữ số 8 bit cho khóa 16 bit (D6).
            </p>
          }
        >
          <p>
            The bf16 example is ours; the book's keys are unsigned integers. CUB lists <code>__nv_bfloat16</code> among its
            radix sort's key types (ch14-facts A2), and PyTorch's MPS sort uses 8-bit digits for 16-bit keys (D6).
          </p>
        </Translated>
      </Beyond>
      <Aside>
        <Translated
          vi={
            <p>
            Lab 14.2 giải các bài tập 1 đến 3 (p. 347); độ rộng chữ số của nó là r của trang này. Xem{' '}
            <a href={href('ch14', 'labs')}>trang lab</a>.
            </p>
          }
        >
          <p>
            Lab 14.2 works exercises 1 to 3 (p. 347); its digit width is the r of this page. See the{' '}
            <a href={href('ch14', 'labs')}>labs page</a>.
          </p>
        </Translated>
      </Aside>
    </>
  );
}
