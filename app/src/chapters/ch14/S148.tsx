import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, realBackends, useReports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  C_CHOICES,
  LAB2,
  PROBLEMS,
  VOCAB,
  fastest,
  measuredLocal,
  sweepC,
  tableEntries,
  tileBytes,
  type Problem,
  type Row,
} from '../../sim/radixcost';
import { BOOK_KEYS, bin, localPass, summarise, warpStores } from '../../sim/radixlocal';

const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const dec = (v: number) => (Math.round(v * 10) / 10).toFixed(1);
const big = (v: number) => v.toLocaleString('en-US');
const kib = (b: number) => (b % 1024 === 0 ? `${big(b / 1024)} KiB` : `${big(b)} B`); // i18n-ignore: units

function Seg<T extends string | number>({ label, value, options, onPick, show }: {
  label: string; value: T; options: readonly T[]; onPick: (v: T) => void; show?: (v: T) => string;
}) {
  return (
    <span style={FIELD}>
      {label}
      <span className="seg" role="group" aria-label={label}>
        {options.map((o) => (
          <button key={String(o)} aria-pressed={value === o} onClick={() => onPick(o)}>
            {show ? show(o) : String(o)}
          </button>
        ))}
      </span>
    </span>
  );
}

// ---------------------------------------------------------------- Fig. 14.13 on the book's keys
const CW = 40;
const X0 = 20;
const cx = (i: number) => X0 + i * CW + CW / 2;

function Fig1413() {
  const [c, setC] = useState(2);
  const p = localPass(BOOK_KEYS, 0, 2, 4, c);
  const stores = warpStores(p, 32, 4, 32);
  const s = summarise(stores);
  const per = 4 * c;
  const local = p.blocks.flatMap((b) => b.sorted);
  const rowY = { in: 30, local: 112, out: 214 };
  return (
    <Bench
      title={tr('Fig. 14.13, two ways', 'Hình 14.13, theo hai cách')}
      task={
        <Translated vi="Chuyển giữa một và hai khóa mỗi luồng và đếm các mũi tên đi song song.">
          Switch between one and two keys per thread and count the arrows that travel in parallel.
        </Translated>
      }
      foot={tr(
        "Model: localPass() in src/sim/radixlocal.ts, a 2-bit digit, the first pass. Which thread copies which slot when it owns two keys is our choice (the book's figure does not say): in store instruction k, thread t copies local slot 4k + t.",
        'Mô hình: localPass() trong src/sim/radixlocal.ts, chữ số 2 bit, lượt đầu. Luồng nào chép ô nào khi nó sở hữu hai khóa là lựa chọn của chúng tôi (hình của sách không nói): ở lệnh ghi thứ k, luồng t chép ô cục bộ 4k + t.',
      )}
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <Seg label={tr('Keys per thread', 'Số khóa mỗi luồng')} value={c} options={[1, 2]} show={(v) => (v === 1 ? tr('1 (Fig. 14.11)', '1 (Hình 14.11)') : tr('2 (Fig. 14.13)', '2 (Hình 14.13)'))} onPick={setC} />
      </div>
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox="0 0 680 262" style={{ width: '100%', minWidth: 560, maxWidth: 760, display: 'block' }} role="img"
          aria-label={tr(`Sixteen keys sorted locally by ${p.blocks.length} blocks and written out`, `Mười sáu khóa được ${p.blocks.length} khối sắp cục bộ rồi ghi ra`)} data-fig={c} data-runs={stores.map((w) => w.runs).join(' ')}>
          <defs>
            <marker id="s148-head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">{/* i18n-ignore: SVG attribute value */}
              <path d="M0,0 L8,4 L0,8 z" fill="var(--ink)" />
            </marker>
          </defs>
          <text x={X0} y={rowY.in - 7} fontSize="11" fontWeight="600" style={{ fill: 'var(--global)' }}>
            <Translated vi="đầu vào (bộ nhớ toàn cục), hai bit thấp được gạch chân">input (global memory), low two bits underlined</Translated>
          </text>
          {BOOK_KEYS.map((k, i) => (
            <g key={i}>
              <rect x={X0 + i * CW + 1} y={rowY.in} width={CW - 2} height={24} rx="2" fill="var(--global-wash)" stroke="var(--global)" />
              <text x={cx(i)} y={rowY.in + 16} fontSize="11" textAnchor="middle">{bin(k, 4).slice(0, 2)}<tspan fontWeight="700" textDecoration="underline">{bin(k, 4).slice(2)}</tspan></text>
            </g>
          ))}
          {p.blocks.map((b) => (
            <g key={b.block}>
              <rect x={X0 + b.first * CW - 2} y={rowY.local - 30} width={per * CW + 4} height={64} rx="5" fill="none" stroke="var(--ink-3)" strokeDasharray="4 3" />
              <text x={X0 + b.first * CW + 2} y={rowY.local - 34} fontSize="10" style={{ fill: 'var(--ink-2)' }}>
                <Translated vi={<>khối {b.block}: {per} khóa, 4 luồng</>}>block {b.block}: {per} keys, 4 threads</Translated>
              </text>
              {b.sorted.map((_, j) => (
                <text key={j} x={cx(b.first + j)} y={rowY.local - 8} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--reg)' }}>t{j % 4}</text>
              ))}
              {b.counts.map((n, d) => {
                if (!n) return null;
                const x1 = X0 + (b.first + b.offsets[d]) * CW + 4;
                const x2 = x1 + n * CW - 8;
                return <path key={d} d={`M${x1},${rowY.local + 28} v4 H${x2} v-4`} fill="none" stroke="var(--shared)" strokeWidth="1.3" />;
              })}
            </g>
          ))}
          {local.map((k, i) => (
            <g key={i} data-local={i}>
              <rect x={X0 + i * CW + 1} y={rowY.local} width={CW - 2} height={24} rx="2" fill="var(--shared-wash)" stroke="var(--shared)" />
              <text x={cx(i)} y={rowY.local + 16} fontSize="11" textAnchor="middle">{bin(k, 4)}</text>
            </g>
          ))}
          {p.writes.map((w) => (
            <line key={`${w.block}-${w.slot}`} x1={cx(p.blocks[w.block].first + w.slot)} y1={rowY.local + 36} x2={cx(w.addr)} y2={rowY.out - 3}
              stroke="var(--ink)" strokeWidth="1.3" markerEnd="url(#s148-head)" />
          ))}
          {p.out.map((k, i) => (
            <g key={i} data-out={i}>
              <rect x={X0 + i * CW + 1} y={rowY.out} width={CW - 2} height={24} rx="2" fill="var(--global-wash)" stroke="var(--global)" />
              <text x={cx(i)} y={rowY.out + 16} fontSize="11" textAnchor="middle">{bin(k, 4)}</text>
            </g>
          ))}
          <text x={X0} y={rowY.out + 40} fontSize="11" fontWeight="600" style={{ fill: 'var(--global)' }}>
            <Translated vi="đầu ra sau lượt (bộ nhớ toàn cục)">output after the pass (global memory)</Translated>
          </text>
        </svg>
      </div>
      <table className="data" style={{ margin: '0.7rem 0 0', maxWidth: '36rem' }} data-fig-summary={c}>
        <tbody>
          <tr><td><Translated vi="số khối">blocks</Translated></td><td className="num" data-sum="blocks">{p.blocks.length}</td></tr>
          <tr><td><Translated vi="số mục bảng (4 giá trị chữ số × số khối)">table entries (4 digit values × blocks)</Translated></td><td className="num" data-sum="table">{p.table.length}</td></tr>
          <tr><td><Translated vi="khóa mỗi bucket cục bộ">keys per local bucket</Translated></td><td className="num" data-sum="bucket">{fmt(per / 4)}</td></tr>
          <tr><td><Translated vi="số lệnh ghi, mỗi lệnh 4 luồng">store instructions, 4 threads each</Translated></td><td className="num" data-sum="stores">{s.stores}</td></tr>
          <tr><td><Translated vi="số dải địa chỉ liên tiếp, mọi lần ghi">runs of consecutive addresses, all stores</Translated></td><td className="num" data-sum="runs"><b>{stores.reduce((a, w) => a + w.runs, 0)}</b></td></tr>
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- the explorer over c
const THREADS = [128, 256, 512, 1024];

interface Limit { key: string; label: string; bytes: number }
function useLimits(): Limit[] {
  const { reports } = useReports();
  const limits = realBackends(reports).flatMap((b) => {
    const d = reports[b].device;
    const field = 'maxThreadgroupMemoryLength' in d ? 'maxThreadgroupMemoryLength' : 'sharedMemPerBlock';
    const v = Number(d[field]);
    return isFinite(v) && v > 0 ? [{ key: b, label: `${String(d.name ?? b)} (${b}): ${field}`, bytes: v }] : [];
  });
  return limits;
}
function useMeasuredC(r: number) {
  const { reports } = useReports();
  return realBackends(reports)
    .map((b) => {
      const rows = benchRows(reports[b], LAB2);
      return { b, device: String(reports[b].device.name ?? b), byC: Object.fromEntries(C_CHOICES.map((c) => [c, measuredLocal(rows, r, c)])) as Record<number, number | undefined> };
    })
    .filter((d) => Object.values(d.byC).some((v) => v != null));
}

function KeysPerThread() {
  const [pid, setPid] = useState<Problem['id']>('n24');
  const [r, setR] = useState(8);
  const [threads, setThreads] = useState(256);
  const limits = useLimits();
  const [limitKey, setLimitKey] = useState<string>('');
  const [typed, setTyped] = useState('');
  const problem = PROBLEMS.find((p) => p.id === pid)!;
  const rows: Row[] = useMemo(() => sweepC(problem, 32, r, threads), [problem, r, threads]);
  const measured = useMeasuredC(r);
  const key = limitKey || limits[0]?.key || 'typed';
  const chosen = limits.find((l) => l.key === key);
  const typedBytes = Number(typed);
  const limit = key === 'typed' ? (typed.trim() !== '' && isFinite(typedBytes) && typedBytes > 0 ? typedBytes : null) : chosen?.bytes ?? null;
  const maxSectors = Math.max(...rows.map((x) => x.local.meanUnits));

  return (
    <Bench
      title={tr('Keys per thread', 'Số khóa mỗi luồng')}
      task={
        <Translated vi="Tăng c và xem các dải dài ra và bảng co lại, cho đến khi ô (tile) không còn vừa bộ nhớ chia sẻ.">
          Raise c and watch the runs shorten and the table shrink, until the tile no longer fits in shared memory.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Mô hình: sweepC() trong src/sim/radixcost.ts trên src/sim/radixlocal.ts. Khóa 32 bit với giá trị 4 byte; bộ nhớ
              chia sẻ mỗi khối là ô gồm c × threads cặp (mỗi cặp 8 byte) cộng 2<sup>r</sup> số đếm bốn byte, không gì khác. Số
              dải và sector được lấy trung bình qua mọi lượt của một mẫu có seed gồm vài khối, warp 32 lane, sector 32 byte của
              NVIDIA. Thanh ghi và occupancy không được mô hình hóa.
            </>
          }
        >
          <>
            Model: sweepC() in src/sim/radixcost.ts on src/sim/radixlocal.ts. 32-bit keys with 4-byte values; shared memory per
            block is the tile of c × threads pairs (8 bytes each) plus 2<sup>r</sup> four-byte counts, nothing else. Runs and
            sectors are averaged over every pass of a seeded sample of a few blocks, 32-lane warps, NVIDIA's 32-byte sectors.
            Registers and occupancy are not modelled.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <Seg label={tr('Problem', 'Bài toán')} value={pid} options={PROBLEMS.map((p) => p.id)} show={(id) => tr(PROBLEMS.find((p) => p.id === id)!.label, id === 'n20' ? '2²⁰ khóa ngẫu nhiên' : id === 'n24' ? '2²⁴ khóa ngẫu nhiên' : id === 'row' ? 'một hàng 151,936 token' : '64 hàng như thế')} onPick={setPid} />
        <Seg label={tr('Digit width r', 'Độ rộng chữ số r')} value={r} options={[1, 2, 4, 6, 8]} onPick={setR} />
        <Seg label={tr('Threads per block', 'Số luồng mỗi khối')} value={threads} options={THREADS} onPick={setThreads} />
      </div>
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ maxWidth: '100%', minWidth: 0 }}>
          <Translated vi="Bộ nhớ chia sẻ một khối được dùng">Shared memory a block may use</Translated>
          <select value={key} aria-label={tr('Shared memory limit', 'Giới hạn bộ nhớ chia sẻ')} onChange={(e) => setLimitKey(e.target.value)} style={{ maxWidth: '100%' }}>
            {limits.map((l) => <option key={l.key} value={l.key}>{l.label} = {big(l.bytes)} {tr('bytes', 'byte')}</option>)}
            <option value="typed">{tr('a number I type', 'một số tôi tự nhập')}</option>
          </select>
        </label>
        {key === 'typed' ? (
          <label className="field">
            <Translated vi="Byte">Bytes</Translated>
            <input type="number" min={0} value={typed} aria-label={tr('Your limit in bytes', 'Giới hạn của bạn tính bằng byte')} placeholder={tr('from doctor', 'lấy từ doctor')} onChange={(e) => setTyped(e.target.value)} />
          </label>
        ) : null}
      </div>
      {limit == null ? (
        <p className="ui" style={{ margin: '0 0 0.8rem' }} data-limit="none">
          <Translated
            vi={
              <>
                {limits.length ? 'Hãy nhập một giới hạn tính bằng byte để kiểm tra từng hàng.' : 'Máy này không có báo cáo thiết bị nào.'}{' '}
                Giới hạn là con số <code>python3 labs/pmpp.py doctor</code> báo: <code>sharedMemPerBlock</code> trên CUDA và HIP,{' '}
                <code>maxThreadgroupMemoryLength</code> trên Metal.
              </>
            }
          >
            {limits.length ? 'Type a limit in bytes to check each row against it.' : 'No device report on this machine.'} The limit is
            what <code>python3 labs/pmpp.py doctor</code> reports: <code>sharedMemPerBlock</code> on CUDA and HIP,{' '}
            <code>maxThreadgroupMemoryLength</code> on Metal.
          </Translated>
        </p>
      ) : null}
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0, minWidth: '50rem', background: 'var(--page)' }} aria-label={tr('Costs by keys per thread', 'Chi phí theo số khóa mỗi luồng')}>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.5rem' }}>c</th>
              <th className="num"><Translated vi="khóa mỗi khối">keys per block</Translated></th>
              <th className="num"><Translated vi="khối mỗi lượt">blocks per pass</Translated></th>
              <th className="num"><Translated vi="số mục bảng">table entries</Translated></th>
              <th className="num"><Translated vi="khóa mỗi bucket cục bộ">keys per local bucket</Translated></th>
              <th className="num"><Translated vi="dải mỗi lần ghi">runs per store</Translated></th>
              <th><Translated vi="sector mỗi lần ghi">sectors per store</Translated></th>
              <th className="num"><Translated vi="bộ nhớ chia sẻ mỗi khối">shared memory per block</Translated></th>
              {measured.map((d) => <th key={d.b} className="num">{d.b}, ms{/* i18n-ignore: backend name and unit */}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((x) => {
              const over = limit != null && x.sharedBytes > limit;
              return (
                <tr key={x.c} data-c={x.c} data-over={over ? 1 : 0} style={over ? { background: 'var(--fail-wash)' } : undefined}>
                  <td style={{ paddingLeft: '0.5rem' }}><b>{x.c}</b></td>
                  <td className="num">{big(x.c * threads)}</td>
                  <td className="num" data-col="blocks">{big(x.blocks)}</td>
                  <td className="num" data-col="table">{big(x.tableEntries)}</td>
                  <td className="num" data-col="bucket">{fmt(x.keysPerBucket)}</td>
                  <td className="num" data-col="runs">{dec(x.local.meanRuns)}</td>
                  <td data-col="sectors" className="nowrap">
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, whiteSpace: 'nowrap' }}>
                      <i style={{ display: 'inline-block', height: 10, width: `${(x.local.meanUnits / maxSectors) * 6}rem`, background: 'var(--global-wash)', border: '1px solid var(--global)' }} />
                      {dec(x.local.meanUnits)}
                      <span className="muted"><Translated vi={<>(tối thiểu {dec(x.local.meanMin)})</>}>(min {dec(x.local.meanMin)})</Translated></span>
                    </span>
                  </td>
                  <td className="num nowrap" data-col="smem">
                    {over ? (
                      <span className="t-fail">
                        <Translated vi={<>{kib(x.sharedBytes)}, vượt {kib(x.sharedBytes - limit!)}</>}>
                          {kib(x.sharedBytes)}, over by {kib(x.sharedBytes - limit!)}
                        </Translated>
                      </span>
                    ) : kib(x.sharedBytes)}
                  </td>
                  {measured.map((d) => <td key={d.b} className="num" data-measured={`${d.b}-${x.c}`}>{d.byC[x.c] != null ? fmt(d.byC[x.c]! * 1000) : <span className="muted"><Translated vi="không có dòng">no row</Translated></span>}</td>)}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="ui" style={{ margin: '0.7rem 0 0' }} data-limit-line={limit ?? ''}>
        {limit != null ? (
          <Translated
            vi={
              <>
                Giới hạn đang dùng: {big(limit)} byte{chosen && key !== 'typed' ? `, như ${chosen.label} đã báo` : ', như bạn đã nhập'}. Các hàng
                màu đỏ không vừa: không thể khởi chạy khối với ô đó.
              </>
            }
          >
            <>
              Limit in use: {big(limit)} bytes{chosen && key !== 'typed' ? `, as ${chosen.label} reported it` : ', as you typed it'}. Rows in red do
              not fit: the block could not be launched with that tile.
            </>
          </Translated>
        ) : null}{' '}
        {measured.length
          ? tr(
              'The millisecond columns are lab 14.2 measured on this machine at its own problem size; the rest is the model.',
              'Các cột mili giây là số đo lab 14.2 trên máy này ở kích thước bài toán riêng của lab; phần còn lại là mô hình.',
            )
          : tr(
              'No lab 14.2 measurement on this machine yet; run python3 labs/pmpp.py run 14.2 --solution --bench and reload for a measured column.',
              'Máy này chưa có số đo nào của lab 14.2; hãy chạy python3 labs/pmpp.py run 14.2 --solution --bench và tải lại để có một cột số đo.',
            )}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const N22 = { rows: 1, rowLength: 2 ** 22 };

export const questionsS148: Record<string, Question> = {
  'ch14/14.8/q-table': {
    kind: 'numeric',
    id: 'ch14/14.8/q-table',
    prompt: (
      <Translated vi={<>2<sup>22</sup> khóa, 256 luồng mỗi khối, r = 8, mười sáu khóa mỗi luồng. Bảng bucket có bao nhiêu mục?</>}>
        <>2<sup>22</sup> keys, 256 threads per block, r = 8, sixteen keys per thread. How many entries does the bucket table have?</>
      </Translated>
    ),
    answer: tableEntries(N22, 8, 256, 16),
    unit: 'entries',
    vi: { unit: 'mục' },
    explain: (
      <Translated
        vi={
          <p>
          Giờ một khối sở hữu 16 × 256 = 4,096 khóa, nên có 2<sup>22</sup> / 4,096 = 1,024 khối, mỗi khối 256 bucket:{' '}
          {big(tableEntries(N22, 8, 256, 16))} mục. Với một khóa mỗi luồng, con số là {big(tableEntries(N22, 8, 256, 1))}, nhiều
          gấp mười sáu lần.
          </p>
        }
      >
        <p>
          A block now owns 16 × 256 = 4,096 keys, so there are 2<sup>22</sup> / 4,096 = 1,024 blocks, each with 256 buckets:{' '}
          {big(tableEntries(N22, 8, 256, 16))} entries. At one key per thread it was {big(tableEntries(N22, 8, 256, 1))}, sixteen
          times more.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.8, p. 345',
  },
  'ch14/14.8/q-smem': {
    kind: 'numeric',
    id: 'ch14/14.8/q-smem',
    prompt: (
      <Translated vi={<>Tám khóa mỗi luồng, 256 luồng, các cặp (uint32 key, uint32 value). Riêng ô (tile) cần bao nhiêu byte bộ nhớ chia sẻ?</>}>
        <>Eight keys per thread, 256 threads, (uint32 key, uint32 value) pairs. How many bytes of shared memory does the tile alone need?</>
      </Translated>
    ),
    answer: tileBytes(256, 8),
    unit: 'bytes',
    vi: { unit: 'byte', source: 'phép tính của khóa học này; sách ch. 5 §5.6 về bộ nhớ chia sẻ và occupancy' },
    explain: (
      <Translated
        vi={
          <p>
          8 × 256 cặp, mỗi cặp 8 byte: {big(tileBytes(256, 8))} byte, chưa kể các số đếm và vùng nháp cho scan. Hãy so nó với giới
          hạn mà <code>python3 labs/pmpp.py doctor</code> báo cho thiết bị của bạn; rồi nhớ rằng một khối dùng nhiều bộ nhớ chia sẻ
          hơn cũng chừa chỗ cho ít khối hơn bên cạnh nó (mục 5.6).
          </p>
        }
      >
        <p>
          8 × 256 pairs of 8 bytes: {big(tileBytes(256, 8))} bytes, before the counts and any scan scratch. Compare it with the
          limit <code>python3 labs/pmpp.py doctor</code> reports for your device; then remember that a block using more shared
          memory also leaves room for fewer blocks beside it (section 5.6).
        </p>
      </Translated>
    ),
    source: 'our arithmetic; book ch. 5 §5.6 on shared memory and occupancy',
  },
  'ch14/14.8/q-waves': {
    kind: 'choice',
    id: 'ch14/14.8/q-waves',
    prompt: (
      <Translated
        vi={
          <>
          Một thiết bị có thể chạy cùng lúc 40 khối của bạn, và phép sắp xếp khởi chạy 16,384 khối mỗi lượt. Gộp các khối bằng
          cách gộp việc cho luồng (thread coarsening) phải hy sinh điều gì?
          </>
        }
      >
        <>
          A device can run 40 of your blocks at a time, and the sort launches 16,384 blocks per pass. What does merging blocks
          by coarsening give up?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Gần như không gì: phần lớn các khối vốn đang chờ một chỗ trống, và khối lớn hơn ghi các dải dài hơn.">
            Almost nothing: most blocks were waiting for a slot anyway, and larger blocks write longer runs.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Tính song song, nên chắc chắn chậm hơn: ít khối hơn nghĩa là ít khóa được sắp hơn tại mỗi thời điểm trên thiết bị.">
            Parallelism, so it must be slower: fewer blocks means fewer keys being sorted at any one moment on the device.
          </Translated>
        ),
        why: (
          <Translated vi="Dù thế nào cũng chỉ 40 khối chạy cùng lúc. Chừng nào còn nhiều hơn hẳn 40 khối, thiết bị vẫn bận như cũ.">
            Only 40 blocks run at once either way. As long as there are well over 40 blocks, the device stays as busy.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Tính đúng đắn: các khối đã gộp việc chia nhau bucket, và sẽ cần phép toán nguyên tử (atomic) để đặt khóa.">
            Correctness: coarsened blocks share buckets, and would need atomics to place their keys.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi khối vẫn sở hữu các khóa và các mục bảng của nó; lần scan vẫn cho mỗi bucket cục bộ một khoảng riêng.">
            Each block still owns its keys and its table entries; the scan still gives every local bucket its own range.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
          Lập luận của sách: nhiều khối nhỏ trả giá cho tính song song bằng các bucket cục bộ nhỏ hơn, và cái giá đó chỉ đáng trả
          nếu các khối thật sự chạy cùng lúc. Với 16,384 khối và chỗ cho 40, chúng chạy theo từng đợt 40, nên gộp chúng lại không
          mất gì cho đến khi còn quá ít khối để lấp đầy thiết bị.
          </p>
        }
      >
        <p>
          The book's argument: many small blocks pay for parallelism with smaller local buckets, and the price is only worth
          paying if those blocks really run at the same time. With 16,384 blocks and room for 40, they run in waves of 40, so
          merging them gives up nothing until there are too few blocks to fill the device.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.8, p. 344',
  },
  'ch14/14.8/q-fig': {
    kind: 'numeric',
    id: 'ch14/14.8/q-fig',
    prompt: (
      <Translated vi={<>Hình 14.13 chia mười sáu khóa cho hai khối bốn luồng. Điều đó ngụ ý mỗi luồng bao nhiêu khóa?</>}>
        <>Fig. 14.13 splits sixteen keys between two blocks of four threads. How many keys per thread does that imply?</>
      </Translated>
    ),
    answer: 2,
    unit: 'keys',
    vi: { unit: 'khóa', source: 'ch. 14 §14.8, Fig. 14.13, p. 345; notes, mục 8 của "Unclear in the PDF"' },
    explain: (
      <Translated
        vi={
          <p>
          16 khóa / (2 khối × 4 luồng) = 2. Hình vẽ bốn luồng của mỗi khối chỉ trên bốn khóa đầu trong tám khóa của khối, và không
          bao giờ nói luồng nào xử lý khóa nào; hai khóa mỗi luồng là cách đọc duy nhất khớp số.
          </p>
        }
      >
        <p>
          16 keys / (2 blocks × 4 threads) = 2. The figure draws each block's four threads over only the first four of its eight
          keys and never says which thread handles which key; two per thread is the only reading that adds up.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.8, Fig. 14.13, p. 345; notes, "Unclear in the PDF" item 8',
  },
};

// ---------------------------------------------------------------- the section
export function S148() {
  const measured = useMeasuredC(8);
  const rowBlock = 256;
  const perThread = Math.ceil(VOCAB / rowBlock);
  return (
    <>
      <Translated
        vi={
          <p>
          Phép sắp xếp cục bộ có một chi phí không liên quan gì đến chữ số. Mỗi khối ghi các bucket cục bộ của riêng nó, nên các
          khóa trải trên càng nhiều khối thì mỗi khối giữ càng ít khóa, bucket của nó càng nhỏ, và các dải của nó càng ngắn. Cái
          giá đó mua được tính song song, và chỉ đáng trả nếu các khối thật sự chạy cùng lúc. Nếu phần cứng vẫn chạy chúng từng
          đợt nối tiếp nhau, thì các bucket nhỏ đã bị trả giá mà không mua được gì.
          </p>
        }
      >
        <p>
          The local sort has a cost that has nothing to do with the digit. Every block writes its own local buckets, so the more
          blocks the keys are spread over, the fewer keys each block holds, the smaller its buckets, and the shorter its runs.
          That price buys parallelism, and it is worth paying only if the blocks really run at the same time. If the hardware
          runs them one wave after another anyway, the small buckets are paid for and nothing is bought.
        </p>
      </Translated>
      <Source section="14.8" pages="344" />
      <Translated
        vi={
          <p>
          Vậy hãy cho mỗi luồng vài khóa. Với c khóa mỗi luồng, một khối B luồng sở hữu c × B khóa, các bucket cục bộ của nó lớn
          gấp c lần trung bình, và các luồng liên tiếp có nhiều khả năng ghi vào các địa chỉ liên tiếp hơn. Hình của sách làm điều
          này với chữ số 2 bit: hai khối bốn luồng trên mười sáu khóa, mỗi bucket cục bộ giữ hai khóa, và các mũi tên tới đầu ra
          đi thành từng cặp song song.
          </p>
        }
      >
        <p>
          So give each thread several keys. With c keys per thread a block of B threads owns c × B keys, its local buckets are
          on average c times larger, and consecutive threads are more likely to write consecutive addresses. The book's figure
          does this with a 2-bit digit: two blocks of four threads over the sixteen keys, each local bucket holding two keys,
          and the arrows to the output come in parallel pairs.
        </p>
      </Translated>
      <Source section="14.8" pages="344–345" figs={tr('Fig. 14.13', 'Hình 14.13')} />
      <Aside>
        <Translated
          vi={
            <p>
            Hình 14.13 vẽ các luồng của mỗi khối chỉ trên một nửa số khóa của khối và không nói luồng nào sở hữu khóa nào. Tám khóa
            cho một khối bốn luồng nghĩa là hai khóa mỗi luồng. Trình khám phá bên dưới làm rõ điều đó.
            </p>
          }
        >
          <p>
            Fig. 14.13 draws each block's threads over only half of its keys and does not say which thread owns which key. Eight
            keys per four-thread block means two keys per thread. The explorer below makes that explicit.
          </p>
        </Translated>
      </Aside>
      <Fig1413 />
      <Ask q={questionsS148['ch14/14.8/q-fig']} />
      <Translated
        vi={
          <p>
          Còn một lợi ích thứ hai. Bảng có một mục cho mỗi bucket của mỗi khối, nên ít khối hơn c lần làm bảng nhỏ đi c lần, và
          lần scan toàn cục giữa các pha rẻ hơn c lần. Ở r = 8 điều đó quan trọng: với 256 luồng mỗi khối và một khóa mỗi luồng,
          mục 14.7 thấy bảng dài bằng dữ liệu.
          </p>
        }
      >
        <p>
          There is a second benefit. The table has one entry per bucket per block, so c times fewer blocks make it c times
          smaller, and the global scan between the phases c times cheaper. At r = 8 that matters: with 256 threads per block and
          one key per thread, section 14.7 found the table as long as the data.
        </p>
      </Translated>
      <Source section="14.8" pages="345" />
      <Ask q={questionsS148['ch14/14.8/q-table']} />

      <h2><Translated vi="Nó tốn gì">What it costs</Translated></h2>
      <Translated
        vi={
          <p>
          Ô (tile) nằm trong bộ nhớ chia sẻ: c × B khóa và cùng số giá trị, 8 byte một cặp, cộng các số đếm của khối. Bộ nhớ chia
          sẻ mỗi khối có giới hạn, và một khối lấy càng nhiều thì càng ít khối vừa trên một multiprocessor bên cạnh nó, chính là
          ngân sách occupancy (mức chiếm dụng) của chương 5. Vậy bộ nhớ chia sẻ chặn trên c, và occupancy có thể giảm từ lâu trước
          mức chặn đó. Trên Metal, giới hạn là <code>maxThreadgroupMemoryLength</code> của thiết bị; trình khám phá đọc nó, hoặc{' '}
          <code>sharedMemPerBlock</code> trên CUDA và HIP, từ báo cáo doctor của bạn.
          </p>
        }
      >
        <p>
          The tile lives in shared memory: c × B keys and as many values, 8 bytes a pair, plus the block's counts. Shared memory
          per block is limited, and the more a block takes, the fewer blocks fit on a multiprocessor beside it, which is chapter
          5's occupancy budget. So shared memory caps c, and occupancy can fall well before that cap. On Metal the limit is the
          device's <code>maxThreadgroupMemoryLength</code>; the explorer reads it, or <code>sharedMemPerBlock</code> on CUDA and
          HIP, from your doctor report.
        </p>
      </Translated>
      <Beyond title={tr('Ours', 'Của khóa học này')}>
        <Translated
          vi={
            <p>
            Ở đây sách không nêu chi phí nào của gộp việc; chi phí này đến từ các chương 5 và 6. Giới hạn bộ nhớ threadgroup của Metal
            và cách định kích thước nó lúc khởi chạy nằm trong ch05-facts C1 và C8.
            </p>
          }
        >
          <p>
            The book names no cost of coarsening here; this one comes from chapters 5 and 6. Metal's threadgroup memory limit and
            how to size it at launch are in ch05-facts C1 and C8.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS148['ch14/14.8/q-smem']} />
      <Predict
        id="ch14/14.8/p-best-c"
        prompt={
          <Translated vi="Ở r = 8, lab 14.2 đo thời gian thiết kế sắp xếp cục bộ của nó với 1, 2, 4 và 8 khóa mỗi luồng. Giá trị nào sẽ nhanh nhất trên thiết bị của bạn?">
            At r = 8, lab 14.2 times its local-sort design with 1, 2, 4 and 8 keys per thread. Which will be fastest on your device?
          </Translated>
        }
        unit={tr('keys per thread', 'khóa mỗi luồng')}
      >
        {(v) =>
          measured.length ? (
            <>
              {measured.map((d) => {
                const best = fastest([1, 2, 4, 8].map((c) => ({ at: c, seconds: d.byC[c] })));
                return best ? (
                  <p key={d.b} data-best-c={`${d.b}-${best.at}`}>
                    <Translated
                      vi={
                        <>
                          <b>{d.device}</b> ({d.b}): nhanh nhất ở <b>c = {best.at}</b>, {fmt(best.seconds * 1000)} ms
                          {best.at === v ? ', đúng như bạn nói' : `; bạn đã nói ${v}`}.
                        </>
                      }
                    >
                      <b>{d.device}</b> ({d.b}): fastest at <b>c = {best.at}</b>, {fmt(best.seconds * 1000)} ms
                      {best.at === v ? ', as you said' : `; you said ${v}`}.
                    </Translated>
                  </p>
                ) : null;
              })}
              <p className="muted">
                <Translated vi="Hãy so nó với hàng mà ở đó số sector của trình khám phá ngừng giảm và bộ nhớ chia sẻ bắt đầu tăng.">
                  Compare it with the row where the explorer's sectors stop falling and the shared memory starts to climb.
                </Translated>
              </p>
            </>
          ) : (
            <Translated
              vi={
                <p data-measured="none">
                Máy này chưa có số đo nào. Hãy chạy <code>python3 labs/pmpp.py run 14.2 --solution --bench</code>, rồi tải lại trang
                này.
                </p>
              }
            >
              <p data-measured="none">
                No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 14.2 --solution --bench</code>, then
                reload this page.
              </p>
            </Translated>
          )
        }
      </Predict>
      <Translated
        vi={
          <p>
          Trình khám phá quét c từ 1 đến 32 cho bài toán, độ rộng chữ số và kích thước khối bạn chọn, và kiểm tra bộ nhớ chia sẻ
          của từng hàng so với giới hạn thiết bị của bạn đã báo, hoặc một giới hạn bạn tự nhập.
          </p>
        }
      >
        <p>
          The explorer sweeps c from 1 to 32 for a problem, a digit width and a block size of your choice, and checks each
          row's shared memory against the limit your device reported, or one you type.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
            Như ở mục 14.7: byte và sector được đếm, không đo thời gian, trên các khóa lấy mẫu. Giới hạn bộ nhớ chia sẻ là con số
            thiết bị của bạn đã báo hoặc con số bạn nhập. Thanh ghi, occupancy và vùng nháp riêng của scan không được mô hình hóa.
            </p>
          }
        >
          <p>
            As in section 14.7: bytes and sectors are counted, not timed, on sampled keys. The shared-memory limit is the figure
            your device reported or the one you type. Registers, occupancy and the scan's own scratch space are not modelled.
          </p>
        </Translated>
      </Simplification>
      <KeysPerThread />
      <Translated
        vi={
          <p>
          Gộp việc làm các dải dài ra và bảng co lại cho đến khi ô không còn vừa. Bài tập 3 yêu cầu đúng điều này, và lab 14.2 đưa
          số khóa mỗi luồng ra thành một thiết lập để bạn đo được thiết bị của mình ngừng có lợi ở đâu.
          </p>
        }
      >
        <p>
          Coarsening lengthens the runs and shrinks the table until the tile no longer fits. Exercise 3 asks for exactly this,
          and lab 14.2 exposes the keys per thread as a setting so that you can measure where your device stops gaining.
        </p>
      </Translated>
      <Source section="14.11" pages="347" />
      <Today checked="2026-11-20">
        <Translated
          vi={
            <p>
            <code>items_per_thread</code> của CUB chính là hệ số này: ví dụ tinh chỉnh trong tài liệu của nó đặt 21 cho OneSweep và
            25 cho downsweep. Tài liệu <code>BlockRadixSort</code> của nó nói hiệu quả tăng theo số phần tử mỗi luồng cho đến khi áp
            lực thanh ghi hoặc bộ nhớ chia sẻ làm occupancy giảm quá nhiều (ch14-facts A4).
            </p>
          }
        >
          <p>
            CUB's <code>items_per_thread</code> is this factor: the tuning example in its documentation sets 21 for OneSweep and 25
            for the downsweep. Its <code>BlockRadixSort</code> documentation says efficiency grows with items per thread until
            register pressure or shared memory lowers occupancy too far (ch14-facts A4).
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS148['ch14/14.8/q-waves']} />

      <h2><Translated vi="Gộp việc đến tận cùng">Coarsening taken to its end</Translated></h2>
      <Translated
        vi={
          <p>
          Lab 14.3, mốc dự án các ứng viên đã sắp xếp, đẩy ý tưởng này đến tận cùng: một khối {rowBlock} luồng cho mỗi hàng từ vựng,
          nên một hàng {big(VOCAB)} token cho mỗi luồng khoảng {perThread} khóa. Con số đó vượt xa mọi ô:{' '}
          {perThread} × {rowBlock} cặp 8 byte là {big(tileBytes(rowBlock, perThread))} byte, nên khối không thể giữ hàng của nó
          trong bộ nhớ chia sẻ và phải xử lý từng phần một. Cái được là số đếm bucket của một hàng không bao giờ rời khối, nên
          không cần scan giữa các khối và không cần bước toàn lưới giữa các lượt. Cái giá là chính cái giá trang này bắt đầu, theo
          chiều ngược lại: ở batch 1, đó là một khối duy nhất cho cả thiết bị, tình huống mà trình khám phá của{' '}
          <a href={href('ch10', '10.10')}>mục 10.10</a> đã cho thấy với softmax một khối mỗi hàng.
          </p>
        }
      >
        <p>
          Lab 14.3, the sorted-candidates milestone, takes the idea as far as it goes: one block of {rowBlock} threads per
          vocabulary row, so a row of {big(VOCAB)} tokens gives each thread about {perThread} keys. That is far past any tile:{' '}
          {perThread} × {rowBlock} pairs of 8 bytes is {big(tileBytes(rowBlock, perThread))} bytes, so the block cannot hold its
          row in shared memory and has to work through it a piece at a time. What it gains is that a row's bucket counts never
          leave the block, so no scan across blocks and no grid-wide step between passes are needed. The price is the one this
          page started with, turned around: at batch 1 that is a single block for the whole device, the situation the explorer
          of <a href={href('ch10', '10.10')}>section 10.10</a> showed for a softmax with one block per row.
        </p>
      </Translated>
      <Beyond title={tr('Ours', 'Của khóa học này')}>
        <Translated
          vi={
            <p>
            Sách không có phép sắp xếp theo hàng hay theo batch; thiết kế này là của lab 14.3, và những gì nó ngụ ý là cách đọc của
            chúng tôi. Kích thước bộ từ vựng là một trong các kích thước trong ch09-facts D7.
            </p>
          }
        >
          <p>
            The book has no row-wise or batched sort; this design is lab 14.3's, and what it implies is our reading. The
            vocabulary size is one of those in ch09-facts D7.
          </p>
        </Translated>
      </Beyond>
    </>
  );
}
