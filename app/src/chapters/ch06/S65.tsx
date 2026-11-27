import { useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { fmt, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { warpAccess } from '../../sim/coalesce';
import {
  BOOK_CAPACITY,
  PITFALLS,
  SECTOR_BYTES,
  VERDICTS,
  calculate,
  deviceSlots,
  elementsOf,
  residentUpperBound,
  warpInstruction,
  warpVectorInstruction,
  type Mapping,
  type Verdict,
} from '../../sim/coarsen';
import { Translated, tr } from '../../lib/i18n';

// labs/ch06/snippets/s65_coarsen_toy.cu, which ./verify.sh front-end-checks.
// snippets-s65.test.ts fails if this copy drifts from the file.
const TOY = `// Section 6.5, pp. 141-142: the book's thread-coarsening toy, made compilable.
// foo stands for one unit of work (the book leaves it abstract); here it doubles
// one element. The bounds checks are ours: the book's toy has none.
__device__ void foo(float* data, unsigned int i) { data[i] = 2.0f*data[i]; }

// Before: one unit of work per thread. Launch one thread per element.
__global__ void fine(float* data, unsigned int n) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) foo(data, i);
}

// After: coarsening factor 4. Launch a quarter as many threads.
#define COARSEN 4
__global__ void coarse(float* data, unsigned int n) {
    unsigned int iStart = COARSEN*(blockIdx.x*blockDim.x + threadIdx.x);
    for (unsigned int c = 0; c < COARSEN; ++c) {   // the coarsening loop
        unsigned int i = iStart + c;
        if (i < n) foo(data, i);
    }
}

// Not in the book: the same factor with a strided mapping. Thread t handles
// t, t + T, t + 2T and t + 3T, where T is the number of threads in the grid.
__global__ void coarseStrided(float* data, unsigned int n) {
    unsigned int T = gridDim.x*blockDim.x;
    unsigned int t = blockIdx.x*blockDim.x + threadIdx.x;
    for (unsigned int c = 0; c < COARSEN; ++c) {
        unsigned int i = t + c*T;
        if (i < n) foo(data, i);
    }
}`;

const pct = (x: number) => `${fmt(100 * x)}%`;
const count = (n: number) => n.toLocaleString('en-US');

// ---------------------------------------------------------------- wave calculator
const TPB_OPTIONS = [64, 128, 256, 512, 1024];
type DeviceChoice = { key: string; label: string; capacity: number | null; note: string };

function WaveCalculator() {
  const { reports } = useReports();
  const [tpb, setTpb] = useState(256);
  const [units, setUnits] = useState(1056 * 256);
  const [factor, setFactor] = useState(1);
  const [dev, setDev] = useState('book');
  const [custom, setCustom] = useState(100);

  const devices: DeviceChoice[] = useMemo(() => {
    const list: DeviceChoice[] = [
      {
        key: 'book',
        get label() {
          return tr("The book's hypothetical device (264)", 'Thiết bị giả định của sách (264)');
        },
        capacity: BOOK_CAPACITY,
        get note() {
          return tr("The book's example on p. 143: a device that runs 264 blocks at the same time. It names no device.", 'Ví dụ của sách ở p. 143: một thiết bị chạy 264 khối cùng lúc. Sách không nêu tên thiết bị nào.');
        },
      },
    ];
    for (const b of realBackends(reports).filter((x) => x === 'cuda' || x === 'hip' || x === 'metal')) {
      const d = reports[b].device;
      const slots = deviceSlots(d);
      const name = String(d.name ?? b);
      if (slots && (b === 'cuda' || b === 'hip'))
        list.push({
          key: b,
          get label() {
            return tr(`Yours: ${name} (${b})`, `Của bạn: ${name} (${b})`);
          },
          capacity: residentUpperBound(slots, tpb),
          get note() {
            return tr(
              `${slots.units} × min(${slots.blocksPerUnit} block slots, ${slots.threadsPerUnit || 'unreported'} thread slots ÷ ${tpb} threads) from this device's report. ` +
                'An upper bound: registers and shared memory per block can only lower it, and the occupancy API (chapter 4) gives the real figure for a real kernel.' +
                (b === 'hip' ? ' On HIP the block-slot field is the number of wave slots, not a limit of its own (ch04-facts B9).' : ''),
              `${slots.units} × min(${slots.blocksPerUnit} chỗ cho khối, ${slots.threadsPerUnit || 'không được báo cáo'} chỗ cho luồng ÷ ${tpb} luồng) từ báo cáo của thiết bị này. ` +
                'Đây là cận trên: thanh ghi và bộ nhớ chia sẻ mỗi khối chỉ có thể làm nó giảm, và API occupancy (chương 4) cho con số thật với một kernel thật.' +
                (b === 'hip' ? ' Trên HIP, trường chỗ cho khối là số chỗ cho wave, không phải một giới hạn riêng (ch04-facts B9).' : ''),
            );
          },
        });
      else
        list.push({
          key: b,
          get label() {
            return tr(`Yours: ${name} (${b})`, `Của bạn: ${name} (${b})`);
          },
          capacity: null,
          get note() {
            return b === 'metal' ? tr('Metal reports no number of blocks a device runs at once, so there is nothing to put here. Use a custom value.', 'Metal không báo cáo số khối một thiết bị chạy cùng lúc, nên không có gì để điền ở đây. Hãy dùng một giá trị tùy chọn.') : tr('This report has no per-SM fields.', 'Báo cáo này không có các trường theo từng SM.');
          },
        });
    }
    list.push({
      key: 'custom',
      get label() {
        return tr('Custom', 'Tùy chọn');
      },
      capacity: Math.max(1, Math.floor(custom)),
      get note() {
        return tr('A number you choose.', 'Một con số do bạn chọn.');
      },
    });
    return list;
  }, [reports, tpb, custom]);

  const device = devices.find((d) => d.key === dev) ?? devices[0];
  const capacity = device.capacity;
  const r = capacity ? calculate(units, tpb, factor, capacity) : null;
  const preset = (blocks: number) => {
    setTpb(256);
    setUnits(blocks * 256);
    setFactor(1);
    setDev('book');
  };

  const MAXROWS = 8;
  const rows = r ? r.fill.perWave : [];
  const shown = rows.length > MAXROWS ? [...rows.slice(0, MAXROWS - 1).map((n, i) => ({ n, w: i })), { n: rows[rows.length - 1], w: rows.length - 1 }] : rows.map((n, i) => ({ n, w: i }));
  const W = 560;
  const rowH = 30;

  return (
    <Bench
      title={tr('Wave calculator', 'Máy tính wave')}
      task={tr("Reproduce the book's two cases, then find a factor that leaves only whole waves.", 'Tái hiện hai trường hợp của sách, rồi tìm một hệ số chỉ để lại các wave trọn vẹn.')}
      foot={tr('Each row is one wave: the blocks the device runs at the same time. Filled: a block is running in that slot. Dashed: the slot is idle.', 'Mỗi hàng là một wave: các khối mà thiết bị chạy cùng lúc. Tô đặc: một khối đang chạy ở chỗ đó. Nét đứt: chỗ đó đang rảnh.')}
    >
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <button onClick={() => preset(1056)}><Translated vi="1056 khối của sách">The book's 1056 blocks</Translated></button>
        <button onClick={() => preset(1584)}><Translated vi="1584 khối của sách">The book's 1584 blocks</Translated></button>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Đơn vị công việc">Work units</Translated>
          <input type="number" min={1} value={units} aria-label={tr('work units', 'đơn vị công việc')} style={{ width: '9rem' }}
            onChange={(e) => setUnits(Math.max(1, Math.floor(Number(e.target.value) || 1)))} />
        </label>
        <label className="field">
          <Translated vi="Số luồng mỗi khối">Threads per block</Translated>
          <select value={tpb} onChange={(e) => setTpb(+e.target.value)}>
            {TPB_OPTIONS.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Hệ số gộp việc">Coarsening factor</Translated>
          <input type="range" min={1} max={64} value={factor} aria-label={tr('coarsening factor slider', 'thanh trượt hệ số gộp việc')} onChange={(e) => setFactor(+e.target.value)} />
          <input type="number" min={1} max={64} value={factor} aria-label={tr('coarsening factor', 'hệ số gộp việc')} style={{ width: '4.5rem' }}
            onChange={(e) => setFactor(Math.min(64, Math.max(1, Math.floor(Number(e.target.value) || 1))))} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Số khối thiết bị chạy cùng lúc">Blocks the device runs at once</Translated>
          <select value={device.key} onChange={(e) => setDev(e.target.value)} aria-label={tr('device', 'thiết bị')}>
            {devices.map((d) => (
              <option key={d.key} value={d.key} disabled={d.capacity == null}>
                {d.label}{d.capacity == null ? tr(': no figure', ': không có số liệu') : d.key !== 'book' && d.key !== 'custom' ? tr(`: at most ${count(d.capacity)}`, `: tối đa ${count(d.capacity)}`) : ''}
              </option>
            ))}
          </select>
        </label>
        {device.key === 'custom' ? (
          <label className="field">
            <Translated vi="Số khối cùng lúc">Blocks at once</Translated>
            <input type="number" min={1} value={custom} aria-label={tr('custom capacity', 'sức chứa tùy chọn')} style={{ width: '6rem' }}
              onChange={(e) => setCustom(Math.max(1, Math.floor(Number(e.target.value) || 1)))} />
          </label>
        ) : null}
      </div>
      <p className="muted ui" style={{ margin: '0 0 0.8rem' }}>{device.note}</p>
      {devices.filter((d) => d.capacity == null).map((d) => (
        <p key={d.key} className="muted ui" style={{ margin: '0 0 0.8rem' }} data-k="no-figure">
          {d.label}: {d.note}
        </p>
      ))}

      {r && capacity ? (
        <>
          <table className="data" style={{ background: 'var(--page)', margin: '0 0 0.8rem' }} aria-label={tr('wave arithmetic', 'phép tính wave')}>
            <tbody>
              <tr><td><Translated vi="luồng">threads</Translated></td><td data-k="threads">{count(r.threads)}</td><td className="muted">⌈{count(units)} ÷ {factor}⌉</td></tr>
              <tr><td><Translated vi="khối">blocks</Translated></td><td data-k="blocks">{count(r.blocks)}</td><td className="muted"><Translated vi={<>⌈luồng ÷ {tpb}⌉</>}>⌈threads ÷ {tpb}⌉</Translated></td></tr>
              <tr><td><Translated vi="wave">waves</Translated></td><td data-k="waves">{fmt(r.waves)}</td><td className="muted"><Translated vi={<>khối ÷ {count(capacity)}</>}>blocks ÷ {count(capacity)}</Translated></td></tr>
              <tr><td><Translated vi="chỗ rảnh">slots idle</Translated></td><td data-k="idle">{pct(r.fill.idle)}</td><td className="muted"><Translated vi={<>trên tất cả {r.fill.waveCount} wave</>}>over all {r.fill.waveCount} wave{r.fill.waveCount === 1 ? '' : 's'}</Translated></td></tr>
              <tr>
                <td><Translated vi="thời gian theo wave">wave time</Translated></td>
                <td data-k="time">{r.time}</td>
                <td className="muted"><Translated vi={<>{r.fill.waveCount} wave × {factor}, so với {r.baseTime} khi không gộp việc, nếu gộp việc không tiết kiệm được gì</>}>{r.fill.waveCount} wave{r.fill.waveCount === 1 ? '' : 's'} × {factor}, against {r.baseTime} uncoarsened, if coarsening saved nothing</Translated></td>
              </tr>
            </tbody>
          </table>
          <p style={{ margin: '0 0 0.8rem' }} data-k="verdict">
            {r.fill.whole ? (
              <span className="chip ok"><Translated vi="chỉ các wave trọn vẹn">only whole waves</Translated></span>
            ) : (
              <span className="chip fail">
                {r.waves < 1 ? tr('less than one wave', 'chưa tới một wave') : tr('partial last wave', 'wave cuối chưa đầy')}<Translated vi={<>: {count(r.fill.tail)} trên {count(capacity)} chỗ đang bận</>}>: {count(r.fill.tail)} of {count(capacity)} slots busy</Translated>
              </span>
            )}
          </p>
          <svg viewBox={`0 0 ${W + 160} ${shown.length * rowH + 30}`} role="img" aria-label={tr('waves of blocks', 'các wave của khối')} style={{ width: '100%', maxWidth: W + 160, display: 'block' }}>
            {shown.map(({ n, w }, k) => {
              const y = k * rowH + 4;
              const fw = (W * n) / capacity;
              const skipped = rows.length > MAXROWS && k === MAXROWS - 1;
              const cells = capacity <= 48;
              return (
                <g key={w}>
                  {skipped ? (
                    <text x={0} y={y - 8} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>… thêm {rows.length - MAXROWS} wave đầy</>}>… {rows.length - MAXROWS} more full waves</Translated></text>
                  ) : null}
                  <text x={0} y={y + 16} fontSize="12" style={{ fill: 'var(--ink-2)' }} data-wave={w + 1} data-fill={n}><Translated vi={<>wave {w + 1}</>}>wave {w + 1}</Translated></text>
                  {cells ? (
                    Array.from({ length: capacity }, (_, s) => {
                      const cw = W / capacity;
                      return (
                        <rect key={s} x={60 + s * cw + 1} y={y + 2} width={Math.max(1, cw - 2)} height={rowH - 10} rx="2"
                          fill={s < n ? 'var(--ink-2)' : 'none'} stroke="var(--ink-3)" strokeDasharray={s < n ? undefined : '3 3'} />
                      );
                    })
                  ) : (
                    <>
                      <rect x={60} y={y + 2} width={W} height={rowH - 10} rx="2" fill="none" stroke="var(--ink-3)" strokeDasharray="4 3" />
                      <rect x={60} y={y + 2} width={fw} height={rowH - 10} rx="2" fill="var(--ink-2)" />
                    </>
                  )}
                  <text x={60 + W + 8} y={y + 16} fontSize="12" style={{ fill: 'var(--ink)' }}>{count(n)}/{count(capacity)}</text>
                </g>
              );
            })}
          </svg>
        </>
      ) : (
        <Translated vi={<p>Hãy chọn một thiết bị có số khối chạy cùng lúc.</p>}>
          <p>Choose a device with a number of blocks it runs at once.</p>
        </Translated>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- who handles what
const STEP_THREADS = 64; // two warps of 32
const WARP = 32;
const PER_ROW = 64; // eight sectors of eight floats

function MappingStepper() {
  const [C, setC] = useState(4);
  const [mapping, setMapping] = useState<Mapping>('contiguous');
  const [c, setCc] = useState(0);
  const cc = Math.min(c, C - 1);
  const units = STEP_THREADS * C;
  const addrs = warpInstruction(cc, C, mapping, WARP, STEP_THREADS);
  const acc = warpAccess(addrs, 4, SECTOR_BYTES);
  const touched = new Set(addrs.map((a) => a / 4));
  const sectors = new Set(acc.segments);
  const mine = new Set<number>();
  for (let lane = 0; lane < WARP; lane++) for (const e of elementsOf(lane, C, mapping, STEP_THREADS)) mine.add(e);
  const perInstr = Array.from({ length: C }, (_, k) => warpAccess(warpInstruction(k, C, mapping, WARP, STEP_THREADS), 4, SECTOR_BYTES));
  const loopSectors = perInstr.reduce((s, a) => s + a.transactions, 0);
  const vec = warpAccess(warpVectorInstruction(WARP), 16, SECTOR_BYTES);
  const rowsN = units / PER_ROW;
  const cell = 13;

  return (
    <Bench
      title={tr('Who handles what', 'Ai xử lý phần nào')}
      task={tr('At C = 4, step through c with each mapping and count the sectors one load instruction of warp 0 touches.', 'Với C = 4, chạy từng bước qua c với mỗi cách ánh xạ và đếm số sector mà một lệnh nạp của warp 0 chạm tới.')}
      foot={<Translated vi={<>Một lệnh nạp của warp 0 (các luồng 0 đến 31) ở vòng lặp c. Sector là đơn vị 32 byte được NVIDIA ghi trong tài liệu (ch06-facts A2): tám float. Các sector có viền là những sector mà lệnh này yêu cầu. Không có cache: hàng cho cả vòng lặp đếm lại sector của từng lệnh từ đầu.</>}>One load instruction of warp 0 (threads 0 to 31) in loop iteration c. Sectors are NVIDIA's documented 32-byte unit (ch06-facts A2): eight floats. Outlined sectors are the ones this instruction requests. No cache: the whole-loop row counts each instruction's sectors afresh.</Translated>}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('coarsening factor C', 'hệ số gộp việc C')}>
          {[1, 2, 4, 8].map((k) => (
            <button key={k} aria-pressed={k === C} onClick={() => { setC(k); setCc(0); }}>C = {k}</button>
          ))}
        </span>
        <span className="seg" role="group" aria-label={tr('mapping', 'cách ánh xạ')}>
          <button aria-pressed={mapping === 'contiguous'} onClick={() => setMapping('contiguous')}><Translated vi="liền kề (của sách)">contiguous (the book's)</Translated></button>
          <button aria-pressed={mapping === 'strided'} onClick={() => setMapping('strided')}><Translated vi="theo bước nhảy">strided</Translated></button>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setCc(Math.max(0, cc - 1))} disabled={cc === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => setCc(Math.min(C - 1, cc + 1))} disabled={cc >= C - 1}><Translated vi="c tiếp theo">Next c</Translated></button>
        <span className="muted" aria-live="polite" data-k="iteration"><Translated vi={<>vòng lặp c = {cc} trên {C - 1}</>}>loop iteration c = {cc} of {C - 1}</Translated></span>
      </div>

      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }} data-k="sectors" data-sectors={acc.transactions}>
        {mapping === 'contiguous' ? (
          <Translated vi={<>Lane k nạp phần tử {C}k + {cc}: các lane cách nhau {4 * C} byte. </>}>
            <>Lane k loads element {C}k + {cc}: lanes are {4 * C} bytes apart. </>
          </Translated>
        ) : (
          <Translated vi={<>Lane k nạp phần tử k + {cc}·{STEP_THREADS}: các lane cách nhau 4 byte. </>}>
            <>Lane k loads element k + {cc}·{STEP_THREADS}: lanes are 4 bytes apart. </>
          </Translated>
        )}
        <Translated vi={<>Lệnh này chạm tới <b>{acc.transactions} sector</b> và dùng {pct(acc.efficiency)} số byte đã nạp.</>}>
          This instruction touches <b>{acc.transactions} sectors</b> and uses {pct(acc.efficiency)} of the bytes fetched.
        </Translated>
      </div>

      <table className="threads" aria-label={tr('elements per thread', 'số phần tử mỗi luồng')} style={{ marginBottom: '0.9rem' }}>
        <thead><tr><th><Translated vi="luồng">thread</Translated></th><th><Translated vi={<>các phần tử nó xử lý (c = 0 … {C - 1})</>}>elements it handles (c = 0 … {C - 1})</Translated></th></tr></thead>
        <tbody>
          {[0, 1, 2, 31].map((t) => (
            <tr key={t}>
              <td>{t}</td>
              <td>{elementsOf(t, C, mapping, STEP_THREADS).map((e, k) => (k === cc ? <b key={k}>{e} </b> : <span key={k}>{e} </span>))}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <svg viewBox={`0 0 ${PER_ROW * cell + 70} ${rowsN * (cell + 8) + 8}`} role="img" aria-label={tr('the array, eight floats per sector', 'mảng, tám float mỗi sector')}
        style={{ width: '100%', maxWidth: PER_ROW * cell + 70, display: 'block' }}>
        {Array.from({ length: rowsN }, (_, row) => (
          <g key={row}>
            <text x={0} y={row * (cell + 8) + cell + 2} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{row * PER_ROW}</text>
            {Array.from({ length: PER_ROW / 8 }, (_, s) => {
              const sec = (row * PER_ROW) / 8 + s;
              const x = 40 + s * 8 * cell;
              const y = row * (cell + 8) + 4;
              return (
                <g key={s}>
                  {Array.from({ length: 8 }, (_, k) => {
                    const e = sec * 8 + k;
                    const on = touched.has(e);
                    return (
                      <rect key={k} x={x + k * cell} y={y} width={cell - 1} height={cell} rx="1"
                        fill={on ? 'var(--global)' : mine.has(e) ? 'var(--global-wash)' : 'var(--cell)'} stroke="var(--rule)" strokeWidth="0.5" />
                    );
                  })}
                  {sectors.has(sec) ? (
                    <rect data-sector={sec} x={x - 1} y={y - 2} width={8 * cell + 1} height={cell + 4} fill="none" stroke="var(--ink)" strokeWidth="1.4" rx="2" />
                  ) : null}
                </g>
              );
            })}
          </g>
        ))}
      </svg>
      <p className="muted ui" style={{ margin: '0.4rem 0 0.9rem' }}>
        <Translated vi="Tô đặc: các phần tử lệnh này nạp. Tô nhạt: các phần tử warp 0 xử lý ở các vòng lặp khác. Nhãn hàng là chỉ số phần tử.">
          Solid: elements this instruction loads. Washed: elements warp 0 handles in other iterations. Row labels are element indices.
        </Translated>
      </p>

      <table className="data" style={{ background: 'var(--page)', margin: 0 }} aria-label={tr('the whole coarsening loop for warp 0', 'toàn bộ vòng lặp gộp việc của warp 0')}>
        <thead><tr><th><Translated vi="warp 0, cả vòng lặp">warp 0, whole loop</Translated></th><th><Translated vi="lệnh nạp">load instructions</Translated></th><th><Translated vi="sector được yêu cầu, đếm theo từng lệnh">sectors requested, counted per instruction</Translated></th><th><Translated vi="byte được dùng">bytes used</Translated></th></tr></thead>
        <tbody>
          <tr data-k="loop">
            <td>{mapping === 'contiguous' ? tr('contiguous, scalar loads', 'liền kề, nạp vô hướng') : tr('strided, scalar loads', 'theo bước nhảy, nạp vô hướng')}</td>
            <td>{C}</td>
            <td>{loopSectors}</td>
            <td>{pct((C * WARP * 4) / (loopSectors * SECTOR_BYTES))}</td>
          </tr>
          {C === 4 && mapping === 'contiguous' ? (
            <tr data-k="vector">
              <td><Translated vi="liền kề, một lần nạp 16 byte mỗi luồng (Fig. 6.12)">contiguous, one 16-byte load per thread (Fig. 6.12)</Translated></td>
              <td>1</td>
              <td>{vec.transactions}</td>
              <td>{pct(vec.efficiency)}</td>
            </tr>
          ) : null}
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- pitfall cards
function PitfallCards() {
  const [pick, setPick] = useState<Record<string, Verdict>>({});
  return (
    <Bench title={tr('Helps or hurts?', 'Có ích hay có hại?')} task={tr('Predict each card before you open it.', 'Hãy dự đoán từng thẻ trước khi mở nó.')} foot={tr("Answers come from the book's three pitfalls (pp. 142–143), its two examples of coarsening (p. 142) and what it says vector loads buy (p. 139).", 'Đáp án lấy từ ba cạm bẫy của sách (pp. 142–143), hai ví dụ gộp việc của sách (p. 142) và điều sách nói về lợi ích của nạp dạng vector (p. 139).')}>
      <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(15rem, 1fr))' }}>
        {PITFALLS.map((p) => {
          const mine = pick[p.id];
          return (
            <div key={p.id} className="panel" data-card={p.id} style={{ background: 'var(--page)', padding: '0.8rem', borderRadius: 4 }}>
              <p style={{ marginTop: 0 }}>{p.scenario}</p>
              <div className="row" role="group" aria-label={tr('your prediction', 'dự đoán của bạn')}>
                {VERDICTS.map((v) => (
                  <button key={v} aria-pressed={mine === v} onClick={() => setPick({ ...pick, [p.id]: v })}>{tr(v, ({ helps: 'có ích', 'little or no change': 'ít hoặc không thay đổi', 'can hurt': 'có thể có hại' } as Record<string, string>)[v] ?? v)}</button>
                ))}
              </div>
              {mine ? (
                <p style={{ marginBottom: 0 }} data-k="reveal">
                  <span className={'chip ' + (mine === p.answer ? 'ok' : 'fail')}>{mine === p.answer ? tr('right', 'đúng') : tr('the answer', 'đáp án')}: {tr(p.answer, ({ helps: 'có ích', 'little or no change': 'ít hoặc không thay đổi', 'can hurt': 'có thể có hại' } as Record<string, string>)[p.answer] ?? p.answer)}</span>{' '}
                  {p.why} <span className="muted">({/[–,]/.test(p.pages) ? 'pp.' : 'p.'} {p.pages})</span>
                </p>
              ) : null}
            </div>
          );
        })}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS65: Record<string, Question> = {
  'ch06/6.5/q-factor': {
    kind: 'numeric',
    id: 'ch06/6.5/q-factor',
    prompt: (
      <Translated
        vi={
          <>
            102,400 đơn vị công việc, 256 luồng mỗi khối, một thiết bị chạy 100 khối cùng lúc. Hệ số gộp việc lớn nhất mà vẫn lấp đầy mọi chỗ trong mọi wave là bao nhiêu?
          </>
        }
      >
        <>
          102,400 work units, 256 threads per block, a device that runs 100 blocks at once. What is the largest coarsening
          factor that still fills every slot in every wave?
        </>
      </Translated>
    ),
    answer: 4,
    explain: (
      <Translated
        vi={
          <p>
            Với hệ số 1 có 102,400 luồng trong 400 khối: 4 wave đầy. Hệ số 2 cho 200 khối (2 wave), hệ số 4 cho 100 khối (một wave đầy). Hệ số 8 cho 50 khối, nửa wave: một nửa thiết bị ngồi không trong khi mỗi khối làm tám đơn vị công việc. Các hệ số ở giữa, như 3, để lại một wave chưa đầy.
          </p>
        }
      >
        <p>
          At factor 1 there are 102,400 threads in 400 blocks: 4 full waves. Factor 2 gives 200 blocks (2 waves), factor 4
          gives 100 (one full wave). Factor 8 gives 50 blocks, half a wave: half the device sits idle while each block does
          eight units of work. Factors in between, such as 3, leave a partial wave.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.5, p. 143 (the same arithmetic as the book’s 1056-block case)',
    vi: { source: 'ch. 6 §6.5, p. 143 (cùng phép tính với trường hợp 1056 khối của sách)' },
  },
  'ch06/6.5/q-independent': {
    kind: 'choice',
    id: 'ch06/6.5/q-independent',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel ReLU: mỗi luồng một phần tử, không có dữ liệu dùng chung, các lần nạp được hợp nhất. Bạn gộp việc với hệ số 4 và giữ nạp vô hướng. Bạn kỳ vọng điều gì?
          </>
        }
      >
        <>
          A ReLU kernel: one element per thread, no shared data, coalesced loads. You coarsen it by 4 and keep scalar loads.
          What do you expect?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Lợi ích rất ít hoặc không có.">Little or no gain.</Translated>, correct: true },
      { label: <Translated vi="Nhanh hơn khoảng 4×: một phần tư số luồng làm cùng khối lượng công việc.">About 4× faster: a quarter of the threads do the same work.</Translated>, why: <Translated vi="Ít luồng hơn, mỗi luồng làm nhiều hơn, thì vẫn là chừng ấy công việc. Lợi ích phải đến từ một chi phí phụ bị loại bỏ.">Fewer threads doing more work each is the same work. The gain has to come from an overhead that goes away.</Translated> },
      { label: <Translated vi="Chậm hơn nhiều, vì số luồng ít hơn bốn lần.">Much slower, because there are four times fewer threads.</Translated>, why: <Translated vi="Với một mảng lớn vẫn còn thừa luồng để lấp đầy thiết bị; cạm bẫy đó cần một lưới (grid) nhỏ.">With a large array there are still plenty of threads to fill the device; that pitfall needs a small grid.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Gộp việc chỉ có lợi khi việc song song hóa có một cái giá: nạp thừa, công việc thừa, đồng bộ hóa, chi phí lệnh. Các luồng độc lập không chia sẻ gì thì chẳng có chi phí nào để loại bỏ (cạm bẫy 1). Ngoại lệ là khi thay đổi đó cũng biến bốn lần nạp vô hướng thành một lần nạp vector, đúng như Fig. 6.12 làm: khi đó một chi phí lệnh thực sự biến mất.
          </p>
        }
      >
        <p>
          Coarsening pays only when parallelising had a cost: redundant loads, redundant work, synchronisation, instruction
          overhead. Independent threads that share nothing have none for it to remove (pitfall 1). The exception is when the
          change also turns four scalar loads into one vector load, which is what Fig. 6.12 does: then an instruction
          overhead does go away.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.5, pp. 142–143',
  },
  'ch06/6.5/q-mapping': {
    kind: 'choice',
    id: 'ch06/6.5/q-mapping',
    prompt: (
      <Translated
        vi={
          <>
            Gộp việc với hệ số 4, luồng i xử lý các phần tử 4i đến 4i + 3 bằng nạp vô hướng trong một vòng lặp theo c. Một lệnh nạp của một warp trông như thế nào?
          </>
        }
      >
        <>
          Coarsening by 4, thread i handles elements 4i to 4i + 3 with scalar loads in a loop over c. What does one load
          instruction of a warp look like?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Các lane cách nhau 16 byte: lệnh chạm tới 16 sector và dùng một phần tư mỗi sector.">Lanes 16 bytes apart: the instruction touches 16 sectors and uses a quarter of each.</Translated>, correct: true },
      { label: <Translated vi="Được hợp nhất hoàn hảo, vì mỗi luồng đọc các địa chỉ liên tiếp.">Perfectly coalesced, because each thread reads consecutive addresses.</Translated>, why: <Translated vi="Hợp nhất truy cập nói về các lane của một lệnh, không phải về một luồng theo thời gian. Trong một lệnh, các lane kề nhau cách nhau bốn float.">Coalescing is about the lanes of one instruction, not about one thread over time. In one instruction, neighbouring lanes are four floats apart.</Translated> },
      { label: <Translated vi="Được hợp nhất trên NVIDIA nhưng không trên AMD, nơi các wavefront 64 lane trải trên số byte gấp đôi.">Coalesced on NVIDIA but not on AMD, whose 64-lane wavefronts span twice as many bytes.</Translated>, why: <Translated vi="Các địa chỉ là như nhau trên mọi thiết bị. NVIDIA ghi trong tài liệu sector 32 byte mà phép đếm này dùng (A2); AMD gom các địa chỉ của một wavefront thành các dòng cache (B1).">The addresses are the same on every device. NVIDIA documents the 32-byte sector this counts with (A2); AMD groups a wavefront's addresses into cache lines (B1).</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Ở vòng lặp c, lane k nạp phần tử 4k + c. Ba mươi hai lane trải trên 128 float, mười sáu sector 32 byte, và chỉ dùng 4 trong mỗi 16 byte đã nạp. Có hai cách sửa: biến bốn lần nạp thành một lần nạp 16 byte mỗi luồng (Fig. 6.12), hoặc ánh xạ theo bước nhảy, i + c·T với T luồng trong lưới, để mọi lệnh lại đọc 32 float kề nhau.
          </p>
        }
      >
        <p>
          In iteration c, lane k loads element 4k + c. Thirty-two lanes span 128 floats, sixteen 32-byte sectors, and use 4 of
          every 16 bytes fetched. Two fixes: make the four loads one 16-byte load per thread (Fig. 6.12), or map strided,
          i + c·T for T threads in the grid, so that every instruction reads 32 neighbouring floats again.
        </p>
      </Translated>
    ),
    source: 'our analysis with the coalescing rule of §6.1; NVIDIA sector size from notes/today/ch06-facts.md A2',
    vi: { source: 'phân tích của khóa học này với quy tắc hợp nhất truy cập của §6.1; kích thước sector của NVIDIA từ notes/today/ch06-facts.md A2' },
  },
  'ch06/6.5/q-occupancy': {
    kind: 'numeric',
    id: 'ch06/6.5/q-occupancy',
    prompt: (
      <Translated
        vi={
          <>
            Lấy SM ví dụ của sách từ chương 4: 65,536 thanh ghi và 2048 chỗ cho luồng. Gộp việc làm số thanh ghi mỗi luồng tăng từ 32 lên 64. Occupancy tối đa có thể đạt bây giờ là bao nhiêu?
          </>
        }
      >
        <>
          Take the book's example SM from chapter 4: 65,536 registers and 2048 thread slots. Coarsening raises registers per
          thread from 32 to 64. What occupancy is now possible at most?
        </>
      </Translated>
    ),
    answer: 50,
    unit: '%',
    explain: (
      <Translated
        vi={
          <p>
            65,536 ÷ 64 = 1024 luồng, một nửa của 2048. Trước đó, 65,536 ÷ 32 = 2048 lấp đầy mọi chỗ. Kernel đã gộp việc giờ chỉ còn một nửa số warp để luân phiên trong khi chờ bộ nhớ, nên lợi ích của nó phải vượt quá phần mà các warp đó từng che giấu (cạm bẫy 3).
          </p>
        }
      >
        <p>
          65,536 ÷ 64 = 1024 threads, half of 2048. Before, 65,536 ÷ 32 = 2048 filled every slot. The coarsened kernel now
          has half the warps to switch between while it waits on memory, so its gain has to exceed what those warps were
          hiding (pitfall 3).
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.5, p. 143; the example limits from ch. 4 §4.7, pp. 85–86',
    vi: { source: 'ch. 6 §6.5, p. 143; các giới hạn ví dụ từ ch. 4 §4.7, pp. 85–86' },
  },
  'ch06/6.5/q-scalability': {
    kind: 'choice',
    id: 'ch06/6.5/q-scalability',
    prompt: <Translated vi={<>Bạn đã tinh chỉnh một hệ số gộp việc trên một GPU. Điều gì xảy ra trên một GPU khác, hoặc với một kích thước đầu vào khác?</>}>You tuned a coarsening factor on one GPU. What happens on another GPU, or with another input size?</Translated>,
    options: [
      { label: <Translated vi="Có thể phải tinh chỉnh lại, thậm chí biên dịch lại: hệ số tốt nhất phụ thuộc vào thiết bị và dữ liệu.">It may need retuning, even recompiling: the best factor depends on the device and the data.</Translated>, correct: true },
      { label: <Translated vi="Nó vẫn dùng được, vì các khối được lập lịch một cách trong suốt.">It carries over, because blocks are scheduled transparently.</Translated>, why: <Translated vi="Lập lịch trong suốt là thứ mà mỗi luồng một phần tử mang lại. Một hệ số cố định ngầm cài sẵn một giả định về số khối thiết bị chạy cùng lúc.">Transparent scheduling is what one thread per element buys. A fixed factor bakes in an assumption about how many blocks the device runs at once.</Translated> },
      { label: <Translated vi="Trình biên dịch chọn một hệ số mới cho từng thiết bị.">The compiler picks a new factor for each device.</Translated>, why: <Translated vi="Hệ số nằm trong mã nguồn của bạn (một hằng như COARSEN); không có gì chọn nó thay bạn.">The factor is in your source (a constant like COARSEN); nothing chooses it for you.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hệ số đúng phụ thuộc vào số khối thiết bị chạy cùng lúc và vào số đơn vị công việc. Kết luận của sách: gộp việc làm khả năng mở rộng kém trong suốt hơn so với mỗi luồng một phần tử.
          </p>
        }
      >
        <p>
          The right factor depends on how many blocks the device runs at once and on how many units of work there are. The
          book's conclusion: coarsening makes scalability less transparent than one thread per element.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.5, p. 143',
  },
};

export function S65() {
  return (
    <>
      <Translated
        vi={
          <p>
            Mọi kernel cho đến giờ đều dùng mức chia nhỏ nhất có thể: mỗi luồng một phần tử đầu ra trong phép cộng vector, một pixel trong chuyển ảnh xám và làm mờ, một phần tử đầu ra trong phép nhân ma trận. Lựa chọn đó mang lại <a href={href('ch04', '4.3')}>khả năng mở rộng trong suốt</a>: trên một thiết bị đủ lớn, mọi thứ chạy cùng lúc; trên một thiết bị nhỏ hơn, phần cứng chạy các khối theo từng wave, và mã không thay đổi.
          </p>
        }
      >
        <p>
          Every kernel so far has used the finest grain there is: one thread per output element in vector addition, per
          pixel in grayscale and blur, per output entry in matrix multiplication. That choice buys{' '}
          <a href={href('ch04', '4.3')}>transparent scalability</a>: on a device large enough, everything runs at once; on
          a smaller one, the hardware runs the blocks in waves, and the code does not change.
        </p>
      </Translated>
      <Source section="6.5" pages="141" />

      <h2>
        <Translated vi="Song song hóa tốn gì">What parallelism costs</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chia công việc cho nhiều luồng không miễn phí. Các khối khác nhau có thể nạp cùng dữ liệu, các luồng có thể lặp lại công việc mà một luồng lẽ ra chỉ cần làm một lần, chúng phải đồng bộ hóa, và mọi luồng đều tự thực thi phép tính chỉ số và kiểm tra biên của mình. Khi các luồng thực sự chạy cùng lúc, cái giá đó mua được tốc độ. Khi phần cứng dù sao cũng chạy chúng hết wave này đến wave khác, nó chẳng mua được gì.
          </p>
        }
      >
        <p>
          Splitting work across threads is not free. Different blocks may load the same data, threads may repeat work one
          thread could have done once, they have to synchronise, and every thread executes its own index arithmetic and
          bounds checks. When the threads truly run at the same time, that price buys speed. When the hardware runs them one
          wave after another anyway, it buys nothing.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Trong trường hợp đó, hãy tự tuần tự hóa một phần công việc: giao cho mỗi luồng vài đơn vị. Đó là <b>gộp việc cho luồng (thread coarsening)</b>. Số đơn vị mỗi luồng là <b>hệ số gộp việc (coarsening factor)</b>, và vòng lặp mà một luồng chạy qua các đơn vị của nó là <b>vòng lặp gộp việc (coarsening loop)</b>.
          </p>
        }
      >
        <p>
          In that case, serialise part of the work yourself: give each thread several units. That is{' '}
          <b>thread coarsening</b>. The number of units per thread is the <b>coarsening factor</b>, and the loop a thread
          runs over its units is the <b>coarsening loop</b>.
        </p>
      </Translated>
      <Source section="6.5" pages="141–142" />
      <Code
        tabs={[{ label: 's65_coarsen_toy.cu', code: TOY, status: tr('front-end-checked; unverified on CUDA and HIP hardware', 'front-end-checked; unverified trên phần cứng CUDA và HIP') }]}
        highlight={[15, 16, 17, 18, 19]}
        caption={
          <Translated vi={<>Ví dụ đồ chơi của sách trước và sau khi gộp việc với hệ số 4, từ <code>labs/ch06/snippets/</code>. Luồng 0 xử lý các phần tử 0 đến 3, luồng 1 các phần tử 4 đến 7. Các kiểm tra biên và kernel thứ ba là của khóa học này.</>}>
            The book's toy before and after coarsening by 4, from <code>labs/ch06/snippets/</code>. Thread 0 handles
            elements 0 to 3, thread 1 elements 4 to 7. The bounds checks and the third kernel are ours.
          </Translated>
        }
      />

      <h2>
        <Translated vi="Bạn đã gặp nó rồi">You have already seen it</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Kernel nạp dạng vector của Fig. 6.12 chính là gộp việc. Mỗi luồng cộng bốn cặp phần tử thay vì một, và chi phí được loại bỏ là chi phí lệnh: bốn lệnh nạp cho mỗi đầu vào trở thành một lần nạp 16 byte. Điều đó chỉ được vì bốn phần tử của luồng nằm cạnh nhau.
          </p>
        }
      >
        <p>
          The vector-load kernel of Fig. 6.12 is coarsening. Each thread adds four pairs of elements instead of one, and the
          overhead removed is instruction overhead: four load instructions per input become one 16-byte load. That only works
          because the thread's four elements sit next to each other.
        </p>
      </Translated>
      <Source section="6.5" pages="142" figs="Fig. 6.12" />
      <Translated
        vi={
          <p>
            Điều này đặt ra một câu hỏi mà sách không hỏi: với nạp vô hướng, cách ánh xạ của sách (luồng i lấy 4i đến 4i + 3) có còn được hợp nhất không? Việc hợp nhất được quyết định theo từng lệnh, trên các lane của một warp.
          </p>
        }
      >
        <p>
          Which raises a question the book does not ask: with scalar loads, is the book's mapping (thread i takes 4i to
          4i + 3) still coalesced? Coalescing is decided per instruction, across the lanes of a warp.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Phép so sánh này là của khóa học này: quy tắc hợp nhất truy cập của §6.1 áp dụng cho ví dụ đồ chơi, đếm theo sector 32 byte của NVIDIA. Cách ánh xạ theo bước nhảy cũng là của khóa học này.
            </p>
          }
        >
          <p>
            This comparison is ours: the coalescing rule of §6.1 applied to the toy, counted in NVIDIA's 32-byte sectors.
            The strided mapping is also ours.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Chạy từng bước qua vòng lặp bên dưới với cả hai cách ánh xạ, mỗi lần một lệnh nạp của warp 0, và đếm số sector mà mỗi lệnh yêu cầu.
          </p>
        }
      >
        <p>
          Step through the loop below with both mappings, one load instruction of warp 0 at a time, and count the sectors
          each instruction requests.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi lần một lệnh, theo sector 32 byte của NVIDIA, không có cache: các sector mà những vòng lặp sau nạp lại sẽ được đếm lại, dù một cache có thể phục vụ chúng.
            </p>
          }
        >
          <p>
            One instruction at a time, in NVIDIA&rsquo;s 32-byte sectors, with no cache: sectors that later iterations
            fetch again are counted again, although a cache might serve them.
          </p>
        </Translated>
      </Simplification>
      <MappingStepper />
      <Ask q={questionsS65['ch06/6.5/q-mapping']} />
      <Translated
        vi={
          <p>
            Vậy cách ánh xạ liền kề của sách đi tự nhiên với nạp dạng vector, còn cách theo bước nhảy đi với nạp vô hướng. Dù cách nào thì mảng cũng được đọc một lần; điều khác nhau là mỗi lệnh kéo theo bao nhiêu sector.
          </p>
        }
      >
        <p>
          So the book's contiguous mapping pairs naturally with vector loads, and the strided one with scalar loads. Either
          way the array is read once; what differs is how many sectors each instruction drags in.
        </p>
      </Translated>

      <h2>
        <Translated vi="Gộp việc cho phép nhân ma trận phân ô">Coarsening the tiled matrix multiplication</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Kernel phân ô của chương 5 là ví dụ còn lại mà sách đưa ra. Mỗi khối nạp các ô của M và N, và các khối kề nhau nạp nhiều ô giống nhau. Ít khối hơn, mỗi khối tính một ô đầu ra lớn hơn, sẽ nạp mỗi ô đầu vào ít lần hơn: chi phí được loại bỏ là việc nạp thừa.
          </p>
        }
      >
        <p>
          Chapter 5's tiled kernel is the other example the book gives. Each block loads tiles of M and N, and neighbouring
          blocks load many of the same tiles. Fewer blocks, each computing a larger tile of the output, load each input tile
          fewer times: the overhead removed is redundant loading.
        </p>
      </Translated>
      <Source section="6.5" pages="142" />
      <Translated
        vi={
          <p>
            Chương 15 làm việc này một cách đầy đủ, và kernel gộp việc, phân ô thanh ghi của nó là mốc dự án GEMM thứ ba của khóa học này.
          </p>
        }
      >
        <p>
          Chapter 15 does this properly, and its coarsened, register-tiled kernel is this course's third GEMM milestone.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md, A15)">{/* i18n-ignore: date and notes reference */}
        <Translated
          vi={
            <p>
              Best Practices Guide của NVIDIA khuyến nghị cùng ý tưởng cho các kernel dùng bộ nhớ chia sẻ: một luồng xử lý vài phần tử, để phần việc chung của chúng chỉ làm một lần. Không có API nào cho việc này; đó là một cách viết kernel.
            </p>
          }
        >
          <p>
            NVIDIA's Best Practices Guide recommends the same idea for shared-memory kernels: one thread processing several
            elements, so that work common to them is done once. There is no API for it; it is a way of writing the kernel.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Ba cách làm sai">Three ways to get it wrong</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách nêu ba cạm bẫy. Hãy thử các thẻ trước; phần còn lại của mục sẽ mổ xẻ từng cái.
          </p>
        }
      >
        <p>
          The book names three pitfalls. Try the cards first; the rest of the section takes each one apart.
        </p>
      </Translated>
      <PitfallCards />

      <h3>
        <Translated vi="Gộp việc khi chẳng có gì để loại bỏ">Coarsening when there is nothing to remove</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Nếu các luồng độc lập, không chia sẻ dữ liệu và không cần phối hợp, thì không có chi phí song song hóa, và gộp việc sẽ thay đổi rất ít.
          </p>
        }
      >
        <p>
          If the threads are independent, share no data and need no coordination, there is no parallelisation overhead, and
          coarsening should change little.
        </p>
      </Translated>
      <Source section="6.5" pages="142–143" />
      <Ask q={questionsS65['ch06/6.5/q-independent']} />

      <h3>
        <Translated vi="Gộp việc quá mức">Coarsening too much</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Gộp việc trao cho phần cứng ít tính song song hơn. Thiết bị của sách chạy 264 khối cùng lúc.
          </p>
        }
      >
        <p>
          Coarsening hands the hardware less parallelism. The book's device runs 264 blocks at once.
        </p>
      </Translated>
      <Source section="6.5" pages="143" />
      <Translated
        vi={
          <p>
            Một lưới 1056 khối là bốn wave đầy; gộp việc với hệ số 8 thì còn 132 khối, nửa wave, và một nửa thiết bị ngồi không. Một lưới 1584 khối là sáu wave; gộp việc với hệ số 4 thì còn 396 khối, một wave rưỡi, và nửa wave cuối để thiết bị trống một nửa: hiệu ứng đuôi từ <a href={href('ch04', '4.3')}>chương 4</a>.
          </p>
        }
      >
        <p>
          A grid of 1056 blocks is four full waves; coarsened by 8 it is 132 blocks, half a wave, and half the device idles.
          A grid of 1584 blocks is six waves; coarsened by 4 it is 396 blocks, one and a half waves, and the last half wave
          leaves the device half empty: the tail effect from <a href={href('ch04', '4.3')}>chapter 4</a>.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mọi khối đều mất cùng thời gian và thiết bị giữ một số khối cố định cùng lúc. Ở đây hệ số không làm thay đổi con số đó, như lẽ ra sẽ xảy ra nếu gộp việc làm tăng thanh ghi hoặc bộ nhớ chia sẻ mỗi khối; hãy tự hạ nó xuống để thấy cạm bẫy đó. “Thời gian theo wave” chỉ tính phần song song bị mất, không tính phần chi phí tiết kiệm được.
            </p>
          }
        >
          <p>
            Every block takes the same time and the device holds a fixed number at once. The factor does not change that
            number here, as it would if coarsening raised registers or shared memory per block; lower it by hand to see that
            pitfall. "Wave time" counts only the parallelism lost, none of the overhead saved.
          </p>
        </Translated>
      </Simplification>
      <WaveCalculator />
      <Translated
        vi={
          <p>
            Ít wave hơn không phải là mục tiêu. Gộp việc với hệ số 8, lưới 1056 khối chạy trong một wave thay vì bốn, nhưng mỗi khối giờ làm tám đơn vị công việc: nếu gộp việc không tiết kiệm được gì, nó tốn thời gian bằng tám wave cũ. Các wave trọn vẹn giữ cho thiết bị bận; còn bản thân lợi ích phải đến từ chi phí được loại bỏ.
          </p>
        }
      >
        <p>
          Fewer waves is not the goal. Coarsened by 8, the 1056-block grid runs in one wave instead of four, but each block
          now does eight units of work: if coarsening saved nothing, it takes as long as eight of the old waves. Whole waves
          keep the device busy; the gain itself has to come from the overhead removed.
        </p>
      </Translated>
      <Ask q={questionsS65['ch06/6.5/q-factor']} label={tr('Work it out', 'Tự tính ra')} />

      <h3>
        <Translated vi="Mất occupancy">Losing occupancy</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Một luồng đã gộp việc thường giữ nhiều trạng thái hơn: vài kết quả đang tích lũy, vài đầu vào. Nếu điều đó làm tăng thanh ghi mỗi luồng hoặc bộ nhớ chia sẻ mỗi khối đủ để hạ occupancy, kernel sẽ có ít warp hơn để luân phiên trong khi chờ, và lợi ích của gộp việc phải bù lại phần đó.
          </p>
        }
      >
        <p>
          A coarsened thread usually keeps more state: several running results, several inputs. If that raises registers per
          thread or shared memory per block enough to lower occupancy, the kernel has fewer warps to switch between while it
          waits, and the coarsening gain has to make up for that.
        </p>
      </Translated>
      <Source section="6.5" pages="143" />
      <Ask q={questionsS65['ch06/6.5/q-occupancy']} label={tr('Work it out', 'Tự tính ra')} />

      <h2>
        <Translated vi="Không hệ số nào mang đi đâu cũng tốt">No factor travels well</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hệ số tốt nhất phụ thuộc vào thiết bị và dữ liệu, và thường phải tìm bằng cách đo. Chuyển sang GPU khác, hoặc kích thước bài toán khác, có thể đồng nghĩa với việc tinh chỉnh lại và biên dịch lại. Đó là cái giá: một kernel đã gộp việc mở rộng kém trong suốt hơn so với kernel mỗi luồng một phần tử.
          </p>
        }
      >
        <p>
          The best factor depends on the device and on the data, and it usually has to be found by measuring. Moving to
          another GPU, or another problem size, can mean retuning it and recompiling. That is the price: a coarsened kernel
          scales less transparently than one with a thread per element.
        </p>
      </Translated>
      <Source section="6.5" pages="143" />
      <Ask q={questionsS65['ch06/6.5/q-scalability']} />
    </>
  );
}
