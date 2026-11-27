import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Aside, Bench, Simplification, Source, Today, type Tier } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type BenchRow } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  DESIGNS,
  DESIGN_LABEL,
  PATTERN_LABEL,
  PRESET,
  isCoarse,
  makeKeep,
  presetMap,
  storeMap,
  type Design,
  type Pattern,
  type StoreMap,
} from '../../sim/compactstore';
import { TRAFFIC_DESIGNS, TRAFFIC_LABEL, traffic, type TrafficDesign } from '../../sim/stablescan';
import { functionSource } from '../../sim/toolchain';
import { StageWriteScene } from './StageWriteScene';

// Section 12.6: shared-memory gathering and thread coarsening for the stable filter (Figs.
// 12.9 and 12.10, pp. 298-300). The explorer runs on src/sim/compactstore.ts (with chapter 6's
// warpAccess); the bytes table on stablescan.ts's traffic().

const TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', blockCounts: 'global', blockOffsets: 'global', outputSize: 'global',
  staged_s: 'shared', warpTotals_s: 'shared', groupTotals: 'shared',
  i: 'reg', key: 'reg', keep: 'reg', staged: 'reg', base: 'reg', rankInWarp: 'reg', warpCount: 'reg', r: 'reg', j: 'reg',
};

const pct = (x: number) => `${Math.round(x * 100)}%`;

function Strip({ m, block }: { m: StoreMap; block: number }) {
  const run = m.runs[block];
  const cap = 256;
  const ROW = 32; // slots per row: 128 bytes
  const segSlots = m.cfg.segmentBytes / 4;
  const start = Math.floor(run.start / ROW) * ROW;
  const end = Math.min(run.start + Math.min(run.length, cap), m.totals.kept);
  const rows = Math.max(1, Math.ceil((end - start) / ROW));
  const C = 17;
  const mine = new Map<number, number>(); // slot -> store number within the block
  m.stores.filter((s) => s.block === block).forEach((s, k) => s.slots.forEach((slot) => mine.set(slot, k)));
  return (
    <div style={{ overflowX: 'auto' }}>
      <svg viewBox={`0 0 ${ROW * C + 60} ${rows * (C + 8) + 6}`} style={{ width: '100%', minWidth: 480, maxWidth: ROW * C + 60, display: 'block' }}
        role="img" aria-label={tr(`Output slots of block ${block}, by the store instruction that writes them`, `Các ô đầu ra của khối ${block}, theo lệnh ghi ghi vào chúng`)} data-strip={block}>
        {Array.from({ length: rows }, (_, r) => (
          <g key={r}>
            <text x={52} y={r * (C + 8) + 13} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>{start + r * ROW}</text>
            {Array.from({ length: ROW }, (_, c) => {
              const j = start + r * ROW + c;
              if (j >= m.totals.kept) return null;
              const k = mine.get(j);
              const x = 58 + c * C;
              const y = r * (C + 8) + 2;
              return (
                <g key={c} data-slot={j} data-store={k ?? ''}>
                  <rect x={x} y={y} width={C - 1} height={C - 1} rx="1.5"
                    fill={k == null ? 'var(--cell)' : k % 2 ? 'var(--global)' : 'var(--global-wash)'}
                    stroke={k == null ? 'var(--rule)' : 'var(--global)'} strokeOpacity={k == null ? 1 : 0.7} />
                  {k != null ? (
                    <text x={x + (C - 1) / 2} y={y + 11.5} fontSize="8.5" textAnchor="middle" style={{ fill: k % 2 ? 'var(--page)' : 'var(--ink)' }}>{k + 1}</text>
                  ) : null}
                  {j % segSlots === 0 ? <line x1={x - 1} y1={y - 2} x2={x - 1} y2={y + C + 1} stroke="var(--ink)" strokeWidth={1.6} /> : null}
                </g>
              );
            })}
          </g>
        ))}
      </svg>
      {run.length > cap ? <p className="muted ui" style={{ margin: '0.2rem 0 0' }}><Translated vi={<>Đang hiển thị {cap} ô đầu tiên trong dải {run.length} ô của khối này.</>}>Showing the first {cap} slots of this block's run of {run.length}.</Translated></p> : null}
    </div>
  );
}

function Explorer() {
  const [n, setN] = useState<number>(PRESET.n);
  const [f, setF] = useState<number>(PRESET.fraction);
  const [pattern, setPattern] = useState<Pattern>(PRESET.pattern);
  const [blockDim, setBlockDim] = useState<number>(PRESET.blockDim);
  const [warp, setWarp] = useState<number>(PRESET.warp);
  const [coarse, setCoarse] = useState<number>(4);
  const [seg, setSeg] = useState<number>(PRESET.segmentBytes);
  const [design, setDesign] = useState<Design>('direct');
  const [block, setBlock] = useState(0);
  const keep = useMemo(() => makeKeep(n, f, pattern, PRESET.seed), [n, f, pattern]);
  const maps = useMemo(
    () => Object.fromEntries(DESIGNS.map((d) => [d, storeMap({ keep, blockDim, warp, coarse, segmentBytes: seg, design: d })])) as Record<Design, StoreMap>,
    [keep, blockDim, warp, coarse, seg],
  );
  const m = maps[design];
  const b = Math.min(block, m.totals.blocks - 1);
  const mine = m.stores.filter((s) => s.block === b);
  const select = (label: string, value: number, set: (v: number) => void, options: number[], disabled = false) => (
    <label className="field">
      {label}
      <select value={value} disabled={disabled} onChange={(e) => { set(Number(e.target.value)); setBlock(0); }} aria-label={label}>
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </label>
  );

  return (
    <Bench
      title={tr('Who writes each output slot?', 'Ai ghi vào mỗi ô đầu ra?')}
      task={tr(
        'Compare Fig. 12.8 with Fig. 12.9 at a quarter kept and at nearly all kept. Then coarsen, with and without staging.',
        'So sánh Fig. 12.8 với Fig. 12.9 khi giữ một phần tư và khi giữ gần như tất cả. Sau đó gộp việc, có và không có gom tạm.',
      )}
      foot={
        <Translated
          vi={
            <>
              Mọi thiết kế đều ghi cùng một đầu ra. Các ô tô màu thuộc về khối đang chọn; con số trong một ô là lệnh ghi
              (của khối này) ghi vào nó, và một vạch đậm đánh dấu ranh giới đoạn. Một đoạn là đơn vị 32 byte mà NVIDIA ghi
              trong tài liệu, hoặc một dòng cache trên AMD; Apple không ghi gì, nên trên Metal kích thước do bạn chọn. Mô
              hình: src/sim/compactstore.ts, đếm các đoạn của mỗi lần ghi bằng mô hình của chương 6.
            </>
          }
        >
          Every design writes the same output. Shaded slots belong to the selected block; the number in a slot is the store
          instruction (of this block) that writes it, and a heavy tick marks a segment boundary. A segment is NVIDIA's
          documented 32-byte unit, or a cache line on AMD; Apple documents none, so on Metal the size is your choice.
          Model: src/sim/compactstore.ts, which counts each store's segments with chapter 6's model.
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          {tr('Design', 'Thiết kế')}
          <select value={design} onChange={(e) => setDesign(e.target.value as Design)} aria-label={tr('Design', 'Thiết kế')} style={{ maxWidth: '100%' }}>
            {DESIGNS.map((d) => <option key={d} value={d}>{tr(DESIGN_LABEL[d], { direct: 'Fig. 12.8: mỗi luồng được giữ tự ghi khóa của mình', staged: 'Fig. 12.9: gom tạm trong bộ nhớ chia sẻ, rồi chép ra', 'staged-coarse': 'Fig. 12.10: gom tạm và gộp việc', 'direct-coarse': 'gộp việc, không gom tạm (của khóa học này)' }[d])}</option>)}
          </select>
        </label>
        <label className="field">
          {tr('Kept:', 'Được giữ:')} {pattern === 'alternate' ? tr('50% (every other)', '50% (cứ cách một)') : pct(f)}
          <input type="range" min="0" max="100" value={Math.round(f * 100)} disabled={pattern === 'alternate'} aria-label={tr('Fraction kept', 'Tỉ lệ được giữ')}
            onChange={(e) => setF(Number(e.target.value) / 100)} />
        </label>
        <label className="field">
          {tr('Pattern', 'Mẫu')}
          <select value={pattern} onChange={(e) => setPattern(e.target.value as Pattern)} aria-label={tr('Pattern', 'Mẫu')}>
            {(['random', 'runs', 'alternate'] as Pattern[]).map((p) => <option key={p} value={p}>{tr(PATTERN_LABEL[p], { random: 'ngẫu nhiên', runs: 'theo dải', alternate: 'cứ cách một khóa' }[p])}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        {select(tr('Keys', 'Số khóa'), n, setN, [64, 128, 256, 512, 1024])}
        {select(tr('Threads per block', 'Số luồng mỗi khối'), blockDim, setBlockDim, [16, 32, 64, 128, 256])}
        {select(tr('Warp width', 'Độ rộng warp'), warp, setWarp, [8, 32, 64])}
        {select(tr('Coarsening factor', 'Hệ số gộp việc'), coarse, setCoarse, [1, 2, 4, 8], !isCoarse(design))}
        {select(tr('Segment bytes', 'Số byte mỗi đoạn'), seg, setSeg, [32, 64, 128])}
        <label className="field">
          {tr('Block', 'Khối')}
          <select value={b} onChange={(e) => setBlock(Number(e.target.value))} aria-label={tr('Block', 'Khối')}>
            {Array.from({ length: m.totals.blocks }, (_, k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
      </div>

      <div className="mx-label t-global">
        <Translated
          vi={
            <>
              dải của khối {b}: các ô đầu ra {m.runs[b].start} đến {m.runs[b].start + m.runs[b].length - 1} ({m.runs[b].length} khóa), được ghi bởi {mine.length} lệnh ghi
            </>
          }
        >
          block {b}'s run: output slots {m.runs[b].start} to {m.runs[b].start + m.runs[b].length - 1} ({m.runs[b].length} keys), written by {mine.length} store instruction{mine.length === 1 ? '' : 's'}
        </Translated>
      </div>
      <Strip m={m} block={b} />

      <div style={{ overflowX: 'auto', margin: '0.9rem 0 0.6rem' }}>
      <table className="data" style={{ background: 'var(--page)', margin: 0, minWidth: '46rem' }} aria-label={tr('All four designs at these settings', 'Cả bốn thiết kế ở các thiết lập này')}>
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}><Translated vi="thiết kế">design</Translated></th>
            <th className="num"><Translated vi="số lệnh ghi">store instructions</Translated></th>
            <th className="num"><Translated vi="số đoạn bị chạm">segments touched</Translated></th>
            <th className="num"><Translated vi="byte hữu ích">useful bytes</Translated></th>
            <th className="num"><Translated vi="số đoạn liền (độ dài trung bình)">chunks (mean length)</Translated></th>
            <th className="num"><Translated vi="gom tạm mỗi khối">staging per block</Translated></th>
            <th className="num"><Translated vi="đầu vào scan mỗi khóa">scan inputs per key</Translated></th>
            <th className="num"><Translated vi="tỉ phần lưu lượng của lần ghi">stores' share of traffic</Translated></th>
          </tr>
        </thead>
        <tbody>
          {DESIGNS.map((d) => {
            const t = maps[d].totals;
            return (
              <tr key={d} data-design={d} style={d === design ? { background: 'var(--rule-soft)', fontWeight: 600 } : undefined}>
                <td style={{ paddingLeft: '0.6rem' }}>{tr(DESIGN_LABEL[d], { direct: 'Fig. 12.8: mỗi luồng được giữ tự ghi khóa của mình', staged: 'Fig. 12.9: gom tạm trong bộ nhớ chia sẻ, rồi chép ra', 'staged-coarse': 'Fig. 12.10: gom tạm và gộp việc', 'direct-coarse': 'gộp việc, không gom tạm (của khóa học này)' }[d])}</td>
                <td className="num" data-stores={t.stores}>{t.stores}</td>
                <td className="num" data-segments={t.segments}>{t.segments}</td>
                <td className="num">{pct(t.efficiency)}</td>
                <td className="num" data-chunks={t.chunks}>{t.chunks} ({fmt(t.meanChunk)})</td>
                <td className="num" data-staging={t.stagingBytes}>{t.stagingBytes ? `${t.stagingBytes} B` : tr('none', 'không có')}</td>
                <td className="num">{fmt(t.scanInputsPerKey)}</td>
                <td className="num">{pct(t.storeShare)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      </div>
      <p className="muted ui" style={{ margin: '0 0 0.8rem' }}>
        <Translated
          vi={
            <>
              {m.totals.kept} trong {m.totals.keys} khóa được giữ, {m.totals.blocks} khối, mỗi khối {m.totals.keysPerBlock} khóa. "Byte
              hữu ích" là tỉ phần byte của các đoạn được nạp có chứa khóa được giữ. "Tỉ phần của lần ghi" đếm các đoạn ghi so với việc
              đọc mỗi khóa một lần.
            </>
          }
        >
          {m.totals.kept} of {m.totals.keys} keys kept, {m.totals.blocks} block{m.totals.blocks === 1 ? '' : 's'} of {m.totals.keysPerBlock} keys. "Useful bytes" is the
          share of the fetched segments' bytes that hold kept keys. "Stores' share" counts the store segments against reading
          every key once.
        </Translated>
      </p>

      <details>
        <summary className="ui"><Translated vi={<>Các lệnh ghi của khối {b} ({mine.length})</>}>Block {b}'s store instructions ({mine.length})</Translated></summary>
        <table className="threads" style={{ marginTop: '0.4rem' }} aria-label={tr('Store instructions of the selected block', 'Các lệnh ghi của khối đang chọn')}>
          <thead>
            <tr><th><Translated vi="lần ghi">store</Translated></th><th><Translated vi="warp">warp</Translated></th><th>{isCoarse(design) ? tr('round', 'vòng') : ''}</th><th><Translated vi="lane hoạt động">active lanes</Translated></th><th><Translated vi="ô đầu ra">output slots</Translated></th><th><Translated vi="đoạn">segments</Translated></th></tr>
          </thead>
          <tbody>
            {mine.slice(0, 40).map((s, k) => {
              const consecutive = s.slots.every((x, q) => q === 0 || x === s.slots[q - 1] + 1);
              return (
                <tr key={s.id}>
                  <td>{k + 1}</td>
                  <td>{s.warp}</td>
                  <td>{isCoarse(design) ? s.round : ''}</td>
                  <td>{s.lanes.length}</td>
                  <td>{consecutive ? `${s.slots[0]}–${s.slots[s.slots.length - 1]}` : `${s.slots.slice(0, 6).join(', ')}${s.slots.length > 6 ? ', …' : ''}`}</td>
                  <td>{s.segments.length}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {mine.length > 40 ? <p className="muted ui"><Translated vi={<>và {mine.length - 40} lệnh nữa</>}>and {mine.length - 40} more</Translated></p> : null}
      </details>
    </Bench>
  );
}

const P_DIRECT = presetMap('direct').totals;
const P_STAGED = presetMap('staged').totals;

/** Lab 12.3's rows nearest a kept fraction: part A ("flags") and part B ("staged", best coarsening factor). */
function measured(rows: BenchRow[], f: number) {
  const near = (name: string) => {
    const rs = rows.filter((r) => r.name === name && num(r.kept_fraction) != null);
    if (!rs.length) return undefined;
    const best = Math.min(...rs.map((r) => Math.abs(num(r.kept_fraction)! - f)));
    const at = rs.filter((r) => Math.abs(num(r.kept_fraction)! - f) === best);
    return at.reduce((a, r) => ((num(r.seconds) ?? Infinity) < (num(a.seconds) ?? Infinity) ? r : a));
  };
  return { flags: near('flags'), staged: near('staged') };
}

function BytesTable() {
  const [f, setF] = useState(0.5);
  const { reports, reload } = useReports();
  const devs = realBackends(reports).map((bk) => ({ bk, device: String(reports[bk].device.name ?? bk), rows: benchRows(reports[bk], 'ch12/lab3_stable') })).filter((d) => d.rows.length);
  const cell = (d: TrafficDesign, rows: BenchRow[]) => {
    if (d === 'single-pass') return <span className="muted"><Translated vi="không có trong lab 12.3">not in lab 12.3</Translated></span>;
    const m = measured(rows, f)[d === 'three-kernels' ? 'flags' : 'staged'];
    if (!m) return <span className="muted"><Translated vi="không có hàng">no row</Translated></span>;
    const eq = num(m.copy_equivalent_bytes_per_key);
    return (
      <>
        {eq != null ? <b>{fmt(eq)}</b> : '?'}
        <Translated vi={<span className="muted"> (giữ {pct(num(m.kept_fraction) ?? 0)}{d === 'staged' ? `, c = ${m.coarse}` : ''}, {fmt(num(m.gbps) ?? 0)} GB/s)</span>}>
          <span className="muted"> (at {pct(num(m.kept_fraction) ?? 0)} kept{d === 'staged' ? `, c = ${m.coarse}` : ''}, {fmt(num(m.gbps) ?? 0)} GB/s)</span>
        </Translated>
      </>
    );
  };
  return (
    <Bench
      title={tr('Bytes per key, by design', 'Số byte mỗi khóa, theo thiết kế')}
      task={tr(
        "Slide the fraction kept. Which part of each design's traffic does it change?",
        'Kéo tỉ lệ được giữ. Nó làm thay đổi phần nào trong lưu lượng của mỗi thiết kế?',
      )}
      foot={
        <Translated
          vi={
            <>
              Mô hình: traffic() trong src/sim/stablescan.ts, khóa 4 byte, khối 256 khóa; cách tính của khóa học này, không
              phải của sách. Một cột đo được xuất hiện cho mỗi thiết bị đã chạy benchmark của lab 12.3: thời gian đo được,
              quy ra số byte mỗi khóa mà một phép sao chép thường sẽ di chuyển trong cùng thời gian đó trên cùng thiết bị, để nó
              đọc theo đơn vị của mô hình.
            </>
          }
        >
          Model: traffic() in src/sim/stablescan.ts, 4-byte keys, blocks of 256 keys; our accounting, not the book's. A
          measured column appears for each device on which lab 12.3's benchmark has run: the measured time expressed as the
          bytes per key a plain copy would move in that time on the same device, so it reads in the model's units.
        </Translated>
      }
    >
      <label className="field" style={{ marginBottom: '0.7rem' }}>
        {tr('Kept:', 'Được giữ:')} {pct(f)}
        <input type="range" min="0" max="100" value={Math.round(f * 100)} aria-label={tr('Fraction kept for the bytes table', 'Tỉ lệ được giữ cho bảng byte')} onChange={(e) => setF(Number(e.target.value) / 100)} />
      </label>
      <div style={{ overflowX: 'auto' }}>
      <table className="data" style={{ background: 'var(--page)', margin: 0, minWidth: '34rem' }}>
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}><Translated vi="thiết kế">design</Translated></th>
            <th className="num"><Translated vi="đọc">reads</Translated></th>
            <th className="num"><Translated vi="ghi">writes</Translated></th>
            <th className="num"><Translated vi="mô hình, byte mỗi khóa">model, bytes per key</Translated></th>
            {devs.map((d) => <th key={d.bk}>{d.device} ({d.bk}), <Translated vi="đo được">measured</Translated></th>)}
          </tr>
        </thead>
        <tbody>
          {TRAFFIC_DESIGNS.map((d) => {
            const t = traffic(d, f, 256);
            return (
              <tr key={d} data-traffic={d}>
                <td style={{ paddingLeft: '0.6rem' }}>{tr(TRAFFIC_LABEL[d], { 'three-kernels': 'bài tập 1: cờ giữ lại, scan, rải (ba kernel)', staged: 'đếm theo khối, scan các số đếm của khối, rải có gom tạm (ba kernel)', 'single-pass': 'bài tập 2: một kernel với scan một lượt' }[d])}</td>
                <td className="num">{fmt(t.reads)}</td>
                <td className="num">{fmt(t.writes)}</td>
                <td className="num" data-total={t.total}><b>{fmt(t.total)}</b></td>
                {devs.map((dv) => <td key={dv.bk}>{cell(d, dv.rows)}</td>)}
              </tr>
            );
          })}
        </tbody>
      </table>
      </div>
      {!devs.length ? (
        <p className="ui" style={{ margin: '0.7rem 0 0' }}>
          <Translated
            vi={
              <>
                Máy này chưa có số đo nào. Chạy <code>python3 labs/pmpp.py run 12.3 --solution --bench</code> trên một máy có GPU
                (lần chạy tự ghi lại kết quả), rồi tải lại, và bảng này sẽ có thêm một cột cho mỗi thiết bị.{' '}
              </>
            }
          >
            No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 12.3 --solution --bench</code> on a
            machine with a GPU (the run records itself), then reload, and this table gains a column per device.{' '}
          </Translated>
          <button className="quiet" onClick={reload}>{tr('Reload results', 'Tải lại kết quả')}</button>
        </p>
      ) : null}
    </Bench>
  );
}

export const questionsS126: Record<string, Question> = {
  'ch12/12.6/q-direct': {
    kind: 'choice',
    id: 'ch12/12.6/q-direct',
    prompt: (
      <Translated vi={<>Trong Fig. 12.8, 12 trong 32 lane của một warp giữ khóa của mình. Các lần ghi của chúng rơi vào đâu?</>}>
        <>In Fig. 12.8, 12 of a warp's 32 lanes keep their keys. Where do their stores land?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Vào 12 ô rải rác khắp đầu ra, tùy độ lệch của mỗi lane được giữ tình cờ rơi vào đâu.">In 12 slots scattered over the output, wherever each kept lane's offset happens to fall.</Translated>,
        why: <Translated vi="Độ lệch của mỗi lane được giữ lớn hơn một so với lane được giữ trước đó, vì scan đếm đúng số khóa được giữ nằm giữa chúng: không có khóa nào.">Each kept lane's offset is one more than the previous kept lane's, because the scan counts exactly the kept keys between them: none.</Translated>,
      },
      {
        label: <Translated vi="Vào 12 ô liên tiếp, theo thứ tự lane.">In 12 consecutive slots, in lane order.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Vào 32 ô: các lane còn lại ghi số 0 vào ô của chúng.">In 32 slots: the other lanes write zeros into theirs.</Translated>,
        why: <Translated vi="Các lane có khóa bị bỏ hoàn toàn không ghi.">Lanes whose key is dropped do not store at all.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Bên trong một warp, phép rải trực tiếp vốn đã là một dải liền. Điều gom tạm sửa được là giữa các warp: mỗi warp giữ
            bất kỳ thứ gì đều phát một lần ghi ngắn của riêng nó, một dải có thể vắt qua ranh giới đoạn, và chỉ việc gom các khóa
            của khối trước mới gộp những dải đó thành ít lần ghi hơn, đầy hơn.
          </p>
        }
      >
        <p>
          Inside a warp the direct scatter is already a single run. What staging fixes is between warps: every warp that keeps
          anything issues its own short store, a run can straddle a segment boundary, and only gathering the block's keys
          first merges those runs into fewer, fuller stores.
        </p>
      </Translated>
    ),
    source: 'ch. 12 §12.6, p. 298; our analysis with src/sim/compactstore.ts',
    vi: { source: 'ch. 12 §12.6, p. 298; phân tích của khóa học này với src/sim/compactstore.ts' },
  },
  'ch12/12.6/q-chunks': {
    kind: 'numeric',
    id: 'ch12/12.6/q-chunks',
    prompt: (
      <Translated vi={<>4096 khóa, khối 128 luồng, hệ số gộp việc 4, gom tạm như trong Fig. 12.10. Bộ lọc ghi bao nhiêu đoạn liền?</>}>
        <>4096 keys, blocks of 128 threads, coarsening factor 4, staged as in Fig. 12.10. How many contiguous chunks does the filter write?</>
      </Translated>
    ),
    answer: 8,
    explain: (
      <Translated vi={<p>Mỗi khối nhận 128 × 4 = 512 khóa và ghi các khóa được giữ của nó thành một đoạn liền: 4096 / 512 = 8 đoạn (nếu khối nào cũng giữ được gì đó).</p>}>
        <p>Each block takes 128 × 4 = 512 keys and writes its kept keys as one chunk: 4096 / 512 = 8 chunks (if every block keeps something).</p>
      </Translated>
    ),
    source: 'ch. 12 §12.6, p. 299, Fig. 12.10',
  },
  'ch12/12.6/q-scan-coarse': {
    kind: 'choice',
    id: 'ch12/12.6/q-scan-coarse',
    prompt: (
      <Translated vi={<>Theo sách, ngoài các đoạn liền dài hơn, gộp việc mang lại gì cho bộ lọc ổn định?</>}>
        <>Besides longer chunks, what does coarsening buy the stable filter, according to the book?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Ít lần đọc khóa từ bộ nhớ toàn cục hơn: mỗi luồng nạp c khóa của nó một lần, giữ trong thanh ghi và dùng lại cho lần ghi.">Fewer global reads of the keys: each thread loads its c keys once, keeps them in registers and reuses them for the store.</Translated>,
        why: <Translated vi="Thiết kế nào cũng đọc mỗi khóa từ bộ nhớ toàn cục; gộp việc thay đổi ai đọc nó, không phải đọc bao nhiêu lần.">Every design reads each key from global memory; coarsening changes who reads it, not how often.</Translated>,
      },
      {
        label: <Translated vi="Tính ổn định: không gộp việc, các khóa được giữ từ những warp kề nhau có thể đổi chỗ trong đầu ra.">Stability: without coarsening, kept keys from neighbouring warps can swap places in the output.</Translated>,
        why: <Translated vi="Bộ lọc vốn đã ổn định; scan cho mỗi khóa đúng ô của nó dù có gộp việc hay không.">The filter was already stable; the scan gives every key its exact slot with or without coarsening.</Translated>,
      },
      {
        label: <Translated vi="Một phép scan rẻ hơn: mỗi luồng một số đếm, hoặc mỗi warp một tổng, đi vào phép scan cấp khối nặng rào chắn thay cho nhiều khóa.">A cheaper scan: one count per thread, or one total per warp, enters the barrier-heavy block scan for many keys.</Translated>,
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Sách gọi scan là phần tốn kém nhất của bộ lọc ổn định, và chương 11 đã cho thấy gộp việc làm gì cho một phép scan:
            mỗi luồng đếm tuần tự các khóa của mình, việc này rẻ, và chỉ một giá trị mỗi luồng đi vào phép scan cấp khối cùng các
            rào chắn của nó. Trong trình khám phá, "đầu vào scan mỗi khóa" giảm theo 1/c.
          </p>
        }
      >
        <p>
          The book calls the scan the most expensive part of the stable filter, and chapter 11 showed what coarsening does for a
          scan: each thread counts its own keys sequentially, which is cheap, and only one value per thread enters the
          block-level scan with its barriers. In the explorer, "scan inputs per key" falls as 1/c.
        </p>
      </Translated>
    ),
    source: 'ch. 12 §12.6, pp. 299–300; ch. 11 §11.6, pp. 267–270',
    vi: { source: 'chương 12 §12.6, pp. 299–300; chương 11 §11.6, pp. 267–270' },
  },
  'ch12/12.6/q-budget': {
    kind: 'numeric',
    id: 'ch12/12.6/q-budget',
    prompt: (
      <Translated vi={<>Gom tạm các khóa 4 byte cho một khối 256 luồng với hệ số gộp việc 8 cần bao nhiêu byte bộ nhớ chia sẻ?</>}>
        <>Staging 4-byte keys for a block of 256 threads with a coarsening factor of 8 needs how many bytes of shared memory?</>
      </Translated>
    ),
    answer: 8192,
    unit: 'bytes',
    explain: (
      <Translated
        vi={
          <p>
            256 × 8 × 4 = 8192. Các cặp (index, value) của lab 12.4 sẽ làm nó gấp đôi. So sánh nó với giới hạn mỗi khối của thiết
            bị bạn từ <code>python3 labs/pmpp.py doctor</code>; mã host Metal của lab 12.3 kiểm tra giới hạn của thiết bị và bỏ qua
            hệ số gộp việc nào không vừa.
          </p>
        }
      >
        <p>
          256 × 8 × 4 = 8192. Lab 12.4's (index, value) pairs would double it. Compare it with your device's per-block limit
          from <code>python3 labs/pmpp.py doctor</code>; lab 12.3's Metal host code checks the device's limit and skips a
          coarsening factor that does not fit.
        </p>
      </Translated>
    ),
    source: 'ch. 12 §12.6, p. 299; notes/today/ch05-facts.md C1, C8',
    vi: { unit: 'byte', source: 'chương 12 §12.6, p. 299; notes/today/ch05-facts.md C1, C8' },
  },
  'ch12/12.6/q-cub': {
    kind: 'choice',
    id: 'ch12/12.6/q-cub',
    prompt: (
      <Translated
        vi={
          <>
            Kernel select của CUB chỉ gom tạm qua bộ nhớ chia sẻ khi một tile giữ nhiều phần tử hơn số luồng nó có, và rải
            trực tiếp trong trường hợp còn lại. Vì sao đó là một quy tắc hợp lý?
          </>
        }
      >
        CUB's select kernel stages through shared memory only when a tile keeps more items than it has threads, and
        scatters directly otherwise. Why is that a sensible rule?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Gom tạm tốn bộ nhớ chia sẻ và một rào chắn, và chỉ có lợi nhờ gộp các lần ghi rải rác, mà khi mỗi luồng giữ một khóa hoặc ít hơn thì chúng chỉ chiếm phần nhỏ lưu lượng.">
            Staging costs shared memory and a barrier and pays only by merging scattered stores, which at one kept key per thread or fewer are a small share of the traffic.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Gom tạm sai ở tỉ lệ thấp: các luồng vượt quá số đếm của tile sẽ chép những mục cũ của mảng gom tạm.">Staging is incorrect at low fractions: threads past the tile's count would copy stale entries of the staging array.</Translated>,
        why: <Translated vi="Nó đúng ở mọi tỉ lệ; câu hỏi là nó có lợi hay không.">It is correct at any fraction; the question is whether it pays.</Translated>,
      },
      {
        label: <Translated vi="Rải trực tiếp thì ổn định còn gom tạm thì không: chép ra theo chỉ số luồng làm đảo thứ tự các khóa được giữ.">The direct scatter is stable and staging is not: copying out by thread index reorders the kept keys.</Translated>,
        why: <Translated vi="Cả hai đều ghi mỗi khóa được giữ tại độ lệch đã scan của nó; cả hai đều ổn định.">Both write every kept key at its scanned offset; both are stable.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong trình khám phá, gộp việc mà không gom tạm thì các lần ghi trở nên rải rác, tệ nhất khi phần lớn khóa được giữ. Khi
            chỉ giữ vài phần trăm, cùng những lần ghi đó chỉ là phần nhỏ trong lưu lượng của kernel, vì mỗi khóa vẫn được đọc một
            lần. Quy tắc là của CUB (ch12-facts A9); lập luận là của khóa học này.
          </p>
        }
      >
        <p>
          In the explorer, coarsen without staging and the stores become scattered, worst when most keys are kept. At a
          few percent kept, the same stores are a small share of the kernel's traffic, because every key is still read once.
          The rule is CUB's (ch12-facts A9); the reasoning is ours.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch12-facts.md A9; our analysis with src/sim/compactstore.ts',
    vi: { source: 'notes/today/ch12-facts.md A9; phân tích của khóa học này với src/sim/compactstore.ts' },
  },
};

/** A lab's reference-solution status, in the labs page's words (LabKit's solutionTabStatus), from this machine's runs. */
function useLabStatus(lab: string) {
  const { reports } = useReports();
  return {
    cu: solutionTabStatus(reports, lab, 'kernel.cu'),
    metal: solutionTabStatus(reports, lab, 'kernel.metal'),
    mm: solutionTabStatus(reports, lab, 'launch.mm'),
  };
}

export function S126() {
  const status = useLabStatus('ch12/lab3_stable');
  const cu = useLabFile('ch12/lab3_stable/solution/kernel.cu');
  const metal = useLabFile('ch12/lab3_stable/solution/kernel.metal');
  const mm = useLabFile('ch12/lab3_stable/solution/launch.mm');
  const pick = (src: string | null, names: string[]) => {
    if (!src) return null;
    const parts = names.map((nm) => functionSource(src, nm)?.text).filter((x): x is string => !!x);
    return parts.length ? parts.join('\n\n') : src;
  };
  const cuPart = pick(cu, ['countKeptKernel', 'stagedScatterKernel', 'launchStableFilterStaged']);
  const metalPart = pick(metal, ['count_kept', 'staged_scatter']);
  const mmPart = pick(mm, ['encodeStableFilterStaged']);
  return (
    <>
      <Translated
        vi={
          <p>
            Tiếp theo sách chuyển sang các lần ghi của Fig. 12.8. Chỉ những luồng có khóa được giữ mới ghi, nên các ô đầu ra nằm
            cạnh nhau có thể do các warp khác nhau ghi, với những lane rảnh nằm xen giữa, và phần cứng có ít cơ hội gộp chúng hơn.
          </p>
        }
      >
        <p>
          The book next turns to the stores of Fig. 12.8. Only threads whose key is kept write, so output slots that
          sit next to each other can be written by different warps, with idle lanes in between, and the hardware gets fewer
          chances to combine them.
        </p>
      </Translated>
      <Source section="12.6" pages="298" />
      <Translated
        vi={
          <p>
            Hãy đọc kỹ điều đó, vì rất dễ hiểu quá đà. Bên trong một warp, các lane được giữ thực sự ghi vào các ô liên tiếp: độ
            lệch của mỗi lane lớn hơn một so với lane được giữ trước đó. Thứ bị mất là giữa các warp. Mỗi warp giữ bất kỳ thứ gì
            đều phát một lần ghi ngắn, trống một phần, của riêng nó, và một dải ngắn vắt qua ranh giới đoạn sẽ chạm hai đoạn.
          </p>
        }
      >
        <p>
          Read that carefully, because it is easy to take it too far. Inside one warp, the kept lanes do write consecutive
          slots: each one's offset is one more than the previous kept lane's. What is lost is between warps. Every warp that
          keeps anything issues its own short, partly empty store, and a short run that crosses a segment boundary touches two
          segments.
        </p>
      </Translated>
      <Ask q={questionsS126['ch12/12.6/q-direct']} />

      <h2><Translated vi="Gom trước, rồi ghi cùng nhau">Gather first, then write together</Translated></h2>
      <Translated
        vi={
          <p>
            Cách sửa: mỗi khối trước hết đặt các khóa được giữ của mình vào bộ nhớ chia sẻ (shared memory), tại độ lệch của chúng
            trong khối, theo thứ tự đầu vào. Sau một rào chắn (barrier), các luồng 0, 1, 2 trở đi chép dải đó ra bộ nhớ toàn cục,
            nên các lane liên tiếp ghi vào các ô liên tiếp và mỗi khối ghi một đoạn liền. Sách gọi đây là một dạng riêng tư hóa
            (privatization): danh sách trong bộ nhớ chia sẻ là phần đầu ra riêng của khối, như ở mục 12.4, nhưng giờ theo thứ tự.
          </p>
        }
      >
        <p>
          The fix: each block first places its kept keys in shared memory, at their offsets within the block, in input order.
          After a barrier, threads 0, 1, 2 and onward copy that run to global memory, so consecutive lanes write consecutive
          slots and each block writes one contiguous chunk. The book calls this a form of privatisation: the shared-memory list
          is the block's private piece of the output, as in section 12.4, but now in order.
        </p>
      </Translated>
      <Source section="12.6" pages="298–299" figs="Fig. 12.9" />
      <Translated
        vi={
          <p>
            Hoạt cảnh ghi cùng những khóa được giữ theo ba cách: phép rải trực tiếp của Fig. 12.8, bản gom tạm, và bản gom tạm
            được gộp việc gấp 2.
          </p>
        }
      >
        <p>
          The scene writes the same kept keys three ways: the direct scatter of Fig. 12.8, the staged version, and the
          staged version coarsened by 2.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Hoạt cảnh thu nhỏ mọi thứ cho vừa: mười sáu khóa của sách, khối tám luồng, warp bốn lane và đoạn 16 byte. Nó đếm các
              lần ghi và các đoạn; không có gì được đo thời gian.
            </p>
          }
        >
          <p>
            The scene shrinks everything to fit: the book's sixteen keys, blocks of eight threads, warps of four lanes
            and 16-byte segments. It counts stores and segments; nothing is timed.
          </p>
        </Translated>
      </Simplification>
      <StageWriteScene />
      <Predict
        id="ch12/12.6/p-stores"
        prompt={
          <Translated
            vi={
              <>
                Trình khám phá bên dưới mở đầu với {PRESET.n} khóa, một nửa được giữ ngẫu nhiên, khối {PRESET.blockDim} luồng,
                warp {PRESET.warp} lane và đoạn {PRESET.segmentBytes} byte. Phép rải trực tiếp phát {P_DIRECT.stores} lệnh
                ghi: warp nào cũng giữ gì đó. Bản gom tạm phát bao nhiêu lệnh?
              </>
            }
          >
            The explorer below opens with {PRESET.n} keys, half kept at random, blocks of {PRESET.blockDim} threads, warps of{' '}
            {PRESET.warp} and {PRESET.segmentBytes}-byte segments. The direct scatter issues {P_DIRECT.stores} store
            instructions: every warp keeps something. How many does the staged version issue?
          </Translated>
        }
        unit={tr('store instructions', 'lệnh ghi')}
      >
        {(v) => (
          <Translated
            vi={
              <p>
                Mô hình cho ra <b>{P_STAGED.stores}</b>{v === P_STAGED.stores ? ', đúng như bạn nói' : ''}. Dải khoảng{' '}
                {Math.round(P_STAGED.meanChunk)} khóa được giữ của mỗi khối được các lane liên tiếp chép ra, nên nó cần bốn hoặc năm
                lần ghi rộng cả warp thay vì tám. Số đoạn bị chạm hầu như không đổi, từ {P_DIRECT.segments} thành {P_STAGED.segments}:
                khi mỗi luồng một khóa, lần ghi của mỗi warp vốn đã là một dải liền.
              </p>
            }
          >
            <p>
              The model says <b>{P_STAGED.stores}</b>{v === P_STAGED.stores ? ', as you said' : ''}. Each block's run of about{' '}
              {Math.round(P_STAGED.meanChunk)} kept keys is copied by consecutive lanes, so it takes four or five warp-wide stores
              instead of eight. The segments touched barely move, from {P_DIRECT.segments} to {P_STAGED.segments}: at one key per
              thread, each warp's store was already one run.
            </p>
          </Translated>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Trình khám phá đếm mọi lệnh ghi của cả bốn thiết kế cùng lúc, cho bất kỳ khóa và kích thước nào bạn đặt. Dải ô cho thấy
            ai ghi dải của khối đang chọn.
          </p>
        }
      >
        <p>
          The explorer counts every store instruction of all four designs at once, for whatever keys and sizes you set. The
          strip shows who writes the selected block's run.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi lần một lệnh ghi, không có cache: một đoạn bị hai lệnh chạm được đếm hai lần. Không đo thời gian, và không tính chi
              phí cho chuyến đi qua bộ nhớ chia sẻ và rào chắn của việc gom tạm. Các lần đọc không được vẽ: thiết kế nào cũng đọc mỗi
              khóa một lần, được hợp nhất.
            </p>
          }
        >
          <p>
            One store instruction at a time, with no cache: a segment two instructions touch is counted twice. No timing, and
            no charge for staging's trip through shared memory and its barrier. Reads are not drawn: every design reads each key
            once, coalesced.
          </p>
        </Translated>
      </Simplification>
      <Explorer />
      <Translated
        vi={
          <p>
            Ba điều cần tìm trong đó. Thứ nhất, với mỗi luồng một khóa, gom tạm giúp nhiều nhất ở tỉ lệ thấp và trung bình, và
            không giúp gì khi gần như mọi khóa được giữ: lúc đó các dải của phép rải trực tiếp vốn đã là những warp đầy. Thứ hai,
            gộp việc theo hệ số <i>c</i> chia số đoạn liền cho <i>c</i>. Thứ ba, gộp việc mà không gom tạm (thiết kế thứ tư, của
            khóa học này) là trường hợp xấu: mỗi luồng giữ <i>c</i> khóa liên tiếp, nên ở mỗi vòng các ô của các lane nằm cách xa
            nhau, và khi phần lớn khóa được giữ, một lần ghi có thể chạm một đoạn riêng cho mỗi lane.
          </p>
        }
      >
        <p>
          Three things to find in it. First, with one key per thread, staging helps most at low and middling fractions and not
          at all when nearly every key is kept: the direct scatter's runs are then already full warps. Second, coarsening by{' '}
          <i>c</i> divides the number of chunks by <i>c</i>. Third, coarsening without staging (the fourth design, ours) is the
          bad case: each thread holds <i>c</i> consecutive keys, so in each round the lanes' slots lie apart, and with most keys
          kept a store can touch a separate segment for every lane.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              "Được hợp nhất" nghĩa là gì thì khác nhau tùy nhà cung cấp: NVIDIA ghi sector 32 byte, AMD dòng cache 64 và 128 byte,
              Apple không ghi đơn vị nào (ch06-facts A2, B1, C1). Vì lý do đó, kích thước đoạn trong trình khám phá là một tham số.
            </p>
          }
        >
          <p>
            What "coalesced" means differs by vendor: NVIDIA documents 32-byte sectors, AMD 64- and 128-byte cache lines, Apple
            no unit at all (ch06-facts A2, B1, C1). The segment size in the explorer is a parameter for that reason.
          </p>
        </Translated>
      </Aside>

      <h2><Translated vi="Gộp việc">Coarsening</Translated></h2>
      <Translated
        vi={
          <p>
            Cho mỗi khối nhiều khóa hơn số luồng nó có làm các đoạn liền ít đi và dài hơn. Sách coi các đoạn nhỏ là cái giá của
            tính song song hạt mịn hơn: đầu ra được chia cho càng nhiều khối, phần của mỗi khối càng ngắn.
          </p>
        }
      >
        <p>
          Giving each block more keys than it has threads makes the chunks fewer and longer. The book frames the small chunks as
          the price of finer-grained parallelism: the more blocks the output is split across, the shorter each one's piece.
        </p>
      </Translated>
      <Source section="12.6" pages="299" figs="Fig. 12.10" />
      <Ask q={questionsS126['ch12/12.6/q-chunks']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Lý do lớn hơn của sách là phép scan. Scan là phần tốn kém của một bộ lọc ổn định, và chương 11 đã cho thấy một phép
            scan được gộp việc làm ít việc hơn và đồng bộ hóa ít hơn: mỗi luồng đếm các khóa của mình bằng một vòng lặp thường, và
            chỉ một con số mỗi luồng, hoặc mỗi warp, đi vào phép scan cấp khối.
          </p>
        }
      >
        <p>
          The book's larger reason is the scan. The scan is the expensive part of a stable filter, and chapter 11 showed that a
          coarsened scan does less work and synchronises less: each thread counts its own keys in a plain loop, and only one
          number per thread, or one per warp, enters the block-level scan.
        </p>
      </Translated>
      <Source section="12.6" pages="299–300" />
      <Ask q={questionsS126['ch12/12.6/q-scan-coarse']} />

      <h2><Translated vi="Mã">The code</Translated></h2>
      <Translated
        vi={
          <p>
            Phần B của lab 12.3 viết bài tập 3 và 4 ở dạng nhiều kernel: đếm các khóa được giữ của mỗi tile, scan các số đếm, rồi
            gom tạm và ghi. Trong một vòng, vị trí của mỗi lane đến từ một lần bỏ phiếu và một popcount, như trong trình chạy từng
            bước theo khối của mục 12.5. Tile đi qua <code>coarse</code> vòng, mỗi vòng <code>BLOCK_DIM</code> khóa liên tiếp, nên
            các lần nạp vẫn được hợp nhất, và mảng gom tạm là bộ nhớ chia sẻ động <code>BLOCK_DIM × coarse × 4</code> byte.
          </p>
        }
      >
        <p>
          Lab 12.3 part B writes exercises 3 and 4 in a multi-kernel form: count the kept keys of each tile, scan the counts,
          then stage and write. Inside a round, each lane's place comes from a vote and a popcount, as in section 12.5's block
          stepper. The tile walks <code>coarse</code> rounds of <code>BLOCK_DIM</code> consecutive keys, so the loads stay
          coalesced, and the staging array is dynamic shared memory of <code>BLOCK_DIM × coarse × 4</code> bytes.
        </p>
      </Translated>
      {cuPart && metalPart && mmPart ? (
        <Code
          tabs={[
            { label: 'CUDA', code: cuPart, tiers: TIERS, status: status.cu },
            { label: tr('Metal kernel', 'Kernel Metal'), code: metalPart, tiers: TIERS, status: status.metal },
            { label: tr('Metal host', 'Host Metal'), code: mmPart, tiers: {}, status: status.mm },
          ]}
          caption={
            <Translated
              vi={
                <>
                  Phần B của lab 12.3, từ <code>labs/ch12/lab3_stable/solution/</code>. Trên Metal, mảng gom tạm là một tham
                  số con trỏ <code>threadgroup</code> mà host đặt kích thước trên encoder, sau khi kiểm tra{' '}
                  <code>maxThreadgroupMemoryLength</code> của thiết bị.
                </>
              }
            >
              Lab 12.3 part B, from <code>labs/ch12/lab3_stable/solution/</code>. On Metal the staging array is a{' '}
              <code>threadgroup</code> pointer parameter whose size the host sets on the encoder, after checking the
              device's <code>maxThreadgroupMemoryLength</code>.
            </Translated>
          }
        />
      ) : null}
      <Ask q={questionsS126['ch12/12.6/q-budget']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Bảng đặt lưu lượng bộ nhớ toàn cục của ba thiết kế cạnh nhau: ba kernel của bài tập 1, cách đếm-rồi-rải của phần B lab
            12.3, và bản một lượt của bài tập 2. Kéo tỉ lệ được giữ và xem phần nào của mỗi tổng thay đổi.
          </p>
        }
      >
        <p>
          The table sets the three designs' global traffic side by side: exercise 1's three kernels, lab 12.3 part B's
          count-then-scatter, and exercise 2's single pass. Slide the fraction kept and see which part of each total moves.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Byte, không phải thời gian: mỗi mảng được đếm một lần cho mỗi kernel đọc hoặc ghi nó. Không được đếm: lưu lượng của
              scan ngoài việc đọc đầu vào và ghi đầu ra, cache, và các lần đọc look-back của bản một lượt, vài word mỗi khối.
            </p>
          }
        >
          <p>
            Bytes, not time: each array is counted once for every kernel that reads or writes it. Not counted: the scan's
            traffic beyond reading its input and writing its output, caches, and the single pass's look-back reads, a
            few words per block.
          </p>
        </Translated>
      </Simplification>
      <BytesTable />

      <h2><Translated vi="Khi nào không nên gom tạm">When not to stage</Translated></h2>
      <Translated
        vi={
          <p>
            Các thư viện production chủ ý đánh đổi điều này. Kernel select của CUB chính là Figs. 12.9 và 12.10 trong mã production,
            và nó không phải lúc nào cũng gom tạm: nó chỉ đi qua bộ nhớ chia sẻ "if the average number of selection_flags items per
            thread is greater than one" (nếu số phần tử selection_flags trung bình mỗi luồng lớn hơn một), và rải trực tiếp trong
            trường hợp còn lại.
          </p>
        }
      >
        <p>
          Production libraries make this trade on purpose. CUB's select kernel is Figs. 12.9 and 12.10 in production code, and
          it does not always stage: it goes through shared memory only "if the average number of selection_flags items per
          thread is greater than one", and scatters directly otherwise.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch12-facts.md, A9)', '2026-11-20 (notes/today/ch12-facts.md, mục A9)')}>
        <Translated vi={<p>Từ <code>agent_select_if.cuh</code> của CUB trong mã nguồn CCCL.</p>}>
          <p>From CUB's <code>agent_select_if.cuh</code> in the CCCL source.</p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Quy tắc của CUB xuất phát từ chi phí của việc gom tạm. Chuyến đi qua bộ nhớ chia sẻ và một rào chắn phải trả trong mọi
            trường hợp; phần tiết kiệm nằm ở việc biến nhiều lần ghi rải rác thành vài lần ghi đầy. Khi phần lớn khóa bị bỏ, các
            lần ghi chỉ là phần nhỏ của một kernel vốn vẫn đọc mọi khóa, nên kể cả cắt giảm mạnh số lệnh ghi cũng tiết kiệm được ít
            trên tổng. Trong trình khám phá, hãy so sánh cột cuối của các thiết kế gộp việc khi giữ vài phần trăm và khi giữ một nửa.
          </p>
        }
      >
        <p>
          CUB's rule follows from what staging costs. A trip through shared memory and a barrier is paid in every case; the
          saving is in turning many scattered stores into a few full ones. When most keys are dropped, the stores are a small
          part of a kernel that reads every key anyway, so even a large cut in store instructions saves little of the total.
          In the explorer, compare the last column of the coarsened designs at a few percent kept and at half.
        </p>
      </Translated>
      <Ask q={questionsS126['ch12/12.6/q-cub']} />

      <h2><Translated vi="Các bài tập được giải ở đâu">Where the exercises are worked</Translated></h2>
      <Translated
        vi={
          <p>
            Bài tập 3 và 4 (riêng tư hóa, rồi gộp việc, cho bộ lọc một kernel) được giải trong phần B của lab 12.3, ở dạng nhiều
            kernel phía trên; lab 12.5 tùy chọn dùng dạng một kernel. Cả hai đều ở{' '}
            <a href={href('ch12', 'labs')}>trang lab</a>.
          </p>
        }
      >
        <p>
          Exercises 3 and 4 (privatisation, then coarsening, of the single-kernel filter) are worked in lab 12.3 part B, in the
          multi-kernel form above; the optional lab 12.5 takes the single-kernel form. Both are on the{' '}
          <a href={href('ch12', 'labs')}>labs page</a>.
        </p>
      </Translated>
      <Source section="12.10" pages="302" />
    </>
  );
}
