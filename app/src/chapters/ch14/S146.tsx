import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, realBackends, useReports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { LAB2, measuredLocal, measuredOneBit } from '../../sim/radixcost';
import {
  BOOK_KEYS,
  BOOK_KEY_BITS,
  UNITS,
  bin,
  directPass,
  exclusiveScan,
  bucketTable,
  localPass,
  localSortBlocks,
  passCount,
  passInput,
  placeFromCounts,
  softmaxKeys,
  starts,
  summarise,
  uniformKeys,
  unitsOf,
  warpStores,
  type Design,
  type Layout,
  type WarpStore,
} from '../../sim/radixlocal';
import { LocalSortScene } from './LocalSortScene';
import { useSolutionStatus } from './solutionStatus';

// ---------------------------------------------------------------- shared bits
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const plural = (n: number, one: string, many = one + 's') => `${n} ${n === 1 ? one : many}`;
const dec = (v: number) => (Math.round(v * 10) / 10).toFixed(1);

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

// ---------------------------------------------------------------- interactive 1: where a warp's writes land
type KeySet = 'book' | 'uniform' | 'softmax';
const KEYSET_LABEL: Record<KeySet, string> = {
  get book() { return tr("the book's sixteen", 'mười sáu khóa của sách'); },
  get uniform() { return tr('4,096 random keys', '4,096 khóa ngẫu nhiên'); },
  get softmax() { return tr('a 4,096-token softmax row', 'một hàng softmax 4,096 token'); },
};
const N_SAMPLE = 4096;
const KEYS: Record<KeySet, readonly number[]> = {
  book: BOOK_KEYS,
  uniform: uniformKeys(N_SAMPLE, 1),
  softmax: softmaxKeys(N_SAMPLE, 1),
};
const ELEM = 4; // every key is stored as a 4-byte unsigned int
const METAL_SIZES = [16, 32, 64, 128];

function BookStrip({ addrs, bounds, out, source, local, lanesFrom }: {
  addrs: number[]; bounds: number[]; out: number[]; source: number[]; local: boolean; lanesFrom: number[];
}) {
  const cw = 680 / 16;
  const hit = new Set(addrs);
  const from = new Set(lanesFrom);
  const tier = local ? 'shared' : 'global';
  return (
    <>
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox="0 0 680 140" style={{ width: '100%', minWidth: 560, maxWidth: 820, display: 'block' }} role="img"
          aria-label={tr("This block's keys and where they land in the output", 'Các khóa của khối này và nơi chúng rơi vào trong đầu ra')} data-strip="book">
          <defs>
            <marker id="s146-head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">{/* i18n-ignore: SVG attribute value */}
              <path d="M0,0 L8,4 L0,8 z" fill="var(--ink)" />
            </marker>
          </defs>
          {lanesFrom.map((src, l) => (
            <text key={l} x={src * cw + cw / 2} y={12} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>t{l}</text>
          ))}
          {source.map((k, i) => (
            <g key={i} data-src={i}>
              <rect x={i * cw + 1} y={18} width={cw - 2} height={24} rx="2" fill={from.has(i) ? `var(--${tier})` : `var(--${tier}-wash)`} stroke={`var(--${tier})`} />
              <text x={i * cw + cw / 2} y={34} fontSize="11" textAnchor="middle" style={{ fill: from.has(i) ? 'var(--page)' : 'var(--ink)' }}>{bin(k, 4)}</text>
            </g>
          ))}
          {[1, 2, 3].map((b) => <line key={b} x1={b * 4 * cw} x2={b * 4 * cw} y1={14} y2={46} stroke="var(--ink-3)" strokeDasharray="3 3" />)}
          {lanesFrom.map((src, l) => (
            <line key={l} x1={src * cw + cw / 2} y1={43} x2={addrs[l] * cw + cw / 2} y2={100} stroke="var(--ink)" strokeWidth="1.6" markerEnd="url(#s146-head)" />
          ))}
          {out.map((k, i) => (
            <g key={i} data-out={i} data-hit={hit.has(i) ? 1 : 0}>
              <rect x={i * cw + 1} y={104} width={cw - 2} height={24} rx="2" fill={hit.has(i) ? 'var(--global)' : 'var(--global-wash)'} stroke="var(--global)" />
              <text x={i * cw + cw / 2} y={120} fontSize="11" textAnchor="middle" style={{ fill: hit.has(i) ? 'var(--page)' : 'var(--ink)' }}>{bin(k, 4)}</text>
            </g>
          ))}
          {bounds.slice(1).map((b, i) => (b > 0 && b < 16 ? <line key={i} x1={b * cw} x2={b * cw} y1={100} y2={134} stroke="var(--ink)" strokeWidth="2.4" /> : null))}
        </svg>
      </div>
      <p className="muted" style={{ margin: '0.3rem 0 0', fontSize: '0.84rem' }}>
        <Translated
          vi={
            <>
              Trên: {local ? 'khóa của mỗi khối sau khi sắp xếp cục bộ, trong bộ nhớ chia sẻ' : 'các khóa như lượt này đọc chúng, trong bộ nhớ toàn cục'}, đường
              nét đứt giữa các khối; các ô đậm hơn thuộc lần ghi này. Dưới: đầu ra trong bộ nhớ toàn cục; các đường dày ngăn cách các bucket.
            </>
          }
        >
          Top: {local ? "each block's keys after its local sort, in shared memory" : 'the keys as this pass reads them, in global memory'}, dashed lines
          between blocks; the darker cells are this store's. Bottom: the output in global memory; thick lines separate its buckets.
        </Translated>
      </p>
    </>
  );
}

function WideStrip({ n, addrs, bounds }: { n: number; addrs: number[]; bounds: number[] }) {
  const W = 680;
  const x = (a: number) => (a / n) * W;
  // A marker above the start of every run, so a run of one key is still visible at this scale.
  const sorted = [...addrs].sort((a, b) => a - b);
  const runStarts = sorted.filter((a, i) => i === 0 || a !== sorted[i - 1] + 1);
  return (
    <>
      <svg viewBox="0 0 680 44" style={{ width: '100%', display: 'block' }} role="img" aria-label={tr("Output of the pass with this warp's writes marked", 'Đầu ra của lượt với các lần ghi của warp này được đánh dấu')} data-strip="wide" data-runs={runStarts.length}>
        {runStarts.map((a) => (
          <path key={a} d={`M${x(a) - 4},2 h8 l-4,7 z`} fill="var(--ink)" data-run-start={a} />
        ))}
        <rect x="0" y="12" width={W} height="26" fill="var(--global-wash)" stroke="var(--global)" />
        {bounds.slice(1, -1).map((b, i) => <line key={i} x1={x(b)} x2={x(b)} y1={12} y2={38} stroke="var(--ink-3)" strokeWidth={bounds.length > 20 ? 0.4 : 1} />)}
        {addrs.map((a, i) => (
          <rect key={i} x={x(a)} y={8} width={Math.max(1.5, W / n)} height={34} fill="var(--global)" />
        ))}
      </svg>
      <p className="muted" style={{ margin: '0.2rem 0 0', fontSize: '0.84rem' }}>
        <Translated
          vi={
            <>
              Toàn bộ đầu ra, {n.toLocaleString('en-US')} khóa, trong bộ nhớ toàn cục. Đậm: những gì lần ghi này ghi, với một tam giác trên mỗi
              đoạn. Đường mảnh: ranh giới bucket.
            </>
          }
        >
          The whole output, {n.toLocaleString('en-US')} keys, in global memory. Dark: what this store writes, with a triangle over each
          run. Thin lines: bucket boundaries.
        </Translated>
      </p>
    </>
  );
}

function Units({ w, unitBytes, noun }: { w: WarpStore; unitBytes: number; noun: string }) {
  const per = Math.max(1, unitBytes / ELEM);
  const written = new Set(w.addrs);
  const shown = w.units.slice(0, 48);
  return (
    <div data-units={w.units.length} style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 4px', alignItems: 'center' }} aria-label={tr(`${noun}s touched`, `các ${({ sector: 'sector', line: 'line', segment: 'phân đoạn' } as Record<string, string>)[noun] ?? noun} bị chạm`)}>
      {shown.map((u, k) => (
        <span key={u} style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
          {k > 0 && u !== shown[k - 1] + 1 ? <span className="muted" style={{ fontSize: '0.8rem' }}>…</span> : null}
          <span data-unit={u} title={tr(`${noun} ${u}: bytes ${u * unitBytes} to ${(u + 1) * unitBytes - 1}`, `${({ sector: 'sector', line: 'line', segment: 'phân đoạn' } as Record<string, string>)[noun] ?? noun} ${u}: byte ${u * unitBytes} đến ${(u + 1) * unitBytes - 1}`)}
            style={{ display: 'inline-flex', border: '1px solid var(--global)', background: 'var(--cell)' }}>
            {Array.from({ length: per }, (_, e) => (
              <i key={e} style={{ width: per > 16 ? 4 : 6, height: 14, background: written.has(u * per + e) ? 'var(--global)' : 'transparent', borderLeft: e ? '1px solid var(--rule-soft)' : 'none' }} />
            ))}
          </span>
        </span>
      ))}
      {w.units.length > shown.length ? <span className="muted"><Translated vi={<>và {w.units.length - shown.length} cái nữa</>}>and {w.units.length - shown.length} more</Translated></span> : null}
    </div>
  );
}

function WarpWrites() {
  const [ks, setKs] = useState<KeySet>('uniform');
  const [B, setB] = useState(256);
  const [W, setW] = useState(32);
  const [r, setR] = useState(1);
  const [pass, setPass] = useState(0);
  const [unitId, setUnitId] = useState('nv-sector');
  const [metal, setMetal] = useState(32);
  const [design, setDesign] = useState<Design>('direct');
  const [sel, setSel] = useState(0);

  const book = ks === 'book';
  const bits = book ? BOOK_KEY_BITS : 32;
  const P = passCount(bits, r);
  const p = Math.min(pass, P - 1);
  const bitsLabel = (sh: number) => (r === 1 ? tr(`bit ${sh}`, `bit ${sh}`) : tr(`bits ${sh} to ${sh + r - 1}`, `các bit ${sh} đến ${sh + r - 1}`));
  const block = book ? 4 : B;
  const unit = UNITS.find((u) => u.id === unitId)!;
  const unitBytes = unit.bytes ?? metal;

  const m = useMemo(() => {
    const input = passInput(KEYS[ks], r, p);
    const shift = p * r;
    const plans = { direct: directPass(input, shift, r, block), local: localPass(input, shift, r, block) };
    const stores = { direct: warpStores(plans.direct, W, ELEM, unitBytes), local: warpStores(plans.local, W, ELEM, unitBytes) };
    const counts = new Array<number>(2 ** r).fill(0);
    for (const k of input) counts[(k >>> shift) & (2 ** r - 1)]++;
    const bounds = [...exclusiveScan(counts), input.length];
    return { input, shift, plans, stores, sum: { direct: summarise(stores.direct), local: summarise(stores.local) }, bounds };
  }, [ks, r, p, block, W, unitBytes]);

  const list = m.stores[design];
  const s = Math.min(sel, list.length - 1);
  const w = list[s];
  const lanes = Math.min(W, block);
  const plan = m.plans[design];
  // Where each lane's key sits before the store: its input slot (direct) or its local slot (local).
  const lanesFrom = list[s].addrs.map((_, l) => w.block * block + w.store * block + w.warp * W + l);
  const firstThread = w.warp * W;
  const noun = unit.noun;
  const pick = (k: KeySet) => {
    setKs(k);
    setPass(0);
    setSel(0);
    if (k === 'book' && r > 4) setR(1);
  };

  return (
    <Bench
      title={tr("Where a warp's writes land", 'Các lần ghi của một warp rơi vào đâu')}
      task={tr(
        'Start with the direct scatter and raise r. Then switch the design to the local sort, and try a block of 4 threads against one of 256.',
        'Bắt đầu với phân tán trực tiếp và tăng r. Rồi chuyển thiết kế sang sắp xếp cục bộ, và thử một khối 4 luồng so với một khối 256 luồng.',
      )}
      foot={
        <Translated
          vi={
            <>
              Mô hình: src/sim/radixlocal.ts. Mỗi lúc một lần ghi rộng cả warp; khóa 4 byte; đầu ra bắt đầu trên ranh giới
              {' '}{({ sector: 'sector', line: 'line', segment: 'phân đoạn' } as Record<string, string>)[noun] ?? noun}. Mỗi lượt thấy các khóa như các lượt trước để lại{p ? ` (ở đây, sau các lượt 0 đến ${p - 1})` : ''}.
              Hàng softmax là tổng hợp (logits phân phối chuẩn, độ lệch chuẩn 2, của khóa học này), các khóa của nó được biến
              đổi cho phép sắp xếp giảm dần như lab 14.3 làm (ch14-facts E1).
            </>
          }
        >
          <>
            Model: src/sim/radixlocal.ts. One warp-wide store at a time; 4-byte keys; the output starts on a {noun} boundary.
            Each pass sees the keys as the earlier passes left them{p ? ` (here, after passes 0 to ${p - 1})` : ''}. The
            softmax row is synthetic (normal logits, standard deviation 2, ours), its keys transformed for a descending sort
            as lab 14.3 does (ch14-facts E1).
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <Seg label={tr('Keys', 'Khóa')} value={ks} options={['book', 'uniform', 'softmax'] as KeySet[]} show={(k) => KEYSET_LABEL[k]} onPick={pick} />
        <Seg label={book ? tr('Threads per block (the figure has 4)', 'Số luồng mỗi khối (hình có 4)') : tr('Threads per block', 'Số luồng mỗi khối')} value={block} options={[4, 128, 256]}
          disabled={(v) => book && v !== 4} onPick={(v) => { setB(v); setSel(0); }} />
        <Seg label={tr('Warp width', 'Độ rộng warp')} value={W} options={[32, 64]} show={(v) => (v === 64 ? tr('64 (AMD gfx9)', '64 (AMD gfx9)') : '32')} onPick={(v) => { setW(v); setSel(0); }} />
      </div>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <Seg label={tr('Digit width r', 'Độ rộng chữ số r')} value={r} options={[1, 2, 4, 8]} disabled={(v) => book && v > 4}
          onPick={(v) => { setR(v); setPass(0); setSel(0); }} />
        {P <= 4 ? (
          <Seg label={tr(`Pass (${bitsLabel(m.shift)})`, `Lượt (${bitsLabel(m.shift)})`)} value={p} options={Array.from({ length: P }, (_, i) => i)}
            onPick={(v) => { setPass(v); setSel(0); }} />
        ) : (
          <label className="field">
            <Translated vi={<>Lượt {p} trong các lượt 0 đến {P - 1} ({bitsLabel(m.shift)})</>}>
              Pass {p} of passes 0 to {P - 1} ({bitsLabel(m.shift)})
            </Translated>
            <input type="range" min={0} max={P - 1} value={p} aria-label={tr('Pass', 'Lượt')} onChange={(e) => { setPass(Number(e.target.value)); setSel(0); }} />
          </label>
        )}
        <Seg label={tr('Design', 'Thiết kế')} value={design} options={['direct', 'local'] as Design[]}
          show={(d) => (d === 'direct' ? tr('direct scatter', 'phân tán trực tiếp') : tr('local sort, then write', 'sắp xếp cục bộ, rồi ghi'))} onPick={setDesign} />
      </div>
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ maxWidth: '100%', minWidth: 0 }}>
          <Translated vi="Đơn vị bộ nhớ">Memory unit</Translated>
          <select value={unitId} aria-label={tr('Memory unit', 'Đơn vị bộ nhớ')} onChange={(e) => setUnitId(e.target.value)} style={{ maxWidth: '100%' }}>
            {UNITS.map((u) => (
              <option key={u.id} value={u.id}>
                {tr(
                  u.label,
                  ({
                    'nv-sector': 'NVIDIA: sector 32 byte',
                    'nv-line': 'NVIDIA: line 128 byte (cache line của sách)',
                    'amd-64': 'AMD: line 64 byte',
                    'amd-128': 'AMD: line 128 byte',
                    metal: 'Metal: không được tài liệu hóa, hãy chọn một kích thước',
                  } as Record<string, string>)[u.id] ?? u.label,
                )}
              </option>
            ))}
          </select>
        </label>
        {unit.bytes == null ? (
          <Seg label={tr('Your segment size (bytes)', 'Kích thước phân đoạn của bạn (byte)')} value={metal} options={METAL_SIZES} onPick={setMetal} />
        ) : null}
        <label className="field" style={{ flex: '1 1 14rem' }}>
          <Translated
            vi={<>Lần ghi {s + 1} / {list.length}: khối {w.block}, {book || block <= W ? `${lanes} luồng của nó` : `warp ${w.warp} (luồng ${firstThread} đến ${firstThread + w.addrs.length - 1})`}</>}
          >
            Store {s + 1} of {list.length}: block {w.block}, {book || block <= W ? `its ${lanes} threads` : `warp ${w.warp} (threads ${firstThread} to ${firstThread + w.addrs.length - 1})`}
          </Translated>
          <input type="range" min={0} max={list.length - 1} value={s} aria-label={tr('Which warp store', 'Lần ghi warp nào')} onChange={(e) => setSel(Number(e.target.value))} />
        </label>
      </div>
      {unit.bytes == null ? (
        <p className="ui" style={{ margin: '0 0 0.8rem' }} data-metal-note>
          <Translated
            vi={
              <>
                Apple không ghi nhận trong tài liệu quy tắc hợp nhất truy cập, kích thước line hay kích thước sector nào
                (ch06-facts C1). Các số đếm dưới đây dùng kích thước {metal} byte bạn đã chọn; chỉ phép đo mới cho biết một máy
                Mac làm gì.
              </>
            }
          >
            Apple documents no coalescing rule, line size or sector size (ch06-facts C1). The counts below use the {metal}-byte
            size you picked; only a measurement says what a Mac does.
          </Translated>
        </p>
      ) : null}

      {book ? (
        <BookStrip addrs={w.addrs} bounds={m.bounds} out={plan.out} lanesFrom={lanesFrom} local={design === 'local'}
          source={design === 'local' ? plan.blocks.flatMap((b) => b.sorted) : m.input} />
      ) : (
        <WideStrip n={plan.n} addrs={w.addrs} bounds={m.bounds} />
      )}

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.9rem', alignItems: 'flex-start' }}>
        <div className="panel" style={{ flex: '1 1 18rem', minWidth: 0 }} aria-live="polite">
          <p style={{ margin: '0 0 0.5rem' }}>
            <b><Translated vi="Lần ghi này.">This store.</Translated></b>{' '}
            <span data-readout="store">
              <Translated
                vi={
                  <>
                    {w.addrs.length} lane ghi {w.runs} đoạn địa chỉ liên tiếp, chạm vào{' '}
                    <b data-readout="units">{w.units.length} {({ sector: 'sector', line: 'line', segment: 'phân đoạn' } as Record<string, string>)[noun] ?? noun}</b> {unitBytes} byte; một đoạn đã căn chỉnh của cùng các khóa đó
                    sẽ cần <span data-readout="min">{w.minUnits}</span>.
                  </>
                }
              >
                {plural(w.addrs.length, 'lane')} write{w.addrs.length === 1 ? 's' : ''} {plural(w.runs, 'run')} of consecutive addresses, touching{' '}
                <b data-readout="units">{plural(w.units.length, noun)}</b> of {unitBytes} bytes; one aligned run of the same keys would need{' '}
                <span data-readout="min">{w.minUnits}</span>.
              </Translated>
            </span>
          </p>
          <p className="muted" style={{ margin: '0 0 0.5rem', fontSize: '0.86rem' }} data-readout="digits">
            <Translated vi={<>Chữ số của các lane: {w.digits.join(' ')}</>}>Lane digits: {w.digits.join(' ')}</Translated>
          </p>
          <Units w={w} unitBytes={unitBytes} noun={noun} />
        </div>
        <div className="panel" style={{ flex: '1 1 18rem', minWidth: 0, overflowX: 'auto' }}>
          <p style={{ margin: '0 0 0.4rem' }}>
            <Translated vi={<><b>Mọi lần ghi của lượt này</b>, tính trung bình</>}>
              <b>Every store of this pass</b>, on average
            </Translated>
          </p>
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr>
                <th />
                <th className="num"><Translated vi="phân tán trực tiếp">direct scatter</Translated></th>
                <th className="num"><Translated vi="sắp xếp cục bộ">local sort</Translated></th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><Translated vi="số lần ghi warp">warp stores</Translated></td>
                <td className="num">{m.sum.direct.stores}</td>
                <td className="num">{m.sum.local.stores}</td>
              </tr>
              <tr>
                <td><Translated vi="số đoạn mỗi lần ghi">runs per store</Translated></td>
                <td className="num" data-avg="direct-runs">{dec(m.sum.direct.meanRuns)}</td>
                <td className="num" data-avg="local-runs">{dec(m.sum.local.meanRuns)}</td>
              </tr>
              <tr>
                <td><Translated vi={<>số {({ sector: 'sector', line: 'line', segment: 'phân đoạn' } as Record<string, string>)[noun] ?? noun} mỗi lần ghi</>}>{noun}s per store</Translated></td>
                <td className="num" data-avg="direct-units">{design === 'direct' ? <b>{dec(m.sum.direct.meanUnits)}</b> : dec(m.sum.direct.meanUnits)}</td>
                <td className="num" data-avg="local-units">{design === 'local' ? <b>{dec(m.sum.local.meanUnits)}</b> : dec(m.sum.local.meanUnits)}</td>
              </tr>
              <tr>
                <td><Translated vi="tối thiểu khi hợp nhất">coalesced minimum</Translated></td>
                <td className="num">{dec(m.sum.direct.meanMin)}</td>
                <td className="num">{dec(m.sum.local.meanMin)}</td>
              </tr>
              <tr>
                <td><Translated vi="lần ghi tệ nhất">worst store</Translated></td>
                <td className="num">{m.sum.direct.maxUnits}</td>
                <td className="num">{m.sum.local.maxUnits}</td>
              </tr>
            </tbody>
          </table>
          <p className="muted" style={{ margin: '0.5rem 0 0', fontSize: '0.86rem' }}>
            <Translated
              vi={
                <>
                  Ở đây phân tán trực tiếp tạo tối đa {Math.min(lanes, 2 ** r)} đoạn mỗi lần ghi: số nhỏ hơn giữa số lane ({lanes}) và
                  số bucket (2<sup>{r}</sup> = {2 ** r}).
                </>
              }
            >
              The direct scatter makes at most {Math.min(lanes, 2 ** r)} runs per store here: the smaller of the lanes ({lanes}) and the
              buckets (2<sup>{r}</sup> = {2 ** r}).
            </Translated>
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive 2: the bucket table
const BOOK_ZEROS = localSortBlocks(BOOK_KEYS, 0, 1, 4).map((b) => b.counts[0]);
const BOOK_LOCAL = localSortBlocks(BOOK_KEYS, 0, 1, 4);

function BucketTable() {
  const [zeros, setZeros] = useState<number[]>(BOOK_ZEROS);
  const [layout, setLayout] = useState<Layout>('digit-major');
  const [sel, setSel] = useState<{ b: number; t: number }>({ b: 2, t: 0 });
  const counts = zeros.map((z) => [z, 4 - z]);
  const isBook = zeros.every((z, i) => z === BOOK_ZEROS[i]);
  const table = bucketTable(counts, layout);
  const scanned = exclusiveScan(table);
  const st = starts(counts, layout);
  const placed = placeFromCounts(counts, layout);
  const byAddr = [...placed].sort((a, b) => a.addr - b.addr);
  const totalZeros = zeros.reduce((a, b) => a + b, 0);
  // A partitioned output holds zeros in its first totalZeros positions and ones after them.
  const misplaced = (i: number) => byAddr[i].digit !== (i < totalZeros ? 0 : 1);
  const firstBad = byAddr.findIndex((_, i) => misplaced(i));
  const ok = firstBad < 0;
  const labels = counts.flatMap((_, b) => [0, 1].map((d) => ({ b, d }))).sort((x, y) =>
    layout === 'digit-major' ? x.d - y.d || x.b - y.b : x.b - y.b || x.d - y.d,
  );
  const me = placed.find((x) => x.block === sel.b && x.slot === sel.t)!;
  const z = zeros[sel.b];
  const keyAt = (b: number, t: number) => (isBook ? bin(BOOK_LOCAL[b].sorted[t], 4) : null);
  const preset = (zs: number[], s: { b: number; t: number }) => {
    setZeros(zs);
    setLayout('digit-major');
    setSel(s);
  };
  const setZ = (b: number, v: number) => setZeros(zeros.map((x, i) => (i === b ? Math.max(0, Math.min(4, v)) : x)));

  return (
    <Bench
      title={tr('The bucket table', 'Bảng bucket')}
      task={tr(
        "Change a block's split between zeros and ones and watch every start move. Click a thread to see where it writes. Then scan the table in the wrong order.",
        'Thay đổi cách chia bit 0 và bit 1 của một khối và xem mọi điểm bắt đầu dịch chuyển. Nhấp vào một luồng để xem nó ghi vào đâu. Rồi scan bảng theo sai thứ tự.',
      )}
      foot={tr(
        'Model: bucketTable(), starts() and placeFromCounts() in src/sim/radixlocal.ts. Four blocks of four keys and a 1-bit digit, as in Fig. 14.9. Your choices here are not recorded.',
        'Mô hình: bucketTable(), starts() và placeFromCounts() trong src/sim/radixlocal.ts. Bốn khối, mỗi khối bốn khóa, và chữ số 1 bit, như trong Hình 14.9. Các lựa chọn của bạn ở đây không được ghi lại.',
      )}
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Mẫu có sẵn">Presets</Translated>
          <span className="seg" role="group" aria-label={tr('Presets', 'Mẫu có sẵn')}>
            <button onClick={() => preset(BOOK_ZEROS, { b: 0, t: 0 })}>{tr('Fig. 14.9', 'Hình 14.9')}</button>
            <button onClick={() => preset(BOOK_ZEROS, { b: 2, t: 1 })}>{tr('p. 342: block 2', 'p. 342: khối 2')}</button>
            <button onClick={() => preset(BOOK_ZEROS, { b: 3, t: 3 })}>{tr('p. 342: block 3', 'p. 342: khối 3')}</button>
          </span>
        </span>
        <Seg label={tr('Table layout before the scan', 'Bố cục bảng trước phép scan')} value={layout} options={['digit-major', 'block-major'] as Layout[]}
          show={(l) => (l === 'digit-major' ? tr('all zero counts, then all one counts', 'mọi số đếm bit 0, rồi mọi số đếm bit 1') : tr('block by block (wrong)', 'từng khối một (sai)'))} onPick={setLayout} />
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.7rem', marginBottom: '1rem' }}>
        {counts.map(([zc, oc], b) => (
          <div key={b} className="panel" style={{ flex: '1 1 9.5rem', minWidth: '9rem', padding: '0.6rem' }} data-block={b}>
            <p style={{ margin: '0 0 0.4rem' }}><b><Translated vi={<>Khối {b}</>}>Block {b}</Translated></b></p>
            <div style={{ display: 'flex', flexWrap: 'nowrap', alignItems: 'center', gap: '0.3rem', marginBottom: '0.45rem' }}>
              <button aria-label={tr(`fewer zeros in block ${b}`, `bớt bit 0 trong khối ${b}`)} onClick={() => setZ(b, zc - 1)} disabled={zc === 0} style={{ padding: '0.15rem 0.5rem' }}>−</button>
              <span className="ui" data-counts={`${zc} ${oc}`} style={{ flex: '1 1 auto', minWidth: 0, textAlign: 'center', fontSize: '0.8rem', lineHeight: 1.2 }}>
                <Translated vi={<>{zc} bit 0, {oc} bit 1</>}>{zc} zero{zc === 1 ? '' : 's'}, {oc} one{oc === 1 ? '' : 's'}</Translated>
              </span>
              <button aria-label={tr(`more zeros in block ${b}`, `thêm bit 0 trong khối ${b}`)} onClick={() => setZ(b, zc + 1)} disabled={zc === 4} style={{ padding: '0.15rem 0.5rem' }}>+</button>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 3 }} role="group" aria-label={tr(`threads of block ${b}`, `các luồng của khối ${b}`)}>
              {[0, 1, 2, 3].map((t) => {
                const d = t < zc ? 0 : 1;
                const on = sel.b === b && sel.t === t;
                return (
                  <button key={t} aria-pressed={on} aria-label={tr(`block ${b} thread ${t}`, `khối ${b} luồng ${t}`)} onClick={() => setSel({ b, t })}
                    style={{ padding: '0.2rem 0', display: 'flex', flexDirection: 'column', alignItems: 'center', lineHeight: 1.2,
                      background: on ? 'var(--shared-wash)' : undefined, borderColor: on ? 'var(--shared)' : undefined, color: on ? 'var(--ink)' : undefined }}>
                    <span className="muted" style={{ fontSize: '0.72rem' }}>t{t}</span>
                    <b style={{ fontSize: '0.86rem' }}>{keyAt(b, t) ?? d}</b>
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      <div className="panel" style={{ overflowX: 'auto', marginBottom: '0.9rem' }}>
        <table className="data" style={{ margin: 0, minWidth: '34rem' }} aria-label={tr('The table and its scan', 'Bảng và phép scan của nó')}>
          <thead>
            <tr>
              <th><Translated vi="phần tử bảng">table entry</Translated></th>
              {labels.map(({ b, d }) => <th key={`${b}-${d}`} className="num"><Translated vi={<>{d ? 'bit 1' : 'bit 0'} b{b}</>}>{d ? 'ones' : 'zeros'} b{b}</Translated></th>)}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><Translated vi="số đếm">count</Translated></td>
              {table.map((v, i) => <td key={i} className="num" data-table={i}>{v}</td>)}
            </tr>
            <tr>
              <td><Translated vi="scan loại trừ">exclusive scan</Translated></td>
              {scanned.map((v, i) => <td key={i} className="num" data-scan={i}><b>{v}</b></td>)}
            </tr>
          </tbody>
        </table>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', alignItems: 'flex-start' }}>
        <div className="panel" style={{ flex: '1 1 16rem' }} aria-live="polite">
          <p style={{ margin: 0 }} data-readout="thread">
            <b><Translated vi={<>Khối {sel.b}, luồng {sel.t}.</>}>Block {sel.b}, thread {sel.t}.</Translated></b>{' '}
            {me.digit === 0 ? (
              <>
                <Translated
                  vi={
                    <>
                      Ô cục bộ {sel.t} của nó nhỏ hơn số đếm bit 0 của khối ({z}), nên nó chứa một bit 0. Nó ghi vào điểm bắt đầu của bucket 0
                      cộng với ô của nó: {st[sel.b][0]} + {sel.t} = <b>{me.addr}</b>.
                    </>
                  }
                >
                  Its local slot {sel.t} is below the block's zero count ({z}), so it holds a zero. It writes to the zero bucket's
                  start plus its slot: {st[sel.b][0]} + {sel.t} = <b>{me.addr}</b>.
                </Translated>
              </>
            ) : (
              <>
                <Translated
                  vi={
                    <>
                      Ô cục bộ {sel.t} của nó không nhỏ hơn số đếm bit 0 của khối ({z}), nên nó chứa một bit 1, bit 1 thứ{' '}
                      {['nhất', 'hai', 'ba', 'tư'][sel.t - z]} của khối. Nó ghi vào điểm bắt đầu của bucket 1 cộng {sel.t} − {z}:{' '}
                      {st[sel.b][1]} + {sel.t - z} = <b>{me.addr}</b>.
                    </>
                  }
                >
                  Its local slot {sel.t} is not below the block's zero count ({z}), so it holds a one, the{' '}
                  {['first', 'second', 'third', 'fourth'][sel.t - z]} of the block's ones. It writes to the one bucket's start
                  plus {sel.t} − {z}: {st[sel.b][1]} + {sel.t - z} = <b>{me.addr}</b>.
                </Translated>
              </>
            )}
          </p>
          <table className="data" style={{ margin: '0.6rem 0 0' }}>
            <thead>
              <tr>
                <th><Translated vi="khối">block</Translated></th>
                {counts.map((_, b) => <th key={b} className="num">{b}</th>)}
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><Translated vi="bucket 0 bắt đầu tại">zero bucket starts at</Translated></td>
                {st.map((x, b) => <td key={b} className="num" data-start={`${b}-0`}>{x[0]}</td>)}
              </tr>
              <tr>
                <td><Translated vi="bucket 1 bắt đầu tại">one bucket starts at</Translated></td>
                {st.map((x, b) => <td key={b} className="num" data-start={`${b}-1`}>{x[1]}</td>)}
              </tr>
            </tbody>
          </table>
        </div>
        <div className="panel" style={{ flex: '1 1 20rem', minWidth: 0 }}>
          <p style={{ margin: '0 0 0.5rem' }}>
            <Translated vi={<><b>Đầu ra</b> (chữ số, và khối cùng luồng nào đã ghi nó)</>}>
              <b>The output</b> (digit, and which block and thread wrote it)
            </Translated>
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, minmax(0, 1fr))', gap: 3 }} data-output={byAddr.map((x) => x.digit).join('')}>
            {byAddr.map((x, i) => {
              const bad = misplaced(i);
              const mine = x.block === sel.b && x.slot === sel.t;
              return (
                <div key={i} data-addr={i} data-bad={bad ? 1 : 0}
                  style={{ border: `1px solid ${bad ? 'var(--fail)' : 'var(--global)'}`, background: bad ? 'var(--fail-wash)' : mine ? 'var(--global)' : 'var(--global-wash)',
                    color: mine && !bad ? 'var(--page)' : 'var(--ink)', borderRadius: 2, textAlign: 'center', fontSize: '0.74rem', lineHeight: 1.25, padding: '0.15rem 0' }}>
                  <span style={{ opacity: 0.75 }}>{i}</span>
                  <br />
                  <b>{x.digit}</b>
                  <br />
                  <span style={{ opacity: 0.8 }}>b{x.block}t{x.slot}</span>
                </div>
              );
            })}
          </div>
          <p style={{ margin: '0.6rem 0 0' }} data-verdict={ok ? 'partitioned' : 'not-partitioned'}>
            {ok ? (
              <span className="t-ok"><Translated vi="Đã phân hoạch: mọi bit 0 đứng trước mọi bit 1, và khóa của mỗi khối theo thứ tự khối.">Partitioned: every zero before every one, and each block's keys in block order.</Translated></span>
            ) : (
              <span className="t-fail">
                <Translated
                  vi={
                    <>
                      Chưa phân hoạch: vị trí {firstBad} chứa một bit 1, do khối {byAddr[firstBad].block} ghi, ở chỗ chỉ dành cho bit 0.
                      Khi scan từng khối một, các bit 1 của một khối bị đặt trước các bit 0 của khối kế tiếp.
                    </>
                  }
                >
                  Not partitioned: position {firstBad} holds a one, written by block {byAddr[firstBad].block}, where only zeros belong.
                  Scanned block by block, a block's ones are placed ahead of the next block's zeros.
                </Translated>
              </span>
            )}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- lab 14.2's measurement
function useLab2() {
  const { reports } = useReports();
  const devs = realBackends(reports).flatMap((b) => {
    const rows = benchRows(reports[b], LAB2);
    const local = measuredLocal(rows, 1, 1);
    const onebit = measuredOneBit(rows);
    return local != null && onebit != null ? [{ b, device: String(reports[b].device.name ?? b), local, onebit }] : [];
  });
  return { devs };
}

function LabListing() {
  const cu = useLabFile('ch14/lab2_multibit/solution/kernel.cu');
  const metal = useLabFile('ch14/lab2_multibit/solution/kernel.metal');
  const st = useSolutionStatus(LAB2);
  if (!cu || !metal) return null;
  return (
    <Code
      tabs={[
        { label: tr('CUDA and HIP', 'CUDA và HIP'), code: cu, status: st.cu },
        { label: 'Metal', code: metal, status: st.metal },
      ]}
      caption={
        <Translated
          vi={
            <>
              Lời giải tham chiếu (solution) của <a href={href('ch14', 'labs')}>lab 14.2</a>, nạp từ{' '}
              <code>labs/ch14/lab2_multibit/solution/</code>. Hãy thử trong lab trước: tự viết ba kernel và để dành phần này cho lúc
              bạn bị kẹt.
            </>
          }
        >
          <>
            The reference solution of <a href={href('ch14', 'labs')}>lab 14.2</a>, loaded from{' '}
            <code>labs/ch14/lab2_multibit/solution/</code>. Try it in the lab first: write the three kernels yourself and keep
            this for when you are stuck.
          </>
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- questions
const Q_COUNTS = [[3, 1], [1, 3], [2, 2], [2, 2]];
const Q_START = starts(Q_COUNTS, 'digit-major')[2][1];
const Q_THREAD = placeFromCounts(Q_COUNTS, 'digit-major').find((x) => x.block === 1 && x.slot === 2)!.addr;
const Q_SECTORS = unitsOf([...Array.from({ length: 12 }, (_, i) => 4 + i), ...Array.from({ length: 20 }, (_, i) => 64 + i)], 4, 32).length;
const P_R8 = summarise(warpStores(directPass(KEYS.uniform, 0, 8, 256), 32, 4, 32));
const P_R1 = summarise(warpStores(directPass(KEYS.uniform, 0, 1, 256), 32, 4, 32));

export const questionsS146: Record<string, Question> = {
  'ch14/14.6/q-start': {
    kind: 'numeric',
    id: 'ch14/14.6/q-start',
    prompt: (
      <Translated
        vi={
          <>
            Bốn khối kết thúc phép sắp xếp cục bộ với số đếm bit 0 là [3, 1, 2, 2] và số đếm bit 1 là [1, 3, 2, 2]. Bucket 1 của khối
            2 bắt đầu ở vị trí đầu ra nào?
          </>
        }
      >
        <>
          Four blocks finish their local sort with zero counts [3, 1, 2, 2] and one counts [1, 3, 2, 2]. At which output
          position does block 2's one bucket start?
        </>
      </Translated>
    ),
    answer: Q_START,
    explain: (
      <Translated
        vi={
          <p>
            Cả 8 bit 0 của cả bốn khối đứng trước. Sau chúng là các bit 1 của những khối trước khối 2: 1 của khối 0 và 3 của khối 1.
            Vậy 8 + 1 + 3 = {Q_START}. Trong bảng bố trí thành [3, 1, 2, 2, 1, 3, 2, 2], đó là giá trị scan loại trừ tại phần tử
            "bit 1 của khối 2".
          </p>
        }
      >
        <p>
          All 8 zeros of all four blocks come first. After them come the ones of the blocks before block 2: block 0's 1 and
          block 1's 3. So 8 + 1 + 3 = {Q_START}. In the table laid out as [3, 1, 2, 2, 1, 3, 2, 2], that is the exclusive
          scan at the entry "ones of block 2".
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.6, Fig. 14.9, pp. 340–341',
  },
  'ch14/14.6/q-thread': {
    kind: 'numeric',
    id: 'ch14/14.6/q-thread',
    prompt: (
      <Translated
        vi={
          <>
            Vẫn bốn khối đó. Khối 1 chứa 1 bit 0 và 3 bit 1, và bucket 1 của nó bắt đầu tại 9. Luồng 2 của khối 1 ghi khóa của nó
            vào đâu?
          </>
        }
      >
        <>
          The same four blocks. Block 1 holds 1 zero and 3 ones, and its one bucket starts at 9. Where does block 1's thread 2
          write its key?
        </>
      </Translated>
    ),
    answer: Q_THREAD,
    explain: (
      <Translated
        vi={
          <p>
            Ô cục bộ của luồng 2 là 2, không nhỏ hơn số đếm bit 0 của khối là 1, nên nó chứa một bit 1: bit 1 thứ hai của khối, ở độ
            lệch 2 − 1 = 1 bên trong bucket. 9 + 1 = {Q_THREAD}.
          </p>
        }
      >
        <p>
          Thread 2's local slot is 2, which is not below the block's zero count of 1, so it holds a one: the second of the
          block's ones, at offset 2 − 1 = 1 inside the bucket. 9 + 1 = {Q_THREAD}.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.6, pp. 341–342',
  },
  'ch14/14.6/q-sectors': {
    kind: 'numeric',
    id: 'ch14/14.6/q-sectors',
    prompt: (
      <Translated
        vi={
          <>
            Một warp 32 lane ghi các khóa 4 byte thành hai đoạn: 12 bit 0 bắt đầu từ byte 16 của một sector 32 byte, và 20 bit 1 bắt
            đầu trên một ranh giới sector ở chỗ khác. Lần ghi chạm vào bao nhiêu sector 32 byte?
          </>
        }
      >
        <>
          A warp of 32 lanes stores 4-byte keys in two runs: 12 zeros starting 16 bytes into a 32-byte sector, and 20 ones
          starting on a sector boundary somewhere else. How many 32-byte sectors does the store touch?
        </>
      </Translated>
    ),
    answer: Q_SECTORS,
    unit: 'sectors',
    vi: { unit: 'sector', source: 'ch14-facts A10; ch06-facts A2 (NVIDIA đếm truy cập của một warp theo sector 32 byte)' },
    explain: (
      <Translated
        vi={
          <p>
            Các bit 0 phủ 48 byte từ độ lệch 16, tức byte 16 đến 63: hai sector. Các bit 1 phủ 80 byte từ một ranh giới: ba sector,
            sector cuối dùng một nửa. Tổng cộng {Q_SECTORS}, trong khi một đoạn 32 khóa đã căn chỉnh cần 4.
          </p>
        }
      >
        <p>
          The zeros cover 48 bytes from offset 16, so bytes 16 to 63: two sectors. The ones cover 80 bytes from a boundary:
          three sectors, the last one half used. {Q_SECTORS} in all, where one aligned run of 32 keys needs 4.
        </p>
      </Translated>
    ),
    source: 'ch14-facts A10; ch06-facts A2 (NVIDIA counts a warp\'s access in 32-byte sectors)',
  },
  'ch14/14.6/q-layout': {
    kind: 'choice',
    id: 'ch14/14.6/q-layout',
    prompt: (
      <Translated vi={<>Vì sao bảng phải liệt kê số đếm bit 0 của mọi khối trước số đếm bit 1 của bất kỳ khối nào?</>}>
        <>Why must the table list every block's zero count before any block's one count?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Nó làm phép scan bảng được hợp nhất: phép scan đọc bảng nhanh hơn khi các chữ số bằng nhau nằm cạnh nhau.">It makes the table scan coalesced: a scan reads the table faster when equal digits sit side by side.</Translated>,
        why: <Translated vi="Bố cục nào cũng là một mảng liền mạch; phép scan đọc nó theo cùng cách. Thứ tự thay đổi những gì phép scan cộng lại, không phải cách nó đọc.">Either layout is one contiguous array; the scan reads it the same way. The order changes what the scan adds up, not how it reads.</Translated>,
      },
      {
        label: <Translated vi="Mọi bit 1 đi sau mọi bit 0 của mọi khối; scan theo thứ tự đó, mỗi phần tử cộng đúng các số đếm đứng trước bucket của nó.">Every one goes after every zero of every block; scanned in that order, each entry sums exactly the counts in front of its bucket.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Thứ tự nào cũng được: phép cộng có tính giao hoán và bảng chứa cùng các số đếm dù thế nào, nên phép scan cho mọi bucket cùng điểm bắt đầu.">Any order works: addition is commutative and the table holds the same counts either way, so the scan gives every bucket the same start.</Translated>,
        why: <Translated vi="Tổng cuối cùng thì như nhau, nhưng một phép scan loại trừ cho một tiền tố khác ở mỗi phần tử. Hãy thử bố cục từng khối một trong trình khám phá.">The final total is the same, but an exclusive scan gives a different prefix at each entry. Try the block-by-block layout in the explorer.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một phép scan loại trừ cho mỗi phần tử tổng của mọi thứ được liệt kê trước nó. Điểm bắt đầu của bucket 1 của khối b phải
            là số khóa thuộc về phía trước nó: mọi bit 0, của mọi khối, cộng các bit 1 của khối 0 đến b − 1. Chỉ một bảng có mọi số
            đếm bit 0 trước, rồi đến các số đếm bit 1 theo thứ tự khối, mới đặt đúng những phần tử đó trước nó.
          </p>
        }
      >
        <p>
          An exclusive scan gives each entry the sum of everything listed before it. The start of block b's one bucket must be
          the number of keys that belong in front of it: all zeros, of every block, plus the ones of blocks 0 to b − 1. Only a
          table with all zero counts first, then the one counts in block order, puts exactly those entries before it.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.6, p. 340',
  },
};

// ---------------------------------------------------------------- the section
export function S146() {
  const { devs } = useLab2();
  return (
    <>
      <Translated
        vi={
          <p>
            Lượt của mục 14.5 đúng và ổn định. Vấn đề nằm ở các lần ghi của nó. Mỗi luồng tính khóa của mình đi đâu và ghi nó vào
            đó, và đích đến phụ thuộc vào bit của khóa, không phải vị trí của luồng. Trong khối 0 của ví dụ của sách, bốn luồng lần
            lượt ghi vào bucket 0, bucket 1, bucket 0 và bucket 1: các luồng kề nhau, địa chỉ cách xa nhau.
          </p>
        }
      >
        <p>
          The pass of section 14.5 is correct and stable. Its stores are the problem. Every thread computes where its key goes
          and stores it there, and the destination depends on the key's bit, not on the thread's position. In block 0 of the
          book's example the four threads write to the zero bucket, the one bucket, the zero bucket and the one bucket in
          turn: neighbouring threads, addresses far apart.
        </p>
      </Translated>
      <Source section="14.6" pages="339–340" figs={tr('Fig. 14.5', 'Hình 14.5')} />
      <Translated
        vi={
          <p>
            Điều đó tốn bao nhiêu? Hãy nhìn vào bên trong một warp. Đích đến của mỗi bit 0 lớn hơn của bit 0 trước nó một đơn vị, vì
            phép scan đếm đúng các khóa ở giữa, và các bit 1 cũng vậy. Nên lần ghi của một warp có tối đa hai đoạn địa chỉ liên
            tiếp. Sách ước tính warp chạm vào hai cache line, hoặc nhiều hơn một chút tùy cách căn chỉnh.
          </p>
        }
      >
        <p>
          How much does that cost? Look inside one warp. Each zero's destination is one more than the previous zero's, because
          the scan counted exactly the keys in between, and the same holds for the ones. So a warp's store is at most two runs
          of consecutive addresses. The book's estimate is that the warp touches two cache lines, or a little more depending on
          alignment.
        </p>
      </Translated>
      <Source section="14.6" pages="340" />
      <Translated
        vi={
          <p>
            Tính theo đơn vị của chính NVIDIA, sector 32 byte, hai đoạn tốn 4 đến 7 sector, trong khi một lần ghi được hợp nhất của 32
            khóa bốn byte tốn 4. Ở một bit đó là một tổn thất vừa phải. Nó tăng theo chữ số: với chữ số r bit, một warp ghi tối đa{' '}
            2<sup>r</sup> đoạn, và với 8 bit trên khóa ngẫu nhiên gần như mỗi lane có một bucket riêng, tiến gần tới một sector cho
            mỗi khóa.
          </p>
        }
      >
        <p>
          In NVIDIA's own unit, the 32-byte sector, the two runs cost 4 to 7 sectors, where one coalesced store of 32 four-byte
          keys costs 4. At one bit that is a modest loss. It grows with the digit: with an r-bit digit a warp writes up to
          2<sup>r</sup> runs, and with 8 bits on random keys almost every lane has a bucket to itself, which approaches one
          sector per key.
        </p>
      </Translated>
      <Today checked="2026-11-20">
        <Translated
          vi={
            <p>
              NVIDIA phục vụ truy cập của một warp theo các sector 32 byte, bốn sector cho một line 128 byte, với thứ tự lane
              bất kỳ (ch06-facts A2, A4). AMD gom các truy cập của một wavefront thành các line 64 hoặc 128 byte (B1), và
              wavefront của nó rộng 64 trên gfx9 (ch05-facts B5). Apple không ghi nhận quy tắc nào trong tài liệu (C1): hãy đo.
              Phép tính sector là của khóa học này (ch14-facts A10).
            </p>
          }
        >
          <p>
            NVIDIA serves a warp's access in 32-byte sectors, four to a 128-byte line, in any lane order (ch06-facts A2, A4).
            AMD groups a wavefront's accesses into 64- or 128-byte lines (B1), and its wavefronts are 64 wide on gfx9
            (ch05-facts B5). Apple documents no rule at all (C1): measure. The sector arithmetic is ours (ch14-facts A10).
          </p>
        </Translated>
      </Today>
      <Predict
        id="ch14/14.6/p-direct-r8"
        prompt={tr(
          'Before you open the explorer: random 32-bit keys, blocks of 256 threads, an 8-bit digit, the direct scatter. How many 32-byte sectors does an average warp store touch?',
          'Trước khi mở trình khám phá: khóa 32 bit ngẫu nhiên, khối 256 luồng, chữ số 8 bit, phân tán trực tiếp. Trung bình một lần ghi của warp chạm vào bao nhiêu sector 32 byte?',
        )}
        unit={tr('sectors', 'sector')}
      >
        {(v) => (
          <p>
            <Translated
              vi={
                <>
                  Mô hình cho <b>{dec(P_R8.meanUnits)}</b> sector mỗi lần ghi, ứng với {dec(P_R8.meanRuns)} đoạn trong 32 lane; ở
                  một bit là {dec(P_R1.meanUnits)}.{' '}
                </>
              }
            >
              The model gives <b>{dec(P_R8.meanUnits)}</b> sectors per store, for {dec(P_R8.meanRuns)} runs among 32 lanes; at
              one bit it is {dec(P_R1.meanUnits)}.{' '}
            </Translated>
            {Math.abs(v - P_R8.meanUnits) <= 3
              ? tr('Close to yours.', 'Gần với con số của bạn.')
              : v < P_R8.meanUnits
                ? tr('More than you said: with 256 buckets, two lanes of a warp rarely share one, so nearly every lane is a run of its own.', 'Nhiều hơn bạn nói: với 256 bucket, hai lane của một warp hiếm khi chung một bucket, nên gần như mỗi lane là một đoạn riêng.')
                : tr('Fewer than you said: a run of one 4-byte key still costs only one sector, and a few lanes do share a bucket.', 'Ít hơn bạn nói: một đoạn chỉ một khóa 4 byte vẫn chỉ tốn một sector, và vài lane có chung bucket.')}{' '}
            <Translated vi="Đặt trình khám phá về r = 8 và kiểm tra.">Set the explorer to r = 8 and check it.</Translated>
          </p>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Trình khám phá chạy một lượt của phép sắp xếp và theo dõi các lần ghi của nó từng warp một. Chọn một lần ghi bằng thanh
            trượt: khung bên trái đếm số đoạn và số đơn vị bộ nhớ mà nó chạm vào, còn bảng bên phải lấy trung bình mọi lần ghi của
            lượt trong cả hai thiết kế, để bạn so sánh mà không cần chuyển qua lại.
          </p>
        }
      >
        <p>
          The explorer runs one pass of a sort and follows its stores one warp at a time. Pick a store with the slider: the left
          panel counts its runs and the memory units it touches, and the table on the right averages every store of the pass in
          both designs, so you can compare them without switching.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi lúc một lần ghi rộng cả warp, không cache, không thời gian: một sector mà hai lần ghi cùng chạm vào được đếm hai lần.
              Kích thước phân đoạn là đơn vị được các hãng ghi trong tài liệu; với Metal kích thước do bạn chọn. Mảng đầu ra bắt đầu
              trên một ranh giới đơn vị.
            </p>
          }
        >
          <p>
            One warp-wide store at a time, no cache, no timing: a sector two stores touch counts twice. Segment sizes are the
            vendors' documented units; for Metal the size is yours. The output array starts on a unit boundary.
          </p>
        </Translated>
      </Simplification>
      <WarpWrites />
      <Translated
        vi={
          <p>
            Ba điều cần tìm. Thứ nhất, phân tán trực tiếp không bao giờ tạo nhiều đoạn mỗi lần ghi hơn số nhỏ hơn giữa độ rộng warp
            và 2<sup>r</sup>, và ở r = 8 nó tiến gần tới độ rộng warp. Thứ hai, sắp xếp cục bộ không thay đổi gì với 4 luồng mỗi
            khối và cắt giảm số đoạn với 256; phần tiếp theo nói vì sao. Thứ ba, trên hàng softmax các chữ số cao chỉ nhận một nhúm
            giá trị, vì chúng chứa số mũ của float, nên các lượt cuối của phép sắp xếp xác suất ghi ít đoạn mà dài, ngay cả khi
            không có sắp xếp cục bộ.
          </p>
        }
      >
        <p>
          Three things to find. First, the direct scatter never makes more runs per store than the smaller of the warp width
          and 2<sup>r</sup>, and at r = 8 it comes close to the warp width. Second, the local sort changes nothing with 4
          threads per block and cuts the runs with 256; the next part says why. Third, on the softmax row the top digits take
          only a handful of values, because they hold the float's exponent, so the last passes of a sort of probabilities
          write few, long runs even without a local sort.
        </p>
      </Translated>
      <Ask q={questionsS146['ch14/14.6/q-sectors']} />

      <h2>
        <Translated vi="Phân tán trong bộ nhớ chia sẻ">Do the scattering in shared memory</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương 6 đưa ra ba lối thoát cho một mẫu truy cập không được hợp nhất: đổi luồng nào xử lý dữ liệu nào, đổi bố cục dữ
            liệu, hoặc làm phần bất quy tắc trong bộ nhớ chia sẻ (shared memory) và di chuyển dữ liệu giữa bộ nhớ chia sẻ và bộ nhớ
            toàn cục một cách có trật tự. Một phép sắp xếp không chọn được bố cục đầu ra, nên chương này đi theo cách thứ ba, như{' '}
            <a href={href('ch12', '12.6')}>mục 12.6</a> đã làm cho bộ lọc ổn định.
          </p>
        }
      >
        <p>
          Chapter 6 offered three ways out of an access pattern that does not coalesce: change which thread handles which data,
          change the layout of the data, or do the irregular part in shared memory and move data between shared and global
          memory in an orderly way. A sort cannot choose its output layout, so this chapter takes the third, as{' '}
          <a href={href('ch12', '12.6')}>section 12.6</a> did for the stable filter.
        </p>
      </Translated>
      <Source section="14.6" pages="340" />
      <Aside>
        <Translated
          vi={
            <p>
              Ba lựa chọn nằm ở <a href={href('ch06', '6.1')}>mục 6.1</a> (sách §6.1, p. 129).
            </p>
          }
        >
          <p>
            The three options are in <a href={href('ch06', '6.1')}>section 6.1</a> (book §6.1, p. 129).
          </p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Trước hết mỗi khối chạy cùng lượt đó chỉ trên các khóa của riêng nó, trong bộ nhớ chia sẻ, với một phép scan trên khối
            thay vì trên toàn bộ đầu vào. Các khóa của khối ra đã được sắp theo chữ số vào các bucket cục bộ. Rồi khối sao chép các
            bucket cục bộ ra bộ nhớ toàn cục, mỗi bucket vào chỗ của nó bên trong bucket toàn cục tương ứng. Bên trong một bucket,
            các luồng kề nhau giờ ghi vào các địa chỉ kề nhau.
          </p>
        }
      >
        <p>
          Each block first runs the same pass on its own keys only, in shared memory, with a scan over the block instead of over
          the whole input. The block's keys come out sorted by the digit into local buckets. Then the block copies its local
          buckets to global memory, each one to its place inside the matching global bucket. Inside a bucket, neighbouring
          threads now write neighbouring addresses.
        </p>
      </Translated>
      <Source section="14.6" pages="340" figs={tr('Fig. 14.8', 'Hình 14.8')} />
      <Translated
        vi={
          <p>
            Hãy xem điều đó thay đổi gì và không thay đổi gì. Trong ví dụ của sách, khối 0 ghi vào các vị trí 0, 1, 8 và 9 trong cả
            hai thiết kế; chỉ luồng nào ghi vào đâu là thay đổi. Lần ghi của một warp tốn số sector mà các địa chỉ của nó chạm vào,
            bất kể thứ tự lane, nên với một khối không rộng hơn một warp thì sắp xếp cục bộ không mang lại gì. Nó có lợi khi một khối
            chứa nhiều warp: sau khi sắp xếp cục bộ, mỗi warp sao chép một dải ô cục bộ liên tiếp, và chúng phần lớn nằm trong một
            bucket cục bộ. Trong trình khám phá, hãy so 4 luồng mỗi khối với 256.
          </p>
        }
      >
        <p>
          Look at what that changes and what it does not. In the book's example, block 0 writes to positions 0, 1, 8 and 9 in
          both designs; only which thread writes which has changed. A warp's store costs the sectors its addresses touch,
          whatever the lane order, so for a block no wider than a warp the local sort buys nothing. It pays when a block holds
          several warps: after the local sort each warp copies a stretch of consecutive local slots, and those mostly lie in one
          local bucket. In the explorer, compare 4 threads per block with 256.
        </p>
      </Translated>
      <Beyond title={tr('Ours', 'Của khóa học này')}>
        <Translated
          vi={
            <p>
              Nhận xét về thứ tự lane suy ra từ quy tắc của NVIDIA rằng truy cập của một warp tốn số sector mà nó chạm vào "in some
              linear or permuted way" (theo một cách tuyến tính hoặc hoán vị nào đó) (ch06-facts A2). Sách không nêu điều này.
            </p>
          }
        >
          <p>
            The point about lane order follows from NVIDIA's rule that a warp's access costs the sectors it touches "in some
            linear or permuted way" (ch06-facts A2). The book does not make it.
          </p>
        </Translated>
      </Beyond>
      <LocalSortScene />

      <h2>
        <Translated vi="Mỗi bucket cục bộ đi về đâu">Where each local bucket goes</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một khối chỉ ghi được các bucket của nó khi biết chúng bắt đầu ở đâu, và điều đó phụ thuộc vào mọi khối khác. Bucket 0
            của khối b đi sau các bucket 0 của khối 0 đến b − 1. Bucket 1 của nó đi sau các bucket 0 của mọi khối và sau các bucket
            1 của khối 0 đến b − 1.
          </p>
        }
      >
        <p>
          A block can only write its buckets once it knows where they start, and that depends on every other block. Block b's
          zero bucket goes after the zero buckets of blocks 0 to b − 1. Its one bucket goes after the zero buckets of all the
          blocks and after the one buckets of blocks 0 to b − 1.
        </p>
      </Translated>
      <Source section="14.6" pages="340" />
      <Translated
        vi={
          <p>
            Một phép scan trả lời tất cả cùng lúc. Sau khi sắp xếp cục bộ, mỗi khối ghi kích thước các bucket của mình vào một bảng,
            và bảng được bố trí theo từng chữ số: số đếm bit 0 của mọi khối, rồi số đếm bit 1 của mọi khối. Một phép scan loại trừ
            trên mảng đó cho, tại mỗi phần tử, số khóa thuộc về phía trước bucket đó, tức điểm bắt đầu của nó. Trong ví dụ của sách,
            các số đếm là [2, 2, 1, 3] và [2, 2, 3, 1], và phép scan cho điểm bắt đầu bucket 0 là [0, 2, 4, 5] và điểm bắt đầu
            bucket 1 là [8, 10, 12, 15].
          </p>
        }
      >
        <p>
          One scan answers all of these at once. After its local sort every block writes its bucket sizes into a table, and the
          table is laid out digit by digit: the zero counts of all blocks, then the one counts of all blocks. An exclusive scan of
          that array gives, at each entry, the number of keys that belong in front of that bucket, which is its start. In the
          book's example the counts are [2, 2, 1, 3] and [2, 2, 3, 1], and the scan gives zero-bucket starts [0, 2, 4, 5] and
          one-bucket starts [8, 10, 12, 15].
        </p>
      </Translated>
      <Source section="14.6" pages="340–341" figs={tr('Fig. 14.9', 'Hình 14.9')} />
      <BucketTable />
      <Ask q={questionsS146['ch14/14.6/q-start']} />
      <Ask q={questionsS146['ch14/14.6/q-layout']} />

      <h2>
        <Translated vi="Giai đoạn ghi">The write phase</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mỗi khối giữ hai số đếm của riêng nó, nên mỗi luồng có thể biết từ chỉ số của mình rằng ô cục bộ của nó thuộc bucket
            nào. Luồng t sao chép ô cục bộ t: nếu t nhỏ hơn số đếm bit 0 của khối, nó ghi vào điểm bắt đầu bucket 0 của khối cộng t,
            ngược lại ghi vào điểm bắt đầu bucket 1 cộng t trừ số đếm bit 0. Trong khối 2, vốn chứa một bit 0, luồng 0 ghi bit 0 và
            các luồng 1 đến 3 ghi ba bit 1 cạnh nhau. Trong khối 3 thì ngược lại: các luồng 0 đến 2 ghi bit 0, luồng 3 ghi bit 1.
          </p>
        }
      >
        <p>
          Each block keeps its own two counts, so every thread can tell from its index which bucket its local slot is in.
          Thread t copies local slot t: if t is below the block's zero count it writes to the block's zero start plus t,
          otherwise to its one start plus t minus the zero count. In block 2, which holds one zero, thread 0 writes the zero and
          threads 1 to 3 write the three ones side by side. In block 3 it is the other way round: threads 0 to 2 write zeros,
          thread 3 the one.
        </p>
      </Translated>
      <Source section="14.6" pages="341–342" />
      <Ask q={questionsS146['ch14/14.6/q-thread']} />

      <h2>
        <Translated vi="Ba kernel mỗi lượt">Three kernels per pass</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bảng tạo ra một sự phụ thuộc trên toàn lưới: không khối nào được ghi cho đến khi mọi khối đã đếm xong và bảng đã được
            scan. Hình 14.7 giấu cùng vấn đề đó bên trong <code>gridExclusiveScan</code> của nó. Lối thoát đơn giản là kết thúc
            kernel tại mỗi điểm như vậy. Một lượt trở thành ba lần khởi chạy: đếm chữ số của mỗi khối, scan bảng, rồi cho mỗi khối
            sắp xếp cục bộ các khóa của mình và phân tán chúng. Các khóa được đọc hai lần, một lần để đếm và một lần để di chuyển, và
            không có gì phải chờ bên trong một kernel. Sách để phần mã làm bài tập 1; lab 14.2 chính là bài tập đó.
          </p>
        }
      >
        <p>
          The table creates a dependence across the whole grid: no block may write until every block has counted and the table
          has been scanned. Fig. 14.7 hid the same problem inside its <code>gridExclusiveScan</code>. The plain way out is to
          end the kernel at each such point. One pass becomes three launches: count each block's digits, scan the table, then
          let every block sort its keys locally and scatter them. The keys are read twice, once to count and once to move, and
          nothing has to wait inside a kernel. The book leaves the code as exercise 1; lab 14.2 is that exercise.
        </p>
      </Translated>
      <Source section="14.11" pages="347" />
      <Today checked="2026-11-20">
        <Translated
          vi={
            <p>
              Radix sort của PyTorch cho GPU Apple chính là thiết kế này viết bằng Metal: các kernel <code>radix_count</code>,{' '}
              <code>radix_scan</code> và <code>radix_scatter</code>, bảng bố trí theo từng chữ số qua các khối như trong Hình 14.9, và
              phép chia theo từng bit cho sắp xếp cục bộ (ch14-facts D6). Metal không ghi nhận rào chắn nào giữa các threadgroup trong
              tài liệu, nên các dispatch riêng là con đường được tài liệu hóa ở đó (C4).
            </p>
          }
        >
          <p>
            PyTorch's radix sort for Apple GPUs is this design written in Metal: kernels <code>radix_count</code>,{' '}
            <code>radix_scan</code> and <code>radix_scatter</code>, the table laid out digit by digit across blocks as in Fig.
            14.9, and a per-bit split for the local sort (ch14-facts D6). Metal documents no barrier across threadgroups, so
            separate dispatches are the documented route there (C4).
          </p>
        </Translated>
      </Today>
      <LabListing />
      <Predict
        id="ch14/14.6/p-local-gain"
        prompt={tr(
          'Lab 14.2 provides the one-bit direct-scatter sort of section 14.5 and asks you for the local-sort design. At r = 1 and one key per thread, how many times faster will the local-sort design be on your device?',
          'Lab 14.2 cung cấp phép sắp xếp phân tán trực tiếp một bit của mục 14.5 và yêu cầu bạn viết thiết kế sắp xếp cục bộ. Với r = 1 và một khóa mỗi luồng, thiết kế sắp xếp cục bộ sẽ nhanh hơn bao nhiêu lần trên thiết bị của bạn?',
        )}
        unit="×"
      >
        {(v) =>
          devs.length ? (
            <>
              {devs.map((d) => {
                const x = d.onebit / d.local;
                return (
                  <p key={d.b} data-measured={d.b}>
                    <Translated
                      vi={
                        <>
                          <b>{d.device}</b> ({d.b}): phép sắp xếp trực tiếp một bit mất {fmt(d.onebit * 1000)} ms và sắp xếp cục bộ ở r
                          = 1, một khóa mỗi luồng, mất {fmt(d.local * 1000)} ms: <b>{fmt(x)}×</b>
                          {x < 1 ? ', nên sắp xếp cục bộ chậm hơn' : ''}. Bạn đã dự đoán {v}×.
                        </>
                      }
                    >
                      <b>{d.device}</b> ({d.b}): the direct one-bit sort took {fmt(d.onebit * 1000)} ms and the local sort at r = 1,
                      one key per thread, {fmt(d.local * 1000)} ms: <b>{fmt(x)}×</b>
                      {x < 1 ? ', so the local sort was slower' : ''}. You predicted {v}×.
                    </Translated>
                  </p>
                );
              })}
              <p className="muted">
                <Translated
                  vi={
                    <>
                      Ở một bit, số sector của trình khám phá chỉ giảm một chút; mọi khác biệt khác giữa hai thiết kế (phép scan
                      bảng thay vì scan trên mọi khóa, lần đọc khóa thứ hai) cũng nằm trong phép đo.
                    </>
                  }
                >
                  At one bit the explorer's sector counts fall only a little; whatever else differs between the two (the
                  table scan instead of a scan over every key, the second read of the keys) is in the measurement too.
                </Translated>
              </p>
            </>
          ) : (
            <p data-measured="none">
              <Translated
                vi={
                  <>
                    Chưa có phép đo nào trên máy này. Hãy chạy <code>python3 labs/pmpp.py run 14.2 --solution --bench</code>, rồi
                    tải lại trang này để so sánh.
                  </>
                }
              >
                No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 14.2 --solution --bench</code>, then
                reload this page to compare.
              </Translated>
            </p>
          )
        }
      </Predict>
    </>
  );
}
