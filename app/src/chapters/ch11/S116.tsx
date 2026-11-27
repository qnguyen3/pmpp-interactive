import { useEffect, useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Bench, Beyond, NoteGroup, Simplification, Source, Today, type Tier } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type BenchRow } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { SEGMENT_MODELS, type SegmentModelId } from '../../sim/coalesce';
import {
  coarsenedSteps,
  coarsenedWork,
  demoValues,
  koggeStoneSteps,
  koggeStoneWork,
  loadTransactions,
  loaderOf,
  ownerOf,
  sequentialWork,
  threePhases,
  warpLoad,
  type LoadMode,
} from '../../sim/coarsescan';
import { functionSource } from '../../sim/toolchain';
import { CoarsePhasesScene } from './CoarsePhasesScene';

const STATUS = {
  get printed() {
    return tr(
      'front-end-checked; emulator-checked (N a multiple of the block segment); unverified on CUDA hardware; as printed',
      'front-end-checked; emulator-checked (N là bội số của đoạn khối); unverified trên phần cứng CUDA; như bản in',
    );
  },
} as const;
const PRINTED = 'ch11/snippets/s116_coarsened_as_printed.cu';
const LAB = 'ch11/lab2_coarsened';
const missing = (s: string | null) => !s || s.startsWith('// could not load');
const TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', buffer_s: 'shared', threadSums: 'shared', warpSums_s: 'shared',
  threadSum: 'reg', prevPartialSum: 'reg', blockSegment: 'reg', threadSegment: 'reg', c: 'reg', buffer_r: 'reg',
};
const range = (n: number, from = 0) => Array.from({ length: n }, (_, i) => from + i);
const int = (v: number) => (Number.isInteger(v) ? v.toLocaleString('en-US') : fmt(v));


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

// ---------------------------------------------------------------- explorer: three phases
const PS = [4, 8, 16, 32, 64, 128, 256, 512, 1024];
const PRESETS: { label: string; P: number; F: number }[] = [
  {
    get label() {
      return tr('Fig. 11.6\'s sixteen', 'Mười sáu phần tử của Fig. 11.6');
    },
    P: 4,
    F: 4,
  },
  {
    get label() {
      return tr('64 threads, factor 8', '64 luồng, hệ số 8');
    },
    P: 64,
    F: 8,
  },
  {
    get label() {
      return tr('factor 1 (P = N)', 'hệ số 1 (P = N)');
    },
    P: 1024,
    F: 1,
  },
  {
    get label() {
      return tr('1,024 threads, factor 16', '1,024 luồng, hệ số 16');
    },
    P: 1024,
    F: 16,
  },
];

function ThreePhasesExplorer() {
  const [P, setP] = useState(4);
  const [F, setF] = useState(4);
  const N = P * F;
  const draw = P <= 16 && F <= 8;
  const ph = useMemo(() => (draw ? threePhases(demoValues(N), P) : null), [draw, N, P]);
  const w = coarsenedWork(N, P);
  const s = coarsenedSteps(N, P);
  const ks = koggeStoneWork(N);
  const kss = koggeStoneSteps(N, P);
  const cellRow = (vals: number[], tier: 'global' | 'reg' | 'cell') => (
    <span style={{ display: 'inline-flex', gap: 2 }}>
      {vals.map((v, i) => (
        <span key={i} style={{
          minWidth: '1.75rem', padding: '0.1rem 0.15rem', textAlign: 'center', fontVariantNumeric: 'tabular-nums', fontSize: '0.8rem', borderRadius: 2,
          border: `1px solid ${tier === 'cell' ? 'var(--rule)' : `var(--${tier})`}`, background: tier === 'cell' ? 'var(--cell)' : `var(--${tier}-wash)`,
        }}>{v}</span>
      ))}
    </span>
  );
  return (
    <Bench
      title={tr('Three phases', 'Ba pha')}
      task={tr('Run the small block first. Then push the factor up and the thread count down, and watch which phase the work moves into.', 'Chạy khối nhỏ trước. Rồi tăng hệ số và giảm số luồng, và xem khối lượng việc dồn sang pha nào.')}
      foot={
        <Translated
          vi={
            <>
              N = BLOCK_DIM × COARSE_FACTOR: đoạn của một khối, như trong Fig. 11.12. Các con số đếm là công thức của sách (pp.
              266, 269–270); "số bước" giả định P đơn vị thực thi, mỗi luồng một đơn vị. Trường hợp nhỏ dùng các số của Fig. 11.6
              với 16 phần tử.
            </>
          }
        >
          <>
            N = BLOCK_DIM × COARSE_FACTOR: one block's segment, as in Fig. 11.12. The counts are the book's formulas (pp. 266,
            269–270); "steps" assume P execution units, one per thread. The small instance uses Fig. 11.6's numbers at 16
            elements.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="muted"><Translated vi="Bộ giá trị có sẵn:">Presets:</Translated></span>
        {PRESETS.map((p) => (
          <button key={p.label} className="quiet" onClick={() => { setP(p.P); setF(p.F); }}>{p.label}</button>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi={<>BLOCK_DIM (số luồng, P): <b>{P}</b></>}>BLOCK_DIM (threads, P): <b>{P}</b></Translated>
          <input type="range" min={0} max={PS.length - 1} value={PS.indexOf(P)} aria-label="BLOCK_DIM" onChange={(e) => setP(PS[+e.target.value])} />
        </label>
        <label className="field">
          <Translated vi={<>COARSE_FACTOR (mỗi luồng): <b>{F}</b></>}>COARSE_FACTOR (per thread): <b>{F}</b></Translated>
          <input type="range" min={1} max={16} value={F} aria-label="COARSE_FACTOR" onChange={(e) => setF(+e.target.value)} />
        </label>
        <Translated vi={<span>N = <b data-n>{int(N)}</b> phần tử mỗi khối</span>}><span>N = <b data-n>{int(N)}</b> elements per block</span></Translated>
      </div>
      <div>
        <div style={{ marginBottom: '1rem' }}>
          {ph ? (
            <div style={{ overflowX: 'auto' }}>
              <table className="data" style={{ margin: 0 }} data-phases>
                <thead>
                  <tr><th><Translated vi="luồng">thread</Translated></th><th><Translated vi="dãy của nó (đầu vào)">its run (input)</Translated></th><th><Translated vi="pha 1: dãy của nó sau scan">phase 1: its run scanned</Translated></th><th className="num"><Translated vi="pha 3 cộng">phase 3 adds</Translated></th><th><Translated vi="kết quả">result</Translated></th></tr>
                </thead>
                <tbody>
                  {ph.runs.map((r, t) => (
                    <tr key={t} data-thread={t}>
                      <td>{t}</td>
                      <td>{cellRow(ph.input.slice(t * F, t * F + F), 'global')}</td>
                      <td data-run={r.join(' ')}>{cellRow(r, 'cell')}</td>
                      <td className="num" data-added={ph.added[t]}>{t === 0 ? '0' : `+${ph.added[t]}`}</td>
                      <td data-final={ph.final.slice(t * F, t * F + F).join(' ')}>{cellRow(ph.final.slice(t * F, t * F + F), 'reg')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="narrate" style={{ margin: '0.6rem 0 0' }} data-phase2>
                <Translated
                  vi={
                    <>
                      Pha 2: các tổng của dãy {ph.totals.join(', ')} đi qua một lần scan khối và trở thành{' '}
                      {ph.scannedTotals.join(', ')}. Luồng <i>t</i> cộng tổng đã scan của luồng <i>t</i> − 1.{' '}
                    </>
                  }
                >
                  Phase 2: the run totals {ph.totals.join(', ')} go through one block scan and become{' '}
                  {ph.scannedTotals.join(', ')}. Thread <i>t</i> adds the scanned total of thread <i>t</i> − 1.{' '}
                </Translated>
                {ph.correct ? <span className="chip ok"><Translated vi="kết quả chính là scan">the result is the scan</Translated></span> : <span className="chip fail"><Translated vi="sai">wrong</Translated></span>}
              </p>
            </div>
          ) : (
            <p className="narrate" style={{ margin: 0 }}>
              <Translated
                vi={
                  <>
                    {P} luồng với mỗi luồng {F} phần tử là quá nhiều để vẽ. Các con số bên dưới vẫn áp dụng; giá trị chỉ được vẽ với
                    16 luồng trở xuống và hệ số từ 8 trở xuống.
                  </>
                }
              >
                {P} threads with {F} elements each is too many to draw. The counts below still apply; the values are drawn
                for 16 threads or fewer and a factor of 8 or less.
              </Translated>
            </p>
          )}
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table className="data" style={{ margin: 0, minWidth: '24rem' }} data-counts>
            <thead>
              <tr><th></th><th className="num"><Translated vi="phép cộng">additions</Translated></th><th className="num"><Translated vi="số bước trên P đơn vị">steps on P units</Translated></th></tr>
            </thead>
            <tbody>
              <tr><td><Translated vi="vòng lặp (Fig. 11.1)">the loop (Fig. 11.1)</Translated></td><td className="num">{int(sequentialWork(N))}</td><td className="num">{int(N)}</td></tr>
              <tr><td><Translated vi="Kogge-Stone, mỗi phần tử một luồng">Kogge-Stone, one thread per element</Translated></td><td className="num" data-k="ks-work">{int(ks)}</td><td className="num" data-k="ks-steps">{int(kss)}</td></tr>
              <tr><td><Translated vi={<><b>gộp việc</b>, P = {P} luồng</>}><b>coarsened</b>, P = {P} threads</Translated></td><td className="num" data-k="work"><b>{int(w.total)}</b></td><td className="num" data-k="steps"><b>{int(s.total)}</b></td></tr>
              {w.parts.map((p, i) => (
                <tr key={p.label}>
                  <td className="muted" style={{ paddingLeft: '1rem' }}>{p.label}</td>
                  <td className="num muted">{int(p.value)}</td>
                  <td className="num muted">{int(s.parts[i].value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted" style={{ margin: '0.6rem 0 0' }} data-ratio>
            <Translated vi={<>Khối lượng việc khi gộp là {fmt(w.total / N)} phép cộng mỗi phần tử (vòng lặp: {fmt(sequentialWork(N) / N)}; Kogge-Stone:{' '}{fmt(ks / N)}).</>}>
              Coarsened work is {fmt(w.total / N)} additions per element (the loop: {fmt(sequentialWork(N) / N)}; Kogge-Stone:{' '}
              {fmt(ks / N)}).
            </Translated>{' '}
            {F === 1
              ? tr('With one element per thread there is no sequential phase: the block scan does all the work, as in Kogge-Stone.', 'Với mỗi luồng một phần tử thì không có pha tuần tự: scan khối làm toàn bộ việc, như Kogge-Stone.')
              : P <= F
                ? tr('With few threads and long runs the count approaches 2N: each element is added once in its run and once more in phase 3.', 'Với ít luồng và dãy dài, con số tiến tới 2N: mỗi phần tử được cộng một lần trong dãy của nó và thêm một lần ở pha 3.')
                : ''}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: one warp, two ways to load
const FS = [1, 2, 4, 8, 16];
const B = 32; // threads in the block drawn: one warp

function LoadExplorer() {
  const [F, setF] = useState(4);
  const [c, setC] = useState(0);
  const [model, setModel] = useState<SegmentModelId>('nvidia');
  const [custom, setCustom] = useState(32);
  const [pick, setPick] = useState<number | null>(null);
  const narrow = useNarrow();
  useEffect(() => setC((x) => Math.min(x, F - 1)), [F]);
  const m = SEGMENT_MODELS[model];
  const seg = m.segmentBytes ?? custom;
  const cc = Math.min(c, F - 1);
  const loads = (['staged', 'direct'] as LoadMode[]).map((mode) => warpLoad(mode, cc, F, B, 32, seg));
  const VW = narrow ? 344 : 704;
  const perSeg = seg / 4;
  const groups = perSeg < B ? B / perSeg : 1; // segments per drawn row
  const sg = groups > 1 ? (narrow ? 3 : 5) : 0; // gap between segments
  const cw = (VW - 4 - (groups - 1) * sg) / B;
  const cx = (col: number) => 2 + col * cw + Math.floor(col / perSeg) * sg;
  const ch = Math.max(9, Math.min(18, cw * 0.8));

  const panel = (mode: LoadMode, k: number) => {
    const L = loads[k];
    const touched = new Set(L.elements);
    const segs = new Set(L.access.segments);
    const H = F * (ch + 3) + 6;
    return (
      <div key={mode} style={{ marginBottom: '0.8rem' }} data-mode={mode}>
        <div style={{ fontWeight: 600, fontSize: '0.88rem', margin: '0 0 0.3rem' }}>
          {mode === 'staged' ? tr('Chunk at a time (Fig. 11.12): thread t loads element c·32 + t', 'Từng khúc một (Fig. 11.12): luồng t nạp phần tử c·32 + t') : tr('Each thread loads its own run: thread t loads element t·F + c', 'Mỗi luồng nạp dãy của chính nó: luồng t nạp phần tử t·F + c')}
        </div>
        <svg viewBox={`0 0 ${VW} ${H}`} role="img" aria-label={tr(`Block segment, ${mode} load`, `Đoạn của khối, kiểu nạp ${mode}`)} style={{ width: '100%', maxWidth: VW, display: 'block' }}>
          {range(F * B).map((e) => {
            const row = Math.floor(e / B);
            const col = e % B;
            const xx = cx(col);
            const yy = 2 + row * (ch + 3);
            const sIdx = Math.floor((e * 4) / seg);
            const on = touched.has(e);
            const inSeg = segs.has(sIdx);
            const isPick = pick === e;
            return (
              <rect key={e} data-el={e} data-on={on ? '1' : '0'} x={xx + 0.5} y={yy} width={cw - 1} height={ch} rx="1.5"
                onClick={() => setPick(e)} style={{ cursor: 'pointer' }}
                fill={on ? 'var(--global)' : inSeg ? 'var(--global-wash)' : 'var(--cell)'}
                stroke={isPick ? 'var(--ink)' : on || inSeg ? 'var(--global)' : 'var(--rule)'}
                strokeWidth={isPick ? 2 : 1} />
            );
          })}
        </svg>
        <div className="row" style={{ marginTop: '0.3rem', gap: '1rem' }} data-readout={mode} data-transactions={L.access.transactions} data-runs={L.runs}>
          <Translated vi={<span><b>{L.access.transactions}</b> {({ sector: 'sector', line: 'dòng', segment: 'đoạn' } as Record<string, string>)[m.unit] ?? m.unit} {seg} byte{m.segmentBytes == null ? ' (bạn tự chọn)' : ''}</span>}>
            <span><b>{L.access.transactions}</b> {m.unit}{L.access.transactions === 1 ? '' : 's'} of {seg} bytes{m.segmentBytes == null ? ' (your choice)' : ''}</span>
          </Translated>
          <Translated vi={<span>{Math.round(L.access.efficiency * 100)}% số byte được nạp có dùng</span>}>
            <span>{Math.round(L.access.efficiency * 100)}% of fetched bytes used</span>
          </Translated>
          <span className="muted">{L.runs === 1 ? tr('one run of consecutive bytes', 'một dãy byte liên tiếp') : tr(`${L.runs} separate runs`, `${L.runs} dãy rời nhau`)}</span>
          <Translated vi={<span className="muted">cả {F} lần nạp: {loadTransactions(mode, F, B, 32, seg)} {({ sector: 'sector', line: 'dòng', segment: 'đoạn' } as Record<string, string>)[m.unit] ?? m.unit}</span>}>
            <span className="muted">all {F} load{F === 1 ? '' : 's'}: {loadTransactions(mode, F, B, 32, seg)} {m.unit}s</span>
          </Translated>
        </div>
      </div>
    );
  };

  const info = pick == null || pick >= F * B ? null : {
    staged: loaderOf('staged', pick, B, F),
    direct: loaderOf('direct', pick, B, F),
    owner: ownerOf(pick, F),
  };

  return (
    <Bench
      title={tr('One warp, two ways to load', 'Một warp, hai cách nạp')}
      task={tr('Step through the load iterations at factor 4 and count the segments each way. Then click any element: who loads it, and who scans it?', 'Đi qua các lần lặp nạp ở hệ số 4 và đếm số segment theo mỗi cách. Rồi bấm vào một phần tử bất kỳ: ai nạp nó, và ai scan nó?')}
      foot={
        <Translated
          vi={
            <>
              Một khối 32 luồng, một warp, đoạn gồm 32 × F số float được vẽ từng khúc: hàng <i>c</i> là khúc <i>c</i>, và các khe
              nhỏ chia nó thành các {({ sector: 'sector', line: 'dòng', segment: 'đoạn' } as Record<string, string>)[m.unit] ?? m.unit} {seg} byte. Ô tô đậm là các phần tử mà một lệnh nạp đọc; ô tô nhạt là phần còn lại
              của các {({ sector: 'sector', line: 'dòng', segment: 'đoạn' } as Record<string, string>)[m.unit] ?? m.unit} mà nó chạm tới. Mỗi lệnh được đếm riêng ({m.facts}).
            </>
          }
        >
          <>
            One block of 32 threads, one warp, segment of 32 × F floats drawn chunk by chunk: row <i>c</i> is chunk <i>c</i>, and
            the small gaps split it into {seg}-byte {m.unit}s. Solid cells are the elements one load instruction reads; tinted
            cells are the rest of the {m.unit}s it touches. Each instruction is counted on its own ({m.facts}).
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span className="seg" role="group" aria-label={tr('Coarsening factor', 'Hệ số gộp việc')}>
          {FS.map((f) => <button key={f} aria-pressed={F === f} onClick={() => setF(f)}>F = {f}</button>)}
        </span>
        <span className="row" style={{ gap: '0.4rem' }}>
          <button onClick={() => setC(Math.max(0, cc - 1))} disabled={cc === 0}>c − 1</button>
          <button className="primary" onClick={() => setC(Math.min(F - 1, cc + 1))} disabled={cc >= F - 1}>c + 1</button>
          <span className="muted" aria-live="polite" data-iter={cc}><Translated vi={<>lần lặp nạp c = {cc} / {F - 1}</>}>load iteration c = {cc} of {F - 1}</Translated></span>
        </span>
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Mô hình segment">Segment model</Translated>
          <select value={model} onChange={(e) => setModel(e.target.value as SegmentModelId)} style={{ maxWidth: '100%' }}>
            {Object.values(SEGMENT_MODELS).map((x) => <option key={x.id} value={x.id}>{tr(x.label, ({ nvidia: 'NVIDIA: sector 32 byte trong dòng cache 128 byte', amd64: 'AMD: dòng cache 64 byte', amd128: 'AMD: dòng cache 128 byte', metal: 'Apple: không có tài liệu' } as Record<string, string>)[x.id] ?? x.label)}</option>)}
          </select>
        </label>
        {m.segmentBytes == null ? (
          <label className="field">
            <Translated vi="Bạn tự chọn (không phải sự thật đã kiểm chứng)">Your choice (not a fact)</Translated>
            <select value={custom} onChange={(e) => setCustom(Number(e.target.value))}>
              {[32, 64, 128].map((v) => <option key={v} value={v}>{v} {tr('bytes', 'byte')}</option>)}
            </select>
          </label>
        ) : null}
      </div>
      {panel('staged', 0)}
      {panel('direct', 1)}
      <p className="narrate" role="status" style={{ margin: 0 }} data-pick>
        {info && pick != null
          ? tr(
              `Element ${pick}: with chunks, thread ${info.staged.thread} loads it in iteration ${info.staged.iter}; loading runs directly, thread ${info.direct.thread} would, in iteration ${info.direct.iter}. Either way thread ${info.owner} scans it, because runs are ${F} long: ${pick} ÷ ${F} = ${info.owner}${pick % F ? `, remainder ${pick % F}` : ''}.`,
              `Phần tử ${pick}: nạp theo khúc thì luồng ${info.staged.thread} nạp nó ở lần lặp ${info.staged.iter}; nạp thẳng theo dãy thì sẽ là luồng ${info.direct.thread}, ở lần lặp ${info.direct.iter}. Đằng nào thì luồng ${info.owner} cũng scan nó, vì các dãy dài ${F}: ${pick} ÷ ${F} = ${info.owner}${pick % F ? `, dư ${pick % F}` : ''}.`,
            )
          : tr(
              `In iteration ${cc} the chunk load reads elements ${loads[0].elements[0]} to ${loads[0].elements[31]}, side by side; the direct load reads ${loads[1].elements.slice(0, 3).join(', ')}, …, ${F} apart. Click an element to see who loads it and who scans it.`,
              `Ở lần lặp ${cc}, kiểu nạp theo khúc đọc các phần tử ${loads[0].elements[0]} đến ${loads[0].elements[31]}, liền kề nhau; kiểu nạp thẳng đọc ${loads[1].elements.slice(0, 3).join(', ')}, …, cách nhau ${F}. Bấm vào một phần tử để xem ai nạp nó và ai scan nó.`,
            )}
        {model === 'metal' ? tr(' Apple documents no coalescing rule or segment size: the count uses the size you picked.', ' Apple không công bố quy tắc hợp nhất truy cập hay kích thước segment nào: con số đếm dùng kích thước bạn chọn.') : ''}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- measured reveal (lab 11.2)
/**
 * Lab 11.2's rows: the provided thread_major kernel (register-tiled, each thread loading its
 * own run straight from global memory) against register_tiled (the same kernel staged
 * through shared memory, Fig. 11.13), at the same block size and factor. Speedup = the
 * direct kernel's time over the staged one's.
 */
function stagingPairs(rows: BenchRow[]): { block: number; coarse: number; speedup: number }[] {
  const find = (kernel: string, block: number, coarse: number) =>
    rows.find((r) => r.kernel === kernel && num(r.block) === block && num(r.coarse) === coarse && num(r.seconds) != null);
  return rows
    .filter((r) => r.kernel === 'register_tiled' && num(r.seconds) != null)
    .flatMap((r) => {
      const block = num(r.block)!;
      const coarse = num(r.coarse)!;
      const d = find('thread_major', block, coarse);
      return d ? [{ block, coarse, speedup: num(d.seconds)! / num(r.seconds)! }] : [];
    })
    .sort((a, b) => a.block - b.block || a.coarse - b.coarse);
}

function MeasuredStaging({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const found = realBackends(reports).flatMap((b) => {
    const pairs = stagingPairs(benchRows(reports[b], LAB));
    const main = pairs.find((p) => p.block === 256 && p.coarse === 4) ?? pairs[0];
    return main ? [{ b, device: String(reports[b].device.name ?? b), main, pairs }] : [];
  });
  if (!found.length)
    return (
      <Translated
        vi={
          <p>
            Máy này chưa có số đo nào. Hãy chạy <code>python3 labs/pmpp.py run 11.2 --solution --bench</code>, rồi quay lại: ô
            này sẽ so dự đoán của bạn với kết quả.
          </p>
        }
      >
        <p>
          No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 11.2 --solution --bench</code>, then come back:
          this box will compare your prediction with the result.
        </p>
      </Translated>
    );
  return (
    <>
      {found.map((f) => (
        <p key={f.b} data-measured={f.main.speedup}>
          <Translated
            vi={
              <>
                Trên <b>{f.device}</b> ({f.b}), với khối {f.main.block} luồng ở hệ số {f.main.coarse}, kernel nạp qua bộ nhớ chia sẻ
                chạy nhanh gấp <b>{fmt(f.main.speedup)}×</b> kernel nạp thẳng
                {Math.abs(f.main.speedup - predicted) <= Math.max(0.25, predicted * 0.25)
                  ? ', gần với dự đoán của bạn.'
                  : f.main.speedup < predicted
                    ? '. Bạn đã kỳ vọng nhiều hơn: các lần lặp sau của kiểu nạp thẳng chạm vào những segment mà lần đầu đã nạp, và bộ nhớ đệm (cache) có thể phục vụ những lần lặp lại đó. Mô hình đếm từng lệnh riêng rẽ.'
                    : '. Bạn đã kỳ vọng ít hơn.'}{' '}
                Theo hệ số ở kích thước khối này:{' '}
                {f.pairs.filter((p) => p.block === f.main.block).map((p) => `${p.coarse}: ${fmt(p.speedup)}×`).join(', ')}.
              </>
            }
          >
            On <b>{f.device}</b> ({f.b}), with {f.main.block}-thread blocks at factor {f.main.coarse}, the staged kernel ran{' '}
            <b>{fmt(f.main.speedup)}×</b> as fast as the direct one
            {Math.abs(f.main.speedup - predicted) <= Math.max(0.25, predicted * 0.25)
              ? ', close to what you predicted.'
              : f.main.speedup < predicted
                ? '. You expected more: the direct load\'s later iterations touch the segments its first one fetched, and a cache can serve those repeats. The model counts each instruction on its own.'
                : '. You expected less.'}{' '}
            By factor at this block size:{' '}
            {f.pairs.filter((p) => p.block === f.main.block).map((p) => `${p.coarse}: ${fmt(p.speedup)}×`).join(', ')}.
          </Translated>
        </p>
      ))}
      <p className="muted"><Translated vi="Đây là số đo từ máy của bạn, không phải giá trị kỳ vọng.">These are measurements from your machine, not expected values.</Translated></p>
    </>
  );
}

// ---------------------------------------------------------------- code
function CoarsenedListing() {
  const printed = useLabFile(PRINTED);
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  const metal = useLabFile(LAB + '/solution/kernel.metal');
  // The labs page's words (LabKit's solutionTabStatus), from this machine's runs.
  const { reports } = useReports();
  const labStatus = { cu: solutionTabStatus(reports, 'ch11/lab2_coarsened', 'kernel.cu'), metal: solutionTabStatus(reports, 'ch11/lab2_coarsened', 'kernel.metal') };
  if (missing(printed)) return null;
  const k = functionSource(printed!, 'scan_kernel');
  if (!k) return null;
  const labCu = !missing(cu) ? functionSource(cu!, 'CoarsenedScanKernel') : null;
  const labMetal = !missing(metal) ? functionSource(metal!, 'coarsened_scan') : null;
  const tabs = [
    { label: tr('Fig. 11.12 as printed', 'Fig. 11.12 như bản in'), code: k.text, tiers: TIERS, status: STATUS.printed },
    ...(labCu ? [{ label: tr('lab 11.2 (CUDA and HIP)', 'lab 11.2 (CUDA và HIP)'), code: labCu.text, tiers: TIERS, status: labStatus.cu }] : []),
    ...(labMetal ? [{ label: tr('lab 11.2 (Metal)', 'lab 11.2 (Metal)'), code: labMetal.text, tiers: TIERS, status: labStatus.metal }] : []),
  ];
  return (
    <Code
      tabs={tabs}
      caption={
        <>
          <Translated
            vi={
              <>
                Fig. 11.12 với số dòng của sách (<code>labs/{PRINTED}</code>, trong đó <code>blockScan</code> là Fig. 11.9). Như
                bản in, nó không hề dùng <code>N</code> và không kiểm tra biên, nên chỉ đúng khi N là bội số của{' '}
                <code>COARSE_FACTOR*BLOCK_DIM</code>; phép kiểm tra trên trình giả lập chạy nó theo cách đó.
              </>
            }
          >
            Fig. 11.12 with the book's line numbers (<code>labs/{PRINTED}</code>, where <code>blockScan</code> is Fig. 11.9). As
            printed it never uses <code>N</code> and has no bounds checks, so it is right only when N is a multiple of{' '}
            <code>COARSE_FACTOR*BLOCK_DIM</code>; the emulator check runs it that way.
          </Translated>{' '}
          {labCu
            ? <Translated vi={<>Các tab lab là lời giải tham chiếu (solution) của lab 11.2 (<code>labs/{LAB}/solution/</code>): mọi chỉ số đều được kiểm tra với N, hệ số là một tham số template (một macro trên Metal), và các tổng của warp được truyền vào, như ở lab 11.1.</>}><>The lab tabs are lab 11.2's reference solution (<code>labs/{LAB}/solution/</code>): every index is checked against N, the factor is a template parameter (a macro on Metal), and the warp totals are passed in, as in lab 11.1.</></Translated>
            : <Translated vi={<>Lời giải tham chiếu của lab 11.2, vốn kiểm tra mọi chỉ số với N, sẽ hiện ở đây khi nó có trên máy này.</>}><>Lab 11.2's reference solution, which checks every index against N, will appear here once it is on this machine.</></Translated>}
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- questions
export const questionsS116: Record<string, Question> = {
  'ch11/11.6/q-work': {
    kind: 'numeric',
    id: 'ch11/11.6/q-work',
    prompt: (
      <Translated vi={<>Một khối P = 256 luồng scan N = 1,024 phần tử có gộp việc. Công thức của sách đếm được bao nhiêu phép cộng?</>}>
        <>A block of P = 256 threads scans N = 1,024 elements with coarsening. How many additions does the book's formula count?</>
      </Translated>
    ),
    answer: 3836,
    explain: (
      <Translated
        vi={
          <p>
            2N + P·log2P − P − N/P = 2,048 + 256·8 − 256 − 4 = 3,836. Theo pha: 768 trong các lần scan tuần tự trên dãy (256
            luồng, mỗi luồng 3), 2,048 trong lần scan khối trên 256 tổng, 1,020 trong các phép cộng cuối (255 luồng, mỗi luồng 4).
            Kogge-Stone với mỗi phần tử một luồng làm 1,024·10 − 1,023 = 9,217; vòng lặp làm 1,023.
          </p>
        }
      >
        <p>
          2N + P·log2P − P − N/P = 2,048 + 256·8 − 256 − 4 = 3,836. By phase: 768 in the sequential run scans (256 threads, 3
          each), 2,048 in the block scan of the 256 totals, 1,020 in the adds (255 threads, 4 each). Kogge-Stone with one thread
          per element does 1,024·10 − 1,023 = 9,217; the loop does 1,023.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.6, pp. 269–270; §11.5, p. 266; model: src/sim/coarsescan.ts',
    vi: { source: 'ch. 11 §11.6, pp. 269–270; §11.5, p. 266; mô hình: src/sim/coarsescan.ts' },
  },
  'ch11/11.6/q-steps': {
    kind: 'numeric',
    id: 'ch11/11.6/q-steps',
    prompt: (
      <Translated vi={<>N = 1,024, P = 32 luồng (và 32 đơn vị thực thi). Scan gộp việc mất bao nhiêu bước?</>}>
        <>N = 1,024, P = 32 threads (and 32 execution units). How many steps does the coarsened scan take?</>
      </Translated>
    ),
    answer: 68,
    explain: (
      <Translated
        vi={
          <p>
            2N/P + log2P − 1 = 64 + 5 − 1 = 68: 31 bước cho lần scan tuần tự 32 phần tử của mỗi luồng, 5 cho lần scan khối trên
            32 tổng, 32 cho các phép cộng cuối. Kogge-Stone trên 32 đơn vị mất (1,024·10)/32 = 320 (p. 266), nên gộp việc giảm số
            bước gần 5 lần trên cùng phần cứng.
          </p>
        }
      >
        <p>
          2N/P + log2P − 1 = 64 + 5 − 1 = 68: 31 steps for each thread's sequential scan of 32 elements, 5 for the block scan of
          32 totals, 32 for the adds. Kogge-Stone on 32 units takes (1,024·10)/32 = 320 (p. 266), so coarsening cuts the steps
          by a factor of almost 5 on the same hardware.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.6, p. 270; §11.5, pp. 266–267', // i18n-ignore: citation
  },
  'ch11/11.6/q-owner': {
    kind: 'numeric',
    id: 'ch11/11.6/q-owner',
    prompt: (
      <Translated
        vi={
          <>
            COARSE_FACTOR = 4, BLOCK_DIM = 256, khối 0. Ở lần lặp nạp <code>c = 2</code>, luồng 5 chép một phần tử vào{' '}
            <code>buffer_s</code>. Về sau luồng nào scan phần tử đó?
          </>
        }
      >
        <>
          COARSE_FACTOR = 4, BLOCK_DIM = 256, block 0. In load iteration <code>c = 2</code>, thread 5 copies one element into{' '}
          <code>buffer_s</code>. Which thread later scans that element?
        </>
      </Translated>
    ),
    answer: 129,
    explain: (
      <Translated
        vi={
          <p>
            Luồng 5 chép phần tử <code>c·BLOCK_DIM + threadIdx.x</code> = 2·256 + 5 = 517 (chunk-major). Các dãy dài 4 và luồng{' '}
            <i>t</i> sở hữu các phần tử 4<i>t</i> đến 4<i>t</i> + 3 (thread-major), nên phần tử 517 thuộc về luồng 517 ÷ 4 = 129.
            Luồng nạp một phần tử và luồng scan nó gần như không bao giờ trùng nhau; rào chắn sau lần nạp là thứ làm điều đó an
            toàn.
          </p>
        }
      >
        <p>
          Thread 5 copies element <code>c·BLOCK_DIM + threadIdx.x</code> = 2·256 + 5 = 517 (chunk-major). Runs are 4 long and
          thread <i>t</i> owns elements 4<i>t</i> to 4<i>t</i> + 3 (thread-major), so element 517 belongs to thread 517 ÷ 4 = 129.
          The thread that loads an element and the thread that scans it are almost never the same; the barrier after the load
          is what makes that safe.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.6, pp. 268–269 (Fig. 11.12, lines 06 and 11); model: src/sim/coarsescan.ts',
    vi: { source: 'ch. 11 §11.6, pp. 268–269 (Fig. 11.12, dòng 06 và 11); mô hình: src/sim/coarsescan.ts' },
  },
  'ch11/11.6/q-sectors': {
    kind: 'numeric',
    id: 'ch11/11.6/q-sectors',
    prompt: (
      <Translated
        vi={
          <>
            Hệ số 4, và mỗi luồng nạp dãy của chính nó thẳng từ bộ nhớ toàn cục. Ở lệnh nạp đầu tiên, một warp 32 lane đọc các số
            float 0, 4, 8, …, 124. Nó chạm tới bao nhiêu sector 32 byte?
          </>
        }
      >
        <>
          Factor 4, and each thread loads its own run directly from global memory. In the first load instruction a 32-lane warp
          reads floats 0, 4, 8, …, 124. How many 32-byte sectors does it touch?
        </>
      </Translated>
    ),
    answer: 16,
    unit: 'sectors',
    vi: { unit: 'sector', source: 'ch. 11 §11.6, p. 269; sector 32 byte của NVIDIA (ch06-facts A2 đến A4); mô hình: coarsescan.ts với coalesce.ts' },
    explain: (
      <Translated
        vi={
          <p>
            Các lane cách nhau 16 byte, nên hai lane dùng chung mỗi sector 32 byte: 32 lane, 16 sector, trong đó một nửa số byte
            được dùng. Một khúc liền gồm 32 số float là 128 byte, 4 sector. Từ hệ số 8 trở lên, mỗi lane có một sector riêng, nên
            con số tăng theo hệ số cho đến khi đạt mỗi lane một sector, 32, gấp tám lần con số 4 của kiểu nạp qua bộ nhớ chia sẻ.
          </p>
        }
      >
        <p>
          The lanes are 16 bytes apart, so two lanes share each 32-byte sector: 32 lanes, 16 sectors, of which half the bytes
          are used. One contiguous chunk of 32 floats is 128 bytes, 4 sectors. At factor 8 and beyond every lane has a sector
          of its own, so the count grows with the factor until it reaches one sector per lane, 32, eight times the staged
          load's 4.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.6, p. 269; NVIDIA\'s 32-byte sector (ch06-facts A2 to A4); model: coarsescan.ts with coalesce.ts',
  },
  'ch11/11.6/q-fewer-blocks': {
    kind: 'choice',
    id: 'ch11/11.6/q-fewer-blocks',
    prompt: (
      <Translated
        vi={
          <>
            Bạn có thể gộp việc theo hai cách: giữ nguyên đoạn của khối và cho nó ít luồng hơn, hoặc giữ nguyên số luồng và cho
            khối một đoạn dài hơn. Cách nào giúp bước liên khối của mục 11.9, và vì sao?
          </>
        }
      >
        <>
          You can coarsen two ways: keep the block's segment and give it fewer threads, or keep the threads and give the block a
          longer segment. Which one helps the inter-block step of section 11.9, and why?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Ít luồng hơn mỗi khối: khối nhỏ hơn xong sớm hơn, nên tổng của chúng đến sớm hơn.">
            Fewer threads per block: smaller blocks finish sooner, so their sums arrive earlier.
          </Translated>
        ),
        why: (
          <Translated vi="Số khối, và do đó số tổng của khối cần ghép lại, vẫn giữ nguyên.">
            The number of blocks, and so of block sums to stitch together, stays the same.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Đoạn dài hơn: nó chia số khối cho hệ số.">The longer segment: it divides the number of blocks by the factor.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Không cách nào: bước liên khối chỉ phụ thuộc vào N.">Neither: the inter-block step depends only on N.</Translated>,
        why: <Translated vi="Nó phụ thuộc vào việc N được chia thành bao nhiêu khối.">It depends on how many blocks N is split into.</Translated>,
      },
      {
        label: (
          <Translated vi="Cả hai như nhau: đằng nào mỗi luồng cũng làm F phần tử, nên lưới làm cùng khối lượng việc.">
            Both equally: either way each thread does F elements, so the grid does the same work.
          </Translated>
        ),
        why: (
          <Translated vi="Tính theo luồng thì đúng. Tính theo lưới, chỉ một trong hai cách làm thay đổi số khối.">
            Per thread, yes. Per grid, only one of them changes how many blocks there are.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mục 11.9 scan các tổng của khối, mỗi khối một tổng, và phải truyền chúng giữa các khối. Giữ BLOCK_DIM và nhân đoạn
            với hệ số làm số khối giảm đi đúng hệ số đó, nên ít tổng hơn và ít lần chuyển giao hơn.
          </p>
        }
      >
        <p>
          Section 11.9 scans the blocks' sums, one per block, and has to pass them between blocks. Keeping BLOCK_DIM and
          multiplying the segment by the factor leaves a factor fewer blocks, so fewer sums and fewer hand-offs.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.6, p. 270',
  },
};

// ---------------------------------------------------------------- the section
export function S116() {
  return (
    <>
      <Translated
        vi={
          <p>
            <a href={href('ch11', '11.5')}>Mục 11.5</a> đã đếm cái giá của Kogge-Stone: khoảng N·log2N phép cộng trong khi vòng
            lặp cần N − 1. Với đủ đơn vị thực thi, phần việc thêm mua được một span ngắn hơn. GPU hiếm khi có nhiều đến thế: giao
            cho một khối nhiều luồng hơn số SM của nó chạy được cùng lúc thì phần cứng đằng nào cũng chạy chúng lần lượt. Khi đó
            phần việc thêm của thuật toán song song bị trả giá mà không được gì. Tốt hơn là bạn tự quyết phần nào chạy nối tiếp,
            và cho phần đó là vòng lặp hiệu quả công việc.
          </p>
        }
      >
        <p>
          <a href={href('ch11', '11.5')}>Section 11.5</a> counted the price of Kogge-Stone: about N·log2N additions where the
          loop needs N − 1. With enough execution units the extra work buys a shorter span. A GPU rarely has that many: give a
          block more threads than its SM can run at once and the hardware runs them in turns anyway. Then the parallel
          algorithm's extra work is paid for nothing. Better to decide yourself which part runs one after another, and make
          that part the work-efficient loop.
        </p>
      </Translated>
      <Source section="11.6" pages="267" />

      <h2><Translated vi="Ba pha">Three phases</Translated></h2>
      <Translated
        vi={
          <p>
            Giao cho mỗi khối một đoạn dài gấp vài lần số luồng của nó, gấp đúng <b>hệ số gộp việc</b> (coarsening factor), và cắt
            đoạn thành mỗi luồng một dãy liền nhau. Pha 1: mọi luồng scan dãy của mình bằng vòng lặp thường, không rào chắn, không
            dùng chung. Pha 2: giá trị cuối của mỗi luồng là tổng của dãy; các tổng đi qua một lần scan khối, tức{' '}
            <code>blockScan</code> của <a href={href('ch11', '11.4')}>mục 11.4</a>. Pha 3: mọi luồng cộng tổng đã scan của luồng
            trước nó vào tất cả giá trị của mình. Đó lại là scan-scan-cộng, xuống một tầng: một đoạn giờ là dãy của một luồng.
          </p>
        }
      >
        <p>
          Give each block a segment several times longer than its thread count, the <b>coarsening factor</b>, and cut the
          segment into one contiguous run per thread. Phase 1: every thread scans its own run with the plain loop, no barrier,
          no sharing. Phase 2: each thread's last value is its run's total; the totals go through one block scan, the{' '}
          <code>blockScan</code> of <a href={href('ch11', '11.4')}>section 11.4</a>. Phase 3: every thread adds the scanned total
          of the thread before it to all of its values. It is scan-scan-add once more, one level down: a segment is now a
          thread's run.
        </p>
      </Translated>
      <Source pages="267–268" figs="Fig. 11.11" />
      <Translated
        vi={
          <p>
            Bắt đầu với các số của Fig. 11.6 trên 4 luồng, mỗi luồng 4 phần tử: ba pha cho ra đúng các hàng của hình đó. Rồi dự
            đoán, trước khi kéo thanh trượt, số phép cộng mỗi phần tử thay đổi thế nào khi hệ số tăng.
          </p>
        }
      >
        <p>
          Start with Fig. 11.6's numbers on 4 threads of 4 elements: the three phases produce exactly the rows of that figure.
          Then predict, before you move a slider, what happens to the additions per element as the factor grows.
        </p>
      </Translated>
      <ThreePhasesExplorer />
      <Translated
        vi={
          <p>
            Hoạt hình theo một khối 4 luồng qua toàn bộ kernel, từng tầng bộ nhớ một, ở dạng dùng thanh ghi của mục kế tiếp.
          </p>
        }
      >
        <p>
          The animation follows one block of 4 threads through the whole kernel, memory tier by memory tier, in the register
          form of the next section.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Hình vẽ được tính từ cùng mô hình với các trình khám phá. Warp của nó là cả khối 4 luồng, và một truy cập được hợp
              nhất được vẽ thành một dãy byte liên tiếp. Không có thời gian.
            </p>
          }
        >
          <p>
            The drawing is computed from the same model as the explorers. Its warp is the whole block of 4 threads, and a
            coalesced access is drawn as one run of consecutive bytes. No timing.
          </p>
        </Translated>
      </Simplification>
      <CoarsePhasesScene />

      <h2><Translated vi="Kernel">The kernel</Translated></h2>
      <Translated
        vi={
          <p>
            Fig. 11.12 đặt hai hằng số, <code>COARSE_FACTOR</code> và <code>BLOCK_DIM</code>; một khối sở hữu{' '}
            <code>COARSE_FACTOR*BLOCK_DIM</code> phần tử bắt đầu từ <code>blockSegment</code> (dòng 02). Nó nạp cả đoạn vào{' '}
            <code>buffer_s</code> (dòng 04 đến 09), scan tại chỗ dãy của từng luồng (dòng 11 đến 14), truyền giá trị cuối của dãy
            cho <code>blockScan</code> (dòng 16 và 17), chia sẻ các tổng đã scan qua <code>threadSums</code> (dòng 19 đến 21),
            cộng (dòng 23 đến 28) và ghi (dòng 31 đến 34). Ba rào chắn của riêng nó, cộng hai rào chắn bên trong{' '}
            <code>blockScan</code>.
          </p>
        }
      >
        <p>
          Fig. 11.12 sets two constants, <code>COARSE_FACTOR</code> and <code>BLOCK_DIM</code>; a block owns{' '}
          <code>COARSE_FACTOR*BLOCK_DIM</code> elements starting at <code>blockSegment</code> (line 02). It loads the whole
          segment into <code>buffer_s</code> (lines 04 to 09), scans each thread's run in place (lines 11 to 14), passes the run's
          last value to <code>blockScan</code> (lines 16 and 17), shares the scanned totals through <code>threadSums</code>{' '}
          (lines 19 to 21), adds (lines 23 to 28) and stores (lines 31 to 34). Three barriers of its own, plus the two inside{' '}
          <code>blockScan</code>.
        </p>
      </Translated>
      <Source pages="268–269" figs="Fig. 11.12" />
      <CoarsenedListing />
      <Translated
        vi={
          <p>
            Có hai sơ suất trong văn bản và mã của sách. Ở p. 269, các tổng được truyền "to the <code>blockSum</code> device
            function from Fig. 11.9"; hàm đó thực ra là <code>blockScan</code>. Và kernel nhận <code>N</code> nhưng không hề đọc
            nó và không kiểm tra chỉ số nào, nên như bản in nó đọc và ghi vượt quá cuối mọi mảng có độ dài không phải bội số của
            đoạn khối. Phiên bản của lab 11.2 bảo vệ mọi lần nạp (phần tử đơn vị, 0, khi vượt cuối) và mọi lần ghi.
          </p>
        }
      >
        <p>
          Two slips in the book's text and listing. On p. 269 the totals are passed "to the <code>blockSum</code> device
          function from Fig. 11.9"; that function is <code>blockScan</code>. And the kernel takes <code>N</code> but never reads
          it and checks no index, so as printed it reads and writes past the end of any array whose length is not a multiple of
          the block segment. Lab 11.2's version guards every load (the identity, 0, past the end) and every store.
        </p>
      </Translated>
      <Source pages="269" />

      <h2><Translated vi="Nạp mà không mất hợp nhất truy cập">Loading without losing coalescing</Translated></h2>
      <Translated
        vi={
          <p>
            Mỗi luồng sở hữu một dãy liền nhau, nên cách nạp hiển nhiên là: luồng <i>t</i> lấy các phần tử của chính nó,{' '}
            <i>t</i>·F + <i>c</i> ở lần lặp <i>c</i>. Ở hệ số 4, luồng 0 đọc phần tử 0, luồng 1 phần tử 4, luồng 2 phần tử 8, tất
            cả trong cùng một lệnh: các địa chỉ của warp cách nhau 16 byte thay vì 4, và truy cập không còn được hợp nhất (
            <a href={href('ch06', '6.1')}>mục 6.1</a>). Fig. 11.12 thay vào đó cắt đoạn thành <code>COARSE_FACTOR</code> khúc,
            mỗi khúc <code>BLOCK_DIM</code> phần tử, và nạp mỗi lần lặp một khúc: luồng <i>t</i> lấy phần tử <i>t</i> của khúc,
            và warp đọc một dãy liền nhau. Lần ghi cũng làm như vậy.
          </p>
        }
      >
        <p>
          Each thread owns a contiguous run, so the obvious load is: thread <i>t</i> fetches its own elements,{' '}
          <i>t</i>·F + <i>c</i> in iteration <i>c</i>. At factor 4, thread 0 reads element 0, thread 1 element 4, thread 2
          element 8, all in the same instruction: the warp's addresses are 16 bytes apart instead of 4, and the access is no
          longer coalesced (<a href={href('ch06', '6.1')}>section 6.1</a>). Fig. 11.12 instead cuts the segment into{' '}
          <code>COARSE_FACTOR</code> chunks of <code>BLOCK_DIM</code> elements and loads one chunk per iteration: thread{' '}
          <i>t</i> takes element <i>t</i> of the chunk, and the warp reads one contiguous run. The store does the same.
        </p>
      </Translated>
      <Source pages="269" />
      <Translated
        vi={
          <p>
            Vậy kernel dùng hai cách ánh xạ chỉ số. Chunk-major, <code>c*BLOCK_DIM + threadIdx.x</code>, cho mọi lần chuyển giữa
            bộ nhớ toàn cục và bộ nhớ chia sẻ. Thread-major, <code>threadIdx.x*COARSE_FACTOR + c</code>, cho phép scan. Bộ nhớ
            chia sẻ là nơi dữ liệu chuyển từ cách này sang cách kia, và rào chắn sau lần nạp là thứ cho phép một luồng đọc các
            phần tử mà luồng khác đã nạp.
          </p>
        }
      >
        <p>
          So the kernel uses two index mappings. Chunk-major, <code>c*BLOCK_DIM + threadIdx.x</code>, for every transfer between
          global and shared memory. Thread-major, <code>threadIdx.x*COARSE_FACTOR + c</code>, for the scan. Shared memory is
          where the data turns from one to the other, and the barrier after the load is what lets a thread read elements other
          threads loaded.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi lần một lệnh nạp, đếm theo đơn vị của nhà cung cấp được chọn (coalesce.ts, mô hình của chương 6). Không có bộ nhớ
              đệm: một segment đã được nạp ở một lần lặp sẽ được đếm lại nếu lần lặp sau chạm tới nó, dù cache thật có thể phục vụ
              nó.
            </p>
          }
        >
          <p>
            One load instruction at a time, counted in the chosen vendor's unit (coalesce.ts, chapter 6's model). No caches: a
            segment fetched by one iteration is counted again if the next one touches it, though a real cache may serve it.
          </p>
        </Translated>
      </Simplification>
      <Ask q={questionsS116['ch11/11.6/q-sectors']} label={tr('Work it out first', 'Tự tính trước')} />
      <LoadExplorer />
      <Ask q={questionsS116['ch11/11.6/q-owner']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Mô hình đếm số sector mỗi lệnh gấp bốn lần cho kiểu nạp thẳng ở hệ số 4. Nhưng các lần lặp 1, 2 và 3 của kiểu nạp
            thẳng chạm đúng những sector mà lần lặp 0 đã chạm, và chương 5 đã cảnh báo rằng bộ nhớ đệm có thể che một phần những
            gì mô hình lưu lượng dự đoán. Lab 11.2 cung cấp kiểu nạp thẳng bên cạnh kiểu nạp qua bộ nhớ chia sẻ của sách, để bạn
            đo xem thiết bị của mình xử lý nó ra sao.
          </p>
        }
      >
        <p>
          The model counts four times as many sectors per instruction for the direct load at factor 4. But the direct load's
          iterations 1, 2 and 3 touch the very sectors iteration 0 touched, and chapter 5 warned that a cache can hide part of
          what a model of traffic predicts. Lab 11.2 provides the direct load beside the book's staged one, so you can measure what
          your device makes of it.
        </p>
      </Translated>
      <Predict
        id="ch11/11.6/p-staging"
        prompt={
          <Translated
            vi={
              <>
                Lab 11.2 đo thời gian kernel phân ô thanh ghi của nó, nạp qua bộ nhớ chia sẻ, so với một bản sao có sẵn nạp mỗi dãy
                thẳng từ bộ nhớ toàn cục. Với khối 256 luồng ở hệ số 4, kernel nạp qua bộ nhớ chia sẻ sẽ nhanh hơn bao nhiêu lần trên
                thiết bị <em>của bạn</em>? Chốt một con số trước khi chạy benchmark.
              </>
            }
          >
            <>
              Lab 11.2 times its register-tiled kernel, staged through shared memory, against a provided copy that loads each
              run straight from global memory. With 256-thread blocks at factor 4, how many times faster will the staged one be
              on <em>your</em> device? Commit a number before running the benchmark.
            </>
          </Translated>
        }
        unit={tr('× faster', '× nhanh hơn')}
      >
        {(v) => <MeasuredStaging predicted={v} />}
      </Predict>

      <h2><Translated vi="Gộp việc tiết kiệm được gì">What coarsening saves</Translated></h2>
      <Translated
        vi={
          <p>
            Đếm phép cộng với N phần tử và P luồng. Các lần scan tuần tự trên dãy làm P·(N/P − 1) = N − P. Lần scan khối trên P
            tổng làm P·log2P. Mọi luồng trừ luồng đầu cộng một tổng vào N/P giá trị của mình: N − N/P. Tổng cộng 2N + P·log2P − P
            − N/P. Khi P gần N, đây là O(N·log2N), lại là Kogge-Stone. Khi P nhỏ hơn N nhiều, nó gần 2N: mỗi phần tử được cộng
            một lần trong dãy của nó và thêm một lần ở pha 3, tức O(N), cùng bậc với vòng lặp.
          </p>
        }
      >
        <p>
          Count additions with N elements and P threads. The sequential run scans do P·(N/P − 1) = N − P. The block scan of P
          totals does P·log2P. Every thread but the first adds a total to its N/P values: N − N/P. In all, 2N + P·log2P − P −
          N/P. When P is close to N this is O(N·log2N), Kogge-Stone again. When P is much smaller than N it is close to 2N: each
          element is added once in its run and once more in phase 3, which is O(N), the order of the loop.
        </p>
      </Translated>
      <Source pages="269–270" />
      <Translated
        vi={
          <p>
            Số bước kể cùng câu chuyện. Scan trên dãy mất N/P − 1, scan khối log2P, các phép cộng cuối N/P: 2N/P + log2P − 1, so
            với khoảng (N·log2N)/P của Kogge-Stone trên cùng P đơn vị. Khi P gần N, cả hai đều là O(log2N); khi P nhỏ hơn N
            nhiều, gộp việc thắng với hệ số gần log2N / 2.
          </p>
        }
      >
        <p>
          Steps tell the same story. The run scan takes N/P − 1, the block scan log2P, the adds N/P: 2N/P + log2P − 1, against
          about (N·log2N)/P for Kogge-Stone on the same P units. For P close to N both are O(log2N); for P much smaller than N,
          coarsening wins by a factor of nearly log2N / 2.
        </p>
      </Translated>
      <Source pages="270" />
      <Ask q={questionsS116['ch11/11.6/q-work']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS116['ch11/11.6/q-steps']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Những lợi ích khác">Other things it buys</Translated></h2>
      <Translated
        vi={
          <p>
            Khối lượng việc cũng chuyển sang phần mã rẻ hơn. Scan trên dãy của một luồng không có rào chắn và không phân kỳ; scan
            khối, vốn có cả hai, giờ chỉ xử lý mỗi luồng một giá trị thay vì F. Đó là cùng sự đánh đổi như phép rút gọn gộp việc
            của chương 10 (<a href={href('ch10', '10.10')}>mục 10.10</a>). Và có hai cách gộp việc: giữ đoạn và dùng ít luồng
            hơn, hoặc giữ số luồng và kéo dài đoạn. Cách thứ hai chia số khối cho hệ số, đúng điều mà bước liên khối của{' '}
            <a href={href('ch11', '11.9')}>mục 11.9</a> muốn.
          </p>
        }
      >
        <p>
          The work also moves to cheaper code. A thread's run scan has no barrier and no divergence; the block scan, which has
          both, now handles one value per thread instead of F. It is the same trade as chapter 10's coarsened reduction (
          <a href={href('ch10', '10.10')}>section 10.10</a>). And there are two ways to coarsen: keep the segment and use fewer
          threads, or keep the threads and lengthen the segment. The second divides the number of blocks by the factor, which
          is what the inter-block step of <a href={href('ch11', '11.9')}>section 11.9</a> wants.
        </p>
      </Translated>
      <NoteGroup>
        <Source pages="270" />
        <Today checked={tr('2026-11-20 (ch11-facts A4)', '2026-11-20 (ch11-facts A4)')}>
          <Translated
            vi={
              <p>
                <code>BLOCK_SCAN_WARP_SCANS</code> của CUB chính là thiết kế này: "Upsweep sequential reduction in registers" (rút gọn
                tuần tự trong thanh ghi), một scan Kogge-Stone trong warp, một pha lan truyền, "Downsweep sequential scan in registers"
                (scan tuần tự trong thanh ghi). NVIDIA ghi chú nó "can often provide lower turnaround latencies when the GPU is
                under-occupied" (thường cho độ trễ hoàn tất thấp hơn khi GPU không được lấp đầy).
              </p>
            }
          >
            <p>
              CUB's <code>BLOCK_SCAN_WARP_SCANS</code> is this design: "Upsweep sequential reduction in registers", a warp
              Kogge-Stone scan, a propagation phase, "Downsweep sequential scan in registers". NVIDIA notes it "can often
              provide lower turnaround latencies when the GPU is under-occupied".
            </p>
          </Translated>
        </Today>
      </NoteGroup>
      <Ask q={questionsS116['ch11/11.6/q-fewer-blocks']} />

      <h2><Translated vi="Nơi bạn gặp điều này: các lần nạp trong cumsum của llama.cpp">Where you meet this: llama.cpp's cumsum loads</Translated></h2>
      <Translated
        vi={
          <p>
            Tổng tích lũy CUDA của llama.cpp gộp việc theo hệ số 4: mỗi luồng scan <code>num_unroll = 4</code> phần tử trong thanh
            ghi trước lần scan trong warp. Nhưng nó nạp chúng liền nhau theo luồng, <code>idx = i + tid * num_unroll</code>, đúng
            mẫu mà p. 269 gọi là không được hợp nhất; sách thì sẽ nạp qua bộ nhớ chia sẻ. Đó là cách chúng tôi đọc mã, không phải
            một phép đo, và trình khám phá ở trên đếm chi phí mỗi lệnh khi không có cache. Lab 11.2 chạy cả hai trên thiết bị của
            bạn: cache có che được khác biệt ở đó hay không là câu hỏi cho benchmark, không phải cho mô hình.
          </p>
        }
      >
        <p>
          llama.cpp's CUDA cumulative sum coarsens by 4: each thread scans <code>num_unroll = 4</code> elements in registers
          before the warp scan. But it loads them thread-contiguously, <code>idx = i + tid * num_unroll</code>, which is exactly
          the pattern p. 269 calls uncoalesced; the book would stage through shared memory. That is our reading of the code, not
          a measurement, and the explorer above counts what it costs per instruction with no cache. Lab 11.2 runs both on your
          device: whether the cache hides the difference there is a question for the benchmark, not for the model.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              <code>ggml-cuda/cumsum.cu</code> của llama.cpp, đọc ngày 2026-11-20 (ch11-facts D1).
            </p>
          }
        >
          <p>
            llama.cpp's <code>ggml-cuda/cumsum.cu</code> as read on 2026-11-20 (ch11-facts D1).
          </p>
        </Translated>
      </Beyond>
    </>
  );
}
