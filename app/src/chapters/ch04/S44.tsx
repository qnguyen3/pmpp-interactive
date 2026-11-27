import { useEffect, useId, useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, num, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { STRAIGHT_LINE, finished, startWarp, step, type WarpRun } from '../../sim/divergence';
import {
  BOOK_MAX_THREADS,
  coordOf,
  cut,
  dims,
  laneOf,
  linearWorking,
  place,
  threadsIn,
  warpOf,
  widthChoices,
  type Dims,
} from '../../sim/warps';
import { Translated, tr } from '../../lib/i18n';

// ---------------------------------------------------------------- drawing a block and its warps
// Warps are told apart by four shades of ink, never by hue: colour in this course
// means a memory tier or a failure.
const SHADES = [9, 30, 54, 80];
const shade = (warp: number) => `color-mix(in srgb, var(--ink) ${SHADES[warp % 4]}%, var(--cell))`; // i18n-ignore: CSS value
const inkOn = (warp: number) => (SHADES[warp % 4] > 45 ? 'var(--page)' : 'var(--ink)');
const MAX_SLICES = 16;

/** The block as the kernel indexes it: x across, y down, one panel per z. */
function BlockView({
  d, width, pick, upTo, onPick,
}: {
  d: Dims;
  width: number;
  /** Linear index of the marked thread. */
  pick: number | null;
  /** When set, only threads with a linear index below it are painted. */
  upTo?: number;
  onPick?: (index: number) => void;
}) {
  const c = Math.max(6, Math.min(26, Math.floor(Math.min(352 / d.x, 352 / d.y))));
  const slices = Math.min(d.z, MAX_SLICES);
  const pad = 14;
  return (
    <div className="row" style={{ alignItems: 'flex-start', gap: '0.9rem 1.1rem' }} data-view="block">
      {Array.from({ length: slices }, (_, z) => (
        <svg key={z} viewBox={`0 0 ${d.x * c + pad + 2} ${d.y * c + pad + 2}`} role="img" aria-label={tr(`Threads of the block with z = ${z}, shaded by warp`, `Các luồng của khối với z = ${z}, tô theo warp`)}
          style={{ width: d.x * c + pad + 2, maxWidth: '100%', flex: 'none' }}>
          <text x={pad} y={10} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{d.z > 1 ? `z = ${z}:  x →` : 'x →'}</text>
          <text x={0} y={pad + 10} fontSize="10" style={{ fill: 'var(--ink-2)' }}>y</text>
          {Array.from({ length: d.x * d.y }, (_, k) => {
            const x = k % d.x;
            const y = Math.floor(k / d.x);
            const p = place(x, y, z, d, width);
            const painted = upTo == null || p.index < upTo;
            const marked = pick === p.index;
            return (
              <g key={k}>
                <rect x={pad + x * c} y={pad + y * c} width={c - 1} height={c - 1} rx="1.5"
                  data-thread={`${x},${y},${z}`} data-index={p.index} data-warp={p.warp} data-lane={p.lane} data-painted={painted ? '' : undefined}
                  fill={marked ? 'var(--page)' : painted ? shade(p.warp) : 'var(--cell)'}
                  stroke={marked ? 'var(--ink)' : 'var(--rule)'} strokeWidth={marked ? 2.5 : 0.6}
                  onMouseEnter={onPick ? () => onPick(p.index) : undefined} onClick={onPick ? () => onPick(p.index) : undefined}
                  style={onPick ? { cursor: 'pointer' } : undefined} />
                {c >= 18 && (painted || marked) ? (
                  <text x={pad + x * c + c / 2 - 0.5} y={pad + y * c + c / 2 + 3.5} fontSize="10" textAnchor="middle" pointerEvents="none" /* i18n-ignore: SVG attribute */
                    style={{ fill: marked ? 'var(--ink)' : inkOn(p.warp) }}>{p.warp}</text>
                ) : null}
              </g>
            );
          })}
        </svg>
      ))}
      {d.z > slices ? <span className="muted"><Translated vi={<>và thêm {d.z - slices} giá trị z nữa, được cắt theo cùng cách</>}>and {d.z - slices} more values of z, cut the same way</Translated></span> : null}
    </div>
  );
}

/** The same threads as the hardware groups them: one row per warp, one cell per lane. */
function WarpRows({ d, width, pick, upTo, onPick }: { d: Dims; width: number; pick: number | null; upTo?: number; onPick?: (index: number) => void }) {
  const hatch = useId();
  const k = cut(d, width);
  const c = width <= 32 ? 13 : 9;
  const left = 56;
  const top = 14;
  const rh = 15;
  const W = left + width * c + 4;
  return (
    <svg viewBox={`0 0 ${W} ${top + k.warps * rh + 4}`} role="img" aria-label={tr('The warps of the block, one row each, one cell per lane', 'Các warp của khối, mỗi warp một hàng, mỗi lane một ô')} data-view="warps"
      style={{ width: W, maxWidth: '100%', display: 'block' }}>
      <defs>
        <pattern id={hatch} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" strokeWidth="1.2" />
        </pattern>
      </defs>
      <text x={left} y={10} fontSize="10" style={{ fill: 'var(--ink-2)' }}><Translated vi="lane 0">lane 0</Translated></text>
      <text x={left + width * c - 2} y={10} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>lane {width - 1}</>}>lane {width - 1}</Translated></text>
      {Array.from({ length: k.warps }, (_, w) => (
        <g key={w}>
          <text x={left - 6} y={top + w * rh + 10.5} fontSize="10.5" textAnchor="end"><Translated vi={<>warp {w}</>}>warp {w}</Translated></text>
          {Array.from({ length: width }, (_, lane) => {
            const index = w * width + lane;
            const real = index < k.threads;
            const painted = real && (upTo == null || index < upTo);
            const marked = pick === index;
            return (
              <g key={lane}>
                <rect x={left + lane * c} y={top + w * rh} width={c - 1} height={rh - 2} rx="1.5"
                  data-index={real ? index : undefined} data-pad={real ? undefined : ''}
                  fill={marked ? 'var(--page)' : painted ? shade(w) : 'var(--cell)'} stroke={marked ? 'var(--ink)' : 'var(--rule)'} strokeWidth={marked ? 2.5 : 0.6}
                  onMouseEnter={real && onPick ? () => onPick(index) : undefined} onClick={real && onPick ? () => onPick(index) : undefined}
                  style={real && onPick ? { cursor: 'pointer' } : undefined} />
                {real ? null : <rect x={left + lane * c} y={top + w * rh} width={c - 1} height={rh - 2} rx="1.5" fill={`url(#${hatch})`} pointerEvents="none" />}{/* i18n-ignore: SVG attribute */}
              </g>
            );
          })}
        </g>
      ))}
    </svg>
  );
}

/** One sentence of arithmetic for a thread of the block. */
function describeThread(index: number, d: Dims, width: number): string {
  const t = coordOf(index, d);
  const where = d.z > 1 ? `(x ${t.x}, y ${t.y}, z ${t.z})` : d.y > 1 ? `(x ${t.x}, y ${t.y})` : `x ${t.x}`;
  const work = linearWorking(t.x, t.y, t.z, d);
  return tr(`Thread ${where}: linear index ${work}. Warp ${index} / ${width} rounded down = ${warpOf(index, width)}; lane ${index} mod ${width} = ${laneOf(index, width)}.`, `Luồng ${where}: chỉ số tuyến tính ${work}. Warp ${index} / ${width} làm tròn xuống = ${warpOf(index, width)}; lane ${index} mod ${width} = ${laneOf(index, width)}.`);
}

// ---------------------------------------------------------------- lab 4.1's Metal probe
interface ProbeRow {
  /** The threadgroup shape the probe kernel was dispatched with. */
  tx: number;
  ty: number;
  /** [[threads_per_simdgroup]] as the kernel saw it. */
  width: number;
  groups: number | undefined;
  /** Whether every SIMD-group was a consecutive run of the linear thread index. */
  consecutive: boolean | null;
}
interface Probe {
  device: string;
  whose: string;
  rows: ProbeRow[];
  /** The one width every row reported, or null when they differ. */
  width: number | null;
  consecutive: boolean | null;
  /** maxTotalThreadsPerThreadgroup of the probe's pipeline, if the lab recorded it. */
  cap: number | null;
}
const LAB1 = 'ch04/lab1_launchconfig';
/** The "probe" rows lab 4.1's driver writes after running the probe kernel on a Metal device, if it has run. */
function metalProbe(reports: Reports): Probe | null {
  const run = reports['metal'];
  if (!run) return null;
  for (const variant of ['work', 'solution']) {
    const all = benchRows(run, LAB1, variant);
    const rows = all
      .filter((r) => r.name === 'probe')
      .flatMap((r): ProbeRow[] => {
        const width = num(r.width_in_kernel);
        const c: unknown = r.consecutive_runs;
        return width
          ? [{ tx: num(r.threadgroup_x) ?? 0, ty: num(r.threadgroup_y) ?? 0, width, groups: num(r.simdgroups_per_threadgroup), consecutive: typeof c === 'boolean' ? c : null }]
          : [];
      });
    if (rows.length)
      return {
        get device() { return String(run.device.name ?? tr('your Metal device', 'thiết bị Metal của bạn')); },
        get whose() { return variant === 'work' ? tr('your probe kernel', 'kernel thăm dò của bạn') : tr("the reference solution's probe kernel", 'kernel thăm dò của lời giải tham chiếu'); },
        rows,
        width: rows.every((r) => r.width === rows[0].width) ? rows[0].width : null,
        consecutive: rows.every((r) => r.consecutive === true) ? true : rows.some((r) => r.consecutive === false) ? false : null,
        cap: num(all.find((r) => r.name === 'queried')?.probe_maxTotalThreadsPerThreadgroup) ?? null,
      };
  }
  return null;
}

const PRESETS: { label: string; d: Dims }[] = [
  { label: '4 × 4', d: dims(4, 4) },
  { label: '8 × 8', d: dims(8, 8) },
  { label: '16 × 16', d: dims(16, 16) },
  { label: '4 × 8 × 2', d: dims(4, 8, 2) },
  { label: '48 × 1', d: dims(48) },
];
const AXIS = ['x', 'y', 'z'] as const;
const count = (n: number, one: string, many = one + 's') => `${n} ${n === 1 ? one : many}`;

function WarpLab() {
  const { reports } = useReports();
  const choices = useMemo(() => widthChoices(reports), [reports]);
  const probe = useMemo(() => metalProbe(reports), [reports]);
  // Opens on 16 x 16, not on the 8 x 8 of the prediction above it, so the drawing does not answer that prediction.
  const [d, setD] = useState<Dims>(dims(16, 16));
  const [key, setKey] = useState('book');
  const [pick, setPick] = useState<number | null>(null);
  const [n, setN] = useState(0);
  const [playing, setPlaying] = useState(false);

  const metal = key === 'metal';
  const chosen = choices.find((c) => c.key === key) ?? choices[0];
  const width = metal ? (probe && probe.consecutive === true ? probe.width : null) : chosen.width;
  // On Metal the cap belongs to a pipeline: the probe's, if lab 4.1 recorded it, else the book's example.
  const cap = metal ? (probe?.cap ?? BOOK_MAX_THREADS) : chosen.maxThreads;
  const threads = threadsIn(d);
  const valid = AXIS.every((a) => Number.isInteger(d[a]) && d[a] >= 1);
  const over = valid && threads > cap;
  const ok = valid && !over && width != null;
  const k = ok ? cut(d, width) : null;
  const at = Math.min(n, threads);

  useEffect(() => {
    setPick(null);
    setN(0);
    setPlaying(false);
  }, [d, key]);
  useEffect(() => {
    if (!playing) return;
    if (!ok || at >= threads) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setN((v) => v + 1), threads > 128 ? 40 : 160);
    return () => window.clearTimeout(id);
  }, [playing, at, threads, ok]);

  const setAxis = (a: (typeof AXIS)[number], v: string) => setD({ ...d, [a]: v.trim() === '' ? 0 : Math.floor(Number(v)) });
  const cur = at > 0 ? at - 1 : null;
  const curT = cur != null ? coordOf(cur, d) : null;
  const prevT = cur != null && cur > 0 ? coordOf(cur - 1, d) : null;

  return (
    <>
      <Bench
        title={tr('Warp painter', 'Trình tô warp')}
        task={tr('Set a block shape and a width. Which threads end up in the same warp?', 'Đặt hình dạng khối và độ rộng. Những luồng nào rơi vào cùng một warp?')}
        foot={
          <>
            <Translated
              vi={
                <>
                  Mỗi sắc độ là một warp; con số trong một ô là warp của nó. Ô gạch chéo là lane không có luồng. Trỏ vào một
                  luồng, ở hình nào cũng được, để xem phép tính của nó.{' '}
                </>
              }
            >
              Each shade is one warp; a number in a cell is its warp. Hatched cells are lanes with no thread. Point at a thread, in either
              drawing, to see its arithmetic.{' '}
            </Translated>
            {!metal
              ? chosen.note
              : probe
                ? tr(`On Metal a threadgroup is capped at ${cap} here: ${probe.cap != null ? "the maxTotalThreadsPerThreadgroup of the probe's pipeline, as lab 4.1 recorded it" : "the book's example, because lab 4.1 recorded no pipeline limit"}.`, `Trên Metal, một threadgroup ở đây bị giới hạn ở ${cap}: ${probe.cap != null ? 'maxTotalThreadsPerThreadgroup của pipeline thăm dò, như lab 4.1 đã ghi lại' : 'ví dụ của sách, vì lab 4.1 không ghi lại giới hạn pipeline nào'}.`)
                : ''}
          </>
        }
      >
        <div className="row" style={{ marginBottom: '0.7rem' }}>
          {AXIS.map((a) => (
            <label className="field" key={a}>{/* i18n-ignore: API name */}
              blockDim.{a}
              <input type="number" min={1} step={1} value={d[a] || ''} aria-label={`blockDim.${a}`} style={{ width: '5.2rem' }} onChange={(e) => setAxis(a, e.target.value)} />{/* i18n-ignore: API name */}
            </label>
          ))}
          <label className="field" style={{ maxWidth: '100%', minWidth: 0 }}>
            <Translated vi="Độ rộng warp">Warp width</Translated>
            <select value={key} aria-label={tr('Warp width', 'Độ rộng warp')} onChange={(e) => setKey(e.target.value)}>
              {choices.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
              <option value="metal">{tr("Metal: what lab 4.1's probe recorded", 'Metal: những gì kernel thăm dò của lab 4.1 ghi lại')}</option>
            </select>
          </label>
        </div>
        <div className="row" style={{ marginBottom: '0.9rem' }}>
          <span className="muted"><Translated vi="Cấu hình mẫu">Presets</Translated></span>
          <span className="seg" role="group" aria-label={tr('Presets', 'Cấu hình mẫu')} style={{ flexWrap: 'wrap' }}>
            {PRESETS.map((p) => (
              <button key={p.label} aria-pressed={p.d.x === d.x && p.d.y === d.y && p.d.z === d.z} onClick={() => setD(p.d)} style={{ whiteSpace: 'nowrap' }}>{p.label}</button>
            ))}
          </span>
        </div>

        {metal ? (
          <div className="narrate" data-metal={probe ? 'probed' : 'none'} style={{ marginBottom: '0.9rem' }}>
            {probe ? (
              <>
                <Translated
                  vi={
                    <p style={{ margin: '0 0 0.5em' }}>
                      Trên <b>{probe.device}</b>, {probe.whose} trong lab 4.1 đã ghi lại điều này, mỗi dòng cho một hình dạng threadgroup mà nó được dispatch:
                    </p>
                  }
                >
                  <p style={{ margin: '0 0 0.5em' }}>
                    On <b>{probe.device}</b>, {probe.whose} in lab 4.1 recorded this, one line per threadgroup shape it was dispatched with:
                  </p>
                </Translated>
                <div style={{ overflowX: 'auto', marginBottom: '0.5em' }}>
                  <table className="threads" style={{ width: 'auto' }}>
                    <thead>
                      <tr><th>threadgroup</th><th><Translated vi="số luồng trong một SIMD-group">threads in a SIMD-group</Translated></th><th>SIMD-groups</th><th><Translated vi="các đoạn liên tiếp của chỉ số tuyến tính">consecutive runs of the linear index</Translated></th></tr>{/* i18n-ignore: Metal terms */}
                    </thead>
                    <tbody>
                      {probe.rows.map((r, i) => (
                        <tr key={i} data-probe-row="">
                          <td>{r.tx} × {r.ty}</td>
                          <td>{r.width}</td>
                          <td>{r.groups ?? tr('not recorded', 'không được ghi')}</td>
                          <td>{r.consecutive == null ? tr('not recorded', 'không được ghi') : r.consecutive ? tr('yes', 'có') : tr('no', 'không')}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {width != null
                  ? tr(`The drawing below applies consecutive runs of ${width} to your block${probe.rows.some((r) => r.tx === d.x && r.ty === d.y && d.z === 1) ? ', which is one of the shapes the probe ran' : ', a shape the probe did not run'}. It is what one pipeline did on one device in these dispatches, not a rule Apple documents.`, `Hình bên dưới áp các đoạn liên tiếp dài ${width} lên khối của bạn${probe.rows.some((r) => r.tx === d.x && r.ty === d.y && d.z === 1) ? ', một trong các hình dạng mà kernel thăm dò đã chạy' : ', một hình dạng mà kernel thăm dò chưa chạy'}. Đó là những gì một pipeline đã làm trên một thiết bị trong các lần dispatch này, không phải quy tắc Apple ghi trong tài liệu.`)
                  : tr('The groups were not all consecutive runs of one width, so no rule here can draw them. The table is the result.', 'Các nhóm không phải tất cả đều là đoạn liên tiếp của cùng một độ rộng, nên không quy tắc nào ở đây vẽ được chúng. Bảng chính là kết quả.')}
              </>
            ) : (
              <Translated
                vi={
                  <>
                    Không có gì để vẽ. Metal nói một threadgroup được cắt thành SIMD-group "in an implementation-defined fashion"
                    (theo cách do cài đặt quyết định) và không ghi độ rộng nào trong tài liệu, nên một hình vẽ dựa trên quy tắc sẽ là
                    bịa ra. Hãy viết kernel thăm dò của lab 4.1, rồi chạy{' '}
                    <code>python3 labs/pmpp.py run 4.1 --bench</code>. Khi lab qua các bài kiểm thử, lần chạy sẽ ghi lại những gì
                    kernel thăm dò thấy trên thiết bị của bạn, và lựa chọn này hiển thị nó.
                  </>
                }
              >
                <>
                  Nothing to draw. Metal says a threadgroup is cut into SIMD-groups "in an implementation-defined fashion" and documents no
                  width, so a drawing made from a rule would be invented. Write the probe kernel of lab 4.1, then run{' '}
                  <code>python3 labs/pmpp.py run 4.1 --bench</code>. Once the lab passes its tests, the run records what the probe saw on
                  your device, and this choice shows it.
                </>
              </Translated>
            )}
          </div>
        ) : null}

        {!valid ? (
          <p><span className="chip fail"><Translated vi="Mỗi chiều của một khối là một số nguyên, từ 1 trở lên.">Each dimension of a block is a whole number, 1 or more.</Translated></span></p>
        ) : over ? (
          <p data-over="">
            <span className="chip fail"><Translated vi={<>{threads} luồng trong một khối</>}>{threads} threads in a block</Translated></span>{' '}
            <Translated vi={<>là nhiều hơn mức {cap} mà lựa chọn này cho phép, nên lần khởi chạy sẽ bị từ chối. Không có gì được vẽ.</>}>
              is more than the {cap} this choice allows, so the launch would be refused. Nothing is drawn.
            </Translated>
          </p>
        ) : k && width != null ? (
          <>
            <div className="row" style={{ marginBottom: '0.8rem' }} data-readout="cut">
              <span><Translated vi={<><b data-out="threads">{k.threads}</b> luồng</>}><b data-out="threads">{k.threads}</b> threads</Translated></span>
              <span><Translated vi={<><b data-out="warps">{k.warps}</b> warp trong khối</>}><b data-out="warps">{k.warps}</b> {k.warps === 1 ? 'warp' : 'warps'} in the block</Translated></span>
              <span><Translated vi={<><b data-out="padded">{k.padded}</b> lane rỗi</>}><b data-out="padded">{k.padded}</b> idle {k.padded === 1 ? 'lane' : 'lanes'}</Translated></span>
              <span data-out="rows">
                {d.y * d.z === 1
                  ? tr('one row: a warp is a run of consecutive x', 'một hàng: một warp là một đoạn x liên tiếp')
                  : k.rowsPerWarp == null
                    ? tr('warp boundaries fall inside rows', 'ranh giới warp rơi vào giữa các hàng')
                    : k.rowsPerWarp >= 1
                      ? tr(`one warp covers ${count(k.rowsPerWarp, 'row')} of the block`, `một warp phủ ${k.rowsPerWarp} hàng của khối`)
                      : tr(`one row of the block is ${count(Math.round(d.x / width), 'warp')}`, `một hàng của khối là ${Math.round(d.x / width)} warp`)}
              </span>
            </div>
            <div className="narrate" role="status" data-readout="thread" style={{ marginBottom: '0.9rem', minHeight: '2.6em' }}>
              {pick != null && pick < k.threads ? describeThread(pick, d, width) : tr('Point at a thread.', 'Hãy trỏ vào một luồng.')}
            </div>
            <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
              <div>
                <div className="mx-label"><Translated vi="Khối theo cách kernel đánh chỉ số">The block as the kernel indexes it</Translated></div>
                {d.x <= 64 ? (
                  <BlockView d={d} width={width} pick={pick} onPick={setPick} />
                ) : (
                  <Translated vi={<p className="muted">Một hàng {d.x} luồng quá dài để vẽ ở đây. Với một hàng, thứ tự tuyến tính chính là threadIdx.x; hãy nhìn các warp.</p>}><p className="muted">A row of {d.x} threads is too long to draw here. With one row, the linear order is just threadIdx.x; look at the warps.</p></Translated>
                )}
              </div>
              <div>
                <div className="mx-label"><Translated vi="Cũng các luồng đó theo thứ tự tuyến tính, cắt thành warp">The same threads in linear order, cut into warps</Translated></div>
                <WarpRows d={d} width={width} pick={pick} onPick={setPick} />
              </div>
            </div>
          </>
        ) : null}
      </Bench>

      <Translated
        vi={
          <p>
            Trình tô cho thấy kết quả. Trình chạy từng bước bên dưới cho thấy đường đi tạo ra kết quả đó, với khối và độ rộng
            bạn đặt ở trên: nó đi qua các luồng theo thứ tự tuyến tính, mỗi lần một chỉ số. Hãy xem điều gì xảy ra với{' '}
            <code>y</code> mỗi khi <code>x</code> hết, và chỉ số đã dịch đi bao xa khi <code>y</code> hoặc <code>z</code> tăng
            một.
          </p>
        }
      >
        <p>
          The painter shows the result. The stepper below shows the walk that produces it, for the block and width you set
          above: it visits the threads in linear order, one index at a time. Watch what happens to <code>y</code> each time{' '}
          <code>x</code> runs out, and how far the index has moved when <code>y</code> or <code>z</code> goes up by one.
        </p>
      </Translated>
      <Bench
        title={tr('From index to lane', 'Từ chỉ số đến lane')}
        task={tr('Step the linear index. Before each warp fills, say which thread will be its last.', 'Tăng chỉ số tuyến tính từng bước. Trước khi mỗi warp đầy, hãy nói luồng nào sẽ là luồng cuối của nó.')}
        foot={tr('The block and the width are the ones set in the painter. The index goes from 0 to one less than the number of threads in the block.', 'Khối và độ rộng là những gì đã đặt trong trình tô. Chỉ số chạy từ 0 đến số luồng trong khối trừ một.')}
      >
        {ok && k && width != null ? (
          <>
            <div className="row" style={{ marginBottom: '0.8rem' }}>
              <button onClick={() => { setPlaying(false); setN(0); }} disabled={at === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
              <button onClick={() => { setPlaying(false); setN(at - 1); }} disabled={at === 0}><Translated vi="Lùi">Back</Translated></button>
              <button className="primary" onClick={() => { setPlaying(false); setN(at + 1); }} disabled={at >= threads}><Translated vi="Bước">Step</Translated></button>
              <button onClick={() => { setPlaying(false); setN(Math.min(threads, (Math.floor(at / width) + 1) * width)); }} disabled={at >= threads}><Translated vi="Lấp đầy warp này">Fill this warp</Translated></button>
              <button onClick={() => setPlaying(!playing)} disabled={at >= threads}>{playing ? tr('Pause', 'Tạm dừng') : tr('Play', 'Chạy')}</button>
              <span className="muted" aria-live="polite"><Translated vi={<>đã đặt {at} trên {threads} luồng</>}>{at} of {threads} threads placed</Translated></span>
            </div>
            <div className="narrate" role="status" data-readout="step" style={{ marginBottom: '0.9rem' }}>
              {cur == null || !curT ? (
                <Translated vi={<>Chưa đặt luồng nào. Chỉ số 0 là luồng (x 0, y 0, z 0); nó vào lane 0 của warp 0. Nhấn Bước.</>}>
                  <>Nothing placed yet. Index 0 is thread (x 0, y 0, z 0); it goes to lane 0 of warp 0. Press Step.</>
                </Translated>
              ) : (
                <>
                  <Translated
                    vi={
                      <>
                        Chỉ số {cur}: luồng (x {curT.x}, y {curT.y}, z {curT.z}), vì {linearWorking(curT.x, curT.y, curT.z, d)}. Nó vào warp{' '}
                        {warpOf(cur, width)}, lane {laneOf(cur, width)}.{' '}
                      </>
                    }
                  >
                    Index {cur}: thread (x {curT.x}, y {curT.y}, z {curT.z}), because {linearWorking(curT.x, curT.y, curT.z, d)}. It goes to warp{' '}
                    {warpOf(cur, width)}, lane {laneOf(cur, width)}.{' '}
                  </Translated>
                  {prevT && curT.z !== prevT.z
                    ? tr(`x and y both ran out, so z went up by one: one step of z is worth ${d.x * d.y} positions (blockDim.x times blockDim.y).`, `x và y đều đã hết, nên z tăng một: một bước của z đáng giá ${d.x * d.y} vị trí (blockDim.x nhân blockDim.y).`)
                    : prevT && curT.y !== prevT.y
                      ? tr(`x ran out, so y went up by one: one step of y is worth ${d.x} positions (blockDim.x).`, `x đã hết, nên y tăng một: một bước của y đáng giá ${d.x} vị trí (blockDim.x).`)
                      : ''}{' '}
                  {laneOf(cur, width) === width - 1 ? tr(`That fills warp ${warpOf(cur, width)}.`, `Thế là warp ${warpOf(cur, width)} đầy.`) : ''}
                  {at >= threads && k.padded > 0 ? tr(` The block is used up and the last warp has ${k.padded} lanes with no thread.`, ` Khối đã dùng hết và warp cuối có ${k.padded} lane không có luồng.`) : ''}
                </>
              )}
            </div>
            <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
              <div>
                {d.x <= 64 ? <BlockView d={d} width={width} pick={cur} upTo={at} /> : <p className="muted"><Translated vi="Một hàng dài: đường đi chỉ là x = 0, 1, 2, ...">One long row: the walk is just x = 0, 1, 2, ...</Translated></p>}
              </div>
              <div>
                <WarpRows d={d} width={width} pick={cur} upTo={at} />
              </div>
            </div>
          </>
        ) : (
          <p className="muted"><Translated vi="Hãy đặt một khối mà trình tô vẽ được, và đường đi sẽ hiện ở đây.">Set a block the painter can draw, and the walk appears here.</Translated></p>
        )}
      </Bench>
    </>
  );
}

// ---------------------------------------------------------------- one fetch, many lanes
const LANE_CHOICES = [
  { v: 8, get label() { return tr("8 lanes (the toy size of the book's Fig. 4.8)", '8 lane (kích thước đồ chơi trong Fig. 4.8 của sách)'); } },
  { v: 32, get label() { return tr("32 lanes (the book's warp)", '32 lane (warp của sách)'); } },
  { v: 64, get label() { return tr('64 lanes (HIP on gfx9)', '64 lane (HIP trên gfx9)'); } },
];
const REG_ROWS = [
  { key: 'r2', label: 'r2' },
  { key: 'r3', label: 'r3' },
  { key: 'r1', label: 'r1' },
] as const;

function FetchStepper() {
  const [lanes, setLanes] = useState(8);
  const [warps, setWarps] = useState<WarpRun[]>(() => [startWarp(0, 8), startWarp(1, 8)]);
  const [last, setLast] = useState<number | null>(null);
  const reset = (w: number) => {
    setLanes(w);
    setWarps([startWarp(0, w), startWarp(1, w)]);
    setLast(null);
  };
  const advance = (which: number[]) => {
    setWarps((ws) => ws.map((w, i) => (which.includes(i) ? step(w) : w)));
    setLast(which[which.length - 1]);
  };
  const said = () => {
    if (last == null) return tr('Nothing has been fetched. Each processing block holds one warp, and each warp has its own program counter.', 'Chưa nạp lệnh nào. Mỗi processing block giữ một warp, và mỗi warp có bộ đếm chương trình riêng.');
    const w = warps[last];
    const ins = STRAIGHT_LINE[w.pc - 1];
    const a = w.regs[0];
    const z = w.regs[lanes - 1];
    const show = (r: typeof a, i: number) =>
      ins.dst === 'r1' ? `${r.r2} + ${r.r3} = ${r.r1}` : tr(`${ins.dst} = ${r[ins.dst]}, for its own i = ${w.first + i}`, `${ins.dst} = ${r[ins.dst]}, với i riêng của nó = ${w.first + i}`);
    return tr(`Processing block ${last} fetched "${ins.text}" once, and all ${lanes} lanes of warp ${last} executed it. Lane 0: ${show(a, 0)}. Lane ${lanes - 1}: ${show(z, lanes - 1)}.`, `Processing block ${last} nạp "${ins.text}" một lần, và cả ${lanes} lane của warp ${last} đều thực thi nó. Lane 0: ${show(a, 0)}. Lane ${lanes - 1}: ${show(z, lanes - 1)}.`);
  };
  return (
    <Bench
      title={tr('One fetch, many lanes', 'Một lần nạp, nhiều lane')}
      task={tr('Step each warp. Count the fetches, then count the additions.', 'Cho từng warp chạy từng bước. Đếm số lần nạp lệnh, rồi đếm số phép cộng.')}
      foot={
        <Translated
          vi={
            <>
              Chương trình gồm ba lệnh không có nhánh; lệnh cuối là ví dụ của chính sách. Đầu vào là những số nhỏ bịa ra, khác
              nhau với mỗi luồng, để bạn có thể kiểm tra một lane bằng tay.
            </>
          }
        >
          <>
            The program is three instructions with no branch; the last is the book's own example. The inputs are small made-up numbers,
            different for every thread, so that you can check a lane by hand.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field" style={{ maxWidth: '100%', minWidth: 0 }}>
          <Translated vi="Độ rộng của một warp">Width of a warp</Translated>
          <select value={lanes} aria-label={tr('Width of a warp', 'Độ rộng của một warp')} onChange={(e) => reset(Number(e.target.value))}>
            {LANE_CHOICES.map((c) => <option key={c.v} value={c.v}>{c.label}</option>)}
          </select>
        </label>
        <button className="primary" onClick={() => advance([0])} disabled={finished(warps[0])}><Translated vi="Bước warp 0">Step warp 0</Translated></button>
        <button onClick={() => advance([1])} disabled={finished(warps[1])}><Translated vi="Bước warp 1">Step warp 1</Translated></button>
        <button onClick={() => advance([0, 1])} disabled={finished(warps[0]) && finished(warps[1])}><Translated vi="Bước cả hai">Step both</Translated></button>
        <button onClick={() => reset(lanes)} disabled={warps.every((w) => w.fetches === 0)}><Translated vi="Bắt đầu lại">Restart</Translated></button>
      </div>
      <div className="narrate" role="status" data-readout="fetch" style={{ marginBottom: '0.9rem' }}>{said()}</div>
      <div className="cols" style={lanes > 8 ? { gridTemplateColumns: 'minmax(0, 1fr)' } : undefined}>
        {warps.map((w, i) => (
          <div className="panel" key={i} data-warp={i} style={{ overflowX: 'auto' }}>
            <div className="mx-label"><Translated vi={<>Processing block {i}: warp {i}, luồng {w.first} đến {w.first + lanes - 1}</>}>Processing block {i}: warp {i}, threads {w.first} to {w.first + lanes - 1}</Translated></div>
            <div className="row" style={{ alignItems: 'flex-start', gap: '0.6rem 1.4rem', marginBottom: '0.6rem' }}>
              <div>
                <div className="muted"><Translated vi="nạp và phân phát lệnh (một bộ cho cả block)">instruction fetch and dispatch (one for the block)</Translated></div>
                <ol style={{ margin: '0.2rem 0 0', paddingLeft: '1.4rem' }} className="mono">
                  {STRAIGHT_LINE.map((ins, k) => (
                    <li key={k} data-pc={k === w.pc ? '' : undefined} style={{ fontWeight: k === w.pc - 1 ? 700 : 400, color: k < w.pc ? 'var(--ink)' : 'var(--ink-2)' }}>
                      {ins.text} <span className="muted" style={{ fontWeight: 400 }}>({ins.meaning}){k === w.pc - 1 ? tr(' ◂ just executed', ' ◂ vừa thực thi') : k === w.pc ? tr(' ◂ next', ' ◂ kế tiếp') : ''}</span>
                    </li>
                  ))}
                </ol>
              </div>
              <div data-out="counts">
                <div><Translated vi={<><b data-out="fetches">{w.fetches}</b> lần nạp lệnh</>}><b data-out="fetches">{w.fetches}</b> {w.fetches === 1 ? 'fetch' : 'fetches'}</Translated></div>
                <div><Translated vi={<><b data-out="laneops">{w.laneOps}</b> phép toán trên lane</>}><b data-out="laneops">{w.laneOps}</b> lane-operations</Translated></div>
              </div>
            </div>
            <table className="threads" aria-label={tr(`Registers of warp ${i}, one column per lane`, `Thanh ghi của warp ${i}, mỗi lane một cột`)} style={{ width: 'auto' }}>
              <thead>
                <tr>
                  <th><Translated vi="lane">lane</Translated></th>
                  {w.regs.map((_, l) => <th key={l} style={{ textAlign: 'right', paddingLeft: '0.35rem' }}>{l}</th>)}
                </tr>
              </thead>
              <tbody>
                {REG_ROWS.map((r) => (
                  <tr key={r.key} className={w.pc > 0 && STRAIGHT_LINE[w.pc - 1].dst === r.key ? 'cur' : ''} data-reg={r.key}>
                    <td className="t-reg">{r.label}</td>
                    {w.regs.map((reg, l) => (
                      <td key={l} className="t-reg" style={{ textAlign: 'right', paddingLeft: '0.35rem' }}>{reg[r.key] == null ? '·' : reg[r.key]}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const P_ID = 'ch04/4.4/p-warp-of-thread';
const P_BLOCK = dims(8, 8);
const P_AT = place(3, 5, 0, P_BLOCK, 32);

export const questionsS44: Record<string, Question> = {
  'ch04/4.4/q-3d': {
    kind: 'numeric',
    id: 'ch04/4.4/q-3d',
    prompt: (
      <Translated
        vi={
          <>
            Một khối có <code>blockDim</code> x 4, y 8, z 2 và độ rộng warp là 32. Warp nào (đếm từ 0) chứa luồng có{' '}
            <code>threadIdx</code> x 1, y 2, z 1?
          </>
        }
      >
        <>
          A block has <code>blockDim</code> x 4, y 8, z 2 and the warp width is 32. Which warp (counting from 0) holds the
          thread with <code>threadIdx</code> x 1, y 2, z 1?
        </>
      </Translated>
    ),
    answer: 1,
    explain: (
      <Translated
        vi={
          <p>
            Chỉ số tuyến tính = x + y·Dx + z·Dx·Dy = 1 + 2·4 + 1·32 = 41. Chia cho 32 và làm tròn xuống được warp 1; phần dư,
            9, là lane của nó. Một lối tắt cho hình dạng này: một lát z là 4·8 = 32 luồng, đúng một warp, nên số warp chính là
            chỉ số z. Đó là ví dụ khối ba chiều của chính sách.
          </p>
        }
      >
        <p>
          Linear index = x + y·Dx + z·Dx·Dy = 1 + 2·4 + 1·32 = 41. Dividing by 32 and rounding down gives warp 1; the
          remainder, 9, is its lane. A shortcut for this shape: one z slice is 4·8 = 32 threads, exactly one warp, so the
          warp number is the z index. That is the book's own example of a three-dimensional block.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.4, pp. 75–76',
  },
  'ch04/4.4/q-ex1ab': {
    kind: 'numeric',
    id: 'ch04/4.4/q-ex1ab',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 1 của sách, phần a và b. Một kernel được khởi chạy trên <code>N = 1024</code> phần tử với{' '}
            <code>(N + 128 - 1)/128</code> khối 128 luồng, trên một thiết bị có warp rộng 32. Lưới có bao nhiêu warp?
          </>
        }
      >
        <>
          The book's exercise 1, parts a and b. A kernel is launched over <code>N = 1024</code> elements with{' '}
          <code>(N + 128 - 1)/128</code> blocks of 128 threads, on a device whose warps are 32 wide. How many warps are in
          the grid?
        </>
      </Translated>
    ),
    answer: 32,
    unit: 'warps',
    explain: (
      <Translated
        vi={
          <p>
            Một khối 128 luồng là 128 / 32 = 4 warp (phần a). Lần khởi chạy có (1024 + 127) / 128 = 8 khối theo số học số
            nguyên, nên lưới chứa 8 · 4 = 32 warp (phần b). Trên một thiết bị rộng 64, cùng lần khởi chạy đó là 16 warp. Sách
            không in đáp án; đáp án này là của chúng tôi.
          </p>
        }
      >
        <p>
          A 128-thread block is 128 / 32 = 4 warps (part a). The launch has (1024 + 127) / 128 = 8 blocks in integer
          arithmetic, so the grid holds 8 · 4 = 32 warps (part b). On a 64-wide device the same launch is 16 warps. The
          book prints no answers; this one is ours.
        </p>
      </Translated>
    ),
    source: 'ch. 4 exercise 1, pp. 90–91',
    vi: { unit: 'warp', source: 'ch. 4 bài tập 1, pp. 90–91' },
  },
  'ch04/4.4/q-pad': {
    kind: 'choice',
    id: 'ch04/4.4/q-pad',
    prompt: (
      <Translated
        vi={
          <>
            Bạn khởi chạy các khối 48 luồng trên một thiết bị có warp rộng 32 và trên một thiết bị có warp rộng 64. Một khối
            chiếm bao nhiêu warp trên mỗi thiết bị, và bao nhiêu lane ngồi không?
          </>
        }
      >
        <>
          You launch 48-thread blocks on a device with 32-wide warps and on one with 64-wide warps. How many warps does
          one block take on each, and how many lanes sit idle?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="2 warp với 16 lane rỗi trên cả hai.">{'2 warps with 16 idle lanes on both.'}</Translated>,
        why: <Translated vi="Với độ rộng 64, 48 luồng vừa trong một warp.">{'At 64 wide, 48 threads fit in one warp.'}</Translated>,
      },
      {
        label: (
          <Translated vi="1.5 warp trên thiết bị thứ nhất và 0.75 warp trên thiết bị thứ hai, không có gì rỗi.">
            {'1.5 warps on the first and 0.75 of a warp on the second, with nothing idle.'}
          </Translated>
        ),
        why: (
          <Translated vi="Warp là nguyên vẹn. Một warp chưa đầy vẫn chiếm tất cả các lane của nó.">
            {'Warps are whole. A warp that is not full still occupies all of its lanes.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="2 warp với 16 lane rỗi trên thiết bị thứ nhất; 1 warp với 16 lane rỗi trên thiết bị thứ hai.">
            {'2 warps with 16 idle lanes on the first; 1 warp with 16 idle lanes on the second.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="2 warp trên thiết bị thứ nhất và 1 trên thiết bị thứ hai, không có gì rỗi.">
            {'2 warps on the first and 1 on the second, with nothing idle.'}
          </Translated>
        ),
        why: (
          <Translated vi="Các lane vẫn ở đó dù có luồng lấp vào hay không: 2 · 32 − 48 trên thiết bị thứ nhất, và 64 − 48 trên thiết bị thứ hai.">
            {'The lanes are there whether or not a thread fills them: 2 · 32 − 48 on the first device, and 64 − 48 on the second.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Số warp trong một khối = số luồng chia cho độ rộng, làm tròn lên; warp cuối được độn thêm. Ở 32: ⌈48 / 32⌉ = 2,
            với 2·32 − 48 = 16 lane không dùng. Ở 64: ⌈48 / 64⌉ = 1, với 64 − 48 = 16 không dùng. Số lane rỗi tình cờ bằng
            nhau; số warp mà bộ lập lịch có để làm việc thì không. Trường hợp rộng 32 là của sách; trường hợp rộng 64 là cùng
            phép tính với độ rộng mà AMD ghi trong tài liệu.
          </p>
        }
      >
        <p>
          Warps in a block = threads divided by the width, rounded up; the last warp is padded. At 32: ⌈48 / 32⌉ = 2, with
          2·32 − 48 = 16 lanes unused. At 64: ⌈48 / 64⌉ = 1, with 64 − 48 = 16 unused. The idle lanes happen to be equal;
          the number of warps the scheduler has to work with is not. The 32-wide case is the book's; the 64-wide one is
          the same arithmetic with AMD's documented width.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.4, p. 75; notes/today/ch04-facts.md B6', // i18n-ignore: citation
  },
  'ch04/4.4/q-rows': {
    kind: 'choice',
    id: 'ch04/4.4/q-rows',
    prompt: (
      <Translated vi={<>Một khối 16 × 16. Một warp phủ bao nhiêu hàng của khối khi warp rộng 32, và khi warp rộng 64?</>}>
        <>A block is 16 × 16. How many rows of the block does one warp cover when warps are 32 wide, and when they are 64 wide?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Một hàng, và hai hàng.">{'One row, and two rows.'}</Translated>,
        why: <Translated vi="Một hàng của khối này là 16 luồng, một nửa warp rộng 32.">{'A row of this block is 16 threads, half of a 32-wide warp.'}</Translated>,
      },
      { label: <Translated vi="Hai hàng, và bốn hàng.">{'Two rows, and four rows.'}</Translated>, correct: true },
      {
        label: <Translated vi="Hai hàng ở cả hai độ rộng.">{'Two rows at both widths.'}</Translated>,
        why: <Translated vi="Một warp rộng hơn lấy một đoạn dài hơn của cùng thứ tự tuyến tính.">{'A wider warp takes a longer run of the same linear order.'}</Translated>,
      },
      {
        label: <Translated vi="Tám hàng, và bốn hàng.">{'Eight rows, and four rows.'}</Translated>,
        why: <Translated vi="Đó là số warp trong khối, không phải số hàng trong một warp.">{'Those are the numbers of warps in the block, not of rows in a warp.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            x được lấp trước, nên một warp là một đoạn các hàng liên tiếp mỗi khi độ dài hàng chia hết độ rộng. 32 / 16 = 2
            hàng; 64 / 16 = 4 hàng. Khối là 8 warp trên thiết bị này và 4 trên thiết bị kia, và một ranh giới hàng vốn nằm giữa
            hai warp ở độ rộng 32 có thể rơi vào bên trong một warp ở độ rộng 64. Phần 4.5 đếm xem điều đó ảnh hưởng thế nào tới
            một phép kiểm tra biên.
          </p>
        }
      >
        <p>
          x fills first, so a warp is a run of consecutive rows whenever the row length divides the width. 32 / 16 = 2
          rows; 64 / 16 = 4 rows. The block is 8 warps on one device and 4 on the other, and a row boundary that fell
          between two warps at 32 can fall inside one at 64. Section 4.5 counts what that does to a boundary check.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.5, p. 82; notes/today/ch04-facts.md B6', // i18n-ignore: citation
  },
  'ch04/4.4/q-hardcode': {
    kind: 'choice',
    id: 'ch04/4.4/q-hardcode',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel tính lane của nó bằng <code>threadIdx.x % 32</code> và dựng lane mask trong một số nguyên 32 bit. Nó
            được biên dịch, không sửa gì, cho một thiết bị có warp rộng 64. Điều gì xảy ra?
          </>
        }
      >
        <>
          A kernel computes its lane as <code>threadIdx.x % 32</code> and builds lane masks in a 32-bit integer. It is
          compiled, unchanged, for a device whose warps are 64 wide. What happens?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không có gì thay đổi: lane được đánh số từ 0 đến 31 trên mọi thiết bị, bất kể độ rộng.">
            {'Nothing changes: lanes are numbered 0 to 31 on every device, whatever its width.'}
          </Translated>
        ),
        why: <Translated vi="Trên thiết bị đó, lane chạy từ 0 đến 63.">{'On that device the lanes run from 0 to 63.'}</Translated>,
      },
      {
        label: (
          <Translated vi="Nó không biên dịch được: trình biên dịch từ chối số lane 32 cho một target rộng 64.">
            {'It fails to compile: the compiler rejects a lane count of 32 for a 64-wide target.'}
          </Translated>
        ),
        why: (
          <Translated vi="Đó là phép tính số nguyên bình thường. Nó biên dịch được ở mọi nơi, đó là lý do lỗi này sống sót.">
            {'It is ordinary integer arithmetic. It compiles everywhere, which is why the mistake survives.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó chạy đúng như đã viết: thiết bị chuyển sang warp rộng 32 cho một kernel được dựng theo cách đó.">
            {'It runs as written: the device switches to 32-wide warps for a kernel built that way.'}
          </Translated>
        ),
        why: (
          <Translated vi="Độ rộng thuộc về phần cứng. Một kernel không thể đặt nó.">
            {'The width belongs to the hardware. A kernel cannot set it.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó chạy, nhưng coi mỗi warp là hai nhóm 32, và mask của nó không thể gọi tên lane 32 đến 63.">
            {'It runs, but treats each warp as two groups of 32, and its masks cannot name lanes 32 to 63.'}
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Tài liệu HIP của AMD nói thẳng điều này: mã giả định 32 có thể chạy trên thiết bị rộng 64, và khi đó chỉ dùng
              một nửa tài nguyên tính toán; và <code>threadIdx.x % warpSize</code> có thể vượt quá 31 ở đó, nên{' '}
              <code>1 &lt;&lt; lane</code> phải trở thành phép dịch 64 bit. Cách sửa là đọc <code>warpSize</code> trong mã
              device và thuộc tính thiết bị ở host.
            </p>
            <p>
              Bạn sẽ gặp đúng điều này trong llama.cpp: <code>argmax.cu</code> của nó được viết cho{' '}
              <code>WARP_SIZE 32</code> từ đầu đến cuối, trong khi các kernel khác trong cùng thư mục hỏi độ rộng vật lý.
            </p>
          </>
        }
      >
        <>
          <p>
            AMD's HIP documentation says it in so many words: code that assumes 32 can run on a 64-wide device, and then
            uses half of the compute resources; and <code>threadIdx.x % warpSize</code> can exceed 31 there, so{' '}
            <code>1 &lt;&lt; lane</code> has to become a 64-bit shift. The fix is to read <code>warpSize</code> in device
            code and the device property on the host.
          </p>
          <p>
            You will meet exactly this in llama.cpp: its <code>argmax.cu</code> is written for{' '}
            <code>WARP_SIZE 32</code> throughout, while other kernels in the same directory ask for the physical width.
          </p>
        </>
      </Translated>
    ),
    source: 'notes/today/ch05-facts.md B5; notes/today/ch04-facts.md D1', // i18n-ignore: citation
  },
};

export function S44() {
  return (
    <>
      <Translated
        vi={
          <p>
            Phần 4.3 đã cho bạn rào chắn. Sách mở đầu phần này bằng quy tắc mà rào chắn phục vụ: bên trong một khối, hãy giả
            định các luồng chạy theo thứ tự bất kỳ, và khi một pha phải xong trước khi pha sau bắt đầu, hãy nói rõ điều đó
            bằng một rào chắn. Quy tắc đó vẫn đứng vững. Phần này nói về một điều mà quy tắc cho phép bạn bỏ qua nhưng hiệu
            năng thì không: phần cứng thực tế gom các luồng của một khối như thế nào khi chạy chúng.
          </p>
        }
      >
        <p>
          Section 4.3 gave you the barrier. The book opens this section with the rule it serves: inside a block, assume
          the threads run in any order whatever, and when one phase must finish before the next starts, say so with a
          barrier. That rule stands. This section is about something the rule lets you ignore and performance does not:
          how the hardware in fact groups the threads of a block when it runs them.
        </p>
      </Translated>
      <Source section="4.4" pages="74" />

      <h2><Translated vi="Một khối được cắt thành các warp">A block is cut into warps</Translated></h2>
      <Translated
        vi={
          <p>
            Một SM không lập lịch các luồng của một khối từng luồng một. Nó cắt khối thành các nhóm có kích thước cố định, gọi
            là <b>warp</b>, và warp là đơn vị nó lập lịch: khi một warp được chọn, mọi luồng của nó cùng thực thi lệnh kế
            tiếp. Sách cho kích thước là 32 luồng “in most implementations to date” (trong hầu hết các cài đặt cho đến nay),
            và trong cùng đoạn đó gọi nó là một chi tiết cài đặt có thể thay đổi.
          </p>
        }
      >
        <p>
          An SM does not schedule the threads of a block one at a time. It cuts the block into groups of a fixed size,
          called <b>warps</b>, and the warp is the unit it schedules: when a warp is chosen, all of its threads execute
          the next instruction together. The book gives the size as 32 threads “in most implementations to date”, and in
          the same paragraph calls it an implementation detail that may change.
        </p>
      </Translated>
      <Source section="4.4" pages="74, 77" figs="Fig. 4.6" />
      <Translated
        vi={
          <p>
            Hãy hiểu lời cảnh báo đó theo nghĩa đen, vì trên ba backend của khóa học này con số đó đã không còn là một con số
            duy nhất. Một kernel đọc độ rộng qua <code>warpSize</code> trên CUDA và HIP; một kernel Metal nhận nó như một tham
            số. Không kernel nào bạn viết nên chứa hằng số 32 làm độ rộng.
          </p>
        }
      >
        <p>
          Take that caution literally, because across the three backends of this course the number is already not one
          number. A kernel reads the width as <code>warpSize</code> on CUDA and HIP; a Metal kernel receives it as a
          parameter. No kernel you write should contain the literal 32 as a width.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch04-facts.md A8, B6, C2; ch05-facts.md B5)', '2026-11-20 (notes/today/ch04-facts.md mục A8, B6, C2; ch05-facts.md mục B5)')}>
        <Translated
          vi={
            <p>
              CUDA: tài liệu liệt kê kích thước warp là 32 cho mọi compute capability và mô tả <code>warpSize</code> là
              “commonly 32” (thường là 32). HIP: 64 trên thiết bị gfx9 (AMD Instinct), 32 trên gfx10 trở lên. Metal: đơn vị là
              SIMD-group; độ rộng của nó là thuộc tính của một pipeline đã biên dịch và không có giá trị nào được ghi trong tài
              liệu.
            </p>
          }
        >
          <p>
            CUDA: the documentation lists a warp size of 32 for every compute capability and describes{' '}
            <code>warpSize</code> as “commonly 32”. HIP: 64 on gfx9 devices (AMD Instinct), 32 on gfx10 and above. Metal:
            the unit is a SIMD-group; its width is a property of a compiled pipeline and has no documented value.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Nếu bạn đã viết mã AVX hay NEON, bạn đã có hình dung đúng: một warp là lượng luồng bằng một vector, và mỗi luồng
            là một <b>lane</b> của nó. Phần còn lại của mục này nói về luồng nào nhận lane nào.
          </p>
        }
      >
        <p>
          If you have written AVX or NEON code, you already have the right picture: a warp is a vector's worth of threads,
          and each thread is one <b>lane</b> of it. The rest of this section is about which thread gets which lane.
        </p>
      </Translated>

      <h2><Translated vi="Đếm warp">Counting warps</Translated></h2>
      <Translated
        vi={
          <p>
            Một khối 256 luồng là 256 / 32 = 8 warp. Nếu một SM giữ ba khối như vậy, nó giữ 24 warp. Khi kích thước khối không
            là bội số của độ rộng, warp cuối được lấp bằng các lane không chứa luồng nào: một khối 48 luồng là 2 warp, và warp
            thứ hai có 16 luồng và 16 lane rỗi. Những lane đó là phần cứng bị giữ chỗ và không làm gì suốt thời gian khối tồn
            tại.
          </p>
        }
      >
        <p>
          A block of 256 threads is 256 / 32 = 8 warps. If an SM holds three such blocks, it holds 24 warps. When the
          block size is not a multiple of the width, the last warp is filled up with lanes that hold no thread: a
          48-thread block is 2 warps, and the second has 16 threads and 16 idle lanes. Those lanes are hardware that is
          reserved and does nothing for as long as the block lives.
        </p>
      </Translated>
      <Source section="4.4" pages="74–75" figs="Fig. 4.6" />
      <Translated
        vi={
          <p>
            Tổng quát, một khối <i>T</i> luồng trên thiết bị độ rộng <i>W</i> là ⌈<i>T</i> / <i>W</i>⌉ warp với{' '}
            ⌈<i>T</i> / <i>W</i>⌉·<i>W</i> − <i>T</i> lane rỗi. Hướng dẫn của NVIDIA cũng nói vậy về thiết bị của họ: số
            luồng nào cũng hợp lệ, và một tổng không phải bội số của 32 để lại các lane không dùng ở warp cuối.
          </p>
        }
      >
        <p>
          In general a block of <i>T</i> threads on a device of width <i>W</i> is ⌈<i>T</i> / <i>W</i>⌉ warps with{' '}
          ⌈<i>T</i> / <i>W</i>⌉·<i>W</i> − <i>T</i> idle lanes. NVIDIA's guide says the same of its own devices: any
          number of threads is legal, and a total that is not a multiple of 32 leaves lanes of the last warp unused.
        </p>
      </Translated>
      <Ask q={questionsS44['ch04/4.4/q-ex1ab']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS44['ch04/4.4/q-pad']} />

      <h2><Translated vi="Những luồng nào chung một warp">Which threads share a warp</Translated></h2>
      <Translated
        vi={
          <p>
            Với khối một chiều, việc cắt đi theo <code>threadIdx.x</code>: warp 0 chứa luồng 0 đến 31, warp 1 chứa 32 đến 63,
            và warp <i>n</i> chứa 32<i>n</i> đến 32(<i>n</i> + 1) − 1. Với hai hoặc ba chiều, các luồng trước hết được xếp
            thành một hàng, theo thứ tự hàng (row-major): toàn bộ <code>y = 0</code> theo <code>x</code> tăng dần, rồi{' '}
            <code>y = 1</code>, và với khối ba chiều thì toàn bộ <code>z = 0</code> trước bất kỳ phần nào của{' '}
            <code>z = 1</code>. Rồi hàng đó được cắt thành các đoạn liên tiếp dài bằng độ rộng.
          </p>
        }
      >
        <p>
          For a one-dimensional block the cut follows <code>threadIdx.x</code>: warp 0 holds threads 0 to 31, warp 1
          holds 32 to 63, and warp <i>n</i> holds 32<i>n</i> to 32(<i>n</i> + 1) − 1. For two or three dimensions the
          threads are first put in one line, in row-major order: all of <code>y = 0</code> in increasing{' '}
          <code>x</code>, then <code>y = 1</code>, and for a three-dimensional block all of <code>z = 0</code> before any
          of <code>z = 1</code>. The line is then cut into consecutive runs of the width.
        </p>
      </Translated>
      <Source section="4.4" pages="75–76" figs="Fig. 4.7" />
      <Translated
        vi={
          <p>
            Viết ra, với <code>blockDim</code> = (<i>Dx</i>, <i>Dy</i>, <i>Dz</i>) và độ rộng <i>W</i>:
          </p>
        }
      >
        <p>
          Written out, with <code>blockDim</code> = (<i>Dx</i>, <i>Dy</i>, <i>Dz</i>) and width <i>W</i>:
        </p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>chỉ số tuyến tính = <code>x + y·Dx + z·Dx·Dy</code></li>
            <li>warp = chỉ số tuyến tính / <i>W</i>, làm tròn xuống</li>
            <li>lane = chỉ số tuyến tính mod <i>W</i></li>
          </ul>
        }
      >
        <ul>
          <li>linear index = <code>x + y·Dx + z·Dx·Dy</code></li>
          <li>warp = linear index / <i>W</i>, rounded down</li>
          <li>lane = linear index mod <i>W</i></li>
        </ul>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Công thức là của chúng tôi. Sách nêu quy tắc bằng lời; NVIDIA cho các bước nhảy, blockDim.x cho y và
              blockDim.x * blockDim.y cho z (facts A8).
            </p>
          }
        >
          <p>
            The formula is ours. The book states the rule in words; NVIDIA gives the strides, blockDim.x for y and
            blockDim.x * blockDim.y for z (facts A8).
          </p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Đó là offset của phần tử (<i>z</i>, <i>y</i>, <i>x</i>) trong một tensor thứ tự hàng có shape (<i>Dz</i>,{' '}
            <i>Dy</i>, <i>Dx</i>): làm phẳng khối như <code>reshape(-1)</code> sẽ làm, và một warp là <i>W</i>{' '}
            phần tử liên tiếp. Sách xét ba trường hợp. Một khối 4 × 4 là 16 luồng, nửa warp, được độn thêm 16 vị trí. Một
            khối 8 × 8 là hai warp: hàng 0 đến 3, rồi hàng 4 đến 7. Một khối x 4, y 8, z 2 (sách viết z trước, thành 2 × 8 ×
            4) cũng là hai warp: toàn bộ <code>z = 0</code>, rồi toàn bộ <code>z = 1</code>.
          </p>
        }
      >
        <p>
          It is the offset of element (<i>z</i>, <i>y</i>, <i>x</i>) in a row-major tensor of shape (<i>Dz</i>,{' '}
          <i>Dy</i>, <i>Dx</i>): flatten the block the way <code>reshape(-1)</code> would, and a warp is <i>W</i>{' '}
          consecutive elements. The book works three cases. A 4 × 4 block is 16 threads, half a warp, padded with 16 more
          positions. An 8 × 8 block is two warps: rows 0 to 3, then rows 4 to 7. A block with x 4, y 8, z 2 (the book
          writes it z first, as 2 × 8 × 4) is also two warps: the whole of <code>z = 0</code>, then the whole of{' '}
          <code>z = 1</code>.
        </p>
      </Translated>
      <Predict
        id={P_ID}
        prompt={
          <Translated
            vi={
              <>
                Một khối 8 × 8, warp rộng 32. Warp nào (đếm từ 0) chứa luồng có <code>threadIdx.x = 3</code>
                , <code>threadIdx.y = 5</code>? Hãy chốt câu trả lời trước khi mở trình tô.
              </>
            }
          >
            <>
              An 8 × 8 block, warps 32 wide. Which warp (counting from 0) holds the thread with <code>threadIdx.x = 3</code>
              , <code>threadIdx.y = 5</code>? Commit before you open the painter.
            </>
          </Translated>
        }
      >
        {(v) => (
          <>
            <p data-reveal={P_ID}>
              {v === P_AT.warp ? tr('Yes: ', 'Đúng: ') : tr(`It is warp ${P_AT.warp}: `, `Đó là warp ${P_AT.warp}: `)}
              <Translated
                vi={
                  <>
                    chỉ số tuyến tính là {linearWorking(3, 5, 0, P_BLOCK)}, và {P_AT.index} / 32 làm tròn xuống là {P_AT.warp},
                    với lane {P_AT.index} mod 32 = {P_AT.lane}. Hàng 0 đến 3 lấp đầy warp 0, nên mọi thứ từ hàng 4 trở đi nằm
                    trong warp 1. Ô được đánh dấu là luồng đó.
                  </>
                }
              >
                the linear index is {linearWorking(3, 5, 0, P_BLOCK)}, and {P_AT.index} / 32 rounded down is {P_AT.warp}, with
                lane {P_AT.index} mod 32 = {P_AT.lane}. Rows 0 to 3 fill warp 0, so everything from row 4 on is in warp 1.
                The marked cell is the thread.
              </Translated>
            </p>
            <BlockView d={P_BLOCK} width={32} pick={P_AT.index} />
          </>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Giờ hãy tự đổi hình dạng và độ rộng. Trình tô mở với khối 16 × 16; cấu hình mẫu 8 × 8 là khối của câu dự đoán. Hãy
            thử các cấu hình mẫu trước, rồi chuyển độ rộng sang 64 trong khi vẫn chọn khối 8 × 8 và 16 × 16, và xem những luồng
            nào đổi bạn đồng hành.
          </p>
        }
      >
        <p>
          Now change the shape and the width yourself. The painter opens on a 16 × 16 block; the 8 × 8 preset is the
          block of the prediction. Try the presets first, then switch the width to 64 with the 8 × 8 and the 16 × 16
          block still selected, and watch which threads change company.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Trình tô áp dụng cách cắt mà NVIDIA mô tả, với bất kỳ độ rộng nào bạn chọn. AMD mô tả thứ tự tuyến tính và tính
              lane bằng <code>threadIdx.x % warpSize</code> trong các ví dụ của họ, nhưng không có câu nào nói warp là các đoạn
              liên tiếp (facts B6). Với Metal không có quy tắc nào để áp dụng, nên lựa chọn đó chỉ vẽ những gì một kernel đã
              ghi lại trên thiết bị của bạn.
            </p>
          }
        >
          <p>
            The painter applies the cut that NVIDIA documents, with whatever width you choose. AMD documents the linear
            order and computes lanes as <code>threadIdx.x % warpSize</code> in its examples, but has no single sentence
            saying warps are consecutive runs (facts B6). For Metal no rule exists to apply, so that choice draws only
            what a kernel recorded on your device.
          </p>
        </Translated>
      </Simplification>
      <WarpLab />
      <Ask q={questionsS44['ch04/4.4/q-3d']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Hình dạng của khối thay đổi điều gì">What the shape of a block changes</Translated></h2>
      <Translated
        vi={
          <p>
            Số luồng quyết định số warp. Hình dạng quyết định ai nằm trong đó. Một khối 16 × 16 và một khối 256 × 1 đều là 8
            warp ở độ rộng 32, nhưng ở khối thứ nhất một warp là hai hàng gồm 16 luồng kề nhau theo <code>x</code>, còn ở
            khối thứ hai nó là 32 luồng kề nhau theo <code>x</code>. Đổi độ rộng sang 64 và khối 16 × 16 thành 4 warp, mỗi
            warp bốn hàng.
          </p>
        }
      >
        <p>
          The number of threads fixes the number of warps. The shape fixes who is in them. A 16 × 16 block and a 256 × 1
          block are both 8 warps at width 32, but in the first a warp is two rows of 16 neighbours in <code>x</code>, and
          in the second it is 32 neighbours in <code>x</code>. Change the width to 64 and the 16 × 16 block is 4 warps of
          four rows each.
        </p>
      </Translated>
      <Source section="4.5" pages="82" />
      <Translated
        vi={
          <p>
            Hướng dẫn của NVIDIA nói khối nhiều chiều chỉ là “for convenience only” (cho tiện) và không ảnh hưởng hiệu năng.
            Điều đó đúng với bản thân lần khởi chạy. Nó không phải toàn bộ câu chuyện với một kernel, vì mọi thứ phần cứng làm
            theo từng warp đều phụ thuộc vào ai nằm trong warp: phân kỳ điều khiển của{' '}
            <a href={href('ch04', '4.5')}>phần 4.5</a>, và ở chương 6 là cả việc hợp nhất các truy cập bộ nhớ của một warp lẫn
            xung đột giữa chúng trong bộ nhớ chia sẻ.
          </p>
        }
      >
        <p>
          NVIDIA's guide says multi-dimensional blocks are “for convenience only” and do not affect performance. That is
          true of the launch itself. It is not the whole story for a kernel, because everything the hardware does per
          warp depends on who is in the warp: the control divergence of <a href={href('ch04', '4.5')}>section 4.5</a>,
          and in chapter 6 both the merging of a warp's memory accesses and the conflicts between them in shared memory.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Câu của NVIDIA: “The use of multi-dimensional thread blocks and grids is for convenience only and does not
              affect performance.” (notes/today/ch04-facts.md A8). Các tham chiếu về sau là tới §6.1 và §6.4 của sách, ở đây
              là <a href={href('ch06', '6.1')}>6.1</a> và <a href={href('ch06', '6.4')}>6.4</a>.
            </p>
          }
        >
          <p>
            NVIDIA's sentence: “The use of multi-dimensional thread blocks and grids is for convenience only and does not
            affect performance.” (notes/today/ch04-facts.md A8). The forward references are to the book's §6.1 and §6.4,
            here <a href={href('ch06', '6.1')}>6.1</a> and <a href={href('ch06', '6.4')}>6.4</a>.
          </p>
        </Translated>
      </Aside>
      <Ask q={questionsS44['ch04/4.4/q-rows']} />

      <h2><Translated vi="Một lệnh, mọi lane">One instruction, every lane</Translated></h2>
      <Translated
        vi={
          <p>
            Một SM thực thi một warp theo kiểu SIMD: single instruction, multiple data (một lệnh, nhiều dữ liệu). Tại mỗi thời
            điểm một lệnh được nạp, và nó được thực thi cho mọi luồng của warp, mỗi luồng trên dữ liệu của riêng nó.
          </p>
        }
      >
        <p>
          An SM executes a warp in SIMD fashion: single instruction, multiple data. At any instant one instruction is
          fetched, and it is executed for every thread of the warp, each on its own data.
        </p>
      </Translated>
      <Source section="4.4" pages="76–77" figs="Fig. 4.8" />
      <Translated
        vi={
          <p>
            Để điều đó rẻ, một SM được dựng từ các <b>processing block</b>: một nhóm streaming processor đứng sau một đơn vị
            nạp và phân phát lệnh. Các warp của SM được rải trên các processing block của nó, và mọi luồng của một warp chạy
            trên cùng một block. Hình của sách vẽ một SM đồ chơi với hai processing block, mỗi block tám lõi; về thiết bị ví dụ
            của mình, sách nói 128 streaming processor của một SM tạo thành bốn processing block, mỗi block 32.
          </p>
        }
      >
        <p>
          To make that cheap an SM is built out of <b>processing blocks</b>: a group of streaming processors behind one
          instruction fetch and dispatch unit. The SM's warps are spread over its processing blocks, and all threads of a
          warp run on the same one. The book's figure draws a toy SM with two processing blocks of eight; of its example
          device it says an SM's 128 streaming processors form four processing blocks of 32.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch04-facts.md A3)', '2026-11-20 (notes/today/ch04-facts.md, mục A3)')}>
        <Translated
          vi={
            <p>
              “Bốn processing block, mỗi block 32” là phát biểu của sách về ví dụ của nó và ở đây là unverified: văn bản của
              NVIDIA về thiết bị đó cho 128 lõi mỗi SM và không nhắc tới processing block. Thuật ngữ này là của NVIDIA, từ một
              whitepaper năm 2017 mô tả các chip cũ hơn với hai block 32 và bốn block 16, mỗi block có bộ lập lịch warp riêng.
            </p>
          }
        >
          <p>
            “Four processing blocks of 32” is the book's statement about its example and is unverified here: NVIDIA's
            text on that device gives 128 cores per SM and does not mention processing blocks. The term is NVIDIA's, from a 2017 whitepaper that describes older chips with two
            blocks of 32 and four of 16, each block with its own warp scheduler.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Khung bên lề của sách giải thích điều này bằng bộ xử lý trong sách giáo khoa kiến trúc máy tính: một đơn vị điều
            khiển, giữ bộ đếm chương trình và thanh ghi lệnh, và một đơn vị xử lý, giữ logic số học và các thanh ghi. Một
            processing block là một đơn vị điều khiển nối với nhiều đơn vị xử lý, mỗi lane một đơn vị. Một thanh ghi lệnh điều
            khiển tất cả, nên các lane không thể khác nhau ở việc chúng làm, chỉ khác ở nội dung thanh ghi: mọi lane thực thi{' '}
            <code>add r1, r2, r3</code>, và <code>r2</code> với <code>r3</code> chứa giá trị khác nhau ở mỗi lane.
          </p>
        }
      >
        <p>
          The book's sidebar explains it with the processor of an architecture textbook: a control unit, holding the
          program counter and the instruction register, and a processing unit, holding the arithmetic logic and the
          registers. A processing block is one control unit wired to many processing units, one per lane. One instruction
          register drives them all, so the lanes cannot differ in what they do, only in what their registers contain:
          every lane executes <code>add r1, r2, r3</code>, and <code>r2</code> and <code>r3</code> hold different values
          in each.
        </p>
      </Translated>
      <Source section="4.4" pages="78–79" figs={tr('sidebar “Warps and SIMD Hardware”', 'khung bên lề “Warps and SIMD Hardware”')} />
      <Translated vi={<p>Hãy chạy từng bước. Hai processing block, như trong hình đồ chơi của sách, mỗi block một warp riêng.</p>}>
        <p>
          Step it. Two processing blocks, as in the book's toy figure, each with its own warp.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Ở đây một lệnh mất một bước cho cả warp, và một warp di chuyển khi bạn nhấn nút của nó. Một SM thật có pipeline,
              độ trễ bộ nhớ và một bộ lập lịch chọn warp (phần 4.6). Mô hình trung thực ở một điểm: mỗi lệnh một lần nạp cho mỗi
              warp, và mỗi lane một bộ thanh ghi riêng.
            </p>
          }
        >
          <p>
            An instruction takes one step for a whole warp here, and a warp moves when you press its button. A real SM
            has pipelines, memory latency and a scheduler that picks the warp (section 4.6). The model is faithful about
            one thing: one fetch per instruction per warp, and one private set of registers per lane.
          </p>
        </Translated>
      </Simplification>
      <FetchStepper />

      <h2><Translated vi="Bạn viết cho một luồng">You write one thread</Translated></h2>
      <Translated
        vi={
          <p>
            Mã nguồn của kernel không nhắc gì đến những điều này. Bạn viết mã vô hướng cho một luồng; việc gom thành warp và
            việc dùng phần cứng SIMD diễn ra bên dưới. Vì bộ lập lịch trên thực tế giữ mọi luồng của một warp ở cùng một lệnh,
            sách gọi mô hình này là <b>SIMT</b>, single instruction, multiple thread (một lệnh, nhiều luồng), và đặt nó tương
            phản với SIMD trên CPU, nơi bạn hoặc trình biên dịch phải viết rõ độ rộng vector bằng intrinsic.
          </p>
        }
      >
        <p>
          Nothing in a kernel's source mentions any of this. You write scalar code for one thread; the grouping into warps
          and the use of the SIMD hardware happen underneath. Because the scheduler in effect holds all threads of a warp
          to the same instruction, the book calls the model <b>SIMT</b>, single instruction, multiple thread, and
          contrasts it with CPU SIMD, where you or the compiler must spell the vector width out with intrinsics.
        </p>
      </Translated>
      <Source section="4.4" pages="77" />
      <Translated
        vi={
          <p>
            NVIDIA cũng đặt ra sự tương phản đó và nói thêm cần bận tâm đến mức nào, bằng một phép so sánh bạn sẽ nhận ra: warp
            với kernel giống như cache line với mã CPU. Bạn có thể bỏ qua nó khi hỏi mã có đúng không. Bạn phải thiết kế theo
            nó khi hỏi mã nhanh đến đâu. Đó là lập trường của phần còn lại của chương này. Chú thích của sách cũng nói vậy:
            SIMT dễ viết hơn SIMD, và các luồng của một warp đi theo những hướng khác nhau vẫn khiến bạn trả giá, đó là nội
            dung phần tiếp theo.
          </p>
        }
      >
        <p>
          NVIDIA draws the same contrast and adds how much to care, with a comparison you will recognise: a warp is to a
          kernel what a cache line is to CPU code. You can ignore it when you ask whether the code is correct. You have to
          design for it when you ask how fast the code is. That is the stance of the rest of this chapter. The book's
          footnote says as much: SIMT is easier to write than SIMD, and threads of a warp that go different ways still
          cost you, which is the next section.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch04-facts.md A9)', '2026-11-20 (notes/today/ch04-facts.md, mục A9)')}>
        <Translated
          vi={
            <p>
              Nguyên văn của NVIDIA: “analogous to the role of cache lines: the cache line size can be safely ignored when
              designing for correctness but must be considered in the code structure when designing for peak performance.”
              (tương tự vai trò của cache line: kích thước cache line có thể bỏ qua an toàn khi thiết kế cho tính đúng đắn,
              nhưng phải được tính đến trong cấu trúc mã khi thiết kế cho hiệu năng đỉnh.)
            </p>
          }
        >
          <p>
            NVIDIA's words: “analogous to the role of cache lines: the cache line size can be safely ignored when
            designing for correctness but must be considered in the code structure when designing for peak performance.”
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Vì sao xây dựng theo cách này">Why build it this way</Translated></h2>
      <Translated
        vi={
          <p>
            Phần cứng điều khiển đắt đỏ: nạp và giải mã lệnh cần logic và truy cập bộ nhớ đệm lệnh. Dùng chung một đơn vị điều
            khiển cho nhiều đơn vị thực thi nghĩa là phần chip dành cho điều khiển nhỏ hơn và phần dành cho số học lớn hơn, đúng
            điều một cỗ máy thông lượng muốn. Sách dự đoán các thiết kế kiểu này sẽ vẫn là chuẩn mực vì lý do đó.
          </p>
        }
      >
        <p>
          Control hardware is expensive: fetching and decoding instructions needs logic and access to the instruction
          cache. Sharing one control unit among many execution units means a smaller share of the chip goes to control
          and a larger share to arithmetic, which is what a throughput machine wants. The book expects designs of this
          kind to remain the norm for that reason.
        </p>
      </Translated>
      <Source section="4.4" pages="77, 79" />
      <Translated
        vi={
          <p>
            Cùng sự tiết kiệm đó trên phần cứng AMD có tỉ lệ khác. Đơn vị gần nhất của AMD với một processing block là SIMD:
            một compute unit của dòng CDNA có bốn SIMD, mỗi SIMD rộng 16 lane, và một lệnh cho warp rộng 64 mất bốn chu kỳ, 16
            luồng mỗi chu kỳ. Một processing block như sách mô tả chạy cả warp trong một bước, nên hãy coi “processing block =
            SIMD” chỉ là sự tương ứng gần đúng.
          </p>
        }
      >
        <p>
          The same economy on AMD hardware has other proportions. AMD's nearest unit to a processing block is the SIMD: a
          compute unit of its CDNA family has four of them, each 16 lanes wide, and one instruction for a 64-wide warp
          takes four cycles, 16 threads per cycle. A processing block as the book describes it runs a whole warp in one
          step, so treat “processing block = SIMD” as a loose match.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Các con số lấy từ tài liệu HIP của AMD (notes/today/ch04-facts.md B1). Việc ghép hai đơn vị là của chúng tôi;
              AMD không đặt chúng cạnh nhau.
            </p>
          }
        >
          <p>
            The figures are from AMD's HIP documentation (notes/today/ch04-facts.md B1). The pairing of the two units is
            ours; AMD does not draw it.
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Các primitive cấp warp, chỉ nêu tên">Warp-level primitives, by name only</Translated></h2>
      <Translated
        vi={
          <p>
            Sách kết thúc phần này bằng việc gọi tên một họ hàm API cho phép các luồng của một warp trao đổi giá trị và đồng bộ
            hóa với nhau. Mã viết bằng chúng coi warp là một tầng riêng giữa khối và luồng. Chúng có thể đem lại lợi ích đáng
            kể, và chương 10, 11 và 12 là nơi sách dùng chúng; không có gì trong chương này cần đến chúng.
          </p>
        }
      >
        <p>
          The book ends the section by naming a family of API functions that let the threads of one warp exchange values
          and synchronise among themselves. Code written with them treats the warp as a level of its own between the
          block and the thread. They can pay off substantially, and chapters 10, 11 and 12 are where
          the book uses them; nothing in this chapter needs them.
        </p>
      </Translated>
      <Source section="4.4" pages="77–78" />
      <Translated
        vi={
          <p>
            Có một khác biệt đáng mang theo đến lúc đó, vì đó là chỗ “một file nguồn cho cả CUDA và HIP” không còn đúng. Các
            hàm này gọi tên lane bằng một bit mask. Trên CUDA, mask là một <code>unsigned</code> 32 bit. Trên HIP nó là 64 bit
            trên mọi thiết bị, kể cả thiết bị rộng 32, và header từ chối mask 32 bit lúc biên dịch. Các hàm SIMD-group của
            Metal trả về một đối tượng <code>simd_vote</code> thay vì một số nguyên thô.
          </p>
        }
      >
        <p>
          One difference is worth carrying until then, because it is where “one source file for CUDA and HIP” stops being
          true. These functions name lanes with a bit mask. On CUDA the mask is a 32-bit <code>unsigned</code>. On HIP it
          is 64 bits on every device, 32-wide ones included, and the header rejects a 32-bit mask at compile time. Metal's
          SIMD-group functions hand back a <code>simd_vote</code> object instead of a raw integer.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch04-facts.md A20, B12, C7)', '2026-11-20 (notes/today/ch04-facts.md, mục A20, B12, C7)')}>
        <Translated
          vi={
            <p>
              CUDA: mọi mask đều là <code>unsigned</code>. HIP: “All mask values returned or accepted by these built-ins are
              64-bit unsigned integer values, even when compiled for a device with 32 threads per warp.” (mọi giá trị mask mà
              các built-in này trả về hoặc nhận vào đều là số nguyên không dấu 64 bit, kể cả khi biên dịch cho thiết bị có 32
              luồng mỗi warp.) Metal: <code>simd_vote</code> bọc một kiểu số nguyên không được xác định.
            </p>
          }
        >
          <p>
            CUDA: all masks are <code>unsigned</code>. HIP: “All mask values returned or accepted by these built-ins are
            64-bit unsigned integer values, even when compiled for a device with 32 threads per warp.” Metal:{' '}
            <code>simd_vote</code> wraps an unspecified integer type.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Trên Metal, hãy hỏi; đừng tự tính">On Metal, ask; do not compute</Translated></h2>
      <Translated
        vi={
          <p>
            Mọi điều ở trên về cách cắt là quy tắc CUDA ghi trong tài liệu. Metal cho bạn cùng chỉ số luồng tuyến tính rồi nói
            việc chia thành SIMD-group là “implementation-defined” (do cài đặt quyết định). Vì vậy một kernel Metal không được
            tự tính nhóm của nó bằng chỉ số / độ rộng. Nó hỏi, bằng ba tham số kernel:{' '}
            <code>[[simdgroup_index_in_threadgroup]]</code>, <code>[[thread_index_in_simdgroup]]</code> và{' '}
            <code>[[threads_per_simdgroup]]</code>. Host đọc độ rộng từ pipeline đã biên dịch qua{' '}
            <code>threadExecutionWidth</code>.
          </p>
        }
      >
        <p>
          Everything above about the cut is CUDA's documented rule. Metal gives you the same linear thread index and then
          says the division into SIMD-groups is “implementation-defined”. A Metal kernel must therefore not work out its
          group as index / width. It asks, with three kernel parameters:{' '}
          <code>[[simdgroup_index_in_threadgroup]]</code>, <code>[[thread_index_in_simdgroup]]</code> and{' '}
          <code>[[threads_per_simdgroup]]</code>. The host reads the width from the compiled pipeline as{' '}
          <code>threadExecutionWidth</code>.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ dùng CUDA. Cách diễn đạt của Metal và ba thuộc tính lấy từ Metal Shading Language specification
              (notes/today/ch04-facts.md C2). Apple không dùng từ warp; phép so sánh là của chúng tôi.
            </p>
          }
        >
          <p>
            The book uses CUDA only. Metal's wording and the three attributes are from the Metal Shading Language
            specification (notes/today/ch04-facts.md C2). Apple does not use the word warp; the comparison is ours.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Lab 4.1 cho bạn viết một kernel thăm dò ghi lại ba giá trị đó cho mọi luồng của một threadgroup. Khi đã chạy nó, hãy
            chọn “Metal” trong trình tô ở trên: nó hiển thị những gì thiết bị của bạn đã làm, và không hiển thị gì nếu bạn chưa
            hỏi. Các lab nằm trên <a href={href('ch04', 'labs')}>trang lab</a>.
          </p>
        }
      >
        <p>
          Lab 4.1 has you write a probe kernel that records those three values for every thread of a threadgroup. Once
          you have run it, choose “Metal” in the painter above: it shows what your device did, and nothing if you have
          not asked it yet. The labs are on the <a href={href('ch04', 'labs')}>labs page</a>.
        </p>
      </Translated>

      <h2><Translated vi="Độ rộng xuất hiện ở đâu trong mã bạn sẽ đọc">Where the width appears in code you will read</Translated></h2>
      <Translated
        vi={
          <p>
            Cả ba dự án mà khóa học này chuẩn bị cho bạn đọc đều mang độ rộng trong mã nguồn, và cả ba đều cho thấy hai thói
            quen: một con số viết cứng ở file này, một truy vấn ở file kế bên.
          </p>
        }
      >
        <p>
          The three projects this course prepares you to read all carry the width in their source, and all three show
          both habits: a hard-coded number in one file, a query in the next.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Đọc từ nhánh mặc định của mỗi dự án ngày 2026-11-20 (notes/today/ch04-facts.md D1 đến D4). File có thể di chuyển;
              hãy tìm theo định danh.
            </p>
          }
        >
          <p>
            Read from each project's default branch on 2026-11-20 (notes/today/ch04-facts.md D1 to D4). Files move;
            search for the identifier.
          </p>
        </Translated>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th><Translated vi="Dự án">Project</Translated></th><th><Translated vi="Viết cứng">Hard-coded</Translated></th><th><Translated vi="Truy vấn">Asked for</Translated></th></tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="PyTorch, CUDA và ROCm">PyTorch, CUDA and ROCm</Translated></td>
            <td><Translated vi={<><code>#define C10_WARP_SIZE 32</code> trên bản dựng CUDA</>}><code>#define C10_WARP_SIZE 32</code> on CUDA builds</Translated></td>
            <td>
              <Translated
                vi={
                  <>
                    trên bản dựng ROCm, <code>C10_WARP_SIZE</code> là một hàm (64 cho mã device gfx9, 32 cho các trường hợp
                    khác), và mã host phải gọi <code>at::cuda::warp_size()</code>
                  </>
                }
              >
                on ROCm builds <code>C10_WARP_SIZE</code> is a function (64 for gfx9 device code, 32 otherwise), and host
                code must call <code>at::cuda::warp_size()</code>
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="llama.cpp, CUDA và HIP">llama.cpp, CUDA and HIP</Translated></td>
            <td><Translated vi={<><code>#define WARP_SIZE 32</code>, độ rộng mà hầu hết kernel được viết cho</>}><code>#define WARP_SIZE 32</code>, the width most kernels are written for</Translated></td>
            <td><Translated vi={<><code>ggml_cuda_get_physical_warp_size()</code>: 64 dưới HIP trên gfx9, ngược lại là 32</>}><code>ggml_cuda_get_physical_warp_size()</code>: 64 under HIP on gfx9, else 32</Translated></td>
          </tr>
          <tr>
            <td>llama.cpp, Metal</td>{/* i18n-ignore: product names */}
            <td><Translated vi={<><code>#define N_SIMDWIDTH 32</code>, kèm chú thích “assuming SIMD group size is 32”</>}><code>#define N_SIMDWIDTH 32</code>, commented “assuming SIMD group size is 32”</Translated></td>
            <td><Translated vi={<>nhóm và lane lấy từ <code>[[simdgroup_index_in_threadgroup]]</code> và <code>[[thread_index_in_simdgroup]]</code></>}>group and lane from <code>[[simdgroup_index_in_threadgroup]]</code> and <code>[[thread_index_in_simdgroup]]</code></Translated></td>
          </tr>
          <tr>
            <td>MLX, Metal</td>{/* i18n-ignore: product names */}
            <td><Translated vi={<><code>constexpr int SIMD_SIZE = 32;</code> trong kernel softmax của nó</>}><code>constexpr int SIMD_SIZE = 32;</code> in its softmax kernel</Translated></td>
            <td><Translated vi={<><code>uint simd_size [[threads_per_simdgroup]]</code> trong kernel arg-reduce của nó</>}><code>uint simd_size [[threads_per_simdgroup]]</code> in its arg-reduce kernel</Translated></td>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Một kernel biết hỏi thì đúng ở mọi độ rộng. Một kernel viết cứng 32 là kernel mà ai đó phải rà soát trước khi nó gặp
            một thiết bị rộng 64, và việc rà soát đó là một đóng góp đầu tiên thực tế cho bất kỳ dự án nào trong số này.
          </p>
        }
      >
        <p>
          A kernel that asks is correct at any width. A kernel that hard-codes 32 is a kernel somebody has to audit before
          it meets a 64-wide device, and that audit is a realistic first contribution to any of these projects.
        </p>
      </Translated>
      <Ask q={questionsS44['ch04/4.4/q-hardcode']} />
      <Translated vi={<p>Tiếp theo: điều gì xảy ra khi các luồng của một warp không thống nhất về lệnh kế tiếp.</p>}>
        <p>
          Next: what happens when the threads of one warp do not agree on the next instruction.
        </p>
      </Translated>
    </>
  );
}
