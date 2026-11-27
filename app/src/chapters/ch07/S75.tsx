import { useEffect, useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Aside, Bench, Beyond, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { href } from '../../lib/router';
import { SEGMENT_MODELS, type SegmentModelId } from '../../sim/coalesce';
import {
  CONV_CORRECT,
  FIG714_M,
  FIG714_T,
  attenuation,
  convRowAccess,
  fig714,
  flatLoadPass,
  gridOf,
  idealIntensity,
  makeProblem,
  organisation,
  outTileOf,
  runAll,
  runTiledBlock,
  stateAt,
  threadInfo,
  tiledIntensity,
  trafficRatioVsConst,
  type ConvConfig,
  type ConvStep,
  type Organisation,
  type RowAccess,
  type RowLayout,
} from '../../sim/convtile';
import { SCHEDULE_LABEL, type Schedule } from '../../sim/tiled';
import { LoadWaitComputeScene } from './LoadWaitComputeScene';

// ---------------------------------------------------------------- shared bits

const CONV_TIERS: Record<string, Tier> = {
  N: 'global', P: 'global', N_s: 'shared', ins: 'shared', F_c: 'const', F: 'const',
  Pvalue: 'reg', row: 'reg', col: 'reg', tx: 'reg', ty: 'reg', tileRow: 'reg', tileCol: 'reg', fRow: 'reg', fCol: 'reg',
};
const f2 = (v: number) => v.toFixed(2);
const count = (n: number) => n.toLocaleString('en-US');
const bytesText = (b: number) => (b >= 1024 ? `${count(b)} B (${fmt(b / 1024)} KiB)` : `${count(b)} B`); // i18n-ignore: numbers and units only
const positive = (v: unknown) => {
  const n = Number(v);
  return isFinite(n) && n > 0 ? n : undefined;
};
const pct = (x: number) => `${Math.round(100 * x)}%`;

/** The lines of a lab file between "// @part <name>" and the next part marker (or the end). */
function labPart(src: string | null, part: string): string | null {
  if (src == null) return null;
  const lines = src.split('\n');
  const start = lines.findIndex((l) => l.includes('@part ' + part));
  if (start < 0) return src;
  let end = lines.findIndex((l, i) => i > start && /@part\s|@end\b/.test(l));
  if (end < 0) end = lines.length;
  return lines.slice(start + 1, end).join('\n').replace(/\s+$/, '');
}

// ---------------------------------------------------------------- device choices for the explorer

interface Limits {
  key: string;
  label: string;
  threads?: number;
  shared?: number;
  note: string;
}
const BOOK_LIMITS: Limits = {
  key: 'book',
  get label() {
    return tr("The book's: 1024 threads per block", 'Của sách: 1024 luồng mỗi khối');
  },
  threads: 1024,
  get note() {
    return tr(
      'The book gives 1024 threads per block (pp. 165, 177) and no shared-memory limit per block in this chapter.',
      'Sách đưa ra 1024 luồng mỗi khối (tr. 165, 177) và không nêu giới hạn bộ nhớ chia sẻ cho mỗi khối trong chương này.',
    );
  },
};
function reportedLimits(reports: Reports): Limits[] {
  return realBackends(reports).flatMap((b) => {
    const d = reports[b].device;
    const name = String(d.name ?? b);
    if (b === 'metal' || d.backend === 'metal')
      return [{
        key: b,
        get label() {
          return tr(`Yours: ${name} (${b})`, `Của bạn: ${name} (${b})`);
        },
        threads: positive(d['maxThreadsPerThreadgroup.width']),
        shared: positive(d.maxThreadgroupMemoryLength),
        get note() {
          return tr(
            'maxThreadsPerThreadgroup is per dimension and for a trivial shader; a compiled kernel may allow fewer threads in total (its pipeline’s maxTotalThreadsPerThreadgroup, which lab 7.2 checks). maxThreadgroupMemoryLength is the threadgroup-memory limit.',
            'maxThreadsPerThreadgroup tính theo từng chiều và cho một shader tầm thường; một kernel đã biên dịch có thể cho phép tổng số luồng ít hơn (maxTotalThreadsPerThreadgroup của pipeline, thứ mà lab 7.2 kiểm tra). maxThreadgroupMemoryLength là giới hạn bộ nhớ threadgroup.',
          );
        },
      }];
    return [{
      key: b,
      get label() {
        return tr(`Yours: ${name} (${b})`, `Của bạn: ${name} (${b})`);
      },
      threads: positive(d.maxThreadsPerBlock),
      shared: positive(d.sharedMemPerBlock),
      get note() {
        return tr(
          'maxThreadsPerBlock and sharedMemPerBlock as your device reported them to python3 labs/pmpp.py doctor.',
          'maxThreadsPerBlock và sharedMemPerBlock như thiết bị của bạn đã báo cho python3 labs/pmpp.py doctor.',
        );
      },
    }];
  });
}

interface Threshold {
  key: string;
  label: string;
  value: number;
}
function reportedThresholds(reports: Reports): Threshold[] {
  return realBackends(reports).flatMap((b) => {
    const r = benchRows(reports[b], 'ch05/lab1_roofline').find((x) => x.name === 'roofline');
    const v = num(r?.threshold_flop_per_byte);
    return v ? [{ key: b, get label() { return tr(`Yours, from lab 5.1 on ${String(reports[b].device.name ?? b)}: ${fmt(v)} FLOP/B`, `Của bạn, từ lab 5.1 trên ${String(reports[b].device.name ?? b)}: ${fmt(v)} FLOP/B`); }, value: v }] : [];
  });
}

// ---------------------------------------------------------------- explorer: tile size, filter size, intensity

function TilePicture({ which, t, r }: { which: Organisation; t: number; r: number }) {
  const IN = t + 2 * r;
  const S = 260;
  const c = S / Math.max(IN, 1);
  const grid = IN <= 40;
  const pass = which === 2 ? flatLoadPass(t, r) : [];
  const maxPass = pass.length ? Math.max(...pass) : 0;
  const passOpacity = (k: number) => (maxPass === 0 ? 0.18 : 0.14 + (0.5 * k) / maxPass);
  return (
    <>
    <svg viewBox={`0 0 ${S + 2} ${S + 2}`} role="img" aria-label={tr('Input tile with its halo and the output tile', 'Ô đầu vào cùng vùng biên của nó và ô đầu ra')} style={{ width: '100%', maxWidth: 300, display: 'block' }}>
      {grid ? (
        Array.from({ length: IN * IN }, (_, i) => {
          const y = Math.floor(i / IN);
          const x = i % IN;
          const halo = y < r || x < r || y >= r + t || x >= r + t;
          const fill = which === 1 ? (halo ? 'var(--cell)' : 'var(--shared-wash)') : 'var(--shared)';
          return (
            <rect key={i} x={1 + x * c} y={1 + y * c} width={Math.max(0.5, c - (c > 6 ? 1.5 : 0.6))} height={Math.max(0.5, c - (c > 6 ? 1.5 : 0.6))}
              fill={fill} fillOpacity={which === 2 ? passOpacity(pass[i]) : 1}
              stroke={which === 1 && halo ? 'var(--shared)' : 'none'} strokeWidth="0.6" strokeDasharray={halo && c > 10 ? '2 1.5' : undefined} />
          );
        })
      ) : (
        <>
          <rect x={1} y={1} width={IN * c} height={IN * c} fill={which === 1 ? 'var(--cell)' : 'var(--shared)'} fillOpacity={which === 1 ? 1 : 0.4}
            stroke="var(--shared)" strokeDasharray="4 3" />
          <rect x={1 + r * c} y={1 + r * c} width={t * c} height={t * c} fill={which === 1 ? 'var(--shared-wash)' : 'var(--shared)'} fillOpacity={which === 1 ? 1 : 0.2} />
        </>
      )}
      {t > 0 ? <rect x={1 + r * c} y={1 + r * c} width={t * c} height={t * c} fill="none" stroke="var(--ink)" strokeWidth="2" /> : null}
    </svg>
    <p className="muted" style={{ margin: '0.4rem 0 0', fontSize: '0.86em', maxWidth: 300 }} data-k="picture">
      {which === 1
        ? tr(`The block is the ${IN} × ${IN} input tile, one thread per element. Inside the outline: the ${t} × ${t} output tile; the threads of the outer ring only load.`, `Khối chính là ô đầu vào ${IN} × ${IN}, mỗi phần tử một luồng. Bên trong đường viền: ô đầu ra ${t} × ${t}; các luồng ở vòng ngoài chỉ nạp.`)
        : tr(`The block is the ${t} × ${t} output tile. It loads the ${IN} × ${IN} input tile in ${maxPass + 1} ${maxPass ? 'passes' : 'pass'} of a flat loop; darker cells come in later passes.`, `Khối chính là ô đầu ra ${t} × ${t}. Nó nạp ô đầu vào ${IN} × ${IN} trong ${maxPass + 1} lượt của một vòng lặp phẳng; các ô đậm hơn được nạp ở các lượt sau.`)}
    </p>
    </>
  );
}

function TileExplorer() {
  const { reports } = useReports();
  const [which, setWhich] = useState<Organisation>(1);
  const [inEdge, setInEdge] = useState(32);
  const [outEdge, setOutEdge] = useState(28);
  const [r, setR] = useState(2);
  const [bytes, setBytes] = useState(4);
  const [thKey, setThKey] = useState('book');
  const [custom, setCustom] = useState('20');
  const [devKey, setDevKey] = useState('book');

  const thresholds: Threshold[] = useMemo(
    () => [
      {
        key: 'book',
        get label() {
          return tr("The book's example: 20 FLOP/B", 'Ví dụ của sách: 20 FLOP/B');
        },
        value: 20,
      },
      ...reportedThresholds(reports),
    ],
    [reports],
  );
  const devices = useMemo(() => [BOOK_LIMITS, ...reportedLimits(reports)], [reports]);
  const dev = devices.find((d) => d.key === devKey) ?? BOOK_LIMITS;
  const customValue = Number(custom);
  const threshold = thKey === 'custom' ? (isFinite(customValue) && customValue > 0 ? customValue : 20) : (thresholds.find((x) => x.key === thKey)?.value ?? 20);

  const t = which === 1 ? Math.max(0, inEdge - 2 * r) : outEdge;
  const m = 2 * r + 1;
  const o = organisation(t, r, which, bytes);
  const ai = tiledIntensity(t, m, bytes);
  const ideal = idealIntensity(m, bytes);
  const table = fig714(bytes);
  const fits = (need: number, cap?: number) => (cap == null ? null : need <= cap);
  const threadsOk = fits(o.threads, dev.threads);
  const sharedOk = fits(o.sharedBytes, dev.shared);
  const bound = (v: number) => (v >= threshold ? 'compute' : 'memory');

  const switchOrg = (w: Organisation) => {
    if (w === which) return;
    if (w === 2) setOutEdge(Math.max(1, Math.min(64, t)));
    else setInEdge(Math.max(1, Math.min(80, outEdge + 2 * r)));
    setWhich(w);
  };
  const pick = (tt: number, mm: number) => {
    const rr = (mm - 1) / 2;
    setR(rr);
    if (which === 1) setInEdge(tt + mm - 1);
    else setOutEdge(tt);
  };

  return (
    <Bench
      title={tr('Tile size, filter size, intensity', 'Kích thước ô, kích thước bộ lọc, cường độ')}
      task={tr(
        'How large must the output tile be to come within 10% of the ideal for a 5 × 5 filter, and can a block of 1024 threads do it? Then compare the two organisations for a 9 × 9 filter.',
        'Ô đầu ra phải lớn đến đâu để đạt tới trong vòng 10% so với mức lý tưởng cho một bộ lọc 5 × 5, và một khối 1024 luồng có làm được không? Rồi so sánh hai cách tổ chức cho một bộ lọc 9 × 9.',
      )}
      foot={
        <Translated
          vi={
            <>
              Mọi con số đều đến từ công thức ở tr. 176 (sim/convtile.ts), cho một khối nằm xa các cạnh của ảnh. Bảng là
              Hình 7.14 được tính lại; với 4 byte mỗi phần tử, nó khớp mọi giá trị được in. Nhấn vào một ô để nạp các kích
              thước của nó. <TierLegend tiers={['shared', 'global', 'reg']} />
            </>
          }
        >
          <>
            Every number comes from the formula on p. 176 (sim/convtile.ts), for a block away from the image&apos;s edges.
            The table is Fig. 7.14 recomputed; at 4 bytes per element it matches every printed value. Click a cell to load
            its sizes. <TierLegend tiers={['shared', 'global', 'reg']} />
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label={tr('Organisation', 'Cách tổ chức')}>
          <button aria-pressed={which === 1} onClick={() => switchOrg(1)}><Translated vi="Khối = ô đầu vào (Hình 7.12)">Block = input tile (Fig. 7.12)</Translated></button>
          <button aria-pressed={which === 2} onClick={() => switchOrg(2)}><Translated vi="Khối = ô đầu ra (Bài tập 11)">Block = output tile (Ex. 11)</Translated></button>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        {which === 1 ? (
          <label className="field">
            <Translated vi={<>Cạnh ô đầu vào: <b>{inEdge}</b></>}>Input tile edge: <b>{inEdge}</b></Translated>
            <input type="range" min={1} max={80} value={inEdge} aria-label={tr('input tile edge', 'cạnh ô đầu vào')} onChange={(e) => setInEdge(+e.target.value)} />
          </label>
        ) : (
          <label className="field">
            <Translated vi={<>Cạnh ô đầu ra: <b>{outEdge}</b></>}>Output tile edge: <b>{outEdge}</b></Translated>
            <input type="range" min={1} max={64} value={outEdge} aria-label={tr('output tile edge', 'cạnh ô đầu ra')} onChange={(e) => setOutEdge(+e.target.value)} />
          </label>
        )}
        <label className="field">
          <Translated vi={<>Bán kính bộ lọc: <b>{r}</b> ({m} × {m})</>}>Filter radius: <b>{r}</b> ({m} × {m})</Translated>
          <input type="range" min={0} max={7} value={r} aria-label={tr('filter radius', 'bán kính bộ lọc')} onChange={(e) => setR(+e.target.value)} />
        </label>
        <label className="field">
          <Translated vi="Số byte mỗi phần tử">Bytes per element</Translated>
          <select value={bytes} onChange={(e) => setBytes(+e.target.value)} aria-label={tr('bytes per element', 'số byte mỗi phần tử')}>
            <option value={4}>{tr('4 (float, the book)', '4 (float, như sách)')}</option>
            <option value={2}>{tr('2 (half or bfloat16)', '2 (half hoặc bfloat16)')}</option>
            <option value={1}>{tr('1 (8-bit)', '1 (số 8-bit)')}</option>
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          <Translated vi="Ngưỡng">Threshold</Translated>
          <select value={thKey} onChange={(e) => setThKey(e.target.value)} aria-label={tr('threshold', 'ngưỡng')}>
            {thresholds.map((x) => <option key={x.key} value={x.key}>{x.label}</option>)}
            <option value="custom">{tr('Custom', 'Tùy chỉnh')}</option>
          </select>
        </label>
        {thKey === 'custom' ? (
          <label className="field">
            FLOP/B
            <input type="number" min={0.1} step="any" value={custom} aria-label={tr('custom threshold', 'ngưỡng tùy chỉnh')} onChange={(e) => setCustom(e.target.value)} style={{ width: '6rem' }} />
          </label>
        ) : null}
        <label className="field">
          <Translated vi="Giới hạn">Limits</Translated>
          <select value={devKey} onChange={(e) => setDevKey(e.target.value)} aria-label={tr('device limits', 'giới hạn thiết bị')}>
            {devices.map((d) => <option key={d.key} value={d.key}>{d.label}</option>)}
          </select>
        </label>
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 3fr)' }}>
        <div>
          <TilePicture which={which} t={t} r={r} />
        </div>
        <div>
          {t <= 0 ? (
            <p className="narrate bad" data-readout="none">
              <Translated
                vi={
                  <>
                    Không có ô đầu ra: một ô đầu vào rộng {inEdge} không chứa đầu ra nào với bán kính {r}. Nó phải rộng hơn{' '}
                    {2 * r}.
                  </>
                }
              >
                No output tile: an input tile {inEdge} wide holds no output for a radius of {r}. It must be wider than{' '}
                {2 * r}.
              </Translated>
            </p>
          ) : (
            <table className="data" style={{ margin: 0 }} data-readout="tile">
              <tbody>
                <tr><td><Translated vi="Ô đầu ra">Output tile</Translated></td><td data-k="out"><Translated vi={<>{t} × {t} = {count(o.outElems)} đầu ra</>}>{t} × {t} = {count(o.outElems)} outputs</Translated></td></tr>
                <tr><td><Translated vi="Ô đầu vào">Input tile</Translated></td><td data-k="in"><Translated vi={<>{o.inTile} × {o.inTile} = {count(o.inElems)} đầu vào ({count(o.haloElems)} trong vùng biên)</>}>{o.inTile} × {o.inTile} = {count(o.inElems)} inputs ({count(o.haloElems)} in the halo)</Translated></td></tr>
                <tr><td><Translated vi="Số đầu vào mỗi đầu ra">Inputs per output</Translated></td><td data-k="ratio">{fmt(o.ratio, 4)}</td></tr>
                <tr>
                  <td><Translated vi="Cường độ (tr. 176)">Intensity (p. 176)</Translated></td>
                  <td>
                    <b data-k="ai">{f2(ai)}</b> FLOP/B{' '}
                    {bound(ai) === 'memory' ? <span className="chip global"><Translated vi="bị giới hạn bởi bộ nhớ">memory-bound</Translated></span> : <span className="chip reg"><Translated vi="bị giới hạn bởi tính toán">compute-bound</Translated></span>}{' '}
                    <span className="muted"><Translated vi={<>tại {fmt(threshold)} FLOP/B</>}>at {fmt(threshold)} FLOP/B</Translated></span>
                  </td>
                </tr>
                <tr>
                  <td><Translated vi="Lý tưởng và hệ số suy giảm">Ideal and attenuation</Translated></td>
                  <td>
                    <Translated
                      vi={
                        <>
                          lý tưởng {m}² / {bytes} = <span data-k="ideal">{f2(ideal)}</span>; hệ số <span data-k="att">{f2(attenuation(t, m))}</span>{' '}
                          ({pct(ai / ideal)} mức lý tưởng)
                        </>
                      }
                    >
                      ideal {m}² / {bytes} = <span data-k="ideal">{f2(ideal)}</span>; factor <span data-k="att">{f2(attenuation(t, m))}</span>{' '}
                      ({pct(ai / ideal)} of the ideal)
                    </Translated>
                  </td>
                </tr>
                <tr>
                  <td><Translated vi="Số luồng mỗi khối">Threads per block</Translated></td>
                  <td data-k="threads">
                    {count(o.threads)}{' '}
                    {threadsOk == null ? null : threadsOk ? <span className="chip ok"><Translated vi="vừa">fits</Translated></span> : <span className="chip fail"><Translated vi={<>vượt {count(dev.threads!)}</>}>over {count(dev.threads!)}</Translated></span>}
                  </td>
                </tr>
                <tr>
                  <td><Translated vi="Bộ nhớ chia sẻ mỗi khối">Shared memory per block</Translated></td>
                  <td data-k="shared">
                    {bytesText(o.sharedBytes)}{' '}
                    {sharedOk == null ? <span className="muted"><Translated vi="(không có giới hạn cho lựa chọn này)">(no limit for this choice)</Translated></span> : sharedOk ? <span className="chip ok"><Translated vi="vừa">fits</Translated></span> : <span className="chip fail"><Translated vi={<>vượt {bytesText(dev.shared!)}</>}>over {bytesText(dev.shared!)}</Translated></span>}
                  </td>
                </tr>
                {which === 1 ? (
                  <tr>
                    <td><Translated vi="Rảnh khi tính toán">Idle while computing</Translated></td>
                    <td data-k="idle"><Translated vi={<>{count(o.idleCompute)} trên {count(o.threads)} luồng ({pct(o.threads ? o.idleCompute / o.threads : 0)})</>}>{count(o.idleCompute)} of {count(o.threads)} threads ({pct(o.threads ? o.idleCompute / o.threads : 0)})</Translated></td>
                  </tr>
                ) : (
                  <tr>
                    <td><Translated vi="Số lần nạp mỗi luồng">Loads per thread</Translated></td>
                    <td data-k="loads">
                      <Translated vi={<>tối đa {o.loadItersFlat} với một vòng lặp phẳng; {o.loadIters2D} với một vòng lặp bước nhảy {t} theo mỗi chiều</>}>
                        at most {o.loadItersFlat} with a flat loop; {o.loadIters2D} with a loop striding {t} in each direction
                      </Translated>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          )}
          <p className="muted" style={{ margin: '0.6rem 0 0', fontSize: '0.86em' }}>{dev.note}</p>
        </div>
      </div>

      <div style={{ marginTop: '1rem', overflowX: 'auto' }}>
        <div className="mx-label">
          <Translated vi={<>Hình 7.14 với {bytes} byte mỗi phần tử, tô màu tại {fmt(threshold)} FLOP/B</>}>
            Fig. 7.14 at {bytes} bytes per element, coloured at {fmt(threshold)} FLOP/B
          </Translated>
        </div>
        <table className="data" aria-label={tr('Fig. 7.14 recomputed', 'Hình 7.14 được tính lại')} style={{ margin: 0, background: 'var(--page)' }}>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.5rem' }}><Translated vi="bộ lọc \ t">filter \ t</Translated></th>
              {FIG714_T.map((tt) => <th key={tt} style={{ textAlign: 'center' }}>{tt}</th>)}
              <th style={{ textAlign: 'center' }}><Translated vi="lý tưởng">ideal</Translated></th>
            </tr>
          </thead>
          <tbody>
            {table.map((row) => (
              <tr key={row.m}>
                <td style={{ paddingLeft: '0.5rem' }}>{row.m} × {row.m}</td>
                {row.values.map((v, j) => {
                  const here = FIG714_T[j] === t && row.m === m;
                  return (
                    <td key={j} data-m={row.m} data-t={FIG714_T[j]} data-bound={bound(v)} style={{ padding: 2 }}>
                      <button
                        className="quiet"
                        onClick={() => pick(FIG714_T[j], row.m)}
                        aria-label={tr(`t ${FIG714_T[j]}, filter ${row.m}: ${f2(v)}`, `t ${FIG714_T[j]}, bộ lọc ${row.m}: ${f2(v)}`)}
                        style={{
                          width: '100%',
                          background: bound(v) === 'memory' ? 'var(--global-wash)' : 'var(--reg-wash)',
                          outline: here ? '2px solid var(--ink)' : undefined,
                          fontVariantNumeric: 'tabular-nums',
                        }}
                      >
                        {f2(v)}
                      </button>
                    </td>
                  );
                })}
                <td data-m={row.m} data-t="ideal" data-bound={bound(row.ideal)} style={{ padding: 2 }}>
                  <span className={'chip ' + (bound(row.ideal) === 'memory' ? 'global' : 'reg')} style={{ display: 'block', textAlign: 'center' }}>{f2(row.ideal)}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted" style={{ margin: '0.4rem 0 0', fontSize: '0.86em' }}>
          <Translated
            vi={
              <>
                <span className="chip global">bị giới hạn bởi bộ nhớ</span> dưới ngưỡng, <span className="chip reg">bị giới hạn bởi tính toán</span> tại hoặc trên ngưỡng.
                {(FIG714_M as readonly number[]).includes(m) && (FIG714_T as readonly number[]).includes(t) ? '' : ' Các kích thước hiện tại không phải một ô của bảng.'}
              </>
            }
          >
            <span className="chip global">memory-bound</span> below the threshold, <span className="chip reg">compute-bound</span> at or above it.
            {(FIG714_M as readonly number[]).includes(m) && (FIG714_T as readonly number[]).includes(t) ? '' : ' The current sizes are not a cell of the table.'}
          </Translated>
        </p>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- one warp's row: coalescing view

function RowStrip({ access, label, seg, op }: { access: RowAccess; label: string; seg: number; op: 'load' | 'store' }) {
  const px = 2.4; // pixels per byte
  return (
    <div style={{ marginBottom: '0.6rem' }}>
      <div className="mx-label">{label}</div>
      {access.rows.map((row, k) => {
        if (!row.lanes) return <p key={k} className="muted" style={{ margin: 0 }}><Translated vi="không có lane nào nằm trong ảnh">no lane is inside the image</Translated></p>;
        const lo = row.startBytes;
        const hi = row.startBytes + row.lanes * 4;
        const s0 = Math.floor(lo / seg);
        const s1 = Math.floor((hi - 1) / seg);
        const base = s0 * seg;
        const w = (s1 - s0 + 1) * seg * px;
        return (
          <svg key={k} viewBox={`0 0 ${w + 2} 34`} style={{ width: w + 2, maxWidth: '100%', display: 'block', marginBottom: 2 }}
            role="img" aria-label={tr(`${s1 - s0 + 1} segments`, `${s1 - s0 + 1} đoạn`)}>
            {Array.from({ length: s1 - s0 + 1 }, (_, i) => (
              <rect key={i} x={1 + i * seg * px} y={2} width={seg * px} height={22} fill="var(--cell)" stroke="var(--ink-3)" />
            ))}
            <rect x={1 + (lo - base) * px} y={8} width={(hi - lo) * px} height={10} fill={op === 'load' ? 'var(--global-wash)' : 'var(--global)'} stroke="var(--global)" />
            {access.rows.length > 1 ? (
              <text x={4} y={33} fontSize="8.5" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>hàng khối {k}</>}>block row {k}</Translated></text>
            ) : null}
          </svg>
        );
      })}
      <div className="muted" style={{ fontSize: '0.86em' }} data-k={label}>
        <Translated
          vi={
            <>
              <b>{access.transactions}</b> đoạn ({access.bytesFetched} B) cho {access.bytesNeeded} B được yêu cầu:{' '}
              dùng {pct(access.efficiency)}
            </>
          }
        >
          <b>{access.transactions}</b> {access.transactions === 1 ? 'segment' : 'segments'} ({access.bytesFetched} B) for {access.bytesNeeded} B requested:{' '}
          {pct(access.efficiency)} used
        </Translated>
      </div>
    </div>
  );
}

function RowCoalescing() {
  const [unit, setUnit] = useState<SegmentModelId>('nvidia');
  const [warp, setWarp] = useState(32);
  const [bx, setBx] = useState(1);
  const [r, setR] = useState(2);
  const [width, setWidth] = useState('1024');
  const [row, setRow] = useState(1);
  const model = SEGMENT_MODELS[unit];
  const W = Math.max(64, Math.floor(Number(width)) || 1024);
  const seg = model.segmentBytes;
  const acc = (layout: RowLayout, op: 'load' | 'store') =>
    seg ? convRowAccess({ layout, op, bx, r, width: W, row, warpSize: warp, segmentBytes: seg }) : null;
  const out = 32 - 2 * r;
  const cases: { layout: RowLayout; op: 'load' | 'store'; label: string }[] = [
    { layout: 'fig712', op: 'load', label: tr(`Fig. 7.12, load: input-tile row, 32 wide, from column ${bx * out - r}`, `Hình 7.12, nạp: hàng của ô đầu vào, rộng 32, từ cột ${bx * out - r}`) },
    { layout: 'fig712', op: 'store', label: tr(`Fig. 7.12, store: output row, ${out} wide, from column ${bx * out}`, `Hình 7.12, ghi: hàng đầu ra, rộng ${out}, từ cột ${bx * out}`) },
    { layout: 'tile32', op: 'load', label: tr(`32-wide tile, load: from column ${bx * 32}`, `ô rộng 32, nạp: từ cột ${bx * 32}`) },
    { layout: 'tile32', op: 'store', label: tr(`32-wide tile, store: from column ${bx * 32}`, `ô rộng 32, ghi: từ cột ${bx * 32}`) },
  ];
  return (
    <Bench
      title={tr('One warp, one tile row', 'Một warp, một hàng của ô')}
      task={tr(
        'Predict how many segments a warp needs to store a 28-float row. Then move the block column and change the image width.',
        'Dự đoán một warp cần bao nhiêu đoạn để ghi một hàng 28 float. Rồi dịch cột khối và đổi chiều rộng ảnh.',
      )}
      foot={
        <Translated
          vi={<>Các con số đến từ sim/convtile.ts (convRowAccess) và quy tắc đoạn trong sim/coalesce.ts. Thanh nhạt là lần nạp, thanh đậm là lần ghi.</>}
        >
          <>Counts come from sim/convtile.ts (convRowAccess) and the segment rule in sim/coalesce.ts. Pale bars are loads, solid bars stores.</>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Quy tắc đoạn">Segment rule</Translated>
          <select value={unit} onChange={(e) => setUnit(e.target.value as SegmentModelId)} aria-label={tr('segment rule', 'quy tắc đoạn')}>
            {(Object.keys(SEGMENT_MODELS) as SegmentModelId[]).map((k) => <option key={k} value={k}>{SEGMENT_MODELS[k].label}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <select value={warp} onChange={(e) => setWarp(+e.target.value)} aria-label={tr('warp width', 'độ rộng warp')}>
            <option value={32}>{tr('32 (NVIDIA; AMD RDNA)', '32 (NVIDIA; AMD dòng RDNA)')}</option>
            <option value={64}>{tr('64 (AMD CDNA)', '64 (AMD dòng CDNA)')}</option>
          </select>
        </label>
        <label className="field">
          <Translated vi={<>Cột khối bx: <b>{bx}</b></>}>Block column bx: <b>{bx}</b></Translated>
          <input type="range" min={0} max={7} value={bx} aria-label={tr('block column', 'cột khối')} onChange={(e) => setBx(+e.target.value)} />
        </label>
        <label className="field">
          <Translated vi={<>Bán kính: <b>{r}</b></>}>Radius: <b>{r}</b></Translated>
          <input type="range" min={1} max={7} value={r} aria-label={tr('radius', 'bán kính')} onChange={(e) => setR(+e.target.value)} />
        </label>
        <label className="field">
          <Translated vi="Chiều rộng ảnh (số float)">Image width (floats)</Translated>
          <input type="number" min={64} value={width} aria-label={tr('image width', 'chiều rộng ảnh')} onChange={(e) => setWidth(e.target.value)} style={{ width: '6rem' }} />
        </label>
        <label className="field">
          <Translated vi={<>Hàng ảnh: <b>{row}</b></>}>Image row: <b>{row}</b></Translated>
          <input type="range" min={0} max={7} value={row} aria-label={tr('image row', 'hàng ảnh')} onChange={(e) => setRow(+e.target.value)} />
        </label>
      </div>
      {seg == null ? (
        <p className="narrate" data-k="no-rule">
          <Translated
            vi={
              <>
                Apple không ghi tài liệu về quy tắc hợp nhất truy cập bộ nhớ (coalescing), kích thước line hay kích thước
                sector cho GPU của mình (notes/today/ch06-facts.md, C1), nên ở đây không có gì để đếm. Trên Metal, lab
                coalescing của chương 6 đo thay vì dự đoán.
              </>
            }
          >
            Apple documents no coalescing rule, line size or sector size for its GPUs (notes/today/ch06-facts.md, C1), so
            there is nothing here to count. On Metal the coalescing lab of chapter 6 measures instead of predicting.
          </Translated>
        </p>
      ) : (
        <>
          <p className="muted" style={{ marginTop: 0 }}>
            <Translated
              vi={
                <>
                  Hàng {row} của ảnh bắt đầu tại byte {count(row * W * 4)}, tức là{' '}
                  {(row * W * 4) % seg === 0 ? `một bội số của ${seg}` : `${(row * W * 4) % seg} byte sau một ranh giới ${seg} byte`}.
                  {warp === 64 ? ' Một warp rộng 64 phủ hai hàng của một khối rộng 32, cách nhau một hàng ảnh, nên mỗi lần truy cập hiện hai dải.' : ''}
                </>
              }
            >
              Row {row} of the image starts at byte {count(row * W * 4)}, which is{' '}
              {(row * W * 4) % seg === 0 ? `a multiple of ${seg}` : `${(row * W * 4) % seg} bytes past a ${seg}-byte boundary`}.
              {warp === 64 ? ' A 64-wide warp covers two rows of a 32-wide block, an image row apart, so each access shows two strips.' : ''}
            </Translated>
          </p>
          <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
            <div>
              {cases.slice(0, 2).map((c) => <RowStrip key={c.label} access={acc(c.layout, c.op)!} label={c.label} seg={seg} op={c.op} />)}
            </div>
            <div>
              {cases.slice(2).map((c) => <RowStrip key={c.label} access={acc(c.layout, c.op)!} label={c.label} seg={seg} op={c.op} />)}
            </div>
          </div>
        </>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- stepper: load, wait, compute

// labs/ch07/snippets/s75_halo_toy.cu, front-end-checked by ./verify.sh. S75.snippets.test.ts
// runs it on the emulator (labs/ch07/snippets/check/s75_halo_toy_test.cu), and fails if this
// copy drifts from the file or if the line numbers in LINE stop pointing at the lines they name.
const HALO_TOY = `#define FILTER_RADIUS 1
#define IN_TILE_DIM 8
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))
__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

__global__ void convolution_tiled_2D_const_mem_kernel(
        const float* N, float* P, int width, int height) {
    int tx = threadIdx.x, ty = threadIdx.y;   // signed from here on
    int col = blockIdx.x*OUT_TILE_DIM + tx - FILTER_RADIUS;
    int row = blockIdx.y*OUT_TILE_DIM + ty - FILTER_RADIUS;
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
    if (row >= 0 && row < height && col >= 0 && col < width) {
        N_s[ty][tx] = N[row*width + col];
    } else {
        N_s[ty][tx] = 0.0f;                   // ghost cell
    }
    __syncthreads();
    int tileCol = tx - FILTER_RADIUS;
    int tileRow = ty - FILTER_RADIUS;
    if (col >= 0 && col < width && row >= 0 && row < height) {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM &&
            tileRow >= 0 && tileRow < OUT_TILE_DIM) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
                for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {
                    Pvalue += F_c[fRow][fCol]
                              * N_s[tileRow+fRow][tileCol+fCol];
                }
            }
            P[row*width + col] = Pvalue;
        }
    }
}

// Not in the book, which shows no host code: one block per output tile.
void launchHaloToy(const float* N_d, float* P_d, int width, int height) {
    dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM);   // the block is the input tile
    dim3 dimGrid((width + OUT_TILE_DIM - 1)/OUT_TILE_DIM,
                 (height + OUT_TILE_DIM - 1)/OUT_TILE_DIM);
    convolution_tiled_2D_const_mem_kernel<<<dimGrid, dimBlock>>>(N_d, P_d, width, height);
}`;
export const LINE = { loadIn: 13, test: 12, ghost: 15, barrier: 17, outside: 20, ring: 21, tap: 26, store: 30 };
/** The stepper shows the kernel only, lines 1 to KERNEL_END of the file. */
export const KERNEL_END = 33;

function lineOf(s: ConvStep | undefined, cfg: ConvConfig): number[] {
  if (!s) return [];
  switch (s.op) {
    case 'load':
      return s.global!.inside ? [LINE.loadIn] : cfg.zeroGhost ? [LINE.ghost] : [LINE.test];
    case 'wait':
    case 'release':
      return [LINE.barrier];
    case 'idle':
      return s.reason === 'outside' ? [LINE.outside] : [LINE.ring, LINE.ring + 1];
    case 'tap':
      return [LINE.tap, LINE.tap + 1];
    case 'store':
      return [LINE.store];
  }
}

const IMAGES = [10, 14];
const SCHEDULES: Schedule[] = ['lockstep', 'in-order', 'reversed', 'shuffled'];
const show = (v: number) => (Number.isNaN(v) ? '?' : String(v));
const tiny = { minWidth: '1.45rem', height: '1.35rem', fontSize: '0.68rem' } as const;

function describe(s: ConvStep, cfg: ConvConfig): string {
  if (s.thread < 0) return tr('The barrier opens: every thread of the block has stored its slot. From here on, N_s is only read.', 'Rào chắn mở: mọi luồng của khối đã ghi ô của mình. Từ đây trở đi, N_s chỉ được đọc.');
  const who = tr(`Thread (tx ${s.tx}, ty ${s.ty})`, `Luồng (tx ${s.tx}, ty ${s.ty})`);
  switch (s.op) {
    case 'load': {
      const g = s.global!;
      if (g.inside) return tr(`${who} loads N[${g.row}][${g.col}] = ${s.write!.value} into N_s[${s.ty}][${s.tx}].`, `${who} nạp N[${g.row}][${g.col}] = ${s.write!.value} vào N_s[${s.ty}][${s.tx}].`);
      return cfg.zeroGhost
        ? tr(`${who}: (${g.row}, ${g.col}) is outside the image, a ghost cell. It stores 0 into N_s[${s.ty}][${s.tx}].`, `${who}: (${g.row}, ${g.col}) nằm ngoài ảnh, là một ô ma. Nó ghi 0 vào N_s[${s.ty}][${s.tx}].`)
        : tr(`${who}: (${g.row}, ${g.col}) is outside the image, and without the else branch it stores nothing. N_s[${s.ty}][${s.tx}] keeps whatever it held.`, `${who}: (${g.row}, ${g.col}) nằm ngoài ảnh, và khi không có nhánh else thì nó không ghi gì. N_s[${s.ty}][${s.tx}] giữ nguyên giá trị cũ, bất kể là gì.`);
    }
    case 'wait':
      return tr(`${who} reaches __syncthreads() and waits for the rest of the block.`, `${who} đến __syncthreads() và chờ phần còn lại của khối.`);
    case 'idle':
      return s.reason === 'outside'
        ? tr(`${who} has no output: its (row, col) is outside the image. It returns.`, `${who} không có đầu ra: (row, col) của nó nằm ngoài ảnh. Nó kết thúc.`)
        : tr(`${who} is on the outer ring (tileRow or tileCol outside 0 to ${outTileOf(cfg) - 1}). It loaded, and now it returns.`, `${who} nằm ở vòng ngoài (tileRow hoặc tileCol ngoài khoảng 0 đến ${outTileOf(cfg) - 1}). Nó đã nạp, và giờ nó kết thúc.`);
    case 'tap': {
      const tp = s.tap!;
      return tr(`${who}, tap (${tp.fRow}, ${tp.fCol}): Pvalue += F_c[${tp.fRow}][${tp.fCol}] × N_s[${tp.slotR}][${tp.slotC}] = ${tp.weight} × ${show(tp.value)}; running sum ${show(s.pvalue!)}.`, `${who}, tap (${tp.fRow}, ${tp.fCol}): Pvalue += F_c[${tp.fRow}][${tp.fCol}] × N_s[${tp.slotR}][${tp.slotC}] = ${tp.weight} × ${show(tp.value)}; tổng tích lũy ${show(s.pvalue!)}.`);
    }
    case 'store':
      return tr(`${who} stores ${show(s.pvalue!)} to P[${s.global!.row}][${s.global!.col}].`, `${who} ghi ${show(s.pvalue!)} vào P[${s.global!.row}][${s.global!.col}].`);
    default:
      return '';
  }
}

function HaloStepper() {
  const [img, setImg] = useState(10);
  const [bx, setBx] = useState(0);
  const [by, setBy] = useState(0);
  const [schedule, setSchedule] = useState<Schedule>('in-order');
  const [barrier, setBarrier] = useState(true);
  const [zeroGhost, setZeroGhost] = useState(true);
  const [loadOffset, setLoadOffset] = useState(true);
  const [n, setN] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [picked, setPicked] = useState<number | null>(null);

  const cfg: ConvConfig = useMemo(() => ({ inTile: 8, r: 1, barrier, zeroGhost, loadOffset }), [barrier, zeroGhost, loadOffset]);
  const prob = useMemo(() => makeProblem(img, img, 1), [img]);
  const grid = gridOf(cfg, prob);
  const locale = useLocale();
  const trace = useMemo(() => runTiledBlock(cfg, prob, bx, by, schedule, 5), [cfg, prob, bx, by, schedule, locale]);
  const total = trace.steps.length;
  const at = Math.min(n, total);
  const st = useMemo(() => stateAt(cfg, trace, at), [cfg, trace, at]);
  const cur: ConvStep | undefined = at > 0 ? trace.steps[at - 1] : undefined;
  const firstBad = trace.violations.length ? trace.violations[0].step + 1 : null;
  const IN = cfg.inTile;
  const R = cfg.r;
  const OUT = outTileOf(cfg);
  const infos = useMemo(
    () => Array.from({ length: IN * IN }, (_, t) => threadInfo(cfg, prob, bx, by, t % IN, Math.floor(t / IN))),
    [cfg, prob, bx, by, IN],
  );
  const sel = picked ?? (cur && cur.thread >= 0 ? cur.thread : 1 * IN + 1);
  const ti = infos[sel];

  useEffect(() => setN(0), [img, bx, by, schedule, barrier, zeroGhost, loadOffset]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setN((x) => x + 1), 110);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);

  const toBarrier = () => {
    for (let i = at; i < total; i++) if (trace.steps[i].op === 'release') return setN(i + 1);
    setN(total);
  };
  const off = loadOffset ? ` − ${R}` : '';
  const curTap = cur?.op === 'tap' ? cur.tap! : undefined;
  // The patch to outline: the computing thread's, or else the inspected thread's if it computes.
  const patch =
    cur && cur.thread >= 0 && (cur.op === 'tap' || cur.op === 'store') ? infos[cur.thread] : infos[sel].active ? infos[sel] : undefined;
  const stored = new Map(st.stores.map((s) => [s.index, s.value]));
  // Where this block's input tile really is (the switch can move it).
  const tileRow0 = by * OUT - (loadOffset ? R : 0);
  const tileCol0 = bx * OUT - (loadOffset ? R : 0);
  const finished = at >= total;
  // The block's own output tile, whatever the switches do to the kernel.
  const mineAt = (y: number, x: number) => y >= by * OUT && y < (by + 1) * OUT && x >= bx * OUT && x < (bx + 1) * OUT;
  const mineCount = Math.max(0, Math.min(img, (by + 1) * OUT) - by * OUT) * Math.max(0, Math.min(img, (bx + 1) * OUT) - bx * OUT);
  const badValues = st.stores.filter((s) => !(s.value === prob.ref[s.index])).length;
  const strays = st.stores.filter((s) => !mineAt(Math.floor(s.index / img), s.index % img)).length;
  const unwritten = mineCount - (st.stores.length - strays);
  const allOk = badValues === 0 && strays === 0 && unwritten === 0;

  return (
    <Bench
      title={tr('Step through one block', 'Đi từng bước qua một khối')}
      task={tr(
        'Step thread (tx 1, ty 1) through its load and its first taps. Then remove one safeguard at a time and find the first failure.',
        'Cho luồng (tx 1, ty 1) đi từng bước qua lần nạp và các tap đầu tiên của nó. Rồi bỏ từng biện pháp bảo vệ một và tìm lỗi đầu tiên.',
      )}
      foot={
        <Translated
          vi={
            <>
              Một khối 8 × 8 luồng, một bộ lọc 3 × 3 (bán kính 1) và một ô đầu ra 6 × 6, như trong Hình 7.13. Bộ lọc là{' '}
              {prob.F.slice(0, 3).join(' ')} / {prob.F.slice(3, 6).join(' ')} / {prob.F.slice(6).join(' ')}, cố ý làm lệch
              để một mảng con bị dịch chỗ sẽ lộ ra. Các ô viền nét đứt là ô ma. Trong lưới luồng, w đánh dấu một luồng đang
              chờ ở rào chắn, ✓ một luồng đã ghi đầu ra của nó, – một luồng rảnh; nhấn vào một luồng để xem nó. <TierLegend tiers={['shared', 'global', 'reg']} />
            </>
          }
        >
          <>
            A block of 8 × 8 threads, a 3 × 3 filter (radius 1) and a 6 × 6 output tile, as in Fig. 7.13. The filter is{' '}
            {prob.F.slice(0, 3).join(' ')} / {prob.F.slice(3, 6).join(' ')} / {prob.F.slice(6).join(' ')}, deliberately
            lopsided so that a shifted patch shows. Dashed slots are ghost cells. In the thread grid, w marks a thread
            waiting at the barrier, ✓ one that has stored its output, – one that is idle; click a thread to inspect it. <TierLegend tiers={['shared', 'global', 'reg']} />
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Ảnh">Image</Translated>
          <select value={img} onChange={(e) => { setImg(+e.target.value); setBx(0); setBy(0); setPicked(null); }} aria-label={tr('image size', 'kích thước ảnh')}>
            {IMAGES.map((w) => <option key={w} value={w}>{w} × {w}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Khối (x, y)">Block (x, y)</Translated>
          <select value={`${bx},${by}`} aria-label={tr('block', 'khối')} onChange={(e) => { const [x, y] = e.target.value.split(',').map(Number); setBx(x); setBy(y); setPicked(null); }}>
            {Array.from({ length: grid.x * grid.y }, (_, i) => [i % grid.x, Math.floor(i / grid.x)]).map(([x, y]) => (
              <option key={`${x},${y}`} value={`${x},${y}`}>({x}, {y})</option>
            ))}
          </select>
        </label>
        <label className="field">
          <Translated vi="Thứ tự luồng">Thread order</Translated>
          <select value={schedule} aria-label={tr('thread order', 'thứ tự luồng')} onChange={(e) => setSchedule(e.target.value as Schedule)}>
            {SCHEDULES.map((s) => <option key={s} value={s}>{SCHEDULE_LABEL[s]}</option>)}
          </select>
        </label>
        <label className="check"><input type="checkbox" checked={barrier} onChange={(e) => setBarrier(e.target.checked)} /> <Translated vi="rào chắn">barrier</Translated></label>
        <label className="check"><input type="checkbox" checked={zeroGhost} onChange={(e) => setZeroGhost(e.target.checked)} /> <Translated vi="đặt ô ma bằng 0">ghost slots set to 0</Translated></label>
        <label className="check"><input type="checkbox" checked={loadOffset} onChange={(e) => setLoadOffset(e.target.checked)} /> <Translated vi="− FILTER_RADIUS trong row và col">− FILTER_RADIUS in row and col</Translated></label>
      </div>

      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setN(0); }} disabled={at === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => { setPlaying(false); setN(at - 1); }} disabled={at === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => { setPlaying(false); setN(at + 1); }} disabled={finished}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => { setPlaying(false); toBarrier(); }} disabled={finished}><Translated vi="Chạy đến rào chắn">Run to barrier</Translated></button>
        <button onClick={() => setPlaying(!playing)} disabled={finished}>{playing ? tr('Pause', 'Tạm dừng') : tr('Play', 'Chạy')}</button>
        <button onClick={() => { setPlaying(false); setN(total); }} disabled={finished}><Translated vi="Chạy đến cuối">Run to end</Translated></button>
        {firstBad ? <button onClick={() => { setPlaying(false); setN(firstBad); }}><Translated vi="Đến lỗi đầu tiên">Go to the first failure</Translated></button> : null}
        <span className="muted" aria-live="polite"><Translated vi={<>bước {at} trên {total}</>}>step {at} of {total}</Translated></span>
      </div>

      <div className={'narrate' + (cur?.violation ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }}>
        {cur ? (
          <>
            {describe(cur, cfg)}
            {cur.violation ? <><br /><b className="t-fail"><Translated vi={<>Lỗi: {cur.violation.message}.</>}>Failure: {cur.violation.message}.</Translated></b></> : null}
          </>
        ) : (
          <Translated
            vi={
              <>
                Chưa có gì chạy. Khối ({bx}, {by}) sở hữu các hàng đầu ra {by * OUT} đến {Math.min(img, by * OUT + OUT) - 1} và các cột{' '}
                {bx * OUT} đến {Math.min(img, bx * OUT + OUT) - 1}, và nạp ô đầu vào tại các hàng {tileRow0} đến {tileRow0 + IN - 1}, các cột{' '}
                {tileCol0} đến {tileCol0 + IN - 1}. Nhấn Bước.
              </>
            }
          >
            <>
              Nothing has run yet. Block ({bx}, {by}) owns output rows {by * OUT} to {Math.min(img, by * OUT + OUT) - 1} and columns{' '}
              {bx * OUT} to {Math.min(img, bx * OUT + OUT) - 1}, and loads the input tile at rows {tileRow0} to {tileRow0 + IN - 1}, columns{' '}
              {tileCol0} to {tileCol0 + IN - 1}. Press Step.
            </>
          </Translated>
        )}
      </div>

      <div className="row" style={{ alignItems: 'flex-start', gap: '1rem 1.4rem' }}>
        <div>
          <div className="mx-label t-global"><Translated vi="N (toàn cục), với vòng ô ma">N (global), with the ghost ring</Translated></div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${img + 2}, auto)` }} data-k="n">
            {Array.from({ length: (img + 2) * (img + 2) }, (_, i) => {
              const y = Math.floor(i / (img + 2)) - 1;
              const x = (i % (img + 2)) - 1;
              const inside = y >= 0 && y < img && x >= 0 && x < img;
              const inTile = y >= tileRow0 && y < tileRow0 + IN && x >= tileCol0 && x < tileCol0 + IN;
              const hot = cur?.op === 'load' && cur.global!.row === y && cur.global!.col === x;
              const cls = hot ? 'hot' : inside ? (inTile ? 'on' : '') : 'void';
              return <span key={i} className={cls} style={{ ...tiny, outline: !inside && inTile ? '1px dashed var(--global)' : undefined }}>{inside ? prob.N[y * img + x] : ''}</span>;
            })}
          </div>
        </div>
        <div>
          <div className="mx-label t-shared"><Translated vi="N_s (chia sẻ, khối này)">N_s (shared, this block)</Translated></div>
          <div className="mx shared" style={{ gridTemplateColumns: `repeat(${IN}, auto)` }} data-k="ns">
            {st.slots.map((s, t) => {
              const r0 = Math.floor(t / IN);
              const c0 = t % IN;
              const tapHere = curTap && curTap.slotR === r0 && curTap.slotC === c0;
              const inPatch = patch && r0 >= patch.tileRow && r0 < patch.tileRow + 3 && c0 >= patch.tileCol && c0 < patch.tileCol + 3;
              const cls = tapHere ? (curTap!.written ? 'hot' : 'bad') : cur?.op === 'load' && cur.thread === t ? 'hot' : s.written ? 'on' : '';
              return (
                <span key={t} className={cls} style={{ ...tiny, borderStyle: s.ghost || !infos[t].inImage ? 'dashed' : undefined, outline: inPatch ? '1.5px solid var(--reg)' : undefined }}>
                  {s.written ? show(s.value) : '·'}
                </span>
              );
            })}
          </div>
          <div className="mx-label" style={{ marginTop: '0.7rem' }}><Translated vi="Các luồng: nhấn vào một luồng để xem">Threads: click one to inspect</Translated></div>
          <div className="mx" style={{ gridTemplateColumns: `repeat(${IN}, auto)` }} data-k="threads">
            {infos.map((x, t) => {
              const state = st.done[t] ? (x.active ? '✓' : '–') : st.waiting[t] ? 'w' : st.last[t] >= 0 ? (x.active ? '…' : '·') : '';
              return (
                <span key={t} role="button" tabIndex={0} title={tr(`thread (tx ${x.tx}, ty ${x.ty})`, `luồng (tx ${x.tx}, ty ${x.ty})`)} aria-label={tr(`thread (tx ${x.tx}, ty ${x.ty})`, `luồng (tx ${x.tx}, ty ${x.ty})`)}
                  onClick={() => setPicked(t)} onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && setPicked(t)}
                  className={cur?.thread === t ? 'hot' : ''}
                  style={{ ...tiny, cursor: 'pointer', opacity: x.active ? 1 : 0.45, outline: t === sel ? '1.5px solid var(--ink)' : undefined }}>
                  {state}
                </span>
              );
            })}
          </div>
        </div>
        <div>
          <div className="mx-label t-global"><Translated vi="P (toàn cục)">P (global)</Translated></div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${img}, auto)` }} data-k="p">
            {prob.ref.map((ref, i) => {
              const y = Math.floor(i / img);
              const x = i % img;
              const mine = mineAt(y, x);
              const v = stored.get(i);
              const cls = v === undefined ? (mine ? 'on' : 'void') : v === ref ? 'good' : 'bad';
              return <span key={i} className={cls} style={tiny}>{v === undefined ? '' : show(v)}</span>;
            })}
          </div>
          {finished ? (
            <p style={{ margin: '0.7rem 0 0', maxWidth: '18rem' }} data-k="verdict">
              {allOk ? (
                <span className="chip ok"><Translated vi={<>{st.stores.length} đầu ra của khối này khớp với giá trị tham chiếu.</>}>This block's {st.stores.length} outputs match the reference.</Translated></span>
              ) : (
                <span className="chip fail"><Translated vi="Ô đầu ra của khối này sai.">This block's output tile is wrong.</Translated></span>
              )}{' '}
              {badValues ? tr(`${badValues} stored values differ from the reference. `, `${badValues} giá trị được ghi khác với giá trị tham chiếu. `) : ''}
              {unwritten > 0 ? tr(`${unwritten} outputs of its own tile were never written. `, `${unwritten} đầu ra trong ô của chính nó chưa bao giờ được ghi. `) : ''}
              {strays ? tr(`${strays} stores landed outside its tile. `, `${strays} lệnh ghi rơi ra ngoài ô của nó. `) : ''}
              {trace.violations.length ? tr(`${trace.violations.length} reads of slots that were not written.`, `${trace.violations.length} lần đọc các ô chưa được ghi.`) : ''}
            </p>
          ) : null}
        </div>
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 2fr)', marginTop: '0.4rem' }}>
        <Code
          tabs={[{ label: tr('kernel (Fig. 7.12, corrected)', 'kernel (Hình 7.12, đã sửa)'), code: HALO_TOY.split('\n').slice(0, KERNEL_END).join('\n'), tiers: CONV_TIERS, status: 'emulator-checked and front-end-checked' }]}
          highlight={lineOf(cur, cfg)}
        />
        <div className="panel" data-k="thread" style={{ marginTop: '1.1rem' }}>
          <b><Translated vi={<>Luồng (tx {ti.tx}, ty {ti.ty})</>}>Thread (tx {ti.tx}, ty {ti.ty})</Translated></b>
          {picked != null ? <> <button className="quiet" onClick={() => setPicked(null)}><Translated vi="theo bước hiện tại">follow the step</Translated></button></> : null}
          <p style={{ margin: '0.4rem 0' }}>
            row = {by}·{OUT} + {ti.ty}{off} = {ti.row}, col = {bx}·{OUT} + {ti.tx}{off} = {ti.col}:{' '}
            {ti.inImage ? <Translated vi={<>nạp N[{ti.row}][{ti.col}] vào N_s[{ti.ty}][{ti.tx}].</>}><>loads N[{ti.row}][{ti.col}] into N_s[{ti.ty}][{ti.tx}].</></Translated> : <Translated vi={<>một ô ma, nên N_s[{ti.ty}][{ti.tx}] nhận {zeroGhost ? '0' : 'không gì cả'}.</>}><>a ghost cell, so N_s[{ti.ty}][{ti.tx}] gets {zeroGhost ? '0' : 'nothing'}.</></Translated>}
          </p>
          <p style={{ margin: 0 }}>
            tileRow = {ti.ty} − {R} = {ti.tileRow}, tileCol = {ti.tx} − {R} = {ti.tileCol}:{' '}
            {ti.active ? (
              <Translated vi={<>tính P[{ti.row}][{ti.col}] từ mảng con có góc là N_s[{ti.tileRow}][{ti.tileCol}]; Pvalue đến lúc này là <span className="t-reg">{show(st.pvalue[sel])}</span>.</>}>
                <>computes P[{ti.row}][{ti.col}] from the patch whose corner is N_s[{ti.tileRow}][{ti.tileCol}]; Pvalue so far <span className="t-reg">{show(st.pvalue[sel])}</span>.</>
              </Translated>
            ) : ti.reason === 'ring' ? (
              <Translated vi={<>ở vòng ngoài, rảnh trong pha tính toán.</>}>
                <>on the outer ring, idle in the compute phase.</>
              </Translated>
            ) : (
              <Translated vi={<>đầu ra của nó sẽ nằm ngoài ảnh, nên nó rảnh.</>}>
                <>its output would be outside the image, so it is idle.</>
              </Translated>
            )}
          </p>
        </div>
      </div>
    </Bench>
  );
}

function HaloOutcomes() {
  const [img, setImg] = useState(10);
  const prob = useMemo(() => makeProblem(img, img, 1), [img]);
  const variants: { label: string; over: Partial<ConvConfig> }[] = [
    { label: tr('as written', 'như đã viết'), over: {} },
    { label: tr('no barrier', 'không có rào chắn'), over: { barrier: false } },
    { label: tr('no ghost zeros', 'không đặt ô ma bằng 0'), over: { zeroGhost: false } },
    { label: tr('no − FILTER_RADIUS', 'không có − FILTER_RADIUS'), over: { loadOffset: false } },
  ];
  return (
    <Bench
      title={tr('Every block, every thread order', 'Mọi khối, mọi thứ tự luồng')}
      task={tr(
        'Before you read the table: which defect can no thread order hide, and which one does lockstep hide?',
        'Trước khi đọc bảng: lỗi nào không thứ tự luồng nào che được, và lỗi nào bị chế độ lockstep che đi?',
      )}
      foot={tr(
        'Each cell runs every block of the image in the model above and compares P with a direct convolution.',
        'Mỗi ô chạy mọi khối của ảnh trong mô hình ở trên và so sánh P với một phép tích chập trực tiếp.',
      )}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Ảnh">Image</Translated>
          <select value={img} onChange={(e) => setImg(+e.target.value)} aria-label={tr('outcome image size', 'kích thước ảnh cho kết quả')}>
            {IMAGES.map((w) => <option key={w} value={w}>{w} × {w}</option>)}
          </select>
        </label>
      </div>
      <table className="data" style={{ background: 'var(--page)', margin: 0 }}>
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}><Translated vi="Kernel">Kernel</Translated></th>
            {SCHEDULES.map((s) => <th key={s}>{SCHEDULE_LABEL[s]}</th>)}
          </tr>
        </thead>
        <tbody>
          {variants.map((v) => (
            <tr key={v.label}>
              <td style={{ paddingLeft: '0.6rem' }}>{v.label}</td>
              {SCHEDULES.map((s) => {
                const r = runAll({ ...CONV_CORRECT, inTile: 8, r: 1, ...v.over }, prob, s, 5);
                return (
                  <td key={s} data-outcome={r.correct ? 'correct' : 'wrong'}>
                    {r.correct ? (
                      <span className="chip ok"><Translated vi="đúng">correct</Translated></span>
                    ) : (
                      <span className="chip fail">
                        {r.wrong.length ? tr(`${r.wrong.length} wrong`, `${r.wrong.length} sai`) : ''}
                        {r.wrong.length && r.missing.length ? ', ' : ''}
                        {r.missing.length ? tr(`${r.missing.length} never written`, `${r.missing.length} chưa bao giờ được ghi`) : ''}
                      </span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- your device against the formula

interface Point {
  b: string;
  device: string;
  tile: number;
  out: number;
  r: number;
  ratio: number;
  speedup: number;
}

function measured(reports: Reports): Point[] {
  return realBackends(reports).flatMap((b) =>
    benchRows(reports[b], 'ch07/lab2_tiledconv')
      .filter((x) => x.name === 'tiled_halo')
      .flatMap((x) => {
        const tile = num(x.tile);
        const rad = num(x.radius) ?? (num(x.m) != null ? (num(x.m)! - 1) / 2 : undefined);
        const out = num(x.out_tile) ?? (tile != null && rad != null ? tile - 2 * rad : undefined);
        const sp = num(x.speedup_vs_const);
        if (tile == null || rad == null || out == null || sp == null || out <= 0) return [];
        const ratio = num(x.model_traffic_ratio) ?? trafficRatioVsConst(out, 2 * rad + 1);
        return [{ b, device: String(reports[b].device.name ?? b), tile, out, r: rad, ratio, speedup: sp }];
      }),
  );
}

function Scatter({ pts }: { pts: Point[] }) {
  const Wd = 420, H = 300, pad = { l: 46, r: 12, t: 12, b: 38 };
  const lo = 0.125, hi = 64;
  const L = (v: number) => Math.log2(Math.max(lo, Math.min(hi, v)));
  const X = (v: number) => pad.l + ((L(v) - L(lo)) / (L(hi) - L(lo))) * (Wd - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((L(v) - L(lo)) / (L(hi) - L(lo))) * (H - pad.t - pad.b);
  const ticks = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64];
  const labelled = new Set([0.125, 0.5, 2, 8, 32]);
  return (
    <svg viewBox={`0 0 ${Wd} ${H}`} role="img" aria-label={tr("Measured speed-up against the formula's traffic ratio", 'Mức tăng tốc đo được so với tỉ lệ lưu lượng theo công thức')} style={{ width: '100%', maxWidth: Wd, display: 'block' }}>
      {ticks.map((v) => (
        <g key={v}>
          <line x1={X(v)} x2={X(v)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <line x1={pad.l} x2={Wd - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          {labelled.has(v) ? (
            <>
              <text x={X(v)} y={H - pad.b + 14} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{v}</text>
              <text x={pad.l - 6} y={Y(v) + 3} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{v}</text>
            </>
          ) : null}
        </g>
      ))}
      <line x1={X(lo)} y1={Y(lo)} x2={X(hi)} y2={Y(hi)} stroke="var(--ink-3)" strokeDasharray="4 3" />
      <text x={X(20)} y={Y(20) - 6} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-3)' }}><Translated vi="đo được = công thức">measured = formula</Translated></text>
      <line x1={pad.l} x2={Wd - pad.r} y1={Y(1)} y2={Y(1)} stroke="var(--ink-3)" />
      <text x={pad.l + 4} y={Y(1) - 5} fontSize="10" style={{ fill: 'var(--ink-3)' }}><Translated vi="không nhanh hơn">no faster</Translated></text>
      {pts.map((p, i) => (
        <circle key={i} cx={X(p.ratio)} cy={Y(p.speedup)} r="4.5" fill="var(--shared)" fillOpacity="0.75" stroke="var(--page)">
          <title>{tr(`${p.device}: tile ${p.tile}, r = ${p.r}: formula ${fmt(p.ratio)}×, measured ${fmt(p.speedup)}×`, `${p.device}: ô ${p.tile}, r = ${p.r}: công thức ${fmt(p.ratio)}×, đo được ${fmt(p.speedup)}×`)}</title>
        </circle>
      ))}
      <text x={(pad.l + Wd - pad.r) / 2} y={H - 6} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink)' }}><Translated vi="công thức: số lần ít lưu lượng toàn cục hơn so với kernel bộ nhớ hằng">formula: times less global traffic than the constant-memory kernel</Translated></text>
      <text x={12} y={(pad.t + H - pad.b) / 2} fontSize="10.5" textAnchor="middle" transform={`rotate(-90 12 ${(pad.t + H - pad.b) / 2})`} style={{ fill: 'var(--ink)' }}><Translated vi="mức tăng tốc đo được">measured speed-up</Translated></text>
    </svg>
  );
}

function MeasuredAgainstFormula({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const pts = measured(reports);
  if (!pts.length)
    return (
      <Translated
        vi={
          <p data-k="no-measurement">
            Máy này chưa có số đo nào. Hãy chạy <code>python3 labs/pmpp.py run 7.2 --solution --bench</code>, rồi tải lại
            trang này: ô này sẽ vẽ từng kích thước ô và bán kính bạn đã chạy so với công thức.
          </p>
        }
      >
        <p data-k="no-measurement">
          No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 7.2 --solution --bench</code>, then
          reload this page: this box will plot each tile size and radius you ran against the formula.
        </p>
      </Translated>
    );
  const below = pts.filter((p) => p.speedup < p.ratio).length;
  const slower = pts.filter((p) => p.speedup < 1).length;
  const book = pts.filter((p) => p.tile === 32 && p.r === 2);
  const devices = [...new Set(pts.map((p) => `${p.device} (${p.b})`))];
  return (
    <div data-k="measurement">
      <p style={{ marginTop: 0 }}>
        <Translated vi={<>Lab 7.2 trên {devices.join(' và ')}: mỗi kích thước khối và bán kính bạn đã chạy là một điểm.</>}>
          Lab 7.2 on {devices.join(' and ')}: one point per block size and radius you ran.
        </Translated>
      </p>
      <Scatter pts={pts} />
      <table className="data" style={{ margin: '0.6rem 0 0', fontSize: '0.86em' }}>
        <thead>
          <tr>
            {devices.length > 1 ? <th><Translated vi="thiết bị">device</Translated></th> : null}
            <th className="num"><Translated vi="khối">block</Translated></th><th className="num">r</th><th className="num"><Translated vi="ô đầu ra">output tile</Translated></th>
            <th className="num"><Translated vi="công thức">formula</Translated></th><th className="num"><Translated vi="đo được">measured</Translated></th>
          </tr>
        </thead>
        <tbody>
          {pts.map((p, i) => (
            <tr key={i}>
              {devices.length > 1 ? <td>{p.device}</td> : null}
              <td className="num">{p.tile}</td><td className="num">{p.r}</td><td className="num">{p.out}</td>
              <td className="num">{fmt(p.ratio)}×</td><td className="num">{fmt(p.speedup)}×</td>
            </tr>
          ))}
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            {below === pts.length
              ? `Cả ${pts.length} số đo đều nằm dưới đường: kernel phân ô tiết kiệm ít thời gian hơn mức mà lưu lượng nó loại bỏ gợi ý.`
              : below === 0
                ? `Cả ${pts.length} số đo đều nằm trên hoặc phía trên đường.`
                : `${below} trên ${pts.length} số đo nằm dưới đường.`}{' '}
            {slower ? `${slower} trong số đó nhỏ hơn 1: ở đó kernel phân ô chậm hơn kernel bộ nhớ hằng. ` : ''}
            Công thức tính mọi lần nạp của kernel bộ nhớ hằng là một chuyến đi tới DRAM; ở chỗ bộ nhớ đệm đã phục vụ phần lớn
            chúng, có ít hơn để tiết kiệm. Và nó hoàn toàn không tính thời gian: không tính việc chờ ở rào chắn, cũng không
            tính vòng luồng ngồi rảnh trong khi các luồng khác tính toán.
            {book.length
              ? ` Với ô 32 và r = 2, bạn đo được ${book.map((p) => `${fmt(p.speedup)}× trên ${p.device}`).join(', ')}, so với dự đoán ${predicted}× của bạn.`
              : ''}
          </p>
        }
      >
        <p>
          {below === pts.length
            ? `All ${pts.length} measurements sit below the line: the tiled kernel saves less time than the traffic it removes would suggest.`
            : below === 0
              ? `All ${pts.length} measurements sit on or above the line.`
              : `${below} of ${pts.length} measurements sit below the line.`}{' '}
          {slower ? `${slower} of them are below 1: there the tiled kernel was slower than the constant-memory kernel. ` : ''}
          The formula counts every load of the constant-memory kernel as a trip to DRAM; where caches already served most
          of them, there was less to save. And it does not count time at all: not the wait at the barrier, nor the ring of
          threads that idles while the others compute.
          {book.length
            ? ` At tile 32 and r = 2 you measured ${book.map((p) => `${fmt(p.speedup)}× on ${p.device}`).join(', ')}, against your prediction of ${predicted}×.`
            : ''}
        </p>
      </Translated>
      <p className="muted">
        <Translated vi="Đây là các số đo từ máy của bạn, không phải giá trị kỳ vọng. Một thiết bị khác có thể chênh lệch rất nhiều theo cả hai hướng.">
          These are measurements from your machine, not expected values. Another device can differ by a wide margin in either direction.
        </Translated>
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- the lab listing

function HaloListing() {
  const { reports } = useReports();
  const cu = labPart(useLabFile('ch07/lab2_tiledconv/solution/kernel.cu'), 'halo');
  const metal = labPart(useLabFile('ch07/lab2_tiledconv/solution/kernel.metal'), 'halo');
  if (!cu || !metal) return null;
  const lab = 'ch07/lab2_tiledconv';
  // The labs page's words (LabKit's solutionTabStatus), so a failed run reads as failed here too.
  const cuStatus = solutionTabStatus(reports, lab, 'kernel.cu');
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: cu, tiers: CONV_TIERS, status: cuStatus },
        { label: tr('Metal kernel', 'Kernel Metal'), code: metal, tiers: CONV_TIERS, status: solutionTabStatus(reports, lab, 'kernel.metal') },
      ]}
      caption={
        <Translated
          vi={
            <>
              Phần vùng biên trong lời giải tham chiếu của lab 7.2, tải từ <code>labs/ch07/lab2_tiledconv/solution/</code>.
              Hãy tự viết bản của bạn trong lab trước khi nghiên cứu nó. Bản build HIP được thiết lập để nhận nguyên tệp{' '}
              <code>.cu</code> này qua <code>pmpp/vendors/hip.h</code>; nó chưa được biên dịch hay chạy trên phần cứng AMD
              (unverified).
            </>
          }
        >
          <>
            The halo part of lab 7.2&apos;s reference solution, loaded from <code>labs/ch07/lab2_tiledconv/solution/</code>.
            Write your own in the lab before studying it. The HIP build is set up to take this <code>.cu</code> file
            unchanged through <code>pmpp/vendors/hip.h</code>; it has not been compiled or run on AMD hardware
            (unverified).
          </>
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- questions

export const questionsS75: Record<string, Question> = {
  'ch07/7.5/q-ratio': {
    kind: 'numeric',
    id: 'ch07/7.5/q-ratio',
    prompt: (
      <Translated
        vi={
          <>
            Một khối tính một ô đầu ra 16 × 16 với bộ lọc 5 × 5. Nó nạp bao nhiêu phần tử đầu vào cho mỗi phần tử đầu ra?
          </>
        }
      >
        <>
          A block computes a 16 × 16 output tile with a 5 × 5 filter. How many input elements does it load per output
          element?
        </>
      </Translated>
    ),
    answer: 1.5625,
    tol: 0.01,
    explain: (
      <Translated
        vi={
          <p>
            Bán kính là 2, nên ô đầu vào rộng 16 + 2 + 2 = 20: 400 đầu vào cho 256 đầu ra, 1.5625 cho mỗi đầu ra. Ví dụ
            nhỏ 4 × 4 của sách cần 64 cho 16, 4 cho mỗi đầu ra. Vùng biên là một vòng có độ rộng do bộ lọc cố định, nên nó
            co lại so với ô khi ô lớn lên.
          </p>
        }
      >
        <p>
          The radius is 2, so the input tile is 16 + 2 + 2 = 20 wide: 400 inputs for 256 outputs, 1.5625 per output. The
          book&apos;s 4 × 4 toy needed 64 for 16, 4 per output. The halo is a ring whose width is fixed by the filter, so
          it shrinks relative to the tile as the tile grows.
        </p>
      </Translated>
    ),
    source: 'ch. 7 §7.5, pp. 172–173',
  },
  'ch07/7.5/q-p175': {
    kind: 'choice',
    id: 'ch07/7.5/q-p175',
    prompt: (
      <Translated
        vi={
          <>
            Với <code>FILTER_RADIUS = 1</code>, phần chữ ở tr. 175 nói luồng (5,5) duyệt qua mảng con có góc trên bên trái
            là <code>N_s[5][5]</code>. Thực ra luồng nào dùng mảng con đó?
          </>
        }
      >
        <>
          With <code>FILTER_RADIUS = 1</code>, the text on p. 175 says thread (5,5) iterates through the patch whose
          upper-left corner is <code>N_s[5][5]</code>. Which thread actually uses that patch?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Luồng (5,5), như được in.">Thread (5,5), as printed.</Translated>,
        why: (
          <Translated vi="tileRow và tileCol của nó là 5 − 1 = 4, nên mảng con của nó bắt đầu tại N_s[4][4].">
            Its tileRow and tileCol are 5 − 1 = 4, so its patch starts at N_s[4][4].
          </Translated>
        ),
      },
      {
        label: <Translated vi="Luồng (4,4).">Thread (4,4).</Translated>,
        why: <Translated vi="Mảng con của nó bắt đầu tại N_s[3][3].">Its patch starts at N_s[3][3].</Translated>,
      },
      { label: <Translated vi="Luồng (6,6).">Thread (6,6).</Translated>, correct: true },
      {
        label: <Translated vi="Luồng (7,7).">Thread (7,7).</Translated>,
        why: (
          <Translated vi="Luồng (7,7) nằm ở vòng ngoài của một khối 8 × 8 và không tính gì cả.">
            Thread (7,7) is on the outer ring of an 8 × 8 block and computes nothing.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mảng con của một luồng bắt đầu tại (tileRow, tileCol) = (ty − 1, tx − 1), nên góc N_s[5][5] thuộc về luồng
            (6,6), chính là luồng mà Hình 7.13 vẽ. Nó tính đầu ra (5,5) của ô, đầu ra cuối cùng. Chữ &quot;(5,5)&quot; trong
            văn bản là một sơ suất (notes, Unclear in the PDF, p. 175).
          </p>
        }
      >
        <p>
          A thread&apos;s patch starts at (tileRow, tileCol) = (ty − 1, tx − 1), so the corner N_s[5][5] belongs to thread
          (6,6), which is the thread Fig. 7.13 draws. It computes output (5,5) of the tile, the last one. The text&apos;s
          &quot;(5,5)&quot; is a slip (notes, Unclear in the PDF, p. 175).
        </p>
      </Translated>
    ),
    source: 'ch. 7 §7.5, p. 175, Fig. 7.13',
    vi: { source: 'ch. 7 §7.5, tr. 175, Hình 7.13' },
  },
  'ch07/7.5/q-one-barrier': {
    kind: 'choice',
    id: 'ch07/7.5/q-one-barrier',
    prompt: (
      <Translated
        vi={<>Phép nhân ma trận phân ô của chương 5 cần hai rào chắn mỗi pha. Vì sao Hình 7.12 chỉ cần một?</>}
      >
        <>Chapter 5&apos;s tiled matrix multiplication needs two barriers per phase. Why does Fig. 7.12 need only one?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="N_s được ghi một lần và không bao giờ bị ghi đè, nên không lần ghi nào phải chờ các lần đọc trước đó.">
            N_s is written once and never overwritten, so no write has to wait for earlier reads.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Mỗi luồng chỉ đọc ô mà chính nó đã nạp, nên không luồng nào phải chờ luồng bên cạnh.">
            Each thread reads only the slot it loaded itself, so no thread waits for a neighbour.
          </Translated>
        ),
        why: (
          <Translated vi="Một luồng đang hoạt động đọc chín ô, tám trong số đó do các luồng bên cạnh ghi.">
            An active thread reads nine slots, eight of them written by its neighbours.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Bộ lọc nằm trong bộ nhớ hằng, thứ giữ cho các lần đọc bộ nhớ chia sẻ đúng thứ tự.">
            The filter sits in constant memory, which keeps the shared-memory reads in order.
          </Translated>
        ),
        why: (
          <Translated vi="Bộ nhớ hằng chứa bộ lọc. Nó không sắp thứ tự gì cả.">
            Constant memory holds the filter. It orders nothing.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó có cần rào chắn thứ hai trước lệnh ghi; sách bỏ sót và gặp may.">
            It does need a second barrier before the store; the book left it out and got lucky.
          </Translated>
        ),
        why: (
          <Translated vi="Bảng ở trên không có rào chắn thứ hai: với một rào chắn, mọi thứ tự luồng đều đúng.">
            The table above has no second barrier: with the one barrier every thread order is correct.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Rào chắn thứ nhất của chương 5 khiến bên đọc chờ bên ghi (đọc sau ghi). Rào chắn thứ hai khiến bên ghi của pha
            sau chờ bên đọc của pha này (ghi sau đọc), vì cùng các ô được dùng lại. Kernel tích chập chỉ có một pha: N_s
            được ghi một lần, đọc sau rào chắn, và không bao giờ được ghi lại. Chỉ có phụ thuộc thứ nhất, nên chỉ cần rào
            chắn thứ nhất.
          </p>
        }
      >
        <p>
          Chapter 5&apos;s first barrier makes readers wait for writers (read after write). Its second makes the next
          phase&apos;s writers wait for this phase&apos;s readers (write after read), because the same slots are reused. The
          convolution kernel has a single phase: N_s is written once, read after the barrier, and never written again. Only
          the first dependence exists, so only the first barrier is needed.
        </p>
      </Translated>
    ),
    source: 'ch. 7 §7.5, p. 174; ch. 5 §5.4, pp. 110–111',
    vi: { source: 'ch. 7 §7.5, tr. 174; ch. 5 §5.4, tr. 110–111' },
  },
  'ch07/7.5/q-ai9': {
    kind: 'numeric',
    id: 'ch07/7.5/q-ai9',
    prompt: (
      <Translated
        vi={
          <>
            Ô đầu vào 32 × 32, bộ lọc 9 × 9, float 4 byte. Công thức của §7.5 cho cường độ tính toán bao nhiêu, tính bằng
            FLOP/B?
          </>
        }
      >
        <>
          Input tile 32 × 32, filter 9 × 9, 4-byte floats. What arithmetic intensity does the §7.5 formula give, in FLOP/B?
        </>
      </Translated>
    ),
    answer: 14.58,
    tol: 0.01,
    unit: 'FLOP/B',
    explain: (
      <Translated
        vi={
          <p>
            Bán kính là 4, nên ô đầu ra là 32 − 8 = 24. FLOP: 2 · 24² · 81 = 93,312. Byte: 4 · (32² + 24²) = 4 · (1024 +
            576) = 6,400. Tỉ lệ 14.58, so với mức lý tưởng 81/4 = 20.25: đạt 72%. Hình 7.14 không có cột cho 24; các cột
            lân cận là 12.46 (t = 16) và 15.27 (t = 28).
          </p>
        }
      >
        <p>
          The radius is 4, so the output tile is 32 − 8 = 24. FLOP: 2 · 24² · 81 = 93,312. Bytes: 4 · (32² + 24²) = 4 ·
          (1024 + 576) = 6,400. Ratio 14.58, against an ideal of 81/4 = 20.25: 72% of it. Fig. 7.14 has no column for 24;
          its neighbours are 12.46 (t = 16) and 15.27 (t = 28).
        </p>
      </Translated>
    ),
    source: 'ch. 7 §7.5, p. 176',
  },
  'ch07/7.5/q-org2': {
    kind: 'choice',
    id: 'ch07/7.5/q-org2',
    prompt: (
      <Translated
        vi={
          <>
            Một bộ lọc 9 × 9 trên thiết bị cho phép 1024 luồng mỗi khối. Hình 7.12 với khối 32 × 32 đạt 14.58 FLOP/B. Thay
            đổi nào nâng cường độ của kernel?
          </>
        }
      >
        <>
          A 9 × 9 filter on a device that allows 1024 threads per block. Fig. 7.12 with a 32 × 32 block reaches 14.58
          FLOP/B. Which change raises the kernel&apos;s intensity?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Cho khối bằng ô đầu ra: 32 × 32 luồng lặp để nạp một ô đầu vào 40 × 40.">
            Make the block the output tile: 32 × 32 threads loop to load a 40 × 40 input tile.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Dùng khối 40 × 40, để ô đầu ra lớn lên thành 32 × 32 và việc nạp vẫn là mỗi luồng một phần tử.">
            Use a 40 × 40 block, so that the output tile grows to 32 × 32 and loading stays one per thread.
          </Translated>
        ),
        why: <Translated vi="Đó là 1600 luồng, vượt giới hạn 1024.">That is 1600 threads, over the limit of 1024.</Translated>,
      },
      {
        label: (
          <Translated vi="Đặt cả N vào bộ nhớ hằng, để các lần đọc đầu vào cũng đi qua constant cache.">
            Put N in constant memory too, so the input reads also go through the constant cache.
          </Translated>
        ),
        why: (
          <Translated vi="Bộ nhớ hằng có 64 KB, nhỏ hơn nhiều so với một ảnh, và cache của nó tuần tự hóa một warp có các luồng đọc các địa chỉ khác nhau, điều mà mọi luồng ở đây đều làm (notes/today/ch07-facts.md, A3).">
            Constant memory is 64 KB, far smaller than an image, and its cache serialises a warp whose threads read
            different addresses, which every thread here does (notes/today/ch07-facts.md, A3).
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Thay vào đó dùng khối 16 × 16, để nhiều khối hơn vừa trên mỗi SM cùng lúc.">
            Use 16 × 16 blocks instead, so that more blocks fit on each SM at the same time.
          </Translated>
        ),
        why: (
          <Translated vi="Ô đầu ra co lại còn 8 và cường độ giảm xuống 8.10.">
            The output tile shrinks to 8 and the intensity falls to 8.10.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Khi khối là ô đầu ra (cách tổ chức 2, Bài tập 11), ô đầu ra là 32 và ô đầu vào là 40: 6,400 B bộ nhớ chia sẻ, và
            2 · 32² · 81 / (4 · (40² + 32²)) = 15.80 FLOP/B, cột t = 32 của Hình 7.14. Cái giá là một vòng lặp nạp: 1600
            phần tử cho 1024 luồng.
          </p>
        }
      >
        <p>
          With the block as the output tile (organisation 2, Exercise 11), the output tile is 32 and the input tile 40:
          6,400 B of shared memory, and 2 · 32² · 81 / (4 · (40² + 32²)) = 15.80 FLOP/B, the t = 32 column of Fig. 7.14.
          The price is a load loop: 1600 elements for 1024 threads.
        </p>
      </Translated>
    ),
    source: 'ch. 7 §7.5, pp. 173, 176–177; Exercise 11, p. 181',
    vi: { source: 'ch. 7 §7.5, tr. 173, 176–177; Bài tập 11, tr. 181' },
  },
  'ch07/7.5/q-ex11': {
    kind: 'numeric',
    id: 'ch07/7.5/q-ex11',
    prompt: (
      <Translated
        vi={
          <>
            Cách tổ chức 2 với ô đầu ra (và khối) 28 × 28 và bộ lọc 5 × 5. Một luồng phải nạp tối đa bao nhiêu phần tử của
            ô đầu vào?
          </>
        }
      >
        <>
          Organisation 2 with a 28 × 28 output tile (and block) and a 5 × 5 filter. At most how many input-tile elements
          must one thread load?
        </>
      </Translated>
    ),
    answer: 2,
    explain: (
      <Translated
        vi={
          <p>
            Ô đầu vào là 32 × 32 = 1024 phần tử và khối có 784 luồng, nên một vòng lặp phẳng (luồng i nạp các phần tử i, i
            + 784, ...) cho mỗi luồng một hoặc hai: 240 luồng nạp hai. Một vòng lặp bước nhảy 28 riêng theo x và theo y thì
            dễ viết hơn nhưng không cân bằng: các luồng ở góc trên bên trái 4 × 4 của ô nạp bốn. (Phép tính của khóa học
            này; sách để việc thiết kế cho bài tập.)
          </p>
        }
      >
        <p>
          The input tile is 32 × 32 = 1024 elements and the block has 784 threads, so a flat loop (thread i loads elements
          i, i + 784, ...) gives each thread one or two: 240 threads load two. A loop that strides by 28 in x and in y
          separately is easier to write but unbalanced: the threads in the tile&apos;s top-left 4 × 4 corner load four.
          (Our arithmetic; the book leaves the design to the exercise.)
        </p>
      </Translated>
    ),
    source: 'ch. 7 Exercise 11, p. 181; our arithmetic',
    vi: { source: 'ch. 7 Bài tập 11, tr. 181; phép tính của khóa học này' },
  },
};

// ---------------------------------------------------------------- the section

export function S75() {
  return (
    <>
      <Translated
        vi={
          <p>
            Mục 7.4 đã gạch bộ lọc khỏi hóa đơn của bộ nhớ toàn cục (global memory). Thứ còn lại là đầu vào: kernel bộ nhớ
            hằng vẫn nạp một phần tử của <code>N</code> cho mỗi phép nhân-cộng, 0.5 FLOP/B. Vậy mà các đầu ra cạnh nhau dùng
            gần như cùng các đầu vào. Cách chữa là của chương 5: để các luồng (thread) của một khối nạp các đầu vào mà chúng
            dùng chung vào bộ nhớ chia sẻ (shared memory) một lần, rồi tính từ đó.
          </p>
        }
      >
        <p>
          Section 7.4 took the filter off global memory&apos;s bill. What remains is the input: the constant-memory kernel
          still loads one element of <code>N</code> for every multiply-add, 0.5 FLOP/B. Yet neighbouring outputs use almost
          the same inputs. The remedy is chapter 5&apos;s: let the threads of a block load the inputs they share into
          shared memory once, then compute from there.
        </p>
      </Translated>
      <Source section="7.5" pages="172" />

      <h2>
        <Translated vi="Ô đầu ra và ô đầu vào">Output tiles and input tiles</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Gọi các đầu ra mà một khối tính là <b>ô đầu ra</b> của nó. Để tính chúng, khối cần nhiều đầu vào hơn thế, vì mỗi
            đầu ra với tới r phần tử ra ngoài nó theo mỗi hướng. <b>Ô đầu vào</b> là ô đầu ra được nới thêm r ở mọi phía, và
            vòng phụ đó là <b>vùng biên (halo)</b>. Các phần tử vùng biên cũng nằm trong ô đầu ra của các khối lân cận, nên
            mỗi phần tử đó được nhiều hơn một khối nạp.
          </p>
        }
      >
        <p>
          Call the outputs that one block computes its <b>output tile</b>. To compute them the block needs more inputs
          than that, because every output reaches r elements past itself in each direction. The <b>input tile</b> is
          the output tile grown by r on every side, and the extra ring is the <b>halo</b>. Halo elements are also inside
          the neighbouring blocks&apos; output tiles, so each of them is loaded by more than one block.
        </p>
      </Translated>
      <Source pages="172–173" figs={tr('Fig. 7.11', 'Hình 7.11')} />
      <Translated
        vi={
          <p>
            Trong hình minh họa nhỏ của sách, một ô đầu ra 4 × 4 với bộ lọc 5 × 5 cần một ô đầu vào 8 × 8: gấp bốn lần số
            phần tử nó tạo ra. Đó là điểm khác với phép nhân ma trận của chương 5, nơi các ô của M, N và P đều cùng kích
            thước. Ở đây chúng không cùng, và phần lớn mục này nói về các hệ quả.
          </p>
        }
      >
        <p>
          In the book&apos;s toy picture, a 4 × 4 output tile with a 5 × 5 filter needs an 8 × 8 input tile: four times as
          many elements as it produces. That is the difference from chapter 5&apos;s matrix multiplication, where the
          tiles of M, N and P were all the same size. Here they are not, and most of this section is about the
          consequences.
        </p>
      </Translated>
      <Ask q={questionsS75['ch07/7.5/q-ratio']} label={tr('Predict before reading on', 'Dự đoán trước khi đọc tiếp')} />
      <Translated
        vi={
          <p>
            Vùng biên là một vòng có độ rộng cố định, nên nó ít quan trọng hơn khi ô lớn lên: 4 đầu vào mỗi đầu ra với ví dụ
            nhỏ, khoảng 1.6 với ô 16 × 16. Dù sao khối thực tế cũng lớn hơn ví dụ nhỏ. Một khối nên có ít nhất một warp
            luồng, và thường là vài warp, vừa để giữ thiết bị bận rộn vừa để có đủ tái sử dụng.
          </p>
        }
      >
        <p>
          The halo is a ring of fixed width, so it matters less as the tile grows: 4 inputs per output for the toy, about
          1.6 for a 16 × 16 tile. Real blocks are bigger than the toy anyway. A block should have at least one warp of
          threads, and usually several, both to keep the device busy and to have enough reuse.
        </p>
      </Translated>
      <Source pages="172" />

      <h2>
        <Translated vi="Hai cách giao việc cho luồng">Two ways to give threads work</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một khối có thể khớp với ô đầu vào hoặc ô đầu ra, không thể cả hai. Sách gọi tên hai lựa chọn. Cho khối có kích
            thước bằng <b>ô đầu vào</b>, và việc nạp là mỗi luồng một phần tử, đúng như chương 5; nhưng trong pha tính toán
            chỉ các luồng nằm trên ô đầu ra có việc, còn r vòng luồng bên ngoài ngồi không. Hoặc cho khối có kích thước bằng{' '}
            <b>ô đầu ra</b>: mọi luồng đều tính, nhưng việc nạp cần một vòng lặp, vì có nhiều phần tử đầu vào hơn số luồng.
          </p>
        }
      >
        <p>
          A block can match the input tile or the output tile, not both. The book names the two choices. Make the block
          the size of the <b>input tile</b>, and loading is one element per thread, exactly as in chapter 5; but in the
          compute phase only the threads over the output tile have work, and the outer r rings of threads sit idle. Or
          make the block the size of the <b>output tile</b>: every thread computes, but loading needs a loop, because
          there are more input elements than threads.
        </p>
      </Translated>
      <Source pages="173" />
      <Translated
        vi={
          <p>
            Sách viết cách thứ nhất (Hình 7.12) và để cách thứ hai làm Bài tập 11. Bạn sẽ gặp cả hai bên dưới, trong trình
            khám phá và trong các câu hỏi. Mục <a href={href('ch07', '7.6')}>7.6</a> nêu tên một kernel Metal thực tế được
            xây theo cách thứ hai.
          </p>
        }
      >
        <p>
          The book writes the first (Fig. 7.12) and leaves the second as Exercise 11. You will meet both below, in the
          explorer and in the questions. Section <a href={href('ch07', '7.6')}>7.6</a> names a real Metal kernel built the
          second way.
        </p>
      </Translated>

      <h2>
        <Translated vi="Xem một khối">Watch one block</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trước khi xem mã, hãy xem một khối với các kích thước của Hình 7.13 trong sách: một khối 8 × 8, một bộ lọc 3 × 3
            và một ô đầu ra 6 × 6, trên góc của một ảnh 10 × 10.
          </p>
        }
      >
        <p>
          Before the code, watch one block at the sizes of the book&apos;s Fig. 7.13: an 8 × 8 block, a 3 × 3 filter and
          a 6 × 6 output tile, over the corner of a 10 × 10 image.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Hoạt ảnh chạy mô hình của trình chạy từng bước với các luồng theo thứ tự chỉ số, mỗi lần một thao tác: không có
              warp, không có thời gian, không có bộ nhớ đệm. Phép đếm cuối cùng của nó đếm số byte mà khối yêu cầu, như sách
              làm.
            </p>
          }
        >
          <p>
            The animation runs the stepper&rsquo;s model with the threads in index order, one operation at a time: no
            warps, no timing, no caches. Its closing tally counts the bytes the block asks for, as the book does.
          </p>
        </Translated>
      </Simplification>
      <LoadWaitComputeScene />

      <h2>
        <Translated vi="Kernel">The kernel</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Đây là kernel phân ô có vùng biên như lời giải tham chiếu của lab 7.2 viết. Hãy đọc nó theo màu trước: các truy
            cập bộ nhớ toàn cục màu hổ phách là một lần nạp mỗi luồng trước rào chắn (barrier) và một lần ghi mỗi luồng đang
            hoạt động sau đó. Mọi thứ trong vòng lặp bộ lọc là bộ nhớ chia sẻ, bộ nhớ hằng hoặc thanh ghi.
          </p>
        }
      >
        <p>
          Here is the halo-tiled kernel as lab 7.2&apos;s reference solution writes it. Read it by colour first: the
          amber global-memory accesses are one load per thread before the barrier and one store per active thread after
          it. Everything in the filter loop is shared memory, constant memory or registers.
        </p>
      </Translated>
      <Source pages="174" figs={tr('Fig. 7.12', 'Hình 7.12')} />
      <HaloListing />
      <Translated
        vi={
          <p>
            Bốn điều gánh thiết kế này. Thứ nhất, <code>- FILTER_RADIUS</code> trong <code>col</code> và <code>row</code>{' '}
            làm hai việc. Với mọi luồng, nó chỉ ra phần tử đầu vào cần nạp, vì ô đầu vào của khối bắt đầu r phần tử trước ô
            đầu ra của nó. Với một luồng có tính toán, cùng (row, col) đó là đầu ra của nó, vì luồng (r, r) nằm trên phần tử
            đầu tiên của ô đầu ra.
          </p>
        }
      >
        <p>
          Four things carry the design. First, the <code>- FILTER_RADIUS</code> in <code>col</code> and <code>row</code>{' '}
          does double duty. For every thread it names the input element to load, because the block&apos;s input tile
          starts r elements before its output tile. For a thread that computes, the same (row, col) is its output, because
          thread (r, r) sits over the output tile&apos;s first element.
        </p>
      </Translated>
      <Source pages="174–175" />
      <Translated
        vi={
          <p>
            Thứ hai, một luồng có phần tử nằm ngoài ảnh ghi <code>0.0f</code> vào ô của nó. Các ô ma của mục 7.1 giờ tồn tại
            thành các số không thật trong <code>N_s</code>, vì vậy vòng lặp bộ lọc không có kiểm tra biên. Thứ ba, có một
            rào chắn, giữa nạp và tính. Thứ tư, <code>tileRow</code> và <code>tileCol</code> là vị trí của luồng so với ô đầu
            ra, đồng thời là góc trên bên trái của mảng con của nó trong <code>N_s</code>. Một luồng có tileRow hoặc tileCol
            âm hoặc vượt quá ô đầu ra thì nằm ở vòng ngoài: nó đã nạp phần tử của mình và giờ không làm gì.
          </p>
        }
      >
        <p>
          Second, a thread whose element lies outside the image stores <code>0.0f</code> in its slot. The ghost cells of
          section 7.1 now exist as real zeros in <code>N_s</code>, which is why the filter loop has no bounds test.
          Third, there is one barrier, between loading and computing. Fourth, <code>tileRow</code> and{' '}
          <code>tileCol</code> are the thread&apos;s position relative to the output tile, and also the upper-left corner
          of its patch in <code>N_s</code>. A thread whose tileRow or tileCol is negative or past the output tile is on the
          outer ring: it has loaded its element and now does nothing.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Đoạn mã như được in không biên dịch được: <code>N_s</code> không có kiểu phần tử, và mảng hằng được khai báo là{' '}
            <code>F_c</code> nhưng lại được dùng là <code>F</code>. Ngoài ra, <code>0.0</code> là một double trong khi ý định
            là <code>0.0f</code>; cái đó biên dịch được, nhưng không phải thứ một kernel float muốn. Lab sửa cả ba. Thêm một
            thay đổi về tính đúng đắn trên mọi backend: <code>threadIdx.x</code> là số không dấu, nên{' '}
            <code>threadIdx.x - FILTER_RADIUS</code> là phép tính không dấu. Sách gán nó vào một <code>int</code> trước khi
            so sánh, cách đó chạy được; nếu so sánh trước khi chuyển kiểu thì một phép kiểm tra như <code>&gt;= 0</code>{' '}
            luôn đúng. Lab sao chép chỉ số luồng vào các <code>int</code> trước.
          </p>
        }
      >
        <p>
          The listing as printed does not compile: <code>N_s</code> has no element type, and the constant array is
          declared as <code>F_c</code> but used as <code>F</code>. Separately, <code>0.0</code> is a double where{' '}
          <code>0.0f</code> is meant; that compiles, but it is not what a float kernel wants. The lab fixes all three. One more change is about correctness on every backend:{' '}
          <code>threadIdx.x</code> is unsigned, so <code>threadIdx.x - FILTER_RADIUS</code> is unsigned arithmetic. The
          book assigns it to an <code>int</code> before comparing, which works; compare it before that conversion and a
          test such as <code>&gt;= 0</code> is always true. The lab copies the thread indices into <code>int</code>s first.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Quy tắc số không dấu được kiểm tra trong notes/today/ch07-facts.md, A8: clang báo phép kiểm tra{' '}
              <code>&gt;= 0</code> trên một biểu thức như vậy là luôn đúng. Kernel của mục 7.6 có đúng phép kiểm tra đó. Các
              thuộc tính vị trí luồng của Metal cũng là số không dấu.
            </p>
          }
        >
          <p>
            The unsigned rule is checked in notes/today/ch07-facts.md, A8: clang reports the <code>&gt;= 0</code> test on
            such an expression as always true. Section 7.6&apos;s kernel has exactly that test. Metal&apos;s thread-position
            attributes are unsigned too.
          </p>
        </Translated>
      </Aside>
      <Ask q={questionsS75['ch07/7.5/q-p175']} label={tr('Check the text against the figure', 'Đối chiếu văn bản với hình')} />

      <h2>
        <Translated vi="Đi từng bước">Step through it</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trình chạy từng bước chạy kernel đã sửa cho mọi luồng của một khối. Hãy theo luồng (tx 1, ty 1) trước: nó nạp
            N[0][0] vào N_s[1][1], chờ, rồi đọc mảng con của nó từ góc N_s[0][0]. Sau đó bỏ chọn một biện pháp bảo vệ, giữ
            thứ tự &quot;mỗi luồng chạy xa nhất có thể&quot;, và nhấn <i>Đến lỗi đầu tiên</i>. Thử tương tự với các luồng
            chạy lockstep.
          </p>
        }
      >
        <p>
          The stepper runs the corrected kernel for every thread of one block. Follow thread (tx 1, ty 1) first: it loads
          N[0][0] into N_s[1][1], waits, and then reads its patch from the corner N_s[0][0]. Then untick one safeguard,
          keep the order &quot;each thread runs as far as it can&quot;, and press <i>Go to the first failure</i>. Try the
          same with the threads in lockstep.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi luồng tiến một thao tác ở mức mã nguồn mỗi lần (một lần nạp, một tap bộ lọc, một lần ghi); phần cứng đan
              xen các lệnh và chạy theo warp. &quot;Lockstep&quot; là một warp đơn được lý tưởng hóa. Bộ nhớ chia sẻ chưa ai
              ghi được đọc ra NaN, trường hợp xấu nhất. Không có thời gian, không có bộ nhớ đệm. Mô hình trung thực về việc
              một luồng thấy giá trị nào trong một ô khi nó đọc ô đó.
            </p>
          }
        >
          <p>
            A thread advances one source-level operation at a time (a load, one filter tap, a store); hardware interleaves
            instructions and runs warps. &quot;Lockstep&quot; is an idealised single warp. Shared memory nobody wrote
            reads as NaN, the worst case. No timing and no caches. The model is faithful about which value a thread finds
            in a slot when it reads it.
          </p>
        </Translated>
      </Simplification>
      <HaloStepper />
      <HaloOutcomes />
      <Translated
        vi={
          <p>
            Mỗi lỗi hỏng theo cách riêng. Không có rào chắn, một luồng chạy trước sẽ đọc các ô mà luồng bên cạnh chưa điền,
            và lockstep che điều đó vì khi đó mọi lần nạp xong trước bất kỳ lần đọc nào. Không có số không cho ô ma, chỉ các
            đầu ra ở viền ảnh bị sai, vì chỉ mảng con của chúng chạm vào ô ma. Không có độ lệch, mọi giá trị được tính đều
            đúng nhưng cả ô đã bị dịch r phần tử xuống dưới và sang phải, nên hàng và cột đầu tiên của P không bao giờ được
            ghi.
          </p>
        }
      >
        <p>
          Each defect fails in its own way. Without the barrier, a thread that runs ahead reads slots its neighbours have
          not filled yet, and lockstep hides it because all loads then finish before any read. Without the ghost zeros,
          only the outputs on the image&apos;s border are wrong, because only their patches touch a ghost slot. Without
          the offset, every value computed is right but the whole tile has moved r elements down and right, so the first
          row and column of P are never written.
        </p>
      </Translated>
      <Ask q={questionsS75['ch07/7.5/q-one-barrier']} />

      <h2>
        <Translated vi="Nó tiết kiệm bao nhiêu lưu lượng">How much traffic it saves</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hãy đếm cho một khối nằm xa các cạnh ảnh, như sách làm. Với ô đầu ra t × t và bộ lọc m × m, khối thực hiện t² ·
            m² phép nhân-cộng, mỗi phép 2 FLOP. Nó nạp ô đầu vào một lần: (t + m − 1)² phần tử 4 byte. Và nó ghi t² đầu ra 4
            byte. Các kernel trước đã bỏ qua lệnh ghi, vì mỗi luồng ghi một lần trong khi nạp rất nhiều lần bên trong vòng
            lặp bộ lọc. Giờ các lần nạp cũng đã rời khỏi vòng lặp, và lệnh ghi không còn nhỏ bé bên cạnh chúng nữa.
          </p>
        }
      >
        <p>
          Count for one block away from the image&apos;s edges, as the book does. With a t × t output tile and an m × m
          filter, the block does t² · m² multiply-adds, 2 FLOP each. It loads the input tile once: (t + m − 1)² elements
          of 4 bytes. And it stores t² outputs of 4 bytes. The earlier kernels ignored their stores, because each thread
          stored once while it loaded inside the filter loop many times. Now the loads have left the loop too, and the
          stores are no longer small beside them.
        </p>
      </Translated>
      <Source pages="176" />
      <Translated
        vi={
          <p>
            Chia ra, và tỉ lệ có một dạng đáng chú ý: (m²/4) · 1 / (½ (1 + (m − 1)/t)² + ½). Thừa số đầu là mức lý tưởng từ
            mục 7.3, thứ mà một kernel chạm vào mỗi phần tử đúng một lần sẽ đạt. Thừa số thứ hai là một hệ số suy giảm nhỏ
            hơn 1, tăng dần về 1 khi t lớn lên, vì một ô lớn hơn nạp mỗi phần tử một lần cho nhiều đầu ra hơn. Với ví dụ nhỏ
            của Hình 7.13 (t = 6, m = 3), nó cho 1.62 FLOP/B so với mức lý tưởng 2.25. Với ví dụ xuyên suốt của sách, bộ lọc
            5 × 5 với khối 32 × 32 và do đó ô đầu ra 28 × 28, nó cho 5.42 so với 6.25.
          </p>
        }
      >
        <p>
          Divide, and the ratio takes a telling shape: (m²/4) · 1 / (½ (1 + (m − 1)/t)² + ½). The first factor is the
          ideal from section 7.3, what a kernel that touched each element exactly once would reach. The second is an
          attenuation factor below 1 that climbs towards 1 as t grows, because a bigger tile loads each element once for
          more outputs. For the toy of Fig. 7.13 (t = 6, m = 3) it gives 1.62 FLOP/B against an ideal of 2.25. For the
          book&apos;s running example, a 5 × 5 filter with a 32 × 32 block and so a 28 × 28 output tile, it gives 5.42
          against 6.25.
        </p>
      </Translated>
      <Source pages="176–177" figs={tr('Fig. 7.14', 'Hình 7.14')} />
      <Translated
        vi={
          <p>
            Hình 7.14 lập bảng công thức cho sáu kích thước bộ lọc và năm độ rộng ô, tô màu bị giới hạn bởi bộ nhớ hoặc bị
            giới hạn bởi tính toán. Hình không nêu tên thiết bị nào, nhưng cách tô màu khớp với ngưỡng 20 FLOP/B của thiết bị
            ví dụ của sách. Trình khám phá tính lại bảng và cho bạn đặt ngưỡng theo thiết bị của chính bạn, từ lab 5.1.
          </p>
        }
      >
        <p>
          Fig. 7.14 tabulates the formula for six filter sizes and five tile widths, coloured memory-bound or
          compute-bound. The figure names no device, but its colouring matches the 20 FLOP/B threshold of the book&apos;s
          example device. The explorer recomputes the table and lets you set the threshold to your own device&apos;s,
          from lab 5.1.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch07-facts.md, A7)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Mọi ô xanh lá của Hình 7.14 đều trên 20 FLOP/B và mọi ô hồng đều dưới, khớp với thiết bị ví dụ của sách (một
              H100 SXM: 67 TFLOPS FP32 và 3.35 TB/s). Đó là ví dụ của sách, không phải GPU của bạn.
            </p>
          }
        >
          <p>
            Every green cell of Fig. 7.14 is above 20 FLOP/B and every pink one below it, which matches the book&apos;s
            example device (an H100 SXM: 67 TFLOPS FP32 and 3.35 TB/s). That is the book&apos;s example, not your GPU.
          </p>
        </Translated>
      </Today>
      <TileExplorer />
      <Translated
        vi={
          <p>
            Hai điều cần rút ra. Cường độ chỉ tiến tới m²/4 với ô lớn: hệ số suy giảm cần t lớn gấp vài lần m. Và khi khối cố
            định ở 32 × 32, một bộ lọc lớn hơn vừa nâng mức lý tưởng vừa thu nhỏ ô đầu ra, nên kernel càng tụt xa mức lý tưởng
            đúng lúc mức lý tưởng trở nên đáng để vươn tới.
          </p>
        }
      >
        <p>
          Two things to take from it. Intensity approaches m²/4 only for large tiles: the attenuation factor needs t
          several times larger than m. And with the block fixed at 32 × 32, a bigger filter both raises the ideal and
          shrinks the output tile, so the kernel falls further behind its ideal just as the ideal becomes worth reaching.
        </p>
      </Translated>
      <Ask q={questionsS75['ch07/7.5/q-ai9']} label={tr('Work it out', 'Tự tính')} />

      <h2>
        <Translated vi="Điều gì giới hạn ô">What limits the tile</Translated>
      </h2>
      <Translated
        vi={
          <p>
            t không thể lớn tùy ý. Trong thiết kế của sách, khối là ô đầu vào, nên giới hạn kích thước khối chặn trên ô đầu
            vào: 1024 luồng trên thiết bị của sách là 32 × 32, để lại 32 − 2r cho đầu ra. Bộ nhớ chia sẻ cũng chặn nó, dù một
            ô 32 × 32 số float chỉ có 4 KB. Chương 8 phát triển các cách vượt qua cả hai giới hạn.
          </p>
        }
      >
        <p>
          t cannot grow at will. In the book&apos;s design the block is the input tile, so the block-size limit caps the
          input tile: 1024 threads on the book&apos;s device is 32 × 32, leaving 32 − 2r for the output. Shared memory caps
          it too, though a 32 × 32 tile of floats is only 4 KB. Chapter 8 develops ways around both limits.
        </p>
      </Translated>
      <Source pages="176" />
      <Translated
        vi={
          <p>
            Trên HIP và Metal, giới hạn thuộc về kernel đã biên dịch. HIP biên dịch một kernel cho khối lớn nhất của thiết bị
            trừ khi <code>__launch_bounds__</code> nói khác và từ chối lần khởi chạy vượt giới hạn đó, nên hãy đọc{' '}
            <code>maxThreadsPerBlock</code> (doctor có báo); trên wavefront rộng 64, một khối 32 × 32 là 16 wavefront. Trên
            Metal, giới hạn là <code>maxTotalThreadsPerThreadgroup</code> của pipeline, và vì một function constant không thể
            định kích thước cho một mảng <code>threadgroup</code>, kích thước ô đến từ một macro tiền xử lý.
          </p>
        }
      >
        <p>
          On HIP and Metal the cap belongs to the compiled kernel. HIP compiles a kernel for the device&apos;s largest
          block unless <code>__launch_bounds__</code> says otherwise and refuses a launch above that bound, so read{' '}
          <code>maxThreadsPerBlock</code> (the doctor reports it); on a 64-wide wavefront a 32 × 32 block is 16
          wavefronts. On Metal the limit is the pipeline&apos;s <code>maxTotalThreadsPerThreadgroup</code>, and because a
          function constant cannot size a <code>threadgroup</code> array, the tile size comes from a preprocessor macro.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch07-facts.md B6, C3; ch05-facts.md C3)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Giới hạn khởi chạy mặc định của HIP là mức tối đa của thiết bị. Function constant của Metal không thể là mảng,
              và không thể định kích thước cho một mảng <code>threadgroup</code>.
            </p>
          }
        >
          <p>
            HIP&apos;s default launch bound is the device maximum. Metal function constants cannot be arrays, and cannot
            size a <code>threadgroup</code> array.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS75['ch07/7.5/q-org2']} />
      <Ask q={questionsS75['ch07/7.5/q-ex11']} label={tr('Work it out', 'Tự tính')} />

      <h2>
        <Translated vi="Ba cái giá của một khối rộng hơn đầu ra của nó">Three costs of a block wider than its output</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách nêu ba điểm kém hiệu quả của Hình 7.12, tất cả đều do khối là ô đầu vào. Thứ nhất, vòng luồng bên ngoài nạp
            rồi ngồi không. Khi kernel bị giới hạn bởi bộ nhớ, điều đó thậm chí có thể có ích, vì có nhiều luồng phát lệnh nạp
            hơn; khi nó bị giới hạn bởi tính toán, nó lãng phí gần một phần tư khối (240 trên 1024 luồng khi r = 2). Thứ hai,
            ô đầu ra không phải lũy thừa của hai (rộng 28 với bộ lọc 5 × 5), nên các hàng mà một warp ghi không thẳng hàng với
            các đơn vị mà bộ nhớ di chuyển. Thứ ba, bộ lọc lớn hơn buộc ô đầu ra nhỏ hơn, làm giảm cường độ và khiến hai điểm
            đầu tệ hơn.
          </p>
        }
      >
        <p>
          The book names three inefficiencies of Fig. 7.12, all caused by the block being the input tile. First, the
          outer ring of threads loads and then idles. When the kernel is memory-bound that may even help, since more
          threads issue loads; when it is compute-bound it wastes nearly a quarter of the block (240 of 1024 threads at r = 2).
          Second, the output tile is not a power of two (28 wide for a 5 × 5 filter), so the rows a warp stores do not
          line up with the units memory moves. Third, a bigger filter forces a smaller output tile, which lowers the
          intensity and makes the first two worse.
        </p>
      </Translated>
      <Source pages="177" />
      <Translated
        vi={
          <p>
            Cái giá thứ hai là thứ cần nhìn thấy. Bên dưới, một warp nạp hàng của nó trong ô đầu vào và ghi hàng của nó trong
            đầu ra, theo quy tắc được ghi trong tài liệu của NVIDIA (sector 32 byte) hoặc cache line của AMD. Hãy dự đoán lệnh
            ghi trước: 28 float là 112 byte, ba sector rưỡi.
          </p>
        }
      >
        <p>
          The second cost is the one to see. Below, one warp loads its row of the input tile and stores its row of the
          output, under NVIDIA&apos;s documented rule (32-byte sectors) or AMD&apos;s cache lines. Predict the store
          first: 28 floats are 112 bytes, three and a half sectors.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi lần một lệnh, không có bộ nhớ đệm, nên các đoạn mà một lệnh sau sẽ thấy đã nằm trong cache được đếm lại. Warp
              là một hàng khối gồm 32 luồng (hai hàng với warp rộng 64). Các hàng ảnh bắt đầu tại byte row × width × 4 từ một
              vùng cấp phát được căn chỉnh.
            </p>
          }
        >
          <p>
            One instruction at a time, no cache, so segments that a later instruction would find cached are counted
            again. The warp is one block row of 32 threads (two rows for a 64-wide warp). Image rows start at byte row ×
            width × 4 from an aligned allocation.
          </p>
        </Translated>
      </Simplification>
      <RowCoalescing />
      <Translated
        vi={
          <p>
            Khi các hàng được căn chỉnh, một lệnh ghi 28 float luôn tốn 4 sector, trong khi 3.5 là đủ: một phần tám lượng
            được di chuyển không được dùng. Đặt chiều rộng ảnh khác bội số của 8 float, và một số hàng bắt đầu giữa sector và
            cần 5. Hàng của ô đầu vào còn tệ hơn: nó bắt đầu r phần tử trước một bội số của 28, nên với mọi khối trừ khối đầu
            tiên, nó vắt qua 5 sector để chuyển lượng dữ liệu đáng 4 sector. Một ô rộng 32 bắt đầu tại bội số của 32 di chuyển
            đúng 4 sector nó dùng.
          </p>
        }
      >
        <p>
          With rows aligned, a 28-float store always takes 4 sectors, where 3.5 would do: an eighth of what is moved is
          not used. Make the image width something other than a multiple of 8 floats, and some rows start mid-sector and
          need 5. The input-tile row is worse: it starts r elements before a multiple of 28, so for every block but the
          first it straddles 5 sectors to deliver 4 sectors&apos; worth. A 32-wide tile starting at a multiple of 32
          moves exactly the 4 sectors it uses.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md A2, A3, B1, C1)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              NVIDIA: các truy cập của một warp được hợp nhất thành các transaction 32 byte, và một dãy float liên tiếp không
              căn chỉnh tốn năm trong khi bốn là đủ. AMD gom các địa chỉ của một wavefront thành các yêu cầu cache line 64 hoặc
              128 byte. Apple không ghi tài liệu về quy tắc coalescing, kích thước line hay kích thước sector.
            </p>
          }
        >
          <p>
            NVIDIA: a warp&apos;s accesses coalesce into 32-byte transactions, and an unaligned run of consecutive floats
            takes five where four would do. AMD groups a wavefront&apos;s addresses into 64- or 128-byte cache-line
            requests. Apple documents no coalescing rule, line size or sector size.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Cách chữa là hai cách mà sách nêu. Cách tổ chức 2 cho mọi luồng tính toán, và cho phép ô đầu ra rộng 32, nên các
            lệnh ghi thẳng hàng. Hoặc hoàn toàn không nạp vùng biên vào bộ nhớ chia sẻ, mà đọc nó từ bộ nhớ toàn cục với hy
            vọng một bộ nhớ đệm đã giữ sẵn nó: đó là <a href={href('ch07', '7.6')}>mục 7.6</a>.
          </p>
        }
      >
        <p>
          The remedies are the two the book names. Organisation 2 makes every thread compute, and lets the output tile
          be 32 wide, so stores line up. Or do not load the halo into shared memory at all, and read it from global memory
          in the hope that a cache already holds it: that is <a href={href('ch07', '7.6')}>section 7.6</a>.
        </p>
      </Translated>

      <h2>
        <Translated vi="Kỹ thuật này xuất hiện ở đâu khi phục vụ LLM">Where this shows up in LLM serving</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Phân ô có vùng biên không phải chi phí chính của một transformer, nhưng nó xuất hiện trong các mô hình mà
            llama.cpp phục vụ ngày nay. Các tệp mô hình của nó cho Mamba, Qwen3-Next, LFM và các mô hình khác chạy một phép
            tích chập 1D nhân quả ngắn trên từng kênh của dòng token. Khi xử lý prompt, kernel CUDA của llama.cpp nạp d_conv
            − 1 + 32 cột mỗi kênh vào bộ nhớ chia sẻ cho 32 đầu ra, chờ ở một rào chắn, và đọc từng cửa sổ từ đó: một ô đầu
            vào có vùng biên, chỉ một phía vì bộ lọc chỉ nhìn về phía sau.
          </p>
        }
      >
        <p>
          Halo tiling is not a transformer&apos;s main cost, but it appears in models that llama.cpp serves today. Its
          model files for Mamba, Qwen3-Next, LFM and others run a short causal 1D convolution over each channel of the
          token stream. For prompt
          processing, llama.cpp&apos;s CUDA kernel loads d_conv − 1 + 32 columns per channel into shared memory for 32
          outputs, waits at a barrier, and reads each window from there: an input tile with a halo, one-sided because
          the filter only looks back.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Từ <code>ssm-conv.cu</code> và các tệp mô hình của llama.cpp, kiểm tra ngày 2026-11-20
              (notes/today/ch07-facts.md D3). Độ rộng bộ lọc của nó là một tham số template được chọn bởi một switch trên 2,
              3, 4, 5, 7, 9 và 15 tap: cùng thủ thuật bán kính lúc biên dịch mà lab 7.2 dùng. Chi phí này nhỏ so với các
              phép nhân ma trận của mô hình.
            </p>
          }
        >
          <p>
            From llama.cpp&apos;s <code>ssm-conv.cu</code> and model files, checked 2026-11-20 (notes/today/ch07-facts.md
            D3). Its filter width is a template parameter chosen by a switch over 2, 3, 4, 5, 7, 9 and 15 taps: the same
            compile-time-radius device lab 7.2 uses. The cost is small next to the model&apos;s matrix multiplications.
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Đo so với công thức">Measured against the formula</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Công thức dự đoán lưu lượng, không phải thời gian. Nếu lab 5.2 cho bạn ít hơn hệ số T của mô hình, bạn đã thấy vì
            sao: mô hình tính mọi lần đọc lặp lại là một chuyến đi tới DRAM, trong khi bộ nhớ đệm phục vụ nhiều lần trong số
            đó. Lab 7.2 đo thời gian kernel phân ô có vùng biên so với một kernel bộ nhớ hằng có cùng bán kính lúc biên dịch,
            qua nhiều kích thước ô và bán kính, và ghi tỉ lệ lưu lượng theo công thức bên cạnh mỗi số đo.
          </p>
        }
      >
        <p>
          The formula predicts traffic, not time. If lab 5.2 gave you less than the model&apos;s factor of T, you have
          seen why: the model counts every repeated read as a trip to DRAM, and caches serve many of them. Lab 7.2
          times the halo-tiled kernel against a constant-memory kernel with the same compile-time radius, over several
          tile sizes and radii, and records the formula&apos;s traffic ratio beside each measurement.
        </p>
      </Translated>
      <Predict
        id="ch07/7.5/p-speedup"
        prompt={
          <Translated
            vi={
              <>
                Với khối 32 × 32 và bộ lọc 5 × 5, công thức nói kernel phân ô yêu cầu ít lưu lượng toàn cục hơn 5.42 / 0.5,
                khoảng 10.8 lần, so với kernel bộ nhớ hằng. Bạn kỳ vọng nó chạy nhanh hơn bao nhiêu lần trên thiết bị{' '}
                <em>của bạn</em>? Hãy chốt một con số trước khi chạy lab 7.2.
              </>
            }
          >
            <>
              With a 32 × 32 block and a 5 × 5 filter, the formula says the tiled kernel asks for 5.42 / 0.5, about 10.8
              times, less global traffic than the constant-memory kernel. How many times faster do you expect it to run on{' '}
              <em>your</em> device? Commit a number before running lab 7.2.
            </>
          </Translated>
        }
        unit={tr('× faster', '× nhanh hơn')}
      >
        {(v) => <MeasuredAgainstFormula predicted={v} />}
      </Predict>
      <Translated
        vi={
          <p>
            Lab và hướng dẫn của nó nằm trên <a href={href('ch07', 'labs')}>trang lab</a>. Phần thứ hai của nó sửa kernel của
            mục 7.6, thứ bạn sẽ đọc tiếp theo.
          </p>
        }
      >
        <p>
          The lab and its instructions are on the <a href={href('ch07', 'labs')}>labs page</a>. Its second part repairs
          section 7.6&apos;s kernel, which you will read next.
        </p>
      </Translated>
    </>
  );
}

