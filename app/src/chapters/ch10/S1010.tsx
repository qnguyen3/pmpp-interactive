import { useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';
import { useLabStatus } from './status';
import { BOOK_CAPACITY, deviceSlots, residentUpperBound } from '../../sim/coarsen';
import {
  CH6_DEVICE_R,
  FIG1021,
  SOFTMAX_BATCHES,
  SOFTMAX_BLOCK,
  SOFTMAX_ROW,
  coarsenSteps,
  fewestStepsC,
  justEnoughC,
  maxUsefulC,
  softmaxPreset,
  sweepC,
  type CoarsenResult,
} from '../../sim/multiblock';
import { functionSource } from '../../sim/toolchain';

const LAB = 'ch10/lab1_sum';
const fmtInt = (n: number) => n.toLocaleString('en-US');
const pct = (v: number) => `${Math.round(v * 100)}%`;

const SUM_TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', output_ref: 'global',
  partialSums_s: 'shared', input_s: 'shared',
  partialSum: 'reg', segment: 'reg', i: 'reg', t: 'reg', c: 'reg', idx: 'reg',
};

// ---------------------------------------------------------------- the listing
/**
 * 1-based lines of the kernel to mark: the segment, the first load, and the serial
 * loop up to its closing brace.
 */
export function coarseningLines(text: string): number[] {
  const lines = text.split('\n');
  const seg = lines.findIndex((l) => /\bsegment\s*=/.test(l));
  const loop = lines.findIndex((l) => /for\s*\(.*\bc\s*=\s*1/.test(l));
  if (seg < 0 || loop < 0) return [];
  let end = loop;
  for (let d = 0, j = loop; j < lines.length; j++) {
    d += (lines[j].match(/\{/g) ?? []).length - (lines[j].match(/\}/g) ?? []).length;
    if (d <= 0 && j > loop) {
      end = j;
      break;
    }
  }
  const first = lines.slice(seg + 1, loop).findIndex((l) => /partialSum\s*=/.test(l));
  return [seg + 1, ...(first >= 0 ? [seg + 2 + first] : []), ...Array.from({ length: end - loop + 1 }, (_, i) => loop + 1 + i)];
}

function CoarsenListing() {
  const status = useLabStatus(LAB);
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  const k = cu ? functionSource(cu, 'CoarsenedSumReductionKernel') : null;
  if (!k)
    return (
      <p className="muted">
        <Translated vi={<>Kernel được nạp từ <code>labs/{LAB}/solution/kernel.cu</code> khi lab 10.1 đã sẵn sàng.</>}>The kernel is loaded from <code>labs/{LAB}/solution/kernel.cu</code> once lab 10.1 is in place.</Translated>
      </p>
    );
  const loop = coarseningLines(k.text);
  return (
    <Code
      tabs={[{ label: 'CUDA', code: k.text, tiers: SUM_TIERS, status: status.cu, highlight: loop }]}
      caption={
        <>
          <Translated
            vi={
              <>
                <code>CoarsenedSumReductionKernel</code> từ lời giải tham chiếu của lab 10.1 (<code>labs/{LAB}/solution/kernel.cu</code>,
                dòng {k.firstLine} đến {k.lastLine}). Các dòng được đánh dấu là phần gộp việc: đoạn tăng theo hệ số, lần nạp đầu
                tiên, và vòng lặp tuần tự chạy trước mọi rào chắn.
              </>
            }
          >
            <code>CoarsenedSumReductionKernel</code> from lab 10.1's reference solution (<code>labs/{LAB}/solution/kernel.cu</code>,
            lines {k.firstLine} to {k.lastLine}). The marked lines are the coarsening: the segment that grows by the factor, the
            first load, and the serial loop that runs before any barrier.
          </Translated>
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- interactive: coarsening explorer
const BLOCKS = [8, 32, 64, 128, 256, 512, 1024];
type RSource = 'fig' | 'slider' | 'ch6' | `yours:${string}`;

interface Yours { backend: string; device: string; R: number }
/** CUDA and HIP reports that carry multiProcessorCount and maxBlocksPerMultiProcessor (Metal reports neither). */
function yoursFrom(reports: Reports, block: number): Yours[] {
  return Object.keys(reports)
    .filter((b) => b === 'cuda' || b === 'hip')
    .flatMap((b) => {
      const d = deviceSlots(reports[b].device as Record<string, unknown>);
      return d ? [{ backend: b, device: String(reports[b].device.name ?? b), R: residentUpperBound(d, block) }] : [];
    });
}

function WaveBar({ r, scale, y, label }: { r: CoarsenResult; scale: number; y: number; label: string }) {
  const X0 = 150;
  const width = (steps: number) => (steps / scale) * 520;
  const perWave = r.stepsPerBlock;
  const drawWaves = r.waves <= 48;
  return (
    <g data-bar={label}>
      <text x={X0 - 8} y={y + 14} fontSize="11.5" textAnchor="end">{label}</text>
      {drawWaves ? (
        Array.from({ length: r.waves }, (_, w) => {
          const x = X0 + width(w * perWave);
          return (
            <g key={w}>
              <rect x={x} y={y} width={width(r.fullPerBlock)} height={20} fill="var(--ink-2)" />
              <rect x={x + width(r.fullPerBlock)} y={y} width={width(r.underPerBlock)} height={20} fill="url(#cx-hatch)" stroke="var(--ink-2)" strokeWidth="0.6" />
            </g>
          );
        })
      ) : (
        <>
          <rect x={X0} y={y} width={width(r.fullSteps)} height={20} fill="var(--ink-2)" />
          <rect x={X0 + width(r.fullSteps)} y={y} width={width(r.underSteps)} height={20} fill="url(#cx-hatch)" stroke="var(--ink-2)" strokeWidth="0.6" />
        </>
      )}
      <text x={X0 + width(r.totalSteps) + 6} y={y + 14} fontSize="11.5">
        <Translated
          vi={
            <>
              {fmtInt(r.totalSteps)} bước{drawWaves ? '' : ` (${fmtInt(r.waves)} đợt)`}
            </>
          }
        >
          {fmtInt(r.totalSteps)} steps{drawWaves ? '' : ` (${fmtInt(r.waves)} waves)`}
        </Translated>
      </text>
    </g>
  );
}

function SweepChart({ base, maxC, C, onPick }: { base: { N: number; block: number; R: number; rows: number }; maxC: number; C: number; onPick: (c: number) => void }) {
  const sw = sweepC(base, maxC);
  const top = Math.max(...sw.map((r) => r.totalSteps));
  const best = fewestStepsC(base, maxC);
  const rule = justEnoughC(base, maxC);
  const X0 = 40;
  const PW = 640;
  const PH = 120;
  const VW = X0 + PW + 20;
  const bw = PW / sw.length;
  const bx = (c: number) => X0 + (c - 1) * bw;
  // Never drawn smaller than its own size, so the text stays legible: a narrow screen
  // scrolls the chart sideways inside its box instead. The legend is HTML, so it wraps.
  return (
    <>
    <p style={{ margin: '0.6rem 0 0.2rem', fontSize: '0.85rem', fontWeight: 600 }}>
      <Translated vi={<>Tổng số bước với C = 1 đến {maxC} (bấm vào một cột để chọn hệ số đó)</>}>Total steps for C = 1 to {maxC} (click a bar to pick that factor)</Translated>
    </p>
    <div style={{ overflowX: 'auto' }} data-part="sweep">
    <svg viewBox={`0 0 ${VW} ${PH + 38}`} role="img" aria-label={tr('Total steps for every coarsening factor', 'Tổng số bước cho mọi hệ số gộp việc')} style={{ width: '100%', minWidth: VW, display: 'block' }}
      data-best={best} data-rule={rule}>
      {sw.map((r) => {
        const h = (r.totalSteps / top) * PH;
        const c = r.input.C;
        return (
          <rect key={c} x={bx(c)} y={20 + PH - h} width={Math.max(0.8, bw - (bw > 3 ? 1 : 0))} height={h}
            fill={c === C ? 'var(--ink)' : r.tooFew ? 'var(--rule)' : 'var(--ink-3)'} onClick={() => onPick(c)} style={{ cursor: 'pointer' }}>
            <title>{tr(`C = ${c}: ${r.blocks} blocks, ${r.waves} wave${r.waves === 1 ? '' : 's'}, ${r.totalSteps} steps`, `C = ${c}: ${r.blocks} khối, ${r.waves} đợt, ${r.totalSteps} bước`)}</title>
          </rect>
        );
      })}
      <line x1={X0} x2={X0 + PW} y1={20 + PH} y2={20 + PH} stroke="var(--rule)" />
      <text x={X0 - 4} y={24} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{fmtInt(top)}</text>
      <text x={X0 - 4} y={20 + PH} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>0</text>
      <text x={bx(1)} y={33 + PH} fontSize="10" style={{ fill: 'var(--ink-2)' }}>1</text>
      <text x={bx(sw.length) + bw} y={33 + PH} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{sw.length}</text>
      <polygon points={`${bx(best) + bw / 2},${23 + PH} ${bx(best) + bw / 2 - 5},${32 + PH} ${bx(best) + bw / 2 + 5},${32 + PH}`} fill="var(--ok)" />
      {rule !== best ? (
        <polygon points={`${bx(rule) + bw / 2},${23 + PH} ${bx(rule) + bw / 2 - 5},${32 + PH} ${bx(rule) + bw / 2 + 5},${32 + PH}`} fill="var(--ink)" />
      ) : null}
    </svg>
    </div>
    <p style={{ margin: '0.3rem 0 0', fontSize: '0.85rem' }} data-part="sweep-legend">
      <span style={{ color: 'var(--ok)' }}><Translated vi={<>▲ ít bước nhất: C = {best}</>}>▲ fewest steps: C = {best}</Translated></span>
      {' · '}
      {rule !== best ? <span><Translated vi={<>▲ quy tắc của sách (C lớn nhất mà vẫn có ít nhất R khối): C = {rule}</>}>▲ book's rule (largest C with at least R blocks): C = {rule}</Translated></span> : <span><Translated vi="quy tắc của sách chọn cùng C đó">the book's rule picks the same C</Translated></span>}
      <br />
      <span className="muted"><Translated vi="Cột nhạt: ít khối hơn R, nên một phần thiết bị ngồi không.">Pale bars: fewer blocks than R, so part of the device idles.</Translated></span>
    </p>
    </>
  );
}

function CoarseningExplorer() {
  const { reports } = useReports();
  const [N, setN] = useState(1 << 20);
  const [nText, setNText] = useState(String(1 << 20));
  const [rows, setRows] = useState(1);
  const [block, setBlock] = useState(256);
  const [C, setCRaw] = useState(1);
  const [cText, setCText] = useState('1');
  const [src, setSrc] = useState<RSource>('ch6');
  const [slider, setSlider] = useState(100);

  const yours = yoursFrom(reports, block);
  const R = src === 'fig' ? 1 : src === 'ch6' ? CH6_DEVICE_R : src === 'slider' ? slider : (yours.find((y) => 'yours:' + y.backend === src)?.R ?? CH6_DEVICE_R);
  const setC = (c: number) => {
    const v = Math.max(1, Math.min(1024, Math.floor(c)));
    setCRaw(v);
    setCText(String(v));
  };
  const setLength = (n: number) => {
    setN(n);
    setNText(String(n));
  };
  const r = coarsenSteps({ N, block, C, R, rows });
  const baseline = coarsenSteps({ N, block, C: 1, R, rows });
  const useful = maxUsefulC(N, block);
  const base = { N, block, R, rows };
  // Chart range: wide enough to show the best factor and the current one, narrow enough
  // that the bars near the best stay readable.
  const reach = Math.min(512, useful);
  const maxC = Math.min(reach, Math.max(16, 3 * fewestStepsC(base, reach), C));

  const preset = (p: { N: number; block: number; C: number; R: RSource; rows?: number }) => {
    setLength(p.N);
    setBlock(p.block);
    setC(p.C);
    setSrc(p.R);
    setRows(p.rows ?? 1);
  };

  return (
    <Bench
      title={tr('Coarsening explorer', 'Trình khám phá gộp việc')}
      task={tr('Start from Fig. 10.21, then raise C on 2²⁰ floats (about a million) until the warning appears. Find a factor that is worse than both its neighbours.', 'Bắt đầu từ Hình 10.21, rồi tăng C trên 2²⁰ float (khoảng một triệu) cho đến khi cảnh báo xuất hiện. Tìm một hệ số tệ hơn cả hai hệ số kề nó.')}
      foot={
        <>
          <Translated
            vi={
              <>
                Đặc: các bước được dùng trọn, mỗi luồng cộng các phần tử của riêng nó (2C − 1 mỗi khối). Gạch chéo: cây, nơi các luồng
                rời đi dần (log2 của kích thước khối, mỗi khối). Các khối chạy theo đợt R khối; một đợt tốn số bước của một khối, dù
                nó chứa ít khối đến đâu. Cột mờ là C = 1 trên cùng thiết bị.
              </>
            }
          >
            Solid: fully used steps, every thread adding its own elements (2C − 1 per block). Hatched: the tree, where threads
            drop out (log2 of the block size per block). Blocks run in waves of R; a wave costs one block's steps, however few
            blocks it holds. The faint bar is C = 1 on the same device.
          </Translated>
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="ui muted"><Translated vi="Thiết lập sẵn:">Presets:</Translated></span>
        <button onClick={() => preset({ ...FIG1021, C: 1, R: 'fig' })}><Translated vi="Hình 10.21 (a)">Fig. 10.21 (a)</Translated></button>
        <button onClick={() => preset({ ...FIG1021, C: 2, R: 'fig' })}><Translated vi="Hình 10.21 (b)">Fig. 10.21 (b)</Translated></button>
        <button onClick={() => preset({ N: 1 << 20, block: 256, C: 1, R: 'ch6' })}><Translated vi="2²⁰ float">2²⁰ floats</Translated></button>
        {SOFTMAX_BATCHES.map((b) => {
          const p = softmaxPreset(b);
          return (
            <button key={b} onClick={() => preset({ N: p.N, block: p.block, C: p.C, R: 'ch6', rows: b })}>
              <Translated vi={<>softmax, batch {b}</>}>softmax, batch {b}</Translated>
            </button>
          );
        })}
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          <Translated vi="N mỗi hàng">N per row</Translated>
          <input type="number" min={1} value={nText} aria-label="N"
            onChange={(e) => { setNText(e.target.value); const v = Math.floor(Number(e.target.value)); if (v >= 1 && v <= 1e10) setN(v); }}
            style={{ width: '8.5rem' }} />
        </label>
        <label className="field">
          <Translated vi="Số hàng">Rows</Translated>
          <input type="number" min={1} value={rows} aria-label={tr('rows', 'số hàng')} onChange={(e) => setRows(Math.max(1, Math.floor(Number(e.target.value)) || 1))} style={{ width: '5rem' }} />
        </label>
        <label className="field">
          <Translated vi="Số luồng mỗi khối">Threads per block</Translated>
          <select value={block} onChange={(e) => setBlock(+e.target.value)} aria-label={tr('threads per block', 'số luồng mỗi khối')}>
            {BLOCKS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Hệ số gộp việc C">Coarsening factor C</Translated>
          <input type="number" min={1} max={1024} value={cText} aria-label="C"
            onChange={(e) => { setCText(e.target.value); const v = Math.floor(Number(e.target.value)); if (v >= 1 && v <= 1024) setCRaw(v); }}
            style={{ width: '5.5rem' }} />
        </label>
        <input type="range" min={1} max={Math.max(64, Math.min(512, useful))} value={Math.min(C, 512)} aria-label={tr('C slider', 'Thanh trượt C')}
          onChange={(e) => setC(+e.target.value)} style={{ minWidth: '12rem' }} />
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <span className="ui muted"><Translated vi="Số khối cùng lúc, R:">Blocks at once, R:</Translated></span>
        <span className="seg" role="group" aria-label={tr('Blocks at once', 'Số khối cùng lúc')}>
          <button aria-pressed={src === 'fig'} onClick={() => setSrc('fig')}><Translated vi="1 (Hình 10.21)">1 (Fig. 10.21)</Translated></button>
          <button aria-pressed={src === 'ch6'} onClick={() => setSrc('ch6')}><Translated vi={<>{CH6_DEVICE_R} (ví dụ của sách, ch. 6)</>}>{CH6_DEVICE_R} (the book's example, ch. 6)</Translated></button>
          <button aria-pressed={src === 'slider'} onClick={() => setSrc('slider')}><Translated vi="tự chọn">choose</Translated></button>
          {yours.map((y) => (
            <button key={y.backend} aria-pressed={src === 'yours:' + y.backend} onClick={() => setSrc(`yours:${y.backend}`)}>
              <Translated vi={<>của bạn ({y.backend})</>}>yours ({y.backend})</Translated>
            </button>
          ))}
        </span>
        {src === 'slider' ? (
          <label className="field">
            R = {slider}
            <input type="range" min={1} max={2048} value={slider} onChange={(e) => setSlider(+e.target.value)} aria-label="R" />
          </label>
        ) : null}
      </div>
      <p className="muted" style={{ margin: '0 0 0.8rem' }} data-testid="r-note">
        {src.startsWith('yours:')
          ? (() => {
              const y = yours.find((x) => 'yours:' + x.backend === src)!;
              return <Translated vi={<>{y.device}: multiProcessorCount × maxBlocksPerMultiProcessor, giới hạn bởi số khe luồng cho khối {block} luồng, cho ra tối đa {fmtInt(y.R)}. Thanh ghi và bộ nhớ chia sẻ chỉ có thể làm nó thấp hơn.</>}><>{y.device}: multiProcessorCount × maxBlocksPerMultiProcessor, capped by thread slots for {block}-thread blocks, gives at most {fmtInt(y.R)}. Registers and shared memory can only lower it.</></Translated>;
            })()
          : yours.length
            ? <Translated vi={<>Một báo cáo CUDA hoặc HIP trên máy này cho ra một lựa chọn "của bạn". Metal không báo cáo con số nào trong hai, nên máy Mac không có.</>}><>A CUDA or HIP report on this machine gives a "yours" option. Metal reports neither number, so a Mac gets none.</></Translated>
            : <Translated vi={<>"Của bạn" xuất hiện khi một báo cáo CUDA hoặc HIP có multiProcessorCount và maxBlocksPerMultiProcessor. Metal không báo cáo cả hai, nên máy Mac không có lựa chọn đó.</>}><>"Yours" appears when a CUDA or HIP report carries multiProcessorCount and maxBlocksPerMultiProcessor. Metal reports neither, so a Mac has no such option.</></Translated>}
      </p>

      <div className="cols" style={{ marginBottom: '0.9rem' }}>
        <div className="panel" data-testid="outputs" data-blocks={r.blocks} data-waves={r.waves} data-total={r.totalSteps}
          data-full={r.fullSteps} data-under={r.underSteps} data-commits={r.commits} data-toofew={r.tooFew ? 1 : 0}>
          <p style={{ margin: 0 }}>
            <Translated
              vi={
                <>
                  Số khối: {rows > 1 ? <>{fmtInt(rows)} hàng × </> : null}ceil({fmtInt(N)} / (2·{C}·{block})) = <b>{fmtInt(r.blocks)}</b>
                </>
              }
            >
              Blocks: {rows > 1 ? <>{fmtInt(rows)} rows × </> : null}ceil({fmtInt(N)} / (2·{C}·{block})) = <b>{fmtInt(r.blocks)}</b>
            </Translated>
          </p>
          <p style={{ margin: '0.35rem 0 0' }}>
            <Translated
              vi={
                <>
                  Số đợt: ceil({fmtInt(r.blocks)} / {fmtInt(R)}) = <b>{fmtInt(r.waves)}</b>
                  {r.waves > 1 && r.lastWave < R ? <> (đợt cuối chứa {fmtInt(r.lastWave)} khối)</> : null}
                </>
              }
            >
              Waves: ceil({fmtInt(r.blocks)} / {fmtInt(R)}) = <b>{fmtInt(r.waves)}</b>
              {r.waves > 1 && r.lastWave < R ? <> (the last holds {fmtInt(r.lastWave)} blocks)</> : null}
            </Translated>
          </p>
          <p style={{ margin: '0.35rem 0 0' }}>
            <Translated
              vi={
                <>
                  Mỗi khối: {r.fullPerBlock} bước dùng trọn + {r.underPerBlock} bước dùng chưa hết = {r.stepsPerBlock} bước
                </>
              }
            >
              Per block: {r.fullPerBlock} fully used + {r.underPerBlock} underused = {r.stepsPerBlock} steps
            </Translated>
          </p>
          <p style={{ margin: '0.35rem 0 0' }}>
            <Translated
              vi={
                <>
                  Tổng: <b>{fmtInt(r.totalSteps)}</b> bước ({fmtInt(r.fullSteps)} dùng trọn, {fmtInt(r.underSteps)} dùng chưa hết)
                </>
              }
            >
              Total: <b>{fmtInt(r.totalSteps)}</b> steps ({fmtInt(r.fullSteps)} fully used, {fmtInt(r.underSteps)} underused)
            </Translated>
          </p>
          <p style={{ margin: '0.35rem 0 0' }}><Translated vi={<>Số lần ghi atomic: <b>{fmtInt(r.commits)}</b>, mỗi khối một lần</>}>Atomic commits: <b>{fmtInt(r.commits)}</b>, one per block</Translated></p>
        </div>
        <div className="panel">
          {r.tooFew ? (
            <p style={{ margin: 0 }} data-testid="warning">
              <Translated
                vi={
                  <>
                    <span className="chip fail">quá ít khối</span> {fmtInt(r.blocks)} khối cho {fmtInt(R)} khe:{' '}
                    {pct(r.busyShare)} thiết bị có việc, {pct(1 - r.busyShare)} ngồi không suốt cả kernel.
                  </>
                }
              >
                <span className="chip fail">too few blocks</span> {fmtInt(r.blocks)} block{r.blocks === 1 ? '' : 's'} for {fmtInt(R)} slots:{' '}
                {pct(r.busyShare)} of the device has work, {pct(1 - r.busyShare)} sits idle for the whole kernel.
              </Translated>
            </p>
          ) : (
            <p style={{ margin: 0 }} data-testid="warning">
              <Translated vi={<><span className="chip ok">thiết bị đã đầy</span> {pct(r.busyShare)} khe khối bận, từng đợt một.</>}><span className="chip ok">the device is full</span> {pct(r.busyShare)} of the block slots are busy, wave by wave.</Translated>
            </p>
          )}
          <p style={{ margin: '0.5rem 0 0' }}>
            <Translated
              vi={
                <>
                  So với C = 1 ({fmtInt(baseline.totalSteps)} bước, {fmtInt(baseline.blocks)} khối):{' '}
                  <b>{r.totalSteps < baseline.totalSteps ? `ít hơn ${fmt(baseline.totalSteps / r.totalSteps)}× số bước` : r.totalSteps === baseline.totalSteps ? 'như nhau' : `nhiều hơn ${fmt(r.totalSteps / baseline.totalSteps)}× số bước`}</b>.
                </>
              }
            >
              Against C = 1 ({fmtInt(baseline.totalSteps)} steps, {fmtInt(baseline.blocks)} blocks):{' '}
              <b>{r.totalSteps < baseline.totalSteps ? `${fmt(baseline.totalSteps / r.totalSteps)}× fewer steps` : r.totalSteps === baseline.totalSteps ? 'the same' : `${fmt(r.totalSteps / baseline.totalSteps)}× more steps`}</b>.
            </Translated>
          </p>
          {C > useful ? (
            <p style={{ margin: '0.5rem 0 0' }} className="muted">
              <Translated vi={<>Quá C = {useful}, một khối đã phủ cả một hàng; hệ số lớn hơn chỉ thêm các lần nạp vượt quá cuối.</>}>Past C = {useful} one block already covers a row; a larger factor only adds loads past the end.</Translated>
            </p>
          ) : null}
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
      <svg viewBox="0 0 760 74" role="img" aria-label={tr('Steps, wave by wave', 'Các bước, từng đợt một')} style={{ width: '100%', minWidth: 760, display: 'block' }}>
        <defs>
          <pattern id="cx-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-2)" strokeWidth="1.3" />
          </pattern>
        </defs>
        <g opacity={0.4}>
          <WaveBar r={baseline} scale={Math.max(baseline.totalSteps, r.totalSteps)} y={6} label="C = 1" />
        </g>
        <WaveBar r={r} scale={Math.max(baseline.totalSteps, r.totalSteps)} y={40} label={`C = ${C}`} />
      </svg>
      </div>
      <SweepChart base={base} maxC={maxC} C={C} onPick={setC} />
    </Bench>
  );
}

// ---------------------------------------------------------------- the measured sweep
function MeasuredCoarsening({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const found = realBackends(reports)
    .filter((b) => b !== 'race')
    .flatMap((b) => {
      const rows = benchRows(reports[b], LAB)
        .filter((r) => r.name === 'any')
        .flatMap((r) => {
          const c = num(r.coarse);
          const g = num(r.gb_per_s);
          return c != null && g != null ? [{ c, g, n: num(r.n), block: num(r.block) }] : [];
        })
        .sort((a, b) => a.c - b.c);
      return rows.length ? [{ b, device: String(reports[b].device.name ?? b), rows }] : [];
    });
  if (!found.length)
    return (
      <p>
        <Translated
          vi={
            <>
              Chưa có số đo nào trên máy này. Chạy <code>python3 labs/pmpp.py run 10.1 --solution --bench</code> trên một máy có
              GPU, rồi tải lại trang: ô này sẽ vẽ băng thông theo hệ số gộp việc.
            </>
          }
        >
          No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 10.1 --solution --bench</code> on a machine
          with a GPU, then reload the page: this box will chart bandwidth against the coarsening factor.
        </Translated>
      </p>
    );
  return (
    <>
      {found.map((f) => {
        const best = f.rows.reduce((a, b) => (b.g > a.g ? b : a));
        const top = best.g;
        const mine = f.rows.reduce((a, b) => (Math.abs(b.c - predicted) < Math.abs(a.c - predicted) ? b : a));
        return (
          <div key={f.b} data-backend={f.b} data-best={best.c} style={{ marginBottom: '0.8rem' }}>
            <p>
              <Translated
                vi={
                  <>
                    Trên <b>{f.device}</b> ({f.b})
                    {best.n ? <>, {fmtInt(best.n)} float{best.block ? <> theo khối {best.block}</> : null},</> : null} hệ số nhanh nhất
                    đo được là <b>C = {best.c}</b>, ở {fmt(best.g)} GB/s.{' '}
                    {mine.c === best.c
                      ? 'Đó chính là hệ số bạn chọn.'
                      : `Lựa chọn của bạn${mine.c !== predicted ? ` (gần nhất đã đo: C = ${mine.c})` : ''} chạy ở ${fmt(mine.g)} GB/s, bằng ${Math.round((mine.g / top) * 100)}% hệ số nhanh nhất.`}
                  </>
                }
              >
                On <b>{f.device}</b> ({f.b})
                {best.n ? <>, {fmtInt(best.n)} floats{best.block ? <> in blocks of {best.block}</> : null},</> : null} the fastest factor
                measured was <b>C = {best.c}</b>, at {fmt(best.g)} GB/s.{' '}
                {mine.c === best.c
                  ? 'That is the one you picked.'
                  : `Your pick${mine.c !== predicted ? ` (nearest measured: C = ${mine.c})` : ''} ran at ${fmt(mine.g)} GB/s, ${Math.round((mine.g / top) * 100)}% of the fastest.`}
              </Translated>
            </p>
            <div style={{ overflowX: 'auto' }}>
            <svg viewBox={`0 0 520 ${f.rows.length * 18 + 4}`} role="img" aria-label={tr(`Bandwidth by coarsening factor on ${f.b}`, `Băng thông theo hệ số gộp việc trên ${f.b}`)} style={{ width: '100%', maxWidth: 520, minWidth: 520, display: 'block' }}>
              {f.rows.map((row, i) => (
                <g key={row.c}>
                  <text x={52} y={i * 18 + 13} fontSize="11" textAnchor="end">C = {row.c}</text>
                  <rect x={58} y={i * 18 + 3} width={(row.g / top) * 360} height={13} fill={row.c === best.c ? 'var(--ok)' : 'var(--ink-3)'} />
                  <text x={64 + (row.g / top) * 360} y={i * 18 + 13} fontSize="11">{fmt(row.g)} GB/s{row.c === best.c ? tr(' (fastest)', ' (nhanh nhất)') : ''}</text>
                </g>
              ))}
            </svg>
            </div>
          </div>
        );
      })}
      <p className="muted">
        <Translated
          vi={
            <>
              Số đo từ máy của bạn, không phải giá trị kỳ vọng. Mô hình bước ở trên không có thời gian bộ nhớ, và ở kích thước này
              kernel bị giới hạn bởi bộ nhớ, nên đường cong bạn đo không nhất thiết có hình dạng của mô hình.
            </>
          }
        >
          Measurements from your machine, not expected values. The step model above has no memory time, and at this size the
          kernel is limited by memory, so the curve you measured need not have the model's shape.
        </Translated>
      </p>
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS1010: Record<string, Question> = {
  'ch10/10.10/q-fig1021': {
    kind: 'numeric',
    id: 'ch10/10.10/q-fig1021',
    prompt: (
      <Translated
        vi={
          <>
            Các khối 8 luồng, N = 64, và thiết bị chạy mỗi lần một khối (R = 1), như ở Hình 10.21. Với hệ số gộp việc 4, tổng
            cộng có bao nhiêu bước?
          </>
        }
      >
        <>
          Blocks of 8 threads, N = 64, and the device runs one block at a time (R = 1), as in Fig. 10.21. How many steps in
          total with a coarsening factor of 4?
        </>
      </Translated>
    ),
    answer: 10,
    unit: 'steps',
    explain: (
      <Translated
        vi={
          <p>
            Một khối phủ 2 × 4 × 8 = 64 phần tử, nên một khối làm hết mọi việc. Mỗi luồng cộng 8 phần tử của nó trong 7 bước dùng
            trọn, rồi 8 tổng từng phần gặp nhau trong một cây log2 8 = 3 bước: 10. Không gộp việc thì là 4 khối, chạy lần lượt,
            mỗi khối 1 + 3 = 4 bước: 16.
          </p>
        }
      >
        <p>
          One block covers 2 × 4 × 8 = 64 elements, so one block does everything. Each thread adds its 8 elements in 7 fully
          used steps, then the 8 partial sums meet in a tree of log2 8 = 3 steps: 10. Without coarsening it is 4 blocks, run
          one after another, of 1 + 3 = 4 steps each: 16.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.10, pp. 247–248 (Fig. 10.21)',
    vi: { unit: 'bước' },
  },
  'ch10/10.10/q-just-enough': {
    kind: 'choice',
    id: 'ch10/10.10/q-just-enough',
    prompt: (
      <Translated
        vi={
          <>
            N = 2<sup>20</sup> float, 256 luồng mỗi khối, trên một thiết bị giả định chạy 256 khối cùng lúc. Quy tắc của sách
            chọn hệ số gộp việc nào?
          </>
        }
      >
        <>
          N = 2<sup>20</sup> floats, 256 threads per block, on a hypothetical device that runs 256 blocks at once. Which
          coarsening factor does the book's rule pick?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="C = 1: song song nhiều nhất.">C = 1: the most parallelism.</Translated>, why: <Translated vi="2048 khối là 8 đợt; mỗi khối trả giá cho cây riêng và lần ghi riêng của nó.">2048 blocks are 8 waves; every block pays for its own tree and its own commit.</Translated> },
      { label: 'C = 4.', why: <Translated vi="512 khối là hai đợt đầy: gấp đôi số pha cây và gấp đôi số lần ghi so với C = 8.">512 blocks are two full waves: twice the tree phases and twice the commits of C = 8.</Translated> },
      { label: 'C = 8.', correct: true },
      { label: 'C = 16.', why: <Translated vi="128 khối lấp một nửa thiết bị; nửa kia không có gì để làm.">128 blocks fill half the device; the other half has nothing to do.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Quy tắc là để lại vừa đủ khối để lấp đầy phần cứng. Với C = 8, một khối phủ 2 × 8 × 256 = 4096 phần tử, nên
            2<sup>20</sup> / 4096 = 256 khối: đúng một đợt. Theo cách đếm bước, đó là 15 + 8 = 23 bước, so với 30 cho C = 4 và 39
            cho C = 16.
          </p>
        }
      >
        <p>
          The rule is to leave just enough blocks to fill the hardware. With C = 8 a block covers 2 × 8 × 256 = 4096 elements,
          so 2<sup>20</sup> / 4096 = 256 blocks: exactly one wave. In the step count that is 15 + 8 = 23 steps, against 30 for
          C = 4 and 39 for C = 16.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.10, p. 248',
  },
  'ch10/10.10/q-batch1': {
    kind: 'choice',
    id: 'ch10/10.10/q-batch1',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel softmax giao cho mỗi hàng một khối. Ở batch 1, trên một thiết bị chạy 100 khối cùng lúc, phép rút gọn giữ
            bận bao nhiêu phần khe khối, và có những lối thoát nào?
          </>
        }
      >
        <>
          A softmax kernel gives each row one block. At batch 1, on a device that runs 100 blocks at once, what fraction of the
          block slots does the reduction keep busy, and what are the ways out?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Toàn bộ: một khối dùng cả thiết bị nếu nó đủ lớn, tối đa 1,024 luồng.">All of them: one block uses the whole device if it is large enough, up to 1,024 threads.</Translated>, why: <Translated vi="Một khối chạy trên một SM; các SM khác không có gì để làm.">A block runs on one SM; the others have nothing to do.</Translated> },
      {
        label: <Translated vi="Khoảng 1%. Chia hàng cho nhiều khối (một lượt thứ hai hoặc một rào chắn toàn lưới), hoặc chấp nhận khi softmax chỉ là một phần nhỏ của bước.">About 1%. Split the row over blocks (a second pass or a grid barrier), or accept it when the softmax is a small part of the step.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Khoảng 1%, và không làm gì được: một hàng phải nằm trong một khối để max và tổng của nó khớp nhau.">About 1%, and nothing can be done: a row must stay in one block for its max and its sum to agree with each other.</Translated>, why: <Translated vi="Mục 10.9 chính là cách chia một hàng cho nhiều khối; lượt thứ hai làm các mảnh khớp nhau.">Section 10.9 is exactly how a row is split across blocks; the second pass makes the pieces agree.</Translated> },
      { label: <Translated vi="Khoảng 1%; tăng hệ số gộp việc, để mỗi luồng phủ nhiều phần hàng hơn.">About 1%; raise the coarsening factor, so that each thread covers more of the row.</Translated>, why: <Translated vi="Gộp việc còn giảm số khối hơn nữa. Nó là nguyên nhân, không phải cách chữa.">Coarsening reduces the number of blocks further. It is the cause, not the cure.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một khối trên 100 khe là 1%. Hai lối thoát là hai nửa của chương này: nhiều khối hơn cho mỗi hàng, nối với nhau bằng
            một lượt thứ hai hoặc, khi cả lưới vừa trên thiết bị, một rào chắn toàn lưới dưới một lần khởi chạy cooperative (cách
            llama.cpp làm cho các hàng dài); hoặc giữ mỗi hàng một khối và chấp nhận thiết bị ngồi không vì softmax chỉ là một
            phần nhỏ của một bước decode so với các phép nhân ma trận. Cách nào đúng là chuyện đo đạc, và đó là việc của lab 10.3.
          </p>
        }
      >
        <p>
          One block of 100 slots is 1%. The two exits are the two halves of this chapter: more blocks per row, joined by a
          second pass or, where the whole grid fits on the device, a grid-wide barrier under a cooperative launch (llama.cpp's
          route for long rows); or keep one block per row and accept the idle device because the softmax is a small part of a
          decode step next to the matrix multiplications. Which is right is a measurement, which is what lab 10.3 is for.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.10, p. 248; notes/today/ch10-facts.md A8, D1', // i18n-ignore: citation only
  },
  'ch10/10.10/q-stride': {
    kind: 'choice',
    id: 'ch10/10.10/q-stride',
    prompt: (
      <Translated
        vi={
          <>
            Trong Hình 10.20, mỗi luồng cộng <code>input[i + c*BLOCK_DIM]</code> với c = 1 .. 2·COARSE_FACTOR − 1. Vì sao bước
            theo <code>BLOCK_DIM</code> thay vì giao cho mỗi luồng 2·COARSE_FACTOR phần tử liên tiếp?
          </>
        }
      >
        <>
          In Fig. 10.20 each thread adds <code>input[i + c*BLOCK_DIM]</code> for c = 1 .. 2·COARSE_FACTOR − 1. Why step by{' '}
          <code>BLOCK_DIM</code> instead of giving each thread 2·COARSE_FACTOR consecutive elements?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Để mỗi luồng cộng các phần tử của nó theo cùng thứ tự như vòng lặp tuần tự.">So that each thread adds its elements in the same order as the sequential loop would.</Translated>, why: <Translated vi="Cả hai cách bố trí đều cộng các phần tử của mỗi luồng theo thứ tự tăng dần, nên điều đó không phân định được; và thứ tự của phép tổng không phải là vấn đề.">Both layouts add each thread's elements in increasing order, so that cannot decide between them; and the order of the sum is not the point.</Translated> },
      {
        label: <Translated vi="Để ở mỗi bước các luồng liên tiếp đọc các phần tử liên tiếp và các lần nạp của một warp được hợp nhất.">So that in each step consecutive threads read consecutive elements and a warp's loads coalesce.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Để tránh xung đột bank bộ nhớ chia sẻ giữa các luồng của một warp.">To avoid shared-memory bank conflicts between the threads of a warp.</Translated>, why: <Translated vi="Các lần nạp này đi tới bộ nhớ toàn cục; bộ nhớ chia sẻ chỉ được dùng sau vòng lặp.">These loads go to global memory; shared memory is used only after the loop.</Translated> },
      { label: <Translated vi="Vì vòng lặp không có rào chắn, và chỉ số theo bước nhảy giữ các luồng khỏi tranh chấp.">Because the loop has no barrier, and strided indices keep threads from racing.</Translated>, why: <Translated vi="Cách nào thì mỗi luồng cũng đọc các phần tử của riêng nó; không có gì được ghi trong vòng lặp.">Each thread reads its own elements either way; nothing is written in the loop.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đây là cách phân chia xen kẽ của histogram gộp việc ở chương 9. Ở bước c, luồng t đọc phần tử{' '}
            <code>segment + t + c·BLOCK_DIM</code>, nên 32 luồng của một warp đọc 32 phần tử kề nhau: một lần truy cập được hợp
            nhất, như mục 6.1 định nghĩa. Với các khúc liên tiếp, các lần đọc của một warp trong một bước sẽ cách nhau
            2·COARSE_FACTOR phần tử. Chia liền khúc là lựa chọn tự nhiên trên CPU, nơi một lõi chạy suốt khúc của riêng nó; trên
            GPU nó làm tản mát mọi lệnh của warp.
          </p>
        }
      >
        <p>
          This is the interleaved partitioning of chapter 9's coarsened histogram. In step c, thread t reads element{' '}
          <code>segment + t + c·BLOCK_DIM</code>, so the 32 threads of a warp read 32 neighbouring elements: one coalesced
          access, as section 6.1 defined it. With consecutive chunks, a warp's reads in one step would be
          2·COARSE_FACTOR elements apart. A contiguous split is the natural choice on a CPU, where one core streams through its
          own chunk; on a GPU it scatters every warp instruction.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.10, p. 247 (Fig. 10.20); ch. 9 §9.5; ch. 6 §6.1', // i18n-ignore: citation only
  },
};

export function S1010() {
  return (
    <>
      <Translated
        vi={
          <p>
            Mục 10.9 giao cho mỗi cặp phần tử một luồng: N phần tử, N/2 luồng, N/2048 khối 1024 luồng. Một đầu vào lớn khởi chạy
            nhiều khối hơn hẳn số phần cứng chạy được cùng lúc, và phần thừa phải chờ. Phần cứng đã tuần tự hóa chúng.
          </p>
        }
      >
        <p>
          Section 10.9 gives every pair of elements a thread: N elements, N/2 threads, N/2048 blocks of 1024. A large input
          launches far more blocks than the hardware runs at once, and the surplus waits. The hardware has serialised them.
        </p>
      </Translated>
      <Source section="10.10" pages="246" />

      <h2><Translated vi="Cái giá của song song hóa được trả theo từng khối">The price of parallelising is paid per block</Translated></h2>
      <Translated
        vi={
          <p>
            Hãy xem mỗi khối tốn gì. Đầu tiên mỗi luồng cộng hai phần tử của nó: một bước, mọi luồng đều bận. Rồi đến cây: một nửa
            số luồng rời đi ở mỗi bước, và mỗi bước cần một rào chắn hoặc một shuffle. Pha thứ hai đó dùng chưa hết phần cứng, và
            mọi khối được khởi chạy đều đi qua nó. Nếu các khối thật sự chạy cạnh nhau, đó là cái giá của song song. Nếu phần cứng
            đằng nào cũng chạy chúng lần lượt, đó là lãng phí.
          </p>
        }
      >
        <p>
          Look at what each block spends. First every thread adds its two elements: one step, every thread busy. Then the tree:
          half the threads drop out at each step, and every step needs a barrier or a shuffle. That second phase underuses the
          hardware, and every block launched goes through it. If the blocks really run side by side, it is the cost of
          parallelism. If the hardware will run them one after another anyway, it is waste.
        </p>
      </Translated>
      <Source section="10.10" pages="246" />
      <Translated
        vi={
          <p>
            Vậy khi đằng nào công việc cũng bị tuần tự hóa, hãy tự tuần tự hóa nó, ở chỗ rẻ: giao cho mỗi luồng nhiều phần tử hơn
            để tự cộng trước khi cây bắt đầu. Đó là gộp việc cho luồng (thread coarsening), từ chương 6.
          </p>
        }
      >
        <p>
          So when the work will be serialised regardless, serialise it yourself, where it is cheap: give each thread more
          elements to add on its own before the tree starts. That is thread coarsening, from chapter 6.
        </p>
      </Translated>
      <Source section="10.10" pages="246" figs="ch. 6 §6.5" />

      <h2><Translated vi="Kernel gộp việc">The coarsened kernel</Translated></h2>
      <Translated
        vi={
          <p>
            Hai thay đổi so với kernel nhiều khối. Đoạn tăng theo hệ số: một khối phủ{' '}
            <code>COARSE_FACTOR·2·blockDim.x</code> phần tử. Và trước bất kỳ phép rút gọn nào, mỗi luồng cộng{' '}
            <code>2·COARSE_FACTOR</code> phần tử trong một vòng lặp, bước theo <code>BLOCK_DIM</code>. Trong vòng lặp đó mọi luồng
            đều bận, không luồng nào chờ luồng nào, không có rào chắn, và ở mỗi bước khối đọc một dải liền nhau. Chỉ sau đó cây
            thông thường mới bắt đầu, trên mỗi luồng một tổng từng phần.
          </p>
        }
      >
        <p>
          Two changes to the multi-block kernel. The segment grows by the factor: a block covers{' '}
          <code>COARSE_FACTOR·2·blockDim.x</code> elements. And before any reduction, each thread adds{' '}
          <code>2·COARSE_FACTOR</code> elements in a loop, stepping by <code>BLOCK_DIM</code>. In that loop every thread is busy,
          no thread waits for another, there is no barrier, and in each step the block reads one contiguous run. Only then does
          the usual tree start, over one partial sum per thread.
        </p>
      </Translated>
      <Source section="10.10" pages="246–247" figs={tr('Figs. 10.19, 10.20', 'Hình 10.19, 10.20')} />
      <CoarsenListing />
      <Translated
        vi={
          <p>
            Trang sách chỉ in một phần kernel này: phần đầu không có kiểu trả về, và mọi thứ sau vòng lặp là một dấu lược thay cho
            phép rút gọn và lần ghi của Hình 10.18. Phiên bản của lab 10.1 là toàn bộ kernel, đã biên dịch và kiểm thử. Chỗ nó khác
            trang sách: nó trả về <code>void</code>; nó nhận <code>N</code> và kiểm tra mọi lần nạp với nó, nên một chỉ số vượt
            quá cuối sẽ cộng 0 (bài tập 5); hệ số là một tham số lúc chạy, <code>coarse</code>, và <code>blockDim.x</code> thay
            cho <code>BLOCK_DIM</code>, để một bản build có thể quét cả hai; và dấu lược được viết ra thành phép rút gọn warp hai
            giai đoạn và lần ghi atomic của Hình 10.18. Một lane mask có kiểu theo từng hãng, và <code>atomicAdd</code> kiểu cũ
            thay cho <code>cuda::atomic_ref</code> dưới HIP, nghĩa là cùng file đó được viết để biên dịch cho HIP; điều đó là
            unverified, vì chưa có gì ở đây được build với ROCm.
          </p>
        }
      >
        <p>
          The page prints this kernel only in part: the header has no return type, and everything after the loop is an
          ellipsis that stands for Fig. 10.18's reduction and commit. Lab 10.1's version is the whole kernel, compiled and
          tested. Where it differs from the page: it returns <code>void</code>; it takes <code>N</code> and checks every load
          against it, so an index past the end adds 0 (exercise 5); the factor is a run-time argument, <code>coarse</code>,
          and <code>blockDim.x</code> stands in for <code>BLOCK_DIM</code>, so one build can sweep both; and the ellipsis is
          written out as the two-stage warp reduction and the atomic commit of Fig. 10.18. A lane mask typed per vendor, and
          the legacy <code>atomicAdd</code> in place of <code>cuda::atomic_ref</code> under HIP, mean the same file is written
          to compile for HIP; that is unverified, since nothing here has been built with ROCm.
        </p>
      </Translated>
      <Source section="10.10" pages="247" figs={tr('Fig. 10.20', 'Hình 10.20')} />
      <Ask q={questionsS1010['ch10/10.10/q-stride']} />

      <h2><Translated vi="Đếm số bước">Counting steps</Translated></h2>
      <Translated
        vi={
          <p>
            Sách đếm lợi ích theo số bước. Lấy 32 phần tử và khối 8 luồng. Hai khối thông thường, chạy lần lượt, mỗi khối mất 4
            bước: 1 bước mọi luồng bận, 3 bước trong cây. Tổng cộng 8 bước, 6 bước trong đó dùng chưa hết. Một khối gộp việc hệ
            số hai làm cùng công việc trong 6 bước: 3 bước bận, mỗi luồng cộng 4 phần tử của nó, rồi cùng cây 3 bước.
          </p>
        }
      >
        <p>
          The book counts the gain in steps. Take 32 elements and 8-thread blocks. Two ordinary blocks, run one after the other,
          take 4 steps each: 1 with every thread busy, 3 in the tree. That is 8 steps, 6 of them underused. One block coarsened
          by two does the same work in 6: 3 busy steps, each thread adding its 4 elements, then the same 3-step tree.
        </p>
      </Translated>
      <Source section="10.10" pages="247–248" figs={tr('Fig. 10.21', 'Hình 10.21')} />
      <Translated
        vi={
          <p>
            Trình khám phá tổng quát hóa cách đếm đó. Mỗi khối, gộp việc hệ số C cho ra 2C − 1 bước bận và giữ cây ở log2 của kích
            thước khối. Các khối chạy theo đợt R khối, số khối thiết bị giữ được cùng lúc, và một đợt tốn số bước của một khối dù
            nó chứa ít khối đến đâu. Hãy bắt đầu với hai thiết lập sẵn của Hình 10.21.
          </p>
        }
      >
        <p>
          The explorer generalises that count. Per block, coarsening by C gives 2C − 1 busy steps and leaves the tree at log2 of
          the block size. Blocks run in waves of R, the number the device holds at once, and a wave costs one block's steps
          however few blocks it holds. Start with the two Fig. 10.21 presets.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Cách tính của sách: mọi bước tốn như nhau, các khối chạy theo đợt trọn vẹn, không gì phải chờ bộ nhớ, và khởi chạy cùng
              atomic là miễn phí. Nó cho thấy gộp việc giúp ở đâu và ngừng giúp ở đâu, không phải một kernel chạy nhanh đến đâu.
            </p>
          }
        >
          <p>
            The book's accounting: every step costs the same, blocks run in whole waves, nothing waits for memory, and launches
            and atomics are free. It shows where coarsening helps and where it stops helping, not how fast a kernel runs.
          </p>
        </Translated>
      </Simplification>
      <CoarseningExplorer />
      <Ask q={questionsS1010['ch10/10.10/q-fig1021']} label={tr('Work it out', 'Tự tính ra')} />

      <h2><Translated vi="Gộp việc đến mức nào">How far to coarsen</Translated></h2>
      <Translated
        vi={
          <p>
            Gộp việc đánh đổi tính song song lấy hiệu suất, và sự đánh đổi có giới hạn. Mỗi lần tăng hệ số lại khởi chạy ít khối
            hơn. Khi số khối ít hơn số thiết bị chạy được cùng lúc, các khe rảnh cứ rảnh suốt cả kernel, và mỗi lần tăng thêm chỉ
            kéo dài đợt duy nhất còn lại. Hệ số tốt nhất để lại vừa đủ khối để lấp đầy phần cứng, nên nó phụ thuộc vào cả kích
            thước đầu vào lẫn thiết bị.
          </p>
        }
      >
        <p>
          Coarsening trades parallelism for efficiency, and the trade runs out. Each step up in the factor launches fewer
          blocks. Once there are fewer blocks than the device can run at once, the idle slots stay idle for the whole kernel,
          and every further step up only lengthens the one wave that remains. The best factor leaves just enough blocks to fill
          the hardware, so it depends on both the input size and the device.
        </p>
      </Translated>
      <Source section="10.10" pages="248" />
      <Translated
        vi={
          <p>
            Trong trình khám phá, quy tắc của sách được đánh dấu trên biểu đồ là hệ số lớn nhất vẫn khởi chạy ít nhất R khối.
            Trên 2<sup>20</sup> float, với 264 khe của thiết bị giả định của sách ở chương 6, nó chọn C = 7, khởi chạy 293 khối:
            một đợt đầy và một đợt thứ hai 29 khối. C = 8 khởi chạy 256 và xong trong một đợt, với 8 khe rảnh. Các đợt trọn vẹn
            khiến quy tắc chỉ là gần đúng, và khiến một số hệ số tệ hơn cả hai hệ số kề nó.
          </p>
        }
      >
        <p>
          In the explorer, the book's rule is marked on the chart as the largest factor that still launches at least R blocks.
          On 2<sup>20</sup> floats, with the 264 slots of the book's hypothetical device in chapter 6, it picks C = 7, which launches 293 blocks: one full wave and a second
          wave of 29. C = 8 launches 256 and finishes in one wave, with 8 slots idle. Whole waves make the rule approximate,
          and they make some factors worse than both neighbours.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Đọc quy tắc là "C lớn nhất mà vẫn có ít nhất R khối" là của chúng tôi, cũng như phép so sánh với hệ số ít bước nhất.
              Sách chỉ phát biểu quy tắc bằng lời.
            </p>
          }
        >
          <p>
            Reading the rule as "the largest C with at least R blocks" is ours, as is the comparison with the fewest-steps
            factor. The book states the rule in words only.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS1010['ch10/10.10/q-just-enough']} />
      <Translated
        vi={
          <p>
            Các thư viện cũng đưa ra lựa chọn này. Khi đọc một phép rút gọn production, hãy tìm số phần tử mà mỗi luồng xử lý
            trước cây: đó là hệ số của mục này dưới một tên khác.
          </p>
        }
      >
        <p>
          Libraries expose the same choice. When you read a production reduction, look for the number of elements each thread
          handles before the tree: that is this section's factor under another name.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch10-facts A13)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Trong CUB, núm chỉnh là <code>items_per_thread</code>; policy ví dụ trong tài liệu của nó, không phải mặc định, đặt 16 hoặc 20.
            </p>
          }
        >
          <p>
            In CUB the knob is <code>items_per_thread</code>; its docs' example policy, not its default, sets 16 or 20.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Mỗi hàng một khối">One block per row</Translated></h2>
      <Translated
        vi={
          <p>
            Giờ đến trường hợp bạn sẽ gặp trong thực tế. Một softmax trên một hàng từ vựng cần một max và một tổng trên mọi logit
            của hàng. Giao cho mỗi hàng một khối {SOFTMAX_BLOCK} luồng, và một hàng {fmtInt(SOFTMAX_ROW)} logit (từ vựng của Qwen3-8B)
            để mỗi luồng tự cộng khoảng {Math.round(SOFTMAX_ROW / SOFTMAX_BLOCK)} logit. Đó là gộp việc đến giới hạn, và hoàn toàn
            không cần lần ghi nào giữa các khối. Nó cũng có nghĩa là mỗi hàng một khối: ở batch 1, một khối cho cả GPU. Hãy thử
            các thiết lập sẵn softmax.
          </p>
        }
      >
        <p>
          Now the case you will meet in practice. A softmax over a vocabulary row needs a max and a sum over every logit of the
          row. Give each row one block of {SOFTMAX_BLOCK} threads, and a {fmtInt(SOFTMAX_ROW)}-logit row (Qwen3-8B's vocabulary)
          leaves each thread about {Math.round(SOFTMAX_ROW / SOFTMAX_BLOCK)} logits to add on its own. That is coarsening at its
          limit, and it needs no commit across blocks at all. It also means one block per row: at batch 1, one block for the
          whole GPU. Try the softmax presets.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Cách đặt vấn đề softmax và kích thước từ vựng là của chúng tôi (notes/today/ch09-facts.md, D7). Các thiết lập sẵn giữ ví
              dụ của sách từ chương 6, một thiết bị giả định với {BOOK_CAPACITY} khe; trên thiết bị của bạn con số khác.
            </p>
          }
        >
          <p>
            The softmax framing and the vocabulary size are ours (notes/today/ch09-facts.md, D7). The presets keep the book's
            example from chapter 6, a hypothetical device with {BOOK_CAPACITY} slots; on your device the number differs.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Đây là lý do softmax CUDA của llama.cpp có đường thứ hai. Khi một hàng quá dài cho bộ nhớ chia sẻ của một khối, thiết bị
            hỗ trợ cooperative launch, và vài điều kiện khác thỏa (không mask, không scaling), nó trải hàng ra nhiều khối: mỗi khối
            ghi max từng phần của nó vào một mảng, tất cả gặp nhau tại một rào chắn toàn lưới, rồi mọi khối rút gọn các phần đó, và
            lặp lại như vậy cho tổng. Đó là lượt thứ hai của mục 10.9 với ranh giới kernel được thay bằng một rào chắn. Và đó là
            lý do lab 10.3 đo softmax mỗi-hàng-một-khối ở batch 1, 8 và 64.
          </p>
        }
      >
        <p>
          This is why llama.cpp's CUDA softmax has a second path. When a row is too long for one block's shared memory, the device
          supports cooperative launch, and a few other conditions hold (no mask, no scaling), it spreads the row over many blocks: each
          writes its partial max to an array, all meet at a grid-wide barrier, then every block reduces the partials, and the
          same again for the sum. That is section 10.9's second pass with the kernel boundary replaced by a barrier. And it is
          why lab 10.3 measures its one-block-per-row softmax at batch 1, 8 and 64.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch10-facts.md A8, D1)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Rào chắn lưới cần một cooperative launch mà cả lưới thường trú cùng lúc; bản build HIP include một header khác cho nó.
              Metal không có rào chắn lưới.
            </p>
          }
        >
          <p>
            The grid barrier needs a cooperative launch whose whole grid is resident at once; the HIP build includes a different
            header for it. Metal has no grid barrier.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS1010['ch10/10.10/q-batch1']} />

      <h2><Translated vi="Bài tập 4 và 5, và thiết bị của bạn">Exercises 4 and 5, and your device</Translated></h2>
      <Translated
        vi={
          <p>
            Bài tập 5 yêu cầu kernel gộp việc cho mọi độ dài; đó là kernel của lab 10.1, ở trên. Bài tập 4 biến nó thành max: bên
            trong khối chỉ toán tử thay đổi, còn lần ghi là chủ đề của bảng ở mục 10.9. Max theo hàng của lab 10.3 là max bạn thật
            sự sẽ dùng. Cả hai lab đều ở <a href={href('ch10', 'labs')}>trang lab</a>.
          </p>
        }
      >
        <p>
          Exercise 5 asks for the coarsened kernel at any length; that is lab 10.1's kernel, shown above. Exercise 4 turns it
          into a max: inside the block only the operator changes, and the commit is the subject of the table in section 10.9.
          Lab 10.3's row max is the max you will actually use. Both labs are on the <a href={href('ch10', 'labs')}>labs page</a>.
        </p>
      </Translated>
      <Source section="10.12" pages="250" figs={tr('exercises 4, 5', 'bài tập 4, 5')} />
      <Predict
        id="ch10/10.10/p-coarse"
        prompt={
          <Translated
            vi={
              <>
                Lab 10.1 đo thời gian kernel gộp việc của nó trên một mảng lớn, 2<sup>26</sup> float trở lên (chương trình điều khiển
                (driver) nhân đôi n trên thiết bị nhanh, tối đa 2<sup>28</sup>, và chỉ bắt đầu thấp hơn khi thiếu bộ nhớ), với khối 256
                luồng, cho các hệ số 1, 2, 4, và cứ thế đến 128. Hệ số nào sẽ nhanh nhất trên thiết bị <em>của bạn</em>?
              </>
            }
          >
            <>
              Lab 10.1 times its coarsened kernel on one large array, 2<sup>26</sup> floats or more (the driver doubles n on a
              fast device, up to 2<sup>28</sup>, and starts lower only where memory is short), with 256-thread blocks, for the
              factors 1, 2, 4, and so on up to 128. Which factor will be fastest on <em>your</em> device?
            </>
          </Translated>
        }
        unit={tr('(coarsening factor)', '(hệ số gộp việc)')}
      >
        {(v) => <MeasuredCoarsening predicted={v} />}
      </Predict>
    </>
  );
}
