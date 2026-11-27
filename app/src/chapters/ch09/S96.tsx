import { useEffect, useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { solutionTabStatus } from '../../components/LabKit';
import { fmt, useReports } from '../../lib/reports';
import { Translated, tr as trText } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  PRESETS,
  PRESET_ORDER,
  SKY_WIDTH,
  STRIPE,
  WORKED_COARSE,
  blockRun,
  makeImage,
  summary,
  trace,
  warpReference,
  workedImage,
  type LaneState,
  type Partitioning,
  type PresetId,
  type WarpState,
} from '../../sim/histruns';

const LAB = 'ch09/lab2_privatised';
const int = (v: number) => v.toLocaleString('en-US');
const pct = (v: number) => `${fmt(v * 100, 3)}%`;

// Identifiers coloured by the memory they live in, as everywhere in the course.
const HIST_TIERS: Record<string, Tier> = {
  image: 'global', bins: 'global', counts: 'global',
  bins_s: 'shared', binsS: 'shared', bins_s_ref: 'shared', localBins: 'shared',
  segment: 'reg', i: 'reg', c: 'reg', b: 'reg', bNext: 'reg', bin_r: 'reg', binR: 'reg', tid: 'reg',
  bid: 'reg', bdim: 'reg', binValue: 'reg',
};

// The presets are generated once, on first use.
const IMAGES = new Map<PresetId, Uint8Array>();
function image(id: PresetId): Uint8Array {
  let x = IMAGES.get(id);
  if (!x) IMAGES.set(id, (x = makeImage(id)));
  return x;
}

// ---------------------------------------------------------------- lab regions

/** The lines between `// region: <name>` and the next `// endregion`, dedented; null if absent. */
function region(src: string | null, name: string): string | null {
  if (!src) return null;
  const lines = src.split('\n');
  const start = lines.findIndex((l) => l.trim() === `// region: ${name}`);
  if (start < 0) return null;
  const end = lines.findIndex((l, i) => i > start && l.trim() === '// endregion');
  if (end < 0) return null;
  const body = lines.slice(start + 1, end);
  const indent = Math.min(...body.filter((l) => l.trim()).map((l) => /^\s*/.exec(l)![0].length));
  return body.map((l) => l.slice(Number.isFinite(indent) ? indent : 0)).join('\n');
}
const linesWith = (text: string, needles: string[]) =>
  text.split('\n').flatMap((l, i) => (needles.some((n) => l.includes(n)) ? [i + 1] : []));

function Listing() {
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  const msl = useLabFile(LAB + '/solution/kernel.metal');
  const { reports } = useReports();
  // The same words as the labs page's status table and reference-solution tabs.
  const st = (f: 'kernel.cu' | 'kernel.metal') => solutionTabStatus(reports, LAB, f);
  const k = region(cu, 'runs');
  const m = region(msl, 'runs');
  if (!k)
    return (
      <Translated
        vi={
          <p className="muted" data-listing="missing">
            Kernel được nạp từ <code>labs/{LAB}/solution/kernel.cu</code> khi lab 9.2 đã sẵn sàng.
          </p>
        }
      >
        <p className="muted" data-listing="missing">
          The kernel is loaded from <code>labs/{LAB}/solution/kernel.cu</code> once lab 9.2 is in place.
        </p>
      </Translated>
    );
  // The flush inside the loop and the final flush (not the commit, which adds bins_s to bins).
  const atomics = (t: string) => linesWith(t, ['fetch_add(bin_r', 'bins_s[b], bin_r']); // i18n-ignore: code fragments to find in the listing
  return (
    <Code
      tabs={[
        {
          label: trText('CUDA and HIP', 'CUDA và HIP'),
          code: k,
          tiers: HIST_TIERS,
          status: st('kernel.cu'),
          highlight: atomics(k),
        },
        ...(m
          ? [{ label: 'Metal', code: m, tiers: HIST_TIERS, status: st('kernel.metal'), highlight: atomics(m) }]
          : []),
      ]}
      caption={
        <Translated
          vi={
            <>
              <code>histogramRuns</code> từ lời giải tham chiếu của lab 9.2 (<code>labs/{LAB}/solution/</code>), là Hình
              9.15 của sách. Các dòng được tô sáng là hai lần xả (flush): lần trong vòng lặp và lần cuối. Hãy tự viết bản của bạn
              trong lab trước khi nghiên cứu phần này.
            </>
          }
        >
          <>
            <code>histogramRuns</code> from lab 9.2's reference solution (<code>labs/{LAB}/solution/</code>), the book's
            Fig. 9.15. The highlighted lines are the two flushes: the one inside the loop and the final one. Write your own
            in the lab before you study this.
          </>
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- stepper: run merging, one warp
type StepData = 'worked' | 'sky' | 'stripes' | 'noise' | 'tokens';
const STEP_DATA: { id: StepData; label: string }[] = [
  {
    id: 'worked',
    get label() {
      return trText('worked example (lane 0: 7, 7, 7, 3, 3, 7, 7, 9)', 'ví dụ mẫu (lane 0: 7, 7, 7, 3, 3, 7, 7, 9)');
    },
  },
  {
    id: 'sky',
    get label() {
      return trText(`sky (rows ${int(SKY_WIDTH)} wide)`, `bầu trời (hàng rộng ${int(SKY_WIDTH)})`);
    },
  },
  {
    id: 'stripes',
    get label() {
      return trText(`short stripes (runs of ${STRIPE})`, `sọc ngắn (chuỗi dài ${STRIPE})`);
    },
  },
  {
    id: 'noise',
    get label() {
      return trText('noise', 'nhiễu');
    },
  },
  {
    id: 'tokens',
    get label() {
      return trText('token ids (Zipf-like)', 'token id (kiểu Zipf)');
    },
  },
];
const B_STEP = [8, 32, 64, 128, 256, 1024];
const C_STEP = [2, 4, 8, 16, 32, 64];

/** A lane's values so far as runs, "7×3 3×2 7×2": each group cost one atomic. */
function runsText(tr: WarpState[], at: number, lane: number): { text: string; count: number } {
  const runs: [number, number][] = [];
  for (let j = 1; j <= at; j++) {
    const v = tr[j].lanes[lane].value;
    if (v == null) continue;
    const last = runs[runs.length - 1];
    if (last && last[0] === v) last[1]++;
    else runs.push([v, 1]);
  }
  const shown = runs.slice(-6).map(([v, n]) => (n > 1 ? `${v}×${n}` : String(v)));
  return { text: (runs.length > 6 ? '… ' : '') + shown.join('  '), count: runs.length };
}

function BranchChip({ l }: { l: LaneState }) {
  switch (l.branch) {
    case 'first':
      return <span className="chip reg"><Translated vi="nạp điểm ảnh đầu tiên">load first pixel</Translated></span>;
    case 'extend':
      return <span className="chip reg"><Translated vi="nối dài: ++bin_r">extend: ++bin_r</Translated></span>;
    case 'flush':
      return (
        <Translated vi={<span className="chip shared">xả {l.amount} vào bin {l.bin}</span>}>
          <span className="chip shared">flush {l.amount} into bin {l.bin}</span>
        </Translated>
      );
    case 'final':
      return (
        <Translated vi={<span className="chip shared">xả lần cuối {l.amount} vào bin {l.bin}</span>}>
          <span className="chip shared">final flush {l.amount} into bin {l.bin}</span>
        </Translated>
      );
    case 'inactive':
      return <span className="muted"><Translated vi="ngoài phạm vi">out of range</Translated></span>;
    default:
      return <span className="muted">–</span>;
  }
}

function describe(st: WarpState): string {
  const { coarse, blockDim, p, lanes } = st.cfg;
  if (st.step === 0)
    return trText(
      `Nothing has run. These are threads 0 to ${lanes - 1} of the block; each handles COARSE_FACTOR = ${coarse} pixels, ${
        p === 'interleaved' ? `blockDim.x = ${blockDim} apart (interleaved, as in the book)` : 'next to each other (contiguous, for comparison)'
      }. Press Step.`,
      `Chưa có gì chạy. Đây là các luồng 0 đến ${lanes - 1} của khối; mỗi luồng xử lý COARSE_FACTOR = ${coarse} điểm ảnh, ${
        p === 'interleaved' ? `cách nhau blockDim.x = ${blockDim} (xen kẽ, như trong sách)` : 'nằm kề nhau (liền kề, để so sánh)'
      }. Bấm Bước.`,
    );
  if (st.step === 1)
    return trText(
      'Lines 11 to 13: each lane checks that its first pixel is in range, loads it into b and sets bin_r = 1. No atomics yet.',
      'Dòng 11 đến 13: mỗi lane kiểm tra điểm ảnh đầu tiên của nó có nằm trong phạm vi không, nạp nó vào b và đặt bin_r = 1. Chưa có phép toán nguyên tử nào.',
    );
  const ext = st.lanes.filter((l) => l.branch === 'extend').map((l) => l.lane);
  const fl = st.lanes.filter((l) => l.branch === 'flush');
  const list = (xs: number[]) =>
    xs.length > 6
      ? trText(`${xs.length} lanes`, `${xs.length} lane`)
      : trText(`lane${xs.length === 1 ? '' : 's'} ${xs.join(', ')}`, `lane ${xs.join(', ')}`);
  if (st.c == null)
    return trText(
      `Lines 29 to 31, after the loop: every lane flushes the run still in its registers, ${st.lanes.filter((l) => l.branch === 'final').length} atomics. This flush is unconditional: the last run has not met a different value, so nothing has flushed it yet.`,
      `Dòng 29 đến 31, sau vòng lặp: mọi lane xả chuỗi còn trong thanh ghi của nó, ${st.lanes.filter((l) => l.branch === 'final').length} phép toán nguyên tử. Lần xả này là vô điều kiện: chuỗi cuối chưa gặp giá trị khác, nên chưa có gì xả nó.`,
    );
  const head = trText(`Iteration c = ${st.c} (lines 15 to 26): `, `Lần lặp c = ${st.c} (dòng 15 đến 26): `);
  if (!fl.length)
    return head + trText(
      'every active lane reads its run’s value again and extends it. Registers only, no atomics.',
      'mọi lane đang hoạt động lại đọc đúng giá trị của chuỗi và nối dài nó. Chỉ dùng thanh ghi, không có phép toán nguyên tử.',
    );
  if (!ext.length)
    return head + trText(
      `every active lane meets a new value and flushes: ${fl.length} atomics, and the warp stays together.`,
      `mọi lane đang hoạt động gặp một giá trị mới và xả: ${fl.length} phép toán nguyên tử, và warp vẫn đi cùng nhau.`,
    );
  return (
    head +
    trText(
      `${list(ext)} extend${ext.length === 1 ? 's' : ''}; ${list(fl.map((l) => l.lane))} flush${fl.length === 1 ? 'es' : ''} (${fl.length} atomic${fl.length === 1 ? '' : 's'}). The warp splits at the new if: the two sides run one after the other.`,
      `${list(ext)} nối dài; ${list(fl.map((l) => l.lane))} xả (${fl.length} phép toán nguyên tử). Warp tách ra ở lệnh if mới: hai nhánh chạy lần lượt nhánh này sau nhánh kia.`,
    )
  );
}

function RunStepper() {
  const [data, setData] = useState<StepData>('worked');
  const [p, setP] = useState<Partitioning>('interleaved');
  const [B, setB] = useState(256);
  const [C, setC] = useState(8);
  const [lanes, setLanes] = useState(8);
  const [block, setBlock] = useState(0);
  const [k, setK] = useState(0);

  const coarse = data === 'worked' ? WORKED_COARSE : C;
  const img = useMemo(() => (data === 'worked' ? workedImage(p, B) : image(data)), [data, p, B]);
  const blocks = Math.ceil(img.length / (B * coarse));
  const blk = Math.min(block, blocks - 1);
  const L = Math.min(lanes, B);
  const cfg = useMemo(() => ({ p, n: img.length, blockDim: B, coarse, block: blk, firstThread: 0, lanes: L }), [p, img, B, coarse, blk, L]);
  const tr = useMemo(() => trace(img, cfg), [img, cfg]);
  const last = tr.length - 1;
  const at = Math.min(k, last);
  const st = tr[at];
  const loopSoFar = tr.slice(2, at + 1).filter((s) => s.c != null && s.lanes.some((l) => l.branch === 'extend' || l.branch === 'flush')).length;
  useEffect(() => setK(0), [data, p, B, C, lanes, block]);

  const ref = useMemo(() => warpReference(img, cfg), [img, cfg]);
  const whole = useMemo(() => blockRun(img, p, B, coarse, blk, L), [img, p, B, coarse, blk, L]);
  const matches = st.counts.every((v, i) => v === ref[i]);

  return (
    <Bench
      title={trText('Run merging, one warp', 'Gộp chuỗi, một warp')}
      task={
        <Translated vi="Chạy từng bước ví dụ mẫu và đếm số phép toán nguyên tử của lane 0. Rồi thử bầu trời, và sọc ngắn ở cả hai thứ tự.">
          Step the worked example and count lane 0's atomics. Then try sky, and short stripes in both orders.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              <span className="t-reg">Nối dài</span> chỉ chạm vào thanh ghi; <span className="t-shared">xả</span> là một
              phép toán nguyên tử phạm vi khối trên bản sao riêng của khối trong bộ nhớ chia sẻ. Dữ liệu là của khóa học này, sinh từ một
              seed cố định: {data === 'worked' ? 'ví dụ mẫu giao cho các lane 0 đến 7 tám dãy bạn thấy ở cột cuối' : PRESETS[data].note}.
            </>
          }
        >
          <>
            <span className="t-reg">Extend</span> touches only registers; <span className="t-shared">flush</span> is one
            block-scope atomic on the block's private copy in shared memory. The data are ours, generated from a fixed
            seed: {data === 'worked' ? 'the worked example hands lanes 0 to 7 the eight sequences you see in the last column' : PRESETS[data].note}.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi="Dữ liệu">Data</Translated>
          <select value={data} aria-label={trText('data', 'dữ liệu')} onChange={(e) => setData(e.target.value as StepData)}>
            {STEP_DATA.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}
          </select>
        </label>
        <span className="seg" role="group" aria-label={trText('Partitioning', 'Cách chia')}>
          {(['interleaved', 'contiguous'] as Partitioning[]).map((x) => (
            <button key={x} aria-pressed={p === x} onClick={() => setP(x)}>
              {x === 'interleaved' ? trText('Interleaved (Fig. 9.15)', 'Xen kẽ (Hình 9.15)') : trText('Contiguous (for comparison)', 'Liền kề (để so sánh)')}
            </button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field">
          blockDim.x
          <select value={B} aria-label={trText('block size', 'kích thước khối')} onChange={(e) => { const v = Number(e.target.value); setB(v); setLanes((l) => Math.min(l, Math.max(8, v))); }}>
            {B_STEP.map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
        <label className="field">
          COARSE_FACTOR
          <select value={coarse} aria-label={trText('coarse factor', 'hệ số gộp việc')} disabled={data === 'worked'} onChange={(e) => setC(Number(e.target.value))}>
            {(data === 'worked' ? [WORKED_COARSE] : C_STEP).map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Số lane trong warp">Lanes in the warp</Translated>
          <select value={lanes} aria-label={trText('lanes', 'số lane')} onChange={(e) => setLanes(Number(e.target.value))}>
            <option value={8}>{trText('8 (easier to read; no vendor’s width)', '8 lane (dễ đọc hơn; không phải độ rộng của nhà cung cấp nào)')}</option>
            <option value={32} disabled={B < 32}>{trText('32 (NVIDIA; AMD RDNA)', '32 lane (NVIDIA; AMD RDNA)')}</option>
            <option value={64} disabled={B < 64}>{trText('64 (AMD CDNA)', '64 lane (AMD CDNA)')}</option>
          </select>
        </label>
        <label className="field">
          <Translated vi="Khối">Block</Translated>
          <select value={blk} aria-label={trText('block', 'khối')} onChange={(e) => setBlock(Number(e.target.value))}>
            {Array.from({ length: Math.min(blocks, 64) }, (_, b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </label>
        <Translated vi={<span className="muted">trên {int(blocks)}</span>}>
          <span className="muted">of {int(blocks)}</span>
        </Translated>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setK(0)} disabled={at === 0}><Translated vi="Làm lại">Restart</Translated></button>
        <button onClick={() => setK(at - 1)} disabled={at === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => setK(at + 1)} disabled={at >= last}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => setK(last)} disabled={at >= last}><Translated vi="Chạy đến cuối">Run to the end</Translated></button>
        <Translated vi={<span className="muted" aria-live="polite">bước {at} trên {last}</span>}>
          <span className="muted" aria-live="polite">step {at} of {last}</span>
        </Translated>
      </div>

      <p className="narrate" role="status" style={{ margin: '0 0 0.8rem' }}>{describe(st)}</p>

      <div className="row" style={{ gap: '3px', marginBottom: '0.8rem' }} aria-label={trText('Where the warp split', 'Nơi warp bị tách')} data-splits={tr.slice(1, at + 1).filter((s) => s.divergent).length}>
        <span className="muted" style={{ fontSize: '0.8rem', marginRight: '0.4rem' }}><Translated vi="Các bước (tô đậm: warp bị tách):">Steps (filled: the warp split):</Translated></span>
        {tr.slice(1).map((s, j) => (
          <span key={j} title={s.c == null ? (j === 0 ? trText('first loads', 'các lần nạp đầu') : trText('final flush', 'lần xả cuối')) : `c = ${s.c}`}
            style={{
              width: 10, height: 14, borderRadius: 2, display: 'inline-block',
              border: '1px solid ' + (j + 1 === at ? 'var(--ink)' : 'var(--rule)'),
              background: j + 1 > at ? 'transparent' : s.divergent ? 'var(--ink-2)' : 'var(--cell)',
            }} />
        ))}
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table className="threads" aria-label={trText('Lanes of the warp', 'Các lane của warp')}>
          <thead>
            <tr>
              <th>lane</th>{/* i18n-ignore: glossary keeps lane */}
              <th><Translated vi="phần tử i">element i</Translated></th>
              <th><Translated vi="giá trị">value</Translated></th>
              <th><Translated vi="bước này">this step</Translated></th>
              <th className="t-reg">b</th>
              <th className="t-reg">bin_r</th>{/* i18n-ignore: register name */}
              <th><Translated vi="nguyên tử">atomics</Translated></th>
              <th><Translated vi="các giá trị đến giờ, theo chuỗi">values so far, as runs</Translated></th>
            </tr>
          </thead>
          <tbody>
            {st.lanes.map((l) => {
              const r = runsText(tr, at, l.lane);
              return (
                <tr key={l.lane} data-lane={l.lane} data-atomics={l.atomics} data-branch={l.branch}>
                  <td>{l.lane}</td>
                  <td>{l.element == null ? '' : int(l.element)}</td>
                  <td>{l.value ?? ''}</td>
                  <td><BranchChip l={l} /></td>
                  <td className="t-reg">{l.b ?? ''}</td>
                  <td className="t-reg">{l.b == null ? '' : l.binR}</td>
                  <td>{l.atomics}</td>
                  <td style={{ whiteSpace: 'nowrap' }}>{r.text}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="row" style={{ marginTop: '0.8rem' }} data-totals data-atomics={st.atomics} data-elements={st.elements} data-divergent={st.divergentSteps}>
        <Translated vi={<span>Phép toán nguyên tử đến giờ: <b>{st.atomics}</b></span>}>
          <span>Atomics so far: <b>{st.atomics}</b></span>
        </Translated>
        <Translated vi={<span>Hình 9.14 lẽ ra đã phát <b>{st.elements}</b>, một cho mỗi điểm ảnh đã đọc</span>}>
          <span>Fig. 9.14 would have issued <b>{st.elements}</b>, one per pixel read</span>
        </Translated>
        <Translated vi={<span>Warp đã bị tách ở <b>{st.divergentSteps}</b> trên {loopSoFar} lần lặp vòng lặp đến giờ</span>}>
          <span>The warp split in <b>{st.divergentSteps}</b> of {loopSoFar} loop iteration{loopSoFar === 1 ? '' : 's'} so far</span>
        </Translated>
      </div>
      {st.done ? (
        <Translated
          vi={
            <p style={{ margin: '0.8rem 0 0' }} data-check={matches ? 'ok' : 'fail'}>
              {matches ? (
                <span className="chip ok" style={{ whiteSpace: 'normal' }}>Các lần xả của những lane này cộng lại bằng một phép đếm thường những gì chúng đã đọc, từng bin một.</span>
              ) : (
                <span className="chip fail" style={{ whiteSpace: 'normal' }}>Các lần xả của những lane này không khớp với một phép đếm thường.</span>
              )}{' '}
              Cả khối ({B} luồng): {int(whole.atomics)} phép toán nguyên tử cho {int(whole.elements)} điểm ảnh
              {whole.match ? ', và bản sao riêng của nó bằng một phép đếm thường.' : ', và bản sao riêng của nó sai.'}
            </p>
          }
        >
          <p style={{ margin: '0.8rem 0 0' }} data-check={matches ? 'ok' : 'fail'}>
            {matches ? (
              <span className="chip ok" style={{ whiteSpace: 'normal' }}>These lanes' flushes add up to a plain count of what they read, bin by bin.</span>
            ) : (
              <span className="chip fail" style={{ whiteSpace: 'normal' }}>These lanes' flushes do not match a plain count.</span>
            )}{' '}
            The whole block ({B} threads): {int(whole.atomics)} atomics for {int(whole.elements)} pixels
            {whole.match ? ', and its private copy equals a plain count.' : ', and its private copy is wrong.'}
          </p>
        </Translated>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: atomics per element
const B_TABLE = [32, 64, 128, 256, 512, 1024];
const C_TABLE = [1, 2, 4, 8, 16, 32, 64];

function AtomicsTable() {
  const [B, setB] = useState(256);
  const [C, setC] = useState(16);
  const [W, setW] = useState(32);
  const rows = useMemo(
    () => PRESET_ORDER.map((id) => ({ id, inter: summary(image(id), 'interleaved', B, C, W), contig: summary(image(id), 'contiguous', B, C, W) })),
    [B, C, W],
  );
  return (
    <Bench
      title={trText('Atomics per element', 'Phép toán nguyên tử trên mỗi phần tử')}
      task={
        <Translated vi="Hình 9.15 giúp những dòng nào? Thay đổi blockDim.x và hệ số rồi xem đáp án nào thay đổi.">
          Which rows does Fig. 9.15 help? Change blockDim.x and the factor and see which answers move.
        </Translated>
      }
      foot={trText(
        `Each preset is ${int(image('noise').length)} bytes, 256 bins. Fig. 9.14 issues one block-scope atomic per pixel; the other columns are Fig. 9.15's atomics divided by pixels. Counts, not time.`,
        `Mỗi preset gồm ${int(image('noise').length)} byte, 256 bin. Hình 9.14 phát một phép toán nguyên tử phạm vi khối cho mỗi điểm ảnh; các cột khác là số phép toán nguyên tử của Hình 9.15 chia cho số điểm ảnh. Số lượng, không phải thời gian.`,
      )}
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          blockDim.x
          <select value={B} aria-label={trText('block size', 'kích thước khối')} onChange={(e) => setB(Number(e.target.value))}>
            {B_TABLE.map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
        <label className="field">
          COARSE_FACTOR
          <select value={C} aria-label={trText('coarse factor', 'hệ số gộp việc')} onChange={(e) => setC(Number(e.target.value))}>
            {C_TABLE.map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <select value={W} aria-label={trText('warp width', 'độ rộng warp')} onChange={(e) => setW(Number(e.target.value))}>
            <option value={32}>{trText('32 lanes (NVIDIA; AMD RDNA)', '32 lane (NVIDIA; AMD RDNA)')}</option>
            <option value={64}>{trText('64 lanes (AMD CDNA)', '64 lane (AMD CDNA)')}</option>
          </select>
        </label>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-atomics-table>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}><Translated vi="Dữ liệu">Data</Translated></th>
              <th className="num"><Translated vi="Hình 9.14">Fig. 9.14</Translated></th>
              <th className="num"><Translated vi="Hình 9.15, như bản in (xen kẽ)">Fig. 9.15, as printed (interleaved)</Translated></th>
              <th className="num"><Translated vi="Số lần lặp mà một warp bị tách">Iterations in which a warp split</Translated></th>
              <th className="num"><Translated vi="Hình 9.15 với các phần tử liền kề (của khóa học này)">Fig. 9.15 with contiguous elements (ours)</Translated></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} data-preset={r.id} data-inter={r.inter.perElement} data-contig={r.contig.perElement} data-split={r.inter.divergentShare ?? ''}>
                <td style={{ paddingLeft: '0.6rem' }}>{PRESETS[r.id].label}</td>
                <td className="num">1</td>
                <td className="num"><b>{fmt(r.inter.perElement)}</b></td>
                <td className="num">{r.inter.divergentShare == null ? trText('no loop', 'không có vòng lặp') : pct(r.inter.divergentShare)}</td>
                <td className="num">{fmt(r.contig.perElement)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

function RunsFraction({ predicted }: { predicted: number }) {
  const s = summary(image('sky'), 'interleaved', 256, 16, 32);
  const v = s.perElement * 100;
  return (
    <>
      <Translated
        vi={
          <p data-fraction={v}>
            Mô hình cho ra <b>{fmt(v)}%</b>.{' '}
            {Math.abs(predicted - v) <= Math.max(1, v * 0.2) ? 'Đó đúng là điều bạn dự đoán.' : predicted > v ? 'Bạn đã kỳ vọng Hình 9.15 tiết kiệm ít hơn.' : 'Bạn đã kỳ vọng nó tiết kiệm nhiều hơn.'}{' '}
            Với <code>blockDim.x = 256</code> và hệ số 16, một khối đọc 4,096 điểm ảnh liên tiếp, đúng một hàng của
            bầu trời này, nên mỗi luồng thấy 16 giá trị bằng nhau và xả một lần.
          </p>
        }
      >
        <p data-fraction={v}>
          The model says <b>{fmt(v)}%</b>.{' '}
          {Math.abs(predicted - v) <= Math.max(1, v * 0.2) ? 'That is what you predicted.' : predicted > v ? 'You expected Fig. 9.15 to save less.' : 'You expected it to save more.'}{' '}
          With <code>blockDim.x = 256</code> and a factor of 16 a block reads 4,096 consecutive pixels, exactly one row of
          this sky, so every thread sees 16 equal values and flushes once.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Ảnh thật hiếm khi gọn gàng như vậy, và tiết kiệm một phép toán nguyên tử không giống với tiết kiệm thời gian. Lab 9.2 đo{' '}
            <code>histogramRuns</code> của bạn so với <code>histogramShared</code> trên ảnh bầu trời riêng của nó và trên các byte phân bố đều: xem{' '}
            <a href={href('ch09', 'labs')}>trang lab</a>.
          </p>
        }
      >
        <p>
          Real images are rarely that tidy, and an atomic saved is not the same as time saved. Lab 9.2 times your{' '}
          <code>histogramRuns</code> against <code>histogramShared</code> on its own sky image and on uniform bytes: see
          the <a href={href('ch09', 'labs')}>labs page</a>.
        </p>
      </Translated>
    </>
  );
}

// ---------------------------------------------------------------- questions
const NOISE32 = () => summary(image('noise'), 'interleaved', 256, 16, 32).divergentShare ?? 0;

export const questionsS96: Record<string, Question> = {
  'ch09/9.6/q-runs-count': {
    kind: 'numeric',
    id: 'ch09/9.6/q-runs-count',
    prompt: (
      <Translated
        vi={
          <>
            Tám điểm ảnh của một luồng, theo thứ tự nó đọc, là 7, 7, 7, 3, 3, 7, 7, 9. Hình 9.15 phát bao nhiêu phép toán
            nguyên tử cho chúng?
          </>
        }
      >
        <>
          A thread's eight pixels, in the order it reads them, are 7, 7, 7, 3, 3, 7, 7, 9. How many atomic operations
          does Fig. 9.15 issue for them?
        </>
      </Translated>
    ),
    answer: 4,
    explain: (
      <Translated
        vi={
          <p>
            Bốn chuỗi: 7 ba lần, 3 hai lần, 7 hai lần, 9 một lần. Vòng lặp xả ba chuỗi đầu khi giá trị tiếp theo
            khác đi (lượng 3, 2 và 2), và lần xả cuối sau vòng lặp cộng chuỗi cuối cùng. Bốn phép toán nguyên tử thay vì
            tám. Chuỗi 7 thứ hai là một chuỗi riêng: luồng chỉ nhớ bin gần nhất.
          </p>
        }
      >
        <p>
          Four runs: 7 three times, 3 twice, 7 twice, 9 once. The loop flushes the first three when the next value
          differs (amounts 3, 2 and 2), and the final flush after the loop adds the last one. Four atomics instead of
          eight. The second run of 7 is a separate run: the thread only remembers the most recent bin.
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.6, pp. 218–219 (Fig. 9.15)',
    vi: { source: 'ch. 9 §9.6, pp. 218–219 (Hình 9.15)' },
  },
  'ch09/9.6/q-final-flush': {
    kind: 'choice',
    id: 'ch09/9.6/q-final-flush',
    prompt: (
      <Translated vi={<>Vì sao lần xả sau vòng lặp trong Hình 9.15 là vô điều kiện?</>}>
        <>Why is the flush after the loop in Fig. 9.15 unconditional?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Nó chỉ cần khi điểm ảnh cuối khác điểm ảnh trước đó; vô điều kiện chỉ là để mã đơn giản hơn.">
            It is needed only when the last pixel differs from the one before; unconditional is just simpler code.
          </Translated>
        ),
        why: (
          <Translated vi="Nếu chúng bằng nhau, chuỗi chứa chúng vẫn chưa được xả. Lần nào cũng cần nó.">
            If they are equal, the run they belong to is still unflushed. It is needed every time.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một chuỗi chỉ được xả khi một giá trị khác đến, nên chuỗi cuối luôn còn nằm trong thanh ghi.">
            A run is flushed only when a different value arrives, so the last run is always still in the registers.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Nó phòng một race với các luồng khác của khối vẫn đang đếm vào cùng bin.">
            It guards against a race with other threads of the block that are still counting into the same bin.
          </Translated>
        ),
        why: (
          <Translated vi="Phép toán nguyên tử làm việc đó. Lần xả cuối tồn tại vì độ trễ một bước, không phải vì các luồng khác.">
            The atomic does that. The final flush exists because of the lag, not because of other threads.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó lo cho các luồng có điểm ảnh vượt quá cuối ảnh và vì thế rời vòng lặp sớm.">
            It covers threads whose pixels run past the end of the image and so leave the loop early.
          </Translated>
        ),
        why: (
          <Translated vi="Mọi luồng đều kết thúc vòng lặp với một chuỗi chưa xả, bất kể điểm ảnh của nó ở đâu.">
            Every thread ends the loop with an unflushed run, wherever its pixels are.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lần xả luôn trễ một phần tử: chỉ biết một chuỗi đã kết thúc khi một giá trị khác đến.
            Bỏ lần xả cuối đi thì mỗi luồng mất một chuỗi: trong ví dụ mẫu, tám số 7 của lane 1 sẽ không bao giờ tới được
            histogram.
          </p>
        }
      >
        <p>
          The flush always lags one element behind: a run is known to have ended only when a different value arrives.
          Drop the final flush and each thread loses one run: in the worked example, lane 1's eight 7s would never reach
          the histogram at all.
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.6, pp. 218–219',
  },
  'ch09/9.6/q-spacing': {
    kind: 'choice',
    id: 'ch09/9.6/q-spacing',
    prompt: (
      <Translated
        vi={
          <>
            <code>blockDim.x = 256</code>, chia xen kẽ như trong Hình 9.15. Đầu vào nào được lợi nhiều nhất từ việc gộp
            chuỗi?
          </>
        }
      >
        <>
          <code>blockDim.x = 256</code>, interleaved partitioning as in Fig. 9.15. Which input gains most from run
          merging?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Một ảnh chụp rộng 4096 điểm ảnh có bầu trời phủ trọn các hàng.">A 4096-pixel-wide photo whose sky covers whole rows.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Các sọc gồm 8 điểm ảnh giống nhau xen kẽ với nhiễu.">Stripes of 8 identical pixels alternating with noise.</Translated>,
        why: (
          <Translated vi="Các điểm ảnh liên tiếp của một luồng cách nhau 256, nên không luồng nào đọc được hai điểm ảnh của cùng một sọc.">
            A thread’s consecutive pixels are 256 apart, so no thread ever reads two pixels of the same stripe.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Nhiễu phân bố đều.">Uniform noise.</Translated>,
        why: (
          <Translated vi="Các giá trị kế nhau bằng nhau khoảng một lần trong 256 lần đọc, bất kể khoảng cách.">
            Neighbouring values are equal about once in 256 reads, whatever the spacing.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Khi chia xen kẽ, các điểm ảnh liên tiếp của một luồng cách nhau <code>blockDim.x</code>, nên một chuỗi chỉ được gộp
            ở nơi ảnh không đổi trên nhiều trăm byte liên tiếp, như các hàng bầu trời trọn vẹn. Chuỗi ngắn thì
            vô hình với mọi luồng. Chia liền kề sẽ thấy chúng, với cái giá là mất hợp nhất truy cập.
          </p>
        }
      >
        <p>
          Under interleaved partitioning a thread's consecutive pixels are <code>blockDim.x</code> apart, so a run merges
          only where the image is constant over many hundreds of consecutive bytes, such as whole rows of sky. Short runs
          are invisible to every thread. Contiguous partitioning would see them, at the price of coalescing.
        </p>
      </Translated>
    ),
    source: 'our analysis of ch. 9 §9.6, Fig. 9.15, checked with the model above',
    vi: { source: 'phân tích của khóa học này về ch. 9 §9.6, Hình 9.15, được kiểm tra bằng mô hình ở trên' },
  },
  'ch09/9.6/q-tokens': {
    kind: 'choice',
    id: 'ch09/9.6/q-tokens',
    prompt: (
      <Translated vi={<>Hình 9.15 có giảm đáng kể số phép toán nguyên tử khi đếm token id của các prompt tiếng Anh không?</>}>
        <>Would Fig. 9.15 cut the atomics much when counting the token ids of English prompts?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Có: các token phổ biến như “the” chiếm một phần lớn của bất kỳ prompt tiếng Anh nào, nên chúng tạo thành các chuỗi dài.">
            Yes: common tokens such as “the” make up a large share of any English prompt, so they form long runs.
          </Translated>
        ),
        why: (
          <Translated vi="Token phổ biến thì xuất hiện thường xuyên, chứ không nằm kề nhau. Chuỗi cần các phần tử kế nhau bằng nhau.">
            Common tokens are frequent, not adjacent. Runs need equal neighbours.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Có, miễn là chuyển cách chia sang liền kề, để một luồng thấy các token kế nhau.">
            Yes, as long as the partitioning is switched to contiguous, so that a thread sees neighbouring tokens.
          </Translated>
        ),
        why: (
          <Translated vi="Khi đó một luồng thấy các token kế nhau, nhưng chúng hiếm khi bằng nhau, và các lần nạp không còn được hợp nhất.">
            Then a thread sees neighbouring tokens, but they are rarely equal, and the loads stop coalescing.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Hầu như không: các token bằng nhau hiếm khi nằm kề nhau, và dù sao một luồng cũng thấy các token cách nhau một độ rộng khối.">
            Hardly: equal tokens are rarely neighbours, and a thread sees tokens a block width apart anyway.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Tần suất không phải là sự kề nhau. Trong preset token id, chưa tới một trên hai mươi cặp kề nhau là bằng nhau, và
            Hình 9.15 của mô hình vẫn phát hơn 0.9 phép toán nguyên tử mỗi token. Thứ giúp được với token là một cấu trúc khác:
            một hàng số đếm cho mỗi request, cập nhật khi token được sinh ra, đó là lab 9.3.
          </p>
        }
      >
        <p>
          Frequency is not adjacency. In the token-id preset fewer than one neighbouring pair in twenty is equal, and the
          model's Fig. 9.15 still issues more than 0.9 atomics per token. What helps with tokens is a different structure:
          one row of counts per request, updated as tokens are generated, which is lab 9.3.
        </p>
      </Translated>
    ),
    source: 'our analysis; the preset is a Zipf-like stand-in, not a real tokenizer',
    vi: { source: 'phân tích của khóa học này; preset là một bản thay thế kiểu Zipf, không phải một tokenizer thật' },
  },
  'ch09/9.6/q-diverge': {
    kind: 'choice',
    id: 'ch09/9.6/q-diverge',
    prompt: (
      <Translated vi={<>Với đầu vào nào, nhánh mới của Hình 9.15 gây ra nhiều phân kỳ nhất?</>}>
        <>For which input does Fig. 9.15's new branch cost the most divergence?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Nhiễu phân bố đều, nơi điểm ảnh tiếp theo của một luồng gần như không bao giờ bằng điểm ảnh trước.">
            Uniform noise, where a thread's next pixel almost never equals its last one.
          </Translated>
        ),
        why: (
          <Translated vi="Gần như mọi lane đều xả ở gần như mọi lần lặp, nên các lane phần lớn đồng thuận.">
            Almost every lane flushes in almost every iteration, so the lanes mostly agree.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một ảnh hằng, nơi mọi điểm ảnh một luồng đọc đều bằng điểm ảnh trước.">
            A constant image, where every pixel a thread reads equals its last one.
          </Translated>
        ),
        why: (
          <Translated vi="Mọi lane đều nối dài ở mọi lần lặp: hoàn toàn không bị tách.">
            Every lane extends in every iteration: no split at all.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Dữ liệu mà trong một lần lặp, một số lane nối dài chuỗi còn các lane khác thì xả.">
            Data on which, in one iteration, some lanes extend a run and others flush.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Phân kỳ cần sự bất đồng trong một warp ở một lần lặp. Không có sự tương đồng thì gần như mọi lane đều xả,
            tương đồng nhiều thì gần như mọi lane đều nối dài, và sách kỳ vọng việc giảm tranh chấp sẽ bù cho các lần tách ở
            giữa. "Phần lớn" không phải là "không bao giờ": trong mô hình, một warp 32 lane trên nhiễu vẫn bị tách ở khoảng{' '}
            {pct(NOISE32())} số lần lặp, mỗi khi một lane tình cờ gặp lại giá trị của chính nó. Preset token id,
            nơi vài id phổ biến khiến một số lane nối dài trong khi số còn lại xả, bị tách thường xuyên hơn nhiều.
          </p>
        }
      >
        <p>
          Divergence needs disagreement inside one warp in one iteration. With no similarity nearly all lanes flush,
          with heavy similarity nearly all extend, and the book expects the reduced contention to pay for the splits in
          between. "Mostly" is not "never": in the model, a 32-lane warp on noise still splits in about{' '}
          {pct(NOISE32())} of its iterations, whenever one lane happens to meet its own value again. The token-id preset,
          where a few common ids make some lanes extend while the rest flush, splits far more often.
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.6, p. 219; the noise figure is from the model above',
    vi: { source: 'ch. 9 §9.6, p. 219; con số về nhiễu lấy từ mô hình ở trên' },
  },
  'ch09/9.6/q-few-bins': {
    kind: 'choice',
    id: 'ch09/9.6/q-few-bins',
    prompt: (
      <Translated
        vi={
          <>
            Giờ mỗi luồng xử lý nhiều điểm ảnh. 2<sup>28</sup> byte đi vào 16 bin. Điều đó khiến cái gì trở nên khả thi
            ở đây mà 256 bin thì không?
          </>
        }
      >
        <>
          Each thread now handles several pixels. 2<sup>28</sup> bytes go into 16 bins. What does that make affordable
          here that 256 bins would not?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Bỏ bước ghi vào bộ nhớ toàn cục, vì 16 số đếm từng phần mỗi khối là quá ít để đáng kể.">
            Skipping the commit to global memory, since 16 partial counts per block are too few to matter.
          </Translated>
        ),
        why: (
          <Translated vi="Các số đếm từng phần vẫn phải tới được histogram công khai.">
            The partial counts still have to reach the public histogram.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một histogram riêng trọn vẹn trong mỗi luồng: 16 bộ đếm, cập nhật không cần phép toán nguyên tử, ghi một lần.">
            A whole private histogram in each thread: 16 counters, updated without atomics, committed once.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Phép tăng thường trên các bin công khai, vì chỉ với 16 bin các luồng hiếm khi va chạm.">
            Plain increments on the public bins, since with only 16 bins the threads rarely collide.
          </Translated>
        ),
        why: (
          <Translated vi="Với 16 bin chúng va chạm liên tục; đó là một race bất kể số bin.">
            With 16 bins they collide constantly; that is a race whatever the bin count.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Phép toán nguyên tử phạm vi device trên bản sao riêng của khối, vì 16 bin chừa chỗ cho phạm vi chậm hơn.">
            Device-scope atomics on the block's private copy, since 16 bins leave room for the slower scope.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là một phạm vi chậm hơn cho cùng công việc, không phải điều mà ít bin cho phép.">
            That is a slower scope for the same work, not something few bins enable.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Riêng tư hóa ở mức luồng bị giới hạn bởi số bin. Mười sáu bộ đếm mỗi luồng là rẻ, trong thanh ghi hoặc trong
            bộ nhớ chia sẻ, và mỗi luồng ghi chúng vào bản sao của khối một lần; 256 bộ đếm mỗi luồng,
            nhân với mọi luồng, sách gọi là đắt đến mức không chấp nhận được. Đó là lý do Hình 9.15 chỉ giữ một bin mỗi luồng
            thay vào đó.
          </p>
        }
      >
        <p>
          Thread-level privatisation is limited by the bin count. Sixteen counters per thread are cheap, in registers or in
          shared memory, and each thread commits them to the block's copy once; 256 per thread,
          times every thread, the book calls prohibitively expensive. That is why Fig. 9.15 keeps a single bin per thread
          instead.
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.6, p. 217',
  },
};

// ---------------------------------------------------------------- the page
export function S96() {
  const tokens = useMemo(() => summary(image('tokens'), 'interleaved', 256, 16, 32), []);
  return (
    <>
      <Translated
        vi={
          <p>
            Gộp việc đã giao cho mỗi luồng nhiều điểm ảnh. Điều đó mở ra một cánh cửa mà các kernel một-điểm-ảnh không có. Trước khi đọc
            tiếp, hãy quyết định đó là gì.
          </p>
        }
      >
        <p>
          Coarsening gave each thread several pixels. That opens a door the one-pixel kernels did not have. Before reading
          on, decide what it is.
        </p>
      </Translated>
      <Ask q={questionsS96['ch09/9.6/q-few-bins']} label={trText('Predict', 'Dự đoán')} />
      <Translated
        vi={
          <p>
            Một luồng có thể đếm riêng, hoàn toàn không cần phép toán nguyên tử, rồi cộng số đếm của nó vào bản sao của khối sau. Điều đó đi
            xa tới đâu phụ thuộc vào số bin. Với một nhúm bin, một luồng có thể giữ cả một histogram riêng, trong
            thanh ghi hoặc trong bộ nhớ chia sẻ, và ghi nó một lần. Với nhiều bin, sách gọi việc đó là đắt đến mức không chấp nhận được:
            với một ảnh 8-bit, hãy hình dung 256 bộ đếm do mỗi luồng giữ.
          </p>
        }
      >
        <p>
          A thread can count privately, with no atomics at all, and add its counts to the block's copy later. How far that
          goes depends on the number of bins. With a handful of bins a thread can keep a whole private histogram, in
          registers or in shared memory, and commit it once. With many bins the book calls that prohibitively expensive:
          for an 8-bit image, picture 256 counters held by every thread.
        </p>
      </Translated>
      <Source section="9.6" pages="217" />

      <h2><Translated vi="Chỉ riêng tư hóa bin bạn đang dùng">Privatise only the bin you are using</Translated></h2>
      <Translated
        vi={
          <p>
            Lối thoát của sách là riêng tư hóa một bin mỗi luồng: bin mà nó dùng gần nhất. Ảnh thật có các chuỗi. Một mảng
            trời là hàng nghìn điểm ảnh giống hệt nhau, và các chuỗi đó vừa là trường hợp tệ nhất về tranh chấp, vì mọi luồng
            đều nện vào cùng một bin, vừa là cơ hội tốt nhất, vì các cập nhật liên tiếp vào một bin có thể được gộp thành một
            phép toán nguyên tử duy nhất.
          </p>
        }
      >
        <p>
          The book's way out is to privatise one bin per thread: the one it used last. Real images have runs. A patch of
          sky is thousands of identical pixels, and those runs are both the worst case for contention, since every thread
          hammers the same bin, and the best opportunity, since consecutive updates to one bin can be merged into a
          single atomic.
        </p>
      </Translated>
      <Source pages="218" />

      <h2><Translated vi="Hình 9.15 làm điều đó thế nào">How Fig. 9.15 does it</Translated></h2>
      <Translated
        vi={
          <p>
            Hai thanh ghi làm việc này: <code>b</code> giữ giá trị của chuỗi hiện tại và <code>bin_r</code> giữ độ dài của nó.
          </p>
        }
      >
        <p>
          Two registers do the work: <code>b</code> holds the value of the current run and <code>bin_r</code> its length.
        </p>
      </Translated>
      <ol>
        <Translated
          vi={
            <li>
              Dòng 11 đến 13: nếu điểm ảnh đầu tiên của luồng nằm trong phạm vi, nạp nó vào <code>b</code> và đặt{' '}
              <code>bin_r = 1</code>.
            </li>
          }
        >
          <li>
            Lines 11 to 13: if the thread's first pixel is in range, load it into <code>b</code> and set{' '}
            <code>bin_r = 1</code>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Dòng 14 đến 26, với mỗi điểm ảnh tiếp theo: nếu nó bằng <code>b</code>, cộng một vào <code>bin_r</code>, việc này chỉ
              chạm vào một thanh ghi. Nếu nó khác, xả: cộng <code>bin_r</code> vào bin riêng của khối cho{' '}
              <code>b</code> bằng một phép toán nguyên tử phạm vi khối, rồi bắt đầu một chuỗi mới với giá trị mới.
            </li>
          }
        >
          <li>
            Lines 14 to 26, for each further pixel: if it equals <code>b</code>, add one to <code>bin_r</code>, which
            touches only a register. If it differs, flush: add <code>bin_r</code> to the block's private bin for{' '}
            <code>b</code> with one block-scope atomic, then start a new run with the new value.
          </li>
        </Translated>
        <Translated vi={<li>Dòng 29 đến 31, sau vòng lặp: xả thêm một lần, vô điều kiện.</li>}>
          <li>Lines 29 to 31, after the loop: flush once more, unconditionally.</li>
        </Translated>
      </ol>
      <Ask q={questionsS96['ch09/9.6/q-final-flush']} label={trText('Predict', 'Dự đoán')} />
      <Translated
        vi={
          <p>
            Lần xả luôn trễ một điểm ảnh, vì chỉ biết một chuỗi đã kết thúc khi một giá trị khác
            đến. Nên khi vòng lặp kết thúc, chuỗi cuối vẫn còn trong thanh ghi, và lần xả cuối là thứ đưa nó vào
            histogram. Một luồng có điểm ảnh đầu tiên vượt quá cuối ảnh bỏ qua toàn bộ phần đó, nhưng nó vẫn tới
            rào chắn và tham gia bước ghi, như ở mục 9.4.
          </p>
        }
      >
        <p>
          The flush always lags one pixel behind, because a run is known to have ended only when a different value
          arrives. So when the loop ends, the last run is still in the registers, and the final flush is what puts it in
          the histogram. A thread whose first pixel is past the end of the image skips all of it, but it still reaches
          the barrier and takes part in the commit, as in section 9.4.
        </p>
      </Translated>
      <Source pages="218–219" figs={trText('Fig. 9.15', 'Hình 9.15')} />
      <Ask q={questionsS96['ch09/9.6/q-runs-count']} label={trText('Work it out', 'Tự tính')} />

      <h2><Translated vi="Xem một warp">Watch one warp</Translated></h2>
      <Translated
        vi={
          <p>
            Trình chạy từng bước chạy Hình 9.15 cho các lane đầu tiên của một khối, mỗi lần một lần lặp. Bắt đầu với ví dụ
            mẫu, nơi lane 0 giữ dãy bạn vừa đếm; cột cuối nhóm các giá trị của mỗi lane thành các chuỗi,
            và mỗi nhóm tốn một phép toán nguyên tử. Rồi chuyển sang bầu trời, và sang sọc ngắn ở cả hai thứ tự.
          </p>
        }
      >
        <p>
          The stepper runs Fig. 9.15 for the first lanes of one block, one iteration at a time. Start with the worked
          example, where lane 0 holds the sequence you just counted; the last column groups each lane's values into runs,
          and each group costs one atomic. Then switch to sky, and to short stripes in both orders.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một warp có các lane chạy mỗi lần lặp cùng nhau. Không có thời gian: nó đếm phép toán nguyên tử và các lần lặp mà các lane
              đi theo hai nhánh khác nhau của lệnh <code>if</code> mới, không đếm chi phí của nhánh nào. Dữ liệu là của khóa học này.
            </p>
          }
        >
          <p>
            One warp whose lanes run each iteration together. No time: it counts atomics and the iterations in which lanes
            take different sides of the new <code>if</code>, not what either costs. The data are ours.
          </p>
        </Translated>
      </Simplification>
      <RunStepper />

      <h2><Translated vi="Khoảng cách quyết định thế nào là một chuỗi">Spacing decides what counts as a run</Translated></h2>
      <Translated
        vi={
          <p>
            Đây là một hệ quả mà sách không nói rõ. Khi chia xen kẽ như Hình 9.15, các điểm ảnh liên tiếp của một luồng
            cách nhau <code>blockDim.x</code>. Một luồng chỉ thấy một chuỗi ở nơi ảnh không đổi trên
            nhiều lần <code>blockDim.x</code> byte liên tiếp: các hàng bầu trời trọn vẹn thì đủ điều kiện, một đoạn tám điểm
            kế nhau bằng nhau thì không, vì không luồng nào đọc được hai điểm trong số đó. Chia liền kề sẽ thấy các chuỗi
            ngắn đó, nhưng chỉ bằng cách từ bỏ hợp nhất truy cập, điều mà mục 9.5 đã cho thấy là đánh đổi sai trên GPU.
          </p>
        }
      >
        <p>
          Here is a consequence the book does not spell out. Under the interleaved partitioning of Fig. 9.15, a thread's
          consecutive pixels are <code>blockDim.x</code> apart. A thread sees a run only where the image is constant over
          many times <code>blockDim.x</code> consecutive bytes: whole rows of sky qualify, a stretch of eight equal
          neighbours does not, because no single thread reads two of them. Contiguous partitioning would see those short
          runs, but only by giving up coalescing, which section 9.5 showed is the wrong trade on a GPU.
        </p>
      </Translated>
      <Beyond title={trText("Ours, not the book's", 'Của khóa học này, không phải của sách')}>
        <Translated
          vi={
            <p>
              Sách lấy bầu trời làm động lực cho việc gộp chuỗi. Lập luận về khoảng cách và preset sọc là của khóa học này: hãy kiểm tra chúng trong
              trình chạy từng bước, nơi sọc tốn khoảng một phép toán nguyên tử mỗi điểm ảnh khi chia xen kẽ và được gộp khi chia
              liền kề.
            </p>
          }
        >
          <p>
            The book motivates run merging with sky. The spacing argument and the stripes preset are ours: check them in
            the stepper, where stripes cost about one atomic per pixel under interleaved partitioning and merge under
            contiguous.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS96['ch09/9.6/q-spacing']} />

      <h2><Translated vi="Cái giá của nó">What it costs</Translated></h2>
      <Translated
        vi={
          <p>
            Sách thẳng thắn về cái giá. Kernel có nhiều câu lệnh hơn và nhiều biến hơn. Trên dữ liệu có ít
            tương đồng giữa các điểm ảnh kế nhau, nó có thể chạy chậm hơn Hình 9.14, vì nó trả giá cho phép so sánh và
            việc ghi chép mà gần như không gộp được gì; trên dữ liệu lặp lại nhiều, nó có thể chạy nhanh hơn nhiều. Lệnh{' '}
            <code>if</code> mới có thể tách một warp, nhưng không có tương đồng thì gần như mọi lane đều xả và tương đồng nhiều thì
            gần như mọi lane đều nối dài, nên các lane phần lớn đồng thuận. Ở nơi chúng bị tách, sách kỳ vọng việc giảm
            tranh chấp sẽ bù lại.
          </p>
        }
      >
        <p>
          The book is frank about the price. The kernel has more statements and more variables. On data with little
          similarity between neighbouring pixels it may run slower than Fig. 9.14, because it pays for the comparison and
          the bookkeeping and merges almost nothing; on highly repetitive data it may run much faster. The new{' '}
          <code>if</code> can split a warp, but with no similarity nearly every lane flushes and with heavy similarity
          nearly every lane extends, so the lanes mostly agree. Where they do split, the book expects the reduced
          contention to make up for it.
        </p>
      </Translated>
      <Source pages="219" />
      <Predict
        id="ch09/9.6/p-runs-fraction"
        prompt={
          <Translated
            vi={
              <>
                Preset bầu trời, <code>blockDim.x = 256</code>, <code>COARSE_FACTOR = 16</code>, xen kẽ. Hình 9.15 phát bao nhiêu
                phần trăm số phép toán nguyên tử của Hình 9.14?
              </>
            }
          >
            <>
              The sky preset, <code>blockDim.x = 256</code>, <code>COARSE_FACTOR = 16</code>, interleaved. What percentage
              of Fig. 9.14's atomics does Fig. 9.15 issue?
            </>
          </Translated>
        }
        unit="%"
      >
        {(v) => <RunsFraction predicted={v} />}
      </Predict>
      <Translated
        vi={
          <p>
            Bảng chạy mọi preset qua cùng một mô hình. Nó đếm phép toán nguyên tử và số lần lặp bị tách, không phải thời gian; lab 9.2
            đo thời gian.
          </p>
        }
      >
        <p>
          The table runs every preset through the same model. It counts atomics and split iterations, not time; lab 9.2
          measures time.
        </p>
      </Translated>
      <AtomicsTable />
      <Ask q={questionsS96['ch09/9.6/q-diverge']} />

      <h2><Translated vi="Mã nguồn">The code</Translated></h2>
      <Translated
        vi={
          <p>
            <code>histogramRuns</code> của lab 9.2 là Hình 9.15 với hệ số gộp việc là một đối số lúc chạy. Sách
            khai báo <code>b</code> là <code>unsigned int</code> và <code>bNext</code> là <code>unsigned char</code>; việc
            trộn lẫn này vô hại, vì cả hai đều giữ một giá trị từ 0 đến 255 và phép so sánh nâng kiểu <code>bNext</code>.
          </p>
        }
      >
        <p>
          Lab 9.2's <code>histogramRuns</code> is Fig. 9.15 with the coarsening factor as a run-time argument. The book
          declares <code>b</code> as <code>unsigned int</code> and <code>bNext</code> as <code>unsigned char</code>; the
          mix is harmless, since both hold a value from 0 to 255 and the comparison promotes <code>bNext</code>.
        </p>
      </Translated>
      <Source pages="218" figs={trText('Fig. 9.15', 'Hình 9.15')} />
      <Listing />
      <Translated
        vi={
          <p>
            Histogram của CUB có cùng ý tưởng này dưới dạng một đường xử lý run-length tùy chọn. Một cách chữa liên quan, mà hướng dẫn profiler
            của NVIDIA đặt lên đầu cho tranh chấp nguyên tử, gộp các cập nhật giữa các lane của một warp bằng shuffle trước khi một lane
            phát phép toán nguyên tử. Nó gộp giữa các lane thay vì giữa các phần tử của một luồng.{' '}
            <a href={href('ch10', '10.7')}>Mục 10.7</a> xây dựng các shuffle, và{' '}
            <a href={href('ch12', '12.3')}>mục 12.3</a> dùng các primitive mức warp để hợp nhất các phép toán nguyên tử của một bộ đếm.
          </p>
        }
      >
        <p>
          CUB's histogram ships the same idea as an optional run-length path. A related remedy, which NVIDIA's profiler
          guide puts first for atomic contention, combines updates across the lanes of a warp with shuffles before one lane
          issues the atomic. It merges across lanes rather than across one thread's elements.{' '}
          <a href={href('ch10', '10.7')}>Section 10.7</a> builds the shuffles, and{' '}
          <a href={href('ch12', '12.3')}>section 12.3</a> uses warp-level primitives to coalesce a counter's atomics.
        </p>
      </Translated>
      <Today checked={trText('2026-11-20 (notes/today/ch09-facts.md D3, A6)', '2026-11-20 (notes/today/ch09-facts.md, mục D3, A6)')}>
        <Translated
          vi={
            <p>
              Đường xử lý run-length của <code>DeviceHistogram</code> trong CUB chính là Hình 9.15, kể cả lần xả cuối, viết bằng{' '}
              <code>atomicAdd_block</code> legacy phạm vi khối. Hướng dẫn phân loại vấn đề của Nsight Compute của NVIDIA liệt kê gộp ở mức warp
              trước, rồi phép rút gọn ở mức khối, rồi riêng tư hóa theo khối.
            </p>
          }
        >
          <p>
            CUB's <code>DeviceHistogram</code> run-length path is Fig. 9.15, final flush included, written with the legacy
            block-scope <code>atomicAdd_block</code>. NVIDIA's Nsight Compute triage guide lists warp-level combining
            first, then block-level reduction, then per-block privatisation.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Token không phải bầu trời">Tokens are not sky</Translated></h2>
      <Translated vi={<p>Đếm token id của một prompt, cho phạt lặp (repetition penalty), cũng là một histogram. Hình 9.15 có giúp được không?</p>}>
        <p>Counting the token ids of a prompt, for repetition penalties, is a histogram too. Would Fig. 9.15 help?</p>
      </Translated>
      <Ask q={questionsS96['ch09/9.6/q-tokens']} label={trText('Predict', 'Dự đoán')} />
      <Translated
        vi={
          <p>
            Các token bằng nhau hiếm khi nằm kề nhau: trong preset token id, một bản thay thế kiểu Zipf, chưa tới một trên hai mươi cặp
            kề nhau là bằng nhau, và khi chia xen kẽ, dù sao một luồng cũng thấy các token cách nhau một độ rộng khối. Trong
            bảng ở trên, Hình 9.15 vẫn phát {fmt(tokens.perElement)} phép toán nguyên tử mỗi token ở <code>blockDim.x = 256</code>{' '}
            và hệ số 16, và một warp 32 lane bị tách ở {pct(tokens.divergentShare ?? 0)} số lần lặp: vài
            id phổ biến khiến một số lane nối dài trong khi số còn lại xả. Token là trường hợp hỗn hợp, chịu cái giá của nhánh
            mà gần như không được phần tiết kiệm nào. Đó là lý do lab 9.3 dùng một cấu trúc khác: một hàng số đếm cho mỗi request,
            cập nhật khi mỗi token được sinh ra.
          </p>
        }
      >
        <p>
          Equal tokens are rarely neighbours: in the token-id preset, a Zipf-like stand-in, fewer than one neighbouring pair in
          twenty is equal, and under interleaved partitioning a thread sees tokens a block width apart anyway. In the
          table above, Fig. 9.15 still issues {fmt(tokens.perElement)} atomics per token at <code>blockDim.x = 256</code>{' '}
          and a factor of 16, and a 32-lane warp splits in {pct(tokens.divergentShare ?? 0)} of its iterations: a few
          common ids make some lanes extend while the rest flush. Tokens are the mixed case, with the cost of the branch
          and almost none of the savings. That is why lab 9.3 uses a different structure: one row of counts per request,
          updated as each token is generated.
        </p>
      </Translated>
    </>
  );
}
