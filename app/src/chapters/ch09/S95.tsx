import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { solutionTabStatus } from '../../components/LabKit';
import { benchRows, fmt, num, useReports, type BenchRow, type Reports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  EXERCISE5,
  FIGS,
  SEGMENT_CHOICES,
  blocksFor,
  index,
  kernelTotals,
  owner,
  plan,
  segmentStart,
  warpFootprint,
  warpsPerBlock,
  type Partitioning,
  type SegmentChoice,
} from '../../sim/histcoarsen';

const LAB = 'ch09/lab2_privatised';
const GPU_BACKENDS = ['metal', 'cuda', 'hip'];
const int = (v: number) => v.toLocaleString('en-US');
const pct = (v: number) => `${fmt(v * 100, 3)}%`;

// Identifiers coloured by the memory they live in, as everywhere in the course.
const HIST_TIERS: Record<string, Tier> = {
  image: 'global', bins: 'global', counts: 'global',
  bins_s: 'shared', binsS: 'shared', bins_s_ref: 'shared', localBins: 'shared',
  segment: 'reg', i: 'reg', c: 'reg', b: 'reg', bin: 'reg', tid: 'reg', bid: 'reg', bdim: 'reg', binValue: 'reg',
};

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
/** 1-based lines of `text` that contain any of the needles. */
const linesWith = (text: string, needles: string[]) =>
  text.split('\n').flatMap((l, i) => (needles.some((n) => l.includes(n)) ? [i + 1] : []));

function useListingStatus() {
  const { reports } = useReports();
  // The same words as the labs page's status table and reference-solution tabs.
  return { cu: solutionTabStatus(reports, LAB, 'kernel.cu'), metal: solutionTabStatus(reports, LAB, 'kernel.metal') };
}

function Listing() {
  const sol = useLabFile(LAB + '/solution/kernel.cu');
  const ref = useLabFile(LAB + '/provided/ref_kernels.cu');
  const msl = useLabFile(LAB + '/solution/kernel.metal');
  const status = useListingStatus();
  const inter = region(sol, 'shared');
  const contig = region(ref, 'contiguous');
  const metal = region(msl, 'shared');
  if (!inter && !contig)
    return (
      <Translated
        vi={
          <p className="muted" data-listing="missing">
            Hai kernel được nạp từ <code>labs/{LAB}/solution/kernel.cu</code> và{' '}
            <code>labs/{LAB}/provided/ref_kernels.cu</code> khi lab 9.2 đã sẵn sàng.
          </p>
        }
      >
        <p className="muted" data-listing="missing">
          The two kernels are loaded from <code>labs/{LAB}/solution/kernel.cu</code> and{' '}
          <code>labs/{LAB}/provided/ref_kernels.cu</code> once lab 9.2 is in place.
        </p>
      </Translated>
    );
  const at = (t: string) => linesWith(t, ['i = segment']);
  const tabs = [
    // Short labels: three tabs and a status line have to share one bar on a phone.
    ...(inter ? [{ label: tr('Interleaved', 'Xen kẽ'), code: inter, tiers: HIST_TIERS, status: status.cu, highlight: at(inter) }] : []),
    ...(contig ? [{ label: tr('Contiguous', 'Liền kề'), code: contig, tiers: HIST_TIERS, status: status.cu, highlight: at(contig) }] : []),
    ...(metal ? [{ label: 'Metal', code: metal, tiers: HIST_TIERS, status: status.metal, highlight: at(metal) }] : []),
  ];
  return (
    <Code
      tabs={tabs}
      caption={
        <Translated
          vi={
            <>
              Xen kẽ: <code>histogramShared</code> từ lời giải tham chiếu của lab 9.2 (<code>labs/{LAB}/solution/</code>), là
              Hình 9.14 của sách, bằng CUDA. Liền kề: bản đối chứng mà lab 9.2 cung cấp, và kiểm thử, trong{' '}
              <code>provided/ref_kernels.cu</code>, là Hình 9.12 của sách. Metal: kernel xen kẽ bằng MSL. Dòng được tô sáng
              là dòng 12 của sách, dòng duy nhất khác nhau. Hãy tự viết bản của bạn trong lab trước khi nghiên cứu các bản này.
            </>
          }
        >
          <>
            Interleaved: <code>histogramShared</code> from lab 9.2's reference solution (<code>labs/{LAB}/solution/</code>), the
            book's Fig. 9.14, in CUDA. Contiguous: the comparator lab 9.2 provides, and tests, in{' '}
            <code>provided/ref_kernels.cu</code>, the book's Fig. 9.12. Metal: the interleaved kernel in MSL. The highlighted
            line is the book's line 12, the only one that differs. Write your own in the lab before you study these.
          </>
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- calculator: blocks, copies, commits
const B_CALC = [32, 64, 128, 256, 512, 1024];
const C_CALC = [1, 2, 4, 8, 16, 32, 64, 128];
const BIG = { n: 4096 * 4096, blockDim: 256, coarse: 16, bins: 256 };

function PlanCalculator() {
  const [n, setN] = useState<number>(BIG.n);
  const [nText, setNText] = useState(String(BIG.n));
  const [B, setB] = useState<number>(BIG.blockDim);
  const [C, setC] = useState<number>(BIG.coarse);
  const [bins, setBins] = useState<number>(BIG.bins);
  const [binsText, setBinsText] = useState(String(BIG.bins));
  const preset = (pn: number, pb: number, pc: number, pbins: number) => {
    setN(pn); setNText(String(pn)); setB(pb); setC(pc); setBins(pbins); setBinsText(String(pbins));
  };
  const p1 = plan(n, B, 1, bins);
  const pc = plan(n, B, C, bins);
  const rows = [
    { k: '9.6', label: tr('Fig. 9.6: no private copies', 'Hình 9.6: không có bản sao riêng'), blocks: blocksFor(n, B, 1), copies: 0, zero: 0, shared: 0, global: p1.unprivatisedAtomics },
    { k: '9.10', label: tr('Fig. 9.10: a copy per block, factor 1', 'Hình 9.10: một bản sao mỗi khối, hệ số 1'), blocks: p1.blocks, copies: p1.copies, zero: p1.zeroWrites, shared: p1.sharedAtomics, global: p1.commitAtomicsMax },
    { k: '9.14', label: tr(`Fig. 9.14: a copy per block, factor ${C}`, `Hình 9.14: một bản sao mỗi khối, hệ số ${C}`), blocks: pc.blocks, copies: pc.copies, zero: pc.zeroWrites, shared: pc.sharedAtomics, global: pc.commitAtomicsMax },
  ];
  const capped = pc.pixelsPerBlock < bins;
  return (
    <Bench
      title={tr('Blocks, copies, commits', 'Khối, bản sao, lần ghi')}
      task={
        <Translated vi="Kiểm tra đáp án của bạn với preset bài tập 5, rồi thay đổi hệ số và số bin.">
          Check your answer with the exercise 5 preset, then change the factor and the bins.
        </Translated>
      }
      foot={
        <Translated vi="Số lượng, không phải thời gian. Một khối chỉ ghi các bin khác không của nó, nên cột cuối là số lần ghi nhiều nhất có thể phát ra.">
          Counts, not time. A block commits only its non-zero bins, so the last column is the most the commits can issue.
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="muted"><Translated vi="Preset:">Presets:</Translated></span>
        <button className="quiet" onClick={() => preset(EXERCISE5.n, EXERCISE5.blockDim, EXERCISE5.coarse, EXERCISE5.bins)}><Translated vi="Bài tập 5 (p. 220)">Exercise 5 (p. 220)</Translated></button>
        <button className="quiet" onClick={() => preset(BIG.n, BIG.blockDim, BIG.coarse, BIG.bins)}><Translated vi="Một ảnh 4096 × 4096, 256 bin">A 4096 × 4096 image, 256 bins</Translated></button>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi="Số điểm ảnh">Pixels</Translated>
          <input type="number" min={1} value={nText} aria-label={tr('pixels', 'số điểm ảnh')}
            onChange={(e) => { setNText(e.target.value); const v = Math.floor(Number(e.target.value)); if (v >= 1 && v <= 2 ** 31) setN(v); }} />
        </label>
        <label className="field">
          <Translated vi="Số luồng mỗi khối">Threads per block</Translated>
          <select value={B} aria-label={tr('threads per block', 'số luồng mỗi khối')} onChange={(e) => setB(Number(e.target.value))}>
            {B_CALC.map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
        <label className="field">
          COARSE_FACTOR
          <select value={C} aria-label={tr('coarse factor', 'hệ số gộp việc')} onChange={(e) => setC(Number(e.target.value))}>
            {C_CALC.map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Số bin">Bins</Translated>
          <input type="number" min={1} value={binsText} aria-label={tr('bins', 'số bin')}
            onChange={(e) => { setBinsText(e.target.value); const v = Math.floor(Number(e.target.value)); if (v >= 1 && v <= 1 << 20) setBins(v); }} />
        </label>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-plan>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}><Translated vi="Kernel">Kernel</Translated></th>
              <th className="num"><Translated vi="Số khối">Blocks</Translated></th>
              <th className="num"><Translated vi="Bản sao riêng được xóa và ghi">Private copies zeroed and committed</Translated></th>
              <th className="num"><Translated vi="Lần ghi bộ nhớ chia sẻ để xóa chúng">Shared-memory writes to zero them</Translated></th>
              <Translated vi={<th className="num"><span className="t-shared">Phép toán nguyên tử phạm vi khối</span> trên các bản sao</th>}>
                <th className="num"><span className="t-shared">Block-scope atomics</span> on the copies</th>
              </Translated>
              <Translated vi={<th className="num"><span className="t-global">Phép toán nguyên tử phạm vi device</span>, nhiều nhất</th>}>
                <th className="num"><span className="t-global">Device-scope atomics</span>, at most</th>
              </Translated>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.k} data-kernel={r.k} data-blocks={r.blocks} data-global={r.global}>
                <td style={{ paddingLeft: '0.6rem' }}>{r.label}</td>
                <td className="num">{int(r.blocks)}</td>
                <td className="num">{r.copies ? int(r.copies) : '–'}</td>
                <td className="num">{r.zero ? int(r.zero) : '–'}</td>
                <td className="num">{r.shared ? int(r.shared) : '–'}</td>
                <td className="num"><b>{int(r.global)}</b></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Translated
        vi={
          <p className="muted" style={{ margin: '0.7rem 0 0', fontSize: '0.86rem' }}>
            Mỗi khối của kernel đã gộp việc xử lý {int(pc.pixelsPerBlock)} điểm ảnh.{' '}
            {capped
              ? `Con số đó ít hơn ${int(bins)} bin, nên một khối có thể lấp đầy nhiều nhất ${int(pc.pixelsPerBlock)} bin và ghi không quá số đó.`
              : `Với ${int(bins)} bin, mọi bin của mọi khối đều có thể khác không, nên cận trên là số khối × số bin.`}{' '}
            Từ hệ số 1 đến hệ số {C}: {int(p1.blocks)} bản sao còn {int(pc.blocks)}
            {p1.commitAtomicsMax ? `, và số lần ghi giảm ${fmt(p1.commitAtomicsMax / Math.max(1, pc.commitAtomicsMax))}×` : ''}.
          </p>
        }
      >
        <p className="muted" style={{ margin: '0.7rem 0 0', fontSize: '0.86rem' }}>
          Each block of the coarsened kernel handles {int(pc.pixelsPerBlock)} pixels.{' '}
          {capped
            ? `That is fewer than ${int(bins)} bins, so a block can fill at most ${int(pc.pixelsPerBlock)} of them and commits no more than that.`
            : `With ${int(bins)} bins, every bin of every block may be non-zero, so the bound is blocks × bins.`}{' '}
          From factor 1 to factor {C}: {int(p1.blocks)} copies become {int(pc.blocks)}
          {p1.commitAtomicsMax ? `, and the commits shrink by ${fmt(p1.commitAtomicsMax / Math.max(1, pc.commitAtomicsMax))}×` : ''}.
        </p>
      </Translated>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: who reads what in iteration c
const N_CHOICES = [24, 4096, 65_536, 1_048_576];
const B_CHOICES = [8, 32, 64, 128, 256];
const WIN = 48; // cells drawn at most

interface XS {
  p: Partitioning;
  B: number;
  C: number;
  n: number;
  block: number;
  warp: number;
  c: number;
  /** 32 or 64, or 0 for a width typed in. */
  wMode: number;
  wCustom: number;
  e: number;
  seg: SegmentChoice['id'];
  segCustom: number;
  thread: number | null;
}
const START: XS = { p: 'contiguous', B: 8, C: 3, n: 24, block: 0, warp: 0, c: 0, wMode: 32, wCustom: 16, e: 1, seg: 'nvidia', segCustom: 64, thread: null };
const PRESETS: { label: string; s: Partial<XS> }[] = [
  {
    get label() {
      return tr('Figs. 9.11 and 9.13', 'Hình 9.11 và 9.13');
    },
    s: { B: 8, C: 3, n: 24, e: 1 },
  },
  {
    get label() {
      return tr('One-byte pixels, factor 32', 'Điểm ảnh một byte, hệ số 32');
    },
    s: { B: 256, C: 32, n: 1_048_576, e: 1 },
  },
  {
    get label() {
      return tr('Token ids, factor 8', 'Token id, hệ số 8');
    },
    s: { B: 256, C: 8, n: 1_048_576, e: 4 },
  },
];
const widthOf = (s: XS) => (s.wMode ? s.wMode : Math.max(1, Math.min(128, Math.floor(s.wCustom) || 1)));
const segOf = (s: XS) => SEGMENT_CHOICES.find((x) => x.id === s.seg)!.bytes ?? s.segCustom;

/** Keep block, warp, iteration and thread inside the launch the other choices define. */
function fix(s: XS): XS {
  const W = widthOf(s);
  const block = Math.max(0, Math.min(s.block, blocksFor(s.n, s.B, s.C) - 1));
  const warp = Math.max(0, Math.min(s.warp, warpsPerBlock(s.B, W) - 1));
  const c = Math.max(0, Math.min(s.c, s.C - 1));
  const inWarp = (t: number) => t >= warp * W && t < Math.min(s.B, (warp + 1) * W);
  return { ...s, block, warp, c, thread: s.thread != null && inWarp(s.thread) ? s.thread : null };
}

/** Warp widths the reader's GPU backends reported, where they report one. */
function useDeviceWidths(): { backend: string; device: string; width: number }[] {
  const { reports } = useReports();
  return GPU_BACKENDS.filter((b) => reports[b]).flatMap((b) => {
    const w = num(Number(reports[b].device.warpSize));
    return w ? [{ backend: b, device: String(reports[b].device.name ?? b), width: w }] : [];
  });
}

function ReadsExplorer() {
  const [s0, setS] = useState<XS>(START);
  const s = fix(s0);
  const set = (o: Partial<XS>) => setS(fix({ ...s, ...o }));
  const W = widthOf(s);
  const seg = segOf(s);
  const choice = SEGMENT_CHOICES.find((x) => x.id === s.seg)!;
  const devices = useDeviceWidths();
  const blocks = blocksFor(s.n, s.B, s.C);
  const warps = warpsPerBlock(s.B, W);
  const fp = warpFootprint({ p: s.p, n: s.n, blockDim: s.B, coarse: s.C, block: s.block, warp: s.warp, c: s.c, warpWidth: W, elemBytes: s.e, segmentBytes: seg });
  const totals = useMemo(
    () =>
      (['contiguous', 'interleaved'] as Partitioning[]).map((p) => ({
        p,
        t: kernelTotals({ p, n: s.n, blockDim: s.B, coarse: s.C, warpWidth: W, elemBytes: s.e, segmentBytes: seg }),
      })),
    [s.n, s.B, s.C, W, s.e, seg],
  );

  // The thread whose line 12 is spelled out: the one clicked, else the warp's first.
  const t = s.thread ?? s.warp * W;
  const S = segmentStart(s.block, s.B, s.C);
  const iT = index(s.p, s.block, t, s.c, s.B, s.C);
  const lane0 = s.warp * W;
  const laneN = Math.min(s.B, lane0 + W) - 1;

  // Which cells of the block's segment to draw: all of them, or a window that starts
  // at the segment holding the warp's first element in this iteration.
  const span = s.B * s.C;
  const len = Math.min(WIN, span);
  const segE = seg / s.e; // elements per segment
  let w0 = S;
  if (span > WIN) {
    const first = fp.elements[0] ?? S;
    w0 = Math.min(Math.max(S, Math.floor(first / segE) * segE), S + span - WIN);
  }
  const cells = Array.from({ length: len }, (_, k) => {
    const i = w0 + k;
    const o = owner(s.p, i, s.B, s.C);
    const mine = o.thread >= lane0 && o.thread <= laneN;
    return { i, o, inRange: i < s.n, hot: mine && o.c === s.c, done: mine && o.c < s.c, picked: o.thread === t };
  });
  const lanesShownInWindow = new Set(cells.filter((x) => x.hot).map((x) => x.o.thread)).size;

  // Segments under the window, and the whole span of segments this load touches.
  const firstSeg = Math.floor((w0 * s.e) / seg);
  const lastSeg = Math.floor(((w0 + len) * s.e - 1) / seg);
  const touched = new Set(fp.segments);
  const cw = 18;
  const pad = 6;
  const vbw = len * cw + 2 * pad;
  const segLo = fp.segments[0] ?? 0;
  const segHi = fp.segments[fp.segments.length - 1] ?? 0;
  const spanSegs = fp.segments.length ? segHi - segLo + 1 : 0;
  // The fewest segments the active lanes could need, if their bytes were consecutive and aligned.
  const best = Math.max(1, Math.ceil(fp.bytesNeeded / seg));
  const sw = spanSegs ? Math.max(2, Math.min(18, Math.floor(860 / spanSegs))) : 18;

  const formula =
    s.p === 'contiguous'
      ? `i = segment + threadIdx.x·COARSE_FACTOR + c = ${int(S)} + ${t}·${s.C} + ${s.c} = ${int(iT)}`
      : `i = segment + c·blockDim.x + threadIdx.x = ${int(S)} + ${s.c}·${s.B} + ${t} = ${int(iT)}`;

  return (
    <Bench
      title={tr('Who reads what in iteration c', 'Ai đọc gì ở lần lặp c')}
      task={
        <Translated vi="Tái hiện hai hình vẽ của sách, rồi kéo hệ số và xem lần nạp của một warp chạm vào những gì.">
          Reproduce the book's two drawings, then drag the factor and watch what one warp's load touches.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Ảnh là dải đầu vào; ô được tô là những gì warp được chọn đọc, đậm ở lần lặp này, nhạt ở các lần
              trước. Số trong một ô là luồng sở hữu nó. Segment được tính theo đơn vị mà nhà cung cấp được chọn
              ghi trong tài liệu ({choice.facts}); mảng bắt đầu ở một ranh giới segment.
            </>
          }
        >
          <>
            The image is the input strip; shaded cells are what the chosen warp reads, dark in this iteration, light in
            earlier ones. The number in a cell is the thread that owns it. Segments are counted in the unit the chosen
            vendor documents ({choice.facts}); the array starts on a segment boundary.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="muted"><Translated vi="Preset:">Presets:</Translated></span>
        {PRESETS.map((x) => (
          <button key={x.label} className="quiet" onClick={() => set({ ...x.s, block: 0, warp: 0, c: 0, thread: null })}>
            {x.label}
          </button>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span className="seg" role="group" aria-label={tr('Partitioning', 'Cách chia')}>
          {(['contiguous', 'interleaved'] as Partitioning[]).map((p) => (
            <button key={p} aria-pressed={s.p === p} onClick={() => set({ p })}>
              {p === 'contiguous' ? tr('Contiguous', 'Liền kề') : tr('Interleaved', 'Xen kẽ')} ({FIGS[p].code})
            </button>
          ))}
        </span>
        <label className="field">
          <Translated vi="Số luồng mỗi khối">Threads per block</Translated>
          <select value={s.B} aria-label={tr('threads per block', 'số luồng mỗi khối')} onChange={(e) => set({ B: Number(e.target.value) })}>
            {B_CHOICES.map((x) => <option key={x} value={x}>{x}{x === 8 ? tr(' (the book’s drawing)', ' (hình vẽ của sách)') : ''}</option>)}
          </select>
        </label>
        <label className="field">
          COARSE_FACTOR = {s.C}
          <input type="range" min={1} max={64} value={s.C} aria-label={tr('coarse factor', 'hệ số gộp việc')} onChange={(e) => set({ C: Number(e.target.value) })} />
        </label>
        <label className="field">
          <Translated vi="Ảnh">Image</Translated>
          <select value={s.n} aria-label={tr('pixels', 'số điểm ảnh')} onChange={(e) => set({ n: Number(e.target.value) })}>
            {N_CHOICES.map((x) => <option key={x} value={x}>{int(x)}{tr(' elements', ' phần tử')}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <select value={s.wMode} aria-label={tr('warp width', 'độ rộng warp')} onChange={(e) => set({ wMode: Number(e.target.value) })}>
            <option value={32}>{tr('32 lanes (NVIDIA; AMD RDNA)', '32 lane (NVIDIA; AMD RDNA)')}</option>
            <option value={64}>{tr('64 lanes (AMD CDNA)', '64 lane (AMD CDNA)')}</option>
            <option value={0}>{tr('another width', 'độ rộng khác')}</option>
          </select>
        </label>
        {s.wMode === 0 ? (
          <label className="field">
            <Translated vi="Số lane">Lanes</Translated>
            <input type="number" min={1} max={128} value={s.wCustom} aria-label={tr('custom warp width', 'độ rộng warp tùy chỉnh')} onChange={(e) => set({ wCustom: Number(e.target.value) })} />
          </label>
        ) : null}
        <label className="field">
          <Translated vi="Phần tử">Element</Translated>
          <select value={s.e} aria-label={tr('element size', 'kích thước phần tử')} onChange={(e) => set({ e: Number(e.target.value) })}>
            <option value={1}>{tr('1-byte pixel', 'điểm ảnh 1 byte')}</option>
            <option value={4}>{tr('4-byte token id', 'token id 4 byte')}</option>
          </select>
        </label>
        <label className="field">
          <Translated vi="Segment bộ nhớ">Memory segment</Translated>
          <select value={s.seg} aria-label={tr('segment', 'segment')} onChange={(e) => set({ seg: e.target.value as SegmentChoice['id'] })}>
            {SEGMENT_CHOICES.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
          </select>
        </label>
        {choice.bytes == null ? (
          <label className="field">
            <Translated vi="Lựa chọn của bạn (không phải dữ kiện)">Your choice (not a fact)</Translated>
            <select value={s.segCustom} aria-label={tr('segment bytes', 'số byte của segment')} onChange={(e) => set({ segCustom: Number(e.target.value) })}>
              {[32, 64, 128].map((v) => <option key={v} value={v}>{v}{tr(' bytes', ' byte')}</option>)}
            </select>
          </label>
        ) : null}
      </div>
      <p className="muted" style={{ margin: '0 0 0.7rem', fontSize: '0.84rem' }}>
        {devices.length
          ? (
            <Translated vi={<>Thiết bị của bạn đã báo: {devices.map((d) => `${d.device} (${d.backend}), ${d.width} lane`).join('; ')}.</>}>
              <>Your devices reported: {devices.map((d) => `${d.device} (${d.backend}), ${d.width} lanes`).join('; ')}.</>
            </Translated>
          )
          : (
            <Translated vi={<>Chưa có độ rộng warp nào trong báo cáo của bạn. Trên Metal, độ rộng được báo theo pipeline, không theo thiết bị.</>}>
              <>No warp width in your reports yet. On Metal the width is reported per pipeline, not per device.</>
            </Translated>
          )}
      </p>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi="Khối">Block</Translated>
          <input type="range" min={0} max={blocks - 1} value={s.block} aria-label={tr('block', 'khối')} disabled={blocks < 2}
            onChange={(e) => set({ block: Number(e.target.value) })} />
        </label>
        <Translated vi={<span className="muted">khối {int(s.block)} trên {int(blocks)}</span>}>
          <span className="muted">block {int(s.block)} of {int(blocks)}</span>
        </Translated>
        {warps > 1 ? (
          <label className="field">
            <Translated vi="Warp">Warp</Translated>
            <select value={s.warp} aria-label={tr('warp', 'warp')} onChange={(e) => set({ warp: Number(e.target.value), thread: null })}>
              {Array.from({ length: warps }, (_, w) => <option key={w} value={w}>{w}{tr(' (threads ', ' (luồng ')}{w * W}{tr(' to ', ' đến ')}{Math.min(s.B, (w + 1) * W) - 1})</option>)}
            </select>
          </label>
        ) : null}
        <span className="row" style={{ gap: '0.4rem' }}>
          <button onClick={() => set({ c: 0 })} disabled={s.c === 0}>c = 0</button>
          <button onClick={() => set({ c: s.c - 1 })} disabled={s.c === 0}><Translated vi="Lùi">Back</Translated></button>
          <button className="primary" onClick={() => set({ c: s.c + 1 })} disabled={s.c >= s.C - 1}><Translated vi="Bước">Step</Translated></button>
        </span>
        <label className="field">
          <Translated vi="Lần lặp c">Iteration c</Translated>
          <input type="range" min={0} max={s.C - 1} value={s.c} aria-label={tr('iteration', 'lần lặp')} disabled={s.C < 2} onChange={(e) => set({ c: Number(e.target.value) })} />
        </label>
        <Translated vi={<span className="muted" aria-live="polite">lần lặp c = {s.c} (0 đến {s.C - 1})</span>}>
          <span className="muted" aria-live="polite">iteration c = {s.c} (0 to {s.C - 1})</span>
        </Translated>
      </div>

      <Translated
        vi={
          <p className="narrate" style={{ margin: '0 0 0.8rem' }} data-formula={iT}>
            Luồng {t} (lane {t - lane0} của warp {s.warp}) trong khối {int(s.block)}, lần lặp {s.c}:{' '}
            <code style={{ whiteSpace: 'normal' }}>{formula}</code>, trong đó segment = COARSE_FACTOR·blockIdx.x·blockDim.x ={' '}
            {s.C}·{int(s.block)}·{s.B} = {int(S)}.
            {iT >= s.n ? ` Vị trí đó đã quá cuối ảnh, nên phép kiểm tra biên bỏ qua nó.` : ''}
            {fp.active > 1 ? ` Luồng kế bên nó trong warp đọc xa hơn ${fp.laneStride} phần tử.` : ''}{' '}
            <span className="muted">Bấm vào một ô để chọn luồng của ô đó.</span>
          </p>
        }
      >
        <p className="narrate" style={{ margin: '0 0 0.8rem' }} data-formula={iT}>
          Thread {t} (lane {t - lane0} of warp {s.warp}) in block {int(s.block)}, iteration {s.c}:{' '}
          <code style={{ whiteSpace: 'normal' }}>{formula}</code>, where segment = COARSE_FACTOR·blockIdx.x·blockDim.x ={' '}
          {s.C}·{int(s.block)}·{s.B} = {int(S)}.
          {iT >= s.n ? ` That is past the end of the image, so the bounds check skips it.` : ''}
          {fp.active > 1 ? ` Its neighbour in the warp reads ${fp.laneStride} element${fp.laneStride === 1 ? '' : 's'} further on.` : ''}{' '}
          <span className="muted">Click a cell to pick its thread.</span>
        </p>
      </Translated>

      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${vbw} 78`} role="img" aria-label={tr("The block's segment of the image, with what the warp reads", 'Đoạn ảnh của khối, cùng những gì warp đọc')}
          style={{ width: '100%', maxWidth: vbw * 1.15, minWidth: len * 14, display: 'block' }}>
          {cells.map((x, k) => {
            const X = pad + k * cw;
            const fill = !x.inRange ? 'transparent' : x.hot ? 'var(--global)' : x.done ? 'var(--global-wash)' : 'var(--cell)';
            return (
              <g key={x.i} data-cell={x.i} data-owner={x.o.thread} data-hot={x.hot ? 1 : 0} style={{ cursor: 'pointer' }}
                onClick={() => set({ thread: x.o.thread, warp: Math.floor(x.o.thread / W) })}>
                {(x.i - S) % 8 === 0 || k === 0 ? (
                  <text x={X + 1} y={10} fontSize="8.5" style={{ fill: 'var(--ink-3)' }}>{int(x.i)}</text>
                ) : null}
                <rect x={X + 1} y={14} width={cw - 2} height={24} rx={2}
                  style={{ fill, stroke: x.picked ? 'var(--ink)' : x.hot || x.done ? 'var(--global)' : 'var(--rule)', strokeWidth: x.picked ? 2 : 1, strokeDasharray: x.inRange ? undefined : '3 2' }} />
                <text x={X + cw / 2} y={30} fontSize={x.o.thread > 99 ? 7 : 9} textAnchor="middle"
                  style={{ fill: x.hot ? 'var(--page)' : x.inRange ? 'var(--ink-2)' : 'var(--ink-3)', fontWeight: x.hot ? 650 : 400 }}>
                  {x.o.thread}
                </text>
              </g>
            );
          })}
          {Array.from({ length: lastSeg - firstSeg + 1 }, (_, k) => {
            const sg = firstSeg + k;
            const a = Math.max(w0, sg * segE);
            const b = Math.min(w0 + len, (sg + 1) * segE);
            const x0 = pad + (a - w0) * cw + 1;
            const x1 = pad + (b - w0) * cw - 1;
            const on = touched.has(sg);
            return (
              <g key={sg}>
                <path d={`M ${x0} 44 V 50 H ${x1} V 44`} style={{ fill: 'none', stroke: on ? 'var(--global)' : 'var(--rule)', strokeWidth: on ? 2 : 1 }} />
                {x1 - x0 > 60 ? (
                  <text x={(x0 + x1) / 2} y={62} fontSize="8.5" textAnchor="middle" style={{ fill: on ? 'var(--global)' : 'var(--ink-3)' }}>
                    <Translated vi={<>segment {int(sg)}{on ? ': bị chạm' : ''}</>}>segment {int(sg)}{on ? ': touched' : ''}</Translated>
                  </text>
                ) : null}
              </g>
            );
          })}
          <text x={pad} y={75} fontSize="8.5" style={{ fill: 'var(--ink-3)' }}>
            {span > WIN
              ? tr(
                  `elements ${int(w0)} to ${int(w0 + len - 1)} of this block's ${int(S)} to ${int(S + span - 1)}; ${lanesShownInWindow} of the warp's ${fp.active} reads fall in this window`,
                  `phần tử ${int(w0)} đến ${int(w0 + len - 1)} trong ${int(S)} đến ${int(S + span - 1)} của khối này; ${lanesShownInWindow} trên ${fp.active} lần đọc của warp rơi vào cửa sổ này`,
                )
              : tr(
                  `this block's segment: elements ${int(S)} to ${int(S + span - 1)}`,
                  `đoạn của khối này: phần tử ${int(S)} đến ${int(S + span - 1)}`,
                )}
          </text>
        </svg>
      </div>

      <div className="mx-label" style={{ margin: '0.7rem 0 0.2rem' }}>
        <Translated vi="Các segment mà lần nạp này chạm vào">Segments this load touches</Translated>
        {spanSegs > 1
          ? tr(
              `: ${fp.transactions} of the ${int(spanSegs)} from segment ${int(segLo)} to ${int(segHi)}`,
              `: ${fp.transactions} trên ${int(spanSegs)} segment từ segment ${int(segLo)} đến ${int(segHi)}`,
            )
          : ''}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${Math.max(1, spanSegs) * sw + 2} 16`} role="img" aria-label={tr('Segments touched by this load', 'Các segment bị lần nạp này chạm vào')}
          style={{ width: '100%', maxWidth: Math.max(1, spanSegs) * sw * 1.5 + 2, height: 16, display: 'block' }} preserveAspectRatio="none" /* i18n-ignore: SVG attribute value */>
          {Array.from({ length: spanSegs }, (_, k) => (
            <rect key={k} x={1 + k * sw} y={1} width={Math.max(1, sw - 1)} height={14}
              style={{ fill: touched.has(segLo + k) ? 'var(--global)' : 'transparent', stroke: 'var(--rule)', strokeWidth: 0.5 }} />
          ))}
        </svg>
      </div>

      <div className="row" style={{ marginTop: '0.7rem' }} data-readout data-elements={fp.elements.join(',')} data-transactions={fp.transactions} data-active={fp.active}>
        <Translated vi={<span>Warp {s.warp}: <b>{fp.active}</b> lane đang hoạt động</span>}>
          <span>Warp {s.warp}: <b>{fp.active}</b> active lane{fp.active === 1 ? '' : 's'}</span>
        </Translated>
        <Translated vi={<span><b>{fp.transactions}</b> segment {seg} byte{choice.bytes == null ? ' (lựa chọn của bạn)' : ''}</span>}>
          <span><b>{fp.transactions}</b> segment{fp.transactions === 1 ? '' : 's'} of {seg} bytes{choice.bytes == null ? ' (your choice)' : ''}</span>
        </Translated>
        <Translated vi={<span>cần <b>{int(fp.bytesNeeded)}</b> byte, nạp về <b>{int(fp.bytesFetched)}</b></span>}>
          <span>needed <b>{int(fp.bytesNeeded)}</b> bytes, fetched <b>{int(fp.bytesFetched)}</b></span>
        </Translated>
        <Translated vi={<span className={'chip ' + (fp.transactions <= best ? 'ok' : fp.transactions >= 4 * best ? 'fail' : 'global')}>{pct(fp.efficiency)} số byte nạp về được dùng</span>}>
          <span className={'chip ' + (fp.transactions <= best ? 'ok' : fp.transactions >= 4 * best ? 'fail' : 'global')}>{pct(fp.efficiency)} of fetched bytes used</span>
        </Translated>
        <span className="muted">
          {fp.transactions <= best
            ? tr('as few segments as these lanes allow', 'ít segment nhất mà các lane này cho phép')
            : tr(`${fmt(fp.transactions / best)}× the fewest these lanes could need`, `${fmt(fp.transactions / best)}× mức ít nhất các lane này có thể cần`)}
        </span>
      </div>

      <div style={{ overflowX: 'auto', marginTop: '0.9rem' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-totals>
          <thead>
            <tr>
              <Translated vi={<th style={{ paddingLeft: '0.6rem' }}>Cả kernel, {int(s.n)} phần tử</th>}>
                <th style={{ paddingLeft: '0.6rem' }}>Whole kernel, {int(s.n)} elements</th>
              </Translated>
              <th className="num"><Translated vi="Lần nạp của warp">Warp loads</Translated></th>
              <th className="num"><Translated vi="Segment đã nạp">Segments fetched</Translated></th>
              <th className="num"><Translated vi="Mỗi lần nạp">Per load</Translated></th>
              <th className="num"><Translated vi="Byte nạp về trên mỗi byte cần">Bytes fetched per byte needed</Translated></th>
            </tr>
          </thead>
          <tbody>
            {totals.map(({ p, t: x }) => (
              <tr key={p} data-p={p} data-transactions={x.transactions} style={{ fontWeight: p === s.p ? 650 : 400 }}>
                <td style={{ paddingLeft: '0.6rem' }}>{p === 'contiguous' ? tr('Contiguous', 'Liền kề') : tr('Interleaved', 'Xen kẽ')} ({FIGS[p].code})</td>
                <td className="num">{int(x.loads)}</td>
                <td className="num">{int(x.transactions)}</td>
                <td className="num">{x.minPerLoad === x.maxPerLoad ? x.minPerLoad : tr(`${x.minPerLoad} to ${x.maxPerLoad}`, `${x.minPerLoad} đến ${x.maxPerLoad}`)}</td>
                <td className="num">{x.bytesNeeded ? fmt(x.bytesFetched / x.bytesNeeded) : '–'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the measured comparison (lab 9.2)
function labRows(reports: Reports, b: string): BenchRow[] {
  const s = benchRows(reports[b], LAB, 'solution');
  return s.length ? s : benchRows(reports[b], LAB, 'work');
}
const pick = (rows: BenchRow[], name: string) => rows.find((r) => r.name === name && num(r.coarse) === 16 && r.data === 'uniform');

function InterleavedGain({ predicted }: { predicted: number }) {
  const { reports, loading } = useReports();
  const found = GPU_BACKENDS.filter((b) => reports[b]).flatMap((b) => {
    const rows = labRows(reports, b);
    const inter = num(pick(rows, 'shared')?.seconds);
    const contig = num(pick(rows, 'contiguous')?.seconds);
    return inter && contig ? [{ b, device: String(reports[b].device.name ?? b), inter, contig, gain: contig / inter }] : [];
  });
  if (loading) return <Translated vi={<p className="muted">Đang đọc kết quả lab của bạn.</p>}><p className="muted">Reading your lab results.</p></Translated>;
  if (!found.length)
    return (
      <Translated
        vi={
          <p data-gain="none">
            Chưa có số đo nào trên máy này. Chạy <code>python3 labs/pmpp.py run 9.2 --solution --bench</code> (nó ghi lại
            các dòng mà ô này đọc), rồi tải lại trang: nó sẽ so sánh dự đoán của bạn với kết quả.
          </p>
        }
      >
        <p data-gain="none">
          No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 9.2 --solution --bench</code> (it records
          the rows this box reads), then reload the page: it will compare your prediction with the result.
        </p>
      </Translated>
    );
  return (
    <>
      {found.map((f) => {
        // First the direction: a prediction and a measurement on opposite sides of 1 are not
        // "close", however near their magnitudes are.
        const opposite = (predicted - 1) * (f.gain - 1) < 0;
        return (
          <Translated
            key={f.b}
            vi={
              <p data-gain={f.gain} data-direction={opposite ? 'opposite' : 'same'}>
                Trên <b>{f.device}</b> ({f.b}), ở hệ số 16 trên các byte phân bố đều, kernel xen kẽ mất {fmt(f.inter * 1e3)} ms và
                kernel liền kề mất {fmt(f.contig * 1e3)} ms: kernel xen kẽ chạy nhanh bằng <b>{fmt(f.gain)}×</b>.{' '}
                {opposite
                  ? f.gain < 1
                    ? 'Kernel liền kề thắng trên thiết bị này, ngược với dự đoán của bạn và với lập luận của sách.'
                    : 'Kernel xen kẽ thắng trên thiết bị này, ngược với dự đoán của bạn.'
                  : predicted > f.gain * 2
                    ? 'Bạn đã kỳ vọng nhiều hơn đáng kể. Số segment là thứ bộ nhớ phải cung cấp; các bộ nhớ đệm giữ dòng cache của một luồng giữa các lần lặp của nó lấy lại được một phần.'
                    : predicted < f.gain / 2
                      ? 'Bạn đã kỳ vọng ít hơn đáng kể.'
                      : 'Con số đó gần với dự đoán của bạn.'}
                {f.b === 'metal'
                  ? ' Apple không ghi trong tài liệu quy tắc hợp nhất truy cập nào (ch09-facts C4), nên trên Mac số đo này là bằng chứng, và nó có thể nghiêng về bất kỳ phía nào.'
                  : f.gain < 1
                    ? ' Các bộ nhớ đệm giữ dòng cache của mỗi luồng giữa các lần lặp có thể bù lại các segment thừa.'
                    : ''}
              </p>
            }
          >
            <p key={f.b} data-gain={f.gain} data-direction={opposite ? 'opposite' : 'same'}>
              On <b>{f.device}</b> ({f.b}), at factor 16 on uniform bytes, the interleaved kernel took {fmt(f.inter * 1e3)} ms and
              the contiguous one {fmt(f.contig * 1e3)} ms: interleaved ran <b>{fmt(f.gain)}×</b> as fast.{' '}
              {opposite
                ? f.gain < 1
                  ? 'Contiguous won on this device, the opposite of your prediction and of the book’s argument.'
                  : 'Interleaved won on this device, the opposite of your prediction.'
                : predicted > f.gain * 2
                  ? 'You expected considerably more. The segment count is what memory must deliver; caches that keep a thread’s line between its iterations claw some of it back.'
                  : predicted < f.gain / 2
                    ? 'You expected considerably less.'
                    : 'That is close to what you predicted.'}
              {f.b === 'metal'
                ? ' Apple documents no coalescing rule (ch09-facts C4), so on a Mac this measurement is the evidence, and it can go either way.'
                : f.gain < 1
                  ? ' Caches that keep each thread’s line between its iterations can make up for the extra segments.'
                  : ''}
            </p>
          </Translated>
        );
      })}
      <Translated vi={<p className="muted">Đây là số đo từ máy của bạn, không phải giá trị kỳ vọng.</p>}>
        <p className="muted">These are measurements from your machine, not expected values.</p>
      </Translated>
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS95: Record<string, Question> = {
  'ch09/9.5/q-index-interleaved': {
    kind: 'numeric',
    id: 'ch09/9.5/q-index-interleaved',
    prompt: (
      <Translated
        vi={
          <>
            Chia xen kẽ (Hình 9.14), <code>blockDim.x = 256</code>, <code>COARSE_FACTOR = 4</code>. Luồng 5 của khối 2
            đọc phần tử nào ở lần lặp <code>c = 3</code>?
          </>
        }
      >
        <>
          Interleaved partitioning (Fig. 9.14), <code>blockDim.x = 256</code>, <code>COARSE_FACTOR = 4</code>. Which
          element does thread 5 of block 2 read in iteration <code>c = 3</code>?
        </>
      </Translated>
    ),
    answer: 2821,
    explain: (
      <Translated
        vi={
          <p>
            segment = COARSE_FACTOR·blockIdx.x·blockDim.x = 4·2·256 = 2048. Ở lần lặp 3, khối phủ dải thứ tư gồm
            256 phần tử, bắt đầu từ 2048 + 3·256 = 2816, và luồng 5 lấy phần tử thứ sáu trong số đó: 2821.
          </p>
        }
      >
        <p>
          segment = COARSE_FACTOR·blockIdx.x·blockDim.x = 4·2·256 = 2048. In iteration 3 the block covers the fourth run of
          256 elements, starting at 2048 + 3·256 = 2816, and thread 5 takes the sixth of them: 2821.
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.5, pp. 216–217 (Fig. 9.14, line 12)',
    vi: { source: 'ch. 9 §9.5, pp. 216–217 (Hình 9.14, dòng 12)' },
  },
  'ch09/9.5/q-index-contiguous': {
    kind: 'numeric',
    id: 'ch09/9.5/q-index-contiguous',
    prompt: (
      <Translated
        vi={
          <>
            Cùng luồng và lần lặp (khối 2, luồng 5, <code>c = 3</code>, <code>blockDim.x = 256</code>,{' '}
            <code>COARSE_FACTOR = 4</code>), nhưng chia liền kề (Hình 9.12). Phần tử nào?
          </>
        }
      >
        <>
          Same thread and iteration (block 2, thread 5, <code>c = 3</code>, <code>blockDim.x = 256</code>,{' '}
          <code>COARSE_FACTOR = 4</code>), but with contiguous partitioning (Fig. 9.12). Which element?
        </>
      </Translated>
    ),
    answer: 2071,
    explain: (
      <Translated
        vi={
          <p>
            Cùng segment, 2048. Dải riêng của luồng 5 bắt đầu sau 5·4 = 20 phần tử, và lần lặp 3 là phần tử thứ tư của
            dải: 2048 + 20 + 3 = 2071. Luồng 6 đọc 2075 trong cùng lần lặp: các lane kế nhau cách nhau COARSE_FACTOR
            phần tử, đó chính là toàn bộ vấn đề của thứ tự này trên GPU.
          </p>
        }
      >
        <p>
          Same segment, 2048. Thread 5's own run starts 5·4 = 20 elements in, and iteration 3 is the fourth element of
          the run: 2048 + 20 + 3 = 2071. Thread 6 reads 2075 in the same iteration: neighbouring lanes are COARSE_FACTOR
          elements apart, which is the whole problem with this order on a GPU.
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.5, p. 215 (Fig. 9.12, line 12)',
    vi: { source: 'ch. 9 §9.5, p. 215 (Hình 9.12, dòng 12)' },
  },
  'ch09/9.5/q-ex5c': {
    kind: 'numeric',
    id: 'ch09/9.5/q-ex5c',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 5c: 524,288 phần tử, 128 bin, 1024 luồng mỗi khối, kernel gộp việc của Hình 9.14 với
            hệ số 4. Số phép toán nguyên tử trên bộ nhớ toàn cục lớn nhất nó có thể phát ra là bao nhiêu?
          </>
        }
      >
        <>
          Exercise 5c: 524,288 elements, 128 bins, 1024 threads per block, the coarsened kernel of Fig. 9.14 with a
          factor of 4. What is the largest number of atomic operations on global memory it can issue?
        </>
      </Translated>
    ),
    answer: 16384,
    explain: (
      <Translated
        vi={
          <p>
            524,288 / (1024·4) = 128 khối. Phép toán nguyên tử của mỗi điểm ảnh đi vào bản sao chia sẻ của khối; bộ nhớ toàn cục chỉ thấy các
            lần ghi, nhiều nhất một cho mỗi bin khác không của mỗi khối: 128·128 = 16,384. Không gộp việc thì là 512·128 = 65,536
            (5b), và không riêng tư hóa thì một cho mỗi phần tử, 524,288 (5a).
          </p>
        }
      >
        <p>
          524,288 / (1024·4) = 128 blocks. Every pixel's atomic goes to the block's shared copy; global memory sees only the
          commits, at most one per non-zero bin per block: 128·128 = 16,384. Without coarsening it was 512·128 = 65,536
          (5b), and without privatisation one per element, 524,288 (5a).
        </p>
      </Translated>
    ),
    source: 'ch. 9 exercise 5, p. 220 (the answer is ours; the book prints none)',
    vi: { source: 'ch. 9 bài tập 5, p. 220 (đáp án của khóa học này; sách không in đáp án)' },
  },
  'ch09/9.5/q-segments': {
    kind: 'numeric',
    id: 'ch09/9.5/q-segments',
    prompt: (
      <Translated
        vi={
          <>
            Điểm ảnh một byte, chia liền kề với <code>COARSE_FACTOR = 32</code>, một warp 32 lane và các
            transaction 32 byte của NVIDIA. Một lần nạp toàn warp chạm vào bao nhiêu segment?
          </>
        }
      >
        <>
          One-byte pixels, contiguous partitioning with <code>COARSE_FACTOR = 32</code>, a 32-lane warp and NVIDIA's
          32-byte transactions. How many segments does one warp-wide load touch?
        </>
      </Translated>
    ),
    answer: 32,
    explain: (
      <Translated
        vi={
          <p>
            Các lane kế nhau cách nhau 32 phần tử, ở đây là 32 byte, nên byte của mỗi lane nằm trong một segment 32 byte
            khác nhau: 32 transaction để cung cấp 32 byte hữu ích, điều mà NVIDIA ghi trong tài liệu là trường hợp tệ nhất cho một lần nạp.
            Thứ tự xen kẽ đọc 32 byte liên tiếp: một segment.
          </p>
        }
      >
        <p>
          Neighbouring lanes are 32 elements, here 32 bytes, apart, so every lane's byte sits in a different 32-byte
          segment: 32 transactions to deliver 32 useful bytes, which NVIDIA documents as the worst case for one load.
          The interleaved order reads 32 consecutive bytes: one segment.
        </p>
      </Translated>
    ),
    source: 'our arithmetic on ch. 9 §9.5, p. 216, with notes/today/ch09-facts.md A7',
    vi: { source: 'phép tính của khóa học này dựa trên ch. 9 §9.5, p. 216, cùng notes/today/ch09-facts.md A7' },
  },
  'ch09/9.5/q-cpu-gpu': {
    kind: 'choice',
    id: 'ch09/9.5/q-cpu-gpu',
    prompt: (
      <Translated vi={<>Vì sao chia liền kề hợp với CPU mà không hợp với GPU?</>}>
        <>Why does contiguous partitioning suit a CPU and not a GPU?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Chia liền kề đọc một số điểm ảnh hai lần ở ranh giới các dải, và bộ nhớ đệm lớn của CPU che đi các lần lặp lại.">
            Contiguous partitioning reads some pixels twice at the run boundaries, and a CPU’s large cache hides the repeats.
          </Translated>
        ),
        why: (
          <Translated vi="Cả hai thứ tự đều đọc mỗi điểm ảnh đúng một lần; tổng của trình khám phá đếm cùng số byte cần cho cả hai.">
            Both orders read every pixel exactly once; the explorer’s totals count the same bytes needed for both.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="GPU không có bộ nhớ đệm dữ liệu, nên mọi truy cập đều đi tới DRAM, nơi phục vụ một dải byte liền không nhanh hơn các byte rải rác.">
            A GPU has no data caches, so every access goes to DRAM, which serves a run of bytes no faster than scattered ones.
          </Translated>
        ),
        why: (
          <Translated vi="GPU có bộ nhớ đệm. Điểm mấu chốt là với hàng nghìn luồng đang chạy, dòng cache của một luồng cụ thể khó mà ở lại trong đó.">
            GPUs have caches. The point is that with thousands of threads in flight, a given thread’s line is unlikely to stay in one.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Dòng cache của một luồng CPU ở lại trong bộ nhớ đệm cho đến điểm ảnh tiếp theo của nó; trên GPU hiếm khi như vậy, nên một warp phải đọc các byte kế nhau cùng lúc.">
            A CPU thread’s line stays cached until its next pixel; on a GPU it seldom does, so a warp must read neighbouring bytes together.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chia liền kề cần một luồng cho mỗi dải điểm ảnh, nhiều luồng hơn chia xen kẽ, và CPU có ít luồng hơn nhiều.">
            Contiguous partitioning needs a thread per run of pixels, more threads than interleaved, and a CPU has far fewer.
          </Translated>
        ),
        why: (
          <Translated vi="Cả hai thứ tự dùng cùng số luồng; chỉ dòng 12 khác nhau.">
            Both orders use the same threads; only line 12 differs.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mẫu truy cập thân thiện với một luồng đi dọc bộ nhớ theo thời gian lại bất lợi cho ba mươi hai lane
            nạp cùng một thời điểm. Trên GPU, đơn vị quan trọng là lần nạp duy nhất của warp, và nó rẻ khi
            địa chỉ của các lane nằm kề nhau.
          </p>
        }
      >
        <p>
          The access pattern that is friendly to one thread walking memory over time is hostile to thirty-two lanes
          loading at the same instant. On the GPU, the unit that matters is the warp's single load, and it is cheap when
          its lanes' addresses are adjacent.
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.5, p. 216; ch. 6 §6.1', // i18n-ignore: citation only
  },
  'ch09/9.5/q-overcoarsen': {
    kind: 'choice',
    id: 'ch09/9.5/q-overcoarsen',
    prompt: (
      <Translated
        vi={
          <>
            2<sup>20</sup> điểm ảnh, khối 1024 luồng và <code>COARSE_FACTOR = 1024</code>, nên cả ảnh chỉ là một
            khối. Ý hay chứ?
          </>
        }
      >
        <>
          2<sup>20</sup> pixels, 1024-thread blocks and <code>COARSE_FACTOR = 1024</code>, so the whole image is one
          block. Good idea?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Có: một bản sao riêng nghĩa là ít việc xóa và ghi nhất có thể, đó chính là mục đích của gộp việc.">
            Yes: one private copy means the least zeroing and committing possible, which is what coarsening is for.
          </Translated>
        ),
        why: (
          <Translated vi="Đúng về chi phí phụ, nhưng không liên quan: thiết bị giờ gần như không có gì để chạy.">
            True of the overhead, and irrelevant: the device now has almost nothing to run.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không: một khối chạy trên một SM và phần còn lại của thiết bị ngồi không; hãy giữ số khối xấp xỉ số khối chạy được cùng lúc.">
            No: one block runs on one SM and the rest of the device idles; keep about as many blocks as run at once.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không: một khối không thể có 1024 luồng trên phần cứng hiện nay, nên lần khởi chạy thất bại trước khi gộp việc kịp có ý nghĩa.">
            No: a block cannot have 1024 threads on current hardware, so the launch fails before coarsening matters.
          </Translated>
        ),
        why: (
          <Translated vi="1024 luồng mỗi khối là mức tối đa trên phần cứng hiện nay, và được phép.">
            1024 threads per block is the maximum on current hardware, and allowed.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không khác gì: gộp việc giữ nguyên tổng công việc và chỉ thay đổi cách chia nó cho các luồng.">
            It makes no difference: coarsening leaves the total work as it was and changes how it is divided among threads.
          </Translated>
        ),
        why: (
          <Translated vi="Công việc thì như nhau; mức song song phơi ra cho phần cứng thì không.">
            The work is the same; the parallelism exposed to the hardware is not.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đây là cạm bẫy thứ hai của chương 6: gộp việc quá nhiều để phần cứng ngồi không. Chi phí phụ mà gộp việc loại bỏ
            chỉ đáng loại bỏ với những khối dù sao cũng sẽ bị chạy tuần tự. Thiết bị của bạn chạy cùng lúc được bao nhiêu khối
            phụ thuộc vào thiết bị và tài nguyên của kernel; chương 6 tính điều đó.
          </p>
        }
      >
        <p>
          This is chapter 6's second pitfall: too much coarsening leaves hardware idle. The overhead coarsening removes
          is worth removing only for blocks that would have been serialised anyway. How many blocks your device runs at
          once depends on the device and on the kernel's resources; chapter 6 works it out.
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.5, p. 214; ch. 6 §6.5, p. 143', // i18n-ignore: citation only
  },
};

// ---------------------------------------------------------------- the page
export function S95() {
  return (
    <>
      <Translated
        vi={
          <p>
            Kernel của mục 9.4 làm hai việc cho mỗi khối mà chẳng liên quan gì tới điểm ảnh nào: nó xóa bản sao bin riêng
            về không trước lần cập nhật đầu tiên, và nó ghi bản sao vào histogram công khai sau lần cuối cùng. Chi phí
            phụ đó trả một lần cho mỗi khối. Nó chỉ mua được gì khi các khối thật sự chạy song song với nhau, mỗi khối giảm bớt
            cho các bin công khai phần tranh chấp của mình. Khởi chạy nhiều khối hơn số thiết bị giữ được cùng lúc thì bộ lập lịch
            sẽ chạy chúng lần lượt; mỗi khối chạy sau vẫn xóa và ghi bản sao của riêng nó, và việc đó chẳng mua được gì.
          </p>
        }
      >
        <p>
          The kernel of section 9.4 does two jobs per block that have nothing to do with any pixel: it zeroes its private
          copy of the bins before the first update, and it commits the copy to the public histogram after the last. That
          overhead is paid once per block. It buys something only while the blocks really run side by side, each sparing
          the public bins its share of the contention. Launch more blocks than the device holds at once and the scheduler
          runs them in turns; every later block still zeroes and commits its own copy, and that work buys nothing.
        </p>
      </Translated>
      <Source section="9.5" pages="214" />
      <Translated
        vi={
          <p>
            Cách chữa là cái mà <a href={href('ch06', '6.5')}>chương 6</a> gọi là gộp việc cho luồng (thread coarsening): khởi chạy ít
            khối hơn và giao cho mỗi luồng nhiều điểm ảnh, để mỗi bản sao riêng phục vụ được nhiều phần của ảnh hơn. Sách gọi
            số điểm ảnh mỗi luồng là <code>COARSE_FACTOR</code>.
          </p>
        }
      >
        <p>
          The remedy is the one <a href={href('ch06', '6.5')}>chapter 6</a> called thread coarsening: launch fewer
          blocks and give each thread several pixels, so that each private copy serves more of the image. The book calls
          the number of pixels per thread <code>COARSE_FACTOR</code>.
        </p>
      </Translated>

      <h2><Translated vi="Gộp việc tiết kiệm được gì">What coarsening saves</Translated></h2>
      <Translated
        vi={
          <p>
            Hãy đếm trước khi đo. Bài tập 5 ở cuối chương so sánh ba kernel trên cùng một đầu vào: không
            riêng tư hóa (Hình 9.6), một bản sao riêng mỗi khối (Hình 9.10), và cùng kernel đó với mỗi luồng xử lý{' '}
            <code>COARSE_FACTOR</code> điểm ảnh (Hình 9.14). Hãy giải phần cuối của nó, rồi kiểm tra trong máy tính bên dưới
            và xem các con số thay đổi thế nào.
          </p>
        }
      >
        <p>
          Count it before you measure it. Exercise 5 at the end of the chapter compares three kernels on one input: no
          privatisation (Fig. 9.6), a private copy per block (Fig. 9.10), and the same with each thread handling{' '}
          <code>COARSE_FACTOR</code> pixels (Fig. 9.14). Work out its last part, then check it in the calculator below
          and see how the counts move.
        </p>
      </Translated>
      <Source pages="220" figs={tr('exercise 5', 'bài tập 5')} />
      <Ask q={questionsS95['ch09/9.5/q-ex5c']} label={tr('Exercise 5c', 'Bài tập 5c')} />
      <Translated
        vi={
          <p>
            Máy tính đếm công việc của ba kernel cho bất kỳ đầu vào nào. Preset bài tập 5 tái hiện câu hỏi;
            preset thứ hai là một ảnh cỡ một tấm ảnh chụp lớn.
          </p>
        }
      >
        <p>
          The calculator counts the three kernels' work for any input. The exercise 5 preset reproduces the question;
          the second preset is an image the size of a large photo.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Số lượng, không phải thời gian. Cột cuối là một cận trên: nó giả định mọi bin mà một khối có thể lấp đầy đều khác không, trong khi phép kiểm tra
              bằng không làm con số thật nhỏ hơn với hầu hết các ảnh.
            </p>
          }
        >
          <p>
            Counts, not time. The last column is a bound: it assumes every bin a block can fill is non-zero, where the zero
            test makes the real number smaller for most images.
          </p>
        </Translated>
      </Simplification>
      <PlanCalculator />
      <Translated
        vi={
          <p>
            Nhìn lại cột phép toán nguyên tử phạm vi khối: một cho mỗi điểm ảnh, bất kể hệ số. Gộp việc thu nhỏ
            chi phí mỗi khối, không phải chi phí mỗi điểm ảnh. Mục 9.6 nhắm vào các phép toán nguyên tử theo từng điểm ảnh.
          </p>
        }
      >
        <p>
          Look at the column of block-scope atomics again: one per pixel, whatever the factor. Coarsening shrinks the
          cost per block, not the cost per pixel. Section 9.6 goes after the per-pixel atomics.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Nên gộp việc tới mức nào? Cho đến khi lần khởi chạy có xấp xỉ bằng số khối mà thiết bị chạy được cùng lúc, và không
            hơn. Con số đó thuộc về thiết bị và kernel của bạn, và{' '}
            <a href={href('ch06', '6.5')}>máy tính đợt (wave) của chương 6</a> là nơi để tính nó.
          </p>
        }
      >
        <p>
          How far should you coarsen? Until the launch has about as many blocks as the device can run at once, and no
          further. That number belongs to your device and your kernel, and{' '}
          <a href={href('ch06', '6.5')}>chapter 6's wave calculator</a> is the place to work it out.
        </p>
      </Translated>
      <Ask q={questionsS95['ch09/9.5/q-overcoarsen']} />

      <h2><Translated vi="Hai cách chia điểm ảnh">Two ways to hand out pixels</Translated></h2>
      <Translated
        vi={
          <p>
            Khi một luồng xử lý nhiều điểm ảnh, bạn phải quyết định là những điểm ảnh nào. Mỗi khối vẫn lấy một đoạn của
            ảnh, dài <code>COARSE_FACTOR·blockDim.x</code> điểm ảnh, bắt đầu từ{' '}
            <code>segment = COARSE_FACTOR·blockIdx.x·blockDim.x</code>. Bên trong đoạn đó, sách so sánh hai
            cách gán, và hai listing của nó chỉ khác nhau một dòng, dòng 12:
          </p>
        }
      >
        <p>
          Once a thread handles several pixels, you must decide which ones. Each block still takes one stretch of the
          image, <code>COARSE_FACTOR·blockDim.x</code> pixels long, starting at{' '}
          <code>segment = COARSE_FACTOR·blockIdx.x·blockDim.x</code>. Inside that stretch the book compares two
          assignments, and its two listings differ in a single line, line 12:
        </p>
      </Translated>
      <Source section="9.5" pages="214–217" figs={tr('Figs. 9.11–9.14', 'Hình 9.11–9.14')} />
      <ul>
        <Translated
          vi={
            <li>
              <b>Chia liền kề</b> (Hình 9.11 và 9.12). Mỗi luồng lấy một dải gồm <code>COARSE_FACTOR</code>{' '}
              điểm ảnh kế nhau: <code>i = segment + threadIdx.x*COARSE_FACTOR + c</code>.
            </li>
          }
        >
          <li>
            <b>Contiguous partitioning</b> (Figs. 9.11 and 9.12). Each thread takes a run of <code>COARSE_FACTOR</code>{' '}
            neighbouring pixels: <code>i = segment + threadIdx.x*COARSE_FACTOR + c</code>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Chia xen kẽ</b> (Hình 9.13 và 9.14). Ở lần lặp <code>c</code>, cả khối lấy{' '}
              <code>blockDim.x</code> điểm ảnh tiếp theo, mỗi luồng một điểm: <code>i = segment + c*blockDim.x + threadIdx.x</code>.
            </li>
          }
        >
          <li>
            <b>Interleaved partitioning</b> (Figs. 9.13 and 9.14). In iteration <code>c</code> the whole block takes the
            next <code>blockDim.x</code> pixels, one per thread: <code>i = segment + c*blockDim.x + threadIdx.x</code>.
          </li>
        </Translated>
      </ul>
      <Translated
        vi={
          <p>
            Cả hai đều đọc mỗi điểm ảnh đúng một lần. Điều khác nhau là một warp yêu cầu bộ nhớ những gì trong một bước, và đó là thứ
            trình khám phá đếm. Bắt đầu với hình vẽ của sách: tám luồng và hệ số 3. Bước từ lần lặp 0 sang
            lần lặp 1 ở mỗi thứ tự và xem các ô được tô cách nhau bao xa. Ở quy mô này, hai thứ tự vẫn tốn
            như nhau: cả dải chỉ 24 byte, nằm trong một segment 32 byte, nên mọi lần nạp đều lấy đúng segment đó. Sự
            khác biệt xuất hiện với preset thứ hai: chọn nó và kéo hệ số từ 1 lên 64.
          </p>
        }
      >
        <p>
          Both read every pixel exactly once. What differs is what one warp asks memory for in one step, and that is what
          the explorer counts. Start with the book's drawings: eight threads and a factor of 3. Step from iteration 0 to
          iteration 1 in each order and watch how far apart the shaded cells are. At this scale the two orders still cost
          the same: the whole strip is 24 bytes, inside one 32-byte segment, so every load fetches that one segment. The
          difference appears with the second preset: take it and drag the factor from 1 to 64.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Hai chỗ sai trong sách: câu "four threads access characters 8 through 15" ở p. 216 ý chỉ tám luồng và các điểm ảnh
              của Hình 9.13, và câu "In Fig. 9.11" ở p. 215 mô tả mã của Hình 9.12.
            </p>
          }
        >
          <p>
            Two slips in the book: p. 216's "four threads access characters 8 through 15" means Fig. 9.13's eight threads
            and pixels, and p. 215's "In Fig. 9.11" describes the code of Fig. 9.12.
          </p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Một warp là thứ quan trọng vì các lane của nó phát lệnh nạp cùng nhau. Trình khám phá cho thấy các lần đọc của warp được chọn
            ở lần lặp được chọn, các segment bộ nhớ mà chúng rơi vào, và tổng cho cả kernel ở cả hai thứ tự.
            Các menu nhà cung cấp thay đổi độ rộng warp và kích thước segment; kích thước phần tử chuyển sang token id.
          </p>
        }
      >
        <p>
          One warp is what matters because its lanes issue the load together. The explorer shows the chosen warp's reads
          in the chosen iteration, the memory segments they fall in, and the totals for the whole kernel in both orders.
          The vendor menus change the warp width and the segment size; the element size switches to token ids.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Segment căn chỉnh cho mỗi lần nạp toàn warp: không có thời gian, không có bộ nhớ đệm, và ảnh bắt đầu ở một ranh giới segment. Apple
              không ghi trong tài liệu kích thước segment nào, nên với Metal kích thước là do bạn chọn.
            </p>
          }
        >
          <p>
            Aligned segments per warp-wide load: no time, no caches, and the image starts on a segment boundary. Apple
            documents no segment size, so for Metal the size is your choice.
          </p>
        </Translated>
      </Simplification>
      <ReadsExplorer />
      <Ask q={questionsS95['ch09/9.5/q-index-interleaved']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS95['ch09/9.5/q-index-contiguous']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS95['ch09/9.5/q-segments']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Vì sao câu trả lời của CPU sai ở đây">Why the CPU answer is wrong here</Translated></h2>
      <Translated
        vi={
          <p>
            Trên CPU, chia liền kề thường là lựa chọn đúng. Một nhúm luồng, mỗi luồng đi dọc đoạn bộ nhớ của riêng mình
            theo thứ tự; dòng cache mà một luồng đã nạp cho một điểm ảnh vẫn còn đó cho điểm tiếp theo, và các luồng
            hiếm khi tranh chấp cùng các dòng cache.
          </p>
        }
      >
        <p>
          On a CPU, contiguous partitioning is usually the right choice. A handful of threads each walk their own stretch
          of memory in order; the cache line a thread loaded for one pixel is still there for the next, and the threads
          seldom contend for the same lines.
        </p>
      </Translated>
      <Source pages="216" />
      <Translated
        vi={
          <p>
            SM của GPU giữ quá nhiều luồng đang chạy đến mức dòng cache của một luồng khó mà sống sót trong bộ nhớ đệm cho đến khi
            luồng đó quay lại lấy điểm ảnh tiếp theo. Điều quyết định chi phí là thời điểm một warp thực thi lệnh nạp: các lane của nó
            phát lệnh cùng nhau và hệ thống bộ nhớ phục vụ chúng theo từng segment nguyên. Các lane đọc byte kế nhau dùng chung
            segment; các lane cách nhau <code>COARSE_FACTOR</code> byte cần mỗi lane một segment khi khoảng cách đạt tới một segment. Đây là
            quy tắc hợp nhất truy cập bộ nhớ (coalescing) của <a href={href('ch06', '6.1')}>chương 6, mục 6.1</a>, áp dụng cho một vòng lặp. NVIDIA và AMD
            ghi quy tắc đó trong tài liệu; Apple thì không, nên trên Mac số đo của lab 9.2 có thể nghiêng về bất kỳ phía nào.
          </p>
        }
      >
        <p>
          A GPU's SM keeps so many threads in flight that one thread's line is unlikely to survive in the cache until that
          thread comes back for its next pixel. What decides the cost is the moment a warp executes the load: its lanes
          issue it together and the memory system serves them in whole segments. Lanes that read neighbouring bytes share
          segments; lanes <code>COARSE_FACTOR</code> bytes apart need one each once the gap reaches a segment. This is the
          coalescing rule of <a href={href('ch06', '6.1')}>chapter 6, section 6.1</a>, applied to a loop. NVIDIA and AMD
          document that rule; Apple documents none, so on a Mac lab 9.2's measurement may go either way.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Sách thêm một gợi ý không kèm mã: mỗi luồng nên đọc bốn điểm ảnh, một từ 32-bit, mỗi lần lặp, để dùng hết
            băng thông giữa các bộ nhớ đệm và các SM.
          </p>
        }
      >
        <p>
          The book adds a hint without code: each thread should read four pixels, one 32-bit word, per iteration, to use
          the bandwidth between the caches and the SMs fully.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch09-facts.md A7, B3, C4; ch05-facts.md B5)', '2026-11-20 (notes/today/ch09-facts.md, mục A7, B3, C4; ch05-facts.md B5)')}>
        <Translated
          vi={
            <p>
              NVIDIA phục vụ lần nạp toàn cục của một warp bằng các transaction 32 byte; một lần nạp toàn warp có thể cần 32 transaction trong
              trường hợp tệ nhất. AMD gộp các lane liên tiếp thành các yêu cầu dòng cache 64 hoặc 128 byte, và warp của AMD có thể rộng 64
              lane. Apple không ghi trong tài liệu quy tắc hợp nhất truy cập nào: trên Mac, benchmark là bằng chứng duy nhất.
            </p>
          }
        >
          <p>
            NVIDIA serves a warp's global load in 32-byte transactions; one warp-wide load can need 32 of them in the
            worst case. AMD merges consecutive lanes into 64- or 128-byte cache-line requests, and its warps may be 64
            lanes wide. Apple documents no coalescing rule: on a Mac the benchmark is the only evidence.
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              Gợi ý bốn-điểm-ảnh-mỗi-lần-lặp không có trong tài liệu hiện hành nào của NVIDIA. Histogram của CUB thì có nạp nhiều mẫu
              mỗi luồng dưới dạng một vector.
            </p>
          }
        >
          <p>
            The four-pixels-per-iteration hint is in no current NVIDIA document. CUB's histogram does load several samples
            per thread as one vector.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS95['ch09/9.5/q-cpu-gpu']} />

      <h2><Translated vi="Mã nguồn">The code</Translated></h2>
      <Translated
        vi={
          <p>
            Kernel xen kẽ là kernel mà lab 9.2 yêu cầu bạn viết; lab cũng cung cấp bản liền kề, để
            bạn có thể đo thời gian cả hai. Trong các lab, hệ số gộp việc là một đối số lúc chạy, trong khi sách dùng một
            hằng số lúc biên dịch, để một bản build có thể quét qua nhiều giá trị.
          </p>
        }
      >
        <p>
          The interleaved kernel is the one lab 9.2 asks you to write; the lab also provides the contiguous version, so
          that you can time the two. In the labs the coarsening factor is a run-time argument, where the book uses a
          compile-time constant, so that one build can sweep it.
        </p>
      </Translated>
      <Source pages="215, 217" figs={tr('Figs. 9.12, 9.14', 'Hình 9.12, 9.14')} />
      <Listing />

      <h2><Translated vi="Đo nó">Measure it</Translated></h2>
      <Translated
        vi={
          <p>
            Hợp nhất truy cập là một khẳng định về phần cứng, nên hãy kiểm tra nó trên phần cứng của bạn. Lab 9.2 đo kernel liền kề được cung cấp so với
            kernel xen kẽ của bạn ở hệ số 16, trên các byte ngẫu nhiên phân bố đều. Hãy chốt một con số trước.
          </p>
        }
      >
        <p>
          Coalescing is a claim about hardware, so test it on yours. Lab 9.2 times the provided contiguous kernel against
          your interleaved one at a factor of 16, on uniform random bytes. Commit to a number first.
        </p>
      </Translated>
      <Predict
        id="ch09/9.5/p-interleaved-gain"
        prompt={
          <Translated
            vi={
              <>
                Ở <code>COARSE_FACTOR = 16</code> trên các byte ngẫu nhiên phân bố đều, bạn kỳ vọng kernel xen kẽ chạy nhanh hơn kernel liền kề
                bao nhiêu lần trên GPU <em>của bạn</em>?
              </>
            }
          >
            <>
              At <code>COARSE_FACTOR = 16</code> on uniform random bytes, how many times faster than the contiguous kernel
              do you expect the interleaved kernel to run on <em>your</em> GPU?
            </>
          </Translated>
        }
        unit={tr('× faster', '× nhanh hơn')}
      >
        {(v) => <InterleavedGain predicted={v} />}
      </Predict>
      <Translated
        vi={
          <p>
            <a href={href('ch09', 'labs')}>Trang lab</a> có mọi thứ lab 9.2 đã đo.
          </p>
        }
      >
        <p>
          The <a href={href('ch09', 'labs')}>labs page</a> has everything lab 9.2 measured.
        </p>
      </Translated>

      <h2><Translated vi="Nơi bạn sẽ gặp nó">Where you will meet it</Translated></h2>
      <Translated
        vi={
          <p>
            Histogram CUDA của PyTorch (<code>torch.histc</code>, <code>torch.bincount</code>) riêng tư hóa trong bộ nhớ chia sẻ, như
            ở mục 9.4, mỗi khi các bin vừa, và duyệt đầu vào bằng <code>FOR_KERNEL_LOOP</code>, một vòng lặp grid-stride: mỗi lượt của
            cả lưới đọc <code>gridDim.x·blockDim.x</code> phần tử liên tiếp tiếp theo, đó là chia
            xen kẽ với bước nhảy là cả lưới. Nó định kích thước lưới bằng một heuristic tên là{' '}
            <code>RATIO_OF_GMEM_ATOMIC_ADD_TO_SMEM_ATOMIC_ADD</code>, cân bằng số lần ghi (kích thước lưới nhân số bin)
            với số phép toán nguyên tử trên bộ nhớ chia sẻ mỗi SM: chính sự đánh đổi của mục này, viết thành mã. Bản MPS giới hạn
            lưới của nó ở 128 threadgroup, một giới hạn mà các tác giả chọn bằng cách đo.
          </p>
        }
      >
        <p>
          PyTorch's CUDA histogram (<code>torch.histc</code>, <code>torch.bincount</code>) privatises in shared memory, as
          in section 9.4, whenever the bins fit, and walks the input with <code>FOR_KERNEL_LOOP</code>, a grid-stride loop: each pass of the
          whole grid reads the next <code>gridDim.x·blockDim.x</code> consecutive elements, which is interleaved
          partitioning with the grid as the stride. It sizes the grid with a heuristic named{' '}
          <code>RATIO_OF_GMEM_ATOMIC_ADD_TO_SMEM_ATOMIC_ADD</code>, which balances the commits (grid size times bins)
          against the shared-memory atomics per SM: this section's trade-off, written as code. The MPS version caps its
          grid at 128 threadgroups, a limit its authors chose by measuring.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách không bàn về PyTorch. Đọc vào 2026-11-20: <code>aten/src/ATen/native/cuda/SummaryOps.cu</code>{' '}
              và các tệp <code>HistogramKernel</code> của MPS (notes/today/ch09-facts.md D1, D2).
            </p>
          }
        >
          <p>
            The book does not discuss PyTorch. Read on 2026-11-20: <code>aten/src/ATen/native/cuda/SummaryOps.cu</code>{' '}
            and the MPS <code>HistogramKernel</code> files (notes/today/ch09-facts.md D1, D2).
          </p>
        </Translated>
      </Beyond>
    </>
  );
}
