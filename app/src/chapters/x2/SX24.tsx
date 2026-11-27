import { useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  LAYOUTS, LAYOUT_IDS, SEGMENT_MODELS, iterationPlan, iterationSpan, loadStep, loadsPerLane, problem, rowSteps,
  segmentBytesOf, type LaneLoad, type LayoutId, type LoadOrder, type Mapping, type SegmentModelId, type Step,
} from '../../sim/qlayout';
import { Translated, tr } from '../../lib/i18n';

// Section X2.4 (ours): lab 22.1's gather GEMV with quantised rows. A stepper over one warp's
// loads (qlayout.ts, which counts segments with coalesce.ts), a Predict on lab X2.1's mapping
// that sets the stepper, an explorer of idle lanes against row length for 32 and 64 lanes,
// and lab X2.2's reference solution loaded from labs/x2/lab2_q4gemv/solution/.

const FACTS = 'notes/today/x2-facts.md';
const LAB_X22 = 'x2/lab2_q4gemv';
const loaded = (t: string | null) => (t && !t.startsWith('// could not load') ? t : null);

export const questionsSX24: Record<string, Question> = {
  'x2/X2.4/q-lane-per-block': {
    kind: 'numeric',
    id: 'x2/X2.4/q-lane-per-block',
    prompt: (
      <Translated
        vi={
          <>
            Các khối Q4_0 18 byte, một lane cho mỗi khối, 32 lane, segment 32 byte. Lần nạp 2 byte đầu tiên của mỗi lane vào
            các mã của nó đọc byte 2 và 3 của khối. Lệnh nạp đó chạm vào bao nhiêu segment?
          </>
        }
      >
        <>
          Q4_0 blocks of 18 bytes, one lane per block, 32 lanes, 32-byte segments. Each lane's first 2-byte load of its
          codes reads bytes 2 and 3 of its block. How many segments does that one load instruction touch?
        </>
      </Translated>
    ),
    answer: 18,
    unit: 'segments',
    explain: (
      <Translated
        vi={
          <p>
            Khối của lane 31 bắt đầu ở 31 × 18 = 558, nên các byte của các lane chạy từ 2 đến 561, và với các lane cách nhau 18
            byte thì không segment 32 byte nào ở giữa bị bỏ qua: 18 segment, nạp 576 byte để dùng 64 byte, khoảng 11 %. Lệnh nạp
            tiếp theo lại nạp hầu hết các segment đó một lần nữa, và chỉ bộ nhớ đệm (cache) mới cứu được điều đó.
          </p>
        }
      >
        <p>
          Lane 31's block starts at 31 × 18 = 558, so the lanes' bytes run from 2 to 561, and with lanes 18 bytes apart no
          32-byte segment in between is skipped: 18 segments, 576 bytes fetched for 64 bytes used, about 11 %. The next
          load instruction fetches most of the same segments again, and only a cache can save that.
        </p>
      </Translated>
    ),
    source: `ch. 6 §6.1, pp. 124–132 (the segment rule); ${FACTS} D1`,
    vi: { unit: 'segment', source: `ch. 6 §6.1, pp. 124–132 (quy tắc segment); ${FACTS} D1` },
  },
  'x2/X2.4/q-four-lanes': {
    kind: 'choice',
    id: 'x2/X2.4/q-four-lanes',
    prompt: (
      <Translated vi={<>Cùng hàng đó với bốn lane cho mỗi khối, mỗi lane nạp 2 byte tiếp theo của nó. Ít hay nhiều segment hơn trên mỗi byte hữu ích, và vì sao?</>}>
        <>The same row with four lanes per block, each lane loading its next 2 bytes. Fewer or more segments per useful byte, and why?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Ít hơn: giờ các lane kề nhau đọc các byte kề nhau của một khối.">{'Fewer: neighbouring lanes now read neighbouring bytes of a block.'}</Translated>, correct: true },
      { label: <Translated vi="Nhiều hơn: bốn lane đọc một khối xung đột với nhau.">{'More: four lanes reading one block conflict with each other.'}</Translated>, why: <Translated vi="Các lane đọc các byte kề nhau của một khối chính là điều hợp nhất truy cập bộ nhớ (coalescing) cần.">{'Lanes reading neighbouring bytes of one block is what coalescing wants.'}</Translated> },
      { label: <Translated vi="Như nhau: 32 lane vẫn nạp tổng cộng 64 byte hữu ích.">{'The same: 32 lanes still load 64 useful bytes in total.'}</Translated>, why: <Translated vi="Số byte hữu ích như nhau; còn chúng rơi vào bao nhiêu segment thì không.">{'The useful bytes are the same; how many segments they fall in is not.'}</Translated> },
      { label: <Translated vi="Ít hơn, vì giờ mỗi lane nạp 4 byte trong một lệnh thay vì 2.">{'Fewer, because each lane now loads 4 bytes in one instruction instead of 2.'}</Translated>, why: <Translated vi="Độ rộng lần nạp vẫn là 2 byte. Điều thay đổi là các lane kề nhau đọc những byte nào.">{'The load width is still 2 bytes. What changed is which bytes neighbouring lanes read.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Với một lane cho mỗi khối, 64 byte của warp nằm rải rác trên 32 khối. Với bốn lane cho mỗi khối, chúng là 8 đoạn 8
            byte, trên 8 khối liên tiếp: 144 byte, 5 segment cho cùng 64 byte hữu ích. Giờ mỗi lane làm một phần tư khối, và tổng
            riêng phần của nó được nhân với d của khối đó một lần.
          </p>
        }
      >
        <p>
          With one lane per block the warp's 64 bytes are scattered over 32 blocks. With four lanes per block they are 8
          runs of 8 bytes, over 8 consecutive blocks: 144 bytes, 5 segments for the same 64 useful bytes. Each lane now does a quarter of a block, and its partial sum is
          scaled once by that block's d.
        </p>
      </Translated>
    ),
    source: `ch. 6 §6.1, pp. 124–132; ${FACTS} A5, D3`, // i18n-ignore: a citation
  },
  'x2/X2.4/q-scale': {
    kind: 'choice',
    id: 'x2/X2.4/q-scale',
    prompt: (
      <Translated vi={<>Vì sao kernel nhân tổng riêng phần của một lane với hệ số tỉ lệ d của khối một lần, thay vì nhân từng tích với d?</>}>
        <>Why does the kernel multiply a lane's partial sum by its block's scale d once, instead of multiplying every product by d?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="d được dùng chung, nên một phép nhân thay cho một phép nhân mỗi trọng số.">{'d is shared, so one multiply replaces one per weight.'}</Translated>, correct: true },
      { label: <Translated vi="Nhân từng tích với d sẽ làm tràn tổng float.">{'Multiplying each product by d would overflow the float sum.'}</Translated>, why: <Translated vi="Ở đây các tích là float; không có gì tràn. Lý do là số phép nhân.">{'The products are floats here; nothing overflows. The reason is the count of multiplies.'}</Translated> },
      { label: <Translated vi="Hệ số tỉ lệ chỉ được biết sau khi vòng lặp đã đọc khối.">{'The scale is only known after the loop has read the block.'}</Translated>, why: <Translated vi="Nó được lưu cùng khối và nạp cùng khối.">{'It is stored with the block and loaded with it.'}</Translated> },
      { label: <Translated vi="Phải như vậy, nếu không kết quả thay đổi: các phép nhân không giao hoán.">{'It must, or the result changes: the multiplies do not commute.'}</Translated>, why: <Translated vi="Hai cách cho kết quả như nhau, sai khác do làm tròn: phép nhân phân phối với phép cộng.">{'The two agree up to rounding: multiplication distributes over addition.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đưa hằng số ra ngoài tổng của khối: d · (tổng của q·x) bằng tổng của d·q·x, một phép nhân cho mỗi phần của khối thay
            vì một phép nhân mỗi trọng số. Mọi kernel thật đều làm vậy, và llama.cpp cùng MLX còn tiến thêm một bước, đưa cả độ
            dời ra ngoài thông qua tổng của các activation (mục{' '}
            <a href={href('x2', 'X2.5')}>X2.5</a>).
          </p>
        }
      >
        <p>
          Factor the constant out of the block's sum: d · (sum of q·x) equals the sum of d·q·x, one multiply per piece
          of a block instead of one per weight. Every real kernel does this, and llama.cpp and MLX go one step
          further and fold the offset out too, through the sum of the activations (section{' '}
          <a href={href('x2', 'X2.5')}>X2.5</a>).
        </p>
      </Translated>
    ),
    source: `${FACTS} D3, D4, E4`,
  },
  'x2/X2.4/q-tail': {
    kind: 'numeric',
    id: 'x2/X2.4/q-tail',
    prompt: (
      <Translated vi={<>in = 1056, tức 33 khối Q4_0 mỗi hàng. Một lane cho mỗi khối, warp 32 lane. Có bao nhiêu lane rảnh trong lần lặp cuối?</>}>
        <>in = 1056, so 33 Q4_0 blocks per row. One lane per block, 32-lane warps. How many lanes are idle in the last iteration?</>
      </Translated>
    ),
    answer: 31,
    unit: 'lanes',
    explain: (
      <Translated
        vi={
          <p>
            Hai lần lặp: 32 khối, rồi 1. Ở lần thứ hai, lane 0 qua được phép kiểm tra biên và 31 lane đi vào nhánh sai. Trên
            một wavefront 64 lane, đó là một lần lặp với 33 lane hoạt động, và lại 31 lane rảnh.
          </p>
        }
      >
        <p>
          Two iterations: 32 blocks, then 1. In the second, lane 0 passes the bounds test and 31 lanes take its false side.
          On a 64-lane wavefront it is one iteration with 33 active lanes, and again 31 idle.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.5, pp. 79–82',
    vi: { unit: 'lane' },
  },
  'x2/X2.4/q-wave64': {
    kind: 'choice',
    id: 'x2/X2.4/q-wave64',
    prompt: (
      <Translated vi={<>Với bốn lane cho mỗi khối, một lần lặp của wavefront 64 lane phủ bao nhiêu khối, và cần thay đổi gì trong mã cho điều đó?</>}>
        <>With four lanes per block, how many blocks does one iteration of a 64-lane wavefront cover, and what in the code must change for it?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="16, và không gì cả, nếu chỉ số khối được tính từ warpSize / 4 thay vì từ 8.">{'16, and nothing, if the block index is computed from warpSize / 4 rather than from 8.'}</Translated>, correct: true },
      { label: <Translated vi="8, vì một khối vẫn có bốn lane, bất kể độ rộng.">{'8, because a block still has four lanes, whatever the width.'}</Translated>, why: <Translated vi="64 lane chia thành nhóm 4 tạo ra 16 nhóm.">{'64 lanes in groups of 4 make 16 groups.'}</Translated> },
      { label: <Translated vi="16, và lane mask của mọi lệnh shuffle phải được viết lại thành một literal 64 bit.">{'16, and every shuffle\'s lane mask must be rewritten as a 64-bit literal.'}</Translated>, why: <Translated vi="HIP cần mask 64 bit trên mọi target, nhưng một mask có kiểu như ~(lane_mask_t)0 vốn đã là mask như vậy, và literal là thứ cần tránh. Số lần lặp suy ra từ warpSize.">{'HIP needs a 64-bit mask on every target, but a typed mask such as ~(lane_mask_t)0 already is one, and a literal is what to avoid. The iteration count follows from warpSize.'}</Translated> },
      { label: <Translated vi="32, vì wavefront rộng gấp đôi warp.">{'32, because the wavefront is twice as wide as a warp.'}</Translated>, why: <Translated vi="Rộng gấp đôi thì nhân đôi 8 khối của warp 32 lane, thành 16.">{'Twice as wide doubles the 8 blocks of a 32-lane warp, to 16.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hãy viết mọi phép ánh xạ theo độ rộng warp mà thiết bị báo cáo: <code>warpSize</code> trong CUDA và HIP,{' '}
            <code>threadExecutionWidth</code> của pipeline (hoặc <code>[[threads_per_simdgroup]]</code>) trong Metal. Một literal
            8 hay 32 là một lỗi chỉ lộ ra trên nhà sản xuất khác.
          </p>
        }
      >
        <p>
          Write every mapping in terms of the warp width the device reports: <code>warpSize</code> in CUDA and HIP, the
          pipeline's <code>threadExecutionWidth</code> (or <code>[[threads_per_simdgroup]]</code>) in Metal. A literal 8
          or 32 is a bug that only shows on another vendor.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch05-facts.md B5; notes/today/ch03-facts.md C9', // i18n-ignore: a citation
  },
};

// ---------------------------------------------------------------- the stepper
export interface StepperConfig {
  layout: LayoutId;
  order: LoadOrder;
  lanesPerBlock: number;
  loadBytes: number;
  warpWidth: number;
  inW: number;
  seg: SegmentModelId;
  assumed: number;
  step: number;
}
export const DEFAULT_STEPPER: StepperConfig = { layout: 'q4_0-blocks', order: 'interleaved', lanesPerBlock: 1, loadBytes: 2, warpWidth: 32, inW: 4096, seg: 'nvidia', assumed: 32, step: 0 };
/** Lab X2.1's mapping: Q8_0 codes in their own array, lane L reads code bytes 4L to 4L + 3. */
export const LAB_X21_MAPPING: StepperConfig = { ...DEFAULT_STEPPER, layout: 'q8_0-split', lanesPerBlock: 8, loadBytes: 4 };
/** Lab X2.2's mapping: four lanes per Q4_0 block, lane s reads qs bytes 4s to 4s + 3 as two 16-bit loads. */
export const LAB_X22_MAPPING: StepperConfig = { ...DEFAULT_STEPPER, order: 'piece', lanesPerBlock: 4, loadBytes: 2 };
const ROW_LENGTHS = [512, 1024, 1056, 4096];

const SW = 680;

/** One memory strip: segments, requested bytes, misaligned bytes in --fail. */
function Strip({ lanes, lo, hi, segBytes, blockLines, scaleBytes, label, kind }: {
  lanes: LaneLoad[]; lo: number; hi: number; segBytes: number; blockLines: number[]; scaleBytes: [number, number][];
  label: string; kind: 'code' | 'scale';
}) {
  const s0 = Math.floor(lo / segBytes) * segBytes;
  const s1 = Math.ceil(hi / segBytes) * segBytes;
  const n = s1 - s0;
  const bx = (a: number) => ((a - s0) / n) * SW;
  const touched = new Set<number>();
  for (const l of lanes) if (l.active) for (let a = l.addr; a < l.addr + l.bytes; a++) touched.add(Math.floor(a / segBytes));
  const segs: number[] = [];
  for (let s = s0 / segBytes; s < s1 / segBytes; s++) segs.push(s);
  return (
    <div style={{ margin: '0 0 0.5rem' }}>
    <div className="muted" style={{ fontSize: '0.8rem', fontFamily: 'var(--sans)' }}>{label}</div>
    <svg viewBox={`0 12 ${SW} 30`} preserveAspectRatio="none" /* i18n-ignore: an SVG attribute value */ style={{ width: '100%', height: 34, display: 'block' }} role="img" aria-label={label} data-strip={kind}>
      {segs.map((s) => (
        <rect key={s} x={bx(s * segBytes)} y={14} width={(segBytes / n) * SW} height={24} vectorEffect="non-scaling-stroke" data-seg={s} data-touched={touched.has(s) ? 1 : 0}
          fill={touched.has(s) ? 'var(--global-wash)' : 'var(--page)'} stroke="var(--rule)" />
      ))}
      {scaleBytes.map(([a, b], i) => (
        <rect key={'s' + i} x={bx(a)} y={30} width={Math.max(0.6, ((b - a) / n) * SW)} height={8} fill="var(--ink-3)" opacity={0.45} />
      ))}
      {blockLines.map((a) => (
        <line key={'b' + a} x1={bx(a)} x2={bx(a)} y1={12} y2={40} stroke="var(--ink-2)" strokeWidth="0.8" />
      ))}
      {lanes.filter((l) => l.active).map((l) => (
        <rect key={l.lane} x={bx(l.addr)} y={16} width={Math.max(0.8, (l.bytes / n) * SW)} height={13}
          fill={l.misaligned ? 'var(--fail)' : 'var(--global)'} data-lane={l.lane} />
      ))}
    </svg>
    <div className="muted" style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontFamily: 'var(--sans)' }}>
      <span><Translated vi={<>byte {s0}</>}>byte {s0}</Translated></span><span><Translated vi={<>{(s1 - s0) / segBytes} segment, mỗi segment {segBytes} byte</>}>{(s1 - s0) / segBytes} segments of {segBytes} bytes</Translated></span><span><Translated vi={<>byte {s1}</>}>byte {s1}</Translated></span>
    </div>
    </div>
  );
}

/** One square per lane, shaded by block, idle lanes empty, misaligned lanes in --fail. */
function Lanes({ step }: { step: Step }) {
  const n = step.codeLanes.length;
  const w = SW / n;
  return (
    <div>
    <svg viewBox={`0 0 ${SW} 20`} preserveAspectRatio="none" /* i18n-ignore: an SVG attribute value */ style={{ width: '100%', height: 20, display: 'block' }} role="img" aria-label={tr('Lanes of the warp', 'Các lane của warp')} data-strip="lanes">
      {step.codeLanes.map((l) => (
        <g key={l.lane} data-lane={l.lane} data-active={l.active ? 1 : 0} data-block={l.block}>
          <rect x={l.lane * w + 0.5} y={2} width={w - 1} height={16}
            fill={!l.active ? 'var(--page)' : l.block % 2 ? 'var(--ink-3)' : 'var(--ink-2)'}
            stroke={l.active && l.misaligned ? 'var(--fail)' : 'var(--rule)'} strokeWidth={l.active && l.misaligned ? 2 : 1} vectorEffect="non-scaling-stroke" />
        </g>
      ))}
    </svg>
    <div className="muted" style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontFamily: 'var(--sans)' }}>
      <span><Translated vi="lane 0">lane 0</Translated></span><span><Translated vi="các lane của warp, tô màu theo khối; ô trống nghĩa là rảnh">the warp's lanes, shaded by block; empty means idle</Translated></span><span><Translated vi={<>lane {n - 1}</>}>lane {n - 1}</Translated></span>
    </div>
    </div>
  );
}

function WarpStepper({ cfg, set }: { cfg: StepperConfig; set: (c: StepperConfig) => void }) {
  const m: Mapping = { layout: cfg.layout, order: cfg.order, lanesPerBlock: cfg.lanesPerBlock, loadBytes: cfg.loadBytes, warpWidth: cfg.warpWidth, inW: cfg.inW };
  const why = problem(m);
  const segBytes = segmentBytesOf(cfg.seg, cfg.assumed);
  const lay = LAYOUTS[cfg.layout];
  const upd = (o: Partial<StepperConfig>) => set({ ...cfg, ...o, step: 0 });
  const steps = why ? [] : rowSteps(m);
  const at = steps[Math.min(cfg.step, steps.length - 1)];
  const step = at ? loadStep(m, at.iteration, at.load, segBytes) : null;
  const plan = why ? null : iterationPlan(cfg.inW, cfg.layout, cfg.lanesPerBlock, cfg.warpWidth);
  const K = why ? 0 : loadsPerLane(m);
  const go = (i: number) => set({ ...cfg, step: Math.max(0, Math.min(steps.length - 1, i)) });
  let strips = null;
  if (step && at && plan) {
    const [lo, hi] = iterationSpan(m, at.iteration);
    const blocks: number[] = [];
    const scaleIn: [number, number][] = [];
    for (let b = at.iteration * plan.blocksPerIteration; b < Math.min(plan.blocks, (at.iteration + 1) * plan.blocksPerIteration); b++) {
      const start = lay.interleaved ? b * (lay.scaleBytes + lay.codeBytes) : b * lay.codeBytes;
      blocks.push(start);
      if (lay.interleaved) scaleIn.push([start, start + lay.scaleBytes]);
    }
    const sl = step.scaleLanes.filter((l) => l.active).map((l) => l.addr);
    strips = (
      <>
        <Strip kind="code" lanes={step.codeLanes} lo={lo} hi={hi} segBytes={segBytes} blockLines={blocks} scaleBytes={scaleIn}
          label={lay.interleaved ? tr('the row, this iteration: grey marks each block’s 2-byte scale, dark the bytes this load asks for', 'hàng, lần lặp này: màu xám đánh dấu hệ số tỉ lệ 2 byte của mỗi khối, màu đậm là các byte mà lần nạp này yêu cầu') : tr('the code array, this iteration: dark marks the bytes this load asks for', 'mảng mã, lần lặp này: màu đậm đánh dấu các byte mà lần nạp này yêu cầu')} />
        <Strip kind="scale" lanes={step.scaleLanes} lo={Math.min(...sl)} hi={Math.max(...sl) + lay.scaleBytes} segBytes={segBytes} blockLines={[]} scaleBytes={[]}
          label={lay.interleaved ? tr('the scale load: each lane reads its block’s d, in the same array', 'lần nạp hệ số tỉ lệ: mỗi lane đọc d của khối mình, trong cùng mảng') : tr('the scale array: each lane reads its block’s scale', 'mảng hệ số tỉ lệ: mỗi lane đọc hệ số tỉ lệ của khối mình')} />
        <Lanes step={step} />
      </>
    );
  }
  const last = plan ? steps.findIndex((s) => s.iteration === plan.iterations - 1) : 0;
  return (
    <Bench
      title={tr('One warp across a quantised row', 'Một warp đi qua một hàng đã lượng tử hóa')}
      task={tr('Pick a layout and a mapping, then step through the loads. Find a mapping that touches few segments per useful byte.', 'Chọn một bố cục và một cách ánh xạ, rồi đi qua từng lần nạp. Tìm một cách ánh xạ chạm vào ít segment trên mỗi byte hữu ích.')}
      foot={
        <Translated
          vi={
            <>
              Mỗi lần một lệnh nạp, đếm theo quy tắc segment của chương 6. Không có bộ nhớ đệm: một segment được một lần nạp lấy
              về sẽ bị lần nạp tiếp theo lấy lại. Màu trong dải lane xen kẽ theo khối. {({ nvidia: 'NVIDIA: sector 32 byte trong dòng 128 byte', amd64: 'AMD: dòng cache 64 byte', amd128: 'AMD: dòng cache 128 byte', metal: 'Apple: không có tài liệu' } as Record<string, string>)[cfg.seg] ?? SEGMENT_MODELS[cfg.seg].label}
              {SEGMENT_MODELS[cfg.seg].segmentBytes == null ? `, nên segment ${cfg.assumed} byte là giả định của bạn` : ''} (
              {SEGMENT_MODELS[cfg.seg].facts}).
            </>
          }
        >
          <>
            One load instruction at a time, counted with chapter 6's segment rule. No cache: a segment fetched by one load is
            fetched again by the next. Shades in the lane strip alternate by block. {SEGMENT_MODELS[cfg.seg].label}
            {SEGMENT_MODELS[cfg.seg].segmentBytes == null ? `, so the ${cfg.assumed}-byte segment is your assumption` : ''} (
            {SEGMENT_MODELS[cfg.seg].facts}).
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field" style={{ maxWidth: '100%', minWidth: 0 }}>
          <Translated vi="Bố cục">Layout</Translated>
          <select value={cfg.layout} aria-label={tr('Layout', 'Bố cục')} style={{ maxWidth: '100%' }} onChange={(e) => upd({ layout: e.target.value as LayoutId })}>
            {LAYOUT_IDS.map((l) => <option key={l} value={l}>{LAYOUTS[l].label}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Số lane mỗi khối">Lanes per block</Translated>
          <span className="seg" data-control="lanes">
            {[1, 2, 4, 8].map((v) => <button key={v} aria-pressed={cfg.lanesPerBlock === v} onClick={() => upd({ lanesPerBlock: v })}>{v}</button>)}
          </span>
        </label>
        <label className="field">
          <Translated vi="Độ rộng lần nạp, byte">Load width, bytes</Translated>
          <span className="seg" data-control="width">
            {[1, 2, 4].map((v) => <button key={v} aria-pressed={cfg.loadBytes === v} onClick={() => upd({ loadBytes: v })}>{v}</button>)}
          </span>
        </label>
        <label className="field">
          <Translated vi="Mỗi lane đọc">Each lane reads</Translated>
          <span className="seg" data-control="order">
            {(['interleaved', 'piece'] as LoadOrder[]).map((v) => (
              <button key={v} aria-pressed={cfg.order === v} onClick={() => upd({ order: v })}>{v === 'interleaved' ? tr('interleaved with its neighbours', 'xen kẽ với các lane bên cạnh') : tr('its own piece', 'phần riêng của nó')}</button>
            ))}
          </span>
        </label>
        <label className="field">
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <span className="seg" data-control="warp">
            {[32, 64].map((v) => <button key={v} aria-pressed={cfg.warpWidth === v} onClick={() => upd({ warpWidth: v })}>{v}</button>)}
          </span>
        </label>
        <label className="field">
          <Translated vi="Độ dài hàng, số trọng số">Row length, weights</Translated>
          <select value={cfg.inW} aria-label={tr('Row length', 'Độ dài hàng')} onChange={(e) => upd({ inW: +e.target.value })}>
            {ROW_LENGTHS.map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className="field" style={{ maxWidth: '100%', minWidth: 0 }}>
          <Translated vi="Segment">Segments</Translated>
          <select value={cfg.seg} aria-label={tr('Segments', 'Segment')} style={{ maxWidth: '100%' }} onChange={(e) => upd({ seg: e.target.value as SegmentModelId })}>
            {(Object.keys(SEGMENT_MODELS) as SegmentModelId[]).map((s) => <option key={s} value={s}>{tr(SEGMENT_MODELS[s].label, ({ nvidia: 'NVIDIA: sector 32 byte trong dòng 128 byte', amd64: 'AMD: dòng cache 64 byte', amd128: 'AMD: dòng cache 128 byte', metal: 'Apple: không có tài liệu' } as Record<string, string>)[s] ?? SEGMENT_MODELS[s].label)}</option>)}
          </select>
        </label>
        {SEGMENT_MODELS[cfg.seg].segmentBytes == null ? (
          <label className="field">
            <Translated vi="Giả định">Assume</Translated>
            <select value={cfg.assumed} aria-label={tr('Assumed segment', 'Segment giả định')} onChange={(e) => upd({ assumed: +e.target.value })}>
              {[32, 64, 128].map((v) => <option key={v} value={v}>{v} {tr('bytes', 'byte')}</option>)}
            </select>
          </label>
        ) : null}
      </div>
      {why ? (
        <p data-problem style={{ margin: 0 }}><span className="chip fail"><Translated vi="không chạy được">cannot run</Translated></span> <Translated vi={<>Với bố cục này, {why}.</>}>With this layout, {why}.</Translated></p>
      ) : step && at && plan ? (
        <>
          <div className="row" style={{ marginBottom: '0.6rem' }}>
            <button className="primary" onClick={() => go(cfg.step + 1)} disabled={cfg.step >= steps.length - 1}><Translated vi="Bước tiếp">Step</Translated></button>
            <button onClick={() => go((at.iteration + 1) * K)} disabled={at.iteration >= plan.iterations - 1}><Translated vi="Lần lặp tiếp">Next iteration</Translated></button>
            <button onClick={() => go(last)}><Translated vi="Lần lặp cuối">Last iteration</Translated></button>
            <button onClick={() => go(0)}><Translated vi="Đặt lại">Reset</Translated></button>
          </div>
          <Translated
            vi={
              <p className="narrate" data-narrate style={{ margin: '0 0 0.5rem' }}>
                Lần lặp {at.iteration + 1} trên {plan.iterations}, lần nạp {at.load + 1} trên {K}: {cfg.warpWidth - step.idle} lane, mỗi lane đọc{' '}
                {cfg.loadBytes} byte từ {Math.min(plan.blocksPerIteration, plan.blocks - at.iteration * plan.blocksPerIteration)}{' '}
                khối
                {step.idle ? `; ${step.idle} lane không qua được phép kiểm tra biên và nằm rảnh` : ''}.
              </p>
            }
          >
            <p className="narrate" data-narrate style={{ margin: '0 0 0.5rem' }}>
              Iteration {at.iteration + 1} of {plan.iterations}, load {at.load + 1} of {K}: {cfg.warpWidth - step.idle} lanes read{' '}
              {cfg.loadBytes} byte{cfg.loadBytes > 1 ? 's' : ''} each from {Math.min(plan.blocksPerIteration, plan.blocks - at.iteration * plan.blocksPerIteration)}{' '}
              block{plan.blocks - at.iteration * plan.blocksPerIteration > 1 ? 's' : ''}
              {step.idle ? `; ${step.idle} lanes fail the bounds test and sit idle` : ''}.
            </p>
          </Translated>
          {strips}
          <div className="cols" style={{ marginTop: '0.6rem' }}>
            <div className="panel" data-panel="code">
              <p style={{ margin: '0 0 0.3rem' }}><b><Translated vi="Lần nạp mã này">This code load</Translated></b></p>
              <table className="data" style={{ margin: 0 }}>
                <tbody>
                  <tr><td><Translated vi="Số segment bị chạm">Segments touched</Translated></td><td className="num" data-code-segments>{step.code.transactions}</td></tr>
                  <tr><td><Translated vi="Byte hữu ích">Useful bytes</Translated></td><td className="num" data-code-needed>{step.code.bytesNeeded}</td></tr>
                  <tr><td><Translated vi="Byte được nạp">Bytes fetched</Translated></td><td className="num">{step.code.bytesFetched}</td></tr>
                  <tr><td><Translated vi="Tỉ lệ hữu ích">Useful fraction</Translated></td><td className="num" data-code-eff={step.code.efficiency.toFixed(3)}>{(100 * step.code.efficiency).toFixed(0)} %</td></tr>
                  <tr>
                    <td><Translated vi="Lần nạp lệch căn chỉnh">Misaligned loads</Translated></td>
                    <td className="num" data-misaligned={step.misaligned}>{step.misaligned ? <span className="chip fail">{step.misaligned}</span> : 0}</td>
                  </tr>
                  <tr><td><Translated vi="Lane rảnh">Idle lanes</Translated></td><td className="num" data-idle={step.idle}>{step.idle}</td></tr>
                </tbody>
              </table>
            </div>
            <div className="panel" data-panel="scale">
              <p style={{ margin: '0 0 0.3rem' }}><b><Translated vi="Lần nạp hệ số tỉ lệ của lần lặp này">This iteration's scale load</Translated></b></p>
              <table className="data" style={{ margin: 0 }}>
                <tbody>
                  <tr><td><Translated vi="Số hệ số tỉ lệ khác nhau">Distinct scales</Translated></td><td className="num">{step.scale.bytesNeeded / lay.scaleBytes}</td></tr>
                  <tr><td><Translated vi="Số segment bị chạm">Segments touched</Translated></td><td className="num" data-scale-segments>{step.scale.transactions}</td></tr>
                  <tr><td><Translated vi="Tỉ lệ hữu ích">Useful fraction</Translated></td><td className="num">{(100 * step.scale.efficiency).toFixed(0)} %</td></tr>
                </tbody>
              </table>
              <Translated
                vi={
                  <p className="muted" style={{ margin: '0.5rem 0 0' }}>
                    {plan.blocks} khối mỗi hàng, {plan.blocksPerIteration} mỗi lần lặp, {K} lần nạp mã mỗi lane mỗi lần lặp.
                  </p>
                }
              >
                <p className="muted" style={{ margin: '0.5rem 0 0' }}>
                  {plan.blocks} blocks per row, {plan.blocksPerIteration} per iteration, {K} code load{K > 1 ? 's' : ''} per lane per iteration.
                </p>
              </Translated>
              {step.misaligned ? (
                <Translated
                  vi={
                    <p style={{ margin: '0.5rem 0 0' }}>
                      Một lần nạp {cfg.loadBytes} byte từ một địa chỉ không phải bội số của {cfg.loadBytes} là hành vi không xác
                      định trong CUDA, HIP và Metal như nhau. Cứ cách một khối Q4_0 lại có một khối có mã bắt đầu lệch 2 byte so
                      với một bội số của 4.
                    </p>
                  }
                >
                  <p style={{ margin: '0.5rem 0 0' }}>
                    A {cfg.loadBytes}-byte load from an address that is not a multiple of {cfg.loadBytes} is undefined behaviour
                    in CUDA, HIP and Metal alike. Every other Q4_0 block's codes start 2 bytes past a multiple of 4.
                  </p>
                </Translated>
              ) : null}
            </div>
          </div>
        </>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- idle lanes against row length
function TailExplorer() {
  const [inW, setIn] = useState(1056);
  const [lpb, setLpb] = useState(1);
  const W = 620;
  const H = 180;
  const pad = { l: 44, r: 12, t: 12, b: 30 };
  const xs: number[] = [];
  for (let v = 32; v <= 16384; v += 32) xs.push(v);
  const X = (v: number) => pad.l + ((Math.log2(v) - 5) / (14 - 5)) * (W - pad.l - pad.r);
  const Y = (f: number) => H - pad.b - f * (H - pad.t - pad.b);
  const line = (ww: number) => xs.map((v) => `${X(v).toFixed(1)},${Y(iterationPlan(v, 'q4_0-blocks', lpb, ww).idleFraction).toFixed(1)}`).join(' ');
  const plans = [32, 64].map((ww) => ({ ww, p: iterationPlan(inW, 'q4_0-blocks', lpb, ww) }));
  return (
    <Bench
      title={tr('Lanes per block against row length', 'Số lane mỗi khối theo độ dài hàng')}
      task={tr('Drag the row length and change the lanes per block. When does a 64-lane wavefront waste a large share of its lanes?', 'Kéo độ dài hàng và thay đổi số lane mỗi khối. Khi nào một wavefront 64 lane lãng phí phần lớn số lane của nó?')}
      foot={<Translated vi={<>Các khối Q4_0 gồm 32 trọng số. Một warp đi qua một hàng; lane rảnh là các lượt lane-lần lặp không qua được phép kiểm tra biên, trên tổng số lượt lane-lần lặp của hàng.</>}><>Q4_0 blocks of 32 weights. One warp walks one row; idle lanes are the lane-iterations that fail the bounds test, over all lane-iterations of the row.</></Translated>}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi={<>Độ dài hàng in = {inW} ({inW / 32} khối)</>}>
            Row length in = {inW} ({inW / 32} blocks)
          </Translated>
          <input type="range" min="1" max="512" value={inW / 32} aria-label={tr('Row length', 'Độ dài hàng')} onChange={(e) => setIn(32 * +e.target.value)} />
        </label>
        <label className="field">
          <Translated vi="Số lane mỗi khối">Lanes per block</Translated>
          <span className="seg" data-control="tail-lanes">
            {[1, 2, 4, 8].map((v) => <button key={v} aria-pressed={lpb === v} onClick={() => setLpb(v)}>{v}</button>)}
          </span>
        </label>
      </div>
      <div className="cols">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Idle share of lanes against row length', 'Tỉ lệ lane rảnh theo độ dài hàng')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
          <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <g key={f}>
              <line x1={pad.l} x2={W - pad.r} y1={Y(f)} y2={Y(f)} stroke="var(--rule-soft)" />
              <text x={pad.l - 5} y={Y(f) + 4} fontSize="10.5" textAnchor="end">{f * 100}%</text>
            </g>
          ))}
          {[32, 128, 512, 2048, 8192].map((v) => (
            <text key={v} x={X(v)} y={H - pad.b + 14} fontSize="10.5" textAnchor="middle">{v}</text>
          ))}
          <text x={(W + pad.l) / 2} y={H - 3} fontSize="10.5" textAnchor="middle"><Translated vi="độ dài hàng, số trọng số (thang logarit)">row length, weights (log scale)</Translated></text>
          <polyline points={line(64)} fill="none" stroke="var(--ink-3)" strokeWidth="1.8" strokeDasharray="5 3" />
          <polyline points={line(32)} fill="none" stroke="var(--ink)" strokeWidth="1.6" />
          <line x1={X(inW)} x2={X(inW)} y1={pad.t} y2={H - pad.b} stroke="var(--reg)" strokeDasharray="3 3" />
          <text x={W - pad.r - 6} y={pad.t + 14} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-3)' }}><Translated vi="64 lane, nét đứt">64 lanes, dashed</Translated></text>
          <text x={W - pad.r - 6} y={pad.t + 28} fontSize="10.5" textAnchor="end"><Translated vi="32 lane">32 lanes</Translated></text>
        </svg>
        <div className="panel" data-panel="tail">
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr><th><Translated vi="Độ rộng warp">Warp width</Translated></th><th className="num"><Translated vi="Số lần lặp">Iterations</Translated></th><th className="num"><Translated vi="Rảnh ở lần cuối">Idle in the last</Translated></th><th className="num"><Translated vi="Rảnh tổng thể">Idle overall</Translated></th></tr>
            </thead>
            <tbody>
              {plans.map(({ ww, p }) => (
                <tr key={ww} data-width={ww} data-iterations={p.iterations} data-idle-last={p.idleLast} data-idle-frac={p.idleFraction.toFixed(4)}>
                  <td>{ww}</td>
                  <td className="num">{p.iterations}</td>
                  <td className="num"><Translated vi={<>{p.idleLast} lane</>}>{p.idleLast} lanes</Translated></td>
                  <td className="num">{(100 * p.idleFraction).toFixed(1)} %</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the code
/** The same words as the labs page's status table (LabKit's solutionTabStatus). */
function solutionStatus(reports: Reports, lab: string): { cu: string; metal: string } {
  return { cu: solutionTabStatus(reports, lab, 'kernel.cu'), metal: solutionTabStatus(reports, lab, 'kernel.metal') };
}

function LabX22Code() {
  const { reports } = useReports();
  const st = solutionStatus(reports, LAB_X22);
  const cu = loaded(useLabFile(`${LAB_X22}/solution/kernel.cu`));
  const metal = loaded(useLabFile(`${LAB_X22}/solution/kernel.metal`));
  if (!cu && !metal)
    return (
      <Translated
        vi={
          <p className="muted" data-code="missing">
            Lời giải tham chiếu (solution) của lab X2.2 chưa có trong <code>labs/{LAB_X22}/solution/</code> trên máy này. Khi có,
            nó sẽ xuất hiện ở đây với một tab CUDA và một tab Metal.
          </p>
        }
      >
        <p className="muted" data-code="missing">
          Lab X2.2's reference solution is not in <code>labs/{LAB_X22}/solution/</code> on this machine yet. When it is,
          it appears here with a CUDA tab and a Metal tab.
        </p>
      </Translated>
    );
  return (
    <Code
      tabs={[
        ...(cu ? [{ label: tr('CUDA and HIP', 'CUDA và HIP'), code: cu, status: st.cu }] : []),
        ...(metal ? [{ label: 'Metal', code: metal, status: st.metal }] : []),
      ]}
      caption={<Translated vi={<>Lời giải tham chiếu của lab X2.2, nạp từ <code>labs/{LAB_X22}/solution/</code>. Hãy tự viết bản của bạn trong lab trước.</>}><>Lab X2.2's reference solution, loaded from <code>labs/{LAB_X22}/solution/</code>. Write your own in the lab first.</></Translated>}
    />
  );
}

// ---------------------------------------------------------------- the section
export function SX24() {
  const q = questionsSX24;
  const [cfg, setCfg] = useState<StepperConfig>(DEFAULT_STEPPER);
  const x21 = loadStep(
    { layout: LAB_X21_MAPPING.layout, lanesPerBlock: LAB_X21_MAPPING.lanesPerBlock, loadBytes: LAB_X21_MAPPING.loadBytes, warpWidth: 32, inW: 4096 },
    0, 0, 32,
  );
  return (
    <>
      <Translated
        vi={
          <p>
            Bạn đã có sẵn hình dạng kernel. GEMV dạng gather của lab 22.1 giao mỗi hàng đầu ra cho một warp; các lane cùng nhau
            đi dọc hàng theo bước nhảy (stride), mỗi lane giữ một tổng chạy cho mỗi hàng của batch, một phép rút gọn (reduction)
            bằng shuffle gom các tổng lại, và lane 0 ghi kết quả. Khi decode, batch tối đa là 8, và mỗi trọng số được nạp một
            lần và dùng cho mọi hàng của batch.
          </p>
        }
      >
        <p>
          You have the kernel shape already. Lab 22.1's gather GEMV gives each output row to one warp; the lanes stride along
          the row together, each keeps a running sum per batch row, a shuffle reduction collects the sums, and lane 0
          writes. At decode the batch is at most 8, and every weight is loaded once and used for every batch row.
        </p>
      </Translated>
      <Source section="22.2" pages="532–536" />
      <Translated
        vi={
          <p>
            llama.cpp cũng chọn như vậy: tới 8 token nó chạy các kernel nhân ma trận với vector, không phải một GEMM phân ô. Phép
            rút gọn warp của chương 10 là phần kết của mọi kernel như thế.
          </p>
        }
      >
        <p>
          llama.cpp makes the same choice: up to 8 tokens it runs matrix-vector kernels, not a tiled GEMM. Chapter 10's warp
          reduction is the end of every one of them.
        </p>
      </Translated>
      <Source section="10.7" pages="237–241" />
      <Beyond title={tr("Ours, not the book's", 'Của khóa học này, không phải của sách')}>
        <Translated
          vi={
            <p>
              Sách không in kernel lượng tử hóa nào. Các cách ánh xạ trên trang này là của khóa học này, được đối chiếu với mã
              nguồn của llama.cpp và MLX trong lượt kiểm tra dữ kiện ngày 2026-11-20 ({FACTS} A5, D3 đến D5; ch15-facts D1).
            </p>
          }
        >
          <p>
            The book prints no quantised kernel. The mappings on this page are ours, read against llama.cpp's and MLX's
            sources in the facts pass on 2026-11-20 ({FACTS} A5, D3 to D5; ch15-facts D1).
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Khối thay đổi điều gì">What blocks change</Translated></h2>
      <Translated
        vi={
          <p>
            Với trọng số float32, đơn vị công việc của một lane là một số float. Với trọng số lượng tử hóa, đó là một phần của
            một khối: vài byte mã dùng chung một hệ số tỉ lệ. Một lane nạp các mã của nó, giải nén chúng bằng mặt nạ và phép
            dịch bit, nhân chúng với các activation rồi cộng, sau đó nhân tổng riêng phần với hệ số tỉ lệ của khối một lần. Giờ
            có hai dòng dữ liệu rời bộ nhớ: các mã, chiếm phần lớn số byte, và các hệ số tỉ lệ, một dòng thứ hai nhỏ.
          </p>
        }
      >
        <p>
          With float32 weights, a lane's unit of work is one float. With quantised weights it is a piece of a block: some
          code bytes that share one scale. A lane loads its codes, unpacks them with masks and shifts, multiplies them by
          the activations and adds, then multiplies the partial sum by the block's scale once. Two streams now leave
          memory: the codes, which are most of the bytes, and the scales, a small second stream.
        </p>
      </Translated>
      <Ask q={q['x2/X2.4/q-scale']} />

      <h2><Translated vi="Ba cách ánh xạ và việc hợp nhất truy cập của chúng">Three mappings and their coalescing</Translated></h2>
      <Translated
        vi={
          <p>
            Các lần nạp của một warp có được hợp nhất hay không là câu hỏi của chương 6: địa chỉ của các lane rơi vào bao nhiêu
            segment đã căn chỉnh? Các khối khiến câu hỏi thú vị theo ba cách. Đặt một lane trên mỗi khối Q4_0 18 byte thì các
            lane kề nhau cách nhau 18 byte. Đặt nhiều lane trên mỗi khối thì chúng đọc các byte kề nhau, đó là cách llama.cpp đọc
            Q4_0, theo từng nửa 16 bit. Hoặc lưu mã và hệ số tỉ lệ trong hai mảng riêng, như lab X2.1 và MLX làm, thì các mã vẫn
            căn chỉnh 4 byte, nên một lane có thể nạp cả một word 32 bit.
          </p>
        }
      >
        <p>
          Whether a warp's loads coalesce is chapter 6's question: how many aligned segments do the lanes' addresses fall
          in? Blocks make it interesting in three ways. Put one lane on each 18-byte Q4_0 block and neighbouring lanes are 18
          bytes apart. Put several lanes on each block and they read neighbouring bytes, which is how llama.cpp reads Q4_0, in
          16-bit halves. Or store codes and scales in separate arrays, as lab X2.1 and MLX do, and the codes stay 4-byte
          aligned, so a lane can load a whole 32-bit word.
        </p>
      </Translated>
      <Source section="6.1" pages="124–132" />
      <Today checked={tr(`2026-11-20 in the facts pass (${FACTS}, A5, C5, D3, D5)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${FACTS}, A5, C5, D3, D5)`)}>
        <Translated
          vi={
            <p>
              llama.cpp đọc mã Q4_0 bằng <code>get_int_b2</code>, "assume at least 2 byte alignment", và giữ{' '}
              <code>get_int_b4</code> 4 byte cho các activation Q8_1. Các kernel Metal của nó đọc cùng các khối đó dưới dạng word{' '}
              <code>uint16_t</code>.
            </p>
          }
        >
          <p>
            llama.cpp reads Q4_0 codes with <code>get_int_b2</code>, "assume at least 2 byte alignment", and keeps the
            4-byte <code>get_int_b4</code> for the Q8_1 activations. Its Metal kernels read the same blocks as{' '}
            <code>uint16_t</code> words.
          </p>
        </Translated>
      </Today>
      <Simplification>
        <Translated
          vi={
            <p>
              Mô hình segment của chương 6: sector 32 byte của NVIDIA, line 64 hoặc 128 byte của AMD. Apple không công bố kích
              thước segment, nên với Metal kích thước là giả định của bạn. Không có bộ nhớ đệm; không hiển thị các lần nạp
              activation.
            </p>
          }
        >
          <p>
            Chapter 6's segment model: NVIDIA's 32-byte sectors, AMD's 64- or 128-byte lines. Apple documents no segment size,
            so for Metal the size is your assumption. No caches; activation loads not shown.
          </p>
        </Translated>
      </Simplification>
      <WarpStepper cfg={cfg} set={setCfg} />
      <Translated
        vi={
          <p>
            Những điều cần tìm. Trên các khối Q4_0, một lane mỗi khối với lần nạp 2 byte chạm 18 segment cho 64 byte hữu ích;
            bốn lane mỗi khối chạm 5. Đổi độ rộng lần nạp thành 4 byte trên các khối Q4_0 và lần nạp của mọi khối chẵn chuyển
            sang đỏ: các mã đó bắt đầu lệch 2 byte so với một bội số của 4. Trên các mảng riêng, cùng các lần nạp 4 byte đều
            căn chỉnh. Và hãy đi đến lần lặp cuối của một hàng 1056 trọng số để thấy các lane rơi ra khỏi phần cuối.
          </p>
        }
      >
        <p>
          Things to find. On Q4_0 blocks, one lane per block with 2-byte loads touches 18 segments for 64 useful bytes; four
          lanes per block touch 5. Switch the load width to 4 bytes on Q4_0 blocks and every even block's load turns red:
          those codes start 2 bytes past a multiple of 4. On the separate arrays the same 4-byte loads are all aligned. And
          step to the last iteration of a 1056-weight row to see lanes fall off the end.
        </p>
      </Translated>
      <Ask q={q['x2/X2.4/q-lane-per-block']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={q['x2/X2.4/q-four-lanes']} />

      <h3><Translated vi="Cách ánh xạ của lab X2.1">Lab X2.1's mapping</Translated></h3>
      <Translated
        vi={
          <p>
            Trong lab X2.1, các mã Q8_0 của một hàng là một mảng liên tục và các hệ số tỉ lệ là một mảng khác. Lane L của warp
            đọc các byte mã 4L đến 4L + 3 của đoạn tiếp theo của hàng j dưới dạng một word 32 bit, nên tám lane dùng chung mỗi
            khối 32 trọng số.
          </p>
        }
      >
        <p>
          In lab X2.1 the Q8_0 codes of a row are one contiguous array and the scales another. Lane L of the warp reads code
          bytes 4L to 4L + 3 of the next stretch of row j as one 32-bit word, so eight lanes share each 32-weight block.
        </p>
      </Translated>
      <Predict
        id="x2/X2.4/p-lab1-segments"
        unit={tr('segments', 'segment')}
        prompt={<Translated vi={<>Với 32 lane và segment 32 byte, một trong các lần nạp mã đó chạm bao nhiêu segment? (Sau đó hỏi điều tương tự với các lần nạp hệ số tỉ lệ.)</>}><>With 32 lanes and 32-byte segments, how many segments does one of those code loads touch? (Then ask the same of the scale loads.)</></Translated>}
      >
        {(c) => (
          <div data-reveal="x21">
            <Translated
              vi={
                <p>
                  <b>{x21.code.transactions} segment</b> cho {x21.code.bytesNeeded} byte hữu ích, mọi byte được nạp đều được dùng
                  {c === x21.code.transactions ? ', đúng như bạn nói' : ''}: 32 lane × 4 byte là 128 byte liên tiếp, đã căn chỉnh.
                  Các lần nạp hệ số tỉ lệ là một lệnh thứ hai: 4 khối, 4 hệ số tỉ lệ FP16 khác nhau, {x21.scale.bytesNeeded} byte
                  trong{' '}
                  {x21.scale.transactions} segment, dùng {(100 * x21.scale.efficiency).toFixed(0)} % của nó. Nhỏ, và chỉ trả một lần mỗi lần lặp.
                </p>
              }
            >
              <p>
                <b>{x21.code.transactions} segments</b> for {x21.code.bytesNeeded} useful bytes, every fetched byte used
                {c === x21.code.transactions ? ', as you said' : ''}: 32 lanes × 4 bytes is 128 consecutive, aligned bytes. The
                scale loads are a second instruction: 4 blocks, 4 distinct FP16 scales, {x21.scale.bytesNeeded} bytes in{' '}
                {x21.scale.transactions} segment, {(100 * x21.scale.efficiency).toFixed(0)} % of it used. Small, and paid once per iteration.
              </p>
            </Translated>
            <button onClick={() => setCfg(LAB_X21_MAPPING)} data-action="load-x21"><Translated vi="Hiển thị trong trình chạy từng bước">Show it in the stepper</Translated></button>
          </div>
        )}
      </Predict>
      <Translated
        vi={
          <p className="muted">
            Đây là câu hỏi của bài tập 4 chương 20 (p. 511), liệu mọi lần nạp của kernel attention có được hợp nhất hay không,
            đặt ra cho một hàng đã lượng tử hóa.
          </p>
        }
      >
        <p className="muted">
          This is the question of chapter 20's exercise 4 (p. 511), whether every load of the attention kernel is coalesced,
          asked of a quantised row.
        </p>
      </Translated>

      <h2><Translated vi="Phần cuối của một hàng">The end of a row</Translated></h2>
      <Translated
        vi={
          <p>
            Vòng lặp chạy khi chỉ số khối của lane còn nằm trong hàng. Khi các khối của hàng không lấp đầy lần lặp cuối, các lane
            vượt quá phần cuối đi vào nhánh sai của phép kiểm tra và chờ. Đó là phân kỳ luồng (divergence) ở biên của chương 4,
            và chi phí của nó phụ thuộc vào độ dài hàng so với độ rộng warp.
          </p>
        }
      >
        <p>
          The loop runs while the lane's block index is inside the row. When the row's blocks do not fill the last
          iteration, the lanes past the end take the false side of the test and wait. That is chapter 4's boundary
          divergence, and its cost depends on how the row length compares with the warp width.
        </p>
      </Translated>
      <Source section="4.5" pages="79–82" />
      <TailExplorer />
      <Translated
        vi={
          <p>
            Với hàng dài, phần đuôi chỉ vài phần trăm. Với hàng ngắn trên một wavefront 64 lane, nó có thể là một nửa số lane:
            một hàng 1024 trọng số có 32 khối Q4_0, lấp đầy đúng một warp 32 lane và một nửa warp 64 lane. Đặt hai lane trên mỗi
            khối lại lấp đầy nó. Cách ánh xạ đúng phụ thuộc vào độ rộng warp, nên nó cần được tính từ độ rộng đó.
          </p>
        }
      >
        <p>
          For long rows the tail is a few percent. For short rows on a 64-lane wavefront it can be half the lanes: a
          1024-weight row has 32 Q4_0 blocks, which fills a 32-lane warp exactly and half of a 64-lane one. Putting two lanes
          on each block fills it again. The right mapping depends on the warp width, so it should be computed from it.
        </p>
      </Translated>
      <Ask q={q['x2/X2.4/q-tail']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Độ rộng warp">Warp width</Translated></h2>
      <Translated
        vi={
          <p>
            Một warp là 32 lane trên NVIDIA, 64 hoặc 32 trên AMD, và trên Metal là bất cứ giá trị nào pipeline báo cáo. Mọi biểu
            thức trên trang này có nêu số lane đều dùng <code>warpSize</code> (CUDA và HIP) hoặc{' '}
            <code>threadExecutionWidth</code> của pipeline (Metal, bên trong kernel là <code>[[threads_per_simdgroup]]</code>),
            không bao giờ là 32.
          </p>
        }
      >
        <p>
          A warp is 32 lanes on NVIDIA, 64 or 32 on AMD, and on Metal whatever the pipeline reports. Every expression on this
          page that names a lane count names <code>warpSize</code> (CUDA and HIP) or the pipeline's{' '}
          <code>threadExecutionWidth</code> (Metal, inside the kernel <code>[[threads_per_simdgroup]]</code>), never 32.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch05-facts.md B5; notes/today/ch03-facts.md C9)">{/* i18n-ignore: a date and a citation */}
        <Translated
          vi={
            <p>
              AMD: "AMD devices return 64 for gfx9 and 32 for gfx10 and above." (các thiết bị AMD trả về 64 với gfx9 và 32 với
              gfx10 trở lên). Apple: "The number of threads in a SIMD group is returned by the threadExecutionWidth of your
              compute pipeline state object." (số luồng trong một SIMD group được trả về qua threadExecutionWidth của đối tượng
              compute pipeline state).
            </p>
          }
        >
          <p>
            AMD: "AMD devices return 64 for gfx9 and 32 for gfx10 and above." Apple: "The number of threads in a SIMD group is
            returned by the threadExecutionWidth of your compute pipeline state object."
          </p>
        </Translated>
      </Today>
      <Ask q={q['x2/X2.4/q-wave64']} />

      <h2><Translated vi="Mã nguồn">The code</Translated></h2>
      <Translated
        vi={
          <p>
            Lab X2.2 yêu cầu kernel này trên các khối Q4_0 của llama.cpp. So với GEMV gather của lab 22.1, ba điều thay đổi: hàng
            được duyệt theo khối thay vì theo số float, mỗi byte mã được giải nén thành hai mã (trọng số j ở nibble thấp, trọng
            số j + 16 ở nibble cao, mỗi mã trừ 8), và phần khối của mỗi lane được nhân với d của nó một lần, với -8 được đưa ra
            ngoài thông qua tổng các activation: d × (Σ q·x − 8 Σ x). Phép rút gọn và luồng ghi duy nhất không đổi.
          </p>
        }
      >
        <p>
          Lab X2.2 asks for this kernel on llama.cpp's Q4_0 blocks. Against lab 22.1's gather, three things change: the
          row is walked in blocks rather than floats, each code byte is unpacked into two codes (weight j in the low
          nibble, weight j + 16 in the high one, each minus 8), and each lane's share of a block is multiplied by its d once, with the -8 folded out through the sum of the
          activations: d × (Σ q·x − 8 Σ x). The reduction and the single writer are unchanged.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Với HIP, cùng file <code>.cu</code> được viết để build qua <code>pmpp/vendors/hip.h</code>: mặt nạ và phép dịch bit
            không cần intrinsic nào, và các lệnh shuffle của phép rút gọn nhận một lane mask có kiểu, thứ mà HIP cần rộng 64 bit
            (unverified: chưa có máy HIP nào chạy nó). Trên Metal, khối là một struct gồm một <code>half d</code> và <code>uchar qs[16]</code>, như llama.cpp
            khai báo, và <code>simd_sum</code> thay cho các lệnh shuffle.
          </p>
        }
      >
        <p>
          For HIP the same <code>.cu</code> is written to build through <code>pmpp/vendors/hip.h</code>: masks and
          shifts need no intrinsic, and the reduction's shuffles take a typed lane mask, which HIP needs 64 bits wide
          (unverified: no HIP machine has run it). On Metal the block is a struct of a <code>half d</code> and <code>uchar qs[16]</code>, as llama.cpp
          declares it, and <code>simd_sum</code> replaces the shuffles.
        </p>
      </Translated>
      <Today checked={tr(`2026-11-20 in the facts pass (${FACTS}, C5, D1, D3)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${FACTS}, C5, D1, D3)`)}>
        <Translated
          vi={
            <p>
              <code>mul_mat_vec_q</code> của llama.cpp giữ một số float cho mỗi kết quả khối trong{' '}
              <code>tmp[ncols_dst][rows_per_cuda_block]</code> và rút gọn giữa các warp qua bộ nhớ chia sẻ (shared memory).
            </p>
          }
        >
          <p>
            llama.cpp's <code>mul_mat_vec_q</code> keeps one float per block result in{' '}
            <code>tmp[ncols_dst][rows_per_cuda_block]</code> and reduces across warps through shared memory.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Cách ánh xạ của nó là bốn lane mỗi khối, mỗi lane đọc 4 byte mã của riêng nó bằng hai lần nạp 16 bit, không bao giờ
            bằng một word 32 bit, vì vấn đề căn chỉnh bạn đã thấy trong trình chạy từng bước.{' '}
            <button onClick={() => { setCfg(LAB_X22_MAPPING); document.querySelector('[aria-label="Một warp đi qua một hàng đã lượng tử hóa"]')?.scrollIntoView({ block: 'start' }); }} data-action="load-x22">
              Hiển thị cách ánh xạ của lab X2.2 trong trình chạy từng bước
            </button>
          </p>
        }
      >
        <p>
          Its mapping is four lanes per block, each reading its own 4 bytes of codes as two 16-bit loads, never as one 32-bit
          word, because of the alignment you saw in the stepper.{' '}
          <button onClick={() => { setCfg(LAB_X22_MAPPING); document.querySelector('[aria-label="One warp across a quantised row"]')?.scrollIntoView({ block: 'start' }); }} data-action="load-x22">
            Show lab X2.2's mapping in the stepper
          </button>
        </p>
      </Translated>
      <LabX22Code />
    </>
  );
}
