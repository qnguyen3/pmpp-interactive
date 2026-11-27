import { useEffect, useId, useMemo, useRef, useState, type ReactNode, type RefObject } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { fmt, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  EXERCISE_4,
  KERNELS,
  finding,
  run,
  timeline,
  type Finding,
  type KernelId,
  type Order,
  type Rule,
  type Step,
} from '../../sim/barrier';
import {
  FIG45,
  TAIL_EXAMPLE,
  capacity,
  deviceFromReport,
  equalDurations,
  isGpuBackend,
  profile,
  schedule,
  tail,
  unevenDurations,
  usage,
  waveSizes,
  waves,
} from '../../sim/blocksched';
import { WavesScene } from './WavesScene';
import { Translated, tr } from '../../lib/i18n';

// The kernels of this page, exactly as stored in labs/ch04/snippets/<name>.cu, where
// the CUDA front-end check parses them. app/src/sim/barrier.test.ts fails if a copy
// here drifts from its file.
const SNIPPETS: Record<KernelId, string> = {
  incorrect_barrier: `__global__ void incorrect_barrier_example(int n) {
    // ...
    if (threadIdx.x % 2 == 0) {
        // ... work of the even threads
        __syncthreads();
    }
    else {
        // ... work of the odd threads
        __syncthreads();
    }
}`,
  barrier_uniform: `__global__ void barrier_uniform(float* data) {
    __shared__ float s[4];                         // the block has 4 threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] = data[i];                      // phase 0: write my slot
    if (blockIdx.x > 0) {
        __syncthreads();
    }
    data[i] = s[(threadIdx.x + 1) % blockDim.x];   // phase 1: read a neighbour's slot
}`,
  barrier_per_thread: `__global__ void barrier_per_thread(float* data) {
    __shared__ float s[4];                         // the block has 4 threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] = data[i];                      // phase 0: write my slot
    if (threadIdx.x > 0) {
        __syncthreads();
    }
    data[i] = s[(threadIdx.x + 1) % blockDim.x];   // phase 1: read a neighbour's slot
}`,
  barrier_early_return: `__global__ void barrier_early_return(float* data, int n) {
    __shared__ float s[4];                         // the block has 4 threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;                            // threads past the end of the data leave here
    s[threadIdx.x] = data[i];                      // phase 0: write my slot
    __syncthreads();
    data[i] = s[0];                                // phase 1: read thread 0's slot
}`,
  barrier_loop: `__global__ void barrier_loop(float* data) {
    __shared__ float s[4];                         // the block has 4 threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // odd threads go round the loop twice, even threads once
    for (unsigned k = 0; k <= threadIdx.x % 2; ++k) {
        s[threadIdx.x] = data[i] + k;              // one round: write my slot
        __syncthreads();
    }
    data[i] = s[(threadIdx.x + 1) % blockDim.x];   // after the loop: read a neighbour's slot
}`,
  barrier_guarded: `__global__ void barrier_guarded(float* data, int n) {
    __shared__ float s[4];                         // the block has 4 threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool mine = i < n;                             // threads past the end stay, and do nothing
    if (mine) s[threadIdx.x] = data[i];            // phase 0: write my slot
    __syncthreads();
    if (mine) data[i] = s[0];                      // phase 1: read thread 0's slot
}`,
};
const TIERS: Record<string, Tier> = { s: 'shared', data: 'global', i: 'reg', k: 'reg', mine: 'reg' };
// The files parse with the CUDA front end; none of them is built or run. Two of them have a
// legal barrier (labs/ch04/snippets/README.md), so only the other four are "wrong on purpose".
const SNIPPET_STATUS = () => tr('front-end-checked; wrong on purpose, never run', 'front-end-checked; cố ý sai, chưa từng chạy');
const SNIPPET_STATUS_LEGAL = () => tr('front-end-checked; never run', 'front-end-checked; chưa từng chạy');
const statusOf = (id: KernelId) => (id === 'barrier_uniform' || id === 'barrier_guarded' ? SNIPPET_STATUS_LEGAL() : SNIPPET_STATUS());

const f1 = (v: number) => v.toFixed(1);
const round1 = (v: number) => Math.round(v * 10) / 10;
const list = (a: number[]) => (a.length <= 1 ? a.join('') : a.slice(0, -1).join(', ') + ' and ' + a[a.length - 1]);
const plural = (a: number[], one: string) => `${one}${a.length === 1 ? '' : 's'} ${list(a)}`;
const wait = (a: number[]) => (a.length === 1 ? 'waits' : 'wait');

/**
 * Width of an element in CSS pixels, so a drawing can use it as its viewBox and keep
 * its text at reading size on a phone. (Chapter 1 has the same helper; a shared one
 * would serve both.)
 */
function useWidth(initial: number, min = 280, max = 640): [RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(initial);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const read = () => setW(Math.max(min, Math.min(max, Math.round(el.clientWidth))));
    read();
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, [min, max]);
  return [ref, w];
}

// ---------------------------------------------------------------- barrier timeline
const AXIS = 5.5; // microseconds shown

/** Fig. 4.3 redrawn from the model: work, then waiting, then everyone continues together. */
function TimelineView({ durations, onChange }: { durations: readonly number[]; onChange?: (i: number, v: number) => void }) {
  const tl = timeline(durations);
  const hatch = useId();
  const svg = useRef<SVGSVGElement>(null);
  const [drag, setDrag] = useState<number | null>(null);
  const [box, W] = useWidth(600, 290, 600);
  const narrow = W < 440;
  const L = narrow ? 54 : 60;
  const R = narrow ? 84 : 92;
  const row = 24;
  const H = durations.length * row + 30;
  const x = (t: number) => L + (t / AXIS) * (W - L - R);
  const timeAt = (clientX: number) => {
    const r = svg.current!.getBoundingClientRect();
    const px = (clientX - r.left) * (W / r.width);
    return Math.max(0.1, Math.min(5, round1(((px - L) / (W - L - R)) * AXIS)));
  };
  return (
    <div ref={box}>
    <svg ref={svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Threads reaching a barrier at different times', 'Các luồng đến rào chắn vào những thời điểm khác nhau')}
      style={{ width: '100%', maxWidth: W, display: 'block', touchAction: onChange ? 'none' : undefined }}
      onPointerMove={(e) => { if (drag != null) onChange?.(drag, timeAt(e.clientX)); }}
      onPointerUp={() => setDrag(null)} onPointerLeave={() => setDrag(null)}>
      <defs>
        <pattern id={hatch} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="6" stroke="var(--ink-3)" strokeWidth="2" />
        </pattern>
      </defs>
      {[0, 1, 2, 3, 4, 5].map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={0} y2={durations.length * row} stroke="var(--rule)" />
          <text x={x(t)} y={durations.length * row + 13} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{t}</text>
        </g>
      ))}
      <text x={x(AXIS)} y={durations.length * row + 26} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi="thời gian, µs">time, µs</Translated></text>
      {durations.map((d, i) => {
        const y = i * row + 4;
        return (
          <g key={i} data-thread={i}>
            <text x={L - 8} y={y + 11.5} fontSize="11" textAnchor="end"><Translated vi={<>luồng {i}</>}>thread {i}</Translated></text>
            <rect x={x(0)} y={y} width={x(d) - x(0)} height={row - 9} fill="var(--ink-2)" />
            {tl.waits[i] > 0 ? (
              <rect x={x(d)} y={y} width={x(tl.release) - x(d)} height={row - 9} fill={`url(#${hatch})`} stroke="var(--ink-3)" strokeWidth="0.6" />
            ) : null}
            <rect x={x(tl.release) + 2} y={y + 3} width={x(AXIS) - x(tl.release) - 2} height={row - 15} fill="var(--ink-2)" opacity="0.28" />
            <text x={W - R + 8} y={y + 11.5} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
              {tl.waits[i] > 0 ? tr(`waits ${f1(tl.waits[i])}`, `chờ ${f1(tl.waits[i])}`) : tr('last to arrive', 'đến cuối cùng')}
            </text>
            {onChange ? (
              <circle cx={x(d)} cy={y + (row - 9) / 2} r="6.5" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.6" style={{ cursor: 'ew-resize' }}
                onPointerDown={(e) => { e.preventDefault(); setDrag(i); }} />
            ) : null}
          </g>
        );
      })}
      <line x1={x(tl.release)} x2={x(tl.release)} y1={0} y2={durations.length * row} stroke="var(--ink)" strokeWidth="2" />
    </svg>
    </div>
  );
}

const TIMELINE_PRESETS: { label: string; d: number[] }[] = [
  { get label() { return tr('exercise 4', 'bài tập 4'); }, d: [...EXERCISE_4] },
  { get label() { return tr('one slow thread', 'một luồng chậm'); }, d: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0] },
  { get label() { return tr('balanced', 'cân bằng'); }, d: [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5] },
];

function BarrierTimeline() {
  const [d, setD] = useState<number[]>([1.4, 2.2, 1.8, 0.9, 2.6, 1.2, 3.4, 2.0]);
  const tl = timeline(d);
  const set = (i: number, v: number) => setD((old) => old.map((o, k) => (k === i ? v : o)));
  return (
    <Bench
      title={tr('Barrier timeline', 'Dòng thời gian của rào chắn')}
      task={tr('Drag the end of a bar, or type a time. Which thread decides how long the others wait?', 'Kéo đầu một thanh, hoặc gõ một thời gian. Luồng nào quyết định các luồng khác chờ bao lâu?')}
      foot={tr('Solid: working towards the barrier. Hatched: waiting at it. The line is the moment the last thread arrives and all are released. Pale: the next phase, which all start together.', 'Tô đặc: đang làm việc để tới rào chắn. Gạch chéo: đang chờ ở đó. Vạch kẻ là thời điểm luồng cuối cùng tới và tất cả được giải phóng. Nhạt: pha kế tiếp, mọi luồng cùng bắt đầu.')}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="muted"><Translated vi="Đặt tám thời gian thành:">Set the eight times to:</Translated></span>
        {TIMELINE_PRESETS.map((p) => (
          <button key={p.label} onClick={() => setD(p.d)}>{p.label}</button>
        ))}
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 1fr)' }}>
        <div><TimelineView durations={d} onChange={set} /></div>
        <div>
          <div className="mx-label"><Translated vi="Thời gian mỗi luồng cần, µs">Time each thread needs, µs</Translated></div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '0.3rem 0.6rem' }}>
            {d.map((v, i) => (
              <label key={i} className="field" style={{ margin: 0 }}>
                <Translated vi={<>luồng {i}</>}>thread {i}</Translated>
                <input type="number" min={0.1} max={5} step={0.1} value={v} aria-label={tr(`time of thread ${i}`, `thời gian của luồng ${i}`)} style={{ width: '4.6rem' }}
                  onChange={(e) => { const n = Number(e.target.value); if (e.target.value !== '' && isFinite(n)) set(i, Math.max(0.1, Math.min(5, round1(n)))); }} />
              </label>
            ))}
          </div>
        </div>
      </div>
      <p className="summary" role="status" style={{ margin: '0.9rem 0 0' }}>
        <Translated
          vi={
            <>
              Tất cả được giải phóng lúc <b data-k="release">{f1(tl.release)}</b> µs, khi luồng {list(tl.slowest).replace(' and ', ' và ')} tới.
              Thời gian chờ: <b data-k="waiting">{f1(tl.waiting)}</b> trên <b data-k="total">{f1(tl.total)}</b> µs thời gian luồng, tức là{' '}
              <b data-k="share">{f1(tl.share * 100)}%</b>.
            </>
          }
        >
          All released at <b data-k="release">{f1(tl.release)}</b> µs, when {plural(tl.slowest, 'thread')} arrive{tl.slowest.length === 1 ? 's' : ''}.
          Waiting: <b data-k="waiting">{f1(tl.waiting)}</b> of <b data-k="total">{f1(tl.total)}</b> µs of thread time, which is{' '}
          <b data-k="share">{f1(tl.share * 100)}%</b>.
        </Translated>
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- bug hunt
type Mark = 'legal' | 'undefined' | 'depends';
const MARKS: { id: Mark; label: string }[] = [
  { id: 'legal', get label() { return tr('Legal', 'Hợp lệ'); } },
  { id: 'undefined', get label() { return tr('Undefined', 'Không xác định'); } },
  { id: 'depends', get label() { return tr('Depends on whose rule you read', 'Tùy bạn đọc quy tắc của ai'); } },
];
interface Case {
  id: KernelId;
  tab: string;
  /** null: shown for comparison, nothing to mark. */
  verdict: Mark | null;
  why: ReactNode;
}
const CASES: Case[] = [
  {
    id: 'incorrect_barrier',
    tab: '1. Fig. 4.4', // i18n-ignore: a figure number
    verdict: 'undefined', // i18n-ignore: verdict key
    why: (
      <Translated
        vi={
          <>
            Các luồng chẵn gọi rào chắn ở dòng 5, các luồng lẻ gọi rào chắn ở dòng 9. Hai lời gọi là hai rào chắn, và không
            cái nào được cả khối đi tới. Sách gọi kết quả là không xác định; hướng dẫn của NVIDIA nói nó "may hang or produce
            unintended side effects" (có thể treo hoặc gây tác dụng phụ ngoài ý muốn).
          </>
        }
      >
        <>
          Even threads call the barrier on line 5, odd threads the one on line 9. Two calls are two barriers, and neither
          is reached by the whole block. The book calls the result undefined; NVIDIA's guide says it "may hang or produce
          unintended side effects".
        </>
      </Translated>
    ),
  },
  {
    id: 'barrier_uniform',
    tab: '2. if (blockIdx.x > 0)',
    verdict: 'legal', // i18n-ignore: verdict key
    why: (
      <Translated
        vi={
          <>
            <code>blockIdx.x</code> giống nhau với mọi luồng của một khối, nên trong mỗi khối, hoặc mọi luồng gọi rào chắn
            hoặc không luồng nào gọi. Hướng dẫn của NVIDIA dùng chính điều kiện này làm ví dụ đúng. Hợp lệ không có nghĩa là
            đúng: hãy chạy khối 0 trong trình chạy từng bước và xem cái gì ngăn cách hai pha của nó.
          </>
        }
      >
        <>
          <code>blockIdx.x</code> is the same for every thread of a block, so in any one block all threads call the
          barrier or none does. NVIDIA's guide uses this very condition as its correct example. Legal is not the same as
          right: run block 0 in the stepper and see what separates its two phases.
        </>
      </Translated>
    ),
  },
  {
    id: 'barrier_per_thread',
    tab: '3. if (threadIdx.x > 0)',
    verdict: 'undefined', // i18n-ignore: verdict key
    why: (
      <Translated
        vi={
          <>
            Điều kiện sai với luồng 0 và đúng với phần còn lại của khối. Hướng dẫn của NVIDIA dùng chính điều kiện này làm ví
            dụ sai. Luồng 0 đi tiếp sang pha thứ hai mà không chờ ai.
          </>
        }
      >
        <>
          The condition is false for thread 0 and true for the rest of the block. NVIDIA's guide uses this very condition
          as its wrong example. Thread 0 goes on to the second phase without waiting for anyone.
        </>
      </Translated>
    ),
  },
  {
    id: 'barrier_early_return',
    get tab() { return tr('4. early return', '4. return sớm'); },
    verdict: 'depends', // i18n-ignore: verdict key
    why: (
      <Translated
        vi={
          <>
            Theo quy tắc của sách (được mọi luồng của khối thực thi, hoặc không luồng nào) thì đây là vi phạm: luồng đã
            return không bao giờ thực thi lời gọi. Hướng dẫn của NVIDIA chờ "all non-exited threads" (mọi luồng chưa thoát),
            và đặc tả của Apple nói một luồng đã kết thúc không chặn rào chắn trên Apple silicon, nên ở đó điều này được phép.
            Các trang của AMD không nói. Mã phải chạy ở mọi nơi thì không dựa vào điều này: xem bản viết lại ở tab cuối.
          </>
        }
      >
        <>
          By the book's rule (executed by all threads of the block, or by none) it is a violation: the thread that
          returned never executes the call. NVIDIA's guide waits for "all non-exited threads", and Apple's specification
          says a thread that has ended does not block a barrier on Apple silicon, so there it is allowed. AMD's pages do
          not say. Code that must run everywhere does not rely on it: see the rewrite in the last tab.
        </>
      </Translated>
    ),
  },
  {
    id: 'barrier_loop',
    get tab() { return tr('5. barrier in a loop', '5. rào chắn trong vòng lặp'); },
    verdict: 'undefined', // i18n-ignore: verdict key
    why: (
      <Translated
        vi={
          <>
            Điều kiện vòng lặp cũng là một điều kiện, và điều kiện này khác nhau bên trong khối. Đặc tả của Apple nói thẳng
            điều này cho vòng lặp: trong mỗi lần lặp, nếu có luồng nào thực thi rào chắn thì tất cả đều phải thực thi. Ở vòng
            thứ hai chỉ các luồng lẻ gọi nó, còn các luồng chẵn đã đang đọc những ô mà các luồng lẻ sắp ghi đè.
          </>
        }
      >
        <>
          A loop condition is a condition, and this one differs inside the block. Apple's specification says it outright
          for loops: in each iteration, if any thread executes the barrier, all of them must. In the second round only
          the odd threads call it, and the even threads are already reading slots the odd ones are about to overwrite.
        </>
      </Translated>
    ),
  },
  {
    id: 'barrier_guarded',
    get tab() { return tr('4, rewritten', '4, viết lại'); },
    verdict: null,
    why: (
      <Translated
        vi={
          <>
            Phép kiểm tra phạm vi giờ bảo vệ công việc chứ không bảo vệ rào chắn. Mọi luồng của khối đều tới dòng 6, nên
            kernel hợp lệ theo mọi cách diễn đạt, với cái giá là các luồng ngoài phạm vi phải ở lại đến cuối.
          </>
        }
      >
        <>
          The range test now guards the work and not the barrier. Every thread of the block reaches line 6, so the kernel
          is legal under every wording, at the cost of out-of-range threads staying until the end.
        </>
      </Translated>
    ),
  },
];
const FINDING: Record<Finding, { cls: string; text: string }> = {
  'stuck-sites': { cls: 'fail', get text() { return tr('stuck: threads wait at two different calls', 'kẹt: các luồng chờ ở hai lời gọi khác nhau'); } },
  'stuck-ended': { cls: 'fail', get text() { return tr('stuck: waiting for a thread that has ended', 'kẹt: đang chờ một luồng đã kết thúc'); } },
  'ran-ahead': { cls: 'fail', get text() { return tr('finished, but a thread started a phase too early', 'đã xong, nhưng một luồng bắt đầu một pha quá sớm'); } },
  partial: { cls: '', get text() { return tr('finished: ended threads were not waited for', 'đã xong: không chờ các luồng đã kết thúc'); } },
  clean: { cls: 'ok', get text() { return tr('finished: every thread met at every barrier', 'đã xong: mọi luồng gặp nhau ở mọi rào chắn'); } },
};
const RULES: { id: Rule; label: string }[] = [
  { id: 'all-arrive', get label() { return tr('every thread must arrive (the book)', 'mọi luồng phải tới (sách)'); } },
  { id: 'exited-excluded', get label() { return tr('ended threads do not block (NVIDIA, Apple)', 'luồng đã kết thúc không chặn (NVIDIA, Apple)'); } },
];

/** One sentence for a step of the barrier model. */
function tell(s: Step, rule: Rule): string {
  if (s.kind === 'advance') {
    const me = s.states[s.thread!];
    const ran = s.lines!.length ? tr(`runs line${s.lines!.length > 1 ? 's' : ''} ${list(s.lines!)}`, `chạy dòng ${list(s.lines!).replace(' and ', ' và ')}`) : tr('has nothing left to run', 'không còn gì để chạy');
    const end = me.status === 'waiting' ? tr(` and waits at the call on line ${me.site}`, ` và chờ ở lời gọi tại dòng ${me.site}`) : tr(' and ends', ' và kết thúc');
    const ahead = s.ahead
      ? tr(` It started phase ${s.ahead.phase} while ${plural(s.ahead.behind, 'thread')} still had earlier work to do: nothing separated the phases.`, ` Nó bắt đầu pha ${s.ahead.phase} trong khi luồng ${list(s.ahead.behind).replace(' and ', ' và ')} vẫn còn việc trước đó: không có gì ngăn cách các pha.`)
      : '';
    return tr(`Thread ${s.thread} ${ran}${end}.${ahead}`, `Luồng ${s.thread} ${ran}${end}.${ahead}`);
  }
  if (s.kind === 'release') {
    if (!s.absent!.length) return tr(`Every thread of the block is at the call on line ${s.site}. The barrier opens and all of them continue.`, `Mọi luồng của khối đều ở lời gọi tại dòng ${s.site}. Rào chắn mở và tất cả cùng đi tiếp.`);
    return tr(`${plural(s.released!, 'Thread')} ${wait(s.released!)} at the call on line ${s.site}. ${plural(s.absent!, 'Thread')} ended without reaching it. Under this rule a thread that has ended does not count, so the barrier opens.`, `Luồng ${list(s.released!).replace(' and ', ' và ')} chờ ở lời gọi tại dòng ${s.site}. Luồng ${list(s.absent!).replace(' and ', ' và ')} đã kết thúc mà không tới đó. Theo quy tắc này, một luồng đã kết thúc không được tính, nên rào chắn mở.`);
  }
  if (s.kind === 'stuck') {
    if (s.reason === 'different-sites')
      return tr(`Nothing can move. ${s.waiting.map((g, i) => `${plural(g.threads, i ? 'thread' : 'Thread')} ${wait(g.threads)} at line ${g.site}`).join('; ')}. Two calls are two barriers, and neither will ever have the whole block.`, `Không gì có thể di chuyển. ${s.waiting.map((g, i) => `${i ? 'luồng' : 'Luồng'} ${list(g.threads).replace(' and ', ' và ')} chờ ở dòng ${g.site}`).join('; ')}. Hai lời gọi là hai rào chắn, và không cái nào sẽ có đủ cả khối.`);
    return tr(`Nothing can move. ${plural(s.waiting[0].threads, 'Thread')} ${wait(s.waiting[0].threads)} at line ${s.waiting[0].site} for ${plural(s.absent!, 'thread')}, which ended without reaching the call${rule === 'all-arrive' ? ' and never will' : ''}.`, `Không gì có thể di chuyển. Luồng ${list(s.waiting[0].threads).replace(' and ', ' và ')} chờ ở dòng ${s.waiting[0].site} cho luồng ${list(s.absent!).replace(' and ', ' và ')}, vốn đã kết thúc mà không tới lời gọi${rule === 'all-arrive' ? ' và sẽ không bao giờ tới' : ''}.`);
  }
  return tr('Every thread has ended.', 'Mọi luồng đã kết thúc.');
}

function BugHunt() {
  const [sel, setSel] = useState(0);
  const [marks, setMarks] = useState<Partial<Record<KernelId, Mark>>>({});
  const [draft, setDraft] = useState<Mark | null>(null);
  const [rule, setRule] = useState<Rule>('all-arrive');
  const [order, setOrder] = useState<Order>('ascending');
  const [block, setBlock] = useState(1);
  const [n, setN] = useState(0);

  const c = CASES[sel];
  const marked = c.verdict == null || marks[c.id] != null;
  const result = useMemo(() => run(KERNELS[c.id].build(4, block), rule, order), [c.id, block, rule, order]);
  const total = result.steps.length;
  const at = Math.min(n, total);
  const step = at > 0 ? result.steps[at - 1] : undefined;
  const states = step?.states;
  useEffect(() => setN(0), [sel, rule, order, block]);
  useEffect(() => setDraft(null), [sel]);

  const hl = !step ? [] : step.kind === 'advance' ? step.lines! : step.kind === 'release' ? [step.site!] : step.waiting.map((g) => g.site);
  const done = at >= total;
  const found = FINDING[finding(result)];
  const score = CASES.filter((k) => k.verdict != null && marks[k.id] === k.verdict).length;
  const answered = CASES.filter((k) => k.verdict != null && marks[k.id] != null).length;

  return (
    <Bench
      title={tr('Is this barrier legal?', 'Rào chắn này có hợp lệ không?')}
      task={tr('Mark each kernel first. Then step it, under both readings of the rule.', 'Đánh giá từng kernel trước. Rồi chạy từng bước, theo cả hai cách đọc quy tắc.')}
      foot={tr("Each kernel runs in a block of 4 threads. In kernels 4 and '4, rewritten' the last thread of the block is past the end of the data.", "Mỗi kernel chạy trong một khối 4 luồng. Ở kernel 4 và '4, viết lại', luồng cuối cùng của khối nằm quá cuối dữ liệu.")}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('Kernel', 'Kernel')} style={{ flexWrap: 'wrap', rowGap: '0.3rem' }}>
          {CASES.map((k, i) => (
            <button key={k.id} aria-pressed={i === sel} onClick={() => setSel(i)}>
              {k.tab}
              {k.verdict != null && marks[k.id] != null ? (marks[k.id] === k.verdict ? ' ✓' : ' ✗') : ''}
            </button>
          ))}
        </span>
        <span className="muted score" aria-live="polite"><Translated vi={<>đã đánh giá {answered} trên 5, đúng {score}</>}>{answered} of 5 marked, {score} right</Translated></span>
      </div>
      {/* The listing gets the full width: its comments say which phase each line belongs to. */}
      <Code tabs={[{ label: c.id + '.cu', code: SNIPPETS[c.id], tiers: TIERS, status: statusOf(c.id) }]} highlight={hl} />
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 2fr)' }}>
        <div>
          {c.verdict != null && !marks[c.id] ? (
            <div className="panel mark">
              <p style={{ margin: '0 0 0.5rem' }}><b><Translated vi="Bạn đánh giá rào chắn này thế nào?">Your verdict on this barrier?</Translated></b></p>
              <div className="q" style={{ margin: 0, padding: 0, border: 0 }}>
                <div className="opts" role="radiogroup">
                  {MARKS.map((m) => (
                    <label key={m.id}>
                      <input type="radio" name={'mark-' + c.id} checked={draft === m.id} onChange={() => setDraft(m.id)} />
                      <span>{m.label}</span>
                    </label>
                  ))}
                </div>
              </div>
              <button className="primary" disabled={!draft} onClick={() => draft && setMarks({ ...marks, [c.id]: draft })}><Translated vi="Chốt đánh giá">Mark it</Translated></button>
            </div>
          ) : (
            <div className="panel verdict-box" data-verdict={c.verdict ?? 'none'}>
              {c.verdict != null ? (
                <p style={{ margin: '0 0 0.4rem' }}>
                  <span className={'chip ' + (marks[c.id] === c.verdict ? 'ok' : 'fail')}>
                    {marks[c.id] === c.verdict ? tr('Right', 'Đúng') : tr('Not quite', 'Chưa đúng')}
                  </span>{' '}
                  <b>{MARKS.find((m) => m.id === c.verdict)!.label}.</b>
                </p>
              ) : (
                <p style={{ margin: '0 0 0.4rem' }}><b><Translated vi="Để so sánh.">For comparison.</Translated></b></p>
              )}
              <p style={{ margin: 0 }}>{c.why}</p>
            </div>
          )}
        </div>
        <div>
          {marked ? (
            <table className="threads" aria-label={tr('Threads of the block', 'Các luồng của khối')} style={{ width: '100%' }}>
              <thead>
                <tr><th><Translated vi="luồng">thread</Translated></th><th><Translated vi="trạng thái">state</Translated></th></tr>
              </thead>
              <tbody>
                {[0, 1, 2, 3].map((t) => {
                  const st = states?.[t];
                  const txt = !st || (st.status === 'ready' && st.line == null) ? tr('not started', 'chưa bắt đầu')
                    : st.status === 'waiting' ? tr(`waiting at line ${st.site}`, `đang chờ ở dòng ${st.site}`)
                    : st.status === 'ended' ? tr('ended', 'đã kết thúc')
                    : tr('released, ready to run', 'đã được giải phóng, sẵn sàng chạy');
                  return (
                    <tr key={t} data-thread={t} data-status={st?.status ?? 'ready'} className={step?.kind === 'advance' && step.thread === t ? 'cur' : ''}>
                      <td><Translated vi={<>luồng {t}</>}>thread {t}</Translated></td>
                      <td>{txt}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          ) : null}
        </div>
      </div>
      {marked ? (
        <>
          <div className="row" style={{ margin: '0.9rem 0 0.7rem' }}>
            <span className="seg" role="group" aria-label={tr('Rule', 'Quy tắc')}>
              {RULES.map((r) => <button key={r.id} aria-pressed={r.id === rule} onClick={() => setRule(r.id)}>{r.label}</button>)}
            </span>
            <span className="seg" role="group" aria-label={tr('Thread order', 'Thứ tự luồng')}>
              <button aria-pressed={order === 'ascending'} onClick={() => setOrder('ascending')}><Translated vi="luồng 0 chạy trước">thread 0 runs first</Translated></button>
              <button aria-pressed={order === 'descending'} onClick={() => setOrder('descending')}><Translated vi="luồng 3 chạy trước">thread 3 runs first</Translated></button>
            </span>
            {c.id === 'barrier_uniform' ? (
              <label className="field">
                <Translated vi="Khối">Block</Translated>
                <select value={block} onChange={(e) => setBlock(+e.target.value)} aria-label={tr('Block', 'Khối')}>
                  <option value={1}>blockIdx.x = 1</option>
                  <option value={0}>blockIdx.x = 0</option>
                </select>
              </label>
            ) : null}
          </div>
          <div className="row" style={{ marginBottom: '0.7rem' }}>
            <button onClick={() => setN(0)} disabled={at === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
            <button onClick={() => setN(at - 1)} disabled={at === 0}><Translated vi="Lùi">Back</Translated></button>
            <button className="primary" onClick={() => setN(at + 1)} disabled={done}><Translated vi="Bước">Step</Translated></button>
            <button onClick={() => setN(total)} disabled={done}><Translated vi="Chạy đến cuối">Run to the end</Translated></button>
            <span className="muted" aria-live="polite"><Translated vi={<>bước {at} trên {total}</>}>step {at} of {total}</Translated></span>
            {done ? <span className={'chip outcome ' + found.cls} data-finding={finding(result)}>{found.text}</span> : null}
          </div>
          <div className={'narrate' + (step && (step.kind === 'stuck' || step.ahead) ? ' bad' : '')} role="status">
            {step ? tell(step, rule) : tr('Nothing has run yet. Each step lets one thread run until it reaches a barrier call or ends; when no thread can run, the waiting threads are released or found stuck.', 'Chưa có gì chạy. Mỗi bước cho một luồng chạy đến khi nó tới một lời gọi rào chắn hoặc kết thúc; khi không luồng nào chạy được, các luồng đang chờ được giải phóng hoặc bị phát hiện là kẹt.')}
          </div>
        </>
      ) : (
        <p className="muted" style={{ margin: '0.9rem 0 0' }}><Translated vi="Hãy đánh giá trước, rồi trình chạy từng bước cho kernel này sẽ mở.">Mark it, then the stepper for this kernel opens.</Translated></p>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- waves explorer
const MAX_WAVE_BLOCKS = 20000;
const intIn = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, Math.round(Number.isFinite(v) ? v : lo)));

function WavesExplorer() {
  const { reports } = useReports();
  const [blocks, setBlocks] = useState<number>(TAIL_EXAMPLE.blocks);
  const [units, setUnits] = useState<number>(TAIL_EXAMPLE.units);
  const [slots, setSlots] = useState<number>(TAIL_EXAMPLE.slotsPerUnit);
  const [uneven, setUneven] = useState(false);

  const cap = capacity(units, slots);
  const s = useMemo(
    () => schedule({ blocks, units, slotsPerUnit: slots, durations: uneven ? unevenDurations(blocks) : equalDurations(blocks) }),
    [blocks, units, slots, uneven],
  );
  const prof = useMemo(() => profile(s), [s]);
  const u = usage(s);
  const tl = tail(blocks, cap);
  const sizes = waveSizes(blocks, cap);
  const w = waves(blocks, cap);
  const mine = Object.keys(reports).filter(isGpuBackend).map((b) => deviceFromReport(b, reports[b].device)).filter((d) => d.units != null);
  const preset = (b: number, un: number, sl: number) => { setBlocks(b); setUnits(un); setSlots(sl); };

  const [box, W] = useWidth(600, 290, 600);
  const H = 190;
  const L = 44;
  const top = 16;
  const span = Math.max(1, s.makespan);
  const x = (t: number) => L + (t / span) * (W - L - 10);
  const y = (r: number) => top + (1 - r / Math.max(1, cap)) * (H - top - 26);
  const hatch = useId();
  const sizeText = sizes.length <= 6 ? sizes.join(', ') : `${sizes.slice(0, 3).join(', ')}, … , ${sizes[sizes.length - 1]}`;

  return (
    <Bench
      title={tr('Waves and the tail', 'Các đợt và phần đuôi')}
      task={tr('Change the block count until the last wave is full. Then make the block times uneven.', 'Đổi số khối cho đến khi đợt cuối đầy. Rồi làm cho thời gian các khối không đều.')}
      foot={tr('Time is in block-times: an equal block takes 1, an uneven one 1, 2, 3 or 4 (seeded). Hatched: slots with no block to run.', 'Thời gian tính bằng đơn vị thời gian khối: một khối đều tốn 1, một khối không đều tốn 1, 2, 3 hoặc 4 (có seed cố định). Gạch chéo: các ô không có khối nào để chạy.')}
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="muted"><Translated vi="Bắt đầu từ:">Start from:</Translated></span>
        <button onClick={() => preset(FIG45.blocks, FIG45.small.units, FIG45.small.slotsPerUnit)}>{tr('Fig. 4.5, small device', 'Fig. 4.5, thiết bị nhỏ')}</button>
        <button onClick={() => preset(FIG45.blocks, FIG45.large.units, FIG45.large.slotsPerUnit)}>{tr('Fig. 4.5, large device', 'Fig. 4.5, thiết bị lớn')}</button>
        <button onClick={() => preset(TAIL_EXAMPLE.blocks, TAIL_EXAMPLE.units, TAIL_EXAMPLE.slotsPerUnit)}><Translated vi="660 trên 264 của sách">the book's 660 on 264</Translated></button>
        {mine.map((d) => (
          <button key={d.name} onClick={() => setUnits(intIn(d.units!, 1, 4096))}><Translated vi={<>{d.name} của bạn: {fmt(d.units!, 6)} từ multiProcessorCount</>}>your {d.name}: {fmt(d.units!, 6)} from multiProcessorCount</Translated></button>
        ))}
      </div>
      {!mine.length ? (
        <Translated
          vi={
            <p className="muted own-device" style={{ margin: '0 0 0.7rem' }}>
              Không có báo cáo thiết bị nào trên máy này chứa số SM, nên không có cấu hình mẫu cho thiết bị của bạn. Một
              thiết bị Metal không có số đó để báo cáo.
            </p>
          }
        >
          <p className="muted own-device" style={{ margin: '0 0 0.7rem' }}>
            No device report on this machine carries an SM count, so there is no preset for your device. A Metal device
            has none to report.
          </p>
        </Translated>
      ) : null}
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          <Translated vi="Số khối trong lưới">Blocks in the grid</Translated>
          <input type="number" min={1} max={MAX_WAVE_BLOCKS} step={1} value={blocks} aria-label={tr('Blocks in the grid', 'Số khối trong lưới')} onChange={(e) => setBlocks(intIn(+e.target.value, 1, MAX_WAVE_BLOCKS))} />
        </label>
        <label className="field">
          <Translated vi="Số SM">SMs</Translated>
          <input type="number" min={1} max={4096} step={1} value={units} aria-label={tr('SMs', 'Số SM')} onChange={(e) => setUnits(intIn(+e.target.value, 1, 4096))} />
        </label>
        <label className="field">
          <Translated vi="Số khối mỗi SM, với kernel này">Blocks per SM, for this kernel</Translated>
          <input type="number" min={1} max={64} step={1} value={slots} aria-label={tr('Blocks per SM', 'Số khối mỗi SM')} onChange={(e) => setSlots(intIn(+e.target.value, 1, 64))} />
        </label>
        <button onClick={() => setBlocks(intIn(Math.ceil(blocks / cap) * cap, 1, MAX_WAVE_BLOCKS))} disabled={blocks % cap === 0}><Translated vi="Làm tròn lưới lên thành các đợt đầy">Round the grid up to full waves</Translated></button>
        <span className="seg" role="group" aria-label={tr('Block times', 'Thời gian khối')}>
          <button aria-pressed={!uneven} onClick={() => setUneven(false)}><Translated vi="thời gian khối đều">equal block times</Translated></button>
          <button aria-pressed={uneven} onClick={() => setUneven(true)}><Translated vi="thời gian khối không đều">uneven block times</Translated></button>
        </span>
      </div>
      <div className="cols plot">
        <div ref={box}>
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Blocks in progress over time', 'Số khối đang chạy theo thời gian')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
            <defs>
              <pattern id={hatch} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <line x1="0" y1="0" x2="0" y2="6" stroke="var(--ink-3)" strokeWidth="1.6" />
              </pattern>
            </defs>
            <rect x={x(0)} y={y(cap)} width={x(span) - x(0)} height={y(0) - y(cap)} fill={`url(#${hatch})`} />
            {prof.map((g) => (
              <rect key={g.from} x={x(g.from)} y={y(g.resident)} width={Math.max(0.4, x(g.to) - x(g.from) - (prof.length <= 40 ? 1.5 : 0))} height={y(0) - y(g.resident)} fill="var(--ink-2)" />
            ))}
            <line x1={x(0)} x2={x(span)} y1={y(cap)} y2={y(cap)} stroke="var(--ink)" strokeDasharray="5 3" />
            <line x1={x(0)} x2={x(span)} y1={y(0)} y2={y(0)} stroke="var(--ink-3)" />
            <text x={L - 6} y={y(cap) + 4} fontSize="10.5" textAnchor="end">{fmt(cap, 6)}</text>
            <text x={L - 6} y={y(0) + 4} fontSize="10.5" textAnchor="end">0</text>
            <text x={x(0)} y={H - 6} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="khởi chạy">launch</Translated></text>
            <text x={x(span)} y={H - 6} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>khối cuối kết thúc, thời điểm {fmt(s.makespan, 6)}</>}>last block ends, time {fmt(s.makespan, 6)}</Translated></text>
            <text x={x(0) + 4} y={top - 4} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="số khối đang chạy">blocks in progress</Translated></text>
          </svg>
        </div>
        <div className="panel stats" role="status">
          <p style={{ margin: '0 0 0.4rem' }}>
            <Translated vi={<>Chạy cùng lúc: {fmt(units, 6)} × {slots} = <b data-k="capacity">{fmt(cap, 6)}</b> khối</>}>
              Runs at once: {fmt(units, 6)} × {slots} = <b data-k="capacity">{fmt(cap, 6)}</b> blocks
            </Translated>
          </p>
          <p style={{ margin: '0 0 0.4rem' }}>
            <Translated vi={<>Số đợt, theo cách đếm của sách: {fmt(blocks, 6)} / {fmt(cap, 6)} = <b data-k="waves">{fmt(w, 4)}</b></>}>
              Waves, by the book's count: {fmt(blocks, 6)} / {fmt(cap, 6)} = <b data-k="waves">{fmt(w, 4)}</b>
            </Translated>
          </p>
          {uneven ? (
            <p style={{ margin: '0 0 0.4rem' }}>
              <Translated vi="Khi thời gian không đều, các khối không còn bắt đầu cùng nhau, nên không có đợt rõ ràng nào để đếm.">
                With uneven times the blocks no longer start together, so there are no clean waves to count.
              </Translated>
            </p>
          ) : (
            <>
              <p style={{ margin: '0 0 0.4rem' }}><Translated vi="Số khối mỗi đợt:">Blocks per wave:</Translated> <span data-k="sizes">{sizeText}</span></p>
              <p style={{ margin: '0 0 0.4rem' }}>
                <Translated vi={<>Đợt cuối: <b data-k="idle">{fmt(tl.idle, 6)}</b> trên {fmt(cap, 6)} ô ngồi không, <b data-k="idle-share">{fmt(tl.idleShare * 100, 3)}%</b></>}>
                  Last wave: <b data-k="idle">{fmt(tl.idle, 6)}</b> of {fmt(cap, 6)} slots idle, <b data-k="idle-share">{fmt(tl.idleShare * 100, 3)}%</b>
                </Translated>{' '}
                {tl.idle === 0 ? <span className="chip ok"><Translated vi="không có đuôi">no tail</Translated></span> : null}
              </p>
            </>
          )}
          <p style={{ margin: 0 }}>
            <Translated vi={<>Từ lúc khởi chạy đến khi khối cuối kết thúc, thiết bị đã dùng <b data-k="utilisation">{fmt(u.utilisation * 100, 3)}%</b> sức chứa khối của nó.</>}>
              From launch to the last block's end the device used <b data-k="utilisation">{fmt(u.utilisation * 100, 3)}%</b> of its block capacity.
            </Translated>
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const TAIL_Q = tail(1000, 264);
export const questionsS43: Record<string, Question> = {
  'ch04/4.3/q-fig44': {
    kind: 'choice',
    id: 'ch04/4.3/q-fig44',
    prompt: (
      <Translated vi={<>Kernel của Fig. 4.4, ở trên, chạy trong một khối 4 luồng. Ai chờ ở đâu?</>}>
        <>The kernel of Fig. 4.4, above, runs in a block of 4 threads. Who waits where?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Cả bốn chờ ở lời gọi tại dòng 5, rồi sau đó cả bốn chờ ở lời gọi tại dòng 9.">
            {'All four wait at the call on line 5, and later all four at the call on line 9.'}
          </Translated>
        ),
        why: (
          <Translated vi="Một luồng chỉ đi một nhánh. Không luồng nào tới được cả hai lời gọi.">
            {'A thread takes one branch. No thread reaches both calls.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không ai chờ lâu: mỗi cặp đủ mặt ở rào chắn của riêng nó, nên cả hai cặp cùng được giải phóng.">
            {'Nobody waits for long: each pair is complete at its own barrier, so both pairs are released at once.'}
          </Translated>
        ),
        why: (
          <Translated vi="Một rào chắn đếm cả khối, không chỉ những luồng tình cờ đi vào nhánh của nó.">
            {'A barrier counts the whole block, not the threads that happened to take its branch.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Luồng 0 và 2 chờ ở lời gọi tại dòng 5, luồng 1 và 3 ở dòng 9, và không lời gọi nào có đủ cả bốn.">
            {'Threads 0 and 2 wait at the call on line 5, 1 and 3 at line 9, and neither call ever gets all four.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Trình biên dịch gộp hai lời gọi thành một rào chắn sau if, nên cả bốn gặp nhau ở đó.">
            {'The compiler merges the two calls into one barrier after the if, so all four meet there.'}
          </Translated>
        ),
        why: (
          <Translated vi="Nó không làm vậy. Hai lời gọi trong mã nguồn là hai rào chắn.">
            {'It does not. Two calls in the source are two barriers.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hai lời gọi trông đối xứng, nhưng mỗi cái là một rào chắn riêng, và mỗi cái cần mọi luồng của khối. Các luồng chẵn
            sẽ không bao giờ thấy một luồng lẻ tới dòng 5, và ngược lại. Sách gọi kết quả là hành vi không xác định và cảnh
            báo về deadlock; hướng dẫn của NVIDIA nói về việc các luồng tới "the same" (cùng một) lời gọi.
          </p>
        }
      >
        <p>
          The two calls look symmetrical, but each is its own barrier, and each needs every thread of the block. The even
          threads will never see an odd thread arrive at line 5, and the other way round. The book calls the result
          undefined behaviour and warns of deadlock; NVIDIA's guide speaks of threads reaching "the same" call.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.3, p. 72 (Fig. 4.4); notes/today/ch04-facts.md A6', // i18n-ignore: citation
  },
  'ch04/4.3/q-uniform': {
    kind: 'choice',
    id: 'ch04/4.3/q-uniform',
    prompt: (
      <Translated
        vi={
          <>
            <code>i</code> là chỉ số toàn cục của luồng, <code>blockIdx.x * blockDim.x + threadIdx.x</code>, và{' '}
            <code>n</code> không phải bội số của kích thước khối. Rào chắn có điều kiện nào dưới đây là hợp lệ?
          </>
        }
      >
        <>
          <code>i</code> is the thread's global index, <code>blockIdx.x * blockDim.x + threadIdx.x</code>, and{' '}
          <code>n</code> is not a multiple of the block size. Which of these conditional barriers is legal?
        </>
      </Translated>
    ),
    options: [
      { label: <code>if (threadIdx.x &lt; blockDim.x / 2) __syncthreads();</code>, why: <Translated vi="Đúng với một nửa mỗi khối và sai với nửa còn lại.">{'True for half of each block and false for the other half.'}</Translated> },
      { label: <code>if (i &lt; n) __syncthreads();</code>, why: <Translated vi="Ở khối cuối, một số luồng qua được phép kiểm tra và một số thì không.">{'In the last block some threads pass the test and some do not.'}</Translated> },
      { label: <code>if (blockIdx.x % 2 == 0) __syncthreads();</code>, correct: true },
      { label: <code>if (data[i] &gt; 0) __syncthreads();</code>, why: <Translated vi="Một giá trị đọc theo từng luồng khác nhau giữa các luồng của một khối.">{'A value read per thread differs between threads of a block.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Phép thử cần áp dụng: hai luồng của cùng một khối có thể nhận câu trả lời khác nhau không?{' '}
            <code>blockIdx.x % 2</code> không thể khác nhau trong một khối, nên mỗi khối hoặc gặp nhau trọn vẹn ở rào chắn hoặc
            bỏ qua nó trọn vẹn. Liệu các khối lẻ sau đó có tính đúng mà không cần rào chắn hay không là một câu hỏi riêng, và
            câu trả lời thường là không.
          </p>
        }
      >
        <p>
          The test to apply: can two threads of the same block get different answers? <code>blockIdx.x % 2</code> cannot
          differ inside a block, so each block either meets at the barrier as a whole or skips it as a whole. Whether the
          odd blocks then compute the right thing without a barrier is a separate question, and the answer is usually no.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.3, p. 72; NVIDIA\'s wording in notes/today/ch04-facts.md A6',
    vi: { source: 'ch. 4 §4.3, p. 72; cách diễn đạt của NVIDIA trong notes/today/ch04-facts.md A6' },
  },
  'ch04/4.3/q-tail': {
    kind: 'numeric',
    id: 'ch04/4.3/q-tail',
    prompt: (
      <Translated
        vi={
          <>
            Một lưới có 1000 khối, tất cả tốn thời gian như nhau. Thiết bị chạy 264 khối cùng lúc. Bao nhiêu phần trăm sức
            chứa khối của nó ngồi không trong đợt cuối?
          </>
        }
      >
        <>
          A grid has 1000 blocks that all take the same time. The device runs 264 of them at once. What percentage of its
          block capacity is idle during the last wave?
        </>
      </Translated>
    ),
    answer: Math.round(TAIL_Q.idleShare * 100),
    tol: 0.02,
    unit: '%',
    explain: (
      <Translated
        vi={
          <p>
            1000 / 264 xấp xỉ 3.79 đợt: ba đợt đầy lấy 792 khối và để lại {TAIL_Q.last} cho đợt thứ tư. Đợt đó để 264 −{' '}
            {TAIL_Q.last} = {TAIL_Q.idle} ô ngồi không, và {TAIL_Q.idle} / 264 xấp xỉ {Math.round(TAIL_Q.idleShare * 100)}%.
            Một lưới 1056 khối, bốn đợt đầy, sẽ không có ô nào ngồi không.
          </p>
        }
      >
        <p>
          1000 / 264 is about 3.79 waves: three full waves take 792 blocks and leave {TAIL_Q.last} for the fourth. That
          wave leaves 264 − {TAIL_Q.last} = {TAIL_Q.idle} slots idle, and {TAIL_Q.idle} / 264 is about{' '}
          {Math.round(TAIL_Q.idleShare * 100)}%. A grid of 1056 blocks, four full waves, would have none.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.3, pp. 73–74 (the method; these numbers are ours)',
    vi: { source: 'ch. 4 §4.3, pp. 73–74 (phương pháp; các con số này là của chúng tôi)' },
  },
  'ch04/4.3/q-gridbarrier': {
    kind: 'choice',
    id: 'ch04/4.3/q-gridbarrier',
    prompt: (
      <Translated vi={<>Một rào chắn toàn lưới giới hạn kích thước lưới. Một rào chắn toàn khối không giới hạn lưới. Vì sao có sự khác biệt?</>}>
        <>A grid-wide barrier limits how large the grid may be. A block-wide barrier puts no limit on the grid. Why the difference?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Rào chắn toàn lưới được cài đặt bằng phần mềm, và chi phí của nó tăng quá nhanh với lưới lớn.">
            {'A grid-wide barrier is implemented in software, and its cost grows too fast for large grids.'}
          </Translated>
        ),
        why: <Translated vi="Nó nặng hơn, nhưng giới hạn không phải về tốc độ.">{'It is heavier, but the limit is not about speed.'}</Translated>,
      },
      {
        label: (
          <Translated vi="Mọi luồng trong phạm vi của rào chắn phải thường trú để tới được nó: một khối luôn thường trú, một lưới lớn thì không.">
            {'Every thread in a barrier\'s scope must be resident to reach it: a block always is, a large grid is not.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Rào chắn toàn lưới cần mọi khối ở trong cùng một cluster, và cluster có giới hạn kích thước.">
            {'A grid-wide barrier needs every block to be in the same cluster, and a cluster has a size limit.'}
          </Translated>
        ),
        why: (
          <Translated vi="Cluster có rào chắn riêng. Rào chắn toàn lưới không đi qua chúng.">
            {'Clusters have their own barrier. The grid-wide one does not go through them.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không có khác biệt: __syncthreads() cũng hỏng khi lưới vượt quá lượng chạy cùng lúc.">
            {'There is no difference: __syncthreads() also fails once the grid exceeds what runs at once.'}
          </Translated>
        ),
        why: (
          <Translated vi="Rào chắn khối chỉ cần chính khối của nó, vốn luôn thường trú trọn vẹn.">
            {'A block barrier only needs its own block, which is always resident as a whole.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một luồng chưa bắt đầu thì không thể tới. Với cách khởi chạy thông thường, phần lớn các khối của một lưới lớn vẫn
            nằm trong danh sách của runtime trong khi những khối đầu tiên chạy, nên một rào chắn trải qua tất cả chúng sẽ
            không bao giờ mở. Đó là lý do rào chắn toàn lưới đi kèm một lời gọi khởi chạy đặc biệt và một lưới không lớn hơn
            lượng thiết bị giữ được cùng lúc, và vì sao rào chắn toàn khối không cần cả hai: runtime chỉ bắt đầu một khối khi
            mọi luồng của nó đều có chỗ.
          </p>
        }
      >
        <p>
          A thread that has not started cannot arrive. With the usual launch, most blocks of a large grid are still on
          the runtime's list while the first ones run, so a barrier across all of them would never open. That is why the
          grid-wide barrier comes with a special launch call and a grid no larger than what the device holds at once, and
          why the block-wide one is free of both: the runtime starts a block only when all its threads have a place.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.3, pp. 70–73; notes/today/ch04-facts.md A7', // i18n-ignore: citation
  },
  'ch04/4.3/q-scal': {
    kind: 'choice',
    id: 'ch04/4.3/q-scal',
    prompt: (
      <Translated vi={<>Thói quen nào phá vỡ khả năng mở rộng trong suốt?</>}>
        <>Which habit breaks transparent scalability?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Hỏi thiết bị số SM của nó và chọn số khối theo đó.">
            {'Asking the device for its SM count and choosing the number of blocks from it.'}
          </Translated>
        ),
        why: (
          <Translated vi="Điều đó tinh chỉnh một lần khởi chạy cho một thiết bị. Kernel vẫn đúng trên mọi thiết bị khác.">
            {'That tunes a launch to a device. The kernel is still correct on any other.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Gọi __syncthreads() nhiều lần trong một kernel, khiến các khối dừng ở nhiều rào chắn.">
            {'Calling __syncthreads() several times in one kernel, so that blocks stop at many barriers.'}
          </Translated>
        ),
        why: (
          <Translated vi="Rào chắn nằm gọn trong một khối, đó là điều kiện cho khả năng mở rộng, không phải mối đe dọa với nó.">
            {'Barriers stay inside a block, which is the condition for scalability, not a threat to it.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Bắt một khối chờ kết quả từ một khối khác của cùng lưới, khối có thể chưa bắt đầu.">
            {'Making one block wait for a result from another block of the same grid that may not have started.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Khởi chạy nhiều khối hơn số thiết bị chạy được cùng lúc, khiến một số khối phải chờ SM trống.">
            {'Launching more blocks than the device can run at once, so that some wait for a free SM.'}
          </Translated>
        ),
        why: (
          <Translated vi="Đó là trường hợp bình thường. Danh sách của runtime hấp thụ nó.">
            {'That is the normal case. The runtime\'s list absorbs it.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Cùng một mã chạy không cần sửa trên mọi thiết bị vì một khối không cần gì từ khối nào khác. Mọi giả định về các
            khối khác, khối nào đã xong, bao nhiêu khối thường trú, chúng ở SM nào, đều là giả định về kích thước của một thiết
            bị và lịch chạy của một lần chạy. Nó có thể đúng trên máy đã được kiểm thử và sai trên một máy lớn hơn hoặc nhỏ
            hơn.
          </p>
        }
      >
        <p>
          The same code runs unchanged on any device because a block needs nothing from any other block. Any assumption
          about the others, which have finished, how many are resident, which SM they are on, is an assumption about one
          device's size and one run's schedule. It may hold on the machine it was tested on and fail on a larger or
          smaller one.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.3, pp. 73–74',
  },
};

/** Blocks in a sampling kernel's grid: one vocabulary, 256 threads per block, `perThread` elements each (ours). */
const SAMPLE_THREADS = 256;
const sampleBlocks = (vocabulary: number, perThread: number) => Math.ceil(vocabulary / (SAMPLE_THREADS * perThread));

export function S43() {
  const q = questionsS43;
  const ex = timeline(EXERCISE_4);
  const exShare = round1(ex.share * 100);
  const cap = capacity(TAIL_EXAMPLE.units, TAIL_EXAMPLE.slotsPerUnit);
  const bookTail = tail(TAIL_EXAMPLE.blocks, cap);
  return (
    <>
      <Translated
        vi={
          <p>
            Các luồng của một khối chạy theo nhịp riêng. Khi một thuật toán có các pha, nơi mọi luồng phải xong một bước trước
            khi bất kỳ luồng nào bắt đầu bước kế tiếp, phải có gì đó giữ các luồng nhanh lại. Trong CUDA, đó là{' '}
            <code>__syncthreads()</code>. Một luồng gọi nó sẽ bị giữ tại lời gọi cho đến khi mọi luồng của khối đều tới đó,
            rồi tất cả được giải phóng cùng nhau. Đây là một <b>rào chắn (barrier)</b>. Hai dấu gạch dưới ở đầu đánh dấu một
            intrinsic: thứ trông như lời gọi hàm và được trình biên dịch biến thẳng thành một lệnh.
          </p>
        }
      >
        <p>
          The threads of a block run at their own pace. When an algorithm has phases, where every thread must finish one
          step before any thread starts the next, something has to hold the fast ones back. In CUDA that is{' '}
          <code>__syncthreads()</code>. A thread that calls it is held at the call until every thread of its block has
          arrived there, and then all are released together. This is a <b>barrier</b>. The two leading underscores mark
          an intrinsic: something that looks like a function call and that the compiler turns straight into an
          instruction.
        </p>
      </Translated>
      <Source section="4.3" pages="70" />
      <Translated
        vi={
          <p>
            Bạn đã biết một thuật toán có các pha. Softmax trên một hàng logits tìm giá trị lớn nhất, rồi cộng các lũy thừa,
            rồi chia. Khi một khối xử lý cả hàng, mỗi pha cần kết quả của mọi luồng từ pha trước, và rào chắn là thứ đứng giữa
            chúng. Chương 10 dựng kernel đó.
          </p>
        }
      >
        <p>
          You already know an algorithm with phases. Softmax over a row of logits finds the maximum, then sums the
          exponentials, then divides. When one block handles the row, each phase needs every thread's result from the
          phase before, and a barrier is what stands between them. Chapter 10 builds that kernel.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Ví dụ softmax là của chúng tôi. Hình ảnh của chính sách là bốn người bạn đi mua sắm riêng và phải cùng quay về
              xe trước khi có ai lái đi (p. 71).
            </p>
          }
        >
          <p>
            The softmax example is ours. The book's own picture is four friends who shop separately and must all be back
            at the car before anyone drives off (p. 71).
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Chờ đợi tốn gì">What waiting costs</Translated></h2>
      <Translated
        vi={
          <p>
            Các luồng tới rào chắn vào những thời điểm khác nhau. Những luồng tới sớm không làm gì cho đến khi luồng cuối cùng
            có mặt. Thời gian ngồi không đó là cái giá của một rào chắn, và một luồng quyết định nó cho cả khối.
          </p>
        }
      >
        <p>
          Threads reach a barrier at different times. Those that arrive early do nothing until the last one is there.
          That idle time is the price of a barrier, and one thread sets it for the whole block.
        </p>
      </Translated>
      <Source section="4.3" pages="71–72" figs={tr('Fig. 4.3, redrawn below; exercise 4, p. 91', 'Fig. 4.3, vẽ lại bên dưới; bài tập 4, p. 91')} />
      <Predict
        id="ch04/4.3/p-waiting"
        prompt={
          <Translated
            vi={
              <>
                Bài tập 4 của sách: tám luồng cần {EXERCISE_4.map((v) => f1(v)).join(', ')} micro giây để tới một rào chắn.
                Bao nhiêu phần trăm tổng thời gian của các luồng, từ lúc bắt đầu đến lúc được giải phóng, là thời gian chờ?
              </>
            }
          >
            <>
              The book's exercise 4: eight threads need {EXERCISE_4.map((v) => f1(v)).join(', ')} microseconds to reach a
              barrier. What percentage of the threads' total time, from start to release, is spent waiting?
            </>
          </Translated>
        }
        unit="%"
      >
        {(v) => (
          <>
            <p>
              <Translated
                vi={
                  <>
                    <b>{f1(exShare)}%</b>. Tất cả được giải phóng lúc {f1(ex.release)} µs, khi luồng chậm nhất tới. Tám luồng
                    nhân {f1(ex.release)} µs là {f1(ex.total)} µs thời gian luồng, trong đó {f1(ex.waiting)} µs là chờ.{' '}
                  </>
                }
              >
                <b>{f1(exShare)}%</b>. All are released at {f1(ex.release)} µs, when the slowest thread arrives. Eight
                threads times {f1(ex.release)} µs is {f1(ex.total)} µs of thread time, of which {f1(ex.waiting)} µs is
                waiting.{' '}
              </Translated>
              {Math.abs(v - exShare) <= 0.5 ? tr('You had it.', 'Bạn đoán đúng.') : tr('Each hatched bar below is one thread\'s share of that.', 'Mỗi thanh gạch chéo bên dưới là phần của một luồng trong đó.')}
            </p>
            <TimelineView durations={EXERCISE_4} />
          </>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Dòng thời gian bên dưới nhận tám thời gian bất kỳ: kéo đầu một thanh hoặc gõ một thời gian, và nó vẽ lúc mỗi luồng
            tới và nó chờ bao lâu.
          </p>
        }
      >
        <p>
          The timeline below takes any eight times: drag the end of a bar or type a time, and it draws when each thread
          arrives and how long it waits.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Dòng thời gian chỉ biết những thời gian bạn đưa cho nó. Nó không nói gì về chi phí của bản thân rào chắn, và
              luồng thật tới theo từng warp, không phải từng luồng một.
            </p>
          }
        >
          <p>
            The timeline knows only the times you give it. It says nothing about what the barrier itself costs, and real
            threads arrive a warp at a time, not one by one.
          </p>
        </Translated>
      </Simplification>
      <BarrierTimeline />
      <Translated
        vi={
          <p>
            Thử "một luồng chậm": bảy luồng dành ba phần tư thời gian để chờ luồng thứ tám. Rồi thử "cân bằng": không ai chờ.
            Một rào chắn rẻ khi pha trước nó giao cho mọi luồng lượng việc xấp xỉ nhau, và đó là điều bạn quyết định khi chia
            việc cho các luồng.
          </p>
        }
      >
        <p>
          Try "one slow thread": seven threads spend three quarters of their time waiting for the eighth. Then "balanced":
          nobody waits. A barrier is cheap when the phase before it gives every thread about the same amount of work, and
          that is something you decide when you divide the work among threads.
        </p>
      </Translated>

      <h2><Translated vi="Các quy tắc">The rules</Translated></h2>
      <Translated
        vi={
          <p>
            Một rào chắn chỉ có thể mở nếu mọi luồng mà nó chờ thực sự tới được. Điều đó đặt điều kiện lên chỗ bạn được phép
            viết nó:
          </p>
        }
      >
        <p>
          A barrier can only open if every thread it waits for actually gets there. That puts conditions on where you
          may write one:
        </p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>một <code>__syncthreads()</code> phải được mọi luồng của khối thực thi, hoặc không luồng nào;</li>
            <li>bên trong một <code>if</code>, điều đó có nghĩa là điều kiện phải cho cùng kết quả với mọi luồng của khối;</li>
            <li>hai lời gọi ở hai chỗ trong mã nguồn là hai rào chắn khác nhau, kể cả khi chúng trông giống nhau.</li>
          </ul>
        }
      >
        <ul>
          <li>a <code>__syncthreads()</code> must be executed by all threads of the block, or by none;</li>
          <li>inside an <code>if</code>, that means the condition has to come out the same for every thread of the block;</li>
          <li>two calls at two places in the source are two different barriers, even when they look alike.</li>
        </ul>
      </Translated>
      <Source section="4.3" pages="72" figs="Fig. 4.4" />
      <Translated
        vi={
          <p>
            Phản ví dụ của sách nằm bên dưới. Nhánh được chọn theo chỉ số của chính luồng, và mỗi nhánh kết thúc bằng một rào
            chắn.
          </p>
        }
      >
        <p>
          The book's counter-example is below. The branch is chosen by the thread's own index, and each branch ends in
          a barrier.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              <b>Lỗi in có vẻ có.</b> Đoạn mã in trong sách không có <code>__global__</code>, và phần chữ gọi các dòng của
              nó là 04, 06 và 10 trong khi đoạn mã đánh số chúng là 3, 5 và 9. Bản ở đây thêm <code>__global__</code> và giữ
              cách đánh số của đoạn mã.
            </p>
          }
        >
          <p>
            <b>Apparent errata.</b> The printed listing has no <code>__global__</code>, and the text refers to its lines
            as 04, 06 and 10 where the listing numbers them 3, 5 and 9. The copy here adds <code>__global__</code> and
            keeps the listing's numbering.
          </p>
        </Translated>
      </Aside>
      <Code
        tabs={[{ label: 'CUDA', code: SNIPPETS.incorrect_barrier, tiers: {}, status: SNIPPET_STATUS() }]}
        highlight={[5, 9]}
        caption={
          <Translated
            vi={
              <>
                Fig. 4.4 của sách, lưu tại <code>labs/ch04/snippets/incorrect_barrier.cu</code>. Nó vi phạm quy tắc thứ
                ba, thông qua quy tắc thứ hai.
              </>
            }
          >
            <>
              The book's Fig. 4.4, stored as <code>labs/ch04/snippets/incorrect_barrier.cu</code>. It breaks the third
              rule, by way of the second.
            </>
          </Translated>
        }
      />
      <Ask q={q['ch04/4.3/q-fig44']} label={tr('Before you step it', 'Trước khi bạn chạy từng bước')} />
      <Translated
        vi={
          <p>
            Sách gọi những gì xảy ra tiếp theo là hành vi không xác định, và cảnh báo rằng một rào chắn đặt sai chỗ có thể cho
            kết quả sai hoặc deadlock: các luồng chờ nhau mãi mãi. Giữ đúng quy tắc là việc của bạn; không có gì kiểm tra hộ
            bạn.
          </p>
        }
      >
        <p>
          The book calls what happens next undefined behaviour, and warns that a misplaced barrier can give wrong results
          or a deadlock: threads waiting for each other for ever. Keeping the rules is your job; nothing checks them for
          you.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch04-facts.md, A6)', '2026-11-20 (notes/today/ch04-facts.md, mục A6)')}>
        <Translated
          vi={
            <p>
              Hướng dẫn của NVIDIA nêu cùng quy tắc đó, và thêm hai chi tiết. Các luồng phải tới "the same" (cùng một) lời
              gọi. Và nó chỉ chờ các luồng "non-exited" (chưa thoát): một luồng đã return không giữ chân những luồng còn lại.
              Một rào chắn dưới một điều kiện không đồng nhất "may hang or produce unintended side effects" (có thể treo hoặc
              gây tác dụng phụ ngoài ý muốn).
            </p>
          }
        >
          <p>
            NVIDIA's guide states the same rule, and two details more. Threads must reach "the same" call. And it waits
            only for "non-exited" threads: one that has returned does not hold the rest. A barrier under a condition that
            is not uniform "may hang or produce unintended side effects".
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Rào chắn này có hợp lệ không?">Is this barrier legal?</Translated></h2>
      <Translated
        vi={
          <p>
            Năm kernel nhỏ. Hãy quyết định với từng kernel xem rào chắn của nó có hợp lệ không trước khi xem, rồi chạy từng bước
            qua một khối bốn luồng. Trình chạy từng bước có thể áp dụng một trong hai cách đọc quy tắc: của sách, nơi mọi luồng
            của khối phải tới, hoặc cách hẹp hơn trong tài liệu của NVIDIA và Apple, nơi một luồng đã kết thúc không được chờ.
          </p>
        }
      >
        <p>
          Five small kernels. Decide for each whether its barrier is legal before you look, then step through a block of
          four threads. The stepper can apply either reading of the rule: the book's, where every thread of the block
          must arrive, or the narrower one in NVIDIA's and Apple's documents, where a thread that has ended is not waited
          for.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              "Kẹt" đại diện cho điều mà các nhà cung cấp gọi là không xác định: một thiết bị thật có thể treo hoặc có thể chạy
              tiếp và tính ra kết quả vô nghĩa. Mỗi lần một luồng di chuyển, xa hết mức có thể, theo một trong hai thứ tự cố
              định. Một lần chạy không bị đánh dấu gì chỉ cho thấy thứ tự này diễn ra suôn sẻ, không phải kernel đúng.
            </p>
          }
        >
          <p>
            "Stuck" stands for what the vendors call undefined: a real device may hang or may run on and compute
            nonsense. One thread moves at a time, as far as it can, in one of two fixed orders. A run with nothing flagged
            shows that this order went well, not that the kernel is correct.
          </p>
        </Translated>
      </Simplification>
      <BugHunt />
      <Translated
        vi={
          <p>
            Kernel 3 theo quy tắc thứ hai đáng xem lại với "luồng 3 chạy trước": mọi thứ hoàn tất và không có gì bị đánh dấu.
            Đó là cách một rào chắn đặt sai chỗ qua được kiểm thử. Điều kiện vẫn không đồng nhất, và với thứ tự còn lại, luồng
            0 đọc một ô trước khi luồng bên cạnh kịp ghi.
          </p>
        }
      >
        <p>
          Kernel 3 under the second rule is worth a second look with "thread 3 runs first": everything finishes and
          nothing is flagged. That is how a misplaced barrier passes a test. The condition is still not uniform, and
          with the other order thread 0 reads a slot before its neighbour has written it.
        </p>
      </Translated>
      <Ask q={q['ch04/4.3/q-uniform']} />

      <h2><Translated vi="Hệ thống hứa gì để đáp lại">What the system promises in return</Translated></h2>
      <Translated
        vi={
          <p>
            Các quy tắc cũng đòi hỏi runtime một điều. Nếu một số luồng của khối có thể bắt đầu trong khi những luồng khác chưa
            có chỗ chạy, các luồng sớm có thể ngồi ở rào chắn vô thời hạn. CUDA loại trừ điều đó bằng cách cấp tài nguyên cho
            một khối như một đơn vị: một khối chỉ bắt đầu khi runtime đã bảo đảm được những gì mọi luồng của nó cần để chạy
            xong. Vì vậy các luồng của một khối chạy gần nhau về thời gian, và một lần chờ ở rào chắn đặt đúng chỗ luôn có giới
            hạn. Đây là nửa kia của việc gán nguyên khối ở phần 4.2.
          </p>
        }
      >
        <p>
          The rules ask something of the runtime too. If some threads of a block could start while others had no place
          to run, the early ones might sit at a barrier indefinitely. CUDA rules that out by assigning resources to a
          block as a unit: a block starts only once the runtime has secured what all of its threads need to finish. So
          the threads of a block run close together in time, and a wait at a correctly placed barrier is always bounded.
          This is the other half of section 4.2's whole-block assignment.
        </p>
      </Translated>
      <Source section="4.3" pages="72–73" />

      <h2><Translated vi="Các phạm vi rộng hơn">Wider scopes</Translated></h2>
      <Translated
        vi={
          <p>
            Tập các luồng mà một rào chắn chờ là <b>phạm vi (scope)</b> của nó. <code>__syncthreads()</code> có phạm vi là
            khối. Có hai phạm vi rộng hơn, cả hai qua Cooperative Groups API: rào chắn toàn cluster cho một thread block
            cluster, và rào chắn toàn lưới.
          </p>
        }
      >
        <p>
          The set of threads a barrier waits for is its <b>scope</b>. <code>__syncthreads()</code> has the block as its
          scope. Two wider ones exist, both through the Cooperative Groups API: a cluster-wide barrier for a thread block
          cluster, and a grid-wide barrier.
        </p>
      </Translated>
      <Source section="4.3" pages="70–71" />
      <Translated
        vi={
          <p>
            Sách gọi rào chắn toàn lưới là nặng hơn và nói nó đi kèm các hạn chế để bảo đảm mọi luồng của lưới thực sự chạy cùng
            lúc. Ví dụ minh họa của nó ở chương 18.
          </p>
        }
      >
        <p>
          The book calls the grid-wide one heavier and says it comes with restrictions that make sure all threads of the
          grid really are running at once. Its worked example is in chapter 18.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch04-facts.md, A7)', '2026-11-20 (notes/today/ch04-facts.md, mục A7)')}>
        <Translated
          vi={
            <p>
              Các hạn chế hiện nay: khởi chạy bằng <code>cudaLaunchCooperativeKernel</code> thay vì{' '}
              <code>&lt;&lt;&lt;&gt;&gt;&gt;</code>, và giữ lưới không lớn hơn lượng thường trú cùng lúc. Dạng đa thiết bị của
              cooperative launch đã bị gỡ bỏ trong CUDA 13.
            </p>
          }
        >
          <p>
            The restrictions today: launch with <code>cudaLaunchCooperativeKernel</code> instead of{' '}
            <code>&lt;&lt;&lt;&gt;&gt;&gt;</code>, and keep the grid no larger than what is resident at once. The
            multi-device form of the cooperative launch was removed in CUDA 13.
          </p>
        </Translated>
      </Today>
      <Ask q={q['ch04/4.3/q-gridbarrier']} />

      <h2><Translated vi="Vì sao các khối có thể chạy theo thứ tự bất kỳ">Why blocks may run in any order</Translated></h2>
      <Translated
        vi={
          <p>
            Hãy lật ngược phạm vi toàn khối. Nếu không rào chắn nào vươn qua biên của một khối, thì không khối nào phải chờ khối
            khác, và không có gì trong mô hình sắp thứ tự chúng. Runtime có thể chạy chúng theo thứ tự bất kỳ, vài khối một lúc
            hay nhiều khối, trên bất cứ thứ gì thiết bị có. Sách gọi kết quả là <b>khả năng mở rộng trong suốt (transparent
            scalability)</b>: cùng một mã chạy không cần sửa trên phần cứng với lượng tài nguyên khác nhau, chậm trên thiết bị
            nhỏ và nhanh hơn trên thiết bị lớn.
          </p>
        }
      >
        <p>
          Turn the block-wide scope around. If no barrier reaches across the edge of a block, then no block ever waits
          for another, and nothing in the model orders them. The runtime may run them in any order, a few at a time or
          many, on whatever the device has. The book calls the result <b>transparent scalability</b>: the same code runs
          unchanged on hardware with different amounts of resources, slowly on a small device and faster on a large one.
        </p>
      </Translated>
      <Source section="4.3" pages="73–74" figs={tr('Fig. 4.5, in the animation below', 'Fig. 4.5, trong hoạt hình bên dưới')} />
      <WavesScene />

      <h2><Translated vi="Các đợt và hiệu ứng đuôi">Waves and the tail effect</Translated></h2>
      <Translated
        vi={
          <p>
            Một nhóm khối chạy cùng lúc là một <b>đợt (wave)</b>, và sách đếm số đợt bằng số khối trong lưới chia cho số khối
            chạy cùng lúc. Tám khối thành bốn đợt trên thiết bị chạy hai khối một lúc, và hai đợt trên thiết bị chạy bốn. Con số
            không nhất thiết là số nguyên. Ví dụ của sách: {TAIL_EXAMPLE.blocks} khối trên một thiết bị chạy {cap} khối cùng lúc
            là {waves(TAIL_EXAMPLE.blocks, cap)} đợt, hai đợt đầy và một đợt cuối {bookTail.last} khối để{' '}
            {Math.round(bookTail.idleShare * 100)}% thiết bị không có gì làm. Đó là <b>hiệu ứng đuôi (tail effect)</b>. Cách
            khắc phục là một số khối là bội số của lượng chạy cùng lúc, điều phụ thuộc vào thiết bị.
          </p>
        }
      >
        <p>
          A group of blocks that runs at the same time is a <b>wave</b>, and the book counts waves as the blocks in the
          grid divided by the blocks that run at once. Eight blocks make four waves on a device that runs two at a time,
          and two waves on one that runs four. The count need not be whole. The book's example: {TAIL_EXAMPLE.blocks}{' '}
          blocks on a device that runs {cap} at once is {waves(TAIL_EXAMPLE.blocks, cap)} waves, two full ones and a last
          wave of {bookTail.last} that leaves {Math.round(bookTail.idleShare * 100)}% of the device with nothing to do.
          That is the <b>tail effect</b>. The remedy is a block count that is a multiple of what runs at once, which
          depends on the device.
        </p>
      </Translated>
      <Source section="4.3" pages="73–74" />
      <Translated
        vi={
          <p>
            Nên nhắm bao nhiêu đợt thì tùy vào các khối. Khi chúng tốn thời gian xấp xỉ nhau, ít đợt cũng ổn và điều quan trọng
            là đợt cuối đầy; chương 6 dùng điều đó. Khi tải của chúng khác nhau, nhiều đợt thì tốt hơn: mỗi ô rảnh sớm sẽ nhận
            khối kế tiếp, và tải được dàn đều giữa các SM.
          </p>
        }
      >
        <p>
          How many waves to aim for depends on the blocks. When they all take about the same time, few waves are fine
          and a full last wave is what matters; chapter 6 uses that. When their loads differ, many waves are better: each
          slot that frees up early takes the next block, and the load evens out across SMs.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              "Wave", "tail effect" và "transparent scalability" là thuật ngữ của sách. Hướng dẫn của NVIDIA không dùng chúng;
              tài liệu HIP của AMD có nói một GPU nhỏ hơn lập lịch các khối "in more waves" (theo nhiều đợt hơn)
              (notes/today/ch04-facts.md, A5, B2).
            </p>
          }
        >
          <p>
            "Wave", "tail effect" and "transparent scalability" are the book's terms. NVIDIA's guide does not use them;
            AMD's HIP documentation does say a smaller GPU schedules blocks "in more waves" (notes/today/ch04-facts.md,
            A5, B2).
          </p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Trình khám phá chạy bộ lập lịch của phần 4.2 trên những lưới quá lớn để chạy từng bước, và vẽ số khối đang chạy từ
            lúc khởi chạy đến khi khối cuối kết thúc.
          </p>
        }
      >
        <p>
          The explorer runs the scheduler of section 4.2 on grids too large to step through, and draws how many blocks
          are in progress from the launch to the end of the last block.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Các khối là những đơn vị không chia nhỏ với thời gian cho trước, và "số khối mỗi SM" là một con số bạn gõ vào.
              Trên thiết bị thật, nó đến từ các ngân sách của phần 4.7, và thời gian một khối cần phụ thuộc vào những gì khác
              đang thường trú.
            </p>
          }
        >
          <p>
            Blocks are atoms with given durations, and "blocks per SM" is a number you type. On a real device it comes
            from section 4.7's budgets, and the time a block takes depends on what else is resident.
          </p>
        </Translated>
      </Simplification>
      <WavesExplorer />
      <Ask q={q['ch04/4.3/q-tail']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Một trường hợp từ suy luận LLM có thể nằm ở đầu kia. Lấy mẫu làm việc trên một vector duy nhất có kích thước bằng
            bộ từ vựng, nên lưới của nó có từ vựng ÷ (số luồng mỗi khối × số phần tử mỗi luồng) khối, làm tròn lên. Một bộ từ
            vựng 32,000 với 256 luồng mỗi khối và một phần tử mỗi luồng là {sampleBlocks(32000, 1)} khối:{' '}
            {fmt(waves(sampleBlocks(32000, 1), cap), 2)} đợt trên con số {cap} của sách. Hãy đặt trình khám phá thành{' '}
            {sampleBlocks(32000, 1)} khối và nhìn các ô ngồi không. Không lựa chọn số khối nào lấp đầy được một thiết bị như
            vậy, và phần đuôi không phải là vấn đề. Không phải lúc nào nó cũng nhỏ như thế: một bộ từ vựng 128,000 với một phần
            tử mỗi luồng là {sampleBlocks(128000, 1)} khối, {fmt(waves(sampleBlocks(128000, 1), cap), 2)} đợt, và phép tính
            phần đuôi lại áp dụng, cho đến khi mỗi luồng nhận bốn phần tử và lưới trở về {sampleBlocks(128000, 4)} khối.
          </p>
        }
      >
        <p>
          One case from LLM inference can sit at the other end. Sampling works on a single vector the size of the
          vocabulary, so its grid has vocabulary ÷ (threads per block × elements per thread) blocks, rounded up. A
          vocabulary of 32,000 at 256 threads per block and one element per thread is {sampleBlocks(32000, 1)} blocks:{' '}
          {fmt(waves(sampleBlocks(32000, 1), cap), 2)} of a wave on the book's {cap}. Set the explorer to{' '}
          {sampleBlocks(32000, 1)} blocks and look at the idle slots. No choice of block count fills such a device, and the
          tail is not the question. It is not always so small: a vocabulary of 128,000 at one element per thread is{' '}
          {sampleBlocks(128000, 1)} blocks, {fmt(waves(sampleBlocks(128000, 1), cap), 2)} waves, and the tail arithmetic
          applies again, until each thread takes four elements and the grid is back to {sampleBlocks(128000, 4)} blocks.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Của chúng tôi. Sách không bàn về lấy mẫu; các con số là phép tính đợt của sách áp dụng cho những bộ từ vựng mà
              chúng tôi chọn là số tròn.
            </p>
          }
        >
          <p>
            Ours. The book does not discuss sampling; the counts are its wave arithmetic applied to vocabularies we chose
            as round numbers.
          </p>
        </Translated>
      </Beyond>
      <Ask q={q['ch04/4.3/q-scal']} />

      <h2><Translated vi="Trên các backend khác">On other backends</Translated></h2>
      <Translated
        vi={
          <p>
            Rào chắn có trên cả ba backend. Điều khác nhau là mỗi nhà cung cấp đã viết ra bao nhiêu phần của quy tắc, và liệu có
            gì rộng hơn một khối hay không.
          </p>
        }
      >
        <p>
          The barrier exists on all three backends. What differs is how much of the rule each vendor has written down,
          and whether anything wider than a block exists.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Bảng này là của chúng tôi, đối chiếu với tài liệu của nhà cung cấp ngày 2026-11-20 (notes/today/ch04-facts.md,
              A6, A7, B3, B11, C3; ch05-facts.md C2). Khóa học này chưa chạy nội dung nào trong bảng trên phần cứng CUDA hay
              HIP.
            </p>
          }
        >
          <p>
            The table is ours, checked against vendor documentation on 2026-11-20 (notes/today/ch04-facts.md, A6, A7,
            B3, B11, C3; ch05-facts.md C2). Nothing in this table has been run on CUDA or HIP hardware by this course.
          </p>
        </Translated>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }} aria-label={tr('Barriers across backends', 'Rào chắn trên các backend')}>
        <thead>
          <tr><th style={{ width: '18%' }}><Translated vi="Cái gì">What</Translated></th><th>CUDA</th><th><Translated vi="HIP trên AMD">HIP on AMD</Translated></th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="Rào chắn toàn khối">Block-wide barrier</Translated></td>
            <td><code>__syncthreads()</code></td>
            <td><Translated vi="cùng cách viết">same spelling</Translated></td>
            <td>
              <Translated
                vi={
                  <>
                    <code>threadgroup_barrier(mem_flags::…)</code>. Cờ chỉ ra vùng nhớ cần sắp thứ tự:{' '}
                    <code>mem_threadgroup</code> cho bộ nhớ threadgroup, <code>mem_device</code> cho các buffer của device
                  </>
                }
              >
                <code>threadgroup_barrier(mem_flags::…)</code>. The flag names the memory to order:{' '}
                <code>mem_threadgroup</code> for threadgroup memory, <code>mem_device</code> for device buffers
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="Quy tắc cho rào chắn dưới điều kiện">Rule for a barrier under a condition</Translated></td>
            <td><Translated vi="có tài liệu: chỉ khi điều kiện đồng nhất trên cả khối">documented: only if the condition is uniform across the block</Translated></td>
            <td><Translated vi="không được nêu trong các trang của AMD. Hãy theo quy tắc của NVIDIA như quy tắc khả chuyển; lựa chọn đó lấy từ tài liệu của CUDA, không phải của AMD">not stated in AMD's pages. Follow NVIDIA's rule as the portable one; that choice is taken from CUDA's documentation, not AMD's</Translated></td>
            <td><Translated vi="có tài liệu, cho câu lệnh điều kiện và cho vòng lặp">documented, for conditionals and for loops</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Một luồng đã kết thúc">A thread that has ended</Translated></td>
            <td><Translated vi={'không chặn các luồng khác ("non-exited threads")'}>does not block the others ("non-exited threads")</Translated></td>
            <td><Translated vi="không được nêu">not stated</Translated></td>
            <td><Translated vi="không chặn các luồng khác, có tài liệu cho Apple silicon">does not block the others, documented for Apple silicon</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Rào chắn toàn cluster">Cluster-wide barrier</Translated></td>
            <td><Translated vi={<><code>cluster.sync()</code>, compute capability 9.0 trở lên</>}><code>cluster.sync()</code>, compute capability 9.0 and up</Translated></td>
            <td><Translated vi={<>có tài liệu về <code>cluster_group</code>; thiết bị nào hỗ trợ nó thì unverified</>}>a <code>cluster_group</code> is documented; which devices support it is unverified</Translated></td>
            <td><Translated vi="không có">none</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Rào chắn toàn lưới">Grid-wide barrier</Translated></td>
            <td><Translated vi={<><code>this_grid().sync()</code>, khởi chạy bằng <code>cudaLaunchCooperativeKernel</code></>}><code>this_grid().sync()</code>, launched with <code>cudaLaunchCooperativeKernel</code></Translated></td>
            <td><Translated vi={<><code>this_grid().sync()</code>, khởi chạy bằng <code>hipLaunchCooperativeKernel</code></>}><code>this_grid().sync()</code>, launched with <code>hipLaunchCooperativeKernel</code></Translated></td>
            <td><Translated vi="không có tài liệu nào. Một pha cần cả lưới xong là một lần dispatch riêng">none documented. A phase that needs the whole grid to finish is a separate dispatch</Translated></td>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Lab 4.3 trên <a href={href('ch04', 'labs')}>trang lab</a> là một cuộc săn lỗi rào chắn trên những kernel bạn biên
            dịch và chạy.
          </p>
        }
      >
        <p>
          Lab 4.3 on the <a href={href('ch04', 'labs')}>labs page</a> is a barrier hunt on kernels you compile and
          run.
        </p>
      </Translated>
    </>
  );
}
