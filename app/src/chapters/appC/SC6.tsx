import { useEffect, useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useNarrow } from '../../lib/narrow';
import { href } from '../../lib/router';
import {
  FIG_HEAD,
  FIG_PAGES,
  PROGRAM_FILE,
  PROGRAM_NAME,
  moved,
  pageList,
  runFig,
  steppable,
  type Counters,
  type FigLine,
  type FigRun,
  type Policy,
  type Program,
  type Step,
  type View,
} from '../../sim/managedmem';
import { TwoUnifiedScene } from './TwoUnifiedScene';

// Section C.6, Unified memory (book pp. 626–627). The stepper runs Fig. C.1 and our
// explicit-copy port on src/sim/managedmem.ts; the listings are the model's lines, which
// SC6.snippets.test.ts ties to labs/appC/snippets/c6_sortfile_*.cu character for character.

type Mode = Extract<Policy, 'bulk' | 'migrate' | 'one'>;
const MODES: Mode[] = ['bulk', 'migrate', 'one'];
const MODE_LABEL: Record<Mode, string> = {
  get bulk() {
    return tr(
      'limited: CUDA 6 on Kepler/Maxwell; today Windows, WSL, some Tegra',
      'hạn chế: CUDA 6 trên Kepler/Maxwell; ngày nay Windows, WSL, một số Tegra',
    );
  },
  get migrate() {
    return tr('full: compute capability 6.0 and later on Linux', 'đầy đủ: compute capability 6.0 trở lên trên Linux');
  },
  get one() {
    return tr('Apple silicon, Metal shared buffer', 'Apple silicon, buffer shared của Metal');
  },
};
const MODE_SHORT: Record<Mode, string> = {
  get bulk() {
    return tr('limited', 'hạn chế');
  },
  get migrate() {
    return tr('full', 'đầy đủ');
  },
  get one() {
    return tr('Apple silicon', 'Apple silicon');
  },
};
const PROGRAMS: Program[] = ['cpu', 'managed', 'explicit']; // Fig. C.1's order, then our third column

// The three listings, aligned row by row like a side-by-side diff: [cpu, managed, explicit]
// line indices into figLines(), or null where a listing has no such line.
const ROWS: (number | null)[][] = [
  [0, 0, 0], // declaration
  [1, 1, 1], // malloc / cudaMallocManaged / malloc
  [null, null, 2], // cudaMalloc
  [2, 2, 3], // fread
  [null, null, 4], // copy in
  [3, 3, 5], // sort
  [4, 4, null], // (blank) / cudaDeviceSynchronize
  [null, null, 6], // copy out
  [5, 5, 7], // use_data
  [null, null, 8], // cudaFree(d_data)
  [6, 6, 9], // free / cudaFree
];

const plural = (n: number, one: string, many = one + 's') => `${n} ${n === 1 ? one : many}`;

// Listing styles (inline, like the rest of the app's interactives).
const MONO: CSSProperties = { fontFamily: 'var(--sans)', fontVariationSettings: "'MONO' 1, 'CASL' 0" }; // i18n-ignore: CSS value
const lnStyle = (chg: boolean, del: boolean, hl: boolean): CSSProperties => ({
  ...MONO,
  display: 'block',
  whiteSpace: 'pre',
  padding: '0 0.45rem 0 0.25rem', // i18n-ignore: CSS value
  minHeight: '1.6em',
  fontWeight: chg ? 680 : undefined,
  textDecoration: del ? 'line-through' : undefined,
  color: del ? 'var(--fail)' : undefined,
  background: hl ? 'var(--rule-soft)' : undefined,
  boxShadow: hl ? 'inset 3px 0 0 var(--ink)' : undefined, // i18n-ignore: CSS value
});

/** One sentence or three about what a line just did, from the model's events. */
function describe(program: Program, mode: Mode, line: FigLine, step: Step | null, fr: FigRun): ReactNode {
  const ev = step?.events ?? [];
  const count = (e: string, extra?: (x: (typeof ev)[number]) => boolean) => ev.filter((x) => x.e === e && (!extra || extra(x))).length;
  const crashed = step?.skipped;
  if (crashed)
    return (
      <Translated vi={<>Chương trình đã dừng: dòng này không bao giờ chạy.</>}>
        <>The program has already stopped: this line never runs.</>
      </Translated>
    );
  const err = step?.errors[0];
  switch (line.role) {
    case 'decl':
      return program === 'explicit'
        ? (
          <Translated vi={<>Hai con trỏ: một cho buffer CPU, một cho buffer GPU. Mô hình của chương 2 cần cả hai.</>}>
            <>Two pointers: one for a CPU buffer, one for a GPU buffer. Chapter 2's model needs both.</>
          </Translated>
        )
        : (
          <Translated vi={<>Khai báo con trỏ. Chưa có bộ nhớ.</>}>
            <>Declares the pointer. No memory yet.</>
          </Translated>
        );
    case 'alloc':
      if (line.op?.op === 'alloc' && line.op.kind === 'managed')
        return mode === 'one' ? (
          <Translated
            vi={
              <>
                Trên Apple silicon, thứ tương ứng là một buffer có lưu trữ shared (<code>newBufferWithLength:options:</code> với{' '}
                <code>MTLResourceStorageModeShared</code>); con trỏ <code>contents</code> của nó là thứ CPU dùng. Không có chế
                độ managed nào để chọn: lưu trữ managed "không có tác dụng trên Apple Silicon".
              </>
            }
          >
            <>
              On Apple silicon the counterpart is a buffer with shared storage (<code>newBufferWithLength:options:</code> with{' '}
              <code>MTLResourceStorageModeShared</code>); its <code>contents</code> pointer is what the CPU uses. There is no
              managed mode to choose: managed storage "has no effect on Apple Silicon".
            </>
          </Translated>
        ) : (
          <Translated
            vi={
              <>
                <code>cudaMallocManaged</code> trả về một con trỏ mà cả CPU và GPU đều dùng được. Trong mô hình này chưa trang
                nào được đặt chỗ: mỗi trang nằm ở nơi nó được chạm tới lần đầu.
              </>
            }
          >
            <>
              <code>cudaMallocManaged</code> returns one pointer that both the CPU and the GPU may use. In this model no page is
              placed yet: each lands where it is first touched.
            </>
          </Translated>
        );
      if (line.op?.op === 'alloc' && line.op.kind === 'device')
        return mode === 'one' ? (
          <Translated
            vi={<>Thứ tương ứng trên Metal là một buffer có lưu trữ private: trong cùng bộ nhớ vật lý, nhưng chỉ GPU dùng.</>}
          >
            <>The counterpart on Metal is a buffer with private storage: in the same physical memory, but GPU only.</>
          </Translated>
        ) : (
          <Translated
            vi={
              <>
                <code>cudaMalloc</code>: {FIG_PAGES} trang bộ nhớ GPU. Mã host không được dereference <code>d_data</code>.
              </>
            }
          >
            <>
              <code>cudaMalloc</code>: {FIG_PAGES} pages of GPU memory. Host code may not dereference <code>d_data</code>.
            </>
          </Translated>
        );
      return (
        <Translated vi={<><code>malloc</code>: bộ nhớ CPU thông thường, chưa được chạm tới.</>}>
          <><code>malloc</code>: ordinary CPU memory, not yet touched.</>
        </Translated>
      );
    case 'read-file':
      return (
        <Translated
          vi={
            <>
              <code>fread</code> ghi cả N byte qua <code>data</code> trên CPU. {count('populate')} trang được đặt trong{' '}
              {mode === 'one' ? 'bộ nhớ duy nhất' : 'bộ nhớ CPU'}, nơi chúng được chạm tới lần đầu.
            </>
          }
        >
          <>
            <code>fread</code> writes all N bytes through <code>data</code> on the CPU. {plural(count('populate'), 'page')} placed in{' '}
            {mode === 'one' ? 'the one memory' : 'CPU memory'}, where {count('populate') === 1 ? 'it was' : 'they were'} first touched.
          </>
        </Translated>
      );
    case 'copy-in':
      return mode === 'one' ? (
        <Translated
          vi={
            <>
              Lần sao chép điền buffer GPU từ buffer CPU: {step!.counters.copiedInMemory} trang được sao chép bên trong một
              bộ nhớ vật lý. Không có đường truyền nào phải đi qua, nhưng đó vẫn là một lần sao chép (trên Metal, một lần
              blit vào buffer private).
            </>
          }
        >
          <>
            The copy fills the GPU buffer from the CPU buffer: {plural(step!.counters.copiedInMemory, 'page')} copied inside one
            physical memory. No link to cross, but it is still a copy (on Metal, a blit into a private buffer).
          </>
        </Translated>
      ) : (
        <Translated
          vi={
            <>
              <code>cudaMemcpy</code> chuyển cả {count('move')} trang sang buffer GPU, bất kể kernel sẽ chạm tới những gì.
            </>
          }
        >
          <>
            <code>cudaMemcpy</code> moves all {plural(count('move'), 'page')} to the GPU buffer, whatever the kernel will touch.
          </>
        </Translated>
      );
    case 'sort':
      if (program === 'cpu')
        return (
          <Translated vi={<>CPU sắp xếp tại chỗ. Không có gì di chuyển.</>}>
            <>The CPU sorts in place. Nothing moves.</>
          </Translated>
        );
      if (program === 'explicit')
        return (
          <Translated
            vi={<>Lời gọi khởi chạy trả về ngay. Kernel sắp xếp <code>d_data</code> trong bộ nhớ GPU: chỉ có truy cập cục bộ.</>}
          >
            <>The launch returns at once. The kernel sorts <code>d_data</code> in GPU memory: local accesses only.</>
          </Translated>
        );
      if (mode === 'one')
        return (
          <Translated
            vi={
              <>
                Trên Metal: mã hóa lệnh dispatch và commit command buffer. GPU đọc và ghi chính những trang đó tại chỗ. Không
                có gì di chuyển.
              </>
            }
          >
            <>
              On Metal: encode the dispatch and commit the command buffer. The GPU reads and writes the same pages in place.
              Nothing moves.
            </>
          </Translated>
        );
      if (mode === 'bulk')
        return (
          <Translated
            vi={
              <>
                Lời gọi khởi chạy trả về ngay. Trước khi kernel chạy, mọi trang managed có bản sao trong bộ nhớ CPU đều chuyển
                sang GPU: {count('move', (x) => x.why === 'bulk')} trang, bất kể kernel chạm tới những gì. Từ giờ cho đến lần
                đồng bộ hóa, CPU không được chạm tới bất kỳ bộ nhớ managed nào.
              </>
            }
          >
            <>
              The launch returns at once. Before the kernel runs, every managed page whose copy is in CPU memory moves to the GPU:{' '}
              {plural(count('move', (x) => x.why === 'bulk'), 'page')}, whatever the kernel touches. From now until the synchronise the
              CPU must not touch any managed memory.
            </>
          </Translated>
        );
      return (
        <Translated
          vi={
            <>
              Lời gọi khởi chạy trả về ngay và chưa có gì di chuyển. Mỗi trang kernel chạm tới gây lỗi trang ở lần chạm đầu
              và di chuyển: {count('fault')} lỗi trang GPU, {count('move')} trang được chuyển.
              {line.op?.op === 'launch' && line.op.reads.length < FIG_PAGES ? ' Các trang không được chạm tới vẫn nằm trong bộ nhớ CPU.' : ''}
            </>
          }
        >
          <>
            The launch returns at once and nothing moves yet. Each page the kernel touches faults on its first touch and migrates:{' '}
            {plural(count('fault'), 'GPU fault')}, {plural(count('move'), 'page')} moved.
            {line.op?.op === 'launch' && line.op.reads.length < FIG_PAGES ? ' Untouched pages stay in CPU memory.' : ''}
          </>
        </Translated>
      );
    case 'sync':
      return mode === 'one' ? (
        <Translated
          vi={<>Trên Metal, <code>waitUntilCompleted</code> trên command buffer: nghĩa vụ duy nhất mà một bộ nhớ chung để lại cho bạn.</>}
        >
          <>On Metal, <code>waitUntilCompleted</code> on the command buffer: the one duty one memory leaves you.</>
        </Translated>
      ) : (
        <Translated
          vi={
            <>
              <code>cudaDeviceSynchronize</code> chờ đến khi kernel chạy xong. Giờ CPU có thể chạm tới dữ liệu
              {mode === 'bulk' ? ', và ở chế độ hạn chế, mới được chạm tới bộ nhớ managed' : ''}.
            </>
          }
        >
          <>
            <code>cudaDeviceSynchronize</code> waits until the kernel has finished. Now the CPU may touch the data
            {mode === 'bulk' ? ', and in limited mode, touch managed memory at all' : ''}.
          </>
        </Translated>
      );
    case 'copy-out':
      return mode === 'one' ? (
        <Translated vi={<>Chờ kernel, rồi sao chép {FIG_PAGES} trang trở lại bên trong cùng bộ nhớ đó.</>}>
          <>Waits for the kernel, then copies {plural(FIG_PAGES, 'page')} back inside the same memory.</>
        </Translated>
      ) : (
        <Translated
          vi={
            <>
              <code>cudaMemcpy</code> thứ hai chờ kernel, rồi chuyển cả {count('move')} trang trở lại.
            </>
          }
        >
          <>
            The second <code>cudaMemcpy</code> waits for the kernel, then moves all {plural(count('move'), 'page')} back.
          </>
        </Translated>
      );
    case 'use':
      if (err?.kind === 'segfault')
        return (
          <Translated
            vi={
              <>
                <b className="t-fail">Segmentation fault.</b> <code>use_data</code> chạm tới bộ nhớ managed trong khi kernel có
                thể vẫn đang chạy, điều mà các hệ thống hạn chế không cho phép. Chương trình dừng ở đây.
              </>
            }
          >
            <>
              <b className="t-fail">Segmentation fault.</b> <code>use_data</code> touches managed memory while the kernel may
              still be running, which limited systems do not allow. The program stops here.
            </>
          </Translated>
        );
      if (err?.kind === 'race')
        return (
          <Translated
            vi={
              <>
                <b className="t-fail">Một race condition.</b> Không sập: CPU được chạm tới bộ nhớ managed khi một kernel đang
                chạy. Nhưng không có gì bắt <code>use_data</code> chờ việc sắp xếp, nên nó có thể đọc các byte chưa sắp xếp
                hoặc mới sắp xếp một phần trong {pageList(err.pages)}.{' '}
                {count('fault') ? `(${count('fault')} lỗi trang CPU đã kéo những trang đó về giữa lúc đang sắp xếp.)` : ''}
              </>
            }
          >
            <>
              <b className="t-fail">A race.</b> No crash: the CPU may touch managed memory while a kernel runs. But nothing makes{' '}
              <code>use_data</code> wait for the sort, so it may read unsorted or partly sorted bytes in {pageList(err.pages)}.{' '}
              {count('fault') ? `(${plural(count('fault'), 'CPU fault')} pulled those pages back mid-sort.)` : ''}
            </>
          </Translated>
        );
      if (err?.kind === 'ordering')
        return (
          <Translated
            vi={
              <>
                <b className="t-fail">Một lỗi thứ tự.</b> Trên một bộ nhớ chung không có gì di chuyển, nên cũng không có gì sập:
                CPU đọc <code>contents</code> trước khi chờ command buffer, và có thể thấy các byte chưa sắp xếp trong{' '}
                {pageList(err.pages)}.
              </>
            }
          >
            <>
              <b className="t-fail">An ordering error.</b> Nothing moves on one memory, so nothing crashes either: the CPU reads{' '}
              <code>contents</code> before waiting for the command buffer, and may see unsorted bytes in {pageList(err.pages)}.
            </>
          </Translated>
        );
      if (program !== 'managed' || mode === 'one')
        return (
          <Translated vi={<><code>use_data</code> đọc mọi byte trên CPU. Không có gì di chuyển.</>}>
            <><code>use_data</code> reads every byte on the CPU. Nothing moves.</>
          </Translated>
        );
      return (
        <Translated
          vi={
            <>
              <code>use_data</code> đọc mọi byte trên CPU.{' '}
              {count('fault')
                ? `Mỗi trang nằm trong bộ nhớ GPU gây lỗi trang và quay về: ${count('fault')} lỗi trang CPU, ${count('move')} trang được chuyển.`
                : 'Mọi trang đều đã nằm trong bộ nhớ CPU.'}
            </>
          }
        >
          <>
            <code>use_data</code> reads every byte on the CPU.{' '}
            {count('fault')
              ? `Each page that lives in GPU memory faults back: ${plural(count('fault'), 'CPU fault')}, ${plural(count('move'), 'page')} moved.`
              : 'Every page is already in CPU memory.'}
          </>
        </Translated>
      );
    case 'free':
      if (line.text.includes('cudaFree') && step?.events.some((x) => x.e === 'wait'))
        return (
          <Translated
            vi={
              <>
                <code>cudaFree</code> "có thể thực hiện đồng bộ hóa ngầm", nên trong mô hình này kernel đã chạy xong vào lúc
                này. Quá muộn cho <code>use_data</code>.
              </>
            }
          >
            <>
              <code>cudaFree</code> "may perform implicit synchronization", so the kernel has finished by now in this model. Too
              late for <code>use_data</code>.
            </>
          </Translated>
        );
      return line.text.includes('cudaFree') ? (
        <Translated vi={<><code>cudaFree</code> giải phóng bộ nhớ.</>}>
          <><code>cudaFree</code> releases the memory.</>
        </Translated>
      ) : (
        <Translated vi={<><code>free</code> giải phóng buffer CPU.</>}>
          <><code>free</code> releases the CPU buffer.</>
        </Translated>
      );
    default:
      return null;
  }
}

/** The buffers of a view, drawn in the memory that holds each page. */
function Memories({ view, mode, hot }: { view: View; mode: Mode; hot: Set<string> }) {
  const bufs = Object.entries(view);
  const name = (b: string, kind: string) =>
    tr(
      `${b} (${kind === 'managed' ? (mode === 'one' ? 'shared buffer' : 'managed') : kind === 'device' ? 'cudaMalloc' : 'malloc'})`,
      `${b} (${kind === 'managed' ? (mode === 'one' ? 'buffer shared' : 'managed') : kind === 'device' ? 'cudaMalloc' : 'malloc'})`,
    );
  const row = (box: 'cpu' | 'gpu' | 'one', b: string) => {
    const v = view[b];
    return (
      <div key={b} style={{ marginBottom: '0.5rem' }}>
        <div className="mx-label" style={{ fontWeight: 500 }}>{name(b, v.kind)}</div>
        <div className={'mx' + (box === 'gpu' ? ' global' : '')} style={{ gridTemplateColumns: `repeat(${v.pages.length}, auto)` }}>
          {v.pages.map((p, i) => {
            const here = box === 'one' ? p.loc === 'one' : p.loc === box || p.loc === 'both';
            const isHot = here && hot.has(`${b}:${i}`);
            return (
              <span key={i} data-box={box} data-buf={b} data-page={i} data-present={here ? 'yes' : 'no'}
                className={here ? (box === 'gpu' ? (isHot ? 'hot' : 'on') : '') : 'void'}
                style={here && box !== 'gpu' ? { borderColor: 'var(--ink-2)', fontWeight: 600, boxShadow: isHot ? 'inset 0 0 0 2px var(--ink)' : undefined } : undefined}>
                {here ? `p${i}` : ''}
              </span>
            );
          })}
        </div>
      </div>
    );
  };
  if (mode === 'one')
    return (
      <div className="panel" data-memory="one">
        <div className="mx-label">
          <Translated vi="một bộ nhớ hệ thống, CPU và GPU cùng dùng">one system memory, used by CPU and GPU</Translated>
        </div>
        {bufs.length ? bufs.map(([b]) => row('one', b)) : (
          <p className="muted" style={{ margin: 0 }}>
            <Translated vi="Chưa cấp phát gì.">Nothing allocated yet.</Translated>
          </p>
        )}
      </div>
    );
  const where = (b: string, box: 'cpu' | 'gpu') => {
    const k = view[b].kind;
    return k === 'managed' || (k === 'host' ? box === 'cpu' : box === 'gpu');
  };
  return (
    <div className="cols" style={{ gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }}>
      {(['cpu', 'gpu'] as const).map((box) => (
        <div key={box} className="panel" data-memory={box}>
          <div className={'mx-label' + (box === 'gpu' ? ' t-global' : '')}>{box === 'cpu' ? tr('CPU memory', 'bộ nhớ CPU') : tr('GPU memory', 'bộ nhớ GPU')}</div>
          {bufs.filter(([b]) => where(b, box)).map(([b]) => row(box, b))}
          {!bufs.some(([b]) => where(b, box)) ? (
            <p className="muted" style={{ margin: 0 }}>
              <Translated vi="Không có gì ở đây.">Nothing here.</Translated>
            </p>
          ) : null}
        </div>
      ))}
    </div>
  );
}

function CounterRow({ c, mode, status }: { c: Counters; mode: Mode; status: ReactNode }) {
  const items: [string, number][] = [
    ['pages to GPU', c.h2d], // i18n-ignore: data key; shown through tr() below
    ['pages to CPU', c.d2h], // i18n-ignore: data key; shown through tr() below
    ['GPU faults', c.gpuFaults], // i18n-ignore: data key; shown through tr() below
    ['CPU faults', c.cpuFaults], // i18n-ignore: data key; shown through tr() below
  ];
  if (mode === 'one') items.push(['copied inside memory', c.copiedInMemory]); // i18n-ignore: data key; shown through tr() below
  return (
    <div className="row" style={{ margin: '0.8rem 0 0' }}>
      {items.map(([label, v]) => (
        <span key={label} data-counter={label}>
          {tr(
            label,
            (
              {
                'pages to GPU': 'trang lên GPU',
                'pages to CPU': 'trang về CPU',
                'GPU faults': 'lỗi trang GPU',
                'CPU faults': 'lỗi trang CPU',
                'copied inside memory': 'sao chép trong bộ nhớ',
              } as Record<string, string>
            )[label] ?? label,
          )}{' '}
          <b data-value="">{v}</b>
        </span>
      ))}
      <span className="muted" data-status="">{status}</span>
    </div>
  );
}

function FigStepper() {
  const narrow = useNarrow();
  const [mode, setMode] = useState<Mode>('migrate');
  const [program, setProgram] = useState<Program>('managed');
  const [half, setHalf] = useState(false);
  const [noSync, setNoSync] = useState(false);
  const [n, setN] = useState(0);
  const runs = useMemo(
    () => Object.fromEntries(PROGRAMS.map((p) => [p, runFig(p, mode, { half, noSync })])) as Record<Program, FigRun>,
    [mode, half, noSync],
  );
  const fr = runs[program];
  const order = useMemo(() => steppable(fr.lines), [fr]);
  const total = order.length;
  const at = Math.min(n, total);
  useEffect(() => setN(0), [mode, program, half, noSync]);

  const curLine = at > 0 ? order[at - 1] : -1;
  const curStepIdx = curLine >= 0 ? fr.stepOf[curLine] : -1;
  const curStep = curStepIdx >= 0 ? fr.run.steps[curStepIdx] : null;
  // The last operation that has run so far decides what the memories look like.
  let lastIdx = -1;
  for (let k = 0; k < at; k++) if (fr.stepOf[order[k]] >= 0) lastIdx = fr.stepOf[order[k]];
  const last = lastIdx >= 0 ? fr.run.steps[lastIdx] : null;
  const view = last?.view ?? fr.run.initial;
  const counters = last?.counters ?? { h2d: 0, d2h: 0, gpuFaults: 0, cpuFaults: 0, remote: 0, invalidations: 0, evictions: 0, copiedInMemory: 0 };
  const hot = new Set((curStep?.events ?? []).filter((e) => e.page >= 0 && e.e !== 'local').map((e) => `${e.buf}:${e.page}`));
  const crashed = !!last?.crashed;
  const status = crashed ? tr('program stopped', 'chương trình đã dừng') : last?.active ? tr('kernel may be running', 'kernel có thể đang chạy') : at > 0 ? tr('no kernel running', 'không có kernel nào đang chạy') : '';

  const cell = (p: Program, idx: number | null) => {
    if (idx === null) return <span style={lnStyle(false, false, false)} />;
    const l = runs[p].lines[idx];
    const hl = p === program && idx === curLine;
    return (
      <span style={lnStyle(l.changed, l.deleted, hl)} data-line={idx} data-program={p} data-current={hl ? 'yes' : undefined}
        data-changed={l.changed ? 'yes' : undefined} data-deleted={l.deleted ? 'yes' : undefined}>
        {'  ' + (l.text.replace(/^ {4}/, '') || ' ')}
      </span>
    );
  };

  return (
    <Bench
      title={tr('Fig. C.1, line by line', 'Fig. C.1, từng dòng một')}
      task={tr(
        'Step the managed port in full mode first. Then change one thing at a time.',
        'Trước hết hãy chạy từng bước bản chuyển managed ở chế độ đầy đủ. Sau đó mỗi lần thay đổi một thứ.',
      )}
      foot={
        <Translated
          vi={
            <>
              Các dòng in đậm khác với bản CPU gốc (Fig. C.1 in chúng màu đỏ). Các trang được đánh số từ p0 đến p
              {FIG_PAGES - 1}; một trang được vẽ trong một bộ nhớ nghĩa là bản sao hợp lệ của nó nằm ở đó. Các đoạn mã là{' '}
              <code>labs/appC/snippets/{PROGRAM_FILE.cpu}</code>, <code>{PROGRAM_FILE.explicit}</code> và{' '}
              <code>{PROGRAM_FILE.managed}</code>, front-end-checked.
            </>
          }
        >
          <>
            Bold lines differ from the CPU original (Fig. C.1 prints them in red). Pages are numbered p0 to p{FIG_PAGES - 1}; a
            page drawn in a memory has its valid copy there. The listings are{' '}
            <code>labs/appC/snippets/{PROGRAM_FILE.cpu}</code>, <code>{PROGRAM_FILE.explicit}</code> and{' '}
            <code>{PROGRAM_FILE.managed}</code>, front-end-checked.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Hệ thống">System</Translated>
          <select value={mode} onChange={(e) => setMode(e.target.value as Mode)} aria-label={tr('System', 'Hệ thống')} style={{ maxWidth: '100%' }}>
            {MODES.map((m) => <option key={m} value={m}>{MODE_LABEL[m]}</option>)}
          </select>
        </label>
        <span className="seg" role="group" aria-label={tr('Program to step', 'Chương trình để chạy từng bước')}>
          {PROGRAMS.map((p) => (
            <button key={p} aria-pressed={p === program} onClick={() => setProgram(p)}>
              {tr(PROGRAM_NAME[p], { cpu: 'bản CPU gốc', explicit: 'sao chép tường minh', managed: 'managed (Fig. C.1)' }[p])}
            </button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="check">
          <input type="checkbox" checked={half} onChange={(e) => setHalf(e.target.checked)} />{' '}
          <Translated vi="kernel chỉ chạm tới nửa đầu">kernel touches only the first half</Translated>
        </label>
        <label className="check">
          <input type="checkbox" checked={noSync} onChange={(e) => setNoSync(e.target.checked)} />{' '}
          <Translated vi={<>xóa <code>cudaDeviceSynchronize()</code></>}>
            <>delete <code>cudaDeviceSynchronize()</code></>
          </Translated>
        </label>
      </div>

      <div role="group" aria-label={tr('The three listings', 'Ba đoạn mã')}
        style={{ display: 'grid', gridTemplateColumns: 'repeat(3, max-content)', gap: '0 0.6rem', fontSize: '0.71rem', lineHeight: 1.6, overflowX: 'auto' }}>
        {PROGRAMS.map((p, k) => (narrow && p !== program ? null : (
          <div key={p} data-listing={p} data-live={p === program ? 'yes' : 'no'}
            style={{ background: 'var(--cell)', border: `1px solid ${p === program ? 'var(--ink-3)' : 'var(--rule)'}`, borderRadius: 4, paddingBottom: '0.3rem', opacity: p === program ? 1 : 0.62 }}>
            <div style={{ fontSize: '0.78rem', fontWeight: 620, padding: '0.25rem 0.4rem', borderBottom: '1px solid var(--rule)', marginBottom: '0.2rem' }}>
              {tr(PROGRAM_NAME[p], { cpu: 'bản CPU gốc', explicit: 'sao chép tường minh', managed: 'managed (Fig. C.1)' }[p])}
              {p === 'explicit' ? tr(' (ours)', ' (của chúng tôi)') : ''}
            </div>
            <span style={lnStyle(false, false, false)}>{FIG_HEAD}</span>
            {ROWS.map((r, i) => <div key={i}>{cell(p, r[k])}</div>)}
            <span style={lnStyle(false, false, false)}>{'}'}</span>
          </div>
        )))}
      </div>
      {narrow ? (
        <p className="muted" style={{ margin: '0.4rem 0 0' }}>
          <Translated vi="Trên màn hình rộng hơn, ba đoạn mã nằm cạnh nhau.">On a wider screen the three listings sit side by side.</Translated>
        </p>
      ) : null}

      <div className="row" style={{ margin: '0.9rem 0 0.7rem' }}>
        <button onClick={() => setN(0)} disabled={at === 0}>
          <Translated vi="Bắt đầu lại">Restart</Translated>
        </button>
        <button onClick={() => setN(at - 1)} disabled={at === 0}>
          <Translated vi="Lùi">Back</Translated>
        </button>
        <button className="primary" onClick={() => setN(at + 1)} disabled={at >= total}>
          <Translated vi="Bước">Step</Translated>
        </button>
        <button onClick={() => setN(total)} disabled={at >= total}>
          <Translated vi="Chạy đến cuối">Run to the end</Translated>
        </button>
        <span className="muted" aria-live="polite">
          <Translated vi={<>dòng {at} trên {total}</>}>
            <>line {at} of {total}</>
          </Translated>
        </span>
      </div>
      <div className={'narrate' + (curStep?.errors.length ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }}>
        {curLine >= 0 ? (
          <>
            <code>{fr.lines[curLine].text.trim()}</code> {describe(program, mode, fr.lines[curLine], curStep, fr)}
          </>
        ) : (
          <Translated
            vi={
              <>
                Chưa có gì chạy. Bấm Bước để chạy dòng đầu tiên của đoạn mã "
                {tr(PROGRAM_NAME[program], { cpu: 'bản CPU gốc', explicit: 'sao chép tường minh', managed: 'managed (Fig. C.1)' }[program])}".
              </>
            }
          >
            <>Nothing has run. Press Step to run the first line of the listing "{PROGRAM_NAME[program]}".</>
          </Translated>
        )}
      </div>
      <Memories view={view} mode={mode} hot={hot} />
      <CounterRow c={counters} mode={mode} status={status} />

      <div className="mx-label" style={{ margin: '1.2rem 0 0.3rem' }}>
        <Translated
          vi={
            <>
              Toàn bộ chương trình, {MODE_SHORT[mode]}{half ? ', kernel chạm tới một nửa' : ''}{noSync ? ', đã xóa lệnh đồng bộ' : ''}
            </>
          }
        >
          <>
            Whole program, {MODE_SHORT[mode]}{half ? ', kernel touches half' : ''}{noSync ? ', synchronise deleted' : ''}
          </>
        </Translated>
      </div>
      <table className="data" style={{ margin: 0, background: 'var(--page)' }} aria-label={tr('Totals for the three programs', 'Tổng cho ba chương trình')}>
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}>
              <Translated vi="Chương trình">Program</Translated>
            </th>
            <th className="num">
              <Translated vi="lên GPU">to GPU</Translated>
            </th>
            <th className="num">
              <Translated vi="về CPU">to CPU</Translated>
            </th>
            <th className="num">
              <Translated vi="lỗi trang GPU">GPU faults</Translated>
            </th>
            <th className="num">
              <Translated vi="lỗi trang CPU">CPU faults</Translated>
            </th>
            {mode === 'one' ? (
              <th className="num">
                <Translated vi="sao chép trong bộ nhớ">copied inside memory</Translated>
              </th>
            ) : null}
            <th>
              <Translated vi="Kết quả">Outcome</Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          {PROGRAMS.map((p) => {
            const r = runs[p].run;
            const e = r.errors[0];
            return (
              <tr key={p} data-program={p} data-moved={moved(r.counters)}>
                <td style={{ paddingLeft: '0.6rem' }}>
                  {tr(PROGRAM_NAME[p], { cpu: 'bản CPU gốc', explicit: 'sao chép tường minh', managed: 'managed (Fig. C.1)' }[p])}
                </td>
                <td className="num">{r.counters.h2d}</td>
                <td className="num">{r.counters.d2h}</td>
                <td className="num">{r.counters.gpuFaults}</td>
                <td className="num">{r.counters.cpuFaults}</td>
                {mode === 'one' ? <td className="num">{r.counters.copiedInMemory}</td> : null}
                <td data-outcome={e ? e.kind : 'ok'}>
                  {e ? (
                    <span className="chip fail">
                      {e.kind === 'segfault' ? tr('segmentation fault', 'segmentation fault') : e.kind === 'race' ? tr('race', 'race condition') : tr('ordering error', 'lỗi thứ tự')}
                    </span>
                  ) : (
                    <span className="chip ok">{tr('correct', 'đúng')}</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Bench>
  );
}

function ShapesTable() {
  return (
    <table className="data wide" style={{ maxWidth: '58rem' }} aria-label={tr('Four shapes of unified memory', 'Bốn dạng unified memory')}>
      <thead>
        <tr>
          <th>
            <Translated vi="Dạng">Shape</Translated>
          </th>
          <th>
            <Translated vi="Ví dụ">Examples</Translated>
          </th>
          <th>
            <Translated vi="Thứ di chuyển">What moves</Translated>
          </th>
          <th>
            <Translated vi="Điều bạn vẫn phải lo">What you still owe</Translated>
          </th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>
            <Translated vi="GPU rời, di chuyển bằng phần mềm">Discrete GPU, migration in software</Translated>
          </td>
          <td>
            <Translated vi="CUDA Unified Memory; bộ nhớ managed của HIP">CUDA Unified Memory; HIP managed memory</Translated>
          </td>
          <td>
            <Translated vi="Các trang, giữa hai bộ nhớ, khi có lỗi trang (hoặc tất cả lúc khởi chạy khi bị hạn chế)">
              Pages, between two memories, on fault (or all at launch when limited)
            </Translated>
          </td>
          <td>
            <Translated vi="Thứ tự; và trên hệ thống hạn chế, CPU không được chạm tới khi kernel đang chạy">
              Ordering; and on limited systems, no CPU touch while a kernel runs
            </Translated>
          </td>
        </tr>
        <tr>
          <td>
            <Translated vi="Hai bộ nhớ vật lý, một bảng trang, nhất quán bằng phần cứng">
              Two physical memories, one page table, coherent in hardware
            </Translated>
          </td>
          <td>Grace Hopper (NVLink-C2C){/* i18n-ignore: product names */}</td>
          <td>
            <Translated vi="Ít lỗi trang hơn: nhất quán ở mức dòng cache, các trang được chuyển theo bộ đếm truy cập">
              Less faulting: coherent at cache-line granularity, pages moved by access counters
            </Translated>
          </td>
          <td>
            <Translated vi="Thứ tự">Ordering</Translated>
          </td>
        </tr>
        <tr>
          <td>
            <Translated vi="Một DRAM chung, quy tắc cache theo từng loại buffer">One shared DRAM, caching rules per buffer type</Translated>
          </td>
          <td>NVIDIA Tegra; Apple silicon{/* i18n-ignore: product names */}</td>
          <td>
            <Translated vi="Không có gì giữa các bộ nhớ. Trên Apple, buffer shared không cần sao chép">
              Nothing between memories. On Apple, a shared buffer needs no copy
            </Translated>
          </td>
          <td>
            <Translated vi="Thứ tự, và chọn loại buffer. Trên một số hệ thống Tegra, bộ nhớ managed theo các quy tắc hạn chế">
              Ordering, and choosing the buffer type. On some Tegra systems managed memory follows the limited rules
            </Translated>
          </td>
        </tr>
        <tr>
          <td>
            <Translated vi="Một HBM chung">One shared HBM</Translated>
          </td>
          <td>AMD MI300A{/* i18n-ignore: product name */}</td>
          <td>
            <Translated vi='Không có gì: "di chuyển trang vật lý vừa không cần thiết vừa không hữu ích"'>
              Nothing: "physical page migration is neither needed nor useful"
            </Translated>
          </td>
          <td>
            <Translated vi="Thứ tự; XNACK vẫn bật, cho các ánh xạ">Ordering; XNACK still on, for the mappings</Translated>
          </td>
        </tr>
      </tbody>
    </table>
  );
}

function PagesReveal({ predicted }: { predicted: number }) {
  const m = moved(runFig('managed', 'migrate').run.counters);
  const e = moved(runFig('explicit', 'migrate').run.counters);
  return (
    <Translated
      vi={
        <p>
          Mô hình cho ra <b>{m}</b>: {m / 2} trang đi sang khi kernel chạm tới chúng, {m / 2} trang quay về khi{' '}
          <code>use_data</code> đọc chúng. Bản chuyển dùng sao chép tường minh chuyển {e}.{' '}
          {predicted === m ? 'Bạn đã đúng.' : predicted < m ? 'Ít hơn mức bạn có thể mong: một con trỏ không có nghĩa là một bộ nhớ.' : 'Mỗi trang đi qua một lần mỗi chiều, không hơn.'}
        </p>
      }
    >
      <p>
        The model says <b>{m}</b>: {m / 2} pages over when the kernel touches them, {m / 2} back when <code>use_data</code> reads
        them. The explicit-copy port moves {e}.{' '}
        {predicted === m ? 'You had it.' : predicted < m ? 'Fewer than you might hope: one pointer does not mean one memory.' : 'Each page crosses once each way, no more.'}
      </p>
    </Translated>
  );
}

export const questionsSC6: Record<string, Question> = {
  'appC/C.6/q-delete-sync': {
    kind: 'choice',
    id: 'appC/C.6/q-delete-sync',
    prompt: (
      <Translated
        vi={
          <>
            Xóa <code>cudaDeviceSynchronize()</code> khỏi <code>sortfile</code> bản managed. Điều gì xảy ra trên một máy trạm
            Windows, và trên Linux với một GPU compute capability 7.x?
          </>
        }
      >
        <>
          Delete <code>cudaDeviceSynchronize()</code> from the managed <code>sortfile</code>. What happens on a Windows
          workstation, and on Linux with a compute capability 7.x GPU?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Cả hai đều chạy: Unified Memory giữ CPU và GPU nhất quán, nên use_data thấy buffer đã sắp xếp.">
            Both work: Unified Memory keeps CPU and GPU coherent, so use_data sees the sorted buffer.
          </Translated>
        ),
        why: (
          <Translated vi="Trang nhất quán không sắp đặt thứ tự gì cả. Không có gì bắt use_data chờ việc sắp xếp, và trên Windows CPU hoàn toàn không được chạm tới bộ nhớ managed khi GPU đang hoạt động.">
            {'Coherent pages do not order anything. Nothing makes use_data wait for the sort, and on Windows the CPU must not touch managed memory at all while the GPU is active.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Windows: use_data gặp segfault, vì chạm tới bộ nhớ managed khi GPU đang hoạt động. Linux: nó có thể đọc dữ liệu chưa sắp xếp.">
            {'Windows: use_data segfaults, touching managed memory while the GPU is active. Linux: it may read unsorted data.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Cả hai đều sập với segmentation fault: không hệ thống nào cho CPU chạm tới bộ nhớ managed trước một lần đồng bộ.">
            {'Both crash with a segmentation fault: no system lets the CPU touch managed memory before a synchronise.'}
          </Translated>
        ),
        why: (
          <Translated vi="Với compute capability 6.0 trở lên trên Linux, CPU được chạm tới bộ nhớ managed khi một kernel đang chạy. Được phép không có nghĩa là đúng.">
            {'With compute capability 6.0 or later on Linux, the CPU may touch managed memory while a kernel runs. Allowed is not the same as correct.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Linux sập, vì các trang gây lỗi trang và quay về CPU trong lúc sắp xếp, còn Windows chạy được.">
            Linux crashes, because pages fault back to the CPU during the sort, and Windows works.
          </Translated>
        ),
        why: (
          <Translated vi="Ngược lại: Windows mới là chế độ hạn chế.">The other way round: Windows is the limited mode.</Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Windows, WSL và một số hệ thống Tegra chạy Unified Memory ở chế độ hạn chế, trong đó mọi truy cập của CPU vào bộ
            nhớ managed khi GPU đang hoạt động, tức là cho đến khi được đồng bộ, đều là segmentation fault. Ví dụ của chính
            NVIDIA ghi <code>y = 20</code> ngay sau một lần khởi chạy: nó chạy trên compute capability 6.x và hỏng trước 6.0
            và trên Windows. Trên Linux với hỗ trợ đầy đủ thì không sập, và đó chính là cái bẫy: <code>use_data</code> giờ
            tranh chấp với việc sắp xếp. Sách nói thẳng: chương trình vẫn phải đồng bộ hóa các truy cập đồng thời.
          </p>
        }
      >
        <p>
          Windows, WSL and some Tegra systems run Unified Memory in the limited mode, where any CPU access to managed memory
          while the GPU is active, that is, until it is synchronised, is a segmentation fault. NVIDIA's own example writes <code>y = 20</code> right after a launch:
          it runs on compute capability 6.x and fails before 6.0 and on Windows. On Linux with full support there is no crash,
          and that is the trap: <code>use_data</code> now races with the sort. The book says it plainly: the program still has
          to synchronise concurrent access.
        </p>
      </Translated>
    ),
    source: 'book §C.6, pp. 626–627; notes/today/appC-facts.md A10',
    vi: { source: 'sách §C.6, pp. 626–627; notes/today/appC-facts.md A10' },
  },
  'appC/C.6/q-oversub-windows': {
    kind: 'choice',
    id: 'appC/C.6/q-oversub-windows',
    prompt: (
      <Translated
        vi={
          <>
            Windows. <code>cudaMallocManaged</code> 1.5 lần dung lượng bộ nhớ GPU, rồi một kernel chạy trên toàn bộ vùng đó.
            Điều gì xảy ra?
          </>
        }
      >
        <>
          Windows. <code>cudaMallocManaged</code> of 1.5 times the GPU's memory, then a kernel over all of it. What happens?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Các trang được nạp theo nhu cầu qua lỗi trang, và các trang ít được dùng gần đây nhất bị đẩy về CPU.">
            Pages fault in on demand, and the least recently used pages are evicted back to the CPU.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là Unified Memory đầy đủ trên Linux. Windows là chế độ hạn chế: không có lỗi trang GPU.">
            That is full Unified Memory on Linux. Windows is the limited mode: no GPU page faults.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó chạy, ở tốc độ của kênh kết nối.">It runs, at the speed of the interconnect.</Translated>
        ),
        why: <Translated vi="Nó không bao giờ đi được đến bước chạy.">It never gets as far as running.</Translated>,
      },
      {
        label: (
          <Translated vi="Sysmem Fallback của driver tràn phần dư sang bộ nhớ hệ thống, nên nó chạy, chậm hơn.">
            {"The driver's Sysmem Fallback spills the excess into system memory, so it runs, more slowly."}
          </Translated>
        ),
        why: (
          <Translated vi="Tính năng đó của driver Windows dành cho các vùng cấp phát thông thường bị hết bộ nhớ GPU. Giới hạn đối với bộ nhớ managed được ghi nhận riêng.">
            {'That Windows driver feature is for ordinary allocations that run out of GPU memory. The limit on managed memory is documented separately.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không được phép: một hệ thống hạn chế không thể cấp phát nhiều bộ nhớ managed hơn dung lượng GPU có.">
            Not allowed: a limited system cannot allocate more managed memory than the GPU has.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hướng dẫn của NVIDIA, về các hệ thống hạn chế: "Chúng không thể cấp phát nhiều bộ nhớ managed hơn kích thước vật
            lý của bộ nhớ GPU." Đó là giới hạn thứ ba của Kepler và Maxwell trong sách, vẫn còn hiệu lực ở bất cứ đâu{' '}
            <code>cudaDevAttrConcurrentManagedAccess</code> bằng 0. Cấp phát vượt dung lượng cần Unified Memory đầy đủ, thứ mà
            mục C.8 mổ xẻ.
          </p>
        }
      >
        <p>
          NVIDIA's guide, on limited systems: "They cannot allocate more managed memory than the physical size of GPU memory."
          That is the book's third Kepler and Maxwell limit, still in force wherever <code>cudaDevAttrConcurrentManagedAccess</code>{' '}
          is 0. Oversubscription needs full Unified Memory, which section C.8 takes apart.
        </p>
      </Translated>
    ),
    source: 'book §C.6, p. 627; notes/today/appC-facts.md A10, A18',
    vi: { source: 'sách §C.6, p. 627; notes/today/appC-facts.md A10, A18' },
  },
  'appC/C.6/q-same-bytes': {
    kind: 'choice',
    id: 'appC/C.6/q-same-bytes',
    prompt: (
      <Translated
        vi={
          <>
            GPU rời, Unified Memory đầy đủ, và kernel chạm tới mọi byte của tệp một lần. So với bản chuyển dùng sao chép
            tường minh, bao nhiêu byte đi qua đường truyền?
          </>
        }
      >
        <>
          Discrete GPU, full Unified Memory, and the kernel touches every byte of the file once. Compared with the explicit-copy
          port, how many bytes cross the link?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Khoảng một nửa: chỉ các trang kernel chạm tới mới di chuyển, phần còn lại nằm yên.">
            About half: only the pages the kernel touches move, and the rest stay put.
          </Translated>
        ),
        why: (
          <Translated vi="Nó chạm tới tất cả, nên tất cả đều di chuyển, rồi quay về khi use_data đọc.">
            It touches all of it, so all of it moves, and comes back when use_data reads it.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Gần như bằng nhau: mỗi trang đi sang một lần và về một lần.">
            About the same: every page in once and out once.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Không byte nào: con trỏ được dùng chung.">None: the pointer is shared.</Translated>,
        why: (
          <Translated vi="Con trỏ được dùng chung; bộ nhớ thì không. Trên GPU rời, các byte vẫn phải đến được bộ nhớ GPU.">
            The pointer is shared; the memory is not. On a discrete GPU the bytes still have to reach GPU memory.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Khoảng gấp đôi: mỗi trang gây lỗi trang hai lần.">About double: every page faults twice.</Translated>
        ),
        why: (
          <Translated vi="Mỗi trang gây lỗi trang một lần mỗi chiều: mỗi chiều đi qua một lần, đúng như hai lần sao chép.">
            Each page faults once in each direction: one crossing each way, as the two copies make.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong trình chạy từng bước, chế độ đầy đủ với kernel chạm tới mọi thứ: {FIG_PAGES} trang đi sang và {FIG_PAGES}{' '}
            trang quay về, đúng bằng {2 * FIG_PAGES} trang mà hai lời gọi <code>cudaMemcpy</code> chuyển. Bộ nhớ managed tiết
            kiệm được một buffer thứ hai và ba dòng mã, không phải lưu lượng. NVIDIA đã nói vậy từ lúc ra mắt: một chương
            trình được tinh chỉnh kỹ dùng stream và <code>cudaMemcpyAsync</code> "rất có thể chạy tốt hơn" một chương trình
            chỉ dùng Unified Memory. Nơi nó thực sự tiết kiệm lưu lượng là khi kernel chỉ chạm tới một phần dữ liệu.
          </p>
        }
      >
        <p>
          In the stepper, full mode with the kernel touching everything: {FIG_PAGES} pages over and {FIG_PAGES} back, the same{' '}
          {2 * FIG_PAGES} the two <code>cudaMemcpy</code> calls move. Managed memory saved a second buffer and three lines, not
          traffic. NVIDIA said as much at launch: a carefully tuned program using streams and <code>cudaMemcpyAsync</code> "may
          very well perform better" than one that only uses Unified Memory. Where it does save traffic is when the kernel
          touches part of the data.
        </p>
      </Translated>
    ),
    source: 'the stepper\'s model (src/sim/managedmem.ts); notes/today/appC-facts.md A9',
    vi: { source: 'mô hình của trình chạy từng bước (src/sim/managedmem.ts); notes/today/appC-facts.md A9' },
  },
  'appC/C.6/q-apple': {
    kind: 'choice',
    id: 'appC/C.6/q-apple',
    prompt: (
      <Translated
        vi={
          <>
            Bạn chuyển <code>sortfile</code> sang Metal trên Apple silicon với buffer shared. Những thay đổi nào của Fig. C.1
            có thứ tương ứng?
          </>
        }
      >
        <>
          You port <code>sortfile</code> to Metal on Apple silicon with shared buffers. Which of Fig. C.1's changes have a
          counterpart?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Cả bốn, cộng thêm một lời gọi didModifyRange sau fread để GPU thấy những gì CPU đã ghi.">
            All four, plus a didModifyRange call after fread so that the GPU sees what the CPU wrote.
          </Translated>
        ),
        why: (
          <Translated vi='didModifyRange thuộc về lưu trữ managed, thứ đã bị deprecated và "không có tác dụng trên Apple Silicon". Và Metal không có lời gọi free.'>
            {'didModifyRange belongs to managed storage, which is deprecated and "has no effect on Apple Silicon". And Metal has no free call.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một buffer shared, lần khởi chạy, và một lần chờ trước use_data. Không có gì di chuyển.">
            A shared buffer, the launch, and a wait before use_data. Nothing migrates.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không cái nào: với unified memory bạn không cần chờ.">None: with unified memory you need no wait.</Translated>
        ),
        why: (
          <Translated vi="Bạn vẫn phải lo thứ tự: Apple yêu cầu bạn bảo đảm các thay đổi của GPU hoàn tất trước khi CPU đọc chúng.">
            {"You still owe ordering: Apple asks you to make sure the GPU's changes complete before the CPU reads them."}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Ba trong bốn thay đổi còn lại: phần cấp phát (một buffer có lưu trữ shared, mà CPU ghi vào <code>contents</code>{' '}
            của nó), lần khởi chạy (mã hóa và commit), và lần chờ (<code>waitUntilCompleted</code>). Thay đổi thứ tư không
            có: buffer Metal là các đối tượng, được giải phóng như mọi đối tượng khác, không có lời gọi free. Không có chế độ
            managed nào để di chuyển gì cả, và cũng không cần.
          </p>
        }
      >
        <p>
          Three of the four changes survive: the allocation (a buffer with shared storage, whose <code>contents</code> the CPU
          writes), the launch (encode and commit), and the wait (<code>waitUntilCompleted</code>). The fourth has none: Metal
          buffers are objects, released like any other, with no free call. There is no managed mode to migrate anything, and
          none is needed.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch05-facts.md C6; notes/today/appC-facts.md C2, C5 (citing ch02-facts C2)',
    vi: { source: 'notes/today/ch05-facts.md C6; notes/today/appC-facts.md C2, C5 (trích dẫn ch02-facts C2)' },
  },
  'appC/C.6/q-hip-xnack': {
    kind: 'choice',
    id: 'appC/C.6/q-hip-xnack',
    prompt: (
      <Translated
        vi={
          <>
            <code>hipMallocManaged</code> trên một GPU AMD với <code>HSA_XNACK</code> chưa được đặt. Kernel đọc dữ liệu từ đâu?
          </>
        }
      >
        <>
          <code>hipMallocManaged</code> on an AMD GPU with <code>HSA_XNACK</code> unset. Where does the kernel read the data from?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Từ bộ nhớ device, sau khi mỗi trang di chuyển sang đó ở lần chạm đầu, như trên NVIDIA.">
            From device memory, after each page migrates there on its first touch, as on NVIDIA.
          </Translated>
        ),
        why: (
          <Translated vi="Di chuyển theo truy cập cần HSA_XNACK=1 và một driver có HMM; thiếu chúng thì tính năng này bị tắt.">
            Access-driven migration needs HSA_XNACK=1 and a driver with HMM; without them it is disabled.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không từ đâu cả: không có XNACK, hipMallocManaged trả về lỗi và không cấp phát gì.">
            Nowhere: without XNACK, hipMallocManaged returns an error and allocates nothing.
          </Translated>
        ),
        why: (
          <Translated vi='Lời gọi vẫn chạy. AMD nói khi đó bộ nhớ managed hoạt động "theo cách suy giảm".'>
            {'The call still works. AMD says managed memory then operates "in a degraded fashion".'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Từ bộ nhớ host, qua đường truyền, như thể đó là bộ nhớ host được ghim.">
            From host memory, over the link, as if it were pinned host memory.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Từ bộ nhớ device, sau khi runtime sao chép toàn bộ vùng cấp phát một lượt lúc khởi chạy.">
            From device memory, after the runtime copies the whole allocation in bulk at the launch.
          </Translated>
        ),
        why: (
          <Translated vi="Không phải điều AMD ghi nhận cho trường hợp này: header nói khi đó lời gọi hoạt động giống hipMallocHost, vốn cấp phát bộ nhớ host.">
            {'Not what AMD documents for this case: the header says the call then behaves the same as hipMallocHost, which allocates host memory.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            AMD: không có <code>HSA_XNACK=1</code> và một kernel driver có HMM, "di chuyển bộ nhớ theo truy cập sẽ bị tắt", và
            header nói <code>hipMallocManaged</code> khi đó "hoạt động giống hipMallocHost". Kernel đọc bộ nhớ host được ghim
            qua đường truyền, tức là zero-copy của C.2, không phải Unified Memory của C.6.
          </p>
        }
      >
        <p>
          AMD: without <code>HSA_XNACK=1</code> and a kernel driver with HMM, "access-driven memory migration will be disabled",
          and the header says <code>hipMallocManaged</code> then "behaves the same as hipMallocHost". The kernel reads pinned host
          memory over the link, which is C.2's zero-copy, not C.6's Unified Memory.
        </p>
      </Translated>
    ),
    source: 'notes/today/appC-facts.md B5',
    vi: { source: 'notes/today/appC-facts.md B5' },
  },
};

export function SC6() {
  return (
    <>
      <Translated
        vi={
          <p>
            CUDA 6 thêm <b>Unified Memory</b>: một vùng các trang <b>managed</b> mà CPU và GPU dùng chung qua một con trỏ.
            Một trang managed có thể nằm trong bộ nhớ CPU, trong bộ nhớ GPU, hoặc cả hai, và runtime CUDA cùng phần cứng di
            chuyển nó và giữ các bản sao nhất quán. Với mã CPU, bộ nhớ managed trông như bộ nhớ CPU; với kernel, nó trông như
            bộ nhớ GPU. Điều nó không làm là sắp đặt thứ tự các truy cập của bạn: nếu CPU và GPU có thể chạm tới cùng dữ liệu
            cùng lúc, chương trình vẫn phải đồng bộ hóa chúng.
          </p>
        }
      >
        <p>
          CUDA 6 added <b>Unified Memory</b>: a pool of <b>managed</b> pages that the CPU and the GPU share through one pointer.
          A managed page can live in CPU memory, in GPU memory, or in both, and the CUDA runtime and the hardware move it and
          keep the copies coherent. To CPU code managed memory looks like CPU memory; to a kernel it looks like GPU memory.
          What it does not do is order your accesses: if the CPU and the GPU may touch the same data at the same time, the
          program still has to synchronise them.
        </p>
      </Translated>
      <Source section="C.6" pages="626" />
      <Translated
        vi={
          <p>
            Giao diện chỉ là một lời gọi. <code>cudaMallocManaged(&amp;ptr, bytes)</code> trả về một con trỏ mà cả mã host lẫn
            kernel đều dereference được, và <code>cudaFree</code> giải phóng nó. Không có buffer thứ hai, không có lời gọi sao
            chép: hệ thống quyết định khi nào byte di chuyển. Mục này nói về những gì điều đó mang lại và cái giá của nó.
          </p>
        }
      >
        <p>
          The interface is one call. <code>cudaMallocManaged(&amp;ptr, bytes)</code> returns a pointer that host code and kernels
          may both dereference, and <code>cudaFree</code> releases it. No second buffer, no copy calls: the system decides when
          bytes move. This section is about what that buys and what it costs.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/appC-facts.md, A9)', '2026-11-20 (notes/today/appC-facts.md, A9)')}>
        <Translated
          vi={
            <p>
              Sách nói năm 2013. NVIDIA công bố Unified Memory vào tháng 11 năm 2013, bộ công cụ phát hành tháng 4 năm 2014,
              và chính NVIDIA giờ nói "ra mắt năm 2014". Chữ ký ngày nay là{' '}
              <code>cudaMallocManaged(void **devPtr, size_t size, unsigned int flags = cudaMemAttachGlobal)</code>; bộ nhớ
              không được xóa về 0. Biến <code>__managed__</code> và các memory pool managed là hai cách khác để có nó.
            </p>
          }
        >
          <p>
            The book says 2013. NVIDIA announced Unified Memory in November 2013, the toolkit shipped in April 2014, and NVIDIA
            itself now says "introduced in 2014". Today's signature is{' '}
            <code>cudaMallocManaged(void **devPtr, size_t size, unsigned int flags = cudaMemAttachGlobal)</code>; the memory is
            not cleared. <code>__managed__</code> variables and managed memory pools are two other ways to get it.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Bốn thay đổi cho một hàm CPU">Four changes to a CPU function</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Fig. C.1 là một hàm CPU nhỏ: cấp phát một buffer, đọc một tệp vào đó, sắp xếp các byte, chuyển chúng đi tiếp. Bản
            chuyển managed thay đổi bốn dòng. <code>malloc</code> thành <code>cudaMallocManaged</code>, việc sắp xếp thành một
            lần khởi chạy kernel, một <code>cudaDeviceSynchronize()</code> xuất hiện ở chỗ bản CPU có một dòng trống, và{' '}
            <code>free</code> thành <code>cudaFree</code>. Các dòng <code>fread</code> và <code>use_data</code> không đổi: CPU
            ghi và đọc qua chính con trỏ mà kernel sắp xếp. Ý của sách là việc chuyển mã: phía host của bản chuyển nhỏ. Một
            kernel sắp xếp song song vẫn phải tồn tại.
          </p>
        }
      >
        <p>
          Fig. C.1 is a small CPU function: allocate a buffer, read a file into it, sort the bytes, hand them on. The managed
          port changes four lines. <code>malloc</code> becomes <code>cudaMallocManaged</code>, the sort becomes a kernel launch,
          a <code>cudaDeviceSynchronize()</code> appears where the CPU version had a blank line, and <code>free</code> becomes{' '}
          <code>cudaFree</code>. The <code>fread</code> and the <code>use_data</code> lines do not change: the CPU writes and
          reads through the same pointer the kernel sorts. The book's point is porting: the host side of the port is small. A
          parallel sort kernel still has to exist.
        </p>
      </Translated>
      <Source section="C.6" pages="627" figs="Fig. C.1" />
      <Translated
        vi={
          <p>
            Trình chạy từng bước thêm một cột thứ ba, cùng bản chuyển đó làm theo cách của chương 2: một buffer thứ hai từ{' '}
            <code>cudaMalloc</code> và hai lời gọi <code>cudaMemcpy</code>. Rồi sau mỗi dòng, nó hỏi mỗi trang của buffer đang
            ở đâu.
          </p>
        }
      >
        <p>
          The stepper adds a third column, the same port done Chapter 2's way: a second buffer from <code>cudaMalloc</code> and
          two <code>cudaMemcpy</code> calls. Then it asks, after every line, where each page of the buffer is.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Cột sao chép tường minh là của chúng tôi. Fig. C.1 là ví dụ CUDA 6 của chính NVIDIA, chỉ so sánh bản CPU với bản
              managed (appC-facts A9).
            </p>
          }
        >
          <p>
            The explicit-copy column is ours. Fig. C.1 is NVIDIA's own CUDA 6 example, which compares only the CPU version and
            the managed one (appC-facts A9).
          </p>
        </Translated>
      </Beyond>
      <Predict
        id="appC/C.6/p-pages"
        prompt={
          <Translated
            vi={
              <>
                Một GPU rời với Unified Memory đầy đủ. Tệp chiếm {FIG_PAGES} trang và việc sắp xếp chạm tới tất cả. Trên toàn
                bộ <code>sortfile</code> bản managed, có bao nhiêu lượt trang đi qua giữa bộ nhớ CPU và bộ nhớ GPU, cộng cả hai
                chiều?
              </>
            }
          >
            <>
              A discrete GPU with full Unified Memory. The file fills {FIG_PAGES} pages and the sort touches all of them. Over the
              whole managed <code>sortfile</code>, how many page crossings between CPU and GPU memory, both directions together?
            </>
          </Translated>
        }
        unit={tr('pages', 'trang')}
      >
        {(v) => <PagesReveal predicted={v} />}
      </Predict>
      <Translated
        vi={
          <p>
            Trước hết hãy chạy từng bước bản chuyển managed ở chế độ đầy đủ, theo dõi hai bộ nhớ. Bảng dưới trình chạy từng
            bước ghi điểm cho cả ba chương trình trên hệ thống bạn chọn.
          </p>
        }
      >
        <p>
          Step the managed port in full mode first, watching the two memories. The table under the stepper keeps the score for
          all three programs on the system you choose.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Tính theo trang, không theo byte hay thời gian, và chỉ có {FIG_PAGES} trang. Một trang quay về CPU ở lần chạm
              đầu tiên của CPU sau lần đồng bộ; hướng dẫn của NVIDIA chỉ nói rằng các hệ thống hạn chế di chuyển bộ nhớ managed
              về "khi GPU được đồng bộ" và phía CPU có thể gặp lỗi trang. Lần chạm đầu đại diện cho các heuristic đặt chỗ của
              driver. Một kernel thực hiện mọi truy cập của nó ngay lúc khởi chạy.
            </p>
          }
        >
          <p>
            Pages, not bytes or time, and {FIG_PAGES} of them. A page goes back to the CPU on the CPU's first touch after the
            synchronise; NVIDIA's guide says only that limited systems migrate managed memory back "when the GPU is
            synchronized" and that the CPU side can fault. First touch stands in for the driver's placement heuristics. A
            kernel makes all its accesses at its launch.
          </p>
        </Translated>
      </Simplification>
      <FigStepper />
      <Translated
        vi={
          <p>
            Có ba điều cần tìm. Với kernel chạm tới mọi thứ, hãy so sánh tổng của bản chuyển managed với tổng của các lần sao
            chép tường minh. Rồi tích <i>kernel chỉ chạm tới nửa đầu</i> và so sánh lại, ở chế độ đầy đủ và chế độ hạn chế:
            chương trình nào trong ba chương trình giờ di chuyển ít hơn? Cuối cùng, chuyển hệ thống sang Apple silicon và xem
            còn lại gì để di chuyển.
          </p>
        }
      >
        <p>
          Three things to find. With the kernel touching everything, compare the managed port's total with the explicit
          copies'. Then tick <i>kernel touches only the first half</i> and compare again, in full mode and in limited mode:
          which of the three programs now moves less? Last, switch the system to Apple silicon and see what is left to move.
        </p>
      </Translated>
      <Ask q={questionsSC6['appC/C.6/q-same-bytes']} />
      <Translated
        vi={
          <p>
            Các đoạn mã là những tệp vượt qua kiểm tra front-end CUDA, nên có hai chỗ khác với trang sách. Hình in{' '}
            <code>cudaDeviceSynchronize()</code> không có dấu chấm phẩy; như được in, kiểm tra front-end CUDA dừng ngay tại
            đó. Và nó lược cấu hình khởi chạy thành <code>&lt;&lt;&lt;...&gt;&gt;&gt;</code>; bản của chúng tôi đặt tên chúng là{' '}
            <code>grid</code> và <code>block</code>, kèm một chú thích nói rằng chúng đại diện cho bất cứ gì một kernel sắp xếp
            thật sẽ chọn.
          </p>
        }
      >
        <p>
          The listings are files that pass the CUDA front-end check, so two things differ from the page. The figure prints{' '}
          <code>cudaDeviceSynchronize()</code> without a semicolon; as printed, the CUDA front-end check stops right there. And
          it elides the launch configuration as <code>&lt;&lt;&lt;...&gt;&gt;&gt;</code>; ours names it <code>grid</code> and{' '}
          <code>block</code>, with a comment saying they stand in for whatever a real sort would choose.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Hai lỗi nhỏ khác trên trang sách không ảnh hưởng đến mã. Văn bản gọi hàm bị thay là <code>qsort()</code> trong khi
            hình gọi nó là <code>qsort_char</code>; các tệp giữ tên trong hình. Và tiêu đề của hình ghi "CUDA
            6&nbsp;&nbsp;code", với hai dấu cách.
          </p>
        }
      >
        <p>
          Two other slips on the page do not touch the code. The text calls the replaced function <code>qsort()</code> while
          the figure calls it <code>qsort_char</code>; the files keep the figure's name. And the figure's heading reads "CUDA
          6&nbsp;&nbsp;code", with a double space.
        </p>
      </Translated>
      <Source section="C.6" pages="627" figs="Fig. C.1" />

      <h2>
        <Translated vi="Những gì Kepler và Maxwell không làm được">What Kepler and Maxwell could not do</Translated>
      </h2>
      <Translated
        vi={
          <p>
            CUDA 6 chạy trên các GPU không thể giữ bộ nhớ CPU và GPU nhất quán bằng phần cứng, nên phần lớn việc di chuyển
            được làm bằng phần mềm, với ba giới hạn. Mọi bộ nhớ managed mà CPU đã sửa đều bị đẩy sang GPU trước mỗi lần khởi
            chạy. CPU và GPU không thể truy cập bộ nhớ managed cùng lúc. Và vùng managed không thể lớn hơn bộ nhớ vật lý của
            GPU.
          </p>
        }
      >
        <p>
          CUDA 6 ran on GPUs that could not keep CPU and GPU memory coherent in hardware, so most of the migration was done in
          software, with three limits. All managed memory the CPU had modified was flushed to the GPU before every launch. The
          CPU and the GPU could not access managed memory at the same time. And the managed pool could be no larger than the
          GPU's physical memory.
        </p>
      </Translated>
      <Source section="C.6" pages="627" />
      <Translated
        vi={
          <p>
            Đây không phải lịch sử. NVIDIA ghi nhận đúng hành vi này ngày nay ở bất cứ đâu{' '}
            <code>cudaDevAttrConcurrentManagedAccess</code> bằng 0, nghĩa là mọi hệ thống Windows, WSL, và một số hệ thống
            Tegra: bộ nhớ managed chuyển sang GPU khi một kernel bắt đầu, CPU không được chạm tới nó khi GPU đang hoạt động
            ("Các truy cập đồng thời của CPU/GPU, kể cả vào các vùng cấp phát bộ nhớ managed khác nhau, sẽ gây ra
            segmentation fault"), nó di chuyển về khi GPU được đồng bộ, và không được cấp phát vượt dung lượng. Đó là chế độ
            hạn chế của trình chạy từng bước.
          </p>
        }
      >
        <p>
          This is not history. NVIDIA documents exactly this behaviour today wherever{' '}
          <code>cudaDevAttrConcurrentManagedAccess</code> is 0, which means every Windows system, WSL, and some Tegra systems:
          managed memory moves to the GPU when a kernel starts, the CPU must not touch it while the GPU is active ("Concurrent
          CPU/GPU accesses, even to different managed memory allocations, will cause a segmentation fault"), it migrates back
          when the GPU is synchronised, and oversubscription is not allowed. That is the stepper's limited mode.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/appC-facts.md, A10)', '2026-11-20 (notes/today/appC-facts.md, A10)')}>
        <Translated
          vi={
            <p>
              Các kiến trúc trong sách mang compute capability 3.x (Kepler) và 5.x (Maxwell); Unified Memory đầy đủ cần 6.0
              (Pascal) trở lên, trên Linux. Hãy truy vấn <code>cudaDevAttrConcurrentManagedAccess</code> thay vì đoán từ GPU.
            </p>
          }
        >
          <p>
            The book's architectures carry compute capability 3.x (Kepler) and 5.x (Maxwell); full Unified Memory needs 6.0
            (Pascal) or later, on Linux. Query <code>cudaDevAttrConcurrentManagedAccess</code> rather than guessing from the
            GPU.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Giờ hãy gỡ lỗi nó. Trong trình chạy từng bước, tích <i>xóa <code>cudaDeviceSynchronize()</code></i> và chạy bản
            chuyển managed trên từng hệ thống. Hãy dự đoán trước: hệ thống nào sập, và hệ thống nào cho kết quả sai mà không
            sập?
          </p>
        }
      >
        <p>
          Now debug it. In the stepper, tick <i>delete <code>cudaDeviceSynchronize()</code></i> and run the managed port in each
          system. Predict first: which systems crash, and which give a wrong answer without crashing?
        </p>
      </Translated>
      <Ask q={questionsSC6['appC/C.6/q-delete-sync']} />
      <Ask q={questionsSC6['appC/C.6/q-oversub-windows']} />

      <h2>
        <Translated vi="Hai thứ cùng được gọi là unified memory">Two things called unified memory</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Nếu bạn chạy mô hình trên Mac, bạn đã nghe "unified memory" nhiều năm nay, và nó mang nghĩa khác. Sự khác biệt
            không nằm ở bức hình hai chiếc hộp; nó nằm ở thứ gì di chuyển, và khi nào. Hãy xem cùng một chương trình trên cả
            hai.
          </p>
        }
      >
        <p>
          If you run models on a Mac you have heard "unified memory" for years, and it means something else. The difference
          is not in a picture of two boxes; it is in what moves, and when. Watch the same program on both.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={<p>Hoạt hình chạy mô hình của trình chạy từng bước và chia sẻ các điều được đơn giản hóa của nó: tính theo trang, không theo byte hay thời gian.</p>}
        >
          <p>The animation runs the stepper's model and shares its simplifications: pages, not bytes or time.</p>
        </Translated>
      </Simplification>
      <TwoUnifiedScene />
      <Translated
        vi={
          <p>
            Thuật ngữ của CUDA chỉ một vùng do phần mềm quản lý trên các bộ nhớ tách biệt: "một tính năng CUDA cho phép bộ
            nhớ managed có thể được tự động di chuyển giữa CPU và các GPU". Thuật ngữ của Apple chỉ phần cứng: "CPU và GPU
            dùng chung bộ nhớ hệ thống". Và đây chỉ là hai trong bốn dạng đang được dùng ngày nay, mỗi dạng có câu trả lời
            riêng cho câu hỏi "cái gì di chuyển, và tôi còn phải lo gì?"
          </p>
        }
      >
        <p>
          CUDA's term names a software-managed pool over separate memories: "a CUDA feature that enables managed memory which
          can be automatically migrated between CPU and GPUs". Apple's names hardware: "the CPU and the GPU share system
          memory". And these are only two of four shapes in use today, each with its own answer to "what moves, and what do I
          still owe?"
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/appC-facts.md, A17, B8; ch05-facts C6)" /* i18n-ignore: date and citation only */>
        <Translated
          vi={
            <p>
              Grace Hopper: CPU và GPU "dùng chung một bảng trang duy nhất cho mỗi tiến trình". Tegra: một DRAM, nhưng mỗi loại
              buffer "có hành vi truy cập và cache khác nhau". MI300A: XNACK "vẫn cần thiết". Apple silicon:{' '}
              <code>MTLStorageModeManaged</code> đã bị deprecated, "lưu trữ managed không có tác dụng trên Apple Silicon".
            </p>
          }
        >
          <p>
            Grace Hopper: CPU and GPU "share a single per-process page table". Tegra: one DRAM, but each buffer type "has
            different accessing and caching behaviors". MI300A: XNACK "is still necessary". Apple silicon:{' '}
            <code>MTLStorageModeManaged</code> is deprecated, "Managed storage has no effect on Apple Silicon".
          </p>
        </Translated>
      </Today>
      <ShapesTable />
      <Translated
        vi={
          <p>
            Một hàng là toàn bộ phụ lục của sách. Ba hàng còn lại là nơi phần lớn sự nhầm lẫn bắt nguồn, và cột cuối giống
            nhau ở mọi nơi: thứ tự là việc của bạn.
          </p>
        }
      >
        <p>
          One row is the whole of the book's appendix. The other three are where most of the confusion comes from, and the
          last column is the same everywhere: ordering is yours.
        </p>
      </Translated>
      <Ask q={questionsSC6['appC/C.6/q-apple']} />

      <h2>
        <Translated vi="Bạn sẽ gặp nó ở đâu">Where you will meet it</Translated>
      </h2>
      <Translated
        vi={
          <p>
            llama.cpp có một công tắc cho việc này. Với <code>GGML_CUDA_ENABLE_UNIFIED_MEMORY=1</code>, backend CUDA của nó cấp
            phát bằng <code>cudaMallocManaged</code> thay vì <code>cudaMalloc</code>, điều mà theo lời chính nó "cho phép
            swap sang RAM hệ thống thay vì sập khi VRAM của GPU cạn", trên Linux. Cùng mã nguồn đó build backend HIP, nơi nó
            còn advise <code>hipMemAdviseSetCoarseGrain</code>, và tài liệu của nó cảnh báo rằng trên HIP công tắc này "làm
            giảm hiệu năng với các GPU không tích hợp". MLX dùng bộ nhớ managed cho các mảng mà host thấy được trên backend
            CUDA của nó và buffer shared trên Metal: cùng một lời hứa, một con trỏ cho cả hai phía, từ hai cơ chế khác nhau.
          </p>
        }
      >
        <p>
          llama.cpp has a switch for this. With <code>GGML_CUDA_ENABLE_UNIFIED_MEMORY=1</code> its CUDA backend allocates with{' '}
          <code>cudaMallocManaged</code> instead of <code>cudaMalloc</code>, which, in its own words, "allows swapping to system
          RAM instead of crashing when the GPU VRAM is exhausted", on Linux. The same source builds the HIP backend, where it
          also advises <code>hipMemAdviseSetCoarseGrain</code>, and its documentation warns that on HIP the switch "hurts
          performance for non-integrated GPUs". MLX uses managed memory for host-visible arrays on its CUDA backend and shared
          buffers on Metal: the same promise, one pointer for both sides, from two different mechanisms.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Đọc từ <code>docs/build.md</code> và <code>ggml-cuda.cu</code> của llama.cpp, và các allocator của MLX, ngày
              2026-11-20 (appC-facts D3, D5; ch02-facts D3).
            </p>
          }
        >
          <p>
            Read from llama.cpp's <code>docs/build.md</code> and <code>ggml-cuda.cu</code>, and MLX's allocators, on
            2026-11-20 (appC-facts D3, D5; ch02-facts D3).
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Trên HIP và Metal">On HIP and Metal</Translated>
      </h2>
      <Translated
        vi={
          <p>
            HIP có cùng các lời gọi, <code>hipMallocManaged</code> và <code>__managed__</code>, đã được hiện thực trên Linux và
            "đang phát triển" trên Windows. Di chuyển theo truy cập cần hai thứ: <code>HSA_XNACK=1</code> trong môi trường và
            một kernel driver có HMM. Thiếu chúng, lời gọi vẫn thành công, nhưng header nói nó "hoạt động giống
            hipMallocHost": kernel đọc bộ nhớ host được ghim qua đường truyền. AMD cũng yêu cầu kích thước theo đơn vị 4 KB
            ("it should be granularity of 4KB"). Metal hoàn toàn không có cấp phát managed: một buffer shared trên một bộ nhớ
            chung không có gì để di chuyển.
          </p>
        }
      >
        <p>
          HIP has the same calls, <code>hipMallocManaged</code> and <code>__managed__</code>, implemented on Linux and "under
          development" on Windows. Migration on access needs two things: <code>HSA_XNACK=1</code> in the environment and a
          kernel driver with HMM. Without them the call still succeeds, but the header says it "behaves the same as
          hipMallocHost": the kernel reads pinned host memory over the link. AMD also asks for sizes at a granularity of 4 KB
          ("it should be granularity of 4KB"). Metal has no managed allocation at all: a shared buffer on one memory has
          nothing to migrate.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ nói về CUDA. HIP lấy từ trang unified memory của AMD và các header ROCm 10.2 (appC-facts B5); Metal lấy
              từ tài liệu về storage mode của Apple (ch05-facts C6).
            </p>
          }
        >
          <p>
            The book covers CUDA only. HIP from AMD's unified memory page and the ROCm 10.2 headers (appC-facts B5); Metal from
            Apple's storage-mode documentation (ch05-facts C6).
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsSC6['appC/C.6/q-hip-xnack']} />
      <Translated
        vi={
          <p>
            Điều làm chế độ đầy đủ khả thi là GPU có thể nhận một lỗi trang và chạy tiếp. Mục{' '}
            <a href={href('appC', 'C.8')}>C.8</a> mổ xẻ một lỗi trang và hỏi khi nào di chuyển trang là câu trả lời sai. Lab
            C.1 trên <a href={href('appC', 'labs')}>trang lab</a> cấp dữ liệu cho một kernel theo ba cách, sao chép, bộ nhớ
            host ánh xạ và bộ nhớ managed, trên chính GPU của bạn.
          </p>
        }
      >
        <p>
          What makes the full mode possible is that a GPU can take a page fault and carry on. Section{' '}
          <a href={href('appC', 'C.8')}>C.8</a> takes one fault apart and asks when moving pages is the wrong answer. Lab C.1 on
          the <a href={href('appC', 'labs')}>labs page</a> feeds one kernel three ways, copies, mapped host memory and managed
          memory, on your own GPU.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Quay lại <a href={href('appC', 'start')}>bản đồ</a>: bước này là C.6, Unified Memory ở dạng hạn chế. Các ô nó
            thay đổi là hàng bộ nhớ managed, và trên Windows, WSL và một số hệ thống Tegra chúng vẫn là hiện trạng ngày nay.
          </p>
        }
      >
        <p>
          Back to <a href={href('appC', 'start')}>the map</a>: this step is C.6, Unified Memory in its limited form. The
          cells it changes are the managed-memory row, and on Windows, WSL and some Tegra systems they are still today's.
        </p>
      </Translated>
    </>
  );
}
