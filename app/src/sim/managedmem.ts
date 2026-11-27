// A page-level model of managed memory for Appendix C, sections C.6 (Unified Memory,
// book pp. 626–627) and C.8 (page fault handling, pp. 628–629), and for the scenes
// "Two things called unified memory" and "One GPU page fault".
//
// What it models: buffers split into pages, and for each page where a valid copy is
// (CPU memory, GPU memory, both as a read-only replica, or the one memory of a
// one-memory system) and which processors hold a mapping to it. A program is a list of
// operations: allocate, CPU read or write, kernel launch with the pages it reads and
// writes, synchronise, explicit copy, free. A policy says what the system does when a
// processor touches a page it does not hold:
//
//   bulk       Limited Unified Memory. At every launch, each managed page whose only copy
//              is in CPU memory moves to the GPU; the CPU must not touch managed memory
//              while a kernel is active (a segmentation fault); no managed allocation
//              beyond GPU memory. Book p. 627 (CUDA 6 on Kepler and Maxwell); today the
//              documented behaviour wherever cudaDevAttrConcurrentManagedAccess is 0:
//              Windows, WSL and some Tegra systems (notes/today/appC-facts.md A10).
//   migrate    Full Unified Memory, the default. A page is placed where it is first
//              touched; a touch by the other processor faults, the owner loses its
//              mapping, the page moves and the toucher gains a mapping (book p. 628; A11).
//              Past the GPU's capacity the least-recently-used page goes back to the CPU.
//   map        cudaMemAdvise SetPreferredLocation(host) plus SetAccessedBy(GPU): the page
//              stays in CPU memory and the GPU reaches it over the link (book p. 628,
//              "mapped ... accessed over the interconnect"; A13).
//   replicate  cudaMemAdvise SetReadMostly: a read by a processor without a copy makes a
//              replica; a write invalidates the other copy (A13: "trading off read
//              bandwidth for write bandwidth").
//   one        One physical memory: an Apple silicon shared buffer (ch05-facts C6).
//              Nothing moves; the program still owes ordering.
//
// What it leaves out (every lesson that shows it says so):
//   - Pages, not bytes, time or page sizes. Counters count pages.
//   - A page's first touch places it where it is touched and is not counted as a fault.
//   - In bulk mode a page goes back to the CPU on the CPU's first touch after the
//     synchronise. NVIDIA's guide says only that managed memory "is migrated back to the
//     CPU when the GPU is synchronized" and that the CPU side can fault (A10).
//   - Fixed policies stand in for the driver's heuristics (A13); least-recently-used
//     eviction stands in for its eviction policy, which our sources do not document.
//   - Hardware-coherent systems (cache-line granularity, access counters; A11, A13) and
//     prefetching are not modelled.
//   - A kernel makes all of its accesses at its launch and stays active until a
//     synchronise, a blocking copy (ch02-facts A4, A5) or a free (which "may perform
//     implicit synchronization", ch02-facts A3; the model always waits there).
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

import { tr } from '../lib/locale.ts';

export type Policy = 'bulk' | 'migrate' | 'map' | 'replicate' | 'one';
export type Proc = 'cpu' | 'gpu';
export type Dir = 'h2d' | 'd2h';
export type BufKind = 'managed' | 'host' | 'device';
/** Where the valid copy of a page is. 'one': the single memory of a one-memory system. */
export type Loc = 'none' | 'cpu' | 'gpu' | 'both' | 'one';

export const POLICY_NAME: Record<Policy, string> = {
  get bulk() { return tr('bulk at launch', 'chuyển hết khi khởi chạy'); },
  get migrate() { return tr('fault and migrate', 'lỗi trang rồi di chuyển'); },
  get map() { return tr('map on access', 'ánh xạ khi truy cập'); },
  replicate: 'read-mostly',
  get one() { return tr('one memory', 'một bộ nhớ'); },
};

export type Op =
  | { op: 'alloc'; buf: string; kind: BufKind; pages: number }
  | { op: 'cpu'; buf: string; write: boolean; pages: number[] }
  | { op: 'launch'; buf: string; reads: number[]; writes: number[] }
  | { op: 'sync' }
  | { op: 'memcpy'; dst: string; src: string }
  | { op: 'free'; buf: string };

export type ErrorKind = 'segfault' | 'race' | 'ordering' | 'oversubscribed';

export type EvKind =
  | 'populate' // first touch places the page
  | 'local' // an access served by the processor's own memory
  | 'fault' // an access to a page the processor has no mapping for
  | 'unmap' // a processor loses its mapping
  | 'move' // a page crosses the link
  | 'map' // a processor gains a mapping
  | 'replay' // the faulting access runs again and completes
  | 'remote' // an access served over the link without moving the page
  | 'invalidate' // a copy is discarded because the other side wrote
  | 'evict' // a page leaves GPU memory to make room
  | 'copy' // an explicit copy inside one physical memory
  | 'wait' // the host waits for the active kernel
  | 'error';

export interface Ev {
  e: EvKind;
  buf: string;
  /** The page, or -1 for an event about the whole operation. */
  page: number;
  /** Who touched, lost or gained a mapping, or where a page was placed. */
  proc?: Proc;
  dir?: Dir;
  why?: 'fault' | 'bulk' | 'evict' | 'copy' | 'replica';
  error?: ErrorKind;
}

export interface Counters {
  /** Pages that crossed the link from CPU memory to GPU memory. */
  h2d: number;
  /** Pages that crossed the link from GPU memory to CPU memory. */
  d2h: number;
  gpuFaults: number;
  cpuFaults: number;
  /** Page accesses served over the link without moving the page (one per page per kernel or CPU op). */
  remote: number;
  invalidations: number;
  evictions: number;
  /** Pages copied from one buffer to another inside one physical memory. */
  copiedInMemory: number;
}

export interface MemError {
  kind: ErrorKind;
  /** Index of the operation that raised it. */
  step: number;
  buf: string;
  pages: number[];
  message: string;
}

export interface PageView {
  loc: Loc;
  cpuMap: boolean;
  gpuMap: boolean;
}
export type View = Record<string, { kind: BufKind; pages: PageView[]; freed: boolean }>;

/** The state after one event: what the scenes draw. */
export interface Frame {
  step: number;
  ev: Ev;
  view: View;
  counters: Counters;
  active: boolean;
}

export interface Step {
  op: Op;
  index: number;
  events: Ev[];
  /** Index into Run.frames of this step's first event (equal to `end` when it has none). */
  start: number;
  end: number;
  errors: MemError[];
  /** The program had already crashed, so this op did not run. */
  skipped: boolean;
  view: View;
  counters: Counters;
  active: boolean;
  crashed: boolean;
}

export interface Run {
  policy: Policy;
  steps: Step[];
  frames: Frame[];
  counters: Counters;
  errors: MemError[];
  crashed: boolean;
  /** The view before any operation. */
  initial: View;
}

export interface SimOptions {
  /** GPU memory, in pages. Default: unlimited. */
  capacity?: number;
}

export const zeroCounters = (): Counters => ({
  h2d: 0, d2h: 0, gpuFaults: 0, cpuFaults: 0, remote: 0, invalidations: 0, evictions: 0, copiedInMemory: 0,
});

/** Pages that crossed the link in either direction. */
export const moved = (c: Counters) => c.h2d + c.d2h;

interface PageState extends PageView {
  lastGpu: number;
}
interface BufState {
  kind: BufKind;
  pages: PageState[];
  freed: boolean;
}

const range = (from: number, to: number) => Array.from({ length: Math.max(0, to - from + 1) }, (_, i) => from + i);
export const pagesFrom = range;

/** "page 3", "pages 0–7", "pages 0, 2 and 5". */
export function pageList(ps: number[]): string {
  const s = [...new Set(ps)].sort((a, b) => a - b);
  if (s.length === 0) return tr('no pages', 'không có trang nào');
  if (s.length === 1) return tr(`page ${s[0]}`, `trang ${s[0]}`);
  if (s[s.length - 1] - s[0] === s.length - 1) return tr(`pages ${s[0]}–${s[s.length - 1]}`, `các trang ${s[0]}–${s[s.length - 1]}`);
  return tr(`pages ${s.slice(0, -1).join(', ')} and ${s[s.length - 1]}`, `các trang ${s.slice(0, -1).join(', ')} và ${s[s.length - 1]}`);
}

/** Runs a program under a policy and records every event. Pure and deterministic. */
export function simulate(policy: Policy, ops: Op[], opts: SimOptions = {}): Run {
  const capacity = opts.capacity ?? Infinity;
  const bufs = new Map<string, BufState>();
  const counters = zeroCounters();
  const frames: Frame[] = [];
  const steps: Step[] = [];
  const errors: MemError[] = [];
  let active = false;
  let crashed = false;
  let clock = 0;
  const inflight = new Map<string, Set<number>>();
  let cur: Ev[] = [];
  let curStep = 0;

  const view = (): View => {
    const v: View = {};
    for (const [name, b] of bufs)
      v[name] = { kind: b.kind, freed: b.freed, pages: b.pages.map((p) => ({ loc: p.loc, cpuMap: p.cpuMap, gpuMap: p.gpuMap })) };
    return v;
  };
  const emit = (ev: Ev) => {
    cur.push(ev);
    frames.push({ step: curStep, ev, view: view(), counters: { ...counters }, active });
  };
  const buf = (name: string) => {
    const b = bufs.get(name);
    if (!b) throw new Error(`no buffer ${name}`);
    return b;
  };
  const onGpu = () => {
    let n = 0;
    for (const b of bufs.values()) for (const p of b.pages) if (p.loc === 'gpu' || p.loc === 'both') n++;
    return n;
  };
  const set = (p: PageState, loc: Loc, cpuMap: boolean, gpuMap: boolean) => {
    p.loc = loc;
    p.cpuMap = cpuMap;
    p.gpuMap = gpuMap;
  };

  // Make room for one more page in GPU memory, evicting the least-recently-used managed page.
  const ensureRoom = (keepBuf: string, keepPage: number) => {
    if (policy === 'one') return;
    while (onGpu() >= capacity) {
      let v: { name: string; i: number; p: PageState } | undefined;
      for (const [name, b] of bufs) {
        if (b.kind !== 'managed') continue;
        for (let i = 0; i < b.pages.length; i++) {
          const p = b.pages[i];
          if ((p.loc === 'gpu' || p.loc === 'both') && !(name === keepBuf && i === keepPage) && (!v || p.lastGpu < v.p.lastGpu))
            v = { name, i, p };
        }
      }
      if (!v) throw new Error('GPU memory is full of pages that cannot be evicted');
      counters.evictions++;
      emit({ e: 'evict', buf: v.name, page: v.i, proc: 'gpu' });
      if (v.p.loc === 'both') {
        set(v.p, 'cpu', true, false); // the CPU still holds a valid copy: drop the replica
        emit({ e: 'unmap', buf: v.name, page: v.i, proc: 'gpu' });
      } else {
        set(v.p, 'gpu', false, false);
        emit({ e: 'unmap', buf: v.name, page: v.i, proc: 'gpu' });
        counters.d2h++;
        set(v.p, 'cpu', false, false);
        emit({ e: 'move', buf: v.name, page: v.i, dir: 'd2h', why: 'evict' }); // i18n-ignore: union member compared by the chapter
        set(v.p, 'cpu', true, false);
      }
    }
  };

  const gpuTouch = (name: string, i: number, write: boolean) => {
    const b = buf(name);
    const p = b.pages[i];
    if (b.kind === 'host') throw new Error('the model does not let a kernel touch a malloc buffer');
    if (policy === 'one' || b.kind === 'device') {
      if (p.loc === 'none') {
        set(p, policy === 'one' ? 'one' : 'gpu', policy === 'one', true);
        emit({ e: 'populate', buf: name, page: i, proc: 'gpu' });
      } else emit({ e: 'local', buf: name, page: i, proc: 'gpu' });
      p.lastGpu = ++clock;
      return;
    }
    if (policy === 'map') {
      if (p.loc === 'none') {
        set(p, 'cpu', true, true);
        emit({ e: 'populate', buf: name, page: i, proc: 'cpu' });
      }
      counters.remote++;
      emit({ e: 'remote', buf: name, page: i, proc: 'gpu' });
      return;
    }
    if (p.loc === 'none') {
      ensureRoom(name, i);
      set(p, 'gpu', false, true);
      p.lastGpu = ++clock;
      emit({ e: 'populate', buf: name, page: i, proc: 'gpu' });
      return;
    }
    if (p.loc === 'gpu') {
      p.lastGpu = ++clock;
      emit({ e: 'local', buf: name, page: i, proc: 'gpu' });
      return;
    }
    if (p.loc === 'both') {
      p.lastGpu = ++clock;
      if (write) {
        counters.invalidations++;
        set(p, 'gpu', false, true);
        emit({ e: 'invalidate', buf: name, page: i, proc: 'cpu' });
      }
      emit({ e: 'local', buf: name, page: i, proc: 'gpu' });
      return;
    }
    // loc === 'cpu': the GPU has no mapping. (Never reached in bulk mode: the launch moved it.)
    counters.gpuFaults++;
    emit({ e: 'fault', buf: name, page: i, proc: 'gpu' });
    const replica = policy === 'replicate' && !write;
    if (!replica) {
      set(p, 'cpu', false, false);
      emit({ e: 'unmap', buf: name, page: i, proc: 'cpu' });
    }
    ensureRoom(name, i);
    counters.h2d++;
    // The page has arrived in GPU memory; the GPU's mapping comes next.
    if (replica) set(p, 'both', true, false);
    else set(p, 'gpu', false, false);
    emit({ e: 'move', buf: name, page: i, dir: 'h2d', why: replica ? 'replica' : 'fault' });
    if (policy === 'replicate' && write) {
      counters.invalidations++;
      emit({ e: 'invalidate', buf: name, page: i, proc: 'cpu' });
    }
    if (replica) set(p, 'both', true, true);
    else set(p, 'gpu', false, true);
    p.lastGpu = ++clock;
    emit({ e: 'map', buf: name, page: i, proc: 'gpu' });
    emit({ e: 'replay', buf: name, page: i, proc: 'gpu' });
  };

  const cpuTouch = (name: string, i: number, write: boolean) => {
    const b = buf(name);
    const p = b.pages[i];
    if (b.kind === 'device') throw new Error('host code may not dereference cudaMalloc memory (ch02-facts A17)');
    if (p.loc === 'none') {
      set(p, policy === 'one' ? 'one' : 'cpu', true, policy === 'one' || (policy === 'map' && b.kind === 'managed'));
      emit({ e: 'populate', buf: name, page: i, proc: 'cpu' });
      return;
    }
    if (p.loc === 'cpu' || p.loc === 'one') {
      emit({ e: 'local', buf: name, page: i, proc: 'cpu' });
      return;
    }
    if (p.loc === 'both') {
      if (write) {
        counters.invalidations++;
        set(p, 'cpu', true, false);
        emit({ e: 'invalidate', buf: name, page: i, proc: 'gpu' });
      }
      emit({ e: 'local', buf: name, page: i, proc: 'cpu' });
      return;
    }
    // loc === 'gpu': the CPU faults and the page comes back.
    counters.cpuFaults++;
    emit({ e: 'fault', buf: name, page: i, proc: 'cpu' });
    const replica = policy === 'replicate' && !write;
    if (!replica) {
      set(p, 'gpu', false, false);
      emit({ e: 'unmap', buf: name, page: i, proc: 'gpu' });
    }
    counters.d2h++;
    // The page has arrived in CPU memory; the CPU's mapping comes next.
    if (replica) set(p, 'both', false, true);
    else set(p, 'cpu', false, false);
    emit({ e: 'move', buf: name, page: i, dir: 'd2h', why: replica ? 'replica' : 'fault' });
    if (policy === 'replicate' && write) {
      counters.invalidations++;
      emit({ e: 'invalidate', buf: name, page: i, proc: 'gpu' });
    }
    if (replica) set(p, 'both', true, true);
    else set(p, 'cpu', true, false);
    emit({ e: 'map', buf: name, page: i, proc: 'cpu' });
    emit({ e: 'replay', buf: name, page: i, proc: 'cpu' });
  };

  const wait = () => {
    if (!active) return;
    active = false;
    inflight.clear();
    emit({ e: 'wait', buf: '', page: -1 });
  };

  const raise = (kind: ErrorKind, name: string, pages: number[], message: string) => {
    const err: MemError = { kind, step: curStep, buf: name, pages, message };
    errors.push(err);
    stepErrors.push(err);
    emit({ e: 'error', buf: name, page: pages.length ? pages[0] : -1, error: kind });
  };
  let stepErrors: MemError[] = [];

  const initial = view();
  ops.forEach((op, index) => {
    curStep = index;
    cur = [];
    stepErrors = [];
    const start = frames.length;
    if (crashed) {
      steps.push({ op, index, events: [], start, end: start, errors: [], skipped: true, view: view(), counters: { ...counters }, active, crashed });
      return;
    }
    switch (op.op) {
      case 'alloc': {
        if (policy === 'bulk' && op.kind === 'managed') {
          let managed = op.pages;
          for (const b of bufs.values()) if (b.kind === 'managed' && !b.freed) managed += b.pages.length;
          if (managed > capacity) {
            raise('oversubscribed', op.buf, [], tr(
              `cudaMallocManaged fails: ${managed} managed pages do not fit in ${capacity} pages of GPU memory, and limited systems cannot allocate more managed memory than the GPU's physical memory`,
              `cudaMallocManaged thất bại: ${managed} trang managed không vừa ${capacity} trang bộ nhớ GPU, và hệ thống hạn chế không thể cấp phát bộ nhớ managed nhiều hơn bộ nhớ vật lý của GPU`,
            ));
            crashed = true;
            break;
          }
        }
        const pages: PageState[] = Array.from({ length: op.pages }, () => ({ loc: 'none' as Loc, cpuMap: false, gpuMap: false, lastGpu: 0 }));
        bufs.set(op.buf, { kind: op.kind, pages, freed: false });
        if (op.kind === 'device')
          pages.forEach((p, i) => {
            if (policy !== 'one') ensureRoom(op.buf, i);
            set(p, policy === 'one' ? 'one' : 'gpu', false, true);
            emit({ e: 'populate', buf: op.buf, page: i, proc: 'gpu' });
          });
        break;
      }
      case 'cpu': {
        const b = buf(op.buf);
        if (b.kind === 'managed' && active) {
          if (policy === 'bulk') {
            raise('segfault', op.buf, op.pages, tr('segmentation fault: the CPU touched managed memory while a kernel was active, which limited systems do not allow', 'segmentation fault: CPU chạm vào bộ nhớ managed khi một kernel đang hoạt động, điều mà hệ thống hạn chế không cho phép'));
            crashed = true;
            break;
          }
          const mine = inflight.get(op.buf);
          const hit = op.pages.filter((i) => mine?.has(i));
          if (hit.length)
            raise(policy === 'one' ? 'ordering' : 'race', op.buf, hit,
              policy === 'one'
                ? tr(`the CPU uses ${pageList(hit)} before waiting for the command buffer that uses ${hit.length === 1 ? 'it' : 'them'}`, `CPU dùng ${pageList(hit)} trước khi chờ command buffer đang dùng ${hit.length === 1 ? 'trang đó' : 'các trang đó'}`)
                : tr(`the CPU touches ${pageList(hit)} while the kernel that uses ${hit.length === 1 ? 'it' : 'them'} may still be running: nothing orders the two`, `CPU chạm vào ${pageList(hit)} trong khi kernel dùng ${hit.length === 1 ? 'trang đó' : 'các trang đó'} có thể vẫn đang chạy: không gì xếp thứ tự giữa hai bên`));
        }
        for (const i of op.pages) cpuTouch(op.buf, i, op.write);
        break;
      }
      case 'launch': {
        if (policy === 'bulk')
          for (const [name, b] of bufs) {
            if (b.kind !== 'managed') continue;
            b.pages.forEach((p, i) => {
              if (p.loc !== 'cpu') return;
              counters.h2d++;
              set(p, 'gpu', false, true);
              p.lastGpu = ++clock;
              emit({ e: 'move', buf: name, page: i, dir: 'h2d', why: 'bulk' }); // i18n-ignore: union member compared by the chapter
            });
          }
        const writes = new Set(op.writes);
        const touched = [...new Set([...op.reads, ...op.writes])].sort((a, b) => a - b);
        for (const i of touched) gpuTouch(op.buf, i, writes.has(i));
        active = true;
        const mine = inflight.get(op.buf) ?? new Set<number>();
        touched.forEach((i) => mine.add(i));
        inflight.set(op.buf, mine);
        break;
      }
      case 'sync':
        wait();
        break;
      case 'memcpy': {
        wait(); // a blocking copy returns only when done, after the kernel before it
        const s = buf(op.src);
        const d = buf(op.dst);
        const dir: Dir = s.kind === 'device' ? 'd2h' : 'h2d';
        d.pages.forEach((p, i) => {
          if (policy === 'one') {
            counters.copiedInMemory++;
            set(p, 'one', d.kind !== 'device', true);
            emit({ e: 'copy', buf: op.dst, page: i });
          } else {
            counters[dir]++;
            if (dir === 'h2d') set(p, 'gpu', false, true);
            else set(p, 'cpu', true, false);
            emit({ e: 'move', buf: op.dst, page: i, dir, why: 'copy' }); // i18n-ignore: union member compared by the chapter
          }
        });
        break;
      }
      case 'free': {
        wait();
        const b = buf(op.buf);
        b.freed = true;
        b.pages.forEach((p) => set(p, 'none', false, false));
        break;
      }
    }
    steps.push({ op, index, events: cur, start, end: frames.length, errors: stepErrors, skipped: false, view: view(), counters: { ...counters }, active, crashed });
  });

  return { policy, steps, frames, counters: { ...counters }, errors, crashed, initial };
}

// ---------------------------------------------------------------- Fig. C.1, three ways
// The CPU original (left of Fig. C.1), our explicit-copy port, and the managed port (right
// of Fig. C.1). Each line's text is the line of labs/appC/snippets/c6_sortfile_*.cu,
// character for character (SC6.snippets.test.ts checks it).

export type Program = 'cpu' | 'explicit' | 'managed';
export const PROGRAM_NAME: Record<Program, string> = {
  get cpu() { return tr('CPU original', 'bản CPU gốc'); },
  get explicit() { return tr('explicit copies', 'sao chép tường minh'); },
  get managed() { return tr('managed (Fig. C.1)', 'bản managed (Fig. C.1)'); },
};
export const PROGRAM_FILE: Record<Program, string> = {
  cpu: 'c6_sortfile_cpu.cu',
  explicit: 'c6_sortfile_explicit.cu',
  managed: 'c6_sortfile_managed.cu',
};
export const FIG_HEAD = 'void sortfile(FILE *fp, int N) {'; // i18n-ignore: CUDA source shown as code
export const FIG_PAGES = 8;

export type LineRole = 'decl' | 'alloc' | 'read-file' | 'copy-in' | 'sort' | 'sync' | 'copy-out' | 'use' | 'free' | 'blank';
export interface FigLine {
  text: string;
  role: LineRole;
  /** At most one operation per line. */
  op: Op | null;
  /** Differs from the CPU original (Fig. C.1 prints these in red). */
  changed: boolean;
  /** Removed by the "delete cudaDeviceSynchronize()" switch: shown struck out, never run. */
  deleted: boolean;
}
export interface FigOptions {
  /** The kernel touches only the first half of the buffer. */
  half?: boolean;
  /** cudaDeviceSynchronize() deleted from the managed port. */
  noSync?: boolean;
  pages?: number;
}

export function figLines(program: Program, o: FigOptions = {}): FigLine[] {
  const n = o.pages ?? FIG_PAGES;
  const all = range(0, n - 1);
  const touched = o.half ? range(0, Math.ceil(n / 2) - 1) : all;
  const L = (text: string, role: LineRole, op: Op | null, changed = false, deleted = false): FigLine => ({ text, role, op, changed, deleted });
  if (program === 'cpu')
    return [
      L('    char *data;', 'decl', null), // i18n-ignore: CUDA source shown as code
      L('    data = (char *)malloc(N);', 'alloc', { op: 'alloc', buf: 'data', kind: 'host', pages: n }),
      L('    fread(data, 1, N, fp);', 'read-file', { op: 'cpu', buf: 'data', write: true, pages: all }), // i18n-ignore: CUDA source shown as code
      L('    qsort_char(data, N, 1);', 'sort', { op: 'cpu', buf: 'data', write: true, pages: touched }), // i18n-ignore: CUDA source shown as code
      L('', 'blank', null),
      L('    use_data(data);', 'use', { op: 'cpu', buf: 'data', write: false, pages: all }),
      L('    free(data);', 'free', { op: 'free', buf: 'data' }),
    ];
  if (program === 'explicit')
    return [
      L('    char *data, *d_data;', 'decl', null, true), // i18n-ignore: CUDA source shown as code
      L('    data = (char *)malloc(N);', 'alloc', { op: 'alloc', buf: 'data', kind: 'host', pages: n }),
      L('    cudaMalloc(&d_data, N);', 'alloc', { op: 'alloc', buf: 'd_data', kind: 'device', pages: n }, true), // i18n-ignore: CUDA source shown as code
      L('    fread(data, 1, N, fp);', 'read-file', { op: 'cpu', buf: 'data', write: true, pages: all }), // i18n-ignore: CUDA source shown as code
      L('    cudaMemcpy(d_data, data, N, cudaMemcpyHostToDevice);', 'copy-in', { op: 'memcpy', dst: 'd_data', src: 'data' }, true), // i18n-ignore: CUDA source shown as code
      L('    qsort_char<<<grid, block>>>(d_data, N, 1);', 'sort', { op: 'launch', buf: 'd_data', reads: touched, writes: touched }, true),
      L('    cudaMemcpy(data, d_data, N, cudaMemcpyDeviceToHost);', 'copy-out', { op: 'memcpy', dst: 'data', src: 'd_data' }, true), // i18n-ignore: CUDA source shown as code
      L('    use_data(data);', 'use', { op: 'cpu', buf: 'data', write: false, pages: all }),
      L('    cudaFree(d_data);', 'free', { op: 'free', buf: 'd_data' }, true),
      L('    free(data);', 'free', { op: 'free', buf: 'data' }),
    ];
  return [
    L('    char *data;', 'decl', null), // i18n-ignore: CUDA source shown as code
    L('    cudaMallocManaged(&data, N);', 'alloc', { op: 'alloc', buf: 'data', kind: 'managed', pages: n }, true), // i18n-ignore: CUDA source shown as code
    L('    fread(data, 1, N, fp);', 'read-file', { op: 'cpu', buf: 'data', write: true, pages: all }), // i18n-ignore: CUDA source shown as code
    L('    qsort_char<<<grid, block>>>(data, N, 1);', 'sort', { op: 'launch', buf: 'data', reads: touched, writes: touched }, true),
    L('    cudaDeviceSynchronize();', 'sync', o.noSync ? null : { op: 'sync' }, true, !!o.noSync),
    L('    use_data(data);', 'use', { op: 'cpu', buf: 'data', write: false, pages: all }),
    L('    cudaFree(data);', 'free', { op: 'free', buf: 'data' }, true),
  ];
}

/** Lines a stepper visits: every line except blank and deleted ones. */
export const steppable = (lines: FigLine[]) => lines.map((l, i) => i).filter((i) => lines[i].role !== 'blank' && !lines[i].deleted);

export interface FigRun {
  lines: FigLine[];
  run: Run;
  /** For each line, the index of its step in run.steps, or -1. */
  stepOf: number[];
}

/**
 * Fig. C.1 as a program under a mode. The CPU original and the explicit port use no
 * managed memory, so only the 'one' mode changes them (their buffers then share one memory).
 */
export function runFig(program: Program, mode: Policy, o: FigOptions = {}): FigRun {
  const lines = figLines(program, o);
  const ops: Op[] = [];
  const stepOf = lines.map((l) => (l.op ? ops.push(l.op) - 1 : -1));
  return { lines, run: simulate(mode, ops), stepOf };
}

// ---------------------------------------------------------------- C.8 explorer: scripts
export type ScriptKind = 'cpu-write' | 'cpu-read' | 'kernel-read' | 'kernel-write' | 'sync';
export interface ScriptOp {
  kind: ScriptKind;
  /** First and last page, inclusive. Ignored by 'sync'. */
  from: number;
  to: number;
}
export const EXPLORE_PAGES = 16;
export const EXPLORE_POLICIES: Policy[] = ['bulk', 'migrate', 'map', 'replicate'];

export function scriptLabel(s: ScriptOp): string {
  const ps = s.from === s.to ? tr(`page ${s.from}`, `trang ${s.from}`) : tr(`pages ${s.from}–${s.to}`, `các trang ${s.from}–${s.to}`);
  switch (s.kind) {
    case 'cpu-write': return tr(`CPU writes ${ps}`, `CPU ghi ${ps}`);
    case 'cpu-read': return tr(`CPU reads ${ps}`, `CPU đọc ${ps}`);
    case 'kernel-read': return tr(`kernel reads ${ps}`, `kernel đọc ${ps}`);
    case 'kernel-write': return tr(`kernel writes ${ps}`, `kernel ghi ${ps}`);
    case 'sync': return tr('synchronise', 'đồng bộ hóa');
  }
}

export function scriptOps(script: ScriptOp[], pages = EXPLORE_PAGES): Op[] {
  const ops: Op[] = [{ op: 'alloc', buf: 'm', kind: 'managed', pages }];
  for (const s of script) {
    const ps = range(Math.max(0, Math.min(s.from, s.to)), Math.min(pages - 1, Math.max(s.from, s.to)));
    if (s.kind === 'sync') ops.push({ op: 'sync' });
    else if (s.kind === 'cpu-write' || s.kind === 'cpu-read') ops.push({ op: 'cpu', buf: 'm', write: s.kind === 'cpu-write', pages: ps });
    else ops.push({ op: 'launch', buf: 'm', reads: s.kind === 'kernel-read' ? ps : [], writes: s.kind === 'kernel-write' ? ps : [] });
  }
  return ops;
}

/** Runs an explorer script; step k of the run is script line k − 1 (step 0 is the allocation). */
export const explore = (policy: Policy, script: ScriptOp[], pages = EXPLORE_PAGES) => simulate(policy, scriptOps(script, pages));

const S = (kind: ScriptKind, from = 0, to = from): ScriptOp => ({ kind, from, to });
const repeat = (n: number, body: ScriptOp[]) => Array.from({ length: n }, () => body).flat();

export type PresetId = 'figC1' | 'one-page' | 'ping-pong' | 'decode';
export const DECODE_TOKENS = 6;
export const PRESETS: { id: PresetId; label: string; script: ScriptOp[] }[] = [
  { id: 'figC1', label: 'Fig. C.1', script: [S('cpu-write', 0, 15), S('kernel-write', 0, 15), S('sync'), S('cpu-read', 0, 15)] },
  { id: 'one-page', get label() { return tr('kernel touches 1 page of 16', 'kernel chạm 1 trang trong 16'); }, script: [S('cpu-write', 0, 15), S('kernel-read', 5), S('sync')] },
  {
    id: 'ping-pong',
    get label() { return tr('CPU and GPU take turns on one counter page', 'CPU và GPU lần lượt dùng một trang bộ đếm'); },
    script: [S('cpu-write', 0, 15), ...repeat(4, [S('kernel-write', 0), S('sync'), S('cpu-write', 0)])],
  },
  {
    id: 'decode',
    get label() { return tr('decode loop re-reading all pages', 'vòng lặp decode đọc lại mọi trang'); },
    // Pages 0–13 are weights, page 15 the logits; the CPU samples from the logits each token.
    script: [S('cpu-write', 0, 15), ...repeat(DECODE_TOKENS, [S('kernel-read', 0, 13), S('kernel-write', 15), S('sync'), S('cpu-read', 15)])],
  },
];

/** The reader's cost model: a price per page that crosses the link and per remote page access. */
export const cost = (c: Counters, perMove: number, perRemote: number) => moved(c) * perMove + c.remote * perRemote;

// ---------------------------------------------------------------- C.8 explorer: a model that does not fit
export type SweepPolicy = 'lru' | 'prefix';
export interface TokenTraffic {
  /** Pages migrated into GPU memory during this token. */
  h2d: number;
  /** Pages evicted back to CPU memory during this token. */
  d2h: number;
  /** Pages read over the link without moving. */
  remote: number;
  /** GPU page faults. */
  faults: number;
}

/**
 * Decoding with weights of `weights` pages in a GPU memory of `capacity` pages: every token
 * reads every weight page in the same order. All weights start in CPU memory.
 *   'lru'    fault and migrate, evicting the least-recently-used page when GPU memory is full.
 *   'prefix' keep the first `capacity` pages resident (moved once, on the first token) and
 *            map the rest from host memory, read over the link every token.
 * Least-recently-used eviction is our stand-in: the driver's policy is not documented in
 * our sources, so the cliff this produces is a property of the model, not of any driver.
 */
export function decodeSweep(weights: number, capacity: number, tokens: number, policy: SweepPolicy): TokenTraffic[] {
  const out: TokenTraffic[] = [];
  if (policy === 'prefix') {
    const kept = Math.min(weights, capacity);
    for (let t = 0; t < tokens; t++)
      out.push({ h2d: t === 0 ? kept : 0, d2h: 0, remote: weights - kept, faults: t === 0 ? kept : 0 });
    return out;
  }
  const lru = new Map<number, true>(); // insertion order = least recently used first
  for (let t = 0; t < tokens; t++) {
    const tr: TokenTraffic = { h2d: 0, d2h: 0, remote: 0, faults: 0 };
    for (let p = 0; p < weights; p++) {
      if (lru.has(p)) {
        lru.delete(p);
        lru.set(p, true);
        continue;
      }
      tr.faults++;
      if (lru.size >= capacity) {
        const victim = lru.keys().next().value as number;
        lru.delete(victim);
        tr.d2h++;
      }
      lru.set(p, true);
      tr.h2d++;
    }
    out.push(tr);
  }
  return out;
}

/** Traffic per token once warm (the last of a few tokens). */
export const warmTraffic = (weights: number, capacity: number, policy: SweepPolicy) => {
  const tr = decodeSweep(weights, capacity, 3, policy);
  return tr[tr.length - 1];
};

// ---------------------------------------------------------------- the page-fault scene
/**
 * One page through a GPU touch and a later CPU touch, under fault and migrate: four
 * managed pages written by the CPU; a kernel reads page 2 (a fault), a second kernel reads
 * it again (local), the host synchronises, and the CPU writes page 2 (the reverse fault).
 */
export const FAULT_PAGE = 2;
export const faultStoryOps = (): Op[] => [
  { op: 'alloc', buf: 'm', kind: 'managed', pages: 4 },
  { op: 'cpu', buf: 'm', write: true, pages: [0, 1, 2, 3] },
  { op: 'launch', buf: 'm', reads: [FAULT_PAGE], writes: [] },
  { op: 'launch', buf: 'm', reads: [FAULT_PAGE], writes: [] },
  { op: 'sync' },
  { op: 'cpu', buf: 'm', write: true, pages: [FAULT_PAGE] },
];
export const faultStory = () => simulate('migrate', faultStoryOps());
