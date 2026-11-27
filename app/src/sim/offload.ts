// The host stub of chapter 2 as a small interpreted program.
//
// One model serves section 2.3 (the narrated scene and the boundary-traffic
// explorer) and section 2.4 (the call-by-call stepper and the bug hunt).
//
//   program(backend, iface)   the ordered calls of the stub for CUDA, HIP, or Metal
//   run(program, upTo, env)   the state of both memories after `upTo` calls
//   mutate(program, defect)   the same stub with exactly one defect
//   outcome(defect, env)      what that defect produces, classified
//   lanes(program)            who waits for whom: host and device, call by call
//   boundaryTraffic(...)      transfers and bytes for a chain of elementwise ops
//   phaseTimes / chainTimes   reading the reader's own lab 2.2 and 2.3 benchmark rows
//
// What the model is faithful about: which bytes are where after each call, how
// many of them crossed between host memory and device memory, which calls make
// the host thread wait, and what the vendors' documentation says about each
// defect (notes/today/ch02-facts.md; the item is named beside every rule).
//
// What it leaves out: time (nothing here has a duration), pinned and managed
// memory, caching allocators, and what a real machine does after undefined
// behaviour. Where the documentation promises nothing the model says
// 'undetermined' and does not guess.

import { tr } from '../lib/locale.ts';

export const FLOAT_BYTES = 4; // float is 4 bytes on all three backends (facts A20)
export const N = 8;
/** What the test harness writes into C_h before the call, so an element nobody wrote is recognisable. */
export const SENTINEL = -1;

const A_VALUES = [3, 1, 4, 1, 5, 9, 2, 6];
const B_VALUES = [2, 7, 1, 8, 2, 8, 1, 8];

/** Small integers: no sum is 0, none is the sentinel, and no element of B is 0. */
export function inputs(n: number = N): { A: number[]; B: number[] } {
  return {
    A: Array.from({ length: n }, (_, i) => A_VALUES[i % A_VALUES.length]),
    B: Array.from({ length: n }, (_, i) => B_VALUES[i % B_VALUES.length]),
  };
}
export function expected(n: number = N): number[] {
  const { A, B } = inputs(n);
  return A.map((a, i) => a + B[i]);
}

/**
 * The course emulator fills fresh "device" memory with 0xA5 bytes
 * (labs/common/include/pmpp/emu/emu_host.h). Read as a float that is about
 * -2.9e-16: small enough to vanish when added to any of the inputs above.
 */
export const EMU_FILL_BYTE = 0xa5;
export const EMU_FILL = new Float32Array(new Uint8Array(4).fill(EMU_FILL_BYTE).buffer)[0];

export type Backend = 'cuda' | 'hip' | 'metal';
export type Iface = 'buffers' | 'host-arrays';
/** Where a program runs: a device as its documentation describes it, or the course emulator. */
export type Env = Backend | 'emu';
export type ArrayId = 'A' | 'B' | 'C';

/**
 * A float, or one of two things the model will not put a number on:
 * 'uncleared'     memory that was allocated and never written (CUDA: "The memory is not cleared", A2)
 * 'undetermined'  anything read or written after the documentation stopped promising a result
 */
export type Cell = number | 'uncleared' | 'undetermined';

export function showCell(c: Cell): string {
  if (c === 'uncleared') return '?';
  if (c === 'undetermined') return '~';
  if (c === EMU_FILL) return 'A5';
  return Number.isInteger(c) ? String(c) : c.toPrecision(2);
}

export interface Ref {
  side: 'host' | 'device';
  array: ArrayId;
}

export type Op =
  | { op: 'alloc'; array: ArrayId; bytes: number } // cudaMalloc, hipMalloc, newBufferWithLength
  | { op: 'alloc-copy'; array: ArrayId; bytes: number } // newBufferWithBytes: allocate, then copy the host array in
  | { op: 'fill'; array: ArrayId } // the data is produced straight into a shared buffer
  | { op: 'copy'; dst: Ref; src: Ref; bytes: number; kind: 'h2d' | 'd2h' }
  | { op: 'encode' } // Metal: record the commands; nothing runs yet
  | { op: 'launch' } // CUDA and HIP: the launch statement. Metal: commit.
  | { op: 'wait' } // waitUntilCompleted
  | { op: 'read' } // Metal: read the result through contents
  | { op: 'free'; array: ArrayId }
  | { op: 'host-deref'; array: ArrayId }; // a defect: host code reads through a device pointer

/** The step groups a benchmark can time (lab 2.2 reports one number per group). */
export type Group = 'allocate' | 'copy in' | 'launch' | 'wait' | 'copy back' | 'free' | 'fill' | 'read';

export interface Call {
  /** Stable name, e.g. 'alloc-A', 'copy-in-B', 'launch', 'copy-back', 'free-C'. */
  id: string;
  /** The book's Part 1, 2 or 3 (Fig. 2.5). */
  part: 1 | 2 | 3;
  group: Group;
  /** Source lines, as in labs/ch02/snippets (a unit test ties them to the files). */
  code: string[];
  /** The runtime or framework calls these lines make, for the tally. */
  api: string[];
  op: Op;
  /** A documented remark specific to this backend. */
  note?: string;
  /** True on the one call a defect changed or added. */
  changed?: boolean;
}

export interface Program {
  backend: Backend;
  iface: Iface;
  /** 'separate': host memory and device memory. 'shared': one memory, as on Apple silicon. */
  memory: 'separate' | 'shared';
  n: number;
  /** Lines shown above the first call (signature, declarations). */
  head: string[];
  calls: Call[];
  tail: string[];
  defect?: Defect;
}

// ---------------------------------------------------------------- programs
const METAL_OPTS = 'options:MTLResourceStorageModeShared';

/** The eight lines that are one launch statement in CUDA (facts C4). */
export const METAL_ENCODE = [
  'id<MTLCommandBuffer> cb = [queue commandBuffer];',
  'id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];',
  '[enc setComputePipelineState:pipeline];', // i18n-ignore: Objective-C source line
  '[enc setBuffer:A offset:0 atIndex:0];', // i18n-ignore: Objective-C source line
  '[enc setBuffer:B offset:0 atIndex:1];', // i18n-ignore: Objective-C source line
  '[enc setBuffer:C offset:0 atIndex:2];', // i18n-ignore: Objective-C source line
  '[enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];', // i18n-ignore: Objective-C source line
  '[enc endEncoding];', // i18n-ignore: Objective-C source line
];
const METAL_ENCODE_API = [
  'commandBuffer', 'computeCommandEncoder', 'setComputePipelineState', 'setBuffer', 'setBuffer', 'setBuffer',
  'dispatchThreads', 'endEncoding',
];

function family(backend: 'cuda' | 'hip', n: number): Program {
  const p = backend; // the two runtimes differ only in this prefix (facts B1)
  const size = n * FLOAT_BYTES;
  const alloc = (a: ArrayId): Call => ({
    id: 'alloc-' + a, part: 1, group: 'allocate',
    code: [`${p}Malloc((void**)&${a}_d, size);`],
    api: [`${p}Malloc`], // i18n-ignore: API name
    op: { op: 'alloc', array: a, bytes: size },
  });
  const copyIn = (a: ArrayId): Call => ({
    id: 'copy-in-' + a, part: 1, group: 'copy in', // i18n-ignore: Group identifier
    code: [`${p}Memcpy(${a}_d, ${a}_h, size, ${p}MemcpyHostToDevice);`],
    api: [`${p}Memcpy`], // i18n-ignore: API name
    op: { op: 'copy', dst: { side: 'device', array: a }, src: { side: 'host', array: a }, bytes: size, kind: 'h2d' },
  });
  const free = (a: ArrayId): Call => ({
    id: 'free-' + a, part: 3, group: 'free',
    code: [`${p}Free(${a}_d);`],
    api: [`${p}Free`], // i18n-ignore: API name
    op: { op: 'free', array: a },
    note: backend === 'hip'
      ? tr('hipFree performs an implicit hipDeviceSynchronize() first (facts B1).', 'hipFree thực hiện một lần hipDeviceSynchronize() ngầm trước (facts B1).')
      : tr('cudaFree may perform implicit synchronization (facts A3).', 'cudaFree có thể thực hiện đồng bộ hóa ngầm (facts A3).'),
  });
  return {
    backend, iface: 'host-arrays', memory: 'separate', n,
    head: [
      'void vecAdd(const float* A_h, const float* B_h, float* C_h, int n) {', // i18n-ignore: C source line
      '    int size = n * sizeof(float);',
      '    float *A_d, *B_d, *C_d;', // i18n-ignore: C source line
    ],
    calls: [
      alloc('A'), alloc('B'), alloc('C'),
      copyIn('A'), copyIn('B'),
      {
        id: 'launch', part: 2, group: 'launch',
        code: ['vecAddKernel<<<(n + 255) / 256, 256>>>(A_d, B_d, C_d, n);'],
        api: [],
        op: { op: 'launch' },
      },
      {
        id: 'copy-back', part: 3, group: 'copy back', // i18n-ignore: Group identifier
        code: [`${p}Memcpy(C_h, C_d, size, ${p}MemcpyDeviceToHost);`],
        api: [`${p}Memcpy`], // i18n-ignore: API name
        op: { op: 'copy', dst: { side: 'host', array: 'C' }, src: { side: 'device', array: 'C' }, bytes: size, kind: 'd2h' },
      },
      free('A'), free('B'), free('C'),
    ],
    tail: ['}'],
  };
}

function metal(iface: Iface, n: number): Program {
  const size = n * FLOAT_BYTES;
  const newBuffer = (a: ArrayId): Call => ({
    id: 'alloc-' + a, part: 1, group: 'allocate',
    code: [`id<MTLBuffer> ${a} = [device newBufferWithLength:size ${METAL_OPTS}];`],
    api: ['newBufferWithLength'],
    op: { op: 'alloc', array: a, bytes: size },
  });
  const encode: Call = { id: 'encode', part: 2, group: 'launch', code: METAL_ENCODE, api: METAL_ENCODE_API, op: { op: 'encode' } };
  const commit: Call = { id: 'launch', part: 2, group: 'launch', code: ['[cb commit];'], api: ['commit'], op: { op: 'launch' } };
  const wait: Call = { id: 'wait', part: 3, group: 'wait', code: ['[cb waitUntilCompleted];'], api: ['waitUntilCompleted'], op: { op: 'wait' } };
  if (iface === 'buffers') {
    const fill = (a: ArrayId): Call => ({
      id: 'fill-' + a, part: 1, group: 'fill',
      code: [`produce${a}((float*)${a}.contents, n);`],
      api: ['contents'],
      op: { op: 'fill', array: a },
    });
    return {
      backend: 'metal', iface, memory: 'shared', n,
      head: [
        'void vecAddInBuffers(id<MTLDevice> device, id<MTLCommandQueue> queue,',
        '                     id<MTLComputePipelineState> pipeline, int n, NSUInteger group) {',
        '    NSUInteger size = n * sizeof(float);',
      ],
      calls: [
        newBuffer('A'), newBuffer('B'), newBuffer('C'),
        fill('A'), fill('B'),
        encode, commit, wait,
        { id: 'read', part: 3, group: 'read', code: ['consume((const float*)C.contents, n);'], api: ['contents'], op: { op: 'read' } },
      ],
      tail: ['}'],
    };
  }
  const wrap = (a: ArrayId): Call => ({
    id: 'copy-in-' + a, part: 1, group: 'copy in', // i18n-ignore: Group identifier
    code: [`id<MTLBuffer> ${a} = [device newBufferWithBytes:${a}_h length:size ${METAL_OPTS}];`],
    api: ['newBufferWithBytes'],
    op: { op: 'alloc-copy', array: a, bytes: size },
  });
  return {
    backend: 'metal', iface, memory: 'shared', n,
    head: [
      'void vecAdd(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,',
      '            NSUInteger group, const float* A_h, const float* B_h, float* C_h, int n) {', // i18n-ignore: Objective-C source line
      '    NSUInteger size = n * sizeof(float);',
    ],
    calls: [
      wrap('A'), wrap('B'), newBuffer('C'),
      encode, commit, wait,
      {
        id: 'copy-back', part: 3, group: 'copy back', // i18n-ignore: Group identifier
        code: ['memcpy(C_h, C.contents, size);'],
        api: ['contents'],
        op: { op: 'copy', dst: { side: 'host', array: 'C' }, src: { side: 'device', array: 'C' }, bytes: size, kind: 'd2h' },
      },
    ],
    tail: ['}'],
  };
}

/**
 * The host stub for a backend. `iface` matters on Metal only: 'buffers' means the
 * caller's data is created in shared buffers, 'host-arrays' keeps the book's
 * interface (ordinary host arrays in and out).
 */
export function program(backend: Backend, iface: Iface = 'buffers', n: number = N): Program {
  return backend === 'metal' ? metal(iface, n) : family(backend, n);
}

/** Every source line of a program in order, with the 1-based line range of each call. */
export function listing(p: Program): { lines: string[]; range: [number, number][] } {
  const lines = [...p.head];
  const range: [number, number][] = [];
  for (const c of p.calls) {
    const first = lines.length + 1;
    for (const l of c.code) lines.push('    ' + l);
    range.push([first, lines.length]);
  }
  lines.push(...p.tail);
  return { lines, range };
}

// ---------------------------------------------------------------- running
export interface DevArray {
  cells: Cell[];
  bytes: number;
  freed: boolean;
  /** A kernel that writes this array has been issued and nothing has waited for it yet. */
  inFlight: boolean;
}

export type FindingKind =
  | 'out-of-bounds' // a copy or the kernel went past an allocation
  | 'use-after-free'
  | 'undefined-per-docs' // pointers that do not match the copy kind (facts A4, B1)
  | 'not-allowed' // host code read through a CUDA device pointer (facts A17)
  | 'slow-zero-copy'; // the same on HIP, as AMD documents it (facts B7)

export interface Finding {
  /** 1-based index of the call that did it. */
  step: number;
  kind: FindingKind;
  detail: string;
}

export interface PointerVar {
  name: string;
  refersTo: 'nothing yet' | 'host memory' | 'device memory' | 'freed device memory' | 'the one shared memory';
  /** May host code read and write through it? */
  hostMayUse: boolean;
}

export interface HostRead {
  /**
   * impossible  the host has no way to see the result here
   * stale       the host can read something, and it is not the finished result
   * valid       the host reads what the kernel produced
   */
  status: 'impossible' | 'stale' | 'valid';
  values: Cell[] | null;
  why: string;
}

export interface State {
  /** How many calls have been executed. */
  step: number;
  /** The caller's arrays in host memory; null when the data lives in buffers. */
  host: Record<ArrayId, Cell[]> | null;
  device: Record<ArrayId, DevArray | null>;
  pointers: PointerVar[];
  /** What the host thread was doing during the last call. */
  hostThread: 'running' | 'blocked';
  /** Device work that finished while the host was inside the last call. */
  ran: string[];
  /** Device work issued and not yet waited for. */
  queue: string[];
  /** separate memory: bytes that crossed the boundary. shared memory: bytes copied between allocations. */
  bytesMoved: number;
  /** Array-sized copies made. */
  copies: number;
  frees: number;
  apiCalls: number;
  kernel: 'not issued' | 'issued' | 'done';
  /** Metal with the data in buffers: what the caller's read through contents obtained, once it has read. */
  consumed: Cell[] | null;
  hostRead: HostRead;
  /** The first thing that went outside what the documentation defines. */
  finding: Finding | null;
  /** Emulator only: AddressSanitizer ended the run at this call. */
  stoppedAt: number | null;
  /** One sentence on what the last call did. */
  says: string;
}

const KERNEL = { cuda: 'vecAddKernel', hip: 'vecAddKernel', metal: 'add_arrays' } as const;

function add(a: Cell, b: Cell): Cell {
  if (a === 'undetermined' || b === 'undetermined') return 'undetermined';
  if (a === 'uncleared' || b === 'uncleared') return 'uncleared';
  return Math.fround(a + b);
}

interface Work extends State {
  tainted: boolean;
  /** How the caller's C_h was last filled by a copy back. */
  copiedBack: 'no' | 'before the kernel finished' | 'in part' | 'yes';
}

function fresh(p: Program, env: Env): Cell {
  if (p.backend === 'metal') return 0; // newBufferWithLength "clears with zero values" (facts C2)
  // CUDA: "The memory is not cleared." (facts A2). HIP: the header does not say; assume the same (facts B1).
  return env === 'emu' ? EMU_FILL : 'uncleared';
}

function note(w: Work, step: number, kind: FindingKind, detail: string, env: Env) {
  if (!w.finding) w.finding = { step, kind, detail };
  if (kind === 'out-of-bounds' || kind === 'use-after-free') {
    if (env === 'emu') w.stoppedAt = w.stoppedAt ?? step; // AddressSanitizer aborts the process
    else w.tainted = true;
  }
}

function runKernel(w: Work, p: Program, step: number, env: Env) {
  const { A, B, C } = w.device;
  w.kernel = 'done';
  if (!A || !B || !C) return;
  for (const [name, arr] of [['A', A], ['B', B], ['C', C]] as const) {
    if (arr.freed) note(w, step, 'use-after-free', tr(`the kernel touches ${name}_d after it was freed`, `kernel chạm vào ${name}_d sau khi nó đã được giải phóng`), env);
    else if (arr.cells.length < p.n)
      note(w, step, 'out-of-bounds', tr(`the kernel indexes ${p.n} elements of ${name}_d, which holds ${arr.cells.length}`, `kernel đánh chỉ số ${p.n} phần tử của ${name}_d, mảng này chứa ${arr.cells.length}`), env);
  }
  if (w.stoppedAt != null) return;
  for (let i = 0; i < Math.min(p.n, C.cells.length); i++)
    C.cells[i] = w.tainted ? 'undetermined' : add(A.cells[i] ?? 'undetermined', B.cells[i] ?? 'undetermined');
  C.inFlight = false;
}

/** Everything issued so far finishes: what a blocking call or a wait brings about. */
function drain(w: Work, p: Program, step: number, env: Env) {
  if (!w.queue.length) return;
  w.ran = [...w.queue];
  w.queue = [];
  runKernel(w, p, step, env);
}

function exec(w: Work, p: Program, call: Call, step: number, env: Env) {
  const o = call.op;
  const hostName = (a: ArrayId) => `${a}_h`;
  const devName = (a: ArrayId) => (p.backend === 'metal' ? tr(`buffer ${a}`, `buffer ${a}`) : `${a}_d`);
  w.hostThread = 'running';
  w.ran = [];
  w.apiCalls += call.api.length;
  switch (o.op) {
    case 'alloc': {
      const cells = Math.floor(o.bytes / FLOAT_BYTES);
      w.device[o.array] = { cells: Array<Cell>(cells).fill(fresh(p, env)), bytes: o.bytes, freed: false, inFlight: false };
      w.says = p.backend === 'metal'
        ? tr(`Buffer ${o.array}: ${o.bytes} bytes, filled with zeros. Both the CPU and the GPU can use it.`, `Buffer ${o.array}: ${o.bytes} byte, được điền toàn số 0. Cả CPU lẫn GPU đều dùng được nó.`)
        : tr(`${devName(o.array)} now holds the address of ${o.bytes} bytes of device memory. The memory is not cleared, and the host cannot read it.`, `${devName(o.array)} giờ giữ địa chỉ của ${o.bytes} byte bộ nhớ device. Vùng nhớ này không được xóa, và host không thể đọc nó.`);
      return;
    }
    case 'alloc-copy': {
      const cells = Math.floor(o.bytes / FLOAT_BYTES);
      const src = w.host![o.array];
      w.device[o.array] = { cells: src.slice(0, cells), bytes: o.bytes, freed: false, inFlight: false };
      w.bytesMoved += o.bytes;
      w.copies += 1;
      w.says = tr(`A new buffer, and ${o.bytes} bytes copied into it from ${hostName(o.array)}: the caller's array was not in a buffer, so it has to be put in one.`, `Một buffer mới, và ${o.bytes} byte được sao chép vào đó từ ${hostName(o.array)}: mảng của bên gọi chưa nằm trong buffer nào, nên phải đưa nó vào một buffer.`);
      return;
    }
    case 'fill': {
      const arr = w.device[o.array]!;
      const data = inputs(p.n)[o.array as 'A' | 'B'];
      for (let i = 0; i < arr.cells.length; i++) arr.cells[i] = data[i];
      w.says = tr(`The caller produces ${o.array} straight into the buffer through its contents pointer. Nothing is copied, because the data never existed anywhere else.`, `Bên gọi tạo ${o.array} thẳng vào buffer qua con trỏ contents của nó. Không có gì được sao chép, vì dữ liệu chưa từng tồn tại ở nơi nào khác.`);
      return;
    }
    case 'copy': {
      const count = Math.floor(o.bytes / FLOAT_BYTES);
      if (p.memory === 'separate') {
        // A blocking copy in the same in-order stream as the launch (facts A4, A5).
        w.hostThread = 'blocked';
        drain(w, p, step, env);
        if (w.stoppedAt != null) return;
      }
      const matches = o.kind === 'h2d' ? o.dst.side === 'device' && o.src.side === 'host' : o.dst.side === 'host' && o.src.side === 'device';
      const cellsOf = (r: Ref): Cell[] | null => (r.side === 'host' ? w.host![r.array] : w.device[r.array]?.cells ?? null);
      const dst = cellsOf(o.dst);
      const src = cellsOf(o.src);
      w.bytesMoved += o.bytes;
      w.copies += 1;
      const toHostC = o.dst.side === 'host' && o.dst.array === 'C';
      if (!matches && env !== 'emu') {
        // "dst and src pointers that do not match the direction of the copy results in an undefined behavior"
        note(w, step, 'undefined-per-docs', tr('the pointers do not match the copy kind', 'các con trỏ không khớp với kiểu sao chép'), env);
        // After undefined behaviour neither array the call was handed has known contents.
        dst?.fill('undetermined');
        src?.fill('undetermined');
        w.says = tr('The destination is a device pointer and the source a host pointer, but the kind says device to host. The documentation calls that undefined behaviour and promises no error code.', 'Đích là một con trỏ device và nguồn là một con trỏ host, nhưng kiểu sao chép lại ghi là device sang host. Tài liệu gọi đó là hành vi không xác định và không hứa hẹn mã lỗi nào.');
        return;
      }
      if (!dst || !src) return;
      for (const r of [o.src, o.dst])
        if (r.side === 'device' && w.device[r.array]?.freed)
          note(w, step, 'use-after-free', tr(`the copy ${r === o.src ? 'reads' : 'writes'} ${devName(r.array)} after it was freed`, `lần sao chép ${r === o.src ? 'đọc' : 'ghi'} ${devName(r.array)} sau khi nó đã được giải phóng`), env);
      if (src.length < count)
        note(w, step, 'out-of-bounds', tr(`the copy reads ${o.bytes} bytes from an allocation of ${src.length * FLOAT_BYTES}`, `lần sao chép đọc ${o.bytes} byte từ một vùng cấp phát cỡ ${src.length * FLOAT_BYTES}`), env);
      if (dst.length < count)
        note(w, step, 'out-of-bounds', tr(`the copy writes ${o.bytes} bytes into an allocation of ${dst.length * FLOAT_BYTES}`, `lần sao chép ghi ${o.bytes} byte vào một vùng cấp phát cỡ ${dst.length * FLOAT_BYTES}`), env);
      if (w.stoppedAt != null) return;
      const inFlight = o.src.side === 'device' && !!w.device[o.src.array]?.inFlight;
      for (let i = 0; i < Math.min(count, dst.length); i++)
        dst[i] = w.tainted || inFlight ? 'undetermined' : src[i] ?? 'undetermined';
      if (toHostC)
        w.copiedBack = w.kernel !== 'done' ? 'before the kernel finished' : count < p.n ? 'in part' : 'yes'; // i18n-ignore: copiedBack identifier, not shown
      const what = tr(`${o.bytes} bytes from ${o.src.side === 'host' ? hostName(o.src.array) : devName(o.src.array)} to ${o.dst.side === 'host' ? hostName(o.dst.array) : devName(o.dst.array)}`, `${o.bytes} byte từ ${o.src.side === 'host' ? hostName(o.src.array) : devName(o.src.array)} sang ${o.dst.side === 'host' ? hostName(o.dst.array) : devName(o.dst.array)}`);
      if (p.memory === 'shared') w.says = tr(`An ordinary memcpy of ${what}: the caller wants its result in its own array.`, `Một memcpy thông thường của ${what}: bên gọi muốn kết quả nằm trong mảng của chính nó.`);
      else if (w.ran.length) w.says = tr(`The host waits in this call: first the kernel issued before it finishes, then ${what} cross the boundary.`, `Host chờ trong lời gọi này: trước hết kernel được phát trước đó chạy xong, rồi ${what} đi qua ranh giới.`);
      else w.says = tr(`The host waits in this call while ${what} cross the boundary.`, `Host chờ trong lời gọi này trong khi ${what} đi qua ranh giới.`);
      if (count < p.n) w.says += tr(` That is ${count} of the ${p.n} elements.`, ` Tức là ${count} trên ${p.n} phần tử.`);
      return;
    }
    case 'encode':
      w.says = tr('Eight calls describe the work: which kernel, which buffers, how many threads. Nothing has been sent to the GPU yet.', 'Tám lời gọi mô tả công việc: kernel nào, buffer nào, bao nhiêu luồng. Chưa có gì được gửi tới GPU.');
      return;
    case 'launch': {
      w.queue.push(KERNEL[p.backend]);
      w.kernel = 'issued';
      const C = w.device.C;
      if (C && !C.freed) C.inFlight = true;
      w.says = p.backend === 'metal'
        ? tr('The command buffer is committed and the call returns. The GPU will run the kernel when it gets to it; the host carries on.', 'Command buffer được commit và lời gọi trả về. GPU sẽ chạy kernel khi đến lượt nó; host tiếp tục chạy.')
        : tr('The launch returns at once. The kernel is set up to run on the device; the host does not wait for it to finish, or even to start.', 'Lệnh khởi chạy trả về ngay. Kernel được thiết lập để chạy trên device; host không chờ nó chạy xong, thậm chí không chờ nó bắt đầu.');
      if (env === 'emu') {
        drain(w, p, step, env); // the emulator runs the grid to completion inside the launch
        w.says = tr('The emulator runs the whole grid inside the launch statement.', 'Trình giả lập chạy toàn bộ lưới bên trong câu lệnh khởi chạy.');
      }
      return;
    }
    case 'wait':
      w.hostThread = 'blocked';
      drain(w, p, step, env);
      w.says = tr('The host blocks here until the GPU has finished the command buffer. Only now are the contents of C the result.', 'Host bị chặn ở đây cho đến khi GPU hoàn tất command buffer. Chỉ đến lúc này nội dung của C mới là kết quả.');
      return;
    case 'read': {
      const C = w.device.C;
      // Before the wait the read succeeds, and what it sees is not defined once the GPU may be writing.
      w.consumed = !C ? [] : w.kernel === 'issued' ? C.cells.map(() => 'undetermined' as Cell) : C.cells.slice();
      w.says = w.kernel === 'done'
        ? tr('The caller reads the sums through the contents pointer of C. No copy: it is the same memory the kernel wrote.', 'Bên gọi đọc các tổng qua con trỏ contents của C. Không có lần sao chép nào: đó chính là vùng nhớ mà kernel đã ghi.')
        : tr('The caller reads C through its contents pointer before the GPU work on it has completed. The read succeeds and the values are not the result.', 'Bên gọi đọc C qua con trỏ contents của nó trước khi công việc của GPU trên nó hoàn tất. Lần đọc thành công và các giá trị không phải là kết quả.');
      return;
    }
    case 'free': {
      // hipFree always synchronises first (B1); cudaFree may (A3).
      if (w.queue.length) {
        w.hostThread = 'blocked';
        drain(w, p, step, env);
        if (w.stoppedAt != null) return;
      }
      const arr = w.device[o.array];
      if (arr) arr.freed = true;
      w.frees += 1;
      w.says = tr(`${devName(o.array)} is returned to the device's pool. The variable still holds the old address, which now means nothing.`, `${devName(o.array)} được trả về vùng nhớ chung (pool) của device. Biến vẫn giữ địa chỉ cũ, giờ không còn ý nghĩa gì.`);
      return;
    }
    case 'host-deref': {
      if (env === 'cuda')
        note(w, step, 'not-allowed', tr(`host code reads ${devName(o.array)}[0]: device memory is only accessible to kernels`, `mã host đọc ${devName(o.array)}[0]: chỉ kernel mới truy cập được bộ nhớ device`), env);
      else if (env === 'hip')
        note(w, step, 'slow-zero-copy', tr(`host code reads ${devName(o.array)}[0]: AMD documents this as slow "zero copy" access`, `mã host đọc ${devName(o.array)}[0]: tài liệu của AMD mô tả đây là truy cập "zero copy" chậm`), env);
      w.says = env === 'emu'
        ? tr('Host code reads through a device pointer. In the emulator host and device share one address space, so the read works and nothing reports it.', 'Mã host đọc qua một con trỏ device. Trong trình giả lập, host và device dùng chung một không gian địa chỉ, nên lần đọc thành công và không có gì báo lại.')
        : env === 'hip'
          ? tr('Host code reads through a device pointer. AMD documents host access to hipMalloc memory as slow "zero copy" access, dependent on configuration.', 'Mã host đọc qua một con trỏ device. Tài liệu của AMD mô tả việc host truy cập bộ nhớ hipMalloc là truy cập "zero copy" chậm, tùy thuộc cấu hình.')
          : tr('Host code reads through a device pointer. CUDA does not allow it and does not document what happens.', 'Mã host đọc qua một con trỏ device. CUDA không cho phép điều này và không ghi trong tài liệu điều gì xảy ra.');
      return;
    }
  }
}

function pointersOf(w: Work, p: Program): PointerVar[] {
  const out: PointerVar[] = [];
  const ids: ArrayId[] = ['A', 'B', 'C'];
  if (w.host)
    for (const a of ids)
      out.push({ name: `${a}_h`, refersTo: p.memory === 'shared' ? 'the one shared memory' : 'host memory', hostMayUse: true }); // i18n-ignore: identifier also shown to the reader
  for (const a of ids) {
    const d = w.device[a];
    if (p.backend === 'metal') {
      if (d) out.push({ name: `${a}.contents`, refersTo: 'the one shared memory', hostMayUse: true }); // i18n-ignore: identifier also shown to the reader
      else out.push({ name: `${a}.contents`, refersTo: 'nothing yet', hostMayUse: false }); // i18n-ignore: identifier also shown to the reader
    } else
      out.push({ name: `${a}_d`, refersTo: !d ? 'nothing yet' : d.freed ? 'freed device memory' : 'device memory', hostMayUse: false }); // i18n-ignore: identifier also shown to the reader
  }
  return out;
}

function hostReadOf(w: Work, p: Program): HostRead {
  if (p.memory === 'separate') {
    const values = w.host!.C.slice();
    if (w.finding?.kind === 'undefined-per-docs')
      return { status: 'stale', values, why: tr('A call with undefined behaviour was handed C_h. Nothing about its contents is known after that.', 'Một lời gọi có hành vi không xác định đã được trao C_h. Sau đó không còn biết gì về nội dung của nó.') };
    if (w.copiedBack === 'no')
      return {
        status: 'impossible', values,
        why: w.device.C
          ? tr('The sums are, or will be, in device memory, and host code cannot read device memory. C_h still holds what the caller left there.', 'Các tổng đang, hoặc sẽ, nằm trong bộ nhớ device, và mã host không thể đọc bộ nhớ device. C_h vẫn giữ những gì bên gọi để lại ở đó.')
          : tr('There is no result anywhere yet. C_h holds what the caller left there.', 'Chưa có kết quả ở đâu cả. C_h giữ những gì bên gọi để lại ở đó.'),
      };
    if (w.copiedBack === 'yes' && !values.includes('undetermined'))
      return { status: 'valid', values, why: tr('The copy back waited for the kernel and then brought every element across.', 'Lần sao chép về đã chờ kernel rồi mang mọi phần tử qua.') };
    return {
      status: 'stale', values,
      why: w.copiedBack === 'in part' ? tr('Only part of the result was copied back.', 'Chỉ một phần kết quả được sao chép về.') : tr('C_h was filled, but not with the finished result.', 'C_h đã được điền, nhưng không phải bằng kết quả hoàn chỉnh.'),
    };
  }
  const C = w.device.C;
  if (!C) return { status: 'impossible', values: null, why: tr('The result buffer does not exist yet.', 'Buffer kết quả chưa tồn tại.') };
  const values = C.cells.slice();
  if (w.kernel === 'done') return { status: 'valid', values, why: tr('The host waited for the command buffer, so the contents are the result.', 'Host đã chờ command buffer, nên những gì trong contents là kết quả.') };
  return {
    status: 'stale', values,
    why: w.kernel === 'issued'
      ? tr('The read works, because the memory is shared. But the GPU may not have started, or may be half way through: these may be zeros, sums, or a mix.', 'Lần đọc thành công, vì bộ nhớ được dùng chung. Nhưng GPU có thể chưa bắt đầu, hoặc đang làm dở: đây có thể là các số 0, các tổng, hoặc lẫn lộn cả hai.')
      : tr('The read works, because the memory is shared. Nothing has been sent to the GPU, so these are the zeros the buffer was created with.', 'Lần đọc thành công, vì bộ nhớ được dùng chung. Chưa có gì được gửi tới GPU, nên đây là các số 0 mà buffer có từ khi được tạo.'),
  };
}

/**
 * The state after the first `upTo` calls of `p`. `env` says whose rules apply:
 * the program's own backend as documented (the default), or the course emulator
 * ('emu', CUDA-family programs only).
 */
export function run(p: Program, upTo: number = p.calls.length, env: Env = p.backend): State {
  const { A, B } = inputs(p.n);
  const callerOwnsArrays = !(p.backend === 'metal' && p.iface === 'buffers');
  const w: Work = {
    step: 0,
    host: callerOwnsArrays ? { A: [...A], B: [...B], C: Array<Cell>(p.n).fill(SENTINEL) } : null,
    device: { A: null, B: null, C: null },
    pointers: [],
    hostThread: 'running',
    ran: [],
    queue: [],
    bytesMoved: 0,
    copies: 0,
    frees: 0,
    apiCalls: 0,
    kernel: 'not issued', // i18n-ignore: kernel state identifier, not shown
    consumed: null,
    hostRead: { status: 'impossible', values: null, why: '' },
    finding: null,
    stoppedAt: null,
    says: '',
    tainted: false,
    copiedBack: 'no',
  };
  const last = Math.max(0, Math.min(upTo, p.calls.length));
  for (let i = 0; i < last && w.stoppedAt == null; i++) {
    exec(w, p, p.calls[i], i + 1, env);
    w.step = i + 1;
  }
  if (w.stoppedAt != null)
    w.says = tr(`AddressSanitizer stops the run here: ${w.finding!.detail}.`, `AddressSanitizer dừng lần chạy tại đây: ${w.finding!.detail}.`);
  w.pointers = pointersOf(w, p);
  w.hostRead = hostReadOf(w, p);
  const state: State & Partial<Work> = { ...w };
  delete state.tainted;
  delete state.copiedBack;
  return state;
}

/** What the caller ends up holding: C_h, or the contents of buffer C when the data lives in buffers. */
export function result(s: State): Cell[] {
  return s.host ? s.host.C : s.consumed ?? s.device.C?.cells ?? [];
}

/** One call of the stub on the two lanes of a timeline: the host thread and the device. */
export interface LaneStep {
  id: string;
  group: Group;
  /** What the host thread does during this call. */
  host: 'running' | 'blocked';
  /** Device work that finishes while the host is inside this call. */
  finishes: string[];
  /** Device work issued and still not waited for when the call returns. */
  pending: string[];
  /** Bytes that have crossed (separate memory) or been copied (shared memory) once the call returns. */
  bytesMoved: number;
}

/** Who waits for whom, call by call. The model has no durations: this is order only. */
export function lanes(p: Program, env: Env = p.backend): LaneStep[] {
  return p.calls.map((c, i) => {
    const s = run(p, i + 1, env);
    return { id: c.id, group: c.group, host: s.hostThread, finishes: s.ran, pending: s.queue, bytesMoved: s.bytesMoved };
  });
}

// ---------------------------------------------------------------- defects
export const DEFECTS = [
  'bytes-as-elements', // a copy is given n where it needs n * sizeof(float)
  'alloc-in-elements', // cudaMalloc(n) with copies of n * sizeof(float)
  'missing-copy-in', // B is never copied to the device
  'copy-back-early', // the copy back comes before the launch
  'free-before-copy-back', // C_d is freed, then copied from
  'swapped-pointers', // destination and source exchanged on the copy back, kind unchanged
  'host-dereference', // host code reads C_d[0]
] as const;
export type Defect = (typeof DEFECTS)[number];

export const DEFECT_LABEL: Record<Defect, string> = {
  get 'bytes-as-elements'() {
    return tr('The copy back is given n as its size, not n * sizeof(float).', 'Lệnh sao chép về được cho n làm kích thước, chứ không phải n * sizeof(float).');
  },
  get 'alloc-in-elements'() {
    return tr('The three allocations ask for n bytes; the copies move n * sizeof(float).', 'Ba vùng cấp phát xin n byte; các lệnh sao chép chuyển n * sizeof(float).');
  },
  get 'missing-copy-in'() {
    return tr('B is never copied to the device.', 'B không bao giờ được sao chép sang device.');
  },
  get 'copy-back-early'() {
    return tr('The copy back comes before the launch.', 'Lệnh sao chép về đứng trước lệnh khởi chạy.');
  },
  get 'free-before-copy-back'() {
    return tr('C_d is freed before the copy back.', 'C_d bị giải phóng trước lệnh sao chép về.');
  },
  get 'swapped-pointers'() {
    return tr('The copy back has its first two arguments exchanged, with the kind unchanged.', 'Lệnh sao chép về bị đổi chỗ hai đối số đầu, với kiểu sao chép giữ nguyên.');
  },
  get 'host-dereference'() {
    return tr('Host code reads C_d[0] directly.', 'Mã host đọc trực tiếp C_d[0].');
  },
};

/**
 * The program with exactly one defect, or null where the mistake cannot be
 * written (a Metal stub with shared buffers has no byte counts on copies, no
 * free call and no copy kind to get wrong).
 */
export function mutate(p: Program, defect: Defect): Program | null {
  const calls = p.calls.map((c) => ({ ...c }));
  const at = (id: string) => calls.findIndex((c) => c.id === id);
  const done = (cs: Call[]): Program => ({ ...p, calls: cs, defect });
  const move = (id: string, before: string) => {
    const [c] = calls.splice(at(id), 1);
    calls.splice(at(before), 0, { ...c, changed: true });
    return done(calls);
  };
  if (p.backend === 'metal') {
    if (p.iface !== 'buffers') return null;
    if (defect === 'missing-copy-in') return done(calls.filter((c) => c.id !== 'fill-B'));
    if (defect === 'copy-back-early') return move('read', 'launch');
    return null;
  }
  const pre = p.backend;
  const elems = p.n; // the count in elements, used where bytes are needed
  switch (defect) {
    case 'bytes-as-elements': {
      const c = calls[at('copy-back')];
      c.code = [`${pre}Memcpy(C_h, C_d, n, ${pre}MemcpyDeviceToHost);`]; // i18n-ignore: C source line
      c.op = { ...(c.op as Extract<Op, { op: 'copy' }>), bytes: elems };
      c.changed = true;
      return done(calls);
    }
    case 'alloc-in-elements':
      for (const a of ['A', 'B', 'C'] as const) {
        const c = calls[at('alloc-' + a)];
        c.code = [`${pre}Malloc((void**)&${a}_d, n);`]; // i18n-ignore: C source line
        c.op = { op: 'alloc', array: a, bytes: elems };
        c.changed = true;
      }
      return done(calls);
    case 'missing-copy-in':
      return done(calls.filter((c) => c.id !== 'copy-in-B'));
    case 'copy-back-early':
      return move('copy-back', 'launch');
    case 'free-before-copy-back':
      return move('free-C', 'copy-back');
    case 'swapped-pointers': {
      const c = calls[at('copy-back')];
      c.code = [`${pre}Memcpy(C_d, C_h, size, ${pre}MemcpyDeviceToHost);`]; // i18n-ignore: C source line
      c.op = { op: 'copy', dst: { side: 'device', array: 'C' }, src: { side: 'host', array: 'C' }, bytes: p.n * FLOAT_BYTES, kind: 'd2h' };
      c.changed = true;
      return done(calls);
    }
    case 'host-dereference':
      calls.splice(at('copy-back') + 1, 0, {
        id: 'host-deref', part: 3, group: 'read',
        code: ['float first = C_d[0];'],
        api: [],
        op: { op: 'host-deref', array: 'C' },
        changed: true,
      });
      return done(calls);
  }
}

export type OutcomeClass =
  | 'no-counterpart' // the mistake cannot be written on this backend
  | 'passes' // the result is right and nothing reports the defect
  | 'wrong-values'
  | 'out-of-bounds'
  | 'use-after-free'
  | 'undefined-per-docs'
  | 'not-documented'
  | 'slow-zero-copy';

export const CLASS_LABEL: Record<OutcomeClass, string> = {
  get 'no-counterpart'() {
    return tr('cannot be written here', 'không thể viết lỗi này ở đây');
  },
  get passes() {
    return tr('right answer, nothing reported', 'đáp án đúng, không có gì được báo');
  },
  get 'wrong-values'() {
    return tr('wrong values', 'giá trị sai');
  },
  get 'out-of-bounds'() {
    return tr('out of bounds', 'truy cập ngoài biên');
  },
  get 'use-after-free'() {
    return tr('use after free', 'dùng sau khi giải phóng');
  },
  get 'undefined-per-docs'() {
    return tr('undefined behaviour per the documentation', 'hành vi không xác định theo tài liệu');
  },
  get 'not-documented'() {
    return tr('not allowed, failure not documented', 'không được phép, lỗi không được ghi trong tài liệu');
  },
  get 'slow-zero-copy'() {
    return tr('slow zero-copy access per AMD\'s documentation', 'truy cập zero-copy chậm theo tài liệu của AMD');
  },
};

export interface Outcome {
  defect: Defect;
  env: Env;
  klass: OutcomeClass;
  /** The mutated program that was run; null when there is no counterpart. */
  program: Program | null;
  /** What the caller holds afterwards. Empty when the run was stopped or has no counterpart. */
  result: Cell[];
  /** Positions that do not hold the right sum. */
  wrong: number[];
  /** An error code the documentation guarantees for this defect. None of the seven has one. */
  errorCode: string | null;
  /** Emulator: the call at which AddressSanitizer ends the run, or null. */
  stoppedAt: number | null;
  finding: Finding | null;
  /** One or two sentences, in the documentation's terms. */
  summary: string;
  /** Where the rule comes from in notes/today/ch02-facts.md. */
  facts: string;
}

interface Texts {
  /** Where the rule comes from in notes/today/ch02-facts.md. */
  facts: string;
  cuda: string;
  /** Defaults to the CUDA text where AMD documents the same thing. */
  hip?: string;
  metal?: string;
  emu: string;
}
const SUMMARY: Record<Defect, Texts> = {
  'bytes-as-elements': {
    facts: 'A4, B1',
    get cuda() {
      return tr('The copy moves n bytes, which is n/4 floats. The rest of C_h keeps what the caller had there. Every call returns success.', 'Lần sao chép chuyển n byte, tức n/4 số float. Phần còn lại của C_h giữ nguyên những gì bên gọi có ở đó. Mọi lời gọi đều trả về thành công.');
    },
    get emu() {
      return tr('Runs to the end: the copy is in bounds, only too short. The first n/4 elements are right and the rest still hold the test\'s sentinel.', 'Chạy đến cuối: lần sao chép nằm trong biên, chỉ là quá ngắn. n/4 phần tử đầu đúng và phần còn lại vẫn giữ giá trị canh (sentinel) của bài kiểm tra.');
    },
  },
  'alloc-in-elements': {
    facts: 'A2, A6',
    get cuda() {
      return tr('Each allocation is a quarter of what the copies and the kernel use, so they write past its end. Nothing in the documentation says device arrays are bounds-checked, and the result is not determined: it may even be right.', 'Mỗi vùng cấp phát chỉ bằng một phần tư lượng mà các lần sao chép và kernel dùng, nên chúng ghi vượt quá cuối vùng. Tài liệu không nói gì về việc mảng trên device được kiểm tra biên, và kết quả không xác định: thậm chí có thể đúng.');
    },
    get emu() {
      return tr('AddressSanitizer stops the run at the first copy in: a write of 4n bytes into an allocation of n.', 'AddressSanitizer dừng lần chạy ở lần sao chép vào đầu tiên: ghi 4n byte vào một vùng cấp phát cỡ n.');
    },
  },
  'missing-copy-in': {
    facts: 'A2, B1, C2',
    get cuda() {
      return tr('The kernel adds A to whatever B_d happened to hold: cudaMalloc does not clear memory. Every element is wrong and no call fails.', 'Kernel cộng A với bất cứ thứ gì B_d tình cờ đang chứa: cudaMalloc không xóa bộ nhớ. Mọi phần tử đều sai và không lời gọi nào thất bại.');
    },
    get hip() {
      return tr('The kernel adds A to whatever B_d happened to hold. AMD\'s header does not say whether hipMalloc clears memory, so assume it does not. Every element is wrong and no call fails.', 'Kernel cộng A với bất cứ thứ gì B_d tình cờ đang chứa. Header của AMD không nói hipMalloc có xóa bộ nhớ hay không, nên hãy giả định là không. Mọi phần tử đều sai và không lời gọi nào thất bại.');
    },
    get metal() {
      return tr('The buffer for B is zero-filled, so the result is A + 0 = A: wrong, and plausible.', 'Buffer cho B được điền số 0, nên kết quả là A + 0 = A: sai, mà trông vẫn hợp lý.');
    },
    get emu() {
      return tr('Runs to the end. The emulator fills fresh device memory with 0xA5 bytes, a float of about -2.9e-16, which vanishes when added to A. The result equals A.', 'Chạy đến cuối. Trình giả lập điền bộ nhớ device mới cấp phát bằng các byte 0xA5, một số float xấp xỉ -2.9e-16, biến mất khi cộng vào A. Kết quả bằng A.');
    },
  },
  'copy-back-early': {
    facts: 'A2, A5, C6',
    get cuda() {
      return tr('C_h receives what C_d held before any kernel wrote it: memory that was never cleared. The kernel then runs and its result is never fetched.', 'C_h nhận những gì C_d chứa trước khi có kernel nào ghi vào: vùng nhớ chưa từng được xóa. Sau đó kernel chạy và kết quả của nó không bao giờ được lấy về.');
    },
    get metal() {
      return tr('The read works, because the memory is shared, and returns the zeros the buffer was created with.', 'Lần đọc thành công, vì bộ nhớ được dùng chung, và trả về các số 0 mà buffer có từ khi được tạo.');
    },
    get emu() {
      return tr('Runs to the end. C_h holds the emulator\'s 0xA5 fill in every element.', 'Chạy đến cuối. Mọi phần tử của C_h giữ giá trị điền 0xA5 của trình giả lập.');
    },
  },
  'free-before-copy-back': {
    facts: 'A3, A6, B1',
    get cuda() {
      return tr('The copy reads memory that has been returned to the pool. No documented error code covers it and the values are not determined: they may still look right.', 'Lần sao chép đọc vùng nhớ đã được trả về pool. Không có mã lỗi nào trong tài liệu bao quát trường hợp này và các giá trị không xác định: chúng vẫn có thể trông đúng.');
    },
    get emu() {
      return tr('AddressSanitizer stops the run at the copy back: a read of freed memory.', 'AddressSanitizer dừng lần chạy ở lần sao chép về: một lần đọc vùng nhớ đã giải phóng.');
    },
  },
  'swapped-pointers': {
    facts: 'A4, B1',
    get cuda() {
      return tr('Pointers that do not match the copy kind are documented as undefined behaviour. cudaErrorInvalidMemcpyDirection exists, but nothing promises you will get it.', 'Tài liệu ghi rằng các con trỏ không khớp với kiểu sao chép là hành vi không xác định. cudaErrorInvalidMemcpyDirection có tồn tại, nhưng không có gì hứa rằng bạn sẽ nhận được nó.');
    },
    get hip() {
      return tr('Pointers that do not match the copy kind are documented as undefined behaviour, in the same words as CUDA\'s. No error code is promised.', 'Tài liệu ghi rằng các con trỏ không khớp với kiểu sao chép là hành vi không xác định, bằng đúng lời lẽ như tài liệu của CUDA. Không có mã lỗi nào được hứa hẹn.');
    },
    get emu() {
      return tr('Runs to the end. The emulator copies as the pointers say, host to device, so C_h is never written and keeps the sentinel.', 'Chạy đến cuối. Trình giả lập sao chép theo hướng các con trỏ chỉ ra, từ host sang device, nên C_h không bao giờ được ghi và giữ nguyên giá trị canh (sentinel).');
    },
  },
  'host-dereference': {
    facts: 'A17, B7, C2, C3',
    get cuda() {
      return tr('The result is right, and the stub then does something CUDA does not allow. NVIDIA does not document what happens; the book says exceptions or other run-time errors.', 'Kết quả đúng, rồi hàm stub làm một việc CUDA không cho phép. NVIDIA không ghi trong tài liệu điều gì xảy ra; sách nói là ngoại lệ hoặc các lỗi thời gian chạy khác.');
    },
    get hip() {
      return tr('The result is right. AMD documents host access to hipMalloc memory as "zero copy" access over a slow interface, dependent on configuration and not tested here.', 'Kết quả đúng. Tài liệu của AMD mô tả việc host truy cập bộ nhớ hipMalloc là truy cập "zero copy" qua một giao tiếp chậm, tùy thuộc cấu hình và không được kiểm tra ở đây.');
    },
    get metal() {
      return tr('Reading a shared buffer through contents is the normal path, once the host has waited for the GPU work on it.', 'Đọc một buffer dùng chung qua contents là cách làm bình thường, một khi host đã chờ công việc của GPU trên nó xong.');
    },
    get emu() {
      return tr('Every test passes. Host and device share one address space in the emulator, so nothing can report the read.', 'Mọi bài kiểm tra đều qua. Trong trình giả lập host và device dùng chung một không gian địa chỉ, nên không gì có thể báo lại lần đọc này.');
    },
  },
};
const NO_COUNTERPART = () =>
  tr(
    'With the data in shared buffers there is no call in which to make this mistake: no byte count on a copy, no copy kind, no free.',
    'Khi dữ liệu nằm trong các buffer dùng chung, không có lời gọi nào để phạm lỗi này: không có số byte trên lần sao chép, không có kiểu sao chép, không có lệnh giải phóng.',
  );

/** What one defect produces under one set of rules. */
export function outcome(defect: Defect, env: Env, n: number = N): Outcome {
  const base = program(env === 'emu' ? 'cuda' : env, 'buffers', n);
  const mutated = mutate(base, defect);
  const summary = SUMMARY[defect];
  const text = summary[env] ?? (env === 'hip' ? summary.cuda : NO_COUNTERPART());
  if (!mutated) {
    // On Metal with shared buffers a host read through contents is the designed path, not a defect.
    const normal = env === 'metal' && defect === 'host-dereference';
    return {
      defect, env, klass: normal ? 'passes' : 'no-counterpart', program: null,
      result: normal ? expected(n) : [], wrong: [], errorCode: null, stoppedAt: null, finding: null,
      summary: text,
      facts: summary.facts,
    };
  }
  const s = run(mutated, mutated.calls.length, env);
  const want = expected(n);
  const res = s.stoppedAt != null ? [] : result(s);
  const wrong = want.flatMap((v, i) => (res[i] === v ? [] : [i]));
  let klass: OutcomeClass;
  if (s.finding)
    klass = s.finding.kind === 'not-allowed' ? 'not-documented' : s.finding.kind;
  else klass = wrong.length ? 'wrong-values' : 'passes';
  return {
    defect, env, klass, program: mutated, result: res, wrong: s.stoppedAt != null ? [] : wrong,
    errorCode: null, stoppedAt: s.stoppedAt, finding: s.finding, summary: text, facts: summary.facts,
  };
}

// ---------------------------------------------------------------- lab 2.2 and 2.3 benchmark rows
// The lessons show the reader's own measurements beside the model and never a
// modelled time. These helpers only read benchmark rows (reports/runs/*.json).
type Row = Record<string, string | number>;

/**
 * Which step group a benchmark column belongs to, by its name. Lab 2.2 reports
 * one time per phase of the stub; the stepper shows each beside its steps.
 *   CUDA, HIP:  alloc_s, h2d_s, kernel_s, d2h_s, free_s
 *   Metal:      wrap_s (create the buffers and copy the inputs in), encode_s,
 *               wait_s, readback_s
 * total_s, gpu_s (the GPU time inside the wait) and the percentages are not phases.
 */
export function groupOfKey(key: string): Group | null {
  const k = key.toLowerCase();
  if (!/(_ms|_us|_s|ms|seconds)$/.test(k)) return null;
  if (/free/.test(k)) return 'free';
  if (/alloc|malloc|newbuffer/.test(k)) return 'allocate';
  if (/d2h|back|out|download|todevicehost/.test(k)) return 'copy back'; // i18n-ignore: Group identifier
  if (/h2d|copy_?in|upload|wrap|(^|_)in_/.test(k)) return 'copy in'; // i18n-ignore: Group identifier
  if (/wait|sync/.test(k)) return 'wait';
  if (/kernel|launch|compute|encode|commit/.test(k)) return 'launch';
  if (/fill/.test(k)) return 'fill';
  if (/read/.test(k)) return 'read';
  return null;
}

/** A time column's value in seconds, from the unit its name ends with; null when it is not a time. */
export function toSeconds(key: string, value: unknown): number | null {
  if (typeof value !== 'number' || !isFinite(value) || value < 0) return null;
  const k = key.toLowerCase();
  if (/_us$/.test(k)) return value * 1e-6;
  if (/(_ms|ms)$/.test(k)) return value * 1e-3;
  if (/(_s|seconds)$/.test(k)) return value;
  return null;
}

/** Seconds per step group from one 'phases' row of lab 2.2. Columns of the same group add up. */
export function phaseTimes(row: Row): Partial<Record<Group, number>> {
  const out: Partial<Record<Group, number>> = {};
  for (const [key, value] of Object.entries(row)) {
    const g = groupOfKey(key);
    const s = toSeconds(key, value);
    if (g == null || s == null) continue;
    out[g] = (out[g] ?? 0) + s;
  }
  return out;
}

/**
 * The two times of one 'chain' row of lab 2.3, in seconds: the chain with its
 * data kept on the device, and the same chain as one round trip per operation.
 * Null unless the row has both.
 */
export function chainTimes(row: Row): { resident: number; roundTrips: number } | null {
  let resident: number | null = null;
  let roundTrips: number | null = null;
  for (const [key, value] of Object.entries(row)) {
    const s = toSeconds(key, value);
    if (s == null) continue;
    const k = key.toLowerCase();
    if (/resident/.test(k)) resident = s;
    else if (/round|trip|transparent/.test(k)) roundTrips = s;
  }
  return resident != null && roundTrips != null ? { resident, roundTrips } : null;
}

// ---------------------------------------------------------------- boundary traffic
export type Strategy = 'transparent' | 'resident';
export type Memory = 'separate' | 'shared';

export interface Transfer {
  /** Index of the operation this transfer belongs to. */
  op: number;
  dir: 'in' | 'out';
  /** What is moved, e.g. 'x', 'bias', 'x after scale'. */
  what: string;
}

export interface Traffic {
  ops: string[];
  transfersIn: number;
  transfersOut: number;
  /** One entry per tensor-sized transfer, in program order. */
  transfers: Transfer[];
  bytesPerTensor: number;
  bytesAcross: number;
  /** Bytes the kernels read and write in device memory: the same whatever the strategy. */
  kernelBytes: number;
  /** bytesAcross / kernelBytes. */
  ratio: number;
}

const CHAIN = [
  { get name() { return tr('add bias', 'cộng bias'); }, operand: 'bias' },
  { get name() { return tr('scale', 'nhân hệ số'); }, operand: 'gain' },
  { get name() { return tr('add residual', 'cộng residual'); }, operand: 'residual' },
];
/** The k elementwise operations of the chain; past three they repeat as "layer 2", "layer 3". */
export function chainOps(k: number): { name: string; operand: string }[] {
  return Array.from({ length: k }, (_, i) => {
    const layer = Math.floor(i / CHAIN.length) + 1;
    const c = CHAIN[i % CHAIN.length];
    return k > CHAIN.length ? { name: `${c.name} ${layer}`, operand: `${c.operand} ${layer}` } : c;
  });
}

/**
 * A chain of k elementwise operations, each combining the running result x with
 * one fresh tensor of n floats.
 *   transparent: every operation is its own offload (two tensors in, one out)
 *   resident:    x and every operand go in once, the final x comes out once
 * With one shared memory, and the data created in shared buffers, nothing crosses.
 */
export function boundaryTraffic(k: number, strategy: Strategy, memory: Memory, n: number): Traffic {
  const ops = chainOps(k);
  const transfers: Transfer[] = [];
  if (memory === 'separate') {
    if (strategy === 'transparent')
      ops.forEach((o, i) => {
        transfers.push({ op: i, dir: 'in', what: 'x' }, { op: i, dir: 'in', what: o.operand }, { op: i, dir: 'out', what: 'x' });
      });
    else {
      transfers.push({ op: 0, dir: 'in', what: 'x' });
      ops.forEach((o, i) => transfers.push({ op: i, dir: 'in', what: o.operand }));
      transfers.push({ op: k - 1, dir: 'out', what: 'x' });
    }
  }
  const bytesPerTensor = n * FLOAT_BYTES;
  const transfersIn = transfers.filter((t) => t.dir === 'in').length;
  const transfersOut = transfers.length - transfersIn;
  const bytesAcross = transfers.length * bytesPerTensor;
  const kernelBytes = k * 3 * bytesPerTensor; // each op reads two tensors and writes one
  return {
    ops: ops.map((o) => o.name), transfersIn, transfersOut, transfers, bytesPerTensor, bytesAcross, kernelBytes,
    ratio: kernelBytes ? bytesAcross / kernelBytes : 0,
  };
}
