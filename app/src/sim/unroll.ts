// Chapter 6, section 6.6: loop unrolling. Runs the issue-timeline explorer, the
// "where does x live?" classification and the numbers in the section's questions.
//
// Sources for the rules (the lesson cites them where they are used):
//  - the book, section 6.6, pp. 144 to 146: a 16-iteration loop with body
//    A(i) B(i), unrolled by 4 into A(i) B(i) ... A(i+3) B(i+3) with i += 4;
//    reordered to A(i) .. A(i+3) B(i) .. B(i+3) when A(i+1) does not depend on
//    A(i) or B(i); a GPU thread stalls at a branch until it is resolved; the
//    compiler unrolls small constant-bound loops on its own; #pragma unroll N,
//    with 1 meaning "do not unroll"; a local array indexed with the loop counter
//    stays in memory, and only after full unrolling (constant indices) can it be
//    promoted to registers;
//  - notes/today/infra-facts.md A7: "If it is absent, the loop will be completely
//    unrolled if its trip count is constant. If it evaluates to 0 or 1, the loop
//    will not be unrolled.";
//  - notes/today/ch06-facts.md A13: NVIDIA's list of what goes to local memory,
//    first among them "Arrays for which it cannot determine that they are indexed
//    with constant quantities"; C4: Apple's "dynamic indexing" of thread-scoped
//    arrays pushes data to device memory.
//
// What this leaves out (the lesson says so beside the explorer):
//  - One thread on its own: no other warps to switch to while it waits, which is
//    the GPU's main way of hiding latency (chapter 4).
//  - In-order issue, one instruction per cycle, a fixed A-to-B latency and a fixed
//    stall after every loop branch. The loop's counter update and compare are
//    folded into the one branch instruction. Real compilers schedule more freely
//    and may hide some of this on their own.
//  - Only B(i) depends on A(i). Nothing depends on B, and A(i+1) is independent of
//    A(i) and B(i), which is the condition under which the book allows reordering.
//  - "Where x lives" ignores the other two reasons NVIDIA gives for local memory
//    (arrays too large for registers, and spilling when a kernel runs out of
//    registers): the snippets are small enough that only indexing decides.
//
// Erasable TypeScript only, and the one import is lib/locale.ts (itself import-free): the
// browser checks import this file directly in Node.

import { tr } from '../lib/locale.ts';

export type CellKind = 'issueA' | 'issueB' | 'branch' | 'stallDep' | 'stallBranch';

export interface Cell {
  kind: CellKind;
  /** The original loop index of the A or B issued, or the one a dependence stall waits for. */
  i?: number;
}

export interface UnrollConfig {
  /** Trip count of the original loop (16 in the book). */
  iterations: number;
  /** Unroll factor; must divide `iterations`. Equal to `iterations` means fully unrolled: no loop left. */
  unroll: number;
  /** Issue the unrolled body as all A's, then all B's. */
  reorder: boolean;
  /** Cycles B(i) must wait if it issues right after A(i); 0 means no wait. */
  latencyAB: number;
  /** Cycles the thread waits after each loop branch before it can issue again. */
  branchStall: number;
}

export interface Timeline {
  cells: Cell[];
  /** Loop branches executed: one per remaining iteration, none when fully unrolled. */
  branches: number;
  /** Iterations of the loop as unrolled (1 when fully unrolled). */
  loopIterations: number;
  stallDep: number;
  stallBranch: number;
  /** stallDep + stallBranch. */
  stalls: number;
  /** Cycles from the first issue to the last. */
  total: number;
}

export const DEFAULTS: UnrollConfig = { iterations: 16, unroll: 1, reorder: false, latencyAB: 3, branchStall: 2 };
export const FACTORS = [1, 2, 4, 8, 16];

/** In-order, one instruction per cycle: the cycle-by-cycle record of one thread running the loop. */
export function schedule(cfg: UnrollConfig): Timeline {
  const { iterations: n, unroll: u, reorder, latencyAB, branchStall } = cfg;
  if (u < 1 || n % u !== 0) throw new Error(`unroll factor ${u} does not divide ${n} iterations`);
  const full = u === n;
  const cells: Cell[] = [];
  const readyB = new Map<number, number>();
  let stallDep = 0;
  let stallBranch = 0;
  let branches = 0;

  const issueA = (i: number) => {
    readyB.set(i, cells.length + 1 + latencyAB);
    cells.push({ kind: 'issueA', i });
  };
  const issueB = (i: number) => {
    const ready = readyB.get(i)!;
    while (cells.length < ready) {
      cells.push({ kind: 'stallDep', i });
      stallDep++;
    }
    cells.push({ kind: 'issueB', i });
  };

  for (let base = 0; base < n; base += u) {
    if (reorder) {
      for (let k = 0; k < u; k++) issueA(base + k);
      for (let k = 0; k < u; k++) issueB(base + k);
    } else {
      for (let k = 0; k < u; k++) {
        issueA(base + k);
        issueB(base + k);
      }
    }
    if (!full) {
      branches++;
      cells.push({ kind: 'branch' });
      for (let s = 0; s < branchStall; s++) {
        cells.push({ kind: 'stallBranch' });
        stallBranch++;
      }
    }
  }
  return {
    cells,
    branches,
    loopIterations: n / u,
    stallDep,
    stallBranch,
    stalls: stallDep + stallBranch,
    total: cells.length,
  };
}

/** The body of one iteration of the unrolled loop, in the book's notation. */
export function bodyText(u: number, reorder: boolean): string {
  const ix = (k: number) => (k === 0 ? 'i' : `i+${k}`);
  const parts: string[] = [];
  if (reorder) {
    for (let k = 0; k < u; k++) parts.push(`A(${ix(k)})`);
    for (let k = 0; k < u; k++) parts.push(`B(${ix(k)})`);
  } else for (let k = 0; k < u; k++) parts.push(`A(${ix(k)})`, `B(${ix(k)})`);
  return parts.join(' ');
}

// ---------------------------------------------------------------- where does x live?
export type Place = 'registers' | 'local memory';

export interface WhereSnippet {
  id: string;
  /** Short name for the card. */
  label: string;
  /** The kernel as it appears in labs/ch06/snippets/s66_where_x_lives.cu (a unit test checks). */
  code: string;
  /** What decides it. */
  array: boolean;
  /** How x is indexed. */
  index: 'none' | 'constant' | 'loop counter' | 'run-time value';
  /** For a loop counter: what happens to the loop. */
  loop?: 'constant bound' | 'run-time bound' | 'unroll 1';
}

/** The book's rule (pp. 145 to 146) with NVIDIA's list (ch06-facts A13): where x is likely to be placed, and why. */
export function whereLives(s: WhereSnippet): { place: Place; rule: string } {
  if (!s.array) return { place: 'registers', rule: tr('A scalar local variable lives in a register (chapter 5).', 'Một biến cục bộ vô hướng nằm trong thanh ghi (chương 5).') };
  if (s.index === 'constant')
    return {
      place: 'registers',
      rule: tr(
        'Every index is a compile-time constant, so each element can be given its own register (pp. 145–146).',
        'Mọi chỉ số đều là hằng số lúc biên dịch, nên mỗi phần tử có thể được cấp một thanh ghi riêng (pp. 145–146).',
      ),
    };
  if (s.index === 'run-time value')
    return {
      place: 'local memory', // i18n-ignore: Place identity compared with ===; S66 shows it through its own tr()
      rule: tr(
        'The compiler cannot tell which element is meant, so the array must sit somewhere it can be indexed: memory, not registers. NVIDIA lists exactly this case (A13).',
        'Trình biên dịch không biết phần tử nào được chọn, nên mảng phải nằm ở nơi có thể đánh chỉ số: bộ nhớ, không phải thanh ghi. NVIDIA liệt kê đúng trường hợp này (A13).',
      ),
    };
  if (s.loop === 'constant bound')
    return {
      place: 'registers',
      rule: tr(
        'A small loop with a constant trip count is fully unrolled by default (p. 145; NVIDIA A7). After that every index is a constant.',
        'Một vòng lặp nhỏ có số lần lặp là hằng số được trải hết theo mặc định (p. 145; NVIDIA A7). Sau đó mọi chỉ số đều là hằng số.',
      ),
    };
  if (s.loop === 'unroll 1')
    return {
      place: 'local memory', // i18n-ignore: Place identity compared with ===; S66 shows it through its own tr()
      rule: tr(
        '#pragma unroll 1 forbids unrolling, so the index stays the loop counter, which is not a constant (p. 145; A7).',
        '#pragma unroll 1 cấm trải vòng lặp, nên chỉ số vẫn là biến đếm vòng lặp, vốn không phải hằng số (p. 145; A7).',
      ),
    };
  return {
    place: 'local memory', // i18n-ignore: Place identity compared with ===; S66 shows it through its own tr()
    rule: tr(
      'With a trip count known only at run time the loop cannot be fully unrolled, so the index stays a variable (A7, A13).',
      'Khi số lần lặp chỉ biết được lúc chạy, vòng lặp không thể được trải hết, nên chỉ số vẫn là một biến (A7, A13).',
    ),
  };
}

export const WHERE_SNIPPETS: WhereSnippet[] = [
  {
    id: 'scalar',
    get label() {
      return tr('one element per thread', 'một phần tử mỗi luồng');
    },
    array: false,
    index: 'none',
    code: `__global__ void scalar(const float* in, float* out) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    float x = in[i];
    out[i] = foo(x);
}`,
  },
  {
    id: 'runtime-bound',
    get label() {
      return tr('coarsened, loop bound known at run time', 'đã gộp việc, cận vòng lặp chỉ biết lúc chạy');
    },
    array: true,
    index: 'loop counter', // i18n-ignore: union member compared in code, not shown
    loop: 'run-time bound', // i18n-ignore: union member compared in code, not shown
    code: `__global__ void runtimeBound(const float* in, float* out, unsigned int count) {
    unsigned int iStart = COARSEN*(blockIdx.x*blockDim.x + threadIdx.x);
    float x[COARSEN];
    for (unsigned int c = 0; c < count; ++c) x[c] = in[iStart + c];
    for (unsigned int c = 0; c < count; ++c) out[iStart + c] = foo(x[c]);
}`,
  },
  {
    id: 'by-hand',
    get label() {
      return tr('coarsened, unrolled by hand', 'đã gộp việc, trải vòng lặp bằng tay');
    },
    array: true,
    index: 'constant',
    code: `__global__ void unrolledByHand(const float* in, float* out) {
    unsigned int iStart = COARSEN*(blockIdx.x*blockDim.x + threadIdx.x);
    float x[COARSEN];
    x[0] = in[iStart];     x[1] = in[iStart + 1];
    x[2] = in[iStart + 2]; x[3] = in[iStart + 3];
    out[iStart] = foo(x[0]);         out[iStart + 1] = foo(x[1]);
    out[iStart + 2] = foo(x[2]);     out[iStart + 3] = foo(x[3]);
}`,
  },
  {
    id: 'unroll-1',
    get label() {
      return tr('coarsened, #pragma unroll 1', 'đã gộp việc, #pragma unroll 1');
    },
    array: true,
    index: 'loop counter', // i18n-ignore: union member compared in code, not shown
    loop: 'unroll 1', // i18n-ignore: union member compared in code, not shown
    code: `__global__ void notUnrolled(const float* in, float* out) {
    unsigned int iStart = COARSEN*(blockIdx.x*blockDim.x + threadIdx.x);
    float x[COARSEN];
#pragma unroll 1
    for (unsigned int c = 0; c < COARSEN; ++c) x[c] = in[iStart + c];
#pragma unroll 1
    for (unsigned int c = 0; c < COARSEN; ++c) out[iStart + c] = foo(x[c]);
}`,
  },
  {
    id: 'runtime-index',
    get label() {
      return tr('one element picked at run time', 'một phần tử được chọn lúc chạy');
    },
    array: true,
    index: 'run-time value', // i18n-ignore: union member compared in code, not shown
    code: `__global__ void runtimeIndex(const float* in, float* out, const int* pick) {
    unsigned int t = blockIdx.x*blockDim.x + threadIdx.x;
    float x[COARSEN];
#pragma unroll
    for (unsigned int c = 0; c < COARSEN; ++c) x[c] = in[COARSEN*t + c];
    out[t] = foo(x[pick[t]]);       // which element: known only at run time
}`,
  },
];
