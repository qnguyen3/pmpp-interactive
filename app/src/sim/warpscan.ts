// A lane-level model of the warp scan of Fig. 11.8 and the block scan of Fig. 11.9, and of
// the scan-scan-add decomposition behind both (book §11.4, pp. 261–265).
//
// What it models:
//   - shuffle-up with the documented out-of-range rule: a lane whose source would be below
//     the start of its group of `width` lanes gets its own value back (CUDA: "the lower
//     delta lanes will remain unchanged", infra-facts A3; HIP: "the thread returns its own
//     var", B3; Metal simd_shuffle_up: "doesn't modify the lower delta lanes", C2);
//   - Fig. 11.8's loop with its participation test written three ways (laneIdx(), the book;
//     threadIdx.x; none), a loop bound and a lane-counting width that may differ from the
//     hardware width (a 64-wide AMD wavefront running code written for 32: ch11-facts B1);
//   - Fig. 11.9: warp scans, the last lane's total into shared memory, one warp scanning the
//     totals, every later warp adding its predecessor's scanned total; its two barriers and
//     its limit of one warp of totals (warps per block ≤ warp width: B1, C2);
//   - the scan-scan-add stages for any segment size (Figs. 11.6, 11.7, 11.11, 11.14);
//   - which lines of labs/ch11/snippets/s114_blockscan_as_printed.cu need a change or a check
//     on each target (PORT_ISSUES; warpscan.test.ts ties the line numbers to the file).
//
// What it leaves out: time. A step is one shuffle and one add for every lane at once (the
// lanes move in lockstep); nothing here knows how long a shuffle or a barrier takes.
// Values are small integers, so float rounding never shows.
//
// Used by section 11.4. It does not import chapter 10's shuffle.ts on purpose: chapter 10
// models shuffle-down reductions, this file shuffle-up scans.

import { tr } from '../lib/locale.ts';

export const FIG_11_6 = [2, 1, 3, 1, 0, 4, 1, 2, 0, 3, 1, 2, 5, 3, 1, 2];

/** Inclusive prefix sum, left to right (Fig. 11.1). */
export function inclusiveScan(v: readonly number[]): number[] {
  const out: number[] = [];
  let run = 0;
  v.forEach((x, i) => {
    run = i === 0 ? x : run + x;
    out.push(run);
  });
  return out;
}

/** Inclusive scan restarted at every multiple of `group`. */
export function groupedScan(v: readonly number[], group: number): number[] {
  const out: number[] = [];
  let run = 0;
  v.forEach((x, i) => {
    run = i % group === 0 ? x : run + x;
    out.push(run);
  });
  return out;
}

/** The values the emulator check gives thread t: (5t mod 7) + 1 (labs/ch11/snippets/check/s114_check.cu). */
export const laneValue = (t: number) => ((5 * t) % 7) + 1;
export const laneValues = (n: number, from = 0) => Array.from({ length: n }, (_, i) => laneValue(from + i));

// ---------------------------------------------------------------- scan, scan, add

export interface ScanScanAdd {
  segSize: number;
  input: number[];
  /** Stage 1: every segment scanned on its own (flattened). */
  stage1: number[];
  /** Each segment's total: the last value of its stage-1 scan. */
  totals: number[];
  /** Stage 2: the totals scanned. */
  stage2: number[];
  /** What stage 3 adds to each segment: 0 for the first, stage2[k − 1] for segment k. */
  added: number[];
  /** Stage 3: the scanned input. */
  final: number[];
  /** Additions in each stage: (segSize − 1) per segment; a scan of the totals; one per element of every segment but the first. */
  additions: { stage1: number; stage2: number; stage3: number };
}

export function scanScanAdd(input: readonly number[], segSize: number): ScanScanAdd {
  if (segSize < 1 || input.length % segSize !== 0) throw new Error('the input must be a whole number of segments');
  const segs = input.length / segSize;
  const stage1 = groupedScan(input, segSize);
  const totals = Array.from({ length: segs }, (_, k) => stage1[k * segSize + segSize - 1]);
  const stage2 = inclusiveScan(totals);
  const added = Array.from({ length: segs }, (_, k) => (k === 0 ? 0 : stage2[k - 1]));
  const final = stage1.map((v, i) => v + added[Math.floor(i / segSize)]);
  return {
    segSize,
    input: [...input],
    stage1,
    totals,
    stage2,
    added,
    final,
    additions: { stage1: segs * (segSize - 1), stage2: Math.max(0, segs - 1), stage3: (segs - 1) * segSize },
  };
}

// ---------------------------------------------------------------- shuffle up

export interface ShflLane {
  lane: number;
  /** The lane whose value was received, or null when the lane got its own value back. */
  src: number | null;
  value: number;
}

/**
 * One shuffle-up across a warp: lane l receives the value of lane l − delta, provided that
 * lane is in the same group of `width` lanes (width defaults to the whole warp, as warpSize
 * does). Otherwise it gets its own value back.
 */
export function shuffleUp(values: readonly number[], delta: number, width = values.length): ShflLane[] {
  return values.map((v, lane) => {
    const pos = lane % width;
    return pos >= delta ? { lane, src: lane - delta, value: values[lane - delta] } : { lane, src: null, value: v };
  });
}

// ---------------------------------------------------------------- Fig. 11.8

export type LaneTest = 'laneIdx' | 'threadIdx' | 'none';
export const LANE_TEST_CODE: Record<LaneTest, string> = {
  laneIdx: 'if(laneIdx() >= stride)',
  threadIdx: 'if(threadIdx.x >= stride)',
  get none() {
    return tr('(no test)', '(không kiểm tra)');
  },
};

export interface WarpScanOptions {
  /** The loop runs while stride < loopBound. The book's is WARP_SIZE, 32. Default: the hardware width. */
  loopBound?: number;
  /** How the code counts lanes: laneIdx() = threadIdx.x mod laneWidth. Default: the hardware width. */
  laneWidth?: number;
  test?: LaneTest;
  /** This warp's index in its block; threadIdx.x = warp·(hardware width) + lane. */
  warp?: number;
  /** The shuffle's width argument. Default: warpSize, the hardware width. */
  shuffleWidth?: number;
}

export interface LaneStep {
  lane: number;
  /** threadIdx.x of this lane. */
  thread: number;
  before: number;
  received: number;
  /** Where the received value came from; null: the lane's own value came back. */
  src: number | null;
  adds: boolean;
  after: number;
  /** It received its own value and added it: it doubled itself. */
  doubled: boolean;
}

export interface WarpScanStep {
  stride: number;
  lanes: LaneStep[];
}

export interface WarpScanTrace {
  /** Hardware width: lanes in the warp. */
  hw: number;
  loopBound: number;
  laneWidth: number;
  test: LaneTest;
  warp: number;
  initial: number[];
  steps: WarpScanStep[];
  final: number[];
  /** The right answer: an inclusive scan within each group of laneWidth lanes, the code's idea of a warp. */
  expected: number[];
  /** Lanes whose final value is not the expected one. */
  wrong: number[];
  /** Lanes that doubled themselves at least once. */
  doubled: number[];
  /** Additions performed. */
  additions: number;
}

export function warpScanTrace(values: readonly number[], o: WarpScanOptions = {}): WarpScanTrace {
  const hw = values.length;
  const loopBound = o.loopBound ?? hw;
  const laneWidth = o.laneWidth ?? hw;
  const test = o.test ?? 'laneIdx';
  const warp = o.warp ?? 0;
  const width = o.shuffleWidth ?? hw;
  let cur = [...values];
  const steps: WarpScanStep[] = [];
  let additions = 0;
  for (let stride = 1; stride < loopBound; stride *= 2) {
    const got = shuffleUp(cur, stride, width);
    const lanes = cur.map((before, lane) => {
      const thread = warp * hw + lane;
      const passes = test === 'none' ? true : test === 'threadIdx' ? thread >= stride : thread % laneWidth >= stride;
      const r = got[lane];
      const after = passes ? before + r.value : before;
      if (passes) additions++;
      return { lane, thread, before, received: r.value, src: r.src, adds: passes, after, doubled: passes && r.src === null };
    });
    steps.push({ stride, lanes });
    cur = lanes.map((l) => l.after);
  }
  const expected = groupedScan(values, laneWidth);
  const wrong = cur.flatMap((v, l) => (v === expected[l] ? [] : [l]));
  const doubled = [...new Set(steps.flatMap((s) => s.lanes.filter((l) => l.doubled).map((l) => l.lane)))].sort((a, b) => a - b);
  return { hw, loopBound, laneWidth, test, warp, initial: [...values], steps, final: cur, expected, wrong, doubled, additions };
}

/**
 * The three ways a port to a 64-wide wavefront can count (ch11-facts B1): the loop bound and
 * the lane numbering both 32 (each half scans as its own warp: right, half the reach), the
 * bound 32 but lanes counted in 64s (lanes 32 to 63 wrong), both 64 (right, full reach).
 */
export type HalfCase = 'both32' | 'mixed' | 'both64';
export const HALF_CASES: { key: HalfCase; loopBound: number; laneWidth: number; label: string; code: string }[] = [
  {
    key: 'both32', loopBound: 32, laneWidth: 32,
    get label() {
      return tr('loop bound 32, lanes counted in 32s', 'cận vòng lặp 32, lane đếm theo nhóm 32');
    },
    code: 'stride < 32; laneIdx() = threadIdx.x % 32',
  },
  {
    key: 'mixed', loopBound: 32, laneWidth: 64,
    get label() {
      return tr('loop bound 32, lanes counted from warpSize (64)', 'cận vòng lặp 32, lane đếm theo warpSize (64)');
    },
    code: 'stride < 32; laneIdx() = threadIdx.x % warpSize',
  },
  {
    key: 'both64', loopBound: 64, laneWidth: 64,
    get label() {
      return tr('warpSize (64) for both', 'warpSize (64) cho cả hai');
    },
    code: 'stride < warpSize; laneIdx() = threadIdx.x % warpSize',
  },
];
export function halfCase(values: readonly number[], key: HalfCase): WarpScanTrace {
  const c = HALF_CASES.find((x) => x.key === key)!;
  return warpScanTrace(values, { loopBound: c.loopBound, laneWidth: c.laneWidth });
}

// ---------------------------------------------------------------- Fig. 11.9

export interface BlockScanResult {
  threads: number;
  width: number;
  numWarps: number;
  /** One warp scans every warp's total: numWarps ≤ width, i.e. width·width ≥ threads. */
  fits: boolean;
  /** After line 03: each warp scanned on its own. */
  afterWarpScan: number[];
  /** warpSums_s after line 07: each warp's total. */
  warpSums: number[];
  /** warpSums_s after line 15: the totals warp 0 managed to scan; entries past its width keep the raw totals. */
  scannedSums: number[];
  final: number[];
  expected: number[];
  wrong: number[];
  barriers: number;
}

/** Fig. 11.9 on one block whose size is a multiple of the warp width. */
export function blockScanFig119(values: readonly number[], width: number): BlockScanResult {
  const threads = values.length;
  if (threads % width !== 0) throw new Error('the block must be a whole number of warps');
  const numWarps = threads / width;
  const afterWarpScan: number[] = [];
  for (let w = 0; w < numWarps; w++) afterWarpScan.push(...warpScanTrace(values.slice(w * width, (w + 1) * width)).final);
  const warpSums = Array.from({ length: numWarps }, (_, w) => afterWarpScan[w * width + width - 1]);
  // Warp 0: lane l loads warpSums_s[l] if l < NUM_WARPS, else 0; scans; writes back where l < NUM_WARPS.
  const loaded = Array.from({ length: width }, (_, l) => (l < numWarps ? warpSums[l] : 0));
  const scanned = warpScanTrace(loaded).final;
  const scannedSums = warpSums.map((s, w) => (w < width ? scanned[w] : s));
  const final = afterWarpScan.map((v, t) => {
    const w = Math.floor(t / width);
    return w > 0 ? v + scannedSums[w - 1] : v;
  });
  const expected = inclusiveScan(values);
  const wrong = final.flatMap((v, t) => (v === expected[t] ? [] : [t]));
  return { threads, width, numWarps, fits: numWarps <= width, afterWarpScan, warpSums, scannedSums, final, expected, wrong, barriers: 2 };
}

/** The condition Fig. 11.9 needs, as a Metal host or kernel check would write it (ch11-facts C2). */
export const oneWarpOfSums = (threads: number, width: number) => width * width >= threads;

export type BlockDesign = 'fig11.3' | 'fig11.5' | 'fig11.9';
/** Barriers one block executes for one block scan of `threads` elements, one per thread. */
export function barriersPerBlockScan(design: BlockDesign, threads: number): number {
  let iters = 0;
  for (let stride = 1; stride < threads; stride *= 2) iters++;
  return design === 'fig11.3' ? 2 * iters : design === 'fig11.5' ? iters : 2;
}

// ---------------------------------------------------------------- masks per backend

export type Backend = 'cuda' | 'hip' | 'metal';
export interface MaskVerdict {
  backend: Backend;
  compiles: boolean;
  maskType: string;
  why: string;
  facts: string;
}
/** Does `__shfl_up_sync(0xffffffff, ...)` (literal) or a typed all-ones mask compile? */
export function maskVerdict(backend: Backend, mask: 'literal' | 'typed'): MaskVerdict {
  if (backend === 'cuda')
    return {
      backend,
      compiles: true,
      maskType: tr('unsigned (32 bits)', 'unsigned (32 bit)'),
      why: tr(
        'The mask parameter is an unsigned; 0xffffffff names all 32 lanes.',
        'Tham số mask là một unsigned; 0xffffffff chỉ cả 32 lane.',
      ),
      facts: 'infra-facts A3',
    };
  if (backend === 'hip')
    return mask === 'literal'
      ? {
          backend,
          compiles: false,
          maskType: tr('a 64-bit integer', 'một số nguyên 64-bit'),
          why: tr(
            'Every HIP _sync intrinsic has a static_assert that the mask is a 64-bit integer, on every AMD target, wave32 included; the literal is a 32-bit unsigned.',
            'Mọi intrinsic _sync của HIP đều có một static_assert rằng mask là một số nguyên 64-bit, trên mọi target AMD, kể cả wave32; literal là một unsigned 32-bit.',
          ),
          facts: 'infra-facts B3; ch11-facts B1',
        }
      : {
          backend,
          compiles: true,
          maskType: 'unsigned long long', // i18n-ignore: C type name
          why: tr(
            'A mask type chosen per vendor (unsigned on CUDA, unsigned long long on HIP) passes; in wave32 mode the upper half is ignored.',
            'Một kiểu mask chọn theo vendor (unsigned trên CUDA, unsigned long long trên HIP) thì qua được; ở chế độ wave32 nửa trên bị bỏ qua.',
          ),
          facts: 'infra-facts B3',
        };
  return {
    backend,
    compiles: true,
    maskType: 'none',
    why: tr(
      'simd_shuffle_up(data, delta) has no mask and no width: it acts on the active threads.',
      'simd_shuffle_up(data, delta) không có mask và không có width: nó tác động lên các luồng đang hoạt động.',
    ),
    facts: 'infra-facts C2; ch11-facts C1',
  };
}

// ---------------------------------------------------------------- Port it

export type PortTarget = 'hip64' | 'hip32' | 'metal';
export const PORT_TARGETS: { key: PortTarget; label: string }[] = [
  {
    key: 'hip64',
    get label() {
      return tr('HIP on a 64-wide wavefront (gfx9)', 'HIP trên một wavefront rộng 64 (gfx9)');
    },
  },
  {
    key: 'hip32',
    get label() {
      return tr('HIP on a 32-wide wavefront (gfx10 and later)', 'HIP trên một wavefront rộng 32 (gfx10 trở về sau)');
    },
  },
  { key: 'metal', label: 'Metal (MSL)' }, // i18n-ignore: backend name
];

export type IssueKind = 'compile error' | 'wrong or limited' | 'undocumented' | 'limit' | 'rewrite';

export interface PortIssue {
  id: string;
  /** 1-based line in labs/ch11/snippets/s114_blockscan_as_printed.cu. */
  line: number;
  /** Other lines of the same issue: clicking any of them counts. */
  also?: number[];
  /** Text that line contains (the test checks it). */
  text: string;
  /** Where the book prints it. */
  where: string;
  /** Per target: why the line needs a change or a check there. Absent: fine on that target. */
  on: Partial<Record<PortTarget, { kind: IssueKind; says: string; facts: string }>>;
}

export const PRINTED_FILE = 'ch11/snippets/s114_blockscan_as_printed.cu';

export const PORT_ISSUES: PortIssue[] = [
  {
    id: 'warp-size',
    line: 7,
    text: '#define WARP_SIZE 32',
    get where() {
      return tr('chapter 10, used by Figs. 11.8 and 11.9', 'chương 10, được Fig. 11.8 và 11.9 dùng');
    },
    on: {
      hip64: {
        kind: 'wrong or limited',
        get says() {
          return tr(
            'Right as printed, but only by accident: laneIdx(), warpIdx(), NUM_WARPS and the loop all count in 32s, so each half of the wavefront scans as its own warp, with half the reach. Change it to warpSize in one place and not the other and lanes 32 to 63 go wrong (below). Use warpSize everywhere, or a compile-time width per architecture.',
            'Đúng như in, nhưng chỉ là tình cờ: laneIdx(), warpIdx(), NUM_WARPS và vòng lặp đều đếm theo nhóm 32, nên mỗi nửa wavefront scan như một warp riêng, với một nửa tầm với. Đổi nó thành warpSize ở chỗ này mà không đổi ở chỗ kia thì các lane 32 đến 63 sai (bên dưới). Hãy dùng warpSize ở mọi chỗ, hoặc một độ rộng lúc biên dịch cho từng kiến trúc.',
          );
        },
        facts: 'ch11-facts B1; ch10-facts B2',
      },
      metal: {
        kind: 'undocumented',
        get says() {
          return tr(
            'Apple documents no SIMD-group width. Read [[threads_per_simdgroup]] in the kernel instead of assuming 32 (the projects read all hard-code 32).',
            'Apple không ghi trong tài liệu độ rộng SIMD-group nào. Hãy đọc [[threads_per_simdgroup]] trong kernel thay vì giả định 32 (các dự án chúng tôi đã đọc đều viết cứng 32).',
          );
        },
        facts: 'ch11-facts C2; infra-facts C5',
      },
    },
  },
  {
    id: 'helpers',
    line: 13,
    also: [14],
    text: '__device__ unsigned int warpIdx() { return threadIdx.x/WARP_SIZE; }', // i18n-ignore: CUDA code
    get where() {
      return tr('chapter 10, pp. 238–239', 'chương 10, pp. 238–239');
    },
    on: {
      metal: {
        kind: 'rewrite',
        get says() {
          return tr(
            'Lane and SIMD-group numbers come from the kernel attributes [[thread_index_in_simdgroup]] and [[simdgroup_index_in_threadgroup]], not from a division of the thread index.',
            'Số lane và số SIMD-group lấy từ các thuộc tính kernel [[thread_index_in_simdgroup]] và [[simdgroup_index_in_threadgroup]], không phải từ phép chia chỉ số luồng.',
          );
        },
        facts: 'ch11-facts C2',
      },
    },
  },
  {
    id: 'mask',
    line: 20,
    text: '__shfl_up_sync(0xffffffff, val, stride)',
    get where() {
      return tr('Fig. 11.8, line 03', 'Fig. 11.8, dòng 03');
    },
    on: {
      hip64: {
        kind: 'compile error',
        get says() {
          return tr(
            'HIP static_asserts a 64-bit mask on every _sync intrinsic; the 32-bit literal does not compile. Give the mask its own type, unsigned on CUDA and unsigned long long on HIP (lab 10.1, llama.cpp).',
            'HIP static_assert một mask 64-bit trên mọi intrinsic _sync; literal 32-bit không biên dịch được. Hãy cho mask kiểu riêng, unsigned trên CUDA và unsigned long long trên HIP (lab 10.1, llama.cpp).',
          );
        },
        facts: 'infra-facts B3; ch11-facts B1',
      },
      hip32: {
        kind: 'compile error',
        get says() {
          return tr(
            'The static_assert applies in wave32 mode too: the mask must be a 64-bit integer, whose upper half is then ignored.',
            'static_assert áp dụng cả ở chế độ wave32: mask phải là một số nguyên 64-bit, khi đó nửa trên của nó bị bỏ qua.',
          );
        },
        facts: 'infra-facts B3',
      },
      metal: {
        kind: 'rewrite',
        get says() {
          return tr(
            'simd_shuffle_up(val, stride): no mask, no width. Lanes below the stride keep their own value, as on CUDA, so the test on the next line stays. Or replace the whole loop with simd_prefix_inclusive_sum(val).',
            'simd_shuffle_up(val, stride): không mask, không width. Các lane dưới stride giữ giá trị của chính chúng, như trên CUDA, nên phép kiểm tra ở dòng tiếp theo vẫn được giữ nguyên. Hoặc thay cả vòng lặp bằng simd_prefix_inclusive_sum(val).',
          );
        },
        facts: 'ch11-facts C1',
      },
    },
  },
  {
    id: 'shared-in-helper',
    line: 34,
    text: '__shared__ float warpSums_s[NUM_WARPS];', // i18n-ignore: CUDA code
    get where() {
      return tr('Fig. 11.9, line 05', 'Fig. 11.9, dòng 05');
    },
    on: {
      hip64: {
        kind: 'undocumented',
        get says() {
          return tr(
            'AMD documents shared memory "allocated statically within the kernel" and says nothing about a __shared__ array inside a __device__ function. Declare it in the kernel and pass a pointer.',
            'AMD ghi trong tài liệu bộ nhớ chia sẻ "allocated statically within the kernel" và không nói gì về một mảng __shared__ bên trong một hàm __device__. Hãy khai báo nó trong kernel và truyền một con trỏ.',
          );
        },
        facts: 'ch11-facts B1',
      },
      hip32: {
        kind: 'undocumented',
        get says() {
          return tr(
            'Same on every AMD target: not documented. Declaring the array in the kernel works on all three backends.',
            'Giống nhau trên mọi target AMD: không có trong tài liệu. Khai báo mảng trong kernel chạy được trên cả ba backend.',
          );
        },
        facts: 'ch11-facts B1',
      },
      metal: {
        kind: 'compile error',
        get says() {
          return tr(
            'MSL rejects it: "variables in the threadgroup address space cannot be declared in a non-qualified function". Declare threadgroup float warpSums[...] in the kernel and pass a threadgroup float* to the helper.',
            'MSL từ chối nó: "variables in the threadgroup address space cannot be declared in a non-qualified function". Hãy khai báo threadgroup float warpSums[...] trong kernel và truyền một threadgroup float* cho hàm phụ.',
          );
        },
        facts: 'ch11-facts C2',
      },
    },
  },
  {
    id: 'one-warp-of-sums',
    line: 40,
    also: [41, 42, 43, 44],
    text: 'if(warpIdx() == 0) {',
    get where() {
      return tr('Fig. 11.9, lines 11 to 17', 'Fig. 11.9, dòng 11 đến 17');
    },
    on: {
      metal: {
        kind: 'limit',
        get says() {
          return tr(
            'One SIMD-group scans all the SIMD-group totals, so it needs (threads per threadgroup) ÷ width ≤ width. At 1,024 threads that holds for width 32 and fails for width 16, and Apple documents neither. Refuse configurations where width × width < threads, or loop.',
            'Một SIMD-group scan mọi tổng của các SIMD-group, nên nó cần (số luồng mỗi threadgroup) ÷ width ≤ width. Ở 1,024 luồng điều đó đúng với width 32 và sai với width 16, và Apple không ghi trong tài liệu độ rộng nào trong hai độ rộng đó. Hãy từ chối các cấu hình có width × width < số luồng, hoặc dùng vòng lặp.',
          );
        },
        facts: 'ch11-facts C2',
      },
    },
  },
];

/** The lines that need a change or a check on a target, in file order. */
export function portIssues(target: PortTarget): (PortIssue & { kind: IssueKind; says: string; facts: string })[] {
  return PORT_ISSUES.filter((p) => p.on[target]).map((p) => ({ ...p, ...p.on[target]! }));
}

/** Every line that belongs to an issue on this target. */
export const issueLines = (p: PortIssue) => [p.line, ...(p.also ?? [])];

/**
 * Score a set of clicked lines against a target's issues: an issue is found when any of its
 * lines was clicked; a clicked line that belongs to no issue is extra.
 */
export function scorePort(target: PortTarget, clicked: readonly number[]): { found: string[]; missed: string[]; extra: number[] } {
  const issues = portIssues(target);
  const got = new Set(clicked);
  const covered = new Set(issues.flatMap(issueLines));
  return {
    found: issues.filter((p) => issueLines(p).some((l) => got.has(l))).map((p) => p.id),
    missed: issues.filter((p) => !issueLines(p).some((l) => got.has(l))).map((p) => p.id),
    extra: [...got].filter((l) => !covered.has(l)).sort((a, b) => a - b),
  };
}
