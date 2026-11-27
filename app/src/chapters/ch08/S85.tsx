import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Bench, Beyond, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { benchRows, fmt, num, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { Translated, tr, useLocale } from '../../lib/i18n';
import {
  BOOK_BLOCK_LIMIT,
  blocksSlab,
  cubeAI,
  extraRegsPerBlock,
  lifetimeCounts,
  planesLoaded,
  sharedBytes,
  slabAI,
  slabAILimit,
  threadsSlab,
  windowRegs,
  type SlabKernel,
} from '../../sim/stencilslab';
import {
  COEFFS,
  CORRECT,
  SCHEDULE_LABEL,
  depthOf,
  gridBlocks,
  idx3,
  interiorBlock,
  isShared,
  makeProblem,
  outWidth,
  position,
  runAll,
  runBlock,
  stateAt,
  type Kernel,
  type Pairing,
  type Schedule,
  type SharedBuf,
  type Slot,
  type ZConfig,
  type ZStep,
} from '../../sim/zmarch';
import { ZMarchScene } from './ZMarchScene';

// ---------------------------------------------------------------- code shown by the stepper
// The two kernels as the stepper shows them: labs/ch08/snippets/zmarch_planes.cu and
// zmarch_registers.cu, which pass the CUDA front-end check. S85.test.ts fails if these
// copies drift from the files. The model (sim/zmarch.ts) runs exactly these statements.
const PLANES_BODY = `// Fig. 8.10 (p. 194), corrected: c3 goes with y-1 and c4 with y+1,
// as in Figs. 8.6 and 8.8. ty and tx stand for threadIdx.y and
// threadIdx.x; the stepper uses tiles of 4 and 6.
#define IN_TILE_DIM 6
#define OUT_TILE_DIM (IN_TILE_DIM - 2)
__constant__ float c0, c1, c2, c3, c4, c5, c6;

__global__ void stencil_planes(const float* in, float* out,
                               unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
    int ty = threadIdx.y, tx = threadIdx.x;
    __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];
    bool jk = j >= 0 && j < N && k >= 0 && k < N;
    if (jk && iStart >= 1 && iStart - 1 < N)
        inPrev_s[ty][tx] = in[(iStart - 1)*N*N + j*N + k];
    if (jk && iStart < N)
        inCurr_s[ty][tx] = in[iStart*N*N + j*N + k];
    for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        if (jk && i + 1 < N)
            inNext_s[ty][tx] = in[(i + 1)*N*N + j*N + k];
        __syncthreads();   // barrier 1
        bool inGrid = i >= 1 && i < N - 1 && j >= 1 && j < N - 1
                   && k >= 1 && k < N - 1;
        bool inner = ty >= 1 && ty < IN_TILE_DIM - 1
                  && tx >= 1 && tx < IN_TILE_DIM - 1;
        if (inGrid && inner)
            out[i*N*N + j*N + k] = c0*inCurr_s[ty][tx]
              + c1*inCurr_s[ty][tx - 1] + c2*inCurr_s[ty][tx + 1]
              + c3*inCurr_s[ty - 1][tx] + c4*inCurr_s[ty + 1][tx]
              + c5*inPrev_s[ty][tx] + c6*inNext_s[ty][tx];
        __syncthreads();   // barrier 2
        inPrev_s[ty][tx] = inCurr_s[ty][tx];
        inCurr_s[ty][tx] = inNext_s[ty][tx];
    }
}`;

const REGISTERS_BODY = `// Fig. 8.12 (p. 197), corrected: the register inNext where the page
// prints inNext_s, c3 with y-1 and c4 with y+1, and registers that
// start at 0 (the page leaves them unset). ty and tx stand for
// threadIdx.y and threadIdx.x; the stepper uses tiles of 4 and 6.
#define IN_TILE_DIM 6
#define OUT_TILE_DIM (IN_TILE_DIM - 2)
__constant__ float c0, c1, c2, c3, c4, c5, c6;

__global__ void stencil_registers(const float* in, float* out,
                                  unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
    int ty = threadIdx.y, tx = threadIdx.x;
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    float inPrev = 0.0f, inCurr = 0.0f, inNext = 0.0f;
    bool jk = j >= 0 && j < N && k >= 0 && k < N;
    if (jk && iStart >= 1 && iStart - 1 < N)
        inPrev = in[(iStart - 1)*N*N + j*N + k];
    if (jk && iStart < N) {
        inCurr = in[iStart*N*N + j*N + k];
        inCurr_s[ty][tx] = inCurr;
    }
    for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        if (jk && i + 1 < N)
            inNext = in[(i + 1)*N*N + j*N + k];
        __syncthreads();   // barrier 1
        bool inGrid = i >= 1 && i < N - 1 && j >= 1 && j < N - 1
                   && k >= 1 && k < N - 1;
        bool inner = ty >= 1 && ty < IN_TILE_DIM - 1
                  && tx >= 1 && tx < IN_TILE_DIM - 1;
        if (inGrid && inner)
            out[i*N*N + j*N + k] = c0*inCurr
              + c1*inCurr_s[ty][tx - 1] + c2*inCurr_s[ty][tx + 1]
              + c3*inCurr_s[ty - 1][tx] + c4*inCurr_s[ty + 1][tx]
              + c5*inPrev + c6*inNext;
        __syncthreads();   // barrier 2
        inPrev = inCurr;
        inCurr = inNext;
        inCurr_s[ty][tx] = inNext;
    }
}`;

// labs/ch08/snippets/rolling_window_indexed.cu, for "Keep the window in registers".
const INDEXED_WINDOW = `// The register-tiled march of Fig. 8.12 with its z window written as a small array
// indexed at run time: w[z % 3] holds plane z, so no values are copied between
// registers. Same arithmetic as the book's kernel. But the indices change with i, and
// NVIDIA documents an array the compiler cannot see is indexed with constants as a
// candidate for local memory, which is device memory (notes/today/ch08-facts.md, A5).
// Ours, not the book's. It parses with the CUDA front end; it is never built or run.
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2)
__constant__ float c0, c1, c2, c3, c4, c5, c6;

__global__ void stencil_indexed_window(const float* in, float* out, unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
    int ty = threadIdx.y, tx = threadIdx.x;
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    float w[3] = {0.0f, 0.0f, 0.0f};                       // w[z % 3] holds plane z
    bool jk = j >= 0 && j < N && k >= 0 && k < N;
    if (jk && iStart >= 1 && iStart - 1 < N) w[(iStart + 2) % 3] = in[(iStart - 1)*N*N + j*N + k];
    if (jk && iStart < N) { w[iStart % 3] = in[iStart*N*N + j*N + k]; inCurr_s[ty][tx] = w[iStart % 3]; }
    for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        if (jk && i + 1 < N) w[(i + 1) % 3] = in[(i + 1)*N*N + j*N + k];   // overwrites plane i - 2
        __syncthreads();
        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1 &&
            ty >= 1 && ty < IN_TILE_DIM - 1 && tx >= 1 && tx < IN_TILE_DIM - 1)
            out[i*N*N + j*N + k] = c0*w[i % 3]
                + c1*inCurr_s[ty][tx - 1] + c2*inCurr_s[ty][tx + 1]
                + c3*inCurr_s[ty - 1][tx] + c4*inCurr_s[ty + 1][tx]
                + c5*w[(i + 2) % 3] + c6*w[(i + 1) % 3];
        __syncthreads();
        inCurr_s[ty][tx] = w[(i + 1) % 3];
    }
}`;

const SNIPPET_STATUS = 'front-end-checked; never run (the stepper runs a tested model of it)'; // i18n-ignore: translated where used
const STENCIL_TIERS: Record<string, Tier> = {
  in: 'global', out: 'global',
  inPrev_s: 'shared', inCurr_s: 'shared', inNext_s: 'shared',
  inPrev: 'reg', inCurr: 'reg', inNext: 'reg', w: 'reg',
  i: 'reg', j: 'reg', k: 'reg', iStart: 'reg', ty: 'reg', tx: 'reg', jk: 'reg', n: 'reg',
  c0: 'const', c1: 'const', c2: 'const', c3: 'const', c4: 'const', c5: 'const', c6: 'const',
};
const LAB_TIERS: Record<string, Tier> = {
  in: 'global', out: 'global',
  inPrev_s: 'shared', inCurr_s: 'shared', inNext_s: 'shared',
  inPrev: 'reg', inCurr: 'reg', inNext: 'reg', i: 'reg', j: 'reg', k: 'reg', iStart: 'reg', n: 'reg',
};

const PRINTED_PAIR = '+ c3*inCurr_s[ty - 1][tx] + c4*inCurr_s[ty + 1][tx]'; // i18n-ignore: code
const SWAPPED_PAIR = '+ c3*inCurr_s[ty + 1][tx] + c4*inCurr_s[ty - 1][tx]   // as printed'; // i18n-ignore: code
// The snippets define the tile as 6; the stepper shows the width of the problem it runs.
// S85.test.ts checks that this line occurs once in each snippet.
const TILE_DEFINE = '#define IN_TILE_DIM 6';
const codeFor = (kernel: Kernel, pairing: Pairing, tile: number) => {
  const body = (kernel === 'planes' ? PLANES_BODY : REGISTERS_BODY).replace(TILE_DEFINE, `#define IN_TILE_DIM ${tile}`);
  return pairing === 'book-8.6' ? body : body.replace(PRINTED_PAIR, SWAPPED_PAIR);
};

/** 1-based lines of the shown code that a step executes. */
function linesFor(code: string, s: ZStep | undefined): number[] {
  if (!s) return [];
  const lines = code.split('\n');
  const find = (needle: string) => lines.findIndex((l) => l.includes(needle)) + 1;
  const run = (needle: string, n: number) => {
    const a = find(needle);
    return a > 0 ? Array.from({ length: n }, (_, x) => a + x) : [];
  };
  const P = !lines.some((l) => l.includes('float inPrev = 0.0f'));
  // A load is a guard line and an assignment below it; a skipped load runs only the guard.
  const load = (needle: string, extra = 0) => {
    const a = find(needle);
    if (a <= 0) return [];
    return s.global?.access === 'skipped' ? [a - 1] : Array.from({ length: 2 + extra }, (_, x) => a - 1 + x);
  };
  switch (s.op) {
    case 'loadPrev': return load(P ? 'inPrev_s[ty][tx] = in[' : 'inPrev = in[');
    case 'loadCurr': return P ? load('inCurr_s[ty][tx] = in[') : load('inCurr = in[', 1);
    case 'loadNext': return load(P ? 'inNext_s[ty][tx] = in[' : 'inNext = in[');
    case 'barrier1': case 'release1': return [find('// barrier 1')];
    case 'barrier2': case 'release2': return [find('// barrier 2')];
    case 'compute': return s.active ? run('if (inGrid && inner)', 5) : run('bool inGrid', 5); // i18n-ignore: code
    case 'shiftPrev': return [find('inPrev_s[ty][tx] = inCurr_s[ty][tx]')];
    case 'shiftCurr': return [find('inCurr_s[ty][tx] = inNext_s[ty][tx]')];
    case 'rotate': return [find('inPrev = inCurr;'), find('inCurr = inNext;')];
    case 'storeCurr': return [find('inCurr_s[ty][tx] = inNext;')];
  }
}

// ---------------------------------------------------------------- the stepper
const PROBLEMS = [
  {
    get label() {
      return tr('tile 4 on a 6 × 6 × 6 grid', 'ô 4 trên lưới 6 × 6 × 6');
    },
    N: 6,
    inTile: 4,
  },
  {
    get label() {
      return tr("tile 6 on a 10 × 10 × 10 grid (the book's picture)", 'ô 6 trên lưới 10 × 10 × 10 (hình của sách)');
    },
    N: 10,
    inTile: 6,
  },
];
const SCHEDULES: Schedule[] = ['lockstep', 'in-order', 'reversed', 'shuffled'];
const KERNEL_LABEL: Record<Kernel, string> = {
  get planes() {
    return tr('three shared planes (Fig. 8.10)', 'ba mặt phẳng chia sẻ (Hình 8.10)');
  },
  get registers() {
    return tr('registers for z (Fig. 8.12)', 'thanh ghi cho z (Hình 8.12)');
  },
};
const PAIRING_LABEL: Record<Pairing, string> = {
  get 'book-8.6'() {
    return tr('c3 with y − 1, as Figs. 8.6 and 8.8', 'c3 với y − 1, như Hình 8.6 và 8.8');
  },
  get 'as-printed-8.10'() {
    return tr('c3 with y + 1, as printed in Figs. 8.10 and 8.12', 'c3 với y + 1, như in trong Hình 8.10 và 8.12');
  },
};
const show = (v: number) => (Number.isNaN(v) ? '?' : String(v));

/** One sentence (or two) saying what a step did. */
function describe(s: ZStep, cfg: ZConfig): ReactNode {
  if (s.thread < 0)
    return s.op === 'release1'
      ? tr(
          `Barrier 1 opens: every thread has loaded its point of plane ${s.i + 1}. Reading neighbours' points of the current plane is safe now.`,
          `Rào chắn 1 mở: mọi luồng đã nạp điểm của nó trong mặt phẳng ${s.i + 1}. Giờ đọc các điểm của láng giềng trong mặt phẳng hiện tại là an toàn.`,
        )
      : tr(
          `Barrier 2 opens: every thread has finished reading for output plane ${s.i}. Overwriting the window is safe now.`,
          `Rào chắn 2 mở: mọi luồng đã đọc xong cho mặt phẳng đầu ra ${s.i}. Giờ ghi đè cửa sổ là an toàn.`,
        );
  const who = tr(`Thread (tx ${s.tx}, ty ${s.ty})`, `Luồng (tx ${s.tx}, ty ${s.ty})`);
  const g = s.global;
  const dest = (s.writes ?? []).map((w) => (isShared(w.buf) ? `${w.buf}[${w.r}][${w.c}]` : tr(`register ${w.buf}`, `thanh ghi ${w.buf}`))).join(tr(' and ', ' và '));
  switch (s.op) {
    case 'loadPrev':
    case 'loadCurr':
    case 'loadNext': {
      if (g!.access === 'skipped') {
        const why =
          g!.z < 0 || g!.z >= cfg.N
            ? tr(`plane ${g!.z} is outside the grid`, `mặt phẳng ${g!.z} nằm ngoài lưới`)
            : tr(`its point (y ${g!.j}, x ${g!.k}) is outside the grid`, `điểm (y ${g!.j}, x ${g!.k}) của nó nằm ngoài lưới`);
        return tr(
          `${who}: ${why}, so the guard skips this load and the slot keeps whatever it held.`,
          `${who}: ${why}, nên điều kiện chặn bỏ qua lần nạp này và ô nhớ giữ nguyên giá trị nó đang có.`,
        );
      }
      const role =
        s.op === 'loadPrev'
          ? tr('the plane before the first output plane', 'mặt phẳng trước mặt phẳng đầu ra đầu tiên')
          : s.op === 'loadCurr'
            ? tr('the first output plane', 'mặt phẳng đầu ra đầu tiên')
            : tr('the next plane', 'mặt phẳng kế tiếp');
      return tr(
        `${who} loads in at (z ${g!.z}, y ${g!.j}, x ${g!.k}) = ${s.writes![0].value} into ${dest}: ${role}.`,
        `${who} nạp in tại (z ${g!.z}, y ${g!.j}, x ${g!.k}) = ${s.writes![0].value} vào ${dest}: ${role}.`,
      );
    }
    case 'barrier1':
      return tr(
        `${who} reaches barrier 1 and waits until every thread has loaded plane ${s.i + 1}.`,
        `${who} đến rào chắn 1 và chờ đến khi mọi luồng đã nạp mặt phẳng ${s.i + 1}.`,
      );
    case 'barrier2':
      return tr(
        `${who} reaches barrier 2 and waits until every thread has finished reading for output plane ${s.i}.`,
        `${who} đến rào chắn 2 và chờ đến khi mọi luồng đã đọc xong cho mặt phẳng đầu ra ${s.i}.`,
      );
    case 'compute': {
      if (!s.active)
        return s.why === 'halo'
          ? tr(
              `${who} sits on the ring of the tile: it fails the guard and computes nothing. It still loads, and it still reaches both barriers.`,
              `${who} nằm trên vành ngoài của ô: nó không qua điều kiện chặn và không tính gì. Nó vẫn nạp, và vẫn đến được cả hai rào chắn.`,
            )
          : tr(
              `${who}: its point of output plane ${s.i} lies on the grid's boundary layer, which the sweep never computes.`,
              `${who}: điểm của nó trong mặt phẳng đầu ra ${s.i} nằm trên lớp biên của lưới, nơi lượt quét không bao giờ tính.`,
            );
      const sum = s.terms!.map((t) => `${COEFFS[t.coeff]}·${show(t.read.value)}`).join(' + ');
      const swapped = cfg.pairing === 'as-printed-8.10';
      return (
        <Translated
          vi={
            <>
              {who} tính out tại (z {g!.z}, y {g!.j}, x {g!.k}) = {sum} = <b>{show(s.value!)}</b>.{' '}
              {cfg.kernel === 'planes'
                ? 'Bảy lần đọc bộ nhớ chia sẻ: tâm, x ± 1 và y ± 1 từ inCurr_s, z − 1 từ inPrev_s, z + 1 từ inNext_s.'
                : 'Bốn lần đọc bộ nhớ chia sẻ, các láng giềng x và y từ inCurr_s; tâm, z − 1 và z + 1 đến từ thanh ghi của chính nó.'}
              {swapped ? ' Hệ số của y − 1 và y + 1 bị đổi chỗ, như bản in.' : ''}
            </>
          }
        >
          <>
            {who} computes out at (z {g!.z}, y {g!.j}, x {g!.k}) = {sum} = <b>{show(s.value!)}</b>.{' '}
            {cfg.kernel === 'planes'
              ? 'Seven shared-memory reads: centre, x ± 1 and y ± 1 from inCurr_s, z − 1 from inPrev_s, z + 1 from inNext_s.'
              : 'Four shared-memory reads, the x and y neighbours from inCurr_s; the centre, z − 1 and z + 1 come from its own registers.'}
            {swapped ? ' The coefficients of y − 1 and y + 1 are swapped, as printed.' : ''}
          </>
        </Translated>
      );
    }
    case 'shiftPrev':
      return tr(
        `${who} copies its own inCurr_s[${s.ty}][${s.tx}] (plane ${s.reads![0].z ?? '–'}) into inPrev_s[${s.ty}][${s.tx}].`,
        `${who} chép inCurr_s[${s.ty}][${s.tx}] của chính nó (mặt phẳng ${s.reads![0].z ?? '–'}) vào inPrev_s[${s.ty}][${s.tx}].`,
      );
    case 'shiftCurr':
      return tr(
        `${who} copies its own inNext_s[${s.ty}][${s.tx}] (plane ${s.reads![0].z ?? '–'}) into inCurr_s[${s.ty}][${s.tx}]. Neighbours read this slot, which is why barrier 2 comes first.`,
        `${who} chép inNext_s[${s.ty}][${s.tx}] của chính nó (mặt phẳng ${s.reads![0].z ?? '–'}) vào inCurr_s[${s.ty}][${s.tx}]. Các láng giềng đọc ô nhớ này, đó là lý do rào chắn 2 phải đến trước.`,
      );
    case 'rotate':
      return tr(
        `${who} moves registers: inPrev = inCurr (plane ${s.reads![0].z ?? '–'}), inCurr = inNext (plane ${s.reads![1].z ?? '–'}). No memory is touched.`,
        `${who} chuyển thanh ghi: inPrev = inCurr (mặt phẳng ${s.reads![0].z ?? '–'}), inCurr = inNext (mặt phẳng ${s.reads![1].z ?? '–'}). Không chạm vào bộ nhớ nào.`,
      );
    case 'storeCurr':
      return tr(
        `${who} writes its register inNext (plane ${s.reads![0].z ?? '–'}) into inCurr_s[${s.ty}][${s.tx}] for its neighbours. Barrier 2 had to come first for the same reason as in Fig. 8.10.`,
        `${who} ghi thanh ghi inNext của nó (mặt phẳng ${s.reads![0].z ?? '–'}) vào inCurr_s[${s.ty}][${s.tx}] cho các láng giềng. Rào chắn 2 phải đến trước vì cùng lý do như trong Hình 8.10.`,
      );
    default:
      return '';
  }
}

const CELL = { minWidth: '1.6rem', height: '1.5rem', fontSize: '0.7rem' } as const;
const SMALL = { minWidth: '1.35rem', height: '1.3rem', fontSize: '0.64rem' } as const;
const STICKY = { position: 'sticky', top: 0, background: 'var(--page)', zIndex: 1 } as const;
const SLOT6 = { minWidth: '1.4rem', height: '1.4rem', fontSize: '0.66rem' } as const;

/** Which plane a buffer holds across the slots that hold anything. */
function holds(slots: Slot[]): string {
  const zs = [...new Set(slots.filter((x) => x.z !== null).map((x) => x.z as number))].sort((a, b) => a - b);
  if (!zs.length) return tr('empty', 'trống');
  return zs.length === 1
    ? tr(`holds plane ${zs[0]}`, `giữ mặt phẳng ${zs[0]}`)
    : tr(`mixed: planes ${zs.join(' and ')}`, `lẫn lộn: các mặt phẳng ${zs.join(' và ')}`);
}

function Stepper() {
  const [pi, setPi] = useState(0);
  const prob0 = PROBLEMS[pi];
  const [kernel, setKernel] = useState<Kernel>('planes');
  const [schedule, setSchedule] = useState<Schedule>('in-order');
  const [b1, setB1] = useState(true);
  const [b2, setB2] = useState(true);
  const [pairing, setPairing] = useState<Pairing>('book-8.6');
  const cfg: ZConfig = useMemo(
    () => ({ N: prob0.N, inTile: prob0.inTile, kernel, barrier1: b1, barrier2: b2, pairing }),
    [prob0, kernel, b1, b2, pairing],
  );
  const inner = useMemo(() => interiorBlock(cfg), [cfg]);
  const [blockKey, setBlockKey] = useState(`${inner.bx},${inner.by},${inner.bz}`);
  const [bx, by, bz] = blockKey.split(',').map(Number);
  const [n, setN] = useState(0);
  const [playing, setPlaying] = useState(false);
  const scroller = useRef<HTMLDivElement | null>(null);

  const prob = useMemo(() => makeProblem(cfg.N), [cfg.N]);
  const locale = useLocale();
  const trace = useMemo(() => runBlock(cfg, prob, bx, by, bz, schedule, 5), [cfg, prob, bx, by, bz, schedule, locale]);
  const total = trace.steps.length;
  const at = Math.min(n, total);
  const st = useMemo(() => stateAt(cfg, trace, at), [cfg, trace, at]);
  const cur: ZStep | undefined = at > 0 ? trace.steps[at - 1] : undefined;
  const firstBad = trace.violations.length ? trace.violations[0].step + 1 : null;
  const T = cfg.inTile;
  const O = outWidth(cfg);
  const D = depthOf(cfg);
  const g = gridBlocks(cfg);
  const code = codeFor(kernel, pairing, T);

  useEffect(() => setN(0), [pi, kernel, schedule, b1, b2, pairing, blockKey]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setN((x) => x + 1), 260);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);
  // Keep the current thread's row in view inside its own scroll box (never scroll the page).
  useEffect(() => {
    const box = scroller.current;
    const row = cur && cur.thread >= 0 ? box?.querySelector<HTMLElement>(`tr[data-thread="${cur.thread}"]`) : null;
    if (box && row) box.scrollTop = Math.max(0, row.offsetTop - box.clientHeight / 2);
  }, [cur]);

  const nextBarrier = () => {
    for (let x = at; x < total; x++) if (trace.steps[x].thread < 0) return setN(x + 1);
    setN(total);
  };

  // Highlights for the current step.
  const slotCls = (buf: SharedBuf, r: number, c: number) => {
    const rd = cur?.reads?.find((x) => x.buf === buf && x.r === r && x.c === c);
    if (rd && !rd.ok && rd.role !== 'copy') return 'bad';
    if (rd) return 'hot';
    if (cur?.writes?.some((w) => w.buf === buf && w.r === r && w.c === c)) return 'hot';
    return '';
  };
  const bufs: SharedBuf[] = kernel === 'planes' ? ['inPrev_s', 'inCurr_s', 'inNext_s'] : ['inCurr_s'];
  const stored = new Map(st.stores.map((x) => [x.index, x.value]));
  const iters = trace.steps.filter((s) => s.op === 'compute').length / (T * T);
  const finished = at >= total;
  const wrongHere = st.stores.filter((x) => x.value !== prob.ref[x.index]).length;
  const c = st.counters;
  const perOutput = c.globalStores ? c.stencilSharedReads / c.globalStores : 0;
  const perThreadIter = (v: number) => (finished && iters ? v / (T * T * iters) : null);

  // The block's part of the input: planes iStart - 1 .. iStart + D, rows ty and columns tx of its tile.
  const zs = Array.from({ length: D + 2 }, (_, q) => trace.iStart - 1 + q);
  const zOut = Array.from({ length: D }, (_, q) => trace.iStart + q);
  const blocks: string[] = [];
  for (let z = 0; z < g.z; z++) for (let y = 0; y < g.xy; y++) for (let x = 0; x < g.xy; x++) blocks.push(`${x},${y},${z}`);

  return (
    <Bench
      title={tr('Step one block through z', 'Cho một khối đi từng bước dọc z')}
      task={tr('Follow one thread through the first iteration, then watch the whole block reach each barrier.', 'Theo một luồng qua vòng lặp đầu tiên, rồi xem cả khối đến từng rào chắn.')}
      foot={
        <Translated
          vi={
            <>
              Các giá trị là số nguyên nhỏ và trọng số là c0…c6 = {COEFFS.join(', ')}, nên bạn có thể kiểm tra một tổng bằng tay. Số
              nhỏ ở góc một ô nhớ là mặt phẳng z mà giá trị của nó đến từ đó. <TierLegend />
            </>
          }
        >
          <>
            Values are small integers and the weights are c0…c6 = {COEFFS.join(', ')}, so you can check a sum by hand. The
            small number in a slot's corner is the plane z its value came from. <TierLegend />
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi="Bài toán">Problem</Translated>
          <select value={pi} onChange={(e) => {
            const k = +e.target.value;
            const nb = interiorBlock({ ...cfg, N: PROBLEMS[k].N, inTile: PROBLEMS[k].inTile });
            setPi(k);
            setBlockKey(`${nb.bx},${nb.by},${nb.bz}`);
          }}>
            {PROBLEMS.map((p, k) => <option key={k} value={k}>{p.label}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Khối (x, y, z)">Block (x, y, z)</Translated>
          <select value={blockKey} onChange={(e) => setBlockKey(e.target.value)}>
            {blocks.map((b) => (
              <option key={b} value={b}>({b.split(',').join(', ')}){b === `${inner.bx},${inner.by},${inner.bz}` ? tr(': touches no edge', ': không chạm rìa nào') : ''}</option>
            ))}
          </select>
        </label>
        <label className="field">
          <Translated vi="Kernel">Kernel</Translated>
          <select value={kernel} onChange={(e) => setKernel(e.target.value as Kernel)}>
            {(['planes', 'registers'] as Kernel[]).map((k) => <option key={k} value={k}>{KERNEL_LABEL[k]}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Thứ tự luồng">Thread order</Translated>
          <select value={schedule} onChange={(e) => setSchedule(e.target.value as Schedule)}>
            {SCHEDULES.map((s) => <option key={s} value={s}>{SCHEDULE_LABEL[s]}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Hệ số">Coefficients</Translated>
          <select value={pairing} onChange={(e) => setPairing(e.target.value as Pairing)}>
            {(['book-8.6', 'as-printed-8.10'] as Pairing[]).map((p) => <option key={p} value={p}>{PAIRING_LABEL[p]}</option>)}
          </select>
        </label>
        <label className="check"><input type="checkbox" checked={b1} onChange={(e) => setB1(e.target.checked)} /> <Translated vi="rào chắn 1">barrier 1</Translated></label>
        <label className="check"><input type="checkbox" checked={b2} onChange={(e) => setB2(e.target.checked)} /> <Translated vi="rào chắn 2">barrier 2</Translated></label>
      </div>

      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setN(0); }} disabled={at === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => { setPlaying(false); setN(at - 1); }} disabled={at === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => { setPlaying(false); setN(at + 1); }} disabled={at >= total}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => { setPlaying(false); nextBarrier(); }} disabled={at >= total}><Translated vi="Chạy đến rào chắn kế tiếp">Run to next barrier</Translated></button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? tr('Pause', 'Tạm dừng') : tr('Play', 'Phát')}</button>
        {firstBad ? <button onClick={() => { setPlaying(false); setN(firstBad); }}><Translated vi="Đến lỗi đầu tiên">Go to the first failure</Translated></button> : null}
        <span className="muted" aria-live="polite"><Translated vi={<>bước {at} trên {total}</>}>step {at} of {total}</Translated></span>
      </div>

      <div className={'narrate' + (cur?.violation ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }}>
        {cur ? (
          <>
            {describe(cur, cfg)}
            {cur.violation ? <><br /><b className="t-fail"><Translated vi={<>Lỗi: {cur.violation.message}.</>}>Failure: {cur.violation.message}.</Translated></b></> : null}
          </>
        ) : (
          <Translated
            vi={
              <>
                Chưa có gì chạy. Khối ({bx}, {by}, {bz}) là một mặt phẳng {T} × {T} luồng. Nó tính các mặt phẳng đầu ra{' '}
                {trace.iStart} đến {trace.iStart + D - 1}, các hàng y {by * O} đến {by * O + O - 1} và các cột x {bx * O} đến{' '}
                {bx * O + O - 1}, ở bất cứ đâu chúng nằm trong lưới. Hãy nhấn Bước.
              </>
            }
          >
            <>
              Nothing has run yet. Block ({bx}, {by}, {bz}) is a plane of {T} × {T} threads. It computes output planes{' '}
              {trace.iStart} to {trace.iStart + D - 1}, rows y {by * O} to {by * O + O - 1} and columns x {bx * O} to{' '}
              {bx * O + O - 1}, wherever those are inside the grid. Press Step.
            </>
          </Translated>
        )}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 7fr) minmax(0, 4fr)' }}>
        <div>
          <Code tabs={[{ label: kernel === 'planes' ? 'zmarch_planes.cu' : 'zmarch_registers.cu', code, tiers: STENCIL_TIERS, status: tr(SNIPPET_STATUS, 'front-end-checked; chưa từng chạy (trình chạy từng bước chạy một mô hình đã kiểm thử của nó)') }]}
            highlight={linesFor(code, cur)} />
        </div>
        <div>
          <div className="row" style={{ alignItems: 'flex-start', gap: '0.9rem' }}>
            {bufs.map((b) => (
              <div key={b} data-buf={b}>
                <div className="mx-label t-shared">{b} <span className="muted" style={{ fontWeight: 400 }}>{holds(st.shared[b])}</span></div>
                <div className="mx shared" style={{ gridTemplateColumns: `repeat(${T}, auto)` }}>
                  {st.shared[b].map((s, q) => (
                    <span key={q} style={T > 4 ? SLOT6 : CELL} className={slotCls(b, Math.floor(q / T), q % T)}>
                      {s.z === null ? '·' : show(s.value)}
                      {s.z !== null ? <sub>{s.z}</sub> : null}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <div className="panel" data-out="counters" data-loads={c.globalLoads} data-stores={c.globalStores}
            data-reads={c.stencilSharedReads} data-copy-reads={c.copySharedReads} data-copy-writes={c.copySharedWrites}
            style={{ marginTop: '0.8rem' }}>
            <table className="data" style={{ margin: 0 }}>
              <tbody>
                {[
                  ['global', tr('global loads', 'lần nạp toàn cục'), String(c.globalLoads), tr(`${st.planes.length} planes touched`, `đã chạm ${st.planes.length} mặt phẳng`), ''],
                  ['global', tr('global stores', 'lần lưu toàn cục'), String(c.globalStores), tr('one per output', 'một lần mỗi đầu ra'), ''],
                  ['shared', tr('shared reads, stencil', 'lần đọc chia sẻ, stencil'), String(c.stencilSharedReads), c.globalStores ? tr(`${fmt(perOutput)} per output`, `${fmt(perOutput)} mỗi đầu ra`) : tr('none yet', 'chưa có'), 'per-output'],
                  ['shared', tr('shared, shifting', 'chia sẻ, dịch cửa sổ'), tr(`${c.copySharedReads} reads, ${c.copySharedWrites} writes`, `${c.copySharedReads} lần đọc, ${c.copySharedWrites} lần ghi`),
                    finished
                      ? tr(
                          `${fmt(perThreadIter(c.copySharedReads)!)} reads + ${fmt(perThreadIter(c.copySharedWrites)!)} writes per thread per iteration`,
                          `${fmt(perThreadIter(c.copySharedReads)!)} lần đọc + ${fmt(perThreadIter(c.copySharedWrites)!)} lần ghi mỗi luồng mỗi vòng lặp`,
                        )
                      : tr('per-thread figures at the end', 'số liệu mỗi luồng có ở cuối'), 'per-thread-iter'],
                  ...(kernel === 'registers' ? [['reg', tr('register moves', 'lần chuyển thanh ghi'), String(c.registerMoves), tr('no memory touched', 'không chạm bộ nhớ'), '']] : []),
                ].map(([tier, label, value, note, key]) => (
                  <tr key={label}>
                    <td style={{ paddingLeft: 0 }}>
                      <span className={'chip ' + tier}>{label}</span>
                      <div className="muted" style={{ fontSize: '0.74rem', marginTop: '0.15rem' }} data-out={key || undefined}>{note}</div>
                    </td>
                    <td className="num" style={{ whiteSpace: 'nowrap' }}>{value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="mx-label" style={{ marginTop: '0.9rem' }}>
        <Translated vi="Các luồng của khối">Threads of the block</Translated>
      </div>
      <div ref={scroller} style={{ maxHeight: '12rem', overflowY: 'auto', position: 'relative' }}>
            <table className="threads" aria-label={tr('Threads of the block', 'Các luồng của khối')}>
              <thead>
                <tr>
                  <th style={STICKY}><Translated vi="luồng">thread</Translated></th><th style={STICKY}><Translated vi="tại (y, x)">at (y, x)</Translated></th><th style={STICKY}><Translated vi="đang làm">doing</Translated></th>
                  {kernel === 'registers' ? <><th style={STICKY} className="t-reg">inPrev</th><th style={STICKY} className="t-reg">inCurr</th><th style={STICKY} className="t-reg">inNext</th></> : null}
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: T * T }, (_, t) => {
                  const tx = t % T;
                  const ty = Math.floor(t / T);
                  const p = position(cfg, bx, by, tx, ty);
                  const last = st.last[t] >= 0 ? trace.steps[st.last[t]] : undefined;
                  const ring = tx === 0 || ty === 0 || tx === T - 1 || ty === T - 1;
                  const doing = st.done[t]
                    ? tr('finished', 'đã xong')
                    : st.waiting[t]
                      ? tr(`waiting at barrier ${last?.op === 'barrier1' ? 1 : 2}`, `đang chờ ở rào chắn ${last?.op === 'barrier1' ? 1 : 2}`)
                      : last
                        ? last.iter < 0 ? tr('loading', 'đang nạp') : tr(`output plane ${last.i}`, `mặt phẳng đầu ra ${last.i}`)
                        : tr('not started', 'chưa bắt đầu');
                  const r = st.regs[t];
                  const reg = (x: Slot) => (x.z === null ? '·' : <>{show(x.value)}<sub className="muted"> z{x.z}</sub></>);
                  return (
                    <tr key={t} data-thread={t} className={cur?.thread === t ? 'cur' : ''}>
                      <td>(tx {tx}, ty {ty}{/* i18n-ignore: identifiers */}){ring ? <span className="muted"><Translated vi=" vành"> ring</Translated></span> : null}</td>
                      <td>({p.j}, {p.k})</td>
                      <td>{doing}</td>
                      {kernel === 'registers' ? <><td className="t-reg">{reg(r.inPrev)}</td><td className="t-reg">{reg(r.inCurr)}</td><td className="t-reg">{reg(r.inNext)}</td></> : null}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
      <div style={{ marginTop: '1rem' }}>
        <div className="mx-label t-global">
          <Translated vi="Đầu vào trong bộ nhớ toàn cục: ô của khối này, mỗi z một mặt phẳng x-y">Input in global memory: this block's tile, one x-y plane per z</Translated>
        </div>
        <div className="row" style={{ alignItems: 'flex-start', gap: '0.8rem' }}>
          {zs.map((z) => {
            const loadingThis = cur?.global?.array === 'in' && cur.global.z === z;
            return (
              <div key={z} data-z={z}>
                <div className="muted" style={{ fontSize: '0.72rem' }}>z = {z}{st.planes.includes(z) ? tr(' (loaded)', ' (đã nạp)') : ''}</div>
                <div className="mx global" style={{ gridTemplateColumns: `repeat(${T}, auto)` }}>
                  {Array.from({ length: T * T }, (_, q) => {
                    const p = position(cfg, bx, by, q % T, Math.floor(q / T));
                    const inside = z >= 0 && z < cfg.N && p.j >= 0 && p.j < cfg.N && p.k >= 0 && p.k < cfg.N;
                    const hot = loadingThis && cur!.global!.j === p.j && cur!.global!.k === p.k;
                    return (
                      <span key={q} style={SMALL} className={!inside ? 'void' : hot ? 'hot' : loadingThis ? 'on' : ''}>
                        {inside ? prob.input[idx3(cfg.N, z, p.j, p.k)] : ''}
                      </span>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
        <div className="mx-label" style={{ marginTop: '0.8rem', color: 'var(--ok)' }}>
          <Translated vi="Đầu ra: các mặt phẳng của khối này (xanh khi bằng kết quả quét trên CPU)">Output: this block's planes (green when equal to the CPU sweep)</Translated>
        </div>
        <div className="row" style={{ alignItems: 'flex-start', gap: '0.8rem' }}>
          {zOut.map((z) => (
            <div key={z} data-out-z={z}>
              <div className="muted" style={{ fontSize: '0.72rem' }}>z = {z}</div>
              <div className="mx global" style={{ gridTemplateColumns: `repeat(${O}, auto)` }}>
                {Array.from({ length: O * O }, (_, q) => {
                  const j = by * O + Math.floor(q / O);
                  const k = bx * O + (q % O);
                  const interior = z >= 1 && z < cfg.N - 1 && j >= 1 && j < cfg.N - 1 && k >= 1 && k < cfg.N - 1;
                  const ix = interior ? idx3(cfg.N, z, j, k) : -1;
                  const v = interior ? stored.get(ix) : undefined;
                  const cls = !interior ? 'void' : v === undefined ? '' : v === prob.ref[ix] ? 'good' : 'bad';
                  return <span key={q} style={SMALL} className={cls} data-state={cls || 'pending'}>{v === undefined ? '' : show(v)}</span>;
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {finished ? (
        <p style={{ margin: '0.9rem 0 0' }} data-out="verdict">
          {wrongHere === 0 ? (
            <span className="chip ok"><Translated vi={<>{st.stores.length} đầu ra của khối này khớp với kết quả quét trên CPU.</>}>This block's {st.stores.length} outputs match the CPU sweep.</Translated></span>
          ) : (
            <span className="chip fail"><Translated vi={<>{wrongHere} trên {st.stores.length} đầu ra của khối này bị sai.</>}>{wrongHere} of this block's {st.stores.length} outputs are wrong.</Translated></span>
          )}{' '}
          {trace.violations.length ? tr(`${trace.violations.length} bad reads of shared memory.`, `${trace.violations.length} lần đọc bộ nhớ chia sẻ sai.`) : ''}
        </p>
      ) : null}
    </Bench>
  );
}

function Outcomes() {
  const [pi, setPi] = useState(0);
  const [shown, setShown] = useState(false);
  const p = PROBLEMS[pi];
  const prob = useMemo(() => makeProblem(p.N), [p.N]);
  const rows = useMemo(() => {
    const variants: { label: string; over: Partial<ZConfig> }[] = [
      { get label() { return tr('both barriers', 'cả hai rào chắn'); }, over: {} },
      { get label() { return tr('without barrier 1', 'không có rào chắn 1'); }, over: { barrier1: false } },
      { get label() { return tr('without barrier 2', 'không có rào chắn 2'); }, over: { barrier2: false } },
    ];
    return (['planes', 'registers'] as Kernel[]).flatMap((kernel) =>
      variants.map((v) => ({
        kernel,
        get label() { return v.label; },
        cells: SCHEDULES.map((s) => runAll({ N: p.N, inTile: p.inTile, kernel, ...CORRECT, ...v.over }, prob, s, 5)),
      })),
    );
  }, [p, prob]);
  return (
    <Bench
      title={tr('Every thread order at once', 'Mọi thứ tự luồng cùng lúc')}
      task={tr('Before you look: which column will be correct in every row?', 'Trước khi xem: cột nào sẽ đúng ở mọi hàng?')}
      foot={tr(
        'Each cell runs every block of the grid through the model above and compares the output with the CPU sweep.',
        'Mỗi ô chạy mọi khối của lưới qua mô hình ở trên và so đầu ra với kết quả quét trên CPU.',
      )}
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi="Bài toán">Problem</Translated>
          <select value={pi} onChange={(e) => setPi(+e.target.value)}>
            {PROBLEMS.map((q, k) => <option key={k} value={k}>{q.label}</option>)}
          </select>
        </label>
        <button className={shown ? 'quiet' : 'primary'} onClick={() => setShown(!shown)} aria-pressed={shown} data-control="reveal">
          {shown ? tr('Hide the results', 'Ẩn kết quả') : tr('Show the results', 'Hiện kết quả')}
        </button>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0 }}>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}>
                <Translated vi="Kernel">Kernel</Translated>
              </th>
              <th>
                <Translated vi="Rào chắn">Barriers</Translated>
              </th>
              {SCHEDULES.map((s) => <th key={s}>{SCHEDULE_LABEL[s]}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.kernel + r.label} data-row={`${r.kernel}/${r.label}`}>
                <td style={{ paddingLeft: '0.6rem' }}>{KERNEL_LABEL[r.kernel]}</td>
                <td>{r.label}</td>
                {r.cells.map((res, k) => {
                  if (!shown)
                    return (
                      <td key={k} data-outcome="hidden" className="muted">
                        ?
                      </td>
                    );
                  const kinds = new Set(res.violations.map((x) => x.kind));
                  return (
                    <td key={k} data-outcome={res.correct ? 'correct' : 'wrong'}>
                      {res.correct ? <span className="chip ok"><Translated vi="đúng">correct</Translated></span> : <span className="chip fail"><Translated vi={<>sai {res.wrong.length} trên {prob.interior.length}</>}>{res.wrong.length} of {prob.interior.length} wrong</Translated></span>}
                      {kinds.has('read-before-write') ? <div className="muted"><Translated vi="đọc trước khi ghi">read before write</Translated></div> : null}
                      {kinds.has('overwritten-before-read') ? <div className="muted"><Translated vi="bị ghi đè trước khi đọc">overwritten before read</Translated></div> : null}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the explorer
const f2 = (v: number) => v.toFixed(2);
const FIELD = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' } as const;
const realDevices = (reports: Reports) => Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const devNum = (v: unknown) => {
  const x = Number(v);
  return isFinite(x) && x > 0 ? x : 0;
};

// Scaled to a phone the plot's labels would be a few pixels tall, so below this width it
// keeps its size and scrolls sideways in its frame (as a scene's stage does).
const SLAB_PLOT_MIN = 600;

function SlabPlot({ t, D, r, bytes }: { t: number; D: number; r: number; bytes: number }) {
  const W = 700;
  const H = 300;
  const pad = { l: 52, r: 18, t: 16, b: 42 };
  const limit = slabAILimit(t, r, bytes);
  const cube8 = cubeAI(8, r, bytes);
  const asCoded = t - 2 * r;
  const yMax = Math.max(limit, cube8) * 1.18;
  const X = (d: number) => pad.l + (Math.log2(d) / 9) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - (v / yMax) * (H - pad.t - pad.b);
  const pts: string[] = [];
  for (let q = 0; q <= 180; q++) {
    const d = 2 ** (q / 20);
    pts.push(`${X(d).toFixed(1)},${Y(slabAI(t, d, r, bytes)).toFixed(1)}`);
  }
  const book = t === 32 && r === 1 && bytes === 4;
  const yt: number[] = [];
  const step = yMax > 4 ? 1 : yMax > 2 ? 0.5 : 0.25;
  for (let v = 0; v <= yMax; v += step) yt.push(v);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('FLOP per byte against slab depth', 'FLOP mỗi byte theo độ sâu tấm')} style={{ width: '100%', minWidth: SLAB_PLOT_MIN, maxWidth: W, display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {[1, 2, 4, 8, 16, 32, 64, 128, 256, 512].map((d) => (
        <g key={d}>
          <line x1={X(d)} x2={X(d)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(d)} y={H - pad.b + 15} fontSize="12.5" textAnchor="middle">{d}</text>
        </g>
      ))}
      {yt.map((v) => (
        <g key={v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 6} y={Y(v) + 4} fontSize="12.5" textAnchor="end">{fmt(v)}</text>
        </g>
      ))}
      <text x={(W + pad.l) / 2} y={H - 8} fontSize="13" textAnchor="middle">
        <Translated vi="số mặt phẳng đầu ra mỗi khối, D (thang log)">output planes per block, D (log scale)</Translated>
      </text>
      <text x={14} y={(H - pad.b + pad.t) / 2} fontSize="13" textAnchor="middle" transform={`rotate(-90 14 ${(H - pad.b + pad.t) / 2})`}>FLOP/B</text>
      <line x1={pad.l} x2={W - pad.r} y1={Y(limit)} y2={Y(limit)} stroke="var(--ink)" strokeDasharray="5 4" />
      <text x={W - pad.r - 4} y={Y(limit) - 6} fontSize="12.5" textAnchor="end" data-mark="limit">
        D → ∞: {f2(limit)}{book ? tr(" (the book's printed 1.52)", ' (con số 1.52 sách in)') : ''}
      </text>
      <line x1={pad.l} x2={W - pad.r} y1={Y(cube8)} y2={Y(cube8)} stroke="var(--ink-3)" strokeDasharray="2 3" />
      <text x={pad.l + 6} y={Y(cube8) + 14} fontSize="12.5" style={{ fill: 'var(--ink-2)' }} data-mark="cube8">
        <Translated vi={<>khối lập phương rộng 8, Hình 8.8: {f2(cube8)}</>}>8-wide cube, Fig. 8.8: {f2(cube8)}</Translated>
      </text>
      <polyline points={pts.join(' ')} fill="none" stroke="var(--shared)" strokeWidth="2.2" />
      {asCoded >= 1 ? (
        <g>
          <circle cx={X(asCoded)} cy={Y(slabAI(t, asCoded, r, bytes))} r="5" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.8" />
          <text x={X(asCoded) + 8} y={Y(slabAI(t, asCoded, r, bytes)) + 16} fontSize="12.5" data-mark="as-coded">
            <Translated vi={<>như đã viết mã, D = {asCoded}: {f2(slabAI(t, asCoded, r, bytes))}</>}>
              as coded, D = {asCoded}: {f2(slabAI(t, asCoded, r, bytes))}
            </Translated>
          </text>
        </g>
      ) : null}
      <rect x={X(D) - 5} y={Y(slabAI(t, D, r, bytes)) - 5} width="10" height="10" transform={`rotate(45 ${X(D)} ${Y(slabAI(t, D, r, bytes))})`} fill="var(--shared)" />
    </svg>
  );
}

function SlabExplorer() {
  const { reports } = useReports();
  const [t, setT] = useState(32);
  const [D, setD] = useState(30);
  const [r, setR] = useState(1);
  const [kind, setKind] = useState<SlabKernel>('planes');
  const [bytes, setBytes] = useState(4);
  const [N, setNN] = useState(256);
  const asCoded = t - 2 * r;
  const ai = slabAI(t, D, r, bytes);
  const limit = slabAILimit(t, r, bytes);
  const life = lifetimeCounts(t, D, r);
  const shared = sharedBytes(kind, t, r, bytes);
  const other = sharedBytes(kind === 'planes' ? 'registers' : 'planes', t, r, bytes);
  const threads = threadsSlab(t);
  const blocksNow = blocksSlab(N, t, D, r);
  const blocksCoded = blocksSlab(N, t, asCoded, r);
  const devices = realDevices(reports);
  const limitsRows = devices.flatMap((b) =>
    // The register-tiled kernel's rows only: lab 8.1 also records the cube kernel's limit (kernel: "cube").
    benchRows(reports[b], 'ch08/lab1_stencil3d').filter((x) => x.name === 'limits' && x.kernel !== 'cube').map((x) => ({ b, row: x })),
  );

  return (
    <Bench
      title={tr('Slab depth and the register budget', 'Độ sâu tấm và ngân sách thanh ghi')}
      task={tr('Find where a deeper slab stops paying, then switch the kernel and watch what moves and what does not.', 'Tìm nơi một tấm sâu hơn thôi đem lại lợi ích, rồi đổi kernel và xem cái gì thay đổi, cái gì không.')}
      foot={
        <Translated
          vi={
            <>
              Mô hình đếm một khối bên trong: mọi đầu vào của ô được nạp một lần, mọi đầu ra được lưu một lần, như ở p. 192. Số thanh
              ghi chỉ tính cửa sổ stencil; trình biên dịch còn thêm chỉ số, địa chỉ và biến tạm. Số liệu thiết bị chỉ xuất hiện ở nơi
              báo cáo của chính bạn có chúng.
            </>
          }
        >
          <>
            The model counts one interior block: every input of its tile loaded once, every output stored once, as on
            p. 192. Register counts are the stencil window only; a compiler adds indices, addresses and temporaries.
            Device figures appear only where your own reports have them.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi={<>Độ rộng ô t = {t}</>}>Tile width t = {t}</Translated>
          <input type="range" min="8" max="32" value={t} aria-label={tr('tile width', 'độ rộng ô')}
            onChange={(e) => { setT(+e.target.value); }} />
        </label>
        <label className="field">
          <Translated vi={<>Độ sâu tấm D = {D}</>}>Slab depth D = {D}</Translated>
          <input type="range" min="0" max="9" step="0.01" value={Math.log2(D)} aria-label={tr('slab depth', 'độ sâu tấm')}
            onChange={(e) => setD(Math.max(1, Math.round(2 ** +e.target.value)))} />
        </label>
        <button className="quiet" onClick={() => setD(asCoded)} aria-pressed={D === asCoded}>
          <Translated vi="như đã viết mã: D = t − 2r">as coded: D = t − 2r</Translated>
        </button>
        <span className="field" style={FIELD}>
          <Translated vi="Bậc stencil r">Stencil order r</Translated>
          <span className="seg" role="group" aria-label={tr('stencil order', 'bậc stencil')}>
            {[1, 2, 3].map((x) => (
              <button key={x} aria-pressed={r === x} onClick={() => { setR(x); if (D === asCoded) setD(t - 2 * x); }}>{x}</button>
            ))}
          </span>
        </span>
        <span className="field" style={FIELD}>
          <Translated vi="Kernel">Kernel</Translated>
          <span className="seg" role="group" aria-label={tr('kernel', 'kernel')}>
            <button aria-pressed={kind === 'planes'} onClick={() => setKind('planes')}>
              <Translated vi="mặt phẳng chia sẻ">shared planes</Translated>
            </button>
            <button aria-pressed={kind === 'registers'} onClick={() => setKind('registers')}>
              <Translated vi="thanh ghi">registers</Translated>
            </button>
          </span>
        </span>
        <label className="field">
          <Translated vi="Số byte mỗi giá trị">Bytes per value</Translated>
          <select value={bytes} onChange={(e) => setBytes(+e.target.value)}>
            <option value={2}>2 (half)</option>{/* i18n-ignore: type names */}
            <option value={4}>4 (float)</option>{/* i18n-ignore: type names */}
            <option value={8}>8 (double)</option>{/* i18n-ignore: type names */}
          </select>
        </label>
        <label className="field">
          <Translated vi="Lưới N">Grid N</Translated>
          <input type="number" min="8" max="4096" value={N} onChange={(e) => setNN(Math.max(8, Math.min(4096, Math.floor(+e.target.value) || 8)))} />
        </label>
      </div>
      <div>
        <div style={{ overflowX: 'auto' }} data-scroll="slab-plot">
          <SlabPlot t={t} D={D} r={r} bytes={bytes} />
        </div>
        <div className="panel" style={{ marginTop: '0.9rem' }} data-out="slab" data-ai={f2(ai)} data-limit={f2(limit)} data-shared={shared} data-blocks={blocksNow} data-planes={planesLoaded(D, r)}>
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              {(
                [
                  [<Translated vi={<>FLOP/B ở D = {D}</>}>FLOP/B at D = {D}</Translated>, <b>{f2(ai)}</b>, tr(`${fmt((100 * ai) / limit)}% of the D → ∞ limit, ${f2(limit)}`, `${fmt((100 * ai) / limit)}% của giới hạn khi D → ∞, ${f2(limit)}`), 'ai'],
                  [tr('Threads per block', 'Số luồng mỗi khối'), threads.toLocaleString('en-US'), tr(`t²; the book's limit is ${BOOK_BLOCK_LIMIT.toLocaleString('en-US')}`, `t²; giới hạn của sách là ${BOOK_BLOCK_LIMIT.toLocaleString('en-US')}`), 'threads'],
                  [<span className="chip shared"><Translated vi="bộ nhớ chia sẻ mỗi khối">shared memory per block</Translated></span>, `${shared.toLocaleString('en-US')} B`,
                    kind === 'planes'
                      ? tr(`${2 * r + 1} planes; with registers ${other.toLocaleString('en-US')} B`, `${2 * r + 1} mặt phẳng; với thanh ghi ${other.toLocaleString('en-US')} B`)
                      : tr(`1 plane; with shared planes ${other.toLocaleString('en-US')} B`, `1 mặt phẳng; với các mặt phẳng chia sẻ ${other.toLocaleString('en-US')} B`), 'shared'],
                  [<span className="chip reg"><Translated vi="thanh ghi cửa sổ mỗi luồng">window registers per thread</Translated></span>, String(kind === 'registers' ? windowRegs(r) : 0),
                    kind === 'registers'
                      ? tr(`+${extraRegsPerBlock(t, r).toLocaleString('en-US')} per block`, `+${extraRegsPerBlock(t, r).toLocaleString('en-US')} mỗi khối`)
                      : tr(
                          `the window is in shared memory; registers would add ${extraRegsPerBlock(t, r).toLocaleString('en-US')} per block`,
                          `cửa sổ nằm trong bộ nhớ chia sẻ; thanh ghi sẽ thêm ${extraRegsPerBlock(t, r).toLocaleString('en-US')} mỗi khối`,
                        ), 'regs'],
                  [tr('Planes loaded per block', 'Số mặt phẳng nạp mỗi khối'), String(planesLoaded(D, r)), tr(`D + 2r; ${2 * r + 1} on chip at once`, `D + 2r; ${2 * r + 1} trên chip cùng lúc`), 'planes'],
                  [tr('Loads and stores per block', 'Số lần nạp và lưu mỗi khối'), `${life.inputs.toLocaleString('en-US')} + ${life.outputs.toLocaleString('en-US')}`, tr('elements over its whole march', 'phần tử trên toàn bộ hành trình'), 'life'],
                  [<Translated vi={<>Số khối cho một lưới {N}³ điểm</>}>Blocks for a grid of {N}³ points</Translated>, blocksNow.toLocaleString('en-US'),
                    D === asCoded ? tr('as coded', 'như đã viết mã') : tr(`${blocksCoded.toLocaleString('en-US')} as coded (D = ${asCoded})`, `${blocksCoded.toLocaleString('en-US')} như đã viết mã (D = ${asCoded})`), 'blocks'],
                ] as [ReactNode, ReactNode, string, string][]
              ).map(([label, value, note, key]) => (
                <tr key={key}>
                  <td style={{ paddingLeft: 0 }}>
                    {label}
                    <div className="muted" style={{ fontSize: '0.76rem', marginTop: '0.1rem' }}>{note}</div>
                  </td>
                  <td className="num" style={{ whiteSpace: 'nowrap' }} data-out={key}>{value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="panel" data-out="devices" style={{ marginTop: '1rem' }}>
        {devices.length === 0 ? (
          <p className="muted" style={{ margin: 0 }}>
            <Translated
              vi={
                <>
                  Máy này chưa có báo cáo thiết bị nào. Chạy <code>python3 labs/pmpp.py doctor</code> để xem các thiết bị của bạn báo cáo
                  gì, và <code>./verify.sh</code> để ghi lại; khi đó ô này sẽ đối chiếu các thiết lập ở trên với chúng.
                </>
              }
            >
              No device reports on this machine yet. Run <code>python3 labs/pmpp.py doctor</code> to see what your devices
              report, and <code>./verify.sh</code> to record it; this box then checks the settings above against them.
            </Translated>
          </p>
        ) : (
          devices.map((b) => {
            const d = reports[b].device;
            const name = String(d.name ?? b);
            const perBlock = devNum(d.sharedMemPerBlock);
            const regs = devNum(d.regsPerBlock);
            const maxT = devNum(d.maxThreadsPerBlock);
            const tg = devNum(d.maxThreadgroupMemoryLength);
            return (
              <p key={b} style={{ margin: '0 0 0.5rem' }} data-device={b}>
                <Translated
                  vi={
                    <>
                      <b>{name}</b> ({b}):{' '}
                      {maxT ? <>maxThreadsPerBlock {maxT.toLocaleString('en-US')}: {threads <= maxT ? <span className="chip ok">vừa</span> : <span className="chip fail">quá nhiều luồng</span>}. </> : null}
                      {perBlock ? <>sharedMemPerBlock {perBlock.toLocaleString('en-US')} B: {shared <= perBlock ? <span className="chip ok">vừa</span> : <span className="chip fail">không vừa</span>}. </> : null}
                      {regs ? <>regsPerBlock {regs.toLocaleString('en-US')}: ở {threads.toLocaleString('en-US')} luồng, tối đa {Math.floor(regs / threads)} thanh ghi mỗi luồng, ít hơn nếu việc cấp phát làm tròn lên. </> : null}
                      {tg ? <>maxThreadgroupMemoryLength {tg.toLocaleString('en-US')} B: {shared <= tg ? <span className="chip ok">vừa</span> : <span className="chip fail">không vừa</span>}. Giới hạn số luồng của nó tính theo pipeline, lab 8.1 có in ra. </> : null}
                    </>
                  }
                >
                  <b>{name}</b> ({b}):{' '}
                  {maxT ? <>maxThreadsPerBlock {maxT.toLocaleString('en-US')}: {threads <= maxT ? <span className="chip ok">fits</span> : <span className="chip fail">too many threads</span>}. </> : null}
                  {perBlock ? <>sharedMemPerBlock {perBlock.toLocaleString('en-US')} B: {shared <= perBlock ? <span className="chip ok">fits</span> : <span className="chip fail">does not fit</span>}. </> : null}
                  {regs ? <>regsPerBlock {regs.toLocaleString('en-US')}: at {threads.toLocaleString('en-US')} threads, at most {Math.floor(regs / threads)} registers per thread, fewer if allocation rounds up. </> : null}
                  {tg ? <>maxThreadgroupMemoryLength {tg.toLocaleString('en-US')} B: {shared <= tg ? <span className="chip ok">fits</span> : <span className="chip fail">does not fit</span>}. Its thread limit is per pipeline, which lab 8.1 prints. </> : null}
                </Translated>
              </p>
            );
          })
        )}
        {limitsRows.length ? (
          <p style={{ margin: '0.4rem 0 0' }} data-out="lab-limits">
            <Translated vi="Lab 8.1 đã đo giới hạn mỗi khối của kernel phân ô thanh ghi:">
              Lab 8.1 measured the register-tiled kernel's limit per block:
            </Translated>{' '}
            {limitsRows
              .map(({ b, row }) =>
                tr(
                  `${b}, tile ${num(row.tile) ?? '?'}: ${num(row.max_threads) ?? '?'} threads (needs ${(num(row.tile) ?? 0) ** 2})`,
                  `${b}, ô ${num(row.tile) ?? '?'}: ${num(row.max_threads) ?? '?'} luồng (cần ${(num(row.tile) ?? 0) ** 2})`,
                ),
              )
              .join('; ')}.
          </p>
        ) : devices.length ? (
          <p className="muted" style={{ margin: '0.4rem 0 0' }}>
            <Translated vi={<>Chạy lab 8.1 với <code>--bench</code> để thêm giới hạn số luồng đo được theo từng kernel vào đây.</>}>
              Run lab 8.1 with <code>--bench</code> to add its measured per-kernel thread limit here.
            </Translated>
          </p>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- erratum hunt
/**
 * What is the matter with a printed line: `error`, a slip in this listing (a name that does
 * not exist, a line number repeated); `inconsistent`, correct on its own but at odds with
 * Figs. 8.6 and 8.8 (notes/ch08.md records the c3/c4 pairing as an inconsistency, not a
 * code error); `null`, nothing. Lines with either problem are the ones to flag.
 */
type Issue = 'error' | 'inconsistent' | null;
interface PrintedLine {
  n: string;
  code: string;
  issue: Issue;
  explain: ReactNode;
}
const INCONSISTENT = (
  <Translated vi={<>Không nhất quán với Hình 8.6 và 8.8: kết quả khác nhau trừ khi c3 = c4. Lab theo 8.6 và 8.8.</>}>
    <>Inconsistent with Figs. 8.6 and 8.8: the results differ unless c3 = c4. The lab follows 8.6 and 8.8.</>
  </Translated>
);
// The lines of Fig. 8.12 (p. 197) that matter, quoted as printed, with their printed numbers.
const PRINTED: PrintedLine[] = [
  {
    n: '27', code: '          + c3*inCurr_s[threadIdx.y+1][threadIdx.x]', issue: 'inconsistent',
    explain: (
      <Translated vi={<>Ở đây c3 nhân với láng giềng y + 1, như trong Hình 8.10. {INCONSISTENT}</>}>
        <>c3 multiplies the y + 1 neighbour here, as in Fig. 8.10. {INCONSISTENT}</>
      </Translated>
    ),
  },
  {
    n: '28', code: '          + c4*inCurr_s[threadIdx.y-1][threadIdx.x]', issue: 'inconsistent',
    explain: (
      <Translated vi={<>Nửa còn lại của cùng cách ghép: c4 với y − 1. {INCONSISTENT}</>}>
        <>The other half of the same pairing: c4 with y − 1. {INCONSISTENT}</>
      </Translated>
    ),
  },
  { n: '33', code: '      __syncthreads();', issue: null, explain: <Translated vi="Rào chắn 2. Đúng.">Barrier 2. Correct.</Translated> },
  { n: '34', code: '      inPrev = inCurr;', issue: null, explain: <Translated vi="Đúng: cửa sổ tiến lên trong thanh ghi.">Correct: the window moves on in registers.</Translated> },
  { n: '35', code: '      inCurr = inNext;', issue: null, explain: <Translated vi="Đúng.">Correct.</Translated> },
  {
    n: '34', code: '      inCurr_s[threadIdx.y][threadIdx.x] = inNext_s;', issue: 'error',
    explain: (
      <Translated
        vi={
          <>
            Kernel này không có <code>inNext_s</code>: Hình 8.12 đã chuyển mặt phẳng kế tiếp vào thanh ghi <code>inNext</code>.
            Như bản in, dòng này không biên dịch được; ý là thanh ghi. Nó cũng được đánh số 34 lần thứ hai.
          </>
        }
      >
        <>This kernel has no <code>inNext_s</code>: Fig. 8.12 moved the next plane into the register <code>inNext</code>. As printed the line does not compile; the register is meant. It is also numbered 34 a second time.</>
      </Translated>
    ),
  },
  { n: '35', code: '    }', issue: 'error', explain: (
      <Translated vi={<>Lại đánh số 35. "Dòng 34" trong văn bản cho lệnh cập nhật bộ nhớ chia sẻ là số 34 thứ hai.</>}>
        <>Numbered 35 again. The text's "line 34" for the shared-memory update means the second 34.</>
      </Translated>
    ) },
  { n: '35', code: '}', issue: 'error', explain: <Translated vi="Và 35 lần thứ ba. Listing lẽ ra phải chạy từ 33 đến 38.">And 35 a third time. The listing should run 33 to 38.</Translated> },
];

function verdictFor(issue: Issue, picked: boolean): string {
  if (issue === 'error') return picked ? tr('Wrong, and you caught it.', 'Sai, và bạn đã bắt được.') : tr('Wrong, and you missed it.', 'Sai, và bạn đã bỏ sót.');
  if (issue === 'inconsistent')
    return picked ? tr('Inconsistent, and you caught it.', 'Không nhất quán, và bạn đã bắt được.') : tr('Inconsistent, and you missed it.', 'Không nhất quán, và bạn đã bỏ sót.');
  return picked ? tr('This one is fine.', 'Dòng này không sao.') : tr('Fine, rightly left alone.', 'Không sao, để nguyên là đúng.');
}

function ErratumHunt() {
  const [picked, setPicked] = useState<Record<number, boolean>>({});
  const [checked, setChecked] = useState(false);
  const flagged = (l: PrintedLine) => l.issue !== null;
  const found = PRINTED.filter((l, q) => flagged(l) && picked[q]).length;
  const real = PRINTED.filter(flagged).length;
  const falseAlarms = PRINTED.filter((l, q) => !flagged(l) && picked[q]).length;
  const goToListing = () => document.getElementById('s85-fig812')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  return (
    <Bench
      title={tr('Fig. 8.12 as printed', 'Hình 8.12 như bản in')}
      task={tr(
        "Click every line that is wrong, or that disagrees with the book's earlier listings, then check.",
        'Nhấn vào mọi dòng sai, hoặc không khớp với các listing trước đó của sách, rồi kiểm tra.',
      )}
      foot={tr(
        "The quoted lines are the book's, as printed on p. 197, with the page's own line numbers. They are shown because they are wrong or inconsistent; do not copy them.",
        'Các dòng được trích là của sách, như in ở p. 197, với số dòng của chính trang đó. Chúng được hiển thị vì chúng sai hoặc không nhất quán; đừng chép chúng.',
      )}
    >
      <div className="panel" style={{ marginBottom: '0.9rem', overflowX: 'auto' }}>
        <div role="group" aria-label={tr('Printed lines', 'Các dòng in')}>
          {PRINTED.map((l, q) => {
            const on = !!picked[q];
            const should = flagged(l);
            const verdict = checked ? (on === should ? 'right' : 'wrong') : '';
            return (
              <div key={q} data-line={q} data-verdict={verdict} data-issue={checked ? l.issue ?? 'none' : undefined} style={{ marginBottom: '0.2rem' }}>
                <button
                  className="quiet"
                  aria-pressed={on}
                  disabled={checked}
                  onClick={() => setPicked({ ...picked, [q]: !on })}
                  style={{ textAlign: 'left', width: '100%', borderColor: checked ? (l.issue === 'error' ? 'var(--fail)' : l.issue === 'inconsistent' ? 'var(--ink-2)' : 'var(--rule)') : undefined }}
                >
                  <code style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{l.n} {l.code}</code>
                </button>
                {checked ? (
                  <div style={{ fontSize: '0.85rem', margin: '0.2rem 0 0.5rem 0.4rem' }}>
                    <b className={on === should ? 't-ok' : 't-fail'}>{verdictFor(l.issue, on)}</b>{' '}
                    <span className="muted">{l.explain}</span>
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      </div>
      <div className="row">
        <button className="primary" disabled={checked} onClick={() => setChecked(true)}><Translated vi="Kiểm tra">Check</Translated></button>
        {checked ? <button className="quiet" onClick={() => { setChecked(false); setPicked({}); }}><Translated vi="Thử lại">Try again</Translated></button> : null}
        {checked ? <button className="quiet" onClick={goToListing}><Translated vi="Xem listing đã sửa">Show the corrected listing</Translated></button> : null}
      </div>
      {checked ? (
        <div className={'verdict ' + (found === real && !falseAlarms ? 'right' : 'info')} data-out="hunt" data-found={found} data-false={falseAlarms}
          style={{ marginTop: '0.8rem', padding: '0.6rem 0.8rem', border: '1px solid var(--rule)', borderRadius: 4 }}>
          <Translated
            vi={
              <p style={{ margin: '0 0 0.4rem' }}>
                <b>Bạn tìm được {found} trên {real} dòng sai hoặc không nhất quán{falseAlarms ? `, và đánh dấu ${falseAlarms} dòng vốn không sao` : ''}.</b>
              </p>
            }
          >
            <p style={{ margin: '0 0 0.4rem' }}>
              <b>You found {found} of the {real} lines that are wrong or inconsistent{falseAlarms ? ` and flagged ${falseAlarms} that ${falseAlarms === 1 ? 'is' : 'are'} fine` : ''}.</b>
            </p>
          </Translated>
          <Translated
            vi={
              <p style={{ margin: 0 }}>
                Còn một chỗ nữa, trong phần văn ở p. 196: bộ nhớ chia sẻ giảm "xuống một phần ba so với kernel trong Hình 8.12". Câu đó
                mô tả Hình 8.12, nên phép so sánh phải là với Hình 8.10. Kernel đã sửa ở trên qua được các bài kiểm thử của lab 8.1; đặt
                hệ số của trình chạy từng bước về "như bản in" để xem cặp bị đổi chỗ biến mọi đầu ra thành màu đỏ.
              </p>
            }
          >
            <p style={{ margin: 0 }}>
              One more, in the prose on p. 196: shared memory falls "to one third of that by the kernel in Fig. 8.12". The
              sentence describes Fig. 8.12, so the comparison is with Fig. 8.10. The corrected kernel above passes lab
              8.1's tests; set the stepper's coefficients to "as printed" to watch the swapped pair turn every output red.
            </p>
          </Translated>
        </div>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- listings from the lab
function partOf(file: string | null, marker: RegExp): string | null {
  if (file == null) return null;
  const lines = file.split('\n');
  const at = lines.findIndex((l) => marker.test(l));
  return at < 0 ? file : lines.slice(at).join('\n');
}

/**
 * Status of a listing lab 8.1 runs, in the labs page's words (LabKit's solutionTabStatus):
 * the shared planes kernel is run by the lab's tests, the second listing is the lab's solution.
 */
function labStatusText(reports: Reports, backend: 'cuda' | 'metal', shared = false): string {
  const st = solutionTabStatus(reports, 'ch08/lab1_stencil3d', backend === 'cuda' ? 'kernel.cu' : 'kernel.metal');
  return shared ? tr(`tested with lab 8.1: ${st}`, `được kiểm thử cùng lab 8.1: ${st}`) : st;
}

function PlanesListing() {
  const { reports } = useReports();
  const cu = useLabFile('ch08/shared/planes.cu');
  const metal = useLabFile('ch08/shared/planes.metal');
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: cu ?? '// loading labs/ch08/shared/planes.cu', tiers: LAB_TIERS, status: labStatusText(reports, 'cuda', true) },
        { label: 'Metal', code: metal ?? '// loading labs/ch08/shared/planes.metal', tiers: LAB_TIERS, status: labStatusText(reports, 'metal', true) },
      ]}
      caption={
        <Translated
          vi={
            <>
              Hình 8.10 như lab 8.1 cung cấp, được nạp từ <code>labs/ch08/shared/</code>. Nó khác với trang sách ở chỗ quan trọng: c3
              nhân với láng giềng y − 1 và c4 với láng giềng y + 1, như trong Hình 8.6 và 8.8. Hình 8.10 bản in để chúng theo chiều
              ngược lại. Bản build HIP dùng nguyên tệp CUDA; đường đó chưa được biên dịch trên phần cứng AMD.
            </>
          }
        >
          <>
            Fig. 8.10 as lab 8.1 provides it, loaded from <code>labs/ch08/shared/</code>. It differs from the page where
            it matters: c3 multiplies the y − 1 neighbour and c4 the y + 1 neighbour, as in Figs. 8.6 and 8.8. The
            printed Fig. 8.10 has them the other way round. The HIP build uses the CUDA file unchanged; that path has not been compiled on AMD hardware yet.
          </>
        </Translated>
      }
    />
  );
}

function RegTiledListing() {
  const { reports } = useReports();
  const cu = partOf(useLabFile('ch08/lab1_stencil3d/solution/kernel.cu'), /Part 2/);
  const metal = partOf(useLabFile('ch08/lab1_stencil3d/solution/kernel.metal'), /^\/\/ Fig\. 8\.12/);
  return (
    <div id="s85-fig812" className="wide" style={{ scrollMarginTop: '1rem' }}>
      <Code
        tabs={[
          { label: 'CUDA', code: cu ?? '// loading labs/ch08/lab1_stencil3d/solution/kernel.cu', tiers: LAB_TIERS, status: labStatusText(reports, 'cuda') },
          { label: 'Metal', code: metal ?? '// loading labs/ch08/lab1_stencil3d/solution/kernel.metal', tiers: LAB_TIERS, status: labStatusText(reports, 'metal') },
        ]}
        caption={
          <Translated
            vi={
              <>
                Hình 8.12, đã sửa: nửa sau lời giải tham chiếu của lab 8.1, được nạp từ{' '}
                <code>labs/ch08/lab1_stencil3d/solution/</code>. Hãy tự viết nó trong{' '}
                <a href={href('ch08', 'labs')}>lab</a> trước khi nghiên cứu. Chú thích của nó liệt kê những chỗ khác với trang sách;
                phần săn lỗi bên dưới nói về những dòng đó.
              </>
            }
          >
            <>
              Fig. 8.12, corrected: the second half of lab 8.1's reference solution, loaded from{' '}
              <code>labs/ch08/lab1_stencil3d/solution/</code>. Write it yourself in the{' '}
              <a href={href('ch08', 'labs')}>lab</a> before studying it. Its comment lists where it differs from the
              page; the hunt below is about those lines.
            </>
          </Translated>
        }
      />
    </div>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS85: Record<string, Question> = {
  'ch08/8.5/q-barrier2': {
    kind: 'choice',
    id: 'ch08/8.5/q-barrier2',
    prompt: (
      <Translated vi={<>Bỏ <code>__syncthreads()</code> thứ hai khỏi Hình 8.10. Điều gì có thể sai?</>}>
        <>Remove the second <code>__syncthreads()</code> from Fig. 8.10. What can go wrong?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Một luồng đọc điểm của láng giềng trong inCurr_s trước khi láng giềng nạp nó.">{'A thread reads a neighbour\'s point of inCurr_s before the neighbour has loaded it.'}</Translated>,
        why: <Translated vi="Đó là điều rào chắn 1 ngăn chặn, và rào chắn 1 vẫn còn đó.">{'That is what barrier 1 prevents, and barrier 1 is still there.'}</Translated>,
      },
      {
        label: <Translated vi="Một luồng nhanh ghi đè inCurr_s bằng mặt phẳng kế tiếp trong khi một láng giềng vẫn đang đọc mặt phẳng hiện tại.">{'A fast thread overwrites inCurr_s with the next plane while a neighbour is still reading the current one.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Không có gì: mỗi luồng chỉ dịch các ô nhớ của chính nó, nên không có hai luồng nào ghi cùng một phần tử.">{'Nothing: each thread shifts only its own slots, so no two threads write the same element.'}</Translated>,
        why: <Translated vi="Mỗi luồng chỉ ghi các ô nhớ của chính nó, nhưng các láng giềng đọc những ô đó. Sở hữu một ô nhớ không làm nó trở thành riêng tư.">{'Each thread writes only its own slots, but its neighbours read those slots. Owning a slot does not make it private.'}</Translated>,
      },
      {
        label: <Translated vi="Chỉ các luồng ở vành bị ảnh hưởng: chúng bỏ qua phép tính và đến bước dịch sớm, nhưng sau đó không ai đọc các ô nhớ của chúng.">{'Only the ring threads are affected: they skip the compute and reach the shift early, but nobody reads their slots after that.'}</Translated>,
        why: <Translated vi="Các luồng ở vành là những luồng dễ ghi đè sớm nhất, chính vì chúng không có gì để tính; thiệt hại rơi vào các luồng bên trong đọc ô nhớ của chúng.">{'The ring threads are the likeliest to overwrite early, precisely because they have nothing to compute; the damage lands on the inner threads that read their slots.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Bước dịch ghi đè inCurr_s, và inCurr_s được tối đa bốn luồng khác đọc cho mặt phẳng này. Một luồng đến bước dịch trước sẽ
            phá hủy một giá trị mà láng giềng vẫn cần: một xung đột ghi sau đọc, phụ thuộc giả của chương 5. Trình chạy từng bước cho
            thấy điều đó: bỏ chọn rào chắn 2 và nhấn "Đến lỗi đầu tiên".
          </p>
        }
      >
        <p>
          The shift overwrites inCurr_s, and inCurr_s is read by up to four other threads for this plane. A thread that
          reaches the shift first destroys a value a neighbour still needs: a write-after-read conflict, the false
          dependence of chapter 5. The stepper shows it: untick barrier 2 and press "Go to the first failure".
        </p>
      </Translated>
    ),
    source: 'ch. 8 §8.5, pp. 194–195; ch. 5 §5.4 for the two kinds of dependence',
    vi: { source: 'ch. 8 §8.5, pp. 194–195; ch. 5 §5.4 về hai loại phụ thuộc' },
  },
  'ch08/8.5/q-uniform-barrier': {
    kind: 'choice',
    id: 'ch08/8.5/q-uniform-barrier',
    prompt: (
      <Translated
        vi={
          <>
            Để tiết kiệm công, một đồng nghiệp thêm <code>if (j &gt;= N || k &gt;= N) return;</code> ở đầu kernel phân ô thanh ghi,
            để các luồng nằm ngoài lưới rời đi ngay. Phát biểu nào đúng?
          </>
        }
      >
        <>
          To save work, a colleague adds <code>if (j &gt;= N || k &gt;= N) return;</code> at the top of the
          register-tiled kernel, so threads that sit beyond the grid leave at once. Which statement is right?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Đầu ra trở nên sai: các luồng bên trong giờ đọc các ô vùng biên mà những luồng đã rời đi lẽ ra phải điền.">{'The output becomes wrong: inner threads now read halo slots that the departed threads should have filled.'}</Translated>,
        why: <Translated vi="Một luồng bên trong chỉ tính khi 1 ≤ j ≤ N − 2, nên các láng giềng ở j ± 1 của nó nằm trong lưới và không bao giờ return. Không ô nhớ nào nó đọc thuộc về một luồng đã rời đi.">{'An inner thread computes only when 1 ≤ j ≤ N − 2, so its neighbours at j ± 1 are inside the grid and never return. No slot it reads belongs to a thread that left.'}</Translated>,
      },
      {
        label: <Translated vi="Trên CUDA khối bị treo ở rào chắn 1, chờ các luồng đã return; trên Apple silicon và AMD nó chạy đúng.">{'On CUDA the block hangs at barrier 1, waiting for threads that returned; on Apple silicon and AMD it runs correctly.'}</Translated>,
        why: <Translated vi="NVIDIA ghi nhận rằng một rào chắn chờ “mọi luồng chưa thoát”, nên CUDA không chờ một luồng đã return, và Apple nói điều tương tự cho Apple silicon. AMD không ghi nhận quy tắc nào, nên không gì bảo đảm nửa về AMD.">{'NVIDIA documents that a barrier waits for "all non-exited threads", so CUDA does not wait for a thread that returned, and Apple says the same for Apple silicon. AMD documents no rule either way, so nothing promises the AMD half.'}</Translated>,
      },
      {
        label: <Translated vi="Đầu ra đúng trên CUDA và trên Apple silicon, nơi một luồng đã thoát không còn được tính ở rào chắn; AMD không ghi nhận quy tắc như vậy, nên nó không khả chuyển.">{'The output is right on CUDA and on Apple silicon, where an exited thread no longer counts at a barrier; AMD documents no such rule, so it is not portable.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Ổn trên cả ba backend, vì một luồng không có việc thì không có gì để đồng bộ.">{'It is fine on all three backends, since a thread with no work has nothing to synchronise with.'}</Translated>,
        why: (
          <Translated
            vi={
              <>
                Ổn ở nơi điều đó được ghi nhận, mà không phải ở đâu cũng vậy. Và một điều kiện chặn đặt muộn hơn lại là chuyện khác: một{' '}
                <code>continue</code> bên trong bỏ qua rào chắn 2 trong khi luồng vẫn còn sống là hành vi không xác định trên CUDA và bị
                đặc tả của Metal cấm.
              </>
            }
          >
            <>
              Fine where it is documented, which is not everywhere. And a guard placed later is a different matter: an
              inner <code>continue</code> that skips barrier 2 while the thread stays alive is undefined on CUDA and
              forbidden by Metal's specification.
            </>
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Đây lại là trường hợp 4 trong phần săn lỗi rào chắn của chương 4. Quy tắc của sách (mọi luồng của khối đến được mọi rào
              chắn, hoặc không luồng nào đến) là cách hiểu an toàn; tài liệu của NVIDIA và Apple cho phép các luồng đã thoát, còn tài
              liệu của AMD không nói gì. Các kernel của lab giữ mọi luồng trong vòng lặp và chỉ chặn các lần nạp và phép tính, điều hợp
              lệ ở mọi nơi.
            </p>
            <p>
              Điều không bao giờ hợp lệ là một luồng vẫn còn sống mà bỏ qua một rào chắn mà các láng giềng của nó đến, chẳng hạn{' '}
              <code>if (!active) continue;</code> đặt trước rào chắn 2. CUDA gọi đó là không xác định, và đặc tả của Metal cấm rõ ràng
              điều đó với các rào chắn bên trong vòng lặp. Không thể trông cậy vào trình giả lập để bắt nó: nó coi hai lời gọi rào chắn
              khác nhau tình cờ khớp cặp là một.
            </p>
          </>
        }
      >
        <>
          <p>
            This is case 4 of chapter 4's barrier hunt again. The book's rule (every thread of the block reaches every
            barrier, or none does) is the safe reading; NVIDIA's and Apple's documents allow threads that have exited,
            and AMD's say nothing either way. The lab's kernels keep every thread in the loop and guard only the loads
            and the compute, which is legal everywhere.
          </p>
          <p>
            What is never legal is a thread that stays alive and skips a barrier its neighbours reach, for instance{' '}
            <code>if (!active) continue;</code> placed before barrier 2. CUDA calls that undefined, and Metal's
            specification forbids it in so many words for barriers inside loops. The emulator cannot be relied on to
            catch it: it treats two different barrier calls that happen to pair up as one.
          </p>
        </>
      </Translated>
    ),
    source:
      'notes/today/ch04-facts.md A6, B3, C3 (checked 2026-11-20); ch. 4 §4.3 bug hunt; labs/README.md for what the emulator models',
    vi: {
      source:
        'notes/today/ch04-facts.md A6, B3, C3 (kiểm tra 2026-11-20); ch. 4 §4.3 săn lỗi; labs/README.md về những gì trình giả lập mô hình hóa',
    },
  },
  'ch08/8.5/q-ex1d': {
    kind: 'numeric',
    id: 'ch08/8.5/q-ex1d',
    prompt: (
      <Translated
        vi={
          <>
            Một lưới 120 × 120 × 120 (kể cả biên) được quét bằng kernel của Hình 8.10, khối 32 × 32, mỗi khối tiến qua{' '}
            <code>OUT_TILE_DIM</code> mặt phẳng như đã viết mã. Lần khởi chạy cần bao nhiêu khối?
          </>
        }
      >
        <>
          A 120 × 120 × 120 grid (boundary included) is swept with the kernel of Fig. 8.10, 32 × 32 blocks, each block
          marching through <code>OUT_TILE_DIM</code> planes as coded. How many blocks does the launch need?
        </>
      </Translated>
    ),
    answer: 64,
    explain: (
      <Translated
        vi={
          <p>
            OUT_TILE_DIM = 30. Theo x và y các khối cách nhau 30, nên ceil(120 / 30) = 4; theo z mỗi khối phủ 30 mặt phẳng, nên lại
            là 4. 4 × 4 × 4 = 64 khối. Cách phân ô 8 × 8 × 8 của Hình 8.8 cần 8,000 khối cho cùng lưới (bài tập 1c).
          </p>
        }
      >
        <p>
          OUT_TILE_DIM = 30. In x and y blocks step by 30, so ceil(120 / 30) = 4; in z each block covers 30 planes, so 4
          again. 4 × 4 × 4 = 64 blocks. The 8 × 8 × 8 tiling of Fig. 8.8 needed 8,000 for the same grid (exercise 1c).
        </p>
      </Translated>
    ),
    source: 'ch. 8 exercise 1d, p. 198',
    vi: { source: 'ch. 8 bài tập 1d, p. 198' },
  },
  'ch08/8.5/q-ex2a': {
    kind: 'numeric',
    id: 'ch08/8.5/q-ex2a',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel stencil 7 điểm có gộp việc theo z dùng khối 32 × 32 và hệ số gộp 16: mỗi khối tính 16 mặt phẳng đầu ra. Một
            khối bên trong nạp bao nhiêu phần tử đầu vào trong suốt vòng đời của nó?
          </>
        }
      >
        <>
          A 7-point stencil kernel with coarsening in z uses 32 × 32 blocks and a coarsening factor of 16: each block
          computes 16 output planes. How many input elements does one interior block load over its lifetime?
        </>
      </Translated>
    ),
    answer: 18432,
    explain: (
      <Translated
        vi={
          <p>
            Một mặt phẳng 32 × 32 cho mỗi mặt phẳng đầu ra, cộng mặt phẳng trước mặt phẳng đầu tiên và mặt phẳng sau mặt phẳng cuối:
            32 × 32 × (16 + 2) = 18,432. Hai mặt phẳng được nạp trước vòng lặp và mỗi vòng lặp nạp một.
          </p>
        }
      >
        <p>
          One plane of 32 × 32 per output plane, plus the plane before the first and the plane after the last: 32 × 32 ×
          (16 + 2) = 18,432. Two planes are loaded before the loop and one per iteration.
        </p>
      </Translated>
    ),
    source: 'ch. 8 exercise 2a, p. 198',
    vi: { source: 'ch. 8 bài tập 2a, p. 198' },
  },
  'ch08/8.5/q-ex2b': {
    kind: 'numeric',
    id: 'ch08/8.5/q-ex2b',
    prompt: (
      <Translated vi={<>Cùng khối bên trong đó, 32 × 32 luồng và 16 mặt phẳng đầu ra. Nó tính bao nhiêu phần tử đầu ra trong suốt vòng đời?</>}>
        <>Same interior block, 32 × 32 threads and 16 output planes. How many output elements does it compute over its lifetime?</>
      </Translated>
    ),
    answer: lifetimeCounts(32, 16, 1).outputs,
    unit: 'elements',
    explain: (
      <Translated
        vi={
          <p>
            Vành ngoài của mặt phẳng 32 × 32 chỉ nạp, nên mỗi mặt phẳng đầu ra có 30 × 30 = 900 điểm, và có 16 mặt phẳng như vậy: 30
            × 30 × 16 = 14,400. Mục "Số lần nạp và lưu mỗi khối" của trình khám phá tấm ở t = 32 và D = 16 cho thấy cùng cặp số,
            18,432 và 14,400.
          </p>
        }
      >
        <p>
          The ring of the 32 × 32 plane only loads, so each output plane is 30 × 30 = 900 points, and there are 16 of them:
          30 × 30 × 16 = 14,400. The slab explorer's "Loads and stores per block" at t = 32 and D = 16 shows the same
          pair, 18,432 and 14,400.
        </p>
      </Translated>
    ),
    source: 'ch. 8 exercise 2b, p. 199',
    vi: { unit: 'phần tử', source: 'ch. 8 bài tập 2b, p. 199' },
  },
  'ch08/8.5/q-ex2c': {
    kind: 'numeric',
    id: 'ch08/8.5/q-ex2c',
    prompt: (
      <Translated
        vi={
          <>
            Cùng khối đó. Cường độ tính toán của nó tính bằng FLOP/B là bao nhiêu, nếu đếm mọi lần nạp và lưu giá trị 4 byte như công
            thức của sách ở p. 192?
          </>
        }
      >
        <>
          Same block. What is its arithmetic intensity in FLOP/B, counting every load and store of 4-byte values as the
          book's formula on p. 192 does?
        </>
      </Translated>
    ),
    answer: 1.43,
    tol: 0.01,
    unit: 'FLOP/B',
    explain: (
      <Translated
        vi={
          <p>
            13 × 14,400 / (4 × (18,432 + 14,400)) = 187,200 / 131,328 = 1.43. Nông hơn 30 mặt phẳng như đã viết mã (1.47), nên thấp
            hơn một chút: hai mặt phẳng z thêm vào được chia cho ít đầu ra hơn.
          </p>
        }
      >
        <p>
          13 × 14,400 / (4 × (18,432 + 14,400)) = 187,200 / 131,328 = 1.43. Shallower than the as-coded 30 planes (1.47),
          so a little lower: the two extra z planes are spread over fewer outputs.
        </p>
      </Translated>
    ),
    source: 'ch. 8 exercises 2b and 2c, p. 199',
    vi: { source: 'ch. 8 bài tập 2b và 2c, p. 199' },
  },
  'ch08/8.5/q-who-reads': {
    kind: 'choice',
    id: 'ch08/8.5/q-who-reads',
    prompt: (
      <Translated vi={<>Trong Hình 8.10, phần tử của những mảng chia sẻ nào được nhiều hơn một luồng đọc?</>}>
        <>In Fig. 8.10, the elements of which shared arrays are read by more than one thread?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Cả ba: mỗi mặt phẳng chứa các giá trị mà cả khối cùng nạp.">{'All three: every plane holds values the whole block loaded together.'}</Translated>, why: <Translated vi="Hãy nhìn các chỉ số: inPrev_s và inNext_s chỉ bao giờ được đọc ở [ty][tx].">{'Look at the indices: inPrev_s and inNext_s are only ever read at [ty][tx].'}</Translated> },
      { label: <Translated vi="Chỉ inCurr_s, qua bốn láng giềng x-y của mỗi luồng.">{'Only inCurr_s, through each thread\'s four x-y neighbours.'}</Translated>, correct: true },
      { label: <Translated vi="inPrev_s và inNext_s, vì mọi luồng đều cần các mặt phẳng trước và sau.">{'inPrev_s and inNext_s, since every thread needs the planes before and after.'}</Translated>, why: <Translated vi="Mỗi luồng cần điểm của chính nó trong chúng, ở [ty][tx], và không cần của ai khác.">{'Every thread needs its own point of them, at [ty][tx], and nobody else\'s.'}</Translated> },
      { label: <Translated vi="Không mảng nào: mỗi luồng chỉ đọc những gì nó đã nạp.">{'None: each thread reads only what it loaded.'}</Translated>, why: <Translated vi="Bốn láng giềng x-y trong inCurr_s do bốn luồng khác nạp.">{'The four x-y neighbours in inCurr_s were loaded by four other threads.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            inPrev_s[ty][tx] và inNext_s[ty][tx] chỉ do luồng có cùng (tx, ty) đọc và không ai khác, nên chúng chưa bao giờ cần được
            chia sẻ. Chỉ inCurr_s được đọc ở [ty ± 1][tx] và [ty][tx ± 1]. Quan sát đó chính là toàn bộ phân ô thanh ghi.
          </p>
        }
      >
        <p>
          inPrev_s[ty][tx] and inNext_s[ty][tx] are read by the thread with the same (tx, ty) and nobody else, so they
          never needed to be shared. Only inCurr_s is read at [ty ± 1][tx] and [ty][tx ± 1]. That observation is all
          register tiling is.
        </p>
      </Translated>
    ),
    source: 'ch. 8 §8.6, p. 196',
  },
  'ch08/8.5/q-global-same': {
    kind: 'choice',
    id: 'ch08/8.5/q-global-same',
    prompt: (
      <Translated vi={<>Phân ô thanh ghi chuyển hai trong ba mặt phẳng ra khỏi bộ nhớ chia sẻ vào thanh ghi. FLOP/B của kernel so với bộ nhớ toàn cục thay đổi thế nào?</>}>
        <>Register tiling moves two of the three planes out of shared memory into registers. What happens to the kernel's FLOP/B against global memory?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Nó tăng gấp ba: khối giờ giữ một mặt phẳng trong bộ nhớ chia sẻ thay vì ba.">{'It triples: the block now keeps one plane in shared memory instead of three.'}</Translated>, why: <Translated vi="Các mặt phẳng di chuyển giữa hai loại bộ nhớ trên chip. Bộ nhớ toàn cục không hề thấy chúng di chuyển.">{'The planes moved between two kinds of on-chip storage. Global memory never saw them move.'}</Translated> },
      { label: <Translated vi="Nó tăng một chút, vì thanh ghi có độ trễ thấp hơn và băng thông cao hơn bộ nhớ chia sẻ.">{'It rises a little, because registers have lower latency and higher bandwidth than shared memory.'}</Translated>, why: <Translated vi="Tốc độ truy cập không thuộc về tỉ lệ. Tỉ lệ đếm số byte di chuyển đến và từ bộ nhớ toàn cục.">{'Speed of access is not part of the ratio. The ratio counts bytes moved to and from global memory.'}</Translated> },
      { label: <Translated vi="Không gì cả: cùng các phần tử được nạp từ bộ nhớ toàn cục, mỗi phần tử một lần.">{'Nothing: the same elements are loaded from global memory, once each.'}</Translated>, correct: true },
      { label: <Translated vi="Nó giảm, vì mặt phẳng hiện tại giờ được ghi hai lần, vào thanh ghi và vào bộ nhớ chia sẻ.">{'It falls, because the current plane is now written twice, to a register and to shared memory.'}</Translated>, why: <Translated vi="Bản chép thêm đi từ thanh ghi sang bộ nhớ chia sẻ, trên chip. Lưu lượng toàn cục không bị động đến.">{'The extra copy goes from a register to shared memory, on chip. Global traffic is untouched.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mọi phần tử vẫn được nạp một lần mỗi khối và mọi đầu ra được lưu một lần, nên FLOP/B giữ nguyên, 1.47 ở t = 32 như đã
            viết mã. Điều thay đổi nằm trên chip: số lần đọc chia sẻ mỗi đầu ra giảm từ 7 xuống 4, các lần sao chép mặt phẳng biến mất, và
            bộ nhớ chia sẻ mỗi khối còn một phần ba. Các bộ đếm của trình chạy từng bước cho thấy cả hai nửa.
          </p>
        }
      >
        <p>
          Every element is still loaded once per block and every output stored once, so FLOP/B is the same, 1.47 at
          t = 32 as coded. What changed is on chip: shared reads per output fall from 7 to 4, the plane copies are gone,
          and shared memory per block is a third. The stepper's counters show both halves.
        </p>
      </Translated>
    ),
    source: 'ch. 8 §8.6, pp. 197–198',
  },
  'ch08/8.5/q-shared-bytes': {
    kind: 'numeric',
    id: 'ch08/8.5/q-shared-bytes',
    prompt: (
      <Translated vi={<>Một khối 32 × 32 của Hình 8.10 dùng bao nhiêu byte bộ nhớ chia sẻ, với giá trị 4 byte?</>}>
        <>How many bytes of shared memory does one 32 × 32 block of Fig. 8.10 use, with 4-byte values?</>
      </Translated>
    ),
    answer: 12288,
    unit: 'B',
    explain: (
      <Translated
        vi={
          <p>
            Ba mặt phẳng: 3 × 32 × 32 × 4 = 12,288 B, "12 KB" của sách. Với phân ô thanh ghi chỉ còn inCurr_s: 32 × 32 × 4 = 4,096 B
            (bài tập 2e).
          </p>
        }
      >
        <p>
          Three planes: 3 × 32 × 32 × 4 = 12,288 B, the book's "12 KB". With register tiling only inCurr_s is left: 32 ×
          32 × 4 = 4,096 B (exercise 2e).
        </p>
      </Translated>
    ),
    source: 'ch. 8 §8.5, p. 195; exercises 2d and 2e, p. 199',
    vi: { source: 'ch. 8 §8.5, p. 195; bài tập 2d và 2e, p. 199' },
  },
  'ch08/8.5/q-launch-fail': {
    kind: 'choice',
    id: 'ch08/8.5/q-launch-fail',
    prompt: (
      <Translated
        vi={
          <>
            Kernel phân ô thanh ghi của bạn biên dịch ra 72 thanh ghi mỗi luồng, không có launch bounds. Bạn khởi chạy nó trên CUDA
            với khối 32 × 32. Điều gì xảy ra?
          </>
        }
      >
        <>
          Your register-tiled kernel compiles to 72 registers per thread, with no launch bounds. You launch it on CUDA
          with 32 × 32 blocks. What happens?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Nó chạy, nhưng chậm hơn, vì ít khối hơn vừa trên một SM.">{'It runs, but more slowly, because fewer blocks fit on an SM.'}</Translated>, why: <Translated vi="Đó là điều xảy ra với số thanh ghi vừa phải. Ở đây riêng một khối đã không vừa.">{'That is what happens at moderate register counts. Here one block alone does not fit.'}</Translated> },
      {
        label: <Translated vi="Lần khởi chạy thất bại: 1,024 luồng × 72 thanh ghi nhiều hơn 64 K thanh ghi mà một khối được dùng.">{'The launch fails: 1,024 threads × 72 registers is more than the 64 K registers a block may use.'}</Translated>,
        correct: true,
      },
      { label: <Translated vi="Trình biên dịch cho các thanh ghi vượt giới hạn tràn sang bộ nhớ cục bộ khi lần khởi chạy được thực hiện.">{'The compiler spills the registers above the limit to local memory when the launch is made.'}</Translated>, why: <Translated vi="Thanh ghi do trình biên dịch cấp phát, đó là lý do --resource-usage báo cáo chúng lúc build. Không có launch bounds, nó không biết khối sẽ lớn đến vậy.">{'Registers are allocated by the compiler, which is why --resource-usage reports them at build time. Without launch bounds it did not know the block would be this large.'}</Translated> },
      { label: <Translated vi="Runtime thu nhỏ khối còn 512 luồng để nó vừa.">{'The runtime shrinks the block to 512 threads so that it fits.'}</Translated>, why: <Translated vi="NVIDIA ghi nhận kết quả: khi một khối cần nhiều thanh ghi hơn số có sẵn, lần khởi chạy thất bại.">{'NVIDIA documents the outcome: when a block needs more registers than are available, the launch fails.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Một khối có thể dùng tối đa 64 K thanh ghi trên mọi compute capability mà NVIDIA liệt kê; ở 1,024 luồng, đó là tối đa
              64 thanh ghi mỗi luồng, và ít hơn nếu việc cấp phát làm tròn lên. Vượt quá mức đó, lần khởi chạy thất bại. Cách sửa được
              ghi nhận là <code>__launch_bounds__(1024)</code>, khiến nvcc gói vừa 64 thanh ghi mỗi luồng, tràn sang bộ nhớ cục bộ
              (local memory) nếu buộc phải vậy.
            </p>
            <p>
              HIP thì khác: không có launch bounds, nó biên dịch cho khối lớn nhất của thiết bị, nên nó giới hạn số thanh ghi (và có
              thể tràn) ngay từ đầu thay vì thất bại lúc khởi chạy. Trên Metal, <code>maxTotalThreadsPerThreadgroup</code> của
              pipeline{' '}
              có thể trả về dưới 1,024, và phần khởi chạy phải kiểm tra nó.
            </p>
          </>
        }
      >
        <>
          <p>
            A block may use at most 64 K registers on every compute capability NVIDIA lists; at 1,024 threads that is 64
            registers per thread at most, and fewer if allocation rounds up. Above that the launch fails. The documented
            fix is <code>__launch_bounds__(1024)</code>, which makes nvcc fit 64 registers per thread, spilling to local
            memory if it must.
          </p>
          <p>
            HIP differs: without launch bounds it compiles for the device's largest block, so it limits registers (and may
            spill) up front instead of failing at launch. On Metal the pipeline's <code>maxTotalThreadsPerThreadgroup</code>{' '}
            may come back below 1,024, and the launcher has to check it.
          </p>
        </>
      </Translated>
    ),
    source: 'notes/today/ch08-facts.md A4, B4, C1 (checked 2026-11-20); ch04-facts A12, A13',
    vi: { source: 'notes/today/ch08-facts.md A4, B4, C1 (kiểm tra 2026-11-20); ch04-facts A12, A13' },
  },
  'ch08/8.5/q-indexed-window': {
    kind: 'choice',
    id: 'ch08/8.5/q-indexed-window',
    prompt: (
      <Translated
        vi={
          <>
            <code>ssm_conv</code> trên CUDA của llama.cpp giữ cửa sổ của nó dưới dạng <code>float x[d_conv]</code> và đọc{' '}
            <code>x[(i + j) % d_conv]</code>. Sách giữ ba biến vô hướng có tên và chép chúng. Dạng nào bị quy tắc được ghi nhận của
            NVIDIA đánh dấu là ứng viên cho bộ nhớ cục bộ, và vì sao điều đó quan trọng?
          </>
        }
      >
        <>
          llama.cpp's CUDA <code>ssm_conv</code> keeps its window as <code>float x[d_conv]</code> and reads{' '}
          <code>x[(i + j) % d_conv]</code>. The book keeps three named scalars and copies them. Which form does NVIDIA's
          documented rule flag as a candidate for local memory, and why does it matter?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Ba biến vô hướng, vì mỗi lần sao chép giữa chúng tốn một lệnh và một thanh ghi mà trình biên dịch phải giữ.">{'The three scalars, because each copy between them costs an instruction and a register the compiler must keep.'}</Translated>, why: <Translated vi="Chép giữa các thanh ghi rẻ và để các giá trị nằm trong thanh ghi. Quy tắc là về cách đánh chỉ số.">{'Copies between registers are cheap and leave the values in registers. The rule is about indexing.'}</Translated> },
      {
        label: <Translated vi="Mảng được đánh chỉ số lúc chạy: bộ nhớ cục bộ nằm trong bộ nhớ thiết bị.">{'The array indexed at run time: local memory lives in device memory.'}</Translated>,
        correct: true,
      },
      { label: <Translated vi="Không dạng nào: mọi thứ khai báo trong kernel mà không có __shared__ đều nằm trong thanh ghi, dù là mảng hay không.">{'Neither: anything declared inside a kernel without __shared__ lives in a register, array or not.'}</Translated>, why: <Translated vi="Cả chương 5 và chương 6 đều cho thấy ngoại lệ: các mảng mà trình biên dịch không thể đánh chỉ số bằng hằng.">{'Chapter 5 and chapter 6 both show the exception: arrays the compiler cannot index with constants.'}</Translated> },
      { label: <Translated vi="Cả hai: mọi cửa sổ theo từng luồng, mảng hay biến vô hướng, đều được đặt trong bộ nhớ cục bộ.">{'Both: any per-thread window, array or scalars, is placed in local memory.'}</Translated>, why: <Translated vi="Biến vô hướng và mảng nhỏ chỉ được đánh chỉ số bằng hằng được nâng lên thanh ghi; đó là lý do unroll quan trọng (ch. 6 §6.6).">{'Scalars and small arrays indexed only with constants are promoted to registers; that is why unrolling matters (ch. 6 §6.6).'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            NVIDIA liệt kê "các mảng mà nó không thể xác định là được đánh chỉ số bằng các đại lượng hằng" đầu tiên trong số các biến
            dễ rơi vào bộ nhớ cục bộ, vốn có độ trễ và băng thông của bộ nhớ thiết bị. Việc nvcc có vẫn giữ cửa sổ của llama.cpp trong
            thanh ghi hay không chưa được kiểm tra; <code>--resource-usage</code> sẽ cho biết, và unroll vòng lặp để mọi chỉ số đều
            là hằng sẽ xóa bỏ câu hỏi.
          </p>
        }
      >
        <p>
          NVIDIA lists "arrays for which it cannot determine that they are indexed with constant quantities" first among
          the variables likely to land in local memory, which has device-memory latency and bandwidth. Whether nvcc keeps
          llama.cpp's window in registers anyway was not checked; <code>--resource-usage</code> would tell, and unrolling
          the loop so every index is a constant removes the question.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch08-facts.md A5, D1 (checked 2026-11-20); ch. 6 §6.6, pp. 145–146',
    vi: { source: 'notes/today/ch08-facts.md A5, D1 (kiểm tra 2026-11-20); ch. 6 §6.6, pp. 145–146' },
  },
  'ch08/8.5/q-diverge32': {
    kind: 'numeric',
    id: 'ch08/8.5/q-diverge32',
    prompt: (
      <Translated
        vi={
          <>
            Một khối 32 × 32 bên trong của kernel phân ô thanh ghi đến điều kiện chặn phép tính. Trên phần cứng NVIDIA, nơi một warp
            gồm 32 luồng, bao nhiêu warp của nó có lane rảnh bên cạnh lane hoạt động?
          </>
        }
      >
        <>
          An interior 32 × 32 block of the register-tiled kernel reaches the compute guard. On NVIDIA hardware, where a
          warp is 32 threads, how many of its warps have idle lanes alongside active ones?
        </>
      </Translated>
    ),
    answer: 30,
    unit: 'warps',
    explain: (
      <Translated
        vi={
          <p>
            Với <code>blockDim.x</code> = 32, mỗi warp là một hàng ty. Hàng 0 và 31 toàn là luồng ở vành: rảnh, nhưng không bị chia.
            Mỗi hàng trong 30 hàng ở giữa có lane 0 và 31 rảnh và 30 lane hoạt động. Tổng cộng 124 trên 1,024 luồng không tính gì.
            Với wavefront rộng 64, mỗi wavefront chứa hai hàng, và cả 16 đều lẫn lộn.
          </p>
        }
      >
        <p>
          With <code>blockDim.x</code> = 32 each warp is one row ty. Rows 0 and 31 are entirely ring threads: idle, but
          not divided. Each of the 30 rows between has lanes 0 and 31 idle and 30 active. In all 124 of 1,024 threads
          compute nothing. With 64-wide wavefronts each wavefront holds two rows, and all 16 are mixed.
        </p>
      </Translated>
    ),
    source: 'ch. 8 §8.5, p. 195 (the guard); warp shapes in notes/today/ch04-facts.md A8 and ch08-facts.md B2',
    vi: { unit: 'warp', source: 'ch. 8 §8.5, p. 195 (điều kiện chặn); hình dạng warp trong notes/today/ch04-facts.md A8 và ch08-facts.md B2' },
  },
};

// ---------------------------------------------------------------- the section
export function S85() {
  return (
    <>
      <Translated
        vi={
          <p>
            Mục 8.4 đã chạm vào một bức tường. Một ô lập phương cần một luồng cho mỗi điểm đầu vào, nên một ô rộng t cần t³ luồng, và
            với 1,024 luồng trong một khối, sách chọn t = 8, nơi vùng biên (halo) giữ kernel ở 0.96 FLOP/B. Lối thoát là thôi dành một
            luồng cho mỗi điểm dọc z. Hãy giao cho mỗi luồng một cột điểm theo z, làm cho khối thành một mặt phẳng x-y duy nhất của ô
            đầu vào, và để mặt phẳng luồng đó tiến qua ô từng mặt phẳng một.
          </p>
        }
      >
        <p>
          Section 8.4 ran into a wall. A cubic tile wants one thread per input point, so a tile t wide needs t³ threads,
          and with 1,024 threads in a block the book settles on t = 8, where the halo holds the kernel to 0.96 FLOP/B. The
          way out is to stop spending a thread on every point along z. Give each thread a column of points in z, make the
          block a single x-y plane of the input tile, and let that plane of threads march through the tile one plane at
          a time.
        </p>
      </Translated>
      <Source section="8.5" pages="193" figs={tr('Fig. 8.9', 'Hình 8.9')} />
      <Translated
        vi={
          <p>
            Đây là <a href={href('ch06', '6.5')}>gộp việc cho luồng (thread coarsening) của chương 6</a>: một luồng làm tuần tự công
            việc mà nhiều luồng từng làm song song, để cắt bớt một chi phí mà bản song song phải trả. Ở đây chi phí đó là vùng biên.
            Mỗi khối nạp một vành điểm mà nó không tính, và với khối nhỏ, những điểm ít được dùng lại ấy bị các khối láng giềng nạp đi
            nạp lại. Một khối tiến theo z phủ một ô sâu hơn nhiều với cùng vành mỗi mặt phẳng, và nạp hai mặt phẳng vùng biên z một
            lần cho cả cột. Khối có t² luồng thay vì t³, nên t = 32 vừa trong 1,024.
          </p>
        }
      >
        <p>
          This is <a href={href('ch06', '6.5')}>chapter 6's thread coarsening</a>: one thread does, in sequence, work that several threads did in parallel,
          to cut an overhead that the parallel version paid. Here the overhead is the halo. Every block loads a ring of
          points it does not compute, and with small blocks those low-reuse points are loaded over and over by neighbouring
          blocks. A marching block covers a much deeper tile with the same ring per plane, and loads the two z halo planes
          once for the whole column. The block has t² threads instead of t³, so t = 32 fits in 1,024.
        </p>
      </Translated>
      <Source section="8.5 and §6.5" pages="141–143, 193, 195" />
      <Translated
        vi={
          <p>
            Hãy xem hành trình một lần trước khi đọc mã. Hoạt cảnh vẽ hình của sách: một ô rộng 6 có khối 6 × 6 luồng đi qua 4 mặt
            phẳng đầu ra.
          </p>
        }
      >
        <p>
          Watch the march once before reading any code. The animation draws the book's picture: a 6-wide tile whose block
          of 6 × 6 threads walks through 4 output planes.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Hoạt cảnh chạy khối theo bước đồng bộ (lockstep), từ cùng mô hình với trình chạy từng bước bên dưới. Không đo thời gian:
              mỗi phần được định nhịp theo thuyết minh.
            </p>
          }
        >
          <p>
            The animation runs the block in lockstep, from the same model as the stepper below. No timing: each part is
            paced for the narration.
          </p>
        </Translated>
      </Simplification>
      <ZMarchScene />

      <h2>
        <Translated vi="Kernel">The kernel</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hình 8.10 giữ ba mặt phẳng của ô trong bộ nhớ chia sẻ (shared memory): <code>inPrev_s</code>, <code>inCurr_s</code> và{' '}
            <code>inNext_s</code>. Trước vòng lặp, mỗi luồng nạp điểm của nó trong hai mặt phẳng đầu. Rồi mỗi lượt của vòng lặp làm
            năm việc. Nạp một mặt phẳng mới vào <code>inNext_s</code>. Chờ ở một rào chắn (barrier). Cho các luồng bên trong tính
            một mặt phẳng đầu ra, bốn láng giềng x-y từ <code>inCurr_s</code> và các láng giềng z từ{' '}
            <code>inPrev_s</code> và <code>inNext_s</code>. Chờ ở rào chắn thứ hai. Dịch: mỗi luồng chép điểm của chính nó từ{' '}
            <code>inCurr_s</code> vào <code>inPrev_s</code> và từ <code>inNext_s</code> vào <code>inCurr_s</code>. Khi đó hai
            trong ba mặt phẳng mà lượt tiếp theo cần đã có sẵn trên chip, nên mỗi lượt nạp một mặt phẳng.
          </p>
        }
      >
        <p>
          Fig. 8.10 keeps three planes of the tile in shared memory: <code>inPrev_s</code>, <code>inCurr_s</code> and{' '}
          <code>inNext_s</code>. Before the loop each thread loads its point of the first two. Each pass of the loop then
          does five things. Load one new plane into <code>inNext_s</code>. Wait at a barrier. Let the inner threads compute
          one output plane, the four x-y neighbours from <code>inCurr_s</code> and the z neighbours from{' '}
          <code>inPrev_s</code> and <code>inNext_s</code>. Wait at a second barrier. Shift: each thread copies its own point
          of <code>inCurr_s</code> into <code>inPrev_s</code> and of <code>inNext_s</code> into <code>inCurr_s</code>. Two of
          the three planes the next pass needs are then already on chip, so each pass loads one plane.
        </p>
      </Translated>
      <Source section="8.5" pages="194–195" figs={tr('Figs. 8.10, 8.11', 'Hình 8.10, 8.11')} />
      <PlanesListing />
      <Translated
        vi={
          <p>
            Theo <a href={href('ch05', '5.4')}>thuật ngữ của chương 5</a>, rào chắn 1 bảo vệ một phụ thuộc đọc sau ghi: một luồng sắp
            đọc các điểm của láng giềng mà những láng giềng đó vừa nạp. Rào chắn 2 bảo vệ một phụ thuộc ghi sau đọc: bước dịch ghi đè{' '}
            <code>inCurr_s</code>, nơi các láng giềng có thể vẫn đang đọc cho mặt phẳng này. Vì sao không có rào chắn thứ ba sau bước
            dịch? Ở đó mỗi luồng chỉ ghi các ô nhớ của chính nó, và lần tiếp theo có ai đó đọc ô nhớ của láng giềng là sau rào chắn 1
            của lượt kế tiếp, mà không luồng nào vượt qua cho đến khi mọi luồng đã dịch xong.
          </p>
        }
      >
        <p>
          In <a href={href('ch05', '5.4')}>chapter 5's vocabulary</a>, barrier 1 protects a read-after-write dependence: a thread is about to read
          neighbours' points that those neighbours have just loaded. Barrier 2 protects a write-after-read dependence: the
          shift overwrites <code>inCurr_s</code>, which neighbours may still be reading for this plane. Why is there no
          third barrier after the shift? Each thread writes only its own slots there, and the next time anyone reads a
          neighbour's slot is after the next pass's barrier 1, which no thread passes until every thread has finished its
          shift.
        </p>
      </Translated>
      <Beyond title={tr('Our reasoning', 'Lập luận của chúng tôi')}>
        <Translated
          vi={
            <p>Sách đưa ra hai rào chắn; lập luận rằng không cần rào chắn thứ ba là của chúng tôi, từ listing.</p>
          }
        >
          <p>The book gives the two barriers; the argument that a third is not needed is ours, from the listing.</p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Xem nó chạy">Watch it run</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trình chạy từng bước chạy thân kernel cho mọi luồng của một khối. Bắt đầu với "mỗi luồng chạy xa nhất có thể" và khối
            không chạm rìa nào, rồi đi từng bước qua luồng (tx 1, ty 1), luồng đầu tiên có tính toán. Sau đó đổi kernel sang thanh
            ghi và so sánh các bộ đếm, và cuối cùng đặt hệ số theo cách ghép in trong sách.
          </p>
        }
      >
        <p>
          The stepper runs the kernel body for every thread of one block. Start with "each thread runs as far as it can"
          and the block that touches no edge, and step through thread (tx 1, ty 1), the first one that computes. Then
          switch the kernel to registers and compare the counters, and finally set the coefficients to the pairing
          printed in the book.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi bước một thao tác ở mức mã nguồn; luồng thật xen kẽ theo từng lệnh và chạy theo warp. "Lockstep" nghĩa là mọi luồng
              hoàn tất một câu lệnh trước khi bất kỳ luồng nào bắt đầu câu tiếp theo, điều mà không khối thật nào hứa hẹn. Không đo thời
              gian, không cache. Bộ nhớ chia sẻ chưa được ghi được coi là NaN, trường hợp xấu nhất: vẽ thành ·, hiện thành ? trong một
              tổng. Mã hiển thị độ rộng ô của bài toán bạn chọn.
            </p>
          }
        >
          <p>
            One source-level operation per step; real threads interleave per instruction and run in warps. "Lockstep"
            means every thread finishes a statement before any thread starts the next, which no real block promises. No
            timing, no caches. Unwritten shared memory is treated as NaN, the worst case: drawn as ·, shown as ? in a
            sum. The code shows the tile width of the problem you pick.
          </p>
        </Translated>
      </Simplification>
      <Stepper />
      <Translated
        vi={
          <p>
            Giờ hãy bỏ chọn rào chắn 2 và nhấn <i>Đến lỗi đầu tiên</i>, rồi bỏ chọn rào chắn 1 thay vào đó. Sau đó đặt thứ tự về
            lockstep và chạy lại từng trường hợp. Bảng chạy mọi tổ hợp trên toàn lưới.
          </p>
        }
      >
        <p>
          Now untick barrier 2 and press <i>Go to the first failure</i>, then untick barrier 1 instead. Then set the
          order to lockstep and run each again. The table runs every combination over the whole grid.
        </p>
      </Translated>
      <Outcomes />
      <Ask q={questionsS85['ch08/8.5/q-barrier2']} />
      <Translated
        vi={
          <p>
            Cột lockstep che giấu cả hai lỗi chỉ vì, trong lockstep, mọi luồng hoàn tất mỗi câu lệnh trước khi bất kỳ luồng nào bắt
            đầu câu tiếp theo: mọi lần nạp của một mặt phẳng xong trước khi có ai đọc, và mọi lần đọc của nó xong trước khi bước dịch
            đầu tiên ghi đè nó. Quy tắc của chương 4 là giả định ngược lại: các luồng của một khối có thể chạy theo bất kỳ thứ tự nào
            so với nhau (p. 74). 36 luồng trong hình của sách đã trải qua hai warp của NVIDIA, còn 16 luồng của bài toán nhỏ, vừa trong
            một warp, cũng không an toàn hơn, vì NVIDIA không còn hứa hẹn lockstep ngay cả trong một warp.
          </p>
        }
      >
        <p>
          The lockstep column hides both bugs only because, in lockstep, every thread finishes each statement before any
          thread starts the next: all of a plane's loads land before anyone reads, and all of its reads are done before the
          first shift overwrites it. Chapter 4's rule is the opposite assumption: the threads of a block may run in any
          order relative to one another (p. 74). The 36 threads of the book's picture already span two NVIDIA warps, and
          the 16 of the small problem, which fit in one, are no safer, because NVIDIA no longer promises lockstep even
          within a warp.
        </p>
      </Translated>
      <Source section="4.3 and §4.5" pages="74, 81" />
      <Today checked="2026-11-20 (notes/today/ch04-facts.md A10)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              NVIDIA: từ compute capability 7.0, cơ chế lập lịch luồng độc lập cho phép các luồng của một warp xen kẽ, nên mã giả định
              một warp chạy lockstep là không hợp lệ và phải đồng bộ hóa tường minh.
            </p>
          }
        >
          <p>
            NVIDIA: from compute capability 7.0, independent thread scheduling lets the threads of one warp interleave, so
            code that assumes a warp runs in lockstep is invalid and must synchronise explicitly.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS85['ch08/8.5/q-uniform-barrier']} />

      <h2>
        <Translated vi="Các con số">The numbers</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lập luận của sách cho việc gộp việc là cường độ tính toán, đếm như ở §8.4: 13 FLOP mỗi đầu ra, mọi phần tử đầu vào của ô
            của một khối được nạp một lần, mọi đầu ra được lưu một lần, mỗi giá trị 4 byte. Hãy chốt một con số trước khi đọc tiếp.
          </p>
        }
      >
        <p>
          The book's case for coarsening is arithmetic intensity, counted as in §8.4: 13 FLOP per output, every input
          element of a block's tile loaded once, every output stored once, 4 bytes each. Commit to a number before
          reading on.
        </p>
      </Translated>
      <Source section="8.4 and §8.5" pages="192, 195" />
      <Predict
        id="ch08/8.5/p-slab-ai"
        prompt={
          <Translated
            vi={
              <>
                Hình 8.10 như đã viết mã, với khối 32 × 32: mỗi khối tiến qua 30 mặt phẳng đầu ra. Một khối bên trong đạt bao nhiêu
                FLOP/B?
              </>
            }
          >
            <>
              Fig. 8.10 as coded, with 32 × 32 blocks: each block marches through 30 output planes. What FLOP/B does one
              interior block reach?
            </>
          </Translated>
        }
        unit="FLOP/B"
      >
        {(v) => (
          <Translated
            vi={
              <>
                <p>
                  Khối nạp 32 mặt phẳng 32 × 32 (30 mặt phẳng của nó cộng một trước và một sau) và lưu 30 × 30 × 30:
                  13 × 27,000 / (4 × (32,768 + 27,000)) = <b>1.47 FLOP/B</b>.{' '}
                  {Math.abs(v - 1.47) < 0.02
                    ? 'Đúng như bạn đã có.'
                    : Math.abs(v - 1.52) < 0.02
                      ? 'Bạn có con số in trong sách, vốn không phải kernel này như đã viết mã.'
                      : 'Hãy đếm các lần nạp theo từng mặt phẳng.'}
                </p>
                <p>
                  Đó đúng là các lần nạp và lưu của một khối lập phương rộng 32, nên tỉ lệ là tỉ lệ của khối lập phương. Sách in 1.52 cho
                  t = 32 (p. 195). 1.52 là con số bạn có được khi chỉ đếm vùng biên x-y, 13 × 900 / (4 × (1,024 + 900)), tức giới hạn khi
                  tấm dày lên không giới hạn; sách không nói nó đã đếm thế nào. Cả hai phép tính là của chúng tôi (facts E2).
                </p>
              </>
            }
          >
            <>
              <p>
                The block loads 32 planes of 32 × 32 (its 30 plus one before and one after) and stores 30 × 30 × 30:
                13 × 27,000 / (4 × (32,768 + 27,000)) = <b>1.47 FLOP/B</b>.{' '}
                {Math.abs(v - 1.47) < 0.02
                  ? 'That is what you had.'
                  : Math.abs(v - 1.52) < 0.02
                    ? 'You have the figure printed in the book, which is not this kernel as coded.'
                    : 'Count the loads plane by plane.'}
              </p>
              <p>
                Those are exactly the loads and stores of a 32-wide cube, so the ratio is the cube's. The book prints 1.52
                for t = 32 (p. 195). 1.52 is what you get by counting only the x-y halo, 13 × 900 / (4 × (1,024 + 900)),
                which is the limit as the slab grows without bound; the book does not say how it counted. Both
                computations are ours (facts E2).
              </p>
            </>
          </Translated>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Trình khám phá biến độ sâu tấm thành một núm vặn riêng. Bắt đầu ở t = 32 và D = 30, rồi kéo D lên và theo dõi cả số khối
            lẫn tỉ lệ.
          </p>
        }
      >
        <p>
          The explorer makes the slab depth a separate knob. Start at t = 32 and D = 30, then drag D up and watch the
          block count as well as the ratio.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              FLOP/B là của một khối bên trong: mọi phần tử của ô được nạp một lần, mọi đầu ra được lưu một lần; không cache, không đo
              thời gian. Số khối bao gồm các khối biên trống một phần. Chỉ thanh ghi cửa sổ; trình biên dịch còn thêm nữa.
            </p>
          }
        >
          <p>
            FLOP/B is one interior block's: every tile element loaded once, every output stored once; no caches, no
            timing. The block count includes the partly empty boundary blocks. Window registers only; a compiler adds
            more.
          </p>
        </Translated>
      </Simplification>
      <SlabExplorer />
      <Translated
        vi={
          <p>
            Vượt quá độ sâu khoảng t, tỉ lệ hầu như không đổi, trong khi mỗi khối làm nhiều việc tuần tự hơn và lưới có ít khối hơn
            để trải ra trên thiết bị. Đó là cạm bẫy "gộp việc quá mức" của chương 6: gộp việc loại bỏ tính song song mà phần cứng có
            thể cần. Lựa chọn của sách, một tấm sâu xấp xỉ bằng độ rộng của ô, nằm ở chỗ đường cong đã phẳng.
          </p>
        }
      >
        <p>
          Past a depth of about t the ratio barely moves, while each block does more serial work and the grid has fewer
          blocks to spread across the device. That is the "too much coarsening" pitfall of chapter 6: coarsening removes
          parallelism the hardware may need. The book's choice, a slab about as deep as the tile is wide, sits where the curve
          has flattened.
        </p>
      </Translated>
      <Source section="6.5" pages="143" />
      <Ask q={questionsS85['ch08/8.5/q-ex1d']} label={tr('Exercise 1d', 'Bài tập 1d')} />
      <Ask q={questionsS85['ch08/8.5/q-ex2a']} label={tr('Exercise 2a', 'Bài tập 2a')} />
      <Ask q={questionsS85['ch08/8.5/q-ex2b']} label={tr('Exercise 2b', 'Bài tập 2b')} />
      <Ask q={questionsS85['ch08/8.5/q-ex2c']} label={tr('Exercise 2c', 'Bài tập 2c')} />
      <Translated
        vi={
          <p>
            Bộ nhớ chia sẻ tăng theo mặt phẳng, không theo khối lập phương: 3 × t² giá trị thay vì t³, 12,288 byte cho ba mặt phẳng
            float 32 × 32, mức mà sách gọi là hợp lý.
          </p>
        }
      >
        <p>
          Shared memory grows with the plane, not the cube: 3 × t² values instead of t³, 12,288 bytes for three 32 × 32
          planes of floats, which the book calls a reasonable level.
        </p>
      </Translated>
      <Source section="8.5" pages="195" />
      <Today checked="2026-11-20 (ch08-facts.md A6, B3, C3)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              12,288 B nằm trong giới hạn được ghi nhận của mỗi nhà cung cấp: mức 48 KB tĩnh mỗi khối của CUDA, quy tắc của HIP rằng
              một khối có thể dùng toàn bộ LDS của một compute unit, và giới hạn threadgroup 32 KB của Metal.
            </p>
          }
        >
          <p>
            12,288 B fits each vendor's documented limit: CUDA's 48 KB static allowance per block, HIP's rule that one
            block may use a compute unit's whole LDS, and Metal's 32 KB threadgroup cap.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Phân ô thanh ghi">Register tiling</Translated>
      </h2>
      <Translated
        vi={
          <p>Trước khi đọc bước tiếp theo của sách, hãy nhìn lại các chỉ số trong Hình 8.10 một lần nữa.</p>
        }
      >
        <p>Before reading the book's next step, look at the indices in Fig. 8.10 once more.</p>
      </Translated>
      <Ask q={questionsS85['ch08/8.5/q-who-reads']} />
      <Translated
        vi={
          <p>
            Chỉ <code>inCurr_s</code> là được chia sẻ theo đúng nghĩa. Mỗi phần tử của <code>inPrev_s</code> và{' '}
            <code>inNext_s</code> có một người đọc, chính luồng đã nạp nó, nên nó có thể nằm trong thanh ghi (register) của luồng đó.
            Hình 8.12 làm đúng như vậy: ba thanh ghi mỗi luồng, <code>inPrev</code>, <code>inCurr</code> và{' '}
            <code>inNext</code>, và một mặt phẳng chia sẻ. Giá trị hiện tại được giữ hai lần, trong thanh ghi cho chính luồng đó và
            trong <code>inCurr_s</code> cho bốn láng giềng của nó. Sách gọi việc giữ dữ liệu trong thanh ghi để tiết kiệm truy cập bộ
            nhớ chia sẻ hoặc toàn cục là phân ô thanh ghi.
          </p>
        }
      >
        <p>
          Only <code>inCurr_s</code> is shared in any real sense. Each element of <code>inPrev_s</code> and{' '}
          <code>inNext_s</code> has one reader, the thread that loaded it, so it can live in that thread's registers. Fig.
          8.12 does exactly that: three registers per thread, <code>inPrev</code>, <code>inCurr</code> and{' '}
          <code>inNext</code>, and one shared plane. The current value is kept twice, in a register for the thread itself
          and in <code>inCurr_s</code> for its four neighbours. The book calls keeping data in registers to save shared or
          global accesses register tiling.
        </p>
      </Translated>
      <Source section="8.6" pages="196–197" figs={tr('Fig. 8.12', 'Hình 8.12')} />
      <RegTiledListing />
      <Translated
        vi={
          <p>
            Listing in trong sách có nhiều sai sót hơn bất kỳ listing nào khác trong chương: một số là lỗi của chính listing, một số
            không khớp với các listing trước đó của chương. Hãy tìm chúng trước khi đọc phần giải thích.
          </p>
        }
      >
        <p>
          The printed listing has more slips than any other in the chapter: some are errors in the listing itself, some
          disagree with the chapter's earlier listings. Find them before reading the explanations.
        </p>
      </Translated>
      <ErratumHunt />
      <Translated
        vi={
          <p>
            Nó đem lại gì. Bộ nhớ chia sẻ mỗi khối giảm còn một phần ba, 4,096 byte ở t = 32. Số lần đọc chia sẻ mỗi đầu ra giảm từ 7
            xuống 4, vì tâm và cả hai láng giềng z giờ đến từ thanh ghi, và các lần sao chép xoay vòng mặt phẳng biến mất: việc xoay vòng
            là hai lần chuyển thanh ghi và một lần ghi giá trị hiện tại mới. Các bộ đếm của trình chạy từng bước cho thấy cả ba. Sách
            nói thanh ghi có độ trễ thấp hơn và băng thông cao hơn bộ nhớ chia sẻ, nên kernel sẽ chạy nhanh hơn.
          </p>
        }
      >
        <p>
          What it buys. Shared memory per block falls to a third, 4,096 bytes at t = 32. Shared reads per output fall from
          7 to 4, because the centre and both z neighbours now come from registers, and the copies that rotated the planes
          are gone: the rotation is two register moves and one write of the new current value. The stepper's counters show
          all three. Registers, the book says, have lower latency and higher bandwidth than shared memory, so the kernel
          should run faster.
        </p>
      </Translated>
      <Source section="8.6" pages="197" />
      <Translated
        vi={
          <p>
            Nó tốn gì. Thêm ba thanh ghi mỗi luồng, 3,072 cho mỗi khối 32 × 32, và một stencil bậc r cần một cửa sổ 2r + 1 giá trị
            (cách đếm của chúng tôi), nên bậc cao hơn tốn hơn. Nếu thiếu thanh ghi, lối thoát của sách là chuyển một số mặt phẳng trở
            lại bộ nhớ chia sẻ: sự đánh đổi giữa hai bên là thứ bạn tinh chỉnh cho từng kernel.
          </p>
        }
      >
        <p>
          What it costs. Three more registers per thread, 3,072 per 32 × 32 block, and a stencil of order r needs a window
          of 2r + 1 values (our count), so higher orders cost more. If registers run short, the book's escape hatch is to move some
          planes back to shared memory: the trade between the two is one you tune per kernel.
        </p>
      </Translated>
      <Source section="8.6" pages="197" />
      <Translated
        vi={
          <p>
            Nó để nguyên điều gì. Lưu lượng toàn cục. Cùng các phần tử được nạp mỗi cái một lần, nên FLOP/B giữ nguyên như trước.
            Phân ô thanh ghi tăng tốc phía trên chip; phía ngoài chip đã được giải quyết bằng gộp việc.
          </p>
        }
      >
        <p>
          What it leaves alone. Global traffic. The same elements are loaded once each, so FLOP/B is exactly what it was.
          Register tiling speeds up the on-chip side; the off-chip side was settled by coarsening.
        </p>
      </Translated>
      <Source section="8.6" pages="198" />
      <Ask q={questionsS85['ch08/8.5/q-global-same']} />
      <Ask q={questionsS85['ch08/8.5/q-shared-bytes']} label={tr('Exercises 2d and 2e', 'Bài tập 2d và 2e')} />

      <h2>
        <Translated vi="Những gì các nhà cung cấp ghi nhận về thanh ghi">What vendors document about registers</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Khẳng định của sách rằng thanh ghi nhanh hơn bộ nhớ chia sẻ không kèm con số nào, và các nhà cung cấp cũng không đưa ra.
            NVIDIA xếp tệp thanh ghi là nơi lưu trữ có độ trễ thấp nhất, dưới bộ nhớ chia sẻ, và nói một lần truy cập thanh ghi thường
            tốn "không thêm chu kỳ xung nhịp nào". AMD gọi thanh ghi là "cấp nhanh nhất" của hệ thống phân cấp. Apple không ghi nhận gì
            về thanh ghi cả.
          </p>
        }
      >
        <p>
          The book's claim that registers beat shared memory comes with no numbers, and the vendors do not supply any
          either. NVIDIA ranks the register file as the lowest-latency storage, below shared memory, and says a register
          access normally costs "zero extra clock cycles". AMD calls registers "the fastest level" of the hierarchy. Apple
          documents nothing about registers at all.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch08-facts.md A3, B4, C4)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>Không nhà cung cấp nào công bố con số độ trễ hay băng thông của thanh ghi hoặc bộ nhớ chia sẻ.</p>
          }
        >
          <p>No vendor publishes a register or shared-memory latency or bandwidth figure.</p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Cái bẫy mà sách bỏ qua nằm ở CUDA. Một khối có thể dùng tối đa 64 K thanh ghi, nên một khối 1,024 luồng chỉ để lại tối đa
            64 thanh ghi mỗi luồng, ít hơn nếu việc cấp phát làm tròn lên. Một kernel được biên dịch dùng nhiều hơn sẽ thất bại lúc
            khởi chạy với{' '}
            <code>cudaErrorLaunchOutOfResources</code>. <code>__launch_bounds__(1024)</code> cho nvcc biết kích thước khối, và
            khi đó nvcc gói vừa ngân sách, tràn sang bộ nhớ cục bộ nếu buộc phải vậy. HIP vốn đã biên dịch cho khối lớn nhất của
            thiết bị trừ khi được chỉ định khác, nên nó giới hạn số thanh ghi, và có thể tràn, thay vì làm lần khởi chạy thất bại; và{' '}
            <code>__launch_bounds__</code> của nó nhận một đối số thứ hai với ý nghĩa khác với của CUDA. Metal có{' '}
            <code>[[max_total_threads_per_threadgroup(n)]]</code> và, từ Metal 4,{' '}
            <code>[[required_threads_per_threadgroup(n)]]</code>; việc cái nào trong số đó thay đổi cách cấp phát thanh ghi hay
            không thì không được ghi nhận. Thứ bạn có thể quan sát ở đó là <code>maxTotalThreadsPerThreadgroup</code> của pipeline,
            mà lab 8.1 in ra và kiểm tra trước khi dispatch một nhóm 32 × 32.
          </p>
        }
      >
        <p>
          The trap the book leaves out is on CUDA. A block may use at most 64 K registers, so a 1,024-thread block leaves at
          most 64 per thread, fewer if allocation rounds up. A kernel compiled to more fails at launch with{' '}
          <code>cudaErrorLaunchOutOfResources</code>. <code>__launch_bounds__(1024)</code> tells nvcc the block size, and
          nvcc then fits the budget, spilling to local memory if it must. HIP already compiles for the device's largest
          block unless told otherwise, so it limits registers, and may spill, rather than failing the launch; and its{' '}
          <code>__launch_bounds__</code> takes a second argument with a different meaning from CUDA's. Metal has{' '}
          <code>[[max_total_threads_per_threadgroup(n)]]</code> and, from Metal 4,{' '}
          <code>[[required_threads_per_threadgroup(n)]]</code>; whether either changes register allocation is not
          documented. What you can observe there is the pipeline's <code>maxTotalThreadsPerThreadgroup</code>, which lab
          8.1 prints and checks before it dispatches a 32 × 32 group.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch08-facts.md A4, B4, C1, C5; ch04-facts B10)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Đặc tả MSL đặt thuộc tính của Metal 4 trên MTLDevice; các header lại đặt nó trên MTLComputePipelineDescriptor.
            </p>
          }
        >
          <p>
            The MSL specification places the Metal 4 property on MTLDevice; the headers have it on
            MTLComputePipelineDescriptor.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Để xem một kernel thực sự dùng gì: <code>nvcc --resource-usage</code> hoặc <code>hipcc --resource-usage</code> lúc
            build, hoặc <code>numRegs</code> từ các thuộc tính hàm lúc chạy. Metal không cung cấp số thanh ghi; trình profiler của
            Xcode hiện một bộ đếm "Occupancy" mà Apple không định nghĩa.
          </p>
        }
      >
        <p>
          To see what a kernel really uses: <code>nvcc --resource-usage</code> or <code>hipcc --resource-usage</code> at
          build time, or <code>numRegs</code> from the function attributes at run time. Metal exposes no register count;
          Xcode's profiler shows an "Occupancy" counter that Apple does not define.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch04-facts A13, B10; ch08-facts C4)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>Trên HIP, numRegs chỉ đếm thanh ghi vector.</p>
          }
        >
          <p>On HIP, numRegs counts vector registers only.</p>
        </Translated>
      </Today>
      <Ask q={questionsS85['ch08/8.5/q-launch-fail']} />

      <h2>
        <Translated vi="Giữ cửa sổ trong thanh ghi">Keep the window in registers</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách xoay cửa sổ bằng cách chép ba biến vô hướng có tên, <code>inPrev = inCurr; inCurr = inNext;</code>, để mọi truy cập
            dùng một tên cố định. Thật dễ bị cám dỗ tránh các lần sao chép bằng một mảng nhỏ và đánh chỉ số theo modulo, để mỗi mặt phẳng
            mới ghi đè ô cũ nhất. Kernel bên dưới làm đúng như vậy. Nó tính ra cùng kết quả như kernel của sách (một unit test chạy một
            mô hình vòng lặp của nó dưới ba thứ tự luồng; bản thân tệp chưa từng chạy), và theo quy tắc được ghi nhận của NVIDIA, cửa
            sổ của nó là ứng viên cho bộ nhớ cục bộ (local memory), vì trình biên dịch không thấy được chỉ số hằng trong{' '}
            <code>w[i % 3]</code>. Bộ nhớ cục bộ là bộ nhớ thiết bị. Đây là <a href={href('ch06', '6.6')}>điểm về nâng lên thanh ghi
            của chương 6</a>: chỉ số hằng, thường nhờ unroll, là thứ cho phép trình biên dịch giữ một mảng theo từng luồng trong thanh
            ghi.
          </p>
        }
      >
        <p>
          The book rotates the window by copying three named scalars, <code>inPrev = inCurr; inCurr = inNext;</code>, so
          every access uses a fixed name. It is tempting to avoid the copies with a small array and modular indexing, so
          that each new plane overwrites the oldest slot. The kernel below does that. It computes the same results as the
          book's kernel (a unit test runs a model of its loop under three thread orders; the file itself has never run),
          and by NVIDIA's documented rule its window is a candidate for local memory, because the compiler cannot see a
          constant index in{' '}
          <code>w[i % 3]</code>. Local memory is device memory. This is <a href={href('ch06', '6.6')}>chapter 6's register-promotion point</a>: constant
          indices, usually by unrolling, are what let the compiler keep a per-thread array in registers.
        </p>
      </Translated>
      <Source section="6.6" pages="145–146" />
      <Today checked="2026-11-20 (ch08-facts.md A5)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>NVIDIA: bộ nhớ cục bộ "nằm trong bộ nhớ thiết bị" và có độ trễ và băng thông của bộ nhớ toàn cục.</p>
          }
        >
          <p>NVIDIA: local memory "resides in device memory" and has the latency and bandwidth of global memory.</p>
        </Translated>
      </Today>
      <Code
        tabs={[{ label: 'rolling_window_indexed.cu', code: INDEXED_WINDOW, tiers: STENCIL_TIERS, status: tr('front-end-checked; never run', 'front-end-checked; chưa từng chạy') }]}
        caption={
          <Translated
            vi={
              <>
                Của chúng tôi, từ <code>labs/ch08/snippets/rolling_window_indexed.cu</code>: Hình 8.12 với cửa sổ là một mảng. Cùng kết
                quả trong mô hình, một câu hỏi khác cho trình biên dịch.
              </>
            }
          >
            <>
              Ours, from <code>labs/ch08/snippets/rolling_window_indexed.cu</code>: Fig. 8.12 with the window as an
              array. Same results in the model, a different question for the compiler.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Cùng hình dạng này xuất hiện trong mã thực tế. <code>ssm_conv</code> trên CUDA của llama.cpp, phép tích chập nhân quả ngắn
            dọc theo token trong các lớp kiểu Mamba, có đúng hình dạng này trên đường xử lý chuỗi ngắn: một luồng cho mỗi kênh tiến
            dọc theo các token, giữ <code>d_conv</code> đầu vào gần nhất trong <code>float x[d_conv]</code> và đọc{' '}
            <code>x[(i + j) % d_conv]</code>. Đó là cửa sổ thanh ghi của chương này trong một chiều. Việc nvcc có giữ mảng đó trong
            thanh ghi hay không thì chúng tôi chưa kiểm tra; <code>--resource-usage</code> trên tệp đó sẽ cho bạn biết.
          </p>
        }
      >
        <p>
          The same shape appears in real code. llama.cpp's CUDA <code>ssm_conv</code>, the short causal convolution along tokens
          in Mamba-style layers, has exactly this shape on its short-sequence path: one thread per channel marches along
          the tokens, keeping the last <code>d_conv</code> inputs in <code>float x[d_conv]</code> and reading{' '}
          <code>x[(i + j) % d_conv]</code>. It is this chapter's register window in one dimension. Whether nvcc keeps that
          array in registers we did not check; <code>--resource-usage</code> on that file would tell you.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch08-facts.md D1)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>Đọc từ nhánh master của llama.cpp, ggml/src/ggml-cuda/ssm-conv.cu. Cùng tệp này build backend HIP.</p>
          }
        >
          <p>Read from llama.cpp master, ggml/src/ggml-cuda/ssm-conv.cu. The same file builds the HIP backend.</p>
        </Translated>
      </Today>
      <Ask q={questionsS85['ch08/8.5/q-indexed-window']} />

      <h2>
        <Translated vi="Trên Metal">On Metal</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Các tab Metal của cả hai listing là cùng các kernel bằng MSL. Có ba chi tiết đáng biết trước khi bạn tự viết:
          </p>
        }
      >
        <p>
          The Metal tabs of both listings are the same kernels in MSL. Three details are worth knowing before you write
          your own:
        </p>
      </Translated>
      <ul>
        <li>
          <Translated
            vi={
              <>
                Một threadgroup 2D tiến qua một lưới 3D vẫn phải khai báo mọi thuộc tính vị trí là <code>uint3</code>. Trộn{' '}
                <code>uint2</code> và <code>uint3</code> là lỗi biên dịch.
              </>
            }
          >
            A 2D threadgroup marching through a 3D grid must still declare every position attribute <code>uint3</code>.
            Mixing <code>uint2</code> and <code>uint3</code> is a compile error.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Rào chắn bên trong vòng z hợp lệ vì mọi luồng chạy cùng số vòng lặp. Metal yêu cầu một rào chắn trong vòng lặp phải được
                mọi luồng của threadgroup đến được ở mọi vòng lặp, hoặc không luồng nào.
              </>
            }
          >
            The barrier inside the z loop is legal because every thread runs the same number of iterations. Metal requires
            that a barrier in a loop be reached by every thread of the threadgroup on every iteration, or by none.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Định kích thước ô bằng một macro hoặc một <code>constexpr</code>. Một function constant, thứ llama.cpp và MLX dùng cho
                kích thước kernel, không được ghi nhận là có thể định kích thước một mảng.
              </>
            }
          >
            Size the tile with a macro or a <code>constexpr</code>. A function constant, which llama.cpp and MLX use for
            kernel sizes, is not documented as able to size an array.
          </Translated>
        </li>
      </ul>
      <Today checked="2026-11-20 (ch08-facts.md C2, C3; ch04-facts C3)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>Từ đặc tả MSL. Lỗi uint2/uint3 cũng đã được tái hiện với trình biên dịch ở đây.</p>
          }
        >
          <p>From the MSL specification. The uint2/uint3 error was also reproduced with the compiler here.</p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Để so sánh, <code>ssm_conv</code> trên Metal của llama.cpp đi con đường khác: một luồng cho mỗi token, mọi đầu vào đọc
            thẳng từ bộ nhớ thiết bị, không bộ nhớ threadgroup và không tiến theo z. Đó là hình dạng của kernel cơ bản trong Hình 8.6,
            dựa vào cache cho phần chồng lấn.
          </p>
        }
      >
        <p>
          For comparison, llama.cpp's Metal <code>ssm_conv</code> takes the other road: one thread per token, every input
          read straight from device memory, no threadgroup memory and no march. That is the shape of the basic kernel in
          Fig. 8.6, relying on caches for the overlap.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch08-facts.md D2)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>Đọc từ nhánh master của llama.cpp, ggml/src/ggml-metal/kernels/ssm.metal.</p>
          }
        >
          <p>Read from llama.cpp master, ggml/src/ggml-metal/kernels/ssm.metal.</p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Tiếp theo sẽ đi đâu">Where this goes next</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hãy mang theo một quy tắc từ mục này: dữ liệu do một luồng đọc thuộc về thanh ghi của luồng đó; dữ liệu do nhiều luồng đọc
            thuộc về bộ nhớ chia sẻ. Chương 15 áp dụng nó cho phép nhân ma trận (
            <a href={href('ch15', '15.4')}>15.4, Phân ô thanh ghi cho các ô đầu vào</a>), nơi mỗi luồng giữ các dải của hai ô đầu
            vào trong thanh ghi và khối chia sẻ các ô; kernel đó là mốc dự án M3. Lab 8.1 yêu cầu bạn viết stencil phân ô thanh ghi
            và đo nó so với ba kernel còn lại.
          </p>
        }
      >
        <p>
          Carry one rule out of this section: data read by one thread belongs in that thread's registers; data read by
          several threads belongs in shared memory. Chapter 15 applies it to matrix multiplication (
          <a href={href('ch15', '15.4')}>15.4, Register tiling of the input tiles</a>), where each thread keeps strips of
          the two input tiles in registers and the block shares the tiles; that kernel is milestone M3.
          Lab 8.1 has you write the register-tiled stencil and measure it against the other three kernels.
        </p>
      </Translated>
      <Source section="15.3 and §15.4" pages="352–360" />
      <Ask q={questionsS85['ch08/8.5/q-diverge32']} />
    </>
  );
}
