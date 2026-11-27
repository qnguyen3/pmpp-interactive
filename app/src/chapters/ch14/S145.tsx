import { useEffect, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import type { Tier } from '../../components/notes';
import { Beyond, Bench, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { useSolutionStatus } from './solutionStatus';
import { BOOK_KEYS, DEFECT_LABEL, bin, parallelPass, randomKeys, type Defect, type ParallelPass } from '../../sim/radixsort';

// One pass as the lesson shows it: the region "pass" of labs/ch14/snippets/s145_radix_iter.cu
// (Fig. 14.7's lines split at the scan into two kernels and a host function), and the region
// "scan" of the same file. The file is front-end-checked and run on the emulator against
// src/sim/radixsort.ts by S145.snippets.test.ts, which also fails if these copies drift or
// if the line numbers in STEPS stop pointing at the lines they name.
const PASS = `__global__ void radix_bits_kernel(unsigned int* input, unsigned int* bits,
        unsigned int N, unsigned int iter) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N) {
        unsigned int key = input[i];
        unsigned int bit = (key >> iter) & 1;
        bits[i] = bit;
    }
}

__global__ void radix_scatter_kernel(unsigned int* input,
        unsigned int* output, unsigned int* bits, unsigned int N,
        unsigned int iter) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N) {
        unsigned int key = input[i];
        unsigned int bit = (key >> iter) & 1;
        unsigned int numOnesBefore = bits[i];
        unsigned int numOnesTotal = bits[N];
        unsigned int dst = (bit == 0)
            ? (i - numOnesBefore)
            : (N - numOnesTotal + numOnesBefore);
        output[dst] = key;
    }
}

// bits has N + 1 entries.
void radix_sort_iter(unsigned int* input, unsigned int* output,
        unsigned int* bits, unsigned int N, unsigned int iter,
        unsigned int blockSize) {
    unsigned int numBlocks = (N + blockSize - 1) / blockSize;
    radix_bits_kernel<<<numBlocks, blockSize>>>(input, bits, N, iter);
    exclusive_scan_kernel<<<1, SCAN_THREADS>>>(bits, N);
    radix_scatter_kernel<<<numBlocks, blockSize>>>(input, output, bits,
                                                   N, iter);
}`;

const SCAN = `// Not the book's code: an exclusive scan of bits[0..N-1] followed by one zero, in one block,
// written in place. Each thread sums a contiguous chunk, the chunk sums are scanned in
// shared memory (Kogge-Stone, section 11.2), and each thread then rewrites its chunk as a
// running sum: chapter 11's coarsened scan (section 11.6) in the reduce-scan-scan order of
// section 11.9, with the chunks left in global memory. bits[N] receives the total.
__global__ void exclusive_scan_kernel(unsigned int* bits, unsigned int N) {
    __shared__ unsigned int sums[SCAN_THREADS];
    unsigned int t = threadIdx.x;
    unsigned int per = (N + blockDim.x - 1) / blockDim.x;
    unsigned int begin = (t * per < N) ? t * per : N;
    unsigned int end = (begin + per < N) ? begin + per : N;
    unsigned int mine = 0;
    for(unsigned int k = begin; k < end; ++k) mine += bits[k];
    sums[t] = mine;
    __syncthreads();
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        unsigned int left = (t >= stride) ? sums[t - stride] : 0;
        __syncthreads();
        sums[t] += left;
        __syncthreads();
    }
    unsigned int run = sums[t] - mine;          // ones in the chunks before this one
    for(unsigned int k = begin; k < end; ++k) {
        unsigned int b = bits[k];
        bits[k] = run;
        run += b;
    }
    if(t == blockDim.x - 1) bits[N] = run;      // after the last chunk: the total
}`;

const SNIPPET_STATUS = () =>
  tr(
    'front-end-checked; emulator-checked; unverified on CUDA and HIP hardware',
    'front-end-checked; emulator-checked; unverified trên phần cứng CUDA và HIP',
  );
const TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', bits: 'global',
  i: 'reg', key: 'reg', bit: 'reg', numOnesBefore: 'reg', numOnesTotal: 'reg', dst: 'reg',
  sums: 'shared', t: 'reg', per: 'reg', begin: 'reg', end: 'reg', mine: 'reg', stride: 'reg', left: 'reg', run: 'reg', b: 'reg', k: 'reg',
};

type Op = 'i' | 'load' | 'bit' | 'store-bit' | 'scan' | 'reload' | 'read' | 'dst' | 'store';
/** The stepper's steps and the lines of PASS each one highlights (pinned by S145.snippets.test.ts). */
export const STEPS: { op: Op; lines: number[]; label: string }[] = [
  { op: 'i', lines: [32, 3], get label() { return tr('kernel 1: i', 'kernel 1: i'); } },
  { op: 'load', lines: [5], get label() { return tr('load the key', 'nạp khóa'); } },
  { op: 'bit', lines: [6], get label() { return tr('extract the bit', 'trích bit'); } },
  { op: 'store-bit', lines: [7], get label() { return tr('store bits[i]', 'ghi bits[i]'); } },
  { op: 'scan', lines: [33], get label() { return tr('scan', 'scan'); } },
  { op: 'reload', lines: [34, 35, 14, 16, 17], get label() { return tr('kernel 2: i, key, bit', 'kernel 2: i, khóa, bit'); } },
  { op: 'read', lines: [18, 19], get label() { return tr('read the scan', 'đọc kết quả scan'); } },
  { op: 'dst', lines: [20, 21, 22], get label() { return tr('destination', 'đích đến'); } },
  { op: 'store', lines: [23], get label() { return tr('store the key', 'ghi khóa'); } },
];
const PREDICT_THREADS = [2, 6, 11, 13];
type Col = 'i' | 'key' | 'bit' | 'before' | 'total' | 'dst';
const ROWS: { col: Col; label: string }[] = [
  { col: 'i', label: 'i' },
  { col: 'key', get label() { return tr('key', 'khóa'); } },
  { col: 'bit', get label() { return tr('bit', 'bit'); } },
  { col: 'before', label: 'numOnesBefore' },
  { col: 'total', label: 'numOnesTotal' },
  { col: 'dst', label: 'dst' }, // i18n-ignore: the kernel's variable name
];
const LAB1 = 'ch14/lab1_twosorts/solution/';
const loaded = (t: string | null) => (t && !t.startsWith('// could not load') ? t : null);

// ---------------------------------------------------------------- the stepper
const LX = 92; // left label column
const CW = 38;
const KEY_Y = 30;
const BITS_Y = 112;
const OUT_Y = 222;

function narrate(pp: ParallelPass, s: number, f: number): string {
  const t = pp.threads[f];
  const n = pp.n;
  const who = tr(`Thread ${t.thread} of block ${t.block}`, `Luồng ${t.thread} của khối ${t.block}`);
  const kb = pp.keys.some((k) => k > 15) ? 8 : 4;
  const key = bin(t.key, kb);
  switch (s) {
    case 0:
      return tr(
        `Nothing has run. ${n} keys, one thread each, in blocks of ${pp.blockDim}; this pass reads bit ${pp.bit}. Click a key or a column of the register table to follow another thread.`,
        `Chưa có gì chạy. ${n} khóa, mỗi khóa một luồng, trong các khối gồm ${pp.blockDim} luồng; lượt này đọc bit ${pp.bit}. Nhấp vào một khóa hoặc một cột của bảng thanh ghi để theo dõi luồng khác.`,
      );
    case 1:
      return tr(
        `The host launches kernel 1. ${who}: i = blockIdx.x·blockDim.x + threadIdx.x = ${t.block}·${pp.blockDim} + ${t.thread} = ${t.i}. (The page's line 03.)`,
        `Host khởi chạy kernel 1. ${who}: i = blockIdx.x·blockDim.x + threadIdx.x = ${t.block}·${pp.blockDim} + ${t.thread} = ${t.i}. (Dòng 03 của trang sách.)`,
      );
    case 2:
      return tr(
        `${who} loads its key: input[${t.i}] = ${key}. Every thread loads one key, so the whole input is read once.`,
        `${who} nạp khóa của nó: input[${t.i}] = ${key}. Mỗi luồng nạp một khóa, nên toàn bộ đầu vào được đọc một lần.`,
      );
    case 3:
      return tr(
        `bit = (key >> ${pp.bit}) & 1 = ${t.bit}: shift the bit of interest to the right end, mask off the rest.`,
        `bit = (key >> ${pp.bit}) & 1 = ${t.bit}: dịch bit cần xét về đầu bên phải, che bỏ phần còn lại.`,
      );
    case 4:
      return tr(
        `${who} stores bits[${t.i}] = ${t.bit}. When every thread has done this, bits holds one bit per key. That is the end of kernel 1.`,
        `${who} ghi bits[${t.i}] = ${t.bit}. Khi mọi luồng đã làm vậy, bits chứa một bit cho mỗi khóa. Đó là phần kết của kernel 1.`,
      );
    case 5:
      if (pp.defect === 'scan-n' || pp.defect === 'last-as-total')
        return tr(
          `The host launches the scan over the ${n} entries of a ${n}-entry bits array. Each entry now holds the ones before it, and no entry holds the total. (On the page this is line 10, gridExclusiveScan.)`,
          `Host khởi chạy phép scan trên ${n} phần tử của một mảng bits ${n} phần tử. Giờ mỗi phần tử chứa số bit 1 đứng trước nó, và không phần tử nào chứa tổng. (Trên trang sách đây là dòng 10, gridExclusiveScan.)`,
        );
      if (pp.defect === 'inclusive')
        return tr(
          `The host launches an inclusive scan over ${n + 1} entries: entry k now counts the ones up to and including key k, not the ones before it.`,
          `Host khởi chạy một phép scan bao gồm trên ${n + 1} phần tử: giờ phần tử k đếm số bit 1 tính đến và bao gồm cả khóa k, chứ không phải số bit 1 đứng trước nó.`,
        );
      return tr(
        `The host launches the scan, one launch over every block's bits: ${n} bits and a zero after them. Entry k now holds the ones before key k, and entry ${n} the total, ${pp.ones}. (On the page this is line 10, gridExclusiveScan.)`,
        `Host khởi chạy phép scan, một lần khởi chạy trên bit của mọi khối: ${n} bit và một số 0 sau chúng. Giờ phần tử k chứa số bit 1 trước khóa k, và phần tử ${n} chứa tổng, ${pp.ones}. (Trên trang sách đây là dòng 10, gridExclusiveScan.)`,
      );
    case 6:
      return tr(
        `The host launches kernel 2. ${who} computes i = ${t.i} again and reloads key ${key}, and recomputes its bit, ${t.bit}: the scan has overwritten bits[${t.i}].`,
        `Host khởi chạy kernel 2. ${who} tính lại i = ${t.i}, nạp lại khóa ${key}, và tính lại bit của nó, ${t.bit}: phép scan đã ghi đè bits[${t.i}].`,
      );
    case 7:
      if (pp.defect === 'scan-n')
        return tr(
          `numOnesBefore = bits[${t.i}] = ${t.onesBefore}. numOnesTotal = bits[${n}]: past the end of a ${n}-entry array. Out of range; what it returns is anyone's guess.`,
          `numOnesBefore = bits[${t.i}] = ${t.onesBefore}. numOnesTotal = bits[${n}]: quá cuối một mảng ${n} phần tử. Ngoài phạm vi; nó trả về gì thì không ai đoán được.`,
        );
      if (pp.defect === 'last-as-total')
        return tr(
          `numOnesBefore = bits[${t.i}] = ${t.onesBefore}. numOnesTotal = bits[${n - 1}] = ${t.total}, the last of the ${n} entries, as Fig. 14.6 suggests. The true total is ${pp.ones}.`,
          `numOnesBefore = bits[${t.i}] = ${t.onesBefore}. numOnesTotal = bits[${n - 1}] = ${t.total}, phần tử cuối trong ${n} phần tử, như Hình 14.6 gợi ý. Tổng thật là ${pp.ones}.`,
        );
      return `numOnesBefore = bits[${t.i}] = ${t.onesBefore}; numOnesTotal = bits[${n}] = ${t.total}.`;
    case 8:
      if (pp.defect === 'zero-formula')
        return tr(
          `With the defect every key uses the zero formula: dst = i − numOnesBefore = ${t.formula}.`,
          `Với lỗi này mọi khóa đều dùng công thức cho bit 0: dst = i − numOnesBefore = ${t.formula}.`,
        );
      return t.bit === 0
        ? tr(
            `bit is 0, so dst = i − numOnesBefore = ${t.formula}: the zeros before it, since every key before it is a zero or a one.`,
            `bit bằng 0, nên dst = i − numOnesBefore = ${t.formula}: số bit 0 đứng trước nó, vì mọi khóa đứng trước nó đều là 0 hoặc 1.`,
          )
        : tr(
            `bit is 1, so dst = N − numOnesTotal + numOnesBefore = ${t.formula}: all the zeros, then the ones before it.`,
            `bit bằng 1, nên dst = N − numOnesTotal + numOnesBefore = ${t.formula}: tất cả các bit 0, rồi các bit 1 đứng trước nó.`,
          );
    default: {
      const bad = pp.collisions.length + pp.holes.length + pp.oobWrites.length + pp.unknown.length;
      const own =
        t.dst == null
          ? tr(`${who} cannot know where to write: its destination depends on the out-of-range read.`, `${who} không thể biết phải ghi vào đâu: đích đến của nó phụ thuộc vào lần đọc ngoài phạm vi.`)
          : t.oobWrite
            ? tr(`${who} writes output[${t.dst}]: past the end of the output.`, `${who} ghi output[${t.dst}]: quá cuối đầu ra.`)
            : tr(`${who} writes output[${t.dst}] = ${key}.`, `${who} ghi output[${t.dst}] = ${key}.`);
      return bad
        ? tr(
            `${own} Across the grid: ${pp.collisions.length} slots written twice, ${pp.holes.length} never written, ${pp.oobWrites.length} writes out of range.`,
            `${own} Trên toàn lưới: ${pp.collisions.length} ô bị ghi hai lần, ${pp.holes.length} ô không bao giờ được ghi, ${pp.oobWrites.length} lần ghi ngoài phạm vi.`,
          )
        : tr(
            `${own} Every slot is written exactly once, and the output is the stable partition by bit ${pp.bit}.`,
            `${own} Mọi ô được ghi đúng một lần, và đầu ra là phép phân hoạch ổn định theo bit ${pp.bit}.`,
          );
    }
  }
}

function PassSvg({ pp, s, focus, setFocus, predict }: { pp: ParallelPass; s: number; focus: number; setFocus: (i: number) => void; predict: boolean }) {
  const n = pp.n;
  const kb = pp.keys.some((k) => k > 15) ? 8 : 4;
  const cx = (j: number) => LX + j * CW + (CW - 4) / 2;
  const totalX = LX + n * CW + 14;
  const w = totalX + CW + 10;
  const scanned = s >= 5;
  const wrote = s >= 9;
  const showDst = (i: number) => s >= 8 && !(predict && PREDICT_THREADS.includes(i));
  const cellCls = (slot: number) => (pp.writers[slot].length > 1 ? 'bad' : pp.writers[slot].length === 0 ? 'bad' : '');
  const sideCount = pp.scan.length > n ? n + 1 : n;
  return (
    <svg viewBox={`0 0 ${w} 278`} role="img" aria-label={tr('One radix pass, thread by thread', 'Một lượt radix, từng luồng một')} style={{ width: '100%', maxWidth: w, minWidth: Math.min(w, 600), display: 'block' }}>
      <defs>
        <marker id="pp-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">{/* i18n-ignore: SVG attribute value */}
          <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--ink-2)' }} />
        </marker>
        <marker id="pp-arrow-bad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">{/* i18n-ignore: SVG attribute value */}
          <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--fail)' }} />
        </marker>
      </defs>
      {Array.from({ length: Math.ceil(n / pp.blockDim) }, (_, b) => (
        <g key={b}>
          <line x1={LX + b * pp.blockDim * CW + 1} x2={LX + (b + 1) * pp.blockDim * CW - 5} y1={14} y2={14} stroke="var(--ink-2)" />
          <text x={LX + b * pp.blockDim * CW + (pp.blockDim * CW) / 2 - 2} y={10} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>khối {b}</>}>block {b}</Translated></text>
        </g>
      ))}
      <text x={6} y={KEY_Y + 17} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}>{tr('input', 'đầu vào')}</text>
      {pp.keys.map((k, i) => (
        <g key={i} data-key-slot={i} onClick={() => setFocus(i)} style={{ cursor: 'pointer' }}>
          <rect x={LX + i * CW} y={KEY_Y} width={CW - 4} height={26} rx="2"
            fill={i === focus ? 'var(--global)' : s >= 2 ? 'var(--global-wash)' : 'var(--cell)'} stroke="var(--global)" />
          <text x={cx(i)} y={KEY_Y + 17} fontSize={kb === 8 ? 8.5 : 11} textAnchor="middle" style={{ fill: i === focus ? 'var(--page)' : 'var(--ink)' }}>{bin(k, kb)}</text>
          <text x={cx(i)} y={KEY_Y + 40} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{i}</text>
        </g>
      ))}
      <text x={6} y={BITS_Y + 17} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}>{scanned ? tr('bits, scanned', 'bits, đã scan') : 'bits'}</text>
      {Array.from({ length: sideCount }, (_, k) => {
        const x = k < n ? LX + k * CW : totalX;
        const v = scanned ? pp.scan[k] : k < n ? (s >= 4 ? pp.bits[k] : null) : 0;
        const hidden = scanned && predict && k < n && PREDICT_THREADS.includes(k);
        const readTotal = s >= 7 && k === n && pp.defect !== 'scan-n' && pp.defect !== 'last-as-total';
        return (
          <g key={k} data-bits-slot={k}>
            <rect x={x} y={BITS_Y} width={CW - 4} height={26} rx="2" fill={readTotal ? 'var(--global-wash)' : 'var(--cell)'} stroke="var(--global)" strokeDasharray={k === n && !scanned ? '3 3' : undefined} />
            <text x={x + (CW - 4) / 2} y={BITS_Y + 17} fontSize="11.5" textAnchor="middle" data-v={hidden ? '' : v ?? ''}>{hidden ? '?' : v ?? ''}</text>
          </g>
        );
      })}
      {pp.scan.length === n ? (
        <g data-missing-total="">
          <rect x={totalX} y={BITS_Y} width={CW - 4} height={26} rx="2" fill={s >= 7 && pp.defect === 'scan-n' ? 'var(--fail-wash)' : 'none'} stroke="var(--fail)" strokeDasharray="3 3" />
          <text x={totalX + (CW - 4) / 2} y={BITS_Y + 17} fontSize="11" textAnchor="middle" style={{ fill: 'var(--fail)' }}>{s >= 7 && pp.defect === 'scan-n' ? '??' : ''}</text>
        </g>
      ) : null}
      <text x={totalX + (CW - 4) / 2} y={BITS_Y + 40} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>[{n}]</text>
      {s === 5 ? (
        <g data-wall="">
          <rect x={LX - 4} y={BITS_Y - 25} width={totalX + CW - LX} height={18} rx="2" fill="var(--ink)" />
          <text x={LX + 6} y={BITS_Y - 12} fontSize="11" style={{ fill: 'var(--page)' }}>{tr("exclusive scan: one launch over every block's bits; no thread goes on until it has finished", 'scan loại trừ: một lần khởi chạy trên bit của mọi khối; không luồng nào đi tiếp cho đến khi nó xong')}</text>
        </g>
      ) : null}
      <text x={6} y={OUT_Y + 17} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}>{tr('output', 'đầu ra')}</text>
      {Array.from({ length: n }, (_, j) => {
        const v = wrote ? pp.out[j] : null;
        const cls = wrote ? cellCls(j) : '';
        return (
          <g key={j} data-out-slot={j} data-out-state={wrote ? (cls ? (pp.writers[j].length ? 'collision' : 'hole') : 'ok') : ''}>
            <rect x={LX + j * CW} y={OUT_Y} width={CW - 4} height={26} rx="2"
              fill={cls ? 'var(--fail-wash)' : 'var(--cell)'} stroke={cls ? 'var(--fail)' : 'var(--global)'} strokeDasharray={wrote && pp.writers[j].length === 0 ? '3 3' : undefined} />
            <text x={cx(j)} y={OUT_Y + 17} fontSize={kb === 8 ? 8.5 : 11} textAnchor="middle" style={{ fill: cls ? 'var(--fail)' : 'var(--ink)' }}>
              {wrote ? (pp.writers[j].length > 1 ? `×${pp.writers[j].length}` : v == null ? '' : bin(v, kb)) : ''}
            </text>
            <text x={cx(j)} y={OUT_Y + 40} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{j}</text>
          </g>
        );
      })}
      {s >= 8
        ? pp.threads.map((t) => {
            if (!showDst(t.i) || t.dst == null) return null;
            const out = t.oobWrite;
            const bad = out || (t.dst >= 0 && t.dst < n && pp.writers[t.dst].length > 1);
            const x2 = out ? totalX + 8 : cx(t.dst);
            return (
              <line key={t.i} data-arrow={t.i} data-to={t.dst} x1={cx(t.i)} y1={BITS_Y + 44} x2={x2} y2={OUT_Y - 3}
                stroke={bad ? 'var(--fail)' : t.i === focus ? 'var(--ink)' : 'var(--ink-2)'} strokeWidth={t.i === focus ? 2.2 : bad ? 1.5 : 1}
                strokeDasharray={wrote ? undefined : '4 3'} opacity={t.i === focus || bad ? 1 : 0.7}
                markerEnd={bad ? 'url(#pp-arrow-bad)' : 'url(#pp-arrow)'} />
            );
          })
        : null}
      {s >= 8 && pp.oobWrites.length ? (
        <text x={totalX + 8} y={OUT_Y + 17} fontSize="10.5" style={{ fill: 'var(--fail)' }}>{tr('out of range', 'ngoài phạm vi')}</text>
      ) : null}
    </svg>
  );
}

function PassStepper() {
  const [source, setSource] = useState<'book' | 'random'>('book');
  const [seed, setSeed] = useState(3);
  const [bit, setBit] = useState(0);
  const [defect, setDefect] = useState<Defect>('none');
  const [s, setS] = useState(0);
  const [focus, setFocus] = useState(6);
  const [predict, setPredict] = useState(false);
  const [guesses, setGuesses] = useState<Record<string, string>>({});
  const [checked, setChecked] = useState(false);

  const keys = useMemo(() => (source === 'book' ? BOOK_KEYS : randomKeys(16, 8, seed)), [source, seed]);
  const maxBit = source === 'book' ? 3 : 7;
  const b = Math.min(bit, maxBit);
  const pp = useMemo(() => parallelPass(keys, b, 4, defect), [keys, b, defect]);
  const truth = useMemo(() => parallelPass(keys, b, 4, 'none'), [keys, b]);
  useEffect(() => setS(0), [keys, b, defect]);
  useEffect(() => {
    setGuesses({});
    setChecked(false);
  }, [keys, b]);

  const hideFor = (i: number) => predict && !checked && PREDICT_THREADS.includes(i);
  const grade = (i: number, what: 'before' | 'dst') => {
    const g = guesses[`${i}:${what}`];
    if (g == null || g.trim() === '') return null;
    const want = what === 'before' ? truth.threads[i].onesBefore : truth.threads[i].dst;
    return Number(g) === want;
  };
  const allRight = PREDICT_THREADS.every((i) => grade(i, 'before') && grade(i, 'dst'));
  const step = STEPS[s - 1];
  const done = s === STEPS.length;
  const ft = pp.threads[focus];
  const badNow = s >= 7 && (ft.oobRead || (s >= 8 && (ft.dst == null || ft.oobWrite)));
  const kb = pp.keys.some((k) => k > 15) ? 8 : 4;
  const colStyle = (i: number) => (i === focus ? { background: 'var(--rule-soft)', fontWeight: 650 } : {});
  const guessBox = (i: number, what: 'before' | 'dst') => (
    <input type="number" aria-label={tr(`thread ${i} ${what === 'before' ? 'ones before' : 'destination'}`, `luồng ${i} ${what === 'before' ? 'số bit 1 phía trước' : 'đích đến'}`)}
      value={guesses[`${i}:${what}`] ?? ''} style={{ width: '2.9rem', padding: '0.05rem 0.2rem' }}
      onClick={(e) => e.stopPropagation()}
      onChange={(e) => setGuesses({ ...guesses, [`${i}:${what}`]: e.target.value })} />
  );
  const mark = (i: number, what: 'before' | 'dst') => {
    if (!(predict && checked && PREDICT_THREADS.includes(i))) return null;
    return grade(i, what) ? <span className="t-ok" data-grade="right"> ✓</span> : <span className="t-fail" data-grade="wrong"> ✗</span>;
  };
  const cell = (t: ParallelPass['threads'][number], col: Col) => {
    switch (col) {
      case 'i': return s >= 1 ? t.i : '';
      case 'key': return s >= 2 ? bin(t.key, kb) : '';
      case 'bit': return s >= 3 ? t.bit : '';
      case 'before':
        return hideFor(t.i) ? guessBox(t.i, 'before') : <>{s >= 7 ? t.onesBefore : ''}{mark(t.i, 'before')}</>;
      case 'total':
        return s >= 7 ? (t.total == null ? <span className="t-fail">??</span> : t.total) : '';
      default:
        return hideFor(t.i) ? guessBox(t.i, 'dst') : (
          <>
            {s >= 8 ? (t.dst == null ? <span className="t-fail">?</span> : t.oobWrite ? <span className="t-fail">{t.dst}</span> : t.dst) : ''}
            {mark(t.i, 'dst')}
          </>
        );
    }
  };

  return (
    <Bench
      title={tr('One pass, thread by thread', 'Một lượt, từng luồng một')}
      task={tr(
        "Follow one thread through both kernels, predict four threads' numbers, then break the pass.",
        'Theo một luồng qua cả hai kernel, dự đoán các con số của bốn luồng, rồi làm hỏng lượt đó.',
      )}
      foot={
        <Translated
          vi={
            <>
              Màu hổ phách là bộ nhớ toàn cục, màu tím là các thanh ghi của một luồng. Nhấp vào một khóa hoặc một cột để
              theo dõi luồng đó. Mô hình: <code>src/sim/radixsort.ts</code>, <code>parallelPass</code> (các bài kiểm thử
              của nó tái hiện các hàng của Hình 14.6).
            </>
          }
        >
          <>
            Amber is global memory, purple a thread's registers. Click a key or a column to follow that thread. Model:{' '}
            <code>src/sim/radixsort.ts</code>, <code>parallelPass</code> (its tests reproduce Fig. 14.6's rows).
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Khóa">Keys</Translated>
          <select value={source} onChange={(e) => setSource(e.target.value as 'book' | 'random')}>
            <option value="book">{tr("the book's sixteen (Fig. 14.5)", 'mười sáu khóa của sách (Hình 14.5)')}</option>
            <option value="random">{tr('sixteen random 8-bit keys', 'mười sáu khóa 8 bit ngẫu nhiên')}</option>
          </select>
        </label>
        {source === 'random' ? <button onClick={() => setSeed(seed + 1)}>{tr('Shuffle', 'Xáo trộn')}</button> : null}
        <label className="field">
          <Translated vi="Bit">Bit</Translated>
          <select value={b} onChange={(e) => setBit(Number(e.target.value))} aria-label={tr('Bit', 'Bit')}>
            {Array.from({ length: maxBit + 1 }, (_, k) => <option key={k} value={k}><Translated vi={<>bit {k}</>}>bit {k}</Translated></option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Kernel">Kernel</Translated>
          <select value={defect} onChange={(e) => setDefect(e.target.value as Defect)} aria-label={tr('Kernel', 'Kernel')}>
            {(Object.keys(DEFECT_LABEL) as Defect[]).map((d) => (
              <option key={d} value={d}>
                {tr(DEFECT_LABEL[d], {
                  none: 'như đã sửa',
                  'zero-formula': 'bit 1 dùng công thức của bit 0',
                  'scan-n': 'scan trên n phần tử',
                  'last-as-total': 'tổng lấy từ phần tử cuối trong n phần tử',
                  inclusive: 'scan bao gồm thay vì loại trừ',
                }[d])}
              </option>
            ))}
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={predict} onChange={(e) => { setPredict(e.target.checked); setChecked(false); }} />{' '}<Translated vi="dự đoán trước">predict first</Translated>
        </label>
      </div>

      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setS(0)} disabled={s === 0}>{tr('Restart', 'Bắt đầu lại')}</button>
        <button onClick={() => setS((x) => Math.max(0, x - 1))} disabled={s === 0}>{tr('Back', 'Lùi')}</button>
        <button className="primary" onClick={() => setS((x) => Math.min(STEPS.length, x + 1))} disabled={done}>{tr('Step', 'Bước')}</button>
        <button onClick={() => setS(STEPS.length)} disabled={done}>{tr('Run to the end', 'Chạy đến cuối')}</button>
        <span className="muted" aria-live="polite"><Translated vi={<>bước {s} / {STEPS.length}{step ? `: ${step.label}` : ''}</>}>step {s} of {STEPS.length}{step ? `: ${step.label}` : ''}</Translated></span>
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 1fr)', gap: '1rem' }}>
        <Code tabs={[{ label: 's145_radix_iter.cu', code: PASS, tiers: TIERS }]} highlight={step ? step.lines : []} />
        <div className={'narrate' + (badNow ? ' bad' : '')} role="status" style={{ marginTop: '1.1rem' }}>
          {narrate(pp, s, focus)}
        </div>
      </div>

      <div className="mx-label t-global"><Translated vi="Bộ nhớ toàn cục">Global memory</Translated></div>
      <PassSvg pp={pp} s={s} focus={focus} setFocus={setFocus} predict={hideFor(PREDICT_THREADS[0])} />

      <div className="mx-label t-reg" style={{ marginTop: '0.8rem' }}><Translated vi="Các thanh ghi của mỗi luồng (nhấp vào một cột để theo dõi luồng đó)">Each thread's registers (click a column to follow that thread)</Translated></div>
      <div style={{ overflowX: 'auto' }}>
        <table className="threads" aria-label={tr("Every thread's registers", 'Các thanh ghi của mọi luồng')} data-part="registers" style={{ width: 'auto' }}>
          <thead>
            <tr>
              <th>block.thread</th>
              {pp.threads.map((t) => (
                <th key={t.i} data-thread={t.i} onClick={() => setFocus(t.i)} style={{ cursor: 'pointer', ...colStyle(t.i) }}>
                  {t.block}.{t.thread}{predict && PREDICT_THREADS.includes(t.i) ? '*' : ''}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ROWS.map((row) => (
              <tr key={row.col}>
                <td>{row.label}</td>
                {pp.threads.map((t) => (
                  <td key={t.i} className="t-reg" data-thread={t.i} data-col={row.col} onClick={() => setFocus(t.i)} style={{ cursor: 'pointer', ...colStyle(t.i) }}>
                    {cell(t, row.col)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {predict ? (
        <div className="row" style={{ marginTop: '0.6rem' }} data-part="predict">
          <button className="primary" onClick={() => setChecked(true)}>{tr('Check my predictions', 'Kiểm tra dự đoán của tôi')}</button>
          {checked ? (
            allRight ? <span className="chip ok"><Translated vi="cả tám con số đều đúng">all eight numbers right</Translated></span> : <span className="chip fail"><Translated vi="có con số bị lệch; bảng giờ hiển thị số của mô hình">some numbers are off; the table now shows the model's</Translated></span>
          ) : (
            <span className="muted"><Translated vi="Các cột đánh dấu *: điền số bit 1 phía trước và đích đến cho kernel đã sửa, rồi kiểm tra.">Columns marked *: fill in ones before and destination for the corrected kernel, then check.</Translated></span>
          )}
        </div>
      ) : null}

      {done ? (
        <p style={{ margin: '0.8rem 0 0' }} data-part="outcome">
          {pp.correct ? (
            <span className="chip ok"><Translated vi="mọi ô được ghi một lần: phép phân hoạch ổn định">every slot written once: the stable partition</Translated></span>
          ) : (
            <>
              <span className="chip fail"><Translated vi="đầu ra sai">wrong output</Translated></span>{' '}
              {[
                pp.collisions.length ? tr(`${pp.collisions.length} slot${pp.collisions.length > 1 ? 's' : ''} written twice (${pp.collisions.join(', ')})`, `${pp.collisions.length} ô bị ghi hai lần (${pp.collisions.join(', ')})`) : '',
                pp.holes.length ? tr(`${pp.holes.length} never written (${pp.holes.join(', ')})`, `${pp.holes.length} ô không bao giờ được ghi (${pp.holes.join(', ')})`) : '',
                pp.oobWrites.length ? tr(`${pp.oobWrites.length} write${pp.oobWrites.length > 1 ? 's' : ''} out of range`, `${pp.oobWrites.length} lần ghi ngoài phạm vi`) : '',
                pp.oobReads.length ? tr(`${pp.oobReads.length} reads of bits[${pp.n}] out of range`, `${pp.oobReads.length} lần đọc bits[${pp.n}] ngoài phạm vi`) : '',
              ].filter(Boolean).join('; ')}.
            </>
          )}
          {defect === 'last-as-total' && pp.correct ? (
            <span className="muted">
              <Translated vi=" Ở đây đúng chỉ vì bit của khóa cuối bằng 0. Hãy thử khóa ngẫu nhiên cho đến khi bit của khóa cuối bằng 1.">
                {' '}Right here only because the last key's bit is 0. Try random keys until the last key's bit is 1.
              </Translated>
            </span>
          ) : null}
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS145: Record<string, Question> = {
  'ch14/14.5/q-dst-one': {
    kind: 'numeric',
    id: 'ch14/14.5/q-dst-one',
    prompt: (
      <Translated
        vi={
          <>
            Một lượt trên n = 20 khóa tìm thấy tổng cộng 9 bit 1. Khóa ở chỉ số 13 có bit 1 và 6 bit 1 đứng trước nó. Đích đến của
            nó là gì?
          </>
        }
      >
        <>
          A pass over n = 20 keys finds 9 ones in total. The key at index 13 has bit 1 and 6 ones before it. What is its
          destination?
        </>
      </Translated>
    ),
    answer: 17,
    explain: (
      <Translated
        vi={
          <p>
            Cả 20 − 9 = 11 bit 0 đứng trước, rồi đến 6 bit 1 đứng trước nó: 11 + 6 = 17, tức là{' '}
            <code>N − numOnesTotal + numOnesBefore</code>.
          </p>
        }
      >
        <p>
          All 20 − 9 = 11 zeros come first, then the 6 ones before it: 11 + 6 = 17, which is{' '}
          <code>N − numOnesTotal + numOnesBefore</code>.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.5, p. 337',
  },
  'ch14/14.5/q-dst-zero': {
    kind: 'numeric',
    id: 'ch14/14.5/q-dst-zero',
    prompt: (
      <Translated vi={<>Cùng lượt đó; giờ khóa ở chỉ số 13 có bit 0, và 6 bit 1 đứng trước nó. Đích đến của nó là gì?</>}>
        <>The same pass; now the key at index 13 has bit 0, and 6 ones before it. What is its destination?</>
      </Translated>
    ),
    answer: 7,
    explain: (
      <Translated
        vi={
          <p>
            Mười ba khóa đứng trước nó và 6 trong số đó là bit 1, nên 7 là bit 0: 13 − 6 = 7, tức là{' '}
            <code>i − numOnesBefore</code>. Tổng không liên quan ở đây.
          </p>
        }
      >
        <p>
          Thirteen keys come before it and 6 of them are ones, so 7 are zeros: 13 − 6 = 7, which is{' '}
          <code>i − numOnesBefore</code>. The total does not enter into it.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.5, p. 337',
  },
  'ch14/14.5/q-total': {
    kind: 'choice',
    id: 'ch14/14.5/q-total',
    prompt: (
      <Translated
        vi={
          <>
            Hình 14.7 đọc <code>bits[N]</code> để lấy tổng số bit 1, và trong Hình 14.6 giá trị scan cuối cùng được hiển thị, 8,
            bằng tổng. Vì sao một phép scan trên N bit là chưa đủ?
          </>
        }
      >
        <>
          Fig. 14.7 reads <code>bits[N]</code> for the total number of ones, and in Fig. 14.6 the last scan value shown,
          8, equals the total. Why is a scan of the N bits not enough?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Đủ rồi: phần tử cuối của một phép scan loại trừ trên N bit chứa tổng của chúng, tức số bit 1.">It is enough: the last entry of an exclusive scan of N bits holds their sum, the number of ones.</Translated>,
        why: <Translated vi="Nó thiếu bit của chính khóa cuối. Các khóa của sách kết thúc bằng 0000, nên điều này bị che đi.">It misses the last key's own bit. The book's keys end with 0000, which hides that.</Translated>,
      },
      {
        label: <Translated vi="Phần tử cuối của nó đếm số bit 1 trước khóa cuối, không tính bit của chính khóa đó. Hãy scan N + 1 phần tử, phần tử cuối là 0.">Its last entry counts the ones before the last key, without that key's own bit. Scan N + 1 entries, the last a zero.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Thay vào đó dùng scan bao gồm ở mọi nơi: phần tử cuối của nó là tổng, và các công thức giữ nguyên.">Use an inclusive scan everywhere instead: its last entry is the total, and the formulas stay as they are.</Translated>,
        why: <Translated vi={'Khi đó mỗi bit 1 nhận nhiều hơn "số bit 1 phía trước" của nó một đơn vị và rơi quá một ô: biến thể scan bao gồm của trình chạy từng bước cho thấy điều này.'}>Then every one gets one more than its "ones before" and lands a slot too far: the stepper's inclusive variant shows it.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trên trang sách, không có gì nói <code>bits[N]</code> đến từ đâu; giá trị cuối của hình khớp với tổng chỉ vì khóa cuối
            là <code>0000</code>. Đoạn mã cấp phát N + 1 phần tử và phép scan của nó ghi tổng vào phần tử cuối. Trong trình chạy
            từng bước, hãy chọn "tổng lấy từ phần tử cuối trong n phần tử" với khóa ngẫu nhiên.
          </p>
        }
      >
        <p>
          On the page, nothing says where <code>bits[N]</code> comes from; the figure's last value matches the total
          only because the last key is <code>0000</code>. The snippet allocates N + 1 entries and its scan writes the
          total into the last one. In the stepper, choose "total from the last of n entries" with random keys.
        </p>
      </Translated>
    ),
    source: 'notes/ch14.md, "Unclear in the PDF" item 4; pp. 338–339',
    vi: { source: 'notes/ch14.md, "Unclear in the PDF" mục 4; pp. 338–339' },
  },
  'ch14/14.5/q-runs': {
    kind: 'numeric',
    id: 'ch14/14.5/q-runs',
    prompt: (
      <Translated
        vi={
          <>
            Một warp 32 luồng trong một lượt 1 bit chứa z bit 0 và 32 − z bit 1, với 0 &lt; z &lt; 32. Nó ghi vào bao nhiêu đoạn
            địa chỉ đầu ra liền nhau?
          </>
        }
      >
        <>
          One warp of 32 threads in a 1-bit pass holds z zeros and 32 − z ones, with 0 &lt; z &lt; 32. Into how many
          contiguous runs of output addresses does it write?
        </>
      </Translated>
    ),
    answer: 2,
    unit: 'runs',
    vi: { unit: 'đoạn', source: 'phân tích của khóa học này về ch. 14 §14.6, p. 340 (notes/today/ch14-facts.md A10)' },
    explain: (
      <Translated
        vi={
          <p>
            Đích đến của mỗi bit 0 lớn hơn của bit 0 trước nó một đơn vị, và các bit 1 cũng vậy, nên warp ghi một đoạn z ô liên
            tiếp và một đoạn 32 − z ô, cách xa nhau. So với một đoạn duy nhất của một lần ghi được hợp nhất, đó là chi phí mà mục{' '}
            <a href={href('ch14', '14.6')}>14.6</a> đặt ra để loại bỏ; tính theo sector 32 byte của NVIDIA thì là 4 đến 7 sector
            thay vì 4.
          </p>
        }
      >
        <p>
          Each zero's destination is one more than the previous zero's, and the same holds for the ones, so the warp
          writes one run of z consecutive slots and one run of 32 − z, far apart. Compared with one run for a coalesced
          store, that is the cost section <a href={href('ch14', '14.6')}>14.6</a> sets out to remove; in NVIDIA's 32-byte
          sectors it is 4 to 7 sectors instead of 4.
        </p>
      </Translated>
    ),
    source: 'our analysis of ch. 14 §14.6, p. 340 (notes/today/ch14-facts.md A10)',
  },
  'ch14/14.5/q-split': {
    kind: 'choice',
    id: 'ch14/14.5/q-split',
    prompt: (
      <Translated vi={<>Vì sao một lượt của lab 14.1 là ba lần khởi chạy thay vì một kernel như Hình 14.7?</>}>
        <>Why is lab 14.1's pass three launches instead of Fig. 14.7's one kernel?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Mọi khối phải ghi bit của mình trước khi bất kỳ khối nào đọc kết quả scan, và chỉ ranh giới giữa các lần khởi chạy mới sắp thứ tự các khối ở mọi nơi.">Every block must write its bits before any block reads the scan, and only a launch boundary orders blocks everywhere.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Kernel không thể gọi hàm khác, nên phép scan phải là một kernel riêng, được khởi chạy từ host giữa hai kernel kia.">Kernels cannot call other functions, so the scan has to be a kernel of its own, launched from the host between the other two.</Translated>,
        why: <Translated vi="Chúng có thể gọi hàm device. Vấn đề là việc chờ các khối khác.">They can call device functions. The problem is waiting for other blocks.</Translated>,
      },
      {
        label: <Translated vi="Phép scan cần nhiều bộ nhớ chia sẻ hơn mức một kernel được dùng, nên nó phải chạy thành các lần khởi chạy riêng.">The scan needs more shared memory than one kernel may use, so it must run as launches of its own.</Translated>,
        why: <Translated vi="Phép scan có thể nằm trong bộ nhớ toàn cục. Điều một kernel không làm được là chờ cả lưới.">The scan can live in global memory. What one kernel cannot do is wait for the whole grid.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trên CUDA và HIP, các lần khởi chạy trên một stream chạy theo thứ tự; trên Metal, các dispatch trong một serial compute
            encoder chạy theo thứ tự và thấy được các lần ghi của nhau. Một lần khởi chạy cooperative với <code>grid.sync()</code>{' '}
            cũng chạy được trên CUDA và HIP, với cái giá là lưới phải đủ nhỏ để thường trú cùng lúc; Metal không có rào chắn toàn
            lưới.
          </p>
        }
      >
        <p>
          On CUDA and HIP, launches on one stream run in order; on Metal, dispatches in one serial compute encoder run in
          order and see each other's writes. A cooperative launch with <code>grid.sync()</code> would also work on CUDA
          and HIP, at the price of a grid small enough to be resident all at once; Metal has no grid-wide barrier.
        </p>
      </Translated>
    ),
    source: 'notes/ch14.md, "Unclear in the PDF" item 6; notes/today/ch04-facts.md A7, C3; ch02-facts.md A5, B3, C9',
    vi: { source: 'notes/ch14.md, "Unclear in the PDF" mục 6; notes/today/ch04-facts.md A7, C3; ch02-facts.md A5, B3, C9' },
  },
};

export function S145() {
  const q = questionsS145;
  const cu = loaded(useLabFile(LAB1 + 'kernel.cu'));
  const metal = loaded(useLabFile(LAB1 + 'kernel.metal'));
  const mm = loaded(useLabFile(LAB1 + 'launch.mm'));
  const lab1 = useSolutionStatus('ch14/lab1_twosorts');
  const tabs = [
    cu ? { label: tr('CUDA (and HIP)', 'CUDA (và HIP)'), code: cu, tiers: TIERS, status: lab1.cu } : null,
    metal ? { label: tr('Metal kernel', 'Kernel Metal'), code: metal, tiers: TIERS, status: lab1.metal } : null,
    mm ? { label: tr('Metal host', 'Host Metal'), code: mm, status: lab1.metal } : null,
  ].filter((t) => t != null);
  return (
    <>
      <Translated
        vi={
          <p>
            Mỗi lượt cần toàn bộ kết quả của lượt trước, nên các lượt chạy lần lượt. Tính song song nằm bên trong một lượt. Vì vậy
            sách viết một kernel cho một lượt và để host khởi chạy nó một lần cho mỗi bit.
          </p>
        }
      >
        <p>
          Each pass needs the whole result of the pass before it, so the passes run one after another. The parallelism is
          inside a pass. The book therefore writes a kernel for one pass and leaves the host to launch it once per bit.
        </p>
      </Translated>
      <Source section="14.5" pages="336" />

      <h2>
        <Translated vi="Mỗi khóa một luồng">One thread per key</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Cách làm thẳng thắn nhất giao cho mỗi khóa một luồng, luồng này tính xem khóa của nó thuộc về đâu trong đầu ra và ghi nó
            vào đó. Sách vẽ mười sáu khóa của mục 14.4 với bốn khối, mỗi khối bốn luồng; luồng{' '}
            <code>i = blockIdx.x * blockDim.x + threadIdx.x</code> sở hữu khóa <code>i</code>, như mọi kernel một chiều kể từ
            chương 2. Đầu vào thật dài hơn nhiều và khối lớn hơn nhiều.
          </p>
        }
      >
        <p>
          The straightforward scheme gives every key a thread, which works out where its key belongs in the output and
          writes it there. The book draws the sixteen keys of section 14.4 with four blocks of four threads; thread{' '}
          <code>i = blockIdx.x * blockDim.x + threadIdx.x</code> owns key <code>i</code>, as in every one-dimensional
          kernel since chapter 2. Real inputs are far longer and blocks far larger.
        </p>
      </Translated>
      <Source section="14.5" pages="336–337" figs={tr('Fig. 14.5', 'Hình 14.5')} />

      <h2>
        <Translated vi="Mỗi khóa đi về đâu">Where each key goes</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một khóa có bit bằng 0 đi vào bucket 0 ở đầu. Mọi bit 0 đứng trước nó cũng đi vào đó, theo thứ tự, nên đích đến của
            nó là số bit 0 đứng trước nó, tức là chỉ số của nó trừ số bit 1 đứng trước nó. Một khóa có bit bằng 1 đi sau tất cả các
            bit 0, phía sau các bit 1 đứng trước nó.
          </p>
        }
      >
        <p>
          A key whose bit is 0 goes to the zero bucket at the front. Every zero before it also goes there, in order, so
          its destination is the number of zeros before it, which is its index minus the number of ones before it. A key
          whose bit is 1 goes after all the zeros, behind the ones before it.
        </p>
      </Translated>
      <Source section="14.5" pages="337" />
      <table className="data" data-part="formulas">
        <thead>
          <tr><th><Translated vi="Bit của khóa">The key's bit</Translated></th><th><Translated vi="Đích đến">Destination</Translated></th></tr>
        </thead>
        <tbody>
          <tr><td>0</td><td><code><Translated vi="i − số bit 1 phía trước">i − ones before</Translated></code></td></tr>
          <tr><td>1</td><td><code><Translated vi="n − tổng số bit 1 + số bit 1 phía trước">n − ones in total + ones before</Translated></code></td></tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Câu giới thiệu công thức thứ hai viết "zero bucket"; ý là bucket 1. Cả hai công thức đều cần một đại lượng mà một luồng
            không thể tự biết: có bao nhiêu bit 1 đứng trước khóa của nó. Đó là một phép scan loại trừ trên các bit, và tổng cũng
            rơi ra từ chính phép scan đó.
          </p>
        }
      >
        <p>
          The sentence that introduces the second formula says "zero bucket"; it means the one bucket. Both formulas
          need one quantity a thread cannot know on its own: how many ones come before its key. That is an exclusive
          scan of the bits, and the total falls out of the same scan.
        </p>
      </Translated>
      <Source section="14.5" pages="337–338" figs={tr('Fig. 14.6', 'Hình 14.6')} />
      <Ask q={q['ch14/14.5/q-dst-one']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={q['ch14/14.5/q-dst-zero']} label={tr('Work it out', 'Tự tính')} />

      <h2>
        <Translated vi="Kernel, từng bước một">The kernel, step by step</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hình 14.7 viết lượt này thành một kernel: tính <code>i</code>, nạp khóa, trích bit bằng{' '}
            <code>(key &gt;&gt; iter) &amp; 1</code>, ghi nó vào <code>bits</code>, gọi <code>gridExclusiveScan(bits, N)</code>,
            đọc lại "số bit 1 phía trước" và tổng, tính đích đến và ghi khóa. Ba điều trên trang sách cần sửa trước khi nó chạy
            được.
          </p>
        }
      >
        <p>
          Fig. 14.7 writes the pass as one kernel: compute <code>i</code>, load the key, extract the bit with{' '}
          <code>(key &gt;&gt; iter) &amp; 1</code>, store it in <code>bits</code>, call{' '}
          <code>gridExclusiveScan(bits, N)</code>, read "ones before" and the total back, compute the destination and
          store the key. Three things on the page need correcting before it runs.
        </p>
      </Translated>
      <Source section="14.5" pages="338–339" figs={tr('Fig. 14.7', 'Hình 14.7')} />
      <ul>
        <Translated
          vi={
            <li>
              <b><code>gridExclusiveScan</code> là mã giả.</b> Mọi luồng của lưới sẽ phải chờ ở giữa kernel cho đến khi mọi khối đã
              ghi bit của mình và phép scan đã xong. Một lần khởi chạy thông thường không hứa rằng mọi khối đều thường trú cùng lúc,
              cũng không hứa thứ tự nào giữa chúng, nên không backend nào cho một kernel chờ như vậy. Đoạn mã dưới đây tách lượt tại
              phép scan thành ba lần khởi chạy, đó là thiết kế của lab 14.1.
            </li>
          }
        >
          <li>
            <b><code>gridExclusiveScan</code> is pseudocode.</b> Every thread of the grid would have to wait in the middle
            of the kernel until every block had written its bits and the scan had finished. An ordinary launch promises
            neither that all blocks are resident at once nor any order between them, so no backend lets a kernel wait
            like that. The snippet below splits the pass at the scan into three launches, which is lab 14.1's design.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Tổng cần N + 1 phần tử.</b> Kernel đọc <code>bits[N]</code>, phần tử mà một phép scan loại trừ trên N bit không bao
              giờ ghi. Giá trị cuối của hình bằng tổng chỉ vì bit của khóa cuối bằng 0. Hãy scan N bit cùng một số 0 sau chúng, để
              phần tử N là tổng.
            </li>
          }
        >
          <li>
            <b>The total needs N + 1 entries.</b> The kernel reads <code>bits[N]</code>, which an exclusive scan of N
            bits never writes. The figure's last value equals the total only because the last key's bit is 0. Scan the N
            bits and a zero after them, so that entry N is the total.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Một lỗi đánh máy.</b> "indeed to right destination indices" ở p. 339 phải là "the right".
            </li>
          }
        >
          <li>
            <b>A typo.</b> "indeed to right destination indices" on p. 339 should read "the right".
          </li>
        </Translated>
      </ul>
      <Translated
        vi={
          <p>
            Việc tách kéo theo một thay đổi nữa: kernel thứ hai tính lại bit của mỗi khóa từ khóa, vì phép scan đã ghi đè{' '}
            <code>bits[i]</code> bằng số bit 1 đứng trước nó. Trang sách giữ <code>bit</code> trong một thanh ghi qua phép scan; khi
            phép scan là một lần khởi chạy riêng, thanh ghi không tồn tại qua nó.
          </p>
        }
      >
        <p>
          One more change follows from the split: the second kernel recomputes each key's bit from the key, because the
          scan has overwritten <code>bits[i]</code> with the ones before it. The page keeps <code>bit</code> in a
          register across the scan; once the scan is a separate launch, registers do not survive it.
        </p>
      </Translated>
      <Beyond title={tr('Our corrections', 'Các chỉnh sửa của chúng tôi')}>
        <Translated
          vi={
            <p>
              Việc tách, N + 1 phần tử và bit được tính lại là của khóa học này (notes/ch14.md, "Unclear in the PDF" mục 4 đến 6).
              Các kernel giữ tên và thứ tự của trang sách, nên các dòng tương ứng được với nhau: dòng 03 của trang sách là dòng 3 và
              14 của chúng tôi, dòng 10 của nó là dòng 33 của chúng tôi.
            </p>
          }
        >
          <p>
            The split, the N + 1 entries and the recomputed bit are ours (notes/ch14.md, "Unclear in the PDF" items 4 to
            6). The kernels keep the page's names and order, so lines map across: the page's line 03 is our lines 3 and 14,
            its line 10 is our line 33.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Hãy chạy từng bước qua lượt này. Trước hết theo một luồng qua cả hai kernel; rồi đánh dấu "dự đoán trước" và điền các
            con số của bốn luồng trước khi chạy phép scan; rồi chọn một kernel bị hỏng và chạy nó đến cuối.
          </p>
        }
      >
        <p>
          Step through the pass. Follow one thread through both kernels first; then tick "predict first" and fill in
          four threads' numbers before you run the scan; then pick a broken kernel and run it to the end.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Phép scan toàn lưới là một bước mà mọi luồng thấy hoàn tất cùng lúc. Mọi luồng cùng tiến từng dòng một, điều mà một lần
              khởi chạy thật không hứa (ở đây không sao: trong một kernel không luồng nào đọc thứ luồng khác ghi). Khối gồm bốn luồng,
              như trong sách; không có thời gian.
            </p>
          }
        >
          <p>
            The grid-wide scan is one step that every thread sees complete at once. All threads advance together line by
            line, which a real launch does not promise (it does not matter here: no thread reads what another writes
            within a kernel). Blocks of four, as in the book; no timing.
          </p>
        </Translated>
      </Simplification>
      <PassStepper />
      <Ask q={q['ch14/14.5/q-total']} />
      <Ask q={q['ch14/14.5/q-split']} />
      <Translated
        vi={
          <p>
            Bất kỳ phép scan loại trừ đúng nào cũng dùng được giữa hai kernel. Phép scan của đoạn mã là cách ngắn nhất mà vẫn đúng
            với mọi độ dài: một khối, mỗi luồng cộng một khúc. Với đầu vào dài, khối duy nhất đó là nút thắt cổ chai, và các phép
            scan nhiều kernel hoặc một lượt của chương 11 mới là lựa chọn thật.
          </p>
        }
      >
        <p>
          Any correct exclusive scan will do between the two kernels. The snippet's is the shortest that is correct for
          any length: one block, each thread summing a chunk. For long inputs that single block is the bottleneck, and
          chapter 11's multi-kernel or single-pass scans are the real choice.
        </p>
      </Translated>
      <Code
        tabs={[{ label: tr('scan', 'scan'), code: SCAN, tiers: TIERS, status: SNIPPET_STATUS() }]}
        caption={
          <Translated
            vi={
              <>
                Cả hai đoạn mã đều từ <code>labs/ch14/snippets/s145_radix_iter.cu</code>. Bài kiểm thử đơn vị của nó chạy nó trên
                trình giả lập CPU của khóa học dưới ba thứ tự luồng: các hàng của Hình 14.6 trên khóa của sách, bốn lượt của Hình 14.4,
                mọi bit của các khóa 8 bit ngẫu nhiên đối chiếu với mô hình, và 1,000 khóa 32 bit ngẫu nhiên được sắp xếp trong 32
                lượt.
              </>
            }
          >
            <>
              Both listings are from <code>labs/ch14/snippets/s145_radix_iter.cu</code>. Its unit test runs it on the
              course's CPU emulator under three thread orders: Fig. 14.6's rows on the book's keys, Fig. 14.4's four
              passes, every bit of random 8-bit keys against the model, and 1,000 random 32-bit keys sorted in 32 passes.
            </>
          </Translated>
        }
      />

      <h2>
        <Translated vi="Khóa mang theo id token">Keys that carry token ids</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một bộ lấy mẫu sắp xếp các cặp (xác suất, id token), không phải khóa trần. Đích đến chỉ phụ thuộc vào khóa, nên mang
            theo một giá trị tốn thêm một lần nạp và một lần ghi trong lần phân tán: đọc <code>values[i]</code> bên cạnh khóa và
            ghi nó vào <code>valuesOut[dst]</code>. Lab 14.3 làm điều này cho mọi hàng của một batch.
          </p>
        }
      >
        <p>
          A sampler sorts (probability, token id) pairs, not bare keys. The destination depends only on the key, so
          carrying a value costs one more load and one more store in the scatter: read <code>values[i]</code> beside the
          key and write it to <code>valuesOut[dst]</code>. Lab 14.3 does this for every row of a batch.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách định nghĩa sắp xếp khóa-giá trị (p. 330) nhưng cả hai kernel của nó đều di chuyển khóa trần; việc mang theo giá
              trị là của khóa học này (notes/ch14.md, "Keys with payloads").
            </p>
          }
        >
          <p>
            The book defines key-value sorting (p. 330) but both of its kernels move bare keys; carrying the value is
            ours (notes/ch14.md, "Keys with payloads").
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Lần phân tán làm gì với bộ nhớ">What the scatter does to memory</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Không có hai luồng nào nhận cùng một đích đến: hai công thức đánh số các bit 0 là 0, 1, 2, … và các bit 1 sau chúng,
            nên lần phân tán không cần phép toán nguyên tử nào. Nhưng các luồng liên tiếp không ghi vào các ô liên tiếp. Trong
            trình chạy từng bước, bốn luồng của khối 0 ghi vào các ô 0, 8, 1 và 9.
          </p>
        }
      >
        <p>
          No two threads get the same destination: the two formulas number the zeros 0, 1, 2, … and the ones after them,
          so the scatter needs no atomics. But consecutive threads do not write consecutive slots. In the stepper, the
          four threads of block 0 write slots 0, 8, 1 and 9.
        </p>
      </Translated>
      <Source section="14.6" pages="339–340" />
      <Ask q={q['ch14/14.5/q-runs']} />

      <h2>
        <Translated vi="Trên từng backend">On each backend</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trên CUDA và HIP, kernel đơn của sách có thể giữ được nhờ một lần khởi chạy cooperative và <code>grid.sync()</code>,
            nhưng khi đó mọi khối phải thường trú cùng lúc, điều này giới hạn kích thước lưới; trình giả lập của khóa học chỉ cho
            phép tổng cộng tối đa 1,024 luồng. Metal hoàn toàn không ghi nhận rào chắn toàn lưới nào trong tài liệu. Các lần khởi
            chạy riêng thì chạy được ở mọi nơi: các lần khởi chạy trên một stream CUDA hoặc HIP chạy theo thứ tự được phát ra, và
            các dispatch trong một serial compute encoder của Metal chạy theo thứ tự và thấy được các lần ghi của nhau.
          </p>
        }
      >
        <p>
          On CUDA and HIP the book's single kernel could survive with a cooperative launch and <code>grid.sync()</code>,
          but every block must then be resident at once, which caps the size of the grid; the course's emulator allows it
          only up to 1,024 threads in all. Metal documents no grid-wide barrier at all. Separate launches work everywhere:
          launches on one CUDA or HIP stream run in the order they were issued, and dispatches in one serial Metal compute
          encoder run in order and see each other's writes.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch04-facts.md A7, C3; infra-facts.md A4; ch02-facts.md A5, B3, C9)', '2026-11-20 (notes/today/ch04-facts.md mục A7, C3; infra-facts.md mục A4; ch02-facts.md mục A5, B3, C9)')}>
        <Translated
          vi={
            <p>
              Đồng bộ hóa toàn lưới cần <code>cudaLaunchCooperativeKernel</code> hoặc <code>hipLaunchCooperativeKernel</code> và một
              lưới vừa với thiết bị cùng lúc. Rào chắn rộng nhất của Metal là một threadgroup. Giới hạn của trình giả lập nằm trong{' '}
              <code>labs/README.md</code>.
            </p>
          }
        >
          <p>
            Grid-wide sync needs <code>cudaLaunchCooperativeKernel</code> or <code>hipLaunchCooperativeKernel</code> and a
            grid that fits on the device at once. Metal's widest barrier is one threadgroup. The emulator's limit is in{' '}
            <code>labs/README.md</code>.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Các thư viện còn đi xa hơn. Cấu trúc theo từng lượt này, một phép scan trên số đếm chữ số rồi một lần phân tán, là thứ
            CUB gọi là nhánh multi-pass của nó; mặc định của nó cho khóa 4 byte là một thiết kế một lượt gộp phép scan vào lần
            phân tán.
          </p>
        }
      >
        <p>
          The libraries go further. This per-pass structure, a scan of digit counts and then a scatter, is what CUB calls
          its multi-pass path; its default for 4-byte keys is a single-pass design that folds the scan into the scatter.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch14-facts.md, A1, B3, C4, D6)', '2026-11-20 (notes/today/ch14-facts.md, mục A1, B3, C4, D6)')}>
        <Translated
          vi={
            <p>
              Mặc định của CUB từ compute capability 6.0 cho khóa từ 4 byte trở lên là OneSweep, dùng decoupled look-back của chương
              11. rocPRIM cũng hiện thực OneSweep. Metal không ghi nhận trong tài liệu bảo đảm tiến triển nào giữa các threadgroup,
              nên look-back không có cơ sở được tài liệu hóa ở đó; radix sort MPS của PyTorch dùng các dispatch đếm, scan và phân
              tán riêng.
            </p>
          }
        >
          <p>
            CUB's default from compute capability 6.0 for keys of 4 bytes or more is OneSweep, which uses chapter 11's
            decoupled look-back. rocPRIM implements OneSweep too. Metal documents no forward progress between
            threadgroups, so a look-back has no documented basis there; PyTorch's MPS radix sort uses separate count, scan
            and scatter dispatches.
          </p>
        </Translated>
      </Today>
      {tabs.length ? (
        <Code
          tabs={tabs}
          caption={
            <Translated
              vi={
                <>
                  Lời giải tham chiếu (solution) của lab 14.1, nạp từ <code>labs/{LAB1}</code>. Hãy thử làm{' '}
                  <a href={href('ch14', 'labs')}>lab</a> trước khi đọc nó.
                </>
              }
            >
              <>
                The reference solution of lab 14.1, loaded from <code>labs/{LAB1}</code>. Try the{' '}
                <a href={href('ch14', 'labs')}>lab</a> before you read it.
              </>
            </Translated>
          }
        />
      ) : null}
    </>
  );
}
