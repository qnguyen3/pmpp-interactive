// Gaussian elimination in the book's one-thread-per-row design (§A.7, pp. 594-599), for
// section A.7's stepper, its Fig. A.11 erratum hunt and its scene.
//
// The algorithm is the book's (Figs. A.9 to A.11): for each column, every thread divides its
// own row by that row's coefficient in the column, a barrier, then every row below the
// current one subtracts the current row, a barrier. A row that already lacks the variable
// skips both (p. 598). Back substitution then runs upward one row at a time, as the figures
// draw it. With pivoting on, each column starts with a max reduction over the remaining
// rows' |coefficient| and a swap that brings the winner to the top (pp. 597-598).
//
// Arithmetic: exact rationals (BigInt fractions, with IEEE-style infinity and NaN so that a
// division by zero behaves as the hardware's would), or FP32 or FP16, every operation
// rounded once to nearest even by minifloat.ts. Not modelled: timing, warps, shared-memory
// layout, fused multiply-add (a multiply and a subtract round twice here).

import { tr } from '../lib/locale.ts';
import { FP16, FP32, add, div, mul, round, sub, type Format } from './minifloat';

// ---------------------------------------------------------------- numbers
type Q = { k: 'q'; n: bigint; d: bigint } | { k: 'inf'; neg: boolean } | { k: 'nan' };

const babs = (x: bigint) => (x < 0n ? -x : x);
function gcd(a: bigint, b: bigint): bigint {
  a = babs(a);
  b = babs(b);
  while (b) [a, b] = [b, a % b];
  return a || 1n;
}
function q(n: bigint, d: bigint = 1n): Q {
  if (d === 0n) return n === 0n ? { k: 'nan' } : { k: 'inf', neg: n < 0n };
  if (d < 0n) {
    n = -n;
    d = -d;
  }
  const g = gcd(n, d);
  return { k: 'q', n: n / g, d: d / g };
}
const NAN: Q = { k: 'nan' };
const neg = (a: Q): Q => (a.k === 'q' ? { k: 'q', n: -a.n, d: a.d } : a.k === 'inf' ? { k: 'inf', neg: !a.neg } : a);
const sgn = (a: Q) => (a.k === 'q' ? (a.n < 0n ? -1 : a.n > 0n ? 1 : 0) : a.k === 'inf' ? (a.neg ? -1 : 1) : 0);

function qadd(a: Q, b: Q): Q {
  if (a.k === 'nan' || b.k === 'nan') return NAN;
  if (a.k === 'inf' && b.k === 'inf') return a.neg === b.neg ? a : NAN;
  if (a.k === 'inf') return a;
  if (b.k === 'inf') return b;
  return q(a.n * b.d + b.n * a.d, a.d * b.d);
}
function qmul(a: Q, b: Q): Q {
  if (a.k === 'nan' || b.k === 'nan') return NAN;
  if (a.k === 'inf' || b.k === 'inf') {
    const s = sgn(a) * sgn(b);
    return s === 0 ? NAN : { k: 'inf', neg: s < 0 };
  }
  return q(a.n * b.n, a.d * b.d);
}
function qdiv(a: Q, b: Q): Q {
  if (a.k === 'nan' || b.k === 'nan') return NAN;
  if (a.k === 'inf' && b.k === 'inf') return NAN;
  if (a.k === 'inf') return { k: 'inf', neg: a.neg !== (sgn(b) < 0) };
  if (b.k === 'inf') return q(0n);
  return q(a.n * b.d, a.d * b.n); // b = 0 gives infinity, or NaN for 0 / 0 (zero counts as +0)
}
const qnum = (a: Q) => (a.k === 'q' ? Number(a.n) / Number(a.d) : a.k === 'inf' ? (a.neg ? -Infinity : Infinity) : NaN);

/** Text for a number: "19/3", "-1/6", "0", "inf", "NaN"; long fractions as an approximate decimal. */
function qtext(a: Q): string {
  if (a.k === 'nan') return 'NaN';
  if (a.k === 'inf') return a.neg ? '-inf' : 'inf';
  if (a.d === 1n && babs(a.n) < 10n ** 9n) return String(a.n);
  if (String(babs(a.n)).length + String(a.d).length <= 8) return `${a.n}/${a.d}`;
  return '≈' + fmtFloat(qnum(a), 11);
}

function fmtFloat(v: number, digits: number): string {
  if (Number.isNaN(v)) return 'NaN';
  if (v === Infinity) return 'inf';
  if (v === -Infinity) return '-inf';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (Number.isInteger(v) && a < 1e7) return String(v);
  if (a >= 1e7 || a < 1e-4) return v.toExponential(digits - 1).replace(/\.?0+e/, 'e');
  return String(Number(v.toPrecision(digits)));
}

export type ArithId = 'exact' | 'fp32' | 'fp16';
export const ARITH_LABEL: Record<ArithId, string> = {
  get exact() {
    return tr('exact fractions', 'phân số chính xác');
  },
  fp32: 'FP32',
  fp16: 'FP16',
};

interface Arith<T> {
  of(n: bigint, d: bigint): T;
  add(a: T, b: T): T;
  sub(a: T, b: T): T;
  mul(a: T, b: T): T;
  div(a: T, b: T): T;
  isZero(a: T): boolean;
  /** |a| > |b|; NaN is never greater. */
  absGreater(a: T, b: T): boolean;
  abs(a: T): T;
  num(a: T): number;
  text(a: T): string;
}

const EXACT: Arith<Q> = {
  of: (n, d) => q(n, d),
  add: qadd,
  sub: (a, b) => qadd(a, neg(b)),
  mul: qmul,
  div: qdiv,
  isZero: (a) => a.k === 'q' && a.n === 0n,
  absGreater: (a, b) => {
    if (a.k === 'nan' || b.k === 'nan') return false;
    if (a.k === 'inf') return b.k !== 'inf';
    if (b.k === 'inf') return false;
    return babs(a.n) * b.d > babs(b.n) * a.d;
  },
  abs: (a) => (a.k === 'q' ? { k: 'q', n: babs(a.n), d: a.d } : a.k === 'inf' ? { k: 'inf', neg: false } : a),
  num: qnum,
  text: qtext,
};

function floatArith(f: Format, digits: number): Arith<number> {
  return {
    of: (n, d) => round(f, Number(n) / Number(d)),
    add: (a, b) => add(f, a, b),
    sub: (a, b) => sub(f, a, b),
    mul: (a, b) => mul(f, a, b),
    div: (a, b) => div(f, a, b),
    isZero: (a) => a === 0,
    absGreater: (a, b) => Math.abs(a) > Math.abs(b),
    abs: (a) => Math.abs(a),
    num: (a) => a,
    text: (a) => fmtFloat(a, digits),
  };
}
const ARITH: Record<ArithId, Arith<unknown>> = {
  exact: EXACT as Arith<unknown>,
  fp32: floatArith(FP32, 8) as Arith<unknown>,
  fp16: floatArith(FP16, 5) as Arith<unknown>,
};

// ---------------------------------------------------------------- systems
export type SystemId = 'book1' | 'book2' | 'tiny';
export interface SystemDef {
  id: SystemId;
  label: string;
  vars: string[];
  /** Augmented rows as exact fractions [numerator, denominator]. */
  rows: [bigint, bigint][][];
  /** The exact solution, as text. */
  solution: string[];
}
const I = (n: number): [bigint, bigint] => [BigInt(n), 1n];

/** The book's first system (p. 594, Figs. A.9 and A.10). */
const BOOK1: SystemDef = {
  id: 'book1',
  get label() {
    return tr("The book's first system (Figs. A.9, A.10)", 'Hệ thứ nhất của sách (Hình A.9, A.10)');
  },
  vars: ['X', 'Y', 'Z'],
  rows: [
    [I(3), I(5), I(2), I(19)],
    [I(2), I(3), I(1), I(11)],
    [I(1), I(2), I(2), I(11)],
  ],
  solution: ['1', '2', '3'],
};
/** The book's second system: equation 1 replaced by 5Y + 2Z = 16 (p. 597, Fig. A.11). */
const BOOK2: SystemDef = {
  id: 'book2',
  get label() {
    return tr("The book's second system (Fig. A.11)", 'Hệ thứ hai của sách (Hình A.11)');
  },
  vars: ['X', 'Y', 'Z'],
  rows: [
    [I(0), I(5), I(2), I(16)],
    [I(2), I(3), I(1), I(11)],
    [I(1), I(2), I(2), I(11)],
  ],
  solution: ['1', '2', '3'],
};

/** ε = 10^-k, k = 1 to 10. */
export const EPS_EXPONENTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

/** Ours, not the book's: [[ε, 1], [1, 1]] x = [1, 2]; x1 = 1 / (1 - ε), x2 = (1 - 2ε) / (1 - ε). */
export function tinySystem(k: number): SystemDef {
  const d = 10n ** BigInt(k);
  const x1 = q(d, d - 1n);
  const x2 = q(d - 2n, d - 1n);
  return {
    id: 'tiny',
    label: tr('2 × 2 with a tiny pivot ε (ours)', '2 × 2 với pivot ε rất nhỏ (của khóa học này)'),
    vars: ['x1', 'x2'],
    rows: [
      [[1n, d], I(1), I(1)],
      [I(1), I(1), I(2)],
    ],
    solution: [qtext(x1), qtext(x2)],
  };
}

export function system(id: SystemId, epsExp = 8): SystemDef {
  return id === 'book1' ? BOOK1 : id === 'book2' ? BOOK2 : tinySystem(epsExp);
}

// ---------------------------------------------------------------- the trace
export type ThreadState = 'ready' | 'working' | 'skipping' | 'idle' | 'waiting' | 'searching' | 'finished';
export type PhaseKind = 'start' | 'pivot' | 'swap' | 'divide' | 'subtract' | 'barrier' | 'backsub' | 'done';

export interface Cell {
  v: number;
  text: string;
}
export interface Phase {
  kind: PhaseKind;
  /** The column being eliminated; for backsub, the row being solved. */
  col: number;
  /** One state per thread; thread t owns row t (the book's row t + 1). */
  threads: ThreadState[];
  /** The augmented matrix after this phase. */
  m: Cell[][];
  /** divide: each row's divisor, null where its thread did not divide. */
  divisors?: (Cell | null)[];
  /** pivot: the max reduction over |coefficient| in rows col..n-1. */
  pivot?: { rows: number[]; abs: Cell[]; levels: number[][]; winner: number };
  swap?: [number, number];
  /** backsub: the columns whose known values were substituted. */
  subst?: number[];
  /** The panel of the book's figure this phase produces, if any: 'Pivoting', 'Step 1', ... */
  step?: string;
  note: string;
}
export interface TraceOpts {
  pivot: boolean;
  arith: ArithId;
}
export interface Trace {
  sys: SystemDef;
  opts: TraceOpts;
  n: number;
  phases: Phase[];
  /** The right-hand column at the end. */
  solution: Cell[];
  /** No infinity or NaN anywhere at the end. */
  finite: boolean;
}

const rowName = (i: number) => tr(`row ${i + 1}`, `hàng ${i + 1}`);
const list = (xs: string[]) => (xs.length <= 1 ? xs.join('') : xs.slice(0, -1).join(', ') + tr(' and ', ' và ') + xs[xs.length - 1]);
const uni = (s: string) => s.replace(/-/g, '−');

export function trace(sys: SystemDef, opts: TraceOpts): Trace {
  const A = ARITH[opts.arith];
  const n = sys.rows.length;
  const m: unknown[][] = sys.rows.map((r) => r.map(([a, b]) => A.of(a, b)));
  const phases: Phase[] = [];
  const cells = (): Cell[][] => m.map((r) => r.map((x) => ({ v: A.num(x), text: A.text(x) })));
  const cell = (x: unknown): Cell => ({ v: A.num(x), text: A.text(x) });
  const V = sys.vars;
  let stepNo = 0;
  const solved = new Set<number>();
  // Every sentence of a note starts with a capital ("row 2 ..." becomes "Row 2 ...").
  const cap = (s: string) => s.replace(/(^|\. )([a-z])/g, (_, a: string, b: string) => a + b.toUpperCase());
  const push = (p: Omit<Phase, 'm'>) => phases.push({ ...p, note: cap(p.note), m: cells() });
  const barrier = (why: string) =>
    push({ kind: 'barrier', col: -1, threads: Array(n).fill('waiting'), note: tr(`Barrier: all ${n} threads wait here ${why}.`, `Rào chắn: cả ${n} luồng chờ ở đây ${why}.`) });
  const base = (i: number): ThreadState => (solved.has(i) ? 'finished' : 'idle');

  push({ kind: 'start', col: 0, threads: Array(n).fill('ready'), note: tr(`Thread t owns row t + 1 and does all the work on it. Nothing has run yet.`, `Luồng t sở hữu hàng t + 1 và làm mọi việc trên hàng đó. Chưa có gì chạy.`) });

  for (let k = 0; k < n; k++) {
    if (opts.pivot) {
      const rows = Array.from({ length: n - k }, (_, j) => k + j);
      const levels: number[][] = [rows];
      let cur = rows;
      while (cur.length > 1) {
        const next: number[] = [];
        for (let j = 0; j < cur.length; j += 2) {
          if (j + 1 === cur.length) next.push(cur[j]);
          else next.push(A.absGreater(m[cur[j + 1]][k], m[cur[j]][k]) ? cur[j + 1] : cur[j]);
        }
        levels.push(next);
        cur = next;
      }
      const winner = cur[0];
      const abs = rows.map((i) => cell(A.abs(m[i][k])));
      push({
        kind: 'pivot',
        col: k,
        threads: Array.from({ length: n }, (_, i) => (i >= k ? 'searching' : base(i))),
        pivot: { rows, abs, levels, winner },
        note:
          tr(`Pivot search for ${V[k]}: a max reduction over |coefficient of ${V[k]}| in ${list(rows.map(rowName))}: `, `Tìm pivot cho ${V[k]}: một phép rút gọn max trên |hệ số của ${V[k]}| trong ${list(rows.map(rowName))}: `) +
          `${list(abs.map((c) => c.text))}. ` +
          (winner === k
            ? tr(`${rowName(k)} already holds the largest, so no swap is needed.`, `${rowName(k)} đã giữ giá trị lớn nhất, nên không cần hoán đổi.`)
            : tr(`${rowName(winner)} wins.`, `${rowName(winner)} thắng.`)),
      });
      if (winner !== k) {
        [m[k], m[winner]] = [m[winner], m[k]];
        push({
          kind: 'swap',
          col: k,
          threads: Array.from({ length: n }, (_, i) => (i === k || i === winner ? 'working' : base(i))),
          swap: [k, winner],
          step: 'Pivoting', // i18n-ignore: identifier also shown to the reader
          note: tr(
            `Threads ${k} and ${winner} exchange their rows: ${rowName(winner)} moves to the top of what is left.`,
            `Luồng ${k} và ${winner} đổi hàng cho nhau: ${rowName(winner)} chuyển lên đầu phần còn lại.`,
          ),
        });
      }
      barrier(winner !== k ? tr('so that every thread sees the swapped rows', 'để mọi luồng đều thấy các hàng đã hoán đổi') : tr('until the search is over', 'cho đến khi việc tìm kiếm kết thúc'));
    }

    // Divide: each remaining row by its own coefficient in column k.
    const divisors: (Cell | null)[] = Array(n).fill(null);
    const threads: ThreadState[] = Array.from({ length: n }, (_, i) => base(i));
    const divided: string[] = [];
    const skipped: number[] = [];
    let zeroPivot = false;
    for (let i = k; i < n; i++) {
      const lead = m[i][k];
      if (i > k && A.isZero(lead)) {
        threads[i] = 'skipping';
        skipped.push(i);
        continue;
      }
      if (A.isZero(lead)) zeroPivot = true;
      threads[i] = 'working';
      divisors[i] = cell(lead);
      divided.push(tr(`${rowName(i)} by ${uni(A.text(lead))}`, `${rowName(i)} cho ${uni(A.text(lead))}`));
      m[i] = m[i].map((x, c) => (c < k ? x : A.div(x, lead)));
    }
    stepNo += 1;
    if (k === n - 1) solved.add(k);
    push({
      kind: 'divide',
      col: k,
      threads,
      divisors,
      step: `Step ${stepNo}`, // i18n-ignore: identifier also shown to the reader
      note:
        tr(`Each thread divides its own row by that row's coefficient of ${V[k]}: ${list(divided)}.`, `Mỗi luồng chia hàng của mình cho hệ số của ${V[k]} trong hàng đó: ${list(divided)}.`) +
        (skipped.length
          ? tr(
              ` ${list(skipped.map(rowName))} ${skipped.length > 1 ? 'have' : 'has'} no ${V[k]}, so ${skipped.length > 1 ? 'their threads skip' : 'its thread skips'} the division.`,
              ` ${list(skipped.map(rowName))} không có ${V[k]}, nên luồng của ${skipped.length > 1 ? 'chúng' : 'nó'} bỏ qua phép chia.`,
            )
          : '') +
        (zeroPivot ? tr(` ${rowName(k)}'s coefficient is 0: x / 0 gives infinity and 0 / 0 gives NaN.`, ` Hệ số của ${rowName(k)} bằng 0: x / 0 cho ra vô cực và 0 / 0 cho ra NaN.`) : '') +
        (k === n - 1 ? tr(` ${rowName(k)} now reads ${V[k]} = ${uni(A.text(m[k][n]))}.`, ` ${rowName(k)} giờ là ${V[k]} = ${uni(A.text(m[k][n]))}.`) : ''),
    });
    barrier(tr('until every division is written', 'cho đến khi mọi phép chia đã được ghi'));

    if (k < n - 1) {
      const sThreads: ThreadState[] = Array.from({ length: n }, (_, i) => base(i));
      const subtracted: number[] = [];
      const sSkipped: number[] = [];
      for (let i = k + 1; i < n; i++) {
        if (A.isZero(m[i][k])) {
          sThreads[i] = 'skipping';
          sSkipped.push(i);
          continue;
        }
        sThreads[i] = 'working';
        subtracted.push(i);
        m[i] = m[i].map((x, c) => (c < k ? x : A.sub(x, m[k][c])));
      }
      stepNo += 1;
      push({
        kind: 'subtract',
        col: k,
        threads: sThreads,
        step: `Step ${stepNo}`, // i18n-ignore: identifier also shown to the reader
        note:
          (subtracted.length
            ? tr(
                `${list(subtracted.map(rowName))} subtract${subtracted.length > 1 ? '' : 's'} ${rowName(k)}, so ${V[k]} leaves ${subtracted.length > 1 ? 'them' : 'it'}.`,
                `${list(subtracted.map(rowName))} trừ đi ${rowName(k)}, nên ${V[k]} biến mất khỏi ${subtracted.length > 1 ? 'chúng' : 'nó'}.`,
              )
            : tr(`No row below has ${V[k]}.`, `Không hàng nào bên dưới có ${V[k]}.`)) +
          (sSkipped.length
            ? tr(
                ` ${list(sSkipped.map(rowName))} already ${sSkipped.length > 1 ? 'lack' : 'lacks'} ${V[k]} and ${sSkipped.length > 1 ? 'skip' : 'skips'}.`,
                ` ${list(sSkipped.map(rowName))} vốn đã không có ${V[k]} và bỏ qua.`,
              )
            : '') +
          tr(` Thread ${k} is done until back substitution: one more idle thread.`, ` Luồng ${k} xong việc cho đến bước thế ngược: thêm một luồng rảnh.`),
      });
      barrier(tr('until every subtraction is written', 'cho đến khi mọi phép trừ đã được ghi'));
    }
  }

  // Back substitution, upward, one row per step, as Figs. A.9 to A.11 draw it.
  for (let j = n - 2; j >= 0; j--) {
    const subst: number[] = [];
    for (let c = j + 1; c < n; c++) {
      const f = m[j][c];
      if (A.isZero(f)) continue;
      subst.push(c);
      m[j][c] = A.sub(m[j][c], A.mul(f, m[c][c]));
      m[j][n] = A.sub(m[j][n], A.mul(f, m[c][n]));
    }
    solved.add(j);
    stepNo += 1;
    push({
      kind: 'backsub',
      col: j,
      threads: Array.from({ length: n }, (_, i) => (i === j ? 'working' : i > j ? 'finished' : 'idle')),
      subst,
      step: `Step ${stepNo}`, // i18n-ignore: identifier also shown to the reader
      note:
        tr(
          `Thread ${j} substitutes ${list(subst.map((c) => `${V[c]} = ${uni(A.text(m[c][n]))}`)) || 'nothing'} into ${rowName(j)}: `,
          `Luồng ${j} thế ${list(subst.map((c) => `${V[c]} = ${uni(A.text(m[c][n]))}`)) || 'không gì'} vào ${rowName(j)}: `,
        ) +
        `${V[j]} = ${uni(A.text(m[j][n]))}.`,
    });
    if (j > 0) barrier(tr('before the next row up reads this one', 'trước khi hàng kế tiếp phía trên đọc hàng này'));
  }
  const solution = m.map((r) => cell(r[n]));
  push({
    kind: 'done',
    col: -1,
    threads: Array(n).fill('finished'),
    note: tr(`Solution: ${list(V.map((v, i) => `${v} = ${uni(solution[i].text)}`))}.`, `Nghiệm: ${list(V.map((v, i) => `${v} = ${uni(solution[i].text)}`))}.`),
  });
  const last = phases[phases.length - 1].m;
  return { sys, opts, n, phases, solution, finite: last.every((r) => r.every((c) => Number.isFinite(c.v))) };
}

// ---------------------------------------------------------------- thread utilisation
/** Fraction of thread slots doing work over the divide and subtract phases of a trace. */
export function busyFraction(t: Trace): number {
  const ph = t.phases.filter((p) => p.kind === 'divide' || p.kind === 'subtract');
  const busy = ph.reduce((s, p) => s + p.threads.filter((x) => x === 'working').length, 0);
  return busy / (ph.length * t.n);
}
/** The same for a dense n × n system: divides keep n, n-1, ..., 1 threads busy and subtracts n-1, ..., 1, so n² of n(2n - 1). */
export const elimBusyFraction = (n: number) => n / (2 * n - 1);

// ---------------------------------------------------------------- the book's figures
export type Caption =
  | { op: 'none' }
  | { op: 'swap'; rows: [number, number] }
  | { op: 'divide'; by: [number, string][] }
  | { op: 'subtract'; minus: number; from: number[]; skipped: number[] }
  | { op: 'substitute'; row: number; vars: string[] };

export interface Panel {
  label: string;
  caption: Caption;
  /** Augmented rows as text, "0" for an empty position. */
  rows: string[][];
  /** "Z" on the panel that solves for Z, and so on. */
  solved?: string;
}

/** Panels as the book's figures lay them out, computed from a trace (rows numbered from 1). */
export function panelsOf(t: Trace): Panel[] {
  const V = t.sys.vars;
  const out: Panel[] = [{ label: 'Original', caption: { op: 'none' }, rows: t.phases[0].m.map((r) => r.map((c) => c.text)) }]; // i18n-ignore: identifier also shown to the reader
  for (const p of t.phases) {
    if (!p.step) continue;
    let caption: Caption = { op: 'none' };
    let solved: string | undefined;
    if (p.kind === 'swap') caption = { op: 'swap', rows: [p.swap![0] + 1, p.swap![1] + 1] };
    if (p.kind === 'divide') {
      caption = {
        op: 'divide',
        by: p.divisors!.flatMap((d, i) => (d && d.text !== '1' ? [[i + 1, d.text] as [number, string]] : [])),
      };
      if (p.col === t.n - 1) solved = V[p.col];
    }
    if (p.kind === 'subtract')
      caption = {
        op: 'subtract',
        minus: p.col + 1,
        from: p.threads.flatMap((s, i) => (s === 'working' ? [i + 1] : [])),
        skipped: p.threads.flatMap((s, i) => (s === 'skipping' ? [i + 1] : [])),
      };
    if (p.kind === 'backsub') {
      caption = { op: 'substitute', row: p.col + 1, vars: p.subst!.map((c) => V[c]) };
      solved = V[p.col];
    }
    out.push({ label: p.step, caption, rows: p.m.map((r) => r.map((c) => c.text)), ...(solved ? { solved } : {}) });
  }
  return out;
}

/** Figs. A.10 (and A.9, the same steps as equations) and A.11, computed. */
export function bookPanels(fig: 'A.10' | 'A.11'): Panel[] {
  return fig === 'A.10'
    ? panelsOf(trace(BOOK1, { pivot: false, arith: 'exact' }))
    : panelsOf(trace(BOOK2, { pivot: true, arith: 'exact' }));
}

const R = (s: string) => s.split(' ');
/** Fig. A.11 as printed (p. 598), transcribed from notes/appA.md; captions shortened to their content. */
export const printedA11: Panel[] = [
  { label: 'Original', caption: { op: 'none' }, rows: [R('0 5 2 16'), R('2 3 1 11'), R('1 2 2 11')] }, // i18n-ignore: identifier also shown to the reader
  { label: 'Pivoting', caption: { op: 'swap', rows: [1, 2] }, rows: [R('2 3 1 11'), R('0 5 2 16'), R('1 2 2 11')] }, // i18n-ignore: identifier also shown to the reader
  { label: 'Step 1', caption: { op: 'divide', by: [[1, '3']] }, rows: [R('1 3/2 1/2 11/2'), R('0 5 2 16'), R('1 2 2 11')] }, // i18n-ignore: identifier also shown to the reader
  { label: 'Step 2', caption: { op: 'subtract', minus: 1, from: [3], skipped: [2] }, rows: [R('1 3/2 1/2 11/2'), R('0 5 2 16'), R('0 1/2 3/2 11/2')] }, // i18n-ignore: identifier also shown to the reader
  { label: 'Step 3', caption: { op: 'divide', by: [[2, '5'], [3, '1/2']] }, rows: [R('1 3/2 1/2 11/2'), R('0 1 2/5 16/5'), R('0 1 3 11')] }, // i18n-ignore: identifier also shown to the reader
  { label: 'Step 4', caption: { op: 'subtract', minus: 2, from: [3], skipped: [] }, rows: [R('1 3/2 1/2 11/2'), R('0 1 2/5 16/5'), R('0 0 13/5 39/5')] }, // i18n-ignore: identifier also shown to the reader
  { label: 'Step 5', caption: { op: 'divide', by: [[3, '13/5']] }, rows: [R('1 5/3 2/3 19/3'), R('0 1 2/5 16/5'), R('0 0 1 3')], solved: 'Z' }, // i18n-ignore: identifier also shown to the reader
  { label: 'Step 6', caption: { op: 'substitute', row: 2, vars: ['Z'] }, rows: [R('1 5/3 2/3 19/3'), R('0 1 0 2'), R('0 0 1 3')], solved: 'Y' }, // i18n-ignore: identifier also shown to the reader
  { label: 'Step 7', caption: { op: 'substitute', row: 1, vars: ['Y', 'Z'] }, rows: [R('1 0 0 1'), R('0 1 0 2'), R('0 0 1 3')], solved: 'X' }, // i18n-ignore: identifier also shown to the reader
];

export interface PanelDiff {
  panel: string;
  what: 'caption' | 'cell';
  row?: number;
  col?: number;
  printed: string;
  computed: string;
}

/** Caption text, the same template for printed and computed panels. */
export function captionText(c: Caption, vars: string[] = ['X', 'Y', 'Z']): string {
  switch (c.op) {
    case 'none':
      return '';
    case 'swap':
      return tr(`swap row ${c.rows[0]} with row ${c.rows[1]}`, `hoán đổi hàng ${c.rows[0]} với hàng ${c.rows[1]}`);
    case 'divide':
      return c.by.length
        ? tr('divide ', 'chia ') + list(c.by.map(([r, d]) => tr(`row ${r} by ${uni(d)}`, `hàng ${r} cho ${uni(d)}`)))
        : tr('no division needed', 'không cần chia');
    case 'subtract':
      return (
        tr(`subtract row ${c.minus} from ${list(c.from.map((r) => `row ${r}`))}`, `lấy ${list(c.from.map((r) => `hàng ${r}`))} trừ hàng ${c.minus}`) +
        (c.skipped.length
          ? tr(
              `; ${list(c.skipped.map((r) => `row ${r}`))} already ${c.skipped.length > 1 ? 'lack' : 'lacks'} ${vars[c.minus - 1]}`,
              `; ${list(c.skipped.map((r) => `hàng ${r}`))} vốn đã không có ${vars[c.minus - 1]}`,
            )
          : '')
      );
    case 'substitute':
      return tr(`substitute ${list(c.vars)} into row ${c.row}`, `thế ${list(c.vars)} vào hàng ${c.row}`);
  }
}

/** Where a printed figure differs from the computed one: captions and cells. */
export function diffPanels(printed: Panel[], computed: Panel[]): PanelDiff[] {
  const out: PanelDiff[] = [];
  printed.forEach((p, i) => {
    const c = computed[i];
    if (JSON.stringify(p.caption) !== JSON.stringify(c.caption))
      out.push({ panel: p.label, what: 'caption', printed: captionText(p.caption), computed: captionText(c.caption) }); // i18n-ignore: union member
    p.rows.forEach((row, r) =>
      row.forEach((v, k) => {
        if (v !== c.rows[r][k]) out.push({ panel: p.label, what: 'cell', row: r, col: k, printed: v, computed: c.rows[r][k] }); // i18n-ignore: union member
      }),
    );
  });
  return out;
}

/** One augmented row as an equation: "3X + 5Y + 2Z = 19", zero terms left out, the book's style. */
export function equation(row: string[], vars: string[]): string {
  const terms: string[] = [];
  row.slice(0, -1).forEach((t, i) => {
    if (t === '0') return;
    const negTerm = t.startsWith('-') && t !== '-inf';
    const mag = negTerm ? t.slice(1) : t;
    const coef = mag === '1' ? '' : /^[\d/.]+$/.test(mag) ? mag : `(${uni(mag)})`;
    const term = coef + vars[i];
    if (!terms.length) terms.push((negTerm ? '−' : '') + term);
    else terms.push((negTerm ? '− ' : '+ ') + term);
  });
  return `${terms.join(' ') || '0'} = ${uni(row[row.length - 1])}`;
}
