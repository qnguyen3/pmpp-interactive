// Speed-up arithmetic for chapter 1 (book §1.3, pp. 9–10): Amdahl's law, and a
// second, cruder model of what caps the speed-up of the accelerated part itself.
//
// The formula is ours. The book prints no formula for Amdahl's law, only three
// worked cases (p. 9): 30% made 100 times faster gives about 1.42; 30% made
// infinitely fast gives no more than 1.43; 99% made 100 times faster gives about
// 50. speedup() reproduces all three (notes/today/ch01-facts.md, A16).
//
// What the model assumes, and what the app says wherever it is shown:
//   - A run has two parts only: one you speed up (a fraction p of the original
//     time) and one you leave alone (1 - p).
//   - The two parts do not overlap in time. Chapter 22 (§22.3, pp. 537–538)
//     relaxes this with host and device work running at once: see overlapped()
//     and moduleSpeedup() at the end of this file, which chapter 22 added.
//   - The untouched part stays the same size. In a real port, moving work to a
//     GPU adds copies and kernel launches to it.
//   - The problem size is fixed.
//   - In the "what caps the kernel" model the accelerated part's speed-up is
//     min(threads x rate per thread, memory cap): threads scale perfectly until
//     memory saturates, the rate per thread is a constant, and the memory cap is
//     a single given number. The book states "about 10x" for that cap (p. 10)
//     and does not derive it; chapter 5 computes it for a kernel and a device.
//
// No imports and no syntax that needs compiling away beyond types, so the
// browser checks can load this file directly.

/**
 * Overall speed-up when a fraction `p` of the original run time is made `s`
 * times faster and the rest is left alone. `s` may be Infinity.
 */
export function speedup(p: number, s: number): number {
  return 1 / (1 - p + p / s);
}

/** The most `speedup(p, s)` can reach, however large `s` is: 1 / (1 - p). */
export function ceiling(p: number): number {
  return 1 / (1 - p);
}

export interface Remaining {
  /** The part left alone, as a share of the original time: 1 - p. */
  untouched: number;
  /** The accelerated part after the speed-up, as a share of the original time: p / s. */
  accelerated: number;
  /** Their sum: the new run time as a share of the original. */
  total: number;
}

/** What is left of the original run time, split into its two parts. */
export function remaining(p: number, s: number): Remaining {
  const untouched = 1 - p;
  const accelerated = p / s;
  return { untouched, accelerated, total: untouched + accelerated };
}

/**
 * The share of the remaining time that is the untouched part. It equals
 * speedup(p, s) / ceiling(p): how much of the ceiling has been reached.
 */
export function untouchedShare(p: number, s: number): number {
  const r = remaining(p, s);
  return r.total === 0 ? 0 : r.untouched / r.total;
}

// Two values that differ by less than this (relatively) are treated as equal,
// so that "exactly at the ceiling" survives binary floating point.
const EPS = 1e-9;
const same = (a: number, b: number) => Math.abs(a - b) <= EPS * Math.max(Math.abs(a), Math.abs(b), 1);

/**
 * The parallel fraction at which a part made `s` times faster gives `target`
 * overall. null when no fraction does (the target is below 1 or above `s`).
 */
export function fractionNeeded(target: number, s: number): number | null {
  if (!(target >= 1) || !(s >= 1)) return null;
  if (same(target, 1)) return 0;
  if (s === Infinity) return 1 - 1 / target;
  if (same(target, s)) return 1;
  if (target > s) return null;
  return (1 - 1 / target) / (1 - 1 / s);
}

/**
 * How many times faster a part that takes a fraction `p` of the time must
 * become to give `target` overall. Infinity when the target is the ceiling
 * itself; null when it is above the ceiling.
 */
export function kernelNeeded(target: number, p: number): number | null {
  if (!(target > 0)) return null;
  const top = ceiling(p);
  if (same(target, top)) return Infinity;
  if (target > top) return null;
  return p / (1 / target - (1 - p));
}

// ------------------------------------------------- what caps the kernel itself

export type Limiter = 'threads' | 'memory';

export interface Capped {
  /** Speed-up of the accelerated part over the one-core baseline. */
  s: number;
  /** Which of the two ceilings set it. */
  limitedBy: Limiter;
}

/**
 * Speed-up of a kernel over a one-core CPU baseline when `threads` threads each
 * run at `ratePerThread` times the speed of that core, and memory saturates at
 * a speed-up of `memoryCap`. Below 1 when there are too few threads to match
 * the core.
 */
export function cappedKernel(threads: number, ratePerThread: number, memoryCap: number): Capped {
  const byThreads = threads * ratePerThread;
  return byThreads < memoryCap ? { s: byThreads, limitedBy: 'threads' } : { s: memoryCap, limitedBy: 'memory' };
}

export interface Limits extends Capped {
  /** Overall speed-up from Amdahl's law with that `s`. */
  overall: number;
  /** Share of the remaining time spent in the untouched part. */
  untouchedShare: number;
}

/** cappedKernel() followed by Amdahl's law for a parallel fraction `p`. */
export function limits(p: number, threads: number, ratePerThread: number, memoryCap: number): Limits {
  const k = cappedKernel(threads, ratePerThread, memoryCap);
  return { ...k, overall: speedup(p, k.s), untouchedShare: untouchedShare(p, k.s) };
}

/**
 * The two thread counts at which the picture changes: below `breakEven` the
 * kernel is slower than the CPU core, and from `saturation` on, memory sets
 * its speed and more threads change nothing.
 */
export function corners(ratePerThread: number, memoryCap: number): { breakEven: number; saturation: number } {
  return { breakEven: 1 / ratePerThread, saturation: memoryCap / ratePerThread };
}

/**
 * What each control would buy on its own, everything else held: `factor`
 * times the threads, `factor` times the memory cap, or an untouched part cut
 * to 1/`factor` of its size. Each entry is the resulting overall speed-up.
 */
export function levers(
  p: number,
  threads: number,
  ratePerThread: number,
  memoryCap: number,
  factor = 10,
): { now: number; threads: number; memory: number; untouched: number } {
  return {
    now: limits(p, threads, ratePerThread, memoryCap).overall,
    threads: limits(p, threads * factor, ratePerThread, memoryCap).overall,
    memory: limits(p, threads, ratePerThread, memoryCap * factor).overall,
    untouched: limits(1 - (1 - p) / factor, threads, ratePerThread, memoryCap).overall,
  };
}

// ------------------------------------------------------------- slider ladders
// Log-spaced stops with round values, so that a slider can land exactly on the
// cases the lesson talks about (30%, 99%, 99.9%; 10, 100, 1,000).

const tidy = (v: number) => Number(v.toPrecision(6));

function ladder(lo: number, hi: number, mantissas: number[]): number[] {
  const out: number[] = [];
  for (let e = Math.floor(Math.log10(lo)) - 1; e <= Math.ceil(Math.log10(hi)); e++)
    for (const m of mantissas) {
      const v = tidy(m * 10 ** e);
      if (v >= lo * (1 - EPS) && v <= hi * (1 + EPS)) out.push(v);
    }
  return [...new Set(out)].sort((a, b) => a - b);
}

/**
 * Parallel fractions from 30% to 99.99%, spaced evenly in the logarithm of the
 * sequential share 1 - p, so the stops get finer as p approaches 1.
 */
export const FRACTIONS: number[] = ladder(0.0001, 0.7, [1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9])
  .map((q) => tidy(1 - q))
  .sort((a, b) => a - b);

/** Speed-ups of the accelerated part, and memory caps: 1 to 1,000. */
export const SPEEDUPS: number[] = ladder(1, 1000, [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8]);

/** Thread counts: 1 to 100,000, whole numbers only. */
export const THREADS: number[] = ladder(1, 100000, [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8]).filter((v) => Number.isInteger(v));

/** Speed of one GPU thread relative to the CPU core of the baseline. */
export const RATES: { value: number; label: string }[] = [
  { value: 1 / 8, label: '1/8' },
  { value: 1 / 4, label: '1/4' },
  { value: 1 / 2, label: '1/2' },
  { value: 1, label: '1' },
];

/** Index of the stop closest to `value`, compared on a log scale. */
export function nearestIndex(stops: number[], value: number, of: (v: number) => number = (v) => v): number {
  let best = 0;
  let bestD = Infinity;
  stops.forEach((v, i) => {
    const d = Math.abs(Math.log(of(v)) - Math.log(of(value)));
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  });
  return best;
}

/** Index into FRACTIONS of the stop closest to `p` (compared on the sequential share). */
export const fractionIndex = (p: number) => nearestIndex(FRACTIONS, p, (v) => 1 - v);

// ---------------------------------------------------------------- the book's cases

export interface BookCase {
  label: string;
  p: number;
  s: number;
  /** How the book states the result (p. 9). */
  book: string;
}

/** The three cases the book works on p. 9. */
export const BOOK_CASES: BookCase[] = [
  { label: '30%, 100×', p: 0.3, s: 100, book: 'about 1.42' }, // i18n-ignore: no-import file loaded by e2e; Wrap.tsx translates
  { label: '30%, unbounded', p: 0.3, s: Infinity, book: 'no more than 1.43' }, // i18n-ignore: no-import file loaded by e2e; S13/Wrap translate
  { label: '99%, 100×', p: 0.99, s: 100, book: 'about 50' }, // i18n-ignore: no-import file loaded by e2e; Wrap.tsx translates
];

// ---------------------------------------------------------------- curves for the plots

/** `n` values from `lo` to `hi` inclusive, evenly spaced on a log scale. */
export function logSpace(lo: number, hi: number, n: number): number[] {
  const a = Math.log(lo);
  const b = Math.log(hi);
  return Array.from({ length: n }, (_, i) => (i === n - 1 ? hi : Math.exp(a + ((b - a) * i) / (n - 1))));
}

/** Overall speed-up against the speed-up of the accelerated part. */
export function speedupCurve(p: number, lo = 1, hi = 1000, n = 61): { s: number; overall: number }[] {
  return logSpace(lo, hi, n).map((s) => ({ s, overall: speedup(p, s) }));
}

/**
 * Kernel and overall speed-up against the number of threads. The two corner
 * thread counts are included so that a polyline through the points has its
 * corners in the right place.
 */
export function limitsCurve(
  p: number,
  ratePerThread: number,
  memoryCap: number,
  lo = 1,
  hi = 100000,
  n = 81,
): { threads: number; s: number; overall: number }[] {
  const c = corners(ratePerThread, memoryCap);
  const xs = [...logSpace(lo, hi, n), c.breakEven, c.saturation].filter((x) => x >= lo && x <= hi).sort((a, b) => a - b);
  return xs.map((threads) => {
    const k = limits(p, threads, ratePerThread, memoryCap);
    return { threads, s: k.s, overall: k.overall };
  });
}

// ======================================================== chapter 22: modules and overlap
// Section 22.3 (pp. 537–538) applies the law to an application made of several
// modules, each either moved to the GPU or left on the host, and adds the case
// chapter 1 leaves out: host and device working at the same time. The book's
// worked numbers: 95% made 100 times faster gives 1 / (5% + 0.95%) = 16.8 (the
// page prints "17x"); with the device's time hidden inside the host's, 1 / 5% = 20
// (notes/ch22.md; notes/today/ch22-facts.md, A9).
//
// What this part assumes, beyond the list at the top of the file:
//   - A host module keeps its original speed; a device module runs `s` times faster.
//   - No copies between host and device and no launch costs (tasklanes.ts models
//     launches).
//   - Overlap: a host module runs at the same time as the device unless it is
//     chained to the device, meaning it produces what the device needs or needs
//     what the device produces within the same step. Chained host time adds to the
//     total; the other host time and the device time run side by side and the
//     longer counts: total = chained host + max(other host, device). With nothing
//     chained that is the book's max(host, device). The chain is ours; the book's
//     example comes out the same either way as long as the device's time fits
//     inside the unchained host work.
//   - Shares are fractions of the original run time and add up to 1.

/**
 * Speed-up when the part left alone (1 - p) and the accelerated part (p / s) run
 * at the same time: the longer of the two sets the run time.
 */
export function overlapped(p: number, s: number): number {
  return 1 / Math.max(1 - p, p / s);
}

export type Place = 'host' | 'device';

export interface AppModule {
  /** Short name, as drawn. */
  name: string;
  /** Fraction of the original run time, 0 to 1. */
  share: number;
  place: Place;
  /** How many times faster the module runs on the device. Ignored on the host. */
  s: number;
  /** Chained to the device's work (see above). Matters only on the host, with overlap on. */
  chained?: boolean;
  /** Where the module has to run, when it cannot be moved. */
  fixed?: Place;
}

/** One module's place in time, as a fraction of the original run time. */
export interface Span {
  /** Index of the module in the list. */
  module: number;
  lane: Place;
  start: number;
  end: number;
}

export interface ModuleResult {
  /** Time on the host, as a fraction of the original run time. */
  host: number;
  /** The part of `host` that is chained to the device. */
  chained: number;
  /** Time on the device, as a fraction of the original run time. */
  device: number;
  /** Device time that runs while unchained host work runs (0 without overlap). */
  hidden: number;
  /** New run time as a fraction of the original. */
  total: number;
  /** 1 / total. */
  speedup: number;
  /** The most any device speed-up gives with this placement: 1 / host (Infinity with nothing on the host). */
  ceiling: number;
  /** Where each module sits in time, for drawing. */
  spans: Span[];
}

/** Time a module takes where it is placed, as a fraction of the original run time. */
export const moduleTime = (m: AppModule) => (m.place === 'device' ? m.share / m.s : m.share);

/**
 * The application's new run time and speed-up when each module runs where it is
 * placed, with or without host and device overlapping.
 */
export function moduleSpeedup(modules: AppModule[], overlap: boolean): ModuleResult {
  const onHost = (m: AppModule) => m.place === 'host';
  const host = modules.filter(onHost).reduce((a, m) => a + m.share, 0);
  const chained = overlap ? modules.filter((m) => onHost(m) && m.chained).reduce((a, m) => a + m.share, 0) : 0;
  const device = modules.filter((m) => !onHost(m)).reduce((a, m) => a + moduleTime(m), 0);
  const spans: Span[] = [];
  let total: number;
  let hidden = 0;
  if (!overlap) {
    // One after another, in list order.
    let t = 0;
    modules.forEach((m, i) => {
      const len = moduleTime(m);
      spans.push({ module: i, lane: m.place, start: t, end: t + len });
      t += len;
    });
    total = t;
  } else {
    // Chained host modules listed before the first device module run first, the
    // other chained ones after; in between, the device's modules and the unchained
    // host modules run side by side, each lane in list order.
    const firstDevice = modules.findIndex((m) => !onHost(m));
    const isBefore = (i: number) => firstDevice >= 0 && i < firstDevice;
    let t = 0;
    modules.forEach((m, i) => {
      if (onHost(m) && m.chained && isBefore(i)) {
        spans.push({ module: i, lane: 'host', start: t, end: t + m.share });
        t += m.share;
      }
    });
    const windowStart = t;
    let h = windowStart;
    let d = windowStart;
    modules.forEach((m, i) => {
      if (onHost(m) && m.chained) return;
      const len = moduleTime(m);
      if (onHost(m)) {
        spans.push({ module: i, lane: 'host', start: h, end: h + len });
        h += len;
      } else {
        spans.push({ module: i, lane: 'device', start: d, end: d + len });
        d += len;
      }
    });
    hidden = Math.min(h, d) - windowStart;
    t = Math.max(h, d);
    modules.forEach((m, i) => {
      if (onHost(m) && m.chained && !isBefore(i)) {
        spans.push({ module: i, lane: 'host', start: t, end: t + m.share });
        t += m.share;
      }
    });
    total = t;
    spans.sort((a, b) => a.module - b.module);
  }
  return { host, chained, device, hidden, total, speedup: 1 / total, ceiling: host > 0 ? 1 / host : Infinity, spans };
}

/**
 * Set module `i` to `share` and scale every other module so that the shares still
 * add up to 1 (equal parts if the others were all 0).
 */
export function rebalance(modules: AppModule[], i: number, share: number): AppModule[] {
  if (modules.length === 1) return [{ ...modules[0], share: 1 }];
  const v = Math.min(1, Math.max(0, share));
  const rest = modules.reduce((a, m, j) => (j === i ? a : a + m.share), 0);
  const n = modules.length - 1;
  return modules.map((m, j) =>
    j === i ? { ...m, share: v } : { ...m, share: rest > 0 ? (m.share / rest) * (1 - v) : (1 - v) / n },
  );
}

/**
 * The molecular-dynamics time step of Fig. 22.2 (p. 537) with the book's split:
 * non-bonded forces are 95% of the original time and 100 times faster on the GPU;
 * everything else stays on the host. How the other 5% divides between the three
 * host modules, and the speed-up those three would get on a GPU, are ours. The
 * chained flags follow the figure's arrows: the non-bonded kernel needs the
 * neighbour list, and the update needs both kinds of force before the next step.
 */
export const MD_STEP: AppModule[] = [
  { name: 'Neighbour list', share: 0.02, place: 'host', s: 4, chained: true }, // i18n-ignore: identifier also shown to the reader; S223 translates by name
  { name: 'Non-bonded forces', share: 0.95, place: 'device', s: 100 }, // i18n-ignore: identifier also shown to the reader; S223 translates by name
  { name: 'Vibrational and rotational forces', share: 0.02, place: 'host', s: 4 }, // i18n-ignore: identifier also shown to the reader; S223 translates by name
  { name: 'Update positions and velocities', share: 0.01, place: 'host', s: 4, chained: true }, // i18n-ignore: identifier also shown to the reader; S223 translates by name
];

/**
 * Ours, for illustration only: one decode step of a language model, with each
 * part's share of the step's time today and no timings. Speed-ups are relative to
 * today's version of each part; the sampler's 10 on a GPU is a placeholder. The
 * next step's kernels need the sampled token, so a host sampler is chained.
 */
export const DECODE_STEP: AppModule[] = [
  { name: 'Projections and attention', share: 0.78, place: 'device', s: 1, fixed: 'device' }, // i18n-ignore: identifier also shown to the reader; S223 translates by name
  { name: 'Elementwise kernels', share: 0.08, place: 'device', s: 1, fixed: 'device' }, // i18n-ignore: identifier also shown to the reader; S223 translates by name
  { name: 'Sampling', share: 0.06, place: 'host', s: 10, chained: true }, // i18n-ignore: identifier also shown to the reader; S223 translates by name
  { name: 'Kernel launches', share: 0.05, place: 'host', s: 1, fixed: 'host' }, // i18n-ignore: identifier also shown to the reader; S223 translates by name
  { name: 'Detokenise and stream out', share: 0.03, place: 'host', s: 1, fixed: 'host' }, // i18n-ignore: identifier also shown to the reader; S223 translates by name
];
