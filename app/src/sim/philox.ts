// Counter-based random numbers for module X1: the model behind section X1.7 ("Philox round
// by round", "Bits to u", the avalanche prediction), the uniforms of section X1.8, and every
// beat of the "Ten rounds of Philox" scene. Nothing here is in the book.
//
// Sources, all in notes/today/x1-facts.md:
//   - A4: Random123's Philox4x32. Constants 0xD2511F53 and 0xCD9E8D57 (multipliers),
//     0x9E3779B9 and 0xBB67AE85 (key bumps), ten rounds by default. One round:
//       lo0 = mulhilo(M0, ctr[0], &hi0); lo1 = mulhilo(M1, ctr[2], &hi1);
//       out = { hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0 }
//     and the key is bumped between rounds (not before the first). The three 10-round
//     known-answer vectors of tests/kat_vectors.
//   - A3: what curand_init(seed, subsequence, offset) does, as OUR DERIVATION from cuRAND's
//     header (described there in words; the header is not reproduced): key = seed (low,
//     high); the subsequence goes into counter words z, w; offset / 4 into x, y; the state
//     index is offset mod 4. Unverified on a device.
//   - D6 and E3: PyTorch's Metal Philox as we read c10/metal/random.h: nine rounds, key and
//     counter words in (high, low) order. Our transcription, never run on a device.
//   - A2, B2, E2: the uniform conversions of cuRAND (x * 2^-32 + 2^-33), rocRAND and
//     hipRAND on AMD (2^-32 + x * 2^-32), and the 24-bit one M5 uses ((x >> 8) * 2^-24).
//
// Arithmetic: a 32-bit word is a JS number in [0, 2^32). The 32 x 32 -> 64-bit product is
// formed exactly from 16-bit halves (the tests compare it with BigInt). Float results
// imitate float32 round-to-nearest-even with Math.fround, and only for the operations these
// conversions perform: an integer-to-float conversion, a multiplication by a power of two
// and one addition, each rounded once, which is what every backend does for them (x1-facts
// C4, E2; the multiplication is exact, so a fused multiply-add gives the same result).
//
// M5's mapping of (seed, stream, step) to counter and key is ours (section X1.7).
//
// Erasable TypeScript only, and no imports: the browser checks import this file directly.

// ---------------------------------------------------------------- constants (A4)
import { tr } from '../lib/locale.ts';

export const M0 = 0xd2511f53;
export const M1 = 0xcd9e8d57;
export const W0 = 0x9e3779b9;
export const W1 = 0xbb67ae85;
export const ROUNDS = 10;

export type Ctr = [number, number, number, number];
export type Key = [number, number];

const TWO32 = 4294967296;

/** The high and low 32 bits of the exact 64-bit product of two 32-bit words. */
export function mulhilo(a: number, b: number): [number, number] {
  const aL = a & 0xffff;
  const aH = a >>> 16;
  const bL = b & 0xffff;
  const bH = b >>> 16;
  const ll = aL * bL; // each partial product is below 2^32
  const mid = aL * bH + aH * bL; // below 2^33
  const hh = aH * bH;
  const low = ll + (mid % 65536) * 65536; // below 2^33, exact in a double
  const hi = (hh + Math.floor(mid / 65536) + Math.floor(low / TWO32)) >>> 0;
  return [hi, low >>> 0];
}

/** One Philox4x32 round (A4). */
export function round(ctr: Ctr, key: Key): Ctr {
  const [hi0, lo0] = mulhilo(M0, ctr[0]);
  const [hi1, lo1] = mulhilo(M1, ctr[2]);
  return [(hi1 ^ ctr[1] ^ key[0]) >>> 0, lo1, (hi0 ^ ctr[3] ^ key[1]) >>> 0, lo0];
}

/** The key schedule between rounds (A4): add the two Weyl constants, modulo 2^32. */
export function bumpKey(key: Key): Key {
  return [(key[0] + W0) >>> 0, (key[1] + W1) >>> 0];
}

/** Philox4x32-R: `rounds` rounds, the key bumped between them. */
export function philox(ctr: Ctr, key: Key, rounds = ROUNDS): Ctr {
  let c: Ctr = [ctr[0] >>> 0, ctr[1] >>> 0, ctr[2] >>> 0, ctr[3] >>> 0];
  let k: Key = [key[0] >>> 0, key[1] >>> 0];
  for (let r = 0; r < rounds; r++) {
    if (r > 0) k = bumpKey(k);
    c = round(c, k);
  }
  return c;
}

export interface Product {
  /** The counter word multiplied, and the constant it is multiplied by. */
  a: number;
  m: number;
  hi: number;
  lo: number;
}
export interface RoundTrace {
  /** 1-based. */
  round: number;
  ctrIn: Ctr;
  /** The key this round uses (already bumped for rounds 2 and later). */
  key: Key;
  /** M0 x ctr[0] and M1 x ctr[2]. */
  p0: Product;
  p1: Product;
  out: Ctr;
}

/** Every round, with its products split into halves, for the stepper and the scene. */
export function philoxTrace(ctr: Ctr, key: Key, rounds = ROUNDS): RoundTrace[] {
  const out: RoundTrace[] = [];
  let c: Ctr = [ctr[0] >>> 0, ctr[1] >>> 0, ctr[2] >>> 0, ctr[3] >>> 0];
  let k: Key = [key[0] >>> 0, key[1] >>> 0];
  for (let r = 0; r < rounds; r++) {
    if (r > 0) k = bumpKey(k);
    const [hi0, lo0] = mulhilo(M0, c[0]);
    const [hi1, lo1] = mulhilo(M1, c[2]);
    const next = round(c, k);
    out.push({ round: r + 1, ctrIn: c, key: k, p0: { a: c[0], m: M0, hi: hi0, lo: lo0 }, p1: { a: c[2], m: M1, hi: hi1, lo: lo1 }, out: next });
    c = next;
  }
  return out;
}

// ---------------------------------------------------------------- known answers (A4, E1)
export interface Kat {
  id: string;
  label: string;
  rounds: number;
  ctr: Ctr;
  key: Key;
  expected: Ctr;
}
/** The three 10-round philox4x32 lines of Random123's tests/kat_vectors (x1-facts A4). */
export const KATS: Kat[] = [
  {
    id: 'zero', get label() { return tr('all zero', 'toàn số 0'); }, rounds: 10,
    ctr: [0, 0, 0, 0], key: [0, 0],
    expected: [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8],
  },
  {
    id: 'ones', get label() { return tr('all ones', 'toàn bit 1'); }, rounds: 10,
    ctr: [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff], key: [0xffffffff, 0xffffffff],
    expected: [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd],
  },
  {
    id: 'pi', get label() { return tr('digits of pi', 'các chữ số của pi'); }, rounds: 10,
    ctr: [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344], key: [0xa4093822, 0x299f31d0],
    expected: [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1],
  },
];

export const hex = (w: number) => (w >>> 0).toString(16).padStart(8, '0');
export const sameWords = (a: readonly number[], b: readonly number[]) => a.length === b.length && a.every((v, i) => v >>> 0 === b[i] >>> 0);

// ---------------------------------------------------------------- 64-bit inputs
export const MAX64 = (1n << 64n) - 1n;
/** Low and high 32-bit words of a value below 2^64. */
export function split64(x: bigint): [number, number] {
  const v = BigInt.asUintN(64, x);
  return [Number(v & 0xffffffffn), Number(v >> 32n)];
}
/** A decimal or 0x-prefixed hexadecimal string as a 64-bit unsigned value, or null. */
export function parse64(s: string): bigint | null {
  const t = s.trim().replace(/[_\s,]/g, '');
  if (!/^(0x[0-9a-fA-F]+|\d+)$/.test(t)) return null;
  const v = BigInt(t);
  return v >= 0n && v <= MAX64 ? v : null;
}

// ---------------------------------------------------------------- M5's mapping (ours)
/**
 * M5: counter = (step low, step high, stream low, stream high), key = (seed low, seed high).
 * By our derivation from cuRAND's header (A3) this is the counter and key behind the first
 * curand() after curand_init(seed, stream, 4 * step): unverified on a device.
 */
export function m5Input(seed: bigint, stream: bigint, step: bigint): { ctr: Ctr; key: Key } {
  const [s0, s1] = split64(step);
  const [r0, r1] = split64(stream);
  const [k0, k1] = split64(seed);
  return { ctr: [s0, s1, r0, r1], key: [k0, k1] };
}
/** The four words for one (seed, stream, step). */
export function m5Words(seed: bigint, stream: bigint, step: bigint): Ctr {
  const { ctr, key } = m5Input(seed, stream, step);
  return philox(ctr, key);
}
/** M5's u for one row and step: word 0, top 24 bits, times 2^-24. */
export function m5U(seed: bigint, stream: bigint, step: bigint): number {
  return m5Uniform(m5Words(seed, stream, step)[0]);
}

/** Add a 64-bit amount to two counter words (low, high), with the carry. */
function add64(lo: number, hi: number, amount: bigint): [number, number] {
  const v = BigInt.asUintN(64, (BigInt(hi >>> 0) << 32n) + BigInt(lo >>> 0) + amount);
  return split64(v);
}
/** Increment a 128-bit counter by one (word x is the lowest). */
function inc128(c: Ctr): Ctr {
  const out: Ctr = [...c] as Ctr;
  for (let i = 0; i < 4; i++) {
    out[i] = (out[i] + 1) >>> 0;
    if (out[i] !== 0) break;
  }
  return out;
}

/**
 * The first `count` values curand() would return after curand_init(seed, subsequence,
 * offset), by OUR DERIVATION from cuRAND's header (x1-facts A3, described, not copied): key
 * from the seed, the subsequence added to counter words z and w, offset / 4 added to x and
 * y, the state index offset mod 4, and the counter incremented after the fourth word.
 */
export function curandAfterInit(seed: bigint, subsequence: bigint, offset: bigint, count = 1): number[] {
  const [k0, k1] = split64(seed);
  const key: Key = [k0, k1];
  let ctr: Ctr = [0, 0, 0, 0];
  const [z, w] = add64(0, 0, BigInt.asUintN(64, subsequence));
  ctr[2] = z;
  ctr[3] = w;
  const [x, y] = add64(0, 0, BigInt.asUintN(64, offset) / 4n);
  ctr[0] = x;
  ctr[1] = y;
  let state = Number(BigInt.asUintN(64, offset) % 4n);
  let words = philox(ctr, key);
  const out: number[] = [];
  for (let i = 0; i < count; i++) {
    out.push(words[state]);
    state++;
    if (state === 4) {
      state = 0;
      ctr = inc128(ctr);
      words = philox(ctr, key);
    }
  }
  return out;
}

// ---------------------------------------------------------------- PyTorch MPS, as we read it (D6, E3)
/** Rounds in PyTorch's Metal Philox by our reading of c10/metal/random.h (D6): nine. */
export const MPS_ROUNDS = 9;
/**
 * PyTorch's Metal philox4::rand(seed, index) as we transcribe it (D6, E3): key words
 * (seed high, seed low), the index in counter words z, w (high, low), nine rounds. Our
 * reading, not run on a device.
 */
export function mpsInput(seed: bigint, index: bigint): { ctr: Ctr; key: Key } {
  const [s0, s1] = split64(seed);
  const [i0, i1] = split64(index);
  return { ctr: [0, 0, i1, i0], key: [s1, s0] };
}
export function mpsPhilox(seed: bigint, index: bigint): Ctr {
  const { ctr, key } = mpsInput(seed, index);
  return philox(ctr, key, MPS_ROUNDS);
}
/** E3's table: PyTorch Metal (transcribed) against Philox4x32-10 in the A3 layout. */
export const E3 = [
  { seed: 0n, index: 0n, mps: [0x7028b1c8, 0xd6594c40, 0x4f2051bb, 0x76d6628e] as Ctr, ten: [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8] as Ctr },
  { seed: 42n, index: 0n, mps: [0x9a822623, 0xfd1355ba, 0x3f3f7f79, 0x4bd4b252] as Ctr, ten: [0x9ceaf053, 0x77f5493b, 0x12bf50ad, 0x5742b3d7] as Ctr },
];

// ---------------------------------------------------------------- avalanche
export interface Avalanche {
  /** differing[r - 1]: how many of the four words differ after round r. */
  differing: number[];
  /** Which words differ after each round. */
  masks: boolean[][];
  /** The first round after which all four words differ, or null. */
  allBy: number | null;
}
/** Run two counters with one key side by side and count the words that differ after each round. */
export function avalanche(a: Ctr, b: Ctr, key: Key, rounds = ROUNDS): Avalanche {
  const ta = philoxTrace(a, key, rounds);
  const tb = philoxTrace(b, key, rounds);
  const masks = ta.map((r, i) => r.out.map((w, j) => w !== tb[i].out[j]));
  const differing = masks.map((m) => m.filter(Boolean).length);
  const k = differing.findIndex((d) => d === 4);
  return { differing, masks, allBy: k < 0 ? null : k + 1 };
}
/** Rounds until all four words differ for streams s and s + 1 at one seed and step (M5's layout). */
export function roundsUntilAllWordsDiffer(seed: bigint, stream: bigint, step: bigint): number | null {
  const A = m5Input(seed, stream, step);
  const B = m5Input(seed, stream + 1n, step);
  return avalanche(A.ctr, B.ctr, A.key).allBy;
}

// ---------------------------------------------------------------- bits to u (A2, B2, E2)
const P32 = 2 ** -32;
const P33 = 2 ** -33;
const P24 = 2 ** -24;
const f32 = Math.fround;

/** cuRAND's curand_uniform: float(x) * 2^-32 + 2^-33 in single precision, range (0, 1] (A2). */
export function curandUniform(x: number): number {
  return f32(f32(x >>> 0) * P32 + P33);
}
/** rocRAND's uniform, which hiprand_uniform uses on AMD: 2^-32 + float(x) * 2^-32, range (0, 1] (B2). */
export function hipUniformAMD(x: number): number {
  return f32(P32 + f32(x >>> 0) * P32);
}
/** M5's conversion: the top 24 bits times 2^-24, exact on every backend, range [0, 1 - 2^-24] (E2). */
export function m5Uniform(x: number): number {
  return (x >>> 8) * P24;
}
export type Conversion = 'curand' | 'hip' | 'm5';
export const CONVERT: Record<Conversion, (x: number) => number> = { curand: curandUniform, hip: hipUniformAMD, m5: m5Uniform };
export const CONVERSION_LABEL: Record<Conversion, string> = {
  get curand() {
    return tr('curand_uniform (NVIDIA)', 'curand_uniform (NVIDIA)');
  },
  get hip() {
    return tr('hiprand_uniform on AMD (rocRAND)', 'hiprand_uniform trên AMD (rocRAND)');
  },
  m5: 'M5: (x >> 8) × 2⁻²⁴',
};

const view = new DataView(new ArrayBuffer(4));
/** The 32 bits of a float32. */
export function f32bits(v: number): number {
  view.setFloat32(0, v);
  return view.getUint32(0);
}
/** The shortest decimal that reads back as the same float32. */
export function f32short(v: number): string {
  const x = f32(v);
  for (let p = 1; p <= 9; p++) {
    const s = x.toPrecision(p);
    if (f32(Number(s)) === x) return Number(s).toString();
  }
  return x.toPrecision(9);
}

/**
 * How many of the 2^32 inputs satisfy pred, by binade analysis. Inside one binade of x
 * [2^k, 2^(k+1)), every conversion above is the same function of x shifted by any multiple
 * of P = 8 float32 steps of that binade (rounding to even depends on whole multiples of two
 * steps, and the results stay in one binade away from the top edge). So the count is one
 * period's count times the number of periods, plus the top edge evaluated input by input.
 * Binades below 2^21 are evaluated input by input. About two million evaluations in all.
 */
export function countInputs(pred: (x: number) => boolean): number {
  let total = pred(0) ? 1 : 0;
  for (let k = 0; k < 32; k++) {
    const lo = 2 ** k;
    const hi = 2 ** (k + 1);
    if (k <= 20) {
      for (let x = lo; x < hi; x++) if (pred(x)) total++;
      continue;
    }
    const step = Math.max(1, 2 ** (k - 23)); // float32 spacing of integers in this binade
    const P = 8 * step;
    let inPeriod = 0;
    for (let x = lo; x < lo + P; x++) if (pred(x)) inPeriod++;
    let edge = 0;
    for (let x = hi - P; x < hi; x++) if (pred(x)) edge++;
    total += inPeriod * ((hi - lo - P) / P) + edge;
  }
  return total;
}

export interface BinadeRow {
  /** x in [2^k, 2^(k+1)); k = -1 stands for x = 0. */
  k: number;
  size: number;
  /** Inputs where curand_uniform and AMD's hiprand_uniform give different floats. */
  differ: number;
  /** Inputs that give exactly 1.0. */
  onesCurand: number;
  onesHip: number;
}
let profileCache: BinadeRow[] | null = null;
/** The binade analysis, binade by binade (cached: about two million evaluations). */
export function binadeProfile(): BinadeRow[] {
  if (profileCache) return profileCache;
  const rows: BinadeRow[] = [];
  const at = (x: number) => ({
    differ: curandUniform(x) !== hipUniformAMD(x) ? 1 : 0,
    oc: curandUniform(x) === 1 ? 1 : 0,
    oh: hipUniformAMD(x) === 1 ? 1 : 0,
  });
  const z = at(0);
  rows.push({ k: -1, size: 1, differ: z.differ, onesCurand: z.oc, onesHip: z.oh });
  for (let k = 0; k < 32; k++) {
    const lo = 2 ** k;
    const hi = 2 ** (k + 1);
    const row: BinadeRow = { k, size: hi - lo, differ: 0, onesCurand: 0, onesHip: 0 };
    const addRange = (a: number, b: number, times: number) => {
      for (let x = a; x < b; x++) {
        const v = at(x);
        row.differ += v.differ * times;
        row.onesCurand += v.oc * times;
        row.onesHip += v.oh * times;
      }
    };
    if (k <= 20) addRange(lo, hi, 1);
    else {
      const P = 8 * Math.max(1, 2 ** (k - 23));
      addRange(lo, lo + P, (hi - lo - P) / P);
      addRange(hi - P, hi, 1);
    }
    rows.push(row);
  }
  profileCache = rows;
  return rows;
}
/** Inputs where the two vendor conversions disagree: 2^24, one in 256 (E2, our arithmetic). */
export function differingInputs(): number {
  return binadeProfile().reduce((s, r) => s + r.differ, 0);
}
/** Inputs that a vendor conversion turns into exactly 1.0: 128 each (A2, E2). */
export function onesCount(kind: 'curand' | 'hip'): number {
  return binadeProfile().reduce((s, r) => s + (kind === 'curand' ? r.onesCurand : r.onesHip), 0);
}

/** The regions of the strip, as the analysis finds them (each is checked in the tests). */
export const DIFFER_REGIONS = [
  { from: 0, to: 2 ** 23, get which() { return tr('every x', 'mọi x'); }, count: 2 ** 23 },
  { from: 2 ** 23, to: 2 ** 24, get which() { return tr('even x', 'x chẵn'); }, count: 2 ** 22 },
  { from: 2 ** 24, to: 2 ** 25, get which() { return tr('x whose float32 value is 2 more than a multiple of 4', 'x có giá trị float32 bằng một bội của 4 cộng thêm 2'); }, count: 2 ** 22 },
];
/** The smallest input both vendor conversions turn into 1.0 (A2). */
export const FIRST_ONE = 0xffffff80;

export interface Sample {
  n: number;
  differ: number;
  onesCurand: number;
  onesHip: number;
  m5AtOne: number;
}
/** A seeded sample of n inputs drawn with Philox itself (counter i, key (seed, 0)). */
export function sampleConversions(n: number, seed = 1): Sample {
  const s: Sample = { n, differ: 0, onesCurand: 0, onesHip: 0, m5AtOne: 0 };
  let words: Ctr = [0, 0, 0, 0];
  for (let i = 0; i < n; i++) {
    if (i % 4 === 0) words = philox([i / 4, 0, 0, 0], [seed >>> 0, 0]);
    const x = words[i % 4];
    const a = curandUniform(x);
    const b = hipUniformAMD(x);
    if (a !== b) s.differ++;
    if (a === 1) s.onesCurand++;
    if (b === 1) s.onesHip++;
    if (m5Uniform(x) >= 1) s.m5AtOne++;
  }
  return s;
}

/** The inputs the "Bits to u" explorer offers as jumps. */
export const JUMPS: { label: string; x: number }[] = [
  { label: '0', x: 0 },
  { label: '255', x: 255 },
  { label: '2²³', x: 2 ** 23 },
  { label: '2²⁴ + 1', x: 2 ** 24 + 1 },
  { label: '2²⁵', x: 2 ** 25 },
  { label: '0xFFFFFF80', x: 0xffffff80 },
];
