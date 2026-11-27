// One warp's vote and aggregated append (section 12.3): which lanes take part, the mask,
// the leader, the count, the one atomic, the broadcast, each lane's rank and slot, and
// what goes wrong in the broken variants. Section 12.3's warp vote visualiser, its
// questions, and beats 4 to 6 of the scene "One vote, one atomic" run on it.
//
// The forms, and where each rule comes from:
//   book       Fig. 12.3 as printed (pp. 292-293): __activemask() inside the branch,
//              leader = __ffs(mask) - 1, count = __popc(mask), the leader's fetch_add,
//              __shfl_sync(mask, j, leader), rank = __popc(mask & ((1 << lane) - 1)).
//              32-bit masks. On NVIDIA the mask is "an instantaneous snapshot" (ch12-facts
//              A1): the passing lanes may arrive as several converged snapshots, each of
//              which aggregates on its own. Which lanes converge is the reader's choice
//              here, not a model of the hardware.
//   ballot     lab 12.1's documented form (A1's advice; llama.cpp and PyTorch, D1 and D4):
//              every lane votes in uniform control flow with __ballot_sync, a warp with no
//              kept key skips, a typed mask (32 bits for CUDA, 64 for HIP; infra B3).
//   metal      lab 12.1's Metal form (C1): simd_prefix_exclusive_sum(keep), simd_sum(keep),
//              simd_is_first(), simd_broadcast_first(), in uniform control flow, so the
//              first active thread is lane 0 whether or not it keeps, and it adds 0 when
//              nobody keeps unless the code checks the count.
//   hip-naive  Fig. 12.3 with only the shuffle mask widened to 64 bits, on a 64-wide
//              wavefront (B1): __activemask() is "the exact active mask" on AMD, but it is
//              stored in an unsigned int, and __ffs, __popc and 1 << lane are 32-bit, so
//              lanes 32 to 63 vanish from the count, their ranks shift past the width of
//              an int (undefined), and a warp whose passing lanes are all >= 32 gets
//              leader -1.
//   bug-ffs    the book's form without the - 1 after __ffs: lane (lowest + 1) leads.
//   bug-below  the book's form with a lanes-below mask that includes the lane itself,
//              (2u << lane) - 1, so every rank is one too high.
//
// Documented rules used for the failures: __ffs(0) is 0 and the result is 1-based (A2); a
// shuffle that reads a lane outside its mask returns an undefined value, and CUDA's
// __shfl_sync reads lane srcLane % width (infra A3, A4); a 32-bit shift by 32 or more is
// undefined (A3). An undefined value is null here and poisons whatever uses it.
//
// Nothing here models time, divergence cost or the order in which snapshots are served
// beyond "in the order given".
//
// Erasable TypeScript only, and no imports: the browser checks import this file directly.
import { tr } from '../lib/locale.ts';

export type Width = 8 | 32 | 64;
export const WIDTHS: readonly Width[] = [8, 32, 64];

export type Form = 'book' | 'ballot' | 'metal' | 'hip-naive' | 'bug-ffs' | 'bug-below';
export const FORMS: readonly Form[] = ['book', 'ballot', 'metal', 'hip-naive', 'bug-ffs', 'bug-below'];
export const FORM_LABEL: Record<Form, string> = {
  get book() { return tr('Fig. 12.3 as printed', 'Fig. 12.3 như in trong sách'); },
  get ballot() { return tr('ballot in uniform control flow (lab 12.1)', 'ballot trong dòng điều khiển đồng nhất (lab 12.1)'); },
  get metal() { return tr('Metal SIMD-group functions (lab 12.1)', 'các hàm SIMD group của Metal (lab 12.1)'); },
  get 'hip-naive'() { return tr('naive HIP port: only the shuffle mask widened', 'bản port HIP ngây thơ: chỉ mở rộng mask của shuffle'); },
  get 'bug-ffs'() { return tr('planted bug: no - 1 after __ffs', 'lỗi cài sẵn: thiếu - 1 sau __ffs'); },
  get 'bug-below'() { return tr('planted bug: lanes-below mask includes the lane', 'lỗi cài sẵn: mask các lane phía dưới bao gồm cả chính lane đó'); },
};
/** Forms that run Fig. 12.3's code inside the branch: only passing lanes take part. */
export const IN_BRANCH: Record<Form, boolean> = {
  book: true, ballot: false, metal: false, 'hip-naive': true, 'bug-ffs': true, 'bug-below': true,
};

export interface VoteInput {
  width: Width;
  /** Lanes whose key is in range and passes the test; length `width`. */
  pass: readonly boolean[];
  /** The counter's value when the warp's atomic arrives. */
  counter: number;
  form: Form;
  /** Book family on NVIDIA: the passing lanes reach __activemask() as these groups, in this order. */
  snapshots?: readonly (readonly number[])[];
  /** Metal: skip the atomic when nobody keeps. */
  guard?: boolean;
}

export interface Atomic {
  /** The lane that performs it. */
  lane: number;
  add: number;
  /** What fetch_add returns: the counter before. */
  before: number;
  after: number;
}

export interface Group {
  /** The lanes that compute this mask together: one snapshot, or the whole warp. */
  lanes: number[];
  /** The mask as the code holds it (book family: the active mask; ballot: the vote; Metal: the vote, for reference). */
  mask: bigint;
  /** Bits in the variable that holds the mask. */
  maskBits: 32 | 64;
  /** __ffs(mask) as computed: 1-based, 0 for no bit. Null for Metal, which has no ffs step. */
  ffs: number | null;
  /** The leader as computed; -1 or a lane number the warp does not have are possible. */
  leader: number;
  /** The leader is one of `lanes`, so the atomic happens. */
  leaderRuns: boolean;
  /** __popc(mask) or simd_sum as computed. */
  count: number;
  atomic: Atomic | null;
  /** What the broadcast gives the lanes; null = undefined. */
  start: number | null;
  /** The broadcast reads a lane that is not in the mask (undefined). */
  shuffleOutside: boolean;
  /** Metal only: ctz of the vote, the lowest keeping lane, 64 when none (C2). */
  ctz: number | null;
  /** The ballot form's warp-uniform "nobody keeps" branch was taken. */
  skipped: boolean;
}

export interface Lane {
  lane: number;
  pass: boolean;
  /** Runs the vote code: inside the branch for the book family, every lane for ballot and Metal. */
  votes: boolean;
  group: number | null;
  /** The lanes-below mask as computed; null = undefined (a shift past the width of its type). */
  below: bigint | null;
  and: bigint | null;
  rank: number | null;
  /** Writes its key (it passes and reaches the store). */
  writes: boolean;
  /** Where it writes; null when it writes nothing or at an undefined address. */
  slot: number | null;
  undefinedSlot: boolean;
}

export type Flag =
  | 'compile-error'
  | 'lane-dropped'
  | 'leader-invalid'
  | 'shuffle-outside'
  | 'slot-undefined'
  | 'overlap'
  | 'out-of-range'
  | 'gap'
  | 'zero-add';
/** Flags that mean the output is wrong. 'zero-add' only wastes an atomic. */
export const FAILURES: readonly Flag[] = ['compile-error', 'lane-dropped', 'leader-invalid', 'shuffle-outside', 'slot-undefined', 'overlap', 'out-of-range', 'gap'];
export const FLAG_TEXT: Record<Flag, string> = {
  get 'compile-error'() { return tr('does not compile', 'không biên dịch được'); },
  get 'lane-dropped'() { return tr('a passing lane is missing from the count', 'một lane đạt điều kiện bị thiếu trong số đếm'); },
  get 'leader-invalid'() { return tr('the leader is not a lane that runs this code, so nobody does the atomic', 'leader không phải một lane chạy đoạn mã này, nên không ai thực hiện phép atomic'); },
  get 'shuffle-outside'() { return tr('the broadcast reads a lane outside the mask: undefined', 'phép broadcast đọc một lane nằm ngoài mask: không xác định'); },
  get 'slot-undefined'() { return tr('a lane writes at an undefined slot', 'một lane ghi vào một vị trí không xác định'); },
  get overlap() { return tr('two lanes write the same slot', 'hai lane ghi cùng một vị trí'); },
  get 'out-of-range'() { return tr('a lane writes outside the reserved range', 'một lane ghi ra ngoài vùng đã đặt trước'); },
  get gap() { return tr('a reserved slot is never written', 'một vị trí đã đặt trước không bao giờ được ghi'); },
  get 'zero-add'() { return tr('an atomic adds 0', 'một phép atomic cộng 0'); },
};

export interface VoteResult {
  input: VoteInput;
  passing: number[];
  groups: Group[];
  lanes: Lane[];
  counterBefore: number;
  counterAfter: number;
  atomics: number;
  /** What a correct append fills: counter .. counter + passing - 1. */
  expected: number[];
  flags: Flag[];
  correct: boolean;
  compileError: string | null;
}

// ---------------------------------------------------------------- bit helpers
const ONE = BigInt(1);
const ZERO = BigInt(0);
const bit = (lane: number) => ONE << BigInt(lane);
export const maskOf = (lanes: readonly number[]): bigint => lanes.reduce((m, l) => m | bit(l), ZERO);
export const lanesOf = (mask: bigint, width: number): number[] => Array.from({ length: width }, (_, l) => l).filter((l) => (mask >> BigInt(l)) & ONE);
export function popc(mask: bigint): number {
  let n = 0;
  for (let m = mask; m > ZERO; m >>= ONE) n += Number(m & ONE);
  return n;
}
/** __ffs / __ffsll: position of the lowest set bit counting from 1, 0 for none. */
export function ffs(mask: bigint): number {
  if (mask === ZERO) return 0;
  let p = 1;
  for (let m = mask; (m & ONE) === ZERO; m >>= ONE) p++;
  return p;
}
/** ctz on a 64-bit vote (MSL): trailing zeros, 64 for none. */
export const ctz64 = (mask: bigint): number => (mask === ZERO ? 64 : ffs(mask) - 1);
const low32 = (m: bigint) => m & ((ONE << BigInt(32)) - ONE);

/** A mask in binary, highest lane on the left, in groups of four. */
export function binary(mask: bigint, bits: number): string {
  const s = mask.toString(2).padStart(bits, '0').slice(-bits);
  return s.replace(/(.{4})(?=.)/g, '$1 ');
}
/** A mask in hex, as many digits as the variable has. */
export const hex = (mask: bigint, bits: number): string => '0x' + mask.toString(16).padStart(Math.ceil(bits / 4), '0');

// ---------------------------------------------------------------- snapshots
/** The passing lanes as one snapshot: what a fully converged warp reports. */
export const oneSnapshot = (pass: readonly boolean[]): number[][] => {
  const p = pass.flatMap((x, l) => (x ? [l] : []));
  return p.length ? [p] : [];
};
/** The passing lanes split in two snapshots: the lower half (rounded up) first, then the rest. */
export function halves(pass: readonly boolean[]): number[][] {
  const p = pass.flatMap((x, l) => (x ? [l] : []));
  if (p.length < 2) return p.length ? [p] : [];
  const k = Math.ceil(p.length / 2);
  return [p.slice(0, k), p.slice(k)];
}

// ---------------------------------------------------------------- the model
export function vote(input: VoteInput): VoteResult {
  const { width, pass, counter, form } = input;
  if (pass.length !== width) throw new Error(`pass has ${pass.length} lanes, the warp ${width}`);
  const passing = pass.flatMap((x, l) => (x ? [l] : []));
  const expected = passing.map((_, k) => counter + k);
  const lanes: Lane[] = pass.map((p, lane) => ({
    lane, pass: p, votes: !IN_BRANCH[form] || p, group: null, below: null, and: null, rank: null, writes: false, slot: null, undefinedSlot: false,
  }));
  const groups: Group[] = [];
  let c = counter;

  const bookFamily = form === 'book' || form === 'bug-ffs' || form === 'bug-below';
  if (bookFamily && width === 64) {
    return {
      input, passing, groups, lanes, counterBefore: counter, counterAfter: counter, atomics: 0, expected,
      flags: ['compile-error'], correct: false,
      compileError:
        tr('A 64-wide wavefront means HIP, and HIP rejects the 32-bit mask in __shfl_sync: "The mask must be a 64-bit integer" (static_assert, ch12-facts B1). Try the naive HIP port.', 'Wavefront rộng 64 nghĩa là HIP, và HIP từ chối mask 32 bit trong __shfl_sync: "The mask must be a 64-bit integer" (static_assert, ch12-facts B1). Hãy thử bản port HIP ngây thơ.'),
    };
  }

  if (bookFamily || form === 'hip-naive') {
    // On AMD the active mask is exact (B1): one snapshot whatever the reader asked for.
    const snaps = form === 'hip-naive' ? oneSnapshot(pass) : (input.snapshots?.map((s) => [...s]) ?? oneSnapshot(pass));
    const seen = new Set<number>();
    for (const s of snaps) for (const l of s) {
      if (!pass[l] || seen.has(l)) throw new Error('snapshots must split the passing lanes');
      seen.add(l);
    }
    if (seen.size !== passing.length) throw new Error('snapshots must cover every passing lane');
    const naive = form === 'hip-naive';
    snaps.forEach((s, g) => {
      const full = maskOf(s);
      // hip-naive: unsigned int activeThreads = __activemask(); keeps the low 32 bits only.
      const mask = naive ? low32(full) : full;
      const f = ffs(mask);
      const leader = form === 'bug-ffs' ? f : f - 1;
      const leaderRuns = s.includes(leader);
      const count = popc(mask);
      let atomic: Atomic | null = null;
      if (leaderRuns) {
        atomic = { lane: leader, add: count, before: c, after: c + count };
        c += count;
      }
      // __shfl_sync(mask, j, leader): legal only if the source lane is in the mask (A4).
      // CUDA reads srcLane % width; a leader of -1 (0xFFFFFFFF as unsigned) has no lane in HIP's docs.
      const src = leader >= 0 ? leader % width : null;
      const shuffleOutside = src == null || !s.includes(src);
      const start = atomic && !shuffleOutside ? atomic.before : null;
      groups.push({ lanes: [...s], mask, maskBits: 32, ffs: f, leader, leaderRuns, count, atomic, start, shuffleOutside, ctz: null, skipped: false });
      for (const l of s) {
        const L = lanes[l];
        L.group = g;
        L.writes = true;
        if (l >= 32) {
          // (1 << laneIdx()) with lane >= 32 shifts a 32-bit int past its width: undefined (A3).
          L.below = null;
          L.and = null;
          L.rank = null;
        } else {
          const below = form === 'bug-below' ? low32((bit(l) << ONE) - ONE) : bit(l) - ONE;
          L.below = below;
          L.and = mask & below;
          L.rank = popc(L.and);
        }
        L.slot = start != null && L.rank != null ? start + L.rank : null;
        L.undefinedSlot = L.slot == null;
      }
    });
  } else if (form === 'ballot') {
    const bits: 32 | 64 = width === 64 ? 64 : 32;
    const mask = maskOf(passing);
    const all = lanes.map((L) => L.lane);
    const skipped = mask === ZERO;
    const f = ffs(mask);
    const leader = f - 1;
    const count = popc(mask);
    let atomic: Atomic | null = null;
    if (!skipped) {
      atomic = { lane: leader, add: count, before: c, after: c + count };
      c += count;
    }
    groups.push({
      lanes: all, mask, maskBits: bits, ffs: f, leader, leaderRuns: !skipped, count, atomic,
      start: atomic ? atomic.before : null, shuffleOutside: false, ctz: null, skipped,
    });
    for (const L of lanes) {
      L.group = 0;
      L.below = bit(L.lane) - ONE;
      L.and = mask & L.below;
      L.rank = popc(L.and);
      L.writes = L.pass;
      L.slot = L.pass && atomic ? atomic.before + L.rank : null;
    }
  } else {
    // Metal, uniform control flow: every lane is active, so the first active thread is lane 0.
    const mask = maskOf(passing);
    const count = passing.length;
    const leader = 0;
    let atomic: Atomic | null = null;
    if (!(input.guard && count === 0)) {
      atomic = { lane: leader, add: count, before: c, after: c + count };
      c += count;
    }
    // uint base = 0; if (simd_is_first()) base = atomic...; base = simd_broadcast_first(base);
    const start = atomic ? atomic.before : 0;
    groups.push({
      lanes: lanes.map((L) => L.lane), mask, maskBits: 64, ffs: null, leader, leaderRuns: true, count, atomic,
      start, shuffleOutside: false, ctz: ctz64(mask), skipped: !atomic,
    });
    for (const L of lanes) {
      L.group = 0;
      // simd_prefix_exclusive_sum(keep ? 1 : 0): the keeping lanes with a lower index.
      L.below = bit(L.lane) - ONE;
      L.and = mask & L.below;
      L.rank = popc(L.and);
      L.writes = L.pass;
      L.slot = L.pass ? start + L.rank : null;
    }
  }

  // ---- what went wrong
  const flags = new Set<Flag>();
  const counted = new Set<number>();
  for (const g of groups) {
    if (g.lanes.some((l) => pass[l]) && !g.leaderRuns) flags.add('leader-invalid');
    if (g.shuffleOutside && g.lanes.length) flags.add('shuffle-outside');
    if (g.atomic && g.atomic.add === 0) flags.add('zero-add');
    for (const l of lanesOf(g.mask, width)) counted.add(l);
  }
  if (form !== 'metal' && passing.some((l) => !counted.has(l))) flags.add('lane-dropped');
  const writers = lanes.filter((L) => L.writes);
  if (writers.some((L) => L.undefinedSlot)) flags.add('slot-undefined');
  const slots = writers.flatMap((L) => (L.slot == null ? [] : [L.slot]));
  if (new Set(slots).size !== slots.length) flags.add('overlap');
  if (slots.some((s) => s < counter || s >= c)) flags.add('out-of-range');
  // Each lane must also stay inside the range its own group reserved.
  for (const L of writers) {
    const a = L.group == null ? null : groups[L.group].atomic;
    if (a && L.slot != null && (L.slot < a.before || L.slot >= a.after)) flags.add('out-of-range');
  }
  const filled = new Set(slots);
  for (let s = counter; s < c; s++) if (!filled.has(s)) flags.add('gap');
  const list = [...flags];
  const correct =
    !list.some((f) => FAILURES.includes(f)) &&
    c - counter === passing.length &&
    slots.length === passing.length &&
    [...slots].sort((a, b) => a - b).every((s, i) => s === expected[i]);
  return {
    input, passing, groups, lanes, counterBefore: counter, counterAfter: c,
    atomics: groups.filter((g) => g.atomic).length, expected, flags: list, correct, compileError: null,
  };
}

/** Pass flags for a warp of `width` lanes with the given lanes passing. */
export function passFrom(width: number, lanes: readonly number[]): boolean[] {
  const p = Array.from({ length: width }, () => false);
  for (const l of lanes) {
    if (!Number.isInteger(l) || l < 0 || l >= width) throw new Error(`lane ${l} is not in a warp of ${width}`);
    p[l] = true;
  }
  return p;
}

// ---------------------------------------------------------------- presets
export type Preset = 'book' | 'none' | 'all' | 'lane31' | 'upper' | 'alternate';
export const PRESET_LABEL: Record<Preset, string> = {
  get book() { return tr("the book's lanes 1, 2, 4, 5", 'các lane 1, 2, 4, 5 của sách'); },
  get none() { return tr('none', 'không lane nào'); },
  get all() { return tr('all', 'tất cả'); },
  get lane31() { return tr('only lane 31', 'chỉ lane 31'); },
  get upper() { return tr('only lanes 32 to 63', 'chỉ các lane 32 đến 63'); },
  get alternate() { return tr('alternating', 'xen kẽ'); },
};
/** Presets that make sense at this width. */
export const presetsFor = (width: Width): Preset[] =>
  (['book', 'none', 'all', 'lane31', 'upper', 'alternate'] as Preset[]).filter((p) => (p === 'lane31' ? width >= 32 : p === 'upper' ? width === 64 : true));
export function preset(p: Preset, width: Width): boolean[] {
  const r = Array.from({ length: width }, (_, l) => l);
  switch (p) {
    case 'book':
      return passFrom(width, [1, 2, 4, 5]);
    case 'none':
      return passFrom(width, []);
    case 'all':
      return passFrom(width, r);
    case 'lane31':
      return passFrom(width, [31]);
    case 'upper':
      return passFrom(width, r.filter((l) => l >= 32));
    case 'alternate':
      return passFrom(width, r.filter((l) => l % 2 === 0));
  }
}

// ---------------------------------------------------------------- Fig. 12.3 against Fig. 12.4
/**
 * The book's own mapping between the two listings (p. 294), by line number in each figure
 * as printed. labs/ch12/snippets/s123_fig12_3.cu and s123_fig12_4_cg.cu keep the book's
 * line numbers; app/src/chapters/ch12/S123.snippets.test.ts pins every line named here.
 */
export interface LinePair {
  id: string;
  /** Lines of Fig. 12.3. */
  fig3: number[];
  /** The text that identifies them (contained in the first line). */
  fig3Text: string;
  /** The matching line of Fig. 12.4. */
  fig4: number;
  fig4Text: string;
  /** What the pair does, for the feedback. */
  step: string;
}
export const LINE_PAIRS: readonly LinePair[] = [
  { id: 'mask', fig3: [7], fig3Text: '__activemask()', fig4: 7, fig4Text: 'coalesced_threads()', get step() { return tr('Which lanes are here: a 32-bit mask, or a group handle that holds it.', 'Những lane nào đang có mặt: một mask 32 bit, hoặc một handle của nhóm chứa mask đó.'); } },
  { id: 'leader', fig3: [10, 11], fig3Text: '__ffs(activeThreads) - 1', fig4: 10, fig4Text: 'thread_rank() == 0', get step() { return tr('The leader: the lowest active lane, which is the member of rank 0.', 'Leader: lane đang hoạt động thấp nhất, cũng là thành viên có rank 0.'); } }, // i18n-ignore: fig3Text is code
  { id: 'count', fig3: [13], fig3Text: '__popc(activeThreads)', fig4: 12, fig4Text: 'num_threads()', get step() { return tr('How many lanes are here: a population count, or the size of the group (the book writes size(), now a legacy alias).', 'Có bao nhiêu lane đang có mặt: một phép đếm bit (population count), hoặc kích thước của nhóm (sách viết size(), nay là một bí danh cũ).'); } },
  { id: 'broadcast', fig3: [20], fig3Text: '__shfl_sync(activeThreads, j, leader)', fig4: 19, fig4Text: 'shfl(j, 0)', get step() { return tr('The leader hands j to the others: by lane number with a mask, or by rank within the group.', 'Leader chuyển j cho các lane khác: theo số lane cùng một mask, hoặc theo rank trong nhóm.'); } },
  { id: 'rank', fig3: [22, 23, 24], fig3Text: '(1 << laneIdx()) - 1', fig4: 21, fig4Text: 'thread_rank()', get step() { return tr('A lane\'s offset: the active lanes below it, counted from the mask, or simply its rank in the group.', 'Độ lệch của một lane: số lane đang hoạt động phía dưới nó, đếm từ mask, hoặc đơn giản là rank của nó trong nhóm.'); } },
];
