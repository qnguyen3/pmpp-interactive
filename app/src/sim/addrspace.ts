// Section C.1: one pointer type for every memory on the GPU.
//
// Part 1, windows. Since Fermi, CUDA's loads and stores work on one generic
// address space (book p. 624). PTX describes the state spaces .local and .shared
// as "windows within the generic address space"; "a generic address maps to
// global memory unless it falls within the window for const, local, or shared
// memory" (appC-facts A2). AMD's compiler documents the same shape, a "flat"
// address space with apertures for scratch and LDS (B1). The toy below has two
// windows on a 16-bit strip; everything else is global.
//
// Simplifications (the page says so too): window positions and sizes are
// invented for the picture (real ones are set by the hardware and not given in
// our sources); one dimension; no caches; the .const and .param windows are
// left out.
//
// Part 2, rules for the "does it compile, which memory" cards. CUDA and HIP
// helpers take generic pointers, so one helper serves both memories and the
// argument's address picks the memory (A2, B1). The Metal Shading Language, as
// its specification is quoted in appC-facts C1, has disjoint address spaces:
// a pointer without one is a compile error, and overloading includes the
// address space, so a helper needs one definition per memory. A second rule
// set records what the Metal compiler on the build machine did at its default
// language version, MSL 4.1: it accepted an unqualified pointer as generic, and
// the call read the memory its argument pointed into. appC-facts C1 records this
// as observed, not documented, and labs/appC/snippets/check.py repeats it; the
// page labels it so.
//
// Only the i18n helper is imported; browser checks load this file directly.

import { tr } from '../lib/locale.ts';

// ---------------------------------------------------------------- part 1: windows
export type Window = 'local' | 'shared' | 'global';
export type TierName = 'reg' | 'shared' | 'global';

export const STRIP_TOP = 0x10000;
export const WINDOWS: { window: Exclude<Window, 'global'>; lo: number; hi: number }[] = [
  { window: 'local', lo: 0xa000, hi: 0xb800 },
  { window: 'shared', lo: 0xd000, hi: 0xe000 },
];

export interface Classified {
  window: Window;
  /** The PTX state space the access goes to. */
  state: '.local' | '.shared' | '.global';
  /** The state-space load the hardware ends up performing. */
  load: 'ld.local' | 'ld.shared' | 'ld.global';
  /**
   * Colour on the page. Local memory takes the global colour, as in chapter 5
   * (ch05/S52.tsx): it is private to a thread, but it physically lives in global
   * memory. The page tells the local window apart by its outline and label.
   */
  tier: TierName;
}

export function classify(addr: number): Classified {
  const w = WINDOWS.find((x) => addr >= x.lo && addr < x.hi)?.window ?? 'global';
  return {
    window: w,
    state: w === 'local' ? '.local' : w === 'shared' ? '.shared' : '.global',
    load: w === 'local' ? 'ld.local' : w === 'shared' ? 'ld.shared' : 'ld.global',
    tier: w === 'shared' ? 'shared' : 'global',
  };
}

/** The three variables the reader can pick, with made-up addresses inside the right windows. */
export const VARIABLES: { id: Window; label: string; code: string; addr: number }[] = [
  { id: 'global', get label() { return tr('an element of a global buffer', 'một phần tử của một buffer toàn cục'); }, code: 'x[37]', addr: 0x1200 + 37 * 4 },
  { id: 'shared', get label() { return tr('an element of a __shared__ tile', 'một phần tử của một ô (tile) __shared__'); }, code: 'tile[5]', addr: 0xd000 + 5 * 4 },
  { id: 'local', get label() { return tr('an element of a per-thread local array', 'một phần tử của một mảng cục bộ riêng của mỗi luồng'); }, code: 'window[2]', addr: 0xa000 + 2 * 4 },
];

// ---------------------------------------------------------------- part 2: cards
export type Lang = 'cuda' | 'hip' | 'msl';
/** Address space of a parameter: CUDA and HIP pointers are generic; MSL ones name a space, or omit it. */
export type ParamSpace = 'generic' | 'device' | 'threadgroup' | 'none';
/** What the call passes: a pointer into global/device memory or into shared/threadgroup memory. */
export type ArgSpace = 'global' | 'shared';
export type Rules = 'spec' | 'observed-4.1';
export type Bin = 'global' | 'shared' | 'error';

export interface Card {
  id: string;
  lang: Lang;
  /** The helper's definition(s), as shown. */
  helper: string;
  /** The call site, as shown. */
  call: string;
  overloads: ParamSpace[];
  arg: ArgSpace;
  caller: 'kernel';
}

export interface Outcome {
  bin: Bin;
  /** The compiler's complaint, for errors. */
  error?: 'missing-space' | 'no-match';
  rule: string;
  source: string;
}

const MSL_SPACE: Record<ArgSpace, ParamSpace> = { global: 'device', shared: 'threadgroup' };

export function judge(card: Card, rules: Rules = 'spec'): Outcome {
  if (card.lang !== 'msl')
    return {
      bin: card.arg,
      rule:
        card.lang === 'cuda'
          ? tr(
              'A plain pointer is generic: the address falls in the shared window or outside it, and the hardware loads from that memory.',
              'Một con trỏ thường là generic: địa chỉ rơi vào cửa sổ của bộ nhớ chia sẻ hoặc nằm ngoài nó, và phần cứng nạp từ bộ nhớ tương ứng.',
            )
          : tr(
              'HIP compiles to AMD\'s "flat" address space: the address falls in the LDS aperture or outside it, as in CUDA.',
              'HIP biên dịch sang không gian địa chỉ "flat" của AMD: địa chỉ rơi vào khe LDS (aperture) hoặc nằm ngoài nó, như trong CUDA.',
            ),
      source:
        card.lang === 'cuda'
          ? 'book p. 624; appC-facts A2' // i18n-ignore: citation
          : tr('appC-facts B1 (the LLVM AMDGPU documentation)', 'appC-facts B1 (tài liệu LLVM AMDGPU)'),
    };
  if (card.overloads.includes('none')) {
    if (rules === 'observed-4.1')
      return {
        bin: card.arg,
        rule: tr(
          'Observed, not documented: at its default language version (MSL 4.1) the Metal compiler on the build machine accepted the unqualified pointer as a generic one, and the call reads the memory its argument points into.',
          'Quan sát được, không có trong tài liệu: ở phiên bản ngôn ngữ mặc định (MSL 4.1), trình biên dịch Metal trên máy build chấp nhận con trỏ không có định danh không gian như một con trỏ generic, và lời gọi đọc bộ nhớ mà đối số của nó trỏ vào.',
        ),
        source: tr(
          'observed on the build machine, recorded in appC-facts C1 against the specification text quoted there; repeated by labs/appC/snippets/check.py',
          'quan sát trên máy build, ghi lại trong appC-facts C1 đối chiếu với đoạn đặc tả được trích ở đó; được lặp lại bởi labs/appC/snippets/check.py',
        ),
      };
    return {
      bin: 'error',
      error: 'missing-space',
      rule: tr(
        'Every pointer must carry an address space (device, threadgroup, constant, thread). A missing one is a compile error.',
        'Mọi con trỏ phải mang một không gian địa chỉ (device, threadgroup, constant, thread). Thiếu nó là lỗi biên dịch.',
      ),
      source: tr('MSL specification §4, appC-facts C1', 'Đặc tả MSL §4, appC-facts C1'),
    };
  }
  const want = MSL_SPACE[card.arg];
  if (card.overloads.includes(want))
    return {
      bin: card.arg,
      rule:
        want === 'threadgroup'
          ? tr(
              'Overloading includes the address space, so the call picks the threadgroup definition. A threadgroup pointer parameter is allowed because a kernel calls the helper.',
              'Nạp chồng (overloading) tính cả không gian địa chỉ, nên lời gọi chọn định nghĩa threadgroup. Tham số con trỏ threadgroup được phép vì một kernel gọi hàm trợ giúp.',
            )
          : tr(
              'Overloading includes the address space, so the call picks the device definition.',
              'Nạp chồng (overloading) tính cả không gian địa chỉ, nên lời gọi chọn định nghĩa device.',
            ),
      source: tr('MSL specification §1.5.1 and §4, appC-facts C1', 'Đặc tả MSL §1.5.1 và §4, appC-facts C1'),
    };
  return {
    bin: 'error',
    error: 'no-match',
    rule: tr(
      `No definition takes a ${want} pointer, and one address space does not convert to another: "no matching function for call".`,
      `Không định nghĩa nào nhận con trỏ ${want}, và một không gian địa chỉ không chuyển được sang không gian khác: "no matching function for call".`,
    ),
    source: tr('MSL specification §1.5.1, appC-facts C1', 'Đặc tả MSL §1.5.1, appC-facts C1'),
  };
}

const CUDA_HELPER = '__device__ float rowMax(const float* p, int n)'; // i18n-ignore: code shown as written
const PAIR = 'float rowMax(device const float* p, uint n)\nfloat rowMax(threadgroup const float* p, uint n)'; // i18n-ignore: code shown as written

export const CARDS: Card[] = [
  { id: 'cuda-shared', lang: 'cuda', helper: CUDA_HELPER, call: '__shared__ float tile[256];\n... rowMax(tile, n)', overloads: ['generic'], arg: 'shared', caller: 'kernel' },
  { id: 'cuda-global', lang: 'cuda', helper: CUDA_HELPER, call: 'rowMax(x + row * n, n)   // x from cudaMalloc', overloads: ['generic'], arg: 'global', caller: 'kernel' }, // i18n-ignore: code shown as written
  { id: 'hip-shared', lang: 'hip', helper: CUDA_HELPER, call: '__shared__ float tile[256];\n... rowMax(tile, n)', overloads: ['generic'], arg: 'shared', caller: 'kernel' },
  { id: 'msl-none', lang: 'msl', helper: 'float rowMax(const float* p, uint n)', call: 'threadgroup float tile[256];\n... rowMax(tile, n)', overloads: ['none'], arg: 'shared', caller: 'kernel' }, // i18n-ignore: code shown as written
  { id: 'msl-device-only', lang: 'msl', helper: 'float rowMax(device const float* p, uint n)', call: 'threadgroup float tile[256];\n... rowMax(tile, n)', overloads: ['device'], arg: 'shared', caller: 'kernel' }, // i18n-ignore: code shown as written
  { id: 'msl-pair-tile', lang: 'msl', helper: PAIR, call: 'threadgroup float tile[256];\n... rowMax(tile, n)', overloads: ['device', 'threadgroup'], arg: 'shared', caller: 'kernel' },
  { id: 'msl-pair-row', lang: 'msl', helper: PAIR, call: 'rowMax(x + row * n, n)   // device const float* x', overloads: ['device', 'threadgroup'], arg: 'global', caller: 'kernel' }, // i18n-ignore: code shown as written
  { id: 'msl-threadgroup', lang: 'msl', helper: 'float rowMax(threadgroup const float* p, uint n)', call: 'kernel void k(...) {\n  threadgroup float tile[256];\n  ... rowMax(tile, n) }', overloads: ['threadgroup'], arg: 'shared', caller: 'kernel' }, // i18n-ignore: code shown as written
];

export const BIN_LABEL: Record<Bin, string> = {
  get global() {
    return tr('compiles, reads global or device memory', 'biên dịch được, đọc bộ nhớ toàn cục hoặc bộ nhớ device');
  },
  get shared() {
    return tr('compiles, reads shared or threadgroup memory', 'biên dịch được, đọc bộ nhớ chia sẻ hoặc bộ nhớ threadgroup');
  },
  get error() {
    return tr('does not compile', 'không biên dịch được');
  },
};

/** Minimum number of definitions so one helper can be called on both memories. */
export function definitionsNeeded(lang: Lang, rules: Rules = 'spec'): number {
  if (lang !== 'msl') return 1;
  return rules === 'observed-4.1' ? 1 : 2;
}
