// Section C.3: why cudaMemcpy had a direction argument, and why
// cudaMemcpyDefault can do without it once there is one virtual address space.
//
// A toy address map with three places memory can be: host memory, GPU 0 and
// GPU 1. In "uva" mode they occupy disjoint ranges of one address space, as
// under CUDA 4's unified virtual address space (book p. 625; appC-facts A5): a
// value lies in at most one range, so the runtime can tell from two pointer
// values which way a copy goes. In "separate" mode each processor has its own
// address space and the three ranges cover the same values, so a value alone
// cannot say where it lives.
//
// An option adds a buffer registered with cudaHostRegister. Under unified
// addressing that memory is one of the documented exceptions: "there exists a
// distinct host and device address for the memory" (A3, A5). The map then shows
// one buffer at two addresses. What cudaMemcpyDefault infers for the device-side
// value is not stated in our sources, so the model says so instead of guessing.
//
// Simplifications (the page says so too): range bases and sizes are invented and
// tiny; one host and two GPUs; the model infers the direction only and does not
// perform or time a copy.
//
// No imports, so browser checks can load this file directly.
import { tr } from '../lib/locale.ts';

export type Space = 'host' | 'gpu0' | 'gpu1';
export type Mode = 'uva' | 'separate';

export interface Range {
  space: Space;
  lo: number;
  /** Exclusive. */
  hi: number;
  label: string;
  /** Set on the two ranges of the registered buffer. */
  registered?: 'host-side' | 'device-side';
}

export const SPACE_LABEL: Record<Space, string> = {
  get host() { return tr('host memory', 'bộ nhớ host'); },
  get gpu0() { return tr('GPU 0', 'GPU 0'); },
  get gpu1() { return tr('GPU 1', 'GPU 1'); },
};

/** The whole toy address space: 0x0000 to 0xFFFF. */
export const TOP = 0x10000;

export const REGISTERED = { host: [0x2000, 0x2800] as const, device: [0x7400, 0x7c00] as const };

export function ranges(mode: Mode, registered = false): Range[] {
  if (mode === 'separate')
    return [
      { space: 'host', lo: 0x1000, hi: 0x7000, label: tr('host memory (its own address space)', 'bộ nhớ host (không gian địa chỉ riêng của nó)') },
      { space: 'gpu0', lo: 0x1000, hi: 0x7000, label: tr('GPU 0 memory (its own address space)', 'bộ nhớ GPU 0 (không gian địa chỉ riêng của nó)') },
      { space: 'gpu1', lo: 0x1000, hi: 0x7000, label: tr('GPU 1 memory (its own address space)', 'bộ nhớ GPU 1 (không gian địa chỉ riêng của nó)') },
    ];
  const out: Range[] = [
    { space: 'host', lo: 0x1000, hi: 0x7000, label: tr('host memory', 'bộ nhớ host') },
    { space: 'gpu0', lo: 0x8000, hi: 0xb800, label: tr('GPU 0 memory', 'bộ nhớ GPU 0') },
    { space: 'gpu1', lo: 0xc000, hi: 0xf800, label: tr('GPU 1 memory', 'bộ nhớ GPU 1') },
  ];
  if (registered) {
    out.push({ space: 'host', lo: REGISTERED.host[0], hi: REGISTERED.host[1], label: tr('registered buffer, host address', 'buffer đã đăng ký, địa chỉ host'), registered: 'host-side' });
    out.push({ space: 'host', lo: REGISTERED.device[0], hi: REGISTERED.device[1], label: tr('registered buffer, device address', 'buffer đã đăng ký, địa chỉ device'), registered: 'device-side' });
  }
  return out;
}

/** Every range containing the value. Under UVA a registered host buffer is reported alone, not with host memory around it. */
export function locate(addr: number, mode: Mode, registered = false): Range[] {
  const hits = ranges(mode, registered).filter((r) => addr >= r.lo && addr < r.hi);
  const reg = hits.filter((r) => r.registered);
  return reg.length ? reg : hits;
}

export type Kind =
  | 'cudaMemcpyHostToHost'
  | 'cudaMemcpyHostToDevice'
  | 'cudaMemcpyDeviceToHost'
  | 'cudaMemcpyDeviceToDevice'
  | 'ambiguous'
  | 'unmapped'
  | 'alias';

export interface Inference {
  kind: Kind;
  /** One sentence that the page shows under the result. */
  detail: string;
  src: Range[];
  dst: Range[];
}

const isHost = (s: Space) => s === 'host';
const where = (r: Range) => (r.space === 'host' ? tr('host memory', 'bộ nhớ host') : SPACE_LABEL[r.space]);

/** What cudaMemcpy(dst, src, n, cudaMemcpyDefault) would infer from the two values. */
export function inferKind(dst: number, src: number, mode: Mode, registered = false): Inference {
  const s = locate(src, mode, registered);
  const d = locate(dst, mode, registered);
  const base = { src: s, dst: d };
  if (!s.length || !d.length) {
    const which = !s.length && !d.length ? tr('Neither value', 'Cả hai giá trị đều') : !s.length ? tr('The source value', 'Giá trị nguồn') : tr('The destination value', 'Giá trị đích');
    return { ...base, kind: 'unmapped', detail: tr(`${which} lies in no allocation, so there is nothing to infer.`, `${which} không nằm trong vùng cấp phát nào, nên không có gì để suy ra.`) };
  }
  if (s.length > 1 || d.length > 1)
    return {
      ...base,
      kind: 'ambiguous',
      detail: tr(`Cannot tell: the ${s.length > 1 ? 'source' : 'destination'} value names a location in ${Math.max(s.length, d.length)} separate address spaces, so you must say the direction yourself.`, `Không xác định được: giá trị ${s.length > 1 ? 'nguồn' : 'đích'} chỉ tới một vị trí trong ${Math.max(s.length, d.length)} không gian địa chỉ riêng biệt, nên bạn phải tự chỉ ra chiều sao chép.`),
    };
  const [a, b] = [s[0], d[0]];
  if (a.registered === 'device-side' || b.registered === 'device-side')
    return {
      ...base,
      kind: 'alias',
      detail:
        tr('This value is the device-side address of the registered buffer: one buffer, two addresses (an exception to unified addressing). Our sources do not say what cudaMemcpyDefault infers for it.', 'Giá trị này là địa chỉ phía device của buffer đã đăng ký: một buffer, hai địa chỉ (một ngoại lệ của định địa chỉ hợp nhất). Các nguồn của chúng tôi không nói cudaMemcpyDefault suy ra gì cho nó.'),
    };
  const kind: Kind = isHost(a.space)
    ? isHost(b.space)
      ? 'cudaMemcpyHostToHost'
      : 'cudaMemcpyHostToDevice'
    : isHost(b.space)
      ? 'cudaMemcpyDeviceToHost'
      : 'cudaMemcpyDeviceToDevice';
  return { ...base, kind, detail: tr(`From ${where(a)} to ${where(b)}, read off the two values.`, `Từ ${where(a)} tới ${where(b)}, đọc ra từ hai giá trị.`) };
}

/** 0x2c00, four hex digits. */
export const hex4 = (a: number) => '0x' + a.toString(16).padStart(4, '0');
