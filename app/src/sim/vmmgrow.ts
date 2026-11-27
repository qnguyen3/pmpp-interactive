// Virtual address space control: Appendix C, §C.9 (book p. 629), and the narrated
// scene "Growing a buffer without copying it".
//
// The book names three calls: cuMemAddressReserve (reserve a virtual range),
// cuMemCreate (physical memory on any device) and cuMemMap (map it anywhere in the
// range), for layouts such as one volume spread over several devices behind one
// pointer. Checked against NVIDIA (notes/today/appC-facts.md A15):
//   - "Mapping the allocation to a region of the address range using cuMemMap does not
//     make the address accessible, and would result in a program crash if accessed by
//     a CUDA kernel. Users must specifically select access control using the
//     cuMemSetAccess function on source and accessing devices."
//   - Teardown order: cuMemUnmap, cuMemRelease, cuMemAddressFree.
//   - The range "can be mapped to memory chunks belonging to any device in the system".
//   - HIP: "If the allocation size is not aligned ... hipMemCreate() will return an
//     out-of-memory error" (B7). CUDA's rule for this is not in our checked sources;
//     the model applies the same rule to both and says so.
//
// Part 1, the stepper: apply() validates one operation against those rules.
//   - Mapping outside the reservation is refused: the book has cuMemMap place memory
//     inside the reserved range.
//   - Mapping over an address that is already mapped, mapping one chunk twice, and
//     granting access over unmapped addresses are refused by the model. Our sources do
//     not say what the driver does; the app says these are the model's rules.
//   - A kernel read of an address with nothing mapped, or of mapped memory the reading
//     device has no access to, is a crash.
//   - Teardown out of the documented order is flagged and not applied. The model does
//     not say what a driver would do out of order.
//
// Part 2, growth: grow() runs one strategy over a sequence of tokens, each adding
// bytesPerToken to a buffer such as a KV cache, and counts bytes copied, the most
// memory allocated at once, memory allocated but unused, and base-pointer changes.
//   - 'double': allocate a buffer twice as large when full, copy, free the old one.
//   - 'exact': allocate exactly the new size at every token, copy, free the old one.
//   - 'vmm': reserve addresses once, map one more chunk whenever the data needs it.
// Simplifications: one sequence on one device; allocation and mapping are not timed;
// no other allocations, so no fragmentation from them; addresses for the copying
// strategies come from a bump allocator that never reuses a freed hole (only the fact
// that the base changes matters, and that holds whatever the allocator does, since the
// old buffer is still live while the new one is filled).
//
// No imports and only erasable TypeScript, so browser checks can load this file
// directly with Node.

import { tr } from '../lib/locale.ts';

export const MiB = 1024 * 1024;

// ================================================================ part 1: the stepper

export type Dev = 0 | 1;

export interface Chunk {
  /** Handle number, h1, h2, ... */
  id: number;
  device: Dev;
  size: number;
  released: boolean;
}
export interface Mapping {
  chunk: number;
  offset: number;
  size: number;
}
export interface Grant {
  offset: number;
  size: number;
  device: Dev;
}

export interface VmmState {
  /** The allocation granularity, a parameter of the model (not a device value). */
  gran: number;
  /** Size of the reserved range (it starts at offset 0), or null when none is reserved. */
  reserved: number | null;
  /** The range was reserved and has been freed. */
  freed: boolean;
  chunks: Chunk[];
  maps: Mapping[];
  grants: Grant[];
  nextId: number;
}

export type Op =
  | { op: 'reserve'; size: number }
  | { op: 'create'; device: Dev; size: number }
  | { op: 'map'; chunk: number; offset: number }
  | { op: 'setAccess'; offset: number; size: number; device: Dev }
  | { op: 'read'; device: Dev; offset: number; size: number }
  | { op: 'unmap'; chunk: number }
  | { op: 'release'; chunk: number }
  | { op: 'free' };

export type OutcomeKind = 'ok' | 'error' | 'crash' | 'order';

export interface ReadPiece {
  offset: number;
  size: number;
  status: 'local' | 'remote' | 'unmapped' | 'no-access';
  /** Which device's memory backs the piece, when something is mapped there. */
  backing?: Dev;
}

export interface Outcome {
  op: Op;
  kind: OutcomeKind;
  /** Short machine-readable code: 'ok', 'misaligned', 'outside', 'no-access', ... */
  code: string;
  /** The CUDA call (or "kernel") this step stands for. */
  call: string;
  /** The HIP counterpart (B7), where there is one. */
  hip?: string;
  text: string;
  pieces?: ReadPiece[];
}

export const CALLS: Record<Op['op'], { cuda: string; hip?: string }> = {
  reserve: { cuda: 'cuMemAddressReserve', hip: 'hipMemAddressReserve' },
  create: { cuda: 'cuMemCreate', hip: 'hipMemCreate' },
  map: { cuda: 'cuMemMap', hip: 'hipMemMap' },
  setAccess: { cuda: 'cuMemSetAccess', hip: 'hipMemSetAccess' },
  read: { cuda: 'kernel' },
  unmap: { cuda: 'cuMemUnmap' },
  release: { cuda: 'cuMemRelease' },
  free: { cuda: 'cuMemAddressFree' },
};

export function initial(gran: number): VmmState {
  return { gran, reserved: null, freed: false, chunks: [], maps: [], grants: [], nextId: 1 };
}

const overlaps = (a: number, as: number, b: number, bs: number) => a < b + bs && b < a + as;

/** Parts of [a, a + size) not covered by the given intervals. */
export function uncovered(intervals: { offset: number; size: number }[], a: number, size: number): { offset: number; size: number }[] {
  const sorted = intervals.filter((i) => overlaps(i.offset, i.size, a, size)).sort((x, y) => x.offset - y.offset);
  const out: { offset: number; size: number }[] = [];
  let at = a;
  const end = a + size;
  for (const i of sorted) {
    if (i.offset > at) out.push({ offset: at, size: Math.min(i.offset, end) - at });
    at = Math.max(at, i.offset + i.size);
    if (at >= end) break;
  }
  if (at < end) out.push({ offset: at, size: end - at });
  return out;
}

/** Removes [a, a + size) from a list of grants, splitting where needed. */
function cutGrants(grants: Grant[], a: number, size: number): Grant[] {
  const out: Grant[] = [];
  for (const g of grants) {
    if (!overlaps(g.offset, g.size, a, size)) {
      out.push(g);
      continue;
    }
    if (g.offset < a) out.push({ ...g, size: a - g.offset });
    const gEnd = g.offset + g.size;
    if (gEnd > a + size) out.push({ ...g, offset: a + size, size: gEnd - (a + size) });
  }
  return out;
}

/** A human-readable size in granules, e.g. "2 granules", "1.5 granules". */
export const granules = (bytes: number, gran: number) => {
  const n = bytes / gran;
  return tr(`${Number(n.toFixed(2))} granule${n === 1 ? '' : 's'}`, `${Number(n.toFixed(2))} granule`);
};

/** Applies one operation. Never throws; refused operations leave the state unchanged. */
export function apply(s: VmmState, op: Op): { state: VmmState; outcome: Outcome } {
  const call = CALLS[op.op];
  const out = (kind: OutcomeKind, code: string, text: string, state: VmmState = s, pieces?: ReadPiece[]) => ({
    state,
    outcome: { op, kind, code, call: call.cuda, hip: call.hip, text, pieces } as Outcome,
  });
  const g = (b: number) => granules(b, s.gran);

  switch (op.op) {
    case 'reserve': {
      if (s.reserved != null) return out('error', 'twice', tr('This model keeps one reserved range; it is already reserved.', 'Mô hình này chỉ giữ một vùng địa chỉ đã đặt trước; vùng đó đã được đặt trước rồi.'));
      if (!(op.size > 0)) return out('error', 'size', tr('A range needs a positive size.', 'Một vùng cần kích thước dương.'));
      return out('ok', 'ok', tr(`Reserved ${g(op.size)} of addresses. Nothing is behind them yet: reserving costs addresses, not memory.`, `Đã đặt trước ${g(op.size)} địa chỉ. Chưa có gì phía sau chúng: đặt trước chỉ tốn địa chỉ, không tốn bộ nhớ.`), {
        ...s,
        reserved: op.size,
        freed: false,
      });
    }
    case 'create': {
      if (!(op.size > 0)) return out('error', 'size', tr('A chunk needs a positive size.', 'Một chunk cần kích thước dương.'));
      if (op.size % s.gran !== 0)
        return out(
          'error',
          'misaligned',
          tr(`Refused: ${g(op.size)} is not a whole number of granules. HIP documents an out-of-memory error from hipMemCreate for this (B7); CUDA's rule is not in our checked sources.`, `Bị từ chối: ${g(op.size)} không phải là một số nguyên granule. HIP ghi trong tài liệu rằng hipMemCreate trả lỗi hết bộ nhớ (out-of-memory) cho trường hợp này (B7); quy tắc của CUDA không có trong các nguồn chúng tôi đã kiểm tra.`),
        );
      const chunk: Chunk = { id: s.nextId, device: op.device, size: op.size, released: false };
      return out('ok', 'ok', tr(`Created chunk h${chunk.id}: ${g(op.size)} of physical memory on GPU ${op.device}. It has no address yet.`, `Đã tạo chunk h${chunk.id}: ${g(op.size)} bộ nhớ vật lý trên GPU ${op.device}. Nó chưa có địa chỉ.`), {
        ...s,
        chunks: [...s.chunks, chunk],
        nextId: s.nextId + 1,
      });
    }
    case 'map': {
      if (s.reserved == null) return out('error', 'no-range', tr('There is no reserved range to map into.', 'Không có vùng đã đặt trước nào để ánh xạ vào.'));
      const c = s.chunks.find((x) => x.id === op.chunk);
      if (!c) return out('error', 'no-chunk', tr(`There is no chunk h${op.chunk}.`, `Không có chunk h${op.chunk}.`));
      if (c.released) return out('error', 'released', tr(`Chunk h${c.id} has been released.`, `Chunk h${c.id} đã được giải phóng.`));
      if (s.maps.some((m) => m.chunk === c.id)) return out('error', 'mapped', tr(`Chunk h${c.id} is already mapped (this model maps a chunk at one place only).`, `Chunk h${c.id} đã được ánh xạ rồi (mô hình này chỉ ánh xạ một chunk ở một chỗ).`));
      if (op.offset < 0 || op.offset + c.size > s.reserved)
        return out('error', 'outside', tr(`Refused: h${c.id} at granule ${op.offset / s.gran} would run past the end of the reserved range.`, `Bị từ chối: h${c.id} tại granule ${op.offset / s.gran} sẽ vượt quá cuối vùng đã đặt trước.`));
      const hit = s.maps.find((m) => overlaps(m.offset, m.size, op.offset, c.size));
      if (hit) return out('error', 'overlap', tr(`Refused by the model: those addresses already have chunk h${hit.chunk} behind them.`, `Mô hình từ chối: các địa chỉ đó đã có chunk h${hit.chunk} phía sau.`));
      return out(
        'ok',
        'ok',
        tr(`Mapped h${c.id} (GPU ${c.device}) at granule ${op.offset / s.gran}. The addresses now have memory behind them, but no device may access it yet.`, `Đã ánh xạ h${c.id} (GPU ${c.device}) tại granule ${op.offset / s.gran}. Giờ các địa chỉ đã có bộ nhớ phía sau, nhưng chưa thiết bị nào được phép truy cập nó.`),
        { ...s, maps: [...s.maps, { chunk: c.id, offset: op.offset, size: c.size }] },
      );
    }
    case 'setAccess': {
      if (s.reserved == null) return out('error', 'no-range', tr('There is no reserved range.', 'Không có vùng đã đặt trước nào.'));
      if (op.offset < 0 || op.size <= 0 || op.offset + op.size > s.reserved) return out('error', 'outside', tr('That range is not inside the reservation.', 'Vùng đó không nằm trong vùng đã đặt trước.'));
      const gaps = uncovered(s.maps, op.offset, op.size);
      if (gaps.length)
        return out('error', 'unmapped', tr(`Refused by the model: granule ${gaps[0].offset / s.gran} has nothing mapped. Grant access over mapped memory.`, `Mô hình từ chối: granule ${gaps[0].offset / s.gran} chưa có gì được ánh xạ. Hãy cấp quyền truy cập trên bộ nhớ đã ánh xạ.`));
      const grants = [...cutGrants(s.grants.filter((x) => x.device === op.device), op.offset, op.size), { offset: op.offset, size: op.size, device: op.device }];
      return out('ok', 'ok', tr(`GPU ${op.device} may now read and write granules ${op.offset / s.gran} to ${(op.offset + op.size) / s.gran - 1}.`, `Giờ GPU ${op.device} được đọc và ghi các granule từ ${op.offset / s.gran} đến ${(op.offset + op.size) / s.gran - 1}.`), {
        ...s,
        grants: [...s.grants.filter((x) => x.device !== op.device), ...grants],
      });
    }
    case 'read': {
      if (s.reserved == null || op.offset < 0 || op.offset + op.size > s.reserved)
        return out('crash', 'unreserved', tr(`The kernel on GPU ${op.device} crashes: those addresses are not in a reserved range.`, `Kernel trên GPU ${op.device} bị crash: các địa chỉ đó không nằm trong vùng đã đặt trước.`));
      const pieces: ReadPiece[] = [];
      for (const gap of uncovered(s.maps, op.offset, op.size)) pieces.push({ ...gap, status: 'unmapped' });
      for (const m of s.maps) {
        if (!overlaps(m.offset, m.size, op.offset, op.size)) continue;
        const a = Math.max(m.offset, op.offset);
        const e = Math.min(m.offset + m.size, op.offset + op.size);
        const backing = (s.chunks.find((c) => c.id === m.chunk) as Chunk).device;
        const mine = s.grants.filter((x) => x.device === op.device);
        const noAccess = uncovered(mine, a, e - a);
        let at = a;
        for (const n of noAccess) {
          if (n.offset > at) pieces.push({ offset: at, size: n.offset - at, status: backing === op.device ? 'local' : 'remote', backing });
          pieces.push({ ...n, status: 'no-access', backing });
          at = n.offset + n.size;
        }
        if (at < e) pieces.push({ offset: at, size: e - at, status: backing === op.device ? 'local' : 'remote', backing });
      }
      pieces.sort((x, y) => x.offset - y.offset);
      const bad = pieces.find((p) => p.status === 'unmapped' || p.status === 'no-access');
      if (bad?.status === 'unmapped')
        return out('crash', 'unmapped', tr(`The kernel on GPU ${op.device} crashes at granule ${bad.offset / s.gran}: nothing is mapped there.`, `Kernel trên GPU ${op.device} bị crash tại granule ${bad.offset / s.gran}: ở đó chưa có gì được ánh xạ.`), s, pieces);
      if (bad)
        return out(
          'crash',
          'no-access',
          tr(`The kernel on GPU ${op.device} crashes at granule ${bad.offset / s.gran}: memory is mapped there, but GPU ${op.device} was never granted access. cuMemMap does not make an address accessible; cuMemSetAccess does (A15).`, `Kernel trên GPU ${op.device} bị crash tại granule ${bad.offset / s.gran}: ở đó đã có bộ nhớ được ánh xạ, nhưng GPU ${op.device} chưa bao giờ được cấp quyền truy cập. cuMemMap không làm cho một địa chỉ truy cập được; cuMemSetAccess mới làm điều đó (A15).`),
          s,
          pieces,
        );
      const remote = pieces.filter((p) => p.status === 'remote').reduce((t, p) => t + p.size, 0);
      return out(
        'ok',
        'ok',
        remote
          ? tr(`The kernel on GPU ${op.device} reads ${g(op.size)} through one pointer: ${g(op.size - remote)} from its own memory and ${g(remote)} from the other GPU's.`, `Kernel trên GPU ${op.device} đọc ${g(op.size)} qua một con trỏ: ${g(op.size - remote)} từ bộ nhớ của chính nó và ${g(remote)} từ bộ nhớ của GPU kia.`)
          : tr(`The kernel on GPU ${op.device} reads ${g(op.size)}, all from its own memory.`, `Kernel trên GPU ${op.device} đọc ${g(op.size)}, tất cả từ bộ nhớ của chính nó.`),
        s,
        pieces,
      );
    }
    case 'unmap': {
      const m = s.maps.find((x) => x.chunk === op.chunk);
      if (!m) return out('error', 'not-mapped', tr(`Chunk h${op.chunk} is not mapped.`, `Chunk h${op.chunk} chưa được ánh xạ.`));
      return out('ok', 'ok', tr(`Unmapped h${op.chunk}. Its addresses are still reserved, and its memory still exists.`, `Đã bỏ ánh xạ h${op.chunk}. Các địa chỉ của nó vẫn được đặt trước, và bộ nhớ của nó vẫn còn.`), {
        ...s,
        maps: s.maps.filter((x) => x !== m),
        grants: cutGrants(s.grants, m.offset, m.size),
      });
    }
    case 'release': {
      const c = s.chunks.find((x) => x.id === op.chunk);
      if (!c) return out('error', 'no-chunk', tr(`There is no chunk h${op.chunk}.`, `Không có chunk h${op.chunk}.`));
      if (c.released) return out('error', 'released', tr(`Chunk h${c.id} is already released.`, `Chunk h${c.id} đã được giải phóng rồi.`));
      if (s.maps.some((m) => m.chunk === c.id))
        return out('order', 'order', tr(`Outside the documented order: h${c.id} is still mapped. Unmap it first (cuMemUnmap, then cuMemRelease, then cuMemAddressFree). Not applied.`, `Ngoài thứ tự được ghi trong tài liệu: h${c.id} vẫn đang được ánh xạ. Hãy bỏ ánh xạ nó trước (cuMemUnmap, rồi cuMemRelease, rồi cuMemAddressFree). Không được áp dụng.`));
      return out('ok', 'ok', tr(`Released h${c.id}: its memory on GPU ${c.device} is given back.`, `Đã giải phóng h${c.id}: bộ nhớ của nó trên GPU ${c.device} được trả lại.`), {
        ...s,
        chunks: s.chunks.map((x) => (x.id === c.id ? { ...x, released: true } : x)),
      });
    }
    case 'free': {
      if (s.reserved == null) return out('error', 'no-range', tr('There is no reserved range to free.', 'Không có vùng đã đặt trước nào để giải phóng.'));
      if (s.maps.length)
        return out('order', 'order', tr('Outside the documented order: chunks are still mapped in the range. Unmap and release them first. Not applied.', 'Ngoài thứ tự được ghi trong tài liệu: vẫn còn chunk được ánh xạ trong vùng. Hãy bỏ ánh xạ và giải phóng chúng trước. Không được áp dụng.'));
      return out('ok', 'ok', tr('Freed the reserved addresses.', 'Đã giải phóng các địa chỉ đã đặt trước.'), { ...s, reserved: null, freed: true, grants: [] });
    }
  }
}

/** Runs a list of operations from an empty state. */
export function run(gran: number, ops: Op[]): { state: VmmState; log: Outcome[] } {
  let state = initial(gran);
  const log: Outcome[] = [];
  for (const op of ops) {
    const r = apply(state, op);
    state = r.state;
    log.push(r.outcome);
  }
  return { state, log };
}

/** Everything is torn down: nothing mapped, every chunk released, the range freed. */
export const clean = (s: VmmState) => s.reserved == null && s.maps.length === 0 && s.chunks.every((c) => c.released);

/** What backs each granule of the range, for drawing: null if nothing is mapped there. */
export function slots(s: VmmState): { chunk: number | null; device: Dev | null; access: [boolean, boolean] }[] {
  if (s.reserved == null) return [];
  const n = Math.floor(s.reserved / s.gran);
  return Array.from({ length: n }, (_, k) => {
    const at = k * s.gran;
    const m = s.maps.find((x) => at >= x.offset && at < x.offset + x.size);
    const c = m ? s.chunks.find((x) => x.id === m.chunk) : undefined;
    const has = (d: Dev) => s.grants.some((x) => x.device === d && at >= x.offset && at < x.offset + x.size);
    return { chunk: m ? m.chunk : null, device: c ? c.device : null, access: [has(0), has(1)] as [boolean, boolean] };
  });
}

/**
 * The book's example (p. 629): one pointer over a volume whose first half lives on
 * GPU 0 and second half on GPU 1, readable by a kernel on GPU 0. Eight granules.
 */
export function bookExample(gran: number): Op[] {
  return [
    { op: 'reserve', size: 8 * gran },
    { op: 'create', device: 0, size: 4 * gran },
    { op: 'create', device: 1, size: 4 * gran },
    { op: 'map', chunk: 1, offset: 0 },
    { op: 'map', chunk: 2, offset: 4 * gran },
    { op: 'setAccess', offset: 0, size: 8 * gran, device: 0 },
    { op: 'read', device: 0, offset: 0, size: 8 * gran },
  ];
}

/** Only the three calls the book lists, then a kernel read: it crashes. */
export function bookThreeCalls(gran: number): Op[] {
  return [
    { op: 'reserve', size: 8 * gran },
    { op: 'create', device: 0, size: 4 * gran },
    { op: 'map', chunk: 1, offset: 0 },
    { op: 'read', device: 0, offset: 0, size: 4 * gran },
  ];
}

// ================================================================ part 2: growth

export type Strategy = 'double' | 'exact' | 'vmm';

export const STRATEGIES: { key: Strategy; label: string }[] = [
  {
    key: 'double',
    get label() {
      return tr('Allocate twice as much and copy', 'Cấp phát gấp đôi rồi sao chép');
    },
  },
  {
    key: 'exact',
    get label() {
      return tr('Reallocate exactly, every token', 'Cấp phát lại vừa đủ, mỗi token');
    },
  },
  {
    key: 'vmm',
    get label() {
      return tr('Reserve once, map chunks', 'Đặt trước một lần, ánh xạ các chunk');
    },
  },
];

export interface GrowOpts {
  tokens: number;
  bytesPerToken: number;
  /** Chunk size: the starting capacity for 'double', the mapping unit for 'vmm'. */
  chunk: number;
  /** 'vmm' only: addresses reserved up front. Defaults to the final size rounded up to a chunk. */
  reserve?: number;
}

export interface Block {
  base: number;
  size: number;
}

export interface GrowStep {
  /** Tokens held after this step, 1-based. */
  token: number;
  live: number;
  /** Bytes allocated (for 'vmm': mapped) after the step. */
  allocated: number;
  /** Most bytes allocated at once during the step: old plus new while copying. */
  during: number;
  copied: number;
  copiedTotal: number;
  /** The buffer's base address after the step. */
  base: number;
  /** The base changed in this step (the first allocation does not count). */
  moved: boolean;
  pointerChanges: number;
  /** The buffer before a move. */
  from?: Block;
  /** The buffer after the step; for 'vmm', the mapped extent. */
  to: Block;
  /** 'vmm': chunks mapped after the step, and how many were mapped in it. */
  mapped?: number;
  newChunks?: number;
  /** 'vmm': the data has outgrown the reservation. */
  overflow?: boolean;
}

export interface GrowResult {
  strategy: Strategy;
  steps: GrowStep[];
  copiedTotal: number;
  /** Most bytes allocated at once over the whole run. */
  peak: number;
  finalLive: number;
  finalAllocated: number;
  /** Allocated (or mapped) but not holding data, at the end. */
  unused: number;
  pointerChanges: number;
  /** 'vmm': addresses reserved. */
  reserved?: number;
  /** 'vmm': the data outgrew the reservation at some step. */
  overflow: boolean;
  /** copiedTotal / finalLive. */
  copyRatio: number;
  /** Largest (allocated during the step) / (data being copied), over steps that copy. */
  peakRatio: number;
}

const ceilDiv = (a: number, b: number) => Math.ceil(a / b);

export function grow(strategy: Strategy, o: GrowOpts): GrowResult {
  const T = Math.max(1, Math.floor(o.tokens));
  const b = o.bytesPerToken;
  const steps: GrowStep[] = [];
  let copiedTotal = 0;
  let peak = 0;
  let peakRatio = 0;
  let pointerChanges = 0;

  if (strategy === 'vmm') {
    const reserved = o.reserve ?? ceilDiv(T * b, o.chunk) * o.chunk;
    const maxChunks = Math.floor(reserved / o.chunk);
    let mapped = 0;
    let overflow = false;
    for (let t = 1; t <= T; t++) {
      const live = t * b;
      const need = ceilDiv(live, o.chunk);
      const want = Math.min(need, maxChunks);
      if (need > maxChunks) overflow = true;
      const newChunks = Math.max(0, want - mapped);
      mapped = Math.max(mapped, want);
      const allocated = mapped * o.chunk;
      peak = Math.max(peak, allocated);
      steps.push({
        token: t,
        live,
        allocated,
        during: allocated,
        copied: 0,
        copiedTotal: 0,
        base: 0,
        moved: false,
        pointerChanges: 0,
        to: { base: 0, size: allocated },
        mapped,
        newChunks,
        overflow: need > maxChunks,
      });
    }
    const last = steps[steps.length - 1];
    return {
      strategy,
      steps,
      copiedTotal: 0,
      peak,
      finalLive: last.live,
      finalAllocated: last.allocated,
      unused: Math.max(0, last.allocated - last.live),
      pointerChanges: 0,
      reserved,
      overflow,
      copyRatio: 0,
      peakRatio: 0,
    };
  }

  let top = 0;
  const alloc = (size: number): Block => {
    const blk = { base: top, size };
    top += size;
    return blk;
  };
  let buf: Block | null = null;
  for (let t = 1; t <= T; t++) {
    const live = t * b;
    const prevLive = (t - 1) * b;
    if (!buf) {
      let cap = strategy === 'double' ? o.chunk : live;
      while (cap < live) cap *= 2;
      buf = alloc(cap);
      peak = Math.max(peak, cap);
      steps.push({ token: t, live, allocated: cap, during: cap, copied: 0, copiedTotal, base: buf.base, moved: false, pointerChanges, to: buf });
      continue;
    }
    const full = live > buf.size;
    if (!full) {
      steps.push({ token: t, live, allocated: buf.size, during: buf.size, copied: 0, copiedTotal, base: buf.base, moved: false, pointerChanges, to: buf });
      continue;
    }
    let cap = strategy === 'double' ? buf.size * 2 : live;
    while (cap < live) cap *= 2;
    const from = buf;
    const to = alloc(cap);
    const during = from.size + to.size;
    copiedTotal += prevLive;
    pointerChanges++;
    peak = Math.max(peak, during);
    if (prevLive > 0) peakRatio = Math.max(peakRatio, during / prevLive);
    buf = to;
    steps.push({ token: t, live, allocated: cap, during, copied: prevLive, copiedTotal, base: to.base, moved: true, pointerChanges, from, to });
  }
  const last = steps[steps.length - 1];
  return {
    strategy,
    steps,
    copiedTotal,
    peak,
    finalLive: last.live,
    finalAllocated: last.allocated,
    unused: last.allocated - last.live,
    pointerChanges,
    overflow: false,
    copyRatio: last.live > 0 ? copiedTotal / last.live : 0,
    peakRatio,
  };
}

/**
 * Bytes per token of a KV cache, the book's Eq. 20.7 (ch. 20 §20.6, pp. 505–506):
 * 2 × layers × hidden dimension × bytes per value, where the book defines the hidden
 * dimension as query heads × head size and the 2 counts K and V.
 */
export const kvBytesPerToken = (layers: number, hidden: number, bytes: number) => 2 * layers * hidden * bytes;

/** The book's Llama 2 7B example at 16 bits (p. 506): 32 layers, hidden dimension 4096. 512 KiB. */
export const LLAMA2_7B_BYTES_PER_TOKEN = kvBytesPerToken(32, 4096, 2);
