// Section C.3: one GPU thread follows a linked list that the host built.
//
// Each node was allocated by the host with one of four allocators. The kernel
// receives the head as an argument and then follows `next` pointers it reads
// from the nodes. Whether each hop is a valid dereference, and what it costs,
// is not decided here: it is the derefmap cell for the node's allocator at the
// chosen setting, "kernel, given the pointer as an argument" for the head and
// "kernel, pointer read from memory" for every later node. So this stepper and
// the map on the start page cannot disagree.
//
// What the model simplifies (the page says so too):
//   - One thread; no timing.
//   - "Invalid dereference" stands for undefined behaviour, which on a GPU may
//     crash the kernel or quietly return garbage. The walk stops there.
//   - Addresses are invented for the picture: one made-up range per allocator,
//     nodes 64 bytes apart within it.

import { cell, type Cell, type EraId, type RowId } from './derefmap';
import { tr } from '../lib/locale.ts';

export type Alloc = 'malloc' | 'cudaHostAlloc' | 'cudaMalloc' | 'cudaMallocManaged';
export type Setting = 'uva' | 'pascal' | 'hmm' | 'um-limited';
export type HopVerdict = 'local' | 'remote' | 'fault' | 'invalid';

export const ALLOCS: Alloc[] = ['malloc', 'cudaHostAlloc', 'cudaMalloc', 'cudaMallocManaged'];

export const SETTINGS: { id: Setting; label: string; era: EraId }[] = [
  {
    id: 'uva',
    get label() {
      return tr('CUDA 4, unified addressing', 'CUDA 4, định địa chỉ thống nhất (unified addressing)');
    },
    era: 'uva',
  },
  {
    id: 'pascal',
    get label() {
      return tr('Pascal and later, managed only (Linux, no HMM)', 'Pascal trở về sau, chỉ bộ nhớ managed (Linux, không HMM)');
    },
    era: 'pascal',
  },
  {
    id: 'hmm',
    get label() {
      return tr('HMM or ATS', 'HMM hoặc ATS');
    },
    era: 'hmm',
  },
  {
    id: 'um-limited',
    get label() {
      return tr('Limited (Windows, WSL)', 'Hạn chế (Windows, WSL)');
    },
    era: 'um-limited',
  },
];

const ROW: Record<Alloc, RowId> = {
  malloc: 'malloc',
  cudaHostAlloc: 'pinned',
  cudaMalloc: 'device',
  cudaMallocManaged: 'managed',
};

/** Made-up base addresses, one range per allocator. */
export const RANGES: Record<Alloc, { base: number; name: string }> = {
  malloc: {
    base: 0x7f2000001000,
    get name() {
      return tr('host heap (malloc)', 'heap của host (malloc)');
    },
  },
  cudaHostAlloc: {
    base: 0x7f3000000000,
    get name() {
      return tr('pinned host memory', 'bộ nhớ ghim (pinned memory) của host');
    },
  },
  cudaMalloc: {
    base: 0x7f4000000000,
    get name() {
      return tr('GPU 0 memory', 'bộ nhớ của GPU 0');
    },
  },
  cudaMallocManaged: {
    base: 0x7f5000000000,
    get name() {
      return tr('managed memory', 'bộ nhớ managed');
    },
  },
};

export const NODE_VALUES = [3, 1, 4, 1, 5];

export const HOP_LABEL: Record<HopVerdict, string> = {
  get local() {
    return tr('read locally', 'đọc cục bộ');
  },
  get remote() {
    return tr('read over the interconnect', 'đọc qua kênh kết nối');
  },
  get fault() {
    return tr('page faults and migrates', 'lỗi trang và di chuyển');
  },
  get invalid() {
    return tr('invalid dereference', 'dereference không hợp lệ');
  },
};

/** 0x7f30_0000_0040: groups of four hex digits, as debuggers often print them. */
export function hex(a: number): string {
  const h = a.toString(16).padStart(12, '0');
  return '0x' + h.replace(/(.{4})(?=.)/g, '$1_');
}

/** Each node's address: its allocator's base plus 64 bytes per earlier node from the same allocator. */
export function addresses(nodes: Alloc[]): number[] {
  const seen: Record<Alloc, number> = { malloc: 0, cudaHostAlloc: 0, cudaMalloc: 0, cudaMallocManaged: 0 };
  return nodes.map((a) => RANGES[a].base + 0x40 * seen[a]++);
}

export function hopVerdict(c: Cell): HopVerdict {
  switch (c.verdict) {
    case 'local':
    case 'bulk': // moved to the GPU at launch, so the read itself is local
      return 'local';
    case 'remote':
      return 'remote';
    case 'fault':
      return 'fault';
    default:
      return 'invalid';
  }
}

export interface Hop {
  /** 0-based node index. */
  index: number;
  alloc: Alloc;
  address: number;
  range: string;
  /** How the kernel got the address: the kernel argument, or the next field of the previous node. */
  via: 'argument' | 'previous node'; // i18n-ignore: union members compared by the chapter, which shows its own text
  verdict: HopVerdict;
  /** The node's value if the read was valid, else null. */
  value: number | null;
  /** Sum of the values read so far, this one included. */
  sum: number;
  /** The derefmap cell that decided the hop. */
  rule: Cell;
}

/** The walk, hop by hop, ending after the last node or at the first invalid hop (which is included). */
export function chase(nodes: Alloc[], setting: Setting, values: number[] = NODE_VALUES): Hop[] {
  const eraId = SETTINGS.find((s) => s.id === setting)!.era;
  const addr = addresses(nodes);
  const hops: Hop[] = [];
  let sum = 0;
  for (let i = 0; i < nodes.length; i++) {
    const rule = cell('cuda', eraId, ROW[nodes[i]], i === 0 ? 'arg' : 'chased');
    const verdict = hopVerdict(rule);
    const ok = verdict !== 'invalid';
    if (ok) sum += values[i];
    hops.push({
      index: i,
      alloc: nodes[i],
      address: addr[i],
      range: RANGES[nodes[i]].name,
      via: i === 0 ? 'argument' : 'previous node', // i18n-ignore: union member, not shown as is
      verdict,
      value: ok ? values[i] : null,
      sum,
      rule,
    });
    if (!ok) break;
  }
  return hops;
}

/** How many node values the kernel reads correctly before its first invalid dereference. */
export function goodBeforeFailure(nodes: Alloc[], setting: Setting): number {
  return chase(nodes, setting).filter((h) => h.verdict !== 'invalid').length;
}

/** The index of the first invalid hop, or null if the walk completes. */
export function firstFailure(nodes: Alloc[], setting: Setting): number | null {
  const h = chase(nodes, setting).find((x) => x.verdict === 'invalid');
  return h ? h.index : null;
}
