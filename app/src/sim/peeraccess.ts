// Peer access between GPUs: Appendix C, §C.5 (book p. 626).
//
// The book: with one physical address space across GPUs, a kernel can load from
// another GPU's memory through a pointer passed as an argument, which replaces the
// host-triggered device-to-device copies a halo exchange (chapter 23) needed: extra
// memory for the copies, plus host synchronisation and copy set-up.
//
// What the book leaves out, from NVIDIA's documentation (notes/today/appC-facts.md A8):
//   - Access must be enabled first, from the accessing device, with
//     cudaDeviceEnablePeerAccess(peerDevice, 0), and "access granted by this call is
//     unidirectional".
//   - "On non-NVSwitch enabled systems, each device can support a system-wide maximum
//     of eight peer connections."
//   - "On Linux specifically, CUDA and the display driver do not support IOMMU-enabled
//     bare-metal PCIe peer-to-peer memory transfer."
//   - Without peer access, copies still work: cudaMemcpyPeer, cudaMemcpyPeerAsync.
//
// What the model simplifies, and what the app says wherever it is shown:
//   - Whether two GPUs can be peers at all is taken as given. On a real system
//     cudaDeviceCanAccessPeer answers that, and it depends on the topology.
//   - Without NVSwitch the GPUs are taken to be peers over PCIe, which is what the
//     IOMMU rule covers. With NVSwitch and the IOMMU on, our sources say nothing, so
//     the model grants nothing and says "not documented" instead of "refused".
//   - "Eight peer connections" is modelled as eight outgoing grants per GPU. The guide
//     does not say how connections are counted.
//   - No bandwidth or latency.
//   - The halo-exchange step lists are our reading of the book's comparison. The
//     "ordering between devices" step is ours: the book does not discuss it.
//
// No imports and only erasable TypeScript, so browser checks can load this file
// directly with Node.
import { tr } from '../lib/locale.ts';

export const MAX_PEERS_WITHOUT_NVSWITCH = 8;

export interface PeerSystem {
  /** Number of GPUs, 2 to 16 in the explorer. */
  n: number;
  nvswitch: boolean;
  /** Bare-metal Linux with the IOMMU on. */
  iommu: boolean;
}

/** grants[i][j]: GPU i may load from GPU j's memory. */
export type Grants = boolean[][];

export function emptyGrants(n: number): Grants {
  return Array.from({ length: n }, () => Array.from({ length: n }, () => false));
}

export function outgoing(g: Grants, i: number): number {
  return g[i].reduce((s, on, j) => s + (on && j !== i ? 1 : 0), 0);
}

export type Refusal = 'self' | 'iommu' | 'limit' | 'undocumented';

export type EnableResult =
  | { ok: true; call: string }
  | { ok: false; reason: Refusal; call: string; text: string };

/** The call a program would make with GPU `from` as the current device, to reach GPU `to`. */
export const enableCall = (from: number, to: number) => tr(`on GPU ${from}: cudaDeviceEnablePeerAccess(${to}, 0)`, `trên GPU ${from}: cudaDeviceEnablePeerAccess(${to}, 0)`);

/** Whether GPU `from` may be granted access to GPU `to`'s memory now. */
export function canEnable(sys: PeerSystem, g: Grants, from: number, to: number): EnableResult {
  const call = enableCall(from, to);
  if (from === to)
    return { ok: false, reason: 'self', call, text: tr(`GPU ${from}'s own memory needs no grant.`, `Bộ nhớ của chính GPU ${from} không cần cấp quyền.`) }; // i18n-ignore: reason is an identifier, not shown
  if (sys.iommu && !sys.nvswitch)
    return {
      ok: false,
      reason: 'iommu', // i18n-ignore: identifier compared by the page, not shown
      call,
      text: tr('Refused: on bare-metal Linux with the IOMMU on, CUDA does not support PCIe peer-to-peer (A8).', 'Bị từ chối: trên Linux chạy trực tiếp trên phần cứng (bare-metal) khi bật IOMMU, CUDA không hỗ trợ peer-to-peer qua PCIe (A8).'),
    };
  if (sys.iommu)
    return {
      ok: false,
      reason: 'undocumented', // i18n-ignore: identifier compared by the page, not shown
      call,
      text: tr("Not documented: NVIDIA's IOMMU rule is about PCIe peer-to-peer, and our sources say nothing about NVSwitch with the IOMMU on (A8). The model grants nothing it cannot source.", 'Không có tài liệu: quy tắc IOMMU của NVIDIA nói về peer-to-peer qua PCIe, và các nguồn của chúng tôi không nói gì về NVSwitch khi bật IOMMU (A8). Mô hình không cấp quyền nào mà nó không có nguồn dẫn chứng.'),
    };
  if (!sys.nvswitch && !g[from][to] && outgoing(g, from) >= MAX_PEERS_WITHOUT_NVSWITCH)
    return {
      ok: false,
      reason: 'limit', // i18n-ignore: identifier compared by the page, not shown
      call,
      text: tr(`Refused: GPU ${from} already has ${MAX_PEERS_WITHOUT_NVSWITCH} peers, the most a device supports without NVSwitch (A8).`, `Bị từ chối: GPU ${from} đã có ${MAX_PEERS_WITHOUT_NVSWITCH} peer, mức tối đa một thiết bị hỗ trợ khi không có NVSwitch (A8).`),
    };
  return { ok: true, call };
}

/** Grants `from` access to `to` if allowed; returns the new grants and the outcome. */
export function enable(sys: PeerSystem, g: Grants, from: number, to: number): { grants: Grants; result: EnableResult } {
  const result = canEnable(sys, g, from, to);
  if (!result.ok) return { grants: g, result };
  const next = g.map((row) => row.slice());
  next[from][to] = true;
  return { grants: next, result };
}

/** Takes a grant back (in the model). */
export function revoke(g: Grants, from: number, to: number): Grants {
  const next = g.map((row) => row.slice());
  next[from][to] = false;
  return next;
}

export interface Deref {
  ok: boolean;
  how: 'local' | 'peer' | 'refused';
  text: string;
}

/** A kernel on GPU `gpu` loads through a pointer to GPU `owner`'s memory. */
export function deref(g: Grants, gpu: number, owner: number): Deref {
  if (gpu === owner) return { ok: true, how: 'local', text: tr(`A load from GPU ${gpu}'s own memory.`, `Một lần nạp từ bộ nhớ của chính GPU ${gpu}.`) };
  if (g[gpu][owner])
    return {
      ok: true,
      how: 'peer',
      text: tr(`A direct load over the fabric: GPU ${gpu} was granted access to GPU ${owner}. The pointer has the same value on both GPUs (A8).`, `Một lần nạp trực tiếp qua kết nối (fabric): GPU ${gpu} đã được cấp quyền truy cập GPU ${owner}. Con trỏ có cùng giá trị trên cả hai GPU (A8).`),
    };
  return {
    ok: false,
    how: 'refused',
    text: tr(`An error: GPU ${gpu} has no access to GPU ${owner}'s memory. A copy still works without it: cudaMemcpyPeer (A8).`, `Lỗi: GPU ${gpu} không có quyền truy cập bộ nhớ của GPU ${owner}. Sao chép vẫn hoạt động khi không có quyền này: cudaMemcpyPeer (A8).`),
  };
}

/**
 * Whether every GPU could load from every other: n − 1 peers each, within the limit.
 * 'unknown' where our sources say nothing (NVSwitch with the IOMMU on).
 */
export function allToAll(sys: PeerSystem): 'possible' | 'impossible' | 'unknown' {
  if (sys.iommu) return sys.nvswitch ? 'unknown' : 'impossible';
  return sys.nvswitch || sys.n - 1 <= MAX_PEERS_WITHOUT_NVSWITCH ? 'possible' : 'impossible';
}

/** True only where all to all is documented to work. */
export function allToAllPossible(sys: PeerSystem): boolean {
  return allToAll(sys) === 'possible';
}

/**
 * Enables as many grants as the rules allow, in row order; reports how many were
 * refused and how many were left out because the rule is not documented.
 */
export function enableAll(sys: PeerSystem): { grants: Grants; granted: number; refused: number; unknown: number } {
  let g = emptyGrants(sys.n);
  let granted = 0;
  let refused = 0;
  let unknown = 0;
  for (let i = 0; i < sys.n; i++)
    for (let j = 0; j < sys.n; j++) {
      if (i === j) continue;
      const r = enable(sys, g, i, j);
      g = r.grants;
      if (r.result.ok) granted++;
      else if (r.result.reason === 'undocumented') unknown++;
      else refused++;
    }
  return { grants: g, granted, refused, unknown };
}

// ---------------------------------------------------------------- one halo exchange

export type Who = 'host' | 'gpu0' | 'gpu1' | 'both';
export type StepKind = 'kernel' | 'wait' | 'copy' | 'order' | 'load';

export interface HaloStep {
  who: Who;
  kind: StepKind;
  text: string;
}

export interface HaloExchange {
  approach: 'copy' | 'peer';
  steps: HaloStep[];
  /** Extra buffers that hold copied halo data. */
  staging: number;
  /** Copies the host sets up. */
  hostCopies: number;
  /** Times the host waits for a GPU. */
  hostWaits: number;
  /** Points where one GPU must wait for the other's step to finish. */
  ordering: number;
  /** Peer loads need GPU 0 → 1 and GPU 1 → 0 enabled; false if a grant is missing. */
  ready: boolean;
  /** The grants that are missing, as [from, to]. */
  missing: [number, number][];
}

/**
 * The steps of one halo exchange between GPU 0 and GPU 1, between stencil step k and
 * step k + 1, done with host-driven copies or with peer loads. For peer loads, pass
 * the current grants to see whether the exchange can run.
 */
export function haloExchange(approach: 'copy' | 'peer', g?: Grants): HaloExchange {
  if (approach === 'copy') {
    const steps: HaloStep[] = [
      { who: 'both', kind: 'kernel', text: tr('Each GPU runs step k on its half and writes its boundary rows.', 'Mỗi GPU chạy bước k trên nửa của mình và ghi các hàng biên của nó.') },
      { who: 'host', kind: 'wait', text: tr('The host waits for GPU 0 to finish.', 'Host chờ GPU 0 xong.') },
      { who: 'host', kind: 'wait', text: tr('The host waits for GPU 1 to finish.', 'Host chờ GPU 1 xong.') },
      { who: 'host', kind: 'copy', text: tr("The host sets up a copy of GPU 0's boundary into a halo buffer on GPU 1.", 'Host thiết lập một lần sao chép biên của GPU 0 vào một buffer vùng biên (halo) trên GPU 1.') },
      { who: 'host', kind: 'copy', text: tr("The host sets up a copy of GPU 1's boundary into a halo buffer on GPU 0.", 'Host thiết lập một lần sao chép biên của GPU 1 vào một buffer vùng biên (halo) trên GPU 0.') },
      { who: 'both', kind: 'kernel', text: tr('Each GPU runs step k + 1, reading its neighbour\'s rows from its own halo buffer.', 'Mỗi GPU chạy bước k + 1, đọc các hàng của GPU láng giềng từ buffer vùng biên của chính nó.') },
    ];
    return { approach, steps, staging: 2, hostCopies: 2, hostWaits: 2, ordering: 1, ready: true, missing: [] };
  }
  const missing: [number, number][] = [];
  if (g) {
    if (!g[0]?.[1]) missing.push([0, 1]);
    if (!g[1]?.[0]) missing.push([1, 0]);
  }
  const steps: HaloStep[] = [
    { who: 'both', kind: 'kernel', text: tr('Each GPU runs step k on its half and writes its boundary rows.', 'Mỗi GPU chạy bước k trên nửa của mình và ghi các hàng biên của nó.') },
    { who: 'both', kind: 'order', text: tr('Both GPUs finish step k before either starts step k + 1.', 'Cả hai GPU xong bước k trước khi một trong hai bắt đầu bước k + 1.') },
    { who: 'gpu0', kind: 'load', text: tr("GPU 0 runs step k + 1 and loads GPU 1's boundary rows through the peer pointer.", 'GPU 0 chạy bước k + 1 và nạp các hàng biên của GPU 1 qua con trỏ peer.') },
    { who: 'gpu1', kind: 'load', text: tr("GPU 1 runs step k + 1 and loads GPU 0's boundary rows through the peer pointer.", 'GPU 1 chạy bước k + 1 và nạp các hàng biên của GPU 0 qua con trỏ peer.') },
  ];
  return { approach, steps, staging: 0, hostCopies: 0, hostWaits: 0, ordering: 1, ready: missing.length === 0, missing };
}
