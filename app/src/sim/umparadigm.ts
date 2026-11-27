// Which Unified Memory paradigm a CUDA (or HIP) device offers, from three device
// attributes. Appendix C, §C.7 (book pp. 627–628) and the labs page.
//
// Not in the book. The table is NVIDIA's (CUDA Programming Guide 2.6.2.1, quoted in
// notes/today/appC-facts.md A14):
//
//   | Paradigm                                   | Attributes                                      |
//   | Limited (Windows, WSL, some Tegra)         | ConcurrentManagedAccess = 0                     |
//   | Full, managed allocations only             | ConcurrentManagedAccess = 1, PageableMemoryAccess = 0 |
//   | Full, all memory, software coherence (HMM) | both 1, PageableMemoryAccessUsesHostPageTables = 0 |
//   | Full, all memory, hardware coherence (ATS) | all three 1                                     |
//
// The capability checklist comes from A10 (the limited paradigm's rules), A11 (full
// support: migration on use, oversubscription, page or cache-line granularity) and
// A12 (system memory needs PageableMemoryAccess; cudaMalloc memory is never
// host-accessible, even with hardware coherence). HIP has attributes with the same
// names and meanings (B5); Metal has none of them (C5).
//
// Two combinations are not rows of NVIDIA's table: UsesHostPageTables = 1 with
// PageableMemoryAccess = 0, and PageableMemoryAccess = 1 with ConcurrentManagedAccess
// = 0. The model reports them as unlisted and does not guess what they mean.
//
// No imports and only erasable TypeScript, so browser checks can load this file
// directly with Node.
import { tr } from '../lib/locale.ts';

export interface Attrs {
  /** cudaDevAttrConcurrentManagedAccess */
  concurrentManagedAccess: boolean;
  /** cudaDevAttrPageableMemoryAccess */
  pageableMemoryAccess: boolean;
  /** cudaDevAttrPageableMemoryAccessUsesHostPageTables */
  usesHostPageTables: boolean;
}

export type ParadigmKey = 'limited' | 'managed' | 'hmm' | 'ats';

export interface Row {
  key: ParadigmKey;
  name: string;
  /** The attribute condition, as NVIDIA's table writes it. */
  condition: string;
  /** Where NVIDIA says you find it. */
  where: string;
  /** Facts items. */
  source: string;
}

export const ROWS: Row[] = [
  {
    key: 'limited',
    get name() { return tr('Limited', 'Hạn chế'); },
    condition: 'ConcurrentManagedAccess = 0',
    get where() { return tr('Windows, including WSL, and some Tegra systems; devices of compute capability below 6.0', 'Windows, kể cả WSL, và một số hệ thống Tegra; các thiết bị có compute capability dưới 6.0'); },
    source: 'A10, A14',
  },
  {
    key: 'managed',
    get name() { return tr('Full, managed allocations only', 'Đầy đủ, chỉ vùng cấp phát managed'); },
    condition: 'ConcurrentManagedAccess = 1, PageableMemoryAccess = 0',
    get where() { return tr('compute capability 6.0 or later without pageable memory access, for example Linux without HMM', 'compute capability 6.0 trở lên nhưng không truy cập được bộ nhớ phân trang (pageable), ví dụ Linux không có HMM'); },
    source: 'A12, A14',
  },
  {
    key: 'hmm',
    get name() { return tr('Full, all memory, software coherence (HMM)', 'Đầy đủ, mọi bộ nhớ, nhất quán bằng phần mềm (HMM)'); },
    get condition() { return tr('both 1, PageableMemoryAccessUsesHostPageTables = 0', 'cả hai bằng 1, PageableMemoryAccessUsesHostPageTables = 0'); },
    get where() { return tr('Linux 6.1.24+, 6.2.11+ or 6.3+, compute capability 7.5 or later, driver 535+ with the open kernel modules', 'Linux 6.1.24+, 6.2.11+ hoặc 6.3+, compute capability 7.5 trở lên, driver 535+ với các kernel module mã nguồn mở'); },
    source: 'A12, A14',
  },
  {
    key: 'ats',
    get name() { return tr('Full, all memory, hardware coherence (ATS)', 'Đầy đủ, mọi bộ nhớ, nhất quán bằng phần cứng (ATS)'); },
    get condition() { return tr('all three 1', 'cả ba bằng 1'); },
    get where() { return tr('Grace Hopper and Grace Blackwell (NVLink chip-to-chip)', 'Grace Hopper và Grace Blackwell (NVLink chip-to-chip)'); },
    source: 'A12, A14',
  },
];

export interface Paradigm {
  row: Row;
  /** False for a combination NVIDIA's table does not list. */
  listed: boolean;
  /** Why it is unlisted, when it is. */
  note?: string;
}

export function paradigm(a: Attrs): Paradigm {
  const row = (k: ParadigmKey) => ROWS.find((r) => r.key === k) as Row;
  if (!a.concurrentManagedAccess) {
    if (a.pageableMemoryAccess || a.usesHostPageTables)
      return {
        row: row('limited'),
        listed: false,
        note: tr('NVIDIA lists the limited paradigm by ConcurrentManagedAccess = 0 alone and does not say what the other two attributes mean alongside it.', 'NVIDIA xác định mô hình hạn chế chỉ bằng ConcurrentManagedAccess = 0 và không nói hai thuộc tính còn lại có nghĩa gì khi đi kèm.'),
      };
    return { row: row('limited'), listed: true };
  }
  if (!a.pageableMemoryAccess) {
    if (a.usesHostPageTables)
      return {
        row: row('managed'),
        listed: false,
        note: tr('UsesHostPageTables = 1 with PageableMemoryAccess = 0 is not a row of NVIDIA\'s table. Read as managed only, since system memory needs PageableMemoryAccess.', 'UsesHostPageTables = 1 cùng PageableMemoryAccess = 0 không phải một hàng trong bảng của NVIDIA. Hiểu là chỉ managed, vì bộ nhớ hệ thống cần PageableMemoryAccess.'),
      };
    return { row: row('managed'), listed: true };
  }
  return { row: row(a.usesHostPageTables ? 'ats' : 'hmm'), listed: true };
}

export type Answer = 'yes' | 'no' | 'not stated';

export interface Check {
  key: 'malloc' | 'concurrent' | 'oversubscribe' | 'fault' | 'cudamalloc-host';
  what: string;
  answer: Answer;
  /** One line of why, with the facts item. */
  why: string;
}

/** What works on a machine with these attributes. */
export function checklist(a: Attrs): Check[] {
  const full = a.concurrentManagedAccess;
  const malloc: Answer = full ? (a.pageableMemoryAccess ? 'yes' : 'no') : a.pageableMemoryAccess ? 'not stated' : 'no'; // i18n-ignore: identifier also shown to the reader
  return [
    {
      key: 'malloc',
      what: tr('A kernel uses a pointer from malloc, new or mmap', 'Một kernel dùng con trỏ từ malloc, new hoặc mmap'),
      answer: malloc,
      why:
        malloc === 'yes'
          ? tr('PageableMemoryAccess = 1: all system memory behaves as unified memory, file-backed mmap included (A12).', 'PageableMemoryAccess = 1: mọi bộ nhớ hệ thống hoạt động như unified memory, kể cả mmap gắn với tệp (A12).')
          : malloc === 'no'
            ? tr('PageableMemoryAccess = 0: only memory allocated as managed behaves as unified memory (A12).', 'PageableMemoryAccess = 0: chỉ bộ nhớ cấp phát dạng managed mới hoạt động như unified memory (A12).')
            : tr('An unlisted combination; NVIDIA does not say (A14).', 'Một tổ hợp không có trong bảng; NVIDIA không nói (A14).'),
    },
    {
      key: 'concurrent',
      what: tr('The CPU touches managed memory while a kernel runs', 'CPU chạm vào bộ nhớ managed khi một kernel đang chạy'),
      answer: full ? 'yes' : 'no',
      why: full
        ? tr('ConcurrentManagedAccess = 1: coherent concurrent access (A11, A14).', 'ConcurrentManagedAccess = 1: truy cập đồng thời và nhất quán (A11, A14).')
        : tr('Not while the GPU is active; concurrent access can end in a segmentation fault (A10).', 'Không được khi GPU đang hoạt động; truy cập đồng thời có thể kết thúc bằng lỗi segmentation fault (A10).'),
    },
    {
      key: 'oversubscribe',
      what: tr('Managed allocations larger than GPU memory', 'Vùng cấp phát managed lớn hơn bộ nhớ GPU'),
      answer: full ? 'yes' : 'no',
      why: full ? tr('Oversubscription is allowed (A11).', 'Được phép cấp phát vượt dung lượng (oversubscription) (A11).') : tr('No more managed memory than the GPU physically has (A10).', 'Không được nhiều bộ nhớ managed hơn lượng GPU thực sự có (A10).'),
    },
    {
      key: 'fault',
      what: tr('Managed pages move on demand, when the GPU touches them', 'Các trang managed di chuyển theo nhu cầu, khi GPU chạm vào chúng'),
      answer: full ? 'yes' : 'no',
      why: full
        ? tr('Migrated when used by another processor, by fault and migrate by default (A11, A13).', 'Được di chuyển khi một bộ xử lý khác dùng tới, mặc định theo cơ chế lỗi trang rồi di chuyển (fault and migrate) (A11, A13).')
        : tr('All managed memory generally moves to the GPU at each launch, and back at synchronisation (A10).', 'Toàn bộ bộ nhớ managed thường được chuyển sang GPU ở mỗi lần khởi chạy, và chuyển về khi đồng bộ hóa (A10).'),
    },
    {
      key: 'cudamalloc-host',
      what: tr('Host code dereferences a cudaMalloc pointer', 'Mã host giải tham chiếu một con trỏ cudaMalloc'),
      answer: 'no',
      why: tr('Never, on any paradigm: hardware coherence does not open GPU-only allocations to the host (A12).', 'Không bao giờ, trên mọi mô hình: nhất quán bằng phần cứng không mở các vùng cấp phát chỉ dành cho GPU cho host (A12).'),
    },
  ];
}

/** The unit in which managed data moves or stays coherent (A10, A11). */
export function granularityText(a: Attrs): string {
  const k = paradigm(a).row.key;
  return k === 'ats'
    ? tr('cache lines: hardware coherence (A11)', 'dòng cache: nhất quán bằng phần cứng (A11)')
    : k === 'limited'
      ? tr('all managed memory, moved in bulk at launch and synchronisation (A10)', 'toàn bộ bộ nhớ managed, chuyển một lần cả khối khi khởi chạy và khi đồng bộ hóa (A10)')
      : tr('pages: software coherence (A11)', 'trang: nhất quán bằng phần mềm (A11)');
}

// ---------------------------------------------------------------- lab C.1's 'paradigm' bench row

/** The minimal shape of a backend's run in reports/runs/*.json (see app/src/lib/reports.ts). */
export interface RunLike {
  results: { lab: string; variant: string; bench: Record<string, string | number>[] }[];
}

const norm = (k: string) => k.toLowerCase().replace(/[^a-z]/g, '');
const truthy = (v: string | number | undefined): boolean | undefined => {
  if (v == null) return undefined;
  if (typeof v === 'number') return isFinite(v) ? v !== 0 : undefined;
  const s = v.trim().toLowerCase();
  if (s === '1' || s === 'true' || s === 'yes') return true;
  if (s === '0' || s === 'false' || s === 'no') return false;
  return undefined;
};

/**
 * Reads the three attributes from a bench row. Field names are matched loosely: lab
 * C.1 (labs/appC/lab1_feeding/main.cpp) writes concurrent_managed, pageable_access and
 * uses_host_page_tables, and the full attribute names (concurrentManagedAccess,
 * cudaDevAttrConcurrentManagedAccess, hipDeviceAttributeConcurrentManagedAccess, ...)
 * work too.
 * Returns null unless all three are present and read as 0 or 1.
 */
export function parseParadigmRow(row: Record<string, string | number>): Attrs | null {
  let cma: boolean | undefined;
  let pma: boolean | undefined;
  let host: boolean | undefined;
  for (const [k, v] of Object.entries(row)) {
    const n = norm(k);
    if (n.includes('useshostpagetables')) host = truthy(v);
    else if (n.includes('concurrentmanaged')) cma = truthy(v);
    else if (n.includes('pageable')) pma = truthy(v);
  }
  if (cma == null || pma == null || host == null) return null;
  return { concurrentManagedAccess: cma, pageableMemoryAccess: pma, usesHostPageTables: host };
}

/** Lab C.1's directory, as the lab runtime names results: "appC/lab1_feeding". */
export const isLabC1 = (lab: string) => /^appC\/lab1(_|$)/.test(lab);

/**
 * The attributes lab C.1 reported on one backend, from its bench row named
 * "paradigm" (solution first, then work), or null when the lab has not run there.
 */
export function paradigmFromRun(run: RunLike | undefined): Attrs | null {
  if (!run) return null;
  const order = ['solution', 'work', 'starter'];
  const rs = run.results
    .filter((r) => isLabC1(r.lab))
    .sort((a, b) => order.indexOf(a.variant) - order.indexOf(b.variant));
  for (const r of rs)
    for (const row of r.bench) {
      if (String(row.name ?? '').toLowerCase() !== 'paradigm') continue;
      const a = parseParadigmRow(row);
      if (a) return a;
    }
  return null;
}
