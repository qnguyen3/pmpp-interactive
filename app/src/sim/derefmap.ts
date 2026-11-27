// Appendix C as data: who may dereference which pointer, era by era.
//
// The book tells the history of CUDA's memory model as a sequence of steps
// (pp. 623–629). This file turns it into a table: for each era, each kind of
// pointer (a row) and each place that might dereference it (a site), one verdict
// class, one sentence on cost, and the source of the rule. The book's own
// assembled table is in notes/appC.md ("Who can dereference what"); the rules
// for today, for HIP and for Metal come from notes/today/appC-facts.md (items
// A1 to D5 and its table "The notes' dereference table, brought up to date").
// A source written as "A12" means that facts file; "ch02-facts A17" means
// notes/today/ch02-facts.md, item A17; "p. 625" means the book.
//
// What the model simplifies, and what the page says wherever it is shown:
//   - One cell is one documented rule. Real systems add conditions (driver
//     version, operating system, device attributes) that the cell's sentence
//     names only when they decide the answer.
//   - Costs are classes, not numbers: local, over the interconnect, pages moving
//     on first touch, pages moved in bulk at launch.
//   - The eras follow the book's order of sections, which is not always the
//     order of the dates (CUDA 2.2 shipped before Fermi products did, A2, A3).
//   - "Windows, WSL and some Tegra systems today" behave like the CUDA 6 era on
//     Kepler and Maxwell (A10); the era's label says so instead of repeating it.
//   - A row the book never discusses (registered memory before unified
//     addressing) is "not documented" rather than a guess.
//
// Its one runtime import is the translation helper `tr`.

import { tr } from '../lib/locale.ts';

export type EraId = 'copies' | 'fermi' | 'zerocopy' | 'uva' | 'kepler' | 'um-limited' | 'pascal' | 'vmm' | 'hmm';
export type Platform = 'cuda' | 'hip' | 'metal';
export type RowId = 'malloc' | 'pinned' | 'registered' | 'device' | 'peer' | 'managed' | 'shared' | 'vmm';
export type SiteId = 'host' | 'arg' | 'chased';
export type Verdict =
  | 'local'
  | 'remote'
  | 'fault'
  | 'bulk'
  | 'other-ptr'
  | 'not-while-running'
  | 'after-enable'
  | 'no'
  | 'not-yet'
  | 'unknown';

export interface Cell {
  verdict: Verdict;
  /** One sentence: what it costs or why not. */
  cost: string;
  /** Book pages and/or facts items, in the notation described above. */
  source: string;
}

export interface Era {
  id: EraId;
  /** Label on the timeline strip: the book's section, or where the era comes from. */
  strip: string;
  title: string;
  /** The release or architecture that brought it. */
  vehicle: string;
  /** The date as the book gives it, or null when the era is not in the book. */
  book: string | null;
  /** The date as our sources give it, with the facts item, when it differs or the book gives none. */
  corrected: { date: string; item: string } | null;
  /** Book pages, or null for the era beyond the book. */
  pages: string | null;
  /** What this step changes, in one sentence. */
  adds: string;
  /** A remark that belongs to this era (an ordering caveat, the systems that still behave this way). */
  note?: string;
}

export const ERAS: Era[] = [
  {
    id: 'copies',
    strip: 'p. 623',
    get title() {
      return tr('Copies only', 'Chỉ có sao chép');
    },
    get vehicle() {
      return tr('early GPU generations, and chapter 2', 'các thế hệ GPU đầu tiên, và chương 2');
    },
    get book() {
      return tr('early generations', 'các thế hệ đầu');
    },
    corrected: null,
    pages: '623',
    get adds() {
      return tr('Two physical memories. Data goes in with cudaMemcpy before a kernel and comes out after it.', 'Hai bộ nhớ vật lý. Dữ liệu được đưa vào bằng cudaMemcpy trước kernel và lấy ra sau kernel.');
    },
  },
  {
    id: 'fermi',
    strip: 'C.1',
    get title() {
      return tr('One address space on the GPU', 'Một không gian địa chỉ trên GPU');
    },
    get vehicle() {
      return tr('Fermi', 'Fermi');
    },
    book: '2009',
    corrected: {
      get date() {
        return tr('whitepaper 2009; products reached the market in March 2010', 'sách trắng (whitepaper) 2009; sản phẩm ra thị trường vào tháng 3 năm 2010');
      },
      item: 'A2',
    },
    pages: '624',
    get adds() {
      return tr('Shared, local and global memory share one address space, so one pointer type serves all three.', 'Bộ nhớ chia sẻ, bộ nhớ cục bộ và bộ nhớ toàn cục dùng chung một không gian địa chỉ, nên một kiểu con trỏ phục vụ được cả ba.');
    },
  },
  {
    id: 'zerocopy',
    strip: 'C.2',
    get title() {
      return tr('Zero-copy host memory', 'Bộ nhớ host zero-copy');
    },
    get vehicle() {
      return tr('CUDA 2.2', 'CUDA 2.2');
    },
    book: '2009',
    corrected: {
      get date() {
        return tr('May 2009, as the book says', 'tháng 5 năm 2009, như sách nói');
      },
      item: 'A3',
    },
    pages: '624–625',
    get adds() {
      return tr('A kernel reads pinned host memory in place, over the interconnect, through a second pointer value.', 'Kernel đọc bộ nhớ host được ghim tại chỗ, qua đường kết nối (interconnect), thông qua một giá trị con trỏ thứ hai.');
    },
    get note() {
      return tr('The map follows the book\'s order. By the dates, CUDA 2.2 (May 2009) came before Fermi products (March 2010).', 'Bản đồ theo thứ tự của sách. Xét theo ngày tháng, CUDA 2.2 (tháng 5 năm 2009) ra trước các sản phẩm Fermi (tháng 3 năm 2010).');
    },
  },
  {
    id: 'uva',
    strip: 'C.3',
    get title() {
      return tr('Unified virtual address space', 'Không gian địa chỉ ảo hợp nhất');
    },
    get vehicle() {
      return tr('CUDA 4', 'CUDA 4');
    },
    book: '2011',
    corrected: {
      get date() {
        return tr('May 2011, as the book says', 'tháng 5 năm 2011, như sách nói');
      },
      item: 'A5',
    },
    pages: '625',
    get adds() {
      return tr('One virtual address space for host and devices: a pointer\'s value says where it lives. Who may dereference it barely changes.', 'Một không gian địa chỉ ảo cho host và các device: giá trị của con trỏ cho biết nó nằm ở đâu. Ai được giải tham chiếu nó thì hầu như không đổi.');
    },
  },
  {
    id: 'kepler',
    strip: 'C.4–C.5',
    get title() {
      return tr('Large addresses and peer GPUs', 'Địa chỉ lớn và GPU ngang hàng');
    },
    get vehicle() {
      return tr('Kepler', 'Kepler');
    },
    book: '2013',
    corrected: {
      get date() {
        return tr('March 2012; and Fermi already had 64-bit addressing and more than 4 GB', 'tháng 3 năm 2012; và Fermi đã có địa chỉ 64-bit và hơn 4 GB');
      },
      item: 'A6',
    },
    pages: '625–626',
    get adds() {
      return tr('Addresses wide enough for more than 4 GB, and one GPU dereferencing another GPU\'s memory.', 'Địa chỉ đủ rộng cho hơn 4 GB, và một GPU giải tham chiếu bộ nhớ của GPU khác.');
    },
  },
  {
    id: 'um-limited',
    strip: 'C.6',
    get title() {
      return tr('Unified Memory, limited', 'Unified Memory, có giới hạn');
    },
    get vehicle() {
      return tr('CUDA 6 on Kepler and Maxwell', 'CUDA 6 trên Kepler và Maxwell');
    },
    book: '2013',
    corrected: {
      get date() {
        return tr('announced November 2013, shipped April 2014', 'công bố tháng 11 năm 2013, phát hành tháng 4 năm 2014');
      },
      item: 'A9',
    },
    pages: '626–627',
    get adds() {
      return tr('Managed memory: one pointer for CPU and GPU, moved to the GPU in bulk at every launch.', 'Bộ nhớ managed: một con trỏ cho CPU và GPU, được chuyển sang GPU cả khối ở mỗi lần khởi chạy.');
    },
    get note() {
      return tr('Windows, WSL and some Tegra systems still behave exactly like this today (A10).', 'Windows, WSL và một số hệ thống Tegra ngày nay vẫn hoạt động đúng như thế này (A10).');
    },
  },
  {
    id: 'pascal',
    strip: 'C.7–C.8',
    get title() {
      return tr('GPU page faults', 'Lỗi trang trên GPU');
    },
    get vehicle() {
      return tr('Pascal and CUDA 8', 'Pascal và CUDA 8');
    },
    book: '2016',
    corrected: {
      get date() {
        return tr('CUDA 8 shipped in September 2016; the book gives 2016 for Pascal', 'CUDA 8 phát hành tháng 9 năm 2016; sách ghi 2016 cho Pascal');
      },
      item: 'A11',
    },
    pages: '627–629',
    get adds() {
      return tr('49-bit addresses and GPU page faults: managed pages move when first touched, not at launch.', 'Địa chỉ 49-bit và lỗi trang (page fault) trên GPU: các trang managed di chuyển khi được chạm lần đầu, không phải lúc khởi chạy.');
    },
    get note() {
      return tr('This is Linux with a GPU of compute capability 6.0 or later and no HMM (A12, A14).', 'Đây là Linux với GPU có compute capability 6.0 trở lên và không có HMM (A12, A14).');
    },
  },
  {
    id: 'vmm',
    strip: 'C.9',
    get title() {
      return tr('Address space control', 'Điều khiển không gian địa chỉ');
    },
    get vehicle() {
      return tr('the virtual memory management calls', 'các lời gọi quản lý bộ nhớ ảo');
    },
    get book() {
      return tr('CUDA 11, no year', 'CUDA 11, không ghi năm');
    },
    corrected: {
      get date() {
        return tr('CUDA 10.2, November 2019', 'CUDA 10.2, tháng 11 năm 2019');
      },
      item: 'A15',
    },
    pages: '629',
    get adds() {
      return tr('Reserve a range of addresses, create memory, map it into the range, and grant access to it.', 'Đặt trước một vùng địa chỉ, tạo bộ nhớ, ánh xạ nó vào vùng đó, và cấp quyền truy cập tới nó.');
    },
  },
  {
    id: 'hmm',
    strip: 'beyond the book', // i18n-ignore: identifier also shown to the reader
    get title() {
      return tr('All host memory', 'Toàn bộ bộ nhớ host');
    },
    get vehicle() {
      return tr('HMM on Linux, or ATS on Grace Hopper and Grace Blackwell', 'HMM trên Linux, hoặc ATS trên Grace Hopper và Grace Blackwell');
    },
    book: null,
    corrected: {
      get date() {
        return tr('NVIDIA\'s HMM post is from August 2023', 'bài viết về HMM của NVIDIA có từ tháng 8 năm 2023');
      },
      item: 'A12',
    },
    pages: null,
    get adds() {
      return tr('Plain malloc memory works in kernels through the host\'s own pointer.', 'Bộ nhớ malloc thường dùng được trong kernel qua chính con trỏ của host.');
    },
    get note() {
      return tr('HMM needs Linux 6.1.24 or later, compute capability 7.5 or later and driver 535 or later with the open kernel modules (A12).', 'HMM cần Linux 6.1.24 trở lên, compute capability 7.5 trở lên và driver 535 trở lên với các module kernel mã nguồn mở (A12).');
    },
  },
];

export const ERA_ORDER: EraId[] = ERAS.map((e) => e.id);
export const era = (id: EraId): Era => ERAS.find((e) => e.id === id)!;

export const ROWS: RowId[] = ['malloc', 'pinned', 'registered', 'device', 'peer', 'managed', 'shared', 'vmm'];
export const SITES: SiteId[] = ['host', 'arg', 'chased'];

export const ROW_LABEL: Record<Platform, Record<RowId, string>> = {
  cuda: {
    get malloc() {
      return tr('malloc\'d host memory', 'bộ nhớ host cấp phát bằng malloc');
    },
    get pinned() {
      return tr('pinned host memory (cudaHostAlloc, cudaMallocHost)', 'bộ nhớ host được ghim (cudaHostAlloc, cudaMallocHost)');
    },
    get registered() {
      return tr('registered host memory (cudaHostRegister)', 'bộ nhớ host đã đăng ký (cudaHostRegister)');
    },
    get device() {
      return tr('cudaMalloc on this GPU', 'cudaMalloc trên GPU này');
    },
    get peer() {
      return tr('cudaMalloc on a peer GPU', 'cudaMalloc trên một GPU ngang hàng (peer)');
    },
    get managed() {
      return tr('managed memory (cudaMallocManaged)', 'bộ nhớ managed (cudaMallocManaged)');
    },
    get shared() {
      return tr('a __shared__ variable', 'một biến __shared__');
    },
    get vmm() {
      return tr('a VMM-mapped range (cuMemMap)', 'một vùng địa chỉ ánh xạ bằng VMM (cuMemMap)');
    },
  },
  hip: {
    get malloc() {
      return tr('malloc\'d host memory', 'bộ nhớ host cấp phát bằng malloc');
    },
    get pinned() {
      return tr('pinned host memory (hipHostMalloc)', 'bộ nhớ host được ghim (hipHostMalloc)');
    },
    get registered() {
      return tr('registered host memory (hipHostRegister)', 'bộ nhớ host đã đăng ký (hipHostRegister)');
    },
    get device() {
      return tr('hipMalloc on this GPU', 'hipMalloc trên GPU này');
    },
    get peer() {
      return tr('hipMalloc on a peer GPU', 'hipMalloc trên một GPU ngang hàng (peer)');
    },
    get managed() {
      return tr('managed memory (hipMallocManaged)', 'bộ nhớ managed (hipMallocManaged)');
    },
    get shared() {
      return tr('a __shared__ variable', 'một biến __shared__');
    },
    get vmm() {
      return tr('a VMM-mapped range (hipMemMap)', 'một vùng địa chỉ ánh xạ bằng VMM (hipMemMap)');
    },
  },
  metal: {
    get malloc() {
      return tr('malloc\'d memory', 'bộ nhớ cấp phát bằng malloc');
    },
    get pinned() {
      return tr('a shared buffer (no pinned memory documented for Metal)', 'một buffer shared (tài liệu của Metal không có bộ nhớ ghim)');
    },
    get registered() {
      return tr('your page-aligned memory, wrapped by newBufferWithBytesNoCopy', 'bộ nhớ căn theo trang của bạn, được bọc bằng newBufferWithBytesNoCopy');
    },
    get device() {
      return tr('a private buffer', 'một buffer private');
    },
    get peer() {
      return tr('another GPU\'s buffer', 'buffer của một GPU khác');
    },
    get managed() {
      return tr('managed storage mode (deprecated)', 'chế độ lưu trữ managed (đã bị deprecated)');
    },
    get shared() {
      return tr('a threadgroup variable', 'một biến threadgroup');
    },
    get vmm() {
      return tr('a placement sparse buffer (Metal 4)', 'một placement sparse buffer (Metal 4)');
    },
  },
};

export const SITE_LABEL: Record<Platform, Record<SiteId, string>> = {
  cuda: { get host() { return tr('host code', 'mã host'); }, get arg() { return tr('kernel, given the pointer as an argument', 'kernel, nhận con trỏ làm đối số'); }, get chased() { return tr('kernel, pointer read from memory', 'kernel, con trỏ đọc từ bộ nhớ'); } },
  hip: { get host() { return tr('host code', 'mã host'); }, get arg() { return tr('kernel, given the pointer as an argument', 'kernel, nhận con trỏ làm đối số'); }, get chased() { return tr('kernel, pointer read from memory', 'kernel, con trỏ đọc từ bộ nhớ'); } },
  metal: { get host() { return tr('CPU code, through contents', 'mã CPU, qua contents'); }, get arg() { return tr('kernel, buffer bound as an argument', 'kernel, buffer được gắn làm đối số'); }, get chased() { return tr('kernel, address read from memory', 'kernel, địa chỉ đọc từ bộ nhớ'); } },
};

export const VERDICT_LABEL: Record<Verdict, string> = {
  get local() {
    return tr('yes, local', 'có, cục bộ');
  },
  get remote() {
    return tr('yes, over the interconnect', 'có, qua đường kết nối');
  },
  get fault() {
    return tr('yes, pages move on first touch', 'có, trang di chuyển ở lần chạm đầu');
  },
  get bulk() {
    return tr('yes, moved in bulk at launch', 'có, chuyển cả khối lúc khởi chạy');
  },
  get 'other-ptr'() {
    return tr('only through a different pointer value', 'chỉ qua một giá trị con trỏ khác');
  },
  get 'not-while-running'() {
    return tr('not while a kernel runs', 'không được trong lúc kernel chạy');
  },
  get 'after-enable'() {
    return tr('only after access is enabled', 'chỉ sau khi bật quyền truy cập');
  },
  get no() {
    return tr('no', 'không');
  },
  get 'not-yet'() {
    return tr('does not exist yet', 'chưa tồn tại');
  },
  get unknown() {
    return tr('not documented', 'không có tài liệu');
  },
};

/** The verdicts under which the dereference works, at some cost. */
export const WORKS: Verdict[] = ['local', 'remote', 'fault', 'bulk'];
export const works = (v: Verdict) => WORKS.includes(v);

/** Colour use: success and failure only; slow and conditional verdicts stay neutral. */
export function tone(v: Verdict): 'ok' | 'fail' | 'neutral' | 'absent' {
  if (v === 'local') return 'ok';
  if (v === 'no') return 'fail';
  if (v === 'not-yet') return 'absent';
  return 'neutral';
}

// ---------------------------------------------------------------- CUDA, era by era
type Grid = Record<RowId, Record<SiteId, Cell>>;
type Patch = Partial<Record<RowId, Partial<Record<SiteId, Cell>>>>;
// Cost and source are read when shown, so they follow the reader's language.
const c = (verdict: Verdict, cost: () => string, source: () => string): Cell => ({
  verdict,
  get cost() {
    return cost();
  },
  get source() {
    return source();
  },
});

const COPIES: Grid = {
  malloc: {
    host: c('local', () => tr('Ordinary host memory, read at host-memory speed.', 'Bộ nhớ host thông thường, đọc ở tốc độ của bộ nhớ host.'), () => 'p. 623'),
    arg: c('no', () => tr('Kernels reach only device memory: copy the data in with cudaMemcpy first.', 'Kernel chỉ chạm tới được bộ nhớ device: trước hết hãy sao chép dữ liệu vào bằng cudaMemcpy.'), () => 'p. 623'),
    chased: c('no', () => tr('A host pointer means nothing to a kernel in this model.', 'Trong mô hình này, một con trỏ host không có nghĩa gì với kernel.'), () => 'p. 623'),
  },
  pinned: {
    host: c('local', () => tr('Host memory the operating system may not page out.', 'Bộ nhớ host mà hệ điều hành không được phép đẩy ra khỏi bộ nhớ (page out).'), () => 'ch. 23, pp. 562–563'),
    arg: c('no', () => tr('In the copy model even pinned memory is reached only through copies.', 'Trong mô hình sao chép, ngay cả bộ nhớ ghim (pinned memory) cũng chỉ được chạm tới qua các lần sao chép.'), () => 'p. 623'),
    chased: c('no', () => tr('In the copy model even pinned memory is reached only through copies.', 'Trong mô hình sao chép, ngay cả bộ nhớ ghim (pinned memory) cũng chỉ được chạm tới qua các lần sao chép.'), () => 'p. 623'),
  },
  registered: {
    host: c('local', () => tr('Your own malloc memory, page-locked where it is (not in the book).', 'Bộ nhớ malloc của chính bạn, được khóa trang ngay tại chỗ (không có trong sách).'), () => 'A3'),
    arg: c('no', () => tr('In the copy model kernels reach only device memory.', 'Trong mô hình sao chép, kernel chỉ chạm tới được bộ nhớ device.'), () => 'p. 623'),
    chased: c('no', () => tr('In the copy model kernels reach only device memory.', 'Trong mô hình sao chép, kernel chỉ chạm tới được bộ nhớ device.'), () => 'p. 623'),
  },
  device: {
    host: c('no', () => tr('Host code may only hand the pointer to API calls and kernels; dereferencing it is undefined.', 'Mã host chỉ được trao con trỏ này cho các lời gọi API và kernel; giải tham chiếu nó là hành vi không xác định.'), () => 'ch. 2 §2.4, p. 30; ch02-facts A17'),
    arg: c('local', () => tr('Device memory: the fast path.', 'Bộ nhớ device: con đường nhanh.'), () => 'p. 623'),
    chased: c('local', () => tr('A device pointer stored in device memory can be followed.', 'Một con trỏ device lưu trong bộ nhớ device có thể được lần theo.'), () => 'p. 625'),
  },
  peer: {
    host: c('no', () => tr('Host code cannot dereference any GPU\'s cudaMalloc memory.', 'Mã host không thể giải tham chiếu bộ nhớ cudaMalloc của bất kỳ GPU nào.'), () => 'ch. 2 §2.4, p. 30'),
    arg: c('no', () => tr('Another GPU\'s data arrives only through copies the host starts.', 'Dữ liệu của một GPU khác chỉ đến được qua các lần sao chép do host khởi động.'), () => 'p. 626'),
    chased: c('no', () => tr('Another GPU\'s data arrives only through copies the host starts.', 'Dữ liệu của một GPU khác chỉ đến được qua các lần sao chép do host khởi động.'), () => 'p. 626'),
  },
  managed: {
    host: c('not-yet', () => tr('Managed memory arrives with CUDA 6 (C.6).', 'Bộ nhớ managed xuất hiện cùng CUDA 6 (C.6).'), () => 'p. 626'),
    arg: c('not-yet', () => tr('Managed memory arrives with CUDA 6 (C.6).', 'Bộ nhớ managed xuất hiện cùng CUDA 6 (C.6).'), () => 'p. 626'),
    chased: c('not-yet', () => tr('Managed memory arrives with CUDA 6 (C.6).', 'Bộ nhớ managed xuất hiện cùng CUDA 6 (C.6).'), () => 'p. 626'),
  },
  shared: {
    host: c('no', () => tr('Shared memory exists only on the GPU, one copy per block, while the block runs.', 'Bộ nhớ chia sẻ (shared memory) chỉ tồn tại trên GPU, mỗi khối một bản, trong lúc khối chạy.'), () => 'ch. 5 §5.2, pp. 98–104'),
    arg: c('no', () => tr('A pointer could point only into global memory, so a helper could not take a pointer to shared data.', 'Một con trỏ chỉ có thể trỏ vào bộ nhớ toàn cục (global memory), nên một hàm phụ trợ không thể nhận con trỏ tới dữ liệu chia sẻ.'), () => 'p. 624'),
    chased: c('no', () => tr('A pointer could point only into global memory.', 'Một con trỏ chỉ có thể trỏ vào bộ nhớ toàn cục.'), () => 'p. 624'),
  },
  vmm: {
    host: c('not-yet', () => tr('Reserve-and-map control arrives with C.9.', 'Khả năng điều khiển kiểu đặt trước rồi ánh xạ (reserve-and-map) xuất hiện cùng C.9.'), () => 'p. 629'),
    arg: c('not-yet', () => tr('Reserve-and-map control arrives with C.9.', 'Khả năng điều khiển kiểu đặt trước rồi ánh xạ (reserve-and-map) xuất hiện cùng C.9.'), () => 'p. 629'),
    chased: c('not-yet', () => tr('Reserve-and-map control arrives with C.9.', 'Khả năng điều khiển kiểu đặt trước rồi ánh xạ (reserve-and-map) xuất hiện cùng C.9.'), () => 'p. 629'),
  },
};

const PATCHES: Record<EraId, Patch> = {
  copies: {},
  fermi: {
    shared: {
      arg: c('local', () => tr('One generic pointer: the address falls in the shared window, so the load reads the block\'s shared memory.', 'Một con trỏ tổng quát (generic): địa chỉ rơi vào cửa sổ shared, nên lệnh nạp đọc bộ nhớ chia sẻ của khối.'), () => 'p. 624; A2'),
      chased: c('local', () => tr('A generic pointer read back from memory works too, inside the block that owns the variable.', 'Một con trỏ tổng quát đọc lại từ bộ nhớ cũng dùng được, bên trong khối sở hữu biến đó.'), () => 'A2; ch. 5 §5.2'),
    },
  },
  zerocopy: {
    pinned: {
      host: c('local', () => tr('cudaHostAlloc with cudaHostAllocMapped: pinned host memory, read at host speed.', 'cudaHostAlloc với cudaHostAllocMapped: bộ nhớ host được ghim, đọc ở tốc độ host.'), () => 'p. 624'),
      arg: c('other-ptr', () => tr('The kernel must be given the value from cudaHostGetDevicePointer, and every access crosses the interconnect.', 'Kernel phải được trao giá trị lấy từ cudaHostGetDevicePointer, và mọi lần truy cập đều đi qua đường kết nối (interconnect).'), () => 'pp. 624–625'),
      chased: c('other-ptr', () => tr('Only if the host wrote device-side values into the structure, not its own pointers.', 'Chỉ khi host ghi các giá trị phía device vào cấu trúc, chứ không phải các con trỏ của chính nó.'), () => 'pp. 624–625'),
    },
    registered: {
      arg: c('unknown', () => tr('The book names only cudaHostAlloc; our sources do not say how registered memory behaved in 2009.', 'Sách chỉ nêu cudaHostAlloc; nguồn của chúng tôi không nói bộ nhớ đã đăng ký (registered) hoạt động thế nào vào năm 2009.'), () => tr('not in the book', 'không có trong sách')),
      chased: c('unknown', () => tr('The book names only cudaHostAlloc; our sources do not say how registered memory behaved in 2009.', 'Sách chỉ nêu cudaHostAlloc; nguồn của chúng tôi không nói bộ nhớ đã đăng ký (registered) hoạt động thế nào vào năm 2009.'), () => tr('not in the book', 'không có trong sách')),
    },
  },
  uva: {
    malloc: {
      arg: c('no', () => tr('One address space is not one access right: a kernel still cannot dereference malloc memory.', 'Một không gian địa chỉ không có nghĩa là một quyền truy cập: kernel vẫn không thể giải tham chiếu bộ nhớ malloc.'), () => 'p. 625'),
      chased: c('no', () => tr('A malloc pointer that the host stored is invalid on the GPU.', 'Một con trỏ malloc do host lưu lại là không hợp lệ trên GPU.'), () => 'p. 625'),
    },
    pinned: {
      arg: c('remote', () => tr('The same pointer now works directly; every access crosses the interconnect.', 'Giờ chính con trỏ đó dùng được trực tiếp; mọi lần truy cập đều đi qua đường kết nối.'), () => 'p. 625; A3'),
      chased: c('remote', () => tr('The book\'s one exception: a pointer read from memory is safe if it points into cudaHostAlloc memory.', 'Ngoại lệ duy nhất của sách: một con trỏ đọc từ bộ nhớ là an toàn nếu nó trỏ vào bộ nhớ cudaHostAlloc.'), () => 'p. 625'),
    },
    registered: {
      arg: c('other-ptr', () => tr('An exception to unified addressing: the device address comes from cudaHostGetDevicePointer, unless the device reports CanUseHostPointerForRegisteredMem.', 'Một ngoại lệ của địa chỉ hợp nhất (unified addressing): địa chỉ device lấy từ cudaHostGetDevicePointer, trừ khi device báo CanUseHostPointerForRegisteredMem.'), () => 'A3, A5'),
      chased: c('other-ptr', () => tr('A pointer the host stored holds the host address, which is not this memory\'s device address.', 'Một con trỏ do host lưu lại giữ địa chỉ host, vốn không phải địa chỉ device của vùng nhớ này.'), () => 'A3, A5'),
    },
    device: {
      host: c('no', () => tr('Unified addresses, but host code still cannot dereference cudaMalloc memory.', 'Địa chỉ đã hợp nhất, nhưng mã host vẫn không thể giải tham chiếu bộ nhớ cudaMalloc.'), () => 'p. 625'),
    },
    peer: {
      arg: c('no', () => tr('The book dates direct peer access to Kepler; NVIDIA\'s account of UVA suggests earlier, and the first architecture could not be confirmed.', 'Sách xếp truy cập ngang hàng (peer) trực tiếp vào thời Kepler; mô tả UVA của NVIDIA gợi ý sớm hơn, và chưa xác nhận được kiến trúc đầu tiên.'), () => 'p. 626; A8'),
    },
  },
  kepler: {
    peer: {
      arg: c('after-enable', () => tr('Pass the pointer and the kernel loads over PCIe or NVLink, once cudaDeviceEnablePeerAccess has granted access.', 'Truyền con trỏ vào và kernel nạp dữ liệu qua PCIe hoặc NVLink, một khi cudaDeviceEnablePeerAccess đã cấp quyền truy cập.'), () => 'p. 626; A8'),
      chased: c('after-enable', () => tr('The peer pointer has the same value on both GPUs, so a stored one can be followed once access is enabled.', 'Con trỏ peer có cùng giá trị trên cả hai GPU, nên một con trỏ đã lưu có thể được lần theo một khi quyền truy cập được bật.'), () => 'A8'),
    },
  },
  'um-limited': {
    managed: {
      host: c('not-while-running', () => tr('The CPU may touch it only while no kernel runs; touching it during one is a segmentation fault.', 'CPU chỉ được chạm vào nó khi không có kernel nào chạy; chạm vào trong lúc một kernel đang chạy là lỗi phân đoạn (segmentation fault).'), () => 'p. 627; A10'),
      arg: c('bulk', () => tr('Every managed page goes to the GPU at launch, and the total cannot exceed GPU memory.', 'Mọi trang managed được chuyển sang GPU lúc khởi chạy, và tổng dung lượng không được vượt quá bộ nhớ GPU.'), () => 'p. 627; A10'),
      chased: c(
        'bulk',
        () => tr('Our inference: one pointer value on both sides, and every managed page on the GPU from the launch, so a host-built list in managed memory should be followable. The book claims it only from Pascal on.', 'Suy luận của chúng tôi: một giá trị con trỏ ở cả hai phía, và mọi trang managed nằm trên GPU từ lúc khởi chạy, nên một danh sách do host dựng trong bộ nhớ managed lẽ ra có thể lần theo được. Sách chỉ khẳng định điều này từ Pascal trở đi.'),
        () => tr('ours, inferred from p. 626 (one virtual address for every variable); the book dates linked-structure traversal to Pascal, pp. 628–629', 'của chúng tôi, suy ra từ p. 626 (một địa chỉ ảo cho mọi biến); sách xếp việc duyệt cấu trúc liên kết vào thời Pascal, pp. 628–629'),
      ),
    },
  },
  pascal: {
    managed: {
      host: c('fault', () => tr('Allowed while kernels run; a page that lives on the GPU is serviced at longer latency.', 'Được phép trong lúc kernel chạy; một trang đang nằm trên GPU được phục vụ với độ trễ lớn hơn.'), () => 'p. 628; A10'),
      arg: c('fault', () => tr('Nothing moves at launch: each first touch faults, and the page migrates or is mapped.', 'Không có gì di chuyển lúc khởi chạy: mỗi lần chạm đầu tiên gây lỗi trang (page fault), và trang được di chuyển hoặc được ánh xạ.'), () => 'p. 628; A11'),
      chased: c('fault', () => tr('Host-built linked structures in managed memory can be followed.', 'Các cấu trúc liên kết do host dựng trong bộ nhớ managed có thể được lần theo.'), () => 'pp. 628–629; A12'),
    },
    malloc: {
      arg: c('no', () => tr('Still no: Pascal\'s page faults cover managed allocations only; plain malloc memory needs HMM or ATS.', 'Vẫn không: lỗi trang của Pascal chỉ áp dụng cho các vùng cấp phát managed; bộ nhớ malloc thường cần HMM hoặc ATS.'), () => 'A12'),
      chased: c('no', () => tr('A malloc pointer the host stored still cannot be followed; the book leaves this open, NVIDIA does not.', 'Một con trỏ malloc do host lưu lại vẫn không thể lần theo; sách để ngỏ điều này, NVIDIA thì không.'), () => 'pp. 628–629; A12'),
    },
  },
  vmm: {
    vmm: {
      host: c('unknown', () => tr('Our sources describe access grants to devices only.', 'Nguồn của chúng tôi chỉ mô tả việc cấp quyền truy cập cho device.'), () => 'A15'),
      arg: c('after-enable', () => tr('Mapping alone is not enough: a kernel may touch the range only after cuMemSetAccess.', 'Chỉ ánh xạ thôi thì chưa đủ: kernel chỉ được chạm vào vùng địa chỉ sau cuMemSetAccess.'), () => 'p. 629; A15'),
      chased: c('after-enable', () => tr('One range and one pointer, even across devices, once access is granted.', 'Một vùng địa chỉ và một con trỏ, kể cả giữa các device, một khi quyền truy cập đã được cấp.'), () => 'p. 629; A15'),
    },
  },
  hmm: {
    malloc: {
      arg: c('fault', () => tr('The host pointer works in kernels; with HMM a first touch faults the page over, with ATS access is coherent in hardware.', 'Con trỏ host dùng được trong kernel; với HMM, lần chạm đầu tiên gây lỗi trang để chuyển trang sang, với ATS truy cập được phần cứng giữ nhất quán.'), () => 'A12'),
      chased: c('fault', () => tr('Host-built lists in plain malloc memory can be followed.', 'Các danh sách do host dựng trong bộ nhớ malloc thường có thể được lần theo.'), () => 'A12'),
    },
    registered: {
      arg: c('remote', () => tr('All host memory is reachable through host pointers; pinned pages do not migrate.', 'Mọi bộ nhớ host đều chạm tới được qua con trỏ host; các trang được ghim không di chuyển.'), () => 'A3; D1'),
      chased: c('remote', () => tr('All host memory is reachable through host pointers; pinned pages do not migrate.', 'Mọi bộ nhớ host đều chạm tới được qua con trỏ host; các trang được ghim không di chuyển.'), () => 'A3; D1'),
    },
    device: {
      host: c('no', () => tr('Even with hardware coherence, host code cannot dereference cudaMalloc memory.', 'Ngay cả khi phần cứng giữ nhất quán, mã host vẫn không thể giải tham chiếu bộ nhớ cudaMalloc.'), () => 'A12'),
    },
  },
};

function applyPatch(g: Grid, p: Patch): Grid {
  const out = {} as Grid;
  for (const r of ROWS) out[r] = { ...g[r], ...(p[r] ?? {}) };
  return out;
}

const CUDA: Record<EraId, Grid> = (() => {
  const all = {} as Record<EraId, Grid>;
  let g = COPIES;
  for (const id of ERA_ORDER) {
    g = applyPatch(g, PATCHES[id]);
    all[id] = g;
  }
  return all;
})();

// ---------------------------------------------------------------- HIP today (ROCm 10.2)
function hipGrid(xnack: boolean): Grid {
  const noX = () => tr('Not GPU-visible without XNACK: it needs HSA_XNACK=1 and an HMM driver, on CDNA2 or later.', 'GPU không nhìn thấy nếu không có XNACK: cần HSA_XNACK=1 và một driver HMM, trên CDNA2 trở lên.');
  const withX = () => tr('A first GPU access causes a recoverable page fault; CDNA2 and later only.', 'Lần GPU truy cập đầu tiên gây một lỗi trang có thể phục hồi; chỉ trên CDNA2 trở lên.');
  return {
    malloc: {
      host: c('local', () => tr('Ordinary host memory.', 'Bộ nhớ host thông thường.'), () => 'B5'),
      arg: xnack ? c('fault', withX, () => 'B5') : c('no', noX, () => 'B5'),
      chased: xnack ? c('fault', withX, () => 'B5') : c('no', noX, () => 'B5'),
    },
    pinned: {
      host: c('local', () => tr('Pinned host memory.', 'Bộ nhớ host được ghim.'), () => 'B2'),
      arg: c('remote', () => tr('The same pointer works in kernels; traffic crosses the interconnect, much slower than on-device bandwidth.', 'Chính con trỏ đó dùng được trong kernel; lưu lượng đi qua đường kết nối, chậm hơn nhiều so với băng thông trên device.'), () => 'B2'),
      chased: c('remote', () => tr('AMD: the pointer can be used on both host and device.', 'AMD: con trỏ dùng được trên cả host lẫn device.'), () => 'B2'),
    },
    registered: {
      host: c('local', () => tr('Your own malloc memory, page-locked where it is.', 'Bộ nhớ malloc của chính bạn, được khóa trang ngay tại chỗ.'), () => 'B5'),
      arg: c('unknown', () => tr('Device-accessible after hipHostRegister; AMD does not say which pointer the kernel should use.', 'Device truy cập được sau hipHostRegister; AMD không nói kernel nên dùng con trỏ nào.'), () => 'B5'),
      chased: c('unknown', () => tr('Device-accessible after hipHostRegister; AMD does not say which pointer the kernel should use.', 'Device truy cập được sau hipHostRegister; AMD không nói kernel nên dùng con trỏ nào.'), () => 'B5'),
    },
    device: {
      host: c('remote', () => tr('Allowed, unlike CUDA: AMD documents host access to hipMalloc memory as zero copy over the interconnect.', 'Được phép, khác với CUDA: tài liệu của AMD mô tả việc host truy cập bộ nhớ hipMalloc là zero copy qua đường kết nối.'), () => 'B3; ch02-facts B7'),
      arg: c('local', () => tr('Device memory: the fast path.', 'Bộ nhớ device: con đường nhanh.'), () => 'B3'),
      chased: c('local', () => tr('A device pointer stored in device memory can be followed.', 'Một con trỏ device lưu trong bộ nhớ device có thể được lần theo.'), () => 'B3'),
    },
    peer: {
      host: c('remote', () => tr('As for any hipMalloc memory: zero copy over the interconnect.', 'Như với mọi bộ nhớ hipMalloc: zero copy qua đường kết nối.'), () => 'B3; ch02-facts B7'),
      arg: c('after-enable', () => tr('Direct loads and stores once hipDeviceEnablePeerAccess has been called.', 'Nạp và ghi trực tiếp một khi đã gọi hipDeviceEnablePeerAccess.'), () => 'B4'),
      chased: c('after-enable', () => tr('Direct loads and stores once hipDeviceEnablePeerAccess has been called.', 'Nạp và ghi trực tiếp một khi đã gọi hipDeviceEnablePeerAccess.'), () => 'B4'),
    },
    managed: xnack
      ? {
          host: c('unknown', () => tr('AMD describes managed pages as migrating automatically; the CPU side is not spelled out.', 'AMD mô tả các trang managed là tự động di chuyển; phía CPU không được nói rõ.'), () => 'B3, B5'),
          arg: c('fault', () => tr('A GPU access to a page outside device memory faults, and the page migrates.', 'Một lần GPU truy cập vào trang nằm ngoài bộ nhớ device gây lỗi trang, và trang được di chuyển.'), () => 'B5'),
          chased: c('fault', () => tr('A GPU access to a page outside device memory faults, and the page migrates.', 'Một lần GPU truy cập vào trang nằm ngoài bộ nhớ device gây lỗi trang, và trang được di chuyển.'), () => 'B5'),
        }
      : {
          host: c('local', () => tr('Without XNACK it behaves as pinned host memory: the pages stay in host memory.', 'Không có XNACK, nó hoạt động như bộ nhớ host được ghim: các trang ở lại bộ nhớ host.'), () => 'B5'),
          arg: c('remote', () => tr('Without XNACK it behaves as pinned host memory, read over the interconnect.', 'Không có XNACK, nó hoạt động như bộ nhớ host được ghim, được đọc qua đường kết nối.'), () => 'B5'),
          chased: c('remote', () => tr('Without XNACK it behaves as pinned host memory, read over the interconnect.', 'Không có XNACK, nó hoạt động như bộ nhớ host được ghim, được đọc qua đường kết nối.'), () => 'B5'),
        },
    shared: {
      host: c('no', () => tr('Shared memory (LDS) exists only on the GPU.', 'Bộ nhớ chia sẻ (LDS) chỉ tồn tại trên GPU.'), () => 'B1'),
      arg: c('local', () => tr('A generic "flat" pointer: the address falls in the LDS aperture.', 'Một con trỏ tổng quát "flat": địa chỉ rơi vào khẩu độ (aperture) LDS.'), () => 'B1'),
      chased: c('local', () => tr('A generic "flat" pointer read back from memory works too, inside the owning block.', 'Một con trỏ tổng quát "flat" đọc lại từ bộ nhớ cũng dùng được, bên trong khối sở hữu.'), () => 'B1'),
    },
    vmm: {
      host: c('unknown', () => tr('Our sources describe access grants to devices only.', 'Nguồn của chúng tôi chỉ mô tả việc cấp quyền truy cập cho device.'), () => 'B7'),
      arg: c('after-enable', () => tr('Usable once hipMemMap and hipMemSetAccess have run.', 'Dùng được một khi hipMemMap và hipMemSetAccess đã chạy.'), () => 'B7'),
      chased: c('after-enable', () => tr('Usable once hipMemMap and hipMemSetAccess have run.', 'Dùng được một khi hipMemMap và hipMemSetAccess đã chạy.'), () => 'B7'),
    },
  };
}

// ---------------------------------------------------------------- Metal today (Apple silicon)
const NOT_DOC = () => tr('Kernels receive buffers by binding; our sources say nothing about following CPU-built pointers to them.', 'Kernel nhận buffer qua việc gắn (binding); nguồn của chúng tôi không nói gì về việc lần theo các con trỏ tới chúng do CPU dựng.');
const METAL: Grid = {
  malloc: {
    host: c('local', () => tr('Ordinary memory.', 'Bộ nhớ thông thường.'), () => 'C2'),
    arg: c('no', () => tr('A kernel sees only buffers you bind: wrap page-aligned memory with newBufferWithBytesNoCopy first.', 'Kernel chỉ thấy các buffer bạn gắn (bind): trước hết hãy bọc bộ nhớ căn theo trang bằng newBufferWithBytesNoCopy.'), () => 'C2; ch02-facts C2'),
    chased: c('no', () => tr('Not GPU-visible until it is wrapped in a buffer.', 'GPU không nhìn thấy nó cho đến khi nó được bọc trong một buffer.'), () => 'C2'),
  },
  pinned: {
    host: c('local', () => tr('contents is the CPU\'s pointer to it: one memory, no copy.', 'contents là con trỏ của CPU tới nó: một bộ nhớ, không sao chép.'), () => 'ch05-facts C6'),
    arg: c('local', () => tr('Bound by object and read in place: on Apple silicon there is no interconnect to cross.', 'Được gắn theo đối tượng và đọc tại chỗ: trên Apple silicon không có đường kết nối nào phải đi qua.'), () => 'C2'),
    chased: c('unknown', NOT_DOC, () => 'C3'),
  },
  registered: {
    host: c('local', () => tr('Your own page-aligned memory, used in place.', 'Bộ nhớ căn theo trang của chính bạn, dùng tại chỗ.'), () => 'ch02-facts C2'),
    arg: c('local', () => tr('Bound like any buffer; llama.cpp wraps memory-mapped weights this way.', 'Được gắn như mọi buffer; llama.cpp bọc các trọng số được ánh xạ bộ nhớ (memory-mapped) theo cách này.'), () => 'C2; D4'),
    chased: c('unknown', NOT_DOC, () => 'C3'),
  },
  device: {
    host: c('no', () => tr('contents is NULL for a buffer with private storage.', 'contents là NULL với một buffer có chế độ lưu trữ private.'), () => 'ch05-facts C6'),
    arg: c('local', () => tr('GPU-only storage, filled by a blit copy.', 'Vùng lưu trữ chỉ dành cho GPU, được điền bằng một lần sao chép blit.'), () => 'C2; ch02-facts C3'),
    chased: c('unknown', NOT_DOC, () => 'C3'),
  },
  peer: {
    host: c('no', () => tr('No peer GPUs on Apple silicon; peer groups are deprecated there.', 'Apple silicon không có GPU ngang hàng (peer); peer group đã bị deprecated ở đó.'), () => 'C4'),
    arg: c('no', () => tr('No peer GPUs on Apple silicon; peer groups are deprecated there.', 'Apple silicon không có GPU ngang hàng (peer); peer group đã bị deprecated ở đó.'), () => 'C4'),
    chased: c('no', () => tr('No peer GPUs on Apple silicon; peer groups are deprecated there.', 'Apple silicon không có GPU ngang hàng (peer); peer group đã bị deprecated ở đó.'), () => 'C4'),
  },
  managed: {
    host: c('local', () => tr('Apple: managed storage has no effect on Apple silicon; use shared.', 'Apple: chế độ lưu trữ managed không có tác dụng trên Apple silicon; hãy dùng shared.'), () => 'ch05-facts C6; C5'),
    arg: c('local', () => tr('Nothing migrates: there is one memory.', 'Không có gì di chuyển: chỉ có một bộ nhớ.'), () => 'C5'),
    chased: c('unknown', NOT_DOC, () => 'C3'),
  },
  shared: {
    host: c('no', () => tr('Threadgroup memory exists only on the GPU, one copy per threadgroup.', 'Bộ nhớ threadgroup chỉ tồn tại trên GPU, mỗi threadgroup một bản.'), () => 'C1'),
    arg: c('local', () => tr('Through a helper whose parameter says threadgroup, called from a kernel: one overload per address space.', 'Qua một hàm phụ trợ có tham số khai báo threadgroup, được gọi từ kernel: mỗi không gian địa chỉ một bản nạp chồng (overload).'), () => 'C1'),
    chased: c('unknown', () => tr('Not documented in our sources.', 'Không được ghi trong nguồn của chúng tôi.'), () => 'C1'),
  },
  vmm: {
    host: c('unknown', () => tr('Not documented in our sources.', 'Không được ghi trong nguồn của chúng tôi.'), () => 'C6'),
    arg: c('local', () => tr('Once its pages are mapped from a placement heap with updateBufferMappings; there is no separate access grant.', 'Một khi các trang của nó được ánh xạ từ một placement heap bằng updateBufferMappings; không có bước cấp quyền truy cập riêng.'), () => 'C6'),
    chased: c('unknown', () => tr('Not documented in our sources.', 'Không được ghi trong nguồn của chúng tôi.'), () => 'C6'),
  },
};

export interface Options {
  /** HIP only: HSA_XNACK=1 with an HMM driver on a GPU that supports it. */
  xnack?: boolean;
}

/** The whole grid for a platform; `eraId` matters for CUDA only (HIP and Metal are today). */
export function grid(platform: Platform, eraId: EraId, opts: Options = {}): Grid {
  if (platform === 'hip') return hipGrid(!!opts.xnack);
  if (platform === 'metal') return METAL;
  return CUDA[eraId];
}

export function cell(platform: Platform, eraId: EraId, row: RowId, site: SiteId, opts: Options = {}): Cell {
  return grid(platform, eraId, opts)[row][site];
}

/** The cells whose verdict differs from the era before (CUDA only). */
export function changed(eraId: EraId): { row: RowId; site: SiteId; from: Verdict; to: Verdict }[] {
  const i = ERA_ORDER.indexOf(eraId);
  if (i <= 0) return [];
  const before = CUDA[ERA_ORDER[i - 1]];
  const now = CUDA[eraId];
  const out: { row: RowId; site: SiteId; from: Verdict; to: Verdict }[] = [];
  for (const r of ROWS)
    for (const s of SITES)
      if (before[r][s].verdict !== now[r][s].verdict) out.push({ row: r, site: s, from: before[r][s].verdict, to: now[r][s].verdict });
  return out;
}

/** The first CUDA era in which a cell's verdict is one of `vs`, or null. */
export function firstEra(row: RowId, site: SiteId, vs: Verdict[]): EraId | null {
  return ERA_ORDER.find((id) => vs.includes(CUDA[id][row][site].verdict)) ?? null;
}

/** Every facts item a cell or an era cites, for the check that each one exists. */
export function citedItems(): string[] {
  const texts: string[] = [];
  for (const id of ERA_ORDER) for (const r of ROWS) for (const s of SITES) texts.push(CUDA[id][r][s].source);
  for (const x of [false, true]) {
    const g = hipGrid(x);
    for (const r of ROWS) for (const s of SITES) texts.push(g[r][s].source);
  }
  for (const r of ROWS) for (const s of SITES) texts.push(METAL[r][s].source);
  for (const e of ERAS) if (e.corrected) texts.push(e.corrected.item);
  return texts;
}
