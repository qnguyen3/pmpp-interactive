// Chapter 6, section 6.8: the book's checklist of optimisations (Fig. 6.13, p. 148)
// as data, and the logic of the three interactives built on it: the decision tool
// (pick a symptom, the rows that relieve it rise), the sorting drill (twelve cards into
// four categories) and the scenario cards.
//
// Sources (the lesson cites them where they are used):
//  - notes/ch06.md, section 6.8: the table reproduced in our wording, the per-row
//    commentary of pp. 149 to 153, and "Unclear in the PDF" items 13 and 14;
//  - notes/ch06.md, sections 6.1 to 6.7 and 6.9, for the costs the chapter states;
//  - notes/today/ch06-facts.md A11 (register pressure of vector loads, alignment) and
//    D1 to D4 (what llama.cpp, MLX and PyTorch do), the only sources for "seen in";
//  - notes/today/infra-facts.md, "Where the backends really differ" (lane masks), for
//    the portability cost of warp-level primitives.
//
// What is ours, not the book's: the "what it costs" column where it says so, the
// "seen in" column, the symptom list and its ranking, the "why this category"
// sentences, and the scenarios' wording. Each states its source.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.
import { tr } from '../lib/locale.ts';

export type Category = 'compute' | 'memory' | 'sync' | 'general';
export const CATEGORIES: readonly Category[] = ['compute', 'memory', 'sync', 'general'];
export const CATEGORY_NAME: Record<Category, string> = {
  get compute() {
    return tr('Compute utilisation', 'Tận dụng tài nguyên tính toán');
  },
  get memory() {
    return tr('Memory utilisation', 'Tận dụng bộ nhớ');
  },
  get sync() {
    return tr('Synchronisation latency', 'Độ trễ đồng bộ hóa');
  },
  get general() {
    return tr('General', 'Chung');
  },
};
/** What each category targets (p. 149), in our words. */
export const CATEGORY_AIM: Record<Category, string> = {
  get compute() {
    return tr('keep the cores busy with useful work', 'giữ cho các lõi luôn bận với việc có ích');
  },
  get memory() {
    return tr('make the memory system do less, or do it more efficiently', 'bắt hệ thống bộ nhớ làm ít việc hơn, hoặc làm hiệu quả hơn');
  },
  get sync() {
    return tr('cut the time threads spend waiting for each other at barriers', 'giảm thời gian các luồng chờ nhau tại rào chắn');
  },
  get general() {
    return tr('which of the above it helps depends on the kernel', 'nó giúp nhóm nào ở trên là tùy vào kernel');
  },
};

export type RowId =
  | 'occupancy'
  | 'unroll'
  | 'divergence'
  | 'coalesce'
  | 'smem-tiling'
  | 'reg-tiling'
  | 'vector'
  | 'banks'
  | 'privatise'
  | 'warp'
  | 'double-buffer'
  | 'coarsen';

export interface Cost {
  text: string;
  /** Where the cost is stated ("p. 153", "facts A11"), or null: then the cost is "not stated in the book". */
  source: string | null;
}

export interface SeenIn {
  /** Project and file, as the facts file names them. */
  where: string;
  what: string;
  /** The facts item, in notes/today/ch06-facts.md. */
  fact: 'D1' | 'D2' | 'D3' | 'D4';
  /** True when the facts item shows the technique is NOT used there. */
  absent?: boolean;
  /** Text that appears in that facts item, so a test can tie the entry to it. */
  token: string;
}

export interface Row {
  id: RowId;
  category: Category;
  name: string;
  /** Book's column "benefit to compute cores", our wording. */
  benefitCompute: string;
  /** Book's column "benefit to memory", our wording; null where the table lists none. */
  benefitMemory: string | null;
  strategies: string[];
  /** Pages of the table row and of the row's commentary in the text. */
  pages: string;
  /** The text's commentary for the row (pp. 149 to 153), in our words. */
  commentary: string;
  cost: Cost;
  /** Later chapters where the book says the row is used again. */
  returnsIn: { ch: number; what: string }[];
  /** Real backends, only from facts D1 to D4. Empty means "not checked". */
  seenIn: SeenIn[];
  /** Where the course teaches it, as [chapter id, section id]. */
  taught: [string, string][];
  /** Why the book files it under its category (primary intent, p. 149). Ours. */
  why: string;
}

export const ROWS: readonly Row[] = [
  {
    id: 'occupancy',
    category: 'compute',
    get name() { return tr('Occupancy tuning', 'Tinh chỉnh occupancy (mức chiếm dụng)'); },
    get benefitCompute() {
      return tr(
        'More resident work to cover pipeline latency',
        'Nhiều việc thường trú hơn để che độ trễ pipeline',
      );
    },
    get benefitMemory() {
      return tr(
        'More memory requests in flight, which covers DRAM latency',
        'Nhiều yêu cầu bộ nhớ đang thực hiện cùng lúc hơn, giúp che độ trễ DRAM',
      );
    },
    get strategies() {
      return [tr('Tune what each block takes from the SM: threads per block, shared memory per block, registers per thread', 'Chỉnh lượng tài nguyên mỗi khối lấy từ SM: số luồng mỗi khối, bộ nhớ chia sẻ mỗi khối, số thanh ghi mỗi luồng')];
    },
    pages: '148–149',
    get commentary() {
      return tr(
        'Chapter 4 introduced it: keep many more threads than cores so there is always something to run. Chapter 5 added shared memory as a third budget, and section 6.2 added that resident warps also keep the memory system busy. From here on the book assumes you keep occupancy healthy whenever an optimisation spends registers or shared memory.',
        'Chương 4 đã giới thiệu nó: giữ nhiều luồng hơn hẳn số lõi để lúc nào cũng có việc để chạy. Chương 5 thêm bộ nhớ chia sẻ làm ngân sách thứ ba, và mục 6.2 bổ sung rằng các warp thường trú còn giữ cho hệ thống bộ nhớ luôn bận. Từ đây trở đi, sách giả định bạn giữ occupancy ở mức tốt mỗi khi một tối ưu hóa tiêu tốn thanh ghi hoặc bộ nhớ chia sẻ.',
      );
    },
    cost: {
      get text() { return tr('Caps the registers and shared memory each thread or block can use, which other optimisations want; chapter 15 deliberately gives occupancy up when other optimisations make up for it.', 'Giới hạn số thanh ghi và bộ nhớ chia sẻ mà mỗi luồng hoặc khối được dùng, vốn là thứ các tối ưu hóa khác cần; chương 15 cố ý hy sinh occupancy khi các tối ưu hóa khác bù lại được.'); },
      source: 'p. 149',
    },
    returnsIn: [{ ch: 15, get what() { return tr('more resources per thread, lower occupancy, compensated by other optimisations', 'nhiều tài nguyên hơn cho mỗi luồng, occupancy thấp hơn, được bù bằng các tối ưu hóa khác'); } }],
    seenIn: [],
    taught: [['ch04', '4.7'], ['ch05', '5.6'], ['ch06', '6.2']],
    get why() {
      return tr(
        'Its first aim is that cores always have a ready warp. More requests in flight is a welcome side effect.',
        'Mục tiêu đầu tiên của nó là để các lõi luôn có một warp sẵn sàng. Nhiều yêu cầu đang thực hiện cùng lúc hơn là một tác dụng phụ đáng mừng.',
      );
    },
  },
  {
    id: 'unroll',
    category: 'compute',
    get name() { return tr('Loop unrolling', 'Trải vòng lặp (loop unrolling)'); },
    get benefitCompute() {
      return tr(
        'Fewer branch instructions, and more independent instructions to schedule, so fewer stalls',
        'Ít lệnh rẽ nhánh hơn, và nhiều lệnh độc lập hơn để lập lịch, nên ít lần đình trệ hơn',
      );
    },
    get benefitMemory() {
      return tr(
        'Can let local arrays be promoted to registers, which removes their memory traffic',
        'Có thể cho phép đưa mảng cục bộ lên thanh ghi, loại bỏ lưu lượng bộ nhớ của chúng',
      );
    },
    get strategies() {
      return [tr('Mostly the compiler does it. Help it by giving loops constant bounds where you can', 'Phần lớn do trình biên dịch làm. Hãy giúp nó bằng cách cho vòng lặp biên là hằng số khi có thể')];
    },
    pages: '148–149',
    get commentary() {
      return tr(
        'A recap of section 6.6. The book assumes the compiler unrolls well enough. Chapters 11 and 15 make it essential for register tiling after coarsening, and chapter 15 also uses it to make up for low occupancy.',
        'Tóm tắt lại mục 6.6. Sách giả định trình biên dịch trải vòng lặp đủ tốt. Chương 11 và 15 biến nó thành điều thiết yếu cho phân ô thanh ghi sau khi gộp việc cho luồng, và chương 15 còn dùng nó để bù cho occupancy thấp.',
      );
    },
    cost: { get text() { return tr('Not stated in the book.', 'Sách không nêu.'); }, source: null },
    returnsIn: [
      { ch: 11, get what() { return tr('register tiling after coarsening, in prefix sum', 'phân ô thanh ghi sau khi gộp việc cho luồng, trong tổng tiền tố'); } },
      { ch: 15, get what() { return tr('register tiling, and making up for low occupancy', 'phân ô thanh ghi, và bù cho occupancy thấp'); } },
    ],
    seenIn: [
      {
        where: 'MLX, steel/defines.h', // i18n-ignore: project name and file path
        get what() {
          return tr(
            'STEEL_PRAGMA_UNROLL, a full-unroll loop pragma used in its Metal GEMM',
            'STEEL_PRAGMA_UNROLL, một pragma trải toàn bộ vòng lặp, dùng trong GEMM Metal của nó',
          );
        },
        fact: 'D3',
        token: 'STEEL_PRAGMA_UNROLL', // i18n-ignore: facts-file text a test matches verbatim
      },
      {
        where: 'llama.cpp, Metal kernels/common.h', // i18n-ignore: project name and file path
        get what() {
          return tr(
            'FOR_UNROLL, the same Clang loop pragma',
            'FOR_UNROLL, cùng loại pragma vòng lặp của Clang',
          );
        },
        fact: 'D2',
        token: 'FOR_UNROLL', // i18n-ignore: facts-file text a test matches verbatim
      },
    ],
    taught: [['ch06', '6.6']],
    get why() {
      return tr(
        'Its first aim is fewer stalls and branches in the instruction stream. Register promotion is a bonus.',
        'Mục tiêu đầu tiên của nó là bớt đình trệ và rẽ nhánh trong dòng lệnh. Việc đưa lên thanh ghi là phần lợi thêm.',
      );
    },
  },
  {
    id: 'divergence',
    category: 'compute',
    get name() { return tr('Reducing control divergence', 'Giảm phân kỳ luồng (divergence)'); },
    get benefitCompute() {
      return tr(
        'High SIMD efficiency: fewer lanes idle while a warp executes',
        'Hiệu suất SIMD cao: ít lane rảnh hơn khi một warp thực thi',
      );
    },
    benefitMemory: null,
    get strategies() {
      return [
        tr(
          'Reassign threads to work or data',
          'Phân lại luồng cho công việc hoặc dữ liệu',
        ),
        tr(
          'Change the data layout',
          'Đổi cách bố trí dữ liệu',
        ),
      ];
    },
    pages: '148–150',
    get commentary() {
      return tr(
        'Part 1 kernels diverged only at boundaries. Later chapters reassign work so that one warp is fully used before the next (reduction), give the threads of a warp similar amounts of work (graph traversal), or reorder data so that one warp gets similar rows (the JDS sparse format).',
        'Các kernel ở Phần 1 chỉ phân kỳ ở biên. Các chương sau phân lại công việc để một warp được dùng hết trước warp kế tiếp (phép rút gọn), giao cho các luồng của một warp lượng việc tương đương nhau (duyệt đồ thị), hoặc sắp lại dữ liệu để một warp nhận các hàng tương tự nhau (định dạng ma trận thưa JDS).',
      );
    },
    cost: { get text() { return tr('Not stated in the book.', 'Sách không nêu.'); }, source: null },
    returnsIn: [
      { ch: 10, get what() { return tr('reduction: use one warp fully before the next', 'phép rút gọn: dùng hết một warp trước warp kế tiếp'); } },
      { ch: 17, get what() { return tr('sparse matrices: the JDS layout', 'ma trận thưa: bố cục JDS'); } },
      { ch: 18, get what() { return tr('graph traversal: vertex-centric versus edge-centric', 'duyệt đồ thị: lấy đỉnh làm trung tâm so với lấy cạnh làm trung tâm'); } },
    ],
    seenIn: [],
    taught: [['ch04', '4.5']],
    get why() {
      return tr(
        'Idle SIMD lanes are idle cores. The table lists no benefit to memory at all.',
        'Lane SIMD rảnh là lõi rảnh. Bảng không ghi lợi ích nào cho bộ nhớ cả.',
      );
    },
  },
  {
    id: 'coalesce',
    category: 'memory',
    get name() { return tr('Coalesced global memory accesses', 'Truy cập bộ nhớ toàn cục được hợp nhất'); },
    get benefitCompute() {
      return tr(
        'Fewer stalls waiting on global memory',
        'Ít đình trệ hơn khi chờ bộ nhớ toàn cục',
      );
    },
    get benefitMemory() {
      return tr(
        'Less global traffic, and fuller use of each burst and cache line',
        'Ít lưu lượng toàn cục hơn, và dùng trọn hơn mỗi burst và dòng bộ nhớ đệm (cache line)',
      );
    },
    get strategies() {
      return [
        tr(
          'Remap threads to data',
          'Ánh xạ lại luồng với dữ liệu',
        ),
        tr(
          'Change the layout: transpose, or turn an array of structures into a structure of arrays',
          'Đổi bố cục: chuyển vị, hoặc biến mảng các cấu trúc thành cấu trúc các mảng',
        ),
        tr(
          'Move data between global and shared memory in a coalesced order and do the irregular access in shared memory (corner turning, packing)',
          'Di chuyển dữ liệu giữa bộ nhớ toàn cục và bộ nhớ chia sẻ theo thứ tự được hợp nhất, và thực hiện truy cập bất quy tắc trong bộ nhớ chia sẻ (corner turning, đóng gói)',
        ),
      ];
    },
    pages: '148, 150–151',
    get commentary() {
      return tr(
        'The threads of a warp should touch neighbouring addresses in the same burst or line. For irregular patterns: stage through shared memory (corner turning in 6.1; packing in filter and sorting; a shared copy to binary-search in merge), remap threads (reduction, chapter 15), or change the layout (ELL and JDS, a transpose, structure of arrays).',
        'Các luồng của một warp nên chạm vào các địa chỉ liền kề trong cùng một burst hoặc dòng. Với các mẫu bất quy tắc: trung chuyển qua bộ nhớ chia sẻ (corner turning ở 6.1; đóng gói trong lọc và sắp xếp; một bản sao chia sẻ để tìm kiếm nhị phân trong trộn), ánh xạ lại luồng (phép rút gọn, chương 15), hoặc đổi bố cục (ELL và JDS, một phép chuyển vị, cấu trúc các mảng).',
      );
    },
    cost: {
      get text() { return tr("Corner turning's store into shared memory is strided and can turn into a bank conflict, which is the next row's problem.", 'Lệnh ghi vào bộ nhớ chia sẻ của corner turning có bước nhảy (stride) và có thể gây xung đột bank, đó là vấn đề của hàng tiếp theo.'); },
      source: 'pp. 139–140',
    },
    returnsIn: [
      { ch: 10, get what() { return tr('remapping threads to data', 'ánh xạ lại luồng với dữ liệu'); } },
      { ch: 12, get what() { return tr('packing scattered writes in shared memory', 'đóng gói các lệnh ghi rải rác trong bộ nhớ chia sẻ'); } },
      { ch: 13, get what() { return tr('coalesced load, then binary search in shared memory', 'nạp được hợp nhất, rồi tìm kiếm nhị phân trong bộ nhớ chia sẻ'); } },
      { ch: 14, get what() { return tr('packing', 'đóng gói'); } },
      { ch: 15, get what() { return tr('remapping threads to data', 'ánh xạ lại luồng với dữ liệu'); } },
      { ch: 17, get what() { return tr('ELL and JDS layouts', 'bố cục ELL và JDS'); } },
    ],
    seenIn: [
      {
        where: 'MLX, steel/gemm/gemm.h', // i18n-ignore: project name and file path
        get what() {
          return tr(
            'a transposed operand is loaded along its contiguous direction and the tile is indexed to match: corner turning',
            'một toán hạng đã chuyển vị được nạp theo hướng liên tục của nó và ô được đánh chỉ số cho khớp: corner turning',
          );
        },
        fact: 'D3',
        token: 'loading along the contiguous direction', // i18n-ignore: facts-file text a test matches verbatim
      },
    ],
    taught: [['ch06', '6.1']],
    get why() {
      return tr(
        'It changes how the memory system is used. Fewer stalls follows from that.',
        'Nó thay đổi cách dùng hệ thống bộ nhớ. Bớt đình trệ là hệ quả của điều đó.',
      );
    },
  },
  {
    id: 'smem-tiling',
    category: 'memory',
    get name() { return tr('Shared memory tiling', 'Phân ô bằng bộ nhớ chia sẻ'); },
    get benefitCompute() {
      return tr(
        'Fewer stalls waiting on global memory',
        'Ít đình trệ hơn khi chờ bộ nhớ toàn cục',
      );
    },
    get benefitMemory() { return tr('Less global traffic', 'Ít lưu lượng toàn cục hơn'); },
    get strategies() {
      return [tr('Keep data a block reuses in shared memory, so it crosses between global memory and the SM once', 'Giữ dữ liệu mà một khối dùng lại trong bộ nhớ chia sẻ, để nó chỉ đi giữa bộ nhớ toàn cục và SM một lần')];
    },
    pages: '148, 151',
    get commentary() {
      return tr(
        'Chapter 5 with matmul. It returns in most patterns: with input and output tiles of different shapes in convolution and stencil, and in more elaborate forms in chapter 15 (matmul) and chapter 16 (sequence alignment).',
        'Chương 5, với phép nhân ma trận. Nó quay lại trong hầu hết các mẫu: với ô đầu vào và ô đầu ra khác hình dạng trong tích chập và stencil, và ở dạng phức tạp hơn trong chương 15 (phép nhân ma trận) và chương 16 (căn chỉnh chuỗi).',
      );
    },
    cost: { get text() { return tr('Spends shared memory, and can lower occupancy when shared memory is what limits it.', 'Tiêu tốn bộ nhớ chia sẻ, và có thể làm giảm occupancy khi bộ nhớ chia sẻ là thứ giới hạn nó.'); }, source: 'p. 153' },
    returnsIn: [
      { ch: 7, get what() { return tr('convolution: input and output tiles of different sizes', 'tích chập: ô đầu vào và ô đầu ra khác kích thước'); } },
      { ch: 8, get what() { return tr('stencil', 'stencil'); } },
      { ch: 15, get what() { return tr('matmul, more elaborate', 'phép nhân ma trận, phức tạp hơn'); } },
      { ch: 16, get what() { return tr('sequence alignment', 'căn chỉnh chuỗi'); } },
    ],
    seenIn: [
      {
        where: 'MLX, steel/gemm/gemm.h', // i18n-ignore: project name and file path
        get what() {
          return tr(
            'GEMM operands staged in threadgroup-memory tiles',
            'các toán hạng GEMM được trung chuyển qua các ô trong bộ nhớ threadgroup',
          );
        },
        fact: 'D3',
        token: 'threadgroup tiles', // i18n-ignore: facts-file text a test matches verbatim
      },
    ],
    taught: [['ch05', '5.3'], ['ch05', '5.4']],
    get why() {
      return tr(
        'It exists to cut global traffic. It is the canonical memory optimisation.',
        'Nó tồn tại để cắt giảm lưu lượng toàn cục. Đây là tối ưu hóa bộ nhớ kinh điển.',
      );
    },
  },
  {
    id: 'reg-tiling',
    category: 'memory',
    get name() { return tr('Register tiling', 'Phân ô thanh ghi'); },
    get benefitCompute() {
      return tr(
        'Fewer stalls waiting on shared memory',
        'Ít đình trệ hơn khi chờ bộ nhớ chia sẻ',
      );
    },
    get benefitMemory() { return tr('Less shared memory traffic', 'Ít lưu lượng bộ nhớ chia sẻ hơn'); },
    get strategies() {
      return [tr('Keep data a warp or a thread reuses in registers, so it crosses between shared memory and registers once', 'Giữ dữ liệu mà một warp hoặc một luồng dùng lại trong thanh ghi, để nó chỉ đi giữa bộ nhớ chia sẻ và thanh ghi một lần')];
    },
    pages: '148, 151',
    get commentary() {
      return tr(
        'Already in use: each thread of the chapter 3 and 5 kernels sums into a register and writes once, so the registers of all threads form a tile of the output. It returns in stencil (a shared tile moved to registers and back) and in prefix sum and chapter 15 (each thread owns several outputs and reuses several inputs).',
        'Đã được dùng sẵn: mỗi luồng của các kernel ở chương 3 và 5 cộng dồn vào một thanh ghi và ghi ra một lần, nên thanh ghi của mọi luồng tạo thành một ô của đầu ra. Nó quay lại trong stencil (một ô chia sẻ được chuyển vào thanh ghi rồi trả về) và trong tổng tiền tố và chương 15 (mỗi luồng sở hữu nhiều đầu ra và dùng lại nhiều đầu vào).',
      );
    },
    cost: { get text() { return tr('Not stated in the book.', 'Sách không nêu.'); }, source: null },
    returnsIn: [
      { ch: 8, get what() { return tr('stencil: shared tile into registers and back', 'stencil: ô chia sẻ vào thanh ghi rồi trả về'); } },
      { ch: 11, get what() { return tr('prefix sum: several outputs per thread', 'tổng tiền tố: nhiều đầu ra cho mỗi luồng'); } },
      { ch: 15, get what() { return tr('matmul: several outputs and inputs per thread', 'phép nhân ma trận: nhiều đầu ra và đầu vào cho mỗi luồng'); } },
    ],
    seenIn: [],
    taught: [['ch05', '5.4']],
    get why() {
      return tr(
        'It removes traffic, this time between shared memory and registers.',
        'Nó loại bỏ lưu lượng, lần này là giữa bộ nhớ chia sẻ và thanh ghi.',
      );
    },
  },
  {
    id: 'vector',
    category: 'memory',
    get name() { return tr('Vector loads and stores', 'Nạp và ghi dạng vector'); },
    get benefitCompute() {
      return tr(
        'Fewer instructions spent managing memory accesses',
        'Ít lệnh hơn dành cho việc quản lý truy cập bộ nhớ',
      );
    },
    get benefitMemory() {
      return tr(
        'More bytes in flight per request, which covers DRAM latency',
        'Nhiều byte đang được truyền hơn trên mỗi yêu cầu, giúp che độ trễ DRAM',
      );
    },
    get strategies() {
      return [tr('Have each thread move 16-byte chunks of contiguous data', 'Cho mỗi luồng di chuyển các chunk 16 byte dữ liệu liên tục')];
    },
    pages: '139, 148, 151',
    get commentary() {
      return tr(
        'It applies to most computations in the book, and because it is simple the book only mentions it where it matters most. Section 6.3 adds the second benefit: a kernel that cannot run at full occupancy still keeps enough bytes in flight. Chapter 15 has such a kernel.',
        'Nó áp dụng được cho hầu hết các phép tính trong sách, và vì nó đơn giản nên sách chỉ nhắc đến ở những chỗ nó quan trọng nhất. Mục 6.3 thêm lợi ích thứ hai: một kernel không thể chạy ở occupancy tối đa vẫn giữ đủ số byte đang được truyền. Chương 15 có một kernel như vậy.',
      );
    },
    cost: {
      get text() { return tr('The tail that is not a multiple of four needs scalar code (pp. 138–139). Vendor addition: wider loads raise register pressure, and the pointer must be aligned to the vector size.', 'Phần đuôi không phải bội số của bốn cần mã vô hướng (pp. 138–139). Bổ sung từ nhà sản xuất: lệnh nạp rộng hơn làm tăng áp lực thanh ghi, và con trỏ phải được căn chỉnh theo kích thước vector.'); },
      source: 'pp. 138–139; facts A11', // i18n-ignore: citation
    },
    returnsIn: [{ ch: 15, get what() { return tr('a kernel below full occupancy', 'một kernel dưới mức occupancy tối đa'); } }],
    seenIn: [
      {
        where: 'PyTorch, aten/src/ATen/native/cuda/MemoryAccess.cuh', // i18n-ignore: project name and file path
        get what() {
          return tr(
            'aligned_vector loads, with the vector width chosen at run time from the pointer alignment',
            'các lệnh nạp aligned_vector, với độ rộng vector được chọn lúc chạy dựa trên căn chỉnh của con trỏ',
          );
        },
        fact: 'D4',
        token: 'can_vectorize_up_to', // i18n-ignore: facts-file text a test matches verbatim
      },
      {
        where: 'llama.cpp, ggml-cuda/common.cuh', // i18n-ignore: project name and file path
        get what() {
          return tr(
            '16-byte copies through int4 on CUDA and HIP',
            'các lần sao chép 16 byte qua int4 trên CUDA và HIP',
          );
        },
        fact: 'D1',
        token: 'ggml_cuda_memcpy_1', // i18n-ignore: facts-file text a test matches verbatim
      },
    ],
    taught: [['ch06', '6.3']],
    get why() {
      return tr(
        'It changes how the memory system is asked: fewer, wider requests.',
        'Nó thay đổi cách yêu cầu hệ thống bộ nhớ: ít yêu cầu hơn nhưng rộng hơn.',
      );
    },
  },
  {
    id: 'banks',
    category: 'memory',
    get name() {
      return tr(
        'Avoiding shared memory bank conflicts',
        'Tránh xung đột bank (bank conflict) trong bộ nhớ chia sẻ',
      );
    },
    get benefitCompute() {
      return tr(
        'Fewer stalls waiting on shared memory',
        'Ít đình trệ hơn khi chờ bộ nhớ chia sẻ',
      );
    },
    get benefitMemory() {
      return tr(
        'Less shared traffic, and better use of the banks',
        'Ít lưu lượng chia sẻ hơn, và dùng các bank tốt hơn',
      );
    },
    get strategies() {
      return [
        tr(
          'Remap threads to data',
          'Ánh xạ lại luồng với dữ liệu',
        ),
        tr(
          'Change the layout, often by padding',
          'Đổi bố cục, thường bằng cách đệm thêm (padding)',
        ),
      ];
    },
    pages: '148, 151–152',
    get commentary() {
      return tr(
        'It returns in prefix sum and chapter 15, where threads load their parts of the input into shared memory with a stride.',
        'Nó quay lại trong tổng tiền tố và chương 15, nơi các luồng nạp phần đầu vào của mình vào bộ nhớ chia sẻ với một bước nhảy (stride).',
      );
    },
    cost: {
      get text() { return tr('Padding costs shared memory, and index arithmetic gets dearer when a dimension stops being a power of two.', 'Padding tốn bộ nhớ chia sẻ, và phép tính chỉ số đắt hơn khi một chiều không còn là lũy thừa của hai.'); },
      source: 'pp. 140–141',
    },
    returnsIn: [
      { ch: 11, get what() { return tr('prefix sum: strided shared-memory loads', 'tổng tiền tố: nạp bộ nhớ chia sẻ có bước nhảy'); } },
      { ch: 15, get what() { return tr('matmul: strided shared-memory loads', 'phép nhân ma trận: nạp bộ nhớ chia sẻ có bước nhảy'); } },
    ],
    seenIn: [
      {
        where: 'MLX, steel/gemm/gemm.h', // i18n-ignore: project name and file path
        get what() {
          return tr(
            'threadgroup tiles padded by 16 bytes',
            'các ô threadgroup được đệm thêm 16 byte',
          );
        },
        fact: 'D3',
        token: 'tgp_padding_a', // i18n-ignore: facts-file text a test matches verbatim
      },
      {
        where: 'llama.cpp, ggml-cuda/mmq.cuh', // i18n-ignore: project name and file path
        get what() {
          return tr(
            'shared tiles padded by 16 bytes "to avoid shared memory bank conflicts"',
            'các ô chia sẻ được đệm thêm 16 byte "để tránh xung đột bank trong bộ nhớ chia sẻ"',
          );
        },
        fact: 'D1',
        token: 'To avoid shared memory bank conflicts', // i18n-ignore: facts-file text a test matches verbatim
      },
    ],
    taught: [['ch06', '6.4']],
    get why() {
      return tr(
        'It is about how the shared memory is used.',
        'Nó liên quan đến cách dùng bộ nhớ chia sẻ.',
      );
    },
  },
  {
    id: 'privatise',
    category: 'memory',
    get name() { return tr('Privatisation', 'Riêng tư hóa (privatization)'); },
    get benefitCompute() {
      return tr(
        'Fewer stalls waiting on atomic updates',
        'Ít đình trệ hơn khi chờ các cập nhật nguyên tử',
      );
    },
    get benefitMemory() {
      return tr(
        'Less contention on, and serialisation of, atomic updates',
        'Ít tranh chấp và ít tuần tự hóa các cập nhật nguyên tử hơn',
      );
    },
    get strategies() {
      return [tr('Apply partial updates to a private copy, and update the public copy once at the end', 'Áp các cập nhật từng phần lên một bản sao riêng, và cập nhật bản sao chung một lần ở cuối')];
    },
    pages: '148, 152',
    get commentary() {
      return tr(
        'Not taught yet; listed for completeness. When many threads or blocks update one output, each updates a private copy and commits it at the end: histogram counters, and the output lists of filter and graph traversal.',
        'Chưa được dạy; liệt kê cho đầy đủ. Khi nhiều luồng hoặc khối cùng cập nhật một đầu ra, mỗi bên cập nhật một bản sao riêng và ghi nó vào bản chung ở cuối: bộ đếm histogram, và danh sách đầu ra của lọc và duyệt đồ thị.',
      );
    },
    cost: {
      get text() { return tr('Each private copy must be committed to the public one at the end; coarsening in chapter 9 exists partly to need fewer copies.', 'Mỗi bản sao riêng phải được ghi vào bản sao chung ở cuối; gộp việc cho luồng ở chương 9 tồn tại một phần là để cần ít bản sao hơn.'); },
      source: 'pp. 152–153',
    },
    returnsIn: [
      { ch: 9, get what() { return tr('histogram counters', 'bộ đếm histogram'); } },
      { ch: 12, get what() { return tr('filter: many threads appending to one list', 'lọc: nhiều luồng nối thêm vào một danh sách'); } },
      { ch: 18, get what() { return tr('graph traversal: frontier lists', 'duyệt đồ thị: danh sách frontier'); } },
    ],
    seenIn: [],
    taught: [],
    get why() {
      return tr(
        'Contention on one memory location is a memory-system problem.',
        'Tranh chấp trên một vị trí bộ nhớ là vấn đề của hệ thống bộ nhớ.',
      );
    },
  },
  {
    id: 'warp',
    category: 'sync',
    get name() { return tr('Warp-level primitives', 'Các hàm nguyên thủy cấp warp'); },
    get benefitCompute() { return tr('Fewer block-wide barriers', 'Ít rào chắn toàn khối hơn'); },
    get benefitMemory() {
      return tr(
        'Less global and shared traffic (the table; the text names only shared memory)',
        'Ít lưu lượng toàn cục và chia sẻ hơn (theo bảng; phần văn bản chỉ nêu bộ nhớ chia sẻ)',
      );
    },
    get strategies() {
      return [tr('Do the steps that need a barrier at warp level, then combine the warps’ results at block level', 'Thực hiện các bước cần rào chắn ở cấp warp, rồi kết hợp kết quả của các warp ở cấp khối')];
    },
    pages: '148, 152',
    get commentary() {
      return tr(
        'Mentioned briefly in chapter 4: threads of a warp can exchange data and synchronise without a block-wide barrier and without going through shared memory. Used in reduction, prefix sum and filter, and in chapter 15 to help register tiling.',
        'Được nhắc qua ở chương 4: các luồng của một warp có thể trao đổi dữ liệu và đồng bộ hóa mà không cần rào chắn toàn khối và không phải đi qua bộ nhớ chia sẻ. Được dùng trong phép rút gọn, tổng tiền tố và lọc, và trong chương 15 để hỗ trợ phân ô thanh ghi.',
      );
    },
    cost: {
      get text() { return tr('Not stated in the book. A portability cost instead (infra facts B3 and C2): a CUDA lane mask must become a 64-bit mask on HIP, and Metal’s SIMD-group functions take no mask, so the same source needs changes per backend.', 'Sách không nêu. Thay vào đó là chi phí về tính di động (infra facts B3 và C2): một lane mask của CUDA phải thành mask 64 bit trên HIP, và các hàm SIMD group của Metal không nhận mask, nên cùng một mã nguồn cần sửa theo từng backend.'); },
      source: null,
    },
    returnsIn: [
      { ch: 10, get what() { return tr('reduction', 'phép rút gọn'); } },
      { ch: 11, get what() { return tr('prefix sum', 'tổng tiền tố'); } },
      { ch: 12, get what() { return tr('filter', 'lọc'); } },
      { ch: 15, get what() { return tr('helping register tiling', 'hỗ trợ phân ô thanh ghi'); } },
    ],
    seenIn: [],
    taught: [['ch04', '4.4'], ['ch04', '4.5']],
    get why() {
      return tr(
        'Its first aim is to replace block-wide barriers with cheaper warp-wide exchange.',
        'Mục tiêu đầu tiên của nó là thay rào chắn toàn khối bằng trao đổi trong warp rẻ hơn.',
      );
    },
  },
  {
    id: 'double-buffer',
    category: 'sync',
    get name() { return tr('Double buffering', 'Buffer kép (double buffering)'); },
    get benefitCompute() {
      return tr(
        'Removes barriers that only enforce false dependences',
        'Bỏ các rào chắn chỉ dùng để đảm bảo phụ thuộc giả',
      );
    },
    benefitMemory: null,
    get strategies() {
      return [tr('Write into a different buffer from the one the preceding reads use, so the write-after-read barrier can go', 'Ghi vào một buffer khác với buffer mà các lần đọc trước đó dùng, để có thể bỏ rào chắn write-after-read')];
    },
    pages: '148, 152',
    get commentary() {
      return tr(
        'A recap of section 6.7. Its real uses are in prefix sum and chapter 15.',
        'Tóm tắt lại mục 6.7. Các ứng dụng thực sự của nó nằm ở tổng tiền tố và chương 15.',
      );
    },
    cost: { get text() { return tr('Two buffers instead of one: twice the memory for the buffer.', 'Hai buffer thay vì một: gấp đôi bộ nhớ cho buffer.'); }, source: 'p. 147' },
    returnsIn: [
      { ch: 11, get what() { return tr('prefix sum', 'tổng tiền tố'); } },
      { ch: 15, get what() { return tr('matmul', 'phép nhân ma trận'); } },
    ],
    seenIn: [
      {
        where: 'llama.cpp, ggml-cuda/fattn-mma-f16.cuh', // i18n-ignore: project name and file path
        get what() {
          return tr(
            'flash attention preloads the next data into a second stage with asynchronous copies',
            'flash attention nạp trước dữ liệu kế tiếp vào tầng thứ hai bằng các lần sao chép bất đồng bộ',
          );
        },
        fact: 'D1',
        token: 'nstages_target', // i18n-ignore: facts-file text a test matches verbatim
      },
      {
        where: 'MLX, steel/gemm/gemm.h', // i18n-ignore: project name and file path
        get what() {
          return tr(
            'not used: the main GEMM loop is single-buffered with two barriers per step, the book’s "before" loop',
            'không dùng: vòng lặp GEMM chính chỉ có một buffer với hai rào chắn mỗi bước, đúng vòng lặp "trước" của sách',
          );
        },
        fact: 'D3',
        absent: true,
        token: 'single-buffered with two barriers', // i18n-ignore: facts-file text a test matches verbatim
      },
    ],
    taught: [['ch06', '6.7']],
    get why() {
      return tr(
        'All it removes is waiting at a barrier.',
        'Tất cả những gì nó loại bỏ là việc chờ tại rào chắn.',
      );
    },
  },
  {
    id: 'coarsen',
    category: 'general',
    get name() { return tr('Thread coarsening', 'Gộp việc cho luồng (thread coarsening)'); },
    get benefitCompute() {
      return tr(
        'Depends on what parallelisation costs in this kernel',
        'Tùy vào chi phí song song hóa trong kernel này',
      );
    },
    get benefitMemory() {
      return tr(
        'Depends on what parallelisation costs in this kernel',
        'Tùy vào chi phí song song hóa trong kernel này',
      );
    },
    get strategies() {
      return [tr('Give each thread several units of work, to cut a parallelisation overhead you are paying for nothing', 'Giao cho mỗi luồng nhiều đơn vị công việc, để cắt một chi phí song song hóa mà bạn đang trả vô ích')];
    },
    pages: '148, 152–153',
    get commentary() {
      return tr(
        'What it removes differs by kernel: redundant input loads by many blocks (stencil, matmul), private copies to commit (histogram), synchronisation and divergence (reduction, prefix sum), redundant work of the parallel algorithm (prefix sum), poor coalescing (filter, sorting).',
        'Thứ nó loại bỏ khác nhau theo từng kernel: các lần nạp đầu vào thừa của nhiều khối (stencil, phép nhân ma trận), các bản sao riêng phải ghi lại (histogram), đồng bộ hóa và phân kỳ (phép rút gọn, tổng tiền tố), việc thừa của thuật toán song song (tổng tiền tố), hợp nhất truy cập kém (lọc, sắp xếp).',
      );
    },
    cost: {
      get text() { return tr('Exposes less parallelism (half-empty or partial waves), and more registers or shared memory per thread can cut occupancy.', 'Phơi ra ít tính song song hơn (các đợt chạy chỉ đầy một nửa hoặc không đầy), và nhiều thanh ghi hoặc bộ nhớ chia sẻ hơn cho mỗi luồng có thể làm giảm occupancy.'); },
      source: 'p. 143',
    },
    returnsIn: [
      { ch: 8, get what() { return tr('stencil: larger output tiles, fewer redundant loads', 'stencil: ô đầu ra lớn hơn, ít lần nạp thừa hơn'); } },
      { ch: 9, get what() { return tr('histogram: fewer private copies to commit', 'histogram: ít bản sao riêng phải ghi lại hơn'); } },
      { ch: 10, get what() { return tr('reduction: less synchronisation and divergence', 'phép rút gọn: ít đồng bộ hóa và phân kỳ hơn'); } },
      { ch: 11, get what() { return tr('prefix sum: less synchronisation, divergence and redundant work', 'tổng tiền tố: ít đồng bộ hóa, phân kỳ và việc thừa hơn'); } },
      { ch: 12, get what() { return tr('filter: better coalescing', 'lọc: hợp nhất truy cập tốt hơn'); } },
      { ch: 14, get what() { return tr('sorting: better coalescing', 'sắp xếp: hợp nhất truy cập tốt hơn'); } },
      { ch: 15, get what() { return tr('matmul: larger output tiles, fewer redundant loads', 'phép nhân ma trận: ô đầu ra lớn hơn, ít lần nạp thừa hơn'); } },
    ],
    seenIn: [],
    taught: [['ch06', '6.5']],
    get why() {
      return tr(
        'The overhead it removes is different in every kernel, so no single category fits.',
        'Chi phí mà nó loại bỏ khác nhau ở mỗi kernel, nên không có một nhóm duy nhất nào phù hợp.',
      );
    },
  },
];

export const rowById = (id: RowId): Row => {
  const r = ROWS.find((x) => x.id === id);
  if (!r) throw new Error('no checklist row ' + id);
  return r;
};

export function categoryCounts(rows: readonly Row[] = ROWS): Record<Category, number> {
  const c: Record<Category, number> = { compute: 0, memory: 0, sync: 0, general: 0 };
  for (const r of rows) c[r.category]++;
  return c;
}

/** "3 + 6 + 2 + 1" */
export const countLine = (c: Record<Category, number>) => CATEGORIES.map((k) => c[k]).join(' + ');

// ---------------------------------------------------------------- symptoms
export type SymptomId =
  | 'global-stall'
  | 'global-traffic'
  | 'shared-serial'
  | 'divergence'
  | 'barrier'
  | 'atomics'
  | 'instructions'
  | 'overhead';

export interface Relief {
  row: RowId;
  /** How this row relieves the symptom, in our words, with the page that supports it. */
  how: string;
  page: string;
}

export interface Symptom {
  id: SymptomId;
  label: string;
  /** What a profiler or a measurement would show. Ours. */
  looksLike: string;
  /** Rows that relieve it, most direct first. */
  relief: Relief[];
  /** Rows that spend the very resource this symptom is short of (p. 153's warning). */
  spends: Relief[];
}

export const SYMPTOMS: readonly Symptom[] = [
  {
    id: 'global-stall',
    get label() {
      return tr(
        'Stalls on global memory, bandwidth far from used',
        'Đình trệ ở bộ nhớ toàn cục, băng thông còn xa mới dùng hết',
      );
    },
    get looksLike() {
      return tr(
        'Warps wait on loads, yet the bytes moved per second are well below what the device can move.',
        'Các warp chờ lệnh nạp, nhưng số byte di chuyển mỗi giây thấp hơn nhiều so với khả năng của thiết bị.',
      );
    },
    relief: [
      { row: 'occupancy', get how() { return tr('more resident warps put more requests in flight', 'nhiều warp thường trú hơn thì có nhiều yêu cầu đang thực hiện cùng lúc hơn'); }, page: 'pp. 135–136' },
      { row: 'vector', get how() { return tr('each request carries more bytes, even when occupancy cannot rise', 'mỗi yêu cầu mang nhiều byte hơn, kể cả khi occupancy không tăng được'); }, page: 'p. 139' },
      { row: 'unroll', get how() { return tr('independent loads can be issued before the first returns', 'các lệnh nạp độc lập có thể được phát trước khi lệnh đầu tiên trả về'); }, page: 'pp. 144–145' },
      { row: 'coalesce', get how() { return tr('fewer, fuller transactions for the same data', 'ít giao dịch hơn và đầy hơn cho cùng lượng dữ liệu'); }, page: 'pp. 126, 150' },
    ],
    spends: [
      { row: 'smem-tiling', get how() { return tr('more shared memory per block can lower occupancy', 'nhiều bộ nhớ chia sẻ hơn cho mỗi khối có thể làm giảm occupancy'); }, page: 'p. 153' },
      { row: 'coarsen', get how() { return tr('more registers per thread can lower occupancy', 'nhiều thanh ghi hơn cho mỗi luồng có thể làm giảm occupancy'); }, page: 'p. 143' },
    ],
  },
  {
    id: 'global-traffic',
    get label() { return tr('Too much global traffic', 'Quá nhiều lưu lượng toàn cục'); },
    get looksLike() {
      return tr(
        'Global memory is near its limit, and the kernel moves many more bytes than its data.',
        'Bộ nhớ toàn cục gần chạm giới hạn, và kernel di chuyển nhiều byte hơn hẳn lượng dữ liệu của nó.',
      );
    },
    relief: [
      { row: 'smem-tiling', get how() { return tr('data a block reuses crosses from global memory once', 'dữ liệu mà một khối dùng lại chỉ đi từ bộ nhớ toàn cục một lần'); }, page: 'p. 151' },
      { row: 'coalesce', get how() { return tr('no bytes of a burst or line fetched and thrown away', 'không byte nào của một burst hay dòng bị nạp về rồi bỏ đi'); }, page: 'pp. 130–131' },
      { row: 'coarsen', get how() { return tr('fewer blocks reload the same input tiles', 'ít khối phải nạp lại cùng các ô đầu vào hơn'); }, page: 'pp. 142, 153' },
      { row: 'unroll', get how() { return tr('a local array promoted to registers stops costing memory traffic', 'một mảng cục bộ được đưa lên thanh ghi không còn tốn lưu lượng bộ nhớ'); }, page: 'pp. 145–146' },
      { row: 'warp', get how() { return tr('the table claims less global traffic; the text does not say why', 'bảng khẳng định ít lưu lượng toàn cục hơn; phần văn bản không nói vì sao'); }, page: 'pp. 148, 152' },
    ],
    spends: [],
  },
  {
    id: 'shared-serial',
    get label() { return tr('Shared-memory accesses serialised', 'Truy cập bộ nhớ chia sẻ bị tuần tự hóa'); },
    get looksLike() {
      return tr(
        'A warp’s shared-memory access is split into several passes (a k-way bank conflict).',
        'Một lần truy cập bộ nhớ chia sẻ của warp bị tách thành nhiều lượt (xung đột bank k chiều).',
      );
    },
    relief: [
      { row: 'banks', get how() { return tr('padding or remapping spreads the warp over all banks', 'padding hoặc ánh xạ lại trải warp ra khắp các bank'); }, page: 'pp. 140, 151' },
      { row: 'reg-tiling', get how() { return tr('values reused from registers are not read from shared memory again', 'các giá trị được dùng lại từ thanh ghi không phải đọc lại từ bộ nhớ chia sẻ'); }, page: 'p. 151' },
    ],
    spends: [{ row: 'coalesce', get how() { return tr('corner turning makes the shared-memory store strided', 'corner turning khiến lệnh ghi vào bộ nhớ chia sẻ có bước nhảy'); }, page: 'pp. 139–140' }],
  },
  {
    id: 'divergence',
    get label() { return tr('SIMD lanes idle (control divergence)', 'Lane SIMD rảnh (phân kỳ luồng)'); },
    get looksLike() {
      return tr(
        'Warps execute with many lanes switched off.',
        'Các warp thực thi với nhiều lane bị tắt.',
      );
    },
    relief: [
      { row: 'divergence', get how() { return tr('reassign work or reorder data so a warp’s threads take the same path', 'phân lại công việc hoặc sắp lại dữ liệu để các luồng của một warp đi cùng một nhánh'); }, page: 'pp. 149–150' },
      { row: 'coarsen', get how() { return tr('in reduction and prefix sum it cuts divergence overhead', 'trong phép rút gọn và tổng tiền tố, nó cắt giảm chi phí phân kỳ'); }, page: 'p. 153' },
    ],
    spends: [],
  },
  {
    id: 'barrier',
    get label() { return tr('Time waiting at barriers', 'Thời gian chờ tại rào chắn'); },
    get looksLike() {
      return tr(
        'Warps sit at __syncthreads() waiting for the rest of their block.',
        'Các warp đứng ở __syncthreads() chờ phần còn lại của khối.',
      );
    },
    relief: [
      { row: 'double-buffer', get how() { return tr('the barrier that only protects a write-after-read goes', 'rào chắn chỉ dùng để bảo vệ một write-after-read được bỏ đi'); }, page: 'p. 147' },
      { row: 'warp', get how() { return tr('steps done inside a warp need no block-wide barrier', 'các bước làm bên trong một warp không cần rào chắn toàn khối'); }, page: 'p. 152' },
      { row: 'coarsen', get how() { return tr('in reduction and prefix sum it cuts synchronisation', 'trong phép rút gọn và tổng tiền tố, nó cắt giảm đồng bộ hóa'); }, page: 'p. 153' },
    ],
    spends: [],
  },
  {
    id: 'atomics',
    get label() { return tr('Contended atomic updates', 'Cập nhật nguyên tử bị tranh chấp'); },
    get looksLike() {
      return tr(
        'Many threads update the same few locations and wait on each other.',
        'Nhiều luồng cập nhật cùng một vài vị trí và chờ lẫn nhau.',
      );
    },
    relief: [
      { row: 'privatise', get how() { return tr('updates go to private copies, committed once', 'các cập nhật đi vào bản sao riêng, được ghi lại một lần'); }, page: 'p. 152' },
      { row: 'coarsen', get how() { return tr('fewer private copies to commit (histogram)', 'ít bản sao riêng phải ghi lại hơn (histogram)'); }, page: 'p. 153' },
    ],
    spends: [],
  },
  {
    id: 'instructions',
    get label() {
      return tr(
        'Many instructions per byte or per result',
        'Nhiều lệnh trên mỗi byte hoặc mỗi kết quả',
      );
    },
    get looksLike() {
      return tr(
        'The cores are busy, but much of what they issue is loads, index arithmetic and branches.',
        'Các lõi đang bận, nhưng phần lớn những gì chúng phát là lệnh nạp, tính chỉ số và rẽ nhánh.',
      );
    },
    relief: [
      { row: 'vector', get how() { return tr('one instruction moves 16 bytes instead of 4', 'một lệnh di chuyển 16 byte thay vì 4'); }, page: 'p. 139' },
      { row: 'unroll', get how() { return tr('fewer loop branches and counter updates', 'ít rẽ nhánh vòng lặp và cập nhật bộ đếm hơn'); }, page: 'p. 144' },
      { row: 'coarsen', get how() { return tr('work common to several outputs is done once per thread', 'việc chung của nhiều đầu ra chỉ làm một lần mỗi luồng'); }, page: 'pp. 141–142' },
    ],
    spends: [{ row: 'banks', get how() { return tr('padding can make index arithmetic dearer', 'padding có thể làm phép tính chỉ số đắt hơn'); }, page: 'pp. 140–141' }],
  },
  {
    id: 'overhead',
    get label() {
      return tr(
        'The same loads or work repeated across threads',
        'Cùng các lần nạp hoặc công việc bị lặp lại giữa các luồng',
      );
    },
    get looksLike() {
      return tr(
        'Neighbouring blocks load the same inputs, or every thread redoes the same setup.',
        'Các khối lân cận nạp cùng đầu vào, hoặc mọi luồng làm lại cùng một bước chuẩn bị.',
      );
    },
    relief: [
      { row: 'coarsen', get how() { return tr('one thread does several units of work and shares the overhead', 'một luồng làm nhiều đơn vị công việc và chia sẻ chi phí phụ'); }, page: 'pp. 141–142' },
      { row: 'smem-tiling', get how() { return tr('one load per block replaces one load per thread', 'một lần nạp mỗi khối thay cho một lần nạp mỗi luồng'); }, page: 'p. 151' },
    ],
    spends: [],
  },
];

export interface RankedRow {
  row: Row;
  /** 'relieves' rows first, in order; then 'spends' (they make it worse); then the rest in book order. */
  role: 'relieves' | 'spends' | 'other';
  relief?: Relief;
  rank: number;
}

export function rankRows(symptom: SymptomId | null): RankedRow[] {
  const s = symptom ? SYMPTOMS.find((x) => x.id === symptom) : undefined;
  if (!s) return ROWS.map((row, i) => ({ row, role: 'other', rank: i }));
  const out: RankedRow[] = [];
  s.relief.forEach((r) => out.push({ row: rowById(r.row), role: 'relieves', relief: r, rank: out.length }));
  const seen = new Set(out.map((x) => x.row.id));
  for (const r of s.spends)
    if (!seen.has(r.row)) {
      out.push({ row: rowById(r.row), role: 'spends', relief: r, rank: out.length });
      seen.add(r.row);
    }
  for (const row of ROWS) if (!seen.has(row.id)) out.push({ row, role: 'other', rank: out.length });
  return out;
}

// ---------------------------------------------------------------- sorting drill
export type Placement = Partial<Record<RowId, Category>>;

export interface SortScore {
  placed: number;
  right: number;
  /** Per category: how many cards you put there, and how many the book has. */
  tally: Record<Category, { yours: number; book: number }>;
  done: boolean;
}

export function sortScore(p: Placement): SortScore {
  const book = categoryCounts();
  const tally = Object.fromEntries(CATEGORIES.map((c) => [c, { yours: 0, book: book[c] }])) as SortScore['tally'];
  let placed = 0;
  let right = 0;
  for (const row of ROWS) {
    const c = p[row.id];
    if (!c) continue;
    placed++;
    tally[c].yours++;
    if (c === row.category) right++;
  }
  return { placed, right, tally, done: placed === ROWS.length };
}

// ---------------------------------------------------------------- scenarios
export type Outcome = 'faster' | 'same' | 'slower';
export const OUTCOME_LABEL: Record<Outcome, string> = {
  get faster() {
    return tr('Faster', 'Nhanh hơn');
  },
  get same() {
    return tr('About the same', 'Gần như không đổi');
  },
  get slower() {
    return tr('Slower', 'Chậm hơn');
  },
};

export interface Scenario {
  id: string;
  /** The kernel and its bottleneck. */
  situation: string;
  /** The proposed change. */
  change: string;
  expect: Outcome;
  /** Why, tied to p. 153's rule. */
  why: string;
  row: RowId;
  page: string;
}

export const SCENARIOS: readonly Scenario[] = [
  {
    id: 'tiling-bandwidth',
    get situation() {
      return tr(
        'A tiled matmul (chapter 5’s kernel, T × T threads per block) moves bytes at close to your measured bandwidth. Its occupancy is at the device maximum, and a block with four times the shared memory and four times the threads would still leave it there.',
        'Một phép nhân ma trận phân ô (kernel của chương 5, T × T luồng mỗi khối) di chuyển byte ở mức gần băng thông bạn đo được. Occupancy của nó đang ở mức tối đa của thiết bị, và một khối với gấp bốn lần bộ nhớ chia sẻ và gấp bốn lần số luồng vẫn giữ nó ở mức đó.',
      );
    },
    get change() {
      return tr(
        'Double the tile width: four times the shared memory and four times the threads per block.',
        'Tăng gấp đôi độ rộng ô: gấp bốn lần bộ nhớ chia sẻ và gấp bốn lần số luồng mỗi khối.',
      );
    },
    expect: 'faster',
    get why() {
      return tr(
        'Tiling spends shared memory, which is not short here, to relieve global traffic, which is the bottleneck. Each input crosses from global memory half as often, and shared memory per thread stays at 8 bytes. This is the case where p. 153 says tiling pays.',
        'Phân ô tiêu tốn bộ nhớ chia sẻ, thứ không thiếu ở đây, để giảm lưu lượng toàn cục, vốn là nút thắt. Mỗi đầu vào đi từ bộ nhớ toàn cục chỉ còn một nửa số lần, và bộ nhớ chia sẻ mỗi luồng vẫn ở mức 8 byte. Đây là trường hợp mà p. 153 nói phân ô có lợi.',
      );
    },
    row: 'smem-tiling',
    page: 'p. 153',
  },
  {
    id: 'padding-no-conflict',
    get situation() {
      return tr(
        'The profiler reports one pass per shared-memory request (no bank conflicts). The kernel is limited by global memory.',
        'Profiler báo một lượt cho mỗi yêu cầu bộ nhớ chia sẻ (không có xung đột bank). Kernel bị giới hạn bởi bộ nhớ toàn cục.',
      );
    },
    get change() { return tr('Pad the shared tile by one column.', 'Đệm thêm một cột vào ô chia sẻ.'); },
    expect: 'same',
    get why() {
      return tr(
        'Padding relieves bank conflicts, and there are none: nothing is relieved. It does cost a little shared memory and index arithmetic (pp. 140–141), so if anything it is marginally slower. A gain is not on the table.',
        'Padding giải tỏa xung đột bank, mà ở đây không có: chẳng có gì được giải tỏa. Nó lại tốn một chút bộ nhớ chia sẻ và phép tính chỉ số (pp. 140–141), nên nếu có khác thì nó chậm hơn chút ít. Không có chuyện được lợi.',
      );
    },
    row: 'banks',
    page: 'p. 153',
  },
  {
    id: 'coarsen-small-grid',
    get situation() {
      return tr(
        'A grid fills the device for just over two waves of blocks, and the kernel has no redundant loads between blocks: each element is read by one thread.',
        'Một lưới lấp đầy thiết bị trong hơn hai đợt khối một chút, và kernel không có lần nạp thừa nào giữa các khối: mỗi phần tử được đọc bởi một luồng.',
      );
    },
    get change() { return tr('Coarsen by a factor of 8.', 'Gộp việc cho luồng với hệ số 8.'); },
    expect: 'slower',
    get why() {
      return tr(
        'There is no overhead to remove (pitfall 1), and the grid shrinks to about a quarter of one wave, so most of the device sits idle (pitfall 2). The book works the same arithmetic: 1056 blocks on 264 slots coarsened by 8 is half a wave.',
        'Không có chi phí phụ nào để loại bỏ (cạm bẫy 1), và lưới co lại còn khoảng một phần tư của một đợt, nên phần lớn thiết bị ngồi không (cạm bẫy 2). Sách làm cùng phép tính này: 1056 khối trên 264 chỗ, gộp việc với hệ số 8, là nửa đợt.',
      );
    },
    row: 'coarsen',
    page: 'pp. 142–143',
  },
  {
    id: 'vector-full',
    get situation() {
      return tr(
        'A streaming kernel (one add per 12 bytes) runs at full occupancy and already moves bytes at about your measured bandwidth.',
        'Một kernel streaming (một phép cộng mỗi 12 byte) chạy ở occupancy tối đa và đã di chuyển byte ở mức xấp xỉ băng thông bạn đo được.',
      );
    },
    get change() {
      return tr(
        'Switch to 16-byte vector loads and stores.',
        'Chuyển sang nạp và ghi vector 16 byte.',
      );
    },
    expect: 'same',
    get why() {
      return tr(
        'The bytes do not change and the memory system is already as busy as it gets. Vector loads cut instruction count and add bytes in flight, which matters below full occupancy (p. 139). Neither is the bottleneck here.',
        'Số byte không đổi và hệ thống bộ nhớ đã bận hết mức. Nạp vector giảm số lệnh và tăng số byte đang được truyền, điều quan trọng khi dưới mức occupancy tối đa (p. 139). Không cái nào là nút thắt ở đây.',
      );
    },
    row: 'vector',
    page: 'p. 153',
  },
  {
    id: 'double-buffer-barrier',
    get situation() {
      return tr(
        'Each iteration reads neighbours’ values from a shared buffer, writes its own, and has two barriers. Barrier waits dominate the time; occupancy has room to spare.',
        'Mỗi lần lặp đọc giá trị của các lân cận từ một buffer chia sẻ, ghi giá trị của mình, và có hai rào chắn. Thời gian chờ rào chắn chiếm phần lớn; occupancy vẫn còn dư.',
      );
    },
    get change() {
      return tr(
        'Use two alternating buffers and drop the barrier between the read and the write.',
        'Dùng hai buffer luân phiên và bỏ rào chắn giữa lần đọc và lần ghi.',
      );
    },
    expect: 'faster',
    get why() {
      return tr(
        'Half the barriers go, and the one that goes only protected a write-after-read (false) dependence. The cost, twice the buffer memory, is spent from a resource that is not short.',
        'Một nửa số rào chắn biến mất, và cái bị bỏ chỉ bảo vệ một phụ thuộc write-after-read (giả). Cái giá, gấp đôi bộ nhớ buffer, được trả bằng một tài nguyên không thiếu.',
      );
    },
    row: 'double-buffer',
    page: 'p. 147',
  },
];
