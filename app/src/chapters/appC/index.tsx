import type { Chapter } from '../types';

// Appendix C, built in the shape of chapter 5: one file per section, each loaded on
// demand and exporting its component and its questions.
export const chapter: Chapter = {
  id: 'appC',
  num: 'C',
  title: 'CUDA memories, address spaces, and pointers',
  pages: '623–629',
  vi: { title: 'Các loại bộ nhớ, không gian địa chỉ và con trỏ của CUDA' },
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '623',
      lede: "Say why chapter 2's copy-everything model stops being enough, step through the eras of CUDA's memory model on one map, and check what this appendix assumes you remember.",
      vi: { title: 'Trước khi bắt đầu', lede: 'Nói được vì sao mô hình sao chép mọi thứ của chương 2 không còn đủ, đi qua từng thời kỳ của mô hình bộ nhớ CUDA trên một bản đồ, và kiểm tra những gì phụ lục này giả định bạn còn nhớ.' },
      load: () => import('./SC0').then((m) => ({ Component: m.SC0, questions: m.questionsSC0 })),
    },
    {
      id: 'C.1', num: 'C.1', title: 'Unified device memory address space', pages: '624',
      lede: 'Write one device helper that takes a plain pointer to shared or global data in CUDA and HIP, and the overloads Metal needs instead.',
      vi: { title: 'Không gian địa chỉ bộ nhớ device hợp nhất', lede: 'Viết một hàm phụ trợ trên device nhận một con trỏ thường tới dữ liệu chia sẻ hoặc toàn cục trong CUDA và HIP, và các bản nạp chồng (overload) mà Metal cần thay vào đó.' },
      load: () => import('./SC1').then((m) => ({ Component: m.SC1, questions: m.questionsSC1 })),
    },
    {
      id: 'C.2', num: 'C.2', title: 'Zero-copy access to host memory', pages: '624–625',
      lede: 'Decide when a kernel should read host memory in place over the interconnect, by estimating what that does to a memory-bound kernel and when copying first pays off.',
      vi: { title: 'Truy cập zero-copy vào bộ nhớ host', lede: 'Quyết định khi nào một kernel nên đọc trực tiếp bộ nhớ host qua đường kết nối, bằng cách ước lượng điều đó gây ra gì cho một kernel bị giới hạn bởi bộ nhớ và khi nào sao chép trước thì đáng.' },
      load: () => import('./SC2').then((m) => ({ Component: m.SC2, questions: m.questionsSC2 })),
    },
    {
      id: 'C.3', num: 'C.3', title: 'Unified virtual address space', pages: '625',
      lede: "Tell from a pointer's value where its memory lives, and say why that alone does not make the pointer safe to dereference or to follow.",
      vi: { title: 'Không gian địa chỉ ảo hợp nhất', lede: 'Biết từ giá trị của một con trỏ rằng bộ nhớ của nó nằm ở đâu, và nói được vì sao chỉ điều đó thôi không làm cho con trỏ an toàn để giải tham chiếu hay để đi theo.' },
      load: () => import('./SC3').then((m) => ({ Component: m.SC3, questions: m.questionsSC3 })),
    },
    {
      id: 'C.4', num: 'C.4', title: 'Large virtual and physical GPU address spaces', pages: '625–626',
      lede: 'Work out what an address width can reach, see why 32-bit GPU addresses forced special allocations, and connect that to the 32-bit limits that still matter in kernels.',
      vi: { title: 'Không gian địa chỉ ảo và vật lý lớn của GPU', lede: 'Tính xem một độ rộng địa chỉ với tới được bao xa, thấy vì sao địa chỉ GPU 32-bit buộc phải dùng các phép cấp phát đặc biệt, và nối điều đó với những giới hạn 32-bit vẫn còn quan trọng trong kernel.' },
      load: () => import('./SC4').then((m) => ({ Component: m.SC4, questions: m.questionsSC4 })),
    },
    {
      id: 'C.5', num: 'C.5', title: 'Unified physical address space', pages: '626',
      lede: "Say what a kernel needs before it can load directly from another GPU's memory, what that replaces, and where it stops.",
      vi: { title: 'Không gian địa chỉ vật lý hợp nhất', lede: 'Nói được một kernel cần gì trước khi có thể nạp trực tiếp từ bộ nhớ của một GPU khác, điều đó thay thế cái gì, và nó dừng ở đâu.' },
      load: () => import('./SC5').then((m) => ({ Component: m.SC5, questions: m.questionsSC5 })),
    },
    {
      id: 'C.6', num: 'C.6', title: 'Unified memory', pages: '626–627',
      lede: "Port a CPU function to managed memory, say where its bytes live after every line on each kind of system, and keep CUDA's Unified Memory apart from Apple's unified memory.",
      vi: { title: 'Unified memory', lede: 'Chuyển một hàm CPU sang bộ nhớ được quản lý (managed memory), nói được các byte của nó nằm ở đâu sau mỗi dòng trên từng loại hệ thống, và phân biệt Unified Memory của CUDA với unified memory của Apple.' },
      load: () => import('./SC6').then((m) => ({ Component: m.SC6, questions: m.questionsSC6 })),
    },
    {
      id: 'C.7', num: 'C.7', title: 'Access to the full host virtual address space', pages: '627–628',
      lede: "Check whether a GPU's address translation can reach every host address, and tell 'can name it' apart from 'may access it' by reading three device attributes.",
      vi: { title: 'Truy cập toàn bộ không gian địa chỉ ảo của host', lede: 'Kiểm tra xem cơ chế dịch địa chỉ của một GPU có với tới mọi địa chỉ host không, và phân biệt \'gọi tên được\' với \'được phép truy cập\' bằng cách đọc ba thuộc tính của device.' },
      load: () => import('./SC7').then((m) => ({ Component: m.SC7, questions: m.questionsSC7 })),
    },
    {
      id: 'C.8', num: 'C.8', title: 'Page fault handling', pages: '628–629',
      lede: 'Predict how many pages a sequence of CPU and GPU accesses moves under bulk migration, fault-driven migration, mapping and replication, and what happens when the data outgrows device memory.',
      vi: { title: 'Xử lý lỗi trang (page fault)', lede: 'Dự đoán một chuỗi truy cập của CPU và GPU di chuyển bao nhiêu trang dưới các cơ chế di chuyển hàng loạt, di chuyển theo lỗi trang, ánh xạ và nhân bản, và điều gì xảy ra khi dữ liệu vượt quá bộ nhớ device.' },
      load: () => import('./SC8').then((m) => ({ Component: m.SC8, questions: m.questionsSC8 })),
    },
    {
      id: 'C.9', num: 'C.9', title: 'Virtual address space control', pages: '629',
      lede: 'Reserve an address range, back it with physical memory piece by piece, grant access, and use that to grow a buffer such as a KV cache without copying it.',
      vi: { title: 'Điều khiển không gian địa chỉ ảo', lede: 'Đặt trước một dải địa chỉ, gắn bộ nhớ vật lý cho nó từng phần một, cấp quyền truy cập, và dùng điều đó để nới rộng một buffer như KV cache mà không phải sao chép.' },
      load: () => import('./SC9').then((m) => ({ Component: m.SC9, questions: m.questionsSC9 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Measure on your own devices which of three ways of feeding one kernel wins for dense, repeated and sparse access, after predicting it.',
      vi: { title: 'Lab', lede: 'Đo trên chính các thiết bị của bạn xem cách nào trong ba cách cấp dữ liệu cho một kernel thắng với truy cập dày đặc, lặp lại và thưa, sau khi đã dự đoán.' },
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '', title: 'Summary', pages: '623–629',
      lede: 'Carry forward what each memory model lets you dereference and what it costs, check yourself across the whole appendix, and see where the PDF looks wrong.',
      vi: { title: 'Tóm tắt', lede: 'Mang theo những gì mỗi mô hình bộ nhớ cho phép bạn giải tham chiếu và cái giá của nó, tự kiểm tra trên toàn bộ phụ lục, và xem chỗ nào bản PDF có vẻ sai.' },
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
