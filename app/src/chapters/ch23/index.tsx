import type { Chapter } from '../types';

// Condensed (PLAN.md): the book's section 23.3 (streams, asynchronous copies, pinned memory,
// events) is kept and taught on one GPU, its first half on page 23.3 and its second half on page
// 23.4. MPI, NCCL and NVSHMEM (§23.2, §23.4, §23.5) get a paragraph each and one comparison
// table on page 23.4, whose displayed number says so; its id stays '23.4' because progress
// keys use it. One file per section, each loaded on demand and exporting its component and questions.
export const chapter: Chapter = {
  id: 'ch23',
  num: '23',
  title: 'Multi-GPU programming',
  pages: '541–576',
  vi: { title: 'Lập trình đa GPU' },
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '541–549',
      lede: "Say what this chapter gives you on one GPU, read the book's running example (a stencil split by rows across GPUs, with halo rows refreshed every step), and check the ideas from chapters 2, 4, 8, 10 and 22 and appendix C that it builds on.",
      vi: {
        title: 'Trước khi bắt đầu',
        lede: 'Nói được chương này cho bạn điều gì trên một GPU, đọc ví dụ xuyên suốt của sách (một stencil được chia theo hàng cho nhiều GPU, với các hàng vùng biên (halo) được làm mới ở mỗi bước), và kiểm tra lại các ý từ các chương 2, 4, 8, 10 và 22 cùng phụ lục C mà nó dựa vào.',
      },
      load: () => import('./S230').then((m) => ({ Component: m.S230, questions: m.questionsS230 })),
    },
    {
      id: '23.3', num: '23.3', title: 'Streams, asynchronous copies and pinned memory', pages: '553–556, 560–563',
      lede: 'Split a job that copies in, computes and copies out across streams so the copies hide behind the kernel, predict how long it takes and what idle time remains, and say why the host buffer must be pinned.',
      vi: {
        title: 'Stream, sao chép bất đồng bộ và bộ nhớ ghim',
        lede: 'Chia một công việc gồm chép vào, tính toán và chép ra cho nhiều stream để các lần sao chép ẩn sau kernel, dự đoán nó mất bao lâu và còn lại bao nhiêu thời gian rảnh, và nói vì sao buffer phía host phải được ghim.',
      },
      load: () => import('./S233').then((m) => ({ Component: m.S233, questions: m.questionsS233 })),
    },
    {
      id: '23.4', num: '23.3–23.5', title: 'Events, priorities, and who waits', pages: '556–575',
      lede: "The second half of the book's section 23.3: order work across streams with events, say what a stream priority can and cannot promise, and find who waits in the book's loop; then, in brief, where NCCL and NVSHMEM (§23.4, §23.5) move that waiting.",
      vi: {
        title: 'Event, độ ưu tiên, và ai phải chờ',
        lede: 'Nửa sau mục 23.3 của sách: sắp thứ tự công việc giữa các stream bằng event, nói được một độ ưu tiên stream có thể và không thể hứa điều gì, và tìm ai phải chờ trong vòng lặp của sách; rồi, ngắn gọn, NCCL và NVSHMEM (§23.4, §23.5) chuyển sự chờ đó đi đâu.',
      },
      load: () => import('./S234').then((m) => ({ Component: m.S234, questions: m.questionsS234 })),
    },
    {
      id: 'labs', num: '', title: 'Lab', pages: null,
      lede: 'One optional lab: split a copy-in, compute, copy-out job across streams and measure what the overlap buys on your GPU against a prediction; on a Mac, keep several buffers in flight so the CPU fills the next chunk while the GPU computes.',
      vi: {
        title: 'Lab',
        lede: 'Một lab không bắt buộc: chia một công việc chép vào, tính toán, chép ra cho nhiều stream và đo xem sự chồng lấp mang lại bao nhiêu trên GPU của bạn so với một dự đoán; trên Mac, giữ nhiều buffer đang được dùng cùng lúc để CPU nạp chunk tiếp theo trong khi GPU tính toán.',
      },
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '23.6', title: 'Summary and exercises', pages: '574–576',
      lede: 'What to carry forward, a mixed check on the whole chapter, where each of the book\'s exercises is worked, the PDF\'s apparent slips and dated claims, and why this chapter has no project milestone.',
      vi: {
        title: 'Tóm tắt và bài tập',
        lede: 'Những gì cần mang theo, một bài kiểm tra tổng hợp cho cả chương, nơi mỗi bài tập của sách được giải, các chỗ có vẻ sai và các khẳng định đã cũ trong PDF, và vì sao chương này không có mốc dự án.',
      },
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
