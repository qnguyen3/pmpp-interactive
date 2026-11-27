import type { Chapter } from '../types';

// One file per section, each loaded on demand and exporting its component and its
// questions (chapter 5 is the template).
export const chapter: Chapter = {
  id: 'ch15',
  num: '15',
  title: 'Advanced optimizations for matrix multiplication',
  pages: '349–370',
  vi: { title: 'Tối ưu hóa nâng cao cho phép nhân ma trận' },
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '349',
      lede: 'See how far your chapter 5 kernel is from the vendor library on your own device, and check what this chapter assumes you remember.',
      vi: { title: 'Trước khi bắt đầu', lede: 'Xem kernel chương 5 của bạn còn cách thư viện của nhà cung cấp bao xa trên chính thiết bị của bạn, và kiểm tra những gì chương này giả định bạn còn nhớ.' },
      load: () => import('./S150').then((m) => ({ Component: m.S150, questions: m.questionsS150 })),
    },
    {
      id: '15.1', num: '15.1', title: 'Background: GEMM and the layouts you will meet', pages: '349–350',
      lede: "Map a linear layer or an attention product onto the book's GEMM, and onto a column-major or row-major library call.",
      vi: { title: 'Bối cảnh: GEMM và các cách bố trí bạn sẽ gặp', lede: 'Ánh xạ một lớp tuyến tính hay một tích attention vào GEMM của sách, và vào một lời gọi thư viện theo thứ tự cột (column-major) hoặc thứ tự hàng (row-major).' },
      load: () => import('./S151').then((m) => ({ Component: m.S151, questions: m.questionsS151 })),
    },
    {
      id: '15.2', num: '15.2', title: 'Data reuse analysis', pages: '350–352',
      lede: 'Compute the arithmetic intensity of any output tile, and say why neither the tile depth nor anything at batch size one can raise it.',
      vi: { title: 'Phân tích tái sử dụng dữ liệu', lede: 'Tính cường độ tính toán của bất kỳ ô đầu ra nào, và nói vì sao cả độ sâu của ô lẫn mọi thứ ở batch kích thước một đều không thể nâng nó lên.' },
      load: () => import('./S152').then((m) => ({ Component: m.S152, questions: m.questionsS152 })),
    },
    {
      id: '15.3', num: '15.3', title: 'Larger tiles with thread coarsening', pages: '352–357',
      lede: 'Write the 128 x 128 coarsened kernel: say which thread owns which outputs, which elements it loads, and what keeps its tile in registers.',
      vi: { title: 'Ô lớn hơn với gộp việc cho luồng', lede: 'Viết kernel đã gộp việc 128 x 128: nói luồng nào sở hữu đầu ra nào, nó nạp những phần tử nào, và điều gì giữ ô của nó trong thanh ghi.' },
      load: () => import('./S153').then((m) => ({ Component: m.S153, questions: m.questionsS153 })),
    },
    {
      id: '15.4', num: '15.4', title: 'Register tiling of the input tiles', pages: '357–360',
      lede: 'Interchange the inner loops so each thread reads every shared value once, and count what that saves.',
      vi: { title: 'Phân ô thanh ghi cho các ô đầu vào', lede: 'Hoán đổi các vòng lặp trong để mỗi luồng đọc mỗi giá trị chia sẻ đúng một lần, và đếm xem điều đó tiết kiệm được bao nhiêu.' },
      load: () => import('./S154').then((m) => ({ Component: m.S154, questions: m.questionsS154 })),
    },
    {
      id: '15.5', num: '15.5', title: 'Coalesced storing of the output tile', pages: '360–361',
      lede: "Re-map threads to four 4 x 4 sub-tiles so that a warp's vector stores coalesce, for 32- and 64-wide warps.",
      vi: { title: 'Ghi ô đầu ra được hợp nhất', lede: 'Ánh xạ lại các luồng vào bốn ô con 4 x 4 để các lệnh ghi vector của một warp được hợp nhất, với warp rộng 32 và 64.' },
      load: () => import('./S155').then((m) => ({ Component: m.S155, questions: m.questionsS155 })),
    },
    {
      id: '15.6', num: '15.6', title: 'Eliminating bank conflicts', pages: '361–363',
      lede: 'Find the bank conflicts of a strip read for any leading dimension, and choose the padding that removes them.',
      vi: { title: 'Loại bỏ xung đột bank', lede: 'Tìm các xung đột bank của một lần đọc dải với bất kỳ chiều đầu (leading dimension) nào, và chọn phần đệm loại bỏ chúng.' },
      load: () => import('./S156').then((m) => ({ Component: m.S156, questions: m.questionsS156 })),
    },
    {
      id: '15.7', num: '15.7', title: 'Occupancy considerations', pages: '363–364',
      lede: 'Budget registers and shared memory for a tile shape, and explain why this kernel runs at low occupancy on purpose.',
      vi: { title: 'Cân nhắc về occupancy', lede: 'Cân đối thanh ghi và bộ nhớ chia sẻ cho một hình dạng ô, và giải thích vì sao kernel này cố ý chạy ở occupancy thấp.' },
      load: () => import('./S157').then((m) => ({ Component: m.S157, questions: m.questionsS157 })),
    },
    {
      id: '15.8', num: '15.8', title: 'Software pipelining', pages: '364–368',
      lede: 'Double-buffer and software-pipeline the tile loop with one barrier, and predict when the overlap pays.',
      vi: { title: 'Pipeline phần mềm', lede: 'Đệm kép và pipeline phần mềm cho vòng lặp ô với một rào chắn, và dự đoán khi nào sự chồng lấp có lợi.' },
      load: () => import('./S158').then((m) => ({ Component: m.S158, questions: m.questionsS158 })),
    },
    {
      id: '15.9', num: '15.9', title: 'Specialized software and hardware support', pages: '368–369',
      lede: 'Say what tensor cores, asynchronous copies and vendor libraries add, which of them each backend has today, and how far your kernel is from the library.',
      vi: { title: 'Hỗ trợ phần mềm và phần cứng chuyên biệt', lede: 'Nói tensor core, sao chép bất đồng bộ và thư viện của nhà cung cấp mang lại gì, mỗi backend hiện có cái nào, và kernel của bạn còn cách thư viện bao xa.' },
      load: () => import('./S159').then((m) => ({ Component: m.S159, questions: m.questionsS159 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Seven labs: build milestone M3 one step at a time, measure every step against lab 5.3 and the vendor library on your devices, then hunt three defects.',
      vi: { title: 'Lab', lede: 'Bảy lab: xây mốc dự án M3 từng bước một, đo mỗi bước so với lab 5.3 và thư viện của nhà cung cấp trên thiết bị của bạn, rồi săn ba lỗi.' },
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '15.10', title: 'Summary and exercises', pages: '370',
      lede: "What to carry forward, a mixed check, where the book's exercises and errata are worked, and milestone M3.",
      vi: { title: 'Tóm tắt và bài tập', lede: 'Những gì cần mang theo, một bài kiểm tra tổng hợp, nơi các bài tập và errata của sách được giải, và mốc dự án M3.' },
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
