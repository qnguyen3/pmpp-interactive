import type { Chapter } from '../types';

// One file per section, each loaded on demand and exporting its component and its
// questions (the shape of chapter 5, the template).
export const chapter: Chapter = {
  id: 'ch10',
  num: '10',
  title: 'Reduction',
  pages: '221–250',
  vi: { title: 'Phép rút gọn' },
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '221',
      lede: 'What this chapter is for, where reduction sits under softmax, max-logit and argmax, and a check on what it assumes from chapters 4 to 9.',
      vi: { title: 'Trước khi bắt đầu', lede: 'Chương này để làm gì, phép rút gọn (reduction) nằm ở đâu bên dưới softmax, max-logit và argmax, và một bài kiểm tra những gì nó giả định bạn còn nhớ từ chương 4 đến 9.' },
      load: () => import('./S100').then((m) => ({ Component: m.S100, questions: m.questionsS100 })),
    },
    {
      id: '10.1', num: '10.1', title: 'Background', pages: '221–223',
      lede: 'Write any reduction as one loop over a binary operator and its identity, and pick a starting value that is safe on real data.',
      vi: { title: 'Bối cảnh', lede: 'Viết mọi phép rút gọn thành một vòng lặp trên một toán tử hai ngôi và phần tử đơn vị của nó, và chọn một giá trị khởi đầu an toàn trên dữ liệu thật.' },
      load: () => import('./S101').then((m) => ({ Component: m.S101, questions: m.questionsS101 })),
    },
    {
      id: '10.2', num: '10.2', title: 'Reduction trees', pages: '223–227',
      lede: 'Draw the tree a parallel reduction computes, count its work and span, and say what the operator must satisfy for the tree to give the loop\'s answer.',
      vi: { title: 'Cây rút gọn', lede: 'Vẽ cây mà một phép rút gọn song song tính, đếm khối lượng công việc và độ dài đường găng (span) của nó, và nói toán tử phải thỏa điều gì để cây cho ra cùng kết quả với vòng lặp.' },
      load: () => import('./S102').then((m) => ({ Component: m.S102, questions: m.questionsS102 })),
    },
    {
      id: '10.3', num: '10.3', title: 'A simple reduction kernel', pages: '227–230',
      lede: 'Map a reduction tree onto one block of threads, step the in-place kernel, and say what its barrier guarantees.',
      vi: { title: 'Một kernel rút gọn đơn giản', lede: 'Ánh xạ một cây rút gọn lên một khối luồng, chạy từng bước kernel tại chỗ, và nói rào chắn của nó bảo đảm điều gì.' },
      load: () => import('./S103').then((m) => ({ Component: m.S103, questions: m.questionsS103 })),
    },
    {
      id: '10.4', num: '10.4', title: 'Reducing control divergence', pages: '230–234',
      lede: 'Count the divergent warps and the resource efficiency of a reduction, and remove most of both with a three-line change.',
      vi: { title: 'Giảm phân kỳ điều khiển', lede: 'Đếm số warp bị phân kỳ và hiệu suất sử dụng tài nguyên của một phép rút gọn, và loại bỏ phần lớn cả hai bằng một thay đổi ba dòng.' },
      load: () => import('./S104').then((m) => ({ Component: m.S104, questions: m.questionsS104 })),
    },
    {
      id: '10.5', num: '10.5', title: 'Reducing memory access divergence', pages: '234–235',
      lede: 'Count the memory requests of each kernel from the addresses a warp touches, and see why the convergent kernel coalesces.',
      vi: { title: 'Giảm phân kỳ truy cập bộ nhớ', lede: 'Đếm số yêu cầu bộ nhớ của mỗi kernel từ các địa chỉ mà một warp chạm tới, và thấy vì sao kernel hội tụ được hợp nhất truy cập.' },
      load: () => import('./S105').then((m) => ({ Component: m.S105, questions: m.questionsS105 })),
    },
    {
      id: '10.6', num: '10.6', title: 'Reducing global memory accesses', pages: '236–237',
      lede: 'Move the tree into shared memory, place its barrier, and count the global traffic that is left.',
      vi: { title: 'Giảm truy cập bộ nhớ toàn cục', lede: 'Chuyển cây vào bộ nhớ chia sẻ (shared memory), đặt rào chắn của nó, và đếm lưu lượng toàn cục còn lại.' },
      load: () => import('./S106').then((m) => ({ Component: m.S106, questions: m.questionsS106 })),
    },
    {
      id: '10.7', num: '10.7', title: 'Reducing synchronization overhead with warp-level primitives', pages: '237–241',
      lede: 'Exchange values between lanes with shuffles, write a warp reduction with no shared memory and no barrier, and make it compile for CUDA, HIP and Metal.',
      vi: { title: 'Giảm chi phí đồng bộ hóa bằng các primitive cấp warp', lede: 'Trao đổi giá trị giữa các lane bằng shuffle, viết một phép rút gọn warp không dùng bộ nhớ chia sẻ và không có rào chắn, và làm cho nó biên dịch được cho CUDA, HIP và Metal.' },
      load: () => import('./S107').then((m) => ({ Component: m.S107, questions: m.questionsS107 })),
    },
    {
      id: '10.8', num: '10.8', title: 'Further reducing synchronization overhead with two-stage warp-wide reduction', pages: '241–244',
      lede: 'Reduce every warp in registers, hand off once through shared memory, and argue why more idle lanes can still be faster.',
      vi: { title: 'Giảm thêm chi phí đồng bộ hóa bằng phép rút gọn toàn warp hai giai đoạn', lede: 'Rút gọn mọi warp trong thanh ghi, chuyển giao một lần qua bộ nhớ chia sẻ, và lập luận vì sao nhiều lane rảnh hơn vẫn có thể nhanh hơn.' },
      load: () => import('./S108').then((m) => ({ Component: m.S108, questions: m.questionsS108 })),
    },
    {
      id: '10.9', num: '10.9', title: 'Reduction for arbitrary length inputs', pages: '244–246',
      lede: 'Reduce any length with many blocks, commit with an atomic, and decide when a second pass is the better design.',
      vi: { title: 'Phép rút gọn cho đầu vào độ dài tùy ý', lede: 'Rút gọn mọi độ dài bằng nhiều khối, ghi kết quả bằng một phép toán nguyên tử (atomic), và quyết định khi nào một lượt thứ hai là thiết kế tốt hơn.' },
      load: () => import('./S109').then((m) => ({ Component: m.S109, questions: m.questionsS109 })),
    },
    {
      id: '10.10', num: '10.10', title: 'Thread coarsening to reduce overhead', pages: '246–248',
      lede: 'Choose a coarsening factor for a given input and device, and see why the best one leaves just enough blocks to fill the hardware.',
      vi: { title: 'Gộp việc cho luồng để giảm chi phí', lede: 'Chọn một hệ số gộp việc cho một đầu vào và thiết bị cho trước, và thấy vì sao hệ số tốt nhất chỉ để lại vừa đủ khối để lấp đầy phần cứng.' },
      load: () => import('./S1010').then((m) => ({ Component: m.S1010, questions: m.questionsS1010 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Three labs on your own GPUs: one sum written three ways and measured, a reduction hunt, and milestone M4, softmax and argmax over logits.',
      vi: { title: 'Lab', lede: 'Ba lab trên GPU của chính bạn: một phép tổng viết theo ba cách và được đo, một cuộc săn lỗi phép rút gọn, và mốc dự án M4, softmax và argmax trên logits.' },
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '10.11', title: 'Summary and exercises', pages: '249–250',
      lede: 'What to carry forward, a mixed check on the whole chapter, where each exercise is worked, the PDF\'s apparent errata, and milestone M4.',
      vi: { title: 'Tóm tắt và bài tập', lede: 'Những gì cần mang theo, một bài kiểm tra tổng hợp cho cả chương, nơi mỗi bài tập được giải, các lỗi in có vẻ có trong bản PDF, và mốc dự án M4.' },
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
