import type { Chapter } from '../types';

// One file per section, each loaded on demand and exporting its component and its
// questions (the shape of chapter 5, the template). The title is the PDF's chapter
// heading, "Scan" (p. 251); the book's subtitle is "And work efficiency in parallel
// algorithms".
export const chapter: Chapter = {
  id: 'ch11',
  num: '11',
  title: 'Scan',
  pages: '251–287',
  vi: { title: 'Scan' },
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '251–252',
      lede: 'What scan does in an inference stack, the chain of kernels this chapter builds, and a check on what it assumes from chapters 4 to 10.',
      vi: { title: 'Trước khi bắt đầu', lede: 'Scan làm gì trong một stack suy luận, chuỗi kernel mà chương này xây dựng, và một bài kiểm tra những gì chương giả định từ chương 4 đến 10.' },
      load: () => import('./S110').then((m) => ({ Component: m.S110, questions: m.questionsS110 })),
    },
    {
      id: '11.1', num: '11.1', title: 'Background', pages: '252–254',
      lede: 'Compute inclusive and exclusive scans, convert one into the other, and say which one decides whether the token that crosses p is kept.',
      vi: { title: 'Bối cảnh', lede: 'Tính scan bao gồm và scan loại trừ, chuyển loại này sang loại kia, và nói được loại nào quyết định token vượt qua p có được giữ lại hay không.' },
      load: () => import('./S111').then((m) => ({ Component: m.S111, questions: m.questionsS111 })),
    },
    {
      id: '11.2', num: '11.2', title: 'Parallel scan with the Kogge-Stone algorithm', pages: '254–258',
      lede: "Trace Kogge-Stone's doubling strides, write its block kernel, and find the race its second barrier prevents.",
      vi: { title: 'Scan song song với thuật toán Kogge-Stone', lede: 'Lần theo các bước nhảy (stride) tăng gấp đôi của Kogge-Stone, viết kernel cho một khối, và tìm race condition mà rào chắn thứ hai của nó ngăn chặn.' },
      load: () => import('./S112').then((m) => ({ Component: m.S112, questions: m.questionsS112 })),
    },
    {
      id: '11.3', num: '11.3', title: 'Double-buffering to reduce synchronization', pages: '258–261',
      lede: 'Tell the true dependence from the false one, remove the false one with two buffers, and say which buffer ends up holding the result.',
      vi: { title: 'Buffer kép để giảm đồng bộ hóa', lede: 'Phân biệt phụ thuộc thật với phụ thuộc giả, loại bỏ phụ thuộc giả bằng hai buffer, và nói được buffer nào cuối cùng chứa kết quả.' },
      load: () => import('./S113').then((m) => ({ Component: m.S113, questions: m.questionsS113 })),
    },
    {
      id: '11.4', num: '11.4', title: 'Warp-level primitives to reduce synchronization', pages: '261–265',
      lede: 'Scan a warp with shuffles and no barrier, build a block scan from warp scans with two barriers, and make it compile and run on CUDA, HIP and Metal.',
      vi: { title: 'Primitive cấp warp để giảm đồng bộ hóa', lede: 'Scan một warp bằng shuffle mà không cần rào chắn, xây scan khối từ các scan warp với hai rào chắn, và làm cho nó biên dịch và chạy được trên CUDA, HIP và Metal.' },
      load: () => import('./S114').then((m) => ({ Component: m.S114, questions: m.questionsS114 })),
    },
    {
      id: '11.5', num: '11.5', title: 'Work efficiency considerations', pages: '265–267',
      lede: 'Count the additions and steps of sequential, Kogge-Stone and Brent-Kung scans, and predict when the parallel one loses.',
      vi: { title: 'Những cân nhắc về hiệu quả công việc', lede: 'Đếm số phép cộng và số bước của scan tuần tự, Kogge-Stone và Brent-Kung, và dự đoán khi nào bản song song thua.' },
      load: () => import('./S115').then((m) => ({ Component: m.S115, questions: m.questionsS115 })),
    },
    {
      id: '11.6', num: '11.6', title: 'Coarsening to improve work-efficiency', pages: '267–270',
      lede: 'Give each thread a run to scan sequentially, keep loads and stores coalesced, and count what coarsening saves.',
      vi: { title: 'Gộp việc cho luồng để cải thiện hiệu quả công việc', lede: 'Giao cho mỗi luồng một đoạn để scan tuần tự, giữ các lần nạp và ghi được hợp nhất, và đếm xem gộp việc tiết kiệm được gì.' },
      load: () => import('./S116').then((m) => ({ Component: m.S116, questions: m.questionsS116 })),
    },
    {
      id: '11.7', num: '11.7', title: 'Register tiling to avoid shared memory access latency', pages: '270–272',
      lede: "Keep each thread's run in registers, use shared memory only to coalesce, and remove the bank conflicts its strided accesses cause.",
      vi: { title: 'Phân ô thanh ghi để tránh độ trễ truy cập bộ nhớ chia sẻ', lede: 'Giữ đoạn của mỗi luồng trong thanh ghi, chỉ dùng bộ nhớ chia sẻ để hợp nhất truy cập, và loại bỏ các xung đột bank mà truy cập theo bước nhảy của nó gây ra.' },
      load: () => import('./S117').then((m) => ({ Component: m.S117, questions: m.questionsS117 })),
    },
    {
      id: '11.8', num: '11.8', title: 'Memory bandwidth considerations', pages: '272–273',
      lede: "Work out scan's operations per byte, turn your measured bandwidth into a ceiling in elements per second, and judge a kernel by the 80% yardstick.",
      vi: { title: 'Những cân nhắc về băng thông bộ nhớ', lede: 'Tính số phép toán trên mỗi byte của scan, biến băng thông bạn đo được thành một trần tính bằng phần tử mỗi giây, và đánh giá một kernel theo thước đo 80%.' },
      load: () => import('./S118').then((m) => ({ Component: m.S118, questions: m.questionsS118 })),
    },
    {
      id: '11.9', num: '11.9', title: 'Consolidating block segments for a global scan', pages: '273–282',
      lede: "Scan any length by stitching blocks together, count each design's bytes per element, and say on which backends the single-pass kernel is safe.",
      vi: { title: 'Hợp nhất các đoạn khối cho một scan toàn cục', lede: 'Scan độ dài bất kỳ bằng cách ghép các khối lại với nhau, đếm số byte mỗi phần tử của từng thiết kế, và nói được kernel một lượt an toàn trên những backend nào.' },
      load: () => import('./S119').then((m) => ({ Component: m.S119, questions: m.questionsS119 })),
    },
    {
      id: '11.10', num: '11.10', title: 'Parallel scan with the Brent-Kung algorithm', pages: '282–285',
      lede: "Trace Brent-Kung's up-sweep and down-sweep, count its 2N − 2 − log₂N additions, and argue why Kogge-Stone still wins inside a warp.",
      vi: { title: 'Scan song song với thuật toán Brent-Kung', lede: 'Lần theo lượt quét lên và quét xuống của Brent-Kung, đếm 2N − 2 − log₂N phép cộng của nó, và lập luận vì sao Kogge-Stone vẫn thắng bên trong một warp.' },
      load: () => import('./S1110').then((m) => ({ Component: m.S1110, questions: m.questionsS1110 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: "Six labs: block scans measured three ways, coarsening against your copy bandwidth, a race hunt, scans of any length (hierarchical everywhere, single-pass on CUDA only), and milestone M5's top-p mask.",
      vi: { title: 'Lab', lede: 'Sáu lab: scan khối đo theo ba cách, gộp việc so với băng thông sao chép của bạn, một cuộc săn race, scan độ dài bất kỳ (phân cấp ở mọi nơi, một lượt chỉ trên CUDA), và mặt nạ top-p của mốc dự án M5.' },
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '11.11', title: 'Summary and exercises', pages: '285–287',
      lede: "What to carry forward, a mixed check, where each exercise is worked, the PDF's apparent errata, and the M5 piece.",
      vi: { title: 'Tóm tắt và bài tập', lede: 'Những gì cần mang theo, một bài kiểm tra tổng hợp, nơi mỗi bài tập được giải, các lỗi in có vẻ có trong PDF, và phần việc của M5.' },
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
