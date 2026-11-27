import type { Chapter } from '../types';

// One file per section, each loaded on demand and exporting its component and its
// questions, in the shape of chapter 5 (the template).
export const chapter: Chapter = {
  id: 'ch03',
  num: '3',
  title: 'Multidimensional grids and data',
  pages: '45–66',
  vi: { title: 'Lưới và dữ liệu đa chiều' },
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '45',
      lede: 'What this chapter is for, the three kernels it builds, and a check on what it assumes from chapter 2.',
      vi: { title: 'Trước khi bắt đầu', lede: 'Chương này để làm gì, ba kernel mà nó xây dựng, và một bài kiểm tra những gì nó giả định bạn đã có từ chương 2.' },
      load: () => import('./S30').then((m) => ({ Component: m.S30, questions: m.questionsS30 })),
    },
    {
      id: '3.1', num: '3.1', title: 'Multidimensional grid organization', pages: '45–48',
      lede: 'Choose a legal dim3 launch for 1D, 2D or 3D work and say what every thread sees in its four built-ins.',
      vi: { title: 'Tổ chức lưới đa chiều', lede: 'Chọn một cấu hình khởi chạy dim3 hợp lệ cho công việc 1D, 2D hoặc 3D và nói mỗi luồng thấy gì trong bốn biến dựng sẵn của nó.' },
      load: () => import('./S31').then((m) => ({ Component: m.S31, questions: m.questionsS31 })),
    },
    {
      id: '3.2', num: '3.2', title: 'Mapping threads to multidimensional data', pages: '48–55',
      lede: 'Take any thread of a 2D or 3D grid to its row, column and flat offset, say which threads are surplus, and write the grayscale kernel correctly.',
      vi: { title: 'Ánh xạ luồng sang dữ liệu đa chiều', lede: 'Đưa bất kỳ luồng nào của một lưới 2D hoặc 3D về hàng, cột và offset phẳng của nó, nói những luồng nào là dư, và viết đúng kernel chuyển ảnh xám.' },
      load: () => import('./S32').then((m) => ({ Component: m.S32, questions: m.questionsS32 })),
    },
    {
      id: '3.3', num: '3.3', title: 'Image blur: a more complex kernel', pages: '55–59',
      lede: "Write a kernel whose threads loop over a neighbourhood, with one guard for surplus threads and another for the picture's edge, and say what each guard prevents.",
      vi: { title: 'Làm mờ ảnh: một kernel phức tạp hơn', lede: 'Viết một kernel mà các luồng lặp qua một vùng lân cận, với một phép kiểm tra cho luồng dư và một phép kiểm tra khác cho mép ảnh, và nói mỗi phép kiểm tra ngăn điều gì.' },
      load: () => import('./S33').then((m) => ({ Component: m.S33, questions: m.questionsS33 })),
    },
    {
      id: '3.4', num: '3.4', title: 'Matrix multiplication', pages: '59–64',
      lede: "Write the one-thread-per-element matmul kernel and its launch for any shape, trace any thread's reads by hand, and say which storage convention your kernel assumes.",
      vi: { title: 'Phép nhân ma trận', lede: 'Viết kernel nhân ma trận mỗi luồng một phần tử cùng cấu hình khởi chạy cho mọi kích thước, tự lần theo các lần đọc của bất kỳ luồng nào, và nói kernel của bạn giả định quy ước lưu trữ nào.' },
      load: () => import('./S34').then((m) => ({ Component: m.S34, questions: m.questionsS34 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Five labs on your own GPUs: grayscale, blur, the naive GEMM that is milestone M1, a thread-granularity measurement, and an index bug hunt.',
      vi: { title: 'Lab', lede: 'Năm lab trên GPU của chính bạn: ảnh xám, làm mờ, GEMM ngây thơ là mốc dự án M1, một phép đo độ mịn công việc của luồng, và một cuộc săn lỗi chỉ số.' },
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '3.5', title: 'Summary and exercises', pages: '64–65',
      lede: 'What to carry forward, a mixed check on the whole chapter, and where the PDF differs from tested code.',
      vi: { title: 'Tóm tắt và bài tập', lede: 'Những gì cần mang theo, một bài kiểm tra tổng hợp cho cả chương, và những chỗ PDF khác với mã đã kiểm thử.' },
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
