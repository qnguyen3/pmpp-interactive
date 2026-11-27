import type { Chapter } from '../types';

// Chapter 5 is the template for every other chapter: one file per section, each
// loaded on demand and exporting its component and its questions.
export const chapter: Chapter = {
  id: 'ch05',
  num: '5',
  title: 'Memory architecture and data locality',
  pages: '93–121',
  vi: { title: 'Kiến trúc bộ nhớ và tính cục bộ dữ liệu' },
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '93',
      lede: 'What this chapter is for, the kernel it takes apart, and a check on what it assumes you remember.',
      vi: { title: 'Trước khi bắt đầu', lede: 'Chương này để làm gì, kernel mà nó mổ xẻ, và một bài kiểm tra những gì nó giả định bạn còn nhớ.' },
      load: () => import('./S50').then((m) => ({ Component: m.S50, questions: m.questions50 })),
    },
    {
      id: '5.1', num: '5.1', title: 'Memory bandwidth as a performance limiter', pages: '93–97',
      lede: 'Tell whether a kernel is held back by arithmetic or by memory, and by how much.',
      vi: { title: 'Băng thông bộ nhớ như một giới hạn hiệu năng', lede: 'Biết một kernel đang bị kìm bởi phép tính số học hay bởi bộ nhớ, và kìm bao nhiêu.' },
      load: () => import('./S51').then((m) => ({ Component: m.S51, questions: m.questions51 })),
    },
    {
      id: '5.2', num: '5.2', title: 'Where data can live', pages: '98–104',
      lede: 'Say which memory each variable of a kernel lands in, who can see it, and how many copies exist.',
      vi: { title: 'Dữ liệu có thể nằm ở đâu', lede: 'Nói được mỗi biến của kernel nằm ở bộ nhớ nào, ai thấy được nó, và có bao nhiêu bản sao.' },
      load: () => import('./S52').then((m) => ({ Component: m.S52, questions: m.questionsS52 })),
    },
    {
      id: '5.3', num: '5.3', title: 'Tiling for reduced memory traffic', pages: '104–107',
      lede: 'See the redundant fetches in the naive kernel and how sharing them inside a block removes them.',
      vi: { title: 'Phân ô để giảm lưu lượng bộ nhớ', lede: 'Thấy những lần nạp thừa trong kernel ngây thơ và cách chia sẻ chúng trong một khối loại bỏ chúng.' },
      load: () => import('./S53').then((m) => ({ Component: m.S53, questions: m.questionsS53 })),
    },
    {
      id: '5.4', num: '5.4', title: 'A tiled matrix multiplication kernel', pages: '108–112',
      lede: 'Write the tiled kernel, place its two barriers, and say what each one protects.',
      vi: { title: 'Một kernel nhân ma trận phân ô', lede: 'Viết kernel phân ô, đặt hai rào chắn của nó, và nói mỗi rào chắn bảo vệ điều gì.' },
      load: () => import('./S54').then((m) => ({ Component: m.S54, questions: m.questionsS54 })),
    },
    {
      id: '5.5', num: '5.5', title: 'Boundary checks', pages: '112–115',
      lede: 'Make the kernel correct for any size and for rectangular matrices.',
      vi: { title: 'Kiểm tra biên', lede: 'Làm cho kernel đúng với mọi kích thước và với ma trận chữ nhật.' },
      load: () => import('./S55').then((m) => ({ Component: m.S55, questions: m.questionsS55 })),
    },
    {
      id: '5.6', num: '5.6', title: 'Memory use and occupancy', pages: '115–118',
      lede: 'Budget shared memory against the number of threads a device can keep resident.',
      vi: { title: 'Sử dụng bộ nhớ và occupancy', lede: 'Cân đối bộ nhớ chia sẻ với số luồng mà một thiết bị có thể giữ thường trú.' },
      load: () => import('./S56').then((m) => ({ Component: m.S56, questions: m.questionsS56 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Five labs on your own GPUs: measure, tile, generalise, debug, and size shared memory at run time.',
      vi: { title: 'Lab', lede: 'Năm lab trên GPU của chính bạn: đo, phân ô, tổng quát hóa, gỡ lỗi, và định kích thước bộ nhớ chia sẻ lúc chạy.' },
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '5.7', title: 'Summary and exercises', pages: '118–121',
      lede: 'What to carry forward, and a mixed check on the whole chapter.',
      vi: { title: 'Tóm tắt và bài tập', lede: 'Những gì cần mang theo, và một bài kiểm tra tổng hợp cho cả chương.' },
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
