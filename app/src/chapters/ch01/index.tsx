import type { Chapter } from '../types';

// A condensed chapter: the book's chapter 1 has no code and no exercises, so it has
// no labs page. One file per section, each loaded on demand and exporting its
// component and its questions (chapter 5 is the template).
export const chapter: Chapter = {
  id: 'ch01',
  num: '1',
  title: 'Introduction',
  pages: '1–20',
  vi: { title: 'Giới thiệu' },
  sections: [
    {
      id: 'start', num: '', title: 'Start here: one budget, two designs', pages: '1–7',
      lede: 'Say why a GPU beats a CPU on many independent operations and loses on one dependent chain, and predict which design finishes a given workload first.',
      vi: { title: 'Bắt đầu: một ngân sách, hai thiết kế', lede: 'Giải thích vì sao GPU thắng CPU trên nhiều phép toán độc lập nhưng thua trên một chuỗi phụ thuộc, và dự đoán thiết kế nào hoàn thành một khối lượng công việc cho trước trước.' },
      load: () => import('./S10').then((m) => ({ Component: m.S10, questions: m.questions10 })),
    },
    {
      id: '1.3', num: '1.3', title: 'Speed-up and what limits it', pages: '7–12',
      lede: 'Put a ceiling on the end-to-end gain from a faster kernel before you write it, and name the four difficulties the rest of the book keeps returning to.',
      vi: { title: 'Tăng tốc và những gì giới hạn nó', lede: 'Đặt trần cho mức lợi đầu cuối từ một kernel nhanh hơn trước khi bạn viết nó, và gọi tên bốn khó khăn mà phần còn lại của sách liên tục quay lại.' },
      load: () => import('./S13').then((m) => ({ Component: m.S13, questions: m.questions13 })),
    },
    {
      id: '1.7', num: '1.7', title: 'The book, this edition, and your path through it', pages: 'xvii–xxvi, 12–18',
      lede: 'Plan the order in which you read the book for your goals, and tell whether a page you find online is talking about this edition.',
      vi: { title: 'Sách, ấn bản này, và lộ trình của bạn qua nó', lede: 'Lên kế hoạch thứ tự đọc sách theo mục tiêu của bạn, và nhận ra một trang bạn tìm thấy trên mạng có đang nói về ấn bản này hay không.' },
      load: () => import('./S17').then((m) => ({ Component: m.S17, questions: m.questions17 })),
    },
    {
      id: 'wrap', num: '', title: 'Summary and check', pages: '1–18',
      lede: "What to carry into chapter 2, a mixed check on the whole chapter, and the places where the book's first chapter is dated or the PDF slips.",
      vi: { title: 'Tóm tắt và kiểm tra', lede: 'Những gì cần mang sang chương 2, một bài kiểm tra tổng hợp cho cả chương, và những chỗ chương đầu của sách đã lỗi thời hoặc bản PDF bị sai.' },
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
