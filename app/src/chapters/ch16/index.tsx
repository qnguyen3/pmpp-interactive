import type { Chapter } from '../types';

// Condensed: two teaching pages (16.3 covers §16.1 to §16.4 and the synchronisation part of
// §16.8; 16.6 covers §16.5 to §16.8 on Smith-Waterman's table), an optional lab and a
// summary. Each section is loaded on demand and exports its component and its questions.
export const chapter: Chapter = {
  id: 'ch16',
  num: '16',
  title: 'Dynamic programming and wavefront parallelism',
  pages: '373–400',
  vi: { title: 'Quy hoạch động và song song theo wavefront' },
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '373–374',
      lede: 'Say what a wavefront is and why a recurrence along a sequence needs one, and check the ideas from chapters 2, 4, 5, 6 and 11 that this chapter builds on.',
      vi: { title: 'Trước khi bắt đầu', lede: 'Nói được wavefront là gì và vì sao một hệ thức truy hồi dọc theo một chuỗi cần đến nó, và kiểm tra lại các ý từ các chương 2, 4, 5, 6 và 11 mà chương này dựa vào.' },
      load: () => import('./S160').then((m) => ({ Component: m.S160, questions: m.questionsS160 })),
    },
    {
      id: '16.3', num: '16.3', title: 'Wavefronts, and who waits for whom', pages: '374–381, 397–399',
      lede: 'Number the cells of a table by the step at which they can run, say why a Floyd-Warshall step can update its table in place, and choose the cheapest synchronisation (a relaunch, a block barrier, flags, or none) that covers exactly the dependencies.',
      vi: { title: 'Wavefront, và ai chờ ai', lede: 'Đánh số các ô của một bảng theo bước mà chúng có thể chạy, nói được vì sao một bước Floyd-Warshall có thể cập nhật bảng của nó tại chỗ, và chọn cách đồng bộ hóa rẻ nhất (khởi chạy lại, rào chắn (barrier) của khối, cờ, hoặc không gì cả) phủ đúng các phụ thuộc.' },
      load: () => import('./S163').then((m) => ({ Component: m.S163, questions: m.questionsS163 })),
    },
    {
      id: '16.6', num: '16.6', title: 'Tiles, and the shear that fills them', pages: '382–398',
      lede: 'Map blocks to tiles and threads to cells along anti-diagonal fronts, shear a tile so every thread works at every step, and say what that costs in launches, parallelism and shared-memory banks.',
      vi: { title: 'Ô, và phép trượt lấp đầy chúng', lede: 'Ánh xạ khối vào ô (tile) và luồng vào ô con dọc theo các front chéo phụ, trượt một ô để mọi luồng đều làm việc ở mọi bước, và nói được điều đó tốn gì về số lần khởi chạy, độ song song và các bank của bộ nhớ chia sẻ (shared memory).' },
      load: () => import('./S166').then((m) => ({ Component: m.S166, questions: m.questionsS166 })),
    },
    {
      id: 'labs', num: '', title: 'Lab', pages: null,
      lede: 'One optional lab: fill a dynamic-programming table with square and sheared tiles, test every cell against a CPU reference on every backend, and measure launches, padding and tile width on your GPU.',
      vi: { title: 'Lab', lede: 'Một lab tùy chọn: điền một bảng quy hoạch động bằng ô vuông và ô đã trượt, kiểm tra từng ô con với bản tham chiếu CPU trên mọi backend, và đo số lần khởi chạy, phần đệm và độ rộng ô trên GPU của bạn.' },
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '16.9', title: 'Summary and exercises', pages: '399–400',
      lede: "What to carry forward, a mixed check, where each of the book's exercises is worked, the PDF's apparent slips, what this chapter left out, and why it has no project milestone.",
      vi: { title: 'Tóm tắt và bài tập', lede: 'Những gì cần mang theo, một bài kiểm tra tổng hợp, nơi từng bài tập của sách được giải, những chỗ có vẻ sai sót trong bản PDF, những gì chương này bỏ qua, và vì sao chương này không có mốc dự án.' },
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
