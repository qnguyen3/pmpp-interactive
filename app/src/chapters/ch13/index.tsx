import type { Chapter } from '../types';

// Condensed: one teaching page (13.3 covers §13.3 to §13.8 of the book), an optional
// lab and a summary. Each section is loaded on demand and exports its component and
// its questions.
export const chapter: Chapter = {
  id: 'ch13',
  num: '13',
  title: 'Merge',
  pages: '303–328',
  vi: { title: 'Trộn (merge)' },
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '303–306',
      lede: 'Say why merge is the first pattern where a thread cannot work out its inputs from its index, and check the ideas from chapters 2, 5 and 6 that this chapter depends on.',
      vi: { title: 'Trước khi bắt đầu', lede: 'Nói được vì sao trộn (merge) là mẫu đầu tiên mà một luồng không thể suy ra đầu vào của mình từ chỉ số của nó, và kiểm tra các ý từ chương 2, 5 và 6 mà chương này dựa vào.' },
      load: () => import('./S130').then((m) => ({ Component: m.S130, questions: m.questionsS130 })),
    },
    {
      id: '13.3', num: '13.3', title: 'Co-rank: split the output, search for the inputs', pages: '306–327',
      lede: 'Use a binary search to find where any slice of the merged output begins in each input, use that to give every thread an equal share of a merge, and say what staging the inputs in shared memory buys and what it wastes.',
      vi: { title: 'Co-rank: chia đầu ra, tìm đầu vào', lede: 'Dùng tìm kiếm nhị phân để tìm nơi bất kỳ lát nào của đầu ra đã trộn bắt đầu trong mỗi đầu vào, dùng điều đó để chia cho mọi luồng một phần bằng nhau của phép trộn, và nói được việc đưa đầu vào vào bộ nhớ chia sẻ (shared memory) mang lại gì và lãng phí gì.' },
      load: () => import('./S133').then((m) => ({ Component: m.S133, questions: m.questionsS133 })),
    },
    {
      id: 'labs', num: '', title: 'Lab', pages: null,
      lede: 'One optional lab: merge two sorted arrays of key–value pairs with co-rank partitioning, first straight from global memory and then through shared memory, tested for stability against std::merge on every backend.',
      vi: { title: 'Lab', lede: 'Một lab tùy chọn: trộn hai mảng đã sắp xếp gồm các cặp khóa–giá trị bằng cách phân chia theo co-rank, trước tiên đọc thẳng từ bộ nhớ toàn cục (global memory) rồi qua bộ nhớ chia sẻ, được kiểm tra tính ổn định so với std::merge trên mọi backend.' },
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '13.9', title: 'Summary and exercises', pages: '327–328',
      lede: "What to carry forward, a mixed check, where each of the book's four exercises is worked, the PDF's apparent slips (most of them in the circular-buffer listings this course leaves out), and why the chapter has no project milestone.",
      vi: { title: 'Tóm tắt và bài tập', lede: 'Những gì cần mang theo, một bài kiểm tra tổng hợp, nơi mỗi bài trong bốn bài tập của sách được giải, những chỗ có vẻ nhầm của bản PDF (phần lớn nằm trong các đoạn mã buffer vòng mà khóa học này bỏ qua), và vì sao chương này không có mốc dự án.' },
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
