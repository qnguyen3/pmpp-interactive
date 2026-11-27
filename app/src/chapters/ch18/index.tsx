import type { Chapter } from '../types';

// Chapter 18, condensed: a start page, two section pages (18.3 covers §18.3 to §18.5, 18.6
// covers §18.6 to §18.8), one optional lab and the summary. Each section loads on demand.
export const chapter: Chapter = {
  id: 'ch18',
  num: '18',
  title: 'Graph traversal',
  pages: '425–451',
  vi: { title: 'Duyệt đồ thị' },
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '425–431',
      lede: "Say what this chapter adds to an inference engineer's toolkit, read a graph stored as CSR, CSC or COO, label BFS levels by hand, and check what the chapter assumes from chapters 4, 9, 11, 12 and 17.",
      vi: { title: 'Trước khi bắt đầu', lede: 'Nói được chương này bổ sung gì vào bộ công cụ của một kỹ sư suy luận, đọc một đồ thị lưu dưới dạng CSR, CSC hoặc COO, tự tay gán nhãn các mức BFS, và kiểm tra những gì chương này giả định từ các chương 4, 9, 11, 12 và 17.' },
      load: () => import('./S180').then((m) => ({ Component: m.S180, questions: m.questionsS180 })),
    },
    {
      id: '18.3', num: '18.3', title: 'Vertices, edges or a frontier', pages: '431–442',
      lede: 'Run one BFS level four ways on four small graphs, count the work each does, and make a vertex join the next frontier exactly once with compare-and-swap on CUDA, HIP and Metal.',
      vi: { title: 'Đỉnh, cạnh hay một frontier', lede: 'Chạy một mức BFS theo bốn cách trên bốn đồ thị nhỏ, đếm khối lượng việc mỗi cách làm, và làm cho một đỉnh vào frontier kế tiếp đúng một lần bằng compare-and-swap trên CUDA, HIP và Metal.' },
      load: () => import('./S183').then((m) => ({ Component: m.S183, questions: m.questionsS183 })),
    },
    {
      id: '18.6', num: '18.6', title: 'Private queues and one launch for every level', pages: '442–449',
      lede: "Build each block's share of the next frontier in shared memory with an overflow path and one global commit, fix the book's one racy line, and run every level in a single cooperative launch on CUDA and HIP, knowing what Metal does instead.",
      vi: { title: 'Hàng đợi riêng và một lần khởi chạy cho mọi mức', lede: 'Dựng phần frontier kế tiếp của mỗi khối trong bộ nhớ chia sẻ (shared memory), có đường xử lý tràn và một lần ghi chung ra bộ nhớ toàn cục, sửa một dòng có race condition của sách, và chạy mọi mức trong một lần khởi chạy cooperative duy nhất trên CUDA và HIP, đồng thời biết Metal làm gì thay thế.' },
      load: () => import('./S186').then((m) => ({ Component: m.S186, questions: m.questionsS186 })),
    },
    {
      id: 'labs', num: '', title: 'Lab', pages: null,
      lede: 'One optional lab: a frontier BFS with a privatised queue, tested against a CPU BFS on every backend and under the race detector, then timed against the fixed cost of one launch per level; on CUDA and HIP, the single-launch version too.',
      vi: { title: 'Lab', lede: 'Một lab tùy chọn: BFS theo frontier với hàng đợi được riêng tư hóa, kiểm tra so với BFS trên CPU ở mọi backend và dưới trình phát hiện race, rồi đo thời gian so với chi phí cố định của một lần khởi chạy cho mỗi mức; trên CUDA và HIP, có cả phiên bản một lần khởi chạy.' },
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '18.9', title: 'Summary and exercises', pages: '449–451',
      lede: 'What to carry forward, a mixed check, where the book\'s three exercises are worked, the PDF\'s apparent slips, and why the chapter has no project milestone.',
      vi: { title: 'Tóm tắt và bài tập', lede: 'Những gì cần mang theo, một bài kiểm tra tổng hợp, nơi ba bài tập của sách được giải, những chỗ có vẻ sai sót trong bản PDF, và vì sao chương này không có mốc dự án.' },
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
