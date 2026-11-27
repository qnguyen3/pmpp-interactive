import type { Chapter } from '../types';

// One file per section, each loaded on demand and exporting its component and its
// questions (the shape of chapter 5, the template).
export const chapter: Chapter = {
  id: 'ch12',
  num: '12',
  title: 'Filter',
  pages: '289–302',
  vi: { title: 'Lọc' },
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '289',
      lede: 'See where compaction sits in an LLM sampler, why frameworks avoid it, and check what the chapter assumes from chapters 4, 6, 9, 10 and 11.',
      vi: { title: 'Trước khi bắt đầu', lede: 'Xem nén gọn (compaction) nằm ở đâu trong một bộ lấy mẫu của LLM, vì sao các framework tránh nó, và kiểm tra những gì chương này giả định từ các chương 4, 6, 9, 10 và 11.' },
      load: () => import('./S120').then((m) => ({ Component: m.S120, questions: m.questionsS120 })),
    },
    {
      id: '12.1', num: '12.1', title: 'Background', pages: '289–290',
      lede: 'Say what a filter must produce, and decide for a given job whether it has to keep input order and whether it can work in place.',
      vi: { title: 'Bối cảnh', lede: 'Nói được một bộ lọc phải tạo ra gì, và với một việc cụ thể, quyết định nó có phải giữ thứ tự đầu vào hay không và có thể làm tại chỗ hay không.' },
      load: () => import('./S121').then((m) => ({ Component: m.S121, questions: m.questionsS121 })),
    },
    {
      id: '12.2', num: '12.2', title: 'A simple parallel unstable filter', pages: '290–291',
      lede: 'Write the one-atomic-per-key filter, explain why its output order changes between runs, and estimate how many atomics its single counter has to absorb.',
      vi: { title: 'Một bộ lọc song song không ổn định đơn giản', lede: 'Viết bộ lọc dùng một atomic cho mỗi khóa, giải thích vì sao thứ tự đầu ra của nó thay đổi giữa các lần chạy, và ước lượng bộ đếm duy nhất của nó phải hấp thụ bao nhiêu atomic.' },
      load: () => import('./S122').then((m) => ({ Component: m.S122, questions: m.questionsS122 })),
    },
    {
      id: '12.3', num: '12.3', title: 'Coalescing atomic operations with warp-level primitives', pages: '291–295',
      lede: 'Give every passing lane its output slot with one vote, one population count and one atomic per warp, and write it so the same idea compiles for CUDA, HIP and Metal.',
      vi: { title: 'Gộp các phép toán nguyên tử bằng các primitive cấp warp', lede: 'Cấp cho mỗi lane vượt qua ô đầu ra của nó chỉ bằng một lần bỏ phiếu, một lần đếm bit và một atomic cho mỗi warp, và viết sao cho cùng một ý tưởng biên dịch được cho CUDA, HIP và Metal.' },
      load: () => import('./S123').then((m) => ({ Component: m.S123, questions: m.questionsS123 })),
    },
    {
      id: '12.4', num: '12.4', title: 'Privatization', pages: '295–297',
      lede: 'Give each block a private list and counter in shared memory, place the three barriers, and commit the block\'s survivors with one atomic and coalesced stores.',
      vi: { title: 'Riêng tư hóa', lede: 'Cấp cho mỗi khối một danh sách và một bộ đếm riêng trong bộ nhớ chia sẻ (shared memory), đặt ba rào chắn (barrier), và ghi những phần tử còn lại của khối bằng một atomic và các lần ghi được hợp nhất.' },
      load: () => import('./S124').then((m) => ({ Component: m.S124, questions: m.questionsS124 })),
    },
    {
      id: '12.5', num: '12.5', title: 'A simple parallel stable filter', pages: '297–298',
      lede: 'Turn keep flags into output slots with an exclusive scan, compute a block\'s slots from warp votes, and say why a stable filter cannot take its slot from a counter.',
      vi: { title: 'Một bộ lọc song song ổn định đơn giản', lede: 'Biến các cờ giữ lại thành ô đầu ra bằng một scan loại trừ, tính các ô của một khối từ các lần bỏ phiếu của warp, và nói vì sao một bộ lọc ổn định không thể lấy ô của nó từ một bộ đếm.' },
      load: () => import('./S125').then((m) => ({ Component: m.S125, questions: m.questionsS125 })),
    },
    {
      id: '12.6', num: '12.6', title: 'Improving memory coalescing with shared memory and thread coarsening', pages: '298–300',
      lede: 'Stage each block\'s survivors in shared memory so consecutive threads write consecutive slots, and pick a coarsening factor from what it does to chunk length, store instructions and the scan.',
      vi: { title: 'Cải thiện hợp nhất truy cập bộ nhớ bằng bộ nhớ chia sẻ và gộp việc cho luồng', lede: 'Gom tạm các phần tử còn lại của mỗi khối trong bộ nhớ chia sẻ để các luồng liên tiếp ghi vào các ô liên tiếp, và chọn hệ số gộp việc dựa trên tác động của nó lên độ dài đoạn, số lệnh ghi và phép scan.' },
      load: () => import('./S126').then((m) => ({ Component: m.S126, questions: m.questionsS126 })),
    },
    {
      id: '12.7', num: '12.7', title: 'In-place stable filter', pages: '300',
      lede: 'State the ordering an in-place filter needs, show why the single-pass scan provides it on CUDA, and choose a design that is documented-safe on HIP and Metal.',
      vi: { title: 'Bộ lọc ổn định tại chỗ', lede: 'Nêu thứ tự mà một bộ lọc tại chỗ cần, chỉ ra vì sao scan một lượt cung cấp nó trên CUDA, và chọn một thiết kế được tài liệu bảo đảm an toàn trên HIP và Metal.' },
      load: () => import('./S127').then((m) => ({ Component: m.S127, questions: m.questionsS127 })),
    },
    {
      id: '12.8', num: '12.8', title: 'Related patterns', pages: '301',
      lede: 'Recognise duplicate removal and row or column removal and insertion as one-directional data movement, compute positions without a scan when the index decides, and say which way the in-place ordering runs.',
      vi: { title: 'Các mẫu liên quan', lede: 'Nhận ra việc loại bỏ phần tử trùng lặp và việc xóa hay chèn hàng hoặc cột đều là di chuyển dữ liệu một chiều, tính vị trí mà không cần scan khi chỉ số quyết định, và nói thứ tự tại chỗ chạy theo chiều nào.' },
      load: () => import('./S128').then((m) => ({ Component: m.S128, questions: m.questionsS128 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Five labs on your own GPUs: the unstable filter three ways across a sweep of fractions, a filter hunt, a stable filter by scan, milestone M5\'s candidate compaction, and an optional in-place single pass.',
      vi: { title: 'Lab', lede: 'Năm lab trên GPU của chính bạn: bộ lọc không ổn định theo ba cách trên một dải tỉ lệ, một cuộc săn lỗi trong bộ lọc, một bộ lọc ổn định bằng scan, phần nén gọn ứng viên của mốc dự án M5, và một bản một lượt tại chỗ tùy chọn.' },
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '12.9', title: 'Summary and exercises', pages: '302',
      lede: 'What to carry forward, a mixed check on the whole chapter, where each exercise is worked, the PDF\'s apparent errata, and the M5 piece.',
      vi: { title: 'Tóm tắt và bài tập', lede: 'Những gì cần mang theo, một bài kiểm tra tổng hợp cho cả chương, nơi mỗi bài tập được giải, những lỗi in có vẻ có trong bản PDF, và phần việc của M5.' },
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
