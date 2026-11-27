import type { Chapter } from '../types';

// Standard depth (PLAN.md). The chapter prints no code and no exercises; the course adds
// two labs, the second of which is milestone M8. One file per section, each loaded on
// demand and exporting its component and its questions.
export const chapter: Chapter = {
  id: 'ch22',
  num: '22',
  title: 'Algorithm selection, problem decomposition, and problem formulation',
  pages: '529–540',
  vi: { title: 'Chọn thuật toán, phân rã bài toán và đặt bài toán' },
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '529',
      lede: 'What this chapter is for, why it has no code of its own, and a check on the earlier chapters it leans on.',
      vi: { title: 'Trước khi bắt đầu', lede: 'Chương này để làm gì, vì sao nó không có mã riêng, và một bài kiểm tra các chương trước mà nó dựa vào.' },
      load: () => import('./S220').then((m) => ({ Component: m.S220, questions: m.questionsS220 })),
    },
    {
      id: '22.1', num: '22.1', title: 'Algorithm selection', pages: '530–531',
      lede: 'Choose between algorithms for one problem by naming what each trades (steps, exposed parallelism, generality, accuracy) on the hardware you actually have.',
      vi: { title: 'Chọn thuật toán', lede: 'Chọn giữa các thuật toán cho cùng một bài toán bằng cách nói rõ mỗi thuật toán đánh đổi gì (số bước, mức song song lộ ra, tính tổng quát, độ chính xác) trên phần cứng bạn thực sự có.' },
      load: () => import('./S221').then((m) => ({ Component: m.S221, questions: m.questionsS221 })),
    },
    {
      id: '22.2', num: '22.2', title: 'Problem decomposition', pages: '532–536',
      lede: 'Decide, for a new computation, whether each thread should own an output or an input, by asking who writes, who conflicts and what needs an atomic.',
      vi: { title: 'Phân rã bài toán', lede: 'Với một phép tính mới, quyết định mỗi luồng nên sở hữu một đầu ra hay một đầu vào, bằng cách hỏi ai ghi, ai xung đột và cái gì cần một phép toán nguyên tử (atomic).' },
      load: () => import('./S222').then((m) => ({ Component: m.S222, questions: m.questionsS222 })),
    },
    {
      id: '22.3', num: '22.3', title: "Application-level considerations: Amdahl's law", pages: '537–538',
      lede: 'Work out the speed-up a whole application gets when only some of its modules run on the GPU, with and without overlap, and what to do about the rest.',
      vi: { title: 'Các cân nhắc ở mức ứng dụng: định luật Amdahl', lede: 'Tính mức tăng tốc mà cả ứng dụng nhận được khi chỉ một số mô-đun của nó chạy trên GPU, có và không có chồng lấp, và phải làm gì với phần còn lại.' },
      load: () => import('./S223').then((m) => ({ Component: m.S223, questions: m.questionsS223 })),
    },
    {
      id: '22.4', num: '22.4', title: 'Problem formulation', pages: '538',
      lede: 'Recognise when re-posing the problem beats any amount of kernel tuning, and what the new formulation costs instead.',
      vi: { title: 'Đặt bài toán', lede: 'Nhận ra khi nào việc đặt lại bài toán thắng mọi công sức tinh chỉnh kernel, và cách đặt mới phải trả giá gì thay vào đó.' },
      load: () => import('./S224').then((m) => ({ Component: m.S224, questions: m.questionsS224 })),
    },
    {
      id: '22.5', num: '22.5', title: 'Batching: latency versus throughput', pages: '539–540',
      lede: 'Predict where a decode step stops being bandwidth-bound as the batch grows, and say what each user pays in latency for the throughput gained.',
      vi: { title: 'Batching: độ trễ đổi lấy thông lượng', lede: 'Dự đoán ở đâu một bước decode thôi bị giới hạn bởi băng thông khi batch lớn dần, và nói mỗi người dùng phải trả bao nhiêu độ trễ cho phần thông lượng có thêm.' },
      load: () => import('./S225').then((m) => ({ Component: m.S225, questions: m.questionsS225 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Two labs on your own GPUs: two decompositions of a decode-time matrix-vector product, and the batching study (milestone M8).',
      vi: { title: 'Lab', lede: 'Hai lab trên GPU của chính bạn: hai cách phân rã một phép nhân ma trận với vector lúc decode, và bài nghiên cứu batching (mốc dự án M8).' },
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '22.6', title: 'Summary', pages: '540',
      lede: "What to carry forward, a mixed check, the PDF's apparent slips, and where milestone M8 stands.",
      vi: { title: 'Tóm tắt', lede: 'Những gì cần mang theo, một bài kiểm tra tổng hợp, những chỗ có vẻ nhầm trong PDF, và mốc dự án M8 đang ở đâu.' },
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
