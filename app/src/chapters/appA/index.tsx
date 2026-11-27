import type { Chapter } from '../types';

// Appendix A, built in the shape of chapter 5: one file per section, each loaded on
// demand and exporting its component and its questions.
export const chapter: Chapter = {
  id: 'appA',
  num: 'A',
  title: 'Numerical considerations',
  pages: '583–600',
  vi: { title: 'Các vấn đề về số học' },
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '583',
      lede: 'Say where rounding enters one step of LLM inference, what this appendix lets you decide about it, and check what it assumes from chapters 5, 10 and 20.',
      vi: { title: 'Trước khi bắt đầu', lede: 'Chỉ ra chỗ làm tròn xen vào một bước suy luận LLM, phụ lục này giúp bạn quyết định gì về nó, và kiểm tra những gì nó giả định từ các chương 5, 10 và 20.' },
      load: () => import('./SA0').then((m) => ({ Component: m.SA0, questions: m.questionsSA0 })),
    },
    {
      id: 'A.1', num: 'A.1', title: 'Floating-point data representation', pages: '583–586',
      lede: "Turn any bit pattern of FP32, FP16, BF16 or the book's toy formats into its value and back, and say what each field buys you.",
      vi: { title: 'Biểu diễn dữ liệu dấu phẩy động', lede: 'Chuyển bất kỳ mẫu bit nào của FP32, FP16, BF16 hoặc các định dạng thu nhỏ của sách thành giá trị của nó và ngược lại, và nói mỗi trường mang lại cho bạn điều gì.' },
      load: () => import('./SA1').then((m) => ({ Component: m.SA1, questions: m.questionsSA1 })),
    },
    {
      id: 'A.2', num: 'A.2', title: 'Representable numbers', pages: '586–590',
      lede: 'Draw every value a small format can hold, see where its gaps are, and predict the rounding error a real format makes near any value.',
      vi: { title: 'Các số biểu diễn được', lede: 'Vẽ mọi giá trị mà một định dạng nhỏ có thể chứa, thấy các khoảng trống của nó nằm ở đâu, và dự đoán sai số làm tròn mà một định dạng thật mắc phải gần bất kỳ giá trị nào.' },
      load: () => import('./SA2').then((m) => ({ Component: m.SA2, questions: m.questionsSA2 })),
    },
    {
      id: 'A.3', num: 'A.3', title: 'Precision and special bit patterns in IEEE format', pages: '590–591',
      lede: 'Classify any bit pattern as zero, denormal, normal, infinity or NaN, predict which operations produce infinities and NaNs, and keep a softmax from producing them.',
      vi: { title: 'Độ chính xác và các mẫu bit đặc biệt trong định dạng IEEE', lede: 'Phân loại bất kỳ mẫu bit nào là số không, denormal, normal, vô cực hay NaN, dự đoán phép toán nào sinh ra vô cực và NaN, và giữ cho một softmax không sinh ra chúng.' },
      load: () => import('./SA3').then((m) => ({ Component: m.SA3, questions: m.questionsSA3 })),
    },
    {
      id: 'A.4', num: 'A.4', title: 'Arithmetic accuracy and rounding', pages: '591–592',
      lede: 'Step an addition through alignment and rounding, measure its error in ULPs under each rounding mode, and say which operations on your GPU are correctly rounded.',
      vi: { title: 'Độ chính xác số học và làm tròn', lede: 'Đi từng bước một phép cộng qua căn chỉnh và làm tròn, đo sai số của nó theo ULP dưới mỗi chế độ làm tròn, và nói phép toán nào trên GPU của bạn được làm tròn đúng.' },
      load: () => import('./SA4').then((m) => ({ Component: m.SA4, questions: m.questionsSA4 })),
    },
    {
      id: 'A.5', num: 'A.5', title: 'Lower-precision formats', pages: '592–593',
      lede: 'Decode FP8 and FP4 values, quantise a block with a shared scale, and work out what fewer bits per weight buy a memory-bound decode step.',
      vi: { title: 'Các định dạng độ chính xác thấp hơn', lede: 'Giải mã các giá trị FP8 và FP4, lượng tử hóa một khối với một hệ số tỉ lệ dùng chung, và tính xem ít bit hơn cho mỗi trọng số mang lại gì cho một bước decode bị giới hạn bởi bộ nhớ.' },
      load: () => import('./SA5').then((m) => ({ Component: m.SA5, questions: m.questionsSA5 })),
    },
    {
      id: 'A.6', num: 'A.6', title: 'Algorithm considerations', pages: '593–594',
      lede: 'Predict how summation order and accumulator width change a long sum, and choose sorted, pairwise or compensated summation for a reduction you write.',
      vi: { title: 'Các cân nhắc về thuật toán', lede: 'Dự đoán thứ tự cộng và độ rộng bộ tích lũy thay đổi một tổng dài thế nào, và chọn phép cộng có sắp xếp, cộng theo cặp hay cộng có bù cho một phép rút gọn bạn viết.' },
      load: () => import('./SA6').then((m) => ({ Component: m.SA6, questions: m.questionsSA6 })),
    },
    {
      id: 'A.7', num: 'A.7', title: 'Linear solvers and numerical stability', pages: '594–599',
      lede: 'Step Gaussian elimination with one thread per row and its barriers, watch a zero or tiny pivot break it, and say why pivoting costs a max reduction per column.',
      vi: { title: 'Bộ giải tuyến tính và tính ổn định số học', lede: 'Đi từng bước phép khử Gauss với một luồng mỗi hàng cùng các rào chắn của nó, xem một phần tử chốt bằng không hoặc rất nhỏ làm hỏng nó, và nói vì sao việc chọn phần tử chốt tốn một phép rút gọn max cho mỗi cột.' },
      load: () => import('./SA7').then((m) => ({ Component: m.SA7, questions: m.questionsSA7 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: "Measure on your own GPU how summation order, accumulator width and compensation change a long sum, what your device's arithmetic really does, and what accuracy costs in time.",
      vi: { title: 'Lab', lede: 'Đo trên GPU của chính bạn thứ tự cộng, độ rộng bộ tích lũy và phép bù thay đổi một tổng dài thế nào, số học của thiết bị bạn thực sự làm gì, và độ chính xác tốn bao nhiêu thời gian.' },
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: 'A.8', title: 'Summary and exercises', pages: '599–600',
      lede: 'Carry forward what decides a floating-point result, check yourself across the appendix, see where each exercise is worked and where the PDF looks wrong, and take the accumulator decision on to X2.',
      vi: { title: 'Tóm tắt và bài tập', lede: 'Mang theo những gì quyết định một kết quả dấu phẩy động, tự kiểm tra trên toàn phụ lục, xem mỗi bài tập được giải ở đâu và chỗ nào PDF có vẻ sai, và mang quyết định về bộ tích lũy sang X2.' },
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
