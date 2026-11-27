import type { Chapter } from '../types';

// Module X2 is not in the book: every section is ours, so no section carries book pages.
// Built in the shape of chapter 5: one file per section, each loaded on demand and
// exporting its component and its questions.
export const chapter: Chapter = {
  id: 'x2',
  num: 'X2',
  title: 'Quantised kernels',
  pages: 'not in the book',
  vi: { title: 'Kernel lượng tử hóa' },
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: null,
      lede: 'See which kernel runs under a quantised linear layer in torchao, MLX and llama.cpp, what this module builds on your GPUs, and check what it assumes from chapters 5, 6, 10, 20, 22 and appendix A.',
      vi: { title: 'Trước khi bắt đầu', lede: 'Xem kernel nào chạy bên dưới một lớp tuyến tính lượng tử hóa trong torchao, MLX và llama.cpp, module này xây dựng gì trên GPU của bạn, và kiểm tra những gì nó giả định từ các chương 5, 6, 10, 20, 22 và phụ lục A.' },
      load: () => import('./SX20').then((m) => ({ Component: m.SX20, questions: m.questionsSX20 })),
    },
    {
      id: 'X2.1', num: 'X2.1', title: 'Bytes per weight set the speed of decode', pages: null,
      lede: 'Predict a decode GEMV’s speed-up from the bytes each weight format moves, see why a separate dequantise step makes it slower than float32, and turn your own measurement into a fraction of your bandwidth.',
      vi: { title: 'Số byte mỗi trọng số quyết định tốc độ decode', lede: 'Dự đoán mức tăng tốc của một GEMV ở bước decode từ số byte mà mỗi định dạng trọng số di chuyển, thấy vì sao một bước giải lượng tử hóa riêng làm nó chậm hơn float32, và biến phép đo của chính bạn thành một phần băng thông của bạn.' },
      load: () => import('./SX21').then((m) => ({ Component: m.SX21, questions: m.questionsSX21 })),
    },
    {
      id: 'X2.2', num: 'X2.2', title: 'Scales, zero points and blocks', pages: null,
      lede: 'Quantise a block of weights by the rules llama.cpp and MLX really use, predict which values clip, and find how block size, bit width and outliers trade error against bytes.',
      vi: { title: 'Hệ số tỉ lệ, điểm không và khối', lede: 'Lượng tử hóa một khối trọng số theo đúng các quy tắc llama.cpp và MLX dùng, dự đoán giá trị nào bị chặn, và tìm cách kích thước khối, độ rộng bit và giá trị ngoại lai đánh đổi sai số với số byte.' },
      load: () => import('./SX22').then((m) => ({ Component: m.SX22, questions: m.questionsSX22 })),
    },
    {
      id: 'X2.3', num: 'X2.3', title: 'Two codes per byte', pages: null,
      lede: 'Pack and unpack 4-bit codes in the orders llama.cpp and MLX really use, read an 18-byte block without a misaligned load, and say what each backend gives you for 4-bit data.',
      vi: { title: 'Hai mã mỗi byte', lede: 'Đóng gói và giải nén mã 4 bit theo đúng thứ tự llama.cpp và MLX dùng, đọc một khối 18 byte mà không có lần nạp lệch căn chỉnh nào, và nói mỗi backend cho bạn gì với dữ liệu 4 bit.' },
      load: () => import('./SX23').then((m) => ({ Component: m.SX23, questions: m.questionsSX23 })),
    },
    {
      id: 'X2.4', num: 'X2.4', title: 'A quantised GEMV, warp by warp', pages: null,
      lede: 'Map warps, lanes and loads onto rows of quantised blocks, watch which mappings coalesce and which lanes sit idle at the end of a row, and read the kernel lab X2.2 asks for.',
      vi: { title: 'Một GEMV lượng tử hóa, từng warp một', lede: 'Ánh xạ warp, lane và các lần nạp lên các hàng khối lượng tử hóa, xem cách ánh xạ nào được hợp nhất và lane nào ngồi không ở cuối hàng, và đọc kernel mà lab X2.2 yêu cầu.' },
      load: () => import('./SX24').then((m) => ({ Component: m.SX24, questions: m.questionsSX24 })),
    },
    {
      id: 'X2.5', num: 'X2.5', title: 'Dequantise, or multiply in integers', pages: null,
      lede: 'Follow one block’s dot product through the float path real Metal kernels use and the integer path llama.cpp uses on CUDA and HIP, prove the offset folds out of the loop, and say what each backend offers for a byte dot product.',
      vi: { title: 'Giải lượng tử hóa, hay nhân bằng số nguyên', lede: 'Theo dõi tích vô hướng của một khối qua đường số thực mà các kernel Metal thật dùng và đường số nguyên mà llama.cpp dùng trên CUDA và HIP, chứng minh phần bù được tách ra khỏi vòng lặp, và nói mỗi backend cung cấp gì cho tích vô hướng theo byte.' },
      load: () => import('./SX25').then((m) => ({ Component: m.SX25, questions: m.questionsSX25 })),
    },
    {
      id: 'X2.6', num: 'X2.6', title: 'Where the accumulation must stay wide', pages: null,
      lede: 'Bound every partial sum of a quantised dot product, choose the accumulator for each level, and watch the narrow choices wrap, overflow or drift.',
      vi: { title: 'Nơi phép cộng dồn phải giữ độ rộng lớn', lede: 'Chặn trên mọi tổng riêng phần của một tích vô hướng lượng tử hóa, chọn bộ cộng dồn cho từng cấp, và xem các lựa chọn hẹp bị quay vòng, tràn hoặc trôi sai số.' },
      load: () => import('./SX26').then((m) => ({ Component: m.SX26, questions: m.questionsSX26 })),
    },
    {
      id: 'X2.7', num: 'X2.7', title: 'The quantised decode step', pages: null,
      lede: 'Put quantised linear layers into milestone M8’s decode step, predict the speed-up from the bytes of the whole step, and judge the error by logits and greedy tokens instead of by weights.',
      vi: { title: 'Bước decode lượng tử hóa', lede: 'Đưa các lớp tuyến tính lượng tử hóa vào bước decode của mốc dự án M8, dự đoán mức tăng tốc từ số byte của cả bước, và đánh giá sai số qua logits và token greedy thay vì qua trọng số.' },
      load: () => import('./SX27').then((m) => ({ Component: m.SX27, questions: m.questionsSX27 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Four labs on your own GPUs: an int8-weight GEMV against float32, a 4-bit GEMV on llama.cpp’s Q4_0 blocks, a hunt for four defects on the integer path, and milestone M9, the quantised decode step.',
      vi: { title: 'Lab', lede: 'Bốn lab trên GPU của chính bạn: một GEMV trọng số int8 so với float32, một GEMV 4 bit trên các khối Q4_0 của llama.cpp, một cuộc săn bốn lỗi cài sẵn trên đường số nguyên, và mốc dự án M9, bước decode lượng tử hóa.' },
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '', title: 'Summary', pages: null,
      lede: 'What to carry forward, a mixed check, where each task of the module is worked, where the sources disagree, and milestone M9.',
      vi: { title: 'Tóm tắt', lede: 'Những gì cần mang theo, một bài kiểm tra tổng hợp, mỗi nhiệm vụ của module được làm ở đâu, các nguồn bất đồng ở đâu, và mốc dự án M9.' },
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
