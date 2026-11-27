import type { Chapter } from '../types';

// One file per section, each loaded on demand and exporting its component and its
// questions (chapter 5 is the template).
export const chapter: Chapter = {
  id: 'ch20',
  num: '20',
  title: 'Large language models',
  pages: '477–512',
  vi: { title: 'Mô hình ngôn ngữ lớn' },
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '477–478',
      lede: 'See which of your milestones this chapter composes into attention, and check the reductions, intensity arithmetic and layouts it assumes you remember.',
      vi: { title: 'Trước khi bắt đầu', lede: 'Xem những mốc dự án nào của bạn được chương này ghép lại thành attention, và kiểm tra các phép rút gọn, phép tính cường độ và cách bố trí dữ liệu mà chương giả định bạn còn nhớ.' },
      load: () => import('./S200').then((m) => ({ Component: m.S200, questions: m.questionsS200 })),
    },
    {
      id: '20.1', num: '20.1', title: 'Transformer architecture', pages: '478–482',
      lede: 'Place every block of a decoder layer on a kernel you have written or will write, and say where its arithmetic goes as the context grows.',
      vi: { title: 'Kiến trúc Transformer', lede: 'Đặt mọi khối của một lớp decoder lên một kernel bạn đã viết hoặc sẽ viết, và nói được phép tính số học của nó đi về đâu khi ngữ cảnh dài ra.' },
      load: () => import('./S201').then((m) => ({ Component: m.S201, questions: m.questionsS201 })),
    },
    {
      id: '20.2', num: '20.2', title: 'Multi-head attention', pages: '482–486',
      lede: 'Write down the shape of every matrix in one causal attention head, say which entries the mask removes, and why subtracting the row max changes nothing but the risk of overflow.',
      vi: { title: 'Multi-head attention', lede: 'Viết ra kích thước của mọi ma trận trong một head attention nhân quả, nói được mặt nạ loại bỏ những phần tử nào, và vì sao trừ đi giá trị lớn nhất của hàng không thay đổi gì ngoài nguy cơ tràn số.' },
      load: () => import('./S202').then((m) => ({ Component: m.S202, questions: m.questionsS202 })),
    },
    {
      id: '20.3', num: '20.3', title: 'Implementing attention in CUDA', pages: '486–488',
      lede: 'Write a causal mask-and-softmax kernel with one block per row, say what each of its barriers protects, and count the global traffic the unfused pipeline pays for S and P.',
      vi: { title: 'Cài đặt attention bằng CUDA', lede: 'Viết một kernel che mặt nạ nhân quả và softmax với một khối cho mỗi hàng, nói được mỗi rào chắn của nó bảo vệ điều gì, và đếm lưu lượng bộ nhớ toàn cục mà pipeline không gộp phải trả cho S và P.' },
      load: () => import('./S203').then((m) => ({ Component: m.S203, questions: m.questionsS203 })),
    },
    {
      id: '20.4', num: '20.4', title: 'KV caching', pages: '488–492',
      lede: 'Predict exactly which entries change when one token is added, and explain why a decode step is a bandwidth-bound matrix-vector pass over the whole cache while prefill is a compute-bound GEMM.',
      vi: { title: 'KV caching', lede: 'Dự đoán chính xác những phần tử nào thay đổi khi thêm một token, và giải thích vì sao một bước decode là một lượt nhân ma trận-vectơ bị giới hạn bởi băng thông trên toàn bộ cache, còn prefill là một GEMM bị giới hạn bởi tính toán.' },
      load: () => import('./S204').then((m) => ({ Component: m.S204, questions: m.questionsS204 })),
    },
    {
      id: '20.5', num: '20.5', title: 'Flash attention', pages: '492–504',
      lede: 'Merge softmax results tile by tile with a running max and denominator, and write the single-kernel forward pass: who owns which data, which barriers it needs, what the book’s listing gets wrong, and what changes on HIP and Metal.',
      vi: { title: 'Flash attention', lede: 'Trộn kết quả softmax từng ô một với giá trị lớn nhất và mẫu số đang cập nhật, và viết lượt forward trong một kernel duy nhất: ai sở hữu dữ liệu nào, cần những rào chắn nào, listing của sách sai ở đâu, và điều gì thay đổi trên HIP và Metal.' },
      load: () => import('./S205').then((m) => ({ Component: m.S205, questions: m.questionsS205 })),
    },
    {
      id: '20.6', num: '20.6', title: 'KV cache arithmetic intensity and memory requirement', pages: '504–508',
      lede: 'Size a KV cache for any model, batch and precision against your device’s memory, and compute why batching lifts the projections’ intensity but not decode attention’s.',
      vi: { title: 'Cường độ tính toán và yêu cầu bộ nhớ của KV cache', lede: 'Định kích thước KV cache cho bất kỳ mô hình, batch và độ chính xác nào so với bộ nhớ của thiết bị bạn, và tính được vì sao batching nâng cường độ của các phép chiếu nhưng không nâng cường độ của attention khi decode.' },
      load: () => import('./S206').then((m) => ({ Component: m.S206, questions: m.questionsS206 })),
    },
    {
      id: '20.7', num: '20.7', title: 'Alleviating the memory requirements of the attention mechanism', pages: '508–510',
      lede: 'Compare MHA, GQA, MQA and MLA by cache size and decode intensity with the definitions straightened out, and explain how paging removes reserved but unused cache memory.',
      vi: { title: 'Giảm yêu cầu bộ nhớ của cơ chế attention', lede: 'So sánh MHA, GQA, MQA và MLA theo kích thước cache và cường độ khi decode, với các định nghĩa được làm rõ, và giải thích cách phân trang loại bỏ phần bộ nhớ cache đã giữ chỗ nhưng không dùng.' },
      load: () => import('./S207').then((m) => ({ Component: m.S207, questions: m.questionsS207 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Five labs: build milestones M6 and M7 one kernel at a time, hunt the race the book’s flash kernel has, and measure decode against the intensity formula on your devices.',
      vi: { title: 'Lab', lede: 'Năm lab: xây các mốc dự án M6 và M7 từng kernel một, săn race condition mà kernel flash của sách mắc phải, và đo decode so với công thức cường độ trên các thiết bị của bạn.' },
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '20.8', title: 'Summary and exercises', pages: '510–511',
      lede: 'What to carry forward, a mixed check, where the book’s exercises and errata are worked, and milestones M6 and M7.',
      vi: { title: 'Tóm tắt và bài tập', lede: 'Những gì cần mang theo, một bài kiểm tra tổng hợp, nơi các bài tập và errata của sách được giải, và các mốc dự án M6 và M7.' },
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
