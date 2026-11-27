import type { Chapter } from '../types';

// Module X1 is not in the book: every section is ours, so no section carries book pages.
// Built in the shape of chapter 5: one file per section, each loaded on demand and
// exporting its component and its questions.
export const chapter: Chapter = {
  id: 'x1',
  num: 'X1',
  title: 'Sampling kernels',
  pages: 'not in the book',
  vi: { title: 'Các kernel lấy mẫu' },
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: null,
      lede: 'See where the sampler sits after the model’s last matmul, which earlier lab supplies each of its stages, and check what this module assumes from chapters 9 to 14.',
      vi: { title: 'Trước khi bắt đầu', lede: 'Xem sampler nằm ở đâu sau phép nhân ma trận cuối cùng của mô hình, lab nào trước đó cung cấp từng giai đoạn của nó, và kiểm tra những gì module này giả định từ chương 9 đến 14.' },
      load: () => import('./SX10').then((m) => ({ Component: m.SX10, questions: m.questionsSX10 })),
    },
    {
      id: 'X1.1', num: 'X1.1', title: 'The sampler chain', pages: null,
      lede: 'Name every stage between logits and a token, say which stages depend on their order, and state the order and tie rule that milestone M5 implements.',
      vi: { title: 'Chuỗi sampler', lede: 'Gọi tên mọi giai đoạn giữa logits và một token, nói giai đoạn nào phụ thuộc vào thứ tự của chúng, và nêu thứ tự cùng quy tắc phân định hòa mà mốc dự án M5 cài đặt.' },
      load: () => import('./SX11').then((m) => ({ Component: m.SX11, questions: m.questionsSX11 })),
    },
    {
      id: 'X1.2', num: 'X1.2', title: 'Temperature, softmax and greedy', pages: null,
      lede: 'Fold temperature into lab 10.3’s softmax, predict how it reshapes a distribution, and say what greedy decoding returns on a tie on each backend.',
      vi: { title: 'Temperature, softmax và greedy', lede: 'Gộp temperature vào softmax của lab 10.3, dự đoán nó định hình lại một phân phối ra sao, và nói greedy decoding trả về gì khi hòa trên từng backend.' },
      load: () => import('./SX12').then((m) => ({ Component: m.SX12, questions: m.questionsSX12 })),
    },
    {
      id: 'X1.3', num: 'X1.3', title: 'Penalties from token counts', pages: null,
      lede: 'Apply repetition, frequency and presence penalties from lab 9.3’s counts in coalesced, memory-bound elementwise passes, and say which sign rule each project uses.',
      vi: { title: 'Các penalty từ số đếm token', lede: 'Áp dụng penalty lặp lại (repetition), tần suất (frequency) và hiện diện (presence) từ số đếm của lab 9.3 trong các lượt theo từng phần tử được hợp nhất, bị giới hạn bởi bộ nhớ, và nói mỗi dự án dùng quy tắc dấu nào.' },
      load: () => import('./SX13').then((m) => ({ Component: m.SX13, questions: m.questionsSX13 })),
    },
    {
      id: 'X1.4', num: 'X1.4', title: 'Top-k is a selection, not a sort', pages: null,
      lede: 'Find the k largest of 150,000 logits without sorting them, by an 8-bit radix select, and predict how much reading and writing that saves over a full sort.',
      vi: { title: 'Top-k là một phép chọn, không phải sắp xếp', lede: 'Tìm k giá trị lớn nhất trong 150,000 logits mà không sắp xếp chúng, bằng một radix select 8-bit, và dự đoán nó tiết kiệm bao nhiêu lượt đọc và ghi so với sắp xếp toàn bộ.' },
      load: () => import('./SX14').then((m) => ({ Component: m.SX14, questions: m.questionsSX14 })),
    },
    {
      id: 'X1.5', num: 'X1.5', title: 'Top-p over a sorted row', pages: null,
      lede: 'Compose lab 14.3’s sort, lab 10.3’s softmax and lab 11.6’s scan into top-p, state its boundary and tie rule exactly, and decide how much of the row you really need to sort.',
      vi: { title: 'Top-p trên một hàng đã sắp xếp', lede: 'Ghép phép sắp xếp của lab 14.3, softmax của lab 10.3 và scan của lab 11.6 thành top-p, nêu chính xác biên và quy tắc phân định hòa của nó, và quyết định bạn thực sự cần sắp xếp bao nhiêu phần của hàng.' },
      load: () => import('./SX15').then((m) => ({ Component: m.SX15, questions: m.questionsSX15 })),
    },
    {
      id: 'X1.6', num: 'X1.6', title: 'Min-p as a filter', pages: null,
      lede: 'Turn min-p into one comparison against max + ln p, run it as lab 12.4’s stable filter or as a prefix count on a sorted row, and predict how temperature placement changes what it keeps.',
      vi: { title: 'Min-p như một bộ lọc', lede: 'Biến min-p thành một phép so sánh với max + ln p, chạy nó như bộ lọc ổn định của lab 12.4 hoặc như một phép đếm tiền tố trên hàng đã sắp xếp, và dự đoán vị trí đặt temperature thay đổi những gì được giữ ra sao.' },
      load: () => import('./SX16').then((m) => ({ Component: m.SX16, questions: m.questionsSX16 })),
    },
    {
      id: 'X1.7', num: 'X1.7', title: 'Random numbers every backend agrees on', pages: null,
      lede: 'Derive one uniform number per row and step from (seed, stream, step) with Philox4x32-10, check it against published test vectors, and convert its bits to a float that is identical on CUDA, HIP, Metal and the CPU.',
      vi: { title: 'Số ngẫu nhiên mà mọi backend đều thống nhất', lede: 'Suy ra một số đều cho mỗi hàng và mỗi bước từ (seed, stream, step) bằng Philox4x32-10, kiểm tra nó với các test vector đã công bố, và chuyển các bit của nó thành một số float giống hệt nhau trên CUDA, HIP, Metal và CPU.' },
      load: () => import('./SX17').then((m) => ({ Component: m.SX17, questions: m.questionsSX17 })),
    },
    {
      id: 'X1.8', num: 'X1.8', title: 'The draw: inverse CDF over a scan', pages: null,
      lede: 'Pick the token where u lands in the running sum of the survivors’ weights with one block per row, make it safe at both ends of [0, 1), and test it for exact agreement across backends despite floating-point differences.',
      vi: { title: 'Lần rút: CDF ngược trên một scan', lede: 'Chọn token nơi u rơi vào tổng tích lũy trọng số của các token còn lại, với một khối cho mỗi hàng, làm nó an toàn ở cả hai đầu của [0, 1), và kiểm tra nó khớp chính xác giữa các backend bất chấp khác biệt dấu phẩy động.' },
      load: () => import('./SX18').then((m) => ({ Component: m.SX18, questions: m.questionsSX18 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Four labs on your own GPUs: top-k by selection against a full sort, top-p and min-p composed and then fused, the M5 sampler end to end with a fixed seed, and a hunt for three defects in a fused filter.',
      vi: { title: 'Lab', lede: 'Bốn lab trên GPU của chính bạn: top-k bằng phép chọn so với sắp xếp toàn bộ, top-p và min-p được ghép rồi gộp, sampler M5 từ đầu đến cuối với seed cố định, và một cuộc săn ba lỗi trong một bộ lọc đã gộp.' },
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '', title: 'Summary', pages: null,
      lede: 'What to carry forward, a mixed check, where every exercise of the module is worked, where the sources disagree, and milestone M5.',
      vi: { title: 'Tóm tắt', lede: 'Những gì cần mang theo, một bài kiểm tra tổng hợp, nơi mỗi bài tập của module được giải, chỗ các nguồn bất đồng, và mốc dự án M5.' },
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
