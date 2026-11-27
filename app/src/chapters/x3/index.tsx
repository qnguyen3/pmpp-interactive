import type { Chapter } from '../types';

// Module X3 is not in the book: every section is ours, so no section carries book pages.
// Built in the shape of chapter 5: one file per section, each loaded on demand and
// exporting its component and its questions.
export const chapter: Chapter = {
  id: 'x3',
  num: 'X3',
  title: 'Reading real backends',
  pages: 'not in the book',
  vi: { title: 'Đọc các backend thực tế' },
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: null,
      lede: 'See what reading a real backend means in this module, which three projects and commits it reads, and check what it assumes from chapters 2 to 22 and module X1.',
      vi: { title: 'Trước khi bắt đầu', lede: 'Xem việc đọc một backend thực tế nghĩa là gì trong học phần này, học phần đọc ba dự án nào ở commit nào, và kiểm tra những gì nó giả định từ chương 2 đến 22 và học phần X1.' },
      load: () => import('./SX30').then((m) => ({ Component: m.SX30, questions: m.questionsSX30 })),
    },
    {
      id: 'X3.1', num: 'X3.1', title: 'Finding the kernel for an op', pages: null,
      lede: 'Start from the one place each project turns an op into a backend call, follow one name, and land in the kernel on CUDA, HIP and Metal.',
      vi: { title: 'Tìm kernel cho một op', lede: 'Bắt đầu từ nơi duy nhất mỗi dự án biến một op thành một lời gọi backend, lần theo một cái tên, và đến được kernel trên CUDA, HIP và Metal.' },
      load: () => import('./SX31').then((m) => ({ Component: m.SX31, questions: m.questionsSX31 })),
    },
    {
      id: 'X3.2', num: 'X3.2', title: 'Elementwise and gather ops', pages: null,
      lede: 'Read the three projects’ elementwise and embedding kernels as chapter 2’s kernel with chapter 6’s coarsening and vector loads, and say how each one picks a variant on CUDA and on Metal.',
      vi: { title: 'Các op từng phần tử và op gather', lede: 'Đọc các kernel từng phần tử và kernel embedding của ba dự án như kernel của chương 2 cộng với gộp việc cho luồng (thread coarsening) và nạp vector của chương 6, và nói mỗi kernel chọn biến thể ra sao trên CUDA và trên Metal.' },
      load: () => import('./SX32').then((m) => ({ Component: m.SX32, questions: m.questionsSX32 })),
    },
    {
      id: 'X3.3', num: 'X3.3', title: 'Row reductions: RMSNorm and softmax', pages: null,
      lede: 'Recognise chapter 10’s block reduction in the RMSNorm and softmax kernels of all three projects, see how an online softmax reads the row twice where lab 10.3’s reads it three times, and say which exponential each kernel calls.',
      vi: { title: 'Rút gọn theo hàng: RMSNorm và softmax', lede: 'Nhận ra phép rút gọn theo khối của chương 10 trong các kernel RMSNorm và softmax của cả ba dự án, thấy một online softmax đọc hàng hai lần trong khi bản của lab 10.3 đọc ba lần, và nói mỗi kernel gọi hàm mũ nào.' },
      load: () => import('./SX33').then((m) => ({ Component: m.SX33, questions: m.questionsSX33 })),
    },
    {
      id: 'X3.4', num: 'X3.4', title: 'Matrix products: the shape picks the kernel', pages: null,
      lede: 'Predict which kernel each project runs for a matrix product of a given shape, from one token to a full batch, and read MLX’s GEMV as lab 22.1’s kernel.',
      vi: { title: 'Phép nhân ma trận: hình dạng chọn kernel', lede: 'Dự đoán mỗi dự án chạy kernel nào cho một phép nhân ma trận có hình dạng cho trước, từ một token đến cả một batch, và đọc GEMV của MLX như kernel của lab 22.1.' },
      load: () => import('./SX34').then((m) => ({ Component: m.SX34, questions: m.questionsSX34 })),
    },
    {
      id: 'X3.5', num: 'X3.5', title: 'Attention: own kernels, libraries and submodules', pages: null,
      lede: 'Say which attention kernel each project runs for prefill and for decode, find it even when it is not in the project’s tree, and read MLX’s CUDA decode kernel against lab 20.4.',
      vi: { title: 'Attention: kernel tự viết, thư viện và submodule', lede: 'Nói mỗi dự án chạy kernel attention nào cho prefill và cho decode, tìm ra nó ngay cả khi nó không nằm trong cây mã của dự án, và đọc kernel decode CUDA của MLX đối chiếu với lab 20.4.' },
      load: () => import('./SX35').then((m) => ({ Component: m.SX35, questions: m.questionsSX35 })),
    },
    {
      id: 'X3.6', num: 'X3.6', title: 'Sampling and greedy decoding', pages: null,
      lede: 'Find each project’s sampler and argmax, say where each runs by default, and predict which token each one picks on an exact tie.',
      vi: { title: 'Lấy mẫu và giải mã greedy', lede: 'Tìm bộ lấy mẫu và argmax của mỗi dự án, nói mỗi cái mặc định chạy ở đâu, và dự đoán mỗi cái chọn token nào khi hòa tuyệt đối.' },
      load: () => import('./SX36').then((m) => ({ Component: m.SX36, questions: m.questionsSX36 })),
    },
    {
      id: 'X3.7', num: 'X3.7', title: 'The Llama step: RMSNorm, RoPE and SwiGLU', pages: null,
      lede: 'Lay out one Llama decode step as launches of the course’s kernels, add the three pieces the course has not built, and keep the two RoPE pairings apart.',
      vi: { title: 'Bước Llama: RMSNorm, RoPE và SwiGLU', lede: 'Trình bày một bước decode của Llama thành các lần khởi chạy kernel của khóa học, thêm ba phần khóa học chưa xây, và phân biệt rõ hai cách ghép cặp RoPE.' },
      load: () => import('./SX37').then((m) => ({ Component: m.SX37, questions: m.questionsSX37 })),
    },
    {
      id: 'X3.8', num: 'X3.8', title: 'Real checkpoints: formats and the traps between them', pages: null,
      lede: 'Read a llama2.c .bin header and a safetensors header by hand, check a download by its size, and list the conversions a Hugging Face Llama needs before the course’s kernels can run it.',
      vi: { title: 'Checkpoint thực tế: định dạng và những cái bẫy giữa chúng', lede: 'Đọc tay một header .bin của llama2.c và một header safetensors, kiểm tra một bản tải về qua kích thước của nó, và liệt kê các phép chuyển đổi mà một Llama của Hugging Face cần trước khi các kernel của khóa học chạy được nó.' },
      load: () => import('./SX38').then((m) => ({ Component: m.SX38, questions: m.questionsSX38 })),
    },
    {
      id: 'X3.9', num: 'X3.9', title: 'Testing against a reference, and contributing', pages: null,
      lede: 'Compare a GPU run with a reference the way the projects do, explain why exact greedy agreement can fail late without any kernel being wrong, run your own kernel inside MLX or PyTorch, and know what each project asks of a kernel change.',
      vi: { title: 'Kiểm thử đối chiếu với tham chiếu, và đóng góp', lede: 'So một lần chạy GPU với một tham chiếu theo cách các dự án làm, giải thích vì sao sự trùng khớp greedy tuyệt đối có thể hỏng muộn mà không kernel nào sai, chạy kernel của chính bạn bên trong MLX hoặc PyTorch, và biết mỗi dự án đòi hỏi gì ở một thay đổi kernel.' },
      load: () => import('./SX39').then((m) => ({ Component: m.SX39, questions: m.questionsSX39 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Two labs on your own GPUs: the RMSNorm, RoPE and SwiGLU kernels a Llama needs plus a review of a buggy port, then milestone M10, a generated Llama-style model run end to end on the course’s kernels and checked token by token against a CPU reference.',
      vi: { title: 'Lab', lede: 'Hai lab trên GPU của chính bạn: các kernel RMSNorm, RoPE và SwiGLU mà một Llama cần cùng việc review một bản port có lỗi, rồi mốc dự án M10, một mô hình kiểu Llama được sinh ra, chạy trọn vẹn trên các kernel của khóa học và được kiểm tra từng token với một tham chiếu CPU.' },
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '', title: 'Summary', pages: null,
      lede: 'What to carry forward, a mixed check, where every exercise of the module is worked, where the sources disagree, and milestone M10.',
      vi: { title: 'Tóm tắt', lede: 'Những gì cần mang theo, một bài kiểm tra tổng hợp, nơi mọi bài tập của học phần được giải, những chỗ các nguồn không thống nhất, và mốc dự án M10.' },
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
