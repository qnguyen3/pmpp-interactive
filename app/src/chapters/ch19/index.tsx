import type { Chapter } from '../types';

// Chapter 19 is standard depth: a page per book section, one lab, and a summary. Each
// section loads on demand and exports its component and its questions.
export const chapter: Chapter = {
  id: 'ch19',
  num: '19',
  title: 'Convolutional neural networks',
  pages: '453–475',
  vi: { title: 'Mạng nơ-ron tích chập' },
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '453–454',
      lede: 'See what a convolution chapter is doing in a course about LLM kernels, what it builds, and the one idea from it that carries over, then check what it assumes from chapters 3, 5, 7 and 15.',
      vi: { title: 'Trước khi bắt đầu', lede: 'Xem một chương về tích chập làm gì trong một khóa học về kernel LLM, nó xây dựng những gì, và một ý tưởng từ nó được mang sang tiếp, rồi kiểm tra những gì nó giả định từ các chương 3, 5, 7 và 15.' },
      load: () => import('./S190').then((m) => ({ Component: m.S190, questions: m.questionsS190 })),
    },
    {
      id: '19.1', num: '19.1', title: 'Convolutional neural networks', pages: '454–459',
      lede: 'Compute any output of a convolutional layer by hand from its input maps and filter bank, and give the shapes of X, F and Y for any layer, batched or not.',
      vi: { title: 'Mạng nơ-ron tích chập', lede: 'Tính tay bất kỳ đầu ra nào của một lớp tích chập từ các map đầu vào và bộ lọc của nó, và cho biết kích thước của X, F và Y cho bất kỳ lớp nào, có batch hay không.' },
      load: () => import('./S191').then((m) => ({ Component: m.S191, questions: m.questionsS191 })),
    },
    {
      id: '19.2', num: '19.2', title: 'A CUDA convolutional layer kernel', pages: '460–463',
      lede: 'Map a 3D grid onto samples, output maps and tiles, write the direct kernel with the bounds checks the book leaves out, and explain why it cannot reuse an input tile across output maps.',
      vi: { title: 'Một kernel lớp tích chập CUDA', lede: 'Ánh xạ một lưới (grid) 3D lên các mẫu, các map đầu ra và các ô, viết kernel trực tiếp với các kiểm tra biên mà sách bỏ qua, và giải thích vì sao nó không thể dùng lại một ô đầu vào cho nhiều map đầu ra.' },
      load: () => import('./S192').then((m) => ({ Component: m.S192, questions: m.questionsS192 })),
    },
    {
      id: '19.3', num: '19.3', title: 'Formulating a convolutional layer as GEMM', pages: '463–472',
      lede: 'Build the unrolled input matrix for any layer, say what it costs, trace any of its elements back to the input with Eq. 19.5, and read the tiled kernel that never stores it.',
      vi: { title: 'Biểu diễn một lớp tích chập dưới dạng GEMM', lede: 'Dựng ma trận đầu vào đã unroll cho bất kỳ lớp nào, nói nó tốn bao nhiêu, truy ngược bất kỳ phần tử nào của nó về đầu vào bằng Eq. 19.5, và đọc kernel phân ô không bao giờ lưu nó.' },
      load: () => import('./S193').then((m) => ({ Component: m.S193, questions: m.questionsS193 })),
    },
    {
      id: '19.4', num: '19.4', title: 'The cuDNN library', pages: '472–474',
      lede: 'Describe a convolution to a vendor library the way cuDNN, MIOpen and Apple’s frameworks expect, translate between their names and the chapter’s, and say what the library decides for you.',
      vi: { title: 'Thư viện cuDNN', lede: 'Mô tả một phép tích chập cho thư viện của nhà sản xuất theo cách cuDNN, MIOpen và các framework của Apple mong đợi, chuyển đổi giữa tên gọi của chúng và của chương, và nói thư viện quyết định thay bạn những gì.' },
      load: () => import('./S194').then((m) => ({ Component: m.S194, questions: m.questionsS194 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'One lab on your own GPUs: a convolutional layer computed directly and as unroll plus the project’s GEMM, timed on five layer shapes, from an early layer to a pointwise one and LeNet C3 at batch 64, against your prediction.',
      vi: { title: 'Lab', lede: 'Một lab trên GPU của chính bạn: một lớp tích chập tính trực tiếp và bằng unroll cộng GEMM của dự án, đo thời gian trên năm kích thước lớp, từ một lớp đầu đến một lớp pointwise và LeNet C3 ở batch 64, so với dự đoán của bạn.' },
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '19.5', title: 'Summary and exercises', pages: '474–475',
      lede: 'What to carry forward, a mixed check, where each of the book’s exercises is worked, the PDF’s apparent errata for this chapter, and how the project’s GEMM was reused.',
      vi: { title: 'Tóm tắt và bài tập', lede: 'Những gì cần mang theo, một bài kiểm tra tổng hợp, nơi mỗi bài tập của sách được giải, errata có vẻ có trong PDF cho chương này, và cách GEMM của dự án được dùng lại.' },
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
