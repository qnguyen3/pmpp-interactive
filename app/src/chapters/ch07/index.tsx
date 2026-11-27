import type { Chapter } from '../types';

// Chapter 7, built on chapter 5's template: one file per section, each loaded on demand
// and exporting its component and its questions.
export const chapter: Chapter = {
  id: 'ch07',
  num: '7',
  title: 'Convolution',
  pages: '159–181',
  vi: { title: 'Tích chập' },
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '159',
      lede: 'What a convolution chapter is doing in a course about LLM kernels, the four kernels it builds, and a check on what it assumes from chapters 3 to 5.',
      vi: { title: 'Trước khi bắt đầu', lede: 'Một chương về tích chập làm gì trong một khóa học về kernel cho LLM, bốn kernel mà chương xây dựng, và một bài kiểm tra những gì chương giả định bạn còn nhớ từ chương 3 đến 5.' },
      load: () => import('./S70').then((m) => ({ Component: m.S70, questions: m.questionsS70 })),
    },
    {
      id: '7.1', num: '7.1', title: 'Background', pages: '159–164',
      lede: "Compute any output of a 1D or 2D convolution by hand, including at the edges, and match the book's definition to what PyTorch's Conv2d computes.",
      vi: { title: 'Kiến thức nền', lede: 'Tự tay tính bất kỳ đầu ra nào của một phép tích chập 1D hay 2D, kể cả ở các cạnh, và đối chiếu định nghĩa của sách với những gì Conv2d của PyTorch tính.' },
      load: () => import('./S71').then((m) => ({ Component: m.S71, questions: m.questionsS71 })),
    },
    {
      id: '7.2', num: '7.2', title: 'Parallel convolution: a basic kernel', pages: '164–166',
      lede: 'Map one thread to one output, write the bounds-checked kernel, and predict which warps diverge at the image edges.',
      vi: { title: 'Tích chập song song: một kernel cơ bản', lede: 'Ánh xạ một luồng (thread) cho một đầu ra, viết kernel có kiểm tra biên, và dự đoán những warp nào phân kỳ ở các cạnh ảnh.' },
      load: () => import('./S72').then((m) => ({ Component: m.S72, questions: m.questionsS72 })),
    },
    {
      id: '7.3', num: '7.3', title: 'Memory bandwidth considerations', pages: '167–168',
      lede: "Work out a convolution's ideal and actual operations per byte, and tell from the filter size alone whether any kernel for it can be compute-bound.",
      vi: { title: 'Những cân nhắc về băng thông bộ nhớ', lede: 'Tính số phép toán trên mỗi byte lý tưởng và thực tế của một phép tích chập, và chỉ từ kích thước bộ lọc mà biết liệu có kernel nào cho nó có thể bị giới hạn bởi tính toán hay không.' },
      load: () => import('./S73').then((m) => ({ Component: m.S73, questions: m.questionsS73 })),
    },
    {
      id: '7.4', num: '7.4', title: 'Constant memory and caching', pages: '168–172',
      lede: 'Put a small read-only table where every thread can read it cheaply on CUDA, HIP and Metal, and say when that helps and when it serialises.',
      vi: { title: 'Bộ nhớ hằng và bộ nhớ đệm', lede: 'Đặt một bảng nhỏ chỉ-đọc ở nơi mọi luồng đều đọc được với chi phí thấp trên CUDA, HIP và Metal, và nói khi nào cách đó có lợi và khi nào nó bị tuần tự hóa.' },
      load: () => import('./S74').then((m) => ({ Component: m.S74, questions: m.questionsS74 })),
    },
    {
      id: '7.5', num: '7.5', title: 'Tiled convolution with halo cells', pages: '172–177',
      lede: 'Size input and output tiles for a filter, write the load-wait-compute kernel, and predict its intensity from tile size and filter radius.',
      vi: { title: 'Tích chập phân ô với ô vùng biên', lede: 'Định kích thước ô đầu vào và ô đầu ra cho một bộ lọc, viết kernel nạp-chờ-tính, và dự đoán cường độ tính toán của nó từ kích thước ô và bán kính bộ lọc.' },
      load: () => import('./S75').then((m) => ({ Component: m.S75, questions: m.questionsS75 })),
    },
    {
      id: '7.6', num: '7.6', title: 'Tiled convolution using caches for halo cells', pages: '178–179',
      lede: 'Keep only the tile interior in shared memory, read the halo through the cache, and find the error in the printed listing.',
      vi: { title: 'Tích chập phân ô dùng bộ nhớ đệm cho ô vùng biên', lede: 'Chỉ giữ phần trong của ô ở bộ nhớ chia sẻ (shared memory), đọc vùng biên (halo) qua bộ nhớ đệm, và tìm lỗi trong đoạn mã in trong sách.' },
      load: () => import('./S76').then((m) => ({ Component: m.S76, questions: m.questionsS76 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Two labs on your own GPUs: the filter in constant memory three ways, then tiled convolution with halos, measured against the formula.',
      vi: { title: 'Lab', lede: 'Hai lab trên GPU của chính bạn: bộ lọc trong bộ nhớ hằng theo ba cách, rồi tích chập phân ô với vùng biên, đo đối chiếu với công thức.' },
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '7.7', title: 'Summary and exercises', pages: '179–181',
      lede: 'What to carry forward, a mixed check, where each of the book\'s exercises is worked, and where the PDF looks wrong.',
      vi: { title: 'Tóm tắt và bài tập', lede: 'Những gì cần mang theo, một bài kiểm tra tổng hợp, nơi từng bài tập của sách được giải, và những chỗ PDF có vẻ sai.' },
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
