import type { Chapter } from '../types';

// Built on chapter 5's template: one file per section, each loaded on demand and exporting
// its component and its questions.
export const chapter: Chapter = {
  id: 'ch14',
  num: '14',
  title: 'Sorting',
  pages: '329–348',
  vi: { title: 'Sắp xếp' },
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '329',
      lede: 'See where sorting sits under a top-p sampler, why top-k needs no full sort, and check what this chapter assumes from chapters 4, 6 and 10 to 13.',
      vi: { title: 'Trước khi bắt đầu', lede: 'Xem sắp xếp nằm ở đâu bên dưới một bộ lấy mẫu top-p, vì sao top-k không cần sắp xếp toàn bộ, và kiểm tra những gì chương này giả định từ các chương 4, 6 và 10 đến 13.' },
      load: () => import('./S140').then((m) => ({ Component: m.S140, questions: m.questionsS140 })),
    },
    {
      id: '14.1', num: '14.1', title: 'Background', pages: '330–331',
      lede: 'Say what a sort must guarantee, decide when you need stability, and build a row-wise sort from two stable passes.',
      vi: { title: 'Bối cảnh', lede: 'Nói được một phép sắp xếp phải bảo đảm điều gì, quyết định khi nào bạn cần tính ổn định, và dựng một phép sắp xếp theo từng hàng từ hai lượt ổn định.' },
      load: () => import('./S141').then((m) => ({ Component: m.S141, questions: m.questionsS141 })),
    },
    {
      id: '14.2', num: '14.2', title: 'Parallel odd-even sort', pages: '331–333',
      lede: 'Step odd-even sort across its threads, say why its pairs never collide, stop it correctly, and set its flag without a data race.',
      vi: { title: 'Sắp xếp chẵn-lẻ song song', lede: 'Chạy từng bước sắp xếp chẵn-lẻ qua các luồng của nó, nói vì sao các cặp của nó không bao giờ va chạm, dừng nó đúng lúc, và đặt cờ của nó mà không có race condition (tranh chấp dữ liệu).' },
      load: () => import('./S142').then((m) => ({ Component: m.S142, questions: m.questionsS142 })),
    },
    {
      id: '14.3', num: '14.3', title: 'Parallel merge sort', pages: '333–334',
      lede: 'Follow a merge tree stage by stage and say how a grid\'s blocks should be shared among its merges.',
      vi: { title: 'Sắp xếp trộn song song', lede: 'Theo dõi một cây trộn qua từng giai đoạn và nói các khối của một lưới nên được chia cho các phép trộn của nó như thế nào.' },
      load: () => import('./S143').then((m) => ({ Component: m.S143, questions: m.questionsS143 })),
    },
    {
      id: '14.4', num: '14.4', title: 'Radix sort', pages: '334–336',
      lede: 'Run an LSD radix sort pass by pass, say why each pass must be stable, and turn float logits into keys that radix sort orders correctly, descending.',
      vi: { title: 'Radix sort', lede: 'Chạy một radix sort LSD qua từng lượt, nói vì sao mỗi lượt phải ổn định, và biến logit dạng float thành khóa mà radix sort sắp đúng theo thứ tự giảm dần.' },
      load: () => import('./S144').then((m) => ({ Component: m.S144, questions: m.questionsS144 })),
    },
    {
      id: '14.5', num: '14.5', title: 'Parallel radix sort', pages: '336–339',
      lede: 'Compute every key\'s destination in a radix pass from one exclusive scan, and split the book\'s one-kernel pass into kernels that run on every backend.',
      vi: { title: 'Radix sort song song', lede: 'Tính đích đến của mọi khóa trong một lượt radix từ một scan loại trừ duy nhất, và tách lượt một-kernel của sách thành các kernel chạy được trên mọi backend.' },
      load: () => import('./S145').then((m) => ({ Component: m.S145, questions: m.questionsS145 })),
    },
    {
      id: '14.6', num: '14.6', title: 'Optimizing for memory coalescing', pages: '339–342',
      lede: 'Count the memory runs a radix scatter makes, and remove most of them with a local sort in shared memory and a scanned table of bucket sizes.',
      vi: { title: 'Tối ưu cho hợp nhất truy cập bộ nhớ', lede: 'Đếm số dải bộ nhớ mà một lần scatter radix tạo ra, và loại bỏ phần lớn chúng bằng một phép sắp xếp cục bộ trong bộ nhớ chia sẻ cùng một bảng kích thước bucket đã scan.' },
      load: () => import('./S146').then((m) => ({ Component: m.S146, questions: m.questionsS146 })),
    },
    {
      id: '14.7', num: '14.7', title: 'Choice of radix value', pages: '342–344',
      lede: 'Choose a radix width by trading passes against bucket count, coalescing and the size of the table scan.',
      vi: { title: 'Chọn giá trị radix', lede: 'Chọn độ rộng radix bằng cách cân số lượt với số bucket, hợp nhất truy cập và kích thước của phép scan bảng.' },
      load: () => import('./S147').then((m) => ({ Component: m.S147, questions: m.questionsS147 })),
    },
    {
      id: '14.8', num: '14.8', title: 'Thread coarsening to improve coalescing', pages: '344–345',
      lede: 'Pick how many keys each thread sorts, and say what it buys in bucket size and table size and what it costs in shared memory.',
      vi: { title: 'Gộp việc cho luồng để cải thiện hợp nhất truy cập', lede: 'Chọn mỗi luồng sắp bao nhiêu khóa, và nói điều đó đem lại gì về kích thước bucket và kích thước bảng, và tốn gì về bộ nhớ chia sẻ.' },
      load: () => import('./S148').then((m) => ({ Component: m.S148, questions: m.questionsS148 })),
    },
    {
      id: '14.9', num: '14.9', title: 'Other parallel sort methods', pages: '345–346',
      lede: 'Place sorting networks, sample sort and MSD radix sort, pick the library call for your backend, and say why top-k needs a selection, not a sort.',
      vi: { title: 'Các phương pháp sắp xếp song song khác', lede: 'Định vị mạng sắp xếp, sample sort và radix sort MSD, chọn lời gọi thư viện cho backend của bạn, và nói vì sao top-k cần một phép chọn, không phải một phép sắp xếp.' },
      load: () => import('./S149').then((m) => ({ Component: m.S149, questions: m.questionsS149 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Three labs on your own GPUs: odd-even and one-bit radix sort, a multi-bit radix sort with local sorting measured against std::stable_sort, and milestone M5, sorted candidates per row.',
      vi: { title: 'Lab', lede: 'Ba lab trên GPU của chính bạn: sắp xếp chẵn-lẻ và radix sort một bit, một radix sort nhiều bit có sắp xếp cục bộ được đo so với std::stable_sort, và mốc dự án M5, các ứng viên đã sắp xếp theo từng hàng.' },
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '14.10', title: 'Summary and exercises', pages: '346–348',
      lede: 'What to carry forward, a mixed check, where each exercise is worked, the PDF\'s apparent errata, and milestone M5.',
      vi: { title: 'Tóm tắt và bài tập', lede: 'Những gì cần mang theo, một bài kiểm tra tổng hợp, nơi mỗi bài tập được giải, các lỗi in có vẻ có trong PDF, và mốc dự án M5.' },
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
