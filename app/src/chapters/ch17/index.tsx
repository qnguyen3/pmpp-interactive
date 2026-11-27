import type { Chapter } from '../types';

// Chapter 17 is condensed: two teaching pages built around the book's format comparison,
// one optional lab, and a summary. Each section loads on demand and exports its component
// and its questions.
export const chapter: Chapter = {
  id: 'ch17',
  num: '17',
  title: 'Sparse matrix computation',
  pages: '401–423',
  vi: { title: 'Tính toán ma trận thưa' },
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '401–404',
      lede: 'Say what SpMV computes and why this course keeps only its techniques, judge when a sparse layout reads fewer bytes than a dense one, and check the ideas from chapters 4, 6, 9 and 11 and appendix A that the chapter builds on.',
      vi: { title: 'Trước khi bắt đầu', lede: 'Nói được SpMV tính gì và vì sao khóa học này chỉ giữ lại các kỹ thuật của nó, đánh giá khi nào một cách lưu ma trận thưa đọc ít byte hơn cách lưu dày đặc, và kiểm tra các ý từ chương 4, 6, 9, 11 và phụ lục A mà chương này dựa vào.' },
      load: () => import('./S170').then((m) => ({ Component: m.S170, questions: m.questionsS170 })),
    },
    {
      id: '17.2', num: '17.2', title: 'Formats decide the kernel', pages: '402–414, 418–421',
      lede: 'Turn a small sparse matrix into COO, CSR, ELL and CSC, give each the SpMV thread mapping it implies, and say for each whether it needs atomics, whether its loads coalesce and where its warps diverge.',
      vi: { title: 'Định dạng quyết định kernel', lede: 'Chuyển một ma trận thưa nhỏ sang COO, CSR, ELL và CSC, gán cho mỗi định dạng cách ánh xạ luồng SpMV mà nó kéo theo, và nói với từng định dạng liệu nó có cần phép toán nguyên tử (atomic) không, các lần nạp của nó có được hợp nhất không và các warp của nó phân kỳ ở đâu.' },
      load: () => import('./S172').then((m) => ({ Component: m.S172, questions: m.questionsS172 })),
    },
    {
      id: '17.5', num: '17.5', title: 'Buying regularity: padding, spilling, sorting', pages: '412–422',
      lede: "Price ELL's padding against CSR's divergence on rows of uneven length, choose a hybrid split or a sort, say when the preprocessing pays for itself, and recognise the same moves in mixture-of-experts routing.",
      vi: { title: 'Mua sự đều đặn: đệm, tràn, sắp xếp', lede: 'Cân phần đệm của ELL với phân kỳ luồng của CSR trên các hàng dài không đều, chọn cách chia lai hoặc sắp xếp, nói khi nào phần tiền xử lý tự bù được chi phí của nó, và nhận ra cùng những nước đi ấy trong việc định tuyến mixture-of-experts.' },
      load: () => import('./S175').then((m) => ({ Component: m.S175, questions: m.questionsS175 })),
    },
    {
      id: 'labs', num: '', title: 'Lab', pages: null,
      lede: 'One optional lab: write CSR and ELL SpMV, test them against a CPU reference on every backend, then measure them on matrices whose row lengths are uniform, spread, sorted and skewed, with a hybrid split, after predicting the outcome.',
      vi: { title: 'Lab', lede: 'Một lab tùy chọn: viết SpMV cho CSR và ELL, kiểm tra chúng với bản tham chiếu trên CPU ở mọi backend, rồi đo chúng trên các ma trận có độ dài hàng đều, trải rộng, đã sắp xếp và lệch, cùng một cách chia lai, sau khi dự đoán kết quả.' },
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '17.8', title: 'Summary and exercises', pages: '421–422',
      lede: "What to carry forward, a mixed check, where each of the book's five exercises is worked, the PDF's apparent slips in this chapter, and why the chapter has no project milestone.",
      vi: { title: 'Tóm tắt và bài tập', lede: 'Những gì cần mang theo, một bài kiểm tra tổng hợp, nơi mỗi bài trong năm bài tập của sách được giải, những chỗ có vẻ sai sót của bản PDF trong chương này, và vì sao chương này không có mốc dự án.' },
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
