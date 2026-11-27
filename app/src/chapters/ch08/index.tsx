import type { Chapter } from '../types';

// Condensed tier (PLAN.md): the PDE background is reduced to the start page; the two
// transferable techniques, thread coarsening and register tiling, get a page each.
export const chapter: Chapter = {
  id: 'ch08',
  num: '8',
  title: 'Stencil computation',
  pages: '183–199',
  vi: { title: 'Tính toán stencil' },
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '183–188',
      lede: 'Say what this chapter adds to your kernel toolkit and why the stencil itself matters little for inference, and check that you have the four ideas from chapters 4 to 7 it builds on.',
      vi: { title: 'Trước khi bắt đầu', lede: 'Nói được chương này bổ sung gì vào bộ công cụ kernel của bạn và vì sao bản thân stencil ít quan trọng với suy luận, rồi kiểm tra xem bạn đã có bốn ý từ chương 4 đến 7 mà chương này dựa vào hay chưa.' },
      load: () => import('./S80').then((m) => ({ Component: m.S80, questions: m.questionsS80 })),
    },
    {
      id: '8.4', num: '8.4', title: 'A 3D tile and its halo', pages: '187–193',
      lede: "Count what a 3D stencil kernel asks of memory, and see why a cube-shaped tile cannot get far: the thread limit caps its side, the halo takes more than half of it, and each warp's load is split across rows.",
      vi: { title: 'Một ô 3D và vùng biên (halo) của nó', lede: 'Đếm xem một kernel stencil 3D đòi hỏi bộ nhớ những gì, và thấy vì sao một ô hình lập phương không đi được xa: giới hạn số luồng chặn cạnh của nó, vùng biên chiếm hơn một nửa, và lần nạp của mỗi warp bị chia ra nhiều hàng.' },
      load: () => import('./S84').then((m) => ({ Component: m.S84, questions: m.questionsS84 })),
    },
    {
      id: '8.5', num: '8.5', title: 'Marching in z, then registers', pages: '193–198',
      lede: 'March a plane of threads through z with three rolling planes and say what each of its two barriers protects, then move the values only one thread reads into registers and say what that buys, what it costs and what it leaves unchanged.',
      vi: { title: 'Tiến theo z, rồi đến thanh ghi', lede: 'Cho một mặt phẳng luồng tiến dọc z với ba mặt phẳng luân phiên và nói mỗi rào chắn trong hai rào chắn của nó bảo vệ điều gì, rồi chuyển các giá trị chỉ một luồng đọc vào thanh ghi và nói điều đó đem lại gì, tốn gì và để nguyên điều gì.' },
      load: () => import('./S85').then((m) => ({ Component: m.S85, questions: m.questionsS85 })),
    },
    {
      id: 'labs', num: '', title: 'Lab', pages: null,
      lede: 'One optional lab: write the basic and the register-tiled 3D stencil, test them on every backend, and measure the chapter\'s model on your own GPU.',
      vi: { title: 'Lab', lede: 'Một lab tùy chọn: viết kernel stencil 3D cơ bản và bản phân ô thanh ghi, kiểm thử chúng trên mọi backend, và đo mô hình của chương trên GPU của chính bạn.' },
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '8.7', title: 'Summary and exercises', pages: '198–199',
      lede: 'What to carry forward, a mixed check on the whole chapter, where each of the book\'s exercises is worked, the PDF\'s apparent slips, and why this chapter has no project milestone.',
      vi: { title: 'Tóm tắt và bài tập', lede: 'Những gì cần mang theo, một bài kiểm tra tổng hợp cho cả chương, nơi từng bài tập của sách được giải, những chỗ có vẻ sai sót trong PDF, và vì sao chương này không có mốc dự án.' },
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
