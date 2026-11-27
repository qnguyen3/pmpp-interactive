import type { Chapter } from '../types';

// One file per section, each loaded on demand and exporting its component and its
// questions (the shape of chapter 5, the template).
export const chapter: Chapter = {
  id: 'ch04',
  num: '4',
  title: 'Compute architecture and scheduling',
  pages: '67–92',
  vi: { title: 'Kiến trúc tính toán và lập lịch' },
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '67',
      lede: 'See what the chapter is for, where it sits under a framework call, and check what it assumes you remember from chapters 1 to 3.',
      vi: { title: 'Trước khi bắt đầu', lede: 'Xem chương này để làm gì, nó nằm ở đâu bên dưới một lệnh gọi framework, và kiểm tra những gì nó giả định bạn còn nhớ từ chương 1 đến 3.' },
      load: () => import('./S40').then((m) => ({ Component: m.S40, questions: m.questionsS40 })),
    },
    {
      id: '4.1', num: '4.1', title: 'Architecture of a modern GPU', pages: '68',
      lede: 'Name the hardware levels a kernel runs on and say which of their sizes your own device will report.',
      vi: { title: 'Kiến trúc của một GPU hiện đại', lede: 'Gọi tên các tầng phần cứng mà một kernel chạy trên đó và nói kích thước nào trong số đó thiết bị của chính bạn sẽ báo cáo.' },
      load: () => import('./S41').then((m) => ({ Component: m.S41, questions: m.questionsS41 })),
    },
    {
      id: '4.2', num: '4.2', title: 'Thread block scheduling', pages: '69–70',
      lede: 'Predict how many blocks of a grid are running at any moment and say what being assigned as a whole block gives the threads inside it.',
      vi: { title: 'Lập lịch khối luồng', lede: 'Dự đoán có bao nhiêu khối của một lưới đang chạy tại mỗi thời điểm và nói việc được gán nguyên cả khối mang lại gì cho các luồng bên trong nó.' },
      load: () => import('./S42').then((m) => ({ Component: m.S42, questions: m.questionsS42 })),
    },
    {
      id: '4.3', num: '4.3', title: 'Synchronisation and transparent scalability', pages: '70–74',
      lede: 'Place a barrier where every thread of a block will reach it, and explain why the same grid runs unchanged on a device of any size.',
      vi: { title: 'Đồng bộ hóa và khả năng mở rộng trong suốt', lede: 'Đặt một rào chắn ở chỗ mọi luồng của khối đều đến được, và giải thích vì sao cùng một lưới chạy không cần sửa trên thiết bị ở mọi kích thước.' },
      load: () => import('./S43').then((m) => ({ Component: m.S43, questions: m.questionsS43 })),
    },
    {
      id: '4.4', num: '4.4', title: 'Warps and SIMD hardware', pages: '74–79',
      lede: 'Work out which threads of a block share a warp for any block shape and any warp width, and say what sharing one means.',
      vi: { title: 'Warp và phần cứng SIMD', lede: 'Tìm ra những luồng nào của một khối cùng chung một warp với mọi hình dạng khối và mọi độ rộng warp, và nói việc chung warp có nghĩa là gì.' },
      load: () => import('./S44').then((m) => ({ Component: m.S44, questions: m.questionsS44 })),
    },
    {
      id: '4.5', num: '4.5', title: 'Control divergence', pages: '79–82',
      lede: 'Predict the passes a warp makes at a branch or a data-dependent loop, count the warps a boundary check splits, and judge when that cost matters.',
      vi: { title: 'Phân kỳ điều khiển', lede: 'Dự đoán số lượt một warp phải chạy tại một nhánh hay một vòng lặp phụ thuộc dữ liệu, đếm số warp bị một phép kiểm tra biên chia tách, và đánh giá khi nào chi phí đó đáng kể.' },
      load: () => import('./S45').then((m) => ({ Component: m.S45, questions: m.questionsS45 })),
    },
    {
      id: '4.6', num: '4.6', title: 'Warp scheduling and latency tolerance', pages: '82–85',
      lede: 'Explain why an SM holds many more warps than it can execute and estimate how many a kernel needs to keep the hardware busy.',
      vi: { title: 'Lập lịch warp và khả năng che độ trễ', lede: 'Giải thích vì sao một SM giữ nhiều warp hơn hẳn số nó có thể thực thi, và ước lượng một kernel cần bao nhiêu warp để giữ phần cứng luôn bận.' },
      load: () => import('./S46').then((m) => ({ Component: m.S46, questions: m.questionsS46 })),
    },
    {
      id: '4.7', num: '4.7', title: 'Resource partitioning and occupancy', pages: '85–87',
      lede: 'Compute occupancy from block slots, thread slots and registers for any device limits, and find the small change that drops it off a cliff.',
      vi: { title: 'Phân chia tài nguyên và occupancy', lede: 'Tính occupancy từ số ô khối, số ô luồng và thanh ghi với mọi giới hạn thiết bị, và tìm thay đổi nhỏ khiến nó tụt dốc đột ngột.' },
      load: () => import('./S47').then((m) => ({ Component: m.S47, questions: m.questionsS47 })),
    },
    {
      id: '4.8', num: '4.8', title: 'Querying device properties', pages: '87–89',
      lede: 'Read the limits this chapter depends on from a CUDA, HIP or Metal device and turn them into a launch configuration.',
      vi: { title: 'Truy vấn thuộc tính thiết bị', lede: 'Đọc các giới hạn mà chương này phụ thuộc từ một thiết bị CUDA, HIP hoặc Metal và biến chúng thành một cấu hình khởi chạy.' },
      load: () => import('./S48').then((m) => ({ Component: m.S48, questions: m.questionsS48 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Four labs on your own GPUs: query the device and size a launch from it, measure what a divergent warp costs, hunt three defects around a barrier, and sweep block size on a kernel that waits on memory.',
      vi: { title: 'Lab', lede: 'Bốn lab trên GPU của chính bạn: truy vấn thiết bị và định kích thước một lần khởi chạy từ đó, đo chi phí của một warp phân kỳ, săn ba lỗi quanh một rào chắn, và quét kích thước khối trên một kernel phải chờ bộ nhớ.' },
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '4.9', title: 'Summary and exercises', pages: '89–92',
      lede: 'Collect what to carry forward and test it on situations the sections did not show.',
      vi: { title: 'Tóm tắt và bài tập', lede: 'Gom lại những gì cần mang theo và kiểm tra chúng trên những tình huống mà các phần trước chưa trình bày.' },
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
