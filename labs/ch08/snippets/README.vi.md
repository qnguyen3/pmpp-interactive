> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Snippets

Các kernel mà bài học của chương 8 hiển thị để phục vụ một trình chạy từng bước hoặc một câu hỏi,
và không phải là lab. Chúng được giữ ở đây dưới dạng tệp `.cu` thật để `./verify.sh` chạy kiểm tra
front-end CUDA trên chúng, và một unit test sẽ thất bại nếu bản sao hiển thị trong bài học lệch
khỏi tệp. Chúng không được dựng hay chạy.

- `zmarch_planes.cu`, `zmarch_registers.cu`: các kernel của trình chạy từng bước ở mục 8.5
  "Step one block through z": Figs. 8.10 và 8.12 của sách, đã sửa (c3 với y-1 và c4
  với y+1 như trong Figs. 8.6 và 8.8; thanh ghi `inNext` ở chỗ Fig. 8.12 in
  `inNext_s`; thanh ghi được khởi tạo), với `ty` và `tx` thay cho `threadIdx.y` và
  `threadIdx.x`. `app/src/sim/zmarch.ts` là một mô hình viết tay của hai kernel này, và
  `app/src/chapters/ch08/S85.test.ts` gắn các bản sao của bài học với các tệp này.
  Các phiên bản đã kiểm thử của cùng các kernel là của lab 8.1 (`../shared/planes.cu` và
  `../lab1_stencil3d/solution/kernel.cu`).
- `rolling_window_indexed.cu`: ví dụ "keep the window in registers" của mục 8.5, cách đi theo z
  với phân ô thanh ghi, cửa sổ của nó là `float w[3]` được đánh chỉ số bằng `z % 3`. Của khóa học
  này, không phải của sách. `app/src/chapters/ch08/S85.test.ts` chạy một mô hình vòng lặp của nó
  dưới ba thứ tự luồng và nhận đúng kết quả của lượt quét CPU; theo quy tắc được NVIDIA ghi trong
  tài liệu, cửa sổ của nó là ứng viên cho bộ nhớ cục bộ (local memory).
- `regtile_two_planes.cu`: câu hỏi của trang tổng kết về double buffering: Fig. 8.12
  (đã sửa) với mặt phẳng chia sẻ được nhân đôi, để vòng lặp chỉ cần một rào chắn (barrier). Của
  khóa học này, không phải của sách. `app/src/chapters/ch08/Wrap.snippets.test.ts` gắn bản sao
  của trang với tệp này và chạy một mô hình vòng lặp của nó từng luồng một dưới ba thứ tự.
