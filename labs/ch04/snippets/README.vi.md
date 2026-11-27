> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Snippets

Các kernel mà bài học của chương 4 hiển thị để phục vụ một câu hỏi, và không phải là lab. Chúng
được giữ ở đây dưới dạng tệp `.cu` thật để `./verify.sh` chạy kiểm tra front-end CUDA trên chúng,
và các unit test sẽ thất bại nếu bản sao hiển thị trong bài học lệch khỏi tệp. Chúng không được
biên dịch hay chạy.

- `incorrect_barrier.cu`: Fig. 4.4 của sách (p. 72) có thêm `__global__`, thứ mà đoạn mã in trong
  sách thiếu. Nó cố ý sai.
- `barrier_uniform.cu`, `barrier_per_thread.cu`, `barrier_early_return.cu`, `barrier_loop.cu`,
  `barrier_guarded.cu`: các kernel trong phần săn lỗi "Is this barrier legal?" của mục 4.3. Tất cả
  trừ cái đầu và cái cuối đều cố ý sai. `app/src/sim/barrier.ts` chứa một mô hình viết tay cho
  từng kernel, và `app/src/sim/barrier.test.ts` gắn số dòng của mô hình và các bản sao trong bài
  học (`app/src/chapters/ch04/S43.tsx`) với các tệp này.
- `device_loop.cu`: vòng lặp truy vấn của mục 4.8 (sách p. 88) với chỉ số `int`, kiểm tra cả hai
  mã trả về, và các trường cho từng SM mà danh sách ở p. 89 bỏ sót. Chỉ có mã host. Bài học
  (`app/src/chapters/ch04/S48.tsx`) nạp tệp này lúc chạy, nên không có bản sao thứ hai để lệch.
  Nó không đọc tần số xung nhịp: `clockRate` không còn là thành viên của `cudaDeviceProp`, và
  header stub không có `cudaDevAttrClockRate`.
- `foo_kernel.cu`: kernel và lần khởi chạy của bài tập 1 trong sách (pp. 90 to 91), đúng như in,
  cho phần dự đoán của mục 4.5. `app/src/sim/divergence.test.ts` gắn bản sao trong bài học
  (`app/src/chapters/ch04/S45.tsx`) và ba điều kiện của mô hình với tệp này.
- `warp_tail_sum.cu`: một phép tính tổng khối của khóa học này mà sáu bước cuối giả định một warp
  chạy cùng nhịp (lockstep), không có rào chắn (barrier), cho câu hỏi đầu tiên của trang tổng kết.
  Nó cố ý sai trên mọi thiết bị không hứa chạy cùng nhịp.
  `app/src/chapters/ch04/Wrap.snippets.test.ts` gắn bản sao của trang
  (`app/src/chapters/ch04/Wrap.tsx`) với tệp này và chạy sáu bước theo kiểu cùng nhịp lẫn không
  theo thứ tự.

Không có ở đây: phần săn lỗi in sai (erratum) của mục 4.8 hiển thị vòng lặp ở p. 88 với các trường
ở p. 89 đúng như in, bao gồm `devProp.clockRate`, thứ mà CUDA 13.0 đã gỡ bỏ. Đoạn mã đó sẽ cố ý
không qua kiểm tra front-end, nên nó chỉ nằm trong `app/src/chapters/ch04/S48.tsx`, và
`app/src/sim/launchconfig.test.ts` gắn nó với `notes/ch04.md` và với tệp facts (A18).
