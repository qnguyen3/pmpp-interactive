> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 5.3: một GEMM tổng quát (mốc dự án M2)

Sách: mục 5.5 (pp. 112-115), Fig. 5.13, và trường hợp ma trận chữ nhật mà sách để lại làm bài tập.

Mở rộng kernel phân ô của bạn thành `C[j x l] = A[j x k] * B[k x l]` cho mọi kích thước. Đây là
kernel đằng sau mọi lớp tuyến tính: j là batch, k là số đầu vào, l là số đầu ra.

Ba điều thay đổi so với lab 5.2: `Width` từng mang ba nghĩa khác nhau; kích thước không còn là bội
của ô, nên mọi lần truy cập bộ nhớ cần kiểm tra biên riêng và số giai đoạn được làm tròn lên; và
lưới (grid) không còn vuông.

- Sửa `work/kernel.cu` và `work/kernel.metal` + `work/launch.mm`.
- Kiểm thử: `python3 labs/pmpp.py run 5.3`   Trình giả lập (emulator) chạy dưới AddressSanitizer,
  nên một kiểm tra biên bị thiếu sẽ được báo cùng dòng mã nguồn.
- Đo: `python3 labs/pmpp.py run 5.3 --bench` đo thời gian một lớp 1024 -> 4096 ở các kích thước
  batch từ 1 đến 1024.

Trước khi đo hiệu năng, hãy dự đoán tỉ lệ GFLOP/s ở batch 256 so với batch 1.
