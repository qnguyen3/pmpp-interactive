> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 3.3: GEMM ngây thơ cho một lớp tuyến tính (mốc dự án M1, nửa đầu)

Sách: mục 3.4 (pp. 59-64), Fig. 3.11 (p. 61), và định nghĩa cho ma trận chữ nhật ở p. 60 mà hình
này thu hẹp lại thành ma trận vuông.

Hãy viết kernel của Fig. 3.11 cho `C[j x l] = A[j x k] * B[k x l]`: `float` theo thứ tự hàng
(row-major), mọi kích thước, mỗi phần tử của C một luồng (thread), khối (block) 16 x 16. Hình có một
`Width`. Bạn có ba kích thước, và mỗi `Width` trong hình trở thành đúng một trong số chúng.

Xét như một lớp tuyến tính, j là batch (chiều đầu tiên), k là số đầu vào và l là số đầu ra. B chứa
trọng số dưới dạng (in, out). PyTorch lưu ma trận chuyển vị, (out, in), và tính `y = x W^T`, nên với
`B = W^T` được dựng trên host, kernel này tính ra cùng `y`. Một trong các test làm đúng như vậy.
PyTorch và ggml (llama.cpp) giữ trọng số ở dạng (out, in) và đọc cả hai toán hạng dọc theo hàng. Lab
này giữ quy ước của sách, trong đó toán hạng thứ hai được đọc xuống theo cột, và lab 5.3 cũng giữ
quy ước đó: cùng thứ tự đối số, cùng cách lưu trữ, để kernel phân ô sau này thay thế kernel này.

- Sửa `work/kernel.cu`: `naiveGemmKernel` và `launchNaiveGemm`. Chiều x của lưới (grid) phủ l, chiều
  y phủ j.
- Sửa `work/kernel.metal` (kernel `naive_gemm`) và `work/launch.mm` (`encodeNaiveGemm`). Kernel nhận
  `[[thread_position_in_grid]]`, lần khởi chạy là `dispatchThreadgroups` với các threadgroup 16 x 16
  trọn vẹn, và kernel giữ điều kiện chặn. Làm tròn lên thành threadgroup trọn vẹn và chặn là cách
  matmul MPS của PyTorch và backend Metal của llama.cpp làm. MLX thì gọi `dispatchThreads`, hàm này
  khởi chạy đúng lưới bạn yêu cầu: không luồng nào nằm ngoài dữ liệu, nên không cần điều kiện chặn.
  Cách nào cũng qua được các test.
- Kiểm thử: `python3 labs/pmpp.py run 3.3`   Bắt đầu với `--backend emu`: nó chạy dưới
  AddressSanitizer, công cụ này dừng ở truy cập đầu tiên ra ngoài một buffer và nêu dòng mã.
  `--backend race` chạy các luồng giả lập đồng thời dưới ThreadSanitizer và báo cáo hai luồng cùng
  lưu vào một phần tử.
- Đo: `python3 labs/pmpp.py run 3.3 --bench` đo thời gian các độ rộng vuông 256, 512 và 1024, rồi
  lớp 1024 -> 4096 của lab 5.3 ở batch 1, 16 và 256.

Các test dùng các kích thước vuông và chữ nhật, một hàng (dạng của decode), một cột (ma trận nhân
vector, bài tập 2 của sách) và chiều trong bằng 1. C được điền sẵn, nên một phần tử bạn không bao giờ
ghi sẽ bị báo cáo. Ở những nơi AddressSanitizer không theo dõi (backend `race` và thiết bị thật),
một giá trị lưu quá cuối C cũng bị báo cáo.

Trước khi đo hiệu năng: tăng gấp đôi độ rộng từ 512 lên 1024 nhân lượng phép tính lên 8 lần. Hãy dự
đoán kernel chạy lâu hơn bao nhiêu lần trên GPU của bạn. Con số bạn nhận được là đường cơ sở mà
chương 5 xuất phát từ đó.

Đây là nửa đầu của M1. Chương 4 hoàn thành nó bằng cách chọn cấu hình khởi chạy từ các truy vấn
device thay vì một kích thước cố định 16 x 16.
