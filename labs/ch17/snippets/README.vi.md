> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Snippets

Các kernel mà trang 17.2 và trang tổng kết của chương 17 hiển thị để giải thích, và không phải là
lab. Trang nạp chúng lúc chạy bằng `useLabFile('ch17/snippets/<name>')`, nên không có bản sao nào
có thể lệch đi; nó hiển thị mỗi tệp không kèm chú thích đầu tệp, thay vào đó là chú thích ảnh của
trang. Trình chạy lab không dựng chúng.

| Tệp | Hiển thị ở | Được kiểm tra bởi |
|---|---|---|
| `spmv_coo.cu` | trang 17.2, "COO: one thread per non-zero" (Fig. 17.5 kèm `#include <cuda/atomic>` mà trang bỏ qua và một struct `COOMatrix`, thứ sách không bao giờ định nghĩa) | kiểm tra front-end CUDA; `app/src/chapters/ch17/snippets-s172.test.ts` ghim các dòng atomic và chạy nó qua `check/spmv_check.cu` |
| `spmv_csr.cu` | trang 17.2, "CSR: one thread per row" (Fig. 17.9 kèm một struct `CSRMatrix`) | như trên: các cận của vòng lặp và phép cộng dồn |
| `spmv_ell.cu` | trang 17.2, "ELL: pad, then turn" (Fig. 17.12 đã sửa: `y[row] += sum` thay cho `y[row] = sum` của trang; ô đệm chứa cột -1 và giá trị 0, nên `colIdx` và `col` có dấu; `stride` thay cho `numRows`; cộng thêm cùng kernel đó lặp qua toàn bộ độ rộng và kiểm tra `col >= 0`) | như trên: dòng chỉ số, phép cộng dồn và phép kiểm tra ô đệm, trong cả hai kernel |
| `spmv_csc.cu` | trang 17.2, "CSC: the same idea, by column" (Fig. 17.18 với tên trường số ít `rowIdx` và `value` thay cho `rowIdxs` và `values` của listing, câu include và một struct `CSCMatrix`) | như trên: `x[col]` được nạp một lần trước vòng lặp, phép cộng nguyên tử |
| `spmv_jds.cu` | trang tổng kết, "Exercise 5, answered" (của khóa học này: sách không cho kernel JDS nào; mỗi hàng đã sắp xếp một luồng, hàng dài nhất trước, `y[row[r]] += sum`) | kiểm tra front-end CUDA; `app/src/chapters/ch17/Wrap.snippets.test.ts` ghim các dòng được tô sáng và chạy nó qua `check/spmv_jds_check.cu` |
| `check/spmv_jds_check.cu` | không ở đâu (một chương trình điều khiển) | kiểm tra front-end CUDA. Được dựng và chạy bởi `Wrap.snippets.test.ts` trên biến thể `emu` (AddressSanitizer): các mảng dựng trên host của Fig. 17.14 đối chiếu với p. 416, kernel trên Fig. 17.1, Fig. 17.14, ma trận của bài tập 1, ma trận lệch 40 x 33 và một ma trận có hàng rỗng dưới ba thứ tự luồng, rồi một bản sao ghi vào `y[r]` thay vì `y[row[r]]` (5 trên 6 hàng sai trên Fig. 17.14, câu hỏi đầu tiên của trang tổng kết). |
| `check/spmv_check.cu` | không ở đâu (một chương trình điều khiển) | kiểm tra front-end CUDA. Được dựng và chạy bởi `snippets-s172.test.ts` như trình chạy lab dựng backend `emu` (bật AddressSanitizer): mọi kernel, cả hai vòng lặp ELL và một bước nhảy ELL làm tròn lên bội của 8, trên Fig. 17.1 và trên một ma trận lệch 40 x 33, dưới ba thứ tự luồng của trình giả lập, so sánh chính xác với một tham chiếu CPU (giá trị nguyên); rồi bản lai của p. 414 chạy phần COO trước, với `y[row] = sum` của trang (4 trên 4 hàng sai) và `+=` của snippet (không hàng nào); rồi Fig. 17.1 với ô đệm chứa cột 0 và `x[0] = +Inf` (3 hàng NaN) hoặc -1 (không hàng nào). Được dựng lại cho `race` (ThreadSanitizer): atomic COO của snippet không bị báo gì, và một bản sao `y[row] +=` thường bị báo ở đúng dòng của nó. Bị bỏ qua khi không tìm thấy trình biên dịch C++ hoặc python3. |

`spmv_coo.cu` và `spmv_csc.cu` include `<cuda/atomic>` mà không có phương án dự phòng cho HIP.
Chúng dành cho kiểm tra front-end CUDA và các trình giả lập; trên HIP, `<cuda/atomic>` đến từ
libhipcxx (`notes/today/infra-facts.md` B2). Không có gì ở đây đã chạy trên GPU: trạng thái trên
trang 17.2 là "emulator-checked and front-end-checked; unverified on CUDA and HIP
hardware".

Kiểm tra front-end CUDA là kiểm tra mà `labs/pmpp.py` chạy: `clang++ -x cuda --cuda-host-only
-nocudainc -nocudalib -fsyntax-only -std=c++17 -I labs/common/include -I
labs/common/include/pmpp/emu/shim -include pmpp/check/cuda_stub.h <file>`.
