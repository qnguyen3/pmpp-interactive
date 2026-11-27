> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Snippets

Mã mà các bài học chương 9 trình bày để phục vụ một lời giải thích, và không phải là
lab. Các bài học nạp những tệp này lúc chạy bằng `useLabFile('ch09/snippets/<name>')`, nên
không có bản sao nào có thể bị lệch. Chúng không được trình chạy lab dựng.

| Tệp | Hiển thị ở | Được kiểm tra bởi |
|---|---|---|
| `histogram_sequential.cpp` | mục 9.1, "One division, one flat array" | `app/src/sim/histogram.test.ts` biên dịch `check/histogram_sequential_test.cpp` bằng trình biên dịch C++ của host, chạy nó trên mọi ảnh dựng sẵn của trình khám phá trong mục và trên các ảnh lớn hơn có seed cố định, rồi so sánh 256 số đếm với mô hình (`app/src/sim/histogram.ts`). Phép kiểm thử bị bỏ qua khi không tìm thấy trình biên dịch C++. |
| `privatised_variants.cu` | mục 9.4, "Seven broken privatised kernels" | Phép kiểm tra front-end CUDA; `app/src/sim/histpriv.test.ts` cố định cách mỗi kernel khác với `histPrivatised` và chạy mọi kernel trên trình giả lập (emulator) (xem hàng tiếp theo) |
| `check/privatised_check.cu` | không ở đâu cả (một chương trình điều khiển (driver)) | Phép kiểm tra front-end CUDA. `app/src/sim/histpriv.test.ts` nối nó vào `privatised_variants.cu`, dịch kết quả bằng `labs/tools/cu2emu.py`, dựng nó giống cách trình chạy lab dựng các backend `emu` và `race`, chạy mỗi kernel trên tám điểm ảnh của bài săn lỗi, và so sánh các bin của mỗi thứ tự luồng với `app/src/sim/histpriv.ts` (`simulate`) và các báo cáo ThreadSanitizer với phép kiểm tra happens-before của mô hình (`races`). Bị bỏ qua khi không tìm thấy trình biên dịch C++; nửa phần race bị bỏ qua khi ThreadSanitizer không dựng được. |
| `wrap_early_exit.cu` | trang tổng kết, "A mixed check", câu hỏi đầu tiên | Phép kiểm tra front-end CUDA; `app/src/chapters/ch09/Wrap.snippets.test.ts` kiểm tra hình dạng của nó, dịch nó bằng `labs/tools/cu2emu.py` cùng với `check/wrap_early_exit_test.cu`, chạy nó trên trình giả lập của khóa học theo ba thứ tự luồng ở 256 và 128 luồng mỗi khối, và so sánh các bin mà nó tính sai với mô hình của trang (`earlyExitLost` trong `Wrap.tsx`). Bị bỏ qua khi không tìm thấy trình biên dịch C++. |
| `check/wrap_early_exit_test.cu` | không ở đâu cả (một chương trình điều khiển) | Phép kiểm tra front-end CUDA; được chạy bởi cùng phép kiểm thử đó. Nó cũng chạy một bản sao của kernel trong đó lệnh trả về sớm được thay bằng một phép chặn trên lần cập nhật, bản này phải đúng ở mọi nơi. |

`histogram_sequential.cpp` là Fig. 9.2 (p. 203) với `const` được thêm vào con trỏ ảnh.
Giống mã in trong sách, nó giả định bên gọi đã đặt `bins` về 0.

`privatised_variants.cu` là Fig. 9.10 (p. 214) được thu nhỏ cho bài săn lỗi (tám bin, mỗi luồng
một điểm ảnh) cùng bảy bản sao cố ý sai. Nó đọc số đếm riêng một lần vào
`count`, trong khi sách đọc `bins_s[b]` hai lần.

`wrap_early_exit.cu` là Fig. 9.10 (p. 214) với một thay đổi, cố ý sai: một luồng không có
điểm ảnh nào trả về trước vòng lặp đặt về 0, nên các bin mà nó phụ trách trong khối cuối cùng không bao giờ được
đặt về 0 hay được ghi nhận.
