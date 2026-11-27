> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Snippets

Mã mà các bài học chương 22 trình bày để phục vụ một câu hỏi, và không phải là lab. Các
tệp là `.cu` thật để `./verify.sh` chạy phép kiểm tra front-end CUDA trên chúng. Chúng
không được trình chạy lab dựng.

| Tệp | Hiển thị ở | Được kiểm tra bởi |
|---|---|---|
| `scatter_conv_plain.cu` | mục 22.2, "This scatter passes on the emulator" | Phép kiểm tra front-end CUDA; `app/src/chapters/ch22/S222.snippets.test.ts` thất bại nếu bản sao trong bài học lệch khỏi tệp này, kiểm tra hình dạng của nó, và chạy nó trên backend `emu` (đúng ở cả ba thứ tự luồng, vì `emu` không bao giờ ngắt một luồng giữa chừng `+=` của nó) và trên backend `race` (ThreadSanitizer báo dòng `+=` tranh chấp với chính nó) |
| `gather_conv.cu` | cùng chỗ đó, bên cạnh nó | Phép kiểm tra front-end CUDA; cùng phép kiểm thử đó: bản sao trong bài học, hình dạng, đúng trên `emu` và không có báo cáo trên `race` |
| `check/conv_check.cu` | không ở đâu cả (một chương trình điều khiển (driver)) | Phép kiểm tra front-end CUDA. Phép kiểm thử nối nó vào hai snippet, dịch kết quả bằng `labs/tools/cu2emu.py` và dựng nó giống cách trình chạy lab dựng `emu` và `race`. Nó cũng chứa một bản sao `atomicAdd` của scatter, được tạo cho phép kiểm tra và không phải mã bài học, bản này phải đúng ở mọi nơi và không có báo cáo nào. Bị bỏ qua khi không tìm thấy trình biên dịch C++; nửa phần race bị bỏ qua khi ThreadSanitizer không dựng được. |

Hai kernel tính cùng một phép tích chập (convolution) có đệm 0,
`out[o] = sum over t of filter[t] * in[o + t - r]`, dạng của chương 7. Sách không in
mã nào trong chương 22 (`notes/ch22.md`, "Code listings"); cả hai kernel là của khóa học này.
