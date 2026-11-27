> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Snippets

Các kernel mà bài học chương 7 trình bày và không phải là lab. Chúng được giữ ở đây dưới dạng
tệp `.cu` thật để `./verify.sh` chạy phép kiểm tra front-end CUDA trên chúng, và các unit test
sẽ thất bại nếu bản sao hiển thị trong bài học lệch khỏi tệp. Chúng không được trình chạy lab
dựng; các chương trình điều khiển (driver) trong `check/` chạy chúng trên trình giả lập (emulator)
CPU từ các unit test.

- `s75_halo_toy.cu`: kernel của Fig. 7.12 trong sách (p. 174), đã sửa, ở các kích thước của
  Fig. 7.13 (`FILTER_RADIUS 1`, `IN_TILE_DIM 8`, một ô đầu ra 6 x 6), cùng một hàm khởi chạy
  của khóa học này (sách không trình bày mã host). Các chỗ sửa so với mã in trong sách:
  `N_s` có kiểu phần tử, mảng hằng được dùng đúng tên mà nó được khai báo
  (`F_c`), `0.0f` thay cho `0.0`, và `threadIdx` được chép sang `int` trước khi
  trừ bất cứ thứ gì. Trình chạy từng bước "Load, wait, compute" của mục 7.5 hiển thị nó và
  tô sáng các dòng của nó; `app/src/chapters/ch07/S75.snippets.test.ts` gắn bản sao trong
  `S75.tsx` với tệp này và cố định các dòng được tô sáng, và mô hình của trình chạy từng bước là
  `app/src/sim/convtile.ts`. Front-end-checked và emulator-checked: cùng phép kiểm thử đó chạy
  nó qua `check/s75_halo_toy_test.cu`.
- `wrap_causal_conv1d.cu`: một phép tích chập (convolution) 1D nhân quả của khóa học này (4 điểm
  lọc, một vùng biên (halo) một phía, khối 128 luồng) cho câu hỏi đầu tiên của trang tổng kết. Nó
  cố ý sai: các luồng vượt quá cuối mảng trả về trước rào chắn (barrier), nên khi khối cuối cùng
  chỉ có 1 hoặc 2 đầu ra, một ô vùng biên được đọc mà không ai ghi. `app/src/chapters/ch07/Wrap.test.ts`
  gắn bản sao của trang (`app/src/chapters/ch07/Wrap.tsx`) với tệp này, chạy một mô hình của
  kernel cho mọi độ dài từ 1 đến 400, và chạy chính tệp này qua
  `check/wrap_causal_conv1d_test.cu`. Front-end-checked và emulator-checked.

## check/

Đây là các chương trình điều khiển, không phải mã bài học. Mỗi chương trình include snippet của nó
bằng `#include "../<snippet>.cu"`, nên phép kiểm tra front-end CUDA phân tích cặp đó như một tệp.
`app/src/chapters/ch07/snippetEmu.ts` chèn snippet vào thay cho dòng đó, dịch kết quả bằng
`labs/tools/cu2emu.py`, dựng nó giống như trình chạy lab dựng backend `emu`
(bật AddressSanitizer), chạy nó và trả về những gì nó in ra. Các phép kiểm thử bị bỏ qua khi
không tìm thấy trình biên dịch C++ hoặc `python3`.

| Chương trình điều khiển | Nó chạy gì | Phép kiểm thử yêu cầu gì |
|---|---|---|
| `check/s75_halo_toy_test.cu` | `launchHaloToy` trên các ảnh 10x10, 14x14, 13x7, 6x6, 1x1 và 37x19 với một bộ lọc 3 x 3 lệch, theo ba thứ tự luồng của trình giả lập, so với một bản tham chiếu CPU có ô ma bằng 0 (chính xác, các số nguyên nhỏ) | mọi đầu ra đúng trong cả 18 lần chạy, exit 0, không có báo cáo AddressSanitizer |
| `check/wrap_causal_conv1d_test.cu` | `causalConv1d` cho mọi n từ 1 đến 400 theo ba thứ tự luồng, bên cạnh một bản đã sửa được tạo cho phép kiểm tra (không trả về sớm; mọi luồng đều nạp, một số 0 ma ở sau cuối mảng, và tới rào chắn trước khi kiểm tra `i < n`) | kernel của trang sai đúng tại n = 1, 2, 129, 130, 257, 258, 385 và 386 ở mọi thứ tự; bản đã sửa đúng với mọi n |
