> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 15.2: phân ô thanh ghi, các vòng lặp được hoán đổi

Sách: mục 15.4 (pp. 357-360), Figs. 15.8 và 15.9. Mốc dự án M3, giai đoạn 2 trên 5.

`work/` chứa lời giải của lab 15.1, đã đổi tên (`gemmRegTileKernel`, `launchGemmRegTile`,
`gemm_regtile`, `encodeGemmRegTile`), với thân của `mm` bị xóa. Viết lại `mm` theo cách
Fig. 15.9 làm: đưa vòng lặp chiều trong `i` ra ngoài cùng; với mỗi `i`, sao chép 8 giá trị
của luồng ở cột `i` trong các hàng `A_s` của nó vào `a_r[8]` và 8 giá trị ở hàng `i` trong
các cột `B_s` của nó vào `b_r[8]`; rồi cộng tích ngoài `a_r[row] * b_r[col]` vào cả 64 phần
tử của ô. Trải hoàn toàn cả năm vòng lặp, nếu không `a_r`, `b_r` và ô sẽ rời khỏi thanh ghi
(register).

## Tệp

- Sửa `mm` trong `work/kernel.cu` và trong `work/kernel.metal`. `work/launch.mm` đã hoàn
  chỉnh: phân ô thanh ghi thay đổi kernel, không thay đổi phần khởi chạy.

## Lệnh

    python3 labs/pmpp.py run 15.2 --backend emu
    python3 labs/pmpp.py run 15.2
    python3 labs/pmpp.py run 15.2 --bench

Các bài kiểm thử là của lab 15.1 (xem README của nó). Mã khởi đầu (starter) để nguyên
thì build được nhưng thất bại: `mm` của nó không cộng gì, nên mọi phần tử của C ra 0.

## Trước khi bạn đo hiệu năng

Đếm trước. Mỗi luồng, mỗi cặp ô, Fig. 15.6 đọc bộ nhớ chia sẻ (shared memory)
8 x 8 x 8 x 2 = 1024 lần; Fig. 15.9 đọc 8 x (8 + 8) = 128 lần, ít hơn 8 lần.
Rồi ghi ra **mức tăng tốc bạn kỳ vọng so với lab 15.1 ở 2048 x 2048 x 2048**, và cân nhắc:

- Trên thiết bị có 32 bank bốn byte, các lần đọc dải vẫn xung đột trong cách bố trí luồng
  này: dải A xung đột 2 chiều, dải B 4 chiều (16 lane đọc các word cách nhau 8). Đó là phép
  tính của khóa học này dựa trên quy tắc bank được NVIDIA ghi nhận
  (`notes/today/ch15-facts.md` E1).
- Các lần ghi không đổi: mỗi luồng vẫn ghi mảng 8 x 8 của nó bằng các lần ghi vô hướng rơi
  cách phần tử của luồng bên cạnh 8 phần tử.
- Một trình biên dịch trải hoàn toàn Fig. 15.6 sẽ thấy mọi chỉ số là hằng và có thể đã tự
  giữ các lần đọc lặp lại trong thanh ghi (lập luận của khóa học này, không phải phát biểu
  của nhà cung cấp). Việc hoán đổi làm cho sự tái sử dụng trở nên tường minh; liệu trước đó
  nó có bị thiếu hay không là điều phép đo cho bạn biết.
- Trên Metal, Apple không ghi nhận cả bank lẫn số thanh ghi mỗi luồng (C2, C3): không có gì
  được ghi nhận dự đoán được kết quả.

`--bench` thêm lời giải của lab 15.1 thành cột `15.1` (được sao chép vào bản build khi bạn
cấu hình), bên cạnh lời giải của lab 5.3, của bạn và của nhà cung cấp.
