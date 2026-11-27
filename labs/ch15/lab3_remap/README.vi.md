> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 15.3: lần ghi được hợp nhất, bốn ô con 4 x 4 và lần ghi float4

Sách: mục 15.5 (pp. 360-361), Fig. 15.10; Bài tập 3 và 2 (p. 370). Sách không cho mã cho mục
này. Mốc dự án M3, giai đoạn 3 trên 5.

Trong lab 15.1 và 15.2, mỗi luồng ghi một mảng 8 x 8, nên trong một lệnh ghi, các luồng cạnh
nhau của một warp ghi vào các địa chỉ cách nhau 8 số float. `work/` chứa lời giải của lab
15.2, đã đổi tên và biến thành một template theo độ rộng warp `W`. Hãy ánh xạ lại các luồng
như mục 15.5 mô tả, và ghi mỗi hàng của một ô con bằng một lần ghi `float4`.

Cách bố trí, cho một warp gồm W luồng (số của sách là W = 32):

- W lane của một warp tạo thành 8 hàng lane, mỗi hàng `LC = W / 8` lane;
- ô của mỗi warp được chia thành 2 x 2 góc phần tư, mỗi phần 32 hàng và `4 * LC` cột, và một
  lane sở hữu ô con 4 x 4 ở cùng vị trí trong cả bốn góc phần tư: ô 8 x 8 logic của nó;
- do đó ô của một warp là 64 x `8 * LC`, và `128 / (8 * LC)` warp nằm ngang qua ô của khối.

Với W = 32, đó là 8 warp của sách xếp 2 x 4, ô warp 64 x 32, góc phần tư 32 x 16, lane xếp
8 x 4. Với W = 64 (AMD CDNA), cùng các quy tắc cho 4 warp xếp 2 x 2, ô warp 64 x 64 và góc
phần tư 32 x 32 (`notes/today/ch15-facts.md` B1). Hãy tự tính bằng tay lane 5 của warp 6
cho cả hai độ rộng trước khi viết mã.

## Tệp

- `work/kernel.cu`: phép ánh xạ (TODO 1), `mm` trên bốn ô con (TODO 2), `writeTile` với các
  lần ghi `float4` (TODO 3), và `launchGemmRemap` (TODO 4), hàm đọc độ rộng warp của thiết bị
  một lần và gọi `launchGemmRemapWidth`.
  `launchGemmRemapWidth`, hàm khởi chạy bản khởi tạo cho một độ rộng cho trước, đã được viết
  sẵn cho bạn: các bài kiểm thử gọi nó với cả hai độ rộng, và truy vấn occupancy của
  benchmark tham chiếu đến cả hai bản khởi tạo, nên mã khởi đầu (starter) phải tạo ra chúng.
- `work/kernel.metal` (`gemm_remap`) và `work/launch.mm` (`encodeGemmRemap`): tương tự trong
  Metal. Host biên dịch kernel với `SIMD_WIDTH` được định nghĩa, và bạn gắn C lần thứ hai
  dưới dạng `device float4*` cho các lần ghi vector, như trong lab 6.2.

Không bao giờ viết độ rộng warp thành một con số bên trong kernel. Trên HIP nó khác nhau giữa
các thiết bị và "can not be assumed to be a compile-time constant on the host"; cách AMD
khuyến nghị là cách được dùng ở đây, một bản khởi tạo template do host chọn (B1).

## Lệnh

    python3 labs/pmpp.py run 15.3 --backend emu
    python3 labs/pmpp.py run 15.3
    python3 labs/pmpp.py run 15.3 --bench

## Các bài kiểm thử kiểm tra gì

Các bài kiểm thử của lab 15.1, mỗi bài chạy với cách bố trí cho W = 32 và cho W = 64 trên mọi
backend, kể cả các trình giả lập: cách bố trí là phép tính chỉ số, nên đúng hay sai không phụ
thuộc vào độ rộng của phần cứng. Trên CUDA và HIP, phần khởi chạy đọc độ rộng của thiết bị
cũng được chạy. Trên Metal, chương trình điều khiển (driver) build `SIMD_WIDTH` 32 và 64 cho
các bài kiểm thử, và cho benchmark thì dùng độ rộng mà pipeline báo. Mã khởi đầu để nguyên
thì build được nhưng thất bại: `writeTile` không ghi gì, nên C vẫn là NaN.

Các hình dạng với N = 1, 3, 7, 9, 131 và 511 cần đường dự phòng vô hướng. Một điều trình giả
lập không thấy được: một lần ghi `float4` vào địa chỉ không căn chỉnh 16 byte vẫn chạy tốt
trên CPU, nên bỏ điều kiện căn chỉnh vẫn qua trên `emu`. Trên Metal, cùng lỗi đó ghi vào sai
phần tử (chỉ số của khung nhìn `float4` bị làm tròn xuống) và thất bại. Với phép ép kiểu con
trỏ của sách trên CUDA, blog nhà phát triển của NVIDIA yêu cầu địa chỉ căn chỉnh 16 byte
(`notes/today/ch06-facts.md` A11), và `cudaMalloc` trả về bộ nhớ căn chỉnh ít nhất 256 byte
(A3), nên N là bội của 4 là đủ. Trên HIP, C đến từ `hipMalloc`, mà tài liệu tham khảo bộ nhớ
của AMD không nêu căn chỉnh nào (ch15-facts B8): việc cùng quy tắc đó là đủ ở đó vẫn là
unverified.

## Trước khi bạn đo hiệu năng

Dự đoán **mức tăng tốc so với lab 15.2 trên hình dạng K ngắn 4096 x 128 x 4096, và ở
2048 x 2048 x 2048.** Cân nhắc:

- Các lần ghi xảy ra một lần cho mỗi phần tử đầu ra, dù K dài bao nhiêu. Với K = 128 có 16
  cặp ô phải nạp và tính cho mỗi ô được ghi; với K = 2048 có 256. Vì vậy một mẫu ghi tốt hơn
  quan trọng nhất khi K ngắn.
- Việc ánh xạ lại sửa các lần ghi và bỏ xung đột bank (bank conflict) 4 chiều của dải B,
  nhưng nó nâng xung đột của dải A từ 2 chiều lên 8 chiều trên 32 bank: tám hàng lane đọc các
  hàng của `A_s` cách nhau 4, tức cách nhau 32 word (E1). Lab 15.4 thêm phần đệm để bỏ nó.
- Trên Metal, Apple không ghi nhận quy tắc hợp nhất truy cập bộ nhớ (coalescing) nào: không
  có gì được ghi nhận dự đoán được kết quả.

## Vượt ra ngoài sách

- Không phải thư viện nào cũng giải mục 15.5 theo cách này. CUTLASS giữ cách bố trí luồng tốt
  nhất cho phần số học và đưa ô đầu ra qua bộ nhớ chia sẻ (shared memory) trong một phần kết
  (epilogue) riêng, rồi ghi nó bằng các truy cập theo sọc, được hợp nhất (A12); kernel Metal
  của llama.cpp làm tương tự cho các ô ở rìa (D2).
- Metal: lần ghi `float4` đi qua một lần gắn thứ hai của C được khai báo `device float4*`,
  dạng vector được Apple ghi nhận, thay vì phép ép kiểu con trỏ của sách
  (`notes/today/ch06-facts.md` C3).
