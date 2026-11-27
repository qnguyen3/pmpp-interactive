> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 9.2: riêng tư hóa trong bộ nhớ chia sẻ, gộp việc và gộp dải

Sách: các mục 9.4 đến 9.6 (pp. 211-219), Figs. 9.10, 9.14 và 9.15.

Hãy viết hai histogram ảnh 256 bin giữ một bản sao riêng của các bin trong bộ nhớ chia sẻ (shared memory):

- `histogramShared` (Metal: `histogram_shared`), Fig. 9.14. Đặt các bin riêng của khối (block) về 0
  bằng một vòng lặp theo bước nhảy (stride), rào chắn (barrier), mỗi luồng (thread) đếm `coarseFactor` điểm ảnh với cách đánh
  chỉ số xen kẽ và phép toán nguyên tử (atomic) phạm vi khối trên các bin riêng, rào chắn, rồi cộng mỗi
  bin riêng khác 0 vào các bin công khai bằng một phép toán nguyên tử phạm vi device.
- `histogramRuns` (Metal: `histogram_runs`), Fig. 9.15. Giống như trên, cộng thêm một bộ đếm dải cho mỗi
  luồng: các điểm ảnh liên tiếp của một luồng có cùng giá trị được đếm trong một thanh ghi (register)
  và chỉ tốn một phép toán nguyên tử khi giá trị thay đổi.

Hệ số gộp việc (coarsening) là một đối số lúc chạy, `coarseFactor`. Sách dùng một macro lúc biên dịch,
`COARSE_FACTOR`; một đối số cho phép một bản dựng được đo ở nhiều hệ số.

- Sửa `work/kernel.cu` (hai kernel) và `work/kernel.metal` (cùng hai kernel đó bằng Metal).
  Các hàm khởi chạy bên dưới kernel và `work/launch.mm` đã được viết sẵn cho bạn: 256 luồng
  mỗi khối (trên Metal, ít hơn nếu pipeline không chạy được 256), và ceil(n / (coarseFactor x
  256)) khối.
- `provided/` chứa hai kernel so sánh của benchmark, mà bạn không sửa:
  `refHistogramBasic` (Fig. 9.6, mỗi điểm ảnh một luồng) và `refHistogramContiguous`
  (Fig. 9.12, riêng tư hóa (privatization) trong bộ nhớ chia sẻ với cách đánh chỉ số liền kề). Chúng cũng được kiểm thử.
- Trên Metal, các hàm atomic chỉ nhận kiểu atomic. Các bin riêng là một mảng `threadgroup
  atomic_uint` khai báo trong thân kernel, được đặt về 0 bằng `atomic_store_explicit` và
  được đọc trong phần ghi nhận bằng `atomic_load_explicit`; `work/kernel.metal` có bảng
  từ vựng. Đặc tả MSL không có ví dụ nào về một mảng atomic threadgroup khai báo theo
  cách đó; đó là dạng mà mã thực tế đang dùng (`notes/today/ch09-facts.md` C1). Nếu trình biên dịch Metal
  của bạn từ chối nó, hãy dùng một tham số `threadgroup atomic_uint* [[threadgroup(0)]]` và
  `setThreadgroupMemoryLength`, như histogram MPS của PyTorch làm (tham số trong
  `HistogramKernel.metal`, độ dài được đặt trong `HistogramKernel.mm`; D2).
- Kiểm thử: `python3 labs/pmpp.py run 9.2`. Bắt đầu với `--backend emu`, rồi `--backend race`.
- Đo (chỉ GPU): `python3 labs/pmpp.py run 9.2 --bench`.

## Các phép kiểm thử làm gì

Mỗi trường hợp so sánh cả 256 số đếm với vòng lặp tuần tự của Fig. 9.2 trong `main.cpp`,
một cách chính xác. Chương trình điều khiển (driver) đặt `bins` về 0 trước mỗi lần khởi chạy.

- Các ảnh, rộng x cao: 1 x 1, 255 x 1, 256 x 1, 257 x 1, 64 x 64, 100 x 100; trên GPU
  còn có 1024 x 1024 và 4096 x 1025. Phần lớn khiến khối cuối cùng có ít điểm ảnh hơn số
  luồng.
- Các hệ số gộp việc 1, 3, 4 và 16.
- Bốn loại ảnh, từ các seed cố định: byte ngẫu nhiên phân bố đều, một ảnh hằng (mọi điểm ảnh
  là 200), một bầu trời (mỗi hàng một giá trị, thay đổi chậm từ hàng này sang hàng khác), và các sọc (các dải 8
  byte bằng nhau).
- Trên GPU, mỗi cấu hình được khởi chạy 5 lần.

Mỗi backend có thể và không thể cho bạn thấy gì ở đây:

- Bộ nhớ chia sẻ mà không ai đặt về 0 sẽ chứa các byte `0xCD` trên `emu` và `race`, mới cho mỗi
  khối, nên thiếu vòng lặp đặt về 0 sẽ luôn sai ở đó. GPU có thể đưa cho bạn các số 0, và
  khi đó các phép kiểm thử vượt qua trên nó. Vượt qua ở đó không phải là bằng chứng.
- Một rào chắn bị thiếu sẽ sai trên `emu` ở ít nhất một trong ba thứ tự luồng, và
  `race` báo cáo nó cùng các dòng liên quan.
- `race` chạy một khối 256 luồng riêng lẻ (nó chạy các khối lớn hơn 128 luồng
  từng khối một), nên nó không thể thấy hai khối tranh chấp nhau trong phần ghi nhận. Một `+=` thường ở đó có thể
  hiện ra trên GPU của bạn dưới dạng số đếm bị mất.
- Hai lỗi vượt qua mọi phép kiểm thử. Một vòng lặp đặt về 0 hoặc ghi nhận giả định mỗi bin một luồng
  (`b < blockDim.x`, hoặc không có vòng lặp nào) cho số đếm đúng, vì các hàm khởi chạy dùng
  256 luồng và có 256 bin; nó vẫn sai với mọi kích thước khối khác, nên hãy viết
  các vòng lặp như sách làm (lab 9.1 kiểm thử kernel pool của nó ở 64 và 128 luồng). Và
  cách đánh chỉ số liền kề (Fig. 9.12) cho số đếm đúng: một luồng lấy điểm ảnh nào
  thay đổi tốc độ, không thay đổi kết quả. Chỉ benchmark mới cho thấy điều đó.

## Trước khi đo hiệu năng

`--bench` đo thời gian, trên 2^26 điểm ảnh (8192 x 8192, nên một hàng bầu trời dài; ít hàng hơn nếu
thiết bị báo quá ít bộ nhớ), byte phân bố đều, ảnh hằng và bầu trời:
`basic` (được cung cấp), `contiguous` (được cung cấp, coarse 16), `shared` của bạn ở coarse 1, 4, 16
và 64, và `runs` của bạn ở coarse 16. Nó in Gpixels/s cho mỗi kernel, lần tốt nhất trong 5 lần chạy, và
ba phép so sánh bên dưới. Hãy viết ra trước:

1. `shared` ở coarse 16 nhanh hơn `basic` bao nhiêu lần trên ảnh hằng
   (mục 9.4 cũng hỏi điều này).
2. Cách xen kẽ (`shared`) nhanh hơn `contiguous` bao nhiêu lần ở coarse 16 trên byte phân bố
   đều (mục 9.5 cũng hỏi điều này). Lập luận của sách là hợp nhất truy cập bộ nhớ (coalescing): trong một bước, các
   luồng của một warp nên đọc các byte lân cận (p. 216). Tài liệu của NVIDIA mô tả quy tắc đó là
   các giao dịch 32 byte mỗi warp (A7) và AMD là các yêu cầu theo dòng bộ nhớ đệm (B3). Tài liệu của Apple không nêu
   quy tắc hợp nhất nào (C4), nên trên Mac, phép đo này là bằng chứng duy nhất bạn có.
3. Liệu `runs` có thắng `shared` trên bầu trời, và trên byte phân bố đều hay không (trang labs ghi lại
   dự đoán này).

Các con số là phép đo trên thiết bị của bạn, không phải của GPU nói chung. Việc đặt `bins` về 0 trước
mỗi lần chạy không được tính thời gian.
