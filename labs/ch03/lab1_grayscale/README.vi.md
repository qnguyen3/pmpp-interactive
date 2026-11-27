> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 3.1: ảnh màu sang ảnh xám

Sách: mục 3.2 (pp. 48-55), Fig. 3.4 (p. 53) và lần khởi chạy ở p. 50.

Hãy viết kernel biến một ảnh RGB thành ảnh xám, và lần khởi chạy của nó: mỗi điểm ảnh một luồng
(thread), một lưới (grid) 2D gồm các khối (block) 2D được làm tròn lên thành số khối nguyên, một
phép kiểm tra phạm vi cho các luồng rơi ra ngoài ảnh, và hai độ lệch phẳng (một byte mỗi điểm ảnh ở
đầu ra, ba byte ở đầu vào). Mức xám là L = 0.299 r + 0.587 g + 0.114 b, cắt bớt thành một byte.

- Sửa `work/kernel.cu` (CUDA / HIP / trình giả lập): `colorToGrayscaleKernel` và
  `launchGrayscale`. Hình dạng khối là một đối số của lần khởi chạy; đừng viết cứng 16.
- Sửa `work/kernel.metal` + `work/launch.mm` (Metal): kernel đọc các buffer `uchar` thông thường
  và lấy vị trí của nó từ `thread_position_in_grid`; lần khởi chạy gắn hai kích thước bằng
  `setBytes` và dùng `dispatchThreads`, nên lưới đúng bằng bức ảnh.
- Kiểm thử: `python3 labs/pmpp.py run 3.1`   Bắt đầu với `--backend emu`.
- Đo: `python3 labs/pmpp.py run 3.1 --bench`

Kích thước được viết theo cách sách viết, hàng trước: 62 nhân 76 là 62 hàng, mỗi hàng 76 điểm ảnh,
nên `height` là 62 và `width` là 76.

## Đừng sao chép Fig. 3.4

Lời giải khác với đoạn mã in trong sách ở bốn chỗ. Nếu muốn, hãy tự tìm chúng trước; nếu không, các
test sẽ tìm giúp bạn.

1. **Phép ép kiểu.** Dòng 19 in `(unsigned char) 0.21f*r+0.71f*g+0.07f*b`. Phép ép kiểu gắn chặt
   hơn `*`, nên nó chỉ áp dụng cho `0.21f`, giá trị này thành 0, và số hạng màu đỏ bị mất. Trình
   biên dịch không nói gì. Hãy đặt cả tổng trong ngoặc.
2. **Các hằng số.** Đoạn mã in 0.21, 0.71, 0.07. Phần văn bản ở p. 52 cho 0.299, 0.587, 0.114, là
   các giá trị lab này dùng. Chúng cộng lại bằng 1, nên màu trắng vẫn là trắng; các hằng số của đoạn
   mã cộng lại bằng 0.99.
3. **Thứ tự đối số.** Đoạn mã nhận `(Pout, Pin, width, height)`, và lời gọi host ở p. 50 truyền
   `(Pin_d, Pout_d, m, n)`. Ở đây đầu vào đứng trước ở cả hai chỗ.
4. **Tên hàm.** Đoạn mã in `colortoGrayscaleConvertion`, phần văn bản in
   `colorToGrayscaleConversion`. Ở đây nó là `colorToGrayscaleKernel`.

Hai điều nhỏ hơn: `Pin` ở đây là `const`, và kích thước lưới được tính bằng phép làm tròn lên số
nguyên ở chỗ p. 50 in `ceil(m/16.0)`.

Kernel Metal không có trong sách. Nó có cùng cái bẫy ép kiểu, vì ngữ pháp giống nhau. Nó có một cái
bẫy riêng: `uchar3` chiếm bốn byte trong Metal Shading Language, nên một con trỏ `uchar3` trên các
điểm ảnh ba byte sẽ đọc sai dữ liệu.

## Các test kiểm tra gì

Đầu ra của bạn được so sánh với một bản tham chiếu CPU, cho phép lệch một mức xám về mỗi phía. Phép
ép kiểu cắt bớt, nên thứ tự mà trình biên dịch cộng ba tích có thể làm byte lệch đi một.

- Các ảnh một màu trước: đỏ, xanh lá và xanh dương thuần (76, 149 và 29) và trắng (255).
- Rồi một dải chuyển màu có nhiễu ở 62 nhân 76 (bức ảnh của sách), 16 nhân 16, 1 nhân 1, 1 nhân
  300, 300 nhân 1, 33 nhân 17 và 64 nhân 80, và trên thiết bị thật là 1500 nhân 2000. Mỗi trường hợp
  chạy với các khối 16 nhân 16, 32 nhân 8 và 8 nhân 32 luồng.
- Buffer đầu ra được lấp đầy bằng một giá trị canh (sentinel) trước mỗi lần khởi chạy. Một điểm ảnh
  vẫn còn giữ giá trị đó là chưa bao giờ được ghi, và thông báo lỗi đếm những điểm ảnh như vậy.
- Trên trình giả lập (emulator), các buffer đúng bằng kích thước bức ảnh và bản dựng dùng
  AddressSanitizer, nên truy cập đầu tiên ra ngoài một buffer sẽ dừng lần chạy và nêu dòng mã trong
  kernel của bạn. Trên thiết bị thật không có gì kiểm tra biên: ở đó các buffer lớn hơn bức ảnh và
  các byte phía sau nó phải được trả về không thay đổi.

## Trước khi đo hiệu năng

Benchmark đo thời gian một bức ảnh lớn với các khối 8 nhân 8, 16 nhân 16, 32 nhân 32, 256 nhân 1 và
1 nhân 256 luồng, bỏ qua mọi hình dạng mà device của bạn không cho phép.

Một khối rộng (256 nhân 1) và một khối cao (1 nhân 256) khởi chạy cùng số luồng, và mọi luồng làm
cùng một lượng việc. Điều khác nhau là các luồng của một khối giữ những điểm ảnh nào: một dải dọc
theo một hàng, hay một dải xuống theo một cột. Tài liệu lập trình của NVIDIA nói "The use of
multi-dimensional thread blocks and grids is for convenience only and does not affect performance";
câu đó nói về việc viết một ánh xạ trong một, hai hay ba chiều. Cùng đoạn đó viết tiếp: các luồng
được đánh số với x thay đổi nhanh nhất, và "This affects how threads are assigned to warps". Hãy
viết ra tỉ số bạn dự đoán, thời gian khối cao chia cho thời gian khối rộng, rồi chạy. Benchmark in
ra tỉ số nó đo được. Vì sao kết quả lại như vậy là chủ đề của chương 4 và 6.
