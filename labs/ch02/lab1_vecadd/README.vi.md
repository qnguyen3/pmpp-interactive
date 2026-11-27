> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 2.1: cộng vector, từ đầu đến cuối

Sách: các mục 2.3 đến 2.6 (pp. 26-39). Kernel là Fig. 2.10 (p. 35), hàm host là Fig. 2.13
(p. 38), phần kiểm tra lỗi là khung bên lề ở p. 32. Phiên bản Metal là của khóa học này, không
phải của sách.

Hãy viết toàn bộ chương trình của chương. Chương trình điều khiển (driver) đóng vai `main` trong
Fig. 2.4: nó sở hữu ba mảng host thông thường và gọi `vecAdd` của bạn. Nó không bao giờ thấy một
con trỏ device.

**CUDA, HIP và trình giả lập (emulator):** `work/kernel.cu`

- `CHECK(call)`: mẫu ở p. 32 dưới dạng một macro, viết một lần. Đoạn mã trên trang sách không biên
  dịch được như khi in; hãy tìm hiểu tại sao trước khi sao chép nó. Bản của bạn in ra chuỗi lỗi, tên
  tệp và số dòng, không giải phóng gì, và khiến hàm bao quanh trả về một giá trị khác không. Không
  dùng `exit()`: chương trình điều khiển kiểm tra điều xảy ra sau một lỗi.
- `vecAddKernel(A, B, C, n)`: mỗi luồng (thread) một phép cộng từng cặp.
- `vecAdd(A_h, B_h, C_h, n, threadsPerBlock)`: cấp phát ba mảng trên device, sao chép A và B vào,
  tính số khối (block) cho kích thước khối được cho, khởi chạy, kiểm tra lần khởi chạy, chờ, sao
  chép C về, giải phóng. Mọi lời gọi runtime đều đi qua `CHECK`. Trả về 0, hoặc khác không khi có
  bất cứ thứ gì thất bại.

**Metal:** `work/kernel.metal` và `work/launch.mm`

- `vec_add`: kernel.
- `vecAdd(device, queue, pipeline, A_h, B_h, C_h, n)`: tạo ba buffer, encode, dispatch đúng `n`
  luồng, commit, chờ, kiểm tra, sao chép kết quả ra. Metal không có mã lỗi: hãy kiểm tra mọi lần
  tạo có ra nil hay không, và kiểm tra trạng thái cùng lỗi của command buffer sau khi chờ.
- `makeVecAddPipeline` được cung cấp sẵn. Nó biến văn bản của `kernel.metal` thành một pipeline
  state. Chương trình điều khiển gọi nó một lần và giữ lại kết quả.

Tên và chữ ký hàm là cố định. `main.cpp` là chương trình điều khiển; bạn không sửa nó.

- Kiểm thử: `python3 labs/pmpp.py run 2.1`   Bắt đầu với `--backend emu`.
- Đo: `python3 labs/pmpp.py run 2.1 --bench` (chỉ trên thiết bị thật)

## Các test tìm gì

Đầu vào nằm trong [0.5, 1.5], nên không tổng đúng nào gần 0, và `C_h` bắt đầu là NaN. Các độ dài
nằm ở cả hai phía của mỗi kích thước khối: 1, 31, 32, 33, 255, 256, 257, 1000, 4097 (thiết bị thật
thêm 2^20 + 3). Trên CUDA, HIP và trình giả lập, mỗi độ dài chạy với 32 và với 256 luồng mỗi khối,
và một trường hợp dùng khối lớn nhất mà device báo cáo.

- NaN còn lại trong `C_h`: không có gì được sao chép về các phần tử đó.
- Phần đuôi sai: quá ít khối. Bộ nhớ device mà không luồng nào ghi chứa một mẫu đánh dấu trên trình
  giả lập, chứa bất cứ thứ gì có sẵn trước đó trên CUDA, và bằng không trên Metal.
- Một báo cáo AddressSanitizer trên trình giả lập: một truy cập ra ngoài một mảng, kèm dòng mã
  nguồn. Thiếu điều kiện chặn, kích thước tính theo phần tử ở chỗ cần tính theo byte, và giải phóng
  trước khi sao chép về đều dẫn đến đây. CUDA không ghi trong tài liệu phép kiểm tra biên nào cho
  mảng trên device, đó là lý do lần chạy đầu tiên là trên trình giả lập.
- Chỉ CUDA, HIP và trình giả lập: một lời gọi yêu cầu một khối lớn gấp đôi mức device cho phép. Nó
  phải trả về khác không mà không kết thúc chương trình, và lời gọi hợp lệ ngay sau đó phải trả về
  0 với các tổng đúng.

## Trước khi đo hiệu năng

Hãy viết ra một con số: ở kích thước lớn nhất, `vecAdd` từ đầu đến cuối của bạn nhanh hơn hay chậm
hơn vòng lặp CPU thông thường bao nhiêu lần? Lớn hơn 1 nghĩa là phiên bản GPU thắng.

Benchmark đo thời gian vòng lặp CPU của Fig. 2.4 và một lời gọi trọn vẹn tới `vecAdd` của bạn cho
ba độ dài (2^16, 2^20 và 2^24 phần tử, ít hơn nếu device báo cáo quá ít bộ nhớ), lấy kết quả tốt
nhất của vài lời gọi sau một lời gọi khởi động không tính giờ. Lời gọi khởi động giữ thời gian khởi
động của runtime nằm ngoài phép đo. Nó in ra tỉ số. Lab 2.2 sau đó cho thấy thời gian đã đi đâu.

## Trạng thái của lời giải tham chiếu

verified on Metal, emulator-checked, front-end-checked. unverified trên CUDA và trên HIP: chưa có
máy nào có một trong hai chạy nó.
