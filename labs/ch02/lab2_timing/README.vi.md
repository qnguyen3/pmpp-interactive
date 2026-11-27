> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 2.2: thời gian đi đâu?

Sách: mục 2.3, p. 28: nhận xét rằng việc chuyển việc ra ngoài một cách "trong suốt" có thể rất kém
hiệu quả vì phải sao chép qua lại, và rằng mã thực tế giữ dữ liệu lớn trên device. Sách không đưa
ra số đo thời gian nào. Phép đo và mọi thứ về Metal là của khóa học này.

Hãy gắn công cụ đo vào vòng đi và về của lab 2.1 và kiểm chứng nhận xét đó trên chính máy của bạn.
Mã khởi đầu (starter) đã chứa một `vecAdd` chạy được dưới một tên mới (lời giải tham chiếu của lab
2.1; hãy dán bản của bạn vào nếu bạn thích). Nó tính đúng các tổng và không đo gì cả.

**CUDA, HIP và trình giả lập (emulator):** `work/kernel.cu`

- Hoàn thành `vecAddTimed(A_h, B_h, C_h, n, syncAfterLaunch, phase_s)`. `phase_s[0..4]` nhận số
  giây dành cho cấp phát, sao chép vào, kernel, sao chép về, giải phóng.
- `syncAfterLaunch != 0`: pha kernel kết thúc sau một lời gọi `cudaDeviceSynchronize()` có kiểm
  tra. `syncAfterLaunch == 0`: đồng hồ được đọc ngay sau lần khởi chạy, không đồng bộ hóa. Đó chính
  là thí nghiệm. Các tổng phải đúng trong cả hai trường hợp.
- `launchVecAdd` và kernel được cung cấp sẵn và chạy được. Chương trình điều khiển (driver) tự đo
  thời gian `launchVecAdd`, với lần đồng bộ hóa riêng của nó, trên các mảng nó đã đặt sẵn trên
  device. Đó là phép đo "thường trú".

Bạn quyết định mỗi lần đọc đồng hồ và mỗi lần đồng bộ hóa đặt ở đâu. Các nhà cung cấp ghi trong tài
liệu hai cái bẫy (`notes/today/ch02-facts.md`, các mục A3, A5 và B1):

- Một lần khởi chạy trả về trước khi lưới (grid) chạy. Một bộ đếm giờ chỉ bao quanh lần khởi chạy
  chỉ đo được lần khởi chạy.
- Một lần sao chép hoặc giải phóng có thể che giấu thời gian chờ. NVIDIA gọi `cudaMemcpy` là đồng
  bộ "for most use cases", và một phiên bản cũ hơn của tài liệu hướng dẫn của họ liệt kê các lần sao
  chép nhỏ từ host sang device là bất đồng bộ. `cudaFree` "may perform implicit synchronization".
  `hipFree` luôn đồng bộ hóa trước. Công việc vẫn đang chạy khi một lời gọi như vậy bắt đầu sẽ bị
  tính vào pha chứa lời gọi đó.

**Metal:** `work/launch.mm` (`work/kernel.metal` được cung cấp hoàn chỉnh)

- Hoàn thành `vecAddTimed(device, queue, pipeline, A_h, B_h, C_h, n, phase_s)`. `phase_s[0..3]`
  nhận số giây dành cho tạo buffer và sao chép các mảng host vào, encode và commit, chờ, và sao chép
  kết quả ra. `phase_s[4]` nhận thời gian GPU mà command buffer báo cáo
  (`GPUEndTime - GPUStartTime`).
- Viết `vecAddResident(queue, pipeline, A, B, C, n)`: cùng lần khởi chạy đó trên các buffer mà
  chương trình điều khiển đã tạo với chế độ lưu trữ dùng chung và đã lấp đầy qua `contents`. Chương
  trình điều khiển đọc kết quả tại chỗ.
- Không có `syncAfterLaunch` trên Metal. Trên CUDA, lần sao chép về có chặn sẽ chờ kernel, nên bỏ
  đồng bộ hóa chỉ chuyển thời gian từ pha này sang pha khác. Trên Metal không có gì chờ thay bạn:
  đọc `contents` trước khi lần chờ trả về là một lỗi về thứ tự, không phải một lựa chọn đo thời
  gian.

`main.cpp` là chương trình điều khiển; bạn không sửa nó.

- Kiểm thử: `python3 labs/pmpp.py run 2.2`   Bắt đầu với `--backend emu`.
- Đo: `python3 labs/pmpp.py run 2.2 --bench` (chỉ trên thiết bị thật)

## Các test thấy được gì và không thấy được gì

- Các tổng đúng, với cả hai giá trị của `syncAfterLaunch`, cho các độ dài không phải bội của kích
  thước khối.
- Mọi pha đều được ghi, hữu hạn và không âm.
- Các pha của bạn cộng lại không vượt quá đồng hồ riêng của chương trình điều khiển bao quanh lời
  gọi (với độ chênh cho phép 5% và 50 micro giây) và ít nhất bằng một nửa của nó. Trên Metal, đó là
  bốn pha phía host; thời gian GPU nằm bên trong lần chờ, nên nó được kiểm tra riêng: khác 0, và
  không dài hơn lời gọi.
- `launchVecAdd` (CUDA, HIP, trình giả lập) và `vecAddResident` (Metal) cho ra các tổng đúng trên
  các mảng mà chương trình điều khiển sở hữu.

Những điều này cũng đúng trên trình giả lập, nơi mọi lời gọi đều đồng bộ. Việc một lần đồng bộ hóa
có đặt đúng chỗ hay không không phải là một test, và trên trình giả lập thì không thể là một test.
Chỉ benchmark trên thiết bị thật mới cho thấy điều đó.

## Trước khi đo hiệu năng

Hãy viết ra hai con số.

1. Ở kích thước lớn nhất, kernel chiếm bao nhiêu phần trăm thời gian từ đầu đến cuối?
2. CUDA và HIP: khi bỏ đồng bộ hóa, pha kernel của bạn sẽ đọc được bao nhiêu phần trăm so với giá
   trị khi có đồng bộ hóa, và pha nào sẽ tăng lên thay vào đó?
   Metal: thời gian của đường thường trú so với tổng thời gian của đường dùng mảng host thì thế nào?

Benchmark chạy ba độ dài (2^16, 2^20 và 2^24 phần tử, ít hơn nếu device báo cáo quá ít bộ nhớ). Mỗi
hàng là lần nhanh nhất trong vài lời gọi sau một lời gọi khởi động không tính giờ. Trên CUDA và HIP,
nó in năm pha của bạn hai lần, có và không có đồng bộ hóa, và đặt pha kernel của bạn cạnh thời gian
của chương trình điều khiển cho `launchVecAdd` trên dữ liệu thường trú: nếu hai số chênh nhau nhiều,
một lần đồng bộ hóa bị thiếu hoặc đặt sai chỗ. Trên Metal, nó in bốn pha của bạn và thời gian GPU,
rồi đến đường thường trú cùng số byte mà đường đó sao chép. Cả hai đều in thời gian của vòng lặp CPU.

## Trạng thái của lời giải tham chiếu

verified on Metal, emulator-checked, front-end-checked. unverified trên CUDA và trên HIP: chưa có
máy nào có một trong hai chạy nó. Vị trí đồng bộ hóa trong `solution/kernel.cu` theo đúng tài liệu
được trích dẫn ở trên và chưa được đo trên device CUDA hay HIP nào.
