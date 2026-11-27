> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 2.3: các phép toán từng phần tử mà một transformer cần (mốc dự án M0)

Sách: các mục 2.3 đến 2.6 (pp. 26-39). Dạng kernel là Fig. 2.10 (p. 35), phép biến đổi affine là
Exercise 9 (Fig. 2.15, p. 42), hàm trợ giúp dùng chung là hàng `__host__ __device__` của Fig. 2.11
(p. 36) và Exercise 10. Sách không áp dụng điều nào trong số này cho bias, scale hay residual, và
không có Metal: những phần đó là của khóa học này.

Hãy viết ba kernel, hàm khởi chạy bên cạnh mỗi kernel, và một hàm host nối chúng thành chuỗi trên
device:

- `addBiasKernel`: `Y[r][j] = X[r][j] + bias[j]` cho một tensor `rows x d` và một bias gồm `d` giá
  trị. Mỗi phần tử một luồng (thread). Lưới (grid) là phẳng, tensor thì không.
- `scaleShiftKernel`: `Y[i] = alpha * X[i] + beta`, thông qua `scaleShift`, một hàm trợ giúp mà
  trình biên dịch dựng hai lần, một lần cho host và một lần cho device.
- `addResidualKernel`: `X[i] += R[i]`, tại chỗ.
- `launchAddBias`, `launchScaleShift`, `launchAddResidual`: mỗi hàm tự tính số khối (block) của nó
  cho 256 luồng mỗi khối và khởi chạy. Không làm gì khác.
- `blockTail`: `y = alpha * (x + bias) + beta + r` cho các mảng host. Một lượt sao chép vào, ba lần
  khởi chạy nối tiếp nhau, một lần kiểm tra và một lần chờ, một lần sao chép ra, rồi giải phóng.

Những hàm này theo các quy ước mà mọi mốc dự án sau đều dùng: tensor `float` theo thứ tự hàng
(row-major) với chiều batch đứng đầu, con trỏ device thô và kích thước tường minh làm đối số kernel,
một hàm `launch<Name>` bên cạnh mỗi kernel, bản tham chiếu CPU trong `main.cpp`. Một hàm khởi chạy
chỉ tính lưới của nó và khởi chạy. Việc chờ và đọc trạng thái lỗi là việc của bên gọi, để bên gọi
có thể xếp hàng nhiều lần khởi chạy trên dữ liệu vẫn nằm trên device. `launchAddBias`,
`launchScaleShift` và `launchAddResidual` là những mảnh mà các mốc dự án sau sẽ gọi.

**CUDA, HIP và trình giả lập (emulator):** `work/kernel.cu`. `scaleShiftOnHost` và macro `CHECK`
được cung cấp sẵn.

**Metal:** `work/kernel.metal` (hàm trợ giúp `scale_shift` và các kernel `add_bias`,
`scale_shift_k`, `add_residual`) và `work/launch.mm` (`encodeAddBias`, `encodeScaleShift`,
`encodeAddResidual`, và `blockTail`). Mỗi hàm encode ghi một lần dispatch vào một encoder mà bên gọi
đã mở. `blockTail` đặt cả ba lần dispatch vào một command buffer và chờ một lần. Apple ghi trong tài
liệu rằng `computeCommandEncoder` thông thường chạy các lần dispatch của nó tuần tự và rằng các
buffer tạo từ device được theo dõi xung đột (hazard-tracked) theo mặc định, nên mỗi lần dispatch
thấy được những gì lần trước nó đã ghi (`notes/today/ch02-facts.md`, mục C9).

Trên Metal không có `scaleShiftOnHost`, và không có test cho nó. Mọi hàm trong một tệp `.metal` đều
là mã GPU, viết bằng một ngôn ngữ riêng và biên dịch tách biệt với chương trình host, và Apple không
ghi trong tài liệu cách nào để dựng một định nghĩa cho cả CPU lẫn GPU
(`notes/today/ch02-facts.md`, mục C7). `scale_shift` trong `kernel.metal` chỉ đóng vai một hàm trợ
giúp `__device__`. Các dự án cần cùng định nghĩa ở cả hai phía của Metal, trong đó có llama.cpp,
dùng chung một header thuần chứa hằng số và struct, không phải hàm (mục D2).

Tên và chữ ký hàm là cố định. `main.cpp` là chương trình điều khiển (driver); bạn không sửa nó.

- Kiểm thử: `python3 labs/pmpp.py run 2.3`   Bắt đầu với `--backend emu`.
- Đo: `python3 labs/pmpp.py run 2.3 --bench` (chỉ trên thiết bị thật)

## Các test tìm gì

Mỗi hàm khởi chạy được kiểm thử riêng, trên các tensor mà chương trình điều khiển đã giữ sẵn trên
device, với các kích thước `1 x 1`, `1 x 257`, `3 x 100`, `4 x 256` và `7 x 33` (thiết bị thật thêm
`64 x 4099`). Chỉ một trong số đó có số phần tử là bội của 256.

- Mọi đầu vào nằm giữa 1 và 4 và mọi đầu ra đúng đều cách xa 0. Đầu ra bắt đầu là NaN, nên một phần
  tử không luồng nào ghi sẽ bị báo cáo đúng như vậy.
- Bias khác nhau theo từng cột, và mọi kích thước trừ `1 x 1` đều có ít hàng hơn cột: một bias đánh
  chỉ số theo hàng, hoặc hàng và cột bị hoán đổi, sẽ cho giá trị sai.
- Đầu vào được đọc lại sau lời gọi và phải không thay đổi. `addResidual` được kiểm tra tại chỗ.
- Kernel scale được so sánh với bản tham chiếu CPU và, trên CUDA, HIP và trình giả lập, so từng phần
  tử với `scaleShiftOnHost`, trong dung sai của harness. Không phải chính xác đến từng bit: một
  trình biên dịch có thể gộp phép nhân và phép cộng thành một phép toán chỉ làm tròn một lần, và
  bản biên dịch cho host và cho device không nhất thiết thống nhất về điều đó.
- Một luồng vượt quá cuối tensor: trên trình giả lập, AddressSanitizer dừng lần chạy và nêu dòng
  kernel. Trên GPU không có gì đảm bảo sẽ báo cáo điều này, nên ở đó chương trình điều khiển đặt một
  khối giá trị đánh dấu phía sau mỗi tensor và kiểm tra chúng còn nguyên.
- `blockTail` được so sánh với tổ hợp CPU của ba phép toán trên cùng các kích thước. Nó phải trả về
  0, không động đến đầu vào, và ghi đè mọi phần tử của `y_h` (bắt đầu là NaN).

## Trước khi đo hiệu năng

Hãy viết ra một con số: ở kích thước benchmark chọn, một chuỗi thường trú gồm ba phép toán nhanh hơn
bao nhiêu lần so với cùng ba phép toán đó làm thành ba vòng đi và về?

Mục 2.3 đếm được 9 lần truyền cỡ tensor so với 5 cho một chuỗi trong đó mỗi phép toán mang theo cả
một tensor của riêng nó. Ở đây bias là một hàng và scale là hai con số, nên trước hết hãy đếm số lần
sao chép cỡ tensor mà mỗi biến thể của chuỗi này thực hiện. Device của bạn quyết định con số đó
tương ứng với bao nhiêu thời gian.

Benchmark tăng dần tensor (4096 cột, số hàng gấp đôi mỗi lần) cho đến khi một kernel chạy đủ lâu để
đo, tối đa 2^26 phần tử, và dừng sớm nếu một lần cấp phát thất bại. Sau đó nó:

- đo thời gian từng kernel riêng trên dữ liệu đã nằm trên device và in lưu lượng bộ nhớ toàn cục
  (global memory) mà nó yêu cầu mỗi giây (12 byte mỗi phần tử cho hai phép cộng, 8 cho phép scale).
  Đó là lưu lượng được yêu cầu, không phải lưu lượng quan sát được: `addBias` yêu cầu cùng một hàng
  bias lặp đi lặp lại;
- đo thời gian một lời gọi `blockTail` của bạn so với cùng ba phép toán đó làm thành ba lần chuyển
  việc ra device riêng biệt mà chương trình điều khiển ghép từ các hàm khởi chạy (hoặc encode) của
  bạn, mỗi lần có sao chép vào và sao chép ra riêng, và in ra tỉ số.

Trên Metal không có bus nào phải vượt qua. Các vòng đi và về sao chép mảng host vào buffer dùng
chung và sao chép kết quả ra mỗi lần (mục C3): thứ được đo ở đó là chi phí của một giao diện nhận
mảng host. Dữ liệu giữ trong buffer dùng chung sẽ không bị sao chép chút nào.

## Trạng thái của lời giải tham chiếu

verified on Metal, emulator-checked, front-end-checked. unverified trên CUDA và trên HIP: chưa có
máy nào có một trong hai chạy nó.
