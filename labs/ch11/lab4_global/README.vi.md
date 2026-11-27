> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 11.4: scan độ dài bất kỳ, theo phân cấp

Sách: mục 11.9 (pp. 273-276), Figs. 11.14 và 11.15.

Các lab 11.1 và 11.2 scan đoạn của từng khối một cách riêng rẽ. Ở đây bạn nối các đoạn thành một
scan bao gồm (inclusive scan) trên toàn bộ n phần tử, với hai thiết kế ba kernel mà sách vẽ:

- **scan-scan-add** (Fig. 11.14). Kernel 1 scan mọi đoạn và ghi tổng của từng đoạn. Kernel 2 scan
  các tổng. Kernel 3 cộng tổng đã scan của mọi đoạn trước đó vào từng phần tử của một đoạn. Sách
  đếm 16 + 8/S byte bộ nhớ toàn cục (global memory) mỗi phần tử, với S phần tử mỗi đoạn (p. 274).
- **reduce-scan-scan** (Fig. 11.15). Kernel 1 chỉ rút gọn mọi đoạn thành tổng của nó (một phép
  rút gọn (reduction) của chương 10). Kernel 2 scan các tổng. Kernel 3 cộng tổng của các đoạn
  trước vào phần tử đầu tiên của đoạn rồi scan đoạn đó, cho ra trực tiếp các giá trị cuối
  cùng. Sách đếm 12 + 12/S byte mỗi phần tử (p. 275).

Sách không in mã cho cả hai, và kernel 2 của sách scan các tổng "in a single block" (p. 273).
Điều đó giới hạn n ở S × S phần tử. Ở đây kernel 2 lại chính là toàn bộ thiết kế, áp dụng cho
các tổng, cho đến khi chúng vừa trong một đoạn. Vì vậy n nào cũng được.

Hợp đồng. Một khối gồm `block` luồng sở hữu một đoạn S = `COARSE_FACTOR` × `block` phần tử liên
tiếp. `COARSE_FACTOR` là 4, được định nghĩa một lần cho mỗi backend (`kernel.cu` cho CUDA,
`launch.mm` cho Metal); chương trình điều khiển (driver) đọc nó qua `coarseFactor()`. Kích thước
khối là lũy thừa của hai từ 64 đến 1024. n chạy từ 1 đến dưới 2^31. Một phần tử ở vị trí n hoặc
sau đó nạp 0, phần tử đơn vị của +, và không được ghi. `scratch` chứa `scratchFloats(n, block)`
số float, và không có giả định nào về nội dung của nó lúc bắt đầu.

**Phần việc của bạn:**

- **`scanSegment`**: `RegisterTiledScanKernel` của bạn từ lab 11.2, chuyển thành một hàm
  device, với hai bổ sung. Nó cộng một `carry` vào phần tử đầu tiên của đoạn trước khi scan
  (kernel 3 của Fig. 11.15), và nó trả về tổng của đoạn.
- **`ScanSegmentsKernel` và `AddBlockSumsKernel`**: kernel 1 và 3 của Fig. 11.14.
- **`ReduceSegmentsKernel` và `ScanWithCarryKernel`**: kernel 1 và 3 của Fig. 11.15. Phép rút
  gọn bao phủ cùng đoạn mà phép scan bao phủ, với cùng các lần nạp được hợp nhất.
- **`scratchFloats`, `launchScanScanAdd`, `launchReduceScanScan`**: lượng scratch mà đệ quy
  cần, và các hàm khởi chạy. Mỗi hàm khởi chạy tự gọi lại chính nó trên các tổng cho kernel 2.
  Hãy bố trí scratch theo từng mức.
- **Metal**: cùng các kernel đó trong `kernel.metal`, và trong `launch.mm` là `scratchFloats`,
  `encodeScanScanAdd` và `encodeReduceScanScan`. Chúng ghi nhiều lần dispatch vào một compute
  encoder duy nhất mà chương trình điều khiển đã mở. Mỗi lần dispatch thấy những gì lần trước
  nó đã ghi. Apple ghi trong tài liệu hai lý do: `computeCommandEncoder` thông thường
  "dispatches its compute commands serially", và các buffer do device tạo ra được theo dõi
  xung đột (hazard-tracked) theo mặc định (ch02 facts C9). Các threadgroup không bao giờ trao
  đổi với nhau bên trong một lần dispatch.

## Tệp

- Sửa `work/kernel.cu` (CUDA, HIP và cả hai trình giả lập): `scanSegment`, bốn kernel,
  `scratchFloats`, `launchScanScanAdd` và `launchReduceScanScan`.
- Sửa `work/kernel.metal` (`scan_segment`, `scan_segments`, `add_block_sums`,
  `reduce_segments`, `scan_with_carry`) và `work/launch.mm` (`scratchFloats`,
  `encodeScanScanAdd`, `encodeReduceScanScan`). Mỗi hàm encode trả về false, không dispatch,
  cho một khối mà một trong các pipeline của nó không chạy được.
- Được cung cấp trong mã khởi đầu (starter), không thuộc bài tập: `warpScan` và `blockScan` từ
  lab 11.1, `warpReduce` từ lab 10.1, và trên Metal là phép kiểm tra `fits`. Bạn có thể thay
  bằng bản của mình nếu muốn.
- Được cung cấp, không sửa, trong `../shared-g13/`: `global_copy.cu` và `.metal` (một phép sao chép
  đơn giản, mốc băng thông) và `global_launch.mm`.

## Lệnh

- Kiểm thử: chạy `python3 labs/pmpp.py run 11.4 --backend emu` trước, rồi `python3 labs/pmpp.py run 11.4`.
- Rào chắn dưới ThreadSanitizer: `python3 labs/pmpp.py run 11.4 --backend race`.
- Đo: `python3 labs/pmpp.py run 11.4 --bench` (chỉ trên thiết bị thật).
- Lời giải tham chiếu: thêm `--solution`.

Lần chạy trên trình giả lập chậm, phần lớn thời gian nằm ở trường hợp ba mức bên dưới.

## Các bài kiểm thử làm gì

- Trên thiết bị thật: n = 1, 2, 255, 256, 257, 1000, 4096, 4097, 65536, 70001, 2^20 + 1 và
  2^24 + 7, với khối 64, 256 và 1024. Với khối 64 (S = 256), 2^24 + 7 là bốn mức: 65,537 đoạn,
  rồi 257, rồi 2, rồi 1.
- Trên `emu`: n tới 4097 với khối 64 và 256, cộng thêm n = 70001 với khối 64. Đó là 274 đoạn,
  các tổng của chúng tạo thành 2 đoạn, các tổng của chúng tạo thành 1: ba mức đệ quy. Trình
  giả lập chậm ở kích thước đó, nên nó chỉ chạy với các số nguyên.
- Trên `race`: n = 1000 và 5000 với khối 64, tức là 4 và 20 khối, nhiều khối cùng lúc.
- Ba loại dữ liệu. Toàn số một, và các số nguyên từ -3 đến 3, có tổng tiền tố chính xác theo
  mọi thứ tự cộng khi tổng |x| còn nhỏ hơn hoặc bằng 2^24, nên ở đó chúng phải khớp chính xác.
  Vượt quá mức đó (toàn số một ở 2^24 + 7), và với các giá trị phân bố đều trong [-1, 1], kết
  quả được so với vòng lặp của Fig. 11.1 chạy bằng double, trong phạm vi 1e-5 nhân tổng |x|
  tính đến phần tử đó, cộng 1e-6. Vì sao nhiều như vậy: mỗi đầu ra đi qua tối đa khoảng 16 phép
  cộng có làm tròn ở mỗi mức của phân cấp. Với 4 mức, trường hợp xấu nhất là
  64 × 2^-24 ≈ 3.8e-6 nhân tổng đó.
- Mỗi thiết kế chạy hai lần trên cùng các buffer. Lần khởi chạy đầu tiên bắt đầu với đầu ra và
  scratch đầy NaN, nên một tổng được đọc trước khi được ghi sẽ lộ ra. Lần khởi chạy thứ hai nhận
  dữ liệu mới và bất cứ thứ gì lần đầu để lại trong scratch, nên một kernel cộng dồn vào
  scratch thay vì ghi đè cũng sẽ lộ ra.
- Các từ canh gác nằm sau đầu ra và scratch. Một luồng nằm quá n mà ghi sẽ bị báo, và một hàm
  khởi chạy dùng nhiều scratch hơn `scratchFloats` đã yêu cầu cũng vậy. Đầu vào phải giữ
  nguyên.
- Kernel chép được cung cấp cũng được kiểm thử, nên một tệp cung cấp bị hỏng sẽ lộ ra.

Mỗi backend có thể bắt được gì:

- `emu` chạy các luồng lần lượt từng luồng theo ba thứ tự cố định, và lấp bộ nhớ chia sẻ
  (shared memory) bằng rác cho mọi khối. Một chỉ số sai, một rào chắn (barrier) bị thiếu và một
  truy cập ngoài biên đều thất bại ở đó mọi lần.
- `race` chạy các luồng của một khối, và nhiều khối, cùng lúc dưới ThreadSanitizer. Nó báo một
  rào chắn bị thiếu bên trong các kernel của bạn. Trong lab này, không phần nào nên truyền dữ liệu
  giữa các khối bên trong một lần khởi chạy. Mọi lần chuyển giao xảy ra giữa các lần khởi chạy,
  nên nó không có gì khác để báo.
- Các trình giả lập chậm và không mô hình hóa thời gian. Chỉ một thiết bị thật mới đo được cái
  giá của hai thiết kế.

## Chỗ các backend khác nhau (không có trong sách)

- **HIP**: như trong lab 11.1, mặt nạ lane cần kiểu 64 bit riêng, và `warpSize` là 64 trên gfx9
  và 32 trên các dòng AMD sau này (ch11 facts B1). Không có gì ở đây dùng atomic. Chưa được
  build hay chạy với ROCm: unverified.
- **Metal**: đây là scan toàn cục của Metal. Metal không ghi trong tài liệu sự tiến triển giữa
  các threadgroup, và một kỹ sư Apple gọi các giả định của scan một lượt là hành vi không xác
  định (ch11 facts C5). Vì vậy kernel đơn của lab 11.5 không có phiên bản Metal. llama.cpp phát
  hành cùng scan-scan-add ba lần dispatch trên Metal, và backend MPS của PyTorch dùng ba lượt
  rút gọn, scan và mang sang (ch11 facts D1, D4).
- **Metal**: một buffer có thể được gắn tại một độ lệch byte (`setBuffer:offset:atIndex:`), và
  đó là cách một buffer scratch chứa mọi mức. Với một mảng `float`, độ lệch phải là bội số của
  4 (ch06 facts C3). Không có buffer null để nói "không có tổng". Vì vậy ở chỗ CUDA có thể
  truyền `nullptr`, Metal dùng một kernel riêng.
- **Thư viện**: `DeviceScan` của CUB làm một lượt với decoupled look-back, và scan của rocPRIM
  cũng vậy; MPSGraph có `cumulativeSum` (ch11 facts A8, B2, C6).

## Trước khi đo hiệu năng, hãy dự đoán

Hãy ghi những điều này ra trước. Bài học hỏi điều đầu tiên dưới id trong ngoặc.

1. reduce-scan-scan sẽ chạy nhanh hơn scan-scan-add bao nhiêu lần ở n = 2^26 trên thiết bị của
   bạn? Phép đếm byte cho 16/12, khoảng 1.33 (`ch11/11.9/p-rss`, được hỏi ở mục 11.9).
2. Mỗi thiết kế sẽ đạt bao nhiêu phần tốc độ của kernel chép? Nếu cả hai di chuyển số byte của
   mình nhanh như phép sao chép di chuyển 8 byte của nó, scan-scan-add sẽ mất gấp đôi thời gian của
   phép sao chép và reduce-scan-scan gấp rưỡi.

## Đọc kết quả

Benchmark scan n = 2^26 số float. Nó dùng ít hơn nếu thiết bị của bạn báo ít hơn bốn lần 512 MB
đầu vào và đầu ra. Nó chạy với khối 256 và 1024 (S = 1024 và 4096). Với mỗi kích thước khối, nó
đo phép sao chép, rồi scan-scan-add, rồi reduce-scan-scan, và kiểm tra từng kết quả trước khi báo.

- `time ms` là lần tốt nhất trong năm lần chạy sau một lần khởi động không đo. Nó bao gồm mọi
  kernel của mọi mức. Một phần tư giây các phép sao chép không đo chạy trước, để kernel được đo đầu
  tiên không phải trả thêm chi phí đánh thức thiết bị.
- `Gelem/s` là n chia cho thời gian.
- `book B/elem` là phép đếm byte bộ nhớ toàn cục mỗi phần tử của sách (pp. 272-275): 8 cho phép
  chép, 16 + 8/S và 12 + 12/S cho các thiết kế. `GB/s book` là con số đó nhân n, chia cho thời
  gian. Đó là một mô hình lưu lượng, không phải phép đo lưu lượng, nên theo cách đếm này một
  thiết kế có thể cho thấy nhiều GB/s hơn phép sao chép.
- `of copy` là thời gian của phép sao chép tính theo tỉ lệ so với thời gian của thiết kế.
- Dòng cuối cho mỗi kích thước khối cho tỉ lệ đo được của hai thiết kế bên cạnh tỉ lệ phép đếm
  byte của sách.

Cùng các hàng đó được đưa vào mục 11.9 và trang labs, bên cạnh các hàng của lab 11.5. Các con số
là phép đo của máy này ở kích thước này. Chúng không phải là thuộc tính của các thiết kế. Hãy
chạy benchmark nhiều hơn một lần trước khi so với dự đoán của bạn. Nếu các lần chạy không khớp
nhau, độ phân tán đó là một phần của những gì bạn đã đo.

## Đi xa hơn

- p. 276 cảnh báo rằng một số phép rút gọn của chương 10 giả định toán tử có tính giao hoán.
  Dòng nào trong `ReduceSegmentsKernel` của bạn giả định điều đó? Hãy xem những phần tử nào một
  luồng kết hợp. Bạn sẽ viết lại nó thế nào cho một toán tử có tính kết hợp nhưng không giao
  hoán?
- Tăng `COARSE_FACTOR` lên 8. S gấp đôi, nên có ít mức hơn, và các số hạng 8/S và 12/S giảm một
  nửa. Thời gian có thay đổi không? Hãy để ý giới hạn bộ nhớ chia sẻ trong chú thích ở đầu
  `kernel.cu`.
- Làm scan thành loại trừ (mục 11.1), như cấp phát bộ nhớ và top-p sampling cần. Những kernel
  nào thay đổi trong mỗi thiết kế?
