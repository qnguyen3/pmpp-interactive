> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 4.2: một warp phân kỳ tốn bao nhiêu

Sách: mục 4.5 (pp. 79-82), Figs. 4.9 và 4.10. Sách trình bày hai dạng này như những đoạn mã bên
trong các hình, không phải như kernel. Ba kernel này là của khóa học này.

Mục 4.5 nói rằng một warp có các luồng (thread) đi những đường khác nhau sẽ được thực thi theo
nhiều lượt, mỗi đường một lượt, trong khi các luồng khác không hoạt động. Sách không nêu cái giá.
Lab này đo một cái giá như vậy trên thiết bị của bạn, với một kernel có mẫu rẽ nhánh do chương
trình điều khiển (driver) kiểm soát.

**Phần việc của bạn:** ba kernel và các lần khởi chạy của chúng. Mọi mảng có `n` phần tử và luồng
`i` của lưới sở hữu phần tử `i`.

- `branchKernel`: `out[i]` là `pathA(in[i], iters)` ở chỗ `which[i] == 0`, và
  `pathB(in[i], iters)` ở mọi chỗ khác. Mỗi luồng chạy một trong hai. Đây là dạng của Fig. 4.9,
  với dữ liệu thay cho `threadIdx.x < 24`.
- `selectKernel`: cùng `out[i]` đó, được viết sao cho luồng điều khiển của không luồng nào phụ
  thuộc vào `which[i]`.
- `loopKernel`: `out[i]` là `in[i]` sau `trips[i]` bước của `stepA`. Đây là dạng của Fig. 4.10.
  `trips[i]` có thể bằng 0.

## Tệp

- Sửa `work/kernel.cu` (CUDA, HIP, trình giả lập): ba kernel và `launchBranch`, `launchSelect`,
  `launchLoop`.
- Sửa `work/kernel.metal` (`branch_kernel`, `select_kernel`, `loop_kernel`) và `work/launch.mm`
  (`encodeBranch`, `encodeSelect`, `encodeLoop`).
- Được cung cấp bên cạnh, không được sửa: `paths.h` và `paths.metal.h`. Chúng định nghĩa `pathA`,
  `pathB`, `stepA` và `stepB`. Hai đường tốn như nhau, mỗi đường `iters` bước. Hãy đọc chú thích
  trong `paths.h` để biết vì sao chúng là vòng lặp chứ không phải biểu thức một dòng.
- Được cung cấp, chỉ chương trình điều khiển dùng: `provided/simd_probe.metal`.

Hai chi tiết về Metal:

- `kernel.metal` có dòng `#include "paths.metal.h"`, nhưng chương trình điều khiển biên dịch nó
  lúc chạy, và Apple ghi rõ rằng mã nguồn được biên dịch theo cách đó "may only import the Metal
  default library" (`notes/today/ch05-facts.md`, C7). Chương trình điều khiển dán header vào trước
  khi biên dịch. Thông báo của trình biên dịch vẫn nêu tên `kernel.metal` và số dòng của nó.
- Dispatch nguyên các threadgroup (`dispatchThreadgroups`), như chú thích trong `launch.mm` giải
  thích.

## Lệnh

- Kiểm thử: `python3 labs/pmpp.py run 4.2 --backend emu`, rồi `python3 labs/pmpp.py run 4.2`.
- Đo: `python3 labs/pmpp.py run 4.2 --bench` (chỉ trên thiết bị thật).

Các bài kiểm thử chạy tám mẫu A và B (toàn A, toàn B, nửa đầu mảng, các warp xen kẽ, nửa số lane
của mỗi warp, các lane xen kẽ, mỗi warp một lane, một nửa ngẫu nhiên có seed) ở các kích thước
khối bằng 1, 2 và 8 lần độ rộng, với độ dài mảng là bội và không là bội của kích thước khối.
`selectKernel` cũng phải khớp với `branchKernel`. `loopKernel` được kiểm thử với số vòng lặp bằng
0, một dãy tăng đã sắp xếp, chính dãy đó bị xáo trộn, và một hằng số. Một bài kiểm thử đạt cho
biết các giá trị đúng. Nó không thể cho biết `selectKernel` của bạn có thật sự tránh được nhánh
rẽ hay không: chỉ benchmark mới cho thấy điều đó.

## Độ rộng đến từ đâu

Các mẫu được dựng từ độ rộng mà thiết bị của bạn báo cáo, không bao giờ từ một hằng số 32.

- CUDA và HIP: `warpSize` từ các thuộc tính thiết bị. Warp của một luồng là chỉ số tuyến tính của
  nó trong khối chia cho độ rộng. NVIDIA công bố cách cắt đó. AMD công bố thứ tự tuyến tính và dùng
  cùng phép tính lane trong các ví dụ của họ (`notes/today/ch04-facts.md`, A8 và B6).
- Metal: `threadExecutionWidth` của pipeline đã biên dịch. Đặc tả để cách cắt một threadgroup
  thành các SIMD-group là "implementation-defined" (C2), nên chương trình điều khiển không tính
  nó. Nó dispatch kernel thăm dò được cung cấp với cùng hình dạng, đọc lại SIMD-group và lane của
  từng luồng, và dựng "mỗi SIMD-group một lane" cùng các mẫu khác từ đó. Đầu ra kiểm thử cho biết
  kernel thăm dò tìm thấy gì. Kernel thăm dò là một kernel khác với kernel của bạn, nên đây là bằng
  chứng về thiết bị của bạn, không phải một bảo đảm về pipeline của bạn. Chú thích trong
  `provided/simd_probe.metal` có chi tiết.
- Trình giả lập (emulator) không chạy các luồng của một warp cùng nhịp và không có mô hình chi
  phí. Nó chạy mọi mẫu với các độ rộng giả định 4, 32 và 64 để kiểm tra logic của các kernel, và
  nó không chạy được benchmark.

## Trước khi đo hiệu năng, hãy dự đoán

Viết cả hai câu trả lời ra trước.

1. Với một nửa số lane của mỗi warp đi mỗi đường, `branchKernel` sẽ chậm hơn bao nhiêu lần so với
   mẫu đã sắp xếp có cùng lượng việc (nửa đầu mảng là A, nửa sau là B)?
2. Trên chính mẫu nửa-nửa đó, tính cả hai đường rồi chọn sẽ nhanh hơn hay chậm hơn nhánh rẽ?

Rồi thêm hai câu nữa để suy luận trước khi bạn xem. Theo mô hình của sách (p. 80), một warp chạy
bao nhiêu lượt khi một lane đi đường B và mọi lane khác đi A? Và với `loopKernel`: cùng các số
vòng lặp, một lần sắp xếp dọc theo mảng và một lần bị xáo trộn. Cái nào chậm hơn, và vì sao trường
hợp mọi số được thay bằng giá trị trung bình lại thuộc về phép so sánh này?

## Đọc đầu ra

Benchmark tăng `iters` cho đến khi trường hợp toàn A chạy đủ lâu để đo thời gian, rồi in mỗi mẫu
một hàng:

- `branch ms`: `branchKernel` trên mẫu đó.
- `sorted, same mix`: `branchKernel` trên một mẫu có cùng số A và B, mọi A ở phía trước. Lượng
  phép tính như nhau; chỉ cách sắp xếp khác.
- `branch / sorted`: cái giá của cách sắp xếp. Đây là con số mà dự đoán thứ nhất của bạn nói tới.
- `select ms` và `select / branch`: dự đoán thứ hai của bạn.

Tiếp theo là ba hàng cho `loopKernel`: số vòng lặp đã sắp xếp, cùng các số đó bị xáo trộn, và mọi
số bằng giá trị trung bình. Độ rộng và kích thước khối được dùng nằm ở dòng tiêu đề và trong mọi
hàng được ghi lại.

Những gì các nhà sản xuất công bố, để đặt cạnh những gì bạn đo được (`notes/today/ch04-facts.md`):

- NVIDIA: một warp phân kỳ "executes each branch path taken, disabling threads that are not on
  that path". Không có con số chi phí nào được nêu (A9).
- AMD: "the hardware serializes both branches, executing each path with only the relevant
  threads active", và predication (điều `selectKernel` làm) là một trong các cách khắc phục mà
  AMD liệt kê (B7).
- Apple không công bố cả cách các luồng của một SIMD-group đi những đường khác nhau được thực thi
  lẫn cái giá của nó (C4). Trên Metal, bảng của bạn là bằng chứng, và không tài liệu nào nói nó
  nên trông thế nào.

Các con số là phép đo của ba kernel này trên máy của bạn ở một kích thước. Chúng không phải là
tính chất của nhánh rẽ nói chung: một nhánh rẽ mà hai phía chỉ dài vài lệnh có thể được biên dịch
thành một phép chọn, và không nhà sản xuất nào công bố khi nào (A9, C4).
