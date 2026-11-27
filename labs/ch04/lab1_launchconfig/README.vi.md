> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 4.1: hỏi thiết bị, rồi định cỡ lần khởi chạy (mốc dự án M1, nửa sau)

Sách: mục 4.8 (pp. 87-89) cho các truy vấn, mục 4.7 (pp. 85-86) cho phép tính số chỗ, mục 4.4
(p. 75) cho cách một khối được cắt thành các warp. Kernel là GEMM ngây thơ của lab 3.3 với tên
gọi của một lớp tuyến tính; nó được cung cấp sẵn.

Lab 3.3 khởi chạy kernel đó với các khối 16 x 16 vì hình trong sách làm vậy. Ở đây bạn tự truy
vấn thiết bị và suy ra khối và lưới (grid) từ những gì nó báo cáo. Việc này hoàn tất M1.

## CUDA, HIP và trình giả lập (emulator): `work/kernel.cu`

Ba hàm. Kernel `linearKernel`, câu lệnh khởi chạy `launchLinearWith` và các hàm ở cuối tệp được
cung cấp sẵn.

1. `queryDevices(info, maxDevices)`: vòng lặp của p. 88. Hỏi số thiết bị, rồi hỏi thuộc tính của
   từng thiết bị, và chép các trường mà `DeviceInfo` liệt kê (`launch_config.h`, được cung cấp,
   không sửa). Kiểm tra cả hai mã trả về. Trang sách thì không.
2. `chooseLaunch(d, rows, cols, &grid, &block)`: một hàm thuần, không có lời gọi CUDA nào bên
   trong, để các bài kiểm thử có thể đưa cho nó mô tả của những thiết bị bạn không sở hữu. Nó
   phải trả về một cấu hình thỏa mọi quy tắc bên dưới, hoặc `false` nếu không có cấu hình nào.
3. `launchLinear(X, W, Y, batch, in, out)`: truy vấn, chọn, khởi chạy qua `launchLinearWith`.
   Chương trình điều khiển (driver) đọc lại cấu hình mà lời gọi đó đã ghi nhận.

Các quy tắc cho `chooseLaunch`. Các bài kiểm thử kiểm tra từng quy tắc riêng và nêu tên quy tắc
bị vi phạm:

- khối nằm trong `maxThreadsPerBlock`, và trong `maxThreadsDim` trên mỗi trục;
- số luồng của nó là một số nguyên lần warp (`warpSize`, không bao giờ là một hằng số viết cứng);
- các khối cỡ đó có thể lấp đầy các chỗ luồng của một SM. Mục 4.7 nêu hai cách để thiếu: khối nhỏ
  đến mức các chỗ khối hết trước, hoặc kích thước của nó không chia hết số chỗ luồng. Một trường
  chỗ bằng 0 nghĩa là thiết bị không báo cáo nó, và quy tắc cần trường đó được bỏ qua;
- khối không cao hơn `rows`, và `block.z` bằng 1;
- lưới phủ `rows x cols` mà không thừa khối nào, `grid.z` bằng 1, và lưới nằm trong
  `maxGridSize`.

Có nhiều khối thỏa tất cả các điều này. Bạn trả về khối nào là quyết định thiết kế của bạn, và
benchmark cho bạn biết quyết định đó đáng giá bao nhiêu. Các quy tắc này là điểm xuất phát của
khóa học này cho một kernel mỗi đầu ra một luồng, không phải khuyến nghị của một nhà sản xuất.

Những chỗ p. 89 không còn khớp với các header (`notes/today/ch04-facts.md`, A16 và A18):

- `devProp.clockRate` đã bị gỡ khỏi `cudaDeviceProp` trong CUDA 13.0. Một tệp đọc nó sẽ không
  biên dịch được ở đó. HIP vẫn còn thành viên này. Đừng đọc nó.
- Trang sách mô tả `regsPerBlock` là số thanh ghi (register) của một SM. Con số đó là
  `regsPerMultiprocessor`. `regsPerBlock` là giới hạn cho một khối.
- `maxThreadsPerMultiProcessor` và `maxBlocksPerMultiProcessor` không có trên trang sách. Chúng
  là số chỗ luồng và số chỗ khối của mục 4.7.
- Vòng lặp của trang sách so sánh một chỉ số `unsigned int` với số đếm `int` và bỏ qua cả hai giá
  trị trả về. Phiên bản vòng lặp của chính NVIDIA dùng `int`, và hướng dẫn của họ nói phải kiểm
  tra mọi giá trị trả về.

Trên HIP, cùng tệp đó được biên dịch qua `pmpp/vendors/hip.h`: mọi trường mà `DeviceInfo` liệt kê
đều có trong `hipDeviceProp_t` với cùng cách viết (`notes/today/ch05-facts.md`, B2). Một số trường
mang nghĩa khác ở đó (ch04 facts B9): `major` và `minor` là số hiệu kiến trúc của AMD,
`maxBlocksPerMultiProcessor` được tính từ số chỗ luồng và không phải là một giới hạn riêng, và
trang tham chiếu của AMD nói một số trường có thể đọc ra 0. Đó là lý do một giá trị 0 làm bỏ qua
quy tắc của nó thay vì bị đem ra làm số chia.

## Metal: `work/kernel.metal` và `work/launch.mm`

Metal chia câu trả lời làm hai. Thiết bị biết tên của nó và các giới hạn của nó cho kernel đơn
giản nhất có thể. Độ rộng của một SIMD-group và giới hạn luồng thực sự thuộc về một kernel đã biên
dịch, nên chúng được đọc từ pipeline state. Thiết bị không có thuộc tính nào cho số lõi, số thanh
ghi hay số luồng thường trú cùng lúc (facts C1, C5).

1. `kernel.metal`: kernel thăm dò `simd_probe`. Mỗi luồng ghi lại
   `[[simdgroup_index_in_threadgroup]]`, `[[thread_index_in_simdgroup]]` và
   `[[threads_per_simdgroup]]` của nó. Trên CUDA, hai giá trị đầu là phép tính trên chỉ số luồng,
   vì cách cắt một khối thành các warp được công bố. Đặc tả của Metal gọi cách cắt một
   threadgroup thành các SIMD-group là do cài đặt quyết định (implementation-defined) (facts C2),
   nên một kernel phải hỏi. Kernel tuyến tính được cung cấp sẵn.
2. `launch.mm`, `queryMetal(device, pipeline)`: điền `MetalInfo`. Bốn trường lấy từ thiết bị, hai
   trường lấy từ pipeline.
3. `launch.mm`, `encodeLinear(...)`: định cỡ threadgroup theo công thức của Apple (facts C6): độ
   rộng `threadExecutionWidth`, chiều cao `maxTotalThreadsPerThreadgroup` chia cho độ rộng đó, rồi
   không cao hơn batch. Dispatch bằng `dispatchThreads`, hàm nhận lưới tính theo luồng và làm các
   threadgroup ở rìa nhỏ lại. Kernel được cung cấp không có kiểm tra biên, nên
   `dispatchThreadgroups` sẽ ghi ra ngoài Y. `dispatchThreads` cần GPU family Apple4 trở lên;
   chương trình điều khiển kiểm tra và báo cho bạn nếu thiết bị của bạn cũ hơn.
4. `launch.mm`, `encodeProbe(...)`: dispatch kernel thăm dò trên lưới và kích thước threadgroup
   mà chương trình điều khiển đưa cho bạn.

## Lệnh

- Kiểm thử: `python3 labs/pmpp.py run 4.1`   Bắt đầu với `--backend emu`.
- Đo: `python3 labs/pmpp.py run 4.1 --bench` (chỉ trên thiết bị thật)
- Bản tham chiếu: thêm `--solution`.

## Các bài kiểm thử làm gì

- `queryDevices`: mọi trường của thiết bị 0 phải bằng giá trị mà bộ kiểm thử tự đọc được.
- `chooseLaunch`: chín thiết bị tổng hợp nhân với bốn hình dạng đầu ra (1 x 4096, 3 x 40,
  64 x 4096, 1000 x 1000). Các thiết bị là những con số được chọn để mỗi thiết bị thử một quy
  tắc: các giới hạn trong ví dụ của sách, một thiết bị có 1536 chỗ luồng và 16 chỗ khối, độ rộng
  64 dưới giới hạn 256 luồng, `maxThreadsDim` nhỏ hơn `maxThreadsPerBlock`, các trường chỗ bằng 0
  ở độ rộng 32 và 64, và một giới hạn lưới quá nhỏ cho phần lớn các hình dạng. Không thiết bị nào
  mô tả một sản phẩm. Việc có tồn tại cấu hình nào hay không được xác định bằng cách thử mọi khối
  trong các giới hạn. Ở chỗ không có cấu hình nào, câu trả lời phải là `false`.
- `launchLinear`: kết quả so với một bản tham chiếu CPU trong một dung sai, cho các đầu ra
  1 x 1000, 3 x 40, 37 x 53 và 64 x 257, và cấu hình nó đã dùng so với các quy tắc cho thiết bị
  thật. Y được điền sẵn, nên một đầu ra bạn không bao giờ ghi sẽ bị báo. Trên thiết bị thật, một
  giá trị được ghi quá cuối Y cũng bị báo; trên trình giả lập, AddressSanitizer dừng ở lần truy
  cập đầu tiên ra ngoài một buffer và nêu tên dòng đó.
- Metal: `MetalInfo` so với giá trị chính chương trình điều khiển đọc được, cho ba pipeline: hai
  kernel của `kernel.metal`, và `linear_layer` được biên dịch lại với giới hạn luồng đặt thấp hơn
  giới hạn của thiết bị (qua `MTLComputePipelineDescriptor.maxTotalThreadsPerThreadgroup`, facts C5).
  Apple mô tả con số của thiết bị là mức nó hỗ trợ "for a trivial shader", và hai kernel này gần
  như tầm thường, nên pipeline của chúng có thể báo cùng con số với thiết bị. Pipeline thứ ba thì
  không thể, và nó để lộ một giá trị đọc từ sai đối tượng. Threadgroup mà `encodeLinear` trả về,
  cho cả hai pipeline tuyến tính, phải không vượt quá `maxTotalThreadsPerThreadgroup` của
  pipeline đó, là bội số của `threadExecutionWidth` của nó và không cao hơn batch. Các bản ghi của
  kernel thăm dò phải thỏa những gì đặc tả hứa (một lane nhỏ hơn độ rộng, không có hai luồng nào
  của một threadgroup cùng SIMD-group và cùng lane, mọi SIMD-group đều đầy trừ có thể cái cuối) và
  không gì hơn: các bài kiểm thử không giả định cách cắt được thực hiện ra sao.

Các thuộc tính thiết bị của trình giả lập là tổng hợp. Chúng mô tả trình giả lập, không phải một
GPU, và độ rộng warp của nó là cố định.

## Trước khi bạn đo hiệu năng

Benchmark đo thời gian một lớp tuyến tính với 1024 đầu vào và 4096 đầu ra (lớp ví dụ của sách ở mục
20.6, lab 5.3 cũng dùng nó) ở batch 1 và batch 64, một lần với cấu hình bạn suy ra và một lần với
các khối cố định 16 x 16, xen kẽ nhau, và báo số giây và GFLOP/s cho mỗi lần. Hãy viết ra hai dự
đoán trước:

1. Ở batch 1, cấu hình bạn suy ra sẽ nhanh hơn các khối cố định 16 x 16 bao nhiêu lần trên GPU của
   bạn? Trên CUDA và HIP, một khối 16 x 16 ở batch 1 có 15 hàng luồng không có việc gì làm. Trên
   Metal, `dispatchThreads` cắt threadgroup cho vừa lưới, nên yêu cầu đó chạy thành 16 x 1. Hãy
   quyết định xem điều đó có thay đổi câu trả lời của bạn không.
2. Độ rộng bạn dự đoán thiết bị của bạn sẽ báo.

Benchmark cũng in ra những gì bạn đã truy vấn, khối và lưới bạn suy ra, và trên Metal là một bản
tóm tắt của kernel thăm dò: độ rộng mà kernel thấy, số SIMD-group mỗi threadgroup, và liệu các
SIMD-group có phải là những đoạn liên tiếp của chỉ số luồng tuyến tính trong các lần dispatch đó
hay không. Một chữ "yes" ở đó là điều GPU của bạn đã làm, không phải điều Metal hứa.

Các con số là phép đo trên máy của bạn. Nếu cấu hình suy ra chậm hơn 16 x 16 ở một kích thước
batch nào đó, đó là một phát hiện về các quy tắc, và là lý do để giữ lựa chọn này trong một hàm
mà bạn có thể thay đổi.
