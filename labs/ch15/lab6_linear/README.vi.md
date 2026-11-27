> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 15.6: lớp tuyến tính nhanh (mốc dự án M3)

Sách: các mục 15.3 đến 15.8 (pp. 352-368) và Bài tập 1 đến 3 (p. 370).

Viết `launchLinear(X, W, Y, batch, in, out)`: `Y[batch x out] = X[batch x in] * W^T`, với W
lưu dạng (out, in) theo thứ tự hàng (row-major), như `nn.Linear` và ggml lưu một trọng số. Đó
đúng là chữ ký của lab 6.4, nên kernel này lắp được vào chỗ của M2. Bên trong là kernel lab
15.5 của bạn, được điều chỉnh theo cách bố trí của dự án. Theo tên gọi của sách, A = X
(M = batch, K = in), B = W^T (N = out), C = Y.

## Phần việc của bạn

Bắt đầu từ kernel lab 15.5 của bạn. `clear`, `loadTile` (giờ dành cho ô X), `mm` và
`writeTile` giữ nguyên. Có ba điều mới:

1. **Ô W.** B = W^T được lưu theo thứ tự cột (column-major): các hàng của W là các cột của B.
   Viết `loadTileT`, hàm đọc ô W theo cách W được lưu (128 hàng của W, mỗi hàng 8 số float dọc
   theo k, một float4 mỗi luồng khi `in % 4 == 0`, các số vô hướng có điền 0 trong trường hợp
   khác) và ghi nó dạng chuyển vị vào `B_s[k][n]`. Khi đó `mm` đọc một dải B như một hàng liên
   tục của `B_s`, như trong sách và trong lab 15.5.
2. **Bước hàng của `B_s`, `LDB_S`.** Mã khởi đầu (starter) đặt nó bằng `bN`, giá trị này đúng
   nhưng gây xung đột. Hãy chọn một giá trị tốt hơn (xem dưới).
3. **Tên gọi.** Kernel là `fastLinearKernel<WARP>`: tham số template độ rộng warp mà lab 15.5
   gọi là W ở đây là WARP, vì W là trọng số. Trên Metal, kernel là `linear_m3`.

Xử lý `in == 0` (`numTiles - 1` của Fig. 15.14 không được tràn vòng). `launchLinearWidth`,
`launchLinear` và `launch.mm` đã hoàn chỉnh.

## Vượt ra ngoài sách (của khóa học này)

Sách nhân hai ma trận thứ tự hàng và không cho mã cho lần nạp vector, lần ghi vector hay cách
bố trí của mục 15.5. Mọi thứ điều chỉnh lab 15.5 theo cách bố trí của dự án là của khóa học
này:

- **Lưu W dạng chuyển vị.** Nếu không, `mm` sẽ đọc một dải B dọc xuống một cột của ô chia sẻ,
  với một bước nhảy (stride). Xoay ô trong bộ nhớ chia sẻ (shared memory) giữ cho lần đọc dải
  là một hàng liên tục; đây là phép xoay góc (corner turning) của lab 6.4 ở quy mô 128 x 8.
- **`LDB_S = bN + 4 = 132`.** Lần ghi chuyển vị ghi dọc xuống các cột của `B_s`. Với 32 bank
  4 byte của NVIDIA (`notes/today/ch15-facts.md` A2), lane t của warp w trong ánh xạ float4
  ghi word `(4(t % 2) + j) * pitch + 16w + t / 2`. Với pitch 132, đó là
  `16(t % 2) + t / 2 + 4j + 16w` mod 32: 32 bank khác nhau. Ánh xạ vô hướng của Fig. 15.5, mà
  kernel này không dùng nhưng người đọc có thể dùng (lane t ghi
  n = 32s + 4w + t / 8, k = t % 8), cho `4(t % 8) + t / 8 + 4w` mod 32: lại là 32 bank khác
  nhau. Với pitch 128, cùng các lần ghi đó là xung đột 2 chiều và 8 chiều. Trình khám phá của
  mục 15.6, chế độ thứ ba, vẽ cả hai ánh xạ. Các hàng vẫn căn chỉnh 16 byte và dải B vẫn liên
  tục. 132 số float là 128 cộng 16 byte, phần đệm mà MLX cho các ô của nó. Apple không ghi
  nhận bank nào (C2), nên trên Metal đây là một phép đo.
- **Độ rộng warp.** Như trong lab 15.5: 32, hoặc 64 cho một wavefront AMD rộng 64, khi đó
  khối trở thành 2 x 2 ô wavefront 64 x 64, lane 8 x 8, góc phần tư 32 x 32
  (ch15-facts B1). `launchLinear` đọc `warpSize` từ thiết bị; trên Metal, chương trình điều
  khiển (driver) biên dịch `SIMD_WIDTH` từ `threadExecutionWidth` của pipeline (C1).
- **Các đường float4.** Lần nạp cần `in % 4 == 0`, lần ghi cần `out % 4 == 0`, và các con trỏ
  phải chính là các vùng cấp phát: căn chỉnh 16 byte trên CUDA, nơi `cudaMalloc` căn chỉnh ít
  nhất 256 byte (ch06-facts A3); trên HIP, tài liệu tham khảo bộ nhớ của AMD không nêu căn
  chỉnh nào cho `hipMalloc` (ch15-facts B8), nên các đường float4 ở đó là unverified. Trên
  Metal, chúng đi qua các buffer được khai báo `device float4*`, cách được Metal ghi nhận,
  không phải một phép ép kiểu con trỏ (ch06-facts C3).
- **Lớp 4096 -> 4096** trong benchmark là lựa chọn của khóa học này, một kích thước ẩn cỡ 7B.

Những chỗ mã khác trang sách, như trong lab 15.5: một khối một chiều 256 luồng (notes,
"Unclear" 7); mọi chỉ số vào ô A có đệm dùng `bK + 1`, kể cả độ lệch hàng (notes, "Unclear"
4); Fig. 15.14 được cụ thể hóa, với buffer được chọn theo tính chẵn lẻ (notes, "Unclear" 8).

## Tệp

- `work/kernel.cu` (CUDA, HIP, các trình giả lập): các hàm trợ giúp, `fastLinearKernel<WARP>`.
- `work/kernel.metal`: `linear_m3`. `work/launch.mm`: `encodeLinear`, đã hoàn chỉnh. Nó gắn
  Y, X và W lần thứ hai, dưới dạng float4, ở các chỉ số 4, 5 và 6.
- Được cung cấp, không sửa, trong `provided/`: `vendor_linear.cu` (cuBLAS, hipBLAS),
  `vendor_linear_mps.mm` (MPS), `occupancy.cu` (số thanh ghi và số khối mỗi SM, CUDA và HIP)
  và `baselines.cmake`, tệp này cũng sao chép lời giải tham chiếu của lab 6.4 vào thư mục build
  với tên đã đổi (`launchLinearM2`, `encodeLinearM2`) để làm mốc so sánh m2.

## Lệnh

- Kiểm thử: `python3 labs/pmpp.py run 15.6 --backend emu` trước, rồi `--backend race`, rồi
  `python3 labs/pmpp.py run 15.6` trên GPU của bạn.
- Đo: `python3 labs/pmpp.py run 15.6 --bench`.

## Các bài kiểm thử làm gì

- Các hình dạng (batch, in, out): (1,1,1), (1,40,260), (3,3,3), (128,8,128), (128,24,128),
  (130,9,131), (5,1,9), (17,33,9), (256,32,256), (3,0,5); trên GPU có thêm (4,300,1000),
  (257,129,511), (1024,1024,1024), (8,4096,4096). `race` chạy (128,24,128), (130,9,131)
  và (1,40,260).
- `(3,0,5)` là `in == 0`: Y phải ra toàn số 0. Nó chạy dưới một bộ canh giờ 30 giây, nên một
  `numTiles - 1` bị tràn vòng (một `numTiles` không dấu không có điều kiện chặn) thất bại với
  lý do đó, trên `emu` và trên Metal, thay vì chạy hàng giờ.
- Mọi hình dạng ở độ rộng warp 32 và 64, trên mọi backend: độ rộng chỉ thay đổi luồng nào sở
  hữu đầu ra nào, nên cả hai phải cho cùng Y. Thêm một trường hợp chạy `launchLinear` (Metal:
  pipeline cho độ rộng của thiết bị) đúng như dự án gọi nó.
- X và W chứa đúng số float của chúng, nên trên `emu` một lần đọc vượt qua một trong hai là
  một lỗi AddressSanitizer chỉ ra dòng đó. Y bắt đầu là NaN và theo sau là một dải bảo vệ mà
  các số float của nó phải vẫn giữ giá trị canh gác sau đó.
- Kết quả được so với `y = x W^T` tính trên CPU bằng double, với dung sai tỉ lệ theo tổng
  |x w| của mỗi đầu ra. Trên GPU, mỗi trường hợp chạy hai lần, Y được đặt lại NaN ở giữa.
- Trên GPU, lời gọi của nhà cung cấp được cung cấp sẵn cũng được kiểm tra với cùng tham chiếu.
  Một lỗi ở đó là lỗi của khóa học, không phải của bạn.
- Tham chiếu CPU, `cpuLinear`, nằm trong `main.cpp` của lab này, như quy ước của dự án yêu
  cầu; `../shared/linear_test.hpp` khai báo nó và chứa bài toán kiểm thử.

## Trước khi bạn đo hiệu năng, hãy dự đoán

Benchmark đo thời gian ba kernel trên lớp 1024 -> 4096 của lab 5.3 và 6.4 ở batch 1, 4, 16,
64, 256 và 1024, và trên một lớp 4096 -> 4096 ở batch 1, 16 và 256: **m2** (kernel của lab
6.4, T = 16), **m3** (của bạn) và **vendor** (cuBLAS, hipBLAS hoặc
`MPSMatrixMultiplication`). Chúng thay phiên nhau trong mỗi vòng và vòng tốt nhất được tính.

Hãy ghi ra hai điều trước:

1. Ở batch 256 trên lớp 1024 -> 4096, tốc độ M3 của bạn bằng bao nhiêu phần trăm tốc độ của
   thư viện nhà cung cấp.
2. Ở batch 1, thứ hạng của m2, m3 và vendor.

Những điều cần cân nhắc:

- Ở batch 1, cả ba đều phải đọc toàn bộ ma trận trọng số một lần; cường độ tính toán của mục
  15.2, 0.5 m n / (m + n), dưới 0.5 FLOP/B khi m = 1. Và một ô khối 128 hàng tính 128 hàng
  trong đó 127 hàng là phần đệm: lớp 1024 -> 4096 cho bao nhiêu khối ở batch 1, và chúng
  chiếm bao nhiêu phần của GPU?
- Một thư viện có thể chọn một kernel khác cho mỗi hình dạng. Chính llama.cpp chuyển sang các
  kernel ma trận-vector cho đến batch 8 (`MMVQ_MAX_BATCH_SIZE`, `MMVF_MAX_BATCH_SIZE`,
  ch15-facts D1), và chú thích riêng của PyTorch gọi gemv của nó là "bw bound" (D4).
- Các thư viện nhà cung cấp dùng những kỹ thuật mà chương này chỉ nêu tên (mục 15.9). Cả ba
  mặc định tính bằng FP32 (A10, B6, C6), nên khoảng cách là do kỹ thuật, không phải độ chính
  xác.
- Trên Metal, Apple không ghi nhận bank hay số thanh ghi (C2, C3), và khuyên không nên để
  nhiều luồng đọc cùng một địa chỉ threadgroup, điều mà cách bố trí của mục 15.5 cố ý làm
  (C2). GEMM của chính MLX dùng ô nhỏ hơn và 64 đến 128 luồng mỗi threadgroup (D3). Đây là
  những dữ kiện để cân nhắc, không phải lý do mà tài liệu đưa ra.
- Mỗi vòng dùng lại cùng X và W, nên một bộ nhớ đệm (cache) đủ lớn để chứa W có thể phục vụ nó.

Benchmark cũng in một hàng về kernel đã biên dịch của bạn: trên CUDA và HIP là số thanh ghi
mỗi luồng và số khối mỗi SM mà runtime báo (lập luận của mục 15.7, được đo), trên Metal là
`threadExecutionWidth` của pipeline, giới hạn luồng và bộ nhớ threadgroup. Nó in chế độ toán
của thư viện nhà cung cấp. Các con số của nó là phép đo trên máy của bạn, không phải kỳ vọng.

## Ghi chú về các thư viện nhà cung cấp

- cuBLAS và hipBLAS dùng thứ tự cột. Lời gọi là `cublasSgemm(h, CUBLAS_OP_T,
  CUBLAS_OP_N, out, batch, in, &1, W, in, X, in, &0, Y, out)`: Y^T = W X^T đọc theo thứ tự cột
  (phép suy dẫn của khóa học này, `provided/vendor_linear.cu`). hipBLAS nhận cùng các đối số.
  MPS dùng thứ tự hàng và chỉ cần `transposeRight: YES`.
- Chế độ toán của handle không bao giờ bị thay đổi. Trên NVIDIA, `NVIDIA_TF32_OVERRIDE=0` tắt
  TF32 ở mọi nơi; trên compute capability 10.0 và 10.3, giả lập BF16x9 có thể được bật mà
  không cần đổi mã qua `CUBLAS_EMULATE_SINGLE_PRECISION`, và khi đó thư viện chạy một thuật
  toán khác (A10). Benchmark in ra cả hai biến.
- `provided/vendor_linear.cu` và `provided/occupancy.cu` chưa được biên dịch cho CUDA hay HIP
  (unverified). Nếu một trong hai không build được, hãy cấu hình với
  `-DPMPP_CH15_VENDOR_BLAS=OFF` hoặc `-DPMPP_CH15_OCCUPANCY_API=OFF` và lab sẽ chạy mà không
  có các hàng đó. Không có lời giải của lab 6.4 thì không có hàng m2.
