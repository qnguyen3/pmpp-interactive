> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 15.1: kernel gộp việc 128 x 128

Sách: mục 15.3 (pp. 352-357), Figs. 15.3 đến 15.7. Mốc dự án M3, giai đoạn 1 trên 5.

Viết phép nhân ma trận ô lớn của sách, `C[M x N] = A[M x K] * B[K x N]`, thứ tự hàng
(row-major), kích thước bất kỳ. Một khối 256 luồng tính một ô 128 x 128 của C. Mỗi luồng sở
hữu một mảng 8 x 8 của ô đó và giữ nó trong thanh ghi (register), trong khi chiều trong chảy
qua bộ nhớ chia sẻ (shared memory), mỗi lần 8 cột của A và 8 hàng của B. Bốn hàm trợ giúp
(`clear`, `loadTile`, `mm`, `writeTile`), kernel `gemmCoarseKernel`, và hàm khởi chạy của nó
`launchGemmCoarse`.

Những chỗ lab cố ý khác trang sách (phần đầu của lời giải tham chiếu (solution) nói lý do):

- Khối là một chiều, 256 luồng. Lời văn nói 16 x 16, nhưng Fig. 15.3 chỉ đọc
  `threadIdx.x`.
- `A_s` và `B_s` được khai báo một lần, ở đầu kernel.
- Kích thước là `int`, và các đối số là `(A, B, C, M, K, N)`: thứ tự của lab 5.3 (số hàng
  của A, chiều trong, số cột của B), nên các chỗ gọi của mốc dự án M2 dùng được nguyên vẹn.
- Các hàm trợ giúp là `static`. Các giai đoạn sau liên kết lời giải của lab này vào
  benchmark của chúng, và trình liên kết của trình giả lập CPU sẽ âm thầm gộp hai hàm inline
  không static trùng tên.
- `__launch_bounds__(256)` là của khóa học này. Nó cho trình biên dịch biết kích thước khối;
  thiếu nó, HIP biên dịch cho khối lớn nhất mà thiết bị chấp nhận và có thể ép bớt thanh ghi
  (`notes/today/ch15-facts.md` B3).

## Tệp

- Sửa `work/kernel.cu` và `work/kernel.metal` + `work/launch.mm`.
- `../shared/` chứa các bài kiểm thử và benchmark mà lab 15.1 đến 15.5 dùng chung
  (`stage_driver.hpp`), các lời gọi GEMM của nhà cung cấp, và logic build sao chép lời giải
  của lab 5.3 vào benchmark của lab này. Bạn không sửa nó.

## Lệnh

    python3 labs/pmpp.py run 15.1 --backend emu   # start here
    python3 labs/pmpp.py run 15.1                 # every backend you have
    python3 labs/pmpp.py run 15.1 --bench         # real devices only

Trên trình giả lập, một rào chắn (barrier) bị thiếu cho kết quả sai dưới mọi thứ tự luồng, và
AddressSanitizer dừng ở lần đọc ngoài biên đầu tiên của A hoặc B và chỉ ra dòng đó.

## Các bài kiểm thử kiểm tra gì

- Các hình dạng được chọn để mỗi cái làm hỏng một thứ: một phần tử duy nhất, một khối với
  một cặp ô, ba cặp ô, ô không đầy theo mọi hướng với K và N không phải bội của 4, một hàng
  decode trải qua ba khối cột, K = 1, bốn khối đầy, N hẹp, và K = 0 (một tổng rỗng: C phải
  ra toàn số 0). Trên GPU có thêm hình vuông 1024 và ba hình dạng lớn có kích thước lẻ. Trường hợp
  K = 0 có 30 giây; một kernel vẫn còn chạy lúc đó sẽ thất bại kèm lý do khả dĩ, một cận
  vòng lặp ô bị tràn vòng (lab 15.5 là nơi lỗi này gây hại).
- C được điền NaN trước mỗi lần chạy và theo sau là một dải bảo vệ 128 hàng chứa một giá trị
  canh gác. Một phần tử mà kernel của bạn không bao giờ ghi sẽ vẫn là NaN; một lần ghi vượt
  quá hàng cuối của C sẽ làm thay đổi dải bảo vệ. Thông báo lỗi nói là trường hợp nào.
- Kết quả được so với một tham chiếu CPU trong một dung sai tăng theo kích thước của các
  tích được cộng (Metal không hứa số học IEEE nghiêm ngặt).
- Trên Metal, pipeline phải chấp nhận 256 luồng mỗi threadgroup. Metal có thể đặt giới hạn
  thấp hơn mức kernel yêu cầu, nên chương trình điều khiển (driver) kiểm tra trước khi chạy
  bất cứ thứ gì.

## Trước khi bạn đo hiệu năng

Hãy ghi ra hai dự đoán.

1. **Mức tăng tốc so với kernel của lab 5.3 ở 2048 x 2048 x 2048.** Cân nhắc ba điều. Mô
   hình cho 4 FLOP/B với các ô 16 x 16 của lab 5.3 và 32 FLOP/B với các ô 128 x 128: lưu
   lượng toàn cục được yêu cầu ít hơn 8 lần. Nhưng vòng lặp trong của Fig. 15.6 vẫn đọc hai
   giá trị từ bộ nhớ chia sẻ cho mỗi phép nhân cộng, đúng như của lab 5.3. Và trên máy mà
   khóa học được chạy thử, bộ nhớ đệm (cache) đã phục vụ phần lớn lưu lượng của lab 5.3, nên
   phân ô (tiling) được lợi ít hơn nhiều so với mô hình của nó nói.
2. **Ở batch 1 trên lớp 1024 -> 4096, nhanh hơn hay chậm hơn lab 5.3?** Đếm số khối mà mỗi
   kernel khởi chạy (benchmark in ra: 256 cho lab 5.3, 32 ở đây) và so với số SM của thiết
   bị bạn từ `python3 labs/pmpp.py doctor` (CUDA và HIP báo `multiProcessorCount`; Metal
   không báo con số như vậy).

`--bench` đo thời gian lời giải của lab 5.3 (T = 16), kernel của bạn, và GEMM của nhà cung
cấp (cuBLAS, hipBLAS hoặc MPS), lần lượt, lấy lần tốt nhất trong 5 lần chạy mỗi cái, trên các
hình vuông 1024, 2048 và 4096, lớp 1024 -> 4096 ở batch 1, 16, 256 và 1024, và hình dạng K
ngắn 4096 x 128 x 4096. Mỗi kernel trước tiên được kiểm tra với tham chiếu CPU. Nó cũng in
những gì thiết bị nói về kernel của bạn: trên CUDA và HIP là số khối mỗi SM mà API occupancy
báo (ước tính của sách là một, p. 363), trên Metal là giới hạn luồng của pipeline, độ rộng
SIMD group và bộ nhớ threadgroup. Các con số là phép đo trên máy của bạn.

## Vượt ra ngoài sách

Không có trong sách; từ `notes/today/ch15-facts.md`.

- Metal: các hàm trợ giúp là `METAL_FUNC`, macro của Apple cho một hàm luôn inline từ các
  header đi kèm; đặc tả MSL không có thuộc tính inline. Các vòng lặp được trải ra với
  `#pragma clang loop unroll(full)`, mà trình biên dịch của Apple chấp nhận và MLX dùng,
  nhưng Apple không ghi nhận cách điều khiển trải vòng lặp nào (C4).
  `[[max_total_threads_per_threadgroup(256)]]` là bản tương ứng của `__launch_bounds__` (C3).
- Các mốc so sánh của nhà cung cấp: cuBLAS là của sách (p. 368); hipBLAS và Metal
  Performance Shaders là của khóa học này (B6, C6). cuBLAS và hipBLAS dùng thứ tự cột
  (column-major), nên tích thứ tự hàng được tính là `C^T = B^T A^T` (A10). Cả ba tính bằng
  FP32 trừ khi được bảo khác; benchmark in ra chế độ toán. Các lời gọi cuBLAS và hipBLAS
  chưa được build trên máy CUDA hay ROCm: nếu chúng không build được, hãy cấu hình với
  `-DPMPP_CH15_VENDOR=OFF` (và `-DPMPP_CH15_OCCUPANCY_API=OFF` cho truy vấn occupancy),
  như lab 4.4 làm với lời gọi occupancy của nó.
