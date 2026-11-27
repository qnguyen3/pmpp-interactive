> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 6.3: gây ra một xung đột bank, rồi đệm để loại bỏ nó

Sách: mục 6.4 (pp. 139-141) và Exercise 5 (p. 155); mục 6.1 (pp. 129-130, Fig. 6.4) cho việc
xoay một ô trong bộ nhớ chia sẻ.

Ví dụ của mục 6.4 là lần ghi rẽ góc (corner turning): các luồng (thread) của một warp ghi xuống
một cột của một ô chia sẻ 32 x 32, mọi luồng đều rơi vào bank 0, và phần cứng phục vụ chúng lần
lượt từng luồng một. Một cột đệm không dùng dịch mỗi hàng đi một bank và xung đột biến mất
(p. 140). Sách chỉ đưa ra hai khai báo và không gì khác. Ở đây bạn viết một kernel mà toàn bộ việc
của nó là phép xoay đó, một phép chuyển vị ma trận, và đo xem phần đệm mang lại gì trên thiết bị
của bạn.

**Phần việc của bạn:** `tiledTransposeKernel(in, out, rows, cols, pitch)` và
`launchTiledTranspose`. `in` là rows x cols và `out` là cols x rows, cả hai theo thứ tự hàng
(row-major), cho mọi kích thước.

- Mỗi ô một khối `TILE_WIDTH x TILE_WIDTH`. Ô chia sẻ đã được khai báo sẵn cho bạn, với chỗ cho
  `TILE_WIDTH + 1` word mỗi hàng, và phần tử (r, c) của ô là `tile[r * pitch + c]`.
- Đọc ô đầu vào theo từng hàng, để các luồng kề nhau đọc các địa chỉ kề nhau (được hợp nhất, mục
  6.1). Một rào chắn (barrier). Ghi ô đầu ra theo từng hàng, nghĩa là mỗi luồng lấy giá trị của nó
  từ ô chia sẻ theo chiều dọc một cột. Lần đọc cột đó là chỗ có xung đột khi pitch là bội của số
  bank.
- Kiểm tra biên cho các ô không đầy, và bảo đảm mọi luồng đều đến được rào chắn.
- `pitch` là một đối số lúc chạy. Chương trình điều khiển (driver) gọi kernel của bạn với
  `pitch = TILE_WIDTH` (không đệm) và `pitch = TILE_WIDTH + 1` (có đệm), nên một kernel phục vụ cả
  hai bố cục.

## Tệp

- Sửa `work/kernel.cu` (CUDA, HIP, trình giả lập): `tiledTransposeKernel` và
  `launchTiledTranspose`. `TILE_WIDTH` được định nghĩa bởi bản dựng (16; 32 cũng được dựng cho
  benchmark), nên đừng viết cứng 16.
- Sửa `work/kernel.metal` (`tiled_transpose`) và `work/launch.mm` (`encodeTiledTranspose`). Metal
  nhận độ rộng ô lúc chạy: 16, và 32 ở chỗ pipeline chấp nhận 1024 luồng mỗi threadgroup.
- Được cung cấp, không sửa: hai công cụ đo, `provided/bank_probe.cu`, `provided/bank_probe.metal`,
  `provided/copy.cu`, `provided/copy.metal` và `provided/bank_probe_launch.mm`.

## Lệnh

- Kiểm thử: `python3 labs/pmpp.py run 6.3 --backend emu`, rồi `python3 labs/pmpp.py run 6.3`.
- Kiểm tra race condition: `python3 labs/pmpp.py run 6.3 --backend race` chạy các luồng giả lập
  đồng thời dưới ThreadSanitizer (chỉ ô 16), công cụ này nêu tên hai dòng của một rào chắn bị
  thiếu hoặc đặt sai chỗ. Một lần chạy sạch là bằng chứng, không phải lời chứng minh.
- Đo: `python3 labs/pmpp.py run 6.3 --bench` (chỉ trên thiết bị thật).

## Các bài kiểm thử làm gì

Phép chuyển vị của bạn chạy trên 1 x 1, 7 x 5, 16 x 16, 17 x 33, 64 x 48, 100 x 1 và 1 x 100, và
trên GPU còn trên 1000 x 1000 và 513 x 2049, mỗi trường hợp ở cả hai pitch. `out` bắt đầu là NaN,
nên một phần tử bạn không bao giờ ghi sẽ bị báo, và phép so sánh là chính xác tuyệt đối: một phép
chuyển vị chỉ di chuyển giá trị và không tính gì. Trên trình giả lập (emulator), một rào chắn bị
thiếu sẽ thất bại ở ít nhất một trong ba thứ tự luồng của nó, và AddressSanitizer dừng ở một lần
truy cập ra ngoài `in` hoặc `out` và nêu tên dòng đó.

Các bài kiểm thử không thể thấy bạn đã dùng pitch nào: đầu ra như nhau theo cả hai cách. Chỉ
benchmark mới thấy được, và chỉ trên một thiết bị mà bố cục có ảnh hưởng.

Hai kiểm tra ngắn cho các công cụ đo chạy trước: kernel sao chép, và kernel thăm dò ở bước nhảy 1,
có giá trị đầu ra đã biết.

## Các công cụ đo

- **Kernel thăm dò bank.** Một khối đúng một warp: `warpSize` lane trên CUDA và HIP,
  `threadExecutionWidth` của chính pipeline thăm dò trên Metal. Lane `l` đọc và ghi word
  `(l * stride) & 1023` của một mảng chia sẻ 1024 số float, mảng và chỉ số của Exercise 5, rất
  nhiều lần, mỗi bước phụ thuộc vào bước trước. Benchmark chạy các bước nhảy 1, 2, 4, 7, 8, 12, 16,
  24, 31, 32 và 33 và báo thời gian cho mỗi lần truy cập toàn warp (một lần đọc hoặc một lần ghi)
  cùng tỉ lệ so với bước nhảy 1. Thứ nó đo là một chuỗi truy cập phụ thuộc nhau của một warp, nên
  độ trễ của mỗi lần truy cập và vòng lặp bao quanh có mặt trong mọi con số. Chú thích trong
  `provided/bank_probe.cu` cho biết nó ngăn trình biên dịch loại bỏ các lần truy cập bằng cách nào.
- **Kernel sao chép.** Đọc và ghi cùng các byte như phép chuyển vị mà không dùng bộ nhớ chia sẻ và
  không xoay. Một phép chuyển vị chạy nhanh xấp xỉ như vậy thì không còn nhiều để thu được từ bố
  cục của nó.

Trình giả lập chạy kernel thăm dò chỉ để kiểm tra đầu ra của nó. Nó không mô hình hóa bank nào và
không mô hình hóa thời gian.

## Trước tiên, tự làm Exercise 5 bằng tay

Với các bước nhảy 32, 31, 24, 16, 12, 8 và 7, hãy tính xem 32 lane của một warp chạm vào những bank
nào và bao nhiêu lane dùng chung mỗi bank, với bố cục của sách là 32 bank mỗi bank 4 byte (p. 139).
Rồi tìm quy tắc cho câu trả lời với mọi bước nhảy mà không cần liệt kê các lane. Các hàng của kernel
thăm dò theo cùng thứ tự, nên bạn có thể đặt câu trả lời của mình bên cạnh chúng.

## Trước khi đo hiệu năng, hãy dự đoán

Viết cả hai ra trước.

1. Kernel thăm dò tốn thời gian mỗi lần truy cập ở bước nhảy 32 lâu hơn bao nhiêu lần so với ở
   bước nhảy 1. Mô hình của NVIDIA nói 32 lượt so với 1, nhưng kernel thăm dò cũng đo cả độ trễ và
   chi phí vòng lặp, nên hãy quyết định bao nhiêu phần của con số 32 đó sẽ hiện ra. AMD và Apple
   hoàn toàn không cần theo mô hình của NVIDIA (bên dưới).
2. Tỉ lệ thời gian chuyển vị ở pitch T so với pitch T + 1, cho T = 16 và cho T = 32. Với T = 16,
   trước hết hãy tính xung đột của lần đọc cột ở pitch 16 và ở pitch 17: một warp 32 luồng phủ hai
   hàng của một khối rộng 16.

## Những gì các nhà sản xuất công bố

Từ `notes/today/ch06-facts.md` và `notes/today/infra-facts.md`, để đặt cạnh những gì bạn đo được:

- NVIDIA: 32 bank, mỗi bank rộng 4 byte. Một yêu cầu có xung đột được tách thành đủ số yêu cầu không
  xung đột cần thiết. Các luồng đọc cùng một word 32 bit thì không xung
  đột. Ô có đệm là ví dụ tính sẵn của chính NVIDIA (A6, A12, infra A10). Nsight Compute đếm xung
  đột trong `l1tex__data_bank_conflicts_pipe_lsu.sum` (A12).
- AMD: 32 bank (CDNA 1 đến 3) hoặc 64 (CDNA 4, RDNA 2 đến 4), mỗi bank 4 byte, phục vụ 16 lane mỗi
  chu kỳ, nên một lần truy cập của một wavefront rộng 64 tốn vài chu kỳ dù không hề có xung đột.
  Hướng dẫn của AMD dùng cùng phần đệm 32 x 33 (B4, infra B10). Với 64 lane ở bước nhảy 32, các
  lane 32 đến 63 quay vòng về đúng các word của lane 0 đến 31 (phép tính của chúng tôi), và AMD nói
  một lần phát (broadcast) từ một địa chỉ đến nhiều luồng được xử lý mà không có xung đột: hãy đọc
  hàng đó cẩn thận.
- Apple: không có cấu trúc bank nào được công bố cho bộ nhớ threadgroup (C2, infra C13). Lời khuyên
  duy nhất của Apple về thứ tự truy cập là để các luồng kề nhau của một quad group chạm vào các
  phần tử kề nhau, và tránh để nhiều luồng đọc cùng một vị trí. Trên Metal, kernel thăm dò là cách
  để bạn tìm ra.

## Đọc đầu ra

- `SIMD width`: độ rộng warp mà kernel thăm dò đã dùng, và nó đến từ đâu.
- Kernel thăm dò bank: mỗi bước nhảy một hàng với thời gian, số nano giây mỗi lần truy cập toàn
  warp, và thời gian so với bước nhảy 1. Số bước tăng dần cho đến khi bước nhảy 1 chạy quá 10 ms.
- Kernel sao chép: cạnh ma trận tăng từ 1024 cho đến khi phép sao chép chạy quá 5 ms, tối đa 8192,
  giữ cả hai ma trận trong phạm vi một phần tư bộ nhớ mà thiết bị của bạn báo cáo.
- Phép chuyển vị: mỗi độ rộng ô và pitch một hàng, với GB/s (8 byte mỗi phần tử: một lần đọc, một
  lần ghi) và `vs copy`, thời gian của phép sao chép chia cho thời gian của phép chuyển vị.
- Trên CUDA và HIP, `TILE_WIDTH` 32 là một bản dựng riêng mà `--bench` cũng chạy. Kernel thăm dò và
  kernel sao chép được đo một lần, bởi bản dựng `TILE_WIDTH` 16.

Các con số là phép đo trên máy của bạn.

## Đi xa hơn

Cách sửa này mở rộng được trong các kernel thật. Phép nhân ma trận lượng tử hóa cho CUDA của
llama.cpp đệm mỗi khối trong bộ nhớ chia sẻ thêm 16 byte "to avoid shared memory bank conflicts" và
kiểm tra phần đệm bằng một `static_assert` (ch06 facts D1). MLX đệm các ô GEMM Metal của nó thêm 16
byte và sắp ô theo thứ tự lưu trữ của toán hạng, tức là rẽ góc cộng với đệm (D3). Vì sao một kernel
có các luồng di chuyển 16 byte mỗi lần lại đệm 16 byte thay vì một word?
