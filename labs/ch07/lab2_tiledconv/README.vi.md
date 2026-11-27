> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 7.2: tích chập phân ô với vùng biên

Sách: các mục 7.5 và 7.6 (pp. 172-179), Figs. 7.12 và 7.15.

Một phép tích chập (convolution) 2D trên một ảnh float `width x height`, thứ tự hàng (row-major),
với một bộ lọc `(2r+1) x (2r+1)` áp dụng không lật, ô ma tính là 0. Hai kernel phân ô (tiling),
cả hai đều đọc bộ lọc từ bộ nhớ hằng (constant memory):

1. **Phần A, hãy viết nó: kernel vùng biên (halo)** (Fig. 7.12). Khối (block) chính là ô đầu vào:
   `IN_TILE_DIM x IN_TILE_DIM` luồng (thread), mỗi luồng nạp một phần tử, ô ma được ghi là
   số 0, rồi một rào chắn (barrier). Trong pha tính toán, vòng ngoài `FILTER_RADIUS` của
   các luồng ngồi không, và mỗi luồng còn lại tạo ra một phần tử của một
   ô đầu ra `OUT_TILE_DIM x OUT_TILE_DIM`, với `OUT_TILE_DIM = IN_TILE_DIM - 2r`.
   Hãy viết `convTiledHaloKernel` và `launchHalo` (khối và lưới (grid) của nó).
2. **Phần B, hãy sửa nó: kernel vùng biên qua bộ nhớ đệm** (Fig. 7.15). Bộ nhớ chia sẻ (shared memory)
   chỉ giữ chính ô đó, cùng kích thước với ô đầu ra; một điểm lọc rơi ra ngoài ô được
   đọc từ bộ nhớ toàn cục (global memory), nơi sách kỳ vọng các khối lân cận đã để nó lại
   trong bộ nhớ đệm (cache) L2. `convCachedKernel` biên dịch được và có **hai lỗi**. Một trong số đó
   được in sẵn trong sách.

Bán kính là một tham số template, `template <int FILTER_RADIUS>`, nên các vòng lặp bộ lọc
có số lần lặp cố định. Một switch được cung cấp chọn kernel đã biên dịch cho bán kính mà
chương trình điều khiển (driver) yêu cầu (1 đến 7), giống cách `ssm_conv` của llama.cpp chọn độ rộng
bộ lọc (`notes/today/ch07-facts.md` D3). `IN_TILE_DIM` và `TILE_DIM` là `TILE_WIDTH`
của bản dựng.

- Sửa `work/kernel.cu` (phần A: `convTiledHaloKernel`, `launchHalo`; phần B:
  `convCachedKernel`), `work/kernel.metal` (`conv_tiled_halo`, `conv_tiled_cached`, cùng
  hai lỗi đó) và `work/launch.mm` (`encodeConvTiledHalo`). Các phần bắt đầu ở
  các dòng `// @part halo` và `// @part cached`. Đừng sửa `main.cpp` hay
  `../shared/`, hay các hàm được cung cấp sau `// @part provided`.
- `TILE_WIDTH` được định nghĩa bởi bản dựng: mặc định 32, và 8 cùng 16 cũng được dựng cho
  benchmark. Metal biên dịch 8, 16 và 32 lúc chạy, truyền `TILE_WIDTH` và
  `FILTER_RADIUS` dưới dạng macro (một function constant không thể đặt kích thước mảng threadgroup, C3); một
  ô chỉ được kiểm thử khi `maxTotalThreadsPerThreadgroup` của pipeline cho phép
  `tile x tile` luồng, nếu không chương trình điều khiển nói nó đã bỏ qua gì và vì sao. Trên HIP,
  một khối 32 x 32 có thể bị từ chối (B6); khi đó phép kiểm thử sẽ nói vậy. Đừng viết cứng 32.
- Kiểm thử: `python3 labs/pmpp.py run 7.2`. Bắt đầu với `--backend emu`.
- Đo: `python3 labs/pmpp.py run 7.2 --bench` (chỉ trên thiết bị thật).

## Các phép kiểm thử kiểm tra gì

Cả hai kernel được so với một bản tham chiếu CPU (ô ma bằng 0, không lật, tích lũy bằng double),
với dung sai co giãn theo tổng của `|f * x|` cho mỗi đầu ra, vì Metal không
mặc định dùng số học IEEE nghiêm ngặt. `P` được điền NaN trước mỗi lần khởi chạy, nên một
đầu ra không ai ghi sẽ thất bại, và trên thiết bị thật, một vùng canh gác sau `P` phải trở về
nguyên vẹn.

- Kernel `const` được cung cấp mà benchmark dùng để so sánh, trên 33x17, bán kính 1 đến 7
  (1 đến 3 trên trình giả lập). Các phép này vượt qua ngay từ đầu.
- Ví dụ 7 x 7 của sách với bộ lọc 5 x 5 (Fig. 7.4, r = 2): toàn bộ đầu ra,
  và `P[2][2] = 321` và `P[1][0] = 112` (pp. 163-164).
- Dữ liệu ngẫu nhiên với bộ lọc ngẫu nhiên (nên không đối xứng), cho mọi bán kính từ 1 đến
  `min(7, (tile - 2) / 2)`, điều này giữ ô đầu ra của kernel vùng biên ở mức 2 trở lên:
  1x1, 7x7, 30x30, 33x17, 100x61, 257x129 và 1000x999 (rộng x cao). Trên
  trình giả lập (emulator), bán kính 1 đến 3 ở 1x1, 7x7, 13x9, 33x17 và 40x31. Phần lớn các chiều rộng và
  chiều cao này không phải bội của ô, nên các ô chìa ra ngoài mép phải và mép dưới.

## Phần A: những điều cần làm đúng

- `threadIdx` và `blockIdx` là số không dấu, các thuộc tính vị trí của Metal cũng vậy.
  `threadIdx.x - FILTER_RADIUS` là một biểu thức không dấu: với luồng 0, nó là một số
  khổng lồ, không phải -2. Hãy chép các chỉ số sang `int` trước khi trừ bất cứ thứ gì.
- **Mọi luồng đều tới rào chắn.** Các luồng ở vòng ngoài không tính gì, nhưng chúng nạp
  các phần tử vùng biên mà các luồng khác đọc. Đặt phép chặn tính toán sau rào chắn, và đừng bao giờ
  `return` trước nó. NVIDIA và Apple nói một rào chắn chỉ chờ các luồng chưa
  thoát, và trình giả lập mô hình hóa điều đó, nên một lệnh trả về sớm có thể không làm treo gì cả;
  tác hại của nó là một phần tử vùng biên không ai nạp (`notes/today/ch04-facts.md` A6
  và C3).
- **Hãy ghi các số 0 ma.** Vòng lặp tính toán không có kiểm tra biên: đó chính là mục đích của
  việc hiện thực hóa các ô ma. Bộ nhớ chia sẻ mà bạn không ghi sẽ giữ bất cứ thứ gì đã có ở đó.
  Trình giả lập điền nó bằng các byte `0xCD` cho mọi khối, nên nó luôn thất bại; GPU
  không hứa hẹn gì cả, và bộ nhớ chia sẻ chưa khởi tạo của nó có thể tình cờ đọc ra
  số 0, nên trên thiết bị, lỗi này có thể vượt qua. Hãy chạy trên `emu`.
- Không có rào chắn, trình giả lập thất bại ở cả ba thứ tự luồng: theo thứ tự,
  luồng `(r, r)` chạy hết vòng lặp tính toán của nó trước khi các luồng sau nó nạp
  bất cứ thứ gì.

## Phần B: một bài sửa lỗi

Phương pháp, như ở lab 5.4:

1. Chạy `python3 labs/pmpp.py run 7.2 --backend emu` trước. AddressSanitizer dừng lần chạy ở
   lần truy cập ngoài biên đầu tiên và chỉ ra dòng đó, và các phép kiểm thử sau điểm dừng đó
   không chạy. Các phép kiểm thử của phần A chạy trước; chúng thất bại cho đến khi bạn viết xong phần A,
   nhưng điều đó không dừng lần chạy.
2. Sửa một chỗ, chạy lại. Những chỗ thất bại sẽ thay đổi dần theo từng bước.
3. Sau đó chạy trên GPU của bạn, và ghi lại GPU đã cho bạn thấy lỗi nào và bằng cách nào.

Mỗi lỗi tự nó gây thất bại. Một lỗi cho đầu ra sai gần như ở khắp nơi. Lỗi kia cho
đầu ra sai chỉ gần mép phải của ảnh, ở các ô chìa ra ngoài mép đó (nên không
xảy ra ở ô 8 trên ảnh rộng 1000, vốn có đúng 125 ô theo chiều ngang), và ở hàng cuối nó
đọc vượt quá cuối `N`.

**Một điều thứ ba cần để ý, mà không phép kiểm thử nào bắt được.** Phép kiểm tra trong ô ở mã khởi đầu
(starter) là của sách: `threadIdx.x - FILTER_RADIUS + fCol >= 0 && ... < TILE_DIM`. Vế trái
là số không dấu, nên `>= 0` luôn đúng. Kernel vẫn đúng, một cách tình cờ: một điểm lọc
bên trái ô bị quấn thành một giá trị trên bốn tỷ, và nửa `< TILE_DIM` loại nó
(`notes/today/ch07-facts.md` A8, được kiểm tra bằng cách biên dịch). Dù vậy vẫn hãy sửa nó, bằng cách kiểm tra
các bản sao `int` của chỉ số, vì lần sửa tiếp theo bỏ cận trên, hoặc kiểm tra
một cận trên khác, sẽ làm hỏng nó mà không báo gì. Clang báo phép so sánh luôn đúng này
dưới `-Wtautological-unsigned-zero-compare`; bản dựng mặc định thì không.

## Lab này khác trang sách ở đâu

- **Fig. 7.15, dòng 21.** Lần đọc bộ nhớ chia sẻ thiếu `- FILTER_RADIUS` mà phép
  kiểm tra ngay trước nó có dùng (`notes/ch07.md`, Unclear in the PDF).
- **Các phép kiểm tra `>= 0` trên giá trị không dấu** trong Fig. 7.15, ở trên.
- **`F_c` và `F`.** Figs. 7.12 và 7.15 khai báo `F_c` và đọc `F`. Ở đây dùng `F_c`
  xuyên suốt, phẳng (15 x 15 float cho bán kính lớn nhất), đánh chỉ số
  `fRow * (2*FILTER_RADIUS+1) + fCol`. `N_s` có kiểu phần tử, các literal là `0.0f`,
  và `N` là `const`.
- **Bán kính** là một tham số template được chọn bằng switch, không phải một `#define`.
- **Mã khởi chạy.** Sách không trình bày mã host cho các kernel này; `launchHalo` và
  `launchCached` được cung cấp là của khóa học này. Khối của Fig. 7.12 là hình vuông `IN_TILE_DIM` và
  lưới có một khối cho mỗi ô đầu ra, đó là điều mà cách đánh chỉ số của sách ngụ ý.
- **Symbol hằng nằm trong `kernel.cu`**, cạnh các kernel và lời gọi
  `cudaMemcpyToSymbol`: một khai báo `extern __constant__` ở tệp khác cần
  relocatable device code trên CUDA và HIP (A1, B3).

## Thí nghiệm

Hãy làm điều này khi mọi phép kiểm thử đã vượt qua. Trước khi chạy `--bench`, hãy viết ra hai dự đoán:

1. Công thức của mục 7.5 (p. 176) với ô đầu vào 32 x 32 và bộ lọc 5 x 5
   (`t = 28`, `m = 5`) cho 5.42 FLOP/B, so với 0.5 FLOP/B của một kernel bộ nhớ hằng
   đọc mọi đầu vào từ bộ nhớ toàn cục (p. 171). Vậy kernel phân ô yêu cầu
   **ít lưu lượng toàn cục hơn khoảng 5.42 / 0.5 = 10.8 lần**. Trên thiết bị của bạn, nó sẽ chạy nhanh hơn
   kernel bộ nhớ hằng bao nhiêu lần?
2. Ở ô 32 và r = 2, kernel qua bộ nhớ đệm có nhanh hơn kernel vùng biên không? Hãy viết
   ra tỉ lệ bạn kỳ vọng: GFLOP/s của `tiled_cached` chia cho GFLOP/s của `tiled_halo`.

Con số 10.8 nói về số byte được yêu cầu. Bộ nhớ đệm vốn đã phục vụ nhiều lần đọc lân cận của
kernel bộ nhớ hằng (p. 168), và một kernel yêu cầu ít lưu lượng bộ nhớ thì
sẽ bị giới hạn bởi thứ khác, nên mức tăng tốc bạn đo được có thể khác xa 10.8 theo
cả hai chiều. Kernel của mục 7.6 đặt cược rằng các lần đọc vùng biên trúng bộ nhớ đệm; liệu
chúng có trúng trên GPU của bạn hay không chính là phép đo (`notes/today/ch07-facts.md` C4 cho Apple, B5 cho
AMD).

`--bench` làm gì: nó nhân đôi cạnh của một ảnh vuông từ 2048 cho đến khi kernel bộ nhớ hằng
được cung cấp ở r = 2 mất hơn 50 ms, tối đa 8192 x 8192, giữ `N` và `P`
trong phạm vi một phần tám bộ nhớ mà thiết bị báo cáo (và 2 GiB). Với mỗi độ rộng ô (trên
CUDA và HIP, là của bản dựng: `pmpp.py run 7.2 --bench` chạy các bản dựng t8, t16 và mặc định;
trên Metal là 8, 16 và 32) và r trong {1, 2, 3, 5, 7} mà ô đầu ra của kernel vùng biên
ít nhất là 2, nó kiểm tra mỗi kernel trên ảnh 100x61 rồi đo thời gian: lần tốt nhất
trong ít nhất 5 lần chạy, lặp lại cho đến khi tổng các lần chạy đạt khoảng một phần tư giây. Một
kernel không vượt qua phép kiểm tra sẽ được báo cáo, không được đo thời gian.

- `const` là đường cơ sở được cung cấp (`../shared/conv_const.*`): Fig. 7.9 của sách, mỗi
  đầu ra một luồng, bộ lọc trong bộ nhớ hằng, cùng bán kính lúc biên dịch như các
  kernel của bạn, và khối 16 x 16 ở mọi bản dựng.
- `out` là ô đầu ra: `tile - 2r` với `tiled_halo`, `tile` với `tiled_cached`.
- GFLOP/s đếm `2 * m^2` cho mỗi đầu ra, tính cả điểm lọc ma (cách đếm của sách, p. 167).
- `x const` là thời gian của kernel bộ nhớ hằng chia cho thời gian của kernel này.
- `model` là FLOP/B của mục 7.5 cho kernel vùng biên ở `t = out`, và `ratio` là con số đó
  chia cho 0.5: hệ số byte được yêu cầu theo công thức, mà mục 7.5 vẽ đối chiếu với
  `x const` đo được.
- Trên CUDA và HIP, mỗi lần chạy được đo thời gian bao gồm lời gọi `cudaMemcpyToSymbol` được cung cấp
  để chép bộ lọc (tối đa 900 bytes), như nhau cho cả ba kernel.

Mọi con số nó in ra là phép đo trên thiết bị của bạn. Không có gì ở đây nói chúng nên
là bao nhiêu.
