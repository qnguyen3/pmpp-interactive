> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 7.1: bộ lọc trong bộ nhớ hằng, ba cách

Sách: các mục 7.2 đến 7.4 (pp. 164-172): Fig. 7.7 (p. 166), Fig. 7.9 (p. 170) và
phương án truyền theo giá trị ở p. 172.

Kernel tích chập (convolution) cơ bản nạp một trọng số bộ lọc và một phần tử đầu vào từ bộ nhớ
toàn cục (global memory) cho mỗi phép nhân cộng, nên mô hình của sách cho nó 0.25 FLOP/B (p. 167).
Mọi luồng (thread) đọc cùng các trọng số theo cùng thứ tự, nên mục 7.4 chuyển bộ lọc vào
bộ nhớ hằng (constant memory), nơi mọi luồng của một warp đọc cùng một địa chỉ được bộ nhớ đệm
hằng phục vụ, và mô hình tăng gấp đôi lên 0.5 FLOP/B (p. 171). Sau đó sách nêu một
phương án không cần lệnh sao chép tường minh: đặt các trọng số vào một struct và truyền nó theo
giá trị cùng lệnh khởi chạy (p. 172). Bạn viết cả ba và đo xem mức tăng gấp đôi có xuất hiện
trên thiết bị của bạn hay không.

**Phần của bạn:** ba kernel và các lệnh khởi chạy của chúng. Cả ba tính cùng một đầu ra: mỗi
đầu ra một luồng, ô ma bằng 0, bộ lọc không bị lật (công thức của sách,
p. 162; `F.conv2d(x, w, padding=r)` của PyTorch tính đúng điều này), bán kính lúc chạy
từ 0 đến 7, và một phép chặn cho các luồng vượt quá mép ảnh.

1. `convBasicKernel` và `launchConvBasic(N, F, P, r, width, height)`: Fig. 7.7, với
   `F` là một con trỏ device. Mã in trong sách đánh chỉ số các con trỏ thường `F` và `P` như
   mảng 2D và không biên dịch được; hãy làm phẳng cả hai. Kernel của sách không có phép chặn
   đầu ra vì nó giả định lưới (grid) khớp đúng với ảnh; kernel của bạn cần có.
2. `F_c`, `convConstKernel` và `launchConvConst(N, F_h, P, r, width, height)`:
   Fig. 7.9. Khai báo `F_c` trong bộ nhớ hằng, phẳng, đủ chỗ cho bộ lọc lớn nhất.
   `F_h` là bộ lọc trong bộ nhớ host; lệnh khởi chạy chép `(2r+1)^2` trọng số của nó vào
   `F_c` bằng `cudaMemcpyToSymbol`, rồi khởi chạy.
3. `convByValueKernel` và `launchConvByValue(N, F_h, P, r, width, height)`: phương án ở
   p. 172. Điền `struct Filter` được cung cấp (904 bytes) trên host và
   truyền nó cho kernel như một đối số `const Filter` thông thường.

Giữ khai báo `__constant__`, kernel và lệnh sao chép trong `kernel.cu`. Sách
nói một kernel ở tệp khác cần một khai báo `extern` (p. 169); trong chế độ whole-program mặc định
của nvcc, đó là lỗi trừ khi mọi thứ được dựng với `-rdc=true`
(HIP: `-fgpu-rdc`). CUDA cũng có qualifier `__grid_constant__` cho các tham số như
struct này; chúng tôi không tìm thấy tài liệu HIP nào về nó, nên tệp này không dùng nó.

Trên Metal không có symbol nào để chép vào: một mảng `constant` ở phạm vi chương trình phải được
khởi tạo trong mã nguồn shader. Ba cách trở thành ba loại đối số kernel:

| | CUDA | Metal |
|---|---|---|
| basic | `const float* F` | `device const float* F`, một buffer |
| const | `__constant__ float F_c[...]` + `cudaMemcpyToSymbol` | `constant float* F`, một buffer trong không gian địa chỉ constant |
| byvalue | `const Filter f`, truyền theo giá trị | `constant Filter& f`, điền bằng `setBytes` (dưới 4 KB) |

## Các tệp

- Sửa `work/kernel.cu` (CUDA, HIP, trình giả lập). `TILE_WIDTH` được định nghĩa bởi bản dựng
  (16); các khối là `TILE_WIDTH x TILE_WIDTH`. Đừng viết cứng 16.
- Sửa `work/kernel.metal` (`conv_basic`, `conv_const`, `conv_by_value`) và
  `work/launch.mm` (`encodeConvBasic`, `encodeConvConst`, `encodeConvByValue`). Chương trình
  điều khiển (driver) truyền 16 làm `tileWidth`.
- Mỗi tệp được chia bằng các dòng `// @part basic`, `// @part const` và
  `// @part byvalue`. Hãy giữ chúng lại: các bài học dùng chúng để hiển thị từng phần một.
- `main.cpp` chứa các phép kiểm thử, bản tham chiếu CPU và benchmark. Bạn không sửa nó.

## Lệnh

- Kiểm thử: `python3 labs/pmpp.py run 7.1 --backend emu` trước, rồi `python3 labs/pmpp.py run 7.1`.
- Đo: `python3 labs/pmpp.py run 7.1 --bench` (chỉ trên thiết bị thật).
- Làm lại từ đầu: `python3 labs/pmpp.py reset 7.1 --yes`.

## Các phép kiểm thử làm gì

Với mỗi kernel trong ba kernel:

- Ảnh 7 x 7 và bộ lọc 5 x 5 của sách (Fig. 7.4). `P[2][2]` phải là 321 và
  `P[1][0]` phải là 112, hai đầu ra mà sách tính ra (pp. 163-164), và mọi
  đầu ra khác phải khớp với bản tham chiếu.
- Ảnh ngẫu nhiên với bộ lọc ngẫu nhiên ở bán kính 0, 1, 2, 3 và 7, trên các ảnh 1 x 1,
  5 x 9, 16 x 16, 33 x 17, 100 x 61 và 257 x 129 (rộng x cao) trên GPU, và
  1 x 1, 5 x 9, 7 x 7 và 33 x 17 trên trình giả lập. Một bộ lọc ngẫu nhiên không đối xứng,
  nên một bộ lọc bị lật, hoặc bị đọc thành `F[fCol][fRow]`, sẽ thất bại ở đây dù nó vượt qua
  ví dụ đối xứng của sách. Bán kính 7 trên ảnh 5 x 9 gần như toàn là điểm lọc ma.

`P` bắt đầu là NaN, nên một đầu ra bạn không bao giờ ghi sẽ bị báo. Bản tham chiếu là một phép
tích chập trực tiếp tích lũy bằng double; dung sai của mỗi đầu ra được co giãn theo tổng của
`|f * x|` trên các điểm lọc của nó, vì Metal không mặc định dùng số học IEEE nghiêm ngặt.

Trên trình giả lập, AddressSanitizer dừng lại ở một lần đọc ngoài `N` (thiếu phép kiểm tra
ô ma) hoặc một lần ghi ngoài `P` (thiếu phép chặn đầu ra) và chỉ ra dòng đó. Trên GPU,
cùng những lỗi đó chỉ hiện ra dưới dạng giá trị sai.

## Trước khi đo hiệu năng, hãy dự đoán

Hãy viết ra những điều này trước.

1. Ở r = 2 (bộ lọc 5 x 5), kernel bộ nhớ hằng sẽ chạy nhanh hơn kernel cơ bản bao nhiêu lần
   trên thiết bị của bạn? Mô hình của sách nói 2x, từ 0.25 lên
   0.5 FLOP/B. Điều đó chỉ đúng nếu cả hai kernel đều bị giới hạn bởi băng thông bộ nhớ và
   không có gì khác được lưu đệm: mô hình tính mọi lần đọc `N` là một chuyến đi tới DRAM,
   dù các luồng lân cận phần lớn đọc cùng các phần tử đầu vào (p. 168).
2. Trên Metal, struct `setBytes` có thắng buffer hằng không? Tài liệu của Apple không nói gì về bộ nhớ đệm
   hằng. Điều họ đã nói, trong một bài nói năm 2016, là không gian địa chỉ `constant`
   dành cho dữ liệu chỉ đọc được dùng lại nhiều, và dữ liệu có các lần đọc
   bị giới hạn tĩnh, như một struct kích thước cố định truyền theo tham chiếu, có thể được nạp
   trước vào thanh ghi (register). Bài nói đó về các GPU iPhone thời ấy; việc Mac của bạn có làm
   điều này hay không thì không có tài liệu nào nêu.

## Benchmark

Với mỗi bán kính 1, 2, 3, 5 và 7, ảnh lớn dần từ 2048 x 2048 cho đến khi kernel cơ bản
mất hơn 50 ms, tối đa 8192 x 8192 và trong phạm vi một phần tư bộ nhớ mà thiết bị của bạn
báo cáo. Rồi cả ba kernel chạy trên ảnh đó với cùng một bộ lọc ngẫu nhiên,
lấy lần tốt nhất trong `--reps` lần chạy (mặc định 5). Mỗi hàng cho thời gian, GFLOP/s, mức tăng
tốc so với kernel cơ bản, và cường độ tính toán theo mô hình của sách cho kernel đó. GFLOP/s đếm
`2 x width x height x m^2` mỗi lần chạy với `m = 2r + 1`, tính cả điểm lọc ma: cách đếm
của sách (p. 167), không phải công việc mà các kernel thực sự làm gần các mép.

Trên Metal, hàng đầu tiên báo `threadExecutionWidth` và
`maxTotalThreadsPerThreadgroup` của pipeline `conv_const`.

Để đặt mỗi kernel cạnh mô hình của nó, hãy lấy băng thông mà lab 5.1 đo được trên cùng
thiết bị. Trần của mô hình là 0.25 lần con số đó (theo GFLOP/s) với kernel cơ bản
và 0.5 lần với hai kernel còn lại. Một kernel vượt xa trần của nó cho bạn biết rằng
giả định "mọi lần đọc đều tới DRAM" của mô hình không mô tả đúng thiết bị của bạn: có thứ gì đó
đang phục vụ sự tái sử dụng. Các con số là phép đo trên máy của bạn.

## Tùy chọn, trên CUDA: bộ lọc đã đi đâu?

`-Xptxas=-v` khiến ptxas liệt kê mức dùng thanh ghi, bộ nhớ chia sẻ và bộ nhớ hằng của mỗi kernel
(CUDA Best Practices Guide). Ví dụ:

```sh
nvcc -c -Xptxas=-v -DTILE_WIDTH=16 labs/ch07/lab1_constconv/work/kernel.cu -o /dev/null
```

Hãy xem 900 bytes của `F_c` xuất hiện ở đâu với `convConstKernel`, và liệu có phần nào
của `Filter` 904 byte xuất hiện với `convByValueKernel` hay không. Tài liệu của NVIDIA nêu rằng các tham số
kernel được truyền qua bộ nhớ hằng, tối đa 32,764 bytes kể từ CUDA 12.1 trên
Volta và các GPU mới hơn. (Lệnh này không được chạy ở đây: máy viết
lab này không có CUDA toolkit.)

## Tài liệu của các nhà cung cấp nói gì

Từ `notes/today/ch07-facts.md`, để đặt cạnh những gì bạn đo được:

- NVIDIA: bộ nhớ hằng được lưu đệm. Nếu mọi luồng của một warp đọc cùng một địa chỉ, nó
  "can be as fast as a register access"; các địa chỉ khác nhau trong một warp được
  phục vụ lần lượt (A3). Chỉ số của bộ lọc không phụ thuộc vào luồng,
  đó là trường hợp tốt nhất. Các tham số kernel đi qua bộ nhớ hằng (A4).
- AMD: `__constant__` giữ nguyên tên và `cudaMemcpyToSymbol` trở thành
  `hipMemcpyToSymbol` (được bản dựng ánh xạ). Bộ nhớ đệm đứng sau nó là bộ nhớ đệm vô hướng,
  chỉ giúp ích khi lần đọc đồng nhất trên toàn wavefront. Chúng tôi không tìm thấy tài liệu
  AMD nào cho biết kích thước bộ nhớ hằng hay giới hạn đối số kernel (B1, B2).
- Apple: tài liệu không nêu bộ nhớ đệm hằng nào, cũng không nêu giới hạn kích thước nào cho buffer hằng;
  `setBytes` nhận dưới 4 KB (C1, C2).

## Lab này khác các trang sách ở đâu

- Fig. 7.7 đánh chỉ số `float* F` và `float* P` bằng hai chỉ số và không cho `inRow` và
  `inCol` kiểu nào. Cả hai được sửa ở đây, và phép chặn đầu ra được thêm vào.
- Fig. 7.9 khai báo `F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1]` cho một bán kính lúc biên dịch.
  Ở đây `F_c` phẳng và có kích thước cho bán kính 7, nên một bản dựng phục vụ mọi bán kính.
- Sách mô tả `cudaMemcpyToSymbol` là báo cho runtime biết dữ liệu sẽ không
  thay đổi trong lúc kernel chạy. Tài liệu của NVIDIA mô tả nó là một lệnh sao chép tới bất kỳ device symbol
  nào, hằng hay không; chính qualifier `__constant__` mới khiến `F_c` chỉ đọc trong kernel (A2).

## Đi xa hơn

Các backend thực tế dùng cách truyền theo giá trị. `conv2d_kernel` CUDA của llama.cpp
nhận các kích thước dưới dạng một đối số `const conv_params`, và các kernel Metal của nó nhận một
tham chiếu struct `constant` được điền bằng `setBytes`. Cả backend CUDA của llama.cpp lẫn
các kernel CUDA gốc của PyTorch đều hoàn toàn không dùng `__constant__`, và llama.cpp giữ các
bảng tra lượng tử hóa trong bộ nhớ toàn cục trên CUDA và trong không gian địa chỉ `constant`
trên Metal (D4). Các bảng đó được đánh chỉ số bằng giá trị đã lượng tử hóa của chính mỗi luồng.
Từ những gì NVIDIA và AMD nói về bộ nhớ đệm hằng của họ, vì sao điều đó lại không hợp với bộ nhớ
hằng, trong khi một bộ lọc tích chập lại hợp?
