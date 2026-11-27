> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab X2.2: một GEMV 4 bit trên các khối Q4_0 của llama.cpp

Không có trong sách. Học phần X2 là của khóa học này, dựa trên phụ lục A, mục A.5 (p. 593) và
phân tích cường độ của chương 15. Các dữ kiện về nhà cung cấp và dự án lấy từ
`notes/today/x2-facts.md`, được kiểm tra ngày 2026-11-20.

Lab X2.1 cắt các trọng số xuống khoảng 1 byte mỗi trọng số, theo một bố cục chọn cho kernel. Lab này
cắt chúng xuống 4.5 bit, theo bố cục bạn sẽ gặp trong một tệp GGUF: Q4_0 của llama.cpp, đúng
như `ggml-common.h` định nghĩa (x2-facts D1).

```cpp
struct BlockQ4_0 { uint16_t d; uint8_t qs[16]; };   // 18 bytes per 32 weights
// d:     the FP16 bits of the block's scale
// qs[j]: code j in the low nibble, code j + 16 in the high nibble (not j and j + 1)
// w = (code - 8) * d,  codes 0..15
```

```
Y[b][j] = sum over k of X[b][k] * w[j][k]      1 <= batch <= 8, in % 32 == 0
```

W là `[out x in/32]` khối, hàng j chứa các trọng số của đầu ra j. X và Y là float32,
thứ tự hàng (row-major), chiều batch đứng trước.

- **Quy tắc là quy tắc `_ref` của llama.cpp** (`quantize_row_q4_0_ref`).
  - Lấy giá trị có độ lớn lớn nhất, kèm dấu của nó; khi hòa thì giá trị đầu tiên thắng.
  - `d = max / -8`. Cực trị có dấu đó rơi đúng vào mã 0, và cực trị đối diện
    bị kẹp về mã 15. Vì vậy d âm khi cực trị dương.
  - Mỗi mã là `min(15, (int8_t)(w / d + 8.5))`, tính từ d dạng float; FP16(d) được
    lưu.
  - Một khối toàn số không lưu d = 0 / -8 = -0, bit FP16 `0x8000`, và các mã bằng 8.

  Đường importance-matrix của llama.cpp chọn d theo cách khác, với cùng bố cục (D1). Lab
  này không mô phỏng nó.
- **Số byte mỗi trọng số**: 18 / 32 = 0.5625, so với 4 cho FP32 và 1.0625 cho Q8_0 (E1).

## Phần của bạn

1. **`work/kernel.cu`: `gemvQ4_0Kernel` và `launchGemvQ4_0`.** Một warp cho mỗi đầu ra j, trong
   các khối gồm 4 warp, như ở lab 22.1 và X2.1.
   - **Cách các lane chia nhau một khối là do bạn chọn**: một lane cho mỗi khối, hoặc nhiều lane (trình
     chạy từng bước của X2.4 cho thấy sự đánh đổi). Nó phải chạy đúng với mọi độ rộng warp, không chỉ 32.
   - **Đọc `qs` theo byte hoặc theo word 16 bit, không bao giờ qua `int` hay `char4`.** Một khối 18 byte
     có căn chỉnh 2, nên cứ cách một khối, `qs` lại bắt đầu ở một địa chỉ chia 4
     dư 2 (A5). llama.cpp đọc nó bằng hai lần nạp 16 bit (`get_int_b2`, A5).
   - **Giải nén bằng mặt nạ và phép dịch bit.** Nibble thấp của byte t là trọng số t; nibble
     cao là trọng số t + 16.
   - **Việc tách số -8 ra được khuyến nghị, không bắt buộc.** Với một khối,
     sum of (q - 8) d x = d (sum of q x - 8 sum of x). Khi đó vòng lặp trong chỉ chạm đến q
     và x (E4). Mọi kernel thật trong x2-facts đều làm vậy: llama.cpp qua trường `s` của Q8_1
     trên CUDA (D3) và qua `sumy` trên Metal (D4), MLX qua `sum` (D5).
2. **`work/kernel.metal`: `gemv_q4_0`**, với khối được khai báo như llama.cpp khai báo trên
   Metal, `struct block_q4_0 { half d; uchar qs[16]; };` (C5). **`work/launch.mm`**: phần
   dispatch trong `encodeGemvQ4_0`.
3. Benchmark, sau khi ghi lại dự đoán của bạn (bên dưới).

## Các tệp

- `work/kernel.cu` (CUDA, HIP và các trình giả lập). Của bạn: `gemvQ4_0Kernel`,
  `launchGemvQ4_0`. Được cung cấp: `warpSum` (static, vì hai lời giải khác được liên kết
  bên cạnh), `deviceWarpSize`, và từ `labs/x2/shared/qformat.h` là `BlockQ4_0` và
  `halfBitsToFloat`.
- `work/kernel.metal`: `gemv_q4_0` là của bạn. `work/launch.mm`: phần dispatch trong
  `encodeGemvQ4_0` là của bạn.
- `provided/unfused.cu`, `provided/unfused.metal`, `provided/unfused_launch.mm`
  (được cung cấp, đúng, chỉ dùng cho benchmark). `launchGemvQ4_0Unfused` và
  `encodeGemvQ4_0Unfused` giải lượng tử hóa W vào một ma trận nháp float32 do bên gọi sở hữu,
  rồi chạy phép gather của lab 22.1 trên đó.
- `labs/x2/shared/qformat.h` và `qfixture.h`: chỉ đọc (README của lab X2.1 liệt kê chúng).
- `main.cpp`: chương trình điều khiển (driver), bộ lượng tử hóa CPU và bản tham chiếu có giải lượng tử hóa. Bạn không cần
  sửa nó.
- `CMakeLists.txt` liên kết lời giải của lab 22.1 (gather FP32), lời giải của lab X2.1 (Q8_0, nếu
  có) và đường không gộp được cung cấp.

Các chữ ký hàm là cố định. Lab X2.4 liên kết chúng:

```cpp
void launchGemvQ4_0(const float* X, const BlockQ4_0* W, float* Y, int batch, int in, int out);
// Metal: kernel gemv_q4_0, struct block_q4_0 { half d; uchar qs[16]; };
//        buffers X 0 (float4), W 1, Y 2, batch 3, in 4, out 5.
void encodeGemvQ4_0(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                    id<MTLBuffer> X, id<MTLBuffer> W, id<MTLBuffer> Y,
                    uint32_t batch, uint32_t in, uint32_t out);
// Provided:
void launchGemvQ4_0Unfused(const float* X, const BlockQ4_0* W, float* Wf, float* Y,
                           int batch, int in, int out);      // Wf: out x in floats of scratch
void encodeGemvQ4_0Unfused(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState> dequant,
                           id<MTLComputePipelineState> gather, id<MTLBuffer> X, id<MTLBuffer> W,
                           id<MTLBuffer> Wf, id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out);
```

## Các lệnh

- Kiểm thử: `python3 labs/pmpp.py run X2.2 --backend emu` trước, rồi
  `python3 labs/pmpp.py run X2.2` cho GPU của bạn.
- Đo: `python3 labs/pmpp.py run X2.2 --bench` (chỉ trên thiết bị thật).
- Lời giải tham chiếu: thêm `--solution`. Làm lại từ đầu: `python3 labs/pmpp.py reset X2.2 --yes`.

## Các test làm gì

- **Bộ lượng tử hóa CPU được kiểm tra trước**, trên hai khối suy ra bằng tay (`qfixture.h`):
  - một khối có cực trị âm, d = +1/8 (`0x3000`);
  - một khối có cực trị dương, d = -1/4 (`0xB400`);
  - một khối toàn số không (`0x8000` và các byte `0x88`).

  Test kiểm tra các bit hệ số tỉ lệ, mọi byte đã đóng gói và các giá trị đã giải lượng tử hóa. Test
  trình mô phỏng của các bài học được dự định dùng cùng các khối đó.
- **Kích thước (batch, in, out):** (1, 32, 1), (1, 64, 5), (2, 96, 7), (3, 256, 64),
  (8, 512, 130), (5, 4096, 3), và (1, 32 x 33, 3). Kích thước cuối có 33 khối mỗi hàng, nên
  cứ cách một hàng lại bắt đầu ở một khối lẻ, tại một địa chỉ chia 4 dư 2. Trên GPU còn có
  (1, 4096, 4096), (8, 4096, 4096), (4, 14336, 4096), (1, 4096, 14336) và (8, 1024, 4099).
- **Các trọng số:**
  - Mỗi khối có độ lớn là một lũy thừa của hai riêng, từ 2^-12 đến 2^12.
  - Mỗi khối có một +peak và một -peak. Cái nào đứng trước thì luân phiên, nên cả hai dấu của d
    đều xuất hiện và peak còn lại bị kẹp về mã 15.
  - Mỗi hàng có một khối với 16 trọng số đầu dương và 16 trọng số cuối âm, nên
    đọc một byte như hai trọng số liền kề sẽ thất bại.
  - Với out >= 3, hàng out / 2 toàn số không, và với từ 2 khối mỗi hàng trở lên thì
    khối cuối của hàng 0 cũng vậy.
- **Kích thước chính xác, NaN và hai lần khởi chạy, như lab X2.1.** X và W chứa đúng số byte của chúng
  (18 cho mỗi khối). Y bắt đầu là NaN, theo sau là các giá trị canh gác (sentinel), và mỗi trường hợp khởi chạy hai lần.
- **Phép so sánh.** Kết quả được so với `dequantize_row_q4_0` theo sau là một tích vô hướng bằng
  double, trong phạm vi 1e-4 x (1 + sum of |x w|), không bao giờ so từng bit.
- **Mốc cơ sở cũng được kiểm tra.** Đường không gộp được cung cấp được kiểm tra trên ba kích thước, nên
  benchmark chỉ so sánh các kernel đúng.

## Mỗi backend bắt được gì

- **`emu`** bắt được chỉ số sai, một nibble ghép với sai trọng số, một hệ số tỉ lệ từ
  sai khối, thiếu số -8, các lần đọc vượt quá X hoặc W, và một lane bỏ qua phép rút gọn. Không có
  cập nhật chung nào, nên không có bản build `race`.
- **Không trình giả lập nào bắt được một lần nạp 4 byte lệch căn chỉnh.** CPU của host cho phép điều đó, và
  AddressSanitizer không kiểm tra căn chỉnh. Một lần đọc `qs` kiểu `int` qua được trên `emu`, nhưng trên GPU,
  cứ cách một khối nó lại là hành vi không xác định: `qs` của khối g bắt đầu ở byte 18g + 2 của W,
  chia 4 dư 2 khi g chẵn (A5). Kích thước 33 khối đặt các khối đó ở vị trí chẵn
  trong một số hàng và vị trí lẻ trong các hàng khác, nhưng một thiết bị vẫn có thể cho kết quả đúng.
  **Hãy kiểm tra bằng cách đọc**: mọi lần nạp từ một khối đều rộng 1 hoặc 2 byte.
- **Metal** kiểm tra kết quả trên GPU của bạn. Cùng quy tắc căn chỉnh đó cũng áp dụng ở đây (C5).
- **Không có gì ở đây kiểm tra một warp 64 lane.** Các trình giả lập mô phỏng warp 32 lane. Hãy kiểm tra bằng
  cách đọc rằng ánh xạ từ lane sang khối của bạn dùng `warpSize`.

## Các backend khác nhau ở đâu (không có trong sách)

- **Đọc khối.** CUDA và HIP: hai lần nạp 16 bit, như `get_int_b2` của llama.cpp (A5).
  Metal: llama.cpp đọc các word 16 bit qua một phép ép kiểu con trỏ (C5). Đặc tả MSL
  không mô tả phép ép kiểu con trỏ nào (infra-facts C7), nên lời giải tham chiếu đọc các `uchar`.
- **Hệ số tỉ lệ.** CUDA và HIP giữ FP16 dưới dạng bit `uint16_t`, được giải mã bằng `halfBitsToFloat`.
  Header của hai nhà cung cấp khác nhau (bảng trong x2-facts), và các bit thì không cần header nào. (Các trình giả lập
  và stub front-end giờ cũng mô phỏng `__half`; lab này không dựa vào điều đó.)
  Metal đọc `half` trực tiếp.
- **Kiểu 4 bit.**
  - CUDA và HIP không có kiểu số nguyên 4 bit nào trong C++ (A4, bảng).
  - MSL giờ có `int4b_format` và `uint4b_format` với `unpack` (Metal 4.0 với SDK 26.4,
    và 4.1). Đặc tả không nói nibble nào đứng trước, cũng không nói họ GPU nào
    hỗ trợ chúng (C2). Cả llama.cpp lẫn MLX đều không dùng chúng (tìm kiếm trong mã).
  - Mặt nạ và phép dịch bit là cách khả chuyển.
- **Ở đây cũng không có tích vô hướng số nguyên.** Đường CUDA và HIP của llama.cpp biến các activation thành int8
  và đưa các nibble không dấu cùng các byte có dấu vào `dp4a`. Trên HIP nó cần một builtin
  khác nhau cho mỗi target (B1, B2, D3). Metal không có tích vô hướng số nguyên (C1).
- **HIP** chưa được build hay chạy ở đây: unverified.

## Trước khi benchmark, hãy dự đoán

Hãy ghi lại những điều này trước:

1. Mức tăng tốc của kernel Q4_0 gộp của bạn ở batch 1 trên 4096 -> 4096:
   - so với F32: trần theo byte là 4 / 0.5625 = 7.11;
   - so với Q8_0: 1.0625 / 0.5625 = 1.89.

   Đó là các tỉ lệ byte. Chúng không phải lời hứa.
2. Mức tăng tốc của đường không gộp so với F32. Nó đọc 0.5625 byte mỗi trọng số, ghi 4
   và đọc lại 4, so với 4 của F32. Mô hình byte cho khoảng 0.47, tức là chậm hơn cả
   việc không lượng tử hóa (A1 trích PyTorch nói đúng điều này về int8 không gộp). Thiết bị của bạn
   có đồng ý không?

Rồi hãy nghĩ về batch 8: số byte activation như nhau cho mọi định dạng, và công việc giải nén
cho mỗi trọng số được chia sẻ cho nhiều hàng hơn.

## Đọc kết quả đầu ra

- **Các lớp và batch** như lab X2.1: 4096 -> 4096, 4096 -> 14336 và 14336 -> 4096, ở
  batch 1, 4 và 8. Hai kích thước sau là của khóa học này.
- **Các kernel:**
  - `gather`: F32, lời giải của lab 22.1.
  - `q8_0`: lời giải của lab X2.1, khi có.
  - `q4_0_unfused`: được cung cấp.
  - `q4_0`: của bạn.

  Chúng chạy luân phiên: một vòng không tính giờ, rồi năm vòng, và mỗi kernel giữ vòng tốt nhất của nó.
- **Các bản sao của W.** Một lần submit có tính giờ đi qua các bản sao trọng số của mỗi định dạng mà
  tổng cộng chứa ít nhất 256 MB, nên không lần khởi chạy nào tìm thấy trọng số của nó trong bộ nhớ đệm (cache) (như lab 22.1).
  Đường không gộp ghi và đọc một ma trận nháp duy nhất. Ma trận đó không được sao chép, nên
  bộ nhớ đệm có thể giúp nó. Thời gian của nó vẫn tính cho mỗi lần khởi chạy, gồm cả giải lượng tử hóa và gather.
- **`W GB/s`** là số byte mà mỗi định dạng lưu (Q4_0: các khối 18 byte) chia cho thời gian.
  Với đường không gộp, điều đó bỏ qua lưu lượng ma trận nháp mà nó cũng di chuyển. Trang labs
  so sánh `W GB/s` với băng thông đo được của lab 5.1.
- **`vs F32` và `vs Q8_0`** là các tỉ lệ thời gian.
- **Các hàng báo cáo:** `qgemv`: `{kernel, format, batch, in, out, seconds, weight_bytes,
  weight_gb_per_s, speedup_vs_f32}`.

Đây là các phép đo trên thiết bị của bạn ở những kích thước này, không phải thuộc tính của các kernel.
Ở chỗ mức tăng tốc gộp thấp hơn 7.11, hãy hỏi kernel còn trả thêm gì cho mỗi trọng số
mà F32 không phải trả: giải nén, hệ số tỉ lệ, số -8. Hãy hỏi cả nó trả gì cho mỗi hàng của X.

## Đi xa hơn

- **Prefill** (của khóa học này, theo `mul_mm` của llama.cpp, x2-facts D4). Với nhiều token, GEMV là
  kernel sai. Giải lượng tử hóa một ô (tile) của W vào bộ nhớ chia sẻ (shared memory), rồi chạy GEMM phân ô
  của lab 15.6 trên đó. Đó là "giải lượng tử hóa, rồi nhân", làm ở mức từng ô thay vì cho
  cả ma trận. So sánh nó với đường không gộp ở batch 64.
- **Activation đã nhân tỉ lệ trước, cách làm quen thuộc trên Metal** (D4, D5, D6). llama.cpp, MLX và PyTorch MPS
  không bao giờ dịch các nibble. Chúng áp mặt nạ tại chỗ, để mỗi nibble bị nhân với 1, 16, 256 hoặc
  4096, và chia các activation tương ứng cho cùng lũy thừa của hai đó, một phép chia chính xác. Hãy thử
  trong `gemv_q4_0` và đo.
- **Q4_1 hoặc định dạng affine của MLX.** Thêm một độ lệch cho mỗi khối, như `d q + m` của Q4_1 (D1) hoặc
  `s q + bias` của MLX (D5). Phép tách của E4 mang độ lệch đi qua tổng của x theo cùng
  cách đó.
