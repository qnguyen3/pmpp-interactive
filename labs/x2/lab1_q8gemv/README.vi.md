> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab X2.1: một GEMV trọng số int8 cho decode batch 1

Không có trong sách. Học phần X2 là của khóa học này. Nó dựa trên phụ lục A, mục A.5 (p. 593): một
định dạng số nhỏ hơn nạp ít byte hơn cho mỗi phép toán, và điều đó làm tăng cường độ tính toán.
Nó cũng xây trên phân tích cường độ của chương 15 và phép gather của lab 22.1. Các dữ kiện về nhà cung cấp và
dự án lấy từ `notes/today/x2-facts.md`, được kiểm tra ngày 2026-11-20.

Lúc decode, mọi lớp tuyến tính là một phép nhân ma trận với vector, và chi phí của nó là đọc W
một lần (lab 22.1). Lưu W bằng ít byte hơn là đòn bẩy duy nhất còn lại. Lab này lưu nó theo
Q8_0 của llama.cpp: các khối 32 trọng số, mỗi khối gồm một mã int8 cho mỗi trọng số cùng một hệ số tỉ lệ
FP16 (x2-facts D1).

```
Y[b][j] = sum over k of X[b][k] * d[j][k / 32] * q[j][k]      1 <= batch <= 8, in % 32 == 0
```

- **Quy tắc là của llama.cpp** (`quantize_row_q8_0_ref`). Với mỗi khối, `d = amax / 127`, mỗi
  mã là `roundf(w / d)` (nửa làm tròn ra xa số không), và d được lưu dưới dạng FP16. Các mã được tính
  từ d dạng float trước khi nó được làm tròn sang FP16. `labs/x2/shared/qformat.h` chứa quy tắc này,
  và chương trình điều khiển (driver) gọi nó.
- **Bố cục là của khóa học này.** Các mã nằm trong một mảng `Wq [out x in]` (int8). Các hệ số tỉ lệ nằm trong một
  mảng thứ hai `Wd [out x in/32]`, dưới dạng bit FP16 trong `uint16_t`. Đó là cách MLX giữ hệ số tỉ lệ
  tách khỏi các trọng số đã đóng gói của nó (D5). `block_q8_0` của llama.cpp xen một hệ số tỉ lệ 2 byte
  với 32 mã, tổng 34 byte, nên các mã của nó chỉ căn chỉnh theo 2 byte (A5). Khi tách riêng, mọi hàng
  mã đều bắt đầu ở một bội số của 32 byte, và bốn mã có thể được đọc thành một word 32 bit.
- **Số byte mỗi trọng số**: 1 + 2/32 = 1.0625, so với 4 cho FP32 (E1).

## Phần của bạn

1. `work/kernel.cu`: viết `codeOf`, `gemvQ8_0Kernel` và `launchGemvQ8_0`. Cách ánh xạ
   là phép gather của lab 22.1:
   - một warp cho mỗi đầu ra j, trong các khối gồm `WARPS_PER_BLOCK` = 4 warp;
   - các lane cùng nhau đi theo bước nhảy dọc hàng j, mỗi lane nạp 4 mã thành một word 32 bit;
   - một tổng trong thanh ghi cho mỗi hàng batch;
   - 4 tích của mỗi word được nhân với hệ số tỉ lệ của khối chúng một lần;
   - `warpSum` trên toàn warp, và lane 0 ghi `Y[b][j]`.

   Không bao giờ dùng hằng 32 cho độ rộng warp. Số 32 trong lab này là `QK8_0`, kích thước khối,
   và nó có tên.
2. `codeOf` biến byte i của một word trở lại thành một mã có dấu. Hãy làm việc đó bằng các phép toán số nguyên
   không phụ thuộc vào cách trình biên dịch chuyển một giá trị ngoài phạm vi sang `int8_t`.
3. `work/kernel.metal`: viết `gemv_q8_0`, cùng kernel đó với mã là `char4` và hệ số tỉ lệ
   là `half`. Trong `work/launch.mm`, viết phần dispatch trong `encodeGemvQ8_0`.
4. Benchmark, sau khi ghi lại dự đoán của bạn (bên dưới).

## Các tệp

- `work/kernel.cu` (CUDA, HIP và các trình giả lập). Của bạn: `codeOf`, `gemvQ8_0Kernel`,
  `launchGemvQ8_0`. Được cung cấp: `warpSum` với kiểu lane mask riêng của nó, `deviceWarpSize`, và
  `halfBitsToFloat` từ `labs/x2/shared/qformat.h`. `warpSum` là `static` vì lời giải của lab
  22.1 được liên kết bên cạnh và có hàm của riêng nó.
- `work/kernel.metal`: `gemv_q8_0` là của bạn.
- `work/launch.mm`: phần dispatch trong `encodeGemvQ8_0` là của bạn. Việc gắn buffer đã được
  cung cấp.
- `main.cpp`: chương trình điều khiển và bản tham chiếu CPU. Bạn không cần sửa nó.
- `labs/x2/shared/qformat.h` (được cung cấp, dùng chung với lab X2.2): `BlockQ4_0`,
  `floatToHalfBits` (làm tròn về gần nhất, hòa thì về chẵn), `halfBitsToFloat` (các phép toán số nguyên
  và một lũy thừa của hai, để không cần kiểu `__half` nào), `quantizeRowQ8_0Ref`,
  `quantizeRowQ4_0Ref`, `dequantizeRowQ8_0`, `dequantizeRowQ4_0`, tất cả đều là vòng lặp đơn giản.
- `labs/x2/shared/qfixture.h` (được cung cấp): các khối dữ liệu cố định suy ra bằng tay mà chương trình điều khiển
  dùng để kiểm tra bộ lượng tử hóa trước tiên.
- `CMakeLists.txt` liên kết lời giải tham chiếu (solution) của lab 22.1 (`launchGemvGather`,
  `encodeGemvGather`) làm mốc cơ sở FP32.

Các chữ ký hàm là cố định. Lab X2.4 liên kết chúng:

```cpp
void launchGemvQ8_0(const float* X, const int8_t* Wq, const uint16_t* Wd, float* Y,
                    int batch, int in, int out);
// Metal: kernel gemv_q8_0. Buffers: X 0 (float4), Wq 1 (char4), Wd 2 (half), Y 3,
//        batch 4, in 5, out 6.
void encodeGemvQ8_0(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                    id<MTLBuffer> X, id<MTLBuffer> Wq, id<MTLBuffer> Wd, id<MTLBuffer> Y,
                    uint32_t batch, uint32_t in, uint32_t out);
```

`encodeGemvQ8_0` ghi vào encoder được truyền cho nó và không commit gì cả.

## Các lệnh

- Kiểm thử: `python3 labs/pmpp.py run X2.1 --backend emu` trước, rồi
  `python3 labs/pmpp.py run X2.1` cho GPU của bạn.
- Đo: `python3 labs/pmpp.py run X2.1 --bench` (chỉ trên thiết bị thật).
- Lời giải tham chiếu: thêm `--solution`. Làm lại từ đầu: `python3 labs/pmpp.py reset X2.1 --yes`.

## Các test làm gì

- **Các phần của chính chương trình điều khiển được kiểm tra trước.**
  - `floatToHalfBits` được kiểm tra trên các giá trị suy ra bằng tay: hòa thì về chẵn, giá trị FP16 hữu hạn
    lớn nhất, tràn thành vô cực, số subnormal, và 1/127. Sau đó mọi mẫu trong 65536
    mẫu FP16 đi qua `halfBitsToFloat` và quay lại.
  - `quantizeRowQ8_0Ref` được kiểm tra trên một khối suy ra bằng tay (`qfixture.h`). Độ lớn lớn nhất
    của nó là 127/32, nên d = 1/32 chính xác (FP16 `0x2800`), và các mã của nó bao gồm các trường hợp
    hòa 2.5 và -2.5, mà `roundf` đưa về 3 và -3. Một khối toàn số không phải cho d = 0
    và mọi mã bằng 0.

  Test trình mô phỏng của các bài học được dự định dùng cùng các khối dữ liệu cố định đó.
- **Kích thước (batch, in, out):** (1, 32, 1), (1, 64, 5), (2, 96, 7), (3, 256, 64),
  (8, 512, 130), (5, 4096, 3). Trên GPU còn có (1, 4096, 4096), (8, 4096, 4096),
  (4, 14336, 4096), (1, 4096, 14336) và (8, 1024, 4099).
- **Các trọng số:**
  - Khối i của hàng j được nhân tỉ lệ với 2^((3j + 5i) mod 25 - 12), nằm giữa 2^-12 và 2^12.
    Các khối liền kề khi đó khác nhau một hệ số 32 trở lên (2^5, hoặc 2^20 ở chỗ phép
    mod quay vòng), nên một hệ số tỉ lệ lấy từ sai khối sẽ cho kết quả sai.
  - Mọi khối đều chứa cả +peak và -peak, nên các mã 127 và -127 xuất hiện trong mọi khối.
  - Với out >= 3, một hàng toàn số không (d = 0).
  - Các hệ số tỉ lệ nhỏ nhất là số subnormal FP16, mà `halfBitsToFloat` phải giải mã được.
- **Kích thước chính xác.** X, Wq và Wd chứa đúng số byte của chúng, nên trên `emu`, một lần đọc vượt quá bất kỳ
  mảng nào sẽ là một báo cáo AddressSanitizer chỉ đúng dòng.
- **Y được kiểm tra hai lần.** Y bắt đầu là NaN, nên một phần tử không ai ghi sẽ thất bại. Theo sau nó
  là một dải bảo vệ gồm các giá trị canh gác (sentinel) phải còn nguyên. Mỗi trường hợp khởi chạy hai lần mà không
  đặt lại Y.
- **Phép so sánh.** Kết quả được so với `y = x dequant(W)^T` tính bằng double trên
  CPU, từ cùng các hệ số tỉ lệ FP16 mà kernel đọc. Dung sai là
  1e-4 x (1 + sum of |x w|) cho đầu ra đó, không bao giờ so từng bit: kernel cộng theo thứ tự
  khác, và Metal không mặc định dùng số học IEEE nghiêm ngặt.

## Mỗi backend bắt được gì

- **`emu`** chạy từng luồng một dưới AddressSanitizer. Nó bắt được chỉ số sai, một
  hệ số tỉ lệ từ sai khối, một byte không được mở rộng dấu, các lần đọc vượt quá Wq hoặc Wd, và một
  lane bỏ qua phép rút gọn. Ở đây không có cập nhật chung nào, nên không có bản build `race`.
- **Metal** kiểm tra cùng các kết quả đó trên GPU của bạn. Nó đọc mã dưới dạng `char4` và hệ số tỉ lệ dưới dạng
  `half`, nên một lỗi trong `codeOf` hay `halfBitsToFloat` không thể lộ ra ở đó. Những phần đó thuộc về
  tệp họ CUDA, và `emu` kiểm tra chúng.
- **Không có gì ở đây kiểm tra một warp 64 lane.** Các trình giả lập mô phỏng warp 32 lane, và Metal
  báo độ rộng của riêng nó. Hãy kiểm tra bằng cách đọc: bước nhảy là `warpSize`, và số 32 duy nhất là
  `QK8_0`.

## Các backend khác nhau ở đâu (không có trong sách)

- **Ở đây không có `__dp4a`.** Các activation là float, nên tích vô hướng bên trong là các phép
  nhân cộng float trên các mã đã chuyển đổi.
  - `__dp4a` của CUDA (sm_61 trở lên, A2) không tồn tại trên HIP. HIP có
    `amd_mixed_dot` và các builtin riêng cho từng target thay thế (B1, B2).
  - Metal hoàn toàn không có tích vô hướng số nguyên cho từng luồng (C1).
  - Đường CUDA và HIP của llama.cpp lượng tử hóa các activation sang int8 ở mỗi token để dùng một
    tích vô hướng số nguyên (D3).
  - Đường Metal của llama.cpp giữ nguyên float, giống lab này (D4).
- **Hệ số tỉ lệ FP16.** CUDA và HIP sẽ dùng `__half2float`, từ hai header khác nhau (bảng
  trong x2-facts). Vì vậy tệp `.cu` giữ các bit trong `uint16_t` và giải mã chúng bằng
  `halfBitsToFloat`, vốn không cần header nào. (Các trình giả lập và stub front-end giờ cũng
  mô phỏng `__half`; lab này không dựa vào điều đó.) Metal có
  `half` sẵn trong ngôn ngữ.
- **Căn chỉnh.** `char4` có kích thước 4 và căn chỉnh 4 trên CUDA (A5) và trên Metal (infra-facts
  C9). Đó là lý do các mã nằm trong mảng riêng của chúng. Lab X2.2 đọc các khối 18 byte thật
  của llama.cpp, nơi điều đó không thể làm được.
- **`dot(half4, half4)` trả về `half`** trong MSL (C1), chính là bộ tích lũy hẹp mà phụ lục A
  cảnh báo. Kernel Metal chuyển sang `float4` trước.
- **HIP** chưa được build hay chạy ở đây: unverified.

## Trước khi benchmark, hãy dự đoán

Hãy ghi lại những điều này trước:

1. Mức tăng tốc của Q8_0 so với F32 ở batch 1 trên lớp 4096 -> 4096. Trần của mô hình
   byte là 4 / 1.0625 = 3.76. Đó là tỉ lệ số byte được lưu, nếu băng thông là
   chi phí duy nhất. Nó không phải một lời hứa.
2. Mức tăng tốc ở batch 8 lớn hơn hay nhỏ hơn ở batch 1, và vì sao. Hãy cân nhắc
   hai điều:
   - Số byte activation tăng theo batch, và chúng như nhau cho cả hai định dạng.
   - Công việc biến một mã thành một float và áp hệ số tỉ lệ của nó được làm một lần cho mỗi
     trọng số và được chia sẻ cho 8 hàng.

## Đọc kết quả đầu ra

- **Các lớp:** 4096 -> 4096, 4096 -> 14336 (một phép chiếu lên của MLP) và 14336 -> 4096 (một
  phép chiếu xuống). Hai kích thước sau là của khóa học này. Batch 1, 4 và 8.
- **Các kernel:** `gather` (F32, lời giải của lab 22.1) và `q8_0` (của bạn). Chúng chạy luân phiên: một
  vòng không tính giờ, rồi năm vòng, và mỗi kernel giữ vòng tốt nhất của nó.
- **Các bản sao của W.** Một lần submit có tính giờ khởi chạy kernel một lần cho mỗi bản sao trọng số của nó,
  đi qua các bản sao mà tổng cộng chứa ít nhất 256 MB cho mỗi định dạng (như lab 22.1). Vì vậy một
  lần khởi chạy không tìm thấy trọng số của nó ở chỗ lần khởi chạy trước để lại. Header
  của mỗi lớp cho biết có bao nhiêu bản sao. `time (us)` là cho mỗi lần khởi chạy. CUDA và HIP đo giờ
  bằng đồng hồ của host; Metal dùng thời điểm bắt đầu và kết thúc trên GPU của command buffer.
- **`W GB/s`** là số byte mà mỗi định dạng lưu (Q8_0: mã và hệ số tỉ lệ) chia cho
  thời gian. Nó không phải tất cả những gì hệ thống bộ nhớ đã di chuyển. Trang labs so sánh nó với băng thông
  đo được của lab 5.1, giới hạn tốc độ của một phép nhân ma trận với vector.
- **`vs F32`** là thời gian F32 chia cho thời gian của kernel này.
- **Các hàng báo cáo:** `qgemv`: `{kernel, format, batch, in, out, seconds, weight_bytes,
  weight_gb_per_s, speedup_vs_f32}`.

Rồi so các con số của bạn với dự đoán. Mức tăng tốc đo được thấp hơn tỉ lệ byte,
hay vượt qua nó, ở đâu? Hãy nhớ hai kernel khác nhau ở nhiều điều hơn là trọng số: kernel này đọc X
bốn float mỗi lần, còn kernel của lab 22.1 đọc một. Các con số là phép đo trên thiết bị của bạn ở những kích thước này,
không phải thuộc tính của các kernel.

## Đi xa hơn

- **Activation số nguyên, như đường CUDA của llama.cpp** (x2-facts D3). Lượng tử hóa mỗi khối
  32 activation sang int8 với hệ số tỉ lệ riêng của nó, một lần cho mỗi token. Khi đó tích vô hướng bên trong
  trở thành một tích vô hướng số nguyên với một lần nhân lại tỉ lệ float cho mỗi khối. Trên CUDA, hãy thử `__dp4a`
  sau `#if` (A2). Các trình giả lập và stub không khai báo nó, nên hãy giữ một phương án dự phòng bốn tích.
  Đo cả lượt chạy thêm và những gì nó mang lại.
- **Bố cục riêng của llama.cpp.** Đọc `block_q8_0` 34 byte xen kẽ của llama.cpp. Các lần
  nạp của bạn trở thành căn chỉnh 2 byte (A5), nên hãy đọc mã thành các word 16 bit, như lab X2.2 làm.
  Đo xem bố cục tách riêng đáng giá bao nhiêu.
