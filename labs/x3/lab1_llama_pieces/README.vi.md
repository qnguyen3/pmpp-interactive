> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab X3.1: những thành phần Llama mà khóa học chưa xây, và một buổi review bản port (các thành phần cho mốc M10)

Không có trong sách: học phần X3 là của khóa học này. Phép rút gọn (reduction) là của lab 10.3 (phép rút gọn
hai giai đoạn của sách, mục 10.8, Fig. 10.16, p. 243, cùng các chỉnh sửa của lab đó). Những gì các kernel
tính, và các quy ước mà test kiểm tra, đến từ các dự án được đọc cho học phần này ngày
2026-11-20 tại các commit được ghim trong `notes/today/x3-facts.md`. Một mục trần như (B2) bên dưới
nằm trong tệp đó.

Các mốc M0 đến M8 cho bạn các kernel theo từng phần tử, các GEMM, softmax và argmax, attention
với KV cache, GEMV và embedding. Một bước decoder Llama cần thêm ba thứ nữa (ghi chú dự án
trong x3-facts): RMSNorm (B2), RoPE (B6) và khối feed-forward có cổng SiLU, SwiGLU (B1).
Phần A là viết chúng ở dạng nhỏ nhất. Phần B là đọc mã theo cách bạn sẽ đọc
llama.cpp: một RMSNorm theo cấu trúc các kernel của nó, với ba lỗi cần tìm.

- Sửa `work/kernel.cu` (CUDA; cũng được build cho HIP và các trình giả lập) và `work/kernel.metal` +
  `work/launch.mm` (Metal). Các tên là cố định: chương trình điều khiển (driver) gọi `launchRmsNorm`,
  `launchRope`, `launchSwiGLU`, `launchRmsNormPort`, các kernel Metal `rms_norm_rows`,
  `rope_rows`, `swiglu`, `rms_norm_port`, và các encoder `encodeRmsNorm`, `encodeRope`,
  `encodeSwiGLU`, `encodeRmsNormPort`, những hàm ghi vào encoder được truyền cho chúng.
- Kiểm thử: `python3 labs/pmpp.py run X3.1`. Bắt đầu với `--backend emu`.
- Race: `python3 labs/pmpp.py run X3.1 --backend race`.
- Đo: `python3 labs/pmpp.py run X3.1 --bench` (chỉ phần A; hãy đưa ra dự đoán bên dưới trước).

## Phần A: hợp đồng

float32 theo thứ tự hàng (row-major), token (hàng) đứng trước, con trỏ thô và kích thước làm tham số.

- `launchRmsNorm(x, w, y, rows, d, eps)`: `y[r][c] = x[r][c] * (1 / sqrtf(mean(x[r]^2) + eps)) * w[c]`,
  với `x`, `y` có kích thước `[rows x d]` và `w` có kích thước `[d]`. `y` có thể là `x`. Một khối 256 luồng cho mỗi
  hàng, với phép tính tổng khối của lab 10.3: lane mask có kiểu, `warpSize`, giai đoạn 2 có điều kiện bảo vệ. Đây là những gì
  llama.cpp, MLX và PyTorch tính cho mỗi hàng (B2); Hugging Face ép kiểu trở lại kiểu đầu vào
  trước phép nhân với trọng số, điều mà ở float32 không thay đổi gì.
- `launchRope(x, pos, cosT, sinT, rows, heads, headDim, mode)`: tại chỗ trên
  `x[rows x heads x headDim]`, `headDim` chẵn. Hàng `r` ở vị trí `pos[r]`. Các bảng có kích thước
  `[maxPos x headDim/2]`, được chương trình điều khiển dựng bằng double với góc `p * 10000^(-2i/headDim)`
  (góc mà llama.cpp, Hugging Face và llama2.c dùng, B6, D1) và làm tròn sang float một lần, nên
  kernel không làm phép lượng giác nào. Cặp `i` của một head dùng phần tử `i` của bảng và ánh xạ
  `(a, b)` thành `(a cos - b sin, a sin + b cos)`:
  - mode 0 xoay các phần tử `(2i, 2i+1)`: NORM của llama.cpp (`rope_norm`), llama2.c;
  - mode 1 xoay các phần tử `(i, i + headDim/2)`: NeoX, `rotate_half` của Hugging Face, mặc định
    của mlx-lm (`rope_traditional = False`).
- `launchSwiGLU(gate, up, out, n)`: `out[i] = gate[i] / (1 + expf(-gate[i])) * up[i]`
  (`op_silu` của llama.cpp, B1). `out` có thể là `gate`.

Metal: `rms_norm_rows` (một threadgroup 256 cho mỗi hàng, `simd_sum`, độ rộng SIMD group đọc
từ pipeline), `rope_rows` và `swiglu`, với cùng các hợp đồng. `encodeRmsNorm`
trả về false, không dispatch, khi pipeline không thể chạy 256 luồng thành các SIMD group trọn vẹn
với số SIMD group không nhiều hơn số lane.

### Hai cách ghép cặp RoPE

Chúng không phải hai cách viết cùng một thứ: trên cùng đầu vào chúng cho đầu ra
khác nhau. Một checkpoint được huấn luyện với cách này chỉ chạy được với cách kia sau khi các hàng trọng số Q và K
của nó được hoán vị. llama.cpp chạy Llama với NORM và hoán vị trọng số của Hugging Face khi
chuyển đổi (B6):

```python
def permute(weights, n_head, n_head_kv):
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2).reshape(weights.shape))
```

áp dụng cho `q_proj.weight` với `(n_head, n_head)` và cho `k_proj.weight` với
`(n_head, n_kv_head)`. Trong mỗi head, nó đưa hàng `k * headDim/2 + j` đến hàng `2j + k`. Chương trình
điều khiển kiểm tra đẳng thức mà điều này mang lại, trên kernel của bạn: mode 1 trên `W x` bằng mode 0 trên
`(P W) x` sau khi hoàn tác phép hoán vị, cho Q (4 head) và cho K (2 KV head dưới 4 head
query). Nó cũng kiểm tra rằng hoán vị K bằng số head của query sẽ phá vỡ đẳng thức
(của khóa học này; x3-facts D6 đọc thấy cùng lỗi đó trong `export.py` của llama2.c). Trước khi
chạy, hãy tự hỏi vì sao cùng phần tử bảng `i` phục vụ cặp `(2i, 2i+1)` ở một mode và
`(i, i + headDim/2)` ở mode kia.

### SwiGLU và vô cực

Test đưa vào `gate = -100`. Ở float, `expf(100)` tràn thành vô cực. Với phép chia IEEE mặc định
của CUDA, `-100 / infinity` là `-0`, nên công thức của llama.cpp cho một kết quả hữu hạn. Dưới
fast math của Metal, có thể là mặc định của runtime này (ch05-facts C7), hành vi của
vô cực là không xác định (ch10-facts C3): hãy viết kernel Metal sao cho không có gì bị tràn. Cùng
hàm đó có một dạng trong đó đối số của hàm mũ không bao giờ dương.

## Phần B: bản port

`work/kernel.cu` (`rmsNormPortKernel`, `launchRmsNormPort`) và `work/kernel.metal`
(`rms_norm_port`, với `encodeRmsNormPort` trong `work/launch.mm`) chứa một RMSNorm mà một đồng nghiệp
đã port theo cấu trúc các kernel của llama.cpp (B2):

- phần khởi chạy chọn 256 luồng cho các hàng dưới 1024 phần tử, ngược lại thì 1024 (đường
  vô hướng của llama.cpp), và một đường float4 khi nó cho là an toàn;
- một lưới (grid) cố định gồm 4 khối đi qua các hàng (khối `b` làm các hàng `b`, `b + 4`, ...), nên một
  khối dùng lại buffer chia sẻ của nó hết hàng này đến hàng khác; bản thân llama.cpp chạy một khối cho mỗi hàng;
- phép rút gọn có hai giai đoạn qua một buffer chia sẻ `s_sum` gồm 64 vị trí (mỗi vị trí cho một lane của
  warp rộng nhất ở đây); hệ số tỉ lệ được chuyển cho khối qua `s_sum[0]`.

Nó biên dịch được, và qua được một số test. Nó có **ba lỗi độc lập**, cùng ba lỗi đó
trong cả hai tệp; trên Metal một trong số chúng nằm trong `launch.mm`, vì ở đó, như trong llama.cpp,
host chọn đường float4. Hãy tìm và sửa cả ba. Hàm `launchRmsNormPort` nhận
cùng tham số như `launchRmsNorm`, và phải cho cùng kết quả với mọi `rows`, `d >= 1`
(không tại chỗ).

Không phải lỗi: lane mask có kiểu và `warpSize`; phép ép kiểu `float*` sang `float4*` trong
`kernel.cu`, vốn là kỹ thuật của sách và không nhà cung cấp nào ghi trong tài liệu (infra-facts A6);
việc gắn mỗi buffer Metal lần thứ hai dưới dạng `device float4*`, cách được Metal ghi trong tài liệu để đọc
bốn float cùng lúc (infra-facts C7); 64 vị trí; lưới 4 khối; quy tắc chọn kích thước khối; `1 /
sqrtf`; việc đọc hệ số tỉ lệ từ `s_sum[0]`.

Phương pháp:

1. `python3 labs/pmpp.py run X3.1 --backend emu` trước. Ba thứ tự luồng của nó là
   tất định; bộ nhớ chia sẻ (shared memory) bắt đầu được lấp đầy bằng byte `0xCD` trong mọi khối, nên một giá trị
   không ai ghi sẽ hiện ra là rác; và AddressSanitizer dừng lần chạy ở lần đọc đầu tiên vượt quá một
   vùng cấp phát và chỉ ra dòng. Các trường hợp test của bản port được sắp xếp để những trường hợp đọc
   vượt quá một vùng cấp phát đến sau cùng.
2. Sửa một chỗ, chạy lại. Các trường hợp thất bại thay đổi khi bạn tiến lên.
3. `python3 labs/pmpp.py run X3.1 --backend race` chạy các luồng cùng lúc dưới
   ThreadSanitizer và chỉ ra hai dòng mã nguồn của một lần ghi và một lần đọc mà không có gì sắp thứ tự.
   Điều này quan trọng ở đây: trình giả lập chạy mỗi luồng đến điểm đồng bộ hóa tiếp theo của nó (một
   rào chắn (barrier) hoặc một warp shuffle) trước khi luồng tiếp theo chạy, nên nó chỉ cho thấy một rào chắn bị thiếu
   khi một luồng ghi vào buffer chia sẻ trước điểm đồng bộ hóa tiếp theo của chính nó.
   ThreadSanitizer thấy được trong cả hai trường hợp, nhưng chỉ trong những lịch chạy đã xảy ra.
4. Rồi chạy trên GPU của bạn: `python3 labs/pmpp.py run X3.1`. Ghi lại GPU đã cho bạn thấy những lỗi nào
   và đã che giấu những lỗi nào. Bộ nhớ chia sẻ hay bộ nhớ threadgroup mà không ai ghi thì chứa bất cứ thứ gì nó đang chứa;
   GPU đặt gì vào đó không phải là điều mà bất kỳ tệp facts nào chúng tôi đã kiểm tra ghi lại.

Các trường hợp, và mỗi kích thước dùng để làm gì: rows 1 và 3 (một hàng cho mỗi khối) và rows lớn hơn 4
(nhiều hàng cho mỗi khối; 6 trên `race`, 9 trên `emu`, 17 trên GPU); `d` 64 và 4096 (bội số của
4, với 256 và 1024 luồng); `d` 577, 1001 và 4097 (không phải bội số của 4). Trên thiết bị thật
và trên `race`, `x` và `w` có theo sau một dải bảo vệ gồm các giá trị 1000.0, nên một lần đọc vượt quá cuối mảng
làm thay đổi kết quả thay vì đọc ra số không; trên `emu` chúng được cấp phát đúng kích thước
để AddressSanitizer bắt được lần đọc.

Khi bạn đã tìm được cả ba, hãy so sánh với llama.cpp: mỗi lỗi là thứ mà các kernel của nó phòng
ngừa (B2; ch10-facts D1, nơi softmax của nó mang chú thích "sync is needed as we
reuse buf_iw across block_reduce invocations, see #26385").

## Các test kiểm tra gì

- RMSNorm: rows 1, 3 và 17 nhân với `d` 1, 7, 64, 128, 576 (độ rộng của SmolLM2-135M, D4), 1000 và
  4097, tại chỗ và không tại chỗ; các hàng có tỉ lệ khác nhau trong cùng một batch, nên một kernel
  dùng tổng của hàng khác sẽ bị bắt; một hàng toàn số không, mà đầu ra phải bằng 0; `eps = 1e-5`
  (llama2.c và SmolLM2, D1, D4). Các đầu ra bắt đầu là NaN, 64 giá trị bảo vệ sau `y` phải
  còn nguyên, và `w` (và `x`, khi không tại chỗ) phải không thay đổi. Backend race chạy
  rows 1 và 3 nhân với `d` 1, 7, 64 và 1000.
- RoPE: rows 1 và 3 ở các vị trí riêng biệt bao gồm 0 và `maxPos - 1` (`maxPos` 4096 trên
  GPU, 512 trên `emu`, 64 trên `race`); 1, 2 và 4 head; `headDim` 8, 64 và 128 (64 và 8 trên
  `race`); cả hai mode so với bản tham chiếu; vị trí 0 phải trả lại chính xác đầu vào ở
  cả hai mode; hai mode phải cho đầu ra khác nhau; đẳng thức hoán vị ở trên.
- SwiGLU: `n` 1, 31, 1000 và 4099 (không có 4099 trên `race`), không tại chỗ và `out == gate`;
  các gate phân bố đều trong [-8, 8] với -100 và +100 được cài ở hai đầu; mọi kết quả hữu hạn.
- Bản port: các trường hợp của phần B.

## Dung sai, và vì sao

Mọi kết quả được so với một bản tham chiếu tính bằng double trên CPU từ cùng các đầu vào float,
theo `|got - ref| <= k * FLT_EPSILON * (1 + |ref|)` (`FLT_EPSILON` là 2^-23; một lần
làm tròn tối đa bằng một nửa giá trị đó, tính tương đối). Phần `1 +` giữ cho cận có ý nghĩa ở chỗ `ref`
bằng 0. Các cận sai số bên dưới tính theo đơn vị `FLT_EPSILON`, tương đối với giá trị.

- RMSNorm, `k = 32`. Tổng bình phương: một luồng cộng lần lượt tối đa `ceil(d / 256)` bình phương
  (17 ở `d = 4097`), rồi 5 tầng của cây warp và thêm 5 tầng ở giai đoạn 2 (6 và 6 với
  warp 64 lane). Mọi số hạng đều dương, nên tổng lệch tối đa một nửa cho mỗi phép cộng
  trên đường dài nhất, cộng thêm phép bình phương: khoảng (17 + 12 + 1) / 2 = 15. Hệ số tỉ lệ nhận một nửa
  số đó qua căn bậc hai, cộng thêm phép chia cho `d`, phép `+ eps`, căn bậc hai và
  nghịch đảo: trên CUDA, `sqrtf` và `/` được làm tròn đúng theo mặc định (ch21-facts A10);
  dưới fast math của Metal, `x / y` nằm trong 2.5 ulp, `1.0 / x` trong 1 và `sqrt` là
  `x * rsqrt(x)` với `rsqrt` trong 2 (ch21-facts C3), tổng cộng tối đa khoảng 7. Rồi hai
  phép nhân. Tổng khoảng 7.5 + 7 + 1, dưới 16; `k = 32` chừa gấp đôi số đó. Thứ tự mà
  `simd_sum` cộng không được ghi trong tài liệu (ch10-facts A4, C1): ngay cả khi nó cộng 32 lane của mình lần lượt
  từng cái một thì cận sẽ vào khoảng 25.
- RoPE, `k = 8`. Mỗi đầu ra là hai phép nhân và một phép cộng các giá trị có độ lớn tối đa 1: dưới 3,
  dù có hay không có phép nhân cộng gộp (fused multiply-add). Bản tham chiếu đọc cùng các bảng float, nên
  sai số làm tròn của chúng không được tính; test vị trí 0 là chính xác.
- SwiGLU, `k = 16`. CUDA: `expf` trong 2 ulp (ch10-facts A11), rồi một phép cộng, một phép chia
  làm tròn đúng và phép nhân: dưới 4. Dạng Metal với `e = exp(-|g|)`: `exp` nhanh
  nằm trong `3 + floor(|2x|)` ulp (ch10-facts C4), lớn khi `|g|` lớn, nhưng khi đó `e`
  nhỏ: với `g >= 0` nó dịch `1 / (1 + e)` tối đa khoảng 1.5; với `g < 0` bản thân kết quả
  nhỏ, và sai số tuyệt đối vẫn dưới khoảng 4 trên [-8, 8]. HIP công bố
  các chênh lệch đo được so với thư viện C++, không phải các cận (ch10-facts B4, ch21-facts B2):
  ở đó là unverified.

Một hàng sai, một cách ghép cặp sai, một trọng số bị thiếu hay một phép rút gọn đọc phải rác sẽ lệch
nhiều bậc độ lớn.

## Trước khi benchmark

Hãy ghi lại dự đoán của bạn (bài học hỏi điều này dưới tên `x3/labs/p-lab1-rmsnorm-bw`): `launchRmsNorm` sẽ đạt
bao nhiêu phần băng thông lab 5.1 của bạn ở rows = 1, d = 4096, và ở
rows = 256? Ở rows = 1, một lời gọi di chuyển khoảng 48 KiB: `x` và `y`, mỗi cái 16 KiB, và trọng số
16 KiB.

Những lực cần cân nhắc, không phải đáp án:

- Một hàng là một khối, và một khối chạy trên một trong các lõi (core) của GPU; phần còn lại phải chờ.
- Mỗi lần khởi chạy tốn một khoảng thời gian cố định, bất kể nó di chuyển bao nhiêu.
- Benchmark gọi kernel nhiều lần trên cùng các buffer, nên dữ liệu vừa trong
  bộ nhớ đệm (cache) không cần đến từ bộ nhớ sau lời gọi đầu tiên. Lab 5.1 truyền qua một tập dữ liệu làm việc lớn hơn
  nhiều so với bất kỳ bộ nhớ đệm nào.

## Benchmark

Chỉ trên thiết bị thật. `launchRmsNorm`, `launchRope` (mode 0) và `launchSwiGLU` ở rows (token)
1, 16 và 256 với `d` = 576 và 4096. RoPE chạy trên các head 128 khi `d` là bội số của
128 (4096 = 32 x 128) và các head 64 trong trường hợp còn lại (576 = 9 x 64, kích thước của SmolLM2-135M, D4);
SwiGLU trên `n = rows x 4d`. GB/s đếm số byte duy nhất mà mỗi lời gọi phải đọc và ghi, một lần:
RMSNorm `x`, `w` và `y`; RoPE `x` hai lần, `pos` và các hàng cos và sin của các vị trí
của các hàng; SwiGLU `gate`, `up` và `out`. Thời gian mỗi lời gọi là kết quả tốt nhất trong 5 lần chạy có tính giờ sau một
lần không tính giờ, mỗi lần chạy lặp lại lời gọi liên tiếp cho đến khi kéo dài ít nhất 2 ms (phương pháp của lab
10.3); thời gian là của thiết bị (event của CUDA và HIP, `GPUStartTime` đến `GPUEndTime` của command buffer
trên Metal). Mỗi kết quả có tính giờ được kiểm tra trước khi được báo cáo. Ở chỗ
`reports/runs/<backend>.json` chứa một băng thông của lab 5.1 (của bản làm của bạn nếu có, ngược lại là của
lời giải), mỗi hàng còn hiển thị tỉ lệ so với nó. Các con số là phép đo trên máy của bạn
ở những kích thước này, không phải mục tiêu.

## Tính khả chuyển

- HIP: `kernel.cu` build được qua `pmpp/vendors/hip.h` với mask có kiểu và `warpSize`.
  Với warp 64 lane, một khối 256 có 4 warp; 64 vị trí của bản port đủ cho một giai đoạn 2 với 64 lane.
  Chưa được biên dịch hay chạy trên phần cứng AMD ở đây: unverified.
- Metal: độ rộng SIMD group lấy từ pipeline (`threadExecutionWidth`), không bao giờ là một
  hằng số; độ dài bộ nhớ threadgroup là bội số của 16 byte (ch05-facts C8). Việc một
  buffer được gắn ở offset 0 có bắt đầu tại ranh giới 16 byte mà một `float4` cần hay không thì không được nêu trong
  các tệp facts chúng tôi đã kiểm tra: unverified, dù mọi test ở đây đều đi qua nó.
- `expf` và `sqrtf` không cần khai báo riêng: bước kiểm tra front-end CUDA của khóa học
  khai báo các hàm toán học độ chính xác đơn trong header stub của nó (một phiên bản trước của
  tệp này khai báo chúng dưới `PMPP_CUDA_FRONTEND_CHECK`, từ trước khi stub có chúng).
