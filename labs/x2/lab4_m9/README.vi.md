> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab X2.4: bước decode đã lượng tử hóa (mốc dự án M9)

Không có trong sách: học phần X2 là của khóa học này. Điểm neo của nó là phụ lục A mục A.5 (p. 593, ghi lại trong
`notes/appA.md`): một định dạng số nhỏ hơn di chuyển ít byte hơn cho mỗi phép toán, điều đó làm tăng
cường độ tính toán của một kernel bị giới hạn bởi băng thông. Một bước decode là một kernel như vậy: ở batch
1, mọi phép chiếu là một phép nhân ma trận với vector đọc mỗi trọng số một lần (`ch22-facts` A5). Các
định dạng là của llama.cpp; bước decode là của lab 22.2. Một mục trần như (D1) bên dưới nằm trong
`notes/today/x2-facts.md`.

Lab này lấy bước decode greedy của lab 22.2 (M8) và lưu trọng số của nó dưới dạng Q8_0 hoặc Q4_0, rồi
đo, trên thiết bị của bạn, bao nhiêu phần của tỉ lệ byte biến thành tốc độ và cái giá của nó về
sai số đầu ra so với bước float32.

## Phần của bạn

(a) `launchEmbedQ(E, tokens, H, batch)`: một phép gather có giải lượng tử hóa. Một luồng cho mỗi phần tử của H, như
trong lab 22.2; luồng sở hữu `H[r][j]` tìm phần tử j của hàng `tokens[r]` của E, và với một
E đã lượng tử hóa thì đọc mã của nó và hệ số tỉ lệ của khối chứa nó. Q8_0: `code * scale`. Q4_0: `(code - 8) *
scale`, trong đó vị trí k của một khối là nibble thấp của byte k khi k < 16 và nibble cao
của byte k - 16 trong trường hợp còn lại (D1: llama.cpp đóng gói trọng số j với trọng số j + 16, không phải hai trọng số liền kề).
Kết quả phải giống từng bit với những gì bộ giải lượng tử hóa của CPU cho ra: một phép nhân float.

(b) `launchQLinear(X, W, Y, batch)`: `Y = X W^T`, dispatch theo `W.fmt` tới `launchGemvGather` của lab 22.1
(F32), `launchGemvQ8_0` của lab X2.1 hoặc `launchGemvQ4_0` của lab X2.2. Batch tối đa
là 8: đây là các kernel nhân ma trận với vector, như của llama.cpp với tối đa 8 token (`ch22-facts` D3).

(c) `launchQDecodeStep(m, s)`: bước decode của lab 22.2 với (a) và (b) thay cho phần embedding và
lớp tuyến tính của nó. Một lần khởi chạy cho mỗi phép toán cho cả batch, không có gì phải chờ:

| Phép toán | Thành phần | Kích thước |
|---|---|---|
| `H = dequantised E[tokens]` | của bạn, `launchEmbedQ` | `[batch x d]` |
| với mỗi lớp: `T = H W_l^T` | của bạn, `launchQLinear` | `[batch x d]` |
| `T += b_l` | M0 `launchAddBias`, tại chỗ | |
| `H += T` | M0 `launchAddResidual` | |
| `logits = H E^T` | lại là `launchQLinear`: trọng số của LM head là E | `[batch x vocab]` |
| `probs = softmax(logits / temperature)` | M4 `launchSoftmaxRows` | `[batch x vocab]` |
| `tokens = argmax(logits)` | M4 `launchArgmaxRows`, trên các token mà bước đã đọc | `[batch]` |

Các trọng số vẫn ở dạng lượng tử hóa trong bộ nhớ thiết bị. Không có gì trong bước ghi ra một bản sao đã giải lượng tử hóa của một
ma trận trọng số: việc đó sẽ di chuyển nhiều byte hơn cả bước float32 (A1, nơi phép đo của chính PyTorch
với một đường int8 không gộp cho kết quả chậm hơn FP16). Các GEMV giải lượng tử hóa bên trong
tích vô hướng của chúng.

Trên Metal, cùng bước đó là `embed_rows_q8_0` và `embed_rows_q4_0` trong `kernel.metal`
(`embed_rows_f32` được cung cấp), và `encodeEmbedQ`, `encodeQLinear` và `encodeQDecodeStep` trong
`launch.mm`, những hàm đặt mọi dispatch vào compute encoder của bên gọi: một command buffer cho mỗi
bước.

## Của khóa học này, không phải của sách

- **Các định dạng.** Q8_0 và Q4_0 là của llama.cpp (D1): các khối 32 trọng số dọc theo một hàng, mỗi khối một
  hệ số tỉ lệ FP16; mã Q8_0 trong [-127, 127], mã Q4_0 từ 0 đến 15 mang nghĩa `code - 8`. Mã và hệ số tỉ lệ của Q8_0
  được giữ trong hai mảng (bố cục của lab X2.1; llama.cpp xen kẽ chúng trong một khối 34 byte);
  Q4_0 đúng là khối 18 byte của llama.cpp. `../shared-g12/qformats.h` chứa các struct
  và các quy tắc CPU.
- **Lượng tử hóa lúc nạp.** Chương trình điều khiển (driver) lượng tử hóa mô hình đồ chơi float32 của lab 22.2 bằng các bộ lượng tử hóa
  `_ref` của llama.cpp (`quantize_row_q8_0_ref`, `quantize_row_q4_0_ref`, D1), quy tắc dùng để tạo một tệp
  mô hình khi không dùng importance matrix. Không có gì được tải xuống.
- **Các cặp định dạng.** Mọi lớp ở một định dạng, embedding gắn liền và LM head ở một định dạng khác:
  `(F32, F32)`, `(Q8_0, Q8_0)`, `(Q4_0, Q4_0)` và `(Q4_0, F32)`. Cặp cuối giữ
  head 32,000 x 4,096 ở float32, và đo xem việc đó tốn gì về thời gian và mua được gì về sai số.
- **Mô hình** là của lab 22.2: embedding, hai khối gồm lớp tuyến tính, bias và residual, LM head
  gắn liền, bộ lấy mẫu; không có attention (chương 20 xây dựng nó).

## M9 đi tiếp đến đâu

Mốc M10 của học phần X3 (lab X3.2) chạy một bước kiểu Llama có lớp tuyến tính là một
`LinearFn`, `(X, const float* W, Y, batch, in, out)`. Nó gọi `launchQLinear` tham chiếu của lab này
(và `encodeQLinear` trên Metal) qua `labs/x3/lab2_tinyllama/provided/m9_linear`,
vốn gắn mỗi con trỏ trọng số float của mô hình với một `QMatrix`, để cùng bước đó chạy được với
trọng số Q8_0 và Q4_0 và được kiểm tra với một bản tham chiếu CPU trên cùng các trọng số đã lượng tử hóa.
Lab X3.2 dùng lời giải tham chiếu (solution), không phải `work/` của bạn, và `provided/pieces.cmake` của nó chỉ dùng
lời giải đó nếu `launchQLinear` có chữ ký mà `qdecode.h` khai báo.

## Các tệp

- `work/kernel.cu` (CUDA, HIP, các trình giả lập): `embedRowsQ8_0Kernel`, `embedRowsQ4_0Kernel`,
  `launchEmbedQ`, `launchQLinear`, `launchQDecodeStep`. `embedRowsF32Kernel` và `halfToFloat`
  được cung cấp.
- `work/kernel.metal`: `embed_rows_q8_0`, `embed_rows_q4_0` (`embed_rows_f32` được cung cấp).
  `work/launch.mm`: `encodeEmbedQ`, `encodeQLinear`, `encodeQDecodeStep`.
- Được cung cấp, không sửa: `qdecode.h` (`QFormat`, `QMatrix`, `QDecodeModel`, `QPipelines`, các
  khai báo của các thành phần; `DecodeState` lấy từ `decode.h` của lab 22.2);
  `provided/pieces.cmake`, tệp sao chép các lời giải tham chiếu của lab 2.3 (M0), lab 10.3 (M4),
  lab 22.1 (phép gather), lab X2.1 và lab X2.2 vào thư mục build, phỏng theo tệp của lab 22.2;
  `provided/step_timer.cu`, cặp event đo giờ một bước trên CUDA và HIP. Nếu lời giải của lab X2.1 hoặc
  X2.2 bị thiếu hoặc có chữ ký khác, `../shared-g12/qpieces.cmake` build một
  GEMV thay thế có cùng hình dạng; những dòng đầu của mỗi lần chạy cho biết cái nào đã được dùng.
- `main.cpp`: các test, benchmark, bước tham chiếu CPU bằng double trên các trọng số đã giải lượng tử hóa,
  và bộ sinh mô hình đồ chơi tất định.

## Các lệnh

- Kiểm thử: `python3 labs/pmpp.py run X2.4 --backend emu` trước, rồi `python3 labs/pmpp.py run
  X2.4` trên GPU của bạn. Lần chạy trên trình giả lập mất vài phút: nó chạy mọi luồng của mọi lần khởi chạy của
  mọi bước dưới ba thứ tự luồng, cho bốn cặp định dạng.
- Đo: `python3 labs/pmpp.py run X2.4 --bench`.

## Các test làm gì

- Mô hình đồ chơi: d = 64, vocab = 96, 2 lớp (d và vocab đều là bội số của 32). Batch 1, 3 và 8 trên
  các trình giả lập, 1 đến 8 trên GPU; 3 bước trên các trình giả lập, 4 trên GPU; mọi cặp định dạng.
- Riêng phần embedding, với E ở mỗi định dạng, trên một batch 7 chứa token 0, token
  `vocab - 1` và một token được dùng hai lần: mọi hàng của H phải giống từng bit với hàng đã giải lượng tử hóa
  của E do CPU tính.
- Sau mỗi bước, với mỗi chuỗi: H và logits được so với một bước CPU tính bằng double
  trên các trọng số đã giải lượng tử hóa (các số float mà bộ giải lượng tử hóa của CPU cho ra), trong phạm vi
  dung sai của lab 22.2 `1e-4 * (1 + mag)`; các xác suất trong phạm vi `1e-20 + 1e-4 * p` của softmax, tính bằng
  double, của logits mà bước của bạn đã tính. Token tiếp theo phải bằng argmax của bản tham chiếu
  ở mọi chỗ hai logits tốt nhất của bản tham chiếu cách nhau xa hơn tổng hai dung sai
  của chúng; ở đó, mọi bước đúng đều phải chọn nó. Lần chạy in ra số token đã được
  so sánh như vậy.
- Tính bất biến theo batch, sau khi các test theo batch đã qua: mọi chuỗi của batch lớn nhất decode, từng bước
  một, đúng y như khi nó chạy một mình.
- Các trọng số (bản sao của mọi định dạng) và các bias không thay đổi sau lần chạy. H, T, logits,
  probs và tokens mỗi cái đều có một dải bảo vệ theo sau phải còn nguyên, và các đầu ra bắt đầu
  là NaN. Các trọng số được cấp phát đúng kích thước của chúng, nên trên `emu` một lần đọc vượt quá một mảng là một
  lỗi AddressSanitizer chỉ ra đúng dòng.
- Trên GPU, còn có kích thước mặc định của lab 22.2 (d = 4096, 2 lớp, vocab = 32,000), một batch 8,
  hai bước, mọi cặp.
- Không chạy trên backend `race`: mã của bạn không có cập nhật chung nào (mọi luồng ghi phần tử
  của riêng nó), và các thành phần đã được kiểm tra race trong lab của riêng chúng.

## Trước khi benchmark, hãy dự đoán

1. Mức tăng tốc của bước `(Q4_0, Q4_0)` so với bước `(F32, F32)` ở batch 1. Hãy bắt đầu từ
   tỉ lệ byte của cả bước (trình khám phá của mục X2.7 tính nó): mọi trọng số được đọc một lần,
   nên mô hình dự đoán tỉ lệ số byte trọng số, khoảng 7.1 cho Q4_0 so với float32 (E1).
   Rồi làm tương tự cho `(Q4_0, F32)`, nơi head 32,000 x 4,096 vẫn ở float32: mô hình byte
   cho 1.21. Ghi lại bạn kỳ vọng bao nhiêu phần của mỗi tỉ lệ còn lại.
2. Trong 32 bước greedy từ các token khởi đầu cố định, bao nhiêu bước khớp với bước float32 ở Q8_0 và
   ở Q4_0?

Những điều cần cân nhắc:

- Một GEMV đã lượng tử hóa làm nhiều việc hơn cho mỗi byte so với phép gather float của lab 22.1: nó giải nén mã, đọc
  hệ số tỉ lệ và nhân với chúng. Việc công việc đó có ẩn được dưới lưu lượng bộ nhớ hay không là điều mà
  phép đo cho bạn biết (A1: một phép đo đã công bố còn cách khá xa tỉ lệ byte).
- Các activation, đầu ra, bias, softmax và argmax không co lại cùng với
  trọng số. Ở batch 1 chúng nhỏ so với trọng số; chúng tăng theo batch.
- Dự đoán 1.21 là một phát biểu về byte. Nếu head chiếm phần lớn số byte của bước, nó cũng
  chiếm phần lớn thời gian, trừ khi có thứ khác giới hạn bước đó.
- Mô hình đồ chơi là ngẫu nhiên và không có attention: token tiếp theo của nó chỉ phụ thuộc vào token hiện tại.
  Mức khớp của nó không nói gì về mô hình đã huấn luyện, và sai số logit của nó chỉ nói việc
  làm tròn dịch chuyển những trọng số cụ thể này bao xa.

## Benchmark đo gì

Chỉ trên thiết bị thật. Kích thước và seed mặc định của lab 22.2: d = 4096, 2 lớp, vocab = 32,000,
embedding gắn liền với LM head. Cả ba định dạng của mọi ma trận đều nằm trên thiết bị cùng lúc; nếu
chúng chiếm hơn một phần tư bộ nhớ mà thiết bị báo cáo, d giảm một nửa cho đến khi vừa,
và kích thước được in ra. Batch 1, 2, 4 và 8. Một bước là một command buffer trên Metal
(`GPUEndTime - GPUStartTime`) và một cặp event trên CUDA và HIP; mỗi con số là kết quả tốt nhất của
vài lần chạy có tính giờ sau một lần không tính giờ.

Các hàng `qstep`: `format_layers`, `format_head`, `batch`, `seconds`, `tokens_per_s`
(`batch / seconds`), `weight_bytes` (E một lần cho head, mọi W_l, các bias: những gì một bước
phải đọc), `weight_gb_per_s`, `speedup_vs_f32` (thời gian của bước `(F32, F32)` ở cùng batch
chia cho thời gian của bước này), `logits_rel_rms` (sau bước đầu tiên, `sqrt(sum (l - l_f32)^2 / sum
l_f32^2)` trên cả batch), `agree_steps` (số bước greedy, tính từ bước đầu, trong đó
mọi chuỗi chọn cùng token với bước float32; 32 nghĩa là không có bước nào khác), và kích thước.
Trang labs thêm mức sử dụng băng thông so với phép đo lab 5.1 của bạn. Đây là các
phép đo trên máy của bạn, không phải giá trị kỳ vọng.

## Tính khả chuyển

- HIP: `kernel.cu` của bạn chỉ dùng luồng, khối và các lần khởi chạy qua runtime, những thứ build được qua
  `pmpp/vendors/hip.h` không cần thay đổi; các hệ số tỉ lệ FP16 được chuyển từ bit của chúng bằng
  `halfToFloat`, nên không cần header `__half` nào (các header `cuda_fp16.h` và `hip_fp16.h`
  khác nhau, infra B12). Các GEMV bạn gọi dùng một lane mask có kiểu và độ rộng warp của thiết bị.
  Không có gì ở đây đã được biên dịch hay chạy trên phần cứng AMD: unverified.
- CUDA: unverified cho đến khi một `reports/verify-*.md` từ một máy CUDA cho thấy lab này qua.
- Metal: `half` là một kiểu lõi của MSL, nên các kernel đọc một hệ số tỉ lệ bằng `as_type<half>` (C5). Một
  khối Q4_0 cũng là 18 byte với căn chỉnh 2 trên Metal (C5).
