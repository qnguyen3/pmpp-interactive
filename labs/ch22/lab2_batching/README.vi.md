> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 22.2: nghiên cứu batching (mốc dự án M8)

Sách: mục 22.5 (p. 539) cho batching, mục 22.2 (pp. 532-533) cho cách phân rã của
embedding, mục 22.3 (pp. 537-538) cho lý do vòng lặp ở lại trên device. Chương này
không in mã nào: mọi kernel và hàm host ở đây là của khóa học này. Một mục đứng riêng như (E1) bên dưới
nằm trong `notes/today/ch22-facts.md`.

Mục 22.5 nói rằng gộp batch các truy vấn của nhiều người dùng biến các tích vector-ma trận của mỗi phép chiếu
thành một tích ma trận dùng lại trọng số: mỗi truy vấn chờ
lâu hơn, cả nhóm xong sớm hơn. Lab này dựng một bước decode greedy từ chính các
thành phần của dự án, cho `batch` chuỗi cùng lúc, và đo xem sớm hơn bao nhiêu trên thiết bị
của bạn, và nó ngừng có lợi ở đâu.

## Phần của bạn

(a) `launchEmbed(E, tokens, H, batch, d)`: `H[r][j] = E[tokens[r]][j]`, mỗi phần tử của H
một luồng (thread). Lấy đầu ra làm trung tâm (mục 22.2): luồng sở hữu `H[i]` gom về đúng một
float mà nó cần. Hai chuỗi ở cùng một token đọc cùng một hàng, và các lần đọc không bao giờ
xung đột. Các id token ở lại trong bộ nhớ device, nơi argmax của bước trước đã ghi
chúng. Trước khi viết, hãy hỏi phiên bản lấy đầu vào làm trung tâm sẽ thế nào (mỗi hàng
của E một luồng) và tiêu chí nào của sách (p. 533) loại bỏ nó.

(b) `launchDecodeStep(m, s)`: một bước decode cho `s.batch` chuỗi, mỗi thao tác một lần khởi chạy
cho cả batch, không có gì phải chờ bên trong bước:

| Thao tác | Thành phần | Hình dạng |
|---|---|---|
| `H = E[tokens]` | của bạn, `launchEmbed` | `[batch x d]` |
| với mỗi lớp: `T = H W_l^T` | `m.linear`: M2, M3 hoặc phép gather của lab 22.1, do chương trình điều khiển (driver) chọn | `[batch x d]` |
| `T += b_l` | M0 `launchAddBias`, tại chỗ | |
| `H += T` | M0 `launchAddResidual` | |
| `logits = H E^T` | lại là `m.linear`: trọng số của LM head là E | `[batch x vocab]` |
| `probs = softmax(logits / temperature)` | M4 `launchSoftmaxRows`, hệ số `1 / s.temperature` | `[batch x vocab]` |
| `tokens = argmax(logits)` | M4 `launchArgmaxRows`, trên các token mà bước đã đọc | `[batch]` |

`decode.h` khai báo mọi thành phần cùng thứ tự đối số của nó. Trên Metal, cùng bước đó là
`embed_rows` trong `kernel.metal`, và `encodeEmbed` cùng `encodeDecodeStep` trong `launch.mm`,
chúng đặt mọi dispatch vào compute encoder của bên gọi; cả bước là một command
buffer.

## Của khóa học này, không phải của sách

- **Mô hình.** Một decoder được cắt gọn còn những gì batching làm thay đổi: embedding, `layers` khối
  gồm lớp tuyến tính, bias và residual, LM head, bộ lấy mẫu. Các hình dạng là của chúng tôi. Embedding được
  buộc với LM head (`Wout = E`), giúp tiết kiệm một ma trận `vocab x d`; E vốn đã được lưu
  `(out, in)` cho LM head, nên không cần chuyển vị.
- **Không có attention.** Chương 20 dựng nó (M6, M7). Nó bị bỏ ra vì nó không gộp batch được:
  mỗi chuỗi có keys và values riêng, nên không có trọng số chung nào để batch
  dùng lại (A6, mục này cũng giải thích cách diễn đạt lỏng hơn ở p. 539). Trong một bước thật, nó thuộc
  về nhóm các phần có chi phí tăng theo batch ngay từ đầu, cạnh các thành phần theo từng phần tử và
  softmax.
- **Xác suất.** Decode greedy chỉ đọc argmax. Bước này vẫn tính `probs`,
  vì bộ lấy mẫu của mốc dự án M5 đọc chúng, nên chi phí của chúng thuộc về
  phép đo.
- **Token tại chỗ.** Argmax ghi các token tiếp theo đè lên các token mà bước đã đọc, nên
  bước tiếp theo bắt đầu trên device mà không cần chép về host rồi chép ngược lại. Ý chính của mục 22.3 là
  bất cứ thứ gì ở lại trên host đều giới hạn mức tăng tốc; một chuyến khứ hồi về host cho mỗi token là một
  phần còn lại như vậy (việc ánh xạ sang decode là của chúng tôi).
- **Thứ tự tuần tự.** Mỗi lần khởi chạy thấy những gì lần trước nó đã ghi: các lần khởi chạy CUDA và HIP
  không có stream chạy theo thứ tự phát lệnh (ch02-facts A5, B3). Trên Metal, compute encoder thường
  "dispatches its compute commands serially" và các buffer do device tạo được
  theo dõi xung đột (ch02-facts C9). Trong một encoder đồng thời, bước này sẽ cần một rào chắn
  bộ nhớ sau mỗi dispatch mà dispatch tiếp theo đọc (C1).

## Các tệp

- `work/kernel.cu` (CUDA, HIP, các trình giả lập): `embedGatherKernel`, `launchEmbed`,
  `launchDecodeStep`.
- `work/kernel.metal`: `embed_rows`. `work/launch.mm`: `encodeEmbed`, `encodeDecodeStep`.
- Được cung cấp, không sửa: `decode.h` (`DecodeModel`, `DecodeState`, `LinearFn` và phiên bản
  Metal tương ứng `EncodeLinearFn`, các khai báo của các thành phần); `provided/pieces.cmake`, tệp này
  chép các lời giải tham chiếu của lab 2.3 (M0), lab 6.4 (M2, đổi tên thành `launchLinearM2`,
  `encodeLinearM2`), lab 10.3 (M4), lab 15.6 (M3, đổi tên thành `launchLinearM3`,
  `encodeLinearM3`, nếu có) và lab 22.1 (`launchGemvGather`, `encodeGemvGather`, nếu
  có) vào thư mục dựng; `provided/step_timer.cu`, cặp event đo thời gian một
  bước trên CUDA và HIP. Chương trình điều khiển biên dịch kernel Metal của mỗi thành phần từ chính
  `kernel.metal` của lab đó, với các hằng số mà chương trình điều khiển của lab đó dùng (TILE_WIDTH 16 cho lab 6.4;
  SIMD_WIDTH từ `threadExecutionWidth` của pipeline cho lab 15.6; các phép kiểm tra của lab 10.3 cho
  256 luồng trong số nguyên lần SIMD-group).
- `main.cpp`: các phép kiểm thử, benchmark, và bản tham chiếu CPU `cpuDecodeStep`.

## Lệnh

- Kiểm thử: `python3 labs/pmpp.py run 22.2 --backend emu` trước, rồi `python3 labs/pmpp.py
  run 22.2` trên GPU của bạn. Lần chạy trên trình giả lập (emulator) chậm: nó chạy
  mọi luồng của mọi lần khởi chạy của mọi bước, theo ba thứ tự luồng, cho mọi GEMM,
  và các ô (tile) rộng 16 của M2 khởi chạy nhiều luồng hơn hẳn so với M3.
- Đo: `python3 labs/pmpp.py run 22.2 --bench`.

## Các phép kiểm thử làm gì

- Mô hình: trên các trình giả lập d = 48, vocab = 301, 2 lớp, batch 1, 2, 3 và 5, bốn
  bước; trên GPU d = 256, vocab = 4099, 2 lớp, batch 1, 2, 3, 8, 17 và 64, sáu
  bước. Mỗi batch chạy với mọi GEMM có mặt: m2 luôn luôn, m3 khi lời giải của lab 15.6
  tồn tại, phép gather (batch 8 trở xuống) khi lời giải của lab 22.1 tồn tại.
- `launchEmbed` riêng lẻ, trên các batch chứa token 0, token `vocab - 1` và một token được dùng
  hai lần: mọi hàng của H phải giống từng bit với hàng của E, các token không bị đụng tới.
- Sau mỗi bước, với mỗi chuỗi: token bằng argmax của một bản tham chiếu CPU
  tính bằng double từ cùng các trọng số float; H và logits khớp với nó trong phạm vi
  `1e-4 * (1 + mag)`, trong đó `mag` là tổng giá trị tuyệt đối của mọi tích và
  bias góp vào phần tử đó; các xác suất khớp trong phạm vi dung sai của lab 10.3
  (`1e-20 + 1e-4 * p`) với softmax, tính bằng double, của logits mà bước của bạn đã tính.
- Mô hình kiểm thử được sinh sao cho ở mọi bước được kiểm thử, logit cao nhất của bản tham chiếu
  hơn logit hạng nhì nhiều hơn `1e-5 * (2 + mag)` của hai logit đó, lớn hơn nhiều so với mức
  làm tròn float32 làm lệch một logit; một seed không thỏa điều này sẽ được thay bằng seed kế tiếp. Trường hợp hòa
  sẽ thuộc về chỉ số nhỏ nhất (quy tắc của M4), nhưng biên độ đó loại trừ chúng.
- Tính bất biến theo batch, sau khi batch đó đã vượt qua: các token của mỗi chuỗi trong batch
  bằng các token của nó khi nó được decode một mình. Trên GPU, phép này chạy một lần cho mỗi GEMM, trên batch lớn nhất
  được kiểm thử. Trên các trình giả lập, nó chạy một lần, trên batch 3, với M3 (hoặc M2 nếu không có lab 15.6),
  để giữ lần chạy ngắn: tính bất biến phụ thuộc vào cách bước của bạn xử lý batch, không phụ thuộc vào
  GEMM nào nó gọi.
- Các trọng số không thay đổi sau lần chạy. H, T, logits, probs và tokens mỗi thứ đều có theo sau
  một dải canh phải còn nguyên, và các đầu ra bắt đầu là NaN, nên một giá trị không ai
  ghi không thể vượt qua. Các trọng số được cấp phát đúng kích thước, nên trên `emu`, một lần đọc vượt quá
  một trong số đó là một lỗi AddressSanitizer chỉ ra dòng đó.
- Không chạy trên backend `race`: mã của bạn không có cập nhật dùng chung nào (mỗi luồng ghi phần tử
  của riêng nó), và các thành phần đã được kiểm thử race trong lab riêng của chúng.

## Trước khi đo hiệu năng, hãy dự đoán

Lấy ngưỡng T từ lab 5.1 của bạn (trình khám phá của mục 22.5 hiển thị nó) và viết ra:

1. Điểm gãy với M3: batch đầu tiên, trong số 1, 2, 4, ..., có thời gian mỗi bước ít nhất
   1.5x thời gian ở batch 1. Dưới điểm đó, thêm chuỗi gần như được đi nhờ miễn phí.
2. Số token mỗi giây ở batch 64 chia cho số token mỗi giây ở batch 1, với M3.

Những điều cần cân nhắc:

- Mô hình đặt điểm gãy của một phép chiếu float32 gần 2T: cường độ tính toán của
  `[B x K] x [K x N]` vào khoảng B/2 FLOP/B khi B nhỏ so với K và N (E1). Benchmark
  in FLOP/B theo mô hình của bước ở mỗi batch; hãy so nó với T.
- Ô khối của M3 là 128 hàng của batch. Mọi batch từ 1 đến 128 khởi chạy cùng một
  lưới (grid) và làm cùng phép tính, trên các hàng điền 0, nên thời gian mỗi bước của nó đi ngang
  trên dải đó, dù bước có bị giới hạn bởi băng thông hay không (E3). Hãy so với các hàng của M2
  (ô 16) và, ở batch nhỏ, với của phép gather: llama.cpp chuyển từ kernel
  ma trận-vector sang ma trận-ma trận ở khoảng 8 token là có lý do (D6).
- Các thành phần theo từng phần tử và softmax tăng theo batch ngay từ đầu: cường độ tính toán của chúng
  không tăng theo B (E2). Các hàng 'piece' cho thấy khi nào chúng bắt đầu đáng kể.
- Bộ nhớ đệm (cache) có thể giữ một phần trọng số nếu bộ nhớ đệm của thiết bị bạn lớn; tài liệu của Apple không nêu kích thước
  bộ nhớ đệm nào (infra-facts C13). Điều đó sẽ đưa băng thông trọng số ở batch 1 lên trên mức lab 5.1
  đo được.
- Trên Metal, lab 5.1 là ngưỡng duy nhất bạn có: Apple không công bố FLOPS đỉnh hay
  băng thông (C13).

## Benchmark đo gì

Chỉ trên thiết bị thật. Mô hình là d = 4096, 2 lớp, vocab = 32000, float32, embedding
buộc với LM head (hình dạng của chúng tôi; khoảng 650 MB trọng số). Nếu trọng số chiếm nhiều
hơn một phần tư bộ nhớ mà thiết bị báo cáo, hoặc việc cấp phát thất bại, d giảm một nửa cho đến khi
vừa; hình dạng và số byte trọng số được in ra. Một bước là một command buffer trên
Metal (`GPUEndTime - GPUStartTime`) và một cặp event trên CUDA và HIP. Mỗi con số là
lần tốt nhất trong vài lần chạy được đo sau một lần không đo.

- Các hàng `step`, cho mỗi GEMM có mặt, ở batch 1, 2, 4, ... tới 1024 (phép gather tới
  8), dừng khi một bước mất hơn 2 s hoặc một lần cấp phát thất bại: số giây, số token mỗi
  giây (`batch / seconds`), số byte trọng số và GB/s trọng số (số byte trọng số chia số giây), và
  FLOP/B theo mô hình: số FLOP lý tưởng của bước chia số byte lý tưởng, mỗi mảng được đọc hoặc ghi
  một lần, `2 B K N` FLOP của các GEMM cộng một cho mỗi phần tử bias và residual (cách đếm của chúng tôi, E1,
  E2). Bảng cũng in số token mỗi giây của mỗi hàng chia cho hàng trước nó.
- Các hàng `one_by_one`, 1 đến 32 chuỗi: chừng ấy bước batch 1 riêng biệt, lần lượt
  nối tiếp nhau trong một command buffer hoặc một cặp event. Đây là "sum of all the
  vector-matrix multiplications" của sách (p. 539), đặt cạnh bước gộp batch cho cùng số
  chuỗi.
- Các hàng `piece` ở batch 1, 16, 256 và batch lớn nhất đo được: mỗi thành phần của bước
  (embed, lớp tuyến tính của một lớp, bias, residual, LM head, softmax, argmax) được đo thời gian riêng. Một
  bước chạy các thành phần lớp tuyến tính, bias và residual một lần cho mỗi lớp.

Các hàng là phép đo trên máy của bạn, không phải kỳ vọng. Ứng dụng tìm điểm gãy từ
các hàng `step`: batch đầu tiên có thời gian mỗi bước ít nhất 1.5x thời gian ở
batch nhỏ nhất đo được (một định nghĩa của chúng tôi; trình khám phá của mục 22.5 dùng cùng định nghĩa đó). `llama-batched-bench` đo cùng đại lượng này trên
một mô hình thật (D3).

## Tính khả chuyển

- HIP: `kernel.cu` của bạn chỉ dùng luồng, khối và lệnh khởi chạy runtime, những thứ dựng được qua
  `pmpp/vendors/hip.h` mà không cần thay đổi. Các thành phần dùng warp shuffle (M4, lab 22.1) dùng một
  lane mask có kiểu, vì HIP cần mask 64 bit, và lặp theo `warpSize`; M3 không có intrinsic
  warp nhưng bố trí các warp của nó theo độ rộng mà thiết bị báo cáo (32 hoặc 64). Không có gì
  ở đây từng được biên dịch hoặc chạy trên phần cứng AMD: unverified.
- CUDA: unverified cho đến khi một `reports/verify-*.md` từ một máy CUDA cho thấy lab này vượt qua.
- Metal: độ rộng SIMD-group của M3 đến từ `threadExecutionWidth` của pipeline, như ở lab
  15.6; tài liệu của Apple không nêu độ rộng cố định nào (ch15-facts C1).
