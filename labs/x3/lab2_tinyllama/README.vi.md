> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab X3.2: một mô hình kiểu Llama từ đầu đến cuối (mốc dự án M10)

Không có trong sách: học phần X3 là của khóa học này. Mô hình là decoder trong `run.c` của llama2.c, tức
kiến trúc Llama của Meta trong khoảng 130 dòng C; định dạng tệp là của llama2.c; báo cáo
sai số là của llama.cpp. Một mục trần như (D1) bên dưới nằm trong `notes/today/x3-facts.md`.

Mọi kernel của một bước decode Llama giờ đã có trong khóa học này: phép gather embedding (M8), lớp
tuyến tính (M2, M3, phép gather của lab 22.1), RMSNorm, RoPE và SwiGLU (lab X3.1), KV cache và
attention cho decode (M7), phép cộng residual (M0), argmax (M4) và bộ lấy mẫu (M5). Lab này
ghép chúng thành một bước decode greedy, chạy một mô hình mà một script trong repository ghi ra
từ một seed cố định, và kiểm tra mọi logit và mọi token so với một bản tham chiếu CPU. Bạn không viết
kernel số học nào. Bạn quyết định thứ tự các thành phần, tham số của chúng, và nơi mỗi
kết quả nằm, đó là phần lớn những gì tệp mô hình của một framework làm.

## Phần của bạn

(a) `advancePositionsKernel` và `launchAdvancePositions` (`advance_positions` và
`encodeAdvancePositions` trên Metal): một luồng cho mỗi chuỗi, chạy ở cuối mỗi bước.

```
out[pos[b] * batch + b] = tokens[b]                       record this step's argmax
if (pos[b] + 1 < promptLen) tokens[b] = prompt[(pos[b] + 1) * batch + b]   still in the prompt
pos[b] += 1;  len[b] += 1;  rngStep[b] += 1
```

Prompt được đưa vào mỗi bước một token qua chính bước đó, như llama2.c làm (D1): khi một
chuỗi còn ở trong prompt của nó, token tiếp theo của prompt thay thế argmax. Kernel này là
thứ cho phép toàn bộ vòng lặp sinh token ở lại trên thiết bị. Trước khi viết nó, hãy nói
host sẽ phải làm gì giữa hai bước nếu không có nó.

(b) `launchLlamaStep(m, s)` (`encodeLlamaStep` trên Metal): một bước cho `s.batch` chuỗi,
một lần khởi chạy cho mỗi phép toán cho cả batch, không có gì chờ GPU hay sao chép về
host. Thứ tự là của bản tham chiếu, và mô hình bước của ứng dụng (`sim/llamastep.ts`) cho thấy nó:

| Phép toán | Thành phần | Kích thước |
|---|---|---|
| `x = E[tokens]` | lab 22.2 (M8) `launchEmbed` | `[batch x d]` |
| với mỗi lớp: `xb = rmsnorm(x) * rmsAtt` | lab X3.1 `launchRmsNorm` | `[batch x d]` |
| `q, k, v = xb wq^T, xb wk^T, xb wv^T` | `m.linear`, ba lời gọi | `[batch x dq]`, `[batch x dkv]` hai lần |
| xoay `q` (h_q head) và `k` (h_kv head) tại `pos` | lab X3.1 `launchRope`, cách ghép cặp `m.ropeMode` | tại chỗ |
| nối `k`, `v` vào cache của lớp l tại `pos` | lab 20.4 (M7) `launchKvAppend` | `[batch x seqLen x h_kv x 128]` |
| `att = softmax(q K^T / sqrt(128)) V` trên `len` hàng | `m.attention`: kernel của lab 20.4, hoặc kernel nhóm của lab 20.5 | `[batch x dq]` |
| `x += att wo^T` | `m.linear`, rồi lab 2.3 (M0) `launchAddResidual` | |
| `xb = rmsnorm(x) * rmsFfn` | lab X3.1 `launchRmsNorm` | |
| `x += (silu(xb w1^T) * (xb w3^T)) w2^T` | `m.linear` hai lần, lab X3.1 `launchSwiGLU`, `m.linear`, M0 | `[batch x hidden]` |
| `logits = (rmsnorm(x) * rmsFinal) E^T` | lab X3.1, rồi `m.linear` với E: LM head được gắn liền | `[batch x vocab]` |
| `tokens = argmax(logits)` | lab 10.3 (M4) `launchArgmaxRows`, hoặc `s.sample` (M5) khi chương trình điều khiển đặt nó | `[batch]` |
| ghi lại, đưa prompt vào, tiến lên | của bạn, `launchAdvancePositions` | |

Ở đây `dq = h_q * 128 = d` và `dkv = h_kv * 128`. `llama.h` khai báo mọi thành phần với
thứ tự tham số của nó và cho biết kích thước của mọi buffer. Hai câu hỏi cần giải quyết trước khi chạy:
attention phải thấy `len` nào để nó đọc hàng bạn vừa nối vào và không đọc gì
sau đó, và các token tiếp theo phải đặt ở đâu để bước kế tiếp bắt đầu trên thiết bị?

## Của khóa học này, không phải của sách

- **Mô hình.** `forward` của llama2.c (D1): RMSNorm với epsilon `1e-5`, attention grouped-query
  với RoPE cơ số 10000, một khối feed-forward có cổng SiLU, bộ phân loại gắn liền với
  embedding. Kích thước head là 128 vì đó là kích thước mà các kernel decode của M7 được biên dịch cho
  (`HEAD_DIM 128`); một mô hình có kích thước head khác cần một kernel decode nhận kích thước head làm
  tham số, và bộ nạp từ chối nó (E2).
- **Tệp.** `provided/make_model.py` ghi tệp `.bin` phiên bản 0 của llama2.c (D1): một header 28 byte
  gồm bảy `int32`, rồi các trọng số float32 theo thứ tự của llama2.c, hai bảng RoPE mà
  llama2.c bỏ qua được ghi thành số không, và một `vocab_size` dương, nghĩa là bộ phân loại chính là
  embedding. Nó chỉ dùng thư viện chuẩn của Python, với bộ sinh splitmix64 riêng
  (không phải module `random`) và `struct.pack('<f')`, nên mọi máy đều ghi ra cùng các byte.
  CMake chạy nó lúc configure vào thư mục build. Không có gì được tải xuống bởi CMake, script,
  chương trình điều khiển hay các test.
- **Các trọng số.** Các giá trị phân bố đều nhân với một hệ số tỉ lệ cho mỗi tensor (`--block-scale`, `--qk-scale`; các
  giá trị mặc định là của chúng tôi), được chọn để các token sinh ra thay đổi và các bước greedy của bản tham chiếu
  có khoảng cách lớn giữa hai logit đầu (xem các test). Chúng không phải một mô hình đã huấn luyện: các token
  không có nghĩa.
- **Một token mỗi bước.** Prompt đi qua bước decode từng token một, như llama2.c
  xử lý nó. M6 (flash attention cho prefill của lab 20.2) không được ghép vào: nó nhận cả
  prompt như một khối `[T x d]` theo bố cục riêng, không phải bố cục cache mà M7 đọc
  (`[batch x seqLen x h_kv x 128]`), và bản tham chiếu xử lý prompt từng token
  một. Một đường prefill sẽ cần M6 ghi vào cache, đó là một thay đổi đối với M6.
- **Không có vòng đi về host trong vòng lặp.** Argmax ghi các token tiếp theo đè lên những token mà bước
  đã đọc; phần tiến lên của bạn ghi chúng vào `out` (`[seqLen x batch]`) và đưa prompt vào. Chương trình
  điều khiển sao chép `out` về một lần, sau bước cuối cùng. Trên CUDA và HIP, các bước được khởi chạy
  liên tiếp trên stream mặc định, vốn chạy chúng theo thứ tự phát lệnh (ch02-facts A5, B3). Trên
  Metal, mỗi bước là một command buffer, và command buffer của mọi bước được commit vào một
  queue trước khi chương trình điều khiển chờ: một queue "maintains an ordered list of command buffers"
  (infra-facts C12), và bên trong một bước, compute encoder thông thường chạy các dispatch tuần tự trên
  các buffer được theo dõi xung đột (hazard-tracked) (ch02-facts C9).
- **Cách ghép cặp RoPE (B6).** `m.ropeMode` 0 xoay các cặp liên tiếp `(2i, 2i+1)` (llama2.c,
  `LLAMA_ROPE_TYPE_NORM` của llama.cpp); 1 xoay `(i, i + 64)` (`rotate_half` của Hugging Face).
  `make_model.py --permute-qk` ghi cùng các trọng số với phép hoán vị hàng của llama.cpp trên `wq`
  và `wk`, phép mà trình chuyển đổi của nó áp dụng cho các checkpoint Hugging Face: mô hình đã hoán vị ở
  mode 0 tính đúng những gì mô hình gốc tính ở mode 1.
- **M9 trong M10.** Lớp gián tiếp `LinearFn` là nơi một lớp tuyến tính đã lượng tử hóa cắm vào. `launchQLinear(X, const QMatrix& W, Y, batch)`
  của lab X2.4 (M9) nhận một bộ mô tả ma trận đã lượng tử hóa,
  không phải `(X, const float* W, Y, batch, in, out)` của `LinearFn`, nên một bộ chuyển đổi được cung cấp
  nối chúng lại (`provided/m9_linear.h`, `.cu`, `.mm`). Chương trình điều khiển lượng tử hóa bảy phép chiếu
  của mọi lớp và embedding gắn liền bằng các quy tắc `_ref` của llama.cpp, như chương trình điều khiển của lab X2.4
  làm (x2-facts D1), giữ các giá trị đã giải lượng tử hóa trong các buffer float của mô hình (embedding
  và bản tham chiếu CPU đọc chúng), và gắn mỗi buffer float với bản sao đã lượng tử hóa của nó. Bộ
  chuyển đổi là một `LinearFn` tra cứu `W` được truyền cho nó và gọi `launchQLinear` của M9 với
  bản sao đã lượng tử hóa; nó không bao giờ đọc `W` dạng float, và một con trỏ nó không được giao là một lỗi
  mà chương trình điều khiển báo cáo. Giao diện `llama.h` của bạn không thay đổi: cùng một bước chạy float32 hoặc
  đã lượng tử hóa. Các trọng số RMSNorm vẫn là float32 (là vector, không phải ma trận mà lớp tuyến tính đọc).
  Nó cần lời giải của lab X2.4 và lab 22.1 (M9 gửi một ma trận float32 tới phép gather) và
  dùng các GEMV của lab X2.1 và X2.2, hoặc các bản thay thế trong `labs/x2/shared-g12`; lần chạy in ra
  cái nào. Các GEMV của M9 nhận batch tối đa 8 và `in` là bội số của 32.

## Các tệp

- `work/kernel.cu` (CUDA, HIP, các trình giả lập): `advancePositionsKernel`, `launchAdvancePositions`,
  `launchLlamaStep`.
- `work/kernel.metal`: `advance_positions`. `work/launch.mm`: `encodeAdvancePositions`,
  `encodeLlamaStep`.
- Được cung cấp, không sửa: `llama.h` (`LlamaConfig` theo thứ tự header của llama2.c, `LlamaModel` với
  con trỏ thiết bị cho mỗi lớp, `LlamaState`, `LinearFn`, `AttentionFn`, `SampleFn` và các bản tương ứng
  trên Metal, khai báo của mọi thành phần); `provided/pieces.cmake`, tệp sao chép các
  lời giải tham chiếu của lab 2.3, 6.4, 10.3, 20.4, 22.2 và X3.1 (bắt buộc) và 15.6, 20.5,
  22.1, X1.3 và X2.4 (nếu có) vào thư mục build (X2.4 cùng với các GEMV của lab X2.1 và X2.2,
  hoặc các bản thay thế của chúng), đổi tên những gì sẽ xung đột, như tệp của lab
  22.2 làm (một thành phần bắt buộc bị thiếu, hoặc có chữ ký khác với của `llama.h`,
  sẽ khiến lab bị bỏ qua kèm một ghi chú nêu tên nó); `provided/m9_linear.h`, `.cu` và `.mm`, bộ
  chuyển đổi M9; `provided/make_model.py`; `provided/step_timer.cu`,
  cặp event đo giờ một bước trên CUDA và HIP (của lab 22.2).
- `main.cpp`: bộ nạp (với kiểm tra kích thước E1), bản tham chiếu CPU (`RefSeq`: forward của llama2.c
  bằng double, greedy chọn chỉ số nhỏ nhất khi hòa, D9), các test và benchmark.
  Chương trình điều khiển biên dịch kernel Metal của mỗi thành phần từ chính `kernel.metal` của lab đó, với
  các hằng số mà chương trình điều khiển của lab đó dùng.

## Các lệnh

- Kiểm thử: `python3 labs/pmpp.py run x3.2 --backend emu` trước, rồi `python3 labs/pmpp.py run
  x3.2` trên GPU của bạn. Trình giả lập chạy mọi luồng của mọi lần khởi chạy của mọi bước dưới ba
  thứ tự luồng: hãy chờ vài phút, không phải vài giây.
- Đo: `python3 labs/pmpp.py run x3.2 --bench`, sau các dự đoán bên dưới.
- Một seed khác: `PMPP_X3_SEED=5 python3 labs/pmpp.py run x3.2` ghi lại các mô hình test lúc
  configure với seed 5 (và lần chạy tiếp theo không có nó sẽ ghi lại các mô hình được cung cấp).

## Các test làm gì

Kích thước, do chương trình điều khiển in ra: trên GPU d = 512 (h_q 4 trên h_kv 2), hidden 1376, 4 lớp,
vocab 4099, seqLen 128, một prompt 4 token và 60 token được sinh (63 bước), batch 1 và 3;
trên các trình giả lập d = 256 (h_q 2 trên h_kv 1), hidden 352, 2 lớp, vocab 300, seqLen 16, một
prompt 3 token và 5 token được sinh (7 bước), batch 1 và 2. Các kích thước của trình giả lập đều
khác nhau (d, dkv = 128, hidden, vocab), nên một kích thước truyền vào sai tham số
cũng thất bại ở đó; d không thể nhỏ hơn mà không mất attention grouped-query ở kích thước head 128, nên
lần chạy trên trình giả lập mất vài phút. Mọi
prompt bắt đầu bằng token 1, BOS của llama2.c (D1). Lớp tuyến tính là phép gather của lab 22.1 trên
GPU và M3 của lab 15.6 trên các trình giả lập, với các ô rộng 128 khởi chạy ít luồng nhất và vì thế
tốn ít nhất cho trình giả lập (lab 22.2 cũng chọn như vậy); M2 nếu không có cái nào.

- **Các tệp mô hình.** Kích thước tệp bằng công thức E1 cho header của nó; checksum của
  256 trọng số đầu và 256 trọng số cuối (FNV-1a 64 trên các float đầu của embedding và các float cuối của
  `rms_final_weight`) bằng giá trị mà script in ra khi CMake chạy nó và giá trị
  ghi trong `provided/pieces.cmake` cho seed được cung cấp (một bộ sinh ghi ra các byte khác
  trên một máy khác sẽ thất bại ở đây); tệp `--permute-qk` chứa đúng phép hoán vị
  `wq` và `wk` của llama.cpp và không có gì khác biệt nữa.
- **Riêng từng thành phần**, trước khi mã của bạn chạy: mỗi thành phần được cung cấp chạy một lần so với một bản tham chiếu CPU
  (embedding từng bit một, RMSNorm, mọi lớp tuyến tính có mặt, RoPE ở cả hai cách ghép cặp tại
  các vị trí 3 và seqLen - 1, phép nối KV, mỗi kernel attention trên một cache mà các hàng chưa đọc
  là NaN, phép cộng residual từng bit một, SwiGLU, argmax với một trường hợp hòa được cài sẵn). Một thất bại
  ở đây là của khóa học, không phải của bạn. Rồi đến riêng phần tiến lên của bạn, tại các vị trí 0, 2 và 5 của một
  prompt 4 token.
- **Logits với teacher forcing.** Mọi bước được đưa vào token của bản tham chiếu (cả chuỗi
  tham chiếu là prompt), nên logits của mọi bước đều có thể được so sánh. Ở mỗi bước, chương trình điều khiển
  in ra NMSE, `sum (got - ref)^2 / sum ref^2` (thước đo của llama.cpp, D7), sai số tuyệt đối
  lớn nhất, và sai số lớn nhất so với dung sai của nó. Một logit qua nếu
  `|got - ref| <= 4e-5 * (1 + mag)`, trong đó `mag = sum_j |xn_j E_vj|` là độ lớn của các số hạng
  trong tích vô hướng của LM head (`xn` là trạng thái ẩn đã chuẩn hóa cuối cùng của bản tham chiếu). Cùng
  dung sai đó áp dụng trên mọi backend; nó được chọn để chế độ toán học mặc định của Metal qua được
  với khoảng dư: trên máy mà lab này được viết, sai số tệ nhất bằng 0.06 dung sai với GEMM của lab
  6.4 hoặc lab 15.6 và 0.02 dung sai với phép gather (Metal), và khoảng 0.013 dung sai trên
  trình giả lập. Đây là các phép đo trên một máy; CUDA và HIP chưa được chạy. Bước cũng phải ghi lại
  argmax của chính nó vào `out`, đưa vào token prompt tiếp theo, và để `pos` và `len` ở step + 1 và
  step + 2.
- **Greedy chạy tự do.** Prompt thật, rồi tất cả các bước mà không có việc gì của host ở giữa,
  và một lần sao chép `out` ở cuối. Các token phải bằng chính xác của bản tham chiếu ở mọi
  bước mà khoảng cách giữa hai logit đầu của bản tham chiếu vượt quá hai lần dung sai logit (tổng dung sai
  của hai logit). Với các seed được cung cấp, mọi bước được test đều đạt, theo một hệ số mà
  chương trình điều khiển in ra. Với một seed khác, một bước có khoảng cách nhỏ bên trong prompt được bỏ qua riêng (bước
  kế tiếp bị ép), và bước đầu tiên như vậy trong phần sinh khiến phần còn lại thành "inconclusive from
  step k", được in ra, không bao giờ âm thầm cho qua: một lần đảo ở đó thay đổi mọi token sau đó.
- **Tính bất biến theo batch.** Các token của mỗi chuỗi trong batch 2 (3 trên GPU) bằng các token của nó khi nó
  được decode một mình.
- **Các kernel khác** (chỉ GPU): mọi lớp tuyến tính và kernel attention khác có mặt,
  với teacher forcing ở batch 1. Trình giả lập sẽ mất vài phút cho mỗi kernel, và các phép kiểm tra
  riêng từng thành phần đã chạy từng cái.
- **M9 trong M10**, nếu có lời giải của lab X2.4: với Q8_0 và với Q4_0 (mọi phép chiếu và
  head ở định dạng đó), riêng lớp của M9 qua bộ chuyển đổi trên `wq` và `w2` so với
  CPU trên các trọng số đã giải lượng tử hóa; rồi bước của bạn với bộ chuyển đổi làm lớp tuyến tính,
  chạy tự do ở batch 2 qua prompt và hai token được sinh trên các trình giả lập (mỗi
  bước giả lập đều chậm; lần chạy phải vừa giới hạn thời gian của `pmpp.py`), ở batch 3 qua cả 63 bước
  trên GPU, với các token phải bằng của lần chạy tham chiếu CPU
  trên cùng các trọng số đã lượng tử hóa, theo cùng quy tắc khoảng cách như trên; và trên GPU, còn có mọi
  logit của mọi bước với teacher forcing. Các mã đã lượng tử hóa, hệ số tỉ lệ và bản sao float phải
  không thay đổi sau đó. Với seed GPU được cung cấp, một chuỗi Q4_0 có một bước mà khoảng cách
  thấp hơn mức của quy tắc (được in là inconclusive từ bước đó trở đi). Lần chạy cũng in ra nơi mà các
  token greedy của mỗi chuỗi lần đầu khác với token của mô hình float32: không phải một test, vì mô hình là
  ngẫu nhiên và một token thay đổi sẽ thay đổi mọi token theo sau.
- **Cách ghép cặp RoPE.** Bản tham chiếu CPU của mô hình đã hoán vị ở mode 0 bằng của mô hình gốc ở
  mode 1 tới 1e-9; rồi mô hình gốc ở mode 1 và mô hình đã hoán vị ở mode 0, mỗi cái
  với teacher forcing (điều này cũng kiểm tra token của mọi bước) và, trên GPU, chạy tự do, so với
  bản tham chiếu mode 1.
- **Bộ lấy mẫu (M5)**, chỉ GPU, nếu có lời giải của lab X1.3: `s.sample` ở temperature 0
  cho chính xác các token greedy; ở temperature 0.8, top-k 40, top-p 0.95, min-p 0.05, với
  một seed cố định và một stream cho mỗi chuỗi, hai lần chạy cho các token giống hệt nhau. Các token được lấy mẫu
  không được so sánh giữa các backend ở đây (lab X1.3 là nơi điều đó được kiểm tra).
- **Giá trị độc và dải bảo vệ.** Các activation, logits và mọi hàng cache bắt đầu là NaN, nên một
  giá trị không ai ghi, hay một hàng cache mà attention không nên đọc, sẽ biến logits thành NaN.
  Mọi đầu ra có theo sau một dải bảo vệ phải còn nguyên; sau mỗi lần chạy, các hàng cache
  dưới số bước đã được ghi và phần còn lại vẫn là NaN; các trọng số, prompt và các bảng
  cos/sin không thay đổi. Trên `emu`, các trọng số và đầu vào được cấp phát đúng kích thước của chúng,
  nên một lần đọc vượt quá một mảng là một lỗi AddressSanitizer chỉ ra đúng dòng.
- Không chạy trên backend `race`: mã của bạn khởi chạy các thành phần và không ghi bộ nhớ chia sẻ (shared memory) nào, và các
  thành phần đã được kiểm tra race trong lab của riêng chúng.

## Trước khi benchmark, hãy dự đoán

Hãy ghi lại hai dự đoán (bài học hỏi chúng dưới tên `x3/labs/p-lab2-bwfraction` và
`x3/labs/p-lab2-attn-share`):

1. Số token mỗi giây ở batch 1, vị trí 16, tính theo tỉ lệ so với trần: băng thông của lab 5.1
   chia cho số byte trọng số mỗi token (benchmark in ra cả hai).
2. Tỉ phần của attention trong thời gian bước ở vị trí 512, biết tỉ phần bạn đo được ở 16.

Những điều cần cân nhắc:

- **Lựa chọn GEMV.** Ở batch 1, mọi phép chiếu là một phép nhân ma trận với vector. Phép gather của lab 22.1
  đọc mỗi trọng số một lần mỗi bước; các ô của lab 6.4 được xây cho ma trận và khởi chạy một
  ô 16 hàng cho một hàng công việc. llama.cpp chuyển từ kernel ma trận-vector sang kernel ma trận-ma trận
  ở khoảng 8 token (ch22-facts D6). Benchmark chạy cả hai.
- **Số lần khởi chạy mỗi token.** Một bước là mười sáu lần khởi chạy cho mỗi lớp cộng thêm năm, mỗi lần đều ngắn ở
  batch 1. Mỗi lần tốn một lần khởi chạy trên CUDA và HIP và một dispatch trên Metal, và không lần nào di chuyển
  trọng số. Bảng chia theo từng thành phần cộng dồn các thành phần được đo giờ riêng; hãy so tổng của nó với bước.
- **KV cache tăng; trọng số thì không.** Số trọng số đọc mỗi token là cố định.
  Attention đọc `2 x layers x batch x len x dkv x 4` byte, tăng theo vị trí:
  ở vị trí 512 (len 513) nó khoảng 30 lần so với ở 16 (len 17). Điều đó có quan trọng hay không phụ thuộc vào việc nó so ra sao với
  số byte trọng số, mà benchmark in ra cạnh nhau.
- **Bộ nhớ đệm (cache).** Trọng số của một mô hình nhỏ có thể vừa một phần trong bộ nhớ đệm của thiết bị, điều sẽ đẩy
  băng thông trọng số lên trên của lab 5.1; Apple không công bố kích thước bộ nhớ đệm nào (infra-facts C13).

## Benchmark đo gì

Chỉ trên thiết bị thật. Mô hình được ghi bởi cùng script đó lúc bench (`--fill tile`: mọi
tensor lặp lại một khối giá trị ngẫu nhiên, nên một tệp lớn được ghi trong vài giây; các con số của nó
không có nghĩa gì): d = 2048 với 16 head query trên 4 head KV, hidden 5632, 8 lớp, vocab
32000, seqLen 1024 (kích thước của chúng tôi). Nếu trọng số chiếm hơn một phần tư bộ nhớ
mà thiết bị báo cáo, d, số head và độ rộng hidden giảm một nửa cho đến khi vừa; kích thước và
số byte trọng số được in ra. Một bước là một command buffer trên Metal (`GPUEndTime -
GPUStartTime`) và một cặp event trên CUDA và HIP (`provided/step_timer.cu`). Mỗi con số
là kết quả tốt nhất của vài lần chạy có tính giờ sau một lần không tính giờ.

- Các hàng `step`, cho mỗi lớp tuyến tính có mặt, ở batch 1 và 4 và các vị trí 16 và 512 (các
  hàng cache trước vị trí đó chứa giá trị hữu hạn, như sau một prompt; giá trị của chúng không
  thay đổi thời gian, và mỗi bước có tính giờ được đặt trở lại vị trí của nó): ms mỗi bước, số token mỗi
  giây (`batch / seconds`), số byte trọng số mỗi token (`weight bytes / batch`), GB/s trọng số, và
  số byte KV cache mà attention đọc.
- Các hàng `piece`, với phép gather (hoặc M2 khi không có lab 22.1): mọi thành phần được đo giờ riêng ở cùng
  các batch và vị trí, nhân với số lần một bước chạy nó, cộng thành bốn nhóm: các phép nhân ma trận,
  attention (nối và attention), norm + RoPE + SwiGLU + residual, embed + LM head + argmax
  + tiến lên; cùng với tỉ phần của attention trong tổng của chúng.
- Một hàng `ceiling`: băng thông của lab 5.1, đọc từ `reports/runs/<backend>.json` khi bạn đã
  chạy benchmark của lab 5.1, chia cho số byte trọng số mỗi token, và số token mỗi
  giây đo được ở batch 1, vị trí 16 tính theo tỉ lệ so với nó.

Các hàng là phép đo trên máy của bạn, không phải giá trị kỳ vọng.

## Một checkpoint thật do bạn tự tải về

Pipeline đọc định dạng của llama2.c, nên checkpoint thật đầu tiên nên thử là một trong các
tinyllamas của Karpathy (MIT, D2): `stories260K.bin` (1,056,540 byte) hoặc `stories15M.bin`. Phép kiểm tra E1 của bộ nạp
tái tạo được cả hai kích thước từ header của chúng. Chúng sẽ không chạy được nguyên trạng: kích thước head
của chúng là 8 và 48, còn các kernel decode của M7 được biên dịch cho 128 (E2), nên trước hết attention phải
nhận kích thước head làm tham số (và hằng số `LLAMA_HEAD_DIM` trong `llama.h`
cũng vậy). Với thay đổi đó, `run.c` với `-t 0.0` là bản tham chiếu để so sánh token.

Một checkpoint Llama của Hugging Face như SmolLM2-135M (Apache-2.0) cần thêm bốn phép chuyển đổi
(D4): bfloat16 sang float32 (chính xác: 16 bit được lưu trở thành nửa trên của một float), cách ghép cặp
RoPE chia đôi (mode 1, hoặc hoán vị `wq` và `wk` như `--permute-qk` làm), 9 head query
trên 3 head KV với kích thước head 64, và RoPE cơ số 100000 và epsilon 1e-5 từ
`config.json` của nó. Đừng dùng `export.py` của llama2.c cho nó: theo cách chúng tôi đọc, phép hoán vị
`k_proj` của nó giả định số head KV bằng số head query (D6). GGUF là định dạng llama.cpp nạp, nhưng
nó thêm metadata có kiểu, phần đệm căn chỉnh và thường là các tensor đã lượng tử hóa (D5).

Không phần nào ở đây được chạy khi lab được viết: đó là một mô tả công việc, unverified.

## Tính khả chuyển

- HIP: `kernel.cu` của bạn chỉ dùng luồng, khối và các lần khởi chạy qua runtime, những thứ build được qua
  `pmpp/vendors/hip.h` không cần thay đổi. Các thành phần có warp shuffle (lab 10.3, lab 20.4 và 20.5,
  lab 22.1, lab X3.1) dùng một lane mask có kiểu và hoặc `warpSize` hoặc một warp logic 32 lane.
  Không có gì ở đây đã được biên dịch hay chạy trên phần cứng AMD: unverified.
- CUDA: unverified cho đến khi một `reports/verify-*.md` từ một máy CUDA cho thấy lab này qua.
- Metal: các kernel decode của lab 20.4 và 20.5 cần SIMD group 32 lane; chương trình điều khiển hỏi
  pipeline (`threadExecutionWidth`) và làm thất bại phép kiểm tra riêng từng thành phần, kèm độ rộng nó tìm thấy,
  nếu pipeline báo một độ rộng khác. Apple không công bố độ rộng cố định nào (ch20-facts C3).
