> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 20.1: mặt nạ nhân quả và softmax (mốc dự án M6, phần một)

Sách: mục 20.3 (pp. 486-488), kernel gộp mặt nạ và softmax của Fig. 20.4 (p. 487). Các phép
rút gọn (reduction) bên trong nó là của lab 10.3 (mốc dự án M4). Chiều batch, mặt nạ căn góc
dưới phải cho `nq < nk` và logsumexp `L` không có trên trang sách: chúng là của khóa học này, và
lý do nằm ở bên dưới. Một mục đứng riêng như (A1) hay (D5) nằm trong
`notes/today/ch20-facts.md`.

Hãy viết bước giữa của attention không gộp: cho các điểm số `S = Q K^T`, tạo ra
`P = softmax(scale * S + M)`, trong đó `M` là mặt nạ nhân quả, và logsumexp của mọi hàng.

```
launchCausalSoftmax(S, P, L, batch, nq, nk, scale)
```

| Tensor | Hình dạng |
|---|---|
| `S` (vào), `P` (ra) | `[batch, nq, nk]` |
| `L` (ra) | `[batch, nq]` |

`float` theo thứ tự hàng (row-major), batch đứng trước. `batch` đếm các cặp (chuỗi, head): một
phần tử cho mỗi head của mỗi chuỗi.

## Hợp đồng

- `1 <= nq <= nk`, `batch >= 1`, mọi điểm số đều hữu hạn.
- **Mặt nạ được căn theo góc dưới phải.** Hàng `r` thấy các cột `c <= r + (nk - nq)`. Với
  `nq == nk` đó là quy tắc của sách, cột không quá hàng (p. 484). Với `nq == 1` hàng duy nhất
  thấy mọi cột: một bước decode, vốn không cần mặt nạ (Fig. 20.6). FlashAttention (từ bản 2.1)
  và MLX căn mặt nạ theo cách này; `is_causal` của PyTorch căn nó theo góc trên trái khi
  `nq < nk`, điều sẽ khiến một truy vấn một token chỉ thấy key 0 (A8, D4, D5).
- Với `m` là giá trị `scale * S` lớn nhất trên các cột nhìn thấy được của một hàng và
  `s = sum of exp(scale * S - m)` trên các cột đó:
  - `P = exp(scale * S - m) / s` trên các cột nhìn thấy được, **đúng bằng 0** trên các cột bị
    che;
  - `L = m + log(s)`, logsumexp của các điểm số đã co giãn nhìn thấy được của hàng.
- Một điểm số bị che không bao giờ được đọc. Giá trị của nó không quan trọng, và các bài kiểm
  thử đặt nó bằng 1e30 để chứng minh điều đó.

`L` là thứ mà kernel flash của lab 20.2 phải tái tạo cho cùng các điểm số, để các bài kiểm thử
của nó có thể so sánh hai bên.

## Thiết kế (của Fig. 20.4)

- **Một khối 256 luồng cho mỗi hàng.** Chiều x của lưới (grid) là hàng (`blockIdx.x` của
  sách), chiều y là phần tử của batch. Luồng `t` xử lý các cột `t, t + 256, t + 512, ...`: gộp
  việc cho luồng (thread coarsening), được hợp nhất (coalescing) ở mọi lượt (p. 488).
- **Mặt nạ là cận của vòng lặp.** Các vòng lặp tìm giá trị lớn nhất và tính tổng dừng tại cột
  nhìn thấy được cuối cùng, nên các phần tử bị che không bao giờ bị chạm tới; vòng lặp đầu ra ghi
  số 0 tường minh cho chúng. Đó là mẹo của sách, và là lý do không có vô cực nào xuất hiện ở đâu
  cả.
- **Ba lượt qua phần nhìn thấy được của hàng**: giá trị lớn nhất, rồi tổng của
  `exp(scale * S - m)`, rồi ghi P. Giữa giá trị lớn nhất và tổng là một rào chắn (barrier).
- **Các phép rút gọn được cho sẵn**: các phép rút gọn khối hai giai đoạn của lab 10.3,
  `blockReduceMax` và `blockReduceSum`, được sao chép nguyên vẹn và ghi nguồn trong tệp. Chúng
  trả kết quả cho mọi luồng qua `scratch[0]`.

Bạn viết kernel có mặt nạ (ba lượt của nó, rào chắn giữa các phép rút gọn, và `L`) và hàm khởi
chạy.

## Chỗ khác với trang sách

Trang sách cần bốn chỉnh sửa, và có hai lựa chọn mà chúng tôi thay đổi:

1. **`gridDim.y` và `gridDim.z` bằng 1, không phải 0** (p. 487 ghi 0). Các chiều lưới không dùng
   mặc định là 1 trên CUDA và HIP (A1). Dù sao ở đây chiều y của lưới vẫn được dùng: nó đi qua
   batch.
2. **Không có `NEG_INFINITY`.** Không header CUDA nào định nghĩa tên đó (A5). Phần tử đơn vị của
   phép max là `-FLT_MAX`, hữu hạn, như ở lab 10.3: dưới fast math của Metal, hành vi của vô cực
   là không xác định (C5), và một hợp đồng hữu hạn duy nhất phục vụ cả ba backend.
3. **Không có `cub::Max()` hay `cub::Sum()`.** Các functor đó đã bị gỡ bỏ trong CCCL 3.0 (infra
   A11), và hipCUB viết khác (B1). Các phép rút gọn M4 thay cho `BlockReduce` và phép phát
   `max_or_sum` của sách: chúng đã trao kết quả cho mọi luồng sẵn rồi.
4. **`L` thay cho `D`.** Sách ghi mẫu số `D` "for reuse during training" (p. 488). Riêng `D`
   không thể dựng lại P, vì giá trị lớn nhất của hàng, mốc mà nó được tính theo, không được lưu.
   FlashAttention-2 lưu logsumexp `m + log(D)` thay vào đó (A7), và lab này cũng vậy.

Hai lựa chọn: kernel tự áp dụng `scale` (Fig. 20.4 đọc S đã được co giãn; sách cho phép cả hai
chỗ, p. 486), để S có thể đến thẳng từ lớp tuyến tính M3; và `exp` và `log` được viết là `expf`
và `logf` trong `kernel.cu`, nói rõ ra là độ chính xác đơn (A5).

## Tệp

- `work/kernel.cu` (CUDA; cũng được biên dịch cho HIP và hai trình giả lập):
  `causalSoftmaxKernel` và `launchCausalSoftmax`. Các phép rút gọn khối M4 ở đầu tệp được cung
  cấp; các TODO được xếp theo thứ tự.
- `work/kernel.metal`: `causal_softmax`, cùng kernel đó với `simd_max` / `simd_sum` và bộ nhớ
  threadgroup (các phép rút gọn Metal của lab 10.3, được cung cấp). `work/launch.mm`:
  `encodeCausalSoftmax` kiểm tra pipeline, gắn các đối số và đặt kích thước vùng nháp; bạn viết
  phần dispatch.
- `main.cpp`: chương trình điều khiển (driver). Tham chiếu trên CPU và các dung sai nằm trong
  `../shared-g11/unfused.hpp` (namespace `causal`), dùng chung với benchmark của lab 20.2.
- Được cung cấp cho benchmark, không sửa: `../shared-g11/unfused.cmake` sao chép `launchLinear`
  tham chiếu của lab 15.6 (M3) và `launchGemm` tham chiếu của lab 5.3 (M2) vào bản build, có đổi
  tên. Nếu thiếu lời giải của lab nào, các hàng của nó sẽ vắng mặt.

## Lệnh

- Kiểm thử: `python3 labs/pmpp.py run 20.1 --backend emu` trước, rồi `--backend race` (bên
  dưới), rồi `python3 labs/pmpp.py run 20.1` trên GPU của bạn.
- Đo: `python3 labs/pmpp.py run 20.1 --bench`.

## Các bài kiểm thử kiểm tra gì

- **Hình dạng**, `(nq, nk)`: (1, 1), (1, 2), (1, 4097), (7, 100), (31, 31), (32, 32),
  (33, 33), (255, 255), (256, 256), (257, 257), (1000, 1000), (2048, 2048), mỗi cái ở batch 1
  và 3. Các hàng 1 và 2 cột để phần lớn luồng rảnh; 255, 256, 257 nằm ở hai bên một lượt của
  khối; (7, 100) có `nq < nk`; 4097 cột cần 17 lượt. `emu` chạy batch 1 và 2 với `nk` tới 300;
  `race` chạy batch 1 với `nk` tới 100.
- **Dữ liệu**, một công thức cho mỗi hàng, trộn lẫn trong một batch để bắt một kernel đọc nhầm
  hàng hoặc nhầm phần tử:
  - điểm số phân bố đều trong [-8, 8];
  - cũng vậy nhưng dịch đi +90 và -90. `exp(90)` vượt khỏi phạm vi của float, nên một kernel bỏ
    qua phép trừ giá trị lớn nhất sẽ trả về vô cực hoặc NaN;
  - điểm số lớn nhất của hàng nằm ở cột nhìn thấy được cuối cùng (một lỗi lệch một ở cận sẽ bỏ
    sót nó);
  - 1e30 ở mọi cột bị che: một kernel đọc một điểm số bị che cho giá trị lớn nhất hoặc tổng sẽ
    nhận một hàng toàn số 0 hoặc NaN;
  - mọi điểm số bằng nhau (P là `1 / visible`).
- **Hệ số co giãn** 1 và `1/sqrt(128)` (`emu` chạy cái thứ hai, `race` cái thứ nhất).
- **Kiểm tra**: P trong dung sai dưới đây và mọi hàng có tổng bằng 1; P bị che đúng bằng 0; L
  trong dung sai của nó; P và L khởi đầu là NaN, nên một phần tử không ai ghi sẽ bị bắt; 64 giá
  trị canh sau P và sau L phải không bị chạm tới.

## Dung sai, và lý do

Tham chiếu được tính bằng double trên CPU từ cùng các điểm số float (`causal::refRow`). Các cận
theo cách suy ra của lab 10.3, ở trường hợp xấu nhất được kiểm thử: 4097 cột, đối số của hàm mũ
xuống tới khoảng -17 ở hệ số co giãn 1.

**P trên một cột nhìn thấy được:** `|got - ref| <= 1e-20 + 5e-5 * ref`, và hàng có tổng bằng 1
trong phạm vi `5e-5`. Sai số tương đối của một P nhiều nhất bằng tổng của:

- `exp` của chính nó: `expf` của CUDA sai trong phạm vi 2 ulp (ch10-facts A11), `exp` chính xác
  của Metal trong phạm vi 4 ulp, `exp` nhanh của nó trong phạm vi `3 + floor(|2x|)` ulp
  (ch10-facts C4): 37 ulp tại x = -17, khoảng 4.4e-6;
- đối số của nó: `scale * S` được làm tròn một lần trước khi trừ `m`, tới nửa ulp của 98 với
  các hàng +90, 3.8e-6 tuyệt đối, nên 3.8e-6 tương đối sau `exp`;
- tổng `s`: mọi số hạng mang cùng hai sai số (cộng lại nhiều nhất 8.2e-6), cộng 17 phép cộng
  trong một luồng và hai cây shuffle 5 bước, 27 lần làm tròn 2^-24, 1.6e-6 (mọi số hạng đều
  dương);
- một phép nghịch đảo và một phép nhân, 1.2e-7.

Tổng cộng nhiều nhất khoảng 1.8e-5; `5e-5` lớn hơn gấp đôi con số đó. Thiếu phép trừ giá trị lớn
nhất, sai hệ số co giãn, sai cận hay một điểm số bị che lọt vào tổng đều lệch hàng bậc độ lớn.
HIP công bố các sai khác đo được, không phải cận (ch10-facts B4); ngoài điều đó, unverified. Số
hạng tuyệt đối chỉ để một xác suất subnormal, thứ mà GPU có thể xả về 0 (flush to zero), không
tự mình làm hỏng một hàng.

**L:** `|got - ref| <= 5e-5 + 2^-21 * |ref|`. `log(s)` biến sai số tương đối của `s` (nhiều nhất
1e-5 theo danh sách trên) thành cùng sai số tuyệt đối đó, và `log` thêm sai số của riêng nó, điều
các tệp facts của chúng tôi không đặt cận (unverified: 5e-5 chừa chỗ cho vài ulp của một giá trị
dưới 9). `m` chính xác tới mức làm tròn của `scale * S` (2^-24 tương đối) và phép cộng cuối làm
tròn thêm một lần (2^-24 tương đối): `2^-21` phủ cả hai, gấp bốn lần.

Hãy viết `expf` và `logf`, không phải `__expf`: cận dùng con số của `expf`.

## Backend race

`emu` chạy từng luồng một và không thể thấy một race condition tình cờ để lại giá trị đúng.
`race` chạy các luồng của một khối cùng lúc dưới ThreadSanitizer. Trường hợp nó có mặt ở đây là
trường hợp của lab 10.3: `blockReduceMax` và `blockReduceSum` dùng chung `scratch`, và mọi luồng
đọc giá trị lớn nhất từ `scratch[0]` ngay trước khi phép rút gọn tổng ghi lại `scratch[0]`. Nếu
không có gì sắp thứ tự hai việc đó, một warp nhanh có thể ghi đè giá trị lớn nhất trước khi một
warp chậm kịp đọc nó. Trên GPU lỗi đó có thể vượt qua mọi bài kiểm thử; ThreadSanitizer chỉ ra
hai dòng mã. Một lượt chạy sạch không phải là bằng chứng.

## Trước khi bạn đo hiệu năng, hãy dự đoán

Benchmark chạy chuỗi attention không gộp ở 8 head, d = 128, N = 1024, 2048 và 4096, mỗi head
một lần khởi chạy:

1. `S = Q K^T`: M3, `launchLinear` tham chiếu của lab 15.6 (bố cục trọng số `(out, in)` đúng là
   `(n, d)` của K);
2. causal softmax của bạn, scale `1/sqrt(d)`, `nq = nk = N`;
3. `O = P V`: M2, `launchGemm` tham chiếu của lab 5.3.

Nó in thời gian của từng bước (tốt nhất trong 5 lượt đo, mỗi lượt lặp lại bước đó cho đến khi
kéo dài ít nhất 2 ms), phần chiếm của softmax trong tổng, GB/s của nó tính trên lưu lượng tối
thiểu (nửa nhìn thấy được của S đọc một lần, P ghi một lần), và số byte mà S và P chiếm. Đầu ra
của mọi bước được kiểm tra trước khi thời gian của nó được báo cáo. Các kích thước cần hơn một
phần tư bộ nhớ mà thiết bị của bạn báo sẽ bị bỏ qua.

**Dự đoán `ch20/labs/p-lab1-share`.** Trước khi chạy: ở N = 4096, d = 128, kernel mặt nạ và
softmax của bạn sẽ chiếm bao nhiêu phần trăm thời gian của chuỗi ba bước?

Gợi ý: phép nhân S làm `2 N^2 d` FLOP (M3 tính toàn bộ S, kể cả nửa bị che). Softmax di chuyển
khoảng `10 N^2` byte: nửa nhìn thấy được của S là `2 N^2` byte, đọc ba lần, và P là `4 N^2` byte,
ghi một lần. Hãy chia mỗi con số cho tốc độ giới hạn nó, trần tính toán và băng thông của bạn từ
lab 5.1; tỉ lệ giữa chúng quyết định phần chiếm. Nếu cả hai kernel chạy ở trần của chúng, một
thiết bị có điểm gãy (ridge point) R FLOP/B sẽ dành thời gian cho phép nhân gấp `2 d / (10 R)` lần
so với softmax. Rồi hãy hỏi M3 và M2 cách trần tính toán bao xa, và softmax của bạn cách trần băng
thông bao xa.

Các con số là phép đo trên máy của bạn, ở các kích thước này, không phải mục tiêu.

## Tính khả chuyển

- HIP: `kernel.cu` được biên dịch qua `pmpp/vendors/hip.h`; các phép rút gọn được cung cấp dùng
  mặt nạ lane có kiểu và `warpSize`, nên một wavefront rộng 64 cho 4 warp mỗi khối 256. Chưa được
  biên dịch hay chạy trên phần cứng AMD ở đây: unverified.
- Metal: `launch.mm` đọc độ rộng SIMD group từ pipeline (`threadExecutionWidth`) và từ chối
  dispatch trừ khi 256 luồng tạo thành các SIMD group trọn vẹn và số SIMD group không nhiều hơn số
  lane của một SIMD group (giai đoạn 2 trao kết quả của mỗi group cho một lane).
- `expf` và `logf` không cần khai báo riêng: phép kiểm tra front-end CUDA của khóa học khai báo
  các hàm toán độ chính xác đơn trong header stub của nó (một phiên bản trước của tệp này khai báo
  chúng dưới `PMPP_CUDA_FRONTEND_CHECK`, từ trước khi stub có chúng).
