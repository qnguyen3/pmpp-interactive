> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab X1.2: top-p và min-p, ghép từ các lab trước, rồi gộp (mốc dự án M5)

Không có trong sách: module X1 là của khóa học này (sách hoàn toàn không có lấy mẫu, PLAN.md). Các
thành phần là của sách: phép rút gọn (reduction) hai giai đoạn của chương 10 (Figs. 10.13 và 10.16,
pp. 240 và 243) và warp scan cùng block scan của chương 11 (Figs. 11.8 và 11.9, p. 264), mà các lab
10.3, 11.6 và 14.3 đã dựng thành một softmax, một mặt nạ top-p và một phép sắp xếp hàng. Thứ tự của hai
bộ lọc, quy tắc top-p và phép kiểm tra min-p là của llama.cpp; việc gộp, lập luận tiền tố và cận
sai số là của chúng tôi. Một mục đứng riêng như (D1) dưới đây nằm trong `notes/today/x1-facts.md`; các mục
khác ghi rõ tệp của chúng. Những gì mã nguồn của llama.cpp làm đã được kiểm tra trong lượt rà soát dữ
kiện ngày 2026-11-20.

Sau phép matmul cuối cùng của mô hình, một bộ lấy mẫu biến một hàng logits thành một token. Top-p giữ
tập nhỏ nhất gồm các token có khả năng cao nhất mà xác suất cộng lại đạt p; min-p giữ các token có xác
suất ít nhất bằng min-p nhân với xác suất lớn nhất. Chuỗi mặc định của llama.cpp chạy top-p, rồi
min-p, rồi nhiệt độ, rồi lần rút (D1). Lab này dựng giai đoạn top-p và min-p hai lần: một lần bằng cách
ghép những gì bạn đã có, một lần dưới dạng một kernel duy nhất.

**Phần A, ghép.** Viết

```cpp
void launchFilterRows(const float* candLogits, int* sortedPos, float* sortedLogits,
                      float* sortedProbs, int* n, void* workspace,
                      int rows, int K, float topP, float minP);
```

từ `launchSortRowsDesc` của lab 14.3, `launchSoftmaxRows` của lab 10.3 và `launchTopPMask` của lab
11.6 (lời giải tham chiếu của chúng được cung cấp sẵn), cộng thêm một kernel của riêng bạn,
`minPCount`, với hàm khởi chạy của nó

```cpp
void launchMinPCount(const float* sortedLogits, const int* keptTopP, int* n,
                     int rows, int K, float minP);
```

hàm này đếm số token qua min-p của mỗi hàng đã sắp xếp và ghi `n` từ số đếm đó và `keptTopP` của lab
11.6. Benchmark cũng gọi nó riêng, để đo thời gian chuỗi sau phép sắp xếp.

**Phần B, gộp.** Viết

```cpp
void launchFilterSortedFused(const float* sortedLogits, float* sortedProbs, int* n,
                             int rows, int K, float topP, float minP);
```

một kernel, `filterSortedFused`, một khối 256 luồng cho mỗi hàng, bắt đầu từ các hàng đã được sắp xếp
và tạo ra cùng `sortedProbs` và `n` trong khi đọc logits hai lần và ghi xác suất một lần.

## Tệp và lệnh

- Sửa `work/kernel.cu` (CUDA; cũng được build cho HIP và các trình giả lập): `minPCount`,
  `launchMinPCount`, `launchFilterRows`, `filterSortedFused`, `launchFilterSortedFused`.
  Sửa `work/kernel.metal` (`min_p_count`, `filter_sorted_fused`) và `work/launch.mm`
  (`encodeMinPCount`, `encodeFilterRows`, `encodeFilterSortedFused`). Các tên là cố định.
- Có sẵn trong `work/kernel.cu`: warp sum, warp scan và block sum của chương 10 và 11, dưới dạng các
  hàm `static`. Hãy để các hàm trợ giúp của bạn cũng là `static`: các bản sao có sẵn bên dưới định
  nghĩa những hàm trùng một số tên, và trình giả lập liên kết mọi thứ vào một chương trình.
- Có sẵn, không được sửa, trong `labs/x1/shared-g12/`: `sortrows_g12.cu`, `softmax_g12.cu` và
  `topp_g12.cu`, các bản sao lời giải tham chiếu của các lab 14.3, 10.3 và 11.6 dưới tên tệp riêng,
  với `laneIdx`, `warpIdx` và `warpsInBlock` được đổi thành `static` để ba tệp liên kết được vào một
  chương trình giả lập. `app/src/chapters/x1/copies-g12.test.ts` kiểm tra rằng không có gì khác lệch
  khỏi bản gốc. Trên Metal, chương trình điều khiển (driver) nạp `solution/kernel.metal` của các bản gốc từ chính thư mục của
  chúng và liên kết `solution/launch.mm` của chúng.
- Kiểm thử: `python3 labs/pmpp.py run x1/lab2`. Bắt đầu với `--backend emu` (vài phút: riêng phép sắp
  xếp của lab 14.3 ở đó đã là hàng chục block scan cho mỗi hàng).
- Race: `python3 labs/pmpp.py run x1/lab2 --backend race`.
- Đo: `python3 labs/pmpp.py run x1/lab2 --bench`, sau khi có các dự đoán bên dưới.

## Hợp đồng

Mỗi hàng chứa K logits hữu hạn; `rows >= 1`, `K >= 1`; thứ tự hàng (row-major), batch trước. Với mỗi
hàng:

- **Chỉ phần A.** `sortedPos` chứa các vị trí của hàng được sắp theo logit giảm dần, logits bằng nhau
  theo vị trí thấp hơn (ổn định). `sortedLogits[i]` là đúng 32 bit của `candLogits[sortedPos[i]]`.
- `sortedProbs` là softmax ở nhiệt độ 1 của hàng đã sắp xếp:
  `exp(l_i - l_0) / sum_j exp(l_j - l_0)`, trong đó `l_0`, logit đầu tiên sau khi sắp xếp, là lớn nhất.
- `n = max(1, min(a, b))`, trong đó
  - `a`, top-p: số token `i` có `i == 0`, hoặc `topP >= 1`, hoặc khối xác suất xếp trước `i`
    (`sortedProbs[0] + ... + sortedProbs[i-1]`) dưới `topP`. Đó là số giữ lại của lab 11.6.
  - `b`, min-p: số token có `sortedLogits[i] >= sortedLogits[0] + lnMinP`, trong đó
    `lnMinP = logf(minP)` được tính một lần, trên host, dưới dạng float. Mọi token khi `minP <= 0`.
- Các đầu vào không bị sửa đổi, và không có gì được ghi quá cuối của bất kỳ đầu ra nào.

Vì sao có từng điều khoản:

- **Top-p giữ token vượt qua p.** Bộ lấy mẫu trên host và trên graph của llama.cpp đều giữ đúng những
  token có khối xác suất trước chúng dưới p (`notes/today/ch11-facts.md` D2), tức là "tập nhỏ nhất có
  khối xác suất ít nhất bằng p" của bài báo top-p (D9). `p >= 1` giữ tất cả vì llama.cpp trả về sớm ở
  đó, và vì khối xác suất trước các token cuối của một phép scan float có thể làm tròn thành 1. README
  của lab 11.6 có phần còn lại.
- **Min-p trên logits, so với giá trị lớn nhất cộng một logarit.** Min-p trên host của llama.cpp tính
  `max_logit + logf(p)` một lần và giữ `logit >= min_logit` (`ch12-facts.md` D3). Trong số học chính
  xác, đó là `p_i >= minP * p_max`, quy tắc của bài báo min-p (D9), không cần softmax. Phép so sánh là
  `>=`: một logit nằm đúng ở ngưỡng vẫn được giữ. Tính `logf` một lần trên host làm ngưỡng là cùng một
  số float trên mọi backend.
- **Top-p, rồi min-p.** Đó là thứ tự mặc định của llama.cpp (D1); mlx-lm chạy cùng hai bộ lọc theo cùng
  thứ tự (D8). Ở đó cả hai được áp dụng trước nhiệt độ.
- **`n` là một giá trị nhỏ nhất.** Cả hai bộ lọc đều giữ một tiền tố của hàng đã sắp xếp: top-p vì khối
  xác suất trước một token chỉ tăng dọc theo hàng, min-p vì các logit chỉ giảm. Ngưỡng của min-p dùng
  logit lớn nhất, thứ mà top-p không bao giờ loại bỏ, nên chạy min-p sau top-p thấy cùng ngưỡng như chạy
  nó trên cả hàng. Một token qua được cả hai đúng khi nó nằm trong tiền tố ngắn hơn. `max(1, ...)` vì
  logit lớn nhất luôn qua top-p và qua min-p với mọi `minP <= 1`; với `minP > 1` không gì qua min-p, và
  một bộ lấy mẫu vẫn phải trả về một token.
- **Phép sắp xếp ổn định**, nên trong hai logits bằng nhau, cái ở vị trí thấp hơn đứng trước. Ở ranh
  giới top-p, đó là quy tắc phá hòa (lab 11.6: "giá trị bằng nhau được phân định theo vị trí, tức là
  theo phép sắp xếp"). mlx-lm thì giữ mọi token bằng với ranh giới (ch11-facts D3).

### Workspace của phần A

`workspace` là một vùng cấp phát mà driver tạo ra, bố trí theo thứ tự này:

| Phần | Kích thước | Dùng bởi |
|---|---|---|
| vùng nháp của phép sắp xếp | `4 * rows * K` uint32 | lab 14.3 |
| `massBefore` | `rows * K` float32 | lab 11.6 |
| `keptTopP` | `rows` int32 | lab 11.6, rồi `minPCount` của bạn |
| `keep` | `rows * K` uint8 | lab 11.6 |

Tổng cộng `21 * rows * K + 4 * rows` byte; `keep` đứng cuối để không có gì sau nó cần căn chỉnh. Trên
Metal, cùng bốn phần đó là bốn buffer, vì `encodeTopPMask` của lab 11.6 gắn mỗi buffer của nó ở
offset 0.

## Thiết kế gợi ý cho phần B

Một khối 256 luồng cho mỗi hàng, theo dạng của lab 11.6:

- Hàng đã được sắp xếp, nên giá trị lớn nhất là phần tử 0: không cần lượt tìm max.
- Lượt 1: tổng của `exp(l - l_0)`, gộp việc cho luồng (thread coarsening) dọc theo hàng, rồi hai giai
  đoạn của chương 10 (`blockSum` có sẵn).
- Lượt 2: các đoạn 1,024 token (4 cho mỗi luồng), với khối xác suất của các đoạn trước được mang theo
  trong một thanh ghi (register). Mỗi warp đọc phần logits của mình một cách được hợp nhất, tính từng
  xác suất, ghi nó vào `sortedProbs` và đặt tạm nó trong bộ nhớ chia sẻ (shared memory); đó cũng là chỗ
  logit đang nằm trong thanh ghi, nên hãy quyết định min-p tại đó. Mỗi lane scan dải 4 xác suất liên
  tiếp của mình trong thanh ghi, khối scan các tổng của dải (warp scan, rồi warp 0 trên các tổng của
  warp), và khối xác suất trước mỗi token, kể cả các đoạn trước, quyết định top-p khi nó còn trong thanh
  ghi.
- Đếm cả hai quyết định cho mỗi luồng, cộng các số đếm bằng một phép rút gọn hai giai đoạn, và để một
  luồng ghi `n`.
- Các mảng chia sẻ được dùng lại: bởi phép rút gọn và phép scan, và bởi đoạn này với đoạn kế tiếp. Với
  mỗi mảng, hãy tìm lần đọc cuối và lần ghi kế tiếp, và đặt một rào chắn (barrier) giữa chúng.

Trên Metal, `simd_sum`, `simd_prefix_exclusive_sum` và `simd_prefix_inclusive_sum` làm trong một lời
gọi điều mà các vòng lặp shuffle làm (`infra-facts.md` C2, ch11-facts C1).

## Những gì các phép kiểm thử kiểm tra

Cả hai phần chạy trên cùng các đầu vào: phần A trên các ứng viên, phần B trên logits đã sắp xếp của bản
tham chiếu.

- Kích thước: rows 1, 3 và 7 nhân K = 1, 2, 31, 40, 255, 256, 257, 1,000, 1,024, 1,025, 4,096,
  32,000 và 151,936 (40 là top-k mặc định của llama.cpp, D1; 151,936 là một kích thước bộ từ vựng,
  `ch09-facts.md` D7). Trình giả lập chạy rows 1 và 3 và K tới 4,099, một batch cho mỗi thiết lập;
  backend race chạy 2 hàng với K = 1, 33, 257, 1,025 và 2,100. Trên cả hai, phần A chỉ chạy ở hai kích
  thước (K = 40 và 1,025; race: 33 và 1,025), trên hàng min-p cài sẵn và trên các hàng softmax ở
  p = 0.9, min-p = 0.05, vì phép sắp xếp của lab 14.3 chiếm phần lớn thời gian của chúng. Các hàng khác
  loại được trộn trong một batch, nên một kernel đọc nhầm hàng sẽ thất bại.
- **Các hàng chính xác, so sánh không có dung sai.** Một nhóm `G = 2^j` logits bằng 0 và phần còn lại
  bằng -1000, ở các vị trí xáo trộn. Với `exp(0) = 1` và `exp(-1000) = 0`, mọi xác suất là `2^-j` hoặc
  0 và mọi tổng riêng phần đều chính xác, theo bất kỳ thứ tự cộng nào. Một `exp` làm tròn đúng cho ra
  chính xác hai giá trị đó; các cận vài ulp của nhà cung cấp (ch10-facts A11, C4) không hứa điều đó.
  Chúng đúng trên Metal và trên các trình giả lập khi lab này được dựng, và là unverified trên CUDA và
  HIP.
  Với `p = (t + 1) / G`, tổng bao gồm ở token t đúng bằng p: token t phải được giữ và token t + 1 bị
  bỏ. t được đặt ở token 0, bên trong dải 4 của một lane (token 129), ở token cuối của đoạn 1,024 đầu
  tiên (token 1,023), ở token đầu của đoạn thứ hai và bên trong nó. `p = 0` và `1e-7` giữ một token;
  `p = 1` và `1.5` giữ mọi token, kể cả phần đuôi -1000.
- **Min-p tại ngưỡng của nó.** Một hàng có logit lớn nhất là 0, với một logit cài sẵn đúng bằng
  `logf(minP)` (0.05 và 0.1) như host tính, phải được giữ, và số float ngay dưới nó, phải không được
  giữ; `p = 1`. Cộng 0 là chính xác dưới mọi chế độ làm tròn hay mọi cách sắp xếp lại phép cộng, nên các
  số đếm này chính xác trên mọi backend. `minP = 1` chỉ giữ các logits bằng với logit lớn nhất.
- **Các hàng softmax**: logits phân bố đều trong [-12, 12], có đỉnh nhọn (một ở 12, phần còn lại trong
  [-12, 0]) và gần phẳng (trong [-1, 1]), ở p = 0.5, 0.9 và 0.95 và min-p = 0, 0.05 và 0.1. Các hàng
  dài có giá trị bằng nhau (bộ sinh có 65,536 mức). `sortedProbs` phải nằm trong dải bên dưới so với bản
  tham chiếu; `n` phải bằng của bản tham chiếu ở mọi nơi mà khối xác suất trước mỗi token của bản tham
  chiếu và khoảng cách của mỗi logit tới ngưỡng min-p nằm ngoài dải, và bên trong dải thì chấp nhận
  một trong hai giá trị kề nhau.
- Luôn luôn: `sortedPos` là một hoán vị theo thứ tự ổn định và `sortedLogits` chứa các bit của ứng
  viên; `1 <= n <= K`; `n` của phần B bằng của phần A ở mọi nơi dải không cho phép lựa chọn; các đầu ra
  bắt đầu là NaN hoặc -1 để mọi thứ chưa từng được ghi đều lộ ra; 64 phần tử canh gác sau mỗi đầu ra và
  sau workspace không bị chạm vào; đầu vào không đổi.
- Các lab có sẵn 14.3, 10.3 và 11.6 cũng được kiểm thử riêng, để một tệp có sẵn bị hỏng lộ ra đúng là
  như vậy chứ không phải như lỗi của bạn.

## Dải dung sai, và vì sao

Bản tham chiếu được tính bằng double từ các logits float: các xác suất, và khối xác suất trước mỗi
token dưới dạng tổng tích lũy chính xác của chúng. Xác suất được so sánh với
`|got - ref| <= 1e-4 * ref`, khối xác suất (qua quyết định top-p) với cùng dải tương đối: dải của lab
11.6, được suy lại ở đây cho các xác suất mà GPU tự tính.

Trường hợp xấu nhất ở hàng lớn nhất được kiểm thử, 151,936 logits, 256 luồng, đối số mũ trong
[-24, 0], mọi số hạng đều dương (nên mỗi lần làm tròn nhiều nhất là `2^-24` của một tổng riêng phần
không lớn hơn tổng cuối cùng):

- `exp`: `expf` của CUDA sai trong phạm vi 2 ulp (`ch10-facts.md` A11); `exp` nhanh của Metal trong
  phạm vi `3 + floor(|2x|)` ulp (ch10-facts C4), 51 ulp hay khoảng `6.1e-6` tương đối ở x = -24. HIP
  công bố các sai khác đo được, không phải cận (ch10-facts B4): unverified ngoài các giá trị đó.
- Đối số `l - l_0` được làm tròn một lần: tới `24 * 2^-24`, khoảng `1.4e-6` tương đối trong `exp`.
- Tổng: mỗi luồng lần lượt cộng tới 594 số hạng (593 lần làm tròn; lần đầu, 0 + x, là chính xác), rồi
  hai tầng cây. Apple không mô tả thứ tự của `simd_sum`, nên tính 63 lần làm tròn cho mỗi tầng: tổng
  cộng 719, `4.3e-5`.
- Một phép nghịch đảo và một phép nhân: `1.2e-7`.

Vậy một xác suất sai trong khoảng `5.1e-5` tương đối với `exp` nhanh của Metal (`3.7e-5` với cây 5 + 3
tầng của CUDA và `expf`). Khối xác suất trước một token là tổng của các xác suất như vậy, tất cả đều
dương, nên nó mang cùng sai số tương đối, cộng với các lần làm tròn của chính phép scan: 3 trong dải
của một lane, 63 và 63 cho hai tầng scan (lại theo thứ tự bất kỳ trên Metal), 148 cho phần mang sang qua
149 đoạn 1,024, và 3 lần kết hợp các phần: 280 lần làm tròn, `1.7e-5`. Tổng cộng khoảng `6.8e-5`, nằm
trong `1e-4`. Chuỗi ghép có cùng cận: softmax của lab 10.3 và phép scan của lab 11.6 cộng theo cùng
dạng. Một mẫu số sai, một phép scan bao gồm hoặc thiếu phần mang sang sẽ lệch cả một xác suất trọn vẹn.

Min-p có một dải riêng trên các hàng softmax: GPU cộng cùng hai số float như host (`l_0 + lnMinP`),
nhưng một trình biên dịch được phép sắp xếp lại phép cộng có thể so sánh `l - l_0` với `lnMinP` thay
vào đó, và điều này có thể khác ở bit cuối. Các logits cách ngưỡng trong phạm vi
`1e-6 * (|l_0| + |lnMinP|)` có thể đi theo hướng nào cũng được. Khi `l_0` bằng 0 (các hàng chính xác và
hàng cài sẵn) thì không có dải.

Vì sao không so sánh `n` chính xác ở mọi nơi: kết quả của một phép scan float phụ thuộc vào cây cộng của
nó, cây này khác nhau giữa kernel của bạn, của lab 11.6 và bản tham chiếu (ch11-facts A8, D4), và Metal
thậm chí không hứa làm tròn về gần nhất (C5). Bên trong dải, đáp án nào cũng là một bộ lấy mẫu đúng.

## Trước khi bạn benchmark

Hai dự đoán (trang labs hỏi chúng dưới tên `x1/labs/p-lab2-fused` và `x1/labs/p-lab2-sort`):

1. **Ở 64 hàng gồm 151,936, kernel gộp sẽ nhanh hơn chuỗi ghép không tính phép sắp xếp bao nhiêu
   lần?** Trước hết hãy đếm số byte mỗi thiết kế phải di chuyển cho mỗi token: softmax của chuỗi ghép
   đọc logits ba lần và ghi xác suất, lab 11.6 đọc chúng và ghi khối xác suất cùng byte giữ lại, min-p
   đọc logits thêm lần nữa; kernel gộp đọc logits hai lần và ghi xác suất một lần. Rồi hãy quyết
   định xem cả hai có chạy ở cùng băng thông không.
2. **Phép sắp xếp sẽ chiếm bao nhiêu phần thời gian của chuỗi ghép, kể cả phép sắp xếp, ở K =
   151,936, và ở K = 40?** Trình khám phá của mục X1.5 là nơi để lập luận về điều này: lab 14.3 làm
   bao nhiêu việc cho mỗi hàng, bất kể K là bao nhiêu, và các bộ lọc làm bao nhiêu?

Benchmark chạy rows 1 và 64 nhân K = 40, 1,024, 32,000 và 151,936 trên các hàng dạng softmax ở top-p
mặc định 0.95 và min-p 0.05 của llama.cpp (D1). Nó đo riêng phép sắp xếp của lab 14.3, toàn bộ
`launchFilterRows` của bạn (kể cả phép sắp xếp), chuỗi ghép sau phép sắp xếp (softmax, mặt nạ top-p,
phép đếm min-p của bạn, trên cùng logits đã sắp xếp), và kernel gộp của bạn trên các logits đã sắp
xếp đó, không tính thời gian phép sắp xếp. Thời gian mỗi lần gọi là tốt nhất của 5 lần chạy đo, mỗi lần
lặp lại lời gọi cho đến khi kéo dài ít nhất 2 ms. Nó in ra số mili giây mỗi lần gọi, GB/s theo số byte
tối thiểu mỗi thiết kế bộ lọc phải di chuyển (29 byte mỗi token cho bản ghép, 12 cho bản gộp), phần
của phép sắp xếp trong chuỗi ghép, và mức tăng tốc của kernel gộp so với chuỗi ghép sau phép sắp
xếp. Mọi kết quả đo được đều được kiểm tra trước khi báo cáo. Các con số là phép đo trên máy của bạn,
không phải mục tiêu.

Một kích thước 1 hàng tối đa là 600 KB mỗi buffer, và một lần chạy đo lặp lại lời gọi trên cùng các
buffer; hãy đọc các kích thước 64 hàng để biết băng thông.

## Tính khả chuyển

- HIP: mặt nạ lane có kiểu và `warpSize` có mặt để `kernel.cu` có thể build qua `pmpp/vendors/hip.h`;
  với warp rộng 64, một khối 256 có 4 warp. Chưa được biên dịch hay chạy trên phần cứng AMD ở đây:
  unverified.
- Metal: `launch.mm` đọc độ rộng SIMD group từ pipeline (`threadExecutionWidth`) và từ chối 256 luồng
  không tạo thành các SIMD group trọn vẹn, hoặc nhiều SIMD group hơn số lane của một SIMD group. Chế độ
  toán mặc định của Metal cho phép sắp xếp lại phép cộng và chế độ làm tròn của nó có thể là làm tròn về
  không (`ch05-facts.md` C7, x1-facts C5): dải ở trên cho phép thứ tự bất kỳ bên trong một SIMD group,
  và các hàng chính xác vẫn chính xác theo thứ tự bất kỳ và ở cả hai chế độ làm tròn.
- Ngưỡng min-p "giữ tất cả" là `-FLT_MAX`, không phải `-INFINITY`: dưới chế độ fast math của Metal, hành
  vi của vô cực là không xác định (`ch10-facts.md` C3), và `-FLT_MAX` nhỏ hơn mọi logit hữu hạn trên mọi
  backend.
- Logits NaN và vô cực nằm ngoài hợp đồng.

## Tiếp theo là gì

Đây là giai đoạn top-p và min-p của mốc M5. Lab X1.3 ghép cùng giai đoạn này từ các bản sao riêng của
nó bên trong toàn bộ bộ lấy mẫu; muốn đi xa hơn, hãy thay kernel gộp của lab này vào chỗ giai đoạn
ghép ở đó và đo sự khác biệt trên cả một bước lấy mẫu. Lab X1.4 săn ba lỗi trong kernel gộp mà bạn
vừa viết.
