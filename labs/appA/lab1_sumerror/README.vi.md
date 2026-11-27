> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab A.1: sai số của phép tính tổng: thứ tự, độ rộng biến tích lũy và phép bù

Sách: mục A.6 (pp. 593-594). Cây là phép rút gọn (reduction) dùng bộ nhớ chia sẻ (shared memory)
của chương 10 (Fig. 10.10, p. 237) đặt sau vòng lặp gộp việc của Fig. 10.20 (p. 247). Phụ lục A
không trình bày mã nào: các kernel, các biến tích lũy 16 bit, vòng lặp có bù, công cụ thăm dò
device, các cận sai số và mọi phép đo là của khóa học này.

Mục A.6 đưa ra hai khẳng định mà bạn có thể đo được. Thứ tự của một tổng float làm thay đổi kết quả
của nó, nên một cây rút gọn và một vòng lặp cho kết quả khác nhau là chính đáng (p. 593). Và để có
thứ gì đó vững chắc hơn việc sắp xếp lại, có phép tính tổng có bù của Kahan, mà sách nêu tên nhưng
không mô tả (p. 594). Lab này tính tổng một mảng float32 theo ba cách, với ba kiểu biến tích lũy, và
đo sai số của mỗi cách so với một bản tham chiếu cùng chi phí của mỗi cách. Đây cũng là câu hỏi mà
một framework phải trả lời khi chọn biến tích lũy: PyTorch cộng các tổng half và bfloat16 bằng float
trên mọi device, trong khi `sum` của MLX trên một mảng float16 trên Metal, theo cách chúng tôi đọc
kernel của nó, vẫn ở float16 (appA-facts D1, D2).

## Phần của bạn

Ba phép tính tổng của `n` giá trị float32, mỗi phép là một template theo kiểu biến tích lũy `AccT`.
Mọi phép cộng được làm tròn về `AccT`; kết quả được ghi vào `out[0]` dưới dạng float.

- **(a) `sumSerial`**: một luồng (thread) cộng `x[0]`, `x[1]`, ... theo thứ tự chỉ số. Đây là thứ
  tự của vòng lặp CPU, thứ tự "sequential" ở p. 593.
- **(b) `sumTree`**, tất định, không dùng atomic. Luồng t của khối (block) b trước hết cộng `coarse`
  phần tử của nó `x[b*B*coarse + t + j*B]`, j = 0 đến coarse - 1, theo đúng thứ tự đó (B là kích
  thước khối; bước nhảy giữ cho các lần nạp của một warp được hợp nhất). Sau đó khối kết hợp B tổng
  của nó bằng cây của Fig. 10.10 (bước nhảy B/2, ..., 2, 1, mỗi tầng một rào chắn), và luồng 0 ghi
  tổng của khối vào `partials[b]`. Một lần khởi chạy thứ hai của cùng kernel, một khối, rút gọn
  `partials` vào `out[0]`.
- **(c) `sumKahan`**: như (b), nhưng vòng lặp của mỗi luồng là một tổng có bù. Cây vẫn giữ nguyên
  dạng thông thường. Mã khởi đầu (starter) CUDA cho sẵn bốn dòng của vòng lặp.

Các hàm khởi chạy, mỗi phép tính tổng một hàm, trả về false mà không khởi chạy khi kích thước nằm
ngoài hợp đồng (n < 1, coarse < 1, block không phải lũy thừa của hai từ 1 đến 1024) hoặc khi bản dựng
này không có kernel cho biến tích lũy đó:

```cpp
bool launchSumSerial(const float* x, float* out, int n, Acc acc);
bool launchSumTree (const float* x, float* partials, float* out, int n, int coarse, int block, Acc acc);
bool launchSumKahan(const float* x, float* partials, float* out, int n, int coarse, int block, Acc acc);
```

`Acc` (`AccFloat`, `AccHalf`, `AccBfloat`) nằm trong `sumerror.h`. Các biến tích lũy hiện có:

| | float | half | bfloat16 |
|---|---|---|---|
| Metal | `float` | `half` | `bfloat` |
| CUDA | `float` | `__half` | `__nv_bfloat16` (compute capability 8.0 trở lên; chương trình điều khiển kiểm tra) |
| HIP | `float` | `__half` | `__hip_bfloat16`, được đặt tên `__nv_bfloat16` bằng một typedef trong `sumerror.h` |
| emu, race | `float` | không có | không có |

Trên CUDA và HIP, các bản khởi tạo 16 bit nằm bên trong các khối `#if PMPP_NARROW_ACC` mà mã khởi
đầu cung cấp. Các trình giả lập (emulator) và stub của phép kiểm tra front-end CUDA mô hình hóa
`__half` nhưng không mô hình hóa `__nv_bfloat16`, và điều kiện `PMPP_NARROW_ACC` trong `sumerror.h`
loại cả hai nhánh 16 bit khỏi những bản dựng đó, nên các khối đó chỉ được nvcc và hipcc biên dịch.

## Các tệp

- Sửa `work/kernel.cu`: `sumSerial`, `sumTree`, `sumKahan` và ba hàm khởi chạy. Các hàm trợ giúp
  `addRn` và `subRn` được cung cấp cho vòng lặp có bù.
- Sửa `work/kernel.metal`: `sum_serial`, `sum_tree`, `sum_kahan`. Chín bản khởi tạo
  `[[host_name("sum_tree_half")]]` ở cuối tệp được cung cấp sẵn.
- Sửa `work/launch.mm`: `encodeSumSerial`, `encodeSumTree`, `encodeSumKahan`. Hai lượt của cây là
  hai lần dispatch trong cùng một encoder bạn được cho, như trong lab 10.1.
- Được cung cấp, không sửa:
  - `main.cpp`: các bộ dữ liệu, bản tham chiếu CPU, mô hình CPU của các thứ tự của bạn, các cận, các
    test và benchmark;
  - `sumerror.h`: `Acc`, các nguyên mẫu hàm khởi chạy, và các header kiểu 16 bit;
  - `fpformat.h`: phép làm tròn FP32, FP16 và BF16 bằng phần mềm, cùng quy tắc với trình khám phá của
    các mục A.1 đến A.4 (`app/src/sim/minifloat.ts`);
  - `../shared/fpprobe.cu`, `../shared/fpprobe.metal`, `../shared/fpprobe_launch.mm`: công cụ thăm
    dò device.

## Lệnh

Trình chạy chấp nhận số của lab, `A.1`, và phần đầu tên thư mục của nó, `appA/lab1`. Mọi lệnh bên
dưới dùng cách thứ hai, phần đầu của id mà các bản ghi lần chạy lưu lại (`appA/lab1_sumerror`).

- Kiểm thử: `python3 labs/pmpp.py run appA/lab1 --backend emu` trước, rồi
  `python3 labs/pmpp.py run appA/lab1`. Trình giả lập thực hiện hơn một trăm lần khởi chạy cho mỗi
  test, mỗi lần dưới ba thứ tự luồng, nên hãy cho nó vài phút.
- Các rào chắn của cây dưới ThreadSanitizer: `python3 labs/pmpp.py run appA/lab1 --backend race`.
- Đo: `python3 labs/pmpp.py run appA/lab1 --bench` (chỉ trên thiết bị thật).
- Lời giải tham chiếu: thêm `--solution`. Bắt đầu lại: `python3 labs/pmpp.py reset appA/lab1 --yes`.

## Công cụ thăm dò device

Trước bất kỳ phép tính tổng nào, chương trình điều khiển (driver) hỏi device ba câu hỏi bằng một
luồng (`../shared/fpprobe.*`). Đầu vào đến từ một buffer, nên trình biên dịch không thể tính ra đáp
án lúc dựng (appA-facts E1).

- `1 + 0.75 ULP`: số float liền trên 1 khi device làm tròn đến gần nhất, 1 khi nó làm tròn về phía
  không. Metal cho phép cả hai (C2); mặc định của CUDA là làm tròn đến gần nhất (A9).
- `2^-126 * 0.5`: số subnormal 2^-127 nếu các số denormal FP32 được giữ, 0 nếu bị xả về
  không. nvcc mặc định giữ chúng (A6); Metal có thể xả (C2); mặc định của HIP không được ghi trong
  tài liệu (B3).
- `(1 + 2^-12)^2 - (1 + 2^-11)`: đúng bằng 2^-24 với một lần làm tròn (một phép nhân-cộng gộp),
  0 khi tích được làm tròn trước (A9).

Công cụ thăm dò được biên dịch với thiết lập mặc định của từng backend. Trên các trình giả lập, nó
mô tả CPU host và trình biên dịch host, không phải một GPU.

## Các test đánh giá một kết quả thế nào

**Bản tham chiếu** là một tổng có bù kiểu Neumaier ở độ chính xác double trên CPU, chính xác hơn
nhiều so với bất kỳ tổng nào ở đây (appA-facts E1). Sai số được đo so với nó, chia tỉ lệ theo
`S = sum of |x|`.

**Dữ liệu**, sinh từ các seed cố định và làm tròn về định dạng của biến tích lũy, với mọi giá trị
nhỏ hơn số chuẩn nhỏ nhất của định dạng được đặt về 0 (không có đầu vào subnormal; Metal có thể xả
chúng, C2):

- `ones`; `ints`, các số nguyên từ -3 đến 3;
- `uniform` trên [0, 1);
- `mixed`: độ lớn từ 2^-14 đến 2^-2, mỗi khoảng nhị phân (binade) có khả năng như nhau; `mixed-asc`
  và `mixed-desc` là cùng các giá trị đó đã sắp xếp; `cancel` là `mixed` với dấu ngẫu nhiên;
- `softmax`: `exp(l - max)` cho các logits l phân bố đều trên [-9, 0], với một số 1 ở một vị trí
  ngẫu nhiên;
- `big-small`: 1, rồi n - 1 bản sao của nửa ULP của 1 trong định dạng đó (2^-24 cho float).

**Kích thước.** emu: n = 1, 2, 33, 1000, 4097; coarse 1, 3, 8; khối 64 và 256. race: n = 33 và
1000; coarse 1 và 3; khối 64. Thiết bị thật: n = 1, 2, 31, 32, 33, 255, 256, 257, 1000, 4097, 100003
và 2^20; coarse 1, 2, 3, 8, 64; khối 64 và 256. Trên Metal, một khối lớn hơn
`maxTotalThreadsPerThreadgroup` của pipeline bị bỏ qua, và việc bỏ qua được in ra. Đầu ra và
partials bắt đầu là NaN, và mỗi trường hợp được khởi chạy hai lần trên cùng các buffer.

**Các trường hợp chính xác (float).** `ones` và `ints` phải cho kết quả chính xác ở mọi thứ tự: mọi
tổng riêng phần là một số nguyên có độ lớn tối đa 3n, nhỏ hơn 2^24, nên không phép cộng nào phải làm
tròn.

**Cận sai số (float, và các định dạng hẹp trên Metal, CUDA, HIP).** Phép suy dẫn là của chúng tôi,
phép phân tích sai số thuận thông thường của một tổng. Mỗi phần tử x đi đến kết quả qua một số d
phép cộng có làm tròn, và mỗi lần làm tròn nhân thứ đi qua nó với (1 + e) với |e| tối đa
u = 2^-p (p = 24 cho float, 11 cho half, 8 cho bfloat16) khi làm tròn đến gần nhất. Metal có thể
làm tròn về phía không thay vào đó (C2), khi đó |e| nhỏ hơn 2u, nên các test dùng v = 2u xuyên suốt.
Tích của d thừa số như vậy nằm trong khoảng `g(d) = d*v / (1 - d*v)` quanh 1, nên

`|result - reference| <= g(D) * S + a`, trong đó D là d lớn nhất trên mọi phần tử.

Cộng vào giá trị 0 ban đầu là chính xác, đó là nguồn gốc của các "- 1". Với L = log2(block),
blocks = ceil(n / (block*coarse)) và c2 = ceil(blocks / block), số lần cộng mà mỗi luồng ở lượt 2
thực hiện:

- serial: D = n - 1;
- tree: D = (coarse - 1) + L + (c2 - 1) + L: vòng lặp của luồng, cây, và lặp lại như vậy ở lượt 2.
  Lưu một tổng riêng phần dưới dạng float là chính xác với cả ba định dạng;
- Kahan: một vòng lặp có bù trên c phần tử chính xác đến khoảng 2u cộng một số hạng theo c*u^2, bất
  kể c là bao nhiêu. Test cho phép `K(c) = 2v + 4*c*v^2` với c > 1 (0 với c = 1) và kết hợp bốn giai
  đoạn: `(1 + K(coarse)) * (1 + g(L)) * (1 + K(c2)) * (1 + g(L)) - 1`. Đây là một ước lượng bậc nhất
  với biên độ rộng rãi, không phải một chứng minh.

`a` cho phép trường hợp một tổng riêng phần subnormal bị xả về không: 2n (serial, tree) hoặc 4n
(Kahan) nhân với số chuẩn nhỏ nhất của định dạng. Ở đây chỉ dữ liệu FP16 mới có thể tạo ra trường
hợp đó.

**Test riêng của Kahan (float).** `big-small` ở coarse 64: luồng 0 của khối 0 cộng 1 rồi 63 nửa ULP
của 1. Một vòng lặp thông thường bỏ mất từng số một, tổng cộng khoảng 32 ULP; vòng lặp có bù phải
nằm trong khoảng 2 ULP quanh bản tham chiếu. Đây là trường hợp cho thấy trình biên dịch có giữ lại
phép bù của bạn hay không.

**Biến tích lũy hẹp (Metal; CUDA và HIP khi được dựng).** Các bộ dữ liệu là `uniform`, `mixed` và
`cancel`, và một trường hợp có S đạt tới 65,504 (giá trị half lớn nhất) bị bỏ qua, nên mọi tổng
riêng phần đều hữu hạn trong FP16. Rồi 4,097 số một qua serial và tree phải có cùng các bit như mô
hình CPU của cùng thứ tự, làm tròn đến gần nhất hoặc về phía không. Mô hình đó là "CPU model" của
`main.cpp`: cùng các phép cộng theo cùng thứ tự, mỗi phép được làm tròn về định dạng bằng phần mềm
qua `fpformat.h`. Một kernel lặng lẽ tích lũy bằng float sẽ trả về 4097 ở đó.

Mã khởi đầu chưa sửa biên dịch được và thất bại ở mọi test tính tổng: các hàm khởi chạy của nó trả
về false mà không khởi chạy. Không có lỗi cài sẵn nào; đây là một lab viết và đo.

## Các test thấy được gì và không thấy được gì

- **emu** chạy từng luồng một theo ba thứ tự, dưới AddressSanitizer: chỉ số sai, một phép đọc vượt
  quá n, một rào chắn bị thiếu. Lab này không dựng tổng half hay bfloat16 nào ở đó, và trình biên
  dịch của nó là của host, vốn mặc định không sắp xếp lại các phép cộng float (appA-facts B5). Vì vậy
  một vòng lặp có bù qua được trên emu không cho bạn biết gì về việc một trình biên dịch GPU làm gì
  với nó.
- **race** chạy các luồng của một khối đồng thời dưới ThreadSanitizer. Cây là một lần cập nhật bộ nhớ
  chia sẻ giữa các rào chắn: một rào chắn đặt sai chỗ sẽ được báo cáo kèm dòng mã nguồn.
- **Metal** là backend duy nhất ở đây kiểm thử các định dạng hẹp và là backend duy nhất có trình
  biên dịch GPU. Nếu `kahan, float, big-small` thất bại trên Metal và qua trên emu, gợi ý bên dưới
  giải thích lý do. Hãy thử tự tìm ra trước: câu hỏi là trình biên dịch của Metal mặc định được phép
  làm gì với `(t - sum) - y`.
- **CUDA và HIP** chạy cùng các test trên thiết bị thật. Việc nvcc có sắp xếp lại các phép cộng float
  hay không không được ghi trong tài liệu (appA-facts A13); test `big-small` của lab này chính là
  phép đo.
- Phép kiểm tra chính xác đến từng bit dùng các số một, nên nó bắt được một biến tích lũy float
  nhưng không bắt được mọi sai sót về thứ tự. Cột `model` của benchmark so sánh mọi bộ dữ liệu với mô
  hình CPU của thứ tự bạn dùng.

<details>
<summary>Gợi ý: Kahan chỉ thất bại trên Metal</summary>

Phần hiệu chỉnh `c = (t - sum) - y` bằng 0 trong số học thực. Chế độ toán mặc định của Metal, chế
độ mà runtime của lab dùng (`MTLCompileOptions` mặc định), được tài liệu mô tả là fast hoặc relaxed,
và cả hai đều cho phép trình biên dịch sắp xếp lại các phép toán dấu phẩy động; chỉ chế độ `safe`
cấm điều đó (appA-facts C3, ch05-facts C7). Đặc tả MSL cảnh báo rằng việc sắp xếp lại làm hỏng
"code that relies on rounding behavior such as (x + 2^52) - 2^52" (C2), và một vòng lặp có bù chính
là loại mã đó. Trên máy dùng để dựng lab này, kernel có bù khi chưa sửa đã vượt cận 2 ULP ở n = 1000,
coarse 64, với kết quả giống đến từng bit với kết quả mà một vòng lặp thông thường trả về trên trình
giả lập: phép bù đã biến mất.

Cách sửa là một dòng ở phạm vi tệp trong `kernel.metal`:

```metal
#pragma METAL fp math_mode(safe)
```

Đặc tả cũng cho phép đặt pragma ở đầu một câu lệnh ghép (C3). Ở phạm vi tệp, nó áp dụng cho mọi
kernel trong tệp, nên benchmark so sánh tree và Kahan dưới cùng một chế độ.
</details>

## Trước khi đo hiệu năng, hãy dự đoán

Hãy viết những điều này ra trước. Trang labs hỏi ba điều đầu tiên dưới các id trong ngoặc.

1. Kernel serial độ chính xác half trả về gì trên device của bạn với 65,536 giá trị uniform
   (`appA/labs/p-lab1-half-loop`). Mô hình của mục A.6, làm tròn đến gần nhất, dừng ở 2048; Metal có
   thể làm tròn về phía không thay vào đó (C2). Hãy tính xem điều đó sẽ cho ra gì, rồi so sánh với
   dòng làm tròn của công cụ thăm dò. Trình khám phá của mục A.2 cho thấy khoảng cách của half tại
   mọi giá trị.
2. Kernel Kahan chạy lâu hơn kernel tree thông thường bao nhiêu lần trên 2^24 số float, ở cùng hệ số
   gộp việc (`appA/labs/p-lab1-kahan-cost`). Vòng lặp có bù làm bốn phép cộng cho mỗi phần tử trong
   khi vòng lặp thông thường làm một; phép tính tổng đọc 4 byte cho mỗi phần tử trong cả hai trường
   hợp.
3. Kernel tree float thông thường sẽ lệch bao nhiêu ULP ở hệ số gộp việc 4096 trên 2^22 số float
   uniform, và Kahan có lệch như vậy ở cùng hệ số không (`appA/labs/p-lab1-coarse`).
4. Thứ tự nào, `mixed-asc` hay `mixed-desc`, được kernel serial xử lý tốt hơn. Lời khuyên sắp xếp
   trước của sách (p. 594) là một câu trả lời; hãy nói vì sao.
5. Device của bạn có giữ các số denormal FP32 và làm tròn đến gần nhất hay không. Dòng đầu tiên của
   benchmark trả lời điều đó.

## Đọc đầu ra

Benchmark in ra bốn bảng. Cùng các hàng đó được ghi vào `reports/runs` dưới `appA/lab1_sumerror`,
nơi các mục A.4 và A.6 và trang labs đọc chúng.

- **probe**: `rounding` (`nearest` hoặc `toward-zero`), `keeps_denormals`, `fuses_mul_add`, cho
  device này dưới thiết lập trình biên dịch mặc định.
- **accuracy**: n = 65,536, coarse 16, block 256 (hoặc lũy thừa của hai lớn nhất mà mọi pipeline
  tree và Kahan cho phép, trên Metal). Mỗi phương pháp, biến tích lũy và bộ dữ liệu một hàng:
  `result`, `reference`, `abs_err`, `rel_err` (tương đối so với S), `ulps` (theo định dạng của biến
  tích lũy, tại bản tham chiếu) và `model`: `same` nếu kết quả của bạn có cùng các bit như mô hình
  CPU của thứ tự đó dưới chế độ làm tròn đã thăm dò. `differs` có thể đến từ một thứ tự khác, từ một
  device đã xả một tổng riêng phần subnormal (mô hình giữ chúng), hoặc từ một trình biên dịch đã
  sắp xếp lại các phép cộng của bạn. Một kết quả bị tràn được in thành `inf` và lưu thành `null`:
  tổng thật của 65,536 số một lớn hơn giá trị half lớn nhất, 65,504.
- **coarse**: tree và Kahan ở float trên 2^22 số float uniform ở coarse 1, 4, 16, 64, 256, 1024 và
  4096: `time ms` (tốt nhất trong các lần chạy), `GB/s` (4 byte mỗi số float đầu vào chia cho thời
  gian đó: số byte mà một phép tính tổng phải đọc, không phải lượng mà hệ thống bộ nhớ đã di chuyển),
  `rel err` và `ulps`. Ở hệ số gộp việc lớn có ít khối, nên hãy dự kiến phải đánh đổi tính song song
  lấy độ chính xác, và xem mỗi phương pháp bắt đầu tốn kém từ đâu.
- **time**: serial, tree và Kahan ở mỗi biến tích lũy, dữ liệu uniform, coarse 16. Tree và Kahan chạy
  trên 2^24 số float, hoặc ít hơn nếu kích thước đó dùng hơn một phần tám bộ nhớ mà device báo cáo
  (dòng in ra cho biết n nào đã chạy). Serial chỉ chạy trên 2^20 số float: một luồng duyệt 2^24 số
  float có thể chạm giới hạn thời gian lệnh của GPU.

Trước mỗi bảng trong hai bảng có đo thời gian, một phần tư giây các phép tính tổng tree float không
tính giờ giữ cho device bận, để kernel đầu tiên được đo không phải gánh thêm thời gian device thức
dậy; trong bảng `coarse`, tree và Kahan chạy xen kẽ vì cùng lý do. Mỗi lần chạy có đo thời gian là
một lời gọi hàm khởi chạy hoặc hàm encode của bạn, bao gồm cả hai lượt, và kết quả tốt nhất của vài
lần chạy sau một lần chạy không tính giờ được báo cáo. Các con số là phép đo trên máy này ở các kích
thước này, không phải tính chất của các kernel. Không có giá trị kỳ vọng nào được in hay lưu.

## Các backend khác nhau ở đâu (không có trong sách)

- **CUDA**: `__fadd_rn` và `__fsub_rn` là các phép cộng mà NVIDIA ghi trong tài liệu là không bao giờ
  được gộp vào một phép nhân-cộng (appA-facts A9); việc chúng có ngăn cả việc sắp xếp lại hay
  không thì không được nêu (A13). `__half` và `__nv_bfloat16` dùng các toán tử thông thường, mỗi `+`
  là một phép cộng làm tròn đến số chẵn gần nhất (A16). BF16 cần compute capability 8.0 trở lên (A2).
- **HIP**: cùng mã nguồn thông qua `pmpp/vendors/hip.h`. HIP-Clang mặc định dùng
  `-fno-associative-math`, nên phép bù tồn tại qua một bản dựng bình thường; `-ffast-math` sẽ cho
  phép trình biên dịch xóa nó (B5). `__fadd_rn` có cùng tên (B4). Kiểu BF16 là `__hip_bfloat16` từ
  `<hip/hip_bf16.h>` (B1), được đổi tên trong `sumerror.h`.
- **Metal**: `half` và `bfloat` là kiểu có sẵn (`bfloat` từ Metal 3.1, C1). Phép làm tròn có thể về
  phía không và các số denormal có thể bị xả (C2): công cụ thăm dò cho bạn biết trường hợp nào. Không
  có `__fadd_rn`; gợi ý ở trên nói về thứ thay thế nó.
- Một bản tham chiếu double trên GPU là khả thi trên CUDA và HIP nhưng không trên Metal, vốn không có
  `double` (C1). Bản tham chiếu chạy trên CPU cho cả ba.

## Đi xa hơn

- **Một cây có bù.** Giữ cặp (sum, c) qua cả cây: ở mỗi tầng, kết hợp hai cặp bằng một phép TwoSum
  của các tổng và cộng cả hai phần hiệu chỉnh. Cột Kahan của bảng `coarse` có thay đổi ở coarse 1,
  nơi cây làm toàn bộ công việc, không?
- **Bộ cộng cắt bớt của sách.** Bài tập 3 (p. 599) hỏi sai số của một bộ cộng làm tròn về phía không.
  Trên CUDA, `__fadd_rz` chính là bộ cộng đó (appA-facts A9): hãy viết một tổng serial với nó và so
  sánh sai số của nó với sai số của `__fadd_rn`. Trên HIP, các dạng `_rz` cần
  `OCML_BASIC_ROUNDED_OPERATIONS` (B4); Metal không có chế độ làm tròn theo từng phép toán (C2).
- **Theo cặp trong thanh ghi.** `sum` của NumPy dùng phép tính tổng theo cặp từng phần
  (appA-facts D4) và phép tính tổng CPU của PyTorch dùng một chuỗi các khúc từ 16 phần tử trở lên
  (D1): nhóm theo vị trí, để các tổng riêng phần tương tự nhau gặp nhau. Hãy làm cho vòng lặp
  của mỗi luồng tính theo cặp trên `coarse` phần tử của nó và so sánh với Kahan trong bảng `coarse`.

## Trạng thái của lời giải tham chiếu

verified on Metal (test và benchmark), emulator-checked (`emu` và `race`), front-end-checked cho
`kernel.cu` và `../shared/fpprobe.cu`. unverified trên CUDA và trên HIP: chưa có máy nào có một
trong hai chạy nó, và các nhánh FP16 và BF16 của `kernel.cu` và `sumerror.h` chưa được bất kỳ
trình biên dịch nào ở đây biên dịch.
