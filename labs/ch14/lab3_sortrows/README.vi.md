> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 14.3: ứng viên đã sắp xếp, mọi hàng giảm dần và ổn định (mốc dự án M5)

Sách: radix sort như một chuỗi các phép phân hoạch ổn định (mục 14.4, pp. 334-336), các công thức
đích của Fig. 14.6 (p. 338) và kernel của Fig. 14.7 (p. 339), sắp xếp cục bộ trong bộ nhớ
chia sẻ (shared memory) (14.6, pp. 339-342), chữ số r bit dưới dạng r lượt cục bộ một bit (14.7, pp. 342-344),
nhiều khóa mỗi luồng (14.8, pp. 344-345). Sách không in mã cho 14.6 đến 14.8 (chúng là
bài tập 1 đến 3, p. 347). Bạn cũng dùng lại histogram bộ nhớ chia sẻ của chương 9 và phép scan
cấp khối của chương 11. Không có trong sách, và là của khóa học này: khóa float, một khối mỗi hàng, offset
chạy theo từng bucket, và phép gom (gather). Một mục đứng riêng như (A3) hay (C1) bên dưới nằm trong `notes/today/ch14-facts.md`.

Hãy viết một hàm host và các kernel đằng sau nó:

```cpp
void launchSortRowsDesc(const float* probs, float* sortedProbs, int* sortedIdx,
                        unsigned int* scratch, int rows, int cols);
```

Với mỗi hàng của `probs`, một tensor float32 `[rows x cols]` (thứ tự hàng (row-major), batch trước),
`sortedIdx` nhận các chỉ số cột theo thứ tự giảm dần của giá trị, các giá trị bằng nhau theo cột
tăng dần, và `sortedProbs` nhận các giá trị theo thứ tự đó. `scratch` chứa `4 x rows x cols`
unsigned int để bạn dùng. Đây là đầu vào cho tổng tích lũy của top-p trong module X1: sắp xếp,
rồi scan.

- Sửa `work/kernel.cu` (CUDA; cũng được build cho HIP và các trình giả lập): `descKeys`,
  `sortRowsPass`, `gatherRows`, `launchSortRowsDesc`. Sửa `work/kernel.metal` (`desc_keys`,
  `sort_rows_pass`, `gather_rows`) và `work/launch.mm` (`encodeSortRowsDesc`, hàm encode
  toàn bộ phép sắp xếp vào một compute encoder duy nhất mà nó được giao).
- Được cung cấp trong các tệp đó: phép scan cấp khối (scan-scan-add của chương 11) và các hàm phụ trợ.
- Kiểm thử: `python3 labs/pmpp.py run 14.3`. Bắt đầu với `--backend emu`, mất vài phút.
- Race: `python3 labs/pmpp.py run 14.3 --backend race` (xem "Backend race" bên dưới).
- Đo: `python3 labs/pmpp.py run 14.3 --bench`, sau khi đưa ra dự đoán bên dưới.

## Hợp đồng

- `rows >= 1`, `cols >= 1`, giá trị bất kỳ: không chỉ bội số của 256 hay của ô (tile) của bạn.
- Float bất kỳ: số âm, `-0.0` và `+0.0`, vô cực, `FLT_MAX`, số subnormal, NaN.
- Thứ tự là thứ tự của khóa sau, sắp xếp tăng dần: lấy 32 bit `u` của float; nếu
  `u` là `-0.0` (`0x80000000`) thì biến nó thành `+0.0`; rồi đặt bit dấu của một giá trị không âm
  và đảo mọi bit của một giá trị âm; rồi đảo toàn bộ khóa. Trên các số, đó là
  thứ tự số giảm dần với hai số không bằng nhau. Trên NaN, đó là các bit: `+NaN` đứng
  trước mọi thứ (kể cả `+inf`) và `-NaN` đứng sau mọi thứ (kể cả `-inf`). Đây là
  phép biến đổi mà CUB tài liệu hóa cho khóa float của nó, kể cả bước số không và quy tắc NaN
  (A3; các mẫu bit được tính trong E1).
- Ổn định: các khóa bằng nhau giữ thứ tự cột tăng dần. Thứ tự giảm dần đến từ việc đảo khóa,
  chứ không phải từ việc đảo ngược một kết quả tăng dần, vốn sẽ liệt kê các phần tử bằng nhau theo cột giảm dần (A3).
- `sortedProbs[r][p]` đúng bằng 32 bit của `probs[r][sortedIdx[r][p]]`. `probs` không bị
  sửa đổi. Không có gì được ghi ra ngoài ba buffer đầu ra.

### Vì sao dùng bit, không dùng float

Các kernel không bao giờ thực hiện một phép toán dấu phẩy động. Hàm khởi chạy đưa cho chúng `probs` và
`sortedProbs` dưới dạng `unsigned int*` (trong CUDA là một phép ép kiểu; trong Metal các buffer được gắn dưới dạng
`device const uint*` và `device uint*`), nên phép kiểm tra `-0.0`, phép kiểm tra dấu và phép sao chép
đều là việc trên số nguyên. Điều đó quan trọng trên Metal: cả hai chế độ toán học mặc định có thể có của nó đều cho phép "no
signed zeros", fast math để NaN và INF không xác định, và số subnormal có thể bị ép về không (flush to zero) trong
phép tính (C3). Một biểu thức phía float `x == 0.0f ? 0.0f : x` có thể bị viết lại; một phép kiểm tra số nguyên thì không.
Trên HIP, `__float_as_uint` có trong header nhưng không có trong tài liệu tham chiếu toán học mà AMD công bố
(B6). Cách nhìn không dấu không cần intrinsic ép kiểu bit nào cả, nên đó là dạng chạy được
ở mọi nơi. (CUDA tài liệu hóa `__float_as_uint`, A8.)

Với chính đầu vào của mốc dự án này, các xác suất softmax, mọi giá trị đều ít nhất là `+0.0`, nên
nhánh dấu không bao giờ kích hoạt (E1). Các phép kiểm thử vẫn có số âm, cả hai số không và vô cực,
vì cùng kernel đó sắp xếp logits.

## Thiết kế (của khóa học này)

Một khối 256 luồng mỗi hàng, hình dạng của lab 10.3. Ba kernel:

1. `descKeys`: khóa của mỗi giá trị (ở trên), và cột của nó làm chỉ số.
2. `sortRowsPass`, khởi chạy một lần cho mỗi chữ số r bit (bạn chọn; gợi ý là 8, tức bốn
   lượt qua 32 bit, chữ số ít quan trọng nhất trước), đọc một buffer khóa và chỉ số
   của `scratch` và ghi vào buffer kia. Mỗi khối:
   - đếm các chữ số của hàng mình trong một histogram bộ nhớ chia sẻ gồm 2^r ngăn, bằng phép toán nguyên tử
     (chương 9);
   - scan 2^r số đếm (loại trừ): chỗ mỗi bucket bắt đầu trong hàng đầu ra;
   - đi qua hàng theo từng ô 256c khóa, c khóa mỗi luồng. Mỗi ô được sắp xếp trong bộ nhớ
     chia sẻ theo chữ số bằng r phép tách một bit ổn định, dùng các công thức của Fig. 14.6 và
     phép scan cấp khối được cung cấp; rồi mỗi khóa được ghi vào offset chạy của bucket của nó, và các
     offset tiến lên theo kích thước bucket của ô.
3. `gatherRows`: `sortedIdx` từ buffer chỉ số cuối cùng, `sortedProbs` bằng cách sao chép các bit của
   `probs` tại các chỉ số đó.

Vì sao lượt này ổn định qua các ô: các ô được một khối duy nhất duyệt theo thứ tự, và offset của
một bucket chỉ tiến về phía trước, nên một khóa của ô trước luôn đặt trước một khóa của
ô sau có cùng chữ số. Offset chạy đó làm thay việc của bảng kích thước bucket theo từng
khối trong sách (Fig. 14.9): với một khối mỗi hàng, chỉ có một khối cần tính. Nó cũng
bỏ được phép scan toàn lưới của Fig. 14.7, thứ mà sách gọi nhưng không bao giờ định nghĩa (notes/ch14.md,
"Unclear in the PDF", mục 6), và lệnh đọc `bits[N]` của nó, vượt một phần tử quá N phần tử mà hình
vẽ cho thấy (mục 4).

Trong một ô, các luồng liên tiếp ghi vào các vị trí liên tiếp của một bucket: lệnh ghi được hợp nhất
của mục 14.6. Với 2^8 bucket và vài nghìn khóa mỗi ô, một bucket chứa vài khóa
mỗi ô trên dữ liệu ngẫu nhiên, nên các lệnh ghi còn xa mới được hợp nhất hoàn toàn: đó là cái giá của một
cơ số lớn mà mục 14.7 mô tả (và A10 đếm theo sector).

## Các phép kiểm thử kiểm tra gì

- Hình dạng: rows 1, 3, 4 và 7 nhân cols 1, 2, 31, 255, 256, 257, 1000, 4096, 32000, 50257 và
  151936 (hai giá trị cuối là kích thước bộ từ vựng của GPT-2 và Qwen3-8B, ch09-facts D7). Trên
  `emu`, rows 1 và 3 nhân cols 1, 2, 31, 255, 256, 257, 1000 và 4099, với một batch cho mỗi số hàng
  ở mỗi kích thước, các công thức dữ liệu luân phiên theo kích thước, vì mọi trường hợp chạy dưới ba
  thứ tự luồng và mỗi thứ tự tốn hàng chục nghìn lần chuyển luồng mỗi hàng. Trên `race`,
  rows 2 nhân cols 1, 31, 257, 1000 và 2100 (hai ô ngay cả ở 8 khóa mỗi luồng), mọi công thức dữ liệu.
- Dữ liệu, trộn lẫn trong một batch để bắt được một kernel đọc hoặc ghi sai hàng:
  - xác suất softmax của logits phân bố đều trong [-12, 12];
  - như trên với các giá trị bằng nhau chính xác được cài sẵn: cột 0 và cols-1 (bằng giá trị lớn nhất của hàng, tạo
    thành ba giá trị bằng nhau), 255 và 256, 5 và 261 (cách nhau một bước nhảy của một luồng), và các dải 40
    giá trị bằng nhau trải qua mọi bội số của 256 từ 512 đến 8192. Một ô là 256c khóa, nên với
    mọi c nguyên từ 1 đến 32, một dải sẽ vượt qua ranh giới ô khi hàng dài hơn
    256c + 20 cột;
  - mọi giá trị bằng nhau (đáp án là 0, 1, ..., cols-1);
  - one-hot (các số không phải giữ thứ tự cột phía sau số một);
  - logits có `-0.0` và `+0.0` xen kẽ mỗi 11 cột, và (trong chừng mực hàng còn
    chỗ) mỗi loại hai giá trị `-inf`, `+inf`, `-FLT_MAX`, `FLT_MAX`, `FLT_MIN`, `-FLT_MIN` và bốn
    số subnormal ở cả hai dấu;
  - như trên với tối đa bốn NaN, mỗi dấu hai cái, hai trong số đó có payload.
- `sortedIdx` bằng đúng tham chiếu (đó là phép kiểm tra tính ổn định); `sortedProbs` bằng đúng
  các bit được gom; mỗi hàng đầu ra là một hoán vị; 64 phần tử canh gác sau cuối
  `sortedProbs`, `sortedIdx` và `scratch` không bị chạm vào; `probs` không thay đổi.
- Không có dung sai: một phép sắp xếp không làm phép tính nào và các giá trị được sao chép dưới dạng bit.
- Tham chiếu, trong `main.cpp`: `std::stable_sort` các chỉ số cột trên khóa số nguyên.
  Không phải `vDSP_vsorti` của Accelerate, mà Apple tài liệu hóa là không ổn định (C2). Trên mọi hàng không có
  NaN, chương trình điều khiển (driver) còn kiểm tra tham chiếu của nó với `std::stable_sort` dùng `>` trên các float,
  cách này coi hai số không là bằng nhau và giữ thứ tự đầu vào (A11). Khi có NaN, `>` không phải là một thứ tự
  yếu chặt (strict weak ordering), nên khóa là tham chiếu duy nhất.

## Backend race

`emu` chạy từng luồng một, nên nó không thấy được hai luồng chạm vào cùng một vùng nhớ mà
không có gì sắp thứ tự chúng nếu kết quả tình cờ ra đúng. `race` chạy các luồng của một khối
cùng lúc dưới ThreadSanitizer và báo những cặp như vậy kèm cả hai dòng mã nguồn. Trong lab này, những
chỗ cần làm đúng là histogram (phép toán nguyên tử, và một rào chắn giữa bước xóa về không, đếm và
đọc), buffer ô (mỗi phép tách đọc nó rồi ghi lại), các vị trí bắt đầu bucket của một
ô, và các offset chạy (được phép phân tán đọc, rồi tiến lên, rồi được ô kế tiếp đọc).

## Con đường thư viện, theo từng backend

Lời khuyên của sách là dùng một phép sắp xếp thư viện trừ khi bạn có lý do để không dùng (p. 347). Với
phép toán này, một phép sắp xếp giảm dần ổn định trên các cặp (khóa, chỉ số) theo từng hàng:

- CUDA: `cub::DeviceSegmentedRadixSort::SortPairsDescending`, hoặc
  `cub::DeviceSegmentedSort::StableSortPairsDescending`. Hàm
  `DeviceSegmentedSort::SortPairsDescending` thông thường được tài liệu hóa là không ổn định (A5). Các giá trị là
  chỉ số token, đúng hình dạng đầu ra của lab này (A2).
- HIP: cùng các tên đó dưới `hipcub::` trong cả hai bản hipCUB của ROCm 10.2 (B1). Trong hipCUB
  4.7.0 chúng chạy segmented radix sort của rocPRIM, và `StableSortPairsDescending` chỉ đơn giản gọi
  `SortPairsDescending` (B1); rocPRIM tài liệu hóa device radix sort của nó là ổn định (B2).
- Metal: không có phép sắp xếp trong MSL và không có phép sắp xếp thư viện ở cấp kernel. Hàm
  `argSort(_:axis:descending:name:)` của MPSGraph (macOS 13) sắp xếp dọc một trục, và Apple không nêu
  tính ổn định hay thứ tự các phần tử bằng nhau cho nó (C1). Nếu các xác suất bằng nhau phải ra theo chỉ số token
  tăng dần, bạn tự viết kernel, như ở đây. Backend MPS của PyTorch có bản riêng: một radix sort với
  thiết kế đếm, scan và phân tán của chương này trong Metal và chữ số 8 bit cho khóa 32 bit. Khóa
  của nó khác với lab này: `-0.0` và `+0.0` vẫn phân biệt, và NaN đứng cuối (D6, cách chúng tôi
  đọc mã nguồn của nó).

### Vì sao kernel này quan trọng trên HIP cho llama.cpp

`argsort` CUDA của llama.cpp sắp xếp các hàng ngắn bằng một mạng bitonic trong bộ nhớ chia sẻ và các hàng dài
bằng CUB. Bản build HIP của nó không định nghĩa đường đi CUB nào cả, và đường đi bitonic cần cả hàng,
được đệm lên lũy thừa của hai, nằm trong bộ nhớ chia sẻ: một hàng 151,936 cột được đệm thành 262,144 int, 1 MiB.
Theo cách chúng tôi đọc `supports_op`, một `argsort` cỡ bộ từ vựng vì thế bị báo là không hỗ trợ
trên HIP và chạy trên một backend khác (D1). Một phép sắp xếp theo hàng mà bộ nhớ chia sẻ không tăng theo
hàng, như phép này, là thứ mà đường đi đó còn thiếu (cách chúng tôi đọc, và chưa được biên dịch hay chạy trên phần cứng
AMD ở đây).

## Trước khi bạn benchmark

Hãy ghi lại hai dự đoán (bài học yêu cầu chúng dưới id `ch14/labs/p-lab3-batch1`):

1. Ở batch 1 và 151,936 cột, kernel của bạn sẽ nhanh hơn `std::stable_sort` của một luồng CPU
   bao nhiêu lần? Một con số nhỏ hơn 1 nghĩa là chậm hơn.
2. Tỉ lệ đó sẽ tăng hay giảm ở batch 64?

Cầu nối của mục 14.8 và mục 10.10 của chương 10 giải thích vì sao một khối mỗi hàng để phần lớn
thiết bị ngồi không ở batch 1.

Benchmark sắp xếp các xác suất softmax ở rows 1, 8 và 64 nhân cols 32,000 và 151,936. Thời gian
mỗi lần gọi là kết quả tốt nhất trong 5 lần chạy được đo trên thiết bị, mỗi lần lặp lại toàn bộ phép sắp xếp (khóa, mọi
lượt, gom) liên tục cho đến khi lần chạy kéo dài ít nhất 20 ms (các lần chạy chỉ gồm một lần gọi ngắn đã
được thấy dao động tới hai lần); nó báo số hàng mỗi giây và số triệu khóa mỗi
giây. Mốc so sánh là `std::stable_sort` dùng `>` trên lần lượt từng hàng cộng phép gom, trên một
luồng CPU, đo trên host bằng một đồng hồ ổn định (tốt nhất trong 5). Mọi kết quả được đo đều được kiểm tra
trước khi báo cáo, và một hình dạng lớn hơn một phần tư bộ nhớ thiết bị được báo cáo sẽ bị
bỏ qua. Các con số là phép đo trên máy của bạn, không phải mục tiêu.

## Tính khả chuyển

- HIP: `kernel.cu` được build qua `pmpp/vendors/hip.h` với shuffle mask có kiểu và
  `warpSize`; với warp rộng 64, một khối 256 có 4 warp. `<cuda/atomic>` trên HIP đến từ
  libhipcxx, một phần của ROCm từ 10.1 (infra-facts B2). Chưa được biên dịch hay chạy trên phần cứng AMD ở đây:
  unverified.
- Metal: `launch.mm` đọc độ rộng SIMD group từ pipeline (`threadExecutionWidth`),
  kiểm tra rằng 256 luồng tạo thành các SIMD group trọn vẹn với số nhóm không nhiều hơn số lane, định
  kích thước bộ nhớ scan cho mỗi SIMD group từ đó (bội số của 16 byte, ch05-facts C8), và kiểm tra các
  mảng threadgroup của kernel có vừa không. Histogram threadgroup là `atomic_uint` (ch09-facts C1).
  Kernel xếp hạng các luồng của nó theo SIMD group và lane, vì việc chia một threadgroup thành
  các SIMD group là do bản cài đặt quyết định (ch04-facts C2) và phép scan chạy theo thứ tự đó. Mọi
  dispatch đi vào một compute encoder duy nhất: một encoder thông thường là tuần tự, và các buffer device được
  theo dõi xung đột (hazard-tracked), nên mỗi lượt thấy được lượt trước (ch02-facts C9).
- Bước kiểm tra front-end CUDA cần thư mục include shim mà `labs/pmpp.py` truyền vào
  (`-I labs/common/include/pmpp/emu/shim`) cho `<cuda/atomic>`.
