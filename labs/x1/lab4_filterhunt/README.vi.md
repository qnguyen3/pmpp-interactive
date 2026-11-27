> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab X1.4: săn lỗi bộ lọc, ba lỗi trong một kernel top-p và min-p gộp

Không có trong sách: module X1 là của khóa học này, các lỗi cũng vậy. Kernel là phần B của lab X1.2,
nên các thành phần của nó là phép rút gọn (reduction) hai giai đoạn của sách (Figs. 10.13 và 10.16,
pp. 240 và 243) cùng warp scan và block scan của sách (Figs. 11.8 và 11.9, p. 264), và các quy tắc của
nó là của llama.cpp (`notes/today/x1-facts.md` D1, `ch11-facts.md` D2, `ch12-facts.md` D3). README của
lab X1.2 nêu đầy đủ hợp đồng; tóm tắt, với mỗi hàng gồm K logits được sắp theo thứ tự giảm dần:

- `sortedProbs` = softmax ở nhiệt độ 1 của hàng;
- `n = max(1, min(a, b))`, trong đó `a` đếm các token có khối xác suất xếp trước chúng dưới `topP`
  (luôn có token 0, mọi token khi `topP >= 1`), và `b` đếm các logits lớn hơn hoặc bằng
  `logit[0] + logf(minP)` (mọi token khi `minP <= 0`).

`work/kernel.cu` và `work/kernel.metal` chứa `filterSortedFused` / `filter_sorted_fused` theo dạng mà
lab X1.2 gợi ý: một khối 256 luồng cho mỗi hàng; lượt 1 cộng `exp(l - l_0)` bằng phép tổng khối hai giai
đoạn; lượt 2 đi qua hàng theo các đoạn 1,024 với khối xác suất của các đoạn trước được mang theo trong
một thanh ghi (register), ghi từng xác suất, quyết định min-p trên từng logit và top-p trên khối xác suất
của từng token, và đếm cả hai; một phép rút gọn biến các số đếm thành `n`. Mảng chia sẻ `warpSums_s`
(`simdSums_s` trên Metal) chứa tổng của khối trước, rồi các tổng của warp cho từng đoạn.

Nó biên dịch được. Nó có **ba lỗi độc lập**. Hãy tìm và sửa cả ba.

- Sửa `work/kernel.cu` (`filterSortedFused`; `launchFilterSortedFused` bên dưới nó là đúng) và
  `work/kernel.metal` (`filter_sorted_fused`). `work/launch.mm` (`encodeFilterSortedFused`) là đúng.
  Cùng ba lỗi đó có mặt trong cả hai kernel.
- Không phải lỗi: mặt nạ lane có kiểu và `warpSize` ở chỗ sách viết `0xffffffff` và 32 (HIP từ chối mặt
  nạ 32 bit và warp AMD có thể có 64 lane, `infra-facts.md` B3); ngưỡng `-FLT_MAX` giữ mọi token khi
  `minP <= 0` (`-INFINITY` không xác định dưới chế độ fast math của Metal, `ch10-facts.md` C3);
  `logf(minP)` được tính một lần trên host và truyền vào, như llama.cpp tính nó; không có lượt tìm max,
  vì hàng đã được sắp xếp; mỗi warp tự đặt tạm phần đoạn của riêng nó; giá trị loại trừ của lane đầu
  tiên được đặt về 0 sau shuffle; `max(1, ...)`; hai số đếm dùng chung một phép rút gọn; và việc các hàm
  trợ giúp là `static`.
- Kiểm thử: `python3 labs/pmpp.py run x1/lab4 --backend emu`, rồi `--backend race`, rồi
  `python3 labs/pmpp.py run x1/lab4` cho GPU của bạn.

## Những gì các phép kiểm thử kiểm tra

Các phép kiểm tra phần B của lab X1.2, ở kích thước nhỏ hơn (chương trình điều khiển, tức driver, include `main.cpp` của lab X1.2):

- Rows 1, 3 và 7 nhân K = 1, 31, 257, 1,000, 1,024, 1,025, 2,100, 4,096, 4,099 và 32,000 trên GPU, nơi
  mỗi batch chạy mười lần trên các đầu ra mới và một lần thất bại cho biết lần khởi chạy nào sai. Trình
  giả lập (emulator) chạy rows 1 và 3 nhân K = 1,000, 1,024, 1,025, 2,100 và 4,099, backend race chạy 2
  hàng với K = 33, 1,025 và 2,100. Các hàng dài hơn một đoạn 1,024 được cố ý đưa vào trên mọi backend.
- **Các hàng chính xác**, so sánh không có dung sai: một nhóm `2^j` logits bằng 0 và phần còn lại bằng
  -1000, nên mọi xác suất là một lũy thừa của hai hoặc 0 và mọi tổng riêng phần chính xác theo mọi thứ
  tự (với điều kiện `exp(0)` đúng bằng 1 và `exp(-1000)` đúng bằng 0, điều mà README của lab X1.2 bàn
  tới).
  p nằm đúng trên tổng bao gồm tại một token (token 0, bên trong dải 4 của một lane, token cuối của đoạn
  đầu tiên, token đầu của đoạn thứ hai, bên trong đoạn thứ hai): token đó phải được giữ và token kế tiếp
  bị bỏ. p = 0 giữ một token, p = 1 giữ mọi token. Một logit cài sẵn đúng bằng `0 + logf(minP)` được giữ,
  số float ngay dưới nó thì không.
- **Các hàng softmax**: xác suất sai trong phạm vi `1e-4` tương đối so với bản tham chiếu bằng double, và
  `n` bằng của bản tham chiếu ở mọi nơi dải không cho phép lựa chọn (README của lab X1.2 suy ra dải này).
- Các đầu ra bắt đầu là NaN và -1, và 64 phần tử canh gác sau mỗi đầu ra phải không bị chạm vào.

## Dự đoán của bạn

Trước khi chạy bất cứ thứ gì, hãy đọc `work/kernel.cu` và ghi lại các lỗi bạn có thể tìm ra bằng cách
đọc. Với mỗi lỗi, hãy ghi lại backend nào bạn kỳ vọng sẽ báo nó: `emu`, `race`, GPU của bạn, hay không
backend nào, và vì sao. Trang labs hỏi một con số từ đó (`x1/labs/p-lab4-emu`): **trong ba lỗi, `emu`
sẽ báo bao nhiêu lỗi?** Rồi hãy chạy `emu`, `race` và GPU của bạn, sửa từng lỗi một, và điền vào:

| Lỗi | Đó là gì | `emu` | `race` | GPU của bạn |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

## Phương pháp

1. Trình giả lập trước: `python3 labs/pmpp.py run x1/lab4 --backend emu`. Nó chạy một luồng cho đến khi
   luồng đó tới một rào chắn (barrier), một shuffle hoặc điểm kết thúc, rồi tới luồng kế tiếp, dưới ba
   thứ tự cố định (theo thứ tự, đảo ngược, xáo trộn).
2. Sửa một thứ, chạy lại. Các trường hợp thất bại thay đổi theo từng bước. Hãy dùng điểm chung của các trường
   hợp thất bại: K nào, loại hàng nào, p nào, và gợi ý trong thông báo. Driver gọi tên một lỗi đã biết khi
   `n` khớp với giá trị mà lỗi đó sẽ cho ra.
3. Hiểu trình giả lập có thể và không thể cho thấy điều gì:
   - **Số học** (một tổng sai, một phép so sánh sai, một giá trị cộng vào sai chỗ) sai ở mọi thứ tự luồng
     và trên mọi backend. Các hàng chính xác biến nó thành một số đếm sai mà không có việc làm tròn nào
     để che giấu.
   - **Bộ nhớ chia sẻ (shared memory) chưa ai ghi** chứa các byte `0xCD`, được lấp mới cho mỗi khối.
   - Điều `emu` không thể cho thấy: nó chỉ giải quyết một rào chắn hay một shuffle khi không luồng nào có
     thể chạy tiếp, nên mọi luồng hoàn thành đoạn công việc của mình tới rào chắn hay shuffle kế tiếp
     trước khi bất kỳ luồng nào bắt đầu đoạn sau đó. Một lỗi mà tác hại của nó cần một warp đi trước warp
     khác cả một đoạn sẽ vẫn bị che giấu ở đó, trong cả ba thứ tự.
4. `python3 labs/pmpp.py run x1/lab4 --backend race` chạy các luồng giả lập thực sự đồng thời dưới
   ThreadSanitizer. Hai luồng chạm vào cùng một vùng nhớ mà không có gì sắp thứ tự chúng sẽ được báo cáo
   cùng cả hai dòng mã nguồn, dù có giá trị nào ra sai hay không. ThreadSanitizer báo mỗi cặp dòng một
   lần mỗi lần chạy, nên một trường hợp sau báo cáo đầu tiên có thể qua trong khi lỗi vẫn còn đó.
5. Rồi tới GPU của bạn: `python3 labs/pmpp.py run x1/lab4`. Ghi lại những lỗi nào nó cho bạn thấy và
   những lỗi nào nó che giấu.

## Đọc bảng của bạn

Dù GPU của bạn cho thấy gì, hãy đọc nó đối chiếu với những gì nhà cung cấp mô tả, vì đó là điều thiết bị
tiếp theo được phép làm:

- Các warp khác nhau chạy độc lập (NVIDIA, `ch04-facts.md` A9; Apple: các SIMD group "make independent
  forward progress ... in the absence of threadgroup barrier operations", tức là tiến triển độc lập khi
  không có thao tác rào chắn threadgroup, `infra-facts.md` C5). Một warp đã đọc một giá trị từ bộ nhớ
  chia sẻ có thể đang ở bất kỳ đâu trong giai đoạn kế tiếp trong khi một warp khác còn chưa đọc nó.
- "Đọc từ hoặc ghi vào cùng một vị trí bộ nhớ mà không có memory fence hay đồng bộ hóa dẫn tới hành vi
  không xác định" (`ch10-facts.md` A1). Một lần ghi sau đọc cũng là một race: softmax của lab 10.3 có một
  rào chắn giữa hai phép rút gọn khối dùng chung một mảng đúng vì lý do đó, và llama.cpp cùng PyTorch đặt
  cùng rào chắn đó (ch10-facts D1, D4).

Một trường hợp qua mười lần trên GPU của bạn đã cho bạn thấy mười lịch chạy của một thiết bị. Nó không
cho bạn thấy rằng kernel là đúng.

## Tính khả chuyển

- HIP: mặt nạ lane có kiểu và `warpSize` có mặt để `kernel.cu` có thể build qua `pmpp/vendors/hip.h`;
  với warp rộng 64, một khối 256 có 4 warp. Chưa được biên dịch hay chạy trên phần cứng AMD ở đây:
  unverified.
- Metal: `launch.mm` đọc độ rộng SIMD group từ pipeline (`threadExecutionWidth`) và từ chối 256 luồng
  không tạo thành các SIMD group trọn vẹn, hoặc nhiều SIMD group hơn số lane của một SIMD group.

Không có gì để benchmark ở đây.
