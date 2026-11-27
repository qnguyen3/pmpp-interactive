> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab X2.3: săn lỗi trên đường số nguyên

Không có trong sách: học phần X2 là của khóa học này, và các lỗi cũng vậy. Các thành phần của kernel là của sách
(phép rút gọn (reduction) bằng warp shuffle của chương 10, việc chuyển giao qua bộ nhớ chia sẻ (shared memory) sau một rào chắn (barrier) của chương 5);
thứ nó tính là đường decode CUDA của llama.cpp, như đã đọc trong lượt kiểm tra dữ kiện ngày 2026-11-20. Một
mục trần như (D3) bên dưới nằm trong `notes/today/x2-facts.md`.

Phần decode CUDA và HIP của llama.cpp không nhân trọng số Q4_0 với activation dạng float. Trước hết nó
lượng tử hóa vector activation sang Q8_1 (mã int8 theo khối 32, một hệ số tỉ lệ `d` và
`s = d * sum of the codes` được tính sẵn), rồi coi mỗi khối 32 trọng số là một tích vô hướng số nguyên
và nhân lại tỉ lệ sang float một lần (D3):

```
Y[b][o] = sum over blocks k of row o of   d_w * (sumi * d_x - 8 * s_x)
sumi    = sum over the 32 positions j of code_w(j) * code_x(j)          (an exact integer)
```

Các mã trọng số là không dấu (0 đến 15, giá trị `(code - 8) * d_w`), và `- 8` được tách ra
một lần cho mỗi khối thông qua `s` (E4). Byte j của một khối Q4_0 giữ trọng số j trong nibble thấp và
trọng số j + 16 trong nibble cao (D1), đó là lý do một mặt nạ `0x0F0F0F0F` trên một word 32 bit
cho bốn trọng số liên tiếp và `>> 4` cho bốn trọng số cách đó 16 vị trí.

`work/kernel.cu` và `work/kernel.metal` chứa GEMV đó theo dạng CUDA của llama.cpp: một khối gồm
4 warp cho mỗi hàng đầu ra, các warp chia nhau các khối của hàng, một phép rút gọn shuffle cho mỗi warp,
các kết quả từng phần của mỗi warp đi qua bộ nhớ chia sẻ, warp 0 hoàn tất. Nó biên dịch được. Nó có **bốn
lỗi độc lập**. Hãy tìm và sửa cả bốn, rồi đo đường số nguyên so với đường float của lab
X2.2.

- Sửa `work/kernel.cu` (`gemvQ4_0Q8_1Kernel`) và `work/kernel.metal` (`gemv_q4_0_q8_1`). Cùng
  bốn lỗi đó có mặt ở cả hai.
- Được cung cấp và đúng, không phải lỗi: các phép chuyển đổi FP16, `dot4_i8`, bộ lượng tử hóa
  (`launchQuantizeQ8_1`, `quantize_q8_1`), `launchGemvQ4_0Q8_1`, toàn bộ `work/launch.mm`; các lần
  nạp 16 bit của trọng số (một khối 18 byte chỉ căn chỉnh theo 2 byte, A5) và các lần nạp 32 bit
  của activation (một khối 36 byte giữ chúng căn chỉnh theo 4 byte, như `get_int_b4` của llama.cpp
  giả định); các nibble không dấu được đưa vào một phép nhân byte có dấu (0 đến 15 là như nhau
  ở cả hai); lane mask có kiểu và `warpSize`.
- Kiểm thử: `python3 labs/pmpp.py run X2.3 --backend emu`, rồi `--backend race`, rồi
  `python3 labs/pmpp.py run X2.3` cho GPU của bạn.
- Đo: `python3 labs/pmpp.py run X2.3 --bench`.

## Dự đoán của bạn

Trước khi chạy bất cứ thứ gì, hãy đọc `work/kernel.cu` và ghi lại các lỗi bạn có thể tìm thấy bằng cách
đọc. Với mỗi lỗi, ghi lại bạn mong `emu`, `race` hay GPU của bạn sẽ cho thấy nó,
và trên những trường hợp test nào:

| Lỗi | Nó là gì | `emu` | `race` | GPU của bạn |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |

Những điều cần cân nhắc: `emu` chạy từng luồng một giữa các điểm đồng bộ hóa và lấp đầy
bộ nhớ chia sẻ bằng các byte `0xCD`; `race` chạy các luồng cùng lúc dưới ThreadSanitizer và chỉ ra
hai dòng va chạm nhau, nhưng chỉ với những va chạm đã thực sự xảy ra; một GPU đủ nhanh để một
rào chắn bị thiếu có thể qua được nhờ may mắn về thời điểm thực thi. Một lỗi trong số học số nguyên chỉ lộ ra trên những đầu vào
chạm tới nó.

## Phương pháp

1. Trình giả lập trước. Ba thứ tự luồng của nó là tất định, nên một lỗi đồng bộ hóa
   thất bại ở mọi lần chạy, và AddressSanitizer chỉ ra dòng của mọi truy cập vượt biên.
2. Sửa một chỗ, chạy lại. Các trường hợp thất bại thay đổi khi bạn tiến lên. Hãy dùng điểm chung của các trường hợp
   thất bại: kích thước nào, loại dữ liệu nào, thứ tự luồng nào.
3. Với lỗi đồng bộ hóa, `--backend race` chỉ ra lần ghi và lần đọc va chạm nhau.
4. Rồi đến GPU của bạn. Ghi lại nó đã cho bạn thấy những lỗi nào và đã che giấu những lỗi nào.

## Các test kiểm tra gì

- Kích thước `(batch, in, out)`: `(1, 32, 1)` (một khối), `(1, 256, 3)`, `(2, 512, 5)`,
  `(8, 1024, 9)` với trọng số thật đi qua bộ lượng tử hóa Q4_0 `_ref` của llama.cpp
  (`../shared-g12/qformats.h`) và activation phân bố đều trong [-1, 1].
- **Khối trường hợp tệ nhất**: mọi mã trọng số là 15 và activation là một hằng số
  dương, nên mọi mã Q8_1 là 127 và tổng số nguyên của mọi khối là 15 x 127 x 32 = 60,960
  (E3). Nó tồn tại để bắt đúng một thứ.
- **j so với j + 16**: trong mọi khối, các nibble thấp là 15 và các nibble cao là 0, hoặc
  ngược lại, nên một kernel ghép một nibble với sai activation không thể qua được.
- **Hệ số tỉ lệ khối là lũy thừa của hai**: `d_w = 2^(k mod 8 - 9)` cho khối k, chính xác trong FP16, với
  các mã ngẫu nhiên.
- Bản tham chiếu là cùng đường số nguyên đó trên CPU: tổng khối số nguyên chính xác, rồi phép nhân lại
  tỉ lệ float bằng double, từ chính các khối Q8_1 mà bộ lượng tử hóa được cung cấp đã ghi. Các khối đó
  trước hết được kiểm tra với Q8_1 của chính CPU (một mã có thể lệch một và một hệ số tỉ lệ lệch một bước
  FP16, vì GPU có thể chia lệch một ulp; số học mặc định của Metal không nghiêm ngặt theo IEEE,
  ch05-facts C7). Dung sai: `1e-5 * (1 + mag)`, trong đó `mag` là tổng trên các khối của
  `|d_w| (|sumi d_x| + 8 |s_x|)`.
- Các đầu ra bắt đầu là NaN, `Xq` là các byte `0xEE`, và 256 byte bảo vệ sau mỗi mảng phải còn nguyên;
  các trọng số phải không thay đổi.
- Mỗi trường hợp cũng in ra, để tham khảo chứ không phải một test, đường số nguyên cách
  đường float bao xa (trọng số đã giải lượng tử hóa nhân với activation float): đó là cái giá của việc làm tròn
  activation sang int8 và `s` sang FP16 (D3).
- GEMV Q4_0 float mà benchmark dùng để so sánh (lời giải của lab X2.2, hoặc một bản thay thế nếu
  không có; dòng đầu tiên của lần chạy cho biết là cái nào) cũng được kiểm tra, nên phép so sánh là
  giữa hai kernel đúng.

Trên GPU, một lỗi đồng bộ hóa có thể qua được nhờ may mắn về thời điểm thực thi. Trên thiết bị Metal mà lab này được
xây dựng, khi chỉ cài riêng lỗi đó, mọi trường hợp đều qua (2026-11-20; của bạn có thể khác).
Qua được trên GPU không chứng tỏ rằng một rào chắn là không cần thiết.

## Trước khi benchmark, hãy dự đoán

Đường nào nhanh hơn trên thiết bị của bạn cho một token decode, **tính cả lượt lượng tử hóa**:
(a) lượng tử hóa activation sang Q8_1, rồi GEMV số nguyên; hay (b) GEMV float gộp của lab X2.2,
vốn giải lượng tử hóa bên trong tích vô hướng và đọc activation float? Còn
ở batch 8 thì sao?

Những điều cần cân nhắc:

- CUDA có một lệnh tích vô hướng byte bốn chiều, `__dp4a`, từ compute capability 6.1 (A2); `dot4_i8`
  của lab này chỉ dùng nó ở đó. Bản build HIP dùng dạng bốn tích thông thường, vì
  HIP không có `__dp4a` và các builtin tích vô hướng số nguyên của nó phụ thuộc vào target (B1, B2). Metal hoàn toàn không có
  lệnh tích vô hướng số nguyên nào (C1), và phần decode Metal của chính llama.cpp vẫn dùng float (D4).
- Lượt lượng tử hóa đọc activation một lần và ghi một phần tư số byte của chúng cộng với các
  hệ số tỉ lệ; so với trọng số thì nó nhỏ, nhưng nó là thêm một lần khởi chạy.
- Cả hai đường đọc cùng số byte trọng số Q4_0, vốn chiếm phần lớn với một token (E1).

## Benchmark đo gì

Chỉ trên thiết bị thật. Kích thước `in -> out` là 4096 -> 4096 và 14336 -> 4096 (lựa chọn của chúng tôi về
các phép chiếu cỡ decode), batch 1 và 8, các khối Q4_0 ngẫu nhiên (công việc của kernel không
phụ thuộc vào giá trị). Các lần chạy có tính giờ liên tiếp đọc các bản sao trọng số khác nhau, để một
lần chạy không tìm thấy trọng số của lần chạy trước trong bộ nhớ đệm (cache): tổng cộng khoảng 512 MB bản sao, hoặc
một phần tư bộ nhớ mà thiết bị báo cáo nếu con số đó nhỏ hơn. Tổng được in ra; Apple
không công bố kích thước bộ nhớ đệm nào (infra-facts C13), nên đó là một lựa chọn, không phải một dữ kiện về GPU của bạn. Mỗi
con số là kết quả tốt nhất trong ít nhất 10 lần chạy có tính giờ sau một lần không tính giờ: một command buffer
(`GPUEndTime - GPUStartTime`) trên Metal, một cặp event trên CUDA và HIP.

Các hàng `qhunt`: `path` (`int8_q8_1` hoặc `float_q4_0`), `batch`, `in`, `out`, `seconds_quantise`,
`seconds_gemv`, `seconds_total` (lượng tử hóa và GEMV trong một lượt; với đường float, chỉ riêng
GEMV), `weight_bytes`, `copies`, `source`. Bảng in ra có thêm số byte trọng số chia cho thời gian của
GEMV. Đây là các phép đo trên máy của bạn, không phải giá trị kỳ vọng.

## Tính khả chuyển

- HIP: mã dùng chung là luồng, khối, bộ nhớ chia sẻ, một rào chắn và các shuffle `_sync` với một
  mask có kiểu đủ chứa 64 bit, tất cả build được qua `pmpp/vendors/hip.h`; `dot4_i8` tránh
  `__dp4a`, thứ HIP không có (B1). Các shuffle của bộ lượng tử hóa dùng width 32, nên trên một
  wavefront 64 lane, mỗi nửa là một nhóm riêng. Không có gì ở đây đã được biên dịch hay chạy trên phần cứng
  AMD: unverified.
- CUDA: unverified cho đến khi một `reports/verify-*.md` từ một máy CUDA cho thấy lab này qua.
- Metal: bộ lượng tử hóa cần độ rộng SIMD group là bội số của 32 (các nhóm 32 giá trị của nó
  không được nằm vắt qua hai SIMD group); `encodeQuantizeQ8_1` từ chối dispatch nếu không như vậy. Apple
  không công bố độ rộng cố định nào (infra-facts C2, C5).
