> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 20.5: decode grouped-query, nạp mỗi hàng K/V một lần

**Hãy làm xong lab 20.4 trước.** Mã khởi đầu (starter) của lab này (`kernel.cu` và
`kernel.metal`) bắt đầu từ kernel theo từng head tham chiếu của lab 20.4, nên mở nó ra sẽ lộ đáp
án của lab đó.

Sách: mục 20.7 (pp. 508-509, Eqs. 20.12 và 20.13) cho MQA và GQA và cường độ tính toán của
chúng; mục 20.5 (Eqs. 20.4 và 20.6) cho phép trộn mà bạn đã viết ở lab 20.4.

Trong một mô hình GQA, `g_q = h_q / h_kv` query head dùng chung mỗi K/V head. Kernel của lab 20.4
cho mỗi query head một khối riêng, nên g_q khối của một nhóm đều đọc cùng các hàng K và V: kernel
yêu cầu bộ nhớ gấp g_q lần số byte mà cache chứa. Hãy viết `launchDecodeAttentionGrouped`, với
đúng các đối số và hợp đồng của `launchDecodeAttention` ở lab 20.4, sao cho mỗi hàng K và V được
đọc một lần và dùng cho cả g_q truy vấn. Đó là điều đường `gqa_ratio` của llama.cpp làm: nhiều Q
head so với một K/V head đã nạp (`notes/today/ch20-facts.md` D1).

`g_q` ở đây là số query head **trên mỗi K/V head**, nghĩa mà Eqs. 20.12 và 20.13 cần (quy ước của
tài liệu tham khảo [13]). G của bài báo GQA, "the number of groups", là `h_q / g_q`, số K/V head;
phần văn của sách trộn lẫn hai nghĩa (E3, và mục "Unclear in the PDF" của notes).

Hợp đồng: của lab 20.4 (`q`, `out` `[batch, h_q, 128]`; `Kc`, `Vc`
`[batch, max_seq, h_kv, 128]`; `len[b]` từ 1 đến `max_seq`; không có mặt nạ), cộng thêm
`g_q <= 8`. Với một nhóm lớn hơn, hãy in một thông điệp và không khởi chạy gì.

## Thiết kế (của khóa học này)

Sách không in kernel GQA nào. Cách nhóm lại thiết kế của lab 20.4 này là của chúng tôi.

- **Một khối 256 luồng cho mỗi (K/V head, chuỗi)**: một lưới (grid) `(h_kv, batch)`. Khối `kvh`
  phục vụ các query head từ `kvh * g_q` đến `kvh * g_q + g_q - 1`.
- **Các warp vẫn chia nhau các key** (warp w nhận các key w, w + 8, ...) và **các lane vẫn chia
  nhau chiều head** (4 giá trị mỗi lane). Với mỗi key, một warp nạp hàng K và hàng V của nó vào
  thanh ghi (register) một lần, rồi tính g_q điểm số (g_q cây bướm) và g_q lần cập nhật online.
- **Mỗi warp giữ g_q trạng thái đang chạy** (m, D, O~), một cho mỗi query head.
- **Phép trộn chạy một lần cho mỗi query head**, qua cùng một buffer chia sẻ: công bố O~ của
  các warp cho head g, rào chắn (barrier), kết hợp bằng Eqs. 20.4 và 20.6, ghi đầu ra của head g,
  rào chắn lần nữa trước khi head g + 1 ghi đè buffer. Giữ O~ của cả tám head cùng lúc sẽ cần 32
  KiB, đó là toàn bộ giới hạn của một threadgroup Metal trên các họ GPU mà bảng của Apple liệt kê
  (C1).
- **Kích thước nhóm là một hằng số lúc biên dịch, `GQ`.** Trong `kernel.cu` kernel là một
  template, `decodeAttentionGroupedKernel<GQ>`, và hàm khởi chạy chọn bản thể bằng một `switch`
  trên `h_q / h_kv`. Trên Metal, chương trình điều khiển (driver) biên dịch `kernel.metal` một lần
  cho mỗi kích thước nhóm với `GQ` được định nghĩa như một macro tiền xử lý (một trong hai cách
  được ghi trong tài liệu để có kích thước lúc biên dịch, C6). Các mảng có kích thước `GQ` và các
  vòng lặp có số lần lặp biết trước là thứ cho phép trình biên dịch giữ các trạng thái trong thanh
  ghi. Đặt kích thước mọi mảng cho 8 head và tắt những cái không dùng lúc chạy vẫn biên dịch được
  và cho đáp án đúng, nhưng phiên bản đầu tiên của lời giải tham chiếu đã làm vậy và đo được chậm
  hơn kernel theo từng head, ngay cả ở `g_q = 1`.

**Metal**: cùng cấu trúc. `simd_sum` trao mỗi điểm số cho mọi lane, nên hãy gọi nó một lần cho
mỗi head của nhóm, từ mọi lane. Giá trị lớn nhất đang chạy bắt đầu ở một giá trị hữu hạn (C5), và
chương trình điều khiển kiểm tra các SIMD group 32 lane (C3).

## Chỗ khác với trang sách

Mọi điều trong README của lab 20.4: một `M_INIT` hữu hạn thay cho `NEG_INFINITY`, một cây bướm
thay cho `cub::WarpReduce`, một warp logic 32 lane với độ rộng 32 ở mọi lệnh shuffle, một mặt nạ
lane có kiểu. Và "≈ g_q" của Eq. 20.13 đếm các phần tử 1 byte; ở fp32 mỗi phần tử K và V vẫn tốn 4
byte, nên cường độ tính toán của kernel của bạn khoảng `g_q / 2` FLOP/B (0.5 của lab 20.4, nhân
g_q).

## Tệp

- `work/kernel.cu`: `decodeAttentionGroupedKernel<GQ>` và `launchDecodeAttentionGrouped`. Nó bắt
  đầu là kernel theo từng head tham chiếu của lab 20.4, đã là một template trên `GQ`; các TODO
  đánh dấu những gì cần nhóm lại: các mảng chia sẻ (1), ánh xạ khối sang head (2), trạng thái của
  mỗi warp (3), vòng lặp qua key (4), phép trộn (5), hàm khởi chạy (6).
- `work/kernel.metal`: `decode_attention_grouped`, cùng điểm xuất phát. `GQ` được chương trình
  điều khiển định nghĩa. `work/launch.mm`: `encodeDecodeAttentionGrouped` đã gắn sẵn các đối số;
  bạn viết phần kiểm tra kích thước và phần dispatch.
- Được cung cấp: lời giải tham chiếu của lab 20.4 được sao chép vào thư mục build với các tên đã
  đổi (`provided/perhead.cmake`). Kernel theo từng head của nó là mốc so sánh của benchmark, và
  phép nối thêm KV của nó cấp dữ liệu cho vòng lặp 16 bước. Bài kiểm thử đầu tiên chạy riêng nó:
  nếu bài đó thất bại, lỗi là của khóa học, không phải của bạn.
- `main.cpp`: chương trình điều khiển và các tham chiếu trên CPU. Các bài kiểm thử là của lab
  20.4, trong `../shared-g12/decode_test.hpp`.

## Lệnh

- Kiểm thử: `python3 labs/pmpp.py run 20.5 --backend emu` trước, rồi `--backend race` (các warp
  trộn kết quả qua bộ nhớ chia sẻ (shared memory) một lần cho mỗi head; ThreadSanitizer chỉ ra
  hai dòng mã của một rào chắn bị thiếu), rồi `python3 labs/pmpp.py run 20.5` trên GPU của bạn.
- Đo: `python3 labs/pmpp.py run 20.5 --bench`.

## Các bài kiểm thử làm gì

Các bài kiểm thử decode của lab 20.4 trên kernel của bạn: `(h_q, h_kv)` = (8, 8), (8, 4),
(8, 2), (8, 1) và (6, 2), tức g_q = 1, 2, 4, 8 và 3; một đến bốn chuỗi mỗi lần gọi với độ dài
riêng (1, 2, 31, 33, 100, 1000 và 4097 trên GPU; tới 33, cộng một trường hợp 200, trên `emu`; tới
40 trên `race`); key ngẫu nhiên và key late max; NaN ở mọi hàng cache mà kernel không được đọc;
một dải giá trị canh sau `out`; một tham chiếu double trên CPU với dung sai của lab 20.4. Rồi vòng
lặp 16 bước của lab 20.4, với phép nối thêm của lab 20.4 và kernel của bạn trong một lần gửi.

Các bài kiểm thử kiểm tra đáp án, không kiểm tra việc nhóm. Một kernel vẫn tính một query head
mỗi khối sẽ vượt qua chúng nếu nó được khởi chạy trên `(h_q, batch)`. Chỉ benchmark mới cho thấy
mỗi hàng giờ có được đọc một lần hay không.

## Trước khi bạn đo hiệu năng, hãy dự đoán

Benchmark chạy ở `h_q = 32`, chiều head 128, `len` = 8192, batch 1 và 4, và `h_kv` = 32, 8 và 4
(g_q = 1, 4 và 8). Nó đo thời gian kernel theo từng head của lab 20.4 và kernel của bạn theo các
vòng xen kẽ và in số mili giây và GB/s cho cả hai. GB/s tính số byte cache **duy nhất**,
`2 x len x h_kv x 128 x 4` cho mỗi chuỗi, cho cả hai kernel, nên con số cao hơn nghĩa là cùng
cache đó được tiêu thụ nhanh hơn. Cột `blocks` là số khối mỗi thiết kế khởi chạy theo hợp đồng
của nó: `batch x h_q` cho kernel theo từng head, `batch x h_kv` cho kernel theo nhóm.

**Dự đoán `ch20/labs/p-lab5-speedup`.** Hãy ghi ra, trước khi chạy: ở `h_kv = 8` (g_q = 4),
batch 1, kernel theo nhóm của bạn sẽ nhanh hơn kernel theo từng head bao nhiêu lần.

Các lực cần cân nhắc (README này không đưa ra phán quyết):

- **Số byte được yêu cầu.** Kernel theo từng head yêu cầu gấp g_q lần số byte duy nhất, còn
  kernel của bạn yêu cầu chúng một lần. Eq. 20.13 nói cường độ tính toán tăng khoảng g_q lần.
- **Bộ nhớ đệm (cache).** g_q khối theo từng head của một nhóm đọc cùng các hàng. Nếu chúng chạy
  gần như cùng lúc, một bộ nhớ đệm giữa các lõi và DRAM có thể phục vụ các lần đọc lặp lại, và
  DRAM thấy ít hơn nhiều so với g_q lần số byte. CUDA nói các khối "may be executed in any order"
  (`notes/today/ch04-facts.md` A5), và đặc tả của Apple không nói gì về thứ tự của các
  threadgroup (`ch12-facts.md` C4), nên chúng có chồng lấn nhau hay không là điều cần đo.
- **Số khối đang bay.** Ở batch 1 và `h_kv = 8`, kernel của bạn khởi chạy 8 khối trong khi kernel
  theo từng head khởi chạy 32. Đạt băng thông đầy đủ cần nhiều yêu cầu đang bay; với một truy vấn
  mỗi head và một batch nhỏ, ít khối có thể không phát đủ yêu cầu. Đó là lý do Flash-Decoding chia
  các key của một head cho nhiều khối (A8).

Các con số được in ra là phép đo trên máy của bạn, không phải kỳ vọng. Trang bài học đặt các tỉ lệ
đo được cạnh cường độ tính toán khoảng g_q của Eq. 20.13.

## Vượt ra ngoài sách (của khóa học này)

- **Chia cả các key cho nhiều khối.** Cho mỗi (K/V head, chuỗi) nhiều khối, mỗi khối xử lý một
  phần các key, và trộn các bộ (m, D, O~) của chúng trong một kernel thứ hai bằng cùng Eqs. 20.4
  và 20.6. Điều đó khôi phục số khối bị mất do việc nhóm. Flash-Decoding làm điều này
  (`num_splits` trong `flash_attn_with_kvcache`, A8), llama.cpp trộn các phần bằng
  `flash_attn_combine_results` (D1), và MLX chuyển sang kernel hai lượt của nó cho GQA từ 4096 key
  trở lên (và từ 1024 trên các thiết bị lớn hơn của nó, D4).
- **Các thư viện nhận K/V có ít head hơn Q.** FlashAttention ánh xạ query head sang K/V head đúng
  như lab này làm (A8), và MLX yêu cầu K và V không được "pre-tiled to match q" (D4). Lặp lại các
  K/V head để tạo một tensor có hình dạng MHA, như `repeat_kv` tham chiếu của Meta làm (D3), tốn
  thêm đúng số byte mà lab này loại bỏ.
