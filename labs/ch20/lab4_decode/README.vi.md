> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 20.4: nối thêm vào KV cache và decode một token (mốc dự án M7)

Sách: mục 20.4 (pp. 488-492, Figs. 20.5 đến 20.7) cho những gì một bước decode tính; mục 20.5
(pp. 493-504, Eqs. 20.3 đến 20.6, Figs. 20.10 đến 20.16) cho online softmax mà bạn dùng để xây
dựng nó; mục 20.6 (pp. 506-507, Eq. 20.9) cho lý do nó bị giới hạn bởi băng thông.

Với một KV cache, một bước decode của một lớp attention nhận một token mới cho mỗi chuỗi: q, k
và v của nó (Fig. 20.6). k và v mới được nối thêm vào cache, rồi q chú ý tới mọi thứ trong cache.
Hãy viết hai hàm host làm việc này cho một batch:

```
launchKvAppend(k_new, v_new, Kc, Vc, pos, batch, max_seq, h_kv)
launchDecodeAttention(q, Kc, Vc, len, out, batch, h_q, h_kv, max_seq, scale)
```

| Tensor | Hình dạng |
|---|---|
| `k_new`, `v_new` | `[batch, h_kv, 128]` |
| `Kc`, `Vc` (cache) | `[batch, max_seq, h_kv, 128]` |
| `q`, `out` | `[batch, h_q, 128]` |
| `pos`, `len` | `[batch]`, int32 |

`float` theo thứ tự hàng (row-major), batch đứng trước, chiều head 128 (`HEAD_DIM`, d của sách).
Bố cục cache là của mã Llama tham chiếu của Meta, được cấp phát trước `max_seq` hàng cho mỗi chuỗi
(`notes/today/ch20-facts.md` D3).

- **Nối thêm.** Ghi hàng `pos[b]` của chuỗi b trong cả hai cache và không gì khác: một phép ghi
  rải theo chỉ số, như `ggml_set_rows` của llama.cpp ghi cache của nó (D3).
- **Decode.** Query head h của chuỗi b chú ý tới `len[b]` hàng đầu tiên của K/V head
  `h / (h_q / h_kv)`, nên một kernel bao quát MHA (`h_kv = h_q`), GQA và MQA (`h_kv = 1`).
  `out = softmax(scale * q K^T) V`, `scale = 1/sqrt(128)`. **Không có mặt nạ**: một truy vấn mới
  duy nhất được phép thấy mọi token trong cache. Fig. 20.6 không có mặt nạ, và mã của Meta chỉ
  dựng mặt nạ khi có hơn một token được xử lý (D3).

Hợp đồng: `0 <= pos[b] < max_seq`, `1 <= len[b] <= max_seq`, `h_q` là bội của `h_kv`, mọi giá
trị đều hữu hạn. Các chuỗi có độ dài khác nhau trong một lần gọi.

## Thiết kế (của khóa học này)

Sách không cho mã nào cho cache, phép nối thêm hay một kernel decode (p. 492). Thiết kế này là của
chúng tôi. Kernel decode Metal của MLX, `sdpa_vector`, có cùng hình dạng (D4).

- **Một khối 256 luồng cho mỗi (head, chuỗi)**: một lưới (grid) `(h_q, batch)`.
- **Các warp chia nhau các key.** Warp logic w (8 warp mỗi khối) nhận các key w, w + 8,
  w + 16, ...
- **Các lane chia nhau chiều head**, theo cách của Fig. 20.10: lane L giữ các phần tử L, L + 32,
  L + 64, L + 96 của q, nên mỗi lần 32 lane đọc một hàng key là một đoạn liên tiếp.
- **Điểm số** của key t là một tổng cây bướm (butterfly) trên warp của bốn tích mỗi lane
  (`__shfl_xor_sync`), để lại tổng ở mọi lane.
- **Mỗi warp giữ bộ (m, D, O~) đang chạy của riêng nó**: giá trị lớn nhất, mẫu số, và đầu ra
  chưa chia cho D. Nó cập nhật chúng từng key một, như Figs. 20.13 đến 20.15 làm từng ô một: giá
  trị lớn nhất mới, co giãn lại D và O~ theo `e^(old max - new max)`, cộng `e^(s - new max)` vào
  D và tích của nó với hàng V vào O~.
- **Cuối cùng 8 warp trộn kết quả qua bộ nhớ chia sẻ (shared memory)** bằng Eqs. 20.4 và 20.6
  (dạng chưa chuẩn hóa): trọng số `e^(m_w - m_all)`, một phép chia (Fig. 20.16). Đây là sơ đồ
  "split-K" mà FlashAttention-1 dùng giữa các warp của một khối (chú thích 7, p. 498; ch20-facts
  A7) và Flash-Decoding dùng giữa các khối luồng (A8).

**Metal**: cùng cấu trúc. `simd_sum` trao mỗi điểm số cho mọi lane, phép trộn đi qua bộ nhớ
threadgroup, và giá trị lớn nhất đang chạy bắt đầu ở một giá trị hữu hạn: dưới fast math mặc định
của Metal, vô cực là không xác định (C5).

## Chỗ khác với trang sách

- Giá trị lớn nhất đang chạy bắt đầu ở `M_INIT = -FLT_MAX/2`, không phải `NEG_INFINITY` của
  sách. Tên đó không được header CUDA nào định nghĩa, và IEEE cho NaN với `(-inf) - (-inf)` (A5).
  Ở đây một warp không thấy key nào (mọi trường hợp `len[b] < 8`) vẫn tham gia phép trộn, nên m của nó
  phải hữu hạn. llama.cpp bắt đầu ở `-FLT_MAX/2`, MLX ở `finite_min`, vì cùng lý do (A5, C5, D1,
  D4).
- Sách lấy mỗi phần tử S từ `cub::WarpReduce` cộng một phép phát từ lane 0 (Fig. 20.12). Cây bướm
  trao tổng cho mọi lane trong một lượt (ch10-facts D1), và `cub::Max()` và `cub::Sum()` mà trang
  sách dùng không còn tồn tại (A2).
- `WARP_SIZE` của sách là warp phần cứng. Ở đây nó là một warp logic 32 lane: mọi lệnh shuffle
  truyền độ rộng `WARP_SIZE`, nên trên một wavefront AMD rộng 64, hai warp logic dùng chung một
  wavefront mà không trộn lẫn (B2). Mặt nạ là `lane_mask_t`, 64 bit trên HIP (B1).
- Trang 502 nói `D *= exp(last_m - m_i[ii])` của Fig. 20.14 tính số hạng `D_B` của Eq. 20.4. Nó
  co giãn lại mẫu số cũ: đó là số hạng `D_A` (notes, "Unclear in the PDF").

## Tệp

- `work/kernel.cu` (CUDA, HIP, các trình giả lập): `warpSumAll`, `kvAppendKernel`,
  `launchKvAppend`, `decodeAttentionKernel`, `launchDecodeAttention`. Các TODO được xếp theo thứ
  tự.
- `work/kernel.metal`: `kv_append`, `decode_attention`. `work/launch.mm`: `encodeKvAppend` và
  `encodeDecodeAttention` đã gắn sẵn các đối số; bạn viết phần dispatch.
- `main.cpp`: chương trình điều khiển (driver) và các tham chiếu trên CPU (`cpuDecodeAttention`,
  và `cpuCausalAttention`, Eq. 20.1 cùng mặt nạ của nó). Các bài kiểm thử nằm trong
  `../shared-g12/decode_test.hpp`, dùng chung với lab 20.5.

## Lệnh

- Kiểm thử: `python3 labs/pmpp.py run 20.4 --backend emu` trước. Rồi `--backend race`: các warp
  trộn kết quả qua bộ nhớ chia sẻ, và ThreadSanitizer chỉ ra hai dòng mã của một rào chắn
  (barrier) bị thiếu. Rồi `python3 labs/pmpp.py run 20.4` trên GPU của bạn.
- Đo: `python3 labs/pmpp.py run 20.4 --bench`.

## Các bài kiểm thử làm gì

- **Nối thêm**: bốn hình dạng, mỗi hình dạng hai lần gọi vào các cache chứa đầy giá trị canh.
  Sau mỗi lần gọi, mọi phần tử của cả hai cache được so sánh từng bit: các hàng tại `pos[b]` chứa
  các hàng mới, mọi chỗ khác giữ giá trị canh của nó. Thông điệp chỉ ra chuỗi, hàng, head và phần
  tử.
- **Decode**: `(h_q, h_kv)` = (8, 8) MHA, (8, 2) GQA, (8, 1) MQA và (6, 2), có g_q bằng 3, không
  phải lũy thừa của hai. Một đến bốn chuỗi mỗi lần gọi, mỗi chuỗi có độ dài riêng: 1, 2, 31, 33,
  100, 1000 và 4097 trên GPU; tới 33, cộng một trường hợp 200, trên `emu`; tới 40 trên `race`.
- Hai loại dữ liệu. Key ngẫu nhiên. Và **late max**: key cuối cùng của mọi chuỗi có điểm số
  khoảng 6 với mọi truy vấn đọc nó, trong khi các key khác có điểm gần 0. Giá trị lớn nhất đang
  chạy nhảy vọt ở tận cuối, chỉ trong một warp, nên trạng thái của các warp khác phải được co
  giãn khoảng e^-6 trong phép trộn. Các trình giả lập chỉ chạy late max.
- Các hàng cache từ `len[b]` trở đi chứa NaN: đọc một hàng như vậy sẽ làm hỏng đầu ra. `out` khởi
  đầu là NaN và theo sau là một dải giá trị canh phải còn nguyên.
- Kết quả được so sánh với một tham chiếu double trên CPU. Dung sai là
  `3e-5 * (1 + |reference|)`; `main.cpp` giải thích vì sao nó đủ chặt để bắt việc bỏ sót một key
  trong 4097.
- **Vòng lặp của mốc dự án**: 16 bước tự hồi quy trên (8, 2) cho các chuỗi có độ dài prompt khác
  nhau (một chuỗi bắt đầu rỗng). Mỗi bước nối thêm và chú ý trong một lần gửi, tại
  `pos[b] = prompt[b] + step`, `len[b] = pos[b] + 1`, và q, k, v của nó là một hàm cố định của
  đầu ra bước trước. Ở mọi bước, mỗi đầu ra phải bằng **hàng cuối cùng của một causal attention
  đầy đủ được tính lại từ đầu** trên mọi token cho tới lúc đó (Eq. 20.1, có cả mặt nạ). Đó là
  khẳng định của mục 20.4, được kiểm thử: với cache, bước đó chỉ cần hàng mới. Cuối cùng cả hai
  cache phải chứa đúng lịch sử, và các hàng sau đó phải vẫn là NaN.

## Trước khi bạn đo hiệu năng, hãy dự đoán

Benchmark đo thời gian riêng kernel decode ở `h_q = h_kv = 32`, chiều head 128 (một lớp MHA cỡ
7B, lựa chọn của chúng tôi), với `len` = 512, 2048, 8192 và 32768 và batch 1 và 8. Nó báo số mili
giây mỗi lần gọi và GB/s tính trên số byte cache mà một lần gọi phải đọc,
`2 x len x h_kv x 128 x 4` cho mỗi chuỗi. Các kích thước có cache không vừa một nửa bộ nhớ mà
thiết bị của bạn báo sẽ bị bỏ qua và được liệt kê. Nó cũng đo thời gian riêng phép nối thêm.

**Dự đoán `ch20/labs/p-lab4-ms`.** Hãy ghi ra, trước khi chạy:

1. Dùng băng thông bạn đã đo ở lab 5.1, thời gian của một lần gọi decode ở `len` = 8192, batch
   1, 32 head kích thước 128 ở fp32.
2. Batch 8 ở cùng độ dài có mất gấp 8 lần thời gian hay không.

Lập luận về cường độ tính toán cho (1): mỗi phần tử 4 byte của K được dùng trong một phép nhân
cộng với q (2 FLOP), và mỗi phần tử của V trong một phép nhân cộng với một xác suất. Tức là khoảng
**0.5 FLOP/B** ở fp32. (Eq. 20.9 ra "about 1" bằng cách đếm một phép toán cho mỗi phần tử ở 1 byte
mỗi phần tử; tài liệu tham khảo [13] ra 1 FLOP/B cho BF16 2 byte, ch20-facts E4. Cùng lập luận,
khác đơn vị.) Con số đó thấp hơn nhiều so với ngưỡng mà lab 5.1 của bạn đã đo, nên thời gian sẽ
gần với số byte / băng thông, và ở `len` 8192 cache là 256 MiB.

Với (2), hãy cân nhắc: mỗi chuỗi có cache riêng, nên batch 8 đọc gấp 8 lần số byte (Fig. 20.17:
attention không được lợi từ batching); nhưng batch 1 chỉ khởi chạy 32 khối còn batch 8 khởi chạy
256, và một kernel không giữ đủ yêu cầu đang bay thì không thể đạt băng thông nó được hứa (các tác
giả của Flash-Decoding nêu điểm này về các batch nhỏ, A8).

Các con số được in ra là phép đo trên máy của bạn, không phải kỳ vọng. Trang bài học đặt GB/s cạnh
băng thông lab 5.1 của bạn.

## Vượt ra ngoài sách (của khóa học này)

- **Split-KV.** Với ít chuỗi và ít head, Flash-Decoding (FlashAttention 2.2) còn chia các key
  giữa các khối luồng và trộn các kết quả từng phần trong một kernel thứ hai bằng cùng phép trộn
  Eq. 20.4 / 20.6; `flash_attn_with_kvcache` đưa nó ra dưới dạng `num_splits` (A8), và
  `flash_attn_combine_results` của llama.cpp chính là phép trộn đó (D1).
- **Kiểm tra với PyTorch.** `scaled_dot_product_attention(..., is_causal=True)` căn mặt nạ theo
  góc trên trái khi truy vấn ngắn hơn các key, nên một truy vấn một token sẽ chỉ thấy key 0. Đừng
  truyền mặt nạ nào cho một truy vấn đơn, như lab này làm, hoặc dùng biến thể góc dưới phải
  (D5).
