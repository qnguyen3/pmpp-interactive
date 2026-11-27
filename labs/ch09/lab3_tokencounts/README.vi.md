> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 9.3: đếm token cho các hình phạt lặp lại

Sách: các mục 9.2 đến 9.5 (pp. 203-217), Figs. 9.6, 9.9 và 9.14. Việc đếm id token,
lần cập nhật ở bước decode và công thức hình phạt không có trong sách. Chúng là phần đếm
của mốc dự án M5; module X1 ghép nó với lấy mẫu top-k và top-p.

Một bộ lấy mẫu có hình phạt tần suất và hình phạt hiện diện cần biết, với mỗi yêu cầu trong
batch, mỗi id từ vựng đã xuất hiện bao nhiêu lần trong ngữ cảnh của yêu cầu đó. Đó là một
histogram có các bin là cả một bộ từ vựng: 50,257 id cho GPT-2, 151,936 cho Qwen3-8B,
201,088 cho gpt-oss-20b (`notes/today/ch09-facts.md` D7). vLLM giữ đúng thứ này trên
GPU, một tensor `int32` kích thước `[max_num_reqs, vocab_size]` được cập nhật bằng các phép cộng nguyên tử (D6).

## Trước khi viết: kiểu riêng tư hóa nào vẫn còn áp dụng được?

Chương 9 chữa tranh chấp bằng cách cho các tập con luồng (thread) những bản sao riêng của các bin. Trước khi
bạn mở `work/`, hãy viết ra với mỗi mục dưới đây liệu nó có áp dụng được cho các bin cỡ bộ từ vựng
hay không, và một câu giải thích vì sao. Dùng con số thiết bị của bạn từ
`python3 labs/pmpp.py doctor` (`sharedMemPerBlock` trên CUDA và HIP,
`maxThreadgroupMemoryLength` trên Metal) ở chỗ cần thiết.

1. Một bản sao riêng cho mỗi khối (block) trong bộ nhớ chia sẻ (shared memory) (Fig. 9.10, p. 214).
2. Một bản sao riêng cho mỗi khối trong bộ nhớ toàn cục (global memory) (Fig. 9.9, p. 212).
3. Một bản sao riêng cho mỗi thread block cluster, trong distributed shared memory (p. 213).
4. Riêng tư hóa (privatization) ở mức luồng: gộp các dải giá trị bằng nhau trong một thanh ghi (register) (Fig. 9.15, p. 218).
5. Batch: mỗi yêu cầu có hàng số đếm của riêng nó.
6. Không đếm lại: trong lúc decode, mỗi bước thêm một token cho mỗi yêu cầu.

Câu trả lời của bạn quyết định `countTokens` trông như thế nào. Câu trả lời của chúng tôi ở cuối tệp này.

## Phần của bạn

Ba kernel, mỗi kernel có một hàm khởi chạy. Thứ tự hàng (row-major), batch trước, id int32, số đếm uint32.

- **`countTokens`**: `counts[r*vocab + t]` = số lần id `t` xuất hiện trong `len` token của hàng
  `r`. Bên gọi đã đặt `counts` về 0. Bất kỳ `len` nào từ 1 trở lên; `vocab` từ
  1,000 (các phép kiểm thử nhỏ) đến 201,088. Dùng API của sách từ mục 9.2,
  `cuda::atomic_ref` và `fetch_add` với `cuda::memory_order_relaxed`, với phạm vi mà thiết kế
  của bạn cần.
- **`appendTokens`**: một bước decode. Hàng `r` nhận thêm một token, `newTokens[r]`; hãy cộng nó vào
  số đếm của hàng đó. Sau đó, các số đếm phải bằng số đếm của toàn bộ `len + 1` token.
- **`applyPenalties`**: với mọi phần tử trong logits của batch,
  `logit -= count * frequencyPenalty + (count > 0) * presencePenalty`. Đây là hình phạt tần suất
  và hiện diện của llama.cpp (D4). Khi cả hai hình phạt bằng 0, logits không được thay đổi.

Hình phạt lặp lại có xét dấu (chia một logit dương, nhân một logit âm) không có ở
đây. Module X1 thêm nó, trong mục X1.3 "Penalties from token counts" và lab X1.3 của nó
(`notes/today/x1-facts.md` D2, D9).

## Các tệp

- Sửa `work/kernel.cu` (CUDA, HIP và cả hai trình giả lập): `countTokens`, `appendTokens`,
  `applyPenalties` và
  - `launchCountTokens(const int* tokens, unsigned int* counts, int batch, int len, int vocab)`
  - `launchAppendTokens(const int* newTokens, unsigned int* counts, int batch, int vocab)`
  - `launchApplyPenalties(float* logits, const unsigned int* counts, int batch, int vocab, float frequencyPenalty, float presencePenalty)`
- Sửa `work/kernel.metal` (`count_tokens`, `append_tokens`, `apply_penalties`) và
  `work/launch.mm` (`encodeCountTokens`, `encodeAppendTokens`, `encodeApplyPenalties`).
  Chữ ký kernel và cách gắn buffer đã được cho sẵn; kích thước lưới (grid) là việc của bạn.
- Được cung cấp, không sửa: `provided/count_pool.cu`, `provided/count_pool.metal` và
  `provided/count_pool.mm`. Đó là thiết kế của Fig. 9.9 áp dụng cho đếm token: mỗi khối
  nhận một bản sao riêng của cả bộ từ vựng trong một pool ở bộ nhớ toàn cục, đếm ở đó bằng
  phép toán nguyên tử (atomic) phạm vi khối, rồi ghi nhận mọi bin khác 0 vào hàng của yêu cầu. Mỗi token một luồng,
  256 luồng mỗi khối, như trong Fig. 9.9. Pool được hàm khởi chạy đặt về 0, và việc đó
  được tính thời gian cùng với nó. Nó là đối chứng của benchmark; các phép kiểm thử cũng chạy nó, nên một
  bản sao bị hỏng sẽ lộ ra.

## Lệnh

- Kiểm thử: `python3 labs/pmpp.py run 9.3 --backend emu` trước, rồi `python3 labs/pmpp.py run 9.3`.
- Phép toán nguyên tử: `python3 labs/pmpp.py run 9.3 --backend race`.
- Đo: `python3 labs/pmpp.py run 9.3 --bench` (chỉ trên thiết bị thật).
- Lời giải tham chiếu (solution): thêm `--solution`.

## Các phép kiểm thử làm gì

- Các hình dạng (batch, len, vocab): trên `emu` và `race`, (1, 1, 1000), (3, 100, 1000) và
  (2, 64, 50257). Thiết bị thật thêm (1, 1, 50257), (3, 1000, 50257), (2, 4096, 151936),
  (4, 777, 201088) và (8, 2048, 151936).
- Bốn phân bố token có seed cho mỗi hình dạng: phân bố đều trên bộ từ vựng; kiểu Zipf, trong đó
  một vài id chiếm phần lớn hàng; một id lặp lại suốt cả hàng (mọi luồng đều đập vào
  cùng một bin); và các hàng chứa id 0 và vocab-1, giúp bắt lỗi sai độ lệch hàng.
- `countTokens`: các số đếm phải khớp chính xác với bản tham chiếu CPU. Một lần thất bại chỉ ra
  (hàng, id) sai đầu tiên và so sánh tổng mọi số đếm với bản tham chiếu: một tổng
  quá nhỏ nghĩa là có lần cập nhật bị mất; tổng đúng nhưng bin sai nghĩa là chỉ số sai.
- `countTokens then appendTokens`: thêm một token mỗi hàng phải cho số đếm của toàn bộ
  `len + 1` token, một cách chính xác. Nếu `countTokens` sai, `appendTokens` vẫn được kiểm tra,
  từ số đếm tham chiếu, và thông báo nói cái nào trong hai cái thất bại.
- `applyPenalties`, một lần cho mỗi hình dạng, trên số đếm của một batch kiểu Zipf (nhiều số 0, một vài
  số đếm lớn) và logits trong [-8, 8]: với các hình phạt (0.5, 0.25), kết quả phải nằm trong
  1e-5 lần độ lớn các số hạng của bản tham chiếu CPU (Metal không mặc định dùng số học
  IEEE nghiêm ngặt); với (0, 0), mọi logit phải trở về không đổi đến từng bit.
- Chương trình điều khiển (driver) đặt `counts` về 0 trước mỗi `countTokens`, như hợp đồng nói bên gọi
  làm: buffer trên device bắt đầu với dữ liệu rác (của Metal được điền bằng các byte `0xA5`).

Mỗi backend có thể bắt được gì:

- `emu` chạy từng luồng một dưới AddressSanitizer, theo ba thứ tự cố định. Nó tìm ra
  chỉ số sai, độ lệch hàng sai một đơn vị, và các lần đọc hoặc ghi vượt quá một buffer. Nó
  không thể thấy một phép toán nguyên tử bị thiếu: một `++` thường luôn cho số đếm đúng ở đó.
- `race` chạy các luồng của một khối, và các khối nhỏ, đồng thời dưới
  ThreadSanitizer. Một `++` thường trong `countTokens` được báo cáo cùng dòng mã nguồn của nó khi hai
  luồng chạy cùng lúc đập vào cùng một id, điều mà các hàng kiểu Zipf và lặp lại khiến
  rất dễ xảy ra. `appendTokens` với phép tăng thường không bị báo cáo, và không được bị báo: mỗi
  hàng do một luồng sở hữu.
- Metal là một GPU thật. Trên máy mà lab này được dựng, một `+= 1` thường trong
  `count_tokens` đã làm mất phần lớn các lần cập nhật của các hàng lặp lại, và mất những lần khác nhau ở
  mỗi lần chạy.
- Không gì ở đây bắt được phạm vi sai. Nếu nhiều khối dùng chung một hàng, một
  lần cập nhật `cuda::thread_scope_block` trên hàng đó là một data race và là hành vi không xác định
  (ch09-facts A2), nhưng cả hai trình giả lập đều bỏ qua phạm vi, và Metal không có phạm vi. Hãy kiểm tra điều đó bằng cách
  đọc mã của bạn.

## Các backend khác nhau ở đâu (không có trong sách)

- **CUDA**: `cuda::atomic_ref<T, Scope>` mặc định dùng phạm vi system và `fetch_add` mặc định dùng
  `seq_cst` khi bạn bỏ chúng đi (ch09-facts A1). Hãy truyền cả hai, như sách làm.
- **HIP**: `<cuda/atomic>` giữ nguyên tên qua libhipcxx, ROCm 10.1 trở về sau, và
  bản dựng HIP phải liên kết nó (infra-facts B2). Phạm vi là cùng một đối số template
  (ch09-facts B1). Không được chạy trên phần cứng AMD ở đây: unverified.
- **Metal**: không có góc nhìn atomic nào cho một mảng thường. `count_tokens` khai báo các số đếm
  là `device atomic_uint*` và cập nhật chúng bằng
  `atomic_fetch_add_explicit(..., memory_order_relaxed)`; một phép đọc-sửa-ghi không nhận
  phạm vi (ch09-facts C1). Kiểu atomic thuộc về khai báo của kernel, không thuộc về
  buffer, vốn chứa các `uint` thường: `append_tokens` và `apply_penalties` khai báo
  cùng buffer đó là `device uint*`. PyTorch cũng gắn một tensor `uint32` thường thành
  `device atomic_uint*` (C1).
- **Pool được cung cấp trên Metal**: việc đặt về 0 là một kernel riêng, được dispatch trước
  kernel đếm trong cùng một compute pass. Pass đó tuần tự, nên dispatch thứ hai
  thấy các lần ghi của dispatch thứ nhất (ch02-facts C9). Lần đọc thường một bin riêng trong
  phần ghi nhận trở thành `atomic_load_explicit`, và rào chắn (barrier) giữa hai pha sắp thứ tự
  bộ nhớ device, `threadgroup_barrier(mem_flags::mem_device)` (ch05-facts C2).

## Trước khi đo hiệu năng, hãy dự đoán

Benchmark chạy batch 8, len 4096, vocab 151,936, trên các hàng phân bố đều và kiểu Zipf. Pool
được cung cấp dùng 256 luồng mỗi khối và mỗi luồng một token: 16 khối mỗi hàng,
128 khối, mỗi khối đếm 256 token vào bản sao 151,936 bin của riêng nó. Hãy viết những điều này
ra trước. Bài học hỏi dự đoán thứ nhất dưới id ghi trong ngoặc.

1. `count_pool` nhanh hơn hay chậm hơn các phép toán nguyên tử trực tiếp của bạn vào hàng của mỗi yêu cầu, và
   khoảng bao nhiêu lần? Hãy lập luận từ những gì một khối của pool làm: bao nhiêu lần cập nhật,
   so với bao nhiêu bin nó phải đặt về 0 và đọc lại để ghi nhận
   (`ch09/labs/p-lab3-pool`).
2. Một bước `appendTokens` rẻ hơn bao nhiêu lần so với việc đếm lại toàn bộ ngữ cảnh
   bằng `countTokens`? Hãy viết ra tỉ lệ công việc trước, rồi tỉ lệ thời gian bạn
   kỳ vọng, và nói vì sao chúng có thể khác nhau.

## Đọc kết quả đầu ra

Benchmark trước hết in kích thước của một bản sao riêng của bộ từ vựng cạnh bộ nhớ
trên chip cho mỗi khối của thiết bị, theo như thiết bị báo cáo. Rồi một bảng, hai khối
hàng (phân bố đều, kiểu Zipf). `pmpp.py` cũng lưu các hàng vào `reports/runs/`, nơi ứng dụng
đọc chúng.

- `count (yours)`: một lần khởi chạy `countTokens`. `counts` được đặt về 0 trước mỗi lần chạy, ngoài
  vùng đo thời gian. `tokens/s` là batch x len chia cho thời gian.
- `count_pool (provided)`: việc đặt pool về 0 và kernel của Fig. 9.9, cả hai đều được đo thời gian. Hàng này cũng
  ghi số khối mỗi hàng và số byte của pool. Nó chỉ chạy nếu pool chiếm nhiều nhất
  một phần tư bộ nhớ mà thiết bị của bạn báo cáo.
- `append (yours)`: một bước decode cho cả batch. `tokens/s` là batch chia cho
  thời gian.
- `penalties (yours)`: mọi logit của batch với các hình phạt (0.5, 0.25). `elements/s` là
  batch x vocab chia cho thời gian. `GB/s` đếm 12 byte mỗi logit (đọc logit, đọc
  số đếm của nó, ghi logit): mức tối thiểu một kernel phải di chuyển, không phải lượng mà hệ thống bộ nhớ
  đã di chuyển. Cùng các buffer được dùng ở mọi lần chạy, nên đây không phải phép đo
  băng thông bộ nhớ của bạn; lab 5.1 đo điều đó.
- Dòng cuối của mỗi khối chia các thời gian đo được: đếm lại so với một lần append, và
  pool so với lần đếm của bạn.
- Mọi thời gian là lần tốt nhất trong ít nhất 10 lần chạy, sau một phần tư giây chạy không đo
  để kernel đầu tiên không phải trả thêm chi phí đánh thức thiết bị. Trên Metal, một thời gian là
  thời gian từ đầu đến cuối trên GPU của một command buffer. Trên CUDA và HIP, nó là thời gian đồng hồ
  thực trên host bao quanh lệnh khởi chạy và một lần đồng bộ, nên chi phí cố định của một lần khởi chạy được tính vào.

Các số đếm mà `count` và `count_pool` để lại được kiểm tra với bản tham chiếu CPU
sau khi đo thời gian; một kết quả sai được báo là sai, không kèm thời gian. Các con số
là phép đo trên máy này ở kích thước này. Chúng không phải là thuộc tính của các kernel.

## Đi xa hơn

- Cách chữa của mục 9.5 cho pool là gộp việc (thread coarsening): ít khối hơn, mỗi khối đếm nhiều token hơn,
  nghĩa là ít bản sao phải đặt về 0 và ghi nhận hơn. Dùng thời gian `count_pool` bạn đo được, hãy tính
  xem 4, 16 và 64 token mỗi luồng sẽ bớt được bao nhiêu ở kích thước của benchmark, và
  số khối ít nhất, mỗi hàng một khối, sẽ làm pha đếm tốn bao nhiêu. Có hệ số gộp việc
  nào đưa pool ngang bằng với phép toán nguyên tử trực tiếp không?
- llama.cpp đếm token cho hình phạt trên CPU bằng một hash map trên một cửa sổ gồm
  `penalty_last_n` token cuối, thêm token mới và bỏ token rời khỏi
  cửa sổ (D4). Hãy viết `removeTokens`, phép giảm cho một cửa sổ trượt, và nói vì sao nó
  cũng không cần phép toán nguyên tử.

## Câu trả lời của chúng tôi cho câu hỏi đầu tiên

Hãy đọc phần này sau khi bạn đã viết câu trả lời của mình.

1. **Bộ nhớ chia sẻ: không.** Một bản sao `uint32` là 196.3 KiB với bộ từ vựng của GPT-2, 593.5 KiB
   với của Qwen3 và 785.5 KiB với của gpt-oss. Chỉ bản của GPT-2 vừa ở đâu đó, sau tùy chọn opt-in trên một
   vài compute capability của NVIDIA; hai bộ từ vựng hiện nay không vừa với giới hạn mỗi khối nào có trong tài liệu
   trên cả ba backend. Giảm một nửa bộ đếm xuống `uint16` cũng không
   cứu được: khi đó chỉ giới hạn opt-in của một compute capability chứa được bản của Qwen3, và
   phép toán nguyên tử 16 bit được giả lập trên CUDA và HIP và không có trên Metal (D7, A1, C2). Hãy so
   với con số `doctor` của bạn; benchmark in cả hai.
2. **Bản sao mỗi khối trong bộ nhớ toàn cục: chúng chạy được, nhưng tốn cả một bộ từ vựng cho mỗi khối.**
   Mỗi khối đặt về 0 và ghi nhận mọi bin trong bản sao của nó, bất kể nó đã đếm gì. Với 256
   token mỗi khối và 151,936 bin, chi phí đó lấn át hẳn số lần cập nhật mà nó tiết kiệm. Benchmark
   đo điều đó.
3. **Cluster: không phải ở đây.** Distributed shared memory cần CUDA compute capability 9.0 trở
   lên (ch05-facts A6), thiết bị HIP báo hỗ trợ cluster launch (infra-facts B4), và
   Metal không có. Không trình giả lập nào mô hình hóa cluster (`labs/README.md`).
4. **Gộp dải: được ít.** Fig. 9.15 có lợi khi có các dải dài một giá trị trong các
   phần tử của một luồng. Với chỉ số xen kẽ, các phần tử đó cách nhau một khối trong
   hàng, và văn bản hiếm khi lặp lại một token ở đó (cách hiểu của chúng tôi, chưa đo). Các hàng kiểm thử
   lặp lại là ngoại lệ: một mô hình bị kẹt trong một vòng lặp.
5. **Batch: có, và miễn phí.** Các hàng của những yêu cầu khác nhau không bao giờ dùng chung một bin. Hàng của mỗi
   yêu cầu vốn đã là một histogram riêng mà không ai trộn, nên tranh chấp chỉ
   giới hạn trong các khối của một hàng, và phép toán nguyên tử cần phạm vi device chỉ vì
   nhiều khối dùng chung hàng đó.
6. **Không đếm lại: có, cho decode.** Sau khi prompt được đếm một lần, mỗi bước thêm
   một token cho mỗi yêu cầu. Mỗi hàng một luồng sở hữu hàng đó, quy tắc owner-computes ở
   p. 201 lại đúng, và một phép tăng thường là đúng. llama.cpp giữ số đếm của nó theo
   cách này trên CPU (D4).
