> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 12.4: nén gọn danh sách ứng viên (một phần của mốc dự án M5)

Sách: bộ lọc ổn định (mục 12.5, Figs. 12.7 và 12.8, pp. 297-298) và phép bỏ phiếu warp cùng
tổng tiền tố nhị phân của nó (mục 12.3, Fig. 12.3, pp. 292-293). Việc mang theo các cặp (cột,
giá trị), một khối cho mỗi hàng và ứng dụng vào lấy mẫu không có trong sách: chúng là của khóa
học này. Một mục đứng riêng như (C4) hay (D2) dưới đây nằm trong `notes/today/ch12-facts.md`.

Một bộ lấy mẫu chỉ giữ một số token (top-k, top-p, min-p) trước hết cần, với mỗi hàng logits,
danh sách các token vượt qua một ngưỡng. Hãy viết danh sách đó. Với một tensor logits float32
`[rows x cols]`, thứ tự hàng (row-major), batch đứng trước, và một ngưỡng cho mỗi hàng:

- `launchCompactCandidates(logits, threshold, outIdx, outVal, counts, rows, cols)` ghi, với
  mọi hàng r, các cặp (cột, logit) có `logit >= threshold[r]`, theo thứ tự cột tăng dần, vào
  `outIdx[r * cols + 0 ..)` và `outVal[r * cols + 0 ..)`, và số lượng của chúng vào
  `counts[r]`.

Một khối 256 luồng cho mỗi hàng. Khối đi qua hàng của nó theo từng vòng `blockDim.x` cột, mỗi
luồng một cột mỗi vòng, nên mỗi vòng đọc một đoạn liên tiếp (được hợp nhất). Ở mỗi vòng, một
luồng tìm vị trí của cặp nó giữ từ ba con số: số cặp được giữ ở các vòng trước, số cặp được giữ
bởi các warp trước trong vòng này, và số cặp được giữ bởi các lane thấp hơn trong warp của nó.
Số cuối đến từ một lần bỏ phiếu warp và tổng tiền tố nhị phân của Fig. 12.3; số ở giữa đến từ
một phép scan trên tổng của các warp trong bộ nhớ chia sẻ (shared memory); số đầu tiên là một
số đếm cộng dồn mà mọi luồng đều giữ. Không có phép toán nguyên tử (atomic) toàn cục, và không
khối nào phải chờ khối khác (mục tiếp theo). Đệm các cặp của một vòng trong bộ nhớ chia sẻ trước
khi ghi chúng ra, như Fig. 12.9 làm, là được phép nhưng không bắt buộc.

- Sửa `work/kernel.cu` (CUDA; cũng được biên dịch cho HIP và các trình giả lập) và
  `work/kernel.metal` + `work/launch.mm` (Metal). Các tên là cố định: chương trình điều khiển
  (driver) gọi `launchCompactCandidates`, kernel Metal là `compact_candidates`, và encoder là
  `encodeCompactCandidates`.
- Được cung cấp, không sửa: `provided/rowwise_serial.cu`, `rowwise_serial.metal` và
  `rowwise_serial.mm` (`launchCompactSerial`, `compact_serial`, `encodeCompactSerial`), một
  luồng cho mỗi hàng chạy vòng lặp min-p phía host của llama.cpp (D3), mốc so sánh của
  benchmark.
- Kiểm thử: `python3 labs/pmpp.py run 12.4`. Hãy bắt đầu với `--backend emu`.
- Race condition: `python3 labs/pmpp.py run 12.4 --backend race` (xem "Backend race" bên dưới).
- Đo: `python3 labs/pmpp.py run 12.4 --bench`.

## Vì sao một khối cho mỗi hàng, và không dùng atomic

- **Thứ tự.** Các bộ lọc không ổn định của sách (Figs. 12.2 đến 12.6) lấy vị trí đầu ra từ một
  bộ đếm nguyên tử, nên thứ tự của các phần tử còn lại phụ thuộc vào luồng nào đến trước (ch09
  A3). Một bộ lấy mẫu đi qua các ứng viên theo thứ tự, hoặc phá thế hòa theo cột, cần cùng một
  đáp án ở mọi lượt chạy, nên lab này yêu cầu bộ lọc ổn định: vị trí của mỗi cặp được giữ là số
  cặp được giữ đứng trước nó (mục 12.5). `torch.topk` của PyTorch từ chối atomic vì cùng lý do
  đó (D4).
- **Chờ đợi.** Bộ lọc ổn định của sách thực hiện phép scan trên toàn lưới (grid)
  (`gridExclusiveScan`, Fig. 12.8), trong đó một khối chờ các khối đứng trước nó. CUDA ghi trong
  tài liệu khi nào việc chờ đó an toàn (A10); AMD không ghi bảo đảm tiến triển nào giữa các
  workgroup (B6) và Apple cũng không giữa các threadgroup (C4). Một khối cho mỗi hàng không cần
  chờ gì cả: số đếm cộng dồn của hàng không bao giờ rời khỏi khối. Đó là thiết kế mà top-k trên
  Metal của llama.cpp dùng (một threadgroup cho mỗi hàng, D2), và nó hợp với bố cục batch nhân
  từ vựng.
- **Tất định không có nghĩa là ổn định.** Phần nén gọn của top-k trên Metal trong llama.cpp là
  tất định nhưng ghi các phần tử còn lại theo thứ tự ưu tiên lane, không theo thứ tự cột (theo
  cách chúng tôi đọc mã nguồn, D2). Các bài kiểm thử ở đây kiểm tra thứ tự cột.

## Hợp đồng

- `rows >= 1`, `cols >= 1`, giá trị bất kỳ: không chỉ bội của 32 hay 256.
- Mọi logit và mọi ngưỡng đều hữu hạn. NaN và vô cực nằm ngoài hợp đồng được kiểm thử, như ở lab
  10.3 (mục cuối).
- Mỗi hàng có `cols` vị trí trong `outIdx` và `outVal`, đủ cho trường hợp mọi cột đều đạt. Các
  vị trí từ `counts[r]` trở đi không được ghi: bên gọi có thể đã đặt thứ gì đó ở đó.
- `outIdx` là `int`, `outVal` là `float`, `counts` là `unsigned int`.

## Các bài kiểm thử kiểm tra gì

- Hình dạng: rows 1, 4 và 7 nhân cols 1, 2, 31, 32, 33, 255, 256, 257, 1000, 4096, 32000,
  50257 và 128256 (ba giá trị cuối có kích thước bằng từ vựng). Trình giả lập dùng rows 1 và 3
  và cols tới 4099; backend race dùng rows 2 và cols tới 1000.
- Logits phân bố đều trong [-12, 12]. Mỗi hàng của một batch có ngưỡng riêng, nên một kernel
  đọc nhầm ngưỡng của hàng khác sẽ giữ sai số cặp:
  - ngay trên giá trị lớn nhất của hàng (không giữ gì), bằng giá trị lớn nhất (chỉ giữ các giá
    trị lớn nhất), ngay dưới giá trị nhỏ nhất (giữ tất cả);
  - các phân vị trên CPU giữ khoảng 1, 10 và 50 phần trăm;
  - min-p 0.1, tính trên host như llama.cpp làm, `max + ln 0.1` (D3);
  - các giá trị được cài đặt đúng bằng ngưỡng tại các cột (1, 2), (31, 32), (63, 64),
    (255, 256) và (0, cols - 1): các lane kề nhau, vắt qua ranh giới của warp 32 lane và warp
    64 lane, vắt qua ranh giới vòng, và ở cả hai đầu. Chúng bắt lỗi dùng `>` thay cho `>=` và
    lỗi lệch một ở mọi chỗ nối;
  - một hàng có mọi phần tử còn lại nằm trong vòng 256 cột cuối cùng, bắt một vòng lặp bỏ qua
    vòng cuối chưa đầy và một số đếm được ghi từ sai vòng.
- Với mọi hàng: `counts[r]` chính xác; `outIdx[0 .. counts[r])` bằng danh sách trên CPU, vốn
  tăng ngặt; `outVal` bằng từng bit với logits tại các cột đó; mọi vị trí quá `counts[r]` không
  đổi (chỉ số -1, và các bit của giá trị được kiểm tra trên host); không có gì được ghi quá cuối
  bất kỳ đầu ra nào.
- Mốc so sánh được cung cấp cũng được kiểm thử, để một tệp được cung cấp bị hỏng sẽ lộ ra ở đây.

Phép so sánh là chính xác, không có dung sai, điều mà các lab của khóa học này thường tránh trên
Metal (chế độ toán mặc định của nó không phải IEEE nghiêm ngặt, ch05-facts C7). Ở đây không phép
tính số học nào chạm vào một giá trị: ngưỡng được tính trên host, kernel so sánh hai số thực hữu
hạn và sao chép một số. Các chế độ toán nới lỏng phép tính số học và cách xử lý NaN, INF và số
không có dấu; một phép so sánh và một phép sao chép không liên quan đến điều nào trong đó (theo
cách chúng tôi đọc C7), nên so sánh bằng chính xác là phép kiểm thử đúng.

## Backend race

`emu` chạy từng luồng một và không thể thấy một race condition tình cờ để lại giá trị đúng.
`race` chạy các luồng của một khối cùng lúc dưới ThreadSanitizer. Trường hợp mà lab này được xây
dựng cho: tổng của các warp ở một vòng nằm trong cùng mảng chia sẻ với tổng của vòng sau. Một
warp đã xong vòng của nó có thể ghi tổng tiếp theo trong khi một warp chậm hơn vẫn đang cộng tổng
của vòng này, nên phải có gì đó sắp thứ tự cho hai việc đó. Trên GPU lỗi đó có thể vượt qua mọi
bài kiểm thử, vì nó có gây hại hay không còn tùy vào việc lập lịch. Khi lab này được xây dựng,
lời giải bị gỡ bỏ phần sắp thứ tự đó đã vượt qua mọi bài kiểm thử `emu`, và `race` đã báo lỗi.

## Mốc dự án: module X1 kết hợp phần này như thế nào

Lab này viết danh sách ứng viên cho một ngưỡng được cho sẵn. Ngưỡng đến từ đâu là việc của bộ
lấy mẫu:

- **Min-p.** Giá trị lớn nhất của hàng `m` từ lab 10.3 cho ngưỡng `m + ln p`: một token đạt khi
  `exp(logit - m) >= p`, tức là khi xác suất của nó ít nhất bằng p lần xác suất lớn nhất.
  llama.cpp tính đúng điều này, `max_logit + logf(p)`, trên logits, nên không cần softmax để
  lọc (D3).
- **Top-k.** Ngưỡng là logit lớn thứ k, đến từ một phép chọn (module X1, mục X1.4: một radix
  select, được đo so với phép sắp xếp của chương 14). Các giá trị hòa tại ngưỡng đều được giữ,
  theo thứ tự cột, nên `counts[r]` có thể vượt quá k. Để giữ đúng k với quy tắc phá hòa của
  llama.cpp, "the lowest columns win" (D1), hãy bỏ các cặp dư có giá trị bằng ngưỡng, bắt đầu từ
  cuối danh sách: các giá trị hòa nằm theo thứ tự cột, nên những cặp bị bỏ là các cột cao nhất.
  Đừng chỉ cắt danh sách tại k: một cặp lớn hơn ngưỡng có thể đứng sau một giá trị hòa.
- **Top-p** (của khóa học này). Khi các ứng viên đã được sắp xếp theo logit (chương 14), một
  phép scan trên xác suất của chúng (chương 11) tìm ra chỗ tổng tích lũy đạt p; phần được giữ là
  một tiền tố của danh sách đã sắp xếp.
- **Vì sao là một danh sách và một số đếm, không phải một mặt nạ.** Các bộ lấy mẫu được xây
  dựng dưới dạng đồ thị tính toán (các bộ lấy mẫu backend của llama.cpp, mlx-lm) giữ toàn bộ hàng
  và đặt các phần tử bị loại thành âm vô cực (D3, D6). MLX tự nói lý do: nó có "limited support
  for operations for which output shapes are dependent on input data" (D6). Một kernel viết tay
  có thể trả về một danh sách và một số đếm, và đó là điều lab này làm.

## Trước khi bạn đo hiệu năng

Hãy ghi ra hai dự đoán (bài học yêu cầu chúng dưới tên `ch12/labs/p-lab4-batch1` và
`ch12/labs/p-lab4-fraction`):

1. Ở 1 hàng gồm 128,256 cột với 1 phần trăm được giữ: thời gian của bản một luồng cho mỗi hàng
   được cung cấp chia cho thời gian của bản một khối cho mỗi hàng của bạn.
2. Ở 64 hàng: thời gian của bản một khối cho mỗi hàng của bạn khi giữ 50 phần trăm chia cho
   thời gian của nó khi giữ 1 phần trăm.

Ý của mục 10.10 cũng áp dụng ở đây: ở batch 1, một khối cho mỗi hàng chỉ dùng một khối của một
GPU có chỗ cho rất nhiều khối. Hãy tự hỏi thêm điều gì thay đổi giữa 1 và 50 phần trăm: số byte
được ghi, hay số vòng?

Benchmark đo thời gian rows 1, 8 và 64 nhân cols 32,000 và 128,256, với ngưỡng từ các phân vị
trên CPU giữ 0.1, 1, 10 và 50 phần trăm, cho kernel của bạn ("block") và mốc so sánh
("serial"). Thời gian mỗi lần gọi là kết quả tốt nhất trong 5 lượt đo, mỗi lượt lặp lại lời gọi
cho đến khi nó kéo dài ít nhất 2 ms. GB/s tính lưu lượng tối thiểu: 4 byte cho mỗi logit được
đọc, 8 cho mỗi cặp được giữ được ghi và 4 cho mỗi hàng cho số đếm. Số hàng mỗi giây là số hàng
chia cho thời gian mỗi lần gọi. Mọi kết quả được đo giờ đều được kiểm tra trước khi báo cáo. Các
con số là phép đo trên máy của bạn, không phải mục tiêu.

## NaN và vô cực

Nằm ngoài hợp đồng được kiểm thử, vì các lý do lab 10.3 nêu: dưới fast math của Metal, hành vi
của NaN và vô cực là không xác định (ch10-facts C3), và Apple mô tả chế độ toán mặc định theo ba
cách khác nhau (ch05-facts C7). Trên CPU, với phép so sánh IEEE, `>=` không bao giờ giữ một logit
NaN và một ngưỡng NaN không giữ gì; dưới fast math một GPU không nhất thiết đồng ý. Các logit bị
che bằng âm vô cực sẽ quay lại ở chương 20.

## Tính khả chuyển

- HIP: `kernel.cu` được biên dịch qua `pmpp/vendors/hip.h`. Phép bỏ phiếu là `__ballot_sync`
  với một mặt nạ có kiểu riêng, 64 bit trên HIP (infra-facts B3), và popcount phải khớp với kiểu
  đó: `__popc` nhận 32 bit, nên trên một wavefront rộng 64 nó sẽ âm thầm chỉ đếm các lane 0 đến
  31; `__popcll` đếm cả 64 (B1). Mặt nạ các lane phía dưới là `((lane_mask_t)1 << lane) - 1`,
  không bao giờ là `1 << lane`, vốn không xác định khi quá lane 31 (A3, B1). llama.cpp phòng cùng
  cái bẫy đó bằng cách xóa popcount cho mọi kiểu khác (D1). Chưa được biên dịch hay chạy trên phần
  cứng AMD ở đây: unverified.
- `__activemask()` bên trong nhánh của Fig. 12.3: NVIDIA ghi trong tài liệu rằng nó không cho
  biết những lane nào đã đi vào nhánh (A1), header của AMD nói là có (B1), Apple không nói gì
  (C3). Phép bỏ phiếu ở đây được mọi lane thực hiện, trong luồng điều khiển đồng nhất, nơi cả ba
  đều đồng ý.
- Metal: `simd_ballot` trả về một `simd_vote` 64 bit bất kể độ rộng SIMD group, và trên phần
  cứng có ít hơn 64 lane các bit cao của nó là không xác định (C2). Thứ hạng,
  `popcount(vote & lanesBelow)`, không bao giờ đọc chúng; tổng của một SIMD group thì có, nên hãy
  đếm nó bằng `simd_sum(keep ? 1u : 0u)` thay vào đó (C1, C2). `simd_prefix_exclusive_sum` xếp
  hạng trong một lời gọi (C1).
- Metal: `launch.mm` đọc độ rộng SIMD group từ pipeline (`threadExecutionWidth`), kiểm tra rằng
  256 luồng tạo thành các SIMD group trọn vẹn với số group không nhiều hơn số lane (ch11-facts
  C2: điều mà một phép scan trên các tổng bởi một SIMD group cần), và đặt kích thước mảng
  threadgroup chứa các tổng từ đó, một bội của 16 byte (ch05-facts C8).
