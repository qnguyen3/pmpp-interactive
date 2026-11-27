> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 10.3: softmax và argmax trên logits (mốc dự án M4)

Sách: phép rút gọn (reduction) hai giai đoạn ở cấp warp (mục 10.8, Fig. 10.16, p. 243), vòng lặp
gộp việc (mục 10.10, Fig. 10.20, p. 247) và phép rút gọn max của bài tập 4 (p. 250). Softmax, argmax
và bộ giải mã không có trong sách: chúng là của khóa học này, được dựng từ những phần đó. Một mục
đứng riêng như (A11) hay (D2) bên dưới nằm trong `notes/today/ch10-facts.md`.

Hãy viết ba kernel theo hàng trên logits được lưu dưới dạng `[rows x cols]` float32, theo thứ tự
hàng (row-major), batch đứng đầu:

- `launchRowMax(x, out, rows, cols)`: giá trị lớn nhất của mỗi hàng.
- `launchSoftmaxRows(x, y, rows, cols, scale)`: `y = softmax(scale * x)` cho mỗi hàng, được tính một
  cách ổn định: một phép rút gọn max, rồi một tổng của `exp(scale * x - max)`, rồi phép chia.
  `scale` là 1/temperature khi bạn lấy mẫu (module X1) và 1/sqrt(d) trong attention (chương 20).
- `launchArgmaxRows(x, idx, rows, cols)`: chỉ số của logit lớn nhất của mỗi hàng, khi hòa thì lấy
  chỉ số nhỏ nhất. Đây là bước greedy decoding.

Mỗi hàng một khối 256 luồng (thread). Mỗi luồng đi qua hàng của nó bằng một vòng lặp có bước nhảy
bằng kích thước khối (được hợp nhất), rồi khối rút gọn bằng phép rút gọn warp hai giai đoạn của lab
10.1: mặt nạ có kiểu, `warpSize`, giai đoạn 2 có kiểm tra điều kiện.

- Sửa `work/kernel.cu` (CUDA; cũng được dựng cho HIP và các trình giả lập) và `work/kernel.metal` +
  `work/launch.mm` (Metal). Các tên là cố định: chương trình điều khiển (driver) gọi
  `launchRowMax`, `launchSoftmaxRows`, `launchArgmaxRows`, các kernel Metal `row_max`,
  `softmax_rows`, `argmax_rows`, và các bộ encode `encodeRowMax`, `encodeSoftmaxRows`,
  `encodeArgmaxRows`.
- Được cung cấp, không sửa: `labs/ch10/shared/rowwise_naive.cu`, `rowwise_naive.metal` và
  `rowwise_naive_launch.mm`, mỗi hàng một luồng, bản cơ sở cho benchmark.
- Kiểm thử: `python3 labs/pmpp.py run 10.3`. Bắt đầu với `--backend emu`.
- Race condition: `python3 labs/pmpp.py run 10.3 --backend race` (xem "Backend race" bên dưới).
- Đo: `python3 labs/pmpp.py run 10.3 --bench`.

## Hợp đồng

- `rows >= 1`, `cols >= 1`, giá trị bất kỳ: không chỉ bội của 32 hay 256.
- Mọi logit đều hữu hạn. NaN và vô cực nằm ngoài hợp đồng được kiểm thử (mục cuối).
- Khi hòa thì lấy chỉ số nhỏ nhất. Cây kết hợp các cặp (giá trị, chỉ số) theo một thứ tự bạn không
  kiểm soát, nên phép kết hợp phải cho cùng một cặp theo bất kỳ thứ tự nào: giá trị lớn hơn thắng,
  và giữa hai giá trị bằng nhau thì chỉ số nhỏ hơn thắng. Quy tắc đó giao hoán và kết hợp trên các
  giá trị hữu hạn, đúng như điều các mục 10.4 và 10.9 yêu cầu ở một toán tử. PyTorch công bố cùng
  quy tắc đó cho `torch.argmax` (notes/today/ch10-facts.md D4); argmax CUDA, argmax Metal và bộ lấy
  mẫu CPU của llama.cpp mỗi cái phá thế hòa theo một cách khác nhau (D2), nên cùng một bước greedy có
  thể chọn các token khác nhau trên các backend khác nhau.
- Phần tử đơn vị của max là `-FLT_MAX`, không phải âm vô cực như sách (p. 222). Dưới fast math của
  Metal, hành vi của vô cực là không xác định (C3), và một phần tử đơn vị hữu hạn giữ ba backend trên
  cùng một hợp đồng. Phần tử đơn vị của argmax là cặp (`-FLT_MAX`, `INT_MAX`): nó thua mọi thế hòa,
  kể cả trước một logit bằng `-FLT_MAX`.

## Các bài kiểm thử kiểm tra gì

- Hình dạng: rows 1, 4 và 7 nhân cols 1, 2, 31, 32, 33, 255, 256, 257, 1000, 4096, 32000, 50257 và
  128256 (ba giá trị cuối có cỡ của một bộ từ vựng). Trình giả lập dùng rows 1 và 3 và cols tối đa
  4099; backend race dùng rows 2 và cols tối đa 1000.
- Dữ liệu, theo từng hàng, được trộn trong một batch để bắt một kernel đọc nhầm hàng: các giá trị
  giống logit phân bố đều trong [-12, 12]; chính các giá trị đó dịch đi +90 và -90 (exp(90) vượt
  quá phạm vi của float, nên một softmax bỏ qua bước trừ max sẽ trả về vô cực hoặc NaN: phụ lục A,
  pp. 590-591); các hàng toàn giá trị bằng nhau (argmax 0, softmax 1/cols); các hàng toàn số âm (bắt
  một phần tử đơn vị max bằng 0); một hàng `-FLT_MAX`; và các giá trị lớn nhất chính xác được cài
  sẵn ở các cột (1, 2), (0, cols-1), (5, 261), (6, 261), (3, 40) và (cols-2, cols-1). Các cặp này
  đặt thế hòa ở các lane kề nhau, ở cả hai đầu, hai lần trong vòng lặp của một luồng, với chỉ số cao
  hơn ở lane thấp hơn, trong hai warp, và ở cuối hàng. Cùng nhau chúng bắt được "giữ cặp của mình
  khi hòa", ">= trong vòng lặp của luồng" và "chỉ số cao nhất thắng".
- Các scale 1, 0.5 và 0.125 cho softmax.
- Max theo hàng chính xác, argmax chính xác, softmax trong dung sai bên dưới, và không gì được ghi
  quá cuối bất kỳ đầu ra nào.
- Greedy decoding (bên dưới).
- Bản cơ sở được cung cấp cũng được kiểm thử, để một tệp được cung cấp bị hỏng sẽ lộ ra ở đây.

## Dung sai, và vì sao

Max và argmax được so sánh chính xác tuyệt đối: một phép max không làm phép tính số học nào.

Softmax được so sánh theo từng phần tử với `|got - ref| <= 1e-20 + 1e-4 * ref`, trong đó `ref` được
tính bằng double trên CPU từ cùng các đầu vào float, và mọi hàng phải có tổng bằng 1 trong phạm vi
`1e-4`. Phần tương đối đến từ các cận sai số, ở hàng lớn nhất được kiểm thử (128,256 cột, đối số
của hàm mũ nằm giữa -24 và 0):

- `exp`: `expf` của CUDA sai lệch trong phạm vi 2 ulp (ch10-facts A11). `exp` chính xác của Metal sai
  lệch trong phạm vi 4 ulp, `exp` nhanh của nó trong phạm vi `3 + floor(|2x|)` ulp (C4), tức là 51
  ulp, khoảng 6e-6 tương đối, tại x = -24. HIP công bố các sai khác đo được so với thư viện C++, không
  phải các cận (1 ulp cho `expf` trên [-10, 10], B4); ngoài phạm vi đó thì là unverified.
- Bản thân đối số `scale * x - max` bị làm tròn một lần: tối đa 1.4e-6 tương đối trong `exp`.
- Tổng: mỗi luồng cộng lần lượt khoảng 500 số hạng, rồi hai giai đoạn cây, tối đa khoảng 540 lần làm
  tròn mỗi lần 2^-24, nên 3.2e-5 tương đối (mọi số hạng đều dương).
- Một phép nghịch đảo và một phép nhân: 1.2e-7.

Tổng cộng khoảng 4e-5 với `exp` nhanh của Metal (6.1e-6 + 1.4e-6 + 3.2e-5 + 1.2e-7) và 3.4e-5 với
`expf` của CUDA; `1e-4` vẫn lớn hơn gấp đôi con số lớn hơn. Một bước trừ max bị thiếu, một scale
sai hay một mẫu số sai sẽ lệch đến nhiều bậc độ lớn. Số hạng tuyệt đối thấp hơn rất nhiều so với xác
suất nhỏ nhất mà các hàng kiểm thử tạo ra (khoảng 1e-14), nên trên thực tế phép kiểm tra là tương
đối; nó có mặt để một kết quả subnormal, mà GPU có thể đưa về 0, không tự làm một hàng
thất bại.

Hãy viết `expf`, không phải `__expf`: cận ở trên dùng con số của `expf`, và sai số của `__expf` tăng
theo |x| (2 + floor(|1.173 x|) ulp, A11). Bản cơ sở được cung cấp cộng cả một hàng trong một luồng,
tệ nhất là mỗi cột một lần làm tròn, nên chương trình điều khiển áp cho nó dung sai
`1e-4 + cols * 2^-24`.

## Backend race

`emu` chạy lần lượt từng luồng và không thấy được một race condition tình cờ để lại giá trị đúng.
`race` chạy các luồng của một khối cùng lúc dưới ThreadSanitizer và báo hai luồng chạm vào cùng vùng
nhớ mà không có gì sắp thứ tự chúng, cùng cả hai dòng mã nguồn. Trường hợp mà lab này được xây dựng
cho: hai phép rút gọn khối liên tiếp dùng chung một mảng chia sẻ. Phép thứ hai ghi vào mảng trong
khi các warp chậm hơn có thể vẫn đang đọc kết quả của phép thứ nhất, nên phải có gì đó sắp thứ tự
chúng (cả llama.cpp lẫn PyTorch đều làm vậy, D1, D4). Trên GPU, lỗi đó có thể đạt mọi bài kiểm thử,
vì việc một warp nhanh có ghi đè kết quả trước khi một warp chậm kịp đọc hay không phụ thuộc vào lập
lịch.

## Greedy decoding (mốc dự án)

Chương trình điều khiển cũng chạy một bộ giải mã greedy đơn giản trên argmax của bạn. "Mô hình" là
một bảng tra logits tất định, `V x V`: sau token t, logits là hàng t, và mọi hàng có giá trị lớn nhất
ở hai cột, một thế hòa chính xác. Với một batch gồm 4 chuỗi, mỗi bước chép các hàng của các token
hiện tại vào một buffer logits `[4 x V]`, gọi `launchArgmaxRows`, và đưa các chỉ số trở lại làm các
token tiếp theo. Sau 16 bước, các chuỗi phải khớp với của CPU.

Đó là greedy decoding từ đầu đến cuối cho một bảng, không phải cho một mô hình ngôn ngữ. Mốc M10
(lab X3.2) chạy một mô hình nhỏ kiểu Llama, được một script trong repository ghi ra từ một seed cố
định, và bước greedy của nó gọi lời giải tham chiếu (solution) của `launchArgmaxRows` trong lab này,
không phải bản `work/` của bạn.

## Trước khi bạn đo hiệu năng

Hãy viết ra hai dự đoán (bài học hỏi chúng dưới id `ch10/labs/p-lab3-naive`):

1. Ở batch 1 và 128,256 cột, softmax mỗi hàng một khối của bạn sẽ nhanh hơn mỗi hàng một luồng bao
   nhiêu lần?
2. Khoảng cách đó sẽ tăng hay giảm ở batch 64?

Trình khám phá của mục 10.10 cho thấy vì sao mỗi hàng một khối để phần lớn GPU rảnh ở batch 1.

Benchmark đo thời gian softmax (scale 1) và argmax ở rows 1, 8 và 64 nhân cols 32,000 và 128,256,
cho các kernel của bạn ("block") và bản cơ sở ("naive"). Thời gian mỗi lần gọi là kết quả tốt nhất
trong 5 lần chạy được đo, mỗi lần lặp lại lời gọi cho đến khi nó kéo dài ít nhất 2 ms. GB/s đếm lưu
lượng tối thiểu: 8 byte mỗi phần tử cho softmax (đọc x một lần, ghi y một lần) và 4 cho argmax (đọc
x một lần). Softmax của bạn đọc mỗi hàng ba lần, nên nó chuyển nhiều hơn mức được tính cho nó. Mọi
kết quả được đo đều được kiểm tra trước khi báo cáo. Các con số là phép đo trên máy của bạn, không
phải mục tiêu.

## NaN và vô cực

Nằm ngoài hợp đồng được kiểm thử, vì ba lý do:

- Fast math của Metal, có thể là mặc định của runtime (Apple mô tả mặc định theo ba cách khác nhau,
  ch05-facts C7), để hành vi của NaN và vô cực là không xác định (C3). Một bài kiểm thử đạt ở đó sẽ
  không chứng minh được gì.
- `fmaxf` của CUDA bỏ một toán hạng NaN (A10), trong khi một phép so sánh như `a > b ? a : b` trả về
  toán hạng thứ hai mỗi khi một trong hai là NaN: cùng một phép rút gọn cho các câu trả lời khác nhau
  tùy cách viết phép max của nó.
- Các thư viện không thống nhất, nên không có một câu trả lời duy nhất để kiểm thử: PyTorch và MLX để
  NaN thắng argmax (D3, D4); argmax CUDA của llama.cpp, theo cách chúng tôi đọc mã nguồn của nó, bỏ
  qua NaN và trả về -1 cho một hàng toàn âm vô cực (D2); `Max` của CUB bỏ qua NaN trong khi `ArgMax`
  của nó không nói gì (A13).

Các logit bị che được đặt thành âm vô cực là trường hợp quan trọng trên thực tế (mặt nạ attention,
các token bị cấm). Chúng sẽ quay lại cùng kernel mặt nạ và softmax của chương 20.

## Tính khả chuyển

- HIP: `kernel.cu` được dựng qua `pmpp/vendors/hip.h` với mặt nạ có kiểu và `warpSize`; với warp
  rộng 64, một khối 256 có 4 warp, và giai đoạn 2 đệm các lane còn lại. Chưa được biên dịch hay chạy
  trên phần cứng AMD ở đây: unverified.
- Metal: `launch.mm` đọc độ rộng SIMD-group từ pipeline (`threadExecutionWidth`), kiểm tra rằng 256
  luồng tạo thành nguyên các SIMD-group với số nhóm không vượt quá số lane, và đặt kích thước bộ nhớ
  threadgroup cho từng SIMD-group từ đó (một bội số của 16 byte, ch05-facts C8).
- `expf` không cần khai báo riêng: kiểm tra front-end CUDA của khóa học khai báo các hàm toán học độ
  chính xác đơn trong header stub của nó. Các phiên bản trước của `kernel.cu` và
  `shared/rowwise_naive.cu` khai báo nó dưới `PMPP_CUDA_FRONTEND_CHECK`, với một chú thích nói rằng
  kiểm tra đó không khai báo hàm CUDA Math API nào; điều đó có từ trước khi header stub có phần toán
  học, và khai báo đó đã được bỏ. Nếu `work/kernel.cu` của bạn vẫn còn nó, nó vô hại.
