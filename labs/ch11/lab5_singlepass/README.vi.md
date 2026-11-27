> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 11.5: scan một lượt với lookback (họ CUDA, tùy chọn)

Sách: mục 11.9 (pp. 276-282), Figs. 11.16(b), 11.17 và 11.18.

Lab 11.4 nối các đoạn bằng nhiều kernel và trả giá bằng lưu lượng: 16 hoặc 12 byte mỗi phần tử
trong khi một scan chỉ cần 8 (pp. 272-276). Ở đây một kernel làm toàn bộ phép scan. Mỗi khối
scan đoạn của nó. Luồng cuối cùng của nó, luồng dẫn đầu, chờ cho đến khi khối đứng trước đã
công bố tổng của mọi thứ tính đến cuối khối đó. Sau đó luồng dẫn đầu công bố điều tương tự cho
khối của mình, và khối cộng tổng phía trước vào các phần tử của nó. Đầu vào được nạp một lần và
đầu ra được ghi một lần.

Hai điều làm cho việc chờ an toàn trên CUDA:

- **Chỉ số khối lấy từ một bộ đếm, không từ `blockIdx.x`** (p. 280). Luồng 0 nhận chỉ số tiếp
  theo bằng một phép tăng nguyên tử (atomic). Khi đó một khối chỉ có thể chờ một khối đã bắt
  đầu. Theo các quy tắc tiến triển (forward progress) trong hướng dẫn lập trình của NVIDIA, một
  khối đã bắt đầu sẽ tiếp tục tiến triển, nên việc chờ sẽ kết thúc (ch11 facts A6 trình bày
  chi tiết). Không có gì hứa rằng khối có `blockIdx.x - 1` đã bắt đầu cả.
- **Cờ là một atomic phạm vi device với release và acquire** (Fig. 11.17). Luồng dẫn đầu ghi
  tổng tích lũy của nó vào `partialSums[bid]` bằng một lần ghi thường, rồi đặt `flags[bid]`
  bằng một lần ghi release. Luồng dẫn đầu tiếp theo đọc cờ bằng một lần nạp acquire và chỉ khi
  đó mới đọc `partialSums[bid - 1]` bằng một lần nạp thường. Đây chính là ví dụ "message
  passing" của NVIDIA (ch11 facts A5).

**Không có phiên bản Metal.** Metal không ghi trong tài liệu sự tiến triển giữa các threadgroup.
Một kỹ sư Apple, khi được hỏi về chính scan nối chuỗi này, đã gọi các giả định của nó là hành vi
không xác định (ch11 facts C5). Trên Metal, lab 11.4 là scan toàn cục.

**Phần việc của bạn** (trong `work/kernel.cu`):

- **Chỉ số khối động** ở đầu `SinglePassScanKernel` (Fig. 11.18, dòng 04-11): một luồng nhận nó
  từ `*blockCounter`, và khối chia sẻ nó.
- **`interBlockScan`** (Fig. 11.17), với luồng dẫn đầu là luồng cuối cùng. Chờ bằng
  `cuda::atomic_ref<unsigned int, cuda::thread_scope_device>` và
  `load(cuda::memory_order_acquire)`. Công bố bằng một lần ghi thường vào `partialSums[bid]`,
  rồi `store(1, cuda::memory_order_release)` trên cờ. Trả về tổng của các khối phía trước cho
  mọi luồng.
- **`launchSinglePassScan`**: đặt về 0 bộ đếm và các cờ (`cudaMemset`) trước mỗi lần khởi chạy.

Scan cấp khối bên dưới (các lab 11.1 và 11.2) được cung cấp. Phần mang sang được cộng trước lần
ghi cũng vậy.

Chỗ khác với trang sách:

- Danh sách tham số của Fig. 11.18 thiếu dấu phẩy trước `unsigned int N` (p. 281).
- Fig. 11.18 không bao giờ dùng N. Mọi lần nạp và ghi ở đây đều được kiểm tra với nó.
- Fig. 11.17 đọc cờ bằng `fetch_add(0, acquire)` và đặt nó bằng `fetch_add(1, release)`. Ví dụ
  của NVIDIA dùng `load` và `store`, các dạng chỉ đọc và chỉ ghi với cùng thứ tự bộ nhớ (ch11
  facts A5).
- Fig. 11.17 khai báo `prevBlockPartialSum_s` bên trong hàm device. Ở đây kernel khai báo nó và
  truyền một con trỏ, như với các tổng warp trong lab 11.1 (ch11 facts B1, C2).
- Sách ngụ ý rằng các cờ bắt đầu bằng 0 (p. 278) và nói rằng bộ đếm được "zero-initialized"
  (p. 280). Hàm khởi chạy làm cả hai trước mỗi lần khởi chạy.

## Tệp

- Sửa `work/kernel.cu` (CUDA, HIP và cả hai trình giả lập): `SinglePassScanKernel`,
  `interBlockScan` và `launchSinglePassScan`. Không có `kernel.metal` hay `launch.mm`.
- Được cung cấp, không sửa: `../shared-g13/global_copy.cu` (một phép sao chép đơn giản, mốc băng
  thông).

## Lệnh

- Kiểm thử: `python3 labs/pmpp.py run 11.5 --backend emu`, rồi
  `python3 labs/pmpp.py run 11.5 --backend race`. Trên máy CUDA,
  `python3 labs/pmpp.py run 11.5`.
- Đo: `python3 labs/pmpp.py run 11.5 --bench` (chỉ trên thiết bị CUDA hoặc HIP).
- Lời giải tham chiếu: thêm `--solution`.

Một khối chờ một cờ mà không ai đặt sẽ không bao giờ ngừng chờ. Trên `emu` và `race`, lần chạy
khi đó bị treo cho đến khi `pmpp.py` bỏ cuộc, sau 15 phút; thay vào đó hãy nhấn Ctrl-C. Các
nguyên nhân thường gặp: thiếu lần ghi release, chờ trên `flags[bid]` thay vì `flags[bid - 1]`,
hoặc khối 0 cũng chờ.

## Các bài kiểm thử làm gì

- Trên thiết bị thật: n = 1, 33, 1000, 4097, 100003 và 2^22, với khối 64 và 256. Trên `emu`: n
  tới 4097 với khối 64 và 256. Trên `race`: n = 33, 1000 và 4097 với khối 64, tức là 1, 4 và 17
  khối, nên ít nhất bốn khối chạy cùng lúc.
- Dữ liệu và dung sai như trong lab 11.4: toàn số một và các số nguyên từ -3 đến 3 phải khớp
  chính xác; các giá trị phân bố đều trong [-1, 1] phải nằm trong phạm vi 1e-5 nhân tổng |x|
  tính đến phần tử đó, cộng 1e-6, so với vòng lặp của Fig. 11.1 chạy bằng double.
- Mỗi trường hợp chạy hai lần trên cùng các buffer. Trước lần khởi chạy đầu tiên, mọi cờ đều
  được đặt, `partialSums` và đầu ra là NaN, và bộ đếm giữ giá trị lớn hơn số khối một đơn vị,
  như thể còn sót lại từ việc khác. Lần khởi chạy thứ hai nhận dữ liệu mới và bất cứ thứ gì
  lần đầu để lại.
- Nếu các cờ không được đặt về 0, một luồng dẫn đầu có thể đọc `partialSums[bid - 1]` trước khi
  khối đó đã ghi nó.
- Nếu bộ đếm không được đặt về 0, các khối nhận chỉ số vượt quá khối cuối cùng. `partialSums`
  và `flags` được theo sau bởi đủ các từ canh gác khác 0 để trường hợp này thất bại kèm thông
  báo thay vì một lần chờ không bao giờ kết thúc.
- Sau mỗi lần khởi chạy, bộ đếm phải bằng số khối. Một kernel không bao giờ nhận chỉ số từ bộ
  đếm sẽ thất bại ở đây, và chỉ ở đây.

## Những gì các trình giả lập thấy được và không thấy được

Đã được kiểm tra khi viết lab này, bằng cách cài từng lỗi vào lời giải tham chiếu:

- `race` báo một cờ relaxed. Khi lần nạp acquire bị đổi thành relaxed, lần ghi release bị đổi
  thành relaxed, hoặc cả hai, ThreadSanitizer báo một data race giữa lần đọc
  `partialSums[bid - 1]` và lần ghi `partialSums[bid]`, ở mọi lần chạy. `emu` cho qua cả ba.
- `race` báo các cờ mà hàm khởi chạy không đặt về 0 (một data race trên `partialSums`). `emu`
  cho qua chúng.
- Cả hai đều báo một bộ đếm mà hàm khởi chạy không đặt về 0, và một kernel không bao giờ nhận
  chỉ số từ bộ đếm.

Những gì không cái nào thấy được:

- `emu` chạy các khối lần lượt theo thứ tự chỉ số. Lookback không bao giờ phải chờ ở đó, và chỉ
  số động luôn bằng `blockIdx.x`.
- Phạm vi luồng bị bỏ qua. Một cờ khai báo với `cuda::thread_scope_block` qua được trên cả hai
  trình giả lập. Trên GPU, đó là một data race, tức hành vi không xác định, vì cờ được một khối
  khác đọc (ch11 facts A5, ch09 facts A2).
- Không cái nào tạo ra được bế tắc của việc chờ khối có `blockIdx.x - 1`. Một kernel dùng
  `blockIdx.x` đã cho đầu ra đúng trên cả hai khi được kiểm tra. Phép kiểm tra bộ đếm ở trên chỉ
  bắt được nó khi kernel không bao giờ chạm vào bộ đếm: một kernel nhận một chỉ số rồi vẫn dùng
  `blockIdx.x` sẽ qua mọi bài kiểm thử ở đây. Hãy đọc mã của bạn để tìm lỗi đó. Trình mô phỏng
  của mục 11.9 cho thấy bế tắc này.
- Một lần chạy `race` sạch không chứng minh là không có race condition. ThreadSanitizer chỉ xem
  xét những lịch chạy đã xảy ra. Qua ở đây không chứng minh kernel chạy được trên GPU.

## Chỗ các backend khác nhau (không có trong sách)

- **HIP**: `cuda::atomic_ref` đến từ libhipcxx (ROCm 10.1 trở lên), giữ nguyên cách viết của
  CUDA. libhipcxx chuyển thứ tự bộ nhớ thẳng xuống các atomic của AMD (ch11 facts B4, infra
  facts B2). Bản build HIP của khóa học liên kết libhipcxx ở nơi ROCm cung cấp nó (unverified:
  chưa có máy ROCm nào build nó). AMD không ghi trong tài liệu bảo đảm tiến triển nào giữa các
  workgroup. Chính rocPRIM của AMD chỉ nhận chỉ số khối bằng một bộ đếm atomic trên một số kiến
  trúc, và đã cần một lần sleep trong vòng lặp chờ trên các dòng MI100 đời đầu (ch11 facts B3).
  Là unverified trên HIP cho đến khi một `reports/verify-*.md` từ một máy AMD cho thấy nó qua.
- **Metal**: không có (ch11 facts C5), như trên.
- **Thư viện**: `DeviceScan` của CUB dùng decoupled look-back, thiết kế của bài tập 5, và nói nó
  "typically proceeds at "memcpy" speeds" (ch11 facts A1, A8). CUB đánh chỉ số các tile theo
  `blockIdx.x`, dựa vào thứ tự khởi chạy, điều mà hướng dẫn CUDA không hứa (ch11 facts A6, cách
  đọc của chúng tôi).

## Trước khi đo hiệu năng, hãy dự đoán

Hãy ghi điều này ra trước. Trang labs hỏi nó dưới id trong ngoặc.

- Kernel một lượt sẽ chạy nhanh hơn scan-scan-add của lab 11.4 bao nhiêu lần ở n = 2^26? Phép
  đếm byte cho 2 (16 byte mỗi phần tử so với 8) (`ch11/labs/p-lab5-single`).
- Và nó sẽ đạt bao nhiêu phần tốc độ của kernel chép? Mọi khối chờ khối đứng trước nó, một chuỗi
  đi qua toàn bộ n/S khối (p. 279).

## Đọc kết quả

Benchmark scan n = 2^26 số float. Nó dùng ít hơn nếu thiết bị của bạn báo ít hơn bốn lần 512 MB
đầu vào và đầu ra. Nó chạy với khối 256 và 1024 (S = 1024 và 4096), đo phép sao chép rồi kernel một
lượt, và kiểm tra phép scan trước khi báo.

- `time ms` là lần tốt nhất trong năm lần chạy sau một lần khởi động không đo. Trên CUDA và HIP,
  nó là thời gian phía host bao quanh hàm khởi chạy và một lần đồng bộ hóa, nên nó bao gồm hai
  lần gọi `cudaMemset` của hàm khởi chạy.
- `GB/s` tính 8 byte mỗi phần tử, phép đếm của sách cho một lượt, chia cho thời gian đó.
- `of copy` là thời gian của phép sao chép tính theo tỉ lệ so với thời gian của kernel.

Hãy chạy benchmark của lab 11.4 trên cùng thiết bị để so sánh. Trang labs đặt các hàng của cả
hai lab cạnh nhau. Các con số là phép đo của máy đó ở kích thước đó. Hãy chạy mỗi benchmark
nhiều hơn một lần; nếu các lần chạy không khớp nhau, độ phân tán đó là một phần của phép đo.

## Đi xa hơn (thiết bị CUDA; chưa được kiểm thử ở đây)

- **Lùi lại chờ (back-off).** CUDA có `__nanosleep(ns)` cho chính kiểu chờ này, và CUB gọi nó
  trong look-back của mình (ch11 facts A7). Hãy sleep một chút bên trong vòng lặp chờ rỗng và
  đo. HIP không ghi trong tài liệu hàm tương ứng nào (ch11 facts B4), và cả các trình giả lập lẫn
  các stub của phép kiểm tra front-end đều không khai báo nó, nên hãy đặt nó trong khối
  `#if defined(PMPP_BACKEND_CUDA)`, macro mà bản build của khóa học chỉ định nghĩa cho backend
  CUDA.
- **Decoupled look-back** (bài tập 5, pp. 279-280). Mỗi khối công bố tổng của riêng nó trước,
  với một trạng thái thứ hai, và luồng dẫn đầu đi ngược qua các khối mà tổng tích lũy chưa sẵn
  sàng, cộng các tổng riêng của chúng, cho đến khi gặp một khối đã sẵn sàng. Chương trình điều
  khiển (driver) chỉ kiểm tra đầu ra, bộ đếm và các từ canh gác, nên nó chấp nhận mọi giá trị
  cờ.
