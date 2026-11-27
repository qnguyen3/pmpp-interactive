> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab X1.1: top-k bằng radix select, đo so với một phép sắp xếp đầy đủ (mốc dự án M5)

Không có trong sách: lấy mẫu là của khóa học này (PLAN.md, module X1). Các thành phần là của sách:
histogram trong bộ nhớ chia sẻ (shared memory) với phép toán nguyên tử (atomic) của chương 9
(Fig. 9.10, pp. 213-214), ý tưởng chữ số có nghĩa lớn nhất trước của mục 14.9 (p. 346), phép bỏ phiếu
warp và thứ hạng bằng popcount của mục 12.3 (Fig. 12.3, pp. 292-293) bên trong phép lọc ổn định của
mục 12.5 (pp. 297-298), và warp scan của chương 11 (Fig. 11.8, p. 264).
Mục X1.4 của bài học giải thích thuật toán và chạy qua nó từng bước. Một mục đứng riêng như (A3) hay (D2)
dưới đây nằm trong `notes/today/ch14-facts.md`; các mục khác ghi rõ tệp của chúng.

Viết một hàm host và hai kernel đứng sau nó:

```cpp
void launchTopKRows(const float* logits, int* outIdx, float* outVal, unsigned int* scratch,
                    int rows, int cols, int k);
```

Với mỗi hàng của `logits`, một tensor float32 `[rows x cols]` (thứ tự hàng (row-major), batch trước),
hãy ghi ra k logits lớn nhất và các cột của chúng, mà không sắp xếp hàng. Lab X1.3, toàn bộ bộ lấy
mẫu, gọi một bản sao lời giải tham chiếu (solution) của lab này làm giai đoạn top-k của nó.

- Sửa `work/kernel.cu` (CUDA; cũng được build cho HIP và các trình giả lập): `topkThreshold`,
  `topkGather`, `launchTopKRows`. Sửa `work/kernel.metal` (`topk_threshold`, `topk_gather`) và
  `work/launch.mm` (`encodeTopKRows`, hàm mã hóa cả hai vào đúng một compute encoder được đưa cho nó).
- Có sẵn trong các tệp đó: các hàm trợ giúp warp của lab 12.4 (mặt nạ lane có kiểu, một popcount và
  một find-first-set khớp với nó, `laneIdx`, `warpIdx`), và khóa của lab 14.3 (`orderedKey`).
- Có sẵn và không được sửa: phép sắp xếp của lab 14.3, mốc so sánh của benchmark. Trên CUDA, HIP và
  các trình giả lập, đó là `labs/x1/shared-g11/sortrows_g11.cu`, một bản sao `solution/kernel.cu` của
  lab 14.3 mà thay đổi duy nhất là `laneIdx` và `warpIdx` là static (để tệp của bạn cũng có thể định
  nghĩa chúng); `app/src/chapters/x1/copies-g11.test.ts` kiểm tra rằng ngoài điều đó không có chỗ nào khác biệt.
  Trên Metal, chính `solution/kernel.metal` và `launch.mm` của lab 14.3 được dùng.
- Kiểm thử: `python3 labs/pmpp.py run x1/lab1`. Bắt đầu với `--backend emu` (khoảng ba phút).
- Race: `python3 labs/pmpp.py run x1/lab1 --backend race`.
- Đo: `python3 labs/pmpp.py run x1/lab1 --bench`, sau khi có dự đoán bên dưới.

## Hợp đồng

- `rows >= 1`, `cols >= 1`, `k >= 1`, kích thước bất kỳ. Mọi logit đều hữu hạn; cho phép `-0.0` và
  `+0.0`.
- Đặt `kk = min(k, cols)`. Đầu ra của hàng `r` là `outIdx[r * k + j]` và `outVal[r * k + j]` với
  `j < kk`: các cột của `kk` logits lớn nhất của hàng, theo **thứ tự cột tăng dần**, và các logit của
  chúng. Bước nhảy (stride) hàng của đầu ra là `k`, không phải `kk`.
- "Lớn nhất" là thứ tự theo khóa của lab 14.3: lấy 32 bit `u` của số float, đổi `-0.0`
  (`0x80000000`) thành `+0.0`, rồi bật bit dấu của một giá trị không âm và đảo mọi bit của một giá trị
  âm. Đọc như số nguyên không dấu, các khóa có cùng thứ tự với các logit, với hai số không bằng nhau
  (CUB mô tả phép biến đổi này, A3).
- Giá trị bằng nhau: khi nhiều cột cùng giữ giá trị thứ `kk`, các cột thấp nhất được giữ. Đây là điều
  radix select CUDA của llama.cpp và `kernel_top_k` trên Metal làm (D2); Hugging Face transformers thì
  giữ mọi token bằng nhau và có thể trả về nhiều hơn k (x1-facts D4).
- `outVal` chứa 32 bit của logit, được sao chép. Các ô `kk .. k-1` của một hàng, mọi thứ sau
  `rows * k`, sau `4 x rows` phần tử của `scratch`, và chính các logit đều giữ nguyên.
- Các kernel không bao giờ thực hiện một phép toán dấu phẩy động: hàm khởi chạy đưa cho chúng các logit
  dưới dạng số nguyên không dấu, như lab 14.3, nên không chế độ toán nào có thể viết lại phép kiểm tra
  `-0.0` (C3).

## Thiết kế (của khóa học này)

Một khối 256 luồng cho mỗi hàng, hai kernel.

1. `topkThreshold`, tối đa bốn lượt trong một lần khởi chạy, mỗi lượt cho một byte của khóa, byte có
   nghĩa lớn nhất trước. Mỗi lượt xóa về không một histogram 256 ngăn trong bộ nhớ chia sẻ, đi qua hàng
   (luồng `t` nhận các cột `t, t + 256, ...`) và, với mỗi khóa có các byte đã quyết định đến giờ bằng
   tiền tố tìm được đến giờ, đếm byte hiện tại của nó một cách nguyên tử. Rồi nó đi qua các ngăn từ 255
   xuống cho đến khi số đếm tích lũy đạt số khóa vẫn còn cần: một warp làm việc này, mỗi lần `warpSize`
   ngăn, với một warp scan và một ballot. Các khóa ở ngăn cao hơn được chọn, các khóa ở ngăn thấp hơn
   bị loại, byte của ngăn vượt ngưỡng nhập vào tiền tố, và số còn cần giảm đi bằng số khóa phía trên
   nó. Sau bốn lượt, tiền tố là `K*`, khóa lớn thứ `kk`. Khi ngay ở một lượt sớm hơn, mọi khóa của ngăn vượt ngưỡng
   đều cần được lấy, tập hợp đã được quyết định và kernel dừng (trình chạy từng bước của mục X1.4 cho thấy
   điều này): khi đó `K*` là tiền tố với các bit chưa quyết định bằng không, khóa nhỏ nhất ngăn đó có
   thể chứa, và hạn mức là cả ngăn, nên bước gom giữ lại toàn bộ ngăn. Dù theo cách nào, nó để lại
   trong `scratch` giá trị `K*` và số khóa bằng `K*` vẫn còn cần (hạn mức cho giá trị bằng nhau).
2. `topkGather`, một lượt theo các vòng 256 cột. Một khóa ở trên (`key > K*`), bằng (`key == K*`),
   hoặc bị loại. Một khóa bằng được giữ khi số khóa bằng đứng trước nó trong hàng, theo thứ tự cột,
   còn dưới hạn mức; vị trí của một khóa được giữ là số khóa được giữ đứng trước nó. Cả hai thứ hạng
   đến từ một ballot và một popcount của các lane bên dưới trong mỗi warp, tổng của các warp trong bộ
   nhớ chia sẻ, và các số đếm đang chạy mang từ vòng này sang vòng khác: phép lọc ổn định của lab 12.4,
   cộng thêm một hạn mức cho giá trị bằng nhau.

Đầu ra theo thứ tự cột vì các thứ hạng theo thứ tự cột: không có phép sắp xếp nào cả.

## Những gì các phép kiểm thử kiểm tra

- Kích thước: rows 1, 4 và 7 nhân cols 1, 2, 31, 255, 256, 257, 1000, 4096, 32000, 50257 và 151936
  (kích thước bộ từ vựng của GPT-2 và Qwen3, ch09-facts D7), và với mỗi cái, k = 1, 2, 40, 256, 1000,
  cols và cols + 5 (tức là lớn hơn hàng) khi k <= cols + 5. Trên `emu`, rows 1 và 3 nhân cols tới
  4099, k = 1, 40, 256 (từ 257 cột), 1000 (chỉ 4099) và cols + 5, một batch cho mỗi số hàng mỗi trường
  hợp với dữ liệu xoay vòng, vì mọi trường hợp chạy dưới ba thứ tự luồng. Trên `race`, rows 2 nhân cols
  1, 33, 257, 1000 và 2100 với cùng các k, mọi loại dữ liệu.
- Dữ liệu, trộn lẫn trong mỗi batch để bắt được một kernel đọc hoặc ghi nhầm hàng:
  - logits phân bố đều trong [-12, 12] (65,536 mức, nên có nhiều giá trị bằng nhau);
  - các hàng có đỉnh nhọn dạng softmax (xác suất dương);
  - mọi logit bằng nhau (đáp án là các cột 0 .. kk-1);
  - các giá trị bằng nhau chính xác ở giá trị thứ k, cài sẵn ở các cột 31 và 32, 63 và 64, 255 và
    256, 0 và cols-1, và các dải 40 giá trị bằng nhau vắt qua một đường nối warp (12..51), đường nối
    vòng đầu tiên (236..275) và một đường nối sau đó (2028..2067), với hạn mức là 1, một nửa hoặc tất
    cả trừ một;
  - `-0.0` và `+0.0` xen kẽ mỗi cột thứ ba với giá trị thứ k là một số không;
  - các khóa chung 8, 16 và 24 bit cao nhất (buộc phải có 2, 3 và 4 lượt; trường hợp cuối chỉ có 256
    giá trị khác nhau);
  - các khóa có mọi byte là 0x00 hoặc 0xFF, và độ lớn gần `FLT_MAX` ở cả hai dấu (để ngăn vượt ngưỡng
    nằm ở một trong hai đầu của lượt đi);
  - chỉ có logits âm.
- So sánh chính xác, không có dung sai: `outIdx` bằng bản tham chiếu; `outVal` giống từng bit với các
  logit ở những cột đó; các ô sau `kk` giữ giá trị canh gác mà chương trình điều khiển (driver) đã ghi (-1 và một mẫu bit NaN);
  64 phần tử canh gác sau `outIdx`, `outVal` và `scratch` không bị chạm vào; các logit không đổi.
- Bản tham chiếu, trong `main.cpp`: `std::stable_sort` các chỉ số cột theo khóa, lớn nhất trước, rồi
  lấy `kk` cái đầu, rồi sắp xếp theo cột.
- Mốc so sánh sắp xếp có sẵn cũng được kiểm tra, trên vài kích thước: k phần tử đầu của mỗi hàng, đặt
  theo thứ tự cột, phải bằng bản tham chiếu.

## Trước khi bạn benchmark

Hãy ghi lại một dự đoán (bài học hỏi nó dưới tên `x1/labs/p-lab1-sort`): ở 64 hàng gồm 151,936 logits
và k = 40, phép chọn của bạn sẽ nhanh hơn phép sắp xếp đầy đủ của lab 14.3 bao nhiêu lần?

Mục X1.4 đếm số byte mỗi thiết kế phải di chuyển cho mỗi khóa: bốn lượt của phép sắp xếp, mỗi lượt đọc
khóa 4 byte để đếm nó, rồi đọc và ghi cặp (key, index) 8 byte để phân tán nó, tổng 80 byte; phép chọn
đọc khóa bốn lần để đếm và một lần để gom, 20 byte, cộng 8 byte cho mỗi cặp được giữ. Tức là lưu lượng
ít hơn khoảng 4 lần. Hãy nói bạn kỳ vọng tỉ lệ thời gian cao hơn hay thấp hơn 4, và vì sao. Ba điều cần
cân nhắc: bước phân tán của phép sắp xếp đi tới 256 bucket, nên các lần ghi của nó được hợp nhất kém
(README của lab 14.3); các phép toán nguyên tử của phép chọn ở các lượt sau đều dồn vào vài ngăn gần
điểm vượt ngưỡng; và cả hai thiết kế chạy một khối cho mỗi hàng, nên ở 64 hàng không thiết kế nào lấp
đầy một thiết bị lớn, và ở 1 hàng cả hai chỉ dùng một multiprocessor (chương 10, mục 10.10).

Benchmark chạy rows 1, 8 và 64 nhân cols 32,000 và 151,936 nhân k = 1, 40 và 1,024, trên logits phân
bố đều trong [-12, 12] với 2^24 mức. Nó báo cáo, cho mỗi lần gọi:

- phép chọn của bạn (cả hai kernel), đo thời gian trên device, lấy tốt nhất của 5 lần chạy, mỗi lần lặp
  lại lời gọi cho đến khi kéo dài ít nhất 2 ms;
- toàn bộ phép sắp xếp của lab 14.3 (khóa, bốn lượt, gom), tốt nhất của 5 lần chạy, mỗi lần ít nhất
  20 ms (lab 14.3 thấy các lần chạy ngắn hơn bị dao động); top k là k phần tử đầu của mỗi hàng đã sắp
  xếp, nên lấy chúng không tốn gì;
- một luồng CPU chạy `std::partial_sort` trên các cặp (id, logit) cho mỗi hàng, như top-k trên host của
  llama.cpp làm với k tới 128 (D3), đo trên host bằng một đồng hồ ổn định;
- số hàng mỗi giây, số byte mỗi khóa ở trên, và các tỉ lệ sort/select và CPU/select.

Mọi kết quả đo được đều được kiểm tra trước khi báo cáo (`std::partial_sort` không ổn định, nên kết
quả của nó được kiểm tra theo giá trị), và một kích thước lớn hơn một phần tư bộ nhớ device được báo
cáo sẽ bị bỏ qua. Các con số là phép đo trên máy của bạn, không phải mục tiêu.

## Tính khả chuyển

- HIP: `kernel.cu` được viết để build qua `pmpp/vendors/hip.h`. Phép bỏ phiếu là 64 bit trên mọi đích
  AMD và `__ballot_sync` cùng `__shfl_*_sync` nhận mặt nạ 64 bit (infra-facts B3), nên mặt nạ lane có
  kiểu riêng và popcount cùng find-first-set khớp với nó (`__popcll`, `__ffsll`); `__popc` trên một
  phiếu 64 bit sẽ âm thầm chỉ đếm 32 lane thấp (ch12-facts B1). Với warp rộng 64, lượt đi qua các ngăn
  mất bốn bước 64 ngăn thay vì tám bước 32 ngăn. Chưa được biên dịch hay chạy trên phần cứng AMD ở đây:
  unverified.
- Metal: histogram là `threadgroup atomic_uint`, đọc bằng `atomic_load_explicit` (ch09-facts C1).
  `simd_ballot` trả về một phiếu 64 bit mà các bit trên độ rộng SIMD group không xác định
  (ch12-facts C2): thứ hạng `popcount(vote & lanes below)` không bao giờ đọc chúng, số đếm đến từ
  `simd_sum`, và lane bật đầu tiên được tìm sau khi che phiếu theo các lane thực sự tồn tại. Các luồng
  lấy cột của mình từ `sg * width + lane`, vì các phép scan chạy theo thứ tự SIMD group và cách một
  threadgroup chia thành các SIMD group là do bản cài đặt quyết định (ch04-facts C2). `launch.mm` đọc
  độ rộng từ pipeline (`threadExecutionWidth`) và từ chối dispatch trừ khi 256 luồng tạo thành các SIMD
  group trọn vẹn và độ rộng chia hết 256 ngăn.
- Con đường thư viện: `cub::DeviceTopK` (đầu ra không có thứ tự, giá trị bằng nhau không xác định) và
  `DeviceBatchedTopK` trên CUDA; `rocprim::topk` trên HIP không có lớp bọc hipCUB; MPSGraph `topK` trên
  Metal không nêu thứ tự cho giá trị bằng nhau (A7, B4, C1). Không cái nào hứa thứ tự cột và quy tắc
  giữ cột thấp nhất khi bằng nhau như lab này, đó là lý do M5 có bản riêng.
- Bước kiểm tra front-end CUDA cần thư mục include shim mà `labs/pmpp.py` truyền vào
  (`-I labs/common/include/pmpp/emu/shim`) cho `<cuda/atomic>`.
