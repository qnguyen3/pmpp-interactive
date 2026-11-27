> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 13.1: trộn hai mảng đã sắp xếp bằng cách phân hoạch co-rank

Sách: mục 13.2 đến 13.6 (pp. 305-320), Figs. 13.2, 13.5 và 13.9, và ý tưởng của
Exercise 3 (p. 328). Phần (b) có hình dạng mà CUB, rocPRIM và MLX đang phát hành, không phải
Figs. 13.11 đến 13.13 của sách (`notes/today/ch13-facts.md` A7, B2, D1).

A và B là các mảng đã sắp xếp gồm các cặp khóa-giá trị, `struct KV { int key; int val; }` (trong
`merge.h`). Bạn ghi `total` đầu ra đầu tiên của phép trộn (merge) ổn định của chúng vào C, chỉ so
sánh khóa. Khi bằng nhau, ưu tiên A, và các phần tử từ cùng một mảng giữ nguyên thứ tự: quy tắc
của sách (pp. 304-305). `total = m + n` là toàn bộ phép trộn. Một `total` nhỏ hơn là của khóa học
này, không phải của sách: đó là phép trộn hai danh sách top-k, nơi chỉ cần k đầu ra đầu tiên.

**Phần của bạn:**

- **(a) Kernel cơ bản** (Figs. 13.2, 13.5, 13.9, đã sửa).
  - `co_rank(k, A, m, B, n)`: trong k đầu ra đầu tiên, bao nhiêu đến từ A. Một phép tìm kiếm nhị
    phân trên `.key`, với min và max viết bằng toán tử ba ngôi và các phép kiểm tra biên đặt trước
    các lần đọc.
  - `merge_sequential`: phép trộn tuần tự hai giai đoạn, với `<=`.
  - `mergeBasicKernel`: mỗi luồng (thread) sở hữu `perThread` đầu ra liên tiếp. Đầu ra đầu tiên
    của nó `k_curr` và `k_next` của luồng kế tiếp được giới hạn ở `total`, và một luồng có
    `k_curr >= total` sẽ trả về. Rồi hai lời gọi `co_rank` và một lời gọi `merge_sequential`.
  - `launchMergeBasic`: khối 128 luồng (`MERGE_BLOCK`), đủ số khối cho `total` đầu ra, đếm bằng
    phép chia làm tròn lên số nguyên. Không khởi chạy gì khi `total` là 0.
- **(b) Trung chuyển qua bộ nhớ chia sẻ (shared memory)**, mỗi khối một ô (tile) gồm
  `128 * perThread` đầu ra.
  - `partitionKernel` lưu `bounds[b] = co_rank(min(b * tile, total), A, m, B, n)`
    cho mọi ranh giới ô `b = 0 .. numTiles`.
  - `mergeTiledKernel`: khối b lấy `A[bounds[b] .. bounds[b+1])` và dải B tạo nên phần còn lại
    của các đầu ra của nó (tìm bằng phép trừ). Nó chép cả hai vào một
    `extern __shared__ KV tile[]` với các luồng liên tiếp trên các phần tử liên tiếp, có một
    `__syncthreads()`, rồi mỗi luồng chạy `co_rank` trên các bản sao chia sẻ và
    `merge_sequential` từ bộ nhớ chia sẻ ra C. Mọi luồng đều đến được rào chắn (barrier): không
    trả về sớm.
  - `launchMergeTiled`: hai lần khởi chạy, phân hoạch rồi đến trộn, với
    `tile * sizeof(KV)` byte bộ nhớ chia sẻ động.

## Các tệp

- Sửa `work/kernel.cu` (CUDA, HIP và cả hai trình giả lập): `co_rank`, `merge_sequential`,
  `mergeBasicKernel`, `partitionKernel`, `mergeTiledKernel`, `launchMergeBasic`,
  `launchMergeTiled`. `coRankKernel` và `launchCoRank` được cung cấp, để kiểm thử có thể gọi
  riêng `co_rank` của bạn.
- Sửa `work/kernel.metal` (`co_rank`, `merge_sequential`, `merge_basic`,
  `merge_partition`, `merge_tiled`; `co_rank_test` được cung cấp) và `work/launch.mm`
  (`encodeMergeBasic`, `encodeMergeTiled`; `encodeCoRank` được cung cấp). Mỗi hàm encode trả về
  false, không encode gì cả, khi `maxTotalThreadsPerThreadgroup` của pipeline dưới 128, hoặc
  (bản phân ô) khi ô không vừa `maxThreadgroupMemoryLength` của thiết bị. Chương trình điều khiển
  in ra mọi thứ nó bỏ qua.
- Đừng sửa `merge.h` hay `main.cpp`.

Trong Metal, `co_rank` và `merge_sequential` đọc bộ nhớ device trong `merge_basic` và bộ nhớ
threadgroup trong `merge_tiled`. Các không gian địa chỉ của Metal tách rời nhau và mỗi con trỏ gắn
với một không gian, nên một hàm trợ giúp `const KV*` duy nhất như trên CUDA thì không làm được.
Starter biến mỗi hàm trợ giúp thành một template theo kiểu con trỏ, như merge sort MPS của PyTorch
làm; MLX thì viết hai overload (ch13-facts C3). Đặc tả MSL cũng không nhắc lại quy tắc của C++ rằng
`&&` bỏ qua toán hạng bên phải (ch13-facts C2), nên trong Metal hãy kiểm tra biên trước và đọc bên
trong `if`.

`encodeMergeTiled` đặt hai dispatch trong một encoder. Runtime mở loại encoder tuần tự mặc định, và
các buffer do device tạo được theo dõi hazard, nên phép trộn bắt đầu sau khi phân hoạch đã ghi
`bounds` (`notes/today/ch02-facts.md` C9). `setThreadgroupMemoryLength` cần một bội của 16 byte
(ch05-facts C8): `tile * 8` luôn là bội đó.

## Lệnh

- Kiểm thử: `python3 labs/pmpp.py run 13.1 --backend emu` trước, rồi `python3 labs/pmpp.py run 13.1`.
- Ô chia sẻ của kernel phân ô: `python3 labs/pmpp.py run 13.1 --backend race`.
- Đo: `python3 labs/pmpp.py run 13.1 --bench` (chỉ thiết bị thật).
- Lời giải tham chiếu: thêm `--solution`.

## Kiểm thử làm gì

Mọi phép kiểm tra đều là bằng nhau chính xác. Một phép trộn chép phần tử và không làm phép tính số
học nào, nên không áp dụng dung sai.

1. **Riêng `co_rank`**, qua kernel được cung cấp. Trên ví dụ xuyên suốt của sách
   (khóa A = 1 7 8 9 10, B = 7 10 10 12) mọi k từ 0 đến 9; các giá trị kỳ vọng đến từ
   `std::merge`, không phải từ một bảng. Rồi các cặp mảng ngẫu nhiên (độ dài 0 đến 40,
   khóa 0 đến 3, nên rất nhiều giá trị bằng nhau) ở mọi k.
2. **Cả hai phép trộn** trên: ví dụ xuyên suốt; (0, 0), nơi không được khởi chạy gì và không được
   sập; (0, 37); (37, 0); (1, 1) với khóa bằng nhau; (100, 100) với mọi khóa bằng nhau; (64, 64)
   với A nằm hoàn toàn dưới B, và ngược lại; (1000, 7); (7, 1000); ngẫu nhiên (300, 257) và
   (4093, 3001), một lần với khóa trong [0, 8) và một lần với khóa phân biệt. Thiết bị thật thêm
   (2^20, 2^20) và (3,000,000, 100,000). Mỗi trường hợp ở `perThread` 1, 3 và 8 (thiết bị thêm
   16), và ở `total` = m + n, (m + n) / 2, 17 và 1, những giá trị không vượt quá m + n.

A[i] có giá trị i và B[j] có giá trị 1,000,000,000 + j, nên mỗi đầu ra cho biết nó đến từ đâu.
Tham chiếu là `std::merge` chỉ so sánh khóa, mà chuẩn C++ đảm bảo là ổn định với dải thứ nhất đứng
trước (ch13-facts A6). Vì vậy một kết quả khớp chính xác của `C[0 .. total)` cũng kiểm tra luôn tính
ổn định. C ban đầu được điền giá trị canh `{INT_MIN, -1}` và có thêm 64 phần tử so với m + n: một
đầu ra bạn chưa từng ghi, và bất cứ thứ gì được ghi tại hoặc sau `C[total]`, đều bị báo.

Mỗi backend bắt được gì:

- `emu` chạy các luồng từng luồng một theo ba thứ tự cố định, điền rác vào bộ nhớ chia sẻ cho mọi
  khối, và dừng ở lần đọc hoặc ghi ngoài biên đầu tiên kèm dòng mã nguồn (AddressSanitizer). Một
  lần đọc `A[-1]` hoặc `B[n]` dừng nó ngay lập tức. Một rào chắn bị thiếu cho đầu ra rác.
- `race` chỉ chạy kernel phân ô, ở cỡ nhỏ: ví dụ xuyên suốt và (300, 257) ở `perThread` 4 và 2
  (2 và 3 khối 128 luồng), với các luồng thực sự đồng thời dưới ThreadSanitizer. Một rào chắn bị
  thiếu được báo là một data race trên ô.
- Thiết bị thật chạy cùng các phép kiểm tra ở kích thước lớn hơn. Không có gì trong lab này kiểm
  tra biên cho các lần đọc trên đó, nên một phép kiểm tra biên sai chỉ lộ ra ở đó nếu nó làm thay
  đổi một đầu ra hoặc thiết bị báo lỗi.

## Đọc một lỗi

- *co_rank(k=6) returned 4, expected 5 (ties go to A)*: phép tìm kiếm của bạn cho câu trả lời mà
  một phép trộn sẽ cho nếu khi bằng nhau ưu tiên B, ở mọi k của trường hợp đó (gợi ý chỉ xuất hiện
  khi đó). Hãy xem phép kiểm tra nào trong hai phép của bạn cho phép khóa bằng nhau.
- *C[6] came from B[1]; a stable merge takes A[4] first (both have key 10)*: khóa đúng, thứ tự
  sai. Hoặc `co_rank` hoặc `merge_sequential` xử lý trường hợp bằng nhau sai cách.
- *C[k] was never written*: một luồng lẽ ra phải tạo ra nó đã không chạy, hoặc tạo ra ít đầu ra
  hơn số nó sở hữu. Hãy đếm số luồng mà lần khởi chạy của bạn tạo ra.
- *C[k] was written, but only the first total = T outputs may be*: có thứ gì đó bị giới hạn ở
  m + n trong khi lẽ ra phải giới hạn ở `total`.
- *C[k] holds {key -842150451, ...}, which is no input element*: một giá trị đọc từ bộ nhớ chia
  sẻ mà chưa luồng nào ghi (trình giả lập điền nó bằng byte 0xCD).
- *AddressSanitizer: heap-buffer-overflow ... in co_rank*: một lần đọc vượt quá một trong hai đầu
  của A hoặc B, thường là một phép kiểm tra biên đứng sau lần đọc mà lẽ ra nó phải bảo vệ.

## Lab này khác trang sách ở đâu

- **Phép chia làm tròn lên số nguyên** (ch13-facts A4). Dòng 03 của Fig. 13.9 tính số lượng mỗi
  luồng là `ceil((m+n)/(blockDim.x*gridDim.x))`. Cả hai toán hạng là số nguyên, nên phép chia đã
  bỏ phần thập phân và `ceil` không thay đổi gì: một số đầu ra không bao giờ được tạo ra. Ở đây số
  lượng mỗi luồng là một đối số và hàm khởi chạy tính lưới từ nó là
  `(total + per*128 - 1) / (per*128)`.
- **`__device__`** cho `co_rank` và `merge_sequential`. Các listing không có qualifier nào; trên
  CUDA và HIP, một hàm không có qualifier chỉ chạy trên host (ch13-facts, "Already settled").
- **`k_curr`**. Sách chỉ giới hạn `k_next`, và dựa vào lần khởi chạy cho các luồng vượt quá cuối.
  Ở đây một luồng có đầu ra đầu tiên tại hoặc sau `total` sẽ trả về.
- **Cặp KV** thay vì `int`, để có thể kiểm thử tính ổn định, và đầu vào `const`.
- **`total`**, tối đa m + n (của khóa học này).
- **Nạp chính xác**, không phải ô gấp 2 của mục 13.6. Kernel phân ô của sách nạp một ô đầy đủ của
  A và một ô đầy đủ của B mỗi vòng lặp và chỉ dùng một nửa những gì nó nạp (p. 318). Lab này tính
  co-rank của mọi khối trước, trong một lần khởi chạy riêng (trong Fig. 13.11, luồng 0 và hai rào
  chắn làm việc này), nên mỗi khối nạp đúng những phần tử nó sẽ dùng: phương án thay thế ở p. 316 và
  ý tưởng của Exercise 3. Mỗi khối một ô, nên không có vòng while và không có việc ghi chép
  `A_consumed`.
- **Không có buffer vòng** (mục 13.7). Nạp chính xác không để lại gì thừa để giữ.
- Trong Metal: template theo kiểu con trỏ và các điều kiện bảo vệ biên tường minh (ở trên).

## Vượt ra ngoài sách

- Các thư viện gọi `co_rank` là "merge path" (CUB, rocPRIM) hoặc "merge partition" (MLX,
  PyTorch MPS) (ch13-facts A1). Phần (b) là thiết kế của họ: một lượt phân hoạch, rồi nạp chính xác
  vào bộ nhớ chia sẻ (A7, B2, D1). Merge MPS của PyTorch không có lượt phân hoạch: mọi luồng của một
  khối tự tính phân hoạch của khối, trong thanh ghi, điều mà chú thích của nó gọi là "same result,
  no barriers" (D2).
- `cub::DeviceMerge` được ghi trong tài liệu là không ổn định, `thrust::merge` là ổn định, và backend
  CUDA của Thrust gọi CUB (A7). Nếu tính ổn định quan trọng với bạn, hãy kiểm tra thư viện hứa gì,
  chứ không phải mã của nó tình cờ làm gì.
- Chế độ `total < m + n` là cách backend Metal của llama.cpp tính top-k cho các hàng ngắn: nó giữ
  top k của mỗi khối, rồi trộn các danh sách theo từng cặp bằng một kernel có hình dạng giống
  Fig. 13.9 dừng ở `top_k` (D3). Một danh sách top-k được sắp xếp lớn nhất trước, nên nó được trộn
  với khóa đổi dấu hoặc phép so sánh đảo ngược. Module X1 xây top-k của nó bằng radix select thay vì
  vậy (mục X1.4, "Top-k is a selection, not a sort"), nên không có gì ở đó phụ thuộc vào chế độ này.

## Thí nghiệm

Khi mọi kiểm thử đã qua, và trước khi chạy `--bench`, hãy viết ra hai con số. Trang labs ghi lại
chúng cho bạn.

1. Với kernel cơ bản, `perThread` 8 nhanh hơn (hoặc chậm hơn) `perThread` 1 bao nhiêu lần? Ở 1,
   mỗi đầu ra trả giá cho hai lần tìm kiếm độ sâu log2, nhưng các luồng cạnh nhau ghi vào các địa
   chỉ cạnh nhau. Ở 8, một lần tìm kiếm được chia cho 8 đầu ra, nhưng ở mỗi bước của vòng lặp trộn,
   các luồng của một warp (một SIMD group trên Metal) chạm vào các địa chỉ cách nhau 8 phần tử.
2. Ở `perThread` 8, kernel phân ô nhanh hơn kernel cơ bản bao nhiêu lần?

Rồi chạy `python3 labs/pmpp.py run 13.1 --bench` và so sánh. Mô hình của bài học đếm số lần truy
cập và bỏ qua bộ nhớ đệm (cache), và mỗi luồng của kernel cơ bản duyệt lát của riêng nó theo thứ
tự, nên bộ nhớ đệm có thể hấp thụ phần lớn sự phân tán. Không câu trả lời nào được hứa trước.

`--bench` làm gì: các khóa là số nguyên 31-bit ngẫu nhiên phân bố đều, đã sắp xếp, với m = n. Nó
tăng m từ 2^20 cho đến khi kernel cơ bản ở `perThread` 8 mất ít nhất 20 ms, hoặc mỗi đầu vào đạt
2^23 phần tử (hoặc các mảng sẽ vượt quá một phần tư bộ nhớ mà thiết bị của bạn báo), với
`total = m + n`. Rồi nó đo kernel cơ bản ở `perThread` 1, 2, 4, 8 và 16 và kernel phân ô ở 4, 8
và 16 (lần tốt nhất trong 5 lần chạy sau một lần khởi động), kiểm tra mọi kết quả với CPU, và in ra
mili giây và GB/s. GB/s là `2 x total x 8 bytes / time`: đọc mỗi phần tử một lần và ghi nó một lần,
lưu lượng ít nhất mà bất kỳ phép trộn nào có thể tạo ra, không phải lượng mà kernel thực sự đã di
chuyển. Thời gian của bản phân ô gồm cả lần khởi chạy phân hoạch. Hãy so sánh GB/s với hàng băng
thông mà lab 5.1 đo trên cùng thiết bị. Một hàng mà thiết bị không chạy được sẽ được in là bị bỏ
qua, kèm lý do.
