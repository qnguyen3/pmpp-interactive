> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 18.1: BFS theo frontier với hàng đợi riêng tư hóa

Sách: các mục 18.5 đến 18.7 (pp. 438-448), Figs. 18.14, 18.15 và 18.17.

Hãy viết tìm kiếm theo chiều rộng theo cách mục 18.6 xây dựng nó: mỗi mức một lần khởi chạy, mỗi
đỉnh của frontier trước một luồng (thread), và mỗi khối (block) gom các đỉnh nó tìm thấy vào một
frontier riêng trong bộ nhớ chia sẻ (shared memory) trước khi chép chúng sang frontier công khai bằng một phép
toán nguyên tử (atomic) cho mỗi khối. Sau đó, tùy chọn, gộp mọi mức vào một lần khởi chạy với các rào chắn (barrier) toàn lưới
(mục 18.7).

Đồ thị ở dạng CSR (p. 427): các láng giềng ra của `v` là từ `dst[srcPtrs[v]]` đến
`dst[srcPtrs[v + 1] - 1]`. `level` là một mảng `unsigned int`; `level[root]` là 0 và mọi
phần tử khác là `UNVISITED` khi mã của bạn bắt đầu. Sau đó, mọi đỉnh mà gốc có thể tới
chứa mức của nó, và mọi đỉnh khác vẫn là `UNVISITED`.

**Phần A (bắt buộc, mọi backend).**

- `visitVertexAtomically` (Metal: `visit_vertex`), Fig. 18.14. "Vẫn chưa được thăm? Vậy nó là
  của tôi, ở mức này" dưới dạng một phép compare-and-swap, đúng với chính xác một trong các luồng thử.
- `bfsPrivatisedKernel` (Metal: `bfs_privatised`), Fig. 18.15. Một frontier riêng gồm
  `PRIVATE_FRONTIER_CAPACITY` ô và một bộ đếm riêng trong bộ nhớ chia sẻ; một phép toán nguyên tử
  phạm vi khối trên bộ đếm riêng; một đường tràn ghi thẳng vào frontier
  công khai khi frontier riêng đã đầy; ba rào chắn khối; một phép toán nguyên tử phạm vi device cho mỗi
  khối để giữ chỗ dải của nó trong frontier công khai; một lần ghi nhận trong đó các luồng lân cận
  ghi vào các địa chỉ lân cận.

**Phần B (tùy chọn; CUDA, HIP và các trình giả lập).**

- `bfsCooperativeKernel`, Fig. 18.17: vòng lặp mức chuyển vào trong kernel. Lưới (grid)
  nhỏ hơn một frontier lớn, nên mỗi luồng đi qua frontier trước theo bước nhảy (stride)
  `grid.num_threads()`, và ba lời gọi `grid.sync()` kết thúc mỗi mức.
- Các dòng được đánh dấu của `launchBfsCooperative`: đặt kích thước lưới sao cho mọi khối đều thường trú
  cùng lúc (một truy vấn occupancy nhân với số SM, pp. 445-446), và khởi chạy nó bằng
  `cudaLaunchCooperativeKernel`.

Metal không có phần B: tài liệu của nó không nêu rào chắn toàn lưới, lệnh khởi chạy cooperative hay
truy vấn occupancy nào để đặt kích thước một lưới thường trú (`notes/today/ch18-facts.md` C1, ch04
C3 và C5). Trên Metal, lần tìm kiếm là một dispatch cho mỗi mức.

## Các tệp và lệnh

- Sửa `work/kernel.cu` (cả hai phần) và `work/kernel.metal` (phần A).
- Đã viết sẵn cho bạn: `launchBfsPrivatised` trong `work/kernel.cu` (đặt bộ đếm về 0, khởi chạy
  `ceil(numPrevFrontier / BLOCK_DIM)` khối, đọc số đếm về, hoán đổi các frontier), phép
  kiểm tra hỗ trợ thiết bị ở đầu `launchBfsCooperative`, và `work/launch.mm` (cùng
  vòng lặp theo mức cho Metal: mỗi mức một command buffer, `waitUntilCompleted`, rồi một lần đọc
  số đếm). Bạn không cần sửa `launch.mm`.
- `provided/ref_kernels.cu`, `.metal` và `ref_launch.mm` chứa các đường cơ sở của benchmark,
  mà bạn không sửa: `push` (Fig. 18.6, mọi đỉnh ở mọi mức, với phép toán nguyên tử relaxed
  thay cho race vô hại của trang), `frontier` (Figs. 18.12 và 18.14, một phép toán nguyên tử công khai
  cho mỗi đỉnh tìm thấy) và `floor` (một kernel rỗng). Chúng cũng được kiểm thử.
- `BLOCK_DIM` là 128 và `PRIVATE_FRONTIER_CAPACITY` là 256 (1 KB bộ nhớ threadgroup trên
  Metal), cả hai được định nghĩa bởi bản dựng. Sức chứa lớn hơn khối, nên một khối có thể
  chứa nhiều mục hơn số luồng nó có.
- Kiểm thử: `python3 labs/pmpp.py run 18.1`. Bắt đầu với `--backend emu`, rồi `--backend race`.
- Đo (chỉ GPU): `python3 labs/pmpp.py run 18.1 --bench`.

Trên HIP, `<cuda/atomic>` đến từ libhipcxx, thứ mà bản dựng HIP của khóa học này không dựa vào
(`notes/today/infra-facts.md` B2), và libhipcxx hiện thực `compare_exchange_strong` bằng một
compare-and-swap yếu ở phạm vi khối và device (ch18-facts B4). Vì vậy, cạnh mỗi
`cuda::atomic_ref`, hãy viết lời gọi legacy dưới `#if defined(__HIPCC__)`: `atomicCAS`
(mạnh, và nó trả về giá trị cũ, không phải bool), `atomicAdd` và `atomicExch`, tất cả đều
relaxed và hợp lệ trên bộ nhớ chia sẻ và bộ nhớ toàn cục (infra B1), như trong lab 12.1. Header
cooperative-groups có đường dẫn khác trên HIP; `#if` cho nó đã được viết sẵn cho bạn.

Trên Metal, phép compare-and-exchange duy nhất là `atomic_compare_exchange_weak_explicit`, có thể
thất bại dù các giá trị bằng nhau (ch18-facts C4). Sau một lần thất bại thật, `expected` chứa
giá trị trong bộ nhớ; sau một lần thất bại giả, nó vẫn chứa giá trị bạn đã truyền vào. Hãy lặp dựa trên điều đó.
Các hàm atomic chỉ nhận kiểu atomic, nên `level` là `device atomic_uint*` và bộ đếm
riêng là một `threadgroup atomic_uint`, được đặt về 0 bằng `atomic_store_explicit`. `vertex` là một
từ khóa dành riêng trong MSL; các tệp gọi nó là `vertexId`.

## Lab này khác trang sách ở đâu

- **Dòng 29 của Fig. 18.15** (dòng 35 của Fig. 18.17). Khi tràn, trang đặt bộ đếm
  riêng về lại sức chứa bằng một lần ghi thường, `numCurrFrontier_s =
  PRIVATE_FRONTIER_CAPACITY;`, trong khi các luồng khác của khối có thể đang làm `fetch_add` trên
  nó. Đó là một data race trong mô hình bộ nhớ của CUDA, khiến nó thành hành vi không xác định,
  bất kể phần cứng làm gì (ch18-facts A7). Hãy ghi qua cùng `atomic_ref` đó (relaxed),
  hoặc bỏ lần ghi và kẹp số đếm bằng `min` sau rào chắn. Phép liệt kê mọi cách xen kẽ
  của chúng tôi cho các trường hợp nhỏ cho thấy phép tính đúng với lần ghi nguyên tử (E1,
  phép kiểm tra của chúng tôi, không phải của nhà cung cấp). Trên Metal, bộ đếm là một kiểu atomic, nên chỉ có thể viết
  lần ghi nguyên tử.
- **Giá trị canh.** Trang gợi ý `UNVISITED` là "`cuda::std::numeric_limits<unsigned>`"
  (p. 433), vốn chỉ nêu tên trait mà thiếu `::max()` (A9).
- **`UINT_MAX`.** Ở đây `UNVISITED` là `UINT_MAX` từ `<climits>`: cùng giá trị, và nó
  không cần thư viện nào, trong khi `<cuda/std/limits>` có trên CUDA và trong libhipcxx nhưng không có trong
  trình giả lập CPU. Metal viết nó là `numeric_limits<uint>::max()` (C5).
- **`__shared__` được đưa lên đầu.** Fig. 18.17 khai báo các biến chia sẻ bên trong vòng lặp mức.
  Một biến `__shared__` sống lâu bằng khối, không phải một lượt của vòng lặp, nên hãy khai báo
  chúng ở đầu kernel; chính lần đặt lại ở đầu mỗi mức mới làm rỗng
  frontier riêng.
- **Sáu đối số.** Trang gọi `cudaLaunchCooperativeKernel` với bốn đối số. CUDA
  chấp nhận điều đó (hai tham số cuối của nó có giá trị mặc định, A1); HIP thì không (B1). Hãy truyền đủ sáu:
  `(..., kernelArgs, 0, 0)`.
- **Rank 64 bit.** `grid.thread_rank()` và `grid.num_threads()` trả về `unsigned long long`
  trên CUDA (A4) và giá trị 32 bit trên HIP (B2). Hãy ép kiểu chúng sang `unsigned int` một lần; trang
  thu hẹp chúng một cách im lặng trong phần đầu vòng lặp.
- **Từ canh trên frontier công khai.** Không mã nào kiểm tra sức chứa của frontier công khai,
  và trang không cho thấy việc cấp phát nó (ghi chú "Unclear in the PDF"). Ở đây mỗi buffer frontier
  công khai có chỗ cho `numVertices` mục, đó là tất cả những gì một lần tìm kiếm thăm mỗi
  đỉnh một lần có thể cần, và chương trình điều khiển (driver) ghi các từ canh sau cả hai buffer và kiểm tra
  chúng sau mỗi lần tìm kiếm. Một đỉnh bị thêm hai lần có thể đẩy một frontier vượt quá cuối của nó; các từ
  canh sẽ cho biết điều đó.
- **Con trỏ thô.** `srcPtrs`, `dst` và `numVertices` được truyền nguyên như vậy; struct
  `CSRGraph` của sách không bao giờ được định nghĩa. Mọi mã trong chương đều tên là `bfs_kernel`;
  các tên ở đây là của khóa học này.

## Các phép kiểm thử làm gì

Mỗi trường hợp so sánh chính xác `level` với một BFS tuần tự trong `main.cpp` (mức là số
nguyên, nên dung sai bằng 0), và kiểm tra các từ canh sau cả hai buffer frontier
công khai. Với các lần tìm kiếm mỗi mức một lần khởi chạy, nó cũng kiểm tra số lần khởi chạy và
kích thước frontier của mỗi mức so với số đỉnh ở mức đó của lần tìm kiếm tuần tự:
nhiều hơn nghĩa là một đỉnh bị thêm hai lần, ít hơn nghĩa là một đỉnh bị mất.

Các đồ thị: đồ thị Fig. 18.1 của sách từ gốc 0 và 2 (các mức của Fig. 18.4); đồ thị
Exercise 1 từ đỉnh 0 (cách chúng tôi đọc hình vẽ ở p. 450); một đỉnh duy nhất; một
gốc không có cạnh ra; hai thành phần liên thông; khuyên và cạnh lặp; một chuỗi (300
đỉnh trên GPU, 100 trên các trình giả lập); một lưới với 4 láng giềng mỗi đỉnh (24 x 24 trên
các trình giả lập, 1024 x 1024 trên GPU); "fan-out", trong đó 128 con tìm thấy 512 cháu
trong một khối so với 256 ô riêng, nên nhiều luồng tràn cùng lúc và lần ghi nhận có
nhiều mục hơn số luồng của khối, và một bản sao với 256 con khiến hai khối
cùng chạy; "fan-out shared", trong đó các con tranh nhau cùng các cháu; một
"stale-expected trap", 0 tới {1, 2}, 1 tới {3}, 2 tới {3, 4}, trong đó luồng của đỉnh 2
thua đỉnh 3 và vẫn phải thắng đỉnh 4; một đỉnh trung tâm bậc 5,000, trong đó một luồng
tràn một mình; và hai đồ thị R-MAT từ các seed cố định (nhỏ trên các trình giả lập). Phần B chạy
cùng các đồ thị đó qua `launchBfsCooperative`; trên `emu` đó là 8 khối 128 luồng,
giới hạn cooperative 1,024 luồng của trình giả lập. Các phép kiểm thử của phần B bị bỏ qua, kèm một thông báo,
khi `launchBfsCooperative` trả về `BFS_COOP_NOT_WRITTEN`, trên Metal, và trên một thiết bị
báo không hỗ trợ lệnh khởi chạy cooperative.

Mỗi backend có thể cho bạn thấy gì:

- `emu` chạy từng luồng một cho đến khi nó tới một rào chắn hoặc tới cuối, theo ba thứ tự.
  Phần lớn các rào chắn khối hoặc lưới bị thiếu sẽ sai ở đó dưới ít nhất một thứ tự, nhưng không phải
  tất cả: với một số rào chắn, lịch thực thi của trình giả lập tình cờ cho ra đáp án đúng, và chỉ
  `race` báo cáo chúng. Một giá trị expected được dùng lại qua các lời gọi sẽ sai ở đó,
  và AddressSanitizer dừng ở một lần truy cập ngoài biên trong bộ nhớ chia sẻ hoặc toàn cục và chỉ ra
  dòng đó. Nó không thể thấy một compare-and-swap bị thiếu hay lần ghi thường của trang ở dòng 29:
  không ai bị ngắt giữa một phép kiểm tra và một lần ghi.
- `race` chạy các luồng thực sự đồng thời dưới ThreadSanitizer và báo cáo hai luồng
  chạm vào cùng vùng nhớ mà không có đồng bộ hóa, kèm cả hai dòng mã nguồn. Đó là nơi
  một phép toán nguyên tử bị thiếu lộ ra.
- GPU của bạn chỉ cho thấy đỉnh trùng hoặc đỉnh bị mất khi lịch thực thi tạo ra chúng, và một
  rào chắn bị thiếu có thể vượt qua ở đó.

## Trước khi đo hiệu năng

`--bench` dựng hai đồ thị với số cạnh xấp xỉ nhau (thu nhỏ lại nếu chúng chiếm
hơn một phần tư bộ nhớ mà thiết bị báo cáo):

- **grid**: 1024 x 1024, 4 láng giềng, gốc ở một góc. 2,046 mức, và không frontier nào
  lớn hơn 1,024 đỉnh: một mạng lưới đường sá thu nhỏ.
- **rmat**: R-MAT từ một seed cố định (bộ sinh và các tham số của nó là của chúng tôi và nằm trong
  `main.cpp`). Một vài mức với frontier lớn: một mạng xã hội thu nhỏ.

Với mỗi đồ thị, nó đo thời gian toàn bộ các lần tìm kiếm, lần tốt nhất trong 5 lần sau một lần chạy không đo, với `level` được đặt lại
ngoài vùng đo thời gian, và kiểm tra mọi kết quả trước khi báo cáo: `push`, `frontier`,
`privatised` của bạn, `cooperative` của bạn (chỉ CUDA và HIP), và `floor`. `floor` chạy cùng
vòng lặp host như các lần tìm kiếm theo mức quanh một kernel rỗng, mỗi mức của đồ thị đó một lần:
đặt bộ đếm về 0, khởi chạy, đọc về 4 byte (trên Metal: một dispatch rỗng, `waitUntilCompleted`,
một lần đọc số đếm). Đó là chi phí cố định của một mức không có tìm kiếm nào bên trong. Mỗi hàng cho
mili giây, micro giây mỗi mức và triệu cạnh được duyệt mỗi giây.

Hãy viết ra ba con số trước:

1. Trên **grid**, phần thời gian của lần tìm kiếm riêng tư hóa của bạn mà `floor` chiếm
   (số giây của `floor` chia cho số giây của `privatised`). Gần 1 nghĩa là chi phí cố định
   của một mức chiếm phần lớn, đó là điều p. 445 nói về các đồ thị như mạng lưới đường sá.
2. Trên **rmat**, `privatised` nhanh hơn `frontier` bao nhiêu lần: mức giảm
   tranh chấp trên một bộ đếm công khai mà riêng tư hóa (privatization) mang lại (p. 442).
3. Trên CUDA hoặc HIP, `cooperative` nhanh hơn `privatised` bao nhiêu lần trên **grid**, và
   trên **rmat**.

Trang labs ghi lại (1), (2) và nửa grid của (3).

Hãy chạy `--bench` nhiều hơn một lần. Một lần tìm kiếm gồm 2,046 mức tí hon phần lớn là chờ host
và device chuyền việc qua lại, và điều đó thay đổi giữa các lần chạy; nếu hai lần chạy
cho tỉ lệ khác nhau, sự khác biệt giữa chúng là một phần của câu trả lời. Sách không đưa ra
con số nào cho bất kỳ điều gì ở đây. Các con số là phép đo trên thiết bị của bạn, không phải của GPU nói
chung.
