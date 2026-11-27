> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 6.2: nạp vector, với phần biên được làm đúng

Sách: mục 6.3 (pp. 138-139), Fig. 6.12 và Exercise 4 (p. 155); mục 6.5 (pp. 141-143) cho vòng
lặp gộp việc.

Fig. 6.12 viết lại phép cộng vector sao cho mỗi luồng (thread) cộng bốn cặp số float với một lần
nạp 16 byte cho mỗi đầu vào và một lần ghi 16 byte. Nó khởi chạy một phần tư số luồng và để phần
biên cho Exercise 4: như in, nó đọc quá cuối mỗi khi n không phải là bội của 4. Rồi mục 6.5 gọi
chính kernel đó là một ví dụ của gộp việc cho luồng (thread coarsening). Lab này tách hai ý tưởng
ra. Bạn viết hai kernel trong đó mỗi luồng sở hữu cùng bốn phần tử, `4i` đến `4i+3`, và đo thời
gian chúng so với mỗi luồng một số float.

**Phần việc của bạn:**

- `vecAdd4Kernel` và `launchVecAdd4`: bốn phần tử mỗi lệnh. Một cụm đầy đủ được cộng bằng một lần
  nạp `float4` từ `x`, một từ `y` và một lần ghi `float4` vào `z` (phép ép kiểu của sách). Một cụm
  cuối không đầy (1 đến 3 phần tử) quay về xử lý từng số float một. Một luồng có cụm bắt đầu quá
  cuối mảng thì không làm gì. Khởi chạy `ceil(ceil(n / 4) / 256)` khối 256 luồng.
- `vecAddCoarseKernel` và `launchVecAddCoarse`: bốn phần tử mỗi luồng, mỗi lần truy cập một số
  float. Đây là vòng lặp gộp việc của mục 6.5 với hệ số 4: `iStart` bằng 4 lần chỉ số luồng toàn
  cục, vòng lặp chạy `c` từ 0 đến 3 dưới `#pragma unroll`, và mỗi phần tử được kiểm tra với n.

## Tệp

- Sửa `work/kernel.cu` (CUDA, HIP, trình giả lập): hai kernel và các lần khởi chạy của chúng.
- Sửa `work/kernel.metal` (`vec_add4`, `vec_add_coarse`) và `work/launch.mm` (`encodeVecAdd4`,
  `encodeVecAddCoarse`). `vec_add4` khai báo ba buffer của nó là `float4` cho các cụm đầy đủ và
  nhận lại chính ba buffer đó dưới dạng `float` thường cho phần đuôi, nên `encodeVecAdd4` gắn mỗi
  buffer hai lần, ở hai chỉ số.
- Được cung cấp, không sửa: bản cơ sở mỗi luồng một số float trong `provided/vecadd.cu`,
  `provided/vecadd.metal` và `provided/vecadd_launch.mm`.

## Lệnh

- Kiểm thử: `python3 labs/pmpp.py run 6.2 --backend emu`, rồi `python3 labs/pmpp.py run 6.2`.
- Đo: `python3 labs/pmpp.py run 6.2 --bench` (chỉ trên thiết bị thật).

## Các bài kiểm thử làm gì

Cả hai kernel chạy với n = 1, 2, 3, 4, 5, 7, 8, 9, 255, 256, 257, 1023, 1024 và 1025, và trên GPU
còn với 2^20, 2^20 + 3 và 10,000,001. Mọi số dư của n mod 4 đều được phủ, với một khối, nhiều
khối, và một khối cuối không đầy.

- Mỗi mảng chứa đúng n số float, không có gì dư phía sau. Trên trình giả lập (emulator), một lần
  truy cập `float4` vươn quá cuối mảng là một lỗi AddressSanitizer có nêu tên dòng. Trên GPU, cùng
  sai lầm đó có thể không bị phát hiện, và đó là lý do bạn bắt đầu với `emu`.
- `z` bắt đầu là NaN, nên một phần tử kernel của bạn không bao giờ ghi sẽ bị báo, và thông báo cho
  biết nó nằm trong một cụm đầy đủ hay trong cụm cuối không đầy.
- Kết quả được so với một bản tham chiếu CPU, trong một dung sai.
- Các con trỏ chính là các vùng cấp phát, nên chúng được căn chỉnh 16 byte. Không bài kiểm thử nào
  truyền một con trỏ lệch căn chỉnh như `x + 1`: trình giả lập không phân biệt được một lần truy
  cập `float4` lệch căn chỉnh với một lần đúng căn chỉnh, nên một bài kiểm thử như vậy không thể
  thất bại ở đó.

## Căn chỉnh, theo từng backend

Phép ép kiểu cần một con trỏ căn chỉnh 16 byte. Những gì mỗi nhà sản xuất công bố
(`notes/today/ch06-facts.md` và `notes/today/infra-facts.md`):

- CUDA: các hướng dẫn tham chiếu của NVIDIA không mô tả phép ép kiểu này. Blog dành cho nhà phát
  triển của NVIDIA thì có, và yêu cầu con trỏ được ép phải căn chỉnh theo kích thước của vector
  (A11). `cudaMalloc` trả về vùng nhớ căn chỉnh ít nhất 256 byte (A3).
- HIP: `float4` tồn tại với tên của CUDA và có căn chỉnh 16 byte trong header; cũng không trang
  nào của AMD mô tả phép ép kiểu này (B5, infra B6).
- Metal: cách được công bố là khai báo buffer `device const float4*` (C3, infra C7), và đó là điều
  `vec_add4` làm. Một phép ép con trỏ vẫn biên dịch được, và MLX cùng llama.cpp dùng nó, nhưng đặc
  tả không mô tả nó. Apple cũng nói chỉ số `uint` có thể ngăn các lần nạp được vector hóa, nên các
  kernel Metal đánh chỉ số bằng `int` (C3). GPU thực sự thực hiện bao nhiêu lần truy cập cho một
  `float4` thì không được công bố.

## Trước khi đo hiệu năng, hãy dự đoán

Viết những điều này ra trước.

1. Mức tăng tốc của `vec4` so với `scalar` ở kích thước lớn nhất mà benchmark đạt tới. Lập luận
   của sách là hai lần nạp vector thay cho tám lần nạp vô hướng, ít hơn 75 % lệnh nạp cho cùng dữ
   liệu (p. 139). Sách cũng nói kernel vô hướng đã được hợp nhất truy cập và chạy ở occupancy tối
   đa (p. 138), và nó chuyển 12 byte cho mỗi phép cộng. Hãy quyết định bao nhiêu phần tiết kiệm về
   lệnh biến thành thời gian.
2. `coarse4` so với `scalar`: nhanh hơn, bằng, hay chậm hơn? Trước khi trả lời, hãy áp quy tắc của
   mục 6.1 cho một lệnh nạp của mỗi kernel: địa chỉ của các luồng kề nhau cách nhau bao xa? (So
   sánh với Exercise 1, pp. 154-155.)

## Đọc đầu ra

Benchmark tăng n từ 2^22 cho đến khi bản cơ sở chạy quá 5 ms, tối đa 2^26, giữ ba mảng trong phạm
vi một phần tư bộ nhớ mà thiết bị của bạn báo cáo. Nó in mỗi kernel một hàng ở n đó:

- `time ms`: kết quả tốt nhất trong năm lần chạy.
- `GB/s`: 12 byte mỗi phần tử chia cho thời gian. Nó đếm các byte mà phép cộng cần, không phải
  những gì hệ thống bộ nhớ đã chuyển.
- `vs scalar`: thời gian của bản cơ sở chia cho thời gian của kernel này.

Các con số là phép đo của ba kernel trên máy của bạn ở một kích thước. Mục 6.3 của bài học đọc
chính những hàng này.

## Đi xa hơn

Các kernel theo từng phần tử của PyTorch cho CUDA và ROCm làm điều Exercise 4 yêu cầu, thêm một
bước nữa: chúng không giả định căn chỉnh. `aten/src/ATen/native/cuda/MemoryAccess.cuh` kiểm tra
địa chỉ của mỗi tensor lúc chạy (`can_vectorize_up_to` trả về 8, 4, 2 hoặc 1) và chọn độ rộng
vector từ đó, quay về truy cập vô hướng khi cần (ch06 facts D4). Backend CUDA và HIP của llama.cpp
chép 16 byte mỗi lần qua cùng loại phép ép kiểu và có một hàm trợ giúp kiểm tra căn chỉnh của
tensor trước (D1). Một bài tập cho riêng bạn: `vecAdd4Kernel` sẽ cần gì để chấp nhận `x + 1`?
