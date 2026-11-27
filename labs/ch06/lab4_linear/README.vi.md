> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 6.4: lớp tuyến tính theo bố cục của framework, có rẽ góc (mốc M2)

Sách: rẽ góc (corner turning), mục 6.1 và Fig. 6.4 (pp. 129-130); đệm ô đã xoay, mục 6.4
(pp. 139-141); Exercise 2 (p. 155).

Lab 5.3 tính C = A B với B được lưu theo (k, l). Các framework không lưu một lớp tuyến tính theo
cách đó: `nn.Linear` của PyTorch giữ trọng số dưới dạng (out_features, in_features) và tính
y = x W^T, và `ggml_mul_mat` của llama.cpp dùng cùng bố cục đó. Đọc theo cách của lab 5.3, W^T đó là
N theo thứ tự cột (column-major) của sách (p. 128): các luồng (thread) kề nhau nạp các số float cách
nhau `in`.

Hãy viết `linearKernel(X, W, Y, batch, in, out)` và `launchLinear`:
`Y[batch x out] = X[batch x in] * W^T`, với W được lưu (out, in) theo thứ tự hàng. Bắt đầu từ kernel
lab 5.3 của bạn (mã khởi đầu (starter) là chính kernel đó được sửa lại cho các đối số này: mọi kích
thước, mọi lần nạp và lần ghi đều có kiểm tra biên, điền số 0). Hai điều thay đổi, cả hai trong ô W:

1. **Rẽ góc.** Nạp ô W sao cho các `threadIdx.x` kề nhau đọc các số float kề nhau của một hàng của
   W, rồi đặt mỗi giá trị vào chỗ mà vòng lặp tính tích chờ nó. Đây là Exercise 2 của sách trong
   bố cục mà dự án cần.
2. **Đệm.** Lần ghi sau khi xoay ghi xuống một cột của ô chia sẻ. Khai báo ô là
   `[TILE_WIDTH][TILE_WIDTH + 1]` để cột đó không rơi vào cùng các bank.

**Quyết định về bố cục, từ đây trở đi.** Trọng số của dự án được lưu (out, in) theo thứ tự hàng,
như `nn.Linear` và ggml lưu chúng; tensor là `float` theo thứ tự hàng với batch đứng đầu; kernel
nhận con trỏ device thô và kích thước tường minh. `launchLinear(X, W, Y, batch, in, out)` là chữ ký
mà chương 15 xây dựng tiếp lên.

- Sửa `work/kernel.cu` và `work/kernel.metal` (`work/launch.mm` đã hoàn chỉnh).
- `TILE_WIDTH` mặc định là 16; một bản dựng 32 cũng được tạo cho benchmark, và Metal biên dịch 16 và
  32 lúc chạy. Đừng viết cứng 16.
- Kiểm thử: `python3 labs/pmpp.py run 6.4`   Bắt đầu với `--backend emu`. Các bài kiểm thử chỉ kiểm
  tra kết quả: lần nạp không được hợp nhất của bản cơ sở được cung cấp cũng đạt. Lần nạp của bạn có
  được hợp nhất hay không thì hiện ra trong benchmark (trên NVIDIA, Nsight Compute đếm nó dưới dạng
  số sector mỗi yêu cầu).
- Đo: `python3 labs/pmpp.py run 6.4 --bench` đo thời gian bản cơ sở được cung cấp
  (`provided/linear_baseline.*`: cùng kernel phân ô nhưng đọc `W[Col * in + k]`, đúng nhưng không
  được hợp nhất) và kernel của bạn trên lớp 1024 -> 4096 của lab 5.3 ở batch 1 đến 1024. Hai kernel
  chạy luân phiên, và kết quả tốt nhất trong vài lần chạy của mỗi kernel được giữ lại.

Trước khi đo hiệu năng, hãy dự đoán mức tăng tốc của kernel bạn so với bản cơ sở ở batch 256,
T = 16. Những điều cần cân nhắc:

- NVIDIA đếm lưu lượng bộ nhớ theo các sector 32 byte. Ở T = 32, lần nạp W của một warp chạm 32
  sector ở bản cơ sở và 4 sector khi có rẽ góc: ít hơn 8 lần. Ở T = 16, một warp 32 luồng phủ hai
  hàng của khối: hãy tính con số ở đó.
- Lần nạp W là một phần của mỗi giai đoạn. Lần nạp ô X, các rào chắn (barrier) và vòng lặp tính tích
  như nhau ở cả hai kernel.
- So sánh những gì phân ô (tiling) mang lại cho bạn ở lab 5.2 với mức lưu lượng giảm T lần mà mô
  hình của nó hứa. Bộ nhớ đệm (cache) có thể phục vụ một phần lưu lượng mà mô hình tính phí. Ở đây,
  trong bản cơ sở, những warp nào khác của cùng khối cần phần còn lại của mỗi sector mà một warp nạp?
- Trên Metal không gì dự đoán được câu trả lời: Apple không công bố quy tắc hợp nhất truy cập nào,
  cũng không công bố bank bộ nhớ threadgroup nào, và để việc chia một threadgroup thành các SIMD-group cho cài
  đặt quyết định. Phép đo của bạn là câu trả lời.

Sau đó, bỏ cột đệm và chạy lại `--bench`. Phần đệm đã mang lại gì trên thiết bị của bạn, ở T = 16 và
ở T = 32? Phép tính bank của mục 6.4 dự đoán có sự khác biệt giữa hai trường hợp trên NVIDIA; hãy
tính xem trường hợp nào trước khi bạn xem.
