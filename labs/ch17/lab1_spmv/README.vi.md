> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 17.1: SpMV dạng CSR và ELL trên các hàng lệch

Sách: mục 17.2 đến 17.5 (pp. 404-415), Figs. 17.5, 17.9 và 17.12.

SpMV là phép nhân ma trận thưa với vector **và cộng dồn**: `y = A·x + y` (p. 403). Bạn viết nó hai
lần, cả hai lần đều mỗi hàng một luồng (thread). Một lần ở dạng CSR, nơi một hàng là một dải các
phần tử nằm giữa hai con trỏ. Một lần ở dạng ELL, nơi mọi hàng được đệm đến cùng độ rộng và hình
chữ nhật đã đệm được lưu theo thứ tự cột (column-major), nên các luồng của một warp đọc các địa chỉ
cạnh nhau ở mọi vòng lặp. Chương trình điều khiển (driver) cũng chạy bản lai của mục 17.5: các hàng
dài hơn K tràn phần đuôi của chúng sang một danh sách COO, mà một kernel được cung cấp cộng vào bằng
phép toán nguyên tử (atomic) trước khi kernel ELL của bạn chạy.

Sau đó benchmark đặt câu hỏi của chương trên thiết bị của bạn: việc đệm, bố cục theo thứ tự cột,
sắp xếp hàng theo độ dài và đẩy các hàng dài sang chỗ khác mang lại gì, trên các ma trận có độ dài
hàng đồng đều, trải rộng, đã sắp xếp, và bị lệch bởi một vài hàng rất dài?

Điều này nằm ở đâu dưới những gì bạn đã biết (của khóa học này, không phải của sách): một batch
lởm chởm là cùng bài toán đó. Đệm mọi chuỗi đến chuỗi dài nhất là ELL; gom chuỗi thành nhóm theo độ
dài là phép sắp xếp JDS; phục vụ vài chuỗi dài qua một đường riêng là bản lai.

## Phần của bạn

- **`spmvCsrKernel`** và **`launchSpmvCsr`**: mỗi hàng một luồng. Lặp qua các phần tử của hàng
  `rowPtrs[row] .. rowPtrs[row+1] - 1`, giữ tổng trong một thanh ghi (register), rồi cộng nó vào
  `y[row]`. Fig. 17.9.
- **`spmvEllKernel`** và **`launchSpmvEll`**: mỗi hàng một luồng, trên các mảng theo thứ tự cột.
  Ô `t` của hàng `r` nằm ở `t * stride + r`, với `stride >= numRows`. Các ô đệm chứa cột -1 và giá
  trị 0. Dừng ở `nnzPerRow[row]` (cách của sách), hoặc đi qua cả `width` ô và bỏ qua những ô có cột
  âm (cách của các thư viện). Rồi cộng tổng vào `y[row]`. Fig. 17.12, với các thay đổi liệt kê bên
  dưới.
- Cả hai hàm khởi chạy: khối gồm `SPMV_BLOCK` luồng (128, từ `spmv.h`), đủ số khối để phủ mọi hàng,
  và hoàn toàn không khởi chạy khi `numRows` là 0.
- Metal: các kernel `spmv_csr` và `spmv_ell`, lấy hàng của chúng từ
  `[[thread_position_in_grid]]`, và `encodeSpmvCsr` và `encodeSpmvEll`, không encode gì khi
  `numRows` là 0 và nếu không thì dispatch mỗi hàng một luồng bằng `dispatchThreads`, trong các
  threadgroup gồm `SPMV_BLOCK` luồng hoặc `maxTotalThreadsPerThreadgroup` của pipeline, lấy giá trị
  nhỏ hơn.

## Lab này khác trang sách ở đâu

- **Con trỏ thô, không phải struct.** Sách truyền một struct `CSRMatrix`, `ELLMatrix` hoặc
  `COOMatrix` cho mỗi kernel và không cho thấy cả struct lẫn mã host nào. Ở đây mỗi mảng là một đối
  số riêng, với kích thước tường minh, ánh xạ một-một vào các buffer binding của Metal
  (`notes/today/ch17-facts.md` A11, C2). Các hàm khởi chạy là của khóa học này.
- **Kernel ELL cộng dồn.** Fig. 17.12 kết thúc bằng `y[row] = sum;`, Fig. 17.9 bằng
  `y[row] += sum;`, và phần chữ gọi cả hai là phép cộng dồn (pp. 409, 412). SpMV là nhân và cộng
  dồn, và mọi thư viện đều tính một phép cộng dồn (ch17-facts A10). Hãy viết `+=` ở cả hai. Đó cũng
  là điều cho phép bản lai chạy phần COO trước: với `=`, kernel ELL của bạn sẽ xóa sạch những gì
  kernel COO đã cộng.
- **Một bước nhảy (stride), không phải numRows.** Fig. 17.12 đi từ một ô của một hàng sang ô kế
  tiếp bằng `numRows`. Ở đây bằng `stride`, có thể lớn hơn. Đó là chính nhận xét của sách (p. 413):
  thêm vài hàng trước khi chuyển vị để mọi vòng lặp bắt đầu ở một địa chỉ được căn lề. Kiểm thử dùng
  `stride = numRows`, `numRows` làm tròn lên 8 và `numRows + 37`.
- **Ô đệm chứa cột -1.** Sách nói giá trị đệm là không nhưng không nói cột của ô đệm chứa gì. Ở đây
  là -1, quy ước của Sliced ELL trong cuSPARSE và ELL trong rocSPARSE (ch17-facts A6, B3). Vậy một
  vòng lặp qua toàn bộ độ rộng phải kiểm tra cột trước khi đọc `x`.
- Kernel COO được cung cấp là Fig. 17.5 đúng như sách viết (`cuda::atomic_ref`,
  `fetch_add`, `memory_order_relaxed`), kèm `#include <cuda/atomic>`, thứ mà listing không cho thấy
  (ch17-facts A1).

## Các tệp

- Sửa `work/kernel.cu` (CUDA, HIP và cả hai trình giả lập):
  - `__global__ void spmvCsrKernel(const int* rowPtrs, const int* colIdx, const float* values, const float* x, float* y, int numRows)`
  - `void launchSpmvCsr(` cùng các đối số đó `)`
  - `__global__ void spmvEllKernel(const int* colIdx, const float* values, const int* nnzPerRow, const float* x, float* y, int numRows, int width, int stride)`
  - `void launchSpmvEll(` cùng các đối số đó `)`
- Sửa `work/kernel.metal` (`spmv_csr`, `spmv_ell`) và `work/launch.mm` (`encodeSpmvCsr`,
  `encodeSpmvEll`). Chữ ký kernel và các buffer binding đã được cho sẵn.
- Được cung cấp, không sửa: `spmv.h` (`SPMV_BLOCK`); `provided/coo_spmv.cu`,
  `provided/coo_spmv.metal` và `provided/coo_spmv.mm` (`launchSpmvCoo` và
  `encodeSpmvCoo`, phần COO của bản lai); `main.cpp` (bộ sinh ma trận, các bộ dựng định dạng,
  tham chiếu CPU, các kiểm thử và benchmark).

## Lệnh

- Kiểm thử: `python3 labs/pmpp.py run 17.1 --backend emu` trước, rồi `python3 labs/pmpp.py run 17.1`.
- Các atomic của bản lai, dưới ThreadSanitizer: `python3 labs/pmpp.py run 17.1 --backend race`.
- Đo: `python3 labs/pmpp.py run 17.1 --bench` (chỉ thiết bị thật).
- Lời giải tham chiếu: thêm `--solution`.

## Kiểm thử làm gì

Mọi định dạng được dựng trên host từ một ma trận được sinh ra, và tham chiếu CPU cộng dồn bằng
double. Một hàng qua khi nó nằm trong `1e-4 × (1 + |y0| + Σ|a·x|)` so với tham chiếu: một dung sai
co giãn theo độ lớn các số hạng của hàng, không bao giờ chỉ là tương đối khi gần không, vì Metal
mặc định không dùng số học IEEE nghiêm ngặt (ch05-facts C7) và một phép cộng nguyên tử float trên
bộ nhớ toàn cục đẩy (flush) các số subnormal về không (ch10-facts A6).

- **Ma trận**, mỗi ma trận được kiểm thử ở CSR và ở ELL: 0 hàng; 1 hàng gồm 12 phần tử trong 20
  cột; 7, 33, 65 và 129 hàng có độ dài ngẫu nhiên (không cái nào là bội của khối); 300 hàng nhân
  305 cột; cứ ba hàng có một hàng rỗng; mọi hàng đều rỗng; mọi hàng dài 5; các hàng dài 1 đến 3 với
  một hàng dài 40; 200 hàng có độ dài ngẫu nhiên từ 0 đến 12. Thiết bị thật thêm bốn ma trận của
  benchmark (bên dưới) ở 65,541 hàng.
- **ELL** ở độ rộng = hàng dài nhất và bước nhảy = `numRows`, `numRows` làm tròn lên 8, và
  `numRows + 37` (cũng làm tròn lên 32 trên các ma trận lớn), và một lần với độ rộng hơn mọi hàng
  3 ô, để có những cột toàn ô đệm.
- **Bản lai** trên bốn trong số các ma trận ở K = 0 (mọi thứ trong COO), 1, độ dài hàng trung vị và
  hàng dài nhất (không có phần COO): kernel COO được cung cấp chạy trước, rồi kernel ELL của bạn
  trên cùng `y`. Kernel COO được cung cấp cũng được kiểm thử riêng, nên một bản sao bị hỏng sẽ lộ ra.
- `y` bắt đầu là các giá trị ngẫu nhiên khác không `y0` và phải kết thúc là `y0 + A·x`. Thông báo
  lỗi cho biết khi mọi hàng sai chỉ chứa `A·x` (bạn đã gán thay vì cộng) hoặc vẫn chứa `y0`
  (không có gì đã chạy).
- `y` có 128 phần tử sau hàng cuối của nó, được đặt một giá trị canh phải còn nguyên. Mọi mảng mà
  kernel của bạn đọc cũng có vài phần tử sau phần cuối, nên một luồng vượt quá hàng cuối vẫn nằm
  trong các buffer, và những gì các phần tử đó chứa khiến nó làm thay đổi vùng bảo vệ của `y`.
- Trên CUDA, HIP, `emu` và `race`, `x` được đưa cho kernel của bạn dưới dạng buffer + 1, và phần
  tử trước `x[0]` là NaN: đọc `x` ở cột -1 của một ô đệm biến hàng thành NaN, và thông báo nói điều
  đó.
- Các ô ELL của các hàng căn lề `numRows .. stride-1` chứa giá trị NaN và cột 0, nên một kernel đi
  theo bước `numRows` thay vì `stride` sẽ thất bại.

Mỗi backend bắt được gì:

- `emu` chạy từng luồng một, theo ba thứ tự cố định, dưới AddressSanitizer: chỉ số sai, thiếu kiểm
  tra hàng, đọc vượt quá cuối một mảng. Nó cũng kiểm tra trường hợp `numRows = 0`: một lần khởi
  chạy không khối nào là lỗi ở đó, như trên HIP (ch02-facts B4). Tài liệu của CUDA không nêu trường
  hợp đó, nên trên CUDA một lần khởi chạy không khối nào có thể lọt qua mà không ai để ý.
- `race` chạy các ma trận 33 và 65 hàng với các luồng đồng thời dưới ThreadSanitizer. Kernel của
  bạn không cần atomic (mỗi hàng do một luồng sở hữu); kernel COO của bản lai thì cần, và một `+=`
  thường ở đó sẽ bị báo kèm dòng mã nguồn của nó.
- **Metal** không có vùng canh NaN trước `x`: fast math mặc định của nó làm việc lan truyền NaN
  không đáng tin cậy, và các hàm encode nhận nguyên buffer, nên không có cách nào có kiểm tra để
  đưa cho kernel `x + 1`. Một kernel đọc `x[-1]` sẽ đọc bất cứ thứ gì nằm trước buffer, điều có thể
  hoặc không lộ ra thành lỗi: đừng trông cậy vào nó, hãy chạy `emu`. Với `dispatchThreads`, không
  có luồng nào vượt quá hàng cuối, nên kiểm tra hàng chỉ được thử đến nếu bạn dispatch nguyên các
  threadgroup. Trong các kiểm thử `numRows = 0`, chương trình điều khiển đặt một bộ đếm vào chỗ
  encoder: mọi dispatch đều được đếm, không được chạy, và làm kiểm thử thất bại.

## Các backend khác nhau ở đâu (không có trong sách)

- **CUDA**: `cuda::atomic_ref<float, cuda::thread_scope_device>` từ `<cuda/atomic>`; hãy truyền thứ
  tự một cách tường minh, như sách làm, vì mặc định là `seq_cst` (ch17-facts A1). Hợp nhất truy cập
  bộ nhớ (coalescing) được tính theo sector 32 byte trên mọi thiết bị mà CUDA hiện tại hỗ trợ, nên
  một vòng lặp ELL bắt đầu được căn lề khi bước nhảy là bội của 8 float (ch17-facts A3, phép tính
  của khóa học này). Một vòng lặp ELL của một warp 32 luồng đọc 128 byte liên tiếp của mỗi mảng.
- **HIP**: cùng mã nguồn. `<cuda/atomic>` đến từ libhipcxx, ROCm 10.1 trở lên (infra-facts B2). AMD
  ghi trong tài liệu các dòng 64 và 128 byte, nên căn lề cần một bước nhảy là bội của 16 hoặc 32
  float; một wavefront 64 lane đọc 256 byte liên tiếp mỗi vòng lặp ELL (ch17-facts B1). Chưa chạy
  trên phần cứng AMD ở đây: unverified.
- **Metal**: `[[thread_position_in_grid]]` là hàng, và `dispatchThreads` tạo đúng `numRows` luồng
  (ch05-facts C4, C5). Kernel COO khai báo `y` là `device atomic_float*` và cộng bằng
  `atomic_fetch_add_explicit(..., memory_order_relaxed)`: Metal 3 và Apple7, bộ nhớ device
  (ch17-facts C1). Kernel ELL của bạn bind cùng buffer đó dưới dạng `device float*`. Các dispatch
  COO và ELL dùng chung một compute pass, chạy chúng theo thứ tự (ch02-facts C9). Apple không ghi
  trong tài liệu đơn vị coalescing, quy tắc căn lề hay mô hình phân kỳ nào (ch17-facts C3): trên
  Metal, benchmark là một phép đo không có gì để đối chiếu.

## Trước khi đo hiệu năng, hãy dự đoán

Benchmark dựng bốn ma trận có cùng số hàng, trung bình khoảng 16 phần tử mỗi hàng, các cột nằm
trong khoảng ±2048 quanh đường chéo:

- `uniform`: mọi hàng dài 16.
- `spread`: độ dài phân bố đều từ 1 đến 31, theo thứ tự ngẫu nhiên.
- `spread_sorted`: cùng các hàng đó, hàng dài nhất trước (phép sắp xếp JDS của mục 17.6). Mỗi hàng
  giữ nguyên các cột của nó, nên các hàng cạnh nhau không còn đọc các phần cạnh nhau của `x`.
- `skewed`: các hàng dài 16, và cứ 512 hàng có một hàng dài 256.

Nó đo `csr`, `ell` (bước nhảy = `numRows`), `ell_aligned` (bước nhảy làm tròn lên bội của 32 float)
và, trên `spread` và `skewed`, `hybrid` (K = 16, độ dài điển hình, COO và ELL được đo cùng nhau).
Hãy viết những điều sau ra trước. Bài học hỏi hai điều đầu dưới các id trong ngoặc.

1. Trên `uniform`, thời gian CSR chia cho thời gian ELL (`ch17/labs/p-lab1-uniform`).
2. Trên `skewed`, thời gian ELL chia cho thời gian CSR, và liệu bản lai có quay về được thời gian
   của CSR hoặc tốt hơn không (`ch17/labs/p-lab1-skewed`). Benchmark in CSR / ELL; trên `skewed`
   nó in thêm ELL / CSR.
3. Không được ghi lại: CSR có chạy nhanh hơn trên `spread_sorted` so với trên `spread` không?
   `ell_aligned` có thắng `ell` không?

Các mô hình của bài học đếm số yêu cầu bộ nhớ và số vòng lặp phân kỳ; chúng bỏ qua bộ nhớ đệm
(cache). Apple hoàn toàn không ghi trong tài liệu mô hình coalescing hay phân kỳ nào
(ch17-facts C3). Không điều gì bạn đã đọc hứa trước câu trả lời nào: đó là lý do bạn đo.

## Đọc kết quả in ra

- Kích thước được chọn theo thiết bị của bạn: 2^16 + 5 hàng, nhân đôi khi CSR trên `uniform` còn
  mất dưới 2 ms, tối đa 2^21 + 5, và chỉ khi ma trận `skewed` ở dạng ELL vẫn còn vừa một phần tư
  bộ nhớ mà thiết bị báo. Phần `+ 5` cố ý làm cho `stride = numRows` không phải bội của 8. Một định
  dạng không vừa sẽ được in là bị bỏ qua, kèm số byte nó cần.
- Kết quả của mọi định dạng được kiểm tra với tham chiếu CPU trước khi được đo; một kết quả sai được
  báo là sai, không kèm thời gian. Một thời gian là lần tốt nhất trong `--reps` lần chạy (mặc định
  5) sau một lần khởi động. Trên Metal, đó là thời gian từ đầu đến cuối của GPU cho một command
  buffer; trên CUDA và HIP, thời gian thực trên host quanh các lần khởi chạy và một lần đồng bộ hóa.
- `width`, `stride`: hình dạng của phần ELL. `pad`: tỉ lệ các ô ELL được lưu mà không chứa phần tử
  nào (gồm cả các hàng căn lề). `COO`: các phần tử bị đẩy sang của bản lai.
- `matrix MB`: lượng mà định dạng lưu (`matrix_bytes`). Một kernel dừng ở `nnzPerRow[row]` không đọc
  các ô đệm, nên đây là lượng một vòng lặp toàn độ rộng sẽ đọc, không phải lượng kernel của bạn đã
  đọc.
- `GB/s`: số byte hữu ích chia cho thời gian. Byte hữu ích là 8 cho mỗi phần tử, cộng 16 cho mỗi
  hàng (một con trỏ hoặc độ dài, `x` một lần, `y` đọc và ghi): lưu lượng ít nhất mà bất kỳ định
  dạng nào có thể tạo ra, cùng một tử số cho mọi định dạng. Nó không phải lượng mà hệ thống bộ nhớ
  đã di chuyển.
- `pmpp.py` cũng lưu mỗi hàng (`name: spmv`, với `matrix`, `format`, `rows`, `nnz`,
  `width`, `stride`, `k` cho bản lai, `coo_entries`, `seconds`, `matrix_bytes`,
  `pad_fraction`, `gbps`) vào `reports/runs/`, nơi ứng dụng đọc chúng.

Các con số là phép đo trên máy của bạn ở kích thước đó. Chúng không phải thuộc tính của các định
dạng.

## Đi xa hơn

- Exercise 5 (p. 422): viết kernel JDS. Ma trận `spread_sorted` đã ở sẵn thứ tự hàng JDS; phần còn
  lại là bố cục theo thứ tự cột không đệm và mảng `iterPtr`.
- Exercise 4 (p. 422) yêu cầu phần COO của bản lai chạy trên host. Hãy đo nó so với kernel được
  cung cấp ở K = 16 trên `skewed`, tính cả các lần sao chép.
- Thay vòng lặp ELL của bạn bằng vòng lặp đúng còn lại (toàn độ rộng có kiểm tra cột, hoặc
  `nnzPerRow`) và đo cả hai trên `spread` và `skewed`. Vòng lặp nào đọc các ô đệm?
