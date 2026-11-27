> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 21.1: tính tổng Coulomb trực tiếp, scatter, gather và gộp việc

Sách: mục 21.2 đến 21.4 (pp. 515-522), Figs. 21.5, 21.6, 21.8 và 21.10, và
Exercise 1 (p. 528).

Một lát của bản đồ thế tĩnh điện, được tính theo ba cách. Mỗi điểm của lát nhận tổng, trên mọi
nguyên tử, của điện tích / khoảng cách. Các nguyên tử đến device theo từng phần đủ nhỏ cho bộ nhớ
hằng (constant memory), và mỗi lần khởi chạy cộng thêm phần đóng góp của một phần (p. 517). Vật lý
không quan trọng ở đây. Điều quan trọng là thứ tự các quyết định mà chương đưa ra: scatter (mỗi đầu
vào một luồng, atomic trên mọi đầu ra), gather (mỗi đầu ra một luồng, không atomic), gộp việc cho
luồng (thread coarsening) (mỗi bốn đầu ra một luồng, toán hạng được tái sử dụng từ thanh ghi), và
bốn đầu ra nào mà một luồng đã gộp việc sở hữu, điều quyết định các lần ghi của nó có được hợp nhất
(coalesced) hay không.

## Phần của bạn

Ba kernel và các hàm khởi chạy của chúng, trong `work/kernel.cu`:

1. Mảng `__constant__` cho một phần: `CHUNK_MAX` nguyên tử, mỗi nguyên tử bốn float (x, y, z,
   điện tích).
2. `dcsScatterKernel` và `launchDcsScatter`: Fig. 21.5. Mỗi nguyên tử của phần một luồng, trong các
   khối 1D gồm `BLOCK_X` luồng. Mỗi luồng duyệt toàn bộ lát và cộng nguyên tử của nó vào mọi điểm
   bằng một `atomicAdd` kiểu float. Listing không có số lượng nguyên tử; kernel của bạn cần có, và
   phải thoát nếu nguyên tử của nó vượt quá cuối phần.
3. `dcsGatherKernel` và `launchDcsGather`: Fig. 21.6. Mỗi điểm một luồng, trong các khối
   `BLOCK_X x BLOCK_Y`. Nó cộng phần đó trong một thanh ghi và cộng tổng vào điểm của mình bằng một
   `+=`. Hãy kiểm tra `i` và `j`: lát không được đệm thành nguyên khối.
4. `dcsCoarsenedKernel` và `launchDcsCoarsened`: Fig. 21.10. Mỗi luồng sở hữu `COARSEN`
   (bốn) điểm của một hàng, cách nhau `blockDim.x`: luồng `t` của một khối lấy các điểm
   `t`, `t + blockDim.x`, `t + 2 blockDim.x` và `t + 3 blockDim.x` trong dải `4 blockDim.x` điểm của
   khối. Đọc mỗi nguyên tử một lần, dùng chung `dy*dy + dz*dz` và điện tích cho cả bốn điểm, và
   kiểm tra mỗi điểm với cuối hàng trước khi ghi nó.

Mỗi hàm khởi chạy là vòng lặp host của p. 517, chính là Exercise 1: chép tối đa `chunk` nguyên tử
vào mảng hằng bằng `cudaMemcpyToSymbol`, khởi chạy, lặp lại cho đến khi mọi nguyên tử đã được cộng.
Phần cuối thường không đầy. `dcs.h` cho hợp đồng:

- lưới là `nz x ny x nx` float, thứ tự hàng (row-major), x thay đổi nhanh nhất, dưới dạng một con
  trỏ device;
- các nguyên tử là `numatoms x 4` float trong bộ nhớ **host**;
- lần khởi chạy **cộng** vào mọi điểm của lát `k` và để yên các lát khác;
- `1 <= chunk <= CHUNK_MAX`.

Cùng ba kernel đó nằm trong `work/kernel.metal` (`dcs_scatter`, `dcs_gather`,
`dcs_coarsened`) và các vòng lặp theo phần của chúng trong `work/launch.mm` (`encodeDcsScatter`,
`encodeDcsGather`, `encodeDcsCoarsened`).

## Các tệp

- Sửa `work/kernel.cu` (CUDA, HIP và cả hai trình giả lập), `work/kernel.metal` và
  `work/launch.mm`.
- `dcs.h`: các kích thước và prototype của các hàm khởi chạy. `CHUNK_MAX` là 1,024 nguyên tử, 16 KB
  bộ nhớ hằng mỗi tệp; `BLOCK_X` là 64 và `BLOCK_Y` là 4; `COARSEN` là 4. Đừng viết cứng các giá trị
  này.
- Được cung cấp, không sửa: `../shared/coarse_adjacent.cu`, `coarse_adjacent.metal` và
  `coarse_adjacent_launch.mm`. Đó là Fig. 21.8 với dòng 05 đã sửa (xem bên dưới): mỗi luồng sở hữu
  bốn điểm **liền kề**. Đó là mốc so sánh mà kernel có bước nhảy (strided) của bạn được đo với, và
  nó có mảng hằng riêng, `atoms_adj`, nên hai bên không thể xung đột. Nó cũng được kiểm thử.
- `main.cpp`: các kiểm thử, tham chiếu CPU và benchmark. Bạn không sửa nó.

## Lệnh

- Kiểm thử: `python3 labs/pmpp.py run 21.1 --backend emu` trước, rồi `--backend race`, rồi
  `python3 labs/pmpp.py run 21.1` cho GPU của bạn.
- Đo: `python3 labs/pmpp.py run 21.1 --bench` (chỉ thiết bị thật).
- Bắt đầu lại: `python3 labs/pmpp.py reset 21.1 --yes`.

## Trên Metal

- Không có symbol nào để chép vào. Tất cả nguyên tử nằm trong một buffer gồm các `float4`, 16 byte
  mỗi nguyên tử, và một phần là chính buffer đó được bind lại ở nguyên tử đầu tiên của phần:
  `setBuffer:atoms offset:16 * start atIndex:1`. Kernel nhận nó dưới dạng
  `constant float4* atoms`. Apple không ghi trong tài liệu bộ nhớ đệm hằng nào phía sau không gian
  địa chỉ đó, cũng không ghi giới hạn kích thước nào cho nó (`notes/today/ch21-facts.md` C1).
- Các hàm atomic của Metal chỉ nhận kiểu atomic. Kernel scatter khai báo lưới là
  `device atomic_float*` và cộng bằng
  `atomic_fetch_add_explicit(&grid[p], v, memory_order_relaxed)`; các kernel gather và gộp việc khai
  báo cùng buffer đó là `device float*`. Atomic float cần GPU Apple7 (M1) trở lên (C2).
- Các dispatch của một compute encoder thông thường chạy theo thứ tự chúng được encode và thấy các
  lần ghi của nhau (`notes/today/ch02-facts.md` C9), nên vòng lặp theo phần không cần rào chắn
  (barrier).
- Dispatch kernel gộp việc bằng `dispatchThreadgroups:threadsPerThreadgroup:`, nguyên các
  threadgroup `BLOCK_X x BLOCK_Y`, `ceil(nx / (4 x BLOCK_X))` threadgroup theo chiều ngang.
  `dispatchThreads` cắt bớt các threadgroup ở rìa lưới (ch05-facts C5), và các luồng bị thiếu của
  một threadgroup bị cắt sở hữu những điểm mà không ai tính. Hãy đọc bước nhảy từ
  `[[dispatch_threads_per_threadgroup]]`, kích thước bạn đã dispatch. Scatter và gather có thể dùng
  `dispatchThreads` với đúng mỗi nguyên tử hoặc mỗi điểm một luồng.

## Trên HIP

Cùng `kernel.cu` được viết để dựng cho HIP mà không cần đổi (unverified: chưa máy HIP nào dựng nó),
vì nó chỉ dùng những gì `pmpp/vendors/hip.h` mang sang nguyên vẹn: luồng và khối, bộ nhớ hằng, một
atomic kiểu cũ và các lời gọi runtime; `cudaMemcpyToSymbol` được ánh xạ thành `hipMemcpyToSymbol`.
(Các intrinsic warp `_sync`, cooperative groups và các header half-precision sẽ không chuyển nguyên
vẹn được.) `atomicAdd` float trên AMD biên dịch thành một phép toán an toàn, có thể là một vòng lặp
compare-and-swap, trừ khi bản dựng truyền `-munsafe-fp-atomics`, và RDNA1 và CDNA1 không có phép
cộng nguyên tử float bằng phần cứng (`notes/today/infra-facts.md` B1, ch21-facts B1). Bản dựng HIP
của khóa học không truyền cờ đó, nên phép đo scatter của bạn trên AMD là của dạng an toàn. AMD không
ghi trong tài liệu kích thước nào cho bộ nhớ hằng; bộ nhớ đệm hằng của nó (scalar cache) chỉ giúp
khi mọi lane của một wavefront đọc cùng một địa chỉ, điều mà các kernel gather làm còn kernel
scatter thì không.

## Kiểm thử làm gì

Mỗi trường hợp chạy một lát đầy đủ qua hàm khởi chạy của bạn và so sánh mỗi điểm của lát `k` với
một tổng tính bằng double trong `main.cpp`. Dung sai tại mỗi điểm là
`1e-4 x (sum over atoms of |charge| / distance)`: nó tính đến fast math của Metal
(ch21-facts C3) và việc các atomic float cộng theo thứ tự không cố định, nên tổng của scatter có
thể khác nhau giữa các lần chạy ở những bit cuối (A6). Không bao giờ so từng bit.

- Các lát, `nx x ny`: 37 x 11 và 300 x 5. Không cái nào là bội của `4 x BLOCK_X`, nên khối cuối của
  mọi hàng thò ra ngoài; 300 là một dải nguyên 256 điểm cộng thêm 44.
- 150 nguyên tử theo các phần 64: ba lần khởi chạy, gồm 64, 64 và 22 nguyên tử. Cũng có cả 150 trong
  một lần khởi chạy với phần `CHUNK_MAX`, và một nguyên tử duy nhất.
- Lát 1 trong 3 và lát 2 trong 3: mọi lát khác chứa một giá trị canh phải còn nguyên từng bit. Lát
  2 là lát cuối, nên một lần ghi vượt quá cuối một hàng ở đó sẽ ra khỏi vùng cấp phát.
- Điện tích phân bố đều trong [-1, 1], cả hai dấu. Tọa độ z của mọi nguyên tử nằm giữa hai mặt
  phẳng lát, nên không khoảng cách nào bằng không. Khoảng cách lưới 0.5, và 0.25 trong một trường
  hợp.
- Trên GPU còn có 1000 x 333 với 2,000 nguyên tử theo các phần 1,024.
- Cả bốn kernel (ba kernel của bạn và kernel được cung cấp) trên mọi trường hợp, và một phép kiểm
  tra rằng gather và gộp việc khớp với nhau trong cùng dung sai.

Lưới được cấp phát đúng `nz x ny x nx` float, nên trên `emu` một lần ghi vượt quá cuối của nó sẽ
dừng lần chạy với một báo cáo AddressSanitizer nêu dòng đó.

## Mỗi backend bắt được gì

- `emu` chạy từng luồng một và không bao giờ ngắt một luồng giữa chừng một `+=`, nên nó **không
  thể thấy một atomic bị thiếu**: scatter với một `+=` thường vẫn qua ở đó. Nó bắt được chỉ số sai,
  thiếu cận (AddressSanitizer), nguyên tử cũ do thiếu kiểm tra số lượng nguyên tử, `=` ở chỗ phải là
  `+=` (các trường hợp ba lần khởi chạy), và thiếu độ lệch lát (các trường hợp có giá trị canh).
- `race` chạy các luồng thực sự đồng thời dưới ThreadSanitizer. Scatter với một `+=` thường bị báo
  ở đó kèm dòng mã nguồn, dù tổng có ra sai hay không, và một kernel gộp việc có các luồng sở hữu
  những điểm chồng lấn cũng vậy. Nó không có kiểm tra biên.
- GPU chỉ cho thấy một atomic bị thiếu dưới dạng tổng sai, và chỉ khi hai luồng tình cờ va chạm về
  thời gian.
- Các trường hợp rộng 37 không thể thấy một `dx1`, `dx2` hay `dx3` sai trong kernel gộp việc: một
  hàng 37 điểm chỉ chứa điểm đầu tiên của mỗi luồng. Các trường hợp rộng 300 thì thấy được.

## Trước khi đo hiệu năng, hãy dự đoán

Hãy viết những điều sau ra trước; trang labs ghi lại chúng.

1. **Scatter so với gather** (`p-scatter`): gather sẽ chạy nhanh hơn scatter bao nhiêu lần trong
   so sánh A trên GPU của bạn? Sách nói atomic làm chậm kernel scatter "significantly" và không cho
   con số nào (p. 518); cũng không nhà cung cấp nào cho (ch21-facts A6). Hãy nghĩ xem mọi luồng của
   một phần đang làm gì vào cùng một thời điểm.
2. **Có bước nhảy so với liền kề** (`p-strided`): kernel có bước nhảy của bạn sẽ chạy nhanh hơn
   kernel liền kề được cung cấp bao nhiêu lần ở phần 64, và khoảng cách đó sẽ tăng hay giảm khi đến
   phần 1,024? Trên NVIDIA, quy tắc được ghi trong tài liệu đếm số sector 32 byte mỗi warp: bốn câu
   lệnh ghi của kernel liền kề chạm 64 sector, của kernel có bước nhảy là 16 (ch21-facts A8, E2).
   Nhưng mỗi lần khởi chạy đọc và ghi mỗi điểm một lần, trong khi vòng lặp nguyên tử chạy `chunk`
   lần mỗi điểm (E3). Dự đoán của chính chúng tôi, unverified, là khoảng cách thu hẹp khi phần lớn
   lên. AMD ghi trong tài liệu việc gom các địa chỉ của một wavefront thành các dòng 64 hoặc 128
   byte (B3, E2), điều này dự đoán được số dòng nhưng không dự đoán được thời gian; ở đó wavefront
   có thể rộng 64 lane và các atomic float phụ thuộc vào một cờ trình biên dịch (B1). Apple không
   ghi trong tài liệu quy tắc coalescing nào, cũng không ghi chi phí nào cho các atomic của nó (C2, C4).

## Benchmark

Chỉ thiết bị thật. Mỗi thời gian là cả một lát, tính mọi phần, lần tốt nhất trong `--reps` lần
chạy (mặc định 5), với các kernel lần lượt thay phiên. Trên CUDA và HIP, nó gồm cả các lần sao chép
`cudaMemcpyToSymbol`; trên Metal, các nguyên tử đã nằm sẵn trong một buffer và một phần chỉ là một
lần bind lại.

- **A**, scatter so với gather: một lát 256 x 256, 1,024 nguyên tử, phần 256. Giữ nhỏ vì scatter chỉ
  có một luồng cho mỗi nguyên tử của một phần và mỗi luồng đó duyệt toàn bộ lát.
- **B**, gather, `coarse_adjacent` và `coarse_strided` (của bạn): một lát 1024 x 1024, 4,096 nguyên
  tử, ở phần 64, 256, 512 và 1,024. 512 nguyên tử là 8 KB, tập làm việc của bộ nhớ đệm hằng mỗi SM
  mà NVIDIA ghi trong tài liệu (A5); việc có gì thay đổi ở đó hay không là unverified, và đây là nơi
  bạn sẽ thấy điều đó.

Mỗi hàng báo kernel (`scatter`, `gather`, `coarse_adjacent`, `coarse_strided`), `nx`, `ny`, số
nguyên tử, phần, số giây mỗi lát, số cặp nguyên tử-điểm mỗi giây, và độ rộng nhóm: `warpSize` trên
CUDA và HIP, `threadExecutionWidth` của pipeline `dcs_coarsened` trên Metal. Các trình khám phá ở
trang 21.2 và 21.5 của bài học cung cấp độ rộng đã ghi của Metal dưới mục "Apple". Các con số là
phép đo trên máy của bạn, không phải thuộc tính của các kernel.

## Tùy chọn, trên CUDA: xem các sector

Các bảng bộ nhớ của Nsight Compute báo số sector mỗi yêu cầu; với các truy cập 32-bit bởi một warp
đầy đủ, 4 là tối ưu (ch21-facts A8). Hãy profile `dcsCoarsenedKernel` và `dcsCoarseAdjacentKernel`
và so sánh các lần ghi toàn cục của chúng. (Không chạy ở đây: máy đã viết lab này không có thiết bị
CUDA.)

## Lab này khác các trang sách ở đâu

- **Fig. 21.8, dòng 05.** Như bản in, `i = blockIdx.x * blockDim.x*COARSEN_FACTOR +
  threadIdx.x` cộng một `threadIdx.x` chưa nhân hệ số trong khi mỗi luồng ghi `i` đến `i+3`, nên các
  luồng cạnh nhau sở hữu những điểm chồng lấn và `3 x blockDim.x - 3` điểm cuối trong dải của mỗi
  khối không được ai ghi (ghi chú, "Unclear in the PDF"; ch21-facts E2). Mốc so sánh được cung cấp
  nhân `threadIdx.x` với hệ số, như phần chữ ở p. 521 mô tả. Cùng dòng 05 đó là đúng trong
  Fig. 21.10, nơi quyền sở hữu là có bước nhảy.
- Lát là một `int k`, và tọa độ z của nó là `k x gridspacing`. Fig. 21.6 truyền một `z` kiểu float
  và tính `k = z / gridspacing`, phép này cắt bỏ phần thập phân; Figs. 21.3 và 21.4 đánh chỉ số lát
  theo hai cách khác.
- Các kích thước là `nx` và `ny`, ở chỗ sách truyền một `dim3 grid`.
- Không kernel nào của sách kiểm tra biên, và Fig. 21.5 hoàn toàn không có số lượng nguyên tử.
  Fig. 21.9 vẽ một lưới được đệm thành nguyên khối, điều mà phần chữ không bao giờ bàn tới; lưới này
  không được đệm, nên mọi kernel ở đây đều kiểm tra biên.
- Chương không nêu tên lời gọi điền bộ nhớ hằng (ghi chú, Exercise 1); đó là
  `cudaMemcpyToSymbol`.
- `atomicAdd` trên `float` của sách là CUDA hiện hành. NVIDIA ngày nay gọi nó là atomic kiểu cũ
  (legacy) và khuyên dùng `cuda::atomic_ref`, thứ mà chương 9 đã dùng (infra-facts A1, A2).

## Đi xa hơn

PyTorch đưa ra lựa chọn của chương này cho gradient của một bảng embedding, và đưa ra khác nhau
trên hai backend. Trên CUDA (và ROCm), nó sắp xếp các token id, việc này chia các hàng thành ngăn,
rồi cho mỗi luồng một đầu ra và ghi một lần: gather, tất định. Nó chỉ dùng một đường đi atomic gộp
(fused) khi một heuristic nói gather sẽ có độ song song kém và các thuật toán tất định không được yêu
cầu. Trên MPS, kernel Metal theo mẫu của Fig. 21.5, mỗi đầu vào một luồng và một phép cộng nguyên tử
vào gradient, và PyTorch báo lỗi cho nó ở chế độ tất định (ch21-facts D1, D2). Từ những gì bạn đã đo
trong so sánh A và những gì A6 nói về thứ tự của các atomic float, bạn sẽ phát hành cách nào trong
hai cách, và khi nào?
