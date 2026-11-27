> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 10.1: một tổng, ba kernel

Sách: các mục 10.6 đến 10.10 (pp. 236-248), Figs. 10.10, 10.13, 10.16, 10.18 và 10.20, và bài tập
5 (p. 250).

Bạn viết ba phép rút gọn (reduction) tính tổng và đo thời gian chúng so với hai kernel đầu tiên của
sách, vốn được cung cấp sẵn. Phép rút gọn khối hai giai đoạn bạn viết ở (b) là khối xây dựng mà lab
10.3 dùng lại cho softmax và argmax của mốc dự án M4.

**Phần việc của bạn:**

- **(a) `SharedMemorySumReductionKernel`** (Fig. 10.10). Khối b rút gọn hàng b gồm `2*blockDim.x`
  số float vào `out[b]`: mỗi luồng (thread) cộng hai phần tử của nó ngay khi nạp chúng, rồi một cây
  trong bộ nhớ chia sẻ (shared memory) hoàn tất hàng đó. Fig. 10.10 ghi `*output` nhưng không có
  tham số `output`; hãy thêm nó lại.
- **(b) `warp_reduce` và `TwoStageWarpLevelSumReductionKernel`** (Figs. 10.13 và 10.16), được viết
  cho khả chuyển. Cùng các hàng như (a). Mỗi warp rút gọn các giá trị của nó bằng shuffle, lane 0
  của mỗi warp chuyển tổng của nó qua bộ nhớ chia sẻ, một rào chắn (barrier), và warp 0 rút gọn các
  tổng đó.
  - Mặt nạ lane có kiểu riêng: `unsigned int` trên CUDA, 64 bit trên HIP. Không bao giờ dùng hằng
    `0xffffffff`.
  - Vòng lặp shuffle bắt đầu từ `warpSize/2`, và lane cùng warp được tính từ `threadIdx.x` và
    `warpSize`, không từ một hằng số 32.
  - Mảng cho từng warp có kích thước theo warp hẹp nhất được công bố: `1024/32` phần tử.
  - Lần đọc ở giai đoạn 2 có kiểm tra điều kiện. Các lane không có warp nào phía sau dùng 0.
- **(c) `CoarsenedSumReductionKernel`** (Figs. 10.18 và 10.20, được hoàn thiện, cộng với bài tập 5).
  Mọi n. Hệ số gộp việc là một đối số lúc chạy, và mỗi luồng trước tiên cộng tuần tự `2*coarse`
  phần tử của nó, cách nhau `blockDim.x`. Một phần tử quá cuối mảng đóng góp 0. Tiếp theo là cùng
  phép rút gọn khối hai giai đoạn, rồi mỗi khối một lần
  `cuda::atomic_ref<float, cuda::thread_scope_device>` `fetch_add(..., cuda::memory_order_relaxed)`,
  cách viết của sách (dưới HIP là `atomicAdd` kiểu cũ: xem bên dưới). Hàm khởi chạy đặt `*out`
  bằng 0 (`cudaMemset`) và khởi chạy `ceil(n / (2*coarse*block))` khối.

Mọi hàm khởi chạy đều nhận kích thước khối, một lũy thừa của hai từ 64 đến 1024, để các bài kiểm
thử có thể thử nhiều giá trị.

## Tệp

- Sửa `work/kernel.cu` (CUDA, HIP và cả hai trình giả lập): ba kernel, `warp_reduce`, và
  `launchSharedMemorySum`, `launchTwoStageWarpSum` và `launchCoarsenedSum`.
- Sửa `work/kernel.metal` (`shared_memory_sum`, `warp_reduce`, `two_stage_warp_sum`,
  `coarsened_sum`) và `work/launch.mm` (`encodeSharedMemorySum`, `encodeTwoStageWarpSum`,
  `encodeCoarsenedSum`). Mỗi hàm encode trả về false, không dispatch, cho một khối mà pipeline của
  nó không chạy được: nhiều luồng hơn `maxTotalThreadsPerThreadgroup` của pipeline, hoặc, với hai
  kernel SIMD-group, một khối không phải là một số nguyên lần SIMD-group gồm `threadExecutionWidth`
  lane hoặc có nhiều SIMD-group hơn số lane của một SIMD-group.
- Được cung cấp, không sửa: `SimpleSumReductionKernel` (Fig. 10.5) và
  `ConvergentSumReductionKernel` (Fig. 10.8) của sách trong `../shared/inplace_sum.cu`,
  `../shared/inplace_sum.metal` và `../shared/inplace_launch.mm`. Mỗi khối rút gọn một hàng, tại
  chỗ.

## Lệnh

- Kiểm thử: `python3 labs/pmpp.py run 10.1 --backend emu` trước, rồi `python3 labs/pmpp.py run 10.1`.
- Phép toán nguyên tử (atomic): `python3 labs/pmpp.py run 10.1 --backend race`.
- Đo: `python3 labs/pmpp.py run 10.1 --bench` (chỉ trên thiết bị thật).
- Lời giải tham chiếu (solution): thêm `--solution`.

## Các bài kiểm thử làm gì

- (a), (b) và cả hai kernel được cung cấp: 1, 3 và 37 hàng với khối 64, 128, 256 và 1024 luồng
  (trình giả lập: 1 và 3 hàng ở 64 và 256; `race`: 2 hàng ở 64; Metal: các khối mà pipeline cho
  phép, và chương trình điều khiển in ra những khối nó bỏ qua). Các kernel được cung cấp cũng được
  kiểm thử, nên một tệp được cung cấp bị hỏng sẽ lộ ra.
- (c): n = 1, 2, 31, 32, 33, 255, 256, 257, 1000, 4097, 100003 và 2^22 (hai giá trị cuối chỉ trên
  thiết bị thật), ở coarse 1, 2, 3 và 8, với khối 64 và 256. Trên `race`: n = 33, 1000 và 4097 với
  khối 64, để phần lớn các trường hợp có ít nhất 4 khối chạy cùng lúc.
- Ba loại dữ liệu. Toàn số một và các số nguyên từ -3 đến 3 có tổng float chính xác bất kể thứ tự
  các phép cộng, nên chúng phải khớp tuyệt đối. Các giá trị phân bố đều trong [-1, 1] phải nằm
  trong 1e-6 lần tổng của |x|, cộng 1e-6. Phần tuyệt đối có mặt vì các phép atomic float của CUDA
  trên bộ nhớ toàn cục đưa các số dưới chuẩn (denormal) về 0 (ch10 facts A6).
- Các đầu ra bắt đầu là NaN, nên một hàng kernel của bạn không bao giờ ghi sẽ bị báo. Mỗi trường
  hợp của (c) được khởi chạy hai lần trên cùng đầu ra, và hàm khởi chạy phải đặt lại `*out` về 0 mỗi
  lần.

Mỗi backend bắt được gì:

- `emu` điền rác vào bộ nhớ chia sẻ cho mỗi khối và chạy các luồng lần lượt từng luồng theo ba thứ
  tự cố định. Một rào chắn đặt sai chỗ, một lần đọc ở giai đoạn 2 không kiểm tra điều kiện, hay một
  luồng trả về trước các lần shuffle sẽ thất bại ở đó mọi lần chạy, dù không phải lúc nào cũng dưới
  cả ba thứ tự.
- `race` chạy các khối đồng thời dưới ThreadSanitizer. Một lệnh `*output += partialSum` thường thay
  cho phép atomic sẽ bị báo cùng dòng mã nguồn. `emu` không thấy được một phép atomic bị thiếu.
- Metal là một GPU thật. Trên máy dùng để xây dựng lab này, một lần đọc ở giai đoạn 2 không kiểm tra
  điều kiện vẫn đạt ở đó, vì bộ nhớ threadgroup chưa được ghi tình cờ chứa toàn số 0. Hãy bắt đầu
  với `emu`.
- Không gì trên máy Mac bắt được hai lỗi của HIP: mặt nạ `0xffffffff` và vòng lặp bắt đầu từ 16. Cả
  hai trình giả lập đều mô hình hóa warp 32 lane, và Metal không có mặt nạ. HIP từ chối mặt nạ 32
  bit lúc biên dịch, vì các hàm `_sync` của nó `static_assert` một mặt nạ 64 bit. Trên một wavefront
  64 lane, vòng lặp bắt đầu từ 16 tính tổng riêng từng nửa của wavefront (ch10 facts B1; infra facts
  B3). Hãy kiểm tra hai điều đó bằng cách đọc mã của bạn.

## Các backend khác nhau ở đâu (không có trong sách)

- **HIP**: kiểu mặt nạ ở trên. `warpSize` là 64 trên gfx9 (Instinct) và 32 trên các dòng AMD sau
  này. Khi đó một khối 1024 luồng có 16 warp, và đó là lý do lần đọc ở giai đoạn 2 cần kiểm tra điều
  kiện (ch10 facts B2). `<cuda/atomic>` đến từ libhipcxx trên HIP, thư viện mà bản dựng HIP của khóa
  học chỉ liên kết ở nơi ROCm cung cấp nó (infra facts B2), nên tệp chỉ include nó khi `__HIPCC__`
  không được định nghĩa, và dưới HIP lần cộng dồn là `atomicAdd(output, value)` kiểu cũ, cũng là
  relaxed và có phạm vi toàn device (infra facts A1, B1). Không gì trong số này đã được biên dịch hay
  chạy với ROCm ở đây: unverified.
- **Metal**: không có mặt nạ; `simd_shuffle_down(v, delta)`. Metal không công bố cách một
  threadgroup được chia thành các SIMD-group, nên chỉ số SIMD-group và lane lấy từ
  `[[simdgroup_index_in_threadgroup]]` và `[[thread_index_in_simdgroup]]`, không từ `t / width`
  (ch04 facts C2). Độ rộng cũng không được công bố: hãy đọc `[[threads_per_simdgroup]]`. Atomic
  float cần `device atomic_float*` và `atomic_fetch_add_explicit` với `memory_order_relaxed`: Metal
  3, Apple7 trở lên (infra facts C1). Metal không có `cudaMemset` bên trong một compute pass. Buffer
  nằm trong shared storage và command buffer của nó chưa được commit, nên `launch.mm` ghi số 0 qua
  `out.contents` (ch05 facts C6).
- Các kernel Metal được cung cấp truyền các tổng riêng phần qua bộ nhớ device, nên rào chắn của
  chúng là `threadgroup_barrier(mem_flags::mem_device)` (ch05 facts C2).

## Trước khi đo hiệu năng, hãy dự đoán

Viết những điều này ra trước. Bài học hỏi chúng dưới các id trong ngoặc.

1. Kernel hai giai đoạn sẽ chạy nhanh hơn kernel đơn giản bao nhiêu lần cho mỗi hàng
   (`ch10/labs/p-lab1-warp`).
2. Hệ số gộp việc nào, từ 1 đến 128, sẽ nhanh nhất trên một mảng 2^26 số float trở lên (chương
   trình điều khiển nhân đôi n trên thiết bị nhanh, tối đa 2^28, và chỉ bắt đầu thấp hơn ở nơi thiếu
   bộ nhớ; xem "Đọc đầu ra") (`ch10/10.10/p-coarse`).
3. 20 lần chạy phiên bản atomic trên cùng đầu vào sẽ cho bao nhiêu kết quả khác nhau
   (`ch10/10.9/p-distinct`).

Phân tích của sách cho 256 phần tử nói hiệu suất dùng tài nguyên thực thi là 30% so với 66% (đơn
giản so với hội tụ, pp. 231 và 234), và số yêu cầu bộ nhớ toàn cục là 141 so với 36 so với 9 (đơn
giản, hội tụ, bộ nhớ chia sẻ, pp. 235 và 237). Thiết bị của bạn quyết định bao nhiêu phần trong đó
biến thành thời gian.

## Đọc đầu ra

Benchmark in ba bảng. Các hàng đó cũng được đưa vào các trang bài học (các mục 10.8, 10.9, 10.10 và
trang labs).

- **rows**: 2^24 số float dưới dạng các hàng 512 (khối 256), cho bốn kernel theo hàng. Mỗi lần chạy
  được đo có bản sao đầu vào riêng vừa được tải lên, vì các kernel được cung cấp ghi đè đầu vào của
  chúng. Một phần tư giây công việc không đo thời gian chạy trước, để kernel đầu tiên được đo không
  phải trả thêm chi phí đánh thức thiết bị. `time ms` là kết quả tốt nhất trong năm lần chạy.
  `GB/s` là số byte của đầu vào (4 mỗi số float) chia cho thời gian đó: số byte mà một phép tính
  tổng phải đọc, không phải những gì hệ thống bộ nhớ đã chuyển.
- **any length**: kernel gộp việc ở coarse 1 đến 128 trên một mảng. n bắt đầu ở 2^26 và nhân đôi
  khi coarse 1 còn chạy dưới nửa mili giây, tối đa 2^28 và một phần tư bộ nhớ mà thiết bị của bạn
  báo cáo. `blocks` là số khối đã được khởi chạy. Trên CUDA và HIP, thời gian bao gồm lệnh
  `cudaMemset` một số float của hàm khởi chạy. Trên Metal thì không, vì số 0 được ghi từ CPU.
- **determinism**: 20 lần chạy kernel gộp việc trên cùng 2^24 số float ở coarse 1, và số mẫu bit
  khác nhau trong 20 kết quả.

Các con số là phép đo của máy này ở các kích thước này. Chúng không phải là tính chất của các kernel.

## Đi xa hơn

Bảng determinism là sự khác biệt giữa hai thiết kế nhiều khối của sách. Một lần cộng dồn bằng
atomic cộng các tổng của khối theo bất kỳ thứ tự nào mà các khối hoàn thành. Các tổng riêng phần
được ghi theo từng khối và được rút gọn bởi một kernel thứ hai thì giữ một thứ tự (pp. 245-246). CUB
gọi hai cách này là `not_guaranteed` và `run_to_run`, và đặt `run_to_run` làm mặc định cho
`cub::DeviceReduce` (ch10 facts A13). Hãy thử: ghi các tổng theo khối của (c) vào một mảng và rút
gọn mảng đó bằng kernel (b) của bạn, rồi đếm lại số kết quả khác nhau. Điều này quan trọng với một
LLM bất cứ khi nào cùng một prompt phải cho ra cùng các token. Một mẫu số softmax thay đổi ở những
bit cuối có thể lật một trường hợp gần hòa trong greedy decoding.
