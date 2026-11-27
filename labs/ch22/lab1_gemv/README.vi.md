> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 22.1: hai cách chia một tích ma trận-vector lúc decode

Sách: mục 22.2 (pp. 532-536), trên hết là các tiêu chí ở p. 533, và mục 22.5
(p. 539) để thấy vì sao decode là một tích ma trận-vector. Chương 22 không in mã nào, nên cả hai
kernel, lỗi cài sẵn và benchmark đều là của khóa học này.

Lúc decode, một vài token đi qua mỗi lớp tuyến tính:
`Y[batch x out] = X[batch x in] * W^T`, với W được lưu (out, in) theo thứ tự hàng (row-major). Đó là
bố cục và thứ tự đối số của `launchLinear` trong dự án (labs 6.4 và 15.6). Với một
batch tối đa 8 (`MAX_BATCH`), đây là vài tích ma trận-vector dùng chung W
(p. 539), và chi phí của nó là đọc W một lần. Lab này chia nó thành công việc cho từng warp theo hai
cách và đo xem thiết bị của bạn thích cách nào hơn.

- **(a) Lấy đầu ra làm trung tâm, một phép gather (của bạn).** Một warp sở hữu cột đầu ra j. Các lane của nó
  cùng đi dọc `in` theo bước nhảy (stride), đọc hàng j của W một cách liền kề, và mỗi lane giữ một
  tổng tích lũy cho mỗi hàng batch trong thanh ghi (register). Mỗi tổng được rút gọn trên toàn warp bằng
  shuffle, và lane 0 ghi `Y[b][j]`. Không ai khác ghi phần tử đó, nên không có
  phép toán nguyên tử (atomic) nào. Kernel có mức song song bằng `out` warp.
- **(b) Split-K, lấy đầu vào làm trung tâm dọc theo `in` (được cung cấp, có một lỗi).** `in` được cắt thành S
  đoạn, và một warp nhận một cặp (đoạn, đầu ra). Nó rút gọn các tổng riêng phần của đoạn
  theo cùng cách, bằng cùng hàm với phép gather của bạn, và cộng chúng vào `Y[b][j]`,
  nơi S warp dùng chung. Lệnh khởi chạy đặt
  Y về 0 trước. Đây là một phép scatter, và p. 533 nói một phép scatter cần gì: nhiều luồng có thể
  cập nhật cùng một đầu ra một lúc, nên cần phép toán nguyên tử để tránh mất cập nhật. Đổi
  lại, nó có số warp gấp S lần.

Sách gọi gather thường là lựa chọn tốt hơn trên thiết bị CUDA, vì mỗi luồng
tích lũy trong thanh ghi của chính nó và không có gì cần phép toán nguyên tử. Nhưng sách cũng nói gather
so với scatter không phải là tiêu chí duy nhất. Sách nêu mức song song mà một
cách phân rã phơi ra, mức dễ dàng để biết đầu vào nào cấp cho đầu ra nào, và
cân bằng tải mà nó mang lại (p. 533). Ở đây hai tiêu chí cuối như nhau cho cả hai cách chia.
Mức song song là chỗ chúng khác nhau. Ví dụ của chính sách về việc trả giá bằng phép toán nguyên tử để có thêm
mức song song và cân bằng là SpMV ở định dạng COO (pp. 535-536). Lab này đo sự
đánh đổi đó trên các hình dạng lớp mà bạn sẽ gặp trong một mô hình thật.

## Phần của bạn

1. **(a)** Hãy viết phần công việc mà cả hai kernel dùng chung, một lần: `warpRowDots` trong `work/kernel.cu`
   (`simdRowDots` trong `work/kernel.metal`). Một warp nhận một đoạn `[k0, k1)` của một hàng
   của W và để lại, ở lane 0, tích vô hướng của nó với mọi hàng batch của X: các tổng
   trong thanh ghi, vòng lặp theo bước nhảy warp và phép rút gọn (reduction) trên warp. Rồi viết `gemvGatherKernel` và
   `launchGemvGather` quanh nó trong `work/kernel.cu`, và `gemv_gather` trong
   `work/kernel.metal` cùng phần dispatch trong `encodeGemvGather` ở `work/launch.mm`.
   Split-K được cung cấp gọi cùng hàm đó trên đoạn của nó, nên nó không tính gì cho đến khi
   hàm của bạn được viết.
   - Đừng bao giờ dùng literal 32 cho độ rộng warp: `warpSize` trong kernel, thuộc tính thiết bị
     trong hàm khởi chạy (được cung cấp: `deviceWarpSize()`), `threadExecutionWidth` của pipeline
     trên Metal. `warpSum` của chương 10, với kiểu lane-mask riêng của nó,
     được cung cấp.
   - Các khối (threadgroup) gồm `WARPS_PER_BLOCK` = 4 warp (SIMD-group); mỗi đầu ra
     một warp.
2. **(b)** Tìm lỗi trong `gemvSplitKKernel` (Metal: `gemv_splitk`). Chạy `emu` trước
   và thấy nó vượt qua, rồi đến `race`. Hãy viết ra vì sao `emu` không thể thấy nó. Sửa nó:
   - CUDA và HIP: `atomicAdd(&Y[...], total)`.
   - Metal: các hàm atomic chỉ nhận kiểu atomic, nên tham số `Y` của kernel trở thành
     `device atomic_float*`, và phép cộng trở thành
     `atomic_fetch_add_explicit(..., memory_order_relaxed)` (infra-facts C1).
     `launch.mm` gắn cùng buffer đó bất kể kiểu của nó trong kernel.
3. Đo hiệu năng cả hai cách phân rã, sau khi viết ra dự đoán của bạn (bên dưới).

## Các tệp

- `work/kernel.cu` (CUDA, HIP và cả hai trình giả lập): `warpRowDots`, `gemvGatherKernel`
  và `launchGemvGather` là của bạn. Sửa lỗi trong `gemvSplitKKernel`, hàm gọi
  `warpRowDots` của bạn.
  `launchGemvSplitK` được cung cấp và đúng: nó đặt Y về 0 bằng `cudaMemset`, rồi
  khởi chạy `ceil(S x out / 4)` khối 4 warp.
- `work/kernel.metal`: `simdRowDots` và `gemv_gather` là của bạn. Sửa `gemv_splitk`.
  `gemv_zero` được cung cấp.
- `work/launch.mm`: phần dispatch trong `encodeGemvGather` là của bạn. `encodeGemvSplitK` được
  cung cấp: nó dispatch `gemv_zero` rồi `gemv_splitk` trong cùng một encoder tuần tự.
- `main.cpp`: chương trình điều khiển (driver) và bản tham chiếu CPU `cpuLinear`. Bạn không cần sửa nó.

Tên và chữ ký là cố định. Lab 22.2 liên kết chúng khi lời giải của lab này có mặt:

```cpp
void launchGemvGather(const float* X, const float* W, float* Y, int batch, int in, int out);
void launchGemvSplitK(const float* X, const float* W, float* Y, int batch, int in, int out, int splits);
// Metal: kernels gemv_gather, gemv_splitk, gemv_zero. Buffers: X 0, W 1, Y 2, batch 3, in 4, out 5, splits 6.
void encodeGemvGather(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                      id<MTLBuffer> X, id<MTLBuffer> W, id<MTLBuffer> Y,
                      uint32_t batch, uint32_t in, uint32_t out);
void encodeGemvSplitK(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> splitPipeline,
                      id<MTLComputePipelineState> zeroPipeline, id<MTLBuffer> X, id<MTLBuffer> W,
                      id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out, uint32_t splits);
```

Cả hai hàm encode đều encode vào encoder được truyền cho chúng và không commit gì, nên
nhiều lần khởi chạy có thể dùng chung một command buffer.

## Lệnh

- Kiểm thử: `python3 labs/pmpp.py run 22.1 --backend emu` trước (khoảng một phút), rồi
  `--backend race`, rồi `python3 labs/pmpp.py run 22.1` cho GPU của bạn.
- Đo: `python3 labs/pmpp.py run 22.1 --bench` (chỉ trên thiết bị thật).
- Lời giải tham chiếu (solution): thêm `--solution`. Làm lại từ đầu: `python3 labs/pmpp.py reset 22.1 --yes`.

## Các phép kiểm thử làm gì

- Các hình dạng (batch, in, out): (1, 1, 1), (1, 31, 5), (2, 33, 7), (3, 100, 64), (8, 257, 130),
  (5, 64, 1). Trên GPU còn có (1, 4096, 16), (8, 4096, 4096), (4, 14336, 4096) và
  (8, 1000, 4099).
- Số đoạn chia: 1, 2, 3, 7, 32 và mỗi độ rộng warp đầu vào một đoạn, `ceil(in / width)`.
  Một số giá trị để lại các đoạn rỗng (S = 32 ở in = 33 cho 17 đoạn 2 phần tử và 15 đoạn
  rỗng), và một số không chia hết `in`. Tên mỗi trường hợp nói các đoạn rơi thế nào.
- Các trình giả lập (emulator) chỉ mô hình hóa warp 32 lane. Ở đó chương trình điều khiển cũng chạy số đoạn cho
  warp 64 lane, `ceil(in / 64)`, điều này thay đổi lane nào đọc gì và có bao nhiêu đoạn
  rỗng.
- Trên các trình giả lập, các trường hợp split-K của hai hình dạng rộng nhất chạy với ít đầu ra hơn:
  (3, 100, 9) và (8, 257, 5). Mỗi shuffle ở đó là một lần chuyển đổi giữa các luồng, và một
  lần khởi chạy split-K có S x out warp. Việc chia đoạn là dọc theo `in`, chiều này được giữ nguyên.
- X và W chứa đúng các float của chúng, nên trên `emu`, một lần đọc vượt quá một trong hai là một
  báo cáo AddressSanitizer chỉ ra dòng đó. Y bắt đầu là NaN, nên một phần tử không ai ghi sẽ
  thất bại. Theo sau nó là một dải canh gồm các giá trị canh phải còn nguyên.
- Mỗi trường hợp khởi chạy hai lần mà không đặt lại Y. Một lần khởi chạy split-K không tự đặt Y
  về 0 sẽ thất bại ở lần thứ hai.
- Kết quả được so với `y = x W^T` tính bằng double trên CPU, trong phạm vi 1e-4 lần
  (1 + tổng |x w| cho đầu ra đó), như ở lab 15.6. Không bao giờ so từng bit: phép toán nguyên tử cộng các
  tổng riêng phần theo thứ tự bất kỳ.
- Trên GPU, thêm một trường hợp: split-K ở (1, 4096, 16) với S = 64, khởi chạy năm lần. Nó
  có 64 warp cộng vào mỗi đầu ra trong 16 đầu ra, trường hợp tranh chấp nặng nhất. Nó thất bại nếu có
  lần khởi chạy nào sai, và nói là bao nhiêu lần.

## Mỗi backend có thể bắt được gì

- **`emu`** chạy một luồng giả lập cho đến khi nó tới một điểm đồng bộ hóa (một
  shuffle, một rào chắn (barrier)) hoặc trả về, rồi đến luồng kế tiếp. Nó bắt được chỉ số sai, đọc vượt quá X
  hoặc W, ghi vượt quá Y, và một lane bỏ qua một shuffle. Nó **không thể** thấy lỗi
  split-K: lần đọc, cộng và ghi `Y[b][j]` của lane 0 không có điểm đồng bộ hóa nào bên trong,
  nên không luồng nào khác chạy chen vào giữa, và tổng luôn đúng.
- **`race`** chạy các luồng của một khối, và nhiều khối, cùng lúc dưới
  ThreadSanitizer. Nó báo hai lần truy cập không đồng bộ vào cùng vùng nhớ kèm các
  dòng mã nguồn. Ở đây cả hai lần truy cập đều nằm trên dòng đọc-sửa-ghi. Nó chỉ có thể
  thấy hai warp va chạm khi chúng chạy cùng lúc. Nó chạy đồng thời các khối tới 128
  luồng, và nó không thể thấy cùng một chỉ số luồng của hai khối đã chạy lần lượt
  trên một luồng hệ điều hành (`labs/README.md`, phần đầu của
  `common/include/pmpp/emu/cuda_emu.h`). Đó là lý do split-K đánh số các warp sao cho
  S warp của một đầu ra nằm cạnh nhau: với khối 4 warp, S = 4 đặt 4 warp của một đầu ra
  vào mỗi khối, và S = 16 lấp đầy 4 khối cho mỗi đầu ra. Các trường hợp race là
  (2, 64, 8) ở S = 4 và S = 16 trước, rồi thêm vài trường hợp nữa. ThreadSanitizer báo mỗi cặp
  dòng một lần mỗi lần chạy, nên một trường hợp sau có thể vượt qua dù lỗi vẫn còn đó.
- **Metal** không có công cụ phát hiện race ở đây. Một lần cập nhật bị mất chỉ hiện ra như một tổng sai, và chỉ
  khi hai SIMD-group tình cờ va chạm về thời gian. Trên máy mà lab này được dựng,
  phiên bản thường cho tổng sai ở phần lớn các trường hợp split-K và ở 5 trong 5 lần khởi chạy
  tranh chấp. Đó là do thời điểm, không phải một sự bảo đảm. Quy tắc của NVIDIA cho GPU của chính họ là
  "Reading from or writing to the same memory location without memory fences or
  synchronization results in undefined behavior" (ch10 facts A1). Một lần chạy vượt qua không
  làm cho kernel đúng.
- **Không gì trên Mac kiểm tra warp 64 lane.** Cả hai trình giả lập đều mô hình hóa warp 32 lane, và
  Metal báo độ rộng của chính nó, nên một literal 32 có thể vượt qua ở mọi nơi tại đây. Hãy kiểm tra bằng cách đọc.
  Vòng lặp shuffle bắt đầu ở `warpSize / 2`, mask có kiểu riêng của nó, và bước nhảy là
  `warpSize`.

## Các backend khác nhau ở đâu (không có trong sách)

- **HIP**: các intrinsic `_sync` `static_assert` một mask 64 bit trên mọi target, do đó có
  `lane_mask_t` (infra-facts B3). `warpSize` là 64 trên gfx9 (Instinct) và 32 trên các
  dòng AMD sau đó (ch05 facts B5). `atomicAdd` cho float có tồn tại, nhưng theo mặc định nó biên dịch thành một
  thao tác an toàn, có thể là một vòng lặp compare-and-swap, trừ khi bản dựng truyền
  `-munsafe-fp-atomics` (infra-facts B1). Bản dựng HIP của khóa học thì không. Không điều gì trong số này
  từng được dựng hoặc chạy với ROCm ở đây: unverified.
- **Metal**: chỉ số SIMD-group và lane đến từ `[[simdgroup_index_in_threadgroup]]` và
  `[[thread_index_in_simdgroup]]`, vì Metal để việc chia một threadgroup thành
  các SIMD-group cho phần hiện thực (ch04 facts C2). Độ rộng SIMD-group là
  `threadExecutionWidth` của pipeline (ch03 facts C9), và `launch.mm` khiến mọi
  threadgroup là một số nguyên lần SIMD-group. `simd_sum` cộng trên một SIMD-group không cần
  mask và cần GPU Apple7 trở lên (infra-facts C2). Phép toán nguyên tử float cần
  `device atomic_float*`, Metal 3 và Apple7 trở lên, và `memory_order_relaxed`
  (infra-facts C1). Nếu `gemv_splitk` không dựng được trên GPU của bạn, chương trình điều khiển báo đó
  là lý do mỗi trường hợp split-K thất bại.
- **Đặt về 0 trên Metal**: một compute pass không có `cudaMemset`. Lab 10.1 ghi số 0 từ
  CPU, cách này chỉ chạy được trước khi command buffer được commit và chỉ cho một
  lần khởi chạy mỗi command buffer. Ở đây `gemv_zero` là một dispatch riêng, trong cùng encoder
  tuần tự với `gemv_splitk`. Một compute encoder thường chạy các dispatch của nó "in dispatched
  order", và các buffer device được theo dõi xung đột, nên `gemv_splitk` thấy các số 0
  (ch02 facts C9).

## Trước khi đo hiệu năng, hãy dự đoán

Hãy viết những điều này ra trước:

1. Trên lớp 4096 -> 16 ở batch 1, cách chia tốt nhất sẽ nhanh hơn gather bao nhiêu lần.
2. Trên lớp 4096 -> 4096 ở batch 1, cùng tỉ lệ đó. Dưới 1 nghĩa là gather thắng.

Những điều cần cân nhắc:

- **Số luồng được phơi ra.** Gather có mỗi đầu ra một warp: 16 warp trên lớp hẹp.
  Split-K có S x 16.
- **Thiết bị của bạn có thể giữ thường trú bao nhiêu warp.** Trên CUDA và HIP,
  `python3 labs/pmpp.py doctor` báo `multiProcessorCount`,
  `maxThreadsPerMultiProcessor` và `warpSize`, và benchmark in ra tích của chúng.
  Metal không có API occupancy (mức chiếm dụng), và công cụ GPU của Xcode hiển thị occupancy thay vào đó (ch06 facts
  C5, ch15 facts C3).
- **Số phép toán nguyên tử mỗi đầu ra**: S.
- **Số byte W được đọc**: như nhau với mọi S. Mỗi warp phụ trách đoạn đọc phần của riêng nó trong
  hàng j một lần.
- **Trên HIP**, một `atomicAdd` float có thể là một vòng lặp compare-and-swap (infra-facts B1).

## Đọc kết quả đầu ra

- Các lớp: 4096 -> 4096; 4096 -> 16, một lớp hẹp như một router; 14336 -> 4096, một
  phép chiếu xuống của MLP. Hai hình dạng cuối là của chúng tôi. Batch 1, 4 và 8.
- Các kernel: gather (S = 1), rồi split-K ở S = 2, 4, 8, ... tới `in / (8 x width)`, để
  mỗi đoạn chứa ít nhất 8 độ rộng warp đầu vào.
- Các kernel chạy luân phiên: một vòng không đo, rồi năm vòng, và mỗi kernel giữ
  vòng tốt nhất của nó.
- Một lần gửi được đo thời gian khởi chạy kernel một lần cho mỗi bản sao của W, đi qua các bản sao mà
  tổng cộng chứa ít nhất 256 MB (tối đa 1024 bản sao). Vì vậy một lần khởi chạy không tìm thấy
  trọng số của nó ở chỗ lần khởi chạy trước để lại. Dòng tiêu đề của mỗi lớp nói có bao
  nhiêu bản sao. `time (us)` là cho mỗi lần khởi chạy. Với split-K nó bao gồm việc đặt về 0
  (`cudaMemset`, hoặc dispatch `gemv_zero` trên Metal), vì đó là một phần của
  lần khởi chạy. CUDA và HIP đo lần gửi bằng đồng hồ host, bao quanh các lần khởi chạy và
  lần đồng bộ cuối. Metal dùng thời điểm bắt đầu và kết thúc trên GPU của command buffer.
- `W GB/s` là 4 x in x out byte chia cho thời gian: số byte W mà một tích ma trận-vector
  phải đọc, không phải mọi thứ hệ thống bộ nhớ đã di chuyển. Trang labs so sánh nó
  với băng thông đo được của lab 5.1, giới hạn tốc độ cho một tích ma trận-vector.
- `warps` là S x out. Dòng cuối của mỗi batch nêu S tốt nhất và tốc độ của nó
  so với gather.
- Các hàng được đưa vào báo cáo dưới dạng hàng `gemv`: `{kernel, splits, batch, in, out, seconds,
  weight_gb_per_s}`.

Rồi hãy đọc các con số của bạn đối chiếu với p. 533. Việc không có phép toán nguyên tử của gather thắng ở đâu, và
mức song song thêm của split-K thắng ở đâu? Hình dạng nào đảo ngược câu trả lời, và vì sao? Các
con số là phép đo trên thiết bị của bạn ở các hình dạng này. Chúng không phải là thuộc tính của các
kernel.

## Đi xa hơn

- **Một backend thực tế vạch ranh giới ở đâu** (của chúng tôi, từ ch22-facts D6): llama.cpp có
  các kernel ma trận-vector riêng cho batch nhỏ và chọn chúng theo số token trong
  batch. Trên CUDA và HIP đó là `mmvf` và `mmvq` tới 8 token, với các ngoại lệ theo từng GPU.
  Trên Metal đó là `mul_mv_ext` cho 2 đến 8 token. Benchmark của lab 22.2 thêm
  phép gather của lab này làm đường cho batch nhỏ với batch tới 8, khi lời giải của lab này
  có mặt.
- **Cách chia thứ ba, không có phép toán nguyên tử** (của chúng tôi): giao cho mỗi đầu ra cả một khối S warp.
  Mỗi warp rút gọn đoạn của nó, và khối kết hợp S tổng riêng phần trong bộ nhớ chia sẻ (shared memory)
  bằng phép rút gọn hai giai đoạn của chương 10. Cách này phơi ra cùng mức song song như split-K
  với S tới số warp trong một khối, và không cần phép toán nguyên tử hay việc đặt về 0. Hãy viết nó và thêm
  nó vào benchmark.
- **Tính tất định**: với phép toán nguyên tử, S tổng riêng phần của một đầu ra được cộng theo bất kỳ
  thứ tự nào mà các warp hoàn thành, nên các bit cuối của Y có thể thay đổi giữa các lần chạy. Gather có
  một thứ tự. Hãy đếm số kết quả khác nhau qua 20 lần chạy, như lab 10.1 làm, và nghĩ xem điều
  đó có ý nghĩa gì với greedy decoding.
