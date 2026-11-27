> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 23.1: giấu các lần sao chép sau kernel

Sách: mục 23.3 (pp. 553-563): stream và tham số khởi chạy thứ tư (p. 556), `cudaStreamSynchronize`
(p. 559), `cudaMemcpyAsync` (p. 560), bộ nhớ ghim (pinned memory) (pp. 562-563). Kernel, đường ống
chia chunk, phiên bản Metal và mọi phép đo là của khóa học này: sách chồng lấp một lần trao đổi vùng
biên (halo) qua mạng với các hàng bên trong của một stencil, và không trình bày đường ống host-device
nào. Mẫu sao chép và thực thi chia giai đoạn (staged) là của NVIDIA (Best Practices Guide 10.1.2,
`notes/today/ch23-facts.md` A7); mẫu Metal là của Apple (C5).

Vòng đi và về của chương 2 làm ba việc lần lượt: sao chép đầu vào sang device, chạy kernel, sao chép
đầu ra về. Trong khi đường truyền sao chép, các lõi (core) của GPU chờ; trong khi các lõi tính toán,
đường truyền chờ. Cách khắc phục của mục 23.3, áp dụng cho một GPU: cắt công việc thành các chunk và
để lần sao chép của chunk 2 chạy trong khi kernel của chunk 1 chạy. Lab này bắt bạn viết đường ống đó
rồi đo xem device của bạn mang lại được bao nhiêu từ nó.

## Kernel

`polyKernel`: `out[i] = p(in[i])`, trong đó p là một đa thức bậc `degree` với các hệ số cố định (0.5
cho lũy thừa chẵn, -0.25 cho lũy thừa lẻ), tính bằng quy tắc Horner: `degree` phép nhân-cộng cho mỗi
phần tử. Nó đại diện cho một lớp tính theo từng phần tử với chi phí bạn có thể điều chỉnh. Benchmark điều
chỉnh nó sao cho kernel chạy lâu xấp xỉ bằng thời gian sao chép đầu vào sang, đó là lúc chồng lấp
có ý nghĩa nhất.

## Phần của bạn

Quy tắc chia chunk (cả hai backend): chunk k là `[k*n/chunks, (k+1)*n/chunks)`, phép chia số nguyên.
Bỏ qua các chunk rỗng (n có thể nhỏ hơn `chunks`). Đánh số các chunk bạn xếp hàng là j = 0, 1, 2, ...
theo thứ tự.

**CUDA, HIP và trình giả lập (emulator)**, trong `work/kernel.cu`: `runStaged(job, streams, chunks, order)`.

1. Tạo `streams` stream bằng `cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)`.
2. Xếp chunk j vào stream `j mod streams`: `cudaMemcpyAsync` phần của nó trong `hostIn` vào `d_in`,
   `launchPoly` từ `d_in` vào `d_out`, `cudaMemcpyAsync` phần của nó trong `d_out` vào `hostOut`.
   Cùng một độ lệch ở mọi chỗ.
3. `Order::Depth` xếp hàng từng chunk một (sao chép vào 0, kernel 0, sao chép ra 0, sao chép vào 1,
   ...). `Order::Breadth` xếp mọi lần sao chép vào, rồi mọi kernel, rồi mọi lần sao chép ra.
4. Đồng bộ hóa mọi stream, hủy mỗi stream đúng một lần, trả về 0. Khi gặp lỗi CUDA, in nó ra và trả
   về 1 (macro `CHECK`).

Chương trình điều khiển (driver) sở hữu `d_in`, `d_out` và các buffer host, vốn được ghim
(`cudaMallocHost`) hoặc phân trang (bộ nhớ heap thông thường) tùy theo test.

**Metal**, trong `work/launch.mm`: `runInFlight(job, buffers, chunks)`, mẫu mà Apple ghi trong tài
liệu để chồng lấp công việc của CPU và GPU (C5). K = `buffers`.

1. K ô, mỗi ô là một cặp buffer dùng chung đủ dài cho chunk lớn nhất, lấy từ `slotBuffer` được cung
   cấp.
2. Một dispatch semaphore bắt đầu ở K.
3. Với chunk j: chờ semaphore; nếu ô `j mod K` vẫn đang giữ một chunk trước đó, gọi `waitFor` command
   buffer của nó và sao chép đầu ra của nó vào `hostOut`; sao chép đầu vào của chunk từ `hostIn` vào
   ô; encode bằng `encodePoly` trong một command buffer riêng; thêm một completion handler chỉ làm
   một việc là báo hiệu semaphore; commit.
4. Sau chunk cuối cùng, gọi `waitFor` các ô vẫn đang chạy, cũ nhất trước, và rút chúng.

Việc rút nằm trên luồng (thread) của bạn. Apple ghi trong tài liệu khi nào một completion
handler chạy ("the GPU device calls immediately after the GPU finishes running the commands",
infra-facts C12), và các tệp facts không ghi lại gì về việc nó chạy trên luồng nào, nên handler chỉ
làm đúng việc mà mẫu của Apple làm ở đó: trả lại token (C5).

## Các tệp

- Sửa, bên dưới dòng `your code below`: `work/kernel.cu` (`runStaged`) và `work/launch.mm`
  (`runInFlight`).
- Được cung cấp phía trên dòng đó:
  - `kernel.cu`: `polyKernel`; `launchPoly(d_in, d_out, count, degree, stream)`, 256 luồng mỗi khối
    (block), hàm này ghi lại mỗi lần khởi chạy cho các test và từ chối một count nhỏ hơn 1 (một lưới
    không khối nào là lỗi khởi chạy trên CUDA; nó từ chối trên mọi backend để không test nào phụ
    thuộc vào việc trình giả lập làm gì với trường hợp đó); `runSerial`, đường cơ sở của chương 2.
  - `launch.mm`: `encodePoly(encoder, inBuf, inOffset, outBuf, outOffset, count, degree)`, hàm này
    ghi lại các buffer được cho và từ chối một khoảng vượt quá cuối buffer; `waitFor`, hàm này chờ,
    báo cáo một command buffer thất bại và cộng dồn thời gian GPU; `runSerialMetal`, đường cơ sở
    trong một command buffer; `slotBuffer`.
- Được cung cấp, không sửa: `work/kernel.metal` (hoàn chỉnh), `overlap.h` (các job, `Order`, mọi
  khai báo), `main.cpp` (chương trình điều khiển, bản tham chiếu CPU, hiệu chuẩn, benchmark),
  `probe.cu` (các công cụ đo cho CUDA và HIP: số copy engine, cấp phát bộ nhớ ghim, đo thời gian
  từng phần).

## Lệnh

- Kiểm thử: `python3 labs/pmpp.py run 23.1 --backend emu` trước, rồi `python3 labs/pmpp.py run 23.1`.
- Đo: `python3 labs/pmpp.py run 23.1 --bench` (chỉ trên thiết bị thật).
- Lời giải tham chiếu: thêm `--solution`.

## Lab này khác sách ở đâu

- **Stream không chặn.** Sách tạo stream bằng `cudaStreamCreate` (p. 556) hoặc bằng
  `cudaStreamDefault` cùng một độ ưu tiên (p. 558). Đó là các stream chặn: bất cứ thứ gì xếp trong
  stream mặc định kiểu cũ (legacy) đều chờ chúng và bắt chúng chờ (ch23-facts A3, A9). `runSerial`
  và các phép đo thời gian của chương trình điều khiển dùng stream mặc định, nên các stream của lab
  này là `cudaStreamNonBlocking`.
- **Không có độ ưu tiên.** Ở đây chỉ có một loại công việc. Độ ưu tiên chỉ là gợi ý cho kernel,
  không có tác dụng với các lần sao chép (A2), và ROCm mặc định bỏ qua chúng (B2).
- **Không có event.** Sách cần event vì ba kernel trong ba stream dùng chung một giá trị chuẩn
  (pp. 559-560). Ở đây ba thao tác của mỗi chunk nằm trong một stream, và chỉ riêng thứ tự của stream
  đó đã đủ để sắp thứ tự cho chúng.
- **"Yêu cầu" bộ nhớ ghim.** Sách nói `cudaMemcpyAsync()` yêu cầu bộ nhớ host được ghim (p. 563).
  Tài liệu tham chiếu của NVIDIA nhẹ nhàng hơn: từ bộ nhớ phân trang (pageable), lời gọi vẫn chạy nhưng "will
  revert to a synchronous behavior which will not overlap with other work" (A6); header của HIP nói
  điều tương tự (B4). Các test chạy cả hai loại; benchmark đo chi phí.
- **Không có sao chép trên Metal.** Trên Apple silicon, một buffer dùng chung đồng thời là bộ nhớ
  của CPU và của GPU, và không có copy engine nào được ghi trong tài liệu (C4). Công việc còn lại
  cần giấu là của CPU: nạp (ghi đầu vào của một chunk vào buffer dùng chung) và rút (đọc
  đầu ra của nó). Một luồng CPU làm cả hai, nên đường ống có hai làn, không phải ba: làn của CPU,
  nạp + rút, và làn của GPU. Dù các buffer chồng lấp tốt đến đâu, một lần chạy cũng mất
  ít nhất bằng làn dài hơn (giới hạn của trình chạy từng bước, n nhân max(fill + drain, kernel) cho
  mỗi chunk). Trường hợp tốt nhất, hai làn dài bằng nhau, là một nửa thời gian tuần tự; chi phí cho
  mỗi chunk, thời gian chờ mỗi ô và việc hai bên dùng chung một bộ nhớ làm hao hụt mức đó; các hàng
  `inflight` đo hao hụt bao nhiêu.

## Các test làm gì

Kích thước: n = 1, 7, 1000, 4099 và 65536 số float; thiết bị thật thêm kích thước của benchmark (xem
bên dưới). Bậc 7. Đầu vào trong [-0.75, 0.75], từ một seed cố định.

- **CUDA, HIP, trình giả lập.** (streams, chunks) = (1, 1), (1, 4), (2, 2), (3, 5), (4, 16),
  (8, 3); với n = 7 thêm (2, 16) và (3, 16), khi các chunk rỗng nằm giữa các chunk đầy; ở kích thước
  của benchmark thêm (4, 64). Cả hai thứ tự, buffer host được ghim và phân trang (cùng một kiểu cấp
  phát trên trình giả lập).
- **Metal.** K = 1, 2, 3, 4 với chunks = 1, 4, 2, 5, 16, 3, và 64 ở kích thước của benchmark.

Với mỗi trường hợp:

1. `hostOut` bằng phép tính Horner của CPU trong `main.cpp` trong một dung sai tương đối (Metal
   không mặc định dùng số học IEEE nghiêm ngặt); `hostOut` bắt đầu là NaN, nên một chunk chưa bao giờ
   được sao chép về sẽ hiện thành NaN. `hostIn` không thay đổi.
2. Họ CUDA, từ bản ghi của `launchPoly`: mọi phần tử được tính đúng một lần, mỗi chunk không rỗng một
   lần khởi chạy trên đúng ranh giới của quy tắc, đầu vào và đầu ra ở cùng độ lệch, không lần khởi
   chạy nào bị từ chối, không lần nào trong stream mặc định, `min(streams, non-empty chunks)` stream
   khác nhau, và các chunk liền kề nằm trong các stream khác nhau khi `streams` từ 2 trở lên.
3. Metal, từ bản ghi của `encodePoly`: mỗi chunk không rỗng một lần dispatch với kích thước theo quy
   tắc, nhiều nhất K buffer đầu vào khác nhau, các chunk liền kề nằm trong các buffer đầu vào khác
   nhau khi K từ 2 trở lên, mỗi command buffer một lần `waitFor`, và chunk j được encode sau nhiều
   nhất `max(0, j - K + 1)` lần chờ. Điều kiện cuối cùng đó là thứ làm nó thành một đường ống: chờ
   nhiều hơn mức ô cần sẽ để GPU rảnh trong khi CPU nạp.
4. Chỉ trình giả lập: AddressSanitizer dừng một chunk chạy quá cuối mảng kèm dòng mã nguồn, và báo
   một stream bị hủy hai lần như một double free kèm cả hai dòng. Một stream không bao giờ bị hủy làm
   trường hợp thất bại thông qua số byte heap còn sống mà AddressSanitizer đếm, con số này phải bằng
   nhau trước và sau lời gọi. (LeakSanitizer sẽ báo cáo nó khi thoát, nhưng trên máy Mac dùng để
   dựng lab này, nó nói "detect_leaks is not supported on this platform".)

Các đường cơ sở được cung cấp cũng được kiểm tra, một cách lặng lẽ: một trường hợp chỉ xuất hiện nếu
nó thất bại. Mã khởi đầu (starter) chưa sửa trả về -1 từ `runStaged` và `runInFlight`, nên mọi
trường hợp đều thất bại.

## Các test không thấy được gì

- **Bất cứ điều gì về thời gian, trên trình giả lập.** Mọi lời gọi stream ở đó đều đồng bộ: một lời
  gọi đã xong khi nó trả về. Vì vậy một `runStaged` không có `cudaStreamSynchronize` trước khi trả
  về, một bản có `cudaDeviceSynchronize` sau mỗi chunk, và các buffer phân trang đều qua. Trên trình
  giả lập, khi lời giải tham chiếu được sửa theo từng cách trong hai cách đầu, mọi trường hợp đều
  qua. Chỉ một benchmark trên GPU mới cho thấy cách thứ hai và thứ ba. Trên GPU, cách đầu tiên là
  một race condition (tranh chấp dữ liệu): lời gọi trả về trong khi các lần sao chép vẫn đang đổ vào
  `hostOut`, và việc test có bắt được hay không phụ thuộc vào thời điểm.
- **Stream của các lần sao chép.** Chỉ các lần khởi chạy được ghi lại. Một lần sao chép xếp trong
  stream khác với kernel của chunk đó là một race condition trên GPU và vô hình trên trình giả lập.
- **Chiều sâu so với chiều rộng.** Cả hai thứ tự phát cùng các thao tác; không có gì được ghi lại
  phân biệt được chúng. Benchmark thì có thể.
- **`k*n` trong một `int`.** Ở kích thước của test nó không bao giờ tràn. Ở kích thước của benchmark
  với 64 chunk thì có thể, đó là lý do thiết bị thật chạy một trường hợp 64 chunk ở đó.
- **Semaphore.** Chờ chính command buffer của ô là đủ để đúng, nên một `runInFlight` không có
  semaphore vẫn qua. Semaphore là cơ chế kiểm soát tốc độ mà Apple ghi trong tài liệu (C5); hãy giữ
  nó, và giữ semaphore cân bằng: trên máy dùng để dựng lab này, giải phóng một dispatch semaphore có
  giá trị đếm thấp hơn giá trị khởi đầu đã làm tiến trình kết thúc.

## Trước khi đo hiệu năng, hãy dự đoán

Hãy dùng trình chạy từng bước trong mục 23.3 của bài học. Trên CUDA và HIP, chương trình điều khiển
làm cho kernel chạy lâu xấp xỉ bằng lần sao chép vào, nên hãy bắt đầu từ preset "equal stages" của
nó. Trên Metal, nó làm cho kernel chạy lâu xấp xỉ bằng nạp + rút cộng lại. Hãy viết
những điều này ra trước; bài học hỏi ba điều đầu tiên dưới các id trong ngoặc.

1. CUDA hoặc HIP: thời gian tuần tự chia cho thời gian chia giai đoạn với 4 stream, 16 chunk, chiều
   sâu trước, bộ nhớ ghim (`ch23/labs/p-staged`). Trên CUDA, trước hết hãy tra `asyncEngineCount`
   (trong `python3 labs/pmpp.py doctor`, hoặc các dòng đầu của benchmark và hàng `parts` của nó) và
   đặt số copy engine của trình chạy từng bước cho khớp.
   Trên HIP con số đó không có ý nghĩa (B3).
2. Cùng tỉ số đó với buffer host phân trang (`ch23/labs/p-pageable`).
3. Metal: thời gian in-flight với 1 buffer chia cho thời gian in-flight với 3 buffer, ở 16 chunk, với
   khung nhìn Metal của trình chạy từng bước, kernel bằng nạp + rút
   (`ch23/labs/p-metal`). Rồi, không ghi lại: 3 buffer sẽ tiến gần đến một nửa thời gian tuần tự
   đến mức nào?
4. Không ghi lại: chiều rộng trước có khác chiều sâu trước trên GPU của bạn không? Trình chạy từng
   bước nói là có thể, với một copy engine, nếu engine đó phục vụ các lần sao chép theo thứ tự
   chúng được phát (E2). NVIDIA không ghi trong tài liệu thứ tự đó lẫn cách các hàng đợi sao chép
   của họ thay đổi nó (A7, A8).

Sau lần chạy, preset "your device" của trình chạy từng bước đặt các hàng `staged` bạn đo được bên cạnh
mô hình.

## Đọc đầu ra

Kích thước là số float vừa trong một phần mười sáu bộ nhớ mà device báo cáo (`totalGlobalMem` trên
CUDA và HIP; giá trị nhỏ hơn giữa `maxBufferLength` và `recommendedMaxWorkingSetSize` trên Metal),
tối đa 64 Mi float (256 MiB mỗi mảng). Mức trần là một tham số của lab này, đặt ra để giữ lần chạy
ngắn, không phải một sự thật về device nào. Các dòng đầu in ra kích thước được chọn và lý do.

- **CUDA và HIP** in `asyncEngineCount` và `concurrentKernels`. NVIDIA: với một copy engine, một
  lần sao chép vào và một lần sao chép ra không thể chạy cùng lúc; với hai thì có thể (A7). Trên
  ROCm, `asyncEngineCount` là một hằng số của runtime trên mọi device (B3).
- **Hiệu chuẩn.** Lần sao chép vào của cả mảng được đo thời gian (bộ nhớ ghim, event, tốt nhất của
  vài lần), rồi đến kernel ở bậc 1, 2, 4, ... cho đến khi nó chạy lâu gấp đôi, rồi một bậc ở giữa
  hai bậc cuối. Bậc có thời gian kernel gần nhất với lần sao chép vào được dùng cho mọi hàng và được
  in ra. Trên Metal, kernel được khớp với nạp + rút thay vào đó: CPU làm cả hai, lần lượt
  trên một luồng, nên tổng đó là làn của nó, và kernel được đo thời gian bên trong `runSerialMetal`
  được cung cấp, đúng như hàng `serial`. Ở bậc thấp, thời gian của kernel có thể không đổi: việc đọc
  và ghi mảng, chứ không phải phép tính, quyết định nó (tỉ số của chương 5). Nếu thời gian không bao
  giờ tăng, hoặc một bậc thấp chạy lâu hơn một bậc cao, thì có thứ khác đang dùng GPU (hoặc, trên
  Metal, CPU) trong lần chạy; trên máy dùng để dựng lab này, điều đó đã xảy ra khi các chương trình
  khác đang bận. Hãy chạy lại trước khi so sánh bất cứ thứ gì.
- **`parts`**: từng phần riêng lẻ trên cả mảng. CUDA và HIP: `h2d_s`, `kernel_s`, `d2h_s`, cùng
  `async_engines` và `concurrent_kernels`. Metal: `fill_s`, `kernel_s` (thời gian GPU của command
  buffer duy nhất trong đường tuần tự), `drain_s`.
- **`firsttouch`** (Metal): tạo một buffer dùng chung cỡ bằng mảng, và lần nạp đầu tiên của nó,
  so với lần nạp vào một buffer đã được dùng. Trên máy dùng để dựng lab này, lần nạp đầu tiên
  mất gấp vài lần, gần như toàn bộ nằm ở lần ghi đầu tiên vào mỗi trang. Đó là lý do `slotBuffer`
  giữ các ô của bạn từ lời gọi này sang lời gọi sau: với buffer tạo bên trong mỗi lời gọi, các hàng
  bên dưới phần lớn sẽ đo thời gian lỗi trang (page fault).
- **`serial`**: đường cơ sở được cung cấp, `host` = `pinned` hoặc `pageable` (CUDA, HIP) hoặc
  `shared` (Metal).
- **`staged`** (CUDA, HIP): `streams` 1, 2, 3, 4, 8 nhân `chunks` 4, 16, 64 nhân `order`, bộ nhớ
  ghim; cộng thêm 4 stream, 16 chunk, chiều sâu, phân trang.
- **`inflight`** (Metal): `buffers` 1 đến 4 nhân `chunks` 4, 16, 64, với `gpu_s`, tổng
  `GPUEndTime - GPUStartTime` trên các command buffer đã truyền cho `waitFor` (ch05-facts C9), và
  `overlap_observed`: CPU có encode một chunk, ngay sau khi nạp nó, trong khi GPU đang chạy
  command buffer của một chunk khác hay không. Cả hai đồng hồ đều là mach time. Đó là một quan sát về
  máy của bạn; Apple ghi trong tài liệu mẫu này, không phải kết quả.
- `seconds` của mỗi hàng là thời gian thực của một lời gọi trọn vẹn (bao gồm cả tạo stream), tốt
  nhất của vài lời gọi sau một lời gọi khởi động không tính giờ. Đầu ra của lời gọi khởi động và
  lời gọi cuối được so sánh với đầu ra của đường tuần tự, bản thân đường này được kiểm tra với CPU
  trên một mẫu; một kết quả sai được báo cáo mà không có thời gian.
- Các dòng cuối chia các thời gian mà dự đoán của bạn đã hỏi.

Các con số là phép đo trên máy của bạn ở một kích thước và một bậc, không phải tính chất của phương
pháp.

## Các backend khác nhau ở đâu (không có trong sách)

- **CUDA**: `asyncEngineCount` copy engine (A7). Theo mặc định, stream được ánh xạ lên 8 hàng
  đợi công việc phần cứng (`CUDA_DEVICE_MAX_CONNECTIONS`); nhiều stream hoạt động hơn số hàng đợi có
  thể làm các stream không liên quan phải chờ nhau (A8). Hãy để ý các hàng 8 stream.
- **HIP**: cùng các lời gọi thông qua `pmpp/vendors/hip.h` (`hipStreamCreateWithFlags`,
  `hipMemcpyAsync`, `hipHostMalloc`). Theo mặc định, mỗi device có nhiều nhất 4 hàng đợi phần cứng;
  nhiều stream hơn sẽ dùng chung chúng theo vòng tròn (ch22-facts B1). Một lần sao chép bất đồng bộ
  từ bộ nhớ phân trang được thực hiện đồng bộ (B4). `asyncEngineCount` là một hằng số (B3). Chưa chạy
  trên phần cứng AMD ở đây.
- **Metal**: không có sao chép host-device cho buffer dùng chung và không có copy engine (C4);
  nhiều buffer cùng chạy với một semaphore và một completion handler là cách chồng lấp được ghi
  trong tài liệu (C5). Theo mặc định, một hàng đợi giữ nhiều nhất 64 command buffer chưa hoàn thành
  (C1); lab này giữ nhiều nhất 4 cái cùng chạy.

Nơi bạn gặp lại điều này: `tensor.to("cuda", non_blocking=True)` của PyTorch là một
`cudaMemcpyAsync` (ch02-facts D4), và chính ghi chú của PyTorch nói rằng từ bộ nhớ phân trang "the
call may block the host" (appC-facts D1). Backend CUDA của llama.cpp tạo các stream của nó với
`cudaStreamNonBlocking` (ch23-facts D3).

## Đi xa hơn

- Cho `runStaged` một thứ tự thứ ba: sao chép vào chunk j + 1 trước khi khởi chạy chunk j. Trên GPU
  của bạn, nó nằm ở đâu giữa chiều sâu và chiều rộng?
- Đo lại các hàng 8 stream với `CUDA_DEVICE_MAX_CONNECTIONS=4` và với `=16` (A8).
- Trên Metal, chuyển việc rút vào completion handler, và nói rõ bạn cần biết gì về luồng
  của handler trước khi tin tưởng nó.

## Trạng thái của lời giải tham chiếu

verified on Metal (test và benchmark), emulator-checked, front-end-checked cho `kernel.cu` và
`probe.cu`. Phép kiểm tra front-end chỉ thấy nhánh trình giả lập của `probeEngines`: `cudaDeviceProp`
của header stub không có `asyncEngineCount`. unverified trên CUDA và trên HIP: chưa có máy nào có
một trong hai chạy nó.
