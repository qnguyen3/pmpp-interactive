> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 12.2: săn lỗi bộ lọc

Sách: các mục 12.2 đến 12.4 (pp. 290-297), Figs. 12.2, 12.3 và 12.6. Các lỗi là của khóa học
này, không phải của sách.

`work/kernel.cu` và `work/kernel.metal` chứa ba bộ lọc không ổn định của lab 12.1, giữ = key <
threshold, được viết đúng ngoại trừ **ba lỗi cài sẵn độc lập**. Chúng biên dịch được. Hãy tìm và
sửa cả ba.

- `filterAtomicKernel` (Metal: `filter_atomic`): một lần cập nhật bộ đếm cho mỗi khóa được giữ.
- `filterWarpAggregatedKernel` (Metal: `filter_warp_aggregated`): một phép toán nguyên tử
  (atomic) mỗi warp, từ một lần bỏ phiếu mà mọi lane đều tham gia.
- `filterPrivatisedKernel` (Metal: `filter_privatised`): một danh sách riêng và bộ đếm riêng
  trong bộ nhớ chia sẻ (shared memory), một phép toán nguyên tử công khai mỗi khối.

Tên, chữ ký và hợp đồng là của lab 12.1: `launchFilterAtomic`,
`launchFilterWarpAggregated`, `launchFilterPrivatised(input, output, n, threshold,
outputSize)`, và trên Metal là `encodeFilterAtomic` cùng các hàm tương tự. Bên gọi đặt
`*outputSize` về 0 trước mỗi lần gọi.

- Sửa `work/kernel.cu` (ba kernel; các hàm khởi chạy bên dưới chúng là đúng) và
  `work/kernel.metal`. Cùng ba lỗi nằm trong cả hai tệp. `work/launch.mm` là đúng.
- Không phải lỗi: `lane_mask_t` và các hàm trợ giúp có kiểu `maskPopc` và `maskFfs`, lần bỏ
  phiếu mà mọi lane tham gia trước khi bất kỳ luồng nào return, chỉ số lane tính từ
  `warpSize`, `atomicAdd` kiểu cũ dưới `__HIPCC__`, và trên Metal là các hàm SIMD group được
  gọi từ mọi luồng. README của lab 12.1 giải thích lý do cho từng điều.
- Kiểm thử: `python3 labs/pmpp.py run 12.2 --backend emu`, rồi `--backend race`, rồi
  `python3 labs/pmpp.py run 12.2` cho GPU của bạn.

## Các bài kiểm thử làm gì

Các phép kiểm tra của lab 12.1, ở kích thước nhỏ hơn. Với mỗi đầu vào, chương trình điều khiển
(driver) kiểm tra rằng `*outputSize` bằng số khóa mà một bộ lọc tuần tự giữ, rằng
`output[0, count)` sau khi sắp xếp bằng danh sách của bộ lọc tuần tự sau khi sắp xếp, và rằng
mọi vị trí từ `count` đến `n` vẫn giữ giá trị canh `0xFFFFFFFF` mà nó đã ghi trước lần gọi. Rồi
nó đặt `*outputSize` về 0 lần nữa, gọi kernel lần thứ hai và kiểm tra cùng những điều đó.

- Kích thước 1, 31, 32, 33, 255, 256, 257, 1000, 4099; trên GPU thêm 65537 và 2^20. Backend
  `race` dừng ở 1000.
- Các khóa phân bố đều trong [0, 2^24) với tỉ lệ được giữ 0, 0.001, 0.03, 0.5, 0.97 và 1, cùng
  các mẫu của lab 12.1 (mọi khóa, không khóa nào, chỉ khóa cuối cùng, chỉ khóa 31 trong mỗi 32,
  chỉ các khóa 32 đến 63 trong mỗi 64, cứ một khóa giữ một khóa bỏ, các đoạn 40).
- Trên GPU mỗi đầu vào được chạy 5 vòng, mỗi vòng hai lần gọi, và một thất bại cho biết nó sai
  trong bao nhiêu vòng. Trên các trình giả lập, một vòng (dưới mỗi thứ tự trong ba thứ tự luồng
  của `emu`).

## Dự đoán của bạn

Trước khi chạy bất cứ thứ gì, hãy đọc `work/kernel.cu` và ghi lại các lỗi bạn có thể tìm ra
bằng cách đọc. Với mỗi lỗi, ghi lại backend nào bạn nghĩ sẽ báo nó: `emu`, `race`, GPU của bạn,
hay không backend nào, và vì sao. Trang labs hỏi một con số từ đó (`ch12/labs/p-lab2-emu`):
trong ba lỗi, `emu` sẽ bắt được bao nhiêu? Rồi chạy `emu`, `race` và GPU của bạn, và điền vào
bảng này khi làm:

| Lỗi | `emu` | `race` | GPU của bạn |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |

## Phương pháp

1. Trình giả lập trước: `python3 labs/pmpp.py run 12.2 --backend emu`. Nó chạy một luồng cho
   đến khi luồng đó gặp một rào chắn (barrier), một lời gọi `_sync` cấp warp hoặc kết thúc, rồi
   đến luồng tiếp theo, dưới ba thứ tự cố định. Hãy biết mỗi loại thất bại trông thế nào ở đó:
   - **Một số đếm sai**: `*outputSize is 3; the sequential filter keeps 12`.
   - **Một lỗ hổng hoặc tràn**: `slot 0 of [0, 1) was never written (it still holds the
     sentinel)`, hoặc `slot 19, at or past the count 19, was written`. Số đếm đã đúng, nên các
     khóa đã đi vào sai chỗ.
   - **Một lần ghi ra ngoài mảng đầu ra.** Trên trình giả lập, mỗi buffer device là đúng một
     khối heap, nên AddressSanitizer dừng lượt chạy ở lần ghi đầu tiên như vậy. Hãy đọc
     `WRITE`, khung đầu tiên (`kernel.cu:` và một số dòng), và địa chỉ nằm sau vùng nhớ bao
     nhiêu byte. Các bài kiểm thử sau điểm dừng không được chạy: sửa lỗi đó, rồi chạy lại.
   - **Một thất bại phụ thuộc vào thứ tự luồng.** Một lỗi thất bại dưới "threads in order" và
     đạt dưới "threads reversed". Trong lịch theo thứ tự, luồng 0 chạy đầu tiên từ mỗi rào chắn
     đến rào chắn tiếp theo; trong lịch đảo ngược nó chạy cuối cùng. Hãy tự hỏi luồng 0 làm gì
     giữa hai rào chắn mà phần còn lại của khối phải làm xong trước.
2. Sửa một chỗ, chạy lại. Các thất bại thay đổi trong quá trình bạn làm.
3. `python3 labs/pmpp.py run 12.2 --backend race` chạy các luồng giả lập thực sự đồng thời dưới
   ThreadSanitizer. Hai luồng chạm vào cùng vùng nhớ mà không đồng bộ hóa được báo cáo kèm cả
   hai lần truy cập và các dòng `kernel.cu:` của chúng, dù kết quả có sai hay không. Một lỗi là
   vô hình với `emu`: không luồng giả lập nào từng bị ngắt giữa lúc đọc, cộng và ghi, nên không
   có gì bị mất ở đó. Một lần truy cập trong cặp có thể nằm bên trong phép toán nguyên tử riêng
   của trình giả lập (`pmpp_emu::atomic_rmw ... fetch_add`): đó là một lời gọi `atomic_ref` của
   bạn, tranh chấp với một lần truy cập thường ở dòng kia. ThreadSanitizer báo mỗi cặp dòng một
   lần mỗi lượt chạy, nên một trường hợp sau báo cáo đầu tiên có thể đạt dù lỗi vẫn còn đó.
4. Rồi đến GPU của bạn: `python3 labs/pmpp.py run 12.2`. Ghi lại những lỗi nó cho bạn thấy, và
   những lỗi nó che đi. Cũng hãy xem những kích thước nào thất bại: một lỗi giữa các warp có thể
   ẩn đi khi toàn bộ đầu vào vừa trong một warp.

Nếu một bản sửa cần phép toán nguyên tử: trên CUDA, dùng `cuda::atomic_ref<unsigned int,
scope>` của sách và `fetch_add(..., cuda::memory_order_relaxed)`, với phạm vi được chọn theo
các luồng dùng chung bộ đếm; trên HIP, dùng `atomicAdd` kiểu cũ bên cạnh nó dưới `#if
defined(__HIPCC__)`, vì ở đó `<cuda/atomic>` đến từ libhipcxx. Bản build HIP của khóa học liên
kết `libhipcxx::libhipcxx` khi CMake tìm thấy nó (nó đi kèm ROCm 10.1 trở lên,
`notes/today/infra-facts.md` B2). Không có nó, mã include `<cuda/atomic>` không build được. Có
nó, chưa máy ROCm nào build mã của khóa học này (unverified). Lời gọi kiểu cũ, được ghi trong
tài liệu cho bộ nhớ chia sẻ và bộ nhớ toàn cục (B1), không cần thêm thư viện nào. Trên Metal,
các hàm atomic chỉ nhận kiểu atomic, nên bản sửa đổi tham số thành `device atomic_uint*` và
dùng `atomic_fetch_add_explicit` (infra-facts C1). Host vẫn cấp phát một bộ đếm 32 bit thường.

## Đọc bảng của bạn

Dù GPU của bạn cho thấy gì, hãy đối chiếu nó với những gì các hãng ghi trong tài liệu, vì đó là
điều mà thiết bị tiếp theo được phép làm:

- Các phép toán nguyên tử trên cùng một địa chỉ được tuần tự hóa, nhưng không theo thứ tự cố định nào
  (`notes/today/ch12-facts.md`, bảng "Already settled elsewhere", trỏ tới ch09 A3 và B1). Đó là
  điều làm Fig. 12.2 không ổn định, và đó là tất cả những gì phép toán nguyên tử hứa. Một chuỗi
  đọc, cộng và ghi thường không hứa gì cả: hai luồng có thể đọc cùng một số đếm.
- Không có rào chắn, không gì nói rằng phần còn lại của khối đã xong một bước. NVIDIA: trên mọi
  GPU hiện nay "independent thread scheduling allows full concurrency between threads,
  regardless of warp" (`notes/today/ch12-facts.md` A1). Apple: các SIMD group của một
  threadgroup "make independent forward progress with respect to each other, in the absence
  of threadgroup barrier operations" (`notes/today/infra-facts.md` C5). Các luồng tình cờ đi
  cùng nhau trên thiết bị của bạn không phải là một lời hứa.
- Không công cụ GPU nào lẽ ra đã chỉ ra bộ đếm toàn cục. `compute-sanitizer --tool racecheck`
  của NVIDIA chỉ kiểm tra bộ nhớ chia sẻ, nên một bộ đếm toàn cục không đồng bộ hóa là vô hình
  với nó, trong khi bộ đếm riêng trong bộ nhớ chia sẻ thì nằm trong tầm của nó; AMD và Apple
  không ghi tài liệu về công cụ phát hiện race nào (ch12-facts, "Already settled elsewhere", trỏ
  tới ch09 A10, B6, C5). Backend `race` là ThreadSanitizer trên các luồng CPU: một lượt chạy
  sạch ở đó cho thấy những lịch chạy đã xảy ra, không phải tất cả.
- Một trong ba lỗi hoàn toàn không phải là race. Nó sai trên mọi backend, mọi lần, và không lịch
  chạy nào che được nó.

Một trường hợp đạt 5 vòng trên GPU của bạn đã cho bạn thấy 10 lần khởi chạy trên một thiết bị.
Nó không cho bạn thấy kernel là đúng.

Không có gì để đo hiệu năng ở đây.
