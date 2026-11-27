> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 12.1: bộ lọc không ổn định, ba cách

Sách: các mục 12.2 đến 12.4 (pp. 290-297), Figs. 12.2, 12.3 và 12.6.

Viết ba bộ lọc không ổn định, không tại chỗ (out-of-place), cho các khóa `unsigned int`. Một
khóa được giữ khi nó nhỏ hơn `threshold`. Cả ba nhận cùng các đối số, `(input, output, n,
threshold, outputSize)`, và để các khóa được giữ trong `output[0, *outputSize)` theo thứ tự bất kỳ.

- `filterAtomicKernel` (Metal: `filter_atomic`), Fig. 12.2. Một luồng (thread) cho mỗi khóa.
  Một luồng giữ khóa của mình thì lấy một vị trí bằng một lần `fetch_add` trên một
  `cuda::atomic_ref` phạm vi device trỏ tới `*outputSize` (relaxed), rồi ghi khóa vào đó.
- `filterWarpAggregatedKernel` (Metal: `filter_warp_aggregated`), Fig. 12.3 ở dạng được ghi
  trong tài liệu. Mọi lane bỏ phiếu bằng `__ballot_sync` trên một mặt nạ đầy đủ kiểu
  `lane_mask_t`; một warp có phiếu bằng 0 thì dừng; lane dẫn đầu là lane giữ khóa thấp nhất
  (`maskFfs(vote) - 1`) và thực hiện một phép toán nguyên tử (atomic) với `maskPopc(vote)`;
  một lệnh shuffle trên mặt nạ đầy đủ trao điểm bắt đầu cho mọi lane; mỗi lane giữ khóa cộng
  thêm số lane giữ khóa nằm dưới nó, `maskPopc(vote & (((lane_mask_t)1 << lane) - 1))`. Lane
  được tính từ `warpSize`.
- `filterPrivatisedKernel` (Metal: `filter_privatised`), Fig. 12.6. Một danh sách riêng gồm
  `BLOCK_DIM` vị trí và một bộ đếm riêng trong bộ nhớ chia sẻ (shared memory), một phép toán
  nguyên tử phạm vi block trên bộ đếm riêng, ba rào chắn (barrier), một phép toán nguyên tử
  phạm vi device cho mỗi khối để giữ chỗ dải của nó trong danh sách chung, và một bước commit
  trong đó luồng k sao chép mục riêng k.

Tệp và lệnh:

- Sửa `work/kernel.cu` (ba kernel và các hàm khởi chạy của chúng `launchFilterAtomic`,
  `launchFilterWarpAggregated`, `launchFilterPrivatised`), `work/kernel.metal` (cùng ba kernel
  đó trong Metal) và `work/launch.mm` (phần dispatch).
- Bên gọi đặt `*outputSize` về 0 trước mỗi lần gọi. Các kernel không làm việc đó.
- `BLOCK_DIM` được định nghĩa bởi bản build (mặc định 256): một khóa cho mỗi luồng, `BLOCK_DIM`
  luồng mỗi khối. Trên Metal một threadgroup có `BLOCK_DIM` luồng, hoặc ít hơn nếu pipeline
  không chạy được nhiều đến thế.
- `lane_mask_t`, `maskPopc` và `maskFfs` đã được viết sẵn cho bạn ở đầu `work/kernel.cu`. Hãy
  đọc lý do trước khi dùng chúng.
- `provided/copy_keys.cu`, `.metal` và `.mm` (`launchCopyKeys`, `copy_keys`,
  `encodeCopyKeys`) sao chép n khóa: mốc tham chiếu băng thông của benchmark. Bạn không sửa
  chúng; chúng cũng được kiểm thử.
- Kiểm thử: `python3 labs/pmpp.py run 12.1`. Bắt đầu với `--backend emu`, rồi `--backend race`.
- Đo (chỉ GPU): `python3 labs/pmpp.py run 12.1 --bench`.

## Khác biệt so với trang sách

- **Vị từ.** `cond(val)` của sách là một chỗ giữ chỗ mà sách không bao giờ định nghĩa. Ở đây
  nó là `val < threshold`, với `threshold` là một đối số của kernel, để các bài kiểm thử và
  benchmark có thể chọn bao nhiêu khóa được giữ.
- **Một mặt nạ có kiểu.** Fig. 12.3 giữ mặt nạ trong một `unsigned int`, dựng mặt nạ các lane
  phía dưới là `(1 << laneIdx()) - 1`, và gọi `__popc` và `__ffs`. Điều đó đúng với warp 32
  lane. Trên HIP nó không biên dịch được, và cách sửa một dòng hiển nhiên thì biên dịch được
  nhưng sai: mọi intrinsic `_sync` ở đó đòi mặt nạ 64 bit, `__ballot_sync` trả về
  `unsigned long long`, `__popc` và `__ffs` nhận 32 bit, và trên một wavefront 64 lane
  `1 << lane` là không xác định với các lane 32 đến 63 (`notes/today/ch12-facts.md` B1). Vì
  vậy mặt nạ có kiểu riêng, `lane_mask_t` (`unsigned` trên CUDA và các trình giả lập,
  `unsigned long long` dưới `__HIPCC__`), các phép đếm bit đi qua các hàm trợ giúp có kiểu mà
  template bị xóa của chúng biến một kiểu đối số sai thành lỗi biên dịch (cách làm của
  llama.cpp, D1), và phép dịch là `(lane_mask_t)1 << lane`.
- **Một ballot thay cho `__activemask`.** Fig. 12.3 gọi `__activemask()` bên trong nhánh và
  coi kết quả là tập các lane đã giữ một khóa. Hướng dẫn của NVIDIA giờ nói `__activemask()`
  "cannot be used to determine which warp lanes execute a given branch": nó là ảnh chụp các
  lane tình cờ đang hội tụ (A1). Kernel của sách vẫn đúng, vì mỗi ảnh chụp giữ chỗ dải riêng
  của nó và lệnh shuffle `_sync` hội tụ lại nó, nhưng "một phép toán nguyên tử mỗi warp" thì
  không được bảo đảm. Một lần bỏ phiếu mà mọi lane đều tham gia, bên ngoài mọi nhánh, cho ra
  đúng các lane giữ khóa, nên ở đây không luồng nào return trước lần bỏ phiếu, các luồng nằm
  ngoài phạm vi bỏ phiếu 0, và mặt nạ của lệnh shuffle là mặt nạ đầy đủ vì mọi lane đều thực
  thi nó. Header của AMD nói `__activemask()` của họ là chính xác (B1); Apple không nói gì về
  chuyện này (C3), đó là lý do kernel Metal cũng gọi các hàm SIMD group từ mọi luồng.
- **Không có cooperative group.** Coalesced group của Fig. 12.4, và `num_threads()` của nó
  (tên hiện nay trong CUDA của `size()` trong sách, A5), không cần ở đây: ballot cho ra cùng
  lane dẫn đầu, cùng số đếm và cùng thứ hạng, và đó là dạng cũng build được cho HIP.
- **Bộ đếm được bên gọi đặt về 0** cho cả ba kernel. Sách chỉ nói vậy cho Fig. 12.2 (p. 291).
- **Phép toán nguyên tử trên HIP.** Trên HIP, `<cuda/atomic>` đến từ libhipcxx của AMD. Bản
  build HIP của khóa học liên kết `libhipcxx::libhipcxx` khi CMake tìm thấy nó (nó đi kèm ROCm
  10.1 trở lên, `notes/today/infra-facts.md` B2). Không có nó, mã include `<cuda/atomic>`
  không build được. Có nó, chưa máy ROCm nào build mã của khóa học này (unverified). Vì vậy lab
  không dựa vào nó: bên cạnh mỗi `cuda::atomic_ref`, hãy viết `atomicAdd` kiểu cũ dưới
  `#if defined(__HIPCC__)`, thứ mà HIP ghi trong tài liệu là relaxed và hợp lệ trên bộ nhớ
  chia sẻ lẫn bộ nhớ toàn cục (B1) và không cần thêm thư viện nào. Lời giải tham chiếu
  (solution) cho thấy mẫu này.

Trên Metal, các hàm atomic chỉ nhận kiểu atomic, nên bộ đếm là `device atomic_uint*` và bộ đếm
riêng là một `threadgroup atomic_uint`; không có phạm vi nào để chọn
(`notes/today/infra-facts.md` C1). Các bước cấp warp là bốn hàm SIMD group, được liệt kê trong
`work/kernel.metal` (ch12-facts C1). Hãy dispatch bằng `dispatchThreadgroups`, để các luồng
nằm quá n tồn tại và tham gia vào chúng.

## Các bài kiểm thử làm gì

Một bộ lọc không ổn định có thể đặt các khóa được giữ theo thứ tự bất kỳ, nên chương trình điều
khiển (driver) không so sánh mảng. Với mỗi đầu vào, nó kiểm tra rằng `*outputSize` bằng số khóa
mà một bộ lọc tuần tự giữ; rằng `output[0, count)`, sau khi sắp xếp, bằng danh sách của bộ lọc
tuần tự, sau khi sắp xếp (không mất gì, không ghi gì hai lần); và rằng mọi vị trí từ `count` đến
`n` vẫn giữ giá trị canh `0xFFFFFFFF` mà nó đã ghi trước lần gọi (các khóa nhỏ hơn 2^24, nên giá
trị canh không bao giờ là một khóa). Rồi nó đặt `*outputSize` về 0 lần nữa, gọi kernel lần thứ
hai và kiểm tra cùng ba điều đó.

- Kích thước: 1, 31, 32, 33, 255, 256, 257, 1000, 4099; trên GPU thêm 65537, 2^20 và
  3 x 2^20 + 17. Backend `race` dừng ở 1000: nó chạy mỗi luồng của một khối như một luồng của
  hệ điều hành.
- Các khóa phân bố đều trong [0, 2^24) với tỉ lệ được giữ 0, 0.001, 0.03, 0.5, 0.97 và 1.
- Các mẫu: mọi khóa, không khóa nào, chỉ khóa cuối cùng, chỉ khóa 31 trong mỗi 32, chỉ các khóa
  32 đến 63 trong mỗi 64, cứ một khóa giữ một khóa bỏ, các đoạn 40 khóa giữ và 40 khóa bỏ vắt
  ngang các warp và các khối. Mẫu "32 đến 63" đặt mọi khóa được giữ ở chỗ mà mặt nạ 32 bit
  không thấy được trên wavefront 64 lane; trên các trình giả lập, Metal và NVIDIA nó chỉ là
  thêm một mẫu.

Mỗi backend có thể cho bạn thấy gì:

- `emu` chạy từng luồng một cho đến khi nó gặp một rào chắn, một lời gọi `_sync` cấp warp hoặc
  kết thúc. Ở đó một rào chắn bị thiếu sẽ sai dưới ít nhất một trong ba thứ tự luồng của nó, và
  AddressSanitizer dừng ở một lần ghi ngoài biên và chỉ ra dòng. Một `++` thường trên bộ đếm
  không bao giờ sai ở đó: không ai bị ngắt giữa chừng.
- `race` chạy các luồng thực sự đồng thời dưới ThreadSanitizer và báo cáo hai luồng chạm vào
  cùng vùng nhớ mà không đồng bộ hóa, kèm cả hai dòng mã nguồn.
- GPU của bạn chỉ cho thấy khóa bị mất hoặc bị nhân đôi khi lịch chạy tình cờ tạo ra chúng.

## Trước khi bạn đo hiệu năng

`--bench` tăng n từ 2^22 cho đến khi kernel sao chép được cung cấp mất ít nhất 3 ms (tối đa 2^26 khóa,
và đầu vào cộng đầu ra tối đa một phần tư bộ nhớ mà thiết bị báo cáo). Rồi nó đo thời gian ba bộ
lọc trên các khóa phân bố đều với tỉ lệ được giữ 0, 0.001, 0.01, 0.03, 0.1, 0.25, 0.5, 0.75 và
1, lấy kết quả tốt nhất trong 5 lần chạy mỗi cấu hình, và kiểm tra chính xác mọi kết quả trước
khi báo cáo. Nó in ra mili giây và GB/s của lưu lượng tối thiểu mà một bộ lọc cần (đọc 4n byte,
ghi 4 byte cho mỗi khóa được giữ, 4 byte cho bộ đếm); 8n byte mỗi lượt của kernel sao chép là cùng con
số đó cho một bộ lọc giữ mọi thứ. Việc đặt `*outputSize` về 0 không được tính giờ.

Hãy ghi ra trước:

1. Ở tỉ lệ được giữ nào thì kernel atomic thường sẽ tụt sau kernel riêng tư hóa
   (privatization) xa nhất, một số từ 0 đến 1 (trang labs ghi nó là `ch12/labs/p-lab1-worst`).
2. Thời gian của kernel atomic thường chia cho thời gian của kernel gộp theo warp ở tỉ lệ 0.5
   (`ch12/labs/p-lab1-agg`).

Mô hình của sách (mục 12.2, p. 291) là mỗi khóa được giữ là một phép toán nguyên tử trên cùng
một bộ đếm, bị tuần tự hóa, nên tranh chấp tăng theo tỉ lệ được giữ, còn kernel riêng tư hóa
thực hiện một phép toán nguyên tử công khai mỗi khối bất kể tỉ lệ. Ngược lại, số byte di chuyển
cũng tăng theo tỉ lệ, và lượng việc của các phép toán nguyên tử riêng cũng vậy.

Một giá trị gần 1 cho dự đoán thứ hai nghĩa là trình biên dịch của bạn đã tự gộp phép toán
nguyên tử thường, để Fig. 12.2 chạy như Fig. 12.3. Sách nói trình biên dịch làm việc này (p.
294). NVIDIA chỉ nói vậy trong một bài blog cập nhật năm 2017, "in many cases", không kèm điều
kiện nào (`notes/today/ch12-facts.md` A7). Backend LLVM của AMD có một pass viết lại một phép
toán nguyên tử cùng địa chỉ thành đúng chuỗi ballot, thứ hạng, lane dẫn đầu và phát tán của Fig.
12.3, bật mặc định từ `-O1` trở lên (B4; đọc trong LLVM upstream, chưa xác nhận cho `amdclang++`
của ROCm 10.2). Apple không ghi tài liệu gì về chuyện này. Chỉ phép đo trên thiết bị của bạn mới
quyết định, và chỉ mã đã biên dịch (PTX hoặc SASS, bản disassembly AMDGPU) mới cho thấy trình
biên dịch đã làm gì.

Các con số là phép đo trên thiết bị của bạn, không phải của GPU nói chung.
