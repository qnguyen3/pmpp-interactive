> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 11.1: một scan khối, ba cách

Sách: các mục 11.2 đến 11.4 (pp. 254-265), Figs. 11.3, 11.5 và 11.8 đến 11.10.

Bạn viết hai scan bao gồm (inclusive scan) ở cấp khối và đo thời gian của chúng so với kernel
hai rào chắn (barrier) của sách (Fig. 11.3), vốn được cung cấp sẵn. Scan khối bạn viết ở (b) là
khối xây dựng mà các lab 11.2, 11.3 và 11.6 dùng lại.

Mọi kernel có cùng một hợp đồng. Khối b scan đoạn `blockDim.x` phần tử của riêng nó, bao gồm,
với phép +, và không biết gì về các khối khác. n có thể là bất kỳ giá trị nào từ 1 trở lên. Một
luồng (thread) nằm quá cuối mảng nạp 0, phần tử đơn vị của +, tham gia mọi rào chắn và mọi lần
shuffle, và không ghi gì. Kích thước khối là một lũy thừa của hai từ 64 đến 1024, được chọn lúc
chạy, nên không có `SEG_SIZE`: hãy dùng `blockDim.x`, và đặt kích thước các mảng chia sẻ theo
khối lớn nhất.

**Phần việc của bạn:**

- **(a) `KoggeStoneDoubleBufferKernel`** (Fig. 11.5). Bắt đầu từ Fig. 11.3 được cung cấp và bỏ
  rào chắn thứ hai của nó bằng hai buffer chia sẻ đổi vai cho nhau ở mỗi vòng lặp. Như vậy còn
  lại một rào chắn mỗi vòng lặp. Các luồng không cộng vẫn phải chép giá trị của mình sang, và
  kết quả được đọc từ `inBuffer_s` sau lần đổi cuối cùng.
- **(b) `warpScan`, `blockScan` và `WarpScanKernel`** (Figs. 11.8 đến 11.10), viết cho khả
  chuyển:
  - Mặt nạ lane có kiểu riêng: `unsigned int` trên CUDA, 64 bit trên HIP. Không bao giờ dùng
    hằng `0xffffffff`.
  - Vòng lặp shuffle chạy khi `stride < warpSize`. `laneIdx()` và `warpIdx()` được tính từ
    `threadIdx.x` và `warpSize`, không từ một hằng số 32.
  - Mảng tổng của các warp được khai báo trong kernel, với kích thước theo warp hẹp nhất được
    ghi trong tài liệu (`1024/32` phần tử), và được truyền cho `blockScan` dưới dạng con trỏ.
    Fig. 11.9 khai báo nó bên trong `blockScan`, điều mà Metal từ chối và HIP không ghi trong
    tài liệu.
  - Lần đọc ở giai đoạn 2 được bảo vệ. Các lane không có warp nào phía sau thì đưa vào 0.
  - Mọi luồng đều gọi `blockScan`, dù nằm trong phạm vi hay không.
- **Chỉ Metal, (b2) `warp_scan_prefix`**: cùng scan khối đó nhưng mỗi scan SIMD group được làm
  bằng một lần gọi `simd_prefix_inclusive_sum`, để bạn so sánh hàm một lần gọi với vòng lặp
  của mình.

## Tệp

- Sửa `work/kernel.cu` (CUDA, HIP và cả hai trình giả lập): `KoggeStoneDoubleBufferKernel`,
  `warpScan`, `blockScan`, `WarpScanKernel`, `launchKoggeStoneDB` và `launchWarpScan`.
- Sửa `work/kernel.metal` (`ks_double_buffer`, `warp_scan`, `block_scan_shuffle`,
  `warp_scan_shuffle`, `block_scan_prefix`, `warp_scan_prefix`) và `work/launch.mm`
  (`encodeKoggeStoneDB`, `encodeWarpScanShuffle`, `encodeWarpScanPrefix`). Mỗi hàm encode trả
  về false, không dispatch, cho một khối mà pipeline của nó không chạy được: nhiều luồng hơn
  `maxTotalThreadsPerThreadgroup` của pipeline, hoặc, với hai kernel SIMD group, một khối
  không phải là một số nguyên SIMD group gồm `threadExecutionWidth` lane, hoặc có nhiều SIMD
  group hơn số lane của một SIMD group.
- Được cung cấp, không sửa, trong `../shared-g11/`: `bs_two_barrier.cu` và `.metal` (Fig. 11.3,
  mốc so sánh), `bs_copy.cu` và `.metal` (một phép sao chép đơn giản, làm mốc băng thông), và
  `bs_launch.mm`.

## Lệnh

- Kiểm thử: chạy `python3 labs/pmpp.py run 11.1 --backend emu` trước, rồi `python3 labs/pmpp.py run 11.1`.
- Rào chắn: `python3 labs/pmpp.py run 11.1 --backend race`.
- Đo: `python3 labs/pmpp.py run 11.1 --bench` (chỉ trên thiết bị thật).
- Lời giải tham chiếu: thêm `--solution`.

## Các bài kiểm thử làm gì

- n = 1, 31, 32, 33, 255, 256, 257, 1000, 4097, 100003 và 2^22, với khối 64, 128, 256 và 1024.
  Trình giả lập chạy n tới 4097 với khối 64 và 256. `race` chạy n = 33 và 1000 với khối 64,
  tức là 1 và 16 khối, nên nhiều khối chạy cùng lúc. Trên Metal, các kernel SIMD group chạy với
  những khối mà pipeline cho phép, và chương trình điều khiển (driver) in ra những khối nó bỏ
  qua.
- Ba loại dữ liệu. Với toàn số một, phần tử i phải thành `i mod block + 1`. Các số nguyên từ -3
  đến 3 có tổng tiền tố chính xác theo mọi thứ tự cộng, nên chúng cũng phải khớp chính xác.
  Các giá trị phân bố đều trong [-1, 1] được so với vòng lặp của Fig. 11.1 chạy bằng double
  trên từng đoạn, trong phạm vi 1e-6 nhân tổng |x| của đoạn tính đến phần tử đó, cộng 1e-6.
- Đầu ra khởi đầu là NaN, nên một phần tử mà kernel của bạn không bao giờ ghi sẽ bị báo. Các
  từ canh gác nằm sau đầu ra tới cuối khối cuối cùng và thêm 16 nữa. Một luồng nằm quá n mà
  ghi sẽ bị báo.
- Các kernel được cung cấp cũng được kiểm thử, nên một tệp cung cấp bị hỏng sẽ lộ ra.

Mỗi backend có thể bắt được gì:

- `emu` lấp bộ nhớ chia sẻ (shared memory) bằng rác cho mọi khối và chạy các luồng lần lượt
  từng luồng theo ba thứ tự cố định. Thiếu nhánh chép, thiếu rào chắn ở (a), và dùng
  `threadIdx.x` thay cho lane trong `warpScan`, mỗi lỗi đều thất bại ở đó trong mọi lần chạy.
- `race` chạy các luồng của một khối, và nhiều khối, cùng lúc dưới ThreadSanitizer. Một rào
  chắn bị thiếu được báo cùng cả hai dòng mã nguồn.
- Một lần đọc giai đoạn 2 không được bảo vệ vẫn qua trên mọi backend ở đây. Trong một scan,
  rác mà một lane thừa đọc chỉ chảy vào các lane sau nó, và không ai đọc chúng. Điều mà phần
  bảo vệ ngăn chặn là một lần đọc quá cuối mảng. Trên một wavefront 64 lane, các lane 32 đến
  63 sẽ đọc quá 32 phần tử (lập luận của chúng tôi, từ ch11 facts B1). Hãy giữ nó.
- Không gì trên máy Mac bắt được hai lỗi HIP. Cả hai trình giả lập đều mô hình hóa warp 32
  lane, và Metal không có mặt nạ. Vì vậy hãy kiểm tra hai lỗi này bằng cách đọc mã của bạn:
  - Một mặt nạ `0xffffffff`. HIP từ chối nó lúc biên dịch, vì các hàm `_sync` của nó
    `static_assert` một mặt nạ 64 bit (ch11 facts B1, infra facts B3).
  - Một hằng 32 cho độ rộng warp. Trên một wavefront 64 lane, một vòng lặp dừng ở bước nhảy
    (stride) 16 sẽ scan mỗi nửa 32 lane riêng rẽ. Kết quả chỉ đúng nếu mọi phép đếm khác (lane,
    warp, phép kiểm tra lane cuối) cũng dùng 32. Trộn lẫn hai cách thì sai (ch11 facts B1).

## Chỗ các backend khác nhau (không có trong sách)

- **HIP**: kiểu mặt nạ ở trên. `warpSize` là 64 trên gfx9 (Instinct) và 32 trên các dòng AMD
  sau này, và từ ROCm 7.0 nó không phải là hằng số lúc biên dịch (ch05 facts B5, ch10 facts
  B2). Các lane dưới bước nhảy nhận lại giá trị của chính mình từ `__shfl_up_sync`, như trên
  CUDA (ch11 facts A3, B1). Không phần nào trong số này đã được build hay chạy với ROCm ở đây:
  unverified.
- **Metal**: không có mặt nạ; `simd_shuffle_up(v, delta)` để nguyên `delta` lane thấp nhất,
  cùng quy tắc như CUDA. Metal không ghi trong tài liệu cách một threadgroup được chia thành
  các SIMD group, nên lane, chỉ số SIMD group và độ rộng lấy từ
  `[[thread_index_in_simdgroup]]`, `[[simdgroup_index_in_threadgroup]]` và
  `[[threads_per_simdgroup]]` (infra facts C5). Một hàm phụ trợ không thể khai báo bộ nhớ
  threadgroup: trình biên dịch báo "variables in the threadgroup address space cannot be
  declared in a non-qualified function" (ch11 facts C2).
- `simd_prefix_inclusive_sum` cộng trên các luồng *đang hoạt động* của SIMD group. Một luồng đã
  return sớm sẽ không cộng cũng không được đếm, thêm một lý do để các luồng nằm quá n ở lại
  trong kernel và đưa vào 0. Apple không ghi trong tài liệu thứ tự các phép cộng của nó, nên
  các bit cuối của nó có thể khác vòng lặp của bạn (ch11 facts C1).
- CUDA không có scan warp một lần gọi cho `float`. Các đường thư viện là `cub::WarpScan` và
  `cooperative_groups::inclusive_scan` trên một tile (ch11 facts A4).

## Trước khi đo hiệu năng, hãy dự đoán

Hãy ghi những điều này ra trước. Bài học hỏi chúng dưới các id trong ngoặc.

1. Với khối 1024, scan warp-shuffle của bạn sẽ chạy nhanh hơn kernel hai rào chắn bao nhiêu
   lần (`ch11/labs/p-lab1-warp`).
2. Riêng việc dùng buffer kép sẽ nhanh hơn kernel hai rào chắn bao nhiêu lần
   (`ch11/11.3/p-double`, được hỏi ở mục 11.3).
3. Trên Metal: vòng lặp shuffle của bạn sẽ nhanh hơn, chậm hơn hay bằng phiên bản một lần gọi
   `simd_prefix_inclusive_sum`? Chênh bao nhiêu?

Sách đếm rào chắn: hai mỗi vòng lặp ở Fig. 11.3, một mỗi vòng lặp ở Fig. 11.5, tổng cộng hai
cho Fig. 11.9 (pp. 258-265). Sách cũng nói một scan di chuyển ít nhất 8 byte mỗi phần tử, và
ngay cả một phép sao chép cũng không đạt băng thông công bố (pp. 272-273). Bao nhiêu phần của số rào
chắn trở thành thời gian thì tùy thiết bị của bạn.

## Đọc kết quả

Benchmark scan n = 2^24 số float (ít hơn nếu thiết bị của bạn báo ít hơn bốn lần 128 MiB đầu
vào và đầu ra cần thiết), mỗi khối một đoạn, với khối 256, 512 và 1024. Với mỗi kích thước khối,
nó đo kernel chép trước, rồi kernel hai rào chắn, (a) của bạn, (b) của bạn và, trên Metal, (b2).

- `time ms` là lần tốt nhất trong năm lần chạy sau một lần khởi động không đo. Một phần tư
  giây các phép sao chép không đo chạy trước, để kernel được đo đầu tiên không phải trả thêm chi
  phí đánh thức thiết bị.
- `GB/s` tính 8 byte mỗi phần tử (một lần đọc 4 byte, một lần ghi 4 byte) chia cho thời gian
  đó: số byte một scan phải di chuyển, không phải số byte hệ thống bộ nhớ đã di chuyển.
- `of copy` là tốc độ của kernel tính theo tỉ lệ so với kernel chép ở cùng kích thước khối.
  Phép sao chép là mức cao nhất máy này đã cho một kernel di chuyển cùng số byte, nên nó là thước
  đo ở đây, không phải đỉnh công bố.

Cùng các hàng đó được đưa vào các trang bài học (mục 11.3 và 11.4 và trang labs). Các con số là
phép đo của máy này ở kích thước này. Chúng không phải là thuộc tính của các kernel.

## Đi xa hơn

- Làm scan warp tự lấp phần tử đơn vị trên Metal: `simd_shuffle_and_fill_up(val, 0.0f, d)` chép
  phần tử đơn vị vào các lane thấp, nên phép kiểm tra lane biến mất. Nó cần Apple8 trở lên, nên
  không chạy trên M1 (ch11 facts C1). MLX dùng nó cho các kiểu 8 byte (ch11 facts D3).
- Biến scan bao gồm thành scan loại trừ, như top-p sampling cần (mục 11.1): dịch một lane bằng
  thêm một lần shuffle, hoặc trừ đi đầu vào. Một trong hai cách trả về đúng những gì scan bao
  gồm của bạn đã tính; cách kia thêm một sai số làm tròn của riêng nó. Cách nào, và vì sao điều
  đó quan trọng ở ranh giới top-p?
