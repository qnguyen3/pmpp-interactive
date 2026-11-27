> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 12.3: bộ lọc ổn định bằng scan

Sách: mục 12.5 (pp. 297-298, Figs. 12.7 và 12.8), mục 12.6 (pp. 298-300, Figs. 12.9 và
12.10) và các bài tập 1, 3 và 4 (p. 302). Một mục đứng riêng như (A1) hay (D4) dưới đây nằm
trong `notes/today/ch12-facts.md`.

Viết hai bộ lọc ổn định, không tại chỗ (out-of-place), cho các khóa `unsigned int`. Một khóa
được giữ khi `key < threshold`. Các khóa được giữ ra theo thứ tự đầu vào, xếp liền từ
`output[0]`, và `*outputSize` nhận số lượng của chúng. Bên gọi bảo đảm `n >= 1`: không có khóa
nào thì không có khóa cuối để ghi số đếm, và các hàm khởi chạy sẽ khởi chạy không khối nào, điều
HIP từ chối (`notes/today/ch02-facts.md` B4), nên không hàm khởi chạy nào phải xử lý trường hợp
đó. Không bộ lọc nào để một khối chờ khối khác, nên cả hai chạy trên CUDA, HIP và Metal. (Fig.
12.8 gọi một phép scan toàn lưới bên trong một kernel, điều cần các khối chờ những khối trước
đó; chỉ CUDA ghi trong tài liệu rằng việc đó an toàn, A10, B6, C4. Lab 12.5 xây dựng phiên bản
đó.)

**Phần A, bài tập 1: ba bước.**

- `keepFlagsKernel`: `keep[i]` bằng 1 với một khóa được giữ, 0 nếu không.
- Phép scan được cung cấp: `offset` = scan loại trừ (exclusive scan) của `keep`.
- `scatterKernel`: một khóa được giữ đi tới `output[offset[i]]`, và luồng của khóa cuối cùng ghi
  `*outputSize = offset[n-1] + keep[n-1]`.
- `launchStableFilterFlags(input, output, n, threshold, outputSize, keep, offset, scanScratch)`
  khởi chạy ba kernel theo thứ tự.

**Phần B, các bài tập 3 và 4 ở dạng rút gọn rồi scan (reduce-then-scan).** Khối `b` sở hữu một
ô (tile) gồm `256 * coarse` khóa và đi qua nó trong `coarse` vòng, mỗi vòng 256 khóa liên tiếp,
nên các lần nạp của mỗi vòng đều được hợp nhất (coalescing).

- `countKeptKernel`: ở mỗi vòng, mỗi warp đếm các khóa được giữ của nó bằng một lần bỏ phiếu và
  một phép popcount; tổng của các warp được cộng trong bộ nhớ chia sẻ (shared memory); một luồng
  ghi `blockCounts[b]`.
- Phép scan được cung cấp biến `blockCounts` thành `blockOffsets`.
- `stagedScatterKernel`: khối đọc lại ô của nó, từng vòng một. Vị trí của mỗi khóa được giữ
  trong một buffer trung gian ở bộ nhớ chia sẻ là (số khóa được giữ của các vòng trước) + (số
  khóa được giữ của các warp trước trong vòng này) + (số lane được giữ nằm dưới nó trong warp
  của nó). Số hạng cuối là tổng tiền tố nhị phân của p. 293: phiếu bầu, AND với mặt nạ các lane
  phía dưới, rồi popcount. Sau các vòng, các luồng liên tiếp sao chép đoạn đã đệm ra
  `output[blockOffsets[b] + j]`. Khối cuối cùng ghi `*outputSize`. Buffer trung gian là
  `extern __shared__`, `256 * coarse * 4` byte, 16 KiB ở coarse 16.
- `launchStableFilterStaged(input, output, n, threshold, outputSize, blockCounts,
  blockOffsets, scanScratch, coarse)`.

Phần B có hình dạng của `torch.nonzero_static` trong PyTorch trên CUDA (đếm mỗi khối, scan các
số đếm, ghi rải, D4) và của `nonzero` trên MPS (ba lần dispatch, D5).

Quy tắc cho các lần bỏ phiếu trong mã CUDA của phần B:

- Bỏ phiếu bằng `__ballot_sync` trên toàn warp, được gọi bởi mọi lane, được giữ hay không, quá
  cuối mảng hay không. Trong luồng điều khiển đồng nhất đó, kết quả chính xác là tập các lane
  được giữ. Đừng gọi `__activemask()` bên trong nhánh như Fig. 12.3: NVIDIA giờ cảnh báo rằng nó
  "cannot be used to determine which warp lanes execute a given branch" (A1).
- Dùng `lane_mask_t` được cung cấp cho mọi mặt nạ và `maskPopc` để đếm một mặt nạ. Trên HIP mặt
  nạ là 64 bit và `__popc` sẽ âm thầm bỏ nửa trên (B1). Không bao giờ viết `0xffffffff`.
- Lane và warp được tính từ `threadIdx.x` và `warpSize`, không bao giờ từ một hằng số 32.

Trên Metal, bỏ phiếu và popcount trở thành `simd_sum(keep ? 1u : 0u)` và thứ hạng trở thành
`simd_prefix_exclusive_sum(keep ? 1u : 0u)` (C1), được gọi bởi mọi luồng.

## Tệp

- Sửa `work/kernel.cu` (CUDA, HIP và cả hai trình giả lập).
- Sửa `work/kernel.metal` (`keep_flags`, `scatter_flags`, `count_kept`, `staged_scatter`; chữ
  ký đã được cho sẵn) và `work/launch.mm` (`encodeStableFilterFlags`,
  `encodeStableFilterStaged`). Mỗi encoder đặt vài lần dispatch, với phép scan được cung cấp ở
  giữa, vào một compute encoder duy nhất mà nó nhận. Một encoder tạo bằng
  `computeCommandEncoder` dạng thông thường sẽ chạy các lần dispatch theo thứ tự và mỗi lần thấy các lần ghi
  của lần trước (`notes/today/ch02-facts.md` C9): bản tương ứng trong Metal của ba lần khởi chạy
  trên một stream CUDA. Hãy đọc độ rộng SIMD group từ `threadExecutionWidth` của pipeline, đặt
  kích thước bộ nhớ threadgroup bằng `setThreadgroupMemoryLength:atIndex:` (bội của 16 byte,
  `ch05-facts.md` C8), và trả về false nếu nó không vừa `maxThreadgroupMemoryLength` của thiết
  bị.
- Được cung cấp, không sửa: `labs/ch12/shared-g12/scan_u32.cu`, `scan_u32.metal` và
  `scan_u32_launch.mm`. Một scan loại trừ phân cấp trên `uint32`: mỗi khối scan một ô 1024 phần
  tử, tổng của các ô được scan theo cùng cách, và một lượt cộng hoàn tất (scan-scan-add, mục
  11.9, Fig. 11.14, pp. 273-274). Không khối nào chờ khối khác. `scanScratchElems(n)` cho kích
  thước vùng nháp của nó; chương trình điều khiển (driver) cấp phát vùng nháp. Cùng các tệp đó
  chứa kernel sao chép mà benchmark dùng để so sánh.

## Lệnh

- Kiểm thử: `python3 labs/pmpp.py run 12.3 --backend emu` trước (vài phút: mỗi luồng GPU chạy
  trên CPU), rồi `python3 labs/pmpp.py run 12.3`.
- Race condition trên bộ nhớ chia sẻ: `python3 labs/pmpp.py run 12.3 --backend race`. Buffer
  trung gian và tổng của mỗi warp được ghi lại ở mỗi vòng; một rào chắn (barrier) bị thiếu hiện
  ra ở đây kèm cả hai dòng mã nguồn, ngay cả khi đáp án tình cờ ra đúng.
- Đo: `python3 labs/pmpp.py run 12.3 --bench` (chỉ trên thiết bị thật).
- Lời giải tham chiếu (solution): thêm `--solution`.

## Các bài kiểm thử kiểm tra gì

- Riêng phép scan được cung cấp, so với một scan loại trừ trên CPU, không tại chỗ và một lần tại
  chỗ.
- n = 1, 31, 32, 33, 255, 256, 257, 1000, 4099, 65537, 2^20 và 3 x 2^20 + 17 trên GPU. Trình
  giả lập chạy phần A tới 4099 cộng 65537 (chỉ ở mức giữ một nửa) và phần B tới 4099; backend
  race chạy n tới 2048.
- Phần A, và phần B ở coarse 1, 2, 4, 8 và 16. Coarse 16 chỉ chạy ở nơi buffer trung gian của
  nó vừa bộ nhớ chia sẻ của thiết bị; nếu không, chương trình điều khiển in ra rằng nó đã bỏ qua.
- Tỉ lệ được giữ 0, 0.001, 0.5, 0.97 và 1, cùng năm mẫu: chỉ khóa đầu tiên, chỉ khóa cuối
  cùng, cứ một khóa giữ một khóa bỏ, các đoạn 12 khóa vắt qua mọi ranh giới ô, và một đoạn lấp
  đúng một ô.
- Các khóa được giữ đều khác nhau và không theo thứ tự tăng, nên một khóa sai chỗ sẽ bị bắt. Số
  đếm phải chính xác, `output[0..count)` phải bằng bộ lọc ổn định trên CPU từng phần tử một, và
  mọi vị trí quá số đếm, kể cả 64 vị trí quá cuối mảng, phải vẫn giữ giá trị canh mà chương
  trình điều khiển đã ghi. Chương trình điều khiển điền rác vào `keep`, `offset`,
  `blockCounts`, `blockOffsets` và vùng nháp của phép scan trước mỗi lượt chạy.

## Trước khi bạn đo hiệu năng

Hãy ghi ra hai dự đoán (trang labs ghi chúng là `ch12/labs/p-lab3-staged` và
`ch12/labs/p-lab3-coarse`):

1. Ở coarse 4 và giữ một nửa số khóa, thời gian của phần A chia cho thời gian của phần B.
2. Hệ số gộp việc cho luồng (thread coarsening) nào sẽ nhanh nhất ở mức giữ một nửa.

Mô hình là phép đếm byte trên mỗi khóa của mục 12.5, ở tỉ lệ được giữ f:

- Các mảng có tên của phần A: đầu vào được đọc hai lần (8), `keep` được ghi một lần và đọc hai
  lần (12), `offset` được ghi một lần và đọc một lần (8), các khóa được giữ được ghi (4f). 28 +
  4f, tức 30 byte mỗi khóa ở mức giữ một nửa.
- Phần B: đầu vào được đọc hai lần (8) và các khóa được giữ được ghi (4f); số đếm của mỗi khối
  chỉ là vài byte cho mỗi nghìn khóa. 8 + 4f, khoảng 10 ở mức giữ một nửa.

Mô hình tính phép scan 8 byte mỗi phần tử (đọc `keep`, ghi `offset`). Phép scan được cung cấp
thực sự di chuyển 16 (lượt cộng của nó đọc và ghi `offset` lần nữa, p. 274), nên lưu lượng thật
của phần A cao hơn mô hình của nó khoảng 8 byte mỗi khóa. Bộ nhớ đệm (cache) quyết định phần còn
lại: lần đọc thứ hai một ô của phần B có thể đến từ bộ nhớ đệm, còn các mảng của phần A thì có
thể không. Thiết bị của bạn quyết định.

Benchmark bắt đầu ở n = 2^24 và nhân đôi n cho đến khi một lần sao chép thường của đầu vào mất 3 ms
(trong giới hạn một phần tám bộ nhớ của thiết bị). Với các tỉ lệ được giữ 0.001, 0.01, 0.1,
0.5, 0.9 và 1, nó đo thời gian phần A ("flags") và phần B ("staged") ở mọi hệ số gộp việc vừa
bộ nhớ. Mỗi hàng cho thời gian (tốt nhất trong 5 lần chạy), GB/s trên lưu lượng tối thiểu
`4n + 4 x kept + 4` byte, số byte mỗi khóa theo mô hình, và "at copy speed": kernel sao chép lẽ
ra đã di chuyển bao nhiêu byte mỗi khóa trong cùng thời gian đó. Hai cột cuối là phép so sánh
cần làm: con số của mô hình so với những gì thiết bị của bạn thực tính. Mọi kết quả được đo giờ
đều được kiểm tra trước khi báo cáo. Các con số là phép đo trên máy của bạn, không phải mục
tiêu.

## Tính khả chuyển

- HIP: các lần bỏ phiếu dùng `lane_mask_t` 64 bit và `__popcll` qua `maskPopc`, và lane được
  tính từ `warpSize`; với warp rộng 64, một khối 256 có 4 warp. Chưa được biên dịch hay chạy
  trên phần cứng AMD ở đây: unverified.
- Metal: các hàm SIMD group cần GPU family Apple7, mà mọi máy Mac Apple silicon đều thuộc
  (infra-facts C2). Các kernel không bao giờ giả định độ rộng SIMD group: mảng cho mỗi SIMD group
  được `launch.mm` đặt kích thước từ `threadExecutionWidth`.
- Thiết kế ba bước không cần bảo đảm tiến triển (forward progress) giữa các khối, đó là lý do nó là mốc khả chuyển cơ sở
  (A10, B6, C4). Lab 12.5 đánh đổi điều đó để đọc mỗi khóa chỉ một lần.
