> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 11.2: gộp việc cho luồng và phân ô thanh ghi

Sách: các mục 11.6 đến 11.8 (pp. 267-273), Figs. 11.11 đến 11.13, và bài tập 4 (p. 286).

Bạn viết scan khối có gộp việc cho luồng (thread coarsening) của Fig. 11.12 và dạng phân ô
thanh ghi của nó, Fig. 11.13, rồi đo thời gian so với một biến thể được cung cấp bỏ qua bộ nhớ
chia sẻ (shared memory) và một kernel chép. Phép scan các tổng của từng luồng là `blockScan` của
bạn từ lab 11.1.

Hợp đồng: khối b scan đoạn `COARSE_FACTOR*blockDim.x` phần tử của riêng nó, bao gồm, với phép +,
và không biết gì về các khối khác. n có thể là bất kỳ giá trị nào từ 1 trở lên. Một phần tử ở
vị trí n hoặc sau đó nạp 0 và không được ghi. Kích thước khối là lũy thừa của hai từ 64 đến
1024. `COARSE_FACTOR` là 1, 2, 4, 8 hoặc 16.

**Phần việc của bạn:**

- **(a) `CoarsenedScanKernel`** (Fig. 11.12). Khối nạp đoạn của nó vào bộ nhớ chia sẻ qua
  `COARSE_FACTOR` lượt. Ở mỗi lượt, luồng t nạp phần tử t của `blockDim.x` phần tử tiếp theo,
  nên các luồng của một warp đọc các địa chỉ liên tiếp. Sau đó mỗi luồng scan dải
  `COARSE_FACTOR` phần tử liên tiếp của riêng nó, tuần tự, trong bộ nhớ chia sẻ. `blockScan`
  scan các tổng của dải, mỗi luồng cộng tổng của các dải đứng trước dải của mình, và khối ghi
  theo lượt, như lúc nạp.
- **(b) `RegisterTiledScanKernel`** (Fig. 11.13). Tương tự, nhưng dải nằm trong một mảng cục bộ
  `buffer_r[COARSE_FACTOR]`. Mỗi phần tử được đọc từ bộ nhớ chia sẻ một lần, và các vòng lặp
  trên nó mang `#pragma unroll`. Bộ nhớ chia sẻ chỉ còn để làm cho các lần nạp và ghi toàn cục
  được liên tục (p. 272).
- Đã sửa so với trang sách: các kernel của sách không bao giờ dùng `N`, nên chúng chỉ đúng khi
  n là bội số của đoạn. Kernel của bạn kiểm tra mọi lần nạp và ghi toàn cục, và một phần tử
  ngoài phạm vi đóng góp 0.
- `COARSE_FACTOR` là một tham số template (`template <unsigned int COARSE_FACTOR>`), nên một
  lần build kiểm thử và đo cả năm hệ số. Các hàm khởi chạy nhận hệ số làm đối số cuối cùng và
  chọn bản khởi tạo bằng một `switch`. Trên Metal, nó là một macro tiền xử lý, được đặt khi
  chương trình điều khiển (driver) build pipeline: mỗi hệ số một pipeline.
- Buffer đoạn là bộ nhớ chia sẻ động, `extern __shared__ float buffer_s[];`, có kích thước theo
  đối số thứ ba của `<<< >>>`. Trên trang sách, nó là một mảng tĩnh `COARSE_FACTOR*BLOCK_DIM`
  với `BLOCK_DIM` cố định lúc biên dịch. Nếu đặt kích thước theo khối lớn nhất, một mảng tĩnh
  sẽ giữ chỗ gấp bốn lần lượng mà một khối 256 luồng dùng, và benchmark sẽ đo cả điều đó. Với
  `COARSE_FACTOR*blockDim.x` tối đa 8192 (hệ số 16 chỉ với khối tối đa 512), tổng vẫn nằm trong
  48 KiB mà một khối được dùng mà không cần đăng ký thêm (ch05 facts A3).

## Tệp

- Sửa `work/kernel.cu` (CUDA, HIP và cả hai trình giả lập): `CoarsenedScanKernel`,
  `RegisterTiledScanKernel`, `blockScan` của bạn cùng các hàm phụ trợ chép từ lab 11.1, và
  `launchCoarsenedScan` và `launchRegisterTiledScan`.
- Sửa `work/kernel.metal` (`coarsened_scan`, `register_tiled_scan`, và `block_scan` của bạn từ
  lab 11.1) và `work/launch.mm` (`encodeCoarsenedScan`, `encodeRegisterTiledScan`). Mỗi hàm
  encode trả về false, không dispatch, cho một cấu hình mà pipeline của nó không chạy được:
  nhiều luồng hơn `maxTotalThreadsPerThreadgroup`, một khối không gồm trọn các SIMD group (tối
  đa `width` SIMD group), hoặc nhiều bộ nhớ threadgroup hơn `maxThreadgroupMemoryLength` của
  thiết bị.
- Được cung cấp, không sửa, trong `../shared-g11/`: `bs_threadmajor.cu` và `.metal` (biến thể
  thread-major, bên dưới), `bs_copy.cu` và `.metal`, và `bs_launch.mm`.

Kernel **thread-major** được cung cấp là Fig. 11.13 không có bước trung chuyển: mỗi luồng nạp
dải `COARSE_FACTOR` phần tử liên tiếp của riêng nó thẳng từ bộ nhớ toàn cục và ghi thẳng trở
lại. Trong một lệnh nạp, các lane của một warp chạm vào các địa chỉ cách nhau `COARSE_FACTOR`
số float, chính là mẫu mà p. 269 cảnh báo. Đây không phải một ví dụ dựng lên cho dễ đánh đổ.
Hàm cumsum CUDA của llama.cpp nạp theo cách này, và scan Metal của MLX cũng để mỗi luồng đọc
các giá trị liên tục của riêng nó (ch11 facts D1, D3). Benchmark của bạn cho thấy cái giá của nó
trên thiết bị của bạn.

## Lệnh

- Kiểm thử: chạy `python3 labs/pmpp.py run 11.2 --backend emu` trước, rồi `python3 labs/pmpp.py run 11.2`.
- Các lần cập nhật bộ nhớ chia sẻ: `python3 labs/pmpp.py run 11.2 --backend race`.
- Đo: `python3 labs/pmpp.py run 11.2 --bench` (chỉ trên thiết bị thật).
- Lời giải tham chiếu: thêm `--solution`.

## Các bài kiểm thử làm gì

- n = 1, 2, 33, 1000, 4097, 65537 và 2^22 + 3, với các hệ số 1, 2, 4, 8 và 16 và khối 64, 256
  và 1024. Một cấu hình mà bộ nhớ không vừa sẽ bị bỏ qua, và chương trình điều khiển in ra lý
  do. Trình giả lập chạy n tới 4097 với các hệ số 1, 4 và 8 và khối 64 và 256. `race` chạy
  n = 1000 với khối 64.
- Cùng ba loại dữ liệu và cùng dung sai như lab 11.1, theo từng đoạn: toàn số một và các số
  nguyên từ -3 đến 3 phải khớp chính xác. Các giá trị phân bố đều trong [-1, 1] phải nằm trong
  phạm vi 1e-6 nhân tổng |x| của đoạn tính đến phần tử đó, cộng 1e-6, so với vòng lặp của Fig.
  11.1 chạy bằng double.
- Được kiểm tra riêng trước phần còn lại: phần tử đầu tiên của mọi đoạn (scan bắt đầu lại ở
  đó), phần tử cuối cùng của mọi đoạn (tổng của đoạn), và phần tử cuối cùng của một đoạn cuối
  không đầy.
- Đầu ra khởi đầu là NaN. Các từ canh gác nằm sau nó, tới cuối đoạn cuối cùng và thêm 16 nữa,
  nên một lần ghi ở vị trí n hoặc sau đó sẽ bị báo.
- Các kernel thread-major và kernel chép được cung cấp cũng được kiểm thử.

Mỗi backend có thể bắt được gì:

- `emu` chạy các luồng lần lượt từng luồng với rác trong bộ nhớ chia sẻ, dưới
  AddressSanitizer. Mỗi lỗi sau đây tự nó thất bại ở đó: một lần nạp không kiểm tra với n
  (AddressSanitizer chỉ ra dòng), một lần ghi không kiểm tra với n (các từ canh gác), một đoạn
  nạp theo một thứ tự và đọc theo thứ tự kia, và thiếu rào chắn (barrier) sau bước nạp hoặc
  trước bước ghi.
- `race` chạy các luồng cùng lúc dưới ThreadSanitizer và báo một rào chắn bị thiếu cùng cả hai
  dòng mã nguồn.
- Metal là một GPU thật. Nó cho thấy kernel của bạn có nhanh không, chứ không chỉ có đúng không.

## Chỗ các backend khác nhau (không có trong sách)

- **HIP**: `extern __shared__`, template và `#pragma unroll` giữ nguyên cách viết (ch05 facts
  B4; infra facts B7). Các quy tắc mặt nạ và độ rộng warp của lab 11.1 vẫn áp dụng cho
  `blockScan` của bạn. Chương trình điều khiển đọc giới hạn bộ nhớ chia sẻ từ thiết bị
  (`sharedMemPerBlock`), vốn khác trên AMD. Không phần nào trong số này đã được build hay chạy
  với ROCm ở đây: unverified.
- **Metal**: bộ nhớ threadgroup động là một tham số kernel `threadgroup float*` được đánh dấu
  `[[threadgroup(0)]]`. Kích thước của nó được đặt bằng `setThreadgroupMemoryLength:atIndex:`,
  theo bội số của 16 byte (ch05 facts C8). Giới hạn là `maxThreadgroupMemoryLength` của thiết
  bị, và các mảng tĩnh của kernel cũng tính vào đó (`staticThreadgroupMemoryLength`, ch05 facts
  C3). Đó là lý do một số cặp hệ số và khối chạy được trên CUDA lại bị bỏ qua trên Metal. MSL
  không ghi trong tài liệu cách điều khiển unroll (infra facts, "Where the backends really
  differ"). Dải là một mảng cục bộ kích thước cố định với số vòng lặp không đổi, và trình biên
  dịch quyết định.
- Trên Metal bạn cũng có thể xây `block_scan` trên `simd_prefix_inclusive_sum` (lab 11.1,
  (b2)). Các bài kiểm thử không quan tâm bạn dùng cách nào.

## Trước khi đo hiệu năng, hãy dự đoán

Hãy ghi những điều này ra trước. Bài học hỏi chúng dưới các id trong ngoặc.

1. Hệ số nào từ 1 đến 16 sẽ nhanh nhất với khối 256 (`ch11/labs/p-lab2-best`).
2. Kernel phân ô thanh ghi sẽ đạt bao nhiêu phần băng thông chép của bạn ở hệ số tốt nhất của
   nó (`ch11/11.8/p-copy`, được hỏi ở mục 11.8).
3. Kernel phân ô thanh ghi (b) của bạn, trung chuyển qua bộ nhớ chia sẻ, sẽ nhanh hơn bao nhiêu
   lần so với kernel thread-major được cung cấp ở cùng hệ số, vốn nạp mỗi dải thẳng từ bộ nhớ
   toàn cục (`ch11/11.6/p-staging`, được hỏi ở mục 11.6 với khối 256, hệ số 4).

Mô hình của sách: với P luồng và N phần tử, gộp việc cho luồng đưa khối lượng công việc từ
khoảng N·log2(N) xuống khoảng 2N + P·log2(P), và số bước xuống khoảng 2N/P + log2(P)
(pp. 269-270). Một scan di chuyển ít nhất 8 byte mỗi phần tử, và một kernel bị giới hạn bởi bộ
nhớ đạt hơn 80% băng thông mà không có truy cập thừa được xem là tối ưu tốt (pp. 272-273). Ở
đây thước đo là kernel chép của chính bạn.

## Đọc kết quả

Benchmark scan n = 2^26 số float (ít hơn nếu đầu vào và đầu ra chiếm hơn một phần tư bộ nhớ
thiết bị của bạn báo), với khối 256 và 1024. Với mỗi kích thước khối, nó đo kernel chép, rồi,
ở mỗi hệ số từ 1 đến 16, các kernel gộp việc và phân ô thanh ghi của bạn cùng kernel
thread-major. Một cấu hình không vừa được in ra là bị bỏ qua, kèm lý do.

- `time ms` là lần tốt nhất trong năm lần chạy sau một lần khởi động không đo. Một phần tư
  giây các phép sao chép không đo chạy trước, để kernel được đo đầu tiên không phải trả thêm chi
  phí đánh thức thiết bị.
- `GB/s` tính 8 byte mỗi phần tử (một lần đọc 4 byte, một lần ghi 4 byte) chia cho thời gian
  đó: số byte một scan phải di chuyển, không phải số byte hệ thống bộ nhớ đã di chuyển.
- `of copy` là tốc độ của kernel tính theo tỉ lệ so với kernel chép ở cùng kích thước khối.

Cùng các hàng đó được đưa vào các trang bài học (mục 11.6, 11.7 và 11.8 và trang labs). Các con
số là phép đo của máy này ở kích thước này. Chúng không phải là thuộc tính của các kernel.

## Đi xa hơn (chưa kiểm thử)

Bài tập 4 (p. 286), bắt đầu từ kernel phân ô thanh ghi của bạn:

- **Nạp vector** cho các lượt từ toàn cục sang chia sẻ và từ chia sẻ sang toàn cục. Kỹ thuật
  của sách trên CUDA ép `float*` thành `float4*`. Không nhà cung cấp nào ghi phép ép đó trong
  tài liệu (infra facts A6, B6). Trên Metal, thay vào đó hãy khai báo buffer là
  `device const float4*`, cách được ghi trong tài liệu để đọc bốn số float một lúc (infra facts
  C7). Khi đó mỗi luồng di chuyển bốn phần tử liền kề mỗi lượt, nên các lượt và chỉ số của
  chúng thay đổi. n không nhất thiết là bội số của 4: hãy xử lý phần đuôi.
- **Đệm thêm (padding)** cho các truy cập bộ nhớ chia sẻ có bước nhảy. Luồng t đọc
  `buffer_s[t*F + c]`, nên các lane cạnh nhau cách nhau F số float. Với 32 bank 4 byte của CUDA
  (infra facts A10), đó là một xung đột bank (bank conflict) với mọi F chẵn (phép tính của
  chúng tôi). Hãy đệm mảng sao cho chỉ số i nằm ở `i + i/32`, và đo xem điều đó có quan trọng
  không. Apple không ghi gì trong tài liệu về bank của bộ nhớ threadgroup (infra facts C13),
  nên trên Metal phép đo là câu trả lời duy nhất.
