> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 14.1: hai phép sắp xếp đầu tiên, chẵn-lẻ và radix một bit

Sách: mục 14.2 (pp. 331-333, Fig. 14.2) và mục 14.5 (pp. 336-339, Figs. 14.6 và
14.7). Một mục đứng riêng như (A9) hay (C4) bên dưới nằm trong `notes/today/ch14-facts.md`.

Hai phép sắp xếp khóa 32 bit không dấu, viết theo đúng cách chương sách xây dựng chúng, và được đo
thời gian so với nhau và so với một luồng CPU.

**Phần A: sắp xếp chẵn-lẻ (Fig. 14.2).**

- `oddEvenStep(data, hasChanged, n, isOddStep)`: luồng t sở hữu cặp (i, i + 1) với
  i = 2t ở bước chẵn và 2t + 1 ở bước lẻ. Nó sắp cặp đó theo thứ tự và, nếu đã
  hoán đổi, đặt `*hasChanged` thành 1 bằng một lệnh ghi nguyên tử (`atomicExch(hasChanged, 1u)`, hoặc
  `cuda::atomic_ref<unsigned int, cuda::thread_scope_device>(*hasChanged).store(1u,
  cuda::memory_order_relaxed)`). Nó trả về ngay khi n < 2 hoặc khi cặp của nó vượt quá
  cuối mảng.
- `launchOddEvenSort(data, flag, n)`: vòng lặp host mà sách mô tả nhưng không trình bày.
  Bước chẵn và bước lẻ xen kẽ nhau, bắt đầu bằng một bước chẵn. Xóa cờ, chạy một bước chẵn
  và bước lẻ theo sau, rồi đọc cờ. Chỉ dừng khi cặp bước đó không có lần
  hoán đổi nào. Trả về số bước đã khởi chạy.

**Phần B: một phép radix sort LSD một bit trên các cặp (khóa, giá trị) (Figs. 14.6 và 14.7).** Fig. 14.7 là
một kernel gọi `gridExclusiveScan` giữa bước trích các bit và bước di chuyển các khóa.
Sách không bao giờ định nghĩa hàm đó, và một phép scan xuyên các khối bên trong một kernel đòi hỏi các
khối chờ nhau: một lần khởi chạy cooperative trên CUDA và HIP, không có gì được tài liệu hóa trên
Metal (C4). Nên ở đây một lượt là ba lần khởi chạy:

- `radixExtractBits(keys, bits, n, bit)`: `bits[i]` = bit `bit` của `keys[i]`, và
  `bits[n] = 0`.
- `launchExclusiveScan(bits, n + 1, scratch)` được cung cấp sẵn: sau đó `bits[i]` là
  số bit một đứng trước khóa i và `bits[n]` là tổng số bit một.
- `radixScatter(keysIn, valsIn, keysOut, valsOut, onesBefore, n, bit)`: hai công thức đích
  của Fig. 14.6; giá trị di chuyển cùng khóa của nó.
- `launchRadixSortOneBit(keys, vals, tmpKeys, tmpVals, bits, scanScratch, n)`: 32 lượt,
  bit ít quan trọng nhất trước, đọc một cặp buffer và ghi vào cặp kia. Các cặp đã sắp xếp
  phải kết thúc trong `keys`/`vals`.

Các giá trị là vị trí đầu vào của các khóa. Trong lab 14.3 chúng là id token. Với vị trí làm
giá trị, phép kiểm thử có thể thấy tính ổn định trực tiếp: giữa các khóa bằng nhau, vị trí phải tăng dần.

## Năm cái bẫy trên trang sách

Mỗi bẫy biến một cách đọc sát chữ chương sách thành một chương trình sai hoặc dễ vỡ. Hãy tự tìm
ra cách sửa; lời giải tham chiếu (solution) ghi trong chú thích của nó chỗ nào khác với
trang sách.

1. **Phép scan N + 1** (`notes/ch14.md`, "Unclear in the PDF", mục 4). Fig. 14.7 đọc
   `bits[N]` như tổng số bit một. Phần tử đó chỉ tồn tại, và chỉ giữ tổng,
   nếu phép scan loại trừ chạy trên N + 1 phần tử. Fig. 14.6 vẽ N. Điều quan trọng là
   phần tử thêm vào, không phải bạn đặt gì vào đó: một phép scan loại trừ không bao giờ mang đầu vào cuối
   cùng của nó vào bất kỳ đầu ra nào. Dù vậy vẫn hãy ghi 0 vào đó, để mảng được scan là "các bit, rồi không gì cả".
2. **Lỗi viết "zero bucket"** (mục 3). Ở p. 337, câu giới thiệu công thức thứ hai
   nói rằng nó dành cho các khóa ánh xạ vào bucket số không. Công thức đó là của bucket số một.
   Hãy dùng các công thức theo nhãn mà hình vẽ gán cho chúng.
3. **Phép `n - 1` không dấu bị tràn vòng.** Fig. 14.2 kiểm tra biên bằng `i < N - 1`. Với N không dấu, N - 1
   tại N = 0 là giá trị không dấu lớn nhất và mọi luồng đều qua. Chương trình điều khiển (driver) không bao giờ khởi chạy
   kernel của bạn với n = 0, nên không phép kiểm thử nào bắt được; dù vậy hãy viết phép kiểm tra sao cho nó không thể tràn vòng.
4. **Quy tắc dừng** (mục 7). Sách nói hãy tiếp tục "until no elements are
   swapped". Một bước yên lặng không chứng minh được gì: `[1, 3, 2, 4]` không hoán đổi gì ở bước đầu tiên, bước chẵn,
   và vẫn chưa được sắp xếp. Hai bước liên tiếp, một chẵn và một lẻ theo thứ tự nào cũng được, cả hai
   yên lặng, cho thấy mọi cặp kề nhau đều đúng thứ tự. Lab này đọc cờ một lần mỗi cặp
   chẵn-lẻ, nên nó chỉ thấy những cặp yên lặng bắt đầu bằng một bước chẵn: nó có thể khởi chạy
   thừa một bước (10 thay vì 9 cho `[8, 1, 2, 3, 4, 5, 6, 7]`) để đổi lấy
   một nửa số lần đọc. Các phép kiểm thử có cả cái bẫy này và một đầu vào có giá trị lớn nhất đứng đầu.
5. **Race trên cờ** (p. 333). Nhiều luồng có thể đặt `hasChanged` trong cùng một bước. Sách
   gọi race này là vô hại vì chúng đều ghi 1, rồi cảnh báo rằng mô hình bộ nhớ C++
   dù sao vẫn coi nó là hành vi không xác định. Mô hình bộ nhớ của NVIDIA nói điều tương tự (A9). Hãy dùng một
   lệnh ghi nguyên tử. Backend `race` báo một lệnh ghi thường là race condition ghi-ghi với
   cả hai dòng mã nguồn; `emu` không thấy được, vì nó không bao giờ ngắt một luồng giữa các
   điểm đồng bộ hóa.

## Tệp

- Sửa `work/kernel.cu` (CUDA, HIP và cả hai trình giả lập): `oddEvenStep`, `launchOddEvenSort`,
  `radixExtractBits`, `radixScatter`, `launchRadixSortOneBit`.
- Sửa `work/kernel.metal` (`odd_even_step`, `radix_extract_bits`, `radix_scatter`; các
  chữ ký hàm đã cho sẵn). Cờ là một `device atomic_uint*` được đặt bằng
  `atomic_store_explicit(..., memory_order_relaxed)`: các hàm nguyên tử của Metal chỉ nhận
  con trỏ kiểu atomic (`notes/today/infra-facts.md` C1).
- Sửa `work/launch.mm`:
  - `runOddEvenSort(queue, step, data, flag, n)` commit một command buffer cho mỗi cặp bước
    chẵn-lẻ, chờ nó, và đọc cờ qua `flag.contents`. Một buffer shared-storage
    không cần sao chép, nhưng CPU chỉ được đọc nó sau khi công việc GPU ghi vào nó đã
    hoàn tất (`notes/today/ch05-facts.md` C6; `waitUntilCompleted`, infra-facts C12).
    Trả về số bước, hoặc -1 nếu một command buffer thất bại.
  - `encodeRadixSortOneBit(enc, extract, scatter, keys, vals, tmpKeys, tmpVals, bits,
    scanScratch, n)` đặt cả 32 lượt, kể cả các dispatch của phép scan được cung cấp, vào
    một compute encoder duy nhất mà nó nhận được. Một encoder tạo bằng `computeCommandEncoder` thông thường chạy
    các dispatch của nó theo thứ tự và mỗi dispatch thấy được những gì dispatch trước đã ghi
    (`notes/today/ch02-facts.md` C9).
- Được cung cấp, không sửa: `labs/ch14/shared-g11/scan.cu`, `scan.metal`, `scan_launch.mm`
  (của khóa học này). Một phép scan loại trừ tại chỗ trên unsigned int cho số lượng bất kỳ, xây dựng như mục 11.9
  xây dựng (scan-scan-add, Fig. 11.14, pp. 273-274): scan từng khối, một phép scan đệ quy trên
  tổng các khối, một phép cộng. Không khối nào chờ khối khác. `scanScratchCount(count)` cho kích thước
  vùng nháp; driver cấp phát vùng nháp đó.

## Lệnh

- Kiểm thử: chạy `python3 labs/pmpp.py run 14.1 --backend emu` trước (vài phút: mọi luồng GPU
  chạy trên CPU), rồi `python3 labs/pmpp.py run 14.1`.
- Race trên cờ: `python3 labs/pmpp.py run 14.1 --backend race`.
- Đo: `python3 labs/pmpp.py run 14.1 --bench` (chỉ trên thiết bị thật).
- Lời giải tham chiếu: thêm `--solution`.

## Các phép kiểm thử kiểm tra gì

- Riêng phép scan được cung cấp so với một phép scan loại trừ trên CPU, để một tệp cung cấp sẵn bị hỏng
  lộ ra đúng là chính nó.
- Chẵn-lẻ: n = 1 (kiểm tra cùng với 2), 3, 4, 8, 33, 256 và 1000; trình giả lập dừng
  ở 256, backend race ở 64. Đầu vào: ngẫu nhiên, đảo ngược, đã sắp xếp, tất cả bằng nhau,
  lớn nhất đứng đầu (trường hợp xấu nhất của sách, p. 333) và cái bẫy ở trên. Kết quả phải
  được sắp xếp và là một hoán vị của đầu vào; số bước phải chẵn (trọn cặp), ít nhất
  2 khi n từ 2 trở lên, và nhiều nhất n + 2 làm tròn lên thành trọn cặp (n + 2 với n chẵn, n + 3
  với n lẻ: phép sắp xếp cần nhiều nhất n bước, rồi cặp chứa lần hoán đổi cuối phải hoàn tất
  và một cặp yên lặng theo sau). Các ô canh gác sau cuối mảng giữ 0, nên một cặp vượt quá
  phần tử cuối sẽ hoán đổi một khóa vào đó và bị bắt.
- Radix: n = 1 (kiểm tra cùng với 2), 31, 32, 33, 1000, 4097, 2^16 và 2^20; trình giả lập
  dừng ở 4097, backend race ở 1000. Khóa: ngẫu nhiên toàn dải (bit cao nhất được đặt khoảng
  một nửa số lần, nên một khóa có dấu hoặc một lượt dừng trước bit 31 sẽ thất bại), 0..15 (nhiều
  khóa bằng nhau), tất cả bằng nhau, đã sắp xếp, đảo ngược, lũy thừa của hai. Các khóa phải bằng kết quả của `std::stable_sort`
  và các giá trị phải bằng đúng hoán vị của nó. Trình giả lập chạy mọi mẫu khóa ở
  n = 1000 và hai mẫu ở mỗi kích thước khác; nó trả giá cho mọi luồng của mọi khối trong
  96 lần khởi chạy trở lên mà một phép sắp xếp thực hiện.
- Trước mỗi lần chạy, driver ghi 0xCDCDCDCD vào mọi ô của `bits`, vùng nháp của phép scan,
  các buffer tạm và 64 ô canh gác sau cuối mỗi mảng. Một lượt ghi quá n sẽ bị bắt;
  trên trình giả lập, AddressSanitizer dừng một lượt ghi quá vùng cấp phát và chỉ ra dòng mã.

## Trước khi bạn benchmark

Hãy ghi lại hai dự đoán (trang labs yêu cầu chúng):

1. Ở n = 4096, phép sắp xếp chẵn-lẻ của bạn sẽ chậm hơn phép radix sort một bit bao nhiêu lần?
   (`ch14/labs/p-lab1-oddeven`)
2. Ở 2^22 cặp, phép radix sort một bit của bạn sẽ nhanh hơn `std::stable_sort` của một luồng CPU bao nhiêu lần?
   (`ch14/labs/p-lab1-cpu`)

Hãy ghi lý do trước, từ hai phép đếm:

- **Số lần khởi chạy và số chuyến khứ hồi host mỗi phép sắp xếp.** Chẵn-lẻ khởi chạy một kernel mỗi bước, khoảng n
  lần (nhiều nhất n + 3), và host đọc lại cờ một lần mỗi cặp bước, nên khoảng
  n / 2 chuyến khứ hồi, mỗi chuyến chờ GPU xong. Radix sort khởi chạy
  32 x (2 + số lần khởi chạy của phép scan) kernel và không bao giờ chờ ở giữa. Phép scan trên n + 1
  phần tử là 1 lần khởi chạy cho tới 1024 phần tử, 3 cho tới 1024 x 1024, 5 khi vượt quá.
- **Số byte mỗi lượt.** Một bước chẵn-lẻ đọc mọi khóa và ghi những khóa nó hoán đổi. Một
  lượt radix đọc các khóa và ghi các bit (8 byte mỗi cặp), scan n + 1 phần tử (phép scan
  được cung cấp đọc và ghi chúng hai lần khi có nhiều hơn một ô: 16), đọc
  khóa, giá trị và bit rồi ghi khóa và giá trị (20). Hãy đếm cho 32 lượt, so với
  O(n log n) phép so sánh và di chuyển của `std::stable_sort`.

Benchmark đo thời gian chẵn-lẻ ở n = 1024 và 4096 (thời gian thực của cả vòng lặp host: các lần khởi chạy
và các lần đọc lại cờ), radix sort một bit ở n = 1024, 4096, 2^16, 2^20 và 2^22, và
`std::stable_sort` của cùng các cặp (khóa, vị trí) trên một luồng CPU với `std::chrono` (một
phép đo phía host). Các hàng radix cho thời gian thiết bị của command buffer trên Metal và
thời gian thực quanh các lần khởi chạy và lần đồng bộ hóa cuối cùng trên CUDA và HIP, cộng một cột
"wall", đồng hồ của host trên mọi backend; "odd-even/radix" chia hai thời gian thực đó. Mỗi
hàng là kết quả tốt nhất trong 5 lần chạy sau một lần khởi động, và mọi kết quả được đo đều được kiểm tra trước khi
báo cáo. Các con số là phép đo trên máy của bạn, không phải mục tiêu.

## Tính khả chuyển

- HIP: `atomicExch` có cùng tên và đối số (infra-facts B1); các kernel không dùng intrinsic
  warp nào, nên không có lane mask. Chưa được biên dịch hay chạy trên phần cứng AMD ở đây: unverified.
- Metal: cờ là `device atomic_uint*` (infra-facts C1); một command buffer cho mỗi cặp
  bước, đọc sau `waitUntilCompleted` (ch05-facts C6); mọi lượt radix trong một encoder
  tuần tự (ch02-facts C9). Lượt ba lần khởi chạy không cần tiến triển giữa các threadgroup, điều mà
  Metal không tài liệu hóa (C4).
- Lời khuyên của chính sách (p. 347) là dùng một phép sắp xếp thư viện, `cub::DeviceRadixSort` trên CUDA
  (ổn định, theo cặp, giảm dần: A2) hoặc `hipcub::` trên HIP (B1). Metal không có phép sắp xếp thư viện
  ở cấp kernel (C1). Lab này để hiểu một phép sắp xếp như vậy làm gì ở mỗi lượt.
