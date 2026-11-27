> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 14.2: radix sort nhiều bit với sắp xếp cục bộ

Sách: các mục 14.6 đến 14.8 (pp. 339-345, Figs. 14.8, 14.9, 14.12 và 14.13) và các bài tập 1,
2 và 3 (p. 347). Sách không đưa mã cho bất kỳ phần nào trong số này; các hình vẽ cho ra thiết kế. Một
mục đứng riêng như (A4) hay (C4) bên dưới nằm trong `notes/today/ch14-facts.md`.

Mốc so sánh của lab này là phần B của lab 14.1, được cung cấp sẵn trong `labs/ch14/shared-g11/`, nên
bạn có thể bắt đầu từ đây; lab 14.1 là tùy chọn.

Một phép radix sort LSD ổn định trên các cặp (khóa, giá trị) làm cả ba bài tập cùng lúc: sắp xếp
cục bộ trong bộ nhớ chia sẻ (shared memory) (bài tập 1), cơ số r bit (bài tập 2) và c khóa mỗi luồng
(bài tập 3). r chạy từ 1 đến 8 và c từ 1 đến 8, cả hai được chọn lúc chạy. Một khối có 256
luồng và sở hữu một ô (tile) gồm 256c cặp liên tiếp; ô cuối có thể không đầy, và phần
đệm của nó không được đếm cũng không được ghi.

Một lượt trên chữ số `(key >> shift) & (2^r - 1)` là ba lần khởi chạy, như mục 14.6 trình
bày:

1. `radixLocalSort(keysIn, valsIn, keysLocal, valsLocal, table, localStarts, n, shift,
   radixBits, keysPerThread)`. Nạp ô vào bộ nhớ chia sẻ với truy cập được hợp nhất (coalesced). Sắp xếp nó theo
   chữ số bằng r phép tách một bit ổn định, bit ít quan trọng nhất của chữ số trước (p. 342). Mỗi phép tách
   là một phép scan loại trừ cấp khối: mỗi luồng đếm số bit một trong c khóa của nó, một phép scan trên
   256 tổng của các luồng cho số bit một đứng trước chúng (scan-scan-add của chương 11 bên trong một
   khối), và hai công thức của Fig. 14.6, áp dụng bên trong ô, cho mỗi cặp vị trí mới của nó.
   Rồi ghi ô đã sắp xếp trở lại cùng vị trí ô, ghi 2^r kích thước bucket của khối
   vào bảng theo chữ số trước `table[d * numBlocks + block]` (Figs. 14.9 và 14.12: số đếm của mọi
   khối cho chữ số 0, rồi cho chữ số 1, ...), và ghi chỗ mỗi bucket bắt đầu bên trong
   ô vào `localStarts[block * 2^r + d]`.
2. `launchExclusiveScan(table, 2^r * numBlocks, scratch)` được cung cấp sẵn: vị trí bắt đầu toàn cục
   của mỗi bucket cục bộ.
3. `radixScatterLocal(keysLocal, valsLocal, keysOut, valsOut, tableScanned, localStarts, n,
   shift, radixBits, keysPerThread)`: cặp ở vị trí ô p với chữ số d đi tới
   `tableScanned[d * numBlocks + block] + (p - localStarts[block * 2^r + d])`. Các luồng liên tiếp
   nhận các vị trí ô liên tiếp, nên bên trong một bucket chúng ghi vào các địa chỉ liên tiếp.
   Đó chính là sự hợp nhất truy cập bộ nhớ mà mục 14.6 nhắm tới.

`launchRadixSortMulti(keys, vals, tmpKeys, tmpVals, table, localStarts, scanScratch, n,
radixBits, keysPerThread)` chạy ceil(32 / r) lượt và để kết quả trong `keys`/`vals`.
Khi r không chia hết 32 (r = 3, 5, 6, 7), lượt cuối chỉ có các bit còn lại; một phép
dịch 32 bit trở lên là không xác định trong C++, nên độ rộng chữ số cuối phải thu nhỏ, chứ không chỉ
riêng mặt nạ. Các kernel nhận `keysPerThread` làm tham số cuối (kích thước của mọi thứ
phụ thuộc vào nó).

## Tệp

- Sửa `work/kernel.cu` (CUDA, HIP và cả hai trình giả lập): `radixLocalSort`,
  `radixScatterLocal`, `launchRadixSortMulti`. Làm phép scan cấp khối trong bộ nhớ chia sẻ với
  rào chắn. Một phép scan bằng warp-shuffle cần lane mask, và trên HIP mask phải là số nguyên 64 bit
  trên mọi target (`notes/today/infra-facts.md` B3); dạng dùng bộ nhớ chia sẻ không có
  mask nào để làm sai. Đừng bao giờ giả định độ rộng warp.
- Sửa `work/kernel.metal` (`radix_local_sort`, `radix_scatter_local`; các chữ ký hàm đã
  cho sẵn). Phép scan cấp khối có thể dùng `simd_prefix_exclusive_sum` và `simd_sum` bên trong mỗi
  SIMD group (infra-facts C2) với một ô threadgroup cho mỗi SIMD group để chứa tổng của chúng. Độ
  rộng SIMD group không phải là hằng số được tài liệu hóa (infra-facts C5): `launch.mm` đọc nó từ
  `threadExecutionWidth` của pipeline, và kernel có nó dưới dạng `[[threads_per_simdgroup]]`.
  Một phép scan làm theo cách này cộng các tổng của luồng theo thứ tự (SIMD group, lane), và cách một
  threadgroup được chia thành các SIMD group là do bản cài đặt định nghĩa (`notes/today/ch04-facts.md`
  C2). Nên trong các phép tách, hãy giao cho mỗi luồng các vị trí ô theo thứ hạng của nó trong thứ tự đó,
  `sg * width + lane`, chứ không theo `thread_position_in_threadgroup`; các lần nạp và ghi được hợp nhất
  có thể giữ vị trí luồng.
- Sửa `work/launch.mm` (`encodeRadixSortMulti`): mọi lượt, kể cả phép scan được cung cấp, trong
  một compute encoder tuần tự duy nhất mà nó được giao (`notes/today/ch02-facts.md` C9). Bộ nhớ
  threadgroup gồm ô (256c x 8 byte) ở `[[threadgroup(0)]]`, các tổng SIMD group ở
  `[[threadgroup(1)]]`, và bất cứ gì kernel tự khai báo
  (`staticThreadgroupMemoryLength`). Các độ dài là bội số của 16 byte và tổng phải vừa
  `maxThreadgroupMemoryLength` (`notes/today/ch05-facts.md` C1, C3, C8); trả về false trước khi
  encode bất cứ thứ gì khi không vừa.
- Được cung cấp, không sửa: `labs/ch14/shared-g11/scan.*` (phép scan của lab 14.1) và
  `onebit_global.cu`, `onebit_global.metal`, `onebit_global_launch.mm`: thiết kế của lab 14.1,
  một bit mỗi lượt với một phép scan trên n + 1 bit và mọi khóa được ghi thẳng vào vị trí toàn
  cục của nó. Đó là mốc so sánh cho nhận định của mục 14.6.

## Lệnh

- Kiểm thử: chạy `python3 labs/pmpp.py run 14.2 --backend emu` trước (vài phút: mọi luồng GPU
  chạy trên CPU), rồi `python3 labs/pmpp.py run 14.2`.
- Race trên bộ nhớ chia sẻ: `python3 labs/pmpp.py run 14.2 --backend race`. Mỗi phép tách
  đều ghi lại ô, nên thiếu một rào chắn sẽ để một luồng đọc một ô mới ghi được một nửa.
  Khi cài lỗi vào 2026-11-20 trong một bản sao của lời giải tham chiếu (bỏ `__syncthreads()` ở cuối
  vòng lặp tách), kết quả là:
  - `emu`: r = 1 qua (rào chắn của bước 3 che khoảng trống); r = 2, c = 1 thất bại ở thứ tự
    luồng đầu tiên với thứ tự sai, rồi AddressSanitizer báo heap-buffer-overflow tại
    lệnh ghi ô `vals_s[dst] = ...`.
  - `race` (chạy lại vào 2026-11-21 sau bản sửa rào chắn hai token, bốn lần chạy, mỗi lần dừng trong
    chưa tới một phút): r = 1 qua; r = 2, c = 1 và c = 2 thất bại. Giờ mỗi lần chạy đều chỉ ra cặp
    của rào chắn bị thiếu: một lệnh ghi ô ở một phép tách (`vals_s[dst] = ...`, hoặc
    `keys_s[dst]`) đối đầu với một lệnh đọc nó ở đầu phép tách kế tiếp (`myVals[k] = vals_s[p]`);
    một số lần chạy còn báo buffer `scan_s` của phép scan được dùng lại qua các phép tách và một race
    ghi-ghi trên `start_s` ở bước 3. Lần chạy sau đó dừng với một SEGV của ThreadSanitizer tại
    `keys_s[dst] = ...` (một chỉ số bị hỏng; backend race không có kiểm tra biên). Khi lỗi này
    được cài lần đầu, biến thể race dùng chung một token ThreadSanitizer giữa mọi
    `__syncthreads()` của một khối, và chỉ race trên `start_s` được báo; giờ nó luân phiên
    hai token cho mỗi rào chắn (`labs/ch14/snippets/README.md`, kernel wrap). Dù vậy, một lần chạy
    race sạch không phải bằng chứng rằng mọi rào chắn đều có mặt. Hãy chạy `emu` trước.
- Đo: `python3 labs/pmpp.py run 14.2 --bench` (chỉ trên thiết bị thật).
- Lời giải tham chiếu: thêm `--solution`.

## Các phép kiểm thử kiểm tra gì

- Mốc so sánh được cung cấp, để một tệp cung cấp sẵn bị hỏng lộ ra đúng là chính nó.
- Trên GPU, mọi r trong 1, 2, 3, 4, 5, 8 (3 và 5 không chia hết 32) với mọi c trong 1, 2, 4,
  ở n = 1, 255, 256, 257, 1000, 4097, 65537 và 2^20, với tám mẫu khóa: ngẫu nhiên
  toàn dải, 0..15 (nhiều khóa bằng nhau), tất cả bằng nhau, đã sắp xếp, đảo ngược, lũy thừa của hai, "mọi khóa chung
  byte thấp" (ở các lượt đầu, mỗi khối đặt tất cả khóa của mình vào một bucket, nên
  bảng gần như toàn số không) và "khóa nhỏ hơn 256" (mọi chữ số trên bit 7 đều bằng không).
- Các trình giả lập CPU trả giá cho mọi luồng của mọi khối trong tối đa 96 lần khởi chạy mỗi phép sắp xếp, nên
  ở đó mỗi (r, c) với c trong 1, 2 (trừ r = 1, c = 2, chậm nhất) chạy ở n = 1, 257 và
  1000 với các mẫu khóa luân phiên qua các cấu hình, cộng mọi mẫu ở r = 4,
  c = 2, n = 600 (hai ô), hai kích thước quanh một ô ở r = 5, hai mẫu nhiều số không
  ở r = 8, c = 2, và (chỉ emu) n = 4097 ở r = 8, c = 1, với bảng 4352
  phần tử cần một phép scan qua nhiều ô.
- Các khóa phải bằng kết quả của `std::stable_sort` và các giá trị, vốn là vị trí đầu vào,
  phải bằng đúng hoán vị của nó: giữa các khóa bằng nhau, vị trí phải tăng dần.
- Trước mỗi lần chạy, chương trình điều khiển (driver) ghi 0xCDCDCDCD vào toàn bộ bảng, `localStarts`, vùng nháp
  của phép scan, các buffer tạm và 64 ô canh gác sau cuối mỗi mảng cặp.
  Một số đếm bucket không được ghi cho lượt này, hoặc một lệnh ghi vào phần đệm, sẽ thất bại.
- Một cấu hình mà thiết bị không chạy được (threadgroup hoặc bộ nhớ chia sẻ vượt giới hạn) được
  in ra là bị bỏ qua kèm lý do; nó không được tính là qua.

## Trước khi bạn benchmark

Hãy đưa ra ba dự đoán (các trang yêu cầu chúng):

1. Sắp xếp cục bộ ở r = 1, c = 1 có thắng phép phân tán trực tiếp được cung cấp không, và gấp bao nhiêu lần?
   (`ch14/14.6/p-local-gain`, ở trang 14.6)
2. Trong r = 1, 2, 4, 6 và 8, giá trị nào nhanh nhất ở c = 1? (`ch14/14.7/p-best-r`, ở trang 14.7)
3. Cấu hình tốt nhất của bạn nhanh hơn `std::stable_sort` của một luồng CPU bao nhiêu lần
   ở 2^22 cặp? (`ch14/labs/p-lab2-cpu`)

Các trình khám phá ở trang 14.7 và 14.8 cho thấy mô hình kỳ vọng gì. Hãy tự đếm:

- **Số lượt**: ceil(32 / r). Mỗi lượt đọc và ghi mỗi cặp hai lần (sắp xếp cục bộ, rồi
  phân tán): 32 byte mỗi cặp mỗi lượt. Mốc so sánh di chuyển khoảng 44 byte mỗi cặp mỗi lượt
  (phép đếm của lab 14.1) nhưng ghi thành nhiều nhất hai dải mỗi warp; nó có 32 lượt.
- **Số dải mỗi warp trong phép phân tán**: với 2^r bucket, các khóa của một warp rơi vào tối đa
  min(2^r, độ rộng warp) dải địa chỉ riêng biệt (A10). Sắp xếp cục bộ gom các khóa của mỗi bucket
  lại với nhau bên trong ô, nên một dải dài bằng bucket của ô: khoảng
  256c / 2^r khóa với khóa ngẫu nhiên.
- **Kích thước bảng**: 2^r x ceil(n / 256c) phần tử, được ghi, scan và đọc mỗi lượt. Ở
  r = 8, c = 1, số phần tử đó bằng số khóa.
- **Bộ nhớ chia sẻ**: 256c x 8 byte cho ô cộng các mảng scan và bucket; ở c = 8
  riêng ô đã là 16 KiB. Chỗ nó cạn phụ thuộc vào thiết bị của bạn (`python3 labs/pmpp.py
  doctor`).

Benchmark bắt đầu ở n = 2^22 và nhân đôi n khi mốc so sánh còn mất dưới 20 ms (trong phạm vi
một phần tám bộ nhớ của thiết bị). Nó đo thời gian mốc so sánh (`onebit_global`), phép sắp xếp của bạn
(`local`) ở r = 1, 2, 4, 6, 8 với c = 1 và ở c = 1, 2, 4, 8 với r = 4 và r = 8,
bỏ qua những gì thiết bị không chạy được, và `std::stable_sort` của cùng các cặp trên một luồng
CPU với `std::chrono` (một phép đo phía host). Mỗi hàng là kết quả tốt nhất trong 5 lần chạy sau một
lần khởi động, trên khóa ngẫu nhiên toàn dải, và mọi kết quả được đo đều được kiểm tra trước khi báo cáo.
Thời gian là thời gian thiết bị của command buffer trên Metal và thời gian thực quanh các lần khởi chạy
và lần đồng bộ hóa cuối cùng trên CUDA và HIP. "Counted GB/s" là lưu lượng được đếm, không phải được đo:
32 byte mỗi cặp mỗi lượt, chia cho thời gian, bỏ qua bảng và phép scan. Các con số là phép đo trên
máy của bạn, không phải mục tiêu.

## Tính khả chuyển

- HIP: lời giải tham chiếu scan trong bộ nhớ chia sẻ, không dùng intrinsic warp và không có lane mask.
  `extern __shared__` giữ nguyên cách viết (`notes/today/ch05-facts.md` B4). Chưa được biên dịch hay
  chạy trên phần cứng AMD ở đây: unverified.
- Metal: tổng tiền tố (prefix sum) SIMD group cần GPU family Apple7, mà mọi máy Mac Apple silicon đều đạt
  (infra-facts C2). Các kernel không bao giờ giả định độ rộng SIMD group, và phép sắp xếp cục bộ xếp hạng
  các luồng của nó theo SIMD group và lane ở mọi chỗ vị trí phải theo thứ tự của phép scan, vì
  việc chia một threadgroup thành các SIMD group là do bản cài đặt quyết định (ch04-facts C2). Phép scan
  được cung cấp (`shared-g11/scan.metal`) cũng làm như vậy. Metal không tài liệu hóa sự tiến triển
  giữa các threadgroup (C4), đó là lý do mỗi lượt là ba dispatch chứ không phải một phép sắp xếp
  một lượt có look-back.
- Các thư viện chọn gì (A4): radix sort của CUB dùng chữ số 8 bit trong đường đi một lượt
  (Onesweep) và chữ số 7 bit trong đường đi nhiều lượt cho khóa rộng hơn một byte, với một
  thiết lập số khóa mỗi luồng theo từng kiến trúc; rocPRIM dùng 8 bit trong 374 trên 384 cấu hình
  Onesweep đã tinh chỉnh (B3); radix sort MPS của PyTorch dùng 8 bit cho khóa 16 và 32 bit
  (D6). Đó là tinh chỉnh của họ cho kernel của họ, không phải quy tắc cho kernel của bạn.
