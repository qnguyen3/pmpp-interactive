> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 12.5: bộ lọc ổn định tại chỗ trong một lượt (tùy chọn)

Sách: các bài tập 2, 3 và 4 (p. 302) và mục 12.7 (p. 300, Fig. 12.11), dựa trên chỉ số khối
động và phép nhìn lại một bước (single look-back) của chương 11 (Figs. 11.17 và 11.18,
pp. 278-281). Một mục đứng riêng như (A9) dưới đây nằm trong `notes/today/ch12-facts.md`.

Chỉ dành cho họ CUDA: không có phiên bản Metal (xem "Vì sao không có phiên bản Metal").

Hãy viết bộ lọc ổn định của lab 12.3 (giữ một khóa khi `key < threshold`, các khóa được giữ
theo thứ tự đầu vào) thành một kernel đọc mỗi khóa đúng một lần, và làm cho nó chạy được tại
chỗ: chương trình điều khiển (driver) gọi nó với `output == input` cũng như với một đầu ra
riêng.

`stableFilterSinglePassKernel`, một khối cho mỗi ô (tile) gồm `BLOCK_DIM * coarse` khóa:

1. Luồng 0 lấy chỉ số của khối từ `*blockCounter` bằng một `fetch_add` relaxed ở phạm vi
   device và chia sẻ nó qua bộ nhớ chia sẻ (shared memory). Khối dùng chỉ số này, không dùng
   `blockIdx.x`.
2. Khối nạp toàn bộ ô của nó vào bộ nhớ chia sẻ động theo các vòng được hợp nhất (coalescing).
   Rào chắn (barrier): mọi lần đọc đầu vào của khối này đã xong.
3. Từng vòng một, nó nén gọn các khóa được giữ, theo thứ tự, về đầu của chính mảng chia sẻ đó:
   một lần bỏ phiếu warp cho mỗi lane được giữ thứ hạng của nó, một phép scan trên tổng của các
   warp cho phần còn lại, như phần ghi rải qua vùng đệm của lab 12.3. Giờ khối biết số đếm của
   nó.
4. Nhìn lại một bước, như Fig. 11.17: luồng 0 chờ cho đến khi `flags[bid - 1]` được bật (một
   acquire), đọc `inclusive[bid - 1]`, số khóa được giữ của mọi khối trước đó, lưu
   `inclusive[bid]`, rồi bật `flags[bid]` (một release). Rào chắn.
5. Khối ghi các khóa được giữ của nó tại độ lệch đó, các luồng liên tiếp trên các khóa liên
   tiếp. Khối cuối cùng ghi `*outputSize`.

`launchStableFilterSinglePass(input, output, n, threshold, outputSize, blockCounter, flags,
inclusive, coarse)` khởi chạy nó. Chương trình điều khiển đặt `*blockCounter` và các cờ về 0
trước mỗi lần gọi; ngoài điều đó, không phần nào trong kernel được giả định chúng bắt đầu từ 0.
Bên gọi bảo đảm `n >= 1`: không có khóa nào thì không có khối cuối để ghi số đếm, và hàm khởi
chạy sẽ khởi chạy không khối nào, điều HIP từ chối (`notes/today/ch02-facts.md` B4). Không dùng
`__restrict__`: `input` và `output` có thể là cùng một mảng.

## Tệp

- Sửa `work/kernel.cu` (CUDA, HIP và cả hai trình giả lập). Mặt nạ lane, `maskPopc`, các hàm
  trợ giúp cho lane và `spinPause()` được cung cấp ở đầu tệp: hãy gọi `spinPause()` một lần mỗi
  vòng của vòng lặp chờ. Nó ngủ một chút trên GPU (`__nanosleep` trên CUDA, ch11-facts A7;
  `__builtin_amdgcn_s_sleep(1)` trên AMD, nơi rocPRIM phát hiện các chip MI100 đời đầu bị treo
  trong một vòng quay trần, ch11-facts B3) và không làm gì trong các trình giả lập.
- `BLOCK_DIM` là 256, và 64 trên backend race (bên dưới).

## Lệnh

- Logic: `python3 labs/pmpp.py run 12.5 --backend emu`.
- Thứ tự: `python3 labs/pmpp.py run 12.5 --backend race`.
- Trên một thiết bị CUDA hoặc HIP: `python3 labs/pmpp.py run 12.5`, và `--bench` để đo.
- Lời giải tham chiếu (solution): thêm `--solution`.

Nếu một trường hợp chạy lâu hơn 120 giây, chương trình điều khiển dừng lại và báo cáo: gần như
chắc chắn một khối đang chờ một cờ không bao giờ được bật (sai chỉ số, một cờ không bao giờ được
lưu, bộ đếm không được dùng). Không có gì treo mãi mãi.

## Vì sao nó đúng khi chạy tại chỗ: hai điều kiện

Quy tắc của mục 12.7: không bao giờ ghi vào một vị trí trước khi khóa từng nằm ở đó đã được đọc.
Một khóa chỉ dịch chuyển xuống chỉ số thấp hơn, nên các lần ghi của một khối rơi vào ô của chính
nó hoặc các ô trước đó. Bên trong khối, rào chắn sau lần nạp sắp các lần đọc trước các lần ghi.
Giữa các khối, lập luận của sách là chuỗi scan của Fig. 12.11: với các khối i < j, read(i) đến
trước scan(i), scan(i) đến trước scan(j), và scan(j) đến trước write(j). Hai điều phải đúng thì
chuỗi đó mới có ý nghĩa trên phần cứng thật.

**1. Chuỗi phải sắp thứ tự bộ nhớ, không chỉ thời gian (A9).** Lệnh lưu cờ phải là một release
và lệnh nạp cờ là một acquire, và lần đọc ô của mọi luồng phải đến trước release, đó chính là
điều rào chắn sau lần nạp mang lại cho bạn. Kernel select của CUB nói đúng điều này trong một
chú thích và chuyển phép nhìn lại của nó từ relaxed sang acquire/release khi phép chọn có thể
diễn ra tại chỗ (A9). Ở đây cờ cũng bảo vệ `inclusive[]`, một mảng thường, nên acquire và
release là cần thiết ngay cả khi không tại chỗ: đó là mẫu truyền thông điệp của NVIDIA
(ch11-facts A5). Khi chạy tại chỗ, chúng bảo vệ cả các khóa.

**2. Việc chờ một khối khác phải được phép kết thúc (A10).** Mô hình thực thi của CUDA giờ nói
rằng một khi một luồng của một khối đã tiến triển, mọi luồng trong khối đó rốt cuộc đều tiến
triển. Một khối đã lấy chỉ số b - 1 từ bộ đếm là đã tiến triển, nên khối chờ nó đang chờ một
khối sẽ hoàn tất. Nếu chờ `blockIdx.x - 1` thay vào đó thì không có gì hứa rằng khối ấy thậm chí
đã bắt đầu; với nhiều khối hơn số khối mà thiết bị chứa được cùng lúc, điều đó có thể gây deadlock. Đây
là lý do bước 1 dùng bộ đếm (lý do của sách, p. 280).

Trên HIP, AMD không ghi trong tài liệu quy tắc tiến triển nào cho các workgroup (B6, ch11-facts
B3). Chính hướng dẫn của họ nói rằng các khối bắt đầu theo thứ tự tăng dần trên phần cứng hiện
tại nhưng API không bảo đảm điều đó, và rocPRIM đã cần các cách khắc phục riêng cho từng kiến
trúc cho cả thứ tự khối lẫn vòng quay chờ. Bộ đếm là lựa chọn phòng thủ ở đó. **HIP: unverified**
cho đến khi một `reports/verify-*.md` từ một máy AMD cho thấy lab này vượt qua.

## Vì sao không có phiên bản Metal

Apple không ghi trong tài liệu sự tiến triển nào giữa các threadgroup (C4), và một kỹ sư Apple,
khi trả lời trên diễn đàn nhà phát triển của Apple, đã gọi việc chờ theo chuỗi của một phép scan
một lượt là hành vi không xác định (ch11-facts C5). Cũng không có rào chắn toàn lưới hay
cooperative launch để dựa vào. Thiết kế tại chỗ an toàn theo tài liệu trên Metal là một
threadgroup đi qua dải của nó theo thứ tự, chỉ với các rào chắn threadgroup, đó chính là kernel
của lab 12.4 cho mỗi hàng. Bộ lọc nhiều lần dispatch có tính khả chuyển là lab 12.3.

## Các trình giả lập có thể và không thể cho thấy gì

- `emu` chạy các khối lần lượt theo thứ tự chỉ số, nên mọi khối đều thấy cờ của khối trước đã
  được bật sẵn: phép nhìn lại không bao giờ phải chờ, và rủi ro khi chạy tại chỗ không thể xuất
  hiện. Nó kiểm tra logic: chỉ số, phần nén gọn, các rào chắn trong một khối, kiểm tra biên (dưới
  AddressSanitizer).
- `race` chạy các khối 64 luồng, bốn khối một lúc, như các luồng hệ điều hành thật dưới
  ThreadSanitizer, với atomic thật và acquire, release thật. Ở đó nhiều khối chạy cùng lúc, nên
  một khối có thể tới phép nhìn lại trước khi khối nó chờ hoàn tất, và một truy cập vào
  `inclusive[]` hay vào các khóa mà không gì sắp thứ tự sẽ được báo cáo kèm cả hai dòng mã
  nguồn. Một lượt chạy sạch không chứng minh rằng không có race condition: chỉ những lịch chạy đã
  xảy ra được kiểm tra.
- Không cái nào nói gì về tốc độ, và vượt qua chúng không chứng minh kernel chạy được trên GPU.
  Các hàng CUDA và HIP trong báo cáo của `./verify.sh` mới làm được điều đó.

### Thí nghiệm tùy chọn: lấy đi thứ tự

Khi kernel của bạn đã vượt qua, hãy đổi lệnh nạp cờ và lệnh lưu cờ thành
`cuda::memory_order_relaxed` và chạy `python3 labs/pmpp.py run 12.5 --backend race`. Hãy dự đoán
trước điều gì sẽ được báo cáo. Khi lab này được xây dựng, ThreadSanitizer đã báo hai race
condition, cùng hai cái đó trong mỗi lần của bốn lượt chạy: lần đọc `inclusive[bid - 1]` của một
khối so với lần ghi nó của khối trước, và, khi chạy tại chỗ, lần ghi một khóa đầu ra của một khối
so với lần đọc khóa đầu vào từng nằm ở vị trí đó của một khối khác. Phần lớn trường hợp đáp án vẫn
ra đúng: trên trình giả lập, cũng như trên GPU, một thứ tự bị thiếu là một rủi ro mà một lượt chạy
có thể gặp hoặc không. Hãy đặt acquire và release trở lại.

## Các bài kiểm thử kiểm tra gì

- n = 1, 33, 257, 1000, 4099, 65537, 2^20 và 3 x 2^20 + 17 trên thiết bị; tới 65537 trên
  `emu` (65537 chỉ ở mức giữ một nửa); 1, 33, 257, 1000 và 2048 trên `race`.
- Coarse 1, 2 và 4. Mọi trường hợp đều chạy không tại chỗ và tại chỗ.
- Tỉ lệ được giữ 0, 0.5 và 1, cùng hai mẫu khiến các khối ghi đè lên đầu vào của các khối trước:
  chỉ giữ các khóa của khối 1 (mọi khóa của nó đều rơi vào vị trí của khối 0), và chỉ giữ các
  khóa của các khối lẻ.
- Số đếm chính xác, và `count` phần tử đầu tiên bằng bộ lọc ổn định trên CPU từng phần tử một.
  Khi không tại chỗ, mọi vị trí quá số đếm phải vẫn giữ giá trị canh mà chương trình điều khiển đã
  ghi. Khi tại chỗ, phần mảng quá số đếm là không xác định, như `remove_if` của Thrust để lại nó
  (A8), và không được kiểm tra.

## Trước khi bạn đo hiệu năng (một thiết bị CUDA hoặc HIP)

Hãy ghi ra một dự đoán (bài học yêu cầu nó dưới tên `ch12/labs/p-lab5-singlepass`): ở coarse 4
và giữ một nửa số khóa, thời gian của phần B lab 12.3 chia cho thời gian của kernel này, khi không
tại chỗ.

Mô hình: bản một lượt đọc mỗi khóa một lần, 4 + 4f byte mỗi khóa ở tỉ lệ được giữ f (6 ở mức giữ
một nửa), trong khi phần B lab 12.3 đọc nó hai lần, 8 + 4f (10). Đối trọng với điều đó, chuỗi nhìn
lại khiến việc chuyển giao của mọi khối phải chờ khối trước nó, và chuỗi dài bằng số khối.

Benchmark đo thời gian n = 2^24 ở các tỉ lệ được giữ 0.01, 0.5 và 1, coarse 1, 4 và 8, không tại
chỗ và tại chỗ. Mỗi lượt đo bắt đầu từ các khóa gốc và các cờ đã đặt về 0; phần đặt lại đó không
được tính giờ. Các hàng `singlepass` cho thời gian (tốt nhất trong 5 lần chạy) và GB/s trên lưu
lượng tối thiểu, `4n + 4 x kept` byte. Trang labs đặt chúng cạnh các hàng `staged` của lab 12.3
từ cùng thiết bị. Mọi kết quả được đo giờ đều được kiểm tra trước. Các con số là phép đo trên máy
của bạn, không phải mục tiêu.
