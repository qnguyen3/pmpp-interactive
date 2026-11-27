> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 15.5: buffer kép và đường ống phần mềm

Sách: mục 15.8 (pp. 364-368), Figs. 15.12 đến 15.14. Mốc dự án M3, giai đoạn 5 trên 5: giai
đoạn cuối cùng trong cách bố trí của sách, nên hàng `stage` của lab này là bậc trên cùng của
chiếc thang bắt đầu từ lab 5.3. Lab 15.6 mới chính là mốc dự án.

`work/` chứa lời giải của lab 15.4, đã đổi tên (`gemmPipelinedKernel`, `launchGemmPipelined`,
`launchGemmPipelinedWidth`, `gemm_pipelined`, `encodeGemmPipelined`), với vòng lặp ô bị lấy
ra. Hãy đặt nó trở lại sao cho vòng lặp `tile` tính với cặp `tile` trong khi nạp cặp
`tile + 1` vào một buffer thứ hai, với một rào chắn (barrier) mỗi vòng lặp thay vì hai:

- hai cặp ô chia sẻ, `A_s[2][...]` và `B_s[2][...]` (TODO 1);
- một phần mở đầu nạp cặp 0 và chờ; một vòng lặp trên `tile < numTiles - 1` chạy `mm` trên
  cặp `tile`, rồi nạp cặp `tile + 1` vào buffer còn lại, rồi chờ; một phần kết chạy `mm` trên
  cặp cuối cùng (TODO 2).

Fig. 15.14 là một khung sườn: các đối số của nó bị lược bỏ, các buffer không được khai báo, và
các kiểu con trỏ của nó không khớp với các ô một chiều của Fig. 15.3 (`notes/ch15.md`,
"Unclear" 8). Sách hoán đổi một con trỏ hiện tại và một con trỏ kế tiếp ở cuối mỗi vòng lặp;
chọn buffer của một cặp theo tính chẵn lẻ của nó, `tile & 1`, như lab 6.5 làm, là cùng một
sự luân phiên. Hai trường hợp hình vẽ để lại cho bạn: `numTiles - 1` làm gì khi K bằng 0 (một
số đếm không dấu sẽ tràn vòng), và phần kết đọc buffer nào.

## Tệp

- `work/kernel.cu` và `work/kernel.metal`: các buffer kép và vòng lặp ô.
  `work/launch.mm` đã hoàn chỉnh: đường ống thay đổi kernel, không thay đổi phần khởi chạy.

## Phương pháp

1. Trước khi viết bất cứ thứ gì, hãy trả lời từ Fig. 15.12: trong hai rào chắn của lab 15.4,
   cái nào bảo vệ một phụ thuộc thật và cái nào bảo vệ một phụ thuộc giả? Vì sao một luồng
   chạy trước không bao giờ ghi đè lên một cặp mà các luồng lân cận vẫn đang đọc, kể cả hai
   vòng lặp sau?
2. `python3 labs/pmpp.py run 15.5 --backend emu`, rồi `--backend race`, rồi phần còn lại.

## Các bài kiểm thử kiểm tra gì

Các bài kiểm thử của lab 15.4, ở cả hai độ rộng. Trên GPU, mỗi trường hợp được khởi chạy 10
lần với C được điền lại NaN trước mỗi lần, và một lần thất bại sẽ cho biết kết quả sai ở bao
nhiêu lần khởi chạy. `(128, 8, 128)` là một cặp ô duy nhất (vòng lặp chạy không lần nào) và `(128, 24, 128)`
là ba cặp. `(5, 0, 9)` không có cặp ô nào: C phải ra toàn số 0. Nó chạy dưới một bộ canh giờ
30 giây, nên lỗi tràn vòng không dấu thất bại ở đó với lý do đó thay vì chạy hàng giờ. Cả hai
dạng của lỗi này đã được kiểm tra là tự thất bại: một `numTiles` không dấu không có điều kiện
chặn thất bại trên `emu` và Metal qua bộ canh giờ; một `numTiles` có dấu không có điều kiện
chặn, mà phần kết khi đó đọc các buffer không ai nạp, thất bại trên `emu` (rác trong C) và đã
qua trên Metal, nơi các buffer đó tình cờ chứa số 0. Backend race chạy từng khối 256 luồng một
dưới ThreadSanitizer trên `(128, 24, 128)` và `(130, 9, 131)`.

Rào chắn bên trong vòng lặp đã được kiểm tra là tự thất bại khi bị bỏ đi:

- trên trình giả lập, mọi trường hợp có nhiều hơn một cặp ô đều thất bại, dưới cả ba thứ tự
  luồng; các trường hợp một cặp thì qua, vì vòng lặp của chúng không bao giờ chạy;
- trên backend race, ThreadSanitizer báo race condition giữa lần ghi trong `loadTile` và lần
  đọc trong `mm`, kèm cả hai dòng mã nguồn;
- trên Metal, GPU duy nhất mà nó được kiểm tra, một số trường hợp chỉ thất bại ở vài lần trong
  10 lần khởi chạy, và một số qua cả 10.

Mã khởi đầu (starter) để nguyên thì build được nhưng thất bại: không có vòng lặp, C ra 0.

## Trước khi bạn đo hiệu năng

Dự đoán **mức tăng tốc so với lab 15.4 ở 2048 x 2048 x 2048.** Trước tiên hãy đọc hàng
`occupancy` (CUDA, HIP) hoặc hàng `pipeline` (Metal) của lab 15.4. Nếu từ hai khối trở lên vừa
trên một SM, các khối khác đã chồng lấp lần nạp của chúng với phần số học của khối bạn, và
phần lợi từ việc chồng lấp bên trong một khối sẽ nhỏ lại (p. 364). Cân nhắc thêm:

- Sự chồng lấp là việc của trình biên dịch (p. 367). Mã nguồn chỉ đặt các lần nạp của cặp
  `tile + 1` và phần số học của cặp `tile` trong cùng một vòng lặp, không có rào chắn ở giữa;
  không có gì buộc trình biên dịch phải xen kẽ chúng.
- Bộ nhớ chia sẻ mỗi khối tăng gấp đôi, lên 17,408 byte với phần đệm
  (`notes/today/ch15-facts.md` E1), điều này có thể làm giảm occupancy ở nơi bộ nhớ chia sẻ là
  thứ giới hạn nó.

Rồi hãy thử thứ tự còn lại bên trong vòng lặp: nạp cặp `tile + 1` trước, rồi chạy `mm` trên
cặp `tile`. Nó cũng đúng như vậy (các bài kiểm thử sẽ cho bạn biết) và nó thay đổi những gì
được giao cho trình biên dịch. Đo cả hai thứ tự trên thiết bị của bạn. Dù bạn thấy gì, đó là
phép đo của thiết bị và trình biên dịch của bạn, không phải một quy tắc.

## Vượt ra ngoài sách

Không có trong sách; từ `notes/today/ch15-facts.md`.

- CUDA (compute capability 8.0 trở lên) có thể sao chép từ bộ nhớ toàn cục sang bộ nhớ chia sẻ
  mà không đi qua thanh ghi (LDGSTS, `cp.async`), và Hopper thêm TMA cho cả ô.
  `cuda::memcpy_async` không bảo đảm LDGSTS, và khẳng định của sách rằng nvcc tự gộp một lần
  nạp vector và một lần ghi thành một không thể kiểm chứng được (A7, A8). Vòng lặp ở đây là
  dạng đồng bộ của cùng ý tưởng đó.
- HIP không ghi nhận cách sao chép bất đồng bộ dùng được nào vào LDS cho việc này; dạng phần
  mềm của lab này là dạng nên dùng ở đó (B5). Metal hoàn toàn không ghi nhận cách nào
  (`ch06-facts.md` C7).
- CUTLASS dùng buffer kép ở hai cấp: các ô chia sẻ, như ở đây, và các dải thanh ghi bên trong
  mỗi luồng (A12).
- Metal: thống kê pipeline của Xcode báo các luồng chờ ở rào chắn bao lâu
  ("Synchronization (barrier)", C3), nơi để xem bớt một rào chắn mang lại được gì.
