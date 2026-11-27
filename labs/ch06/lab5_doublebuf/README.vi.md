> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 6.5: buffer kép, với một trình phát hiện race condition theo dõi

Sách: mục 6.7 (pp. 146-147), áp dụng cho kernel phân ô của mục 5.4 (Fig. 5.9, p. 108) mà bạn đã
viết ở lab 5.2.

`work/kernel.cu` và `work/kernel.metal` chứa phép nhân ma trận phân ô của lab 5.2 với rào chắn
(barrier) thứ hai bị xóa. Nó biên dịch được, và nó sai. Hãy làm nó đúng với **một rào chắn mỗi giai
đoạn**, mà không đặt lại rào chắn thứ hai: cho mỗi toán hạng hai ô, để các giai đoạn luân phiên giữa
hai cặp ô, giữ một rào chắn sau các lần nạp và không có rào chắn nào sau tích vô hướng. Ma trận
vuông có độ rộng là bội của `TILE_WIDTH`, như ở lab 5.2.

- Sửa `work/kernel.cu` (`doubleBufferedMatmulKernel`) và `work/kernel.metal`
  (`double_buffered_matmul`). Các lần khởi chạy, `launchDoubleBufferedMatmul` trong `kernel.cu` và
  `encodeDoubleBufferedMatmul` trong `work/launch.mm`, là của lab 5.2 và đều đúng: buffer kép thay
  đổi kernel, không thay đổi lần khởi chạy.
- `provided/` chứa kernel hai rào chắn của lab 5.2 cho CUDA và Metal, đã đổi tên. Benchmark đo thời
  gian kernel của bạn so với nó. Bạn không sửa nó.
- `TILE_WIDTH` được định nghĩa bởi bản dựng (mặc định 16; 8 và 32 cũng được dựng cho benchmark;
  Metal nhận 16 và, ở chỗ thiết bị nhận threadgroup 32 x 32, nhận 32 lúc chạy). Đừng viết cứng 16.

## Phương pháp

1. **Trước khi thay đổi bất cứ điều gì**, hãy chạy mã khởi đầu (starter) trên backend race:

       python3 labs/pmpp.py run 6.5 --backend race

   Các luồng (thread) giả lập chạy đồng thời dưới ThreadSanitizer. Báo cáo của nó nêu tên hai dòng
   của `work/kernel.cu`: một lần ghi vào một ô, và phép nhân-cộng đọc các ô. Hãy viết ra, bằng lời
   của bạn, luồng nào làm gì, theo thứ tự nào, để báo cáo đó có thể xảy ra, và đó là loại nào trong
   hai loại phụ thuộc của mục 6.7. (`python3
   labs/pmpp.py doctor` cho biết trình biên dịch của bạn có dựng được backend race hay không.)
2. `python3 labs/pmpp.py run 6.5 --backend emu`. Mã khởi đầu đạt ở độ rộng 16 và thất bại ở 32 và
   64, dưới cả ba thứ tự luồng. Hãy giải thích trường hợp độ rộng 16 trước khi đi tiếp.
3. Sửa `work/kernel.cu`. Chạy `--backend race` và `--backend emu` cho đến khi cả hai đều sạch.
4. Thực hiện cùng thay đổi đó trong `work/kernel.metal`, rồi chạy `python3 labs/pmpp.py run 6.5`
   trên mọi backend bạn có. Trên GPU, mỗi trường hợp được khởi chạy 10 lần trên cùng đầu vào, với
   đầu ra được điền lại NaN trước mỗi lần khởi chạy, và một trường hợp thất bại cho biết nó sai
   trong bao nhiêu lần khởi chạy.
5. Trả lời TODO 4 trong chú thích: vì sao một luồng chạy trước không thể ghi đè lên một ô mà các
   luồng lân cận vẫn đang đọc, kể cả hai giai đoạn sau?

Vòng lặp của lab 5.2 ghi trước và đọc sau, nên rào chắn bảo vệ sự phụ thuộc giả (ghi sau đọc) là rào
chắn thứ hai của nó. Trong vòng lặp minh họa của sách ở p. 146, lần đọc đến trước, và ở đó nó là rào
chắn thứ nhất. Sách hoán đổi hai con trỏ buffer ở cuối mỗi vòng lặp (p. 147); chọn cặp ô theo số
giai đoạn cũng làm được cùng việc đó.

## Các bài kiểm thử bắt được gì

Mỗi trường hợp sau đã được kiểm tra là tự nó thất bại, trên trình giả lập (giá trị sai dưới mọi thứ
tự luồng), trên backend race (một báo cáo ThreadSanitizer cùng với giá trị sai) và trên Metal:

- **Mã khởi đầu**: một cặp ô và không có rào chắn thứ hai. Một luồng làm xong tích vô hướng của nó
  sẽ nạp các phần tử của giai đoạn tiếp theo đè lên những vị trí mà các luồng lân cận chưa đọc (ghi
  sau đọc). Trên trình giả lập, nó thất bại từ hai giai đoạn trở lên; với một giai đoạn thì không có
  giai đoạn tiếp theo nào để ghi đè lên gì cả.
- **Hai cặp ô, không có rào chắn nào**: một luồng đọc những vị trí mà các luồng lân cận chưa ghi
  (đọc sau ghi). Thất bại ngay cả với một giai đoạn.
- **Rào chắn đặt sau tích vô hướng thay vì sau các lần nạp**: cùng lỗi đọc sau ghi. Không gì buộc
  một luồng chờ các lần nạp của luồng lân cận trước khi nó đọc.
- **Tích vô hướng đọc cặp `(ph + 1) & 1`**: nó đọc cặp mà các giai đoạn khác nạp, không phải cặp mà
  giai đoạn này đã nạp.

Trên GPU, một số trường hợp này đạt ở một số lần khởi chạy. Các trình giả lập thì không để chúng đạt.

## Một lần chạy sạch là bằng chứng, không phải lời chứng minh

Backend race chỉ xem xét những cách lập lịch đã xảy ra, và ThreadSanitizer báo mỗi cặp dòng mã
nguồn một lần mỗi lần chạy. Ba thứ tự luồng của trình giả lập là ba cách lập lịch, được chọn để khắc
nghiệt. Mười lần khởi chạy trên GPU là mười cách lập lịch. Một kernel đạt tất cả chúng vẫn chưa được
chứng minh là đúng. Lập luận trong TODO 4 mới là thứ chứng minh điều đó.

## Trước khi bạn đo hiệu năng

    python3 labs/pmpp.py run 6.5 --bench

đo thời gian kernel hai rào chắn được cung cấp và kernel của bạn trên cùng các ma trận, luân phiên,
ở mỗi độ rộng ô mà backend của bạn dựng (CUDA và HIP: 8, 16 và 32, dưới dạng các chương trình riêng;
Metal: 16, và 32 ở chỗ thiết bị nhận được). Độ rộng tăng từ 512 cho đến khi kernel hai rào chắn chạy
quá 50 ms, tối đa 4096. Chỉ trên thiết bị thật.

Hãy viết ra một dự đoán trước: **ở T = 16, bạn dự đoán mức tăng tốc bao nhiêu khi dùng một rào chắn
mỗi giai đoạn thay vì hai?** Hãy đưa ra một con số, và lý do. Hai điều cần cân nhắc: một giai đoạn
có bao nhiêu rào chắn so với T phép nhân-cộng mà mỗi luồng làm trong đó, và thay đổi này đã phải trả
gì (gấp đôi bộ nhớ chia sẻ mỗi khối, điều mà chương 5 cho thấy có thể làm giảm occupancy).

Rồi so sánh với mã thật. GEMM Metal dùng trong sản xuất của MLX giữ hai rào chắn mỗi bước K, chính là
vòng lặp "trước" của sách (`notes/today/ch06-facts.md`, D3). Dù phép đo của bạn nói gì, đó là phép
đo trên thiết bị của bạn và kernel này, không phải một quy luật.

## Ngoài sách

Không có trong sách; từ `notes/today/ch06-facts.md`.

- Vòng lặp hai buffer, một rào chắn chỉ cần rào chắn khối, nên cùng tệp `.cu` biên dịch được cho HIP
  (B8). Trên Metal, đó là `threadgroup_barrier(mem_flags::mem_threadgroup)` một lần mỗi giai đoạn
  (C7).
- Hai cặp ô `float` 32 x 32 là 16 KB, gấp đôi của lab 5.2. Chương trình điều khiển (driver) so sánh
  những gì kernel cần với những gì thiết bị báo cáo (`sharedMemPerBlock` trên CUDA và HIP, bộ nhớ
  threadgroup của pipeline so với `maxThreadgroupMemoryLength` trên Metal) thay vì giả định một
  giới hạn.
- Ngày nay NVIDIA công bố buffer kép chủ yếu cùng với các phép sao chép bất đồng bộ: ô tiếp theo
  được nạp vào một buffer trong khi các luồng tính trên buffer kia (A16). Đó là hướng chương 15 đưa
  ý tưởng này đi tiếp. AMD chỉ công bố phép sao chép bất đồng bộ vào LDS cho CDNA 3 và 4, qua một
  intrinsic của trình biên dịch (B8). Metal hoàn toàn không công bố phép sao chép bất đồng bộ nào
  vào bộ nhớ threadgroup (C7), nên trên Metal, dạng đồng bộ trong lab này là dạng bạn có.
