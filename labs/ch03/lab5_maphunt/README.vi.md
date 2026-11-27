> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 3.5: săn lỗi ánh xạ

Sách: các mục 3.2 và 3.3 (pp. 48-59). Kernel không phải là một đoạn mã của sách. Nó được dựng từ hai
thứ có trong sách: ánh xạ 3D từ luồng sang dữ liệu (p. 55) và vòng lặp có cắt biên cùng biến đếm của
kernel làm mờ (Fig. 3.8, p. 57), giới hạn trên một trục.

`work/` chứa một phép lấy trung bình theo cửa sổ dọc trục chuỗi (seq) của một tensor float
`[batch, seq, hidden]`:

    out[b][t][h] = mean of in[b][s][h] over s = t-r .. t+r, for the s that exist

Thứ tự hàng (row-major), batch đứng đầu. Hidden nằm trên x, seq trên y, batch trên z, theo các khối
(block) 8 x 8 x 4 luồng (thread). Nó biên dịch được, và có **bốn lỗi cài sẵn độc lập**. Hãy tìm và
sửa cả bốn.

- Sửa `work/kernel.cu` (kernel `seqMeanKernel` và lần khởi chạy `launchSeqMean` của nó), và
  `work/kernel.metal` + `work/launch.mm` (`seq_mean`, `encodeSeqMean`). Cùng bốn lỗi nằm trong tệp
  CUDA và trong cặp tệp Metal.
- Kiểm thử: `python3 labs/pmpp.py run 3.5 --backend emu`, rồi `python3 labs/pmpp.py run 3.5`.

## Trước lần chạy đầu tiên

Test đầu tiên là một khối lập phương. Mỗi test trong bốn test kế tiếp đi ra xa nó theo một hướng:

| Test | batch | seq | hidden | r |
|---|---|---|---|---|
| `cube` | 4 | 16 | 16 | 0 |
| `wide` | 4 | 6 | 20 | 0 |
| `tall` | 4 | 20 | 6 | 0 |
| `odd batch` | 5 | 16 | 16 | 0 |
| `window` | 4 | 16 | 16 | 1 |

Hãy đọc kernel và lần khởi chạy, và viết ra bạn dự đoán test nào trong năm test sẽ thất bại. Rồi
chạy và so sánh. `cube` qua được dù có đủ cả bốn lỗi. Đó chính là mục đích của nó: một test mà các
kích thước đều bằng nhau, đều là bội của hình dạng khối, và cửa sổ chỉ rộng một phần tử thì không
phân biệt được một ánh xạ đúng với ánh xạ này. Các test tên `mixed` sau năm test đó là các tổ hợp,
và qua được khi cả bốn lỗi đã được sửa.

## Phương pháp

1. Trình giả lập (emulator) trước: `python3 labs/pmpp.py run 3.5 --backend emu`. Sửa một chỗ, chạy
   lại. Các lỗi thay đổi khi bạn đi tiếp.
2. Hãy biết mỗi loại lỗi trông thế nào ở đó:
   - **Một truy cập ra ngoài tensor.** Các buffer của trình giả lập đúng bằng tensor, nên
     AddressSanitizer dừng lần chạy ở truy cập đầu tiên như vậy. Báo cáo rất dài. Hãy đọc ba thứ gần
     đầu báo cáo: `READ` hoặc `WRITE`, khung đầu tiên (`kernel.cu:` và một số dòng), và câu nói địa
     chỉ nằm trước hoặc sau buffer bao nhiêu byte. Các test sau điểm dừng không được chạy.
   - **Một truy cập vào sai phần tử bên trong tensor.** Địa chỉ hợp lệ, nên AddressSanitizer không
     nói gì. Bạn nhận được các giá trị sai, và thông báo của test cho biết phần tử sai đầu tiên và
     bao nhiêu phần tử không luồng nào ghi.
   - **Một sai sót trong mã host.** Lần khởi chạy là C++ thông thường tính vài số nguyên. Không công
     cụ nào gắn cờ một lưới (grid) sai hình dạng. Bạn chỉ thấy tác động của nó lên đầu ra.
3. Mỗi test chạy dưới ba thứ tự luồng. Một kernel đúng cho cùng đầu ra dưới cả ba, vì mỗi phần tử
   chỉ có một luồng ghi. Nếu một test báo cáo các giá trị sai khác nhau giữa các thứ tự, kết quả phụ
   thuộc vào luồng nào chạy cuối: có hai luồng đang ghi cùng một phần tử.
   `python3 labs/pmpp.py run 3.5 --backend race` chạy các luồng đồng thời dưới ThreadSanitizer, công
   cụ này báo cáo một cặp như vậy kèm dòng mã nguồn (`python3 labs/pmpp.py doctor` cho biết backend
   `race` có dựng được với trình biên dịch của bạn hay không).
4. Hãy dùng các kích thước. Một test thất bại đã đi ra xa `cube` theo một hướng, và hướng đó là thứ
   lỗi cần để lộ ra.
5. Rồi chạy trên GPU của bạn: `python3 labs/pmpp.py run 3.5`. Một lần chạy thông thường ở đó không
   nêu dòng mã nào. Trên device, chương trình điều khiển (driver) đặt các phần tử canh trước và sau
   mỗi tensor, nên một lần ghi lạc rơi vào vùng nhớ mà chương trình điều khiển sở hữu và được đếm sau đó. Với mỗi lỗi, hãy
   ghi lại lần chạy trên GPU cho bạn thấy gì: một giá trị sai, một phần tử chưa bao giờ được ghi,
   hay chỉ một lần ghi vào các phần tử canh. Một chương trình thực tế không có phần tử canh. Ở đó,
   một lỗi thuộc loại cuối cùng sẽ ghi vào bất cứ thứ gì nằm sau tensor, và mọi kết quả bạn kiểm tra
   vẫn đúng.

Không có gì để đo hiệu năng ở đây.
