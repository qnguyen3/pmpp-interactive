> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 5.4: săn race condition

Sách: các mục 5.4 và 5.5.

`work/kernel.cu` và `work/kernel.metal` chứa một phép nhân ma trận phân ô, có kiểm tra biên, với
**ba lỗi độc lập**. Nó biên dịch được. Hãy tìm và sửa cả ba.

Phương pháp:

1. Chạy `python3 labs/pmpp.py run 5.4 --backend emu` trước. Các thứ tự luồng của trình giả lập
   (emulator) là tất định, nên một lỗi đồng bộ hóa thất bại mọi lần, và AddressSanitizer dừng ở
   lần truy cập ngoài biên đầu tiên và nêu tên dòng đó.
2. Sửa một chỗ, chạy lại. Các thất bại thay đổi khi bạn tiến lên.
3. Để có ý kiến thứ hai về lỗi đồng bộ hóa, `python3 labs/pmpp.py run 5.4
   --backend race` chạy các luồng cùng lúc dưới ThreadSanitizer. Nó nêu tên hai dòng mã nguồn mà
   lần ghi và lần đọc va chạm nhau, cho biết chỗ thiếu một rào chắn (barrier). Nó chỉ thấy các va
   chạm đã xảy ra trong lần chạy đó, nên một báo cáo sạch không phải là bằng chứng.
4. Rồi chạy trên GPU của bạn: `python3 labs/pmpp.py run 5.4`. Ghi lại lỗi nào GPU đã cho bạn thấy
   và lỗi nào nó che đi.

Ở đây không có gì để đo hiệu năng.
