> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 3.4: mỗi phần tử, mỗi hàng hay mỗi cột một luồng

Sách: bài tập 1a và 1b (p. 64), trên kernel của Fig. 3.11 (p. 61).

Fig. 3.11 cho mỗi phần tử của P một luồng (thread) riêng. Bài tập yêu cầu hai thiết kế thô hơn: một
luồng tính cả một hàng đầu ra, và một luồng tính cả một cột đầu ra. Ma trận vuông `Width x Width`,
như trong bài tập.

Kernel theo từng phần tử và lần khởi chạy của nó được cung cấp sẵn trong tệp. **Phần của bạn:**
`rowKernel` với `launchRow`, và `colKernel` với `launchCol`, mỗi cái trên một lưới (grid) 1D gồm các
khối (block) 256 luồng có điều kiện chặn.

- Sửa `work/kernel.cu` (CUDA / HIP / trình giả lập).
- Sửa `work/kernel.metal` (`row_matmul`, `col_matmul`) và `work/launch.mm` (`encodeRow`,
  `encodeCol`). Hai kernel nhận một `[[thread_position_in_grid]]` vô hướng và lần dispatch là 1D.
- Kiểm thử: `python3 labs/pmpp.py run 3.4`   Bắt đầu với `--backend emu`: AddressSanitizer dừng ở
  truy cập đầu tiên ra ngoài một buffer và nêu dòng mã. `--backend race` chạy các luồng giả lập
  đồng thời và báo cáo hai luồng cùng lưu vào một phần tử.
- Đo: `python3 labs/pmpp.py run 3.4 --bench`

Mỗi kernel được kiểm thử và báo cáo riêng, nên mã khởi đầu (starter) chưa sửa qua được với kernel
được cung cấp và thất bại với hai kernel của bạn. Các độ rộng bao gồm những giá trị để phần lớn một
khối 256 luồng không có việc.

Benchmark đo thời gian cả ba kernel ở một độ rộng. Chương trình điều khiển (driver) chọn độ rộng đó:
nó tăng gấp đôi độ rộng cho đến khi kernel theo từng phần tử chạy đủ lâu để đo, và dừng sớm hơn nếu
thêm một lần gấp đôi sẽ đẩy kernel chậm nhất vượt quá nửa giây mỗi lần chạy. Nó in ra độ rộng đã
chọn và lý do.

Trước khi đo hiệu năng: hãy xếp hạng ba kernel từ nhanh nhất đến chậm nhất, và dự đoán mỗi hàng một
luồng chậm hơn mỗi phần tử một luồng bao nhiêu lần ở độ rộng của benchmark. Cả ba làm cùng lượng
phép tính. Chúng khác nhau ở số luồng chia sẻ lượng việc đó và hướng mà mỗi luồng ghi.

Bài tập 1c hỏi ưu và nhược điểm của từng thiết kế. Hãy viết câu trả lời của bạn khi đã có số liệu.
