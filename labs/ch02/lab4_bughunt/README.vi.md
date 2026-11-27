> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 2.4: săn lỗi, một hàm stub gần như chạy được

Sách: các mục 2.4 đến 2.6 (pp. 28-39): Fig. 2.10 (kernel), Fig. 2.13 (hàm host), khung bên lề về
kiểm tra lỗi (p. 32). Bản thân chương trình là của khóa học này.

`work/` chứa một chương trình hoàn chỉnh và biên dịch được. `scalePrefix` nhận một mảng host gồm `n`
số float, nhân `len` số đầu tiên với `s` trên device, và phải giữ nguyên chính xác phần đệm từ `len`
đến `n`. Đây là dạng áp dụng một phép toán lên phần hợp lệ của một batch có đệm.

- `work/kernel.cu` (CUDA / HIP / trình giả lập) có **bốn lỗi cài sẵn độc lập**: kernel
  `scalePrefixKernel` và hàm host `scalePrefix` đều nằm trong đó.
- `work/kernel.metal` + `work/launch.mm` (Metal) có **ba**. Kernel Metal được dispatch theo các
  threadgroup trọn vẹn, như backend Metal của llama.cpp làm, nên nó tự tính chỉ số của mình theo
  cách một kernel CUDA làm.

Mỗi test lấp đầy mảng bằng các giá trị giữa 1 và 2, gọi `scalePrefix` với `s = 3`, và kiểm tra toàn
bộ mảng: phần đầu được nhân, phần đệm không đổi đến từng bit. Các kích thước đến theo một thứ tự:
trước hết là các mảng có phần đệm sau phần đầu, rồi các mảng không có phần đệm, rồi (họ CUDA) một
lần khởi chạy mà device không chạy được.

Phương pháp:

1. Chạy `python3 labs/pmpp.py run 2.4 --backend emu` trước. Trình giả lập (emulator) chạy kernel
   dưới AddressSanitizer, công cụ này dừng ở truy cập đầu tiên ra ngoài một mảng device và nêu dòng
   mã.
2. Sửa một chỗ, chạy lại. Các lỗi thay đổi khi bạn đi tiếp: một lỗi cài sẵn có thể che giấu lỗi
   khác cho đến khi nó được sửa.
3. Rồi chạy trên GPU của bạn: `python3 labs/pmpp.py run 2.4`. Ghi lại những lỗi nào GPU tự cho bạn
   thấy và những lỗi nào nó che giấu.

Trước khi chạy bất cứ thứ gì trên GPU, hãy dự đoán: trong số các lỗi cài sẵn, bao nhiêu lỗi GPU sẽ
tự báo cáo, qua một mã lỗi hoặc một lần sập, thay vì một giá trị sai mà chỉ test so với bản tham
chiếu CPU mới lộ ra? Hãy viết con số đó ra, rồi so sánh với những gì trình giả lập đã báo cáo.

Không có gì để đo hiệu năng ở đây.

## Sau cuộc săn

Hãy đọc phần này khi mọi test đã qua. Nó nói về việc mỗi backend làm gì với những sai sót này, và
nó nêu tên chúng.

- **Các luồng vượt quá `len`.** Khi các luồng thừa vẫn còn rơi vào trong mảng, chúng nhân cả phần
  đệm, điều mà mọi backend đều thể hiện thành một giá trị sai. Khi `len` bằng `n` và không phải bội
  của kích thước khối (trường hợp `n = 1000`), chúng ghi vượt quá cuối mảng device. Trình giả lập
  dừng ở đó và nêu dòng kernel. GPU không được đảm bảo sẽ báo cáo điều này
  (`notes/today/ch02-facts.md`, các mục A6 và C5): trên GPU, trường hợp đó có thể vẫn qua.
  Lỗi này vô hình chừng nào số khối còn bị làm tròn xuống, vì khi đó không có luồng nào thừa.
- **Không khối nào.** Với số khối làm tròn xuống, một phần đầu ngắn hơn một khối sẽ không khởi chạy
  gì cả. Trình giả lập từ chối lần khởi chạy đó vì cấu hình không hợp lệ. HIP ghi trong tài liệu
  một lỗi cho trường hợp này (mục B4). Tài liệu của NVIDIA không nói CUDA làm gì (mục A8). Trên
  Metal, một lần dispatch không threadgroup nào đã không chạy gì và không báo cáo gì khi lab này
  được dựng; tệp facts không có tuyên bố nào của Apple về điều này. Vì vậy các test chỉ chấp nhận
  một kết quả: các giá trị đúng và giá trị trả về 0.
- **Số byte mà thực ra là số phần tử.** Ba phần tư mảng không bao giờ đến được device. Thứ mà kernel
  tìm thấy ở đó thay vào phụ thuộc vào backend: mẫu điền của trình giả lập, bất cứ thứ gì bộ nhớ
  đang chứa trên CUDA (`cudaMalloc` không xóa nó, mục A2), số không trên Metal
  (`newBufferWithLength` xóa nó, mục C2).
- **Lần khởi chạy không được kiểm tra** (chỉ họ CUDA). Một câu lệnh khởi chạy không có giá trị trả
  về. Mẫu của NVIDIA và AMD là đọc trạng thái lỗi ngay sau đó (các mục A6 và B2). Trên trình giả
  lập, `cudaDeviceSynchronize` theo sau không nói gì về một lần khởi chạy bị từ chối, nên phiên bản
  không kiểm tra trả về 0 và test cuối cùng thất bại. Việc lời gọi đồng bộ hóa của CUDA hay HIP có
  báo cáo sự từ chối hay không không được nêu trong tài liệu chúng tôi đã kiểm tra, nên trên GPU
  test đó có thể qua dù lỗi vẫn còn. Metal không có thứ tương ứng: các lời gọi dispatch của nó
  không trả về gì và không có trạng thái lỗi nào để đọc (mục C5), đó là lý do phiên bản Metal có ba
  lỗi.

## Trạng thái của lời giải tham chiếu

verified on Metal, emulator-checked, front-end-checked. unverified trên CUDA và trên HIP: chưa có
máy nào có một trong hai chạy nó. Mỗi lỗi cài sẵn, khi được đặt riêng lẻ trở lại vào lời giải, đều
đã được chứng minh là thất bại: cả bốn trên trình giả lập, ba lỗi Metal trên Metal.
