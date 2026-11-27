> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 15.7: săn lỗi trong một lớp tuyến tính nhanh

Sách: mục 15.6 và 15.8 (pp. 361-368), Fig. 15.12; Bài tập 2 (p. 370).

`work/kernel.cu` và `work/kernel.metal` chứa một lớp tuyến tính nhanh theo cách bố trí của dự
án, kernel của lab 15.6: `Y[batch x out] = X[batch x in] * W^T`, W lưu dạng (out, in) theo
thứ tự hàng (row-major). Nó có **ba lỗi cài sẵn độc lập**, giống nhau trong cả hai tệp. Nó
biên dịch được, và một số hình dạng qua. Hãy tìm và sửa cả ba. `work/launch.mm` là đúng.

Trước khi chạy bất cứ thứ gì, hãy đọc kernel và ghi ra, với mỗi lỗi bạn nghi ngờ, những
backend bạn kỳ vọng sẽ để lộ nó: `emu`, `race`, Metal, CUDA hoặc HIP. Sau đó so sánh. GPU của
bạn đã che giấu những lỗi nào?

Phương pháp (của lab 5.4):

1. `python3 labs/pmpp.py run 15.7 --backend emu` trước. Các thứ tự luồng của trình giả lập
   (emulator) là tất định, nên một lỗi đồng bộ hóa thất bại mọi lần, và AddressSanitizer dừng
   ở lần truy cập đầu tiên nằm ngoài một vùng cấp phát và chỉ ra dòng đó.
2. Sửa một thứ, chạy lại. Các thất bại thay đổi theo từng bước.
3. `python3 labs/pmpp.py run 15.7 --backend race` chạy các luồng cùng lúc dưới
   ThreadSanitizer. Nó chỉ ra hai dòng mã nguồn có truy cập va chạm nhau, cho biết chỗ thiếu
   một rào chắn (barrier). Nó chỉ thấy những va chạm đã xảy ra trong lần chạy đó, nên một báo
   cáo sạch không phải là bằng chứng.
4. Rồi trên GPU của bạn: `python3 labs/pmpp.py run 15.7`. Ghi lại những lỗi nó để lộ và những
   lỗi nó che giấu.

Các phép kiểm tra là của lab 15.6: cùng các hình dạng ở độ rộng warp 32 và 64, Y được điền
NaN, X và W được cấp phát đúng bằng số float của chúng, một dải bảo vệ sau Y mà không kernel
nào được ghi, và một tham chiếu CPU với một dung sai. Hãy đọc mỗi thông báo thất bại nói gì: hàng nào sai,
hình dạng nào thất bại và hình dạng nào qua, dải bảo vệ có thay đổi không. Trên GPU, mỗi
trường hợp được khởi chạy mười lần, Y được đặt lại NaN mỗi lần, và thông báo đếm bao nhiêu
lần khởi chạy bị sai.

Không có gì để đo hiệu năng ở đây.
