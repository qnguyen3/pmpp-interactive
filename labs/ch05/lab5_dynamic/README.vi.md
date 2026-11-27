> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 5.5: độ rộng ô chọn lúc chạy

Sách: mục 5.6 (pp. 116-118), Fig. 5.14. Hình này có vài lỗi như in; bài học liệt kê chúng.

Hãy viết kernel phân ô, có kiểm tra biên, với độ rộng ô là một đối số thay vì một `#define`, dùng
bộ nhớ chia sẻ (shared memory) có kích thước động:

- CUDA: một mảng `extern __shared__` không khai báo kích thước, được tự tay chia thành hai ô, với
  kích thước tính bằng byte được truyền làm tham số khởi chạy thứ ba.
- Metal: một tham số kernel `threadgroup float*` gắn với `[[threadgroup(0)]]`, được host đặt kích
  thước bằng `setThreadgroupMemoryLength:atIndex:`.

Chương trình điều khiển (driver) hỏi thiết bị các giới hạn của nó, suy ra ô lớn nhất vừa được, và
kiểm thử các độ rộng ô 4, 8, 16 và ô lớn nhất đó bằng một tệp nhị phân duy nhất.

- Sửa `work/kernel.cu` và `work/kernel.metal` + `work/launch.mm`.
- Kiểm thử: `python3 labs/pmpp.py run 5.5`
- Đo: `python3 labs/pmpp.py run 5.5 --bench`. Nó đo thời gian các độ rộng 512, 1024, 2048 và
  4096 ở mọi độ rộng ô là lũy thừa của hai mà vừa được. Đó là các độ rộng mà benchmark của lab 5.2
  có thể chốt lại (nó nhân đôi từ 512 cho đến khi kernel ngây thơ của nó đủ chậm để đo, tối đa
  4096), nên trang labs so sánh hai kernel ở độ rộng mà lab 5.2 đã dùng. Hãy chạy cả benchmark của
  lab 5.2, trên cùng backend.

Trước khi đo hiệu năng, hãy dự đoán: ở cùng độ rộng ô (16) và cùng độ rộng ma trận, kernel này
nhanh hơn, chậm hơn hay bằng kernel của lab 5.2, nơi độ rộng ô là một hằng số lúc biên dịch? Vì sao?
