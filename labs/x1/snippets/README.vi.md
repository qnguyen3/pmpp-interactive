> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Snippets

Mã mà các bài học của module X1 hiển thị và không phải là một lab. Các tệp là mã nguồn thật để
bước kiểm tra front-end CUDA trong `./verify.sh` phân tích các tệp `.cu`, và để một bài học hiển
thị đúng tệp đã được kiểm tra chứ không phải một bản sao có thể lệch đi. Trình chạy lab không build
chúng. Các chương trình điều khiển (driver) trong `check/` chạy chúng; phép kiểm thử của mỗi driver
nằm trong `app/src/sim/`. Mỗi snippet một hàng. Mọi thứ ở đây là của khóa học này: lấy mẫu không có
trong sách.

| Tệp | Hiển thị ở | Cách | Driver | Kiểm thử |
|---|---|---|---|---|
| `philox.cu` | mục X1.7, "Mã nguồn": Philox4x32-10, cách ánh xạ bộ đếm và khóa của M5, và phép chuyển 24 bit sang u | nạp lúc chạy | `check/philox_kat.cu` | `app/src/sim/philox.test.ts` |
| `philox.metal` | mục X1.7, thẻ Metal của cùng bảng đó | nạp lúc chạy | `check/philox_metal_run.mm` | `app/src/sim/philox.test.ts` |
| `draw_search.cu` | mục X1.8: lần rút cuối cùng, một khối cho mỗi hàng, với hai rào chắn mà trình chạy từng bước cho bạn gỡ bỏ | nạp lúc chạy | `check/draw_check.cu` | `app/src/sim/draw.test.ts` |

Trang tóm tắt không hiển thị kernel nào của riêng nó, nên không có snippet cho trang tổng kết.

## `philox.cu` và `philox.metal`

Vòng, các hằng số và lịch khóa của Random123, được viết lại ở đây (không sao chép). `philox.cu` tạo
tích 32 x 32 ra 64 bit bằng một phép nhân `unsigned long long`, không cần intrinsic nào
(`__umulhi` cũng được: trình giả lập (emulator) của khóa học và stub front-end giờ chấp nhận nó).
`philox.metal` lấy nửa cao từ `mulhi` của MSL và nửa thấp từ tích 32 bit có tràn vòng
(x1-facts C4), như chính bản MSL của Random123. `kernel.metal` của lab X1.3 dùng dạng còn lại, một
tích `ulong`; cả hai đều qua cùng các vector đáp án đã biết.
Thông báo kiểu BSD của Random123 được chép lại trong cả hai tệp; các header của cuRAND theo giấy
phép của NVIDIA và không được dùng. Những gì phép kiểm thử kiểm tra:

- `philox.cu`, được dịch bởi `labs/tools/cu2emu.py` và chạy trên trình giả lập: ba vector đáp án đã
  biết 10 vòng của Random123 (x1-facts A4), tính bởi một kernel và trên host, và u cho 64 hàng
  (seed, stream, step), tất cả được so sánh với `app/src/sim/philox.ts`;
- `philox.metal`, biên dịch ở Metal 3.2 và 4.1 và chạy trên GPU của máy Mac bởi
  `check/philox_metal_run.mm`: cùng các vector đó và cùng 64 giá trị u.

Trạng thái: `philox.metal` là verified on Metal (được phép kiểm thử biên dịch và chạy ở đây);
`philox.cu` là emulator-checked và front-end-checked, unverified trên phần cứng CUDA và HIP.

## `draw_search.cu`

Lần rút của M5 đứng riêng: tổng tính bằng một phép rút gọn (reduction) khối hai giai đoạn (mục 10.8),
rồi các đoạn gồm `blockDim.x * 4` trọng số với một tổng gộp đang chạy, một phép scan khối trên đoạn
vượt ngưỡng (mục 11.4), và lane đầu tiên có tổng bao gồm lớn hơn hẳn u x total, tìm bằng một ballot
(mục 12.3), với phương án dự phòng là trọng số dương cuối cùng. Những gì phép kiểm thử kiểm tra:

- trên trình giả lập, theo cả ba thứ tự luồng, 27 hàng token so với `app/src/sim/draw.ts`;
- khi bỏ rào chắn A (biên dịch với một define), các token mà `huntDraw` của mô hình dự đoán cho
  lịch chạy theo thứ tự và lịch đảo ngược, một số trong đó sai;
- khi bỏ rào chắn B, các token đúng ở cả ba thứ tự của trình giả lập (trình giả lập không thấy được
  lần ghi sau đọc này giữa các warp), và trên biến thể race, một báo cáo ThreadSanitizer chỉ ra lần
  đọc tổng và lần ghi tổng của warp; khi có đủ cả hai rào chắn, không có báo cáo nào.

Các bản build emulator và race khiến `draw.test.ts` mất khoảng một phút. Mỗi phần được bỏ qua ở nơi
không tìm thấy trình biên dịch C++, `python3` hoặc ThreadSanitizer hoạt động được.

Trạng thái: emulator-checked và front-end-checked, unverified trên phần cứng CUDA và HIP. Không có
phiên bản Metal của snippet này; lab X1.3 chứa lần rút trên Metal (`labs/x1/lab3_sampler`).
