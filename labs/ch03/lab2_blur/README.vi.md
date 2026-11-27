> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 3.2: làm mờ ảnh

Sách: mục 3.3 (pp. 55-59), Fig. 3.8 (p. 57).

Hãy viết kernel làm mờ và lần khởi chạy của nó. Mỗi điểm ảnh đầu ra một luồng (thread), như lab 3.1,
nhưng giờ mỗi luồng chạy một vòng lặp: nó lấy trung bình các điểm ảnh đầu vào trong một vùng vuông
quanh vị trí của chính nó, bỏ đi phần vùng nhô ra ngoài mép ảnh.

- Sửa `work/kernel.cu` (CUDA / HIP / trình giả lập): `blurKernel` và `launchBlur`, với các khối
  (block) 16 nhân 16.
- Sửa `work/kernel.metal` + `work/launch.mm` (Metal). Lần khởi chạy dùng `dispatchThreadgroups`,
  hàm này nhận số nhóm nên làm tròn lên giống như một lưới (grid) CUDA. Vị trí mà Metal trao cho
  kernel là kiểu không dấu; các độ lệch bên trong một vùng thì không.
- Kiểm thử: `python3 labs/pmpp.py run 3.2`   Bắt đầu với `--backend emu`.
- Đo: `python3 labs/pmpp.py run 3.2 --bench`

Một điểm khác với trang sách: sách cố định bán kính vùng bằng một hằng số `BLUR_SIZE` định nghĩa bên
ngoài kernel. Ở đây bán kính là một đối số kernel `r`, để benchmark có thể thay đổi nó. Vùng có cạnh
2r + 1 điểm ảnh. (Chú thích ở dòng 09 của Fig. 3.8 gọi nó là "BLUR_SIZE x BLUR_SIZE box"; phần văn
bản ở p. 58 viết đúng.)

## Các test kiểm tra gì

Đầu ra của bạn được so sánh với một bản tham chiếu CPU, cho phép lệch một mức xám về mỗi phía.

- Kích thước và bán kính: 62 nhân 76 với r = 1, 2 và 3; 1 nhân 1; 1 nhân 9 và 9 nhân 1; 3 nhân 3
  với r = 3, khi vùng lớn hơn bức ảnh; 33 nhân 17 với r = 7; trên thiết bị thật thêm 1500 nhân 2000.
  Kích thước là số hàng nhân số cột, như sách viết.
- Một bức ảnh đồng nhất phải được trả về đồng nhất.
- Một lỗi được báo cáo bằng hai con số: số điểm ảnh biên sai (những điểm có vùng nhô ra ngoài mép)
  và số điểm ảnh bên trong sai. Biên sai mà bên trong đúng chỉ ra phép kiểm tra bên trong hoặc số
  bạn chia. Bên trong sai chỉ ra ánh xạ từ luồng sang điểm ảnh.
- Đầu ra được lấp đầy bằng một giá trị canh (sentinel) trước mỗi lần khởi chạy; một điểm ảnh vẫn
  còn giữ giá trị đó là chưa bao giờ được ghi.
- Trên trình giả lập (emulator), các buffer đúng bằng kích thước bức ảnh và bản dựng dùng
  AddressSanitizer, nên một truy cập ra ngoài buffer sẽ dừng lần chạy và nêu dòng mã. Trên thiết bị
  thật, các buffer lớn hơn bức ảnh và các byte phía sau đầu ra phải được trả về không thay đổi.
- Trường hợp đầu tiên là ngoại lệ của "đúng bằng kích thước": kernel được cho 62 hàng trên cùng của
  một buffer 66 hàng. Một phép đọc chạy quá mép phải của một hàng sẽ rơi vào đầu hàng kế tiếp, vốn
  là vùng nhớ hợp lệ, nên không công cụ kiểm tra biên nào phản đối được. Chỉ các giá trị mới cho
  thấy điều đó. Trong một buffer đúng kích thước, cùng sai sót đó chỉ bị bắt ở hàng cuối cùng.

## Trước khi đo hiệu năng

Benchmark làm mờ một bức ảnh lớn với r = 1, 2, 4 và 8.

Đi từ r = 1 lên r = 4, mỗi luồng đọc 81 giá trị thay vì 9. Hãy viết ra bạn dự đoán kernel sẽ chạy
lâu hơn bao nhiêu lần trên GPU của bạn, rồi chạy. Benchmark in ra tỉ số nó đo được.
