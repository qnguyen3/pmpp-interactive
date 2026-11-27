> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Snippets

Mã mà các bài học của chương 3 hiển thị để phục vụ một lời giải thích hoặc một câu hỏi, và không
phải là lab. Các tệp được giữ ở đây dưới dạng mã nguồn `.cu` (và `.metal`) thật để phép kiểm tra
front-end CUDA trong `./verify.sh` phân tích cú pháp mọi tệp `.cu`, và để bài học hiển thị đúng
tệp đã được kiểm tra chứ không phải một bản sao có thể bị lệch. Trình chạy lab không biên dịch hay
chạy chúng; `check.py` (bên dưới) chạy những tệp mà bài học gọi là emulator-checked.

Cách bài học dùng chúng:

- nạp lúc chạy bằng `useLabFile('ch03/snippets/<name>')`; hoặc
- sao chép vào một tệp section, với một unit test bên cạnh section đó sẽ thất bại nếu bản sao và
  tệp khác nhau, và ghim mọi dòng mà bài học nhắc đến theo số dòng. Hãy sửa test cùng với tệp.

| Tệp | Hiển thị ở | Cách dùng | Test trong `app/src/chapters/ch03/` |
|---|---|---|---|
| `vec_add_1d.cu`, `launch_1d.cu` | "Before you start", mục 3.1 | nạp | `S31.snippets.test.ts` (số dòng) |
| `launch_fig31.cu`, `launch_checked.cu` | mục 3.1 | nạp | `S31.snippets.test.ts` (số dòng) |
| `grayscale_kernel.cu`, `grayscale_kernel.metal` | mục 3.2, đoạn mã và trình chạy từng bước | sao chép | `S32.snippets.test.ts` |
| `grayscale_as_printed.cu` | mục 3.2, phần săn lỗi in ấn (erratum) | sao chép | `S32.snippets.test.ts` |
| `blur_kernel.cu` | mục 3.3, đoạn mã và trình chạy từng bước | sao chép | `S33.snippets.test.ts` |
| `matmul_kernel.cu` | mục 3.4, đoạn mã, trình chạy từng bước và lần khởi chạy | sao chép | `S34.test.ts` |
| `matvec_options.cu` | mục 3.4, bài tập 2 của sách (bốn phương án) | sao chép | `S34.test.ts` |
| `blur_unsigned.cu` | mục 3.3, "Signed arithmetic" (được mô tả, không hiển thị) | không | được `check.py` chạy |
| `scale_rows_defect.cu` | trang tổng kết, câu hỏi đầu tiên | sao chép | `Wrap.snippets.test.ts` |

Một số tệp cố ý sai hoặc kém. `grayscale_as_printed.cu` chứa các dòng mã của Fig. 3.4 đúng như
sách in, kể cả phép ép kiểu và các hằng số; nó phân tích cú pháp được, và không bao giờ được trở
thành nền tảng của một lab. `scale_rows_defect.cu` đọc một phần tử mà một luồng khác ghi; câu hỏi
của trang tổng kết nói về điều đó. `blur_unsigned.cu` giữ mọi chỉ số ở kiểu không dấu, nên hai mệnh
đề của nó luôn đúng; nó chạy được nhưng không nên sao chép theo. Trong `matvec_options.cu` chỉ có
`matvecOption2` là đúng.

## Chạy chúng: `check.py`

```sh
python3 labs/ch03/snippets/check.py            # build in a temporary directory
python3 labs/ch03/snippets/check.py --build DIR
```

Nó dịch từng snippet và chương trình điều khiển (driver) tương ứng trong `check/` bằng
`labs/tools/cu2emu.py`, biên dịch chúng theo cách trình chạy lab biên dịch backend `emu`
(bật AddressSanitizer, `-Wall -Wextra`), chạy mọi trường hợp dưới ba thứ tự luồng của trình giả
lập (emulator), và thất bại nếu bất cứ điều gì bài học khẳng định về snippet không đúng:

| Snippet | Điều được kiểm tra | Được khẳng định ở |
|---|---|---|
| `launch_checked.cu` | `(32, 32, 2)` bị từ chối, buffer không bị động đến, trạng thái lỗi được đặt lại; `(1, 1, 128)`, một lưới 65,536 khối theo trục y và một chiều bằng không bị từ chối; trình giả lập từ chối 1 MiB bộ nhớ chia sẻ động | mục 3.1 |
| `scale_rows_defect.cu` | cột 0 kết thúc ở 1.0; 4, 64 và 37 trên 200 phần tử bằng 1.0 khi chạy theo thứ tự, đảo ngược, xáo trộn | trang tổng kết |
| `blur_kernel.cu` | khớp với một bản tham chiếu CPU trên sáu kích thước ảnh | mục 3.3 |
| `blur_unsigned.cu` | cùng đầu ra với `blur_kernel.cu`; `-Wall -Wextra` cảnh báo về phép so sánh có dấu/không dấu (và không cảnh báo với `blur_kernel.cu`) | mục 3.3 |
| `grayscale_kernel.cu` | sai khác không quá một mức xám so với một bản tham chiếu CPU; đỏ, xanh lá, xanh dương, trắng cho ra 76, 149, 29, 255 | mục 3.2 |
| `matvec_options.cu` | phương án 2 tính B c, phương án 3 tính Bᵀ c (N = 37, khối 16) | mục 3.4 |

Các chương trình điều khiển là CUDA thuần. Trên một device CUDA, `launch_checked_test.cu` sẽ kiểm
tra cùng các giới hạn được tài liệu hóa như trên trình giả lập; chỉ trường hợp bộ nhớ chia sẻ động
là riêng cho trình giả lập. Nó không dành cho HIP: AMD ghi trong tài liệu rằng chiều z của khối
được tới 1024, nên `(1, 1, 128)` hợp lệ ở đó. Chưa có bản dựng nào cho device: chưa có gì ở đây
từng chạy trên GPU.
