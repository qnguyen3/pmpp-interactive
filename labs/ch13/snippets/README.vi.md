> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Snippets

Mã mà bài học của chương 13 hiển thị và không phải là một lab. Các tệp được giữ ở đây dưới dạng mã
nguồn `.cu` thật để kiểm tra front-end CUDA trong `./verify.sh` phân tích chúng, và để bài học hiển
thị đúng tệp đã được kiểm tra chứ không phải một bản sao có thể lệch đi. Trình chạy lab không dựng
chúng; `check.py` chạy những tệp mà bài học gọi là emulator-checked.

| Tệp | Hiển thị ở | Cách | Kiểm thử trong `app/src/chapters/ch13/` |
|---|---|---|---|
| `merge_basic.cu` | mục 13.3: `co_rank` (với các dòng được tô sáng của trình khám phá), `merge_sequential` và `merge_basic_kernel` | sao chép | `snippets-s133.test.ts` |

`merge_basic.cu` là Figs. 13.2, 13.5 và 13.9 đã sửa: `__device__` cho các hàm trợ giúp, đầu vào
`const int*`, toán tử ba ngôi cho min và max, các chú thích của Fig. 13.5 được sửa, phép chia làm
tròn lên số nguyên ở dòng 03 của Fig. 13.9, và `k_curr` được kẹp lại (của khóa học này). Mỗi hàm
giữ cách đánh số dòng của sách; bài học nhắc đến các dòng bằng những số đó và kiểm thử ghim chúng
lại. Bài học cũng suy ra biến thể "swapped tie tests" của `co_rank` từ tệp này (hai toán tử được
hoán đổi); kiểm thử kiểm tra rằng đúng hai dòng đó thay đổi.

## Chạy chúng: `check.py`

```sh
python3 labs/ch13/snippets/check.py            # build in a temporary directory
python3 labs/ch13/snippets/check.py --build DIR
```

Nó dịch snippet và chương trình điều khiển (driver) của nó trong `check/` bằng
`labs/tools/cu2emu.py`, biên dịch chúng như trình chạy lab biên dịch backend `emu` (bật
AddressSanitizer, `-Wall -Wextra`), và chạy mọi trường hợp dưới ba thứ tự luồng của trình giả lập.

| Snippet | Kiểm tra gì | Nêu ở |
|---|---|---|
| `merge_basic.cu` | bằng `std::merge` trên các số nguyên đã sắp xếp với nhiều khóa trùng nhau; ví dụ xuyên suốt trên 3 và 4 luồng và trên nhiều luồng hơn số đầu ra; 1,000 đầu ra trên 3 khối 128 luồng; đầu vào rỗng; không có gì được ghi tại hoặc sau `C[m + n]` | mục 13.3 |

Số nguyên thuần không thể cho thấy tính ổn định; lab 13.1 trộn các cặp khóa-giá trị để làm điều đó.
Không có gì ở đây đã chạy trên GPU.
