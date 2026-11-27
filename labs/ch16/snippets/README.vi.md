> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Snippets

Mã mà các bài học của chương 16 hiển thị và không phải là một lab. Các tệp được giữ ở đây dưới dạng
mã nguồn `.cu` thật để phép kiểm tra front-end CUDA trong `./verify.sh` phân tích được chúng, và để
bài học hiển thị đúng tệp đã được kiểm tra chứ không phải một bản sao có thể lệch dần. Chúng không được
trình chạy lab build; `check.py` chạy những tệp mà bài học gọi là emulator-checked.

| Tệp | Hiển thị ở | Cách | Bài kiểm thử trong `app/src/chapters/ch16/` |
|---|---|---|---|
| `fw_bottomup.cu` | mục 16.3: kernel Floyd-Warshall (cùng các dòng được tô sáng của trình chạy từng bước) và vòng lặp host của nó | sao chép | `snippets-s163.test.ts` |

`fw_bottomup.cu` là Fig. 16.4 đã được sửa, ba thay đổi so với trang sách:

1. "Không có đường đi" là số nguyên `FW_INF = INT_MAX / 2`. Trang sách so sánh một `int` với
   `INFINITY` của `<cmath>`, một `float` mà không `int` nào bằng, nên các lối thoát sớm của nó không bao giờ kích hoạt, và
   với `INT_MAX` làm "không có cạnh" thì phép cộng sẽ tràn số (ch16-facts A3). Một nửa `INT_MAX`
   chừa chỗ cho một phép cộng.
2. Không luồng nào return trước `__syncthreads()`. Trang sách cho các luồng ngoài phạm vi return trước;
   CUDA ghi trong tài liệu rằng các luồng đã thoát không bắt rào chắn (barrier) phải chờ chúng, HIP không ghi điều đó, còn
   Metal chỉ ghi điều đó cho Apple silicon (ch04-facts A6, B3, C3). Thay vào đó, một cờ `inRange`
   canh giữ phần việc. Lệnh `return` sau rào chắn chỉ phụ thuộc vào giá trị trong bộ nhớ chia sẻ,
   nên nó như nhau cho cả khối.
3. Vòng lặp host không có `cudaDeviceSynchronize()` giữa các lần khởi chạy: các lần khởi chạy trong một stream
   chạy theo thứ tự (ch02-facts A5).

Bài học hiển thị kernel và vòng lặp host thành hai đoạn mã, mỗi đoạn có nguyên văn trong tệp;
bài kiểm thử kiểm tra điều đó, và cố định các số dòng mà trình chạy từng bước tô sáng.

## Chạy chúng: `check.py`

```sh
python3 labs/ch16/snippets/check.py            # build in a temporary directory
python3 labs/ch16/snippets/check.py --build DIR
```

Nó dịch mỗi snippet và chương trình điều khiển (driver) của nó trong `check/` bằng `labs/tools/cu2emu.py`, biên dịch
chúng theo cách trình chạy lab biên dịch backend `emu` (bật AddressSanitizer, `-Wall -Wextra`),
và chạy mọi trường hợp dưới ba thứ tự luồng của trình giả lập (emulator).

| Snippet | Kiểm tra gì | Phát biểu ở |
|---|---|---|
| `fw_bottomup.cu` | mọi bước bằng một bước CPU tuần tự trên bảng của Fig. 16.5 như in trong sách, với 8, 3 và 32 luồng mỗi khối; sau bước a và b, bước c thay đổi đúng a đến e (23), b đến e (19), g đến e (23) và h đến e (13); vòng lặp host bằng CPU trên các đồ thị ngẫu nhiên có trọng số không âm, từ 1 đến 64 đỉnh, với các khối để một số luồng nằm ngoài phạm vi | mục 16.3 |

Trình giả lập chạy từng luồng một; nó không thể cho thấy phép cập nhật tại chỗ sẽ làm gì
nếu hàng k hoặc cột k thay đổi trong một bước (điều đó cần một `dist[k][k]` âm, thứ mà
trình chạy từng bước của bài học mô phỏng thay vào đó). Chưa có gì ở đây chạy trên GPU.
