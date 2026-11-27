> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Snippets

Các kernel mà các trang bài học của chương 21 hiển thị, và không phải là lab. Chúng được giữ ở đây
dưới dạng tệp `.cu` thật để kiểm tra front-end CUDA trong `./verify.sh` phân tích từng tệp, để
`check.py` có thể chạy chúng trên trình giả lập (emulator) CPU, và để bài học hiển thị đúng tệp đã
được kiểm tra chứ không phải một bản sao có thể lệch đi. Trình chạy lab không dựng chúng.

Bài học dùng chúng thế nào: mục 21.2 (`app/src/chapters/ch21/S212.tsx`) giữ một bản sao của phần
mỗi tệp nằm dưới dòng đánh dấu của nó (`// ---- shown in the lesson ----`, hoặc
`// ---- as printed ----`). `app/src/chapters/ch21/snippets.test.ts` thất bại nếu một bản sao và
tệp của nó khác nhau, và ghim các dòng mà trang nhắc tới. Hãy sửa kiểm thử cùng với tệp.

| Tệp | Nó là gì | Hiển thị ở |
|---|---|---|
| `scatter.cu` | Fig. 21.5 kèm số lượng nguyên tử và một cận cho nó (listing không có cái nào trong hai) | mục 21.2, "Scatter" |
| `gather.cu` | Fig. 21.6 kèm kiểm tra biên cho `i` và `j` (listing dựa vào việc đệm) | mục 21.2, "Gather" |
| `coarsen_printed.cu` | 34 dòng mã của Fig. 21.8 đúng như bản in, giữ số dòng, một chú thích được viết lại | mục 21.2, săn lỗi in |
| `coarsen_adjacent.cu` | Fig. 21.8 đã sửa: dòng 05 nhân chỉ số luồng với hệ số; kiểm tra biên cho từng điểm | mục 21.2, "Coarsening" |
| `coarsen_strided.cu` | Fig. 21.10 kèm kiểm tra biên cho từng điểm | mục 21.2, "Coalescing" |
| `bias_gather.cu` | của khóa học này, không phải của sách: một logit bias ở dạng gather của Fig. 21.6, cho một câu hỏi; chỉ front-end-checked, không bao giờ chạy (không có trong `check.py`) | trang tổng kết, phần kiểm tra tổng hợp (gắn bởi `Wrap.snippets.test.ts`) |

`coarsen_printed.cu` cố ý sai và không bao giờ được làm cơ sở cho một lab: như bản in, dòng 05 cộng
một `threadIdx.x` chưa nhân hệ số trong khi các dòng 30 đến 33 ghi `i` đến `i + 3`, nên các luồng
cạnh nhau chồng lấn với một `+=` thông thường và phần lớn các điểm của mỗi khối không bao giờ được
ghi (`notes/ch21.md`, "Unclear in the PDF"; `notes/today/ch21-facts.md` E2).

Mọi tệp đã sửa tính chỉ số lát là `k = z / gridspacing` (dạng của Fig. 21.6; Fig. 21.5 viết
`(grid.x*grid.y*z) / gridspacing`). Phép chia là chính xác với khoảng cách 0.5 Å của sách, đó là
khoảng cách mà các phép kiểm tra dùng; một kernel sản xuất sẽ truyền `k` dưới dạng số nguyên.
`sqrtf` không cần khai báo cho kiểm tra front-end: header stub của nó khai báo các hàm toán học độ
chính xác đơn.

## Chạy chúng: `check.py`

```sh
python3 labs/ch21/snippets/check.py            # build in a temporary directory
python3 labs/ch21/snippets/check.py --build DIR
```

Nó biên dịch mỗi snippet với chương trình điều khiển (driver) `check/dcs_test.cu` (mỗi snippet một
`-DKIND_...`, `CHUNK_SIZE` 64), được dịch bởi `labs/tools/cu2emu.py` và dựng như trình chạy lab
dựng backend `emu` (bật AddressSanitizer, `-Wall -Wextra`, cảnh báo làm kiểm tra thất bại). Chương
trình điều khiển làm đúng những gì p. 517 mô tả trên host: với mỗi lát, chép lát vào, chép các
nguyên tử vào bộ nhớ hằng (constant memory) từng phần một với mỗi phần một lần khởi chạy, chép lát
về. Mọi trường hợp chạy dưới ba thứ tự luồng của trình giả lập.

| Snippet | Kiểm tra gì | Nêu ở |
|---|---|---|
| `scatter.cu` | bằng một tổng float64 trong phạm vi 1e-5 của tổng các đóng góp tuyệt đối, qua hai lát và ba phần, với ba hình dạng khởi chạy; không có gì được ghi vượt quá lưới | mục 21.2 |
| `gather.cu` | như trên, trên các lát không phải bội của khối | mục 21.2 |
| `coarsen_adjacent.cu` | như trên, trên các lát không phải bội của 4 × `blockDim.x` | mục 21.2 |
| `coarsen_strided.cu` | như trên | mục 21.2 |
| `coarsen_printed.cu` | ở `blockDim.x` = 32 trên một hàng 128 điểm: các điểm 35 đến 127 không bao giờ được ghi, 3 đến 31 bị ghi bốn lần, 0 đến 2 và 32 đến 34 từ một đến ba lần; kết quả khác với tổng | mục 21.2, săn lỗi in |

Điều này không thể cho thấy: trình giả lập chạy từng luồng một, nên nó không thể thấy rằng kernel
scatter cần các atomic của nó (một `+=` thường cũng sẽ qua ở đây; backend `race` của lab 21.1 là
công cụ cho việc đó), và nó không mô phỏng coalescing, bộ nhớ đệm hay thời gian. Không có gì ở đây
đã chạy trên GPU.
