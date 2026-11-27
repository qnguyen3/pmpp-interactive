> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Snippets

Mã mà các bài học của chương 23 hiển thị và không phải là lab. Các tệp được giữ ở đây dưới dạng mã
nguồn thật để phép kiểm tra front-end CUDA trong `./verify.sh` phân tích cú pháp các tệp `.cu`, và
để bài học hiển thị đúng tệp đã được kiểm tra chứ không phải một bản sao có thể bị lệch. Trình chạy
lab không biên dịch chúng; `check.py` chạy chúng.

| Tệp | Hiển thị ở | Cách dùng | Test trong `app/src/chapters/ch23/` |
|---|---|---|---|
| `staged_loop.cu` | mục 23.3, trình chạy từng bước "Copy, compute, copy back", tab CUDA: vòng lặp chia giai đoạn (staged), phát lệnh theo chiều sâu trước và theo chiều rộng trước | nạp bằng `useLabFile`; trình chạy từng bước tô sáng các dòng mà `app/src/sim/chunkpipe.ts` (`LINES`) nói đã phát từng thao tác | `S233.snippets.test.ts` kiểm tra các dòng đó đúng là những lời gọi mà mô hình nói |
| `staged_loop_metal.mm`, `staged_loop.metal` | cùng trình chạy từng bước, khung nhìn Metal: K buffer dùng chung đang hoạt động cùng lúc trên Apple silicon | như trên | như trên |

Cả hai là của chúng tôi, dựa theo vòng lặp sao chép và thực thi chia giai đoạn của NVIDIA (CUDA C++
Best Practices Guide, mục 10.1.2; `notes/today/ch23-facts.md` A7) và mẫu "Synchronizing CPU and GPU
work" của Apple (C5). Chúng chia công việc theo cách lab 23.1 làm: chunk `c` là `[c*n/chunks,
(c+1)*n/chunks)`, các chunk rỗng bị bỏ qua, và chunk không rỗng thứ j đi vào stream (hoặc ô) `j % S`.
Chúng cố ý nhỏ hơn lab: không kiểm tra lỗi, và bên gọi tạo và đồng bộ hóa các stream.

## Chạy chúng: `check.py`

```sh
python3 labs/ch23/snippets/check.py            # build in a temporary directory
python3 labs/ch23/snippets/check.py --build DIR # or $PMPP_BUILD_DIR/ch23-snippets
```

| Snippet | Điều được kiểm tra | Trạng thái nó hỗ trợ |
|---|---|---|
| `staged_loop.cu` | front end CUDA (clang, các header stub của khóa học) chấp nhận nó và chương trình điều khiển (driver) `check/staged_loop_test.cu` của nó | front-end-checked |
| `staged_loop.cu` | trên trình giả lập (emulator) CPU (dịch bằng `labs/tools/cu2emu.py`, bật AddressSanitizer), cả hai thứ tự phát lệnh đều cho `2x + 1` ở mọi phần tử, với buffer host được ghim, cho các chunk bằng nhau, các chunk không đều (độ dài 2, 3, 2, 3), nhiều chunk hơn số phần tử (chunk rỗng), 1 phần tử trong 64 chunk, nhiều stream hơn số chunk, một stream, và n = 0; không có gì được ghi ra ngoài đầu ra | emulator-checked |
| `staged_loop_metal.mm` với `staged_loop.metal` | biên dịch bằng clang++ và chạy trên GPU của máy Mac này (kernel được biên dịch lúc chạy) với K = 1, 2, 3 và nhiều buffer hơn số chunk, với các chunk bằng nhau, không đều và rỗng và n = 0, so với một bản tham chiếu CPU theo một dung sai; không có gì được ghi ra ngoài đầu ra | verified on Metal |

Trình giả lập chạy các stream một cách đồng bộ, nên nó kiểm tra mỗi chunk tính ra gì và kết quả nằm
ở đâu, chứ không bao giờ kiểm tra có gì chồng lấp hay không; độ chồng lấp là phép đo của lab 23.1.
Chưa có gì ở đây từng chạy trên phần cứng CUDA hoặc HIP.

## Trang 23.3–23.5 (id 23.4): các vòng lặp của sách (chỉ front-end-checked, chưa bao giờ chạy)

| Tệp | Hiển thị ở | Trạng thái | Test trong `app/src/chapters/ch23/` |
|---|---|---|---|
| `overlap_loop.cu` | trang 23.3–23.5 (id 23.4), bên cạnh câu đố "What enforces each ordering?": các vùng được đánh dấu `[setup]` và `[loop]` (Fig. 23.14 với số dòng từ 01 đến 43, dòng 19 đã được sửa) | front-end-checked | `S234.snippets.test.ts`: các bản sao trong bài học bằng các vùng đã bỏ thụt lề; mọi dòng mà `app/src/sim/haloloop.ts` và câu đố trích dẫn đều chứa đúng lời gọi chúng nêu tên; phép kiểm tra front-end qua được, và thất bại ở dòng 19 như trang sách in |
| `nccl_exchange.cu` | trang 23.3–23.5 (id 23.4), đoạn về NCCL: vùng `[exchange]`, Fig. 23.20 các dòng 27 đến 44 | front-end-checked | cùng test đó |
| `check/mpi_nccl_decls.h` | không gì cả (chỉ có khai báo) | | được cả hai include; khai báo các lời gọi MPI và NCCL mà stub của khóa học thiếu, với chữ ký lấy từ `notes/today/ch23-facts.md` A10, A12; các khai báo lại hai lời gọi độ ưu tiên của CUDA (A1) giờ cũng có trong stub, giống hệt, và vẫn biên dịch được |

Cả hai tệp cần MPI (và NCCL) và mỗi rank một GPU, điều mà khóa học không giả định, nên chúng được
phân tích cú pháp và không bao giờ được biên dịch hay chạy. `check.py` không động đến chúng; tệp
vitest ở trên tự chạy clang, với cùng các cờ như `labs/pmpp.py`, và bị bỏ qua khi không tìm thấy
clang++.

## Trang tổng kết: mã của một câu hỏi (chỉ front-end-checked, chưa bao giờ chạy)

| Tệp | Hiển thị ở | Trạng thái | Test trong `app/src/chapters/ch23/` |
|---|---|---|---|
| `rerecord.cu` | trang tổng kết, câu hỏi `ch23/wrap/q-rerecord`: vùng `[shown]`, một event được ghi hai lần trong stream `s1` với một lần chờ trong `s2` giữa hai lần ghi | front-end-checked | `Wrap.snippets.test.ts`: bản sao trên trang bằng vùng đó; phép kiểm tra front-end qua được |

Của chúng tôi, không phải của sách. Điều câu hỏi hỏi (một lần chờ lấy event theo trạng thái được ghi
tại thời điểm lần chờ được phát, `notes/today/ch23-facts.md` A5) không thể hiển thị trên trình giả
lập của khóa học, vốn chạy mọi lời gọi stream một cách đồng bộ, nên tệp được phân tích cú pháp và
không bao giờ chạy. `rerecordFixed` trong cùng tệp là thứ tự mà lời giải thích của câu hỏi đưa ra.
