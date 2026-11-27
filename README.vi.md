> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# PMPP, học trọn qua

[English](README.md) | Tiếng Việt

Một bản học tập tương tác của *Programming Massively Parallel Processors* (ấn bản thứ 5),
làm cho một người đọc: các bài học có mô phỏng và hoạt ảnh có thuyết minh trong một ứng dụng
web chạy cục bộ, và các lab chạy trên chính GPU của bạn qua CUDA, HIP hoặc Metal.

Trạng thái (2026-11-21): mọi chương đã được xây dựng. Đó là các chương 1 đến 24, phụ lục A và C
(B được bỏ qua: kiến thức cơ bản về học sâu), và ba mô-đun mà sách không có: X1 kernel lấy mẫu,
X2 kernel lượng tử hóa, X3 đọc các backend thực tế. Có 79 lab, và chúng mang các mốc dự án
M0 đến M10 (xem `PLAN.md`, mục 5). Chương 5 là bản thử nghiệm đầu tiên và là khuôn mẫu.
`PROGRESS.md` ghi trạng thái hiện tại và những việc còn mở; kế hoạch nằm trong `PLAN.md`.

## Chạy ứng dụng

```sh
./start.sh          # then open http://localhost:5173
```

Cần Node.js 20 trở lên. Tiến độ của bạn được lưu vào `progress/progress.json`; hãy commit nó
hoặc đồng bộ nó để mang theo giữa các máy.

Ứng dụng có tiếng Anh và tiếng Việt: nút chuyển ngôn ngữ nằm ở chân thanh điều hướng bên trái.
Để dịch hoặc rà soát một trang, hãy bắt đầu từ `docs/i18n/README.md`.

## Chạy các lab

```sh
./verify.sh                         # build + test + benchmark everything this machine supports,
                                    # writes reports/verify-<time>-<os>-<backends>.md
./verify.sh --quick                 # same without benchmarks
./verify.sh --backend hip           # one backend only (repeatable), e.g. skip the slow emulators
python3 labs/pmpp.py doctor         # which backends were found, and what the device reports
python3 labs/pmpp.py run 5.2        # test your work for lab 5.2 on every backend
python3 labs/pmpp.py run 5.2 --bench
python3 labs/pmpp.py run 5.2 --backend emu
python3 labs/pmpp.py run 5.2 --solution
```

Ý nghĩa các lệnh trên: `./verify.sh` build, kiểm thử và benchmark mọi thứ máy này hỗ trợ, rồi
ghi `reports/verify-<time>-<os>-<backends>.md`; `--quick` làm như vậy nhưng bỏ benchmark;
`--backend hip` chỉ chạy một backend (có thể lặp lại), chẳng hạn để bỏ qua các trình giả lập
chậm; `doctor` cho biết tìm thấy những backend nào và thiết bị báo cáo gì; `run 5.2` kiểm thử
bản làm của bạn cho lab 5.2 trên mọi backend.

Cần CMake 3.21 trở lên, một trình biên dịch C++17 và Python 3. Các backend được tự động phát
hiện và những backend không có sẽ được bỏ qua:

| Backend | Cần | Trạng thái của các lời giải tham chiếu |
|---|---|---|
| `metal` | macOS có Xcode command line tools | verified on Metal (máy Mac của tác giả), trừ các lab 11.5, 12.5 và 18.1 phần B, vốn chỉ dành cho họ CUDA |
| `emu` | không cần gì thêm | emulator-checked: một trình giả lập trên CPU cho tập con của CUDA, chỉ kiểm tra logic. Giới hạn thời gian kiểm thử của nó là 2400 s (900 s trên thiết bị; `PMPP_TEST_TIMEOUT` ghi đè giá trị này), vì bộ kiểm thử của lab X3.2 chạy lần lượt từng luồng |
| `race` | không cần gì thêm | cùng trình giả lập đó với các luồng chạy đồng thời dưới ThreadSanitizer, cho những lab chọn tham gia |
| `cuda` | NVIDIA CUDA toolkit | front-end-checked; **chưa từng chạy trên thiết bị CUDA** |
| `hip` | AMD ROCm (`ROCM_PATH` hoặc `/opt/rocm`) | **unverified**, chưa từng được biên dịch |

Nếu bạn có máy CUDA hoặc ROCm: hãy chạy `./verify.sh` ở đó và giữ lại báo cáo nó ghi ra. Báo
cáo ghi lỗi build và lỗi kiểm thử đủ chi tiết để sửa từ đó.

## Bố cục

```
app/        the web app; app/src/chapters/chNN/ holds a chapter's lessons
content/    narration scripts, audio and manifests
labs/       build system, runtime, CPU emulator, and labs/chNN/labM_name/{starter,work,solution}
notes/      notes on every chapter of the book, with page references
progress/   your progress file
reports/    verify reports
```

Nghĩa là: `app/` là ứng dụng web, mỗi `app/src/chapters/chNN/` chứa các bài học của một chương;
`content/` chứa kịch bản thuyết minh, âm thanh và manifest; `labs/` chứa hệ thống build, runtime,
trình giả lập CPU và các lab `labs/chNN/labM_name/{starter,work,solution}`; `notes/` chứa ghi chú
cho mọi chương của sách, kèm số trang; `progress/` chứa tệp tiến độ của bạn; `reports/` chứa các
báo cáo verify.

Tệp PDF của sách không nằm trong kho mã và không cần để chạy bất cứ thứ gì.
