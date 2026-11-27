> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Đoạn mã (snippets)

Mã mà các bài học của chương 12 hiển thị để phục vụ một lời giải thích, và không phải là một
lab. Các bài học nạp những tệp này lúc chạy bằng `useLabFile('ch12/snippets/<name>')`, nên
không có bản sao nào bị lệch. Trình chạy lab không build chúng.

| Tệp | Hiển thị ở | Được kiểm tra bởi |
|---|---|---|
| `s123_fig12_3.cu` | mục 12.3, "Fig. 12.3, step by step" và "Which line is which?" (các dòng 01 đến 29 của sách; dòng 08 gán cho `j` một giá trị) | Kiểm tra front-end CUDA; `app/src/chapters/ch12/S123.snippets.test.ts` cố định các dòng của sách và các dòng mà bài học tô sáng, và chạy nó trên trình giả lập (emulator) qua `check/s123_check.cu` |
| `s123_fig12_4_cg.cu` | mục 12.3, "With cooperative groups" (26 dòng của Fig. 12.4, cùng header và namespace mà trang bỏ qua, `num_threads()` thay cho `size()` kiểu cũ, và `j = 0`) | Kiểm tra front-end CUDA; `S123.snippets.test.ts`, như trên, với `check/s123_check.cu` được build cùng `-DSNIPPET_CG` |
| `check/s123_check.cu` | không đâu cả (một chương trình điều khiển (driver)) | Kiểm tra front-end CUDA, cả hai biến thể. Được build và chạy bởi `S123.snippets.test.ts` (như cách trình chạy lab build backend `emu`, bật AddressSanitizer): trong ba thứ tự luồng, các vị trí mà mỗi lane đạt ghi vào phải khớp `app/src/sim/warpvote.ts` từng lane một, và 4,000 khóa phải ra đúng với các khóa được giữ của mỗi warp nằm cạnh nhau. Bị bỏ qua khi không tìm thấy trình biên dịch C++ hoặc python3. |
| `s124_privatised_variants.cu` | mục 12.4, trình chạy từng bước và bảng kết quả (Fig. 12.6 với bốn luồng mỗi khối, và bốn bản sao mà mỗi bản bỏ một rào chắn hoặc tăng bộ đếm riêng bằng `++`) | Kiểm tra front-end CUDA; `app/src/sim/privfilter.test.ts` cố định mọi thân kernel theo `app/src/sim/privfilter.ts` và chạy tệp trên `emu` (ba thứ tự) và `race` qua `check/s124_privatised_check.cu` |
| `check/s124_privatised_check.cu` | không đâu cả (một chương trình điều khiển) | Kiểm tra front-end CUDA. `privfilter.test.ts` nối nó vào đoạn mã, dịch kết quả bằng `labs/tools/cu2emu.py`, và chạy một kernel với một thứ tự mỗi tiến trình, để một lượt chạy bị AddressSanitizer dừng không che các lượt khác. |
| `wrap_pairs.cu` | trang tổng kết, kernel của câu hỏi kiểm tra hỗn hợp đầu tiên (`candidatePairs`, của khóa học này, cố tình sai: hai vị trí của một cặp đến từ hai bộ đếm) | Kiểm tra front-end CUDA; `app/src/chapters/ch12/Wrap.snippets.test.ts` cố định hai lời gọi `fetch_add` mà câu hỏi nói tới và chạy nó trên trình giả lập qua `check/wrap_pairs_test.cu` |
| `check/wrap_pairs_test.cu` | không đâu cả (một chương trình điều khiển) | Kiểm tra front-end CUDA. Được build và chạy bởi `Wrap.snippets.test.ts` (như cách trình chạy lab build backend `emu`, bật AddressSanitizer): trong cả ba thứ tự luồng, cả hai bộ đếm phải đúng và mọi cặp phải khớp, đó chính là ý của câu hỏi (emu chạy một luồng đến hết rồi mới tới luồng tiếp theo, nên không có gì chạy giữa hai phép toán nguyên tử của một luồng); một bản sao lấy một vị trí cho mỗi cặp cũng phải đạt. Bị bỏ qua khi không tìm thấy trình biên dịch C++ hoặc python3. |

`wrap_pairs.cu` và đoạn mã Fig. 12.4 include `<cuda/atomic>` mà không có phương án dự phòng
cho HIP. Chúng dành cho kiểm tra front-end CUDA và các trình giả lập; lab 12.1 cho thấy cách
viết khả chuyển. Trên HIP, `<cuda/atomic>` đến từ libhipcxx, thư viện mà bản build HIP của
khóa học liên kết dưới tên `libhipcxx::libhipcxx` khi CMake tìm thấy nó (nó đi kèm ROCm 10.1
trở lên, `notes/today/infra-facts.md` B2). Không có nó, mã include `<cuda/atomic>` không build
được. Có nó, chưa máy ROCm nào build mã của khóa học này (unverified).

Kiểm tra front-end CUDA là kiểm tra mà `labs/pmpp.py` chạy: `clang++ -x cuda --cuda-host-only
-nocudainc -nocudalib -fsyntax-only -std=c++17 -I labs/common/include -I
labs/common/include/pmpp/emu/shim -include pmpp/check/cuda_stub.h <file>`. Không có include
`shim`, `<cuda/atomic>` sẽ không được tìm thấy.
