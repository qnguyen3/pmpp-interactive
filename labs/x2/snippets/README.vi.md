> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Snippets

Mã mà các bài học của học phần X2 hiển thị để phục vụ một lời giải thích, và không phải là một lab.
Các bài học nạp những tệp này lúc chạy bằng `useLabFile('x2/snippets/<name>')`, nên không có
bản sao nào bị lệch đi. Trình chạy lab không build chúng; các unit test biên dịch những tệp C++ phía host
thông qua một chương trình kiểm tra trong `check/` và so sánh kết quả với mô hình của bài học.

| Tệp | Hiển thị ở | Được kiểm tra bởi |
|---|---|---|
| `q4_0_ref.h` | mục X2.2 ("Q4_0: the signed extreme over -8") và X2.3 ("The alignment trap"): một khối Q4_0 của llama.cpp trên CPU, được lượng tử hóa theo quy tắc `_ref`, đóng gói với trọng số j và j + 16 trong byte j, được giải lượng tử hóa, đọc thành các word 32 bit từ hai nửa 16 bit, và tính tích vô hướng với số 8 được tách ra (notes/today/x2-facts.md D1, D3, A5, E4) | `app/src/sim/quant.test.ts` biên dịch `check/q4_0_ref_test.cpp` (`-std=c++17 -Wall -Wextra -Werror -O2 -ffp-contract=off`), chạy nó trên khối của cảnh minh họa, khối q-q40-clip, một khối toàn số không, một cực trị dương và sáu khối sinh từ seed, rồi so sánh các bit FP16 của d, cả 16 byte, cả 32 số thực đã giải lượng tử hóa, bốn word và tích vô hướng đã tách, từng bit một, với `app/src/sim/quant.ts`; nó cũng kiểm tra rằng `sizeof(BlockQ4_0)` bằng 18. Bị bỏ qua khi không tìm thấy trình biên dịch C++. |
| `accumulators.md` | mục X2.6 ("Safe or unsafe?"): bốn đoạn mã, một MSL và ba CUDA, khác nhau ở chỗ cái gì giữ tổng (`half4` vào `dot`, một `__half2` với `__hfma2`, một tổng khối kiểu `short`, một tổng khối kiểu `int` rồi chuyển sang float). Mục này phân tích tệp theo các tiêu đề `##` của nó. | `app/src/sim/qdot.snippets.test.ts` kiểm tra rằng tệp có bốn đoạn mã của `FRAGMENTS` theo đúng thứ tự, rằng mỗi đoạn khai báo bộ tích lũy của nó và cộng vào đó đúng như `FRAGMENT_MODEL` trong `app/src/sim/qdot.ts` mô tả, và rằng đoạn 3 và 4 chỉ khác nhau ở kiểu của `sumi`, mà mô hình cộng bằng int16 và int32. Sau đó nó ghi ba đoạn CUDA (2, 3, 4), đúng như tệp in ra, vào `check/accumulators_check.cu`, chuyển đổi bằng `labs/tools/cu2emu.py`, build với trình giả lập (`-ffp-contract=off`, bật AddressSanitizer) và chạy trên các hàng của `runFragment`: kết quả trường hợp tệ nhất bằng nhau và sai số tương đối với dữ liệu thực tế bằng nhau, từng bit một. Nó cũng cho chương trình kiểm tra, kể cả các đoạn mã, đi qua bước kiểm tra front-end của CUDA, và kiểm tra rằng một lời gọi chỉ dành cho host trong một đoạn mã sẽ làm bước đó thất bại. Trạng thái: đoạn 2 đến 4 emulator-checked và front-end-checked, unverified trên CUDA; đoạn 1 (MSL) chưa được biên dịch, unverified. Bị bỏ qua khi không tìm thấy trình biên dịch C++ hoặc `python3`. |
| `check/accumulators_check.cu` | không ở đâu cả (chương trình kiểm tra của hàng phía trên) | Include mỗi đoạn CUDA dưới dạng `<id>.inc` từ thư mục tạm của test, trong một kernel một luồng; `dot4` là `__dp4a`. Lượt quét front-end của `labs/pmpp.py` thấy nó mà không có các đoạn mã và chỉ kiểm tra chương trình kiểm tra. Không bao giờ được build cho HIP (không có `__dp4a`, x2-facts B1) hay chạy trên GPU. |

`accumulators.md` là Markdown, không phải `.cu` hay `.metal`, vì các đoạn mã không phải là những hàm
hoàn chỉnh và trang web phân tích tệp theo các tiêu đề của nó. Từ 2026-11-21, stub front-end và
cả hai trình giả lập đều khai báo `__half`, `__half2`, `char4`, `uchar4` và `__dp4a`
(`labs/README.md`), chính điều đó cho phép chương trình kiểm tra chạy các đoạn CUDA. `q4_0_ref.h` là của khóa học này, C++ host thuần, được viết từ các dòng llama.cpp trích trong tệp facts.
Phép chuyển đổi FP16 của nó là thủ tục làm tròn về số chẵn gần nhất của riêng chúng tôi; việc `GGML_FP32_TO_FP16`
của llama.cpp có làm tròn giống vậy trên mọi host hay không thì chưa được kiểm tra. Không có gì ở đây chạy trên GPU.
