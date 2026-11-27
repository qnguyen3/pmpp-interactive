> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Snippets

Mã mà các bài học của Phụ lục A hiển thị để phục vụ một lời giải thích, và không phải là lab.
Bài học nạp các tệp này lúc chạy bằng `useLabFile('appA/snippets/<name>')`, nên không có bản sao
nào để lệch. Trình chạy lab không biên dịch chúng; các unit test biên dịch những tệp C++ chạy trên
host thông qua một chương trình kiểm tra trong `check/` và so sánh kết quả với mô hình của bài học.

| Tệp | Hiển thị ở | Được kiểm tra bởi |
|---|---|---|
| `bf16_round.h` | mục A.4, "FP32 to BF16 on the bits": phép làm tròn FP32 sang BF16 của llama.cpp (làm tròn đến số chẵn gần nhất trên các bit số nguyên, NaN bị ép thành quiet; `ggml-impl.h`, notes/today/appA-facts.md D3) cùng với phép cắt bớt thông thường bên cạnh | `app/src/sim/fpadd.test.ts` biên dịch `check/bf16_round_test.cpp` bằng trình biên dịch C++ của host, chạy nó trên các preset của trình khám phá, các NaN, các trường hợp hòa và 400 mẫu bit ngẫu nhiên, rồi so sánh từng kết quả, chính xác đến từng bit, với `bf16FromBits` trong `app/src/sim/fpadd.ts`. Bị bỏ qua khi không tìm thấy trình biên dịch C++. |
| `mx_quantize.h` | mục A.5, dưới cảnh "Quantising one block": một khối MXFP4 (32 mã E2M1 và một hệ số tỉ lệ UE8M0, theo quy tắc của OCP MX v1.1 §6.3; notes/today/appA-facts.md A4), được lượng tử hóa và giải mã trên CPU | `app/src/sim/mxblock.test.ts` biên dịch `check/mx_quantize_test.cpp` (`-Wall -Wextra`, không cho phép cảnh báo nào), chạy nó trên khối của cảnh, khối q-outlier, một khối toàn số không, một giá trị lớn nhất là lũy thừa của hai, các trường hợp hòa làm tròn về số chẵn, các số float subnormal và mười khối sinh từ seed, rồi so sánh byte hệ số tỉ lệ, cả 32 mã và từng số float đã giải mã, chính xác đến từng bit, với `quantizeBlock` trong `app/src/sim/mxblock.ts`; nó cũng kiểm tra `sizeof(MxBlock)` bằng 17, đúng kích thước `block_mxfp4` của llama.cpp. Bị bỏ qua khi không tìm thấy trình biên dịch C++. |
| `kahan.h` | phần săn lỗi của mục A.6, "Kahan, four ways": một vòng lặp thông thường và bốn vòng lặp có bù (phiên bản 1 bị đảo dấu phần hiệu chỉnh, phiên bản 2 là của Kahan, phiên bản 3 cập nhật phần hiệu chỉnh sau tổng, phiên bản 4 xóa nó đi như fast math có thể làm) | `app/src/sim/fpsum.test.ts` kiểm tra tệp có vòng lặp thông thường và bốn phiên bản theo đúng thứ tự của phần săn lỗi, rồi biên dịch `check/kahan_check.cpp` (`-std=c++17 -Wall -Wextra -O2`, không fast math) và so sánh từng phiên bản, chính xác đến từng bit, với `kahanVariant` trong `app/src/sim/fpsum.ts` trên năm bộ dữ liệu. Bị bỏ qua khi không tìm thấy trình biên dịch C++. |
| `check/fpformat_test.cpp` | không hiển thị trong bài học nào: phép kiểm tra chéo giữa `fpformat.h` của lab A.1 (mô hình CPU của phép làm tròn FP32, FP16 và BF16) và `app/src/sim/minifloat.ts` của trình khám phá | `app/src/sim/minifloat.test.ts` biên dịch nó với `../lab1_sumerror/fpformat.h` và so sánh trên hơn 30,000 phép làm tròn và 15,000 phép cộng ở FP32, FP16 và BF16, cả hai chế độ làm tròn, chính xác đến từng bit. Bị bỏ qua khi không tìm thấy trình biên dịch C++. |

`bf16_round.h` là C++ thuần chạy trên host. Hai dòng tính kết quả của nó là của llama.cpp; phép
kiểm tra NaN và việc dùng `memcpy` thay cho union là của chúng tôi. Không có gì ở đây chạy trên GPU.

`mx_quantize.h` là của chúng tôi, C++ thuần chạy trên host: dạng mà một bản tham chiếu CPU cho một
kernel MXFP4 thường có. Nó gói hai mã vào mỗi byte, phần tử chẵn ở nửa byte thấp, khớp với thứ tự
phần tử của mô hình; chúng tôi không khẳng định một thư viện cụ thể nào đó gói nửa byte theo cùng
cách.

`kahan.h` là của chúng tôi, C++ thuần chạy trên host; thuật toán của Kahan là kiến thức chuẩn của
giải tích số (sách chỉ trích dẫn nó, p. 594).
