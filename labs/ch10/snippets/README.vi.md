> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Snippets

Mã mà bài học của chương 10 hiển thị để phục vụ một lời giải thích, và không phải là lab. Bài học
nạp các tệp này lúc chạy bằng `useLabFile('ch10/snippets/<name>')`, nên không có bản sao nào để lệch.
Chúng không được trình chạy lab biên dịch.

| Tệp | Hiển thị ở | Được kiểm tra bởi |
|---|---|---|
| `fold.h` | mục 10.1, "One loop for every reduction" | `app/src/sim/reducetree.test.ts` biên dịch `check/fold_test.cpp` bằng trình biên dịch C++ của host, chạy nó, và so sánh từng số float, đúng từng bit, với mô hình đằng sau trình khám phá phép gộp của mục đó (`app/src/sim/reducetree.ts`). Bài kiểm thử bị bỏ qua khi không tìm thấy trình biên dịch C++. |
| `s103_inplace.cu` | các mục 10.3 và 10.4, các trình chạy từng bước (Figs. 10.5 và 10.8 như in, một khối) | Kiểm tra front-end CUDA; `app/src/sim/reduction.test.ts` cố định các số dòng mà trình chạy từng bước tô sáng và chạy cả hai kernel trên trình giả lập (xem bên dưới) |
| `s104_mirrored.cu` | mục 10.4, "Mirror it" (bài tập 3, câu trả lời của chúng tôi) | Kiểm tra front-end CUDA; `app/src/chapters/ch10/S104.snippets.test.ts` gắn các phần của trang với nó; chạy trên trình giả lập như bên dưới |
| `s106_barrier_hunt.cu` | mục 10.6, "Where does the barrier go?" | Kiểm tra front-end CUDA; `app/src/sim/reduction.test.ts` cố định các số dòng của nó và chạy mọi kernel trên trình giả lập |
| `s107_warp_reduce_as_printed.cu` | mục 10.7, tab "as printed (CUDA only)" của `warp_reduce` (Fig. 10.13 và các hàm trợ giúp ở pp. 238–239, như in) | Kiểm tra front-end CUDA; `app/src/chapters/ch10/S107.snippets.test.ts` cố định các dòng của sách và chạy nó trên trình giả lập qua `check/s107_warp_reduce_test.cu` |
| `s107_hybrid.cu` | mục 10.7, "Fig. 10.14, corrected" (các dòng 01 đến 19 của sách, mặt nạ và độ rộng khả chuyển) | Kiểm tra front-end CUDA; `S107.snippets.test.ts` cố định các số dòng mà bài học nêu tên, kiểm tra rằng kiểu mặt nạ, `warpIdx` và `warp_reduce` của nó giống của lab 10.1 đến từng ký tự, và chạy nó trên trình giả lập qua `check/s107_hybrid_test.cu` ở 64 và 1,024 luồng |
| `check/s107_warp_reduce_test.cu`, `check/s107_hybrid_test.cu` | không ở đâu (chương trình điều khiển) | Kiểm tra front-end CUDA. Được `S107.snippets.test.ts` dựng và chạy (như trình chạy lab dựng backend `emu`, bật AddressSanitizer); những gì chúng in ra được so với `app/src/sim/shuffle.ts`. Bị bỏ qua khi không tìm thấy trình biên dịch C++ hoặc python3. |
| `wrap_centre.cu` | trang tổng kết, kernel của câu hỏi kiểm tra tổng hợp đầu tiên (`centreRows`, của chúng tôi, cố ý sai: chỉ luồng 0 giữ tổng của hàng) | Kiểm tra front-end CUDA; `app/src/chapters/ch10/Wrap.snippets.test.ts` kiểm tra rằng kiểu mặt nạ, các define, `warpIdx`, `laneIdx` và `warp_reduce` của nó giống của lab 10.1 đến từng ký tự, và chạy nó trên trình giả lập qua `check/wrap_centre_test.cu` |
| `check/wrap_centre_test.cu` | không ở đâu (một chương trình điều khiển) | Kiểm tra front-end CUDA. Được `Wrap.snippets.test.ts` dựng và chạy (như trình chạy lab dựng backend `emu`, bật AddressSanitizer): với khối 256 và 64 luồng và theo ba thứ tự luồng, kernel của trang phải đúng chính xác ở những cột mà luồng 0 ghi, và một bản sao chuyển giá trị trung bình trở lại qua bộ nhớ chia sẻ sau một rào chắn phải đúng ở mọi nơi. Bị bỏ qua khi không tìm thấy trình biên dịch C++ hoặc python3. |
| `check/reduction_check.cu` | không ở đâu (một chương trình điều khiển) | Kiểm tra front-end CUDA. `app/src/sim/reduction.test.ts` nối nó vào sau ba tệp ở trên, chuyển đổi kết quả bằng `labs/tools/cu2emu.py`, dựng nó như trình chạy lab dựng backend `emu` (bật AddressSanitizer), chạy mỗi kernel theo ba thứ tự luồng của trình giả lập, và so sánh đầu ra cùng các mảng với `app/src/sim/reduction.ts`. Bị bỏ qua khi không tìm thấy trình biên dịch C++. |

`fold.h` là mã C++ host thuần: vòng lặp tuần tự của Fig. 10.2 dưới dạng một template, một phép max
trên một hàng bắt đầu từ `-FLT_MAX`, và một phép argmax trên các cặp (giá trị, chỉ số) bắt đầu từ
phần tử 0 (chú thích 1, p. 222) và giữ chỉ số nhỏ hơn khi hòa.

`s106_barrier_hunt.cu` cố ý sai, trừ `SharedMemorySumReductionKernel` (Fig. 10.10 với tham số
`output` bị thiếu đã được thêm lại) và `MaxThenSumKernel`. Hai kết quả của lần chạy trên trình giả
lập được trích dẫn trong mục 10.6: `BarrierInsideIfKernel` đạt ở cả ba thứ tự, vì trình giả lập,
giống cách diễn đạt của NVIDIA, cho các luồng đã thoát được miễn rào chắn (mô hình của bài học giữ
quy tắc chặt hơn của sách và báo nó bị kẹt); và `LessOrEqualKernel` dừng với một báo cáo
AddressSanitizer trên dòng phép cộng của nó. Hai kernel max-rồi-tổng tính tổng `x - max` thay vì
`exp(x - max)`: chúng được viết khi header stub của kiểm tra front-end chưa khai báo hàm toán học
device nào (giờ nó khai báo `expf`), và câu hỏi về rào chắn mà chúng đặt ra không phụ thuộc vào thứ
được cộng.

`s107_hybrid_test.cu` cũng chứa hai biến thể của Fig. 10.14 được viết cho phép kiểm tra, không phải
cho bài học: một biến thể không có rào chắn sau vòng lặp (dòng 11), mà mục 10.7 nói là không cần với
cách kernel được đánh chỉ số, và một biến thể đối chứng chuyển giao sớm một cấp và để mỗi lane cộng
hai vị trí mà các luồng khác đã ghi. Phép kiểm tra yêu cầu biến thể thứ nhất đúng ở cả ba thứ tự
luồng và biến thể đối chứng sai ở ít nhất một thứ tự.
