> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 20.3: săn race condition trong flash attention

Sách: mục 20.5 (pp. 492-504), Figs. 20.9 đến 20.16. Một mục đứng riêng như (A3) nằm trong
`notes/today/ch20-facts.md`.

`work/kernel.cu` và `work/kernel.metal` chứa kernel flash attention của lab 20.2 (cùng hợp
đồng, cùng các bài kiểm thử) với **ba lỗi cài sẵn độc lập**, cùng ba lỗi đó trong cả hai tệp. Mã
biên dịch được. Hãy tìm và sửa cả ba, và ghi lại backend nào đã cho bạn thấy lỗi nào.

Bạn biết mỗi lỗi thuộc loại nào, nhưng không biết nó nằm ở đâu:

1. một rào chắn (barrier) bị thiếu giữa các lane của một warp;
2. một rào chắn bị thiếu trên toàn khối;
3. một giá trị sai trong phần sổ sách của online softmax: không liên quan đến đồng bộ hóa.

Nếu bạn chưa làm lab 20.2, hãy đọc README của nó trước: các mục "Thiết kế (của sách)" ("The design") và
"Chỗ khác với trang sách" ("Where this differs from the page") mô tả kernel đúng.

## Trước khi chạy bất cứ thứ gì, hãy dự đoán

Hãy chép bảng này và điền vào mỗi ô "fails" (thất bại), "reported" (ThreadSanitizer chỉ ra các
dòng mã dù các con số có thể đúng) hoặc "passes" (vượt qua). Trang labs hỏi đáp án của bạn cho lỗi
1 dưới tên `ch20/labs/q-detector`.

| Lỗi | `emu` | `race` | GPU của bạn |
|---|---|---|---|
| 1. rào chắn warp | | | |
| 2. rào chắn khối | | | |
| 3. sổ sách | | | |

Những gì bạn cần để suy luận:

- `emu` chạy từng luồng một. Một luồng chạy cho đến điểm đồng bộ hóa tiếp theo của nó
  (`__syncthreads()`, bất kỳ lời gọi `_sync` nào của warp, kể cả shuffle), rồi luồng tiếp theo
  chạy, theo ba thứ tự cố định. Một lệnh shuffle là một điểm hẹn của cả warp: không lane nào rời
  khỏi nó trước khi mọi lane đã tới.
- `race` chạy các luồng cùng lúc dưới ThreadSanitizer. Cũng như trên GPU, ở đó các lệnh shuffle
  không sắp thứ tự bộ nhớ (A3); `__syncwarp()` và `__syncthreads()` thì có.
- GPU của bạn chạy các warp theo bất kỳ thứ tự và mức chồng lấn nào nó muốn, và phần cứng có thể
  giữ các lane của một warp đi cùng nhau dù mã có yêu cầu hay không (trên GPU Apple, việc một SIMD
  group có chạy đồng bộ từng bước hay không không được ghi trong tài liệu, C4).

## Phương pháp

Giống như của lab 5.4:

1. `python3 labs/pmpp.py run 20.3 --backend emu`. Các thứ tự luồng của nó là tất định, nên một
   lỗi mà nó thấy được sẽ thất bại mọi lần, và AddressSanitizer dừng ở một truy cập ngoài biên và
   chỉ ra dòng mã. Nó chạy chậm trên kernel này: hãy khởi động nó và đọc mã trong khi nó chạy.
2. Sửa một chỗ, chạy lại. Các thất bại sẽ thay đổi dần khi bạn sửa.
3. `python3 labs/pmpp.py run 20.3 --backend race`. ThreadSanitizer chỉ ra hai dòng mã nguồn có
   lần ghi và lần đọc va chạm nhau, cho biết chỗ một rào chắn bị thiếu. Một hàm mà trình biên dịch
   đã nội tuyến có thể hiện ra là `flashAttentionKernel` không kèm số dòng; phía còn lại của cặp
   vẫn chỉ ra dòng của nó. Nó chỉ thấy những va chạm đã xảy ra trong lượt chạy đó, và báo mỗi cặp
   dòng một lần mỗi lượt chạy, nên một bài kiểm thử sau đó của cùng kernel có thể vượt qua: một
   báo cáo sạch không phải là bằng chứng.
4. Rồi đến GPU của bạn: `python3 labs/pmpp.py run 20.3`. Hãy ghi lại những lỗi nó cho bạn thấy
   và những lỗi nó che giấu. Một GPU vượt qua cũng không phải là bằng chứng.
5. Khi cả ba backend đều vượt qua, hãy so sánh `work/` với `solution/`, nơi mỗi dòng được sửa
   đều được đánh dấu.

Hữu ích khi săn lỗi: dữ liệu "late max" khiến giá trị lớn nhất đang chạy tăng ở mọi ô, nên mọi
sai sót trong một lần co giãn lại lộ ra ở đó trước tiên; thông điệp lỗi chỉ ra hàng, head và công
thức dữ liệu.

## Tệp

- `work/kernel.cu` (CUDA; cũng được biên dịch cho HIP và hai trình giả lập) và
  `work/kernel.metal`: phần bạn phải sửa.
- `work/launch.mm`: phía host Metal tham chiếu của lab 20.2 (nó chọn `B_c` cho thiết bị của
  bạn, biên dịch `kernel.metal` với giá trị đó, kiểm tra pipeline và dispatch). Nó không có lỗi.
- `main.cpp`: chương trình điều khiển (driver). Các bài kiểm thử, tham chiếu trên CPU và các dung
  sai là của lab 20.2, trong `../shared-g11/flash_test.hpp`: cùng các hình dạng trên GPU, các
  hình dạng nhỏ trên trình giả lập (`emu`: n = 1, 33, 64, 70; `race`: n = 33 và 40).

Không có gì để đo hiệu năng ở đây: lab 20.2 đo thời gian cùng kernel đó.
