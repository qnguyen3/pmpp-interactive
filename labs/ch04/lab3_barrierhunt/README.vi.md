> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 4.3: săn rào chắn

Sách: mục 4.3 (các quy tắc cho `__syncthreads()`, p. 72), mục 4.4 (p. 74: trong một khối, hãy
giả định các luồng chạy theo bất kỳ thứ tự nào) và bài tập 5 (p. 91). Kernel này là của khóa học
này, không phải một đoạn mã in trong sách.

`work/kernel.cu` và `work/kernel.metal` chứa một kernel đảo ngược tại chỗ phần mảng của mỗi khối.
Khối b sở hữu các phần tử từ `b*blockDim.x` đến `(b+1)*blockDim.x`, và phần cuối cùng dừng ở chỗ
mảng dừng. Mỗi luồng (thread) đọc phần tử của nó vào một thanh ghi (register), rồi ghi nó vào vị
trí đối xứng trong phần của mình. Nó biên dịch được, và có **ba lỗi độc lập**. Hãy tìm và sửa cả
ba.

- Sửa `work/kernel.cu` (`reverseBlocksKernel`; lần khởi chạy `launchReverseBlocks(a, n,
  threadsPerBlock)` bên dưới nó là đúng) và `work/kernel.metal` (`reverse_blocks`).
  `work/launch.mm` cũng đúng. Cùng ba lỗi đó có trong cả hai kernel.
- Kiểm thử: `python3 labs/pmpp.py run 4.3 --backend emu`, rồi `python3 labs/pmpp.py run 4.3`.

Chương trình điều khiển (driver) chọn kích thước khối: 32, 48, 64 và 96 luồng, rồi một, hai và
bốn lần độ rộng mà thiết bị của bạn báo cáo (`warpSize` trên CUDA và HIP,
`threadExecutionWidth` của pipeline trên Metal). Mỗi kích thước khối được kiểm thử với độ dài
mảng là bội của nó và với độ dài không phải bội.

## Phương pháp

1. Trình giả lập (emulator) trước: `python3 labs/pmpp.py run 4.3 --backend emu`. Nó chạy một
   luồng cho đến khi luồng đó gặp một rào chắn (barrier) hoặc trả về, rồi đến luồng tiếp theo,
   theo ba thứ tự cố định. Một kernel cần rào chắn mà không có sẽ sai ở đó mọi lần.
2. Sửa một chỗ, chạy lại. Các thất bại thay đổi khi bạn tiến lên. Hãy dùng điểm chung của các
   trường hợp đạt và không đạt: những kích thước khối nào, và độ dài có phải là bội hay không.
3. Biết mỗi loại thất bại trông thế nào trên trình giả lập:
   - **Giá trị sai.** Thông báo nêu phần tử sai đầu tiên, khối của nó và luồng của nó.
   - **Một lần ghi ra ngoài mảng.** Buffer của trình giả lập đúng bằng mảng, nên
     AddressSanitizer dừng lần chạy ở lần ghi đầu tiên như vậy. Hãy đọc ba thứ gần đầu báo cáo
     của nó: `WRITE`, khung đầu tiên (`kernel.cu:` và một số dòng), và câu cho biết địa chỉ nằm
     trước hoặc sau vùng nhớ bao nhiêu byte. Các bài kiểm thử sau điểm dừng không được chạy.
4. `python3 labs/pmpp.py run 4.3 --backend race` chạy các luồng giả lập đồng thời dưới
   ThreadSanitizer. Một lần đọc và một lần ghi cùng một phần tử mà không có rào chắn ở giữa sẽ
   được báo cùng cả hai dòng mã nguồn (`python3 labs/pmpp.py doctor` cho biết trình biên dịch
   của bạn có dựng được backend `race` hay không).
5. Rồi đến GPU của bạn: `python3 labs/pmpp.py run 4.3`. Ở đó mảng nằm giữa các phần tử canh
   gác, mỗi hình dạng được khởi chạy 40 lần trên dữ liệu mới, và một trường hợp thất bại cho biết
   nó sai trong bao nhiêu lần trên 40.

Rào chắn của Metal nhận một cờ nêu tên vùng nhớ mà các lần truy cập vào đó được nó sắp thứ tự.
Các kernel trao đổi dữ liệu qua bộ nhớ threadgroup truyền `mem_flags::mem_threadgroup`. Hãy quyết
định xem đó có phải là cờ đúng ở đây không.

## Thí nghiệm (bài tập 5)

Bài tập 5 đưa ra cho bạn một khẳng định: với 32 luồng trong một khối, có thể bỏ `__syncthreads()`.
Hãy kiểm tra nó trên chính thiết bị của bạn khi mọi bài kiểm thử đã đạt.

Trước khi chạy, hãy viết ra hai dự đoán. Khi bỏ rào chắn và khối rộng đúng một warp (trường hợp
được đánh dấu `1 x width`), các bài kiểm thử trên GPU của bạn có thất bại không? Còn với khối rộng
bốn warp (`4 x width`)?

1. Xóa rào chắn khỏi `work/kernel.cu` (hoặc `work/kernel.metal`) đã sửa của bạn.
2. `python3 labs/pmpp.py run 4.3 --backend emu`: mọi trường hợp đều thất bại, theo mọi thứ tự.
3. `python3 labs/pmpp.py run 4.3` trên GPU của bạn, năm lần. Với mỗi kích thước khối, ghi lại một
   trường hợp thất bại trong bao nhiêu lần chạy trên năm, và trong bao nhiêu lần khởi chạy trên 40.
4. Đặt rào chắn trở lại.

Dù bảng của bạn cho thấy gì, hãy đọc nó đối chiếu với những gì các nhà sản xuất công bố, vì đó là
điều thiết bị tiếp theo được phép làm:

- Độ rộng không phải là hằng số. NVIDIA gọi `warpSize` là "commonly 32"; dưới HIP nó là 64 trên
  các thiết bị gfx9 của AMD và 32 trên gfx10 trở lên; Apple không công bố giá trị nào, chỉ nói chỗ
  đọc nó (`notes/today/ch04-facts.md`, A8, B6, C2).
- NVIDIA không hứa rằng các luồng của một warp luôn cùng nhịp: từ compute capability 7.0 trở đi,
  mỗi luồng có bộ đếm chương trình riêng, và lời khuyên của NVIDIA là đồng bộ hóa tường minh
  (A10). AMD có hứa chạy cùng nhịp (lockstep) trong một warp (B5). Apple không nói gì theo hướng
  nào (C2).
- Một khối rộng hơn một warp là nhiều warp, và không gì ngoài một rào chắn sắp thứ tự chúng với
  nhau. NVIDIA: "different warps execute independently" (A9). AMD: compute unit chuyển sang một
  warp thường trú khác mỗi khi một warp bị đình trệ (B8). Apple: các SIMD-group "make independent
  forward progress with respect to each other, in the absence of threadgroup barrier operations"
  (C2).

Một trường hợp đạt 40 lần đã cho bạn thấy 40 cách lập lịch. Nó không cho bạn thấy kernel đúng. Ba
thứ tự của trình giả lập cũng là những cách lập lịch, được chọn để khắc nghiệt.

Ở đây không có gì để đo hiệu năng.
