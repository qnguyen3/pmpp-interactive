> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 10.2: săn lỗi phép rút gọn

Sách: các mục 10.7 đến 10.9 (pp. 237-246): `warp_reduce` (Fig. 10.13), kernel hai giai đoạn
(Fig. 10.16) và lần cộng dồn nhiều khối (Fig. 10.18), được mở rộng cho mọi độ dài như bài tập 5 yêu
cầu (p. 250). Các lỗi là của khóa học này, không phải của sách, trừ một lỗi mà chính Fig. 10.16 của
sách có như in.

`work/kernel.cu` và `work/kernel.metal` chứa `reduceAnyKernel`, một phép tính tổng trên mọi độ dài
`n`. Các khối có 128 luồng (thread). Mỗi luồng cộng hai phần tử, nên một khối phủ 256; mỗi warp rút
gọn các giá trị của nó bằng shuffle, warp 0 rút gọn các tổng của từng warp, và luồng 0 của mọi khối
cộng tổng của khối vào `*out`, thứ mà hàm khởi chạy đặt bằng 0 trước. Nó biên dịch được. Nó có **bốn
lỗi độc lập**. Hãy tìm và sửa cả bốn.

- Sửa `work/kernel.cu` (`reduceAnyKernel`; `launchReduceAny(in, out, n)` bên dưới nó là đúng) và
  `work/kernel.metal` (`reduce_any`). `work/launch.mm` cũng đúng. Cùng bốn lỗi đó có trong cả hai
  kernel.
- Không phải lỗi: mặt nạ lane có kiểu riêng và vòng lặp shuffle bắt đầu từ `warpSize/2`, trong khi
  sách viết `0xffffffff` và `WARP_SIZE/2` với `WARP_SIZE` bằng 32. HIP từ chối mặt nạ 32 bit và warp
  của AMD có thể rộng 64 (`notes/today/ch10-facts.md` B1). Trên Metal, độ rộng SIMD-group lấy từ
  `[[threads_per_simdgroup]]`, vì Apple không công bố giá trị nào.
- Kiểm thử: `python3 labs/pmpp.py run 10.2 --backend emu`, rồi `--backend race`, rồi
  `python3 labs/pmpp.py run 10.2` cho GPU của bạn.

Chương trình điều khiển (driver) kiểm thử các độ dài là bội của 256 và các độ dài không phải bội
(trên các trình giả lập tối đa 4097). Mỗi trường hợp được khởi chạy hai lần trên cùng buffer đầu ra
với dữ liệu khác nhau. Dữ liệu là các số nguyên từ 1 đến 8 và mọi tổng riêng phần đều nhỏ hơn 2^24,
nên mọi thứ tự cộng đều cho ra đúng cùng một số float: một tổng sai là một lỗi, không bao giờ do
làm tròn.

## Dự đoán của bạn

Trước khi chạy bất cứ thứ gì, hãy đọc `work/kernel.cu` và viết ra những lỗi bạn tìm được bằng cách
đọc. Với mỗi lỗi, hãy viết ra bạn dự đoán GPU của bạn sẽ để lộ nó (một tổng sai) hay chỉ các trình
giả lập bắt được, và vì sao. Trang labs hỏi một con số từ đó (`ch10/labs/p-lab2-gpu`): trong chín độ
dài mà GPU của bạn chạy, kernel như được giao sẽ đúng ở bao nhiêu độ dài trong cả 20 lần khởi chạy?
Rồi chạy `emu`, `race` và GPU của bạn, và so sánh. Điền bảng này khi bạn tiến hành:

| Lỗi | `emu` | `race` | GPU của bạn |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |

## Phương pháp

1. Trình giả lập (emulator) trước: `python3 labs/pmpp.py run 10.2 --backend emu`. Nó chạy một luồng
   cho đến khi luồng đó gặp một rào chắn (barrier), một lần shuffle hoặc điểm kết thúc của nó, rồi
   đến luồng tiếp theo, theo ba thứ tự cố định, nên một rào chắn bị thiếu sẽ sai ở đó mọi lần.
2. Sửa một chỗ, chạy lại. Các thất bại thay đổi khi bạn tiến lên. Hãy dùng điểm chung của các
   trường hợp đạt và không đạt: những độ dài nào, những thứ tự luồng nào.
3. Biết mỗi loại thất bại trông thế nào trên trình giả lập:
   - **Một lần shuffle đọc một lane không tham gia vào nó.** Trình giả lập trả về các byte `0xCD`,
     mà khi đọc như số float là một số âm lớn, và in một ghi chú nêu tên dòng của lần shuffle và
     lane mà nó đã đọc.
   - **Một lần đọc ra ngoài một mảng.** Mỗi buffer device, và bộ nhớ `__shared__` của mỗi khối, là
     đúng một khối heap ở đó, nên AddressSanitizer dừng lần chạy ở lần đọc đầu tiên như vậy.
     Hãy đọc `READ`, khung đầu tiên (`kernel.cu:` và một số dòng), và địa chỉ nằm sau vùng nhớ bao
     nhiêu byte. Các bài kiểm thử sau điểm dừng không được chạy.
   - **Bộ nhớ chia sẻ chưa ai ghi** chứa các byte `0xCD`, được điền mới cho mỗi khối: lại là giá
     trị sai, không phải điểm dừng.
4. `python3 labs/pmpp.py run 10.2 --backend race` chạy các luồng giả lập, và hai khối mỗi lần, thực
   sự đồng thời dưới ThreadSanitizer. Hai luồng chạm vào cùng vùng nhớ mà không có đồng bộ hóa sẽ
   bị báo cùng cả hai dòng mã nguồn, dù tổng có ra sai hay không. Một lỗi vô hình với `emu` và chỉ
   hiện ra ở đây. ThreadSanitizer báo mỗi cặp dòng một lần mỗi lần chạy, nên một trường hợp sau báo
   cáo đầu tiên có thể đạt dù lỗi vẫn còn đó.
5. Rồi đến GPU của bạn: `python3 labs/pmpp.py run 10.2`. Mỗi trường hợp được khởi chạy 20 lần (mười
   cặp) và một trường hợp thất bại cho biết nó sai trong bao nhiêu lần. Ghi lại lỗi nào GPU của bạn
   đã cho bạn thấy, và lỗi nào nó che đi.

Nếu một cách sửa cần phép toán nguyên tử (atomic): trên CUDA, `cuda::atomic_ref` của sách
(Fig. 10.18, cần `#include <cuda/atomic>`) hoặc `atomicAdd` kiểu cũ; trên HIP, `atomicAdd` kiểu cũ,
vì `<cuda/atomic>` đến từ libhipcxx ở đó và bản dựng HIP của khóa học không liên kết nó
(`notes/today/infra-facts.md` B2; lời giải tham chiếu chọn bằng `#if defined(__HIPCC__)`); trên
Metal, các hàm atomic chỉ nhận kiểu atomic, nên chính con trỏ phải được khai báo là atomic
(`notes/today/infra-facts.md` C1).

## Đọc bảng của bạn

Dù GPU của bạn cho thấy gì, hãy đọc nó đối chiếu với những gì các nhà sản xuất công bố, vì đó là
điều thiết bị tiếp theo được phép làm (`notes/today/ch10-facts.md` và `infra-facts.md`):

- Đọc một lane không tham gia vào một lần shuffle là không xác định. NVIDIA: "If the target thread
  is inactive, the retrieved value is undefined" (ch10 A3). Apple: "A thread needs to only read data
  from another active thread in the SIMD-group" (infra C2). Một giá trị không xác định có thể là giá
  trị đúng, mọi lần, trên một thiết bị.
- Bộ nhớ chia sẻ hoặc bộ nhớ threadgroup bắt đầu với bất cứ thứ gì đã có ở đó. Không gì hứa sẽ là
  số 0.
- Không có rào chắn, các warp tiến triển độc lập với nhau (NVIDIA: "different warps execute
  independently", ch04 A9; Apple: các SIMD-group "make independent forward progress", infra C5).
  Rào chắn cũng là thứ sắp xếp các lần ghi của một warp xảy ra trước các lần đọc của warp khác
  (ch10 A1).
- Hai luồng đọc và ghi cùng một vị trí mà không có đồng bộ hóa là một race condition (tranh chấp dữ
  liệu), và NVIDIA gọi đó là hành vi không xác định: "Reading from or writing to the same memory
  location without memory fences or synchronization results in undefined behavior" (ch10 A1). Điều
  đó đúng với hai khối cũng như với hai warp.

Một trường hợp đạt 20 lần trên GPU của bạn đã cho bạn thấy 20 cách lập lịch và câu trả lời của một
thiết bị cho một lần đọc không xác định. Nó không cho bạn thấy kernel đúng.

Ở đây không có gì để đo hiệu năng.
