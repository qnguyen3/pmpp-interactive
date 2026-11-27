> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Đoạn mã (snippets)

Mã mà các bài học của chương 14 trình bày và không phải là một lab. Các tệp là mã nguồn `.cu` thật
để bước kiểm tra front-end CUDA trong `./verify.sh` phân tích được chúng, và để bài học hiển thị
đúng tệp đã được kiểm tra chứ không phải một bản sao có thể lệch đi. Chúng không được trình chạy lab
build. Các chương trình điều khiển (driver) trong `check/` chạy chúng trên trình giả lập (emulator)
CPU của khóa học; phép kiểm thử của mỗi driver nằm trong `app/src/chapters/ch14/`. Mỗi đoạn mã một hàng.

| Tệp | Xuất hiện ở | Cách dùng | Driver | Kiểm thử |
|---|---|---|---|---|
| `s142_oddeven.cu` | mục 14.2: `sort_kernel` (Fig. 14.2, 13 dòng của trang sách) và vòng lặp host `oddEvenSort` | nạp lúc chạy | `check/s142_check.cu` | `S142.snippets.test.ts` |
| `s144_floatkey.cu` | mục 14.4: phép biến đổi float thành khóa và một kernel áp dụng nó cho một hàng | vùng `floatkey` của nó, ghim vào bản sao của bài học | `check/s144_check.cu` | `S144.snippets.test.ts` |
| `s145_radix_iter.cu` | mục 14.5: Fig. 14.7 tách tại phép scan thành ba lần khởi chạy, cùng phép scan một khối nằm giữa chúng | các vùng của nó, ghim vào các bản sao của bài học, từng dòng một | `check/s145_check.cu` | `S145.snippets.test.ts` |
| `wrap_blocksort.cu` | trang tổng kết, câu hỏi `ch14/wrap/q-blocksort`: một phép sắp xếp chẵn-lẻ cấp khối, cố ý không có rào chắn giữa các bước | nạp lúc chạy, chỉ phần kernel | `check/wrap_check.cu` (chứa bản đã sửa) | `Wrap.snippets.test.ts` |

Mỗi phép kiểm thử dịch đoạn mã và driver của nó bằng `labs/tools/cu2emu.py` đặt cạnh nhau, để
`#include "../<snippet>.cu"` của driver tìm thấy bản dịch, rồi build với AddressSanitizer dưới ba
thứ tự luồng của trình giả lập. Mỗi phép kiểm thử được bỏ qua ở nơi không tìm thấy trình biên dịch C++
hoặc `python3`. Cả bốn tệp đều qua bước kiểm tra front-end CUDA. Chưa có gì ở đây chạy trên GPU, và
không tệp nào được build cho HIP.

## `s142_oddeven.cu`

Fig. 14.2 với hai dòng được đổi (dòng 05 `i + 1 < N`, để `N == 0` không thể tràn vòng; dòng 10
`atomicExch`, để cờ không phải là một race condition), cộng thêm một vòng lặp host chỉ dừng sau hai
bước yên lặng liên tiếp. Phép kiểm thử kiểm tra:

- trên trình giả lập (bật AddressSanitizer, ba thứ tự luồng): các hàng và cờ của Fig. 14.1 từng bước
  một, các bước mà vòng lặp host khởi chạy và đầu ra của nó cho các trường hợp trong
  `src/sim/oddeven.ts`, khóa ngẫu nhiên so với `std::sort`, hai khối mỗi bước, và `N == 0` không chạm
  vào gì;
- trên biến thể phát hiện race (ThreadSanitizer, các luồng chạy đồng thời thật sự): một bước của
  Fig. 14.2 đúng như in (lệnh thường `*hasChanged = 1;`, biên dịch với `-DSNIPPET_AS_PRINTED`) trên đầu
  vào bị đảo ngược được báo là race condition tại dòng đó; `atomicExch` của đoạn mã thì không. Phép này
  được bỏ qua ở nơi không tìm thấy ThreadSanitizer hoạt động được.

## `s144_floatkey.cu`

Phép biến đổi của mục 14.4, trên các bit của float dưới dạng `unsigned int` (cách này không cần
intrinsic ép kiểu bit trên backend nào, và trên Metal một phép kiểm tra bằng không phía float có thể bị
viết lại). Phép kiểm thử so sánh từng khóa với `src/sim/floatkey.ts`, từng bit một, trên các giá trị của
E1 (`notes/today/ch14-facts.md`), hai NaN và 600 mẫu bit khác, và kiểm tra thứ tự giảm dần cùng thứ tự
các phần tử bằng nhau của một phép sắp xếp ổn định trên các khóa đó.

## `s145_radix_iter.cu`

Fig. 14.7 dưới dạng ba kernel theo thứ tự trên một stream: trích các bit, một phép scan loại trừ trên
N + 1 phần tử (của khóa học này, một khối, có nhãn "Not the book's code"), phân tán. Phép kiểm thử kiểm
tra các hàng của Fig. 14.6 cho cả bốn bit, bốn lượt của Fig. 14.4, khóa 8 bit ngẫu nhiên trên mọi bit so
với `src/sim/radixsort.ts`, và 1,000 khóa 32 bit ngẫu nhiên được sắp xếp trong 32 lượt so với `std::sort`.

## `wrap_blocksort.cu`

Một khối sắp xếp tới `MAXN` khóa trong bộ nhớ chia sẻ (shared memory) với n bước chẵn-lẻ cố định: một
mạng sắp xếp bên trong một khối. Nó không có rào chắn giữa các bước, đó chính là lỗi mà câu hỏi nhắm
tới. Driver chứa bản đã sửa, thêm một `__syncthreads()` ở cuối thân vòng lặp, và phép kiểm thử kiểm tra
rằng đây là khác biệt duy nhất. Trên trình giả lập, vốn chạy từng luồng một giữa các rào chắn, kernel như
được trình bày để lại 64, 129 và 256 khóa chưa được sắp xếp (nhưng mọi giá trị vẫn có mặt) ở cả ba thứ tự
luồng, còn bản đã sửa thì sắp xếp được chúng, và cả 256 danh sách gồm tám số 0 và 1, ở cả ba thứ tự.
Không có phép kiểm thử biến thể race. Khi thử vào 2026-11-20, biến thể race không báo gì cho kernel như
được trình bày: lúc đó nó đánh dấu mọi `__syncthreads()` của một khối bằng một token ThreadSanitizer dùng
chung, điều này che giấu các race giữa hai rào chắn. Giờ nó luân phiên hai token cho mỗi rào chắn. Khi thử
lại bằng tay vào 2026-11-21 (không tất định, và không thuộc phép kiểm thử), nó báo race condition trong
kernel như được trình bày ở phép so sánh và cả hai lệnh ghi (dòng 20, 22 và 23); dòng nào xuất hiện, và
liệu giá trị có bị mất trong trường hợp 256 khóa hay không, thay đổi theo từng lần chạy. Nó không báo gì
cho bản đã sửa. Một lần chạy race sạch vẫn chỉ là bằng chứng, không phải chứng minh.
