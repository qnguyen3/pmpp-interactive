> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 16.1: wavefront phân ô, ô vuông rồi đến ô trượt

Sách: mục 16.6 (pp. 384-389, Figs. 16.8 to 16.12), mục 16.7 (pp. 389-396,
Figs. 16.15 to 16.20), và bài tập ở p. 386.

Bảng là ma trận điểm Smith-Waterman. `H` gồm (L_seq + 1) x (L_seq + 1) số int,
thứ tự hàng (row-major), với hàng 0 và cột 0 bằng không. Chuỗi `a` đánh chỉ số các hàng, `b` các cột:

    H[r][q] = max(0, H[r-1][q-1] + s, H[r][q-1] + SW_GAP, H[r-1][q] + SW_GAP)
    s = SW_MATCH if a[r-1] == b[q-1], else SW_MISMATCH

`sw.h` chứa các giá trị trong ví dụ của sách (+3, -3 và khoảng trống bằng 2, lưu là -2; p. 382).
Sách không giải thích chúng, và không có gì trong lab này phụ thuộc vào chúng. Một ô cần
các ô láng giềng phía bắc, phía tây và phía tây bắc, nên mỗi đường chéo phụ là một wavefront:
mọi ô của nó đều có thể chạy khi hai đường chéo phụ trước nó đã xong.

Được cung cấp sẵn: bản tham chiếu CPU (trong `main.cpp`) và một bản cơ sở trong `../shared/`
khởi chạy một lần cho mỗi đường chéo phụ, một luồng cho mỗi ô, mọi đầu vào đọc từ bộ nhớ toàn cục (global memory):
2 L_seq - 1 lần khởi chạy (p. 384; sách để mã của nó làm bài tập). Bạn viết hai
phiên bản phân ô (tiling) cần ít lần khởi chạy hơn nhiều.

## Phần 1: ô vuông (mục 16.6)

Một ô (tile) W x W cho mỗi khối gồm W luồng, giữ trong bộ nhớ chia sẻ (shared memory), và một lần khởi chạy cho mỗi
*mặt sóng (front)* (một đường chéo phụ của các ô): 2T - 1 lần khởi chạy với T = ceil(L_seq / W) ô mỗi cạnh.

- `max4`, `sqLoadN`, `sqLoadW`, `sqLoadNW`: Fig. 16.10 có thêm bước nhảy (stride) theo hàng. Phần tử
  (row, col) của ô là `swTile[row * stride + col]`, với `stride = W + pad`.
- `swSquareKernel(H, a, b, L, d, firstRow, stride)`, L = L_seq + 1: hàng ô của khối là
  `firstRow + blockIdx.x` và cột ô của nó là `d` trừ đi giá trị đó. Trong vòng lặp tính toán,
  chỉ số luồng là hàng trong ô. Có 2W - 1 bước; một luồng chỉ tính ô con của nó
  khi ô con nằm trong ô và trong bảng, và mọi luồng đều đi qua một
  `__syncthreads()` sau mỗi bước, bên ngoài phép kiểm tra đó.
- `sqStoreTile`: ở đây chỉ số luồng là cột trong ô, nên mỗi hàng của ô đến `H`
  thành một mảnh liền mạch, được hợp nhất (p. 389).
- `launchSwSquare(H, a, b, Lseq, W, pad)`: đúng các ô của mỗi mặt sóng (bài tập
  ở p. 386; sách khởi chạy T khối mỗi lần và để các khối thừa thoát ra),
  khối gồm W luồng, `W * stride` số int bộ nhớ chia sẻ động.

## Phần 2: ô trượt (mục 16.7)

Dịch mỗi hàng của ô sang trái thêm một cột so với hàng phía trên (một phép trượt với
hệ số m = -1, Fig. 16.14). Khi đó mỗi bước bên trong ô là một đường chéo phụ đầy đủ gồm W
ô, và một ô mất W bước thay vì 2W - 1. Cái giá: một hàng ô giờ phải chạy
sau hàng phía trên hai cột ô, nên có 3T - 1 mặt sóng, và các rìa của
bảng cắt ngang các ô.

- `hyInitTile`: đặt ô về không, rồi một rào chắn (barrier) (Fig. 16.19). Các ô con nằm ngoài
  bảng không bao giờ được tính, nhưng các vị trí của chúng vẫn được đọc như ô láng giềng.
- `hyLoadN`, `hyLoadW`, `hyLoadNW`: Fig. 16.17. Ô được lưu dưới dạng hình vuông, ô con
  của bước `q_tile` nằm ở cột `q_tile`, nên các ô láng giềng không còn ở chỗ cũ. Sách
  suy ra ô phía bắc và để hai ô còn lại cho bạn (p. 394).
- `hyStoreTile`: Fig. 16.18, có phép trượt, bỏ qua các ô con nằm ngoài bảng ở mọi phía.
- `swHyperKernel`: Fig. 16.16, cùng tham số với kernel ô vuông.
- `launchSwHyper`: đúng các ô của mỗi mặt sóng trong 3T - 1 mặt sóng.

Viết mọi phép tính chỉ số bằng `int`. Câu `const int tile_col = d - blockIdx.x` của sách
trừ các giá trị không dấu và trông cậy vào việc chuyển sang `int` cho ra một số
âm, điều mà C++17 để tùy trình biên dịch (`notes/today/ch16-facts.md` A3).

## Metal

`kernel.metal` chứa `sw_square` và `sw_hyper`. Ô là một tham số của kernel,
`threadgroup int* swTile [[threadgroup(0)]]`, được truyền cho các hàm trợ giúp `static inline` mang
tên như bên CUDA: một hàm trợ giúp MSL không thể tự khai báo bộ nhớ threadgroup (ch16-facts C3).
Rào chắn là `threadgroup_barrier(mem_flags::mem_threadgroup)`.

Trong `launch.mm`, `encodeSwSquare` và `encodeSwHyper` ghi một dispatch cho mỗi mặt sóng vào
encoder được truyền cho chúng: `setBytes` với `SwParams` của mặt sóng đó, rồi
`setThreadgroupMemoryLength` (một bội số của 16 byte, ch05-facts C8), rồi các
threadgroup. Encoder chạy các dispatch theo thứ tự và các buffer được
theo dõi xung đột (hazard-tracked), nên mỗi mặt sóng thấy được những gì các mặt sóng trước đã ghi (ch02-facts C9). Đó
là hình thức của Metal cho việc khởi chạy lại giữa các mặt sóng như trong sách.

## Các tệp

- `work/kernel.cu`: các hàm trợ giúp, `swSquareKernel`, `swHyperKernel`, `launchSwSquare`,
  `launchSwHyper`.
- `work/kernel.metal`: `sw_square`, `sw_hyper` và các hàm trợ giúp của chúng.
- `work/launch.mm`: `encodeSwSquare`, `encodeSwHyper`.
- Chỉ đọc: `main.cpp` (kiểm thử, benchmark, bản tham chiếu CPU), `sw.h`, `CMakeLists.txt`,
  `../shared/baseline.cu`, `../shared/baseline.metal`, `../shared/baseline_launch.mm`.

## Các lệnh

    python3 labs/pmpp.py run 16.1 --backend emu     # start here
    python3 labs/pmpp.py run 16.1 --backend race    # threads at the same time, under ThreadSanitizer
    python3 labs/pmpp.py run 16.1                   # every backend found
    python3 labs/pmpp.py run 16.1 --bench           # also benchmark (real devices only)

## Các bài kiểm thử kiểm tra gì

Mọi ô của bảng, kể cả hàng 0 và cột 0, phải bằng đúng bản tham chiếu CPU,
và 256 số int được cấp phát sau bảng phải giữ nguyên. Trước mỗi lần chạy,
phần bên trong bảng chứa 1234567, nên một ô không ai ghi sẽ được báo là "never
written". Cả hai kernel với pad 0 và 1, cùng bản cơ sở:

| Backend | L_seq | W |
|---|---|---|
| emu | 1, 2, 7, 8, 9, 31, 33, 50 | 3, 4, 5, 8, 16 |
| race | 9, 20 | 4, 8 |
| metal, cuda, hip | 1, 31, 32, 33, 100, 1000, 2049 | 8, 16, 32, 64 |

Trên thiết bị thật, một W cần nhiều luồng hơn hoặc nhiều bộ nhớ chia sẻ (threadgroup) hơn cho mỗi
nhóm so với mức thiết bị báo cáo sẽ bị bỏ qua kèm một thông báo.

Đầu vào cho mỗi độ dài: các chuỗi ngẫu nhiên trên 4 ký hiệu (3 seed), hai chuỗi
giống hệt nhau, hai chuỗi hoàn toàn không khớp (một bảng toàn số không), và một cặp có cột đầu tiên
khớp thường xuyên (`b[0]` là ký hiệu xuất hiện nhiều nhất của `a`). Cặp cuối có mặt vì
kernel ô trượt: rìa trái của bảng cắt ngang các ô trượt, và một ô ở cột 1
đọc ô láng giềng từ các vị trí trong ô mà không luồng nào tính. Trên trình giả lập (emulator), bộ nhớ chia sẻ
không ai ghi chứa các byte `0xCD`, một số int âm lớn, và phép lấy max với 0 che đi phần lớn
điều đó: thiếu `hyInitTile` chỉ lộ ra ở chỗ mà giá trị khớp của một ô cột 1 lẽ ra phải thắng, và cặp này
tạo ra đúng chỗ đó. Trên GPU, không gì bảo đảm một vị trí chưa ghi chứa gì, nên cùng lỗi đó
có thể vượt qua ở đó, ở một số lần chạy hoặc ở tất cả: hãy dựa vào trình giả lập cho lỗi này. Trên
trình giả lập, ba độ dài lớn nhất chạy ba trong số các đầu vào này (seed ngẫu nhiên 1, giống hệt,
cột đầu tiên), vì trình giả lập phải trả giá cho mọi rào chắn của mọi luồng.

Mỗi lỗi sau, khi là lỗi duy nhất trong một lời giải đúng, đều thất bại trên `emu`: rào chắn
bị gỡ khỏi vòng lặp bước (ở kernel nào cũng vậy); `<` thay cho `<=` ở cột cuối của các ô
trượt; lệnh gọi `hyInitTile` bị gỡ; `q >= 1` bị bỏ khỏi phép kiểm tra tính toán của ô
trượt; `sqLoadNW` thiếu trường hợp `q_tile == 0`; thiếu một ô trong số khối của mỗi mặt sóng
(ở kernel nào cũng vậy); và `stride = W + 1` mà chỉ có `W * W` số int bộ nhớ chia sẻ.
Một rào chắn bị dời vào trong phép kiểm tra biên, ở kernel nào cũng vậy, cũng thất bại dưới cả ba thứ tự
luồng. Ba trong số này dừng lần chạy bằng một báo cáo AddressSanitizer chỉ ra dòng
(`q >= 1` bị bỏ, thiếu trường hợp `q_tile == 0`, bộ nhớ chia sẻ thiếu); các lỗi còn lại
hiện ra thành các ô sai.

## Trước khi bạn đo hiệu năng

Benchmark tăng L_seq từ 1024, gấp đôi mỗi lần, cho đến khi kernel ô vuông với W = 32 mất ít
nhất 20 ms hoặc L_seq đạt 8192. Sau đó nó đo thời gian bản cơ sở, và cả hai kernel với W = 8,
16, 32 và 64 (nếu thiết bị cho phép) với pad 0 và 1. Mỗi hàng cho thấy số lần khởi chạy,
thời gian tốt nhất và GCUPS (số lần cập nhật ô mỗi giây / 1e9); các dòng cuối là ba
tỉ số dưới đây, đã đo. Hãy chạy ít nhất hai lần: một khác biệt không tồn tại qua
lần chạy thứ hai là nhiễu. Ghi ra ba con số trước:

1. **Ô vuông với W = 32 nhanh hơn bản cơ sở bao nhiêu lần?** Ủng hộ: 2T - 1
   lần khởi chạy thay vì 2 L_seq - 1, và ô láng giềng lấy từ bộ nhớ chia sẻ. Phản đối: một mặt sóng
   chứa nhiều nhất T ô, mỗi ô một khối gồm W luồng, và bên trong một ô, một bước có
   từ 1 đến W ô con cần tính, trung bình khoảng W/2 (Eq. 16.3).
2. **Thời gian ô trượt chia cho thời gian ô vuông, với W = 32, không đệm.** Ủng hộ: W bước mỗi ô
   thay vì 2W - 1, mọi luồng đều bận ở mọi bước (Eq. 16.4). Phản đối: 3T - 1 lần khởi chạy
   thay vì 2T - 1, nhiều nhất khoảng T/2 + 1 ô mỗi mặt sóng thay vì T, thêm các ô không trọn
   ở các rìa (ch16-facts E2), và mỗi bước ghi và đọc một cột của
   hình vuông được lưu, cách nhau W số int từ luồng này sang luồng kế.
3. **Thời gian ô trượt có một khoảng trống mỗi hàng chia cho không có, với W = 32.** Theo
   mô hình bank của NVIDIA (32 bank, mỗi bank 4 byte), cột không đệm là một xung đột 31 đến 32 chiều ở
   W = 32, và một khoảng trống mỗi hàng loại bỏ nó (ch16-facts E1, phép tính của chúng tôi trên mô hình đó).
   AMD có 32 hoặc 64 bank tùy kiến trúc (B3). Apple không ghi tài liệu về
   bank nào của bộ nhớ threadgroup cả (infra-facts C13), nên trên Metal đây là một phép đo
   không có mô hình nào của nhà sản xuất đứng sau.

Rồi hãy nói việc đệm đẩy kernel **ô vuông** theo hướng nào. Các truy cập của nó chạy dọc một
đường chéo phụ của ô, cách nhau W - 1 số int từ luồng này sang luồng kế, mà cùng mô hình
NVIDIA đó cho là không xung đột khi không đệm; một khoảng trống mỗi hàng biến khoảng cách đó thành W số int, mà
mô hình cho là có xung đột, tới 32 chiều ở W = 32 (E1). Sách chỉ đệm ô trượt.

## Chỗ khác với sách

- Mọi phép tính chỉ số đều là `int` (như trên).
- Mỗi lần khởi chạy phủ đúng các ô của một mặt sóng, thông qua `firstRow` (bài tập ở
  p. 386).
- Việc đệm là một khoảng trống mỗi hàng ô, thông qua `stride`. Macro `pad()` của sách chèn một
  khoảng trống sau mỗi `NUM_BANKS` số int, tức là một khoảng mỗi hàng chỉ khi W bằng số
  bank; với W = 64 trên NVIDIA nó để lại xung đột 2 chiều trong khi một khoảng trống mỗi hàng thì không
  (`notes/ch16.md`, "Unclear in the PDF"; ch16-facts E1).
- Phép ghi của ô trượt kiểm tra `q >= 1`. Fig. 16.18 chỉ kiểm tra `q < L` với `L` không dấu,
  nên với một `q` âm, phép kiểm tra thất bại do giá trị bị quay vòng; với `L` kiểu `int` nó sẽ ghi vào
  hàng phía trên.
- Một hằng số khoảng trống duy nhất. Sách đặt tên hai mức phạt có cùng giá trị, và gán
  tên theo một cách ở Eq. 16.2 và theo cách ngược lại ở Fig. 16.6 cùng kernel (`notes/ch16.md`,
  "Unclear in the PDF"); với giá trị bằng nhau thì không có gì khác biệt.
- Các đoạn mã để `MATCH`, `MISMATCH`, `DELETION`, `INSERTION` và kiểu chuỗi `T`
  không được định nghĩa; ở đây chúng là các hằng số của `sw.h` và `int`.

## Vượt ra ngoài sách

- Bài tập 6 thay `max4` bằng hàm nội tại DPX `__vimax3_s32_relu(nw + s, w + gap,
  n + gap)`, cho cùng giá trị với mọi đầu vào. Nó chỉ có trên CUDA; bảng tính năng của
  NVIDIA ghi nó là lệnh gốc trên compute capability 9.0 và 10.x và là vài
  lệnh ở những nơi khác, và chính mã nguồn CCCL của NVIDIA báo một lỗi trình biên dịch ở
  các chuỗi phụ thuộc của nó trước CUDA 13.4 (ch16-facts A2). HIP không có hàm DPX nào (B2).
  Dạng gần nhất trên Metal là `max(max3(a, b, c), 0)` (Metal 2.1 trở lên, C2). Không được kiểm thử
  ở đây; nếu bạn thử, hãy giữ `max4` làm phương án dự phòng và so sánh hai cách.
