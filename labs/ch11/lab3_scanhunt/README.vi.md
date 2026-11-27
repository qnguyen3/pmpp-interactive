> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 11.3: săn race condition trong scan

Sách: race condition (tranh chấp dữ liệu) mà rào chắn (barrier) thứ hai của Kogge-Stone ngăn
chặn (mục 11.2, pp. 257-258), các phụ thuộc thật và giả của mục 11.3 (pp. 258-259), scan warp
và scan khối (mục 11.4, Figs. 11.8 và 11.9, p. 264) và scan có gộp việc cho luồng (thread
coarsening), phân ô thanh ghi (mục 11.6 và 11.7, Figs. 11.12 và 11.13, pp. 268 và 271). Các lỗi
là của chúng tôi. Một mục đứng riêng như (D4) bên dưới nằm trong `notes/today/ch11-facts.md`.

`work/kernel.cu` và `work/kernel.metal` chứa một scan **loại trừ** theo từng hàng, theo dạng
mà llama.cpp, MLX và PyTorch phát hành cho các hàng theo batch (D1, D3, D4): `out[r][j]` là tổng
của `in[r][0..j-1]`, và `out[r][0]` là 0.

- Mỗi hàng một khối. Khối đi qua hàng của nó theo từng đoạn `COARSE_FACTOR * blockDim.x` phần
  tử (4 phần tử mỗi luồng) và mang theo tổng của các đoạn trước.
- Mỗi warp nạp phần đoạn của nó bằng các lần đọc được hợp nhất vào bộ nhớ chia sẻ (shared
  memory), rồi mỗi lane đọc lại 4 phần tử liên tiếp và scan chúng tuần tự trong thanh ghi.
  Trung chuyển qua bộ nhớ chia sẻ là cách của sách (Figs. 11.12 và 11.13 trung chuyển cả đoạn);
  trung chuyển theo từng warp là của chúng tôi. llama.cpp và MLX không trung chuyển: mỗi luồng
  nạp dải của riêng nó trực tiếp (D1, D3).
- Tổng dải của các lane được scan bằng vòng lặp shuffle của Fig. 11.8; lane cuối của mỗi warp
  ghi tổng của warp vào `warpSums_s`; warp 0 scan các tổng đó (Fig. 11.9). Cùng một
  `warpSums_s` được dùng lại cho mọi đoạn.
- Mỗi lane cộng phần đứng trước dải của nó, và warp ghi phần của mình trở lại, được hợp nhất.

Nó biên dịch được. Nó có **ba lỗi độc lập**. Hãy tìm và sửa cả ba.

- Sửa `work/kernel.cu` (`ExclusiveRowScanKernel`; `launchExclusiveRowScan(in, out, rows,
  cols, block)` bên dưới nó là đúng) và `work/kernel.metal` (`exclusive_row_scan`).
  `work/launch.mm` (`encodeExclusiveRowScan`) cũng đúng. Cùng ba lỗi đó nằm trong cả hai
  kernel.
- Không phải lỗi: mặt nạ lane có kiểu và `warpSize` ở chỗ sách viết `0xffffffff` và 32 (HIP từ
  chối mặt nạ 32 bit và warp AMD có thể có 64 lane, B1); các phép kiểm tra biên, vốn Figs.
  11.12 và 11.13 không có; mỗi warp trung chuyển phần của riêng nó thay vì cả đoạn của khối;
  tổng của lane trước lấy từ một lần shuffle thay vì từ một mảng chia sẻ; và, trên Metal, chỉ
  số lane được truyền vào hàm phụ trợ, vốn không thể đọc thuộc tính luồng hay khai báo bộ nhớ
  threadgroup (C2).
- Kiểm thử: `python3 labs/pmpp.py run 11.3 --backend emu`, rồi `--backend race`, rồi
  `python3 labs/pmpp.py run 11.3` cho GPU của bạn.

## Các bài kiểm thử kiểm tra gì

- Khối 64 và 256 luồng; 1 và 3 hàng nhân 1, 33, 1000, 4097 và 32000 cột. Với 64 luồng, một
  đoạn là 256 phần tử, nên 1000 cột là 4 đoạn và 32000 là 125. Trình giả lập dừng ở 4097 cột;
  backend race chạy 2 hàng với 1, 33 và 1000 cột, 64 luồng.
- Ba loại hàng, trộn trong một batch để việc đọc nhầm hàng sẽ thất bại: toàn số một (nên
  `out[r][j] = j`), số nguyên từ -3 đến 3, và giá trị phân bố đều trong [-1, 1].
- Số một và số nguyên được so chính xác. Mọi tổng một phần là một số nguyên nhỏ hơn rất nhiều
  so với 2^24, mà một số float giữ chính xác, nên mọi thứ tự cộng cho cùng một số float: một
  sai khác là lỗi, không bao giờ là do làm tròn.
- `out[r][0]` phải chính xác là 0, phần tử đơn vị của +.
- Đầu ra khởi đầu là NaN, nên một phần tử không ai ghi sẽ hiện là NaN, và 64 từ canh gác sau
  cuối mảng phải còn nguyên.
- Trên thiết bị, mỗi trường hợp chạy 10 lần với dữ liệu mới, và một lần thất bại cho biết bao
  nhiêu lần khởi chạy bị sai.

## Dự đoán của bạn

Trước khi chạy bất cứ gì, hãy đọc `work/kernel.cu` và ghi ra những lỗi bạn tìm được bằng cách
đọc. Với mỗi lỗi, ghi ra GPU của bạn sẽ cho thấy nó (một giá trị sai) hay chỉ các trình giả lập
mới bắt được, và vì sao. Trang labs hỏi một con số từ đó (`ch11/labs/p-lab3-gpu`): **chỉ chạy
trên GPU của bạn, bao nhiêu trong ba lỗi sẽ làm một bài kiểm thử thất bại?** Sau đó chạy `emu`,
`race` và GPU của bạn, sửa từng lỗi một, và điền vào:

| Lỗi | `emu` | `race` | GPU của bạn |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |

## Phương pháp

1. Trình giả lập trước: `python3 labs/pmpp.py run 11.3 --backend emu`. Nó chạy một luồng cho
   đến khi luồng đó gặp một rào chắn, một lần shuffle hoặc điểm kết thúc, rồi luồng tiếp theo,
   theo ba thứ tự cố định.
2. Sửa một chỗ, chạy lại. Các trường hợp thất bại thay đổi dần khi bạn sửa. Hãy dùng điểm chung của
   các trường hợp qua và trượt: kích thước khối nào, bao nhiêu đoạn, loại hàng nào, cột nào sai
   đầu tiên.
3. Hãy biết các thất bại của trình giả lập trông thế nào:
   - **Bộ nhớ chia sẻ chưa ai ghi** chứa các byte `0xCD`, được lấp mới cho mọi khối. Dưới dạng
     float, đó là khoảng -4.3e8.
   - **Một giá trị đúng bằng gấp đôi giá trị đáng có**, hoặc lệch đúng bằng tổng của cả một
     dải, chỉ về số học, không phải thời điểm: nó thất bại ở mọi thứ tự luồng và trên mọi
     backend.
   - Điều `emu` không cho thấy được: nó chỉ giải quyết một rào chắn hay một lần shuffle khi
     không luồng nào chạy tiếp được nữa. Vì vậy mọi luồng hoàn tất phần việc của mình tới rào
     chắn hoặc lần shuffle tiếp theo trước khi bất kỳ luồng nào bắt đầu phần việc sau đó. Một
     lỗi mà tác hại của nó cần một warp nhanh đi trước một warp chậm cả một phần việc sẽ bị ẩn
     ở đó.
4. `python3 labs/pmpp.py run 11.3 --backend race` chạy các luồng giả lập thực sự đồng thời dưới
   ThreadSanitizer. Hai luồng chạm vào cùng một vùng nhớ mà không có gì sắp thứ tự chúng sẽ
   được báo cùng cả hai dòng mã nguồn, dù có giá trị nào bị sai hay không. ThreadSanitizer báo
   mỗi cặp dòng một lần mỗi lần chạy, nên một trường hợp sau lần báo đầu tiên có thể qua trong
   khi lỗi vẫn còn đó.
5. Sau đó là GPU của bạn: `python3 labs/pmpp.py run 11.3`. Ghi lại những lỗi nó đã cho bạn thấy
   và những lỗi nó đã giấu.

## Đọc bảng của bạn

Dù GPU của bạn cho thấy gì, hãy đối chiếu nó với những gì các nhà cung cấp ghi trong tài liệu,
vì đó là những gì thiết bị tiếp theo được phép làm:

- Shuffle-up để `delta` lane thấp nhất giữ giá trị của chính chúng, trên CUDA, HIP và Metal như
  nhau (A3, B1, C1). Một lane có được dùng giá trị nó nhận hay không tùy vào vị trí của nó
  trong warp.
- Các lane của một warp: NVIDIA cảnh báo rằng "the ability for threads to diverge and reconverge
  at sub-warp granularity makes such assumptions invalid" về việc chạy đồng bộ (lockstep), và
  yêu cầu `__syncwarp()`, vốn "provides memory ordering among the participating threads within
  a warp" (`notes/today/ch04-facts.md` A10). AMD ghi trong tài liệu việc chạy lockstep cho các
  lane của nó (ch04 B5); Apple cung cấp `simdgroup_barrier`, "an execution and memory barrier"
  (`infra-facts.md` C4). Mã chạy qua chỉ vì các lane tình cờ chạy lockstep vẫn có data race
  như thường.
- Các warp khác nhau chạy độc lập (NVIDIA, ch04 A9; Apple: SIMD group "make independent forward
  progress ... in the absence of threadgroup barrier operations", infra C5). Một warp đã xong
  đoạn của nó có thể đang ở bất kỳ đâu trong đoạn tiếp theo.
- "Reading from or writing to the same memory location without memory fences or
  synchronization results in undefined behavior" (`ch10-facts.md` A1). Một lần ghi sau một lần
  đọc cũng là race condition: rào chắn thứ hai của sách trong Fig. 11.3 tồn tại chỉ để sắp thứ
  tự cho một trường hợp như vậy (pp. 257-258), và scan MPS của PyTorch có một rào chắn mà chú
  thích của nó nói rằng thiếu nó thì "long (multi-iteration) scans race" (D4).

Một trường hợp đã qua 10 lần trên GPU của bạn cho bạn thấy 10 lịch chạy của một thiết bị. Nó
không cho bạn thấy kernel là đúng.

## Dung sai cho các hàng phân bố đều

Các hàng số nguyên là chính xác. Một hàng phân bố đều được so bằng
`|got - ref| <= 2 * depth * 2^-24 * (|x_0| + ... + |x_{j-1}|) + 1e-6`, trong đó `ref` là tổng
tính bằng double (vòng lặp của Fig. 11.1, chuyển thành loại trừ) và `depth = chunks per row + 16`.
Đó là dung sai của lab 11.1 (`1e-6` nhân tổng độ lớn, cộng `1e-6`, trong một đoạn tối đa 1024)
với số hạng tương đối được tăng lên cho phần mang sang: khoảng `2e-6` cho một hàng một đoạn,
`1.7e-5` cho 125 đoạn.

Vì sao nó tăng: mọi đầu ra là các đầu vào được cộng dọc theo một cây, và mỗi phép cộng làm tròn
tối đa `2^-24` tương đối. Một đầu vào đến được đầu ra của nó qua tối đa 3 phép cộng trong dải
của lane, 6 trong scan warp (warp tối đa 64 lane), 5 trong scan các tổng warp, một mỗi đoạn
trong phần mang sang, và 3 phép kết hợp các phần: tối đa `chunks + 17`, và phép cộng đầu tiên
của một dải (0 + x) là chính xác. Vì vậy sai số tối đa khoảng `depth * 2^-24` nhân tổng các độ
lớn. Hệ số 2 chừa chỗ cho việc làm tròn trong chính cận trên. Một lỗi ở đây làm giá trị lệch
đi bằng cả những phần tử nguyên vẹn, lớn hơn thế nhiều bậc độ lớn.

## Tính khả chuyển

- HIP: mặt nạ lane có kiểu và `warpSize` cho phép `kernel.cu` build qua `pmpp/vendors/hip.h`.
  Với warp rộng 64, một khối 64 là một warp: một lỗi số học cần warp thứ hai chỉ lộ ra với 256
  luồng. Chưa được biên dịch hay chạy trên phần cứng AMD ở đây: unverified.
- Metal: `launch.mm` đọc độ rộng SIMD group từ pipeline (`threadExecutionWidth`) và từ chối một
  threadgroup không gồm trọn các SIMD group, hoặc có nhiều SIMD group hơn số lane của một SIMD
  group.

Lab 11.6 xây kernel top-p của nó trên phiên bản đã sửa của scan này.

Không có gì để đo hiệu năng ở đây.
