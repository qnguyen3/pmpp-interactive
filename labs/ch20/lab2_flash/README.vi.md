> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 20.2: lượt thuận của flash attention (mốc dự án M6, phần hai)

Sách: mục 20.5 (pp. 492-504): online softmax của Eqs. 20.3 đến 20.6, cách phân ô (tiling) của
Fig. 20.8 và kernel của Figs. 20.9 đến 20.16; các bài tập 1 và 3 (p. 511). Chiều batch và chiều
head, `n` bất kỳ, logsumexp `L`, các ô bị bỏ qua và mọi thứ trên Metal không có trên trang sách:
chúng là của khóa học này, và lý do nằm ở bên dưới. Một mục đứng riêng như (A3) hay (C1) nằm trong
`notes/today/ch20-facts.md`.

Hãy viết attention nhân quả thành một kernel duy nhất, để `n x n` điểm số của lab 20.1 không bao
giờ tồn tại trong bộ nhớ thiết bị:

```
launchFlashAttention(Q, K, V, O, L, batch, heads, n, scale)
```

| Tensor | Hình dạng |
|---|---|
| `Q`, `K`, `V` (vào), `O` (ra) | `[batch, heads, n, 128]` |
| `L` (ra) | `[batch, heads, n]` |

`float` theo thứ tự hàng (row-major), batch đứng trước, chiều head 128 (`HEAD_DIM`, d của sách).

## Hợp đồng

- `n >= 1`, giá trị bất kỳ: không chỉ bội của 32. `batch * heads` khối dọc theo chiều y của
  lưới (grid).
- Nhân quả: hàng `r` thấy các cột `c <= r` (ma trận vuông, nên mặt nạ của sách và mặt nạ căn góc
  dưới phải của lab 20.1 trùng nhau).
- Với `s = scale * Q K^T`, `m` là `s` nhìn thấy được lớn nhất của hàng và `D` là tổng của
  `exp(s - m)` trên các cột nhìn thấy được: `O = (sum of exp(s - m) V) / D` và
  `L = m + log(D)`, cùng `L` mà lab 20.1 ghi cho cùng các điểm số.
- Không có buffer `n x n` nào cả. Chương trình điều khiển (driver) chỉ cấp phát Q, K, V, O và
  L, và kiểm tra điều đó.

## Thiết kế (của sách)

- **Một khối 512 luồng cho mỗi dải gồm `B_r = 32` hàng của Q**: 16 warp, mỗi warp 2 hàng
  (`B_r_warp`). Chiều x của lưới đi qua các dải bằng vòng lặp ngoài của sách (`i` bắt đầu ở
  `blockIdx.x` và tăng theo bước nhảy (stride) `gridDim.x`, p. 497); chiều y của lưới đi qua các cặp
  (chuỗi, head), "additional dimensions of thread blocks" mà p. 495 nhắc tới.
- **Lưu trữ** (p. 497): mỗi luồng giữ phần của nó trong các hàng Q và hàng O của warp, cùng giá
  trị lớn nhất đang chạy `m_i` và mẫu số `D_i` của hai hàng của nó, trong các mảng cục bộ được
  dự định nằm trong thanh ghi (register). Khối dùng chung ba ô: `KT_j` (một ô của K, chuyển vị và
  có đệm), `S_i` (các điểm số của dải so với ô đó, được P ghi đè tại chỗ) và `V_j`.
- **Các lane được xen kẽ** (Fig. 20.10): lane L giữ các cột L, L + 32, L + 64, L + 96 của một
  hàng Q hoặc O. Để tính một điểm số, mọi lane cần mọi phần tử của hàng Q, nên lane sở hữu phát
  từng phần tử bằng một lệnh shuffle (Fig. 20.12).
- **Hai vòng lặp.** Qua các ô j: nạp `KT_j` và `V_j`, rào chắn (barrier), rồi mỗi warp đi qua
  hai hàng của nó (`ii`): tính đoạn của S và giá trị lớn nhất của nó, cập nhật `m` và co giãn lại
  `D`, biến S thành P tại chỗ và cộng tổng của nó vào `D`, co giãn lại O và cộng P nhân `V_j`.
  Rồi đến rào chắn thứ hai, trước khi lần nạp tiếp theo ghi đè các ô (Fig. 20.9 dòng 39). Sau ô
  cuối cùng, `store_O` chia cho `D` một lần.
- **`B_c`**, độ rộng ô, là `TILE_WIDTH` từ bản build: 32 như trên trang sách (mặc định) và 16,
  cũng được build cho benchmark. Mỗi lane sở hữu nhiều nhất một cột của một ô, nên với `B_c = 16`
  các lane 16 đến 31 không có cột nào: chúng vẫn tham gia mọi lệnh shuffle và mang theo phần tử
  đơn vị (`-FLT_MAX` cho max, 0 cho tổng).

**Được cung cấp, đã sửa sẵn**: các hằng số, kiểu mặt nạ lane và warp logic 32 lane, `load_Q`,
`load_KT_and_V` (điền 0 khi quá `n`, đệm qua `addr()`) và `store_O` (kiểm tra biên, ghi
`O = O~ / D` và `L = m + log(D)`).

**Phần của bạn**: `initialize` (bài tập 3), `compute_S_and_max`, `update_m_and_D`,
`compute_P_and_update_D`, `compute_O`, hai vòng lặp của kernel với cả hai rào chắn khối, và hàm
khởi chạy (bài tập 1). Trên Metal, thêm phần của `launch.mm` được mô tả ở mục "Metal".

## Chỗ khác với trang sách

1. **Một rào chắn warp mà trang sách bỏ sót** (A3). Trong `compute_P_and_update_D` mỗi lane ghi
   một phần tử của hàng P của warp; trong `compute_O` mọi lane đọc cả hàng. Các lệnh shuffle ở
   giữa không sắp thứ tự bộ nhớ ("These intrinsics do not imply a memory barrier"), nên không gì
   bảo đảm các lần đọc thấy các lần ghi. `__syncwarp()` sau lần ghi P thì bảo đảm. llama.cpp có
   cùng rào chắn đó ở cùng chỗ (D1). Lab 20.3 xoay quanh nó.
2. **Không có `NEG_INFINITY`** (A5). Không header CUDA nào định nghĩa nó, và IEEE cho NaN với
   `(-inf) - (-inf)`. Mặt nạ được kiểm tra theo chỉ số, nên không điểm số bị che nào từng được
   dùng; phần tử đơn vị của max là `-FLT_MAX`; giá trị lớn nhất đang chạy bắt đầu ở
   `-FLT_MAX/2`, như của llama.cpp, để không phép trừ nào có thể tràn số. Đáp án của bài tập 3
   cho O và D, 0, là của FlashAttention-2 (A7).
3. **Không dùng CUB.** `cub::Max()` và `cub::Sum()` đã không còn (infra A11), và một
   `WarpReduce::TempStorage` duy nhất dùng chung cho cả 16 warp như trên trang sách không phải
   cách dùng mà CUB ghi trong tài liệu (A2). Một cây bướm (butterfly) `__shfl_xor_sync` để lại
   max hoặc tổng ở mọi lane, nên phép phát từ lane 0 của trang sách biến mất.
4. **Một mặt nạ có kiểu và một độ rộng tường minh ở mọi lệnh shuffle.** `0xFFFFFFFF` không biên
   dịch được cho HIP (B1). Trên một wavefront AMD rộng 64, warp 32 lane của sách là nửa
   wavefront, và shuffle của HIP, có độ rộng mặc định 64, sẽ đọc vắt qua hai nửa; truyền độ rộng
   32 giữ mỗi lần trao đổi bên trong warp logic của nó (B2).
5. **`L` ở chỗ trang sách ghi `out_D`.** Riêng mẫu số không thể dựng lại P nếu thiếu giá trị lớn
   nhất của hàng, thứ trang sách không bao giờ ghi; FlashAttention-2 lưu `m + log(D)` (A7).
6. **`n` bất kỳ.** Trang sách chia N cho `B_r` và `B_c`, nên N phải là bội của cả hai (notes,
   "Unclear in the PDF"). Ở đây số ô được làm tròn lên, các hàm nạp điền 0 vào các hàng quá `n`,
   mặt nạ cũng che các cột `>= n`, và `store_O` bỏ qua các hàng `>= n`.
7. **Các ô được khai báo trong kernel** và truyền cho các hàm trợ giúp. Metal từ chối mảng
   threadgroup trong hàm trợ giúp (C2), và AMD chỉ ghi trong tài liệu phạm vi kernel (B3).
8. **Các ô nằm hoàn toàn phía trên đường chéo có thể được bỏ qua**, như FlashAttention-2 làm
   (A7); trang sách chạy mọi ô cho mọi dải. Lời giải tham chiếu bỏ qua chúng. Tùy chọn: một
   kernel chạy chúng vẫn vượt qua, chỉ tốn thêm thời gian.
9. `d` của trang sách ở đây là `HEAD_DIM` (một macro tên `d` sẽ viết lại mọi `d` trong tệp), và
   `exp`, `log` được viết là `expf`, `logf`.

## Metal

Cùng cấu trúc, với từ vựng của Metal (bảng ở đầu `kernel.metal`): `simd_broadcast` cho phép phát
Q (lane nguồn giống nhau ở mọi lane, như nó yêu cầu), `simd_max` và `simd_sum` cho các phép rút
gọn (kết quả có ở mọi lane), `simdgroup_barrier(mem_flags::mem_threadgroup)` cho rào chắn warp
(C4) và `threadgroup_barrier(mem_flags::mem_threadgroup)` cho các rào chắn khối.

`launch.mm` cũng là phần của bạn:

- **Chọn `B_c` theo thiết bị.** Các ô của sách ở `B_c = 32` chiếm 37,376 byte; Apple ghi trong
  tài liệu 32 KB bộ nhớ threadgroup (C1), và việc tạo pipeline thất bại. Hãy lấy `B_c` lớn nhất
  trong {32, 16, 8} có các ô vừa `maxThreadgroupMemoryLength` (16 cho 18,688 byte, F1).
- **Biên dịch với giá trị đó.** `kernel.metal` đọc `TILE_WIDTH`; hãy truyền nó qua
  `MTLCompileOptions.preprocessorMacros` (C6).
- **Kiểm tra pipeline.** Bố cục lane cần các SIMD group đúng 32 (`threadExecutionWidth`; Apple
  không ghi trong tài liệu độ rộng cố định nào, C3) và 512 luồng mỗi threadgroup
  (`maxTotalThreadsPerThreadgroup`, một thuộc tính của pipeline). Nếu một trong hai không đạt,
  hãy trả về nil kèm một thông điệp nói rõ cái nào.
- **Dispatch** một threadgroup 512 luồng cho mỗi dải theo chiều x và cho mỗi (chuỗi, head) theo
  chiều y.

## Tệp

- `work/kernel.cu` (CUDA; cũng được biên dịch cho HIP và hai trình giả lập). Các TODO được xếp
  theo thứ tự.
- `work/kernel.metal` và `work/launch.mm` (Metal).
- `main.cpp`: chương trình điều khiển. Các bài kiểm thử, tham chiếu trên CPU và các dung sai nằm
  trong `../shared-g11/flash_test.hpp`, dùng chung với lab 20.3.
- Được cung cấp cho benchmark, không sửa: `../shared-g11/unfused.cmake` sao chép `launchLinear`
  tham chiếu của lab 15.6 (M3), `launchCausalSoftmax` tham chiếu của lab 20.1 và `launchGemm`
  tham chiếu của lab 5.3 (M2) vào bản build, có đổi tên. Nếu thiếu một trong các lời giải đó, các
  hàng không gộp sẽ vắng mặt.

## Lệnh

- Kiểm thử: `python3 labs/pmpp.py run 20.2 --backend emu` trước (chậm: mỗi trường hợp mất một
  lúc; nó chạy bốn trường hợp), rồi `--backend race` (bên dưới), rồi
  `python3 labs/pmpp.py run 20.2` trên GPU của bạn.
- Đo: `python3 labs/pmpp.py run 20.2 --bench`. Trên CUDA và HIP lệnh này cũng chạy bản build
  `B_c = 16`.

## Các bài kiểm thử kiểm tra gì

- **Hình dạng**: `n` = 1, 2, 31, 32, 33, 63, 64, 65, 100, 257, 1000, 2048 (bội và không phải bội
  của `B_r` và `B_c`), mỗi cái với (batch, heads) = (1, 1) và (2, 3). `emu` chạy `n` = 1, 33 (hai
  head), 64 và 70; `race` chạy `n` = 33 và 40.
- **Dữ liệu**, một công thức cho mỗi head, mọi công thức ở mọi hình dạng:
  - Q, K, V phân bố đều trong [-1, 1];
  - **giá trị lớn nhất đến muộn**: điểm số của cột c tăng khoảng 0.04 mỗi cột, nên giá trị lớn
    nhất đang chạy tăng ở mọi ô và mọi lần co giãn lại đều quan trọng. Một lần co giãn lại bị
    thiếu hoặc đặt sai chỗ sẽ thất bại ở đây;
  - mọi điểm số dịch đi khoảng +90. `exp(90)` vượt khỏi phạm vi của float, nên một kernel không
    trừ giá trị lớn nhất sẽ tràn số (một độ dịch có hàm mũ vẫn vừa trong một float sẽ không bắt
    được lỗi đó).
  - V có mức khác nhau ở mọi cột, để một sự nhầm lẫn cột trong O lộ ra.
- **Kiểm tra**: mọi hàng của O và L của mọi head so với một tham chiếu double trên CPU, với các
  dung sai dưới đây; O và L khởi đầu là NaN, nên một phần tử không ai ghi sẽ bị bắt; 64 giá trị
  canh sau O và sau L phải không bị chạm tới; và không có buffer thiết bị nào gồm `n x n` số
  float.

## Dung sai, và lý do

Tham chiếu (`flash::refRow`) tính mỗi hàng bằng double từ cùng các đầu vào float. Mỗi hàng có
cận riêng, từ dữ liệu riêng của nó. Với u = 2^-24 (một lần làm tròn) và, cho hàng đó:

- `A`, giá trị `scale * sum |q k|` lớn nhất trên các cột nhìn thấy được (các số hạng của một tích
  vô hướng lớn đến đâu);
- `X`, giá trị `m - s` lớn nhất (các số mũ đi xuống sâu đến đâu);
- `T`, số ô mà hàng trải qua;

sai số tương đối của đầu ra chưa chuẩn hóa và của `D` nhiều nhất khoảng

```
e = 2 (d + 2) u A  +  u X  +  2u (3 + 2X)  +  2u (3T + 2X)  +  (B_c + 2T + 8) u
```

từng số hạng một:

- **Điểm số.** Một tích vô hướng 128 số hạng được cộng dồn theo thứ tự, rồi co giãn: nhiều nhất
  `(d + 1)` lần làm tròn của các số hạng có tổng không lớn hơn `A`, và `s - m` mang sai số của hai
  điểm số: `2 (d + 2) u A` tuyệt đối trong số mũ.
- **`s - m`** được làm tròn một lần: `u X`.
- **Hàm mũ.** Cận lớn nhất trong ba backend là `exp` nhanh của Metal, `3 + floor(|2x|)` ulp
  (ch10-facts C4); `expf` của CUDA trong phạm vi 2 ulp (ch10-facts A11). Một ulp nhiều nhất là
  `2u` tương đối: `2u (3 + 2X)`.
- **Các hệ số co giãn lại** `exp(old max - new max)`, một cho mỗi ô: đối số của chúng cộng lại
  nhiều nhất bằng `X` trên cả hàng, và mỗi cái có cùng cận ulp: `2u (3T + 2X)`.
- **Các tổng.** Mọi số hạng đều dương. `P V` cộng `B_c` tích mỗi ô, sai số của mỗi ô tương đối
  so với phần đóng góp của ô đó; `O` và `D` đang chạy được co giãn lại và cộng thêm một lần mỗi ô
  (`2T`); cây bướm (5 bước), phép chia và một chút biên dư tạo nên con số 8.

Các phép kiểm tra cho phép gấp đôi con số đó, vì cả tử số lẫn mẫu số đều mang nó:

- O: `|got - ref| <= 2 e * sum_c p_c |v_c| + 1e-7` cho mỗi cột, trong đó `p` là xác suất tham
  chiếu. Phần tuyệt đối chỉ có ý nghĩa với một đầu ra gần 0.
- L: `|got - ref| <= 2 e + 1e-5 + 2^-22 |ref|`. `log` biến sai số tương đối của `D` thành cùng
  sai số tuyệt đối đó; cận riêng của `log` không có trong tệp facts nào của chúng tôi (unverified:
  `1e-5` chừa chỗ cho nó); `m` và phép cộng cuối mỗi cái làm tròn một lần.

Đây là một ước lượng bậc nhất, không phải một chứng minh; hệ số 2 là biên dư của nó. Một lần co
giãn lại bị quên, một giá trị lớn nhất sai hay một rào chắn bị thiếu đều lệch hàng bậc độ lớn.
HIP công bố các sai số đo được thay vì cận (ch10-facts B4).

## Backend race

`emu` chạy từng luồng một, và một luồng chạy cho đến điểm đồng bộ hóa tiếp theo của nó. Mỗi lệnh
shuffle là một điểm hẹn của cả warp, nên khi bất kỳ lane nào rời cây bướm tính tổng, mọi lane đã
ghi P của nó: trên `emu`, rào chắn warp bị thiếu ở mục 1 phía trên không bao giờ lộ ra. `race`
chạy các luồng cùng lúc dưới ThreadSanitizer, và ở đó các lệnh shuffle, cũng như trên GPU, không
sắp thứ tự bộ nhớ: nó báo lần ghi P (kèm dòng mã) và lần đọc trong `compute_O` là một race
condition. Nó cũng báo một rào chắn khối bị thiếu giữa các ô, chỉ ra lần nạp ô và lần đọc va chạm
với nó. Một lượt chạy sạch không phải là bằng chứng. Lab 20.3 bắt bạn tìm cả hai.

## Trước khi bạn đo hiệu năng, hãy dự đoán

Benchmark đo thời gian flash attention so với chuỗi không gộp của lab 20.1 ở 8 head, d = 128,
n = 512, 1024, 2048 và 4096, batch 1:

- flash: kernel của bạn, một lần khởi chạy cho mọi head;
- không gộp: cho mỗi head, `S = Q K^T` với M3 (tham chiếu của lab 15.6), causal softmax tham chiếu
  của lab 20.1, `O = P V` với M2 (tham chiếu của lab 5.3).

Nó in số mili giây mỗi lần gọi (tốt nhất trong 5 lượt đo, mỗi lượt lặp lại lời gọi cho đến khi
kéo dài ít nhất 2 ms), GFLOP/s của attention nhân quả tính là `2 n^2 d` mỗi head (một nửa của
`4 n^2 d` của hai phép nhân đầy đủ, vì một nửa S bị che; đường không gộp vẫn tính nửa bị che), tỉ
lệ giữa chúng, và số byte mà đường không gộp cấp phát cho S và P. Cả hai đầu ra đều được kiểm tra
trước khi thời gian của chúng được báo cáo. Các kích thước không vừa một phần tư bộ nhớ thiết bị
được báo sẽ bị bỏ qua, và dòng đó nói rõ như vậy.

**Dự đoán `ch20/labs/p-lab2-ratio`.** Trước khi chạy: ở n = 4096, thời gian flash chia cho thời
gian không gộp trên thiết bị của bạn.

Hai lực kéo theo hai hướng ngược nhau; bài học không nói bên nào thắng trên thiết bị của bạn.

- Flash loại bỏ S và P khỏi bộ nhớ thiết bị. Với mỗi head, đường không gộp ghi S, đọc nó trong
  softmax, ghi P và đọc lại nó trong `P V`: ít nhất `2 n^2 x 4` byte được ghi và chừng ấy được
  đọc, 128 MiB mỗi head ở n = 4096.
- Hai phép nhân của flash là các vòng lặp warp đơn giản của sách: một lệnh shuffle cho mỗi phép
  nhân cộng trong phần điểm số, các FMA vô hướng từ bộ nhớ chia sẻ (shared memory) trong `P V`.
  Đường không gộp chạy chúng trên các GEMM M3 và M2 đã được tinh chỉnh của bạn. Chú thích 8 của
  sách (pp. 500-501) nói các phép nhân này có thể dùng những tối ưu nâng cao hơn của chương 15;
  các kernel thực tế dùng phần cứng ma trận (A6, D2).

Các con số là phép đo trên máy của bạn, ở các kích thước này, không phải mục tiêu.

## Tính khả chuyển

- HIP: `kernel.cu` được biên dịch qua `pmpp/vendors/hip.h`. Warp logic 32 lane, mặt nạ có kiểu
  và độ rộng tường minh cho phép hai warp logic dùng chung một wavefront rộng 64 mà không trộn lẫn
  (B2). Chưa được biên dịch hay chạy trên phần cứng AMD ở đây: unverified. Phần đệm `addr()` được
  tinh chỉnh cho 32 bank; AMD có 32 hoặc 64 (B3).
- Metal: xem phía trên. `B_c` mà thiết bị của bạn nhận được in ra ở đầu lượt chạy kiểm thử.
- `expf` và `logf` không cần khai báo riêng: phép kiểm tra front-end CUDA của khóa học khai báo
  các hàm toán độ chính xác đơn trong header stub của nó.
