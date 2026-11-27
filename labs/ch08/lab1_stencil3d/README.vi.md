> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 8.1: một stencil 3D, mỗi điểm một luồng, rồi đi theo trục z với thanh ghi

Sách: mục 8.2 đến 8.6 (pp. 187-198), Figs. 8.6 và 8.12, bài tập 1b (p. 198). Hai kernel so sánh
mà bạn được cho là Figs. 8.8 và 8.10.

Một lượt quét stencil 7 điểm trên một lưới N x N x N số float, thứ tự hàng (row-major) với x thay
đổi nhanh nhất. Mỗi điểm bên trong của `out` trở thành một tổng có trọng số của cùng điểm đó trong
`in` và sáu láng giềng của nó theo x, y và z. Các điểm biên giữ giá trị đầu vào: chương trình điều
khiển (driver) chép `in` vào `out` trước mỗi lần khởi chạy, và các kernel của bạn chỉ ghi các điểm
bên trong. Các trọng số và láng giềng mà mỗi trọng số nhân với nằm trong `stencil.h`.

Hãy viết hai kernel, bằng CUDA và bằng Metal:

1. **Cơ bản** (Fig. 8.6): `stencilBasicKernel` và `launchStencilBasic`. Mỗi điểm lưới một luồng
   (thread), mọi toán hạng đọc từ bộ nhớ toàn cục (global memory), khối 8 x 8 x 8 (bài tập 1b).
2. **Phân ô thanh ghi** (Fig. 8.12): `stencilRegTiledKernel` và `launchStencilRegTiled`.
   Một khối (block) gồm TILE_WIDTH x TILE_WIDTH luồng, một mặt phẳng x-y của ô đầu vào, đi qua
   một tấm gồm `depth` mặt phẳng đầu ra theo z, bắt đầu ở mặt phẳng `blockIdx.z * depth`.
   Mỗi luồng giữ các giá trị trước, hiện tại và kế tiếp của riêng nó trong thanh ghi (register);
   mặt phẳng hiện tại cũng được phản chiếu trong một mảng chia sẻ, vì đó là mặt phẳng duy nhất mà
   các luồng khác đọc. Lưới (grid) có `ceil(N / (TILE_WIDTH - 2))` khối theo x và y và
   `ceil(N / depth)` theo z.

- Sửa `work/kernel.cu` (cả hai kernel và cả hai hàm khởi chạy), `work/kernel.metal`
  (`stencil_basic`, `stencil_regtile`) và `work/launch.mm` (`encodeStencilBasic`,
  `encodeStencilRegTiled`). Đừng sửa `stencil.h`, `main.cpp` hay `../shared/`.
- `TILE_WIDTH` được định nghĩa bởi bản dựng: mặc định 32, và 16 và 8 cũng được dựng cho
  benchmark. Metal biên dịch 8, 16 và 32 lúc chạy; 32 chỉ được kiểm thử ở nơi
  `maxTotalThreadsPerThreadgroup` của pipeline ít nhất là 1024, nếu không chương trình điều khiển
  in ra giới hạn đó và nói rằng nó đã bỏ qua T = 32. Đừng viết cứng 32.
- Kiểm thử: `python3 labs/pmpp.py run 8.1`. Bắt đầu với `--backend emu`. Trình giả lập (emulator)
  cần vài phút cho lab này (ở N = 63, một lần khởi chạy có vài trăm nghìn luồng, chạy từng luồng
  một, dưới ba thứ tự). Nó kiểm thử các lưới nhỏ trước, nên hãy sửa những gì N = 3, 4 và 5 cho thấy
  trước khi chờ đến N = 63.
- Kiểm tra race: `python3 labs/pmpp.py run 8.1 --backend race` chạy các luồng của một khối đồng
  thời dưới ThreadSanitizer (N = 3, 4, 5 và 34, T = 32). Kernel của bạn chia sẻ `inCurr_s` và bảo
  vệ nó bằng hai rào chắn (barrier); nếu thiếu một cái hoặc đặt sai chỗ, công cụ này nêu tên hai
  dòng mã nguồn có các lần truy cập không được sắp thứ tự, ngay cả trong một lần chạy mà các con
  số tình cờ ra đúng. Một lần chạy sạch là bằng chứng, không phải chứng minh.
- Đo: `python3 labs/pmpp.py run 8.1 --bench` (chỉ thiết bị thật).

## Lab này khác trang sách ở đâu

- **c3 và c4.** Figs. 8.6 và 8.8 nhân láng giềng y-1 với c3 và láng giềng y+1 với c4.
  Figs. 8.10 và 8.12 in theo chiều ngược lại. Lab dùng cách ghép đầu tiên ở mọi nơi
  (`stencil.h`), và bảy trọng số của kiểm thử đều khác nhau, nên cách ghép in trong Fig. 8.12
  sẽ thất bại.
- **`inNext_s`.** Dòng cuối của Fig. 8.12 ghi `inNext_s` vào mặt phẳng chia sẻ. Kernel đó không có
  `inNext_s`; giá trị cần ghi là thanh ghi `inNext`.
- **Độ sâu tấm.** Tấm của sách sâu `OUT_TILE_DIM` mặt phẳng. Ở đây nó là đối số `depth`, để
  benchmark có thể so sánh các tấm; với `depth = TILE_WIDTH - 2`, kernel của bạn chính là kernel
  của sách.
- `in` là `const` và bảy trọng số đến dưới dạng một đối số truyền theo giá trị. Các listing để
  `c0` đến `c6` không được khai báo.

## Đừng trả về sớm

Mọi luồng của một khối phải đến được cả hai rào chắn ở mọi vòng lặp của vòng z. Một luồng ở rìa
khối, hoặc ở biên của lưới, không tính đầu ra nào nhưng vẫn nạp phần tử vùng biên (halo) của nó,
thứ mà các láng giềng của nó đọc. Hãy đặt điều kiện quanh các lần nạp và quanh lần ghi; đừng bao
giờ `return` trước các rào chắn của vòng lặp.

Trình giả lập tự nó không thể cho bạn thấy một lần trả về sớm. Hướng dẫn của NVIDIA nói một rào
chắn chờ mọi luồng *chưa thoát*, Apple nói điều tương tự cho Apple silicon, và trình giả lập mô
phỏng đúng như vậy: một luồng đã trả về thì không còn được tính. Điều trình giả lập bắt được là
hậu quả, chẳng hạn một giá trị vùng biên không ai nạp. AMD không nêu quy tắc nào cho trường hợp
này. Quy tắc khả chuyển là của sách: mọi luồng của khối thực thi mọi rào chắn
(`notes/today/ch04-facts.md` A6 và C3).

## Đọc một lỗi

- *Those points still hold the input: nothing was written there.* Kernel hoàn toàn không ghi các
  điểm bên trong đó (hoặc đã không chạy).
- *N boundary points changed.* Có thứ gì đó đã ghi ra ngoài phần bên trong, thường là do thiếu
  hoặc sai điều kiện bảo vệ phần bên trong của lưới.
- *The kernel wrote past the end of out.* Chương trình điều khiển đặt một vùng bảo vệ sau `out` và
  kiểm tra nó sau mỗi lần khởi chạy.
- Trên trình giả lập, một rào chắn bị thiếu cho giá trị sai trong hầu hết mọi trường hợp dưới cả
  ba thứ tự luồng (một thứ tự may mắn có thể che giấu nó trong một lưới rất nhỏ, nên hãy đọc cả
  ba), và AddressSanitizer dừng lần chạy ở lần đọc hoặc ghi ngoài biên đầu tiên và nêu dòng đó (ví
  dụ một lần nạp mặt phẳng `i + 1` khi `i + 1` đã vượt quá lưới). Các kiểm thử sau điểm dừng đó
  không chạy.

## Thí nghiệm

Hãy làm điều này khi mọi kiểm thử đã qua. Trước khi chạy `--bench`, hãy viết ra ba dự đoán:

1. Mô hình của sách: kernel cơ bản đòi 32 byte mỗi điểm đầu ra (bảy lần nạp 4 byte và một lần
   ghi), 0.41 FLOP/B (p. 189). Kernel phân ô thanh ghi ở độ sâu tấm của sách, T = 32 và 30 mặt
   phẳng, nạp 32 x 32 x 32 giá trị và ghi 30 x 30 x 30 mỗi khối: 1.47 FLOP/B. (Sách in 1.52 ở
   p. 195. 1.52 là con số bạn nhận được khi chỉ đếm vùng biên x-y, giới hạn khi tấm sâu lên vô
   hạn; sách không nói nó đã đếm thế nào. Phép tính đó là của khóa học này,
   `notes/today/ch08-facts.md` E2. Các tấm sâu hơn bên dưới kiểm tra giới hạn đó.)
   Như vậy là **ít hơn khoảng 3.6 lần số byte**. Bạn kỳ vọng nó nhanh hơn bao nhiêu lần trên GPU
   của bạn?
2. Kernel ba mặt phẳng chia sẻ được cung cấp (Fig. 8.10) di chuyển đúng cùng lượng dữ liệu toàn
   cục như kernel của bạn (p. 198); kernel của bạn đổi hai mặt phẳng chia sẻ lấy thanh ghi. Bạn kỳ
   vọng kernel của bạn nhanh hơn nó bao nhiêu lần?
3. Độ sâu tấm nào sẽ nhanh nhất: T - 2 của sách, 8, 16, 64, hay một tấm cho cả cột?

Bộ nhớ đệm (cache) vốn đã phục vụ nhiều lần đọc láng giềng của kernel cơ bản, như bạn đã thấy ở
chương 5: hệ số của một mô hình nói kernel *đòi* bộ nhớ ít hơn bao nhiêu, và mức tăng tốc bạn đo
được sẽ nhỏ hơn ở bất cứ đâu bộ nhớ đệm đã làm sẵn việc đó. Con số 3.6 là một phát biểu về số byte
được yêu cầu, không phải một lời hứa.

`--bench` làm gì: nó tăng N cho đến khi kernel cơ bản chạy đủ lâu để đo, mà không để `in` và `out`
cộng lại vượt quá 2 GiB hoặc một phần tám bộ nhớ mà thiết bị của bạn báo. Rồi nó đo mọi kernel
trên lưới đó (lần tốt nhất trong ít nhất 5 lần chạy, lặp lại cho đến khi các lần chạy cộng lại
khoảng một phần tư giây, vì những lượt quét đơn lẻ ngắn như thế này dao động nhiều) và sau đó kiểm
tra từng kết quả với CPU; một kernel có đầu ra sai sẽ bị báo, không được đo. Mỗi dòng hiển thị số
mili giây, số điểm bên trong mỗi giây, và "GB/s at 8 B/point": một lần nạp 4 byte và một lần ghi
4 byte mỗi điểm bên trong, lưu lượng của mức lý tưởng 13/8 FLOP/B, không phải lượng mà kernel thực
sự đòi từ bộ nhớ. Mức tăng tốc được tính so với kernel cơ bản và, với kernel phân ô thanh ghi của
bạn, so với kernel mặt phẳng cùng độ rộng ô.

Các độ sâu tấm là T - 2 (của sách), 8, 16, 64 và N, được đánh dấu "1 slab". Lưới có
`ceil(N / depth)` tấm và tấm 0 bắt đầu ở mặt phẳng 0, một mặt phẳng biên, nên độ sâu N là độ sâu
duy nhất đặt cả cột vào một tấm; N - 1 hoặc N - 2 sẽ khởi chạy một tấm thứ hai với tối đa một mặt
phẳng để tính. Các dòng "limits" ghi lại số luồng mỗi khối mà thiết bị của bạn cho phép đối với
kernel khối lập phương và kernel phân ô thanh ghi của bạn ở mỗi độ rộng ô
(`maxTotalThreadsPerThreadgroup` của pipeline Metal, thuộc về từng kernel đã biên dịch, hoặc
`maxThreadsPerBlock` trên CUDA và HIP).

Trên CUDA, một khối 32 x 32 để lại tối đa 64 thanh ghi mỗi luồng; vượt quá mức đó, lần khởi chạy
thất bại thay vì chạy chậm (`notes/today/ch08-facts.md` A4). Nếu một lần khởi chạy T = 32 có lúc
báo "too many resources requested", đó là lý do; `nvcc
--resource-usage` cho thấy số thanh ghi của kernel.
