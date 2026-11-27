> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 9.1: săn race, hai kernel histogram

Sách: các mục 9.2 đến 9.4 (pp. 203-213), Figs. 9.6 và 9.9. Các lỗi là của khóa học này, không phải
của sách.

`work/kernel.cu` và `work/kernel.metal` chứa hai histogram 256 bin của một ảnh 8 bit,
do một đồng nghiệp viết:

- `histogramBasic` (Metal: `histogram_basic`), Fig. 9.6: mỗi điểm ảnh một luồng (thread), mỗi luồng cộng
  một vào bin của điểm ảnh của nó.
- `histogramPool` (Metal: `histogram_pool`), Fig. 9.9: mỗi khối (block) một histogram riêng trong một
  vùng chung (pool) ở bộ nhớ toàn cục (global memory), được cập nhật bằng phép toán nguyên tử (atomic)
  phạm vi khối, rồi ghi nhận các bin riêng khác 0 vào các bin công khai.

Giữa hai kernel có **ba lỗi độc lập**. Mọi thứ đều biên dịch được. Hãy tìm và sửa
cả ba.

- Sửa `work/kernel.cu` (hai kernel; `launchHistogramBasic` và `launchHistogramPool`
  bên dưới chúng là đúng) và `work/kernel.metal`. Cùng ba lỗi đó có ở cả hai.
  `work/launch.mm` là đúng và có mặt cho đầy đủ.
- Bên gọi đặt `bins` và pool về 0. Pool chứa ceil(width x height /
  threadsPerBlock) x 256 bộ đếm: mỗi khối một histogram riêng.
- Kiểm thử: `python3 labs/pmpp.py run 9.1 --backend emu`, rồi `--backend race`, rồi
  `python3 labs/pmpp.py run 9.1` cho GPU của bạn.
- Đo (chỉ GPU): `python3 labs/pmpp.py run 9.1 --bench`.

## Các phép kiểm thử làm gì

Mỗi trường hợp so sánh cả 256 số đếm với vòng lặp tuần tự của Fig. 9.2, nằm trong
`main.cpp`. Số đếm là số nguyên, nên phép so sánh là chính xác.

- Các ảnh, rộng x cao: 1 x 1, 63 x 1, 64 x 1, 100 x 10, 64 x 64; trên GPU còn có
  1024 x 1024 và 3000 x 1001.
- Bốn loại ảnh, từ các seed cố định: byte ngẫu nhiên phân bố đều, một ảnh hằng (mọi
  điểm ảnh là 200), 0 và 255 thành các dải dài, một dải chuyển màu mượt.
- Khối 64 và 128 luồng, ít hơn 256 bin, nên vòng lặp ghi nhận phải đi theo bước nhảy (stride); trên
  GPU còn có 256 và khối lớn nhất mà kernel có thể chạy.
- Các trường hợp của kernel cơ bản chạy trước, rồi đến của pool, nên mỗi lỗi hiện ra trong các trường hợp
  của chính kernel chứa nó. Trường hợp đầu tiên của mỗi kernel là ảnh hằng trong 64 khối 64
  luồng: mọi luồng đều muốn cùng một bin, đầu vào có tranh chấp nặng nhất có thể.
- Trên GPU, mỗi cấu hình được khởi chạy 20 lần trên các buffer vừa được đặt về 0, và một lần thất bại
  cho biết trong bao nhiêu lần khởi chạy các số đếm bị sai. Trên trình giả lập (emulator), nó được khởi chạy một lần.

## Dự đoán của bạn

Trước khi chạy bất cứ thứ gì, hãy đọc `work/kernel.cu` và viết ra các lỗi bạn có thể tìm thấy bằng
cách đọc. Với mỗi lỗi, hãy viết ra backend nào bạn kỳ vọng sẽ cho thấy nó, và vì sao. Rồi chạy
`emu`, `race` và GPU của bạn, và điền vào bảng này trong khi làm:

| Lỗi | `emu` | `race` | GPU của bạn |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |

## Phương pháp

1. Trình giả lập trước: `python3 labs/pmpp.py run 9.1 --backend emu`. Nó chạy một luồng cho đến khi
   luồng đó tới một rào chắn (barrier) hoặc tới cuối, rồi đến luồng kế tiếp, theo ba thứ tự cố định. Một
   rào chắn bị thiếu luôn cho đáp án sai ở đó. Một phép toán nguyên tử bị thiếu thì không bao giờ: không
   luồng nào bị ngắt giữa chừng `++bins[b]`, nên trình giả lập không thể làm mất một
   lần cập nhật. Hãy kỳ vọng thấy một trong ba lỗi ở đây, và hai lỗi còn ẩn.
2. `python3 labs/pmpp.py run 9.1 --backend race` chạy các luồng giả lập thực sự
   đồng thời dưới ThreadSanitizer, mỗi lần vài khối, mỗi khối tới 128 luồng. Hai
   luồng chạm vào cùng vùng nhớ mà không có đồng bộ hóa sẽ được báo cáo cùng cả hai
   lần truy cập, dù số đếm có ra sai hay không. Hãy đọc số dòng `kernel.cu:`.
   Một lần truy cập trong một cặp có thể nằm bên trong phép toán nguyên tử của chính trình giả lập
   (`pmpp_emu::atomic_rmw ... fetch_add`): đó là một lời gọi `atomic_ref` của bạn, tranh chấp
   với một lần truy cập thường ở dòng kia. ThreadSanitizer báo mỗi cặp dòng một lần
   mỗi lần chạy, nên hãy đọc trường hợp đầu tiên của mỗi kernel; một trường hợp sau có thể vượt qua dù race
   vẫn còn đó.
3. Sửa một chỗ, chạy lại. Những chỗ thất bại sẽ thay đổi dần theo từng bước.
4. Rồi đến GPU của bạn: `python3 labs/pmpp.py run 9.1`. Ghi lại nó đã cho bạn thấy những lỗi nào, và
   đã giấu những lỗi nào.

Nếu một chỗ sửa cần phép toán nguyên tử, hãy viết nó như sách làm: một `cuda::atomic_ref<unsigned int,
scope>` trên bộ đếm và `fetch_add(..., cuda::memory_order_relaxed)`. Chọn phạm vi
theo các luồng dùng chung bộ đếm: `cuda::thread_scope_block` nếu tất cả nằm trong một
khối, `cuda::thread_scope_device` nếu chúng ở các khối khác nhau. Bản sao riêng giữ
phạm vi khối. Trên Metal, các hàm atomic chỉ nhận kiểu atomic, nên chỗ sửa đổi
tham số thành `device atomic_uint*` và dùng `atomic_fetch_add_explicit`
(`notes/today/ch09-facts.md` C1); host vẫn cấp phát các bộ đếm 32 bit thường.

Một rào chắn Metal nêu tên vùng nhớ có các thao tác mà nó sắp thứ tự:
`threadgroup_barrier(mem_flags::...)`. Kernel pool cần cờ nào, và vì sao? Các
cờ và điều mỗi cờ đảm bảo nằm trong `notes/today/ch05-facts.md` C2.

## Đọc bảng của bạn

Dù GPU của bạn cho thấy gì, hãy đọc nó đối chiếu với những gì tài liệu của các nhà cung cấp nêu, vì đó là điều
thiết bị tiếp theo được phép làm (`notes/today/ch09-facts.md`):

- Các phép toán nguyên tử trên một vị trí được tuần tự hóa theo thứ tự không cố định; một lần ghi thường từ nhiều
  luồng của một warp vào một vị trí khiến "which thread performs the final write"
  không xác định (NVIDIA, A3). AMD: các phép toán nguyên tử cùng vị trí "are serialized in an undefined
  order" (B1). Về hình thức, một `++bins[b]` thường từ hai luồng là một data race và là hành vi
  không xác định, nên một số đếm bị mất là kết quả điển hình, không phải kết quả được hứa hẹn (A2, A3).
- Metal: "Objects of atomic types are free from data races" (C1). Đặc tả không nói gì
  về các lần truy cập thường.
- Không công cụ GPU nào có thể chỉ ra các dòng của bạn. `compute-sanitizer --tool racecheck` của NVIDIA
  chỉ kiểm tra bộ nhớ chia sẻ (shared memory), nên nó không thể thấy race trên các bin toàn cục (A10); tài liệu của AMD không nêu
  công cụ phát hiện race nào (B6); Shader Validation của Metal không liệt kê phép kiểm tra race nào (C5). Backend `race`
  là ThreadSanitizer trên các luồng CPU (E1): một lần chạy sạch ở đó cho thấy các lịch thực thi đã
  xảy ra, không phải tất cả.

Một cấu hình vượt qua 20 lần trên GPU của bạn đã cho bạn thấy 20 lịch thực thi trên một thiết bị. Nó
không cho bạn thấy rằng kernel là đúng.

## Trước khi đo hiệu năng

`--bench` đo thời gian cả hai kernel trên 2^26 điểm ảnh (ít hơn nếu thiết bị báo quá ít bộ nhớ
cho ảnh và pool), 256 luồng mỗi khối, trên byte phân bố đều và trên ảnh
hằng, lấy lần tốt nhất trong 5 lần chạy. Hãy viết ra trước:

1. Kernel cơ bản mất thời gian gấp bao nhiêu lần trên ảnh hằng so với trên byte phân bố đều
   (mục 9.3 hỏi cùng câu hỏi và tiết lộ đáp án).
2. Kernel pool nhanh hơn kernel cơ bản bao nhiêu lần trên ảnh hằng
   (trang labs ghi lại dự đoán này).

Mô hình của sách (mục 9.3, pp. 209-210) cho phép tới 256x cho câu đầu tiên: các phép toán nguyên tử trên
một vị trí không thể chồng lấn, và byte phân bố đều rải chúng ra 256 vị trí. Hai điều
có thể khiến con số thực nhỏ hơn nhiều. GPU thực hiện các phép toán nguyên tử phạm vi device trong bộ nhớ đệm (cache) L2,
không phải trong DRAM (NVIDIA, A4; AMD, B2; tài liệu của Apple không nói gì, C4), nên mỗi phép ngắn hơn. Và
trình biên dịch có thể gộp các phép toán nguyên tử cùng địa chỉ của một warp thành một: NVIDIA chỉ nói điều này trong
một bài blog cập nhật năm 2017, "in many cases", không kèm điều kiện (A9); trình biên dịch của AMD chỉ làm vậy
khi nó chứng minh được cả wavefront dùng một địa chỉ, điều mà một bin đọc từ
ảnh không thỏa (B5).

Các con số là phép đo trên thiết bị của bạn. Việc đặt các bin và pool về 0 trước mỗi
lần chạy không được tính thời gian.

## Thêm một điều về phạm vi

Sửa phần ghi nhận bằng `cuda::thread_scope_block` thay vì phạm vi device và mọi phép kiểm thử
ở đây vẫn vượt qua: các trình giả lập bỏ qua phạm vi (`labs/README.md`), và Metal không có phạm vi. Nó
vẫn sai. Theo mô hình bộ nhớ của CUDA, hai khối cập nhật một bin công khai là một data race,
vì mỗi lần cập nhật "is not atomic at a scope that includes the thread that performed the
other operation", và bất kỳ race nào như vậy "results in undefined behavior" (A2). Số đúng trên
một backend không chứng minh được gì về backend tiếp theo.
