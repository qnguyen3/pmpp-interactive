> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 4.4: kích thước khối trên một kernel chờ bộ nhớ

Sách: mục 4.6 (pp. 82-85) cho lý do một warp đang chờ không tốn gì khi có warp khác sẵn sàng, mục
4.7 (pp. 85-87) cho phép tính occupancy, các bài tập 6 đến 8 (pp. 91-92). Kernel này là của khóa
học này, không phải một đoạn mã in trong sách.

Hãy viết một phép gom (gather) theo dạng của một embedding bag:

    out[i] = sum over j < k of table[idx[i*k + j]]      for i < n

Các chỉ số là ngẫu nhiên và bảng thì lớn, nên gần như mọi bước của vòng lặp là một lần nạp từ bộ
nhớ toàn cục (global memory) mà không gì đoán trước được. Đây là loại kernel mà mục 4.6 nói tới:
nó dành thời gian để chờ, và thứ nó cần từ SM là những warp khác chạy trong lúc đó. Nó cũng là
dạng của phép tra embedding token ở tầng dưới cùng của một mô hình.

- Sửa `work/kernel.cu`:
  - `gatherSumKernel` và `launchGatherSum(table, idx, out, n, k, threadsPerBlock)`: mỗi đầu ra
    một luồng (thread) trên một lưới 1D, kích thước khối là một đối số lúc chạy.
  - `modelBlocksPerSm(limits, threadsPerBlock, regsPerThread)` và `modelOccupancy(...)`: phép
    tính của sách trên số chỗ khối, số chỗ luồng và số thanh ghi (register). Hãy tự tính tay các
    trường hợp của sách trước, rồi viết hàm cho ra cùng các câu trả lời. Một khối chiếm chỗ luồng
    theo nguyên warp, độ rộng lấy từ `limits.warpSize`, và một giới hạn bằng 0 nghĩa là thiết bị
    không báo cáo nó: bỏ qua quy tắc cần giới hạn đó. `sm_limits.h` chứa struct; nó được cung cấp
    và không được sửa.
- Sửa `work/kernel.metal` (`gather_sum`) và `work/launch.mm` (`encodeGatherSum`, với kích thước
  threadgroup là một đối số). Không có mô hình nào phải viết cho Metal: Apple không công bố giới
  hạn cho từng lõi, số thanh ghi hay hàm occupancy nào (`notes/today/ch04-facts.md`, C5).
- `provided/occupancy_api.cu` thuộc về chương trình điều khiển (driver). Trên CUDA và HIP, nó hỏi
  runtime của nhà sản xuất xem bao nhiêu khối của kernel bạn vừa trên một SM.
- Kiểm thử: `python3 labs/pmpp.py run 4.4`   Bắt đầu với `--backend emu`.
- Đo: `python3 labs/pmpp.py run 4.4 --bench` (chỉ trên thiết bị thật)

## Các bài kiểm thử làm gì

Kernel được so với một bản tham chiếu CPU, trong một dung sai, cho nhiều n, k và kích thước khối
dựng từ độ rộng mà thiết bị của bạn báo cáo: một lần độ rộng, một rưỡi, hai, ba, khối lớn nhất mà
thiết bị nhận, và 7 luồng. Phần lớn để lại một khối cuối không đầy. `out` được điền sẵn, nên một
đầu ra bạn không bao giờ ghi sẽ bị báo. Trên thiết bị thật, một giá trị được ghi quá cuối `out`
sẽ bị báo; trên trình giả lập (emulator), AddressSanitizer dừng ở lần truy cập đầu tiên ra ngoài
một buffer và nêu tên dòng đó.

Mô hình chạy trên những giới hạn mà không ai cần sở hữu (CUDA, HIP và trình giả lập):

- bảy kích thước khối được tính sẵn trong sách trên thiết bị ví dụ của sách: 100% cho 1024 xuống
  đến 64 luồng mỗi khối, 50% cho 32, 75% cho 768;
- thanh ghi trên cùng thiết bị đó: 31 và 33 thanh ghi mỗi luồng ở 512 luồng mỗi khối (4 khối,
  rồi 3), và 64 thanh ghi mỗi luồng (50% ở mọi kích thước khối);
- các bài tập 6, 7 và 8;
- một thiết bị có 1536 chỗ luồng và 16 chỗ khối, ở đó 768 cho 100% và 1024 cho 66.7%. Các GPU hiện
  nay khác nhau ở những giới hạn này (facts A12), nên câu trả lời của sách đúng trên ví dụ của sách
  chứ không đúng ở mọi nơi;
- các khối không phải nguyên số warp (48 luồng chiếm hai warp khi một warp là 32), và độ rộng 64;
- các giới hạn bằng 0, và các khối hoàn toàn không chạy được.

Thanh ghi là phần yếu nhất của mô hình. Sách nhân số luồng với số thanh ghi mỗi luồng. NVIDIA nói
thanh ghi được chia cho các warp và không công bố độ chi tiết của việc chia; AMD cấp phát chúng cho
từng wavefront theo từng gói và có một nhóm thanh ghi thứ hai mà không trường thiết bị nào hiển thị
(facts A13, B10). Với một kernel thật, hàm occupancy của nhà sản xuất là căn cứ, và đó là lý do
benchmark in nó cạnh mô hình của bạn.

## Trước khi bạn đo hiệu năng

Benchmark chạy cùng phép gom đó (một bảng có kích thước tính từ bộ nhớ của thiết bị bạn, từ 32 MB
đến 256 MB) ở 1, 2, 3, 4, 6, 8, 12, 16, 24 và 32 lần độ rộng của thiết bị, đến khối lớn nhất mà
thiết bị hoặc pipeline chấp nhận. Các kích thước được đo thời gian theo từng vòng. Với mỗi kích
thước, nó báo thời gian tốt nhất, thời gian trung vị và số lần tra mỗi giây.

Hãy viết dự đoán của bạn ra trước: kích thước khối nào sẽ chậm nhất trên thiết bị của bạn, và chậm
hơn kích thước nhanh nhất bao nhiêu lần?

Trên CUDA và HIP có thêm hai cột: số khối mỗi SM và occupancy mà mô hình của bạn cho với các giới
hạn thiết bị báo cáo, và các giá trị tương ứng từ hàm của nhà sản xuất cho kernel đã biên dịch của
bạn. Ở chỗ chúng không khớp, hãy tìm xem mô hình của bạn đã bỏ sót tài nguyên nào.

- Mô hình của bạn không được cho số thanh ghi trừ khi bạn cung cấp: không trường thiết bị nào chứa
  nó. Trình biên dịch in ra con số này (`nvcc --resource-usage`, `hipcc --resource-usage`). Truyền
  nó bằng `PMPP_REGS_PER_THREAD=<n> python3 labs/pmpp.py run 4.4 --bench`.
- Trên AMD, `maxBlocksPerMultiProcessor` được tính từ số chỗ luồng và không phải một giới hạn
  riêng, kích thước khối được làm tròn lên thành nguyên wavefront, và trường thanh ghi chỉ đếm
  thanh ghi vector (facts B9, B10). Benchmark nói điều này khi nó chạy trên HIP.
- `provided/occupancy_api.cu` chưa được biên dịch bởi nvcc hay một trình biên dịch HIP. Nếu nó
  không dựng được với bộ công cụ của bạn, hãy cấu hình một lần với
  `cmake -S labs -B labs/build -DPMPP_CH04_OCCUPANCY_API=OFF` và benchmark sẽ chạy mà không có các
  cột của nhà sản xuất.

Trên Metal, benchmark chỉ là một lượt quét thời gian. Không có con số occupancy nào để đặt cạnh nó,
nên điều bạn có thể nói là những gì bạn đã đo.

Hai dòng cuối của đầu ra cho tỉ lệ kích thước chậm nhất so với nhanh nhất, và để so sánh, lần chạy
trung vị của một kích thước cách lần chạy tốt nhất của chính nó bao xa. Một khoảng cách giữa hai
kích thước không lớn hơn con số thứ hai đó là do máy, không phải do kích thước khối. Các con số là
phép đo trên thiết bị của bạn; một lượt quét phẳng cũng là một kết quả.
