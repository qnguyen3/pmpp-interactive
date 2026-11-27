> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 5.2: kernel phân ô

Sách: mục 5.4 (pp. 108-112), Fig. 5.9.

Hãy viết phép nhân ma trận phân ô (tiling): hai ô (tile) trong bộ nhớ chia sẻ (shared memory), mỗi
luồng (thread) mỗi giai đoạn nạp một phần tử của M và một của N, hai rào chắn (barrier). Ma trận
vuông có độ rộng là bội của `TILE_WIDTH`; lab 5.3 bỏ cả hai giới hạn này.

- Sửa `work/kernel.cu` và `work/kernel.metal` + `work/launch.mm`.
- `TILE_WIDTH` được định nghĩa bởi bản dựng (mặc định 16; 8 và 32 cũng được dựng cho benchmark;
  Metal nhận từ 4 đến 32 lúc chạy). Đừng viết cứng 16.
- Kiểm thử: `python3 labs/pmpp.py run 5.2`   Bắt đầu với `--backend emu`.
- Đo: `python3 labs/pmpp.py run 5.2 --bench` (so sánh với một kernel ngây thơ được cung cấp)

Trước khi đo hiệu năng, hãy viết ra mức tăng tốc bạn dự đoán ở T = 16 và độ rộng ô nào bạn dự đoán
sẽ nhanh nhất. Mô hình của bài học nói lưu lượng giảm T lần; thiết bị của bạn quyết định bao nhiêu
phần trong đó biến thành thời gian.
