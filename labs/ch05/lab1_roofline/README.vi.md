> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 5.1: trần của bạn ở đâu?

Sách: mục 5.1 (pp. 93-97) và kernel của Fig. 3.11 (p. 61).

Hai kernel thăm dò được cung cấp sẵn. Một kernel (cộng vector) bị giới hạn bởi băng thông bộ nhớ,
kernel kia (một chuỗi phép nhân-cộng trong thanh ghi) bị giới hạn bởi tính toán. Phép đo của chúng
cho ra ngưỡng của thiết bị bạn tính bằng FLOP/B.

**Phần việc của bạn:** viết phép nhân ma trận ngây thơ (mỗi phần tử đầu ra một luồng) và lần khởi
chạy của nó, cho mọi độ rộng.

- Sửa `work/kernel.cu` (CUDA / HIP / trình giả lập) và `work/kernel.metal` + `work/launch.mm` (Metal).
- Kiểm thử: `python3 labs/pmpp.py run 5.1`
- Đo: `python3 labs/pmpp.py run 5.1 --bench`

Trước khi đo hiệu năng, hãy dự đoán: mô hình nói một kernel 0.25 FLOP/B có thể đạt tối đa một phần
tư con số băng thông của bạn tính bằng GFLOP/s. Kernel của bạn sẽ rơi xuống dưới mức đó, đúng mức
đó, hay vượt lên trên? Rồi giải thích những gì bạn thấy (mục 5.1 của bài học bàn về điều này).

Các con số là phép đo của hai kernel đơn giản trên máy của bạn, không phải đỉnh do nhà sản xuất công bố.
