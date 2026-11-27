> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 6.1: hợp nhất truy cập bộ nhớ trên thiết bị của bạn

Sách: mục 6.1 (pp. 124-132), Figs. 6.2 và 6.3, và nhận xét về các lần ghi (p. 130).

Hãy viết ba kernel nhỏ chỉ khác nhau ở địa chỉ mà mỗi luồng (thread) chạm vào, rồi đo xem mỗi mẫu
tốn bao nhiêu ở đây. Không kernel nào là đoạn mã in trong sách: chúng là những kernel thăm dò quy
tắc mà mục 6.1 phát biểu.

1. `stridedReadKernel`: luồng `i` chép `in[i * stride]` sang `out[i]`, với `i < n`. Lần ghi liên
   tiếp ở mọi bước nhảy (stride); lần đọc chỉ liên tiếp ở bước nhảy 1.
2. `transposeReadRowsKernel`: `in` có kích thước rows x cols, `out` là cols x rows. Luồng
   (x = cột, y = hàng) đọc `in[row * cols + col]` và ghi `out[col * rows + row]`.
3. `transposeWriteRowsKernel`: cách gán ngược lại, nên lần ghi vào `out` là phía liên tiếp và lần
   đọc từ `in` là phía có bước nhảy.

Mỗi kernel có hàm khởi chạy bên cạnh (`launchStridedRead`, `launchTransposeReadRows`,
`launchTransposeWriteRows`); các phép chuyển vị dùng khối 16 x 16 và một lưới (grid) được làm tròn
lên để phủ ma trận. Một kernel sao chép được cung cấp sẵn (`provided/copy.*`): nó đọc và ghi liên
tiếp, và băng thông của nó là con số để so sánh các kernel khác.

- Sửa `work/kernel.cu` và `work/kernel.metal` + `work/launch.mm`. Trên Metal, hãy đánh chỉ số bằng
  `int`, không phải `uint`: đó là lời khuyên của Apple cho việc đánh chỉ số mảng trong MSL.
- Kiểm thử: `python3 labs/pmpp.py run 6.1`   Bắt đầu với `--backend emu`. Đầu vào chứa đúng
  `n * stride` số float, nên AddressSanitizer nêu tên dòng của mọi lần đọc vượt quá nó. Các đầu ra
  được điền NaN trước mỗi lần chạy, nên một phần tử bạn không bao giờ ghi sẽ thất bại.
- Đo: `python3 labs/pmpp.py run 6.1 --bench` (chỉ trên thiết bị thật). Nó đo thời gian kernel sao
  chép, phép đọc có bước nhảy ở các bước nhảy 1 đến 64 trên một đầu vào 2^25 số float, cả hai
  phép chuyển vị trên một ma trận 4096 x 4096 (nhỏ hơn nếu thiết bị của bạn báo quá ít bộ nhớ),
  và in độ rộng SIMD mà thiết bị của bạn báo cáo.

Trước khi đo hiệu năng, hãy viết ra hai dự đoán.

1. **Bước nhảy 2 tính theo phần trăm của bước nhảy 1.** Hướng dẫn của NVIDIA đếm lưu lượng bộ nhớ
   theo các sector 32 byte và nói một lần nạp bước nhảy 2 chỉ dùng một nửa mỗi sector mà nó nạp.
   Con số mà lab này in ra đếm các byte kernel của bạn dùng, cả đọc lẫn ghi, và lần ghi liên tiếp
   ở mọi bước nhảy. Nếu thời gian tỉ lệ với số byte được chuyển, mô hình sector sẽ cho con số đó
   bao nhiêu phần trăm? Rồi quyết định xem bộ nhớ đệm (cache) của thiết bị bạn có thể làm thay đổi
   nó không. Trên AMD, đơn vị là một dòng cache 64 hoặc 128 byte. Apple không công bố quy tắc hợp
   nhất truy cập, kích thước dòng hay kích thước sector nào, nên trên máy Mac phép đo này là câu
   trả lời duy nhất.
2. **Phép chuyển vị ngây thơ nào nhanh hơn**, đọc theo hàng (ghi có bước nhảy) hay ghi theo hàng
   (đọc có bước nhảy), và nhanh hơn bao nhiêu? Sách nói các lần ghi không được hợp nhất có thể gây
   hại nhiều hơn các lần nạp không được hợp nhất (p. 130); NVIDIA gắn chi phí thêm của các lần ghi
   rải rác với ECC trên bộ nhớ GDDR. Một khối 16 x 16 đọc và ghi cùng 256 phần tử theo cả hai cách:
   có gì giữ các dòng của phía có bước nhảy lại trên chip trong khi các luồng khác của khối dùng
   chúng không?

Các con số bạn nhận được là phép đo trên máy của bạn, không phải giá trị kỳ vọng.
