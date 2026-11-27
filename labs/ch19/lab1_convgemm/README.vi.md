> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 19.1: một lớp tích chập theo hai cách

Sách: mục 19.2 (pp. 460-463, Figs. 19.5 và 19.7) và mục 19.3 (pp. 463-472,
Eq. 19.4, Eq. 19.5, Fig. 19.11).

Hãy tính một lớp tích chập (convolution), lớp của sách (stride 1, không đệm), theo hai cách, rồi đo
chúng với nhau và với implicit GEMM của Fig. 19.11.

- X là N x C x H x W, F là M x C x K x K, Y là N x M x H_out x W_out (NCHW), tất cả đều là `float`
  theo thứ tự hàng (row-major). H_out = H - K + 1, W_out = W - K + 1, và P = H_out * W_out là số
  điểm ảnh trong một bản đồ đầu ra.

## Phần của bạn

**(1) Trực tiếp.** `convDirectKernel` và `launchConvDirect(X, F, Y, N, C, H, W, M, K)`:
Fig. 19.7, đã sửa. Mỗi điểm ảnh đầu ra một luồng (thread), khối gồm `TILE_WIDTH x TILE_WIDTH`
luồng, lưới (grid) `(M, T, N)`, với chỉ số ô được tuyến tính hóa theo từng hàng như trong
Fig. 19.5 nhưng dùng phép chia làm tròn lên. Mọi đầu ra đều được kiểm tra biên, và bạn tự tuyến
tính hóa `X[n, c, h + p, w + q]` của trang sách.

**(2) Trải + GEMM.** `unrollKernel` và `launchUnroll(X, n, Bt, C, H, W, K)` ghi đầu vào đã trải của
mẫu n ở dạng chuyển vị, thành Bt = B^T: P hàng, mỗi hàng ứng với một điểm ảnh đầu ra, mỗi hàng chứa
C*K*K giá trị của mảng vùng của điểm ảnh đó theo thứ tự (c, p, q) của sách. Rồi
`launchConvGemm(X, F, Y, Bt, N, C, H, W, M, K)`: với mỗi mẫu, trải vào buffer tạm `Bt` và gọi một
lần GEMM M2 của dự án, của lab 6.4

```
launchLinear(X, W, Y, batch, in, out)   // Y[batch x out] = X[batch x in] * W^T, W stored (out, in)
```

Hãy chọn sáu đối số của nó sao cho Y của mẫu ra dạng M x P (NCHW) và F không bao giờ bị chép. Một
lời gọi đó là điểm chính của lab: chữ ký GEMM của dự án có thể ghép nối được.

**Metal.** `conv_direct` và `unroll_input` nằm trong `kernel.metal`; `encodeConvDirect`,
`encodeUnroll` và `encodeConvGemm` trong `launch.mm`. `encodeConvGemm` lặp qua các mẫu trong một
compute encoder và gọi `encodeLinearAt` được cung cấp, hàm này bind kernel `linear` của lab 6.4 với
một độ lệch byte vào mỗi buffer. Một compute encoder thông thường chạy các dispatch của nó theo thứ
tự và các buffer được theo dõi hazard, nên một buffer tạm phục vụ được mọi mẫu
(`notes/today/ch02-facts.md` C9).

Fig. 19.11 đã sửa được cung cấp và đo bên cạnh hai đường đi của bạn; bạn không viết nó.

## Mã khác trang sách ở đâu

- Fig. 19.5 chia `W_out / TILE_WIDTH`, điều này để lại 12 hàng và 12 cột của đầu ra 28 x 28 của
  LeNet C1 không được tính. Hãy làm tròn lên, và để các luồng không có điểm ảnh nào thì không làm
  gì.
- Fig. 19.7 không có kiểm tra biên và chữ ký của nó thiếu H, W và M, những thứ mà các chỉ số tuyến
  tính hóa cần. Trang sách cũng đặt tên các biến `dim3` là `blockDim` và `gridDim`; lab dùng
  `dimBlock` và `dimGrid`, như lab 6.4, để chúng không che các biến có sẵn cùng tên.
- Sách không cho kernel trải nào. Nó mô tả việc trải tường minh và chi phí của nó (p. 467) rồi
  chuyển sang phiên bản ngầm định. Kernel ở đây là của khóa học này.
- Fig. 19.11 (được cung cấp, `provided/implicit_gemm.*`): số giai đoạn được làm tròn lên (phép
  `(C*K*K) / TILE_WIDTH` của trang chạy một giai đoạn cho C*K*K = 25 và bỏ mất 9 trong 25 số hạng),
  việc nạp F, việc gom X và việc ghi được kiểm tra biên với điền số không (sách để điều này làm bài
  tập, p. 470), và mã host mà sách không cho thấy khởi chạy `(ceil(P / 16), ceil(M / 16), N)` khối.

## Các tệp

- `work/kernel.cu` (CUDA, HIP, các trình giả lập): `convDirectKernel`, `launchConvDirect`,
  `unrollKernel`, `launchUnroll`, `launchConvGemm`.
- `work/kernel.metal`: `conv_direct`, `unroll_input`.
- `work/launch.mm`: `encodeConvDirect`, `encodeUnroll`, `encodeConvGemm`. Hàm cuối cùng của nó,
  `convGemmScratchProblem`, được cung cấp: trước khi chương trình điều khiển (driver) cấp phát buffer tạm, nó kiểm tra B^T của một mẫu với `maxBufferLength` của thiết bị (Metal giới hạn một buffer
  ở đó, ít nhất 256 MB theo tài liệu của Apple, `notes/today/ch19-facts.md` C2) và thất bại kèm một
  thông báo thay vì cấp phát.
- Được cung cấp, không sửa, trong `provided/`:
  - `gemm.cmake` chọn kernel nào của lab 6.4 mà đường đi GEMM của bạn gọi (bên dưới).
  - `linear_at.mm`: `encodeLinearAt`, `encodeLinear` của lab 6.4 kèm độ lệch byte.
  - `implicit_gemm.cu`, `implicit_gemm.metal`, `implicit_gemm_launch.mm`: Fig. 19.11 đã sửa, dưới
    dạng `launchConvImplicitGemm` và `encodeConvImplicitGemm`.
- `main.cpp`: chương trình điều khiển. `cpuConvLayer`, tham chiếu CPU, là Fig. 19.4 với các chỉ số
  được tuyến tính hóa và tổng được giữ ở double.

**GEMM nào.** Mặc định, đường đi gọi lời giải tham chiếu của lab 6.4. Để gọi kernel lab 6.4 của
chính bạn, đặt `PMPP_CH19_GEMM=work`; `PMPP_CH19_GEMM=m2` hoặc không đặt thì quay về mặc định. Lựa
chọn được đọc khi các lab được cấu hình, việc mà `pmpp.py` làm trước mỗi lần chạy. Kiểm thử và
benchmark in ra cái nào đã chạy ("m2 reference" hoặc "m2 yours").

## Lệnh

- Kiểm thử: `python3 labs/pmpp.py run 19.1 --backend emu` trước. Nó chạy dưới AddressSanitizer, nên
  một lần đọc hoặc ghi vượt quá X, F, Bt hoặc Y sẽ dừng kèm dòng mã nguồn. Rồi `--backend race`,
  rồi `python3 labs/pmpp.py run 19.1` trên GPU của bạn.
- Đo: `python3 labs/pmpp.py run 19.1 --bench`.
- Với M2 của chính bạn: `PMPP_CH19_GEMM=work python3 labs/pmpp.py run 19.1 --bench`.

## Kiểm thử làm gì

Các hình dạng (N, C, H, W, M, K), trên mọi backend:

- Ví dụ của sách, (1, 3, 3, 3, 2, 2), với các số nguyên của Fig. 19.2. Y phải chính xác là
  `14 20 15 24 / 12 24 17 26`: số nguyên nhỏ là chính xác trong float, nên không có dung sai.
- Cùng ví dụ đó hai lần, (2, 3, 3, 3, 2, 2).
- LeNet C1, (2, 1, 32, 32, 6, 5): C*K*K = 25, đầu ra 28 x 28 và M = 6, không cái nào là bội của 16.
  Đây là những hình dạng mà Figs. 19.5 và 19.11 như bản in làm sai.
- LeNet C3, (2, 6, 14, 14, 16, 5): C*K*K = 150, đầu ra 10 x 10.
- Pointwise, (2, 8, 7, 9, 5, 1). Một điểm ảnh đầu ra, (1, 4, 5, 5, 3, 5), nên P = 1.
  Không vuông, (1, 2, 9, 17, 3, 3). C*K*K là bội của 16, (1, 16, 6, 6, 16, 2).
- Trên GPU còn có (4, 64, 56, 56, 64, 3), (8, 256, 14, 14, 256, 3) và (2, 3, 224, 224, 64, 3).
- `race` chạy ví dụ của sách, (1, 2, 9, 17, 3, 3) và (1, 6, 14, 14, 16, 5): GEMM và kernel ngầm
  định chia sẻ các ô (tile) qua các rào chắn (barrier).

Bốn phép kiểm tra cho mỗi hình dạng:

- **(a) Riêng phép trải.** Mọi phần tử của B^T của mọi mẫu được so sánh chính xác với một phép trải
  trên CPU, nên một lỗi chỉ số được báo là lỗi trải, kèm hàng, cột và phần tử X lẽ ra phải ở đó, chứ
  không phải là một GEMM sai.
- **(b) Đường đi trực tiếp** và **(c) đường đi trải + GEMM**, so với `cpuConvLayer` với một dung sai
  co giãn theo tổng |x f| của mỗi đầu ra. Nếu (a) qua mà (c) thất bại, thông báo chỉ vào các đối số
  của lời gọi GEMM của bạn.
- **(d) Implicit GEMM được cung cấp.** Một lỗi ở đó là lỗi của khóa học, và thông báo nói như vậy.

X và F chứa chính xác các số float của chúng, và trên các trình giả lập Bt chứa đúng P x C*K*K.
Y bắt đầu là NaN và theo sau là một dải bảo vệ gồm các số float canh phải còn nguyên; trên GPU, Bt
cũng có một dải như vậy. Trên GPU, mỗi phép kiểm tra chạy hai lần, với Y và Bt được đặt lại NaN ở
giữa.

Starter chưa đụng tới dựng được và thất bại: các kernel rỗng của nó để Bt và Y là NaN.

## Trước khi đo hiệu năng, hãy dự đoán

Benchmark chỉ chạy trên thiết bị thật (`emu` nói nó bị bỏ qua). Năm hình dạng lớp
(N, C, H, W, M, K):

| lớp | hình dạng | GEMM mỗi mẫu, M x C*K*K x P | độ phình, Eq. 19.4 |
|---|---|---|---|
| early | (8, 3, 224, 224, 64, 3) | 64 x 27 x 49284 | 8.84 |
| middle | (8, 64, 56, 56, 64, 3) | 64 x 576 x 2916 | 8.37 |
| late | (8, 256, 14, 14, 256, 3) | 256 x 2304 x 144 | 6.61 |
| pointwise | (8, 256, 14, 14, 256, 1) | 256 x 256 x 196 | 1 |
| small | (64, 6, 14, 14, 16, 5), LeNet C3 ở batch 64 | 16 x 150 x 100 | 12.76 |

Các tỉ lệ phình được tính từ Eq. 19.4, K^2 H_out W_out / (H W), không phải đo. Với mỗi lớp, ba
đường đi lần lượt chạy và lần tốt nhất trong vài vòng được tính: kernel trực tiếp của bạn; trải +
GEMM của bạn, được đo như cả lớp, rồi riêng N lần trải của nó, rồi riêng N lần GEMM của nó trên một
buffer tạm đã điền sẵn; và implicit GEMM được cung cấp.

Hãy viết ra ba con số trước:

1. Trên lớp late, trải + GEMM nhanh hơn hay chậm hơn trực tiếp bao nhiêu lần.
2. Cùng tỉ lệ đó trên lớp early.
3. Trên lớp early, bao nhiêu phần thời gian của trải + GEMM dành cho chính phép trải.

Những điều cần cân nhắc:

- Kernel trực tiếp yêu cầu 0.25 FLOP/B: mỗi đầu ra, C*K*K phép nhân-cộng và hai lần nạp cho mỗi
  phép (ghi chú về mục 19.2). Nhưng các luồng cạnh nhau đọc các mảng vùng chồng lấn, và bộ nhớ đệm
  (cache) có thể phục vụ phần lớn việc đó. Hãy so sánh kết quả ngây thơ của lab 5.1 của bạn với trần
  0.25 FLOP/B của nó trên roofline ở mục 5.1 của bài học.
- Phép trải ghi B^T và GEMM đọc lại nó. B^T bằng tỉ lệ phình nhân với kích thước X của một mẫu.
- Số ô với ô 16 x 16. Late: 16 hàng ô, 9 cột ô, 144 giai đoạn. Early: 4 hàng ô và 3081 cột ô,
  nhưng chỉ 2 giai đoạn, giai đoạn thứ hai rộng 11 trên 16. Mỗi trường hợp, một ô được tái sử dụng
  bao nhiêu?
- Lớp small thực hiện 64 lần khởi chạy trải và 64 lần khởi chạy GEMM, mỗi GEMM là một hàng ô nhân 7
  cột ô.
- Phép trải thực hiện một phép chia số nguyên và một phép lấy dư (mỗi loại vài lần) cho mỗi phần tử
  nó ghi. NVIDIA ghi trong tài liệu rằng phép chia số nguyên tốn kém ("up to 20 instructions"), AMD
  ghi là "many more hardware cycles than multiplication", và phát biểu duy nhất của Apple là một bài
  nói năm 2016 (`notes/today/ch19-facts.md` A8, B5, C6). Kernel trực tiếp thực hiện một phép chia và
  một phép lấy dư mỗi luồng, rồi C*K*K phép nhân-cộng.
- Phép trải của lời giải cho mỗi luồng một phần tử của B^T theo đúng thứ tự của B^T, nên các luồng
  cạnh nhau ghi các float cạnh nhau và đọc X dọc theo một hàng của mảng vùng. Đánh số các luồng dọc
  theo v thay vào đó sẽ đọc X liên tiếp và ghi B^T với bước nhảy (stride) C*K*K. Bạn không thể có
  cả hai nếu không trung chuyển qua bộ nhớ chia sẻ (shared memory). Bạn đã chọn cách nào?
- Backend CUDA và HIP của llama.cpp ghi trong tài liệu một ngưỡng cắt: một lớp có `IC * KW * KH < 64`
  và `OW * OH < 512` được chuyển tới kernel trực tiếp thay vì implicit GEMM của nó
  (`notes/today/ch19-facts.md` D1). Quy tắc đó sẽ gửi lớp nào trong năm lớp tới đó? Đó là một lựa
  chọn production cho kernel và phần cứng của llama.cpp, không phải một dự đoán cho kernel và phần
  cứng của bạn.

Benchmark in ra, cho mỗi lớp, ba đường đi theo ms và GFLOP/s (2 N M C K^2 P / time), riêng các lần
trải và các lần GEMM kèm phần của phép trải trong lớp, và tỉ lệ của trải + GEMM so với trực tiếp.
Nó cũng in ra GEMM nào đã chạy và, trên Metal, kích thước buffer tạm bên cạnh `maxBufferLength`. Các
con số của nó là phép đo trên máy của bạn, không phải giá trị kỳ vọng. Trang của lab so sánh ba dự
đoán của bạn với chúng.

## Vượt ra ngoài sách (của khóa học này)

- **GEMM mỗi mẫu.** Trải cả batch thành một ma trận C*K*K x N*P sẽ cho một GEMM lớn, nhưng kết quả
  của nó ra dạng M x N x P (CNHW) và cần một phép hoán vị để thành NCHW. llama.cpp thực hiện lần
  chép đó; đường đi CUDA chậm của PyTorch lặp qua các mẫu như lab này
  (`notes/today/ch19-facts.md` D2, D4, E2).
- **B^T, không phải B.** Lưu ma trận đã trải với mỗi hàng một điểm ảnh đầu ra là điều mà bố cục
  trọng số (out, in) của `launchLinear` yêu cầu. Đó cũng là hướng của `cudnnIm2Col` đã lỗi thời
  trong cuDNN, của CUTLASS và của `ggml_im2col` trong llama.cpp (A4, A7, D2).
- **K = 1.** B chính là X, nhìn dưới dạng C x H*W, nên phép trải chỉ chuyển vị. llama.cpp, MLX và
  PyTorch đều bỏ qua nó và gọi GEMM trực tiếp trên X (D1, D3, D4).
- Nếu bạn so sánh với `torch.nn.functional.conv2d` trên GPU NVIDIA, một lời gọi float32 có thể chạy
  bằng TF32 trừ khi bạn tắt nó; hãy so sánh với một dung sai và in ra thiết lập đó (A6).
