> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 15.4: phần đệm và lần nạp vector

Sách: mục 15.6 (pp. 361-363), Fig. 15.11; Bài tập 1 (p. 370); lập luận cho lần nạp vector
khi occupancy thấp (mục 15.7, p. 364). Mốc dự án M3, giai đoạn 4 trên 5.

`work/` chứa lời giải của lab 15.3, đã đổi tên (`gemmPaddedKernel`, `launchGemmPadded`,
`launchGemmPaddedWidth`, `gemm_padded`, `encodeGemmPadded`), với `A_s` không có phần đệm và
thân của `loadTile` bị xóa. Hai thay đổi:

1. **Phần đệm.** Cho `A_s` chiều đầu (leading dimension) `bK + 1` ở mọi nơi nó được đánh chỉ số: khai báo của
   nó, lần ghi vào bộ nhớ chia sẻ trong `loadTile`, bước nhảy (stride) hàng của các lần đọc dải
   trong `mm`, và độ lệch bắt đầu của luồng trong lời gọi `mm`. Sách nêu ba chỗ đầu mà không
   nêu chỗ cuối (p. 363; `notes/ch15.md`, "Unclear" 4). Bỏ nó đi và xem cái gì thất bại.
2. **Lần nạp vector** (Bài tập 1). Mỗi ô của kernel này là 1024 số float, 256 `float4`: một
   lần nạp `float4` mỗi luồng mỗi ô, khi an toàn. Các hàng của ô A căn chỉnh 16 byte khi K là
   bội của 4, của ô B khi N là bội của 4 (trên CUDA, A và B đến từ `cudaMalloc`, căn chỉnh ít
   nhất 256 byte; trên HIP chúng đến từ `hipMalloc`, mà tài liệu tham khảo bộ nhớ của AMD
   không nêu căn chỉnh nào (ch15-facts B8), nên cùng quy tắc ở đó là unverified), và bốn cột
   phải nằm trong ma trận. Nếu không, giữ đường vô hướng điền số 0 phía ngoài ma trận. Ghi vào
   bộ nhớ chia sẻ từng số float một: một hàng có đệm của `A_s` là 9 số float, nên một lần ghi
   `float4` ở đó sẽ không được căn chỉnh, và NVIDIA không ghi nhận quy tắc bank nào cho truy
   cập bộ nhớ chia sẻ 128 bit (`notes/today/ch15-facts.md` A2).

Sách viết khai báo `A_s[bM*(bK + 1)]`. Hãy đặt tên cho chiều đầu có đệm
(`#define LDA_S (bK + 1)`), viết `A_s[bM * LDA_S]`, và dùng cùng tên đó trong mọi chỉ số vào
`A_s`, để không chỉ số nào có thể giữ độ rộng cũ.

## Tệp

- `work/kernel.cu`: khai báo có đệm, mọi chỉ số vào `A_s`, và `loadTile`.
- `work/kernel.metal` (`gemm_padded`) và `work/launch.mm` (`encodeGemmPadded`): tương tự trong
  Metal, nơi các lần nạp vector đi qua các lần gắn thứ hai của A và B được khai báo
  `device const float4*`, như C đã làm cho các lần ghi.

## Lệnh

    python3 labs/pmpp.py run 15.4 --backend emu
    python3 labs/pmpp.py run 15.4
    python3 labs/pmpp.py run 15.4 --bench

## Các bài kiểm thử kiểm tra gì

Các bài kiểm thử của lab 15.3, ở cả hai độ rộng. Các hình dạng có K hoặc N không phải bội của
4 (1, 3, 7, 9, 129, 131, 511) chạy đường dự phòng vô hướng; các hình dạng khác chạy đường
vector. Mã khởi đầu (starter) để nguyên thì build được nhưng thất bại: `loadTile` không ghi gì,
nên `mm` nhân bất cứ thứ gì đang có trong bộ nhớ chia sẻ.

Độ lệch bắt đầu bị thiếu được phát hiện: với khai báo và chiều đầu có đệm nhưng
`&A_s[tRow * bK]`, mọi hình dạng kiểm thử có từ 5 hàng trở lên đều thất bại, trên trình giả
lập và trên Metal. Các hình dạng có 1, 3 hoặc 4 hàng thì qua, vì mọi hàng của chúng thuộc về
các luồng có ô con bắt đầu ở hàng 0 của ô, nơi hai độ lệch trùng nhau. Một bài kiểm thử chỉ
dùng ma trận nhỏ sẽ bỏ sót lỗi này.

## Trước khi bạn đo hiệu năng

Dự đoán **mức tăng tốc so với lab 15.3 ở 2048 x 2048 x 2048.** Cân nhắc:

- Trên 32 bank, xung đột dải A của cách bố trí lab 15.3 đi từ 8 chiều xuống không còn: các
  hàng cách nhau 4 bắt đầu cách nhau 36 word, ở các bank 0, 4, ..., 28 (Fig. 15.11; E1).
- Lần nạp vector giảm số lệnh nạp mỗi cặp ô từ 8 xuống 2 mỗi luồng. Lập luận của sách là một
  kernel chạy ít luồng mỗi SM phát ra quá ít yêu cầu bộ nhớ để dùng hết băng thông, và các yêu
  cầu rộng hơn bù lại điều đó (p. 364).
- Trên thiết bị AMD có 64 bank, các con số xung đột là mô hình của khóa học này; AMD không ghi
  nhận cách một truy cập 64 lane được chia ra (B2).
- Trên Metal, Apple hoàn toàn không ghi nhận bank của bộ nhớ threadgroup (C2): đây là một phép
  đo không có dự đoán nào từ nhà cung cấp.

## Vượt ra ngoài sách

- Chiều đầu nào bỏ được xung đột dải A (phép tính của khóa học này, E1): mọi số lẻ trên 32
  bank; mọi số không chia hết cho 4 trên 64 bank. MLX đệm các ô Metal của nó thêm 16 byte, một
  thực hành không có tài liệu nào của Apple đứng sau (C2, D3).
