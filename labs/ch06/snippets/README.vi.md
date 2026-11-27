> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Snippets

Các kernel mà bài học của chương 6 hiển thị để phục vụ một câu hỏi hay một bài tập, và không phải là
lab. Chúng được giữ ở đây dưới dạng tệp `.cu` thật để `./verify.sh` chạy kiểm tra front-end CUDA
trên chúng, và các unit test sẽ thất bại nếu bản sao hiển thị trong bài học lệch khỏi tệp. Chúng
không được biên dịch hay chạy.

- `s61_foo_kernel.cu`: kernel của bài tập 1 trong sách (pp. 154 to 155), đúng như in, mỗi dòng được
  đánh số là một dòng mã nguồn để số dòng của bài học trùng với của sách (sách ngắt dòng 01 và 07).
  Bài luyện của mục 6.1 phân loại chín lần truy cập của nó.
  `app/src/chapters/ch06/snippets-s61.test.ts` gắn bản sao trong bài học
  (`app/src/chapters/ch06/S61.tsx`) với tệp này và kiểm tra rằng mô hình truy cập
  (`app/src/sim/coalesce.ts`, `exercise1`) đọc cùng chín biểu thức trên cùng các dòng.
- `s65_coarsen_toy.cu`: ví dụ minh họa gộp việc của sách (pp. 141 to 142) trước và sau khi gộp việc
  theo hệ số 4, với các kiểm tra biên của chúng tôi và biến thể ánh xạ có bước nhảy của chúng tôi.
  Mục 6.5 hiển thị nó; `app/src/chapters/ch06/snippets-s65.test.ts` gắn bản sao trong `S65.tsx` với
  tệp này và cố định các dòng được tô sáng.
- `s66_unroll.cu`: vòng lặp `A(i) B(i)` 16 lần lặp của sách (p. 144) dưới dạng một kernel, rồi được
  trải vòng lặp theo hệ số 4 và sắp xếp lại bằng tay, rồi với `#pragma unroll 4`. A và B là của
  chúng tôi. Mục 6.6 hiển thị nó; `snippets-s66.test.ts` gắn bản sao trong `S66.tsx` với nó.
- `s66_where_x_lives.cu`: năm phiên bản của một phép tính đã gộp việc cho phần phân loại "where does
  x live?" của mục 6.6. `app/src/sim/unroll.ts` chứa các bản sao mà trang hiển thị và
  `app/src/sim/unroll.test.ts` kiểm tra từng bản có trong tệp này đúng từng chữ. Việc `x` cuối cùng
  nằm ở đâu là quy tắc của sách và danh sách của NVIDIA (ch06-facts A13); không gì ở đây được biên
  dịch bằng nvcc để kiểm chứng.
- `s67_exchange.cu`: vòng lặp trao đổi của sách (pp. 146 to 147), một buffer chia sẻ với hai rào
  chắn và hai buffer với một rào chắn, được làm cho biên dịch được (phép tính, các khai báo và
  `swapPointers` là của chúng tôi). Mục 6.7 hiển thị nó; `snippets-s67.test.ts` gắn bản sao trong
  `S67.tsx` với nó và kiểm tra rằng các vòng lặp của trình chạy từng bước
  (`app/src/sim/doublebuf.ts`, `loopSource`) là các vòng lặp của tệp này, đúng từng dòng.
- `wrap_scores.cu`: một kernel của chúng tôi cho hai câu hỏi ở trang tổng kết, các điểm attention
  được đọc từ một key cache lưu theo (token, head, dim). Các lần đọc `K` của nó không được hợp nhất,
  và đó là trọng tâm của các câu hỏi. `app/src/chapters/ch06/snippets-wrap.test.ts` gắn bản sao
  trong `app/src/chapters/ch06/Wrap.tsx` với tệp này và cố định các số sector mà các câu hỏi lấy từ
  `app/src/sim/coalesce.ts`.
