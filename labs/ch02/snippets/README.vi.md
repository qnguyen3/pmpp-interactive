> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Snippets

Mã mà các bài học của chương 2 hiển thị để phục vụ một lời giải thích hoặc một câu hỏi, và không
phải là lab. Các tệp được giữ ở đây dưới dạng mã nguồn thật để phép kiểm tra front-end CUDA trong
`./verify.sh` phân tích cú pháp mọi tệp `.cu`, và để bài học hiển thị đúng tệp đã được kiểm tra
chứ không phải một bản sao có thể bị lệch. Trình chạy lab không biên dịch hay chạy chúng.

Cách bài học dùng chúng:

- nạp lúc chạy bằng `useLabFile('ch02/snippets/<name>')`; hoặc
- sao chép vào một tệp section, với một unit test bên cạnh section đó sẽ thất bại nếu bản sao và
  tệp khác nhau. Hãy sửa test cùng với tệp.

| Tệp | Hiển thị ở | Cách dùng | Test |
|---|---|---|---|
| `vecadd_sequential.cu` | mục 2.3, vòng lặp trước khi được chuyển đi | nạp | `app/src/chapters/ch02/S23.test.ts` |
| `vecadd_stub.cu` | mục 2.4, hàm stub và trình chạy từng bước của nó | nạp | `S24.test.ts`, `app/src/sim/offload.test.ts` |
| `vecadd_stub_metal.mm` | mục 2.4, trình chạy từng bước cho Metal | được test của mô hình đọc | `app/src/sim/offload.test.ts` |
| `ex9_foo.cu` | mục 2.6, bài tập 9 của sách | sao chép | `S26.snippets.test.ts` |
| `saxpy_stub_defect.cu` | trang tổng kết, câu hỏi đầu tiên | sao chép | `Wrap.snippets.test.ts` |
| `count_above_race.cu` | trang tổng kết, câu hỏi thứ ba | sao chép | `Wrap.snippets.test.ts` |

`vecadd_stub_metal.mm` không thuộc bản dựng nào. Chú thích đầu tệp của nó cho lệnh để biên dịch
và chạy nó thủ công trên máy Mac.

Hai trong số này cố ý sai. `saxpy_stub_defect.cu` không bao giờ sao chép `y` sang device dù
kernel của nó đọc `y`. `count_above_race.cu` cập nhật một biến đếm từ nhiều luồng mà không dùng
phép toán nguyên tử (atomic). Cả hai đều phân tích cú pháp được, và không tệp nào được phép trở
thành nền tảng của một lab.
