> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Snippets

Mã mà các bài học của học phần X3 hiển thị và không phải là một lab. Các tệp là mã nguồn thật, nên một
bài học hiển thị đúng tệp đã được kiểm tra chứ không phải một bản sao có thể bị lệch đi. Trình chạy lab
không build chúng; test của mỗi tệp nằm trong `app/src/sim/`. Mỗi snippet một hàng. Mọi thứ ở đây đều là của khóa học này:
học phần X3 không có trong sách.

| Tệp | Hiển thị ở | Cách | Test |
|---|---|---|---|
| `ckpt_check.cpp` | mục X3.8, "Check a download by its size": đọc một header llama2.c phiên bản 0, dự đoán kích thước tệp, so sánh với kích thước trên đĩa | nạp lúc chạy | `app/src/sim/ckptformat.test.ts` |

## `ckpt_check.cpp`

Chỉ có C++ phía host (không có kernel). Công thức kích thước là `notes/today/x3-facts.md` E1 áp lên thứ tự
`memory_map_weights` của run.c (D1). Test biên dịch nó với `-Wall -Wextra -Werror`, ghi
các tệp thưa (sparse file) mang header của stories15M và stories260K (D2) với độ dài mà mô hình dự đoán,
một tệp thiếu 4,096 byte, một tệp có `vocab_size` âm và một tệp ngắn hơn một header, chạy
chương trình trên từng tệp và so sánh đầu ra cùng mã thoát của nó với `app/src/sim/ckptformat.ts`.
