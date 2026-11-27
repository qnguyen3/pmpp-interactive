> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Đoạn mã (snippets)

Mã mà các bài học của chương 20 hiển thị để phục vụ một lời giải thích, và không phải là một
lab. Các bài học nạp những tệp này lúc chạy bằng `useLabFile('ch20/snippets/<name>')`, nên
không có bản sao nào bị lệch. Trình chạy lab không build chúng.

| Tệp | Hiển thị ở | Được kiểm tra bởi |
|---|---|---|
| `s203_softmax_corrected.cu` | mục 20.3, trình chạy từng bước "Step one row" (Fig. 20.4 từng dòng một, đã sửa: `-FLT_MAX`, `cuda::maximum<>{}` và `cuda::std::plus<>{}`, log-sum-exp `L` thay cho `D`) | Kiểm tra front-end CUDA (với các khai báo thay thế cho vài tên CUB và libcu++ mà nó dùng, dưới `PMPP_CUDA_FRONTEND_CHECK`); `app/src/sim/masksoftmax.test.ts` cố định 28 dòng của nó theo các dòng mà mô hình của trình chạy từng bước (`app/src/sim/masksoftmax.ts`) tô sáng. Không được build bằng nvcc và không được chạy: unverified trên phần cứng CUDA. |

Không trang nào khác của chương hiển thị mã không phải là tệp lab: listing của mục 20.5 là lời
giải tham chiếu (solution) của lab 20.2 (`labs/ch20/lab2_flash/solution/`), listing thứ hai
của mục 20.3 là của lab 20.1, và các dòng llama.cpp của mục 20.5 là trích dẫn từ
`notes/today/ch20-facts.md` D1 (được cố định bởi `app/src/chapters/ch20/S205.test.ts`), không
phải mã của khóa học. Trang tổng kết không hiển thị kernel nào; các câu hỏi của nó lấy số liệu
từ các mô hình, được cố định bởi `app/src/chapters/ch20/Wrap.test.ts`.
`app/src/sim/attn-consistency.test.ts` kiểm tra rằng bốn mô hình attention của chương khớp với
nhau và với các hằng số của lab 20.2.
