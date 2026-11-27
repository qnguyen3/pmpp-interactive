> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Snippets

Các kernel mà các trang 18.3 và 18.6 của chương 18 trình bày để phục vụ lời giải thích, và
không phải là lab. Các trang nạp chúng lúc chạy bằng `useLabFile('ch18/snippets/<name>')`
và hiển thị hoặc một vùng (đoạn văn bản giữa `// region: <name>` và `// endregion`) hoặc một
hàm, nên không có bản sao nào có thể bị lệch. Chúng không được trình chạy lab dựng. Trang tổng kết
không hiển thị kernel nào và không có tệp nào ở đây.

| Tệp | Hiển thị ở | Được kiểm tra bởi |
|---|---|---|
| `s183_variants.cu` | trang 18.3: Figs. 18.6, 18.8, 18.10, 18.12 và 18.14 như bản in và như bản đã sửa (phép toán nguyên tử relaxed cho các race vô hại; một nhánh HIP với `atomicCAS` legacy trong Fig. 18.14), cùng tám dạng thăm đỉnh của trình chạy từng bước "Two threads, one neighbour", năm dạng trong đó cố ý sai | Phép kiểm tra front-end CUDA; `app/src/chapters/ch18/S183.snippets.test.ts` cố định mọi vùng và chạy chúng qua `check/s183_check.cu` |
| `check/s183_check.cu` | không ở đâu cả (một chương trình điều khiển (driver)) | Phép kiểm tra front-end CUDA. Được `S183.snippets.test.ts` dựng và chạy trên biến thể `emu` theo thứ tự thuận và ngược: toàn bộ các lần tìm kiếm với mỗi kernel trong tám kernel, so với `app/src/sim/bfs.ts`, và một mức của mỗi kịch bản trình chạy từng bước với mỗi dạng thăm đỉnh, so với `app/src/sim/bfsvisit.ts`; `visitWeakReset` được chạy với giới hạn thời gian và không được trả về |
| `s186_privatised_variants.cu` | trang 18.6, trình chạy từng bước "One block's private frontier": Fig. 18.15 đã sửa (một lần ghi nguyên tử, hoặc một phép kẹp, cho dòng 29) và các bản sao với lần ghi thường của trang, mỗi bản bỏ một rào chắn (barrier) khối, một lần ghi nhận dạng `if` và không kiểm tra tràn | Phép kiểm tra front-end CUDA; `app/src/sim/bfspriv.test.ts` cố định thân mỗi kernel với `app/src/sim/bfspriv.ts` và chạy nó qua `check/s186_privatised_check.cu` trên `emu` (ba thứ tự luồng) và `race` (ThreadSanitizer, báo cáo lần ghi thường); `S186.snippets.test.ts` cố định những gì trang trích dẫn |
| `s186_coop_swap.cu` | trang 18.6, trình chạy từng bước "Three grid barriers" và đoạn mã khởi chạy: phép hoán đổi frontier của Fig. 18.17 không có hàng đợi riêng, các bản sao mỗi bản bỏ một rào chắn lưới (grid), kernel một rào chắn của Exercise 3, và cách đặt kích thước cùng lệnh khởi chạy sáu đối số của sách | Phép kiểm tra front-end CUDA; `app/src/sim/gridbfs.test.ts` cố định thân mỗi kernel với `app/src/sim/gridbfs.ts` và chạy nó qua `check/s186_coop_check.cu` trên `emu`, kể cả một lưới thừa một khối |
| `check/s186_privatised_check.cu`, `check/s186_coop_check.cu` | không ở đâu cả (các chương trình điều khiển) | Phép kiểm tra front-end CUDA (mỗi tệp include snippet của nó khi được phân tích riêng). Các phép kiểm thử nối mỗi chương trình điều khiển vào snippet của nó với `PMPP_SNIPPET_APPENDED` được định nghĩa, dịch kết quả bằng `labs/tools/cu2emu.py` và dựng nó giống cách trình chạy lab dựng các trình giả lập (emulator) |

Không có gì ở đây từng được chạy trên GPU. Các trang ghi trạng thái "emulator-checked and
front-end-checked; unverified on CUDA and HIP hardware". Các nhánh HIP được viết dựa trên
`notes/today/ch18-facts.md` (B1, B2, B4) và chưa được biên dịch cho HIP.

Phép kiểm tra front-end CUDA là phép mà `labs/pmpp.py` chạy: `clang++ -x cuda --cuda-host-only
-nocudainc -nocudalib -fsyntax-only -std=c++17 -I labs/common/include -I
labs/common/include/pmpp/emu/shim -include pmpp/check/cuda_stub.h <file>`.
