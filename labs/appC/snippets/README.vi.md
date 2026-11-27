> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Snippets

Mã mà các bài học của Phụ lục C hiển thị để giải thích, và không phải là một lab. Chúng được giữ
ở đây dưới dạng tệp thật để `./verify.sh` chạy kiểm tra front-end CUDA trên các tệp `.cu`, và các
unit test sẽ thất bại nếu bản sao mà một bài học hiển thị lệch khỏi tệp. Trình chạy lab không dựng
hay chạy chúng; `check.py` (bên dưới) biên dịch và chạy các tệp Metal.

| Tệp | Hiển thị ở | Được kiểm tra bởi |
|---|---|---|
| `c1_helper_generic.cu` | mục C.1, "The helper, in both languages", tab CUDA: một `rowMax` với con trỏ thường, được gọi trên bộ nhớ toàn cục và trên bộ nhớ chia sẻ | kiểm tra front-end CUDA; `app/src/chapters/appC/SC1.snippets.test.ts` kiểm tra rằng bản sao của bài học chính là tệp này |
| `c1_helper_overloads.metal` | cùng khối đó, tab Metal: cặp overload `rowMax` mà MSL cần, mỗi overload cho một không gian địa chỉ | `SC1.snippets.test.ts`, như trên; `check.py` biên dịch nó ở mọi phiên bản MSL từ 2.0 đến 4.1 và chạy kernel của nó trên GPU, đối chiếu với một tham chiếu CPU ("verified on Metal" của tab) |
| `c3_chase.cu` | mục C.3, "Follow the list": kernel một luồng duyệt một danh sách liên kết được dựng trên host | kiểm tra front-end CUDA; `app/src/chapters/appC/SC3.snippets.test.ts` kiểm tra bản sao của bài học và các con số mà câu hỏi của nó nêu |
| `c6_sortfile_cpu.cu` | mục C.6, "Fig. C.1, line by line": bản gốc chạy trên CPU (Fig. C.1, ô bên trái, p. 627) | kiểm tra front-end CUDA; `app/src/chapters/appC/SC6.snippets.test.ts` kiểm tra rằng các dòng của trình chạy từng bước (`app/src/sim/managedmem.ts`, `figLines('cpu')`) chính là `sortfile` của tệp này, khớp từng ký tự |
| `c6_sortfile_managed.cu` | cùng trình chạy từng bước đó: bản chuyển sang bộ nhớ được quản lý (Fig. C.1, ô bên phải) | như trên, `figLines('managed')`; kiểm thử còn kiểm tra rằng nó khác bản gốc CPU đúng ở bốn dòng mà hình đánh dấu |
| `c6_sortfile_explicit.cu` | cùng trình chạy từng bước đó: cột thứ ba của khóa học này, bản chuyển dùng `cudaMalloc` và hai lời gọi `cudaMemcpy` (không có trong sách) | như trên, `figLines('explicit')` |
| `c9_vmm_decls.h`, `c9_vmm_decls_hip.h` | mục C.9, các tab "CUDA Driver API" và "HIP": các khai báo reserve, create, map và grant | không được biên dịch ở đâu cả (stub front-end CUDA không có Driver API, và kiểm tra front-end chỉ đọc tệp `.cu`); `app/src/chapters/appC/SC9.test.ts` kiểm tra rằng các bản sao của bài học chính là các tệp này và mọi khai báo trong chúng đều được trích trong `notes/today/appC-facts.md` (A15, B7) |
| `wrap_layer_table.cu` | trang tổng kết, câu hỏi đầu tiên của "A mixed check": một bảng các lớp được quản lý chứa con trỏ `cudaMalloc`, với một lần đọc từ host vào một trong số chúng, cố ý | kiểm tra front-end CUDA; `app/src/chapters/appC/Wrap.snippets.test.ts` kiểm tra bản sao của trang và các con số của hai câu hỏi số của trang |

Ba tệp `c6_` chỉ là mã host: sách không cho định nghĩa của hàm sắp xếp hay của `use_data`, nên cả
hai đều được khai báo mà không bao giờ được định nghĩa. Có hai điểm trong chúng khác với trang
sách: dấu chấm phẩy bị thiếu sau `cudaDeviceSynchronize()` được thêm vào (nếu để nguyên như bản in, kiểm tra
front-end dừng ở đó), và `<<<...>>>` bị lược bớt trở thành `grid` và `block`, được khai báo một lần
ở phạm vi tệp làm vật thay thế. Tên `qsort_char` của hình được giữ nguyên, dù phần chữ của sách
nói `qsort()`.

## Chạy các tệp Metal: `check.py`

```
python3 labs/appC/snippets/check.py              # build in a temporary directory
python3 labs/appC/snippets/check.py --build DIR  # or $PMPP_BUILD_DIR/appC-snippets
```

Cần macOS với bộ công cụ Metal của Xcode (`xcrun metal`). Nó kiểm tra:

| Cái gì | Kỳ vọng | Nêu ở |
|---|---|---|
| `c1_helper_overloads.metal`, biên dịch với mọi `-std` mà trình biên dịch cung cấp cho macOS (`macos-metal2.0` đến `metal4.1`) và không có `-std` nào | lần nào cũng biên dịch được | mục C.1, chú thích của tab Metal |
| cùng tệp đó, chạy trên GPU bởi `check/c1_run.mm` (kernel `row_max_twice`, nhiều hình dạng) | cả hai giá trị lớn nhất của mọi hàng đều bằng một tham chiếu CPU | trạng thái của tab Metal, "verified on Metal" |
| năm thẻ MSL của mục C.1 (`app/src/sim/addrspace.ts`, `CARDS`), mỗi thẻ được viết ra thành một hàm trợ giúp và một kernel | đến `metal4.0`: đúng như đặc tả xếp chúng (`msl-none` thất bại với "pointer type must have explicit address space qualifier", `msl-device-only` với "no matching function for call", ba thẻ còn lại biên dịch được); ở `metal4.1` và mặc định, chỉ `msl-none` thay đổi, và biên dịch được | mục C.1, chú thích cuối của các thẻ và công tắc "observed" của chúng |
| `check/c1_generic_probe.metal`, một hàm trợ giúp với con trỏ không có qualifier | bị từ chối đến `metal4.0`, được chấp nhận ở `metal4.1` và mặc định; chạy trên GPU, nó trả về đúng các giá trị lớn nhất từ bộ nhớ device, threadgroup và thread | mục C.1, "Observed on the build machine"; `notes/today/appC-facts.md` C1 |

`check/c1_generic_probe.metal` không phải MSL hợp lệ theo đặc tả của nó và không được hiển thị như
mã để sao chép. Không có gì trong `check.py` chạy trên CUDA hay HIP.
