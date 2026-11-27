> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab C.1: ba cách cấp dữ liệu cho một kernel

Sách: phụ lục C, mục C.2 (pp. 624-625), C.3 (p. 625), C.6 (pp. 626-627, Fig. C.1)
và C.8 (p. 628). Kernel là một phép tra embedding theo nghĩa của sách (ch. 20, p. 480).
Các lượt chạy, các đường đi Metal, các phép thăm dò và mọi phép đo là của khóa học này: phụ lục
không cho mã nào cho C.2 và không cho thời gian đo nào.

Phụ lục C là lịch sử về việc dữ liệu của một kernel có thể nằm ở đâu và con trỏ nào chạm tới nó.
Lab này cho một kernel ba chỗ ở trên mỗi backend và đo thời gian của cả chuyến đi, đúng chuyến mà
Fig. C.1 mô tả: CPU tạo ra một bảng, kernel đọc nó, CPU dùng kết quả. Kernel được cung cấp sẵn.
Bạn viết phía host.

## Kernel

`gatherAdd`: `out[r*cols + c] += table[ids[r]*cols + c]`. Hàng r của kết quả là hàng
`ids[r]` của bảng: một phép tra token embedding, trong đó bảng là một từ vựng gồm các hàng
embedding và các id là token id. Nó chạy `passes` lần trên cùng dữ liệu, nên kết quả bằng
`passes` lần các hàng được gom. Các lượt chạy đại diện cho một kernel đọc đầu vào của nó nhiều
hơn một lần.

## Phần của bạn

Mỗi đường đi nhận một `FeedJob` (`feeding.h`): `rows` và `cols` của bảng, `count` id trong bộ nhớ
host mà chương trình điều khiển (driver) sở hữu, `passes`, và hai callback. `produce` ghi bảng,
như `fread` làm trong Fig. C.1; `consume` đọc kết quả, như `use_data` làm. Mỗi đường đi:

1. cấp phát bảng, các id và kết quả trong loại bộ nhớ riêng của nó;
2. gọi `produce` đúng một lần để ghi bảng, và chép các id vào;
3. khởi đầu kết quả bằng không;
4. khởi chạy kernel `passes` lần, rồi đảm bảo lượt cuối đã hoàn tất;
5. gọi `consume` đúng một lần, với một con trỏ mà CPU được phép đọc;
6. giải phóng những gì nó đã cấp phát và trả về 0, hoặc in ra cái gì thất bại và trả về 1.

**CUDA, HIP và trình giả lập (emulator)**, trong `work/kernel.cu`:

- `feedExplicit`: mô hình đơn giản của sách (p. 623). `produce` ghi vào bộ nhớ `malloc`;
  `cudaMalloc` ba mảng; `cudaMemcpy` bảng và các id vào; `cudaMemset` kết quả; các lượt chạy;
  `cudaMemcpy` kết quả về; `consume`.
- `feedMapped`: zero-copy (C.2), mà NVIDIA ngày nay gọi là mapped memory. `cudaHostAlloc`
  với `cudaHostAllocMapped` cho cả ba mảng; `produce` ghi tại chỗ; kernel nhận cùng các con trỏ
  đó; đồng bộ hóa; `consume` tại chỗ.
- `feedManaged`: Fig. C.1. `cudaMallocManaged` với `cudaMemAttachGlobal` cho cả ba;
  `produce`, các lượt chạy, `cudaDeviceSynchronize`, `consume`, tất cả qua cùng các con trỏ.

**Metal**, trong `work/launch.mm`:

- `feedPrivateCopy`: mô hình tường minh. `produce` ghi vào một buffer trung chuyển dùng chung;
  một blit, trong một command buffer riêng, chép nó vào một buffer private; các lượt chạy trên
  buffer private; một blit thứ hai chép kết quả vào một buffer dùng chung; `consume`.
- `feedNoCopy`: bộ nhớ host căn theo trang, độ dài làm tròn lên thành số trang nguyên, được bọc
  bằng `newBufferWithBytesNoCopy` (ch02-facts C2). llama.cpp đưa trọng số mô hình được ánh xạ bộ
  nhớ của nó cho Metal theo cách này (appC-facts D4).
- `feedShared`: `newBufferWithLength` với shared storage; `produce` ghi qua `contents`.

## Các tệp

- Sửa `work/kernel.cu` và `work/launch.mm`, bên dưới dòng `your code below`.
- Được cung cấp, không sửa:
  - phía trên dòng đó, kernel `gatherAddKernel` và `launchGatherAdd` (một lượt, 256
    luồng mỗi khối); `encodeGatherAdd` (một lượt vào một compute encoder đang mở) và
    `waitFor` (chờ một command buffer, báo lỗi nếu có, cộng dồn thời gian GPU của nó vào công
    việc chép hoặc kernel). Cả hai hàm khởi chạy đều ghi lại bảng mà chúng nhận được, để phục vụ
    kiểm thử;
  - `work/kernel.metal`, kernel Metal, hoàn chỉnh;
  - `feeding.h`, job và mọi khai báo;
  - `main.cpp`, chương trình điều khiển, kèm tham chiếu CPU;
  - `../shared/probe.cu`, các công cụ đo CUDA và HIP của benchmark.

## Lệnh

- Kiểm thử: `python3 labs/pmpp.py run appC/lab1_feeding --backend emu` trước, rồi
  `python3 labs/pmpp.py run appC/lab1_feeding`. Trình chạy cũng chấp nhận số của lab,
  `C.1`, thay cho thư mục.
- Đo: `python3 labs/pmpp.py run appC/lab1_feeding --bench` (chỉ thiết bị thật).
- Lời giải tham chiếu: thêm `--solution`.

## Lab này khác sách ở đâu

- **Mapped memory ngày nay không cần con trỏ thứ hai.** Sách nói con trỏ `cudaHostAlloc`
  không thể truyền cho kernel, và mã host phải lấy một con trỏ khác từ
  `cudaHostGetDevicePointer()` (p. 624). Đó là CUDA 2.2. Mọi tiến trình CUDA 64-bit ngày nay
  đều có unified addressing, và NVIDIA ghi rằng bộ nhớ `cudaHostAlloc` truy cập được từ device
  qua cùng con trỏ, có hay không có cờ mapped, và `cudaHostGetDevicePointer()` là không cần thiết
  cho nó (appC-facts A3, A5). Kiểm thử kiểm tra rằng kernel nhận đúng con trỏ mà `produce` đã ghi
  qua. Trình giả lập không cung cấp `cudaHostGetDevicePointer`, nên một lời gọi tới nó không dựng
  được ở đó. Hãy giữ cờ mapped: trên HIP nó làm bộ nhớ trở thành fine-grained (B2).
- **Hai cái tên cho bộ nhớ ghim (pinned memory).** Sách cấp phát bộ nhớ ghim bằng `cudaHostAlloc`
  ở đây và bằng `cudaMallocHost` ở chương 23, mà không liên hệ chúng với nhau. Chúng là cùng một
  kiểu cấp phát, có và không có cờ: `cudaHostAllocDefault` "causes cudaHostAlloc() to
  emulate cudaMallocHost()" (infra-facts A8).
- **Fig. C.1 như bản in** bỏ mất dấu chấm phẩy sau `cudaDeviceSynchronize()` và lược bớt cấu hình
  khởi chạy. `feedManaged` có cả hai và kiểm tra mọi lời gọi.
- **Hai nghĩa của "unified memory".** Nghĩa của sách (CUDA 6) là một vùng các trang mà runtime
  di chuyển giữa hai bộ nhớ vật lý. Nghĩa của Apple là một bộ nhớ vật lý duy nhất mà không gì phải
  di chuyển (appC-facts, ghi chú thuật ngữ). Metal không có chế độ managed trên Apple silicon
  (ch05-facts C6), nên đường đi thứ ba của nó là một buffer dùng chung.

## Kiểm thử làm gì

Các hình dạng, rows x cols kiểu float: 1 x 1, 7 x 3, 64 x 33 và 130 x 17; thiết bị thật thêm
4096 x 256. Hai bộ id: dày (mỗi hàng một lần, theo thứ tự) và thưa (một hàng trong tám,
làm tròn lên, rút từ một seed cố định có lặp lại, luôn gồm hàng cuối, nơi lỗi lệch một lộ ra).
Một lượt và ba lượt. Với mỗi đường đi và mỗi trường hợp:

1. Kết quả mà `consume` thấy bằng `passes` lần các hàng được gom của tham chiếu CPU trong
   `main.cpp`, trong một dung sai. Mọi giá trị đúng cũng là một số nguyên chính xác
   (bảng chứa số nguyên và kernel chỉ cộng), và điều đó cũng được kiểm tra.
2. `produce` và `consume` mỗi hàm được gọi đúng một lần, với con trỏ khác null.
3. Cấu trúc. Với `feedMapped`, `feedManaged`, `feedNoCopy` và `feedShared`, bảng mà hàm khởi chạy
   được cung cấp ghi lại chính là vùng nhớ mà `produce` đã ghi: một con trỏ phục vụ cả hai phía.
   Với `feedExplicit` và `feedPrivateCopy` thì không, nên đã có một lần sao chép, và trên Metal buffer
   mà kernel đọc có private storage.
4. `feedNoCopy`: con trỏ mà `produce` ghi qua là bội của kích thước trang (`getpagesize()`).

Starter chưa đụng tới trả về -1 từ mọi hàm, nên mọi trường hợp đều thất bại.

## Kiểm thử thấy được và không thấy được gì

- **Một lần đồng bộ hóa bị thiếu.** Trên trình giả lập, mọi lời gọi đều đồng bộ, và
  `cudaHostAlloc` và `cudaMallocManaged` là các cấp phát host thông thường. Một `feedMapped` hay
  `feedManaged` không có `cudaDeviceSynchronize` trước `consume` vẫn qua ở đó, và không có hiệu ứng
  thời gian nào lộ ra. Chỉ thiết bị thật mới cho thấy lỗi: dưới dạng một race trên Linux với
  unified memory đầy đủ (`consume` đọc trong khi GPU có thể vẫn đang ghi), và, với bộ nhớ được quản
  lý, dưới dạng một segmentation fault ở nơi `ConcurrentManagedAccess` là 0, bao gồm mọi hệ thống
  Windows và WSL (appC-facts A10).
- **Một kết quả không bắt đầu từ không.** Các cấp phát mới của trình giả lập chứa các byte được đọc
  như một float khoảng -3e-16. Cộng vào một giá trị của bảng, nó biến mất, trừ nơi bảng chứa 0.
  Phép kiểm tra số nguyên của kiểm thử 1 bắt được nó ở bất cứ đâu một hàng được gom chứa số 0, điều
  xảy ra trong các trường hợp 64 x 33 và 130 x 17. Trên Metal, `newBufferWithLength` xóa sạch những
  gì nó cấp phát (ch02-facts C2); bộ nhớ `posix_memalign` thì không nhất thiết sạch. Trên máy dùng
  để xây lab này, bỏ `memset` khỏi `feedNoCopy` làm phần lớn các trường hợp của nó thất bại.
- **Dùng kết quả trước khi chờ, trên Metal.** Trên máy dùng để xây lab này, kết quả của đường đi
  dùng chung khi đó đọc ra toàn số không và mọi trường hợp thất bại. Metal không hứa kết quả đó; nó
  chỉ đặt việc sắp thứ tự lên vai bạn (ch05-facts C6).
- **Căn lề, trên Metal.** Trên cùng máy đó, `newBufferWithBytesNoCopy` chấp nhận bộ nhớ không căn
  theo trang, và kết quả vẫn đúng. Tài liệu của Apple yêu cầu bộ nhớ căn theo trang
  (ch02-facts C2), và kiểm thử 4 là kiểm thử thất bại.
- Rò rỉ bộ nhớ không được kiểm tra, việc mỗi lần giải phóng có khớp với kiểu cấp phát của nó cũng
  không (trên trình giả lập, tất cả đều là `free`).
- Kiểm thử không thể biết một đường đi có nhanh hay không. Đó là việc của benchmark.

## Trước khi đo hiệu năng, hãy dự đoán

Một công cụ để dự đoán, quy tắc hòa vốn của C.2. Nó là của khóa học này, rút ra từ lời khuyên của
sách chỉ dùng bộ nhớ zero-copy cho dữ liệu mà kernel đọc "occasionally and sparsely" (p. 625). Trên
một GPU rời, đường đi tường minh chuyển cả bảng qua đường kết nối một lần rồi đọc bộ nhớ device.
Đường đi mapped chỉ chuyển những gì kernel chạm tới, nhưng đi qua đường kết nối ở mỗi lần chạm. Bộ
nhớ device nhanh hơn đường kết nối nhiều (sách nói đường kết nối dưới một phần mười, p. 625; hàng
thăm dò của benchmark đo cả hai trên máy của bạn). Vậy đại khái mapped chỉ thắng khi kernel đọc ít
hơn khoảng một lượng bằng cả bảng trong suốt vòng đời của nó. Benchmark in lượng đó dưới tên
`touched`, số byte của bảng mà các lượt chạy đọc chia cho kích thước bảng: 1 cho `dense1`,
8 cho `dense8`, 1/64 cho `sparse1`.

Hãy viết ra những điều sau trước. Bài học hỏi ba điều đầu dưới các id trong ngoặc.

1. CUDA hoặc HIP, `sparse1`: thời gian tường minh chia cho thời gian mapped (`appC/labs/p-sparse`).
2. CUDA hoặc HIP, `dense8`: thời gian mapped chia cho thời gian tường minh (`appC/labs/p-repeat`).
3. Metal, `dense8`: thời gian private-copy chia cho thời gian shared, và liệu no-copy có khác
   shared chút nào không (`appC/labs/p-metal`). Trên Apple silicon, không có đường kết nối nào
   tách bộ nhớ của CPU khỏi bộ nhớ của GPU (appC-facts C2). Apple nói Metal "may apply
   additional optimizations to private resources" và không nói là những tối ưu nào.
4. Tra mô hình managed của bạn (benchmark CUDA và HIP in nó đầu tiên; trang bài học của C.7 giải
   thích bốn mô hình) và nói `managed` sẽ giống đường đi nào: tường minh sau lượt đầu trên Linux
   với unified memory đầy đủ, nơi một trang thường di chuyển tới bộ xử lý nào đang dùng nó (A11);
   một lần di chuyển hàng loạt toàn bộ bộ nhớ được quản lý khi GPU bắt đầu, và ngược về khi bạn
   đồng bộ hóa, trên Windows và WSL (A10); mapped trên AMD không có XNACK, nơi `hipMallocManaged`
   "behaves the same as hipMallocHost" (B5).

## Đọc kết quả in ra

Bảng có 256 float mỗi hàng và số hàng bằng số vừa trong một phần tám bộ nhớ mà thiết bị báo
(`totalGlobalMem` trên CUDA và HIP; giá trị nhỏ hơn giữa `maxBufferLength` và
`recommendedMaxWorkingSetSize` trên Metal), tối đa 256 MiB. Mức trần 256 MiB là một tham số của lab
này, để giữ một lần chạy ngắn, không phải dữ kiện về thiết bị nào. Các dòng đầu in ra số byte được
chọn và lý do.

- CUDA và HIP in mô hình unified memory: ba thuộc tính của appC-facts A14 và tên của mô hình mà
  chúng tạo thành. Rồi đến phép thăm dò: một `cudaMemcpy` từ host ghim sang device và một
  `cudaMemcpy` từ device sang device với kích thước bằng bảng, đo bằng event, lấy lần tốt nhất trong
  vài lần. Băng thông device tính số byte đọc cộng số byte ghi. Tỉ lệ giữa chúng là câu trả lời của
  máy bạn cho câu "less than 10%" của sách. `feedExplicit` chép từ bộ nhớ `malloc`, không phải bộ
  nhớ ghim, nên lần sao chép của nó có thể chạy dưới con số của phép thăm dò (ch02-facts A16).
- Metal in ra liệu địa chỉ CPU của một buffer (`contents`) có bằng địa chỉ GPU của nó
  (`gpuAddress`, macOS 13 trở lên) không, cho một buffer shared và một buffer no-copy. Apple không
  ghi câu trả lời nào: đây là một quan sát trên máy của bạn (appC-facts C3).
- Rồi mỗi workload và đường đi một hàng. `seconds` là toàn bộ lời gọi feed, lần tốt nhất trong vài
  lần gọi sau một lần khởi động không đo: cấp phát, `produce`, các lần sao chép, các lượt chạy, đồng bộ
  hóa, `consume` và giải phóng. Trên Metal, `GPU copy` và `GPU kernels` cộng dồn
  `GPUEndTime - GPUStartTime` của các command buffer mà lời gọi đó đã truyền cho `waitFor` dưới dạng
  công việc chép và kernel (ch05-facts C9).
- Việc cấp phát cố ý nằm trong thời gian đo: hàm của Fig. C.1 cũng cấp phát. Một bộ cấp phát host
  có thể trả lại bộ nhớ nó đang giữ từ lần gọi trước, trong khi `newBufferWithLength` xóa sạch những
  gì nó trao ra (ch02-facts C2), nên một phần chênh lệch giữa hai đường đi có thể đến từ cấp phát
  chứ không phải từ các lần sao chép. Hãy nhìn các cột GPU trước khi kết luận.
- Các dòng cuối chia các thời gian mà dự đoán của bạn đã hỏi.

Kết quả của lần gọi khởi động và của lần gọi cuối được kiểm tra; một kết quả sai được báo mà không
kèm thời gian. Các con số là phép đo trên máy của bạn ở một kích thước bảng, không phải thuộc tính
của các đường đi.

## Các backend khác nhau ở đâu (không có trong sách)

- **CUDA**: mapped memory là bộ nhớ host ghim, một con trỏ cho cả hai phía dưới unified
  addressing, mọi lần truy cập đều đi qua đường kết nối (A3). Bộ nhớ được quản lý gây lỗi trang và
  di chuyển từng trang ở nơi `ConcurrentManagedAccess` là 1 (A11), và di chuyển hàng loạt ở nơi nó
  là 0 (A10).
- **HIP**: cùng các lời gọi qua `pmpp/vendors/hip.h`. `hipHostMalloc` với cờ mapped cho bộ nhớ
  fine-grained (B2). `hipMallocManaged` chỉ di chuyển trang khi bật XNACK (`HSA_XNACK=1`) và có một
  driver hỗ trợ HMM; nếu không, nó hoạt động như bộ nhớ host ghim (B5). `cudaMallocManaged` và
  `cudaMemAttachGlobal` đến được HIP qua hai define tạm trong `feeding.h` cho đến khi header dùng
  chung ánh xạ chúng. Chưa chạy trên phần cứng AMD ở đây.
- **Metal**: một bộ nhớ. Một buffer shared không cần chép (ch05-facts C6); một buffer private không
  có địa chỉ CPU (ch02-facts C2); một buffer no-copy bọc bộ nhớ căn theo trang của chính bạn (C2).
  Không có chế độ managed nào và không có việc di chuyển trang khi GPU gặp lỗi trang được ghi trong
  tài liệu cho Apple silicon (appC-facts C5).

## Đi xa hơn

- `feedExplicit` trung chuyển qua bộ nhớ `malloc`. Hãy trung chuyển qua `cudaMallocHost` thay vào
  đó và so sánh lần sao chép của nó với con số của phép thăm dò. Một buffer trung chuyển có còn đáng
  giữ khi nó đã được ghim, hay `produce` nên ghi thẳng vào buffer ghim và lần sao chép đi thẳng từ đó?
- Prefetch bảng sang device trong `feedManaged` trước lượt đầu tiên. Lời gọi này đã thay đổi trong
  CUDA 13 để nhận một `cudaMemLocation` (appC-facts A14), còn header mặc định của HIP vẫn nhận một
  device kiểu `int` (B6), nên dòng đó cần một `#if`. Không trình giả lập nào có lời gọi này: phần
  này chỉ dành cho thiết bị thật.

## Trạng thái của lời giải tham chiếu

verified on Metal (kiểm thử và benchmark), emulator-checked, front-end-checked cho
`kernel.cu` và `../shared/probe.cu`. Kiểm tra front-end chỉ thấy nhánh trình giả lập của truy vấn
mô hình: các nhánh CUDA và HIP của nó dùng các thuộc tính device mà header stub không khai báo.
Trên CUDA và trên HIP: unverified, chưa có máy nào có CUDA hay HIP chạy nó.
