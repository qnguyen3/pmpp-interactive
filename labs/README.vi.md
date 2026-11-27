> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Các lab

Mỗi lab (bài thực hành) nằm trong `chNN/labM_name/` và có ba bản sao của các tệp bạn sửa:

- `starter/`  điểm xuất phát chưa bị đụng tới (mã khởi đầu)
- `work/`     **của bạn**: sửa các tệp này (ban đầu nó là bản sao của `starter/`)
- `solution/` lời giải tham chiếu; chỉ xem sau khi bạn đã tự thử

Các tệp là `kernel.cu` (CUDA; cùng tệp này được dựng cho HIP và cho trình giả lập (emulator)
trên CPU), `kernel.metal` (Metal Shading Language) và `launch.mm` (phía host của Metal).
`main.cpp` là chương trình điều khiển (driver) kiểm thử và benchmark; bạn không cần sửa nó.

```sh
python3 labs/pmpp.py run 5.2                # test your work on every backend found
python3 labs/pmpp.py run 5.2 --backend emu  # one backend: emu, race, metal, cuda or hip
python3 labs/pmpp.py run 5.2 --bench        # also benchmark (real devices only)
python3 labs/pmpp.py run 5.2 --bench --reps 5   # benchmark five times, keeping every repetition
python3 labs/pmpp.py run 5.2 --solution     # the reference solution instead of yours
python3 labs/pmpp.py run 5.2 --full         # sanitizer reports in full (they are condensed by default)
python3 labs/pmpp.py reset 5.2 --yes        # throw your work away and start again
python3 labs/pmpp.py doctor                 # backends found, device properties
python3 labs/pmpp.py starters               # is every current starter in the starter history?
```

`work/` của bạn được tính là *chưa đụng tới* (untouched) khi nó giống hệt `starter/` hiện tại
hoặc bất kỳ phiên bản nào trước đó của nó được ghi trong `labs/starter-history.json`. Khi một
starter đã được sửa kể từ lúc bản sao của bạn được tạo, `run` và `doctor` sẽ báo điều đó:
`reset <lab> --yes` mang về starter đã sửa (chỉ làm vậy nếu bạn chưa bắt đầu lab; nó ghi đè
`work/`).

### Một lần chạy ghi lại và in ra những gì

- Kết quả của mỗi lần chạy được ghi vào `reports/runs/<backend>.json`, nơi ứng dụng đọc. Một lần
  `run` thông thường (không có `--bench`) giữ lại benchmark gần nhất của một kernel vẫn còn qua
  kiểm thử, kèm thời điểm nó được đo.
- `--reps N` chạy benchmark N lần, lần lượt nối tiếp nhau, và giữ mỗi lần lặp cùng thời điểm nó
  bắt đầu (`bench_reps` trong JSON, mọi lần lặp trong một báo cáo `verify`). Ứng dụng hiển thị
  lần đầu tiên. Hãy dùng nó trên một máy có thời gian đo thay đổi giữa các lần chạy; trình chạy
  không tính trung bình và không chọn lần tốt nhất, nó ghi lại đúng những gì đã đo.
- `--no-test` đo hiệu năng mà không kiểm thử. Kết quả kiểm thử gần nhất chỉ được giữ khi bản dựng
  đọc đúng những tệp như cũ: mã nguồn của biến thể, `main.cpp` và các header của lab cùng các tệp
  `provided/`, các mã nguồn dùng chung và được cung cấp mà lab biên dịch (và các tệp `.metal` bên
  cạnh chúng, mà Metal biên dịch lúc chạy), các tệp của những lab trước mà một lab mốc dự án nêu
  trong compile definitions của nó, mọi header mà các tệp phụ thuộc của trình biên dịch liệt kê, và
  chính các compile definitions. Bất cứ thứ gì thay đổi, kết quả kiểm thử sẽ bị bỏ.
  (Với Ninja generator, phần header bị thiếu; Makefile generator mặc định thì có.)
- Một báo cáo sanitizer được rút gọn: dòng tiêu đề của báo cáo đầu tiên (lỗi gì và ở địa chỉ nào),
  các stack của nó được cắt chỉ còn những frame trong mã nguồn của chính lab, và dòng SUMMARY của
  nó; các báo cáo sau chỉ còn dòng SUMMARY. Một ghi chú cho biết bao nhiêu frame đã bị bỏ.
  `--full` (trên `run`, `selftest` và `verify`) hiển thị toàn bộ. Báo cáo data race của
  ThreadSanitizer trên `race` giữ dạng ngắn riêng của chúng (cả hai lần truy cập và dòng mã nguồn
  của chúng).
- Một backend mà máy có nhưng lab không dựng cho nó sẽ nói lý do: "no Metal build for
  this lab" khi biến thể không có `kernel.metal` và `launch.mm`, "not built for the
  race-detection emulator" khi không có `RACE`, hoặc một thư viện mà lab liên kết bị thiếu ở đây.

### Thiết lập bản dựng

- `PMPP_BUILD_DIR=<dir>` dựng ở một chỗ khác thay vì `labs/build`.
- `PMPP_ONLY=ch06` chỉ cấu hình các lab của một chương.
- `PMPP_CMAKE_ARGS="-DPMPP_CH15_VENDOR=OFF -DCMAKE_BUILD_TYPE=Debug"` truyền đối số cho bước
  cấu hình (tách như shell sẽ tách). CMake giữ một giá trị `-D` trong cache của thư mục dựng sau
  khi bạn thôi truyền nó: hãy đặt lại nó một cách tường minh, hoặc dùng một `PMPP_BUILD_DIR` mới.
  Các công tắc mà bản dựng biết: `PMPP_DISABLE_RACE`, `PMPP_DISABLE_METAL`, `PMPP_DISABLE_CUDA`,
  `PMPP_DISABLE_HIP`, `PMPP_EMU_SANITIZE=OFF`, và các tùy chọn riêng của từng lab mà README của
  chúng nêu.
- `doctor` in ra thuộc tính thiết bị của mỗi backend. Trên CUDA và HIP, chúng gồm
  `asyncEngineCount` và `concurrentKernels` (liệu các lần sao chép và các kernel có thể chồng lên nhau
  hay không, chương 22 và 23); trên ROCm, `asyncEngineCount` là một hằng số của runtime, không
  phải số lượng copy engine (`notes/today/ch23-facts.md`, B3).

## Trình giả lập

`emu` chạy kernel `.cu` chưa sửa đổi của bạn trên CPU. Nó tồn tại để logic kernel có thể được
kiểm thử trên một máy không có thiết bị CUDA hay HIP, và để các lỗi đồng bộ hóa thất bại mọi lần
thay vì chỉ thỉnh thoảng.

- Mọi kiểm thử chạy dưới ba thứ tự luồng tất định (theo thứ tự, đảo ngược, xáo trộn).
  Một luồng chạy cho đến khi nó gặp một điểm đồng bộ hóa (`__syncthreads()`, một lời gọi
  `*_sync` cấp warp, một `sync()` của cooperative groups) hoặc trả về, rồi đến luồng tiếp theo.
- Nó được dựng với AddressSanitizer: một lần đọc hoặc ghi ngoài biên trong kernel sẽ dừng lần chạy
  với một báo cáo nêu rõ dòng mã nguồn.
- Nó mô phỏng phép tính chỉ số, bộ nhớ chia sẻ (shared memory), rào chắn (barrier), di chuyển dữ
  liệu cấp warp, phép toán nguyên tử (atomic) và biên. Nó không mô phỏng thời gian, bộ nhớ đệm,
  hợp nhất truy cập bộ nhớ (coalescing), xung đột bank (bank conflict) hay occupancy (mức chiếm
  dụng). "Thuộc tính thiết bị" của nó là tổng hợp.
- Qua được trên trình giả lập không chứng minh một kernel chạy được trên GPU.

### Biến thể phát hiện race condition

Vì `emu` không bao giờ ngắt một luồng giữa hai điểm đồng bộ hóa, nó không thể thấy một atomic bị
thiếu: `bins[b]++` từ một nghìn luồng ở đó luôn "đúng". Biến thể thứ hai, backend `race`, tồn tại
cho việc đó:

```sh
python3 labs/pmpp.py run <lab> --backend race   # only labs that opted in (see "Adding a lab")
```

- Các luồng của một khối, và nhiều khối, chạy thực sự đồng thời dưới dạng luồng của hệ điều hành,
  và bản dựng dùng ThreadSanitizer. Hai luồng giả lập chạm vào cùng vùng nhớ mà không đồng bộ hóa
  sẽ bị báo cáo kèm cả hai dòng mã nguồn, ngay cả khi lần chạy đó tình cờ cho ra đúng số. Một báo
  cáo làm trường hợp kiểm thử đang chạy lúc nó xuất hiện thất bại.
- `__syncthreads`, `__syncwarp`, grid `sync()` là rào chắn thật và được tính là đồng bộ hóa.
  Atomic là atomic thật. Warp vote và shuffle là điểm hẹn thật nhưng, như trên GPU, chúng **không**
  sắp thứ tự các lần truy cập bộ nhớ.
- Nó không lặp lại được (hệ điều hành lập lịch các luồng), nó không có kiểm tra biên
  (ThreadSanitizer và AddressSanitizer không thể kết hợp), và một lần chạy sạch không chứng minh
  là không có race: chỉ những lịch chạy đã xảy ra được xem xét, và ThreadSanitizer báo mỗi cặp dòng
  mã nguồn một lần mỗi lần chạy. Hai khối mà trình giả lập tình cờ chạy nối tiếp nhau trên cùng
  các luồng hệ điều hành chỉ được kiểm tra với nhau cho các chỉ số luồng khác nhau.
- Nó được bật khi ThreadSanitizer của trình biên dịch hoạt động trên máy
  (`python3 labs/pmpp.py doctor` nói lý do nếu không). Nó chậm: hãy giữ kích thước bài toán của
  nó nhỏ như của `emu`.

### Những gì được và không được mô phỏng

Chú thích đầu tệp của `common/include/pmpp/emu/cuda_emu.h` là tài liệu tham chiếu; đây là cùng
danh sách đó, ngắn hơn. "Dừng kernel" nghĩa là: lần khởi chạy kết thúc, `cudaGetLastError()` trả về
`cudaErrorLaunchFailure`, kiểm thử thất bại với một thông báo nêu các dòng mã nguồn. Không có gì bị treo.

| Tính năng | Được mô phỏng | Không được mô phỏng |
|---|---|---|
| Luồng, khối, lưới | chỉ số, kích thước, biến cục bộ của từng luồng; tối đa 1024 luồng mỗi khối và giới hạn theo từng chiều của CUDA (khối 1024 x 1024 x 64, lưới (2^31 - 1) x 65535 x 65535, Programming Guide Table 33): một lần khởi chạy vượt quá sẽ bị từ chối với `cudaErrorInvalidConfiguration` và một thông báo | mọi thứ liên quan đến thời gian |
| `__shared__`, `extern __shared__` | một bản sao cho mỗi khối, được điền `0xCD` cho mọi khối; kích thước động lấy từ lần khởi chạy (48 KiB, 96 KiB sau `cudaFuncSetAttribute`) | giới hạn của bộ nhớ chia sẻ khai báo tĩnh, bố cục, xung đột bank |
| `__syncthreads` và `_count/_and/_or` | rào chắn trên các luồng còn sống của một khối; một rào chắn không thể đạt tới sẽ dừng kernel | các rào chắn trong mã phân kỳ tình cờ ghép cặp được coi là một |
| Warp | `warpSize` là 32; warp *w* là các luồng 32*w*..32*w*+31 theo thứ tự tuyến tính; warp cuối có thể không đầy | thực thi đồng bộ từng bước, phân kỳ, lập lịch |
| `__shfl*_sync`, `__ballot_sync`, `__any_sync`, `__all_sync`, `__match_*_sync`, `__syncwarp` | một điểm hẹn của các lane chưa thoát được nêu trong mask, tại cùng một dòng mã nguồn; `width`; hành vi ngoài phạm vi đã được tài liệu hóa. Dừng kernel khi mask không chứa lane gọi, width sai, hoặc một lane được nêu đang chờ ở chỗ khác. Đọc từ một lane không tham gia trả về các byte `0xCD` và in một ghi chú | toán hạng half và bfloat16; một lane được nêu mà sau đó thoát mà không gọi được coi là đã thoát |
| `__activemask`, `coalesced_threads()` | các lane của warp đến cùng vị trí lời gọi trước khi có lane nào đi tiếp | hội tụ thật: GPU có thể báo ít lane hơn |
| `__popc`, `__clz`, `__ffs`, `__brev` và các dạng `ll` | các phép toán bit | (chữ ký unverified, xem bên dưới) |
| `__dp4a` (`int`, `unsigned`, `char4`, `uchar4`), `__byte_perm`, `__umulhi`, `__mulhi`, `__umul64hi`, `__float_as_uint` và các phép ép kiểu bit khác | các overload mà CUDA Math API tài liệu hóa; `__dp4a` mở rộng dấu cho byte ở các dạng có dấu và tràn vòng tổng 32-bit của nó | việc HIP không có `__dp4a` (x2-facts B1): mã gọi nó qua được ở đây và trong kiểm tra front-end, nhưng không dựng được cho HIP |
| `expf`, `exp2f`, `logf`, `log2f`, `sqrtf`, `rsqrtf`, `powf`, `fmaf`, `tanhf`, `erff`, `fminf`, `fmaxf`, `fabsf`, `copysignf`; `__expf`, `__logf`, `__powf`, `__sinf`, `__cosf`, `__fdividef`, `__saturatef` | các giá trị, lấy từ `<cmath>` của host | giới hạn sai số của NVIDIA, sai số xấp xỉ của các intrinsic nhanh, flush-to-zero; các intrinsic `_rn` không được cung cấp |
| `__half`, `__half2` (`#include <cuda_fp16.h>`) | binary16 chính xác từng bit: chuyển đổi làm tròn về số chẵn gần nhất và `__hadd/__hsub/__hmul/__hdiv` (cùng các dạng `2`, các toán tử), `__hfma`/`__hfma2` làm tròn một lần, `__low2float`, `__floats2half2_rn` và các chuyển đổi khác | payload của NaN, các biến thể `_sat`/`_relu`/làm tròn có hướng, shuffle và atomic cho half, `__nv_bfloat16`, tốc độ. Header của HIP là `<hip/hip_fp16.h>` |
| `atomicAdd/Sub/Exch/Min/Max/Inc/Dec/CAS/And/Or/Xor` | các overload mà NVIDIA tài liệu hóa, trả về giá trị cũ | các dạng half, bfloat16, `float2/float4` và 128-bit |
| `cuda::atomic_ref`, `cuda::atomic` (`<cuda/atomic>`) | load, store, exchange, compare_exchange, fetch_add/sub/and/or/xor/min/max trên các kiểu 1, 2, 4, 8 byte | thread scope và memory order không có tác dụng; wait/notify |
| `__constant__`, `cudaMemcpyToSymbol/FromSymbol` | một biến toàn cục mà host điền qua các lời gọi symbol, có kiểm tra kích thước | việc ép chỉ đọc trong kernel, kích thước của bộ nhớ hằng |
| `char1..4`, `uchar1..4`, `short1..4`, `ushort1..4`, `float2..4`, `int2..4`, `uint2..4`, `double2..4` (kích thước và căn lề của Table 49), `make_*`, `__forceinline__`, `__noinline__`, `__restrict__`, `#pragma unroll` | các kiểu và hàm trợ giúp; các qualifier biên dịch được | inline và unroll không có ý nghĩa gì ở đây (clang chấp nhận `#pragma unroll`; gcc bỏ qua nó, kèm cảnh báo dưới `-Wall`) |
| `<cooperative_groups.h>` | `this_thread_block()`, `this_grid()` với `sync()` toàn lưới dưới `cudaLaunchCooperativeKernel` (mọi khối cùng sống một lúc; lưới x khối tối đa 1024 luồng, nếu không thì `cudaErrorCooperativeLaunchTooLarge`), `coalesced_threads()`, `tiled_partition<1..32>(block)` | cluster, tile lớn hơn, `binary_partition`, `labeled_partition`, `reduce`, `scan`, `memcpy_async`, kernel template trong một lần khởi chạy cooperative |
| Stream (có độ ưu tiên), event, `cudaMemcpyAsync`, `cudaMallocHost`, `cudaMallocManaged`, `cudaFuncSetAttribute`, `cudaFuncGetAttributes`, `cudaGetDevice`, occupancy | các lời gọi tồn tại với prototype đã được tài liệu hóa; mọi thứ đều đồng bộ | bất đồng bộ và chồng lấp; độ ưu tiên (phạm vi là 0 đến 0, như một thiết bị không có độ ưu tiên); `cudaEventElapsedTime` là thời gian thực trên host, `cudaOccupancyMaxActiveBlocksPerMultiprocessor` là một con số tổng hợp, và `cudaFuncGetAttributes` không báo thanh ghi nào: không điều nào trong đó nói lên gì về một GPU |

Những gì trình giả lập cần từ một tệp `.cu` (tất cả cũng đều là CUDA hợp lệ):

- `#include <cuda/atomic>`, `#include <cooperative_groups.h>` và `#include <cuda_fp16.h>` ở nơi bạn
  dùng chúng, như với nvcc.
- Khai báo `__shared__` bên trong một hàm, viết dưới dạng khai báo thông thường
  (`__shared__ float a[N][M];`, `__shared__ float t[(TILE + 2) * (TILE + 2)];`), không nấp sau macro.
  Biểu thức nào cũng có thể làm kích thước mảng; nhưng không được có bộ khởi tạo.
- Một kernel khởi chạy bằng `cudaLaunchCooperativeKernel` phải là một hàm `__global__` thông thường
  (không phải template) được định nghĩa trong một tệp `.cu`.
- Kernel không được overload: tên đứng trước `<<<` phải xác định đúng một hàm.

Chưa được đối chiếu với tài liệu của NVIDIA (viết từ trí nhớ về các header của toolkit):
prototype của `__popc`, `__popcll`, `__clz`, `__clzll`, `__ffs`, `__ffsll`, `__brev`,
`__brevll`, và căn lề 4 byte của `__half2`. Mọi thứ khác trong
`common/include/pmpp/check/cuda_stub.h` đều nêu nguồn của nó (các dữ kiện thêm vào ngày 2026-11-21
nằm ở `notes/today/infra-facts.md`, nhóm E).

### Tự kiểm thử

`labs/selftest/` chứa các kiểm thử của chính trình giả lập: các tệp `.cu` nhỏ viết bằng CUDA thuần,
chạy trên `emu` và `race` (và trên `cuda`, nơi chúng đối chiếu mô hình của trình giả lập với
thiết bị thật). `./verify.sh` chạy chúng và liệt kê chúng trong báo cáo của nó;
`python3 labs/pmpp.py selftest [name] [--backend race]` chỉ chạy riêng chúng và không ghi gì.
`race_hist_plain` và `race_negatives` là các data race cài sẵn: trên `race`, chúng chỉ qua nếu
ThreadSanitizer báo cáo chúng.

## Thêm một lab

Một dòng trong `CMakeLists.txt` của lab (`pmpp_add_lab(...)`, xem `cmake/Lab.cmake`) và một
`main.cpp` xây trên `common/include/pmpp/harness.hpp`. Mã được cung cấp mà lab biên dịch nằm trong
`SHARED_CU` và `SHARED_MM`; `SHARED_DIR ../shared-g11` nêu thư mục chứa nó
(khi đó tên trong `SHARED_CU`/`SHARED_MM` có thể chỉ là tên tệp), thêm nó vào đường dẫn include
và truyền nó cho chương trình điều khiển dưới dạng `PMPP_SHARED_DIR` (và `PMPP_METAL_SHARED_DIR`
trên Metal, vốn nếu không sẽ trỏ tới `../shared`). `LIBS` liên kết một thư viện trên mọi backend và
`LIBS_CUDA`, `LIBS_HIP`, `LIBS_METAL`, `LIBS_EMU`, `LIBS_RACE` trên một backend (`CUDA::cublas`,
`roc::hipblas`, `"-framework MetalPerformanceShaders"`); khi một CMake target được nêu không tồn
tại trên máy, bản dựng lab cho backend đó bị bỏ qua và `run` nói điều đó. Thêm `RACE` vào dòng đó
để dựng lab cho cả trình giả lập phát hiện race condition (hãy làm vậy cho mọi lab về atomic hoặc
các cập nhật không đồng bộ khác), và cho `race` cùng kích thước bài toán nhỏ như `emu`:
`pmpp::backend()` trả về `"race"`, và `PMPP_BACKEND_RACE` được định nghĩa trong `main.cpp`. Một lab
chỉ hoàn thành khi lời giải của nó qua trên Metal và trình giả lập, starter chưa đụng tới của nó
dựng được và thất bại, và mỗi lỗi cài sẵn đã được chứng minh là tự nó gây thất bại.

### Sửa một starter

`work/` thuộc về người đọc và không bao giờ bị ghi đè, nên một starter được sửa thông qua lịch sử
starter: `labs/starter-history.json` chứa sha256 của mọi phiên bản mà các tệp `starter/` của mỗi
lab từng có (các phiên bản trước 2026-11-21 được đọc từ lịch sử git). Hãy sửa `starter/`, rồi chạy
`python3 labs/pmpp.py starters --record`, lệnh này nối thêm phiên bản mới; đừng bao giờ xóa một mục.
Một `work/` bằng bất kỳ phiên bản đã ghi nào vẫn là `untouched` trong báo cáo và ứng dụng (nó vẫn
là một starter, và phải thất bại), và `run` cùng `doctor` báo cho người đọc biết rằng `reset` mang
về bản đã sửa. `python3 labs/pmpp.py starters` thất bại khi một starter hiện tại chưa được ghi, và
`python3 -m unittest discover -s labs/tests` cũng vậy, lệnh này còn kiểm thử các quy tắc so khớp.
