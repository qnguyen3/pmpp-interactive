> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab X1.3: mốc M5, một bộ lấy mẫu GPU theo batch

Không có trong sách: lấy mẫu là của khóa học này (PLAN.md, module X1). Mọi giai đoạn trừ ba là kernel
của một lab trước: các mức phạt của lab 9.3, softmax và argmax của lab 10.3, phép chọn top-k của lab
X1.1, phép sắp xếp của lab 14.3, mặt nạ top-p của lab 11.6. Bạn viết những gì chưa lab nào có: một số
ngẫu nhiên dựa trên bộ đếm cho mỗi hàng, mức phạt lặp lại, lần rút cuối cùng, và việc ghép nối. Một mục
đứng riêng như (A4) hay (E2) dưới đây nằm trong `notes/today/x1-facts.md`; các mục khác ghi rõ tệp của chúng.

```cpp
size_t sampleWorkspaceBytes(int rows, int vocab, int topK);
void launchSampleRows(const float* logits, const unsigned int* counts,
                      const unsigned long long* seeds, const unsigned long long* streams,
                      const unsigned long long* steps, int* tokens, float* uOut, void* workspace,
                      int rows, int vocab, int topK, float topP, float minP, float temperature,
                      float repeatPenalty, float frequencyPenalty, float presencePenalty);
```

Vào là logits `[rows x vocab]` và số đếm token, ra là một id token cho mỗi hàng. Mỗi hàng mang theo
seed, stream và step của riêng nó, và token là một hàm của những giá trị đó và của dữ liệu: chạy lại
và bạn nhận được cùng token, trên mọi backend, với mọi lần rút không nằm trong sai số làm tròn của một
ranh giới (xem bên dưới). Mốc M10 (lab X3.2) gọi nó trên GPU khi lời giải của lab này có mặt.

- Sửa `work/kernel.cu` (CUDA; cũng được build cho HIP và các trình giả lập): `philox4x32_10`,
  `philoxKatKernel`, `uniformFromBits`, `applyRepetition`, `drawRows`, `sampleWorkspaceBytes`,
  `launchSampleRows`. Các hàm khởi chạy `launchPhiloxKat`, `launchApplyRepetition`, `launchDrawRows`
  và `launchDrawRowsU` (lối vào cho kiểm thử: u từ một buffer) đã được viết sẵn cho bạn.
- Sửa `work/kernel.metal` (`philox4x32_10`, `philox_kat`, `uniform_from_bits`,
  `apply_repetition`, `draw_rows`) và `work/launch.mm` (`encodePhiloxKat`,
  `encodeApplyRepetition`, `encodeDrawRows`, `encodeSampleRows`).
- Có sẵn trong các tệp đó: bước sao chép workspace (`copyRows`, `copy_rows`, `encodeCopyRows`) và các
  hàm trợ giúp warp.
- Có sẵn và không được sửa, trong `labs/x1/shared-g11/`: các bản sao của lab 9.3 (`penalties_g11.cu`),
  lab 10.3 (`softmax_g11.cu`), lab 11.6 (`topp_g11.cu`), lab 14.3 (`sortrows_g11.cu`) và lab X1.1
  (`topk_g11.cu`, `.metal`, `_launch.mm`), mỗi bản có tên gốc tệp riêng và các hàm trợ giúp là static
  để chúng có thể được liên kết với nhau; `app/src/chapters/x1/copies-g11.test.ts` kiểm tra rằng không
  có gì khác lệch khỏi bản gốc. Và `minp_g11.cu`, `.metal`, `_launch.mm`: min-p dưới dạng một phép đếm
  trên hàng đã sắp xếp, và số ứng viên còn lại (xem bên dưới). Trên Metal, chính các tệp
  `solution/kernel.metal` của các chương trước được nạp theo đường dẫn và các tệp `launch.mm` của chúng
  được liên kết vào; `sampler_metal.h` khai báo các encoder của chúng, các pipeline và workspace.
- Kiểm thử: `python3 labs/pmpp.py run x1/lab3`. Bắt đầu với `--backend emu` (vài phút).
- Race: `python3 labs/pmpp.py run x1/lab3 --backend race`.
- Đo: `python3 labs/pmpp.py run x1/lab3 --bench`, sau khi có các dự đoán bên dưới.

## Chuỗi xử lý

Thứ tự mặc định của llama.cpp (D1): các mức phạt, top-k, top-p, min-p, rồi nhiệt độ, rồi lần rút. Việc
cắt bớt diễn ra trước nhiệt độ; Hugging Face transformers làm theo chiều ngược lại (D4), và mlx-lm chạy
top-p trước top-k (D8). M5 theo llama.cpp.

| Giai đoạn | Kernel | Ghi ra, cho mỗi hàng |
|---|---|---|
| sao chép | `copyRows` (có sẵn) | các logits, `vocab` số float: logits của bên gọi là chỉ đọc |
| lặp lại | của bạn | các token đã gặp: một logit `<= 0` nhân với mức phạt, một logit dương chia cho nó |
| tần suất, hiện diện | lab 9.3 `launchApplyPenalties` | `logit -= count * f + (count > 0) * p` |
| tham lam (`temperature <= 0`) | lab 10.3 `launchArgmaxRows` | token: logit lớn nhất, id thấp nhất khi bằng nhau; rồi dừng |
| top-k | lab X1.1 `launchTopKRows` | `K` cặp (cột, logit), theo thứ tự cột, khi bằng nhau thì giữ các cột thấp nhất |
| sắp xếp | lab 14.3 `launchSortRowsDesc` | `K` logits, lớn nhất trước, ổn định, và các vị trí của chúng |
| softmax | lab 10.3 `launchSoftmaxRows`, tỉ lệ 1 | `K` xác suất |
| top-p | lab 11.6 `launchTopPMask` | `kept`: giữ vị trí `i` khi `i == 0`, `p >= 1`, hoặc khối xác suất trước `i` dưới `p` |
| min-p, kết hợp | `launchMinPCombine` (có sẵn) | `n = max(1, min(kept, countMinP))`, trong đó `countMinP` đếm các logits đã sắp xếp ít nhất bằng logit đầu cộng `ln(minP)`; `minP <= 0` tắt nó |
| lần rút | của bạn | token |

`K` là `topK`, hoặc `vocab` khi `topK <= 0` hoặc `topK >= vocab` (llama.cpp: "<= 0 để dùng kích thước
vocab", D1): con đường chậm, một phép chọn và một phép sắp xếp trên cả hàng. Workspace chứa mọi giá trị
trung gian: các logits đã phạt (`rows x vocab` số float), các cột và logits của phép chọn cùng vùng nháp
của nó (`4 x rows` uint), các logits đã sắp xếp và vị trí của chúng, vùng nháp của phép sắp xếp
(`4 x rows x K` uint), các xác suất, khối xác suất và cờ giữ lại của lab 11.6 (`rows x K` byte) cùng các
số đếm, và `n`. `sampleWorkspaceBytes` cho biết đó là bao nhiêu byte; trên Metal, nơi các encoder của các
lab trước gắn mọi buffer ở offset 0, `sampler_metal.h` tạo một buffer cho mỗi giá trị trung gian thay vào
đó.

Ở mọi giai đoạn, khi bằng nhau thì id token thấp nhất thắng: phép chọn giữ các cột thấp nhất và trao
chúng đi theo thứ tự cột, phép sắp xếp của lab 14.3 là ổn định, argmax ưu tiên chỉ số thấp nhất.

## Những phần bạn viết

**Philox4x32-10** (A4): mười vòng trên một bộ đếm 128 bit và một khóa 64 bit. Một vòng nhân từ 0 của bộ
đếm với `0xD2511F53` và từ 2 với `0xCD9E8D57`, mỗi phép là một tích đầy đủ 32 x 32 ra 64 bit, và tạo
`(hi1 ^ w1 ^ k0, lo1, hi0 ^ w3 ^ k1, lo0)`; giữa các vòng, khóa tăng thêm `0x9E3779B9` và `0xBB67AE85`.
Hãy viết tích này dưới dạng phép nhân `unsigned long long` (Metal: `ulong`): `__umulhi` có trên CUDA và
HIP (A5, B3), và trình giả lập (emulator) cùng stub front-end của khóa học cũng chấp nhận nó, nhưng dạng
tích không cần intrinsic nào và là cùng một mã nguồn trên Metal. Mã của Random123 được phân phối theo một
thông báo kiểu BSD (Copyright 2010-2011, D. E. Shaw Research), được chép lại trong lời giải; các header
của cuRAND theo giấy phép của NVIDIA và không được dùng. MSL hoàn toàn không có hàm sinh số ngẫu nhiên
nào (C1).

**Cách ánh xạ của M5** (của chúng tôi): counter = (step thấp, step cao, stream thấp, stream cao), key =
(seed thấp, seed cao), từ 0. Theo cách chúng tôi đọc header của cuRAND, đó là lần gọi `curand()` đầu
tiên sau `curand_init(seed, stream, 4 * step, &s)` (A3, unverified trên thiết bị).

**u** = `(w >> 8) * 2^-24`. Một số nguyên dưới 2^24 chuyển sang float chính xác và một lũy thừa của hai
co giãn chính xác, nên u là cùng một số float trên mọi backend dưới mọi chế độ toán, trong
`[0, 1 - 2^-24]`, không bao giờ bằng 1 (E2). Các phép chuyển riêng của cuRAND và rocRAND khác nhau ở
khoảng 0.4 % đầu vào (B2), đó là một lý do M5 tự chuyển các bit.

**Lặp lại** (D2): với một token có count > 0, một logit `<= 0` được nhân với mức phạt và một logit dương
được chia. Chú thích của llama.cpp: "Bài báo học thuật mô tả kỹ thuật này thực ra chỉ chia, nhưng điều đó
sẽ khiến các token có logit âm trở nên có khả năng cao hơn, rõ ràng là sai."

**Lần rút**, một khối cho mỗi hàng như kernel lấy mẫu của FlashInfer (A6): trọng số
`w_i = exp((l_i - l_0) * invT)` trên `n` token còn lại (`l_0` là lớn nhất, `invT = 1 / T` được tính một
lần trên host); tổng bằng một phép rút gọn (reduction) khối; mục tiêu `u * total`; rồi duyệt hàng theo
các đoạn `blockDim.x * 4` trọng số, bốn trọng số liên tiếp cho mỗi luồng, với tổng của các đoạn trước
được mang theo. Một đoạn mà tổng của nó không thể vượt qua mục tiêu thì bị bỏ qua sau phép scan của nó.
Trong đoạn vượt qua, bên thắng là vị trí đầu tiên có trọng số dương mà tổng bao gồm của nó lớn hơn hẳn
mục tiêu: lane đầu tiên như vậy của một warp tìm bằng ballot, warp thấp nhất có lane đó. Nếu việc làm
tròn không để lại vị trí nào như vậy, trọng số dương cuối cùng thắng (FlashInfer có cùng phương án dự
phòng, A6). Với u trong `[0, 1)` và "lớn hơn hẳn", một trọng số bằng không không bao giờ có thể thắng (A2).

## Những lần rút nào phải khớp

Bản tham chiếu CPU trong `main.cpp` lấy mẫu mọi hàng bằng độ chính xác double, sau khi các mức phạt được
áp dụng bằng float theo thứ tự của llama.cpp (nên một token chưa gặp giữ nguyên logit của nó đến từng
bit). Một device tính cùng chuỗi đó bằng float, và chế độ toán mặc định của Metal còn có thể gộp phép
tính, sắp xếp lại phép cộng và làm tròn về không (C5, ch05-facts C7), nên một device có thể rơi vào phía
bên kia của một ranh giới mà CPU chỉ vượt qua sát nút. Vì vậy bản tham chiếu ghi lại, cho mọi quyết định,
nó còn cách việc bị lật bao xa, và phép kiểm thử chỉ so sánh token cho những lần rút **dứt khoát**. Các
dải dưới đây là của chúng tôi; chúng được cố ý đặt rộng, để một kernel đúng không bao giờ thất bại, và
một lần rút nằm trong dải được đếm và in ra, không bao giờ bị đánh trượt. Hơn một nửa số lần rút của mỗi
trường hợp phải dứt khoát, nếu không phép kiểm thử báo cáo một vấn đề với chính dữ liệu của nó.

- Một logit đã phạt: giá trị của token chưa gặp là chính xác ở mọi nơi. Giá trị của token đã gặp đi qua
  một phép nhân hoặc chia và một phép trừ mà một device có thể làm tròn khác hoặc gộp lại: dải
  `e = 8 ulp` của độ lớn lớn nhất có liên quan.
- Top-k: chưa quyết định nếu một token được giữ nào đó trừ đi dải của nó nhỏ hơn hoặc bằng một token bị
  bỏ nào đó cộng dải của nó (một trong hai đã bị phạt). Các giá trị chưa phạt bằng nhau không phải chưa
  quyết định: id thấp nhất thắng trên mọi backend.
- Top-p và min-p: mỗi cái là một phép đếm tiền tố. Một vị trí có khối xác suất trước nó cách `p` trong
  phạm vi `2^-24 x (512 + K / 256)` (softmax float, một tổng rút gọn theo khối, phép scan theo đoạn của
  lab 11.6 với một tiền tố được mang theo), hoặc có logit cách ngưỡng min-p trong phạm vi `2 ulp + e`,
  có thể đi theo hướng nào cũng được; lần rút chỉ chưa quyết định nếu điều đó có thể thay đổi `n`.
- Thứ tự: hai token còn lại kề nhau có dải chồng lên nhau có thể được sắp theo chiều kia; một lần rút
  rơi vào một trong hai là chưa quyết định.
- Lần rút: chưa quyết định nếu `u x total` cách một ranh giới của các trọng số tích lũy trong phạm vi
  `total x (2^-14 + n x 2^-22) + 2 invT (e_0 + max e)`. Phần tương đối bao trùm `exp` float và phép scan
  theo đoạn; phần còn lại bao trùm các logits đã phạt.

Các phép kiểm thử in ra số lần rút rơi vào trong một dải, cùng lý do cho lần đầu tiên.

## Những gì các phép kiểm thử kiểm tra

1. **Đáp án đã biết của Philox**, trên device: ba vector 10 vòng của Random123 (A4): bộ đếm và khóa đều
   bằng không cho `6627e8d5 e169c58d bc57ac4c 9b00dbd8`; đều bằng một cho
   `408f276d 41c83b0e a20bc7c6 6d5451fd`; các chữ số của pi cho
   `d16cfe09 94fdcceb 5001e420 24126ea1`. Chương trình điều khiển (driver) cũng kiểm tra Philox trên host của chính nó với chúng.
2. **u từng bit** cho 4,096 bộ ba (seed, stream, step) qua `drawRows` với một ứng viên mỗi hàng (64 trên
   `emu`, 32 trên `race`), bắt đầu với seed 0, stream 0, step 0, mà từ 0 phải là `0x6627e8d5`. Có cả các
   seed, stream và step lớn hơn 2^32.
3. **Riêng lần rút** qua `launchDrawRowsU` với u cài sẵn: các trọng số bằng nhau (`exp(0) = 1`, nên mọi
   tổng là một số nguyên chính xác) với `u = m x 2^-24` và `u x n` chính xác, phải cho ra vị trí
   `floor(u x n)`, kể cả các đường nối của những đoạn đầu (vị trí 1023 và 1024 của một hàng 2,048 trọng
   số, 2047 và 2048 của 4,096); các trọng số bằng không (`exp(-1e4)` tràn dưới thành 0) không bao giờ được
   thắng, ở u = 0, ở mục tiêu bằng một tổng, và ở `u = 1 - 2^-24`; một token còn lại duy nhất; và
   `u = 1`, giá trị không thể đến từ phép chuyển nhưng đặt mục tiêu ở tổng, như việc làm tròn có thể gây
   ra, để phương án dự phòng phải chọn trọng số dương cuối cùng. Các ứng viên sau `n` mang một trọng số
   khổng lồ mà kernel không bao giờ được phép đọc, và các vị trí được ánh xạ sang id qua một hoán vị.
4. **Lặp lại** trên 2 hàng gồm 1,000 logits (300 trên `emu`) ở cả hai dấu, các số không chính xác ở cả
   hai dấu và số đếm từ 0 đến 3: mức phạt 1.1 và 2.0 sai trong phạm vi 2 ulp so với quy tắc của llama.cpp
   (một số không vẫn là không), mức phạt 1 và các token chưa gặp giống từng bit.
5. **Toàn bộ bộ lấy mẫu** so với bản tham chiếu CPU: rows 1, 3 và 8 nhân vocab 1,000, 4,099, 32,000 và
   151,936; thiết lập là các giá trị mặc định của llama.cpp (top-k 40, top-p 0.95, min-p 0.05, T 0.8) với
   các mức phạt 1.1, 0.1, 0.1, rồi lần lượt từng thiết lập top-k 1, top-k 0 (cả hàng), top-p 1, min-p 0,
   T 0 (tham lam, id thấp nhất), T 2, và không có mức phạt; 64 seed x 4 step cho mỗi trường hợp, một
   stream cho mỗi hàng. Các logits có một cặp bằng nhau cài sẵn gồm hai token đầu chưa gặp, các số không ở
   cả hai dấu trong số các token đã gặp, và ở các hàng lẻ có hai token chưa gặp bằng nhau ở giá trị lớn
   nhất. Các lần rút dứt khoát phải khớp chính xác; u phải khớp từng bit (và không được ghi khi T <= 0);
   mọi token phải là một id trong bộ từ vựng; 64 phần tử canh gác sau `tokens` và `uOut` và (CUDA, HIP,
   các trình giả lập) sau `sampleWorkspaceBytes` không bị chạm vào; logits và số đếm không đổi. Trên
   `emu`: rows 1 và 3 nhân vocab 1,000 và 4,099, top-k nhiều nhất 40 (không có trường hợp cả hàng), mỗi
   (kích thước, thiết lập) thứ tư, 1 seed x 1 step. Trên `race`: rows 2 nhân vocab 300 và 1,000, mọi
   thiết lập, 2 seed x 2 step. Vòng lặp mô phỏng bên dưới dùng bộ từ vựng 1,000 (300 trên `emu`).
6. **Một vòng lặp decode mô phỏng**: 4 hàng, 16 step (trên `emu` là 2 hàng, 4 step), logits từ một bảng
   cố định, số đếm được giữ trên device bởi `appendTokens` của lab 9.3 sau mỗi step, ở các giá trị mặc
   định của llama.cpp có mức phạt. CPU trước tiên tìm một seed mà mọi lần rút đều dứt khoát dọc theo quỹ
   đạo của chính nó; các chuỗi token của bạn phải bằng nó.
7. **Digest của token**: các hash FNV-1a của những lần rút dứt khoát của một trường hợp cố định ở giá trị
   mặc định, các hàng 1,000. Trên GPU, trường hợp đầy đủ là 8 hàng x 16 seed x 8 step (1,024 lần rút),
   được ghi vào JSON của lần chạy dưới tên `tokens_digest` cùng `decisive_draws` và `draws`. Phần lõi của
   nó, 2 hàng x 2 seed x 2 step đầu tiên (8 lần rút), được hash riêng thành `core_digest` cùng
   `core_draws`; các trình giả lập, quá chậm cho trường hợp đầy đủ, chỉ chạy và ghi phần lõi. Cùng một
   digest trên hai backend nghĩa là cùng các token trên những lần rút đó; phần lõi so sánh mọi backend,
   digest đầy đủ so sánh các GPU.

## Trước khi bạn benchmark

Hãy ghi lại hai dự đoán (bài học hỏi chúng dưới tên `x1/labs/p-lab3-cpu` và `x1/labs/p-lab3-batch`):

1. Ở batch 1 và 151,936 token với các giá trị mặc định của llama.cpp, thời gian mỗi bước lấy mẫu của
   GPU chia cho thời gian của một luồng CPU là bao nhiêu? Dưới 1 nghĩa là GPU thắng.
2. Một bước ở batch 64 sẽ lâu hơn ở batch 1 bao nhiêu lần?

Hãy lập luận từ việc mỗi hàng một khối (chương 10, mục 10.10: ở batch 1 chỉ một multiprocessor làm việc
và phần còn lại chờ) và từ những giai đoạn nào đọc cả bộ từ vựng: bước sao chép và các mức phạt đọc và
ghi mọi logit, phép chọn đọc hàng tới năm lần; mọi thứ sau top-k chỉ chạm 40 giá trị mỗi hàng.

Benchmark chạy rows 1, 8 và 64 nhân vocab 32,000 và 151,936 ở các giá trị mặc định của llama.cpp với các
mức phạt bật, và con đường cả hàng (top-k 0) ở rows 1 và 8 nhân 151,936. Nó báo cáo:

- số ms mỗi bước lấy mẫu cho `launchSampleRows` (trên Metal là `encodeSampleRows`) và số token mỗi giây
  (số hàng chia cho thời gian đó);
- từng giai đoạn được đo riêng: các mức phạt (sao chép, lặp lại, tần suất và hiện diện), phép chọn
  top-k, sắp xếp k, softmax, top-p, min-p và kết hợp, lần rút;
- một luồng CPU chạy cùng chuỗi bằng float (llama.cpp mặc định lấy mẫu trên host, D1; top-k bằng
  `std::partial_sort`, ch14-facts D3), và tỉ lệ GPU / CPU.

Mỗi thời gian là tốt nhất của 5 lần chạy, mỗi lần lặp lại lời gọi cho đến khi kéo dài ít nhất 2 ms; mọi
kết quả đo được đều được kiểm tra với bản tham chiếu CPU trước (các lần rút dứt khoát), và một kích thước
lớn hơn một phần tư bộ nhớ device được báo cáo sẽ bị bỏ qua. Thời gian trên Metal là thời gian GPU của
command buffer (`GPUEndTime - GPUStartTime`); thời gian CUDA và HIP là thời gian thực trên host quanh các
lần khởi chạy và một lần đồng bộ; thời gian CPU đo bằng đồng hồ ổn định trên host. Các con số là phép đo
trên máy của bạn, không phải mục tiêu.

## Tính khả chuyển

- HIP: `kernel.cu` được viết để build qua `pmpp/vendors/hip.h`. Mặt nạ lane là 64 bit trên mọi đích AMD
  (infra-facts B3), và find-first-set khớp với nó (`__ffsll`). Dạng tích 64 bit của Philox không cần
  intrinsic nào. hipRAND tồn tại với các tên của cuRAND, nhưng `hiprand_uniform` của nó trên AMD chuyển
  bit khác với `curand_uniform` (B2): với Philox và phép chuyển riêng của M5, u là cùng một đoạn mã trên
  mọi backend, và phép kiểm thử đáp án đã biết kiểm tra điều đó. Chưa được biên dịch hay chạy trên phần
  cứng AMD ở đây: unverified.
- Metal: Philox trong MSL với các tích `ulong` (Metal 2.2 trở lên, ch14-facts C3); `mulhi` cũng cho được
  nửa cao (C4). `simd_ballot` trả về một phiếu 64 bit mà các bit trên độ rộng SIMD group không xác định
  (ch12-facts C2), nên lần rút che nó theo các lane thực có trước khi gọi `ctz`. Các vị trí đến từ
  `sg * width + lane` (ch04-facts C2). `launch.mm` gắn một buffer thay thế ở chỗ `uIn` hoặc `uOut` vắng
  mặt và truyền cờ thay vào; workspace là một buffer cho mỗi giá trị trung gian (ở trên), nên phần canh
  gác phía CUDA sau workspace không có tương ứng trên Metal.
- Bước kiểm tra front-end CUDA cần thư mục include shim mà `labs/pmpp.py` truyền vào
  (`-I labs/common/include/pmpp/emu/shim`) cho các bản sao có include `<cuda/atomic>`.
