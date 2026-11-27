> Bản tiếng Việt của [README.md](README.md). Bản tiếng Anh là bản gốc; nếu hai bản khác nhau, hãy theo bản tiếng Anh.

# Lab 11.6: mặt nạ giữ lại cho top-p (mốc dự án M5, phần top-p)

Sách: scan loại trừ (mục 11.1, p. 253) được tính bằng scan theo hàng của lab 11.3, chính là scan
có gộp việc cho luồng (thread coarsening), phân ô thanh ghi của Figs. 11.12 và 11.13 (pp. 268 và
271) trên scan warp và scan khối của Figs. 11.8 và 11.9 (p. 264). Top-p sampling không có trong
sách: quy tắc, mặt nạ và phép đếm là của chúng tôi. Một mục đứng riêng như (D2) bên dưới nằm
trong `notes/today/ch11-facts.md`.

Top-p (nucleus) sampling giữ lại tập nhỏ nhất gồm các token có khả năng cao nhất mà tổng xác
suất đạt p, rồi lấy mẫu từ các token đó. Sau khi các xác suất được sắp xếp, tập đó là một tiền
tố của mỗi hàng, và tìm nó là một phép scan. Hãy viết một kernel theo hàng:

`launchTopPMask(probs, massBefore, keep, kept, rows, cols, p)`

- `probs`: `[rows x cols]` float32, thứ tự hàng (row-major), batch đứng đầu. Mỗi hàng là một đầu
  ra softmax được sắp xếp giảm dần và có tổng bằng 1 trong phạm vi làm tròn.
- `massBefore`: `[rows x cols]` float32. Khối lượng xác suất xếp hạng hoàn toàn trước mỗi token:
  một scan loại trừ của hàng.
- `keep`: `[rows x cols]` uint8, 1 = giữ, 0 = bỏ.
- `kept`: `[rows]` int32, số lượng số một trong mỗi hàng của `keep`.

Tệp:

- Sửa `work/kernel.cu` (`TopPMaskKernel`, `launchTopPMask`; CUDA, cũng được build cho HIP và các
  trình giả lập) và `work/kernel.metal` (`top_p_mask`) + `work/launch.mm` (`encodeTopPMask`).
  Các tên là cố định.
- Được cung cấp, không sửa, trong `labs/ch11/shared-g12/`: `topp_naive.cu` và `.metal` (một
  luồng mỗi hàng, vòng lặp của Fig. 11.1: mốc so sánh), `topp_copy.cu` và `.metal` (một phép
  chép với cùng 9 byte mỗi token: thước đo) và `topp_launch.mm`.
- Kiểm thử: `python3 labs/pmpp.py run 11.6`. Bắt đầu với `--backend emu`.
- Race condition: `python3 labs/pmpp.py run 11.6 --backend race`.
- Đo: `python3 labs/pmpp.py run 11.6 --bench`.

## Hợp đồng

Với mọi hàng và token i:

- `massBefore[i] = probs[0] + ... + probs[i-1]`, và `massBefore[0] = 0` chính xác.
- `keep[i] = 1` đúng khi `i == 0`, hoặc `p >= 1`, hoặc `massBefore[i] < p`.
- `kept = ` số lượng số một trong hàng của `keep`.

Vì sao có từng điều khoản:

- **Token vượt qua p được giữ lại.** Bộ lấy mẫu phía host của llama.cpp cộng từng token vào một
  tổng tích lũy và dừng khi tổng đạt p, giữ lại token đã làm tổng đạt mức đó; bộ lấy mẫu dạng
  đồ thị của nó làm tương tự với một tổng tích lũy và thêm một token; transformers và vLLM sắp
  xếp tăng dần và bỏ một token khi xác suất của nó cộng khối lượng xếp hạng dưới nó nhỏ hơn
  hoặc bằng 1 - p (D2, D4). Với các hàng có tổng bằng 1, tất cả đều giữ lại đúng những token
  có khối lượng đứng trước nhỏ hơn p (phép suy dẫn của chúng tôi, D2 và D4). Đó là scan loại
  trừ của mục 11.1, và sai lệch một vị trí giữa "exclusive < p" và "inclusive < p" chính là
  toàn bộ khác biệt giữa giữ và bỏ token vượt ngưỡng.
- **Token 0 luôn được giữ.** Khối lượng đứng trước nó là 0, nên `massBefore < p` giữ nó với mọi
  p > 0; điều khoản này cũng bao phủ p = 0. vLLM cũng giữ ít nhất một token bất kể p là bao
  nhiêu (D4).
- **p >= 1 giữ mọi token.** llama.cpp trả về sớm với `p >= 1.0f` (D2). Không có điều đó, khối
  lượng float đứng trước các token cuối có thể làm tròn thành 1 hoặc nhỉnh hơn một chút, và
  p = 1 sẽ bỏ những token mà người dùng yêu cầu "không top-p" mong được giữ.
- **Các token bằng nhau được phân định theo vị trí**, tức là theo phép sắp xếp: trong hai token
  có xác suất bằng nhau nằm ở hai phía của p, token được sắp trước được giữ. mlx-lm thay vào đó
  giữ mọi token bằng với token ở ranh giới (D3). Đó là một phương án khác được nêu ra, không
  phải quy tắc của lab này.

Mặt nạ giữ lại không nhất thiết là một tiền tố. Hai khối lượng cạnh nhau được tính dọc theo các
cây cộng khác nhau, nên bên trong dải dung sai bên dưới, một số 0 có thể đứng trước một số 1.
`kept` đếm số một, không phải độ dài của một tiền tố, và phép nén gọn (compaction) của chương 12
không giả định điều đó.

## Thiết kế gợi ý

Scan theo hàng đã sửa của lab 11.3 (mỗi hàng một khối 256 luồng, mỗi đoạn 4 phần tử cho mỗi luồng,
một tiền tố được mang sang), với phép so ngưỡng và phép đếm được gộp vào, để các xác suất
chỉ được đọc một lần:

- Khi khối lượng của mỗi token còn trong thanh ghi, quyết định giữ, ghi nó, và đếm nó.
- Sau vòng lặp đoạn, cộng các số đếm của các luồng bằng phép rút gọn (reduction) hai giai đoạn
  của chương 10, và để một luồng ghi `kept`.
- Trên Metal, `simd_prefix_exclusive_sum` và `simd_prefix_inclusive_sum` làm trong một lần gọi
  điều mà vòng lặp của Fig. 11.8 làm, và `simd_sum` cộng các số đếm (C1, `infra-facts.md` C2).
  Các mảng threadgroup phải được khai báo trong kernel (C2).

## Các bài kiểm thử kiểm tra gì

- Hình dạng: 1, 4 và 7 hàng nhân 1, 2, 31, 32, 33, 255, 256, 257, 1000, 4096, 32000, 50257 và
  128256 cột (ba giá trị cuối có kích thước bằng bộ từ vựng; của GPT-2 là 50,257, ch09-facts
  D7). Trình giả lập dùng 1 và 3 hàng và tới 4099 cột; backend race dùng 2 hàng và 1, 33, 257,
  1000 và 2100 cột (ba đoạn 1024, nên tiền tố được mang sang được thử). Các loại hàng khác nhau
  được trộn trong một batch, nên một kernel đọc nhầm hàng sẽ bị bắt.
- **Ranh giới chính xác.** Các hàng dyadic: mọi xác suất là bội số nguyên của 2^-24 và hàng có
  tổng chính xác bằng 1, nên mọi tổng một phần đều chính xác trong float, theo mọi thứ tự cộng.
  p là 0.5 hoặc 0.75, và mỗi hàng được dựng sao cho khối lượng bao gồm của một token đúng bằng
  p: ở token 0 (p bằng xác suất đầu tiên: giữ đúng 1 token), bên trong dải 4 phần tử của một
  lane, ở cuối một đoạn 1024, ở vị trí một phần ba hàng, trước một phần đuôi toàn số 0, và bốn token
  0.25 theo sau là các số 0 (các token bằng nhau nằm hai phía p: p = 0.5 giữ 2). Token có khối
  lượng bao gồm bằng p phải được giữ và token tiếp theo bị bỏ. Các hàng này được so chính xác,
  điều này bắt được cả "inclusive < p" lẫn "massBefore <= p".
- **Hàng softmax**, với p = 0.1, 0.5, 0.9, 0.95 và 0.99 (các trình giả lập: 0.5 và 0.99):
  softmax của các logits phân bố đều trong [-12, 12], tính bằng double, làm tròn về float và
  sắp xếp; một hàng nhọn (một logit lớn hơn phần còn lại 40); một hàng gần như phẳng; và một
  hàng có nửa sau chính xác bằng 0. `massBefore` phải nằm trong dải dung sai bên dưới so với
  tham chiếu, quyết định giữ phải đúng ở mọi chỗ tham chiếu cách p xa hơn dải đó, và bên trong
  dải thì quyết định nào cũng được chấp nhận.
- **Các giá trị biên của p**: 0 và 1e-7 giữ đúng token 0; 1 và 1.5 giữ mọi token, kể cả các phần
  đuôi toàn số 0 mà khối lượng đứng trước chính xác bằng 1 (các trình giả lập: 0 và 1).
- Luôn luôn: `massBefore[r][0]` chính xác là 0; `keep` chỉ chứa 0 và 1; `keep[i]` khớp với quy
  tắc áp dụng cho `massBefore[i]` mà bạn đã ghi; `kept` bằng số lượng số một của hàng. Các đầu ra
  khởi đầu là NaN, 0xAB và -1, nên bất cứ thứ gì không bao giờ được ghi sẽ lộ ra, và các phần
  tử canh gác sau cuối mỗi đầu ra phải còn nguyên.
- Mốc so sánh và phép sao chép được cung cấp cũng được kiểm thử, để một tệp cung cấp bị hỏng sẽ lộ
  ra ở đây.

## Dải dung sai, và vì sao

Tham chiếu là vòng lặp của Fig. 11.1 bằng double trên các xác suất float, chuyển thành loại
trừ: khối lượng chính xác đứng trước mỗi token, tới khoảng 1e-11 tương đối. `massBefore[i]` của
bạn phải thỏa `|got - ref| <= 1e-4 * ref`.

Cận cho thiết kế gợi ý, theo cách lab 10.3 đã suy ra cận của nó: mọi xác suất đều lớn hơn hoặc
bằng 0, nên mọi tổng một phần dọc đường đều nhỏ hơn hoặc bằng `ref`, và mỗi lần làm tròn thêm
tối đa `2^-24` của nó. Khối lượng của một token đi qua tối đa

- 3 phép cộng trong dải 4 phần tử của lane (phép đầu tiên, 0 + x, là chính xác);
- 63 trong scan SIMD group hoặc warp, và 63 trong scan các tổng warp: một vòng lặp Kogge-Stone
  chỉ cần log2 của độ rộng, nhưng Apple không ghi trong tài liệu thứ tự của
  `simd_prefix_*_sum` (C1), nên chúng tôi cho phép mọi thứ tự với tối đa 64 số hạng;
- 126 trong phần mang sang, một cho mỗi đoạn 1024 ở 128,256 token;
- 3 phép kết hợp các phần;

258 lần làm tròn, nên tối đa `(1 + 2^-24)^258 - 1`, khoảng `1.5e-5` tương đối. `1e-4` chừa ra
gấp sáu lần mức đó, và bao phủ một thiết kế có các đoạn nhỏ hơn bốn lần. Một chỉ số sai, một
token bị thiếu hoặc một scan bao gồm sẽ lệch cả một xác suất, và dải dung sai bắt được điều đó ở
mọi chỗ xác suất đó lớn hơn `1e-4` của khối lượng đứng trước nó. Các hàng dyadic hoàn toàn không
có dải dung sai.

Vì sao không so từng bit: kết quả của một scan dấu phẩy động phụ thuộc vào cây cộng của nó.
CUB, Thrust, rocPRIM và cumsum CUDA của PyTorch ghi trong tài liệu rằng kết quả có thể khác nhau
giữa các lần chạy trừ khi yêu cầu chế độ tất định (A8, B2, D4), và transformers và vLLM tính cùng
quy tắc từ một scan tăng dần so với 1 - p, nên một token nằm trong phạm vi làm tròn của p có thể
rơi vào các phía khác nhau ở các thư viện khác nhau (D4). Bên trong dải dung sai, câu trả lời nào
cũng là một bộ lấy mẫu đúng.

Mốc so sánh được cung cấp cộng cả một hàng trong một luồng, một lần làm tròn mỗi token, nên nó
được áp dung sai `1e-4 + cols * 2^-24`.

## Trước khi đo hiệu năng

Hãy ghi ra dự đoán của bạn (trang labs hỏi nó dưới id `ch11/labs/p-lab6-copy`): **với 64 hàng
128,256 token, kernel của bạn sẽ đạt bao nhiêu phần băng thông chép của bạn?** Hai điều cần cân
nhắc: phép sao chép chạy một luồng mỗi token, ở đây là 32,064 khối 256; thiết kế gợi ý chạy một khối
mỗi hàng, 64 khối, mỗi khối đi qua 126 đoạn lần lượt với các rào chắn (barrier) khối trong mỗi
đoạn. Mục 11.8 gọi một kernel bị giới hạn bởi bộ nhớ đạt hơn 80% băng thông, không có truy cập
thừa, là được tối ưu tốt (p. 273).

Benchmark đo phép sao chép, mốc so sánh một luồng mỗi hàng và kernel của bạn ở rows x cols =
1 x 128,256, 8 x 128,256, 64 x 128,256 và 256 x 32,000, trên các hàng softmax đã sắp xếp với
p = 0.9. Thời gian mỗi lần gọi là lần tốt nhất trong 5 lần chạy có đo, mỗi lần lặp lại lời gọi
cho đến khi kéo dài ít nhất 2 ms. Nó in ra số mili giây mỗi lần gọi, GB/s với 9 byte mỗi token
(đọc 4, ghi 4 + 1), phần băng thông chép bạn đạt được, và mức tăng tốc so với mốc so sánh. Mọi
kết quả được đo đều được kiểm tra trước khi báo. Các con số là phép đo trên máy của bạn, không
phải mục tiêu.

Một hình dạng 1 hoặc 8 hàng chỉ vài megabyte, và một lần chạy có đo lặp lại lời gọi trên cùng
các buffer nhiều lần. Nếu GB/s của phép sao chép ở đó cao hơn ở 64 hàng, các lần lặp đó đang được
phục vụ từ thứ gì đó nhanh hơn bộ nhớ chính, và "of copy" đang so với thứ đó. Hãy đọc tỉ lệ ở
64 x 128,256 và 256 x 32,000 cho câu hỏi về băng thông; đọc các hình dạng nhỏ để biết một khối
mỗi hàng làm được gì ở batch 1.

## Tính khả chuyển

- HIP: mặt nạ lane có kiểu và `warpSize` cho phép `kernel.cu` build qua `pmpp/vendors/hip.h`;
  với warp rộng 64, một khối 256 có 4 warp. Chưa được biên dịch hay chạy trên phần cứng AMD ở
  đây: unverified.
- Metal: `launch.mm` đọc độ rộng SIMD group từ pipeline (`threadExecutionWidth`) và từ chối 256
  luồng nếu không gồm trọn các SIMD group, hoặc có nhiều SIMD group hơn số lane của một SIMD
  group. Chế độ toán mặc định của Metal cho phép sắp xếp lại phép kết hợp (ch05-facts C7): dải
  dung sai ở trên đã cho phép mọi thứ tự bên trong một SIMD group, và các hàng dyadic chính xác
  theo mọi thứ tự.
- NaN và vô cực nằm ngoài hợp đồng: các hàng là đầu ra softmax.

## Bài này dẫn tới đâu

Đây là phần top-p của mốc dự án M5 (số đếm token của lab 9.3 là phần đầu tiên). Chương 12 nén
gọn các token được giữ thành một danh sách ngắn, chương 14 sắp xếp các logits mà lab này nhận
được ở dạng đã sắp xếp, và module X1 lấy mẫu từ nucleus: hình ảnh khúc xúc xích của mục 11.1
(p. 252), với một số ngẫu nhiên phân bố đều thay cho con dao.
